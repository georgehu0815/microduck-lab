"""Train, evaluate, render, and export macOS-native MicroDuck RLX recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from copy import deepcopy
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlx.environments.microduck_recipes import (
    RECIPES,
    SWING_REWARD_WEIGHTS,
    get_recipe,
    make_recipe_env,
    make_single_recipe_env,
    recipe_metrics,
    validate_locomotion_forward_command,
    validate_reward_weights,
    validate_stilt_options,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

RECIPE_PPO_DEFAULTS = {
    "bridge": {
        "learning_rate": 1e-5, "gamma": 0.99, "clip_coefficient": 0.1,
        "update_epochs": 2, "entropy_coefficient": 0.0,
        "max_grad_norm": 0.5, "initial_std": 0.03,
    },
    "backflip": {
        "learning_rate": 3e-6, "gamma": 0.99, "clip_coefficient": 0.2,
        "update_epochs": 2, "entropy_coefficient": 0.0,
        "max_grad_norm": 0.5, "initial_std": 0.03,
    },
    "dance": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_coefficient": 0.2,
        "update_epochs": 5,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 0.5,
        "initial_std": math.exp(-0.5),
    },
    "swing": {
        "learning_rate": 1e-4,
        "gamma": 0.995,
        "clip_coefficient": 0.1,
        "update_epochs": 3,
        "entropy_coefficient": 0.002,
        "max_grad_norm": 1.0,
        "initial_std": 0.1,
    },
    "running": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_coefficient": 0.2,
        "update_epochs": 5,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 0.5,
        "initial_std": math.exp(-0.5),
    },
    "stilts": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "clip_coefficient": 0.2,
        "update_epochs": 5,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 0.5,
        "initial_std": math.exp(-0.5),
    },
}


@dataclass(frozen=True)
class ArtifactPaths:
    directory: Path
    checkpoint: Path
    metadata: Path
    onnx: Path


def artifact_paths(
    recipe: str,
    *,
    output_dir: Path | None = None,
    checkpoint: Path | None = None,
    onnx_output: Path | None = None,
) -> ArtifactPaths:
    get_recipe(recipe)
    directory = (
        output_dir.expanduser()
        if output_dir is not None
        else (
            checkpoint.expanduser().parent
            if checkpoint is not None
            else REPOSITORY_ROOT / "runs" / recipe
        )
    )
    checkpoint_path = (
        checkpoint.expanduser()
        if checkpoint is not None
        else directory / f"{recipe}.safetensors"
    )
    onnx_path = (
        onnx_output.expanduser()
        if onnx_output is not None
        else directory / f"{recipe}.onnx"
    )
    return ArtifactPaths(
        directory=directory,
        checkpoint=checkpoint_path,
        metadata=checkpoint_path.with_suffix(checkpoint_path.suffix + ".json"),
        onnx=onnx_path,
    )


def _weight_overrides(value: str) -> dict[str, float]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--weight-overrides must be valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("--weight-overrides must be a JSON object")
    result: dict[str, float] = {}
    for key, weight in decoded.items():
        if (
            not isinstance(key, str)
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
        ):
            raise ValueError("reward weights must map string names to numbers")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"reward weight {key!r} must be finite and non-negative")
        result[key] = float(weight)
    return result


def _recipe_weight_overrides(args: argparse.Namespace) -> dict[str, float]:
    return validate_reward_weights(
        args.recipe,
        _weight_overrides(args.weight_overrides),
    )


def _environment(args: argparse.Namespace, *, normalize: bool):
    gamma = getattr(args, "gamma", RECIPE_PPO_DEFAULTS[args.recipe]["gamma"])
    return make_recipe_env(
        args.recipe,
        backflip_mode="landing" if getattr(args, "command", None) == "train" else "showcase",
        num_envs=args.num_envs,
        backend=args.backend,
        seed=args.seed,
        actuator=args.actuator,
        domain_rand=args.domain_rand,
        obs_noise=args.obs_noise,
        action_delay=args.action_delay,
        random_yaw=args.random_yaw if args.recipe != "swing" else False,
        max_episode_s=args.max_episode_s,
        weight_overrides=_recipe_weight_overrides(args),
        dance_clip=getattr(args, "dance_clip", None),
        dance_pose_sigma=getattr(args, "dance_pose_sigma", None),
        locomotion_forward_command=args.locomotion_forward_command,
        stilt_height_cm=args.stilt_height_cm,
        stilt_blend=args.stilt_blend,
        stilt_mass_kg=args.stilt_mass_kg,
        swing_initial_angle_deg=args.swing_initial_angle_deg,
        swing_initial_rate_rad_s=args.swing_initial_rate_rad_s,
        swing_planar_actions=args.swing_planar_actions,
        normalize_observations=normalize,
        normalize_rewards=normalize and getattr(args, "normalize_rewards", args.recipe == "swing"),
        gamma=gamma,
        bridge_curriculum=getattr(args, "bridge_curriculum", False),
    )


def _training_progress_callback(
    total_timesteps: int,
    num_envs: int,
    rollout_steps: int,
    telemetry_path: Path | None = None,
) -> Callable[[dict[str, Any], int], None]:
    import numpy as np

    completed_episodes = 0
    rollout_size = num_envs * rollout_steps

    def report(info: dict[str, Any], step: int) -> None:
        nonlocal completed_episodes
        current_step = min(total_timesteps, step + num_envs)
        returns: list[float] = []
        if "episode" in info and "_episode" in info:
            mask = np.asarray(info["_episode"], dtype=bool)
            if mask.any():
                returns = np.asarray(info["episode"]["r"])[mask].astype(float).tolist()
                completed_episodes += len(returns)
        rollout_complete = current_step == total_timesteps or (
            rollout_size > 0 and current_step % rollout_size == 0
        )
        if not returns and not rollout_complete:
            return
        if telemetry_path is not None and returns:
            with telemetry_path.open("a") as stream:
                stream.write(json.dumps({
                    "phase": "episodes", "env_steps": current_step,
                    "returns": returns,
                    "lengths": np.asarray(info["episode"]["l"])[mask].astype(int).tolist(),
                    "mean_raw_return": float(np.mean(returns)),
                }, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "event": "training_episode",
                    "steps": current_step,
                    "total": total_timesteps,
                    "mean_reward": float(np.mean(returns)) if returns else None,
                    "episodes": completed_episodes,
                }
            ),
            flush=True,
        )

    return report


def _training_rollout_observer(
    total_timesteps: int,
    algorithm: Any,
    telemetry_path: Path | None = None,
    checkpoint_callback: Callable[[int], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    import numpy as np

    def report(event: dict[str, Any]) -> None:
        measured = {**event, "env_steps": int(algorithm.step)}
        if event.get("phase") == "collection":
            measured["mean_reward"] = float(
                np.asarray(algorithm.buffer.rewards, dtype=np.float64).mean()
            )
        if telemetry_path is not None:
            with telemetry_path.open("a") as stream:
                stream.write(json.dumps(measured, allow_nan=False) + "\n")
        if event.get("phase") == "update":
            if telemetry_path is not None:
                print(json.dumps({"event - ppo_microduck_studio.py:244": "ppo_update", **measured}), flush=True)
            if checkpoint_callback is not None:
                checkpoint_callback(int(algorithm.step))
        if event.get("phase") != "collection":
            return
        rewards = np.asarray(algorithm.buffer.rewards, dtype=np.float64)
        print(
            json.dumps(
                {
                    "event": "training_progress",
                    "source": "rollout",
                    "steps": min(total_timesteps, int(algorithm.step)),
                    "total": total_timesteps,
                    "mean_reward": float(rewards.mean()),
                }
            ),
            flush=True,
        )

    return report


def _recipe_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if args.recipe == "backflip":
        from rlx.environments.backflip import protocol_metadata
        options = protocol_metadata()
    if args.recipe == "dance" and getattr(args, "dance_clip", None) is not None:
        clip = args.dance_clip.expanduser().resolve(strict=True)
        options = {"dance_clip": str(clip), "dance_clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}
    if args.recipe == "dance" and getattr(args, "dance_pose_sigma", None) is not None:
        options["dance_pose_sigma"] = args.dance_pose_sigma
    if args.recipe == "swing":
        options = {
            "swing_initial_angle_deg": args.swing_initial_angle_deg,
            "swing_initial_rate_rad_s": args.swing_initial_rate_rad_s,
            "swing_planar_actions": args.swing_planar_actions,
        }
    if args.recipe == "stilts":
        height, blend, mass = validate_stilt_options(
            args.stilt_height_cm,
            args.stilt_blend,
            args.stilt_mass_kg,
        )
        options = {
            "stilt_height_cm": height,
            "stilt_blend": blend,
            "stilt_mass_kg": mass,
        }
    if (
        args.recipe in {"running", "stilts"}
        and args.locomotion_forward_command is not None
    ):
        options["locomotion_forward_command"] = args.locomotion_forward_command
    return options


def _initialization_provenance(source: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    source = source.expanduser().resolve(strict=True)
    hashes = _source_hashes(source, "checkpoint")
    return {
        "kind": "checkpoint",
        "source_checkpoint": str(source),
        "source_sha256": hashes[str(source)],
        "source_sidecar_sha256": hashes[str(source.with_suffix(source.suffix + ".json"))],
        "loaded_metadata": deepcopy(metadata),
    }


def _metadata(
    args: argparse.Namespace, steps: int, *, initialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = {
        "recipe": args.recipe,
        "steps": steps,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_episode_s": args.max_episode_s,
        "backend": args.backend,
        "actuator": args.actuator,
        "reward_weights": _recipe_weight_overrides(args),
        "freeze_observation_normalization": getattr(args, "freeze_observation_normalization", False),
        "freeze_reward_normalization": getattr(args, "freeze_reward_normalization", False),
        "recipe_options": _recipe_options(args),
        "bridge_curriculum": getattr(args, "bridge_curriculum", False),
        "randomization": {
            "domain_rand": args.domain_rand,
            "obs_noise": args.obs_noise,
            "action_delay": args.action_delay,
            "random_yaw": args.random_yaw if args.recipe != "swing" else False,
        },
        "ppo": {
            "total_timesteps": args.total_timesteps,
            "num_steps": args.num_steps,
            "num_minibatches": args.num_minibatches,
            "update_epochs": args.update_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "normalize_advantages": args.normalize_advantages,
            "clip_coefficient": args.clip_coefficient,
            "clip_value_loss": args.clip_value_loss,
            "entropy_coefficient": args.entropy_coefficient,
            "value_coefficient": args.value_coefficient,
            "max_grad_norm": args.max_grad_norm,
            "learning_rate": args.learning_rate,
            "initial_std": getattr(args, "initial_std", RECIPE_PPO_DEFAULTS[args.recipe]["initial_std"]),
            "normalize_rewards": getattr(args, "normalize_rewards", args.recipe == "swing"),
        },
    }
    if initialization is not None:
        metadata["initialization"] = deepcopy(initialization)
        if "teacher_assisted" in initialization["loaded_metadata"]:
            metadata["teacher_assisted"] = initialization["loaded_metadata"]["teacher_assisted"]
    return metadata


def train(args: argparse.Namespace) -> dict[str, object]:
    freeze_observations = getattr(args, "freeze_observation_normalization", False)
    if freeze_observations and args.init_from is None and not (args.recipe == "backflip" and args.total_timesteps > 4):
        raise ValueError("--freeze-observation-normalization requires --init-from")
    paths = artifact_paths(
        args.recipe,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        onnx_output=args.onnx_output,
    )
    paths.directory.mkdir(parents=True, exist_ok=True)
    if args.recipe == "bridge" and args.init_from is None and args.total_timesteps > 4:
        from rlx.models.bridge_bootstrap import initialize_bridge
        args.init_from = initialize_bridge(paths.directory / "initialization/bridge.safetensors",
                                           seed=args.seed, initial_std=args.initial_std)
        freeze_observations = True
        args.freeze_observation_normalization = True
    if args.recipe == "backflip" and args.init_from is None and args.total_timesteps > 4:
        from rlx.models.backflip_bootstrap import initialize_landing
        args.init_from = initialize_landing(paths.directory / "initialization/backflip.safetensors",
                                            seed=args.seed, initial_std=args.initial_std, gamma=args.gamma,
                                            episode_seconds=args.max_episode_s or 12,
                                            normalize_rewards=args.normalize_rewards,
                                            weight_overrides=_recipe_weight_overrides(args))
        freeze_observations = True
        args.freeze_observation_normalization = True
    env = _environment(args, normalize=True)
    try:
        env.freeze_observation_normalization = freeze_observations
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np

        from rlx.algorithms.ppo import PPO, PPOConfig
        from rlx.buffers.rollout_buffer import RolloutBuffer
        from rlx.models.microduck import (
            create_actor_critic,
            load_checkpoint,
            save_checkpoint,
        )

        np.random.seed(args.seed)
        mx.random.seed(args.seed)
        config = PPOConfig(
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=args.normalize_advantages,
            clip_coefficient=args.clip_coefficient,
            clip_value_loss=args.clip_value_loss,
            entropy_coefficient=args.entropy_coefficient,
            value_coefficient=args.value_coefficient,
            max_grad_norm=args.max_grad_norm,
        )
        resumed_steps = 0
        initialization = None
        if args.init_from is not None:
            loaded = load_checkpoint(args.init_from.expanduser())
            initialization = _initialization_provenance(args.init_from, loaded["metadata"])
            network = loaded["model"]
            env.observation_rms.mean = loaded["mean"].astype(np.float64)
            env.observation_rms.var = loaded["variance"].astype(np.float64)
            env.observation_rms.count = loaded["count"]
            if freeze_observations:
                env.epsilon = loaded["epsilon"]
                env.clip = loaded["clip"]
            if loaded["return_mean"] is not None:
                env.return_rms.mean = np.asarray(
                    loaded["return_mean"], dtype=np.float64
                )
                env.return_rms.var = np.asarray(
                    loaded["return_variance"], dtype=np.float64
                )
                env.return_rms.count = loaded["return_count"]
            if args.recipe == "backflip" and (loaded["metadata"].get("reward_calibration") or loaded["metadata"].get("freeze_reward_normalization")):
                env.freeze_reward_normalization = True
                args.freeze_reward_normalization = True
            resumed_steps = int(loaded["metadata"].get("steps", 0))
        else:
            network = create_actor_critic(
                initial_std=args.initial_std
            )
        mx.eval(network.parameters())
        algorithm = PPO(
            config=config,
            env=env,
            network=network,
            optimizer=optim.Adam(learning_rate=args.learning_rate),
            buffer=RolloutBuffer(
                config.num_steps,
                env.observation_space,
                env.action_space,
                gamma=config.gamma,
                num_envs=config.num_envs,
            ),
            key=mx.random.key(args.seed),
        )
        telemetry_path = paths.directory / "training-metrics.jsonl"
        telemetry_path.write_text("")
        checkpoint_interval = getattr(args, "checkpoint_interval", 0)
        next_checkpoint = checkpoint_interval

        def snapshot_training(step: int, *, initial: bool = False) -> None:
            nonlocal next_checkpoint
            if not initial and (not checkpoint_interval or step < next_checkpoint):
                return
            destination = paths.directory / "checkpoints" / f"step-{step:09d}.safetensors"
            save_checkpoint(
                destination, network, env.observation_rms.mean,
                env.observation_rms.var, env.observation_rms.count,
                return_mean=env.return_rms.mean, return_variance=env.return_rms.var,
                return_count=env.return_rms.count, epsilon=env.epsilon, clip=env.clip,
                metadata=_metadata(args, resumed_steps + step, initialization=initialization),
            )
            if not initial:
                next_checkpoint = (step // checkpoint_interval + 1) * checkpoint_interval

        snapshot_training(0, initial=True)
        algorithm.train(
            args.total_timesteps,
            callback=_training_progress_callback(
                args.total_timesteps,
                args.num_envs,
                args.num_steps,
                telemetry_path,
            ),
            observer=_training_rollout_observer(
                args.total_timesteps, algorithm, telemetry_path, snapshot_training,
            ),
        )
        save_checkpoint(
            paths.checkpoint,
            network,
            env.observation_rms.mean,
            env.observation_rms.var,
            env.observation_rms.count,
            return_mean=env.return_rms.mean,
            return_variance=env.return_rms.var,
            return_count=env.return_rms.count,
            epsilon=env.epsilon,
            clip=env.clip,
            metadata=_metadata(args, resumed_steps + algorithm.step, initialization=initialization),
        )
    finally:
        env.close()

    result: dict[str, object] = {
        "command": "train",
        "recipe": args.recipe,
        "checkpoint": str(paths.checkpoint),
        "metadata": str(paths.metadata),
        "steps": resumed_steps + algorithm.step,
        "trained_steps": algorithm.step,
        "resumed_from": str(args.init_from.expanduser()) if args.init_from else None,
        "training_metrics": str(telemetry_path),
    }
    if args.export_onnx:
        from rlx.export.microduck_onnx import export_deterministic_actor

        result["onnx"] = str(export_deterministic_actor(paths.checkpoint, paths.onnx))
    return result


def export(args: argparse.Namespace) -> dict[str, object]:
    from rlx.export.microduck_onnx import export_deterministic_actor

    paths = artifact_paths(
        args.recipe,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        onnx_output=args.output,
    )
    output = export_deterministic_actor(paths.checkpoint, paths.onnx)
    return {
        "command": "export",
        "recipe": args.recipe,
        "checkpoint": str(paths.checkpoint),
        "output": str(output),
    }


def _policy_from_checkpoint(checkpoint: Path) -> Callable[[Any], Any]:
    import mlx.core as mx
    from rlx.models.microduck import load_checkpoint, normalize_observations

    loaded = load_checkpoint(checkpoint)

    def infer(observation: Any) -> Any:
        normalized = normalize_observations(
            mx.array(observation, dtype=mx.float32),
            loaded["mean"],
            loaded["variance"],
            epsilon=loaded["epsilon"],
            clip=loaded["clip"],
        )
        return loaded["model"].deterministic(normalized)

    return infer


def _policy_from_onnx(policy: Path) -> Callable[[Any], Any]:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def infer(observation: Any) -> Any:
        return session.run(
            [output_name],
            {input_name: np.asarray(observation, dtype=np.float32)},
        )[0]

    return infer


def _source_paths(args: argparse.Namespace) -> tuple[Path, str]:
    if args.policy is not None:
        return args.policy.expanduser(), "policy"
    paths = artifact_paths(
        args.recipe,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
    )
    if args.checkpoint is not None:
        return paths.checkpoint, "checkpoint"
    if paths.onnx.exists():
        return paths.onnx, "policy"
    return paths.checkpoint, "checkpoint"


def _source_hashes(source: Path, source_type: str) -> dict[str, str]:
    paths = [source]
    if source_type == "checkpoint":
        paths.append(source.with_suffix(source.suffix + ".json"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _validate_backflip_source(source: Path, source_type: str) -> None:
    from rlx.environments.backflip import protocol_metadata

    if source_type == "checkpoint":
        metadata = json.loads(source.with_suffix(source.suffix + ".json").read_text())["metadata"]
    else:
        import onnx
        properties = {item.key: item.value for item in onnx.load(source).metadata_props}
        metadata = json.loads(properties.get("rlx_metadata", "{}"))
    if metadata.get("recipe") != "backflip":
        raise ValueError("Backflip source must identify its landing-policy recipe")
    saved = metadata.get("recipe_options", metadata)
    current = protocol_metadata()
    for key in ("backflip_protocol_version", "stand_policy_sha256"):
        if saved.get(key) != current[key]:
            raise ValueError(f"Backflip source {key} does not match the current composite controller")


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np
    from rlx.environments.dance_evaluation import DanceEvaluation
    from rlx.environments.locomotion_evaluation import LocomotionEvaluation
    from rlx.environments.swing_evaluation import SwingEvaluation, SwingEvaluationPlan
    from rlx.environments.backflip_evaluation import BackflipEvaluation
    from rlx.environments.bridge_evaluation import BridgeEvaluation

    source, source_type = _source_paths(args)
    if not source.is_file():
        raise FileNotFoundError(f"{source_type} does not exist: {source}")
    if args.recipe == "backflip":
        _validate_backflip_source(source, source_type)
    source_hashes = _source_hashes(source, source_type)
    infer = (
        _policy_from_onnx(source)
        if source_type == "policy"
        else _policy_from_checkpoint(source)
    )
    plan = SwingEvaluationPlan(min_bidirectional_span_deg=args.swing_min_span_deg) if args.recipe == "swing" else None
    swing = SwingEvaluation(args.num_envs, plan) if plan is not None else None
    bridge = BridgeEvaluation(args.num_envs, 1000) if args.recipe == "bridge" else None
    dance_steps = round((args.max_episode_s or get_recipe(args.recipe).default_episode_s) * 50)
    if args.recipe == "dance":
        from microduck_local.motion import load_clip
        from rlx.environments.microduck_recipes import _resolve_dance_clip
        clip_name, clip_directory = _resolve_dance_clip(args.dance_clip)
        dance_steps = max(dance_steps, load_clip(clip_name, clip_directory).steps)
    dance = DanceEvaluation(
        args.num_envs,
        dance_steps,
    ) if args.recipe == "dance" else None
    locomotion_steps = {"running": 600, "stilts": 500}.get(args.recipe)
    locomotion = (
        LocomotionEvaluation(args.recipe, args.num_envs, locomotion_steps)
        if locomotion_steps is not None
        else None
    )
    backflip = BackflipEvaluation(args.num_envs, max(300, round((args.max_episode_s or 12) * 50))) if args.recipe == "backflip" else None
    evaluation = {
        "mode": args.evaluation_mode, "seed": args.seed, "num_envs": args.num_envs,
        "steps_per_env": args.eval_steps,
        "environment": {
            "recipe": args.recipe, "backend": args.backend, "actuator": args.actuator,
            "max_episode_s": args.max_episode_s or get_recipe(args.recipe).default_episode_s,
            "domain_rand": args.domain_rand, "obs_noise": args.obs_noise,
            "action_delay": args.action_delay, "random_yaw": args.random_yaw if args.recipe != "swing" else False,
            "recipe_options": _recipe_options(args),
            "reward_weights": {**(SWING_REWARD_WEIGHTS if args.recipe == "swing" else {}), **_recipe_weight_overrides(args)},
        },
        "swing_criteria": plan.to_dict() if plan is not None else None,
        "backflip_criteria": backflip.criteria.to_dict() if backflip is not None else None,
        "dance_criteria": {**dance.criteria.to_dict(), "required_steps": dance_steps} if dance is not None else None,
        "locomotion_criteria": (
            locomotion.criteria.to_dict() if locomotion is not None else None
        ),
    }
    env = _environment(args, normalize=False)
    metrics: dict[str, list[float]] = defaultdict(list)
    try:
        observation, state, _ = env.reset(None)
        rollout_returns = np.zeros(args.num_envs, dtype=np.float64)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        termination_count = 0
        truncation_count = 0
        action_abs_sum = 0.0
        action_abs_max = 0.0
        finite = bool(np.isfinite(np.asarray(observation)).all())
        for _ in range(args.eval_steps):
            action = infer(observation)
            action_values = np.asarray(action, dtype=np.float32)
            observation, state, reward, terminated, truncated, info = env.step(
                None, state, action
            )
            reward_values = np.asarray(reward, dtype=np.float64)
            terminated_values = np.asarray(terminated, dtype=bool)
            truncated_values = np.asarray(truncated, dtype=bool)
            observation_values = np.asarray(observation)
            rollout_returns += reward_values
            termination_count += int(terminated_values.sum())
            truncation_count += int(truncated_values.sum())
            action_abs_sum += float(np.abs(action_values).sum())
            action_abs_max = max(action_abs_max, float(np.abs(action_values).max()))
            finite = finite and bool(
                np.isfinite(observation_values).all()
                and np.isfinite(reward_values).all()
                and np.isfinite(action_values).all()
            )
            for env_index, item in enumerate(info["infos"]):
                measured = item.get("recipe_metrics", {})
                if backflip is not None:
                    backflip.observe(env_index, measured, terminated=bool(terminated_values[env_index]), truncated=bool(truncated_values[env_index]))
                if swing is not None:
                    swing.observe(env_index, measured, terminated=bool(terminated_values[env_index]),
                                  truncated=bool(truncated_values[env_index]))
                if dance is not None:
                    dance.observe(env_index, item.get("dance_state", {}),
                                  terminated=bool(terminated_values[env_index]),
                                  truncated=bool(truncated_values[env_index]))
                if locomotion is not None:
                    locomotion.observe(
                        env_index,
                        measured,
                        bool(terminated_values[env_index]),
                        bool(truncated_values[env_index]),
                    )
                if bridge is not None:
                    bridge.observe(env_index, measured,
                                   bool(terminated_values[env_index]),
                                   bool(truncated_values[env_index]))
                for name, value in measured.items():
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        finite = False
                        continue
                    if math.isfinite(numeric_value):
                        metrics[name].append(numeric_value)
                    else:
                        finite = False
            episode_mask = np.asarray(info["_episode"], dtype=bool)
            if episode_mask.any():
                completed_returns.extend(
                    np.asarray(info["episode"]["r"])[episode_mask]
                    .astype(float)
                    .tolist()
                )
                completed_lengths.extend(
                    np.asarray(info["episode"]["l"])[episode_mask].astype(int).tolist()
                )

        episode_array = np.asarray(completed_returns, dtype=np.float64)
        length_array = np.asarray(completed_lengths, dtype=np.float64)
        score = (
            float(episode_array.mean())
            if episode_array.size
            else float(rollout_returns.mean())
        )
        transitions = args.eval_steps * args.num_envs
        failures: list[str] = []
        if not finite:
            failures.append("non-finite rollout values")
        if args.min_mean_return is not None and score < args.min_mean_return:
            failures.append(
                f"mean return {score:.6g} is below {args.min_mean_return:.6g}"
            )
        if len(completed_returns) < args.min_episodes:
            failures.append(
                f"completed episodes {len(completed_returns)} is below "
                f"{args.min_episodes}"
            )
        metric_summary = {
            name: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in sorted(metrics.items())
            if values
        }
        if _source_hashes(source, source_type) != source_hashes:
            failures.append("policy source changed during evaluation")
        if args.recipe == "backflip" and _recipe_options(args) != evaluation["environment"]["recipe_options"]:
            failures.append("backflip protocol or stand dependency changed during evaluation")
        pipeline_passed = finite and not failures
        swing_assessment = swing.report() if swing is not None else None
        dance_assessment = dance.report() if dance is not None else None
        locomotion_assessment = (
            locomotion.report() if locomotion is not None else None
        )
        backflip_assessment = backflip.report() if backflip is not None else None
        bridge_assessment = bridge.report() if bridge is not None else None
        skill_status = "not_assessed"
        if bridge_assessment is not None and args.evaluation_mode == "skill":
            skill_status = "passed" if pipeline_passed and bridge_assessment["passed"] else "failed"
            failures.extend(bridge_assessment["failures"])
        if backflip_assessment is not None and args.evaluation_mode == "skill":
            skill_status = "passed" if pipeline_passed and backflip_assessment["passed"] else "failed"
            failures.extend(backflip_assessment["failures"])
        if dance_assessment is not None and args.evaluation_mode == "skill":
            skill_status = "passed" if pipeline_passed and dance_assessment["passed"] else "failed"
            failures.extend(dance_assessment["failures"])
        if swing_assessment is not None:
            for key, episode_key in (("swing_span_deg", "peak_to_peak_span_deg"),
                                     ("swing_bidirectional_span_deg", "bidirectional_span_deg")):
                spans = [episode[episode_key] for episode in swing_assessment["episodes"]]
                if spans:
                    metric_summary[key] = {
                        "mean": float(np.mean(spans)), "median": float(np.median(spans)),
                        "min": float(np.min(spans)), "max": float(np.max(spans)),
                    }
            if args.evaluation_mode == "skill":
                skill_status = "passed" if pipeline_passed and swing_assessment["passed"] else "failed"
                failures.extend(swing_assessment["failures"])
        if locomotion_assessment is not None and args.evaluation_mode == "skill":
            skill_status = (
                "passed"
                if pipeline_passed and locomotion_assessment["passed"]
                else "failed"
            )
            failures.extend(locomotion_assessment["failures"])
        return _finite_json({
            "command": "eval",
            "evaluation_mode": args.evaluation_mode,
            "evaluation": evaluation,
            "source_sha256": source_hashes[str(source)],
            "source_files_sha256": source_hashes,
            "pipeline_passed": pipeline_passed,
            "skill_status": skill_status,
            "swing_assessment": swing_assessment,
            "dance_assessment": dance_assessment,
            "locomotion_assessment": locomotion_assessment,
            "backflip_assessment": backflip_assessment,
            "bridge_assessment": bridge_assessment,
            "recipe": args.recipe,
            "source": str(source),
            "source_type": source_type,
            "steps": args.eval_steps,
            "num_envs": args.num_envs,
            "transitions": transitions,
            "episodes": len(completed_returns),
            "mean_return": score,
            "episode_return": {
                "mean": float(episode_array.mean()) if episode_array.size else None,
                "std": float(episode_array.std()) if episode_array.size else None,
                "min": float(episode_array.min()) if episode_array.size else None,
                "max": float(episode_array.max()) if episode_array.size else None,
            },
            "episode_length": {
                "mean": float(length_array.mean()) if length_array.size else None,
                "min": int(length_array.min()) if length_array.size else None,
                "max": int(length_array.max()) if length_array.size else None,
            },
            "rollout_return": {
                "mean": float(rollout_returns.mean()),
                "std": float(rollout_returns.std()),
                "min": float(rollout_returns.min()),
                "max": float(rollout_returns.max()),
            },
            "recipe_metrics": metric_summary,
            "terminated": termination_count,
            "truncated": truncation_count,
            "termination_rate": termination_count / transitions,
            "reward_per_transition": float(rollout_returns.sum() / transitions),
            "action_abs_mean": action_abs_sum / (transitions * 14),
            "action_abs_max": action_abs_max,
            "finite": finite,
            "passed": finite and not failures,
            "failures": failures,
        })
    finally:
        env.close()


def _render_policy_source(source: Path, source_type: str):
    if source_type == "policy":
        infer_batch = _policy_from_onnx(source)
    else:
        infer_batch = _policy_from_checkpoint(source)

    def infer(observation):
        import numpy as np

        return np.asarray(infer_batch(observation[None, :]), dtype=np.float32)[0]

    return infer


def _camera(recipe: str, name: str, distance: float):
    import mujoco

    angles = {
        "side": (90.0, -8.0),
        "front": (180.0, -8.0),
        "three-quarter": (125.0, -14.0),
    }
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = angles[name]
    camera.distance = distance or get_recipe(recipe).default_render_distance
    return camera


def render(args: argparse.Namespace) -> dict[str, object]:
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import numpy as np
    from microduck_local.render_rollout import build_sheet, sheet_indices

    recipe_options = _recipe_options(args)
    source, source_type = _source_paths(args)
    if args.recipe == "backflip":
        _validate_backflip_source(source, source_type)
    if not source.is_file():
        raise FileNotFoundError(f"{source_type} does not exist: {source}")
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_hashes(source, source_type)
    infer = _render_policy_source(source, source_type)
    render_seconds = (
        args.render_seconds
        if args.render_seconds is not None
        else (
            args.max_episode_s
            if args.max_episode_s is not None
            else get_recipe(args.recipe).default_episode_s
        )
    )
    env = make_single_recipe_env(
        args.recipe,
        seed=args.seed,
        actuator=args.actuator,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=render_seconds,
        weight_overrides=_recipe_weight_overrides(args),
        dance_clip=getattr(args, "dance_clip", None),
        dance_pose_sigma=getattr(args, "dance_pose_sigma", None),
        locomotion_forward_command=args.locomotion_forward_command,
        stilt_height_cm=args.stilt_height_cm,
        stilt_blend=args.stilt_blend,
        stilt_mass_kg=args.stilt_mass_kg,
        swing_initial_angle_deg=args.swing_initial_angle_deg,
        swing_initial_rate_rad_s=args.swing_initial_rate_rad_s,
        swing_planar_actions=args.swing_planar_actions,
    )
    if args.recipe == "dance":
        env.unwrapped.terminate_on_fall = False
    width = args.width - args.width % 2
    height = args.height - args.height % 2
    renderer = mujoco.Renderer(env.unwrapped.model, height=height, width=width)
    camera = _camera(args.recipe, args.camera, args.distance)
    stride = max(1, round(50 / args.fps))
    real_fps = 50 / stride
    render_steps = max(1, round(render_seconds * 50))
    outputs: list[dict[str, object]] = []
    try:
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            frames: list[np.ndarray] = []
            captions: list[list[str]] = []
            step = 0
            reset_count = 0
            while step < render_steps:
                if step % stride == 0:
                    trunk = env.unwrapped.data.xpos[env.unwrapped.trunk_body_id]
                    camera.lookat[:] = (
                        float(trunk[0]),
                        float(trunk[1]),
                        float(trunk[2]),
                    )
                    if args.recipe == "bridge":
                        camera.lookat[:] = (0.0, 0.0, 0.35)
                    renderer.update_scene(env.unwrapped.data, camera=camera)
                    frame = renderer.render().copy()
                    if args.recipe == "bridge":
                        from PIL import Image, ImageDraw
                        image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image)
                        measured = recipe_metrics(env.unwrapped, args.recipe)
                        draw.rectangle((0, 0, width, 42), fill=(12, 18, 24))
                        draw.text((10, 7), f"UNASSISTED BRIDGE | attempt {reset_count + 1} | diagnostic rollout", fill=(140, 245, 193))
                        draw.text((10, 24), f"progress {measured['bridge_progress_m']:.2f} m | crossing {bool(measured['bridge_crossed'])} | resets {reset_count}", fill="white")
                        frame = np.asarray(image)
                    if args.recipe == "backflip":
                        from PIL import Image, ImageDraw
                        image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, width, 42), fill=(12, 18, 24))
                        draw.text((10, 7), f"ASSISTED SHOWCASE | {env.unwrapped.stage.upper()}", fill=(140, 245, 193))
                        draw.text((10, 24), "Spotter launch > learned landing > pretrained stand handoff", fill="white")
                        frame = np.asarray(image)
                    frames.append(frame)
                    metric_values = recipe_metrics(env.unwrapped, args.recipe)
                    compact = ", ".join(
                        f"{key}={float(value):.3g}"
                        for key, value in list(metric_values.items())[:3]
                    )
                    if args.recipe == "bridge":
                        compact = (f"crossed={int(metric_values['bridge_crossed'])} "
                                   f"progress={metric_values['bridge_progress_m']:.2f}m "
                                   f"hold={metric_values['bridge_assistance']:.0f}")
                    captions.append(
                        [
                            f"frame {len(frames) - 1}  t={step / 50:.2f}s",
                            f"recipe={args.recipe}",
                            compact or "metrics unavailable",
                            *([f"ASSISTED SHOWCASE: {env.unwrapped.stage}", "spotter launch / PPO landing / alpha_stand handoff"] if args.recipe == "backflip" else []),
                        ]
                    )
                action = infer(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                step += 1
                if (terminated or truncated) and step < render_steps:
                    reset_count += 1
                    observation, _ = env.reset(seed=args.seed + episode + reset_count)
            if not frames:
                raise RuntimeError("render produced no frames")
            video = output / f"ep{episode}.mp4"
            with imageio.get_writer(
                video,
                fps=real_fps,
                macro_block_size=None,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            indices = sheet_indices(len(frames), args.sheet_frames)
            if args.recipe == "backflip" and render_seconds >= 6 and len(indices) >= 4:
                moments = [*np.linspace(0., 3., len(indices) - 2), render_seconds / 2, (len(frames) - 1) / real_fps]
                indices = [min(len(frames) - 1, round(moment * real_fps)) for moment in moments]
            sheet = output / f"ep{episode}_sheet.png"
            build_sheet(
                [frames[index] for index in indices],
                [captions[index] for index in indices],
                [
                    f"{source.name}  recipe={args.recipe}  episode={episode}",
                    f"{step} control steps ({step / 50:.2f}s), resets={reset_count}",
                ],
                [
                    "Deterministic policy rollout. Inspect posture, contacts, "
                    "and motion before treating reward as evidence."
                ],
                [False] * len(indices),
                sheet,
            )
            outputs.append({"mp4": str(video), "contact_sheet": str(sheet),
                            "seed": args.seed + episode, "control_steps": step,
                            "reset_count": reset_count, "frames": len(frames), "fps": real_fps})
    finally:
        renderer.close()
        env.close()
    if _source_hashes(source, source_type) != source_hashes:
        raise RuntimeError("policy source changed during rendering")
    if args.recipe == "backflip" and _recipe_options(args) != recipe_options:
        raise RuntimeError("backflip protocol or stand dependency changed during rendering")
    return {
        "command": "render",
        "recipe": args.recipe,
        "source": str(source),
        "source_type": source_type,
        "output": str(output),
        "render_seconds": render_seconds,
        "source_sha256": source_hashes[str(source)],
        "source_files_sha256": source_hashes,
        "environment": {"actuator": args.actuator, "seed": args.seed,
                        "domain_rand": False, "obs_noise": False, "action_delay": False,
                        "random_yaw": False, "recipe_options": recipe_options},
        "episodes": outputs,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _recipe_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", choices=sorted(RECIPES), required=True)


def _common(parser: argparse.ArgumentParser) -> None:
    _recipe_argument(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=_positive_int, default=16)
    parser.add_argument("--backend", default="fork")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actuator", choices=("xml", "bam"), default="xml")
    parser.add_argument("--max-episode-s", type=_positive_float)
    parser.add_argument("--weight-overrides", default="{}")
    parser.add_argument("--dance-clip", type=Path)
    parser.add_argument("--dance-pose-sigma", type=_positive_float)
    parser.add_argument("--bridge-curriculum", action="store_true",
                        help="training only: progressively release bridge physics assistance")
    parser.add_argument(
        "--locomotion-forward-command",
        metavar="M_S",
        type=_positive_float,
        help="pin running/stilts to a forward vx command in metres per second",
    )
    parser.add_argument(
        "--domain-rand", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--obs-noise", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--action-delay", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--random-yaw", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--stilt-height-cm", type=float, default=2.0)
    parser.add_argument("--stilt-blend", type=float, default=0.0)
    parser.add_argument("--stilt-mass-kg", type=float)
    parser.add_argument(
        "--swing-initial-angle-deg",
        type=_non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--swing-initial-rate-rad-s",
        type=_non_negative_float,
        default=0.0,
    )
    parser.add_argument(
        "--swing-planar-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )


def _policy_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--policy", type=Path)


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.recipe == "basketball":
        parser.error("basketball requires the recurrent ppo_microduck_balance.py adapter")
    if getattr(args, "bridge_curriculum", False) and (args.recipe != "bridge" or args.command != "train"):
        parser.error("--bridge-curriculum is only valid for bridge training; evaluation is unassisted")
    if args.command == "export":
        return
    if args.dance_clip is not None:
        if args.recipe != "dance":
            parser.error("--dance-clip requires --recipe dance")
        if not args.dance_clip.expanduser().is_file():
            parser.error(f"--dance-clip does not exist: {args.dance_clip}")
    if args.dance_pose_sigma is not None and args.recipe != "dance":
        parser.error("--dance-pose-sigma requires --recipe dance")
    if (
        args.locomotion_forward_command is not None
        and args.recipe not in {"running", "stilts"}
    ):
        parser.error(
            "--locomotion-forward-command requires --recipe running or stilts"
        )
    if args.command == "train" and args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be non-negative (0 disables periodic snapshots)")
    if args.command == "eval":
        if args.recipe == "bridge" and args.evaluation_mode == "skill":
            horizon = round((args.max_episode_s or 20) * 50)
            if horizon < 1000 or args.eval_steps % horizon:
                parser.error("Bridge skill evaluation requires complete episode-horizon multiples, at least 20 seconds each")
        if args.recipe in RECIPES and args.backend == "fork":
            parser.error(f"{args.recipe.title()} evaluation needs control-step physical metrics; fork returns terminal metrics only; use --backend dummy or --backend subproc")
        if args.min_episodes < 0:
            parser.error("--min-episodes must be non-negative")
        if not 0 < args.swing_min_span_deg <= 180:
            parser.error("--swing-min-span-deg must be in (0, 180]")
        if args.min_mean_return is not None and not math.isfinite(args.min_mean_return):
            parser.error("--min-mean-return must be finite")
        if args.recipe == "swing" and args.evaluation_mode == "skill" and (
            args.swing_initial_angle_deg != 0 or args.swing_initial_rate_rad_s != 0
        ):
            parser.error("Swing skill evaluation must start still; use --evaluation-mode pipeline for assisted-start diagnostics")
    if args.command == "train":
        if args.freeze_observation_normalization and args.init_from is None and not (args.recipe in {"backflip", "bridge"} and args.total_timesteps > 4):
            parser.error("--freeze-observation-normalization requires --init-from")
        batch_size = args.num_envs * args.num_steps
        if args.num_minibatches > batch_size:
            parser.error("--num-minibatches cannot exceed num-envs * num-steps")
        if batch_size % args.num_minibatches:
            parser.error("num-envs * num-steps must be divisible by --num-minibatches")
        if args.total_timesteps < batch_size:
            parser.error("--total-timesteps must cover at least one rollout batch")
        if args.onnx_output is not None and not args.export_onnx:
            parser.error("--onnx-output cannot be combined with --no-export-onnx")
        if args.init_from is not None and not args.init_from.expanduser().is_file():
            parser.error(f"--init-from does not exist: {args.init_from}")
    if args.swing_initial_angle_deg > 30:
        parser.error("--swing-initial-angle-deg cannot exceed 30")
    if args.swing_initial_rate_rad_s > 1:
        parser.error("--swing-initial-rate-rad-s cannot exceed 1")
    try:
        _recipe_weight_overrides(args)
        validate_locomotion_forward_command(
            args.recipe, args.locomotion_forward_command
        )
        if args.recipe == "stilts":
            validate_stilt_options(
                args.stilt_height_cm,
                args.stilt_blend,
                args.stilt_mass_kg,
            )
    except ValueError as exc:
        parser.error(str(exc))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    _common(train_parser)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--init-from", type=Path)
    train_parser.add_argument(
        "--freeze-observation-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the loaded observation normalizer fixed; requires --init-from",
    )
    train_parser.add_argument("--checkpoint-interval", type=int, default=100_000)
    train_parser.add_argument("--initial-std", type=_positive_float)
    train_parser.add_argument("--normalize-rewards", action=argparse.BooleanOptionalAction, default=None)
    train_parser.add_argument(
        "--total-timesteps", type=_positive_int, default=1_000_000
    )
    train_parser.add_argument("--num-steps", type=_positive_int, default=24)
    train_parser.add_argument("--num-minibatches", type=_positive_int, default=4)
    train_parser.add_argument("--update-epochs", type=_positive_int)
    train_parser.add_argument("--gamma", type=_probability)
    train_parser.add_argument("--gae-lambda", type=_probability, default=0.95)
    train_parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--clip-coefficient", type=_positive_float)
    train_parser.add_argument(
        "--clip-value-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--entropy-coefficient", type=float)
    train_parser.add_argument("--value-coefficient", type=_positive_float, default=1.0)
    train_parser.add_argument("--max-grad-norm", type=_positive_float)
    train_parser.add_argument("--learning-rate", type=_positive_float)
    train_parser.add_argument(
        "--export-onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--onnx-output", type=Path)

    eval_parser = commands.add_parser("eval")
    _common(eval_parser)
    eval_parser.set_defaults(backend="dummy")
    _policy_source(eval_parser)
    eval_parser.add_argument("--eval-steps", type=_positive_int)
    eval_parser.add_argument("--evaluation-mode", choices=("pipeline", "skill"), default="skill",
                             help="pipeline checks execution only; skill mode applies full per-episode physical criteria")
    eval_parser.add_argument("--swing-min-span-deg", type=_positive_float, default=150.0,
                             help="minimum symmetric bidirectional span (twice the smaller side's peak)")
    eval_parser.add_argument("--min-mean-return", type=float)
    eval_parser.add_argument("--min-episodes", type=int, default=0)

    render_parser = commands.add_parser("render")
    _common(render_parser)
    _policy_source(render_parser)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--episodes", type=_positive_int, default=1)
    render_parser.add_argument("--width", type=_positive_int, default=640)
    render_parser.add_argument("--height", type=_positive_int, default=360)
    render_parser.add_argument("--fps", type=_positive_int, default=25)
    render_parser.add_argument("--sheet-frames", type=_positive_int, default=12)
    render_parser.add_argument(
        "--render-seconds",
        type=_positive_float,
        help=(
            "render-only rollout horizon; defaults to the recipe horizon and "
            "does not change training episodes"
        ),
    )
    render_parser.add_argument(
        "--camera",
        choices=("side", "front", "three-quarter"),
        default="side",
    )
    render_parser.add_argument("--distance", type=_positive_float)

    export_parser = commands.add_parser("export")
    _recipe_argument(export_parser)
    export_parser.add_argument("--output-dir", type=Path)
    export_parser.add_argument("--checkpoint", type=Path)
    export_parser.add_argument("--output", type=Path)

    args = parser.parse_args(argv)
    if args.recipe == "basketball":
        parser.error("basketball requires the recurrent ppo_microduck_balance.py adapter")
    if args.recipe == "backflip" and getattr(args, "backend", None) == "fork":
        args.backend = "dummy"
    if args.command == "train":
        defaults = RECIPE_PPO_DEFAULTS[args.recipe]
        if args.normalize_rewards is None:
            args.normalize_rewards = args.recipe in {"swing", "backflip"}
        for name in (
            "learning_rate",
            "gamma",
            "clip_coefficient",
            "update_epochs",
            "entropy_coefficient",
            "max_grad_norm",
            "initial_std",
        ):
            if getattr(args, name) is None:
                setattr(args, name, defaults[name])
    if args.command == "eval" and args.eval_steps is None:
        args.eval_steps = {"swing": 1200, "running": 600, "backflip": 600, "bridge": 1000}.get(args.recipe, 500)
    _validate_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = {
        "train": train,
        "eval": evaluate,
        "render": render,
        "export": export,
    }[args.command](args)
    print(json.dumps(result, sort_keys=True))
    if args.command == "eval" and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
