"""Train, evaluate, export, and render the MicroDuck drawing policy."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rlx.environments.drawing import (
    ACTION_DIM,
    CONTRACT_VERSION,
    OBS_DIM,
    REWARD_WEIGHTS,
)


PIPELINE_VERSION = "microduck-drawing-pipeline-v1"
DEFAULT_OBS_SCALE = np.ones(OBS_DIM, dtype=np.float32)
DEFAULT_OBS_SCALE[67:70] = 100.0
DEFAULT_OBS_SCALE[73:79] = 10.0
DEFAULT_INITIAL_STD = 0.02
REQUIRED_CASES = (
    ("nominal", {}),
    ("offset", {"offset": (0.004, -0.003)}),
    ("scale", {"scale": 0.9}),
    ("friction", {"friction": 0.8}),
)
TEACHER_SPREAD_CASES = (
    ("nominal", {}),
    ("offset_positive", {"offset": (0.004, -0.003)}),
    ("scale_small", {"scale": 0.9}),
    ("offset_negative", {"offset": (-0.004, 0.003)}),
    ("scale_large", {"scale": 1.1}),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


class DrawingObsScaleExtractor(BaseFeaturesExtractor):
    """SB3-compatible fixed observation scaling embedded in saved actors."""

    def __init__(self, observation_space: Any, scales: Sequence[float] | None = None):
        super().__init__(observation_space, OBS_DIM)
        values = DEFAULT_OBS_SCALE if scales is None else np.asarray(scales, np.float32)
        if values.shape != (OBS_DIM,) or not np.isfinite(values).all():
            raise ValueError(f"observation scales must be finite with shape ({OBS_DIM},)")
        self.register_buffer("scales", torch.as_tensor(values, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations * self.scales


def policy_kwargs(initial_std: float = DEFAULT_INITIAL_STD) -> dict[str, Any]:
    if not math.isfinite(initial_std) or not 0.0 < initial_std < 1.0:
        raise ValueError("initial_std must be finite and in (0, 1)")
    return {
        "activation_fn": nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "features_extractor_class": DrawingObsScaleExtractor,
        "features_extractor_kwargs": {"scales": DEFAULT_OBS_SCALE.tolist()},
        "log_std_init": math.log(initial_std),
        "normalize_images": False,
        "ortho_init": False,
    }


class DeterministicActor(nn.Module):
    """Raw-observation deterministic actor used for ONNX export."""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(observation)
        latent = self.mlp_extractor.forward_actor(features)
        return torch.clamp(self.action_net(latent), -1.0, 1.0)


def checkpoint_metadata(
    *,
    max_episode_s: float,
    seed: int,
    total_timesteps: int,
    assistance: Sequence[float],
    teacher_dataset: Path | None = None,
) -> dict[str, Any]:
    from rlx.environments import drawing as drawing_environment

    return {
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "recipe": "drawing",
        "actuator": "xml",
        "recipe_options": {},
        "reward_weights": dict(REWARD_WEIGHTS),
        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "max_episode_s": float(max_episode_s),
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "assistance_curriculum": [float(value) for value in assistance],
        "observation_scale": DEFAULT_OBS_SCALE.tolist(),
        "teacher_dataset": None if teacher_dataset is None else str(teacher_dataset),
        "policy_output": "15 absolute normalized actuator offsets",
        "ppo_actor_source": "policy network only; teacher controller is not queried",
        "deployment_scope": "simulation_only",
        "hardware_compatible": False,
        "simulation_actuators": {
            "joint_servos": {
                "kp": 8.0,
                "kv": 0.2,
                "torque_limit_nm": 0.96,
            },
            "jaw_servo": {
                "kp": 0.6,
                "torque_limit_nm": 0.04,
            },
            "note": "Strong simulation-only servos; not a hardware deployment recipe.",
        },
        "source_hashes": {
            "pipeline_sha256": sha256_file(Path(__file__)),
            "environment_sha256": sha256_file(Path(drawing_environment.__file__)),
        },
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


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def make_env(
    *,
    seed: int,
    max_episode_s: float,
    assistance: float = 0.0,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    friction: float = 1.2,
) -> Any:
    from rlx.environments.drawing import DrawingEnv

    return DrawingEnv(
        seed=seed,
        max_episode_s=max_episode_s,
        assistance=assistance,
        scale=scale,
        offset=offset,
        friction=friction,
    )


def collect_teacher_dataset(
    *,
    episodes: int,
    seed: int,
    max_episode_s: float,
    assistance: float,
    domain_spread: bool = False,
) -> dict[str, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    scales: list[float] = []
    offset_x: list[float] = []
    offset_y: list[float] = []
    assessments: list[str] = []
    for episode in range(episodes):
        _, environment_options = teacher_environment_options(
            episode, domain_spread=domain_spread
        )
        env = make_env(
            seed=seed + episode,
            max_episode_s=max_episode_s,
            assistance=assistance,
            **environment_options,
        )
        try:
            observation, _ = env.reset(seed=seed + episode)
            for step in range(env.max_steps):
                action = np.asarray(env.teacher_action(), dtype=np.float32)
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(action.copy())
                episode_ids.append(episode)
                step_ids.append(step)
                scales.append(float(environment_options.get("scale", 1.0)))
                offset = environment_options.get("offset", (0.0, 0.0))
                offset_x.append(float(offset[0]))
                offset_y.append(float(offset[1]))
                observation, reward, terminated, truncated, _ = env.step(action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            assessments.append(
                json.dumps(env.assessment(), sort_keys=True, allow_nan=False)
            )
        finally:
            env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "episode": np.asarray(episode_ids, dtype=np.int32),
        "step": np.asarray(step_ids, dtype=np.int32),
        "scale": np.asarray(scales, dtype=np.float32),
        "offset_x": np.asarray(offset_x, dtype=np.float32),
        "offset_y": np.asarray(offset_y, dtype=np.float32),
        "teacher_assistance": np.full(len(actions), assistance, dtype=np.float32),
        "teacher_assessments_json": np.asarray(assessments),
    }


def teacher_environment_options(
    episode: int,
    *,
    domain_spread: bool,
) -> tuple[str, dict[str, Any]]:
    if not domain_spread:
        return "nominal", {}
    name, options = TEACHER_SPREAD_CASES[episode % len(TEACHER_SPREAD_CASES)]
    return name, dict(options)


def collect_dagger_dataset(
    model: Any,
    *,
    episodes: int,
    seed: int,
    max_episode_s: float,
    teacher_probability: float,
    domain_spread: bool = False,
) -> dict[str, np.ndarray]:
    """Label policy-visited unassisted states with corrective teacher actions."""

    if not 0 <= teacher_probability <= 1:
        raise ValueError("teacher_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ids: list[int] = []
    step_ids: list[int] = []
    scales: list[float] = []
    offset_x: list[float] = []
    offset_y: list[float] = []
    assessments: list[str] = []
    for episode in range(episodes):
        _, environment_options = teacher_environment_options(
            episode, domain_spread=domain_spread
        )
        env = make_env(
            seed=seed + episode,
            max_episode_s=max_episode_s,
            assistance=0.0,
            **environment_options,
        )
        try:
            observation, _ = env.reset(seed=seed + episode)
            for step in range(env.max_steps):
                teacher_action = np.asarray(env.teacher_action(), dtype=np.float32)
                policy_action, _ = model.predict(observation, deterministic=True)
                execute_teacher = rng.random() < teacher_probability
                executed_action = teacher_action if execute_teacher else policy_action
                observations.append(np.asarray(observation, dtype=np.float32).copy())
                actions.append(teacher_action.copy())
                episode_ids.append(episode)
                step_ids.append(step)
                scales.append(float(environment_options.get("scale", 1.0)))
                offset = environment_options.get("offset", (0.0, 0.0))
                offset_x.append(float(offset[0]))
                offset_y.append(float(offset[1]))
                observation, reward, terminated, truncated, _ = env.step(executed_action)
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            assessments.append(
                json.dumps(env.assessment(), sort_keys=True, allow_nan=False)
            )
        finally:
            env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "episode": np.asarray(episode_ids, dtype=np.int32),
        "step": np.asarray(step_ids, dtype=np.int32),
        "scale": np.asarray(scales, dtype=np.float32),
        "offset_x": np.asarray(offset_x, dtype=np.float32),
        "offset_y": np.asarray(offset_y, dtype=np.float32),
        "teacher_assistance": np.zeros(len(actions), dtype=np.float32),
        "teacher_assessments_json": np.asarray(assessments),
    }


def concatenate_datasets(
    datasets: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not datasets:
        raise ValueError("at least one dataset is required")
    keys = set(datasets[0])
    if any(set(dataset) != keys for dataset in datasets):
        raise ValueError("datasets must share fields")
    return {
        key: np.concatenate([dataset[key] for dataset in datasets], axis=0)
        for key in keys
    }


def save_teacher_dataset(
    output: Path,
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    max_episode_s: float,
    assistance: float,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    metadata = {
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "seed": seed,
        "max_episode_s": max_episode_s,
        "teacher_assistance": assistance,
        "records": int(len(arrays["actions"])),
        "fields": {key: list(value.shape) for key, value in arrays.items()},
        "dataset_sha256": sha256_file(output),
    }
    assessments = [
        json.loads(value) for value in arrays.get("teacher_assessments_json", [])
    ]
    provenance_errors = teacher_provenance_errors(arrays, assessments)
    metadata["teacher_assessments"] = assessments
    metadata["assistance_provenance_valid"] = not provenance_errors
    metadata["assistance_provenance_errors"] = provenance_errors
    metadata["teacher_passed_all"] = (
        bool(assessments)
        and not provenance_errors
        and all(assessment.get("passed") is True for assessment in assessments)
    )
    write_json(output.with_suffix(output.suffix + ".json"), metadata)
    return metadata


def teacher_provenance_errors(
    arrays: dict[str, np.ndarray],
    assessments: Sequence[dict[str, Any]] | None = None,
) -> list[str]:
    required = {"episode", "teacher_assistance", "teacher_assessments_json"}
    missing = sorted(required.difference(arrays))
    if missing:
        return [f"missing provenance fields: {', '.join(missing)}"]
    episode_ids = np.asarray(arrays["episode"])
    assistance = np.asarray(arrays["teacher_assistance"], dtype=np.float64)
    parsed = (
        list(assessments)
        if assessments is not None
        else [json.loads(value) for value in arrays["teacher_assessments_json"]]
    )
    errors = []
    for episode_index, assessment in enumerate(parsed):
        values = assistance[episode_ids == episode_index]
        if not len(values):
            errors.append(f"episode {episode_index} has no teacher records")
            continue
        unique = np.unique(values)
        assessed = assessment.get("assistance")
        if (
            len(unique) != 1
            or not isinstance(assessed, (int, float))
            or not math.isfinite(float(assessed))
            or not np.isclose(unique[0], float(assessed), atol=1e-7)
        ):
            errors.append(
                f"episode {episode_index} record assistance does not match assessment"
            )
            continue
        expected_unassisted = bool(np.isclose(unique[0], 0.0, atol=1e-7))
        if assessment.get("unassisted") is not expected_unassisted:
            errors.append(
                f"episode {episode_index} unassisted flag contradicts assistance"
            )
    return errors


def load_teacher_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as dataset:
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        arrays = {key: np.asarray(value) for key, value in dataset.items()}
    if observations.ndim != 2 or observations.shape[1] != OBS_DIM:
        raise ValueError(f"teacher observations must have shape (N, {OBS_DIM})")
    if actions.shape != (len(observations), ACTION_DIM):
        raise ValueError(f"teacher actions must have shape (N, {ACTION_DIM})")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("teacher dataset contains non-finite values")
    provenance_errors = teacher_provenance_errors(arrays)
    if provenance_errors:
        raise ValueError(
            "teacher dataset assistance provenance is invalid: "
            + "; ".join(provenance_errors)
        )
    return observations, actions


def create_ppo(env: Any, *, seed: int, learning_rate: float, initial_std: float) -> Any:
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        policy_kwargs=policy_kwargs(initial_std),
        seed=seed,
        device="cpu",
        verbose=0,
    )


def behavior_clone(
    model: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    round_index: int = 0,
) -> list[dict[str, float]]:
    if len(observations) == 0:
        raise ValueError("behavior cloning requires at least one teacher record")
    generator = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    device = model.policy.device
    history: list[dict[str, float]] = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        order = generator.permutation(len(observations))
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            obs = torch.as_tensor(observations[indices], device=device)
            target = torch.as_tensor(actions[indices], device=device)
            prediction = model.policy._predict(obs, deterministic=True)
            loss = torch.mean(torch.square(prediction - target))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "round": round_index,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
            }
        )
    model.policy.set_training_mode(False)
    return history


def write_history_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_histories(
    path: Path,
    bc_history: Sequence[dict[str, Any]],
    ppo_history: Sequence[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        [row["epoch"] for row in bc_history],
        [row["loss"] for row in bc_history],
    )
    axes[0].set(title="Behavior cloning", xlabel="epoch", ylabel="MSE")
    if ppo_history:
        axes[1].plot(
            [row["timesteps"] for row in ppo_history],
            [row["mean_rollout_reward"] for row in ppo_history],
            label="rollout reward",
        )
        axes[1].plot(
            [row["timesteps"] for row in ppo_history],
            [row["loss"] for row in ppo_history],
            label="PPO loss",
        )
        axes[1].legend()
    axes[1].set(title="On-policy PPO", xlabel="timesteps")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


class PPOHistoryCallback:
    """Factory wrapper that avoids importing SB3 at module import time."""

    @staticmethod
    def create(
        history: list[dict[str, float]], total_timesteps: int
    ) -> Any:
        from stable_baselines3.common.callbacks import BaseCallback

        class Callback(BaseCallback):
            def _on_step(self) -> bool:
                return True

            def _on_rollout_end(self) -> None:
                rewards = np.asarray(self.model.rollout_buffer.rewards)
                values = self.model.logger.name_to_value
                steps = min(
                    int(total_timesteps), int(self.model.num_timesteps)
                )
                mean_reward = float(np.mean(rewards))
                history.append(
                    {
                        "timesteps": steps,
                        "mean_rollout_reward": mean_reward,
                        "loss": float(values.get("train/loss", 0.0)),
                        "policy_gradient_loss": float(
                            values.get("train/policy_gradient_loss", 0.0)
                        ),
                        "value_loss": float(values.get("train/value_loss", 0.0)),
                        "approx_kl": float(values.get("train/approx_kl", 0.0)),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "training_progress",
                            "steps": steps,
                            "total": int(total_timesteps),
                            "total_timesteps": int(total_timesteps),
                            "mean_reward": mean_reward,
                            "normalize_rewards": False,
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )

        return Callback()


def curriculum_timesteps(
    total_timesteps: int,
    assistance: Sequence[float],
    *,
    rollout_steps: int = 256,
) -> list[int]:
    if total_timesteps < 1 or not assistance or rollout_steps < 1:
        raise ValueError("PPO curriculum needs timesteps and at least one stage")
    if total_timesteps % rollout_steps:
        raise ValueError("total timesteps must be an exact rollout-block multiple")
    rollout_count = total_timesteps // rollout_steps
    if rollout_count < len(assistance):
        raise ValueError("each assistance stage requires at least one rollout block")
    base, remainder = divmod(rollout_count, len(assistance))
    return [
        (base + int(index < remainder)) * rollout_steps
        for index in range(len(assistance))
    ]


def run_ppo_curriculum(
    model: Any,
    *,
    total_timesteps: int,
    assistance: Sequence[float],
    seed: int,
    max_episode_s: float,
    history: list[dict[str, float]],
    env_factory: Callable[..., Any] = make_env,
) -> None:
    allocations = curriculum_timesteps(
        total_timesteps,
        assistance,
        rollout_steps=int(getattr(model, "n_steps", 256)),
    )
    active_env = model.get_env()
    try:
        for stage, (stage_assistance, timesteps) in enumerate(
            zip(assistance, allocations, strict=False)
        ):
            env = env_factory(
                seed=seed + stage * 10_000,
                max_episode_s=max_episode_s,
                assistance=float(stage_assistance),
            )
            model.set_env(env)
            if active_env is not None and active_env is not env:
                active_env.close()
            active_env = env
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=stage == 0,
                callback=PPOHistoryCallback.create(history, total_timesteps),
                progress_bar=False,
            )
    finally:
        if active_env is not None:
            active_env.close()


def _onnx_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    keys = (
        "pipeline_version",
        "contract_version",
        "recipe",
        "actuator",
        "max_episode_s",
        "observation_dim",
        "action_dim",
    )
    result = {key: json.dumps(metadata[key], sort_keys=True) for key in keys}
    result["recipe_options"] = json.dumps(metadata["recipe_options"], sort_keys=True)
    result["reward_weights"] = json.dumps(metadata["reward_weights"], sort_keys=True)
    result["observation_scale"] = json.dumps(
        metadata["observation_scale"], separators=(",", ":")
    )
    result["rlx_metadata"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return result


def export_policy(model: Any, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    import onnx

    output.parent.mkdir(parents=True, exist_ok=True)
    actor = DeterministicActor(model.policy).cpu().eval()
    sample = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        actor,
        sample,
        output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(output)
    del graph.metadata_props[:]
    for key, value in _onnx_metadata(metadata).items():
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    onnx.save(graph, output)
    parity = compare_onnx(model, output)
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "parity": parity,
    }


def compare_onnx(model: Any, onnx_path: Path) -> dict[str, float | bool]:
    import onnxruntime as ort

    rng = np.random.default_rng(9917)
    observations = rng.normal(0.0, 0.1, size=(8, OBS_DIM)).astype(np.float32)
    torch_actions, _ = model.predict(observations, deterministic=True)
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_actions = session.run(
        None, {session.get_inputs()[0].name: observations}
    )[0]
    error = float(np.max(np.abs(torch_actions - onnx_actions)))
    return {"max_abs_error": error, "passed": error <= 1e-5}


def immutable_training_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"checkpoint", "checkpoint_sha256", "onnx"}
    }


def export_checkpoint(checkpoint: Path, onnx_output: Path) -> dict[str, Any]:
    model = load_ppo(checkpoint)
    sidecar = Path(str(checkpoint) + ".json")
    if sidecar.is_file():
        payload = json.loads(sidecar.read_text())
        metadata = immutable_training_metadata(payload)
    else:
        metadata = checkpoint_metadata(
            max_episode_s=32.0,
            seed=0,
            total_timesteps=0,
            assistance=(0.0,),
        )
        payload = {
            **metadata,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    export = export_policy(model, onnx_output, metadata)
    payload["onnx"] = export
    write_json(sidecar, payload)
    return {
        **export,
        "metadata_path": str(sidecar.resolve()),
        "metadata_sha256": sha256_file(sidecar),
    }


def save_checkpoint(
    model: Any,
    checkpoint: Path,
    metadata: dict[str, Any],
    *,
    onnx_output: Path | None = None,
) -> dict[str, Any]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    actual_checkpoint = checkpoint if checkpoint.suffix == ".zip" else checkpoint.with_suffix(".zip")
    onnx_path = onnx_output or actual_checkpoint.with_name(
        actual_checkpoint.name.removesuffix(".zip") + ".onnx"
    )
    export = export_policy(model, onnx_path, metadata)
    payload = {
        **metadata,
        "checkpoint": str(actual_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(actual_checkpoint),
        "onnx": export,
    }
    write_json(Path(str(actual_checkpoint) + ".json"), payload)
    return payload


def load_ppo(checkpoint: Path, env: Any | None = None) -> Any:
    from stable_baselines3 import PPO

    return PPO.load(checkpoint, env=env, device="cpu")


def resolve_onnx(checkpoint: Path, onnx_output: Path | None = None) -> Path:
    if checkpoint.suffix == ".onnx":
        return checkpoint
    candidate = onnx_output or checkpoint.with_name(
        checkpoint.name.removesuffix(".zip") + ".onnx"
    )
    if candidate.is_file():
        return candidate
    export_checkpoint(checkpoint, candidate)
    return candidate


class OnnxPolicy:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        batch = np.asarray(observation, dtype=np.float32)[None, :]
        action = self.session.run(None, {self.input_name: batch})[0][0]
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)


def rollout_episode(
    policy: Callable[[np.ndarray], np.ndarray],
    env: Any,
    *,
    seed: int,
    action_override: str | None = None,
) -> dict[str, Any]:
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    while True:
        if action_override == "null":
            action = np.zeros(ACTION_DIM, dtype=np.float32)
        elif action_override == "open_jaw":
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[-1] = 1.0
        else:
            action = policy(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break
    assessment = env.assessment()
    assessment.setdefault(
        "ink_contact_fraction",
        float(assessment.get("contact_steps", 0)) / max(1, steps),
    )
    return {
        "seed": seed,
        "steps": steps,
        "return": total_reward,
        "drawing_assessment": assessment,
        "drawing_payload": env.drawing_payload(),
    }


def aggregate_evaluations(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    assessments = [result["drawing_assessment"] for result in results]
    passed = bool(assessments) and all(item.get("passed") is True for item in assessments)
    numeric_keys = (
        "coverage",
        "precision",
        "symmetric_chamfer_m",
        "length_ratio",
        "grasp_fraction",
        "pen_up_leak_fraction",
        "max_normal_force_n",
        "ink_contact_fraction",
    )
    means: dict[str, float | None] = {}
    for key in numeric_keys:
        values = [
            float(item[key])
            for item in assessments
            if item.get(key) is not None and math.isfinite(float(item[key]))
        ]
        means[key] = None if not values else float(np.mean(values))
    return {
        "passed": passed,
        "required_episode_count": len(results),
        "passed_episode_count": sum(item.get("passed") is True for item in assessments),
        "mean_return": (
            None if not results else float(np.mean([item["return"] for item in results]))
        ),
        "means": means,
    }


def studio_drawing_assessment(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = aggregate_evaluations(results)
    episodes = []
    for episode_index, result in enumerate(results):
        assessment = dict(result["drawing_assessment"])
        episodes.append(
            {
                **assessment,
                "env_index": 0,
                "episode_index": episode_index,
                "measured_steps": int(result["steps"]),
                "seed": int(result["seed"]),
                "case": result.get("case", "nominal"),
            }
        )
    return {
        **aggregate,
        "unassisted": bool(episodes)
        and all(
            episode.get("unassisted") is True
            and float(episode.get("assistance", 1.0)) == 0.0
            for episode in episodes
        ),
        "episodes": episodes,
        "criteria": {
            "requires_unassisted": True,
            "all_required_cases_and_seeds_must_pass": True,
        },
    }


def evaluate_onnx(
    onnx_path: Path,
    *,
    seed: int,
    eval_episodes: int,
    max_episode_s: float,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    policy = OnnxPolicy(onnx_path)
    cases: dict[str, list[dict[str, Any]]] = {}
    required: list[dict[str, Any]] = []
    for case_index, (name, case_options) in enumerate(REQUIRED_CASES):
        case_results = []
        for episode in range(eval_episodes):
            episode_seed = seed + case_index * 1_000 + episode
            env = make_env(
                seed=episode_seed,
                max_episode_s=max_episode_s,
                assistance=0.0,
                **case_options,
            )
            try:
                result = rollout_episode(policy, env, seed=episode_seed)
                result["case"] = name
                result["environment"] = {
                    "assistance": 0.0,
                    "scale": case_options.get("scale", 1.0),
                    "offset": list(case_options.get("offset", (0.0, 0.0))),
                    "friction": case_options.get("friction", 1.2),
                }
                case_results.append(result)
                required.append(result)
            finally:
                env.close()
        cases[name] = case_results
    controls: dict[str, dict[str, Any]] = {}
    for control in ("null", "open_jaw"):
        env = make_env(seed=seed + 50_000, max_episode_s=max_episode_s, assistance=0.0)
        try:
            controls[control] = rollout_episode(
                policy, env, seed=seed + 50_000, action_override=control
            )
        finally:
            env.close()
    drawing_assessment = studio_drawing_assessment(required)
    source = onnx_path.resolve()
    checkpoint = (
        checkpoint_path.resolve()
        if checkpoint_path is not None
        else source.with_name(source.name.removesuffix(".onnx") + ".zip")
    )
    metadata_path = Path(str(checkpoint) + ".json")
    source_files = {str(source): sha256_file(source)}
    training_source_hashes = None
    if metadata_path.is_file():
        source_files[str(metadata_path.resolve())] = sha256_file(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        if isinstance(metadata.get("source_hashes"), dict):
            training_source_hashes = dict(metadata["source_hashes"])
    from rlx.environments import drawing as drawing_environment

    evaluation_source_hashes = {
        "pipeline_sha256": sha256_file(Path(__file__)),
        "environment_sha256": sha256_file(Path(drawing_environment.__file__)),
    }
    environment_source_match = (
        training_source_hashes is not None
        and training_source_hashes.get("environment_sha256")
        == evaluation_source_hashes["environment_sha256"]
    )
    pipeline_passed = all(
        math.isfinite(float(result["return"])) and result["steps"] > 0
        for result in required
    )
    passed = bool(pipeline_passed and drawing_assessment["passed"])
    return {
        "schema_version": 1,
        "command": "eval",
        "recipe": "drawing",
        "evaluation_mode": "skill",
        "created_at": utc_now(),
        "pipeline_version": PIPELINE_VERSION,
        "contract_version": CONTRACT_VERSION,
        "source": str(source),
        "source_type": "policy",
        "source_sha256": source_files[str(source)],
        "metadata_sha256": (
            source_files.get(str(metadata_path.resolve()))
            if metadata_path.is_file()
            else None
        ),
        "source_files_sha256": source_files,
        "training_source_hashes": training_source_hashes,
        "evaluation_source_hashes": evaluation_source_hashes,
        "environment_source_match": environment_source_match,
        "reevaluation_notice": (
            None
            if environment_source_match
            else "Evaluation used the current environment source; training metadata "
            "records the environment source loaded for training."
        ),
        "onnx": str(source),
        "onnx_sha256": source_files[str(source)],
        "evaluation_policy": "deterministic ONNX",
        "acceptance_scope": "strict unassisted required cases only",
        "finite": pipeline_passed,
        "pipeline_passed": pipeline_passed,
        "skill_status": "passed" if passed else "failed",
        "passed": passed,
        "success": passed,
        "evaluation_settings_match": True,
        "evaluation": {
            "mode": "skill",
            "seed": seed,
            "eval_episodes": eval_episodes,
            "environment": {
                "recipe": "drawing",
                "actuator": "xml",
                "max_episode_s": max_episode_s,
                "assistance": 0.0,
                "recipe_options": {},
                "reward_weights": dict(REWARD_WEIGHTS),
            },
        },
        "cases": cases,
        "controls": controls,
        "drawing_assessment": drawing_assessment,
    }


def comparison_evidence(
    bc_report: dict[str, Any], ppo_report: dict[str, Any]
) -> dict[str, Any]:
    bc = bc_report["drawing_assessment"]
    ppo = ppo_report["drawing_assessment"]
    metric_deltas = {}
    for key in ("coverage", "precision", "grasp_fraction"):
        before = bc["means"].get(key)
        after = ppo["means"].get(key)
        metric_deltas[key] = None if before is None or after is None else after - before
    return {
        "bc_passed": bc["passed"],
        "ppo_passed": ppo["passed"],
        "mean_return_delta": (
            None
            if bc["mean_return"] is None or ppo["mean_return"] is None
            else ppo["mean_return"] - bc["mean_return"]
        ),
        "metric_deltas": metric_deltas,
        "ppo_improvement_claimed": False,
        "note": "Deltas are reported as finite-run evidence, not a success guarantee.",
    }


def evaluation_with_bc_baseline(
    ppo_report: dict[str, Any],
    bc_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        **ppo_report,
        "bc_baseline": bc_report,
        "comparison": comparison_evidence(bc_report, ppo_report),
    }


def evaluate_policy_with_neighboring_bc(
    onnx_path: Path,
    *,
    seed: int,
    eval_episodes: int,
    max_episode_s: float,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    report = evaluate_onnx(
        onnx_path,
        seed=seed,
        eval_episodes=eval_episodes,
        max_episode_s=max_episode_s,
        checkpoint_path=checkpoint_path,
    )
    bc_onnx = onnx_path.with_name("bc_policy.onnx")
    if onnx_path.name != "policy.onnx" or not bc_onnx.is_file():
        return report
    bc_checkpoint = onnx_path.with_name("bc_policy.zip")
    bc_report = evaluate_onnx(
        bc_onnx,
        seed=seed,
        eval_episodes=eval_episodes,
        max_episode_s=max_episode_s,
        checkpoint_path=bc_checkpoint if bc_checkpoint.is_file() else None,
    )
    return evaluation_with_bc_baseline(report, bc_report)


def _trace_strokes(trace: Sequence[Sequence[Any]]) -> list[list[tuple[float, float]]]:
    strokes: dict[int, list[tuple[float, float]]] = {}
    for row in trace:
        strokes.setdefault(int(row[4]), []).append((float(row[2]), float(row[3])))
    return [strokes[key] for key in sorted(strokes)]


def contact_svg(trace: Sequence[Sequence[Any]]) -> str:
    strokes = _trace_strokes(trace)
    paths = []
    for stroke in strokes:
        if not stroke:
            continue
        commands = [f"M {stroke[0][0] * 1000:.3f} {-stroke[0][1] * 1000:.3f}"]
        commands.extend(f"L {y * 1000:.3f} {-z * 1000:.3f}" for y, z in stroke[1:])
        paths.append(f'  <path d="{" ".join(commands)}" />')
    return "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-105 -310 210 170">',
            "  <metadata>Actual MuJoCo pencil-tip canvas contacts</metadata>",
            '  <rect x="-105" y="-310" width="210" height="170" fill="#faf7ed"/>',
            '  <g fill="none" stroke="#171717" stroke-width="0.8" '
            'stroke-linecap="round" stroke-linejoin="round">',
            *paths,
            "  </g>",
            "</svg>",
        )
    )


def draw_contact_image(
    trace: Sequence[Sequence[Any]], *, width: int = 600, height: int = 460
) -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (width, height), "#faf7ed")
    draw = ImageDraw.Draw(image)
    margin = 24

    def project(point: tuple[float, float]) -> tuple[int, int]:
        y, z = point
        px = margin + int((y + 0.095) / 0.190 * (width - 2 * margin))
        py = height - margin - int((z - 0.1515) / 0.146 * (height - 2 * margin))
        return px, py

    for stroke in _trace_strokes(trace):
        if len(stroke) == 1:
            x, y = project(stroke[0])
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="#171717")
        elif stroke:
            draw.line([project(point) for point in stroke], fill="#171717", width=2)
    return image


def add_contact_geoms(scene: Any, payload: dict[str, Any]) -> int:
    """Add actual same-stroke contact segments to the transient render scene."""

    import mujoco

    points = payload.get("points", [])
    added = 0
    for previous, current in zip(points, points[1:]):
        if int(previous[3]) != int(current[3]) or scene.ngeom >= scene.maxgeom:
            continue
        start = np.asarray(previous[:3], dtype=np.float64)
        end = np.asarray(current[:3], dtype=np.float64)
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.array([0.04, 0.04, 0.04, 1.0], dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.0007,
            start,
            end,
        )
        scene.ngeom += 1
        added += 1
    return added


def policy_source_label(onnx_path: Path) -> str:
    name = onnx_path.stem.lower()
    if "teacher" in name:
        return "teacher policy"
    if name.startswith("bc_") or "baseline" in name:
        return "BC policy"
    return "PPO policy"


def write_raw_contact_trace(
    output: Path,
    trace: Sequence[Sequence[Any]],
    *,
    controller: str,
    seed: int,
) -> dict[str, str]:
    columns = ("t", "x", "y", "z", "strokeId", "force", "requested_down")
    rows = [list(row) for row in trace]
    csv_path = output / "raw_contact_trace.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
    json_path = output / "raw_contact_trace.json"
    write_json(
        json_path,
        {
            "controller": controller,
            "seed": seed,
            "columns": list(columns),
            "rows": rows,
        },
    )
    return {
        "csv": str(csv_path.resolve()),
        "json": str(json_path.resolve()),
    }


def render_onnx(
    onnx_path: Path | None,
    output: Path,
    *,
    seed: int,
    render_seconds: float,
    max_episode_s: float,
    width: int = 640,
    height: int = 480,
    fps: int = 25,
    policy_label: str | None = None,
    controller: str = "onnx",
    checkpoint_argument: Path | None = None,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw

    from rlx.environments.drawing_reference import reference_svg, swimming_duck_strokes

    if controller not in {"onnx", "teacher"}:
        raise ValueError("controller must be 'onnx' or 'teacher'")
    if controller == "onnx" and onnx_path is None:
        raise ValueError("ONNX rendering requires an ONNX source")
    output.mkdir(parents=True, exist_ok=True)
    policy = OnnxPolicy(onnx_path) if controller == "onnx" else None
    horizon = min(render_seconds, max_episode_s)
    env = make_env(seed=seed, max_episode_s=max_episode_s, assistance=0.0)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames: list[np.ndarray] = []
    observation, _ = env.reset(seed=seed)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.07, 0.0, 0.20]
    camera.distance = 0.58
    camera.azimuth = -45
    camera.elevation = -12
    control_steps = max(1, round(horizon / 0.02))
    frame_stride = max(1, round(50 / fps))
    total_reward = 0.0
    label_text = (
        policy_label
        or (
            policy_source_label(onnx_path)
            if controller == "onnx" and onnx_path is not None
            else "teacher controller"
        )
    )
    visible_contact_segments = 0
    try:
        for step in range(control_steps):
            action = (
                policy(observation)
                if policy is not None
                else np.asarray(env.teacher_action(), dtype=np.float32)
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            if step % frame_stride == 0:
                renderer.update_scene(env.data, camera=camera)
                visible_contact_segments = add_contact_geoms(
                    renderer.scene, env.drawing_payload()
                )
                frame = Image.fromarray(renderer.render())
                ink = draw_contact_image(env.trace, width=180, height=138)
                frame.paste(ink, (width - 192, height - 150))
                label = ImageDraw.Draw(frame)
                label.text(
                    (12, 12),
                    f"{label_text} | actual contact ink | t={env.data.time:.2f}s",
                    fill="white",
                )
                frames.append(np.asarray(frame))
            if terminated or truncated:
                break
        trace = [list(row) for row in env.trace]
        assessment = env.assessment()
        drawing_payload = env.drawing_payload()
    finally:
        renderer.close()
        env.close()
    if not frames:
        raise RuntimeError("render produced no frames")
    imageio.mimsave(output / "rollout.mp4", frames, fps=fps, macro_block_size=1)
    sheet_count = min(12, len(frames))
    indices = np.linspace(0, len(frames) - 1, sheet_count).astype(int)
    thumb_width, thumb_height = width // 3, height // 3
    sheet = Image.new("RGB", (thumb_width * 4, thumb_height * 3), "#111418")
    for index, frame_index in enumerate(indices):
        thumb = Image.fromarray(frames[frame_index]).resize((thumb_width, thumb_height))
        sheet.paste(thumb, ((index % 4) * thumb_width, (index // 4) * thumb_height))
    sheet.save(output / "frame_sheet.png")
    actual = draw_contact_image(trace)
    actual.save(output / "actual_contact_drawing.png")
    (output / "actual_contact_drawing.svg").write_text(contact_svg(trace))
    (output / "reference.svg").write_text(reference_svg())
    reference_trace = []
    for stroke_id, stroke in enumerate(swimming_duck_strokes(), start=1):
        reference_trace.extend(
            [
                0.0,
                0.0,
                float(point[0]),
                0.2245 + float(point[1]),
                stroke_id,
                0.0,
                True,
            ]
            for point in stroke
        )
    draw_contact_image(reference_trace).save(output / "reference.png")
    raw_contact_trace = write_raw_contact_trace(
        output, trace, controller=controller, seed=seed
    )
    from rlx.environments import drawing as drawing_environment

    report = {
        "created_at": utc_now(),
        "controller": controller,
        "source": (
            str(onnx_path.resolve())
            if controller == "onnx" and onnx_path is not None
            else None
        ),
        "source_type": "policy" if controller == "onnx" else "teacher_controller",
        "onnx": (
            str(onnx_path.resolve())
            if controller == "onnx" and onnx_path is not None
            else None
        ),
        "checkpoint_argument": (
            None
            if checkpoint_argument is None
            else str(checkpoint_argument.resolve())
        ),
        "checkpoint_used": controller == "onnx",
        "controller_provenance": (
            {
                "environment_source": str(
                    Path(drawing_environment.__file__).resolve()
                ),
                "environment_sha256": sha256_file(
                    Path(drawing_environment.__file__).resolve()
                ),
                "pipeline_source": str(Path(__file__).resolve()),
                "pipeline_sha256": sha256_file(Path(__file__).resolve()),
                "method": "DrawingEnv.teacher_action",
            }
            if controller == "teacher"
            else None
        ),
        "seed": seed,
        "render_seconds": horizon,
        "trajectory_horizon_seconds": max_episode_s,
        "frames": len(frames),
        "return": total_reward,
        "drawing_assessment": assessment,
        "drawing_payload": drawing_payload,
        "raw_contact_trace": raw_contact_trace,
        "ink_source": "actual MuJoCo pencil-tip contact trace",
        "policy_source_label": label_text,
        "visible_contact_segments": visible_contact_segments,
    }
    write_json(output / "render.json", report)
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and (
        not args.output.is_dir() or any(args.output.iterdir())
    ):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_path = args.dataset or args.output / "teacher_dataset.npz"
    if args.dataset is None:
        arrays = collect_teacher_dataset(
            episodes=args.teacher_episodes,
            seed=args.seed,
            max_episode_s=args.max_episode_s,
            assistance=args.teacher_assistance,
            domain_spread=args.teacher_domain_spread,
        )
        save_teacher_dataset(
            dataset_path,
            arrays,
            seed=args.seed,
            max_episode_s=args.max_episode_s,
            assistance=args.teacher_assistance,
        )
    observations, actions = load_teacher_dataset(dataset_path)
    initial_env = make_env(
        seed=args.seed,
        max_episode_s=args.max_episode_s,
        assistance=float(args.assistance[0]),
    )
    model = create_ppo(
        initial_env,
        seed=args.seed,
        learning_rate=args.learning_rate,
        initial_std=args.initial_std,
    )
    bc_history = behavior_clone(
        model,
        observations,
        actions,
        epochs=args.bc_epochs,
        batch_size=args.bc_batch_size,
        learning_rate=args.bc_learning_rate,
        seed=args.seed,
    )
    dataset_parts = [
        {
            key: value.copy()
            for key, value in np.load(dataset_path).items()
        }
    ]
    dagger_artifacts = []
    for round_index in range(1, args.dagger_rounds + 1):
        teacher_probability = args.dagger_teacher_probability ** round_index
        dagger = collect_dagger_dataset(
            model,
            episodes=args.dagger_episodes,
            seed=args.seed + round_index * 100_000,
            max_episode_s=args.max_episode_s,
            teacher_probability=teacher_probability,
            domain_spread=args.teacher_domain_spread,
        )
        dagger_path = args.output / f"dagger_round_{round_index}.npz"
        dagger_metadata = save_teacher_dataset(
            dagger_path,
            dagger,
            seed=args.seed + round_index * 100_000,
            max_episode_s=args.max_episode_s,
            assistance=0.0,
        )
        dagger_metadata["teacher_probability"] = teacher_probability
        write_json(dagger_path.with_suffix(dagger_path.suffix + ".json"), dagger_metadata)
        dagger_artifacts.append(dagger_metadata)
        dataset_parts.append(dagger)
        combined = concatenate_datasets(dataset_parts)
        observations = combined["observations"]
        actions = combined["actions"]
        bc_history.extend(
            behavior_clone(
                model,
                observations,
                actions,
                epochs=args.dagger_bc_epochs,
                batch_size=args.bc_batch_size,
                learning_rate=args.bc_learning_rate,
                seed=args.seed + round_index,
                round_index=round_index,
            )
        )
    metadata = checkpoint_metadata(
        max_episode_s=args.max_episode_s,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        assistance=args.assistance,
        teacher_dataset=dataset_path,
    )
    metadata["teacher_dataset_sha256"] = sha256_file(dataset_path)
    metadata["dagger"] = {
        "rounds": args.dagger_rounds,
        "episodes_per_round": args.dagger_episodes,
        "initial_teacher_probability": args.dagger_teacher_probability,
        "artifacts": dagger_artifacts,
    }
    metadata["training_config"] = {
        "bc_epochs": args.bc_epochs,
        "bc_batch_size": args.bc_batch_size,
        "bc_learning_rate": args.bc_learning_rate,
        "dagger_bc_epochs": args.dagger_bc_epochs,
        "learning_rate": args.learning_rate,
        "initial_std": args.initial_std,
        "eval_episodes": args.eval_episodes,
        "render_seconds": args.render_seconds,
        "teacher_domain_spread": args.teacher_domain_spread,
    }
    bc_metadata = {**metadata, "training_stage": "behavior_cloning_baseline"}
    bc_artifact = save_checkpoint(
        model,
        args.output / "bc_policy.zip",
        bc_metadata,
    )
    ppo_history: list[dict[str, float]] = []
    run_ppo_curriculum(
        model,
        total_timesteps=args.total_timesteps,
        assistance=args.assistance,
        seed=args.seed,
        max_episode_s=args.max_episode_s,
        history=ppo_history,
    )
    ppo_metadata = {**metadata, "training_stage": "behavior_cloning_then_on_policy_ppo"}
    policy_artifact = save_checkpoint(
        model,
        args.output / "policy.zip",
        ppo_metadata,
        onnx_output=args.output / "policy.onnx",
    )
    write_history_csv(args.output / "bc_history.csv", bc_history)
    write_history_csv(args.output / "ppo_history.csv", ppo_history)
    plot_histories(args.output / "training_curves.png", bc_history, ppo_history)
    bc_eval = evaluate_onnx(
        Path(bc_artifact["onnx"]["path"]),
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        max_episode_s=args.max_episode_s,
        checkpoint_path=Path(bc_artifact["checkpoint"]),
    )
    ppo_eval = evaluate_onnx(
        Path(policy_artifact["onnx"]["path"]),
        seed=args.seed,
        eval_episodes=args.eval_episodes,
        max_episode_s=args.max_episode_s,
        checkpoint_path=Path(policy_artifact["checkpoint"]),
    )
    evaluation = evaluation_with_bc_baseline(ppo_eval, bc_eval)
    write_json(args.output / "eval.json", evaluation)
    render_report = None
    if args.render_after_train:
        render_report = render_onnx(
            Path(policy_artifact["onnx"]["path"]),
            args.output / "render",
            seed=args.seed,
            render_seconds=args.render_seconds,
            max_episode_s=args.max_episode_s,
            policy_label="PPO policy",
            controller="onnx",
        )
    summary = {
        "created_at": utc_now(),
        "policy": policy_artifact,
        "bc_policy": bc_artifact,
        "evaluation": str((args.output / "eval.json").resolve()),
        "render": render_report,
        "training_budget": {
            "teacher_records": len(observations),
            "bc_epochs": args.bc_epochs,
            "dagger_rounds": args.dagger_rounds,
            "ppo_timesteps": args.total_timesteps,
        },
        "success_guarantee": False,
    }
    write_json(args.output / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--total-timesteps", type=_positive_int, required=True)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--max-episode-s", type=_positive_float, default=32.0)
    train_parser.add_argument(
        "--assistance",
        type=_probability,
        nargs="+",
        default=[1.0, 0.3, 0.0],
        metavar="LEVEL",
    )
    train_parser.add_argument("--teacher-assistance", type=_probability, default=0.0)
    train_parser.add_argument("--teacher-episodes", type=_positive_int, default=4)
    train_parser.add_argument(
        "--teacher-domain-spread",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train_parser.add_argument("--dataset", type=Path)
    train_parser.add_argument("--bc-epochs", type=_positive_int, default=40)
    train_parser.add_argument("--dagger-rounds", type=int, default=2)
    train_parser.add_argument("--dagger-episodes", type=_positive_int, default=2)
    train_parser.add_argument("--dagger-bc-epochs", type=_positive_int, default=12)
    train_parser.add_argument(
        "--dagger-teacher-probability", type=_probability, default=0.5
    )
    train_parser.add_argument("--bc-batch-size", type=_positive_int, default=512)
    train_parser.add_argument("--bc-learning-rate", type=_positive_float, default=3e-4)
    train_parser.add_argument("--learning-rate", type=_positive_float, default=3e-5)
    train_parser.add_argument("--initial-std", type=_positive_float, default=DEFAULT_INITIAL_STD)
    train_parser.add_argument("--eval-episodes", type=_positive_int, default=1)
    train_parser.add_argument("--render-seconds", type=_positive_float, default=32.0)
    train_parser.add_argument(
        "--render-after-train",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    data_parser = commands.add_parser("data")
    data_parser.add_argument("--output", type=Path, required=True)
    data_parser.add_argument("--episodes", type=_positive_int, default=4)
    data_parser.add_argument("--seed", type=int, default=1)
    data_parser.add_argument("--max-episode-s", type=_positive_float, default=32.0)
    data_parser.add_argument("--assistance", type=_probability, default=0.0)
    data_parser.add_argument(
        "--teacher-domain-spread",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    eval_parser = commands.add_parser("eval")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--eval-output", type=Path, required=True)
    eval_parser.add_argument("--eval-episodes", type=_positive_int, default=1)
    eval_parser.add_argument("--seed", type=int, default=1)
    eval_parser.add_argument("--max-episode-s", type=_positive_float, default=32.0)

    render_parser = commands.add_parser("render")
    render_parser.add_argument("--checkpoint", type=Path, required=True)
    render_parser.add_argument("--render-output", type=Path, required=True)
    render_parser.add_argument("--render-seconds", type=_positive_float, default=32.0)
    render_parser.add_argument("--seed", type=int, default=1)
    render_parser.add_argument("--max-episode-s", type=_positive_float, default=32.0)
    render_parser.add_argument("--policy-label")
    render_parser.add_argument(
        "--controller",
        choices=("onnx", "teacher"),
        default="onnx",
    )

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--onnx-output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("checkpoint", "dataset"):
        path = getattr(args, name, None)
        if path is not None and not path.is_file():
            parser.error(f"{name} does not exist: {path}")
    if hasattr(args, "max_episode_s") and not 4 <= args.max_episode_s <= 120:
        parser.error("--max-episode-s must be between 4 and 120")
    if (
        args.command == "train"
        and (
            not 0.0 < args.initial_std < 1.0
            or (
                args.render_after_train
                and args.render_seconds > args.max_episode_s
            )
        )
    ):
        parser.error("--initial-std must be in (0, 1) and render horizon must fit episode")
    if args.command == "train" and args.dagger_rounds < 0:
        parser.error("--dagger-rounds must be non-negative")
    if args.command == "train" and (
        args.total_timesteps % 256
        or args.total_timesteps < 256 * len(args.assistance)
    ):
        parser.error(
            "--total-timesteps must be a multiple of 256 with at least one "
            "rollout block per assistance stage"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "data":
        arrays = collect_teacher_dataset(
            episodes=args.episodes,
            seed=args.seed,
            max_episode_s=args.max_episode_s,
            assistance=args.assistance,
            domain_spread=args.teacher_domain_spread,
        )
        result = save_teacher_dataset(
            args.output,
            arrays,
            seed=args.seed,
            max_episode_s=args.max_episode_s,
            assistance=args.assistance,
        )
    elif args.command == "train":
        result = train(args)
    elif args.command == "export":
        result = export_checkpoint(args.checkpoint, args.onnx_output)
    elif args.command == "eval":
        onnx_path = resolve_onnx(args.checkpoint)
        result = evaluate_policy_with_neighboring_bc(
            onnx_path,
            seed=args.seed,
            eval_episodes=args.eval_episodes,
            max_episode_s=args.max_episode_s,
            checkpoint_path=(
                args.checkpoint if args.checkpoint.suffix != ".onnx" else None
            ),
        )
        write_json(args.eval_output, result)
    else:
        onnx_path = (
            resolve_onnx(args.checkpoint)
            if args.controller == "onnx"
            else None
        )
        result = render_onnx(
            onnx_path,
            args.render_output,
            seed=args.seed,
            render_seconds=args.render_seconds,
            max_episode_s=args.max_episode_s,
            policy_label=args.policy_label,
            controller=args.controller,
            checkpoint_argument=args.checkpoint,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
