"""Fork-safe MicroDuck dance training, export, evaluation, viewing, and demo."""

from __future__ import annotations

import argparse
import importlib
import os
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rlx.environments.microduck import make_microduck_env

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLIP_DIRECTORY = REPOSITORY_ROOT / "assets" / "clips"
CLIP_NAME = "dance-120bpm"
BEHAVIOR_ID = "imitate"
DEFAULT_WEIGHT_OVERRIDES = {
    "travel": 0.0,
    "stay_home": 1.0,
    "face_home": 1.0,
}
DANCE_ACTION_LIMIT = 1.0


def _weight_overrides(value: str) -> dict[str, float]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--weight-overrides must be valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("--weight-overrides must be a JSON object")  # noqa: TRY004
    merged = dict(DEFAULT_WEIGHT_OVERRIDES)
    for key, weight in decoded.items():
        if (
            not isinstance(key, str)
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
        ):
            raise ValueError("reward weights must map string names to numbers")  # noqa: TRY004
        if weight < 0:
            raise ValueError(f"reward weight {key!r} must be non-negative")
        merged[key] = float(weight)
    return merged


def _environment(args: argparse.Namespace, *, normalize: bool):
    return make_microduck_env(
        num_envs=args.num_envs,
        backend=args.backend,
        seed=args.seed,
        behavior_id=BEHAVIOR_ID,
        clip_name=CLIP_NAME,
        clip_directory=CLIP_DIRECTORY,
        weight_overrides=_weight_overrides(args.weight_overrides),
        domain_rand=args.domain_rand,
        obs_noise=args.obs_noise,
        action_delay=args.action_delay,
        random_yaw=args.random_yaw,
        max_episode_s=args.max_episode_s,
        normalize_observations=normalize,
    )


def _training_metadata(args: argparse.Namespace, steps: int) -> dict[str, Any]:
    return {
    "recipe": "dance",
    "policy_action_limit": DANCE_ACTION_LIMIT,
        "behavior_id": BEHAVIOR_ID,
        "clip_name": CLIP_NAME,
        "steps": steps,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "max_episode_s": args.max_episode_s,
        "backend": args.backend,
        "reward_weights": _weight_overrides(args.weight_overrides),
        "randomization": {
            "domain_rand": args.domain_rand,
            "obs_noise": args.obs_noise,
            "action_delay": args.action_delay,
            "random_yaw": args.random_yaw,
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
        },
    }


def _training_progress_callback(
    total_timesteps: int, num_envs: int, rollout_steps: int
) -> Callable[[dict[str, Any], int], None]:
    import numpy as np

    completed_episodes = 0
    rollout_size = num_envs * rollout_steps

    def report(info: dict[str, Any], step: int) -> None:
        nonlocal completed_episodes

        current_step = min(total_timesteps, step + num_envs)
        episode_returns: list[float] = []
        if "episode" in info and "_episode" in info:
            mask = np.asarray(info["_episode"], dtype=bool)
            if mask.any():
                episode_returns = (
                    np.asarray(info["episode"]["r"])[mask].astype(float).tolist()
                )
                completed_episodes += len(episode_returns)

        rollout_complete = current_step == total_timesteps or (
            rollout_size > 0 and current_step % rollout_size == 0
        )
        if not episode_returns and not rollout_complete:
            return

        print(
            json.dumps(
                {
                    "event": "training_progress",
                    "steps": current_step,
                    "total": total_timesteps,
                    "mean_reward": (
                        float(np.mean(episode_returns)) if episode_returns else None
                    ),
                    "episodes": completed_episodes,
                }
            ),
            flush=True,
        )

    return report


def train(args: argparse.Namespace) -> dict[str, object]:
    env = _environment(args, normalize=True)
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np

        from rlx.algorithms.ppo import PPO, PPOConfig
        from rlx.buffers.rollout_buffer import RolloutBuffer
        from rlx.models.microduck import create_actor_critic, save_checkpoint

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
        network = create_actor_critic(action_limit=DANCE_ACTION_LIMIT)
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
        algorithm.train(
            args.total_timesteps,
            callback=_training_progress_callback(
                args.total_timesteps, args.num_envs, args.num_steps
            ),
        )
        save_checkpoint(
            args.checkpoint,
            network,
            env.observation_rms.mean,
            env.observation_rms.var,
            env.observation_rms.count,
            epsilon=env.epsilon,
            clip=env.clip,
            metadata=_training_metadata(args, algorithm.step),
        )
    finally:
        env.close()

    result: dict[str, object] = {
        "command": "train",
        "checkpoint": str(args.checkpoint),
        "steps": algorithm.step,
        "metadata": str(args.checkpoint.with_suffix(args.checkpoint.suffix + ".json")),
    }
    if args.export_onnx:
        from rlx.export.microduck_onnx import export_deterministic_actor

        target = args.onnx_output or args.checkpoint.with_suffix(".onnx")
        result["onnx"] = str(export_deterministic_actor(args.checkpoint, target))
    return result


def export(args: argparse.Namespace) -> dict[str, object]:
    from rlx.export.microduck_onnx import export_deterministic_actor

    path = export_deterministic_actor(args.checkpoint, args.output)
    return {
        "command": "export",
        "checkpoint": str(args.checkpoint),
        "output": str(path),
    }


def _policy_from_checkpoint(checkpoint: Path) -> Callable[[Any], Any]:
    from rlx.models.microduck import load_checkpoint, normalize_observations

    loaded = load_checkpoint(checkpoint)

    def infer(observation: Any) -> Any:
        normalized = normalize_observations(
            observation,
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
        # Exported policies contain observation normalization, so raw environment
        # observations are passed directly to ONNX.
        raw = np.asarray(observation, dtype=np.float32)
        return session.run([output_name], {input_name: raw})[0]

    return infer


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    env = _environment(args, normalize=False)
    try:
        import numpy as np

        source = args.policy or args.checkpoint
        infer = (
            _policy_from_onnx(args.policy)
            if args.policy is not None
            else _policy_from_checkpoint(args.checkpoint)
        )
        observation, state, _ = env.reset(None)
        rollout_returns = np.zeros(args.num_envs, dtype=np.float64)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []
        termination_count = 0
        truncation_count = 0
        action_abs_sum = 0.0
        action_abs_max = 0.0
        finite = True
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
        passed = finite
        failures: list[str] = []
        if not finite:
            failures.append("non-finite rollout values")
        if args.min_mean_return is not None and score < args.min_mean_return:
            passed = False
            failures.append(
                f"mean return {score:.6g} is below {args.min_mean_return:.6g}"
            )
        if len(completed_returns) < args.min_episodes:
            passed = False
            failures.append(
                f"completed episodes {len(completed_returns)} is below {args.min_episodes}"
            )
        return {
            "command": "eval",
            "source": str(source),
            "source_type": "policy" if args.policy is not None else "checkpoint",
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
            "terminated": termination_count,
            "truncated": truncation_count,
            "termination_rate": termination_count / transitions,
            "reward_per_transition": float(rollout_returns.sum() / transitions),
            "action_abs_mean": action_abs_sum / (transitions * action_values.shape[-1]),
            "action_abs_max": action_abs_max,
            "finite": finite,
            "passed": passed,
            "failures": failures,
        }
    finally:
        env.close()


def render(args: argparse.Namespace) -> dict[str, object]:
    source = args.policy or args.checkpoint
    source = source.expanduser()
    if not source.is_file():
        kind = "policy" if args.policy is not None else "checkpoint"
        raise FileNotFoundError(f"{kind} does not exist: {source}")

    def run_renderer(policy: Path) -> None:
        command = [
            sys.executable,
            "-m",
            "microduck_local.render_rollout",
            "--policy",
            str(policy),
            "--behavior",
            BEHAVIOR_ID,
            "--out",
            str(args.output),
            "--episodes",
            str(args.episodes),
            "--seconds",
            str(args.max_episode_s),
            "--seed",
            str(args.seed),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--sheet-frames",
            str(args.sheet_frames),
            "--camera",
            args.camera,
            "--env",
            f"MICRODUCK_CLIPS_DIR={CLIP_DIRECTORY}",
            "--env",
            f"MICRODUCK_CLIP={CLIP_NAME}",
        ]
        subprocess.run(command, check=True)

    if args.policy is not None:
        run_renderer(source)
    else:
        from rlx.export.microduck_onnx import export_deterministic_actor

        # Keep the temporary policy alive until the rendering subprocess exits.
        with tempfile.TemporaryDirectory(prefix="rlx-microduck-render-") as directory:
            policy = export_deterministic_actor(
                source, Path(directory) / f"{source.stem}.onnx"
            )
            run_renderer(policy)
    return {
        "command": "render",
        "source": str(source),
        "source_type": "policy" if args.policy is not None else "checkpoint",
        "output": str(args.output),
    }


def _make_view_environment(seed: int, max_episode_s: float):
    """Build one deterministic dance environment without vector workers."""
    behaviors = importlib.import_module("microduck_local.behaviors")
    variable = "MICRODUCK_CLIPS_DIR"
    previous = os.environ.get(variable)
    try:
        os.environ[variable] = str(CLIP_DIRECTORY)
        env = behaviors.BehaviorEnv(
            behavior_id=BEHAVIOR_ID,
            clip_name=CLIP_NAME,
            seed=seed,
            max_episode_s=max_episode_s,
            domain_rand=False,
            obs_noise=False,
            action_delay=False,
            random_yaw=False,
        )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous
    return env


def _run_view_loop(
    policy: Path,
    env,
    *,
    seed: int,
    viewer_module,
    ort_module,
) -> None:
    """Run an ONNX policy in a native passive viewer until it is closed."""
    import numpy as np

    contract = importlib.import_module("microduck_local.contract")
    session = ort_module.InferenceSession(
        str(policy), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    observation, _ = env.reset(seed=seed)
    next_step = time.monotonic()

    with viewer_module.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            batch = np.asarray(observation, dtype=np.float32)[None, :]
            action = np.asarray(session.run(None, {input_name: batch})[0])
            expected = (1, *env.action_space.shape)
            if action.shape != expected:
                raise ValueError(
                    f"ONNX policy action batch must have shape {expected}, "
                    f"got {action.shape}"
                )
            if not np.isfinite(action).all():
                raise RuntimeError(
                    "ONNX policy produced non-finite actions; retrain or select "
                    "a valid checkpoint before opening the viewer"
                )
            observation, _, terminated, truncated, _ = env.step(action[0])
            if terminated or truncated:
                observation, _ = env.reset(seed=seed)
            viewer.sync()

            next_step += contract.CTRL_DT
            delay = next_step - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step = time.monotonic()


def view(args: argparse.Namespace) -> dict[str, object]:
    """Run a checkpoint or ONNX policy in the native MuJoCo viewer."""
    source = (args.policy or args.checkpoint).expanduser()
    if not source.is_file():
        kind = "policy" if args.policy is not None else "checkpoint"
        raise FileNotFoundError(f"{kind} does not exist: {source}")

    env = _make_view_environment(args.seed, args.max_episode_s)
    try:
        viewer_module = importlib.import_module("mujoco.viewer")
        ort_module = importlib.import_module("onnxruntime")
        if args.policy is not None:
            try:
                _run_view_loop(
                    source,
                    env,
                    seed=args.seed,
                    viewer_module=viewer_module,
                    ort_module=ort_module,
                )
            except KeyboardInterrupt:
                pass
        else:
            from rlx.export.microduck_onnx import export_deterministic_actor

            with tempfile.TemporaryDirectory(prefix="rlx-microduck-view-") as directory:
                policy = export_deterministic_actor(
                    source, Path(directory) / f"{source.stem}.onnx"
                )
                try:
                    _run_view_loop(
                        policy,
                        env,
                        seed=args.seed,
                        viewer_module=viewer_module,
                        ort_module=ort_module,
                    )
                except KeyboardInterrupt:
                    pass
    finally:
        env.close()
    return {
        "command": "view",
        "source": str(source),
        "source_type": "policy" if args.policy is not None else "checkpoint",
    }


def _parse_child_payload(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    completed.check_returncode()
    return json.loads(completed.stdout.strip().splitlines()[-1])


def demo(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "smoke.safetensors"
    policy = output / "smoke.onnx"
    common = [
        "--num-envs",
        str(args.num_envs),
        "--backend",
        args.backend,
        "--seed",
        str(args.seed),
        "--max-episode-s",
        str(args.max_episode_s),
        "--no-domain-rand",
        "--no-obs-noise",
        "--no-action-delay",
        "--no-random-yaw",
    ]
    train_result = _parse_child_payload(
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "train",
                "--checkpoint",
                str(checkpoint),
                "--onnx-output",
                str(policy),
                "--num-steps",
                "2",
                "--num-minibatches",
                "1",
                "--update-epochs",
                "1",
                "--total-timesteps",
                str(args.num_envs * 2),
                *common,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    )
    eval_result = _parse_child_payload(
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "eval",
                "--policy",
                str(policy),
                "--eval-steps",
                str(args.eval_steps),
                *common,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    )
    stages: list[dict[str, Any]] = [
        train_result,
        {"command": "export", "output": str(policy), "automatic": True},
        eval_result,
    ]
    if args.render:
        render_command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "render",
            "--policy",
            str(policy),
            "--output",
            str(output / "render"),
            "--episodes",
            "1",
            "--max-episode-s",
            str(args.max_episode_s),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--sheet-frames",
            str(args.sheet_frames),
            "--camera",
            args.camera,
        ]
        stages.append(
            _parse_child_payload(
                subprocess.run(
                    render_command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            )
        )
    return {
        "command": "demo",
        "label": "smoke",
        "output": str(output),
        "passed": bool(eval_result["passed"]),
        "stages": stages,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-envs", type=_positive_int, default=16)
    parser.add_argument("--backend", default="fork")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-episode-s", type=_positive_float, default=4.0)
    parser.add_argument("--weight-overrides", default="{}")
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


def _policy_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--policy", type=Path, help="ONNX policy using raw observations"
    )


def _render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=_positive_int, default=1280)
    parser.add_argument("--height", type=_positive_int, default=720)
    parser.add_argument("--fps", type=_positive_int, default=30)
    parser.add_argument("--sheet-frames", type=_positive_int, default=12)
    parser.add_argument(
        "--camera", choices=("side", "front", "three-quarter"), default="side"
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "train":
        batch_size = args.num_envs * args.num_steps
        if args.num_minibatches > batch_size:
            parser.error("--num-minibatches cannot exceed num-envs * num-steps")
        if batch_size % args.num_minibatches:
            parser.error("num-envs * num-steps must be divisible by --num-minibatches")
        if args.total_timesteps < batch_size:
            parser.error("--total-timesteps must cover at least one rollout batch")
        if args.onnx_output is not None and not args.export_onnx:
            parser.error("--onnx-output cannot be combined with --no-export-onnx")
    if hasattr(args, "weight_overrides"):
        try:
            _weight_overrides(args.weight_overrides)
        except ValueError as exc:
            parser.error(str(exc))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train")
    _common(train_parser)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument(
        "--total-timesteps", type=_positive_int, default=1_000_000
    )
    train_parser.add_argument("--num-steps", type=_positive_int, default=24)
    train_parser.add_argument("--num-minibatches", type=_positive_int, default=4)
    train_parser.add_argument("--update-epochs", type=_positive_int, default=5)
    train_parser.add_argument("--gamma", type=_probability, default=0.99)
    train_parser.add_argument("--gae-lambda", type=_probability, default=0.95)
    train_parser.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train_parser.add_argument("--clip-coefficient", type=_positive_float, default=0.2)
    train_parser.add_argument(
        "--clip-value-loss", action=argparse.BooleanOptionalAction, default=True
    )
    train_parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    train_parser.add_argument("--value-coefficient", type=_positive_float, default=1.0)
    train_parser.add_argument("--max-grad-norm", type=_positive_float, default=0.5)
    train_parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    train_parser.add_argument(
        "--export-onnx", action=argparse.BooleanOptionalAction, default=True
    )
    train_parser.add_argument("--onnx-output", type=Path)

    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)

    eval_parser = commands.add_parser("eval")
    _common(eval_parser)
    _policy_source(eval_parser)
    eval_parser.add_argument("--eval-steps", type=_positive_int, default=500)
    eval_parser.add_argument("--min-mean-return", type=float)
    eval_parser.add_argument("--min-episodes", type=int, default=0)

    render_parser = commands.add_parser("render")
    _common(render_parser)
    _policy_source(render_parser)
    render_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for rendered videos and contact sheets",
    )
    render_parser.add_argument("--episodes", type=_positive_int, default=1)
    _render_options(render_parser)

    view_parser = commands.add_parser("view")
    _policy_source(view_parser)
    view_parser.add_argument("--seed", type=int, default=1)
    view_parser.add_argument(
        "--max-episode-s", type=_positive_float, default=4.0
    )

    demo_parser = commands.add_parser("demo")
    demo_parser.add_argument("--output", type=Path, required=True)
    demo_parser.add_argument("--num-envs", type=_positive_int, default=2)
    demo_parser.add_argument("--backend", default="fork")
    demo_parser.add_argument("--seed", type=int, default=1)
    demo_parser.add_argument("--max-episode-s", type=_positive_float, default=0.2)
    demo_parser.add_argument("--eval-steps", type=_positive_int, default=4)
    demo_parser.add_argument(
        "--render", action=argparse.BooleanOptionalAction, default=False
    )
    _render_options(demo_parser)

    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = {
        "train": train,
        "export": export,
        "eval": evaluate,
        "render": render,
        "view": view,
        "demo": demo,
    }[args.command](args)
    print(json.dumps(result, sort_keys=True))
    if args.command in {"eval", "demo"} and not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
