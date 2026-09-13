"""Studio-compatible adapter for standalone recurrent basketball PPO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence


RLX_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RLX_ROOT.parent
DEFAULT_SOURCE_CHECKPOINT = (
    WORKSPACE_ROOT / "microduck-playground/artifacts/basketball/checkpoint.pt"
)
DEFAULT_OUTPUT = RLX_ROOT / "runs/studio/basketball/basketball-studio"
TRAIN_EPISODE_SECONDS = 10.0
ROLLING_COMMAND = (0.15, 0.0, 0.0)
ZERO_COMMAND = (0.0, 0.0, 0.0)
PARITY_LIMIT = 3e-5


def _load_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _learner() -> Any:
    return _load_module(
        "_rlx_basketball_learner",
        Path(__file__).with_name("ppo_microduck_basketball.py"),
    )


def _evaluator() -> Any:
    return _load_module(
        "_rlx_basketball_evaluator",
        RLX_ROOT / "scripts/eval_basketball_local.py",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be finite")
    return parsed


def _empty_weights(value: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"must be valid JSON: {exc.msg}"
        ) from exc
    if decoded != {}:
        raise argparse.ArgumentTypeError(
            "basketball uses its sealed standalone reward and requires {}"
        )
    return {}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recipe", choices=("basketball",), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-envs", type=_positive_int, default=4)
    parser.add_argument("--backend", default="standalone")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actuator", choices=("bam", "xml"), default="bam")
    parser.add_argument("--max-episode-s", type=_positive_float)
    parser.add_argument("--weight-overrides", type=_empty_weights, default={})
    parser.add_argument(
        "--domain-rand", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--obs-noise", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--action-delay", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--random-yaw", action=argparse.BooleanOptionalAction, default=True
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    _common(train)
    train.add_argument("--checkpoint", type=Path)
    train.add_argument("--init-from", type=Path)
    train.add_argument("--onnx-output", type=Path)
    train.add_argument("--total-timesteps", type=_positive_int, default=640)
    train.add_argument("--num-steps", type=_positive_int, default=32)
    train.add_argument("--num-minibatches", type=_positive_int, default=1)
    train.add_argument("--checkpoint-interval", type=_nonnegative_int, default=0)
    train.add_argument("--initial-std", type=_positive_float, default=0.03)
    train.add_argument(
        "--normalize-rewards",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    train.add_argument(
        "--freeze-observation-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument("--update-epochs", type=_positive_int, default=5)
    train.add_argument("--learning-rate", type=_positive_float, default=2e-5)
    train.add_argument("--gamma", type=_finite_float, default=0.99)
    train.add_argument("--gae-lambda", type=_finite_float, default=0.95)
    train.add_argument("--clip-coefficient", type=_positive_float, default=0.2)
    train.add_argument("--entropy-coefficient", type=_finite_float, default=0.01)
    train.add_argument("--value-coefficient", type=_positive_float, default=1.0)
    train.add_argument("--max-grad-norm", type=_positive_float, default=1.0)
    train.add_argument("--target-kl", type=_positive_float, default=0.02)
    train.add_argument("--save-interval", type=_positive_int)
    train.add_argument("--hold", type=_finite_float, default=0.0)
    train.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    train.add_argument(
        "--normalize-advantages",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--clip-value-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    evaluate = commands.add_parser("eval")
    _common(evaluate)
    source = evaluate.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--policy", type=Path)
    evaluate.add_argument("--eval-steps", type=_positive_int)
    evaluate.add_argument(
        "--evaluation-mode",
        choices=("pipeline", "skill"),
        default="skill",
    )
    evaluate.add_argument("--min-mean-return", type=_finite_float)
    evaluate.add_argument("--min-episodes", type=_nonnegative_int, default=0)
    evaluate.add_argument("--pushes", action="store_true")

    render = commands.add_parser("render")
    _common(render)
    source = render.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--policy", type=Path)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--episodes", type=_positive_int, default=1)
    render.add_argument("--width", type=_positive_int, default=640)
    render.add_argument("--height", type=_positive_int, default=360)
    render.add_argument("--fps", type=_positive_int, default=25)
    render.add_argument("--sheet-frames", type=_positive_int, default=12)
    render.add_argument("--render-seconds", type=_positive_float)
    render.add_argument(
        "--camera",
        choices=("side", "front", "three-quarter"),
        default="three-quarter",
    )
    render.add_argument("--distance", type=_positive_float)
    render.add_argument("--pushes", action="store_true")

    import_parser = commands.add_parser("import")
    import_parser.add_argument("--recipe", choices=("basketball",), required=True)
    import_parser.add_argument("--source-dir", type=Path, required=True)
    import_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "import":
        return
    if args.actuator != "bam":
        parser.error("basketball Studio artifacts require --actuator bam")
    for name in ("domain_rand", "obs_noise"):
        if getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is unsupported for basketball")
    if args.command == "train":
        if not 0 <= args.hold <= 1:
            parser.error("--hold must be between zero and one")
        if args.curriculum:
            from rlx.environments.basketball import HOLD_LEVELS

            if args.hold not in HOLD_LEVELS:
                parser.error(f"--curriculum requires --hold in {HOLD_LEVELS}")
        if args.checkpoint is not None and args.checkpoint.suffix != ".pt":
            parser.error("basketball recurrent checkpoints must use a .pt path")
        fixed = {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coefficient": 0.2,
            "update_epochs": 5,
            "entropy_coefficient": 0.01,
            "value_coefficient": 1.0,
            "max_grad_norm": 1.0,
        }
        for name, expected in fixed.items():
            if not math.isclose(getattr(args, name), expected):
                parser.error(
                    f"--{name.replace('_', '-')} is fixed at {expected:g} "
                    "by the standalone learner"
                )
        if args.num_minibatches != 1:
            parser.error("standalone recurrent PPO uses one full-sequence minibatch")
        if args.normalize_rewards:
            parser.error("standalone basketball PPO does not normalize rewards")
        if not args.freeze_observation_normalization:
            parser.error("basketball observation normalization must remain frozen")
        if not args.normalize_advantages:
            parser.error("standalone basketball PPO always normalizes advantages")
        if args.clip_value_loss:
            parser.error("standalone basketball PPO does not implement clipped value loss")
    if args.command == "eval":
        if args.min_mean_return is not None:
            parser.error("basketball evaluation is physical-metric based; --min-mean-return is unsupported")
        if args.min_episodes:
            parser.error("basketball evaluation uses fixed first-fall trials; --min-episodes is unsupported")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _output_directory(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        return checkpoint.expanduser().resolve().parent
    return DEFAULT_OUTPUT


def _adjustments(args: argparse.Namespace) -> list[dict[str, Any]]:
    adjustments: list[dict[str, Any]] = []
    if args.backend != "standalone":
        adjustments.append(
            {
                "option": "backend",
                "requested": args.backend,
                "applied": "standalone",
                "reason": "basketball uses its single-process MuJoCo evaluator",
            }
        )
    if args.command == "train" and args.max_episode_s is not None and not math.isclose(
        args.max_episode_s, TRAIN_EPISODE_SECONDS
    ):
        adjustments.append(
            {
                "option": "max_episode_s",
                "requested": args.max_episode_s,
                "applied": TRAIN_EPISODE_SECONDS,
                "reason": "the standalone recurrent learner has a sealed 10-second horizon",
            }
        )
    if not args.random_yaw:
        adjustments.append(
            {
                "option": "random_yaw",
                "requested": False,
                "applied": True,
                "reason": "the standalone basketball protocol evaluates randomized headings",
            }
        )
    if not args.action_delay:
        adjustments.append(
            {
                "option": "action_delay",
                "requested": False,
                "applied": True,
                "reason": "BasketballEnv and the source policy use inherited action delay",
            }
        )
    return adjustments


def _metadata(
    args: argparse.Namespace,
    *,
    steps: int,
    source: dict[str, Any] | None,
    adjustments: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "recipe": "basketball",
        "steps": int(steps),
        "seed": int(args.seed),
        "num_envs": int(args.num_envs),
        "max_episode_s": TRAIN_EPISODE_SECONDS,
        "backend": "standalone",
        "actuator": "bam",
        "reward_weights": {},
        "recipe_options": {},
        "randomization": {
            "domain_rand": False,
            "obs_noise": False,
            "action_delay": True,
            "random_yaw": True,
        },
        "recurrent_state": {
            "kind": "lstm",
            "reset": "zeroed on episode reset",
            "carried": True,
        },
        "normalization_frozen": True,
        "training_assistance": {"hold": args.hold, "curriculum": args.curriculum},
        "adapter_adjustments": adjustments,
        "source": source,
    }


def _append_onnx_metadata(
    onnx_path: Path,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    parity: dict[str, Any],
) -> None:
    errors = [float(parity.get("max_abs_error", math.inf)), float(parity.get("max_state_abs_error", 0))]
    if any(not math.isfinite(error) or error > PARITY_LIMIT for error in errors):
        raise RuntimeError(
            f"refusing to annotate ONNX before parity passes: {errors}"
        )
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            "rlx_metadata": json.dumps(metadata, sort_keys=True),
            "rlx_checkpoint_sha256": sha256_file(checkpoint_path),
            "controller_scope": "recurrent basketball actor with carried LSTM state",
        }
    )
    helper.set_model_props(model, properties)
    onnx.checker.check_model(model)
    temporary = onnx_path.with_suffix(onnx_path.suffix + ".metadata.tmp")
    onnx.save(model, temporary)
    temporary.replace(onnx_path)


def _copy_artifact(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination


def _verify_local_parity(checkpoint: Path, policy: Path) -> dict[str, Any]:
    import torch

    from rlx.models.basketball import BasketballActor, compare_actor_to_onnx

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError("import checkpoint is not a local recurrent basketball checkpoint")
    actor = BasketballActor()
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    parity = compare_actor_to_onnx(actor, policy)
    errors = [float(parity["max_abs_error"]), float(parity.get("max_state_abs_error", 0))]
    if any(not math.isfinite(error) or error > PARITY_LIMIT for error in errors):
        raise RuntimeError(f"import candidate ONNX parity failed: {errors}")
    return parity


def import_run(source_dir: Path | str, output_dir: Path | str) -> dict[str, Any]:
    source = Path(source_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    source_checkpoint = source / "checkpoint.pt"
    source_policy = source / "policy.onnx"
    source_summary = source / "summary.json"
    source_paths = (source_checkpoint, source_policy, source_summary)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"basketball import source is missing: {path}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to overwrite an existing Studio run: {output}")

    source_hashes = {str(path): sha256_file(path) for path in source_paths}
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("basketball import summary must be a JSON object")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = _copy_artifact(source_checkpoint, output / "checkpoint.pt")
    policy = _copy_artifact(source_policy, output / "policy.onnx")
    parity = _verify_local_parity(checkpoint, policy)
    settings = summary.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("basketball import summary settings must be an object")
    metadata = {
        "recipe": "basketball",
        "steps": int(summary.get("local_steps", 0)),
        "seed": int(settings.get("seed", 42)),
        "num_envs": int(settings.get("num_envs", 1)),
        "max_episode_s": TRAIN_EPISODE_SECONDS,
        "backend": "standalone",
        "actuator": "bam",
        "reward_weights": {},
        "recipe_options": {},
        "randomization": {
            "domain_rand": False,
            "obs_noise": False,
            "action_delay": True,
            "random_yaw": True,
        },
        "recurrent_state": {
            "kind": "lstm",
            "reset": "zeroed on episode reset",
            "carried": True,
        },
        "normalization_frozen": True,
        "imported_candidate": {
            "source_dir": str(source),
            "source_files_sha256": source_hashes,
            "continuation_kind": summary.get("continuation_kind"),
            "full_upstream_resume": bool(summary.get("full_upstream_resume", False)),
            "source_checkpoint_sha256": summary.get("source_checkpoint_sha256"),
            "source_onnx_sha256": summary.get("source_onnx_sha256"),
        },
    }
    _append_onnx_metadata(policy, checkpoint, metadata, parity)
    annotated_parity = _verify_local_parity(checkpoint, policy)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    _write_json(sidecar, {"metadata": metadata})
    if {str(path): sha256_file(path) for path in source_paths} != source_hashes:
        raise RuntimeError("basketball import source changed while copying")
    result = {
        "command": "import",
        "recipe": "basketball",
        "source": str(source),
        "output": str(output),
        "checkpoint": str(checkpoint),
        "metadata": str(sidecar),
        "onnx": str(policy),
        "steps": int(summary.get("local_steps", 0)),
        "source_files_sha256": source_hashes,
        "artifact_hashes": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "policy_sha256": sha256_file(policy),
            "metadata_sha256": sha256_file(sidecar),
        },
        "onnx_parity_before_metadata": parity,
        "onnx_parity_after_metadata": annotated_parity,
    }
    result_path = _write_json(output / "result.json", result)
    result["result_json"] = str(result_path)
    _write_json(result_path, result)
    return result


def train(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_directory(args)
    target_checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else output / "checkpoint.pt"
    )
    target_onnx = (
        args.onnx_output.expanduser().resolve()
        if args.onnx_output is not None
        else output / "policy.onnx"
    )
    source_checkpoint = (
        args.init_from.expanduser().resolve()
        if args.init_from is not None
        else DEFAULT_SOURCE_CHECKPOINT
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"basketball source checkpoint does not exist: {source_checkpoint}")
    source_policy = source_checkpoint.with_name("policy.onnx")
    source_hashes = {
        str(path): sha256_file(path) for path in (source_checkpoint, source_policy)
    }
    if source_checkpoint.parent != output and (
        target_checkpoint in (source_checkpoint, source_policy)
        or target_onnx in (source_checkpoint, source_policy)
    ):
        raise ValueError("training output cannot overwrite external source artifacts")
    output.mkdir(parents=True, exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix="training-", dir=output))
    staged_source = _copy_artifact(source_checkpoint, invocation / "source" / "checkpoint.pt")
    staged_policy = _copy_artifact(source_policy, invocation / "source" / "policy.onnx")
    if (
        sha256_file(staged_source) != source_hashes[str(source_checkpoint)]
        or sha256_file(staged_policy) != source_hashes[str(source_policy)]
    ):
        raise RuntimeError("basketball source changed while staging training")
    batch_steps = args.num_envs * args.num_steps
    updates = math.ceil(args.total_timesteps / batch_steps)
    applied_steps = updates * batch_steps
    adjustments = _adjustments(args)
    if applied_steps != args.total_timesteps:
        adjustments.append(
            {
                "option": "total_timesteps",
                "requested": args.total_timesteps,
                "applied": applied_steps,
                "reason": "rounded up to a complete recurrent rollout batch",
            }
        )
    save_interval = args.save_interval
    if save_interval is None:
        save_interval = (
            updates + 1
            if args.checkpoint_interval == 0
            else max(1, math.ceil(args.checkpoint_interval / batch_steps))
        )
    learner_args = argparse.Namespace(
        checkpoint=staged_source,
        output=invocation / "learner",
        num_envs=args.num_envs,
        updates=updates,
        steps=args.num_steps,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        save_interval=save_interval,
        initial_std=args.initial_std,
        hold=args.hold,
        curriculum=args.curriculum,
        actuator="bam",
        seed=args.seed,
        command=ROLLING_COMMAND,
        pushes=False,
    )
    summary = _learner().train(learner_args)
    generated_checkpoint = Path(summary["checkpoint"]).resolve()
    generated_onnx = Path(summary["onnx"]).resolve()
    if {str(path): sha256_file(path) for path in (source_checkpoint, source_policy)} != source_hashes:
        raise RuntimeError("basketball source changed during training")
    total_steps = int(summary.get("total_steps", summary["local_steps"]))
    source = {
        "checkpoint": str(source_checkpoint),
        "checkpoint_sha256": source_hashes[str(source_checkpoint)],
        "source_files_sha256": source_hashes,
        "archived_checkpoint": str(staged_source),
        "archived_policy": str(staged_policy),
        "continuation_kind": summary.get("continuation_kind"),
        "full_upstream_resume": bool(summary.get("full_upstream_resume", False)),
    }
    metadata = _metadata(args, steps=total_steps, source=source, adjustments=adjustments)
    published_policy = _copy_artifact(generated_onnx, invocation / "studio-policy.onnx")
    _append_onnx_metadata(published_policy, generated_checkpoint, metadata, summary["onnx_parity"])
    checkpoint = _copy_artifact(generated_checkpoint, target_checkpoint)
    policy = _copy_artifact(published_policy, target_onnx)
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".json")
    _write_json(sidecar, {"metadata": metadata})
    metrics_path = output / "training-metrics.jsonl"
    metrics_path.write_text(
        "".join(
            json.dumps(
                {
                    "phase": "update",
                    "env_steps": index * batch_steps,
                    **row,
                },
                allow_nan=False,
            )
            + "\n"
            for index, row in enumerate(summary.get("updates", ()), start=1)
        ),
        encoding="utf-8",
    )
    result = {
        "command": "train",
        "recipe": "basketball",
        "checkpoint": str(checkpoint),
        "metadata": str(sidecar),
        "onnx": str(policy),
        "steps": total_steps,
        "trained_steps": int(summary["local_steps"]),
        "requested_steps": int(args.total_timesteps),
        "training_metrics": str(metrics_path),
        "source_sha256": source_hashes[str(source_checkpoint)],
        "source_files_sha256": source_hashes,
        "resumed_from": str(source_checkpoint) if args.init_from else None,
        "continuation_kind": summary.get("continuation_kind"),
        "full_upstream_resume": False,
        "training_directory": str(invocation),
        "training_assistance": metadata["training_assistance"],
        "artifact_hashes": {
            "checkpoint_sha256": sha256_file(checkpoint),
            "policy_sha256": sha256_file(policy),
            "metadata_sha256": sha256_file(sidecar),
        },
        "onnx_parity": summary["onnx_parity"],
        "adapter_adjustments": adjustments,
        "learner_summary": summary,
    }
    result_path = _write_json(output / "result.json", result)
    result["result_json"] = str(result_path)
    _write_json(result_path, result)
    return result


def _policy_source(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.policy is not None:
        policy = args.policy.expanduser().resolve()
        checkpoint = None
    else:
        checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint is not None
            else _output_directory(args) / "checkpoint.pt"
        )
        candidates = (
            checkpoint.with_suffix(".onnx"),
            checkpoint.parent / "policy.onnx",
            checkpoint.parent / "basketball.onnx",
        )
        policy = next((candidate for candidate in candidates if candidate.is_file()), candidates[1])
    if not policy.is_file():
        raise FileNotFoundError(f"basketball recurrent ONNX policy does not exist: {policy}")
    if checkpoint is None:
        candidate = policy.parent / "checkpoint.pt"
        checkpoint = candidate if candidate.is_file() else None
    return policy, checkpoint


def _source_hashes(policy: Path, checkpoint: Path | None) -> dict[str, str]:
    paths = [policy]
    if checkpoint is not None and checkpoint.is_file():
        paths.append(checkpoint)
    return {str(path): sha256_file(path) for path in paths}


def _trial_failures(trials: Sequence[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for trial in trials:
        label = f"seed {trial['seed']} command {trial['command']}"
        failures.extend(f"{label}: {failure}" for failure in trial.get("failures", ()))
    return failures


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    evaluator = _evaluator()
    policy_path, checkpoint_path = _policy_source(args)
    before_hashes = _source_hashes(policy_path, checkpoint_path)
    output = _output_directory(args)
    output.mkdir(parents=True, exist_ok=True)
    seconds = (
        args.eval_steps * evaluator.CTRL_DT
        if args.eval_steps is not None
        else args.max_episode_s or evaluator.DEFAULT_SECONDS
    )
    evaluator.criteria_for(seconds, ROLLING_COMMAND)
    policy = evaluator.RecurrentOnnxPolicy(policy_path)
    commands = (ZERO_COMMAND, ROLLING_COMMAND)
    seeds = tuple(args.seed + index for index in range(args.num_envs))
    trials = [
        evaluator.run_trial(
            policy=policy,
            seed=seed,
            command=command,
            seconds=seconds,
            actuator="bam",
            pushes=args.pushes,
            render_path=None,
            width=640,
            height=360,
        )
        for command in commands
        for seed in seeds
    ]
    summary = evaluator.summarize_trials(trials)
    after_hashes = _source_hashes(policy_path, checkpoint_path)
    pipeline_failures = []
    if before_hashes != after_hashes:
        pipeline_failures.append("policy source changed during evaluation")
    if any(trial["measured_steps"] < 1 for trial in trials):
        pipeline_failures.append("one or more trials produced no control steps")
    pipeline_passed = not pipeline_failures
    rolling_cases = [
        case for case in summary["cases"] if case["classification"] == "commanded_rolling"
    ]
    forward_rolling_passed = bool(rolling_cases) and all(case["passed"] for case in rolling_cases)
    rolling_passed = False
    success = False
    skill_status = (
        ("passed" if success else "failed")
        if args.evaluation_mode == "skill"
        else "not_assessed"
    )
    failures = [*pipeline_failures, *_trial_failures(trials)]
    failures.append("full steering is not assessed: lateral and yaw command trials are not implemented")
    if not rolling_passed:
        failures.append("commanded rolling criteria did not pass for every evaluated seed")
    task_assessment = {
        "recipe": "basketball",
        "success": success,
        "passed": success,
        "rolling_passed": rolling_passed,
        "forward_rolling_passed": forward_rolling_passed,
        "steering_coverage": {"forward": True, "lateral": False, "yaw": False},
        "hold_zero": all(trial["hold_zero"] for trial in trials),
        "summary": summary,
        "failures": list(dict.fromkeys(failures)),
    }
    report = {
        "schema_version": 1,
        "evaluator": str(Path(evaluator.__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(evaluator.__file__).resolve()),
        "protocol": {
            "control_dt_s": evaluator.CTRL_DT,
            "seconds": seconds,
            "seeds": list(seeds),
            "commands": [list(command) for command in commands],
            "actuator": "bam",
            "pushes": bool(args.pushes),
            "hold": 0,
            "curriculum": False,
            "obs_noise": False,
            "domain_rand": False,
            "action_delay": True,
            "random_yaw": True,
            "first_fall": True,
            "automatic_resets": False,
        },
    }
    result = {
        "command": "eval",
        "recipe": "basketball",
        "evaluation_mode": args.evaluation_mode,
        "source": str(policy_path),
        "source_type": "policy",
        "source_sha256": before_hashes[str(policy_path)],
        "source_files_sha256": before_hashes,
        "steps": round(seconds / evaluator.CTRL_DT),
        "num_envs": args.num_envs,
        "transitions": len(trials) * round(seconds / evaluator.CTRL_DT),
        "finite": pipeline_passed,
        "pipeline_passed": pipeline_passed,
        "skill_status": skill_status,
        "passed": success,
        "success": success,
        "failures": task_assessment["failures"],
        "task_assessment": task_assessment,
        "report": report,
        "basketball_assessment": summary,
        "trials": trials,
        "evaluation": {
            "mode": args.evaluation_mode,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "steps_per_env": round(seconds / evaluator.CTRL_DT),
            "environment": {
                "recipe": "basketball",
                "backend": "standalone",
                "actuator": "bam",
                "max_episode_s": seconds,
                "domain_rand": False,
                "obs_noise": False,
                "action_delay": True,
                "random_yaw": True,
                "hold": 0,
                "recipe_options": {},
                "reward_weights": {},
            },
        },
        "adapter_adjustments": _adjustments(args),
    }
    evaluation_path = _write_json(output / "evaluation.json", result)
    result["evaluation_path"] = str(evaluation_path)
    result_path = _write_json(output / "result.json", result)
    result["result_json"] = str(result_path)
    _write_json(result_path, result)
    return result


def render(args: argparse.Namespace) -> dict[str, Any]:
    if args.episodes != 1:
        raise ValueError("basketball Studio rendering currently requires --episodes 1")
    evaluator = _evaluator()
    policy_path, checkpoint_path = _policy_source(args)
    source_hashes = _source_hashes(policy_path, checkpoint_path)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seconds = args.render_seconds or args.max_episode_s or evaluator.DEFAULT_SECONDS
    evaluator.criteria_for(seconds, ROLLING_COMMAND)
    policy = evaluator.RecurrentOnnxPolicy(policy_path)
    trial = evaluator.run_trial(
        policy=policy,
        seed=args.seed,
        command=ROLLING_COMMAND,
        seconds=seconds,
        actuator="bam",
        pushes=args.pushes,
        render_path=output,
        width=args.width,
        height=args.height,
    )
    if _source_hashes(policy_path, checkpoint_path) != source_hashes:
        raise RuntimeError("policy source changed during rendering")
    render_receipt = trial.get("render")
    if not isinstance(render_receipt, dict):
        raise RuntimeError("basketball evaluator did not produce render artifacts")
    adjustments = _adjustments(args)
    if args.fps != 25:
        adjustments.append(
            {
                "option": "fps",
                "requested": args.fps,
                "applied": 25,
                "reason": "the first-fall renderer records every second 50 Hz control frame",
            }
        )
    if args.sheet_frames != 12:
        adjustments.append(
            {
                "option": "sheet_frames",
                "requested": args.sheet_frames,
                "applied": "5-second intervals",
                "reason": "the physical evaluator uses time-based evidence sampling",
            }
        )
    if args.camera != "three-quarter":
        adjustments.append(
            {
                "option": "camera",
                "requested": args.camera,
                "applied": "three-quarter",
                "reason": "the basketball evidence renderer uses its tracking camera",
            }
        )
    if args.distance is not None:
        adjustments.append(
            {
                "option": "distance",
                "requested": args.distance,
                "applied": "automatic",
                "reason": "the basketball tracking camera owns its framing distance",
            }
        )
    result = {
        "command": "render",
        "recipe": "basketball",
        "source": str(policy_path),
        "source_type": "policy",
        "source_sha256": source_hashes[str(policy_path)],
        "source_files_sha256": source_hashes,
        "checkpoint_sha256": (
            source_hashes.get(str(checkpoint_path)) if checkpoint_path is not None else None
        ),
        "output": str(output),
        "render_seconds": seconds,
        "environment": {
            "actuator": "bam",
            "seed": args.seed,
            "domain_rand": False,
            "obs_noise": False,
            "action_delay": True,
            "random_yaw": True,
            "hold": 0,
            "recipe_options": {},
        },
        "episodes": [
            {
                "mp4": render_receipt["video"],
                "contact_sheet": render_receipt["contact_sheet"],
                "seed": args.seed,
                "control_steps": trial["measured_steps"],
                "reset_count": trial["automatic_resets"],
                "fps": 25,
            }
        ],
        "task_assessment": {
            "success": bool(trial["controlled_rolling"]),
            "passed": bool(trial["controlled_rolling"]),
            "rolling_passed": bool(trial["controlled_rolling"]),
            "hold_zero": bool(trial["hold_zero"]),
            "failures": trial["failures"],
        },
        "trial": trial,
        "artifact_hashes": {
            "rollout_mp4_sha256": sha256_file(Path(render_receipt["video"])),
            "contact_sheet_sha256": sha256_file(Path(render_receipt["contact_sheet"])),
        },
        "adapter_adjustments": adjustments,
    }
    render_path = _write_json(output / "render.json", result)
    result["render_json"] = str(render_path)
    result_path = _write_json(output / "result.json", result)
    result["result_json"] = str(result_path)
    _write_json(render_path, result)
    _write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = {
        "train": train,
        "eval": evaluate,
        "render": render,
        "import": lambda parsed: import_run(parsed.source_dir, parsed.output_dir),
    }[args.command](args)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    if args.command == "eval" and not result["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
