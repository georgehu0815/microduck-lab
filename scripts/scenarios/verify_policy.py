from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for source_root in (ROOT / "microduck_local" / "src", ROOT / "rlx"):
    sys.path.insert(0, str(source_root))

SCENARIOS = ("dance", "swing", "running", "stilts", "backflip")
PPO_FIELDS = (
    "mean_loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approximate_kl",
    "clip_fraction",
    "explained_variance",
)


class VerificationError(RuntimeError):
    pass


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _number(value: Any, label: str, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerificationError(f"{label} must be numeric")
    if not math.isfinite(value):
        raise VerificationError(f"{label} must be finite")
    if integer and not isinstance(value, int):
        raise VerificationError(f"{label} must be an integer")
    return value


def _finite_numbers(value: Any, label: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        _number(value, label)
    elif isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_numbers(item, f"{label}[{index}]")


def validate_metadata(
    sidecar: Any, scenario: str, trained_steps: int
) -> dict[str, Any]:
    if not isinstance(sidecar, dict):
        raise VerificationError("checkpoint sidecar must contain a JSON object")
    if _number(sidecar.get("observation_dim"), "observation_dim", integer=True) != 61:
        raise VerificationError("checkpoint observation_dim must be 61")
    if _number(sidecar.get("action_dim"), "action_dim", integer=True) != 14:
        raise VerificationError("checkpoint action_dim must be 14")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, dict):
        raise VerificationError("checkpoint sidecar metadata must be a JSON object")
    if metadata.get("recipe") != scenario:
        raise VerificationError(
            f"checkpoint recipe must be {scenario!r}, got {metadata.get('recipe')!r}"
        )
    steps = _number(metadata.get("steps"), "metadata.steps", integer=True)
    if steps < trained_steps:
        raise VerificationError(
            f"metadata.steps {steps} is below required {trained_steps}"
        )
    return {
        "observation_dim": 61,
        "action_dim": 14,
        "recipe": scenario,
        "metadata_steps": steps,
    }


def read_telemetry(path: Path) -> list[dict[str, Any]]:
    events = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise VerificationError(
                f"{path}: invalid JSON on line {line_number}: {error.msg}"
            ) from error
        if not isinstance(event, dict):
            raise VerificationError(f"{path}: line {line_number} must be an object")
        events.append(event)
    return events


def validate_telemetry(
    events: list[dict[str, Any]], trained_steps: int
) -> dict[str, Any]:
    if not events:
        raise VerificationError("training telemetry is empty")
    for index, event in enumerate(events, 1):
        _finite_numbers(event, f"telemetry line {index}")
    collections = [event for event in events if event.get("phase") == "collection"]
    updates = [event for event in events if event.get("phase") == "update"]
    if not collections:
        raise VerificationError("training telemetry has no collection events")
    if not updates:
        raise VerificationError("training telemetry has no update events")
    for index, event in enumerate(updates, 1):
        for field in PPO_FIELDS:
            _number(event.get(field), f"update {index}.{field}")
        optimizer_steps = _number(
            event.get("optimizer_steps"),
            f"update {index}.optimizer_steps",
            integer=True,
        )
        if optimizer_steps <= 0:
            raise VerificationError(
                f"update {index}.optimizer_steps must be greater than zero"
            )
    collected = sum(
        _number(event.get("steps"), "collection.steps", integer=True)
        for event in collections
    )
    final_steps = _number(
        events[-1].get("env_steps"), "final telemetry env_steps", integer=True
    )
    if collected < trained_steps:
        raise VerificationError(
            f"collected transitions {collected} are below required {trained_steps}"
        )
    if final_steps < trained_steps:
        raise VerificationError(
            f"final local env_steps {final_steps} are below required {trained_steps}"
        )
    return {
        "events": len(events),
        "collections": len(collections),
        "updates": len(updates),
        "optimizer_steps": sum(event["optimizer_steps"] for event in updates),
        "collected_transitions": collected,
        "final_local_env_steps": final_steps,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_contract(policy: Path) -> tuple[dict[str, Any], Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise VerificationError(f"ONNX dependencies unavailable: {error}") from error

    try:
        model = onnx.load(str(policy))
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(
            str(policy),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as error:
        raise VerificationError(f"ONNX validation failed: {error}") from error
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise VerificationError("ONNX policy must have exactly one input and one output")
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    if input_info.type != "tensor(float)" or input_info.shape != ["batch", 61]:
        raise VerificationError(
            f"ONNX input must be float [batch, 61], got {input_info.type} {input_info.shape}"
        )
    if output_info.type != "tensor(float)" or output_info.shape != ["batch", 14]:
        raise VerificationError(
            f"ONNX output must be float [batch, 14], got {output_info.type} {output_info.shape}"
        )
    rng = np.random.default_rng(20260909)
    batches = [np.zeros((1, 61), dtype=np.float32)]
    sampled = rng.standard_normal((8, 61), dtype=np.float32)
    sampled[0] = 0
    batches.append(sampled)
    checks = []
    for values in batches:
        output = session.run([output_info.name], {input_info.name: values})[0]
        if output.dtype != np.float32 or output.shape != (len(values), 14):
            raise VerificationError(
                f"ONNX runtime returned {output.dtype} {output.shape}, "
                f"expected float32 ({len(values)}, 14)"
            )
        if not bool(np.isfinite(output).all()):
            raise VerificationError(
                f"ONNX runtime returned non-finite values for batch {len(values)}"
            )
        checks.append({"batch": len(values), "finite": True})
    return {
        "checker_valid": True,
        "provider": "CPUExecutionProvider",
        "threads": 1,
        "input": {"type": input_info.type, "shape": input_info.shape},
        "output": {"type": output_info.type, "shape": output_info.shape},
        "runtime_checks": checks,
    }, np.concatenate(batches)


def verify(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    policy = args.policy.expanduser().resolve(strict=True)
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    telemetry_path = checkpoint.parent / "training-metrics.jsonl"
    if not sidecar_path.is_file():
        raise VerificationError(f"checkpoint sidecar not found: {sidecar_path}")
    if not telemetry_path.is_file():
        raise VerificationError(f"sibling telemetry not found: {telemetry_path}")
    sidecar = json.loads(sidecar_path.read_text())
    metadata = validate_metadata(sidecar, args.scenario, args.trained_steps)
    telemetry = validate_telemetry(
        read_telemetry(telemetry_path), args.trained_steps
    )
    onnx_contract, observations = _onnx_contract(policy)
    from rlx.export.microduck_onnx import parity_error

    try:
        parity = parity_error(checkpoint, policy, observations)
    except Exception as error:
        raise VerificationError(f"MLX/ONNX parity check failed: {error}") from error
    if not math.isfinite(parity):
        raise VerificationError("MLX/ONNX parity error is not finite")
    if parity > 1e-4:
        raise VerificationError(
            f"MLX/ONNX maximum absolute error {parity:.9g} exceeds 0.0001"
        )
    return {
        "passed": True,
        "scenario": args.scenario,
        "trained_steps": args.trained_steps,
        "artifact_scope": "checkpoint, telemetry, ONNX contract, runtime, and parity",
        "skill_evaluation_performed": False,
        "skill_pass": None,
        "paths": {
            "checkpoint": str(checkpoint),
            "sidecar": str(sidecar_path),
            "onnx": str(policy),
            "telemetry": str(telemetry_path),
        },
        "sha256": {
            "checkpoint": _sha256(checkpoint),
            "sidecar": _sha256(sidecar_path),
            "onnx": _sha256(policy),
            "telemetry": _sha256(telemetry_path),
        },
        "metadata": metadata,
        "telemetry": telemetry,
        "onnx": onnx_contract,
        "mlx_onnx_max_abs_error": parity,
        "checked_transitions": telemetry["final_local_env_steps"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--trained-steps", type=_positive_int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify(args)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, VerificationError) as error:
        print(f"policy artifact verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
