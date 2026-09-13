"""Replay tennis metrics evidence through real physics and add bounded videos."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np


from microduck_arm_experiments.tennis_return import (
    TennisReturnEnv,
    experiment_provenance,
)
from microduck_arm_v1c.whole_body_learning import source_provenance


VARIANTS = ("nominal", "small", "large")
CONTROLS = ("open_jaw", "hold")
PHYSICS_TOLERANCE = 1e-9
TIME_TOLERANCE_S = 1e-9
ACTION_TOLERANCE = 1e-7
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 25


class ReplayError(ValueError):
    """Evidence is incomplete, inconsistent, or not exactly replayable."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"Invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ReplayError(f"Expected JSON object: {path}")
    return value


def _read_telemetry(path: Path) -> list[dict]:
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"Invalid telemetry JSONL: {path}") from error
    if not records or not all(isinstance(record, dict) for record in records):
        raise ReplayError(f"Telemetry must contain JSON objects: {path}")
    return records


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _safe_file_hashes(directory: Path, result: dict) -> dict[str, str]:
    files = result.get("files")
    if not isinstance(files, dict) or "telemetry.jsonl" not in files:
        raise ReplayError(f"Missing bound telemetry hash: {directory / 'result.json'}")
    verified = {}
    for name, expected in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or Path(name).is_absolute()
            or not isinstance(expected, str)
        ):
            raise ReplayError(f"Unsafe or invalid result file binding: {directory}")
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ReplayError(f"Result file hash mismatch: {path}")
        verified[name] = expected
    return verified


def _verify_snapshots(source: Path, evaluation: dict) -> dict[str, str]:
    provenance = evaluation.get("experiment_hashes")
    if not isinstance(provenance, dict):
        raise ReplayError("Missing experiment_hashes")
    python_sources = provenance.get("python_sources")
    task_spec_hash = provenance.get("task_spec")
    if not isinstance(python_sources, dict) or not isinstance(task_spec_hash, str):
        raise ReplayError("Incomplete experiment_hashes")
    snapshot = source / "source-snapshot"
    expected_names = set(python_sources) | {"tennis-return.json"}
    if not snapshot.is_dir():
        raise ReplayError("Missing source-snapshot")
    actual_names = {
        path.name
        for path in snapshot.iterdir()
        if not (path.name == "__pycache__" and path.is_dir())
    }
    if actual_names != expected_names:
        raise ReplayError("Source snapshot file set mismatch")
    hashes = {}
    for name, expected in python_sources.items():
        path = snapshot / name
        if Path(name).name != name or not path.is_file() or sha256(path) != expected:
            raise ReplayError(f"Python source snapshot hash mismatch: {path}")
        hashes[name] = expected
    task_spec = snapshot / "tennis-return.json"
    if not task_spec.is_file() or sha256(task_spec) != task_spec_hash:
        raise ReplayError(f"Task spec snapshot hash mismatch: {task_spec}")
    hashes[task_spec.name] = task_spec_hash
    return dict(sorted(hashes.items()))


def _terminal_sample(record: dict) -> dict:
    trace_fields = {
        "step",
        "observation_before",
        "action",
        "observation_after",
        "done",
    }
    return {key: value for key, value in record.items() if key not in trace_fields}


def preflight_source(source: Path) -> dict:
    source = Path(source).resolve()
    evaluation_path = source / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    if evaluation.get("evidence_type") != "metrics_only":
        raise ReplayError("Source must be a metrics_only evidence bundle")
    if evaluation.get("video_seeds") not in ([], None):
        raise ReplayError("Source metrics bundle must not declare video seeds")
    release_mode = evaluation.get("release_mode")
    if not isinstance(release_mode, str):
        raise ReplayError("Missing release_mode")
    current_experiment = experiment_provenance(release_mode)
    if evaluation.get("experiment_hashes") != current_experiment:
        raise ReplayError("Frozen experiment_provenance does not match repository")
    current_source = source_provenance()
    if evaluation.get("source_provenance") != current_source:
        raise ReplayError("Frozen source_provenance does not match repository")
    snapshots = _verify_snapshots(source, evaluation)

    seeds = evaluation.get("requested_seeds")
    records = evaluation.get("records")
    controls = evaluation.get("negative_controls")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
        or not isinstance(records, list)
        or not isinstance(controls, list)
    ):
        raise ReplayError("Incomplete episode matrix")
    expected = {(variant, seed) for variant in VARIANTS for seed in seeds}
    actual = {
        (record.get("variant"), record.get("seed"))
        for record in records
        if isinstance(record, dict)
    }
    controls_by_name = {
        record.get("controller"): record
        for record in controls
        if isinstance(record, dict)
    }
    if len(records) != len(expected) or actual != expected:
        raise ReplayError("Incomplete episode matrix")
    if len(controls) != len(CONTROLS) or set(controls_by_name) != set(CONTROLS):
        raise ReplayError("Incomplete negative controls")

    episodes = {}
    ordered = [
        (f"{record['variant']}-{record['seed']}", record)
        for record in records
    ] + [(control, controls_by_name[control]) for control in CONTROLS]
    for directory_name, evaluation_record in ordered:
        directory = source / directory_name
        result_path = directory / "result.json"
        result = _read_json(result_path)
        if result != evaluation_record:
            raise ReplayError(f"Evaluation/result mismatch: {result_path}")
        expected_controller = (
            directory_name if directory_name in CONTROLS else "teacher"
        )
        expected_variant = (
            "nominal"
            if directory_name in CONTROLS
            else directory_name.rsplit("-", 1)[0]
        )
        if (
            result.get("variant") != expected_variant
            or result.get("candidate") != evaluation.get("candidate")
            or result.get("controller") != expected_controller
            or result.get("release_mode") != release_mode
            or result.get("experiment_provenance") != current_experiment
        ):
            raise ReplayError(f"Episode identity/provenance mismatch: {result_path}")
        if (directory / "rollout.mp4").exists():
            raise ReplayError(f"Metrics source unexpectedly contains video: {directory}")
        file_hashes = _safe_file_hashes(directory, result)
        telemetry_path = directory / "telemetry.jsonl"
        telemetry = _read_telemetry(telemetry_path)
        if result.get("steps") != len(telemetry):
            raise ReplayError(f"Telemetry step count mismatch: {directory}")
        for index, record in enumerate(telemetry, start=1):
            required = {
                "step",
                "action",
                "time_s",
                "ball_position_m",
                "joint_positions_rad",
                "phase",
                "done",
                "success",
                "failure_reason",
            }
            if not required <= record.keys() or record["step"] != index:
                raise ReplayError(f"Incomplete telemetry at {directory}:{index}")
            if record["done"] is not (index == len(telemetry)):
                raise ReplayError(f"Invalid terminal sequence: {directory}:{index}")
        terminal = telemetry[-1]
        if _terminal_sample(terminal) != result.get("last_sample"):
            raise ReplayError(f"Terminal telemetry does not equal last_sample: {directory}")
        if (
            terminal.get("success") != result.get("success")
            or terminal.get("failure_reason") != result.get("failure_reason")
        ):
            raise ReplayError(f"Terminal outcome mismatch: {directory}")
        episodes[directory_name] = {
            "result": result,
            "telemetry": telemetry,
            "telemetry_sha256": file_hashes["telemetry.jsonl"],
            "result_sha256": sha256(result_path),
        }
    return {
        "source": source,
        "evaluation": evaluation,
        "evaluation_sha256": sha256(evaluation_path),
        "experiment_provenance": current_experiment,
        "source_provenance": current_source,
        "snapshot_hashes": snapshots,
        "episodes": episodes,
        "ordered_directories": tuple(name for name, _ in ordered),
    }


def _max_abs(expected, actual, field: str, step: int) -> float:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    if (
        expected_array.shape != actual_array.shape
        or not np.isfinite(expected_array).all()
        or not np.isfinite(actual_array).all()
    ):
        raise ReplayError(f"Invalid {field} at step {step}")
    return float(np.max(np.abs(expected_array - actual_array), initial=0.0))


def _frame_writer(path: Path):
    import imageio.v2 as imageio

    return imageio.get_writer(
        path,
        fps=FRAME_RATE,
        codec="libx264",
        macro_block_size=16,
    )


def _overlay_frame(frame: np.ndarray, result: dict, record: dict) -> np.ndarray:
    from PIL import Image, ImageDraw

    if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
        raise ReplayError(f"Unexpected render shape: {frame.shape}")
    image = Image.fromarray(frame)
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, FRAME_WIDTH, 66), fill="black")
    drawing.text((8, 5), "SIMULATION ACTION REPLAY", fill="white")
    drawing.text(
        (8, 25),
        (
            f"{result['variant']} | seed {result['seed']} | "
            f"{result['controller']} | phase {record['phase']}"
        ),
        fill="white",
    )
    status = (
        "TASK PASS"
        if record["success"]
        else record["failure_reason"] or "REPLAY IN PROGRESS"
    )
    drawing.text(
        (8, 45),
        f"t={record['time_s']:.3f}s | {status}",
        fill="yellow",
    )
    return np.asarray(image)


def replay_episode(
    directory: Path,
    result: dict,
    telemetry: list[dict],
    source_context: dict,
    *,
    render: bool = True,
    env_factory: Callable[..., Any] = TennisReturnEnv,
    writer_factory: Callable[[Path], Any] = _frame_writer,
) -> dict:
    directory = Path(directory)
    validation_path = directory / "replay-validation.json"
    video_path = directory / "rollout.mp4"
    temporary_video = directory / ".rollout.action-replay.tmp.mp4"
    for path in (validation_path, video_path, temporary_video):
        if path.exists():
            raise ReplayError(f"Replay output already exists: {path}")

    validation = {
        "complete": False,
        "claimed_video_pass": False,
        "generation_method": "action_replay",
        "simulation_only": True,
        "hardware_release": False,
        "ppo": False,
        "trained_policy": False,
        "physics_tolerance": PHYSICS_TOLERANCE,
        "time_tolerance_s": TIME_TOLERANCE_S,
        "action_tolerance": ACTION_TOLERANCE,
        "steps_expected": len(telemetry),
        "steps_replayed": 0,
        "frames_written": 0,
        "substep_monitoring": "ReturnMonitor.advance observed on every env.step physics substep",
        "substep_monitor_calls": 0,
        "source_evaluation_sha256": source_context["evaluation_sha256"],
        "source_result_sha256": source_context["result_sha256"],
        "source_telemetry_sha256": source_context["telemetry_sha256"],
        "source_snapshot_hashes": source_context["snapshot_hashes"],
        "max_time_error_s": 0.0,
        "max_ball_position_error_m": 0.0,
        "max_joint_position_error_rad": 0.0,
        "max_action_roundtrip_error": 0.0,
    }
    environment = None
    writer = None
    try:
        environment = env_factory(
            variant=result["variant"],
            candidate=result["candidate"],
            max_steps=result["max_steps"],
            gripper=result["gripper"],
            release_mode=result["release_mode"],
        )
        environment.reset(result["seed"])
        original_advance = environment.monitor.advance

        def monitored_advance(sample, duration):
            validation["substep_monitor_calls"] += 1
            return original_advance(sample, duration)

        environment.monitor.advance = monitored_advance
        if render:
            writer = writer_factory(temporary_video)
        final_actual = None
        for index, expected in enumerate(telemetry, start=1):
            action64 = np.asarray(expected["action"], dtype=np.float64)
            action = np.asarray(expected["action"], dtype=np.float32)
            if (
                action.shape != (15,)
                or not np.isfinite(action).all()
                or np.any(action < -1 - ACTION_TOLERANCE)
                or np.any(action > 1 + ACTION_TOLERANCE)
            ):
                raise ReplayError(f"Invalid action at step {index}")
            action_error = _max_abs(action64, action, "action", index)
            validation["max_action_roundtrip_error"] = max(
                validation["max_action_roundtrip_error"],
                action_error,
            )
            if action_error > ACTION_TOLERANCE:
                raise ReplayError(f"Action precision mismatch at step {index}")

            _, done, actual = environment.step(action)
            final_actual = actual
            previous_action = np.asarray(
                environment.robot.previous_action,
                dtype=np.float32,
            )
            if previous_action.shape != action.shape or not np.array_equal(
                previous_action,
                action,
            ):
                raise ReplayError(f"Applied action mismatch at step {index}")
            for timestamp in (expected["time_s"], actual["time_s"]):
                if type(timestamp) not in (float, int) or not math.isfinite(timestamp):
                    raise ReplayError(f"Invalid finite timestamp at step {index}")
            time_error = abs(float(expected["time_s"]) - float(actual["time_s"]))
            ball_error = _max_abs(
                expected["ball_position_m"],
                actual["ball_position_m"],
                "ball_position_m",
                index,
            )
            joint_error = _max_abs(
                expected["joint_positions_rad"],
                actual["joint_positions_rad"],
                "joint_positions_rad",
                index,
            )
            validation["max_time_error_s"] = max(
                validation["max_time_error_s"],
                time_error,
            )
            validation["max_ball_position_error_m"] = max(
                validation["max_ball_position_error_m"],
                ball_error,
            )
            validation["max_joint_position_error_rad"] = max(
                validation["max_joint_position_error_rad"],
                joint_error,
            )
            if time_error > TIME_TOLERANCE_S:
                raise ReplayError(f"Time mismatch at step {index}: {time_error}")
            if ball_error > PHYSICS_TOLERANCE:
                raise ReplayError(
                    f"ball_position_m mismatch at step {index}: {ball_error}"
                )
            if joint_error > PHYSICS_TOLERANCE:
                raise ReplayError(
                    f"joint_positions_rad mismatch at step {index}: {joint_error}"
                )
            for field, replayed in (
                ("done", done),
                ("phase", actual.get("phase")),
                ("success", actual.get("success")),
                ("failure_reason", actual.get("failure_reason")),
            ):
                if expected.get(field) != replayed:
                    raise ReplayError(
                        f"{field} mismatch at step {index}: "
                        f"{expected.get(field)!r} != {replayed!r}"
                    )
            validation["steps_replayed"] = index
            if writer is not None and (index % 2 == 0 or done):
                writer.append_data(_overlay_frame(environment.render(), result, actual))
                validation["frames_written"] += 1
        if not environment.done or validation["steps_replayed"] != len(telemetry):
            raise ReplayError("Replay did not terminate exactly with telemetry")
        if final_actual is None or (
            final_actual.get("success") != result["success"]
            or final_actual.get("failure_reason") != result["failure_reason"]
        ):
            raise ReplayError("Replay terminal outcome does not match result")
        if writer is not None:
            writer.close()
            writer = None
            temporary_video.replace(video_path)
        validation.update(
            complete=True,
            claimed_video_pass=render,
            zero_error=(
                validation["max_time_error_s"] == 0
                and validation["max_ball_position_error_m"] == 0
                and validation["max_joint_position_error_rad"] == 0
            ),
            video_sha256=sha256(video_path) if render else None,
            telemetry_sha256=sha256(directory / "telemetry.jsonl"),
        )
        _write_json(validation_path, validation)
        return validation
    except Exception as error:
        validation["error"] = f"{type(error).__name__}: {error}"
        validation["incomplete"] = True
        _write_json(validation_path, validation)
        raise
    finally:
        if writer is not None:
            writer.close()
        if temporary_video.exists():
            temporary_video.unlink()
        if not validation["complete"] and video_path.exists():
            video_path.unlink()
        if environment is not None:
            environment.close()


def _replay_job(job: dict) -> tuple[str, dict]:
    directory_name = job["directory_name"]
    validation = replay_episode(
        Path(job["output"]) / directory_name,
        job["result"],
        job["telemetry"],
        job["source_context"],
    )
    return directory_name, validation


def _verify_original_source(preflight: dict) -> None:
    source = preflight["source"]
    if sha256(source / "evaluation.json") != preflight["evaluation_sha256"]:
        raise ReplayError("Original source evaluation changed during replay")
    if _verify_snapshots(source, preflight["evaluation"]) != preflight["snapshot_hashes"]:
        raise ReplayError("Original source snapshot changed during replay")
    for name, episode in preflight["episodes"].items():
        directory = source / name
        if sha256(directory / "result.json") != episode["result_sha256"]:
            raise ReplayError(f"Original source result changed during replay: {name}")
        _safe_file_hashes(directory, episode["result"])


def replay_bundle(source: Path, output: Path, workers: int = 1) -> dict:
    if workers < 1:
        raise ReplayError("workers must be positive")
    output = Path(output)
    if output.resolve().is_relative_to(Path(source).resolve()):
        raise ReplayError("Output must be outside the immutable source bundle")
    if output.exists():
        raise ReplayError(f"Output already exists: {output}")
    preflight = preflight_source(source)
    source = preflight["source"]
    shutil.copytree(source, output)
    for name, episode in preflight["episodes"].items():
        copied = output / name / "telemetry.jsonl"
        if sha256(copied) != episode["telemetry_sha256"]:
            raise ReplayError(f"Copied telemetry changed: {copied}")

    source_metrics = {
        "path": str(source),
        "sha256": preflight["evaluation_sha256"],
    }
    jobs = []
    for directory_name in preflight["ordered_directories"]:
        episode = preflight["episodes"][directory_name]
        jobs.append(
            {
                "directory_name": directory_name,
                "output": str(output),
                "result": episode["result"],
                "telemetry": episode["telemetry"],
                "source_context": {
                    "evaluation_sha256": preflight["evaluation_sha256"],
                    "result_sha256": episode["result_sha256"],
                    "telemetry_sha256": episode["telemetry_sha256"],
                    "snapshot_hashes": preflight["snapshot_hashes"],
                },
            }
        )

    validations = {}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        replayed = executor.map(_replay_job, jobs) if executor else map(_replay_job, jobs)
        for directory_name, validation in replayed:
            validations[directory_name] = validation
            print(
                json.dumps(
                    {
                        "directory": directory_name,
                        "steps": validation["steps_replayed"],
                        "zero_error": validation["zero_error"],
                        "video": validation["claimed_video_pass"],
                    }
                ),
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    after_source = source_provenance()
    after_experiment = experiment_provenance(
        preflight["evaluation"]["release_mode"]
    )
    after_snapshot_hashes = _verify_snapshots(output, preflight["evaluation"])
    after_source_evaluation_hash = sha256(source / "evaluation.json")
    _verify_original_source(preflight)
    if (
        after_source != preflight["source_provenance"]
        or after_experiment != preflight["experiment_provenance"]
        or after_snapshot_hashes != preflight["snapshot_hashes"]
        or after_source_evaluation_hash != preflight["evaluation_sha256"]
    ):
        raise ReplayError("Source changed during replay; output remains incomplete")

    evaluation = copy.deepcopy(preflight["evaluation"])
    updated = {}
    for directory_name in preflight["ordered_directories"]:
        directory = output / directory_name
        result = _read_json(directory / "result.json")
        telemetry_hash = sha256(directory / "telemetry.jsonl")
        if telemetry_hash != preflight["episodes"][directory_name]["telemetry_sha256"]:
            raise ReplayError(f"Telemetry changed during replay: {directory}")
        result.update(
            evidence_type="metrics_and_video",
            generation_method="action_replay",
            source_metrics={
                **source_metrics,
                "result_sha256": preflight["episodes"][directory_name][
                    "result_sha256"
                ],
                "telemetry_path": f"{directory_name}/telemetry.jsonl",
                "telemetry_sha256": telemetry_hash,
            },
        )
        result["files"] = {
            **result["files"],
            "rollout.mp4": sha256(directory / "rollout.mp4"),
            "replay-validation.json": sha256(
                directory / "replay-validation.json"
            ),
        }
        _write_json(directory / "result.json", result)
        updated[directory_name] = result

    evaluation["records"] = [
        updated[f"{record['variant']}-{record['seed']}"]
        for record in evaluation["records"]
    ]
    evaluation["negative_controls"] = [
        updated[record["controller"]]
        for record in evaluation["negative_controls"]
    ]
    evaluation.update(
        video_seeds=list(evaluation["requested_seeds"]),
        evidence_type="metrics_and_video",
        generation_method="action_replay",
        source_metrics=source_metrics,
        replay_validation={
            "complete": True,
            "episodes": len(validations),
            "all_substeps_monitored": True,
            "physics_tolerance": PHYSICS_TOLERANCE,
            "time_tolerance_s": TIME_TOLERANCE_S,
            "action_tolerance": ACTION_TOLERANCE,
            "source_provenance_before": preflight["source_provenance"],
            "source_provenance_after": after_source,
            "source_snapshot_hashes_before": preflight["snapshot_hashes"],
            "source_snapshot_hashes_after": after_snapshot_hashes,
            "source_evaluation_sha256_before": preflight["evaluation_sha256"],
            "source_evaluation_sha256_after": after_source_evaluation_hash,
            "experiment_provenance_before": preflight[
                "experiment_provenance"
            ],
            "experiment_provenance_after": after_experiment,
        },
    )
    _write_json(output / "evaluation.json", evaluation)
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        report = replay_bundle(args.source, args.out, args.workers)
    except ReplayError as error:
        parser.exit(1, f"replay failed: {error}\n")
    print(
        json.dumps(
            {
                "episodes": report["episodes"],
                "video_seeds": report["video_seeds"],
                "evidence_type": report["evidence_type"],
                "generation_method": report["generation_method"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
