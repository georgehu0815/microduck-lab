"""Decode tennis experiment videos, cross-check telemetry, and make a contact sheet."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import TypeGuard

from PIL import Image, ImageDraw

from microduck_arm_v1c.config import sha256


VARIANTS = ("nominal", "small", "large")
CONTROLS = ("open_jaw", "hold")
LEGACY_VIDEO_SEEDS = (0,)
TRACE_FIELDS = {
    "step",
    "observation_before",
    "action",
    "observation_after",
    "done",
}
PHASES = ("approach", "grasp", "lift", "carry", "lower", "release", "retreat", "success")
MAX_CONTROL_SAMPLE_S = 0.020001


class EvidenceError(ValueError):
    pass


def _trusted_source_provenance() -> dict:
    from microduck_arm_v1c.whole_body_learning import source_provenance

    return source_provenance()


def _verify_baseline(evaluation: dict) -> None:
    try:
        trusted = _trusted_source_provenance()
    except Exception as error:
        raise EvidenceError("Cannot establish trusted baseline source_provenance") from error
    if not trusted.get("files") or evaluation.get("source_provenance") != trusted:
        raise EvidenceError("Frozen baseline source_provenance does not match trusted repository")


def _verify_scope(record: dict, location: Path, *, evaluation: bool = False) -> None:
    if record.get("simulation_only") is not True or record.get("hardware_release") is not False:
        raise EvidenceError(f"Invalid simulation_only/hardware_release scope: {location}")
    if evaluation or "training_steps" in record:
        if type(record.get("training_steps")) is not int or record["training_steps"] != 0:
            raise EvidenceError(f"Invalid training_steps scope: {location}")
    feasibility = record.get("feasibility")
    if feasibility is not None and (
        not isinstance(feasibility, dict) or feasibility.get("hardware_release") is not False
    ):
        raise EvidenceError(f"Invalid feasibility hardware_release scope: {location}")


def _verify_ground_lift(directory: Path, result: dict, telemetry: list[dict]) -> bool:
    events = result.get("phase_events")
    if (
        not isinstance(events, list)
        or not events
        or not all(isinstance(event, dict) for event in events)
        or [event.get("phase") for event in events] != list(PHASES[:len(events)])
        or events[0].get("time_s") != 0
    ):
        raise EvidenceError(f"Invalid phase_events sequence: {directory}")
    times = [event.get("time_s") for event in events]
    if (
        any(type(time) not in (int, float) or not math.isfinite(time) for time in times)
        or any(current <= previous for previous, current in zip(times, times[1:]))
    ):
        raise EvidenceError(f"Invalid phase_events times: {directory}")
    event_index = 0
    previous_time = 0.
    for step, sample in enumerate(telemetry, start=1):
        raw_time = sample.get("time_s")
        if not isinstance(raw_time, (int, float)) or isinstance(raw_time, bool):
            raise EvidenceError(f"Invalid telemetry sequence: {directory}")
        time = float(raw_time)
        if (
            not math.isfinite(time)
            or time <= previous_time
            or type(sample.get("step")) is not int
            or sample["step"] != step
            or sample.get("done") is not (step == len(telemetry))
        ):
            raise EvidenceError(f"Invalid telemetry sequence: {directory}")
        while event_index + 1 < len(events) and times[event_index + 1] <= time:
            event_index += 1
        if sample.get("phase") != events[event_index]["phase"]:
            raise EvidenceError(f"phase_events disagree with telemetry: {directory}")
        previous_time = time
    if event_index != len(events) - 1:
        raise EvidenceError(f"phase_events extend beyond telemetry: {directory}")
    lifted = any(sample["phase"] in PHASES[3:] for sample in telemetry)
    if lifted != any(event["phase"] == "carry" for event in events):
        raise EvidenceError(f"Ground lift phase evidence mismatch: {directory}")
    if result.get("ground_lift_passed") is not lifted:
        raise EvidenceError(f"ground_lift_passed disagrees with phase evidence: {directory}")
    return lifted


def _finite_number(value) -> TypeGuard[float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _finite_vector(value, length: int) -> TypeGuard[list[float]]:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _episode_profile(
    result: dict,
    task_spec: dict,
    evaluation_experiment_id,
    directory: Path,
) -> tuple[str, dict]:
    profiles = task_spec.get("release_profiles")
    mode = result.get("release_mode")
    if profiles is None:
        if (
            mode not in (None, "supported")
            or evaluation_experiment_id != "tennis-ground-return-v1"
            or task_spec.get("experiment_id") != "tennis-ground-return-v1"
        ):
            raise EvidenceError(
                f"Omitted release_mode is not historical supported evidence: {directory}"
            )
        mode = "supported"
        gates = task_spec.get("acceptance")
    elif isinstance(profiles, dict) and isinstance(mode, str) and mode in profiles:
        profile = profiles[mode]
        gates = profile.get("acceptance") if isinstance(profile, dict) else None
    else:
        raise EvidenceError(f"Successful episode has no snapshotted release profile: {directory}")
    if mode not in {"supported", "half_height"} or not isinstance(gates, dict):
        raise EvidenceError(f"Invalid snapshotted release profile: {directory}")
    required = {
        "placement_speed_max_m_s",
        "bin_hold_s",
        "retreat_clearance_m",
        "fall_tilt_rad",
        "minimum_trunk_height_m",
    }
    if mode == "half_height":
        required |= {
            "release_ball_center_max_wall_height_fraction",
            "release_upward_velocity_max_m_s", "release_lateral_speed_max_m_s",
            "airborne_upward_velocity_max_m_s", "airborne_lateral_speed_max_m_s",
            "maximum_free_fall_interval_s", "side_contact_containment_tolerance_m",
        }
    if any(not _finite_number(gates.get(key)) for key in required):
        raise EvidenceError(f"Incomplete snapshotted physical acceptance gates: {directory}")
    return mode, gates


def _bin_geometry(task_spec: dict, directory: Path) -> tuple[list[float], list[float], float, float]:
    container = task_spec.get("bin")
    if not isinstance(container, dict):
        raise EvidenceError(f"Missing snapshotted bin geometry: {directory}")
    center = container.get("center_xy_m")
    size = container.get("inside_size_xy_m")
    bottom = container.get("bottom_thickness_m")
    wall_height = container.get("wall_height_m")
    if (
        not _finite_vector(center, 2)
        or not _finite_vector(size, 2)
        or any(value <= 0 for value in size)
        or not _finite_number(bottom)
        or not _finite_number(wall_height)
        or wall_height <= 0
    ):
        raise EvidenceError(f"Invalid snapshotted bin geometry: {directory}")
    return center, [value / 2 for value in size], float(bottom), float(wall_height)


def _recomputed_containment(
    sample: dict,
    *,
    mode: str,
    gates: dict,
    bin_geometry: tuple[list[float], list[float], float, float],
    directory: Path,
) -> bool:
    position = sample["ball_position_m"]
    radius = float(sample["ball_radius_m"])
    center, half, bottom, wall_height = bin_geometry
    vertical_inside = (
        position[2] + radius <= bottom + wall_height + 1e-9
        and position[2] - radius >= bottom - 0.001 - 1e-9
    )
    overrun = [
        abs(position[axis] - center[axis]) + radius - half[axis]
        for axis in range(2)
    ]
    geometric = vertical_inside and all(value <= 1e-12 for value in overrun)
    if mode == "supported" or geometric:
        return geometric

    tolerance = gates.get("side_contact_containment_tolerance_m")
    declared_penetration = sample.get("maximum_ball_bin_side_penetration_m")
    declared_sides = sample.get("ball_bin_side_contact_geoms")
    if (
        not _finite_number(tolerance)
        or tolerance < 0
        or not _finite_number(declared_penetration)
        or declared_penetration < 0
        or not isinstance(declared_sides, list)
        or not all(isinstance(name, str) for name in declared_sides)
    ):
        raise EvidenceError(f"Missing half-height contact containment telemetry: {directory}")
    expected_sides = set()
    for axis, label in enumerate(("x", "y")):
        if overrun[axis] > 1e-12:
            side = "high" if position[axis] >= center[axis] else "low"
            expected_sides.add(f"bin_{label}_{side}")
    return bool(
        vertical_inside
        and expected_sides
        and expected_sides <= set(declared_sides)
        and max(overrun) <= tolerance + 1e-12
        and float(declared_penetration) <= tolerance + 1e-12
    )


def _verify_success_physical(
    directory: Path,
    result: dict,
    telemetry: list[dict],
    task_spec: dict,
    evaluation_experiment_id,
) -> dict | None:
    if result.get("success") is not True:
        return None
    mode, gates = _episode_profile(
        result,
        task_spec,
        evaluation_experiment_id,
        directory,
    )
    bin_geometry = _bin_geometry(task_spec, directory)
    scalar_fields = (
        "tilt_rad",
        "trunk_height_m",
        "ball_radius_m",
        "ball_bottom_m",
        "ball_speed_m_s",
        "tcp_ball_distance_m",
        "carry_distance_m",
    )
    boolean_fields = (
        "bottom_supported",
        "any_robot_ball_contact",
        "joint_limit_violation",
        "ball_inside_bin",
        "jaw_open",
        "bilateral_grasp",
    )
    previous_time = 0.0
    maximum_sample_gap = 0.0
    for sample in telemetry:
        if (
            any(not _finite_number(sample.get(key)) for key in scalar_fields)
            or not _finite_vector(sample.get("ball_position_m"), 3)
            or sample["ball_radius_m"] <= 0
            or any(type(sample.get(key)) is not bool for key in boolean_fields)
            or not isinstance(sample.get("forbidden_contacts"), list)
            or not all(
                isinstance(name, str) for name in sample["forbidden_contacts"]
            )
        ):
            raise EvidenceError(f"Successful episode has missing or non-finite physical telemetry: {directory}")
        time = float(sample["time_s"])
        maximum_sample_gap = max(maximum_sample_gap, time - previous_time)
        previous_time = time
        if maximum_sample_gap > MAX_CONTROL_SAMPLE_S:
            raise EvidenceError(f"Successful episode physical telemetry is not sampled at control rate: {directory}")
        if (
            sample["forbidden_contacts"]
            or sample["joint_limit_violation"]
            or sample["tilt_rad"] > gates["fall_tilt_rad"] + 1e-12
            or sample["trunk_height_m"] < gates["minimum_trunk_height_m"] - 1e-12
        ):
            raise EvidenceError(f"Successful episode contains a physical safety failure: {directory}")
        if mode == "half_height" and not _finite_vector(sample.get("ball_velocity_m_s"), 3):
            raise EvidenceError(f"Successful half-height episode is missing physical velocity telemetry: {directory}")

    events = {event["phase"]: float(event["time_s"]) for event in result["phase_events"]}
    if set(events) != set(PHASES):
        raise EvidenceError(f"Successful episode has incomplete phase_events: {directory}")
    release_time = events["release"]
    release_samples = [
        sample for sample in telemetry
        if abs(float(sample["time_s"]) - release_time) <= MAX_CONTROL_SAMPLE_S + 1e-9
        and sample["phase"] in {"lower", "release"}
    ]
    if mode == "supported":
        release_ok = any(
            sample["bilateral_grasp"]
            and sample["bottom_supported"]
            and sample["ball_speed_m_s"] <= gates["placement_speed_max_m_s"] + 1e-12
            and _recomputed_containment(
                sample,
                mode=mode,
                gates=gates,
                bin_geometry=bin_geometry,
                directory=directory,
            )
            for sample in release_samples
        )
    else:
        required_half_gates = (
            "release_ball_center_max_wall_height_fraction",
            "release_upward_velocity_max_m_s",
            "release_lateral_speed_max_m_s",
        )
        if any(not _finite_number(gates.get(key)) for key in required_half_gates):
            raise EvidenceError(f"Incomplete half-height release gates: {directory}")
        authorization_height = result["last_sample"].get("release_authorization_height_m")
        authorization_velocity = result["last_sample"].get("release_authorization_velocity_m_s")
        center_max = (
            bin_geometry[2]
            + gates["release_ball_center_max_wall_height_fraction"] * bin_geometry[3]
        )
        if (
            not _finite_number(authorization_height)
            or not _finite_vector(authorization_velocity, 3)
            or authorization_height
            < bin_geometry[2] + result["last_sample"]["ball_radius_m"] - 0.001 - 1e-9
            or authorization_height > center_max + 1e-9
            or math.sqrt(sum(value * value for value in authorization_velocity))
            > gates["placement_speed_max_m_s"] + 1e-12
            or authorization_velocity[2] > gates["release_upward_velocity_max_m_s"] + 1e-12
            or math.hypot(*authorization_velocity[:2])
            > gates["release_lateral_speed_max_m_s"] + 1e-12
        ):
            raise EvidenceError(f"Half-height release authorization violates the physical profile: {directory}")
        release_ok = any(
            sample["bilateral_grasp"]
            and sample["ball_position_m"][2] <= center_max + 1e-9
            and _recomputed_containment(
                sample,
                mode="supported",
                gates=gates,
                bin_geometry=bin_geometry,
                directory=directory,
            )
            and math.sqrt(sum(value * value for value in sample["ball_velocity_m_s"]))
            <= gates["placement_speed_max_m_s"] + 1e-12
            and sample["ball_velocity_m_s"][2]
            <= gates["release_upward_velocity_max_m_s"] + 1e-12
            and math.hypot(*sample["ball_velocity_m_s"][:2])
            <= gates["release_lateral_speed_max_m_s"] + 1e-12
            for sample in release_samples
        )
    if not release_ok:
        raise EvidenceError(f"Successful episode lacks sampled physical release-entry evidence: {directory}")

    for sample in telemetry:
        if sample["phase"] not in {"release", "retreat", "success"}:
            continue
        contained = _recomputed_containment(
            sample,
            mode=mode,
            gates=gates,
            bin_geometry=bin_geometry,
            directory=directory,
        )
        declared = (
            sample.get("ball_physically_contained_in_bin")
            if mode == "half_height"
            else sample.get("ball_inside_bin")
        )
        if type(declared) is not bool or declared is not contained or not contained:
            raise EvidenceError(f"Successful episode fails recomputed physical containment: {directory}")
        if mode == "half_height":
            maximum_fall = sample.get("maximum_free_fall_interval_s")
            if not _finite_number(maximum_fall) or not 0 <= maximum_fall <= gates["maximum_free_fall_interval_s"] + 1e-9:
                raise EvidenceError(f"Successful episode exceeds maximum free-fall interval: {directory}")
            airborne = not sample["bottom_supported"] and not sample["any_robot_ball_contact"]
            velocity = sample["ball_velocity_m_s"]
            if airborne and (velocity[2] > gates["airborne_upward_velocity_max_m_s"] + 1e-12 or math.hypot(*velocity[:2]) > gates["airborne_lateral_speed_max_m_s"] + 1e-12):
                raise EvidenceError(f"Successful episode contains measured throw velocity: {directory}")

    def stable(sample: dict) -> bool:
        return bool(
            sample["phase"] in {"retreat", "success"}
            and sample["jaw_open"]
            and sample["bottom_supported"]
            and not sample["any_robot_ball_contact"]
            and sample["ball_speed_m_s"] <= gates["placement_speed_max_m_s"] + 1e-12
            and sample["tcp_ball_distance_m"]
            >= sample["ball_radius_m"] + gates["retreat_clearance_m"] - 1e-12
            and (
                sample.get("ball_physically_contained_in_bin")
                if mode == "half_height"
                else sample["ball_inside_bin"]
            )
        )

    success_time = events["success"]
    required_duration = float(gates["bin_hold_s"])
    required_start = success_time - required_duration
    window = [
        sample for sample in telemetry
        if float(sample["time_s"]) > required_start + 1e-9
    ]
    if (
        not window
        or window[-1] is not telemetry[-1]
        or float(window[0]["time_s"]) - required_start > maximum_sample_gap + 1e-9
        or any(
            float(current["time_s"]) - float(previous["time_s"])
            > maximum_sample_gap + 1e-9
            for previous, current in zip(window, window[1:])
        )
        or not all(stable(sample) for sample in window)
    ):
        raise EvidenceError(f"Successful episode lacks a continuous sampled final stability window: {directory}")
    sampled_duration = success_time - float(window[0]["time_s"])
    exact_duration = next(
        (
            result["last_sample"].get(key)
            for key in ("final_stable_duration_s", "bin_stable_duration_s", "stable_duration_s")
            if key in result["last_sample"]
        ),
        None,
    )
    if exact_duration is not None and (
        not _finite_number(exact_duration)
        or exact_duration < required_duration - 1e-9
    ):
        raise EvidenceError(f"Invalid declared monitor stability duration: {directory}")
    return {
        "verification": "independent_sampled_acceptance_envelope_50Hz",
        "substep_replay": False,
        "release_mode": mode,
        "required_final_stability_start_s": required_start,
        "sampled_final_stability_s": sampled_duration,
        "maximum_control_sample_gap_s": maximum_sample_gap,
        "exact_monitor_stability_s": exact_duration,
    }


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise EvidenceError(f"Missing required report: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"Invalid JSON report: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise EvidenceError(f"Missing telemetry: {path}")
    try:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"Invalid telemetry JSONL: {path}") from error
    if not records or not all(isinstance(record, dict) for record in records):
        raise EvidenceError(f"Telemetry must contain JSON objects: {path}")
    return records


def _require_identity(record: dict, expected: dict, path: Path) -> None:
    actual = {key: record.get(key) for key in expected}
    if actual != expected:
        raise EvidenceError(
            f"Result identity mismatch for {path}: expected {expected}, got {actual}"
        )


def _verify_result_files(directory: Path, result: dict) -> dict[str, str]:
    files = result.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceError(f"Missing result file hashes: {directory / 'result.json'}")
    hashes = {}
    for name, expected_hash in sorted(files.items()):
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).is_absolute()
        ):
            raise EvidenceError(f"Unsafe result file name in {directory}: {name!r}")
        path = directory / name
        if not path.is_file():
            raise EvidenceError(f"Missing result file: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise EvidenceError(f"Result file hash mismatch: {path}")
        hashes[name] = actual_hash
    if "telemetry.jsonl" not in hashes:
        raise EvidenceError(f"Telemetry is not bound by result.json: {directory}")
    return hashes


def _verify_terminal(directory: Path, result: dict) -> tuple[list[dict], dict]:
    telemetry = _read_jsonl(directory / "telemetry.jsonl")
    terminal = telemetry[-1]
    if terminal.get("done") is not True:
        raise EvidenceError(f"Telemetry is not terminal: {directory}")
    if result.get("steps") != len(telemetry) or terminal.get("step") != len(telemetry):
        raise EvidenceError(f"Telemetry step count mismatch: {directory}")
    terminal_sample = {
        key: value for key, value in terminal.items() if key not in TRACE_FIELDS
    }
    if terminal_sample != result.get("last_sample"):
        raise EvidenceError(f"Terminal telemetry does not equal last_sample: {directory}")
    if (
        terminal.get("success") != result.get("success")
        or terminal.get("failure_reason") != result.get("failure_reason")
    ):
        raise EvidenceError(f"Terminal outcome mismatch: {directory}")
    if result.get("success") is True and result.get("failure_reason") is not None:
        raise EvidenceError(f"Successful result has a failure reason: {directory}")
    return telemetry, terminal


def _verify_episode(
    root: Path,
    directory_name: str,
    evaluation_record: dict,
    expected_identity: dict,
    task_spec: dict,
    evaluation_experiment_id,
) -> dict:
    directory = root / directory_name
    result_path = directory / "result.json"
    result = _read_json(result_path)
    _require_identity(result, expected_identity, result_path)
    if result != evaluation_record:
        raise EvidenceError(
            f"result.json does not equal its evaluation record: {result_path}"
        )
    _verify_scope(result, result_path)
    file_hashes = _verify_result_files(directory, result)
    telemetry, terminal = _verify_terminal(directory, result)
    ground_lift = _verify_ground_lift(directory, result, telemetry)
    physical_success = _verify_success_physical(
        directory,
        result,
        telemetry,
        task_spec,
        evaluation_experiment_id,
    )
    return {
        "directory": directory_name,
        "result": result,
        "terminal": terminal,
        "telemetry": telemetry,
        "ground_lift_passed": ground_lift,
        "physical_success": physical_success,
        "hashes": {
            "result.json": sha256(result_path),
            **file_hashes,
        },
    }


def _verify_snapshots(root: Path, evaluation: dict) -> dict[str, str]:
    experiment_hashes = evaluation.get("experiment_hashes")
    if not isinstance(experiment_hashes, dict):
        raise EvidenceError("Unsupported evaluation: missing experiment_hashes")
    python_sources = experiment_hashes.get("python_sources")
    task_spec_hash = experiment_hashes.get("task_spec")
    if not isinstance(python_sources, dict) or not python_sources:
        raise EvidenceError(
            "Unsupported evaluation: missing experiment_hashes.python_sources"
        )
    if not isinstance(task_spec_hash, str):
        raise EvidenceError(
            "Unsupported evaluation: missing experiment_hashes.task_spec"
        )
    snapshot = root / "source-snapshot"
    if not snapshot.is_dir():
        raise EvidenceError(
            "Unsupported evidence bundle: source-snapshot is required"
        )
    expected_names = set(python_sources) | {"tennis-return.json"}
    actual_paths = sorted(
        path for path in snapshot.iterdir()
        if not (path.name == "__pycache__" and path.is_dir())
    )
    if any(not path.is_file() for path in actual_paths):
        raise EvidenceError("Source snapshot must contain only files")
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise EvidenceError(
            f"Source snapshot file set mismatch: expected {sorted(expected_names)}, "
            f"got {sorted(actual_names)}"
        )
    hashes = {}
    for name, expected_hash in sorted(python_sources.items()):
        if Path(name).name != name or not name.endswith(".py"):
            raise EvidenceError(f"Invalid Python source snapshot name: {name!r}")
        path = snapshot / name
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise EvidenceError(f"Python source snapshot hash mismatch: {path}")
        hashes[name] = actual_hash
    task_spec = snapshot / "tennis-return.json"
    actual_task_spec_hash = sha256(task_spec)
    if actual_task_spec_hash != task_spec_hash:
        raise EvidenceError(f"Task spec snapshot hash mismatch: {task_spec}")
    hashes[task_spec.name] = actual_task_spec_hash
    return hashes


def _verify_release_profile(root: Path, evaluation: dict, episodes: dict) -> None:
    if "release_mode" not in evaluation:
        return
    spec = _read_json(root / "source-snapshot" / "tennis-return.json")
    profiles = spec.get("release_profiles", {})
    mode = evaluation["release_mode"]
    if not isinstance(mode, str) or mode not in profiles:
        raise EvidenceError("Unknown snapshotted release_mode")
    profile = profiles[mode]
    if not isinstance(profile, dict):
        raise EvidenceError("Invalid snapshotted release profile")
    expected = {
        "release_mode": mode,
        "experiment_id": profile.get("experiment_id"),
        "acceptance_version": profile.get("acceptance_version"),
        "acceptance": profile.get("acceptance"),
    }
    if any(value is None or evaluation.get(key) != value for key, value in expected.items()):
        raise EvidenceError("Evaluation release profile disagrees with snapshot")
    for key, value in expected.items():
        if evaluation["experiment_hashes"].get(key) != value:
            raise EvidenceError("Experiment provenance release profile mismatch")
    for name, episode in episodes.items():
        result = episode["result"]
        if any(result.get(key) != value for key, value in expected.items()):
            raise EvidenceError(f"Episode release profile mismatch: {name}")
        for sample in episode["telemetry"]:
            if any(sample.get(key) != value for key, value in expected.items() if key != "acceptance"):
                raise EvidenceError(f"Telemetry release profile mismatch: {name}")


def preflight_evidence(root: Path) -> dict:
    root = Path(root)
    evaluation_path = root / "evaluation.json"
    evaluation = _read_json(evaluation_path)
    requested_seeds = evaluation.get("requested_seeds")
    has_video_seeds = "video_seeds" in evaluation
    video_seeds = evaluation.get("video_seeds")
    if (
        not isinstance(requested_seeds, list)
        or not requested_seeds
        or any(type(seed) is not int or seed < 0 for seed in requested_seeds)
        or len(requested_seeds) != len(set(requested_seeds))
    ):
        raise EvidenceError(
            "Unsupported evaluation: missing valid requested_seeds"
        )
    if not has_video_seeds:
        video_seeds = list(LEGACY_VIDEO_SEEDS)
    if (
        not isinstance(video_seeds, list)
        or not video_seeds
        or any(type(seed) is not int or seed < 0 for seed in video_seeds)
        or len(video_seeds) != len(set(video_seeds))
        or not set(video_seeds).issubset(requested_seeds)
    ):
        raise EvidenceError(
            "Unsupported evaluation: video_seeds must be distinct nonnegative "
            "integers selected from requested_seeds"
        )
    expected_video_directories = frozenset(
        {f"{variant}-{seed}" for variant in VARIANTS for seed in video_seeds}
        | set(CONTROLS)
    )
    candidate = evaluation.get("candidate")
    records = evaluation.get("records")
    controls = evaluation.get("negative_controls")
    if not isinstance(candidate, str):
        raise EvidenceError("Evaluation is missing candidate identity")
    if not isinstance(records, list) or not isinstance(controls, list):
        raise EvidenceError("Evaluation records are incomplete")
    expected_record_ids = {
        (variant, seed) for variant in VARIANTS for seed in requested_seeds
    }
    actual_record_ids = {
        (record.get("variant"), record.get("seed"))
        for record in records
        if isinstance(record, dict)
    }
    if (
        len(records) != len(expected_record_ids)
        or actual_record_ids != expected_record_ids
    ):
        raise EvidenceError("Evaluation does not contain the complete episode matrix")
    controls_by_name = {
        record.get("controller"): record
        for record in controls
        if isinstance(record, dict)
    }
    if len(controls) != len(CONTROLS) or set(controls_by_name) != set(CONTROLS):
        raise EvidenceError("Evaluation does not contain both negative controls")
    if (
        evaluation.get("episodes") != len(records)
        or evaluation.get("successes")
        != sum(record.get("success") is True for record in records)
        or evaluation.get("all_tasks_passed")
        != all(record.get("success") is True for record in records)
        or evaluation.get("negative_controls_passed")
        != all(record.get("success") is False for record in controls)
    ):
        raise EvidenceError("Evaluation summary does not match its records")
    snapshots = _verify_snapshots(root, evaluation)
    task_spec = _read_json(root / "source-snapshot" / "tennis-return.json")
    _verify_scope(evaluation, evaluation_path, evaluation=True)
    _verify_baseline(evaluation)
    episodes = {}
    for record in records:
        variant = record["variant"]
        seed = record["seed"]
        directory_name = f"{variant}-{seed}"
        episodes[directory_name] = _verify_episode(
            root,
            directory_name,
            record,
            {
                "variant": variant,
                "candidate": candidate,
                "seed": seed,
                "controller": "teacher",
            },
            task_spec,
            evaluation.get("experiment_id"),
        )
    for control in CONTROLS:
        episodes[control] = _verify_episode(
            root,
            control,
            controls_by_name[control],
            {
                "variant": "nominal",
                "candidate": candidate,
                "seed": 0,
                "controller": control,
            },
            task_spec,
            evaluation.get("experiment_id"),
        )
    ground_lifts = sum(
        episode["ground_lift_passed"]
        for name, episode in episodes.items()
        if name not in CONTROLS
    )
    if type(evaluation.get("ground_lifts")) is not int or evaluation["ground_lifts"] != ground_lifts:
        raise EvidenceError("Evaluation ground_lifts disagrees with phase evidence")
    expected_result_paths = {
        (root / name / "result.json").resolve() for name in episodes
    }
    actual_result_paths = {path.resolve() for path in root.rglob("result.json")}
    if actual_result_paths != expected_result_paths:
        raise EvidenceError("Unexpected or missing result.json directories")
    expected_videos = {
        (root / name / "rollout.mp4").resolve()
        for name in expected_video_directories
    }
    actual_videos = {path.resolve() for path in root.rglob("rollout.mp4")}
    if actual_videos != expected_videos:
        raise EvidenceError(
            "Expected exactly the videos declared by evaluation.video_seeds "
            "plus open_jaw and hold controls"
        )
    for name in expected_video_directories:
        if "rollout.mp4" not in episodes[name]["hashes"]:
            raise EvidenceError(f"Video is not bound by result.json: {root / name}")
    _verify_release_profile(root, evaluation, episodes)
    verified_video_count = len(expected_video_directories)
    return {
        "root": root,
        "evaluation": evaluation,
        "evaluation_sha256": sha256(evaluation_path),
        "snapshots": snapshots,
        "episodes": episodes,
        "video_directories": tuple(sorted(expected_video_directories)),
        "verified_video_count": verified_video_count,
    }


def verify_videos(preflight: dict) -> dict:
    root = preflight["root"]
    output = root / "video-verification"
    output.mkdir(exist_ok=False)
    videos = [
        root / directory / "rollout.mp4"
        for directory in preflight["video_directories"]
    ]
    if len(videos) != preflight["verified_video_count"]:
        raise EvidenceError("Preflight verified_video_count is inconsistent")
    sheet = Image.new("RGB", (1600, 270 * len(videos)), "white")
    drawing = ImageDraw.Draw(sheet)
    records = []
    for row, video in enumerate(videos):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(video),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=width,height,nb_read_frames,avg_frame_rate,duration",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        episode = preflight["episodes"][video.parent.name]
        telemetry = episode["telemetry"]
        terminal = episode["terminal"]
        expected_frames = sum(
            record["step"] % 2 == 0 or record["done"] for record in telemetry
        )
        if (
            int(probe["nb_read_frames"]) != expected_frames
            or (probe["width"], probe["height"]) != (640, 480)
            or probe["avg_frame_rate"] != "25/1"
        ):
            raise EvidenceError(f"Video/telemetry mismatch: {video}")
        for column, fraction in enumerate((0.0, 0.25, 0.50, 0.75, 0.95)):
            timestamp = float(probe["duration"]) * fraction
            frame = output / f"{video.parent.name}-{column}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    str(frame),
                ],
                check=True,
            )
            with Image.open(frame) as image:
                sheet.paste(
                    image.resize((320, 240)),
                    (column * 320, row * 270 + 30),
                )
            status = "TASK PASS" if terminal["success"] else "NOT TASK PASS"
            drawing.text(
                (column * 320 + 5, row * 270 + 8),
                f"{video.parent.name} | {timestamp:.2f}s | {status}",
                fill="black",
            )
        records.append(
            {
                "directory": video.parent.name,
                "path": str(video.relative_to(root)),
                "sha256": episode["hashes"]["rollout.mp4"],
                "probe": probe,
                "frames_match_telemetry": True,
                "full_decode_passed": True,
                "task_success": terminal["success"],
                "task_failure": terminal["failure_reason"],
                "result_sha256": episode["hashes"]["result.json"],
                "telemetry_sha256": episode["hashes"]["telemetry.jsonl"],
            }
        )
    if len(records) != preflight["verified_video_count"]:
        raise EvidenceError("Decoded verified_video_count is inconsistent")
    sheet_path = output / "contact-sheet.jpg"
    sheet.save(sheet_path, quality=92)
    report = {
        "videos": records,
        "verified_video_count": preflight["verified_video_count"],
        "video_integrity_passed": True,
        "this_is_not_task_success": True,
        "authentication_scope": "internal_hash_consistency_not_cryptographic_attestation",
        "acceptance_verification": "independent_sampled_acceptance_envelope_50Hz",
        "substep_replay": False,
        "evaluation_sha256": preflight["evaluation_sha256"],
        "episode_hashes": {
            name: episode["hashes"]
            for name, episode in sorted(preflight["episodes"].items())
        },
        "source_snapshot_hashes": preflight["snapshots"],
        "verifier_sha256": sha256(__file__),
        "sheet_sha256": sha256(sheet_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = verify_videos(preflight_evidence(args.root))
    print(
        json.dumps(
            {
                "verified_video_count": report["verified_video_count"],
                "task_successes": sum(
                    record["task_success"] for record in report["videos"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
