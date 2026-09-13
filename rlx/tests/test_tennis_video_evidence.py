from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/verify_tennis_videos.py"
SPEC = importlib.util.spec_from_file_location("tennis_video_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(autouse=True)
def trusted_baseline(tmp_path, monkeypatch):
    source = tmp_path / "frozen-baseline.py"
    source.write_text("frozen baseline\n", encoding="utf-8")

    def provenance():
        return {
            "files": {"baseline": {"path": str(source), "sha256": _sha256(source)}},
            "required_families": ["baseline"],
            "invalidation_rule": "frozen baseline",
        }

    monkeypatch.setattr(verifier, "_trusted_source_provenance", provenance)
    return source


def _write_episode(
    root: Path,
    directory_name: str,
    *,
    variant: str,
    controller: str,
    seed: int = 0,
    include_video: bool = True,
) -> dict:
    directory = root / directory_name
    directory.mkdir()
    terminal_sample = {
        "phase": "approach",
        "time_s": 0.02,
        "success": False,
        "failure_reason": "timeout",
    }
    terminal = {
        "step": 1,
        "observation_before": [0.0],
        "action": [0.0],
        "observation_after": [0.0],
        "done": True,
        **terminal_sample,
    }
    telemetry = directory / "telemetry.jsonl"
    telemetry.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    rollout = directory / "rollout.mp4"
    if include_video:
        rollout.write_bytes(b"synthetic-video")
    metrics = directory / "metrics.json"
    metrics.write_text('{"synthetic": true}\n', encoding="utf-8")
    files = [telemetry, metrics]
    if include_video:
        files.append(rollout)
    result = {
        "variant": variant,
        "candidate": "A",
        "seed": seed,
        "controller": controller,
        "success": False,
        "failure_reason": "timeout",
        "steps": 1,
        "last_sample": terminal_sample,
        "phase_events": [{"phase": "approach", "time_s": 0.0}],
        "ground_lift_passed": False,
        "simulation_only": True,
        "hardware_release": False,
        "feasibility": {"hardware_release": False},
        "files": {
            path.name: _sha256(path)
            for path in files
        },
    }
    _write_json(directory / "result.json", result)
    return result


def _bundle(
    tmp_path: Path,
    *,
    requested_seeds: tuple[int, ...] = (0,),
    video_seeds: tuple[int, ...] | None = None,
) -> Path:
    root = tmp_path / "evidence"
    root.mkdir()
    snapshot = root / "source-snapshot"
    snapshot.mkdir()
    python_sources = {}
    for name in ("__init__.py", "tennis_controller.py", "tennis_return.py"):
        path = snapshot / name
        path.write_text(f"{name}\n", encoding="utf-8")
        python_sources[name] = _sha256(path)
    task_spec = snapshot / "tennis-return.json"
    task_spec.write_text('{"task": "tennis"}\n', encoding="utf-8")
    records = [
        _write_episode(
            root,
            f"{variant}-{seed}",
            variant=variant,
            controller="teacher",
            seed=seed,
            include_video=seed in (
                verifier.LEGACY_VIDEO_SEEDS if video_seeds is None else video_seeds
            ),
        )
        for variant in verifier.VARIANTS
        for seed in requested_seeds
    ]
    controls = [
        _write_episode(
            root,
            control,
            variant="nominal",
            controller=control,
        )
        for control in verifier.CONTROLS
    ]
    evaluation = {
        "experiment_id": "tennis-ground-return-v1",
        "episodes": len(records),
        "successes": 0,
        "ground_lifts": 0,
        "simulation_only": True,
        "hardware_release": False,
        "training_steps": 0,
        "source_provenance": verifier._trusted_source_provenance(),
        "requested_seeds": list(requested_seeds),
        "all_tasks_passed": False,
        "negative_controls_passed": True,
        "records": records,
        "negative_controls": controls,
        "candidate": "A",
        "experiment_hashes": {
            "python_sources": python_sources,
            "task_spec": _sha256(task_spec),
        },
    }
    if video_seeds is not None:
        evaluation["video_seeds"] = list(video_seeds)
    _write_json(root / "evaluation.json", evaluation)
    return root


def _rewrite_telemetry_binding(root: Path, directory_name: str) -> None:
    result_path = root / directory_name / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["files"]["telemetry.jsonl"] = _sha256(
        root / directory_name / "telemetry.jsonl"
    )
    _write_json(result_path, result)
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    collection = (
        evaluation["negative_controls"]
        if directory_name in verifier.CONTROLS
        else evaluation["records"]
    )
    for index, record in enumerate(collection):
        if (
            record["variant"] == result["variant"]
            and record["seed"] == result["seed"]
            and record["controller"] == result["controller"]
        ):
            collection[index] = result
            break
    _write_json(evaluation_path, evaluation)


def _add_success_evidence(root: Path, directory_name: str = "nominal-0") -> None:
    spec_path = root / "source-snapshot/tennis-return.json"
    task_spec = {
        "task": "tennis",
        "experiment_id": "tennis-ground-return-v1",
        "bin": {
            "center_xy_m": [0.38, 0.12],
            "inside_size_xy_m": [0.18, 0.18],
            "wall_height_m": 0.10,
            "bottom_thickness_m": 0.004,
        },
        "acceptance": {
            "minimum_lift_clearance_m": 0.02,
            "lift_hold_s": 0.5,
            "minimum_carry_distance_m": 0.12,
            "placement_speed_max_m_s": 0.03,
            "bin_hold_s": 2.0,
            "retreat_clearance_m": 0.02,
            "fall_tilt_rad": 0.6,
            "minimum_trunk_height_m": 0.065,
        },
    }
    _write_json(spec_path, task_spec)
    phases = [
        ("approach", 0.0),
        ("grasp", 0.10),
        ("lift", 0.20),
        ("carry", 0.30),
        ("lower", 0.40),
        ("release", 0.50),
        ("retreat", 0.60),
        ("success", 2.60),
    ]
    events = [{"phase": phase, "time_s": time} for phase, time in phases]
    rows = []
    for step in range(1, 131):
        time = step * 0.02
        phase = max(
            (phase for phase, event_time in phases if event_time <= time),
            key=verifier.PHASES.index,
        )
        in_bin = verifier.PHASES.index(phase) >= verifier.PHASES.index("lower")
        stable = phase in {"retreat", "success"}
        sample = {
            "time_s": time,
            "phase": phase,
            "success": phase == "success",
            "failure_reason": None,
            "tilt_rad": 0.1,
            "trunk_height_m": 0.1,
            "forbidden_contacts": [],
            "joint_limit_violation": False,
            "ball_position_m": [0.38, 0.12, 0.0375 if in_bin else 0.08],
            "ball_radius_m": 0.0335,
            "ball_bottom_m": 0.004 if in_bin else 0.0465,
            "ball_speed_m_s": 0.01,
            "tcp_ball_distance_m": 0.06 if stable else 0.002,
            "carry_distance_m": 0.2,
            "bottom_supported": in_bin,
            "any_robot_ball_contact": not stable,
            "ball_inside_bin": in_bin,
            "jaw_open": phase in {"approach", "release", "retreat", "success"},
            "bilateral_grasp": phase in {"grasp", "lift", "carry", "lower", "release"},
        }
        rows.append({
            "step": step,
            "observation_before": [0.0],
            "action": [0.0],
            "observation_after": [0.0],
            "done": phase == "success",
            **sample,
        })
    telemetry_path = root / directory_name / "telemetry.jsonl"
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_path = root / directory_name / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        success=True,
        failure_reason=None,
        steps=len(rows),
        last_sample={
            key: value for key, value in rows[-1].items()
            if key not in verifier.TRACE_FIELDS
        },
        phase_events=events,
        ground_lift_passed=True,
    )
    _write_json(result_path, result)
    _rewrite_telemetry_binding(root, directory_name)
    evaluation_path = root / "evaluation.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["experiment_hashes"]["task_spec"] = _sha256(spec_path)
    evaluation["successes"] = 1
    evaluation["ground_lifts"] = 1
    _write_json(evaluation_path, evaluation)


def test_preflight_accepts_complete_hash_consistent_bundle(tmp_path):
    root = _bundle(tmp_path)

    evidence = verifier.preflight_evidence(root)

    assert set(evidence["episodes"]) == {
        "nominal-0",
        "small-0",
        "large-0",
        "open_jaw",
        "hold",
    }
    assert evidence["episodes"]["nominal-0"]["hashes"]["metrics.json"] == _sha256(
        root / "nominal-0/metrics.json"
    )
    assert evidence["verified_video_count"] == 5


def test_preflight_accepts_success_with_independent_physical_evidence(tmp_path):
    root = _bundle(tmp_path)
    _add_success_evidence(root)

    evidence = verifier.preflight_evidence(root)

    physical = evidence["episodes"]["nominal-0"]["physical_success"]
    assert physical["release_mode"] == "supported"
    assert physical["sampled_final_stability_s"] == pytest.approx(1.98)
    assert physical["verification"] == "independent_sampled_acceptance_envelope_50Hz"
    assert physical["substep_replay"] is False


@pytest.mark.parametrize("boundary_time", [0.60, 0.62])
def test_final_dwell_window_is_open_only_at_exact_start(tmp_path, boundary_time):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    path = root / "nominal-0/telemetry.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if abs(row["time_s"] - boundary_time) < 1e-9:
            row["tcp_ball_distance_m"] = 0.04
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _rewrite_telemetry_binding(root, "nominal-0")
    if boundary_time == 0.60:
        verifier.preflight_evidence(root)
    else:
        with pytest.raises(verifier.EvidenceError, match="stability window"):
            verifier.preflight_evidence(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ball_position_m", None, "physical telemetry"),
        ("ball_speed_m_s", float("nan"), "physical telemetry"),
        ("forbidden_contacts", ["robot_bin_collision"], "physical safety"),
        ("joint_limit_violation", True, "physical safety"),
        ("tilt_rad", 0.7, "physical safety"),
        ("ball_position_m", [0.55, 0.12, 0.0375], "containment"),
        ("bottom_supported", False, "stability window"),
        ("any_robot_ball_contact", True, "stability window"),
        ("tcp_ball_distance_m", 0.04, "stability window"),
    ],
)
def test_preflight_rejects_tampered_success_physical_evidence(
    tmp_path, field, value, message
):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    telemetry_path = root / "nominal-0/telemetry.jsonl"
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    if field in {"bottom_supported", "any_robot_ball_contact", "tcp_ball_distance_m"}:
        targets = [row for row in rows if row["phase"] in {"retreat", "success"}]
    elif field == "ball_position_m" and value == [0.55, 0.12, 0.0375]:
        targets = [row for row in rows if row["phase"] in {"release", "retreat", "success"}]
    else:
        targets = [rows[-1]]
    for row in targets:
        row[field] = value
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    result_path = root / "nominal-0/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["last_sample"] = {
        key: value for key, value in rows[-1].items()
        if key not in verifier.TRACE_FIELDS
    }
    _write_json(result_path, result)
    _rewrite_telemetry_binding(root, "nominal-0")

    with pytest.raises(verifier.EvidenceError, match=message):
        verifier.preflight_evidence(root)


def test_preflight_rejects_success_with_phase_only_synthetic_trace(tmp_path):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    path = root / "nominal-0/result.json"
    result = json.loads(path.read_text())
    result.update(
        last_sample={
            "phase": "success",
            "time_s": 0.02,
            "success": True,
            "failure_reason": None,
        },
        phase_events=[
            {"phase": phase, "time_s": index * 0.001}
            for index, phase in enumerate(verifier.PHASES)
        ],
        ground_lift_passed=True,
        steps=1,
    )
    terminal = {
        "step": 1,
        "observation_before": [0.0],
        "action": [0.0],
        "observation_after": [0.0],
        "done": True,
        **result["last_sample"],
    }
    (root / "nominal-0/telemetry.jsonl").write_text(json.dumps(terminal) + "\n")
    _write_json(path, result)
    _rewrite_telemetry_binding(root, "nominal-0")

    with pytest.raises(verifier.EvidenceError, match="physical telemetry"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_unsafe_first_sample_after_exact_dwell_start(tmp_path):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    telemetry_path = root / "nominal-0/telemetry.jsonl"
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    row = next(row for row in rows if row["time_s"] == pytest.approx(0.62))
    row["bottom_supported"] = False
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _rewrite_telemetry_binding(root, "nominal-0")

    with pytest.raises(verifier.EvidenceError, match="stability window"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_gap_inside_final_dwell_window(tmp_path):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    telemetry_path = root / "nominal-0/telemetry.jsonl"
    rows = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
    ]
    del rows[80]
    for step, row in enumerate(rows, start=1):
        row["step"] = step
    result_path = root / "nominal-0/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["steps"] = len(rows)
    result["last_sample"] = {
        key: value for key, value in rows[-1].items()
        if key not in verifier.TRACE_FIELDS
    }
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(result_path, result)
    _rewrite_telemetry_binding(root, "nominal-0")

    with pytest.raises(verifier.EvidenceError, match="control rate"):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize(
    ("mode", "side", "penetration", "expected"),
    [
        ("half_height", "bin_x_low", 0.0002, True),
        ("half_height", "bin_x_high", 0.0002, False),
        ("half_height", "bin_x_low", 0.0006, False),
        ("supported", "bin_x_low", 0.0002, False),
    ],
)
@pytest.mark.parametrize("force_qualified", [False, True])
def test_recomputed_containment_only_allows_qualified_half_height_side_contact(
    tmp_path, mode, side, penetration, expected, force_qualified
):
    sample = {
        "ball_position_m": [0.3233, 0.12, 0.0375],
        "ball_radius_m": 0.0335,
        "ball_bin_side_contact_geoms": [side],
        "maximum_ball_bin_side_penetration_m": penetration,
        "active_contacts": [{
            "geoms": [side, "object_geom"],
            "penetration_m": penetration,
        }] if force_qualified else [],
    }
    contained = verifier._recomputed_containment(
        sample,
        mode=mode,
        gates={"side_contact_containment_tolerance_m": 0.0005},
        bin_geometry=([0.38, 0.12], [0.09, 0.09], 0.004, 0.10),
        directory=tmp_path,
    )

    assert contained is expected


@pytest.mark.parametrize("tamper", [None, "throw", "free_fall"])
def test_half_height_success_physical_profile_is_accepted(tmp_path, tamper):
    root = _bundle(tmp_path)
    _add_success_evidence(root)
    telemetry = [
        json.loads(line)
        for line in (root / "nominal-0/telemetry.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for sample in telemetry:
        sample.update(
            ball_velocity_m_s=[0.001, 0.0, 0.0],
            ball_physically_contained_in_bin=sample["ball_inside_bin"],
            ball_bin_side_contact_geoms=[],
            maximum_ball_bin_side_penetration_m=0.0,
            maximum_free_fall_interval_s=0.0,
            active_contacts=[],
        )
    result = json.loads((root / "nominal-0/result.json").read_text())
    result["release_mode"] = "half_height"
    result["last_sample"].update(
        release_authorization_height_m=0.0375,
        release_authorization_velocity_m_s=[0.001, 0.0, 0.0],
    )
    gates = {
        **json.loads(
            (root / "source-snapshot/tennis-return.json").read_text()
        )["acceptance"],
        "release_ball_center_max_wall_height_fraction": 0.5,
        "release_upward_velocity_max_m_s": 0.005,
        "release_lateral_speed_max_m_s": 0.02,
        "side_contact_containment_tolerance_m": 0.0005,
        "airborne_upward_velocity_max_m_s": 0.03,
        "airborne_lateral_speed_max_m_s": 0.08,
        "maximum_free_fall_interval_s": 0.2,
    }
    task_spec = {
        "bin": {
            "center_xy_m": [0.38, 0.12],
            "inside_size_xy_m": [0.18, 0.18],
            "wall_height_m": 0.10,
            "bottom_thickness_m": 0.004,
        },
        "release_profiles": {
            "half_height": {"acceptance": gates},
        },
    }

    if tamper == "throw":
        telemetry[27].update(bottom_supported=False, any_robot_ball_contact=False,
                             ball_velocity_m_s=[0.09, 0.0, 0.04], ball_speed_m_s=0.098488578)
    elif tamper == "free_fall":
        telemetry[27]["maximum_free_fall_interval_s"] = 0.201
    if tamper:
        with pytest.raises(verifier.EvidenceError, match="throw velocity|free-fall interval"):
            verifier._verify_success_physical(tmp_path, result, telemetry, task_spec,
                                              "tennis-half-height-gravity-return-v2")
        return
    physical = verifier._verify_success_physical(
        tmp_path,
        result,
        telemetry,
        task_spec,
        "tennis-half-height-gravity-return-v2",
    )

    assert physical["release_mode"] == "half_height"
    assert physical["substep_replay"] is False


@pytest.mark.parametrize("tamper", [None, "evaluation", "provenance", "result", "telemetry", "unknown_mode"])
def test_release_profile_is_bound_to_snapshot_and_every_episode(tmp_path, tamper):
    (tmp_path / "source-snapshot").mkdir()
    profile = {
        "experiment_id": "tennis-half-height-gravity-return-v2",
        "acceptance_version": "half-height-gravity-drop-v1",
        "acceptance": {"require_bottom_support_before_release": False},
    }
    _write_json(tmp_path / "source-snapshot" / "tennis-return.json", {
        "release_profiles": {"half_height": profile},
    })
    declared = {"release_mode": "half_height", **profile}
    evaluation = {**declared, "experiment_hashes": dict(declared)}
    episodes = {"nominal-0": {
        "result": dict(declared),
        "telemetry": [{key: value for key, value in declared.items() if key != "acceptance"}],
    }}
    if tamper == "evaluation":
        evaluation["acceptance"] = {"require_bottom_support_before_release": True}
    elif tamper == "provenance":
        evaluation["experiment_hashes"]["release_mode"] = "supported"
    elif tamper == "result":
        episodes["nominal-0"]["result"]["release_mode"] = "supported"
    elif tamper == "telemetry":
        episodes["nominal-0"]["telemetry"][0]["release_mode"] = "supported"
    elif tamper == "unknown_mode":
        evaluation["release_mode"] = "unrecorded_rule_change"
    if tamper is None:
        verifier._verify_release_profile(tmp_path, evaluation, episodes)
    else:
        with pytest.raises(verifier.EvidenceError, match="release"):
            verifier._verify_release_profile(tmp_path, evaluation, episodes)


def test_preflight_accepts_declared_multi_seed_videos(tmp_path):
    seeds = tuple(range(10))
    root = _bundle(
        tmp_path,
        requested_seeds=seeds,
        video_seeds=seeds,
    )

    evidence = verifier.preflight_evidence(root)

    assert set(evidence["video_directories"]) == {
        *(f"{variant}-{seed}" for variant in verifier.VARIANTS for seed in seeds),
        *verifier.CONTROLS,
    }
    assert evidence["verified_video_count"] == 32


def test_preflight_rejects_declared_missing_video(tmp_path):
    root = _bundle(
        tmp_path,
        requested_seeds=(0, 1),
        video_seeds=(0, 1),
    )
    (root / "large-1/rollout.mp4").unlink()

    with pytest.raises(verifier.EvidenceError, match="Missing result file"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_unexpected_video(tmp_path):
    root = _bundle(
        tmp_path,
        requested_seeds=(0, 1),
        video_seeds=(0,),
    )
    (root / "large-1/rollout.mp4").write_bytes(b"unexpected-video")

    with pytest.raises(verifier.EvidenceError, match="Expected exactly"):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize(
    "video_seeds",
    [
        None,
        True,
        [],
        [True],
        [0.0],
        [0, 0],
        [-1],
        [1],
    ],
)
def test_preflight_rejects_invalid_video_seeds(tmp_path, video_seeds):
    root = _bundle(tmp_path)
    path = root / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["video_seeds"] = video_seeds
    _write_json(path, evaluation)

    with pytest.raises(verifier.EvidenceError, match="video_seeds"):
        verifier.preflight_evidence(root)


def test_preflight_preserves_historical_default_video_seed_zero(tmp_path):
    root = _bundle(tmp_path, requested_seeds=(0, 1))

    evidence = verifier.preflight_evidence(root)

    assert set(evidence["video_directories"]) == {
        "nominal-0",
        "small-0",
        "large-0",
        "open_jaw",
        "hold",
    }
    assert evidence["verified_video_count"] == 5


def test_preflight_rejects_videos_in_arbitrary_folders(tmp_path):
    root = _bundle(tmp_path)
    arbitrary = root / "arbitrary"
    arbitrary.mkdir()
    (arbitrary / "rollout.mp4").write_bytes(b"unbound-video")

    with pytest.raises(verifier.EvidenceError, match="Expected exactly"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_missing_evaluation_report(tmp_path):
    root = _bundle(tmp_path)
    (root / "evaluation.json").unlink()

    with pytest.raises(verifier.EvidenceError, match="Missing required report"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_terminal_mismatch_with_consistent_file_hash(tmp_path):
    root = _bundle(tmp_path)
    telemetry = root / "nominal-0/telemetry.jsonl"
    terminal = json.loads(telemetry.read_text(encoding="utf-8"))
    terminal["phase"] = "grasp"
    telemetry.write_text(json.dumps(terminal) + "\n", encoding="utf-8")
    _rewrite_telemetry_binding(root, "nominal-0")

    with pytest.raises(verifier.EvidenceError, match="does not equal last_sample"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_result_not_equal_to_evaluation_record(tmp_path):
    root = _bundle(tmp_path)
    result_path = root / "nominal-0/result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["controller"] = "hold"
    _write_json(result_path, result)

    with pytest.raises(verifier.EvidenceError, match="identity mismatch"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_tampered_telemetry(tmp_path):
    root = _bundle(tmp_path)
    telemetry = root / "nominal-0/telemetry.jsonl"
    telemetry.write_text(
        telemetry.read_text(encoding="utf-8").replace("approach", "grasp"),
        encoding="utf-8",
    )

    with pytest.raises(verifier.EvidenceError, match="file hash mismatch"):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize(
    "relative_path",
    ["source-snapshot/tennis_return.py", "source-snapshot/tennis-return.json"],
)
def test_preflight_rejects_tampered_source_snapshot(tmp_path, relative_path):
    root = _bundle(tmp_path)
    snapshot = root / relative_path
    snapshot.write_text("rewritten\n", encoding="utf-8")

    with pytest.raises(verifier.EvidenceError, match="snapshot hash mismatch"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_legacy_bundle_without_source_snapshot(tmp_path):
    root = _bundle(tmp_path)
    for path in (root / "source-snapshot").iterdir():
        path.unlink()
    (root / "source-snapshot").rmdir()

    with pytest.raises(verifier.EvidenceError, match="Unsupported evidence bundle"):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ground_lifts", 1),
        ("ground_lifts", False),
        ("simulation_only", False),
        ("simulation_only", 1),
        ("hardware_release", True),
        ("hardware_release", 0),
        ("training_steps", 1),
        ("training_steps", False),
        ("training_steps", None),
        ("source_provenance", None),
    ],
)
def test_preflight_rejects_forged_evaluation_fields(tmp_path, field, value):
    root = _bundle(tmp_path)
    path = root / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation[field] = value
    _write_json(path, evaluation)

    with pytest.raises(verifier.EvidenceError, match=field):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize("directory", ["nominal-0", "open_jaw", "hold"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ground_lift_passed", True),
        ("simulation_only", False),
        ("hardware_release", True),
        ("training_steps", 123),
        ("feasibility", {"hardware_release": True}),
    ],
)
def test_preflight_rejects_rebound_result_scope(tmp_path, directory, field, value):
    root = _bundle(tmp_path)
    path = root / directory / "result.json"
    result = json.loads(path.read_text())
    result[field] = value
    _write_json(path, result)
    _rewrite_telemetry_binding(root, directory)

    with pytest.raises(verifier.EvidenceError, match=field):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize("tamper", ["missing", "extra", "path", "hash", "families", "rule"])
def test_preflight_rejects_forged_baseline_manifest(tmp_path, tamper):
    root = _bundle(tmp_path)
    path = root / "evaluation.json"
    evaluation = json.loads(path.read_text())
    provenance = evaluation["source_provenance"]
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text("untrusted replacement\n")
    if tamper == "missing":
        provenance["files"].clear()
    elif tamper == "extra":
        provenance["files"]["extra"] = {"path": str(unrelated), "sha256": _sha256(unrelated)}
    elif tamper == "path":
        provenance["files"]["baseline"] = {"path": str(unrelated), "sha256": _sha256(unrelated)}
    elif tamper == "hash":
        provenance["files"]["baseline"]["sha256"] = "0" * 64
    elif tamper == "families":
        provenance["required_families"] = []
    else:
        provenance["invalidation_rule"] = "not frozen"
    _write_json(path, evaluation)

    with pytest.raises(verifier.EvidenceError, match="source_provenance"):
        verifier.preflight_evidence(root)


def test_preflight_rejects_changed_trusted_baseline(tmp_path, trusted_baseline):
    root = _bundle(tmp_path)
    trusted_baseline.write_text("changed baseline\n")

    with pytest.raises(verifier.EvidenceError, match="source_provenance"):
        verifier.preflight_evidence(root)


def test_preflight_fails_closed_when_trusted_baseline_unavailable(tmp_path, monkeypatch):
    root = _bundle(tmp_path)

    def unavailable():
        raise FileNotFoundError("missing trusted baseline")

    monkeypatch.setattr(verifier, "_trusted_source_provenance", unavailable)
    with pytest.raises(verifier.EvidenceError, match="Cannot establish trusted baseline"):
        verifier.preflight_evidence(root)


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_preflight_accepts_historical_experiment_snapshot(tmp_path, version):
    root = _bundle(tmp_path)
    cache = root / "source-snapshot/__pycache__"
    cache.mkdir()
    (cache / "tennis_controller.cpython-312.pyc").write_bytes(b"unused bytecode")
    snapshot = root / "source-snapshot/tennis_controller.py"
    snapshot.write_text(f"historical controller {version}\n")
    path = root / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["experiment_hashes"]["python_sources"][snapshot.name] = _sha256(snapshot)
    _write_json(path, evaluation)

    evidence = verifier.preflight_evidence(root)

    assert evidence["snapshots"][snapshot.name] == _sha256(snapshot)


@pytest.mark.parametrize("extra", ["unexpected.py", "unexpected-directory"])
def test_preflight_rejects_extra_snapshot_sources(tmp_path, extra):
    root = _bundle(tmp_path)
    path = root / "source-snapshot" / extra
    if path.suffix == ".py":
        path.write_text("extra source\n")
    else:
        path.mkdir()

    with pytest.raises(verifier.EvidenceError, match="Source snapshot"):
        verifier.preflight_evidence(root)


def _add_lift_evidence(root, directory="nominal-0"):
    path = root / directory / "result.json"
    result = json.loads(path.read_text())
    rows = []
    events = []
    for index, phase in enumerate(verifier.PHASES[:4]):
        events.append({"phase": phase, "time_s": index * 0.1})
        sample = {
            "phase": phase,
            "time_s": index * 0.1 + 0.02,
            "success": False,
            "failure_reason": "timeout" if phase == "carry" else None,
        }
        rows.append({"step": index + 1, "done": phase == "carry", **sample})
    (root / directory / "telemetry.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result.update(last_sample=sample, phase_events=events, steps=4, ground_lift_passed=True)
    _write_json(path, result)
    _rewrite_telemetry_binding(root, directory)
    path = root / "evaluation.json"
    evaluation = json.loads(path.read_text())
    evaluation["ground_lifts"] = int(directory not in verifier.CONTROLS)
    _write_json(path, evaluation)


@pytest.mark.parametrize("directory", ["nominal-0", "hold"])
def test_preflight_recomputes_lifts_without_counting_controls(tmp_path, directory):
    root = _bundle(tmp_path)
    _add_lift_evidence(root, directory)

    evidence = verifier.preflight_evidence(root)

    assert evidence["episodes"][directory]["ground_lift_passed"] is True


@pytest.mark.parametrize("tamper", ["phase", "time", "record", "summary", "event"])
def test_preflight_rejects_inconsistent_lift_evidence(tmp_path, tamper):
    root = _bundle(tmp_path)
    _add_lift_evidence(root)
    path = root / "nominal-0/result.json"
    result = json.loads(path.read_text())
    if tamper == "phase":
        result["phase_events"][1]["phase"] = "lower"
    elif tamper == "time":
        result["phase_events"][1]["time_s"] = 0.13
    elif tamper == "record":
        result["ground_lift_passed"] = False
    elif tamper == "event":
        result["phase_events"].pop()
    _write_json(path, result)
    _rewrite_telemetry_binding(root, "nominal-0")
    if tamper == "summary":
        path = root / "evaluation.json"
        evaluation = json.loads(path.read_text())
        evaluation["ground_lifts"] = 0
        _write_json(path, evaluation)

    with pytest.raises(verifier.EvidenceError, match="phase|ground_lift"):
        verifier.preflight_evidence(root)
