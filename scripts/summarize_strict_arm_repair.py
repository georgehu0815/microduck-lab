from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlx/examples"))

import ppo_microduck_arm as pipeline  # noqa: E402
from rlx.environments import arm  # noqa: E402


CASE_NAMES = {
    "arm-reach-v1": "末端触达",
    "arm-pick-place-v1": "抓取放置",
    "arm-relocate-v1": "绕障移物",
    "arm-carry-v1": "航点搬运",
    "arms-handover-v1": "双臂交接（5 g 课程）",
    "arms-co-carry-v1": "双臂共同搬运（17 g 总重）",
}
TRAINING_SEEDS = (101, 202, 303)
PRIMARY_SEED_BASE = 80000
VALIDATION_SEED_BASE = 110000
SEEDS_PER_CONFIG = 10
MINIMUM_PER_CONFIG_SUCCESS_RATE = 0.8
PLOT_FIELDS = (
    "rollout_episode_reward_mean",
    "policy_gradient_loss",
    "value_loss",
    "entropy_loss",
    "approx_kl",
    "clip_fraction",
)
HISTORICAL = ROOT / "rlx/runs/arm/recheck-20260912-v2"
NEGATIVE_SEED_BASE = 140000
NEGATIVE_CONTROLLERS = {
    "arm-reach-v1": ("zero",),
    "arm-pick-place-v1": ("zero", "open_gripper"),
    "arm-relocate-v1": ("zero", "open_gripper"),
    "arm-carry-v1": ("zero", "open_gripper"),
    "arms-handover-v1": ("zero", "open_gripper", "disable_right"),
    "arms-co-carry-v1": ("zero", "open_gripper", "disable_right"),
}
VERIFICATION_LOGS = {
    "rlx": ("rlx-arm-95-pytest.log", 95, r"95 passed"),
    "backend": ("backend-arm-lab-unittest.log", 21, r"Ran 21 tests"),
    "control": ("arm-control-unittest.log", 35, r"Ran 35 tests"),
    "hardware_software": ("hardware-md-arm-t1-unittest.log", 6, r"Ran 6 tests"),
    "video": ("arm-video-evidence-unittest.log", 8, r"Ran 8 tests"),
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required evidence is missing: {path}")
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path, directory: Path) -> str:
    return Path(os.path.relpath(path, directory)).as_posix()


def expected_samples(
    seed_base: int, seeds_per_config: int = SEEDS_PER_CONFIG
) -> dict[tuple[int, int], dict[str, float]]:
    return {
        (
            config_index,
            seed_base + config_index * pipeline.EVAL_CONFIG_SEED_STRIDE + offset,
        ): options
        for config_index, options in enumerate(pipeline.EVAL_CONFIGS)
        for offset in range(seeds_per_config)
    }


def validate_evaluation(
    path: Path,
    *,
    case_id: str,
    metadata: dict[str, Any],
    seed_base: int,
) -> dict[str, Any]:
    report = read_json(path)
    expected = expected_samples(seed_base)
    actual = [(row["config_index"], row["seed"]) for row in report["results"]]
    expected_manifest = pipeline.independent_eval_seeds(seed_base, SEEDS_PER_CONFIG)

    if report["case_id"] != case_id or report["controller"] != "ppo_residual":
        raise ValueError(f"Evaluation case/controller mismatch: {path}")
    if report["source_hashes"] != metadata["source_hashes"]:
        raise ValueError(f"Evaluation source hashes mismatch: {path}")
    if report["model_sha256"] != metadata["model_sha256"]:
        raise ValueError(f"Evaluation model hash mismatch: {path}")
    if report["checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError(f"Evaluation checkpoint hash mismatch: {path}")
    if report.get("evaluation_seeds") != expected_manifest:
        raise ValueError(f"Evaluation seed manifest mismatch: {path}")
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"Incomplete or duplicate evaluation matrix: {path}")

    for row in report["results"]:
        identity = (row["config_index"], row["seed"])
        if row["options"] != expected[identity]:
            raise ValueError(f"Evaluation perturbation mismatch at {identity}: {path}")
        if row["passed"] != all(row["gates"].values()):
            raise ValueError(
                f"Episode verdict disagrees with gates at {identity}: {path}"
            )

    episodes = len(report["results"])
    successes = sum(row["passed"] for row in report["results"])
    per_config = [
        sum(
            row["passed"]
            for row in report["results"]
            if row["config_index"] == config_index
        )
        / SEEDS_PER_CONFIG
        for config_index in range(len(pipeline.EVAL_CONFIGS))
    ]
    statistical_gate = (
        successes / episodes >= arm.SPECS[case_id].success_rate
        and min(per_config) >= MINIMUM_PER_CONFIG_SUCCESS_RATE
    )
    if report["episodes"] != episodes or report["successes"] != successes:
        raise ValueError(f"Evaluation totals do not reconcile: {path}")
    if report["per_config_success_rate"] != per_config:
        raise ValueError(f"Per-config rates do not reconcile: {path}")
    if report["passed"] != statistical_gate:
        raise ValueError(f"Statistical gate does not reconcile: {path}")

    failures = [row for row in report["results"] if not row["passed"]]
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "episodes": episodes,
        "successes": successes,
        "failures": len(failures),
        "failed_gates": dict(
            Counter(
                gate
                for row in failures
                for gate, passed in row["gates"].items()
                if not passed
            )
        ),
        "all_episodes_pass": successes == episodes,
        "statistical_gate_passed": statistical_gate,
        "per_config_success_rate": per_config,
        "wilson_95": report["wilson_95"],
        "seed_manifest": expected_manifest,
    }


def validate_primary_summary(
    training: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    protocol = read_json(training / "strict-protocol.json")
    if protocol["source_hashes"] != pipeline.sources():
        raise ValueError("Primary protocol source hashes are stale")
    if protocol["steps_per_checkpoint"] != 32768:
        raise ValueError("Primary protocol does not specify 32768 steps/checkpoint")
    if protocol["cases"] != list(arm.CASES):
        raise ValueError("Primary protocol case order differs from the pipeline")
    if protocol["training_seeds"] != list(TRAINING_SEEDS):
        raise ValueError("Primary protocol training seeds differ")
    if protocol["evaluation_seed_base"] != PRIMARY_SEED_BASE:
        raise ValueError("Primary protocol seed base differs")
    if protocol["seeds_per_config"] != SEEDS_PER_CONFIG:
        raise ValueError("Primary protocol seeds-per-config differs")
    if protocol["perturbations"] != pipeline.EVAL_CONFIGS:
        raise ValueError("Primary protocol perturbations differ")

    summary = read_json(training / "strict-summary.json")
    identities = {(row["case_id"], row["seed"]) for row in summary["jobs"]}
    expected_identities = {
        (case_id, seed) for case_id in arm.CASES for seed in TRAINING_SEEDS
    }
    if identities != expected_identities or len(summary["jobs"]) != 18:
        raise ValueError("Primary strict summary does not contain exactly 18 jobs")

    episodes = sum(row["primary"]["episodes"] for row in checkpoints)
    successes = sum(row["primary"]["successes"] for row in checkpoints)
    if summary["execution_completed"] is not True:
        raise ValueError("Primary strict run is incomplete")
    if summary["episodes"] != episodes or summary["successes"] != successes:
        raise ValueError("Primary strict summary totals do not reconcile")
    for job in summary["jobs"]:
        checkpoint = next(
            row
            for row in checkpoints
            if row["case_id"] == job["case_id"] and row["training_seed"] == job["seed"]
        )
        primary = checkpoint["primary"]
        if (
            job["episodes"] != primary["episodes"]
            or job["successes"] != primary["successes"]
        ):
            raise ValueError("Primary per-checkpoint summary totals do not reconcile")
        if job["all_episodes_pass"] != primary["all_episodes_pass"]:
            raise ValueError("Primary pass-all status does not reconcile")
        if job["statistical_gate_passed"] != primary["statistical_gate_passed"]:
            raise ValueError("Primary statistical status does not reconcile")
    return {
        "path": str((training / "strict-summary.json").resolve()),
        "sha256": digest(training / "strict-summary.json"),
        "episodes": episodes,
        "successes": successes,
        "all_episodes_pass": successes == episodes,
        "all_checkpoints_meet_statistical_gate": all(
            row["primary"]["statistical_gate_passed"] for row in checkpoints
        ),
        "seed_min": PRIMARY_SEED_BASE,
        "seed_max": PRIMARY_SEED_BASE
        + (len(pipeline.EVAL_CONFIGS) - 1) * pipeline.EVAL_CONFIG_SEED_STRIDE
        + SEEDS_PER_CONFIG
        - 1,
    }


def validate_heldout_summary(
    validation: Path, training: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    protocol_path = validation / "protocol.json"
    summary_path = validation / "summary.json"
    protocol = read_json(protocol_path)
    summary = read_json(summary_path)
    if protocol["source_hashes"] != pipeline.sources():
        raise ValueError("Held-out protocol source hashes are stale")
    if protocol["seed_base"] != VALIDATION_SEED_BASE:
        raise ValueError("Held-out protocol seed base differs")
    if protocol["seeds_per_config"] != SEEDS_PER_CONFIG:
        raise ValueError("Held-out protocol seeds-per-config differs")
    if protocol["perturbations"] != pipeline.EVAL_CONFIGS:
        raise ValueError("Held-out protocol perturbations differ")
    if protocol["training_seeds"] != list(TRAINING_SEEDS):
        raise ValueError("Held-out protocol training seeds differ")
    if Path(protocol["source_batch"]).resolve() != training.resolve():
        raise ValueError(
            "Held-out protocol is not bound to the requested training batch"
        )
    if summary["source_hashes"] != pipeline.sources():
        raise ValueError("Held-out summary source hashes are stale")
    if summary["execution_completed"] is not True or summary.get("errors"):
        raise ValueError("Held-out evaluation is incomplete")

    identities = {(job["case_id"], job["training_seed"]) for job in summary["jobs"]}
    expected_identities = {
        (case_id, seed) for case_id in arm.CASES for seed in TRAINING_SEEDS
    }
    if identities != expected_identities or len(summary["jobs"]) != 18:
        raise ValueError("Held-out summary does not contain exactly 18 jobs")

    episodes = sum(row["validation"]["episodes"] for row in checkpoints)
    successes = sum(row["validation"]["successes"] for row in checkpoints)
    if summary["episodes"] != episodes or summary["successes"] != successes:
        raise ValueError("Held-out summary totals do not reconcile")
    for job in summary["jobs"]:
        checkpoint = next(
            row
            for row in checkpoints
            if row["case_id"] == job["case_id"]
            and row["training_seed"] == job["training_seed"]
        )
        heldout = checkpoint["validation"]
        if (
            job["episodes"] != heldout["episodes"]
            or job["successes"] != heldout["successes"]
        ):
            raise ValueError("Held-out per-checkpoint totals do not reconcile")
        if job["passed"] != heldout["statistical_gate_passed"]:
            raise ValueError("Held-out statistical status does not reconcile")
        if (
            Path(job["checkpoint"]).resolve()
            != Path(checkpoint["checkpoint"]).resolve()
        ):
            raise ValueError("Held-out summary points at a different checkpoint")
        if job["evaluation_sha256"] != heldout["sha256"]:
            raise ValueError("Held-out evaluation hash differs from its summary")

    all_episodes_pass = successes == episodes
    all_statistical = all(
        row["validation"]["statistical_gate_passed"] for row in checkpoints
    )
    if summary["all_episodes_pass"] != all_episodes_pass:
        raise ValueError("Held-out pass-all status does not reconcile")
    if summary["all_checkpoints_meet_threshold"] != all_statistical:
        raise ValueError("Held-out statistical status does not reconcile")
    return {
        "path": str(summary_path.resolve()),
        "sha256": digest(summary_path),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": digest(protocol_path),
        "episodes": episodes,
        "successes": successes,
        "all_episodes_pass": all_episodes_pass,
        "all_checkpoints_meet_statistical_gate": all_statistical,
        "seed_min": VALIDATION_SEED_BASE,
        "seed_max": VALIDATION_SEED_BASE
        + (len(pipeline.EVAL_CONFIGS) - 1) * pipeline.EVAL_CONFIG_SEED_STRIDE
        + SEEDS_PER_CONFIG
        - 1,
    }


def validate_repair(
    training: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    audit_path = training / "repair-source-audit.json"
    audit = read_json(audit_path)
    expected_sources = pipeline.sources()
    if audit["changed_functions"] != ["teacher_targets"]:
        raise ValueError("Repair audit is not limited to teacher_targets")
    for field in (
        "task_gates_unchanged",
        "step_reward_and_physics_unchanged",
        "make_xml_unchanged",
    ):
        if audit.get(field) is not True:
            raise ValueError(f"Repair audit does not preserve {field}")
    if audit["new_environment_sha256"] != expected_sources["environment"]:
        raise ValueError("Repair audit environment hash is stale")

    repaired_source = (
        Path(checkpoints[0]["directory"])
        / "sources"
        / pipeline.SOURCE_ARCHIVE_NAMES["environment"]
    )
    historical_source = HISTORICAL / "frozen-sources/rlx/rlx/environments/arm.py"
    if digest(repaired_source) != audit["new_environment_sha256"]:
        raise ValueError("Archived repaired environment differs from the audit")
    if digest(historical_source) != audit["old_environment_sha256"]:
        raise ValueError("Archived historical environment differs from the audit")
    old_lines = historical_source.read_text().splitlines()
    new_lines = repaired_source.read_text().splitlines()
    changes = list(difflib.ndiff(old_lines, new_lines))
    added = [line[2:] for line in changes if line.startswith("+ ")]
    removed = [line[2:] for line in changes if line.startswith("- ")]
    if len(added) != 6 or removed:
        raise ValueError("Repair is not the audited six-line additive change")
    repair_text = "\n".join(added)
    required_fragments = (
        'self.case_id == "arms-co-carry-v1"',
        "self.phase in (2, 3, 4)",
        "fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)",
        "position = begin + (end - begin) * blend",
        "self.grasp_offsets",
    )
    if not all(fragment in repair_text for fragment in required_fragments):
        raise ValueError("Repair lines do not match synchronized quintic interpolation")
    if "[3.0, 2.0, 3.0, 6.0, 3.0, 2.0, 3.0]" not in repaired_source.read_text():
        raise ValueError("Co-carry phase durations are not the audited 3/6/3 seconds")
    return {
        **audit,
        "path": str(audit_path.resolve()),
        "sha256": digest(audit_path),
        "added_lines": 6,
        "removed_lines": 0,
        "lift_transport_descent_seconds": [3.0, 6.0, 3.0],
        "interpolation": "synchronized quintic Cartesian targets",
    }


def validate_regressions(
    training: Path, checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    path = training / "known-failure-regressions.json"
    report = read_json(path)
    if report["episodes"] != 21 or len(report["results"]) != 21:
        raise ValueError("Known-failure regression evidence must contain 21 outcomes")
    checkpoint_map = {
        (row["case_id"], row["training_seed"]): row for row in checkpoints
    }
    for row in report["results"]:
        identity = (row["case_id"], row["training_seed"])
        checkpoint = checkpoint_map.get(identity)
        if checkpoint is None:
            raise ValueError(f"Regression references an unknown checkpoint: {identity}")
        if (
            Path(row["checkpoint"]).resolve()
            != Path(checkpoint["checkpoint"]).resolve()
        ):
            raise ValueError("Regression checkpoint path mismatch")
        if row["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]:
            raise ValueError("Regression checkpoint hash mismatch")
        if row["model_sha256"] != checkpoint["model_sha256"]:
            raise ValueError("Regression model hash mismatch")
        if row["source_hashes"] != checkpoint["source_hashes"]:
            raise ValueError("Regression source hashes mismatch")
        if row["case_id"] != "arms-co-carry-v1":
            raise ValueError("Known-failure regression includes an unexpected case")
        if row["episode"]["seed"] != row["original_episode"]["seed"]:
            raise ValueError("Known-failure regression changed the evaluation seed")
        if row["episode"]["options"] != row["original_episode"]["options"]:
            raise ValueError("Known-failure regression changed perturbations")
        if row["episode"]["passed"] != all(row["episode"]["gates"].values()):
            raise ValueError("Known-failure regression verdict disagrees with gates")

    successes = sum(row["episode"]["passed"] for row in report["results"])
    if report["successes"] != successes or report["all_passed"] != (successes == 21):
        raise ValueError("Known-failure regression totals do not reconcile")
    historical_summary = HISTORICAL / "summary.json"
    if digest(historical_summary) != report["historical_summary_sha256"]:
        raise ValueError("Known-failure regression historical summary changed")
    if Path(report["historical_batch"]).resolve() != HISTORICAL.resolve():
        raise ValueError("Known-failure regression points at another historical batch")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "episodes": 21,
        "successes": successes,
        "failures": 21 - successes,
        "all_passed": successes == 21,
    }


def validate_timestep(training: Path) -> dict[str, Any] | None:
    path = training / "timestep-convergence.json"
    if not path.is_file():
        return None
    report = read_json(path)
    if report["source_hashes"] != pipeline.sources():
        raise ValueError("Timestep evidence source hashes are stale")
    expected = {
        (case_id, timestep) for case_id in arm.CASES for timestep in (0.002, 0.001)
    }
    actual = [(row["case_id"], row["physics_timestep_s"]) for row in report["results"]]
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError("Timestep evidence must contain the expected 12 outcomes")
    for row in report["results"]:
        if row["controller"] != "authored_ik_fsm_teacher_not_ppo":
            raise ValueError("Timestep evidence controller identity is incorrect")
        if row["passed"] != row["metrics"]["passed"]:
            raise ValueError("Timestep evidence verdict does not reconcile")
        if row["passed"] != all(row["metrics"]["gates"].values()):
            raise ValueError("Timestep evidence gates do not reconcile")
    successes = sum(row["passed"] for row in report["results"])
    if report["all_passed"] != (successes == 12):
        raise ValueError("Timestep evidence total does not reconcile")
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "outcomes": 12,
        "successes": successes,
        "all_passed": successes == 12,
        "physics_timesteps_s": [0.002, 0.001],
        "controller": "authored_ik_fsm_teacher_not_ppo",
    }


def validate_negative_controls(training: Path) -> dict[str, Any] | None:
    directory = training / "negative-controls"
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        return None
    summary = read_json(summary_path)
    if summary["source_hashes"] != pipeline.sources():
        raise ValueError("Negative-control source hashes are stale")

    expected_identities = {
        (case_id, controller)
        for case_id, controllers in NEGATIVE_CONTROLLERS.items()
        for controller in controllers
    }
    outcomes = []
    for case_id, controller in sorted(expected_identities):
        path = directory / f"{case_id}-{controller}.json"
        report = read_json(path)
        expected = expected_samples(NEGATIVE_SEED_BASE, 1)
        actual = [(row["config_index"], row["seed"]) for row in report["results"]]
        if report["case_id"] != case_id or report["controller"] != controller:
            raise ValueError(f"Negative-control identity mismatch: {path}")
        if report["source_hashes"] != pipeline.sources():
            raise ValueError(f"Negative-control source hashes mismatch: {path}")
        if report["model_sha256"] != arm.model_hash(case_id):
            raise ValueError(f"Negative-control model hash mismatch: {path}")
        if report["evaluation_seeds"] != pipeline.independent_eval_seeds(
            NEGATIVE_SEED_BASE, 1
        ):
            raise ValueError(f"Negative-control seed manifest mismatch: {path}")
        if len(actual) != 10 or set(actual) != set(expected):
            raise ValueError(f"Negative-control matrix is incomplete: {path}")
        for row in report["results"]:
            identity = (row["config_index"], row["seed"])
            if row["options"] != expected[identity]:
                raise ValueError(f"Negative-control perturbation mismatch: {path}")
            if row["passed"] != all(row["gates"].values()):
                raise ValueError(f"Negative-control verdict mismatch: {path}")
        unexpected = sum(row["passed"] for row in report["results"])
        if report["episodes"] != 10 or report["successes"] != unexpected:
            raise ValueError(f"Negative-control totals do not reconcile: {path}")
        outcomes.append(
            {
                "case_id": case_id,
                "controller": controller,
                "episodes": 10,
                "unexpected_successes": unexpected,
                "path": str(path.resolve()),
                "sha256": digest(path),
            }
        )

    summary_rows = summary.get("results", [])
    identities = {(row["case_id"], row["controller"]) for row in summary_rows}
    if len(summary_rows) != 13 or identities != expected_identities:
        raise ValueError("Negative-control summary does not contain 13 controllers")
    episodes = sum(row["episodes"] for row in outcomes)
    unexpected = sum(row["unexpected_successes"] for row in outcomes)
    if summary["episodes"] != episodes:
        raise ValueError("Negative-control episode total does not reconcile")
    if summary["unexpected_successes"] != unexpected:
        raise ValueError("Negative-control unexpected-success total does not reconcile")
    if summary["all_passed"] != (unexpected == 0):
        raise ValueError("Negative-control gate does not reconcile")
    return {
        "path": str(summary_path.resolve()),
        "sha256": digest(summary_path),
        "episodes": episodes,
        "unexpected_task_successes": unexpected,
        "all_passed": unexpected == 0,
        "seed_min": NEGATIVE_SEED_BASE,
        "seed_max": NEGATIVE_SEED_BASE
        + (len(pipeline.EVAL_CONFIGS) - 1) * pipeline.EVAL_CONFIG_SEED_STRIDE,
        "interpretation": (
            "The deliberately disabled controllers failed the tasks as expected; "
            "this is not task-success evidence."
        ),
        "results": outcomes,
    }


def validate_verification_logs(training: Path) -> dict[str, Any] | None:
    directory = training / "verification"
    existing = [
        directory / filename
        for filename, _, _ in VERIFICATION_LOGS.values()
        if (directory / filename).is_file()
    ]
    if not existing:
        return None
    if len(existing) != len(VERIFICATION_LOGS):
        raise FileNotFoundError("Verification logs are only partially present")

    results = {}
    for name, (filename, tests, pattern) in VERIFICATION_LOGS.items():
        path = directory / filename
        text = path.read_text()
        passed = re.search(pattern, text) is not None and (
            name == "rlx" or re.search(r"^OK$", text, re.MULTILINE) is not None
        )
        if not passed:
            raise ValueError(f"Verification log does not show a clean pass: {path}")
        results[name] = {
            "tests": tests,
            "passed": True,
            "path": str(path.resolve()),
            "sha256": digest(path),
        }
    return {
        "all_passed": all(row["passed"] for row in results.values()),
        "tests": sum(row["tests"] for row in results.values()),
        "hardware_verified": False,
        "note": (
            "The hardware test log covers software tests for hardware support; "
            "it is not physical hardware verification."
        ),
        "results": results,
    }


def validate_historical() -> dict[str, Any]:
    summary_path = HISTORICAL / "summary.json"
    index_path = HISTORICAL / "index.zh-CN.html"
    manifest_path = HISTORICAL / "video-evidence.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    if summary["episodes"] != 1800 or summary["successes"] != 1798:
        raise ValueError("Historical recheck no longer records 1798/1800")
    failed_videos = sum(not video["passed"] for video in manifest["videos"])
    if failed_videos != 7:
        raise ValueError("Historical video evidence no longer retains seven failures")
    if not index_path.is_file():
        raise FileNotFoundError(f"Historical video index is missing: {index_path}")
    return {
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": digest(summary_path),
        "index_path": str(index_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest(manifest_path),
        "episodes": 1800,
        "successes": 1798,
        "failed_videos_retained": failed_videos,
    }


def video_status(validation: Path, heldout_summary: dict[str, Any]) -> dict[str, Any]:
    index_path = validation / "index.zh-CN.html"
    manifest_path = validation / "video-evidence.json"
    generated = manifest_path.is_file()
    if generated and not index_path.is_file():
        raise FileNotFoundError(
            "Held-out video manifest exists but index.zh-CN.html is missing"
        )
    result = {
        "generated": generated,
        "index_path": str(index_path.resolve()),
        "manifest_path": str(manifest_path.resolve()),
    }
    if generated:
        manifest = read_json(manifest_path)
        if manifest["summary_sha256"] != heldout_summary["sha256"]:
            raise ValueError("Held-out video manifest is bound to another summary")
        if manifest["evidence_verification_passed"] is not True:
            raise ValueError("Held-out video evidence is not verified")
        if (
            manifest["all_experiment_episodes_pass"]
            != heldout_summary["all_episodes_pass"]
        ):
            raise ValueError("Held-out video manifest pass-all status differs")
        result.update(
            {
                "manifest_sha256": digest(manifest_path),
                "video_count": manifest["video_count"],
                "evidence_verification_passed": manifest[
                    "evidence_verification_passed"
                ],
            }
        )
    return result


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Training metrics are missing: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Training metrics are empty: {path}")
    for field in PLOT_FIELDS:
        if not any(row.get(field) is not None for row in rows):
            raise ValueError(f"Training metric {field!r} is absent from {path}")
    return rows


def plot_case(case_id: str, checkpoints: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for checkpoint in checkpoints:
        rows = load_metrics(Path(checkpoint["metrics_path"]))
        for axis, field in zip(axes.flat, PLOT_FIELDS):
            valid = [row for row in rows if row.get(field) is not None]
            axis.plot(
                [row["env_steps"] for row in valid],
                [row[field] for row in valid],
                label=f"seed {checkpoint['training_seed']}",
                linewidth=1,
            )
            axis.set_title(field)
            axis.set_xlabel("environment steps")
            axis.grid(alpha=0.2)
    axes.flat[0].legend()
    figure.suptitle(
        f"{case_id} | authored IK/FSM + 0.02 * PPO residual | simulation only"
    )
    figure.savefig(output / f"{case_id}-training.png", dpi=120)
    plt.close(figure)


def collect_checkpoints(training: Path, validation: Path) -> list[dict[str, Any]]:
    rows = []
    for case_id in arm.CASES:
        for seed in TRAINING_SEEDS:
            directory = training / case_id / f"seed-{seed}"
            metadata = read_json(directory / "training.json")
            pipeline.validate_training_evidence(directory, case_id, metadata)
            if metadata["train_seed"] != seed:
                raise ValueError(f"Training seed mismatch: {directory}")
            metrics_path = directory / "metrics.jsonl"
            load_metrics(metrics_path)
            primary = validate_evaluation(
                directory / "evaluation.json",
                case_id=case_id,
                metadata=metadata,
                seed_base=PRIMARY_SEED_BASE,
            )
            heldout = validate_evaluation(
                validation / case_id / f"seed-{seed}" / "evaluation.json",
                case_id=case_id,
                metadata=metadata,
                seed_base=VALIDATION_SEED_BASE,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "case_name_zh_CN": CASE_NAMES[case_id],
                    "training_seed": seed,
                    "directory": str(directory.resolve()),
                    "training_metadata_path": str(
                        (directory / "training.json").resolve()
                    ),
                    "training_metadata_sha256": digest(directory / "training.json"),
                    "checkpoint": str((directory / "ppo-residual.zip").resolve()),
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "model_sha256": metadata["model_sha256"],
                    "source_hashes": metadata["source_hashes"],
                    "actual_env_steps": metadata["actual_env_steps"],
                    "metrics_path": str(metrics_path.resolve()),
                    "metrics_sha256": digest(metrics_path),
                    "primary": primary,
                    "validation": heldout,
                }
            )
    if len(rows) != 18:
        raise ValueError("Expected exactly 18 training checkpoints")
    return rows


def result_status(
    primary: dict[str, Any],
    validation: dict[str, Any],
    regressions: dict[str, Any],
    timestep: dict[str, Any] | None,
    negative_controls: dict[str, Any] | None,
    verification: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    failures = []
    if not primary["all_episodes_pass"]:
        failures.append("primary_pass_all")
    if not validation["all_episodes_pass"]:
        failures.append("heldout_pass_all")
    if not regressions["all_passed"]:
        failures.append("known_failure_regressions")
    if timestep is not None and not timestep["all_passed"]:
        failures.append("timestep_convergence")
    if negative_controls is not None and not negative_controls["all_passed"]:
        failures.append("negative_controls")
    if verification is not None and not verification["all_passed"]:
        failures.append("verification_logs")
    return not failures, failures


def markdown_report(result: dict[str, Any], output: Path) -> str:
    primary = result["primary"]
    heldout = result["validation"]
    regressions = result["known_failure_regressions"]
    status = "成功" if result["success"] else "未成功"
    lines = [
        "# 严格机械臂修复 v3 结果",
        "",
        f"生成时间：{result['generated_at']}。",
        "",
        f"**最终结论：{status}。** 本结论要求主评估逐回合全过、独立留出评估逐回合全过、21 个已知失败回归全过；统计门槛单独报告，不能替代逐回合全过。",
        "",
        "## 验收总览",
        "",
        "| 证据 | 成功/总数 | 逐回合全过 | 统计门槛 |",
        "|---|---:|---|---|",
        f"| 主评估 seed {primary['seed_min']}..{primary['seed_max']} | {primary['successes']}/{primary['episodes']} | {'通过' if primary['all_episodes_pass'] else '未通过'} | {'18/18 通过' if primary['all_checkpoints_meet_statistical_gate'] else '未全部通过'} |",
        f"| 独立留出 seed {heldout['seed_min']}..{heldout['seed_max']} | {heldout['successes']}/{heldout['episodes']} | {'通过' if heldout['all_episodes_pass'] else '未通过'} | {'18/18 通过' if heldout['all_checkpoints_meet_statistical_gate'] else '未全部通过'} |",
        f"| 已知失败回归 | {regressions['successes']}/{regressions['episodes']} | {'通过' if regressions['all_passed'] else '未通过'} | 不适用 |",
    ]
    timestep = result["timestep_convergence"]
    if timestep is not None:
        lines.append(
            f"| 物理步长检查 | {timestep['successes']}/{timestep['outcomes']} | {'通过' if timestep['all_passed'] else '未通过'} | 不适用 |"
        )
    negative_controls = result["negative_controls"]
    if negative_controls is not None:
        lines.append(
            f"| 负对照 | 意外成功 {negative_controls['unexpected_task_successes']}/{negative_controls['episodes']} | "
            f"{'通过' if negative_controls['all_passed'] else '未通过'} | 不适用 |"
        )
    verification = result["verification"]
    if verification is not None:
        lines.append(
            f"| 软件测试日志 | {verification['tests']} 项通过 | "
            f"{'通过' if verification['all_passed'] else '未通过'} | 不适用 |"
        )
    lines += [
        "",
        f"18 个检查点的实际训练环境步数合计为 **{result['actual_training_env_steps']:,}**；该数字由各 `training.json` 的 `actual_env_steps` 求和，不使用预期值代替。",
        "",
        "## 18 个检查点结果矩阵",
        "",
        "| 任务 | 训练 seed | 主评估 | 主逐回合全过 | 主统计门槛 | 独立留出 | 留出逐回合全过 | 留出统计门槛 |",
        "|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in result["checkpoints"]:
        primary_row = row["primary"]
        validation_row = row["validation"]
        lines.append(
            f"| {row['case_name_zh_CN']} | {row['training_seed']} | "
            f"[{primary_row['successes']}/{primary_row['episodes']}]({relative(Path(primary_row['path']), output)}) | "
            f"{'通过' if primary_row['all_episodes_pass'] else '未通过'} | "
            f"{'通过' if primary_row['statistical_gate_passed'] else '未通过'} | "
            f"[{validation_row['successes']}/{validation_row['episodes']}]({relative(Path(validation_row['path']), output)}) | "
            f"{'通过' if validation_row['all_episodes_pass'] else '未通过'} | "
            f"{'通过' if validation_row['statistical_gate_passed'] else '未通过'} |"
        )

    lines += [
        "",
        "## 修复边界",
        "",
        "- `repair-source-audit.json` 将生产改动限定为 `teacher_targets` 中新增的 6 行：双臂共同搬运在抬升、运输、下降阶段使用同步五次平滑笛卡尔目标。",
        "- 抬升/运输/下降时长保持 3/6/3 秒；任务质量、奖励、物理、门槛和评估器均未改变。",
        "- 保留原有保守掉落评估器；本次修复没有通过放宽 `no_drop` 或其他安全门槛取得结果。",
        "- 控制器仍是人工 IK/定时状态机加 0.02×PPO 残差，不是纯 PPO，也没有重新训练 BC 的主张，更不声称 PPO 优于教师。",
        "- 这是 MuJoCo 仿真证据，不是硬件验证。双臂共同搬运为现有 17 g 总成，双臂交接为现有 5 g 课程；不声称双臂 20 g 目标已经完成。",
        "",
        "## 支持性检查",
        "",
    ]
    if timestep is not None:
        lines.append(
            f"- 教师控制器物理步长检查：{timestep['successes']}/{timestep['outcomes']}，覆盖 2 ms 与 1 ms；这是教师轨迹/仿真步长检查，不是 PPO 性能比较。"
        )
    if negative_controls is not None:
        lines.append(
            f"- 负对照：seed {negative_controls['seed_min']}..{negative_controls['seed_max']}，"
            f"{negative_controls['episodes']} 回合中意外任务成功 {negative_controls['unexpected_task_successes']} 次。"
            "这里的“通过”表示故意禁用的控制器按预期未完成任务，不表示负对照完成了任务。"
        )
    if verification is not None:
        lines.append(
            f"- 保存的软件测试日志共记录 {verification['tests']} 项通过：RLX 95、后端 21、控制 35、硬件支持软件测试 6、视频证据 8。"
            "其中硬件支持测试不构成实体硬件验证。"
        )
    if timestep is None and negative_controls is None and verification is None:
        lines.append("- 未提供可选支持性检查工件。")
    lines += [
        "",
        "## 训练曲线与原始日志",
        "",
        "每张图包含真实回合平均奖励、策略损失、价值损失、熵损失、近似 KL 和裁剪比例；三条线分别对应训练 seed 101、202、303。",
    ]
    for case_id in arm.CASES:
        case_rows = [row for row in result["checkpoints"] if row["case_id"] == case_id]
        logs = "、".join(
            f"[seed {row['training_seed']} 原始 metrics.jsonl]({relative(Path(row['metrics_path']), output)})"
            for row in case_rows
        )
        lines += [
            "",
            f"### {CASE_NAMES[case_id]}",
            "",
            f"![{case_id} 训练奖励与损失]({case_id}-training.png)",
            "",
            logs,
        ]

    lines += [
        "",
        "## 视频与历史证据",
        "",
    ]
    video = result["validation_videos"]
    video_link = relative(Path(video["index_path"]), output)
    if video["generated"]:
        lines.append(
            f"- [独立留出验证视频索引]({video_link})：视频清单存在，记录 {video['video_count']} 个已生成视频。"
        )
    else:
        lines.append(
            f"- [独立留出验证视频索引（预留路径）]({video_link})：尚无 `video-evidence.json`，因此本报告不声称视频已经生成。"
        )
    historical = result["historical"]
    lines.append(
        f"- [旧版 recheck-20260912-v2 视频索引]({relative(Path(historical['index_path']), output)})：历史结果保留为 1798/1800，并保留 7 个失败视频；它不计入 v3 成功判定。"
    )
    lines += [
        "",
        "## 可审计文件",
        "",
        f"- [主评估汇总]({relative(Path(primary['path']), output)})",
        f"- [独立留出汇总]({relative(Path(heldout['path']), output)})",
        f"- [21 个已知失败回归]({relative(Path(regressions['path']), output)})",
        f"- [修复源码审计]({relative(Path(result['repair']['path']), output)})",
    ]
    if timestep is not None:
        lines.append(
            f"- [12 个物理步长结果]({relative(Path(timestep['path']), output)})"
        )
    if negative_controls is not None:
        lines.append(
            f"- [130 回合负对照汇总]({relative(Path(negative_controls['path']), output)})"
        )
    if verification is not None:
        for name, row in verification["results"].items():
            lines.append(f"- [{name} 测试日志]({relative(Path(row['path']), output)})")
    lines.append("")
    if not result["success"]:
        lines += [
            "## 未通过项",
            "",
            "存在失败时不得报告成功。当前未通过项："
            + "、".join(f"`{failure}`" for failure in result["failed_requirements"])
            + "。",
            "",
        ]
    return "\n".join(lines)


def summarize(training: Path, validation: Path, output: Path) -> dict[str, Any]:
    checkpoints = collect_checkpoints(training, validation)
    primary = validate_primary_summary(training, checkpoints)
    heldout = validate_heldout_summary(validation, training, checkpoints)
    repair = validate_repair(training, checkpoints)
    regressions = validate_regressions(training, checkpoints)
    timestep = validate_timestep(training)
    negative_controls = validate_negative_controls(training)
    verification = validate_verification_logs(training)
    historical = validate_historical()
    videos = video_status(validation, heldout)
    success, failed_requirements = result_status(
        primary,
        heldout,
        regressions,
        timestep,
        negative_controls,
        verification,
    )
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "failed_requirements": failed_requirements,
        "simulation_only": True,
        "hardware_verified": False,
        "control_method": "authored IK/timed FSM + 0.02 * PPO residual",
        "pure_ppo": False,
        "bc_retrained": False,
        "ppo_improvement_over_teacher_claimed": False,
        "dual_arm_20g_goal_claimed": False,
        "payloads": {
            "arms_co_carry_actual_assembly_mass_kg": 0.017,
            "arms_handover_curriculum_mass_kg": 0.005,
        },
        "training_batch": str(training.resolve()),
        "validation_batch": str(validation.resolve()),
        "source_hashes": pipeline.sources(),
        "actual_training_env_steps": sum(
            row["actual_env_steps"] for row in checkpoints
        ),
        "checkpoint_count": len(checkpoints),
        "primary": primary,
        "validation": heldout,
        "known_failure_regressions": regressions,
        "timestep_convergence": timestep,
        "negative_controls": negative_controls,
        "verification": verification,
        "repair": repair,
        "historical": historical,
        "validation_videos": videos,
        "checkpoints": checkpoints,
    }

    output.mkdir(parents=True, exist_ok=True)
    for case_id in arm.CASES:
        plot_case(
            case_id,
            [row for row in checkpoints if row["case_id"] == case_id],
            output,
        )
    write_json(output / "results.json", result)
    (output / "RESULTS.zh-CN.md").write_text(markdown_report(result, output) + "\n")
    print(
        json.dumps(
            {
                "success": success,
                "primary": f"{primary['successes']}/{primary['episodes']}",
                "validation": f"{heldout['successes']}/{heldout['episodes']}",
                "known_failure_regressions": (
                    f"{regressions['successes']}/{regressions['episodes']}"
                ),
                "actual_training_env_steps": result["actual_training_env_steps"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize strict arm repair training and holdout evidence"
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=ROOT / "rlx/runs/arm/strict-repair-20260912-v3",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "rlx/runs/arm/strict-heldout-20260912-v3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/robot-arm-design/repair-v3",
    )
    args = parser.parse_args()
    result = summarize(
        args.training.resolve(), args.validation.resolve(), args.output.resolve()
    )
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
