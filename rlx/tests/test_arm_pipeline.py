from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import time

import numpy as np
import pytest

from rlx.environments import arm


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "examples/ppo_microduck_arm.py"
SPEC = importlib.util.spec_from_file_location("ppo_microduck_arm_pipeline_tests", PIPELINE_PATH)
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)


def write_training_evidence(output: Path, case_id: str) -> dict:
    output.mkdir(exist_ok=True)
    checkpoint = output / "ppo-residual.zip"
    checkpoint.write_bytes(b"checkpoint")
    archive = output / "sources"
    archive.mkdir()
    source_hashes = pipeline.sources()
    for name, source in pipeline.source_paths().items():
        shutil.copy2(source, archive / pipeline.SOURCE_ARCHIVE_NAMES[name])
    supplemental = pipeline.supplemental_provenance()
    for name, source in pipeline.supplemental_paths().items():
        if source.is_file():
            shutil.copy2(
                source, archive / pipeline.SUPPLEMENTAL_ARCHIVE_NAMES[name]
            )
    scene = output / "scene.xml"
    scene.write_text(arm.make_xml(case_id))
    return {
        "case_id": case_id,
        "source_hashes": source_hashes,
        "supplemental_provenance": supplemental,
        "model_sha256": arm.model_hash(case_id),
        "checkpoint_sha256": pipeline.digest(checkpoint),
    }


def test_source_hashes_preserve_two_field_integration_contract():
    assert set(pipeline.sources()) == {"environment", "pipeline"}
    assert set(pipeline.supplemental_provenance()) == {
        "experiment_spec",
        "project",
        "lockfile",
    }


@pytest.mark.parametrize("seed_base", [0, 49999])
def test_independent_evaluation_rejects_training_or_low_seed_ranges(seed_base):
    with pytest.raises(ValueError, match="seed_base must be >= 50000"):
        pipeline.independent_eval_seeds(seed_base, 10)


@pytest.mark.parametrize("seeds_per_config", [0, 101])
def test_independent_evaluation_rejects_invalid_config_sample_counts(
    seeds_per_config,
):
    with pytest.raises(ValueError, match="seeds_per_config"):
        pipeline.independent_eval_seeds(50000, seeds_per_config)


def test_evaluate_records_independent_seed_manifest(monkeypatch, tmp_path):
    def fake_episode(case_id, controller, model, seed, options):
        return {
            "passed": True,
            "case_id": case_id,
            "controller": controller,
            "seed": seed,
            "options": options,
            "steps": 10,
            "control_semantics": {"physics_assistance_count": 0},
        }, []

    monkeypatch.setattr(pipeline, "run_episode", fake_episode)
    report = pipeline.evaluate(
        "arm-reach-v1",
        "teacher",
        None,
        tmp_path / "evaluation.json",
        seeds_per_config=2,
        seed_base=51000,
    )
    assert report["episodes"] == 20
    assert report["evaluation_seeds"] == {
        "independent": True,
        "minimum_allowed_seed": 50000,
        "seed_base": 51000,
        "seed_min": 51000,
        "seed_max": 51901,
        "seeds_per_config": 2,
        "config_count": 10,
    }
    assert json.loads((tmp_path / "evaluation.json").read_text())["passed"] is True


def test_training_evidence_rejects_case_and_model_relabeling(tmp_path):
    metadata = write_training_evidence(tmp_path, "arm-pick-place-v1")
    with pytest.raises(ValueError, match="Policy case mismatch"):
        pipeline.validate_training_evidence(
            tmp_path, "arm-relocate-v1", metadata
        )

    metadata["case_id"] = "arm-relocate-v1"
    with pytest.raises(ValueError, match="Policy/model provenance mismatch"):
        pipeline.validate_training_evidence(
            tmp_path, "arm-relocate-v1", metadata
        )


def test_training_evidence_rejects_modified_archived_source(tmp_path):
    metadata = write_training_evidence(tmp_path, "arm-pick-place-v1")
    archived_pipeline = (
        tmp_path / "sources" / pipeline.SOURCE_ARCHIVE_NAMES["pipeline"]
    )
    archived_pipeline.write_text(archived_pipeline.read_text() + "\n")
    with pytest.raises(ValueError, match="Archived source provenance mismatch"):
        pipeline.validate_training_evidence(
            tmp_path, "arm-pick-place-v1", metadata
        )


def test_payload_metadata_distinguishes_actual_curriculum_and_goal_mass():
    handover = arm.ArmEnv("arms-handover-v1")
    co_carry = arm.ArmEnv("arms-co-carry-v1")
    try:
        handover_mass = pipeline.payload_metadata(handover)
        assert handover_mass["actual_transported_assembly_mass_kg"] == pytest.approx(
            0.005
        )
        assert handover_mass["design_goal_payload_mass_kg"] == pytest.approx(0.020)
        assert handover_mass["stage"] == "reduced_5g_curriculum_before_20g_goal"

        co_carry_mass = pipeline.payload_metadata(co_carry)
        assert co_carry_mass["modeled_object_body_mass_kg"] == pytest.approx(0.012)
        assert co_carry_mass["loose_payload_mass_kg"] == pytest.approx(0.005)
        assert co_carry_mass[
            "actual_transported_assembly_mass_kg"
        ] == pytest.approx(0.017)
        assert co_carry_mass["design_goal_payload_mass_kg"] == pytest.approx(0.020)
    finally:
        handover.close()
        co_carry.close()


def test_controller_semantics_separate_authored_control_from_physics_assistance():
    teacher = pipeline.controller_semantics("teacher", steps=600)
    assert teacher["authored_action_steps"] == 600
    assert teacher["physics_assistance"] is False
    assert teacher["learned_action"] is False

    residual = pipeline.controller_semantics("ppo_residual", steps=800)
    assert residual["authored_action_steps"] == 800
    assert residual["learned_action"] is True
    assert "not end-to-end planning" in residual["summary"]

    bc = pipeline.controller_semantics("bc")
    assert bc["authored_action_steps"] == 0
    assert bc["authored_scheduler_conditioning"] is True
    assert "diagnostic" in bc["summary"]


def test_trace_callback_records_real_completed_episode_rewards(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    callback = pipeline.TraceCallback(metrics_path)
    callback.model = SimpleNamespace(
        num_timesteps=1,
        start_time=time.time_ns(),
        logger=SimpleNamespace(name_to_value={}),
    )
    callback.num_timesteps = 1
    callback.locals = {
        "rewards": np.array([1.25]),
        "dones": np.array([False]),
        "infos": [{}],
    }
    assert callback._on_step() is True
    callback._on_rollout_start()

    callback.model.num_timesteps = 2
    callback.num_timesteps = 2
    callback.locals = {
        "rewards": np.array([2.75]),
        "dones": np.array([True]),
        "infos": [{
            "passed": True,
            "gates": {"no_drop": True},
            "physics_assistance_count": 0,
        }],
    }
    assert callback._on_step() is True
    callback._on_rollout_start()

    episode = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert episode == {
        "env_steps": 2,
        "return": 4.0,
        "length": 2,
        "success": True,
        "gates": {"no_drop": True},
        "physics_assistance_count": 0,
    }
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
    ]
    assert metrics[0]["rollout_episode_reward_mean"] is None
    assert metrics[0]["rollout_episode_count"] == 0
    assert metrics[1]["rollout_episode_reward_mean"] == pytest.approx(4.0)
    assert metrics[1]["rollout_episode_count"] == 1
