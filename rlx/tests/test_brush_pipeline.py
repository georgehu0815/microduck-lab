import importlib.util
from pathlib import Path
import sys
import json
from types import SimpleNamespace
import hashlib

import gymnasium as gym
import numpy as np
import torch
import pytest
import onnx
from stable_baselines3.common.vec_env import DummyVecEnv


PATH = Path(__file__).resolve().parents[1] / "examples/ppo_microduck_brush.py"
SPEC = importlib.util.spec_from_file_location("brush_pipeline_test_module", PATH)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


class ToyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (93,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (15,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(93, np.float32), {}

    def step(self, action):
        return np.zeros(93, np.float32), float(action[4]), True, False, {}


def test_rejected_ppo_update_restores_actor_and_logs_real_steps():
    torch.set_num_threads(1)
    env = DummyVecEnv([ToyEnv])
    model = pipeline.AnchoredPPO(pipeline.drawing_feedback.DrawingFeedbackPolicy, env, seed=7,
                                n_steps=8, batch_size=8, n_epochs=1, learning_rate=.001)
    model.anchor_observations = torch.zeros(2, 93)
    with torch.no_grad():
        model.anchor_actions = model.policy.feedback_actor(model.anchor_observations).clone()
    model.anchor_limit = 0
    model.history = []
    before = {key: value.clone() for key, value in model.policy.feedback_actor.state_dict().items()}
    model.learn(16)
    assert model.num_timesteps == 16
    assert len(model.history) == 2
    assert not any(row["accepted_update"] for row in model.history)
    for key, value in before.items():
        torch.testing.assert_close(value, model.policy.feedback_actor.state_dict()[key], rtol=0, atol=0)
    env.close()


def test_policy_evaluation_never_queries_teacher(monkeypatch):
    def forbidden(self):
        raise AssertionError("teacher queried during learned-policy evaluation")
    monkeypatch.setattr(pipeline.brush.BrushEnv, "teacher_action", forbidden)
    result, records = pipeline.run_episode(lambda obs: np.zeros(15, np.float32), seed=901, control="open_jaw")
    assert result["steps"] > 0
    assert not result["drawing_assessment"]["passed"]
    assert records[0] == []


def test_validation_cases_are_distinct_from_training_randomizations():
    assert pipeline.EVAL_CASES["offset"] not in pipeline.TRAIN_CASES
    assert pipeline.EVAL_CASES["scale"] not in pipeline.TRAIN_CASES
    assert pipeline.EVAL_CASES["friction"] not in pipeline.TRAIN_CASES
    assert pipeline.EVAL_CASES["compliance"] not in pipeline.TRAIN_CASES
    assert len(pipeline.EVAL_CASES) == 10


def test_teacher_dataset_requires_matching_bytes_and_source(tmp_path):
    dataset = tmp_path / "teacher.npz"
    np.savez(dataset, observations=np.zeros((1, 93)), actions=np.zeros((1, 15)))
    with pytest.raises(ValueError, match="provenance"):
        pipeline.validate_dataset(dataset)
    metadata = {"sha256": pipeline.digest(dataset), "source_hashes": pipeline.source_hashes()}
    dataset.with_suffix(".json").write_text(json.dumps(metadata))
    assert pipeline.validate_dataset(dataset) == metadata
    dataset.write_bytes(b"changed dataset")
    with pytest.raises(ValueError, match="hash/source mismatch"):
        pipeline.validate_dataset(dataset)


def test_export_has_static_contract_widths_and_parity(tmp_path):
    env = DummyVecEnv([ToyEnv])
    model = pipeline.AnchoredPPO(pipeline.drawing_feedback.DrawingFeedbackPolicy, env, seed=71,
                                n_steps=8, batch_size=8, n_epochs=1)
    model.anchor_observations = torch.zeros(2, 93)
    output = tmp_path / "policy.onnx"
    assert pipeline.export(model, output, {"contract_version": "microduck-brush-v2"})["passed"]
    graph = onnx.load(output)
    assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 93
    assert graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 15
    env.close()


def test_source_guard_rejects_mid_operation_changes(monkeypatch):
    original = pipeline.source_hashes()
    pipeline.assert_sources(original)
    monkeypatch.setattr(pipeline, "source_hashes", lambda: {**original, "actor_sha256": "changed"})
    with pytest.raises(RuntimeError, match="changed during"):
        pipeline.assert_sources(original)


def test_evaluator_accepts_archived_pipeline_but_not_changed_physics(tmp_path):
    policy = tmp_path / "policy.onnx"
    checkpoint = policy.with_suffix(".zip")
    policy.write_bytes(b"policy")
    checkpoint.write_bytes(b"checkpoint")
    archive = tmp_path / "sources/ppo_microduck_brush.py"
    archive.parent.mkdir()
    archive.write_bytes(b"original training pipeline")
    for module in (pipeline.brush, pipeline.brush_reference, pipeline.drawing_feedback,
                   pipeline.drawing, pipeline.drawing_reference):
        source = Path(module.__file__)
        (archive.parent / source.name).write_bytes(source.read_bytes())
    sources = {**pipeline.source_hashes(), "pipeline_sha256": pipeline.digest(archive)}
    metadata = {"recipe": "drawing", "actuator": "xml", "contract_version": "microduck-brush-v2",
                "observation_dim": 93, "action_dim": 15, "max_episode_s": 120.0,
                "recipe_options": {}, "reward_weights": pipeline.brush.REWARD_WEIGHTS,
                "source_hashes": sources, "checkpoint_sha256": pipeline.digest(checkpoint)}
    sidecar = {**metadata, "checkpoint_sha256": pipeline.digest(checkpoint),
               "onnx": {"sha256": pipeline.digest(policy)}}
    policy.with_suffix(".zip.json").write_text(json.dumps(sidecar))
    assert pipeline.provenance_errors(policy, metadata) == []
    metadata["checkpoint_sha256"] = "different checkpoint"
    assert any("checkpoint hashes differ" in error for error in pipeline.provenance_errors(policy, metadata))
    metadata["checkpoint_sha256"] = pipeline.digest(checkpoint)
    archived_actor = archive.parent / "drawing_feedback.py"
    original_actor = archived_actor.read_bytes()
    archived_actor.write_bytes(b"modified archived actor")
    assert any("archive" in error and "drawing_feedback.py" in error for error in pipeline.provenance_errors(policy, metadata))
    archived_actor.unlink()
    assert any("archive" in error and "drawing_feedback.py" in error for error in pipeline.provenance_errors(policy, metadata))
    archived_actor.write_bytes(original_actor)
    metadata["source_hashes"]["environment_sha256"] = "different physics"
    assert any("environment/reference/actor" in error for error in pipeline.provenance_errors(policy, metadata))
    metadata["source_hashes"] = sources = {**pipeline.source_hashes(), "pipeline_sha256": pipeline.digest(archive)}
    archive.write_bytes(b"tampered archive")
    assert any("archive" in error for error in pipeline.provenance_errors(policy, metadata))
    archive.unlink()
    assert any("archive" in error for error in pipeline.provenance_errors(policy, metadata))


@pytest.mark.parametrize("replace_during_rollout", [False, True])
def test_evaluator_records_executed_bytes_not_replacement(tmp_path, monkeypatch, replace_during_rollout):
    policy = tmp_path / "policy.onnx"
    original = b"original executed policy"
    policy.write_bytes(original)
    metadata = {"source_hashes": pipeline.source_hashes()}

    def session_factory(model, **kwargs):
        assert model == original
        assert isinstance(model, bytes)
        return SimpleNamespace(get_modelmeta=lambda: SimpleNamespace(
            custom_metadata_map={"rlx_metadata": json.dumps(metadata)}))

    def episode(policy_callable, *, seed, options=None, control=None):
        if replace_during_rollout:
            policy.write_bytes(b"replacement")
        return {"seed": seed, "steps": 10, "return": 1.0,
                "drawing_assessment": {"passed": control is None, "coverage": 1.0, "precision": 1.0}}, []

    reports = []

    def save(path, report):
        policy.write_bytes(b"replacement during report finalization")
        reports.append(report)

    monkeypatch.setattr(pipeline.ort, "InferenceSession", session_factory)
    monkeypatch.setattr(pipeline, "provenance_errors", lambda path, metadata: [])
    monkeypatch.setattr(pipeline, "run_episode", episode)
    monkeypatch.setattr(pipeline, "write_json", save)
    pipeline.evaluate_command(SimpleNamespace(onnx=policy, out=tmp_path / "eval.json", seed=10001, episodes=4))
    report = reports[0]
    expected = hashlib.sha256(original).hexdigest()
    assert report["source_sha256"] == expected
    assert report["source_files_sha256"][str(policy)] == expected
    assert report["passed"] is not replace_during_rollout
    assert bool(report["provenance_errors"]) is replace_during_rollout
