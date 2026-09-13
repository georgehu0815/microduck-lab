from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from gymnasium import spaces

from microduck_arm_v1 import learning


class FakeIntegratedArmEnv(gym.Env):
    def __init__(
        self,
        case="reach",
        mode="free",
        max_steps=250,
        payload_kg=0.01,
    ):
        self.case = case
        self.mode = mode
        self.max_steps = max_steps
        self.payload_kg = payload_kg
        self.observation_space = spaces.Box(-10, 10, (64,), np.float32)
        self.action_space = spaces.Box(-1, 1, (15,), np.float32)
        self.action_scales = np.asarray(
            [0.5] * 10 + [1.2, 1.25, 1.65, 1.4, 0.16], np.float32
        )
        self.action_offsets = np.asarray(
            [0.0] * 10 + [0.0, -0.45, 0.15, 0.0, -0.16], np.float32
        )
        self.step_count = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.full(64, (seed or 0) % 3, np.float32), {}

    def teacher_action(self):
        return np.full(15, 0.2, np.float32)

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        self.step_count += 1
        terminated = self.step_count >= 3
        success = bool((self.np_random_seed or 0) % 2 == 0)
        info = {
            "success": success if terminated else False,
            "failure_reason": None if success else "teacher_missed",
            "diagnostics": {"step": self.step_count},
        }
        observation = np.full(64, self.step_count, np.float32)
        return observation, 1.0, terminated, False, info

    @property
    def np_random_seed(self):
        state = self.np_random.bit_generator.state
        return int(state["state"]["state"] % 2)


@pytest.fixture(autouse=True)
def fake_env_module(monkeypatch):
    module = types.ModuleType("microduck_arm_v1.env")
    module.IntegratedArmEnv = FakeIntegratedArmEnv
    module.__file__ = __file__
    monkeypatch.setitem(sys.modules, "microduck_arm_v1.env", module)


def test_collect_demonstrations_filters_failures_and_keeps_diagnostics(
    tmp_path, monkeypatch
):
    outcomes = iter([True, False])

    class DeterministicFake(FakeIntegratedArmEnv):
        def reset(self, **kwargs):
            result = super().reset(**kwargs)
            self.success = next(outcomes)
            return result

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info["success"] = self.success if terminated else False
            info["failure_reason"] = None if self.success else "teacher_missed"
            return observation, reward, terminated, truncated, info

    monkeypatch.setattr(learning, "_env_class", lambda: DeterministicFake)
    dataset = learning.collect_demonstrations(
        "reach", "free", episodes=2, seed=10, out=tmp_path
    )

    with np.load(dataset) as payload:
        assert payload["observations"].shape == (3, 64)
        assert payload["actions"].shape == (3, 15)
    metadata = json.loads((tmp_path / "demonstrations.json").read_text())
    assert metadata["episodes_successful"] == 1
    assert metadata["episodes_selected"] == 1
    assert metadata["episodes"][1]["failure_reason"] == "teacher_missed"
    assert metadata["episodes"][1]["diagnostics"] == {"step": 3}
    assert set(metadata["provenance"]["files"]) == {
        "spec",
        "config",
        "environment",
        "learning",
        "model",
    }


def test_collect_demonstrations_refuses_empty_success_dataset(
    tmp_path, monkeypatch
):
    class FailingFake(FakeIntegratedArmEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            info["success"] = False
            info["failure_reason"] = "never_succeeded"
            return observation, reward, terminated, truncated, info

    monkeypatch.setattr(learning, "_env_class", lambda: FailingFake)
    with pytest.raises(RuntimeError, match="refusing to create a fake"):
        learning.collect_demonstrations(
            "reach", "free", episodes=1, seed=3, out=tmp_path
        )
    metadata = json.loads((tmp_path / "demonstrations.json").read_text())
    assert metadata["episodes_successful"] == 0
    assert not (tmp_path / "demonstrations.npz").exists()


def test_bc_checkpoint_logs_losses_and_exports_with_normalization(tmp_path):
    observations = np.random.default_rng(4).normal(size=(32, 64)).astype(np.float32)
    actions = np.tanh(observations @ np.ones((64, 15), np.float32) * 0.05)
    dataset = tmp_path / "data.npz"
    np.savez_compressed(dataset, observations=observations, actions=actions)

    checkpoint = learning.train_bc(dataset, tmp_path / "bc-run", epochs=2, seed=7)
    policy = learning.load_bc(checkpoint)
    exported = learning.export_policy(checkpoint, tmp_path / "bc.onnx")

    assert checkpoint.is_file()
    assert len((tmp_path / "bc-run" / "bc-loss.jsonl").read_text().splitlines()) == 2
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )
    got = session.run(None, {"observation": observations[:3]})[0]
    want = policy(torch.from_numpy(observations[:3])).detach().numpy()
    np.testing.assert_allclose(got, want, atol=1e-5, rtol=1e-5)
    metadata = json.loads(Path(str(exported) + ".json").read_text())
    assert metadata["normalization"]["embedded"] is True
    assert metadata["standalone_policy"] is True


def test_residual_wrapper_uses_physical_scales_and_reports_composition():
    env = FakeIntegratedArmEnv(max_steps=1)
    wrapped = learning.TeacherResidualEnv(env)
    wrapped.reset(seed=2)
    _, _, _, _, info = wrapped.step(np.ones(15, np.float32))

    expected = np.full(15, 0.2, np.float32)
    expected += 0.02 / env.action_scales
    np.testing.assert_allclose(env.last_action, expected)
    assert info["learning_wrapper"]["residual_scale_rad"] == pytest.approx(0.02)


def test_ppo_checkpoint_has_raw_eval_wrapper_metadata_and_residual_export(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(learning, "_env_class", lambda: FakeIntegratedArmEnv)
    checkpoint = learning.train_ppo(
        "reach",
        "fixture",
        tmp_path / "ppo-run",
        total_timesteps=16,
        seed=5,
        max_steps=3,
        residual=True,
    )
    metadata = json.loads((tmp_path / "ppo-run" / "ppo.json").read_text())

    assert checkpoint.is_file()
    assert metadata["fixture_mode_is_bench_aid_only"] is True
    assert metadata["learning_wrapper"]["standalone_policy"] is False
    assert metadata["action_schema"]["action_scales_rad"] == pytest.approx(
        FakeIntegratedArmEnv().action_scales
    )
    assert metadata["learning_wrapper"]["residual_scale_rad"] == pytest.approx(0.02)
    assert "model.predict" in metadata["raw_sb3_evaluation"]["call"]
    assert set(metadata["provenance"]["files"]) == {
        "spec",
        "config",
        "environment",
        "learning",
        "model",
    }
    training_rows = [
        json.loads(line)
        for line in (tmp_path / "ppo-run" / "ppo-training.jsonl")
        .read_text()
        .splitlines()
    ]
    assert training_rows[-1]["phase"] == "training_end"
    assert "train/loss" in training_rows[-1]["metrics"]
    raw = learning.evaluate_sb3_checkpoint(
        checkpoint, np.zeros((2, 64), np.float32)
    )
    assert raw.shape == (2, 15)

    exported = learning.export_policy(checkpoint, tmp_path / "residual.onnx")
    graph_metadata = {
        item.key: item.value for item in onnx.load(exported).metadata_props
    }
    embedded = json.loads(graph_metadata["microduck_arm_v1"])
    assert embedded["action_semantics"] == "normalized residual"
    assert embedded["standalone_policy"] is False
    assert embedded["learning_wrapper"]["residual_scale_rad"] == pytest.approx(0.02)

    evaluation = learning.evaluate_policy(
        checkpoint,
        "reach",
        "fixture",
        seeds=[90, 91],
        out=tmp_path / "ppo-evaluation.json",
        max_steps=3,
    )
    report = json.loads(evaluation.read_text())
    assert report["episodes"] == 2
    assert report["fixture_only"] is True
    assert report["free_mode_validated"] is False
    assert report["hardware_deployment_qualified"] is False
