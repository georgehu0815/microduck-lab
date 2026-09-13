from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnxruntime as ort
import pytest
import torch
from gymnasium import spaces

from microduck_arm_v1c import learning


class TinyEnv(gym.Env):
    def __init__(
        self,
        case="reach",
        mode="fixture",
        candidate="A",
        max_steps=3,
        payload_kg=0.01,
    ):
        self.case = case
        self.mode = mode
        self.candidate = candidate
        self.max_steps = max_steps
        self.payload_kg = payload_kg
        self.dt = 0.02
        self.observation_space = spaces.Box(-10, 10, (64,), np.float32)
        self.action_space = spaces.Box(-1, 1, (15,), np.float32)
        self.action_scales = np.asarray(
            [0.5] * 10 + [1.0, 1.25, 1.5, 2.0, 0.25], dtype=np.float32
        )
        self.action_offsets = np.linspace(-0.2, 0.2, 15, dtype=np.float32)
        self.steps = 0
        self.last_action = None

    def normalize_targets(self, targets):
        return np.clip(
            (np.asarray(targets) - self.action_offsets) / self.action_scales,
            -1,
            1,
        ).astype(np.float32)

    def denormalize_action(self, action):
        return self.action_offsets + self.action_scales * np.asarray(action)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.full(64, float((seed or 0) % 7), np.float32), {}

    def teacher_action(self):
        action = np.linspace(-0.4, 0.4, 15, dtype=np.float32)
        action[10:] += self.steps * 0.01
        return action

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        self.steps += 1
        terminated = self.steps >= self.max_steps
        info = {
            "success": terminated and self.last_action[10] > -0.5,
            "failure_reason": None if terminated else None,
            "diagnostics": {"steps": self.steps},
        }
        return (
            np.full(64, self.steps, np.float32),
            float(self.last_action[10]),
            terminated,
            False,
            info,
        )


@pytest.fixture(autouse=True)
def frozen_fake_env(monkeypatch):
    module = types.ModuleType("microduck_arm_v1c.env")
    module.IntegratedArmEnv = TinyEnv
    module.ENV_FROZEN = True
    module.__file__ = __file__
    monkeypatch.setitem(sys.modules, "microduck_arm_v1c.env", module)


def _dataset(tmp_path: Path) -> Path:
    observations = np.random.default_rng(2).normal(size=(12, 64)).astype(np.float32)
    actions = np.random.default_rng(3).uniform(-0.7, 0.7, (12, 15)).astype(
        np.float32
    )
    episode_ids = np.repeat(np.arange(4), 3)
    path = tmp_path / "data.npz"
    np.savez_compressed(
        path,
        observations=observations,
        teacher_actions=actions,
        episode_ids=episode_ids,
    )
    return path


def test_actor_and_residual_masks_keep_legs_on_teacher():
    teacher = np.linspace(-0.5, 0.5, 15, dtype=np.float32)
    absolute = learning.compose_arm_action(
        teacher, np.full(5, 0.75, dtype=np.float32)
    )
    np.testing.assert_array_equal(absolute[:10], teacher[:10])
    np.testing.assert_array_equal(absolute[10:], np.full(5, 0.75))

    env = TinyEnv(max_steps=1)
    wrapped = learning.TeacherResidualEnv(env)
    assert wrapped.action_space.shape == (5,)
    wrapped.reset(seed=4)
    teacher = env.teacher_action()
    raw = np.ones(5, dtype=np.float32)
    _, _, _, _, info = wrapped.step(raw)
    expected = teacher.copy()
    expected[10:] += 0.02 / env.action_scales[10:]
    np.testing.assert_allclose(env.last_action[:10], expected[:10])
    np.testing.assert_allclose(env.last_action[10:], expected[10:], atol=1e-7)
    assert info["learning_wrapper"]["action_mask"] == pytest.approx(
        learning.ACTION_MASK
    )


def test_episode_split_has_no_frame_leakage_and_is_recorded(tmp_path):
    dataset = _dataset(tmp_path)
    checkpoint = learning.train_bc(
        dataset,
        tmp_path / "run",
        epochs=1,
        seed=7,
        validation_fraction=0.25,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    split = payload["split_provenance"]
    assert split["unit"] == "whole_episode"
    assert split["frame_leakage"] is False
    assert set(split["train_episode_ids"]).isdisjoint(split["validation_episode_ids"])
    assert split["train_samples"] + split["validation_samples"] == 12


def test_weighted_arm_loss_ignores_legs_and_applies_weights():
    target = torch.zeros((1, 15))
    prediction = target.clone()
    prediction[:, :10] = 1000
    assert learning.weighted_arm_mse(prediction, target).item() == 0.0

    prediction[:, 10:] = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    uniform = learning.weighted_arm_mse(prediction, target).item()
    weighted = learning.weighted_arm_mse(
        prediction, target, np.asarray([5, 1, 1, 1, 1], np.float32)
    ).item()
    assert weighted > uniform


def test_bc_export_is_five_arm_outputs_with_embedded_normalization(tmp_path):
    checkpoint = learning.train_bc(
        _dataset(tmp_path), tmp_path / "run", epochs=2, seed=11
    )
    policy = learning.load_bc(checkpoint)
    exported = learning.export_policy(checkpoint, tmp_path / "policy.onnx")
    observations = np.random.default_rng(9).normal(size=(3, 64)).astype(np.float32)
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {"observation": observations})[0]
    with torch.no_grad():
        expected = policy(torch.from_numpy(observations)).numpy()
    assert actual.shape == (3, 5)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    metadata = json.loads(Path(str(exported) + ".json").read_text())
    assert metadata["explicit_adapter_required"] is True
    assert metadata["direct_legacy_or_hardware_compatible"] is False
    assert metadata["output"]["shape"] == ["batch", 5]


def test_dagger_labels_on_policy_states_without_goal_teleport(tmp_path):
    checkpoint = learning.train_bc(
        _dataset(tmp_path), tmp_path / "run", epochs=1, seed=13
    )
    dataset = learning.collect_dagger(
        checkpoint,
        tmp_path / "dagger",
        episodes=2,
        seed=20,
        max_steps=2,
    )
    with np.load(dataset) as payload:
        np.testing.assert_array_equal(
            payload["behavior_actions"][:, :10], payload["teacher_actions"][:, :10]
        )
        assert not np.allclose(
            payload["behavior_actions"][:, 10:], payload["teacher_actions"][:, 10:]
        )
    metadata = json.loads((tmp_path / "dagger" / "demonstrations.json").read_text())
    assert metadata["collection"] == "dagger_on_policy_states"
    assert metadata["goal_teleportation"] is False
    assert metadata["teacher_labels"].endswith("before step")


def test_ppo_logs_required_metrics_exports_arm_only_and_counts_failures(
    tmp_path, monkeypatch
):
    checkpoint = learning.train_ppo(
        tmp_path / "ppo",
        total_timesteps=16,
        seed=17,
        max_steps=2,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "ppo" / "ppo-training.jsonl").read_text().splitlines()
    ]
    final = rows[-1]
    assert final["phase"] == "training_end"
    assert {
        "raw_reward_sum",
        "raw_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    } <= set(final)
    metadata = json.loads((tmp_path / "ppo" / "ppo.json").read_text())
    assert metadata["public_action_dim"] == 15
    assert metadata["policy_action_dim"] == 5
    assert metadata["leg_residuals_applied"] is False

    exported = learning.export_policy(checkpoint, tmp_path / "ppo.onnx")
    session = ort.InferenceSession(
        str(exported), providers=["CPUExecutionProvider"]
    )
    assert session.run(
        None, {"observation": np.zeros((2, 64), np.float32)}
    )[0].shape == (2, 5)

    class FailingTiny(TinyEnv):
        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if terminated:
                info["success"] = False
                info["failure_reason"] = "forced_failure"
            return observation, reward, terminated, truncated, info

    monkeypatch.setattr(learning, "_env_class", lambda: FailingTiny)
    report_path = learning.evaluate_policy(
        checkpoint,
        seeds=[31, 32],
        out=tmp_path / "evaluation.json",
        max_steps=2,
    )
    report = json.loads(report_path.read_text())
    assert report["full_physics_rollout"] is True
    assert report["failures"] == 2
    assert report["failure_counts"] == {"forced_failure": 2}


def test_teacher_evaluation_uses_full_physics_and_counts_failures(tmp_path):
    report_path = learning.evaluate_teacher(
        seeds=[31, 32],
        out=tmp_path / "teacher-evaluation.json",
        max_steps=2,
    )
    report = json.loads(report_path.read_text())
    assert report["policy"] == "deterministic env.teacher_action"
    assert report["full_physics_rollout"] is True
    assert report["successes"] == 2
    assert report["failures"] == 0


def test_training_refuses_unfrozen_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules["microduck_arm_v1c.env"], "ENV_FROZEN", False)
    with pytest.raises(RuntimeError, match="ENV_FROZEN"):
        learning.train_bc(_dataset(tmp_path), tmp_path / "run", epochs=1, seed=1)


def test_provenance_includes_locomotion_contract_and_shipped_policies():
    provenance = learning.source_provenance()
    files = provenance["files"]
    assert "v1c_locomotion" in files
    assert "microduck_local_contract_dependency" in files
    assert files["shipped_policy_alpha_stand"]["sha256"] == (
        "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b"
    )
    assert files["shipped_policy_alpha_walking"]["sha256"] == (
        "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
    )


def test_aggregate_datasets_remaps_episodes_and_preserves_sources(tmp_path):
    first = learning.collect_demonstrations(
        tmp_path / "first", episodes=2, seed=10, max_steps=2
    )
    second = learning.collect_demonstrations(
        tmp_path / "second", episodes=2, seed=20, max_steps=2
    )
    merged = learning.aggregate_datasets([first, second], tmp_path / "merged")

    with np.load(merged) as dataset:
        assert dataset["observations"].shape == (8, 64)
        assert dataset["teacher_actions"].shape == (8, 15)
        assert dataset["behavior_actions"].shape == (8, 15)
        np.testing.assert_array_equal(
            np.unique(dataset["episode_ids"]), np.arange(4)
        )
    metadata = json.loads(merged.with_suffix(".json").read_text())
    assert metadata["episode_ids_remapped"] is True
    assert len(metadata["source_datasets"]) == 2
    assert {
        source["dataset_sha256"] for source in metadata["source_datasets"]
    } == {learning._sha256(first), learning._sha256(second)}
    assert metadata["source_datasets"][0]["episode_id_map"] == {"0": 0, "1": 1}
    assert metadata["source_datasets"][1]["episode_id_map"] == {"0": 2, "1": 3}


def test_bounded_runner_orders_stages_and_records_heldout_provenance(
    tmp_path, monkeypatch
):
    calls = []

    def artifact(name, suffix):
        path = tmp_path / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return path

    def collect(out, episodes, seed, **kwargs):
        calls.append(("collect", episodes, seed))
        return artifact(f"collect-{seed}", ".npz")

    def train_bc(dataset, out, epochs, seed, *args):
        calls.append(("bc", Path(dataset).name, epochs, seed))
        return artifact(f"bc-{len(calls)}", ".pt")

    def collect_dagger(checkpoint, out, episodes, seed, **kwargs):
        calls.append(("dagger", Path(checkpoint).name, episodes, seed))
        return artifact("dagger", ".npz")

    def aggregate(inputs, out):
        calls.append(("aggregate", [Path(path).name for path in inputs]))
        return artifact("merged", ".npz")

    def train_ppo(out, total_timesteps, seed, **kwargs):
        calls.append(("ppo", total_timesteps, seed))
        return artifact("ppo", ".zip")

    def export(checkpoint, out):
        calls.append(("export", Path(checkpoint).name))
        path = artifact(f"export-{Path(checkpoint).stem}", ".onnx")
        Path(str(path) + ".json").write_text(
            json.dumps({"parity_max_abs_error": 0.0})
        )
        return path

    def evaluate(checkpoint, seeds, out, **kwargs):
        calls.append(("evaluate", Path(checkpoint).name, list(seeds)))
        path = artifact(f"evaluate-{Path(checkpoint).stem}", ".json")
        path.write_text(
            json.dumps(
                {
                    "successes": 1,
                    "failures": 0,
                    "failure_counts": {},
                    "records": [],
                }
            )
        )
        return path

    def evaluate_teacher(seeds, out, **kwargs):
        calls.append(("evaluate_teacher", list(seeds)))
        path = artifact("evaluate-teacher", ".json")
        path.write_text(
            json.dumps(
                {
                    "successes": 1,
                    "failures": 0,
                    "failure_counts": {},
                    "records": [],
                }
            )
        )
        return path

    monkeypatch.setattr(learning, "collect_demonstrations", collect)
    monkeypatch.setattr(learning, "train_bc", train_bc)
    monkeypatch.setattr(learning, "collect_dagger", collect_dagger)
    monkeypatch.setattr(learning, "aggregate_datasets", aggregate)
    monkeypatch.setattr(learning, "train_ppo", train_ppo)
    monkeypatch.setattr(learning, "export_policy", export)
    monkeypatch.setattr(learning, "evaluate_policy", evaluate)
    monkeypatch.setattr(learning, "evaluate_teacher", evaluate_teacher)
    monkeypatch.setattr(learning, "_last_jsonl", lambda path: {"loss": 0.1})

    manifest_path = learning.run_bounded_pipeline(
        tmp_path / "pipeline",
        demonstration_episodes=2,
        dagger_episodes=2,
        bc_epochs=1,
        merged_bc_epochs=2,
        ppo_timesteps=16,
        demonstration_seed=10,
        dagger_seed=20,
        ppo_seed=30,
        heldout_seeds=[40, 41],
        max_steps=5,
    )
    assert [call[0] for call in calls] == [
        "collect",
        "bc",
        "dagger",
        "aggregate",
        "bc",
        "ppo",
        "export",
        "export",
        "export",
        "evaluate_teacher",
        "evaluate",
        "evaluate",
        "evaluate",
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["seeds"]["collection"] == [10, 11, 20, 21]
    assert manifest["seeds"]["heldout"] == [40, 41]
    assert manifest["seeds"]["heldout_disjoint_from_collection"] is True
    assert set(manifest["artifacts"]) == {
        "demonstrations",
        "initial_bc",
        "dagger",
        "merged_demonstrations",
        "merged_bc",
        "ppo",
        "initial_bc_onnx",
        "dagger_bc_onnx",
        "ppo_onnx",
        "teacher_evaluation",
        "initial_bc_evaluation",
        "dagger_bc_evaluation",
        "ppo_evaluation",
    }
    summary = json.loads((tmp_path / "pipeline" / "training-summary.json").read_text())
    assert summary["honesty"]["bc_passed"] is True
    assert summary["onnx_parity"] == {"bc": 0.0, "dagger": 0.0, "ppo": 0.0}


def test_bounded_runner_refuses_unfrozen_or_overlapping_heldout(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sys.modules["microduck_arm_v1c.env"], "ENV_FROZEN", False)
    with pytest.raises(RuntimeError, match="ENV_FROZEN"):
        learning.run_bounded_pipeline(tmp_path / "pipeline")

    monkeypatch.setattr(sys.modules["microduck_arm_v1c.env"], "ENV_FROZEN", True)
    with pytest.raises(ValueError, match="overlap"):
        learning.run_bounded_pipeline(
            tmp_path / "pipeline",
            demonstration_episodes=2,
            demonstration_seed=10,
            heldout_seeds=[11],
        )


def test_collection_aborts_without_artifact_when_source_drifts(
    tmp_path, monkeypatch
):
    snapshots = iter(
        [
            {"files": {"env": {"path": "env.py", "sha256": "before"}}},
            {"files": {"env": {"path": "env.py", "sha256": "after"}}},
        ]
    )
    monkeypatch.setattr(learning, "source_provenance", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="changed during teacher"):
        learning.collect_demonstrations(
            tmp_path / "drifted-collection",
            episodes=2,
            seed=1,
            max_steps=1,
        )
    assert not (tmp_path / "drifted-collection" / "demonstrations.npz").exists()


def test_training_aborts_without_checkpoint_when_source_drifts(
    tmp_path, monkeypatch
):
    snapshots = iter(
        [
            {"files": {"learning": {"path": "learning.py", "sha256": "before"}}},
            {"files": {"learning": {"path": "learning.py", "sha256": "after"}}},
        ]
    )
    monkeypatch.setattr(learning, "source_provenance", lambda: next(snapshots))
    with pytest.raises(RuntimeError, match="changed during behavior cloning"):
        learning.train_bc(
            _dataset(tmp_path),
            tmp_path / "drifted-training",
            epochs=1,
            seed=1,
        )
    assert not (tmp_path / "drifted-training" / "bc.pt").exists()


def test_evaluation_aborts_without_report_when_source_drifts(
    tmp_path, monkeypatch
):
    snapshots = iter(
        [
            {"files": {"env": {"path": "env.py", "sha256": "before"}}},
            {"files": {"env": {"path": "env.py", "sha256": "after"}}},
        ]
    )
    monkeypatch.setattr(learning, "source_provenance", lambda: next(snapshots))
    report = tmp_path / "drifted-evaluation.json"
    with pytest.raises(RuntimeError, match="changed during teacher evaluation"):
        learning.evaluate_teacher([1], report, max_steps=1)
    assert not report.exists()


def test_collection_metadata_stores_preoperation_provenance(tmp_path):
    dataset = learning.collect_demonstrations(
        tmp_path / "stable-collection",
        episodes=2,
        seed=1,
        max_steps=1,
    )
    metadata = json.loads(dataset.with_suffix(".json").read_text())
    assert metadata["source_provenance_captured"] == "before_collection"
    learning._validate_provenance(metadata["source_provenance"])


def test_bounded_runner_preserves_existing_attempt_directory(tmp_path):
    root = tmp_path / "attempt-1"
    root.mkdir()
    marker = root / "keep.txt"
    marker.write_text("existing")
    with pytest.raises(FileExistsError, match="new attempt directory"):
        learning.run_bounded_pipeline(root)
    assert marker.read_text() == "existing"
