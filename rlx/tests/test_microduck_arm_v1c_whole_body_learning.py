from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
import pytest
from gymnasium import spaces

from microduck_arm_v1c import learning
from microduck_arm_v1c import whole_body_learning as whole_body


REAL_DYNAMIC_FEATURES = whole_body._dynamic_features
REAL_SOURCE_PROVENANCE = whole_body.source_provenance


class TinyEnv(gym.Env):
    def __init__(
        self,
        case="reach",
        mode="free",
        candidate="A",
        max_steps=3,
        payload_kg=0.01,
        render_mode=None,
    ):
        self.case = case
        self.mode = mode
        self.candidate = candidate
        self.max_steps = max_steps
        self.payload_kg = payload_kg
        self.dt = 0.02
        self.render_mode = render_mode
        self.data = types.SimpleNamespace(time=0.0)
        self.observation_space = spaces.Box(-10, 10, (64,), np.float32)
        self.action_space = spaces.Box(-1, 1, (15,), np.float32)
        self.action_scales = np.asarray(
            [0.5] * 10 + [1.0, 1.25, 1.5, 2.0, 0.25],
            dtype=np.float32,
        )
        self.action_offsets = np.linspace(-0.2, 0.2, 15, dtype=np.float32)
        self.steps = 0
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.last_action = None
        value = float((seed or 0) % 7)
        return np.full(64, value, np.float32), {}

    def teacher_action(self):
        return np.linspace(-0.4, 0.4, 15, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (15,) or not np.isfinite(action).all():
            raise ValueError("invalid action")
        self.last_action = action
        self.steps += 1
        self.data.time = self.steps * self.dt
        terminated = self.steps >= self.max_steps
        success = bool(terminated and np.max(np.abs(action)) <= 1)
        info = {
            "success": success,
            "failure_reason": None if success else "timeout",
            "diagnostics": {"steps": self.steps},
        }
        return (
            np.full(64, self.steps, np.float32),
            float(-np.square(action).mean()),
            terminated,
            False,
            info,
        )

    def render(self):
        return np.zeros(
            (whole_body.RENDER_HEIGHT, whole_body.RENDER_WIDTH, 3),
            dtype=np.uint8,
        )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    module = types.ModuleType("microduck_arm_v1c.env")
    module.IntegratedArmEnv = TinyEnv
    module.ENV_FROZEN = True
    module.__file__ = __file__
    monkeypatch.setitem(sys.modules, "microduck_arm_v1c.env", module)

    path = Path(__file__).resolve()
    provenance = {
        "files": {
            "whole_body_test_source": {
                "path": str(path),
                "sha256": learning._sha256(path),
            }
        },
        "required_families": ["test"],
        "invalidation_rule": "test source must remain unchanged",
    }
    monkeypatch.setattr(whole_body, "source_provenance", lambda: provenance)
    monkeypatch.setattr(
        whole_body,
        "_dynamic_features",
        lambda env: np.asarray([0.1, -0.2, 0.3, 1.0, 0.0], dtype=np.float32),
    )


def test_zero_residual_exactly_reproduces_teacher():
    teacher = np.linspace(-0.8, 0.8, 15, dtype=np.float32)
    scales = np.linspace(0.25, 1.75, 15, dtype=np.float32)

    actual = whole_body.compose_residual_action(
        teacher,
        np.zeros(15, dtype=np.float32),
        scales,
    )

    np.testing.assert_array_equal(actual, teacher)


def test_separate_leg_and_arm_caps_are_applied_in_radians_with_clipping():
    teacher = np.zeros(15, dtype=np.float32)
    scales = np.asarray([0.5] * 10 + [1.0] * 5, dtype=np.float32)

    actual = whole_body.compose_residual_action(
        teacher,
        np.full(15, 2.0, dtype=np.float32),
        scales,
        leg_residual_rad=0.01,
        arm_residual_rad=0.03,
    )

    np.testing.assert_allclose(actual[:10], 0.02)
    np.testing.assert_allclose(actual[10:], 0.03)
    assert actual.dtype == np.float32

    saturated = whole_body.compose_residual_action(
        np.full(15, 0.999, dtype=np.float32),
        np.ones(15, dtype=np.float32),
        scales,
        leg_residual_rad=0.1,
        arm_residual_rad=0.1,
    )
    np.testing.assert_array_equal(saturated, np.ones(15, dtype=np.float32))


@pytest.mark.parametrize(
    ("teacher", "residual", "scales", "message"),
    [
        (np.zeros(14), np.zeros(15), np.ones(15), "teacher_action"),
        (np.zeros(15), np.full(15, np.nan), np.ones(15), "residual_action"),
        (np.zeros(15), np.zeros(15), np.zeros(15), "action_scales"),
    ],
)
def test_residual_composition_rejects_invalid_inputs(
    teacher,
    residual,
    scales,
    message,
):
    with pytest.raises(ValueError, match=message):
        whole_body.compose_residual_action(teacher, residual, scales)


def test_dynamic_observation_uses_local_velocity_and_force_contacts(monkeypatch):
    class Item:
        def __init__(self, item_id, name=None):
            self.id = item_id
            self.name = name

    class Model:
        geom_bodyid = np.asarray([10, 0, 11, 0])

        def body(self, name):
            return Item({"ankle_left": 10, "ankle_right": 11}[name])

        def geom(self, geom_id):
            return Item(
                geom_id,
                ["ankle_left_collision", "floor", "ankle_right_collision", "object"][
                    geom_id
                ],
            )

    class Contact:
        def __init__(self, first, second):
            self.geom1 = first
            self.geom2 = second

    data = types.SimpleNamespace(
        ncon=2,
        contact=[Contact(0, 1), Contact(2, 3)],
    )
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            model=Model(),
            data=data,
            trunk_id=4,
        )
    )
    calls = {}

    def object_velocity(model, actual_data, object_type, object_id, result, local):
        calls["velocity"] = (object_type, object_id, local)
        result[:] = [4.0, 5.0, 6.0, 1.0, -2.0, 3.0]

    def contact_force(model, actual_data, contact_index, result):
        result[:] = 0
        result[0] = 0.02

    monkeypatch.setattr(mujoco, "mj_objectVelocity", object_velocity)
    monkeypatch.setattr(mujoco, "mj_contactForce", contact_force)

    actual = REAL_DYNAMIC_FEATURES(env)

    np.testing.assert_array_equal(actual, [1.0, -2.0, 3.0, 1.0, 0.0])
    assert calls["velocity"] == (mujoco.mjtObj.mjOBJ_BODY, 4, 1)


def test_source_provenance_includes_every_v1c_python_source(monkeypatch):
    monkeypatch.setattr(
        learning,
        "source_provenance",
        lambda: {
            "files": {},
            "required_families": [],
            "invalidation_rule": "base",
        },
    )

    provenance = REAL_SOURCE_PROVENANCE()
    recorded = {Path(record["path"]) for record in provenance["files"].values()}
    source_root = Path(whole_body.__file__).resolve().parent
    expected = set(source_root.rglob("*.py"))

    assert expected <= recorded
    assert source_root / "gait.py" in recorded
    assert "teacher and gait logic" in provenance["invalidation_rule"]


def test_wrapper_exposes_69_observations_fifteen_actions_and_no_waist():
    env = TinyEnv(max_steps=1)
    wrapped = whole_body.WholeBodyResidualEnv(
        env,
        leg_residual_rad=0.01,
        arm_residual_rad=0.03,
    )
    assert wrapped.action_space.shape == (15,)
    observation, _ = wrapped.reset(seed=3)
    assert wrapped.observation_space.shape == (69,)
    assert observation.shape == (69,)
    np.testing.assert_array_equal(
        observation[-5:],
        np.asarray([0.1, -0.2, 0.3, 1.0, 0.0], dtype=np.float32),
    )

    observation, _, terminated, _, info = wrapped.step(
        np.zeros(15, dtype=np.float32)
    )

    assert terminated
    assert observation.shape == (69,)
    np.testing.assert_array_equal(env.last_action, env.teacher_action())
    record = info["learning_wrapper"]
    assert record["control_profile"] == whole_body.CONTROL_PROFILE
    assert record["waist_action_indices"].size == 0
    np.testing.assert_allclose(record["residual_caps_rad"][:10], 0.01)
    np.testing.assert_allclose(record["residual_caps_rad"][10:], 0.03)


def test_short_training_export_and_same_seed_evaluation(tmp_path, monkeypatch):
    checkpoint = whole_body.train_ppo(
        tmp_path / "run",
        total_timesteps=16,
        seed=17,
        max_steps=2,
        leg_residual_rad=0.01,
        arm_residual_rad=0.03,
    )
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["artifact_version"] == 2
    assert metadata["control_profile"] == "whole_body_residual_obs69_v2"
    assert metadata["policy_action_dim"] == 15
    assert metadata["leg_residual_cap_rad"] == pytest.approx(0.01)
    assert metadata["arm_residual_cap_rad"] == pytest.approx(0.03)
    assert metadata["waist_joint_present"] is False
    assert metadata["observation_dim"] == 69
    assert metadata["deployment_contract"] == "obs69-action15:whole-body-residual-v2"
    assert metadata["observation_contract"] == {
        "base_observation_indices": [0, 64],
        "foot_contact_force_threshold_n": 0.01,
        "hardware_foot_sensors_claimed": False,
        "left_foot_force_contact_index": 67,
        "right_foot_force_contact_index": 68,
        "simulator_state_only": True,
        "trunk_local_linear_velocity_indices": [64, 67],
    }
    assert metadata["legacy_61x14_contract_compatible"] is False
    assert metadata["source_provenance_captured"] == "before_training"

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "whole-body-residual-ppo-training.jsonl")
        .read_text()
        .splitlines()
    ]
    assert rows[-1]["phase"] == "training_end"
    assert {
        "raw_reward_sum",
        "raw_loss",
        "value_loss",
        "entropy_loss",
        "approx_kl",
        "clip_fraction",
    } <= set(rows[-1])

    exported = whole_body.export_policy(checkpoint, tmp_path / "policy.onnx")
    session = ort.InferenceSession(
        str(exported),
        providers=["CPUExecutionProvider"],
    )
    assert session.run(
        None,
        {"observation": np.zeros((3, 69), dtype=np.float32)},
    )[0].shape == (3, 15)
    export_metadata = json.loads(Path(str(exported) + ".json").read_text())
    assert export_metadata["output"]["shape"] == ["batch", 15]
    assert export_metadata["input"]["shape"] == ["batch", 69]
    assert export_metadata["parity_max_abs_error"] <= 1e-5
    assert export_metadata["teacher_adapter_required"] is True
    assert export_metadata["legacy_61x14_contract_compatible"] is False

    report_path = whole_body.evaluate_policy(
        checkpoint,
        seeds=[31, 32],
        out=tmp_path / "evaluation.json",
        max_steps=2,
    )
    report = json.loads(report_path.read_text())
    assert report["heldout_seeds"] == [31, 32]
    assert report["same_seeds_for_teacher_and_ppo"] is True
    assert report["teacher"]["successes"] == 2
    assert report["ppo"]["successes"] == 2
    assert report["teacher"]["failures"] == 0
    assert report["ppo"]["failures"] == 0
    assert report["observation_dim"] == 69
    assert report["simulator_state_only"] is True
    assert report["hardware_foot_sensors_claimed"] is False
    assert report["legacy_61x14_contract_compatible"] is False

    rendered_frames = []

    class FakeWriter:
        def __init__(self, path):
            self.path = Path(path)

        def append_data(self, frame):
            rendered_frames.append(np.asarray(frame))

        def close(self):
            self.path.write_bytes(b"whole-body-test-video")

    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda path, **kwargs: FakeWriter(path),
    )
    rendered_report_path = whole_body.evaluate_onnx_render(
        exported,
        seeds=[41],
        out=tmp_path / "rendered",
        max_steps=2,
    )
    rendered_report = json.loads(rendered_report_path.read_text())
    assert rendered_report["same_seeds_for_teacher_and_onnx"] is True
    assert rendered_report["failure_episodes_saved"] is True
    assert rendered_report["complete_step_telemetry_saved"] is True
    assert rendered_report["render"] == {
        "fps": 25,
        "height": 480,
        "labels": [
            "SIMULATION ONLY | teacher",
            "SIMULATION ONLY | teacher+wholebodyresidual",
        ],
        "width": 640,
    }
    assert rendered_report["source_provenance_unchanged"] is True
    assert rendered_report["source_provenance_before"] == rendered_report[
        "source_provenance_after"
    ]
    assert len(rendered_frames) == 2
    assert all(frame.shape == (480, 640, 3) for frame in rendered_frames)
    for controller in ("teacher", "teacher-wholebodyresidual"):
        episode_dir = tmp_path / "rendered" / f"{controller}-seed-41"
        assert (episode_dir / "rollout.mp4").is_file()
        assert (episode_dir / "result.json").is_file()
        telemetry = [
            json.loads(line)
            for line in (episode_dir / "telemetry.jsonl").read_text().splitlines()
        ]
        assert len(telemetry) == 2
        assert telemetry[-1]["success"] is True
        assert telemetry[-1]["observation"][-5:] == pytest.approx(
            [0.1, -0.2, 0.3, 1.0, 0.0]
        )

    blocked = tmp_path / "blocked"
    conflict = blocked / "teacher-wholebodyresidual-seed-42" / "result.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        whole_body.evaluate_onnx_render(
            exported,
            seeds=[42],
            out=blocked,
            max_steps=2,
        )
    assert not (blocked / "teacher-seed-42").exists()


def test_training_aborts_before_checkpoint_when_source_drifts(
    tmp_path,
    monkeypatch,
):
    snapshots = iter(
        [
            {"files": {"module": {"path": "module.py", "sha256": "before"}}},
            {"files": {"module": {"path": "module.py", "sha256": "after"}}},
        ]
    )
    monkeypatch.setattr(whole_body, "source_provenance", lambda: next(snapshots))

    with pytest.raises(RuntimeError, match="whole-body PPO training"):
        whole_body.train_ppo(
            tmp_path / "drifted",
            total_timesteps=8,
            seed=1,
            max_steps=1,
        )
    assert not (tmp_path / "drifted" / "whole-body-residual-ppo.zip").exists()


def test_training_cli_forwards_versioned_residual_configuration(
    tmp_path,
    monkeypatch,
    capsys,
):
    expected = tmp_path / "checkpoint.zip"
    calls = []

    def fake_train(out, steps, seed, **kwargs):
        calls.append((out, steps, seed, kwargs))
        return expected

    monkeypatch.setattr(whole_body, "train_ppo", fake_train)
    assert (
        whole_body.main(
            [
                "train-ppo",
                "--out",
                str(tmp_path / "run"),
                "--steps",
                "32",
                "--seed",
                "7",
                "--leg-residual-rad",
                "0.004",
                "--arm-residual-rad",
                "0.025",
                "--case",
                "walking_carry",
            ]
        )
        == 0
    )

    assert calls == [
        (
            str(tmp_path / "run"),
            32,
            7,
            {
                "case": "walking_carry",
                "mode": "free",
                "candidate": "A",
                "max_steps": learning.DEFAULT_MAX_STEPS,
                "payload_kg": learning.DEFAULT_PAYLOAD_KG,
                "leg_residual_rad": 0.004,
                "arm_residual_rad": 0.025,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["artifact"] == str(expected)


def test_onnx_render_cli_forwards_full_episode_configuration(
    tmp_path,
    monkeypatch,
    capsys,
):
    expected = tmp_path / "rendered" / "evaluation.json"
    calls = []

    def fake_evaluate(onnx, seeds, out, **kwargs):
        calls.append((onnx, list(seeds), out, kwargs))
        return expected

    monkeypatch.setattr(whole_body, "evaluate_onnx_render", fake_evaluate)
    assert (
        whole_body.main(
            [
                "evaluate-onnx-render",
                "--onnx",
                str(tmp_path / "policy.onnx"),
                "--out",
                str(tmp_path / "rendered"),
                "--seeds",
                "7,9",
                "--case",
                "walking_carry",
            ]
        )
        == 0
    )
    assert calls == [
        (
            str(tmp_path / "policy.onnx"),
            [7, 9],
            str(tmp_path / "rendered"),
            {
                "case": "walking_carry",
                "mode": "free",
                "candidate": "A",
                "max_steps": learning.DEFAULT_MAX_STEPS,
                "payload_kg": learning.DEFAULT_PAYLOAD_KG,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["artifact"] == str(expected)
