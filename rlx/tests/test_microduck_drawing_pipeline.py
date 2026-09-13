"""Cheap contracts for the standalone MicroDuck drawing pipeline."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ppo_microduck_drawing.py"
)
SPEC = importlib.util.spec_from_file_location("ppo_microduck_drawing", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scales", torch.as_tensor(module.DEFAULT_OBS_SCALE))

    def forward(self, observation):
        return observation * self.scales


class TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(module.OBS_DIM, 32), nn.Tanh())

    def forward_actor(self, features):
        return self.actor(features)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.features_extractor = TinyFeatures()
        self.mlp_extractor = TinyMlp()
        self.action_net = nn.Linear(32, module.ACTION_DIM)


class TinyModel:
    def __init__(self):
        self.policy = TinyPolicy()

    def predict(self, observations, deterministic=True):
        assert deterministic is True
        actor = module.DeterministicActor(self.policy).eval()
        with torch.no_grad():
            actions = actor(torch.as_tensor(observations, dtype=torch.float32)).numpy()
        return actions, None


def test_cli_contract_matches_drawing_studio_flags(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"checkpoint")

    train = module.parse_args(
        [
            "train",
            "--output",
            str(tmp_path / "run"),
            "--total-timesteps",
            "768",
            "--seed",
            "7",
            "--max-episode-s",
            "12",
            "--assistance",
            "1",
            ".3",
            "0",
        ]
    )
    evaluate = module.parse_args(
        [
            "eval",
            "--checkpoint",
            str(checkpoint),
            "--eval-output",
            str(tmp_path / "eval.json"),
            "--eval-episodes",
            "2",
        ]
    )
    render = module.parse_args(
        [
            "render",
            "--checkpoint",
            str(checkpoint),
            "--render-output",
            str(tmp_path / "render"),
            "--render-seconds",
            "4",
        ]
    )
    export = module.parse_args(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--onnx-output",
            str(tmp_path / "policy.onnx"),
        ]
    )

    assert train.assistance == [1.0, 0.3, 0.0]
    assert train.teacher_assistance == 0.0
    assert train.dagger_rounds == 2
    assert train.render_after_train is False
    assert train.teacher_domain_spread is False
    assert train.total_timesteps == 768
    assert evaluate.eval_episodes == 2
    assert render.render_seconds == 4
    assert render.controller == "onnx"
    assert export.onnx_output.name == "policy.onnx"


def test_fixed_observation_scaling_emphasizes_target_difference():
    extractor = module.DrawingObsScaleExtractor(
        gym.spaces.Box(-np.inf, np.inf, (module.OBS_DIM,), dtype=np.float32)
    )
    observation = torch.ones(2, module.OBS_DIM)
    scaled = extractor(observation)

    torch.testing.assert_close(
        scaled[:, 67:70], torch.full((2, 3), 100.0)
    )
    torch.testing.assert_close(scaled[:, :67], torch.ones(2, 67))
    assert "scales" in dict(extractor.named_buffers())


def test_behavior_clone_preserves_configured_action_std():
    class DummyDrawingEnv(gym.Env):
        observation_space = gym.spaces.Box(
            -np.inf, np.inf, (module.OBS_DIM,), dtype=np.float32
        )
        action_space = gym.spaces.Box(
            -1.0, 1.0, (module.ACTION_DIM,), dtype=np.float32
        )

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(module.OBS_DIM, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(module.OBS_DIM, dtype=np.float32),
                0.0,
                False,
                False,
                {},
            )

    configured_std = 0.005
    model = module.create_ppo(
        DummyDrawingEnv(),
        seed=7,
        learning_rate=3e-6,
        initial_std=configured_std,
    )
    observations = np.zeros((2, module.OBS_DIM), dtype=np.float32)
    actions = np.zeros((2, module.ACTION_DIM), dtype=np.float32)

    module.behavior_clone(
        model,
        observations,
        actions,
        epochs=1,
        batch_size=2,
        learning_rate=3e-4,
        seed=7,
    )

    expected = torch.full_like(model.policy.log_std, configured_std)
    torch.testing.assert_close(model.policy.log_std.exp(), expected)


def test_exported_onnx_is_deterministic_and_matches_torch(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    model = TinyModel()
    metadata = module.checkpoint_metadata(
        max_episode_s=32,
        seed=3,
        total_timesteps=100,
        assistance=(1.0, 0.3, 0.0),
    )

    result = module.export_policy(model, tmp_path / "policy.onnx", metadata)

    assert result["parity"]["passed"] is True
    assert result["parity"]["max_abs_error"] <= 1e-5
    assert len(result["sha256"]) == 64
    properties = {
        item.key: item.value for item in onnx.load(tmp_path / "policy.onnx").metadata_props
    }
    assert json.loads(properties["rlx_metadata"]) == metadata


def test_exported_onnx_loads_through_real_drawing_studio_boundary(
    tmp_path, monkeypatch
):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    local_src = MODULE_PATH.parents[2] / "microduck_local" / "src"
    monkeypatch.syspath_prepend(str(local_src))
    sys.modules.pop("microduck_local.studio_policies", None)
    from microduck_local import studio_policies

    runs_root = tmp_path / "studio"
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(runs_root))
    policy_path = runs_root / "drawing" / "integration" / "policy.onnx"
    metadata = module.checkpoint_metadata(
        max_episode_s=4.0,
        seed=11,
        total_timesteps=1,
        assistance=(0.0,),
    )
    module.export_policy(TinyModel(), policy_path, metadata)

    loaded = studio_policies.studio_metadata(policy_path)
    env = studio_policies.make_studio_env(policy_path, seed=11)
    try:
        observation, _ = env.reset(seed=11)
        assert loaded == metadata
        assert observation.shape == (module.OBS_DIM,)
        assert env.action_space.shape == (module.ACTION_DIM,)
        assert env.studio_recipe == "drawing"
    finally:
        env.close()


def test_export_cli_updates_sidecar_hash_without_recursive_metadata(
    tmp_path, monkeypatch, capsys
):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"checkpoint")
    metadata = module.checkpoint_metadata(
        max_episode_s=4.0,
        seed=13,
        total_timesteps=256,
        assistance=(0.0,),
    )
    original_source_hashes = dict(metadata["source_hashes"])
    sidecar = Path(str(checkpoint) + ".json")
    module.write_json(
        sidecar,
        {
            **metadata,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": module.sha256_file(checkpoint),
            "onnx": {"path": "stale", "sha256": "0" * 64},
        },
    )
    monkeypatch.setattr(module, "load_ppo", lambda _checkpoint: TinyModel())
    output = tmp_path / "policy.onnx"

    assert module.main(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--onnx-output",
            str(output),
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    updated = json.loads(sidecar.read_text())
    properties = {item.key: item.value for item in onnx.load(output).metadata_props}
    embedded = json.loads(properties["rlx_metadata"])
    assert updated["onnx"]["sha256"] == module.sha256_file(output)
    assert result["sha256"] == updated["onnx"]["sha256"]
    assert result["metadata_sha256"] == module.sha256_file(sidecar)
    assert updated["source_hashes"] == original_source_hashes
    assert embedded == metadata
    assert "onnx" not in embedded
    assert "checkpoint_sha256" not in embedded


def test_required_assessment_fails_closed_unless_every_episode_passes():
    passing = {
        "return": 2.0,
        "drawing_assessment": {
            "passed": True,
            "coverage": 0.9,
            "precision": 0.8,
            "grasp_fraction": 1.0,
            "pen_up_leak_fraction": 0.0,
            "max_normal_force_n": 0.2,
            "ink_contact_fraction": 0.95,
            "symmetric_chamfer_m": 0.001,
            "length_ratio": 1.0,
        },
    }
    failing = {
        **passing,
        "drawing_assessment": {**passing["drawing_assessment"], "passed": False},
    }

    assert module.aggregate_evaluations([passing, passing])["passed"] is True
    aggregate = module.aggregate_evaluations([passing, failing])
    assert aggregate["passed"] is False
    assert aggregate["passed_episode_count"] == 1


def test_studio_drawing_assessment_has_unassisted_episode_evidence():
    result = {
        "seed": 4,
        "steps": 1600,
        "case": "nominal",
        "return": 1.0,
        "drawing_assessment": {
            "passed": True,
            "unassisted": True,
            "assistance": 0.0,
            "coverage": 0.99,
            "precision": 0.98,
            "ink_contact_fraction": 0.9,
        },
    }

    assessment = module.studio_drawing_assessment([result])

    assert assessment["passed"] is True
    assert assessment["unassisted"] is True
    assert assessment["episodes"][0]["measured_steps"] == 1600
    assert assessment["episodes"][0]["case"] == "nominal"


def test_evaluation_report_binds_source_metadata_and_studio_fields(
    tmp_path, monkeypatch
):
    onnx = tmp_path / "policy.onnx"
    checkpoint = tmp_path / "policy.zip"
    metadata = tmp_path / "policy.zip.json"
    onnx.write_bytes(b"onnx")
    checkpoint.write_bytes(b"checkpoint")
    metadata.write_text("{}")

    class FakePolicy:
        def __init__(self, path):
            assert path == onnx

        def __call__(self, observation):
            return np.zeros(module.ACTION_DIM, np.float32)

    class FakeEnv:
        max_steps = 2

        def reset(self, seed=None):
            self.steps = 0
            return np.zeros(module.OBS_DIM, np.float32), {}

        def step(self, action):
            self.steps += 1
            return (
                np.zeros(module.OBS_DIM, np.float32),
                1.0,
                False,
                self.steps == 2,
                {},
            )

        def assessment(self):
            return {
                "passed": True,
                "unassisted": True,
                "assistance": 0.0,
                "contact_steps": 2,
                "coverage": 1.0,
                "precision": 1.0,
            }

        def drawing_payload(self):
            return {"points": []}

        def close(self):
            pass

    monkeypatch.setattr(module, "OnnxPolicy", FakePolicy)
    monkeypatch.setattr(module, "make_env", lambda **kwargs: FakeEnv())

    report = module.evaluate_onnx(
        onnx,
        seed=7,
        eval_episodes=1,
        max_episode_s=32,
        checkpoint_path=checkpoint,
    )

    assert report["source_sha256"] == module.sha256_file(onnx)
    assert report["metadata_sha256"] == module.sha256_file(metadata)
    assert report["source_files_sha256"][str(onnx.resolve())] == report["source_sha256"]
    assert report["recipe"] == "drawing"
    assert report["evaluation_mode"] == "skill"
    assert report["skill_status"] == "passed"
    assert report["drawing_assessment"]["unassisted"] is True
    assert set(report["evaluation_source_hashes"]) == {
        "pipeline_sha256",
        "environment_sha256",
    }
    assert all(
        len(value) == 64 for value in report["evaluation_source_hashes"].values()
    )
    assert report["environment_source_match"] is False
    assert report["reevaluation_notice"] is not None
    assert len(report["drawing_assessment"]["episodes"]) == 4
    assert all(
        episode["ink_contact_fraction"] == 1.0
        for episode in report["drawing_assessment"]["episodes"]
    )


def test_bc_and_ppo_are_reported_separately_without_unsupported_claim():
    baseline = {
        "drawing_assessment": {
            "passed": False,
            "mean_return": 10.0,
            "means": {"coverage": 0.4, "precision": 0.5, "grasp_fraction": 0.8},
        }
    }
    ppo = {
        "drawing_assessment": {
            "passed": False,
            "mean_return": 12.0,
            "means": {"coverage": 0.45, "precision": 0.48, "grasp_fraction": 0.85},
        }
    }

    comparison = module.comparison_evidence(baseline, ppo)

    assert comparison["mean_return_delta"] == 2.0
    assert comparison["metric_deltas"]["coverage"] == pytest.approx(0.05)
    assert comparison["ppo_improvement_claimed"] is False

    report = module.evaluation_with_bc_baseline(ppo, baseline)
    assert report["drawing_assessment"] == ppo["drawing_assessment"]
    assert report["bc_baseline"] == baseline
    assert report["comparison"]["ppo_improvement_claimed"] is False


def test_standalone_policy_eval_restores_neighboring_bc_comparison(
    tmp_path, monkeypatch
):
    policy = tmp_path / "policy.onnx"
    baseline = tmp_path / "bc_policy.onnx"
    checkpoint = tmp_path / "policy.zip"
    bc_checkpoint = tmp_path / "bc_policy.zip"
    for path in (policy, baseline, checkpoint, bc_checkpoint):
        path.write_bytes(b"artifact")
    calls = []

    def fake_evaluate(path, **kwargs):
        calls.append((path, kwargs))
        coverage = 0.6 if path == policy else 0.7
        return {
            "source": str(path),
            "drawing_assessment": {
                "passed": False,
                "mean_return": coverage,
                "means": {
                    "coverage": coverage,
                    "precision": coverage,
                    "grasp_fraction": 1.0,
                },
            },
        }

    monkeypatch.setattr(module, "evaluate_onnx", fake_evaluate)
    report = module.evaluate_policy_with_neighboring_bc(
        policy,
        seed=101,
        eval_episodes=8,
        max_episode_s=32,
        checkpoint_path=checkpoint,
    )

    assert [call[0] for call in calls] == [policy, baseline]
    assert calls[1][1]["checkpoint_path"] == bc_checkpoint
    assert calls[1][1]["seed"] == 101
    assert calls[1][1]["eval_episodes"] == 8
    assert report["bc_baseline"]["source"] == str(baseline)
    assert report["comparison"]["metric_deltas"]["coverage"] == pytest.approx(-0.1)
    assert report["comparison"]["ppo_improvement_claimed"] is False


def test_curriculum_uses_real_learn_calls_and_never_teacher_controller(monkeypatch):
    closed = []

    class FakeEnv:
        def __init__(self, assistance):
            self.assistance = assistance

        def close(self):
            closed.append(self.assistance)

    class FakeModel:
        def __init__(self):
            self.env = FakeEnv(-1)
            self.calls = []
            self.n_steps = 2

        def get_env(self):
            return self.env

        def set_env(self, env):
            self.env = env

        def learn(self, **kwargs):
            self.calls.append(kwargs)

    monkeypatch.setattr(
        module.PPOHistoryCallback,
        "create",
        staticmethod(lambda history, total_timesteps: "callback"),
    )
    model = FakeModel()
    factory_calls = []

    def factory(**kwargs):
        factory_calls.append(kwargs)
        return FakeEnv(kwargs["assistance"])

    module.run_ppo_curriculum(
        model,
        total_timesteps=12,
        assistance=(1.0, 0.3, 0.0),
        seed=5,
        max_episode_s=8,
        history=[],
        env_factory=factory,
    )

    assert [call["total_timesteps"] for call in model.calls] == [4, 4, 4]
    assert [call["reset_num_timesteps"] for call in model.calls] == [True, False, False]
    assert [call["assistance"] for call in factory_calls] == [1.0, 0.3, 0.0]
    assert closed == [-1, 1.0, 0.3, 0.0]


def test_ppo_rollout_emits_studio_training_progress(capsys):
    history = []
    callback = module.PPOHistoryCallback.create(history, total_timesteps=768)
    callback.model = type(
        "Model",
        (),
        {
            "num_timesteps": 256,
            "rollout_buffer": type(
                "Buffer", (), {"rewards": np.asarray([[1.0], [3.0]])}
            )(),
            "logger": type(
                "Logger",
                (),
                {"name_to_value": {"train/loss": 0.5}},
            )(),
        },
    )()

    callback._on_rollout_end()

    event = json.loads(capsys.readouterr().out)
    assert event == {
        "event": "training_progress",
        "steps": 256,
        "total": 768,
        "total_timesteps": 768,
        "mean_reward": 2.0,
        "normalize_rewards": False,
    }
    assert history[0]["timesteps"] == 256
    assert history[0]["mean_rollout_reward"] == 2.0


def test_dataset_contains_observed_state_action_and_provenance(tmp_path):
    arrays = {
        "observations": np.zeros((3, module.OBS_DIM), np.float32),
        "actions": np.zeros((3, module.ACTION_DIM), np.float32),
        "rewards": np.ones(3, np.float32),
        "episode": np.zeros(3, np.int32),
        "step": np.arange(3, dtype=np.int32),
        "scale": np.ones(3, np.float32),
        "offset_x": np.zeros(3, np.float32),
        "offset_y": np.zeros(3, np.float32),
        "teacher_assistance": np.zeros(3, np.float32),
        "teacher_assessments_json": np.asarray(
            [
                json.dumps(
                    {"passed": True, "assistance": 0.0, "unassisted": True}
                )
            ]
        ),
    }
    output = tmp_path / "teacher.npz"

    metadata = module.save_teacher_dataset(
        output,
        arrays,
        seed=9,
        max_episode_s=32,
        assistance=1.0,
    )
    observations, actions = module.load_teacher_dataset(output)
    sidecar = json.loads((tmp_path / "teacher.npz.json").read_text())

    assert observations.shape == (3, module.OBS_DIM)
    assert actions.shape == (3, module.ACTION_DIM)
    assert metadata["dataset_sha256"] == sidecar["dataset_sha256"]
    assert sidecar["contract_version"] == "microduck-drawing-v1"
    assert sidecar["teacher_passed_all"] is True


def test_dataset_assistance_provenance_mismatch_fails_closed(tmp_path):
    arrays = {
        "observations": np.zeros((3, module.OBS_DIM), np.float32),
        "actions": np.zeros((3, module.ACTION_DIM), np.float32),
        "rewards": np.ones(3, np.float32),
        "episode": np.zeros(3, np.int32),
        "step": np.arange(3, dtype=np.int32),
        "scale": np.ones(3, np.float32),
        "offset_x": np.zeros(3, np.float32),
        "offset_y": np.zeros(3, np.float32),
        "teacher_assistance": np.ones(3, np.float32),
        "teacher_assessments_json": np.asarray(
            [
                json.dumps(
                    {"passed": True, "assistance": 0.0, "unassisted": True}
                )
            ]
        ),
    }
    output = tmp_path / "invalid-teacher.npz"

    metadata = module.save_teacher_dataset(
        output,
        arrays,
        seed=9,
        max_episode_s=32,
        assistance=0.0,
    )

    assert metadata["assistance_provenance_valid"] is False
    assert metadata["teacher_passed_all"] is False
    with pytest.raises(ValueError, match="assistance provenance is invalid"):
        module.load_teacher_dataset(output)


def test_teacher_domain_spread_cycles_offset_and_scale_without_changing_defaults():
    assert module.teacher_environment_options(3, domain_spread=False) == (
        "nominal",
        {},
    )
    cases = [
        module.teacher_environment_options(index, domain_spread=True)
        for index in range(5)
    ]

    assert cases[0] == ("nominal", {})
    assert cases[1][1]["offset"] == (0.004, -0.003)
    assert cases[2][1]["scale"] == 0.9
    assert cases[3][1]["offset"] == (-0.004, 0.003)
    assert cases[4][1]["scale"] == 1.1


def test_contact_svg_labels_actual_contact_data():
    trace = [
        [0.0, 0.160, -0.01, 0.220, 1, 0.1, True],
        [0.02, 0.160, 0.00, 0.221, 1, 0.1, True],
        [0.04, 0.160, 0.01, 0.222, 2, 0.1, True],
    ]

    svg = module.contact_svg(trace)

    assert "Actual MuJoCo pencil-tip canvas contacts" in svg
    assert svg.count("<path ") == 2
    assert "reference" not in svg.lower()


def test_raw_contact_trace_writes_native_rows_with_force_and_request(tmp_path):
    trace = [
        [0.02, 0.160, -0.01, 0.220, 1, 0.284, True],
        [0.04, 0.160, 0.00, 0.221, 1, 0.190, False],
    ]

    paths = module.write_raw_contact_trace(
        tmp_path, trace, controller="teacher", seed=4
    )

    csv_lines = (tmp_path / "raw_contact_trace.csv").read_text().splitlines()
    payload = json.loads((tmp_path / "raw_contact_trace.json").read_text())
    assert csv_lines[0] == "t,x,y,z,strokeId,force,requested_down"
    assert csv_lines[1].endswith(",0.284,True")
    assert payload == {
        "controller": "teacher",
        "seed": 4,
        "columns": [
            "t",
            "x",
            "y",
            "z",
            "strokeId",
            "force",
            "requested_down",
        ],
        "rows": trace,
    }
    assert paths["csv"].endswith("raw_contact_trace.csv")
    assert paths["json"].endswith("raw_contact_trace.json")


def test_render_policy_labels_distinguish_bc_teacher_and_ppo():
    assert module.policy_source_label(Path("bc_policy.onnx")) == "BC policy"
    assert module.policy_source_label(Path("teacher_reference.onnx")) == "teacher policy"
    assert module.policy_source_label(Path("policy.onnx")) == "PPO policy"


def test_teacher_render_cli_keeps_checkpoint_required_but_selects_teacher(tmp_path):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not used by teacher rendering")

    args = module.parse_args(
        [
            "render",
            "--checkpoint",
            str(checkpoint),
            "--render-output",
            str(tmp_path / "teacher-render"),
            "--controller",
            "teacher",
            "--render-seconds",
            "32",
        ]
    )

    assert args.controller == "teacher"
    assert args.checkpoint == checkpoint


def test_teacher_render_main_does_not_resolve_or_bind_onnx(
    tmp_path, monkeypatch, capsys
):
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"required argument, intentionally unused")
    captured = {}

    def fail_resolve(_checkpoint):
        raise AssertionError("teacher rendering must not resolve ONNX")

    def fake_render(onnx_path, output, **kwargs):
        captured.update(
            {
                "onnx_path": onnx_path,
                "output": output,
                **kwargs,
            }
        )
        return {
            "controller": "teacher",
            "source": None,
            "source_type": "teacher_controller",
            "checkpoint_used": False,
        }

    monkeypatch.setattr(module, "resolve_onnx", fail_resolve)
    monkeypatch.setattr(module, "render_onnx", fake_render)

    assert (
        module.main(
            [
                "render",
                "--checkpoint",
                str(checkpoint),
                "--render-output",
                str(tmp_path / "teacher-render"),
                "--controller",
                "teacher",
            ]
        )
        == 0
    )

    assert captured["onnx_path"] is None
    assert captured["controller"] == "teacher"
    assert captured["checkpoint_argument"] == checkpoint
    assert json.loads(capsys.readouterr().out)["source"] is None


def test_contact_geoms_use_only_actual_same_stroke_segments(monkeypatch):
    calls = []

    class FakeMujoco:
        class mjtGeom:
            mjGEOM_CAPSULE = 3

        @staticmethod
        def mjv_initGeom(*args):
            calls.append(("init", args))

        @staticmethod
        def mjv_connector(*args):
            calls.append(("connector", args))

    monkeypatch.setitem(__import__("sys").modules, "mujoco", FakeMujoco)
    scene = SimpleScene(maxgeom=8)
    payload = {
        "points": [
            [0.159, -0.01, 0.22, 1],
            [0.159, 0.00, 0.221, 1],
            [0.159, 0.01, 0.222, 2],
        ]
    }

    added = module.add_contact_geoms(scene, payload)

    assert added == 1
    assert scene.ngeom == 1
    assert [call[0] for call in calls] == ["init", "connector"]


class SimpleScene:
    def __init__(self, maxgeom):
        self.maxgeom = maxgeom
        self.ngeom = 0
        self.geoms = [object() for _ in range(maxgeom)]


def test_checkpoint_metadata_has_deployment_recipe_contract():
    metadata = module.checkpoint_metadata(
        max_episode_s=32,
        seed=1,
        total_timesteps=2048,
        assistance=(1.0, 0.3, 0.0),
    )

    assert metadata["recipe"] == "drawing"
    assert metadata["actuator"] == "xml"
    assert metadata["recipe_options"] == {}
    assert metadata["reward_weights"] == module.REWARD_WEIGHTS
    assert metadata["contract_version"] == "microduck-drawing-v1"
    assert metadata["observation_dim"] == 83
    assert metadata["action_dim"] == 15
    assert metadata["deployment_scope"] == "simulation_only"
    assert metadata["hardware_compatible"] is False
    assert metadata["simulation_actuators"]["joint_servos"] == {
        "kp": 8.0,
        "kv": 0.2,
        "torque_limit_nm": 0.96,
    }
    assert set(metadata["source_hashes"]) == {
        "pipeline_sha256",
        "environment_sha256",
    }
