import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIP = ROOT / "assets" / "clips" / "dance-120bpm.json"


def load_dance_example():
    spec = importlib.util.spec_from_file_location(
        "ppo_microduck_dance", ROOT / "examples" / "ppo_microduck_dance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dance_clip_is_safe_looping_120_bpm():
    data = json.loads(CLIP.read_text())
    assert data["version"] == 1
    assert data["bpm"] == 120
    assert data["loop"] is True
    assert data["duration"] % (60 / data["bpm"]) == 0
    times = [key["t"] for key in data["keys"]]
    assert times == sorted(times)
    poses = np.asarray([key["joints"] for key in data["keys"]])
    assert poses.shape == (5, 14)
    np.testing.assert_array_equal(poses[0], poses[-1])
    assert np.isfinite(poses).all()
    assert np.max(np.abs(np.diff(poses, axis=0))) < 0.25


def test_clip_loads_and_loops():
    from microduck_local import motion

    clip = motion.load_clip("dance-120bpm", ROOT / "assets" / "clips")
    assert clip.loop
    assert clip.steps == 50
    np.testing.assert_allclose(clip.at(0)[0], clip.at(clip.steps)[0])


def test_model_export_and_example_modules_are_runtime_lazy():
    script = """
import importlib.util, sys
import rlx.models.microduck
import rlx.export.microduck_onnx
spec = importlib.util.spec_from_file_location('dance_example', 'examples/ppo_microduck_dance.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
for root in ('mlx', 'torch', 'tensorboard', 'wandb', 'onnxruntime'):
    assert not any(n == root or n.startswith(root + '.') for n in sys.modules), root
class Reached(Exception): pass
def fake(**kwargs):
    for root in ('mlx', 'torch', 'tensorboard', 'wandb', 'onnxruntime'):
        assert not any(n == root or n.startswith(root + '.') for n in sys.modules), root
    raise Reached
module.make_microduck_env = fake
try:
    module.main(['eval', '--checkpoint', '/tmp/missing', '--num-envs', '1'])
except Reached:
    pass
else:
    raise AssertionError('factory not reached')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_default_reward_weights_merge_user_json():
    module = load_dance_example()
    assert module._weight_overrides("{}") == {
        "travel": 0.0,
        "stay_home": 1.0,
        "face_home": 1.0,
    }
    assert module._weight_overrides('{"stay_home": 2.5, "pose_match": 8}') == {
        "travel": 0.0,
        "stay_home": 2.5,
        "face_home": 1.0,
        "pose_match": 8.0,
    }
    with pytest.raises(ValueError, match="JSON object"):
        module._weight_overrides("[]")
    with pytest.raises(ValueError, match="non-negative"):
        module._weight_overrides('{"travel": -1}')


def test_ppo_flags_and_validation():
    module = load_dance_example()
    defaults = module.parse_args(
        [
            "train",
            "--checkpoint",
            "/tmp/defaults.safetensors",
        ]
    )
    assert defaults.learning_rate == 3e-4
    assert defaults.value_coefficient == 1.0
    metadata = module._training_metadata(defaults, 42)
    assert metadata["ppo"]["learning_rate"] == 3e-4
    assert metadata["ppo"]["value_coefficient"] == 1.0
    args = module.parse_args(
        [
            "train",
            "--checkpoint",
            "/tmp/dance.safetensors",
            "--num-envs",
            "2",
            "--num-steps",
            "4",
            "--num-minibatches",
            "2",
            "--total-timesteps",
            "8",
            "--gae-lambda",
            "0.9",
            "--no-normalize-advantages",
            "--clip-coefficient",
            "0.1",
            "--no-clip-value-loss",
            "--entropy-coefficient",
            "0.02",
            "--value-coefficient",
            "0.7",
            "--max-grad-norm",
            "0.8",
            "--no-export-onnx",
        ]
    )
    assert args.gae_lambda == 0.9
    assert args.normalize_advantages is False
    assert args.clip_coefficient == 0.1
    assert args.clip_value_loss is False
    assert args.entropy_coefficient == 0.02
    assert args.value_coefficient == 0.7
    assert args.max_grad_norm == 0.8
    assert args.export_onnx is False
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "train",
                "--checkpoint",
                "/tmp/x",
                "--num-envs",
                "2",
                "--num-steps",
                "3",
                "--num-minibatches",
                "4",
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "train",
                "--checkpoint",
                "/tmp/x",
                "--no-export-onnx",
                "--onnx-output",
                "/tmp/x.onnx",
            ]
        )


def test_microduck_actor_log_std_is_constrained():
    import mlx.core as mx

    from rlx.models.microduck import (
        ACTION_DIM,
        ACTOR_LOG_STD_MAX,
        ACTOR_LOG_STD_MIN,
        OBSERVATION_DIM,
        create_actor_critic,
    )

    model = create_actor_critic()
    model.actor_log_std = mx.array([-20.0, 5.0] * (ACTION_DIM // 2))
    distribution, _ = model(mx.zeros((2, OBSERVATION_DIM)))
    mx.eval(distribution.log_std)
    values = np.asarray(distribution.log_std)
    assert values.min() == ACTOR_LOG_STD_MIN
    assert values.max() == ACTOR_LOG_STD_MAX

    model.constrain_actor_log_std()
    mx.eval(model.actor_log_std)
    np.testing.assert_allclose(
        np.asarray(model.actor_log_std),
        [ACTOR_LOG_STD_MIN, ACTOR_LOG_STD_MAX] * (ACTION_DIM // 2),
    )


def test_bounded_dance_actor_and_onnx_share_action_limit(tmp_path):
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed")
    import mlx.core as mx
    import onnxruntime as ort

    from rlx.export.microduck_onnx import export_deterministic_actor
    from rlx.models.microduck import create_actor_critic, save_checkpoint

    model = create_actor_critic(action_limit=1.0)
    mx.eval(model.parameters())
    final_layer = model.actor_mean.layers[6]
    final_layer.weight = mx.zeros_like(final_layer.weight)
    final_layer.bias = mx.full_like(final_layer.bias, 20.0)
    checkpoint = tmp_path / "bounded-dance.safetensors"
    save_checkpoint(
        checkpoint,
        model,
        np.zeros(61, dtype=np.float32),
        np.ones(61, dtype=np.float32),
        1.0,
        metadata={"recipe": "dance", "policy_action_limit": 1.0},
    )

    policy = export_deterministic_actor(checkpoint, tmp_path / "bounded-dance.onnx")
    session = ort.InferenceSession(str(policy), providers=["CPUExecutionProvider"])
    output = session.run(
        None,
        {session.get_inputs()[0].name: np.zeros((2, 61), dtype=np.float32)},
    )[0]
    assert np.max(np.abs(output)) <= 1.0
    np.testing.assert_allclose(output, np.ones((2, 14)), atol=1e-6)


def test_gae_does_not_cross_truncated_or_terminated_boundaries():
    import mlx.core as mx

    from rlx.utils import compute_generalized_advantage_estimate

    rewards = mx.array([[1.0], [100.0], [10.0], [1000.0]])
    values = mx.zeros_like(rewards)
    terminations = mx.array([[False], [False], [True], [False]])
    truncations = mx.array([[True], [False], [False], [False]])
    advantages = compute_generalized_advantage_estimate(
        rewards,
        values,
        terminations,
        last_value=mx.array([0.0]),
        last_termination=mx.array([False]),
        gamma=1.0,
        gae_lambda=1.0,
        truncations=truncations,
        truncation_values=mx.zeros_like(values),
    )
    np.testing.assert_allclose(
        np.asarray(advantages).squeeze(-1),
        [1.0, 110.0, 10.0, 1000.0],
    )


def test_ppo_finite_guard_raises_clear_runtime_error():
    import mlx.core as mx

    from rlx.algorithms.ppo import PPO

    with pytest.raises(RuntimeError, match="PPO gradients contains non-finite"):
        PPO._assert_finite("gradients", {"actor": mx.array([0.0, mx.nan])})


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_save_checkpoint_rejects_non_finite_tensors(tmp_path, invalid):
    import mlx.core as mx

    from rlx.models.microduck import create_actor_critic, save_checkpoint

    model = create_actor_critic()
    mx.eval(model.parameters())
    model.actor_log_std = mx.full(model.actor_log_std.shape, invalid)
    checkpoint = tmp_path / "invalid.safetensors"
    with pytest.raises(RuntimeError, match="non-finite tensor.*actor_log_std"):
        save_checkpoint(
            checkpoint,
            model,
            np.zeros(61, dtype=np.float32),
            np.ones(61, dtype=np.float32),
            1.0,
        )
    assert not checkpoint.exists()
    assert not checkpoint.with_suffix(".safetensors.json").exists()


def test_save_checkpoint_rejects_non_finite_normalizer(tmp_path):
    import mlx.core as mx

    from rlx.models.microduck import create_actor_critic, save_checkpoint

    model = create_actor_critic()
    mx.eval(model.parameters())
    checkpoint = tmp_path / "invalid-normalizer.safetensors"
    variance = np.ones(61, dtype=np.float32)
    variance[7] = np.nan
    with pytest.raises(RuntimeError, match="normalizer.variance"):
        save_checkpoint(
            checkpoint,
            model,
            np.zeros(61, dtype=np.float32),
            variance,
            1.0,
        )
    assert not checkpoint.exists()


def test_checkpoint_roundtrip_and_onnx_dynamic_batch_parity(tmp_path):
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed")
    import mlx.core as mx

    from rlx.export.microduck_onnx import export_deterministic_actor, parity_error
    from rlx.models.microduck import (
        create_actor_critic,
        load_checkpoint,
        save_checkpoint,
    )

    mx.random.seed(17)
    model = create_actor_critic()
    mx.eval(model.parameters())
    mean = np.linspace(-0.2, 0.2, 61, dtype=np.float32)
    variance = np.linspace(0.5, 1.5, 61, dtype=np.float32)
    checkpoint = tmp_path / "dance.safetensors"
    save_checkpoint(
        checkpoint,
        model,
        mean,
        variance,
        123.0,
        return_mean=np.array(1.25),
        return_variance=np.array(2.5),
        return_count=456.0,
    )
    loaded = load_checkpoint(checkpoint)
    assert loaded["count"] == 123.0
    np.testing.assert_array_equal(loaded["mean"], mean)
    assert loaded["return_mean"] == pytest.approx(1.25)
    assert loaded["return_variance"] == pytest.approx(2.5)
    assert loaded["return_count"] == pytest.approx(456.0)
    onnx_path = export_deterministic_actor(checkpoint, tmp_path / "dance.onnx")
    rng = np.random.default_rng(4)
    for batch_size in (1, 7):
        batch = rng.normal(size=(batch_size, 61)).astype(np.float32)
        assert parity_error(checkpoint, onnx_path, batch) < 2e-5


def test_swing_onnx_bakes_action_clip(tmp_path):
    if importlib.util.find_spec("onnxruntime") is None:
        pytest.skip("onnxruntime is not installed")
    import mlx.core as mx
    import onnxruntime as ort

    from rlx.export.microduck_onnx import export_deterministic_actor
    from rlx.models.microduck import create_actor_critic, save_checkpoint

    model = create_actor_critic()
    mx.eval(model.parameters())
    final_layer = model.actor_mean.layers[6]
    final_layer.weight = mx.zeros_like(final_layer.weight)
    final_layer.bias = mx.full_like(final_layer.bias, 3.0)
    checkpoint = tmp_path / "swing.safetensors"
    save_checkpoint(
        checkpoint,
        model,
        np.zeros(61, dtype=np.float32),
        np.ones(61, dtype=np.float32),
        1.0,
        metadata={"recipe": "swing"},
    )
    onnx_path = export_deterministic_actor(checkpoint, tmp_path / "swing.onnx")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    result = session.run(
        None, {session.get_inputs()[0].name: np.zeros((1, 61), dtype=np.float32)}
    )[0]
    np.testing.assert_array_equal(result, np.ones((1, 14), dtype=np.float32))


def test_onnx_evaluation_receives_raw_observations(monkeypatch, tmp_path):
    module = load_dance_example()
    raw = np.full((2, 61), 3.25, dtype=np.float32)
    observed = []

    class FakeEnv:
        def reset(self, key):
            return raw.copy(), {}, {}

        def step(self, key, state, action):
            assert np.asarray(action).shape == (2, 14)
            info = {
                "_episode": np.array([False, True]),
                "episode": {"r": np.array([0.0, 7.0]), "l": np.array([0, 2])},
            }
            return (
                raw.copy(),
                {},
                np.array([1.0, 2.0]),
                np.array([False, False]),
                np.array([False, True]),
                info,
            )

        def close(self):
            pass

    monkeypatch.setattr(module, "_environment", lambda args, normalize: FakeEnv())

    def fake_policy(path):
        def infer(observation):
            observed.append(np.asarray(observation).copy())
            return np.zeros((2, 14), dtype=np.float32)

        return infer

    monkeypatch.setattr(module, "_policy_from_onnx", fake_policy)
    policy = tmp_path / "dance.onnx"
    args = module.parse_args(
        [
            "eval",
            "--policy",
            str(policy),
            "--num-envs",
            "2",
            "--eval-steps",
            "2",
            "--min-mean-return",
            "8",
            "--min-episodes",
            "2",
        ]
    )
    result = module.evaluate(args)
    assert all(np.array_equal(item, raw) for item in observed)
    assert result["source_type"] == "policy"
    assert result["episodes"] == 2
    assert result["episode_return"]["mean"] == 7.0
    assert result["truncated"] == 2
    assert result["finite"] is True
    assert result["passed"] is False
    assert len(result["failures"]) == 1


def test_training_progress_reports_steps_and_completed_episode_reward(capsys):
    module = load_dance_example()
    callback = module._training_progress_callback(8, 2, 2)

    callback(
        {
            "_episode": np.array([False, False]),
            "episode": {"r": np.array([0.0, 0.0])},
        },
        0,
    )
    assert capsys.readouterr().out == ""

    callback(
        {
            "_episode": np.array([False, True]),
            "episode": {"r": np.array([0.0, 7.5])},
        },
        2,
    )
    first = json.loads(capsys.readouterr().out)
    assert first == {
        "event": "training_progress",
        "steps": 4,
        "total": 8,
        "mean_reward": 7.5,
        "episodes": 1,
    }

    callback(
        {
            "_episode": np.array([False, False]),
            "episode": {"r": np.array([0.0, 0.0])},
        },
        6,
    )
    final = json.loads(capsys.readouterr().out)
    assert final["steps"] == 8
    assert final["mean_reward"] is None
    assert final["episodes"] == 1


def test_eval_cli_threshold_failure_exits_two(monkeypatch, capsys):
    module = load_dance_example()
    monkeypatch.setattr(
        module,
        "evaluate",
        lambda args: {"command": "eval", "passed": False, "failures": ["low"]},
    )
    with pytest.raises(SystemExit) as exc:
        module.main(["eval", "--checkpoint", "/tmp/missing", "--min-mean-return", "1"])
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["failures"] == ["low"]


def test_render_checkpoint_exports_temporary_policy_and_forwards_options(
    monkeypatch, tmp_path
):
    from rlx.export import microduck_onnx

    module = load_dance_example()
    checkpoint = tmp_path / "dance.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "rendered"
    observed = {}

    def fake_export(source, target):
        assert source == checkpoint
        target.write_bytes(b"onnx")
        observed["policy"] = target
        return target

    def fake_run(command, check):
        assert check is True
        policy = Path(command[command.index("--policy") + 1])
        assert policy == observed["policy"] and policy.is_file()
        observed["command"] = command

    monkeypatch.setattr(microduck_onnx, "export_deterministic_actor", fake_export)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.parse_args(
        [
            "render",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--width",
            "1920",
            "--height",
            "1080",
            "--fps",
            "25",
            "--sheet-frames",
            "9",
            "--camera",
            "front",
        ]
    )
    result = module.render(args)
    assert result["source_type"] == "checkpoint"
    command = observed["command"]
    for flag, value in {
        "--width": "1920",
        "--height": "1080",
        "--fps": "25",
        "--sheet-frames": "9",
        "--camera": "front",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert not observed["policy"].exists()


def test_render_policy_is_forwarded_without_export(monkeypatch, tmp_path):
    module = load_dance_example()
    policy = tmp_path / "dance.onnx"
    policy.write_bytes(b"onnx")
    observed = {}
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, check: observed.update(command=command, check=check),
    )
    args = module.parse_args(
        [
            "render",
            "--policy",
            str(policy),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    result = module.render(args)
    assert result["source_type"] == "policy"
    assert observed["command"][observed["command"].index("--policy") + 1] == str(policy)


def test_render_source_validation_and_positive_dimensions(tmp_path):
    module = load_dance_example()
    with pytest.raises(SystemExit):
        module.parse_args(["render", "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "render",
                "--policy",
                "/tmp/a",
                "--checkpoint",
                "/tmp/b",
                "--output",
                str(tmp_path),
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            ["render", "--policy", "/tmp/a", "--output", str(tmp_path), "--width", "0"]
        )
    args = module.parse_args(
        ["render", "--policy", "/tmp/missing.onnx", "--output", str(tmp_path)]
    )
    with pytest.raises(FileNotFoundError, match="policy does not exist"):
        module.render(args)


def test_demo_is_labeled_smoke_and_optionally_renders(monkeypatch, tmp_path):
    module = load_dance_example()
    calls = []

    def fake_run(command, check, capture_output, text):
        del check, capture_output, text
        calls.append(command)
        subcommand = command[2]
        payload = {
            "train": {"command": "train", "steps": 4},
            "eval": {"command": "eval", "passed": True},
            "render": {"command": "render", "output": "render"},
        }[subcommand]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload) + "\n",
            check_returncode=lambda: None,
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    args = module.parse_args(
        ["demo", "--output", str(tmp_path), "--render", "--width", "800"]
    )
    result = module.demo(args)
    assert result["label"] == "smoke"
    assert result["passed"] is True
    assert [stage["command"] for stage in result["stages"]] == [
        "train",
        "export",
        "eval",
        "render",
    ]
    assert calls[0][2] == "train" and "--onnx-output" in calls[0]
    assert calls[1][2] == "eval" and "--policy" in calls[1]
    assert calls[2][2] == "render" and "--width" in calls[2]


def test_gated_end_to_end_dance_demo(tmp_path):
    import os

    if os.environ.get("RUN_MICRODUCK_INTEGRATION") != "1":
        pytest.skip("set RUN_MICRODUCK_INTEGRATION=1 for the dance workflow")
    environment = {
        **os.environ,
        "PYTHONPATH": f"{ROOT}:{ROOT.parent / 'microduck_local' / 'src'}",
    }
    result = subprocess.run(
        [
            sys.executable,
            "examples/ppo_microduck_dance.py",
            "demo",
            "--output",
            str(tmp_path / "smoke"),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["command"] == "demo"
    assert payload["label"] == "smoke"
    assert payload["passed"] is True
    assert (tmp_path / "smoke" / "smoke.safetensors").is_file()
    assert (tmp_path / "smoke" / "smoke.onnx").is_file()


def test_view_parser_supports_checkpoint_or_policy(tmp_path):
    module = load_dance_example()
    checkpoint = tmp_path / "dance.safetensors"
    policy = tmp_path / "dance.onnx"
    args = module.parse_args(["view", "--checkpoint", str(checkpoint)])
    assert args.checkpoint == checkpoint
    assert args.policy is None
    assert args.seed == 1
    assert args.max_episode_s == 4.0
    args = module.parse_args(
        ["view", "--policy", str(policy), "--seed", "7", "--max-episode-s", "2"]
    )
    assert args.policy == policy
    assert args.seed == 7
    assert args.max_episode_s == 2.0
    with pytest.raises(SystemExit):
        module.parse_args(["view"])
    with pytest.raises(SystemExit):
        module.parse_args(
            ["view", "--checkpoint", str(checkpoint), "--policy", str(policy)]
        )


def test_view_checkpoint_exports_policy_for_entire_loop(monkeypatch, tmp_path):
    from rlx.export import microduck_onnx

    module = load_dance_example()
    checkpoint = tmp_path / "dance.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    observed = {}

    class FakeEnv:
        def close(self):
            observed["closed"] = True

    def fake_export(source, target):
        assert source == checkpoint
        target.write_bytes(b"onnx")
        return target

    def fake_loop(policy, env, **kwargs):
        assert policy.is_file()
        assert env is observed["env"]
        observed["policy"] = policy
        observed["loop_kwargs"] = kwargs

    observed["env"] = FakeEnv()
    monkeypatch.setattr(module, "_make_view_environment", lambda seed, seconds: observed["env"])
    monkeypatch.setattr(module, "_run_view_loop", fake_loop)
    monkeypatch.setattr(microduck_onnx, "export_deterministic_actor", fake_export)
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(name=name),
    )
    args = module.parse_args(["view", "--checkpoint", str(checkpoint)])
    result = module.view(args)
    assert result["source_type"] == "checkpoint"
    assert observed["closed"] is True
    assert not observed["policy"].exists()


def test_view_policy_loop_steps_resets_and_syncs(monkeypatch, tmp_path):
    module = load_dance_example()
    policy = tmp_path / "dance.onnx"
    policy.write_bytes(b"onnx")
    observed = {"resets": 0, "steps": 0, "syncs": 0}

    class FakeInput:
        name = "observations"

    class FakeSession:
        def __init__(self, path, providers):
            assert Path(path) == policy
            assert providers == ["CPUExecutionProvider"]

        def get_inputs(self):
            return [FakeInput()]

        def run(self, outputs, inputs):
            assert outputs is None
            assert inputs["observations"].shape == (1, 61)
            return [np.zeros((1, 14), dtype=np.float32)]

    class FakeOrt:
        InferenceSession = FakeSession

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def is_running(self):
            return observed["steps"] < 2

        def sync(self):
            observed["syncs"] += 1

    class FakeViewerModule:
        @staticmethod
        def launch_passive(model, data):
            assert model == "model" and data == "data"
            return FakeViewer()

    class FakeEnv:
        model = "model"
        data = "data"
        action_space = SimpleNamespace(shape=(14,))

        def reset(self, seed):
            assert seed == 3
            observed["resets"] += 1
            return np.zeros(61, dtype=np.float32), {}

        def step(self, action):
            assert action.shape == (14,)
            observed["steps"] += 1
            return np.zeros(61, dtype=np.float32), 0.0, observed["steps"] == 1, False, {}

    real_import = module.importlib.import_module

    def fake_import(name):
        if name == "microduck_local.contract":
            return SimpleNamespace(CTRL_DT=0.0)
        return real_import(name)

    monkeypatch.setattr(module.importlib, "import_module", fake_import)
    module._run_view_loop(
        policy,
        FakeEnv(),
        seed=3,
        viewer_module=FakeViewerModule,
        ort_module=FakeOrt,
    )
    assert observed == {"resets": 2, "steps": 2, "syncs": 2}


def test_view_missing_source_raises(tmp_path):
    module = load_dance_example()
    args = module.parse_args(["view", "--policy", str(tmp_path / "missing.onnx")])
    with pytest.raises(FileNotFoundError, match="policy does not exist"):
        module.view(args)


def test_view_loop_rejects_non_finite_actions(monkeypatch, tmp_path):
    module = load_dance_example()
    policy = tmp_path / "invalid.onnx"
    policy.write_bytes(b"onnx")

    class FakeSession:
        def __init__(self, path, providers):
            pass

        def get_inputs(self):
            return [SimpleNamespace(name="observations")]

        def run(self, outputs, inputs):
            return [np.full((1, 14), np.nan, dtype=np.float32)]

    class FakeViewer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def is_running(self):
            return True

        def sync(self):
            raise AssertionError("invalid actions must fail before sync")

    class FakeEnv:
        model = object()
        data = object()
        action_space = SimpleNamespace(shape=(14,))

        def reset(self, seed):
            return np.zeros(61, dtype=np.float32), {}

        def step(self, action):
            raise AssertionError("invalid actions must not reach the simulator")

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda name: SimpleNamespace(CTRL_DT=0.02),
    )
    with pytest.raises(RuntimeError, match="non-finite actions"):
        module._run_view_loop(
            policy,
            FakeEnv(),
            seed=1,
            viewer_module=SimpleNamespace(
                launch_passive=lambda model, data: FakeViewer()
            ),
            ort_module=SimpleNamespace(InferenceSession=FakeSession),
        )
