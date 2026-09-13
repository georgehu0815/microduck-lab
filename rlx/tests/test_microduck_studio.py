import importlib.util
import inspect
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

import rlx.environments.microduck_recipes as recipe_module
from rlx.environments.microduck_recipes import (
    RECIPES,
    default_stilt_mass_kg,
    get_recipe,
    make_recipe_env,
    make_single_recipe_env,
    recipe_metrics,
    validate_reward_weights,
    validate_stilt_options,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples" / "ppo_microduck_studio.py"


def write_v3_dance_clip(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "name": "Custom v3 dance",
                "duration": 0.04,
                "loop": True,
                "keys": [
                    {
                        "t": 0.0,
                        "joints": [0.0] * 14,
                        "rootPitch": 0.0,
                        "rootYaw": 0.2,
                        "rootRoll": -0.1,
                        "rootPosition": [0.0, 0.0, 0.12],
                    },
                    {
                        "t": 0.02,
                        "joints": [0.1] * 14,
                        "rootPitch": 0.05,
                        "rootYaw": 0.3,
                        "rootRoll": -0.2,
                        "rootPosition": [0.01, 0.0, 0.13],
                    },
                    {
                        "t": 0.04,
                        "joints": [0.0] * 14,
                        "rootPitch": 0.0,
                        "rootYaw": 0.2,
                        "rootRoll": -0.1,
                        "rootPosition": [0.0, 0.0, 0.12],
                    },
                ],
            }
        )
    )
    return path


def load_studio_example():
    spec = importlib.util.spec_from_file_location("ppo_microduck_studio", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recipe_registry_has_the_seven_studio_recipes():
    assert tuple(sorted(RECIPES)) == ("backflip", "basketball", "bridge", "dance", "running", "stilts", "swing")
    assert get_recipe("running").behavior_id == "run"
    assert get_recipe("swing").behavior_id is None
    with pytest.raises(ValueError, match="unknown MicroDuck recipe"):
        get_recipe("moonwalk")


def test_bridge_evaluation_requires_complete_horizons():
    module = load_studio_example()
    with pytest.raises(SystemExit):
        module.parse_args(["eval", "--recipe", "bridge", "--backend", "dummy", "--eval-steps", "1200"])
    assert module.parse_args(["eval", "--recipe", "bridge", "--backend", "dummy", "--eval-steps", "2000"]).eval_steps == 2000


def test_bridge_factory_applies_requested_spawn_randomization():
    env = make_single_recipe_env("bridge", random_yaw=True, domain_rand=False, obs_noise=False, action_delay=False)
    try:
        assert env.unwrapped._bridge_random_yaw is True
        assert env.unwrapped.action_delay is False
    finally:
        env.close()


def test_stilt_option_normalization_and_validation():
    assert validate_stilt_options(10, 0.5, None) == (
        10.0,
        0.5,
        default_stilt_mass_kg(10.0),
    )
    assert validate_stilt_options(2, 0, 0.02) == (2.0, 0.0, 0.02)
    with pytest.raises(ValueError, match="height"):
        validate_stilt_options(0.2, 0.5, None)
    with pytest.raises(ValueError, match="blend"):
        validate_stilt_options(10, 1.1, None)
    with pytest.raises(ValueError, match="mass"):
        validate_stilt_options(10, 0.5, 0)


def test_reward_weight_validation_is_recipe_specific_and_finite():
    assert validate_reward_weights("swing", {"swing_height": 12}) == {
        "swing_height": 12.0
    }
    with pytest.raises(ValueError, match="unknown swing reward"):
        validate_reward_weights("swing", {"pose_match": 4})
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_reward_weights("dance", {"pose_match": -1})
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_reward_weights("running", {"keep_pace": float("nan")})


def test_dance_recipe_loads_explicit_v3_clip_path_without_fallback(tmp_path):
    clip_path = write_v3_dance_clip(tmp_path / "custom.clip.json")
    env = make_single_recipe_env(
        "dance",
        dance_clip=str(clip_path),
        seed=3,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
    )
    try:
        assert env.unwrapped.clip.name == "Custom v3 dance"
        assert env.unwrapped.clip.steps == 2
        np.testing.assert_allclose(env.unwrapped.clip.at(1)[0], 0.1)
    finally:
        env.close()


def test_dance_recipe_keeps_bundled_120bpm_clip_by_default():
    env = make_single_recipe_env(
        "dance",
        seed=3,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
    )
    try:
        assert env.unwrapped.clip.name == "dance-120bpm"
    finally:
        env.close()


def test_dance_recipe_rejects_missing_custom_clip_instead_of_using_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_single_recipe_env("dance", dance_clip=tmp_path / "missing.clip.json")


def test_dance_clip_argument_is_rejected_for_other_recipes(tmp_path):
    clip_path = write_v3_dance_clip(tmp_path / "custom.clip.json")
    with pytest.raises(ValueError, match="only valid for the dance recipe"):
        make_single_recipe_env("running", dance_clip=clip_path)
    with pytest.raises(ValueError, match="only valid for the dance recipe"):
        make_recipe_env("running", dance_clip=clip_path)


def test_dance_pose_sigma_factory_defaults_are_opt_in():
    assert (
        inspect.signature(make_single_recipe_env)
        .parameters["dance_pose_sigma"]
        .default
        is None
    )
    assert (
        inspect.signature(make_recipe_env).parameters["dance_pose_sigma"].default
        is None
    )


@pytest.mark.parametrize(
    "invalid_sigma",
    [0.0, -0.1, float("nan"), float("inf"), float("-inf")],
)
def test_dance_pose_sigma_must_be_finite_and_positive(invalid_sigma):
    with pytest.raises(ValueError, match="finite and positive"):
        make_single_recipe_env("dance", dance_pose_sigma=invalid_sigma)
    with pytest.raises(ValueError, match="finite and positive"):
        make_recipe_env("dance", dance_pose_sigma=invalid_sigma)


@pytest.mark.parametrize("recipe", ["running", "stilts", "swing"])
def test_dance_pose_sigma_is_rejected_for_other_recipes(recipe):
    with pytest.raises(ValueError, match="only valid for the dance recipe"):
        make_single_recipe_env(recipe, dance_pose_sigma=0.15)
    with pytest.raises(ValueError, match="only valid for the dance recipe"):
        make_recipe_env(recipe, dance_pose_sigma=0.15)


def test_per_joint_dance_pose_match_formula_bounds_and_dynamic_ranking():
    references = np.array(
        [
            [-0.3, -0.15, 0.0, 0.15, 0.3],
            [-0.15, 0.0, 0.15, 0.3, -0.3],
            [0.0, 0.15, 0.3, -0.3, -0.15],
            [0.15, 0.3, -0.3, -0.15, 0.0],
        ],
        dtype=float,
    )

    class FakeClip:
        def at(self, step):
            return references[step], 0.0

    env = SimpleNamespace(clip=FakeClip(), step_count=0)
    sigma = 0.15
    tracked_scores = []
    constant_scores = []
    constant_pose = references.mean(axis=0)
    for step, target in enumerate(references):
        env.step_count = step
        env._joint_qpos = lambda target=target: target.copy()
        tracked_scores.append(
            recipe_module._per_joint_dance_pose_match(env, sigma=sigma)
        )
        env._joint_qpos = lambda: constant_pose.copy()
        constant_scores.append(
            recipe_module._per_joint_dance_pose_match(env, sigma=sigma)
        )

    assert tracked_scores == pytest.approx([1.0] * len(references))
    assert all(0.0 <= score <= 1.0 for score in constant_scores)
    assert np.mean(tracked_scores) > np.mean(constant_scores)

    env.step_count = 0
    current = references[0] + np.array([0.0, sigma, -sigma, 2 * sigma, -2 * sigma])
    env._joint_qpos = lambda: current
    expected = np.mean(
        np.exp(-np.square((current - references[0]) / sigma))
    )
    assert recipe_module._per_joint_dance_pose_match(
        env, sigma=sigma
    ) == pytest.approx(expected)


def test_dance_pose_sigma_substitution_is_instance_local_and_preserves_term_row():
    env_kwargs = {
        "seed": 3,
        "domain_rand": False,
        "obs_noise": False,
        "action_delay": False,
        "random_yaw": False,
        "max_episode_s": 0.1,
    }
    default_env = make_single_recipe_env("dance", **env_kwargs)
    custom_env = make_single_recipe_env(
        "dance", dance_pose_sigma=0.15, **env_kwargs
    )
    fresh_default_env = make_single_recipe_env("dance", **env_kwargs)
    try:
        default_env.reset(seed=3)
        custom_env.reset(seed=3)
        fresh_default_env.reset(seed=3)

        default_row = next(
            row for row in default_env.unwrapped._term_rows if row[0] == "pose_match"
        )
        custom_row = next(
            row for row in custom_env.unwrapped._term_rows if row[0] == "pose_match"
        )
        fresh_default_row = next(
            row
            for row in fresh_default_env.unwrapped._term_rows
            if row[0] == "pose_match"
        )

        assert custom_row[:3] == default_row[:3]
        assert fresh_default_row[3] is default_row[3]
        assert custom_row[3] is not default_row[3]

        target, _ = default_env.unwrapped.clip.at(default_env.unwrapped.step_count)
        current = default_env.unwrapped._joint_qpos()
        squared_error_sum = float(np.square(current - target).sum())
        expected_default = 0.5 * np.exp(
            -squared_error_sum / 6.0**2
        ) + 0.5 * np.exp(-squared_error_sum / 2.0**2)
        assert default_row[3](default_env.unwrapped) == pytest.approx(
            expected_default
        )

        custom_target, _ = custom_env.unwrapped.clip.at(
            custom_env.unwrapped.step_count
        )
        custom_current = custom_env.unwrapped._joint_qpos()
        expected_custom = np.mean(
            np.exp(-np.square((custom_current - custom_target) / 0.15))
        )
        assert custom_row[3](custom_env.unwrapped) == pytest.approx(expected_custom)
        _, custom_terms = custom_env.unwrapped._compute_reward()
        assert custom_terms[custom_row[1]] == pytest.approx(
            custom_row[2] * expected_custom
        )
    finally:
        default_env.close()
        custom_env.close()
        fresh_default_env.close()


def test_vector_dance_recipe_uses_explicit_v3_clip_path(tmp_path):
    clip_path = write_v3_dance_clip(tmp_path / "vector.clip.json")
    env = make_recipe_env(
        "dance",
        num_envs=1,
        backend="dummy",
        dance_clip=clip_path,
        dance_pose_sigma=0.2,
        seed=5,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
    )
    try:
        dance_env = env.envs.envs[0].unwrapped
        assert dance_env.clip.name == "Custom v3 dance"
        pose_row = next(row for row in dance_env._term_rows if row[0] == "pose_match")
        assert pose_row[3].keywords == {"sigma": 0.2}
    finally:
        env.close()


def test_dance_recipe_metrics_include_pose_and_dynamic_tracking_errors():
    class FakeClip:
        steps = 4
        duration = 0.08

        def at(self, step):
            return np.full(14, step * 0.02), 0.0

    env = SimpleNamespace(
        clip=FakeClip(),
        step_count=2,
        _joint_qpos=lambda: np.full(14, 0.03),
        _joint_vel=lambda: np.full(14, 0.75),
        _projected_gravity=lambda: np.array([0.0, 0.0, -0.8]),
        _trunk_xpos=np.array([0.0, 0.0, 0.22]),
    )

    metrics = recipe_metrics(env, "dance")

    assert metrics == pytest.approx(
        {
            "upright": 0.8,
            "height_m": 0.22,
            "pose_rmse_rad": 0.01,
            "pose_velocity_rmse_rad_s": 0.25,
        }
    )


def test_recipe_artifact_names_and_argument_defaults(tmp_path):
    module = load_studio_example()
    for recipe in RECIPES:
        paths = module.artifact_paths(recipe, output_dir=tmp_path / recipe)
        assert paths.checkpoint.name == f"{recipe}.safetensors"
        assert paths.metadata.name == f"{recipe}.safetensors.json"
        assert paths.onnx.name == f"{recipe}.onnx"

    args = module.parse_args(
        [
            "train",
            "--recipe",
            "stilts",
            "--output-dir",
            str(tmp_path),
            "--stilt-height-cm",
            "12",
            "--stilt-blend",
            "0.75",
            "--stilt-mass-kg",
            "0.04",
        ]
    )
    assert args.stilt_height_cm == 12
    assert args.stilt_blend == 0.75
    assert args.stilt_mass_kg == 0.04
    swing_args = module.parse_args(["train", "--recipe", "swing"])
    assert swing_args.learning_rate == 1e-4
    assert swing_args.gamma == 0.995
    assert swing_args.clip_coefficient == 0.1
    assert swing_args.update_epochs == 3
    assert swing_args.entropy_coefficient == 0.002
    assert swing_args.max_grad_norm == 1.0
    assert swing_args.swing_initial_angle_deg == 0
    assert swing_args.swing_initial_rate_rad_s == 0
    assert swing_args.init_from is None
    render_args = module.parse_args(
        [
            "render",
            "--recipe",
            "dance",
            "--policy",
            "/tmp/dance.onnx",
            "--output",
            str(tmp_path / "render"),
            "--render-seconds",
            "120",
        ]
    )
    assert render_args.render_seconds == 120
    assert render_args.max_episode_s is None
    export_args = module.parse_args(
        [
            "export",
            "--recipe",
            "swing",
            "--checkpoint",
            str(tmp_path / "swing.safetensors"),
        ]
    )
    assert export_args.recipe == "swing"
    with pytest.raises(SystemExit):
        module.parse_args(["train", "--recipe", "stilts", "--stilt-blend", "1.5"])
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "train",
                "--recipe",
                "swing",
                "--weight-overrides",
                '{"pose_match": 4}',
            ]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            ["train", "--recipe", "swing", "--swing-initial-angle-deg", "31"]
        )
    with pytest.raises(SystemExit):
        module.parse_args(
            ["train", "--recipe", "swing", "--swing-initial-rate-rad-s", "1.1"]
        )


def test_swing_reset_can_start_with_bounded_curriculum_motion():
    env = make_single_recipe_env(
        "swing",
        seed=7,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
        swing_initial_angle_deg=12,
        swing_initial_rate_rad_s=0.35,
    )
    try:
        observation, _ = env.reset(seed=7)
        metrics = env.unwrapped.recipe_metrics()
        assert observation.shape == (61,)
        assert np.isfinite(observation).all()
        assert abs(metrics["swing_angle_deg"]) <= 12.1
        assert abs(metrics["swing_rate_rad_s"]) <= 0.36
        assert metrics["valid_geometry"] == 1
        _, terms = env.unwrapped._compute_reward()
        assert terms["swing_peak_progress"] == pytest.approx(0.0, abs=1e-12)
        assert terms["swing_height"] == pytest.approx(0.0, abs=1e-12)
        assert terms["swing_energy"] == pytest.approx(0.0, abs=1e-12)
    finally:
        env.close()


@pytest.mark.parametrize(
    ("recipe", "term", "kwargs"),
    [
        ("swing", "action_rate_penalty", {}),
        (
            "stilts",
            "upright",
            {
                "stilt_height_cm": 10.0,
                "stilt_blend": 0.5,
                "stilt_mass_kg": 0.03,
            },
        ),
    ],
)
def test_native_recipe_reward_override_can_disable_one_term(recipe, term, kwargs):
    default_env = make_single_recipe_env(
        recipe,
        seed=4,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
        **kwargs,
    )
    custom_env = make_single_recipe_env(
        recipe,
        seed=4,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
        weight_overrides={term: 0.0},
        **kwargs,
    )
    try:
        default_env.reset(seed=4)
        custom_env.reset(seed=4)
        if recipe == "swing":
            default_env.unwrapped._policy_last_action[:] = 1.0
            custom_env.unwrapped._policy_last_action[:] = 1.0
        _, default_terms = default_env.unwrapped._compute_reward()
        _, custom_terms = custom_env.unwrapped._compute_reward()
        assert default_terms[term] != 0.0
        assert custom_terms[term] == 0.0
    finally:
        default_env.close()
        custom_env.close()


def test_metadata_records_recipe_reward_overrides():
    module = load_studio_example()
    args = module.parse_args(
        [
            "train",
            "--recipe",
            "swing",
            "--weight-overrides",
            '{"swing_height": 12}',
        ]
    )
    assert module._metadata(args, 24)["reward_weights"] == {"swing_height": 12.0}


@pytest.mark.parametrize("recipe", ["dance", "running", "swing", "stilts"])
def test_single_recipe_environment_reset_and_step_are_finite(recipe):
    kwargs = {}
    if recipe == "stilts":
        kwargs = {
            "stilt_height_cm": 10.0,
            "stilt_blend": 0.5,
            "stilt_mass_kg": 0.03,
        }
    env = make_single_recipe_env(
        recipe,
        seed=3,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        max_episode_s=0.1,
        **kwargs,
    )
    try:
        observation, _ = env.reset(seed=3)
        assert observation.shape == (61,)
        assert np.isfinite(observation).all()
        next_observation, reward, terminated, truncated, info = env.step(
            np.zeros(14, dtype=np.float32)
        )
        assert next_observation.shape == (61,)
        assert np.isfinite(next_observation).all()
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info["recipe_metrics"]
        assert all(np.isfinite(value) for value in info["recipe_metrics"].values())
    finally:
        env.close()


def test_training_progress_json_matches_studio_protocol(capsys):
    module = load_studio_example()
    callback = module._training_progress_callback(4, 1, 2)
    callback(
        {
            "_episode": np.array([True]),
            "episode": {"r": np.array([2.5]), "l": np.array([1])},
        },
        0,
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "event": "training_episode",
        "steps": 1,
        "total": 4,
        "mean_reward": 2.5,
        "episodes": 1,
    }


@pytest.mark.parametrize(
    ("recipe", "normalize_rewards"),
    [("swing", True), ("dance", False), ("running", False), ("stilts", False)],
)
def test_training_environment_normalizes_only_swing_rewards(
    monkeypatch, recipe, normalize_rewards
):
    module = load_studio_example()
    captured = {}
    sentinel = object()

    def fake_make_recipe_env(name, **kwargs):
        captured.update(recipe=name, **kwargs)
        return sentinel

    monkeypatch.setattr(module, "make_recipe_env", fake_make_recipe_env)
    args = module.parse_args(["train", "--recipe", recipe])

    assert module._environment(args, normalize=True) is sentinel
    assert captured["recipe"] == recipe
    assert captured["normalize_observations"] is True
    assert captured["normalize_rewards"] is normalize_rewards
    assert captured["gamma"] == args.gamma


@pytest.mark.parametrize("command", ["eval", "render"])
def test_non_training_environment_uses_recipe_gamma(monkeypatch, command, tmp_path):
    module = load_studio_example()
    captured = {}
    sentinel = object()

    def fake_make_recipe_env(name, **kwargs):
        captured.update(recipe=name, **kwargs)
        return sentinel

    monkeypatch.setattr(module, "make_recipe_env", fake_make_recipe_env)
    argv = [command, "--recipe", "swing", "--policy", str(tmp_path / "swing.onnx")]
    if command == "render":
        argv.extend(["--output", str(tmp_path / "render")])
    args = module.parse_args(argv)

    assert not hasattr(args, "gamma")
    assert module._environment(args, normalize=False) is sentinel
    assert captured["normalize_observations"] is False
    assert captured["normalize_rewards"] is False
    assert captured["gamma"] == module.RECIPE_PPO_DEFAULTS["swing"]["gamma"]


def test_rollout_progress_updates_reward_history_before_episode_end(capsys):
    module = load_studio_example()
    algorithm = SimpleNamespace(
        step=8,
        buffer=SimpleNamespace(
            rewards=np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        ),
    )
    observer = module._training_rollout_observer(20, algorithm)
    observer({"phase": "update", "steps": 4})
    assert capsys.readouterr().out == ""
    observer({"phase": "collection", "steps": 4})
    assert json.loads(capsys.readouterr().out) == {
        "event": "training_progress",
        "source": "rollout",
        "steps": 8,
        "total": 20,
        "mean_reward": 4.0,
    }


def test_swing_evaluation_reports_per_environment_peak_to_peak_span(
    monkeypatch, tmp_path
):
    module = load_studio_example()

    class FakeEnv:
        def reset(self, key):
            return np.zeros((2, 61), dtype=np.float32), {}, {}

        def step(self, key, state, action):
            step = getattr(self, "step_index", 0)
            self.step_index = step + 1
            angles = ((-10.0, 20.0), (30.0, -20.0))[step]
            infos = [
                {"recipe_metrics": {"swing_angle_deg": angle}}
                for angle in angles
            ]
            return (
                np.zeros((2, 61), dtype=np.float32),
                {},
                np.ones(2, dtype=np.float32),
                np.zeros(2, dtype=bool),
                np.zeros(2, dtype=bool),
                {
                    "infos": infos,
                    "_episode": np.zeros(2, dtype=bool),
                    "episode": {
                        "r": np.zeros(2, dtype=np.float32),
                        "l": np.zeros(2, dtype=np.int64),
                    },
                },
            )

        def close(self):
            pass

    policy = tmp_path / "swing.onnx"
    policy.touch()
    monkeypatch.setattr(module, "_environment", lambda args, normalize: FakeEnv())
    monkeypatch.setattr(
        module,
        "_policy_from_onnx",
        lambda path: lambda observation: np.zeros((2, 14), dtype=np.float32),
    )
    args = module.parse_args(
        [
            "eval",
            "--recipe",
            "swing",
            "--policy",
            str(policy),
            "--num-envs",
            "2",
            "--eval-steps",
            "2",
        ]
    )

    result = module.evaluate(args)

    assert result["recipe_metrics"]["swing_span_deg"] == {
        "mean": 40.0,
        "median": 40.0,
        "min": 40.0,
        "max": 40.0,
    }


def test_running_train_smoke_command_writes_recipe_named_artifacts(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "train",
            "--recipe",
            "running",
            "--output-dir",
            str(tmp_path),
            "--num-envs",
            "1",
            "--backend",
            "dummy",
            "--num-steps",
            "1",
            "--num-minibatches",
            "1",
            "--update-epochs",
            "1",
            "--total-timesteps",
            "1",
            "--max-episode-s",
            "0.1",
            "--no-domain-rand",
            "--no-obs-noise",
            "--no-action-delay",
            "--no-random-yaw",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 and "No Metal device available" in result.stderr:
        pytest.skip("MLX Metal is unavailable in this test subprocess")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["recipe"] == "running"
    assert Path(payload["checkpoint"]).name == "running.safetensors"
    assert Path(payload["metadata"]).name == "running.safetensors.json"
    assert Path(payload["onnx"]).name == "running.onnx"
    assert (tmp_path / "running.safetensors").is_file()
    assert (tmp_path / "running.safetensors.json").is_file()
    assert (tmp_path / "running.onnx").is_file()
