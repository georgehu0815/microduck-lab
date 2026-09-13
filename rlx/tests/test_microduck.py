import importlib

import gymnasium as gym
import numpy as np
import pytest

from rlx.environments.microduck import MicroDuckVecEnv, make_microduck_env


class FakeVecEnv:
    def __init__(self, observation_dim=61, action_dim=14, num_envs=3):
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            -4.0, 4.0, (action_dim,), dtype=np.float32
        )
        self.closed = False
        self.actions = None
        self._step = 0

    def reset(self):
        return np.arange(self.num_envs * 61, dtype=np.float64).reshape(
            self.num_envs, 61
        )

    def step(self, actions):
        self.actions = actions
        self._step += 1
        observation = np.full((self.num_envs, 61), self._step, dtype=np.float64)
        reward = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        done = np.array([True, True, False])
        infos = [
            {"episode": {"r": 11.0, "l": 7}, "terminal_observation": np.full(61, 4., dtype=np.float32)},
            {"TimeLimit.truncated": True, "terminal_observation": np.full(61, 8., dtype=np.float32)},
            {},
        ]
        return observation, reward, done, infos

    def close(self):
        self.closed = True


def test_reset_ignores_rng_key_and_returns_float32_mlx_batch():
    env = MicroDuckVecEnv(FakeVecEnv())
    mx = importlib.import_module("mlx.core")
    observation, state, info = env.reset(mx.random.key(123))

    assert observation.shape == (3, 61)
    assert observation.dtype == mx.float32
    assert state == {}
    assert info == {}


def test_step_converts_actions_and_splits_termination_from_truncation():
    inner = FakeVecEnv()
    env = MicroDuckVecEnv(inner)
    mx = importlib.import_module("mlx.core")
    env.reset(None)
    observation, state, reward, terminated, truncated, info = env.step(
        mx.random.key(1), {"ignored": mx.array(1)}, mx.zeros((3, 14))
    )

    assert observation.shape == (3, 61)
    assert observation.dtype == mx.float32
    assert reward.dtype == mx.float32
    assert state == {}
    np.testing.assert_array_equal(np.asarray(terminated), [True, False, False])
    np.testing.assert_array_equal(np.asarray(truncated), [False, True, False])
    assert inner.actions.dtype == np.float32
    assert inner.actions.shape == (3, 14)
    np.testing.assert_array_equal(info["infos"][1]["terminal_observation"], np.full(61, 8.))


def test_step_rejects_non_finite_actions_before_calling_inner_environment():
    inner = FakeVecEnv()
    env = MicroDuckVecEnv(inner)
    mx = importlib.import_module("mlx.core")
    env.reset(None)

    actions = np.zeros((3, 14), dtype=np.float32)
    actions[1, 4] = np.nan
    with pytest.raises(RuntimeError, match=r"actions contain non-finite.*\[1, 4\]"):
        env.step(None, {}, mx.array(actions))

    assert inner.actions is None


def test_step_clips_finite_actions_to_declared_action_space():
    inner = FakeVecEnv()
    env = MicroDuckVecEnv(inner)
    mx = importlib.import_module("mlx.core")
    env.reset(None)

    env.step(None, {}, mx.full((3, 14), 100.0))

    np.testing.assert_array_equal(inner.actions, np.full((3, 14), 4.0))


def test_running_normalization_rejects_non_finite_observations():
    inner = FakeVecEnv()
    env = MicroDuckVecEnv(inner, normalize_observations=True)
    env.reset(None)
    observation = np.zeros((3, 61), dtype=np.float32)
    observation[2, 9] = np.inf

    with pytest.raises(RuntimeError, match="running statistics require.*finite"):
        env._normalize_observation(observation)


def test_autoreset_episode_statistics_match_logger_shape_and_passthrough():
    env = MicroDuckVecEnv(FakeVecEnv())
    mx = importlib.import_module("mlx.core")
    env.reset(None)
    *_, info = env.step(None, {}, mx.zeros((3, 14)))

    np.testing.assert_array_equal(info["_episode"], [True, True, False])
    np.testing.assert_allclose(info["episode"]["r"], [11.0, 2.0, 3.0])
    np.testing.assert_array_equal(info["episode"]["l"], [7, 1, 1])

    *_, second_info = env.step(None, {}, mx.zeros((3, 14)))
    np.testing.assert_allclose(second_info["episode"]["r"], [11.0, 2.0, 6.0])
    np.testing.assert_array_equal(second_info["episode"]["l"], [7, 1, 2])


def test_observation_and_reward_normalization_are_finite_and_clipped():
    env = MicroDuckVecEnv(
        FakeVecEnv(),
        normalize_observations=True,
        normalize_rewards=True,
        clip=1.0,
    )
    mx = importlib.import_module("mlx.core")
    observation, _, _ = env.reset(None)
    _, _, reward, *_ = env.step(None, {}, mx.zeros((3, 14)))

    assert np.isfinite(np.asarray(observation)).all()
    assert np.isfinite(np.asarray(reward)).all()
    assert np.max(np.abs(np.asarray(observation))) <= 1.0
    assert np.max(np.abs(np.asarray(reward))) <= 1.0
    assert env.observation_rms.count > env.num_envs
    assert env.return_rms.count > env.num_envs


def test_close_is_idempotent_and_context_manager_closes():
    inner = FakeVecEnv()
    with MicroDuckVecEnv(inner) as env:
        assert env is not None
    assert inner.closed


def test_calibrated_reward_scale_stays_fixed():
    env = MicroDuckVecEnv(FakeVecEnv(), normalize_rewards=True, freeze_reward_normalization=True)
    env.return_rms.var = np.array(4.)
    env.return_rms.count = 9600
    mx = importlib.import_module("mlx.core")
    env.reset(None)
    _, _, reward, *_ = env.step(None, {}, mx.zeros((3, 14)))
    np.testing.assert_allclose(np.asarray(reward), [0.5, 1., 1.5])
    assert env.return_rms.var == 4.
    assert env.return_rms.count == 9600
    env.close()


@pytest.mark.parametrize(
    ("observation_dim", "action_dim", "message"),
    [(60, 14, "observation space"), (61, 13, "action space")],
)
def test_dimension_validation(observation_dim, action_dim, message):
    with pytest.raises(ValueError, match=message):
        MicroDuckVecEnv(FakeVecEnv(observation_dim, action_dim))


def test_missing_microduck_dependency_has_actionable_error(monkeypatch):
    real_import_module = importlib.import_module

    def missing(name, package=None):
        if name.startswith("microduck_local"):
            error = ModuleNotFoundError("No module named 'microduck_local'")
            error.name = "microduck_local"
            raise error
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(ImportError, match="uv pip install -e ../microduck_local"):
        make_microduck_env(num_envs=1)


def test_behavior_factory_options_and_clip_environment_restoration(monkeypatch, tmp_path):
    import os
    import rlx.environments.microduck as microduck

    captured = []

    class FakeBehaviorEnv:
        def __init__(self, **kwargs):
            captured.append((kwargs, os.environ.get("MICRODUCK_CLIPS_DIR")))

    class Behaviors:
        BEHAVIORS = {"imitate": object()}
        BehaviorEnv = FakeBehaviorEnv

    class Walk:
        class MicroduckWalkEnv:
            pass

    class Vec:
        @staticmethod
        def make_vec_env(factories, backend=None):
            assert backend == "fork"
            for factory in factories:
                factory()
            return FakeVecEnv(num_envs=2)

    monkeypatch.setattr(microduck, "_import_microduck_modules",
                        lambda: (Vec, Walk, Behaviors))
    monkeypatch.setenv("MICRODUCK_CLIPS_DIR", "original")
    env = make_microduck_env(
        num_envs=2, backend="fork", behavior_id="imitate",
        clip_name="dance", clip_directory=tmp_path,
        weight_overrides={"pose_match": 7.0},
    )
    assert os.environ["MICRODUCK_CLIPS_DIR"] == "original"
    assert len(captured) == 2
    assert all(value == str(tmp_path) for _, value in captured)
    assert captured[0][0]["behavior_id"] == "imitate"
    assert captured[0][0]["clip_name"] == "dance"
    assert captured[0][0]["weight_overrides"] == {"pose_match": 7.0}
    env.close()


def test_clip_environment_restored_when_vector_creation_fails(monkeypatch, tmp_path):
    import os
    import rlx.environments.microduck as microduck

    class Behaviors:
        BEHAVIORS = {"imitate": object()}
        class BehaviorEnv:
            pass

    class Vec:
        @staticmethod
        def make_vec_env(factories, backend=None):
            assert os.environ["MICRODUCK_CLIPS_DIR"] == str(tmp_path)
            raise RuntimeError("worker failed")

    monkeypatch.setattr(microduck, "_import_microduck_modules",
                        lambda: (Vec, object(), Behaviors))
    monkeypatch.delenv("MICRODUCK_CLIPS_DIR", raising=False)
    with pytest.raises(RuntimeError, match="worker failed"):
        make_microduck_env(num_envs=1, behavior_id="imitate",
                           clip_directory=tmp_path)
    assert "MICRODUCK_CLIPS_DIR" not in os.environ
