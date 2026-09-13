"""Contract and honesty checks for the local basketball task."""

import math

import mujoco
import numpy as np
import pytest

from rlx.environments.basketball import (
    BALL_MASS, BALL_RADIUS, HOLD_LEVELS, REWARD_WEIGHTS,
    BasketballEnv, basketball_model, curriculum_level,
)


@pytest.fixture(scope="module")
def model():
    return basketball_model("xml")


@pytest.fixture
def env(model):
    instance = BasketballEnv(model=model, actuator="xml", command=(0, 0, 0), random_yaw=False)
    instance.reset(seed=101)
    yield instance
    instance.close()


def test_reference_geometry_and_free_ball(env):
    assert env.model.nq == 28
    assert env.model.nv == 26
    assert env.model.nu == 14
    assert env.model.body_mass[env.ball_body_id] == pytest.approx(BALL_MASS)
    assert env.model.geom_size[env.ball_geom_id, 0] == pytest.approx(BALL_RADIUS)
    assert env.model.joint("basketball_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.geom_condim[env.ball_geom_id] == 4
    np.testing.assert_allclose(env.model.geom_friction[env.ball_geom_id], [1.2, .01, .001])


def test_actor_contract_has_no_ball_state(env):
    initial, _ = env.reset(seed=202)
    env.data.qpos[env.ball_qpos_adr] += .03
    mujoco.mj_forward(env.model, env.data)
    changed = env._get_obs()
    assert initial.shape == (61,)
    assert initial.dtype == np.float32
    np.testing.assert_array_equal(initial, changed)
    np.testing.assert_array_equal(changed[51:61], np.zeros(10))
    env.set_command((.08, -.02, .1))
    np.testing.assert_allclose(env._get_obs()[48:51], [.08, -.02, .1])


def test_reset_is_seeded_and_restores_ball(env):
    first_obs, first_info = env.reset(seed=303)
    env.step(np.zeros(14))
    second_obs, second_info = env.reset(seed=303)
    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_info == second_info
    assert second_info["root_above_ball_m"] == pytest.approx(.247)
    assert second_info["ball_position"] == [0, 0, .121]
    assert second_info["hold"] == 0


def test_zero_hold_removes_all_assistance(env):
    env.hold = 1
    env.data.qpos[env.ball_qpos_adr] += .02
    mujoco.mj_forward(env.model, env.data)
    env._apply_assistance()
    assert np.linalg.norm(env.data.xfrc_applied) > 0
    env.hold = 0
    env._apply_assistance()
    np.testing.assert_array_equal(env.data.xfrc_applied, np.zeros_like(env.data.xfrc_applied))


def test_assistance_is_reported_not_hidden(env):
    env.hold = .25
    _, reward, _, _, info = env.step(np.zeros(14))
    assert math.isfinite(reward)
    assert info["hold"] == .25


def test_ball_rotational_assistance_uses_world_torque(env):
    env.hold = 1
    env.data.qpos[env.ball_qpos_adr + 3:env.ball_qpos_adr + 7] = [2 ** -.5, 0, 0, 2 ** -.5]
    env.data.qvel[env.ball_qvel_adr + 3:env.ball_qvel_adr + 6] = [1, 0, 0]
    mujoco.mj_forward(env.model, env.data)
    env._apply_assistance()
    np.testing.assert_allclose(env.data.xfrc_applied[env.ball_body_id, 3:], [0, -.2, 0], atol=1e-12)


def test_curriculum_and_fixed_rewards(env):
    assert HOLD_LEVELS == (1, .5, .25, .1, .03, 0)
    assert curriculum_level(0, 6) == 1
    assert curriculum_level(5, 60) == 5
    assert curriculum_level(3, 1.49) == 2
    assert curriculum_level(0, .2) == 0
    assert curriculum_level(3, 3) == 3
    initial_reward, initial_terms = env._compute_reward()
    env.hold = 1
    reward, terms = env._compute_reward()
    assert reward == initial_reward
    assert terms == initial_terms
    with pytest.raises(TypeError):
        REWARD_WEIGHTS["upright"] = 2
    assert all(value <= 0 for name, value in terms.items() if name.endswith("penalty"))


def test_curriculum_promotes_only_completed_episode(env):
    env.curriculum = True
    env.stage = 0
    env.hold = 1
    env.step_count = 350
    env.reset(seed=1)
    assert env.stage == 0
    env.step_count = 350
    env._episode_finished = True
    env.reset(seed=1)
    assert env.stage == 1
    assert env.hold == .5


def test_first_fall_terminates_and_requires_reset(env):
    env.data.qpos[0] = .3
    mujoco.mj_forward(env.model, env.data)
    _, _, terminated, _, info = env.step(np.zeros(14))
    assert terminated
    assert "offset" in info["termination_reasons"]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(14))


def test_foot_contacts_do_not_count_floor_as_ball(env):
    env.data.qpos[2] = .09
    env.data.qpos[env.ball_qpos_adr] = 1
    mujoco.mj_forward(env.model, env.data)
    metrics = env.metrics()
    assert metrics["foot_ball_contacts"] == 0
    assert metrics["robot_floor_contact"]


@pytest.mark.parametrize("action", [np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)])
def test_rejects_invalid_actions(env, action):
    with pytest.raises(ValueError, match="actions"):
        env.step(action)


@pytest.mark.parametrize("hold", [-1, 1.1, float("nan")])
def test_rejects_invalid_hold(hold):
    with pytest.raises(ValueError, match="hold"):
        BasketballEnv(hold=hold)
