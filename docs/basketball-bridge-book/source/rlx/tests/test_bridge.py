"""Contract, physics-curriculum, and crossing-honesty checks for BridgeEnv."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv
from rlx.environments.bridge import (
    ASSISTANCE_LEVELS,
    BRIDGE_HALF_LENGTH,
    BRIDGE_HALF_WIDTH,
    BRIDGE_MASS,
    CURRICULUM_KNOBS,
    END_PLATFORM_X,
    FOOT_CONTACT_SETTLE_M,
    PLATFORM_TOP_Z,
    SPAWN_X_LEVELS,
    START_PLATFORM_X,
    BridgeEnv,
    bridge_model,
    curriculum_level,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def model():
    return bridge_model()


@pytest.fixture
def env(model):
    instance = BridgeEnv(
        model=model,
        command=(0.18, 0, 0),
        random_yaw=False,
    )
    instance.reset(seed=101)
    yield instance
    instance.close()


def test_model_is_a_free_suspended_narrow_plank(env):
    assert env.model.nq == 28
    assert env.model.nv == 26
    assert env.model.nu == 14
    assert env.model.joint("bridge_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.body_mass[env.bridge_body_id] == pytest.approx(BRIDGE_MASS)
    np.testing.assert_allclose(
        env.model.geom_size[env.bridge_geom_id],
        (BRIDGE_HALF_LENGTH, BRIDGE_HALF_WIDTH, 0.015),
    )
    assert env.model.ntendon == 4
    assert env.model.geom("start_platform").pos[2] == pytest.approx(PLATFORM_TOP_Z / 2)
    assert env.model.geom("end_platform").pos[2] == pytest.approx(PLATFORM_TOP_Z / 2)


def test_gym_contract_and_bridge_state_is_privileged(env):
    observation, info = env.reset(seed=202)
    assert observation.shape == (61,)
    assert observation.dtype == np.float32
    assert env.action_space.shape == (14,)
    assert info["bridge_position"][2] == pytest.approx(PLATFORM_TOP_Z - 0.015)
    before = observation.copy()
    env.data.qpos[env.bridge_qpos_adr + 1] += 0.03
    mujoco.mj_forward(env.model, env.data)
    after = env._get_obs()
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(after[51:61], np.zeros(10))
    env.navigation = False
    env.set_command((0.12, -0.01, 0.05))
    np.testing.assert_allclose(env._get_obs()[48:51], (0.12, -0.01, 0.05))
    env.navigation = True


def test_reset_is_seeded_and_starts_on_safe_platform(env):
    first_obs, first_info = env.reset(seed=303)
    env.step(np.zeros(14))
    second_obs, second_info = env.reset(seed=303)
    np.testing.assert_array_equal(first_obs, second_obs)
    assert first_info == second_info
    assert second_info["trunk_x_m"] == pytest.approx(START_PLATFORM_X - 0.05)
    assert second_info["supported_feet"] >= 1
    assert second_info["robot_ground_contact"] is False
    assert second_info["crossing_accepted"] is False
    assert all(
        isinstance(value, float)
        for value in env.recipe_metrics().values()
    )
    assert set(env.recipe_metrics()) >= {
        "bridge_crossed",
        "bridge_progress_m",
        "bridge_assistance",
        "robot_floor_contact",
        "nonfoot_support",
        "feet_support",
        "bridge_contact_observed",
    }


def test_assistance_is_plank_only_and_zero_is_force_free(env):
    env.reset(seed=404)
    env.assistance = 1
    env.data.qpos[env.bridge_qpos_adr + 1] += 0.025
    mujoco.mj_forward(env.model, env.data)
    env._apply_physics_curriculum()
    assert np.linalg.norm(env.data.xfrc_applied[env.bridge_body_id]) > 0
    np.testing.assert_array_equal(
        env.data.xfrc_applied[env.trunk_body_id],
        np.zeros(6),
    )
    env.assistance = 0
    env._apply_physics_curriculum()
    np.testing.assert_array_equal(
        env.data.xfrc_applied,
        np.zeros_like(env.data.xfrc_applied),
    )


def test_reward_is_exact_inherited_locomotion_reward():
    assert BridgeEnv._compute_reward is MicroduckWalkEnv._compute_reward
    assert dict(CURRICULUM_KNOBS) == {
        "assistance": "plank restoring force and level torque only",
        "spawn_x": "progressive physical start position from plank to launch platform",
    }


def test_curriculum_changes_only_assistance(env):
    assert ASSISTANCE_LEVELS == (1, 0.6, 0.3, 0.1, 0)
    assert curriculum_level(0, crossed=True, elapsed_s=5) == 1
    assert curriculum_level(4, crossed=True, elapsed_s=5) == 4
    assert curriculum_level(3, crossed=False, elapsed_s=1.9) == 2
    assert curriculum_level(0, crossed=False, elapsed_s=0.2) == 0
    assert curriculum_level(2, crossed=False, elapsed_s=4) == 2

    env.curriculum = True
    env.stage = 0
    env.assistance = 1
    env._episode_finished = True
    env._crossed = True
    env.step_count = 300
    env.reset(seed=1)
    assert env.stage == 1
    assert env.assistance == 0.6

    unassisted = BridgeEnv(model=env.model)
    assisted = BridgeEnv(model=env.model, curriculum=True)
    try:
        assert unassisted.max_steps == 1000
        assert unassisted.assistance == 0
        assert assisted.assistance == 1
        np.testing.assert_allclose(unassisted.fixed_command, (0.25, 0, 0))
        assert unassisted.action_delay is False
    finally:
        unassisted.close()
        assisted.close()


def test_crossing_requires_full_traversal_two_end_feet_and_no_ground(env):
    env.reset(seed=505)
    env.data.qpos[0] = 0
    env.data.qpos[3:7] = (1, 0, 0, 0)
    env.data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
    env.data.qpos[2] = (
        env.stand_z + PLATFORM_TOP_Z - FOOT_CONTACT_SETTLE_M
    )
    mujoco.mj_forward(env.model, env.data)
    assert env.metrics()["feet_on_bridge"] > 0
    end_x = END_PLATFORM_X
    env.data.qpos[0] = end_x
    env.data.qpos[3:7] = (1, 0, 0, 0)
    env.data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
    env.data.qpos[2] = (
        env.stand_z + PLATFORM_TOP_Z - FOOT_CONTACT_SETTLE_M
    )
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    metrics = env.metrics()
    assert metrics["full_traversal"]
    assert metrics["feet_on_end_platform"] == 2
    assert metrics["foot_ground_contacts"] == 0
    assert metrics["crossing_accepted"]

    env._crossed = False
    env.data.qpos[0] = end_x - 0.2
    mujoco.mj_forward(env.model, env.data)
    assert not env.metrics()["crossing_accepted"]

    env._crossed = False
    env._bridge_contact_observed = True
    env.data.qpos[0] = end_x + 0.65
    env.data.qpos[2] = 0.10
    mujoco.mj_forward(env.model, env.data)
    grounded = env.metrics()
    assert grounded["robot_ground_contact"]
    assert not grounded["crossing_accepted"]


def test_crossing_cannot_skip_suspended_plank_contact_history(env):
    env.reset(seed=506)
    env.data.qpos[0] = END_PLATFORM_X
    env.data.qpos[3:7] = (1, 0, 0, 0)
    env.data.qpos[env.joint_qpos_adr] = C.DEFAULT_POSE
    env.data.qpos[2] = (
        env.stand_z + PLATFORM_TOP_Z - FOOT_CONTACT_SETTLE_M
    )
    mujoco.mj_forward(env.model, env.data)
    metrics = env.metrics()
    assert metrics["full_traversal"]
    assert metrics["feet_on_end_platform"] == 2
    assert not metrics["bridge_contact_observed"]
    assert not metrics["crossing_accepted"]


def test_curriculum_spawn_ladder_reaches_final_launch_platform(env):
    curriculum_env = BridgeEnv(model=env.model, curriculum=True)
    try:
        for stage, expected_x in enumerate(SPAWN_X_LEVELS):
            curriculum_env.stage = stage
            curriculum_env.assistance = ASSISTANCE_LEVELS[stage]
            _, metrics = curriculum_env.reset(seed=44)
            assert metrics["trunk_x_m"] == pytest.approx(expected_x)
        assert SPAWN_X_LEVELS[-1] == pytest.approx(START_PLATFORM_X - 0.05)
    finally:
        curriculum_env.close()


def test_navigation_commander_uses_standard_twist_slots(env):
    env.reset(seed=507)
    env.data.qpos[1] = 0.10
    yaw = math.radians(10)
    env.data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    mujoco.mj_forward(env.model, env.data)
    observation = env._get_obs()
    assert observation[48] > 0
    assert observation[49] < 0
    assert observation[50] < 0


def test_navigation_commander_stops_forward_on_destination_platform(env):
    env.reset(seed=508)
    env.data.qpos[0] = END_PLATFORM_X - 0.049
    mujoco.mj_forward(env.model, env.data)
    observation = env._get_obs()
    assert observation[48] == pytest.approx(0.0, abs=1e-7)
    assert observation[49] != pytest.approx(0.0)


def test_plank_moves_from_contact_without_teleporting_robot(env):
    env.reset(seed=606)
    initial_plank = env.data.qpos[
        env.bridge_qpos_adr:env.bridge_qpos_adr + 7
    ].copy()
    initial_robot_x = float(env.data.qpos[0])
    env.data.qpos[0] = -0.10
    env.data.qpos[2] = (
        env.stand_z + PLATFORM_TOP_Z - FOOT_CONTACT_SETTLE_M
    )
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    for _ in range(20):
        env.step(np.zeros(14))
        if env._episode_finished:
            break
    changed = env.data.qpos[env.bridge_qpos_adr:env.bridge_qpos_adr + 7]
    assert np.linalg.norm(changed - initial_plank) > 1e-5
    assert initial_robot_x == pytest.approx(START_PLATFORM_X - 0.05)
    assert math.isfinite(env.metrics()["bridge_roll_deg"])


@pytest.mark.parametrize(
    "action",
    [np.zeros(13), np.full(14, np.nan), np.full(14, np.inf)],
)
def test_rejects_invalid_actions(env, action):
    env.reset(seed=707)
    with pytest.raises(ValueError, match="actions"):
        env.step(action)


@pytest.mark.parametrize("assistance", [-1, 1.1, float("nan")])
def test_rejects_invalid_assistance(assistance):
    with pytest.raises(ValueError, match="assistance"):
        BridgeEnv(assistance=assistance)


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime is not installed",
)
def test_shipped_stand_and_walk_onnx_accept_bridge_observation(env):
    import onnxruntime as ort

    observation, _ = env.reset(seed=808)
    for name in ("alpha_stand.onnx", "alpha_walking.onnx"):
        session = ort.InferenceSession(
            str(ROOT / "microduck" / "policies" / name),
            providers=["CPUExecutionProvider"],
        )
        input_name = session.get_inputs()[0].name
        action = session.run(None, {input_name: observation[None]})[0]
        assert action.shape == (1, 14)
        assert np.isfinite(action).all()
