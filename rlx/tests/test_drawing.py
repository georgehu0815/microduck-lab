import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from rlx.environments.drawing import ACTION_DIM, CONTRACT_VERSION, DrawingEnv, OBS_DIM, drawing_xml


def test_separate_contract_and_real_free_pencil():
    env = DrawingEnv()
    obs, _ = env.reset(seed=1)
    assert CONTRACT_VERSION == "microduck-drawing-v1"
    assert C.OBS_DIM == 61 and C.NUM_JOINTS == 14
    assert obs.shape == (OBS_DIM,) == (83,)
    assert env.model.nu == ACTION_DIM == 15
    assert env.model.actuator(14).name == "mouth_servo"
    assert env.model.joint("pencil_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert env.model.neq == 0
    assert np.isfinite(obs).all()
    assert np.allclose(obs[6:20], 0)
    assert env.model.actuator_gainprm[0, 0] == 8
    assert 'name="drawing_canvas"' in drawing_xml()


def test_reset_is_seeded_and_clears_all_ink():
    env = DrawingEnv()
    first, _ = env.reset(seed=17)
    pencil = env.data.qpos[env.pencil_qpos:].copy()
    for _ in range(10):
        env.step(env.teacher_action())
    assert len(env.trace) > 0
    second, _ = env.reset(seed=17)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(pencil, env.data.qpos[env.pencil_qpos:])
    assert env.drawing_payload()["points"] == []
    env.reset(seed=18)
    assert not np.array_equal(pencil, env.data.qpos[env.pencil_qpos:])


def test_opening_fifteenth_control_releases_pencil():
    env = DrawingEnv()
    for _ in range(100):
        action = env.teacher_action()
        action[14] = 1
        _, _, done, _, info = env.step(action)
        if done:
            break
    assert info["dropped"]
    assert not env.assessment()["passed"]


def test_contact_marks_are_not_reference_or_requested_pen_state():
    env = DrawingEnv()
    action = np.zeros(15, np.float32)
    action[14] = -.8333333
    for _ in range(100):
        env.step(action)
    assert env.trace
    assert all(row[5] >= .0001 for row in env.trace)
    assert any(not row[6] for row in env.trace)
    assert env.assessment()["coverage"] < .1
    assert not env.assessment()["passed"]
    env.reset()
    env.data.qpos[env.pencil_qpos] -= .03
    mujoco.mj_forward(env.model, env.data)
    env.step(action)
    assert env.trace == []


def test_no_support_forces_in_final_stage_and_negative_penalties():
    env = DrawingEnv(assistance=0)
    for _ in range(30):
        _, _, _, _, info = env.step(env.teacher_action())
        assert np.count_nonzero(env.data.xfrc_applied) == 0
        assert info["terms"]["action_rate_penalty"] <= 0
        assert info["terms"]["force_penalty"] <= 0


def test_pencil_is_not_clamped_by_original_head_collision_meshes():
    env = DrawingEnv()
    shaft = env.model.geom("pencil_shaft").id
    allowed = {env.model.geom(name).id for name in ("upper_beak_pad", "lower_beak_pad", "drawing_canvas", "floor")}
    for _ in range(50):
        env.step(env.teacher_action())
        for contact in env.data.contact:
            pair = set(contact.geom)
            if shaft in pair:
                assert (pair - {shaft}).issubset(allowed)


def test_peak_force_includes_earlier_physics_substeps(monkeypatch):
    env = DrawingEnv()
    samples = iter([3., .1, .2, .05])

    def contacts():
        return np.array([next(samples), .2, .2]), None

    monkeypatch.setattr(env, "_contacts", contacts)
    env.step(env.teacher_action())
    assert env.max_force == 3.
    assert env.contact_forces[0] == .05


def test_velocity_lag_preserves_the_original_61_channel_contract():
    env = DrawingEnv()
    velocities = env.data.qvel[env.joint_dofs].copy()
    observation, *_ = env.step(env.teacher_action())
    np.testing.assert_array_equal(observation[20:34], velocities)


def test_teacher_feasibility_is_not_a_learned_policy_claim():
    env = DrawingEnv(seed=4)
    for _ in range(env.max_steps):
        _, _, done, truncated, _ = env.step(env.teacher_action())
        if done or truncated:
            break
    result = env.assessment()
    assert result["completed"]
    assert result["passed"]
    assert result["grasp_fraction"] >= .9
    assert result["coverage"] >= .85


@pytest.mark.parametrize("action", [np.zeros(14), np.full(15, np.nan)])
def test_reject_incompatible_actions(action):
    with pytest.raises(ValueError):
        DrawingEnv().step(action)


@pytest.mark.parametrize("kwargs", [{"assistance": -1}, {"friction": float("nan")}, {"offset": (1, 1)}, {"scale": 2}, {"max_episode_s": 0}])
def test_invalid_physics_inputs_fail_closed(kwargs):
    with pytest.raises(ValueError):
        DrawingEnv(**kwargs)
