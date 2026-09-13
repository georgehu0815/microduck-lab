import mujoco
import numpy as np
import pytest

from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.model import compile_model
from microduck_arm_v1c.validation import evaluate


@pytest.mark.parametrize("case", ["reach", "pick_place", "obstacle_relocation", "stationary_waypoint_carry"])
@pytest.mark.parametrize("seed", [0, 2, 9])
def test_free_base_stationary_tasks_without_fixture(case, seed):
    env = IntegratedArmEnv(case=case, mode="free")
    try:
        env.reset(seed=seed)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, info = env.step(env.teacher_action())
            assert not np.any(env.data.xfrc_applied)
            assert not np.any(env.data.qfrc_applied)
            if terminated or truncated:
                break
        assert info["success"], info["failure_reason"]
        assert info["diagnostics"]["static_support_margin_m"] >= .01
        assert not info["fixture_assisted"]
        assert not info["hardware_ready"]
    finally:
        env.close()


def test_contact_refinement_preserves_free_roots_and_friction():
    model = compile_model()
    assert model.opt.noslip_iterations == 10
    assert model.joint("trunk_base_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("task_object_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    for side in ("left", "right"):
        np.testing.assert_allclose(model.geom_friction[model.geom(f"arm_pad_{side}").id], [1., .005, .001])
    assert model.body_mass[model.body("task_object").id] == pytest.approx(.01)
    np.testing.assert_allclose(model.actuator_forcerange[10:], np.tile([-.1, .1], (5, 1)))


@pytest.mark.parametrize("seeds,candidates", [((), ("A",)), ((0, 0), ("A",)), ((0,), ()), ((0,), ("C",))])
def test_empty_or_duplicate_matrix_cannot_claim_success(tmp_path, seeds, candidates):
    with pytest.raises(ValueError):
        evaluate(tmp_path, seeds=seeds, candidates=candidates)


def test_carry_target_tracks_body_frame_not_table_waypoint(monkeypatch):
    env = IntegratedArmEnv(case="walking_carry", mode="free")
    try:
        env.reset(seed=0)
        env.phase = 4
        env.carry_anchor = np.array([.08, -.01, .06])
        observed = []
        def capture_target(target):
            observed.append(target.copy())
            return env.home[10:14].copy()
        monkeypatch.setattr(env, "_inverse_kinematics", capture_target)
        before = env.data.qpos.copy()
        env.teacher_action()
        expected = env.data.xpos[env.trunk_id] + env.data.xmat[env.trunk_id].reshape(3, 3) @ env.carry_anchor
        expected += env.data.site_xpos[env.tcp_id] - env.data.xpos[env.object_id]
        np.testing.assert_allclose(observed[0], expected)
        np.testing.assert_array_equal(env.data.qpos, before)
    finally:
        env.close()


@pytest.mark.parametrize("case", ["stowed_arm_walking", "walking_carry"])
@pytest.mark.parametrize("seed", [0, 2, 9])
def test_free_walking_preserves_original_contact_and_carry_gates(case, seed):
    env = IntegratedArmEnv(case=case, mode="free")
    try:
        env.reset(seed=seed)
        for _ in range(env.max_steps):
            _, _, terminated, truncated, info = env.step(env.teacher_action())
            assert not np.any(env.data.xfrc_applied)
            if terminated or truncated:
                break
        assert info["success"], info["failure_reason"]
        assert len(env.support_events) >= 3
        assert info["diagnostics"]["forward_displacement_m"] >= .05
        assert env.hold_ticks * env.dt >= 1.0
        if case == "walking_carry":
            assert env.ever_lifted
            assert info["diagnostics"]["bilateral_grasp"]
            assert not info["diagnostics"]["object_supported"]
            relative = env.data.xmat[env.trunk_id].reshape(3, 3).T @ (env.data.xpos[env.object_id] - env.data.xpos[env.trunk_id])
            assert np.linalg.norm(relative - env.carry_anchor) < .025
    finally:
        env.close()
