from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microduck_arm_design.wingpod import physics_fingerprint
from microduck_arm_design.wingpod_camera import (
    CAMERA_POD, LENS_SPACING_M, WingPodCameraEnv, camera_spec, optical_frame,
)
from microduck_arm_experiments.tennis_return import TennisReturnEnv


@pytest.fixture
def camera_env():
    env = WingPodCameraEnv(gripper="wide_candidate", release_mode="half_height")
    env.reset(0)
    yield env
    env.close()


def test_camera_version_preserves_physical_model(camera_env):
    base = TennisReturnEnv(gripper="wide_candidate", release_mode="half_height")
    try:
        base.reset(0)
        assert physics_fingerprint(base.robot.model) == physics_fingerprint(camera_env.robot.model)
        assert camera_env.robot.model.nu == 15
    finally:
        base.close()


def test_camera_attached_to_fixed_chest_with_orthogonal_axes(camera_env):
    model, data = camera_env.robot.model, camera_env.robot.data
    left, forward, up = optical_frame(model, data, "left")
    right, _, _ = optical_frame(model, data, "right")
    assert np.linalg.norm(left - right) == pytest.approx(LENS_SPACING_M)
    assert np.dot(forward, up) == pytest.approx(0)
    assert np.linalg.norm(forward) == pytest.approx(1)
    assert model.body("arm_mount").jntnum == 0
    with pytest.raises(ValueError):
        optical_frame(model, data, "invalid")


def test_camera_visual_geometry_does_not_modify_state(camera_env):
    model, data = camera_env.robot.model, camera_env.robot.data
    before = physics_fingerprint(model)
    state = data.qpos.copy()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    camera_env.overlay.update(scene, data)
    assert scene.ngeom > 180
    np.testing.assert_array_equal(data.qpos, state)
    assert physics_fingerprint(model) == before


def test_camera_mount_is_not_on_moving_arm(camera_env):
    model, data = camera_env.robot.model, camera_env.robot.data
    before = optical_frame(model, data, "left")[0]
    address = model.joint("arm_base_yaw").qposadr[0]
    data.qpos[address] += .3
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(optical_frame(model, data, "left")[0], before)


def test_camera_spec_is_candidate_not_hardware_claim():
    spec = camera_spec()
    assert spec["hardware_release"] is False
    assert spec["hardware_model"] is None
    assert spec["mass_kg"] is None
    assert spec["vision_policy_trained"] is False
    assert 2 * CAMERA_POD.half_size[1] == pytest.approx(.028)


def test_camera_version_preserves_dynamics(camera_env):
    base = TennisReturnEnv(gripper="wide_candidate", release_mode="half_height")
    try:
        base.reset(0)
        for _ in range(100):
            mujoco.mj_step(base.robot.model, base.robot.data)
            mujoco.mj_step(camera_env.robot.model, camera_env.robot.data)
        np.testing.assert_array_equal(base.robot.data.qpos, camera_env.robot.data.qpos)
        np.testing.assert_array_equal(base.robot.data.qvel, camera_env.robot.data.qvel)
    finally:
        base.close()
