from dataclasses import replace

import mujoco
import numpy as np

from microduck_arm_design.wingpod import physics_fingerprint
from microduck_arm_design.wingpod_camera import CAMERA_POD, WingPodCameraEnv, optical_frame
from microduck_arm_design.wingpod_camera_soft import (
    PALETTE, SOFT_POD, WingPodSoftCameraEnv, soft_camera_spec,
)


def test_soft_face_retains_camera_envelope_and_candidate_status():
    assert replace(SOFT_POD, color=CAMERA_POD.color) == CAMERA_POD
    spec = soft_camera_spec()
    assert spec["hardware_release"] is False
    assert spec["vision_policy_trained"] is False
    assert spec["physical_model_changed"] is False
    assert spec["mass_kg"] is None
    assert spec["palette"]["cream"][:3] == (1., .97, .87)


def test_soft_overlay_and_dynamics_match_original():
    original = WingPodCameraEnv(gripper="wide_candidate", release_mode="half_height")
    soft = WingPodSoftCameraEnv(gripper="wide_candidate", release_mode="half_height")
    try:
        original.reset(0)
        soft.reset(0)
        model, data = soft.robot.model, soft.robot.data
        before = physics_fingerprint(model)
        state = data.qpos.copy()
        scene = mujoco.MjvScene(model, maxgeom=2000)
        soft.overlay.update(scene, data)
        np.testing.assert_array_equal(state, data.qpos)
        colors = [geom.rgba for geom in scene.geoms[:scene.ngeom]]
        for key in ("cream", "sage", "peach", "honey", "navy"):
            assert any(np.allclose(color, PALETTE[key]) for color in colors)
        assert scene.ngeom < scene.maxgeom
        assert model.nu == 15
        assert physics_fingerprint(model) == before == physics_fingerprint(original.robot.model)
        for eye in ("left", "right"):
            np.testing.assert_array_equal(optical_frame(model, data, eye),
                                          optical_frame(original.robot.model, original.robot.data, eye))
        for _ in range(100):
            mujoco.mj_step(model, data)
            mujoco.mj_step(original.robot.model, original.robot.data)
        np.testing.assert_array_equal(data.qpos, original.robot.data.qpos)
        np.testing.assert_array_equal(data.qvel, original.robot.data.qvel)
    finally:
        original.close()
        soft.close()
