"""WingPod Camera v2: chest-camera styling and simulated optical views only."""

from dataclasses import asdict

import mujoco
import numpy as np

from microduck_arm_design.wingpod import HONEY, Pod, WingPodEnv, WingPodOverlay


BODY = "arm_mount"
CAMERA_POD = Pod("chest_camera", BODY, (.0145, 0, -.026), (.0025, .014, .0085), .0015, (.06, .13, .18, 1))
LENS_SPACING_M = .013
LENS_X_M = .0185
LENS_Z_M = -.025
FOV_Y_DEG = 70.0
OPTICAL_FORWARD = np.array([1., 0., 0.])
OPTICAL_UP = np.array([0., 0., 1.])
LENS_ROTATION = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def camera_spec():
    return {
        "version": "WingPod Camera v2",
        "reference": "user-provided duck-chest-camera-corrected.png",
        "mount": "fixed chest / lower front of arm root, not moving arm or gripper",
        "body": BODY,
        "housing": asdict(CAMERA_POD),
        "units": "metres",
        "optical_centers_local": {eye: [LENS_X_M, side * LENS_SPACING_M / 2, LENS_Z_M]
                                  for eye, side in (("left", 1), ("right", -1))},
        "optical_forward_local": OPTICAL_FORWARD.tolist(),
        "optical_up_local": OPTICAL_UP.tolist(),
        "synthetic_fov_y_deg": FOV_Y_DEG,
        "lens_center_spacing_m": LENS_SPACING_M,
        "sensor_type": "two simulated RGB viewpoints; no qualified stereo/depth hardware",
        "hardware_model": None,
        "mass_kg": None,
        "power_w": None,
        "hardware_release": False,
        "vision_policy_trained": False,
        "dimensions_status": "appearance-envelope proposal, not a selected camera datasheet",
    }


class WingPodCameraOverlay(WingPodOverlay):
    def update(self, scene, data):
        super().update(scene, data)
        self._rounded(scene, data, CAMERA_POD)
        for side in (-1, 1):
            center = np.array([.0175, side * LENS_SPACING_M / 2, LENS_Z_M])
            for radius, depth, color in (
                (.0040, .00045, (.23, .44, .53, 1)),
                (.0033, .00050, (.025, .045, .065, 1)),
                (.0021, .00055, (.045, .20, .28, 1)),
            ):
                self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_CYLINDER,
                           center, (radius, depth, 0), color, LENS_ROTATION)
            self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                       (.0182, center[1] - .00065, LENS_Z_M + .0007),
                       (.00015, .00045, .00065), (.57, .84, .89, 1))
        self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                   (.0176, 0, -.032), (.0005, .0035, .00055), HONEY)


def optical_frame(model, data, eye):
    if eye not in ("left", "right"):
        raise ValueError("eye must be left or right")
    body_id = model.body(BODY).id
    rotation = data.xmat[body_id].reshape(3, 3)
    side = 1 if eye == "left" else -1
    center = np.array([LENS_X_M, side * LENS_SPACING_M / 2, LENS_Z_M])
    return data.xpos[body_id] + rotation @ center, rotation @ OPTICAL_FORWARD, rotation @ OPTICAL_UP


def render_optical_view(renderer, model, data, eye="left"):
    renderer.update_scene(data)
    WingPodCameraOverlay(model).update(renderer.scene, data)
    position, forward, up = optical_frame(model, data, eye)
    near = .002
    half_height = near * np.tan(np.deg2rad(FOV_Y_DEG) / 2)
    for camera in renderer.scene.camera:
        camera.pos[:] = position
        camera.forward[:] = forward
        camera.up[:] = up
        camera.frustum_near = near
        camera.frustum_far = 5
        camera.frustum_bottom = -half_height
        camera.frustum_top = half_height
        camera.frustum_center = 0
        camera.orthographic = 0
    return renderer.render().copy()


class WingPodCameraEnv(WingPodEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overlay = WingPodCameraOverlay(self.robot.model)
