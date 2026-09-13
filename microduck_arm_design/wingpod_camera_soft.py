"""Soft camera-face revision, isolated from physical and historical models."""

from dataclasses import asdict, replace

import mujoco
import numpy as np

from microduck_arm_design.wingpod import WingPodEnv, WingPodOverlay
from microduck_arm_design.wingpod_camera import (
    BODY, CAMERA_POD, LENS_ROTATION, LENS_SPACING_M, LENS_Z_M, camera_spec,
)


PALETTE = {
    "cream": (1.0, .97, .87, 1),
    "sage": (.53, .60, .54, 1),
    "rim": (.31, .40, .39, 1),
    "navy": (.035, .12, .19, 1),
    "glint": (.73, .88, .88, 1),
    "peach": (.94, .68, .52, 1),
    "honey": (1.0, .76, .29, 1),
}
SOFT_POD = replace(CAMERA_POD, color=PALETTE["cream"])


def soft_camera_spec():
    spec = camera_spec()
    spec.update({
        "version": "WingPod Camera v2 Soft",
        "reference": "user-provided duck-soft-frame-finished.png",
        "housing": asdict(SOFT_POD),
        "palette": PALETTE,
        "revision_scope": "render-only face colors, layered lenses, cheeks and beak",
        "physical_model_changed": False,
        "historical_case_palette": "original graphite v2; not soft-revision footage",
    })
    return spec


class WingPodSoftCameraOverlay(WingPodOverlay):
    def update(self, scene, data):
        super().update(scene, data)
        self._rounded(scene, data, SOFT_POD)
        for side in (-1, 1):
            center = np.array([.0175, side * LENS_SPACING_M / 2, LENS_Z_M])
            for radius, position, depth, color in (
                (.0040, .0175, .00045, "sage"),
                (.0035, .0179, .00030, "rim"),
                (.0028, .0182, .00025, "navy"),
            ):
                center[0] = position
                self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_CYLINDER,
                           center, (radius, depth, 0), PALETTE[color], LENS_ROTATION)
            self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                       (.0185, center[1] - .00065, LENS_Z_M + .0008),
                       (.00015, .00045, .00065), PALETTE["glint"])
            self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                       (.0171, side * .0103, -.0304),
                       (.0002, .00135, .00065), PALETTE["peach"])
        self._geom(scene, data, BODY, mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                   (.0176, 0, -.0315), (.0007, .0032, .0010), PALETTE["honey"])


class WingPodSoftCameraEnv(WingPodEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.overlay = WingPodSoftCameraOverlay(self.robot.model)
