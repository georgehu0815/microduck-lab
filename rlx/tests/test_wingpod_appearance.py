from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microduck_arm_design.wingpod import (
    DETAILS, PODS, WingPodOverlay, appearance_spec, apply_palette, physics_fingerprint,
)
from microduck_arm_experiments.tennis_return import TennisReturnEnv


@pytest.fixture
def environment():
    env = TennisReturnEnv(gripper="wide_candidate", release_mode="half_height")
    env.reset(0)
    yield env
    env.close()


def test_palette_preserves_all_nonvisual_arrays_and_contact_pads(environment):
    model = environment.robot.model
    before = physics_fingerprint(model)
    pads = [model.geom(name).rgba.copy() for name in ("arm_pad_left", "arm_pad_right")]
    apply_palette(model)
    assert physics_fingerprint(model) == before
    assert model.nu == 15
    assert np.allclose(model.body("arm_forearm").pos, [.055, 0, 0])
    assert np.allclose(model.body("arm_hand").pos, [.050, 0, 0])
    assert np.allclose(model.site("arm_tcp").pos, [.055, 0, 0])
    left_inner = model.body("arm_finger_left").pos[1] + model.geom("arm_pad_left").pos[1] - model.geom("arm_pad_left").size[1]
    right_inner = model.body("arm_finger_right").pos[1] + model.geom("arm_pad_right").pos[1] + model.geom("arm_pad_right").size[1]
    assert left_inner - right_inner == pytest.approx(.075)
    for name, color in zip(("arm_pad_left", "arm_pad_right"), pads):
        np.testing.assert_array_equal(model.geom(name).rgba, color)


def test_scene_overlay_does_not_touch_physics_or_state(environment):
    model, data = environment.robot.model, environment.robot.data
    before = physics_fingerprint(model)
    positions, velocities = data.qpos.copy(), data.qvel.copy()
    scene = mujoco.MjvScene(model, maxgeom=2000)
    overlay = WingPodOverlay(model)
    overlay.update(scene, data)
    assert scene.ngeom > 150
    assert all(np.isfinite(scene.geoms[index].pos).all() for index in range(scene.ngeom))
    assert before == physics_fingerprint(model)
    np.testing.assert_array_equal(positions, data.qpos)
    np.testing.assert_array_equal(velocities, data.qvel)


def test_palette_does_not_change_dynamics(environment):
    model = environment.robot.model
    twin = mujoco.MjData(model)
    mujoco.mj_copyData(twin, model, environment.robot.data)
    for _ in range(50):
        mujoco.mj_step(model, twin)
    apply_palette(model)
    for _ in range(50):
        mujoco.mj_step(model, environment.robot.data)
    np.testing.assert_array_equal(twin.qpos, environment.robot.data.qpos)
    np.testing.assert_array_equal(twin.qvel, environment.robot.data.qvel)


def test_fingerprint_detects_physical_mutation(environment):
    before = physics_fingerprint(environment.robot.model)
    environment.robot.model.body_mass[-1] += .001
    assert physics_fingerprint(environment.robot.model) != before


def test_cad_spec_is_review_only():
    spec = appearance_spec()
    assert spec["hardware_release"] is False
    assert spec["physical_mass_kg"] is None
    assert all(0 < pod.radius < min(pod.half_size) for pod in PODS)
    assert len(spec["details"]) == len(DETAILS)


def test_export_shares_the_render_spec(tmp_path):
    import csv
    import json
    from scripts.build_wingpod_design import build_cad

    build_cad(tmp_path)
    spec = json.loads((tmp_path / "appearance-spec.json").read_text())
    assert spec == json.loads(json.dumps(appearance_spec()))
    with (tmp_path / "shell-dimensions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(PODS)
    for row, pod in zip(rows, PODS):
        assert row["body"] == pod.body
        assert float(row["width_mm"]) == pytest.approx(pod.half_size[0] * 2000)
    cad = (tmp_path / "wingpod-review-shells.scad").read_text()
    assert cad.count("rotate(") == len(DETAILS)
