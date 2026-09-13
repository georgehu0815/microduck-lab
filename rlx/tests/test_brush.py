"""Focused contracts for the compliant, color-loading brush environment."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rlx.environments.brush import (
    ACTION_DIM,
    CONTRACT_VERSION,
    OBS_DIM,
    BrushEnv,
    brush_xml,
)
from rlx.environments.brush_reference import (
    BRUSH_RADIUS,
    COLORS,
    COLOR_NAMES,
    PALETTE_Z,
    brush_trajectory,
    color_strokes,
)
from rlx.environments.drawing import (
    ACTION_DIM as DRAWING_ACTION_DIM,
    CANVAS_X,
    CANVAS_Z,
    CONTRACT_VERSION as DRAWING_CONTRACT_VERSION,
    OBS_DIM as DRAWING_OBS_DIM,
    DrawingEnv,
)


def _place_tip_in_well(env: BrushEnv, color: int) -> None:
    well = env.model.geom(f"paint_well_{color}").id
    target_tip = env.data.geom_xpos[well].copy()
    target_tip[0] -= env.model.geom_size[well, 0] + BRUSH_RADIUS - 0.0008
    env.data.qpos[env.pencil_qpos : env.pencil_qpos + 3] += target_tip - env.tip
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)

    tip = env.model.geom("pencil_tip").id
    assert any(set(contact.geom) == {tip, well} for contact in env.data.contact)


def _set_exact_authored_trace(env: BrushEnv) -> None:
    env.trace = []
    env.color_trace = []
    for stroke_id, (color, points) in enumerate(
        color_strokes(env.scale, env.offset), start=1
    ):
        for y, z in points:
            env.trace.append(
                [0.0, CANVAS_X, float(y), float(CANVAS_Z + z), stroke_id, 0.1, True]
            )
            env.color_trace.append(color)
    env.step_count = env.max_steps
    env.grasp_steps = env.max_steps
    env.contact_steps = len(env.trace)
    env.pen_up_steps = 1
    env.pen_up_leaks = 0
    env.max_force = 0.1
    env.max_compression = 0.001
    env.failed = False
    env.paint_events = [
        {"time": float(color), "color": color, "force_n": 0.1}
        for color in range(4)
    ]


def test_brush_has_separate_contract_without_changing_drawing():
    env = BrushEnv(seed=1)
    observation, _ = env.reset(seed=1)
    drawing = DrawingEnv(seed=1)
    drawing_observation, _ = drawing.reset(seed=1)

    assert CONTRACT_VERSION == "microduck-brush-v2"
    assert observation.shape == (OBS_DIM,) == (93,)
    assert env.action_space.shape == (ACTION_DIM,) == (15,)
    assert env.model.nu == ACTION_DIM
    assert np.isfinite(observation).all()

    assert DRAWING_CONTRACT_VERSION == "microduck-drawing-v1"
    assert drawing_observation.shape == (DRAWING_OBS_DIM,) == (83,)
    assert drawing.action_space.shape == (DRAWING_ACTION_DIM,) == (15,)
    assert "bristle_compression" not in {
        drawing.model.joint(index).name for index in range(drawing.model.njnt)
    }


def test_xml_has_free_brush_passive_compression_and_four_physical_wells():
    model = mujoco.MjModel.from_xml_string(brush_xml())
    free_joint = model.joint("pencil_free")
    spring = model.joint("bristle_compression")
    spring_dof = spring.dofadr[0]

    assert free_joint.type == mujoco.mjtJoint.mjJNT_FREE
    assert model.neq == 0
    assert spring.type == mujoco.mjtJoint.mjJNT_SLIDE
    np.testing.assert_allclose(spring.axis, [1, 0, 0])
    np.testing.assert_allclose(spring.range, [-0.003, 0.0003])
    np.testing.assert_allclose(model.jnt_stiffness[spring.id], 25.0)
    np.testing.assert_allclose(model.dof_damping[spring_dof], 0.32)
    np.testing.assert_allclose(model.dof_armature[spring_dof], 0.001)
    np.testing.assert_allclose(model.jnt_solref[spring.id], [0.01, 1.0])
    assert spring.id not in model.actuator_trnid[:, 0]

    wells = [model.geom(f"paint_well_{color}") for color in range(4)]
    assert len({well.id for well in wells}) == 4
    np.testing.assert_allclose([well.pos[2] for well in wells], PALETTE_Z)
    np.testing.assert_allclose([well.rgba[:3] for well in wells], COLORS)
    assert all(model.geom_contype[well.id] == 8 for well in wells)
    assert all(model.geom_conaffinity[well.id] == 4 for well in wells)


def test_color_load_requires_actual_tip_well_contact_force(monkeypatch):
    env = BrushEnv(seed=2)
    contacted_color = 2
    assert int(env.trajectory["color"][env.step_count]) != contacted_color
    _place_tip_in_well(env, contacted_color)

    def set_contact_force(_model, _data, _index, wrench):
        wrench[:] = 0

    monkeypatch.setattr(mujoco, "mj_contactForce", set_contact_force)
    env._contacts()
    assert env.loaded_color == -1
    assert env.paint_events == []

    def set_active_contact_force(_model, _data, _index, wrench):
        wrench[:] = 0
        wrench[0] = 0.02

    monkeypatch.setattr(mujoco, "mj_contactForce", set_active_contact_force)
    env._contacts()
    assert env.loaded_color == contacted_color
    assert env._well_contact == contacted_color
    assert [event["color"] for event in env.paint_events] == [contacted_color]


def test_dry_brush_makes_no_ink_and_payload_colors_align_to_actual_marks(
    monkeypatch,
):
    env = BrushEnv(seed=3)
    mark = np.array([CANVAS_X, 0.004, CANVAS_Z - 0.003])

    def canvas_contact(_env):
        return np.array([0.02, 0.2, 0.2]), mark.copy()

    monkeypatch.setattr(DrawingEnv, "_contacts", canvas_contact)
    _, dry_mark = env._contacts()
    assert dry_mark is None
    assert env.loaded_color == -1

    env.loaded_color = 3
    env.step(np.zeros(ACTION_DIM, np.float32))
    payload = env.drawing_payload()
    assert len(env.trace) == len(env.color_trace) == 1
    assert len(payload["points"]) == len(payload["colors"]) == 1
    assert payload["colors"] == [3]
    assert payload["points"][0][1:] == [
        pytest.approx(mark[1]),
        pytest.approx(mark[2]),
        env.trace[0][4],
    ]


def test_reset_clears_loaded_pigment_events_and_color_trace():
    env = BrushEnv(seed=5)
    env.loaded_color = 1
    env.paint_events.append({"time": 1.0, "color": 1, "force_n": 0.1})
    env.trace.append([0.0, CANVAS_X, 0.0, CANVAS_Z, 1, 0.1, True])
    env.color_trace.append(1)

    env.reset(seed=5)

    assert env.loaded_color == -1
    assert env.paint_events == []
    assert env.trace == []
    assert env.color_trace == []
    assert env.drawing_payload()["points"] == []
    assert env.drawing_payload()["colors"] == []


def test_unassisted_steps_apply_no_support_and_penalties_are_nonpositive():
    env = BrushEnv(seed=6, assistance=0)
    for _ in range(20):
        _, _, done, truncated, info = env.step(env.teacher_action())
        assert np.count_nonzero(env.data.xfrc_applied) == 0
        assert info["terms"]["action_rate_penalty"] <= 0
        assert info["terms"]["force_penalty"] <= 0
        if done or truncated:
            break


def test_acceptance_rejects_wrong_palette_order():
    env = BrushEnv(seed=7)
    _set_exact_authored_trace(env)
    env.paint_events[1], env.paint_events[2] = (
        env.paint_events[2],
        env.paint_events[1],
    )

    result = env.assessment()

    assert all(result["by_color"][name]["coverage"] >= 0.95 for name in COLOR_NAMES)
    assert all(result["by_color"][name]["precision"] >= 0.95 for name in COLOR_NAMES)
    assert result["passed"] is False


def test_acceptance_rejects_wrong_trace_colors():
    env = BrushEnv(seed=8)
    _set_exact_authored_trace(env)
    env.color_trace = [(color + 1) % 4 for color in env.color_trace]

    assert env.assessment()["passed"] is False


def test_authored_color_strokes_and_trajectory_are_deterministic_geometry():
    first = color_strokes()
    second = color_strokes()

    assert len(first) == 37
    assert [sum(color == expected for color, _ in first) for expected in range(4)] == [
        26,
        5,
        4,
        2,
    ]
    assert [color for color, _ in first] == sorted(color for color, _ in first)
    for (first_color, first_points), (second_color, second_points) in zip(
        first, second, strict=True
    ):
        assert first_color == second_color
        np.testing.assert_array_equal(first_points, second_points)

    transformed = color_strokes(scale=0.8, offset=(0.003, -0.002))
    for (_, original), (_, changed) in zip(first, transformed, strict=True):
        np.testing.assert_allclose(changed, original * 0.8 + [0.003, -0.002])

    trajectory = brush_trajectory(steps=6000)
    repeated = brush_trajectory(steps=6000)
    assert set(trajectory) == {
        "points",
        "xyz",
        "mode",
        "color",
        "pen_down",
        "velocity",
    }
    for key in trajectory:
        np.testing.assert_array_equal(trajectory[key], repeated[key])
    assert 5900 <= len(trajectory["xyz"]) <= 6000
    assert trajectory["points"].shape == trajectory["velocity"].shape
    assert trajectory["xyz"].shape == (len(trajectory["points"]), 3)
    assert np.isfinite(trajectory["xyz"]).all()
    np.testing.assert_array_equal(
        trajectory["pen_down"],
        (trajectory["xyz"][:, 0] >= CANVAS_X - BRUSH_RADIUS - 0.00015)
        & (trajectory["mode"] != 1),
    )
    assert np.any(trajectory["pen_down"] & (trajectory["mode"] == 0))
    assert not np.any(trajectory["pen_down"] & (trajectory["mode"] == 1))
    assert list(dict.fromkeys(trajectory["color"])) == [0, 1, 2, 3]


def test_seed4_teacher_has_high_color_coverage_all_contacts_and_safe_physics():
    env = BrushEnv(seed=4, max_episode_s=120)
    for _ in range(env.max_steps):
        _, _, done, truncated, _ = env.step(env.teacher_action())
        if done or truncated:
            break

    result = env.assessment()

    assert result["completed"]
    assert result["coverage"] >= 0.95
    assert result["precision"] >= 0.95
    assert all(result["by_color"][name]["coverage"] >= 0.95 for name in COLOR_NAMES)
    assert all(result["by_color"][name]["precision"] >= 0.95 for name in COLOR_NAMES)
    assert [event["color"] for event in result["palette_contacts"]] == [0, 1, 2, 3]
    assert result["max_compression_m"] < 0.0035
    assert result["max_normal_force_n"] < 0.5
