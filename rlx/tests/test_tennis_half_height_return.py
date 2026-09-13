from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

from microduck_arm_experiments.tennis_return import (
    ReturnMonitor,
    TennisReturnEnv,
    release_profile,
    task_spec,
)


def _sample(**changes):
    velocity = np.asarray(changes.pop("ball_velocity_m_s", [0.0, 0.0, 0.0]))
    sample = {
        "tilt_rad": 0.0,
        "trunk_height_m": 0.12,
        "forbidden_contacts": [],
        "joint_limit_violation": False,
        "bilateral_grasp": True,
        "ball_inside_bin": True,
        "ball_speed_m_s": float(np.linalg.norm(velocity)),
        "ball_velocity_m_s": velocity.tolist(),
        "ball_center_height_m": 0.05,
        "tcp_ball_distance_m": 0.06,
        "jaw_open": False,
        "ball_supported": False,
        "ball_bottom_m": 0.0165,
        "carry_distance_m": 0.13,
        "above_bin": False,
        "bottom_supported": False,
        "any_robot_ball_contact": True,
        "ball_radius_m": 0.0335,
        "release_height_ok": True,
    }
    sample.update(changes)
    return sample


def _authorized_monitor():
    monitor = ReturnMonitor(
        release_mode="half_height",
        phase="lower",
        lifted=True,
        origin_checked=True,
    )
    monitor.advance(_sample(ball_velocity_m_s=[0.0, 0.0, -0.02]), 0.2)
    assert monitor.phase == "release"
    assert monitor.release_authorized
    return monitor


def test_profiles_preserve_supported_defaults_and_copy_gates_per_monitor():
    spec = task_spec()
    supported = ReturnMonitor()
    first = ReturnMonitor(release_mode="half_height")
    second = ReturnMonitor(release_mode="half_height")

    assert supported.release_mode == "supported"
    assert supported.gates == spec["acceptance"]
    assert supported.gates == release_profile("supported")["acceptance"]
    assert first.gates == release_profile("half_height")["acceptance"]
    first.gates["placement_speed_max_m_s"] = 99.0
    assert second.gates["placement_speed_max_m_s"] == 0.03


def test_half_height_sample_requires_full_containment_and_correct_center_height():
    env = TennisReturnEnv(release_mode="half_height", max_steps=1)
    try:
        env.reset(0)
        robot = env.robot
        container = env.spec["bin"]
        center = np.asarray(container["center_xy_m"])
        maximum = (
            container["bottom_thickness_m"]
            + 0.5 * container["wall_height_m"]
        )
        address = robot.object_qpos

        robot.data.qpos[address : address + 3] = [*center, maximum + 0.001]
        mujoco.mj_forward(robot.model, robot.data)
        sample = env.sample()
        assert sample["ball_inside_bin"]
        assert not sample["release_height_ok"]

        robot.data.qpos[address : address + 3] = [*center, maximum]
        mujoco.mj_forward(robot.model, robot.data)
        sample = env.sample()
        assert sample["release_height_ok"]
        assert sample["release_center_max_m"] == pytest.approx(maximum)

        half = np.asarray(container["inside_size_xy_m"]) / 2
        robot.data.qpos[address : address + 3] = [
            center[0] + half[0] - env.radius / 2,
            center[1],
            maximum,
        ]
        mujoco.mj_forward(robot.model, robot.data)
        sample = env.sample()
        assert not sample["ball_inside_bin"]
        assert not sample["release_height_ok"]
    finally:
        env.close()


def test_half_height_post_release_containment_requires_measured_side_contact():
    env = TennisReturnEnv(release_mode="half_height", max_steps=1)
    try:
        env.reset(0)
        robot = env.robot
        container = env.spec["bin"]
        center = np.asarray(container["center_xy_m"])
        half = np.asarray(container["inside_size_xy_m"]) / 2
        bottom_center = container["bottom_thickness_m"] + env.radius
        address = robot.object_qpos

        robot.data.qpos[address : address + 3] = [
            center[0],
            center[1] + half[1] - env.radius + 14e-6,
            bottom_center,
        ]
        mujoco.mj_forward(robot.model, robot.data)
        touching = env.sample()
        assert not touching["ball_geometrically_inside_bin"]
        assert touching["ball_physically_contained_in_bin"]
        assert touching["contact_containment_used"]
        assert touching["ball_bin_side_contact_geoms"] == ["bin_y_high"]
        assert touching["maximum_ball_bin_side_penetration_m"] == pytest.approx(14e-6)
        assert touching["side_contact_containment_tolerance_m"] == pytest.approx(0.0005)

        robot.data.qpos[address : address + 3] = [
            center[0],
            center[1] + half[1] - env.radius + 0.000501,
            bottom_center,
        ]
        mujoco.mj_forward(robot.model, robot.data)
        escaped = env.sample()
        assert escaped["ball_bin_side_contact_geoms"] == ["bin_y_high"]
        assert escaped["maximum_ball_bin_side_penetration_m"] > 0.0005
        assert not escaped["ball_physically_contained_in_bin"]

        robot.data.qpos[address : address + 3] = [
            center[0],
            center[1] + half[1] + container["wall_thickness_m"] + env.radius + 0.001,
            bottom_center,
        ]
        mujoco.mj_forward(robot.model, robot.data)
        no_contact_escape = env.sample()
        assert no_contact_escape["ball_bin_side_contact_geoms"] == []
        assert not no_contact_escape["ball_physically_contained_in_bin"]

        robot.data.qpos[address : address + 3] = [
            *center,
            container["bottom_thickness_m"] + container["wall_height_m"] + env.radius + 1e-6,
        ]
        mujoco.mj_forward(robot.model, robot.data)
        above_rim = env.sample()
        assert not above_rim["ball_physically_contained_in_bin"]
    finally:
        env.close()


def test_negative_distance_side_contact_does_not_require_diagnostic_force(monkeypatch):
    env = TennisReturnEnv(release_mode="half_height", max_steps=1)
    try:
        env.reset(0)
        robot = env.robot
        container = env.spec["bin"]
        center = np.asarray(container["center_xy_m"])
        half = np.asarray(container["inside_size_xy_m"]) / 2
        robot.data.qpos[robot.object_qpos : robot.object_qpos + 3] = [
            center[0],
            center[1] + half[1] - env.radius + 17e-6,
            container["bottom_thickness_m"] + env.radius,
        ]
        mujoco.mj_forward(robot.model, robot.data)

        def zero_contact_force(model, data, index, force):
            del model, data, index
            force.fill(0.0)

        monkeypatch.setattr(mujoco, "mj_contactForce", zero_contact_force)
        sample = env.sample()
        assert sample["active_contacts"] == []
        assert sample["ball_bin_side_contact_geoms"] == ["bin_y_high"]
        assert sample["maximum_ball_bin_side_penetration_m"] == pytest.approx(17e-6)
        assert sample["ball_physically_contained_in_bin"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "velocity",
    [
        [0.0, 0.0, 0.031],
        [0.021, 0.0, 0.0],
        [0.0, 0.0, -0.031],
    ],
)
def test_authorization_rejects_measured_throw_or_excess_speed(velocity):
    monitor = ReturnMonitor(
        release_mode="half_height",
        phase="lower",
        lifted=True,
        origin_checked=True,
    )
    monitor.advance(_sample(ball_velocity_m_s=velocity), 0.2)

    assert monitor.phase == "lower"
    assert not monitor.release_authorized
    assert monitor.failure is None


def test_wrong_height_and_early_grasp_loss_cannot_authorize_release():
    wrong_height = ReturnMonitor(
        release_mode="half_height",
        phase="lower",
        lifted=True,
        origin_checked=True,
    )
    wrong_height.advance(_sample(release_height_ok=False), 1.0)
    assert wrong_height.phase == "lower"
    assert not wrong_height.release_authorized

    early_loss = ReturnMonitor(
        release_mode="half_height",
        phase="lower",
        lifted=True,
        origin_checked=True,
    )
    early_loss.advance(_sample(bilateral_grasp=False), 0.01)
    assert early_loss.failure == "premature_release_or_throw"
    assert not early_loss.release_authorized


@pytest.mark.parametrize(
    "velocity",
    [
        [0.0, 0.0, 0.031],
        [0.081, 0.0, -0.2],
    ],
)
def test_post_authorization_upward_or_lateral_throw_is_rejected(velocity):
    monitor = _authorized_monitor()
    monitor.advance(
        _sample(
            ball_velocity_m_s=velocity,
            jaw_open=True,
            any_robot_ball_contact=False,
        ),
        0.01,
    )

    assert monitor.failure == "measured_throw_velocity"
    assert not monitor.success


def test_natural_downward_fall_is_allowed_but_final_settle_is_unchanged():
    monitor = _authorized_monitor()
    assert monitor.release_height_m == pytest.approx(0.05)
    assert monitor.release_velocity_m_s == pytest.approx([0.0, 0.0, -0.02])

    falling = _sample(
        ball_velocity_m_s=[0.0, 0.0, -0.45],
        jaw_open=True,
        any_robot_ball_contact=False,
    )
    monitor.advance(falling, 0.05)
    assert monitor.phase == "release"
    assert monitor.failure is None
    monitor.advance(falling, 0.05)
    assert monitor.phase == "retreat"
    assert monitor.maximum_free_fall_interval_s == pytest.approx(0.1)

    settled = _sample(
        ball_velocity_m_s=[0.0, 0.0, 0.0],
        jaw_open=True,
        bottom_supported=True,
        any_robot_ball_contact=False,
    )
    monitor.advance(settled, 1.999)
    assert not monitor.success
    monitor.advance(settled, 0.001)
    assert monitor.success


def test_contact_qualified_containment_applies_only_after_authorization():
    monitor = _authorized_monitor()
    monitor.advance(
        _sample(
            ball_inside_bin=False,
            ball_physically_contained_in_bin=True,
            ball_velocity_m_s=[0.0, 0.0, 0.0],
            jaw_open=True,
            bottom_supported=True,
            any_robot_ball_contact=False,
        ),
        0.1,
    )
    assert monitor.phase == "retreat"
    assert monitor.failure is None

    escaped = _authorized_monitor()
    escaped.advance(
        _sample(
            ball_inside_bin=False,
            ball_physically_contained_in_bin=False,
            ball_velocity_m_s=[0.0, 0.0, 0.0],
            jaw_open=True,
            bottom_supported=True,
            any_robot_ball_contact=False,
        ),
        0.01,
    )
    assert escaped.failure == "ball_left_bin"


def test_excessive_unsupported_interval_fails_and_records_maximum():
    monitor = _authorized_monitor()
    bound = monitor.gates["maximum_free_fall_interval_s"]
    monitor.advance(
        _sample(
            ball_velocity_m_s=[0.0, 0.0, -0.2],
            jaw_open=True,
            any_robot_ball_contact=False,
        ),
        bound + 0.001,
    )

    assert monitor.failure == "free_fall_interval_exceeded"
    assert monitor.maximum_free_fall_interval_s == pytest.approx(bound + 0.001)


def test_half_height_bounds_and_optional_navigation_api_are_explicit():
    profile = release_profile("half_height")
    gates = copy.deepcopy(profile["acceptance"])
    assert profile["experiment_id"] == "tennis-half-height-gravity-return-v2"
    assert gates["placement_speed_max_m_s"] == 0.03
    assert gates["release_upward_velocity_max_m_s"] == 0.005
    assert gates["release_lateral_speed_max_m_s"] == 0.02
    assert gates["airborne_upward_velocity_max_m_s"] == 0.03
    assert gates["airborne_lateral_speed_max_m_s"] == 0.08
    assert gates["maximum_free_fall_interval_s"] == 0.2
    assert gates["side_contact_containment_tolerance_m"] == 0.0005

    env = TennisReturnEnv(
        release_mode="half_height",
        plan_navigation=True,
        max_steps=1,
    )
    try:
        env.reset(0)
        sample = env.sample()
        assert env.plan_navigation
        assert sample["plan_navigation"]
        assert sample["navigation_prediction"] is None
        assert not sample["navigation_qualified"]
        assert sample["navigation_profile"] is None
    finally:
        env.close()
