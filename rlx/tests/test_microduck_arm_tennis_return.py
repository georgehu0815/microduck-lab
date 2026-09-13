from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from microduck_arm_experiments.tennis_return import (
    ReturnMonitor,
    TennisReturnEnv,
    VARIANTS,
    feasibility,
    geometry_report,
    run_episode,
    scene_tree,
    task_spec,
    experiment_provenance,
)
from microduck_arm_experiments.tennis_controller import BinNavigator, cached_release_clear, crouch_offset, ground_arm_ik
from microduck_arm_v1c.model import compile_model
from microduck_arm_v1c.whole_body_learning import source_provenance


def _compile(variant="nominal", gripper="stock"):
    return mujoco.MjModel.from_xml_string(
        ET.tostring(scene_tree(variant, gripper=gripper), encoding="unicode")
    )


def _sample(**changes):
    sample = {
        "tilt_rad": 0.0,
        "trunk_height_m": 0.12,
        "forbidden_contacts": [],
        "joint_limit_violation": False,
        "bilateral_grasp": False,
        "ball_inside_bin": False,
        "ball_speed_m_s": 0.0,
        "tcp_ball_distance_m": 0.02,
        "jaw_open": True,
        "ball_supported": True,
        "ball_bottom_m": 0.0,
        "carry_distance_m": 0.0,
        "above_bin": False,
        "bottom_supported": False,
        "any_robot_ball_contact": False,
        "ball_radius_m": VARIANTS["nominal"][0] / 2,
    }
    sample.update(changes)
    return sample


def _complete_put_back(monitor, *, starts_inside=False, retreat_seconds=2.0):
    monitor.advance(
        _sample(
            ball_inside_bin=starts_inside,
            tcp_ball_distance_m=0.0,
            jaw_open=True,
        ),
        0.1,
    )
    monitor.advance(_sample(bilateral_grasp=True), 0.2)
    monitor.advance(
        _sample(
            bilateral_grasp=True,
            ball_supported=False,
            ball_bottom_m=0.021,
        ),
        0.5,
    )
    monitor.advance(
        _sample(
            bilateral_grasp=True,
            ball_supported=False,
            ball_bottom_m=0.12,
            carry_distance_m=0.13,
            above_bin=True,
        ),
        0.1,
    )
    monitor.advance(
        _sample(
            bilateral_grasp=True,
            ball_inside_bin=True,
            bottom_supported=True,
            any_robot_ball_contact=True,
        ),
        0.2,
    )
    monitor.advance(
        _sample(
            ball_inside_bin=True,
            bottom_supported=True,
            jaw_open=True,
        ),
        0.1,
    )
    monitor.advance(
        _sample(
            ball_inside_bin=True,
            bottom_supported=True,
            jaw_open=True,
            tcp_ball_distance_m=0.06,
        ),
        retreat_seconds,
    )


@pytest.mark.parametrize("variant", ["small", "nominal", "large"])
def test_real_ball_dimensions_mass_shell_inertia_and_floor_spawn(variant):
    diameter, mass = VARIANTS[variant]
    radius = diameter / 2
    root = scene_tree(variant)
    model = _compile(variant)
    ball = root.find(".//body[@name='task_object']")
    body_id = model.body("task_object").id
    geom_id = model.geom("object_geom").id

    assert float(ball.get("pos").split()[2]) == pytest.approx(radius)
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    assert model.geom_size[geom_id, 0] == pytest.approx(radius)
    assert model.body_mass[body_id] == pytest.approx(mass)
    np.testing.assert_allclose(
        model.body_inertia[body_id],
        np.full(3, 2 * mass * radius**2 / 3),
        rtol=1e-9,
        atol=0,
    )


def test_type_1_and_type_2_corners_are_exact_and_feasibility_is_not_release():
    spec = task_spec()["ball"]
    assert VARIANTS["small"] == pytest.approx(
        (spec["type_1_2_diameter_range_m"][0], spec["type_1_2_mass_range_kg"][0])
    )
    assert VARIANTS["large"] == pytest.approx(
        (spec["type_1_2_diameter_range_m"][1], spec["type_1_2_mass_range_kg"][1])
    )
    for variant in ("small", "large"):
        report = feasibility(variant)
        assert report["ball_diameter_m"] == pytest.approx(VARIANTS[variant][0])
        assert report["ball_mass_kg"] == pytest.approx(VARIANTS[variant][1])
        assert not report["training_authorized_by_feasibility"]
        assert not report["hardware_release"]


@pytest.mark.parametrize("variant", ["small", "large"])
def test_type_1_and_type_2_corner_variants_reset_on_the_floor(variant):
    env = TennisReturnEnv(variant=variant, max_steps=1)
    try:
        env.reset(seed=7)
        position = env.robot.data.xpos[env.robot.object_id]
        expected_xy = np.asarray(env.spec["ball"]["initial_xy_m"])
        jitter = env.spec["ball"]["spawn_jitter_m"]
        assert position[2] - env.radius == pytest.approx(0, abs=1e-12)
        assert np.all(np.abs(position[:2] - expected_xy) <= jitter)
    finally:
        env.close()


def test_stock_scene_preserves_free_robot_and_stock_gripper_contract():
    baseline = compile_model()
    model = _compile(gripper="stock")

    assert model.nu == baseline.nu == 15
    assert model.joint("trunk_base_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("task_object_free").type == mujoco.mjtJoint.mjJNT_FREE
    assert [model.joint(i).name for i in range(model.njnt)] == [
        baseline.joint(i).name for i in range(baseline.njnt)
    ]
    assert [model.actuator(i).name for i in range(model.nu)] == [
        baseline.actuator(i).name for i in range(baseline.nu)
    ]
    np.testing.assert_allclose(model.actuator_forcerange, baseline.actuator_forcerange)
    for side in ("left", "right"):
        actual = model.geom(f"arm_pad_{side}").id
        expected = baseline.geom(f"arm_pad_{side}").id
        np.testing.assert_allclose(model.geom_pos[actual], baseline.geom_pos[expected])
        np.testing.assert_allclose(model.geom_size[actual], baseline.geom_size[expected])
        assert model.geom_type[actual] == baseline.geom_type[expected]


def test_bin_has_bottom_and_four_fully_colliding_walls_and_ball_is_not_welded():
    model = _compile()
    names = {"bin_bottom", "bin_x_low", "bin_x_high", "bin_y_low", "bin_y_high"}

    assert {model.geom(i).name for i in range(model.ngeom)} >= names
    for name in names:
        geom_id = model.geom(name).id
        assert model.geom_contype[geom_id] != 0
        assert model.geom_conaffinity[geom_id] != 0
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert {
        int(model.eq_obj1id[0]),
        int(model.eq_obj2id[0]),
    } == {
        model.joint("arm_gripper").id,
        model.joint("arm_gripper_follower").id,
    }


def test_scene_generation_does_not_modify_frozen_baseline_sources():
    before = source_provenance()
    for gripper in ("stock", "wide_candidate"):
        for variant in VARIANTS:
            scene_tree(variant, gripper=gripper)
    assert source_provenance() == before


def test_wide_candidate_has_75mm_gap_without_changing_pivots_or_gearing():
    stock = _compile(gripper="stock")
    wide = _compile(gripper="wide_candidate")
    spec = task_spec()["wide_gripper_candidate"]
    report = feasibility(gripper="wide_candidate")

    assert report["maximum_open_parallel_pad_gap_m"] == pytest.approx(0.075)
    assert report["equatorial_grasp_aperture_pass"]
    assert spec["pad_center_x_m"] == pytest.approx(0.048)
    assert spec["tcp_offset_m"] == pytest.approx(0.055)
    assert spec["stem_outward_offset_m"] == pytest.approx(0.004)
    assert wide.site_pos[wide.site("arm_tcp").id, 0] == pytest.approx(0.055)
    for side in ("left", "right"):
        sign = 1 if side == "left" else -1
        body_id = wide.body(f"arm_finger_{side}").id
        stock_body_id = stock.body(f"arm_finger_{side}").id
        np.testing.assert_allclose(wide.body_pos[body_id], stock.body_pos[stock_body_id])
        pad_id = wide.geom(f"arm_pad_{side}").id
        assert wide.geom_pos[pad_id, 0] == pytest.approx(0.048)
        assert abs(wide.geom_pos[pad_id, 1]) == pytest.approx(0.028)
        stem_id = wide.geom(f"arm_wide_stem_{side}").id
        assert wide.geom_pos[stem_id, 1] == pytest.approx(sign * 0.032)
    np.testing.assert_allclose(wide.eq_data, stock.eq_data)
    np.testing.assert_allclose(wide.actuator_forcerange, stock.actuator_forcerange)
    assert wide.nu == stock.nu == 15


def test_wide_candidate_models_colliding_structural_bridge_and_stem_mass():
    stock = _compile(gripper="stock")
    wide = _compile(gripper="wide_candidate")
    spec = task_spec()["wide_gripper_candidate"]
    expected_each = (
        np.prod(spec["bridge_full_size_m"])
        + np.prod(spec["stem_full_size_m"])
    ) * spec["assumed_density_kg_m3"]

    for side in ("left", "right"):
        body_id = wide.body(f"arm_finger_{side}").id
        stock_body_id = stock.body(f"arm_finger_{side}").id
        for part in ("bridge", "stem"):
            geom_id = wide.geom(f"arm_wide_{part}_{side}").id
            assert wide.geom_contype[geom_id] != 0
            assert wide.geom_conaffinity[geom_id] != 0
        assert wide.body_mass[body_id] - stock.body_mass[stock_body_id] == pytest.approx(
            expected_each
        )


@pytest.mark.parametrize("variant", ["small", "nominal", "large"])
def test_wide_open_ball_at_tcp_clears_palm_and_gripper_motor(variant):
    model = _compile(variant, gripper="wide_candidate")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ball_qpos = model.jnt_qposadr[model.joint("task_object_free").id]
    data.qpos[ball_qpos : ball_qpos + 3] = data.site_xpos[
        model.site("arm_tcp").id
    ]
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    for obstacle in ("arm_palm", "arm_gripper_motor"):
        distance = mujoco.mj_geomDistance(
            model,
            data,
            model.geom("object_geom").id,
            model.geom(obstacle).id,
            1.0,
            np.zeros(6),
        )
        assert distance > 0, (variant, obstacle, distance)


def test_wide_geometry_contact_sweep_is_pad_first_for_every_ball_variant():
    report = geometry_report()

    assert report["all_geometry_screens_passed"]
    assert not report["dynamic_task_passed"]
    assert not report["hardware_release"]
    assert {record["variant"] for record in report["records"]} == set(VARIANTS)
    for record in report["records"]:
        assert record["open_ball_intersections"] == []
        contact = record["first_bilateral_pad_intersection"]
        assert contact is not None
        assert contact["pads_only"]
        assert contact["contact_geoms"] == ["arm_pad_left", "arm_pad_right"]


def test_fully_controlled_synthetic_put_back_passes_monitor_only():
    monitor = ReturnMonitor()
    _complete_put_back(monitor)

    assert monitor.success
    assert monitor.phase == "success"
    assert monitor.failure is None
    assert not feasibility(gripper="wide_candidate")["training_authorized_by_feasibility"]
    assert not feasibility(gripper="wide_candidate")["hardware_release"]


def test_wide_teacher_reaches_real_ground_lift_without_self_collision():
    env = TennisReturnEnv(gripper="wide_candidate", max_steps=900)
    try:
        env.reset(seed=0)
        while not env.done and not env.monitor.lifted:
            _, _, info = env.step(env.teacher_action())
            assert not info["forbidden_contacts"]
            assert not np.any(env.robot.data.qfrc_applied)
            assert not np.any(env.robot.data.xfrc_applied)
        assert env.monitor.lifted
        assert info["bilateral_grasp"]
        assert not info["ball_supported"]
        assert info["ball_bottom_m"] >= task_spec()["acceptance"]["minimum_lift_clearance_m"]
        assert not info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("seed", [0, 2, 3, 5])
def test_nominal_full_return_keeps_original_physical_acceptance(seed):
    env = TennisReturnEnv(gripper="wide_candidate", max_steps=4000)
    try:
        env.reset(seed=seed)
        while not env.done:
            _, _, info = env.step(env.teacher_action())
        assert info["success"], info["failure_reason"]
        assert info["bottom_supported"] and info["ball_inside_bin"]
        assert not info["any_robot_ball_contact"]
        assert info["ball_speed_m_s"] <= task_spec()["acceptance"]["placement_speed_max_m_s"]
        assert info["tcp_ball_distance_m"] >= env.radius + task_spec()["acceptance"]["retreat_clearance_m"]
        assert [event["phase"] for event in env.phase_events] == [
            "approach", "grasp", "lift", "carry", "lower", "release", "retreat", "success"
        ]
        assert not np.any(env.robot.data.qfrc_applied)
        assert not np.any(env.robot.data.xfrc_applied)
    finally:
        env.close()


def test_small_ball_supported_release_completes_real_full_return():
    env = TennisReturnEnv(variant="small", gripper="wide_candidate", max_steps=4000)
    try:
        env.reset(seed=8)
        while not env.done:
            _, _, info = env.step(env.teacher_action())
        assert info["success"], info["failure_reason"]
        assert env.controller.release_strategy == "supported_hold"
        assert not env.controller.release_prediction[0]["success"]
        assert env.controller.release_prediction[-1]["success"]
        assert info["bottom_supported"] and info["ball_inside_bin"]
        assert not info["any_robot_ball_contact"]
        assert not np.any(env.robot.data.qfrc_applied)
        assert not np.any(env.robot.data.xfrc_applied)
    finally:
        env.close()


@pytest.mark.parametrize("moving", [False, True])
def test_navigation_recovery_requires_stalled_measured_progress(monkeypatch, moving):
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(0)
        robot = env.robot
        navigator = BinNavigator()
        monkeypatch.setattr(navigator.policy, "leg_targets", lambda *args, **kwargs: np.zeros(10))
        for step in range(400):
            robot.data.time = step * robot.dt
            robot.data.xpos[robot.trunk_id, 0] = max(0., robot.data.time - 2) * .03 if moving else 0.
            targets = navigator.leg_targets(robot)
            assert np.max(np.abs(targets)) <= .060001
            if robot.data.time < 4.5:
                assert navigator.recovery_started is None
        assert (navigator.recovery_started is None) == moving
    finally:
        env.close()


def test_crouch_and_ik_planning_do_not_mutate_live_physics():
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(seed=0)
        robot = env.robot
        state = robot.data.qpos.copy()
        velocity = robot.data.qvel.copy()
        controls = robot.data.ctrl.copy()
        offset = crouch_offset(robot, .02)
        assert offset.shape == (10,)
        assert np.isfinite(offset).all()
        joints, reachable = ground_arm_ik(robot, robot.data.site_xpos[robot.tcp_id], 0., .055)
        assert reachable
        assert np.allclose(joints, robot.home[10:14], atol=1e-6)
        assert np.array_equal(robot.data.qpos, state)
        assert np.array_equal(robot.data.qvel, velocity)
        assert np.array_equal(robot.data.ctrl, controls)
    finally:
        env.close()


@pytest.mark.parametrize("reachable", [False, True])
def test_release_holds_last_safe_plan_instead_of_initial_pose(monkeypatch, reachable):
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(seed=0)
        robot = env.robot
        controller = env.controller
        env.monitor.phase = "release"
        controller.release_targets = robot.home.copy()
        controller.release_tcp = robot.data.site_xpos[robot.tcp_id].copy()
        controller.release_pitch = 1.4
        controller.release_jaw = 0.
        controller.release_axis = np.array([1., 0., 0.])
        controller.release_started = robot.data.time
        controller.placement_crouch = np.zeros(10)
        previous_safe = np.array([.1, .3, .5, .6])
        controller.safe_release_joints = previous_safe.copy()
        before = robot.data.qpos.copy()
        monkeypatch.setattr(
            "microduck_arm_experiments.tennis_controller.ground_arm_ik",
            lambda *args: (np.array([.1, .7, .2, .5]), reachable),
        )
        targets = controller.placement_targets(robot.home.copy())
        np.testing.assert_array_equal(targets[10:14], previous_safe)
        np.testing.assert_array_equal(robot.data.qpos, before)
        assert controller.release_blocked
        assert not controller.release_opening
        assert not env.monitor.success
        controller.release_started = robot.data.time - 5.
        controller.release_settled_s = .5
        controller.placement_targets(robot.home.copy())
        assert not controller.release_opening
        assert not controller.cached_release_screen_passed
        calls = []
        monkeypatch.setattr(env, "sample", lambda: {
            "bottom_supported": True, "ball_inside_bin": True, "forbidden_contacts": [],
        })

        def reject_sweep(robot, joints):
            calls.append(joints.copy())
            return False

        monkeypatch.setattr("microduck_arm_experiments.tennis_controller.cached_release_clear", reject_sweep)
        controller.placement_targets(robot.home.copy())
        assert len(calls) == 1
        assert not controller.release_opening
        assert not controller.cached_release_screen_passed
    finally:
        env.close()


def test_cached_release_sweep_leaves_live_physics_unchanged():
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(seed=0)
        robot = env.robot
        before = tuple(array.copy() for array in (robot.data.qpos, robot.data.qvel, robot.data.ctrl))
        assert isinstance(cached_release_clear(robot, robot.home[10:14]), bool)
        assert not cached_release_clear(robot, np.array([0., .8, 0., 0.]))
        for actual, expected in zip((robot.data.qpos, robot.data.qvel, robot.data.ctrl), before):
            np.testing.assert_array_equal(actual, expected)
        assert robot.data.time == 0.
    finally:
        env.close()


def test_experiment_provenance_includes_controller_helpers():
    provenance = experiment_provenance()
    assert {"tennis_return.py", "tennis_controller.py", "__init__.py"} <= provenance["python_sources"].keys()
    assert all(len(value) == 64 for value in provenance["python_sources"].values())


def test_in_bin_and_above_bin_telemetry_is_json_serializable():
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(seed=0)
        robot = env.robot
        container = env.spec["bin"]
        for height in (container["bottom_thickness_m"] + env.radius, .16):
            robot.data.qpos[robot.object_qpos:robot.object_qpos + 3] = [*container["center_xy_m"], height]
            mujoco.mj_forward(robot.model, robot.data)
            sample = env.sample()
            assert type(sample["ball_inside_bin"]) is bool
            assert type(sample["above_bin"]) is bool
            json.dumps(sample, allow_nan=False)
    finally:
        env.close()


def test_non_pad_ball_collision_is_rejected_from_real_contacts():
    env = TennisReturnEnv(gripper="wide_candidate")
    try:
        env.reset(seed=0)
        robot = env.robot
        robot.data.qpos[robot.object_qpos:robot.object_qpos + 3] = robot.data.geom_xpos[robot.model.geom("arm_palm").id]
        mujoco.mj_forward(robot.model, robot.data)
        sample = env.sample()
        assert "non_pad_ball_collision" in sample["forbidden_contacts"]
        assert sample["active_contacts"]
    finally:
        env.close()


@pytest.mark.parametrize(
    "invalid_start",
    [
        {"ball_inside_bin": True},
        {"ball_bottom_m": 0.0011},
        {"ball_bottom_m": -0.0011},
        {"carry_distance_m": 0.0011},
        {"any_robot_ball_contact": True},
    ],
)
def test_invalid_approach_origin_cannot_be_counted_as_a_return(invalid_start):
    monitor = ReturnMonitor()
    monitor.advance(_sample(**invalid_start), 0.01)

    assert monitor.failure == "invalid_floor_start"
    assert not monitor.success


def test_premature_release_or_throw_after_lift_fails():
    monitor = ReturnMonitor(phase="carry", lifted=True)
    monitor.advance(_sample(bilateral_grasp=False), 0.01)
    assert monitor.failure == "premature_release_or_throw"
    assert not monitor.success


def test_authorized_release_without_bottom_support_hard_fails():
    monitor = ReturnMonitor(phase="release", lifted=True, release_authorized=True)
    monitor.advance(
        _sample(
            ball_inside_bin=True,
            bottom_supported=False,
            any_robot_ball_contact=False,
            jaw_open=True,
        ),
        0.01,
    )
    assert monitor.failure == "unsupported_or_fast_release"


def test_authorized_release_cannot_hide_unsafe_relift_behind_robot_contact():
    monitor = ReturnMonitor(
        phase="release",
        lifted=True,
        release_authorized=True,
        origin_checked=True,
    )
    monitor.advance(
        _sample(
            ball_inside_bin=True,
            bottom_supported=False,
            any_robot_ball_contact=True,
            ball_speed_m_s=0.5,
            jaw_open=True,
        ),
        0.01,
    )

    assert monitor.failure == "unsupported_or_fast_release"
    assert not monitor.success


def test_released_ball_leaving_bin_is_a_drop_failure():
    monitor = ReturnMonitor(phase="retreat", lifted=True, release_authorized=True)
    monitor.advance(
        _sample(
            ball_inside_bin=False,
            bottom_supported=False,
            any_robot_ball_contact=False,
            jaw_open=True,
        ),
        0.01,
    )
    assert monitor.failure == "ball_left_bin"
    assert not monitor.success


def test_rim_resting_without_bottom_support_cannot_advance():
    monitor = ReturnMonitor(phase="lower", lifted=True)
    monitor.advance(
        _sample(
            bilateral_grasp=True,
            ball_inside_bin=True,
            bottom_supported=False,
            any_robot_ball_contact=True,
        ),
        1.0,
    )
    assert monitor.phase == "lower"
    assert not monitor.release_authorized
    assert not monitor.success


def test_full_sphere_must_fit_inside_bin_not_just_its_center():
    env = TennisReturnEnv(max_steps=1)
    try:
        env.reset(seed=0)
        container = env.spec["bin"]
        center = np.asarray(container["center_xy_m"])
        half = np.asarray(container["inside_size_xy_m"]) / 2
        position = np.array(
            [
                center[0] + half[0] - env.radius / 2,
                center[1],
                container["bottom_thickness_m"] + env.radius,
            ]
        )
        assert np.all(np.abs(position[:2] - center) < half)
        env.robot.data.qpos[env.robot.object_qpos : env.robot.object_qpos + 3] = position
        mujoco.mj_forward(env.robot.model, env.robot.data)
        assert not env.sample()["ball_inside_bin"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"tilt_rad": 0.61}, "fall"),
        ({"trunk_height_m": 0.064}, "fall"),
        ({"ball_speed_m_s": float("nan")}, "nonfinite_telemetry"),
        ({"forbidden_contacts": ["robot_bin_collision"]}, "robot_bin_collision"),
        ({"joint_limit_violation": True}, "joint_limit"),
    ],
)
def test_monitor_fail_closed_conditions(changes, failure):
    monitor = ReturnMonitor()
    monitor.advance(_sample(**changes), 0.01)
    assert monitor.failure == failure
    assert not monitor.success


@pytest.mark.parametrize(
    "changes",
    [
        {"ball_inside_bin": False},
        {"bottom_supported": False},
        {"any_robot_ball_contact": True},
        {"ball_speed_m_s": 0.031},
        {"tcp_ball_distance_m": 0.04},
    ],
)
def test_retreat_is_blocked_by_drop_contact_speed_or_insufficient_clearance(changes):
    monitor = ReturnMonitor(phase="retreat", lifted=True, release_authorized=True)
    sample = _sample(
        ball_inside_bin=True,
        bottom_supported=True,
        jaw_open=True,
        tcp_ball_distance_m=0.06,
    )
    sample.update(changes)
    monitor.advance(sample, 2.1)
    assert not monitor.success


def test_retreat_requires_full_two_second_hold():
    monitor = ReturnMonitor(phase="retreat", lifted=True, release_authorized=True)
    sample = _sample(
        ball_inside_bin=True,
        bottom_supported=True,
        jaw_open=True,
        tcp_ball_distance_m=0.06,
    )

    monitor.advance(sample, 1.999)
    assert not monitor.success
    monitor.advance(sample, 0.001)
    assert monitor.success


def test_small_failure_rollout_keeps_telemetry_and_uses_no_applied_forces(tmp_path):
    env = TennisReturnEnv(variant="small", max_steps=1)
    try:
        first = env.reset(seed=4)
        before = env.robot._observation().copy()
        after, done, info = env.step(env.teacher_action("hold"))
        assert done
        assert first.shape == before.shape == after.shape == (64,)
        assert np.isfinite(first).all() and np.isfinite(after).all()
        assert not np.any(env.robot.data.qfrc_applied)
        assert not np.any(env.robot.data.xfrc_applied)
        assert not info["success"]
    finally:
        env.close()

    output = tmp_path / "failed-rollout"
    result = run_episode(output, variant="small", control="hold", max_steps=2)
    records = [
        json.loads(line)
        for line in (output / "telemetry.jsonl").read_text().splitlines()
    ]
    assert not result["success"]
    assert result["simulation_only"] and not result["hardware_release"]
    assert "not_trained_return_or_navigation" in result["controller_scope"]
    assert result["feasibility"]["training_authorized_by_feasibility"] is False
    assert records
    assert len(records) == result["steps"]
    assert all(len(row["observation_before"]) == 64 for row in records)
    assert all(len(row["observation_after"]) == 64 for row in records)
    assert all(np.isfinite(row["observation_before"]).all() for row in records)
    assert all(np.isfinite(row["observation_after"]).all() for row in records)
    for previous, current in zip(records, records[1:]):
        np.testing.assert_allclose(
            previous["observation_after"],
            current["observation_before"],
            rtol=0,
            atol=0,
        )


def test_episode_output_directory_is_never_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError):
        run_episode(output, max_steps=1)
    assert marker.read_text() == "preserve"
