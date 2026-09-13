from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from microduck_arm_v1.config import CASES, load_design
from microduck_arm_v1.env import IntegratedArmEnv
from microduck_arm_v1.model import build_artifacts, compile_model
from microduck_arm_v1.validation import hardware_acceptance, torque_com_table, validate_model, workspace_scan


def test_model_contract_and_hardware_claims():
    result = validate_model()
    assert result["passed"]
    assert result["nu"] == 15
    assert not result["hardware_release"]
    assert not result["task_success_proven"]


def test_generated_mjcf_and_urdf_are_consistent(tmp_path):
    manifest = build_artifacts(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "microduck-arm-v1-free.xml"))
    assert model.nu == 15
    robot = ET.parse(tmp_path / "microduck-arm-v1.urdf").getroot()
    links = {node.get("name") for node in robot.findall("link")}
    joints = robot.findall("joint")
    assert len(joints) == len(links) - 1
    assert len([joint for joint in joints if joint.get("type") == "revolute"]) == 16
    assert robot.find("joint[@name='arm_gripper_follower']/mimic").get("multiplier") == "-1"
    mass = sum(float(node.get("value")) for node in robot.findall("link/inertial/mass"))
    assert mass == pytest.approx(model.body_mass.sum() - model.body_mass[model.body("task_object").id])
    assert manifest["urdf_limitations"]


@pytest.mark.parametrize("case", CASES)
def test_all_cases_reset_and_step_with_explicit_status(case):
    env = IntegratedArmEnv(case=case, max_steps=3)
    try:
        observation, info = env.reset(seed=41)
        assert observation.shape == (64,)
        assert env.action_space.shape == (15,)
        assert info["simulation_only"] and not info["hardware_ready"]
        action = env.teacher_action()
        observation, reward, terminated, truncated, info = env.step(action)
        assert np.isfinite(observation).all() and np.isfinite(reward)
        assert isinstance(info["success"], bool)
    finally:
        env.close()


def test_same_seed_reset_is_exact():
    env = IntegratedArmEnv()
    try:
        first, _ = env.reset(seed=81)
        second, _ = env.reset(seed=81)
        assert np.array_equal(first, second)
    finally:
        env.close()


@pytest.mark.parametrize("action", [np.zeros(14), np.full(15, np.nan), np.full(15, np.inf), np.full(15, 1.2)])
def test_invalid_action_rejected_before_physics(action):
    env = IntegratedArmEnv()
    try:
        env.reset(seed=1)
        state = env.data.qpos.copy()
        with pytest.raises(ValueError):
            env.step(action)
        assert np.array_equal(env.data.qpos, state)
    finally:
        env.close()


def test_fixture_never_qualifies_walking():
    with pytest.raises(ValueError, match="free-base"):
        IntegratedArmEnv(case="walking_carry", mode="fixture")


def test_gripper_mirror_and_positive_motor_mass():
    design = load_design()
    model = compile_model()
    assert design["arm"]["motor_mass_kg"] * 5 == pytest.approx(.09)
    assert model.neq == 1
    assert model.eq_data[0, 1] == -1
    assert model.joint("arm_gripper").range[0] == -.32
    assert model.joint("arm_gripper_follower").range[1] == .32


def test_static_support_polygon_is_not_visual_foot_hull():
    assert IntegratedArmEnv.support_margin([], [0, 0, 0]) is None
    square = [(-.02, -.02, 0), (.02, -.02, 0), (.02, .02, 0), (-.02, .02, 0)]
    assert IntegratedArmEnv.support_margin(square, [0, 0, 0]) == pytest.approx(.02)
    assert IntegratedArmEnv.support_margin(square, [.03, 0, 0]) == pytest.approx(-.01)


def test_reach_requires_actual_motion():
    env = IntegratedArmEnv(mode="fixture")
    try:
        env.reset(seed=101)
        assert np.linalg.norm(env.goal - env.tcp_start) >= .020
    finally:
        env.close()


def test_teacher_has_no_phase_side_effect():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    try:
        env.reset(seed=101)
        before = env.phase
        env.teacher_action()
        assert env.phase == before
    finally:
        env.close()


def test_default_horizon_includes_release_and_hold():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    try:
        assert env.max_steps * env.dt > 7 * 3 + env.design["task_targets"]["hold_time_s"]
    finally:
        env.close()


def test_grasp_loss_after_lift_is_failure_even_above_initial_height():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    try:
        env.reset(seed=101)
        env.ever_lifted = True
        env.data.qpos[env.object_qpos + 2] += .03
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, info = env.step(env.normalize_targets(env.home))
        assert terminated and not info["success"] and info["failure_reason"] == "object_drop"
    finally:
        env.close()


def test_reach_orientation_is_measured_and_required():
    env = IntegratedArmEnv(mode="fixture", max_steps=2)
    try:
        env.reset(seed=101)
        env.goal = env.tcp_start.copy()
        env.goal_axis *= -1
        _, _, _, _, info = env.step(env.normalize_targets(env.home))
        assert info["diagnostics"]["tcp_axis_error_rad"] > 3.0
        assert env.hold_ticks == 0 and not info["success"]
    finally:
        env.close()


def test_walking_displacement_without_stepping_does_not_count():
    env = IntegratedArmEnv(case="stowed_arm_walking", max_steps=2)
    try:
        env.reset(seed=101)
        env.trunk_start[0] -= .06
        _, _, _, _, info = env.step(env.normalize_targets(env.home))
        assert info["diagnostics"]["forward_displacement_m"] > .05
        assert env.hold_ticks == 0 and not info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("case", ["walking_carry", "pick_place"])
def test_release_authorization_is_revoked_outside_placement_window(case):
    env = IntegratedArmEnv(case=case, max_steps=2)
    try:
        env.reset(seed=101)
        env.ever_lifted = True
        env.release_authorized = True
        _, _, terminated, _, info = env.step(env.normalize_targets(env.home))
        assert not env.release_authorized
        assert terminated and info["failure_reason"] == "object_drop"
    finally:
        env.close()


def test_reach_full_orientation_detects_roll():
    env = IntegratedArmEnv(mode="fixture")
    try:
        env.reset(seed=101)
        env.goal_axis = np.array([1., 0., 0.])
        env.data.site_xmat[env.tcp_id] = np.diag([1., -1., -1.]).ravel()
        diagnostics = env.diagnostics()
        assert diagnostics["tcp_axis_error_rad"] == pytest.approx(0)
        assert diagnostics["tcp_orientation_error_rad"] == pytest.approx(np.pi)
    finally:
        env.close()


def test_torque_and_workspace_do_not_claim_hardware_pass(tmp_path):
    torque = torque_com_table(tmp_path, samples=3)
    assert torque["maximum_required_qualified_continuous_nm_at_3x"] > 0
    assert not torque["torque_release_passed"]
    workspace = workspace_scan(tmp_path / "workspace.json", samples=3)
    assert workspace["samples"] == 3
    assert not workspace["usable_workspace_certified"]


def test_missing_hardware_evidence_fails_closed(tmp_path):
    assert not hardware_acceptance()["evidence_complete"]
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"all_passed": True}))
    result = hardware_acceptance(evidence)
    assert not result["evidence_complete"] and not result["hardware_release"]
