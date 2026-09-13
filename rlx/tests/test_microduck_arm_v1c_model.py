import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from microduck_arm_v1c.config import CASES, DESIGN_ID, load_design
from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.model import build_artifacts, compile_model, mass_properties
from microduck_arm_v1c.validation import hardware_acceptance, rollout, torque_com_table, workspace_scan


@pytest.mark.parametrize("candidate", ["A", "B"])
def test_consistent_candidate_model(candidate):
    model = compile_model(candidate=candidate)
    design = load_design(candidate)
    assert model.nu == 15
    assert [model.joint(int(joint_id)).name for joint_id in model.actuator_trnid[:, 0]] == design["joint_order"]
    assert model.body("arm_upper").pos == pytest.approx([0, 0, .018])
    assert model.eq_data[0, 1] == -1
    assert all(np.linalg.eigvalsh(record["inertia_body_frame_kg_m2"]).min() > 0 for record in mass_properties(model))


def test_urdf_mjcf_export(tmp_path):
    manifest = build_artifacts(tmp_path)
    assert not manifest["hardware_ready"] and not manifest["fabrication_ready"]
    for candidate in ("A", "B"):
        model = mujoco.MjModel.from_xml_path(str(tmp_path / f"microduck-arm-v1c-{candidate}-free.xml"))
        tree = ET.parse(tmp_path / f"microduck-arm-v1c-{candidate}.urdf")
        assert tree.getroot().get("name") == f"{DESIGN_ID}-{candidate}"
        assert model.nu == 15


@pytest.mark.parametrize("case", CASES)
def test_free_contract_reset_and_step(case):
    env = IntegratedArmEnv(case=case, max_steps=3)
    first, _ = env.reset(seed=2)
    second, _ = env.reset(seed=2)
    assert np.array_equal(first, second)
    observation, reward, _, _, info = env.step(env.teacher_action())
    assert observation.shape == (64,) and np.isfinite(reward)
    assert info["design_id"] == DESIGN_ID and not info["hardware_ready"]
    env.close()


@pytest.mark.parametrize("case", CASES[:4])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("candidate", ["A", "B"])
def test_fixture_real_task_regression(case, seed, candidate):
    result = rollout(case, "fixture", candidate=candidate, seed=seed)
    assert result["success"], result["failure_reason"]
    assert result["fixture_assisted"] and not result["hardware_ready"]
    if case != "reach":
        assert result["ever_lifted"] and result["ever_grasped"]
    if case == "stationary_waypoint_carry":
        assert result["waypoints_completed"] == 2


@pytest.mark.parametrize("control", ["null", "open_jaw"])
def test_negative_controls_cannot_pass(control):
    result = rollout("pick_place", "fixture", control=control, max_steps=700)
    assert not result["success"] and not result["ever_lifted"]


def test_loss_of_grasp_after_lift_fails():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    env.reset(seed=1)
    env.ever_lifted = True
    _, _, terminated, _, info = env.step(env.normalize_targets(env.home))
    assert terminated and info["failure_reason"] == "object_drop"


def test_open_request_is_not_actual_opening_authorization():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    env.reset(seed=1)
    env.ever_lifted = True
    env.phase = 7
    env.release_ready = True
    env.release_jaw_at_entry = 0.
    env.goal = env.data.xpos[env.object_id].copy()
    _, _, terminated, _, info = env.step(env.normalize_targets(env.home))
    assert terminated and info["failure_reason"] == "object_drop"
    assert not env.release_authorized


def test_equality_inventory_contains_only_gripper_coupling():
    model = compile_model()
    assert model.neq == 1
    assert model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert {int(model.eq_obj1id[0]), int(model.eq_obj2id[0])} == {model.joint("arm_gripper").id, model.joint("arm_gripper_follower").id}
    assert model.joint("trunk_base_freejoint").type == mujoco.mjtJoint.mjJNT_FREE
    assert model.joint("task_object_free").type == mujoco.mjtJoint.mjJNT_FREE


def test_phase_is_event_driven_and_teacher_is_pure():
    env = IntegratedArmEnv(case="pick_place", mode="fixture")
    env.reset(seed=1)
    env.steps = 900
    before = env.phase
    env.teacher_action()
    assert env.phase == before
    env.step(env.normalize_targets(env.home))
    assert env.phase == 0


def test_limits_invalid_action_and_walking_fixture_rejection():
    with pytest.raises(ValueError):
        IntegratedArmEnv(case="walking_carry", mode="fixture")
    env = IntegratedArmEnv(mode="fixture")
    env.reset(seed=1)
    for action in (np.zeros(14), np.full(15, np.nan), np.full(15, 1.1)):
        initial = env.data.qpos.copy()
        with pytest.raises(ValueError):
            env.step(action)
        np.testing.assert_array_equal(initial, env.data.qpos)
    target = env.home.copy()
    target[0] += .3
    target[10] += .3
    env.step(env.normalize_targets(target))
    assert env.command_targets[0] - env.home[0] == pytest.approx(.12)
    assert env.command_targets[10] - env.home[10] == pytest.approx(.008)


def test_engineering_gates_fail_closed(tmp_path):
    result = hardware_acceptance()
    assert not result["hardware_release"] and not result["physical_tests_executed"]
    assert all(not gate["passed"] for gate in result["gates"].values())
    torque = torque_com_table(tmp_path, samples=2)
    assert torque["rows"] == 50 and not torque["torque_release_passed"]
    workspace = workspace_scan(tmp_path, samples=2)
    assert workspace["balance_safe_qualified"] is None
