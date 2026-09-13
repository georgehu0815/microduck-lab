from __future__ import annotations

import mujoco
import numpy as np
import pytest

from rlx.environments import arm


MANIPULATION_CASES = tuple(
    case_id for case_id in arm.CASES if case_id != "arm-reach-v1"
)
EVALUATOR_STAGES = ("initial", "grasped", "lifted", "released")


def run_teacher(case_id: str, seed: int = 71):
    env = arm.ArmEnv(case_id)
    env.reset(seed=seed)
    stages = [env.evaluator_stage]
    while not (env.terminated or env.truncated):
        env.step(env.teacher_action())
        if env.evaluator_stage != stages[-1]:
            stages.append(env.evaluator_stage)
    return env, stages


def test_reach_gate_rejects_stale_hold_counter():
    env = arm.ArmEnv("arm-reach-v1")
    try:
        env.reach_run = 50
        assert env.task_gates()["reach_position_and_axis_hold"] is False
    finally:
        env.close()


def test_reach_teacher_satisfies_current_pose_axis_and_hold():
    env = arm.ArmEnv("arm-reach-v1")
    env.reset(seed=71)
    try:
        while not (env.terminated or env.truncated):
            env.step(env.teacher_action())
        assert env.task_gates()["reach_position_and_axis_hold"] is True
    finally:
        env.close()


def test_contact_audit_runs_at_every_physics_substep(monkeypatch):
    env = arm.ArmEnv("arm-pick-place-v1")
    calls = 0
    original = env._measure_contacts

    def audited():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(env, "_measure_contacts", audited)
    try:
        env.step(np.zeros(env.action_dim, dtype=np.float32))
        assert calls == round(arm.CONTROL_DT / env.model.opt.timestep)
    finally:
        env.close()


def test_airborne_grip_loss_over_20ms_counts_as_drop():
    env = arm.ArmEnv("arm-pick-place-v1")
    try:
        env.evaluator_stage = "lifted"
        env.evaluator_events = [
            {"stage": "grasped", "time_s": 0.1},
            {"stage": "lifted", "time_s": 1.2},
        ]
        env.ever_lifted = True
        env.grip_loss_s = 0.019
        env.data.qpos[env.object_qpos:env.object_qpos + 3] = env.start + [
            0,
            0,
            0.04,
        ]
        mujoco.mj_forward(env.model, env.data)
        env.step(np.zeros(env.action_dim, dtype=np.float32))
        assert env.drop_count >= 1
        assert env.task_gates()["no_drop"] is False
    finally:
        env.close()


def test_unsafe_table_support_after_lift_counts_as_drop():
    env = arm.ArmEnv("arm-pick-place-v1")
    try:
        env.evaluator_stage = "lifted"
        env.evaluator_events = [
            {"stage": "grasped", "time_s": 0.1},
            {"stage": "lifted", "time_s": 1.2},
        ]
        env.ever_lifted = True
        env.ever_airborne = True
        env.data.qpos[env.object_qpos:env.object_qpos + 3] = [
            env.start[0],
            env.start[1],
            0.008,
        ]
        mujoco.mj_forward(env.model, env.data)
        env._measure_contacts()
        assert env.object_supported is True
        env.step(np.zeros(env.action_dim, dtype=np.float32))
        assert env.drop_count >= 1
        assert env.task_gates()["no_drop"] is False
    finally:
        env.close()


@pytest.mark.parametrize("case_id", MANIPULATION_CASES)
def test_manipulation_gates_fail_closed_from_initial_state(case_id):
    env = arm.ArmEnv(case_id)
    try:
        gates = env.task_gates()
        assert env.evaluator_stage == "initial"
        assert gates["lift_with_contact_1s"] is False
        assert gates["released_and_settled_2s"] is False
        assert env.completed is False
    finally:
        env.close()


@pytest.mark.parametrize("case_id", MANIPULATION_CASES)
def test_stale_history_cannot_satisfy_current_terminal_gate(case_id):
    env = arm.ArmEnv(case_id)
    try:
        env.max_lift_run = 10_000
        env.max_settle_run = 10_000
        env.controlled_release = True
        env.waypoint_index = 3
        env.path_length = 1.0
        env.handover_verified = True
        env.max_receiver_hold = 10_000
        env.transport_steps = 100
        env.dual_grasp_steps = 100
        env.max_transport_tilt = 0.0
        env.max_payload_slip = 0.0
        env.load_share_samples = 100
        env.load_share_sum[:] = 100.0

        gates = env.task_gates()
        assert env.evaluator_stage == "initial"
        assert gates["ordered_completion"] is False
        assert gates["released_and_settled_2s"] is False
    finally:
        env.close()


def test_pick_place_gate_thresholds_are_isolated():
    env = arm.ArmEnv("arm-pick-place-v1")
    try:
        env.max_lift_run = 49
        assert env.task_gates()["lift_with_contact_1s"] is False
        env.settle_run = 99
        assert env.task_gates()["released_and_settled_2s"] is False
    finally:
        env.close()


def test_carry_gates_are_isolated():
    env = arm.ArmEnv("arm-carry-v1")
    try:
        env.waypoint_index = 2
        assert env.task_gates()["three_waypoints"] is False
        env.path_length = 0.079
        assert env.task_gates()["path_80mm"] is False
        env.max_transport_tilt = np.deg2rad(10)
        assert env.task_gates()["tilt_below_10deg"] is False
    finally:
        env.close()


def test_handover_gates_are_isolated():
    env = arm.ArmEnv("arms-handover-v1")
    try:
        env.handover_verified = False
        assert env.task_gates()["handover_overlap_500ms"] is False
        env.max_receiver_hold = 49
        assert env.task_gates()["receiver_only_hold_1s"] is False
    finally:
        env.close()


def test_co_carry_gates_are_isolated():
    env = arm.ArmEnv("arms-co-carry-v1")
    try:
        env.peak_internal_force = 0.999
        assert env.task_gates()["internal_force_bounded"] is True
        env.peak_internal_force = 1.0
        assert env.task_gates()["internal_force_bounded"] is False
        env.drop_count = 0
        assert env.task_gates()["no_drop"] is True
        env.drop_count = 1
        assert env.task_gates()["no_drop"] is False
        env.transport_steps = 100
        env.dual_grasp_steps = 94
        assert env.task_gates()["shared_grasp_fraction"] is False
        env.max_transport_tilt = np.deg2rad(5)
        assert env.task_gates()["tilt_below_5deg"] is False
        env.max_payload_slip = 0.005
        assert env.task_gates()["payload_slip_below_5mm"] is False
        env.load_share_samples = 24
        env.load_share_sum[:] = 100.0
        assert env.task_gates()["both_arms_support_load"] is False
        env.data.qpos[env.object_qpos:env.object_qpos + 2] = env.start[:2] + [
            0.049,
            0,
        ]
        mujoco.mj_forward(env.model, env.data)
        assert env.task_gates()["translated_50mm"] is False
    finally:
        env.close()


@pytest.mark.parametrize("case_id", MANIPULATION_CASES)
def test_teacher_uses_ordered_actual_evaluator_transitions(case_id):
    env, stages = run_teacher(case_id)
    try:
        assert stages == list(EVALUATOR_STAGES)
        assert env.task_gates()["ordered_completion"] is True
        assert env.task_gates()["released_and_settled_2s"] is True
    finally:
        env.close()


@pytest.mark.parametrize("case_id", MANIPULATION_CASES)
def test_terminal_gate_rejects_each_nonterminal_current_state(case_id):
    env, _ = run_teacher(case_id, seed=91)
    try:
        assert env.task_gates()["released_and_settled_2s"] is True

        env.object_supported = False
        assert env.task_gates()["released_and_settled_2s"] is False
        env.object_supported = True

        original_qpos = env.data.qpos.copy()
        env.data.qpos[env.object_qpos] += 0.020
        mujoco.mj_forward(env.model, env.data)
        assert env.task_gates()["released_and_settled_2s"] is False
        env.data.qpos[:] = original_qpos
        mujoco.mj_forward(env.model, env.data)

        for addresses in env.qpos_indices:
            env.data.qpos[addresses[5]] = 0.001
        mujoco.mj_forward(env.model, env.data)
        assert env.task_gates()["released_and_settled_2s"] is False
        env.data.qpos[:] = original_qpos
        mujoco.mj_forward(env.model, env.data)

        original_goal = env.goal.copy()
        env.data.qpos[env.object_qpos:env.object_qpos + 3] = env.tip(0)
        env.goal = env.tip(0)
        mujoco.mj_forward(env.model, env.data)
        assert env.task_gates()["released_and_settled_2s"] is False
        env.data.qpos[:] = original_qpos
        env.goal = original_goal
        mujoco.mj_forward(env.model, env.data)

        env.contacts[:] = True
        assert env.task_gates()["released_and_settled_2s"] is False
        env.contacts[:] = False

        object_dof = env.model.jnt_dofadr[env.model.joint("object_free").id]
        original_qvel = env.data.qvel.copy()
        for start, stop in ((0, 3), (3, 6)):
            env.data.qvel[:] = original_qvel
            env.data.qvel[object_dof + start:object_dof + stop] = 0.5
            mujoco.mj_forward(env.model, env.data)
            assert env.task_gates()["released_and_settled_2s"] is False
        env.data.qvel[:] = original_qvel
        mujoco.mj_forward(env.model, env.data)

        env.settle_run = 99
        assert env.task_gates()["released_and_settled_2s"] is False
    finally:
        env.close()


@pytest.mark.parametrize("controller", ["zero", "open_gripper"])
def test_pick_place_negative_controls_do_not_pass(controller):
    env = arm.ArmEnv("arm-pick-place-v1")
    env.reset(seed=31)
    try:
        while not (env.terminated or env.truncated):
            if controller == "zero":
                action = np.zeros(env.action_dim, dtype=np.float32)
            else:
                action = env.teacher_action()
                action[5::6] = 1.0
            env.step(action)
        metrics = env.metrics()
        assert metrics["passed"] is False
        assert metrics["gates"]["released_and_settled_2s"] is False
    finally:
        env.close()
