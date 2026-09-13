"""Bounded simulation-only docking and release strategy screening."""

from __future__ import annotations

import copy

import numpy as np


RELEASE_STRATEGIES = ("legacy", "supported_hold")
FORECAST_STEPS = 800
SUPPORTED_JAW_SPEED_RAD_S = 0.025
FORECAST_PROVENANCE = {
    "forecast_source": "simulator_truth_clone_teacher",
    "ppo": False,
    "hardware_eligible": False,
    "simulation_only": True,
    "hardware_release": False,
}


def _onnx_session_memo(environment):
    """Share immutable ONNX handles and omit any live renderer from clones."""
    memo = {}
    robot = getattr(environment, "robot", None)
    renderer = getattr(robot, "_renderer", None)
    if renderer is not None:
        memo[id(renderer)] = None
    holders = [
        getattr(robot, "leg_policy", None),
        getattr(robot, "walk_policy", None),
    ]
    navigator = getattr(getattr(environment, "controller", None), "navigator", None)
    holders.append(getattr(navigator, "policy", None))
    for holder in holders:
        session = getattr(holder, "session", None)
        if session is not None:
            memo[id(session)] = session
    return memo


def _clone_environment(environment):
    return copy.deepcopy(environment, _onnx_session_memo(environment))


def _terminal_info(environment, latest):
    monitor = getattr(environment, "monitor", None)
    controller = getattr(environment, "controller", None)
    failure = (
        latest.get("failure_reason")
        or getattr(monitor, "failure", None)
    )
    success = bool(
        getattr(environment, "done", False)
        and latest.get("success", False)
        and getattr(monitor, "success", False)
        and failure is None
    )
    return {
        "success": success,
        "failure_reason": failure,
        "terminal_phase": latest.get("phase")
        or getattr(monitor, "phase", None),
        "controller_stage": latest.get("controller_stage")
        or getattr(controller, "stage", None),
    }


def _forecast_strategy(environment, strategy):
    clone = _clone_environment(environment)
    latest = {}
    rollout_steps = 0
    initial_step = getattr(clone, "steps", None)
    try:
        controller = clone.controller
        controller.release_strategy = strategy
        for rollout_steps in range(1, FORECAST_STEPS + 1):
            if getattr(clone, "done", False):
                break
            result = clone.step(clone.teacher_action())
            latest = result[-1] if isinstance(result[-1], dict) else {}
            if getattr(clone, "done", False):
                break
        terminal = _terminal_info(clone, latest)
        terminal_step = getattr(clone, "steps", None)
        forecast_steps = (
            terminal_step - initial_step
            if isinstance(initial_step, int) and isinstance(terminal_step, int)
            else rollout_steps
        )
        return {
            **FORECAST_PROVENANCE,
            "strategy": strategy,
            **terminal,
            "forecast_steps": forecast_steps,
            "start_step": initial_step,
            "terminal_step": terminal_step,
            "done": bool(getattr(clone, "done", False)),
            "horizon_exhausted": bool(
                not getattr(clone, "done", False)
                and forecast_steps >= FORECAST_STEPS
            ),
            "simulation_only": True,
            "hardware_release": False,
        }
    except Exception as error:
        terminal_step = getattr(clone, "steps", None)
        forecast_steps = (
            terminal_step - initial_step
            if isinstance(initial_step, int) and isinstance(terminal_step, int)
            else rollout_steps
        )
        return {
            **FORECAST_PROVENANCE,
            "strategy": strategy,
            "success": False,
            "failure_reason": "forecast_error",
            "terminal_phase": getattr(getattr(clone, "monitor", None), "phase", None),
            "controller_stage": getattr(
                getattr(clone, "controller", None), "stage", None
            ),
            "forecast_steps": forecast_steps,
            "start_step": initial_step,
            "terminal_step": terminal_step,
            "done": bool(getattr(clone, "done", False)),
            "horizon_exhausted": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "simulation_only": True,
            "hardware_release": False,
        }
    finally:
        close = getattr(clone, "close", None)
        if close is not None:
            close()


def select_release_strategy(environment):
    """Return the first forecast-successful strategy and serializable evidence.

    Legacy is intentionally screened first. A failed or incomplete forecast is
    never converted into a least-bad release recommendation.
    """
    evidence = []
    strategies = ("half_height_hold", "half_height_slow", "half_height_fast") if getattr(environment, "release_mode", "supported") == "half_height" else RELEASE_STRATEGIES
    for strategy in strategies:
        record = _forecast_strategy(environment, strategy)
        evidence.append(record)
        if record["success"]:
            return strategy, evidence
    return None, evidence


def _screen_docking_candidate(environment, drop):
    clone = _clone_environment(environment)
    latest = {}
    steps = 0
    try:
        clone.controller.docking_forecasting = True
        clone.controller.docking_qualified = True
        clone.controller.navigator.arrived = True
        clone.controller.placement_drop_m = drop
        for steps in range(1, min(2000, environment.max_steps - environment.steps) + 1):
            _, _, latest = clone.step(clone.teacher_action())
            if clone.done:
                break
        terminal = _terminal_info(clone, latest)
        return {
            **FORECAST_PROVENANCE,
            **terminal,
            "time_s": float(environment.robot.data.time),
            "position_m": environment.robot.data.xpos[environment.robot.trunk_id].tolist(),
            "forecast_steps": steps,
            "placement_drop_m": drop,
        }
    finally:
        clone.close()


def screen_docking_pose(environment):
    candidates = []
    for drop in (.025, .030, .035, .040):
        prediction = _screen_docking_candidate(environment, drop)
        candidates.append(prediction)
        if prediction["success"]:
            return {**prediction, "candidates": candidates}
    return {**candidates[-1], "placement_drop_m": None, "candidates": candidates}


def _release_support_is_live(sample):
    return bool(
        sample["bottom_supported"]
        and sample["ball_inside_bin"]
        and not sample["forbidden_contacts"]
    )


def supported_release_targets(controller, targets, allow_opening=True, require_support=True, jaw_speed=SUPPORTED_JAW_SPEED_RAD_S):
    """Hold the arm command, retaining active legs and optionally opening."""
    from .tennis_controller import cached_release_clear

    env = controller.environment
    robot = env.robot
    result = np.asarray(targets).copy()
    crouch = controller.placement_crouch
    if crouch is None:
        crouch = np.zeros(10)
    result[:10] = targets[:10] + crouch
    result[[2, 4, 7, 9]] += -.2 * np.array([-1, 1, 1, -1])

    if controller.release_targets is None:
        controller.release_targets = robot.command_targets.copy()
        controller.release_started = float(robot.data.time)
        controller.safe_release_joints = robot.data.qpos[
            robot.qpos_indices[10:14]
        ].copy()
        controller.release_jaw = float(robot.command_targets[-1])
        if allow_opening:
            sample = env.sample()
            controller.cached_release_screen_passed = bool(
                (_release_support_is_live(sample) if require_support else sample["release_height_ok"] and not sample["forbidden_contacts"])
                and cached_release_clear(robot, controller.safe_release_joints)
                and cached_release_clear(robot, controller.release_targets[10:14])
            )
        else:
            controller.cached_release_screen_passed = False
        controller.release_blocked = not controller.cached_release_screen_passed

    result[10:14] = controller.release_targets[10:14]
    compensation = robot.design["arm"]["gravity_compensation"][
        "maximum_position_offset_rad"
    ]
    result[10:14] -= np.clip(
        robot.data.qfrc_bias[robot.dof_indices[10:14]]
        / robot.design["arm"]["simulation_kp"],
        -compensation,
        compensation,
    )

    if not allow_opening:
        controller.release_blocked = True
        controller.release_opening = False
        result[-1] = controller.release_jaw
        controller.stage = "release_no_safe_plan"
        return result

    sample = env.sample()
    live_support = _release_support_is_live(sample) if require_support else sample.get("ball_physically_contained_in_bin", sample["ball_inside_bin"]) and not sample["forbidden_contacts"]
    settled = live_support and sample["ball_speed_m_s"] < .005
    controller.release_settled_s = (
        controller.release_settled_s + robot.dt if settled else 0.
    )
    if not live_support:
        controller.release_blocked = True
    opening_ready = (
        controller.cached_release_screen_passed
        and live_support
        and controller.release_settled_s >= .3
    )
    controller.release_opening |= opening_ready
    if controller.release_opening and live_support:
        controller.release_jaw = min(
            0.,
            controller.release_jaw + jaw_speed * robot.dt,
        )
    result[-1] = controller.release_jaw
    controller.stage = (
        ("release_supported_hold" if require_support else "release_half_height_drop")
        if controller.cached_release_screen_passed and live_support
        else "release_supported_hold_blocked"
    )
    return result
