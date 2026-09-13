from types import SimpleNamespace

import numpy as np

from microduck_arm_experiments.tennis_controller import BinNavigator, NavigationProfile
from microduck_arm_experiments.tennis_navigation import (
    NAVIGATION_PROFILES,
    forecast_navigation,
    select_navigation_profile,
)
from microduck_arm_experiments.tennis_return import TennisReturnEnv


def test_navigation_forecast_isolates_live_physics_and_policy_state():
    environment = TennisReturnEnv(gripper="wide_candidate", max_steps=3)
    try:
        environment.reset(0)
        environment.controller.navigator = BinNavigator()
        robot = environment.robot
        positions = robot.data.qpos.copy()
        velocities = robot.data.qvel.copy()
        targets = robot.command_targets.copy()
        history = robot.leg_policy.last_action.copy()
        result = forecast_navigation(environment, NAVIGATION_PROFILES[1])
        assert result["forecast_steps"] == 3
        assert not result["success"]
        assert result["failure_reason"] == "timeout"
        assert result["profile"]["warmup_s"] == 1.5
        assert not environment.controller.navigation_forecasting
        assert environment.controller.navigator.profile == NavigationProfile()
        assert environment.steps == 0
        np.testing.assert_array_equal(robot.data.qpos, positions)
        np.testing.assert_array_equal(robot.data.qvel, velocities)
        np.testing.assert_array_equal(robot.command_targets, targets)
        np.testing.assert_array_equal(robot.leg_policy.last_action, history)
        assert not np.any(robot.data.qfrc_applied)
        assert not np.any(robot.data.xfrc_applied)
    finally:
        environment.close()


def test_navigation_selection_requires_complete_success_and_preserves_order(monkeypatch):
    calls = []

    def forecast(environment, profile):
        calls.append(profile)
        return {"success": profile == environment, "profile": profile}

    monkeypatch.setattr("microduck_arm_experiments.tennis_navigation.forecast_navigation", forecast)
    selected, evidence = select_navigation_profile(NAVIGATION_PROFILES[3])
    assert selected == NAVIGATION_PROFILES[3]
    assert calls == list(NAVIGATION_PROFILES[:4])
    assert len(evidence) == 4
    selected, evidence = select_navigation_profile(None)
    assert selected is None
    assert len(evidence) == len(NAVIGATION_PROFILES)
    assert all(not record["success"] for record in evidence)


def test_navigation_forecast_fails_closed_and_closes_clone_on_error(monkeypatch):
    clone = SimpleNamespace(
        controller=SimpleNamespace(navigator=SimpleNamespace()), done=False,
        closed=False,
    )

    def teacher_action():
        raise RuntimeError("synthetic inference failure")

    clone.teacher_action = teacher_action
    clone.close = lambda: setattr(clone, "closed", True)
    monkeypatch.setattr("microduck_arm_experiments.tennis_navigation._clone_environment", lambda _: clone)
    result = forecast_navigation(SimpleNamespace(steps=0), NavigationProfile())
    assert not result["success"]
    assert result["failure_reason"] == "navigation_forecast_error"
    assert result["simulation_only"] and not result["hardware_release"]
    assert not result["ppo"] and not result["hardware_eligible"]
    assert clone.closed
