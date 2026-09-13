import copy
import json
from types import SimpleNamespace

import numpy as np

from microduck_arm_experiments.tennis_release import (
    _clone_environment,
    screen_docking_pose,
    select_release_strategy,
    supported_release_targets,
)


def test_docking_forecast_preserves_live_model_state_and_policy_history():
    from microduck_arm_experiments.tennis_controller import BinNavigator
    from microduck_arm_experiments.tennis_return import TennisReturnEnv

    environment = TennisReturnEnv(gripper="wide_candidate", max_steps=3)
    try:
        environment.reset(0)
        environment.controller.navigator = BinNavigator()
        robot = environment.robot
        positions = robot.data.qpos.copy()
        velocities = robot.data.qvel.copy()
        controls = robot.data.ctrl.copy()
        forces = robot.model.actuator_forcerange.copy()
        history = robot.leg_policy.last_action.copy()
        prediction = screen_docking_pose(environment)
        assert not prediction["success"]
        assert prediction["forecast_steps"] == 3
        assert prediction["simulation_only"]
        assert not prediction["hardware_release"]
        np.testing.assert_array_equal(robot.data.qpos, positions)
        np.testing.assert_array_equal(robot.data.qvel, velocities)
        np.testing.assert_array_equal(robot.data.ctrl, controls)
        np.testing.assert_array_equal(robot.model.actuator_forcerange, forces)
        np.testing.assert_array_equal(robot.leg_policy.last_action, history)
        assert environment.steps == 0
        assert not environment.controller.navigator.arrived
        assert not environment.controller.docking_forecasting
    finally:
        environment.close()


def test_docking_selects_first_verified_posture_and_rejects_all_failed(monkeypatch):
    calls = []

    def forecast(environment, drop):
        calls.append(drop)
        return {"success": drop == environment, "placement_drop_m": drop}

    monkeypatch.setattr(
        "microduck_arm_experiments.tennis_release._screen_docking_candidate", forecast
    )
    selected = screen_docking_pose(.035)
    assert selected["success"]
    assert selected["placement_drop_m"] == .035
    assert calls == [.025, .030, .035]
    rejected = screen_docking_pose(None)
    assert not rejected["success"]
    assert rejected["placement_drop_m"] is None
    assert len(rejected["candidates"]) == 4


class ImmutableSession:
    def __deepcopy__(self, memo):
        raise AssertionError("ONNX session must be memo-shared")


class LiveRenderer:
    def __init__(self):
        self.closed = False

    def __deepcopy__(self, memo):
        raise AssertionError("live renderer must be omitted from clone")

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self):
        self.session = ImmutableSession()
        self.history = []


class FakeEnvironment:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.robot = SimpleNamespace(
            leg_policy=FakePolicy(),
            walk_policy=None,
            _renderer=LiveRenderer(),
        )
        self.monitor = SimpleNamespace(
            phase="release", success=False, failure=None
        )
        self.controller = SimpleNamespace(
            environment=self,
            navigator=SimpleNamespace(policy=FakePolicy()),
            release_strategy=None,
            stage="release",
        )
        self.done = False
        self.steps = 0
        self.closed = False

    def teacher_action(self):
        self.robot.leg_policy.history.append(self.steps)
        return np.zeros(1)

    def step(self, action):
        del action
        self.steps += 1
        outcome = self.outcomes[self.controller.release_strategy]
        self.done = True
        self.monitor.success = outcome
        self.monitor.failure = None if outcome else "unsafe_release"
        self.monitor.phase = "success" if outcome else "release"
        return None, self.done, {
            "success": outcome,
            "failure_reason": self.monitor.failure,
            "phase": self.monitor.phase,
            "controller_stage": self.controller.stage,
        }

    def close(self):
        self.closed = True


def test_environment_clone_is_independent_except_for_onnx_sessions():
    environment = FakeEnvironment({"legacy": True, "supported_hold": True})

    clone = _clone_environment(environment)

    assert clone is not environment
    assert clone.controller.environment is clone
    assert clone.robot.leg_policy is not environment.robot.leg_policy
    assert (
        clone.robot.leg_policy.session
        is environment.robot.leg_policy.session
    )
    assert (
        clone.controller.navigator.policy.session
        is environment.controller.navigator.policy.session
    )
    assert clone.robot._renderer is None
    assert not environment.robot._renderer.closed
    clone.robot.leg_policy.history.append("clone")
    assert environment.robot.leg_policy.history == []


def test_planner_prefers_legacy_and_does_not_mutate_live_state_or_history():
    environment = FakeEnvironment({"legacy": True, "supported_hold": True})
    environment.steps = 37
    before = copy.deepcopy(
        {
            "done": environment.done,
            "steps": environment.steps,
            "strategy": environment.controller.release_strategy,
            "history": environment.robot.leg_policy.history,
        }
    )

    strategy, evidence = select_release_strategy(environment)

    assert strategy == "legacy"
    assert [record["strategy"] for record in evidence] == ["legacy"]
    assert evidence[0]["success"]
    assert evidence[0]["start_step"] == 37
    assert evidence[0]["terminal_step"] == 38
    assert evidence[0]["forecast_steps"] == 1
    assert evidence[0]["simulation_only"]
    assert not evidence[0]["hardware_release"]
    assert evidence[0]["forecast_source"] == "simulator_truth_clone_teacher"
    assert not evidence[0]["ppo"]
    assert not evidence[0]["hardware_eligible"]
    assert json.loads(json.dumps(evidence)) == evidence
    assert environment.done == before["done"]
    assert environment.steps == before["steps"]
    assert environment.controller.release_strategy == before["strategy"]
    assert environment.robot.leg_policy.history == before["history"]
    assert not environment.robot._renderer.closed


def test_planner_uses_supported_hold_only_after_legacy_fails():
    environment = FakeEnvironment({"legacy": False, "supported_hold": True})

    strategy, evidence = select_release_strategy(environment)

    assert strategy == "supported_hold"
    assert [record["strategy"] for record in evidence] == [
        "legacy",
        "supported_hold",
    ]
    assert [record["success"] for record in evidence] == [False, True]


def test_all_candidates_failing_returns_none_instead_of_unsafe_fallback():
    environment = FakeEnvironment({"legacy": False, "supported_hold": False})

    strategy, evidence = select_release_strategy(environment)

    assert strategy is None
    assert len(evidence) == 2
    assert not any(record["success"] for record in evidence)


def test_inconsistent_positive_info_cannot_authorize_release():
    class InconsistentEnvironment(FakeEnvironment):
        def step(self, action):
            del action
            self.steps += 1
            self.done = True
            self.monitor.success = False
            self.monitor.failure = None
            self.monitor.phase = "success"
            return None, self.done, {
                "success": True,
                "failure_reason": None,
                "phase": "success",
                "controller_stage": self.controller.stage,
            }

    environment = InconsistentEnvironment(
        {"legacy": True, "supported_hold": True}
    )

    strategy, evidence = select_release_strategy(environment)

    assert strategy is None
    assert len(evidence) == 2
    assert not any(record["success"] for record in evidence)


def test_supported_hold_freezes_arm_uses_active_legs_and_opens_slowly(monkeypatch):
    qpos = np.linspace(-.2, .2, 15)
    command_targets = np.linspace(-.3, .3, 15)
    command_targets[-1] = -.16
    robot = SimpleNamespace(
        home=np.zeros(15),
        command_targets=command_targets,
        qpos_indices=np.arange(15),
        dof_indices=np.arange(15),
        data=SimpleNamespace(
            time=10.,
            qpos=qpos,
            qfrc_bias=np.r_[np.zeros(10), [.2, -.2, .1, -.1], 0.],
        ),
        design={
            "arm": {
                "gravity_compensation": {
                    "maximum_position_offset_rad": .05
                },
                "simulation_kp": 10.,
            }
        },
        dt=.1,
    )
    sample = {
        "bottom_supported": True,
        "ball_inside_bin": True,
        "forbidden_contacts": [],
        "ball_speed_m_s": 0.,
    }
    env = SimpleNamespace(robot=robot, sample=lambda: sample)
    controller = SimpleNamespace(
        environment=env,
        placement_crouch=np.full(10, .01),
        release_targets=None,
        release_started=None,
        safe_release_joints=None,
        release_jaw=None,
        release_settled_s=.2,
        release_opening=False,
        release_blocked=False,
        cached_release_screen_passed=False,
        stage="release",
    )
    monkeypatch.setattr(
        "microduck_arm_experiments.tennis_controller.cached_release_clear",
        lambda actual_robot, joints: actual_robot is robot and joints.shape == (4,),
    )
    targets = np.linspace(-.1, .1, 15)

    result = supported_release_targets(controller, targets)

    expected_legs = targets[:10] + .01
    expected_legs[[2, 4, 7, 9]] += -.2 * np.array([-1, 1, 1, -1])
    np.testing.assert_allclose(result[:10], expected_legs)
    compensation = np.array([.02, -.02, .01, -.01])
    np.testing.assert_allclose(
        result[10:14],
        robot.command_targets[10:14] - compensation,
    )
    assert controller.release_opening
    assert np.isclose(
        result[-1],
        robot.command_targets[-1] + .025 * robot.dt,
    )
    assert controller.stage == "release_supported_hold"

    jaw_before_support_loss = controller.release_jaw
    sample["forbidden_contacts"] = ["robot_bin_collision"]
    blocked = supported_release_targets(controller, targets)
    assert controller.release_blocked
    assert blocked[-1] == jaw_before_support_loss
    assert controller.stage == "release_supported_hold_blocked"

    controller.release_targets = None
    controller.release_opening = False
    controller.release_settled_s = 1.
    sample["forbidden_contacts"] = []
    screened = []

    def reject_command_pose(actual_robot, joints):
        screened.append(joints.copy())
        return not np.array_equal(joints, actual_robot.command_targets[10:14])

    monkeypatch.setattr(
        "microduck_arm_experiments.tennis_controller.cached_release_clear", reject_command_pose
    )
    blocked = supported_release_targets(controller, targets)
    assert len(screened) == 2
    assert not controller.cached_release_screen_passed
    assert not controller.release_opening
    assert blocked[-1] == robot.command_targets[-1]


def test_supported_hold_fails_closed_when_initial_screen_rejects(monkeypatch):
    robot = SimpleNamespace(
        command_targets=np.zeros(15),
        qpos_indices=np.arange(15),
        dof_indices=np.arange(15),
        data=SimpleNamespace(
            time=3., qpos=np.zeros(15), qfrc_bias=np.zeros(15)
        ),
        design={
            "arm": {
                "gravity_compensation": {
                    "maximum_position_offset_rad": .05
                },
                "simulation_kp": 10.,
            }
        },
        dt=.1,
    )
    sample = {
        "bottom_supported": True,
        "ball_inside_bin": True,
        "forbidden_contacts": [],
        "ball_speed_m_s": 0.,
    }
    env = SimpleNamespace(robot=robot, sample=lambda: sample)
    controller = SimpleNamespace(
        environment=env,
        placement_crouch=np.zeros(10),
        release_targets=None,
        release_started=None,
        safe_release_joints=None,
        release_jaw=None,
        release_settled_s=1.,
        release_opening=False,
        release_blocked=False,
        cached_release_screen_passed=False,
        stage="release",
    )
    monkeypatch.setattr(
        "microduck_arm_experiments.tennis_controller.cached_release_clear",
        lambda robot, joints: False,
    )

    result = supported_release_targets(controller, np.zeros(15))

    assert controller.release_blocked
    assert not controller.cached_release_screen_passed
    assert not controller.release_opening
    assert result[-1] == 0.
    assert controller.stage == "release_supported_hold_blocked"


def test_no_safe_plan_holds_jaw_without_screening_or_success_latch(monkeypatch):
    robot = SimpleNamespace(
        command_targets=np.r_[np.zeros(14), -.16],
        qpos_indices=np.arange(15),
        dof_indices=np.arange(15),
        data=SimpleNamespace(
            time=3., qpos=np.zeros(15), qfrc_bias=np.zeros(15)
        ),
        design={
            "arm": {
                "gravity_compensation": {
                    "maximum_position_offset_rad": .05
                },
                "simulation_kp": 10.,
            }
        },
        dt=.1,
    )
    env = SimpleNamespace(
        robot=robot,
        sample=lambda: {
            "bottom_supported": True,
            "ball_inside_bin": True,
            "forbidden_contacts": [],
            "ball_speed_m_s": 0.,
        },
    )
    controller = SimpleNamespace(
        environment=env,
        placement_crouch=np.zeros(10),
        release_targets=None,
        release_started=None,
        safe_release_joints=None,
        release_jaw=None,
        release_settled_s=1.,
        release_opening=True,
        release_blocked=False,
        cached_release_screen_passed=True,
        stage="release",
    )
    monkeypatch.setattr(
        "microduck_arm_experiments.tennis_controller.cached_release_clear",
        lambda robot, joints: (_ for _ in ()).throw(
            AssertionError("no-safe-plan hold must not screen for opening")
        ),
    )

    result = supported_release_targets(
        controller, np.zeros(15), allow_opening=False
    )

    assert result[-1] == -.16
    assert controller.release_blocked
    assert not controller.release_opening
    assert not controller.cached_release_screen_passed
    assert controller.stage == "release_no_safe_plan"
