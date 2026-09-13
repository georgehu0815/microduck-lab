from __future__ import annotations

import pytest

from microduck_arm_v1c.safety import Supervisor, evaluate_faults


def _live_supervisor() -> Supervisor:
    supervisor = Supervisor(command_deadline_s=0.1, heartbeat_deadline_s=0.2)
    assert supervisor.record_command(1.0, 10)
    assert supervisor.record_heartbeat(1.0, 20)
    assert supervisor.arm(1.0)
    return supervisor


@pytest.mark.parametrize("channel", ["command", "heartbeat"])
@pytest.mark.parametrize("invalid", ["missing", "stale", "replayed", "future"])
def test_arm_rejects_invalid_packets_before_arming(channel, invalid):
    supervisor = Supervisor()
    for stream in ("command", "heartbeat"):
        record = getattr(supervisor, f"record_{stream}")
        if stream == channel and invalid == "missing":
            continue
        timestamp = 1.0
        if stream == channel:
            if invalid == "stale":
                timestamp = 0.0
            elif invalid == "future":
                timestamp = 2.0
        assert record(timestamp, 1)
        if stream == channel and invalid == "replayed":
            assert not record(timestamp, 1)

    assert not supervisor.arm(1.0)
    assert supervisor.armed is False


def test_arm_requires_explicit_clock_and_fresh_packets():
    supervisor = Supervisor()
    with pytest.raises(TypeError):
        supervisor.arm()
    assert not supervisor.arm(0.0)
    supervisor.record_command(1.0, 1)
    supervisor.record_heartbeat(1.0, 1)
    assert supervisor.arm(1.0)
    assert not supervisor.arm(2.0)
    assert supervisor.armed is False


@pytest.mark.parametrize("fault", ["estop", "overcurrent", "undervoltage"])
def test_arm_rejects_latched_fault_until_explicit_reset(fault):
    supervisor = _live_supervisor()
    supervisor.evaluate(1.01, **{fault: True})
    supervisor.disarm()
    assert not supervisor.arm(1.02)
    assert supervisor.armed is False
    assert supervisor.reset()
    assert supervisor.arm(1.03)


@pytest.mark.parametrize("now_s", [float("nan"), float("inf"), -1.0, 0.9])
def test_arm_rejects_invalid_or_backwards_clock(now_s):
    supervisor = _live_supervisor()
    with pytest.raises(ValueError):
        supervisor.arm(now_s)
    assert supervisor.armed is False


def test_command_and_heartbeat_deadlines_are_independent():
    supervisor = _live_supervisor()
    assert supervisor.record_heartbeat(1.15, 21)

    status = supervisor.evaluate(1.21)

    assert status["stop_required"]
    assert status["active_faults"] == ["command_timeout"]
    assert status["command_age_s"] == pytest.approx(0.21)
    assert status["heartbeat_age_s"] == pytest.approx(0.06)


def test_repeated_read_or_heartbeat_packets_cannot_mask_stalled_policy():
    supervisor = _live_supervisor()
    assert supervisor.record_heartbeat(1.05, 21)
    assert supervisor.record_heartbeat(1.10, 22)

    status = supervisor.evaluate(1.101)

    assert "command_timeout" in status["active_faults"]
    assert "heartbeat_timeout" not in status["active_faults"]
    assert not status["motion_allowed"]


def test_timestamp_and_sequence_replays_are_rejected_until_fresh_packet():
    supervisor = _live_supervisor()

    assert not supervisor.record_command(1.0, 11)
    assert "command_replay" in supervisor.evaluate(1.01)["active_faults"]
    assert not supervisor.record_command(1.01, 10)
    assert supervisor.record_command(1.02, 11)
    assert "command_replay" not in supervisor.evaluate(1.02)["active_faults"]

    assert not supervisor.record_heartbeat(0.99, 21)
    assert "heartbeat_replay" in supervisor.evaluate(1.02)["active_faults"]


@pytest.mark.parametrize("fault", ["estop", "overcurrent", "undervoltage"])
def test_faults_latch_and_reset_requires_disarmed_clear_inputs(fault):
    supervisor = _live_supervisor()

    status = supervisor.evaluate(1.01, **{fault: True})
    assert fault in status["latched_faults"]
    assert not supervisor.reset()

    supervisor.disarm()
    assert not supervisor.reset(**{fault: True})
    assert fault in supervisor.evaluate(1.02)["latched_faults"]
    assert supervisor.reset()
    assert supervisor.evaluate(1.03)["latched_faults"] == []


def test_evaluation_time_must_be_monotonic():
    supervisor = _live_supervisor()
    supervisor.evaluate(1.1)

    with pytest.raises(ValueError, match="monotonic"):
        supervisor.evaluate(1.09)


def test_scenario_table_asserts_only_supervisor_logic():
    report = evaluate_faults()

    assert report["passed"]
    assert report["scope"] == "supervisorlogic_only"
    assert report["release_status"] == "hardware_unverified"
    assert report["hardware_verified"] is False
    assert report["hardware_release"] is False
    assert report["hardware_claims"] == []
    assert report["physical_tests_executed"] is False
    assert report["real_motor_io"] is False
    assert set(report["tests"]) == {
        "short",
        "reversal",
        "DATA short",
        "overcurrent",
        "stall",
        "disconnect",
        "intermittent",
        "controller loss",
        "estop",
    }
    for name, test in report["tests"].items():
        assert test["hypothetical_physical_scenario"] == name
        assert test["state_machine_assertion_passed"] is True
        assert test["physical_fault_qualified"] is False
        assert test["actual"] == test["expected"]
        assert "status" not in test
        assert "passed" not in test
    assert all(
        test["evidence"] == "emulated_inputs_only"
        for test in report["tests"].values()
    )
    assert "does not guarantee torque-off" in report["bus_watchdog"]
    assert "does not refresh command age" in report["repeated_read_packets_note"]
