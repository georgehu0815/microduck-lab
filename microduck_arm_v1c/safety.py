from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


LATCHED_FAULTS = ("estop", "overcurrent", "undervoltage")
BUS_WATCHDOG_LIMITATION = (
    "Bus Watchdog requests a stop; it does not guarantee torque-off."
)


def _nonnegative_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return value


@dataclass
class Supervisor:
    """Deterministic simulation/HIL safety state machine with no device I/O."""

    command_deadline_s: float = 0.1
    heartbeat_deadline_s: float = 0.1
    armed: bool = False
    last_command_timestamp_s: float | None = field(default=None, init=False)
    last_command_seq: int | None = field(default=None, init=False)
    last_heartbeat_timestamp_s: float | None = field(default=None, init=False)
    last_heartbeat_seq: int | None = field(default=None, init=False)
    latched_faults: set[str] = field(default_factory=set, init=False)
    _command_replay: bool = field(default=False, init=False, repr=False)
    _heartbeat_replay: bool = field(default=False, init=False, repr=False)
    _last_evaluation_s: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.command_deadline_s = _nonnegative_finite(
            self.command_deadline_s, "command_deadline_s"
        )
        self.heartbeat_deadline_s = _nonnegative_finite(
            self.heartbeat_deadline_s, "heartbeat_deadline_s"
        )

    @staticmethod
    def _valid_packet(
        timestamp_s: float,
        seq: int,
        previous_timestamp_s: float | None,
        previous_seq: int | None,
    ) -> tuple[float, int] | None:
        timestamp_s = _nonnegative_finite(timestamp_s, "timestamp_s")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise ValueError("seq must be a nonnegative integer")
        if previous_timestamp_s is not None and timestamp_s <= previous_timestamp_s:
            return None
        if previous_seq is not None and seq <= previous_seq:
            return None
        return timestamp_s, seq

    def record_command(self, timestamp_s: float, seq: int) -> bool:
        packet = self._valid_packet(
            timestamp_s,
            seq,
            self.last_command_timestamp_s,
            self.last_command_seq,
        )
        if packet is None:
            self._command_replay = True
            return False
        self.last_command_timestamp_s, self.last_command_seq = packet
        self._command_replay = False
        return True

    def record_heartbeat(self, timestamp_s: float, seq: int) -> bool:
        packet = self._valid_packet(
            timestamp_s,
            seq,
            self.last_heartbeat_timestamp_s,
            self.last_heartbeat_seq,
        )
        if packet is None:
            self._heartbeat_replay = True
            return False
        self.last_heartbeat_timestamp_s, self.last_heartbeat_seq = packet
        self._heartbeat_replay = False
        return True

    accept_command = record_command
    accept_heartbeat = record_heartbeat

    def arm(self, now_s: float) -> bool:
        self.armed = False
        if self.evaluate(now_s)["active_faults"]:
            return False
        self.armed = True
        return True

    def disarm(self) -> None:
        self.armed = False

    def reset(
        self,
        *,
        estop: bool = False,
        overcurrent: bool = False,
        undervoltage: bool = False,
    ) -> bool:
        if self.armed or estop or overcurrent or undervoltage:
            return False
        self.latched_faults.clear()
        return True

    def evaluate(
        self,
        now_s: float,
        *,
        estop: bool = False,
        overcurrent: bool = False,
        undervoltage: bool = False,
    ) -> dict[str, Any]:
        now_s = _nonnegative_finite(now_s, "now_s")
        if self._last_evaluation_s is not None and now_s < self._last_evaluation_s:
            raise ValueError("now_s must be monotonic")
        self._last_evaluation_s = now_s

        asserted = {
            "estop": bool(estop),
            "overcurrent": bool(overcurrent),
            "undervoltage": bool(undervoltage),
        }
        self.latched_faults.update(
            fault for fault, active in asserted.items() if active
        )

        command_age = self._age(now_s, self.last_command_timestamp_s)
        heartbeat_age = self._age(now_s, self.last_heartbeat_timestamp_s)
        active_faults = set(self.latched_faults)
        if command_age is None:
            active_faults.add("command_missing")
        elif command_age < 0.0:
            active_faults.add("command_from_future")
        elif command_age > self.command_deadline_s:
            active_faults.add("command_timeout")
        if heartbeat_age is None:
            active_faults.add("heartbeat_missing")
        elif heartbeat_age < 0.0:
            active_faults.add("heartbeat_from_future")
        elif heartbeat_age > self.heartbeat_deadline_s:
            active_faults.add("heartbeat_timeout")
        if self._command_replay:
            active_faults.add("command_replay")
        if self._heartbeat_replay:
            active_faults.add("heartbeat_replay")

        motion_allowed = self.armed and not active_faults
        return {
            "scope": "supervisorlogic_only",
            "real_motor_io": False,
            "armed": self.armed,
            "motion_allowed": motion_allowed,
            "stop_required": self.armed and not motion_allowed,
            "active_faults": sorted(active_faults),
            "latched_faults": sorted(self.latched_faults),
            "command_age_s": command_age,
            "heartbeat_age_s": heartbeat_age,
            "command_deadline_s": self.command_deadline_s,
            "heartbeat_deadline_s": self.heartbeat_deadline_s,
            "bus_watchdog": BUS_WATCHDOG_LIMITATION,
        }

    @staticmethod
    def _age(now_s: float, timestamp_s: float | None) -> float | None:
        return None if timestamp_s is None else now_s - timestamp_s


def _primed_supervisor() -> Supervisor:
    supervisor = Supervisor()
    supervisor.record_command(0.0, 0)
    supervisor.record_heartbeat(0.0, 0)
    supervisor.arm(0.0)
    return supervisor


def _projection(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "stop_required": status["stop_required"],
        "active_faults": status["active_faults"],
        "latched_faults": status["latched_faults"],
    }


def evaluate_faults() -> dict[str, Any]:
    """Check supervisor logic; physical scenario names are hypothetical only."""

    cases: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []

    short = _primed_supervisor()
    cases.append(
        (
            "short",
            "emulated overcurrent input",
            {
                "stop_required": True,
                "active_faults": ["overcurrent"],
                "latched_faults": ["overcurrent"],
            },
            _projection(short.evaluate(0.01, overcurrent=True)),
        )
    )

    reversal = _primed_supervisor()
    cases.append(
        (
            "reversal",
            "emulated undervoltage input",
            {
                "stop_required": True,
                "active_faults": ["undervoltage"],
                "latched_faults": ["undervoltage"],
            },
            _projection(reversal.evaluate(0.01, undervoltage=True)),
        )
    )

    data_short = _primed_supervisor()
    cases.append(
        (
            "DATA short",
            "emulated loss of command and heartbeat updates",
            {
                "stop_required": True,
                "active_faults": ["command_timeout", "heartbeat_timeout"],
                "latched_faults": [],
            },
            _projection(data_short.evaluate(0.101)),
        )
    )

    overcurrent = _primed_supervisor()
    cases.append(
        (
            "overcurrent",
            "emulated overcurrent input",
            {
                "stop_required": True,
                "active_faults": ["overcurrent"],
                "latched_faults": ["overcurrent"],
            },
            _projection(overcurrent.evaluate(0.01, overcurrent=True)),
        )
    )

    stall = _primed_supervisor()
    stall.record_heartbeat(0.08, 1)
    cases.append(
        (
            "stall",
            "emulated fresh read/heartbeat traffic with stale policy command",
            {
                "stop_required": True,
                "active_faults": ["command_timeout"],
                "latched_faults": [],
            },
            _projection(stall.evaluate(0.101)),
        )
    )

    disconnect = _primed_supervisor()
    cases.append(
        (
            "disconnect",
            "emulated loss of command and heartbeat updates",
            {
                "stop_required": True,
                "active_faults": ["command_timeout", "heartbeat_timeout"],
                "latched_faults": [],
            },
            _projection(disconnect.evaluate(0.101)),
        )
    )

    intermittent = _primed_supervisor()
    intermittent.record_command(0.0, 0)
    cases.append(
        (
            "intermittent",
            "emulated repeated command packet",
            {
                "stop_required": True,
                "active_faults": ["command_replay"],
                "latched_faults": [],
            },
            _projection(intermittent.evaluate(0.01)),
        )
    )

    controller_loss = _primed_supervisor()
    controller_loss.record_command(0.08, 1)
    cases.append(
        (
            "controller loss",
            "emulated fresh command traffic with stale controller heartbeat",
            {
                "stop_required": True,
                "active_faults": ["heartbeat_timeout"],
                "latched_faults": [],
            },
            _projection(controller_loss.evaluate(0.101)),
        )
    )

    estop = _primed_supervisor()
    cases.append(
        (
            "estop",
            "emulated E-stop input",
            {
                "stop_required": True,
                "active_faults": ["estop"],
                "latched_faults": ["estop"],
            },
            _projection(estop.evaluate(0.01, estop=True)),
        )
    )

    tests = {
        name: {
            "hypothetical_physical_scenario": name,
            "emulated_inputs": injection,
            "evidence": "emulated_inputs_only",
            "expected": expected,
            "actual": actual,
            "state_machine_assertion_passed": actual == expected,
            "physical_fault_qualified": False,
        }
        for name, injection, expected, actual in cases
    }
    return {
        "scope": "supervisorlogic_only",
        "scenario_note": (
            "Physical scenario names are hypothetical; emulated inputs test "
            "supervisor logic, not physical fault detection or protection."
        ),
        "release_status": "hardware_unverified",
        "hardware_verified": False,
        "hardware_release": False,
        "hardware_claims": [],
        "physical_tests_executed": False,
        "real_motor_io": False,
        "bus_watchdog": BUS_WATCHDOG_LIMITATION,
        "repeated_read_packets_note": (
            "Read or heartbeat traffic does not refresh command age and cannot "
            "mask a stalled policy."
        ),
        "tests": tests,
        "passed": all(
            test["state_machine_assertion_passed"] for test in tests.values()
        ),
    }
