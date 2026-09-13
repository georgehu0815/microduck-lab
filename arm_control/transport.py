"""Arm transport boundary and deterministic simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NoReturn, Protocol

from .contracts import ARM_JOINTS, DeviceContract


HARDWARE_UNAVAILABLE_REASON = (
    "hardware arm transport is unavailable in this phase: servo identity, torque "
    "configuration, physical E-stop feedback, exclusive body ownership, and an "
    "independent watchdog are not commissioned"
)


class TransportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArmSample:
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    currents_ma: tuple[float, ...]
    temperatures_c: tuple[float, ...]
    voltage_v: float

    def validate(self, contract: DeviceContract) -> None:
        fields = (
            ("positions", self.positions),
            ("velocities", self.velocities),
            ("currents_ma", self.currents_ma),
            ("temperatures_c", self.temperatures_c),
        )
        for name, values in fields:
            if len(values) != len(contract.joints) or any(not math.isfinite(value) for value in values):
                raise TransportError(f"invalid {name} sample")
        if not math.isfinite(self.voltage_v):
            raise TransportError("invalid voltage sample")
        for joint, value in zip(contract.joints, self.positions):
            if value < joint.minimum or value > joint.maximum:
                raise TransportError(f"{joint.name} measured outside configured limits")


class ArmTransport(Protocol):
    hardware: bool

    def read(self) -> ArmSample: ...

    def write_positions(self, positions: tuple[float, ...]) -> None: ...

    def protective_stop(self) -> None: ...

    def emergency_stop(self) -> None: ...

    def estop_released(self) -> bool: ...

    def close(self) -> None: ...


class LockedTransport:
    hardware = False
    locked = True

    def _locked(self) -> NoReturn:
        raise TransportError("transport is locked")

    def read(self) -> ArmSample:
        self._locked()

    def write_positions(self, positions: tuple[float, ...]) -> None:
        self._locked()

    def protective_stop(self) -> None:
        self._locked()

    def emergency_stop(self) -> None:
        self._locked()

    def estop_released(self) -> bool:
        return False

    def close(self) -> None:
        return None


class SimulatedTransport:
    hardware = False
    locked = False

    def __init__(self, contract: DeviceContract):
        self.contract = contract
        self.positions: tuple[float, ...] = tuple(
            0.015 if joint.name == "gripper_width" else min(max(0.0, joint.minimum), joint.maximum)
            for joint in ARM_JOINTS
        )
        self.velocities: tuple[float, ...] = (0.0,) * 6
        self.writes: list[tuple[float, ...]] = []
        self.protective_stops = 0
        self.estops = 0
        self.closed = False
        self.fault: str | None = None
        self._estop_released = True

    def read(self) -> ArmSample:
        if self.closed:
            raise TransportError("transport closed")
        if self.fault:
            raise TransportError(self.fault)
        return ArmSample(
            self.positions,
            self.velocities,
            (0.0,) * 6,
            (25.0,) * 6,
            5.0,
        )

    def write_positions(self, positions: tuple[float, ...]) -> None:
        if self.closed:
            raise TransportError("transport closed")
        if len(positions) != 6 or any(not math.isfinite(value) for value in positions):
            raise TransportError("invalid position command")
        previous = self.positions
        self.positions = tuple(positions)
        self.velocities = tuple((new - old) * 50.0 for old, new in zip(previous, positions))
        self.writes.append(self.positions)

    def protective_stop(self) -> None:
        self.protective_stops += 1
        self.velocities = (0.0,) * 6

    def emergency_stop(self) -> None:
        self.estops += 1
        self.velocities = (0.0,) * 6
        self._estop_released = False

    def release_estop_for_test(self) -> None:
        self._estop_released = True

    def estop_released(self) -> bool:
        return self._estop_released

    def close(self) -> None:
        self.closed = True
