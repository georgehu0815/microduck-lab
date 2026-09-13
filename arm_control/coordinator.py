"""Lease-based 50 Hz arm coordinator and safety state machine."""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .contracts import (
    COMMAND_WATCHDOG_S,
    CONTROL_PERIOD_S,
    DEVICES,
    MAX_LEASE_MS,
    ContractError,
    require_exact_keys,
    validate_action,
    validate_contract_shape,
    validate_mode,
)
from .interlock import BodyInterlock
from .transport import (
    HARDWARE_UNAVAILABLE_REASON,
    ArmSample,
    ArmTransport,
    TransportError,
)


class ControlError(RuntimeError):
    def __init__(self, code: str, message: str, data: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}


class SafetyState(str, Enum):
    LOCKED = "LOCKED"
    READY = "READY"
    EXECUTING = "EXECUTING"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"
    ESTOP = "ESTOP"


@dataclass
class Lease:
    session_id: str
    epoch: int
    operator: str
    mode: str
    contract_id: str
    devices: tuple[str, ...]
    expires_at: float
    lease_duration_s: float
    next_sequence: int = 0
    last_command_at: float | None = None
    actions: dict[str, tuple[float, ...]] = field(default_factory=dict)


@dataclass
class Runtime:
    transport: ArmTransport
    state: SafetyState
    sample: ArmSample | None = None
    target: tuple[float, ...] | None = None
    velocity: tuple[float, ...] = (0.0,) * 6
    limited_by: tuple[str, ...] = ()
    fault: str | None = None


RESET_ACK = "I inspected the arm and cleared the cause"


class Coordinator:
    def __init__(
        self,
        transports: dict[str, ArmTransport],
        body_interlock: BodyInterlock,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        if not transports:
            raise ValueError("at least one arm transport is required")
        unknown = transports.keys() - DEVICES.keys()
        if unknown:
            raise ValueError(f"unknown devices: {sorted(unknown)}")
        if any(transport.hardware for transport in transports.values()):
            raise ContractError(HARDWARE_UNAVAILABLE_REASON)
        self.clock = clock
        self.body_interlock = body_interlock
        self._lock = threading.RLock()
        self._epoch = 0
        self._last_tick: float | None = None
        self._service_fault: str | None = None
        self.leases: dict[str, Lease] = {}
        self.devices = {
            device_id: Runtime(
                transport=transport,
                state=(
                    SafetyState.LOCKED
                    if getattr(transport, "locked", False)
                    else SafetyState.READY
                ),
            )
            for device_id, transport in transports.items()
        }

    def capabilities(self) -> dict[str, Any]:
        with self._lock:
            return {
                "service": "microduck-arm-control",
                "wire": "JSON-RPC 2.0 NDJSON over Unix socket",
                "control_hz": 50,
                "hardware_available": False,
                "hardware_hard_denied": True,
                "hardware_unavailable_reason": HARDWARE_UNAVAILABLE_REASON,
                "service_operational": self._service_fault is None,
                "safety_certified": False,
                "body_servo_ids_unchanged": [*range(10, 15), *range(20, 25), *range(30, 35)],
                "body_imu_id_unchanged": 200,
                "devices": {
                    device_id: {
                        "servo_ids": list(DEVICES[device_id].servo_ids),
                        "joint_order": [joint.name for joint in DEVICES[device_id].joints],
                        "joint_map_hash": DEVICES[device_id].joint_map_hash,
                        "state": runtime.state.value,
                        "hardware": runtime.transport.hardware,
                    }
                    for device_id, runtime in self.devices.items()
                },
                "contracts": {
                    "md-arm-table-v1": {
                        "mode": "single_arm",
                        "observation_dim": 66,
                        "action_dim": 6,
                    },
                    "md-dualarm-table-v1": {
                        "mode": "dual_arm",
                        "observation_dim": 116,
                        "action_dim": 12,
                    },
                },
            }

    def acquire(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(
            params,
            {"operator", "mode", "contract_id", "observation_dim", "action_dim", "devices", "lease_ms"},
        )
        operator = params["operator"]
        devices = params["devices"]
        if not isinstance(operator, str) or not operator.strip():
            raise ControlError("INVALID_PARAMS", "operator is required")
        if not isinstance(devices, list) or not devices or any(device not in self.devices for device in devices):
            raise ControlError("INVALID_PARAMS", "devices must name available arms")
        if len(set(devices)) != len(devices):
            raise ControlError("INVALID_PARAMS", "devices must be unique")
        lease_ms = params["lease_ms"]
        if isinstance(lease_ms, bool) or not isinstance(lease_ms, int) or not 20 <= lease_ms <= MAX_LEASE_MS:
            raise ControlError("INVALID_PARAMS", f"lease_ms must be an integer in [20, {MAX_LEASE_MS}]")
        try:
            contract = validate_contract_shape(
                params["contract_id"], params["observation_dim"], params["action_dim"], devices
            )
            mode = validate_mode(params["mode"], contract)
        except ContractError as exc:
            raise ControlError("INVALID_PARAMS", str(exc)) from exc

        with self._lock:
            if self._service_fault is not None:
                raise ControlError(
                    "LOCKED",
                    f"coordinator service is failed: {self._service_fault}",
                )
            now = self.clock()
            self._expire(now)
            for device_id in devices:
                runtime = self.devices[device_id]
                if runtime.state is not SafetyState.READY:
                    raise ControlError("NOT_READY", f"{device_id} is {runtime.state.value}")
                if self._lease_for_device(device_id) is not None:
                    raise ControlError("BUSY", f"{device_id} already has an owner")
            if not self.body_interlock.exclusive:
                raise ControlError("LOCKED", "exclusive body interlock is unavailable")
            session_id = secrets.token_urlsafe(24)
            if not self.body_interlock.acquire(session_id):
                raise ControlError("BODY_INTERLOCK", "could not establish body hold")
            self._epoch += 1
            lease = Lease(
                session_id=session_id,
                epoch=self._epoch,
                operator=operator,
                mode=mode,
                contract_id=contract.contract_id,
                devices=tuple(devices),
                expires_at=now + lease_ms / 1000.0,
                lease_duration_s=lease_ms / 1000.0,
                last_command_at=now,
                actions={device_id: (0.0,) * 6 for device_id in devices},
            )
            self.leases[session_id] = lease
            return {
                "session_id": session_id,
                "session_epoch": lease.epoch,
                "expires_monotonic_ns": int(lease.expires_at * 1e9),
                "joint_map_hashes": {
                    device_id: DEVICES[device_id].joint_map_hash for device_id in devices
                },
            }

    def command(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(
            params,
            {
                "session_id",
                "session_epoch",
                "sequence",
                "deadline_monotonic_ns",
                "mode",
                "contract_id",
                "observation_dim",
                "action_dim",
                "joint_map_hashes",
                "actions",
            },
        )
        with self._lock:
            now = self.clock()
            self._expire(now)
            lease = self._get_lease(params["session_id"], params["session_epoch"])
            if params["mode"] != lease.mode:
                raise ControlError("CONTRACT_MISMATCH", "mode changed during lease")
            if params["contract_id"] != lease.contract_id:
                raise ControlError("CONTRACT_MISMATCH", "contract_id changed during lease")
            try:
                validate_contract_shape(
                    params["contract_id"],
                    params["observation_dim"],
                    params["action_dim"],
                    list(lease.devices),
                )
            except ContractError as exc:
                raise ControlError("INVALID_PARAMS", str(exc)) from exc
            sequence = params["sequence"]
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise ControlError("INVALID_PARAMS", "sequence must be a non-negative integer")
            if sequence != lease.next_sequence:
                raise ControlError("SEQUENCE", f"expected sequence {lease.next_sequence}, got {sequence}")
            deadline_ns = params["deadline_monotonic_ns"]
            if isinstance(deadline_ns, bool) or not isinstance(deadline_ns, int):
                raise ControlError("INVALID_PARAMS", "deadline_monotonic_ns must be an integer")
            if deadline_ns <= int(now * 1e9):
                self._protective_stop(lease.devices, "command deadline expired")
                raise ControlError("DEADLINE", "command deadline has expired")
            if deadline_ns > int((now + MAX_LEASE_MS / 1000.0) * 1e9):
                raise ControlError("INVALID_PARAMS", "command deadline is beyond maximum lease horizon")
            hashes = params["joint_map_hashes"]
            actions = params["actions"]
            if not isinstance(hashes, dict) or set(hashes) != set(lease.devices):
                raise ControlError("JOINT_MAP", "joint_map_hashes must exactly match leased devices")
            if not isinstance(actions, dict) or set(actions) != set(lease.devices):
                raise ControlError("INVALID_PARAMS", "actions must exactly match leased devices")
            validated: dict[str, tuple[float, ...]] = {}
            for device_id in lease.devices:
                if hashes[device_id] != DEVICES[device_id].joint_map_hash:
                    raise ControlError("JOINT_MAP", f"{device_id} joint map mismatch")
                try:
                    validated[device_id] = validate_action(actions[device_id])
                except ContractError as exc:
                    raise ControlError("INVALID_PARAMS", str(exc)) from exc
                if self.devices[device_id].state not in (SafetyState.READY, SafetyState.EXECUTING):
                    raise ControlError("NOT_READY", f"{device_id} is {self.devices[device_id].state.value}")
            try:
                maintained = self.body_interlock.maintain(lease.session_id)
            except Exception as exc:
                self._protective_stop(
                    lease.devices,
                    f"body interlock maintenance failed: {exc}",
                )
                raise ControlError(
                    "BODY_INTERLOCK",
                    "body hold maintenance failed",
                ) from exc
            if not maintained:
                self._protective_stop(lease.devices, "body interlock lost")
                raise ControlError("BODY_INTERLOCK", "body hold could not be maintained")
            lease.actions = validated
            lease.last_command_at = now
            lease.expires_at = now + lease.lease_duration_s
            lease.next_sequence += 1
            for device_id in lease.devices:
                self.devices[device_id].state = SafetyState.EXECUTING
            return {"accepted": True, "sequence": sequence}

    def release(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(params, {"session_id", "session_epoch"})
        with self._lock:
            lease = self._get_lease(params["session_id"], params["session_epoch"])
            release_error = self._release_lease(lease)
            if release_error:
                raise ControlError(
                    "BODY_INTERLOCK",
                    f"body hold release failed: {release_error}",
                )
            return {"released": True}

    def stop(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(params, {"devices", "reason"})
        devices = self._validate_devices(params["devices"])
        if not isinstance(params["reason"], str) or not params["reason"].strip():
            raise ControlError("INVALID_PARAMS", "stop reason is required")
        with self._lock:
            self._protective_stop(devices, params["reason"])
            return {"stopped": True, "state": SafetyState.PROTECTIVE_STOP.value}

    def estop(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(params, {"reason"})
        if not isinstance(params["reason"], str) or not params["reason"].strip():
            raise ControlError("INVALID_PARAMS", "estop reason is required")
        with self._lock:
            for runtime in self.devices.values():
                try:
                    runtime.transport.emergency_stop()
                except Exception as exc:
                    runtime.fault = str(exc)
                runtime.state = SafetyState.ESTOP
                runtime.velocity = (0.0,) * 6
            for lease in list(self.leases.values()):
                self._release_lease(lease)
            return {"stopped": True, "state": SafetyState.ESTOP.value}

    def reset(self, params: dict[str, Any]) -> dict[str, Any]:
        require_exact_keys(params, {"devices", "acknowledgement", "physical_estop_released"})
        devices = self._validate_devices(params["devices"])
        if params["acknowledgement"] != RESET_ACK:
            raise ControlError("MANUAL_RESET_REQUIRED", "exact manual reset acknowledgement required")
        if params["physical_estop_released"] is not True:
            raise ControlError("MANUAL_RESET_REQUIRED", "physical_estop_released must be true")
        with self._lock:
            if self._service_fault is not None:
                raise ControlError(
                    "LOCKED",
                    f"coordinator service is failed: {self._service_fault}",
                )
            samples: dict[str, ArmSample] = {}
            for device_id in devices:
                runtime = self.devices[device_id]
                if runtime.state not in (SafetyState.PROTECTIVE_STOP, SafetyState.ESTOP):
                    raise ControlError("INVALID_STATE", f"{device_id} is not stopped")
                if runtime.state is SafetyState.ESTOP and not runtime.transport.estop_released():
                    raise ControlError("ESTOP_LATCHED", f"{device_id} physical E-stop release is not verified")
                try:
                    sample = runtime.transport.read()
                    sample.validate(DEVICES[device_id])
                except Exception as exc:
                    self._protective_stop(devices, f"reset validation failed: {exc}")
                    raise ControlError("TRANSPORT", f"{device_id}: {exc}") from exc
                samples[device_id] = sample
            for device_id in devices:
                runtime = self.devices[device_id]
                sample = samples[device_id]
                runtime.sample = sample
                runtime.target = sample.positions
                runtime.velocity = (0.0,) * 6
                runtime.fault = None
                runtime.limited_by = ()
                runtime.state = SafetyState.READY
            return {"reset": True, "state": SafetyState.READY.value}

    def state(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            self._expire(now)
            return {
                "monotonic_ns": int(now * 1e9),
                "service_fault": self._service_fault,
                "devices": {
                    device_id: {
                        "state": runtime.state.value,
                        "positions": list(runtime.sample.positions) if runtime.sample else None,
                        "targets": list(runtime.target) if runtime.target else None,
                        "limited_by": list(runtime.limited_by),
                        "fault": runtime.fault,
                        "owner": (
                            lease.operator if (lease := self._lease_for_device(device_id)) else None
                        ),
                    }
                    for device_id, runtime in self.devices.items()
                },
            }

    def tick(self) -> None:
        with self._lock:
            now = self.clock()
            dt = CONTROL_PERIOD_S if self._last_tick is None else now - self._last_tick
            self._last_tick = now
            self._expire(now)
            if dt <= 0 or dt > CONTROL_PERIOD_S * 1.5:
                active = [
                    device_id
                    for device_id, runtime in self.devices.items()
                    if runtime.state is SafetyState.EXECUTING
                ]
                if active:
                    self._protective_stop(active, f"control tick outside 50 Hz budget: {dt:.6f}s")
                return
            for active_lease in list(self.leases.values()):
                try:
                    maintained = self.body_interlock.maintain(active_lease.session_id)
                except Exception as exc:
                    maintained = False
                    reason = f"body interlock maintenance failed: {exc}"
                else:
                    reason = "body interlock lost"
                if not maintained:
                    self._protective_stop(active_lease.devices, reason)
            for device_id, runtime in self.devices.items():
                if runtime.state in (SafetyState.LOCKED, SafetyState.ESTOP):
                    continue
                lease = self._lease_for_device(device_id)
                try:
                    sample = runtime.transport.read()
                    sample.validate(DEVICES[device_id])
                    runtime.sample = sample
                    if runtime.target is None:
                        runtime.target = sample.positions
                except TransportError as exc:
                    runtime.fault = str(exc)
                    affected = lease.devices if lease is not None else (device_id,)
                    self._protective_stop(affected, f"transport read failed: {exc}")
                    continue
                if runtime.state is not SafetyState.EXECUTING or lease is None:
                    continue
                if lease.last_command_at is None or now - lease.last_command_at > COMMAND_WATCHDOG_S:
                    self._protective_stop(lease.devices, "five-tick command watchdog expired")
                    continue
                action = lease.actions[device_id]
                target, velocity, limited = self._integrate(device_id, runtime, action, dt)
                try:
                    runtime.transport.write_positions(target)
                except TransportError as exc:
                    runtime.fault = str(exc)
                    self._protective_stop(lease.devices, f"transport write failed: {exc}")
                    continue
                runtime.target = target
                runtime.velocity = velocity
                runtime.limited_by = tuple(limited)

    def close(self) -> None:
        self.fail_safe("coordinator closed")
        with self._lock:
            for runtime in self.devices.values():
                try:
                    runtime.transport.close()
                except Exception as exc:
                    runtime.fault = (
                        f"{runtime.fault}; transport close failed: {exc}"
                        if runtime.fault
                        else f"transport close failed: {exc}"
                    )

    def fail_safe(self, reason: str) -> None:
        with self._lock:
            self._service_fault = reason
            devices = tuple(
                device_id
                for device_id, runtime in self.devices.items()
                if runtime.state not in (SafetyState.LOCKED, SafetyState.ESTOP)
            )
            if devices:
                self._protective_stop(devices, reason)
            for lease in list(self.leases.values()):
                self._release_lease(lease)

    def _integrate(
        self,
        device_id: str,
        runtime: Runtime,
        action: tuple[float, ...],
        dt: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...], list[str]]:
        contract = DEVICES[device_id]
        target = list(runtime.target or runtime.sample.positions)  # type: ignore[union-attr]
        velocity = list(runtime.velocity)
        limited: list[str] = []
        for index, (joint, normalized) in enumerate(zip(contract.joints, action)):
            requested_velocity = normalized * joint.max_velocity
            max_change = joint.max_acceleration * dt
            delta_velocity = max(-max_change, min(max_change, requested_velocity - velocity[index]))
            next_velocity = velocity[index] + delta_velocity
            if next_velocity != requested_velocity:
                limited.append(f"{joint.name}:acceleration")
            next_target = target[index] + next_velocity * dt
            clamped = max(joint.minimum, min(joint.maximum, next_target))
            if clamped != next_target:
                limited.append(f"{joint.name}:position")
                next_velocity = 0.0
            target[index] = clamped
            velocity[index] = next_velocity
        return tuple(target), tuple(velocity), limited

    def _expire(self, now: float) -> None:
        for lease in list(self.leases.values()):
            if now > lease.expires_at:
                self._protective_stop(lease.devices, "lease expired")
                continue
            if lease.last_command_at is not None and now - lease.last_command_at > COMMAND_WATCHDOG_S:
                self._protective_stop(lease.devices, "five-tick command watchdog expired")

    def _protective_stop(self, devices: tuple[str, ...] | list[str], reason: str) -> None:
        expanded_devices = set(devices)
        affected_sessions = {
            lease.session_id
            for device_id in tuple(expanded_devices)
            if (lease := self._lease_for_device(device_id)) is not None
        }
        for session_id in affected_sessions:
            lease = self.leases.get(session_id)
            if lease is not None:
                expanded_devices.update(lease.devices)
        for device_id in expanded_devices:
            runtime = self.devices[device_id]
            if runtime.state is SafetyState.ESTOP:
                continue
            try:
                runtime.transport.protective_stop()
            except Exception as exc:
                runtime.fault = f"{reason}; protective stop failed: {exc}"
            else:
                runtime.fault = reason
            runtime.velocity = (0.0,) * 6
            runtime.state = SafetyState.PROTECTIVE_STOP
        for session_id in affected_sessions:
            lease = self.leases.get(session_id)
            if lease:
                self._release_lease(lease)

    def _release_lease(self, lease: Lease) -> str | None:
        self.leases.pop(lease.session_id, None)
        release_error = None
        try:
            self.body_interlock.release(lease.session_id)
        except Exception as exc:
            release_error = str(exc)
        for device_id in lease.devices:
            runtime = self.devices[device_id]
            if release_error:
                if runtime.state not in (
                    SafetyState.PROTECTIVE_STOP,
                    SafetyState.ESTOP,
                ):
                    try:
                        runtime.transport.protective_stop()
                    except Exception as exc:
                        release_error = (
                            f"{release_error}; protective stop failed: {exc}"
                        )
                runtime.fault = (
                    f"{runtime.fault}; body interlock release failed: {release_error}"
                    if runtime.fault
                    else f"body interlock release failed: {release_error}"
                )
                if runtime.state is not SafetyState.ESTOP:
                    runtime.state = SafetyState.PROTECTIVE_STOP
                runtime.velocity = (0.0,) * 6
            if runtime.state is SafetyState.EXECUTING:
                runtime.state = SafetyState.READY
                runtime.velocity = (0.0,) * 6
        return release_error

    def _get_lease(self, session_id: Any, epoch: Any) -> Lease:
        if not isinstance(session_id, str):
            raise ControlError("INVALID_PARAMS", "session_id must be a string")
        lease = self.leases.get(session_id)
        if lease is None:
            raise ControlError("LEASE", "unknown or expired session")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != lease.epoch:
            raise ControlError("LEASE", "session_epoch mismatch")
        return lease

    def _lease_for_device(self, device_id: str) -> Lease | None:
        return next((lease for lease in self.leases.values() if device_id in lease.devices), None)

    def _validate_devices(self, value: Any) -> tuple[str, ...]:
        if not isinstance(value, list) or not value or any(device not in self.devices for device in value):
            raise ControlError("INVALID_PARAMS", "devices must name available arms")
        if len(set(value)) != len(value):
            raise ControlError("INVALID_PARAMS", "devices must be unique")
        return tuple(value)
