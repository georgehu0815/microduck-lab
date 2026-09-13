from __future__ import annotations

import math
import unittest
from typing import Any

from arm_control.contracts import DEVICES, LEFT_ARM, RIGHT_ARM, ContractError
from arm_control.coordinator import RESET_ACK, ControlError, Coordinator, SafetyState
from arm_control.interlock import RobotdStopAdapter, SimulatedBodyInterlock
from arm_control.transport import LockedTransport, SimulatedTransport


class Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class HardwareLikeTransport(SimulatedTransport):
    hardware = True


def acquire_params(devices: list[str] | None = None) -> dict:
    devices = devices or ["left-arm"]
    dual = len(devices) == 2
    return {
        "operator": "test",
        "mode": "dual_arm" if dual else "single_arm",
        "contract_id": "md-dualarm-table-v1" if dual else "md-arm-table-v1",
        "observation_dim": 116 if dual else 66,
        "action_dim": 12 if dual else 6,
        "devices": devices,
        "lease_ms": 500,
    }


def command_params(coordinator: Coordinator, lease: dict, sequence: int = 0, action=None) -> dict:
    devices = list(lease["joint_map_hashes"])
    dual = len(devices) == 2
    default_action = action if action is not None else [1, 0, 0, 0, 0, 0]
    return {
        "session_id": lease["session_id"],
        "session_epoch": lease["session_epoch"],
        "sequence": sequence,
        "deadline_monotonic_ns": int((coordinator.clock() + 0.1) * 1e9),
        "mode": "dual_arm" if dual else "single_arm",
        "contract_id": "md-dualarm-table-v1" if dual else "md-arm-table-v1",
        "observation_dim": 116 if dual else 66,
        "action_dim": 12 if dual else 6,
        "joint_map_hashes": {
            device: DEVICES[device].joint_map_hash for device in devices
        },
        "actions": {device: list(default_action) for device in devices},
    }


class CoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.transport = SimulatedTransport(LEFT_ARM)
        self.interlock = SimulatedBodyInterlock()
        self.coordinator = Coordinator(
            {"left-arm": self.transport},
            self.interlock,
            clock=self.clock,
        )

    def test_per_arm_ownership_is_exclusive(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        with self.assertRaisesRegex(ControlError, "already has an owner"):
            self.coordinator.acquire(acquire_params())
        self.coordinator.release(
            {"session_id": lease["session_id"], "session_epoch": lease["session_epoch"]}
        )
        self.assertIsNone(self.interlock.held_by)

    def test_body_interlock_release_failure_removes_authority_and_stops_arm(self) -> None:
        class FailingReleaseInterlock(SimulatedBodyInterlock):
            def release(self, owner: str) -> None:
                raise RuntimeError("release failed")

        transport = SimulatedTransport(LEFT_ARM)
        coordinator = Coordinator(
            {"left-arm": transport},
            FailingReleaseInterlock(),
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params())
        coordinator.command(command_params(coordinator, lease))

        with self.assertRaisesRegex(ControlError, "release failed"):
            coordinator.release(
                {
                    "session_id": lease["session_id"],
                    "session_epoch": lease["session_epoch"],
                }
            )

        self.assertEqual(coordinator.leases, {})
        self.assertEqual(transport.protective_stops, 1)
        self.assertEqual(
            coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.PROTECTIVE_STOP.value,
        )

    def test_close_stops_active_transport_and_releases_body_authority(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        self.coordinator.command(command_params(self.coordinator, lease))

        self.coordinator.close()

        self.assertEqual(self.coordinator.leases, {})
        self.assertIsNone(self.interlock.held_by)
        self.assertEqual(self.transport.protective_stops, 1)
        self.assertTrue(self.transport.closed)
        self.assertFalse(self.coordinator.capabilities()["service_operational"])

    def test_dual_arm_lease_is_atomic_and_requires_dual_mode(self) -> None:
        coordinator = Coordinator(
            {
                "left-arm": SimulatedTransport(DEVICES["left-arm"]),
                "right-arm": SimulatedTransport(DEVICES["right-arm"]),
            },
            SimulatedBodyInterlock(),
            clock=self.clock,
        )
        left = coordinator.acquire(acquire_params(["left-arm"]))
        with self.assertRaisesRegex(ControlError, "already has an owner"):
            coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        coordinator.release(
            {"session_id": left["session_id"], "session_epoch": left["session_epoch"]}
        )
        bad_mode = acquire_params(["left-arm", "right-arm"])
        bad_mode["mode"] = "single_arm"
        with self.assertRaisesRegex(ControlError, "requires mode='dual_arm'"):
            coordinator.acquire(bad_mode)
        dual = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        self.assertEqual(set(dual["joint_map_hashes"]), {"left-arm", "right-arm"})

    def test_locked_transport_cannot_be_acquired(self) -> None:
        coordinator = Coordinator(
            {"left-arm": LockedTransport()},
            SimulatedBodyInterlock(),
            clock=self.clock,
        )
        self.assertEqual(
            coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.LOCKED.value,
        )
        with self.assertRaisesRegex(ControlError, "LOCKED"):
            coordinator.acquire(acquire_params())

    def test_sequence_deadline_jointmap_and_finite_actions_are_validated(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        bad_sequence = command_params(self.coordinator, lease, sequence=1)
        with self.assertRaisesRegex(ControlError, "expected sequence 0"):
            self.coordinator.command(bad_sequence)

        bad_hash = command_params(self.coordinator, lease)
        bad_hash["joint_map_hashes"]["left-arm"] = "0" * 64
        with self.assertRaisesRegex(ControlError, "joint map mismatch"):
            self.coordinator.command(bad_hash)

        nonfinite = command_params(self.coordinator, lease, action=[math.nan, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(ControlError, "finite"):
            self.coordinator.command(nonfinite)

        expired = command_params(self.coordinator, lease)
        expired["deadline_monotonic_ns"] = int((self.clock() - 0.01) * 1e9)
        with self.assertRaisesRegex(ControlError, "expired"):
            self.coordinator.command(expired)
        self.assertEqual(
            self.coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.PROTECTIVE_STOP.value,
        )

    def test_normalized_velocity_is_acceleration_and_position_limited_at_50hz(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        self.coordinator.command(command_params(self.coordinator, lease))
        self.coordinator.tick()
        self.assertEqual(len(self.transport.writes), 1)
        self.assertAlmostEqual(self.transport.writes[0][0], 0.0004, places=8)
        state = self.coordinator.state()["devices"]["left-arm"]
        self.assertIn("base_yaw:acceleration", state["limited_by"])

        self.transport.positions = (
            LEFT_ARM.joints[0].maximum,
            *self.transport.positions[1:],
        )
        runtime = self.coordinator.devices["left-arm"]
        runtime.target = self.transport.positions
        self.clock.advance(0.02)
        self.coordinator.tick()
        self.assertEqual(self.transport.writes[-1][0], LEFT_ARM.joints[0].maximum)
        self.assertIn("base_yaw:position", self.coordinator.state()["devices"]["left-arm"]["limited_by"])

    def test_five_tick_watchdog_latches_protective_stop_and_requires_manual_reset(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        self.coordinator.command(command_params(self.coordinator, lease))
        self.clock.advance(0.101)
        state = self.coordinator.state()["devices"]["left-arm"]
        self.assertEqual(state["state"], SafetyState.PROTECTIVE_STOP.value)
        self.assertEqual(self.transport.protective_stops, 1)

        with self.assertRaisesRegex(ControlError, "acknowledgement"):
            self.coordinator.reset(
                {
                    "devices": ["left-arm"],
                    "acknowledgement": "reset",
                    "physical_estop_released": True,
                }
            )
        result = self.coordinator.reset(
            {
                "devices": ["left-arm"],
                "acknowledgement": RESET_ACK,
                "physical_estop_released": True,
            }
        )
        self.assertEqual(result["state"], SafetyState.READY.value)

    def test_accepted_commands_renew_the_bounded_lease(self) -> None:
        lease = self.coordinator.acquire({**acquire_params(), "lease_ms": 120})
        self.clock.advance(0.08)
        self.coordinator.command(command_params(self.coordinator, lease))
        self.clock.advance(0.08)
        self.assertEqual(
            self.coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.EXECUTING.value,
        )

    def test_estop_is_latched_until_transport_confirms_physical_release(self) -> None:
        self.coordinator.estop({"reason": "test"})
        with self.assertRaisesRegex(ControlError, "not verified"):
            self.coordinator.reset(
                {
                    "devices": ["left-arm"],
                    "acknowledgement": RESET_ACK,
                    "physical_estop_released": True,
                }
            )
        self.transport.release_estop_for_test()
        self.coordinator.reset(
            {
                "devices": ["left-arm"],
                "acknowledgement": RESET_ACK,
                "physical_estop_released": True,
            }
        )
        self.assertEqual(
            self.coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.READY.value,
        )

    def test_estop_continues_across_all_arms_after_unexpected_transport_error(self) -> None:
        class FailingEstopTransport(SimulatedTransport):
            def emergency_stop(self) -> None:
                raise RuntimeError("estop callback failed")

        left = FailingEstopTransport(LEFT_ARM)
        right = SimulatedTransport(RIGHT_ARM)
        coordinator = Coordinator(
            {"left-arm": left, "right-arm": right},
            SimulatedBodyInterlock(),
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        coordinator.command(command_params(coordinator, lease))
        coordinator.estop({"reason": "test"})

        states = coordinator.state()["devices"]
        self.assertEqual(states["left-arm"]["state"], SafetyState.ESTOP.value)
        self.assertEqual(states["right-arm"]["state"], SafetyState.ESTOP.value)
        self.assertEqual(right.estops, 1)
        self.assertEqual(coordinator.leases, {})

    def test_hardware_transport_and_authorization_boolean_are_unconditionally_denied(self) -> None:
        with self.assertRaisesRegex(ContractError, "unavailable in this phase"):
            Coordinator(
                {"left-arm": HardwareLikeTransport(LEFT_ARM)},
                SimulatedBodyInterlock(),
                clock=self.clock,
            )
        with self.assertRaisesRegex(TypeError, "hardware_authorized"):
            coordinator_constructor: Any = Coordinator
            coordinator_constructor(
                {"left-arm": SimulatedTransport(LEFT_ARM)},
                SimulatedBodyInterlock(),
                clock=self.clock,
                hardware_authorized=True,
            )

    def test_capabilities_state_hardware_is_unconditionally_unavailable(self) -> None:
        capabilities = self.coordinator.capabilities()
        self.assertFalse(capabilities["hardware_available"])
        self.assertTrue(capabilities["hardware_hard_denied"])
        self.assertIn(
            "independent watchdog",
            capabilities["hardware_unavailable_reason"],
        )

    def test_simulated_transport_telemetry_is_fixed_and_not_hardware_feedback(self) -> None:
        sample = self.transport.read()
        self.assertFalse(self.transport.hardware)
        self.assertEqual(sample.currents_ma, (0.0,) * 6)
        self.assertEqual(sample.temperatures_c, (25.0,) * 6)
        self.assertEqual(sample.voltage_v, 5.0)

    def test_nonexclusive_body_adapter_cannot_acquire_even_simulation(self) -> None:
        coordinator = Coordinator(
            {"left-arm": SimulatedTransport(LEFT_ARM)},
            RobotdStopAdapter("/does/not/exist"),
            clock=self.clock,
        )
        with self.assertRaisesRegex(ControlError, "exclusive body interlock"):
            coordinator.acquire(acquire_params())

    def test_acquisition_starts_command_watchdog_immediately(self) -> None:
        lease = self.coordinator.acquire(acquire_params())
        self.clock.advance(0.101)
        self.coordinator.tick()

        self.assertNotIn(lease["session_id"], self.coordinator.leases)
        self.assertIsNone(self.interlock.held_by)
        self.assertEqual(self.transport.protective_stops, 1)
        self.assertEqual(
            self.coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.PROTECTIVE_STOP.value,
        )

    def test_body_ownership_is_maintained_before_each_write(self) -> None:
        interlock = SimulatedBodyInterlock()

        class OrderedTransport(SimulatedTransport):
            def __init__(self):
                super().__init__(LEFT_ARM)
                self.minimum_maintains_before_write = 0

            def write_positions(self, positions: tuple[float, ...]) -> None:
                if interlock.maintain_count < self.minimum_maintains_before_write:
                    raise AssertionError("write occurred before body ownership maintenance")
                super().write_positions(positions)

        transport = OrderedTransport()
        coordinator = Coordinator(
            {"left-arm": transport},
            interlock,
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params())
        coordinator.command(command_params(coordinator, lease))
        transport.minimum_maintains_before_write = interlock.maintain_count + 1
        coordinator.tick()

        self.assertEqual(len(transport.writes), 1)
        self.assertGreaterEqual(
            interlock.maintain_count,
            transport.minimum_maintains_before_write,
        )

    def test_lost_body_ownership_stops_both_arms_before_any_write(self) -> None:
        left = SimulatedTransport(LEFT_ARM)
        right = SimulatedTransport(RIGHT_ARM)
        interlock = SimulatedBodyInterlock()
        coordinator = Coordinator(
            {"left-arm": left, "right-arm": right},
            interlock,
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        coordinator.command(command_params(coordinator, lease))
        interlock.held_by = None
        coordinator.tick()

        self.assertEqual(left.writes, [])
        self.assertEqual(right.writes, [])
        self.assertEqual(left.protective_stops, 1)
        self.assertEqual(right.protective_stops, 1)
        self.assertEqual(coordinator.leases, {})
        states = coordinator.state()["devices"]
        self.assertEqual(states["left-arm"]["state"], SafetyState.PROTECTIVE_STOP.value)
        self.assertEqual(states["right-arm"]["state"], SafetyState.PROTECTIVE_STOP.value)

    def test_body_interlock_exception_during_command_stops_full_lease(self) -> None:
        class FailingInterlock(SimulatedBodyInterlock):
            def maintain(self, owner: str) -> bool:
                raise RuntimeError("interlock unavailable")

        left = SimulatedTransport(LEFT_ARM)
        right = SimulatedTransport(RIGHT_ARM)
        coordinator = Coordinator(
            {"left-arm": left, "right-arm": right},
            FailingInterlock(),
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))

        with self.assertRaisesRegex(ControlError, "maintenance failed"):
            coordinator.command(command_params(coordinator, lease))

        self.assertEqual(left.protective_stops, 1)
        self.assertEqual(right.protective_stops, 1)
        self.assertEqual(coordinator.leases, {})

    def test_one_arm_transport_failure_stops_both_leased_arms(self) -> None:
        left = SimulatedTransport(LEFT_ARM)
        right = SimulatedTransport(RIGHT_ARM)
        coordinator = Coordinator(
            {"left-arm": left, "right-arm": right},
            SimulatedBodyInterlock(),
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        coordinator.command(command_params(coordinator, lease))
        left.fault = "read failed"
        coordinator.tick()

        self.assertEqual(left.protective_stops, 1)
        self.assertEqual(right.protective_stops, 1)
        self.assertEqual(right.writes, [])
        self.assertEqual(coordinator.leases, {})

    def test_multi_arm_reset_is_atomic_when_one_transport_fails(self) -> None:
        left = SimulatedTransport(LEFT_ARM)
        right = SimulatedTransport(RIGHT_ARM)
        coordinator = Coordinator(
            {"left-arm": left, "right-arm": right},
            SimulatedBodyInterlock(),
            clock=self.clock,
        )
        lease = coordinator.acquire(acquire_params(["left-arm", "right-arm"]))
        coordinator.command(command_params(coordinator, lease))
        coordinator.stop(
            {"devices": ["left-arm"], "reason": "test dual-arm stop propagation"}
        )
        right.fault = "read failed"

        with self.assertRaisesRegex(ControlError, "read failed"):
            coordinator.reset(
                {
                    "devices": ["left-arm", "right-arm"],
                    "acknowledgement": RESET_ACK,
                    "physical_estop_released": True,
                }
            )

        states = coordinator.state()["devices"]
        self.assertEqual(states["left-arm"]["state"], SafetyState.PROTECTIVE_STOP.value)
        self.assertEqual(states["right-arm"]["state"], SafetyState.PROTECTIVE_STOP.value)


if __name__ == "__main__":
    unittest.main()
