from __future__ import annotations

import argparse
import threading
import unittest
from unittest.mock import Mock, patch

from arm_control.contracts import ContractError
from arm_control.daemon import _tick_loop, build_parser, create_coordinator
from arm_control.coordinator import ControlError, Coordinator, SafetyState
from arm_control.interlock import SimulatedBodyInterlock
from arm_control.transport import SimulatedTransport
from arm_control.contracts import LEFT_ARM


class DaemonSafetyTests(unittest.TestCase):
    def test_current_hardware_path_refuses_before_opening_serial(self) -> None:
        args = argparse.Namespace(
            transport="dynamixel",
            devices="left-arm",
        )
        with (
            patch("arm_control.contracts._load_secure_json") as loader,
            patch("arm_control.dynamixel.PosixSerialStream.open") as opener,
        ):
            with self.assertRaisesRegex(ContractError, "unavailable in this phase"):
                create_coordinator(args)
            loader.assert_not_called()
            opener.assert_not_called()

    def test_daemon_exposes_no_manifest_or_authorization_enable_options(self) -> None:
        option_strings = {
            option
            for action in build_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--deployment-manifest", option_strings)
        self.assertNotIn("--bench-authorization", option_strings)
        self.assertNotIn("--hardware-authorized", option_strings)

    def test_unexpected_ticker_failure_stops_coordinator_and_rpc_service(self) -> None:
        transport = SimulatedTransport(LEFT_ARM)
        interlock = SimulatedBodyInterlock()
        coordinator = Coordinator({"left-arm": transport}, interlock)
        lease = coordinator.acquire(
            {
                "operator": "test",
                "mode": "single_arm",
                "contract_id": "md-arm-table-v1",
                "observation_dim": 66,
                "action_dim": 6,
                "devices": ["left-arm"],
                "lease_ms": 500,
            }
        )
        stopped = threading.Event()
        shutdown = Mock()

        with patch.object(coordinator, "tick", side_effect=RuntimeError("boom")):
            _tick_loop(coordinator, stopped, shutdown)

        self.assertTrue(stopped.is_set())
        shutdown.assert_called_once_with()
        self.assertNotIn(lease["session_id"], coordinator.leases)
        self.assertIsNone(interlock.held_by)
        self.assertEqual(transport.protective_stops, 1)
        self.assertEqual(
            coordinator.state()["devices"]["left-arm"]["state"],
            SafetyState.PROTECTIVE_STOP.value,
        )
        self.assertIn("RuntimeError: boom", coordinator.state()["service_fault"])
        self.assertFalse(coordinator.capabilities()["service_operational"])
        with self.assertRaisesRegex(ControlError, "service is failed"):
            coordinator.acquire(
                {
                    "operator": "test",
                    "mode": "single_arm",
                    "contract_id": "md-arm-table-v1",
                    "observation_dim": 66,
                    "action_dim": 6,
                    "devices": ["left-arm"],
                    "lease_ms": 500,
                }
            )


if __name__ == "__main__":
    unittest.main()
