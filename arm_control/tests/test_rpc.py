from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
import unittest

from arm_control.contracts import LEFT_ARM
from arm_control.coordinator import Coordinator
from arm_control.interlock import SimulatedBodyInterlock
from arm_control.rpc import UnixRpcServer
from arm_control.transport import SimulatedTransport


class RpcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.temp.name, "arm.sock")
        self.coordinator = Coordinator(
            {"left-arm": SimulatedTransport(LEFT_ARM)},
            SimulatedBodyInterlock(),
        )
        self.server = UnixRpcServer(self.coordinator, self.socket_path, mode=0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 1.0
        while not os.path.exists(self.socket_path) and time.monotonic() < deadline:
            time.sleep(0.005)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=1.0)
        self.temp.cleanup()

    def call(self, request: dict) -> dict:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(self.socket_path)
            client.sendall(json.dumps(request).encode() + b"\n")
            return json.loads(client.makefile("rb").readline())

    def test_real_unix_socket_serves_capabilities_with_restrictive_mode(self) -> None:
        response = self.call(
            {"jsonrpc": "2.0", "id": 1, "method": "manipulation.capabilities"}
        )
        self.assertEqual(response["result"]["contracts"]["md-arm-table-v1"]["observation_dim"], 66)
        self.assertEqual(os.stat(self.socket_path).st_mode & 0o777, 0o600)

    def test_unknown_envelope_and_params_are_rejected(self) -> None:
        response = self.call(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "manipulation.capabilities",
                "fabricated": True,
            }
        )
        self.assertEqual(response["error"]["code"], -32600)

        response = self.call(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "manipulation.acquire",
                "params": {
                    "operator": "test",
                    "mode": "single_arm",
                    "contract_id": "md-arm-table-v1",
                    "observation_dim": 66,
                    "action_dim": 6,
                    "devices": ["left-arm"],
                    "lease_ms": 500,
                    "ui_claimed_capability": True,
                },
            }
        )
        self.assertEqual(response["error"]["code"], -32602)
        self.assertIn("unknown fields", response["error"]["message"])


if __name__ == "__main__":
    unittest.main()
