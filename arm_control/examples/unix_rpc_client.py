#!/usr/bin/env python3
"""Minimal Unix JSON-RPC client for the simulation daemon."""

from __future__ import annotations

import json
import socket
import sys
import time


def call(client: socket.socket, request_id: int, method: str, params: dict | None = None) -> dict:
    request = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
    return json.loads(client.makefile("rb").readline())


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/microduck-arm.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(path)
        capabilities = call(client, 1, "manipulation.capabilities")["result"]
        device = "left-arm"
        acquired = call(
            client,
            2,
            "manipulation.acquire",
            {
                "operator": "local-example",
                "mode": "single_arm",
                "contract_id": "md-arm-table-v1",
                "observation_dim": 66,
                "action_dim": 6,
                "devices": [device],
                "lease_ms": 500,
            },
        )["result"]
        print(json.dumps(capabilities, indent=2))
        print(
            json.dumps(
                call(
                    client,
                    3,
                    "manipulation.command",
                    {
                        "session_id": acquired["session_id"],
                        "session_epoch": acquired["session_epoch"],
                        "sequence": 0,
                        "deadline_monotonic_ns": time.monotonic_ns() + 100_000_000,
                        "mode": "single_arm",
                        "contract_id": "md-arm-table-v1",
                        "observation_dim": 66,
                        "action_dim": 6,
                        "joint_map_hashes": {
                            device: capabilities["devices"][device]["joint_map_hash"]
                        },
                        "actions": {device: [0.1, 0, 0, 0, 0, 0]},
                    },
                ),
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
