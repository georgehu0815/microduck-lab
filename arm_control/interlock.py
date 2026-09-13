"""Body-motion interlock adapters.

The arm coordinator never writes body joints. It only talks to an adapter which can
request a stop from the existing robot authority. Current robotd has no exclusive
movement lease, so its Unix adapter intentionally cannot authorize hardware arm motion.
"""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Protocol


class BodyInterlock(Protocol):
    exclusive: bool

    def acquire(self, owner: str) -> bool: ...

    def maintain(self, owner: str) -> bool: ...

    def release(self, owner: str) -> None: ...


@dataclass
class SimulatedBodyInterlock:
    exclusive: bool = True
    held_by: str | None = None
    acquire_count: int = 0
    maintain_count: int = 0

    def acquire(self, owner: str) -> bool:
        self.acquire_count += 1
        if self.held_by not in (None, owner):
            return False
        self.held_by = owner
        return True

    def maintain(self, owner: str) -> bool:
        self.maintain_count += 1
        return self.held_by == owner

    def release(self, owner: str) -> None:
        if self.held_by == owner:
            self.held_by = None


class RobotdStopAdapter:
    """Request body stop through robotd, without claiming an exclusive lease.

    This adapter is useful for integration observation and documentation, but
    ``exclusive`` remains false. A hardware coordinator rejects it because another
    robotd client can submit a later movement intent.
    """

    exclusive = False

    def __init__(self, socket_path: str = "/run/robotd.sock", timeout_s: float = 0.25):
        self.socket_path = socket_path
        self.timeout_s = timeout_s

    def _stop(self) -> bool:
        request = {"jsonrpc": "2.0", "id": 1, "method": "robot.stop"}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout_s)
                client.connect(self.socket_path)
                client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                reply = client.makefile("rb").readline(65537)
        except OSError:
            return False
        if not reply or len(reply) > 65536:
            return False
        try:
            decoded = json.loads(reply)
        except json.JSONDecodeError:
            return False
        return decoded.get("error") is None and decoded.get("result", {}).get("accepted") is True

    def acquire(self, owner: str) -> bool:
        return self._stop()

    def maintain(self, owner: str) -> bool:
        return self._stop()

    def release(self, owner: str) -> None:
        return None
