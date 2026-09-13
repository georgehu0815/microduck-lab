"""Strict JSON-RPC 2.0 NDJSON Unix-socket server."""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path
from typing import Any

from .contracts import ContractError
from .coordinator import ControlError, Coordinator

MAX_LINE = 64 * 1024
METHODS = {
    "manipulation.capabilities": "capabilities",
    "manipulation.acquire": "acquire",
    "manipulation.command": "command",
    "manipulation.release": "release",
    "manipulation.stop": "stop",
    "manipulation.estop": "estop",
    "manipulation.reset": "reset",
    "manipulation.state": "state",
}
ERROR_CODES = {
    "INVALID_PARAMS": -32602,
    "METHOD_NOT_FOUND": -32601,
    "BUSY": 1,
    "LOCKED": 2,
    "NOT_READY": 3,
    "LEASE": 4,
    "SEQUENCE": 5,
    "DEADLINE": 6,
    "JOINT_MAP": 7,
    "CONTRACT_MISMATCH": 8,
    "BODY_INTERLOCK": 9,
    "MANUAL_RESET_REQUIRED": 10,
    "ESTOP_LATCHED": 11,
    "TRANSPORT": 12,
    "INVALID_STATE": 13,
}


class UnixRpcServer:
    def __init__(self, coordinator: Coordinator, socket_path: str, mode: int = 0o660):
        self.coordinator = coordinator
        self.socket_path = Path(socket_path)
        self.mode = mode
        self._listener: socket.socket | None = None
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
        except BaseException:
            listener.close()
            raise
        os.chmod(self.socket_path, self.mode)
        listener.listen(16)
        listener.settimeout(0.2)
        self._listener = listener
        try:
            while not self._stop.is_set():
                try:
                    client, _ = listener.accept()
                except TimeoutError:
                    continue
                threading.Thread(target=self._serve_client, args=(client,), daemon=True).start()
        finally:
            listener.close()
            self._listener = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def shutdown(self) -> None:
        self._stop.set()

    def _serve_client(self, client: socket.socket) -> None:
        with client:
            reader = client.makefile("rb")
            while not self._stop.is_set():
                line = reader.readline(MAX_LINE + 1)
                if not line:
                    return
                if self._stop.is_set():
                    return
                if len(line) > MAX_LINE or not line.endswith(b"\n"):
                    self._send(client, self._error(None, -32600, "request line is too large"))
                    return
                response = self.dispatch_bytes(line)
                if response is not None:
                    self._send(client, response)

    def dispatch_bytes(self, line: bytes) -> dict[str, Any] | None:
        try:
            request = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error(None, -32700, "parse error")
        if not isinstance(request, dict):
            return self._error(None, -32600, "request must be an object")
        request_id = request.get("id")
        if self._stop.is_set():
            return self._error(request_id, -32603, "service is shutting down")
        if set(request) - {"jsonrpc", "id", "method", "params"}:
            return self._error(request_id, -32600, "unknown request envelope fields")
        if request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
            return self._error(request_id, -32600, "invalid JSON-RPC request")
        method = request["method"]
        target = METHODS.get(method)
        if target is None:
            return self._error(request_id, -32601, f"unknown method {method}")
        params = request.get("params", {})
        if not isinstance(params, dict):
            return self._error(request_id, -32602, "params must be an object")
        try:
            if target in ("capabilities", "state"):
                if params:
                    raise ControlError("INVALID_PARAMS", f"{method} takes no parameters")
                result = getattr(self.coordinator, target)()
            else:
                result = getattr(self.coordinator, target)(params)
        except (ControlError, ContractError) as exc:
            code_name = exc.code if isinstance(exc, ControlError) else "INVALID_PARAMS"
            return self._error(
                request_id,
                ERROR_CODES.get(code_name, -32603),
                str(exc),
                {"kind": code_name, **getattr(exc, "data", {})},
            )
        except Exception as exc:
            return self._error(request_id, -32603, f"internal error: {exc}")
        if "id" not in request:
            return None
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _send(client: socket.socket, response: dict[str, Any]) -> None:
        client.sendall(json.dumps(response, separators=(",", ":"), allow_nan=False).encode() + b"\n")

    @staticmethod
    def _error(
        request_id: Any,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"code": code, "message": message}
        if data:
            error["data"] = data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}
