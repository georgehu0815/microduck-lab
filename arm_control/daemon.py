"""Command-line entry point for the arm-control Unix service."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Callable

from .contracts import DEVICES, ContractError
from .coordinator import Coordinator
from .interlock import SimulatedBodyInterlock
from .rpc import UnixRpcServer
from .transport import HARDWARE_UNAVAILABLE_REASON, LockedTransport, SimulatedTransport


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bounded Microduck arm control daemon")
    parser.add_argument("--socket", default="/run/microduck-arm/control.sock")
    parser.add_argument("--transport", choices=("locked", "simulation", "dynamixel"), default="locked")
    parser.add_argument("--devices", default="left-arm,right-arm")
    return parser


def create_coordinator(args: argparse.Namespace) -> Coordinator:
    device_ids = tuple(part.strip() for part in args.devices.split(",") if part.strip())
    if not device_ids or any(device_id not in DEVICES for device_id in device_ids):
        raise ContractError("--devices must contain left-arm and/or right-arm")
    if args.transport == "locked":
        return Coordinator(
            {device_id: LockedTransport() for device_id in device_ids},
            SimulatedBodyInterlock(),
        )
    if args.transport == "simulation":
        return Coordinator(
            {device_id: SimulatedTransport(DEVICES[device_id]) for device_id in device_ids},
            SimulatedBodyInterlock(),
        )
    raise ContractError(HARDWARE_UNAVAILABLE_REASON)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        coordinator = create_coordinator(args)
    except (ContractError, OSError, RuntimeError) as exc:
        print(f"arm-control: refused to start: {exc}", file=sys.stderr)
        return 2

    server = UnixRpcServer(coordinator, args.socket)
    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()
        server.shutdown()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ticker = threading.Thread(
        target=_tick_loop,
        args=(coordinator, stopped, server.shutdown),
        daemon=True,
    )
    ticker.start()
    try:
        server.serve_forever()
    finally:
        stopped.set()
        ticker.join(timeout=1.0)
        coordinator.close()
    return 0


def _tick_loop(
    coordinator: Coordinator,
    stopped: threading.Event,
    shutdown: Callable[[], None] | None = None,
) -> None:
    period = 0.02
    deadline = time.monotonic()
    while not stopped.is_set():
        deadline += period
        try:
            coordinator.tick()
        except BaseException as exc:
            try:
                coordinator.fail_safe(
                    f"control ticker failed: {type(exc).__name__}: {exc}"
                )
            finally:
                stopped.set()
                if shutdown is not None:
                    shutdown()
            return
        stopped.wait(max(0.0, deadline - time.monotonic()))


if __name__ == "__main__":
    raise SystemExit(main())
