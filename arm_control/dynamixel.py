"""Dynamixel Protocol 2 packet codec and offline XL330 transport preparation.

No serial device can be opened in this phase. Packet encoding, decoding, calibration
mapping, and fake-byte-stream tests remain available for offline preparation.
"""

from __future__ import annotations

import math
import os
import select
import struct
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import DeviceContract
from .transport import HARDWARE_UNAVAILABLE_REASON, ArmSample, TransportError

HEADER = b"\xff\xff\xfd\x00"
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_STATUS = 0x55
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83
BROADCAST_ID = 0xFE

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
PRESENT_BLOCK_LEN = 21  # current(2), velocity(4), position(4), ..., voltage(2), temperature(1)


class ByteStream(Protocol):
    def write_all(self, data: bytes) -> None: ...

    def read_exact(self, size: int, timeout_s: float) -> bytes: ...

    def close(self) -> None: ...


def crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def _stuff(body: bytes) -> bytes:
    output = bytearray()
    for byte in body:
        output.append(byte)
        if len(output) >= 3 and output[-3:] == b"\xff\xff\xfd":
            output.append(0xFD)
    return bytes(output)


def _unstuff(body: bytes) -> bytes:
    return body.replace(b"\xff\xff\xfd\xfd", b"\xff\xff\xfd")


def encode_instruction(device_id: int, instruction: int, parameters: bytes = b"") -> bytes:
    if not 0 <= device_id <= 0xFE:
        raise ValueError("invalid Dynamixel ID")
    stuffed = _stuff(bytes([instruction]) + parameters)
    length = len(stuffed) + 2
    packet = HEADER + bytes([device_id]) + struct.pack("<H", length) + stuffed
    return packet + struct.pack("<H", crc16(packet))


def encode_read(device_id: int, address: int, length: int) -> bytes:
    return encode_instruction(device_id, INST_READ, struct.pack("<HH", address, length))


def encode_write(device_id: int, address: int, value: bytes) -> bytes:
    return encode_instruction(device_id, INST_WRITE, struct.pack("<H", address) + value)


def encode_sync_read(device_ids: tuple[int, ...], address: int, length: int) -> bytes:
    return encode_instruction(
        BROADCAST_ID,
        INST_SYNC_READ,
        struct.pack("<HH", address, length) + bytes(device_ids),
    )


def encode_sync_write(device_ids: tuple[int, ...], address: int, values: tuple[bytes, ...]) -> bytes:
    if len(device_ids) != len(values) or not values:
        raise ValueError("sync write IDs and values must be non-empty and aligned")
    width = len(values[0])
    if any(len(value) != width for value in values):
        raise ValueError("sync write values must have equal width")
    params = bytearray(struct.pack("<HH", address, width))
    for device_id, value in zip(device_ids, values):
        params.append(device_id)
        params.extend(value)
    return encode_instruction(BROADCAST_ID, INST_SYNC_WRITE, bytes(params))


@dataclass(frozen=True)
class StatusPacket:
    device_id: int
    error: int
    parameters: bytes


def decode_status(packet: bytes) -> StatusPacket:
    if len(packet) < 11 or packet[:4] != HEADER:
        raise TransportError("invalid Dynamixel status header")
    declared = struct.unpack_from("<H", packet, 5)[0]
    if len(packet) != declared + 7:
        raise TransportError("Dynamixel status length mismatch")
    expected_crc = struct.unpack_from("<H", packet, len(packet) - 2)[0]
    if crc16(packet[:-2]) != expected_crc:
        raise TransportError("Dynamixel status CRC mismatch")
    body = _unstuff(packet[7:-2])
    if len(body) < 2 or body[0] != INST_STATUS:
        raise TransportError("not a Dynamixel status packet")
    return StatusPacket(packet[4], body[1], body[2:])


class PosixSerialStream:
    fd: int

    def __init__(self, fd: int):
        raise TransportError(HARDWARE_UNAVAILABLE_REASON)

    @classmethod
    def open(cls, path: str) -> "PosixSerialStream":
        raise TransportError(HARDWARE_UNAVAILABLE_REASON)

    def write_all(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise TransportError("serial write made no progress")
            view = view[written:]

    def read_exact(self, size: int, timeout_s: float) -> bytes:
        deadline = time.monotonic() + timeout_s
        output = bytearray()
        while len(output) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransportError("serial read timeout")
            readable, _, _ = select.select([self.fd], [], [], remaining)
            if not readable:
                raise TransportError("serial read timeout")
            chunk = os.read(self.fd, size - len(output))
            if not chunk:
                raise TransportError("serial device closed")
            output.extend(chunk)
        return bytes(output)

    def close(self) -> None:
        os.close(self.fd)


class Protocol2Bus:
    def __init__(self, stream: ByteStream, timeout_s: float = 0.03):
        self.stream = stream
        self.timeout_s = timeout_s

    def send(self, packet: bytes) -> None:
        self.stream.write_all(packet)

    def receive_status(self) -> StatusPacket:
        header = self.stream.read_exact(7, self.timeout_s)
        if header[:4] != HEADER:
            raise TransportError("unexpected serial bytes before Dynamixel status")
        length = struct.unpack_from("<H", header, 5)[0]
        return decode_status(header + self.stream.read_exact(length, self.timeout_s))

    def request(self, packet: bytes, expected_id: int) -> StatusPacket:
        self.send(packet)
        status = self.receive_status()
        if status.device_id != expected_id:
            raise TransportError(f"expected status from {expected_id}, got {status.device_id}")
        if status.error:
            raise TransportError(f"Dynamixel {expected_id} returned error 0x{status.error:02x}")
        return status

    def close(self) -> None:
        self.stream.close()


class Xl330Transport:
    hardware = True
    locked = False

    def __init__(self, contract: DeviceContract, bus: Protocol2Bus, calibration: list[dict[str, Any]]):
        self.contract = contract
        self.bus = bus
        self.calibration = self._validate_calibration(contract, calibration)

    @staticmethod
    def _validate_calibration(
        contract: DeviceContract,
        entries: list[dict[str, Any]],
    ) -> tuple[dict[str, float], ...]:
        if len(entries) != 6:
            raise TransportError("calibration must contain six entries")
        output = []
        for joint, entry in zip(contract.joints, entries):
            if not isinstance(entry, dict) or entry.get("name") != joint.name:
                raise TransportError(f"calibration order mismatch at {joint.name}")
            required = {"name", "zero_count", "direction", "counts_per_unit"}
            if set(entry) != required:
                raise TransportError(f"calibration fields mismatch at {joint.name}")
            values = {key: float(entry[key]) for key in required - {"name"}}
            if any(not math.isfinite(value) for value in values.values()):
                raise TransportError(f"non-finite calibration at {joint.name}")
            if values["direction"] not in (-1.0, 1.0) or values["counts_per_unit"] <= 0:
                raise TransportError(f"invalid calibration at {joint.name}")
            output.append(values)
        return tuple(output)

    def _to_count(self, index: int, value: float) -> int:
        calibration = self.calibration[index]
        count = calibration["zero_count"] + (
            calibration["direction"] * value * calibration["counts_per_unit"]
        )
        if not 0 <= count <= 4095:
            raise TransportError(f"{self.contract.joints[index].name} maps outside XL330 position range")
        return round(count)

    def _from_count(self, index: int, count: int) -> float:
        calibration = self.calibration[index]
        return (
            (count - calibration["zero_count"])
            / calibration["counts_per_unit"]
            / calibration["direction"]
        )

    def read(self) -> ArmSample:
        self.bus.send(encode_sync_read(self.contract.servo_ids, ADDR_PRESENT_CURRENT, PRESENT_BLOCK_LEN))
        blocks = {}
        for _ in self.contract.servo_ids:
            status = self.bus.receive_status()
            if (
                status.device_id not in self.contract.servo_ids
                or status.device_id in blocks
                or status.error
                or len(status.parameters) != PRESENT_BLOCK_LEN
            ):
                raise TransportError(f"invalid status block for Dynamixel {status.device_id}")
            blocks[status.device_id] = status.parameters

        positions = []
        velocities = []
        currents = []
        temperatures = []
        voltages = []
        for index, expected_id in enumerate(self.contract.servo_ids):
            block = blocks[expected_id]
            currents.append(abs(struct.unpack_from("<h", block, 0)[0]) * 1.0)
            velocity_count = struct.unpack_from("<i", block, 2)[0]
            motor_rad_s = velocity_count * 0.229 * 2.0 * math.pi / 60.0
            calibration = self.calibration[index]
            velocities.append(
                motor_rad_s
                * (4096.0 / (2.0 * math.pi))
                / calibration["counts_per_unit"]
                / calibration["direction"]
            )
            position_count = struct.unpack_from("<i", block, 6)[0]
            positions.append(self._from_count(index, position_count))
            voltages.append(struct.unpack_from("<H", block, 18)[0] * 0.1)
            temperatures.append(float(block[20]))
        sample = ArmSample(
            tuple(positions),
            tuple(velocities),
            tuple(currents),
            tuple(temperatures),
            sum(voltages) / len(voltages),
        )
        sample.validate(self.contract)
        return sample

    def write_positions(self, positions: tuple[float, ...]) -> None:
        if len(positions) != 6 or any(not math.isfinite(value) for value in positions):
            raise TransportError("invalid arm target")
        values = tuple(struct.pack("<i", self._to_count(index, value)) for index, value in enumerate(positions))
        self.bus.send(encode_sync_write(self.contract.servo_ids, ADDR_GOAL_POSITION, values))

    def protective_stop(self) -> None:
        return None

    def emergency_stop(self) -> None:
        failures = []
        for device_id in self.contract.servo_ids:
            try:
                self.bus.request(encode_write(device_id, ADDR_TORQUE_ENABLE, b"\x00"), device_id)
            except TransportError as exc:
                failures.append(str(exc))
        if failures:
            raise TransportError("; ".join(failures))

    def estop_released(self) -> bool:
        return False

    def close(self) -> None:
        self.bus.close()


def open_xl330_transport(
    contract: DeviceContract,
    bus_path: str,
    calibration: list[dict[str, Any]],
) -> Xl330Transport:
    raise TransportError(HARDWARE_UNAVAILABLE_REASON)
