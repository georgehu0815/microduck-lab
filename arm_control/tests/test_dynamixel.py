from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

from arm_control.dynamixel import (
    HEADER,
    INST_STATUS,
    PosixSerialStream,
    crc16,
    decode_status,
    encode_instruction,
    encode_read,
    encode_sync_write,
    encode_write,
    open_xl330_transport,
)
from arm_control.contracts import LEFT_ARM
from arm_control.transport import TransportError


class DynamixelPacketTests(unittest.TestCase):
    def test_protocol2_ping_matches_documented_packet(self) -> None:
        self.assertEqual(
            encode_instruction(1, 0x01),
            bytes.fromhex("FF FF FD 00 01 03 00 01 19 4E"),
        )

    def test_read_and_write_encode_little_endian_addresses(self) -> None:
        read = encode_read(40, 132, 4)
        write = encode_write(40, 64, b"\x00")
        self.assertIn(bytes.fromhex("02 84 00 04 00"), read)
        self.assertIn(bytes.fromhex("03 40 00 00"), write)
        self.assertEqual(struct.unpack("<H", read[-2:])[0], crc16(read[:-2]))
        self.assertEqual(struct.unpack("<H", write[-2:])[0], crc16(write[:-2]))

    def test_sync_write_contains_each_reserved_id_and_value(self) -> None:
        packet = encode_sync_write((40, 41), 116, (struct.pack("<i", 2048), struct.pack("<i", 1024)))
        self.assertIn(bytes([40]) + struct.pack("<i", 2048), packet)
        self.assertIn(bytes([41]) + struct.pack("<i", 1024), packet)

    def test_status_crc_and_error_are_checked(self) -> None:
        body = bytes([INST_STATUS, 0]) + b"\x01\x02"
        prefix = HEADER + bytes([40]) + struct.pack("<H", len(body) + 2) + body
        packet = prefix + struct.pack("<H", crc16(prefix))
        status = decode_status(packet)
        self.assertEqual(status.device_id, 40)
        self.assertEqual(status.parameters, b"\x01\x02")

        corrupt = packet[:-1] + bytes([packet[-1] ^ 0xFF])
        with self.assertRaisesRegex(TransportError, "CRC"):
            decode_status(corrupt)

    def test_public_serial_open_paths_are_hard_denied_before_os_access(self) -> None:
        with self.assertRaisesRegex(TransportError, "unavailable in this phase"):
            PosixSerialStream(123)

        with patch("arm_control.dynamixel.os.open") as os_open:
            with self.assertRaisesRegex(TransportError, "unavailable in this phase"):
                PosixSerialStream.open("/dev/microduck-arm-left")
            os_open.assert_not_called()

        with patch.object(PosixSerialStream, "open") as serial_open:
            with self.assertRaisesRegex(TransportError, "unavailable in this phase"):
                open_xl330_transport(LEFT_ARM, "/dev/microduck-arm-left", [])
            serial_open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
