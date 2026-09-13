from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

from arm_control.contracts import (
    BODY_IMU_ID,
    BODY_SERVO_IDS,
    DEVICES,
    LEFT_ARM,
    RIGHT_ARM,
    ContractError,
    load_hardware_deployment,
)


class ContractTests(unittest.TestCase):
    def test_arm_ids_are_separate_from_body_and_imu(self) -> None:
        arm_ids = set(LEFT_ARM.servo_ids + RIGHT_ARM.servo_ids)
        self.assertTrue(arm_ids.isdisjoint(BODY_SERVO_IDS))
        self.assertNotIn(BODY_IMU_ID, arm_ids)
        self.assertEqual(LEFT_ARM.servo_ids, tuple(range(40, 46)))
        self.assertEqual(RIGHT_ARM.servo_ids, tuple(range(50, 56)))

    def test_joint_map_hash_is_stable_and_side_specific(self) -> None:
        self.assertEqual(len(LEFT_ARM.joint_map_hash), 64)
        self.assertNotEqual(LEFT_ARM.joint_map_hash, RIGHT_ARM.joint_map_hash)
        self.assertEqual(LEFT_ARM.joint_map_hash, DEVICES["left-arm"].joint_map_hash)

    def test_offline_hardware_files_validate_calibration_ids_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "deployment.json"
            auth_path = root / "authorization.json"
            calibration = [
                {
                    "name": joint.name,
                    "zero_count": 2048,
                    "direction": 1,
                    "counts_per_unit": 500.0 if joint.unit == "rad" else 50000,
                }
                for joint in LEFT_ARM.joints
            ]
            manifest = {
                "schema": "microduck-arm-deployment-v1",
                "deployment_id": "bench-a",
                "calibrated": True,
                "calibration_sha256": "",
                "bus_paths": {"left-arm": "/dev/microduck-arm-left"},
                "devices": {
                    "left-arm": {
                        "servo_ids": list(LEFT_ARM.servo_ids),
                        "joint_map_hash": LEFT_ARM.joint_map_hash,
                        "calibration": calibration,
                    }
                },
            }
            manifest["calibration_sha256"] = hashlib.sha256(
                json.dumps(
                    manifest["devices"],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            authorization = {
                "schema": "microduck-arm-bench-authorization-v1",
                "deployment_id": "bench-a",
                "authorized_bench": True,
                "expires_unix_s": 2000,
                "purpose": "supervised no-load range test",
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            auth_path.write_text(json.dumps(authorization), encoding="utf-8")
            os.chmod(manifest_path, 0o600)
            os.chmod(auth_path, 0o600)

            loaded = load_hardware_deployment(
                manifest_path,
                auth_path,
                now_unix_s=1000,
                require_root_owner=False,
            )
            self.assertEqual(loaded["manifest"]["deployment_id"], "bench-a")

            authorization["expires_unix_s"] = 999
            auth_path.write_text(json.dumps(authorization), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "expired"):
                load_hardware_deployment(
                    manifest_path,
                    auth_path,
                    now_unix_s=1000,
                    require_root_owner=False,
                )

    def test_capability_files_cannot_be_group_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            auth = Path(directory) / "authorization.json"
            path.write_text("{}", encoding="utf-8")
            auth.write_text("{}", encoding="utf-8")
            os.chmod(path, 0o620)
            with self.assertRaisesRegex(ContractError, "group/world writable"):
                load_hardware_deployment(path, auth, require_root_owner=False)


if __name__ == "__main__":
    unittest.main()
