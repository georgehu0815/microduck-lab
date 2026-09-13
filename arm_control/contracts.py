"""Immutable arm contracts and deployment-manifest validation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONTROL_HZ = 50
CONTROL_PERIOD_S = 1.0 / CONTROL_HZ
COMMAND_WATCHDOG_S = 5 * CONTROL_PERIOD_S
MAX_LEASE_MS = 500
MAX_JOINT_VELOCITY_RAD_S = 0.4
MAX_GRIPPER_VELOCITY_M_S = 0.01
MAX_JOINT_ACCELERATION_RAD_S2 = 1.0
MAX_GRIPPER_ACCELERATION_M_S2 = 0.025

BODY_SERVO_IDS = tuple(range(10, 15)) + tuple(range(20, 25)) + tuple(range(30, 35))
BODY_IMU_ID = 200
LEFT_SERVO_IDS = tuple(range(40, 46))
RIGHT_SERVO_IDS = tuple(range(50, 56))


class ContractError(ValueError):
    pass


@dataclass(frozen=True)
class JointSpec:
    name: str
    unit: str
    minimum: float
    maximum: float
    max_velocity: float
    max_acceleration: float


ARM_JOINTS = (
    JointSpec("base_yaw", "rad", math.radians(-100), math.radians(100), 0.4, 1.0),
    JointSpec("shoulder_pitch", "rad", math.radians(-60), math.radians(80), 0.4, 1.0),
    JointSpec("elbow_pitch", "rad", math.radians(-100), math.radians(100), 0.4, 1.0),
    JointSpec("wrist_pitch", "rad", math.radians(-90), math.radians(90), 0.4, 1.0),
    JointSpec("wrist_roll", "rad", math.radians(-90), math.radians(90), 0.4, 1.0),
    JointSpec("gripper_width", "m", 0.0, 0.03, 0.01, 0.025),
)


@dataclass(frozen=True)
class DeviceContract:
    device_id: str
    servo_ids: tuple[int, ...]
    joints: tuple[JointSpec, ...] = ARM_JOINTS

    @property
    def joint_map_hash(self) -> str:
        payload = {
            "device_id": self.device_id,
            "servo_ids": self.servo_ids,
            "joints": [
                {
                    "name": joint.name,
                    "unit": joint.unit,
                    "minimum": joint.minimum,
                    "maximum": joint.maximum,
                    "max_velocity": joint.max_velocity,
                    "max_acceleration": joint.max_acceleration,
                }
                for joint in self.joints
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


LEFT_ARM = DeviceContract("left-arm", LEFT_SERVO_IDS)
RIGHT_ARM = DeviceContract("right-arm", RIGHT_SERVO_IDS)
DEVICES = {device.device_id: device for device in (LEFT_ARM, RIGHT_ARM)}


@dataclass(frozen=True)
class PolicyContract:
    contract_id: str
    observation_dim: int
    action_dim: int
    device_count: int


SINGLE_CONTRACT = PolicyContract("md-arm-table-v1", 66, 6, 1)
DUAL_CONTRACT = PolicyContract("md-dualarm-table-v1", 116, 12, 2)
POLICY_CONTRACTS = {
    SINGLE_CONTRACT.contract_id: SINGLE_CONTRACT,
    DUAL_CONTRACT.contract_id: DUAL_CONTRACT,
}


def require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str] = set()) -> None:
    missing = required - value.keys()
    unknown = value.keys() - required - optional
    if missing:
        raise ContractError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ContractError(f"unknown fields: {', '.join(sorted(unknown))}")


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def validate_action(values: Any) -> tuple[float, ...]:
    if not isinstance(values, list) or len(values) != 6:
        raise ContractError("action must contain exactly 6 values")
    action = tuple(finite_number(value, f"action[{index}]") for index, value in enumerate(values))
    if any(abs(value) > 1.0 for value in action):
        raise ContractError("normalized actions must be within [-1, 1]")
    return action


def validate_contract_shape(contract_id: str, observation_dim: Any, action_dim: Any, devices: list[str]) -> PolicyContract:
    contract = POLICY_CONTRACTS.get(contract_id)
    if contract is None:
        raise ContractError(f"unsupported contract_id {contract_id!r}")
    if observation_dim != contract.observation_dim or action_dim != contract.action_dim:
        raise ContractError(
            f"{contract_id} requires observation_dim={contract.observation_dim} "
            f"and action_dim={contract.action_dim}"
        )
    if len(devices) != contract.device_count:
        raise ContractError(f"{contract_id} requires {contract.device_count} device(s)")
    if contract is DUAL_CONTRACT and set(devices) != set(DEVICES):
        raise ContractError("dual-arm contract requires left-arm and right-arm")
    return contract


def validate_mode(mode: Any, contract: PolicyContract) -> str:
    expected = "single_arm" if contract.device_count == 1 else "dual_arm"
    if mode != expected:
        raise ContractError(f"{contract.contract_id} requires mode={expected!r}")
    return expected


def _load_secure_json(path: str | os.PathLike[str], *, require_root_owner: bool) -> dict[str, Any]:
    file_path = Path(path)
    info = file_path.lstat()
    if not stat.S_ISREG(info.st_mode) or file_path.is_symlink():
        raise ContractError(f"{file_path} must be a regular, non-symlink file")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ContractError(f"{file_path} must not be group/world writable")
    required_uid = 0 if require_root_owner else os.geteuid()
    if info.st_uid != required_uid:
        owner = "root" if require_root_owner else f"uid {required_uid}"
        raise ContractError(f"{file_path} must be owned by {owner}")

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON number {token}")

    try:
        value = json.loads(
            file_path.read_text(encoding="utf-8"),
            parse_constant=reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read {file_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{file_path} must contain a JSON object")
    return value


def load_hardware_deployment(
    manifest_path: str | os.PathLike[str],
    authorization_path: str | os.PathLike[str],
    *,
    now_unix_s: float | None = None,
    require_root_owner: bool = True,
) -> dict[str, Any]:
    """Validate a prospective deployment document for offline review only.

    Passing this validation does not authorize hardware and no daemon path
    consumes the returned data in this phase.
    """

    manifest = _load_secure_json(manifest_path, require_root_owner=require_root_owner)
    authorization = _load_secure_json(authorization_path, require_root_owner=require_root_owner)
    require_exact_keys(
        manifest,
        {
            "schema",
            "deployment_id",
            "calibrated",
            "calibration_sha256",
            "bus_paths",
            "devices",
        },
    )
    require_exact_keys(
        authorization,
        {"schema", "deployment_id", "authorized_bench", "expires_unix_s", "purpose"},
    )
    if manifest["schema"] != "microduck-arm-deployment-v1":
        raise ContractError("unsupported deployment manifest schema")
    if authorization["schema"] != "microduck-arm-bench-authorization-v1":
        raise ContractError("unsupported bench authorization schema")
    if manifest["calibrated"] is not True:
        raise ContractError("deployment is not calibrated")
    digest = manifest["calibration_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ContractError("calibration_sha256 must be a lowercase SHA-256 digest")
    if authorization["authorized_bench"] is not True:
        raise ContractError("bench authorization is not granted")
    if authorization["deployment_id"] != manifest["deployment_id"]:
        raise ContractError("bench authorization does not match deployment_id")
    expiry = finite_number(authorization["expires_unix_s"], "expires_unix_s")
    if expiry <= (time.time() if now_unix_s is None else now_unix_s):
        raise ContractError("bench authorization has expired")
    if not isinstance(authorization["purpose"], str) or not authorization["purpose"].strip():
        raise ContractError("bench authorization purpose is required")
    if not isinstance(manifest["bus_paths"], dict) or not isinstance(manifest["devices"], dict):
        raise ContractError("bus_paths and devices must be objects")
    unknown_devices = manifest["devices"].keys() - DEVICES.keys()
    if unknown_devices:
        raise ContractError(f"unknown arm devices: {', '.join(sorted(unknown_devices))}")
    if set(manifest["bus_paths"]) != set(manifest["devices"]):
        raise ContractError("bus_paths must exactly match the deployment devices")

    for device_id, contract in DEVICES.items():
        if device_id not in manifest["devices"]:
            continue
        device = manifest["devices"][device_id]
        if not isinstance(device, dict):
            raise ContractError(f"devices.{device_id} must be an object")
        require_exact_keys(device, {"servo_ids", "joint_map_hash", "calibration"})
        if tuple(device["servo_ids"]) != contract.servo_ids:
            raise ContractError(f"{device_id} servo IDs do not match the reserved arm bus IDs")
        if device["joint_map_hash"] != contract.joint_map_hash:
            raise ContractError(f"{device_id} joint_map_hash mismatch")
        calibration = device["calibration"]
        if not isinstance(calibration, list) or len(calibration) != 6:
            raise ContractError(f"{device_id} calibration must contain 6 joint entries")
        for joint, entry in zip(contract.joints, calibration):
            if not isinstance(entry, dict):
                raise ContractError(f"{device_id} calibration entry must be an object")
            require_exact_keys(entry, {"name", "zero_count", "direction", "counts_per_unit"})
            if entry["name"] != joint.name:
                raise ContractError(f"{device_id} calibration order mismatch at {joint.name}")
            zero = finite_number(entry["zero_count"], f"{device_id}.{joint.name}.zero_count")
            direction = finite_number(entry["direction"], f"{device_id}.{joint.name}.direction")
            scale = finite_number(
                entry["counts_per_unit"], f"{device_id}.{joint.name}.counts_per_unit"
            )
            if direction not in (-1.0, 1.0) or scale <= 0:
                raise ContractError(f"{device_id} invalid calibration at {joint.name}")
            endpoints = (zero + direction * joint.minimum * scale, zero + direction * joint.maximum * scale)
            if min(endpoints) < 0 or max(endpoints) > 4095:
                raise ContractError(f"{device_id} {joint.name} limits map outside XL330 position range")
        if not isinstance(manifest["bus_paths"][device_id], str) or not manifest["bus_paths"][device_id].startswith("/dev/"):
            raise ContractError(f"missing bus path for {device_id}")

    if not manifest["devices"]:
        raise ContractError("deployment manifest contains no arm devices")
    calibrated_payload = json.dumps(
        manifest["devices"],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    actual_digest = hashlib.sha256(calibrated_payload).hexdigest()
    if digest != actual_digest:
        raise ContractError("calibration_sha256 does not match devices calibration payload")
    return {"manifest": manifest, "authorization": authorization}
