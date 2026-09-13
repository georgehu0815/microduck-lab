from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "hardware/microduck-arm-v1/spec/design.json"
DESIGN_ID = "microduck-single-arm-v1-b"
CASES = ("reach", "pick_place", "obstacle_relocation", "stationary_waypoint_carry", "stowed_arm_walking", "walking_carry")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_design():
    design = json.loads(SPEC_PATH.read_text())
    if design["design_id"] != DESIGN_ID or len(set(design["joint_order"])) != 15:
        raise ValueError("Invalid v1-B morphology contract")
    if len(design["home_rad"]) != 15 or design["arm"]["actuator_count"] != 5:
        raise ValueError("Expected ten leg and five arm actuators")
    source = ROOT / design["source_model"]
    if sha256(source) != design["source_model_sha256"]:
        raise ValueError("Source Microduck MJCF changed; explicitly review and revise provenance")
    return design
