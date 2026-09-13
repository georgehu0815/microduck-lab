from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "hardware/microduck-arm-v1c/spec/design.json"
VENDOR_SOURCE_MANIFEST = (
    ROOT / "hardware/microduck-arm-v1c/vendor/SOURCE-MANIFEST.md"
)
DESIGN_ID = "microduck-single-arm-v1-c"
CASES = (
    "reach",
    "pick_place",
    "obstacle_relocation",
    "stationary_waypoint_carry",
    "stowed_arm_walking",
    "walking_carry",
)
BASE_MODEL_MODULE = "microduck_arm_v1.model"
BASE_ENV_MODULE = "microduck_arm_v1.env"
_CANDIDATES = frozenset({"A", "B"})


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _validate_mapping(design: dict) -> None:
    expected_joints = design["joint_order"][10:]
    mapping = design["actuator_mapping"]
    if len(mapping) != 5:
        raise ValueError("Expected five explicit v1-C arm actuator mappings")
    for index, item in enumerate(mapping):
        expected = {
            "joint_id": f"J{index + 1}",
            "motor_id": f"M{index + 1}",
            "joint_name": expected_joints[index],
            "action_index": index + 10,
            "provisional_bus_id": index + 21,
        }
        if item != expected:
            raise ValueError("Invalid v1-C joint, motor, action, or bus mapping")


def _validate_baseline(design: dict) -> None:
    if design.get("design_id") != DESIGN_ID:
        raise ValueError("Invalid v1-C design identity")
    joint_order = design.get("joint_order", [])
    if len(joint_order) != 15 or len(set(joint_order)) != 15:
        raise ValueError("Invalid v1-C 15-joint morphology contract")
    if len(design.get("home_rad", [])) != 15:
        raise ValueError("Expected explicit 15-joint v1-C home pose")
    arm = design.get("arm", {})
    if arm.get("actuator_count") != 5 or arm.get("pose_joint_count") != 4:
        raise ValueError("Expected ten leg and five arm actuators")
    if arm.get("required_continuous_torque_margin") != 3.0:
        raise ValueError("v1-C requires a 3x continuous torque margin")
    if arm.get("shoulder_pivot_local_m") != [0, 0, 0.018]:
        raise ValueError("v1-C shoulder pivot must align with the motor center")
    pads = arm.get("contact_pads", {})
    if (
        pads.get("count") != 2
        or pads.get("full_size_m") != [0.024, 0.005, 0.008]
        or pads.get("center_local_m") != [0.023, 0, 0]
        or pads.get("mass_each_kg") != 0.005
        or pads.get("solref") != [0.004, 1]
        or pads.get("solimp") != [0.95, 0.99, 0.001]
        or pads.get("status") != "unmeasured_contact_screening"
    ):
        raise ValueError("Invalid v1-C analytical contact-pad screening contract")
    compensation = arm.get("gravity_compensation", {})
    if (
        compensation.get("mode") != "teacher_position_target_feedforward"
        or compensation.get("maximum_position_offset_rad") != 0.04
        or compensation.get("state_source") != "simulator_truth_qfrc_bias"
        or compensation.get("hardware_status") != "unqualified"
    ):
        raise ValueError("Invalid v1-C teacher gravity feedforward contract")
    control = design.get("control", {})
    if control.get("arm_joint_slew_limit_rad_s") != 0.4:
        raise ValueError("v1-C arm slew screening limit must remain 0.4 rad/s")
    if control.get("leg_joint_slew_limit_rad_s") != 6.0:
        raise ValueError("v1-C leg slew screening limit must remain 6 rad/s")
    _validate_mapping(design)

    categories = design["engineering_bom"]["categories"]
    total = sum(item["mass_kg"] for item in categories.values())
    if abs(total - design["engineering_bom"]["target_total_kg"]) > 1e-12:
        raise ValueError("v1-C cartridge BOM does not sum to its target")
    gear = arm["gripper_gear_candidate"]
    center_distance = (
        gear["module_m"] * (gear["left_teeth"] + gear["right_teeth"]) / 2
    )
    if abs(center_distance - gear["center_distance_m"]) > 1e-12:
        raise ValueError("Invalid v1-C equal-gear center-distance formula")
    if design["mass_properties"]["principal_inertia_release_gate"] != "fail":
        raise ValueError("Unmeasured principal inertia must fail release")
    if (
        design["mass_properties"]["status"]
        != "analytical_box_estimate_not_CAD_or_measured"
    ):
        raise ValueError("v1-C inertia estimates must not be represented as CAD")
    ledger = design.get("simulation_mass_ledger", {})
    modeled_total = sum(ledger.get("components_kg", {}).values())
    if abs(modeled_total - 0.166) > 1e-12:
        raise ValueError("v1-C modeled arm geom mass must be 0.166 kg")
    if abs(ledger.get("modeled_arm_geoms_excluding_mount_kg", 0) - 0.146) > 1e-12:
        raise ValueError("v1-C modeled arm mass excluding mount must be 0.146 kg")
    if ledger.get("target_enforcement") != "not_forced_to_match_simulation_geom_mass":
        raise ValueError("Engineering mass target must remain distinct from sim mass")
    scene = design.get("simulation_scene", {})
    if scene.get("scenario_id") != "raised_tabletop_v1_c":
        raise ValueError("v1-C requires the distinct raised-table scenario")
    if scene.get("old_floor_level_success_transferable") is not False:
        raise ValueError("Old floor-level task success cannot qualify v1-C")
    if scene.get("fixture_classification") != (
        "raised_table_fixture_screening_not_original_task"
    ):
        raise ValueError("Fixture evidence must be labeled as a distinct raised-table task")
    if scene.get("surface_top_m") != scene.get("tabletop_top_z_m"):
        raise ValueError("Model surface_top_m alias must match tabletop_top_z_m")
    expected_object_z = scene["tabletop_top_z_m"] + scene["object_half_height_m"]
    if abs(scene["object_initial_center_m"][2] - expected_object_z) > 1e-12:
        raise ValueError("v1-C object center must rest on the raised tabletop")
    source = ROOT / design["source_model"]
    if sha256(source) != design["source_model_sha256"]:
        raise ValueError(
            "Source Microduck MJCF changed; explicitly review and revise provenance"
        )
    provenance = design.get("source_provenance", {})
    if provenance.get("vendor_source_manifest") != str(
        VENDOR_SOURCE_MANIFEST.relative_to(ROOT)
    ):
        raise ValueError("Invalid v1-C vendor source manifest reference")
    if provenance.get("shop_rating_values_ingested") is not False:
        raise ValueError("Shop ratings must not be mixed into manufacturer ratings")
    if not VENDOR_SOURCE_MANIFEST.is_file():
        raise ValueError("Missing v1-C vendor SOURCE-MANIFEST")


def load_design(candidate: str = "A") -> dict:
    if not isinstance(candidate, str) or candidate not in _CANDIDATES:
        raise ValueError(f"Unknown v1-C candidate: {candidate!r}; expected A or B")
    design = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    _validate_baseline(design)

    selected = copy.deepcopy(design["candidates"][candidate])
    motor_skus = selected["motor_skus_by_motor"]
    if set(motor_skus) != {f"M{index}" for index in range(1, 6)}:
        raise ValueError("Candidate must assign exactly M1 through M5")
    catalog = design["motor_catalog"]
    if any(sku not in catalog for sku in motor_skus.values()):
        raise ValueError("Candidate references an unknown motor SKU")
    if any(catalog[sku]["mass_kg"] != 0.018 for sku in motor_skus.values()):
        raise ValueError("Both qualified candidates must retain 18 g servo mass")

    normalized = copy.deepcopy(design)
    normalized["candidate"] = candidate
    normalized["candidate_configuration"] = selected
    normalized["arm"]["motor_sku"] = (
        "XL330-M288-T"
        if len(set(motor_skus.values())) == 1
        else "mixed:3xXL330-M288-T+2xXL330-M077-T"
    )
    normalized["arm"]["motor_skus_by_motor"] = motor_skus
    normalized["arm"]["motor_gear_ratios_by_motor"] = {
        motor: catalog[sku]["gear_ratio"] for motor, sku in motor_skus.items()
    }
    normalized["arm"]["motor_mass_kg"] = 0.018
    normalized["arm"]["motor_total_mass_kg"] = sum(
        catalog[sku]["mass_kg"] for sku in motor_skus.values()
    )
    normalized["simulation_base"] = {
        "model_module": BASE_MODEL_MODULE,
        "env_module": BASE_ENV_MODULE,
        "compatibility": "v1-B geometry keys preserved; candidate overrides are explicit",
    }
    return normalized
