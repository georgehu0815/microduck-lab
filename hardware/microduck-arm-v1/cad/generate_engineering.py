#!/usr/bin/env python3
"""Generate MicroDuck single-arm v1-B engineering review artifacts.

The generator uses only the Python standard library. It intentionally does not
emit fabrication-ready servo interfaces, a released electrical design, or any
hardware test result.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "spec" / "design.json"
NOTICE = "ENGINEERING PROTOTYPE NOT FABRICATION RELEASE"


class SpecError(ValueError):
    """Raised when the canonical design specification is absent or inconsistent."""


def format_number(value: float) -> str:
    return "0" if abs(value) < 1e-12 else f"{value:g}"


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"canonical specification missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON in canonical specification: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecError("canonical specification root must be an object")
    return value


def parse_spec(raw: dict[str, Any], sha256: str) -> dict[str, Any]:
    """Parse and validate the current canonical v1-B schema."""

    if raw.get("schema_version") != 1:
        raise SpecError("schema_version must equal 1")
    if raw.get("design_id") != "microduck-single-arm-v1-b":
        raise SpecError("design_id must equal microduck-single-arm-v1-b")
    if raw.get("status") != (
        "simulation_prototype_not_fabrication_or_hardware_release"
    ):
        raise SpecError("status must remain simulation prototype only")
    if raw.get("units") != {
        "length": "m",
        "mass": "kg",
        "angle": "rad",
        "torque": "N*m",
    }:
        raise SpecError("units must match the canonical SI schema")
    if not isinstance(raw.get("revision"), str) or not raw["revision"]:
        raise SpecError("revision must be a non-empty string")

    canonical_arm = raw.get("arm")
    canonical_power = raw.get("power")
    if isinstance(canonical_arm, dict) and isinstance(canonical_power, dict):
        if canonical_arm.get("gripper_coupling") != (
            "opposite_z_rotation_equal_gear_ratio_left_driven_right_passive"
        ):
            raise SpecError("arm.gripper_coupling does not match v1-B")
        if canonical_arm.get("gravity_compensation") != "not_implemented_or_assumed":
            raise SpecError("arm.gravity_compensation does not match v1-B")
        if canonical_power.get("topology") != "dedicated_arm_pack_positive_domain":
            raise SpecError("power.topology does not match v1-B")
        if canonical_power.get("body_positive_connected_to_arm") is not False:
            raise SpecError("arm power must remain separate from body positive")
        if canonical_power.get("pack_bms_charger") is not None:
            raise SpecError("power.pack_bms_charger must remain unresolved")
        if canonical_power.get("qualified_regulator") is not None:
            raise SpecError("power.qualified_regulator must remain unresolved")
        if canonical_power.get("protection_and_wire_ratings") is not None:
            raise SpecError(
                "power.protection_and_wire_ratings must remain unresolved"
            )
        envelope_w_d_h = canonical_arm.get("motor_envelope_m")
        links_m = canonical_arm.get("link_lengths_m")
        joint_ranges = canonical_arm.get("joint_ranges_rad")
        if (
            isinstance(envelope_w_d_h, list)
            and len(envelope_w_d_h) == 3
            and isinstance(links_m, list)
            and len(links_m) == 3
            and isinstance(joint_ranges, list)
            and len(joint_ranges) == 5
            and isinstance(canonical_arm.get("servo_centers_local_m"), dict)
        ):
            try:
                normalized = {
                    "variant": raw["design_id"],
                    "pose_axes": canonical_arm["pose_joint_count"],
                    "actuator_count": canonical_arm["actuator_count"],
                    "motor_model": canonical_arm["motor_sku"],
                    "motor_mass_g": float(canonical_arm["motor_mass_kg"]) * 1000,
                    "motor_envelope_mm": [
                        float(envelope_w_d_h[0]) * 1000,
                        float(envelope_w_d_h[2]) * 1000,
                        float(envelope_w_d_h[1]) * 1000,
                    ],
                    "shoulder_elbow_mm": float(links_m[0]) * 1000,
                    "elbow_wrist_mm": float(links_m[1]) * 1000,
                    "wrist_tcp_mm": float(links_m[2]) * 1000,
                    "jaw_center_offset_mm": float(
                        canonical_arm["gripper_pivot_half_spacing_m"]
                    )
                    * 1000,
                    "jaw_length_mm": float(
                        canonical_arm["gripper_finger_length_m"]
                    )
                    * 1000,
                    "jaw_ratio": -1.0,
                    "gripper_max_closure_rad": float(
                        canonical_arm["gripper_max_closure_rad"]
                    ),
                    "gripper_left_range_rad": [
                        float(joint_ranges[4][0]),
                        float(joint_ranges[4][1]),
                    ],
                    "gripper_right_range_rad": [
                        -float(joint_ranges[4][1]),
                        -float(joint_ranges[4][0]),
                    ],
                    "trunk_target_g": float(
                        canonical_arm["mount_mass_target_kg"]
                    )
                    * 1000,
                    "cartridge_target_g": float(canonical_arm["mass_target_kg"])
                    * 1000,
                    "power_voltage_v": float(
                        canonical_power["arm_voltage_target_v"]
                    ),
                    "revision": raw["revision"],
                    "status": raw["status"],
                    "replacement_mass_assumptions": raw[
                        "replacement_mass_assumptions"
                    ],
                    "replacement_mass_status": raw["replacement_mass_status"],
                    "required_continuous_torque_margin": float(
                        canonical_arm["required_continuous_torque_margin"]
                    ),
                    "servo_centers_local_mm": {
                        name: [float(value) * 1000 for value in center]
                        for name, center in canonical_arm[
                            "servo_centers_local_m"
                        ].items()
                    },
                    "canonical_sha256": sha256,
                }
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                raise SpecError(f"invalid or missing canonical field: {exc}") from exc
            return validate_normalized(normalized)

    raise SpecError("canonical specification does not match the required schema")


def validate_normalized(normalized: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "variant": "microduck-single-arm-v1-b",
        "pose_axes": 4,
        "actuator_count": 5,
        "motor_model": "M288",
        "motor_mass_g": 18.0,
        "motor_envelope_mm": [20.0, 34.0, 26.0],
        "shoulder_elbow_mm": 55.0,
        "elbow_wrist_mm": 50.0,
        "wrist_tcp_mm": 30.0,
        "jaw_center_offset_mm": 12.0,
        "jaw_length_mm": 30.0,
        "jaw_ratio": -1.0,
        "gripper_max_closure_rad": 0.32,
        "gripper_left_range_rad": [-0.32, 0.0],
        "gripper_right_range_rad": [0.0, 0.32],
        "trunk_target_g": 20.0,
        "cartridge_target_g": 160.0,
        "power_voltage_v": 5.0,
        "required_continuous_torque_margin": 3.0,
        "servo_centers_local_mm": {
            "yaw": [0.0, 0.0, -25.0],
            "shoulder": [0.0, 0.0, 18.0],
            "elbow": [45.0, 0.0, 20.0],
            "wrist": [25.0, 0.0, 20.0],
            "gripper": [7.0, 0.0, 22.0],
        },
    }
    errors: list[str] = []
    for name, expected_value in expected.items():
        actual = normalized[name]
        if actual is None:
            errors.append(f"missing {name}")
            continue
        if name == "motor_model":
            if actual != "XL330-M288-T":
                errors.append(f"{name}={actual!r}, expected XL330-M288-T")
        elif name == "motor_envelope_mm":
            try:
                numeric = [float(value) for value in actual]
            except (TypeError, ValueError):
                errors.append(f"{name} must be a three-number list")
            else:
                if numeric != expected_value:
                    errors.append(f"{name}={numeric!r}, expected {expected_value!r}")
                normalized[name] = numeric
        elif name == "servo_centers_local_mm":
            if not isinstance(actual, dict):
                errors.append(f"{name} must be an object")
            else:
                converted = {
                    key: [float(value) for value in center]
                    for key, center in actual.items()
                }
                if converted != expected_value:
                    errors.append(
                        f"{name}={converted!r}, expected {expected_value!r}"
                    )
                normalized[name] = converted
        elif isinstance(expected_value, list):
            try:
                numeric = [float(value) for value in actual]
            except (TypeError, ValueError):
                errors.append(f"{name} must be a number list")
            else:
                if numeric != expected_value:
                    errors.append(f"{name}={numeric!r}, expected {expected_value!r}")
                normalized[name] = numeric
        elif isinstance(expected_value, float):
            try:
                if float(actual) != expected_value:
                    errors.append(f"{name}={actual!r}, expected {expected_value!r}")
                normalized[name] = float(actual)
            except (TypeError, ValueError):
                errors.append(f"{name}={actual!r}, expected a number")
        elif actual != expected_value:
            errors.append(f"{name}={actual!r}, expected {expected_value!r}")

    if errors:
        raise SpecError("; ".join(errors))

    replacement_expected = {
        "relocated_electronics": (0.070, [-0.010, 0.0, 0.018]),
        "dedicated_arm_pack": (0.080, [-0.040, 0.0, 0.015]),
        "power_and_communications": (0.025, [-0.020, 0.0, 0.034]),
        "sensor_shell": (0.008, [-0.020, 0.0, 0.059]),
    }
    replacement = normalized.get("replacement_mass_assumptions")
    if not isinstance(replacement, dict):
        raise SpecError("missing replacement_mass_assumptions")
    if set(replacement) != set(replacement_expected):
        raise SpecError("replacement_mass_assumptions keys do not match schema")
    replacement_errors: list[str] = []
    for name, (mass_kg, position_m) in replacement_expected.items():
        item = replacement.get(name)
        if not isinstance(item, dict):
            replacement_errors.append(f"missing replacement mass {name}")
            continue
        if float(item.get("mass_kg", -1)) != mass_kg:
            replacement_errors.append(f"{name} mass must remain {mass_kg} kg")
        actual_position = [float(value) for value in item.get("pos_m", [])]
        if actual_position != position_m:
            replacement_errors.append(
                f"{name} position {actual_position!r}, expected {position_m!r}"
            )
        size = item.get("size_m")
        if (
            not isinstance(size, list)
            or len(size) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in size
            )
        ):
            replacement_errors.append(f"{name} size_m must contain three numbers")
    if replacement_errors:
        raise SpecError("; ".join(replacement_errors))

    if normalized.get("replacement_mass_status") != (
        "unmeasured_explicit_placeholder_budgets_not_manufacturer_parts_or_verified_packaging"
    ):
        raise SpecError("replacement masses must remain explicit unmeasured placeholders")

    if normalized["status"] != (
        "simulation_prototype_not_fabrication_or_hardware_release"
    ):
        raise SpecError("status must remain simulation prototype only")

    return normalized


def svg_document(width: int, height: int, body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" role="img"
     aria-label="{NOTICE}">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #17212b; }}
    .title {{ font-size: 24px; font-weight: 700; }}
    .subtitle {{ font-size: 14px; font-weight: 700; fill: #9b1c1c; }}
    .label {{ font-size: 14px; }}
    .small {{ font-size: 12px; }}
    .tiny {{ font-size: 10px; }}
    .part {{ fill: #dce4e8; stroke: #263746; stroke-width: 2; }}
    .blank {{ fill: #fff1c7; stroke: #8a5a00; stroke-width: 2; }}
    .link {{ stroke: #d06b16; stroke-width: 12; stroke-linecap: round; }}
    .axis {{ fill: white; stroke: #263746; stroke-width: 2; }}
    .dim {{ fill: none; stroke: #087f5b; stroke-width: 1.5; }}
    .data {{ fill: none; stroke: #087f5b; stroke-width: 3; }}
    .power {{ fill: none; stroke: #c92a2a; stroke-width: 4; }}
    .ground {{ fill: none; stroke: #343a40; stroke-width: 4; }}
    .box {{ fill: #f7f9fa; stroke: #263746; stroke-width: 2; }}
    .unresolved {{ fill: #fff4d6; stroke: #a66b00; stroke-width: 2; }}
    .forbidden {{ fill: #fff0f0; stroke: #c92a2a; stroke-width: 2; }}
  </style>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4"
            orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 z" fill="#087f5b"/>
    </marker>
  </defs>
{body}
</svg>
"""


def dim(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 - 8
    return (
        f'  <line class="dim" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        'marker-start="url(#arrow)" marker-end="url(#arrow)"/>\n'
        f'  <text class="small" x="{mx}" y="{my}" text-anchor="middle">'
        f"{html.escape(label)}</text>\n"
    )


def mechanical_svg(spec: dict[str, Any], source_label: str) -> str:
    scale = 4.0
    shoulder = (120.0, 285.0)
    elbow = (shoulder[0] + spec["shoulder_elbow_mm"] * scale, 245.0)
    wrist = (elbow[0] + spec["elbow_wrist_mm"] * scale, 285.0)
    tcp = (wrist[0] + spec["wrist_tcp_mm"] * scale, 285.0)
    envelope = " x ".join(f"{v:g}" for v in spec["motor_envelope_mm"])
    centers = spec["servo_centers_local_mm"]
    center_text = ", ".join(
        f"{name} [{','.join(format_number(value) for value in centers[name])}]"
        for name in ("yaw", "shoulder", "elbow", "wrist", "gripper")
    )
    left_range = spec["gripper_left_range_rad"]
    right_range = spec["gripper_right_range_rad"]
    motor_total_g = spec["actuator_count"] * spec["motor_mass_g"]
    body = f"""
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="34" y="40">MicroDuck single arm v1-B - mechanical dimensions</text>
  <text class="subtitle" x="34" y="64">{NOTICE}</text>
  <text class="small" x="34" y="84">Canonical input: {html.escape(source_label)} | revision {html.escape(str(spec["revision"]))} | SHA-256 {spec["canonical_sha256"][:12]}...</text>

  <text class="label" x="34" y="120">Side elevation - nominal kinematic axis distances</text>
  <rect class="blank" x="70" y="310" width="100" height="30"/>
  <text class="tiny" x="120" y="329" text-anchor="middle">TRUNK MOUNT BLANK - {spec["trunk_target_g"]:g} g target</text>
  <circle class="axis" cx="{shoulder[0]}" cy="{shoulder[1]}" r="7"/>
  <line class="link" x1="{shoulder[0]}" y1="{shoulder[1]}" x2="{elbow[0]}" y2="{elbow[1]}"/>
  <circle class="axis" cx="{elbow[0]}" cy="{elbow[1]}" r="7"/>
  <line class="link" x1="{elbow[0]}" y1="{elbow[1]}" x2="{wrist[0]}" y2="{wrist[1]}"/>
  <circle class="axis" cx="{wrist[0]}" cy="{wrist[1]}" r="7"/>
  <line class="link" x1="{wrist[0]}" y1="{wrist[1]}" x2="{tcp[0]}" y2="{tcp[1]}"/>
  <circle cx="{tcp[0]}" cy="{tcp[1]}" r="7" fill="#087f5b"/>
  <text class="small" x="{tcp[0] + 10}" y="{tcp[1] - 10}">TCP</text>
{dim(shoulder[0], 190, elbow[0], 190, f'{spec["shoulder_elbow_mm"]:g} mm shoulder-elbow nominal')}
{dim(elbow[0], 375, wrist[0], 375, f'{spec["elbow_wrist_mm"]:g} mm elbow-wrist nominal')}
{dim(wrist[0], 190, tcp[0], 190, f'{spec["wrist_tcp_mm"]:g} mm wrist-TCP nominal')}

  <rect class="unresolved" x="34" y="415" width="832" height="132"/>
  <text class="label" x="50" y="443">MOTOR ENVELOPE AND SUPPORT-BLANK BOUNDARY</text>
  <text class="small" x="50" y="468">{spec["actuator_count"]} {html.escape(spec["motor_model"])} candidates, {spec["motor_mass_g"]:g} g each; body envelope {envelope} mm for clearance only.</text>
  <text class="small" x="50" y="491">Source mesh depth is about 29 mm while manufacturer body depth is 26 mm. Do not rescale or infer a mating face.</text>
  <text class="small" x="50" y="514">No horn pilot, bolt circle, screw size/depth, bearing seat, cable relief, tolerance, or hole pattern is released.</text>
  <text class="tiny" x="50" y="529">Kinematic sketch omits motor boxes. Canonical local centers (mm): {center_text}.</text>

  <text class="label" x="34" y="580">Top view - counter-rotating geared gripper concept</text>
  <line x1="80" y1="700" x2="520" y2="700" stroke="#adb5bd" stroke-dasharray="6 5"/>
  <circle class="part" cx="235" cy="652" r="48"/>
  <circle class="part" cx="235" cy="748" r="48"/>
  <circle class="axis" cx="235" cy="652" r="5"/>
  <circle class="axis" cx="235" cy="748" r="5"/>
  <line x1="235" y1="652" x2="355" y2="677" stroke="#d06b16" stroke-width="14" stroke-linecap="round"/>
  <line x1="235" y1="748" x2="355" y2="723" stroke="#d06b16" stroke-width="14" stroke-linecap="round"/>
  <text class="small" x="370" y="683">jaw A closes clockwise</text>
  <text class="small" x="370" y="725">jaw B closes counter-clockwise</text>
{dim(170, 652, 170, 700, "12 mm")}
{dim(170, 700, 170, 748, "12 mm")}
{dim(235, 815, 355, 815, f'{spec["jaw_length_mm"]:g} mm jaw length')}
  <text class="small" x="570" y="640">Equal pitch gears</text>
  <text class="small" x="570" y="664">ratio = {spec["jaw_ratio"]:g}</text>
  <text class="small" x="570" y="688">axes parallel to +Z</text>
  <text class="small" x="570" y="712">centers y = +/-{spec["jaw_center_offset_mm"]:g} mm</text>
  <text class="small" x="570" y="736">rotary jaws, no sliders</text>
  <text class="small" x="570" y="760">left q [{format_number(left_range[0])},{format_number(left_range[1])}], right q [{format_number(right_range[0])},{format_number(right_range[1])}] rad</text>
  <text class="small" x="570" y="784">right = -left; max closure {spec["gripper_max_closure_rad"]:g} rad</text>
  <text class="small" x="570" y="808">gear teeth/module/backlash TBD</text>

  <rect class="unresolved" x="34" y="858" width="832" height="104"/>
  <text class="label" x="50" y="886">MASS AND LOAD TARGETS - UNVERIFIED</text>
  <text class="small" x="50" y="910">Central trunk mount target: {spec["trunk_target_g"]:g} g. Complete arm cartridge target: {spec["cartridge_target_g"]:g} g. Motors alone: {motor_total_g:g} g.</text>
  <text class="small" x="50" y="934">Required shoulder continuous capability evidence: at least {spec["required_continuous_torque_margin"]:g}x measured worst-case gravity torque. Evidence is missing.</text>
  <text class="tiny" x="34" y="1000">Dimensioned review drawing; not a scale-controlled manufacturing drawing. TBD mating machining must remain solid/blank.</text>
  <text class="tiny" x="34" y="1014">Motor envelopes use canonical local centers; overall packaging, interference, cable routing, and table clearance remain unverified.</text>
"""
    return svg_document(900, 1032, body)


def wiring_svg(spec: dict[str, Any], source_label: str) -> str:
    motor_rows = []
    labels = ("M1 BASE", "M2 SHOULDER", "M3 ELBOW", "M4 WRIST", "M5 GRIPPER")
    for index, label in enumerate(labels):
        y = 180 + index * 78
        motor_rows.append(
            f"""
  <rect class="box" x="665" y="{y}" width="195" height="58"/>
  <text class="label" x="678" y="{y + 22}">{label}</text>
  <text class="tiny" x="678" y="{y + 43}">1 GND | 2 VDD | 3 DATA</text>
  <rect class="unresolved" x="550" y="{y + 9}" width="78" height="38"/>
  <text class="tiny" x="589" y="{y + 32}" text-anchor="middle">F{index + 1} TBD</text>
  <line class="power" x1="505" y1="{y + 28}" x2="550" y2="{y + 28}"/>
  <line class="power" x1="628" y1="{y + 28}" x2="665" y2="{y + 28}"/>
  <line class="data" x1="420" y1="{y + 48}" x2="665" y2="{y + 48}"/>
  <line class="ground" x1="450" y1="{y + 56}" x2="665" y2="{y + 56}"/>
"""
        )
    return svg_document(
        900,
        780,
        f"""
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="32" y="40">MicroDuck arm v1-B - bench wiring concept</text>
  <text class="subtitle" x="32" y="64">{NOTICE}</text>
  <text class="small" x="32" y="84">Canonical input: {html.escape(source_label)} | SHA-256 {spec["canonical_sha256"][:12]}... | ratings TBD</text>

  <rect class="unresolved" x="32" y="120" width="170" height="92"/>
  <text class="label" x="48" y="146">SEPARATE ARM PACK</text>
  <text class="small" x="48" y="168">5 V domain target</text>
  <text class="small" x="48" y="188">pack/BMS/connector TBD</text>
  <rect class="unresolved" x="235" y="130" width="82" height="52"/>
  <text class="tiny" x="276" y="152" text-anchor="middle">F_MAIN</text>
  <text class="tiny" x="276" y="168" text-anchor="middle">RATING TBD</text>
  <rect class="unresolved" x="350" y="120" width="155" height="92"/>
  <text class="label" x="366" y="146">DC DISCONNECT</text>
  <text class="small" x="366" y="168">candidate function</text>
  <text class="small" x="366" y="188">device/rating TBD</text>
  <line class="power" x1="202" y1="149" x2="235" y2="149"/>
  <line class="power" x1="317" y1="149" x2="350" y2="149"/>
  <line class="power" x1="505" y1="149" x2="505" y2="520"/>

  <rect class="box" x="32" y="265" width="235" height="105"/>
  <text class="label" x="48" y="291">HOST USB + U2D2</text>
  <text class="small" x="48" y="314">bench candidate</text>
  <text class="small" x="48" y="336">mobile integration: TBD</text>
  <text class="small" x="48" y="358">TTL: 1 GND, 2 N/C, 3 DATA</text>
  <rect class="unresolved" x="305" y="238" width="115" height="92"/>
  <text class="label" x="362.5" y="265" text-anchor="middle">POWER HUB</text>
  <text class="small" x="362.5" y="288" text-anchor="middle">bench candidate</text>
  <text class="tiny" x="362.5" y="311" text-anchor="middle">branch protection TBD</text>
  <line class="data" x1="267" y1="348" x2="420" y2="348"/>
  <line class="data" x1="420" y1="228" x2="420" y2="552"/>
  <line class="ground" x1="267" y1="365" x2="292" y2="365"/>
  <line class="ground" x1="450" y1="228" x2="450" y2="600"/>
  <line class="ground" x1="202" y1="191" x2="292" y2="191"/>
  <line class="ground" x1="292" y1="191" x2="292" y2="600"/>
  <line class="ground" x1="292" y1="600" x2="450" y2="600"/>
{''.join(motor_rows)}

  <rect class="forbidden" x="32" y="635" width="828" height="95"/>
  <text class="label" x="48" y="661">PROHIBITED / 禁止</text>
  <text class="small" x="48" y="684">Never connect ARM_5V to original MicroDuck battery positive, body VDD, USB VBUS, or body actuator DATA.</text>
  <text class="small" x="48" y="707">No mains wiring instructions are provided. No fuse, BMS, pack, cable, connector, or disconnect rating is approved.</text>
""",
    )


def circuit_svg(spec: dict[str, Any], source_label: str) -> str:
    return svg_document(
        900,
        620,
        f"""
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="32" y="40">MicroDuck arm v1-B - power/data circuit boundary</text>
  <text class="subtitle" x="32" y="64">{NOTICE}</text>
  <text class="small" x="32" y="84">Canonical input: {html.escape(source_label)} | SHA-256 {spec["canonical_sha256"][:12]}... | logical connectivity only</text>

  <rect class="unresolved" x="42" y="130" width="150" height="84"/>
  <text class="label" x="117" y="156" text-anchor="middle">5 V ARM SOURCE</text>
  <text class="small" x="117" y="180" text-anchor="middle">pack + BMS TBD</text>
  <text class="small" x="117" y="201" text-anchor="middle">ratings unresolved</text>
  <rect class="unresolved" x="245" y="145" width="95" height="54"/>
  <text class="small" x="292.5" y="169" text-anchor="middle">MAIN FUSE</text>
  <text class="tiny" x="292.5" y="187" text-anchor="middle">TBD</text>
  <rect class="unresolved" x="395" y="145" width="120" height="54"/>
  <text class="small" x="455" y="169" text-anchor="middle">DISCONNECT</text>
  <text class="tiny" x="455" y="187" text-anchor="middle">TBD</text>
  <rect class="unresolved" x="570" y="130" width="150" height="84"/>
  <text class="small" x="645" y="156" text-anchor="middle">5 BRANCHES</text>
  <text class="small" x="645" y="180" text-anchor="middle">protection TBD</text>
  <text class="small" x="645" y="201" text-anchor="middle">connector TBD</text>
  <rect class="box" x="770" y="130" width="90" height="84"/>
  <text class="label" x="815" y="161" text-anchor="middle">M1-M5</text>
  <text class="small" x="815" y="185" text-anchor="middle">VDD</text>
  <text class="small" x="815" y="205" text-anchor="middle">GND</text>
  <line class="power" x1="192" y1="155" x2="245" y2="155"/>
  <line class="power" x1="340" y1="155" x2="395" y2="155"/>
  <line class="power" x1="515" y1="155" x2="570" y2="155"/>
  <line class="power" x1="720" y1="155" x2="770" y2="155"/>
  <line class="ground" x1="117" y1="214" x2="117" y2="240"/>
  <line class="ground" x1="117" y1="240" x2="815" y2="240"/>
  <line class="ground" x1="815" y1="214" x2="815" y2="240"/>

  <rect class="box" x="42" y="310" width="150" height="92"/>
  <text class="label" x="117" y="338" text-anchor="middle">HOST USB</text>
  <text class="small" x="117" y="362" text-anchor="middle">logic only</text>
  <text class="small" x="117" y="384" text-anchor="middle">VBUS no-connect</text>
  <rect class="box" x="265" y="310" width="150" height="92"/>
  <text class="label" x="340" y="338" text-anchor="middle">U2D2 TTL</text>
  <text class="small" x="340" y="362" text-anchor="middle">1 GND | 2 N/C</text>
  <text class="small" x="340" y="384" text-anchor="middle">3 DATA</text>
  <rect class="box" x="570" y="310" width="150" height="92"/>
  <text class="label" x="645" y="338" text-anchor="middle">TTL MULTIDROP</text>
  <text class="small" x="645" y="362" text-anchor="middle">DATA + ARM_GND</text>
  <text class="small" x="645" y="384" text-anchor="middle">topology review TBD</text>
  <rect class="box" x="770" y="310" width="90" height="92"/>
  <text class="label" x="815" y="348" text-anchor="middle">M1-M5</text>
  <text class="small" x="815" y="374" text-anchor="middle">TTL</text>
  <line class="data" x1="192" y1="348" x2="265" y2="348"/>
  <line class="data" x1="415" y1="348" x2="570" y2="348"/>
  <line class="data" x1="720" y1="348" x2="770" y2="348"/>
  <line class="ground" x1="340" y1="402" x2="340" y2="470"/>
  <line class="ground" x1="117" y1="199" x2="117" y2="470"/>
  <line class="ground" x1="117" y1="470" x2="645" y2="470"/>
  <line class="ground" x1="645" y1="402" x2="645" y2="470"/>
  <text class="small" x="380" y="493" text-anchor="middle">single common ARM_GND signal reference</text>

  <rect class="forbidden" x="42" y="530" width="818" height="58"/>
  <text class="small" x="58" y="555">NO CONNECT: original MicroDuck battery +, body VDD, body DATA, USB VBUS to ARM_5V, U2D2 pin 2.</text>
  <text class="small" x="58" y="577">This circuit does not qualify pack/BMS/fuses/connectors or prove mobile suitability.</text>
""",
    )


def netlist(spec: dict[str, Any], source_label: str) -> dict[str, Any]:
    motor_refs = [
        f"M{index}" for index in range(1, spec["actuator_count"] + 1)
    ]
    nets = [
        {
            "name": "ARM_5V_SOURCE",
            "nodes": ["QUALIFIED_SOURCE_ASSEMBLY:POS", "F_MAIN:IN"],
            "status": "unresolved_source_chain",
        },
        {
            "name": "ARM_5V_FUSED",
            "nodes": ["F_MAIN:OUT", "SW_DC:IN"],
            "status": "unresolved_ratings",
        },
        {
            "name": "ARM_5V_ENABLED",
            "nodes": ["SW_DC:OUT"]
            + [
                f"F_M{index}:IN"
                for index in range(1, spec["actuator_count"] + 1)
            ],
            "status": "unresolved_ratings",
        },
        {
            "name": "ARM_GND",
            "nodes": [
                "QUALIFIED_SOURCE_ASSEMBLY:NEG",
                "U2D2_TTL:1_GND",
                *[f"{ref}:1_GND" for ref in motor_refs],
            ],
            "status": "required_common_signal_reference",
        },
        {
            "name": "TTL_DATA",
            "nodes": [
                "U2D2_TTL:3_DATA",
                *[f"{ref}:3_DATA" for ref in motor_refs],
            ],
            "status": "bench_candidate_mobile_topology_tbd",
        },
    ]
    for index in range(1, spec["actuator_count"] + 1):
        nets.append(
            {
                "name": f"M{index}_VDD",
                "nodes": [f"F_M{index}:OUT", f"M{index}:2_VDD"],
                "status": "branch_rating_unresolved",
            }
        )
    return {
        "schema_version": "1.0",
        "design_id": spec["variant"],
        "title": "MicroDuck single arm v1-B logical power and TTL netlist",
        "notice": NOTICE,
        "canonical_spec": source_label,
        "canonical_revision": spec["revision"],
        "canonical_spec_sha256": spec["canonical_sha256"],
        "release": {
            "fabrication_release": False,
            "electrical_release": False,
            "hardware_acceptance": "not_run",
        },
        "architecture": {
            "actuator_count": 5,
            "actuator_model_candidate": spec["motor_model"],
            "arm_voltage_domain_v": spec["power_voltage_v"],
            "arm_power_domain": "separate_from_original_microduck_positive",
            "bench_interface_candidate": "U2D2 plus compatible hub",
            "mobile_interface": "TBD",
        },
        "components": {
            "BAT1": {
                "function": "separate arm battery source",
                "part_number": None,
                "voltage_rating": None,
                "current_rating": None,
                "status": "unresolved",
            },
            "BMS1": {
                "function": "battery protection/management",
                "part_number": None,
                "voltage_rating": None,
                "current_rating": None,
                "status": "unresolved",
            },
            "F_MAIN": {
                "function": "arm feeder protection",
                "part_number": None,
                "rating": None,
                "trip_curve": None,
                "status": "unresolved",
            },
            "SW_DC": {
                "function": "manual DC disconnect or e-stop power pole",
                "part_number": None,
                "dc_interrupt_rating": None,
                "status": "unresolved",
            },
            "U2D2": {
                "function": "bench USB to TTL interface candidate",
                "part_number": "ROBOTIS U2D2",
                "status": "bench_candidate_mobile_tbd",
            },
            "HUB1": {
                "function": "bench TTL/power distribution candidate",
                "part_number": None,
                "rating": None,
                "branch_protection": None,
                "status": "unresolved_not_cad_ready",
            },
            **{
                f"F_M{index}": {
                    "function": f"M{index} branch protection",
                    "part_number": None,
                    "rating": None,
                    "status": "unresolved",
                }
                for index in range(1, spec["actuator_count"] + 1)
            },
            **{
                ref: {
                    "function": function,
                    "candidate": spec["motor_model"],
                    "mass_g": spec["motor_mass_g"],
                    "pins": {"1": "GND", "2": "VDD", "3": "DATA"},
                    "status": "candidate_requires_identity_and_bench_verification",
                }
                for ref, function in zip(
                    motor_refs,
                    ("base", "shoulder", "elbow", "wrist", "rotary_gripper"),
                    strict=True,
                )
            },
        },
        "nets": nets,
        "explicit_no_connects": [
            "ORIGINAL_MICRODUCK_BATTERY:POS",
            "ORIGINAL_MICRODUCK_BODY_BUS:VDD",
            "ORIGINAL_MICRODUCK_BODY_BUS:DATA",
            "USB:VBUS_TO_ARM_5V",
            "U2D2_TTL:2_NC",
        ],
        "unresolved_release_gates": [
            "battery pack chemistry, cell configuration, capacity, and enclosure",
            "BMS compatibility and all voltage/current/fault ratings",
            "main and per-motor fuse part numbers, ratings, and trip behavior",
            "wire gauge, insulation, flex life, routing, and strain relief",
            "connector families, mating cycles, polarity keying, and current ratings",
            "DC disconnect/e-stop device and interrupt rating",
            "distribution hub topology and branch protection",
            "mobile compute/interface architecture",
            "measured startup, transient, continuous, and fault currents",
        ],
        "internal_source_topology": (
            "intentionally unspecified until the exact pack, BMS, charger, "
            "regulator, regenerative behavior, and protection architecture are reviewed"
        ),
        "prohibited_instructions": [
            "No mains wiring or exposed-AC assembly is part of this package.",
            "Do not connect the arm source to the original MicroDuck battery positive.",
            "Do not infer a safe fuse or connector rating from arithmetic stall-current sums.",
        ],
    }


def scad_parameters(spec: dict[str, Any], source_label: str) -> str:
    values = ", ".join(f"{value:g}" for value in spec["motor_envelope_mm"])
    centers = spec["servo_centers_local_mm"]

    def vector(values: list[float]) -> str:
        return "[" + ", ".join(format_number(value) for value in values) + "]"

    return f"""/*
Generated from {source_label}.
Canonical spec SHA-256: {spec["canonical_sha256"]}.
{NOTICE}
Do not hand-edit; regenerate with cad/generate_engineering.py.
*/
design_revision = "{str(spec["revision"]).replace('"', '')}";
motor_model = "{spec["motor_model"]}";
motor_count = {spec["actuator_count"]};
motor_mass_g = {spec["motor_mass_g"]:g};
motor_envelope_mm = [{values}];
servo_center_yaw_local_mm = {vector(centers["yaw"])};
servo_center_shoulder_local_mm = {vector(centers["shoulder"])};
servo_center_elbow_local_mm = {vector(centers["elbow"])};
servo_center_wrist_local_mm = {vector(centers["wrist"])};
servo_center_gripper_local_mm = {vector(centers["gripper"])};
shoulder_elbow_mm = {spec["shoulder_elbow_mm"]:g};
elbow_wrist_mm = {spec["elbow_wrist_mm"]:g};
wrist_tcp_mm = {spec["wrist_tcp_mm"]:g};
jaw_center_offset_mm = {spec["jaw_center_offset_mm"]:g};
jaw_length_mm = {spec["jaw_length_mm"]:g};
jaw_coupling_ratio = {spec["jaw_ratio"]:g};
gripper_max_closure_rad = {spec["gripper_max_closure_rad"]:g};
gripper_left_range_rad = [{format_number(spec["gripper_left_range_rad"][0])}, {format_number(spec["gripper_left_range_rad"][1])}];
gripper_right_range_rad = [{format_number(spec["gripper_right_range_rad"][0])}, {format_number(spec["gripper_right_range_rad"][1])}];
trunk_mount_target_g = {spec["trunk_target_g"]:g};
arm_cartridge_target_g = {spec["cartridge_target_g"]:g};
"""


def release_report(spec: dict[str, Any], source_label: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "design_id": spec["variant"],
        "notice": NOTICE,
        "canonical_spec": source_label,
        "canonical_revision": spec["revision"],
        "canonical_spec_sha256": spec["canonical_sha256"],
        "fabrication_release": False,
        "manufacture_check": "FAIL_CLOSED",
        "blocking_gates": [
            "servo and horn official revision-controlled mating drawing unavailable",
            "purchased motor, horn, and fastener stack not measured",
            "mesh depth approximately 29 mm differs from 26 mm body envelope",
            "mount holes, horn pilot/bolt circle, thread engagement, and tolerances TBD",
            "gear module, tooth count, backlash, retention, and material TBD",
            "dual-side support/bearing geometry and load path TBD",
            "pack/BMS/fuse/wire/connector/disconnect selections and ratings TBD",
            "shoulder continuous torque evidence >=3x measured gravity torque missing",
            "20 g trunk and 160 g cartridge mass targets unverified",
            "overall packaging, self-interference, cable routing, and table clearance unverified",
            "hardware SOP and acceptance tests not executed",
        ],
    }


def export_pdfs(svg_paths: list[Path]) -> list[Path]:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        return []
    pdf_paths = []
    for svg_path in svg_paths:
        pdf_path = svg_path.with_suffix(".pdf")
        subprocess.run(
            [converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            check=True,
        )
        pdf_paths.append(pdf_path)
    return pdf_paths


def write_outputs(
    spec_path: Path, source_label: str, output_root: Path = ROOT
) -> dict[str, Any]:
    try:
        spec_bytes = spec_path.read_bytes()
    except FileNotFoundError as exc:
        raise SpecError(f"canonical specification missing: {spec_path}") from exc
    spec_sha256 = hashlib.sha256(spec_bytes).hexdigest()
    spec = parse_spec(load_json(spec_path), spec_sha256)
    cad_dir = output_root / "cad"
    electrical_dir = output_root / "electrical"
    cad_dir.mkdir(parents=True, exist_ok=True)
    electrical_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[Path, str] = {
        cad_dir / "design_parameters.scad": scad_parameters(spec, source_label),
        cad_dir / "mechanical-dimensions.svg": mechanical_svg(spec, source_label),
        electrical_dir / "wiring.svg": wiring_svg(spec, source_label),
        electrical_dir / "circuit.svg": circuit_svg(spec, source_label),
        electrical_dir / "netlist.json": json.dumps(
            netlist(spec, source_label), indent=2, ensure_ascii=False
        )
        + "\n",
        cad_dir / "manufacture-check.json": json.dumps(
            release_report(spec, source_label), indent=2, ensure_ascii=False
        )
        + "\n",
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    pdf_paths = export_pdfs(
        [
            cad_dir / "mechanical-dimensions.svg",
            electrical_dir / "wiring.svg",
            electrical_dir / "circuit.svg",
        ]
    )
    generated_paths = [*outputs, *pdf_paths]
    return {
        "status": "generated_review_artifacts",
        "notice": NOTICE,
        "canonical_spec_sha256": spec_sha256,
        "files": [str(path.relative_to(output_root)) for path in generated_paths],
        "pdf_export": "generated" if pdf_paths else "rsvg-convert_unavailable",
        "fabrication_release": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--source-label",
        default="hardware/microduck-arm-v1/spec/design.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT,
        help="Root containing cad/ and electrical/ output directories.",
    )
    parser.add_argument(
        "--manufacture-check",
        action="store_true",
        help="Return nonzero while any fabrication/electrical gate is unresolved.",
    )
    args = parser.parse_args()
    try:
        receipt = write_outputs(args.spec, args.source_label, args.output_root)
    except SpecError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}))
        return 2
    if args.manufacture_check:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "reason": "unresolved mechanical and electrical release gates",
                }
            )
        )
        return 3
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
