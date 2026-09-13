#!/usr/bin/env python3
"""Generate review drawings and non-precision STL artifacts for MD-Arm-T1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from shared_components import write_outputs as write_shared_outputs


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "spec" / "md-arm-t1.json"
NETLIST_PATH = ROOT / "netlists" / "arm-ttl-netlist.json"
DEFAULT_OUTPUT = ROOT / "generated"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def svg_document(width: int, height: int, body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
     viewBox="0 0 {width} {height}" role="img">
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #16202a; }}
    .title {{ font-size: 24px; font-weight: 700; }}
    .label {{ font-size: 14px; }}
    .small {{ font-size: 12px; }}
    .dim {{ stroke: #0b6e69; stroke-width: 1.5; fill: none; }}
    .part {{ stroke: #243746; stroke-width: 2; fill: #d9e2e8; }}
    .link {{ stroke: #d47716; stroke-width: 12; stroke-linecap: round; }}
    .axis {{ fill: #ffffff; stroke: #243746; stroke-width: 2; }}
    .blocked {{ fill: #fff4d6; stroke: #a66b00; stroke-width: 2; }}
    .power {{ stroke: #c92a2a; stroke-width: 4; fill: none; }}
    .ground {{ stroke: #30343b; stroke-width: 4; fill: none; }}
    .data {{ stroke: #087f5b; stroke-width: 3; fill: none; }}
    .box {{ fill: #f7f9fa; stroke: #243746; stroke-width: 2; }}
    .danger {{ fill: #a61e1e; font-weight: 700; }}
  </style>
  <defs>
    <marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4"
            orient="auto-start-reverse">
      <path d="M0,0 L8,4 L0,8 z" fill="#0b6e69"/>
    </marker>
  </defs>
{body}
</svg>
"""


def dimension(x1: float, y1: float, x2: float, y2: float, label: str) -> str:
    mx = (x1 + x2) / 2
    my = (y1 + y2) / 2 - 7
    return (
        f'  <line class="dim" x1="{x1:.1f}" y1="{y1:.1f}" '
        f'x2="{x2:.1f}" y2="{y2:.1f}" marker-start="url(#arrow)" '
        f'marker-end="url(#arrow)"/>\n'
        f'  <text class="label" x="{mx:.1f}" y="{my:.1f}" '
        f'text-anchor="middle">{label}</text>\n'
    )


def station_plan_panel(
    x: float,
    y: float,
    width: float,
    title: str,
    spacing_mm: float,
    classification: str,
    detail: str,
) -> str:
    scale = 0.78
    center_x = x + width / 2
    center_y = y + 76
    half_spacing = spacing_mm * scale / 2
    left_x = center_x - half_spacing
    right_x = center_x + half_spacing
    base_size = 30
    return f"""
  <g>
    <rect class="box" x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="132"/>
    <text class="label" x="{center_x:.1f}" y="{y + 22:.1f}" text-anchor="middle">{title}</text>
    <text class="small" x="{center_x:.1f}" y="{y + 41:.1f}" text-anchor="middle">{classification}</text>
    <rect class="part" x="{left_x - base_size / 2:.1f}" y="{center_y - base_size / 2:.1f}"
          width="{base_size}" height="{base_size}"/>
    <rect class="part" x="{right_x - base_size / 2:.1f}" y="{center_y - base_size / 2:.1f}"
          width="{base_size}" height="{base_size}"/>
{dimension(left_x, center_y + 29, right_x, center_y + 29, f"{spacing_mm:g} mm base centers")}
    <text class="small" x="{center_x:.1f}" y="{y + 120:.1f}" text-anchor="middle">{detail}</text>
  </g>
"""


def generate_mechanical_svg(spec: dict) -> str:
    geom = spec["design_geometry"]
    upper = geom["upper_arm_axis_distance_mm"]
    fore = geom["forearm_axis_distance_mm"]
    wrist = geom["wrist_to_tcp_mm"]
    reach = geom["maximum_geometric_reach_mm"]
    servo = spec["manufacturer_candidate"]["body_envelope_w_h_d_mm"]["value"]
    tray = geom["catch_tray_mm"]
    spacing = geom["dual_arm_base_spacing_mm"]
    handover = spacing["simulation_layouts"]["arms-handover-v1"][
        "base_center_spacing"
    ]
    co_carry = spacing["simulation_layouts"]["arms-co-carry-v1"][
        "base_center_spacing"
    ]
    station_requirement = spacing["physical_station_requirement"]

    scale = 3.1
    base_x, base_y = 115.0, 300.0
    p1 = (base_x + upper * scale, base_y - 55)
    p2 = (p1[0] + fore * scale, p1[1] + 30)
    tcp = (p2[0] + wrist * scale, p2[1] - 14)

    body = f"""
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="36" y="42">MD-Arm-T1 mechanical concept</text>
  <text class="small" x="36" y="65">DESIGN ONLY - servo mounting interface BLOCKED</text>
  <text class="small" x="36" y="85">Link sketch only. Shared motor geometry: shared-xl330-comparison.svg (same source as Microduck).</text>

  <line x1="45" y1="{base_y + 50}" x2="780" y2="{base_y + 50}"
        stroke="#59636e" stroke-width="4"/>
  <text class="label" x="48" y="{base_y + 74}">fixed tabletop plane</text>

  <rect class="part" x="{base_x - 31}" y="{base_y - 35}" width="62" height="85"/>
  <circle class="axis" cx="{base_x}" cy="{base_y}" r="8"/>
  <line class="link" x1="{base_x}" y1="{base_y}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}"/>
  <rect class="part" x="{p1[0] - 31:.1f}" y="{p1[1] - 40:.1f}" width="62" height="80"/>
  <circle class="axis" cx="{p1[0]:.1f}" cy="{p1[1]:.1f}" r="8"/>
  <line class="link" x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}"/>
  <rect class="part" x="{p2[0] - 31:.1f}" y="{p2[1] - 40:.1f}" width="62" height="80"/>
  <circle class="axis" cx="{p2[0]:.1f}" cy="{p2[1]:.1f}" r="8"/>
  <line class="link" x1="{p2[0]:.1f}" y1="{p2[1]:.1f}" x2="{tcp[0]:.1f}" y2="{tcp[1]:.1f}"/>
  <circle cx="{tcp[0]:.1f}" cy="{tcp[1]:.1f}" r="7" fill="#087f5b"/>
  <text class="label" x="{tcp[0] + 12:.1f}" y="{tcp[1] - 8:.1f}">TCP</text>

{dimension(base_x, base_y - 92, p1[0], p1[1] - 37, f"{upper:g} mm axis distance")}
{dimension(p1[0], p1[1] + 69, p2[0], p2[1] + 54, f"{fore:g} mm axis distance")}
{dimension(p2[0], p2[1] - 70, tcp[0], tcp[1] - 48, f"{wrist:g} mm to TCP")}

  <rect class="blocked" x="36" y="430" width="744" height="98" rx="4"/>
  <text class="label" x="52" y="458">BLOCKED FABRICATION INTERFACE</text>
  <text class="small" x="52" y="482">Servo rectangles are only the published {servo[0]:g} x {servo[1]:g} x {servo[2]:g} mm W x H x D envelope.</text>
  <text class="small" x="52" y="504">No mounting holes, horn pilot, bolt circle, bearing seats, or manufacturing tolerances are defined.</text>

  <text class="label" x="36" y="570">Straight geometric reach: {reach:g} mm</text>
  <text class="label" x="36" y="594">Coupled gripper opening target: 0-30 mm (mapping unverified)</text>
  <text class="label" x="36" y="618">Physical payload target: 20 g (UNVERIFIED)</text>

  <g transform="translate(520,550)">
    <rect x="0" y="0" width="{tray['width'] * 0.9:.1f}" height="{tray['depth'] * 0.45:.1f}"
          fill="#e7f5ff" stroke="#1971c2" stroke-width="2"/>
    <text class="small" x="8" y="22">soft catch tray</text>
    <text class="small" x="8" y="42">{tray['width']:g} x {tray['depth']:g} x {tray['wall_height']:g} mm concept</text>
  </g>

  <text class="small" x="36" y="690">Drawing is dimensioned for design review, not scale-controlled manufacturing.</text>

  <line x1="36" y1="718" x2="864" y2="718" stroke="#59636e" stroke-width="1"/>
  <text class="title" x="36" y="750">Dual-arm station plan-view inputs</text>
  <text class="small" x="36" y="772">Conceptual base envelopes only - no clamp, rail, lock, mounting holes, or tolerances.</text>

{station_plan_panel(36, 790, 258, "Handover simulation", handover, "SIMULATION MODEL INPUT", "arms-handover-v1")}
{station_plan_panel(321, 790, 258, "Historical candidate", spacing["nominal"], "HISTORICAL DESIGN INPUT", f"adjustment candidate {spacing['adjustable_minimum']:g}-{spacing['adjustable_maximum']:g} mm")}
{station_plan_panel(606, 790, 258, "Co-carry simulation", co_carry, "SIMULATION MODEL INPUT", "arms-co-carry-v1")}

  <rect class="blocked" x="36" y="942" width="828" height="70" rx="4"/>
  <text class="label" x="52" y="968">PHYSICAL STATION REQUIREMENT: adjust {station_requirement['minimum_adjustable_center_spacing']:g}-{station_requirement['maximum_adjustable_center_spacing']:g} mm, or redesign and revalidate.</text>
  <text class="small danger" x="52" y="992">SIMULATION LAYOUT ONLY - PHYSICAL FIT UNVERIFIED; fit claim: {station_requirement['fit_claim']}.</text>
"""
    return svg_document(900, 1040, body)


def generate_wiring_svg(netlist: dict) -> str:
    actuator_rows = []
    for index, name in enumerate(("J1", "J2", "J3", "J4", "J5", "G")):
        y = 158 + index * 70
        actuator_rows.append(
            f"""
  <rect class="box" x="650" y="{y}" width="215" height="52"/>
  <text class="label" x="664" y="{y + 20}">{name} XL330-M288-T</text>
  <text class="small" x="664" y="{y + 40}">1 GND | 2 VDD | 3 DATA</text>
  <rect class="box" x="535" y="{y + 8}" width="75" height="34"/>
  <text class="small" x="572.5" y="{y + 30}" text-anchor="middle">F_{name}</text>
  <line class="power" x1="485" y1="{y + 25}" x2="535" y2="{y + 25}"/>
  <line class="power" x1="610" y1="{y + 25}" x2="650" y2="{y + 25}"/>
  <line class="data" x1="390" y1="{y + 43}" x2="650" y2="{y + 43}"/>
  <line class="ground" x1="420" y1="{y + 50}" x2="650" y2="{y + 50}"/>
"""
        )

    body = f"""
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text class="title" x="32" y="40">MD-Arm-T1 per-arm power and TTL wiring</text>
  <text class="small" x="32" y="63">Duplicate as an electrically independent branch for the second arm.</text>

  <rect class="box" x="32" y="105" width="150" height="80"/>
  <text class="label" x="48" y="130">Independent PSU</text>
  <text class="small" x="48" y="152">regulated 5.0 V</text>
  <text class="small" x="48" y="172">10 A candidate, TBD</text>

  <rect class="box" x="225" y="116" width="80" height="50"/>
  <text class="small" x="265" y="146" text-anchor="middle">F_MAIN</text>
  <rect class="box" x="350" y="105" width="150" height="80"/>
  <text class="label danger" x="365" y="132">LATCHING E-STOP</text>
  <text class="small" x="365" y="154">DC-rated pole per arm</text>
  <text class="small" x="365" y="174">manual reset</text>
  <line class="power" x1="182" y1="128" x2="225" y2="128"/>
  <line class="power" x1="305" y1="128" x2="350" y2="128"/>
  <line class="power" x1="500" y1="128" x2="485" y2="128"/>
  <line class="power" x1="485" y1="128" x2="485" y2="533"/>

  <rect class="box" x="32" y="235" width="270" height="112"/>
  <text class="label" x="48" y="263">ROBOTIS U2D2 - TTL port</text>
  <text class="small" x="48" y="287">pin 1 GND</text>
  <text class="small danger" x="48" y="309">pin 2 N/C</text>
  <text class="small" x="48" y="331">pin 3 DATA</text>

  <line class="data" x1="302" y1="326" x2="390" y2="326"/>
  <line class="data" x1="390" y1="183" x2="390" y2="551"/>
  <line class="ground" x1="302" y1="282" x2="420" y2="282"/>
  <line class="ground" x1="420" y1="183" x2="420" y2="610"/>
  <line class="ground" x1="182" y1="168" x2="315" y2="168"/>
  <line class="ground" x1="315" y1="168" x2="315" y2="610"/>
  <line class="ground" x1="315" y1="610" x2="420" y2="610"/>
  <text class="small" x="394" y="574">TTL DATA multidrop</text>
  <text class="small" x="424" y="594">common ARM_GND reference</text>

{''.join(actuator_rows)}

  <rect class="blocked" x="32" y="635" width="828" height="96" rx="4"/>
  <text class="label" x="48" y="662">PROHIBITED CONNECTIONS</text>
  <text class="small" x="48" y="686">No original Microduck battery positive, body VDD, or body DATA. U2D2 does not power servos.</text>
  <text class="small" x="48" y="708">Fuse values, conductors, connectors, distribution board, and e-stop DC ratings remain selection gates.</text>

  <text class="small" x="32" y="768">Official pins: U2D2 TTL = 1 GND, 2 N/C, 3 DATA; XL330 TTL = 1 GND, 2 VDD, 3 DATA. Accessed {netlist['source_accessed']}.</text>
"""
    return svg_document(900, 800, body)


Triangle = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def box_triangles(
    origin: tuple[float, float, float],
    size: tuple[float, float, float],
) -> list[Triangle]:
    x, y, z = origin
    dx, dy, dz = size
    v = [
        (x, y, z),
        (x + dx, y, z),
        (x + dx, y + dy, z),
        (x, y + dy, z),
        (x, y, z + dz),
        (x + dx, y, z + dz),
        (x + dx, y + dy, z + dz),
        (x, y + dy, z + dz),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    return [(v[a], v[b], v[c]) for a, b, c in faces]


def triangle_normal(triangle: Triangle) -> tuple[float, float, float]:
    a, b, c = triangle
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    n = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    length = math.sqrt(sum(value * value for value in n))
    if length == 0:
        return (0.0, 0.0, 0.0)
    return tuple(value / length for value in n)


def ascii_stl(name: str, triangles: Iterable[Triangle]) -> str:
    lines = [f"solid {name}"]
    for triangle in triangles:
        normal = triangle_normal(triangle)
        lines.append(
            f"  facet normal {normal[0]:.8g} {normal[1]:.8g} {normal[2]:.8g}"
        )
        lines.append("    outer loop")
        for vertex in triangle:
            lines.append(
                f"      vertex {vertex[0]:.8g} {vertex[1]:.8g} {vertex[2]:.8g}"
            )
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def generate_link_blanks_stl(spec: dict) -> str:
    geom = spec["design_geometry"]
    width = 12.0
    thickness = 5.0
    gap = 8.0
    triangles: list[Triangle] = []
    y = 0.0
    for length in (
        geom["upper_arm_axis_distance_mm"],
        geom["forearm_axis_distance_mm"],
        geom["wrist_to_tcp_mm"],
    ):
        triangles.extend(box_triangles((0.0, y, 0.0), (length, width, thickness)))
        y += width + gap
    return ascii_stl("md_arm_t1_link_blanks_no_mounting_holes", triangles)


def generate_servo_envelope_stl(spec: dict) -> str:
    width, height, depth = spec["manufacturer_candidate"][
        "body_envelope_w_h_d_mm"
    ]["value"]
    triangles = box_triangles((0.0, 0.0, 0.0), (width, depth, height))
    return ascii_stl("xl330_published_body_envelope_not_mounting_model", triangles)


def generate_catch_tray_stl(spec: dict) -> str:
    tray = spec["design_geometry"]["catch_tray_mm"]
    width = tray["width"]
    depth = tray["depth"]
    base = tray["base_thickness"]
    wall_h = tray["wall_height"]
    wall = tray["wall_thickness"]
    boxes = [
        ((0.0, 0.0, 0.0), (width, depth, base)),
        ((0.0, 0.0, base), (width, wall, wall_h)),
        ((0.0, depth - wall, base), (width, wall, wall_h)),
        ((0.0, wall, base), (wall, depth - 2 * wall, wall_h)),
        ((width - wall, wall, base), (wall, depth - 2 * wall, wall_h)),
    ]
    triangles: list[Triangle] = []
    for origin, size in boxes:
        triangles.extend(box_triangles(origin, size))
    return ascii_stl("md_arm_t1_soft_catch_tray_concept", triangles)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(output_dir: Path) -> list[Path]:
    spec = load_json(SPEC_PATH)
    netlist = load_json(NETLIST_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "md-arm-t1-mechanical.svg": generate_mechanical_svg(spec),
        "md-arm-t1-wiring.svg": generate_wiring_svg(netlist),
        "md-arm-t1-link-blanks.stl": generate_link_blanks_stl(spec),
        "xl330-body-envelope.stl": generate_servo_envelope_stl(spec),
        "md-arm-t1-catch-tray.stl": generate_catch_tray_stl(spec),
    }
    paths = []
    for name, content in outputs.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)

    paths.extend(write_shared_outputs(output_dir))

    manifest = {
        "schema_version": "1.0",
        "design_id": spec["design_id"],
        "revision": spec["revision"],
        "generator": "scripts/generate.py",
        "generator_dependencies": ["Python standard library"],
        "openscad_required": False,
        "fabrication_release": False,
        "servo_mount_status": spec["interfaces"]["servo_mount"]["status"],
        "payload_verification": spec["payload"]["verification"],
        "station_layout_contract": {
            "historical_candidate_mm": {
                "nominal": spec["design_geometry"]["dual_arm_base_spacing_mm"][
                    "nominal"
                ],
                "adjustable_minimum": spec["design_geometry"][
                    "dual_arm_base_spacing_mm"
                ]["adjustable_minimum"],
                "adjustable_maximum": spec["design_geometry"][
                    "dual_arm_base_spacing_mm"
                ]["adjustable_maximum"],
            },
            "simulation_base_center_spacing_mm": {
                name: layout["base_center_spacing"]
                for name, layout in spec["design_geometry"][
                    "dual_arm_base_spacing_mm"
                ]["simulation_layouts"].items()
            },
            "required_physical_adjustment_mm": {
                "minimum": spec["design_geometry"]["dual_arm_base_spacing_mm"][
                    "physical_station_requirement"
                ]["minimum_adjustable_center_spacing"],
                "maximum": spec["design_geometry"]["dual_arm_base_spacing_mm"][
                    "physical_station_requirement"
                ]["maximum_adjustable_center_spacing"],
            },
            "fit_claim": spec["design_geometry"]["dual_arm_base_spacing_mm"][
                "physical_station_requirement"
            ]["fit_claim"],
            "verification": spec["design_geometry"]["dual_arm_base_spacing_mm"][
                "physical_station_requirement"
            ]["verification"],
        },
        "outputs": [
            {
                "path": path.name,
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(paths)
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths + [manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    outputs = write_outputs(args.output_dir)
    print(
        json.dumps(
            {
                "status": "generated",
                "output_dir": str(args.output_dir),
                "files": [path.name for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
