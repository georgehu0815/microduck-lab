"""Audit shared source geometry without modifying training or robot assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[3]
HARDWARE = ROOT / "hardware/md-arm-t1"
ROBOT_XML = (
    ROOT / "rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions_backlash.xml"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_triangles(path: Path) -> list:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("Truncated binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if count == 0 or len(data) != 84 + count * 50:
        raise ValueError("Invalid binary STL length")
    triangles = [
        [
            struct.unpack_from("<3f", data, 84 + index * 50 + 12 + vertex * 12)
            for vertex in range(3)
        ]
        for index in range(count)
    ]
    if not all(
        math.isfinite(value) for face in triangles for point in face for value in point
    ):
        raise ValueError("Nonfinite STL coordinates")
    return triangles


def audit_components() -> dict:
    tree = ET.parse(ROBOT_XML).getroot()
    meshdir = tree.find("compiler").get("meshdir", "")
    assets = {
        item.get("name", Path(item.get("file")).stem): item
        for item in tree.findall("./asset/mesh")
    }
    mesh = assets["xl330"]
    if mesh.get("scale", "1 1 1") != "1 1 1":
        raise ValueError("Source mesh scale changed: re-review shared coordinates")
    source = ROBOT_XML.parent / meshdir / mesh.get("file")
    faces = read_triangles(source)
    bounds = [
        [
            min(point[axis] for face in faces for point in face) * 1000,
            max(point[axis] for face in faces for point in face) * 1000,
        ]
        for axis in range(3)
    ]
    inventory = []
    motors = []
    for body in tree.findall(".//body"):
        for geom in body.findall("geom"):
            if geom.get("mesh") is None:
                continue
            item = {
                "body": body.get("name"),
                "mesh": geom.get("mesh"),
                "class": geom.get("class"),
                "pos": geom.get("pos", "0 0 0"),
                "quat": geom.get("quat", "1 0 0 0"),
            }
            inventory.append(item)
            if item["mesh"] == "xl330":
                motors.append(item)
    spec_path = HARDWARE / "spec/md-arm-t1.json"
    spec = json.loads(spec_path.read_text())
    manufacturer = spec["manufacturer_candidate"]
    part = source.with_suffix(".part")
    return {
        "schema_version": 1,
        "date": "2026-09-12",
        "source_xml": str(ROBOT_XML.relative_to(ROOT)),
        "source_xml_sha256": digest(ROBOT_XML),
        "spec_sha256": digest(spec_path),
        "microduck_exact_variant": None,
        "variant_status": "XL330 family only; XML and Onshape metadata do not identify gearbox SKU",
        "arm_candidate": manufacturer["model"],
        "fabrication_release": False,
        "manufacturer_body_w_h_d_mm": manufacturer["body_envelope_w_h_d_mm"]["value"],
        "mesh": {
            "source": str(source.relative_to(ROOT)),
            "sha256": digest(source),
            "triangle_count": len(faces),
            "unit_scale_to_mm": 1000,
            "bounds_xyz_mm": bounds,
            "bounds_size_xyz_mm": [high - low for low, high in bounds],
            "centering_translation_mm": [-(low + high) / 2 for low, high in bounds],
            "cad_orientation": "rotate +90 degrees about Z after centering; X=width,Y=depth,Z=height",
            "use_limit": "source visual reference, not precision mounting or validated collision geometry",
            "depth_difference_status": "29 mm mesh versus 26 mm published body; feature cause unverified; do not rescale",
        },
        "onshape_part": json.loads(part.read_text()),
        "onshape_part_sha256": digest(part),
        "motor_instances": motors,
        "mesh_inventory": inventory,
        "actuator_count": len(tree.find("actuator")),
        "backlash_joint_count": len(tree.findall(".//joint[@class='backlash']")),
        "chosen_actuator": tree.find(
            "./default/default[@class='chosen_actuator']/position"
        ).attrib,
        "dynamics_transfer": "blocked: fitted body gains/force/backlash are not arm or manufacturer ratings",
        "training_sources_modified": False,
    }


def comparison_svg(audit: dict) -> str:
    faces = read_triangles(ROOT / audit["mesh"]["source"])
    center = audit["mesh"]["centering_translation_mm"]
    projections = [(1, 2, 0), (0, 2, 1), (1, 0, 2)]
    definitions = []
    for index, (horizontal, vertical, depth) in enumerate(projections):
        polygons = []
        for face in sorted(
            faces, key=lambda triangle: sum(point[depth] for point in triangle)
        ):
            points = " ".join(
                f"{(point[horizontal] * 1000 + center[horizontal]) * 5:.3f},"
                f"{-(point[vertical] * 1000 + center[vertical]) * 5:.3f}"
                for point in face
            )
            polygons.append(
                f'<polygon points="{points}" fill="#718992" stroke="#304955" stroke-width=".12"/>'
            )
        definitions.append(f'<g id="view-{index}">{"".join(polygons)}</g>')
    uses = []
    labels = ["Front: mesh 20 x 34 mm", "Side: mesh 29 x 34 mm", "Top: mesh 20 x 29 mm"]
    for row, title in enumerate(
        ["Microduck source motor", "Arm shared reference: identical geometry"]
    ):
        baseline = 210 + row * 270
        uses.append(f'<text x="35" y="{baseline - 110}" class="heading">{title}</text>')
        for column, label in enumerate(labels):
            xpos = 160 + column * 295
            uses.append(
                f'<use href="#view-{column}" transform="translate({xpos} {baseline})"/>'
            )
            uses.append(
                f'<text x="{xpos}" y="{baseline + 110}" text-anchor="middle">{label}</text>'
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="940" height="755" viewBox="0 0 940 755">
<style>text{{font:16px sans-serif;fill:#16333c}}.heading{{font-size:21px;font-weight:bold}}</style>
<rect width="940" height="755" fill="white"/><defs>{"".join(definitions)}</defs>
<text x="35" y="38" class="heading">XL330 shared geometry / Microduck + MD-Arm-T1</text>
<text x="35" y="65">Same STL, same scale; design reference only, NOT fabrication released.</text>
{"".join(uses)}
<text x="35" y="639">Published BODY: W20 x H34 x D26 mm. Source VISUAL mesh: X29 x Y20 x Z34 mm.</text>
<text x="35" y="667">Do not shrink the 29 mm mesh to 26 mm. Mount holes / shaft datum / tolerances unverified.</text>
<text x="35" y="695">Body gearbox SKU unknown. Arm candidate: M288. Shared appearance != shared dynamics.</text>
<text x="35" y="726">Source SHA256: {audit["mesh"]["sha256"][:32]}...</text></svg>"""


def write_outputs(output: Path) -> list[Path]:
    audit = audit_components()
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / audit["mesh"]["source"]
    relative_mesh = Path(os.path.relpath(source, output)).as_posix()
    width, height, depth = audit["manufacturer_body_w_h_d_mm"]
    shift = audit["mesh"]["centering_translation_mm"]
    scad = f'''servo_w_mm = {width};
servo_h_mm = {height};
servo_d_mm = {depth};
module microduck_xl330_reference() {{
    rotate([0, 0, 90])
        translate({json.dumps(shift)})
            scale([1000, 1000, 1000])
                import("{relative_mesh}");
}}
'''
    xml = f'''<mujoco model="shared_xl330_visual_comparison">
  <compiler angle="radian"/>
  <asset><mesh name="shared_xl330" file="{escape(relative_mesh)}"/></asset>
  <worldbody>
    <light pos="0 0 0.3"/>
    <camera name="comparison" pos="0 -.24 .16" xyaxes="1 0 0 0 .5 .8660254"/>
    <body name="microduck_reference" pos="-.035 0 .04"><geom mesh="shared_xl330" type="mesh" contype="0" conaffinity="0" rgba=".4 .5 .55 1"/></body>
    <body name="arm_reference" pos=".035 0 .04"><geom mesh="shared_xl330" type="mesh" contype="0" conaffinity="0" rgba=".4 .5 .55 1"/></body>
  </worldbody>
</mujoco>'''
    contents = {
        "component-consistency.json": json.dumps(audit, indent=2) + "\n",
        "shared-xl330.scad": scad,
        "shared-xl330-comparison.svg": comparison_svg(audit),
        "shared-xl330-preview.xml": xml,
    }
    paths = []
    for name, content in contents.items():
        path = output / name
        path.write_text(content, encoding="utf-8")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HARDWARE / "generated")
    args = parser.parse_args()
    print(json.dumps([str(path) for path in write_outputs(args.output_dir)], indent=2))


if __name__ == "__main__":
    main()
