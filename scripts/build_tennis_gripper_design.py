"""Generate review-only CAD and a dimension diagram from the experiment spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_arm_experiments.tennis_return import SPEC, task_spec
from microduck_arm_v1c.config import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = task_spec()["wide_gripper_candidate"]
    offset = spec["pad_outward_offset_each_m"] * 1000
    density = spec["assumed_density_kg_m3"]
    primitives = []
    for side, sign in (("left", 1), ("right", -1)):
        pivot_y = sign * 12
        primitives.extend([
            (f"{side}_bridge", [spec["bridge_center_x_m"] * 1000, pivot_y + sign * offset / 2, spec["bridge_center_z_m"] * 1000], [value * 1000 for value in spec["bridge_full_size_m"]], "orange"),
            (f"{side}_stem", [spec["stem_center_x_m"] * 1000, pivot_y + sign * (offset + spec["stem_outward_offset_m"] * 1000), spec["bridge_center_z_m"] * 1000], [value * 1000 for value in spec["stem_full_size_m"]], "orange"),
            (f"{side}_pad", [spec["pad_center_x_m"] * 1000, pivot_y + sign * offset, 0], [24, 5, 8], "green"),
        ])
    scad = ['echo("CONCEPT ONLY: no shaft/horn holes, retention, tolerance or strength release");',
            'echo("Units mm. Neutral/open pose. Pads are contact references, not printed PA12.");']
    for name, center, size, color in primitives:
        scad.append(f'color("{color}") translate({json.dumps(center)}) cube({json.dumps(size)}, center=true);')
    (args.out / "wide-gripper-concept.scad").write_text("\n".join(scad) + "\n")
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="730" viewBox="0 0 1100 730">',
           '<rect width="1100" height="730" fill="#f5f7fa"/>',
           '<g font-family="Arial, sans-serif" fill="#162335">',
           '<text x="40" y="48" font-size="27" font-weight="bold">MicroDuck · Tennis wide-gripper candidate / 网球宽口夹爪</text>',
           '<text x="40" y="81" font-size="17" fill="#b54708">REVIEW ONLY / 仅设计审查：未提供轴孔、轮毂固定、装配公差或强度认证</text>',
           '<rect x="108" y="278" width="80" height="104" rx="4" fill="#8f9eae"/>',
           '<text x="136" y="335" font-size="18" fill="white">M5</text>',
           '<circle cx="340" cy="330" r="134" fill="#e9ef76" fill-opacity="0.65" stroke="#859e24" stroke-width="2"/>',
           '<text x="299" y="334" font-size="18">Ø67 mm</text>']
    for name, center, size, color in primitives:
        left = 120 + (center[0] - size[0] / 2) * 4
        top = 330 - (center[1] + size[1] / 2) * 4
        fill = "#3c9c73" if color == "green" else "#e69f35"
        svg.append(f'<rect x="{left}" y="{top}" width="{size[0]*4}" height="{size[1]*4}" fill="{fill}" stroke="#27394c"/>')
    svg.extend([
        '<circle cx="120" cy="282" r="5" fill="#162335"/><circle cx="120" cy="378" r="5" fill="#162335"/>',
        '<path d="M480 180H550 M480 480H550 M530 180V480" fill="none" stroke="#162335" stroke-width="2"/>',
        '<text x="545" y="326" font-size="23" font-weight="bold">75 mm</text>',
        '<text x="545" y="352" font-size="16">net opening</text>',
        '<text x="545" y="376" font-size="16">净开口</text>',
        '<text x="690" y="156" font-size="20" font-weight="bold">Unchanged / 保持不变</text>',
        '<text x="690" y="188" font-size="17">5 arm actuators / 5 个执行器</text>',
        '<text x="690" y="220" font-size="17">24 mm gear-axis spacing</text>',
        '<text x="690" y="252" font-size="17">Mirrored jaw travel: 0–0.32 rad</text>',
        '<text x="690" y="310" font-size="20" font-weight="bold">Changed / 新候选尺寸</text>',
        '<text x="690" y="342" font-size="17">Each pad offset: 28 mm outward</text>',
        '<text x="690" y="374" font-size="17">Pad center: x = 48 mm</text>',
        '<text x="690" y="406" font-size="17">TCP: x = 55 mm (was 30 mm)</text>',
        '<text x="690" y="438" font-size="17">Pads: 24 × 5 × 8 mm</text>',
        '<text x="40" y="559" font-size="18">Top view, open pose. Orange: structure; green: contact pads; yellow: real-sized ball.</text>',
        '<text x="40" y="592" font-size="18">俯视张开姿态。保留电机和轴距；向外移指、前移 TCP，避免网球撞到掌部和电机。</text>',
        '<text x="40" y="637" font-size="17" fill="#b54708">Geometry fit ≠ stable grasp ≠ ground pickup ≠ return-to-bin success ≠ hardware release.</text>',
        '<text x="40" y="674" font-size="17">制造前仍需轴孔/轮毂、保持件、公差、材料强度、负载扭矩、地面可达性与整机平衡验证。</text>',
        '</g></svg>'])
    (args.out / "wide-gripper-dimensions.svg").write_text("\n".join(svg) + "\n")
    volume = sum(
        size[0] * size[1] * size[2] / 1e9
        for name, center, size, color in primitives if color == "orange"
    )
    manifest = {
        "source_spec_sha256": sha256(SPEC), "generator_sha256": sha256(__file__),
        "structural_addition_mass_kg": volume * density,
        "structural_mass_status": "analytical_blank_volume_excludes_unresolved_horns_fasteners",
        "units": "mm", "files": {name: sha256(args.out / name) for name in ("wide-gripper-concept.scad", "wide-gripper-dimensions.svg")},
        "fabrication_release": False, "hardware_release": False,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
