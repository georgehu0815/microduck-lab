"""Build review-only WingPod artwork, CAD envelope and source-bound replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microduck_arm_design.wingpod import (
    DETAILS, PODS, WingPodEnv, WingPodOverlay, appearance_spec, apply_palette,
    physics_fingerprint,
)
from microduck_arm_experiments.tennis_return import TennisReturnEnv
from scripts.replay_tennis_evidence import preflight_source, replay_episode, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/microduck-arm-v1c/tennis-return/half-height-planned-v13"
OUTPUT = ROOT / "artifacts/microduck-arm-v1c/appearance/wingpod-v1"
FONT = "/System/Library/Fonts/Avenir Next.ttc"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render_product(environment, renderer, azimuth, styled):
    robot = environment.robot
    camera = mujoco.MjvCamera()
    camera.lookat[:] = robot.data.xpos[robot.model.body("trunk_base").id] + [.018, 0, .005]
    camera.distance, camera.azimuth, camera.elevation = .48, azimuth, -25
    renderer.update_scene(robot.data, camera=camera)
    for index in range(renderer.scene.ngeom):
        geom = renderer.scene.geoms[index]
        if geom.objtype == mujoco.mjtObj.mjOBJ_SITE:
            geom.rgba[3] = 0
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
            name = robot.model.geom(geom.objid).name
            if name and any(word in name for word in ("bin_", "object", "ball", "goal")):
                geom.rgba[3] = 0
    if styled:
        WingPodOverlay(robot.model).update(renderer.scene, robot.data)
    return renderer.render().copy()


def build_cad(output):
    lines = [
        '// REVIEW ENVELOPES ONLY: no holes, joints, clearance or hardware release.',
        '// Units mm; each part is in its named frozen body local frame.',
        '$fn=32;', 'wall=1.2;',
        'module rounded_box(half,r) {',
        '  hull() for (xx=[-1,1], yy=[-1,1], zz=[-1,1])',
        '    translate([xx*(half[0]-r),yy*(half[1]-r),zz*(half[2]-r)]) sphere(r=r);',
        '}',
    ]
    dimensions = ["part,body,cx_mm,cy_mm,cz_mm,width_mm,depth_mm,height_mm,fillet_mm,status"]
    for index, pod in enumerate(PODS):
        half = [round(value * 1000, 4) for value in pod.half_size]
        center = [round(value * 1000, 4) for value in pod.center]
        radius = pod.radius * 1000
        lines.extend([
            f'module {pod.name}() {{',
            f'  translate({center}) difference() {{',
            f'    rounded_box({half}, {radius});',
            f'    rounded_box([for (value={half}) value-wall], max(.2,{radius}-wall));',
            '  }', '}',
            f'translate([{index % 5 * 65},{index // 5 * 65},0]) {pod.name}();',
        ])
        values = [pod.name, pod.body, *center, *[2 * value for value in half], radius, "UNVERIFIED_ENVELOPE"]
        dimensions.append(",".join(str(value) for value in values))
    lines.append('// Decorative details share the exact render specification; solids, not manufactured parts.')
    for detail in DETAILS:
        index = next(index for index, pod in enumerate(PODS) if pod.body == detail.body)
        center = [round(value * 1000, 4) for value in detail.center]
        size = [round(value * 1000, 4) for value in detail.size]
        shape = f'scale({size}) sphere(r=1);' if detail.kind == "ellipsoid" else f'cylinder(r={size[0]},h={2 * size[1]},center=true);'
        lines.append(f'color({list(detail.color[:3])}) translate([{index % 5 * 65},{index // 5 * 65},0]) translate({center}) rotate([{detail.rotate_x_deg},0,0]) {shape}')
    (output / "wingpod-review-shells.scad").write_text("\n".join(lines) + "\n")
    (output / "shell-dimensions.csv").write_text("\n".join(dimensions) + "\n")
    write_json(output / "appearance-spec.json", appearance_spec())


def build_artwork(output, turntable):
    environment = TennisReturnEnv(gripper="wide_candidate", release_mode="half_height")
    environment.reset(0)
    model = environment.robot.model
    before = physics_fingerprint(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        original = render_product(environment, renderer, 130, False)
        Image.fromarray(original).save(output / "before.png")
        apply_palette(model)
        views = {}
        for name, angle in (("hero", 130), ("side", 90), ("front", 0), ("rear", 210)):
            frame = render_product(environment, renderer, angle, True)
            Image.fromarray(frame).save(output / f"{name}.png")
            views[name] = frame
        board = Image.new("RGB", (2000, 1050), "#F4F0E8")
        draw = ImageDraw.Draw(board)
        heading = ImageFont.truetype(FONT, 62)
        label = ImageFont.truetype(FONT, 25)
        draw.text((55, 25), "MICRODUCK  /  WINGPOD", fill="#253337", font=heading)
        draw.text((58, 105), "A softer wing. The same movement.", fill="#616963", font=label)
        board.paste(Image.fromarray(original), (30, 185))
        board.paste(Image.fromarray(views["hero"]), (1010, 185))
        draw.text((55, 155), "01   EXISTING ENGINEERING MODEL", fill="#616963", font=label)
        draw.text((1035, 155), "02   PORCELAIN + HONEY / APPEARANCE STUDY", fill="#616963", font=label)
        draw.text((55, 935), "55 / 50 mm links   |   75 mm open grip   |   15 actuators unchanged", fill="#253337", font=label)
        draw.text((55, 980), "Render-only shells. Physical mass, clearance, cooling and fabrication are NOT verified.", fill="#8A5027", font=label)
        board.save(output / "before-after.jpg", quality=95)
        board.save(output / "before-after.pdf", resolution=150)
        if turntable:
            with imageio.get_writer(output / "turntable.mp4", fps=25, codec="libx264", macro_block_size=16) as writer:
                for index in range(200):
                    frame = Image.fromarray(render_product(environment, renderer, 130 + index * 1.8, True))
                    caption = ImageDraw.Draw(frame)
                    caption.rectangle((0, 0, 960, 64), fill="#253337")
                    caption.text((20, 10), "WINGPOD / VISUAL STUDY - NOT HARDWARE VALIDATION", fill="white", font=label)
                    writer.append_data(np.asarray(frame))
        after = physics_fingerprint(model)
        if after != before:
            raise ValueError("Product rendering changed physics")
        write_json(output / "appearance-validation.json", {
            "physical_model_unchanged": True, "before_sha256": before, "after_sha256": after,
            "actuators": model.nu, "source_evaluation_sha256": sha256(SOURCE / "evaluation.json"),
            "hardware_release": False, "physical_shell_clearance_verified": False,
        })
    finally:
        renderer.close()
        environment.close()


def replay_all(output, video_cases):
    source = preflight_source(SOURCE)
    appearance_sources = {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / "microduck_arm_design/wingpod.py", Path(__file__).resolve(),
    )}
    validations = {}
    for name in source["ordered_directories"]:
        episode = source["episodes"][name]
        directory = output / "replays" / name
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(SOURCE / name / "telemetry.jsonl", directory / "telemetry.jsonl")
        shutil.copy2(SOURCE / name / "result.json", directory / "result.json")
        context = {key: source[key] for key in ("evaluation_sha256", "snapshot_hashes")}
        context.update({key: episode[key] for key in ("result_sha256", "telemetry_sha256")})
        validations[name] = replay_episode(directory, episode["result"], episode["telemetry"], context,
                                           render=name in video_cases, env_factory=WingPodEnv)
        print(f"{name}: exact replay={validations[name]['zero_error']}", flush=True)
    if any(sha256(ROOT / path) != expected for path, expected in appearance_sources.items()):
        raise ValueError("Appearance sources changed during replay")
    write_json(output / "replay-summary.json", {
        "type": "appearance_action_replay", "source_evaluation_sha256": source["evaluation_sha256"],
        "appearance_sources": appearance_sources,
        "positive_successes": sum(source["episodes"][name]["result"]["success"] for name in validations),
        "negative_controls": 2, "all_complete": all(value["complete"] for value in validations.values()),
        "all_zero_error": all(value["zero_error"] for value in validations.values()),
        "steps": sum(value["steps_replayed"] for value in validations.values()),
        "physics_substeps": sum(value["substep_monitor_calls"] for value in validations.values()),
        "new_ppo_training": False, "hardware_release": False, "episodes": validations,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artwork", action="store_true")
    parser.add_argument("--turntable", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--video-cases", nargs="*", default=["nominal-0", "small-5", "large-6"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    build_cad(args.output)
    if args.artwork:
        build_artwork(args.output, args.turntable)
    if args.replay:
        replay_all(args.output, set(args.video_cases))


if __name__ == "__main__":
    main()
