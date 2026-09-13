"""Render the independent WingPod Camera v2 appearance study and MP4."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microduck_arm_design.wingpod import WingPodOverlay, physics_fingerprint
from microduck_arm_design.wingpod_camera import (
    BODY, WingPodCameraEnv, camera_spec, render_optical_view,
)


OUTPUT = Path("artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2")
FONT = "/System/Library/Fonts/Avenir Next.ttc"


def product_frame(env, renderer, azimuth=130, distance=.48, closeup=False, original=False):
    model, data = env.robot.model, env.robot.data
    camera = mujoco.MjvCamera()
    if closeup:
        camera.lookat[:] = data.xpos[model.body(BODY).id] + [.012, 0, -.018]
    else:
        camera.lookat[:] = data.xpos[model.body("trunk_base").id] + [.018, 0, .005]
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, -15 if closeup else -25
    renderer.update_scene(data, camera=camera)
    for index in range(renderer.scene.ngeom):
        geom = renderer.scene.geoms[index]
        if geom.objtype == mujoco.mjtObj.mjOBJ_SITE:
            geom.rgba[3] = 0
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0:
            name = model.geom(geom.objid).name
            if name and any(word in name for word in ("bin_", "object", "ball", "goal")):
                geom.rgba[3] = 0
    overlay = WingPodOverlay(model) if original else env.overlay
    overlay.update(renderer.scene, data)
    return renderer.render().copy()


def caption(frame, title, subtitle):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 960, 76), fill="#253337")
    draw.text((24, 10), title, font=ImageFont.truetype(FONT, 27), fill="#FFF9E9")
    draw.text((24, 46), subtitle, font=ImageFont.truetype(FONT, 17), fill="#D4DFDD")
    draw.rectangle((0, 680, 960, 720), fill="#253337")
    draw.text((24, 690), "SIMULATED DESIGN PREVIEW  |  Camera hardware and vision policy not validated",
              font=ImageFont.truetype(FONT, 17), fill="#F5C575")
    return np.asarray(image)


def build(output, preview_only=False):
    output.mkdir(parents=True, exist_ok=True)
    env = WingPodCameraEnv(gripper="wide_candidate", release_mode="half_height")
    env.reset(0)
    model, data = env.robot.model, env.robot.data
    before = physics_fingerprint(model)
    state = data.qpos.copy()
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        for name, options in (
            ("hero", {}), ("reference-angle", {"azimuth": 130}),
            ("camera-eyes-detail", {"distance": .16, "closeup": True}),
            ("before-camera", {"original": True}),
        ):
            Image.fromarray(product_frame(env, renderer, **options)).save(output / f"{name}.png")
        for eye in ("left", "right"):
            Image.fromarray(render_optical_view(renderer, model, data, eye)).save(output / f"{eye}-eye-rgb.png")
        if not preview_only:
            video = output / "wingpod-camera-eyes.mp4"
            with imageio.get_writer(video, fps=25, codec="libx264", quality=8,
                                    macro_block_size=16, pixelformat="yuv420p") as writer:
                for index in range(500):
                    if index < 100:
                        frame = product_frame(env, renderer, azimuth=115 + index * .2)
                        title, subtitle = "WINGPOD CAMERA v2 / HELLO, WORLD", "Twin camera eyes on the fixed chest. Porcelain, graphite and honey."
                    elif index < 250:
                        frame = product_frame(env, renderer, azimuth=135 + (index - 100) * 2.4)
                        title, subtitle = "ONE WING. TWO CAMERA EYES.", "360-degree appearance review / existing 15 actuators retained"
                    elif index < 425:
                        frame = product_frame(env, renderer, azimuth=115 + (index - 250) * .1,
                                              distance=.16, closeup=True)
                        title, subtitle = "A FRIENDLY LOOK / FIXED CHEST MOUNT", "28 x 17 x 5 mm proposed housing / 13 mm optical-center spacing"
                    else:
                        frame = product_frame(env, renderer, azimuth=130)
                        title, subtitle = "READY FOR THE NEXT DESIGN REVIEW", "Render-only camera module / no new training or hardware performance claim"
                    writer.append_data(caption(frame, title, subtitle))
            reader = imageio.get_reader(video)
            decoded = 0
            try:
                metadata = reader.get_meta_data()
                for frame in reader:
                    assert frame.shape == (720, 960, 3)
                    decoded += 1
            finally:
                reader.close()
            assert decoded == 500
            (output / "video-verification.json").write_text(json.dumps({
                "file": video.name, "decoded_frames": decoded, "fps": metadata["fps"],
                "duration_seconds": decoded / metadata["fps"], "resolution": [960, 720],
                "type": "static_pose_camera_orbit_and_detail", "task_evaluation": False,
            }, indent=2) + "\n")
        assert before == physics_fingerprint(model)
        np.testing.assert_array_equal(state, data.qpos)
        (output / "camera-spec.json").write_text(json.dumps(camera_spec(), indent=2) + "\n")
        (output / "appearance-validation.json").write_text(json.dumps({
            "physics_unchanged_during_render": True, "physics_sha256": before,
            "qpos_unchanged": True, "actuators": model.nu, "hardware_release": False,
            "new_task_evaluation": False,
        }, indent=2) + "\n")
    finally:
        renderer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    build(args.output, args.preview_only)
