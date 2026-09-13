"""Render and source-verify the soft camera revision without replacing history."""

import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from microduck_arm_design.wingpod import physics_fingerprint
from microduck_arm_design.wingpod_camera_soft import WingPodSoftCameraEnv, soft_camera_spec
from scripts.build_wingpod_camera import FONT, product_frame
from scripts.replay_tennis_evidence import preflight_source, replay_episode, sha256


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2-soft"
SOURCE = ROOT / "artifacts/microduck-arm-v1c/tennis-return/half-height-planned-v13"


def source_hashes():
    paths = ["microduck_arm_design/wingpod.py", "microduck_arm_design/wingpod_camera.py",
             "microduck_arm_design/wingpod_camera_soft.py", "scripts/build_wingpod_camera.py",
             "scripts/build_wingpod_camera_soft.py", "scripts/replay_tennis_evidence.py"]
    return {path: sha256(ROOT / path) for path in paths}


def verify_video(path, shape):
    frames = 0
    with imageio.get_reader(path) as reader:
        metadata = reader.get_meta_data()
        for frame in reader:
            if frame.shape != shape:
                raise ValueError("Unexpected video resolution")
            frames += 1
    if metadata["fps"] != 25 or frames == 0:
        raise ValueError("Unexpected frame rate or empty video")
    return {"file": path.name, "sha256": sha256(path), "decoded_frames": frames,
            "fps": 25, "duration_seconds": frames / 25,
            "resolution": [shape[1], shape[0]]}


def appearance(output, preview_only):
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    env = WingPodSoftCameraEnv(gripper="wide_candidate", release_mode="half_height")
    env.reset(0)
    model, data = env.robot.model, env.robot.data
    before, state = physics_fingerprint(model), data.qpos.copy()
    renderer = mujoco.Renderer(model, height=720, width=960)
    try:
        for name, options in (("hero", {}), ("camera-eyes-detail", {"distance": .16, "closeup": True})):
            Image.fromarray(product_frame(env, renderer, **options)).save(output / f"{name}.png")
        if not preview_only:
            video = output / "wingpod-camera-eyes.mp4"
            with imageio.get_writer(video, fps=25, codec="libx264", quality=8,
                                    macro_block_size=16, pixelformat="yuv420p") as writer:
                for index in range(500):
                    closeup = 150 <= index < 400
                    frame = product_frame(env, renderer, azimuth=120 + 12 * np.sin(index / 90),
                                          distance=.16 if closeup else .48, closeup=closeup)
                    image = Image.fromarray(frame)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 960, 76), fill="#FFF8E7")
                    draw.text((24, 10), "WINGPOD CAMERA v2 / SOFT & SWEET",
                              font=ImageFont.truetype(FONT, 27), fill="#384B49")
                    draw.text((24, 46), "Cream face / sage lens rims / peach cheeks / honey beak",
                              font=ImageFont.truetype(FONT, 17), fill="#52645E")
                    draw.rectangle((0, 680, 960, 720), fill="#FFF8E7")
                    draw.text((24, 690), "RENDER-ONLY DESIGN  |  No new hardware or vision-policy validation",
                              font=ImageFont.truetype(FONT, 17), fill="#52645E")
                    writer.append_data(np.asarray(image))
            report = verify_video(video, (720, 960, 3))
            if report["decoded_frames"] != 500:
                raise ValueError("Incomplete appearance video")
            report.update({"design": "WingPod Camera v2 Soft", "task_evaluation": False,
                           "appearance_sources": hashes})
            (output / "video-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        if before != physics_fingerprint(model) or hashes != source_hashes():
            raise ValueError("Model or source changed during rendering")
        np.testing.assert_array_equal(state, data.qpos)
        (output / "camera-spec.json").write_text(json.dumps(soft_camera_spec(), indent=2) + "\n")
        (output / "appearance-validation.json").write_text(json.dumps({
            "physics_unchanged_during_render": True, "physics_sha256": before,
            "qpos_unchanged": True, "actuators": model.nu, "hardware_release": False,
            "appearance_sources": hashes,
        }, indent=2) + "\n")
    finally:
        renderer.close()
        env.close()


def task(output):
    source = preflight_source(SOURCE)
    episode = source["episodes"]["nominal-0"]
    result = episode["result"]
    if not result["success"]:
        raise ValueError("Source nominal episode failed")
    hashes = source_hashes()
    directory = output / "tennis-return-nominal-0"
    directory.mkdir(parents=True, exist_ok=False)
    for filename in ("result.json", "telemetry.jsonl"):
        shutil.copy2(SOURCE / "nominal-0" / filename, directory / filename)
    context = {key: source[key] for key in ("evaluation_sha256", "snapshot_hashes")}
    context.update({key: episode[key] for key in ("result_sha256", "telemetry_sha256")})
    replay = replay_episode(directory, result, episode["telemetry"], context,
                            env_factory=WingPodSoftCameraEnv)
    if not replay["complete"] or not replay["zero_error"]:
        raise ValueError("Soft replay differs from source")
    video = verify_video(directory / "rollout.mp4", (480, 640, 3))
    if video["decoded_frames"] != replay["frames_written"] or hashes != source_hashes():
        raise ValueError("Frame count or source hashes changed")
    board = Image.new("RGB", (1280, 1530), "#FFF8E7")
    draw = ImageDraw.Draw(board)
    with imageio.get_reader(directory / "rollout.mp4") as reader:
        for position, index in enumerate((0, 200, 400, 750, 1125, video["decoded_frames"] - 1)):
            left, top = position % 2 * 640, position // 2 * 510
            board.paste(Image.fromarray(reader.get_data(index)), (left, top))
            draw.text((left + 12, top + 486), f"Soft revision / {index / 25:.2f}s", fill="#384B49")
    board.save(directory / "sequence-contact-sheet.jpg", quality=95)
    report = {"design": "WingPod Camera v2 Soft", "success": result["success"],
              "source_episode": "nominal-0", "source_directory": str(SOURCE.relative_to(ROOT)),
              "phase_events": result["phase_events"], "replay": replay, "video": video,
              "appearance_and_renderer_sha256": hashes, "new_training": False,
              "vision_control": False, "hardware_release": False,
              "scope": "One soft-style exact action replay; 32-case library retains original graphite styling.",
              "release_note": "Low supported placement; not a demonstrated half-bin airborne drop."}
    (directory / "camera-task-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"success": report["success"], "zero_error": replay["zero_error"],
                      "video": video}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    appearance(args.output, args.preview_only)
    if not args.preview_only:
        task(args.output)
