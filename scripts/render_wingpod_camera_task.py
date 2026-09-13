"""Replay a source-bound tennis return with WingPod Camera v2 visuals."""

import argparse
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from microduck_arm_design.wingpod_camera import WingPodCameraEnv
from scripts.replay_tennis_evidence import preflight_source, replay_episode, sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/microduck-arm-v1c/tennis-return/half-height-planned-v13"
OUTPUT = ROOT / "artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/tennis-return-nominal-0"


def build(output):
    source = preflight_source(SOURCE)
    episode = source["episodes"]["nominal-0"]
    result = episode["result"]
    if not result["success"]:
        raise ValueError("Source episode did not pass")
    tracked = [
        ROOT / "microduck_arm_design/wingpod.py",
        ROOT / "microduck_arm_design/wingpod_camera.py",
        ROOT / "scripts/replay_tennis_evidence.py",
        Path(__file__).resolve(),
    ]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in tracked}
    output.mkdir(parents=True, exist_ok=False)
    for filename in ("result.json", "telemetry.jsonl"):
        shutil.copy2(SOURCE / "nominal-0" / filename, output / filename)
    context = {key: source[key] for key in ("evaluation_sha256", "snapshot_hashes")}
    context.update({key: episode[key] for key in ("result_sha256", "telemetry_sha256")})
    validation = replay_episode(output, result, episode["telemetry"], context,
                                env_factory=WingPodCameraEnv)
    if not validation["complete"] or not validation["zero_error"]:
        raise ValueError("Replay did not exactly reproduce the recorded outcome")
    video = output / "rollout.mp4"
    reader = imageio.get_reader(video)
    samples = {}
    decoded = 0
    try:
        metadata = reader.get_meta_data()
        sample_indices = {0, 200, 400, 750, 1125, validation["frames_written"] - 1}
        for index, frame in enumerate(reader):
            if frame.shape != (480, 640, 3):
                raise ValueError(f"Unexpected frame shape: {frame.shape}")
            if index in sample_indices:
                samples[index] = Image.fromarray(frame)
            decoded += 1
    finally:
        reader.close()
    if decoded != validation["frames_written"] or metadata["fps"] != 25:
        raise ValueError("Video frame count or frame rate mismatch")
    board = Image.new("RGB", (1280, 3 * 510), "#253337")
    draw = ImageDraw.Draw(board)
    for position, (index, image) in enumerate(samples.items()):
        left, top = (position % 2) * 640, (position // 2) * 510
        board.paste(image, (left, top))
        draw.text((left + 12, top + 486), f"Video time {index / 25:.2f}s", fill="white")
    board.save(output / "sequence-contact-sheet.jpg", quality=95)
    for path, expected in hashes.items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Source changed during rendering: {path}")
    report = {
        "design": "WingPod Camera v2", "task": "ground tennis-ball pickup and bin return",
        "success": result["success"], "phase_events": result["phase_events"],
        "source_directory": str(SOURCE.relative_to(ROOT)), "source_episode": "nominal-0",
        "appearance_and_renderer_sha256": hashes, "replay": validation,
        "video": {"file": video.name, "sha256": sha256(video), "decoded_frames": decoded,
                  "fps": metadata["fps"], "duration_s": decoded / metadata["fps"],
                  "resolution": [640, 480]},
        "new_training": False, "vision_control": False, "hardware_release": False,
        "scope": "Single simulated action replay; no new all-case evaluation. Camera styling only.",
        "release_note": "Low supported placement, not a demonstrated half-bin airborne drop.",
    }
    (output / "camera-task-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"success": report["success"], "video": str(video),
                      "frames": decoded, "zero_error": validation["zero_error"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output)
