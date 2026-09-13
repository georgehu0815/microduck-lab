"""Serially render and verify every WingPod tennis episode, including controls."""

import json
from pathlib import Path
import shutil

import imageio.v2 as imageio

from microduck_arm_design.wingpod_camera import WingPodCameraEnv
from scripts.replay_tennis_evidence import preflight_source, replay_episode, sha256, _completed_replay


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/microduck-arm-v1c/tennis-return/half-height-planned-v13"
OUTPUT = ROOT / "artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/task-cases"


def build():
    source = preflight_source(SOURCE)
    paths = [ROOT / "microduck_arm_design/wingpod.py", ROOT / "microduck_arm_design/wingpod_camera.py",
             ROOT / "scripts/replay_tennis_evidence.py", Path(__file__).resolve()]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in paths}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in source["ordered_directories"]:
        episode = source["episodes"][name]
        result = episode["result"]
        directory = OUTPUT / name
        context = {key: source[key] for key in ("evaluation_sha256", "snapshot_hashes")}
        context.update({key: episode[key] for key in ("result_sha256", "telemetry_sha256")})
        if not directory.exists():
            directory.mkdir()
            for filename in ("result.json", "telemetry.jsonl"):
                shutil.copy2(SOURCE / name / filename, directory / filename)
        job = {"output": str(OUTPUT), "directory_name": name, "source_context": context,
               "result": result, "telemetry": episode["telemetry"]}
        validation = _completed_replay(job, env_factory=WingPodCameraEnv)
        receipt = directory / "camera-receipt.json"
        if validation is not None:
            if not receipt.exists() or json.loads(receipt.read_text())["appearance_sources"] != hashes:
                raise ValueError(f"Existing replay has no matching camera appearance provenance: {name}")
        else:
            validation = replay_episode(directory, result, episode["telemetry"], context,
                                        env_factory=WingPodCameraEnv)
        if not validation["complete"] or not validation["zero_error"]:
            raise ValueError(f"Incomplete or discrepant replay: {name}")
        reader = imageio.get_reader(directory / "rollout.mp4")
        frames = 0
        try:
            fps = reader.get_meta_data()["fps"]
            for frame in reader:
                if frame.shape != (480, 640, 3):
                    raise ValueError(f"Unexpected frame size: {name}")
                frames += 1
        finally:
            reader.close()
        if frames != validation["frames_written"] or fps != 25:
            raise ValueError(f"Video count/rate mismatch: {name}")
        control = name in ("hold", "open_jaw")
        matched = result["success"] is (not control)
        record = {"id": name, "variant": "control" if control else result["variant"],
                  "seed": result["seed"], "kind": "negative_control" if control else "positive",
                  "taskSuccess": result["success"], "expectedOutcomeMatched": matched,
                  "video": f"cases/{name}.mp4", "receipt": f"cases/{name}.json",
                  "durationSeconds": frames / fps, "frames": frames, "zeroError": validation["zero_error"]}
        if not matched:
            raise ValueError(f"Unexpected task outcome: {name}")
        for path, expected in hashes.items():
            if sha256(ROOT / path) != expected:
                raise ValueError("Appearance/render sources changed during replay")
        receipt.write_text(json.dumps({"case": record, "appearance_sources": hashes, "replay": validation,
                                       "failure_reason": result["failure_reason"], "phase_events": result["phase_events"],
                                       "hardware_release": False, "vision_control": False, "new_training": False}, indent=2) + "\n")
        cases.append(record)
        print(f"{name}: outcome matched={matched}, {frames} frames fully decoded", flush=True)
    summary = {"positivePassed": sum(case["taskSuccess"] for case in cases if case["kind"] == "positive"),
               "positiveTotal": sum(case["kind"] == "positive" for case in cases),
               "controlsMatched": sum(case["expectedOutcomeMatched"] for case in cases if case["kind"] == "negative_control"),
               "controlsTotal": sum(case["kind"] == "negative_control" for case in cases)}
    if summary != {"positivePassed": 30, "positiveTotal": 30, "controlsMatched": 2, "controlsTotal": 2}:
        raise ValueError("Incomplete case matrix")
    (OUTPUT / "cases.json").write_text(json.dumps({"cases": cases, "summary": summary,
        "scope": "Tennis return, 3 size variants x 10 seeds plus 2 negative controls; not all standalone arm skills",
        "hardware_release": False, "vision_control": False, "new_training": False}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    build()
