import json
from pathlib import Path
import subprocess

from .config import CASES, sha256
from .validation import rollout, write_json, provenance


def verify_video(path):
    path = Path(path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_read_frames,duration", "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    if len(streams) != 1 or int(streams[0]["nb_read_frames"]) < 1 or float(streams[0]["duration"]) <= 0:
        raise ValueError("Invalid or empty video")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, check=True)
    return {"path": str(path), "sha256": sha256(path), "decode_passed": True, **streams[0]}


def render_all(output):
    output = Path(output)
    records = []
    for candidate in ("A", "B"):
        for mode in ("fixture", "free"):
            for case in CASES:
                if mode == "fixture" and "walking" in case:
                    continue
                name = f"{candidate}-{mode}-{case}"
                path = output / f"{name}.mp4"
                result = rollout(case, mode, candidate, output=output / name, video=path)
                records.append({"case": case, "mode": mode, "candidate": candidate, "task_success": result["success"],
                                "failure_reason": result["failure_reason"], "video": verify_video(path)})
                print(name, result["success"], result["failure_reason"], flush=True)
    manifest = {"scope": "full_episode_simulation_videos_including_failures_not_hardware", "records": records,
                "all_videos_decodable": all(record["video"]["decode_passed"] for record in records),
                "all_tasks_passed": all(record["task_success"] for record in records), "source_hashes": provenance()}
    write_json(output / "video-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    render_all(arguments.out)
