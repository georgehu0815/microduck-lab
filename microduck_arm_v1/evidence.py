from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess

from .config import CASES, ROOT, sha256
from .validation import provenance, rollout, write_json


def require_current(record):
    if record.get("provenance") != provenance():
        raise ValueError("Stale or missing source provenance: rerun this evaluation")
    if record.get("hardware_ready") is not False or record.get("simulation_only") is not True:
        raise ValueError("Evaluation must explicitly remain simulation-only")
    if not isinstance(record.get("success"), bool):
        raise ValueError("Evaluation success must be an explicit boolean")


def negative_controls(output, seeds=(910000, 910001), max_steps=1500):
    output = Path(output)
    records = []
    for case in CASES[:4]:
        for controller in ("no_actuation", "open_gripper"):
            for seed in seeds:
                record = rollout(case, "fixture", seed, max_steps, controller)
                label = f"fixture-{case}-{controller}-seed-{seed}.json"
                write_json(output / label, record)
                records.append({key: value for key, value in record.items() if key != "trace"})
    summary = {
        "scope": "fixture_negative_controls_not_hardware_validation",
        "records": records,
        "provenance": provenance(),
        "interpretation": "Open gripper is not a negative for reach. Failed controls on cases with a failed teacher do not establish task validity.",
    }
    write_json(output / "negative-controls.json", summary)
    return summary


def audit_evaluation(evaluation_path, output):
    evaluation_path, output = Path(evaluation_path), Path(output)
    summary = json.loads(evaluation_path.read_text())
    if summary.get("provenance") != provenance():
        raise ValueError("Evaluation summary is not from current sources")
    if not summary.get("records"):
        raise ValueError("Empty evaluation is not evidence")
    videos, episodes = [], []
    for record in summary["records"]:
        require_current(record)
        label = f"{record['mode']}-{record['case']}-seed-{record['seed']}"
        trace_path = evaluation_path.parent / f"{label}.json"
        trace = json.loads(trace_path.read_text())
        require_current(trace)
        if {key: value for key, value in trace.items() if key != "trace"} != record:
            raise ValueError(f"Summary/trace mismatch: {label}")
        if len(trace["trace"]) != record["steps"] or record["steps"] < 1:
            raise ValueError(f"Incomplete trace: {label}")
        episodes.append({"path": str(trace_path), "sha256": sha256(trace_path), "success": record["success"]})
        if "video" not in record:
            continue
        path = Path(record["video"]["path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != record["video"]["sha256"]:
            raise ValueError(f"Video hash mismatch: {label}")
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
            "-of", "json", str(path),
        ], text=True))["streams"][0]
        expected_frames = (record["steps"] + 1) // 2 + int(record["steps"] % 2 == 0)
        if int(probe["nb_read_frames"]) != expected_frames:
            raise ValueError(f"Video omits recorded steps: {label}")
        if probe["codec_name"] != "h264" or probe["r_frame_rate"] != "25/1":
            raise ValueError(f"Unexpected video codec/rate: {label}")
        videos.append({"label": label, "controller": record["controller"], "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                       "success": record["success"], "failure_reason": record["failure_reason"], "probe": probe})
    result = {"scope": "artifact_integrity_not_task_or_hardware_success", "integrity_passed": True,
              "evaluation_sha256": sha256(evaluation_path), "episodes": episodes, "videos": videos,
              "video_sampling": "25fps from 50Hz control; terminal frame included; encoding may add up to one frame interval",
              "provenance": provenance()}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evidence-index.json", result)
    write_gallery(videos, output)
    return result


def write_gallery(videos, output):
    output = Path(output)
    cards = []
    for video in videos:
        link = html.escape(os.path.relpath(ROOT / video["path"], output), quote=True)
        title = html.escape(video["label"] + " · " + video["controller"])
        status = "PASS (simulation only)" if video["success"] else "FAIL: " + html.escape(str(video["failure_reason"]))
        cards.append(f'<article><h2>{title}</h2><p>{status}</p><video controls preload="metadata" src="{link}"></video></article>')
    gallery = '<!doctype html><html lang="en"><meta charset="utf-8"><title>MicroDuck v1-B evidence / 视频证据</title><style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1100px;margin:auto;padding:24px}article{margin:24px 0}video{width:100%;max-width:640px}h2{font-size:18px}</style><h1>MicroDuck v1-B · Video evidence / 视频证据</h1><p>SIMULATION ONLY. Fixture assistance is not mobile validation. Failures are preserved.<br>仅仿真；固定台架成功不等于移动操作成功。全部失败记录保留。</p>' + "".join(cards) + '</html>'
    (output / "videos.html").write_text(gallery)


def merge_galleries(indices, output):
    videos = []
    for index in indices:
        record = json.loads(Path(index).read_text())
        if record.get("provenance") != provenance() or record.get("integrity_passed") is not True:
            raise ValueError("Only current audited evidence can enter the combined gallery")
        for video in record["videos"]:
            if sha256(ROOT / video["path"]) != video["sha256"]:
                raise ValueError("Video changed since audit")
            videos.append(video)
    if not videos:
        raise ValueError("No audited videos supplied")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {"integrity_passed": True, "provenance": provenance(), "videos": videos, "count": len(videos), "hardware_release": False}
    write_json(output / "all-videos-index.json", result)
    write_gallery(videos, output)
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate controls and audit actual v1-B video evidence")
    parser.add_argument("command", choices=("controls", "audit", "gallery"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--evaluation")
    parser.add_argument("--indices", nargs="+")
    arguments = parser.parse_args()
    if arguments.command == "audit":
        if not arguments.evaluation:
            parser.error("--evaluation is required for audit")
        result = audit_evaluation(arguments.evaluation, arguments.out)
    elif arguments.command == "gallery":
        if not arguments.indices:
            parser.error("--indices is required for gallery")
        result = merge_galleries(arguments.indices, arguments.out)
    else:
        result = negative_controls(arguments.out)
    print(json.dumps({key: value for key, value in result.items() if key not in ("records", "episodes", "videos")}, indent=2))


if __name__ == "__main__":
    main()
