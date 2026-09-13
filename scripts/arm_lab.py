from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np

from rlx.environments.arm import ArmEnv, CASES, model_hash


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "rlx/runs/arm"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_run_file(path):
    return path.is_file() and path.resolve().is_relative_to(RUNS.resolve())


def read_run_record(path):
    if not safe_run_file(path) or path.stat().st_size > 16_000_000:
        return {}
    record = json.loads(path.read_text())
    return record if isinstance(record, dict) else {}


class ArmLab:
    def __init__(self):
        self.lock = threading.RLock()
        self.environment = ArmEnv(CASES[0])
        self.environment.reset(seed=1)

    def state(self):
        with self.lock:
            state = self.environment.state()
            state["terminated"] = bool(state["terminated"])
            state["truncated"] = bool(state["truncated"])
            return state

    def command(self, operation, payload):
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        with self.lock:
            if operation == "reset":
                if set(payload) - {"case_id", "seed"}:
                    raise ValueError("Unexpected reset fields")
                case_id = payload.get("case_id", self.environment.case_id)
                seed = payload.get("seed", 1)
                if case_id not in CASES or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
                    raise ValueError("Invalid case or seed")
                replacement = ArmEnv(case_id)
                replacement.reset(seed=seed)
                self.environment.close()
                self.environment = replacement
            elif operation == "step":
                if set(payload) != {"action"} or not isinstance(payload["action"], list):
                    raise ValueError("Step requires action array only")
                if any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in payload["action"]):
                    raise ValueError("Actions must be numbers")
                action = np.asarray(payload["action"], dtype=float)
                if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("Actions must be finite and normalized to [-1,1]")
                self.environment.controller = "manual_simulation"
                self.environment.step(action)
            elif operation == "teacher":
                ticks = payload.get("ticks", 1)
                if set(payload) - {"ticks"} or isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 50:
                    raise ValueError("Teacher ticks must be integer 1..50")
                self.environment.controller = "authored_ik_fsm_teacher_not_ppo"
                for _ in range(ticks):
                    if self.environment.terminated or self.environment.truncated:
                        break
                    self.environment.step(self.environment.teacher_action())
            else:
                raise ValueError("Unsupported operation; hardware and training are not exposed here")
            return self.state()


def run_catalog():
    items = []
    environment_source = ROOT / "rlx/rlx/environments/arm.py"
    pipeline_source = ROOT / "rlx/examples/ppo_microduck_arm.py"
    expected_sources = {"environment": file_hash(environment_source), "pipeline": file_hash(pipeline_source)}
    if not RUNS.exists():
        return {"runs": [], "hardware_enabled": False}
    paths = sorted(set(RUNS.rglob("evaluation.json")) | set(RUNS.rglob("teacher-evaluation.json")) | set(RUNS.rglob("bc-evaluation.json")))
    for path in paths[-128:]:
        try:
            report = read_run_record(path)
            if report.get("case_id") not in CASES:
                continue
            sources_match = report.get("source_hashes") == expected_sources
            model_match = report.get("model_sha256") == model_hash(report["case_id"])
            expected_controller = {"evaluation.json": "ppo_residual", "teacher-evaluation.json": "teacher", "bc-evaluation.json": "bc"}[path.name]
            context_match = report.get("controller") == expected_controller and report.get("simulation_only") is True and report.get("hardware_verified") is False
            checkpoint_match = True
            if expected_controller in ("ppo_residual", "bc"):
                checkpoint = path.parent / ("ppo-residual.zip" if expected_controller == "ppo_residual" else "bc.zip")
                checkpoint_match = safe_run_file(checkpoint) and report.get("checkpoint_sha256") == file_hash(checkpoint)
                metadata_path = path.parent / ("training.json" if expected_controller == "ppo_residual" else "bc-training.json")
                metadata = read_run_record(metadata_path)
                checkpoint_match = checkpoint_match and all(metadata.get(key) == expected for key, expected in {
                    "case_id": report["case_id"], "model_sha256": model_hash(report["case_id"]),
                    "source_hashes": expected_sources, "checkpoint_sha256": report.get("checkpoint_sha256"),
                }.items())
            metrics = {key: value for key, value in report.items() if key != "results"}
            metrics["sources_match"] = sources_match
            metrics["checkpoint_match"] = checkpoint_match
            metrics["model_match"] = model_match
            metrics["context_match"] = context_match
            item = {"case_id": report["case_id"], "run_id": path.parent.name + ":" + report.get("controller", "unknown"),
                    "path": str(path.relative_to(RUNS)), "controller": report.get("controller", "unknown"),
                    "passed": bool(report.get("passed") is True and sources_match and model_match and checkpoint_match and context_match), "metrics": metrics}
            video = path.parent / "rollout.mp4"
            receipt_path = path.parent / "render-receipt.json"
            if safe_run_file(video) and safe_run_file(receipt_path):
                receipt = read_run_record(receipt_path)
                receipt_matches = all(receipt.get(key) == expected for key, expected in {
                    "video_sha256": file_hash(video), "source_hashes": expected_sources,
                    "case_id": report["case_id"], "controller": report.get("controller"),
                    "model_sha256": model_hash(report["case_id"]),
                    "checkpoint_sha256": report.get("checkpoint_sha256") if report.get("controller") == "ppo_residual" else None,
                }.items())
                if receipt_matches and sources_match and model_match and checkpoint_match and context_match:
                    item["video_path"] = str(video.relative_to(RUNS))
            items.append(item)
        except (ValueError, OSError, KeyError, TypeError):
            continue
    return {"runs": items, "hardware_enabled": False}


class Handler(BaseHTTPRequestHandler):
    lab: ArmLab

    def json_response(self, status, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.json_response(200, {"ok": True, "service": "microduck-arm-simulation", "hardware_enabled": False, "cases": list(CASES)})
        elif parsed.path == "/state":
            self.json_response(200, self.lab.state())
        elif parsed.path == "/runs":
            self.json_response(200, run_catalog())
        elif parsed.path == "/artifact":
            self.artifact(parse_qs(parsed.query).get("path", [""])[0])
        else:
            self.json_response(404, {"error": "Unknown endpoint"})

    def artifact(self, relative):
        path = (RUNS / relative).resolve()
        allowed = {".mp4", ".png", ".jpg", ".json", ".csv"}
        if not relative or not path.is_relative_to(RUNS.resolve()) or not path.is_file() or path.suffix not in allowed:
            self.json_response(404, {"error": "Artifact not found"})
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        requested = self.headers.get("Range")
        if requested:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
            if not match:
                self.json_response(416, {"error": "Unsupported byte range"})
                return
            start = int(match.group(1))
            end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
            if start > end or start >= size:
                self.json_response(416, {"error": "Byte range outside artifact"})
                return
        self.send_response(206 if requested else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if requested:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            while remaining:
                data = handle.read(min(65536, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def do_POST(self):
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json_response(415, {"error": "JSON content type required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Body limit is 16384 bytes")
            payload = json.loads(self.rfile.read(length), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))
            result = self.lab.command(urlparse(self.path).path.lstrip("/"), payload)
        except (ValueError, TypeError, RuntimeError) as error:
            self.json_response(400, {"error": str(error), "hardware_enabled": False})
            return
        self.json_response(200, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8812)
    args = parser.parse_args()
    Handler.lab = ArmLab()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Arm simulation http://127.0.0.1:{args.port}; hardware LOCKED", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        Handler.lab.environment.close()


if __name__ == "__main__":
    main()
