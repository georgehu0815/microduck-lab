#!/usr/bin/env python3
"""Collect one deterministic basketball reward trace without retraining."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RLX_ROOT = ROOT / "rlx"
MICRODUCK_LOCAL_SRC = ROOT / "microduck_local" / "src"
sys.path.insert(0, str(RLX_ROOT))
sys.path.insert(0, str(MICRODUCK_LOCAL_SRC))

POLICY = ROOT / "rlx/artifacts/basketball-local-20260910/policy.onnx"
ENVIRONMENT_SOURCE = ROOT / "rlx/rlx/environments/basketball.py"
OUTPUT_DIR = ROOT / "docs/basketball-bridge-book/assets/raw"
CSV_PATH = OUTPUT_DIR / "basketball_reward_trace.csv"
JSON_PATH = OUTPUT_DIR / "basketball_reward_trace.json"

SETTINGS = {
    "actuator": "bam",
    "command": [0.15, 0.0, 0.0],
    "control_dt_s": 0.02,
    "curriculum": False,
    "domain_rand": False,
    "hold": 0.0,
    "max_episode_s": 10.0,
    "obs_noise": False,
    "pushes": False,
    "random_yaw": True,
    "requested_steps": 500,
    "seed": 101,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecurrentPolicy:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if [value.name for value in inputs] != ["obs", "h_in", "c_in"]:
            raise ValueError("basketball policy must use obs/h_in/c_in inputs")
        if [value.name for value in outputs] != ["actions", "h_out", "c_out"]:
            raise ValueError("basketball policy must use actions/h_out/c_out outputs")
        hidden_shape = inputs[1].shape
        hidden_size = int(hidden_shape[-1])
        self.hidden = np.zeros((1, 1, hidden_size), dtype=np.float32)
        self.cell = np.zeros_like(self.hidden)

    def act(self, observation: np.ndarray) -> np.ndarray:
        actions, self.hidden, self.cell = self.session.run(
            ["actions", "h_out", "c_out"],
            {
                "obs": np.asarray(observation, dtype=np.float32).reshape(1, 61),
                "h_in": self.hidden,
                "c_in": self.cell,
            },
        )
        return np.asarray(actions[0], dtype=np.float32)


def collect() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from rlx.environments.basketball import BasketballEnv

    environment = BasketballEnv(
        actuator=SETTINGS["actuator"],
        command=tuple(SETTINGS["command"]),
        curriculum=SETTINGS["curriculum"],
        domain_rand=SETTINGS["domain_rand"],
        hold=SETTINGS["hold"],
        max_episode_s=SETTINGS["max_episode_s"],
        obs_noise=SETTINGS["obs_noise"],
        pushes=SETTINGS["pushes"],
        random_yaw=SETTINGS["random_yaw"],
        seed=SETTINGS["seed"],
    )
    policy = RecurrentPolicy(POLICY)
    rows: list[dict[str, Any]] = []
    cumulative_reward = 0.0
    termination_reasons: list[str] = []
    try:
        observation, _ = environment.reset(seed=SETTINGS["seed"])
        for step in range(1, SETTINGS["requested_steps"] + 1):
            action = policy.act(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            reward = float(reward)
            cumulative_reward += reward
            reasons = info.get("termination_reasons", [])
            termination_reasons = (
                [reasons] if isinstance(reasons, str) else [str(value) for value in reasons]
            )
            rows.append(
                {
                    "step": step,
                    "elapsed_s": step * SETTINGS["control_dt_s"],
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "tilt_deg": float(info["tilt_deg"]),
                    "root_ball_offset_m": float(info["root_ball_offset_m"]),
                    "foot_ball_contacts": int(info["foot_ball_contacts"]),
                    "ball_speed_mps": float(info["ball_speed_mps"]),
                    "action_abs_mean": float(np.mean(np.abs(action))),
                    "action_abs_max": float(np.max(np.abs(action))),
                }
            )
            if terminated or truncated:
                break
    finally:
        environment.close()

    if not rows:
        raise RuntimeError("reward trace produced no samples")
    summary = {
        "schema_version": 1,
        "label": "NEW deterministic diagnostic, NOT historical training reward",
        "collected_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "settings": SETTINGS,
        "policy": str(POLICY.relative_to(ROOT)),
        "policy_sha256": sha256(POLICY),
        "environment_source": str(ENVIRONMENT_SOURCE.relative_to(ROOT)),
        "environment_source_sha256": sha256(ENVIRONMENT_SOURCE),
        "collector": str(Path(__file__).resolve().relative_to(ROOT)),
        "collector_sha256": sha256(Path(__file__).resolve()),
        "recorded_steps": len(rows),
        "elapsed_s": rows[-1]["elapsed_s"],
        "stopped_on_first_done": True,
        "automatic_resets": 0,
        "terminated": rows[-1]["terminated"],
        "truncated": rows[-1]["truncated"],
        "termination_reasons": termination_reasons,
        "reward_sum": cumulative_reward,
        "reward_mean": cumulative_reward / len(rows),
        "reward_min": min(row["reward"] for row in rows),
        "reward_max": max(row["reward"] for row in rows),
    }
    return rows, summary


def main() -> None:
    if not POLICY.is_file():
        raise FileNotFoundError(POLICY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary = collect()
    fieldnames = list(rows[0])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
