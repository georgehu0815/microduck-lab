from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
import torch

from .env import IntegratedArmEnv
from .learning import _checkpoint_metadata, _validate_provenance, load_bc
from .validation import provenance, write_json
from .config import sha256


def render_checkpoint(checkpoint, kind, output, seed=201, max_steps=250):
    checkpoint, output = Path(checkpoint), Path(output)
    metadata = _checkpoint_metadata(checkpoint)
    _validate_provenance(metadata["provenance"])
    if kind == "bc":
        policy = load_bc(checkpoint)
        controller = "behavior_cloning"
    elif kind == "ppo":
        policy = PPO.load(checkpoint, device="cpu")
        controller = "teacher_plus_residual_ppo"
        if metadata.get("learning_wrapper", {}).get("kind") != "teacher_residual":
            raise ValueError("This renderer requires the documented teacher-residual PPO checkpoint")
    else:
        raise ValueError("Expected bc or ppo")
    output.mkdir(parents=True, exist_ok=True)
    label = f"fixture-reach-seed-{seed}"
    video = output / f"{label}.mp4"
    env = IntegratedArmEnv(case="reach", mode="fixture", max_steps=max_steps)
    observation, info = env.reset(seed=seed)
    trace, reward_sum = [], 0.0
    try:
        with imageio.get_writer(str(video), fps=25, codec="libx264", quality=7) as writer:
            for step in range(max_steps):
                if kind == "bc":
                    with torch.no_grad():
                        action = policy(torch.from_numpy(observation[None])).numpy()[0]
                else:
                    residual, _ = policy.predict(observation, deterministic=True)
                    scale = np.asarray(metadata["learning_wrapper"]["residual_normalized_scale"], dtype=np.float32)
                    action = np.clip(env.teacher_action() + scale * residual, -1, 1)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                trace.append({"step": step, "reward": reward, "action": action.tolist(), "diagnostics": info["diagnostics"], "failure_reason": info["failure_reason"]})
                if step % 2 == 0 or terminated or truncated:
                    image = Image.fromarray(env.render())
                    painter = ImageDraw.Draw(image)
                    painter.rectangle((0, 0, 640, 62), fill=(12, 18, 25))
                    painter.text((10, 6), f"SIMULATION ONLY | fixture reach | {controller}", fill="white")
                    status = "PASS" if info["success"] else info["failure_reason"] or "running"
                    painter.text((10, 24), f"t={env.data.time:.2f}s seed={seed} {status} ckpt={sha256(checkpoint)[:12]}", fill="white")
                    painter.text((10, 42), "FIXTURE ASSISTANCE; NOT MOBILE OR HARDWARE VALIDATION", fill=(255, 190, 60))
                    writer.append_data(np.asarray(image))
                if terminated or truncated:
                    break
    finally:
        env.close()
    record = {"case": "reach", "mode": "fixture", "controller": controller, "seed": seed, "steps": len(trace), "return": reward_sum,
              "success": info["success"], "failure_reason": info["failure_reason"], "final_diagnostics": info["diagnostics"],
              "simulation_only": True, "hardware_ready": False, "provenance": provenance(), "trace": trace,
              "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "learning_provenance": metadata["provenance"],
              "video": {"path": str(video), "sha256": sha256(video), "frames_include_full_rollout_at_25fps": True}}
    write_json(output / f"{label}.json", record)
    write_json(output / "evaluation.json", {"scope": "checkpoint_replay_fixture_only", "records": [{key: value for key, value in record.items() if key != "trace"}], "provenance": provenance()})
    return {key: value for key, value in record.items() if key not in ("trace", "learning_provenance", "final_diagnostics")}


def main():
    parser = argparse.ArgumentParser(description="Record an actual v1-B fixture reach checkpoint rollout")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=("bc", "ppo"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--max-steps", type=int, default=250)
    arguments = parser.parse_args()
    print(json.dumps(render_checkpoint(arguments.checkpoint, arguments.kind, arguments.out, arguments.seed, arguments.max_steps), indent=2))


if __name__ == "__main__":
    main()
