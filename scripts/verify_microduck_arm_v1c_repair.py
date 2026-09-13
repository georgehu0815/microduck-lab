"""Evaluate every v1-C case using exported whole-body ONNX, retaining failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from microduck_arm_v1c.config import CASES, sha256
from microduck_arm_v1c import whole_body_learning as whole


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--video-seeds", default="0")
    parser.add_argument("--teacher", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    video_seeds = {int(value) for value in args.video_seeds.split(",") if value}
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or not video_seeds.issubset(seeds):
        parser.error("seeds must be unique nonnegative integers; video seeds must be evaluated")
    if args.out.exists():
        raise FileExistsError(f"Preserve existing evidence; choose a new output directory: {args.out}")
    snapshot = whole.source_provenance()
    models = {case: whole._load_onnx(args.learning_root / case / "policy.onnx") for case in CASES}
    records = []
    controllers = ("teacher", "teacher+wholebodyresidual") if args.teacher else ("teacher+wholebodyresidual",)
    for candidate in ("A", "B"):
        for case in CASES:
            session, metadata = models[case]
            caps = metadata["residual_caps_rad"]
            for controller in controllers:
                for seed in seeds:
                    output = args.out / f"{candidate}-{case}-{controller}-{seed}"
                    result = whole._rollout_episode(
                        seed=seed,
                        controller=controller,
                        predict_residual=lambda observation: whole._onnx_residual(session, observation),
                        case=case,
                        mode="free",
                        candidate=candidate,
                        max_steps=2500,
                        payload_kg=.01,
                        leg_residual_rad=float(caps[0]),
                        arm_residual_rad=float(caps[10]),
                        telemetry_path=output / "telemetry.jsonl",
                        video_path=output / "rollout.mp4" if seed in video_seeds else None,
                    )
                    result.update(case=case, candidate=candidate, onnx_sha256=metadata["sha256"])
                    whole._assert_provenance_unchanged(snapshot, "repair matrix episode")
                    whole.learning._write_json(output / "result.json", result)
                    records.append(result)
                    print(json.dumps({key: result[key] for key in ("case", "candidate", "seed", "controller", "success", "failure_reason")}), flush=True)
    expected = len(CASES) * 2 * len(seeds) * len(controllers)
    if len(records) != expected:
        raise RuntimeError("Incomplete matrix cannot pass")
    report = {
        "episodes": expected,
        "successes": sum(record["success"] for record in records),
        "all_passed": all(record["success"] for record in records),
        "controllers": {name: whole._summarize([record for record in records if record["controller"] == name]) for name in controllers},
        "source_provenance": snapshot,
        "runner_sha256": sha256(__file__),
        "simulation_only": True,
        "hardware_release": False,
        "observation_dim": 69,
        "policy_action_dim": 15,
        "teacher_required": True,
        "waist_joint_present": False,
        "video_seeds": sorted(video_seeds),
    }
    whole._assert_provenance_unchanged(snapshot, "repair matrix report")
    whole.learning._write_json(args.out / "policy-matrix.json", report)
    print(json.dumps({key: report[key] for key in ("episodes", "successes", "all_passed")}), flush=True)


if __name__ == "__main__":
    main()
