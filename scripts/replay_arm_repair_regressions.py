from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlx/examples"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from stable_baselines3 import PPO
    import ppo_microduck_arm as pipeline

    torch.set_num_threads(1)
    old = json.loads((args.historical / "summary.json").read_text())
    samples = []
    for job in old["jobs"]:
        samples.extend({"original_training_seed": job["training_seed"], "episode": row} for row in job["failures"])
        samples.extend({"original_training_seed": job["training_seed"], "episode": row["replay"]} for row in job["historical_failure_replays"])
    rows = []
    for training_seed in (101, 202, 303):
        directory = args.training / "arms-co-carry-v1" / f"seed-{training_seed}"
        metadata = json.loads((directory / "training.json").read_text())
        pipeline.validate_training_evidence(directory, "arms-co-carry-v1", metadata)
        checkpoint = directory / "ppo-residual.zip"
        model = PPO.load(checkpoint)
        for sample in samples:
            expected = sample["episode"]
            episode, _ = pipeline.run_episode("arms-co-carry-v1", "ppo_residual", model, expected["seed"], expected["options"])
            row = {"case_id": "arms-co-carry-v1", "training_seed": training_seed,
                   "original_training_seed": sample["original_training_seed"],
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": pipeline.digest(checkpoint),
                   "model_sha256": metadata["model_sha256"], "source_hashes": pipeline.sources(),
                   "original_episode": expected, "episode": episode}
            rows.append(row)
            print(json.dumps({"training_seed": training_seed, "evaluation_seed": expected["seed"], "passed": episode["passed"],
                              "failed_gates": [name for name, passed in episode["gates"].items() if not passed]}), flush=True)
    report = {"historical_summary_sha256": pipeline.digest(args.historical / "summary.json"),
              "historical_batch": str(args.historical.resolve()), "episodes": len(rows),
              "successes": sum(row["episode"]["passed"] for row in rows),
              "all_passed": all(row["episode"]["passed"] for row in rows), "results": rows}
    pipeline.write_json(args.training / "known-failure-regressions.json", report)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
