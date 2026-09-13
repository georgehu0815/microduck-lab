from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlx/examples"))


def run_job(case: str, seed: int, output: Path) -> dict:
    directory = output / case / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(directory), "--job", "--case", case, "--seed", str(seed)]
    with (directory / "strict-job.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    summary_path = directory / "strict-summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"case_id": case, "seed": seed, "execution_completed": False}
    summary["exit_code"] = result.returncode
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    import torch
    import ppo_microduck_arm as pipeline
    from rlx.environments import arm

    torch.set_num_threads(1)
    if args.job:
        learner = pipeline.train(args.case, args.output, 32768, args.seed)
        metadata = json.loads((args.output / "training.json").read_text())
        pipeline.validate_training_evidence(args.output, args.case, metadata)
        report = pipeline.evaluate(args.case, "ppo_residual", learner, args.output / "evaluation.json", 10, 80000)
        report["checkpoint_sha256"] = pipeline.digest(args.output / "ppo-residual.zip")
        pipeline.write_json(args.output / "evaluation.json", report)
        summary = {"case_id": args.case, "seed": args.seed, "execution_completed": True,
                   "episodes": report["episodes"], "successes": report["successes"],
                   "statistical_gate_passed": report["passed"], "all_episodes_pass": report["successes"] == report["episodes"],
                   "source_hashes": pipeline.sources()}
        pipeline.write_json(args.output / "strict-summary.json", summary)
        print(json.dumps(summary), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    pipeline.write_json(args.output / "strict-protocol.json", {
        "source_hashes": pipeline.sources(), "steps_per_checkpoint": 32768,
        "cases": list(arm.CASES), "training_seeds": [101, 202, 303],
        "evaluation_seed_base": 80000, "seeds_per_config": 10, "perturbations": pipeline.EVAL_CONFIGS,
        "strict_gate": "1800/1800; no skipped cases, no relaxed episode gates", "retrained": True,
    })
    cases = ["arms-co-carry-v1", *[case for case in arm.CASES if case != "arms-co-carry-v1"]]
    jobs = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_job, case, seed, args.output) for case in cases for seed in (101, 202, 303)]
        for future in as_completed(futures):
            result = future.result()
            jobs.append(result)
            print(json.dumps(result), flush=True)
    summary = {"execution_completed": len(jobs) == 18 and all(job["exit_code"] == 0 for job in jobs),
               "all_episodes_pass": len(jobs) == 18 and all(job.get("all_episodes_pass", False) for job in jobs),
               "episodes": sum(job.get("episodes", 0) for job in jobs),
               "successes": sum(job.get("successes", 0) for job in jobs), "jobs": jobs}
    pipeline.write_json(args.output / "strict-summary.json", summary)
    if not summary["all_episodes_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
