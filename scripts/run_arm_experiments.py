from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "rlx/examples/ppo_microduck_arm.py"
CASES = ("arm-reach-v1", "arm-pick-place-v1", "arm-relocate-v1", "arm-carry-v1", "arms-handover-v1", "arms-co-carry-v1")


def execute_job(root, case_id, seed, steps, episodes):
    output = root / case_id / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    operations = ["train", "eval"]
    if seed == 101:
        operations += ["teacher", "negative", "bc"]
    for operation in operations:
        command = [sys.executable, str(PIPELINE), operation, "--case", case_id,
                   "--output", str(output), "--seed", str(seed), "--steps", str(steps),
                   "--seeds-per-config", str(episodes)]
        start = time.monotonic()
        with (output / f"{operation}.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        event = {"case_id": case_id, "seed": seed, "operation": operation,
                 "exit_code": completed.returncode, "elapsed_s": time.monotonic() - start,
                 "command": command}
        print(json.dumps(event), flush=True)
        if completed.returncode:
            return {"case_id": case_id, "seed": seed, "execution_completed": False, "failed_operation": operation}
    return {"case_id": case_id, "seed": seed, "execution_completed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seeds-per-config", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh batch directory to preserve prior evidence")
    args.output.mkdir(parents=True)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_job, args.output, case_id, seed, args.steps, args.seeds_per_config)
                   for seed in (101, 202, 303) for case_id in CASES]
        outcomes = [future.result() for future in as_completed(futures)]
    report = {"execution_completed": all(item["execution_completed"] for item in outcomes),
              "does_not_imply_experiments_pass": True,
              "elapsed_s": time.monotonic() - start, "jobs": outcomes}
    (args.output / "batch-execution.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
