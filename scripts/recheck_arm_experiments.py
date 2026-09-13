from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlx/examples"))


def recheck_checkpoint(source: str, destination: str, seed_base: int) -> dict:
    import torch
    from stable_baselines3 import PPO
    import ppo_microduck_arm as pipeline

    torch.set_num_threads(1)
    source_path = Path(source)
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source_path / "training.json").read_text())
    case_id = metadata["case_id"]
    pipeline.validate_training_evidence(source_path, case_id, metadata)
    checkpoint = source_path / "ppo-residual.zip"
    model = PPO.load(checkpoint)
    started = time.monotonic()
    report = pipeline.evaluate(
        case_id, "ppo_residual", model, output / "evaluation.json", 10, seed_base
    )
    report["checkpoint_sha256"] = pipeline.digest(checkpoint)
    report["checkpoint_path"] = str(checkpoint.resolve())
    report["training_metadata_sha256"] = pipeline.digest(source_path / "training.json")
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    pipeline.write_json(output / "evaluation.json", report)
    previous = json.loads((source_path / "evaluation.json").read_text())
    regressions = []
    for old in previous["results"]:
        if old["passed"]:
            continue
        replay, _ = pipeline.run_episode(
            case_id, "ppo_residual", model, old["seed"], old["options"]
        )
        regressions.append({
            "original": old,
            "replay": replay,
            "same_verdict": old["passed"] == replay["passed"],
            "same_gates": old["gates"] == replay["gates"],
        })
    pipeline.write_json(output / "historical-failure-replays.json", {
        "checkpoint_sha256": pipeline.digest(checkpoint),
        "source_hashes": pipeline.sources(),
        "results": regressions,
    })
    failed = [row for row in report["results"] if not row["passed"]]
    result = {
        "case_id": case_id,
        "training_seed": int(source_path.name.split("-")[-1]),
        "checkpoint": str(checkpoint.resolve()),
        "evaluation": str((output / "evaluation.json").resolve()),
        "evaluation_sha256": pipeline.digest(output / "evaluation.json"),
        "episodes": report["episodes"],
        "successes": report["successes"],
        "passed": report["passed"],
        "wilson_95": report["wilson_95"],
        "per_config_success_rate": report["per_config_success_rate"],
        "failures": failed,
        "historical_failure_replays": regressions,
        "elapsed_s": time.monotonic() - started,
    }
    pipeline.write_json(output / "recheck.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh evaluation without retraining or replacing prior evidence")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=80000)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error("--workers must be between 1 and 6")
    from rlx.environments import arm
    import ppo_microduck_arm as pipeline

    pipeline.independent_eval_seeds(args.seed_base, 10)
    sources = [args.source / case / f"seed-{seed}" for case in arm.CASES for seed in (101, 202, 303)]
    new_seeds = {
        args.seed_base + index * pipeline.EVAL_CONFIG_SEED_STRIDE + offset
        for index in range(len(pipeline.EVAL_CONFIGS)) for offset in range(10)
    }
    for source in sources:
        prior = json.loads((source / "evaluation.json").read_text())
        if new_seeds.intersection(row["seed"] for row in prior["results"]):
            raise ValueError("Fresh evaluation seed set overlaps historical evidence")
        metadata = json.loads((source / "training.json").read_text())
        pipeline.validate_training_evidence(source, source.parent.name, metadata)
    args.output.mkdir(parents=True, exist_ok=False)
    pipeline.write_json(args.output / "protocol.json", {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_batch": str(args.source.resolve()),
        "source_hashes": pipeline.sources(),
        "runner_sha256": pipeline.digest(__file__),
        "seed_base": args.seed_base,
        "seeds_per_config": 10,
        "perturbations": pipeline.EVAL_CONFIGS,
        "training_seeds": [101, 202, 303],
        "case_minimum_success_rates": {case: arm.SPECS[case].success_rate for case in arm.CASES},
        "minimum_per_config_success_rate": 0.8,
        "all_episodes_pass_is_separate_from_threshold_pass": True,
        "retraining": False,
        "simulation_only": True,
    })
    jobs = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(recheck_checkpoint, str(source), str(args.output / source.parent.name / source.name), args.seed_base): source
            for source in sources
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                jobs.append(result)
                print(json.dumps({key: result[key] for key in ("case_id", "training_seed", "episodes", "successes", "passed", "elapsed_s")}), flush=True)
            except Exception as error:
                errors.append({"checkpoint_directory": str(futures[future]), "error": repr(error)})
                print(json.dumps(errors[-1]), flush=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed": len(jobs) == 18 and not errors,
        "all_checkpoints_meet_threshold": len(jobs) == 18 and all(job["passed"] for job in jobs),
        "all_episodes_pass": len(jobs) == 18 and all(job["successes"] == job["episodes"] for job in jobs),
        "episodes": sum(job["episodes"] for job in jobs),
        "successes": sum(job["successes"] for job in jobs),
        "hardware_verified": False,
        "source_hashes": pipeline.sources(),
        "errors": errors,
        "jobs": sorted(jobs, key=lambda job: (job["case_id"], job["training_seed"])),
    }
    pipeline.write_json(args.output / "summary.json", summary)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
