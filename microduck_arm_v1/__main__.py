from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import CASES


def main():
    parser = argparse.ArgumentParser(description="MicroDuck v1-B prototype engineering tools; no hardware actuation")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--out", default="hardware/microduck-arm-v1/generated")
    check = commands.add_parser("check")
    check.add_argument("--out", default="artifacts/microduck-arm-v1/model-check.json")
    for name in ("torque", "workspace"):
        child = commands.add_parser(name)
        child.add_argument("--out", required=True)
        child.add_argument("--samples", type=int, default=1000)
        child.add_argument("--seed", type=int, default=101)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--episodes", type=int, default=2)
    evaluate.add_argument("--max-steps", type=int, default=1500)
    evaluate.add_argument("--videos", action="store_true")
    for name in ("demonstrate", "train-ppo"):
        child = commands.add_parser(name)
        child.add_argument("--case", choices=CASES, default="reach")
        child.add_argument("--mode", choices=("free", "fixture"), default="free")
        child.add_argument("--out", required=True)
        child.add_argument("--seed", type=int, default=101)
        child.add_argument("--max-steps", type=int, default=1500)
        child.add_argument("--episodes" if name == "demonstrate" else "--steps", type=int, default=10 if name == "demonstrate" else 2048)
    bc = commands.add_parser("train-bc")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--out", required=True)
    bc.add_argument("--epochs", type=int, default=20)
    bc.add_argument("--seed", type=int, default=101)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    hardware = commands.add_parser("hardware-check")
    hardware.add_argument("--evidence")
    arguments = parser.parse_args()
    from . import validation
    if arguments.command == "build":
        from .model import build_artifacts
        result = build_artifacts(arguments.out)
    elif arguments.command == "check":
        result = validation.validate_model()
        validation.write_json(arguments.out, result)
    elif arguments.command == "torque":
        result = validation.torque_com_table(arguments.out, arguments.samples, arguments.seed)
    elif arguments.command == "workspace":
        result = validation.workspace_scan(arguments.out, arguments.samples, arguments.seed)
    elif arguments.command == "evaluate":
        if arguments.episodes < 1:
            parser.error("episodes must be positive")
        result = validation.evaluate_suite(arguments.out, tuple(range(910000, 910000 + arguments.episodes)), arguments.max_steps, arguments.videos)
    elif arguments.command == "hardware-check":
        result = validation.hardware_acceptance(arguments.evidence)
    else:
        from . import learning
        if arguments.command == "demonstrate":
            result = learning.collect_demonstrations(arguments.case, arguments.mode, arguments.episodes, arguments.seed, arguments.out, max_steps=arguments.max_steps)
        elif arguments.command == "train-ppo":
            result = learning.train_ppo(arguments.case, arguments.mode, arguments.out, arguments.steps, arguments.seed, max_steps=arguments.max_steps)
        elif arguments.command == "train-bc":
            result = learning.train_bc(arguments.dataset, arguments.out, arguments.epochs, arguments.seed)
        else:
            result = learning.export_policy(arguments.checkpoint, arguments.out)
        result = {"artifact": str(result)}
    print(json.dumps(result, indent=2, default=str, allow_nan=False))
    if arguments.command == "hardware-check" and not result["evidence_complete"]:
        return 2
    if arguments.command == "check" and not result["passed"]:
        return 1
    if arguments.command == "evaluate" and not result["all_robot_cases_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
