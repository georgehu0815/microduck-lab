import argparse
import json

from .model import build_artifacts
from .validation import contact_convergence, evaluate, hardware_acceptance, rollout, torque_com_table, workspace_scan, write_json


def main():
    parser = argparse.ArgumentParser(description="MicroDuck v1-C simulation engineering tools; no hardware I/O")
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model")
    model.add_argument("--out", required=True)
    test = commands.add_parser("evaluate")
    test.add_argument("--out", required=True)
    test.add_argument("--seeds", type=int, default=10)
    convergence = commands.add_parser("contact-convergence")
    convergence.add_argument("--out", required=True)
    for name in ("torque", "workspace"):
        command = commands.add_parser(name)
        command.add_argument("--out", required=True)
        command.add_argument("--samples", type=int, default=200 if name == "torque" else 10000)
    acceptance = commands.add_parser("acceptance")
    acceptance.add_argument("--out", required=True)
    render = commands.add_parser("render")
    render.add_argument("--case", required=True)
    render.add_argument("--mode", choices=("fixture", "free"), required=True)
    render.add_argument("--candidate", choices=("A", "B"), default="A")
    render.add_argument("--seed", type=int, default=0)
    render.add_argument("--payload-kg", type=float, default=.01)
    render.add_argument("--out", required=True)
    render.add_argument("--video", required=True)
    args = parser.parse_args()
    if args.command == "model":
        result = build_artifacts(args.out)
    elif args.command == "evaluate":
        if args.seeds < 1:
            parser.error("seeds must be positive")
        result = evaluate(args.out, seeds=range(args.seeds))
        result = {key: result[key] for key in ("by_mode", "all_tasks_passed", "negative_controls_passed")}
    elif args.command == "torque":
        result = torque_com_table(args.out, samples=args.samples)
    elif args.command == "contact-convergence":
        result = contact_convergence(args.out)
        result = {"refined_settings_passed": result["refined_settings_passed"], "episodes": len(result["records"])}
    elif args.command == "workspace":
        result = workspace_scan(args.out, samples=args.samples)
        result = {key: result[key] for key in ("samples", "reachable", "reachable_collision_free", "balance_safe_qualified")}
    elif args.command == "acceptance":
        result = hardware_acceptance()
        write_json(args.out, result)
    else:
        result = rollout(args.case, args.mode, args.candidate, args.seed, payload_kg=args.payload_kg, output=args.out, video=args.video)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
