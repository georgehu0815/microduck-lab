"""Run a fresh, isolated Microduck training-to-video pipeline without a web server."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "rlx/examples/ppo_microduck_studio.py"
SCENARIOS = ("dance", "swing", "running", "stilts", "backflip")
RECIPE_DIRECTORY = ROOT / "docs/remaining-scenarios-e2e/recipes"
COMMON_OPTIONS = {
    "seed": "seed", "actuator": "actuator", "maxEpisodeS": "max-episode-s",
    "domainRand": "domain-rand", "obsNoise": "obs-noise",
    "actionDelay": "action-delay", "randomYaw": "random-yaw",
    "danceClip": "dance-clip", "dancePoseSigma": "dance-pose-sigma",
    "locomotionForwardCommand": "locomotion-forward-command",
    "stiltHeightCm": "stilt-height-cm", "stiltBlend": "stilt-blend",
    "stiltMassKg": "stilt-mass-kg", "swingInitialAngleDeg": "swing-initial-angle-deg",
    "swingInitialRateRadS": "swing-initial-rate-rad-s", "swingPlanarActions": "swing-planar-actions",
    "rewardWeights": "weight-overrides",
}
TRAIN_OPTIONS = {
    "totalTimesteps": "total-timesteps", "numEnvs": "num-envs",
    "numSteps": "num-steps", "numMinibatches": "num-minibatches",
    "updateEpochs": "update-epochs", "learningRate": "learning-rate", "gamma": "gamma",
    "clipCoefficient": "clip-coefficient", "entropyCoefficient": "entropy-coefficient",
    "initialStd": "initial-std", "maxGradNorm": "max-grad-norm",
    "normalizeRewards": "normalize-rewards", "checkpointInterval": "checkpoint-interval",
    "freezeObservationNormalization": "freeze-observation-normalization",
    "gaeLambda": "gae-lambda", "normalizeAdvantages": "normalize-advantages",
    "clipValueLoss": "clip-value-loss", "valueCoefficient": "value-coefficient",
}
ALLOWED_KEYS = set(COMMON_OPTIONS) | set(TRAIN_OPTIONS) | {
    "experimentId", "runName", "profile", "evalSteps", "renderSeconds",
    "resumeFromCheckpoint", "swingMinSpanDeg",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS)
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--run-name", help="fresh run name; existing directories are never overwritten")
    parser.add_argument("--output-dir", type=Path, help="fresh output directory instead of the Studio run directory")
    parser.add_argument("--recipe-json", type=Path, help="one-stage Studio recipe; continuation requires --init-from")
    parser.add_argument("--init-from", type=Path, help="explicit checkpoint with its .json sidecar; skips default warm-up stages")
    parser.add_argument("--dance-clip", type=Path, help="Dance motion clip JSON; defaults to tracked dance-120bpm")
    parser.add_argument("--steps", type=positive_int, help="override timesteps per PPO stage; not a skill guarantee")
    parser.add_argument("--num-envs", type=positive_int, help="override training environments")
    parser.add_argument("--eval-envs", type=positive_int, help="evaluation lanes (default: 8 full, 1 smoke)")
    parser.add_argument("--seed", type=int, help="training seed (default: recipe seed, or 7)")
    parser.add_argument("--eval-seed", type=int, default=101)
    parser.add_argument("--dry-run", action="store_true", help="print commands without importing MLX, creating files, or training")
    return parser.parse_args(argv)


def default_recipes(scenario: str) -> list[dict]:
    if scenario in {"running", "stilts"}:
        return [read_json(RECIPE_DIRECTORY / f"{scenario}-base.json"),
                read_json(RECIPE_DIRECTORY / f"{scenario}.json")]
    if scenario == "swing":
        return [read_json(RECIPE_DIRECTORY / "swing.json")]
    recipe = {
        "experimentId": scenario, "numEnvs": 16, "numSteps": 128,
        "numMinibatches": 4, "seed": 7, "actuator": "xml",
        "domainRand": False, "obsNoise": False, "actionDelay": False, "randomYaw": False,
        "normalizeRewards": True, "checkpointInterval": 100000,
    }
    if scenario == "dance":
        recipe.update(totalTimesteps=4001792, maxEpisodeS=8, updateEpochs=4,
                      learningRate=1e-4, initialStd=0.1, entropyCoefficient=0,
                      dancePoseSigma=0.2, rewardWeights={
                          "pose_match": 40, "rotation_match": 4, "gentle_head": 1,
                          "no_limit_parking": 1, "no_slip": 0.5, "no_spin": 0.3,
                          "on_feet": 5, "save_energy": 0.0005, "soft_landings": 0.75,
                          "stick_it": 5, "travel": 0,
                      })
    else:
        recipe.update(totalTimesteps=401408, maxEpisodeS=12, updateEpochs=2,
                      learningRate=3e-6, initialStd=0.03, entropyCoefficient=0)
    return [recipe]


def cli_options(recipe: dict, mapping: dict) -> list[str]:
    arguments = []
    for key, option in mapping.items():
        value = recipe.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            arguments.append(f"--{'' if value else 'no-'}{option}")
        else:
            arguments.extend([f"--{option}", json.dumps(value) if isinstance(value, dict) else str(value)])
    return arguments


def make_plan(args) -> dict:
    if args.dance_clip and args.scenario != "dance":
        raise ValueError("--dance-clip only applies to Dance")
    if args.init_from:
        args.init_from = args.init_from.expanduser().resolve(strict=True)
        sidecar = args.init_from.with_suffix(args.init_from.suffix + ".json")
        if read_json(sidecar).get("metadata", {}).get("recipe") != args.scenario:
            raise ValueError("--init-from metadata does not match the scenario")
    recipes = default_recipes(args.scenario)
    if args.recipe_json:
        custom = read_json(args.recipe_json.expanduser().resolve(strict=True))
        if not isinstance(custom, dict) or custom.get("experimentId", args.scenario) != args.scenario:
            raise ValueError("recipe JSON must be an object for this scenario")
        unknown = set(custom) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unsupported recipe fields: {sorted(unknown)}")
        if custom.get("resumeFromCheckpoint") and not args.init_from:
            raise ValueError("continuation recipe requires an explicit --init-from checkpoint")
        recipes = [{**recipes[-1], **custom}]
        recipes[0]["freezeObservationNormalization"] = custom.get("freezeObservationNormalization", False)
    elif args.init_from or args.profile == "smoke":
        recipes = recipes[-1:]
    name = args.run_name or f"{args.scenario}-pipeline-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", name):
        raise ValueError("run name must be 1–48 letters, digits, hyphens or underscores, starting alphanumeric")
    output = (args.output_dir or ROOT / "rlx/runs/studio" / args.scenario / name).expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists; choose a fresh run name or output directory: {output}")
    inputs = [Path(__file__), NATIVE]
    if args.recipe_json:
        inputs.append(args.recipe_json.expanduser().resolve())
    if args.init_from:
        inputs.extend([args.init_from, args.init_from.with_suffix(args.init_from.suffix + ".json")])
    commands = []
    initialization = args.init_from
    if args.scenario == "swing" and initialization is None and not args.recipe_json:
        bootstrap = ROOT / "rlx/scripts/bootstrap_swing_e2e.py"
        teacher = RECIPE_DIRECTORY / "swing-teacher.json"
        inputs.extend([bootstrap, teacher])
        destination = output / "initialization"
        command = [sys.executable, str(bootstrap), "--source", str(teacher), "--output", str(destination),
                   "--seed", str(args.seed if args.seed is not None else recipes[-1].get("seed", 7))]
        if args.profile == "smoke":
            command.extend(["--iterations", "0", "--max-samples", "1200", "--epochs", "1", "--train-seconds", "5", "--total-seconds", "90"])
        commands.append({"name": "initialize-swing", "argv": command})
        initialization = destination / "swing.safetensors"
    for index, original in enumerate(recipes):
        recipe = deepcopy(original)
        recipe.update(seed=args.seed if args.seed is not None else recipe.get("seed", 7),
                      actuator=recipe.get("actuator", "xml"))
        if args.scenario == "dance":
            clip = args.dance_clip or Path(recipe.get("danceClip") or ROOT / "rlx/assets/clips/dance-120bpm.json")
            clip = clip.expanduser().resolve(strict=True)
            recipe["danceClip"] = str(clip)
            document = read_json(clip)
            duration = float(document.get("duration") or document["keys"][-1]["t"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("dance clip duration must be finite and positive")
            recipe["maxEpisodeS"] = max(recipe.get("maxEpisodeS", 8), duration)
            inputs.append(clip)
        if args.profile == "smoke":
            recipe.update(totalTimesteps=32, numEnvs=2, numSteps=16, numMinibatches=1,
                          updateEpochs=1, checkpointInterval=0)
        if args.steps:
            recipe["totalTimesteps"] = args.steps
        if args.num_envs:
            recipe["numEnvs"] = args.num_envs
        batch = recipe["numEnvs"] * recipe["numSteps"]
        if recipe["totalTimesteps"] < batch or batch % recipe["numMinibatches"]:
            raise ValueError("steps must cover a rollout; num-envs × numSteps must be divisible by numMinibatches")
        if recipe.get("freezeObservationNormalization") and initialization is None:
            raise ValueError("frozen observation normalization requires --init-from or the default Swing bootstrap")
        destination = output if index == len(recipes) - 1 else output / "stages" / f"stage-{index + 1}"
        command = [sys.executable, str(NATIVE), "train", "--recipe", args.scenario,
                   "--output-dir", str(destination), "--no-export-onnx"]
        command.extend(cli_options(recipe, COMMON_OPTIONS | TRAIN_OPTIONS))
        if initialization:
            command.extend(["--init-from", str(initialization)])
        commands.append({"name": f"train-{index + 1}", "argv": command, "recipe": recipe})
        initialization = destination / f"{args.scenario}.safetensors"
    policy = output / f"{args.scenario}.onnx"
    checkpoint = output / f"{args.scenario}.safetensors"
    commands.append({"name": "export", "argv": [sys.executable, str(NATIVE), "export", "--recipe", args.scenario,
                                               "--checkpoint", str(checkpoint), "--output", str(policy)]})
    commands.append({"name": "verify-policy", "argv": [sys.executable, str(Path(__file__).with_name("verify_policy.py")),
                    "--checkpoint", str(checkpoint), "--policy", str(policy), "--scenario", args.scenario,
                    "--trained-steps", str(recipe["totalTimesteps"]), "--output", str(output / "verification.json")]})
    evaluation_recipe = {**recipe, "seed": args.eval_seed}
    episode_steps = round(recipe["maxEpisodeS"] * 50)
    eval_steps = max(episode_steps, {"swing": 1200, "running": 600, "stilts": 500, "backflip": 600}.get(args.scenario, 0), recipe.get("evalSteps", 0))
    if args.profile == "smoke":
        eval_steps = 32
    common = ["--recipe", args.scenario, "--policy", str(policy), *cli_options(evaluation_recipe, COMMON_OPTIONS)]
    commands.append({"name": "evaluate", "argv": [sys.executable, str(NATIVE), "eval", *common,
                    "--backend", "dummy", "--num-envs", str(args.eval_envs or (1 if args.profile == "smoke" else 8)),
                    "--eval-steps", str(eval_steps), "--evaluation-mode", "pipeline" if args.profile == "smoke" else "skill",
                    "--swing-min-span-deg", str(recipe.get("swingMinSpanDeg", 150))]})
    seconds = 1.0 if args.profile == "smoke" else max(eval_steps / 50, recipe.get("renderSeconds", 0))
    commands.append({"name": "render", "argv": [sys.executable, str(NATIVE), "render", *common,
                    "--output", str(output / "render"), "--episodes", "1", "--render-seconds", str(seconds),
                    "--width", "640", "--height", "360", "--fps", "25", "--sheet-frames", "12"]})
    return {"scenario": args.scenario, "profile": args.profile, "run_name": name, "output": str(output),
            "input_hashes": {str(item): digest(item) for item in inputs}, "commands": commands,
            "render_seconds": seconds, "visual_review_required": True,
            "scope": "Local nominal simulation; smoke is pipeline-only, and Backflip uses launch/stand assistance."}


def preflight(plan: dict) -> None:
    import imageio_ffmpeg
    import mlx.core as mx
    import numpy as np

    for package in ("onnx", "onnxruntime", "imageio", "PIL", "mujoco"):
        importlib.import_module(package)
    mx.eval(mx.ones((2,)) + 1)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-version"], check=True, capture_output=True)
    specification = importlib.util.spec_from_file_location("scenario_native", NATIVE)
    native = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = native
    specification.loader.exec_module(native)
    for step in plan["commands"]:
        if step["argv"][1] != str(NATIVE):
            continue
        arguments_to_check = step["argv"][2:].copy()
        if "--init-from" in arguments_to_check:
            position = arguments_to_check.index("--init-from")
            del arguments_to_check[position:position + 2]
            if "--freeze-observation-normalization" in arguments_to_check:
                arguments_to_check.remove("--freeze-observation-normalization")
        try:
            native.parse_args(arguments_to_check)
        except SystemExit as error:
            raise ValueError(f"invalid native arguments for {step['name']}") from error
    evaluation_command = next(step["argv"] for step in plan["commands"] if step["name"] == "evaluate")
    arguments = native.parse_args(evaluation_command[2:])
    arguments.num_envs = 1
    environment = native._environment(arguments, normalize=False)
    try:
        observation, state, _ = environment.reset(None)
        if observation.shape != (1, 61) or not np.isfinite(observation).all():
            raise ValueError("invalid reset observations; expected finite [1,61]")
        observation, _, reward, _, _, _ = environment.step(None, state, np.zeros((1, 14), np.float32))
        if observation.shape != (1, 61) or not np.isfinite(observation).all() or not np.isfinite(reward).all():
            raise ValueError("invalid preflight transition")
    finally:
        environment.close()


def run_command(step: dict, output: Path) -> tuple[int, dict]:
    print(f"\n=== {step['name']} ===", flush=True)
    last_result = None
    with (output / f"{step['name']}.log").open("w") as log:
        with subprocess.Popen(step["argv"], cwd=ROOT / "rlx", stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    try:
                        candidate = json.loads(line)
                        if isinstance(candidate, dict):
                            last_result = candidate
                    except ValueError:
                        pass
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    if code not in ({0, 2} if step["name"] == "evaluate" else {0}):
        raise RuntimeError(f"{step['name']} exited {code}; see {output / (step['name'] + '.log')}")
    if last_result is None:
        raise RuntimeError(f"{step['name']} produced no JSON result")
    return code, last_result


def verify_media(output: Path, expected_seconds: float) -> dict:
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image

    video = output / "render/ep0.mp4"
    sheet = output / "render/ep0_sheet.png"
    with (output / "verify-media.log").open("w") as log:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-xerror", "-i", str(video),
                        "-f", "null", "-"], check=True, stdout=log, stderr=log)
    with imageio.get_reader(video) as reader:
        metadata = reader.get_meta_data()
    if abs(metadata["duration"] - expected_seconds) > 0.15:
        raise ValueError(f"MP4 duration {metadata['duration']} does not match {expected_seconds}s")
    with Image.open(sheet) as image:
        image.verify()
    return {"duration": metadata["duration"], "mp4_sha256": digest(video), "sheet_sha256": digest(sheet)}


def execute(plan: dict) -> int:
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    report = {"scenario": plan["scenario"], "profile": plan["profile"], "status": "running", "stages": [],
              "visual_review_required": True, "hardware_deployment_validated": False}
    evaluation = None
    try:
        print("Checking runtime, input assets, and the 61-observation / 14-action contract…", flush=True)
        preflight(plan)
        report["stages"].append({"name": "preflight", "passed": True})
        for step in plan["commands"]:
            code, result = run_command(step, output)
            report["stages"].append({"name": step["name"], "exit_code": code, "result": result})
            write_json(output / "pipeline.json", report)
            if step["name"].startswith("train-"):
                if result.get("trained_steps", 0) < step["recipe"]["totalTimesteps"]:
                    raise ValueError("training stopped before its requested transition budget")
            if step["name"] == "evaluate":
                evaluation = result
                write_json(output / "evaluation.json", evaluation)
            if step["name"] == "render":
                write_json(output / "render.json", result)
        report["media"] = verify_media(output, plan["render_seconds"])
        if any(digest(Path(file)) != expected for file, expected in plan["input_hashes"].items()):
            raise ValueError("an input changed during the pipeline")
        policy = output / f"{plan['scenario']}.onnx"
        if evaluation.get("source_sha256") != digest(policy):
            raise ValueError("evaluation does not match the exported ONNX")
        passed = evaluation.get("passed") is True and evaluation.get("pipeline_passed") is True
        if plan["profile"] == "full":
            passed = passed and evaluation.get("skill_status") == "passed"
        report["status"] = ("smoke_passed" if plan["profile"] == "smoke" else "skill_passed") if passed else "evaluation_failed"
        report["skill_status"] = evaluation.get("skill_status", "not_assessed")
        print(f"\n{report['status']}: {output}\nReview render/ep0.mp4 and render/ep0_sheet.png before judging the motion.", flush=True)
        return 0 if passed else 2
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Interrupted by user")
        return 130
    finally:
        write_json(output / "pipeline.json", report)


def main(argv=None) -> int:
    try:
        plan = make_plan(parse_args(argv))
        if "--dry-run" in (argv if argv is not None else sys.argv[1:]):
            print(json.dumps(plan, indent=2))
            return 0
        return execute(plan)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
