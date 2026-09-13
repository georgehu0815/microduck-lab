"""Reproducible learned feedback + conservative PPO for contact-only color painting."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rlx.environments import brush, brush_reference, drawing, drawing_reference
from rlx.models import drawing_feedback

TRAIN_CASES = (
    {}, {"offset": (.003, -.002)}, {"offset": (-.003, .002)},
    {"scale": .9}, {"scale": 1.1}, {"friction": 1.0},
    {"stiffness": 20.0}, {"stiffness": 35.0},
)
EVAL_CASES = {
    "nominal": {}, "offset": {"offset": (.002, .002)},
    "scale": {"scale": .95}, "friction": {"friction": .9},
    "compliance": {"stiffness": 30.0},
    "combined_left": {"offset": (-.002, -.001), "scale": 1.04, "friction": .95, "stiffness": 23.0},
    "combined_small": {"offset": (.001, -.002), "scale": .92, "friction": 1.1, "stiffness": 32.0},
    "combined_right": {"offset": (.004, .001), "scale": 1.06, "friction": .9, "stiffness": 28.0},
    "boundary_large": {"offset": (.005, -.003), "scale": 1.12, "friction": .8, "stiffness": 40.0},
    "boundary_soft": {"offset": (-.004, .003), "scale": .88, "friction": 1.0, "stiffness": 18.0},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    return {"pipeline_sha256": digest(__file__), "environment_sha256": digest(brush.__file__),
            "reference_sha256": digest(brush_reference.__file__), "actor_sha256": digest(drawing_feedback.__file__),
            "base_environment_sha256": digest(drawing.__file__), "base_reference_sha256": digest(drawing_reference.__file__)}


def assert_sources(expected):
    if source_hashes() != expected:
        raise RuntimeError("Source package changed during the operation; artifacts are not accepted")


def validate_dataset(path):
    metadata_path = Path(path).with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError("Teacher dataset requires adjacent JSON provenance")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("sha256") != digest(path) or metadata.get("source_hashes") != source_hashes():
        raise ValueError("Teacher dataset hash/source mismatch; regenerate from the pinned source")
    return metadata


def provenance_errors(path, metadata):
    errors = []
    required = {"recipe": "drawing", "actuator": "xml", "contract_version": brush.CONTRACT_VERSION,
                "observation_dim": 93, "action_dim": 15, "max_episode_s": 120.0,
                "recipe_options": {}, "reward_weights": brush.REWARD_WEIGHTS}
    for key, value in required.items():
        if metadata.get(key) != value:
            errors.append(f"metadata mismatch: {key}")
    recorded = metadata.get("source_hashes", {})
    current = source_hashes()
    if set(recorded) != set(current) or any(recorded.get(key) != value for key, value in current.items() if key != "pipeline_sha256"):
        errors.append("training environment/reference/actor package differs from evaluation source")
    archived_sources = {
        "pipeline_sha256": "ppo_microduck_brush.py", "environment_sha256": "brush.py",
        "reference_sha256": "brush_reference.py", "actor_sha256": "drawing_feedback.py",
        "base_environment_sha256": "drawing.py", "base_reference_sha256": "drawing_reference.py",
    }
    for key, filename in archived_sources.items():
        archived = Path(path).parent / "sources" / filename
        if not archived.exists() or digest(archived) != recorded.get(key):
            errors.append(f"training source archive missing or hash mismatch: {filename}")
    checkpoint = Path(path).with_suffix(".zip")
    sidecar_path = Path(str(checkpoint) + ".json")
    if not sidecar_path.exists() or not checkpoint.exists():
        return errors + ["checkpoint or metadata missing"]
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("checkpoint_sha256") != digest(checkpoint):
        errors.append("checkpoint hash mismatch")
    if metadata.get("checkpoint_sha256") != sidecar.get("checkpoint_sha256"):
        errors.append("embedded ONNX and sidecar checkpoint hashes differ")
    if sidecar.get("onnx", {}).get("sha256") != digest(path):
        errors.append("ONNX sidecar hash mismatch")
    if sidecar.get("source_hashes") != metadata.get("source_hashes"):
        errors.append("embedded and sidecar source packages differ")
    return errors


def run_episode(policy, *, seed, options=None, collect=False, control=None):
    env = brush.BrushEnv(seed=seed, **(options or {}))
    obs, _ = env.reset(seed=seed)
    observations, labels, rewards = [], [], []
    for _ in range(env.max_steps):
        if collect:
            observations.append(obs.copy())
            labels.append(env.teacher_action())
        action = np.asarray(policy(obs) if policy else env.teacher_action(), np.float32)
        if control == "null":
            action[:] = 0
            action[14] = -.8333333
        elif control == "open_jaw":
            action[14] = 1
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break
    assessment = env.assessment()
    result = {"seed": seed, "environment": options or {}, "steps": env.step_count,
              "return": float(sum(rewards)), "drawing_assessment": assessment}
    env.close()
    return result, (observations, labels, rewards)


def torch_actor(model):
    def action(observation):
        with torch.no_grad():
            return model.policy.feedback_actor(torch.from_numpy(observation)).numpy()
    return action


def data_command(args):
    pinned_sources = source_hashes()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    obs, actions, rewards, episodes, records = [], [], [], [], []
    for index, options in enumerate(TRAIN_CASES):
        result, values = run_episode(None, seed=args.seed + index, options=options, collect=True)
        observations, labels, returns = values
        obs.extend(observations)
        actions.extend(labels)
        rewards.extend(returns)
        episodes.extend([index] * len(observations))
        records.append(result)
        print(json.dumps({"data_episode": index, "steps": result["steps"], "assessment": result["drawing_assessment"]}), flush=True)
    assert_sources(pinned_sources)
    target = output / "teacher.npz"
    np.savez_compressed(target, observations=np.asarray(obs, np.float32), actions=np.asarray(actions, np.float32),
                        rewards=np.asarray(rewards, np.float32), episode=np.asarray(episodes))
    write_json(output / "teacher.json", {"controller": "Jacobian teacher, not PPO", "sha256": digest(target),
               "source_hashes": pinned_sources, "episodes": records})
    assert_sources(pinned_sources)


class AnchoredPPO(PPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["anchor_observations", "anchor_actions", "history"]

    def train(self):
        state = deepcopy(self.policy.state_dict())
        optimizer = deepcopy(self.policy.optimizer.state_dict())
        super().train()
        with torch.no_grad():
            differences = self.policy.feedback_actor(self.anchor_observations) - self.anchor_actions
            drift = float(differences.abs().max())
        accepted = bool(np.isfinite(drift) and drift <= self.anchor_limit)
        if not accepted:
            self.policy.load_state_dict(state)
            self.policy.optimizer.load_state_dict(optimizer)
        entry = {"steps": self.num_timesteps, "accepted_update": accepted, "anchor_max_drift": drift}
        for name in ("policy_gradient_loss", "value_loss", "approx_kl", "clip_fraction", "entropy_loss"):
            entry[name] = float(self.logger.name_to_value.get("train/" + name, 0))
        entry["reward_mean"] = float(self.rollout_buffer.rewards.mean())
        self.history.append(entry)
        print(json.dumps({"event": "training_progress", "steps": self.num_timesteps,
                          "total": self._total_timesteps, "total_timesteps": self._total_timesteps,
                          "mean_reward": entry["reward_mean"], "normalize_rewards": False,
                          "training_progress": entry}), flush=True)


def export(model, destination, metadata):
    pinned_sources = source_hashes()
    actor = model.policy.feedback_actor.cpu().eval()
    sample = torch.zeros((1, brush.OBS_DIM), dtype=torch.float32)
    torch.onnx.export(actor, sample, str(destination), input_names=["obs"], output_names=["actions"],
                      dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=17, dynamo=False)
    graph = onnx.load(destination)
    graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value = brush.OBS_DIM
    graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = brush.ACTION_DIM
    onnx.helper.set_model_props(graph, {"rlx_metadata": json.dumps(metadata), "contract_version": json.dumps(brush.CONTRACT_VERSION)})
    onnx.checker.check_model(graph)
    onnx.save(graph, destination)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    probe = model.anchor_observations[:128].numpy()
    with torch.no_grad():
        expected = actor(torch.from_numpy(probe)).numpy()
    difference = float(np.max(np.abs(expected - session.run(None, {"obs": probe})[0])))
    if difference > 1e-5:
        raise RuntimeError(f"ONNX parity failed: {difference}")
    assert_sources(pinned_sources)
    return {"max_absolute_error": difference, "passed": True}


def train_command(args):
    pinned_sources = source_hashes()
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.out)
    if (output / "policy.zip").exists():
        raise FileExistsError("Choose a new brush run directory; existing checkpoints are immutable")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "sources"
    archive.mkdir(exist_ok=True)
    for source in (__file__, brush.__file__, brush_reference.__file__, drawing_feedback.__file__, drawing.__file__, drawing_reference.__file__):
        shutil.copy2(source, archive / Path(source).name)
    assert_sources(pinned_sources)
    if args.data is None:
        dataset_dir = output / "dataset"
        data_command(argparse.Namespace(out=str(dataset_dir), seed=args.seed))
        args.data = str(dataset_dir / "teacher.npz")
    dataset = np.load(args.data)
    validate_dataset(args.data)
    assert_sources(pinned_sources)
    observations, actions = dataset["observations"], dataset["actions"]
    env = DummyVecEnv([lambda: brush.BrushEnv(seed=args.seed)])
    model = AnchoredPPO(drawing_feedback.DrawingFeedbackPolicy, env, device="cpu", seed=args.seed,
                        n_steps=6144, batch_size=512, n_epochs=2, learning_rate=args.learning_rate,
                        gamma=.995, gae_lambda=.95, clip_range=.05, ent_coef=0, vf_coef=.05,
                        max_grad_norm=.3, target_kl=.002, verbose=0)
    fits = [model.policy.fit_feedback(observations, actions, ridge=args.ridge)]
    print(json.dumps({"feedback_fit": fits[-1]}), flush=True)
    for iteration in range(args.dagger):
        additions, labels = [], []
        diagnostics = []
        for index, options in enumerate(TRAIN_CASES[:5]):
            result, (obs, targets, _) = run_episode(torch_actor(model), seed=args.seed + 100 + iteration * 10 + index, options=options, collect=True)
            additions.extend(obs)
            labels.extend(targets)
            diagnostics.append(result)
        observations = np.concatenate((observations, np.asarray(additions, np.float32)))
        actions = np.concatenate((actions, np.asarray(labels, np.float32)))
        fit = model.policy.fit_feedback(observations, actions, ridge=args.ridge)
        fits.append(fit)
        write_json(output / f"dagger-{iteration}.json", {"fit": fit, "episodes": diagnostics})
        print(json.dumps({"dagger": iteration, "fit": fit, "passes": sum(row["drawing_assessment"]["passed"] for row in diagnostics)}), flush=True)
        assert_sources(pinned_sources)
    model.anchor_observations = torch.from_numpy(observations[::max(1, len(observations)//4096)].copy())
    with torch.no_grad():
        model.anchor_actions = model.policy.feedback_actor(model.anchor_observations).clone()
    model.anchor_limit = args.anchor_limit
    model.history = []
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "recipe": "drawing", "actuator": "xml",
                "recipe_options": {}, "reward_weights": brush.REWARD_WEIGHTS, "contract_version": brush.CONTRACT_VERSION,
                "observation_dim": brush.OBS_DIM, "action_dim": 15, "max_episode_s": 120.0,
                "pipeline_version": "microduck-brush-pipeline-v2", "source_hashes": pinned_sources,
                "deployment_scope": "simulation_only", "hardware_compatible": False,
                "training": vars(args), "dataset_sha256": digest(args.data), "fits": fits,
                "controller": "learned polynomial feedback + PPO; no runtime Jacobian teacher",
                "planner": "authored color-stroke sequence and physical palette targets, not learned visual planning"}
    metadata["training"] = {key: value for key, value in metadata["training"].items() if not callable(value)}
    assert_sources(pinned_sources)
    model.save(output / "bc.zip")
    metadata["checkpoint_sha256"] = digest(output / "bc.zip")
    export(model, output / "bc.onnx", metadata)
    metadata["onnx"] = {"sha256": digest(output / "bc.onnx")}
    write_json(output / "bc.zip.json", metadata)
    assert_sources(pinned_sources)
    baseline, _ = run_episode(torch_actor(model), seed=args.seed + 400)
    write_json(output / "bc-validation.json", baseline)
    print(json.dumps({"bc_validation": baseline["drawing_assessment"]}), flush=True)
    model.learn(total_timesteps=args.steps)
    assert_sources(pinned_sources)
    model.save(output / "policy.zip")
    metadata["checkpoint_sha256"] = digest(output / "policy.zip")
    metadata.pop("onnx", None)
    metadata["total_timesteps"] = model.num_timesteps
    metadata["accepted_ppo_updates"] = sum(row["accepted_update"] for row in model.history)
    metadata["onnx_parity"] = export(model, output / "policy.onnx", metadata)
    metadata["onnx_sha256"] = digest(output / "policy.onnx")
    metadata["onnx"] = {"sha256": metadata["onnx_sha256"]}
    write_json(output / "policy.zip.json", metadata)
    assert_sources(pinned_sources)
    write_json(output / "training.json", {"metadata": metadata, "history": model.history})
    with (output / "ppo_history.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(model.history[0]))
        writer.writeheader()
        writer.writerows(model.history)
    env.close()


def evaluate_command(args):
    source_path = Path(args.onnx).resolve()
    policy_bytes = source_path.read_bytes()
    sidecar_path = source_path.with_suffix(".zip.json")
    start_hashes = {str(path): digest(path) for path in (source_path, sidecar_path, source_path.with_suffix(".zip")) if path.exists()}
    start_hashes[str(source_path)] = hashlib.sha256(policy_bytes).hexdigest()
    for archived in sorted((source_path.parent / "sources").glob("*.py")):
        start_hashes[str(archived.resolve())] = digest(archived)
    start_sources = source_hashes()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(policy_bytes, sess_options=session_options, providers=["CPUExecutionProvider"])
    metadata = json.loads(session.get_modelmeta().custom_metadata_map["rlx_metadata"])
    errors = provenance_errors(source_path, metadata)
    if errors:
        raise ValueError("Evaluation provenance gate failed: " + "; ".join(errors))
    policy = lambda obs: session.run(None, {"obs": obs[None]})[0][0]
    results = []
    for case_index, (case, options) in enumerate(EVAL_CASES.items()):
        for episode in range(args.episodes):
            result, _ = run_episode(policy, seed=args.seed + case_index*1000 + episode, options=options)
            result["case"] = case
            results.append(result)
            print(json.dumps({"case": case, "seed": result["seed"], "assessment": result["drawing_assessment"]}), flush=True)
    controls = {}
    for control in ("null", "open_jaw"):
        controls[control], _ = run_episode(policy, seed=args.seed+50000, control=control)
    episodes = [{**result["drawing_assessment"], "seed": result["seed"], "case": result["case"], "measured_steps": result["steps"]} for result in results]
    finite = all(np.isfinite(result["return"]) and result["steps"] > 0 for result in results + list(controls.values()))
    if source_hashes() != start_sources or any(not Path(path).exists() or digest(path) != value for path, value in start_hashes.items()):
        errors.append("source changed during evaluation")
    pipeline_passed = bool(finite and not errors)
    passed = pipeline_passed and args.episodes >= 4 and all(result["passed"] for result in episodes) and all(not row["drawing_assessment"]["passed"] for row in controls.values())
    owned = dict(start_hashes)
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "recipe": "drawing",
              "command": "eval", "evaluation_mode": "skill", "contract_version": brush.CONTRACT_VERSION,
              "source_type": "policy", "source_files_sha256": owned,
              "finite": finite, "pipeline_passed": pipeline_passed, "success": passed,
              "provenance_errors": errors, "unique_conditions": len(EVAL_CASES),
              "minimum_seeds_per_condition": 4,
              "skill_status": "passed" if passed else "failed", "evaluation_settings_match": pipeline_passed,
              "evaluation": {"mode": "skill", "seed": args.seed, "eval_episodes": args.episodes,
                 "environment": {"recipe": "drawing", "actuator": "xml", "max_episode_s": 120.0,
                    "assistance": 0, "recipe_options": {}, "reward_weights": brush.REWARD_WEIGHTS}},
              "source": str(source_path), "source_sha256": start_hashes[str(source_path)], "source_hashes": start_sources,
              "environment_source_match": metadata["source_hashes"]["environment_sha256"] == start_sources["environment_sha256"],
              "training_pipeline_source_match": metadata["source_hashes"]["pipeline_sha256"] == start_sources["pipeline_sha256"],
              "training_source_hashes": metadata["source_hashes"],
              "passed": passed, "results": results, "controls": controls,
              "drawing_assessment": {"passed": passed, "unassisted": True, "episodes": episodes,
                 "accepted_episodes": sum(row["passed"] for row in episodes), "required_episodes": len(episodes),
                 "mean_coverage": float(np.mean([row["coverage"] for row in episodes])),
                 "mean_precision": float(np.mean([row["precision"] for row in episodes]))}}
    write_json(args.out, report)
    print(json.dumps(report), flush=True)


def export_command(args):
    checkpoint = Path(args.checkpoint)
    metadata = json.loads(Path(str(checkpoint) + ".json").read_text())
    model = AnchoredPPO.load(checkpoint, device="cpu")
    model.anchor_observations = torch.from_numpy(np.random.default_rng(71).normal(0, .01, (128, brush.OBS_DIM)).astype(np.float32))
    metadata.pop("onnx", None)
    metadata.pop("onnx_sha256", None)
    metadata["checkpoint_sha256"] = digest(checkpoint)
    metadata["export_source_hashes"] = source_hashes()
    parity = export(model, Path(args.onnx_output), metadata)
    metadata["onnx"] = {"sha256": digest(args.onnx_output), "parity": parity}
    metadata["onnx_sha256"] = digest(args.onnx_output)
    write_json(str(checkpoint) + ".json", metadata)
    print(json.dumps({"command": "export", "recipe": "drawing", "onnx": args.onnx_output, "parity": parity}), flush=True)


def render_command(args):
    from render_microduck_brush import render_brush
    report = render_brush(onnx_path=Path(args.onnx), output=Path(args.out), seed=args.seed, seconds=args.seconds)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data")
    data.add_argument("--out", required=True)
    data.add_argument("--seed", type=int, default=11)
    data.set_defaults(func=data_command)
    train = sub.add_parser("train")
    train.add_argument("--out", required=True)
    train.add_argument("--data")
    train.add_argument("--seed", type=int, default=101)
    train.add_argument("--steps", type=int, default=49152)
    train.add_argument("--dagger", type=int, default=2)
    train.add_argument("--ridge", type=float, default=.01)
    train.add_argument("--learning-rate", type=float, default=1e-7)
    train.add_argument("--anchor-limit", type=float, default=.00025)
    train.set_defaults(func=train_command)
    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--onnx", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seed", type=int, default=10001)
    evaluate.add_argument("--episodes", type=int, default=4)
    evaluate.set_defaults(func=evaluate_command)
    exporter = sub.add_parser("export")
    exporter.add_argument("--checkpoint", required=True)
    exporter.add_argument("--onnx-output", required=True)
    exporter.set_defaults(func=export_command)
    renderer = sub.add_parser("render")
    renderer.add_argument("--onnx", required=True)
    renderer.add_argument("--out", required=True)
    renderer.add_argument("--seed", type=int, default=501)
    renderer.add_argument("--seconds", type=float, default=120.0)
    renderer.set_defaults(func=render_command)
    args = parser.parse_args()
    if args.command == "train" and (args.steps < 6144 or args.dagger < 0 or not 0 < args.learning_rate <= .0001 or not 0 < args.anchor_limit <= .001):
        parser.error("brush training requires steps>=6144, dagger>=0, positive learning rate<=1e-4 and anchor limit<=.001")
    if args.command == "eval" and args.episodes < 1:
        parser.error("evaluation requires at least one episode per condition")
    torch.set_num_threads(1)
    args.func(args)


if __name__ == "__main__":
    main()
