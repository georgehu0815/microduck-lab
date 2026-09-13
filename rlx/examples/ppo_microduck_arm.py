from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rlx.environments import arm


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "rlx/runs/arm"
RLX_ROOT = Path(arm.__file__).resolve().parents[2]
RESIDUAL_SCALE = .02
INDEPENDENT_EVAL_SEED_MIN = 50000
EVAL_CONFIG_SEED_STRIDE = 100
EVAL_CONFIGS = [
    {"position_noise": .001}, {"position_noise": .0005},
    {"position_noise": .0015}, {"position_noise": .002},
    {"mass_scale": .9}, {"mass_scale": 1.1},
    {"friction_scale": .9}, {"friction_scale": 1.1},
    {"position_noise": .0015, "mass_scale": .95, "friction_scale": .95},
    {"position_noise": .0015, "mass_scale": 1.05, "friction_scale": 1.05},
]
CURRICULUM = {
    "arm-reach-v1": {
        "nominal_payload_mass_kg": 0.0,
        "design_goal_payload_mass_kg": 0.0,
        "stage": "no_payload_reach",
    },
    "arm-pick-place-v1": {
        "nominal_payload_mass_kg": 0.020,
        "design_goal_payload_mass_kg": 0.020,
        "stage": "20g_goal",
    },
    "arm-relocate-v1": {
        "nominal_payload_mass_kg": 0.020,
        "design_goal_payload_mass_kg": 0.020,
        "stage": "20g_goal",
    },
    "arm-carry-v1": {
        "nominal_payload_mass_kg": 0.020,
        "design_goal_payload_mass_kg": 0.020,
        "stage": "20g_goal",
    },
    "arms-handover-v1": {
        "nominal_payload_mass_kg": 0.005,
        "design_goal_payload_mass_kg": 0.020,
        "stage": "reduced_5g_curriculum_before_20g_goal",
    },
    "arms-co-carry-v1": {
        "nominal_payload_mass_kg": 0.017,
        "design_goal_payload_mass_kg": 0.020,
        "stage": "17g_total_assembly_below_20g_goal",
    },
}
SOURCE_ARCHIVE_NAMES = {
    "environment": "arm.py",
    "pipeline": "ppo_microduck_arm.py",
}
SUPPLEMENTAL_ARCHIVE_NAMES = {
    "experiment_spec": "EXPERIMENTS.md",
    "project": "pyproject.toml",
    "lockfile": "uv.lock",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def source_paths():
    return {
        "environment": Path(arm.__file__).resolve(),
        "pipeline": Path(__file__).resolve(),
    }


def supplemental_paths():
    return {
        "experiment_spec": ROOT / "docs/robot-arm-design/EXPERIMENTS.md",
        "project": RLX_ROOT / "pyproject.toml",
        "lockfile": RLX_ROOT / "uv.lock",
    }


def sources():
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required provenance sources are missing: {missing}")
    return {name: digest(path) for name, path in paths.items()}


def supplemental_provenance():
    return {
        name: digest(path)
        for name, path in supplemental_paths().items()
        if path.is_file()
    }


def repository_provenance():
    relative_sources = []
    for path in (*source_paths().values(), *supplemental_paths().values()):
        try:
            relative_sources.append(str(path.relative_to(RLX_ROOT)))
        except ValueError:
            continue
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RLX_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--", *relative_sources],
            cwd=RLX_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "source_worktree_status": None}
    return {"commit": commit, "source_worktree_status": status}


def runtime_versions():
    packages = ("gymnasium", "mujoco", "numpy", "stable-baselines3", "torch")
    return {package: importlib.metadata.version(package) for package in packages}


def payload_metadata(env):
    case_id = env.case_id
    object_mass = float(env.model.body_mass[env.object_id])
    loose_payload_mass = 0.0
    if case_id == "arms-co-carry-v1":
        loose_payload_mass = float(env.model.body_mass[env.model.body("payload").id])
    transported_mass = 0.0 if case_id == "arm-reach-v1" else object_mass + loose_payload_mass
    return {
        "modeled_object_body_mass_kg": object_mass,
        "loose_payload_mass_kg": loose_payload_mass,
        "actual_transported_assembly_mass_kg": transported_mass,
        **CURRICULUM[case_id],
    }


def nominal_payload_metadata(case_id):
    env = arm.ArmEnv(case_id)
    try:
        return payload_metadata(env)
    finally:
        env.close()


def controller_semantics(controller, steps=None):
    semantics = {
        "controller": controller,
        "physics_assistance": False,
        "physics_assistance_count": 0,
        "authored_action_steps": 0,
        "authored_scheduler_conditioning": False,
        "learned_action": False,
        "summary": controller,
    }
    if controller == "teacher":
        semantics.update({
            "authored_action_steps": steps,
            "authored_scheduler_conditioning": True,
            "summary": "authored IK/FSM teacher; no learned action and no physics assistance",
        })
    elif controller == "ppo_residual":
        semantics.update({
            "authored_action_steps": steps,
            "authored_scheduler_conditioning": True,
            "learned_action": True,
            "summary": "authored IK/FSM nominal control plus learned bounded PPO residual; not end-to-end planning",
        })
    elif controller == "bc":
        semantics.update({
            "authored_scheduler_conditioning": True,
            "learned_action": True,
            "summary": "learned full-action BC conditioned on the authored time/FSM scheduler; diagnostic, not autonomous planning",
        })
    elif controller in ("open_gripper", "disable_right"):
        semantics.update({
            "authored_action_steps": steps,
            "authored_scheduler_conditioning": True,
            "summary": f"authored IK/FSM negative control with {controller} intervention",
        })
    elif controller == "zero":
        semantics["summary"] = "zero-action negative control"
    return semantics


def independent_eval_seeds(seed_base, seeds_per_config):
    if seed_base < INDEPENDENT_EVAL_SEED_MIN:
        raise ValueError(
            f"Independent evaluation seed_base must be >= {INDEPENDENT_EVAL_SEED_MIN}"
        )
    if not 1 <= seeds_per_config <= EVAL_CONFIG_SEED_STRIDE:
        raise ValueError(
            f"seeds_per_config must be in [1, {EVAL_CONFIG_SEED_STRIDE}]"
        )
    seeds = [
        seed_base + config_index * EVAL_CONFIG_SEED_STRIDE + seed_index
        for config_index in range(len(EVAL_CONFIGS))
        for seed_index in range(seeds_per_config)
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Independent evaluation seeds overlap between configurations")
    return {
        "independent": True,
        "minimum_allowed_seed": INDEPENDENT_EVAL_SEED_MIN,
        "seed_base": seed_base,
        "seed_min": min(seeds),
        "seed_max": max(seeds),
        "seeds_per_config": seeds_per_config,
        "config_count": len(EVAL_CONFIGS),
    }


def validate_training_evidence(output, case_id, metadata):
    output = Path(output)
    expected_sources = sources()
    expected_model = arm.model_hash(case_id)
    if metadata.get("case_id") != case_id:
        raise ValueError(
            f"Policy case mismatch: checkpoint is {metadata.get('case_id')!r}, requested {case_id!r}"
        )
    if metadata.get("model_sha256") != expected_model:
        raise ValueError("Policy/model provenance mismatch")
    if metadata.get("source_hashes") != expected_sources:
        raise ValueError("Policy/source provenance mismatch")
    checkpoint = output / "ppo-residual.zip"
    if metadata.get("checkpoint_sha256") != digest(checkpoint):
        raise ValueError("Policy/checkpoint provenance mismatch")
    archive = output / "sources"
    for name, expected_hash in expected_sources.items():
        archived = archive / SOURCE_ARCHIVE_NAMES[name]
        if not archived.is_file() or digest(archived) != expected_hash:
            raise ValueError(f"Archived source provenance mismatch: {name}")
    for name, expected_hash in metadata.get("supplemental_provenance", {}).items():
        archived = archive / SUPPLEMENTAL_ARCHIVE_NAMES[name]
        if not archived.is_file() or digest(archived) != expected_hash:
            raise ValueError(f"Archived supplemental provenance mismatch: {name}")
    scene = output / "scene.xml"
    if not scene.is_file() or digest(scene) != expected_model:
        raise ValueError("Archived scene provenance mismatch")


class ResidualArmEnv(gym.Wrapper):
    def __init__(self, case_id, seed=0):
        super().__init__(arm.ArmEnv(case_id))
        self.seed_value = seed
        self.episodes = 0

    def reset(self, **kwargs):
        self.episodes += 1
        kwargs.setdefault("seed", self.seed_value + self.episodes)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.env.controller = "scripted_ik_fsm_plus_ppo_residual"
        nominal = self.env.teacher_action()
        return self.env.step(np.clip(nominal + RESIDUAL_SCALE * np.asarray(action), -1, 1))


class TraceCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.episodes_path = self.path.with_name("episodes.jsonl")
        self.episode_returns = None
        self.episode_lengths = None
        self.completed_episode_returns = []

    def _on_step(self):
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals["infos"]
        if self.episode_returns is None:
            self.episode_returns = np.zeros_like(rewards)
            self.episode_lengths = np.zeros(rewards.shape, dtype=np.int64)
        self.episode_returns += rewards
        self.episode_lengths += 1
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            episode_return = float(self.episode_returns[index])
            row = {
                "env_steps": int(self.num_timesteps),
                "return": episode_return,
                "length": int(self.episode_lengths[index]),
                "success": bool(info.get("passed", False)),
                "gates": info.get("gates", {}),
                "physics_assistance_count": int(
                    info.get("physics_assistance_count", 0)
                ),
            }
            with self.episodes_path.open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            self.completed_episode_returns.append(episode_return)
            self.episode_returns[index] = 0
            self.episode_lengths[index] = 0
        return True

    def _on_rollout_start(self):
        if not self.model.num_timesteps:
            return
        episode_count = len(self.completed_episode_returns)
        row = {"env_steps": self.model.num_timesteps, "elapsed_s": (time.time_ns() - self.model.start_time) / 1e9}
        row.update({key.replace("train/", ""): float(value) for key, value in self.model.logger.name_to_value.items() if key.startswith("train/") and np.isscalar(value)})
        row["rollout_episode_reward_mean"] = (
            float(np.mean(self.completed_episode_returns))
            if episode_count
            else None
        )
        row["rollout_episode_count"] = episode_count
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.completed_episode_returns.clear()

    def _on_training_end(self):
        self._on_rollout_start()


def run_episode(case_id, controller="teacher", model=None, seed=1, options=None, trace=False):
    env = arm.ArmEnv(case_id)
    observation, _ = env.reset(seed=seed, options=options)
    rows = []
    total_reward = 0.0
    started = time.monotonic()
    while not (env.terminated or env.truncated):
        if controller == "zero":
            action = np.zeros(env.action_dim)
        elif controller in ("teacher", "open_gripper", "disable_right"):
            action = env.teacher_action()
            if controller == "open_gripper":
                action[5::6] = 1
            if controller == "disable_right":
                action[6:] = 0
        elif controller == "ppo_residual":
            nominal = env.teacher_action()
            residual, _ = model.predict(observation, deterministic=True)
            action = np.clip(nominal + RESIDUAL_SCALE * residual, -1, 1)
        elif controller == "bc":
            env.teacher_targets()
            action, _ = model.predict(observation, deterministic=True)
        else:
            raise ValueError("Unknown controller")
        observation, reward, _, _, info = env.step(action)
        total_reward += reward
        if trace:
            rows.append({"step": env.steps, "time": env.steps * arm.CONTROL_DT, "action": np.asarray(action).tolist(), "metrics": info})
    metrics = env.metrics()
    semantics = controller_semantics(controller, env.steps)
    semantics["physics_assistance_count"] = int(metrics.get("assistance_count", 0))
    result = metrics | {"case_id": case_id, "controller": controller, "seed": seed,
                        "options": options or {}, "steps": env.steps, "return": total_reward,
                        "wall_seconds": time.monotonic() - started,
                        "control_semantics": semantics,
                        "payload_mass": payload_metadata(env)}
    env.close()
    return result, rows


def evaluate(case_id, controller, model, output, seeds_per_config=10, seed_base=50000):
    seed_manifest = independent_eval_seeds(seed_base, seeds_per_config)
    source_identity = sources()
    results = []
    for config_index, options in enumerate(EVAL_CONFIGS):
        for seed_index in range(seeds_per_config):
            seed = seed_base + config_index * EVAL_CONFIG_SEED_STRIDE + seed_index
            result, _ = run_episode(case_id, controller, model, seed, options)
            result["config_index"] = config_index
            results.append(result)
    successes = sum(result["passed"] for result in results)
    per_config = [sum(result["passed"] for result in results if result["config_index"] == index) / seeds_per_config for index in range(len(EVAL_CONFIGS))]
    fraction = successes / len(results)
    score = 1.96
    denominator = 1 + score * score / len(results)
    center = (fraction + score * score / (2 * len(results))) / denominator
    radius = score * np.sqrt(fraction * (1 - fraction) / len(results) + score * score / (4 * len(results)**2)) / denominator
    aggregate_semantics = controller_semantics(
        controller, sum(result["steps"] for result in results)
    )
    aggregate_semantics["physics_assistance_count"] = sum(
        result["control_semantics"]["physics_assistance_count"]
        for result in results
    )
    report = {"case_id": case_id, "controller": controller, "episodes": len(results), "successes": successes,
              "success_rate": fraction, "per_config_success_rate": per_config,
              "wilson_95": [float(center - radius), float(center + radius)],
              "passed": fraction >= arm.SPECS[case_id].success_rate and min(per_config) >= .8,
              "source_hashes": source_identity, "model_sha256": arm.model_hash(case_id),
              "simulation_only": True, "hardware_verified": False,
              "policy_semantics": controller_semantics(controller)["summary"],
              "control_semantics": aggregate_semantics,
              "payload_curriculum": CURRICULUM[case_id],
              "evaluation_seeds": seed_manifest,
              "results": results}
    if sources() != source_identity:
        raise RuntimeError("Source files changed during evaluation; report rejected")
    write_json(output, report)
    return report


def behavior_clone(case_id, output, episodes=10, epochs=30, seed=101):
    observations, actions = [], []
    env = arm.ArmEnv(case_id)
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        while not (env.terminated or env.truncated):
            action = env.teacher_action()
            observations.append(observation.copy())
            actions.append(action.copy())
            observation, _, _, _, _ = env.step(action)
    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    np.savez_compressed(output / "teacher-data.npz", observations=observation_array, actions=action_array)
    learner = PPO("MlpPolicy", env, seed=seed, n_steps=256, batch_size=256, policy_kwargs={"net_arch": [128, 128]}, verbose=0)
    optimizer = torch.optim.Adam(learner.policy.parameters(), lr=.001)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for indices in np.array_split(rng.permutation(len(observation_array)), max(1, len(observation_array) // 256)):
            inputs = torch.tensor(observation_array[indices])
            labels = torch.tensor(action_array[indices])
            predicted = learner.policy.get_distribution(inputs).distribution.mean
            loss = (predicted - labels).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        losses.append({"epoch": epoch + 1, "action_mse": float(np.mean(epoch_losses))})
    learner.save(output / "bc")
    write_json(output / "bc-training.json", {"source_hashes": sources(), "episodes": episodes,
               "samples": len(actions), "epochs": epochs, "losses": losses, "dataset_sha256": digest(output / "teacher-data.npz"),
               "policy_semantics": controller_semantics("bc")["summary"],
               "control_semantics": controller_semantics("bc"),
               "payload_curriculum": CURRICULUM[case_id],
               "claim": "scheduler-conditioned supervised full-action diagnostic; no autonomous planning or task success inferred from MSE"})
    env.close()
    return learner


def train(case_id, output, timesteps, seed):
    output.mkdir(parents=True, exist_ok=True)
    source_identity = sources()
    supplemental_identity = supplemental_provenance()
    if (output / "training.json").exists() or (output / "ppo-residual.zip").exists():
        raise ValueError("Use a fresh run directory; existing training evidence is immutable")
    torch.set_num_threads(1)
    env = DummyVecEnv([lambda: ResidualArmEnv(case_id, seed)])
    algorithm_config = {
        "algorithm": "stable_baselines3.PPO",
        "n_steps": 256,
        "n_envs": env.num_envs,
        "rollout_transitions_per_update": 256 * env.num_envs,
        "batch_size": 256,
        "n_epochs": 4,
        "learning_rate": 1e-4,
        "clip_range": .1,
        "gamma": .99,
        "gae_lambda": .95,
        "entropy_coef": .001,
        "value_coef": .5,
        "max_grad_norm": .5,
        "target_kl": .02,
        "policy_net_arch": [64, 64],
        "log_std_init": -3.5,
    }
    model = PPO(
        "MlpPolicy", env, seed=seed, n_steps=algorithm_config["n_steps"],
        batch_size=algorithm_config["batch_size"], n_epochs=algorithm_config["n_epochs"],
        learning_rate=algorithm_config["learning_rate"], gamma=algorithm_config["gamma"],
        gae_lambda=algorithm_config["gae_lambda"], clip_range=algorithm_config["clip_range"],
        ent_coef=algorithm_config["entropy_coef"], vf_coef=algorithm_config["value_coef"],
        max_grad_norm=algorithm_config["max_grad_norm"], target_kl=algorithm_config["target_kl"],
        policy_kwargs={"net_arch": algorithm_config["policy_net_arch"],
                       "log_std_init": algorithm_config["log_std_init"]},
        verbose=0,
    )
    initial = torch.cat([parameter.detach().flatten() for parameter in model.policy.parameters()]).clone()
    model.save(output / "initial-residual")
    training_started = time.monotonic()
    model.learn(total_timesteps=timesteps, callback=TraceCallback(output / "metrics.jsonl"))
    training_wall_seconds = time.monotonic() - training_started
    model.save(output / "ppo-residual")
    final = torch.cat([parameter.detach().flatten() for parameter in model.policy.parameters()])
    metadata = {"case_id": case_id, "train_seed": seed, "actual_env_steps": model.num_timesteps,
                "method": "residual PPO over authored IK/FSM; NOT an end-to-end learned planner",
                "residual_scale": RESIDUAL_SCALE, "normalization": "raw SI observations, no running normalizer",
                "source_hashes": source_identity, "model_sha256": arm.model_hash(case_id),
                "checkpoint_sha256": digest(output / "ppo-residual.zip"),
                "parameter_l2_change": float(torch.linalg.vector_norm(final - initial)),
                "observation_dim": env.observation_space.shape[0], "action_dim": env.action_space.shape[0],
                "hardware_enabled": False, "simulation_only": True,
                "requested_env_steps": timesteps,
                "optimizer_updates": int(model._n_updates),
                "training_wall_seconds": training_wall_seconds,
                "training_episodes": int(env.envs[0].episodes),
                "algorithm_config": algorithm_config,
                "runtime_versions": runtime_versions(),
                "repository": repository_provenance(),
                "supplemental_provenance": supplemental_identity,
                "control_semantics": controller_semantics("ppo_residual", model.num_timesteps),
                "payload_mass": nominal_payload_metadata(case_id)}
    archive = output / "sources"
    archive.mkdir(exist_ok=True)
    for name, path in source_paths().items():
        shutil.copy2(path, archive / SOURCE_ARCHIVE_NAMES[name])
    for name, path in supplemental_paths().items():
        if path.is_file():
            shutil.copy2(path, archive / SUPPLEMENTAL_ARCHIVE_NAMES[name])
    (output / "scene.xml").write_text(arm.make_xml(case_id))
    if sources() != source_identity:
        raise RuntimeError("Source files changed during training; checkpoint is unverified")
    write_json(output / "training.json", metadata)
    env.close()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["teacher", "bc", "train", "eval", "negative", "assets"])
    parser.add_argument("--case", choices=arm.CASES, default=arm.CASES[0])
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "development")
    parser.add_argument("--steps", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--seeds-per-config", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.operation == "assets":
        arm.write_assets(args.output)
        return
    if args.operation == "teacher":
        report = evaluate(args.case, "teacher", None, args.output / "teacher-evaluation.json", args.seeds_per_config)
    elif args.operation == "bc":
        learner = behavior_clone(args.case, args.output, seed=args.seed)
        report = evaluate(args.case, "bc", learner, args.output / "bc-evaluation.json", args.seeds_per_config)
    elif args.operation == "train":
        learner = train(args.case, args.output, args.steps, args.seed)
        report = {"trained": True, "steps": learner.num_timesteps, "evaluated": False}
    elif args.operation == "eval":
        metadata = json.loads((args.output / "training.json").read_text())
        validate_training_evidence(args.output, args.case, metadata)
        learner = PPO.load(args.output / "ppo-residual.zip")
        report = evaluate(args.case, "ppo_residual", learner, args.output / "evaluation.json", args.seeds_per_config)
        report["checkpoint_sha256"] = digest(args.output / "ppo-residual.zip")
        write_json(args.output / "evaluation.json", report)
    else:
        controllers = ["zero", "open_gripper"] if args.case != arm.CASES[0] else ["zero"]
        if args.case.startswith("arms-"):
            controllers.append("disable_right")
        reports = {controller: evaluate(args.case, controller, None, args.output / f"negative-{controller}.json", 1, 70000) for controller in controllers}
        report = {"passed": all(result["successes"] == 0 for result in reports.values()), "controllers": {key: value["successes"] for key, value in reports.items()}}
        write_json(args.output / "negative-controls.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
