from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


PIPELINE_VERSION = 1
ACTION_DIM = 15
OBSERVATION_DIM = 64
RESIDUAL_SCALE_RAD = 0.02


def _env_class():
    return importlib.import_module("microduck_arm_v1.env").IntegratedArmEnv


def _make_env(
    case: str,
    mode: str,
    max_steps: int,
    payload_kg: float,
):
    return _env_class()(
        case=case,
        mode=mode,
        max_steps=max_steps,
        payload_kg=payload_kg,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(
            json.dumps(_jsonable(value), sort_keys=True, allow_nan=False) + "\n"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance() -> dict[str, Any]:
    from .config import SPEC_PATH

    paths = {
        "spec": Path(SPEC_PATH).resolve(),
        "config": Path(
            importlib.import_module("microduck_arm_v1.config").__file__
        ).resolve(),
        "environment": Path(
            importlib.import_module("microduck_arm_v1.env").__file__
        ).resolve(),
        "learning": Path(__file__).resolve(),
        "model": Path(
            importlib.import_module("microduck_arm_v1.model").__file__
        ).resolve(),
    }
    return {
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "invalidation_rule": (
            "Any source/spec/environment SHA-256 change invalidates this run and "
            "requires recollection, retraining, export, and evaluation."
        ),
    }


def _validate_provenance(provenance: dict[str, Any] | None) -> None:
    if not provenance:
        raise ValueError("checkpoint metadata has no provenance")
    for name, record in provenance["files"].items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(
                f"{name} provenance changed; this checkpoint is invalid for reuse"
            )


def _versions() -> dict[str, str]:
    names = (
        "gymnasium",
        "mujoco",
        "numpy",
        "onnx",
        "onnxruntime",
        "stable-baselines3",
        "torch",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _is_success(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _default_action_encoding() -> tuple[np.ndarray, np.ndarray]:
    from .config import load_design

    design = load_design()
    ranges = np.asarray(design["arm"]["joint_ranges_rad"], dtype=np.float32)
    scales = np.concatenate(
        [
            np.full(10, design["control"]["pose_action_scale_rad"], np.float32),
            (ranges[:, 1] - ranges[:, 0]) / 2.0,
        ]
    )
    offsets = np.concatenate(
        [
            np.asarray(design["home_rad"][:10], dtype=np.float32),
            (ranges[:, 1] + ranges[:, 0]) / 2.0,
        ]
    )
    return scales, offsets


def _env_action_encoding(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    base = env.unwrapped
    scales = np.asarray(base.action_scales, dtype=np.float32)
    offsets = np.asarray(base.action_offsets, dtype=np.float32)
    if scales.shape != (ACTION_DIM,) or offsets.shape != (ACTION_DIM,):
        raise ValueError("environment action scales and offsets must have shape (15,)")
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("environment action scales must be finite and positive")
    if not np.isfinite(offsets).all():
        raise ValueError("environment action offsets must be finite")
    return scales, offsets


def _dataset_paths(out: str | Path) -> tuple[Path, Path]:
    path = Path(out)
    if path.suffix == ".npz":
        return path, path.with_suffix(".json")
    return path / "demonstrations.npz", path / "demonstrations.json"


def _bc_paths(out: str | Path) -> tuple[Path, Path, Path]:
    path = Path(out)
    if path.suffix in {".pt", ".pth"}:
        return path, path.with_suffix(".json"), path.with_name(path.stem + "-loss.jsonl")
    return path / "bc.pt", path / "bc.json", path / "bc-loss.jsonl"


def _ppo_paths(out: str | Path) -> tuple[Path, Path, Path, Path]:
    path = Path(out)
    if path.suffix == ".zip":
        return (
            path,
            path.with_suffix(".json"),
            path.with_name(path.stem + "-training.jsonl"),
            path.with_name(path.stem + "-episodes.jsonl"),
        )
    return (
        path / "ppo.zip",
        path / "ppo.json",
        path / "ppo-training.jsonl",
        path / "ppo-episodes.jsonl",
    )


def collect_demonstrations(
    case: str,
    mode: str,
    episodes: int,
    seed: int,
    out: str | Path,
    max_steps: int = 1500,
    payload_kg: float = 0.01,
    include_failed: bool = False,
) -> Path:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    dataset_path, metadata_path = _dataset_paths(out)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_records = []
    env = _make_env(case, mode, max_steps, payload_kg)
    try:
        action_scales, action_offsets = _env_action_encoding(env)
        for episode_index in range(episodes):
            observation, reset_info = env.reset(seed=seed + episode_index)
            episode_observations = []
            episode_actions = []
            total_reward = 0.0
            terminated = False
            truncated = False
            info: dict[str, Any] = dict(reset_info or {})
            steps = 0
            while not (terminated or truncated) and steps < max_steps:
                action = np.asarray(env.teacher_action(), dtype=np.float32)
                if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                    raise RuntimeError(
                        "teacher_action must return a finite normalized shape-(15,) action"
                    )
                observation_array = np.asarray(observation, dtype=np.float32)
                if observation_array.shape != (OBSERVATION_DIM,):
                    raise RuntimeError(
                        "IntegratedArmEnv observation must have shape (64,)"
                    )
                episode_observations.append(observation_array)
                episode_actions.append(action.copy())
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            success = _is_success(info.get("success"))
            selected = bool(success or include_failed)
            if selected:
                observations.extend(episode_observations)
                actions.extend(episode_actions)
            episode_records.append(
                {
                    "episode": episode_index,
                    "seed": seed + episode_index,
                    "steps": steps,
                    "return": total_reward,
                    "success": success,
                    "selected": selected,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "failure_reason": info.get("failure_reason"),
                    "diagnostics": info.get("diagnostics"),
                }
            )
    finally:
        env.close()
    metadata = {
        "artifact": "microduck_arm_v1_demonstrations",
        "pipeline_version": PIPELINE_VERSION,
        "case": case,
        "mode": mode,
        "episodes_requested": episodes,
        "episodes_successful": sum(row["success"] for row in episode_records),
        "episodes_selected": sum(row["selected"] for row in episode_records),
        "samples_selected": len(actions),
        "include_failed": include_failed,
        "selection_rule": (
            "successful episodes only"
            if not include_failed
            else "successful and failed episodes for imitation diagnostics"
        ),
        "seed": seed,
        "max_steps": max_steps,
        "payload_kg": payload_kg,
        "observation_schema": _observation_schema(),
        "action_schema": _action_schema(action_scales, action_offsets),
        "episodes": episode_records,
        "fixture_only": mode == "fixture",
        "free_mode_validated": False,
        "hardware_deployment_qualified": False,
        "provenance": _provenance(),
        "versions": _versions(),
    }
    _write_json(metadata_path, metadata)
    if not observations:
        raise RuntimeError(
            "No teacher episodes were selected; refusing to create a fake training "
            "dataset. Inspect demonstrations.json or set include_failed=True for an "
            "imitation-only diagnostic."
        )
    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32)
    if observation_array.ndim != 2:
        raise RuntimeError("environment observations must have a fixed vector shape")
    np.savez_compressed(
        dataset_path,
        observations=observation_array,
        actions=action_array,
    )
    metadata["observation_dim"] = int(observation_array.shape[1])
    metadata["action_dim"] = int(action_array.shape[1])
    metadata["dataset_sha256"] = _sha256(dataset_path)
    _write_json(metadata_path, metadata)
    return dataset_path


class BCPolicy(torch.nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int = ACTION_DIM,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = observation_dim
        for hidden_size in hidden_sizes:
            layers.extend((torch.nn.Linear(width, hidden_size), torch.nn.Tanh()))
            width = hidden_size
        layers.extend((torch.nn.Linear(width, action_dim), torch.nn.Tanh()))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


class NormalizedBCPolicy(torch.nn.Module):
    def __init__(
        self,
        policy: BCPolicy,
        observation_mean: torch.Tensor | np.ndarray,
        observation_std: torch.Tensor | np.ndarray,
    ):
        super().__init__()
        self.policy = policy
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_std",
            torch.as_tensor(observation_std, dtype=torch.float32),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.observation_mean) / self.observation_std
        return self.policy(normalized)


def train_bc(
    dataset_path: str | Path,
    out: str | Path,
    epochs: int,
    seed: int,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    dataset_path = Path(dataset_path)
    with np.load(dataset_path) as dataset:
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
    if observations.shape[1:] != (OBSERVATION_DIM,) or len(observations) == 0:
        raise ValueError("dataset observations must have shape (N, 64)")
    if actions.shape != (len(observations), ACTION_DIM):
        raise ValueError("dataset actions must have shape (N, 15)")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("dataset contains non-finite values")
    checkpoint_path, metadata_path, loss_path = _bc_paths(out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    loss_path.unlink(missing_ok=True)
    dataset_metadata_path = dataset_path.with_suffix(".json")
    dataset_metadata = (
        json.loads(dataset_metadata_path.read_text())
        if dataset_metadata_path.is_file()
        else None
    )
    _seed_everything(seed)
    torch.set_num_threads(1)
    observation_mean = observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = observations.std(axis=0, dtype=np.float64).astype(np.float32)
    observation_std = np.maximum(observation_std, 1e-6)
    normalized = (observations - observation_mean) / observation_std
    policy = BCPolicy(observations.shape[1])
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    batch_size = min(256, len(observations))
    started = time.monotonic()
    losses = []
    policy.train()
    for epoch in range(epochs):
        batch_losses = []
        permutation = rng.permutation(len(observations))
        for start in range(0, len(observations), batch_size):
            indices = permutation[start : start + batch_size]
            inputs = torch.from_numpy(normalized[indices])
            targets = torch.from_numpy(actions[indices])
            loss = torch.square(policy(inputs) - targets).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        row = {
            "epoch": epoch + 1,
            "action_mse": float(np.mean(batch_losses)),
        }
        losses.append(row)
        _append_jsonl(loss_path, row)
    policy.eval()
    checkpoint = {
        "artifact": "microduck_arm_v1_bc_checkpoint",
        "pipeline_version": PIPELINE_VERSION,
        "observation_dim": int(observations.shape[1]),
        "action_dim": ACTION_DIM,
        "hidden_sizes": [128, 128],
        "observation_mean": torch.from_numpy(observation_mean),
        "observation_std": torch.from_numpy(observation_std),
        "state_dict": policy.state_dict(),
        "seed": seed,
        "observation_schema": _observation_schema(),
        "action_schema": _action_schema(),
        "provenance": _provenance(),
    }
    torch.save(checkpoint, checkpoint_path)
    metadata = {
        "artifact": checkpoint["artifact"],
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "demonstration_selection": (
            {
                key: dataset_metadata.get(key)
                for key in (
                    "episodes_requested",
                    "episodes_successful",
                    "episodes_selected",
                    "samples_selected",
                    "include_failed",
                    "selection_rule",
                )
            }
            if dataset_metadata is not None
            else "dataset metadata unavailable"
        ),
        "samples": len(observations),
        "observation_dim": int(observations.shape[1]),
        "action_dim": ACTION_DIM,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "seed": seed,
        "loss_log": str(loss_path.resolve()),
        "initial_action_mse": losses[0]["action_mse"],
        "final_action_mse": losses[-1]["action_mse"],
        "training_wall_seconds": time.monotonic() - started,
        "normalization": "checkpointed per-dimension mean/std",
        "action_semantics": "per-joint affine normalized absolute target",
        "task_mastery_claimed": False,
        "fixture_only": (
            dataset_metadata.get("mode") == "fixture"
            if dataset_metadata is not None
            else None
        ),
        "free_mode_validated": False,
        "hardware_deployment_qualified": False,
        "provenance": checkpoint["provenance"],
        "versions": _versions(),
    }
    _write_json(metadata_path, metadata)
    return checkpoint_path


def load_bc(checkpoint_path: str | Path) -> NormalizedBCPolicy:
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if checkpoint.get("artifact") != "microduck_arm_v1_bc_checkpoint":
        raise ValueError("not a microduck_arm_v1 BC checkpoint")
    _validate_provenance(checkpoint.get("provenance"))
    policy = BCPolicy(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        tuple(checkpoint["hidden_sizes"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    return NormalizedBCPolicy(
        policy,
        checkpoint["observation_mean"],
        checkpoint["observation_std"],
    ).eval()


class TeacherResidualEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.action_space.shape != (ACTION_DIM,):
            raise ValueError("IntegratedArmEnv action space must have shape (15,)")
        self.action_scales, self.action_offsets = _env_action_encoding(env)
        self.residual_normalized_scale = (
            RESIDUAL_SCALE_RAD / self.action_scales
        ).astype(np.float32)

    def step(self, action):
        teacher = np.asarray(self.env.unwrapped.teacher_action(), dtype=np.float32)
        residual = np.asarray(action, dtype=np.float32)
        if teacher.shape != (ACTION_DIM,) or residual.shape != (ACTION_DIM,):
            raise ValueError("teacher and residual actions must have shape (15,)")
        composed = np.clip(
            teacher + self.residual_normalized_scale * residual,
            -1.0,
            1.0,
        )
        observation, reward, terminated, truncated, info = self.env.step(composed)
        info = dict(info)
        info["learning_wrapper"] = {
            "kind": "teacher_residual",
            "teacher_action": teacher,
            "raw_residual_action": residual,
            "composed_action": composed,
            "residual_scale_rad": RESIDUAL_SCALE_RAD,
            "residual_normalized_scale": self.residual_normalized_scale,
            "action_scales_rad": self.action_scales,
            "action_offsets_rad": self.action_offsets,
        }
        return observation, reward, terminated, truncated, info


class _PPOLogCallback(BaseCallback):
    def __init__(self, training_path: Path, episodes_path: Path):
        super().__init__()
        self.training_path = training_path
        self.episodes_path = episodes_path
        self.returns: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
        infos = self.locals["infos"]
        if self.returns is None:
            self.returns = np.zeros_like(rewards)
            self.lengths = np.zeros_like(rewards, dtype=np.int64)
        self.returns += rewards
        self.lengths += 1
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            _append_jsonl(
                self.episodes_path,
                {
                    "timesteps": int(self.num_timesteps),
                    "return": float(self.returns[index]),
                    "length": int(self.lengths[index]),
                    "success": _is_success(info.get("success")),
                    "failure_reason": info.get("failure_reason"),
                    "diagnostics": info.get("diagnostics"),
                },
            )
            self.returns[index] = 0.0
            self.lengths[index] = 0
        return True

    def _on_rollout_end(self) -> None:
        values = {
            key: float(value)
            for key, value in self.model.logger.name_to_value.items()
            if np.isscalar(value)
        }
        _append_jsonl(
            self.training_path,
            {
                "timesteps": int(self.num_timesteps),
                "elapsed_seconds": time.monotonic() - self.started,
                "metrics": values,
            },
        )


def _observation_schema() -> dict[str, Any]:
    return {
        "shape": [OBSERVATION_DIM],
        "fields": [
            {"name": "joint_position", "size": 15},
            {"name": "joint_velocity", "size": 15},
            {"name": "gravity_vector", "size": 3},
            {"name": "angular_velocity", "size": 3},
            {"name": "tcp_relative", "size": 3, "source": "simulation_truth"},
            {"name": "object_relative", "size": 3, "source": "simulation_truth"},
            {"name": "goal_relative", "size": 3, "source": "simulation_truth"},
            {"name": "command_velocity", "size": 3},
            {"name": "previous_action", "size": 15},
            {"name": "phase", "size": 1},
        ],
        "hardware_estimator_required": [
            "tcp_relative",
            "object_relative",
            "goal_relative",
        ],
        "hardware_deployment_qualified": False,
    }


def _action_schema(
    action_scales: np.ndarray | None = None,
    action_offsets: np.ndarray | None = None,
) -> dict[str, Any]:
    if action_scales is None or action_offsets is None:
        action_scales, action_offsets = _default_action_encoding()
    return {
        "shape": [ACTION_DIM],
        "range": [-1.0, 1.0],
        "semantics": "per-joint affine normalized absolute target",
        "denormalization": "target_rad = action_offsets_rad + action_scales_rad * action",
        "action_scales_rad": np.asarray(action_scales, dtype=np.float32).tolist(),
        "action_offsets_rad": np.asarray(action_offsets, dtype=np.float32).tolist(),
        "leg_encoding": "home_rad + 0.5 * action for indices 0:10",
        "arm_encoding": "full joint-range midpoint plus half-span times action",
        "gripper_range_rad": [
            float(action_offsets[-1] - action_scales[-1]),
            float(action_offsets[-1] + action_scales[-1]),
        ],
    }


def _wrapper_metadata(
    residual: bool,
    action_scales: np.ndarray,
    action_offsets: np.ndarray,
) -> dict[str, Any]:
    if not residual:
        return {
            "kind": "none",
            "raw_sb3_action_semantics": "normalized absolute command",
            "standalone_policy": True,
        }
    return {
        "kind": "teacher_residual",
        "raw_sb3_action_semantics": "normalized residual in [-1, 1]",
        "teacher_action_semantics": "normalized absolute command",
        "composition": "clip(teacher + residual_normalized_scale * raw_policy, -1, 1)",
        "residual_scale_rad": RESIDUAL_SCALE_RAD,
        "action_scales_rad": action_scales.tolist(),
        "action_offsets_rad": action_offsets.tolist(),
        "residual_normalized_scale": (
            RESIDUAL_SCALE_RAD / action_scales
        ).tolist(),
        "standalone_policy": False,
        "external_runtime_requirement": "IntegratedArmEnv.teacher_action",
    }


def train_ppo(
    case: str,
    mode: str,
    out: str | Path,
    total_timesteps: int,
    seed: int,
    max_steps: int = 1500,
    residual: bool = True,
    payload_kg: float = 0.01,
) -> Path:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    checkpoint_path, metadata_path, training_path, episodes_path = _ppo_paths(out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.unlink(missing_ok=True)
    episodes_path.unlink(missing_ok=True)
    _seed_everything(seed)
    torch.set_num_threads(1)
    episode_counter = {"value": 0}
    encoding: dict[str, np.ndarray] = {}

    def factory():
        base = _make_env(case, mode, max_steps, payload_kg)
        action_scales, action_offsets = _env_action_encoding(base)
        encoding["scales"] = action_scales
        encoding["offsets"] = action_offsets

        class SeededEnv(gym.Wrapper):
            def reset(self, **kwargs):
                if "seed" not in kwargs:
                    kwargs["seed"] = seed + episode_counter["value"]
                    episode_counter["value"] += 1
                return self.env.reset(**kwargs)

        seeded = SeededEnv(base)
        return TeacherResidualEnv(seeded) if residual else seeded

    vec_env = DummyVecEnv([factory])
    observation_dim = int(vec_env.observation_space.shape[0])
    if observation_dim != OBSERVATION_DIM:
        vec_env.close()
        raise RuntimeError("IntegratedArmEnv observation space must have shape (64,)")
    n_steps = max(8, min(128, total_timesteps))
    batch_size = n_steps
    algorithm_config = {
        "algorithm": "stable_baselines3.PPO",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": 2,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "policy_net_arch": [64, 64],
        "normalization": "raw observations; no running observation normalizer",
    }
    started = time.monotonic()
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            device="cpu",
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=algorithm_config["n_epochs"],
            learning_rate=algorithm_config["learning_rate"],
            gamma=algorithm_config["gamma"],
            gae_lambda=algorithm_config["gae_lambda"],
            policy_kwargs={"net_arch": algorithm_config["policy_net_arch"]},
            verbose=0,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=_PPOLogCallback(training_path, episodes_path),
            progress_bar=False,
        )
        _append_jsonl(
            training_path,
            {
                "timesteps": int(model.num_timesteps),
                "phase": "training_end",
                "metrics": {
                    key: float(value)
                    for key, value in model.logger.name_to_value.items()
                    if np.isscalar(value)
                },
            },
        )
        model.save(checkpoint_path)
        actual_timesteps = int(model.num_timesteps)
        optimizer_updates = int(model._n_updates)
    finally:
        vec_env.close()
    metadata = {
        "artifact": "microduck_arm_v1_ppo_checkpoint",
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "case": case,
        "mode": mode,
        "fixture_mode_is_bench_aid_only": mode == "fixture",
        "payload_kg": payload_kg,
        "max_steps": max_steps,
        "requested_timesteps": total_timesteps,
        "actual_timesteps": actual_timesteps,
        "optimizer_updates": optimizer_updates,
        "observation_dim": observation_dim,
        "action_dim": ACTION_DIM,
        "seed": seed,
        "algorithm_config": algorithm_config,
        "observation_schema": _observation_schema(),
        "action_schema": _action_schema(encoding["scales"], encoding["offsets"]),
        "learning_wrapper": _wrapper_metadata(
            residual, encoding["scales"], encoding["offsets"]
        ),
        "raw_sb3_evaluation": {
            "loader": "stable_baselines3.PPO.load",
            "call": "model.predict(observation, deterministic=True)",
            "returns": (
                "raw normalized residual; compose with teacher using learning_wrapper"
                if residual
                else "normalized absolute action"
            ),
        },
        "training_log": str(training_path.resolve()),
        "episode_log": str(episodes_path.resolve()),
        "training_wall_seconds": time.monotonic() - started,
        "task_mastery_claimed": False,
        "success_evaluation": "not performed by train_ppo",
        "hardware_deployment_qualified": False,
        "free_mode_validated": False,
        "provenance": _provenance(),
        "versions": _versions(),
    }
    _write_json(metadata_path, metadata)
    return checkpoint_path


def evaluate_sb3_checkpoint(
    checkpoint_path: str | Path,
    observations: np.ndarray,
) -> np.ndarray:
    model = PPO.load(Path(checkpoint_path), device="cpu")
    actions, _ = model.predict(
        np.asarray(observations, dtype=np.float32),
        deterministic=True,
    )
    return np.asarray(actions, dtype=np.float32)


def evaluate_policy(
    checkpoint_path: str | Path,
    case: str,
    mode: str,
    seeds: list[int] | tuple[int, ...] | range,
    out: str | Path,
    max_steps: int = 1500,
    payload_kg: float = 0.01,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _checkpoint_metadata(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (RuntimeError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        checkpoint = None
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get("artifact") == "microduck_arm_v1_bc_checkpoint"
    ):
        _validate_provenance(checkpoint.get("provenance"))
        controller = "behavior_cloning"
        policy = load_bc(checkpoint_path)
        wrapper = {"kind": "none"}
    else:
        _validate_provenance(metadata.get("provenance"))
        controller = "ppo"
        policy = PPO.load(checkpoint_path, device="cpu")
        wrapper = metadata.get("learning_wrapper", {"kind": "none"})
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("evaluation seeds must not be empty")
    records = []
    for seed in seed_values:
        env = _make_env(case, mode, max_steps, payload_kg)
        try:
            observation, _ = env.reset(seed=seed)
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                if controller == "behavior_cloning":
                    with torch.no_grad():
                        action = policy(
                            torch.from_numpy(
                                np.asarray(observation, dtype=np.float32)[None]
                            )
                        ).numpy()[0]
                else:
                    action, _ = policy.predict(
                        np.asarray(observation, dtype=np.float32),
                        deterministic=True,
                    )
                    action = np.asarray(action, dtype=np.float32)
                    if wrapper.get("kind") == "teacher_residual":
                        action = np.clip(
                            np.asarray(env.teacher_action(), dtype=np.float32)
                            + np.asarray(
                                wrapper["residual_normalized_scale"],
                                dtype=np.float32,
                            )
                            * action,
                            -1.0,
                            1.0,
                        )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            records.append(
                {
                    "seed": seed,
                    "success": _is_success(info.get("success")),
                    "failure_reason": info.get("failure_reason"),
                    "return": total_reward,
                    "steps": steps,
                    "diagnostics": info.get("diagnostics"),
                }
            )
        finally:
            env.close()
    successes = sum(row["success"] for row in records)
    report = {
        "artifact": "microduck_arm_v1_policy_evaluation",
        "pipeline_version": PIPELINE_VERSION,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "controller": controller,
        "case": case,
        "mode": mode,
        "fixture_only": mode == "fixture",
        "free_mode_validated": False,
        "hardware_deployment_qualified": False,
        "seeds": seed_values,
        "episodes": len(records),
        "successes": successes,
        "success_rate": successes / len(records),
        "claim_scope": (
            "Observed held-out fixture simulation episodes only; no free-mode, "
            "unseen-distribution, or hardware success claim."
        ),
        "records": records,
        "provenance": _provenance(),
        "versions": _versions(),
    }
    _write_json(output_path, report)
    return output_path


class _DeterministicPPOActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            observation, self.policy.features_extractor
        )
        latent = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent), -1.0, 1.0)


def _checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path.with_suffix(".json")
    if metadata_path.is_file():
        return json.loads(metadata_path.read_text())
    return {}


def export_policy(checkpoint_path: str | Path, out: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(out)
    if output_path.suffix != ".onnx":
        output_path = output_path / "policy.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = _checkpoint_metadata(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (RuntimeError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        checkpoint = None
    if (
        isinstance(checkpoint, dict)
        and checkpoint.get("artifact") == "microduck_arm_v1_bc_checkpoint"
    ):
        _validate_provenance(checkpoint.get("provenance"))
        actor = load_bc(checkpoint_path).cpu().eval()
        observation_dim = int(checkpoint["observation_dim"])
        export_metadata = {
            "source": "behavior_cloning",
            "action_semantics": "normalized absolute command",
            "standalone_policy": True,
            "normalization": {
                "kind": "mean_std",
                "embedded": True,
                "mean": checkpoint["observation_mean"].tolist(),
                "std": checkpoint["observation_std"].tolist(),
            },
            "action_schema": checkpoint["action_schema"],
            "observation_schema": checkpoint["observation_schema"],
            "hardware_deployment_qualified": False,
            "provenance": checkpoint["provenance"],
        }
    else:
        _validate_provenance(checkpoint_metadata.get("provenance"))
        model = PPO.load(checkpoint_path, device="cpu")
        actor = _DeterministicPPOActor(model.policy).cpu().eval()
        observation_dim = int(model.observation_space.shape[0])
        wrapper = checkpoint_metadata.get("learning_wrapper", {"kind": "unknown"})
        export_metadata = {
            "source": "stable_baselines3.PPO",
            "action_semantics": (
                "normalized residual"
                if wrapper.get("kind") == "teacher_residual"
                else "normalized absolute command"
            ),
            "standalone_policy": wrapper.get("kind") != "teacher_residual",
            "normalization": {"kind": "identity", "embedded": True},
            "action_schema": checkpoint_metadata.get(
                "action_schema", _action_schema()
            ),
            "observation_schema": checkpoint_metadata.get(
                "observation_schema", _observation_schema()
            ),
            "learning_wrapper": wrapper,
            "hardware_deployment_qualified": False,
            "provenance": checkpoint_metadata["provenance"],
        }
    sample = torch.zeros(1, observation_dim, dtype=torch.float32)
    torch.onnx.export(
        actor,
        sample,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    graph = onnx.load(output_path)
    metadata_item = graph.metadata_props.add()
    metadata_item.key = "microduck_arm_v1"
    metadata_item.value = json.dumps(
        _jsonable(export_metadata), sort_keys=True, separators=(",", ":")
    )
    onnx.save(graph, output_path)
    rng = np.random.default_rng(20260913)
    observations = rng.normal(
        0.0, 0.25, size=(8, observation_dim)
    ).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(observations)).numpy()
    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    actual = session.run(
        None, {session.get_inputs()[0].name: observations}
    )[0]
    max_abs_error = float(np.max(np.abs(expected - actual)))
    if max_abs_error > 1e-5:
        raise RuntimeError(
            f"ONNX parity failed with max absolute error {max_abs_error}"
        )
    sidecar = Path(str(output_path) + ".json")
    _write_json(
        sidecar,
        {
            "artifact": "microduck_arm_v1_onnx_policy",
            "pipeline_version": PIPELINE_VERSION,
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "observation_dim": observation_dim,
            "action_dim": ACTION_DIM,
            "input_name": "observation",
            "output_name": "action",
            "parity_max_abs_error": max_abs_error,
            **export_metadata,
            "versions": _versions(),
        },
    )
    return output_path
