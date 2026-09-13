from __future__ import annotations

import argparse
import importlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import numpy as np
import torch

from microduck_arm_v1.learning import (
    BCPolicy as ArmPolicy,
    NormalizedBCPolicy as NormalizedArmPolicy,
    _seed_everything,
    _sha256,
    _versions,
)

PIPELINE_VERSION = 1
OBSERVATION_DIM = 64
ACTION_DIM = 15
ARM_START = 10
ARM_DIM = 5
ARM_ACTION_INDICES = np.arange(ARM_START, ACTION_DIM)
ACTION_MASK = np.r_[np.zeros(ARM_START), np.ones(ARM_DIM)].astype(np.float32)
MAX_RESIDUAL_RAD = 0.02
DEFAULT_CASE = "reach"
DEFAULT_MODE = "fixture"
DEFAULT_CANDIDATE = "A"
DEFAULT_MAX_STEPS = 2500
DEFAULT_PAYLOAD_KG = 0.01
MAX_RUNNER_EPISODES = 64
MAX_RUNNER_EPOCHS = 500
MAX_RUNNER_PPO_TIMESTEPS = 1_000_000
MAX_RUNNER_STEPS_PER_EPISODE = 5000


def _env_module():
    return importlib.import_module("microduck_arm_v1c.env")


def _env_class():
    return _env_module().IntegratedArmEnv


def _require_env_frozen() -> None:
    if getattr(_env_module(), "ENV_FROZEN", False) is not True:
        raise RuntimeError(
            "microduck_arm_v1c.env.ENV_FROZEN must be True before collecting "
            "training data or running optimization"
        )


def make_env(
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> gym.Env:
    if candidate not in {"A", "B"}:
        raise ValueError("candidate must be 'A' or 'B'")
    return _env_class()(
        case=case,
        mode=mode,
        candidate=candidate,
        max_steps=max_steps,
        payload_kg=payload_kg,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(item) for item in list(value)]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return "-inf" if value < 0 else "inf"
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_jsonable(value), sort_keys=True) + "\n")


def _module_path(name: str) -> Path | None:
    try:
        path = Path(importlib.import_module(name).__file__).resolve()
        return path if path.is_file() else None
    except (ImportError, TypeError):
        return None


def source_provenance() -> dict[str, Any]:
    sources = {"v1c_learning": Path(__file__).resolve()}
    modules = {
        "v1c_env": "microduck_arm_v1c.env",
        "v1c_config": "microduck_arm_v1c.config",
        "v1c_model": "microduck_arm_v1c.model",
        "v1c_locomotion": "microduck_arm_v1c.locomotion",
        "microduck_local_contract_dependency": "microduck_local.contract",
        "v1b_env_dependency": "microduck_arm_v1.env",
        "v1b_config_dependency": "microduck_arm_v1.config",
        "v1b_learning_dependency": "microduck_arm_v1.learning",
        "v1b_model_dependency": "microduck_arm_v1.model",
    }
    for label, name in modules.items():
        path = _module_path(name)
        if path:
            sources[label] = path
    for family, name in (
        ("v1c", "microduck_arm_v1c.config"),
        ("v1b_dependency", "microduck_arm_v1.config"),
    ):
        try:
            config = importlib.import_module(name)
            spec = Path(config.SPEC_PATH).resolve()
            design = json.loads(spec.read_text())
            sources[f"{family}_spec"] = spec
            sources[f"{family}_source_model"] = Path(
                config.ROOT, design["source_model"]
            ).resolve()
        except (AttributeError, ImportError, KeyError, OSError, ValueError):
            pass
    contract_path = _module_path("microduck_local.contract")
    if contract_path is None:
        raise RuntimeError("microduck_local.contract is required for v1C provenance")
    policy_directory = contract_path.parents[2] / "policies"
    for name in ("alpha_stand.onnx", "alpha_walking.onnx"):
        policy_path = (policy_directory / name).resolve()
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"required shipped locomotion policy is missing: {policy_path}"
            )
        sources[f"shipped_policy_{policy_path.stem}"] = policy_path
    return {
        "files": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in sorted(sources.items())
        },
        "required_families": [
            "microduck_arm_v1c",
            "microduck_arm_v1",
            "microduck_local.contract",
            "shipped_legacy_locomotion_policies",
        ],
        "invalidation_rule": (
            "Any recorded v1C source, v1B dependency, legacy contract, or shipped "
            "locomotion policy hash change invalidates continued training, export, "
            "and evaluation."
        ),
    }


def _validate_provenance(provenance: dict[str, Any] | None) -> None:
    if not provenance or not provenance.get("files"):
        raise ValueError("artifact has no source provenance")
    for label, record in provenance["files"].items():
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"{label} source changed; artifact reuse is invalid")


def _assert_provenance_unchanged(
    initial: dict[str, Any],
    operation: str,
) -> None:
    current = source_provenance()
    if current != initial:
        initial_files = initial.get("files", {})
        current_files = current.get("files", {})
        changed = sorted(
            label
            for label in set(initial_files) | set(current_files)
            if initial_files.get(label) != current_files.get(label)
        )
        detail = ", ".join(changed) if changed else "provenance schema"
        raise RuntimeError(
            f"source provenance changed during {operation}: {detail}; "
            "discard this run and restart from a new output directory"
        )


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _action_encoding(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    scales = _array(env.unwrapped.action_scales, (ACTION_DIM,), "action_scales")
    offsets = _array(env.unwrapped.action_offsets, (ACTION_DIM,), "action_offsets")
    if np.any(scales <= 0):
        raise ValueError("action_scales must be positive")
    return scales, offsets


def normalize_targets(env: gym.Env, targets_rad: np.ndarray) -> np.ndarray:
    targets = _array(targets_rad, (ACTION_DIM,), "targets_rad")
    normalizer = getattr(env.unwrapped, "normalize_targets", None)
    if callable(normalizer):
        return _array(normalizer(targets), (ACTION_DIM,), "normalized action")
    scales, offsets = _action_encoding(env)
    return np.clip((targets - offsets) / scales, -1, 1).astype(np.float32)


def denormalize_action(env: gym.Env, action: np.ndarray) -> np.ndarray:
    action = _array(action, (ACTION_DIM,), "action")
    if np.any(np.abs(action) > 1.000001):
        raise ValueError("action must be within [-1, 1]")
    denormalizer = getattr(env.unwrapped, "denormalize_action", None)
    if callable(denormalizer):
        return _array(
            denormalizer(action), (ACTION_DIM,), "denormalized targets"
        )
    scales, offsets = _action_encoding(env)
    return offsets + scales * action


def compose_arm_action(teacher_action: np.ndarray, arm_action: np.ndarray) -> np.ndarray:
    result = _array(teacher_action, (ACTION_DIM,), "teacher_action").copy()
    result[ARM_START:] = np.clip(
        _array(arm_action, (ARM_DIM,), "arm_action"), -1, 1
    )
    return result


def compose_residual_action(
    teacher_action: np.ndarray,
    residual_action: np.ndarray,
    action_scales: np.ndarray,
    max_residual_rad: float = MAX_RESIDUAL_RAD,
) -> np.ndarray:
    result = _array(teacher_action, (ACTION_DIM,), "teacher_action").copy()
    residual = np.asarray(residual_action, dtype=np.float32)
    arm_residual = residual if residual.shape == (ARM_DIM,) else residual[ARM_START:]
    if residual.shape not in {(ARM_DIM,), (ACTION_DIM,)} or not np.isfinite(
        arm_residual
    ).all():
        raise ValueError("residual_action must be finite with shape (5,) or (15,)")
    if not np.isfinite(max_residual_rad) or max_residual_rad <= 0:
        raise ValueError("max_residual_rad must be finite and positive")
    scales = _array(action_scales, (ACTION_DIM,), "action_scales")
    result[ARM_START:] = np.clip(
        result[ARM_START:]
        + np.clip(arm_residual, -1, 1) * max_residual_rad / scales[ARM_START:],
        -1,
        1,
    )
    return result


def split_episode_ids(
    episode_ids: np.ndarray, validation_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(episode_ids)
    unique = np.unique(ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("episode_ids must be a non-empty vector")
    if len(unique) < 2:
        raise ValueError("at least two complete episodes are required")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    return np.sort(shuffled[count:]), np.sort(shuffled[:count])


def split_masks(
    episode_ids: np.ndarray, validation_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_ids, validation_ids = split_episode_ids(
        episode_ids, validation_fraction, seed
    )
    train = np.isin(episode_ids, train_ids)
    validation = np.isin(episode_ids, validation_ids)
    if np.any(train & validation) or not np.all(train | validation):
        raise RuntimeError("episode split leaked or dropped frames")
    return train, validation, {
        "unit": "whole_episode",
        "frame_leakage": False,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_episode_ids": train_ids.tolist(),
        "validation_episode_ids": validation_ids.tolist(),
        "train_samples": int(train.sum()),
        "validation_samples": int(validation.sum()),
    }


def weighted_arm_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    prediction = prediction[..., ARM_START:] if prediction.shape[-1] == ACTION_DIM else prediction
    target = target[..., ARM_START:] if target.shape[-1] == ACTION_DIM else target
    if prediction.shape != target.shape or prediction.shape[-1] != ARM_DIM:
        raise ValueError("prediction and target must align on five arm dimensions")
    weight = torch.ones(ARM_DIM, dtype=prediction.dtype, device=prediction.device)
    if weights is not None:
        weight = torch.as_tensor(weights, dtype=prediction.dtype, device=prediction.device)
        if weight.shape != (ARM_DIM,) or torch.any(weight <= 0):
            raise ValueError("arm loss weights must be five positive values")
    return torch.mean(torch.square(prediction - target) * weight / weight.mean())


def _space_schema(space: gym.Space) -> dict[str, Any]:
    if not isinstance(space, gym.spaces.Box):
        raise ValueError("v1C learning requires Box spaces")
    return {
        "shape": list(space.shape),
        "dtype": str(space.dtype),
        "low": space.low,
        "high": space.high,
    }


def _environment_schema(env: gym.Env) -> dict[str, Any]:
    scales, offsets = _action_encoding(env)
    return {
        "observation": _space_schema(env.observation_space),
        "public_action": _space_schema(env.action_space),
        "arm_policy_output": {
            "shape": [ARM_DIM],
            "indices": ARM_ACTION_INDICES,
            "range": [-1, 1],
        },
        "action_scales_rad": scales,
        "action_offsets_rad": offsets,
        "denormalization": "target_rad = offset_rad + scale_rad * normalized_action",
        "dt_s": float(env.unwrapped.dt),
        "hardware_truth_estimator_claimed": False,
    }


def _paths(out: str | Path, kind: str) -> tuple[Path, Path, Path | None]:
    path = Path(out)
    suffix = {"data": ".npz", "bc": ".pt", "ppo": ".zip"}[kind]
    stem = {"data": "demonstrations", "bc": "bc", "ppo": "ppo"}[kind]
    artifact = path if path.suffix == suffix else path / f"{stem}{suffix}"
    metadata = artifact.with_suffix(".json")
    log = None if kind == "data" else artifact.with_name(f"{stem}-training.jsonl")
    return artifact, metadata, log


def _collect(
    *,
    out: str | Path,
    episodes: int,
    seed: int,
    case: str,
    mode: str,
    candidate: str,
    max_steps: int,
    payload_kg: float,
    behavior: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    collection: str,
) -> Path:
    _require_env_frozen()
    initial_provenance = source_provenance()
    if episodes < 1:
        raise ValueError("episodes must be positive")
    dataset_path, metadata_path, _ = _paths(out, "data")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    observations, labels, actions, episode_ids, records = [], [], [], [], []
    env = make_env(case, mode, candidate, max_steps, payload_kg)
    try:
        schema = _environment_schema(env)
        for episode_id in range(episodes):
            observation, _ = env.reset(seed=seed + episode_id)
            terminated = truncated = False
            total_reward, steps, info = 0.0, 0, {}
            while not (terminated or truncated):
                observation = _array(observation, (OBSERVATION_DIM,), "observation")
                teacher = _array(
                    env.unwrapped.teacher_action(), (ACTION_DIM,), "teacher_action"
                )
                action = teacher if behavior is None else behavior(observation, teacher)
                action = _array(action, (ACTION_DIM,), "behavior action")
                observations.append(observation.copy())
                labels.append(teacher.copy())
                actions.append(action.copy())
                episode_ids.append(episode_id)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            records.append(
                {
                    "episode_id": episode_id,
                    "seed": seed + episode_id,
                    "steps": steps,
                    "return": total_reward,
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                }
            )
    finally:
        env.close()
    _assert_provenance_unchanged(initial_provenance, collection)
    np.savez_compressed(
        dataset_path,
        observations=np.asarray(observations, np.float32),
        teacher_actions=np.asarray(labels, np.float32),
        actions=np.asarray(labels, np.float32),
        behavior_actions=np.asarray(actions, np.float32),
        episode_ids=np.asarray(episode_ids, np.int64),
    )
    _write_json(
        metadata_path,
        {
            "artifact": "microduck_arm_v1c_demonstrations",
            "pipeline_version": PIPELINE_VERSION,
            "collection": collection,
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "max_steps": max_steps,
            "payload_kg": payload_kg,
            "episodes": records,
            "samples": len(observations),
            "teacher_labels": "teacher_action evaluated on each visited state before step",
            "goal_teleportation": False,
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "environment_schema": schema,
            "dataset_sha256": _sha256(dataset_path),
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_collection",
            "versions": _versions(),
        },
    )
    return dataset_path


def collect_demonstrations(
    out: str | Path,
    episodes: int,
    seed: int,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    return _collect(
        out=out,
        episodes=episodes,
        seed=seed,
        case=case,
        mode=mode,
        candidate=candidate,
        max_steps=max_steps,
        payload_kg=payload_kg,
        behavior=None,
        collection="teacher",
    )


def _dataset_metadata_path(dataset_path: Path) -> Path:
    return dataset_path.with_suffix(".json")


def _validated_dataset(
    dataset_path: str | Path,
    *,
    require_metadata: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as dataset:
        required = {"observations", "episode_ids"}
        missing = required.difference(dataset.files)
        if missing:
            raise ValueError(f"dataset is missing arrays: {sorted(missing)}")
        label_key = "teacher_actions" if "teacher_actions" in dataset else "actions"
        if label_key not in dataset:
            raise ValueError("dataset is missing teacher_actions")
        observations = np.asarray(dataset["observations"], np.float32)
        teacher_actions = np.asarray(dataset[label_key], np.float32)
        behavior_actions = np.asarray(
            dataset["behavior_actions"]
            if "behavior_actions" in dataset
            else teacher_actions,
            np.float32,
        )
        episode_ids = np.asarray(dataset["episode_ids"], np.int64)
    sample_count = len(observations)
    if observations.shape != (sample_count, OBSERVATION_DIM):
        raise ValueError("observations must have shape (N, 64)")
    if teacher_actions.shape != (sample_count, ACTION_DIM):
        raise ValueError("teacher actions must have shape (N, 15)")
    if behavior_actions.shape != (sample_count, ACTION_DIM):
        raise ValueError("behavior actions must have shape (N, 15)")
    if episode_ids.shape != (sample_count,):
        raise ValueError("episode_ids must have shape (N,)")
    if sample_count == 0:
        raise ValueError("dataset must contain samples")
    if not (
        np.isfinite(observations).all()
        and np.isfinite(teacher_actions).all()
        and np.isfinite(behavior_actions).all()
    ):
        raise ValueError("dataset arrays must be finite")

    metadata_path = _dataset_metadata_path(path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if require_metadata and metadata is None:
        raise ValueError(f"dataset metadata is required: {metadata_path}")
    if metadata is not None:
        if metadata.get("dataset_sha256") != _sha256(path):
            raise RuntimeError(f"dataset hash does not match metadata: {path}")
        _validate_provenance(metadata.get("source_provenance"))
    return {
        "observations": observations,
        "teacher_actions": teacher_actions,
        "behavior_actions": behavior_actions,
        "episode_ids": episode_ids,
    }, metadata


def aggregate_datasets(
    dataset_paths: Iterable[str | Path],
    out: str | Path,
) -> Path:
    initial_provenance = source_provenance()
    paths = [Path(path) for path in dataset_paths]
    if len(paths) < 2:
        raise ValueError("dataset aggregation requires at least two inputs")
    output_path, metadata_path, _ = _paths(out, "data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged = {
        "observations": [],
        "teacher_actions": [],
        "behavior_actions": [],
        "episode_ids": [],
    }
    source_records = []
    next_episode_id = 0
    for source_index, path in enumerate(paths):
        arrays, metadata = _validated_dataset(path, require_metadata=True)
        source_episode_ids = np.unique(arrays["episode_ids"])
        episode_map = {
            int(source_id): next_episode_id + offset
            for offset, source_id in enumerate(source_episode_ids.tolist())
        }
        remapped_ids = np.asarray(
            [episode_map[int(source_id)] for source_id in arrays["episode_ids"]],
            dtype=np.int64,
        )
        for key in ("observations", "teacher_actions", "behavior_actions"):
            merged[key].append(arrays[key])
        merged["episode_ids"].append(remapped_ids)
        source_records.append(
            {
                "source_index": source_index,
                "path": str(path.resolve()),
                "dataset_sha256": _sha256(path),
                "metadata_sha256": _sha256(_dataset_metadata_path(path)),
                "collection": metadata.get("collection"),
                "samples": len(arrays["observations"]),
                "episode_id_map": {
                    str(source_id): merged_id
                    for source_id, merged_id in episode_map.items()
                },
                "source_provenance": metadata["source_provenance"],
            }
        )
        next_episode_id += len(source_episode_ids)
    _assert_provenance_unchanged(initial_provenance, "dataset aggregation")
    arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in merged.items()
    }
    np.savez_compressed(
        output_path,
        observations=arrays["observations"],
        teacher_actions=arrays["teacher_actions"],
        actions=arrays["teacher_actions"],
        behavior_actions=arrays["behavior_actions"],
        episode_ids=arrays["episode_ids"],
    )
    _write_json(
        metadata_path,
        {
            "artifact": "microduck_arm_v1c_aggregated_demonstrations",
            "pipeline_version": PIPELINE_VERSION,
            "collection": "dataset_aggregation",
            "samples": len(arrays["observations"]),
            "episodes": next_episode_id,
            "episode_ids_remapped": True,
            "source_datasets": source_records,
            "dataset_sha256": _sha256(output_path),
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_aggregation",
            "versions": _versions(),
        },
    )
    return output_path


class ArmActionAdapter:
    """Expands five learned arm values into the simulator's public 15 actions."""

    def __init__(self, env: gym.Env):
        self.env = env

    def absolute(self, arm_action: np.ndarray) -> np.ndarray:
        return compose_arm_action(self.env.unwrapped.teacher_action(), arm_action)

    def residual(self, arm_residual: np.ndarray) -> np.ndarray:
        scales, _ = _action_encoding(self.env)
        return compose_residual_action(
            self.env.unwrapped.teacher_action(), arm_residual, scales
        )


def train_bc(
    dataset_path: str | Path,
    out: str | Path,
    epochs: int,
    seed: int,
    validation_fraction: float = 0.2,
    arm_loss_weights: Iterable[float] | None = None,
) -> Path:
    _require_env_frozen()
    initial_provenance = source_provenance()
    if epochs < 1:
        raise ValueError("epochs must be positive")
    dataset_path = Path(dataset_path)
    with np.load(dataset_path) as dataset:
        observations = np.asarray(dataset["observations"], np.float32)
        key = "teacher_actions" if "teacher_actions" in dataset else "actions"
        labels = np.asarray(dataset[key], np.float32)
        episode_ids = np.asarray(dataset["episode_ids"])
    if observations.shape != (len(observations), OBSERVATION_DIM):
        raise ValueError("observations must have shape (N, 64)")
    if labels.shape != (len(observations), ACTION_DIM):
        raise ValueError("teacher actions must have shape (N, 15)")
    weights = np.ones(ARM_DIM, np.float32)
    if arm_loss_weights is not None:
        weights = _array(list(arm_loss_weights), (ARM_DIM,), "arm_loss_weights")
        if np.any(weights <= 0):
            raise ValueError("arm_loss_weights must be positive")
    train, validation, split = split_masks(episode_ids, validation_fraction, seed)
    mean = observations[train].mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(
        observations[train].std(0, dtype=np.float64).astype(np.float32), 1e-6
    )
    normalized = (observations - mean) / std
    checkpoint_path, metadata_path, log_path = _paths(out, "bc")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    _seed_everything(seed)
    torch.set_num_threads(1)
    policy = ArmPolicy(OBSERVATION_DIM, ARM_DIM)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    indices = np.flatnonzero(train)
    batch_size = min(256, len(indices))
    started = time.monotonic()
    for epoch in range(epochs):
        losses = []
        shuffled = rng.permutation(indices)
        policy.train()
        for start in range(0, len(indices), batch_size):
            batch = shuffled[start : start + batch_size]
            loss = weighted_arm_mse(
                policy(torch.from_numpy(normalized[batch])),
                torch.from_numpy(labels[batch, ARM_START:]),
                weights,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        policy.eval()
        with torch.no_grad():
            val_loss = weighted_arm_mse(
                policy(torch.from_numpy(normalized[validation])),
                torch.from_numpy(labels[validation, ARM_START:]),
                weights,
            )
        _append_jsonl(
            log_path,
            {
                "epoch": epoch + 1,
                "raw_train_arm_mse": np.mean(losses),
                "raw_validation_arm_mse": float(val_loss),
            },
        )
    _assert_provenance_unchanged(initial_provenance, "behavior cloning")
    checkpoint = {
        "artifact": "microduck_arm_v1c_bc_checkpoint",
        "pipeline_version": PIPELINE_VERSION,
        "observation_dim": OBSERVATION_DIM,
        "arm_action_dim": ARM_DIM,
        "hidden_sizes": [128, 128],
        "observation_mean": torch.from_numpy(mean),
        "observation_std": torch.from_numpy(std),
        "state_dict": policy.state_dict(),
        "arm_loss_weights": weights.tolist(),
        "split_provenance": split,
        "source_provenance": initial_provenance,
        "source_provenance_captured": "before_training",
    }
    torch.save(checkpoint, checkpoint_path)
    _write_json(
        metadata_path,
        {
            "artifact": checkpoint["artifact"],
            "checkpoint_sha256": _sha256(checkpoint_path),
            "dataset_sha256": _sha256(dataset_path),
            "epochs": epochs,
            "seed": seed,
            "loss_active_indices": ARM_ACTION_INDICES,
            "leg_targets_used_in_loss": False,
            "arm_loss_weights": weights,
            "split_provenance": split,
            "output_contract": "five normalized arm actions; explicit adapter required",
            "normalization_embedded_on_export": True,
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "training_wall_seconds": time.monotonic() - started,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_training",
            "versions": _versions(),
        },
    )
    return checkpoint_path


def load_bc(checkpoint_path: str | Path) -> NormalizedArmPolicy:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("artifact") != "microduck_arm_v1c_bc_checkpoint":
        raise ValueError("not a v1C behavior-cloning checkpoint")
    _validate_provenance(checkpoint.get("source_provenance"))
    policy = ArmPolicy(
        int(checkpoint["observation_dim"]),
        int(checkpoint["arm_action_dim"]),
        tuple(checkpoint["hidden_sizes"]),
    )
    policy.load_state_dict(checkpoint["state_dict"])
    return NormalizedArmPolicy(
        policy, checkpoint["observation_mean"], checkpoint["observation_std"]
    ).eval()


def collect_dagger(
    checkpoint_path: str | Path,
    out: str | Path,
    episodes: int,
    seed: int,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    policy = load_bc(checkpoint_path)

    def behavior(observation: np.ndarray, teacher: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            arm = policy(torch.from_numpy(observation[None])).numpy()[0]
        return compose_arm_action(teacher, arm)

    return _collect(
        out=out,
        episodes=episodes,
        seed=seed,
        case=case,
        mode=mode,
        candidate=candidate,
        max_steps=max_steps,
        payload_kg=payload_kg,
        behavior=behavior,
        collection="dagger_on_policy_states",
    )


class TeacherResidualEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if env.action_space.shape != (ACTION_DIM,):
            raise ValueError("v1C public action space must have shape (15,)")
        self.action_scales, self.action_offsets = _action_encoding(env)
        self.action_mask = ACTION_MASK.copy()
        self.action_space = gym.spaces.Box(-1, 1, (ARM_DIM,), np.float32)

    def step(self, action):
        raw = _array(action, (ARM_DIM,), "residual action")
        teacher = _array(
            self.env.unwrapped.teacher_action(), (ACTION_DIM,), "teacher_action"
        )
        composed = compose_residual_action(teacher, raw, self.action_scales)
        observation, reward, terminated, truncated, info = self.env.step(composed)
        info = dict(info)
        info["learning_wrapper"] = {
            "kind": "teacher_residual_arm_mask",
            "action_mask": self.action_mask,
            "teacher_action": teacher,
            "raw_residual_action": raw,
            "composed_action": composed,
            "max_residual_rad_per_arm_joint": MAX_RESIDUAL_RAD,
        }
        return observation, reward, terminated, truncated, info


def _ppo_callback(path: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.raw_return = self.last_raw_return = 0.0

        def _on_step(self) -> bool:
            self.raw_return += float(np.asarray(self.locals["rewards"]).mean())
            return True

        def _on_rollout_end(self) -> None:
            self.last_raw_return = self.raw_return
            _append_jsonl(path, self.metrics())
            self.raw_return = 0.0

        def metrics(self) -> dict[str, Any]:
            values = self.model.logger.name_to_value
            return {
                "timesteps": int(self.num_timesteps),
                "raw_reward_sum": self.last_raw_return or self.raw_return,
                "raw_loss": values.get("train/loss"),
                "value_loss": values.get("train/value_loss"),
                "entropy_loss": values.get("train/entropy_loss"),
                "approx_kl": values.get("train/approx_kl"),
                "clip_fraction": values.get("train/clip_fraction"),
            }

    return Callback()


def train_ppo(
    out: str | Path,
    total_timesteps: int,
    seed: int,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    _require_env_frozen()
    initial_provenance = source_provenance()
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be positive")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    checkpoint_path, metadata_path, log_path = _paths(out, "ppo")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    _seed_everything(seed)
    torch.set_num_threads(1)
    details = {}

    def factory():
        env = make_env(case, mode, candidate, max_steps, payload_kg)
        details["schema"] = _environment_schema(env)
        return TeacherResidualEnv(env)

    vec_env = DummyVecEnv([factory])
    n_steps = max(8, min(128, total_timesteps))
    started = time.monotonic()
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            seed=seed,
            device="cpu",
            n_steps=n_steps,
            batch_size=n_steps,
            n_epochs=2,
            learning_rate=3e-4,
            policy_kwargs={"net_arch": [64, 64]},
            verbose=0,
        )
        callback = _ppo_callback(log_path)
        model.learn(total_timesteps, callback=callback, progress_bar=False)
        _append_jsonl(log_path, {"phase": "training_end", **callback.metrics()})
    finally:
        vec_env.close()
    _assert_provenance_unchanged(initial_provenance, "PPO training")
    model.save(checkpoint_path)
    _write_json(
        metadata_path,
        {
            "artifact": "microduck_arm_v1c_ppo_checkpoint",
            "checkpoint_sha256": _sha256(checkpoint_path),
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "max_steps": max_steps,
            "payload_kg": payload_kg,
            "requested_timesteps": total_timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "public_action_dim": ACTION_DIM,
            "policy_action_dim": ARM_DIM,
            "action_mask": ACTION_MASK,
            "leg_residuals_applied": False,
            "max_residual_rad_per_arm_joint": MAX_RESIDUAL_RAD,
            "environment_schema": details["schema"],
            "training_metrics": [
                "raw_reward_sum",
                "raw_loss",
                "value_loss",
                "entropy_loss",
                "approx_kl",
                "clip_fraction",
            ],
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "training_wall_seconds": time.monotonic() - started,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_training",
            "versions": _versions(),
        },
    )
    return checkpoint_path


class _PPOArmActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy.action_space.shape != (ARM_DIM,):
            raise ValueError("v1C PPO policy must have five actions")
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            observation, self.policy.features_extractor
        )
        latent = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent), -1, 1)


def _metadata(checkpoint_path: Path) -> dict[str, Any]:
    path = checkpoint_path.with_suffix(".json")
    return json.loads(path.read_text()) if path.is_file() else {}


def _load_artifact(checkpoint_path: Path):
    metadata = _metadata(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (RuntimeError, EOFError, ValueError, TypeError, pickle.UnpicklingError):
        checkpoint = None
    if isinstance(checkpoint, dict) and checkpoint.get("artifact") == "microduck_arm_v1c_bc_checkpoint":
        return "bc", load_bc(checkpoint_path), checkpoint["source_provenance"]
    from stable_baselines3 import PPO

    _validate_provenance(metadata.get("source_provenance"))
    return "ppo", PPO.load(checkpoint_path, device="cpu"), metadata["source_provenance"]


def export_policy(checkpoint_path: str | Path, out: str | Path) -> Path:
    initial_provenance = source_provenance()
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(out)
    if output_path.suffix != ".onnx":
        output_path /= "arm-policy.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kind, loaded, provenance = _load_artifact(checkpoint_path)
    actor = loaded if kind == "bc" else _PPOArmActor(loaded.policy).eval()
    semantics = (
        "normalized absolute arm targets"
        if kind == "bc"
        else "normalized arm residuals for explicit teacher adapter"
    )
    torch.onnx.export(
        actor,
        torch.zeros(1, OBSERVATION_DIM),
        output_path,
        input_names=["observation"],
        output_names=["arm_action"],
        dynamic_axes={"observation": {0: "batch"}, "arm_action": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    probe = np.random.default_rng(20260913).normal(
        0, 0.2, (8, OBSERVATION_DIM)
    ).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(probe)).numpy()
    actual = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    ).run(None, {"observation": probe})[0]
    if expected.shape != (8, ARM_DIM) or actual.shape != expected.shape:
        raise RuntimeError("ONNX export must produce exactly five arm actions")
    parity = float(np.max(np.abs(expected - actual)))
    if parity > 1e-5:
        raise RuntimeError(f"ONNX parity failed: max absolute error {parity}")
    _assert_provenance_unchanged(initial_provenance, "ONNX export")
    _write_json(
        Path(str(output_path) + ".json"),
        {
            "artifact": "microduck_arm_v1c_arm_onnx",
            "path": str(output_path.resolve()),
            "sha256": _sha256(output_path),
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source": "behavior_cloning" if kind == "bc" else "stable_baselines3.PPO",
            "input": {"name": "observation", "shape": ["batch", OBSERVATION_DIM]},
            "output": {"name": "arm_action", "shape": ["batch", ARM_DIM]},
            "action_indices": ARM_ACTION_INDICES,
            "action_semantics": semantics,
            "normalization": "mean/std embedded in model" if kind == "bc" else "identity",
            "explicit_adapter_required": True,
            "direct_legacy_or_hardware_compatible": False,
            "adapter_rule": (
                "Use the deterministic teacher command for the same state, then "
                "replace arm indices for BC or apply capped 0.02-rad PPO residuals."
            ),
            "parity_max_abs_error": parity,
            "hardware_deployment_qualified": False,
            "source_provenance": provenance,
            "export_source_provenance": initial_provenance,
            "source_provenance_captured": "before_export",
            "versions": _versions(),
        },
    )
    return output_path


def evaluate_policy(
    checkpoint_path: str | Path,
    seeds: Iterable[int],
    out: str | Path,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    initial_provenance = source_provenance()
    checkpoint_path = Path(checkpoint_path)
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("evaluation seeds must not be empty")
    kind, policy, _ = _load_artifact(checkpoint_path)
    records, schema = [], None
    for seed in seed_values:
        env = make_env(case, mode, candidate, max_steps, payload_kg)
        try:
            schema = schema or _environment_schema(env)
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            total_reward, steps, info = 0.0, 0, {}
            while not (terminated or truncated):
                teacher = _array(
                    env.unwrapped.teacher_action(), (ACTION_DIM,), "teacher_action"
                )
                if kind == "bc":
                    with torch.no_grad():
                        arm = policy(
                            torch.from_numpy(np.asarray(observation, np.float32)[None])
                        ).numpy()[0]
                    action = compose_arm_action(teacher, arm)
                else:
                    residual, _ = policy.predict(
                        np.asarray(observation, np.float32), deterministic=True
                    )
                    action = compose_residual_action(
                        teacher, residual, _action_encoding(env)[0]
                    )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            records.append(
                {
                    "seed": seed,
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                    "return": total_reward,
                    "steps": steps,
                    "diagnostics": info.get("diagnostics"),
                }
            )
        finally:
            env.close()
    failures = Counter(
        row["failure_reason"] or "unspecified" for row in records if not row["success"]
    )
    _assert_provenance_unchanged(initial_provenance, "policy evaluation")
    output_path = Path(out)
    _write_json(
        output_path,
        {
            "artifact": "microduck_arm_v1c_policy_evaluation",
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "payload_kg": payload_kg,
            "full_physics_rollout": True,
            "episodes": len(records),
            "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "failure_counts": dict(sorted(failures.items())),
            "records": records,
            "environment_schema": schema,
            "claim_scope": "simulation rollouts only",
            "hardware_truth_estimator_claimed": False,
            "hardware_deployment_qualified": False,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_evaluation",
            "versions": _versions(),
        },
    )
    return output_path


def evaluate_teacher(
    seeds: Iterable[int],
    out: str | Path,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    initial_provenance = source_provenance()
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("evaluation seeds must not be empty")
    records, schema = [], None
    for seed in seed_values:
        env = make_env(case, mode, candidate, max_steps, payload_kg)
        try:
            schema = schema or _environment_schema(env)
            observation, _ = env.reset(seed=seed)
            terminated = truncated = False
            total_reward, steps, info = 0.0, 0, {}
            while not (terminated or truncated):
                action = _array(
                    env.unwrapped.teacher_action(), (ACTION_DIM,), "teacher_action"
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
            records.append(
                {
                    "seed": seed,
                    "success": bool(info.get("success", False)),
                    "failure_reason": info.get("failure_reason"),
                    "return": total_reward,
                    "steps": steps,
                    "diagnostics": info.get("diagnostics"),
                }
            )
        finally:
            env.close()
    failures = Counter(
        row["failure_reason"] or "unspecified" for row in records if not row["success"]
    )
    _assert_provenance_unchanged(initial_provenance, "teacher evaluation")
    output_path = Path(out)
    _write_json(
        output_path,
        {
            "artifact": "microduck_arm_v1c_teacher_evaluation",
            "policy": "deterministic env.teacher_action",
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "payload_kg": payload_kg,
            "full_physics_rollout": True,
            "episodes": len(records),
            "successes": sum(row["success"] for row in records),
            "failures": sum(not row["success"] for row in records),
            "failure_counts": dict(sorted(failures.items())),
            "records": records,
            "environment_schema": schema,
            "claim_scope": "simulation rollouts only",
            "hardware_truth_estimator_claimed": False,
            "hardware_deployment_qualified": False,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_evaluation",
            "versions": _versions(),
        },
    )
    return output_path


def _last_jsonl(path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    return rows[-1] if rows else {}


def _bounded_integer(name: str, value: int, maximum: int, minimum: int = 1) -> int:
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def run_bounded_pipeline(
    out: str | Path,
    *,
    demonstration_episodes: int = 6,
    dagger_episodes: int = 6,
    bc_epochs: int = 20,
    merged_bc_epochs: int = 20,
    ppo_timesteps: int = 2048,
    demonstration_seed: int = 101,
    dagger_seed: int = 201,
    ppo_seed: int = 301,
    heldout_seeds: Iterable[int] = (401, 402, 403),
    validation_fraction: float = 0.2,
    arm_loss_weights: Iterable[float] | None = None,
    case: str = DEFAULT_CASE,
    mode: str = DEFAULT_MODE,
    candidate: str = DEFAULT_CANDIDATE,
    max_steps: int = DEFAULT_MAX_STEPS,
    payload_kg: float = DEFAULT_PAYLOAD_KG,
) -> Path:
    """Run the bounded v1C learning sequence after the environment is frozen."""
    _require_env_frozen()
    initial_provenance = source_provenance()
    demonstration_episodes = _bounded_integer(
        "demonstration_episodes",
        demonstration_episodes,
        MAX_RUNNER_EPISODES,
        minimum=2,
    )
    dagger_episodes = _bounded_integer(
        "dagger_episodes", dagger_episodes, MAX_RUNNER_EPISODES
    )
    bc_epochs = _bounded_integer("bc_epochs", bc_epochs, MAX_RUNNER_EPOCHS)
    merged_bc_epochs = _bounded_integer(
        "merged_bc_epochs", merged_bc_epochs, MAX_RUNNER_EPOCHS
    )
    ppo_timesteps = _bounded_integer(
        "ppo_timesteps", ppo_timesteps, MAX_RUNNER_PPO_TIMESTEPS
    )
    max_steps = _bounded_integer(
        "max_steps", max_steps, MAX_RUNNER_STEPS_PER_EPISODE
    )
    evaluation_seeds = [int(seed) for seed in heldout_seeds]
    if not evaluation_seeds:
        raise ValueError("heldout_seeds must not be empty")
    if len(evaluation_seeds) > MAX_RUNNER_EPISODES:
        raise ValueError(
            f"heldout_seeds must contain at most {MAX_RUNNER_EPISODES} seeds"
        )
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("heldout_seeds must be unique")
    collection_seeds = set(
        range(demonstration_seed, demonstration_seed + demonstration_episodes)
    )
    collection_seeds.update(range(dagger_seed, dagger_seed + dagger_episodes))
    overlap = collection_seeds.intersection(evaluation_seeds)
    if overlap:
        raise ValueError(
            f"heldout seeds overlap collection seeds: {sorted(overlap)}"
        )
    loss_weights = (
        None if arm_loss_weights is None else list(arm_loss_weights)
    )

    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"pipeline output directory is not empty: {root}; "
            "use a new attempt directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    env_args = {
        "case": case,
        "mode": mode,
        "candidate": candidate,
        "max_steps": max_steps,
        "payload_kg": payload_kg,
    }
    demonstrations = collect_demonstrations(
        root / "demonstrations",
        demonstration_episodes,
        demonstration_seed,
        **env_args,
    )
    initial_bc = train_bc(
        demonstrations,
        root / "bc-initial",
        bc_epochs,
        demonstration_seed,
        validation_fraction,
        loss_weights,
    )
    dagger = collect_dagger(
        initial_bc,
        root / "dagger",
        dagger_episodes,
        dagger_seed,
        **env_args,
    )
    merged = aggregate_datasets(
        [demonstrations, dagger],
        root / "demonstrations-merged",
    )
    merged_bc = train_bc(
        merged,
        root / "bc-merged",
        merged_bc_epochs,
        demonstration_seed,
        validation_fraction,
        loss_weights,
    )
    ppo = train_ppo(
        root / "ppo",
        ppo_timesteps,
        ppo_seed,
        **env_args,
    )
    initial_bc_onnx = export_policy(
        initial_bc, root / "exports" / "bc-initial-arm.onnx"
    )
    dagger_bc_onnx = export_policy(
        merged_bc, root / "exports" / "bc-dagger-merged-arm.onnx"
    )
    ppo_onnx = export_policy(ppo, root / "exports" / "ppo-residual-arm.onnx")
    teacher_evaluation = evaluate_teacher(
        evaluation_seeds,
        root / "evaluation-teacher.json",
        **env_args,
    )
    initial_bc_evaluation = evaluate_policy(
        initial_bc,
        evaluation_seeds,
        root / "evaluation-bc-initial.json",
        **env_args,
    )
    dagger_bc_evaluation = evaluate_policy(
        merged_bc,
        evaluation_seeds,
        root / "evaluation-bc-dagger-merged.json",
        **env_args,
    )
    ppo_evaluation = evaluate_policy(
        ppo,
        evaluation_seeds,
        root / "evaluation-ppo.json",
        **env_args,
    )
    artifacts = {
        "demonstrations": demonstrations,
        "initial_bc": initial_bc,
        "dagger": dagger,
        "merged_demonstrations": merged,
        "merged_bc": merged_bc,
        "ppo": ppo,
        "initial_bc_onnx": initial_bc_onnx,
        "dagger_bc_onnx": dagger_bc_onnx,
        "ppo_onnx": ppo_onnx,
        "teacher_evaluation": teacher_evaluation,
        "initial_bc_evaluation": initial_bc_evaluation,
        "dagger_bc_evaluation": dagger_bc_evaluation,
        "ppo_evaluation": ppo_evaluation,
    }
    _assert_provenance_unchanged(initial_provenance, "bounded pipeline")
    manifest_path = root / "pipeline.json"
    _write_json(
        manifest_path,
        {
            "artifact": "microduck_arm_v1c_bounded_pipeline",
            "pipeline_version": PIPELINE_VERSION,
            "environment_frozen": True,
            "bounds": {
                "max_episodes_per_collection": MAX_RUNNER_EPISODES,
                "max_epochs_per_bc_stage": MAX_RUNNER_EPOCHS,
                "max_ppo_timesteps": MAX_RUNNER_PPO_TIMESTEPS,
                "max_steps_per_episode": MAX_RUNNER_STEPS_PER_EPISODE,
            },
            "configuration": {
                **env_args,
                "demonstration_episodes": demonstration_episodes,
                "dagger_episodes": dagger_episodes,
                "bc_epochs": bc_epochs,
                "merged_bc_epochs": merged_bc_epochs,
                "ppo_timesteps": ppo_timesteps,
                "validation_fraction": validation_fraction,
                "arm_loss_weights": loss_weights,
            },
            "seeds": {
                "demonstration_start": demonstration_seed,
                "dagger_start": dagger_seed,
                "ppo": ppo_seed,
                "collection": sorted(collection_seeds),
                "heldout": evaluation_seeds,
                "heldout_disjoint_from_collection": True,
            },
            "stages": [
                "collect_demonstrations",
                "train_bc_initial",
                "collect_dagger_on_policy_states",
                "aggregate_datasets",
                "train_bc_merged",
                "train_residual_ppo",
                "export_initial_bc_onnx",
                "export_dagger_merged_bc_onnx",
                "export_ppo_onnx",
                "evaluate_teacher_heldout_full_physics",
                "evaluate_initial_bc_heldout_full_physics",
                "evaluate_dagger_merged_bc_heldout_full_physics",
                "evaluate_ppo_heldout_full_physics",
            ],
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
                for name, path in artifacts.items()
            },
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_pipeline",
            "versions": _versions(),
        },
    )
    summary_path = root / "training-summary.json"
    evaluations = {
        name: json.loads(path.read_text())
        for name, path in {
            "teacher": teacher_evaluation,
            "bc": initial_bc_evaluation,
            "dagger": dagger_bc_evaluation,
            "ppo": ppo_evaluation,
        }.items()
    }
    _write_json(
        summary_path,
        {
            "artifact": "microduck_arm_v1c_training_summary",
            "pipeline_manifest": str(manifest_path.resolve()),
            "commands": [
                "collect_demonstrations(...)",
                "train_bc(demonstrations, ...)",
                "collect_dagger(initial_bc, ...)",
                "aggregate_datasets([demonstrations, dagger], ...)",
                "train_bc(merged_demonstrations, ...)",
                "train_ppo(...)",
                "export_policy(initial_bc|merged_bc|ppo, ...)",
                "evaluate_teacher(heldout_seeds, ...)",
                "evaluate_policy(initial_bc|merged_bc|ppo, heldout_seeds, ...)",
            ],
            "configuration": {
                **env_args,
                "demonstration_episodes": demonstration_episodes,
                "dagger_episodes": dagger_episodes,
                "bc_epochs": bc_epochs,
                "merged_bc_epochs": merged_bc_epochs,
                "ppo_timesteps": ppo_timesteps,
                "demonstration_seed": demonstration_seed,
                "dagger_seed": dagger_seed,
                "ppo_seed": ppo_seed,
                "heldout_seeds": evaluation_seeds,
                "validation_fraction": validation_fraction,
                "arm_loss_weights": loss_weights,
            },
            "training_metrics": {
                "bc": _last_jsonl(initial_bc.with_name("bc-training.jsonl")),
                "dagger_merged_bc": _last_jsonl(
                    merged_bc.with_name("bc-training.jsonl")
                ),
                "ppo": _last_jsonl(ppo.with_name("ppo-training.jsonl")),
            },
            "evaluation": {
                name: {
                    "successes": report["successes"],
                    "failures": report["failures"],
                    "failure_counts": report["failure_counts"],
                    "records": report["records"],
                }
                for name, report in evaluations.items()
            },
            "onnx_parity": {
                name: json.loads(Path(str(path) + ".json").read_text())[
                    "parity_max_abs_error"
                ]
                for name, path in {
                    "bc": initial_bc_onnx,
                    "dagger": dagger_bc_onnx,
                    "ppo": ppo_onnx,
                }.items()
            },
            "honesty": {
                "simulation_only": True,
                "hardware_deployment_qualified": False,
                "hardware_truth_estimator_claimed": False,
                "bc_passed": evaluations["bc"]["failures"] == 0,
                "dagger_passed": evaluations["dagger"]["failures"] == 0,
                "ppo_passed": evaluations["ppo"]["failures"] == 0,
            },
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_pipeline",
        },
    )
    return manifest_path


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--mode", choices=("fixture", "free"), default=DEFAULT_MODE)
    parser.add_argument("--candidate", choices=("A", "B"), default=DEFAULT_CANDIDATE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--payload-kg", type=float, default=DEFAULT_PAYLOAD_KG)


def _parse_weights(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if len(result) != ARM_DIM:
        raise argparse.ArgumentTypeError("expected five comma-separated weights")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded simulation-only learning tools for MicroDuck arm v1C"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demonstrate = commands.add_parser("demonstrate")
    demonstrate.add_argument("--out", required=True)
    demonstrate.add_argument("--episodes", type=int, default=10)
    demonstrate.add_argument("--seed", type=int, default=101)
    _add_env_args(demonstrate)
    bc = commands.add_parser("train-bc")
    bc.add_argument("--dataset", required=True)
    bc.add_argument("--out", required=True)
    bc.add_argument("--epochs", type=int, default=20)
    bc.add_argument("--seed", type=int, default=101)
    bc.add_argument("--validation-fraction", type=float, default=0.2)
    bc.add_argument("--arm-loss-weights", type=_parse_weights)
    dagger = commands.add_parser("dagger")
    dagger.add_argument("--checkpoint", required=True)
    dagger.add_argument("--out", required=True)
    dagger.add_argument("--episodes", type=int, default=10)
    dagger.add_argument("--seed", type=int, default=101)
    _add_env_args(dagger)
    ppo = commands.add_parser("train-ppo")
    ppo.add_argument("--out", required=True)
    ppo.add_argument("--steps", type=int, default=2048)
    ppo.add_argument("--seed", type=int, default=101)
    _add_env_args(ppo)
    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seeds", default="101,102,103")
    _add_env_args(evaluate)
    args = parser.parse_args(argv)
    env_args = {
        name: getattr(args, name)
        for name in ("case", "mode", "candidate", "max_steps", "payload_kg")
        if hasattr(args, name)
    }
    if args.command == "demonstrate":
        result = collect_demonstrations(
            args.out, args.episodes, args.seed, **env_args
        )
    elif args.command == "train-bc":
        result = train_bc(
            args.dataset,
            args.out,
            args.epochs,
            args.seed,
            args.validation_fraction,
            args.arm_loss_weights,
        )
    elif args.command == "dagger":
        result = collect_dagger(
            args.checkpoint, args.out, args.episodes, args.seed, **env_args
        )
    elif args.command == "train-ppo":
        result = train_ppo(args.out, args.steps, args.seed, **env_args)
    elif args.command == "export":
        result = export_policy(args.checkpoint, args.out)
    else:
        result = evaluate_policy(
            args.checkpoint,
            (int(item) for item in args.seeds.split(",")),
            args.out,
            **env_args,
        )
    print(json.dumps({"artifact": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
