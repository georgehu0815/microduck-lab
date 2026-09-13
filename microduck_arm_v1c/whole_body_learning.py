from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import gymnasium as gym
import mujoco
import numpy as np
import torch

from microduck_arm_v1c import learning


ARTIFACT_VERSION = 2
CONTROL_PROFILE = "whole_body_residual_obs69_v2"
BASE_OBSERVATION_DIM = learning.OBSERVATION_DIM
DYNAMIC_OBSERVATION_DIM = 5
OBSERVATION_DIM = BASE_OBSERVATION_DIM + DYNAMIC_OBSERVATION_DIM
ACTION_DIM = learning.ACTION_DIM
LEG_DIM = learning.ARM_START
ARM_DIM = learning.ARM_DIM
DEFAULT_LEG_RESIDUAL_RAD = 0.01
DEFAULT_ARM_RESIDUAL_RAD = learning.MAX_RESIDUAL_RAD
MAX_TRAIN_TIMESTEPS = learning.MAX_RUNNER_PPO_TIMESTEPS
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
RENDER_FPS = 25
CONTACT_FORCE_THRESHOLD_N = 0.01


def source_provenance() -> dict[str, Any]:
    provenance = learning.source_provenance()
    files = dict(provenance["files"])
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root).as_posix()
        files[f"v1c_python_{relative}"] = {
            "path": str(path),
            "sha256": learning._sha256(path),
        }
    return {
        **provenance,
        "files": dict(sorted(files.items())),
        "invalidation_rule": (
            provenance["invalidation_rule"]
            + " Changes to any Python source under microduck_arm_v1c, including "
            "teacher and gait logic, also invalidate training, export, and evaluation."
        ),
    }


def _assert_provenance_unchanged(
    initial: dict[str, Any],
    operation: str,
) -> None:
    current = source_provenance()
    if current == initial:
        return
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


def residual_caps(
    leg_residual_rad: float = DEFAULT_LEG_RESIDUAL_RAD,
    arm_residual_rad: float = DEFAULT_ARM_RESIDUAL_RAD,
) -> np.ndarray:
    values = np.asarray([leg_residual_rad, arm_residual_rad], dtype=np.float32)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("leg and arm residual caps must be finite and positive")
    return np.r_[
        np.full(LEG_DIM, values[0], dtype=np.float32),
        np.full(ARM_DIM, values[1], dtype=np.float32),
    ]


def compose_residual_action(
    teacher_action: np.ndarray,
    residual_action: np.ndarray,
    action_scales: np.ndarray,
    *,
    leg_residual_rad: float = DEFAULT_LEG_RESIDUAL_RAD,
    arm_residual_rad: float = DEFAULT_ARM_RESIDUAL_RAD,
) -> np.ndarray:
    teacher = learning._array(teacher_action, (ACTION_DIM,), "teacher_action")
    residual = learning._array(residual_action, (ACTION_DIM,), "residual_action")
    scales = learning._array(action_scales, (ACTION_DIM,), "action_scales")
    if np.any(scales <= 0):
        raise ValueError("action_scales must be positive")
    caps = residual_caps(leg_residual_rad, arm_residual_rad)
    return np.clip(
        teacher + np.clip(residual, -1, 1) * caps / scales,
        -1,
        1,
    ).astype(np.float32)


def _dynamic_features(env: gym.Env) -> np.ndarray:
    base = env.unwrapped
    velocity = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        base.model,
        base.data,
        mujoco.mjtObj.mjOBJ_BODY,
        int(base.trunk_id),
        velocity,
        1,
    )

    contacts = np.zeros(2, dtype=np.float32)
    foot_ids = (
        int(base.model.body("ankle_left").id),
        int(base.model.body("ankle_right").id),
    )
    force = np.zeros(6, dtype=np.float64)
    for contact_index in range(base.data.ncon):
        contact = base.data.contact[contact_index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        geom_names = tuple(base.model.geom(geom_id).name for geom_id in geom_ids)
        if "floor" not in geom_names:
            continue
        mujoco.mj_contactForce(base.model, base.data, contact_index, force)
        if float(force[0]) <= CONTACT_FORCE_THRESHOLD_N:
            continue
        bodies = {int(base.model.geom_bodyid[geom_id]) for geom_id in geom_ids}
        for side, body_id in enumerate(foot_ids):
            if body_id in bodies:
                contacts[side] = 1.0

    result = np.concatenate([velocity[3:].astype(np.float32), contacts])
    if result.shape != (DYNAMIC_OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise FloatingPointError("invalid whole-body dynamic observation")
    return result


def augment_observation(env: gym.Env, observation: np.ndarray) -> np.ndarray:
    base = learning._array(
        observation,
        (BASE_OBSERVATION_DIM,),
        "base observation",
    )
    result = np.concatenate([base, _dynamic_features(env)]).astype(np.float32)
    if result.shape != (OBSERVATION_DIM,) or not np.isfinite(result).all():
        raise FloatingPointError("invalid 69-dimensional whole-body observation")
    return result


class WholeBodyResidualEnv(gym.Wrapper):
    """Apply a bounded 15-action residual over the existing full-body teacher."""

    def __init__(
        self,
        env: gym.Env,
        *,
        leg_residual_rad: float = DEFAULT_LEG_RESIDUAL_RAD,
        arm_residual_rad: float = DEFAULT_ARM_RESIDUAL_RAD,
    ):
        super().__init__(env)
        if env.observation_space.shape != (BASE_OBSERVATION_DIM,):
            raise ValueError("v1C observation space must have shape (64,)")
        if env.action_space.shape != (ACTION_DIM,):
            raise ValueError("v1C public action space must have shape (15,)")
        self.action_scales, self.action_offsets = learning._action_encoding(env)
        self.residual_caps_rad = residual_caps(
            leg_residual_rad,
            arm_residual_rad,
        )
        self.leg_residual_rad = float(leg_residual_rad)
        self.arm_residual_rad = float(arm_residual_rad)
        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.r_[
                np.asarray(env.observation_space.low, dtype=np.float32),
                np.full(3, -np.inf, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            ],
            high=np.r_[
                np.asarray(env.observation_space.high, dtype=np.float32),
                np.full(3, np.inf, dtype=np.float32),
                np.ones(2, dtype=np.float32),
            ],
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return augment_observation(self.env, observation), info

    def step(self, action):
        raw = learning._array(action, (ACTION_DIM,), "residual action")
        teacher = learning._array(
            self.env.unwrapped.teacher_action(),
            (ACTION_DIM,),
            "teacher_action",
        )
        composed = compose_residual_action(
            teacher,
            raw,
            self.action_scales,
            leg_residual_rad=self.leg_residual_rad,
            arm_residual_rad=self.arm_residual_rad,
        )
        observation, reward, terminated, truncated, info = self.env.step(composed)
        info = dict(info)
        info["learning_wrapper"] = {
            "artifact_version": ARTIFACT_VERSION,
            "control_profile": CONTROL_PROFILE,
            "teacher_action": teacher,
            "raw_residual_action": raw,
            "composed_action": composed,
            "residual_caps_rad": self.residual_caps_rad,
            "leg_action_indices": np.arange(LEG_DIM, dtype=np.int64),
            "arm_action_indices": np.arange(LEG_DIM, ACTION_DIM, dtype=np.int64),
            "waist_action_indices": np.asarray([], dtype=np.int64),
        }
        return (
            augment_observation(self.env, observation),
            reward,
            terminated,
            truncated,
            info,
        )


def _paths(out: str | Path) -> tuple[Path, Path, Path]:
    path = Path(out)
    checkpoint = (
        path
        if path.suffix == ".zip"
        else path / "whole-body-residual-ppo.zip"
    )
    metadata = checkpoint.with_suffix(".json")
    log = checkpoint.with_name("whole-body-residual-ppo-training.jsonl")
    return checkpoint, metadata, log


def _callback(path: Path):
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
            learning._append_jsonl(path, self.metrics())
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
    *,
    case: str = learning.DEFAULT_CASE,
    mode: str = "free",
    candidate: str = learning.DEFAULT_CANDIDATE,
    max_steps: int = learning.DEFAULT_MAX_STEPS,
    payload_kg: float = learning.DEFAULT_PAYLOAD_KG,
    leg_residual_rad: float = DEFAULT_LEG_RESIDUAL_RAD,
    arm_residual_rad: float = DEFAULT_ARM_RESIDUAL_RAD,
) -> Path:
    learning._require_env_frozen()
    total_timesteps = learning._bounded_integer(
        "total_timesteps",
        total_timesteps,
        MAX_TRAIN_TIMESTEPS,
    )
    caps = residual_caps(leg_residual_rad, arm_residual_rad)
    initial_provenance = source_provenance()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    checkpoint_path, metadata_path, log_path = _paths(out)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.unlink(missing_ok=True)
    learning._seed_everything(seed)
    torch.set_num_threads(1)
    details: dict[str, Any] = {}

    def factory():
        env = learning.make_env(case, mode, candidate, max_steps, payload_kg)
        wrapped = WholeBodyResidualEnv(
            env,
            leg_residual_rad=leg_residual_rad,
            arm_residual_rad=arm_residual_rad,
        )
        details["base_environment_schema"] = learning._environment_schema(env)
        details["observation_space"] = learning._space_schema(
            wrapped.observation_space
        )
        return wrapped

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
        callback = _callback(log_path)
        model.learn(total_timesteps, callback=callback, progress_bar=False)
        learning._append_jsonl(
            log_path,
            {"phase": "training_end", **callback.metrics()},
        )
    finally:
        vec_env.close()

    _assert_provenance_unchanged(initial_provenance, "whole-body PPO training")
    model.save(checkpoint_path)
    learning._write_json(
        metadata_path,
        {
            "artifact": "microduck_arm_v1c_whole_body_residual_ppo_checkpoint",
            "artifact_version": ARTIFACT_VERSION,
            "control_profile": CONTROL_PROFILE,
            "checkpoint_sha256": learning._sha256(checkpoint_path),
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "max_steps": max_steps,
            "payload_kg": payload_kg,
            "requested_timesteps": total_timesteps,
            "actual_timesteps": int(model.num_timesteps),
            "observation_dim": OBSERVATION_DIM,
            "public_action_dim": ACTION_DIM,
            "policy_action_dim": ACTION_DIM,
            "deployment_contract": "obs69-action15:whole-body-residual-v2",
            "legacy_61x14_contract_compatible": False,
            "residual_caps_rad": caps,
            "leg_residual_cap_rad": leg_residual_rad,
            "arm_residual_cap_rad": arm_residual_rad,
            "teacher_retained": True,
            "leg_residuals_applied": True,
            "arm_residuals_applied": True,
            "waist_joint_present": False,
            "observation_contract": {
                "base_observation_indices": [0, BASE_OBSERVATION_DIM],
                "trunk_local_linear_velocity_indices": [64, 67],
                "left_foot_force_contact_index": 67,
                "right_foot_force_contact_index": 68,
                "foot_contact_force_threshold_n": CONTACT_FORCE_THRESHOLD_N,
                "simulator_state_only": True,
                "hardware_foot_sensors_claimed": False,
            },
            "observation_limitations": [
                "foot contacts are derived from MuJoCo contact forces",
                "no hardware foot-sensor contract is claimed",
            ],
            "base_environment_schema": details["base_environment_schema"],
            "observation_space": details["observation_space"],
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
            "versions": learning._versions(),
        },
    )
    return checkpoint_path


class _PPOWholeBodyActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy.action_space.shape != (ACTION_DIM,):
            raise ValueError("whole-body PPO policy must have fifteen actions")
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.policy.extract_features(
            observation,
            self.policy.features_extractor,
        )
        latent = self.policy.mlp_extractor.forward_actor(features)
        return torch.clamp(self.policy.action_net(latent), -1, 1)


def _load_ppo(checkpoint_path: str | Path):
    from stable_baselines3 import PPO

    path = Path(checkpoint_path)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"whole-body checkpoint metadata is required: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("artifact")
        != "microduck_arm_v1c_whole_body_residual_ppo_checkpoint"
        or metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("control_profile") != CONTROL_PROFILE
        or metadata.get("observation_dim") != OBSERVATION_DIM
        or metadata.get("policy_action_dim") != ACTION_DIM
    ):
        raise ValueError("not a compatible v1C whole-body residual PPO checkpoint")
    learning._validate_provenance(metadata.get("source_provenance"))
    if metadata.get("checkpoint_sha256") != learning._sha256(path):
        raise RuntimeError("whole-body checkpoint hash does not match metadata")
    return PPO.load(path, device="cpu"), metadata


def export_policy(checkpoint_path: str | Path, out: str | Path) -> Path:
    initial_provenance = source_provenance()
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(out)
    if output_path.suffix != ".onnx":
        output_path /= "whole-body-residual-policy.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model, checkpoint_metadata = _load_ppo(checkpoint_path)
    actor = _PPOWholeBodyActor(model.policy).eval()

    torch.onnx.export(
        actor,
        torch.zeros(1, OBSERVATION_DIM),
        output_path,
        input_names=["observation"],
        output_names=["residual_action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "residual_action": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    import onnxruntime as ort

    probe = np.random.default_rng(20260913).normal(
        0,
        0.2,
        (8, OBSERVATION_DIM),
    ).astype(np.float32)
    with torch.no_grad():
        expected = actor(torch.from_numpy(probe)).numpy()
    actual = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    ).run(None, {"observation": probe})[0]
    if expected.shape != (8, ACTION_DIM) or actual.shape != expected.shape:
        raise RuntimeError("ONNX export must produce exactly fifteen residual actions")
    parity = float(np.max(np.abs(expected - actual)))
    if parity > 1e-5:
        raise RuntimeError(f"ONNX parity failed: max absolute error {parity}")

    _assert_provenance_unchanged(initial_provenance, "whole-body ONNX export")
    learning._write_json(
        Path(str(output_path) + ".json"),
        {
            "artifact": "microduck_arm_v1c_whole_body_residual_onnx",
            "artifact_version": ARTIFACT_VERSION,
            "control_profile": CONTROL_PROFILE,
            "path": str(output_path.resolve()),
            "sha256": learning._sha256(output_path),
            "source_checkpoint": str(checkpoint_path.resolve()),
            "source_checkpoint_sha256": learning._sha256(checkpoint_path),
            "input": {"name": "observation", "shape": ["batch", OBSERVATION_DIM]},
            "output": {"name": "residual_action", "shape": ["batch", ACTION_DIM]},
            "action_semantics": (
                "normalized whole-body residuals for explicit deterministic "
                "teacher composition"
            ),
            "deployment_contract": "obs69-action15:whole-body-residual-v2",
            "legacy_61x14_contract_compatible": False,
            "residual_caps_rad": checkpoint_metadata["residual_caps_rad"],
            "teacher_adapter_required": True,
            "direct_legacy_or_hardware_compatible": False,
            "waist_joint_present": False,
            "simulator_state_only": True,
            "hardware_foot_sensors_claimed": False,
            "observation_contract": checkpoint_metadata["observation_contract"],
            "parity_max_abs_error": parity,
            "simulation_only": True,
            "hardware_deployment_qualified": False,
            "source_provenance": checkpoint_metadata["source_provenance"],
            "export_source_provenance": initial_provenance,
            "source_provenance_captured": "before_export",
            "versions": learning._versions(),
        },
    )
    return output_path


def _load_onnx(onnx_path: str | Path):
    import onnxruntime as ort

    path = Path(onnx_path)
    metadata_path = Path(str(path) + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"ONNX policy and metadata are required: {path}")
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("artifact") != "microduck_arm_v1c_whole_body_residual_onnx"
        or metadata.get("artifact_version") != ARTIFACT_VERSION
        or metadata.get("control_profile") != CONTROL_PROFILE
        or metadata.get("deployment_contract")
        != "obs69-action15:whole-body-residual-v2"
        or metadata.get("input", {}).get("shape") != ["batch", OBSERVATION_DIM]
        or metadata.get("output", {}).get("shape") != ["batch", ACTION_DIM]
    ):
        raise ValueError("not a compatible v1C 69x15 whole-body residual ONNX policy")
    learning._validate_provenance(metadata.get("source_provenance"))
    learning._validate_provenance(metadata.get("export_source_provenance"))
    if metadata.get("sha256") != learning._sha256(path):
        raise RuntimeError("whole-body ONNX hash does not match metadata")
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (
        len(inputs) != 1
        or inputs[0].name != "observation"
        or inputs[0].shape[-1] != OBSERVATION_DIM
        or len(outputs) != 1
        or outputs[0].name != "residual_action"
        or outputs[0].shape[-1] != ACTION_DIM
    ):
        raise RuntimeError("ONNX runtime schema is not the declared 69x15 contract")
    return session, metadata


def _onnx_residual(session, observation: np.ndarray) -> np.ndarray:
    result = session.run(
        ["residual_action"],
        {"observation": np.asarray(observation, dtype=np.float32)[None]},
    )[0]
    return learning._array(result[0], (ACTION_DIM,), "ONNX residual action")


def _output_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"expected non-empty rollout output is missing: {path}")
    digest = learning._sha256(path)
    if digest != learning._sha256(path):
        raise RuntimeError(f"rollout output changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def _rollout_episode(
    *,
    seed: int,
    controller: str,
    predict_residual: Callable[[np.ndarray], np.ndarray] | None,
    case: str,
    mode: str,
    candidate: str,
    max_steps: int,
    payload_kg: float,
    leg_residual_rad: float,
    arm_residual_rad: float,
    telemetry_path: Path | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    if controller not in {"teacher", "ppo", "teacher+wholebodyresidual"}:
        raise ValueError(f"unknown whole-body controller: {controller}")
    if controller != "teacher" and predict_residual is None:
        raise ValueError("residual controller requires a predictor")
    for path in (telemetry_path, video_path):
        if path is not None and path.exists():
            raise FileExistsError(f"refusing to overwrite rollout output: {path}")

    base_env = learning.make_env(case, mode, candidate, max_steps, payload_kg)
    env = WholeBodyResidualEnv(
        base_env,
        leg_residual_rad=leg_residual_rad,
        arm_residual_rad=arm_residual_rad,
    )
    writer = None
    frame_count = 0
    try:
        if telemetry_path is not None:
            telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        if video_path is not None:
            import imageio.v2 as imageio

            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                video_path,
                fps=RENDER_FPS,
                codec="libx264",
                macro_block_size=16,
            )
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        total_reward, steps, info = 0.0, 0, {}
        while not (terminated or truncated):
            if controller == "teacher":
                residual = np.zeros(ACTION_DIM, dtype=np.float32)
            else:
                residual = learning._array(
                    predict_residual(observation),
                    (ACTION_DIM,),
                    "predicted residual action",
                )
            observation, reward, terminated, truncated, info = env.step(residual)
            total_reward += float(reward)
            steps += 1
            wrapper_info = info["learning_wrapper"]
            if telemetry_path is not None:
                learning._append_jsonl(
                    telemetry_path,
                    {
                        "step": steps,
                        "time_s": float(env.unwrapped.data.time),
                        "controller": controller,
                        "simulation_only": True,
                        "reward": float(reward),
                        "cumulative_return": total_reward,
                        "observation": observation,
                        "teacher_action": wrapper_info["teacher_action"],
                        "residual_action": wrapper_info["raw_residual_action"],
                        "composed_action": wrapper_info["composed_action"],
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "failure_reason": info.get("failure_reason"),
                        "diagnostics": info.get("diagnostics"),
                    },
                )
            if writer is not None and (steps % 2 == 0 or terminated or truncated):
                from PIL import Image, ImageDraw

                raw_frame = np.asarray(env.unwrapped.render())
                if raw_frame.shape != (RENDER_HEIGHT, RENDER_WIDTH, 3):
                    raise RuntimeError(
                        "whole-body renderer must produce 640x480 RGB frames"
                    )
                frame = Image.fromarray(raw_frame)
                drawing = ImageDraw.Draw(frame)
                drawing.rectangle((0, 0, RENDER_WIDTH, 26), fill="black")
                drawing.text(
                    (8, 7),
                    (
                        f"SIMULATION ONLY | {controller} | v1-C {candidate} | "
                        f"{mode} | {case} | t={env.unwrapped.data.time:.2f}s"
                    ),
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
                frame_count += 1
        record = {
            "seed": seed,
            "controller": controller,
            "success": bool(info.get("success", False)),
            "failure_reason": info.get("failure_reason"),
            "return": total_reward,
            "steps": steps,
            "diagnostics": info.get("diagnostics"),
            "simulation_only": True,
            "saved_regardless_of_success": True,
        }
    finally:
        if writer is not None:
            writer.close()
        env.close()

    outputs = {}
    if telemetry_path is not None:
        outputs["telemetry"] = _output_record(telemetry_path)
    if video_path is not None:
        outputs["video"] = {
            **_output_record(video_path),
            "frames": frame_count,
            "fps": RENDER_FPS,
            "width": RENDER_WIDTH,
            "height": RENDER_HEIGHT,
            "label": f"SIMULATION ONLY | {controller}",
        }
    if outputs:
        record["outputs"] = outputs
    return record


def _rollout(
    *,
    seeds: list[int],
    controller: str,
    model,
    case: str,
    mode: str,
    candidate: str,
    max_steps: int,
    payload_kg: float,
    leg_residual_rad: float,
    arm_residual_rad: float,
) -> list[dict[str, Any]]:
    records = []
    for seed in seeds:
        predictor = None
        if controller != "teacher":
            def predictor(observation):
                return model.predict(
                    np.asarray(observation, np.float32),
                    deterministic=True,
                )[0]

        records.append(
            _rollout_episode(
                seed=seed,
                controller=controller,
                predict_residual=predictor,
                case=case,
                mode=mode,
                candidate=candidate,
                max_steps=max_steps,
                payload_kg=payload_kg,
                leg_residual_rad=leg_residual_rad,
                arm_residual_rad=arm_residual_rad,
            )
        )
    return records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = Counter(
        row["failure_reason"] or "unspecified"
        for row in records
        if not row["success"]
    )
    return {
        "episodes": len(records),
        "successes": sum(row["success"] for row in records),
        "failures": sum(not row["success"] for row in records),
        "failure_counts": dict(sorted(failures.items())),
        "records": records,
    }


def evaluate_policy(
    checkpoint_path: str | Path,
    seeds: Iterable[int],
    out: str | Path,
    *,
    case: str = learning.DEFAULT_CASE,
    mode: str = "free",
    candidate: str = learning.DEFAULT_CANDIDATE,
    max_steps: int = learning.DEFAULT_MAX_STEPS,
    payload_kg: float = learning.DEFAULT_PAYLOAD_KG,
) -> Path:
    initial_provenance = source_provenance()
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("evaluation seeds must not be empty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("evaluation seeds must be unique")
    model, checkpoint_metadata = _load_ppo(checkpoint_path)
    leg_cap = float(checkpoint_metadata["leg_residual_cap_rad"])
    arm_cap = float(checkpoint_metadata["arm_residual_cap_rad"])
    common = {
        "seeds": seed_values,
        "case": case,
        "mode": mode,
        "candidate": candidate,
        "max_steps": max_steps,
        "payload_kg": payload_kg,
        "leg_residual_rad": leg_cap,
        "arm_residual_rad": arm_cap,
    }
    teacher_records = _rollout(controller="teacher", model=None, **common)
    ppo_records = _rollout(controller="ppo", model=model, **common)
    _assert_provenance_unchanged(initial_provenance, "whole-body policy evaluation")

    output_path = Path(out)
    learning._write_json(
        output_path,
        {
            "artifact": "microduck_arm_v1c_whole_body_residual_evaluation",
            "artifact_version": ARTIFACT_VERSION,
            "control_profile": CONTROL_PROFILE,
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": learning._sha256(Path(checkpoint_path)),
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "payload_kg": payload_kg,
            "heldout_seeds": seed_values,
            "same_seeds_for_teacher_and_ppo": True,
            "full_physics_rollout": True,
            "teacher": _summarize(teacher_records),
            "ppo": _summarize(ppo_records),
            "residual_caps_rad": checkpoint_metadata["residual_caps_rad"],
            "waist_joint_present": False,
            "observation_dim": OBSERVATION_DIM,
            "deployment_contract": "obs69-action15:whole-body-residual-v2",
            "legacy_61x14_contract_compatible": False,
            "simulator_state_only": True,
            "hardware_foot_sensors_claimed": False,
            "observation_contract": checkpoint_metadata["observation_contract"],
            "observation_limitations": checkpoint_metadata["observation_limitations"],
            "claim_scope": "simulation rollouts only",
            "hardware_truth_estimator_claimed": False,
            "hardware_deployment_qualified": False,
            "source_provenance": initial_provenance,
            "source_provenance_captured": "before_evaluation",
            "versions": learning._versions(),
        },
    )
    return output_path


def evaluate_onnx_render(
    onnx_path: str | Path,
    seeds: Iterable[int],
    out: str | Path,
    *,
    case: str = learning.DEFAULT_CASE,
    mode: str = "free",
    candidate: str = learning.DEFAULT_CANDIDATE,
    max_steps: int = learning.DEFAULT_MAX_STEPS,
    payload_kg: float = learning.DEFAULT_PAYLOAD_KG,
) -> Path:
    initial_provenance = source_provenance()
    seed_values = [int(seed) for seed in seeds]
    if not seed_values:
        raise ValueError("evaluation seeds must not be empty")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("evaluation seeds must be unique")
    output_dir = Path(out)
    report_path = output_dir / "evaluation.json"
    expected_outputs = [report_path]
    for controller in ("teacher", "teacher+wholebodyresidual"):
        for seed in seed_values:
            episode_dir = output_dir / f"{controller.replace('+', '-')}-seed-{seed}"
            expected_outputs.extend(
                [
                    episode_dir / "result.json",
                    episode_dir / "telemetry.jsonl",
                    episode_dir / "rollout.mp4",
                ]
            )
    conflicts = [path for path in expected_outputs if path.exists()]
    if conflicts:
        raise FileExistsError(
            "refusing to overwrite evaluation outputs: "
            + ", ".join(str(path) for path in conflicts)
        )

    session, onnx_metadata = _load_onnx(onnx_path)
    caps = learning._array(
        onnx_metadata["residual_caps_rad"],
        (ACTION_DIM,),
        "ONNX residual caps",
    )
    leg_cap = float(caps[0])
    arm_cap = float(caps[LEG_DIM])
    if not (
        np.allclose(caps[:LEG_DIM], leg_cap)
        and np.allclose(caps[LEG_DIM:], arm_cap)
    ):
        raise ValueError("ONNX metadata must declare uniform leg and arm caps")

    records: dict[str, list[dict[str, Any]]] = {
        "teacher": [],
        "teacher+wholebodyresidual": [],
    }
    for controller in records:
        predictor = (
            None
            if controller == "teacher"
            else lambda observation: _onnx_residual(session, observation)
        )
        for seed in seed_values:
            episode_dir = output_dir / f"{controller.replace('+', '-')}-seed-{seed}"
            result_path = episode_dir / "result.json"
            record = _rollout_episode(
                seed=seed,
                controller=controller,
                predict_residual=predictor,
                case=case,
                mode=mode,
                candidate=candidate,
                max_steps=max_steps,
                payload_kg=payload_kg,
                leg_residual_rad=leg_cap,
                arm_residual_rad=arm_cap,
                telemetry_path=episode_dir / "telemetry.jsonl",
                video_path=episode_dir / "rollout.mp4",
            )
            learning._write_json(result_path, record)
            record["outputs"]["result"] = _output_record(result_path)
            records[controller].append(record)

    _assert_provenance_unchanged(
        initial_provenance,
        "whole-body ONNX evaluation and rendering",
    )
    final_provenance = source_provenance()
    if final_provenance != initial_provenance:
        raise RuntimeError("source provenance changed before evaluation report write")
    output_dir.mkdir(parents=True, exist_ok=True)
    learning._write_json(
        report_path,
        {
            "artifact": "microduck_arm_v1c_whole_body_residual_onnx_evaluation",
            "artifact_version": ARTIFACT_VERSION,
            "control_profile": CONTROL_PROFILE,
            "onnx": str(Path(onnx_path).resolve()),
            "onnx_sha256": learning._sha256(Path(onnx_path)),
            "case": case,
            "mode": mode,
            "candidate": candidate,
            "payload_kg": payload_kg,
            "heldout_seeds": seed_values,
            "same_seeds_for_teacher_and_onnx": True,
            "full_physics_rollout": True,
            "teacher": _summarize(records["teacher"]),
            "teacher+wholebodyresidual": _summarize(
                records["teacher+wholebodyresidual"]
            ),
            "failure_episodes_saved": True,
            "complete_step_telemetry_saved": True,
            "render": {
                "width": RENDER_WIDTH,
                "height": RENDER_HEIGHT,
                "fps": RENDER_FPS,
                "labels": [
                    "SIMULATION ONLY | teacher",
                    "SIMULATION ONLY | teacher+wholebodyresidual",
                ],
            },
            "observation_dim": OBSERVATION_DIM,
            "policy_action_dim": ACTION_DIM,
            "deployment_contract": "obs69-action15:whole-body-residual-v2",
            "legacy_61x14_contract_compatible": False,
            "simulator_state_only": True,
            "hardware_foot_sensors_claimed": False,
            "waist_joint_present": False,
            "observation_contract": onnx_metadata["observation_contract"],
            "output_preflight_no_overwrite": True,
            "all_episode_outputs_verified_after_write": True,
            "source_provenance_before": initial_provenance,
            "source_provenance_after": final_provenance,
            "source_provenance_unchanged": True,
            "versions": learning._versions(),
        },
    )
    _output_record(report_path)
    _assert_provenance_unchanged(
        initial_provenance,
        "whole-body ONNX evaluation report write",
    )
    return report_path


def _add_env_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case", default=learning.DEFAULT_CASE)
    parser.add_argument("--mode", choices=("fixture", "free"), default="free")
    parser.add_argument(
        "--candidate",
        choices=("A", "B"),
        default=learning.DEFAULT_CANDIDATE,
    )
    parser.add_argument("--max-steps", type=int, default=learning.DEFAULT_MAX_STEPS)
    parser.add_argument("--payload-kg", type=float, default=learning.DEFAULT_PAYLOAD_KG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Versioned simulation-only 15-action residual PPO for MicroDuck arm v1C"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-ppo")
    train.add_argument("--out", required=True)
    train.add_argument("--steps", type=int, default=2048)
    train.add_argument("--seed", type=int, default=101)
    train.add_argument(
        "--leg-residual-rad",
        type=float,
        default=DEFAULT_LEG_RESIDUAL_RAD,
    )
    train.add_argument(
        "--arm-residual-rad",
        type=float,
        default=DEFAULT_ARM_RESIDUAL_RAD,
    )
    _add_env_args(train)

    export = commands.add_parser("export")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--seeds", default="101,102,103")
    _add_env_args(evaluate)

    onnx_evaluate = commands.add_parser("evaluate-onnx-render")
    onnx_evaluate.add_argument("--onnx", required=True)
    onnx_evaluate.add_argument("--out", required=True)
    onnx_evaluate.add_argument("--seeds", default="101,102,103")
    _add_env_args(onnx_evaluate)

    args = parser.parse_args(argv)
    if args.command == "train-ppo":
        result = train_ppo(
            args.out,
            args.steps,
            args.seed,
            case=args.case,
            mode=args.mode,
            candidate=args.candidate,
            max_steps=args.max_steps,
            payload_kg=args.payload_kg,
            leg_residual_rad=args.leg_residual_rad,
            arm_residual_rad=args.arm_residual_rad,
        )
    elif args.command == "export":
        result = export_policy(args.checkpoint, args.out)
    elif args.command == "evaluate":
        result = evaluate_policy(
            args.checkpoint,
            (int(item) for item in args.seeds.split(",")),
            args.out,
            case=args.case,
            mode=args.mode,
            candidate=args.candidate,
            max_steps=args.max_steps,
            payload_kg=args.payload_kg,
        )
    else:
        result = evaluate_onnx_render(
            args.onnx,
            (int(item) for item in args.seeds.split(",")),
            args.out,
            case=args.case,
            mode=args.mode,
            candidate=args.candidate,
            max_steps=args.max_steps,
            payload_kg=args.payload_kg,
        )
    print(json.dumps({"artifact": str(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
