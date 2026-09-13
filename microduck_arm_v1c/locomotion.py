"""Simulation-only adapter from legacy Microduck policies to v1-C legs.

The shipped policies consume the fixed 61-observation / 14-action contract:
left leg (5), head/neck (4), right leg (5).  The v1-C model instead exposes
left leg (5), right leg (5), arm (5).  This adapter reconstructs the legacy
observation from real simulated trunk and leg state, virtualizes the missing
head state, and returns absolute targets for only the ten v1-C leg joints.

It is an experimental simulator bridge, not a hardware controller or a
sim-to-real claim.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np

from microduck_local import contract as legacy


POLICY_SHA256 = {
    "alpha_stand.onnx": (
        "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b"
    ),
    "alpha_walking.onnx": (
        "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c"
    ),
}

class StandConfiguration(TypedDict):
    name: str
    policy_name: str
    source_policy_sha256: str
    action_gain: float
    virtual_head: str
    screening: str
    hardware_ready: bool


SCREENED_STAND_CONFIGURATION: StandConfiguration = {
    "name": "alpha-stand-v1c-fixed-head-gain-1p25",
    "policy_name": "alpha_stand.onnx",
    "source_policy_sha256": POLICY_SHA256["alpha_stand.onnx"],
    "action_gain": 1.25,
    "virtual_head": "fixed",
    "screening": "free-base simulation only",
    "hardware_ready": False,
}

LEGACY_LEG_INDICES = np.asarray(legacy.LEG_JOINT_IDS, dtype=np.int64)
LEGACY_HEAD_INDICES = np.asarray(legacy.HEAD_JOINT_IDS, dtype=np.int64)
V1C_LEG_JOINT_NAMES = tuple(legacy.JOINT_NAMES[index] for index in LEGACY_LEG_INDICES)
DEFAULT_LEG_TARGETS = legacy.DEFAULT_POSE[LEGACY_LEG_INDICES].copy()


def policy_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LegacyLegPolicyAdapter:
    """Run a legacy 61x14 policy against the free-base v1-C simulation.

    ``leg_targets`` returns ten absolute joint angles in v1-C leg order.  The
    caller owns composition with the five arm targets and must pass the full
    target through ``env.normalize_targets`` before ``env.step``.
    """

    def __init__(
        self,
        policy_path: str | Path,
        *,
        action_gain: float = 1.0,
        twist_command: Sequence[float] = (0.0, 0.0, 0.0),
        virtual_head: str = "fixed",
        virtual_head_slew_rad_s: float = 6.0,
        verify_source_hash: bool = True,
        configuration_name: str = "custom-legacy-leg-adapter",
    ):
        path = Path(policy_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not np.isfinite(action_gain) or action_gain <= 0:
            raise ValueError("action_gain must be finite and positive")
        command = np.asarray(twist_command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("twist_command must be a finite 3-vector")
        if virtual_head not in {"fixed", "track"}:
            raise ValueError("virtual_head must be 'fixed' or 'track'")
        if not np.isfinite(virtual_head_slew_rad_s) or virtual_head_slew_rad_s <= 0:
            raise ValueError("virtual_head_slew_rad_s must be finite and positive")
        if not configuration_name:
            raise ValueError("configuration_name must be non-empty")

        source_hash = policy_sha256(path)
        expected_hash = POLICY_SHA256.get(path.name)
        if verify_source_hash and expected_hash is None:
            raise ValueError(f"No approved source hash for policy {path.name!r}")
        if verify_source_hash and source_hash != expected_hash:
            raise ValueError(
                f"Source policy hash mismatch for {path.name}: {source_hash}"
            )

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, legacy.OBS_DIM]:
            raise ValueError("Legacy policy input must have shape [1, 61]")
        if len(outputs) != 1 or outputs[0].shape != [1, legacy.NUM_JOINTS]:
            raise ValueError("Legacy policy output must have shape [1, 14]")

        self.policy_path = path.resolve()
        self.source_sha256 = source_hash
        self.action_gain = float(action_gain)
        self.twist_command = command.copy()
        self.virtual_head = virtual_head
        self.virtual_head_slew_rad_s = float(virtual_head_slew_rad_s)
        self.configuration_name = configuration_name
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.provenance = {
            "configuration_name": configuration_name,
            "source_policy": str(self.policy_path),
            "source_policy_sha256": source_hash,
            "source_contract": "obs61-action14:left5-head4-right5",
            "target_contract": "v1c-left5-right5-absolute-leg-targets",
            "virtual_missing_head": virtual_head,
            "action_gain": self.action_gain,
            "simulation_only": True,
            "hardware_ready": False,
            "domain_shift": "legacy headed robot policy adapted to headless v1-C",
        }
        self.reset()

    @classmethod
    def screened_stand(cls, policy_directory: str | Path) -> "LegacyLegPolicyAdapter":
        """Construct the best bounded free-base standing configuration."""
        config = SCREENED_STAND_CONFIGURATION
        return cls(
            Path(policy_directory) / config["policy_name"],
            action_gain=config["action_gain"],
            virtual_head=config["virtual_head"],
            configuration_name=config["name"],
        )

    def reset(self) -> None:
        """Reset virtual head state and policy action history."""
        self.virtual_head_q = legacy.DEFAULT_POSE[LEGACY_HEAD_INDICES].copy()
        self.virtual_head_dq = np.zeros(4, dtype=np.float32)
        self.last_action = np.zeros(legacy.NUM_JOINTS, dtype=np.float32)
        self.last_observation = np.zeros(legacy.OBS_DIM, dtype=np.float32)

    def observation(
        self,
        env,
        *,
        twist_command: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Build the legacy policy observation from live v1-C simulator state."""
        if env.mode != "free":
            raise ValueError("Legacy locomotion adapter requires the free-base model")
        command = (
            self.twist_command
            if twist_command is None
            else np.asarray(twist_command, dtype=np.float32)
        )
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("twist_command must be a finite 3-vector")

        joint_ids = np.asarray(
            [env.model.joint(name).id for name in V1C_LEG_JOINT_NAMES],
            dtype=np.int64,
        )
        qpos_indices = env.model.jnt_qposadr[joint_ids]
        dof_indices = env.model.jnt_dofadr[joint_ids]
        leg_q = env.data.qpos[qpos_indices].astype(np.float32)
        leg_dq = env.data.qvel[dof_indices].astype(np.float32)

        joint_q = legacy.DEFAULT_POSE.copy()
        joint_dq = np.zeros(legacy.NUM_JOINTS, dtype=np.float32)
        joint_q[LEGACY_LEG_INDICES] = leg_q
        joint_q[LEGACY_HEAD_INDICES] = self.virtual_head_q
        joint_dq[LEGACY_LEG_INDICES] = leg_dq
        joint_dq[LEGACY_HEAD_INDICES] = self.virtual_head_dq

        rotation = env.data.xmat[env.trunk_id].reshape(3, 3)
        obs = np.zeros(legacy.OBS_DIM, dtype=np.float32)
        obs[0:3] = env.data.sensor("imu_ang_vel").data
        obs[3:6] = rotation.T @ np.array([0.0, 0.0, -1.0])
        obs[6:20] = joint_q - legacy.DEFAULT_POSE
        obs[20:34] = joint_dq
        obs[34:48] = self.last_action
        obs[48:51] = command
        if not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite legacy locomotion observation")
        self.last_observation = obs.copy()
        return obs

    def leg_targets(
        self,
        env,
        *,
        twist_command: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Return absolute targets for the ten v1-C leg joints."""
        observation = self.observation(env, twist_command=twist_command)
        action = self.session.run(
            [self.output_name],
            {self.input_name: observation[None]},
        )[0][0].astype(np.float32)
        if action.shape != (legacy.NUM_JOINTS,) or not np.isfinite(action).all():
            raise FloatingPointError("Legacy policy returned an invalid action")

        applied = np.clip(action, -4.0, 4.0)
        targets = (
            DEFAULT_LEG_TARGETS + self.action_gain * applied[LEGACY_LEG_INDICES]
        ).astype(np.float32)
        self._advance_virtual_head(applied, float(env.dt))
        self.last_action = action.copy()
        return targets

    def _advance_virtual_head(self, action: np.ndarray, dt: float) -> None:
        if self.virtual_head == "fixed":
            self.virtual_head_q[:] = legacy.DEFAULT_POSE[LEGACY_HEAD_INDICES]
            self.virtual_head_dq[:] = 0.0
            return
        target = (
            legacy.DEFAULT_POSE[LEGACY_HEAD_INDICES]
            + action[LEGACY_HEAD_INDICES]
        )
        requested_velocity = (target - self.virtual_head_q) / dt
        self.virtual_head_dq = np.clip(
            requested_velocity,
            -self.virtual_head_slew_rad_s,
            self.virtual_head_slew_rad_s,
        ).astype(np.float32)
        self.virtual_head_q += self.virtual_head_dq * dt
