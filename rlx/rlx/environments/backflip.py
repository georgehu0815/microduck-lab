"""Assisted launch, unassisted landing policy, and explicit stand handoff."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.contract import DEFAULT_POSE

PROTOCOL_VERSION = "spotter-launch-landing-stand-v2"
RELEASE_ROTATION = 5.3
STAND_POLICY = Path(__file__).resolve().parents[3] / "microduck/policies/alpha_stand.onnx"
LANDING_REWARD_WEIGHTS = {
    "landing_upright": 3.0,
    "landing_pose": 2.0,
    "landing_settle": 1.0,
    "landing_joint_speed_penalty": 0.05,
    "landing_action_rate_penalty": 0.02,
    "landing_action_size_penalty": 0.01,
}


def landing_reward_terms(gravity, joint_offsets, gyro, joint_velocity, action, previous_action):
    upright = float(np.clip((1.0 - gravity[2]) / 2.0, 0.0, 1.0))
    offsets_squared = np.square(joint_offsets)
    pose = float(np.mean(0.5 * np.exp(-offsets_squared / 0.6**2)
                         + 0.5 * np.exp(-offsets_squared / 0.15**2)))
    return {
        "landing_upright": upright,
        "landing_pose": upright * pose,
        "landing_settle": upright * pose * float(np.exp(-np.sum(np.square(gyro)) / 2.0)),
        "landing_joint_speed_penalty": -float(np.mean(np.minimum(np.square(joint_velocity) / 100.0, 1.0))),
        "landing_action_rate_penalty": -float(np.mean(np.minimum(np.square(action - previous_action), 1.0))),
        "landing_action_size_penalty": -float(np.mean(np.minimum(np.square(action) / 16.0, 1.0))),
    }


def protocol_metadata() -> dict:
    return {
        "backflip_protocol_version": PROTOCOL_VERSION,
        "stand_policy_sha256": hashlib.sha256(STAND_POLICY.read_bytes()).hexdigest(),
        "stand_policy": str(STAND_POLICY),
        "assisted_showcase": True,
        "assist_measurement": "assist_torque_norm is the legacy generalized-force zero-detection channel; use assist_force_n and assist_torque_nm for physical units",
        "launch_release_rotation_rad": RELEASE_ROTATION,
        "launch": {"duration_s": 1.2, "terminal_rate_rad_s": 5.0,
                   "peak_target_height_m": 0.32, "release_target_height_m": 0.22,
                   "pitch_kp": 1.5, "pitch_kd": 0.24, "pitch_torque_limit_nm": 1.3,
                   "vertical_kp": 90.0, "vertical_kd": 8.5, "vertical_force_limit_n": 15.0,
                   "lateral_damping": 3.0, "roll_yaw_damping": 0.1},
        "minimum_landing_policy_steps": 10,
        "minimum_pre_handoff_stable_steps": 10,
        "minimum_handoff_rotation_rad": 5.8,
        "training_mode": "unassisted landing from assisted-release states",
        "handoff": "pretrained alpha_stand after braked upright foot contact",
        "landing_reward_version": "observable-two-scale-pose-v1",
        "landing_reward_weights": dict(LANDING_REWARD_WEIGHTS),
    }


@lru_cache(maxsize=1)
def stand_session():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(STAND_POLICY), sess_options=options, providers=["CPUExecutionProvider"])


def stand_action(observation: np.ndarray) -> np.ndarray:
    session = stand_session()
    return session.run(None, {session.get_inputs()[0].name: observation[None].astype(np.float32)})[0][0]


class BackflipEnv(BehaviorEnv):
    def __init__(self, *, backflip_mode="showcase", **kwargs):
        if backflip_mode not in {"landing", "showcase"}:
            raise ValueError("backflip_mode must be landing or showcase")
        self.backflip_mode = backflip_mode
        self.stage = "launch"
        self.landing_policy_steps = 0
        self.landing_stable_steps = 0
        self.maximum_rotation = 0.0
        self.assist_torque_norm = 0.0
        self._last_observation = np.zeros(61, dtype=np.float32)
        super().__init__("backflip", spotter=False, standing_spawns=True,
                         spawn_overrides={"MICRODUCK_EPISODE_S": str(kwargs.get("max_episode_s", 12))}, **kwargs)

    def _compute_reward(self):
        self._lifetime_steps += 1
        self.foot_contact_state = self._foot_contacts()
        self.behavior.state_fn(self)
        terms = landing_reward_terms(
            self._projected_gravity(), self.data.qpos[self.joint_qpos_adr] - DEFAULT_POSE,
            self._gyro, self._joint_vel(), self.last_action, self.prev_action,
        )
        terms = {key: terms[key] * self.weight_overrides.get(key, weight)
                 for key, weight in LANDING_REWARD_WEIGHTS.items()}
        anchor_weight = self.weight_overrides.get("landing_teacher_action_penalty", 0.0)
        if anchor_weight and self.stage == "landing":
            expected_action = stand_action(self._last_observation)
            action_error = float(np.mean(np.square(self.last_action - expected_action)))
            terms["landing_teacher_action_penalty"] = -anchor_weight * (1.0 - math.exp(-action_error / 0.1**2))
        if self.stage != "landing":
            terms = dict.fromkeys(terms, 0.0)
        return float(sum(terms.values())), terms

    def _release(self):
        self.stage = "landing"
        self.spotter_active = False
        self.data.qfrc_applied[:] = 0.0
        self.assist_torque_norm = 0.0

    def _launch_step(self):
        progress = float(np.clip(self.step_count / 60.0, 0., 1.))
        target_rotation = (-2 * progress**3 + 3 * progress**2) * RELEASE_ROTATION + (progress**3 - progress**2) * 6.0
        target_rate = ((-6 * progress**2 + 6 * progress) * RELEASE_ROTATION + (3 * progress**2 - 2 * progress) * 6.0) / 1.2
        height_progress = float(np.clip(target_rotation / RELEASE_ROTATION, 0., 1.))
        blend = 2 * height_progress if height_progress <= 0.5 else 2 * height_progress - 1
        blend = blend**2 * (3 - 2 * blend)
        target_height = 0.12 + blend * 0.20 if height_progress <= 0.5 else 0.32 - blend * 0.10
        pitch = 1.5 * (target_rotation - self._bf_rot) + 0.24 * (target_rate + float(self._gyro[1]))
        vertical = float(np.sum(self.model.body_mass)) * 9.81 + 90 * (target_height - self._trunk_xpos[2]) - 8.5 * self.data.qvel[2]
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[:2] = -3.0 * self.data.qvel[:2]
        self.data.qfrc_applied[2] = np.clip(vertical, 0., 15.)
        self.data.qfrc_applied[3:6] = [-0.1 * self._gyro[0], -np.clip(pitch, -1.3, 1.3), -0.1 * self._gyro[2]]
        self.spotter_active = True
        self.assist_torque_norm = float(np.linalg.norm(self.data.qfrc_applied))
        result = super().step(stand_action(self._last_observation))
        self._last_observation = result[0]
        self.maximum_rotation = max(self.maximum_rotation, float(self._bf_rot))
        return result

    def _launch_ready(self):
        ready = self._bf_rot >= RELEASE_ROTATION and 3.5 <= -float(self._gyro[1]) <= 6.5
        if not ready and self.step_count >= 78:
            raise RuntimeError("Spotter did not reach the declared release-state window")
        return ready

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.stage = "launch"
        self.landing_policy_steps = 0
        self.landing_stable_steps = 0
        self.maximum_rotation = 0.0
        self.assist_torque_norm = 0.0
        self._last_observation = observation
        if self.backflip_mode == "landing":
            for _ in range(250):
                observation, _, terminated, truncated, info = self._launch_step()
                if self._launch_ready():
                    break
                if terminated or truncated:
                    raise RuntimeError("Episode horizon too short to prepare a landing release")
            else:
                raise RuntimeError("Spotter launch did not reach its release angle")
            self._release()
            self.step_count = 0
            observation = self._get_obs()
            self._last_observation = observation
        return observation, info

    def _handoff_ready(self):
        return self._bf_rot >= 5.8 and self.landing_stable_steps >= 10

    def _landing_stable(self):
        return (
            all(self.foot_contact_state.values()) and not self._nonfoot_contact()
            and float(-self._projected_gravity()[2]) >= 0.9
            and float(self._trunk_xpos[2]) >= 0.105
            and float(np.linalg.norm(self._gyro)) < math.sqrt(2)
        )

    def step(self, action):
        if self.backflip_mode == "showcase":
            if self.stage == "launch" and self._launch_ready():
                self._release()
            if self.stage == "launch":
                return self._launch_step()
            if self.stage == "landing" and self._handoff_ready():
                self.stage = "stand"
        self.data.qfrc_applied[:] = 0.0
        self.assist_torque_norm = 0.0
        self.spotter_active = False
        if self.stage == "stand":
            action = stand_action(self._last_observation)
        else:
            if self.landing_policy_steps == 0:
                self._delayed_action = np.asarray(action, dtype=np.float32).copy()
            self.landing_policy_steps += 1
        result = super().step(action)
        self._last_observation = result[0]
        self.maximum_rotation = max(self.maximum_rotation, float(self._bf_rot))
        if self.stage == "landing":
            self.landing_stable_steps = self.landing_stable_steps + 1 if self._landing_stable() else 0
        return result

    def _nonfoot_contact(self):
        foot_bodies = {int(self.model.geom_bodyid[geom]) for geom in self.foot_geoms.values()}
        nonfoot = False
        for contact in self.data.contact:
            first, second = int(contact.geom1), int(contact.geom2)
            if self.floor_geom in (first, second):
                other = second if first == self.floor_geom else first
                nonfoot |= int(self.model.geom_bodyid[other]) not in foot_bodies
        return nonfoot

    def recipe_metrics(self):
        return {
            "rotation_rad": self.maximum_rotation,
            "spotter_active": float(self.spotter_active),
            "assist_torque_norm": self.assist_torque_norm,
            "assist_force_n": float(np.linalg.norm(self.data.qfrc_applied[:3])),
            "assist_torque_nm": float(np.linalg.norm(self.data.qfrc_applied[3:6])),
            "launch_complete": float(self.stage != "launch"),
            "landing_policy_steps": float(self.landing_policy_steps),
            "handed_off": float(self.stage == "stand"),
            "upright": float(-self._projected_gravity()[2]),
            "height_m": float(self._trunk_xpos[2]),
            "both_feet": float(all(self.foot_contact_state.values())),
            "nonfoot_contact": float(self._nonfoot_contact()),
            "gyro_norm": float(np.linalg.norm(self._gyro)),
        }
