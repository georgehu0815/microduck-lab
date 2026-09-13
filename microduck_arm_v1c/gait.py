"""Stateful free-base walking controller for the v1-C simulation.

The controller owns its policy history and route state.  It returns only the
ten absolute leg targets so the same controller can be composed with either a
stowed arm or the walking-carry arm controller.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import mujoco
import numpy as np

from .config import ROOT
from .locomotion import LegacyLegPolicyAdapter


class GaitPhase(str, Enum):
    CLEAR_OBSTACLE = "clear_obstacle"
    ADVANCE = "advance"
    RECOVER = "recover"


class IndependentWalkingController:
    """Feedback wrapper around the shipped walking policy.

    The raised-table scene blocks a straight approach.  The controller first
    requests a strong reverse/side-stepping bout until measured lateral and
    arm-to-table clearances are both sufficient, then advances around the
    table edge. Excess tilt temporarily selects a zero-velocity recovery
    command without discarding policy history.
    """

    def __init__(
        self,
        policy_path: str | Path | None = None,
        *,
        clearance_y_m: float = -0.12,
        clearance_distance_m: float = 0.19,
        recovery_tilt_rad: float = 0.18,
        max_leg_excursion_rad: float = 0.5,
        world_frame_commands: bool = False,
    ):
        self.policy = LegacyLegPolicyAdapter(
            policy_path
            or ROOT / "microduck_local" / "policies" / "alpha_walking.onnx",
            action_gain=2.0,
            verify_source_hash=True,
            configuration_name="v1c-independent-feedback-walking",
        )
        self.clearance_y_m = float(clearance_y_m)
        self.clearance_distance_m = float(clearance_distance_m)
        self.recovery_tilt_rad = float(recovery_tilt_rad)
        self.max_leg_excursion_rad = float(max_leg_excursion_rad)
        self.world_frame_commands = bool(world_frame_commands)
        self.provenance = {
            **self.policy.provenance,
            "controller": "stateful_feedback_route",
            "leg_targets_only": True,
            "scene_strategy": "lateral-clearance-before-forward-advance",
            "max_leg_excursion_rad": self.max_leg_excursion_rad,
            "route_command_frame": "world" if self.world_frame_commands else "body",
            "clearance_y_m": self.clearance_y_m,
            "clearance_distance_m": self.clearance_distance_m,
            "clearance_action_gain": 1.5,
            "advance_action_gain": 1.2,
            "recovery_action_gain": 1.25,
            "loaded_world_forward_command_m_s": .20,
            "loaded_corridor_offset_y_m": -.20,
        }
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.phase = GaitPhase.CLEAR_OBSTACLE
        self.active_phase = self.phase
        self.origin_xy: np.ndarray | None = None
        self.last_command = np.zeros(3, dtype=np.float32)
        self.last_clearance_distance_m = 0.0
        self.last_raw_targets = np.zeros(10, dtype=np.float32)
        self.last_saturated_joint_count = 0

    def leg_targets(self, env) -> np.ndarray:
        """Return ten finite absolute leg-joint targets in v1-C order."""
        if env.mode != "free":
            raise ValueError("Independent walking requires the free-base model")
        if self.origin_xy is None:
            self.origin_xy = env.data.xpos[env.trunk_id, :2].copy()

        displacement = env.data.xpos[env.trunk_id, :2] - self.origin_xy
        rotation = env.data.xmat[env.trunk_id].reshape(3, 3)
        tilt = float(
            np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
        )
        self.last_clearance_distance_m = self._mount_surface_distance(env)

        if tilt > self.recovery_tilt_rad:
            phase = GaitPhase.RECOVER
            command = np.zeros(3, dtype=np.float32)
            self.policy.action_gain = 1.25
        elif self.phase == GaitPhase.CLEAR_OBSTACLE:
            cleared = (
                displacement[1] <= self.clearance_y_m
                and self.last_clearance_distance_m >= self.clearance_distance_m
            )
            if cleared:
                self.phase = GaitPhase.ADVANCE
            phase = self.phase
            if phase == GaitPhase.CLEAR_OBSTACLE:
                command = np.array([-0.30, -0.30, 0.0], dtype=np.float32)
                self.policy.action_gain = 1.5
            else:
                command = self._advance_command()
        else:
            phase = GaitPhase.ADVANCE
            command = self._advance_command()

        if self.world_frame_commands:
            if phase == GaitPhase.ADVANCE:
                corridor_error = self.origin_xy[1] - .20 - env.data.xpos[env.trunk_id, 1]
                command[:2] = [.20, np.clip(2.0 * corridor_error, -.2, .2)]
            command[:2] = np.clip(rotation[:2, :2].T @ command[:2], -.3, .3)
        raw_targets = self.policy.leg_targets(env, twist_command=command)
        lower = np.maximum(
            env.joint_limits[:10, 0],
            env.home[:10] - self.max_leg_excursion_rad,
        )
        upper = np.minimum(
            env.joint_limits[:10, 1],
            env.home[:10] + self.max_leg_excursion_rad,
        )
        targets = np.clip(raw_targets, lower, upper)
        if targets.shape != (10,) or not np.isfinite(targets).all():
            raise FloatingPointError("Invalid independent gait target")
        self.last_raw_targets = raw_targets.copy()
        self.last_saturated_joint_count = int(
            np.count_nonzero(np.abs(targets - raw_targets) > 1e-7)
        )
        self.last_command = command
        self.active_phase = phase
        return targets.astype(np.float32)

    def _advance_command(self) -> np.ndarray:
        self.policy.action_gain = 1.2
        return np.array([0.18, -0.30, 0.0], dtype=np.float32)

    @staticmethod
    def _mount_surface_distance(env) -> float:
        fromto = np.empty(6, dtype=float)
        return float(
            mujoco.mj_geomDistance(
                env.model,
                env.data,
                env.model.geom("arm_mount_plate").id,
                env.model.geom("work_surface").id,
                1.0,
                fromto,
            )
        )
