"""Suspended narrow-bridge environment for Microduck locomotion prototypes."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv

ROBOT_DIR = Path(__file__).resolve().parents[1] / "mjlab_microduck/robot/microduck"

CTRL_DT = C.CTRL_DT
GROUND_Z = 0.0
PLATFORM_TOP_Z = 0.24
PLATFORM_HALF_LENGTH = 0.30
PLATFORM_HALF_WIDTH = 0.24
START_PLATFORM_X = -0.85
END_PLATFORM_X = 0.85
BRIDGE_HALF_LENGTH = 0.55
BRIDGE_HALF_WIDTH = 0.065
BRIDGE_HALF_THICKNESS = 0.015
BRIDGE_MASS = 0.45
BRIDGE_CENTER_Z = PLATFORM_TOP_Z - BRIDGE_HALF_THICKNESS
FOOT_CONTACT_SETTLE_M = 0.003
ASSISTANCE_LEVELS = (1.0, 0.6, 0.3, 0.1, 0.0)
SPAWN_X_LEVELS = (0.0, -0.30, -0.55, -0.75, START_PLATFORM_X - 0.05)

CURRICULUM_KNOBS = MappingProxyType({
    "assistance": "plank restoring force and level torque only",
    "spawn_x": "progressive physical start position from plank to launch platform",
})


def _suspension_length(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def bridge_model(actuator: str = "xml") -> mujoco.MjModel:
    """Build a free suspended plank between two fixed safe platforms."""
    if actuator not in ("xml", "bam"):
        raise ValueError("actuator must be xml or bam")

    spec = mujoco.MjSpec.from_file(str(ROBOT_DIR / "robot_allcollisions.xml"))
    spec.option.timestep = C.PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = 12
    spec.option.ls_iterations = 24

    ground_mat = spec.add_material(name="bridge_ground_mat", rgba=(0.10, 0.12, 0.14, 1))
    platform_mat = spec.add_material(name="bridge_platform_mat", rgba=(0.28, 0.31, 0.34, 1))
    plank_mat = spec.add_material(name="bridge_plank_mat", rgba=(0.52, 0.31, 0.13, 1))
    cable_mat = spec.add_material(name="bridge_cable_mat", rgba=(0.75, 0.77, 0.78, 1))
    del ground_mat, platform_mat, plank_mat, cable_mat

    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=(0, 0, GROUND_Z),
        size=(0, 0, 0.05),
        material="bridge_ground_mat",
        friction=(1.0, 0.005, 0.0001),
    )
    for name, x in (
        ("start_platform", START_PLATFORM_X),
        ("end_platform", END_PLATFORM_X),
    ):
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=2,
            pos=(x, 0, PLATFORM_TOP_Z / 2),
            size=(PLATFORM_HALF_LENGTH, PLATFORM_HALF_WIDTH, PLATFORM_TOP_Z / 2),
            material="bridge_platform_mat",
            friction=(1.1, 0.005, 0.0001),
            priority=1,
        )

    plank = spec.worldbody.add_body(
        name="bridge_plank",
        pos=(0, 0, BRIDGE_CENTER_Z),
    )
    plank.add_freejoint(name="bridge_freejoint")
    plank.add_geom(
        name="bridge_surface",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=2,
        size=(BRIDGE_HALF_LENGTH, BRIDGE_HALF_WIDTH, BRIDGE_HALF_THICKNESS),
        mass=BRIDGE_MASS,
        material="bridge_plank_mat",
        friction=(1.15, 0.006, 0.0002),
        priority=1,
        condim=4,
    )

    anchor_z = 0.77
    anchor_y = 0.17
    attach_y = BRIDGE_HALF_WIDTH - 0.008
    attach_x = BRIDGE_HALF_LENGTH - 0.10
    vertical = anchor_z - BRIDGE_CENTER_Z
    cable_length = _suspension_length(0.0, anchor_y - attach_y, vertical)
    for longitudinal, x in (("start", -attach_x), ("end", attach_x)):
        spec.worldbody.add_geom(
            name=f"bridge_{longitudinal}_crossbar", type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=(x, 0, anchor_z + 0.014), size=(0.014, 0.24, 0.014),
            material="bridge_cable_mat", contype=0, conaffinity=0, group=2,
        )
        for post_side in (-1, 1):
            spec.worldbody.add_geom(
                name=f"bridge_{longitudinal}_post_{post_side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=(x, post_side * 0.225, anchor_z / 2),
                size=(0.014, 0.014, anchor_z / 2), material="bridge_cable_mat",
                contype=0, conaffinity=0, group=2,
            )
        for lateral, anchor_side, attach_side in (
            ("left", anchor_y, attach_y),
            ("right", -anchor_y, -attach_y),
        ):
            name = f"bridge_{longitudinal}_{lateral}"
            spec.worldbody.add_site(
                name=f"{name}_anchor",
                pos=(x, anchor_side, anchor_z),
                size=(0.004,),
                rgba=(0.75, 0.77, 0.78, 1),
            )
            plank.add_site(
                name=f"{name}_attach",
                pos=(x, attach_side, 0),
                size=(0.004,),
                rgba=(0.75, 0.77, 0.78, 1),
            )
            tendon = spec.add_tendon(
                name=f"{name}_cable",
                stiffness=320.0,
                damping=2.0,
                springlength=(0.0, cable_length - 0.0015),
                limited=True,
                range=(0.0, cable_length + 0.008),
                width=0.0015,
                rgba=(0.75, 0.77, 0.78, 1),
                solref_limit=(0.015, 1.0),
                solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
            )
            tendon.wrap_site(f"{name}_anchor")
            tendon.wrap_site(f"{name}_attach")

    spec.worldbody.add_light(pos=(0, -1.5, 3), dir=(0, 0.3, -1),
                             diffuse=(0.85, 0.85, 0.85), ambient=(0.3, 0.3, 0.3))
    spec.add_key(
        name="STAND",
        qpos=np.concatenate((
            (0, 0, 0.12, 1, 0, 0, 0),
            C.DEFAULT_POSE,
            (0, 0, BRIDGE_CENTER_Z, 1, 0, 0, 0),
        )),
        ctrl=C.DEFAULT_POSE,
    )
    model = spec.compile()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    return model


def curriculum_level(stage: int, *, crossed: bool, elapsed_s: float) -> int:
    """Advance after an honest crossing; retreat after an immediate failure."""
    if (
        not 0 <= stage < len(ASSISTANCE_LEVELS)
        or not math.isfinite(elapsed_s)
        or elapsed_s < 0
    ):
        raise ValueError("invalid curriculum stage or elapsed time")
    if crossed:
        return min(stage + 1, len(ASSISTANCE_LEVELS) - 1)
    if elapsed_s < 2.0:
        return max(stage - 1, 0)
    return stage


class BridgeEnv(MicroduckWalkEnv):
    """Gymnasium environment for crossing a physically suspended narrow plank.

    The policy receives the standard 61-float Microduck observation only.
    Bridge pose, world progress, contacts, and crossing state are privileged
    metrics. Assistance acts on the plank, never on the robot, and the inherited
    locomotion reward is identical at every curriculum level.
    """

    def __init__(
        self,
        *,
        max_episode_s: float = 20.0,
        seed: int | None = None,
        obs_noise: bool = False,
        domain_rand: bool = False,
        pushes: bool = False,
        random_yaw: bool = False,
        actuator: str = "xml",
        action_delay: bool = False,
        command: tuple[float, float, float] | None = (0.25, 0.0, 0.0),
        navigation: bool = True,
        assistance: float | None = None,
        curriculum: bool = False,
        model: mujoco.MjModel | None = None,
    ) -> None:
        if not math.isfinite(max_episode_s) or max_episode_s <= 0:
            raise ValueError("episode length must be positive and finite")
        if assistance is None:
            assistance = 1.0 if curriculum else 0.0
        if not math.isfinite(assistance) or not 0 <= assistance <= 1:
            raise ValueError("assistance must be finite and between zero and one")
        if curriculum and assistance not in ASSISTANCE_LEVELS:
            raise ValueError("curriculum assistance must be one of ASSISTANCE_LEVELS")

        self.fixed_command = self._validate_command(command)
        self.navigation = bool(navigation)
        self.assistance = float(assistance)
        self.curriculum = bool(curriculum)
        self.stage = (
            ASSISTANCE_LEVELS.index(assistance)
            if assistance in ASSISTANCE_LEVELS
            else len(ASSISTANCE_LEVELS) - 1
        )
        self.pushes = bool(pushes)
        self._bridge_random_yaw = bool(random_yaw)
        self._episode_finished = False
        self._crossed = False
        self._bridge_contact_observed = False
        self._next_push = math.inf
        self._last_push_force = 0.0

        super().__init__(
            max_episode_s=max_episode_s,
            command_resample_s=5.0,
            obs_noise=obs_noise,
            domain_rand=domain_rand,
            action_delay=action_delay,
            random_yaw=False,
            seed=seed,
            actuator_force=actuator,
            model=model if model is not None else bridge_model(actuator),
            terminate_on_fall=False,
            height_termination=False,
        )

        self.bridge_body_id = self.model.body("bridge_plank").id
        self.bridge_geom_id = self.model.geom("bridge_surface").id
        self.bridge_qpos_adr = int(self.model.joint("bridge_freejoint").qposadr[0])
        self.bridge_qvel_adr = int(self.model.joint("bridge_freejoint").dofadr[0])
        self.start_platform_geom = self.model.geom("start_platform").id
        self.end_platform_geom = self.model.geom("end_platform").id
        self.support_geoms = {
            self.start_platform_geom,
            self.bridge_geom_id,
            self.end_platform_geom,
        }
        self._bridge_nominal_position = np.array((0.0, 0.0, BRIDGE_CENTER_Z))

    @staticmethod
    def _validate_command(command):
        if command is None:
            return None
        array = np.asarray(command, dtype=np.float32)
        if array.shape != (3,) or not np.isfinite(array).all():
            raise ValueError("command must contain three finite numbers")
        return array.copy()

    def set_command(self, command: tuple[float, float, float]) -> None:
        validated = self._validate_command(command)
        if validated is None:
            raise ValueError("set_command requires three numbers")
        self.fixed_command = validated
        self.twist_cmd[:] = validated

    def _sample_commands(self) -> None:
        if self.fixed_command is not None:
            self.twist_cmd[:] = self.fixed_command
        else:
            self.twist_cmd[:] = (
                self._rng.uniform(0.12, 0.24),
                self._rng.uniform(-0.025, 0.025),
                self._rng.uniform(-0.12, 0.12),
            )
        self.head_cmd[:] = 0
        self.body_cmd[:] = 0

    def _update_navigation_command(self) -> None:
        if not self.navigation or self.fixed_command is None:
            return
        rotation = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        heading = rotation[:, 0].astype(np.float64)
        heading[2] = 0
        norm = float(np.linalg.norm(heading))
        heading = heading / norm if norm > 1e-9 else np.array((1.0, 0.0, 0.0))
        side = np.array((-heading[1], heading[0], 0.0))
        reached_destination = (
            self.data.xpos[self.trunk_body_id, 0] >= END_PLATFORM_X - 0.05
        )
        desired_world = np.array((
            0.0 if reached_destination else float(self.fixed_command[0]),
            float(np.clip(-1.2 * self.data.xpos[self.trunk_body_id, 1], -0.12, 0.12)),
            0.0,
        ))
        yaw_error = math.atan2(float(heading[1]), float(heading[0]))
        self.twist_cmd[:] = (
            float(desired_world @ heading),
            float(desired_world @ side),
            float(np.clip(-1.5 * yaw_error, -0.6, 0.6)),
        )

    def _get_obs(self) -> np.ndarray:
        self._update_navigation_command()
        return super()._get_obs()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.curriculum and self._episode_finished:
            self.stage = curriculum_level(
                self.stage,
                crossed=self._crossed,
                elapsed_s=self.step_count * CTRL_DT,
            )
            self.assistance = ASSISTANCE_LEVELS[self.stage]

        self._episode_finished = False
        self._crossed = False
        self._bridge_contact_observed = False
        super().reset(seed=seed, options=options)

        spawn_x = (
            SPAWN_X_LEVELS[self.stage]
            if self.curriculum
            else START_PLATFORM_X - 0.05
        )
        self.data.qpos[:3] = (
            spawn_x,
            self._rng.uniform(-0.015, 0.015),
            self.stand_z + PLATFORM_TOP_Z - FOOT_CONTACT_SETTLE_M,
        )
        yaw = (
            self._rng.uniform(-math.radians(5), math.radians(5))
            if self._bridge_random_yaw
            else 0.0
        )
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.qpos[self.joint_qpos_adr] = (
            C.DEFAULT_POSE + self._rng.uniform(-0.025, 0.025, C.NUM_JOINTS)
        )

        roll = self._rng.uniform(-math.radians(1.5), math.radians(1.5))
        pitch = self._rng.uniform(-math.radians(0.8), math.radians(0.8))
        rotation = np.empty(4)
        roll_quat = np.array((math.cos(roll / 2), math.sin(roll / 2), 0, 0))
        pitch_quat = np.array((math.cos(pitch / 2), 0, math.sin(pitch / 2), 0))
        mujoco.mju_mulQuat(rotation, pitch_quat, roll_quat)
        self.data.qpos[
            self.bridge_qpos_adr:self.bridge_qpos_adr + 7
        ] = (*self._bridge_nominal_position, *rotation)
        self.data.qvel[:] = 0
        self.data.qvel[self.bridge_qvel_adr + 3:self.bridge_qvel_adr + 6] = (
            self._rng.uniform(-0.025, 0.025, 3)
        )
        self.data.xfrc_applied[:] = 0
        self.data.ctrl[:] = self.data.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(self.model, self.data)
        if self.bam is not None:
            self.bam.reset(self.data.qpos[self.joint_qpos_adr])
        self.prev_joint_vel[:] = self._joint_vel()
        self._next_push = self._rng.uniform(1.5, 3.0) if self.pushes else math.inf
        self._last_push_force = 0.0
        return self._get_obs(), self.metrics()

    def _apply_physics_curriculum(self) -> None:
        self.data.xfrc_applied[:] = 0
        self._last_push_force = 0.0
        if self.assistance:
            pos = self.data.xpos[self.bridge_body_id]
            vel = self.data.qvel[self.bridge_qvel_adr:self.bridge_qvel_adr + 3]
            rotation = self.data.xmat[self.bridge_body_id].reshape(3, 3)
            angular_world = (
                rotation
                @ self.data.qvel[self.bridge_qvel_adr + 3:self.bridge_qvel_adr + 6]
            )
            displacement = pos - self._bridge_nominal_position
            restoring_force = -45.0 * displacement - 4.0 * vel
            restoring_force[2] = -25.0 * displacement[2] - 2.5 * vel[2]
            restoring_torque = (
                3.5 * np.cross(rotation[:, 2], np.array((0.0, 0.0, 1.0)))
                - 0.18 * angular_world
            )
            self.data.xfrc_applied[self.bridge_body_id, :3] = (
                self.assistance * restoring_force
            )
            self.data.xfrc_applied[self.bridge_body_id, 3:] = (
                self.assistance * restoring_torque
            )

        if self.step_count * CTRL_DT >= self._next_push:
            self._last_push_force = float(self._rng.choice((-3.0, 3.0)))
            self.data.xfrc_applied[self.trunk_body_id, 1] = self._last_push_force
            self._next_push += self._rng.uniform(1.5, 3.0)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (C.NUM_JOINTS,) or not np.isfinite(action).all():
            raise ValueError("actions must be finite with shape (14,)")
        if self._episode_finished:
            raise RuntimeError("reset is required after a terminal episode")

        self._apply_physics_curriculum()
        observation, reward, terminated, truncated, info = super().step(action)
        metrics = self.metrics()
        reasons = []
        if metrics["robot_ground_contact"]:
            reasons.append("robot_ground_contact")
        if metrics["tilt_deg"] > 70:
            reasons.append("tilt")
        if metrics["trunk_height_m"] < 0.10:
            reasons.append("height")
        if (
            abs(metrics["trunk_lateral_m"]) > PLATFORM_HALF_WIDTH + 0.12
            and metrics["supported_feet"] == 0
        ):
            reasons.append("fell_off_side")
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            reasons.append("nonfinite_physics")

        terminated = terminated or bool(reasons)
        self._episode_finished = terminated or truncated
        info.update(metrics)
        info["termination_reasons"] = reasons
        if self._episode_finished:
            info["episode_rewards"] = dict(self.reward_sums)
        return observation, reward, terminated, truncated, info

    def _contact_state(self) -> dict:
        feet = self.foot_geoms
        foot_supports: dict[str, set[int]] = {"left": set(), "right": set()}
        foot_ground = {"left": False, "right": False}
        robot_ground = False
        robot_bridge = False
        nonfoot_support = False
        foot_geom_ids = set(feet.values())
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            pair = {geom1, geom2}
            for side, foot in feet.items():
                if foot not in pair:
                    continue
                other = geom2 if geom1 == foot else geom1
                if other in self.support_geoms:
                    foot_supports[side].add(other)
                if other == self.floor_geom:
                    foot_ground[side] = True
            if self.floor_geom in pair:
                other = geom2 if geom1 == self.floor_geom else geom1
                body_id = int(self.model.geom_bodyid[other])
                if body_id not in (0, self.bridge_body_id):
                    robot_ground = True
            if self.bridge_geom_id in pair:
                other = geom2 if geom1 == self.bridge_geom_id else geom1
                body_id = int(self.model.geom_bodyid[other])
                if body_id not in (0, self.bridge_body_id):
                    robot_bridge = True
            if pair & self.support_geoms:
                support = next(iter(pair & self.support_geoms))
                other = geom2 if geom1 == support else geom1
                body_id = int(self.model.geom_bodyid[other])
                if body_id not in (0, self.bridge_body_id) and other not in foot_geom_ids:
                    nonfoot_support = True
        return {
            "foot_supports": foot_supports,
            "foot_ground": foot_ground,
            "robot_ground": robot_ground,
            "robot_bridge": robot_bridge,
            "nonfoot_support": nonfoot_support,
        }

    def _foot_contacts(self) -> dict[str, bool]:
        contacts = self._contact_state()["foot_supports"]
        return {side: bool(surfaces) for side, surfaces in contacts.items()}

    def metrics(self) -> dict:
        contacts = self._contact_state()
        foot_supports = contacts["foot_supports"]
        feet_on_bridge = (
            int(self.bridge_geom_id in foot_supports["left"])
            + int(self.bridge_geom_id in foot_supports["right"])
        )
        self._bridge_contact_observed = (
            self._bridge_contact_observed or feet_on_bridge > 0
        )
        left_end = self.end_platform_geom in foot_supports["left"]
        right_end = self.end_platform_geom in foot_supports["right"]
        full_traversal = float(self.data.xpos[self.trunk_body_id, 0]) >= (
            END_PLATFORM_X - 0.08
        )
        accepted = (
            full_traversal
            and left_end
            and right_end
            and not any(contacts["foot_ground"].values())
            and not contacts["robot_ground"]
            and not contacts["nonfoot_support"]
            and self._bridge_contact_observed
        )
        self._crossed = self._crossed or accepted

        rotation = self.data.xmat[self.bridge_body_id].reshape(3, 3)
        plank_up = rotation[:, 2]
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        pitch = math.atan2(
            float(-rotation[2, 0]),
            math.sqrt(float(rotation[2, 1] ** 2 + rotation[2, 2] ** 2)),
        )
        gravity = self._projected_gravity()
        return {
            "elapsed_s": self.step_count * CTRL_DT,
            "assistance": self.assistance,
            "curriculum_stage": self.stage,
            "command": self.twist_cmd.tolist(),
            "trunk_x_m": float(self.data.xpos[self.trunk_body_id, 0]),
            "trunk_lateral_m": float(self.data.xpos[self.trunk_body_id, 1]),
            "trunk_height_m": float(self.data.xpos[self.trunk_body_id, 2]),
            "tilt_deg": math.degrees(
                math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))
            ),
            "bridge_position": self.data.xpos[self.bridge_body_id].tolist(),
            "bridge_roll_deg": math.degrees(roll),
            "bridge_pitch_deg": math.degrees(pitch),
            "bridge_up_z": float(plank_up[2]),
            "bridge_speed_mps": float(
                np.linalg.norm(
                    self.data.qvel[
                        self.bridge_qvel_adr:self.bridge_qvel_adr + 3
                    ]
                )
            ),
            "left_support": sorted(foot_supports["left"]),
            "right_support": sorted(foot_supports["right"]),
            "supported_feet": int(bool(foot_supports["left"]))
            + int(bool(foot_supports["right"])),
            "feet_on_bridge": feet_on_bridge,
            "feet_on_end_platform": int(left_end) + int(right_end),
            "foot_ground_contacts": int(contacts["foot_ground"]["left"])
            + int(contacts["foot_ground"]["right"]),
            "robot_ground_contact": contacts["robot_ground"],
            "robot_bridge_contact": contacts["robot_bridge"],
            "nonfoot_support": contacts["nonfoot_support"],
            "full_traversal": full_traversal,
            "bridge_contact_observed": self._bridge_contact_observed,
            "crossing_accepted": self._crossed,
            "push_force_n": self._last_push_force,
        }

    def recipe_metrics(self) -> dict[str, float]:
        """Numeric-only privileged metrics for training logs and evaluation."""
        metrics = self.metrics()
        return {
            "bridge_crossed": float(metrics["crossing_accepted"]),
            "bridge_progress_m": float(
                metrics["trunk_x_m"] - (START_PLATFORM_X - 0.05)
            ),
            "bridge_assistance": float(self.assistance),
            "robot_floor_contact": float(metrics["robot_ground_contact"]),
            "nonfoot_support": float(metrics["nonfoot_support"]),
            "feet_support": float(metrics["supported_feet"]),
            "bridge_contact_observed": float(
                metrics["bridge_contact_observed"]
            ),
            "feet_on_bridge": float(metrics["feet_on_bridge"]),
            "feet_on_end_platform": float(metrics["feet_on_end_platform"]),
            "bridge_roll_deg": float(metrics["bridge_roll_deg"]),
            "bridge_pitch_deg": float(metrics["bridge_pitch_deg"]),
            "bridge_speed_mps": float(metrics["bridge_speed_mps"]),
            "robot_tilt_deg": float(metrics["tilt_deg"]),
            "robot_lateral_m": float(metrics["trunk_lateral_m"]),
        }
