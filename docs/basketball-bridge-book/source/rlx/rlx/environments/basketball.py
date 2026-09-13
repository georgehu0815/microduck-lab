"""CPU MuJoCo basketball prototype grounded in the playground b11 release."""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "microduck-playground"
BALL_RADIUS = 0.12
BALL_MASS = 0.62
CTRL_DT = C.CTRL_DT
HOLD_LEVELS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
REWARD_WEIGHTS = MappingProxyType({
    "linear_tracking": 1.0,
    "yaw_tracking": 0.5,
    "upright": 1.0,
    "centered": 2.0,
    "feet_on_ball": 1.0,
    "height": 1.0,
    "ball_speed_penalty": 0.05,
    "angular_velocity_penalty": 0.05,
    "action_rate_penalty": 0.2,
})


def basketball_model(actuator: str = "bam") -> mujoco.MjModel:
    if actuator not in ("bam", "xml"):
        raise ValueError("actuator must be bam or xml")
    robot_dir = Path(__file__).resolve().parents[1] / "mjlab_microduck/robot/microduck"
    asset_dir = REFERENCE / "src/mjlab_microduck/robot/assets/basketball"
    spec = mujoco.MjSpec.from_file(str(robot_dir / "robot_allcollisions.xml"))
    spec.option.timestep = C.PHYSICS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.iterations = 10
    spec.option.ls_iterations = 20
    ground_texture = spec.add_texture(name="basketball_ground")
    ground_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    ground_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    ground_texture.rgb1 = [.22, .27, .31]
    ground_texture.rgb2 = [.16, .20, .24]
    ground_texture.width = 256
    ground_texture.height = 256
    ground_material = spec.add_material(name="basketball_ground_mat")
    ground_textures = list(ground_material.textures)
    ground_textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "basketball_ground"
    ground_material.textures = ground_textures
    ground_material.texuniform = True
    ground_material.texrepeat = [4, 4]
    spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05], material="basketball_ground_mat",
        friction=[1, 0.005, 0.0001],
    )
    spec.worldbody.add_light(pos=[0, -1, 3], dir=[0, 0, -1])
    texture = spec.add_texture(name="basketball_tex")
    texture.type = mujoco.mjtTexture.mjTEXTURE_2D
    texture.file = str(asset_dir / "basketball.png")
    material = spec.add_material(name="basketball_mat")
    textures = list(material.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "basketball_tex"
    material.textures = textures
    material.specular = 0.15
    spec.add_mesh(name="basketball_mesh", file=str(asset_dir / "basketball.obj"))
    ball = spec.worldbody.add_body(name="basketball", pos=[0, 0, BALL_RADIUS + 0.001])
    ball.add_freejoint(name="basketball_freejoint")
    ball.add_geom(
        name="ball_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0, 0], mass=BALL_MASS, friction=[1.2, 0.01, 0.001],
        priority=1, condim=4, rgba=[1, 0.45, 0.02, 0], group=3,
    )
    ball.add_geom(
        name="ball_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="basketball_mesh", material="basketball_mat",
        mass=0, contype=0, conaffinity=0, group=2,
    )
    spec.add_key(
        name="STAND", qpos=np.concatenate((
            [0, 0, 2 * BALL_RADIUS + 0.128, 1, 0, 0, 0],
            C.DEFAULT_POSE, [0, 0, BALL_RADIUS + 0.001, 1, 0, 0, 0],
        )), ctrl=C.DEFAULT_POSE,
    )
    model = spec.compile()
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    return model


def curriculum_level(stage: int, seconds: float) -> int:
    if not 0 <= stage < len(HOLD_LEVELS) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("invalid curriculum stage or episode duration")
    if seconds >= 6:
        return min(stage + 1, len(HOLD_LEVELS) - 1)
    if seconds < 1.5:
        return max(stage - 1, 0)
    return stage


class BasketballEnv(MicroduckWalkEnv):
    """A blind 61D actor steps on a sphere; all assistance is explicit in info.

    This is a local actuator/physics adaptation, not a bitwise mjlab port.
    The sealed bounded reward subset is unchanged across curriculum stages.
    """

    def __init__(
        self, *, max_episode_s: float = 10, hold: float = 0,
        curriculum: bool = False, command: tuple[float, float, float] | None = None,
        seed: int | None = None, actuator: str = "bam", obs_noise: bool = False,
        domain_rand: bool = False, pushes: bool = False, random_yaw: bool = True,
        model: mujoco.MjModel | None = None,
    ):
        if not math.isfinite(hold) or not 0 <= hold <= 1:
            raise ValueError("hold must be finite and between zero and one")
        if not math.isfinite(max_episode_s) or max_episode_s <= 0:
            raise ValueError("episode length must be positive and finite")
        if curriculum and hold not in HOLD_LEVELS:
            raise ValueError("curriculum hold must be one of HOLD_LEVELS")
        self.hold = float(hold)
        self.curriculum = curriculum
        self.stage = HOLD_LEVELS.index(hold) if hold in HOLD_LEVELS else 0
        self.fixed_command = self._validate_command(command)
        self.pushes = pushes
        self._episode_finished = False
        super().__init__(
            max_episode_s=max_episode_s, command_resample_s=5,
            obs_noise=obs_noise, domain_rand=domain_rand, action_delay=True,
            random_yaw=random_yaw, seed=seed, actuator_force=actuator,
            model=model if model is not None else basketball_model(actuator),
            terminate_on_fall=False, height_termination=False,
        )
        self.ball_body_id = self.model.body("basketball").id
        self.ball_geom_id = self.model.geom("ball_sphere").id
        self.ball_qpos_adr = int(self.model.joint("basketball_freejoint").qposadr[0])
        self.ball_qvel_adr = int(self.model.joint("basketball_freejoint").dofadr[0])
        self.foot_site_ids = [self.model.site(name).id for name in ("left_foot", "right_foot")]
        self._ball_anchor = np.array([0, 0, BALL_RADIUS + 0.001])
        self._next_push = math.inf

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
        elif self._rng.random() < 0.25:
            self.twist_cmd[:] = 0
        else:
            self.twist_cmd[:] = self._rng.uniform([-.15, -.1, -.5], [.15, .1, .5])
        self.head_cmd[:] = 0
        self.body_cmd[:] = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self.curriculum and self._episode_finished:
            self.stage = curriculum_level(self.stage, self.step_count * CTRL_DT)
            self.hold = HOLD_LEVELS[self.stage]
        self._episode_finished = False
        super().reset(seed=seed, options=options)
        self.data.qpos[:2] = self._rng.uniform(-.01, .01, 2)
        self.data.qpos[2] = 2 * BALL_RADIUS + .128
        yaw = self._rng.uniform(-math.pi, math.pi) if self.random_yaw else 0
        roll, pitch = self._rng.uniform(-math.radians(2), math.radians(2), 2)
        yaw_quat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        pitch_quat = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
        roll_quat = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
        rotation = np.empty(4)
        mujoco.mju_mulQuat(rotation, yaw_quat, pitch_quat)
        mujoco.mju_mulQuat(self.data.qpos[3:7], rotation, roll_quat)
        self.data.qpos[self.joint_qpos_adr] = C.DEFAULT_POSE + self._rng.uniform(-.05, .05, 14)
        self.data.qpos[self.ball_qpos_adr:self.ball_qpos_adr + 7] = [0, 0, BALL_RADIUS + .001, 1, 0, 0, 0]
        self.data.qvel[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.ctrl[:] = self.data.qpos[self.joint_qpos_adr]
        mujoco.mj_forward(self.model, self.data)
        if self.bam is not None:
            self.bam.reset()
        self.prev_joint_vel[:] = 0
        self._next_push = self._rng.uniform(1.5, 3) if self.pushes else math.inf
        return self._get_obs(), self.metrics()

    def _apply_assistance(self) -> None:
        self.data.xfrc_applied[:] = 0
        if self.hold == 0:
            return
        ball_pos = self.data.xpos[self.ball_body_id]
        ball_vel = self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr + 6]
        ball_force = -400 * (ball_pos - self._ball_anchor) - 20 * ball_vel[:3]
        ball_force[2] = -20 * ball_vel[2]
        self.data.xfrc_applied[self.ball_body_id, :3] = self.hold * ball_force
        ball_rotation = self.data.xmat[self.ball_body_id].reshape(3, 3)
        self.data.xfrc_applied[self.ball_body_id, 3:] = -self.hold * .2 * (ball_rotation @ ball_vel[3:])
        rotation = self.data.xmat[self.trunk_body_id].reshape(3, 3)
        up = rotation[:, 2]
        angular_world = rotation @ self.data.qvel[3:6]
        duck_force = -40 * (self.data.xpos[self.trunk_body_id] - ball_pos) - 4 * self.data.qvel[:3]
        duck_force[2] = 0
        self.data.xfrc_applied[self.trunk_body_id, :3] = self.hold * duck_force
        self.data.xfrc_applied[self.trunk_body_id, 3:] = self.hold * (
            3 * np.cross(up, [0, 0, 1]) - .08 * angular_world
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("actions must be finite with shape (14,)")
        if self._episode_finished:
            raise RuntimeError("reset is required after a terminal episode")
        self._apply_assistance()
        if self.step_count * CTRL_DT >= self._next_push:
            self.data.qvel[:2] = self._rng.uniform(-.09, .09, 2)
            self._next_push += self._rng.uniform(1.5, 3)
        observation, reward, terminated, truncated, info = super().step(action)
        metrics = self.metrics()
        reasons = []
        if metrics["root_above_ball_m"] < BALL_RADIUS + .05:
            reasons.append("height")
        if metrics["root_ball_offset_m"] > .16:
            reasons.append("offset")
        if metrics["tilt_deg"] > 55:
            reasons.append("tilt")
        if metrics["robot_floor_contact"]:
            reasons.append("robot_floor_contact")
        if metrics["body_ball_contact"]:
            reasons.append("body_ball_contact")
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            reasons.append("nonfinite_physics")
        terminated = terminated or bool(reasons)
        self._episode_finished = terminated or truncated
        info.update(metrics)
        info["termination_reasons"] = reasons
        if self._episode_finished:
            info["episode_rewards"] = dict(self.reward_sums)
        return observation, reward, terminated, truncated, info

    def _compute_reward(self):
        relative = self.data.xpos[self.trunk_body_id] - self.data.xpos[self.ball_body_id]
        gravity = self._projected_gravity()
        feet = self.data.site_xpos[self.foot_site_ids]
        gaps = np.linalg.norm(feet - self.data.xpos[self.ball_body_id], axis=1) - (BALL_RADIUS + .012)
        ball_velocity = self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr + 2]
        terms = {
            "linear_tracking": math.exp(-float(np.sum((self.body_lin_vel()[:2] - self.twist_cmd[:2]) ** 2)) / .1),
            "yaw_tracking": math.exp(-float((self._gyro[2] - self.twist_cmd[2]) ** 2) / .5),
            "upright": math.exp(-float(np.sum(gravity[:2] ** 2)) / .09),
            "centered": math.exp(-float(np.sum(relative[:2] ** 2)) / .0016),
            "feet_on_ball": float(np.mean(np.exp(-gaps ** 2 / .0004))),
            "height": math.exp(-float((relative[2] - BALL_RADIUS - .125) ** 2) / .0009),
            "ball_speed_penalty": -float(min(np.sum(ball_velocity ** 2), 100)),
            "angular_velocity_penalty": -float(min(np.sum(self._gyro[:2] ** 2), 100)),
            "action_rate_penalty": -float(min(np.sum((self.last_action - self.prev_action) ** 2), 100)),
        }
        weighted = {key: value * REWARD_WEIGHTS[key] * CTRL_DT for key, value in terms.items()}
        reward = sum(weighted.values())
        if not math.isfinite(reward):
            raise FloatingPointError("nonfinite basketball reward")
        return reward, weighted

    def metrics(self) -> dict:
        ball_pos = self.data.xpos[self.ball_body_id]
        relative = self.data.xpos[self.trunk_body_id] - ball_pos
        foot_geoms = set(self.foot_geoms.values())
        touching = set()
        floor_contact = False
        body_ball_contact = False
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.ball_geom_id in pair:
                other = next(iter(pair - {self.ball_geom_id}), self.ball_geom_id)
                if other in foot_geoms:
                    touching.add(other)
                elif self.model.geom_bodyid[other] not in (0, self.ball_body_id):
                    body_ball_contact = True
            if self.floor_geom in pair:
                other = next(iter(pair - {self.floor_geom}), self.floor_geom)
                if self.model.geom_bodyid[other] not in (0, self.ball_body_id):
                    floor_contact = True
        ball_velocity = self.data.qvel[self.ball_qvel_adr:self.ball_qvel_adr + 6]
        return {
            "tilt_deg": math.degrees(math.acos(float(np.clip(-self._projected_gravity()[2], -1, 1)))),
            "root_ball_offset_m": float(np.linalg.norm(relative[:2])),
            "root_above_ball_m": float(relative[2]),
            "ball_speed_mps": float(np.linalg.norm(ball_velocity[:2])),
            "ball_position": ball_pos.tolist(),
            "ball_angular_speed_rad_s": float(np.linalg.norm(ball_velocity[3:])),
            "body_forward_mps": float(self.body_lin_vel()[0]),
            "body_lateral_mps": float(self.body_lin_vel()[1]),
            "body_yaw_rate_rad_s": float(self._gyro[2]),
            "foot_ball_contacts": len(touching),
            "body_ball_contact": body_ball_contact,
            "robot_floor_contact": floor_contact,
            "hold": self.hold,
            "elapsed_s": self.step_count * CTRL_DT,
        }
