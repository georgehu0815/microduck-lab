"""MicroDuck recipe registry and macOS-native MuJoCo environments."""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from rlx.environments.microduck import MicroDuckVecEnv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VENDORED_ROBOT_DIR = REPOSITORY_ROOT / "rlx" / "mjlab_microduck" / "robot" / "microduck"
CLIP_DIRECTORY = REPOSITORY_ROOT / "assets" / "clips"

SWING_STRING_LENGTH = 0.38
SWING_STRING_STIFFNESS = 2000.0
SWING_STRING_LIMIT = 0.395
SWING_HANG_LENGTH = 0.3822
SWING_ANCHOR_HEIGHT = 0.75
SWING_ATTACHMENT_Y = 0.088
SWING_ATTACHMENT_Z = 0.085
SWING_BOTTOM_TRUNK_Z = SWING_ANCHOR_HEIGHT - SWING_HANG_LENGTH - SWING_ATTACHMENT_Z

MIN_STILT_HEIGHT_CM = 0.8
MAX_STILT_HEIGHT_CM = 300.0
DEFAULT_STILT_MASS_BASE_KG = 0.012
DEFAULT_STILT_MASS_PER_CM_KG = 0.001


@dataclass(frozen=True)
class MicroDuckRecipe:
    name: str
    behavior_id: str | None
    default_episode_s: float
    default_render_distance: float


RECIPES = {
    "dance": MicroDuckRecipe("dance", "imitate", 4.0, 0.70),
    "swing": MicroDuckRecipe("swing", None, 24.0, 1.15),
    "running": MicroDuckRecipe("running", "run", 12.0, 0.70),
    "stilts": MicroDuckRecipe("stilts", None, 10.0, 0.85),
    "backflip": MicroDuckRecipe("backflip", "backflip", 12.0, 0.85),
    "basketball": MicroDuckRecipe("basketball", None, 60.0, 1.1),
    "bridge": MicroDuckRecipe("bridge", None, 20.0, 2.6),
}

STILT_REWARD_WEIGHTS = {
    "track_lin_vel": 2.5,
    "track_ang_vel": 1.0,
    "upright": 3.0,
    "pose": 0.0,
    "head_pose": 0.25,
    "feet_air_time": 2.0,
    "ang_vel_xy_penalty": 0.05,
}

SWING_REWARD_WEIGHTS = {
    "swing_peak_progress": 224.0,
    "swing_height": 8.0,
    "swing_late_height": 24.0,
    "swing_energy": 0.5,
    "swing_lateral_penalty": 3.0,
    "swing_lateral_barrier_penalty": 8.0,
    "swing_lateral_velocity_penalty": 3.0,
    "swing_out_of_plane_penalty": 1.0,
    "swing_alignment_penalty": 18.0,
    "swing_alignment_barrier_penalty": 4.0,
    "string_slack_penalty": 8.0,
    "string_extension_penalty": 12.0,
    "invalid_episode_penalty": 10.0,
    "action_rate_penalty": 0.03,
    "joint_torque_penalty": 0.001,
    "joint_limit_penalty": 1.0,
}

RECIPE_REWARD_KEYS = {
    "basketball": set(),
    "bridge": set(),
    "backflip": {
        "landing_upright", "landing_pose", "landing_settle",
        "landing_joint_speed_penalty", "landing_action_rate_penalty", "landing_action_size_penalty",
        "landing_teacher_action_penalty",
    },
    "dance": {
        "pose_match",
        "rotation_match",
        "stick_it",
        "on_feet",
        "no_slip",
        "no_spin",
        "travel",
        "gentle_head",
        "soft_landings",
        "no_limit_parking",
        "save_energy",
    },
    "running": {
        "keep_pace",
        "track_turn",
        "yaw_tracking",
        "air_time",
        "flight",
        "stay_upright",
        "pose",
        "head_up",
        "foot_clearance",
        "plant_the_foot",
        "smooth_moves",
        "no_limit_parking",
        "calm_roll",
    },
    "stilts": {*STILT_REWARD_WEIGHTS, "yaw_tracking"},
    "swing": set(SWING_REWARD_WEIGHTS),
}


def get_recipe(name: str) -> MicroDuckRecipe:
    try:
        return RECIPES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown MicroDuck recipe {name!r}; choose from {sorted(RECIPES)}"
        ) from exc


def validate_reward_weights(
    recipe: str,
    weights: dict[str, float] | None,
) -> dict[str, float]:
    get_recipe(recipe)
    normalized: dict[str, float] = {}
    for key, value in (weights or {}).items():
        if key not in RECIPE_REWARD_KEYS[recipe]:
            raise ValueError(f"unknown {recipe} reward weight {key!r}")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"reward weight {key!r} must be finite and non-negative")
        normalized[key] = weight
    return normalized


def _validate_dance_pose_sigma(
    recipe: str,
    dance_pose_sigma: float | None,
) -> float | None:
    if dance_pose_sigma is None:
        return None
    if recipe != "dance":
        raise ValueError("dance_pose_sigma is only valid for the dance recipe")
    sigma = float(dance_pose_sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("dance_pose_sigma must be finite and positive")
    return sigma


def _per_joint_dance_pose_match(env: gym.Env, *, sigma: float) -> float:
    target, _ = env.clip.at(env.step_count)
    scaled_error = (env._joint_qpos() - target) / sigma
    return float(np.mean(np.exp(-np.square(scaled_error))))


def _running_flight(env: gym.Env) -> float:
    """Dense aerial shaping, independent of the per-foot air-time threshold.

    Upright 0.8 -> 1.0 and command-directed speed 0 -> 0.4 m/s ramp
    credit to one. No credit for grounded, stationary or opposing motion;
    backward/sideways commands use their own body-frame direction.
    """
    contacts = env._foot_contacts()
    if contacts["left"] or contacts["right"]:
        return 0.0
    command = env.twist_cmd[:2]
    command_norm = math.hypot(*command)
    if not math.isfinite(command_norm) or command_norm == 0.0:
        return 0.0
    upright = float(-env._projected_gravity()[2])
    speed = float(np.dot(env.body_lin_vel()[:2], command / command_norm))
    if not math.isfinite(upright) or not math.isfinite(speed):
        return 0.0
    return float(np.clip((upright - 0.8) / 0.2, 0.0, 1.0)
                 * np.clip(speed / 0.4, 0.0, 1.0))


def _locomotion_yaw_tracking(env: gym.Env) -> float:
    yaw_rate = float(env._gyro[2])
    yaw_command = float(env.twist_cmd[2])
    if not math.isfinite(yaw_rate) or not math.isfinite(yaw_command):
        return 0.0
    scaled_error = (yaw_rate - yaw_command) / 0.25
    return float(math.exp(-(scaled_error * scaled_error)))


def _replace_dance_pose_match(env: gym.Env, sigma: float) -> None:
    replacement = partial(_per_joint_dance_pose_match, sigma=sigma)
    rows = []
    replaced = False
    for key, output_key, weight, fn in env._term_rows:
        if key == "pose_match":
            fn = replacement
            replaced = True
        rows.append((key, output_key, weight, fn))
    if not replaced:
        raise RuntimeError("dance recipe is missing its pose_match reward term")
    env._term_rows = tuple(rows)


def _resolve_dance_clip(
    dance_clip: str | Path | None,
) -> tuple[str, Path]:
    if dance_clip is None:
        return "dance-120bpm", CLIP_DIRECTORY

    path = Path(dance_clip).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"dance clip must be a JSON file, got {path}")
    if path.suffix != ".json":
        raise ValueError(f"dance clip must end in .json, got {path.name!r}")

    clip_name = path.name[: -len(path.suffix)]
    motion = importlib.import_module("microduck_local.motion")
    motion.load_clip(clip_name, path.parent)
    return clip_name, path.parent


def default_stilt_mass_kg(height_cm: float) -> float:
    return DEFAULT_STILT_MASS_BASE_KG + height_cm * DEFAULT_STILT_MASS_PER_CM_KG


def validate_stilt_options(
    height_cm: float,
    blend: float,
    mass_kg: float | None,
) -> tuple[float, float, float]:
    height_cm = float(height_cm)
    blend = float(blend)
    if not MIN_STILT_HEIGHT_CM <= height_cm <= MAX_STILT_HEIGHT_CM:
        raise ValueError(
            f"stilt height must be in [{MIN_STILT_HEIGHT_CM:g}, "
            f"{MAX_STILT_HEIGHT_CM:g}] cm"
        )
    if not 0.0 <= blend <= 1.0:
        raise ValueError("stilt blend must be between 0 and 1")
    resolved_mass = (
        default_stilt_mass_kg(height_cm) if mass_kg is None else float(mass_kg)
    )
    if resolved_mass <= 0.0:
        raise ValueError("stilt mass must be positive")
    return height_cm, blend, resolved_mass


def validate_locomotion_forward_command(
    recipe: str,
    command_m_s: float | None,
) -> float | None:
    if command_m_s is None:
        return None
    if recipe not in {"running", "stilts"}:
        raise ValueError(
            "locomotion_forward_command is only valid for running and stilts"
        )
    if isinstance(command_m_s, bool):
        raise ValueError(
            "locomotion_forward_command must be finite, greater than 0, "
            "and at most 1.5"
        )
    command = float(command_m_s)
    if not math.isfinite(command) or not 0.0 < command <= 1.5:
        raise ValueError(
            "locomotion_forward_command must be finite, greater than 0, "
            "and at most 1.5"
        )
    return command


def _pin_locomotion_forward_command(env: gym.Env, command_m_s: float) -> None:
    sample_commands = env._sample_commands

    def sample_pinned_command() -> None:
        sample_commands()
        env.twist_cmd[:] = (command_m_s, 0.0, 0.0)

    env._sample_commands = sample_pinned_command


def _rounded_rectangle_ring(
    width: float,
    length: float,
    radius: float,
    samples_per_corner: int = 8,
) -> list[tuple[float, float]]:
    radius = min(radius, width / 2.0, length / 2.0)
    corners = (
        (width / 2.0 - radius, length / 2.0 - radius, 0.0),
        (-width / 2.0 + radius, length / 2.0 - radius, math.pi / 2.0),
        (-width / 2.0 + radius, -length / 2.0 + radius, math.pi),
        (width / 2.0 - radius, -length / 2.0 + radius, 3.0 * math.pi / 2.0),
    )
    ring: list[tuple[float, float]] = []
    for center_x, center_y, start_angle in corners:
        for sample in range(samples_per_corner):
            angle = start_angle + math.pi * sample / (2.0 * samples_per_corner)
            ring.append(
                (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                )
            )
    return ring


def _stilt_mesh_data(
    height_cm: float,
    blend: float,
) -> tuple[list[float], list[int]]:
    def lerp(start: float, end: float) -> float:
        return start + blend * (end - start)

    mm = 0.001
    bottom = _rounded_rectangle_ring(
        lerp(22.0, 12.0) * mm,
        lerp(32.0, 12.0) * mm,
        lerp(3.5, 6.0) * mm,
    )
    top = _rounded_rectangle_ring(
        lerp(23.0, 22.0) * mm,
        lerp(41.0, 22.0) * mm,
        lerp(4.5, 11.0) * mm,
    )
    count = len(bottom)
    height = height_cm * 0.01
    vertices = [(x, y, -height) for x, y in bottom]
    vertices.extend((x, y, 0.0) for x, y in top)
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, 0.0))
    faces: list[tuple[int, int, int]] = []
    for current in range(count):
        following = (current + 1) % count
        faces.extend(
            (
                (current, following, count + following),
                (current, count + following, count + current),
                (bottom_center, following, current),
                (top_center, count + current, count + following),
            )
        )
    return [value for vertex in vertices for value in vertex], [
        value for face in faces for value in face
    ]


def _build_stilt_model(
    height_cm: float,
    blend: float,
    mass_kg: float,
):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(VENDORED_ROBOT_DIR / "scene_walk.xml"))
    vertices, faces = _stilt_mesh_data(height_cm, blend)
    mesh_name = "rlx_stilt_cartridge"
    spec.add_mesh(name=mesh_name, uservert=vertices, userface=faces)
    for side in ("left", "right"):
        old_collision = spec.geom(f"{side}_foot_collision")
        old_collision.name = f"{side}_original_sole_disabled"
        old_collision.contype = 0
        old_collision.conaffinity = 0

        old_site = spec.site(f"{side}_foot")
        site_pos = tuple(float(value) for value in old_site.pos)
        site_quat = tuple(float(value) for value in old_site.quat)
        old_site.name = f"{side}_original_foot_site"

        stilt = spec.body(f"ankle_{side}").add_body(
            name=f"stilt_{side}",
            pos=site_pos,
            quat=site_quat,
        )
        stilt.add_geom(
            name=f"{side}_foot_collision",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            mass=mass_kg,
            group=3,
            priority=1,
            friction=(1.0, 0.005, 0.0001),
        )
        stilt.add_geom(
            name=f"{side}_stilt_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh_name,
            mass=0.0,
            contype=0,
            conaffinity=0,
            group=2,
            rgba=(0.45, 0.34, 0.85, 1.0),
        )
        stilt.add_site(
            name=f"{side}_foot",
            pos=(0.0, 0.0, -height_cm * 0.01),
        )

    stand = spec.key("STAND")
    stand.qpos[2] = float(stand.qpos[2]) + height_cm * 0.01
    return spec.compile()


def _build_swing_model():
    import mujoco

    spec = mujoco.MjSpec.from_file(str(VENDORED_ROBOT_DIR / "scene.xml"))
    for name in (
        "swing_seat_retained",
        "swing_retention_strap",
        "swing_retention_buckle",
        "swing_retention_bumper_0",
        "swing_retention_bumper_1",
        "swing_retention_bumper_2",
    ):
        spec.add_mesh(
            name=name,
            file=str(VENDORED_ROBOT_DIR / "assets" / f"{name}.stl"),
        )
    spec.add_material(name="swing_seat_material", rgba=(0.055, 0.062, 0.068, 1.0))
    spec.add_material(name="swing_strap_material", rgba=(0.075, 0.080, 0.085, 1.0))
    spec.add_material(name="swing_bumper_material", rgba=(0.12, 0.13, 0.14, 1.0))
    spec.add_material(name="swing_buckle_material", rgba=(0.24, 0.25, 0.26, 1.0))
    spec.add_material(name="swing_frame_material", rgba=(0.28, 0.18, 0.11, 1.0))

    frame_segments = {
        "swing_frame_front_left": ((0.34, 0.34, 0.0), (0.0, 0.25, 0.755)),
        "swing_frame_back_left": ((-0.34, 0.34, 0.0), (0.0, 0.25, 0.755)),
        "swing_frame_front_right": ((0.34, -0.34, 0.0), (0.0, -0.25, 0.755)),
        "swing_frame_back_right": ((-0.34, -0.34, 0.0), (0.0, -0.25, 0.755)),
        "swing_frame_crossbar": ((0.0, -0.27, 0.755), (0.0, 0.27, 0.755)),
    }
    for name, endpoints in frame_segments.items():
        spec.worldbody.add_geom(
            name=name,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=endpoints[0] + endpoints[1],
            size=(0.009 if name == "swing_frame_crossbar" else 0.007,),
            material="swing_frame_material",
            contype=0,
            conaffinity=0,
            group=2,
        )

    trunk = spec.body("trunk_base")
    trunk.pos = (0.0, 0.0, SWING_BOTTOM_TRUNK_Z)
    payload = trunk.add_body(name="swing_seat_payload")
    payload.add_geom(
        name="swing_seat_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_seat_retained",
        material="swing_seat_material",
        contype=0,
        conaffinity=0,
        mass=0.150,
        group=2,
    )
    payload.add_geom(
        name="swing_retention_strap_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_retention_strap",
        material="swing_strap_material",
        contype=0,
        conaffinity=0,
        mass=0.004,
        group=2,
    )
    payload.add_geom(
        name="swing_retention_buckle_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="swing_retention_buckle",
        material="swing_buckle_material",
        contype=0,
        conaffinity=0,
        mass=0.002,
        group=2,
    )
    for index in range(3):
        payload.add_geom(
            name=f"swing_retention_bumper_{index}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"swing_retention_bumper_{index}",
            material="swing_bumper_material",
            contype=0,
            conaffinity=0,
            mass=0.002 / 3.0,
            group=2,
        )

    for side, y in (("left", SWING_ATTACHMENT_Y), ("right", -SWING_ATTACHMENT_Y)):
        spec.worldbody.add_site(
            name=f"swing_anchor_{side}",
            pos=(0.0, y, SWING_ANCHOR_HEIGHT),
            size=(0.003,),
            rgba=(0.55, 0.38, 0.22, 1.0),
        )
        trunk.add_site(
            name=f"swing_attach_{side}",
            pos=(0.0, y, SWING_ATTACHMENT_Z),
            size=(0.003,),
            rgba=(0.55, 0.38, 0.22, 1.0),
        )
        string = spec.add_tendon(
            name=f"swing_string_{side}",
            stiffness=SWING_STRING_STIFFNESS,
            springlength=(0.0, SWING_STRING_LENGTH),
            limited=True,
            range=(0.0, SWING_STRING_LIMIT),
            width=0.0016,
            rgba=(0.92, 0.87, 0.70, 1.0),
            solref_limit=(0.02, 1.0),
            solimp_limit=(0.90, 0.95, 0.001, 0.5, 2.0),
        )
        string.wrap_site(f"swing_anchor_{side}")
        string.wrap_site(f"swing_attach_{side}")
    return spec.compile()


_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _recipe_model(key: tuple[Any, ...], builder: Callable[[], Any], cache: bool):
    if not cache:
        return builder()
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = builder()
        _MODEL_CACHE[key] = model
    return model


def _locomotion_recipe_metrics(env: gym.Env) -> dict[str, float]:
    gravity = env._projected_gravity()
    forward_speed, lateral_speed, _ = env.heading_lin_vel()
    heading = env._trunk_xmat.reshape(3, 3)[:, 0].astype(np.float64)
    heading[2] = 0.0
    heading_norm = float(np.linalg.norm(heading))
    if heading_norm > 1e-9:
        heading /= heading_norm
    else:
        heading[:] = (1.0, 0.0, 0.0)
    contacts = env._foot_contacts()
    return {
        "forward_speed_m_s": float(forward_speed),
        "command_forward_m_s": float(env.twist_cmd[0]),
        "command_lateral_m_s": float(env.twist_cmd[1]),
        "command_yaw_rad_s": float(env.twist_cmd[2]),
        "heading_forward_speed_m_s": float(forward_speed),
        "heading_lateral_speed_m_s": float(lateral_speed),
        "world_x_m": float(env._trunk_xpos[0]),
        "world_y_m": float(env._trunk_xpos[1]),
        "heading_forward_x": float(heading[0]),
        "heading_forward_y": float(heading[1]),
        "upright": float(max(0.0, -gravity[2])),
        "left_foot_contact": float(contacts["left"]),
        "right_foot_contact": float(contacts["right"]),
    }


@lru_cache(maxsize=1)
def _native_environment_classes():
    import mujoco

    walk_env = importlib.import_module("microduck_local.walk_env")
    contract = importlib.import_module("microduck_local.contract")
    default_pose = contract.DEFAULT_POSE
    ctrl_dt = float(contract.CTRL_DT)

    class StiltEnv(walk_env.MicroduckWalkEnv):
        def __init__(
            self,
            *,
            height_cm: float,
            blend: float,
            mass_kg: float,
            cache_model: bool,
            weight_overrides: dict[str, float] | None = None,
            **kwargs: Any,
        ) -> None:
            self.stilt_height_cm = height_cm
            self.stilt_blend = blend
            self.stilt_mass_kg = mass_kg
            self.weight_overrides = dict(weight_overrides or {})
            model = _recipe_model(
                ("stilts", height_cm, blend, mass_kg),
                lambda: _build_stilt_model(height_cm, blend, mass_kg),
                cache_model,
            )
            kwargs.setdefault("height_termination", False)
            kwargs.setdefault("zero_command_prob", 0.25)
            kwargs.setdefault("turn_in_place_prob", 0.05)
            kwargs.setdefault("forward_command_prob", 0.45)
            super().__init__(model=model, **kwargs)

        def _sample_commands(self) -> None:
            r = self._rng
            if r.uniform() < self.zero_command_prob:
                self.twist_cmd[:] = 0.0
            else:
                self.twist_cmd[:] = (
                    r.uniform(-0.12, 0.25),
                    r.uniform(-0.06, 0.06),
                    r.uniform(-0.35, 0.35),
                )
            self.head_cmd[:] = [
                r.uniform(lo, hi) for lo, hi in contract.HEAD_CMD_RANGES
            ]
            self.body_cmd[:] = [
                r.uniform(lo, hi) for lo, hi in contract.BODY_CMD_RANGES
            ]

        def _compute_reward(self):
            reward, terms = super()._compute_reward()
            base_weights = {
                "track_lin_vel": self.W_TRACK_LIN,
                "track_ang_vel": self.W_TRACK_ANG,
                "upright": self.W_UPRIGHT,
                "pose": self.W_POSE,
                "head_pose": self.W_HEAD_POSE,
                "feet_air_time": self.W_AIR_TIME,
                "ang_vel_xy_penalty": self.W_ANG_VEL_XY,
            }
            for key, default_weight in STILT_REWARD_WEIGHTS.items():
                weight = self.weight_overrides.get(key, default_weight)
                terms[key] *= weight / base_weights[key]
            yaw_tracking_weight = self.weight_overrides.get("yaw_tracking", 0.0)
            if yaw_tracking_weight > 0.0:
                terms["yaw_tracking"] = (
                    yaw_tracking_weight * _locomotion_yaw_tracking(self)
                )
            terms["action_rate_penalty"] *= 0.2
            return float(sum(terms.values())), terms

        def recipe_metrics(self) -> dict[str, float]:
            return {
                **_locomotion_recipe_metrics(self),
                "stilt_height_cm": self.stilt_height_cm,
                "stilt_blend": self.stilt_blend,
                "stilt_mass_kg": self.stilt_mass_kg,
            }

    class SwingEnv(walk_env.MicroduckWalkEnv):
        _SEATED_POSE = np.array(
            [
                0.0,
                0.0,
                -0.4079,
                1.35,
                0.0,
                0.3491,
                0.3491,
                0.0,
                0.0,
                0.0,
                0.0,
                0.4079,
                -1.35,
                0.0,
            ],
            dtype=np.float32,
        )

        def __init__(
            self,
            *,
            cache_model: bool,
            weight_overrides: dict[str, float] | None = None,
            initial_angle_deg: float = 0.0,
            initial_rate_rad_s: float = 0.0,
            planar_actions: bool = False,
            **kwargs: Any,
        ) -> None:
            self.weight_overrides = dict(weight_overrides or {})
            self.initial_angle_deg = float(initial_angle_deg)
            self.initial_rate_rad_s = float(initial_rate_rad_s)
            self.planar_actions = bool(planar_actions)
            model = _recipe_model(("swing",), _build_swing_model, cache_model)
            kwargs.update(
                model=model,
                terminate_on_fall=False,
                height_termination=False,
                random_yaw=False,
                command_resample_s=1_000_000.0,
            )
            super().__init__(**kwargs)
            self._policy_last_action = np.zeros(14, dtype=np.float32)
            self._policy_prev_action = np.zeros(14, dtype=np.float32)
            self._swing_site_ids = np.array(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                    for name in (
                        "swing_anchor_left",
                        "swing_anchor_right",
                        "swing_attach_left",
                        "swing_attach_right",
                    )
                ]
            )
            self._swing_tendon_ids = np.array(
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, name)
                    for name in ("swing_string_left", "swing_string_right")
                ]
            )

        def _reset_episode_state(self) -> None:
            super()._reset_episode_state()
            self._policy_last_action = np.zeros(14, dtype=np.float32)
            self._policy_prev_action = np.zeros(14, dtype=np.float32)
            self._swing_peak_positive = 0.0
            self._swing_peak_negative = 0.0
            self._swing_invalid_steps = 0
            self._swing_invalid = False
            self._swing_assisted_energy = 0.0

        def _sample_commands(self) -> None:
            self.twist_cmd[:] = 0.0
            self.head_cmd[:] = 0.0
            self.body_cmd[:] = 0.0

        def reset(self, **kwargs: Any):
            super().reset(**kwargs)
            angle = math.radians(
                self._rng.uniform(
                    -self.initial_angle_deg,
                    self.initial_angle_deg,
                )
            )
            rate = self._rng.uniform(
                -self.initial_rate_rad_s,
                self.initial_rate_rad_s,
            )
            self.data.qpos[:7] = (
                SWING_HANG_LENGTH * math.sin(angle),
                0.0,
                SWING_ANCHOR_HEIGHT
                - SWING_HANG_LENGTH * math.cos(angle)
                - SWING_ATTACHMENT_Z,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            self.data.qpos[self.joint_qpos_adr] = self._SEATED_POSE
            self.data.qvel[:] = 0.0
            self.data.qvel[0] = SWING_HANG_LENGTH * math.cos(angle) * rate
            self.data.qvel[2] = SWING_HANG_LENGTH * math.sin(angle) * rate
            self.data.ctrl[:] = self._SEATED_POSE
            # The actuator delay stores transformed actions, not raw policy history.
            self._delayed_action = (self._SEATED_POSE - default_pose).clip(-4.0, 4.0)
            mujoco.mj_forward(self.model, self.data)
            swing_state = self._swing_state()
            reset_energy = min(
                2.0,
                1.0
                - math.cos(float(swing_state["angle"]))
                + 0.5
                * (SWING_STRING_LENGTH / 9.81)
                * float(swing_state["angle_rate"]) ** 2,
            )
            assisted_amplitude = math.acos(1.0 - reset_energy)
            self._swing_peak_positive = assisted_amplitude
            self._swing_peak_negative = assisted_amplitude
            self._swing_assisted_energy = reset_energy
            if self.bam is not None:
                self.bam.reset(self.data.qpos[self.joint_qpos_adr])
            self.prev_joint_vel = self._joint_vel().copy()
            return self._get_obs(), {}

        def step(self, action: np.ndarray):
            bounded = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
            if self.planar_actions:
                bounded = bounded.copy()
                bounded[[0, 1, 7, 8, 9, 10]] = 0.0
                for left, right in ((2, 11), (3, 12), (4, 13)):
                    sagittal = 0.5 * (bounded[left] - bounded[right])
                    bounded[left] = sagittal
                    bounded[right] = -sagittal
            self._policy_prev_action = self._policy_last_action
            self._policy_last_action = bounded.copy()
            transformed = self._SEATED_POSE - default_pose + 0.7 * bounded
            return super().step(transformed)

        def _get_obs(self) -> np.ndarray:
            rotation = self._trunk_xmat.reshape(3, 3)
            body_y_axis = rotation[:, 1]
            self.twist_cmd[:] = (0.0, body_y_axis[0], body_y_axis[2])
            self.head_cmd[:] = 0.0
            self.body_cmd[:] = 0.0
            observation = super()._get_obs()
            observation[34:48] = self._policy_last_action
            return observation

        def _site_velocity(self, site_id: int) -> np.ndarray:
            velocity = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_SITE,
                int(site_id),
                velocity,
                0,
            )
            return velocity[3:].copy()

        def _swing_state(self) -> dict[str, Any]:
            sites = self.data.site_xpos[self._swing_site_ids]
            anchor = 0.5 * (sites[0] + sites[1])
            attach = 0.5 * (sites[2] + sites[3])
            attach_velocity = 0.5 * (
                self._site_velocity(self._swing_site_ids[2])
                + self._site_velocity(self._swing_site_ids[3])
            )
            rope = attach - anchor
            x, y, z = rope
            radius_sq = max(float(x * x + z * z), 1e-8)
            angle = math.atan2(float(x), float(-z))
            angle_rate = float(
                (-z * attach_velocity[0] + x * attach_velocity[2]) / radius_sq
            )
            lengths = self.data.ten_length[self._swing_tendon_ids].copy()
            slack = float(np.maximum(SWING_STRING_LENGTH - lengths - 5e-4, 0.0).max())
            imbalance = float(abs(lengths[0] - lengths[1]))
            anchor_span = sites[1] - sites[0]
            attach_span = sites[3] - sites[2]
            denom = max(
                float(np.linalg.norm(anchor_span) * np.linalg.norm(attach_span)),
                1e-8,
            )
            alignment = 1.0 - float(
                np.clip(np.dot(anchor_span, attach_span) / denom, -1.0, 1.0)
            )
            return {
                "angle": angle,
                "angle_rate": angle_rate,
                "lateral_m": float(y),
                "lateral_velocity_m_s": float(self.data.qvel[1]),
                "lengths": lengths,
                "slack_m": slack,
                "imbalance_m": imbalance,
                "alignment": alignment,
            }

        @staticmethod
        def _validity(state: dict[str, Any]) -> float:
            extension = max(float(np.max(state["lengths"])) - SWING_STRING_LENGTH, 0.0)
            cost = (
                (max(abs(state["lateral_m"]) - 0.012, 0.0) / 0.004) ** 2
                + (max(state["alignment"] - 0.04, 0.0) / 0.005) ** 2
                + (max(state["slack_m"] - 0.0095, 0.0) / 0.003) ** 2
                + (max(extension - 0.012, 0.0) / 0.001) ** 2
            )
            return math.exp(-min(cost, 20.0))

        def _compute_reward(self):
            self._lifetime_steps += 1
            state = self._swing_state()
            angle = state["angle"]
            rate = state["angle_rate"]
            validity = self._validity(state)

            positive = max(0.0, min(math.pi, angle))
            negative = max(0.0, min(math.pi, -angle))
            paid_positive = min(
                max(positive - self._swing_peak_positive, 0.0),
                8.0 * ctrl_dt,
            )
            paid_negative = min(
                max(negative - self._swing_peak_negative, 0.0),
                8.0 * ctrl_dt,
            )
            frontier = (
                ((self._swing_peak_positive + paid_positive) / math.pi) ** 2
                - (self._swing_peak_positive / math.pi) ** 2
                + ((self._swing_peak_negative + paid_negative) / math.pi) ** 2
                - (self._swing_peak_negative / math.pi) ** 2
            ) / (2.0 * ctrl_dt)
            self._swing_peak_positive = max(self._swing_peak_positive, positive)
            self._swing_peak_negative = max(self._swing_peak_negative, negative)

            longest = float(np.max(state["lengths"]))
            cord_invalid = longest > 0.394 or longest < 0.370
            invalid = (
                abs(state["lateral_m"]) > 0.020
                or state["alignment"] > 0.050
                or cord_invalid
            )
            self._swing_invalid_steps = self._swing_invalid_steps + 1 if invalid else 0
            self._swing_invalid = self._swing_invalid or self._swing_invalid_steps >= 2
            if self._swing_invalid:
                validity = 0.0

            height = max(0.0, min(2.0, 1.0 - math.cos(angle)))
            progress = min(1.0, self.step_count / max(self.max_steps, 1))
            energy = max(
                0.0,
                min(
                    2.0,
                    height + 0.5 * (SWING_STRING_LENGTH / 9.81) * rate * rate,
                ),
            )
            height_gain = max(height - self._swing_assisted_energy, 0.0)
            energy_gain = max(energy - self._swing_assisted_energy, 0.0)
            lateral = state["lateral_m"]
            alignment = state["alignment"]
            extension = max(longest - 0.392, 0.0)
            world_ang = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.trunk_body_id,
                world_ang,
                0,
            )
            action_rate = float(
                ((self._policy_last_action - self._policy_prev_action) ** 2).sum()
            )
            joint_ids = self.model.actuator_trnid[:, 0]
            joint_positions = self.data.qpos[self.model.jnt_qposadr[joint_ids]]
            joint_ranges = self.model.jnt_range[joint_ids]
            lower_excess = np.maximum(joint_ranges[:, 0] - joint_positions, 0.0)
            upper_excess = np.maximum(joint_positions - joint_ranges[:, 1], 0.0)
            joint_limit = float(
                (np.square(lower_excess) + np.square(upper_excess)).sum()
            )
            invalid_severity = max(
                (abs(lateral) / 0.020) ** 2,
                alignment / 0.050,
                (extension / 0.002) ** 2,
            )
            weights = {**SWING_REWARD_WEIGHTS, **self.weight_overrides}
            terms = {
                "swing_peak_progress": weights["swing_peak_progress"] * frontier * validity,
                "swing_height": weights["swing_height"] * height_gain * validity,
                "swing_late_height": weights["swing_late_height"] * height_gain * progress**2 * validity,
                "swing_energy": weights["swing_energy"] * energy_gain * validity,
                "swing_lateral_penalty": -weights["swing_lateral_penalty"] * (lateral / 0.03) ** 2,
                "swing_lateral_barrier_penalty": -weights["swing_lateral_barrier_penalty"]
                * (max(abs(lateral) - 0.012, 0.0) / 0.008) ** 2,
                "swing_lateral_velocity_penalty": -weights["swing_lateral_velocity_penalty"]
                * (state["lateral_velocity_m_s"] / 0.1) ** 2,
                "swing_out_of_plane_penalty": -weights["swing_out_of_plane_penalty"]
                * (world_ang[0] ** 2 + world_ang[2] ** 2)
                / (1.5 * 1.5),
                "swing_alignment_penalty": -weights["swing_alignment_penalty"] * alignment,
                "swing_alignment_barrier_penalty": -weights["swing_alignment_barrier_penalty"]
                * (max(alignment - 0.04, 0.0) / 0.01) ** 2,
                "string_slack_penalty": -weights["string_slack_penalty"]
                * ((state["slack_m"] / 0.01) ** 2 + (state["imbalance_m"] / 0.01) ** 2),
                "string_extension_penalty": -weights["string_extension_penalty"] * min((extension / 0.002) ** 2, 16.0),
                "invalid_episode_penalty": -weights["invalid_episode_penalty"]
                * (1.0 + min(invalid_severity, 8.0) if self._swing_invalid else 0.0),
                "action_rate_penalty": -weights["action_rate_penalty"] * action_rate,
                "joint_torque_penalty": -weights["joint_torque_penalty"]
                * float(np.square(self.data.actuator_force).sum()),
                "joint_limit_penalty": -weights["joint_limit_penalty"] * joint_limit,
            }
            return float(sum(terms.values())), terms

        def recipe_metrics(self) -> dict[str, float]:
            """Sample geometry and spring-only tension estimates, excluding constraint forces."""
            state = self._swing_state()
            lengths = state["lengths"]
            spring_tensions = SWING_STRING_STIFFNESS * np.maximum(
                lengths - SWING_STRING_LENGTH, 0.0
            )
            return {
                "swing_angle_deg": math.degrees(state["angle"]),
                "swing_abs_angle_deg": abs(math.degrees(state["angle"])),
                "swing_rate_rad_s": state["angle_rate"],
                "lateral_offset_m": state["lateral_m"],
                "string_slack_m": state["slack_m"],
                "string_imbalance_m": state["imbalance_m"],
                "string_left_m": float(lengths[0]),
                "string_right_m": float(lengths[1]),
                "string_left_tension_n": float(spring_tensions[0]),
                "string_right_tension_n": float(spring_tensions[1]),
                "alignment_penalty": state["alignment"],
                "valid_geometry": float(not self._swing_invalid),
            }

    return StiltEnv, SwingEnv


class _RecipeMetricsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, recipe: str):
        super().__init__(env)
        self.recipe = recipe

    def step(self, action):
        active_command = (
            self.env.twist_cmd.copy()
            if self.recipe in {"running", "stilts"}
            else None
        )
        observation, reward, terminated, truncated, info = self.env.step(action)
        metrics = recipe_metrics(self.env, self.recipe)
        if active_command is not None:
            metrics.update(
                command_forward_m_s=float(active_command[0]),
                command_lateral_m_s=float(active_command[1]),
                command_yaw_rad_s=float(active_command[2]),
            )
        info = dict(info)
        info["recipe_metrics"] = metrics
        if self.recipe == "dance":
            target, _ = self.env.clip.at(self.env.step_count)
            info["dance_state"] = {
                "phase_step": int(self.env.step_count),
                "current_joints": self.env._joint_qpos().astype(float).tolist(),
                "target_joints": target.astype(float).tolist(),
                **metrics,
            }
        return observation, reward, terminated, truncated, info


def recipe_metrics(env: gym.Env, recipe: str) -> dict[str, float]:
    metrics_fn = getattr(env, "recipe_metrics", None)
    if metrics_fn is not None:
        return metrics_fn()
    if recipe == "running":
        return _locomotion_recipe_metrics(env)
    if recipe == "dance":
        target, _ = env.clip.at(env.step_count)
        next_target, _ = env.clip.at(env.step_count + 1)
        current = env._joint_qpos()
        target_velocity = (next_target - target) / (
            env.clip.duration / env.clip.steps
        )
        gravity = env._projected_gravity()
        return {
            "upright": float(max(0.0, -gravity[2])),
            "height_m": float(env._trunk_xpos[2]),
            "pose_rmse_rad": float(np.sqrt(np.mean(np.square(current - target)))),
            "pose_velocity_rmse_rad_s": float(
                np.sqrt(np.mean(np.square(env._joint_vel() - target_velocity)))
            ),
        }
    return {}


def make_single_recipe_env(
    recipe: str,
    *,
    backflip_mode: str = "showcase",
    seed: int = 0,
    actuator: str = "xml",
    domain_rand: bool = True,
    obs_noise: bool = True,
    action_delay: bool = True,
    random_yaw: bool = True,
    max_episode_s: float | None = None,
    weight_overrides: dict[str, float] | None = None,
    dance_clip: str | Path | None = None,
    dance_pose_sigma: float | None = None,
    locomotion_forward_command: float | None = None,
    stilt_height_cm: float = 2.0,
    stilt_blend: float = 0.0,
    stilt_mass_kg: float | None = None,
    swing_initial_angle_deg: float = 0.0,
    swing_initial_rate_rad_s: float = 0.0,
    swing_planar_actions: bool = False,
    cache_model: bool = False,
    bridge_curriculum: bool = False,
) -> gym.Env:
    spec = get_recipe(recipe)
    if dance_clip is not None and recipe != "dance":
        raise ValueError("dance_clip is only valid for the dance recipe")
    dance_pose_sigma = _validate_dance_pose_sigma(recipe, dance_pose_sigma)
    locomotion_forward_command = validate_locomotion_forward_command(
        recipe, locomotion_forward_command
    )
    weight_overrides = validate_reward_weights(recipe, weight_overrides)
    episode_s = spec.default_episode_s if max_episode_s is None else max_episode_s
    common = {
        "seed": seed,
        "actuator_force": actuator,
        "domain_rand": domain_rand,
        "obs_noise": obs_noise,
        "action_delay": action_delay,
        "random_yaw": random_yaw,
        "max_episode_s": episode_s,
    }
    if recipe == "bridge":
        from rlx.environments.bridge import BridgeEnv
        env = BridgeEnv(seed=seed, actuator=actuator, max_episode_s=episode_s,
                        domain_rand=domain_rand, obs_noise=obs_noise,
                        action_delay=action_delay, random_yaw=random_yaw,
                        curriculum=bridge_curriculum)
    elif recipe == "basketball":
        from rlx.environments.basketball import BasketballEnv
        env = BasketballEnv(seed=seed, actuator=actuator, max_episode_s=episode_s,
                            domain_rand=domain_rand, obs_noise=obs_noise,
                            random_yaw=random_yaw, hold=0, command=(0, 0, 0))
    elif recipe == "backflip":
        from rlx.environments.backflip import BackflipEnv
        env = BackflipEnv(**common, backflip_mode=backflip_mode, weight_overrides=weight_overrides)
    elif recipe in {"dance", "running"}:
        behaviors = importlib.import_module("microduck_local.behaviors")
        kwargs = {
            **common,
            "behavior_id": spec.behavior_id,
            "weight_overrides": dict(weight_overrides or {}),
        }
        if recipe == "dance":
            clip_name, clip_directory = _resolve_dance_clip(dance_clip)
            kwargs["clip_name"] = clip_name
            variable = "MICRODUCK_CLIPS_DIR"
            missing = object()
            previous: object | str = os.environ.get(variable, missing)
            try:
                os.environ[variable] = str(clip_directory)
                env = behaviors.BehaviorEnv(**kwargs)
            finally:
                if previous is missing:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = str(previous)
            if dance_pose_sigma is not None:
                _replace_dance_pose_match(env, dance_pose_sigma)
        else:
            env = behaviors.BehaviorEnv(**kwargs)
            if weight_overrides.get("flight", 0.0) > 0.0:
                env._term_rows += (("flight", "flight", 0.0, _running_flight),)
            if weight_overrides.get("yaw_tracking", 0.0) > 0.0:
                env._term_rows += (
                    (
                        "yaw_tracking",
                        "yaw_tracking",
                        0.0,
                        _locomotion_yaw_tracking,
                    ),
                )
    else:
        StiltEnv, SwingEnv = _native_environment_classes()
        if recipe == "stilts":
            height_cm, blend, mass_kg = validate_stilt_options(
                stilt_height_cm, stilt_blend, stilt_mass_kg
            )
            env = StiltEnv(
                **common,
                height_cm=height_cm,
                blend=blend,
                mass_kg=mass_kg,
                cache_model=cache_model,
                weight_overrides=weight_overrides,
            )
        else:
            env = SwingEnv(
                **common,
                cache_model=cache_model,
                weight_overrides=weight_overrides,
                initial_angle_deg=swing_initial_angle_deg,
                initial_rate_rad_s=swing_initial_rate_rad_s,
                planar_actions=swing_planar_actions,
            )
    if locomotion_forward_command is not None:
        _pin_locomotion_forward_command(env, locomotion_forward_command)
    return _RecipeMetricsWrapper(env, recipe)


def make_recipe_env(
    recipe: str,
    *,
    backflip_mode: str = "showcase",
    num_envs: int = 16,
    backend: str | None = None,
    seed: int = 0,
    actuator: str = "xml",
    domain_rand: bool = True,
    obs_noise: bool = True,
    action_delay: bool = True,
    random_yaw: bool = True,
    max_episode_s: float | None = None,
    weight_overrides: dict[str, float] | None = None,
    dance_clip: str | Path | None = None,
    dance_pose_sigma: float | None = None,
    locomotion_forward_command: float | None = None,
    stilt_height_cm: float = 2.0,
    stilt_blend: float = 0.0,
    stilt_mass_kg: float | None = None,
    swing_initial_angle_deg: float = 0.0,
    swing_initial_rate_rad_s: float = 0.0,
    swing_planar_actions: bool = False,
    normalize_observations: bool = False,
    normalize_rewards: bool = False,
    gamma: float = 0.99,
    epsilon: float = 1e-8,
    clip: float = 10.0,
    bridge_curriculum: bool = False,
) -> MicroDuckVecEnv:
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}")
    get_recipe(recipe)
    if dance_clip is not None and recipe != "dance":
        raise ValueError("dance_clip is only valid for the dance recipe")
    dance_pose_sigma = _validate_dance_pose_sigma(recipe, dance_pose_sigma)
    locomotion_forward_command = validate_locomotion_forward_command(
        recipe, locomotion_forward_command
    )
    dance_clip_directory = (
        _resolve_dance_clip(dance_clip)[1] if recipe == "dance" else None
    )
    vec_env = importlib.import_module("microduck_local.vec_env")
    resolved_backend = vec_env.resolve_backend(backend)
    if recipe == "backflip" and resolved_backend == "fork":
        resolved_backend = "dummy"
    cache_model = resolved_backend == "fork"

    def factory(rank: int):
        def initialize():
            return make_single_recipe_env(
                recipe,
                backflip_mode=backflip_mode,
                seed=seed + rank,
                actuator=actuator,
                domain_rand=domain_rand,
                obs_noise=obs_noise,
                action_delay=action_delay,
                random_yaw=random_yaw,
                max_episode_s=max_episode_s,
                weight_overrides=weight_overrides,
                dance_clip=dance_clip,
                dance_pose_sigma=dance_pose_sigma,
                locomotion_forward_command=locomotion_forward_command,
                stilt_height_cm=stilt_height_cm,
                stilt_blend=stilt_blend,
                stilt_mass_kg=stilt_mass_kg,
                swing_initial_angle_deg=swing_initial_angle_deg,
                swing_initial_rate_rad_s=swing_initial_rate_rad_s,
                swing_planar_actions=swing_planar_actions,
                cache_model=cache_model,
                bridge_curriculum=bridge_curriculum,
            )

        return initialize

    variable = "MICRODUCK_CLIPS_DIR"
    missing = object()
    previous: object | str = os.environ.get(variable, missing)
    try:
        if recipe == "dance":
            os.environ[variable] = str(dance_clip_directory)
        envs = vec_env.make_vec_env(
            [factory(rank) for rank in range(num_envs)],
            backend=resolved_backend,
        )
    finally:
        if recipe == "dance":
            if previous is missing:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = str(previous)
    try:
        return MicroDuckVecEnv(
            envs,
            normalize_observations=normalize_observations,
            normalize_rewards=normalize_rewards,
            gamma=gamma,
            epsilon=epsilon,
            clip=clip,
        )
    except Exception:
        envs.close()
        raise


__all__ = [
    "CLIP_DIRECTORY",
    "RECIPES",
    "MicroDuckRecipe",
    "default_stilt_mass_kg",
    "get_recipe",
    "make_recipe_env",
    "make_single_recipe_env",
    "recipe_metrics",
    "validate_locomotion_forward_command",
    "validate_stilt_options",
]
