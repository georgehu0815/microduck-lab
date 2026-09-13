"""Ground tennis-ball return experiment with fail-closed, no-throw acceptance."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from microduck_arm_v1.model import numbers
from microduck_arm_v1c.config import ROOT, load_design, sha256
from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.model import make_tree
from microduck_arm_v1c.whole_body_learning import source_provenance
from .tennis_controller import GroundReturnController


SPEC = ROOT / "hardware/microduck-arm-v1c/experiments/tennis-return.json"
VARIANTS = {"nominal": (.067, .058), "small": (.0654, .056), "large": (.0686, .0594)}
RELEASE_MODES = ("supported", "half_height")


def task_spec():
    return json.loads(SPEC.read_text())


def release_profile(release_mode="supported"):
    if release_mode not in RELEASE_MODES:
        raise ValueError(f"release_mode must be one of {RELEASE_MODES}")
    spec = task_spec()
    profile = copy.deepcopy(spec["release_profiles"][release_mode])
    if release_mode == "supported" and profile["acceptance"] != spec["acceptance"]:
        raise ValueError("supported release profile must match historical acceptance")
    return profile


def experiment_provenance(release_mode="supported"):
    profile = release_profile(release_mode)
    return {
        "task_spec": sha256(SPEC),
        "python_sources": {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "release_mode": release_mode,
        "experiment_id": profile["experiment_id"],
        "acceptance_version": profile["acceptance_version"],
        "acceptance": copy.deepcopy(profile["acceptance"]),
    }


def scene_tree(variant="nominal", candidate="A", gripper="stock"):
    if gripper not in {"stock", "wide_candidate"}:
        raise ValueError("unknown gripper")
    diameter, mass = VARIANTS[variant]
    radius = diameter / 2
    spec = task_spec()
    root = make_tree("free", .01, candidate)
    root.set("model", f"tennis-ground-return-v1-{candidate}-{variant}-{gripper}")
    if gripper == "wide_candidate":
        fingers = spec["wide_gripper_candidate"]
        offset = fingers["pad_outward_offset_each_m"]
        full_size = np.asarray(fingers["bridge_full_size_m"])
        root.find(".//site[@name='arm_tcp']").set("pos", numbers([fingers["tcp_offset_m"], 0, 0]))
        for side, sign in (("left", 1), ("right", -1)):
            finger = root.find(f".//body[@name='arm_finger_{side}']")
            finger.find(f"geom[@name='arm_pad_{side}']").set("pos", numbers([fingers["pad_center_x_m"], sign * offset, 0]))
            ET.SubElement(finger, "geom", name=f"arm_wide_bridge_{side}", type="box",
                          pos=numbers([fingers["bridge_center_x_m"], sign * offset / 2, fingers["bridge_center_z_m"]]),
                          size=numbers(full_size / 2), mass=str(float(np.prod(full_size) * fingers["assumed_density_kg_m3"])),
                          rgba="0.95 0.50 0.12 1", contype="1", conaffinity="3")
            stem_size = np.asarray(fingers["stem_full_size_m"])
            ET.SubElement(finger, "geom", name=f"arm_wide_stem_{side}", type="box",
                          pos=numbers([fingers["stem_center_x_m"], sign * (offset + fingers["stem_outward_offset_m"]), fingers["bridge_center_z_m"]]),
                          size=numbers(stem_size / 2), mass=str(float(np.prod(stem_size) * fingers["assumed_density_kg_m3"])),
                          rgba="0.95 0.50 0.12 1", contype="1", conaffinity="3")
    world = root.find("worldbody")
    for name in ("work_surface", "obstacle"):
        world.remove(world.find(f"geom[@name='{name}']"))
    ball = root.find(".//body[@name='task_object']")
    ball.set("pos", numbers([*spec["ball"]["initial_xy_m"], radius]))
    geom = ball.find("geom[@name='object_geom']")
    geom.set("type", "sphere")
    geom.set("size", str(radius))
    geom.set("mass", str(mass))
    geom.set("friction", numbers(spec["ball"]["friction"]))
    geom.set("rgba", "0.8 0.95 0.05 1")
    inertia = 2 * mass * radius**2 / 3
    ET.SubElement(ball, "inertial", pos="0 0 0", mass=str(mass), diaginertia=numbers([inertia] * 3))
    container = spec["bin"]
    center = np.array(container["center_xy_m"])
    half = np.array(container["inside_size_xy_m"]) / 2
    wall = container["wall_thickness_m"]
    bottom = container["bottom_thickness_m"]
    height = container["wall_height_m"]
    parts = [("bottom", [*center, bottom / 2], [*(half + wall), bottom / 2])]
    for axis, label in enumerate(("x", "y")):
        for sign, side in ((-1, "low"), (1, "high")):
            position = np.array([*center, bottom + height / 2])
            position[axis] += sign * (half[axis] + wall / 2)
            size = np.array([*(half + wall), height / 2])
            size[axis] = wall / 2
            parts.append((f"{label}_{side}", position, size))
    for name, position, size in parts:
        ET.SubElement(world, "geom", name=f"bin_{name}", type="box", pos=numbers(position),
                      size=numbers(size), rgba="0.15 0.42 0.72 1", contype="1", conaffinity="3",
                      friction="0.8 0.005 0.001")
    root.find(".//site[@name='goal']").set("pos", numbers([*center, bottom + radius]))
    return root


def feasibility(variant="nominal", gripper="stock"):
    if gripper not in {"stock", "wide_candidate"}:
        raise ValueError("unknown gripper")
    diameter, mass = VARIANTS[variant]
    arm = load_design()["arm"]
    opening = 2 * arm["gripper_pivot_half_spacing_m"] - arm["contact_pads"]["full_size_m"][1]
    if gripper == "wide_candidate":
        opening += 2 * task_spec()["wide_gripper_candidate"]["pad_outward_offset_each_m"]
    reach = sum(arm["link_lengths_m"])
    if gripper == "wide_candidate":
        reach += task_spec()["wide_gripper_candidate"]["tcp_offset_m"] - arm["link_lengths_m"][-1]
    return {
        "ball_diameter_m": diameter, "ball_mass_kg": mass, "gripper": gripper,
        "maximum_open_parallel_pad_gap_m": opening,
        "equatorial_grasp_aperture_pass": opening > diameter,
        "aperture_deficit_m": max(0., diameter - opening),
        "ball_only_torque_at_geometric_reach_nm": mass * 9.81 * reach,
        "arm_geometric_reach_m": reach,
        "qualified_payload_kg": None,
        "training_authorized_by_feasibility": False,
        "blockers": (["equatorial_grasp_aperture"] if opening <= diameter else ["wide_finger_structure_unqualified"]) + ["58g_class_payload_unqualified", "whole_body_ground_reach_unqualified"],
        "scope": "open_pad_width_only_not_a_proof_of_collision_free_entry_or_grasp",
        "hardware_release": False,
    }


def geometry_report(gripper="wide_candidate", release_mode="supported"):
    profile = release_profile(release_mode)
    records = []
    for variant in VARIANTS:
        environment = TennisReturnEnv(variant=variant, gripper=gripper, release_mode=release_mode)
        try:
            environment.reset(0)
            robot = environment.robot
            robot.data.qpos[robot.object_qpos:robot.object_qpos + 3] = robot.data.site_xpos[robot.tcp_id]
            follower = robot.model.jnt_qposadr[robot.model.joint("arm_gripper_follower").id]
            open_contacts = []
            first_pad_contact = None
            for angle in np.linspace(0, .32, 321):
                robot.data.qpos[robot.qpos_indices[-1]] = -angle
                robot.data.qpos[follower] = angle
                mujoco.mj_forward(robot.model, robot.data)
                names = set()
                penetrations = []
                for contact in robot.data.contact:
                    pair = {robot.model.geom(contact.geom1).name, robot.model.geom(contact.geom2).name}
                    if "object_geom" in pair and contact.dist < 0:
                        names.update(pair - {"object_geom"})
                        penetrations.append(float(-contact.dist))
                if angle == 0:
                    open_contacts = sorted(names)
                if {"arm_pad_left", "arm_pad_right"}.issubset(names):
                    first_pad_contact = {"each_joint_closure_rad": float(angle), "each_joint_closure_deg": float(np.degrees(angle)),
                                         "contact_geoms": sorted(names), "maximum_penetration_m": max(penetrations),
                                         "pads_only": names == {"arm_pad_left", "arm_pad_right"}}
                    break
            records.append({"variant": variant, "dimensions": feasibility(variant, gripper),
                            "open_ball_intersections": open_contacts,
                            "first_bilateral_pad_intersection": first_pad_contact,
                            "geometric_screen_passed": not open_contacts and first_pad_contact is not None and first_pad_contact["pads_only"]})
        finally:
            environment.close()
    return {"gripper": gripper, "release_mode": release_mode, "experiment_id": profile["experiment_id"],
            "acceptance_version": profile["acceptance_version"], "acceptance": copy.deepcopy(profile["acceptance"]),
            "records": records, "all_geometry_screens_passed": all(record["geometric_screen_passed"] for record in records),
            "method": "kinematic_contact_sweep_at_home_pose_1mrad_steps_not_physical_grip_or_task_success",
            "dynamic_task_passed": False, "hardware_release": False}


@dataclass
class ReturnMonitor:
    release_mode: str = "supported"
    gates: dict | None = field(default=None, repr=False)
    origin_checked: bool = False
    phase: str = "approach"
    dwell_s: float = 0.
    lifted: bool = False
    release_authorized: bool = False
    failure: str | None = None
    success: bool = False
    release_height_m: float | None = None
    release_velocity_m_s: list[float] | None = None
    free_fall_interval_s: float = 0.
    maximum_free_fall_interval_s: float = 0.
    experiment_id: str = field(init=False)
    acceptance_version: str = field(init=False)

    def __post_init__(self):
        profile = release_profile(self.release_mode)
        self.experiment_id = profile["experiment_id"]
        self.acceptance_version = profile["acceptance_version"]
        self.gates = copy.deepcopy(profile["acceptance"] if self.gates is None else self.gates)

    @staticmethod
    def _measured_ball_velocity(sample):
        velocity = sample.get("ball_velocity_m_s")
        if velocity is None:
            return None
        velocity = np.asarray(velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            return None
        return velocity

    def _half_height_velocity_ok(self, velocity):
        gates = self.gates
        return bool(
            np.linalg.norm(velocity) <= gates["placement_speed_max_m_s"]
            and velocity[2] <= gates["release_upward_velocity_max_m_s"]
            and np.linalg.norm(velocity[:2]) <= gates["release_lateral_speed_max_m_s"]
        )

    def advance(self, sample, dt):
        if self.failure or self.success:
            return
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        for value in sample.values():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                self.failure = "nonfinite_telemetry"
                return
        gates = self.gates
        if self.phase == "approach" and not self.origin_checked:
            self.origin_checked = True
            if sample["ball_inside_bin"] or abs(sample["ball_bottom_m"]) > .001 or sample["carry_distance_m"] > .001 or sample["any_robot_ball_contact"]:
                self.failure = "invalid_floor_start"
                return
        if sample["tilt_rad"] > gates["fall_tilt_rad"] or sample["trunk_height_m"] < gates["minimum_trunk_height_m"]:
            self.failure = "fall"
        elif sample["forbidden_contacts"]:
            self.failure = sorted(sample["forbidden_contacts"])[0]
        elif sample["joint_limit_violation"]:
            self.failure = "joint_limit"
        elif self.lifted and not sample["bilateral_grasp"] and not self.release_authorized:
            self.failure = "premature_release_or_throw"
        containment_ok = (
            sample.get("ball_physically_contained_in_bin", sample["ball_inside_bin"])
            if self.release_mode == "half_height" and self.release_authorized
            else sample["ball_inside_bin"]
        )
        if not self.failure and self.release_authorized and not containment_ok:
            self.failure = "ball_left_bin"
        elif not self.failure and self.release_mode == "supported" and self.release_authorized and (
            not sample["bottom_supported"] or sample["ball_speed_m_s"] > gates["placement_speed_max_m_s"]
        ):
            self.failure = "unsupported_or_fast_release"
        velocity = self._measured_ball_velocity(sample) if self.release_mode == "half_height" else None
        if self.release_mode == "half_height" and self.release_authorized:
            airborne = not sample["bottom_supported"] and not sample["any_robot_ball_contact"]
            self.free_fall_interval_s = self.free_fall_interval_s + dt if airborne else 0.
            self.maximum_free_fall_interval_s = max(
                self.maximum_free_fall_interval_s, self.free_fall_interval_s
            )
            if velocity is None:
                self.failure = self.failure or "missing_release_velocity"
            elif airborne and (
                velocity[2] > gates["airborne_upward_velocity_max_m_s"]
                or np.linalg.norm(velocity[:2]) > gates["airborne_lateral_speed_max_m_s"]
            ):
                self.failure = self.failure or "measured_throw_velocity"
            elif self.free_fall_interval_s > gates["maximum_free_fall_interval_s"] + 1e-9:
                self.failure = self.failure or "free_fall_interval_exceeded"
        if self.failure:
            return
        speed_ok = sample["ball_speed_m_s"] <= gates["placement_speed_max_m_s"]
        transitions = {
            "approach": (sample["tcp_ball_distance_m"] < .008 and sample["jaw_open"], .1, "grasp"),
            "grasp": (sample["bilateral_grasp"], .2, "lift"),
            "lift": (sample["bilateral_grasp"] and not sample["ball_supported"] and sample["ball_bottom_m"] >= gates["minimum_lift_clearance_m"], gates["lift_hold_s"], "carry"),
            "carry": (sample["bilateral_grasp"] and sample["carry_distance_m"] >= gates["minimum_carry_distance_m"] and sample["above_bin"], .1, "lower"),
            "retreat": (sample["jaw_open"] and containment_ok and sample["bottom_supported"] and not sample["any_robot_ball_contact"] and speed_ok and sample["tcp_ball_distance_m"] >= sample["ball_radius_m"] + gates["retreat_clearance_m"], gates["bin_hold_s"], "success"),
        }
        if self.release_mode == "supported":
            transitions["lower"] = (
                sample["bilateral_grasp"] and sample["ball_inside_bin"]
                and sample["bottom_supported"] and speed_ok,
                .2,
                "release",
            )
            transitions["release"] = (
                sample["jaw_open"] and not sample["any_robot_ball_contact"]
                and sample["bottom_supported"] and speed_ok,
                .1,
                "retreat",
            )
        else:
            transitions["lower"] = (
                sample["bilateral_grasp"] and sample["release_height_ok"]
                and velocity is not None and self._half_height_velocity_ok(velocity),
                gates["release_authorization_hold_s"],
                "release",
            )
            transitions["release"] = (
                sample["jaw_open"] and not sample["any_robot_ball_contact"],
                .1,
                "retreat",
            )
        condition, duration, next_phase = transitions[self.phase]
        self.dwell_s = self.dwell_s + dt if condition else 0.
        if self.dwell_s + 1e-9 >= duration:
            if next_phase == "release":
                if "ball_center_height_m" in sample:
                    self.release_height_m = float(sample["ball_center_height_m"])
                measured_velocity = self._measured_ball_velocity(sample)
                if measured_velocity is not None:
                    self.release_velocity_m_s = measured_velocity.tolist()
            self.phase, self.dwell_s = next_phase, 0.
            self.lifted |= next_phase == "carry"
            self.release_authorized |= next_phase == "release"
            self.success = next_phase == "success"


class TennisReturnEnv:
    """Separate controller/acceptance contract, preserving frozen v1-C sources."""

    def __init__(
        self,
        variant="nominal",
        candidate="A",
        max_steps=750,
        gripper="stock",
        release_mode="supported",
        plan_navigation=False,
    ):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be positive")
        if type(plan_navigation) is not bool:
            raise ValueError("plan_navigation must be a bool")
        profile = release_profile(release_mode)
        self.variant, self.candidate, self.max_steps = variant, candidate, max_steps
        self.gripper = gripper
        self.release_mode = release_mode
        self.plan_navigation = plan_navigation
        self.acceptance_profile = profile
        self.spec = task_spec()
        self.radius = VARIANTS[variant][0] / 2
        self.robot = IntegratedArmEnv(case="reach", mode="free", candidate=candidate)
        model = mujoco.MjModel.from_xml_string(ET.tostring(scene_tree(variant, candidate, gripper), encoding="unicode"))
        if [model.joint(index).name for index in range(model.njnt)] != [self.robot.model.joint(index).name for index in range(self.robot.model.njnt)]:
            raise ValueError("Experiment must preserve the robot joint mapping")
        if [model.body(index).name for index in range(model.nbody)] != [self.robot.model.body(index).name for index in range(self.robot.model.nbody)]:
            raise ValueError("Experiment must preserve the robot body mapping")
        self.robot.model = model
        self.robot.data = mujoco.MjData(model)
        self.robot.ik_data = mujoco.MjData(model)
        self.robot.robot_mass = float(model.body_mass[self.robot.robot_bodies].sum())
        self.monitor = ReturnMonitor(release_mode=release_mode, gates=profile["acceptance"])
        self.controller = GroundReturnController(self)
        self.done = True

    def reset(self, seed=0):
        self.robot.reset(seed=seed)
        position = np.array([*self.spec["ball"]["initial_xy_m"], self.radius])
        position[:2] += self.robot.np_random.uniform(-self.spec["ball"]["spawn_jitter_m"], self.spec["ball"]["spawn_jitter_m"], 2)
        address = self.robot.object_qpos
        self.robot.data.qpos[address:address + 3] = position
        mujoco.mj_forward(self.robot.model, self.robot.data)
        self.robot.object_start = position.copy()
        self.robot.goal = position.copy()
        self.monitor = ReturnMonitor(
            release_mode=self.release_mode,
            gates=self.acceptance_profile["acceptance"],
        )
        self.controller.reset()
        self.done = False
        self.steps = 0
        self.phase_events = [{"time_s": 0., "phase": "approach"}]
        return self.robot._observation().copy()

    def sample(self):
        robot = self.robot
        result = robot.diagnostics()
        position = robot.data.xpos[robot.object_id].copy()
        container = self.spec["bin"]
        center = np.asarray(container["center_xy_m"])
        half = np.asarray(container["inside_size_xy_m"]) / 2
        bottom = container["bottom_thickness_m"]
        rim = bottom + container["wall_height_m"]
        inside_xy = bool(np.all(np.abs(position[:2] - center) + self.radius <= half))
        bottom_supported, supported, robot_ball = False, False, False
        ball_bin_side_contacts = set()
        maximum_ball_bin_side_penetration = 0.
        forbidden = set(result["forbidden_contacts"])
        active_contacts = []
        force = np.zeros(6)
        for index in range(robot.data.ncon):
            contact = robot.data.contact[index]
            names = {robot.model.geom(contact.geom1).name, robot.model.geom(contact.geom2).name}
            bodies = {int(robot.model.geom_bodyid[contact.geom1]), int(robot.model.geom_bodyid[contact.geom2])}
            if contact.dist <= 0 and "object_geom" in names:
                side_contacts = {
                    name for name in names
                    if name and name.startswith(("bin_x_", "bin_y_"))
                }
                if side_contacts:
                    ball_bin_side_contacts.update(side_contacts)
                    maximum_ball_bin_side_penetration = max(
                        maximum_ball_bin_side_penetration,
                        float(max(0., -contact.dist)),
                    )
            mujoco.mj_contactForce(robot.model, robot.data, index, force)
            if contact.dist > 0 or force[0] <= .002:
                continue
            active_contacts.append({"geoms": sorted(name or f"unnamed_geom_{geom}" for name, geom in (
                (robot.model.geom(contact.geom1).name, contact.geom1),
                (robot.model.geom(contact.geom2).name, contact.geom2))),
                "penetration_m": float(max(0., -contact.dist)), "normal_force_n": float(force[0])})
            if "object_geom" in names:
                bottom_supported |= "bin_bottom" in names
                supported |= "floor" in names or any(name and name.startswith("bin_") for name in names)
                robot_ball |= bool(bodies - {0, robot.object_id})
                if bodies - {0, robot.object_id} and not names & {"arm_pad_left", "arm_pad_right"}:
                    forbidden.add("non_pad_ball_collision")
            if any(name and name.startswith("bin_") for name in names) and bodies - {0, robot.object_id}:
                forbidden.add("robot_bin_collision")
        velocity_index = robot.model.jnt_dofadr[robot.model.joint("task_object_free").id]
        ball_velocity = robot.data.qvel[velocity_index:velocity_index + 3].copy()
        joints = robot.data.qpos[robot.qpos_indices]
        release_center_max = (
            bottom + self.monitor.gates["release_ball_center_max_wall_height_fraction"] * container["wall_height_m"]
            if self.release_mode == "half_height"
            else bottom + self.radius
        )
        vertical_inside = bool(
            position[2] + self.radius <= rim
            and position[2] - self.radius >= bottom - .001
        )
        ball_inside_bin = bool(inside_xy and vertical_inside)
        physical_containment = ball_inside_bin
        contact_containment_used = False
        containment_tolerance = (
            self.monitor.gates["side_contact_containment_tolerance_m"]
            if self.release_mode == "half_height"
            else 0.
        )
        if self.release_mode == "half_height" and vertical_inside and not inside_xy:
            overrun = np.abs(position[:2] - center) + self.radius - half
            expected_contacts = set()
            for axis, label in enumerate(("x", "y")):
                if overrun[axis] > 0:
                    side = "high" if position[axis] >= center[axis] else "low"
                    expected_contacts.add(f"bin_{label}_{side}")
            contact_containment_used = bool(
                expected_contacts
                and expected_contacts <= ball_bin_side_contacts
                and float(np.max(overrun)) <= containment_tolerance + 1e-12
                and maximum_ball_bin_side_penetration <= containment_tolerance + 1e-12
            )
            physical_containment = contact_containment_used
        result.update(
            ball_position_m=position.tolist(), ball_radius_m=self.radius,
            ball_center_height_m=float(position[2]),
            ball_bottom_m=float(position[2] - self.radius), bottom_supported=bottom_supported,
            ball_supported=supported, any_robot_ball_contact=robot_ball,
            ball_inside_bin=ball_inside_bin,
            ball_geometrically_inside_bin=ball_inside_bin,
            ball_physically_contained_in_bin=physical_containment,
            contact_containment_used=contact_containment_used,
            ball_bin_side_contact_geoms=sorted(ball_bin_side_contacts),
            maximum_ball_bin_side_penetration_m=maximum_ball_bin_side_penetration,
            side_contact_containment_tolerance_m=containment_tolerance,
            above_bin=bool(inside_xy and position[2] - self.radius >= rim + .01),
            ball_velocity_m_s=ball_velocity.tolist(),
            ball_speed_m_s=float(np.linalg.norm(ball_velocity)),
            ball_lateral_speed_m_s=float(np.linalg.norm(ball_velocity[:2])),
            ball_vertical_velocity_m_s=float(ball_velocity[2]),
            release_height_ok=bool(
                ball_inside_bin
                and position[2] <= release_center_max + 1e-9
                and (self.release_mode == "half_height" or bottom_supported)
            ),
            release_center_max_m=float(release_center_max),
            release_mode=self.release_mode,
            experiment_id=self.monitor.experiment_id,
            acceptance_version=self.monitor.acceptance_version,
            release_authorization_height_m=self.monitor.release_height_m,
            release_authorization_velocity_m_s=self.monitor.release_velocity_m_s,
            free_fall_interval_s=self.monitor.free_fall_interval_s,
            maximum_free_fall_interval_s=self.monitor.maximum_free_fall_interval_s,
            carry_distance_m=float(np.linalg.norm(position[:2] - robot.object_start[:2])),
            tcp_ball_distance_m=float(np.linalg.norm(robot.data.site_xpos[robot.tcp_id] - position)),
            jaw_open=bool(joints[-1] >= -.05), forbidden_contacts=sorted(forbidden),
            active_contacts=active_contacts, joint_positions_rad=joints.tolist(),
            commanded_joint_positions_rad=robot.command_targets.tolist(),
            joint_limit_violation=bool(np.any(joints < robot.joint_limits[:, 0] - .08) or np.any(joints > robot.joint_limits[:, 1] + .08)),
            maximum_nominal_joint_excursion_rad=float(max(0., np.max(np.maximum(robot.joint_limits[:, 0] - joints, joints - robot.joint_limits[:, 1])))),
            evaluator_joint_tolerance_rad=.08,
            release_plan_blocked=bool(self.controller.release_blocked),
            cached_release_screen_passed=bool(self.controller.cached_release_screen_passed),
            release_strategy=self.controller.release_strategy,
            release_prediction=self.controller.release_prediction,
            docking_prediction=self.controller.docking_prediction,
            docking_qualified=bool(self.controller.docking_qualified),
            placement_drop_m=self.controller.placement_drop_m,
            navigation_recovery_started_s=self.controller.navigator.recovery_started if self.controller.navigator else None,
            navigation_prediction=self.controller.navigation_prediction,
            navigation_qualified=bool(self.controller.navigation_qualified),
            navigation_profile=asdict(self.controller.navigator.profile) if self.controller.navigator else None,
            plan_navigation=self.plan_navigation,
        )
        return result

    def teacher_action(self, control="teacher"):
        if control not in {"teacher", "open_jaw", "hold"}:
            raise ValueError("unknown controller")
        robot = self.robot
        if self.gripper == "wide_candidate":
            return robot.normalize_targets(self.controller.targets(control))
        targets = robot.home.copy()
        targets[:10] = robot.leg_policy.leg_targets(robot)
        if control == "hold":
            return robot.normalize_targets(targets)
        ball = robot.data.xpos[robot.object_id].copy()
        container = self.spec["bin"]
        goal = np.array([*container["center_xy_m"], container["bottom_thickness_m"] + self.radius])
        phase = self.monitor.phase
        if phase in {"approach", "grasp"}:
            tcp_target = ball
        elif phase == "lift":
            tcp_target = robot.object_start + [0, 0, .04]
        elif phase == "carry":
            tcp_target = goal + [0, 0, container["wall_height_m"] + .02]
        elif phase in {"lower", "release"}:
            tcp_target = goal
        else:
            tcp_target = goal + [0, 0, container["wall_height_m"] + .05]
        if phase not in {"approach", "release", "retreat", "success"} and control != "open_jaw":
            targets[-1] = robot.joint_limits[-1, 0]
        robot.goal = tcp_target.copy()
        targets[10:14] = robot._inverse_kinematics(tcp_target)
        return robot.normalize_targets(targets)

    def step(self, action):
        if self.done:
            raise RuntimeError("reset required")
        robot = self.robot
        target = robot.denormalize_action(action)
        speeds = np.array([robot.design["control"]["leg_joint_slew_limit_rad_s"]] * 10 + [robot.design["control"]["arm_joint_slew_limit_rad_s"]] * 5)
        robot.command_targets += np.clip(target - robot.command_targets, -speeds * robot.dt, speeds * robot.dt)
        robot.data.ctrl[:] = robot.command_targets
        for _ in range(robot.substeps):
            mujoco.mj_step(robot.model, robot.data)
            previous_phase = self.monitor.phase
            self.monitor.advance(self.sample(), robot.model.opt.timestep)
            if previous_phase != self.monitor.phase:
                self.phase_events.append({"time_s": float(robot.data.time), "phase": self.monitor.phase})
            if self.monitor.failure or self.monitor.success:
                break
        self.steps += 1
        robot.steps = self.steps
        robot.previous_action = np.asarray(action, dtype=np.float32).copy()
        self.done = bool(self.monitor.failure or self.monitor.success or self.steps >= self.max_steps)
        info = self.sample()
        info.update(phase=self.monitor.phase, success=self.monitor.success,
                    controller_stage=self.controller.stage if self.gripper == "wide_candidate" else "legacy_diagnostic",
                    failure_reason=self.monitor.failure or ("timeout" if self.done and not self.monitor.success else None))
        return robot._observation().copy(), self.done, info

    def render(self):
        robot = self.robot
        if robot._renderer is None:
            robot._renderer = mujoco.Renderer(robot.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.18, .03, .10]
        camera.distance, camera.azimuth, camera.elevation = .95, 125, -28
        robot._renderer.update_scene(robot.data, camera=camera)
        return robot._renderer.render().copy()

    def close(self):
        self.robot.close()


def run_episode(
    output,
    variant="nominal",
    candidate="A",
    seed=0,
    control="teacher",
    max_steps=750,
    video=False,
    gripper="stock",
    release_mode="supported",
    plan_navigation=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = TennisReturnEnv(
        variant=variant,
        candidate=candidate,
        max_steps=max_steps,
        gripper=gripper,
        release_mode=release_mode,
        plan_navigation=plan_navigation,
    )
    writer = None
    try:
        observation = environment.reset(seed)
        if video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(output / "rollout.mp4", fps=25, codec="libx264", macro_block_size=16)
        with (output / "telemetry.jsonl").open("x") as telemetry:
            while not environment.done:
                action = environment.teacher_action(control)
                observation = environment.robot._observation().copy()
                next_observation, done, info = environment.step(action)
                telemetry.write(json.dumps({"step": environment.steps, "observation_before": observation.tolist(),
                                            "action": action.tolist(), "observation_after": next_observation.tolist(),
                                            "done": done, **info}, allow_nan=False) + "\n")
                observation = next_observation
                if writer is not None and (environment.steps % 2 == 0 or done):
                    from PIL import Image, ImageDraw
                    image = Image.fromarray(environment.render())
                    drawing = ImageDraw.Draw(image)
                    drawing.rectangle((0, 0, 640, 46), fill="black")
                    drawing.text((8, 5), f"SIMULATION DIAGNOSTIC | tennis return | {variant} | {control}", fill="white")
                    status = "TASK PASS" if info["success"] else info["failure_reason"] or "NOT YET PASSED"
                    drawing.text((8, 25), f"t={info['time_s']:.2f}s | {info['phase']} | {status}", fill="yellow")
                    writer.append_data(np.asarray(image))
        profile = release_profile(release_mode)
        result = {"variant": variant, "candidate": candidate, "seed": seed, "controller": control,
                  "success": info["success"], "failure_reason": info["failure_reason"], "steps": environment.steps,
                  "last_sample": info, "feasibility": feasibility(variant, gripper), "gripper": gripper, "simulation_only": True,
                  "release_mode": release_mode, "experiment_id": profile["experiment_id"],
                  "acceptance_version": profile["acceptance_version"], "acceptance": copy.deepcopy(profile["acceptance"]),
                  "experiment_provenance": experiment_provenance(release_mode),
                  "release_authorization_height_m": environment.monitor.release_height_m,
                  "release_authorization_velocity_m_s": environment.monitor.release_velocity_m_s,
                  "maximum_free_fall_interval_s": environment.monitor.maximum_free_fall_interval_s,
                  "plan_navigation": plan_navigation,
                  "navigation_prediction": environment.controller.navigation_prediction,
                  "navigation_qualified": bool(environment.controller.navigation_qualified),
                  "navigation_profile": asdict(environment.controller.navigator.profile) if environment.controller.navigator else None,
                  "hardware_release": False, "phase_events": environment.phase_events,
                  "ground_lift_passed": environment.monitor.lifted, "max_steps": max_steps,
                  "controller_scope": "whole_body_IK_legacy_leg_ONNX_and_simulator_truth_clone_teacher_not_PPO_or_hardware_runtime" if gripper == "wide_candidate" else "stock_legs_stand_plus_arm_IK_diagnostic_not_trained_return_or_navigation"}
    finally:
        if writer is not None:
            writer.close()
        environment.close()
    result["files"] = {path.name: sha256(path) for path in output.iterdir() if path.is_file()}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def run_episode_job(job):
    return run_episode(**job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--video-seeds", default="", help="Comma-separated seeds to render; empty produces metrics-only screening, not a video-verifiable bundle")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--gripper", choices=("stock", "wide_candidate"), default="stock")
    parser.add_argument("--release-mode", choices=RELEASE_MODES, default="half_height")
    parser.add_argument("--plan-navigation", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    video_seeds = {int(value) for value in args.video_seeds.split(",") if value}
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0 or not video_seeds.issubset(seeds):
        parser.error("distinct nonnegative seeds required; video seeds must be included")
    if args.workers < 1:
        parser.error("workers must be positive")
    snapshot = source_provenance()
    profile = release_profile(args.release_mode)
    own_hashes = experiment_provenance(args.release_mode)
    args.out.mkdir(parents=True, exist_ok=False)
    source_directory = args.out / "source-snapshot"
    source_directory.mkdir()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(source, source_directory / source.name)
    shutil.copyfile(SPEC, source_directory / SPEC.name)
    records = []
    jobs = []
    (args.out / "geometry-check.json").write_text(
        json.dumps(geometry_report(args.gripper, args.release_mode), indent=2) + "\n"
    )
    for variant in VARIANTS:
        root = scene_tree(variant, gripper=args.gripper)
        ET.indent(root)
        ET.ElementTree(root).write(args.out / f"{variant}.xml", encoding="utf-8", xml_declaration=True)
        for seed in seeds:
            jobs.append({"output": args.out / f"{variant}-{seed}", "variant": variant, "seed": seed,
                         "max_steps": args.max_steps, "video": seed in video_seeds, "gripper": args.gripper,
                         "release_mode": args.release_mode, "plan_navigation": args.plan_navigation})
    executor = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    try:
        results = executor.map(run_episode_job, jobs) if executor else map(run_episode_job, jobs)
        for result in results:
            records.append(result)
            print(json.dumps({key: result[key] for key in ("variant", "seed", "success", "failure_reason")}), flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)
    negatives = [
        run_episode(
            args.out / control,
            control=control,
            max_steps=args.max_steps,
            video=bool(video_seeds),
            gripper=args.gripper,
            release_mode=args.release_mode,
            plan_navigation=args.plan_navigation,
        )
        for control in ("open_jaw", "hold")
    ]
    if snapshot != source_provenance() or own_hashes != experiment_provenance(args.release_mode):
        raise RuntimeError("Source changed during evaluation; evidence cannot be published")
    report = {"experiment_id": profile["experiment_id"], "acceptance_version": profile["acceptance_version"],
              "acceptance": copy.deepcopy(profile["acceptance"]), "release_mode": args.release_mode,
              "plan_navigation": args.plan_navigation, "episodes": len(records),
              "successes": sum(record["success"] for record in records),
              "ground_lifts": sum(record["ground_lift_passed"] for record in records),
              "max_steps": args.max_steps, "requested_seeds": list(seeds),
              "video_seeds": sorted(video_seeds),
              "evidence_type": "metrics_and_video" if video_seeds else "metrics_only",
              "all_tasks_passed": all(record["success"] for record in records),
              "negative_controls_passed": all(not record["success"] for record in negatives),
              "records": records, "negative_controls": negatives,
              "status": "EXPERIMENTAL_NO_HARDWARE_QUALIFICATION", "gripper": args.gripper,
              "candidate": "A", "candidate_comparison_qualified": False,
              "source_provenance": snapshot, "experiment_hashes": own_hashes,
              "training_steps": 0, "hardware_release": False, "simulation_only": True}
    if args.gripper == "wide_candidate":
        from .tennis_release import FORECAST_PROVENANCE

        report["planning_provenance"] = dict(FORECAST_PROVENANCE)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("episodes", "successes", "all_tasks_passed", "status")}), flush=True)
    return 0 if report["all_tasks_passed"] and report["negative_controls_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
