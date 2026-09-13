from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .config import CASES, DESIGN_ID, SPEC_PATH, load_design, sha256
from .model import compile_model


class IntegratedArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, case="reach", mode="free", max_steps=1500, payload_kg=0.01, render_mode=None):
        if case not in CASES:
            raise ValueError(f"Unknown case: {case}")
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if mode == "fixture" and case in ("stowed_arm_walking", "walking_carry"):
            raise ValueError("Walking tasks require the free-base model, never a fixture")
        self.design = load_design()
        self.case, self.mode, self.max_steps = case, mode, max_steps
        self.payload_kg = payload_kg
        self.render_mode = render_mode
        self.model = compile_model(mode, payload_kg)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.dt = self.design["control"]["dt_s"]
        self.substeps = round(self.dt / self.model.opt.timestep)
        self.home = np.array(self.design["home_rad"], dtype=float)
        self.joint_ids = np.array([self.model.joint(name).id for name in self.design["joint_order"]])
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        self.action_scales = np.full(15, self.design["control"]["pose_action_scale_rad"])
        self.action_offsets = self.home.copy()
        self.action_scales[10:] = np.diff(self.joint_limits[10:], axis=1).ravel() / 2
        self.action_offsets[10:] = self.joint_limits[10:].mean(axis=1)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(15,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(64,), dtype=np.float32)
        self.contract = DESIGN_ID + ":obs64-action15:v1"
        self.trunk_id = self.model.body("trunk_base").id
        self.object_id = self.model.body("task_object").id
        self.tcp_id = self.model.site("arm_tcp").id
        self.object_qpos = self.model.jnt_qposadr[self.model.joint("task_object_free").id]
        self.root_joint_id = None if mode == "fixture" else self.model.joint("trunk_base_freejoint").id
        self.robot_bodies = np.array([body_id for body_id in range(1, self.model.nbody) if body_id != self.object_id])
        self.robot_mass = float(self.model.body_mass[self.robot_bodies].sum())
        self.foot_body_ids = {self.model.body(name).id for name in ("ankle_left", "ankle_right")}
        self._renderer = None
        self._done = True

    def normalize_targets(self, targets):
        return np.clip((np.asarray(targets) - self.action_offsets) / self.action_scales, -1, 1).astype(np.float32)

    def denormalize_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (15,) or not np.isfinite(action).all():
            raise ValueError("Expected finite v1-B action contract with shape (15,)")
        if np.any(np.abs(action) > 1.000001):
            raise ValueError("Action outside normalized [-1,1] contract")
        return np.clip(self.action_offsets + self.action_scales * action, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Reset options are not supported; use a versioned scenario configuration")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_indices] = self.home
        self.data.ctrl[:] = self.home
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = [0.125, -0.018, 0.103]
        self.data.qpos[self.object_qpos:self.object_qpos + 2] += self.np_random.uniform(-0.004, 0.004, 2)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.previous_action = self.normalize_targets(self.home)
        self.command_targets = self.home.copy()
        self.phase = 0
        self.hold_ticks = 0
        self.lift_ticks = 0
        self.ever_lifted = False
        self.ever_grasped = False
        self.release_authorized = False
        self.maximum_lifted_height = -np.inf
        self.support_events = []
        self.candidate_support = None
        self.candidate_support_ticks = 0
        self.carry_anchor = None
        self.waypoints_completed = 0
        self.forbidden_contacts = set()
        self.object_start = self.data.xpos[self.object_id].copy()
        self.trunk_start = self.data.xpos[self.trunk_id].copy()
        self.tcp_start = self.data.site_xpos[self.tcp_id].copy()
        self.goal = np.array([0.15, 0.024, 0.102])
        if self.case == "reach":
            self.goal = self.tcp_start + self.np_random.uniform([-.012, -.025, .020], [.006, .025, .028])
        mount_origin = self.data.xpos[self.model.body("arm_mount").id]
        horizontal_axis = np.array([self.goal[0] - mount_origin[0], self.goal[1] - mount_origin[1], 0])
        self.goal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)
        self.model.site_pos[self.model.site("goal").id] = self.goal
        self.waypoints = [np.array([0.12, -.01, .135]), np.array([.145, .024, .135])]
        self.command_velocity = np.array([.015, 0, 0]) if "walking" in self.case else np.zeros(3)
        self._done = False
        self.last_diagnostics = self.diagnostics()
        return self._observation(), self._info(False, None)

    def _observation(self):
        rotation = self.data.xmat[self.trunk_id].reshape(3, 3)
        origin = self.data.xpos[self.trunk_id]
        observation = np.concatenate([
            self.data.qpos[self.qpos_indices] - self.home,
            self.data.qvel[self.dof_indices],
            rotation.T @ np.array([0, 0, -1.0]),
            self.data.sensor("imu_ang_vel").data,
            rotation.T @ (self.data.site_xpos[self.tcp_id] - origin),
            rotation.T @ (self.data.xpos[self.object_id] - origin),
            rotation.T @ (self.goal - origin),
            self.command_velocity, self.previous_action, [self.phase / 7.0],
        ]).astype(np.float32)
        if observation.shape != (64,) or not np.isfinite(observation).all():
            raise FloatingPointError("Non-finite observation or schema mismatch")
        return observation

    def _contact_state(self):
        feet, grasp_sides, contacts, maximum_force = [], set(), set(), 0.0
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.dist > 0:
                continue
            first, second = int(contact.geom1), int(contact.geom2)
            names = {self.model.geom(first).name, self.model.geom(second).name}
            bodies = {int(self.model.geom_bodyid[first]), int(self.model.geom_bodyid[second])}
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force = abs(float(force[0]))
            maximum_force = max(maximum_force, normal_force)
            if "floor" in names and bodies & self.foot_body_ids and normal_force > .01:
                feet.append(contact.pos.copy())
            if "object_geom" in names:
                for side in ("left", "right"):
                    if f"arm_pad_{side}" in names and normal_force > .002:
                        grasp_sides.add(side)
                if "obstacle" in names:
                    contacts.add("object_obstacle")
            if "floor" in names and not (bodies & self.foot_body_ids) and self.object_id not in bodies:
                contacts.add("body_ground")
            if ("work_surface" in names or "obstacle" in names) and any(name and name.startswith("arm_") for name in names):
                contacts.add("arm_environment")
            if len(bodies) == 2 and 0 not in bodies and self.object_id not in bodies:
                contacts.add("robot_self_collision")
        return feet, grasp_sides, contacts, maximum_force

    @staticmethod
    def support_margin(points, point):
        if len(points) < 3:
            return None
        vertices = sorted(set(tuple(vertex[:2]) for vertex in points))
        if len(vertices) < 3:
            return None
        def cross(origin, first, second):
            return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])
        lower, upper = [], []
        for sequence, half in ((vertices, lower), (vertices[::-1], upper)):
            for vertex in sequence:
                while len(half) >= 2 and cross(half[-2], half[-1], vertex) <= 0:
                    half.pop()
                half.append(vertex)
        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            return None
        return float(min(cross(start, end, point) / np.linalg.norm(np.subtract(end, start)) for start, end in zip(hull, hull[1:] + hull[:1])))

    def diagnostics(self):
        feet, grasp, forbidden, peak_force = self._contact_state()
        masses = self.model.body_mass[self.robot_bodies]
        com = np.sum(self.data.xipos[self.robot_bodies] * masses[:, None], axis=0) / masses.sum()
        loaded_com = com.copy()
        if len(grasp) == 2:
            object_mass = self.model.body_mass[self.object_id]
            loaded_com = (com * self.robot_mass + self.data.xipos[self.object_id] * object_mass) / (self.robot_mass + object_mass)
        tilt = float(np.arccos(np.clip(self.data.xmat[self.trunk_id].reshape(3, 3)[2, 2], -1, 1)))
        tool_axis = self.data.site_xmat[self.tcp_id].reshape(3, 3)[:, 0]
        target_rotation = np.column_stack([self.goal_axis, np.cross([0, 0, 1], self.goal_axis), [0, 0, 1]])
        orientation_cosine = (np.trace(target_rotation.T @ self.data.site_xmat[self.tcp_id].reshape(3, 3)) - 1) / 2
        object_supported = any(
            "object_geom" in {self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name}
            and bool({self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name} & {"floor", "work_surface", "obstacle"})
            for contact in self.data.contact
        )
        return {
            "time_s": float(self.data.time), "robot_mass_kg": self.robot_mass,
            "robot_com_m": com.tolist(), "loaded_system_com_m": loaded_com.tolist(), "support_contact_count": len(feet),
            "static_support_margin_m": self.support_margin(feet, loaded_com), "tilt_rad": tilt,
            "trunk_height_m": float(self.data.xpos[self.trunk_id, 2]),
            "tcp_error_m": float(np.linalg.norm(self.data.site_xpos[self.tcp_id] - self.goal)),
            "tcp_axis_error_rad": float(np.arccos(np.clip(tool_axis @ self.goal_axis, -1, 1))),
            "tcp_orientation_error_rad": float(np.arccos(np.clip(orientation_cosine, -1, 1))),
            "object_supported": object_supported,
            "object_height_m": float(self.data.xpos[self.object_id, 2]),
            "object_goal_error_m": float(np.linalg.norm(self.data.xpos[self.object_id] - self.goal)),
            "bilateral_grasp": len(grasp) == 2, "forbidden_contacts": sorted(forbidden),
            "max_contact_normal_force_n": peak_force,
            "arm_actuator_force_nm": self.data.actuator_force[10:].tolist(),
            "forward_displacement_m": float(self.data.xpos[self.trunk_id, 0] - self.trunk_start[0]),
            "fixture_assisted": self.mode == "fixture", "physical_measurements": False,
        }

    def _info(self, success, failure):
        return {"case": self.case, "mode": self.mode, "design_id": DESIGN_ID, "contract": self.contract,
                "spec_sha256": sha256(SPEC_PATH), "success": bool(success), "failure_reason": failure,
                "phase": self.phase, "diagnostics": self.last_diagnostics,
                "simulation_only": True, "fixture_assisted": self.mode == "fixture",
                "hardware_ready": False, "actor_pose_source": "simulator_truth_estimator_not_implemented"}

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() required before stepping a finished episode")
        previously_grasped = self.last_diagnostics["bilateral_grasp"]
        target = self.denormalize_action(action)
        delta = self.design["control"]["joint_speed_limit_rad_s"] * self.dt
        self.command_targets += np.clip(target - self.command_targets, -delta, delta)
        self.data.ctrl[:] = self.command_targets
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            _, _, forbidden, _ = self._contact_state()
            self.forbidden_contacts.update(forbidden)
        self.steps += 1
        self.phase = 0 if self.case in ("reach", "stowed_arm_walking") else min(int(self.steps * self.dt // 3), 7)
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        self.last_diagnostics = self.diagnostics()
        diagnostics = self.last_diagnostics
        self.ever_grasped |= diagnostics["bilateral_grasp"]
        object_velocity_index = self.model.jnt_dofadr[self.model.joint("task_object_free").id]
        placement_velocity = float(np.linalg.norm(self.data.qvel[object_velocity_index:object_velocity_index + 3]))
        release_window = self.case in ("pick_place", "obstacle_relocation", "stationary_waypoint_carry") and self.phase >= 6 and self.ever_lifted and diagnostics["object_goal_error_m"] < .010 and placement_velocity < .030 and self.command_targets[-1] >= -.05
        self.release_authorized = bool(release_window and (previously_grasped or self.release_authorized))
        if self.ever_grasped:
            self.maximum_lifted_height = max(self.maximum_lifted_height, diagnostics["object_height_m"])
        if diagnostics["bilateral_grasp"] and diagnostics["object_height_m"] - self.object_start[2] >= .020:
            self.lift_ticks += 1
            self.ever_lifted |= self.lift_ticks * self.dt >= 1.0
        else:
            self.lift_ticks = 0
        object_relative = self.data.xmat[self.trunk_id].reshape(3, 3).T @ (self.data.xpos[self.object_id] - self.data.xpos[self.trunk_id])
        if self.ever_lifted and diagnostics["bilateral_grasp"] and self.carry_anchor is None:
            self.carry_anchor = object_relative.copy()
        failure = None
        if diagnostics["tilt_rad"] > .6 or diagnostics["trunk_height_m"] < .065:
            failure = "fall"
        elif self.forbidden_contacts:
            failure = sorted(self.forbidden_contacts)[0]
        elif self.ever_lifted and not diagnostics["bilateral_grasp"] and not self.release_authorized:
            failure = "object_drop"
        current = self.data.qpos[self.qpos_indices]
        if np.any(current < self.joint_limits[:, 0] - .08) or np.any(current > self.joint_limits[:, 1] + .08):
            failure = "joint_limit"
        target_met = diagnostics["tcp_error_m"] < .010 and diagnostics["tcp_orientation_error_rad"] < self.design["task_targets"]["orientation_tolerance_rad"] if self.case == "reach" else False
        if self.case in ("pick_place", "obstacle_relocation", "stationary_waypoint_carry"):
            target_met = self.ever_lifted and self.release_authorized and diagnostics["object_goal_error_m"] < .010 and placement_velocity < .030 and not diagnostics["bilateral_grasp"] and abs(self.data.qpos[self.qpos_indices[-1]]) < .05
            if self.case == "stationary_waypoint_carry":
                if self.waypoints_completed < len(self.waypoints) and diagnostics["bilateral_grasp"] and np.linalg.norm(self.data.xpos[self.object_id] - self.waypoints[self.waypoints_completed]) < .010:
                    self.waypoints_completed += 1
                target_met &= self.waypoints_completed == len(self.waypoints)
        if "walking" in self.case:
            active_feet = set()
            force = np.zeros(6)
            for contact_index in range(self.data.ncon):
                contact = self.data.contact[contact_index]
                if "floor" in {self.model.geom(contact.geom1).name, self.model.geom(contact.geom2).name}:
                    mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                    if force[0] > .01:
                        active_feet.update({int(self.model.geom_bodyid[contact.geom1]), int(self.model.geom_bodyid[contact.geom2])} & self.foot_body_ids)
            upright = diagnostics["tilt_rad"] < .25 and failure is None
            if not upright:
                self.support_events.clear()
            support = next(iter(active_feet)) if len(active_feet) == 1 and upright else None
            self.candidate_support_ticks = self.candidate_support_ticks + 1 if support is not None and support == self.candidate_support else 0
            self.candidate_support = support
            displacement = diagnostics["forward_displacement_m"]
            if support is not None and self.candidate_support_ticks * self.dt >= .10:
                if not self.support_events or (support != self.support_events[-1][0] and displacement - self.support_events[-1][1] >= .005):
                    self.support_events.append((support, displacement, self.data.time))
            recent_events = [event for event in self.support_events if self.data.time - event[2] <= 4.0]
            target_met = displacement >= .05 and len(recent_events) >= 3 and upright
            if self.case == "stowed_arm_walking":
                target_met &= bool(np.max(np.abs(current[10:] - self.home[10:])) < .10)
            if self.case == "walking_carry":
                target_met &= self.ever_lifted and diagnostics["bilateral_grasp"] and not diagnostics["object_supported"] and diagnostics["object_height_m"] - self.object_start[2] >= .020 and self.carry_anchor is not None and np.linalg.norm(object_relative - self.carry_anchor) < .025
        if self.mode == "free" and "walking" not in self.case:
            margin = diagnostics["static_support_margin_m"]
            target_met &= margin is not None and margin >= self.design["task_targets"]["static_margin_m"]
        self.hold_ticks = self.hold_ticks + 1 if target_met and failure is None else 0
        success = self.hold_ticks * self.dt >= self.design["task_targets"]["hold_time_s"]
        reward = -diagnostics["tcp_error_m"] - .05 * diagnostics["tilt_rad"] - .0001 * float(np.sum(self.data.qvel[self.dof_indices] ** 2))
        if "walking" in self.case:
            reward += diagnostics["forward_displacement_m"]
        reward += 1.0 if success else 0.0
        reward -= 1.0 if failure else 0.0
        terminated = bool(success or failure)
        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            failure = "timeout"
        self._done = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info(success, failure)

    def _inverse_kinematics(self, target, initial=None):
        self.ik_data.qpos[:] = self.data.qpos
        if initial is not None:
            self.ik_data.qpos[self.qpos_indices[10:14]] = initial
        translation = np.zeros((3, self.model.nv))
        rotation = np.zeros((3, self.model.nv))
        for _ in range(40):
            mujoco.mj_forward(self.model, self.ik_data)
            error = target - self.ik_data.site_xpos[self.tcp_id]
            tool_axis = self.ik_data.site_xmat[self.tcp_id].reshape(3, 3)[:, 0]
            if np.linalg.norm(error) < .0005 and abs(tool_axis[2]) < .02:
                break
            mujoco.mj_jacSite(self.model, self.ik_data, translation, rotation, self.tcp_id)
            orientation_row = .04 * np.cross(tool_axis, [0, 0, 1]) @ rotation[:, self.dof_indices[10:14]]
            jacobian = np.vstack([translation[:, self.dof_indices[10:14]], orientation_row])
            error = np.append(error, -.04 * tool_axis[2])
            update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .00001 * np.eye(4), error)
            selected = self.ik_data.qpos[self.qpos_indices[10:14]] + np.clip(update, -.15, .15)
            self.ik_data.qpos[self.qpos_indices[10:14]] = np.clip(selected, self.joint_limits[10:14, 0], self.joint_limits[10:14, 1])
        return self.ik_data.qpos[self.qpos_indices[10:14]].copy()

    def teacher_action(self):
        target = self.home.copy()
        if self.case == "stowed_arm_walking":
            return self.normalize_targets(target)
        tcp_goal = self.goal.copy()
        if self.case != "reach":
            object_center = self.data.xpos[self.object_id].copy()
            stages = [self.object_start + [0, 0, .035], self.object_start,
                      self.object_start, self.object_start + [0, 0, .035],
                      self.waypoints[0], self.waypoints[1], self.goal, self.goal + [0, 0, .03]]
            tcp_goal = stages[self.phase]
            if 2 <= self.phase <= 6:
                target[-1] = -.32
            if self.phase == 2:
                tcp_goal = object_center
            if self.case == "walking_carry" and self.phase >= 4:
                tcp_goal = self.data.xpos[self.trunk_id] + [.065, 0, .05]
        target[10:14] = self._inverse_kinematics(tcp_goal)
        return self.normalize_targets(target)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.035, 0, .13]
        camera.distance = .65
        camera.azimuth = 135
        camera.elevation = -20
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
