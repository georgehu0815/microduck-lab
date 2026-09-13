from __future__ import annotations

import mujoco
import numpy as np

from microduck_arm_v1.env import IntegratedArmEnv as BaselineEnv
from .config import DESIGN_ID, ROOT, SPEC_PATH, load_design, sha256
from .locomotion import LegacyLegPolicyAdapter
from .gait import IndependentWalkingController
from .model import compile_model

ENV_FROZEN = True


class IntegratedArmEnv(BaselineEnv):
    def __init__(self, case="reach", mode="free", candidate="A", max_steps=2500, payload_kg=.01, render_mode=None):
        super().__init__(case=case, mode=mode, max_steps=max_steps, payload_kg=payload_kg, render_mode=render_mode)
        self.design = load_design(candidate)
        self.candidate = candidate
        self.model = compile_model(mode, payload_kg, candidate)
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.contract = DESIGN_ID + ":obs64-action15:v1"
        self.leg_policy = LegacyLegPolicyAdapter.screened_stand(ROOT / "microduck_local/policies") if mode == "free" else None
        self.walk_policy = IndependentWalkingController(world_frame_commands=case == "walking_carry") if mode == "free" and "walking" in case else None
        self.leg_cache_step = -1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        surface_top = self.design.get("simulation_scene", {}).get("surface_top_m", .145)
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = [.115, -.018, surface_top + .006]
        self.data.qpos[self.object_qpos:self.object_qpos + 2] += self.np_random.uniform(-.004, .004, 2)
        mujoco.mj_forward(self.model, self.data)
        self.object_start = self.data.xpos[self.object_id].copy()
        if self.case != "reach":
            self.goal = np.array([.125, .024, surface_top + .006])
        self.waypoints = [np.array([.112, -.01, surface_top + .045]), np.array([.125, .024, surface_top + .045])]
        if self.mode == "free":
            for waypoint in self.waypoints:
                waypoint[2] = surface_top + self.design["whole_body_teacher"]["free_transfer_clearance_m"]
        self.model.site_pos[self.model.site("goal").id] = self.goal
        self.phase_started_at = 0.
        self.phase_ready_ticks = 0
        self.phase_events = []
        self.release_ready = False
        self.release_jaw_at_entry = 0.
        self.carry_joint_targets = None
        self.leg_cache_step = -1
        for policy in (self.leg_policy, self.walk_policy):
            if policy:
                policy.reset()
        self.last_diagnostics = self.diagnostics()
        return self._observation(), self._info(False, None)

    def _info(self, success, failure):
        result = super()._info(success, failure)
        result.update(design_id=DESIGN_ID, spec_sha256=sha256(SPEC_PATH), candidate=self.candidate,
                      scene="raised_table_screening_not_original_low_table", controller="event_conditioned_IK_FSM",
                      phase_events=getattr(self, "phase_events", []))
        result["leg_controller"] = self.leg_policy.provenance if self.leg_policy else "fixture"
        result["walking_controller"] = self.walk_policy.provenance if self.walk_policy else None
        if self.walk_policy:
            result["walking_route"] = {
                "phase": self.walk_policy.active_phase.value,
                "command": self.walk_policy.last_command.tolist(),
                "mount_table_clearance_m": self.walk_policy.last_clearance_distance_m,
            }
        result["whole_body_teacher"] = self.design["whole_body_teacher"]
        result["noslip_iterations"] = self.model.opt.noslip_iterations
        return result

    def phase_target(self):
        target = [self.object_start + [0, 0, .035], self.object_start,
                self.object_start, self.object_start + [0, 0, .035],
                self.waypoints[0], self.waypoints[1], self.goal, self.goal][self.phase].copy()
        if self.mode == "free":
            tuning = self.design["whole_body_teacher"]
            if self.phase in (0, 3):
                target[2] = self.object_start[2] + tuning["free_approach_lift_m"]
            elif self.phase in (1, 2):
                target[2] += tuning["free_pickup_tcp_offset_m"]
        return target

    def advance_phase(self):
        if self.case in ("reach", "stowed_arm_walking") or self.phase == 7:
            return
        diagnostics = self.diagnostics()
        distance = np.linalg.norm(self.data.site_xpos[self.tcp_id] - self.phase_target())
        ready = distance < .006
        if self.phase == 2:
            ready = diagnostics["bilateral_grasp"]
        elif self.phase == 3:
            ready = self.ever_lifted
            if self.case == "walking_carry":
                transport = np.asarray(self.design["whole_body_teacher"]["transport_arm_targets_rad"])
                ready &= diagnostics["bilateral_grasp"] and np.max(np.abs(self.data.qpos[self.qpos_indices[10:14]] - transport)) < .055
                ready &= np.max(np.abs(self.data.qvel[self.dof_indices[10:14]])) < .08
                ready &= np.linalg.norm(self.data.sensor("imu_ang_vel").data) < .10
                ready &= np.max(np.abs(self.data.qvel[self.dof_indices[:10]])) < .15
        elif self.phase in (4, 5):
            ready = diagnostics["bilateral_grasp"] and np.linalg.norm(self.data.xpos[self.object_id] - self.phase_target()) < .006
        elif self.phase == 6:
            ready = diagnostics["object_goal_error_m"] < .008 and diagnostics["object_supported"]
        if self.case == "walking_carry" and self.phase >= 4:
            ready = False
        self.phase_ready_ticks = self.phase_ready_ticks + 1 if ready else 0
        dwell = 1.0 if self.case == "walking_carry" and self.phase == 3 else .2
        if self.phase_ready_ticks * self.dt >= dwell:
            if self.case == "walking_carry" and self.phase == 3:
                self.carry_joint_targets = np.asarray(self.design["whole_body_teacher"]["transport_arm_targets_rad"])
            if self.phase == 6:
                self.release_tcp = self.data.site_xpos[self.tcp_id].copy()
                self.release_jaw_at_entry = float(self.data.qpos[self.qpos_indices[-1]])
                self.release_ready = True
            self.phase_events.append({"time_s": float(self.data.time), "from": self.phase, "to": self.phase + 1})
            self.phase += 1
            self.phase_started_at = float(self.data.time)
            self.phase_ready_ticks = 0

    def teacher_action(self):
        target = self.home.copy()
        if self.leg_policy:
            if self.leg_cache_step != self.steps:
                walking = self.walk_policy and (self.case == "stowed_arm_walking" or self.phase >= 4)
                policy = self.walk_policy if walking else self.leg_policy
                self.leg_cache = policy.leg_targets(self)
                self.leg_cache_step = self.steps
            target[:10] = self.leg_cache
            if "walking" not in self.case or self.phase < 4 and self.case == "walking_carry":
                tuning = self.design["whole_body_teacher"]
                trim = tuning["reach_pitch_trim_rad" if self.case == "reach" else "manipulation_pitch_trim_rad"]
                target[tuning["pitch_trim_indices"]] += trim * np.asarray(tuning["pitch_trim_signs"])
        if self.case == "stowed_arm_walking":
            return self.normalize_targets(target)
        tcp_goal = self.goal if self.case == "reach" else self.phase_target()
        if self.case != "reach" and 2 <= self.phase <= 6:
            target[-1] = -.32
        if self.case != "reach" and 3 <= self.phase <= 6 and self.last_diagnostics["bilateral_grasp"]:
            tcp_goal = tcp_goal + self.data.site_xpos[self.tcp_id] - self.data.xpos[self.object_id]
        if self.case == "walking_carry" and self.phase >= 4 and self.carry_anchor is not None:
            rotation = self.data.xmat[self.trunk_id].reshape(3, 3)
            object_goal = self.data.xpos[self.trunk_id] + rotation @ self.carry_anchor
            tcp_goal = object_goal + self.data.site_xpos[self.tcp_id] - self.data.xpos[self.object_id]
        if self.case != "reach" and self.phase == 7:
            tcp_goal = self.release_tcp.copy()
        target[10:14] = self._inverse_kinematics(tcp_goal)
        offset_limit = self.design["arm"]["gravity_compensation"]["maximum_position_offset_rad"]
        target[10:14] += np.clip(self.ik_data.qfrc_bias[self.dof_indices[10:14]] / self.design["arm"]["simulation_kp"], -offset_limit, offset_limit)
        if self.case == "walking_carry" and self.phase == 3 and self.ever_lifted:
            target[10:14] = self.design["whole_body_teacher"]["transport_arm_targets_rad"]
        if self.case == "walking_carry" and self.phase >= 4 and self.carry_joint_targets is not None:
            target[10:14] = self.carry_joint_targets
        return self.normalize_targets(target)

    def step(self, action):
        if self._done:
            raise RuntimeError("reset() required before stepping a finished episode")
        previously_grasped = self.last_diagnostics["bilateral_grasp"]
        target = self.denormalize_action(action)
        speeds = np.array([self.design["control"]["leg_joint_slew_limit_rad_s"]] * 10 + [self.design["control"]["arm_joint_slew_limit_rad_s"]] * 5)
        delta = speeds * self.dt
        self.command_targets += np.clip(target - self.command_targets, -delta, delta)
        self.data.ctrl[:] = self.command_targets
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            _, _, forbidden, _ = self._contact_state()
            self.forbidden_contacts.update(forbidden)
        self.steps += 1
        self.advance_phase()
        self.previous_action = np.asarray(action, dtype=np.float32).copy()
        self.last_diagnostics = self.diagnostics()
        diagnostics = self.last_diagnostics
        self.ever_grasped |= diagnostics["bilateral_grasp"]
        object_velocity_index = self.model.jnt_dofadr[self.model.joint("task_object_free").id]
        placement_velocity = float(np.linalg.norm(self.data.qvel[object_velocity_index:object_velocity_index + 3]))
        physically_opening = self.data.qpos[self.qpos_indices[-1]] > self.release_jaw_at_entry + .003
        release_window = self.case in ("pick_place", "obstacle_relocation", "stationary_waypoint_carry") and self.phase == 7 and self.release_ready and physically_opening and self.ever_lifted and diagnostics["object_goal_error_m"] < .010 and diagnostics["object_supported"] and target[-1] >= -.05
        self.release_authorized = bool(release_window and (self.release_authorized or (previously_grasped and placement_velocity < .030)))
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
        if not terminated and self.case not in ("reach", "stowed_arm_walking", "walking_carry") and self.data.time - self.phase_started_at > 12.0:
            failure = "phase_timeout"
            truncated = True
        self._done = terminated or truncated
        return self._observation(), float(reward), terminated, truncated, self._info(success, failure)

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        scene_center = np.array([.10, 0., .13])
        trunk = self.data.xpos[self.trunk_id]
        camera.lookat[:] = (scene_center + trunk) / 2
        camera.lookat[2] = .13
        camera.distance = max(.70, .60 + float(np.linalg.norm(trunk[:2] - scene_center[:2])))
        camera.azimuth = 135
        camera.elevation = -20
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()
