"""Simulation-only ground access; commands actuators, never the free roots."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from microduck_arm_v1c.config import ROOT
from microduck_arm_v1c.locomotion import LegacyLegPolicyAdapter


@dataclass(frozen=True)
class NavigationProfile:
    warmup_s: float = 2.
    cruise_m_s: float = .18
    slow_m_s: float = .16
    gain_scale: float = 1.
    waypoint_y_m: float = .11


class BinNavigator:
    def __init__(self):
        self.policy = LegacyLegPolicyAdapter(
            ROOT / "microduck_local/policies/alpha_walking.onnx",
            action_gain=1.05, configuration_name="tennis-feedback-navigation",
        )
        self.started = None
        self.command = np.zeros(2)
        self.yaw_command = 0.
        self.gain = 1.05
        self.hold_s = 0.
        self.arrived = False
        self.progress_position = None
        self.progress_time = None
        self.recovery_started = None
        self.profile = NavigationProfile()

    def leg_targets(self, robot):
        if self.started is None:
            self.started = float(robot.data.time)
        position = robot.data.xpos[robot.trunk_id, :2]
        if self.progress_position is None or np.linalg.norm(self.command) < .1 or np.linalg.norm(position - self.progress_position) >= .005:
            self.progress_position = position.copy()
            self.progress_time = float(robot.data.time)
        if self.recovery_started is None and position[0] < .205 and robot.data.time - self.progress_time >= 2.5:
            self.recovery_started = float(robot.data.time)
        recovering = self.recovery_started is not None
        rotation = robot.data.xmat[robot.trunk_id].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        waypoint = position[1] < .10 and position[0] < .145
        route = np.array([.10, self.profile.waypoint_y_m] if waypoint else [.225 if recovering else .22, .12])
        error = route - position
        distance = float(np.linalg.norm(error))
        final_distance = float(np.linalg.norm(np.array([.22, .12]) - position))
        docking_region = (.220 <= position[0] <= .235 and abs(position[1] - .12) <= .035) if recovering else (.215 <= position[0] <= .233 and abs(position[1] - .12) <= .04)
        elapsed = robot.data.time - self.started
        if elapsed < self.profile.warmup_s or docking_region or self.arrived:
            desired = np.zeros(2)
            desired_gain = 1.05
        else:
            slow = not recovering and not waypoint and final_distance <= .09
            desired = (self.profile.slow_m_s if slow else self.profile.cruise_m_s) * error / max(distance, 1e-12)
            if waypoint:
                desired[1] = max(desired[1], .16)
                desired[0] = min(desired[0], .13)
            desired_gain = (1.12 if slow else 1.2) * self.profile.gain_scale
        desired_yaw = 0. if elapsed < self.profile.warmup_s or self.arrived or recovering and docking_region else np.clip(-.7 * yaw, -.45, .45)
        self.yaw_command += np.clip(desired_yaw - self.yaw_command, -.8 * robot.dt, .8 * robot.dt)
        delta = desired - self.command
        self.command += delta * min(1., .35 * robot.dt / max(np.linalg.norm(delta), 1e-12))
        self.gain += np.clip(desired_gain - self.gain, -.4 * robot.dt, .4 * robot.dt)
        self.policy.action_gain = self.gain
        command = np.clip(rotation[:2, :2].T @ self.command, -.22, .22)
        targets = self.policy.leg_targets(robot, twist_command=[*command, self.yaw_command])
        if recovering and not docking_region and not self.arrived:
            recovery_time = robot.data.time - self.recovery_started
            ramp = min(1., recovery_time / 2)
            targets[[1, 6]] += .06 * ramp * np.sin(2 * np.pi * recovery_time)
        root_dof = robot.model.jnt_dofadr[robot.root_joint_id]
        speed = np.linalg.norm(robot.data.qvel[root_dof:root_dof + 2])
        settled = docking_region and speed <= .035 and abs(yaw) <= .20 and rotation[2, 2] > np.cos(.25)
        self.hold_s = self.hold_s + robot.dt if settled else 0.
        self.arrived |= self.hold_s >= 1.
        return targets


def ground_arm_ik(robot, target, pitch, tcp_length):
    rotation = robot.data.xmat[robot.trunk_id].reshape(3, 3)
    shoulder = robot.data.xpos[robot.model.body("arm_yaw").id] + rotation @ np.asarray(robot.design["arm"]["shoulder_pivot_local_m"])
    relative = rotation.T @ (np.asarray(target) - shoulder)
    upper, forearm = robot.design["arm"]["link_lengths_m"][:2]
    radial = np.linalg.norm(relative[:2]) - tcp_length * np.cos(pitch)
    height = relative[2] + tcp_length * np.sin(pitch)
    cosine = (radial**2 + height**2 - upper**2 - forearm**2) / (2 * upper * forearm)
    elbow = np.arccos(np.clip(cosine, -1, 1))
    shoulder_pitch = np.arctan2(-height, radial) - np.arctan2(forearm * np.sin(elbow), upper + forearm * np.cos(elbow))
    joints = np.array([np.arctan2(relative[1], relative[0]), shoulder_pitch, elbow, pitch - shoulder_pitch - elbow])
    reachable = abs(cosine) <= 1 and np.all(joints >= robot.joint_limits[10:14, 0]) and np.all(joints <= robot.joint_limits[10:14, 1])
    return joints, bool(reachable)


def placement_ik(robot, target, initial, tcp_length, minimum_pitch=.4):
    candidates = []
    translation = np.zeros((3, robot.model.nv))
    rotation = translation.copy()
    scratch = robot.ik_data
    for pitch in np.linspace(minimum_pitch, 1.7, 27):
        joints, reachable = ground_arm_ik(robot, target, pitch, tcp_length)
        if not reachable:
            continue
        scratch.qpos[:] = robot.data.qpos
        scratch.qpos[robot.qpos_indices[10:14]] = joints
        mujoco.mj_forward(robot.model, scratch)
        collision = False
        for contact in scratch.contact:
            bodies = {int(robot.model.geom_bodyid[contact.geom1]), int(robot.model.geom_bodyid[contact.geom2])}
            names = {robot.model.geom(contact.geom1).name, robot.model.geom(contact.geom2).name}
            arm_contact = any(name and name.startswith("arm_") for name in names)
            if contact.dist < -1e-5 and arm_contact and robot.object_id not in bodies:
                collision = True
                break
        if collision:
            continue
        mujoco.mj_jacSite(robot.model, scratch, translation, rotation, robot.tcp_id)
        payload_torque = robot.model.body_mass[robot.object_id] * 9.81 * translation[2, robot.dof_indices[10:14]]
        torque = scratch.qfrc_bias[robot.dof_indices[10:14]] + payload_torque
        cost = np.max(np.abs(torque)) + .015 * np.linalg.norm(joints - initial)
        candidates.append((cost, joints))
    return min(candidates, key=lambda candidate: candidate[0])[1] if candidates else None


def crouch_offset(robot, drop):
    """Solve on scratch data; leave the live free base and feet untouched."""
    data = mujoco.MjData(robot.model)
    data.qpos[:] = robot.data.qpos
    mujoco.mj_forward(robot.model, data)
    bodies = [robot.model.body(name).id for name in ("ankle_left", "ankle_right")]
    positions = [data.xpos[body].copy() for body in bodies]
    rotations = [data.xmat[body].reshape(3, 3).copy() for body in bodies]
    original = data.qpos[robot.qpos_indices[:10]].copy()
    root_address = robot.model.jnt_qposadr[robot.root_joint_id]
    data.qpos[root_address + 2] -= drop
    translation = np.zeros((3, robot.model.nv))
    rotation_jacobian = translation.copy()
    for _ in range(40):
        mujoco.mj_forward(robot.model, data)
        residual, rows = [], []
        for body, position, rotation in zip(bodies, positions, rotations):
            mujoco.mj_jacBody(robot.model, data, translation, rotation_jacobian, body)
            current = data.xmat[body].reshape(3, 3)
            angular_error = sum(np.cross(current[:, axis], rotation[:, axis]) for axis in range(3)) / 2
            residual.extend([*(position - data.xpos[body]), *(.03 * angular_error)])
            rows.extend(np.vstack([translation[:, robot.dof_indices[:10]], .03 * rotation_jacobian[:, robot.dof_indices[:10]]]))
        jacobian = np.asarray(rows)
        update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(12) * 1e-7, np.asarray(residual))
        data.qpos[robot.qpos_indices[:10]] += np.clip(update, -.1, .1)
    return data.qpos[robot.qpos_indices[:10]] - original


def cached_release_clear(robot, joints):
    """Screen cached arm and jaw opening on scratch data, not the live robot."""
    scratch = mujoco.MjData(robot.model)
    scratch.qpos[:] = robot.data.qpos
    scratch.qpos[robot.qpos_indices[10:14]] = joints
    follower = robot.model.jnt_qposadr[robot.model.joint("arm_gripper_follower").id]
    jaw = robot.data.qpos[robot.qpos_indices[-1]]
    for angle in np.linspace(jaw, 0., 9):
        scratch.qpos[robot.qpos_indices[-1]] = angle
        scratch.qpos[follower] = -angle
        mujoco.mj_forward(robot.model, scratch)
        for contact in scratch.contact:
            names = {robot.model.geom(contact.geom1).name, robot.model.geom(contact.geom2).name}
            if contact.dist >= 0 or not any(name and name.startswith("arm_") for name in names):
                continue
            if "object_geom" in names and names & {"arm_pad_left", "arm_pad_right"}:
                continue
            return False
    return True


class GroundReturnController:
    """Feedback posture and event-conditioned arm coordination, not new PPO."""

    def __init__(self, environment):
        self.environment = environment
        self.reset()

    def reset(self):
        self.offset = None
        self.lift_started = None
        self.lift_joints = None
        self.rise_started = None
        self.navigator = None
        self.navigation_forecasting = False
        self.navigation_prediction = None
        self.navigation_qualified = False
        self.placement_target = None
        self.placement_joints = None
        self.placement_crouch = None
        self.lower_started = None
        self.release_targets = None
        self.release_tcp = None
        self.release_started = None
        self.release_pitch = None
        self.release_jaw = None
        self.release_axis = None
        self.release_settled_s = 0.
        self.release_opening = False
        self.safe_release_joints = None
        self.release_blocked = False
        self.cached_release_screen_passed = False
        self.release_strategy = None
        self.release_prediction = []
        self.docking_forecasting = False
        self.docking_qualified = False
        self.docking_prediction = []
        self.next_docking_screen_s = 25.
        self.placement_drop_m = .025
        self.retreat_origin = None
        self.stage = "settle"
        self.ik_reachable = False

    def targets(self, control):
        env = self.environment
        robot = env.robot
        if control == "teacher" and getattr(env, "release_mode", "supported") == "half_height" and getattr(env, "plan_navigation", False) and self.navigator is not None and not self.navigation_forecasting and self.navigation_prediction is None:
            from .tennis_navigation import select_navigation_profile

            profile, self.navigation_prediction = select_navigation_profile(env)
            self.navigation_qualified = profile is not None
            if profile is not None:
                self.navigator.profile = profile
            else:
                self.navigator.arrived = True
        if getattr(env, "plan_navigation", False) and self.navigation_prediction is not None and not self.navigation_qualified and not self.navigation_forecasting:
            targets = robot.command_targets.copy()
            targets[:10] = self.navigator.leg_targets(robot)
            self.stage = "navigation_no_verified_route"
            return targets
        if self.navigator is not None and not self.docking_qualified and not self.docking_forecasting:
            position = robot.data.xpos[robot.trunk_id, :2]
            root = robot.model.jnt_dofadr[robot.root_joint_id]
            close = .210 <= position[0] <= .240 and abs(position[1] - .12) <= .04
            stable = np.linalg.norm(robot.data.qvel[root:root + 2]) < .01
            if close and stable and robot.data.time >= self.next_docking_screen_s:
                from .tennis_release import screen_docking_pose

                self.next_docking_screen_s = float(robot.data.time) + 5.
                prediction = screen_docking_pose(env)
                self.docking_prediction.append(prediction)
                if prediction["success"]:
                    self.navigator.arrived = True
                    self.docking_qualified = True
                    self.placement_drop_m = prediction["placement_drop_m"]
        if env.monitor.phase == "release" and self.release_strategy is None:
            from .tennis_release import select_release_strategy

            strategy, self.release_prediction = select_release_strategy(env)
            self.release_strategy = strategy or "hold_unverified"
        targets = robot.home.copy()
        targets[:10] = robot.leg_policy.leg_targets(robot)
        if control == "hold" or robot.data.time < 2:
            return targets
        if self.offset is None:
            self.offset = crouch_offset(robot, .020)
        crouch = np.clip((robot.data.time - 2) / 2, 0, 1)
        targets[:10] += crouch * self.offset
        targets[[2, 4, 7, 9]] += crouch * -.2 * np.array([-1, 1, 1, -1])
        self.stage = "crouch"
        if robot.data.time <= 4:
            return targets
        phase = env.monitor.phase
        self.stage = "ground_approach"
        ball = robot.data.xpos[robot.object_id].copy()
        targets[10:14], self.ik_reachable = ground_arm_ik(robot, ball, 1., env.spec["wide_gripper_candidate"]["tcp_offset_m"])
        if phase not in {"approach", "grasp"}:
            if self.lift_started is None:
                self.lift_started = float(robot.data.time)
                self.lift_joints = robot.data.qpos[robot.qpos_indices[10:14]].copy()
            stow = np.array([self.lift_joints[0], -1.1, 1.7, .8])
            duration = float(np.max(np.abs(stow - self.lift_joints)) / .25)
            progress = np.clip((robot.data.time - self.lift_started) / duration, 0, 1)
            targets[10:14] = self.lift_joints + progress * (stow - self.lift_joints)
            self.stage = "folded_lift" if progress < 1 else "loaded_hold"
            if progress >= 1:
                if self.rise_started is None:
                    self.rise_started = float(robot.data.time)
                rise = np.clip((robot.data.time - self.rise_started) / 4, 0, 1)
                targets[:10] -= rise * self.offset
                targets[[2, 4, 7, 9]] -= rise * -.2 * np.array([-1, 1, 1, -1])
                raised = np.array([stow[0], -1.65, 1.78, 1.2])
                targets[10:14] = stow + rise * (raised - stow)
                self.stage = "loaded_rise" if rise < 1 else "transport_ready"
                if rise >= 1:
                    if self.navigator is None:
                        self.navigator = BinNavigator()
                    targets[:10] = self.navigator.leg_targets(robot)
                    self.stage = "arrived" if self.navigator.arrived else "navigate"
                    if self.navigator.arrived and self.docking_qualified:
                        targets = self.placement_targets(targets)
        compensation = robot.design["arm"]["gravity_compensation"]["maximum_position_offset_rad"]
        targets[10:14] += np.clip(robot.data.qfrc_bias[robot.dof_indices[10:14]] / robot.design["arm"]["simulation_kp"], -compensation, compensation)
        if phase not in {"approach", "release", "retreat", "success"} and control != "open_jaw":
            targets[-1] = -.16
        return targets

    def placement_targets(self, targets):
        env = self.environment
        robot = env.robot
        if env.monitor.phase == "release" and (self.release_strategy in {"supported_hold", "hold_unverified"} or str(self.release_strategy).startswith("half_height_")):
            from .tennis_release import supported_release_targets

            half_height = getattr(env, "release_mode", "supported") == "half_height"
            speed = {"half_height_hold": .08, "half_height_slow": .025, "half_height_fast": .15}.get(self.release_strategy, .025)
            return supported_release_targets(self, targets, allow_opening=self.release_strategy != "hold_unverified", require_support=not half_height, jaw_speed=speed)
        if env.monitor.phase == "release":
            if self.release_targets is None:
                self.release_targets = robot.command_targets.copy()
                self.release_tcp = robot.data.site_xpos[robot.tcp_id].copy()
                self.release_pitch = float(sum(robot.data.qpos[robot.qpos_indices[11:14]]))
                self.release_jaw = float(robot.data.qpos[robot.qpos_indices[-1]])
                self.release_axis = robot.data.site_xmat[robot.tcp_id].reshape(3, 3)[:, 0].copy()
                self.release_started = float(robot.data.time)
                self.safe_release_joints = robot.data.qpos[robot.qpos_indices[10:14]].copy()
            self.stage = "release"
            legs = targets[:10].copy() + self.placement_crouch
            legs[[2, 4, 7, 9]] += -.2 * np.array([-1, 1, 1, -1])
            targets = self.release_targets.copy()
            targets[:10] = legs
            elapsed = robot.data.time - self.release_started
            tcp_target = self.release_tcp - np.array([0., 0., min(.004, .001 + .01 * elapsed)])
            tcp_target[0] += min(.010, .004 * max(0., elapsed - 1.))
            opening = max(0., robot.data.qpos[robot.qpos_indices[-1]] - self.release_jaw)
            tcp_target += self.release_axis * (env.radius - .012) * opening
            joints, reachable = ground_arm_ik(robot, tcp_target, self.release_pitch, env.spec["wide_gripper_candidate"]["tcp_offset_m"])
            safe = reachable and joints[1] < .62
            if safe:
                self.safe_release_joints = joints.copy()
            self.release_blocked |= not safe
            targets[10:14] = self.safe_release_joints
            object_dof = robot.model.jnt_dofadr[robot.model.joint("task_object_free").id]
            speed = np.linalg.norm(robot.data.qvel[object_dof:object_dof + 3])
            self.release_settled_s = self.release_settled_s + robot.dt if speed < .001 else 0.
            opening_ready = self.release_settled_s >= .3 and (elapsed >= 4.2 or self.release_blocked and elapsed >= 1.)
            if self.release_blocked:
                self.stage = "release_cached_pose"
                if opening_ready and not self.release_opening:
                    sample = env.sample()
                    self.cached_release_screen_passed = bool(
                        sample["bottom_supported"] and sample["ball_inside_bin"]
                        and not sample["forbidden_contacts"]
                        and cached_release_clear(robot, self.safe_release_joints)
                    )
                opening_ready &= self.cached_release_screen_passed
            self.release_opening |= opening_ready
            targets[-1] = 0. if self.release_opening else self.release_targets[-1]
            return targets
        if self.placement_target is None:
            self.placement_target = robot.data.site_xpos[robot.tcp_id].copy()
            self.placement_joints = robot.data.qpos[robot.qpos_indices[10:14]].copy()
        container = env.spec["bin"]
        lateral_margin = container["inside_size_xy_m"][1] / 2 - env.radius - .008
        lateral_goal = np.clip(robot.data.xpos[robot.trunk_id, 1], container["center_xy_m"][1] - lateral_margin, container["center_xy_m"][1] + lateral_margin)
        goal = np.array([container["center_xy_m"][0] - container["inside_size_xy_m"][0] / 2 + env.radius + .004,
                         lateral_goal, container["bottom_thickness_m"] + env.radius - .001])
        if getattr(env, "release_mode", "supported") == "half_height":
            goal[0] += .006
            goal[2] = container["bottom_thickness_m"] + .5 * container["wall_height_m"] + .003
        if env.monitor.phase == "carry":
            goal[2] = container["bottom_thickness_m"] + env.radius - .001 + container["wall_height_m"] + .032
            self.stage = "over_bin"
        elif env.monitor.phase == "lower":
            if self.lower_started is None:
                self.lower_started = float(robot.data.time)
                self.placement_crouch = crouch_offset(robot, self.placement_drop_m)
            progress = np.clip((robot.data.time - self.lower_started) / 2, 0, 1)
            targets[:10] += progress * self.placement_crouch
            targets[[2, 4, 7, 9]] += progress * -.2 * np.array([-1, 1, 1, -1])
            self.stage = "lower_into_bin"
        elif env.monitor.phase in {"release", "retreat", "success"}:
            targets[:10] += self.placement_crouch
            targets[[2, 4, 7, 9]] += -.2 * np.array([-1, 1, 1, -1])
            if self.retreat_origin is None:
                self.retreat_origin = robot.data.site_xpos[robot.tcp_id].copy()
                self.placement_target = self.retreat_origin.copy()
            goal = self.retreat_origin + np.array([0., 0., .08])
            self.stage = env.monitor.phase
        delta = goal - self.placement_target
        speed = .025 if env.monitor.phase == "retreat" else .008
        candidate = self.placement_target + delta * min(1., speed * robot.dt / max(np.linalg.norm(delta), 1e-12))
        minimum_pitch = .4 if env.monitor.phase == "carry" else float(np.clip(1.4 - (candidate[2] - .10) * 10, .4, 1.4))
        if env.monitor.phase == "retreat":
            minimum_pitch = 1.4
        joints = placement_ik(robot, candidate, self.placement_joints, env.spec["wide_gripper_candidate"]["tcp_offset_m"], minimum_pitch)
        if joints is not None:
            self.placement_target = candidate
            self.placement_joints = joints
        targets[10:14] = self.placement_joints
        current_arm = robot.data.qpos[robot.qpos_indices[10:14]]
        if env.monitor.phase != "carry":
            targets[13] = sum(self.placement_joints[1:]) - sum(current_arm[1:3])
        return targets
