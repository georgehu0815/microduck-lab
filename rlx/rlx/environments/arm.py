from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


CONTROL_DT = 0.02
PHYSICS_DT = 0.002
LINK_LENGTHS = (0.065, 0.055, 0.030)
SHOULDER_HEIGHT = 0.080
CASES = (
    "arm-reach-v1", "arm-pick-place-v1", "arm-relocate-v1",
    "arm-carry-v1", "arms-handover-v1", "arms-co-carry-v1",
)
JOINT_LIMITS = np.array([[-1.745, 1.745], [-1.047, 1.396], [-1.745, 1.745],
                         [-1.571, 1.571], [-1.571, 1.571]])
VELOCITY_LIMITS = np.array([0.4] * 5 + [0.01])
CONTRACTS = {1: ("md-arm-table-v1", 66, 6), 2: ("md-dualarm-table-v1", 116, 12)}


@dataclass(frozen=True)
class TaskSpec:
    horizon: float
    success_rate: float
    mass: float


SPECS = dict(zip(CASES, (
    TaskSpec(12, .95, .02), TaskSpec(20, .90, .02),
    TaskSpec(25, .90, .02), TaskSpec(30, .90, .02),
    TaskSpec(30, .85, .005), TaskSpec(35, .85, .017),
)))


def vector_text(values):
    return " ".join(f"{float(value):.9g}" for value in values)


def arm_xml(prefix, base):
    return f"""
    <body name="{prefix}_base" pos="{vector_text(base)}">
      <geom name="{prefix}_mount" type="box" pos="0 0 .018" size=".023 .023 .018" mass=".08" rgba=".18 .22 .25 1"/>
      <body name="{prefix}_yaw" pos="0 0 {SHOULDER_HEIGHT}">
        <joint name="{prefix}_base_yaw" axis="0 0 1" range="-1.745 1.745"/>
        <geom type="sphere" size=".012" mass=".018"/>
        <body name="{prefix}_shoulder">
          <joint name="{prefix}_shoulder_pitch" axis="0 1 0" range="-1.047 1.396"/>
          <geom name="{prefix}_upper" type="capsule" fromto="0 0 0 .065 0 0" size=".007" mass=".038"/>
          <body name="{prefix}_elbow" pos=".065 0 0">
            <joint name="{prefix}_elbow_pitch" axis="0 1 0" range="-1.745 1.745"/>
            <geom name="{prefix}_forearm" type="capsule" fromto="0 0 0 .055 0 0" size=".006" mass=".033"/>
            <body name="{prefix}_wrist" pos=".055 0 0">
              <joint name="{prefix}_wrist_pitch" axis="0 1 0" range="-1.571 1.571"/>
              <geom type="sphere" size=".007" mass=".018"/>
              <body name="{prefix}_hand">
                <joint name="{prefix}_wrist_roll" axis="1 0 0" range="-1.571 1.571"/>
                <geom name="{prefix}_palm" type="box" pos=".003 0 0" size=".007 .019 .009" mass=".026" rgba=".18 .25 .3 1"/>
                <site name="{prefix}_tcp" pos=".030 0 0" size=".002" rgba="0 1 .5 1"/>
                <body name="{prefix}_finger_left" pos="0 .002 0">
                  <joint name="{prefix}_grip_left" type="slide" axis="0 1 0" range="0 .015" damping=".8" armature=".0001"/>
                  <geom name="{prefix}_pad_left" type="box" pos=".025 0 0" size=".012 .002 .008" mass=".004" condim="4" friction="1.2 .005 .001" rgba=".1 .8 .5 1"/>
                </body>
                <body name="{prefix}_finger_right" pos="0 -.002 0">
                  <joint name="{prefix}_grip_right" type="slide" axis="0 -1 0" range="0 .015" damping=".8" armature=".0001"/>
                  <geom name="{prefix}_pad_right" type="box" pos=".025 0 0" size=".012 .002 .008" mass=".004" condim="4" friction="1.2 .005 .001" rgba=".1 .8 .5 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>"""


def make_xml(case_id, timestep=PHYSICS_DT):
    if case_id not in CASES:
        raise ValueError(f"Unknown arm case: {case_id}")
    dual = case_id.startswith("arms-")
    half_span = .121 if case_id == "arms-co-carry-v1" else .075
    bases = [(0, -half_span, 0), (0, half_span, 0)] if dual else [(0, 0, 0)]
    prefixes = ["left", "right"] if dual else ["left"]
    actuators, equalities = [], []
    for prefix in prefixes:
        for joint in ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"):
            actuators.append(f'<position name="{prefix}_{joint}_servo" joint="{prefix}_{joint}" kp="8" kv=".15" forcerange="-.16 .16"/>')
        actuators.append(f'<position name="{prefix}_grip_servo" joint="{prefix}_grip_left" kp="350" kv="2" ctrlrange="0 .015" forcerange="-1 1"/>')
        equalities.append(f'<joint joint1="{prefix}_grip_right" joint2="{prefix}_grip_left" polycoef="0 1 0 0 0" solref=".004 1"/>')
    if case_id == "arms-co-carry-v1":
        object_geoms = '''<geom name="object_tray" type="box" size=".025 .047 .003" mass=".008" rgba=".4 .6 .9 1"/>
        <geom name="object_handle_left" type="box" pos="0 -.04 .006" size=".007 .007 .006" mass=".002"/>
        <geom name="object_handle_right" type="box" pos="0 .04 .006" size=".007 .007 .006" mass=".002"/>'''
        payload = '<body name="payload" pos=".05 0 .014"><freejoint name="payload_free"/><geom name="payload_geom" type="box" size=".005 .005 .005" mass=".005" rgba=".95 .4 .2 1"/></body>'
    elif case_id == "arms-handover-v1":
        object_geoms = '<geom name="object_bar" type="box" size=".007 .025 .008" mass=".005" rgba="1 .65 .2 1"/>'
        payload = ""
    else:
        object_geoms = '<geom name="object_block" type="box" size=".009 .009 .009" mass=".02" rgba="1 .65 .2 1"/>'
        payload = ""
    return f'''<mujoco model="md-arm-t1-{case_id}">
      <compiler angle="radian"/>
      <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" iterations="60"/>
      <visual><global offwidth="1280" offheight="720"/><quality shadowsize="1024"/></visual>
      <default><joint damping=".03" armature=".00003"/>
        <geom friction="1 .005 .0001" solref=".006 1" solimp=".95 .99 .001" rgba=".65 .75 .8 1"/>
        <position ctrllimited="false" forcelimited="true"/></default>
      <worldbody>
        <light pos="0 -.3 .8" dir="0 .3 -.8"/>
        <camera name="overview" pos=".36 -.38 .30" xyaxes=".72 .69 0 -.32 .34 .88"/>
        <camera name="top" pos=".07 0 .48" xyaxes="1 0 0 0 1 0"/>
        <geom name="table" type="box" pos=".06 0 -.012" size=".22 .20 .012" rgba=".12 .16 .20 1"/>
        {''.join(arm_xml(prefix, base) for prefix, base in zip(prefixes, bases))}
        <body name="object" pos=".085 -.02 .009"><freejoint name="object_free"/>{object_geoms}</body>
        {payload}
        <body name="obstacle" pos=".09 0 -.05"><geom name="obstacle_geom" type="box" size=".018 .006 .01" rgba=".8 .25 .3 1"/></body>
        <site name="goal" type="box" pos=".085 .025 .001" size=".016 .016 .001" rgba=".1 .85 .5 .35"/>
      </worldbody>
      <contact>{''.join(f'<exclude body1="{prefix}_elbow" body2="{prefix}_hand"/>' for prefix in prefixes)}</contact>
      <equality>{''.join(equalities)}</equality>
      <actuator>{''.join(actuators)}</actuator>
    </mujoco>'''


def inverse_kinematics(target, base, pinch_axis="y"):
    relative = np.asarray(target) - np.asarray(base)
    radius = float(np.hypot(relative[0], relative[1]))
    yaw = math.atan2(relative[1], relative[0])
    vertical_down = SHOULDER_HEIGHT - (relative[2] + LINK_LENGTHS[2])
    cosine = (radius**2 + vertical_down**2 - LINK_LENGTHS[0]**2 - LINK_LENGTHS[1]**2) / (2 * LINK_LENGTHS[0] * LINK_LENGTHS[1])
    if abs(cosine) > 1.0001:
        raise ValueError(f"Unreachable TCP target {target}")
    elbow = math.acos(float(np.clip(cosine, -1, 1)))
    shoulder = math.atan2(vertical_down, radius) - math.atan2(LINK_LENGTHS[1] * math.sin(elbow), LINK_LENGTHS[0] + LINK_LENGTHS[1] * math.cos(elbow))
    wrist = math.pi / 2 - shoulder - elbow
    roll = yaw if pinch_axis == "y" else yaw - math.copysign(math.pi / 2, yaw or 1)
    result = np.array([yaw, shoulder, elbow, wrist, roll])
    if np.any(result < JOINT_LIMITS[:, 0] - .002) or np.any(result > JOINT_LIMITS[:, 1] + .002):
        raise ValueError(f"TCP target violates joint envelope: {target}: {result}")
    return result


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, case_id="arm-pick-place-v1", timestep=PHYSICS_DT, render_mode=None):
        if case_id not in CASES or not np.isclose(CONTROL_DT / timestep, round(CONTROL_DT / timestep)):
            raise ValueError("Invalid case or physics timestep")
        self.case_id = case_id
        self.arm_count = 2 if case_id.startswith("arms-") else 1
        self.contract, self.observation_dim, self.action_dim = CONTRACTS[self.arm_count]
        self.prefixes = ["left", "right"][:self.arm_count]
        half_span = .121 if case_id == "arms-co-carry-v1" else .075
        self.bases = [np.array([0, -half_span, 0]), np.array([0, half_span, 0])] if self.arm_count == 2 else [np.zeros(3)]
        self.xml = make_xml(case_id, timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.nominal_mass = self.model.body_mass.copy()
        self.nominal_inertia = self.model.body_inertia.copy()
        self.nominal_friction = self.model.geom_friction.copy()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Box(-1, 1, (self.action_dim,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.observation_dim,), np.float32)
        self.joint_ids, self.qpos_indices, self.dof_indices, self.tcp_ids = [], [], [], []
        for prefix in self.prefixes:
            joints = [self.model.joint(f"{prefix}_{name}").id for name in ("base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "grip_left")]
            self.joint_ids.append(joints)
            self.qpos_indices.append(self.model.jnt_qposadr[joints])
            self.dof_indices.append(self.model.jnt_dofadr[joints])
            self.tcp_ids.append(self.model.site(f"{prefix}_tcp").id)
        self.object_id = self.model.body("object").id
        self.object_qpos = self.model.jnt_qposadr[self.model.joint("object_free").id]
        self.object_geoms = set(np.flatnonzero(self.model.geom_bodyid == self.object_id).tolist())
        self.pad_ids = [[self.model.geom(f"{prefix}_pad_{side}").id for side in ("left", "right")] for prefix in self.prefixes]
        self.geom_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "" for geom in range(self.model.ngeom)]
        self.geom_owners = [next((index for index, prefix in enumerate(self.prefixes) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or "").startswith(prefix + "_")), -1) for body in self.model.geom_bodyid]
        self.table_id = self.model.geom("table").id
        self.obstacle_id = self.model.geom("obstacle_geom").id
        self.grasp_geoms = [{self.model.geom(f"object_handle_{prefix}").id} for prefix in self.prefixes] if case_id == "arms-co-carry-v1" else [self.object_geoms] * self.arm_count
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        mass_scale = float(options.get("mass_scale", 1))
        friction_scale = float(options.get("friction_scale", 1))
        if not .5 <= mass_scale <= 1.5 or not .5 <= friction_scale <= 1.5:
            raise ValueError("Mass/friction perturbation outside declared simulation envelope")
        self.model.body_mass[:] = self.nominal_mass
        self.model.body_inertia[:] = self.nominal_inertia
        self.model.geom_friction[:] = self.nominal_friction
        self.model.body_mass[self.object_id] *= mass_scale
        self.model.body_inertia[self.object_id] *= mass_scale
        for geom_id in self.object_geoms:
            self.model.geom_friction[geom_id, 0] *= friction_scale
        self.reset_options = options.copy()
        self.seed_value = seed
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.controller = "manual"
        self.terminated = False
        self.truncated = False
        self.prev_action = np.zeros(self.action_dim)
        self.joint_targets = np.zeros(self.action_dim)
        self.phase = 0
        self.phase_steps = 0
        self.lift_run = 0
        self.max_lift_run = 0
        self.settle_run = 0
        self.max_settle_run = 0
        self.reach_run = 0
        self.handover_overlap = 0
        self.handover_verified = False
        self.receiver_hold = 0
        self.max_receiver_hold = 0
        self.invalid_contacts = 0
        self.arm_collisions = 0
        self.drop_count = 0
        self.ever_lifted = False
        self.ever_airborne = False
        self.peak_height = 0.0
        self.peak_force = 0.0
        self.max_payload_slip = 0.0
        self.max_transport_tilt = 0.0
        self.dual_grasp_steps = 0
        self.transport_steps = 0
        self.path_length = 0.0
        self.waypoint_index = 0
        self.completed = False
        self.reward_components = {}
        self.controlled_release = False
        self.load_share_sum = np.zeros(self.arm_count)
        self.load_share_samples = 0
        self.contact_support = np.zeros(self.arm_count)
        self.object_supported = False
        self.evaluator_stage = "initial"
        self.evaluator_events = []
        self.self_collisions = 0
        self.grip_loss_s = 0.0
        self.peak_internal_force = 0.0
        self.support_sum = 0.0
        self.support_valid_samples = 0
        self.authored_action_calls = 0
        perturbation = float(options.get("position_noise", .001))
        jitter = self.np_random.uniform(-perturbation, perturbation, 2)
        if self.case_id == "arms-co-carry-v1":
            self.start = np.array([.045, 0, .0031])
            self.goal = np.array([-.007, 0, .0031])
            self.grasp_offsets = [np.array([0, -.04, .006]), np.array([0, .04, .006])]
        elif self.case_id == "arms-handover-v1":
            self.start = np.array([.075, -.018, .0081])
            self.goal = np.array([.075, .030, .0081])
            self.grasp_offsets = [np.array([0, -.012, .003]), np.array([0, .012, .003])]
        else:
            self.start = np.array([.085, -.022, .0091])
            self.goal = np.array([.085, .025, .0091])
            self.grasp_offsets = [np.zeros(3)]
        self.start[:2] += jitter
        self.goal[:2] += jitter
        if self.case_id == "arm-relocate-v1":
            self.start[1] = -.032 + jitter[1]
            self.goal[1] = .034 + jitter[1]
        if self.case_id == "arm-reach-v1":
            self.goal = np.array([.087 + jitter[0], .018 + jitter[1], .030])
        self.waypoints = [self.start + np.array([.015, .025, .030]), self.start + np.array([0, .040, .030]), self.goal + np.array([0, 0, .030])]
        self.data.qpos[self.object_qpos:self.object_qpos + 3] = self.start
        self.data.qpos[self.object_qpos + 3:self.object_qpos + 7] = [1, 0, 0, 0]
        self.model.site_pos[self.model.site("goal").id] = self.goal
        self.model.body_pos[self.model.body("obstacle").id] = [.085 + jitter[0], .002 + jitter[1], .010] if self.case_id == "arm-relocate-v1" else [0, 0, -.05]
        if self.case_id == "arms-co-carry-v1":
            payload_address = self.model.jnt_qposadr[self.model.joint("payload_free").id]
            self.data.qpos[payload_address:payload_address + 3] = self.start + [0, 0, .0082]
            self.data.qpos[payload_address + 3:payload_address + 7] = [1, 0, 0, 0]
        for index in range(self.arm_count):
            home = self.start + self.grasp_offsets[index] + [0, 0, .038]
            if self.case_id == "arms-handover-v1" and index == 1:
                home = self.goal + self.grasp_offsets[index] + [0, 0, .035]
            target = inverse_kinematics(home, self.bases[index], "x" if self.arm_count == 2 else "y")
            self.data.qpos[self.qpos_indices[index][:5]] = target
            self.data.qpos[self.qpos_indices[index][5]] = .015
            other = self.model.joint(f"{self.prefixes[index]}_grip_right").qposadr[0]
            self.data.qpos[other] = .015
            self.data.ctrl[index * 6:index * 6 + 5] = target
            self.data.ctrl[index * 6 + 5] = .015
            self.joint_targets[index * 6:index * 6 + 5] = target
            self.joint_targets[index * 6 + 5] = .030
        mujoco.mj_forward(self.model, self.data)
        self.previous_object = self.object_position.copy()
        self.contacts = np.zeros((self.arm_count, 2), dtype=bool)
        self.last_progress = 0.0
        return self.observation(), self.metrics()

    @property
    def object_position(self):
        return self.data.xpos[self.object_id]

    def tip(self, index):
        return self.data.site_xpos[self.tcp_ids[index]].copy()

    def widths(self):
        return [float(2 * self.data.qpos[address[5]]) for address in self.qpos_indices]

    def _local_block(self, index):
        positions = self.data.qpos[self.qpos_indices[index]].copy()
        velocities = self.data.qvel[self.dof_indices[index]].copy()
        positions[5] *= 2
        velocities[5] *= 2
        rotation = self.data.site_xmat[self.tcp_ids[index]].reshape(3, 3)
        tcp_position = self.tip(index) - (self.bases[index] if self.arm_count == 1 else 0)
        return np.concatenate([positions[:5], velocities[:5], [positions[5], velocities[5]],
                               self.prev_action[index * 6:(index + 1) * 6], tcp_position, rotation[:, :2].T.ravel()])

    def observation(self):
        rotation = self.data.xmat[self.object_id].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        object_block = np.concatenate([self.object_position, rotation[:, :2].T.ravel(), velocity[3:], velocity[:3]])
        current_goal = self.waypoints[min(self.waypoint_index, 2)] if self.case_id == "arm-carry-v1" and self.waypoint_index < 3 else self.goal
        goal_rotation = np.eye(3)
        if self.case_id == "arm-reach-v1":
            object_block[:] = 0
            goal_rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        goal_block = np.concatenate([current_goal, goal_rotation[:, :2].T.ravel()])
        time_left = [max(0, 1 - self.steps * CONTROL_DT / SPECS[self.case_id].horizon)]
        if self.arm_count == 1:
            active_obstacle = self.case_id == "arm-relocate-v1"
            obstacle = np.concatenate([self.model.body_pos[self.model.body("obstacle").id], [.018, .006, .010], [0, 1]]) if active_obstacle else np.zeros(8)
            result = np.concatenate([self._local_block(0), object_block, goal_block,
                                     self.contacts.ravel(), time_left, [self.case_id != "arm-reach-v1", 1], obstacle,
                                     [self.waypoint_index / 3, self.case_id != "arm-carry-v1" or self.waypoint_index >= 3]])
        else:
            left_rotation = self.data.site_xmat[self.tcp_ids[0]].reshape(3, 3)
            right_rotation = self.data.site_xmat[self.tcp_ids[1]].reshape(3, 3)
            relative = np.concatenate([left_rotation.T @ (self.tip(1) - self.tip(0)), (left_rotation.T @ right_rotation)[:, :2].T.ravel()])
            semantic_phase = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}.get(self.phase, 5)
            if self.case_id == "arms-handover-v1":
                semantic_phase = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4, 6: 3, 7: 3, 8: 4}.get(self.phase, 5)
            stage = np.eye(6)[semantic_phase]
            payload = np.zeros(7)
            if self.case_id == "arms-co-carry-v1":
                payload_id = self.model.body("payload").id
                payload_velocity = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, payload_id, payload_velocity, 0)
                payload[:3] = rotation.T @ (self.data.xpos[payload_id] - self.object_position) - [0, 0, .0082]
                payload[3:6] = rotation.T @ (payload_velocity[3:] - velocity[3:])
                payload[6] = 1
            tilt = [math.atan2(rotation[2, 1], rotation[2, 2]), math.atan2(-rotation[2, 0], np.hypot(rotation[2, 1], rotation[2, 2]))]
            result = np.concatenate([self._local_block(0), self._local_block(1), object_block, goal_block,
                                     self.contacts.ravel(), time_left, [1, 1], relative,
                                     self.object_position - self.tip(0), self.object_position - self.tip(1),
                                     stage, tilt, [self.object_supported], payload])
        if result.shape != (self.observation_dim,) or not np.isfinite(result).all():
            raise RuntimeError("Arm observation contract violation")
        return result.astype(np.float32)

    def teacher_targets(self):
        elapsed = self.steps * CONTROL_DT
        targets = []
        if self.case_id == "arm-reach-v1":
            return [(self.goal.copy(), .030)]
        phase_durations = [2.5, 2.0, 2.0, 3.0, 2.5, 2.0, 2.5]
        if self.case_id == "arms-co-carry-v1":
            phase_durations = [3.0, 2.0, 3.0, 6.0, 3.0, 2.0, 3.0]
        bounds = np.cumsum(phase_durations)
        self.phase = int(np.searchsorted(bounds, elapsed, side="right"))
        raised = self.start + [0, 0, .032]
        destination = self.goal + [0, 0, .032]
        if self.case_id == "arms-co-carry-v1" and self.phase in (2, 3, 4):
            fraction = float(np.clip((elapsed - bounds[self.phase - 1]) / phase_durations[self.phase], 0, 1))
            blend = fraction ** 3 * (10 - 15 * fraction + 6 * fraction ** 2)
            begin, end = {2: (self.start, raised), 3: (raised, destination), 4: (destination, self.goal)}[self.phase]
            position = begin + (end - begin) * blend
            return [(position + offset, .012) for offset in self.grasp_offsets]
        if self.case_id == "arm-carry-v1":
            phase_durations = [2.5, 2, 2, 2.5, 2.5, 2.5, 2.5, 2, 2.5]
            self.phase = int(np.searchsorted(np.cumsum(phase_durations), elapsed, side="right"))
        if self.case_id == "arms-handover-v1":
            phase_durations = [2.5, 2, 2.5, 3, 2.5, 2, 3, 2.5, 2, 2.5]
            self.phase = int(np.searchsorted(np.cumsum(phase_durations), elapsed, side="right"))
            handoff = np.array([.075, 0, self.start[2] + .032])
            for index in range(2):
                home = self.start if index == 0 else self.goal
                position = home + self.grasp_offsets[index] + [0, 0, .035]
                width = .030
                if index == 0:
                    if self.phase == 0:
                        position = self.start + self.grasp_offsets[0]
                    elif self.phase == 1:
                        position, width = self.start + self.grasp_offsets[0], .006
                    elif self.phase == 2:
                        position, width = raised + self.grasp_offsets[0], .006
                    elif self.phase in (3, 4):
                        position, width = handoff + self.grasp_offsets[0], .006
                    elif self.phase == 5:
                        position = handoff + self.grasp_offsets[0]
                    elif self.phase >= 6:
                        position = handoff + self.grasp_offsets[0] + [0, -.012, .020]
                else:
                    if self.phase == 3:
                        position = handoff + self.grasp_offsets[1]
                    elif self.phase in (4, 5):
                        position, width = handoff + self.grasp_offsets[1], .006
                    elif self.phase == 6:
                        position, width = destination + self.grasp_offsets[1], .006
                    elif self.phase == 7:
                        position, width = self.goal + self.grasp_offsets[1], .006
                    elif self.phase == 8:
                        position = self.goal + self.grasp_offsets[1]
                targets.append((position, width))
            return targets
        for index in range(self.arm_count):
            offset = self.grasp_offsets[index]
            if self.phase == 0:
                position, width = self.start + offset, .030
            elif self.phase == 1:
                position, width = self.start + offset, .006
            elif self.phase == 2:
                position, width = raised + offset, .006
            elif self.case_id == "arm-carry-v1" and self.phase in (3, 4, 5):
                position, width = self.waypoints[self.phase - 3] + offset, .006
            else:
                phase = self.phase - (2 if self.case_id == "arm-carry-v1" else 0)
                if phase == 3:
                    position, width = destination + offset, .006
                elif phase == 4:
                    position, width = self.goal + offset, .006
                elif phase == 5:
                    position, width = self.goal + offset, .030
                else:
                    position, width = destination + offset, .030
            targets.append((position, .012 if self.case_id == "arms-co-carry-v1" and width == .006 else width))
        return targets

    def teacher_action(self):
        self.authored_action_calls += 1
        action = []
        for index, (target, width) in enumerate(self.teacher_targets()):
            angles = inverse_kinematics(target, self.bases[index], "x" if self.arm_count == 2 else "y")
            desired = np.concatenate([angles, [width]])
            error = desired - self.joint_targets[index * 6:(index + 1) * 6]
            requested = error / (VELOCITY_LIMITS * CONTROL_DT)
            requested[:5] /= max(1, float(np.max(np.abs(requested[:5]))))
            if self.case_id == "arms-co-carry-v1":
                requested[:5] *= .5
            if self.case_id == "arms-handover-v1" and self.phase == 7:
                requested[:5] *= .4
            placement_phase = 6 if self.case_id == "arm-carry-v1" else 4
            if self.arm_count == 1 and self.phase == placement_phase:
                requested[:5] *= .5
            requested[5] = np.clip(requested[5], -1, 1)
            action.extend(requested)
        return np.asarray(action, dtype=np.float32)

    def _measure_contacts(self):
        self.contacts[:] = False
        self.contact_support[:] = 0
        contact_forces = np.zeros((self.arm_count, 3))
        self.object_supported = False
        force = np.zeros(6)
        bad_contact = False
        arm_collision = False
        self_collision = False
        table_id = self.table_id
        obstacle_id = self.obstacle_id
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.object_geoms & pair:
                if table_id in pair:
                    self.object_supported = True
                for arm_index, pads in enumerate(self.pad_ids):
                    for pad_index, pad in enumerate(pads):
                        if pad in pair and self.grasp_geoms[arm_index] & pair:
                            self.contacts[arm_index, pad_index] = True
                            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                            self.peak_force = max(self.peak_force, float(abs(force[0])))
                            world_force = contact.frame.reshape(3, 3).T @ force[:3]
                            object_force = world_force * (1 if int(contact.geom2) in self.object_geoms else -1)
                            self.contact_support[arm_index] += object_force[2]
                            contact_forces[arm_index] += object_force
            names = [self.geom_names[geom] for geom in pair]
            if obstacle_id in pair and table_id not in pair and self.case_id == "arm-relocate-v1":
                bad_contact = True
            if table_id in pair and any("pad_" in name or "palm" in name or "forearm" in name for name in names):
                bad_contact = True
            owners = [self.geom_owners[geom] for geom in pair]
            if min(owners) >= 0 and len(set(owners)) > 1:
                arm_collision = True
            elif min(owners) >= 0:
                self_collision = True
        self.invalid_contacts += int(bad_contact)
        self.arm_collisions += int(arm_collision)
        self.self_collisions += int(self_collision)
        if self.arm_count == 2:
            self.peak_internal_force = max(self.peak_internal_force, float(np.linalg.norm((contact_forces[0] - contact_forces[1])[:2]) / 2))

    def _transition(self, stage):
        self.evaluator_stage = stage
        self.evaluator_events.append({"stage": stage, "time_s": float(self.data.time)})

    def _terminal_state_valid(self):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, velocity, 0)
        return bool(self.object_supported and not np.any(self.contacts)
                    and np.linalg.norm(self.object_position - self.goal) <= .010
                    and np.linalg.norm(velocity[3:]) < .020 and np.linalg.norm(velocity[:3]) < .20
                    and all(width >= .026 for width in self.widths())
                    and all(np.linalg.norm(self.tip(index) - self.object_position) >= .030 for index in range(self.arm_count)))

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError("Action must be finite and match the arm contract")
        if self.terminated or self.truncated:
            raise RuntimeError("Reset a finished episode before stepping")
        action = np.clip(action, -1, 1)
        previous_illegal = self.invalid_contacts
        previous_drop = self.drop_count
        previous_targets = self.joint_targets.copy()
        previous_contacts = self.contacts.copy()
        self.joint_targets += action * np.tile(VELOCITY_LIMITS, self.arm_count) * CONTROL_DT
        for index in range(self.arm_count):
            start = index * 6
            self.joint_targets[start:start + 5] = np.clip(self.joint_targets[start:start + 5], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            self.joint_targets[start + 5] = np.clip(self.joint_targets[start + 5], 0, .030)
            self.data.ctrl[start:start + 5] = self.joint_targets[start:start + 5]
            self.data.ctrl[start + 5] = self.joint_targets[start + 5] / 2
        for _ in range(round(CONTROL_DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
            self._measure_contacts()
            if self.evaluator_stage != "initial" and self.object_position[2] - self.start[2] > .010:
                self.ever_airborne = True
            if self.ever_airborne and self.evaluator_stage != "released":
                distance = float(np.linalg.norm(self.object_position - self.goal))
                vertical_speed = abs(float(self.data.qvel[self.model.jnt_dofadr[self.model.joint("object_free").id] + 2]))
                grasp_now = np.any(np.all(self.contacts, axis=1))
                self.grip_loss_s = 0 if grasp_now or self.object_supported else self.grip_loss_s + self.model.opt.timestep
                safe_placement = distance <= .010 and vertical_speed < .030 and (grasp_now or np.any(previous_contacts))
                if (self.object_supported and not safe_placement) or self.grip_loss_s > .020:
                    self.drop_count = max(1, self.drop_count)
            if self.evaluator_stage == "released" and self.object_position[2] - self.start[2] > .010:
                self.drop_count = max(1, self.drop_count)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite arm physics")
        self.steps += 1
        height = float(self.object_position[2] - self.start[2])
        self.peak_height = max(self.peak_height, height)
        grasped = np.any(np.all(self.contacts, axis=1))
        if self.evaluator_stage == "initial" and grasped:
            self._transition("grasped")
        self.lift_run = self.lift_run + 1 if height >= .020 and grasped else 0
        self.max_lift_run = max(self.max_lift_run, self.lift_run)
        if self.max_lift_run >= 50:
            self.ever_lifted = True
            if self.evaluator_stage == "grasped":
                self._transition("lifted")
        self.path_length += float(np.linalg.norm(self.object_position - self.previous_object))
        self.previous_object = self.object_position.copy()
        if height > .015:
            self.transport_steps += 1
            self.dual_grasp_steps += int(np.all(self.contacts))
            rotation = self.data.xmat[self.object_id].reshape(3, 3)
            self.max_transport_tilt = max(self.max_transport_tilt, math.acos(float(np.clip(rotation[2, 2], -1, 1))))
        if self.case_id == "arms-co-carry-v1":
            payload_local = self.data.xmat[self.object_id].reshape(3, 3).T @ (self.data.xpos[self.model.body("payload").id] - self.object_position)
            self.max_payload_slip = max(self.max_payload_slip, float(np.linalg.norm(payload_local - [0, 0, .0082])))
        if self.case_id == "arms-handover-v1":
            self.handover_overlap = self.handover_overlap + 1 if np.all(self.contacts) and height > .015 else 0
            if self.handover_overlap >= 25:
                self.handover_verified = True
            receiving = self.handover_verified and np.all(self.contacts[1]) and not np.any(self.contacts[0]) and height > .015
            self.receiver_hold = self.receiver_hold + 1 if receiving else 0
            self.max_receiver_hold = max(self.max_receiver_hold, self.receiver_hold)
        if self.case_id == "arm-carry-v1" and self.waypoint_index < 3:
            if np.linalg.norm(self.object_position - self.waypoints[self.waypoint_index]) <= .010 and grasped:
                self.waypoint_index += 1
        object_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.object_id, object_velocity, 0)
        position_error = float(np.linalg.norm(self.object_position - self.goal))
        if self.evaluator_stage == "lifted" and self.object_supported and position_error <= .010 and height < .006 and abs(object_velocity[5]) < .030:
            if (np.any(previous_contacts) or np.any(self.contacts)) and np.any(action[5::6] > .05):
                self.controlled_release = True
                self._transition("released")
        if self.arm_count == 2 and height > .015 and np.linalg.norm(object_velocity[3:]) < .010 and np.all(self.contacts):
            total_mass = self.model.body_mass[self.object_id]
            if self.case_id == "arms-co-carry-v1":
                total_mass += self.model.body_mass[self.model.body("payload").id]
            self.load_share_sum += self.contact_support / (total_mass * 9.81)
            self.load_share_samples += 1
            support_ratio = float(self.contact_support.sum() / (total_mass * 9.81))
            self.support_sum += support_ratio
            self.support_valid_samples += int(.8 <= support_ratio <= 1.2)
        settled = self.evaluator_stage == "released" and self._terminal_state_valid()
        self.settle_run = self.settle_run + 1 if settled else 0
        self.max_settle_run = max(self.max_settle_run, self.settle_run)
        reach_error = float(np.linalg.norm(self.tip(0) - self.goal))
        tip_axis = self.data.site_xmat[self.tcp_ids[0]].reshape(3, 3)[:, 0]
        angle_error = math.acos(float(np.clip(-tip_axis[2], -1, 1)))
        self.reach_run = self.reach_run + 1 if reach_error <= .005 and angle_error <= math.radians(5) else 0
        gates = self.task_gates()
        self.completed = all(gates.values())
        self.terminated = self.completed or self.object_position[2] < -.015
        if self.object_position[2] < -.015:
            self.drop_count += 1
        self.truncated = self.steps * CONTROL_DT >= SPECS[self.case_id].horizon and not self.terminated
        progress = -reach_error if self.case_id == "arm-reach-v1" else -position_error
        self.reward_components = {"progress": 20 * (progress - self.last_progress),
                                  "grasp": float(grasped) * .01, "lift": max(height, 0) * .1,
                                  "success": 5 * float(self.completed), "effort": -.0001 * float(np.square(action).sum()),
                                  "action_change": -.0001 * float(np.square(action - self.prev_action).sum()),
                                  "collision": -float(self.invalid_contacts > previous_illegal),
                                  "drop": -2 * float(self.drop_count > previous_drop)}
        reward = sum(self.reward_components.values())
        self.last_progress = progress
        self.prev_action = (self.joint_targets - previous_targets) / (np.tile(VELOCITY_LIMITS, self.arm_count) * CONTROL_DT)
        return self.observation(), float(reward), self.terminated, self.truncated, self.metrics()

    def task_gates(self):
        gates = {"no_illegal_contact": self.invalid_contacts == 0, "no_arm_collision": self.arm_collisions == 0, "no_self_collision": self.self_collisions == 0, "no_drop": self.drop_count == 0}
        if self.case_id == "arm-reach-v1":
            tip_axis = self.data.site_xmat[self.tcp_ids[0]].reshape(3, 3)[:, 0]
            gates["reach_position_and_axis_hold"] = bool(self.reach_run >= 50 and np.linalg.norm(self.tip(0) - self.goal) <= .005 and -tip_axis[2] >= math.cos(math.radians(5)))
            return gates
        gates.update({"lift_with_contact_1s": self.max_lift_run >= 50, "controlled_release": self.controlled_release,
                      "ordered_completion": [event["stage"] for event in self.evaluator_events] == ["grasped", "lifted", "released"],
                      "released_and_settled_2s": self.settle_run >= 100 and self.evaluator_stage == "released" and self._terminal_state_valid()})
        if self.case_id == "arm-carry-v1":
            gates.update({"three_waypoints": self.waypoint_index >= 3, "path_80mm": self.path_length >= .080, "tilt_below_10deg": self.max_transport_tilt < math.radians(10)})
        if self.case_id == "arms-handover-v1":
            gates.update({"handover_overlap_500ms": self.handover_verified, "receiver_only_hold_1s": self.max_receiver_hold >= 50})
        if self.case_id == "arms-co-carry-v1":
            gates.update({"shared_grasp_fraction": self.dual_grasp_steps / max(self.transport_steps, 1) >= .95,
                          "tilt_below_5deg": self.max_transport_tilt < math.radians(5), "payload_slip_below_5mm": self.max_payload_slip < .005,
                          "both_arms_support_load": self.load_share_samples >= 25 and np.all(self.load_share_sum / max(1, self.load_share_samples) >= .20),
                          "net_support_consistent": self.load_share_samples >= 25 and self.support_valid_samples / max(1, self.load_share_samples) >= .95,
                          "internal_force_bounded": self.peak_internal_force < 1.0,
                          "translated_50mm": np.linalg.norm(self.object_position[:2] - self.start[:2]) >= .050})
        return {key: bool(value) for key, value in gates.items()}

    def metrics(self):
        return {"passed": bool(self.completed), "gates": self.task_gates(), "object_position": self.object_position.tolist(),
                "goal": self.goal.tolist(), "position_error_m": float(np.linalg.norm(self.object_position - self.goal)),
                "tip_error_m": float(np.linalg.norm(self.tip(0) - self.goal)), "peak_lift_m": self.peak_height,
                "grasp_hold_s": self.max_lift_run * CONTROL_DT, "settled_s": self.max_settle_run * CONTROL_DT,
                "invalid_contacts": self.invalid_contacts, "arm_collisions": self.arm_collisions,
                "drop_count": self.drop_count, "peak_contact_force_n": self.peak_force,
                "contact_flags": self.contacts.tolist(), "object_path_m": self.path_length,
                "transport_tilt_deg": math.degrees(self.max_transport_tilt), "payload_slip_m": self.max_payload_slip,
                "dual_grasp_fraction": self.dual_grasp_steps / max(self.transport_steps, 1),
                "receiver_hold_s": self.max_receiver_hold * CONTROL_DT, "waypoints_reached": self.waypoint_index,
                "reward_components": self.reward_components.copy(), "reset_options": self.reset_options,
                "object_mass_kg": float(self.model.body_mass[self.object_id]),
                "load_share": (self.load_share_sum / max(1, self.load_share_samples)).tolist(), "load_share_samples": self.load_share_samples,
                "assembly_mass_kg": float(self.model.body_mass[self.object_id]) + (float(self.model.body_mass[self.model.body("payload").id]) if self.case_id == "arms-co-carry-v1" else 0),
                "evaluator_stage": self.evaluator_stage, "evaluator_events": self.evaluator_events.copy(), "self_collisions": self.self_collisions,
                "peak_internal_force_n": self.peak_internal_force, "net_support_mean": self.support_sum / max(1, self.load_share_samples),
                "support_valid_fraction": self.support_valid_samples / max(1, self.load_share_samples),
                "physics_assistance_count": 0, "authored_action_calls": self.authored_action_calls,
                "simulation_only": True, "physical_hardware_verified": False}

    def state(self):
        names = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule", mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder"}
        geoms = []
        for index in range(self.model.ngeom):
            geom_type = self.model.geom_type[index]
            if geom_type not in names:
                continue
            quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(quaternion, self.data.geom_xmat[index])
            geoms.append({"name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or f"geom_{index}",
                          "type": names[geom_type], "size": self.model.geom_size[index].tolist(),
                          "pos": self.data.geom_xpos[index].tolist(), "quat": quaternion.tolist(), "rgba": self.model.geom_rgba[index].tolist()})
        return {"case_id": self.case_id, "seed": self.seed_value, "step": self.steps, "time": self.steps * CONTROL_DT,
                "contract": self.contract, "observation_dim": self.observation_dim, "action_dim": self.action_dim,
                "controller": self.controller, "stage": str(self.phase), "terminated": self.terminated,
                "truncated": self.truncated, "hardware_enabled": False, "joints": np.concatenate([self.data.qpos[address] for address in self.qpos_indices]).tolist(),
                "geoms": geoms, "metrics": self.metrics()}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def write_assets(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {case: (directory / f"{case}.xml").write_text(make_xml(case)) for case in CASES}


def model_hash(case_id):
    return hashlib.sha256(make_xml(case_id).encode()).hexdigest()
