"""Contact-only pencil drawing, with a separate simulated mouth interface."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from microduck_local import contract as C
from rlx.environments.drawing_reference import make_trajectory, swimming_duck_strokes, assess_drawing

CONTRACT_VERSION = "microduck-drawing-v1"
OBS_DIM = 83
ACTION_DIM = 15
ROBOT_DIR = Path(__file__).resolve().parents[1] / "mjlab_microduck/robot/microduck"
CANVAS_X = 0.160
CANVAS_Z = 0.2245
PENCIL_RADIUS = 0.0015
REWARD_WEIGHTS = {"tracking": 2.0, "contact": 0.4, "grasp": 0.3, "upright": 0.3,
                  "action_rate_penalty": 0.01, "force_penalty": 0.02}


def drawing_xml() -> str:
    root = ET.parse(ROBOT_DIR / "robot_allcollisions.xml").getroot()
    root.find("compiler").set("meshdir", str(ROBOT_DIR / "assets"))
    ET.SubElement(root, "option", timestep="0.005", integrator="implicitfast", iterations="50", cone="elliptic")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 .05", rgba=".10 .13 .16 1", friction="1 .005 .0001", contype="17", conaffinity="1")
    ET.SubElement(world, "light", pos="0 -1 2", dir="0 0 -1", diffuse=".9 .9 .9")
    ET.SubElement(world, "light", pos="-1 -1 1", dir="1 1 -1", diffuse=".8 .8 .8", ambient=".25 .25 .25")
    for side in (-1, 1):
        ET.SubElement(world, "geom", name=f"easel_leg_{side}", type="capsule", fromto=f".22 {side*.11} 0 .19 {side*.07} .35", size=".008", rgba=".5 .28 .12 1", group="2")
    ET.SubElement(world, "geom", name="easel_rear_leg", type="capsule", fromto=".34 0 0 .19 0 .35", size=".008", rgba=".5 .28 .12 1", group="2")
    ET.SubElement(world, "geom", name="drawing_board", type="box", pos=f"{CANVAS_X+.007} 0 {CANVAS_Z}", size=".005 .105 .082", rgba=".53 .32 .15 1", group="2")
    ET.SubElement(world, "geom", name="drawing_canvas", type="box", pos=f"{CANVAS_X+.001} 0 {CANVAS_Z}", size=".001 .095 .073", rgba=".98 .96 .89 1", contype="8", conaffinity="4", friction=".15 .001 .00001", group="2", solref=".01 1")
    head = root.find(".//body[@name='jaw_soft']")
    mouth = np.array([-.00809334, 0, -.0777383])
    upper = mouth + np.array([.003, 0, -.002])
    pad_args = dict(type="box", size=".0015 .005 .009", rgba=".16 .19 .20 1", group="2", contype="4", conaffinity="4", condim="6", friction="1.2 .015 .001", solref=".01 1", solimp=".95 .99 .0001", mass=".001")
    ET.SubElement(head, "geom", name="upper_beak_pad", pos=" ".join(map(str, upper)), **pad_args)
    pivot = mouth + np.array([-.003, 0, .012])
    jaw = ET.SubElement(head, "body", name="drawing_lower_beak", pos=" ".join(map(str, pivot)))
    ET.SubElement(jaw, "joint", name="mouth_open", type="hinge", axis="0 1 0", range="-.12 .6", damping=".005", armature=".00001", frictionloss="0")
    ET.SubElement(jaw, "geom", name="lower_beak_pad", pos="0 0 -.014", **pad_args)
    ET.SubElement(jaw, "geom", name="mouth_adapter", type="box", pos="-.002 0 -.006", size=".001 .008 .008", rgba=".95 .48 .07 1", mass=".002", contype="0", conaffinity="0", group="2")
    actuator = root.find("actuator")
    for servo in actuator.findall("position"):
        servo.set("kp", "8")
        servo.set("kv", ".2")
    ET.SubElement(actuator, "position", name="mouth_servo", joint="mouth_open", kp=".6", kv=".006", ctrlrange="-.12 .6", forcerange="-.04 .04")
    pencil = ET.SubElement(world, "body", name="drawing_pencil", pos=".1117 0 .2245")
    ET.SubElement(pencil, "freejoint", name="pencil_free")
    ET.SubElement(pencil, "geom", name="pencil_shaft", type="capsule", fromto="-.048 0 0 .045 0 0", size=".0015", mass=".002", rgba=".95 .63 .08 1", group="2", contype="4", conaffinity="20", condim="6", friction="1.2 .015 .001", solref=".01 1")
    ET.SubElement(pencil, "geom", name="pencil_tip", type="sphere", pos=".048 0 0", size=".0007", mass=".00005", rgba=".10 .10 .10 1", group="2", contype="4", conaffinity="24", friction=".15 .001 .00001", solref=".01 1")
    ET.SubElement(pencil, "site", name="drawing_tip", pos=".048 0 0", size=".0005", group="3")
    ET.SubElement(head, "site", name="grasp_tip_proxy", pos=f"{mouth[0]} 0 {mouth[2]-.091}", size=".0005", group="3")
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    return ET.tostring(root, encoding="unicode")


@lru_cache(maxsize=1)
def _model():
    return mujoco.MjModel.from_xml_string(drawing_xml())


class DrawingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    studio_recipe = "drawing"

    def __init__(self, seed=0, max_episode_s=32.0, assistance=0.0, scale=1.0, offset=(0., 0.), friction=1.2):
        super().__init__()
        if not (4 <= max_episode_s <= 120 and 0 <= assistance <= 1 and .5 <= scale <= 1.3):
            raise ValueError("invalid drawing horizon, assistance, or scale")
        if not np.isfinite(friction) or not .1 <= friction <= 2.0 or np.shape(offset) != (2,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > .02:
            raise ValueError("invalid drawing friction or canvas offset")
        self.model = _model() if friction == 1.2 else mujoco.MjModel.from_xml_string(drawing_xml())
        if friction != 1.2:
            for name in ("upper_beak_pad", "lower_beak_pad", "pencil_shaft"):
                self.model.geom(name).friction[0] = friction
        self.data = mujoco.MjData(self.model)
        self.max_episode_s = float(max_episode_s)
        self.max_steps = round(max_episode_s / C.CTRL_DT)
        self.assistance, self.scale, self.offset = float(assistance), float(scale), tuple(offset)
        self.trajectory = make_trajectory(self.max_steps, scale=self.scale, offset=self.offset)
        self._trunk_id = self.model.body("trunk_base").id
        self._head_id = self.model.body("jaw_soft").id
        self._pencil_id = self.model.body("drawing_pencil").id
        self._tip_id = self.model.site("drawing_tip").id
        self.joint_qpos = np.array([self.model.joint(name).qposadr[0] for name in C.JOINT_NAMES])
        self.joint_dofs = np.array([self.model.joint(name).dofadr[0] for name in C.JOINT_NAMES])
        self.mouth_qpos = self.model.joint("mouth_open").qposadr[0]
        self.mouth_dof = self.model.joint("mouth_open").dofadr[0]
        self.pencil_qpos = self.model.joint("pencil_free").qposadr[0]
        self.action_space = gym.spaces.Box(-1., 1., (ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.twist_cmd = np.zeros(3, np.float32)
        self.head_cmd = np.zeros(4, np.float32)
        self.body_cmd = np.zeros(6, np.float32)
        self.reset(seed=seed)

    @property
    def tip(self):
        return self.data.site_xpos[self._tip_id].copy()

    @property
    def target(self):
        index = min(self.step_count, self.max_steps - 1)
        point = self.trajectory["points"][index]
        pen_down = self.trajectory["pen_down"][index]
        return np.array([CANVAS_X + (-.00055 if pen_down else -.004), point[0], CANVAS_Z + point[1]])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = [0, 0, .12, 1, 0, 0, 0]
        self.data.qpos[self.joint_qpos] = C.DEFAULT_POSE
        self.data.qpos[self.mouth_qpos] = -.025
        self.data.qpos[self.pencil_qpos:self.pencil_qpos+7] = [.1117, 0, .2245, 1, 0, 0, 0]
        self.data.qpos[self.pencil_qpos+1] += self.np_random.uniform(-.00005, .00005)
        self.data.ctrl[:14] = C.DEFAULT_POSE
        self.data.ctrl[14] = -.06
        self.step_count = 0
        self.last_action = np.zeros(15, np.float32)
        self.last_action[14] = -.8333333
        self.prev_joint_vel = np.zeros(14, np.float32)
        self.contact_forces = np.zeros(3)
        self.trace = []
        self._stroke_id = 0
        self._in_contact = False
        self.contact_steps = self.grasp_steps = self.pen_up_leaks = self.pen_up_steps = 0
        self.max_force = 0.
        self.failed = False
        mujoco.mj_forward(self.model, self.data)
        self._last_tip = self.tip
        self.tip_velocity = np.zeros(3)
        return self._get_obs(), {}

    def _get_obs(self):
        rotation = self.data.xmat[self._trunk_id].reshape(3, 3)
        obs = np.zeros(83, np.float32)
        obs[:3] = self.data.sensor("angular-velocity").data
        obs[3:6] = rotation.T @ np.array([0., 0., -1.])
        obs[6:20] = self.data.qpos[self.joint_qpos] - C.DEFAULT_POSE
        obs[20:34] = self.prev_joint_vel
        obs[34:48] = self.last_action[:14]
        obs[61:64] = [self.data.qpos[self.mouth_qpos], self.data.qvel[self.mouth_dof], self.last_action[14]]
        obs[64:67] = rotation.T @ (self.tip - self.data.xpos[self._trunk_id])
        obs[67:70] = rotation.T @ (self.target - self.tip)
        obs[70:73] = rotation.T @ self.data.xmat[self._pencil_id].reshape(3, 3)[:, 0]
        obs[73:76] = rotation.T @ self.tip_velocity
        index = min(self.step_count, self.max_steps - 1)
        obs[76:79] = rotation.T @ np.r_[0, self.trajectory["velocity"][index]]
        obs[79] = self.trajectory["pen_down"][index]
        obs[80:83] = self.contact_forces
        return obs

    def _contacts(self):
        forces = np.zeros(3)
        canvas = self.model.geom("drawing_canvas").id
        tip = self.model.geom("pencil_tip").id
        shaft = self.model.geom("pencil_shaft").id
        pads = [self.model.geom(name).id for name in ("upper_beak_pad", "lower_beak_pad")]
        mark = None
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.efc_address < 0:
                continue
            pair = set(contact.geom)
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            force = max(0., float(wrench[0]))
            if pair == {canvas, tip}:
                forces[0] += force
                if force >= .0001:
                    mark = contact.pos.copy()
            for pad_index, pad in enumerate(pads):
                if pair == {shaft, pad}:
                    forces[pad_index + 1] += force
        return forces, mark

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (15,) or not np.isfinite(action).all():
            raise ValueError("drawing requires 15 finite controls")
        action = np.clip(action, -1, 1)
        previous = self.last_action.copy()
        self.data.ctrl[:14] = C.DEFAULT_POSE + action[:14]
        self.data.ctrl[14] = .24 + .36 * action[14]
        self.prev_joint_vel = self.data.qvel[self.joint_dofs].copy()
        requested_down = bool(self.trajectory["pen_down"][min(self.step_count, self.max_steps-1)])
        self.pen_up_steps += int(not requested_down)
        target = self.target
        mark = None
        mark_force = 0.
        grasped_substeps = 0
        for _ in range(C.DECIMATION):
            self.data.xfrc_applied[:] = 0
            if self.assistance:
                self.data.xfrc_applied[self._trunk_id, :3] = self.assistance * (250 * (np.array([0, 0, .12]) - self.data.xpos[self._trunk_id]) - 10 * self.data.qvel[:3])
                rotation = self.data.xmat[self._trunk_id].reshape(3, 3)
                self.data.xfrc_applied[self._trunk_id, 3:] = self.assistance * (-2 * np.cross([0, 0, 1], rotation[:, 2]) - .15 * self.data.qvel[3:6])
            mujoco.mj_step(self.model, self.data)
            self.contact_forces, substep_mark = self._contacts()
            self.max_force = max(self.max_force, float(self.contact_forces[0]))
            grasped_substeps += int(np.min(self.contact_forces[1:]) > .0001)
            if substep_mark is not None:
                mark = substep_mark
                mark_force = float(self.contact_forces[0])
        self.tip_velocity = (self.tip - self._last_tip) / C.CTRL_DT
        self._last_tip = self.tip
        if mark is not None:
            if not self._in_contact:
                self._stroke_id += 1
            self.trace.append([float(self.data.time), float(mark[0]), float(mark[1]), float(mark[2]), self._stroke_id, mark_force, requested_down])
            self.contact_steps += 1
            self.pen_up_leaks += int(not requested_down)
        self._in_contact = mark is not None
        grasp = grasped_substeps == C.DECIMATION
        self.grasp_steps += int(grasp)
        self.last_action = action.copy()
        self.step_count += 1
        upright = float(self.data.xmat[self._trunk_id].reshape(3, 3)[2, 2])
        distance = float(np.linalg.norm(self.tip - target))
        terms = {"tracking": 2 * np.exp(-(distance/.006)**2), "contact": .4 * float((mark is not None) == requested_down),
                 "grasp": .3 * grasp, "upright": .3 * max(0, upright),
                 "action_rate_penalty": -.01 * float(np.square(action - previous).sum()),
                 "force_penalty": -.02 * max(0., self.contact_forces[0] - .5)**2}
        fallen = bool(self.data.xpos[self._trunk_id, 2] < .075 or upright < .7)
        dropped = bool(self.tip[2] < .13)
        self.failed |= fallen or dropped
        return self._get_obs(), float(sum(terms.values())), fallen or dropped, self.step_count >= self.max_steps, {"terms": terms, "distance_m": distance, "grasp": grasp, "fallen": fallen, "dropped": dropped}

    def teacher_action(self):
        target = self.target
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacobian, None, self.tip, self._head_id)
        head_dofs = self.joint_dofs[C.HEAD_JOINT_IDS]
        matrix = jacobian[:, head_dofs]
        change = matrix.T @ np.linalg.solve(matrix @ matrix.T + .00003*np.eye(3), target - self.tip)
        action = np.zeros(15, np.float32)
        action[C.HEAD_JOINT_IDS] = self.last_action[C.HEAD_JOINT_IDS] + np.clip(.3 * change, -.025, .025)
        rotation = self.data.xmat[self._trunk_id].reshape(3, 3)
        correction = float(np.arctan2(rotation[0, 2], rotation[2, 2]) + .1 * self.data.qvel[4])
        action[4], action[13] = correction, -correction
        action[14] = -.8333333
        return np.clip(action, -1, 1)

    def drawing_payload(self):
        return {"points": [[CANVAS_X-.0003, row[2], row[3], row[4]] for row in self.trace], "contract": CONTRACT_VERSION, "assistance": self.assistance}

    def assessment(self):
        recorded = []
        for stroke_id in sorted({row[4] for row in self.trace}):
            recorded.append(np.array([[row[2], row[3]-CANVAS_Z] for row in self.trace if row[4] == stroke_id]))
        reference = [stroke + np.array(self.offset) for stroke in swimming_duck_strokes(self.scale)]
        result = assess_drawing(reference, recorded)
        result.update({"grasp_fraction": self.grasp_steps / max(1, self.step_count), "contact_steps": self.contact_steps,
                       "max_normal_force_n": self.max_force, "pen_up_leak_fraction": self.pen_up_leaks / max(1, self.pen_up_steps),
                       "ink_contact_fraction": self.contact_steps / max(1, self.step_count),
                       "completed": self.step_count >= self.max_steps, "fallen_or_dropped": self.failed, "assistance": self.assistance,
                       "unassisted": self.assistance == 0, "contract_version": CONTRACT_VERSION})
        result["passed"] = bool(result["passed"] and result["completed"] and not self.failed and self.assistance == 0 and result["grasp_fraction"] >= .9 and self.max_force < .5 and result["pen_up_leak_fraction"] < .05)
        return result
