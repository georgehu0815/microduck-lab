"""Compliant free brush, physical palette loading, and contact-only color painting."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

from microduck_local import contract as C
from rlx.environments.drawing import DrawingEnv, drawing_xml, CANVAS_X, CANVAS_Z
from rlx.environments.drawing_reference import assess_drawing
from rlx.environments.brush_reference import (
    BRUSH_RADIUS, COLORS, COLOR_NAMES, PALETTE_X, PALETTE_Y, PALETTE_Z,
    brush_trajectory, color_strokes,
)

CONTRACT_VERSION = "microduck-brush-v2"
OBS_DIM = 93
ACTION_DIM = 15
REWARD_WEIGHTS = {"tracking": 2.0, "contact": 0.4, "grasp": 0.3, "upright": 0.3,
                  "action_rate_penalty": 0.01, "force_penalty": 0.02, "color": 0.4}


def brush_xml(stiffness=25.0, friction=1.2):
    root = ET.fromstring(drawing_xml())
    root.find("option").set("timestep", ".002")
    pencil = root.find(".//body[@name='drawing_pencil']")
    pencil.remove(pencil.find("geom[@name='pencil_tip']"))
    pencil.remove(pencil.find("site[@name='drawing_tip']"))
    shaft = pencil.find("geom[@name='pencil_shaft']")
    shaft.set("fromto", "-.048 0 0 .038 0 0")
    shaft.set("rgba", ".22 .32 .42 1")
    ET.SubElement(pencil, "geom", name="brush_ferrule", type="capsule", fromto=".03 0 0 .04 0 0", size=".0022", mass=".0001", rgba=".6 .65 .7 1", contype="0", conaffinity="0", group="2")
    tuft = ET.SubElement(pencil, "body", name="brush_compliant_tuft", pos=".048 0 0")
    ET.SubElement(tuft, "joint", name="bristle_compression", type="slide", axis="1 0 0", range="-.003 .0003", stiffness=str(stiffness), damping=".32", armature=".001", solreflimit=".01 1")
    ET.SubElement(tuft, "geom", name="pencil_tip", type="sphere", size=str(BRUSH_RADIUS), mass=".00003", rgba=".60 .43 .25 1", contype="4", conaffinity="24", friction=".12 .001 .00001", solref=".012 1", group="2")
    for index, displacement in enumerate((-0.0007, 0.0, 0.0007)):
        ET.SubElement(tuft, "geom", name=f"brush_hair_{index}", type="capsule", fromto=f"-.008 {displacement} 0 0 {displacement} 0", size=".0003", mass=".000005", rgba=".64 .46 .27 1", contype="0", conaffinity="0", group="2")
    ET.SubElement(tuft, "site", name="drawing_tip", size=".0003", group="3")
    world = root.find("worldbody")
    ET.SubElement(world, "geom", name="color_palette", type="box", pos=f"{PALETTE_X+.004} {PALETTE_Y} .2245", size=".002 .007 .026", rgba=".78 .73 .63 1", contype="0", conaffinity="0", group="2")
    for index, (height, rgb) in enumerate(zip(PALETTE_Z, COLORS)):
        ET.SubElement(world, "geom", name=f"paint_well_{index}", type="box", pos=f"{PALETTE_X+.001} {PALETTE_Y} {height}", size=".001 .004 .004", rgba=" ".join(map(str, (*rgb, 1))), contype="8", conaffinity="4", friction=".1 .001 .00001", solref=".012 1", group="2")
    for name in ("upper_beak_pad", "lower_beak_pad", "pencil_shaft"):
        root.find(f".//geom[@name='{name}']").set("friction", f"{friction} .015 .001")
    return ET.tostring(root, encoding="unicode")


class BrushEnv(DrawingEnv):
    def __init__(self, seed=0, max_episode_s=120.0, assistance=0.0, scale=1.0,
                 offset=(0.0, 0.0), friction=1.2, stiffness=25.0):
        gym.Env.__init__(self)
        if not (70 <= max_episode_s <= 180 and assistance == 0 and .8 <= scale <= 1.2):
            raise ValueError("brush needs 70–180 seconds, no support, and scale .8–1.2")
        if not (10 <= stiffness <= 60 and .5 <= friction <= 2) or np.shape(offset) != (2,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > .006:
            raise ValueError("invalid brush physical parameters")
        self.model = mujoco.MjModel.from_xml_string(brush_xml(stiffness, friction))
        self.data = mujoco.MjData(self.model)
        self.assistance, self.scale, self.offset = 0.0, scale, tuple(offset)
        self.trajectory = brush_trajectory(round(max_episode_s / C.CTRL_DT), scale, offset)
        self.max_steps = len(self.trajectory["xyz"])
        self.max_episode_s = self.max_steps * C.CTRL_DT
        self._trunk_id = self.model.body("trunk_base").id
        self._head_id = self.model.body("jaw_soft").id
        self._pencil_id = self.model.body("drawing_pencil").id
        self._tip_id = self.model.site("drawing_tip").id
        self.joint_qpos = np.array([self.model.joint(name).qposadr[0] for name in C.JOINT_NAMES])
        self.joint_dofs = np.array([self.model.joint(name).dofadr[0] for name in C.JOINT_NAMES])
        self.mouth_qpos = self.model.joint("mouth_open").qposadr[0]
        self.mouth_dof = self.model.joint("mouth_open").dofadr[0]
        self.pencil_qpos = self.model.joint("pencil_free").qposadr[0]
        self.spring_qpos = self.model.joint("bristle_compression").qposadr[0]
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), dtype=np.float32)
        self.twist_cmd = np.zeros(3, np.float32)
        self.head_cmd = np.zeros(4, np.float32)
        self.body_cmd = np.zeros(6, np.float32)
        self.reset(seed=seed)

    @property
    def target(self):
        return self.trajectory["xyz"][min(self.step_count, self.max_steps - 1)].copy()

    def reset(self, *, seed=None, options=None):
        self.loaded_color = -1
        self.paint_events = []
        self.color_trace = []
        self._contact_color = -1
        self._well_contact = -1
        self.max_compression = 0.0
        return super().reset(seed=seed, options=options)

    def _get_obs(self):
        obs = np.zeros(OBS_DIM, np.float32)
        obs[:83] = super()._get_obs()
        index = min(self.step_count, self.max_steps - 1)
        if self.loaded_color >= 0:
            obs[83 + self.loaded_color] = 1
        obs[87 + self.trajectory["color"][index]] = 1
        obs[91] = self.data.qpos[self.spring_qpos] * 1000
        obs[92] = self.trajectory["mode"][index] / 2
        return obs

    def _contacts(self):
        forces, mark = super()._contacts()
        tip = self.model.geom("pencil_tip").id
        self._well_contact = -1
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.efc_address < 0 or tip not in contact.geom:
                continue
            for color in range(4):
                if self.model.geom(f"paint_well_{color}").id not in contact.geom:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
                if wrench[0] > .0001:
                    self._well_contact = color
                    self.max_force = max(self.max_force, float(wrench[0]))
                    if color != self.loaded_color:
                        self.paint_events.append({"time": float(self.data.time), "color": color, "force_n": float(wrench[0])})
                    self.loaded_color = color
        self.max_compression = max(self.max_compression, -float(self.data.qpos[self.spring_qpos]))
        if mark is not None and self.loaded_color >= 0:
            self._contact_color = self.loaded_color
            return forces, mark
        return forces, None

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("brush requires 15 finite controls")
        action = np.clip(action, -1, 1)
        previous = self.last_action.copy()
        self.data.ctrl[:14] = C.DEFAULT_POSE + action[:14]
        self.data.ctrl[14] = .24 + .36 * action[14]
        self.prev_joint_vel = self.data.qvel[self.joint_dofs].copy()
        index = min(self.step_count, self.max_steps - 1)
        requested_down = bool(self.trajectory["pen_down"][index])
        self.pen_up_steps += int(not requested_down)
        desired_color = int(self.trajectory["color"][index])
        target = self.target
        mark, mark_force, mark_color = None, 0.0, -1
        grasped_substeps = 0
        substeps = round(C.CTRL_DT / self.model.opt.timestep)
        for _ in range(substeps):
            self.data.xfrc_applied[:] = 0
            mujoco.mj_step(self.model, self.data)
            self.contact_forces, substep_mark = self._contacts()
            self.max_force = max(self.max_force, float(self.contact_forces[0]))
            grasped_substeps += int(np.min(self.contact_forces[1:]) > .0001)
            if substep_mark is not None:
                mark, mark_force, mark_color = substep_mark, float(self.contact_forces[0]), self._contact_color
        self.tip_velocity = (self.tip - self._last_tip) / C.CTRL_DT
        self._last_tip = self.tip
        if mark is not None:
            if not self._in_contact or (self.color_trace and self.color_trace[-1] != mark_color):
                self._stroke_id += 1
            self.trace.append([float(self.data.time), *map(float, mark), self._stroke_id, mark_force, requested_down])
            self.color_trace.append(mark_color)
            self.contact_steps += 1
            self.pen_up_leaks += int(not requested_down)
        self._in_contact = mark is not None
        grasp = grasped_substeps == substeps
        self.grasp_steps += int(grasp)
        self.last_action = action.copy()
        self.step_count += 1
        upright = float(self.data.xmat[self._trunk_id].reshape(3, 3)[2, 2])
        distance = float(np.linalg.norm(self.tip - target))
        terms = {"tracking": 2 * np.exp(-(distance/.006)**2), "contact": .4 * float((mark is not None) == requested_down),
                 "grasp": .3 * grasp, "upright": .3 * max(0, upright),
                 "action_rate_penalty": -.01 * float(np.square(action - previous).sum()),
                 "force_penalty": -.02 * max(0., self.contact_forces[0] - .5)**2,
                 "color": .4 * float(self.loaded_color == desired_color)}
        fallen = bool(self.data.xpos[self._trunk_id, 2] < .075 or upright < .7)
        dropped = bool(self.tip[2] < .13)
        self.failed |= fallen or dropped
        info = {"terms": terms, "distance_m": distance, "grasp": grasp, "fallen": fallen, "dropped": dropped, "loaded_color": self.loaded_color}
        return self._get_obs(), float(sum(terms.values())), fallen or dropped, self.step_count >= self.max_steps, info

    def drawing_payload(self):
        points = [[CANVAS_X-.0003, row[2], row[3], row[4]] for row in self.trace]
        return {"points": points, "colors": list(self.color_trace), "palette": COLORS,
                "brush_radius": BRUSH_RADIUS, "contract": CONTRACT_VERSION, "assistance": 0}

    def assessment(self):
        reference = color_strokes(self.scale, self.offset)
        by_color = {}
        for color, name in enumerate(COLOR_NAMES):
            recorded = []
            for stroke_id in sorted({row[4] for row in self.trace}):
                points = [[row[2], row[3] - CANVAS_Z] for row, ink in zip(self.trace, self.color_trace) if row[4] == stroke_id and ink == color]
                if points:
                    recorded.append(np.asarray(points))
            by_color[name] = assess_drawing([points for ink, points in reference if ink == color], recorded)
        grasp = self.grasp_steps / max(1, self.step_count)
        leaks = self.pen_up_leaks / max(1, self.pen_up_steps)
        completed = self.step_count >= self.max_steps
        passed = all(item["passed"] and item["coverage"] >= .95 and item["precision"] >= .95 for item in by_color.values()) and completed and not self.failed and grasp >= .9 and self.max_force < .5 and leaks < .05 and self.max_compression < .0035 and [event["color"] for event in self.paint_events] == list(range(4))
        return {"passed": bool(passed), "unassisted": True, "assistance": 0,
                "contract_version": CONTRACT_VERSION, "by_color": by_color,
                "coverage": float(np.mean([item["coverage"] for item in by_color.values()])),
                "precision": float(np.mean([item["precision"] for item in by_color.values()])),
                "grasp_fraction": grasp, "max_normal_force_n": self.max_force,
                "pen_up_leak_fraction": leaks, "completed": completed,
                "fallen_or_dropped": self.failed, "max_compression_m": self.max_compression,
                "palette_contacts": self.paint_events, "contact_steps": self.contact_steps}
