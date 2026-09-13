from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import ROOT, SPEC_PATH, load_design, sha256


def numbers(values):
    return " ".join(f"{float(value):.10g}" for value in values)


def box(parent, name, center, dimensions, mass, color, collision=True):
    return ET.SubElement(parent, "geom", name=name, type="box", pos=numbers(center),
                         size=numbers(np.array(dimensions) / 2), mass=str(mass),
                         rgba=color, contype="1" if collision else "0", conaffinity="3" if collision else "0")


def make_tree(mode="free", payload_kg=0.01):
    if mode not in ("free", "fixture"):
        raise ValueError("mode must be free or explicitly labeled fixture")
    if not np.isfinite(payload_kg) or not 0 < payload_kg <= 0.05:
        raise ValueError("Object mass must be positive and at most 0.05 kg")
    design = load_design()
    source = ROOT / design["source_model"]
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", f"{design['design_id']}-{mode}")
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    root.find("compiler").set("balanceinertia", "false")
    ET.SubElement(root, "option", timestep=str(design["control"]["physics_dt_s"]), integrator="implicitfast", gravity="0 0 -9.81", iterations="100")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7")
    world = root.find("worldbody")
    trunk = world.find("body[@name='trunk_base']")
    removed_neck = trunk.find("body[@name='neck']")
    removed_joints = {node.get("name") for node in removed_neck.iter("joint")}
    trunk.remove(removed_neck)
    if mode == "fixture":
        trunk.remove(trunk.find("freejoint"))
    for actuator in list(root.find("actuator")):
        if actuator.get("joint") in removed_joints:
            root.find("actuator").remove(actuator)
    defaults = root.find("default")
    arm_default = ET.SubElement(defaults, "default", {"class": "integrated_arm"})
    ET.SubElement(arm_default, "joint", type="hinge", damping="0.012", frictionloss="0.002", armature="0.00002")
    ET.SubElement(arm_default, "geom", contype="1", conaffinity="3", friction="0.8 0.005 0.001", condim="4")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.1", rgba="0.20 0.24 0.28 1", friction="0.9 0.005 0.001", contype="1", conaffinity="3")
    ET.SubElement(world, "light", pos="0.3 -0.4 1.2", dir="-0.3 0.4 -1", diffuse="0.9 0.9 0.9")
    for name, properties in design["replacement_mass_assumptions"].items():
        lump = ET.SubElement(trunk, "body", name=f"assumed_{name}", pos=numbers(properties["pos_m"]), childclass="integrated_arm")
        box(lump, f"assumed_{name}_geom", [0, 0, 0], properties["size_m"], properties["mass_kg"], "0.25 0.36 0.45 0.65")
    arm = design["arm"]
    mount = ET.SubElement(trunk, "body", name="arm_mount", pos=numbers(arm["mount_pos_m"]), childclass="integrated_arm")
    box(mount, "arm_mount_plate", [0, 0, -0.008], [0.044, 0.044, 0.004], 0.020, "0.4 0.5 0.55 1")
    motor_color = "0.14 0.20 0.27 1"
    box(mount, "arm_yaw_motor", arm["servo_centers_local_m"]["yaw"], arm["motor_envelope_m"], 0.018, motor_color)
    yaw = ET.SubElement(mount, "body", name="arm_yaw")
    ET.SubElement(yaw, "joint", name="arm_base_yaw", axis="0 0 1", range=numbers(arm["joint_ranges_rad"][0]))
    box(yaw, "arm_shoulder_motor", arm["servo_centers_local_m"]["shoulder"], arm["motor_envelope_m"], 0.018, motor_color)
    shoulder = ET.SubElement(yaw, "body", name="arm_upper")
    ET.SubElement(shoulder, "joint", name="arm_shoulder_pitch", axis="0 1 0", range=numbers(arm["joint_ranges_rad"][1]))
    upper, forearm, tool = arm["link_lengths_m"]
    ET.SubElement(shoulder, "geom", name="arm_upper_link", type="capsule", fromto=numbers([0, 0, 0, upper, 0, 0]), size="0.004", mass="0.018", rgba="0.95 0.63 0.14 1")
    box(shoulder, "arm_elbow_motor", arm["servo_centers_local_m"]["elbow"], arm["motor_envelope_m"], 0.018, motor_color)
    elbow = ET.SubElement(shoulder, "body", name="arm_forearm", pos=numbers([upper, 0, 0]))
    ET.SubElement(elbow, "joint", name="arm_elbow_pitch", axis="0 1 0", range=numbers(arm["joint_ranges_rad"][2]))
    ET.SubElement(elbow, "geom", name="arm_forearm_link", type="capsule", fromto=numbers([0, 0, 0, forearm, 0, 0]), size="0.0035", mass="0.016", rgba="0.95 0.63 0.14 1")
    box(elbow, "arm_wrist_motor", arm["servo_centers_local_m"]["wrist"], arm["motor_envelope_m"], 0.018, motor_color)
    hand = ET.SubElement(elbow, "body", name="arm_hand", pos=numbers([forearm, 0, 0]))
    ET.SubElement(hand, "joint", name="arm_wrist_pitch", axis="0 1 0", range=numbers(arm["joint_ranges_rad"][3]))
    box(hand, "arm_gripper_motor", arm["servo_centers_local_m"]["gripper"], arm["motor_envelope_m"], 0.018, motor_color)
    box(hand, "arm_palm", [0, 0, 0], [0.010, 0.030, 0.006], 0.012, "0.5 0.6 0.65 1")
    ET.SubElement(hand, "site", name="arm_tcp", pos=numbers([tool, 0, 0]), size="0.002", rgba="0 1 0.5 1")
    for side, sign in (("left", 1), ("right", -1)):
        finger = ET.SubElement(hand, "body", name=f"arm_finger_{side}", pos=numbers([0, sign * arm["gripper_pivot_half_spacing_m"], 0]))
        joint_name = "arm_gripper" if side == "left" else "arm_gripper_follower"
        limits = [-arm["gripper_max_closure_rad"], 0] if side == "left" else [0, arm["gripper_max_closure_rad"]]
        ET.SubElement(finger, "joint", name=joint_name, axis="0 0 1", range=numbers(limits))
        ET.SubElement(finger, "geom", name=f"arm_pad_{side}", type="capsule", fromto=numbers([0.008, 0, 0, tool, 0, 0]), size="0.0025", mass="0.005", friction="1.0 0.005 0.001", rgba="0.2 0.8 0.5 1")
    ET.SubElement(root.find("equality"), "joint", name="gripper_gear_coupling", joint1="arm_gripper_follower", joint2="arm_gripper", polycoef="0 -1 0 0 0", solref="0.004 1")
    for index, joint in enumerate(design["joint_order"][10:]):
        ET.SubElement(root.find("actuator"), "position", name=joint, joint=joint, kp=str(arm["simulation_kp"]), kv=str(arm["simulation_kv"]), forcerange=numbers([-arm["simulation_force_limit_nm"], arm["simulation_force_limit_nm"]]), ctrlrange=numbers(arm["joint_ranges_rad"][index]))
    box(world, "work_surface", [0.135, 0, 0.090], [0.110, 0.130, 0.010], 0, "0.50 0.42 0.32 1")
    box(world, "obstacle", [0.145, 0.030, 0.112], [0.012, 0.018, 0.034], 0, "0.9 0.3 0.2 1")
    item = ET.SubElement(world, "body", name="task_object", pos="0.125 -0.018 0.103")
    ET.SubElement(item, "freejoint", name="task_object_free")
    box(item, "object_geom", [0, 0, 0], [0.012, 0.012, 0.012], payload_kg, "0.8 0.3 0.65 1")
    ET.SubElement(world, "site", name="goal", pos="0.15 0.015 0.115", size="0.008", rgba="0.2 0.95 0.4 0.35")
    ET.SubElement(root, "custom")
    ET.SubElement(root.find("custom"), "text", name="evidence_class", data="fixture_assisted" if mode == "fixture" else "free_base_unvalidated")
    return root


def compile_model(mode="free", payload_kg=0.01):
    return mujoco.MjModel.from_xml_string(ET.tostring(make_tree(mode, payload_kg), encoding="unicode"))


def quaternion_rpy(quaternion):
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, quaternion)
    rotation = matrix.reshape(3, 3)
    pitch = np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0]))
    return [np.arctan2(rotation[2, 1], rotation[2, 2]), pitch, np.arctan2(rotation[1, 0], rotation[0, 0])]


def export_urdf(model, output):
    robot = ET.Element("robot", name="microduck_single_arm_v1_b")
    source_tree = make_tree()
    source_bodies = {body.get("name"): body for body in source_tree.iter("body")}
    ET.SubElement(robot, "link", name="world")
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name == "task_object":
            continue
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=numbers(model.body_ipos[body_id]), rpy=numbers(quaternion_rpy(model.body_iquat[body_id])))
        ET.SubElement(inertial, "mass", value=str(model.body_mass[body_id]))
        inertia = model.body_inertia[body_id]
        ET.SubElement(inertial, "inertia", ixx=str(inertia[0]), iyy=str(inertia[1]), izz=str(inertia[2]), ixy="0", ixz="0", iyz="0")
        parent_id = int(model.body_parentid[body_id])
        parent_name = "world" if parent_id == 0 else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
        joint_ids = list(range(model.body_jntadr[body_id], model.body_jntadr[body_id] + model.body_jntnum[body_id]))
        primary = [joint_id for joint_id in joint_ids if not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id).startswith("passive_")]
        joint_id = primary[0] if primary else None
        kind = "fixed" if joint_id is None else ("floating" if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE else "revolute")
        joint_name = name + "_fixed" if joint_id is None else mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint = ET.SubElement(robot, "joint", name=joint_name, type=kind)
        ET.SubElement(joint, "parent", link=parent_name)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=numbers(model.body_pos[body_id]), rpy=numbers(quaternion_rpy(model.body_quat[body_id])))
        if kind == "revolute":
            if np.linalg.norm(model.jnt_pos[joint_id]) > 1e-9:
                raise ValueError("URDF exporter requires zero local joint offsets; revise frame conversion")
            ET.SubElement(joint, "axis", xyz=numbers(model.jnt_axis[joint_id]))
            actuator_ids = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
            effort = float(np.max(np.abs(model.actuator_forcerange[actuator_ids]))) if len(actuator_ids) else 0.1
            ET.SubElement(joint, "limit", lower=str(model.jnt_range[joint_id, 0]), upper=str(model.jnt_range[joint_id, 1]), effort=str(effort), velocity="0.4")
            if joint_name == "arm_gripper_follower":
                ET.SubElement(joint, "mimic", joint="arm_gripper", multiplier="-1", offset="0")
        for geom_id in range(model.body_geomadr[body_id], model.body_geomadr[body_id] + model.body_geomnum[body_id]):
            geom_type = model.geom_type[geom_id]
            if geom_type not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CAPSULE):
                continue
            for tag in ("visual", "collision"):
                if tag == "collision" and model.geom_contype[geom_id] == model.geom_conaffinity[geom_id] == 0:
                    continue
                element = ET.SubElement(link, tag)
                ET.SubElement(element, "origin", xyz=numbers(model.geom_pos[geom_id]), rpy=numbers(quaternion_rpy(model.geom_quat[geom_id])))
                geometry = ET.SubElement(element, "geometry")
                size = model.geom_size[geom_id]
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    ET.SubElement(geometry, "box", size=numbers(2 * size))
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    ET.SubElement(geometry, "sphere", radius=str(size[0]))
                elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    ET.SubElement(geometry, "cylinder", radius=str(size[0]), length=str(2 * (size[1] + size[0])))
                else:
                    mesh_id = int(model.geom_dataid[geom_id])
                    mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                    source_mesh = ROOT / load_design()["source_model"]
                    original_geom = source_bodies[name].findall("geom")[geom_id - model.body_geomadr[body_id]]
                    original_quaternion = np.fromstring(original_geom.get("quat", "1 0 0 0"), sep=" ")
                    original_quaternion /= np.linalg.norm(original_quaternion)
                    element.find("origin").set("xyz", original_geom.get("pos", "0 0 0"))
                    element.find("origin").set("rpy", numbers(quaternion_rpy(original_quaternion)))
                    ET.SubElement(geometry, "mesh", filename=os.path.relpath(source_mesh.parent / "assets" / f"{mesh_name}.stl", Path(output).parent))
    ET.indent(robot)
    ET.ElementTree(robot).write(output, encoding="utf-8", xml_declaration=True)


def build_artifacts(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for mode in ("free", "fixture"):
        root = make_tree(mode)
        meshdir = Path(root.find("compiler").get("meshdir"))
        root.find("compiler").set("meshdir", os.path.relpath(meshdir, output))
        ET.indent(root)
        path = output / f"microduck-arm-v1-{mode}.xml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        compiled = mujoco.MjModel.from_xml_path(str(path.resolve()))
        if compiled.nu != 15:
            raise ValueError("Generated actuator contract is not 15-dimensional")
        paths.append(path)
    urdf = output / "microduck-arm-v1.urdf"
    export_urdf(compile_model(), urdf)
    paths.append(urdf)
    manifest = {"design_id": load_design()["design_id"], "spec_sha256": sha256(SPEC_PATH), "source_sha256": load_design()["source_model_sha256"], "status": "unvalidated_engineering_prototype", "files": {path.name: sha256(path) for path in paths}, "urdf_limitations": ["URDF removes colocated backlash DOFs; MJCF remains dynamics authority", "Capsules use conservative cylinder approximations", "URDF gear mimic is kinematic, not a simulated gearbox", "Joint origins match this model's zero joint offsets", "Relative mesh dependencies require the source repository assets", "No ROS controller or hardware deployment qualification"]}
    (output / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
