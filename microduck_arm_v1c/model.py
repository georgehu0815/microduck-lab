from __future__ import annotations

import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from microduck_arm_v1.model import make_tree as legacy_tree, export_urdf, numbers
from .config import DESIGN_ID, SPEC_PATH, load_design, sha256


def make_tree(mode="free", payload_kg=0.01, candidate="A"):
    design = load_design(candidate)
    root = legacy_tree(mode, payload_kg)
    root.set("model", f"{DESIGN_ID}-{candidate}-{mode}")
    root.find("option").set("noslip_iterations", str(design["simulation_solver"]["noslip_iterations"]))
    arm = design["arm"]
    root.find(".//body[@name='arm_upper']").set("pos", numbers(arm["shoulder_pivot_local_m"]))
    for index, motor in enumerate(("yaw", "shoulder", "elbow", "wrist", "gripper")):
        geom = root.find(f".//geom[@name='arm_{motor}_motor']")
        sku = arm["motor_skus_by_motor"][f"M{index + 1}"]
        geom.set("mass", str(design["motor_catalog"][sku]["mass_kg"]))
        geom.set("pos", numbers(arm["servo_centers_local_m"][motor]))
    root.find(".//body[@name='arm_mount']").set("pos", numbers(arm["mount_pos_m"]))
    root.find(".//body[@name='arm_forearm']").set("pos", numbers([arm["link_lengths_m"][0], 0, 0]))
    root.find(".//body[@name='arm_hand']").set("pos", numbers([arm["link_lengths_m"][1], 0, 0]))
    for side in ("left", "right"):
        properties = arm["contact_pads"]
        pad = root.find(f".//geom[@name='arm_pad_{side}']")
        del pad.attrib["fromto"]
        pad.set("type", "box")
        pad.set("pos", numbers(properties["center_local_m"]))
        pad.set("size", numbers(np.asarray(properties["full_size_m"]) / 2))
        pad.set("mass", str(properties["mass_each_kg"]))
        pad.set("solref", numbers(properties["solref"]))
        pad.set("solimp", numbers(properties["solimp"]))
    scene = design.get("simulation_scene", {})
    surface_top = scene.get("surface_top_m", .145)
    root.find(".//geom[@name='work_surface']").set("pos", numbers([.13, 0, surface_top - .005]))
    root.find(".//geom[@name='obstacle']").set("pos", numbers([.145, 0, surface_top + .017]))
    root.find(".//body[@name='task_object']").set("pos", numbers([.115, -.018, surface_top + .006]))
    for index, actuator in enumerate(root.find("actuator")[10:]):
        actuator.set("forcerange", numbers([-arm["simulation_force_limit_nm"], arm["simulation_force_limit_nm"]]))
        actuator.set("kp", str(arm["simulation_kp"]))
        actuator.set("kv", str(arm["simulation_kv"]))
    ET.SubElement(root.find("custom"), "text", name="actuator_candidate", data=candidate)
    ET.SubElement(root.find("custom"), "text", name="qualification", data="analytical_inertias_identical_AB_screening_limits_not_measured")
    return root


def compile_model(mode="free", payload_kg=.01, candidate="A"):
    return mujoco.MjModel.from_xml_string(ET.tostring(make_tree(mode, payload_kg, candidate), encoding="unicode"))


def mass_properties(model):
    records = []
    for body_id in range(1, model.nbody):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.body_iquat[body_id])
        rotation = rotation.reshape(3, 3)
        tensor = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        records.append({"body": model.body(body_id).name, "mass_kg": float(model.body_mass[body_id]),
                        "com_local_m": model.body_ipos[body_id].tolist(), "inertia_body_frame_kg_m2": tensor.tolist(),
                        "status": "compiled_model_estimate_not_measured"})
    return records


def build_artifacts(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for candidate in ("A", "B"):
        for mode in ("free", "fixture"):
            root = make_tree(mode=mode, candidate=candidate)
            root.find("compiler").set("meshdir", os.path.relpath(root.find("compiler").get("meshdir"), output))
            ET.indent(root)
            path = output / f"microduck-arm-v1c-{candidate}-{mode}.xml"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            compiled = mujoco.MjModel.from_xml_path(str(path.resolve()))
            if compiled.nu != 15:
                raise ValueError("Invalid v1-C actuator contract")
            files.append(path)
        urdf = output / f"microduck-arm-v1c-{candidate}.urdf"
        export_urdf(compile_model(candidate=candidate), urdf)
        tree = ET.parse(urdf)
        tree.getroot().set("name", f"{DESIGN_ID}-{candidate}")
        tree.write(urdf, encoding="utf-8", xml_declaration=True)
        files.append(urdf)
    properties = output / "mass-com-inertia.json"
    properties.write_text(json.dumps(mass_properties(compile_model()), indent=2) + "\n")
    files.append(properties)
    manifest = {"design_id": DESIGN_ID, "spec_sha256": sha256(SPEC_PATH), "files": {path.name: sha256(path) for path in files},
                "hardware_ready": False, "fabrication_ready": False,
                "AB_limitations": "Equal mass and unqualified equal dynamics; cannot infer energy/torque superiority.",
                "urdf_limitations": "Backlash DOFs collapsed; capsules approximated by conservative cylinders; relative upstream meshes required."}
    (output / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
