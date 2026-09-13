from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from .config import CASES, DESIGN_ID, ROOT, SPEC_PATH, load_design, sha256
from .env import IntegratedArmEnv
from .model import compile_model, make_tree


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def provenance():
    files = [SPEC_PATH, Path(__file__), ROOT / "microduck_arm_v1/model.py", ROOT / "microduck_arm_v1/env.py", ROOT / "microduck_arm_v1/config.py"]
    return {"design_id": DESIGN_ID, "files": {str(path.relative_to(ROOT)): sha256(path) for path in files}, "source_mjcf_sha256": load_design()["source_model_sha256"], "mujoco_version": mujoco.__version__}


def validate_model():
    design = load_design()
    model = compile_model()
    fixture = compile_model("fixture")
    root = make_tree()
    actuator_names = [model.actuator(index).name for index in range(model.nu)]
    checks = {
        "15_action_order": actuator_names == design["joint_order"],
        "floating_robot_root": model.jnt_type[model.joint("trunk_base_freejoint").id] == mujoco.mjtJoint.mjJNT_FREE,
        "fixture_explicitly_different": fixture.nq == model.nq - 7 and fixture.nv == model.nv - 6,
        "positive_link_mass_and_inertia": bool(np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)),
        "finite_dynamics_arrays": bool(np.isfinite(model.body_inertia).all() and np.isfinite(model.body_mass).all()),
        "no_object_or_root_weld": root.findall(".//weld") == [],
        "only_mechanical_gripper_equality": len(root.find("equality")) == 1 and root.find("equality/joint").get("name") == "gripper_gear_coupling",
        "no_neck_actuators": not any(name in actuator_names for name in ("head_pitch", "neck_pitch", "head_yaw", "head_roll")),
        "arm_link_length_contract": abs(sum(design["arm"]["link_lengths_m"]) - .135) < 1e-12,
        "physical_gripper_hinges": all(model.jnt_type[model.joint(name).id] == mujoco.mjtJoint.mjJNT_HINGE for name in ("arm_gripper", "arm_gripper_follower")),
        "five_motor_envelopes": len([geom for geom in root.iter("geom") if geom.get("name", "").startswith("arm_") and geom.get("name", "").endswith("_motor")]) == 5,
        "arm_torque_is_not_stall_rating": bool(np.allclose(model.actuator_forcerange[10:], [-.10, .10])),
    }
    original = mujoco.MjModel.from_xml_path(str(ROOT / design["source_model"]))
    neck = original.body("neck").id
    removed_mass = 0.0
    for body_id in range(1, original.nbody):
        parent = body_id
        while parent > 0 and parent != neck:
            parent = int(original.body_parentid[parent])
        if parent == neck:
            removed_mass += original.body_mass[body_id]
    robot_mass = float(model.body_mass.sum() - model.body_mass[model.body("task_object").id])
    return {"scope": "simulation_model_structure_only", "passed": all(checks.values()), "checks": checks,
            "provenance": provenance(), "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "mass_ledger": {"source_robot_kg": float(original.body_mass.sum()), "removed_modeled_neck_subtree_kg": float(removed_mass), "retained_modeled_structure_kg": float(original.body_mass.sum() - removed_mass), "new_robot_kg": robot_mass, "new_components_including_relocated_electronics_kg": robot_mass - float(original.body_mass.sum() - removed_mass), "all_added_masses_unmeasured": True},
            "hardware_release": False, "packaging_verified": False, "task_success_proven": False}


def torque_com_table(output, samples=100, seed=101, payload_kg=.020):
    if samples < 1:
        raise ValueError("samples must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = IntegratedArmEnv(mode="fixture", payload_kg=payload_kg)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    translation, rotation = np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv))
    try:
        for sample in range(samples):
            angles = env.home[10:14] if sample == 0 else rng.uniform(env.joint_limits[10:14, 0], env.joint_limits[10:14, 1])
            env.data.qpos[env.qpos_indices[10:14]] = angles
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            mujoco.mj_jacSite(env.model, env.data, translation, rotation, env.tcp_id)
            torque = env.data.qfrc_bias[env.dof_indices[10:14]] + translation[:, env.dof_indices[10:14]].T @ np.array([0, 0, payload_kg * 9.81])
            robot_com = np.average(env.data.xipos[env.robot_bodies], weights=env.model.body_mass[env.robot_bodies], axis=0)
            loaded_com = (robot_com * env.robot_mass + env.data.site_xpos[env.tcp_id] * payload_kg) / (env.robot_mass + payload_kg)
            row = {"sample": sample, "payload_kg": payload_kg, "robot_mass_kg": env.robot_mass}
            row.update({f"joint_{index}_rad": float(value) for index, value in enumerate(angles)})
            row.update({f"joint_{index}_gravity_payload_nm": float(value) for index, value in enumerate(torque)})
            row.update({f"loaded_com_{axis}_m": float(value) for axis, value in zip("xyz", loaded_com)})
            row["max_required_qualified_continuous_nm_at_3x"] = 3 * float(np.max(np.abs(torque)))
            row["collision_free_at_pose"] = not bool(env._contact_state()[2])
            rows.append(row)
    finally:
        env.close()
    with (output / "torque-com.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"scope": "quasistatic_screening_not_dynamic_or_hardware_validation", "samples": samples, "seed": seed, "payload_at_tcp_assumption_kg": payload_kg,
               "maximum_required_qualified_continuous_nm_at_3x": max(row["max_required_qualified_continuous_nm_at_3x"] for row in rows),
               "qualified_continuous_capability_nm": None, "torque_release_passed": False,
               "notes": ["Uses qfrc_bias and TCP Jacobian; no payload weld", "Includes infeasible/colliding poses with explicit flag; not a rated operating envelope", "No acceleration, tool contact or motor thermal qualification", "Masses and packaging include provisional assumptions"], "provenance": provenance()}
    write_json(output / "torque-com-summary.json", summary)
    return summary


def workspace_scan(output, samples=1000, seed=101):
    if samples < 1:
        raise ValueError("samples must be positive")
    env = IntegratedArmEnv(mode="fixture")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    reachable, collision_free = 0, 0
    points = []
    try:
        for index in range(samples):
            target = rng.uniform([.03, -.08, .09], [.17, .08, .22])
            angles = env._inverse_kinematics(target, initial=env.home[10:14])
            env.data.qpos[env.qpos_indices[10:14]] = angles
            mujoco.mj_forward(env.model, env.data)
            error = float(np.linalg.norm(target - env.data.site_xpos[env.tcp_id]))
            axis_error = abs(float(env.data.site_xmat[env.tcp_id].reshape(3, 3)[2, 0]))
            reached = error < .002 and axis_error < .02
            forbidden = env._contact_state()[2]
            reachable += int(reached)
            collision_free += int(reached and not forbidden)
            points.append({"index": index, "target_m": target.tolist(), "q_rad": angles.tolist(), "error_m": error, "tool_forward_z": axis_error, "ik_reached": reached, "forbidden_contacts": sorted(forbidden)})
    finally:
        env.close()
    result = {"scope": "fixture_kinematic_horizontal_tool_screen_only", "samples": samples, "seed": seed, "ik_reached": reachable, "ik_and_collision_screen_passed": collision_free, "balance_safe": None, "usable_workspace_certified": False,
              "orientation_constraint": "tool_forward_horizontal_only_not_arbitrary_6D_pose", "sampling_box_m": [[.03, -.08, .09], [.17, .08, .22]],
              "limitations": ["Restricted horizontal-tool IK is not arbitrary task-orientation workspace", "No free-base dynamics or thermal/torque qualification", "Sampled coverage is not an exhaustive reachability proof"], "provenance": provenance(), "points": points}
    write_json(output, result)
    return {key: value for key, value in result.items() if key != "points"}


def rollout(case, mode, seed, max_steps=1500, controller="teacher", video_path=None):
    if controller not in ("teacher", "zero", "no_actuation", "open_gripper"):
        raise ValueError("controller must be teacher, zero, no_actuation or open_gripper")
    env = IntegratedArmEnv(case=case, mode=mode, max_steps=max_steps)
    writer = None
    observation, info = env.reset(seed=seed)
    if controller == "no_actuation":
        env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    trace, reward_sum = [], 0.0
    try:
        if video_path is not None:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(video_path), fps=25, codec="libx264", quality=7)
        for step in range(max_steps):
            action = np.zeros(15, dtype=np.float32) if controller in ("zero", "no_actuation") else env.teacher_action()
            if controller == "open_gripper":
                action[-1] = 1.0
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            trace.append({"step": step, "reward": reward, "action": action.tolist(), "diagnostics": info["diagnostics"], "failure_reason": info["failure_reason"]})
            if writer is not None and (step % 2 == 0 or terminated or truncated):
                image = Image.fromarray(env.render())
                painter = ImageDraw.Draw(image)
                painter.rectangle((0, 0, 640, 62), fill=(12, 18, 25))
                painter.text((10, 6), f"SIMULATION ONLY | {case} | {mode} | {controller}", fill="white")
                painter.text((10, 24), f"t={env.data.time:.2f}s  seed={seed}  success={info['success']}  {info['failure_reason'] or 'running'}", fill="white")
                painter.text((10, 42), "FIXTURE-ASSISTED: NOT MOBILE VALIDATION" if mode == "fixture" else "FLOATING BASE: NO HARDWARE VALIDATION", fill=(255, 190, 60))
                writer.append_data(np.asarray(image))
            if terminated or truncated:
                break
        record = {"case": case, "mode": mode, "controller": controller, "seed": seed, "steps": len(trace), "return": reward_sum,
                  "success": info["success"], "failure_reason": info["failure_reason"], "final_diagnostics": info["diagnostics"],
                  "simulation_only": True, "hardware_ready": False, "provenance": provenance(), "trace": trace}
    finally:
        if writer is not None:
            writer.close()
        env.close()
    if video_path is not None:
        record["video"] = {"path": str(video_path), "sha256": sha256(video_path), "frames_include_full_rollout_at_25fps": True}
    return record


def evaluate_suite(output, seeds=(910000, 910001), max_steps=1500, videos=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in ("free", "fixture"):
        cases = CASES if mode == "free" else CASES[:4]
        for case in cases:
            for seed in seeds:
                label = f"{mode}-{case}-seed-{seed}"
                video = output / f"{label}.mp4" if videos and seed == seeds[0] else None
                record = rollout(case, mode, seed, max_steps, video_path=video)
                write_json(output / f"{label}.json", record)
                records.append({key: value for key, value in record.items() if key != "trace"})
    summary = {"scope": "exploratory_simulation_case_evaluation_not_preregistered_release_suite", "records": records,
               "free_base": {"passed": sum(item["success"] for item in records if item["mode"] == "free"), "episodes": sum(item["mode"] == "free" for item in records)},
               "fixture": {"passed": sum(item["success"] for item in records if item["mode"] == "fixture"), "episodes": sum(item["mode"] == "fixture" for item in records)},
               "all_robot_cases_passed": all(item["success"] for item in records if item["mode"] == "free"), "hardware_release": False, "provenance": provenance()}
    write_json(output / "evaluation.json", summary)
    return summary


def hardware_acceptance(evidence_path=None):
    design = load_design()
    required = list(design["unresolved_release_requirements"])
    record = {} if evidence_path is None else json.loads(Path(evidence_path).read_text())
    checks = {"baseline_matches": record.get("design_id") == DESIGN_ID, "spec_hash_matches": record.get("spec_sha256") == sha256(SPEC_PATH)}
    evidence_root = Path(evidence_path).resolve().parent if evidence_path else ROOT
    for name in required:
        item = record.get("requirements", {}).get(name, {})
        evidence = item.get("evidence", [])
        valid = isinstance(evidence, list) and len(evidence) > 0 and item.get("reviewer") and item.get("procedure") and item.get("result") == "pass"
        for artifact in evidence if isinstance(evidence, list) else []:
            if not isinstance(artifact, dict):
                valid = False
                continue
            path = evidence_root / artifact.get("path", "")
            valid = bool(valid and path.is_file() and artifact.get("sha256") == sha256(path))
        checks[name] = bool(valid)
    return {"scope": "hardware_evidence_completeness_check_not_physical_test", "evidence_complete": all(checks.values()), "checks": checks,
            "hardware_release": False, "physical_tests_executed_by_this_command": False,
            "required_next_step": "independent engineering review of real measured evidence; no actuator I/O is implemented"}
