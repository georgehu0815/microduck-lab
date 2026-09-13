from __future__ import annotations

import csv
import json
from pathlib import Path

import mujoco
import numpy as np

from .config import CASES, DESIGN_ID, ROOT, SPEC_PATH, load_design, sha256
from .env import IntegratedArmEnv
from .model import compile_model
from .safety import evaluate_faults


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def provenance():
    sources = sorted((ROOT / "microduck_arm_v1c").glob("*.py"))
    sources += sorted((ROOT / "microduck_arm_v1").glob("*.py"))
    sources += [SPEC_PATH, ROOT / load_design()["source_model"]]
    sources += [ROOT / "microduck_local/src/microduck_local/contract.py"]
    sources += [ROOT / "microduck_local/policies" / name for name in ("alpha_stand.onnx", "alpha_walking.onnx")]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sources}


def assert_unchanged(snapshot):
    if snapshot != provenance():
        raise RuntimeError("Source changed during execution; discard this run and repeat with frozen sources")


def rollout(case, mode, candidate="A", seed=0, payload_kg=.01, max_steps=2500, output=None, control="teacher", video=None, solver_iterations=None, physics_dt=None):
    source_snapshot = provenance()
    if control not in ("teacher", "null", "open_jaw"):
        raise ValueError("Unknown controller")
    env = IntegratedArmEnv(case=case, mode=mode, candidate=candidate, max_steps=max_steps, payload_kg=payload_kg)
    if solver_iterations is not None:
        if not isinstance(solver_iterations, int) or not 0 <= solver_iterations <= 100:
            env.close()
            raise ValueError("solver_iterations must be an integer in [0, 100]")
        env.model.opt.noslip_iterations = solver_iterations
    if physics_dt is not None:
        if physics_dt not in (.001, .002):
            env.close()
            raise ValueError("Convergence screening supports only 1 ms or 2 ms physics")
        env.model.opt.timestep = physics_dt
        env.substeps = round(env.dt / physics_dt)
    writer = None
    frame_count = 0
    telemetry = []
    mechanical_work = 0.
    try:
        if video is not None:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw
            video = Path(video)
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(video, fps=25, codec="libx264", macro_block_size=16)
        observation, info = env.reset(seed=seed)
        for step in range(max_steps):
            action = env.teacher_action() if control != "null" else env.normalize_targets(env.home)
            if control == "open_jaw":
                targets = env.denormalize_action(action)
                targets[-1] = 0
                action = env.normalize_targets(targets)
            if video is not None and step % 2 == 0:
                frame = Image.fromarray(env.render())
                drawing = ImageDraw.Draw(frame)
                drawing.rectangle((0, 0, 640, 26), fill="black")
                drawing.text((8, 7), f"SIMULATION ONLY | v1-C {candidate} | {mode} | {case} | t={env.data.time:.2f}s", fill="white")
                writer.append_data(np.asarray(frame))
                frame_count += 1
            observation, reward, terminated, truncated, info = env.step(action)
            power = float(np.sum(np.abs(env.data.actuator_force * env.data.qvel[env.dof_indices])))
            mechanical_work += power * env.dt
            telemetry.append({"step": step + 1, "reward": reward, "phase": env.phase,
                              "action": action.tolist(), "command_targets_rad": env.command_targets.tolist(),
                              "joint_position_rad": env.data.qpos[env.qpos_indices].tolist(),
                              "joint_velocity_rad_s": env.data.qvel[env.dof_indices].tolist(),
                              "mechanical_absolute_power_w": power, **info["diagnostics"]})
            if terminated or truncated:
                break
        result = {**info, "seed": seed, "payload_kg": payload_kg, "steps": env.steps, "control": control,
                  "solver": {"noslip_iterations": env.model.opt.noslip_iterations, "physics_dt_s": env.model.opt.timestep},
                  "ever_grasped": env.ever_grasped, "ever_lifted": env.ever_lifted,
                  "waypoints_completed": env.waypoints_completed, "support_events": env.support_events,
                  "mechanical_absolute_work_j": mechanical_work, "electrical_energy_j": None,
                  "energy_note": "Mechanical work is not battery consumption; motor losses and regeneration unqualified.",
                  "source_hashes": source_snapshot}
        if video is not None:
            writer.close()
            writer = None
            result["video"] = {"path": str(video), "sha256": sha256(video), "frames": frame_count, "fps": 25}
        if output:
            assert_unchanged(source_snapshot)
            output = Path(output)
            write_json(output / "result.json", result)
            with (output / "telemetry.jsonl").open("w") as handle:
                for record in telemetry:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
        assert_unchanged(source_snapshot)
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


def evaluate(output, seeds=range(10), candidates=("A", "B"), max_steps=2500):
    seeds, candidates = tuple(seeds), tuple(candidates)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Evaluation requires distinct nonnegative integer seeds")
    if not candidates or len(set(candidates)) != len(candidates) or any(candidate not in ("A", "B") for candidate in candidates):
        raise ValueError("Evaluation requires distinct A/B candidates")
    source_snapshot = provenance()
    output = Path(output)
    records = []
    for candidate in candidates:
        for mode in ("fixture", "free"):
            for case in CASES:
                if mode == "fixture" and "walking" in case:
                    continue
                for seed in seeds:
                    path = output / f"{candidate}-{mode}-{case}-{seed}"
                    records.append(rollout(case, mode, candidate, seed, max_steps=max_steps, output=path))
    negatives = []
    for control in ("null", "open_jaw"):
        negatives.append(rollout("pick_place", "fixture", control=control, max_steps=max_steps, output=output / control))
    by_mode = {mode: {"passed": sum(record["success"] for record in records if record["mode"] == mode),
                      "total": sum(record["mode"] == mode for record in records)} for mode in ("fixture", "free")}
    result = {"design_id": DESIGN_ID, "scenario": "raised_table_v1c_10g_screening",
              "records": records, "by_mode": by_mode, "negative_controls": negatives,
              "negative_controls_passed": all(not record["success"] for record in negatives),
              "all_tasks_passed": all(record["success"] for record in records),
              "hardware_release": False, "source_hashes": source_snapshot}
    assert_unchanged(source_snapshot)
    write_json(output / "evaluation.json", result)
    return result


def contact_convergence(output, seeds=(0, 2, 9)):
    seeds = tuple(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Convergence screening requires nonnegative integer seeds")
    snapshot = provenance()
    output = Path(output)
    records = []
    for iterations, timestep in ((0, .002), (5, .002), (10, .002), (20, .002), (10, .001)):
        for seed in seeds:
            record = rollout(
                "pick_place", "free", seed=seed,
                solver_iterations=iterations, physics_dt=timestep,
                output=output / f"noslip-{iterations}-dt-{timestep}-seed-{seed}",
            )
            records.append(record)
    result = {
        "purpose": "contact_solver_sensitivity_not_physical_friction_qualification",
        "records": records,
        "refined_settings_passed": all(record["success"] for record in records if record["solver"]["noslip_iterations"] > 0),
        "unchanged": ["geometry", "mass", "friction", "actuator_force_limits", "task_success_criteria"],
        "hardware_release": False,
        "source_hashes": snapshot,
    }
    assert_unchanged(snapshot)
    write_json(output / "contact-convergence.json", result)
    return result


def torque_com_table(output, samples=200, seed=101):
    source_snapshot = provenance()
    if samples < 1:
        raise ValueError("samples must be positive")
    env = IntegratedArmEnv(mode="fixture")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    translation = np.zeros((3, env.model.nv))
    rotation = np.zeros_like(translation)
    inertia = np.zeros((env.model.nv, env.model.nv))
    try:
        for sample in range(samples):
            env.data.qpos[env.qpos_indices[10:14]] = env.home[10:14] if sample == 0 else rng.uniform(env.joint_limits[10:14, 0], env.joint_limits[10:14, 1])
            env.data.qvel[:] = 0
            mujoco.mj_forward(env.model, env.data)
            mujoco.mj_fullM(env.model, env.data, inertia)
            mujoco.mj_jacSite(env.model, env.data, translation, rotation, env.tcp_id)
            jacobian = translation[:, env.dof_indices[10:]]
            robot_com = np.average(env.data.xipos[env.robot_bodies], weights=env.model.body_mass[env.robot_bodies], axis=0)
            for payload in (0., .01, .02, .03, .05):
                gravity = env.data.qfrc_bias[env.dof_indices[10:]] + jacobian.T @ [0, 0, payload * 9.81]
                selected_inertia = inertia[np.ix_(env.dof_indices[10:], env.dof_indices[10:])] + payload * jacobian.T @ jacobian
                acceleration_envelope = np.sum(np.abs(selected_inertia), axis=1) * 2.0
                contact_envelope = np.sum(np.abs(jacobian), axis=0) * 1.0
                loaded_com = (robot_com * env.robot_mass + env.data.site_xpos[env.tcp_id] * payload) / (env.robot_mass + payload)
                for joint_index in range(5):
                    rows.append({"sample": sample, "joint": f"J{joint_index + 1}", "payload_kg": payload,
                                 "gravity_nm": float(gravity[joint_index]),
                                 "inertia_acceleration_bound_nm_at_2rad_s2": float(acceleration_envelope[joint_index]),
                                 "tcp_force_bound_nm_at_1N_per_axis": float(contact_envelope[joint_index]),
                                 "required_qualified_torque_nm_at_3x": 3 * float(abs(gravity[joint_index]) + acceleration_envelope[joint_index] + contact_envelope[joint_index]),
                                 "gripper_squeeze_load_included": False,
                                 "loaded_com_x_m": loaded_com[0], "loaded_com_y_m": loaded_com[1], "loaded_com_z_m": loaded_com[2],
                                 "collision_free": not bool(env._contact_state()[2])})
    finally:
        env.close()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "joint-torque-com.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"samples": samples, "rows": len(rows), "payloads_kg": [0, .01, .02, .03, .05],
               "scope": "inverse_dynamics_screening_not_task_or_thermal_qualification",
               "assumptions": ["point payload at TCP, not weld", "2rad/s2 independent joint acceleration bounds", "1N per world axis TCP contact bound", "zero velocity; no Coriolis envelope", "gripper squeeze and compliance unqualified"],
               "qualified_continuous_torque_nm": None, "torque_release_passed": False, "source_hashes": source_snapshot}
    assert_unchanged(source_snapshot)
    write_json(output / "torque-summary.json", summary)
    return summary


def workspace_scan(output, samples=1000, seed=101):
    source_snapshot = provenance()
    if samples < 1:
        raise ValueError("samples must be positive")
    env = IntegratedArmEnv(mode="fixture")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    rows = []
    try:
        for sample in range(samples):
            target = rng.uniform([.025, -.08, .145], [.17, .08, .25])
            angles = env._inverse_kinematics(target, initial=env.home[10:14])
            env.data.qpos[env.qpos_indices[10:14]] = angles
            mujoco.mj_forward(env.model, env.data)
            error = float(np.linalg.norm(target - env.data.site_xpos[env.tcp_id]))
            reached = error < .002 and abs(env.data.site_xmat[env.tcp_id].reshape(3, 3)[2, 0]) < .02
            rows.append({"sample": sample, "target_m": target.tolist(), "position_error_m": error, "reachable": bool(reached),
                         "collision_free": not bool(env._contact_state()[2]), "balance_qualified": False})
    finally:
        env.close()
    result = {"samples": samples, "reachable": sum(row["reachable"] for row in rows),
              "reachable_collision_free": sum(row["reachable"] and row["collision_free"] for row in rows),
              "balance_safe_qualified": None, "scope": "fixture_geometric_screening_not_dynamic_workspace",
              "points": rows, "source_hashes": source_snapshot}
    assert_unchanged(source_snapshot)
    write_json(Path(output) / "workspace.json", result)
    return result


def hardware_acceptance():
    model = compile_model()
    arm_body = model.body("arm_mount").id
    arm_mass = sum(model.body_mass[body_id] for body_id in range(arm_body, model.body("task_object").id))
    return {"design_id": DESIGN_ID, "hardware_release": False, "physical_tests_executed": False,
            "modeled_cartridge_kg": float(arm_mass), "cartridge_target_kg": .160,
            "model_cartridge_mass_target_met": bool(arm_mass <= .160),
            "gates": {name: {"passed": False, "reason": reason} for name, reason in {
                "fabrication": "Mating dimensions, bearing support and tolerances unverified",
                "mass_com_inertia": "Only analytical geometry and inherited mass placeholders",
                "torque": "No qualified continuous operating envelope at 3x margin",
                "power": "No selected pack/converter/protection thermal and fault qualification",
                "dynamic_balance": "Free-base task acceptance requires full evaluated motion",
                "hardware": "No robot connected; no physical tests executed",
                "deployment": "No qualified estimator/mobile interface; new policy contract not stock drop-in",
            }.items()}, "emulated_safety": evaluate_faults()}
