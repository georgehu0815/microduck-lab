"""Build the eighth-case report from saved measurements, not target artwork."""

from __future__ import annotations

import hashlib
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
RUNS = ROOT / "rlx/runs/studio/drawing"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def episode_summary(report):
    episodes = report["drawing_assessment"]["episodes"]
    return {
        "episodes": len(episodes),
        "passed_episodes": sum(bool(item["passed"]) for item in episodes),
        "mean_coverage": statistics.mean(item["coverage"] for item in episodes),
        "mean_precision": statistics.mean(item["precision"] for item in episodes),
        "mean_grasp_fraction": statistics.mean(item["grasp_fraction"] for item in episodes),
        "completed_episodes": sum(bool(item["completed"]) for item in episodes),
        "max_normal_force_n": max(item["max_normal_force_n"] for item in episodes),
        "passed": bool(report["drawing_assessment"]["passed"]),
    }


def zoom_drawing(report, title):
    image = Image.new("RGB", (720, 540), "#fffcef")
    draw = ImageDraw.Draw(image)
    points = report["drawing_payload"]["points"]
    for previous, current in zip(points, points[1:]):
        if previous[3] == current[3]:
            line = [(360 + point[1] * 11500, 280 - (point[2] - .2245) * 11500) for point in (previous, current)]
            draw.line(line, fill="#26292b", width=2)
    draw.text((18, 16), title, fill="#26292b")
    draw.text((18, 512), "Fixed 62.6 x 47.0 mm view; actual contact points, not reference strokes", fill="#555555")
    return image


def make_gif(source, destination):
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(source),
        "-vf", "setpts=PTS/4,fps=6,scale=440:-1:flags=lanczos,split[main][palette];[palette]palettegen[colors];[main][colors]paletteuse",
        "-loop", "0", str(destination),
    ], check=True)


def plot_training(run, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (run / "bc_history.csv").open() as stream:
        bc_rows = list(csv.DictReader(stream))
    with (run / "ppo_history.csv").open() as stream:
        ppo_rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(range(1, len(bc_rows)+1), [float(row["loss"]) for row in bc_rows])
    axes[0, 0].set(title="BC + DAgger supervised loss", xlabel="Cumulative supervised epoch", ylabel="Action MSE", yscale="log")
    steps = [float(row["timesteps"]) for row in ppo_rows]
    axes[0, 1].plot(steps, [float(row["mean_rollout_reward"]) for row in ppo_rows])
    axes[0, 1].set(title="On-policy sampled reward", xlabel="PPO transitions", ylabel="Mean reward per control step")
    axes[1, 0].plot(steps[1:], [float(row["loss"]) for row in ppo_rows[1:]], color="#d55e00")
    axes[1, 0].set(title="PPO combined loss (previous update)", xlabel="PPO transitions", ylabel="SB3 final minibatch loss")
    axes[1, 1].plot(steps[1:], [float(row["approx_kl"]) for row in ppo_rows[1:]], color="#007f5f")
    axes[1, 1].set(title="Policy update size", xlabel="PPO transitions", ylabel="Approximate KL (previous update)")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    figure.suptitle(f"{run.name} — training diagnostics, NOT task acceptance")
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def main():
    from rlx.environments.drawing import drawing_xml
    from rlx.environments.drawing_reference import reference_svg

    media = OUTPUT / "media"
    media.mkdir(exist_ok=True)
    teacher = json.loads((OUTPUT / "teacher-render/render.json").read_text())
    assert teacher["controller"] == "teacher" and teacher["checkpoint_used"] is False
    runs = []
    for name in ("drawing-pilot-01", "drawing-refinement-02"):
        run = RUNS / name
        report = json.loads((run / "eval.json").read_text())
        metadata = json.loads((run / "policy.zip.json").read_text())
        assert report["source_sha256"] == digest(run / "policy.onnx")
        assert report["metadata_sha256"] == digest(run / "policy.zip.json")
        record = {
            "run": name,
            "training_config": metadata["training_config"],
            "timesteps": metadata["total_timesteps"],
            "assistance_curriculum": metadata["assistance_curriculum"],
            "training_source_hashes": metadata["source_hashes"],
            "ppo": episode_summary(report),
            "bc": episode_summary(report["bc_baseline"]),
            "comparison": report["comparison"],
            "evaluation_sha256": digest(run / "eval.json"),
            "policy_sha256": digest(run / "policy.onnx"),
            "checkpoint_sha256": digest(run / "policy.zip"),
            "render": json.loads((run / "render/render.json").read_text())["drawing_assessment"],
        }
        runs.append(record)
        plot_training(run, media / f"{name}-curves.png")
    chosen = RUNS / "drawing-refinement-02"
    learned_render = json.loads((chosen / "render/render.json").read_text())
    assert learned_render["controller"] == "onnx"
    zoom_drawing(teacher, "Jacobian teacher - NOT learned PPO").save(media / "teacher-drawing.png")
    zoom_drawing(learned_render, "Learned PPO - diagnostic, NOT accepted").save(media / "ppo-drawing.png")
    make_gif(OUTPUT / "teacher-render/rollout.mp4", media / "teacher.gif")
    make_gif(chosen / "render/rollout.mp4", media / "drawing.gif")
    public = ROOT / "duck-viewer/public/experiments/drawing"
    public.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(chosen / "render/rollout.mp4", public / "rollout.mp4")
    shutil.copyfile(chosen / "render/actual_contact_drawing.svg", public / "preview.svg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(chosen / "render/rollout.mp4"), "-frames:v", "1", str(public / "poster.jpg")], check=True)
    (OUTPUT / "scene.xml").write_text(drawing_xml())
    (OUTPUT / "reference.svg").write_text(reference_svg())
    software = {name: version(name) for name in ("mujoco", "numpy", "torch", "stable-baselines3", "gymnasium", "onnx", "onnxruntime", "Pillow", "imageio", "imageio-ffmpeg", "matplotlib", "pytest")}
    (OUTPUT / "requirements-reproduction.txt").write_text("".join(f"{name}=={installed}\n" for name, installed in software.items()))
    software.update({"python": platform.python_version(), "platform": platform.platform()})
    save_json(OUTPUT / "environment.json", software)
    result = {
        "date": "2026-09-11", "author": "George Hu", "scenario": "drawing",
        "teacher": teacher["drawing_assessment"], "runs": runs,
        "learned_skill_accepted": all(run["ppo"]["passed"] for run in runs),
        "scope": "simulation-only, preloaded free pencil, independent 83/15 interface, strong XML servos",
        "teacher_video_sha256": digest(OUTPUT / "teacher-render/rollout.mp4"),
        "current_sources": {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / "rlx/rlx/environments/drawing.py",
            ROOT / "rlx/rlx/environments/drawing_reference.py",
            ROOT / "rlx/examples/ppo_microduck_drawing.py",
        )},
    }
    save_json(OUTPUT / "results.json", result)
    rows = []
    for run in runs:
        for kind in ("bc", "ppo"):
            value = run[kind]
            rows.append(f"| {run['run']} / {kind.upper()} | {value['passed_episodes']}/{value['episodes']} | {value['mean_coverage']:.1%} | {value['mean_precision']:.1%} | {value['completed_episodes']}/{value['episodes']} |")
    metrics = teacher["drawing_assessment"]
    text = f"""# 第八案例结果 / Drawing results

Author: George Hu · 2026-09-11

## 结论 / Conclusion

完整绘画案例、物料场景、数据生成、BC/DAgger/PPO 训练、独立 ONNX 导出、评估、视频和 Viewer 接口已经实现。
**学习策略尚未通过完整绘画验收。教师成功不能当作 PPO 成功。**

The complete simulation/training/evaluation pipeline exists. **Neither learned PPO trial is accepted.**
The Jacobian teacher establishes feasibility only; it is not a trained policy.

## 教师可行性 / Teacher feasibility

Full 32-second, unassisted, seed-4 rollout: coverage **{metrics['coverage']:.2%}**,
precision **{metrics['precision']:.2%}**, symmetric Chamfer **{metrics['symmetric_chamfer_m']*1000:.3f} mm**,
grasp retention **{metrics['grasp_fraction']:.2%}**, peak force **{metrics['max_normal_force_n']:.3f} N**.
This is one nominal feasibility rollout, not generalization evidence or hardware validation.

![Actual teacher contact drawing](media/teacher-drawing.png)

## 两次真实训练 / Two actual training trials

Each trial performs **32,768 on-policy PPO transitions** after supervised warm-start;
total PPO transitions: **65,536**. Evaluations run the deterministic exported ONNX.
Case families: nominal, shifted target, 90% target scale, and reduced pad friction.
BC and PPO use matching evaluation seeds/configurations within each report.

| Run / controller | Accepted episodes | Mean coverage | Mean precision | Full episodes |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

`pilot-01`: BC 40 epochs, 2 DAgger rounds, support curriculum 1 → 0.3 → 0.
`refinement-02`: BC 200 epochs, 4 DAgger rounds, spread teacher targets, no base support,
smaller PPO learning rate and exploration standard deviation. Exact settings/hashes are in [results.json](results.json).
Do not compare different seeds as a causal PPO improvement experiment. Both stored matched BC/PPO
comparisons decline in coverage; `ppo_improvement_claimed` remains false.

![Actual PPO contact drawing, diagnostic](media/ppo-drawing.png)
![Pilot reward and loss](media/drawing-pilot-01-curves.png)
![Refinement reward and loss](media/drawing-refinement-02-curves.png)

## 诊断 / Diagnosis

- **观察 / Observed:** the contact-only teacher completes the drawing; releasing the jaw drops the pencil.
- **观察 / Observed:** BC imitates local actions more accurately than the PPO refinements reproduce the whole drawing.
- **观察 / Observed:** both learned evaluations fail the strict full-episode geometry/contact/balance gates.
- **推断 / Inference:** behavior-cloning distribution shift and PPO exploration/update drift remain important risks.
  This is not a proven single root cause; the next experiment should compare a residual actor, pressure control,
  and conservative policy updates with matched seeds, while preserving the same acceptance rules.
- No target pixels, welds, hidden pencil attachments, or teacher actions are used during PPO evaluation.
  The pencil remains a free MuJoCo body; the old head collision meshes cannot clamp it.

## 文件与复现 / Evidence and reproduction

- [Chinese workflow](README.md) / [English workflow](README.en.md): BOM, exact commands, assumptions, algorithm and contracts.
- [Teacher video](teacher-render/rollout.mp4), [contact sheet](teacher-render/frame_sheet.png), [render report](teacher-render/render.json).
- Learned run artifacts: `rlx/runs/studio/drawing/drawing-pilot-01/` and `drawing-refinement-02/`.
- Each run includes datasets, provenance sidecars, BC/PPO checkpoints, ONNX, CSV training histories,
  reward/loss plots, source-bound evaluation and actual-contact render files.
- [Eight-case browser receipt](evidence/verification.json), [runtime versions](environment.json), [generated scene](scene.xml).
  Regenerate the scene after checkout: its mesh directory points to the current workspace.
- To regenerate this measured report and the README GIFs from existing artifacts:
  `PYTHONPATH=rlx:microduck_local/src rlx/.venv-microduck/bin/python docs/drawing-case/build_report.py`.

The GIFs are **4× time-compressed** versions of actual rollouts, not real-time robot motion.
Reference target artwork is [reference.svg](reference.svg); it is never substituted for the measured ink.
The separate `microduck-drawing-v1` 83/15 interface, preloaded pencil and stronger XML gains are
simulation assumptions. No stock-robot deployment, autonomous pickup, or physical material purchase is claimed.
"""
    (OUTPUT / "RESULTS.md").write_text(text)
    print(json.dumps({"output": str(OUTPUT), "teacher_passed": metrics["passed"], "learned_skill_accepted": result["learned_skill_accepted"]}))


if __name__ == "__main__":
    main()
