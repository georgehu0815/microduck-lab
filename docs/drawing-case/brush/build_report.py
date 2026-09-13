"""Publish measured brush artifacts; never substitute target geometry for paint."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(report):
    episodes = report["drawing_assessment"]["episodes"]
    return {"passed": report["passed"], "episodes": len(episodes), "accepted": sum(item["passed"] for item in episodes),
            "coverage": sum(item["coverage"] for item in episodes)/len(episodes),
            "precision": sum(item["precision"] for item in episodes)/len(episodes),
            "max_force_n": max(item["max_normal_force_n"] for item in episodes),
            "max_compression_mm": 1000*max(item["max_compression_m"] for item in episodes),
            "max_leak_fraction": max(item["pen_up_leak_fraction"] for item in episodes),
            "min_grasp_fraction": min(item["grasp_fraction"] for item in episodes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="brush-color-v2-03")
    args = parser.parse_args()
    run = ROOT / "rlx/runs/studio/drawing" / args.run
    evaluation, baseline = read(run / "eval.json"), read(run / "bc-eval.json")
    rendered, metadata = read(run / "render/render.json"), read(run / "policy.zip.json")
    assert evaluation["passed"] and not evaluation["provenance_errors"]
    assert baseline["passed"] and not baseline["provenance_errors"]
    assert evaluation["source_sha256"] == digest(run / "policy.onnx") == rendered["source_sha256"]
    assert rendered["assessment"]["passed"] and not rendered["teacher_diagnostic"]
    ppo, bc = summarize(evaluation), summarize(baseline)
    media = OUTPUT / "media"
    media.mkdir(exist_ok=True)
    for name in ("painting.png", "frame_sheet.png", "rollout.mp4", "raw_trace.csv", "raw_trace.json", "render.json"):
        shutil.copy2(run / "render" / name, media / name)
    for name in ("eval.json", "bc-eval.json", "training.json", "policy.zip.json"):
        shutil.copy2(run / name, OUTPUT / name)
    rows = list(csv.DictReader((run / "ppo_history.csv").open()))
    steps = [int(row["steps"]) for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, label in zip(axes.flat,
        ("reward_mean", "value_loss", "policy_gradient_loss", "anchor_max_drift"),
        ("Mean per-step rollout reward", "PPO value loss (separate scale)", "PPO policy-gradient loss", "Maximum action drift from BC anchor")):
        axis.plot(steps, [float(row[key]) for row in rows], marker="o")
        axis.set_title(label)
        axis.set_xlabel("PPO transitions")
        axis.grid(alpha=.25)
    axes[1, 1].axhline(.00025, color="red", linestyle="--", label="update bound")
    axes[1, 1].legend()
    figure.suptitle("Measured training history · BC warm-start + anchored PPO · George Hu")
    figure.savefig(media / "reward-loss.png", dpi=160)
    plt.close(figure)
    reader = imageio.get_reader(str(media / "rollout.mp4"))
    fps = float(reader.get_meta_data()["fps"])
    frames = []
    last = None
    for index, frame in enumerate(reader):
        last = frame
        if index % max(1, round(fps)) == 0:
            frames.append(Image.fromarray(frame).resize((480, 360)))
    reader.close()
    if last is not None:
        frames.append(Image.fromarray(last).resize((480, 360)))
        Image.fromarray(last).save(media / "poster.jpg", quality=92)
    frames[0].save(media / "brush.gif", save_all=True, append_images=frames[1:], duration=250, loop=0)
    public = ROOT / "duck-viewer/public/experiments/drawing"
    shutil.copy2(media / "poster.jpg", public / "poster.jpg")
    shutil.copy2(media / "rollout.mp4", public / "rollout.mp4")
    summary = {"author": "George Hu", "date": "2026-09-11", "run": args.run,
               "policy_sha256": digest(run / "policy.onnx"), "ppo": ppo, "bc": bc,
               "total_timesteps": metadata["total_timesteps"], "accepted_updates": metadata["accepted_ppo_updates"],
               "render": {"passed": rendered["assessment"]["passed"], "seconds": rendered["elapsed_seconds"], "fps": rendered["fps"]},
               "ppo_improvement_claimed": False, "fixed_authored_planner": True,
               "provenance": {
                   "environment_source_match": evaluation["environment_source_match"],
                   "training_pipeline_source_match": evaluation.get("training_pipeline_source_match"),
                   "provenance_errors": evaluation["provenance_errors"],
               },
               "versions": {package: version(package) for package in ("mujoco", "stable-baselines3", "torch", "numpy", "onnx", "onnxruntime", "gymnasium")}}
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    document = f"""# 彩色画刷：实测结果 / Measured color-brush results

Author: George Hu · 2026-09-11

**发布模型 `{args.run}`：独立 ONNX 验收 {ppo['accepted']}/{ppo['episodes']} 通过。**
固定游泳鸭模板，十组不同场景各四个初始种子；包括组合位置/大小/摩擦/柔软度扰动。
这是学习到的运动反馈控制器 + PPO，不是学习视觉理解或任意构图。

![Actual brush contact painting, not the reference target](media/painting.png)

| 指标 / Metric | BC + DAgger | BC + DAgger + PPO |
|---|---:|---:|
| 通过回合 / Accepted | {bc['accepted']}/{bc['episodes']} | {ppo['accepted']}/{ppo['episodes']} |
| 平均覆盖率 / Coverage | {bc['coverage']:.4%} | {ppo['coverage']:.4%} |
| 平均精度 / Precision | {bc['precision']:.4%} | {ppo['precision']:.4%} |
| 最大笔尖/调色盘法向力 / Peak force | {bc['max_force_n']:.6f} N | {ppo['max_force_n']:.6f} N |
| 最大毛束压缩 / Peak compression | {bc['max_compression_mm']:.4f} mm | {ppo['max_compression_mm']:.4f} mm |

每个回合约 {rendered['elapsed_seconds']:.2f} 秒；颜色分别是金黄、橙、深灰和蓝。
PPO 运行 {metadata['total_timesteps']:,} 个真实 on-policy transition，{metadata['accepted_ppo_updates']} 次更新满足预设漂移上限。
**BC 基线本身已能绘画；这些数据不支持“PPO 比 BC 更好”的因果结论。**
所谓 accepted update 只代表通过动作漂移约束，不代表画质提高。

![Actual training reward and loss, independent axes](media/reward-loss.png)
![Six actual 3D rollout frames](media/frame_sheet.png)

## 验收 / Acceptance

- 每种颜色覆盖率、精度均至少 95%，曲线距离容差 1.5 mm；完整回合，不摔倒、不掉笔。
- 双喙夹持比例 ≥90%；笔尖/调色盘最大法向力 <0.5 N；毛束最大压缩 <3.5 mm；抬笔漏画 <5%。
- 四种颜料须按计划通过真实接触依次取得；空动作和张嘴对照不能通过。
- 评估一次性读取 immutable ONNX bytes；环境、reference、actor、base source 与归档训练 pipeline 均有 SHA-256 门禁。
- 当前 evaluator 可与训练 pipeline 不同；归档训练脚本必须匹配内嵌 hash，报告以 `training_pipeline_source_match` 明示差异。
- 最终录像来自同一 ONNX：`{summary['policy_sha256']}`，没有运行时教师控制器。

## 训练尝试与边界 / Trials and limits

`brush-color-v2-01` 是初步成功模型，20/20 小范围测试通过；其 ONNX 动态宽度注记被 Lab 严格接口拒绝，故不作为发布模型。
`brush-color-v2-02` 改用教师数据种子 101 后，BC 验证失败，未提升为成功示例。
发布模型使用教师数据种子 11、训练种子 101；这显示训练仍有种子敏感性，不能声称任意重训都保证成功。
三个 run 的训练与诊断保留在 `rlx/runs/studio/drawing/`，不删除失败实验。
训练改善同时更改了工具、动作表示和数据流程，不能把所有改善单独归因于柔软画刷或 PPO。
当前 actor 追踪预编排的笔画/取色目标，不学习视觉规划，也不具备错过蘸色后的自主重规划。
毛束是一个被动弹簧接触近似，装饰毛丝不是逐根柔性有限元；不模拟颜料流体或混色。
画刷预装在嘴中，不包含自主拾取、清洗或真实硬件验证。

## 文件 / Files

- [中文复现](README.md) · [English workflow](README.en.md) · [旧 PPO 失败诊断](DIAGNOSIS.md)
- [完整实时视频](media/rollout.mp4) · [4× GIF](media/brush.gif) · [原始接触 CSV](media/raw_trace.csv)
- [最终评估](eval.json) · [匹配 BC 基线](bc-eval.json) · [训练记录](training.json) · [结果数据](results.json)
- [Studio API 验证](evidence/workflow.json)

English summary: the released learned controller passes the fixed-template colored-duck benchmark.
Painting pixels come only from physical brush contact after physical palette loading. The matched BC baseline
already succeeds; PPO completed genuine on-policy updates but is not claimed to improve painting quality.
This is a simulation-only motor-control result, not arbitrary image generation or hardware certification.
"""
    (OUTPUT / "RESULTS.md").write_text(document)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
