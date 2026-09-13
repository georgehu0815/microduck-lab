import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/microduck-arm-v1c"
DOCS = Path(__file__).resolve().parent


def read(path):
    return json.loads((OUTPUT / path).read_text())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    evaluation = read("evaluation-final/evaluation.json")
    learning = read("learning/attempt-2/training-summary.json")
    hardware = read("hardware-acceptance.json")
    videos = read("videos-final/video-manifest.json")
    learned_videos = read("videos-learned/video-manifest.json")
    workspace = read("engineering-final/workspace.json")
    payload = read("payload-screen/summary.json")
    pytest_log = (OUTPUT / "pytest-final.txt").read_text().splitlines()[-1]
    case_rows = []
    case_names = {"reach": "Reach", "pick_place": "Pick/place", "obstacle_relocation": "Obstacle move", "stationary_waypoint_carry": "Waypoint carry", "stowed_arm_walking": "Walk (stowed)", "walking_carry": "Walk + carry"}
    for mode in ("fixture", "free"):
        for case in sorted({record["case"] for record in evaluation["records"] if record["mode"] == mode}):
            selected = [record for record in evaluation["records"] if record["mode"] == mode and record["case"] == case]
            reasons = sorted({record["failure_reason"] for record in selected if record["failure_reason"]})
            case_rows.append(f"| {mode} | {case_names[case]} | {sum(record['success'] for record in selected)}/{len(selected)} | {', '.join(reasons) or 'PASS'} |")
    policy_rows = [f"| {name} | {result['successes']}/5 | {json.dumps(result['failure_counts'])} |" for name, result in learning["evaluation"].items()]
    payload_rows = [f"| {int(record['payload_kg'] * 1000)} | {case_names[record['case']]} | {record['success']} | {record['failure_reason'] or 'PASS'} |" for record in payload["records"]]
    metrics = learning["training_metrics"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    training_root = OUTPUT / "learning/attempt-2"
    for axis, directory, title in ((axes[0, 0], "bc-initial", "BC arm MSE"), (axes[0, 1], "bc-merged", "DAgger merged BC arm MSE")):
        rows = [json.loads(line) for line in (training_root / directory / "bc-training.jsonl").read_text().splitlines()]
        for key, label in (("raw_train_arm_mse", "train"), ("raw_validation_arm_mse", "validation episodes")):
            axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], label=label)
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.legend()
    rows = [json.loads(line) for line in (training_root / "ppo/ppo-training.jsonl").read_text().splitlines()]
    for axis, key, title in ((axes[1, 0], "raw_reward_sum", "PPO rollout reward sum (raw)"), (axes[1, 1], "value_loss", "PPO value loss (raw)")):
        selected = [row for row in rows if row.get(key) is not None]
        axis.plot([row["timesteps"] for row in selected], [row[key] for row in selected])
        axis.set_title(title)
        axis.set_xlabel("training timesteps")
    figure.suptitle("Simulation-only training; lower loss is not task qualification")
    figure.savefig(OUTPUT / "training-curves.png", dpi=160)
    plt.close(figure)
    common = f"""
## Test evidence / 测试证据

`{pytest_log}`

Ruff and Python compilation pass. Standalone mypy remains blocked by resolving
the workspace's `microduck_local.contract` import; this is not claimed as a passed
whole-project type check. Runtime policy-contract regression tests pass.

| Mode | Case | Success / total | Failure |
|---|---|---|---|
{chr(10).join(case_rows)}

Candidates A/B, seeds 0–9, 10 g object. Fixture results are **not** free-base qualification.
The table top is 145 mm; v1-B low-table results are not directly comparable.

## BC → DAgger → residual PPO

8 teacher episodes (seeds 0–7); 4 DAgger episodes (seeds 20–23).
20 BC epochs, 20 merged-data BC epochs, 2,048 PPO timesteps (seed 40).
Held-out seeds: 101–105. Case: reach, fixture, candidate A.
Five arm outputs with an explicit fifteen-action adapter; not stock 14-action ONNX.

| Policy | Held-out success | Failures |
|---|---|---|
{chr(10).join(policy_rows)}

Final metrics (raw, not smoothed):

```json
{json.dumps(metrics, indent=2)}
```

PPO retains the IK teacher with a maximum 0.02 rad arm residual.
Matching the teacher's 5/5 is not evidence that PPO is better than the teacher.
BC loss reduction does not imply closed-loop success. DAgger still has a failure.

![Raw learning curves / 原始训练曲线](../../artifacts/microduck-arm-v1c/training-curves.png)

## Workspace / 工作空间

{workspace['samples']} sampled target positions; {workspace['reachable']} IK-reachable;
{workspace['reachable_collision_free']} reachable and collision-free.
Balance-qualified workspace: **NOT ESTABLISHED**.
Full mass/CoM/inertia tables are analytical model estimates, not measured CAD release.
Joint torque CSV contains 5,000 screening rows; qualified continuous torque is unknown.
The 3× torque release gate remains closed.

## Payload screening / 负载筛查

Candidate A, fixture, seed 0 only: these are not payload ratings or reliability estimates.

| Payload g | Case | Task success | Failure |
|---|---|---|---|
{chr(10).join(payload_rows)}

## Videos / 视频

{len(videos['records'])} full IK/FSM episode videos, plus {len(learned_videos['records'])} actual ONNX policy videos.
All are simulation recordings; successful and failed episodes are retained.
Each video has full ffmpeg decode verification, ffprobe metadata and SHA-256.
See `artifacts/microduck-arm-v1c/videos-final/video-manifest.json` and
`artifacts/microduck-arm-v1c/videos-learned/video-manifest.json`.

## Release / 放行

Modeled cartridge: {hardware['modeled_cartridge_kg'] * 1000:.0f} g;
target: {hardware['cartridge_target_kg'] * 1000:.0f} g — **FAIL**.
Supervisor tests check emulated inputs, not real shorts/reversed wiring or device safety.
Hardware release: **FALSE**. Fabrication, battery, continuous torque, thermal,
sensor/estimator, mobile control and physical fault qualification remain blocked.

Final evidence directories: `evaluation-final`, `engineering-final`, `videos-final`,
`videos-learned`, `learning/attempt-2` under `artifacts/microduck-arm-v1c/`.
Earlier root-level learning outputs, `engineering`, and `videos-attempt-1-interrupted`
are superseded diagnostics, not current release evidence.
"""
    introductions = {
        "en": "# v1-C measured simulation results\n\n**NOT ALL EXPERIMENTS PASS.** Software regression checks and the fixture task matrix pass; free-base manipulation/walking and physical release do not. Preserve those failures rather than relaxing acceptance.\n",
        "zh-CN": "# v1-C 仿真验证结果\n\n**尚未达到全部实验通过。** 软件回归测试和台架任务矩阵通过；自由基座操作、行走和实物放行仍未通过。禁止降低标准或把夹具支撑当作机器人自身平衡。\n\n改进已落实：J1–J5 映射、A/B 电机候选、独立电源域研究、夹具和承力件 BOM、惯量和扭矩表、实际开爪判据、上电前新鲜心跳检查、源码变更中止以及端到端视频证据。混合电机没有重量优势；166 g 模型超出 160 g 目标。\n\n学习结果仅覆盖台架 reach：BC 未通过，DAgger 仍有一例失败，残差 PPO 与教师均为 5/5，并不能证明 PPO 优于教师。机械臂负载表仅为种子 0 的台架筛查，不能宣称为实物承重能力。\n",
    }
    for language, introduction in introductions.items():
        localized = common
        if language == "zh-CN":
            translations = {
                "| Reach |": "| 伸手定位 |",
                "| Pick/place |": "| 抓取放置 |",
                "| Obstacle move |": "| 绕障搬移 |",
                "| Waypoint carry |": "| 定点搬运 |",
                "| Walk (stowed) |": "| 收臂行走 |",
                "| Walk + carry |": "| 行走搬运 |",
                "Ruff and Python compilation pass. Standalone mypy remains blocked by resolving\nthe workspace's `microduck_local.contract` import; this is not claimed as a passed\nwhole-project type check. Runtime policy-contract regression tests pass.": "Ruff 和 Python 编译检查通过。独立 mypy 检查仍无法解析工作区的 `microduck_local.contract` 导入，因此不宣称全项目类型检查通过；实际运行的策略接口回归测试通过。",
                "| Mode | Case | Success / total | Failure |": "| 模式 | 案例 | 成功 / 总数 | 失败原因 |",
                "Candidates A/B, seeds 0–9, 10 g object. Fixture results are **not** free-base qualification.\nThe table top is 145 mm; v1-B low-table results are not directly comparable.": "候选 A/B，种子 0–9，物体 10 g。fixture 是固定台架，free 是自由基座；台架结果**不能**用于整机平衡放行。桌面高 145 mm，不能与 v1-B 的低桌面任务直接比较。",
                "8 teacher episodes (seeds 0–7); 4 DAgger episodes (seeds 20–23).\n20 BC epochs, 20 merged-data BC epochs, 2,048 PPO timesteps (seed 40).\nHeld-out seeds: 101–105. Case: reach, fixture, candidate A.\nFive arm outputs with an explicit fifteen-action adapter; not stock 14-action ONNX.": "教师示范 8 回合（种子 0–7）；DAgger 采集 4 回合（种子 20–23）。初始 BC 和合并数据 BC 各训练 20 轮；PPO 训练 2,048 步（种子 40）。留出评估种子为 101–105，任务为候选 A 的台架 reach。ONNX 输出 5 个手臂动作，必须经过显式 15 动作适配器，不能直接替换原有 14 动作策略。",
                "| Policy | Held-out success | Failures |": "| 策略 | 留出集成功数 | 失败原因 |",
                "Final metrics (raw, not smoothed):": "最后一轮原始指标（未平滑；字段名保留日志原文）：",
                "PPO retains the IK teacher with a maximum 0.02 rad arm residual.\nMatching the teacher's 5/5 is not evidence that PPO is better than the teacher.\nBC loss reduction does not imply closed-loop success. DAgger still has a failure.": "PPO 仍保留 IK 教师，手臂残差上限为 0.02 rad。与教师同为 5/5 不代表优于教师。BC 损失下降不等于闭环任务成功；DAgger 仍有一例失败。",
                "sampled target positions;": "个采样目标位置；",
                "IK-reachable;": "个 IK 可达目标；",
                "reachable and collision-free.": "个同时满足可达和无碰撞条件。",
                "Balance-qualified workspace: **NOT ESTABLISHED**.": "通过平衡验证的工作空间：**尚未建立**。",
                "Full mass/CoM/inertia tables are analytical model estimates, not measured CAD release.\nJoint torque CSV contains 5,000 screening rows; qualified continuous torque is unknown.\nThe 3× torque release gate remains closed.": "质量、重心和完整惯量张量表来自解析模型估计，不是实测或加工放行的 CAD 数据。关节扭矩 CSV 包含 5,000 行筛查记录；合格连续扭矩仍未知，3 倍裕量门槛保持未通过。",
                "Candidate A, fixture, seed 0 only: these are not payload ratings or reliability estimates.": "仅测试候选 A、固定台架、种子 0，不能据此发布额定载荷或可靠性指标。",
                "| Payload g | Case | Task success | Failure |": "| 载荷 g | 案例 | 任务成功 | 失败原因 |",
                "full IK/FSM episode videos, plus": "段完整 IK/状态机回合视频，另有",
                "actual ONNX policy videos.": "段实际运行 ONNX 的策略视频。",
                "All are simulation recordings; successful and failed episodes are retained.\nEach video has full ffmpeg decode verification, ffprobe metadata and SHA-256.\nSee": "全部为仿真录像，同时保留成功和失败回合。每段均经过 ffmpeg 全片解码验证，并记录 ffprobe 元数据和 SHA-256。索引文件：",
                "Modeled cartridge:": "模型中的机械臂模块质量：",
                "target:": "目标：",
                "Supervisor tests check emulated inputs, not real shorts/reversed wiring or device safety.\nHardware release: **FALSE**. Fabrication, battery, continuous torque, thermal,\nsensor/estimator, mobile control and physical fault qualification remain blocked.": "监督器测试仅检查模拟输入，不是真实短路、反接或器件安全试验。硬件放行：**否**。加工、电池、连续扭矩、热性能、传感器/状态估计、移动控制和实体故障验证均仍待完成。",
                "Final evidence directories:": "最终证据目录：",
                "under `artifacts/microduck-arm-v1c/`.": "，均位于 `artifacts/microduck-arm-v1c/` 下。",
                "Earlier root-level learning outputs, `engineering`, and `videos-attempt-1-interrupted`\nare superseded diagnostics, not current release evidence.": "较早的根目录训练产物、`engineering` 和 `videos-attempt-1-interrupted` 仅保留为历史诊断，不属于当前放行证据。",
            }
            for original, translated in translations.items():
                localized = localized.replace(original, translated)
        (DOCS / f"RESULTS.{language}.md").write_text(introduction + localized)


if __name__ == "__main__":
    main()
