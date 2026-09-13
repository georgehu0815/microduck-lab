from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "arm-reach-v1": "末端触达", "arm-pick-place-v1": "抓取放置",
    "arm-relocate-v1": "绕障移物", "arm-carry-v1": "航点搬运",
    "arms-handover-v1": "双臂交接（5 g）", "arms-co-carry-v1": "共同搬托盘（17 g）",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(report):
    return f"{report['successes']}/{report['episodes']}"


def relative(path, directory):
    import os

    return Path(os.path.relpath(path, directory)).as_posix()


def plot_case(case_id, reports, output):
    fields = ("rollout_episode_reward_mean", "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction")
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for report in reports:
        rows = [json.loads(line) for line in (report["directory"] / "metrics.jsonl").read_text().splitlines()]
        for axis, field in zip(axes.flat, fields):
            valid = [row for row in rows if row.get(field) is not None]
            axis.plot([row["env_steps"] for row in valid], [row[field] for row in valid], label=f"seed {report['training']['train_seed']}", linewidth=1)
            axis.set_title(field)
            axis.set_xlabel("environment steps")
            axis.grid(alpha=.2)
    axes.flat[0].legend()
    figure.suptitle(case_id + " | authored IK/FSM + 0.02 * PPO residual | simulation only")
    figure.savefig(output / f"{case_id}-training.png", dpi=120)
    plt.close(figure)


def summarize(batch, output):
    output.mkdir(parents=True, exist_ok=True)
    expected_sources = {"environment": digest(ROOT / "rlx/rlx/environments/arm.py"), "pipeline": digest(ROOT / "rlx/examples/ppo_microduck_arm.py")}
    summary = {"batch": str(batch.relative_to(ROOT)), "source_hashes": expected_sources, "simulation_only": True, "hardware_verified": False, "cases": []}
    lines = ["# 机械臂训练与验收结果", "", "日期：2026-09-12。自动汇总真实运行文件；不是实物能力证明。", "",
             "**控制方法：人工 IK/定时 FSM + 0.02×PPO 残差，不是纯 PPO 从零学会操作。** 三个训练种子使用同一套独立测试配置，不能把配对回放当作300个独立场景。", "",
             "## 主结果矩阵", "", "| 任务 | 教师 | BC诊断 | PPO 101 | PPO 202 | PPO 303 | 3种子均过门槛 |", "|---|---:|---:|---:|---:|---:|---|"]
    details = []
    total_steps = 0
    for case_id, title in NAMES.items():
        reports = []
        for seed in (101, 202, 303):
            directory = batch / case_id / f"seed-{seed}"
            training, evaluation = read(directory / "training.json"), read(directory / "evaluation.json")
            if training["source_hashes"] != expected_sources or evaluation["source_hashes"] != expected_sources:
                raise ValueError(f"Stale source evidence: {directory}")
            if training["case_id"] != case_id or evaluation["case_id"] != case_id or training["model_sha256"] != evaluation["model_sha256"]:
                raise ValueError(f"Case/model mismatch: {directory}")
            if digest(directory / "ppo-residual.zip") != training["checkpoint_sha256"] or training["checkpoint_sha256"] != evaluation["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint mismatch: {directory}")
            total_steps += training["actual_env_steps"]
            reports.append({"directory": directory, "training": training, "evaluation": evaluation})
        teacher = read(reports[0]["directory"] / "teacher-evaluation.json")
        baseline = read(reports[0]["directory"] / "bc-evaluation.json")
        negatives = read(reports[0]["directory"] / "negative-controls.json")
        for report in (teacher, baseline):
            if report["source_hashes"] != expected_sources:
                raise ValueError(f"Stale baseline: {case_id}")
        passed = all(report["evaluation"]["passed"] for report in reports)
        lines.append(f"| {title} | {score(teacher)} | {score(baseline)} | " + " | ".join(score(report["evaluation"]) for report in reports) + f" | {'通过' if passed else '未通过'} |")
        failures = Counter(gate for report in reports for episode in report["evaluation"]["results"] if not episode["passed"] for gate, valid in episode["gates"].items() if not valid)
        summary["cases"].append({"case_id": case_id, "all_three_seeds_passed": passed, "teacher": {key: teacher[key] for key in ("successes", "episodes", "passed")}, "bc": {key: baseline[key] for key in ("successes", "episodes", "passed")}, "negative_controls": negatives, "ppo": [{"seed": report["training"]["train_seed"], **{key: report["evaluation"][key] for key in ("successes", "episodes", "passed", "wilson_95", "per_config_success_rate")}} for report in reports], "failure_gates": dict(failures)})
        plot_case(case_id, reports, output)
        details += ["", f"## {title} / {case_id}", "", "| 训练种子 | 步数 | 训练秒数 | 参数L2变化 | 平均测试回报 | Wilson95% | 最差扰动组 |", "|---|---:|---:|---:|---:|---|---:|"]
        for report in reports:
            training, evaluation = report["training"], report["evaluation"]
            interval = evaluation["wilson_95"]
            details.append(f"| {training['train_seed']} | {training['actual_env_steps']} | {training['training_wall_seconds']:.2f} | {training['parameter_l2_change']:.4f} | {statistics.mean(episode['return'] for episode in evaluation['results']):.4f} | {interval[0]:.1%}–{interval[1]:.1%} | {min(evaluation['per_config_success_rate']):.0%} |")
        directory = reports[0]["directory"]
        details += ["", f"![{case_id} reward and loss]({case_id}-training.png)", "",
                    f"- [全部运行文件]({relative(batch / case_id, output)}/)：源代码快照、场景、检查点、逐回合奖励、PPO损失、每个测试回合的门槛及故障。",
                    f"- 负对照成功数：`{json.dumps(negatives['controllers'], ensure_ascii=False)}`；负对照验证{'通过' if negatives['passed'] else '未通过'}。",
                    f"- PPO失败门槛计数：`{json.dumps(dict(failures), ensure_ascii=False)}`。"]
        receipt = directory / "render-receipt.json"
        if receipt.is_file():
            rendered = read(receipt)
            if rendered["source_hashes"] != expected_sources or rendered["video_sha256"] != digest(directory / "rollout.mp4"):
                raise ValueError(f"Stale video: {case_id}")
            details += [f"- [真实MP4]({relative(directory / 'rollout.mp4', output)})、[视频回执]({relative(receipt, output)})、[6帧接触图]({relative(directory / 'contact-sheet.png', output)})；回放seed={rendered['seed']}，单次门槛{'通过' if rendered['episode']['metrics']['passed'] else '未通过'}。"]
        details.append(f"- [训练曲线原始日志]({relative(directory / 'metrics.jsonl', output)})；[BC训练记录]({relative(directory / 'bc-training.json', output)})。")
    summary["actual_ppo_environment_steps"] = total_steps
    summary["all_six_composite_tasks_passed"] = all(case["all_three_seeds_passed"] for case in summary["cases"])
    lines += ["", f"真实PPO环境步总数：**{total_steps:,}**。各检查点独立记录参数变化，不用教师结果冒充PPO评估。", "",
              "BC诊断失败不隐藏，不因训练MSE降低宣布任务学会。即使复合控制通过，教师已完成绝大部分任务，因此本实验不能证明PPO优于教师。测试扰动较窄；不能推断开放世界泛化或实体部署。",
              "", "## 不包括的验证", "", "- 实物20g载荷、夹力/温升、CAD装配公差、电源与保险协调、急停响应均未验证。", "- 双臂交接仅5g课程，协作搬运17g总重，不冒充双臂20g能力。", "- ONNX仅导出残差演员，必须配合相同教师、契约和限幅；原61→14身体策略不可替换。", "- 软件互锁不是功能安全认证；HTTP仿真与硬件Unix协调器尚不是同一已放行的实物执行链。"]
    (output / "RESULTS.md").write_text("\n".join(lines + details) + "\n")
    (output / "results-summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"all_six_composite_tasks_passed": summary["all_six_composite_tasks_passed"], "steps": total_steps}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/robot-arm-design")
    args = parser.parse_args()
    summarize(args.batch.resolve(), args.output.resolve())
