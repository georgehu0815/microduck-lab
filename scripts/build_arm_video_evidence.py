from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASE_NAMES = {
    "arm-reach-v1": "末端触达",
    "arm-pick-place-v1": "抓取放置",
    "arm-relocate-v1": "绕障移物",
    "arm-carry-v1": "航点搬运",
    "arms-handover-v1": "双臂交接（5 g课程）",
    "arms-co-carry-v1": "双臂搬托盘（17 g总重）",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def assert_values_equal(expected: object, actual: object, name: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f"Different metric keys at {name}")
        for key, value in expected.items():
            assert_values_equal(value, actual[key], f"{name}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Different metric list at {name}")
        for index, value in enumerate(expected):
            assert_values_equal(value, actual[index], f"{name}[{index}]")
    elif isinstance(expected, bool):
        if type(actual) is not bool or expected != actual:
            raise ValueError(f"Different metric boolean at {name}")
    elif isinstance(expected, (int, float)):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(expected, actual, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Different metric number at {name}")
    elif expected != actual:
        raise ValueError(f"Different metric value at {name}")


def validate_batch(summary: dict, protocol: dict) -> None:
    expected_jobs = {(case, seed) for case in CASE_NAMES for seed in protocol["training_seeds"]}
    identities = [(job["case_id"], job["training_seed"]) for job in summary["jobs"]]
    if len(identities) != len(expected_jobs) or set(identities) != expected_jobs:
        raise ValueError("Missing, duplicate, or unexpected checkpoint jobs")
    episodes = successes = 0
    checkpoints_passed = []
    for job in summary["jobs"]:
        path = Path(job["evaluation"])
        report = json.loads(path.read_text())
        if digest(path) != job["evaluation_sha256"] or report["source_hashes"] != protocol["source_hashes"]:
            raise ValueError("Evaluation identity differs from protocol/summary")
        if report["checkpoint_sha256"] != digest(Path(job["checkpoint"])):
            raise ValueError("Evaluation checkpoint identity mismatch")
        if report["case_id"] != job["case_id"] or report["controller"] != "ppo_residual":
            raise ValueError("Evaluation case or controller mismatch")
        expected_samples = {
            (index, protocol["seed_base"] + index * 100 + offset)
            for index in range(len(protocol["perturbations"])) for offset in range(protocol["seeds_per_config"])
        }
        actual_samples = [(row["config_index"], row["seed"]) for row in report["results"]]
        if len(actual_samples) != len(expected_samples) or set(actual_samples) != expected_samples:
            raise ValueError("Evaluation seed/config coverage differs from protocol")
        for row in report["results"]:
            if row["options"] != protocol["perturbations"][row["config_index"]]:
                raise ValueError("Evaluation perturbations differ from protocol")
            if row["passed"] != all(row["gates"].values()):
                raise ValueError("Episode verdict disagrees with gates")
        total = len(report["results"])
        passed = sum(row["passed"] for row in report["results"])
        groups = [sum(row["passed"] for row in report["results"] if row["config_index"] == index) / protocol["seeds_per_config"] for index in range(len(protocol["perturbations"]))]
        threshold_passed = passed / total >= protocol["case_minimum_success_rates"][job["case_id"]] and min(groups) >= protocol["minimum_per_config_success_rate"]
        if report["episodes"] != total or report["successes"] != passed or job["episodes"] != total or job["successes"] != passed:
            raise ValueError("Episode totals do not reconcile")
        if report["passed"] != threshold_passed or job["passed"] != threshold_passed or report["per_config_success_rate"] != groups:
            raise ValueError("Statistical gate does not reconcile")
        if job["failures"] != [row for row in report["results"] if not row["passed"]]:
            raise ValueError("New failures omitted from summary")
        historical_path = Path(protocol["source_batch"]) / job["case_id"] / f"seed-{job['training_seed']}" / "evaluation.json"
        historical = json.loads(historical_path.read_text())
        if historical["source_hashes"] != protocol["source_hashes"] or historical["checkpoint_sha256"] != report["checkpoint_sha256"]:
            raise ValueError("Historical evidence has wrong source/checkpoint identity")
        old_failures = [row for row in historical["results"] if not row["passed"]]
        if [row["original"] for row in job["historical_failure_replays"]] != old_failures:
            raise ValueError("Historical failures omitted from replays")
        for replay in job["historical_failure_replays"]:
            if replay["original"]["seed"] != replay["replay"]["seed"] or replay["original"]["options"] != replay["replay"]["options"]:
                raise ValueError("Historical replay changed seed or perturbations")
        episodes += total
        successes += passed
        checkpoints_passed.append(threshold_passed)
    if summary["episodes"] != episodes or summary["successes"] != successes:
        raise ValueError("Batch totals do not reconcile")
    if summary["all_episodes_pass"] != (episodes == successes) or summary["all_checkpoints_meet_threshold"] != all(checkpoints_passed):
        raise ValueError("Batch status conflates threshold pass and episode pass")


def select_episodes(summary: dict) -> list[dict]:
    selected = {}
    for job in summary["jobs"]:
        report = json.loads(Path(job["evaluation"]).read_text())
        if digest(Path(job["evaluation"])) != job["evaluation_sha256"]:
            raise ValueError("Evaluation changed after summary creation")
        candidates = []
        if job["training_seed"] == 101:
            candidates.append(("first-new-seed", report["results"][0]))
        candidates.extend(("new-failure", row) for row in report["results"] if not row["passed"])
        candidates.extend(("historical-failure-replay", row["replay"]) for row in job["historical_failure_replays"])
        for reason, episode in candidates:
            key = (job["checkpoint"], episode["seed"], json.dumps(episode["options"], sort_keys=True))
            if key in selected:
                selected[key]["selection_reasons"].append(reason)
                continue
            selected[key] = {
                "id": f"{job['case_id']}-train{job['training_seed']}-eval{episode['seed']}",
                "case_id": job["case_id"],
                "training_seed": job["training_seed"],
                "checkpoint": job["checkpoint"],
                "evaluation_model_sha256": report.get("model_sha256"),
                "selection_reasons": [reason],
                "expected_episode": episode,
            }
    return sorted(selected.values(), key=lambda item: (
        0 if "first-new-seed" in item["selection_reasons"] else 1,
        list(CASE_NAMES).index(item["case_id"]), item["training_seed"], item["expected_episode"]["seed"],
    ))


def repaired_episodes(report: dict, summary: dict) -> list[dict]:
    historical_path = Path(report["historical_batch"]) / "summary.json"
    if digest(historical_path) != report["historical_summary_sha256"]:
        raise ValueError("Historical regression source changed")
    historical = json.loads(historical_path.read_text())
    originals = {}
    for job in historical["jobs"]:
        episodes = job["failures"] + [row["replay"] for row in job["historical_failure_replays"]]
        for episode in episodes:
            originals[(job["training_seed"], episode["seed"])] = episode
    training_seeds = {job["training_seed"] for job in summary["jobs"]}
    required = {(seed, *identity) for seed in training_seeds for identity in originals}
    actual = [(row["training_seed"], row["original_training_seed"], row["episode"]["seed"]) for row in report["results"]]
    if len(actual) != len(required) or set(actual) != required:
        raise ValueError("Known-failure regression coverage is incomplete")
    selected = []
    for row in report["results"]:
        matching_jobs = [job for job in summary["jobs"] if job["case_id"] == row["case_id"] and job["training_seed"] == row["training_seed"]]
        if len(matching_jobs) != 1 or Path(matching_jobs[0]["checkpoint"]).resolve() != Path(row["checkpoint"]).resolve():
            raise ValueError("Repaired regression is not bound to the validated training seed/checkpoint")
        expected = originals[(row["original_training_seed"], row["episode"]["seed"])]
        if row["original_episode"] != expected or row["episode"]["options"] != expected["options"]:
            raise ValueError("Repaired regression changed the original failure conditions")
        if row["source_hashes"] != summary["source_hashes"] or row["checkpoint_sha256"] != digest(Path(row["checkpoint"])):
            raise ValueError("Repaired regression has incorrect provenance")
        if row["episode"]["passed"] != all(row["episode"]["gates"].values()):
            raise ValueError("Repaired regression verdict disagrees with gates")
        if row["training_seed"] == row["original_training_seed"]:
            selected.append({"id": f"{row['case_id']}-train{row['training_seed']}-eval{row['episode']['seed']}",
                             "case_id": row["case_id"], "training_seed": row["training_seed"],
                             "checkpoint": row["checkpoint"], "evaluation_model_sha256": row["model_sha256"],
                             "selection_reasons": ["repaired-known-failure"], "expected_episode": row["episode"]})
    successes = sum(row["episode"]["passed"] for row in report["results"])
    if report["episodes"] != len(actual) or report["successes"] != successes or report["all_passed"] != (successes == len(actual)):
        raise ValueError("Repaired regression totals do not reconcile")
    return selected


def assert_episode_matches(expected: dict, receipt: dict) -> None:
    episode = receipt["episode"]
    metrics = episode["metrics"]
    if episode["seed"] != expected["seed"] or episode["steps"] != expected["steps"]:
        raise ValueError("Video seed or episode length differs from evaluation")
    if metrics["passed"] != expected["passed"] or metrics["gates"] != expected["gates"]:
        raise ValueError("Video verdict/gates differ from evaluation")
    defaults = {"position_noise": 0.001, "mass_scale": 1.0, "friction_scale": 1.0}
    if defaults | episode["reset_options"] != defaults | expected["options"]:
        raise ValueError("Video perturbations differ from evaluation")
    for key in ("drop_count", "invalid_contacts", "arm_collisions", "self_collisions", "waypoints_reached"):
        if metrics[key] != expected[key]:
            raise ValueError(f"Video {key} differs from evaluation")
    for key in ("position_error_m", "tip_error_m", "peak_lift_m", "peak_contact_force_n", "peak_internal_force_n", "payload_slip_m", "transport_tilt_deg"):
        if not math.isclose(metrics[key], expected[key], rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Video {key} differs from evaluation")
    if not math.isclose(episode["return"], expected["return"], rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("Video return differs from evaluation")
    result_metadata = {"case_id", "controller", "seed", "options", "steps", "return", "wall_seconds", "control_semantics", "payload_mass", "config_index"}
    expected_metrics = {key: value for key, value in expected.items() if key not in result_metadata}
    actual_metrics = dict(metrics)
    for values in (expected_metrics, actual_metrics):
        if "reset_options" in values:
            values["reset_options"] = defaults | values["reset_options"]
    assert_values_equal(expected_metrics, actual_metrics, "terminal_metrics")


def verify_video(directory: Path, selected: dict, sources: dict) -> dict:
    receipt = json.loads((directory / "render-receipt.json").read_text())
    if receipt["source_hashes"] != sources:
        raise ValueError("Video uses different training/environment sources")
    if receipt["renderer_source_sha256"] != digest(ROOT / "rlx/examples/render_microduck_arm.py"):
        raise ValueError("Video renderer source identity mismatch")
    if receipt["checkpoint_sha256"] != digest(Path(selected["checkpoint"])):
        raise ValueError("Video uses a different checkpoint")
    if receipt["case_id"] != selected["case_id"] or receipt["controller"] != "ppo_residual":
        raise ValueError("Video case/controller mismatch")
    if receipt["model_sha256"] != selected["evaluation_model_sha256"]:
        raise ValueError("Video model differs from evaluation")
    required_media = {"rollout.mp4", "contact-sheet.png", "telemetry.jsonl"}
    if not required_media.issubset(receipt["media_hashes"]):
        raise ValueError("Required evidence missing from media hashes")
    if receipt["video_sha256"] != receipt["media_hashes"]["rollout.mp4"]:
        raise ValueError("Duplicate video hash fields disagree")
    if receipt["telemetry_sha256"] != receipt["media_hashes"]["telemetry.jsonl"]:
        raise ValueError("Duplicate telemetry hash fields disagree")
    if receipt["episode_sha256"] != canonical_digest(receipt["episode"]):
        raise ValueError("Canonical episode hash mismatch")
    for name, expected in receipt["media_hashes"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Video receipt hash mismatch: {name}")
    if "telemetry.jsonl" not in receipt["media_hashes"]:
        raise ValueError("Telemetry is not bound to video receipt")
    assert_episode_matches(selected["expected_episode"], receipt)
    trace = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    if [row["step"] for row in trace] != list(range(1, receipt["episode"]["steps"] + 1)):
        raise ValueError("Telemetry contains missing or duplicate steps")
    if trace[-1]["metrics"] != receipt["episode"]["metrics"]:
        raise ValueError("Final telemetry does not match final video metrics")
    if not math.isclose(sum(row["reward"] for row in trace), receipt["episode"]["return"], abs_tol=1e-8):
        raise ValueError("Telemetry rewards do not sum to video return")
    running_reward = 0.0
    for row in trace:
        running_reward += row["reward"]
        if not math.isclose(row["cumulative_return"], running_reward, abs_tol=1e-8) or not math.isclose(row["time"], row["step"] * 0.02, abs_tol=1e-9):
            raise ValueError("Telemetry timestamp or running return mismatch")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate:format=duration",
        "-of", "json", str(directory / "rollout.mp4"),
    ], check=True, capture_output=True, text=True)
    media = json.loads(probe.stdout)
    stream = media["streams"][0]
    if (stream["width"], stream["height"]) != (1280, 720):
        raise ValueError("Unexpected video resolution")
    if int(stream["nb_read_frames"]) != receipt["episode"]["frames"]:
        raise ValueError("Decoded frame count differs from receipt")
    if not math.isclose(float(media["format"]["duration"]), receipt["episode"]["frames"] / receipt["episode"]["fps"], abs_tol=0.01):
        raise ValueError("Video duration differs from recorded frame rate")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(directory / "rollout.mp4"), "-f", "null", "-"], check=True, capture_output=True)
    events = []
    previous_stage = None
    safety_gates = ("no_drop", "no_illegal_contact", "no_arm_collision", "no_self_collision", "internal_force_bounded")
    failed_seen = set()
    for row in trace:
        metrics = row["metrics"]
        stage = metrics["evaluator_stage"]
        if stage != previous_stage:
            events.append({"time": row["time"], "step": row["step"], "event": f"stage: {stage}"})
            previous_stage = stage
        for gate in safety_gates:
            if metrics["gates"].get(gate) is False and gate not in failed_seen:
                events.append({"time": row["time"], "step": row["step"], "event": f"safety gate false: {gate}"})
                failed_seen.add(gate)
    write_json(directory / "timeline.json", events)
    return {
        **selected,
        "receipt": str((directory / "render-receipt.json").resolve()),
        "video_sha256": receipt["video_sha256"],
        "receipt_sha256": digest(directory / "render-receipt.json"),
        "duration_seconds": float(media["format"]["duration"]),
        "decoded_frames": int(stream["nb_read_frames"]),
        "full_decode_passed": True,
        "matches_evaluation": True,
        "telemetry_steps": len(trace),
        "events": events,
        "passed": receipt["episode"]["metrics"]["passed"],
        "failed_gates": [name for name, passed in receipt["episode"]["metrics"]["gates"].items() if not passed],
    }


def build_index(output: Path, summary: dict, videos: list[dict]) -> None:
    protocol = json.loads((output / "protocol.json").read_text())
    job_count = len(summary["jobs"])
    old_failure_count = sum(len(job["historical_failure_replays"]) for job in summary["jobs"])
    group_count = len(protocol["perturbations"])
    seeds_per_group = protocol["seeds_per_config"]
    last_seed = protocol["seed_base"] + (group_count - 1) * 100 + seeds_per_group - 1
    rows = []
    for case in CASE_NAMES:
        jobs = [job for job in summary["jobs"] if job["case_id"] == case]
        successes = sum(job["successes"] for job in jobs)
        episodes = sum(job["episodes"] for job in jobs)
        rows.append(f"<tr><td>{CASE_NAMES[case]}</td><td>{successes}/{episodes}</td><td>{sum(job['passed'] for job in jobs)}/{len(jobs)}</td><td>{'、'.join(str(job['successes']) + '/' + str(job['episodes']) for job in jobs)}</td></tr>")
    cards = []
    for index, video in enumerate(videos):
        base = f"videos/{video['id']}"
        expected = video["expected_episode"]
        gates = "".join(f"<li>{'✓' if passed else '✗'} {html.escape(gate)}</li>" for gate, passed in expected["gates"].items())
        events = " ".join(f"<button data-video='video-{index}' data-time='{event['time']}'>{event['time']:.2f}s · {html.escape(event['event'])}</button>" for event in video["events"])
        result = "PASS" if video["passed"] else "FAIL"
        reasons = {"first-new-seed": "预先选定：训练seed101的第一个新评估回合（不是择优）", "new-failure": "新评估失败回合（全部保留）", "historical-failure-replay": "历史失败回合复现", "repaired-known-failure": "修复回归：保留原失败seed与扰动，使用修复后重新训练的检查点"}
        captions = "；".join(reasons[reason] for reason in video["selection_reasons"])
        cards.append(f"""<article><h2>{CASE_NAMES[video['case_id']]} · <span class='{result.lower()}'>{result}</span></h2>
<p>{captions}。训练seed {video['training_seed']}；评估seed {expected['seed']}；扰动 <code>{html.escape(json.dumps(expected['options']))}</code></p>
<video id='video-{index}' controls preload='none' poster='{base}/contact-sheet.png' src='{base}/rollout.mp4'></video>
<p>{video['duration_seconds']:.2f}s / 1280×720 / {video['decoded_frames']}帧；逐步遥测 {video['telemetry_steps']}行；累计奖励 {expected['return']:.5f}；掉落 {expected['drop_count']}；峰值内力 {expected['peak_internal_force_n']:.4f}N。</p>
<p>{events}</p><p><a href='{base}/rollout.mp4'>下载MP4</a> · <a href='{base}/contact-sheet.png'>关键帧</a> · <a href='{base}/telemetry.jsonl'>逐步遥测</a> · <a href='{base}/render-receipt.json'>视频回执</a> · <a href='{base}/timeline.json'>事件时间线</a></p>
<details><summary>最终验收门槛与视频SHA-256</summary><ul>{gates}</ul><code>{video['video_sha256']}</code></details></article>""")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>机械臂复测 · 完整视频证据</title>
<style>body{{max-width:1120px;margin:32px auto;padding:0 20px;font:17px/1.65 system-ui;color:#17233a;background:#f4f6fa}}h1{{line-height:1.2}}article,.panel{{background:white;padding:24px;border-radius:12px;margin:24px 0}}video{{width:100%;background:#10131a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd3df;padding:9px;text-align:left}}code{{overflow-wrap:anywhere;font-size:13px}}button{{padding:8px;margin:3px;cursor:pointer}}.fail{{color:#b42632}}.pass{{color:#146843}}a{{color:#1752a0}}@media print{{video,button{{display:none}}article{{break-inside:avoid}}}}</style>
<h1>机械臂复测与视频证据</h1><p>2026-09-12 · MuJoCo仿真 · IK／状态机 + 0.02×PPO残差 · 未进行实体硬件验证</p>
<section class='panel'><h2>先看结论，不隐藏失败</h2><p>独立新种子评估 <strong>{summary['successes']}/{summary['episodes']}</strong> 回合成功；达到原有统计验收门槛的检查点 <strong>{sum(job['passed'] for job in summary['jobs'])}/{job_count}</strong>。逐回合全部通过：<strong>{'是' if summary['all_episodes_pass'] else '否'}</strong>。达到成功率门槛不等于100%可靠。</p>
<p>保留{job_count}个冻结检查点，不重训、不改奖励、不改物理和门槛。每个检查点：{group_count}类扰动×{seeds_per_group}个新种子，seed{protocol['seed_base']}–{last_seed}；种子在不同检查点间复用，因此{summary['episodes']}回合不是{summary['episodes']}个互异随机种子。{old_failure_count}次历史失败另行重放，不计入{summary['episodes']}回合分母。</p>
<p>视频采样规则：六案例各选训练seed101的第一个新评估回合，另录全部新失败及全部历史失败。短片不能证明所有回合成功；完整统计以逐回合JSON为准。视频提供全帧解码校验、源码/模型/检查点哈希、逐步遥测、最终门槛和事件跳转。</p>
<p>三种单臂搬物负载20g；交接仅5g；托盘总重17g，后两者不代表20g设计目标已完成。纯PPO从零学习、上机安全和实体通电均未验证。</p>
<p><a href='summary.json'>完整新评估结果</a> · <a href='protocol.json'>预先固定的测试协议</a> · <a href='video-evidence.json'>视频核验清单</a></p>
<table><tr><th>案例</th><th>成功/回合</th><th>通过门槛/检查点</th><th>训练seed101 /202 /303</th></tr>{''.join(rows)}</table></section>
{''.join(cards)}<script>document.addEventListener('click',event=>{{const button=event.target.closest('button[data-video]');if(!button)return;const video=document.getElementById(button.dataset.video);video.currentTime=Number(button.dataset.time);video.play().catch(()=>{{}});}});</script></html>"""
    (output / "index.zh-CN.html").write_text(page)
    repair_path = output / "repair-regressions.json"
    if repair_path.exists():
        repairs = json.loads(repair_path.read_text())
        notice = f"<section class='panel'><h2>原失败条件的修复回归</h2><p>保留原seed和扰动、使用重新训练的三个检查点交叉验证：{repairs['successes']}/{repairs['episodes']}回合通过。下方逐一录制与原训练种子对应的修复回放，未删除原失败录像。</p><p><a href='repair-regressions.json'>21次交叉回归原始证据</a></p></section>"
        page = page.replace("<section class='panel'><h2>先看结论", notice + "<section class='panel'><h2>先看结论")
        (output / "index.zh-CN.html").write_text(page)


def build_compilation(output: Path, videos: list[dict], name: str) -> dict:
    selected = [video for video in videos if (
        "first-new-seed" in video["selection_reasons"] if name == "six-cases-overview" else not video["passed"]
    )]
    concat = output / f"{name}.ffconcat"
    concat.write_text("ffconcat version 1.0\n" + "".join(
        f"file 'videos/{video['id']}/rollout.mp4'\n" for video in selected
    ))
    destination = output / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", str(destination)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(destination), "-f", "null", "-"], check=True, capture_output=True)
    offset = 0.0
    chapters = []
    for video in selected:
        chapters.append({"start_s": offset, "end_s": offset + video["duration_seconds"],
                         "id": video["id"], "passed": video["passed"], "source_sha256": video["video_sha256"]})
        offset += video["duration_seconds"]
    result = {"path": destination.name, "sha256": digest(destination), "duration_s": offset,
              "full_decode_passed": True, "chapters": chapters,
              "description": "Concatenated evidence videos; no interpolation, speed changes, or success-only selection"}
    write_json(output / f"{name}.chapters.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--regressions", type=Path)
    parser.add_argument("--export-residual", action="store_true")
    args = parser.parse_args()
    output = args.batch.resolve()
    summary = json.loads((output / "summary.json").read_text())
    if not summary["execution_completed"]:
        raise ValueError("Cannot build complete evidence from an incomplete evaluation")
    protocol = json.loads((output / "protocol.json").read_text())
    validate_batch(summary, protocol)
    locked_paths = [Path(__file__).resolve(), ROOT / "rlx/examples/render_microduck_arm.py",
                    ROOT / "rlx/examples/ppo_microduck_arm.py", ROOT / "rlx/rlx/environments/arm.py",
                    output / "summary.json", output / "protocol.json"]
    locked_paths.extend(Path(job["evaluation"]) for job in summary["jobs"])
    repairs = None
    if args.regressions:
        repairs = json.loads(args.regressions.read_text())
        repaired_episodes(repairs, summary)
        write_json(output / "repair-regressions.json", repairs)
        locked_paths.append(output / "repair-regressions.json")
    identity = {str(path): digest(path) for path in locked_paths}
    lock_path = output / "evidence-source-lock.json"
    if lock_path.exists() and json.loads(lock_path.read_text()) != identity:
        raise ValueError("Evidence sources changed after video batch was frozen")
    if not lock_path.exists():
        write_json(lock_path, identity)
    selected = select_episodes(summary)
    if repairs is not None:
        selected.extend(repaired_episodes(repairs, summary))
    videos = []
    for item in selected:
        if {str(path): digest(path) for path in locked_paths} != identity:
            raise ValueError("Frozen evidence changed during rendering")
        directory = output / "videos" / item["id"]
        completed_render = args.resume and (directory / "render-receipt.json").is_file()
        if not args.verify_only and not completed_render:
            directory.mkdir(parents=True, exist_ok=False)
            command = [sys.executable, str(ROOT / "rlx/examples/render_microduck_arm.py"),
                       "--case", item["case_id"], "--checkpoint", item["checkpoint"],
                       "--output", str(directory), "--seed", str(item["expected_episode"]["seed"]),
                       "--fps", "25", "--trace"]
            for key, value in item["expected_episode"]["options"].items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            if args.export_residual and "first-new-seed" in item["selection_reasons"]:
                command.append("--export")
            write_json(directory / "command.json", command)
            with (directory / "render.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        verified = verify_video(directory, item, summary["source_hashes"])
        videos.append(verified)
        print(json.dumps({key: verified[key] for key in ("id", "passed", "failed_gates", "duration_seconds", "telemetry_steps")}), flush=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "summary_sha256": digest(output / "summary.json"),
                "evidence_verification_passed": True, "all_experiment_episodes_pass": summary["all_episodes_pass"],
                "all_checkpoints_meet_threshold": summary["all_checkpoints_meet_threshold"],
                "known_failure_regressions_passed": repairs["all_passed"] if repairs else None,
                "video_count": len(videos), "videos": videos}
    if {str(path): digest(path) for path in locked_paths} != identity:
        raise ValueError("Frozen evidence changed before publication")
    write_json(output / "video-evidence.json", manifest)
    build_index(output, summary, videos)
    compilations = [build_compilation(output, videos, "six-cases-overview")]
    if any(not video["passed"] for video in videos):
        compilations.append(build_compilation(output, videos, "all-failures"))
    write_json(output / "compilations.json", compilations)
    page = (output / "index.zh-CN.html").read_text()
    links = " · ".join(f"<a href='{item['path']}'>{'六案例连续播放' if item['path'].startswith('six') else '全部失败连续播放'}（{item['duration_s']:.1f}s）</a>" for item in compilations)
    page = page.replace("<h1>机械臂复测与视频证据</h1>", "<h1>机械臂复测与视频证据</h1><p>" + links + "</p>")
    (output / "index.zh-CN.html").write_text(page)


if __name__ == "__main__":
    main()
