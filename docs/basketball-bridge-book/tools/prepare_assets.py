#!/usr/bin/env python3
"""Prepare deterministic, source-backed assets for the basketball/bridge book."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/microduck-book-matplotlib")
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/microduck-book-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
BOOK = ROOT / "docs" / "basketball-bridge-book"
ASSETS = BOOK / "assets"
RAW = ASSETS / "raw"
EVIDENCE = ASSETS / "evidence"
DIAGRAMS = ASSETS / "diagrams"

SOURCES = {
    "basketball_local_summary": ROOT
    / "rlx/artifacts/basketball-local-20260910/summary.json",
    "basketball_mixed_summary": ROOT
    / "rlx/artifacts/basketball-mixed-20260910/summary.json",
    "basketball_evaluation": ROOT
    / "docs/basketball-showcase/evidence/evaluation.json",
    "basketball_release_validation": ROOT
    / "microduck-playground/experiments/basketball/validation.json",
    "basketball_training_lineage": ROOT
    / "microduck-playground/experiments/basketball/TRAINING.md",
    "basketball_environment": ROOT / "rlx/rlx/environments/basketball.py",
    "bridge_training_metrics": ROOT
    / "rlx/runs/studio/bridge/bridge-studio-02/training-metrics.jsonl",
    "bridge_evaluation": ROOT
    / "rlx/runs/studio/bridge/bridge-studio-02/evaluation.json",
    "bridge_result": ROOT / "rlx/runs/studio/bridge/bridge-studio-02/result.json",
    "bridge_checkpoint_metadata": ROOT
    / "rlx/runs/studio/bridge/bridge-studio-02/bridge.safetensors.json",
    "bridge_render_evidence": ROOT
    / "rlx/runs/studio/bridge/bridge-studio-02/render/evidence.json",
    "bridge_environment": ROOT / "rlx/rlx/environments/bridge.py",
}

OBSERVED_IMAGES = {
    "observed-basketball-zero-command-contact-sheet.png": {
        "source": ROOT
        / "docs/basketball-showcase/evidence/render/"
        "cmd-p0_000-p0_000-p0_000/seed-101/contact-sheet.png",
        "caption": (
            "Observed deterministic basketball rollout contact sheet, seed 101, "
            "zero velocity command, 60-second evaluation."
        ),
    },
    "observed-basketball-forward-command-contact-sheet.png": {
        "source": ROOT
        / "docs/basketball-showcase/evidence/render/"
        "cmd-p0_150-p0_000-p0_000/seed-101/contact-sheet.png",
        "caption": (
            "Observed deterministic basketball rollout contact sheet, seed 101, "
            "+0.15 m/s forward command, 60-second evaluation."
        ),
    },
    "observed-bridge-rollout-contact-sheet.png": {
        "source": ROOT
        / "rlx/runs/studio/bridge/bridge-studio-02/render/ep0_sheet.png",
        "caption": (
            "Observed bridge-studio-02 deterministic rollout contact sheet: "
            "1,000 control steps (20-second capture) with 2 resets; this is not "
            "one sustained 20-second crossing attempt, and crossing was not achieved."
        ),
    },
    "observed-bridge-studio-screenshot.png": {
        "source": ROOT
        / "docs/bridge-showcase/evidence/studio/suspended-bridge-desktop.png",
        "caption": (
            "Observed Studio experiments screen showing the suspended-bridge "
            "training pilot and its failed-crossing status."
        ),
    },
    "observed-live-scenes.png": {
        "source": ROOT / "docs/bridge-showcase/evidence/studio/live-scenes-desktop.png",
        "caption": (
            "Observed Studio live-scenes desktop screenshot showing genuine "
            "rendered experiment scenes."
        ),
    },
    "observed-seven-cases.png": {
        "source": ROOT / "docs/bridge-showcase/evidence/studio/seven-cases.png",
        "caption": (
            "Observed Studio desktop screenshot showing the seven available "
            "experiment cases."
        ),
    },
}

COLORS = {
    "ink": "#17202A",
    "muted": "#5D6D7E",
    "grid": "#D5D8DC",
    "local": "#117864",
    "mixed": "#B03A2E",
    "reward": "#1F618D",
    "loss": "#8E44AD",
    "orange": "#D35400",
    "gold": "#B7950B",
    "bridge": "#7E5109",
    "platform": "#566573",
    "green": "#1E8449",
    "red": "#C0392B",
}

PNG_METADATA = {
    "Software": "matplotlib",
    "Title": "Microduck basketball and bridge book asset",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": COLORS["muted"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata=PNG_METADATA,
        pil_kwargs={"compress_level": 9},
    )
    plt.close(fig)


def rolling_mean(values: list[float], window: int) -> list[float]:
    recent: deque[float] = deque()
    total = 0.0
    result: list[float] = []
    for value in values:
        recent.append(value)
        total += value
        if len(recent) > window:
            total -= recent.popleft()
        result.append(total / len(recent))
    return result


def validate_sources() -> None:
    missing = [
        relative(path) if path.is_absolute() and ROOT in path.parents else str(path)
        for path in [*SOURCES.values(), *(item["source"] for item in OBSERVED_IMAGES.values())]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required source files:\n" + "\n".join(missing))


def reset_output_dirs() -> None:
    for directory in (RAW, EVIDENCE, DIAGRAMS):
        directory.mkdir(parents=True, exist_ok=True)
    generated_names = {
        "README.md",
        "captions_manifest.json",
        "provenance.json",
        "SHA256SUMS",
    }
    for path in ASSETS.iterdir():
        if path.is_file() and path.name in generated_names:
            path.unlink()
    preserved_raw = {
        "basketball_reward_trace.csv",
        "basketball_reward_trace.json",
    }
    for directory in (RAW, EVIDENCE, DIAGRAMS):
        for path in directory.iterdir():
            if path.is_file() and not (
                directory == RAW and path.name in preserved_raw
            ):
                path.unlink()


def copy_raw_sources() -> dict[str, Path]:
    copied: dict[str, Path] = {}
    for source_id, source in SOURCES.items():
        suffix = "".join(source.suffixes)
        destination = RAW / f"{source_id}{suffix}"
        shutil.copyfile(source, destination)
        copied[source_id] = destination
    return copied


def copy_observed_images() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name, details in OBSERVED_IMAGES.items():
        source = details["source"]
        destination = EVIDENCE / name
        shutil.copyfile(source, destination)
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"Copy hash mismatch for {name}")
        manifest.append(
            {
                "asset": relative(destination),
                "asset_kind": "observed_evidence",
                "caption": details["caption"],
                "source": relative(source),
                "source_sha256": sha256(source),
            }
        )
    return manifest


def extract_basketball_updates(
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, summary in summaries.items():
        updates = summary.get("updates")
        settings = summary.get("settings", {})
        if not isinstance(updates, list) or not updates:
            raise ValueError(f"{run_id} has no update history")
        steps_per_update = int(settings["num_envs"]) * int(settings["steps"])
        if len(updates) != int(summary["local_updates"]):
            raise ValueError(f"{run_id} update count disagrees with local_updates")
        if len(updates) * steps_per_update != int(summary["local_steps"]):
            raise ValueError(f"{run_id} update steps disagree with local_steps")
        for update in updates:
            rows.append(
                {
                    "run_id": run_id,
                    "update": update["update"],
                    "env_steps": int(update["update"]) * steps_per_update,
                    "loss": update.get("loss", ""),
                    "policy_loss": update.get("policy_loss", ""),
                    "value_loss": update.get("value_loss", ""),
                    "approx_kl": update.get("approx_kl", ""),
                    "entropy": update.get("entropy", ""),
                    "gradient_norm": update.get("gradient_norm", ""),
                    "epochs_completed": update.get("epochs_completed", ""),
                    "kl_early_stop": update.get("kl_early_stop", ""),
                    "reward": "",
                    "reward_status": "not_recorded_in_source_summary",
                }
            )
    return rows


def extract_bridge_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "phase",
        "env_steps",
        "steps",
        "seconds",
        "mean_reward",
        "mean_raw_return",
        "returns",
        "lengths",
        "optimizer_steps",
        "mean_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
    ]
    extracted: list[dict[str, Any]] = []
    for row in rows:
        output = {key: row.get(key, "") for key in fields}
        for key in ("returns", "lengths"):
            if isinstance(output[key], list):
                output[key] = json.dumps(output[key], separators=(",", ":"))
        extracted.append(output)
    return extracted


def extract_basketball_evaluation(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "seed",
        "command_forward_mps",
        "command_lateral_mps",
        "command_yaw_rad_s",
        "horizon_s",
        "measured_steps",
        "survived",
        "signed_commanded_progress_m",
        "integrated_absolute_ball_travel_m",
        "integrated_ball_rotation_rad",
        "foot_ball_contact_fraction",
        "max_tilt_deg",
        "max_root_ball_offset_m",
        "mean_abs_body_velocity_tracking_error_mps",
        "sustained_balance",
        "controlled_rolling",
        "random_ball_movement",
        "learned_success",
    ]
    rows: list[dict[str, Any]] = []
    for trial in evaluation["trials"]:
        command = trial["command"]
        row = {key: trial.get(key, "") for key in fields}
        row.update(
            {
                "command_forward_mps": command[0],
                "command_lateral_mps": command[1],
                "command_yaw_rad_s": command[2],
            }
        )
        rows.append(row)
    return rows


def extract_bridge_evaluation(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    fields = [
        "env_index",
        "episode_index",
        "measured_steps",
        "max_progress_m",
        "bridge_contact_observed",
        "floor_contact_steps",
        "nonfoot_support_steps",
        "assisted_steps",
        "terminated",
        "truncated",
        "crossed",
        "passed",
        "failures",
    ]
    rows: list[dict[str, Any]] = []
    for episode in evaluation["bridge_assessment"]["episodes"]:
        row = {key: episode.get(key, "") for key in fields}
        row["failures"] = " | ".join(episode.get("failures", []))
        rows.append(row)
    return rows


def write_extracted_data(
    basketball_summaries: dict[str, dict[str, Any]],
    bridge_metrics: list[dict[str, Any]],
    basketball_evaluation: dict[str, Any],
    bridge_evaluation: dict[str, Any],
    bridge_result: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    basketball_updates = extract_basketball_updates(basketball_summaries)
    bridge_metric_rows = extract_bridge_metrics(bridge_metrics)
    basketball_trials = extract_basketball_evaluation(basketball_evaluation)
    bridge_episodes = extract_bridge_evaluation(bridge_evaluation)

    write_csv(
        RAW / "basketball_training_updates.csv",
        [
            "run_id",
            "update",
            "env_steps",
            "loss",
            "policy_loss",
            "value_loss",
            "approx_kl",
            "entropy",
            "gradient_norm",
            "epochs_completed",
            "kl_early_stop",
            "reward",
            "reward_status",
        ],
        basketball_updates,
    )
    write_csv(
        RAW / "bridge_training_metrics.csv",
        [
            "phase",
            "env_steps",
            "steps",
            "seconds",
            "mean_reward",
            "mean_raw_return",
            "returns",
            "lengths",
            "optimizer_steps",
            "mean_loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "approximate_kl",
            "clip_fraction",
            "explained_variance",
        ],
        bridge_metric_rows,
    )
    write_csv(
        RAW / "basketball_evaluation_trials.csv",
        [
            "seed",
            "command_forward_mps",
            "command_lateral_mps",
            "command_yaw_rad_s",
            "horizon_s",
            "measured_steps",
            "survived",
            "signed_commanded_progress_m",
            "integrated_absolute_ball_travel_m",
            "integrated_ball_rotation_rad",
            "foot_ball_contact_fraction",
            "max_tilt_deg",
            "max_root_ball_offset_m",
            "mean_abs_body_velocity_tracking_error_mps",
            "sustained_balance",
            "controlled_rolling",
            "random_ball_movement",
            "learned_success",
        ],
        basketball_trials,
    )
    write_csv(
        RAW / "bridge_evaluation_episodes.csv",
        [
            "env_index",
            "episode_index",
            "measured_steps",
            "max_progress_m",
            "bridge_contact_observed",
            "floor_contact_steps",
            "nonfoot_support_steps",
            "assisted_steps",
            "terminated",
            "truncated",
            "crossed",
            "passed",
            "failures",
        ],
        bridge_episodes,
    )

    run_rows = []
    basketball_source_keys = {
        "local": "basketball_local_summary",
        "mixed": "basketball_mixed_summary",
    }
    for run_id, summary in basketball_summaries.items():
        settings = summary["settings"]
        run_rows.append(
            {
                "case": "basketball",
                "run_id": run_id,
                "source": relative(SOURCES[basketball_source_keys[run_id]]),
                "seed": settings["seed"],
                "num_envs": settings["num_envs"],
                "rollout_steps_per_env": settings["steps"],
                "updates": summary["local_updates"],
                "recorded_env_steps": summary["local_steps"],
                "independence_note": (
                    "separate continuation run from the same upstream checkpoint; "
                    "fresh local critic and optimizer"
                ),
            }
        )
    run_rows.append(
        {
            "case": "bridge",
            "run_id": "bridge_studio_02",
            "source": relative(SOURCES["bridge_training_metrics"]),
            "seed": read_json(SOURCES["bridge_checkpoint_metadata"])["metadata"]["seed"],
            "num_envs": read_json(SOURCES["bridge_checkpoint_metadata"])["metadata"][
                "num_envs"
            ],
            "rollout_steps_per_env": read_json(SOURCES["bridge_checkpoint_metadata"])[
                "metadata"
            ]["ppo"]["num_steps"],
            "updates": len([row for row in bridge_metrics if row["phase"] == "update"]),
            "recorded_env_steps": bridge_result["trained_steps"],
            "independence_note": (
                "single bridge-studio-02 run warm-started from alpha_walking actor; "
                "fresh critic and optimizer"
            ),
        }
    )
    write_csv(
        RAW / "run_inventory.csv",
        [
            "case",
            "run_id",
            "source",
            "seed",
            "num_envs",
            "rollout_steps_per_env",
            "updates",
            "recorded_env_steps",
            "independence_note",
        ],
        run_rows,
    )

    return {
        "basketball_updates": basketball_updates,
        "bridge_metrics": bridge_metric_rows,
        "basketball_trials": basketball_trials,
        "bridge_episodes": bridge_episodes,
        "run_inventory": run_rows,
    }


def plot_basketball_training(
    summaries: dict[str, dict[str, Any]],
) -> tuple[Path, str]:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    specs = [
        ("local", "Fixed command", COLORS["local"]),
        ("mixed", "Randomized commands", COLORS["mixed"]),
    ]
    for run_id, label, color in specs:
        summary = summaries[run_id]
        updates = summary["updates"]
        step_size = summary["settings"]["num_envs"] * summary["settings"]["steps"]
        x = [item["update"] * step_size for item in updates]
        axes[0, 0].plot(x, [item["loss"] for item in updates], color=color, label=label)
        axes[0, 1].plot(
            x, [item["value_loss"] for item in updates], color=color, label=label
        )
        axes[1, 0].plot(
            x, [item["policy_loss"] for item in updates], color=color, label=label
        )

    axes[0, 0].set_title("Recorded PPO total loss")
    axes[0, 0].set_ylabel("loss (source field: loss)")
    axes[0, 1].set_title("Recorded value loss")
    axes[0, 1].set_ylabel("value loss (source field: value_loss)")
    axes[1, 0].set_title("Recorded policy loss")
    axes[1, 0].set_ylabel("policy loss (source field: policy_loss)")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("environment transitions (update x envs x rollout steps)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5,
        0.62,
        "REWARD HISTORY NOT RECORDED",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=COLORS["red"],
        transform=axes[1, 1].transAxes,
    )
    axes[1, 1].text(
        0.5,
        0.38,
        "The selected basketball summary.json files contain\n"
        "loss, policy_loss, value_loss, KL, entropy, and gradient norm,\n"
        "but no per-update reward or episode-return series.\n"
        "No reward curve is inferred or synthesized.",
        ha="center",
        va="center",
        color=COLORS["ink"],
        transform=axes[1, 1].transAxes,
    )
    fig.suptitle(
        "Basketball continuation training: observed loss histories",
        fontsize=14,
        fontweight="bold",
    )
    path = EVIDENCE / "basketball-training-curves.png"
    save_figure(fig, path)
    return (
        path,
        "Observed basketball PPO losses for two separate local continuation runs. "
        "The empty reward panel documents that reward history is absent from both "
        "source summaries.",
    )


def plot_basketball_reward_trace() -> tuple[Path, str] | None:
    csv_path = RAW / "basketball_reward_trace.csv"
    json_path = RAW / "basketball_reward_trace.json"
    if not csv_path.is_file() or not json_path.is_file():
        return None
    metadata = read_json(json_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("basketball reward diagnostic CSV is empty")
    required = {"step", "elapsed_s", "reward", "cumulative_reward"}
    if not required.issubset(rows[0]):
        raise ValueError("basketball reward diagnostic CSV has unexpected schema")

    elapsed = [float(row["elapsed_s"]) for row in rows]
    rewards = [float(row["reward"]) for row in rows]
    cumulative = [float(row["cumulative_reward"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), constrained_layout=True)
    axes[0].plot(elapsed, rewards, color=COLORS["reward"], linewidth=1.2)
    axes[0].plot(
        elapsed,
        rolling_mean(rewards, 25),
        color=COLORS["orange"],
        linewidth=1.8,
        label="trailing mean (25 steps)",
    )
    axes[0].set_ylabel("environment reward per step")
    axes[0].set_title("Per-step reward returned by BasketballEnv")
    axes[0].legend()
    axes[1].plot(elapsed, cumulative, color=COLORS["local"], linewidth=1.8)
    axes[1].set_ylabel("cumulative environment reward")
    axes[1].set_xlabel("elapsed simulation time (s)")
    axes[1].set_title("Cumulative reward within this diagnostic episode")
    axes[1].text(
        0.01,
        0.96,
        f"Policy SHA256 {metadata['policy_sha256'][:12]}... | "
        f"seed {metadata['settings']['seed']} | "
        f"command +0.15 m/s | recorded {metadata['recorded_steps']} steps | "
        f"no resets",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        color=COLORS["muted"],
    )
    fig.suptitle(
        "NEW deterministic diagnostic, NOT historical training reward",
        fontsize=14,
        fontweight="bold",
        color=COLORS["red"],
    )
    path = EVIDENCE / "basketball-reward-trace.png"
    save_figure(fig, path)
    return (
        path,
        "NEW deterministic diagnostic, NOT historical training reward: per-step "
        "and cumulative rewards from one fixed-command BasketballEnv rollout, "
        "with recurrent ONNX state carried and no resets.",
    )


def plot_bridge_training(rows: list[dict[str, Any]]) -> tuple[Path, str]:
    collection = [row for row in rows if row["phase"] == "collection"]
    updates = [row for row in rows if row["phase"] == "update"]
    episodes = [row for row in rows if row["phase"] == "episodes"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    axes[0, 0].plot(
        [row["env_steps"] for row in collection],
        [row["mean_reward"] for row in collection],
        color=COLORS["reward"],
        linewidth=1.6,
    )
    axes[0, 0].set_title("Collection mean reward")
    axes[0, 0].set_ylabel("mean reward per transition\n(source: mean_reward)")

    episode_x = [row["env_steps"] for row in episodes]
    episode_y = [row["mean_raw_return"] for row in episodes]
    axes[0, 1].scatter(
        episode_x, episode_y, color=COLORS["gold"], alpha=0.35, s=14, label="episode"
    )
    axes[0, 1].plot(
        episode_x,
        rolling_mean(episode_y, 10),
        color=COLORS["orange"],
        linewidth=1.8,
        label="trailing mean (10 episodes)",
    )
    axes[0, 1].set_title("Completed-episode raw return")
    axes[0, 1].set_ylabel("raw episode return\n(source: mean_raw_return)")
    axes[0, 1].legend()

    update_x = [row["env_steps"] for row in updates]
    axes[1, 0].plot(
        update_x,
        [row["mean_loss"] for row in updates],
        color=COLORS["loss"],
        label="mean_loss",
    )
    axes[1, 0].plot(
        update_x,
        [row["value_loss"] for row in updates],
        color=COLORS["local"],
        linestyle="--",
        label="value_loss",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("PPO optimizer losses")
    axes[1, 0].set_ylabel("loss (log scale; recorded fields)")
    axes[1, 0].legend()

    axes[1, 1].plot(
        update_x,
        [row["policy_loss"] for row in updates],
        color=COLORS["mixed"],
        label="policy_loss",
    )
    axes[1, 1].axhline(0, color=COLORS["muted"], linewidth=0.8)
    axes[1, 1].set_title("PPO policy loss")
    axes[1, 1].set_ylabel("policy loss (source: policy_loss)")

    for ax in axes.flat:
        ax.set_xlabel("environment transitions (source: env_steps)")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.suptitle(
        "Bridge-studio-02: observed reward and loss histories",
        fontsize=14,
        fontweight="bold",
    )
    path = EVIDENCE / "bridge-training-curves.png"
    save_figure(fig, path)
    return (
        path,
        "Observed bridge-studio-02 collection rewards, episode returns, and PPO "
        "losses over 32,768 recorded environment transitions.",
    )


def plot_evaluation(
    basketball_evaluation: dict[str, Any], bridge_evaluation: dict[str, Any]
) -> tuple[Path, str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), constrained_layout=True)

    trials = [
        trial
        for trial in basketball_evaluation["trials"]
        if trial["command"][0] == 0.15
    ]
    seeds = [str(trial["seed"]) for trial in trials]
    progress = [trial["signed_commanded_progress_m"] for trial in trials]
    absolute = [trial["integrated_absolute_ball_travel_m"] for trial in trials]
    width = 0.36
    positions = list(range(len(seeds)))
    axes[0].bar(
        [value - width / 2 for value in positions],
        progress,
        width,
        color=COLORS["local"],
        label="signed commanded progress",
    )
    axes[0].bar(
        [value + width / 2 for value in positions],
        absolute,
        width,
        color=COLORS["gold"],
        label="absolute ball travel",
    )
    criterion = (
        trials[0]["criteria"]["min_command_progress_ratio"]
        * trials[0]["command"][0]
        * trials[0]["horizon_s"]
    )
    axes[0].axhline(
        criterion,
        color=COLORS["red"],
        linestyle="--",
        label=f"progress criterion = {criterion:.2f} m",
    )
    axes[0].set_xticks(positions, seeds)
    axes[0].set_xlabel("evaluation seed (+0.15 m/s command)")
    axes[0].set_ylabel("distance (m)")
    axes[0].set_title("Basketball: movement was not controlled rolling")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axes[0].text(
        0.02,
        0.93,
        "All 3 trials survived 60 s; 0/3 met controlled-rolling criteria.",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        color=COLORS["ink"],
    )

    episodes = bridge_evaluation["bridge_assessment"]["episodes"]
    bridge_progress = [episode["max_progress_m"] for episode in episodes]
    axes[1].bar(
        range(1, len(episodes) + 1),
        bridge_progress,
        color=COLORS["bridge"],
    )
    threshold = bridge_evaluation["bridge_assessment"]["criteria"]["min_progress_m"]
    axes[1].axhline(
        threshold,
        color=COLORS["red"],
        linestyle="--",
        label=f"end-to-end target = {threshold:.1f} m",
    )
    axes[1].set_xlabel("evaluated episode record")
    axes[1].set_ylabel("maximum progress (m)")
    axes[1].set_title("Bridge: no evaluated episode crossed")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    axes[1].text(
        0.02,
        0.93,
        f"Best observed progress: {max(bridge_progress):.3f} m; "
        f"passed: {bridge_evaluation['bridge_assessment']['passed_episodes']}/"
        f"{len(episodes)}.",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        color=COLORS["ink"],
    )

    fig.suptitle(
        "Independent evaluation evidence (not training curves)",
        fontsize=14,
        fontweight="bold",
    )
    path = EVIDENCE / "evaluation-outcomes.png"
    save_figure(fig, path)
    return (
        path,
        "Observed independent evaluation outcomes: commanded basketball travel "
        "and bridge maximum progress. Thresholds are computed directly from the "
        "recorded evaluation criteria.",
    )


def diagram_header(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(
        0,
        5.72,
        "SCHEMATIC - NOT OBSERVED EVIDENCE",
        color=COLORS["red"],
        fontsize=8,
        fontweight="bold",
    )
    ax.text(0, 5.25, title, color=COLORS["ink"], fontsize=15, fontweight="bold")
    ax.text(0, 4.88, subtitle, color=COLORS["muted"], fontsize=9)


def draw_pipeline_diagram() -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    diagram_header(
        ax,
        "Evidence preparation pipeline",
        "Source artifacts remain distinct from extracted data, plots, and diagrams.",
    )
    boxes = [
        (0.2, "Run artifacts\nJSON / JSONL", COLORS["reward"]),
        (2.7, "Schema checks\n+ CSV extraction", COLORS["local"]),
        (5.2, "Observed plots\n+ copied frames", COLORS["gold"]),
        (7.7, "Book assets\nhashes + captions", COLORS["mixed"]),
    ]
    for x, label, color in boxes:
        ax.add_patch(
            Rectangle((x, 2.15), 1.8, 1.35, facecolor="white", edgecolor=color, lw=2)
        )
        ax.text(x + 0.9, 2.82, label, ha="center", va="center", color=COLORS["ink"])
    for x in (2.05, 4.55, 7.05):
        ax.add_patch(
            FancyArrowPatch(
                (x, 2.82),
                (x + 0.55, 2.82),
                arrowstyle="-|>",
                mutation_scale=14,
                color=COLORS["muted"],
                lw=1.4,
            )
        )
    ax.text(
        5,
        1.25,
        "Rule: missing fields remain missing. No synthetic training series.",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["red"],
    )
    path = DIAGRAMS / "schematic-evidence-pipeline.png"
    save_figure(fig, path)
    return path, "Schematic of the deterministic evidence-preparation pipeline."


def draw_basketball_geometry() -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    diagram_header(
        ax,
        "Basketball balance geometry",
        "Dimensions shown are model constants; robot drawing is illustrative.",
    )
    ground_y = 1.0
    ball_center = (5.0, 2.15)
    ball_radius = 1.05
    ax.plot([0.8, 9.2], [ground_y, ground_y], color=COLORS["platform"], lw=2)
    ax.add_patch(
        Circle(
            ball_center,
            ball_radius,
            facecolor="#F39C12",
            edgecolor=COLORS["orange"],
            lw=2,
        )
    )
    ax.plot([4.55, 4.75], [3.14, 3.65], color=COLORS["ink"], lw=4)
    ax.plot([5.45, 5.25], [3.14, 3.65], color=COLORS["ink"], lw=4)
    ax.add_patch(Rectangle((4.55, 3.62), 0.9, 0.65, facecolor="#D6EAF8", ec=COLORS["ink"]))
    ax.add_patch(Circle((5.0, 4.55), 0.28, facecolor="#F4D03F", ec=COLORS["ink"]))
    ax.add_patch(
        FancyArrowPatch(
            (5.0, 2.15),
            (6.05, 2.15),
            arrowstyle="<->",
            mutation_scale=14,
            color=COLORS["red"],
            lw=1.5,
        )
    )
    ax.text(5.55, 2.32, "radius = 0.12 m", color=COLORS["red"], fontsize=9)
    ax.add_patch(
        Arc(
            ball_center,
            1.5,
            1.5,
            angle=0,
            theta1=25,
            theta2=135,
            color=COLORS["reward"],
            lw=2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (4.42, 2.64),
            (4.26, 2.38),
            arrowstyle="-|>",
            mutation_scale=12,
            color=COLORS["reward"],
        )
    )
    ax.text(2.05, 3.1, "free 6-DoF ball\nmass = 0.62 kg", color=COLORS["ink"])
    ax.add_patch(
        FancyArrowPatch(
            (6.35, 3.7),
            (8.2, 3.7),
            arrowstyle="-|>",
            mutation_scale=15,
            color=COLORS["green"],
            lw=2,
        )
    )
    ax.text(6.35, 3.95, "velocity command", color=COLORS["green"])
    ax.text(
        1.0,
        0.35,
        "Actor input: 61D proprioception and commands; no ball state in actor input.",
        color=COLORS["muted"],
        fontsize=9,
    )
    path = DIAGRAMS / "schematic-basketball-geometry.png"
    save_figure(fig, path)
    return path, "Schematic basketball geometry using recorded model constants."


def draw_bridge_geometry() -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    diagram_header(
        ax,
        "Suspended bridge geometry",
        "Model dimensions and crossing target; simplified side/top projection.",
    )
    ax.add_patch(
        Rectangle((0.8, 1.05), 2.0, 0.8, facecolor="#AAB7B8", ec=COLORS["platform"])
    )
    ax.add_patch(
        Rectangle((7.2, 1.05), 2.0, 0.8, facecolor="#AAB7B8", ec=COLORS["platform"])
    )
    ax.add_patch(
        Rectangle((2.8, 1.55), 4.4, 0.22, facecolor="#AF601A", ec=COLORS["bridge"])
    )
    for x in (3.2, 6.8):
        ax.plot([x, x], [1.0, 4.2], color=COLORS["platform"], lw=3)
        ax.plot([x - 0.1, x + 0.1], [4.2, 4.2], color=COLORS["platform"], lw=3)
        ax.plot([x, x + (1.2 if x < 5 else -1.2)], [4.0, 1.75], color=COLORS["muted"], lw=1.5)
    ax.add_patch(
        FancyArrowPatch(
            (2.8, 2.15),
            (7.2, 2.15),
            arrowstyle="<->",
            mutation_scale=14,
            color=COLORS["red"],
        )
    )
    ax.text(5.0, 2.32, "plank length = 1.10 m", ha="center", color=COLORS["red"])
    ax.add_patch(
        FancyArrowPatch(
            (1.6, 0.62),
            (8.4, 0.62),
            arrowstyle="<->",
            mutation_scale=14,
            color=COLORS["reward"],
        )
    )
    ax.text(
        5.0,
        0.28,
        "evaluation end-to-end progress target = 1.60 m",
        ha="center",
        color=COLORS["reward"],
    )
    ax.text(4.95, 1.18, "width = 0.13 m", ha="center", color=COLORS["ink"])
    ax.text(0.9, 1.95, "start platform", color=COLORS["ink"])
    ax.text(7.45, 1.95, "end platform", color=COLORS["ink"])
    ax.text(
        5.0,
        4.45,
        "Four spring-damper suspension tendons; free plank body",
        ha="center",
        color=COLORS["muted"],
    )
    path = DIAGRAMS / "schematic-bridge-geometry.png"
    save_figure(fig, path)
    return path, "Schematic suspended-bridge geometry and evaluation target."


def draw_ppo_clipping() -> tuple[Path, str]:
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    ax.set_xlim(0.55, 1.45)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True)
    ax.set_xlabel("probability ratio r = new policy / old policy")
    ax.set_ylabel("surrogate objective contribution")
    ax.set_title(
        "PPO clipping (theoretical schematic, not measured training data)",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.56,
        1.37,
        "SCHEMATIC - THEORETICAL CURVES",
        color=COLORS["red"],
        fontsize=8,
        fontweight="bold",
    )
    epsilon = 0.1
    ratios = [0.55 + index * 0.002 for index in range(451)]
    positive = [min(ratio, 1 + epsilon) for ratio in ratios]
    negative = [-max(ratio, 1 - epsilon) for ratio in ratios]
    ax.plot(ratios, positive, color=COLORS["green"], lw=2, label="positive advantage")
    ax.plot(ratios, negative, color=COLORS["mixed"], lw=2, label="negative advantage")
    ax.axvspan(1 - epsilon, 1 + epsilon, color="#D5F5E3", alpha=0.55)
    ax.axvline(1 - epsilon, color=COLORS["muted"], linestyle="--")
    ax.axvline(1 + epsilon, color=COLORS["muted"], linestyle="--")
    ax.text(0.9, -1.35, "1 - epsilon", ha="center", color=COLORS["muted"])
    ax.text(1.1, -1.35, "1 + epsilon", ha="center", color=COLORS["muted"])
    ax.legend(loc="center right")
    ax.text(
        0.57,
        -1.12,
        "Illustration uses epsilon = 0.10, matching bridge-studio-02 metadata.",
        color=COLORS["ink"],
        fontsize=9,
    )
    path = DIAGRAMS / "schematic-ppo-clipping.png"
    save_figure(fig, path)
    return path, "Theoretical PPO clipping schematic, explicitly not observed data."


def write_readme(
    basketball_summaries: dict[str, dict[str, Any]],
    bridge_metrics: list[dict[str, Any]],
    basketball_evaluation: dict[str, Any],
    bridge_evaluation: dict[str, Any],
) -> None:
    phase_counts = {
        phase: len([row for row in bridge_metrics if row["phase"] == phase])
        for phase in ("collection", "update", "episodes")
    }
    bridge_progress = [
        episode["max_progress_m"]
        for episode in bridge_evaluation["bridge_assessment"]["episodes"]
    ]
    readme = f"""# Basketball and bridge book assets

Generated by `../tools/prepare_assets.py` from repository-local artifacts. The
script installs nothing and uses the existing Python, Matplotlib, and Pillow
environment.

## Evidence versus diagrams

- `evidence/` contains observed plots made from recorded values and byte-for-byte
  copies of genuine screenshots/contact sheets.
- `diagrams/` contains explanatory drawings. Every diagram is visibly labeled
  **SCHEMATIC** and must not be presented as rollout or measurement evidence.
- `raw/` contains copied source records and extracted CSV tables.

## Recorded training schemas

### Basketball

Two separate continuation runs are plotted:

- `basketball_local`: {basketball_summaries["local"]["local_updates"]} updates,
  {basketball_summaries["local"]["local_steps"]:,} environment transitions.
- `basketball_mixed`: {basketball_summaries["mixed"]["local_updates"]} updates,
  {basketball_summaries["mixed"]["local_steps"]:,} environment transitions.

Both summaries record `loss`, `policy_loss`, `value_loss`, `approx_kl`,
`entropy`, and `gradient_norm` per update. They do **not** record per-update
reward, mean reward, or episode return. The basketball figure therefore shows
the real loss curves and an explicit missing-reward panel; it does not synthesize
a reward curve. These are separate local continuation runs from the same
upstream checkpoint, each described as an actor warm-start with a fresh local
critic and optimizer.

If present, `evidence/basketball-reward-trace.png` is a separate **new
deterministic diagnostic**, not historical training reward. It is collected by
`../tools/collect_reward_trace.py` from the fixed-command ONNX policy in one
`BasketballEnv` episode with recurrent state carried, no resets, and first-done
stopping. Its CSV and settings receipt live in `raw/basketball_reward_trace.*`.

### Bridge

`bridge-studio-02/training-metrics.jsonl` contains
{phase_counts["collection"]} collection rows, {phase_counts["update"]} update
rows, and {phase_counts["episodes"]} episode rows through 32,768 environment
transitions. Reward axes use the exact fields `mean_reward` (collection mean per
transition) and `mean_raw_return` (completed episode return). Loss axes use
`mean_loss`, `value_loss`, and `policy_loss`; the total/value panel uses a log
scale because the recorded magnitudes are large.

## Independent evaluation evidence

- Basketball: {basketball_evaluation["summary"]["trial_count"]} first-fall trials,
  seeds 101/202/303 at zero and +0.15 m/s forward commands. All survived the
  60-second horizon, but commanded rolling passed 0/3 trials.
- Bridge: {len(bridge_progress)} recorded episode assessments over 3,000
  transitions. Best progress was {max(bridge_progress):.6f} m against the
  1.6 m crossing target; passed episodes: 0.

Training curves and evaluation plots answer different questions and are kept in
separate files.

## Reproducibility and provenance

- `provenance.json` records every source path, source SHA256, copied raw path,
  and generated-file lineage.
- `SHA256SUMS` hashes every generated/copied asset except itself.
- `captions_manifest.json` labels each image as `observed_evidence`,
  `observed_plot`, or `schematic`.
- `raw/run_inventory.csv` lists known seeds, environment counts, rollout sizes,
  update counts, and environment-transition counts.

Run from the repository root:

```bash
# One-shot simulation collection; do not run during routine book builds.
rlx/.venv-microduck/bin/python \
  docs/basketball-bridge-book/tools/collect_reward_trace.py

# Deterministic extraction/plotting from existing records, including the trace.
python3 docs/basketball-bridge-book/tools/prepare_assets.py
```
"""
    (ASSETS / "README.md").write_text(readme, encoding="utf-8")


def build_provenance(
    copied_sources: dict[str, Path],
    generated_assets: list[dict[str, Any]],
) -> None:
    source_records = []
    for source_id, source in SOURCES.items():
        source_records.append(
            {
                "id": source_id,
                "source_path": relative(source),
                "source_sha256": sha256(source),
                "source_bytes": source.stat().st_size,
                "copied_path": relative(copied_sources[source_id]),
                "copied_sha256": sha256(copied_sources[source_id]),
            }
        )
    trace_csv = RAW / "basketball_reward_trace.csv"
    trace_json = RAW / "basketball_reward_trace.json"
    if trace_csv.is_file() and trace_json.is_file():
        trace_metadata = read_json(trace_json)
        source_records.append(
            {
                "id": "basketball_reward_trace_diagnostic",
                "source_path": relative(trace_csv),
                "source_sha256": sha256(trace_csv),
                "source_bytes": trace_csv.stat().st_size,
                "settings_path": relative(trace_json),
                "settings_sha256": sha256(trace_json),
                "label": trace_metadata["label"],
                "policy_sha256": trace_metadata["policy_sha256"],
                "collected_at_utc": trace_metadata["collected_at_utc"],
            }
        )
    data = {
        "schema_version": 1,
        "generator": relative(Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "policy": {
            "training_data": "observed repository artifacts only",
            "missing_values": "left absent and documented; never synthesized",
            "schematics": "clearly labeled and stored separately",
        },
        "sources": source_records,
        "generated_assets": generated_assets,
    }
    write_json(ASSETS / "provenance.json", data)


def write_checksums() -> None:
    files = sorted(
        path
        for path in ASSETS.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256(path)}  {path.relative_to(ASSETS).as_posix()}" for path in files]
    (ASSETS / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    validate_sources()
    reset_output_dirs()
    configure_plotting()

    basketball_summaries = {
        "local": read_json(SOURCES["basketball_local_summary"]),
        "mixed": read_json(SOURCES["basketball_mixed_summary"]),
    }
    bridge_metrics = read_jsonl(SOURCES["bridge_training_metrics"])
    basketball_evaluation = read_json(SOURCES["basketball_evaluation"])
    bridge_evaluation = read_json(SOURCES["bridge_evaluation"])
    bridge_result = read_json(SOURCES["bridge_result"])

    if max(row.get("env_steps", 0) for row in bridge_metrics) != bridge_result["trained_steps"]:
        raise ValueError("bridge metrics do not reach the result.json trained_steps")
    if not basketball_evaluation["summary"]["all_evaluated_seeds_reported"]:
        raise ValueError("basketball evaluation reports missing seeds")

    copied_sources = copy_raw_sources()
    extracted = write_extracted_data(
        basketball_summaries,
        bridge_metrics,
        basketball_evaluation,
        bridge_evaluation,
        bridge_result,
    )

    image_manifest = copy_observed_images()
    schematic_sources = {
        "schematic-evidence-pipeline.png": (
            "constructed explanatory diagram of this generator"
        ),
        "schematic-basketball-geometry.png": [
            relative(SOURCES["basketball_environment"]),
        ],
        "schematic-bridge-geometry.png": [
            relative(SOURCES["bridge_environment"]),
            relative(SOURCES["bridge_evaluation"]),
        ],
        "schematic-ppo-clipping.png": [
            relative(SOURCES["bridge_checkpoint_metadata"]),
        ],
    }
    draw_functions = [
        lambda: plot_basketball_training(basketball_summaries),
        lambda: plot_bridge_training(bridge_metrics),
        lambda: plot_evaluation(basketball_evaluation, bridge_evaluation),
        draw_pipeline_diagram,
        draw_basketball_geometry,
        draw_bridge_geometry,
        draw_ppo_clipping,
    ]
    reward_trace_plot = plot_basketball_reward_trace()
    if reward_trace_plot is not None:
        draw_functions.insert(1, lambda: reward_trace_plot)
    for draw in draw_functions:
        result = draw()
        if result is None:
            continue
        path, caption = result
        image_manifest.append(
            {
                "asset": relative(path),
                "asset_kind": (
                    "schematic" if path.parent == DIAGRAMS else "observed_plot"
                ),
                "caption": caption,
                "source": [
                    relative(SOURCES["basketball_local_summary"]),
                    relative(SOURCES["basketball_mixed_summary"]),
                ]
                if path.name == "basketball-training-curves.png"
                else [
                    relative(RAW / "basketball_reward_trace.csv"),
                    relative(RAW / "basketball_reward_trace.json"),
                ]
                if path.name == "basketball-reward-trace.png"
                else [
                    relative(SOURCES["bridge_training_metrics"]),
                ]
                if path.name == "bridge-training-curves.png"
                else [
                    relative(SOURCES["basketball_evaluation"]),
                    relative(SOURCES["bridge_evaluation"]),
                ]
                if path.name == "evaluation-outcomes.png"
                else schematic_sources[path.name],
            }
        )

    write_json(ASSETS / "captions_manifest.json", image_manifest)
    write_readme(
        basketball_summaries,
        bridge_metrics,
        basketball_evaluation,
        bridge_evaluation,
    )

    generated_assets = []
    for path in sorted(ASSETS.rglob("*")):
        if path.is_file() and path.name not in {"provenance.json", "SHA256SUMS"}:
            generated_assets.append(
                {
                    "path": relative(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "derived_from": (
                        "extracted repository source data"
                        if path.suffix == ".csv"
                        else "see captions_manifest.json"
                        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                        else "copied repository source or generated documentation"
                    ),
                }
            )
    build_provenance(copied_sources, generated_assets)
    write_checksums()

    print(
        json.dumps(
            {
                "assets": relative(ASSETS),
                "basketball_update_rows": len(extracted["basketball_updates"]),
                "bridge_metric_rows": len(extracted["bridge_metrics"]),
                "basketball_evaluation_trials": len(extracted["basketball_trials"]),
                "bridge_evaluation_episodes": len(extracted["bridge_episodes"]),
                "images": len(image_manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
