"""Reproducibly audit a trained backflip policy against named controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audit_dance import studio_module


CONTROL_HZ = 50
ROLLOUT_STEPS = 600
DEFAULT_SEEDS = (301, 302, 303, 304, 305, 306, 307, 308)
POLICY_NAMES = ("final", "initialization", "zero", "stand")
METRIC_NAMES = (
    "rotation_rad",
    "spotter_active",
    "assist_torque_norm",
    "launch_complete",
    "landing_policy_steps",
    "handed_off",
    "upright",
    "height_m",
    "both_feet",
    "nonfoot_contact",
    "gyro_norm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    events = [
        json.loads(line)
        for line in payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    return events, hashlib.sha256(payload).hexdigest()


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_finite_json(value), indent=2, allow_nan=False) + "\n"
    )


def _parse_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--seeds must be a comma-separated list of integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must not contain duplicates")
    return seeds


def _metadata(path: Path) -> dict[str, Any]:
    document = _read_json(_sidecar(path))
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"checkpoint sidecar has no metadata object: {_sidecar(path)}")
    return metadata


def _single_policy(infer_batch: Callable[[Any], Any]) -> Callable[[np.ndarray], np.ndarray]:
    def infer(observation: np.ndarray) -> np.ndarray:
        action = np.asarray(infer_batch(observation[None, :]), dtype=np.float32)
        if action.shape != (1, 14):
            raise ValueError(f"policy returned {action.shape}, expected (1, 14)")
        return action[0]

    return infer


def _checkpoint_policy(studio: Any, checkpoint: Path) -> Callable[[np.ndarray], np.ndarray]:
    loader = getattr(studio, "_policy_from_checkpoint", None)
    if loader is not None:
        return _single_policy(loader(checkpoint))
    return studio._render_policy_source(checkpoint, "checkpoint")


def _onnx_policy(studio: Any, policy: Path) -> Callable[[np.ndarray], np.ndarray]:
    loader = getattr(studio, "_policy_from_onnx", None)
    if loader is not None:
        return _single_policy(loader(policy))
    return studio._render_policy_source(policy, "policy")


def _stable(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["upright"] >= 0.9
        and metrics["height_m"] >= 0.105
        and metrics["both_feet"] == 1.0
        and metrics["nonfoot_contact"] == 0.0
        and metrics["gyro_norm"] < math.sqrt(2.0)
        and metrics["spotter_active"] == 0.0
        and metrics["assist_torque_norm"] == 0.0
    )


def rollout(
    infer: Callable[[np.ndarray], np.ndarray],
    *,
    mode: str,
    seed: int,
    reward_weights: dict[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    from rlx.environments.backflip import BackflipEnv
    from rlx.environments.backflip_evaluation import BackflipEvaluation

    env = BackflipEnv(
        backflip_mode=mode,
        seed=seed,
        actuator="xml",
        max_episode_s=ROLLOUT_STEPS / CONTROL_HZ,
        domain_rand=False,
        obs_noise=False,
        action_delay=False,
        random_yaw=False,
        weight_overrides=reward_weights,
    )
    assessor = BackflipEvaluation(1, ROLLOUT_STEPS) if mode == "showcase" else None
    trace: dict[str, list[Any]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "rewards",
            "qpos",
            "terminated",
            "truncated",
            "stage",
            *METRIC_NAMES,
        )
    }
    terminated = truncated = False
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(ROLLOUT_STEPS):
            action = np.asarray(infer(observation), dtype=np.float32)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError(
                    f"policy action must be finite shape (14,), got {action.shape}"
                )
            trace["observations"].append(observation.copy())
            trace["actions"].append(action.copy())
            observation, reward, terminated, truncated, _ = env.step(action)
            metrics = env.recipe_metrics()
            if assessor is not None:
                assessor.observe(0, metrics, terminated=terminated, truncated=truncated)
            trace["rewards"].append(float(reward))
            trace["qpos"].append(env.unwrapped.data.qpos.copy())
            trace["terminated"].append(bool(terminated))
            trace["truncated"].append(bool(truncated))
            trace["stage"].append(
                {"launch": 0, "landing": 1, "stand": 2}[env.unwrapped.stage]
            )
            for name in METRIC_NAMES:
                trace[name].append(float(metrics[name]))
            if terminated or truncated:
                break
    finally:
        env.close()

    arrays = {name: np.asarray(values) for name, values in trace.items()}
    steps = len(arrays["rewards"])
    final_metrics = {
        name: float(arrays[name][-1]) if steps else None for name in METRIC_NAMES
    }
    stable_values = np.asarray(
        [
            _stable({name: float(arrays[name][index]) for name in METRIC_NAMES})
            for index in range(steps)
        ],
        dtype=np.bool_,
    )
    arrays["stable"] = stable_values
    last_100_complete = steps >= 100
    stable_last_100_steps = (
        int(np.count_nonzero(stable_values[-100:])) if last_100_complete else 0
    )
    common = {
        "seed": seed,
        "mode": mode,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "raw_return": float(arrays["rewards"].sum()),
        "reward_per_step": (
            float(arrays["rewards"].mean()) if steps else None
        ),
        "final": final_metrics,
        "stable_last_100_steps": stable_last_100_steps,
        "stable_last_100_passed": bool(
            steps == ROLLOUT_STEPS
            and truncated
            and not terminated
            and stable_last_100_steps == 100
        ),
        "finite": all(
            bool(np.isfinite(values).all())
            for name, values in arrays.items()
            if name not in {"terminated", "truncated", "stable"}
        ),
    }
    if assessor is not None:
        assessment = assessor.report()
        common.update(
            assessment=assessment,
            passed=bool(assessment["passed"] and common["finite"]),
            acceptance_scope="strict assisted showcase assessor",
        )
    else:
        common.update(
            passed=bool(common["stable_last_100_passed"] and common["finite"]),
            acceptance_scope=(
                "unassisted assisted-release landing; no stand-policy handoff; "
                "all final 100 steps physically stable"
            ),
        )
    return common, arrays


def _save_trace(
    output: Path,
    *,
    policy_name: str,
    mode: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    prefix: str = "",
) -> str:
    name = f"{prefix}{policy_name}-{mode}-seed{seed}.npz"
    np.savez_compressed(output / name, **arrays)
    return name


def _training_evidence(
    events: list[dict[str, Any]],
    *,
    requested_steps: int,
    final_checkpoint_steps: int,
) -> dict[str, Any]:
    collections = [event for event in events if event.get("phase") == "collection"]
    updates = [event for event in events if event.get("phase") == "update"]
    episodes = [event for event in events if event.get("phase") == "episodes"]
    required_update_keys = (
        "mean_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
    )
    finite = bool(updates) and all(
        all(
            key in event
            and isinstance(event[key], (int, float))
            and math.isfinite(float(event[key]))
            for key in required_update_keys
        )
        for event in updates
    )
    update_steps = [int(event["env_steps"]) for event in updates]
    monotonic = all(
        current > previous
        for previous, current in zip(update_steps, update_steps[1:])
    )
    observed_steps = sum(int(event.get("steps", 0)) for event in collections)
    return {
        "normalization": {
            "collection_mean_reward": "normalized rollout reward per transition",
            "episode_mean_raw_return": "raw, unnormalized completed-episode return",
        },
        "finite_update_metrics": finite,
        "monotonic_update_steps": monotonic,
        "collections": len(collections),
        "updates": len(updates),
        "episode_reports": len(episodes),
        "optimizer_steps": sum(int(event.get("optimizer_steps", 0)) for event in updates),
        "observed_transitions": observed_steps,
        "requested_transitions": requested_steps,
        "final_checkpoint_steps": final_checkpoint_steps,
        "passed": bool(
            requested_steps > 0
            and finite
            and monotonic
            and observed_steps >= requested_steps
            and final_checkpoint_steps >= requested_steps
        ),
        "last_update": updates[-1] if updates else None,
    }


def _select_history(
    checkpoints: list[Path],
    *,
    final_steps: int,
) -> list[dict[str, Any]]:
    candidates = [
        {"path": checkpoint, "steps": int(_metadata(checkpoint)["steps"])}
        for checkpoint in checkpoints
    ]
    if not candidates:
        return []
    targets = (
        ("start", 0),
        ("quarter", round(final_steps * 0.25)),
        ("half", round(final_steps * 0.5)),
        ("last_available", max(item["steps"] for item in candidates)),
    )
    selected: list[dict[str, Any]] = []
    for label, target in targets:
        at_or_before = [
            candidate for candidate in candidates if candidate["steps"] <= target
        ]
        item = (
            max(at_or_before, key=lambda candidate: candidate["steps"])
            if at_or_before
            else min(candidates, key=lambda candidate: candidate["steps"])
        )
        selected.append(
            {
                "label": label,
                "target_steps": target,
                "steps": item["steps"],
                "path": item["path"],
            }
        )
    return selected


def _summarize_policy(
    evaluations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode, reports in evaluations.items():
        raw_returns = [float(report["raw_return"]) for report in reports]
        summary[mode] = {
            "seeds": [int(report["seed"]) for report in reports],
            "passed_seeds": sum(bool(report["passed"]) for report in reports),
            "all_seeds_passed": bool(reports) and all(
                bool(report["passed"]) for report in reports
            ),
            "raw_return_mean": float(np.mean(raw_returns)),
            "raw_return_min": float(np.min(raw_returns)),
            "raw_return_max": float(np.max(raw_returns)),
            "final_upright_mean": float(
                np.mean([report["final"]["upright"] for report in reports])
            ),
            "final_height_m_mean": float(
                np.mean([report["final"]["height_m"] for report in reports])
            ),
            "final_both_feet_fraction": float(
                np.mean([report["final"]["both_feet"] for report in reports])
            ),
            "final_gyro_norm_mean": float(
                np.mean([report["final"]["gyro_norm"] for report in reports])
            ),
            "stable_last_100_fraction_mean": float(
                np.mean(
                    [
                        report["stable_last_100_steps"] / 100.0
                        for report in reports
                    ]
                )
            ),
        }
    return summary


def _paired_landing_refinement(
    initial_reports: list[dict[str, Any]],
    candidate_reports: list[dict[str, Any]],
    *,
    candidate: str,
) -> dict[str, Any]:
    initial = {
        int(report["seed"]): float(report["raw_return"])
        for report in initial_reports
    }
    final = {
        int(report["seed"]): float(report["raw_return"])
        for report in candidate_reports
    }
    paired = []
    for seed in sorted(initial.keys() & final.keys()):
        delta = final[seed] - initial[seed]
        paired.append(
            {
                "seed": seed,
                "initialization_raw_return": initial[seed],
                "final_raw_return": final[seed],
                "delta": delta,
                "relative_gain": (
                    delta / abs(initial[seed])
                    if abs(initial[seed]) > 1e-12
                    else None
                ),
                "improved": delta > 0.0,
            }
        )
    initial_mean = float(np.mean([entry["initialization_raw_return"] for entry in paired]))
    final_mean = float(np.mean([entry["final_raw_return"] for entry in paired]))
    delta_mean = final_mean - initial_mean
    improved = sum(entry["delta"] > 0.0 for entry in paired)
    regressed = sum(entry["delta"] < 0.0 for entry in paired)
    return {
        "candidate": candidate,
        "criterion": (
            "positive aggregate paired unassisted-landing raw-return delta and "
            "more improved seeds than regressed seeds"
        ),
        "per_seed": paired,
        "aggregate": {
            "initialization_raw_return_mean": initial_mean,
            "final_raw_return_mean": final_mean,
            "delta_mean": delta_mean,
            "relative_gain": (
                delta_mean / abs(initial_mean) if abs(initial_mean) > 1e-12 else None
            ),
            "seeds_improved": improved,
            "seeds_regressed": regressed,
            "seeds_unchanged": len(paired) - improved - regressed,
            "seed_count": len(paired),
        },
        "passed": bool(paired and delta_mean > 0.0 and improved > regressed),
    }


def _landing_refinement(
    evaluations: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    return _paired_landing_refinement(
        evaluations["initialization"]["landing"],
        evaluations["final"]["landing"],
        candidate="final",
    )


def _transfer_metrics(value: Any, prefix: str = "") -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            metrics.extend(_transfer_metrics(item, name))
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        metrics.append((prefix, float(value)))
    return metrics


def _plot_mean_band(
    axis: Any,
    entries: list[dict[str, Any]],
    mode: str,
    key: Callable[[dict[str, Any]], float],
    *,
    label: str,
) -> None:
    steps = sorted({int(entry["steps"]) for entry in entries})
    means, lows, highs = [], [], []
    for step in steps:
        values = [
            key(report)
            for entry in entries
            if int(entry["steps"]) == step
            for report in entry["evaluations"][mode]
        ]
        means.append(float(np.mean(values)))
        lows.append(float(np.min(values)))
        highs.append(float(np.max(values)))
    axis.plot(steps, means, "o-", label=label)
    axis.fill_between(steps, lows, highs, alpha=0.15)


def plots(
    output: Path,
    *,
    events: list[dict[str, Any]],
    history: list[dict[str, Any]],
    imitation_history: list[dict[str, Any]],
    transfer_parity: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "legend.fontsize": 10})

    collections = [event for event in events if event.get("phase") == "collection"]
    episodes = [event for event in events if event.get("phase") == "episodes"]
    updates = [event for event in events if event.get("phase") == "update"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(
        [event["env_steps"] for event in collections],
        [event["mean_reward"] for event in collections],
        linewidth=0.7,
    )
    axes[0, 0].set(
        title="Normalized PPO collection reward",
        ylabel="Normalized mean reward per transition",
    )
    axes[0, 1].plot(
        [event["env_steps"] for event in episodes],
        [event["mean_raw_return"] for event in episodes],
        linewidth=0.7,
    )
    axes[0, 1].set(
        title="Raw stochastic training episode return",
        ylabel="Raw unnormalized return",
    )
    if history:
        _plot_mean_band(
            axes[1, 0],
            history,
            "showcase",
            lambda report: float(report["raw_return"]),
            label="Assisted showcase, deterministic checkpoint",
        )
        _plot_mean_band(
            axes[1, 0],
            history,
            "landing",
            lambda report: float(report["raw_return"]),
            label="Unassisted landing, deterministic checkpoint",
        )
        axes[1, 0].legend(fontsize=10)
    axes[1, 0].set(
        title="Raw deterministic checkpoint return",
        ylabel="Raw unnormalized 600-step return",
    )
    if imitation_history:
        axes[1, 1].plot(
            [entry["epoch"] for entry in imitation_history],
            [entry["mean_imitation_loss"] for entry in imitation_history],
            linewidth=0.9,
        )
        axes[1, 1].set(
            title="Legacy behavior-cloning initialization history",
            xlabel="Imitation epoch",
            ylabel="Mean imitation MSE",
        )
    elif transfer_parity is not None:
        metrics = _transfer_metrics(transfer_parity)
        error_metrics = [
            (name, value)
            for name, value in metrics
            if "error" in name.lower() or "diff" in name.lower()
        ]
        displayed = (error_metrics or metrics)[:8]
        if displayed:
            labels, values = zip(*displayed)
            axes[1, 1].bar(range(len(values)), values)
            axes[1, 1].set_xticks(
                range(len(labels)),
                [label.rsplit(".", 1)[-1] for label in labels],
                rotation=30,
                ha="right",
            )
            axes[1, 1].set_ylabel("Recorded parity value")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "transfer-parity.json contains no scalar metrics",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
        axes[1, 1].set(title="Exact initialization-transfer parity evidence")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No optional initialization history was supplied",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set(title="Initialization evidence")
    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xlabel("PPO environment transitions")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Backflip optimization history; smooth curves do not establish physical success"
    )
    figure.savefig(output / "reward-learning.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    keys = (
        "mean_loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "explained_variance",
        "seconds",
    )
    for axis, key in zip(axes.flat, keys):
        axis.plot(
            [event["env_steps"] for event in updates],
            [event[key] for event in updates],
            linewidth=0.7,
        )
        axis.set(title=key, xlabel="PPO environment transitions", ylabel=key)
        axis.grid(alpha=0.2)
    figure.suptitle("Raw PPO update metrics; unsmoothed full history")
    figure.savefig(output / "ppo-losses.png", dpi=300)
    plt.close(figure)
    for filename, subset in (("ppo-objectives.png", keys[:4]), ("ppo-diagnostics.png", keys[4:])):
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        for axis, key in zip(axes.flat, subset):
            axis.plot([event["env_steps"] for event in updates], [event[key] for event in updates], linewidth=0.9)
            axis.set(title=key, xlabel="PPO environment transitions", ylabel=key)
            axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            axis.grid(alpha=0.2)
        figure.suptitle("Backflip PPO — full, unsmoothed training history")
        figure.savefig(output / filename, dpi=300)
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    if history:
        for mode, label in (
            ("showcase", "Assisted showcase"),
            ("landing", "Unassisted landing"),
        ):
            _plot_mean_band(
                axes[0, 0],
                history,
                mode,
                lambda report: float(report["final"]["upright"]),
                label=label,
            )
            _plot_mean_band(
                axes[0, 1],
                history,
                mode,
                lambda report: float(report["final"]["height_m"]),
                label=label,
            )
            _plot_mean_band(
                axes[1, 0],
                history,
                mode,
                lambda report: report["stable_last_100_steps"] / 100.0,
                label=label,
            )
            _plot_mean_band(
                axes[1, 1],
                history,
                mode,
                lambda report: float(bool(report["passed"])),
                label=label,
            )
    axes[0, 0].set(title="Final upright", ylabel="-gravity z")
    axes[0, 1].set(title="Final trunk height", ylabel="Meters")
    axes[1, 0].set(
        title="Stable fraction of final 100 steps",
        ylabel="Fraction meeting every stability threshold",
    )
    axes[1, 1].set(
        title="Physical acceptance across audit seeds",
        ylabel="Passed fraction",
        ylim=(-0.03, 1.03),
    )
    for axis in axes.flat:
        axis.set_xlabel("PPO environment transitions")
        axis.grid(alpha=0.2)
        if history:
            axis.legend(fontsize=8)
    figure.suptitle(
        "Backflip checkpoint learning curves; acceptance remains seed-wise and physical"
    )
    figure.savefig(output / "learning-curves.png", dpi=300)
    plt.close(figure)


def _write_report(output: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Backflip audit",
        "",
        f"- Verdict: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Run: `{result['run']}`",
        f"- Seeds: `{','.join(str(seed) for seed in result['seeds'])}`",
        f"- Initialization: `{result['initialization_evidence']['method']}`",
        "- Scope: deterministic nominal local simulation at 50 Hz.",
        "- Showcase credit: assisted launch, learned landing window, and explicit "
        "pretrained stand handoff.",
        "- Landing credit: assisted-release initialization followed by 600 policy "
        "steps with no handoff; all final 100 steps must meet every stability threshold.",
        "- Curves are optimization diagnostics only and are not used as physical "
        "success evidence.",
        f"- All-seed physical success: "
        f"`{result['gates']['all_seed_physical_success']}`",
        f"- PPO refinement: `{result['gates']['ppo_refinement']}` "
        "(paired unassisted-landing raw returns, not curve shape).",
        "",
        "## Policy comparison",
        "",
        "| Policy | Mode | Passed seeds | Raw return mean | Final upright | "
        "Final height (m) | Final feet | Final gyro | Stable last 100 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy_name in POLICY_NAMES:
        for mode in ("showcase", "landing"):
            summary = result["summary"][policy_name][mode]
            lines.append(
                f"| {policy_name} | {mode} | {summary['passed_seeds']}/"
                f"{len(summary['seeds'])} | {summary['raw_return_mean']:.6g} | "
                f"{summary['final_upright_mean']:.4f} | "
                f"{summary['final_height_m_mean']:.4f} | "
                f"{summary['final_both_feet_fraction']:.3f} | "
                f"{summary['final_gyro_norm_mean']:.4f} | "
                f"{summary['stable_last_100_fraction_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Paired PPO refinement",
            "",
            "| Seed | Initialization raw return | Final raw return | Delta | Relative gain |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {entry['seed']} | {entry['initialization_raw_return']:.6g} | "
                f"{entry['final_raw_return']:.6g} | {entry['delta']:.6g} | "
                + (
                    f"{entry['relative_gain']:.2%} |"
                    if entry["relative_gain"] is not None
                    else "n/a |"
                )
                for entry in result["ppo_refinement"]["per_seed"]
            ],
            (
                f"| **Aggregate** | "
                f"**{result['ppo_refinement']['aggregate']['initialization_raw_return_mean']:.6g}** | "
                f"**{result['ppo_refinement']['aggregate']['final_raw_return_mean']:.6g}** | "
                f"**{result['ppo_refinement']['aggregate']['delta_mean']:.6g}** | "
                + (
                    f"**{result['ppo_refinement']['aggregate']['relative_gain']:.2%}** |"
                    if result["ppo_refinement"]["aggregate"]["relative_gain"]
                    is not None
                    else "**n/a** |"
                )
            ),
            "",
            f"- Seeds improved: "
            f"`{result['ppo_refinement']['aggregate']['seeds_improved']}/"
            f"{result['ppo_refinement']['aggregate']['seed_count']}`",
            f"- Refinement criterion: `{result['ppo_refinement']['criterion']}`",
            "",
            "### Saved checkpoint refinement",
            "",
            "| Checkpoint steps | Paired seeds | Mean delta | Relative gain | Seeds improved |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {entry['steps']} | "
                f"{entry['initialization_refinement']['aggregate']['seed_count']} | "
                f"{entry['initialization_refinement']['aggregate']['delta_mean']:.6g} | "
                + (
                    f"{entry['initialization_refinement']['aggregate']['relative_gain']:.2%} | "
                    if entry["initialization_refinement"]["aggregate"][
                        "relative_gain"
                    ]
                    is not None
                    else "n/a | "
                )
                + f"{entry['initialization_refinement']['aggregate']['seeds_improved']} |"
                for entry in result["history"]
            ],
            "",
            "## Verification gates",
            "",
            f"- Final showcase all seeds: `{result['gates']['final_showcase_all_seeds']}`",
            f"- Final unassisted landing all seeds: "
            f"`{result['gates']['final_landing_all_seeds']}`",
            f"- ONNX parity: `{result['gates']['onnx_parity']}` "
            f"(final max absolute error "
            f"`{result['onnx']['final_parity_max_abs_error']:.6g}`, initialization "
            f"`{result['onnx']['initialization_parity_max_abs_error']:.6g}`)",
            f"- ONNX 61-to-14 contract: `{result['gates']['onnx_contract']}`",
            f"- Training evidence complete and finite: "
            f"`{result['gates']['training_evidence']}`",
            f"- Protocol provenance matches final checkpoint: "
            f"`{result['gates']['protocol_provenance']}`",
            f"- Final initialization hashes match audited initialization artifacts: "
            f"`{result['gates']['initialization_provenance']}`",
            f"- Requested checkpoint history present or explicitly skipped: "
            f"`{result['gates']['history_evidence']}`",
            "",
            "See `audit.json` for per-seed assessor reports, `provenance.json` "
            "for SHA256 inputs, and the NPZ files for retained per-step evidence.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def _missing_result(args: argparse.Namespace, missing: list[Path]) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": False,
        "status": "missing_required_artifacts",
        "run": str(args.run),
        "output": str(args.output),
        "missing": [str(path) for path in missing],
        "message": "The audit did not run because required artifacts are missing.",
    }
    _write_json(args.output / "audit.json", result)
    (args.output / "report.md").write_text(
        "# Backflip audit\n\n"
        "**FAIL:** required final artifacts are missing.\n\n"
        + "\n".join(f"- `{path}`" for path in missing)
        + "\n"
    )
    print(json.dumps(result, allow_nan=False))
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    args.run = args.run.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    final_checkpoint = args.run / "backflip.safetensors"
    final_sidecar = _sidecar(final_checkpoint)
    final_onnx = args.run / "backflip.onnx"
    bc_checkpoint = args.run / "initialization" / "backflip.safetensors"
    bc_sidecar = _sidecar(bc_checkpoint)
    bc_onnx = args.run / "initialization" / "backflip.onnx"
    imitation_history_path = (
        args.run / "initialization" / "imitation-history.json"
    )
    transfer_parity_path = args.run / "initialization" / "transfer-parity.json"
    training_metrics_path = args.run / "training-metrics.jsonl"
    required = (
        final_checkpoint,
        final_sidecar,
        final_onnx,
        bc_checkpoint,
        bc_sidecar,
        bc_onnx,
        training_metrics_path,
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        return _missing_result(args, missing)
    _write_json(
        args.output / "audit.json",
        {
            "passed": False,
            "status": "audit_in_progress",
            "run": str(args.run),
            "output": str(args.output),
        },
    )

    from rlx.environments.backflip import protocol_metadata, stand_action

    studio = studio_module()
    final_metadata = _metadata(final_checkpoint)
    bc_metadata = _metadata(bc_checkpoint)
    initialization_method = str(
        bc_metadata.get("initialization_method")
        or bc_metadata.get("bootstrap")
        or "unspecified"
    )
    method_lower = initialization_method.lower()
    uses_transfer = (
        transfer_parity_path.is_file()
        or (
            "behavior clon" not in method_lower
            and any(
                marker in method_lower
                for marker in ("transfer", "exact", "onnx", "normalizer")
            )
        )
    )
    if uses_transfer and not transfer_parity_path.is_file():
        return _missing_result(args, [transfer_parity_path])
    reward_weights = {
        key: float(value)
        for key, value in final_metadata.get("reward_weights", {}).items()
    }
    if not reward_weights:
        reward_weights = {
            key: float(value)
            for key, value in protocol_metadata()["landing_reward_weights"].items()
        }

    policies = {
        "final": _onnx_policy(studio, final_onnx),
        "initialization": _onnx_policy(studio, bc_onnx),
        "zero": lambda observation: np.zeros(14, dtype=np.float32),
        "stand": stand_action,
    }
    evaluations: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"showcase": [], "landing": []} for name in POLICY_NAMES
    }
    trace_files: list[str] = []
    for policy_name, infer in policies.items():
        for mode in ("showcase", "landing"):
            for seed in args.seeds:
                report, arrays = rollout(
                    infer,
                    mode=mode,
                    seed=seed,
                    reward_weights=reward_weights,
                )
                evaluations[policy_name][mode].append(report)
                trace_files.append(
                    _save_trace(
                        args.output,
                        policy_name=policy_name,
                        mode=mode,
                        seed=seed,
                        arrays=arrays,
                    )
                )

    history: list[dict[str, Any]] = []
    history_selection: list[dict[str, Any]] = []
    history_seeds = () if args.skip_history else args.seeds[: min(3, len(args.seeds))]
    checkpoints = sorted((args.run / "checkpoints").glob("*.safetensors"))
    if not args.skip_history:
        history_selection = _select_history(
            checkpoints,
            final_steps=int(final_metadata["steps"]),
        )
        for selected in history_selection:
            checkpoint = selected["path"]
            infer = _checkpoint_policy(studio, checkpoint)
            selected_seeds = (
                args.seeds if selected["steps"] == 401408 else history_seeds
            )
            entry = {
                key: value for key, value in selected.items() if key != "path"
            }
            entry["checkpoint"] = str(checkpoint)
            entry["evaluations"] = {"showcase": [], "landing": []}
            for mode in ("showcase", "landing"):
                for seed in selected_seeds:
                    report, arrays = rollout(
                        infer,
                        mode=mode,
                        seed=seed,
                        reward_weights=reward_weights,
                    )
                    entry["evaluations"][mode].append(report)
                    trace_files.append(
                        _save_trace(
                            args.output,
                            policy_name=selected["label"],
                            mode=mode,
                            seed=seed,
                            arrays=arrays,
                            prefix="history-",
                        )
                    )
            entry["initialization_refinement"] = _paired_landing_refinement(
                evaluations["initialization"]["landing"],
                entry["evaluations"]["landing"],
                candidate=f"checkpoint-{selected['steps']}",
            )
            history.append(entry)

    with np.load(
        args.output / f"final-showcase-seed{args.seeds[0]}.npz"
    ) as retained:
        first_observations = retained["observations"].astype(np.float32)
    native_actions = np.asarray(
        studio._policy_from_checkpoint(final_checkpoint)(first_observations),
        dtype=np.float32,
    )
    onnx_actions = np.asarray(
        studio._policy_from_onnx(final_onnx)(first_observations),
        dtype=np.float32,
    )
    final_parity_error = float(np.max(np.abs(native_actions - onnx_actions)))
    bc_native_actions = np.asarray(
        studio._policy_from_checkpoint(bc_checkpoint)(first_observations),
        dtype=np.float32,
    )
    bc_onnx_actions = np.asarray(
        studio._policy_from_onnx(bc_onnx)(first_observations),
        dtype=np.float32,
    )
    bc_parity_error = float(np.max(np.abs(bc_native_actions - bc_onnx_actions)))

    import onnx

    onnx_model = onnx.load(str(final_onnx))
    onnx.checker.check_model(onnx_model)
    input_dims = [
        dimension.dim_value or dimension.dim_param
        for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_dims = [
        dimension.dim_value or dimension.dim_param
        for dimension in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    bc_onnx_model = onnx.load(str(bc_onnx))
    onnx.checker.check_model(bc_onnx_model)
    bc_input_dims = [
        dimension.dim_value or dimension.dim_param
        for dimension in bc_onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    bc_output_dims = [
        dimension.dim_value or dimension.dim_param
        for dimension in bc_onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    contract_passed = bool(
        input_dims == ["batch", 61]
        and output_dims == ["batch", 14]
        and bc_input_dims == ["batch", 61]
        and bc_output_dims == ["batch", 14]
    )

    events, training_metrics_sha256 = _read_jsonl(training_metrics_path)
    requested_steps = int(final_metadata.get("ppo", {}).get("total_timesteps", 0))
    training = _training_evidence(
        events,
        requested_steps=requested_steps,
        final_checkpoint_steps=int(final_metadata.get("steps", 0)),
    )
    imitation_history = (
        _read_json(imitation_history_path)
        if not uses_transfer and imitation_history_path.is_file()
        else []
    )
    transfer_parity = (
        _read_json(transfer_parity_path)
        if transfer_parity_path.is_file()
        else None
    )
    current_protocol = protocol_metadata()
    recorded_protocol = final_metadata.get("recipe_options", {})
    protocol_keys = (
        "backflip_protocol_version",
        "stand_policy_sha256",
        "launch_release_rotation_rad",
        "minimum_landing_policy_steps",
        "training_mode",
        "handoff",
        "landing_reward_version",
        "landing_reward_weights",
    )
    final_protocol_matches = all(
        recorded_protocol.get(key) == current_protocol.get(key)
        for key in protocol_keys
    )
    bc_protocol_matches = all(
        bc_metadata.get(key) == current_protocol.get(key) for key in protocol_keys
    )
    protocol_matches = bool(final_protocol_matches and bc_protocol_matches)
    initialization = final_metadata.get("initialization", {})
    initialization_matches = bool(
        initialization.get("kind") == "checkpoint"
        and initialization.get("source_sha256") == _sha256(bc_checkpoint)
        and initialization.get("source_sidecar_sha256") == _sha256(bc_sidecar)
    )

    provenance_files = [
        final_checkpoint,
        final_sidecar,
        final_onnx,
        bc_checkpoint,
        bc_sidecar,
        bc_onnx,
        training_metrics_path,
        Path(current_protocol["stand_policy"]),
        Path(__file__).resolve().parents[1] / "rlx/environments/backflip.py",
        Path(__file__).resolve().parents[1]
        / "rlx/environments/backflip_evaluation.py",
        Path(__file__).resolve().parents[1] / "examples/ppo_microduck_studio.py",
        Path(__file__).resolve().parent / "audit_dance.py",
        *[
            path
            for selected in history_selection
            for path in (selected["path"], _sidecar(selected["path"]))
        ],
    ]
    provenance_files.extend(
        path
        for path in (
            transfer_parity_path if uses_transfer else imitation_history_path,
        )
        if path.is_file()
    )
    provenance = {
        "run": str(args.run),
        "audit_script": str(Path(__file__).resolve()),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "seeds": list(args.seeds),
        "history_seeds": list(history_seeds),
        "history_skipped": bool(args.skip_history),
        "environment": {
            "control_hz": CONTROL_HZ,
            "steps": ROLLOUT_STEPS,
            "actuator": "xml",
            "domain_rand": False,
            "obs_noise": False,
            "action_delay": False,
            "random_yaw": False,
            "reward_weights": reward_weights,
        },
        "files": {
            str(path): _sha256(path)
            for path in dict.fromkeys(provenance_files)
        },
        "retained_evidence": {
            name: _sha256(args.output / name) for name in sorted(trace_files)
        },
    }
    provenance["files"][str(training_metrics_path)] = training_metrics_sha256
    _write_json(args.output / "provenance.json", provenance)

    summary = {
        name: _summarize_policy(evaluations[name]) for name in POLICY_NAMES
    }
    refinement = _landing_refinement(evaluations)
    all_seed_physical_success = bool(
        summary["final"]["showcase"]["all_seeds_passed"]
        and summary["final"]["landing"]["all_seeds_passed"]
    )
    gates = {
        "final_showcase_all_seeds": summary["final"]["showcase"][
            "all_seeds_passed"
        ],
        "final_landing_all_seeds": summary["final"]["landing"]["all_seeds_passed"],
        "all_seed_physical_success": all_seed_physical_success,
        "ppo_refinement": refinement["passed"],
        "onnx_parity": final_parity_error < 1e-4 and bc_parity_error < 1e-4,
        "onnx_contract": contract_passed,
        "training_evidence": training["passed"],
        "protocol_provenance": protocol_matches,
        "initialization_provenance": initialization_matches,
        "history_evidence": bool(args.skip_history or len(history_selection) == 4),
    }
    result = {
        "passed": all(gates.values()),
        "status": "complete",
        "run": str(args.run),
        "output": str(args.output),
        "seeds": list(args.seeds),
        "history_seeds": list(history_seeds),
        "scope": {
            "hardware_safe": False,
            "showcase": (
                "assisted launch plus learned landing window plus pretrained "
                "stand handoff; not an unassisted learned backflip"
            ),
            "landing": (
                "assisted-release initialization, then learned policy only for "
                "600 steps with no handoff"
            ),
            "curve_warning": (
                "reward and loss curves are optimization diagnostics and cannot "
                "establish physical skill"
            ),
        },
        "gates": gates,
        "summary": summary,
        "ppo_refinement": refinement,
        "evaluations": evaluations,
        "history": history,
        "history_selection": [
            {key: value for key, value in selected.items() if key != "path"}
            | {"checkpoint": str(selected["path"])}
            for selected in history_selection
        ],
        "training": training,
        "initialization_evidence": {
            "method": initialization_method,
            "imitation_history": {
                "present": bool(imitation_history),
                "epochs": len(imitation_history),
                "first": imitation_history[0] if imitation_history else None,
                "last": imitation_history[-1] if imitation_history else None,
            },
            "transfer_parity": transfer_parity,
        },
        "onnx": {
            "final_parity_max_abs_error": final_parity_error,
            "initialization_parity_max_abs_error": bc_parity_error,
            "parity_threshold": 1e-4,
            "final_input_shape": input_dims,
            "final_output_shape": output_dims,
            "initialization_input_shape": bc_input_dims,
            "initialization_output_shape": bc_output_dims,
            "contract_passed": contract_passed,
        },
        "protocol": {
            "current": current_protocol,
            "recorded_final": recorded_protocol,
            "recorded_initialization": {
                key: bc_metadata.get(key) for key in protocol_keys
            },
            "matches": protocol_matches,
        },
        "initialization_provenance": {
            "recorded": initialization,
            "audited_initialization_checkpoint_sha256": _sha256(bc_checkpoint),
            "audited_initialization_sidecar_sha256": _sha256(bc_sidecar),
            "matches": initialization_matches,
        },
        "initialization": bc_metadata,
        "trace_files": sorted(trace_files),
        "provenance": "provenance.json",
        "report": "report.md",
        "plots": ["reward-learning.png", "ppo-losses.png", "learning-curves.png"],
    }
    plots(
        args.output,
        events=events,
        history=history,
        imitation_history=imitation_history,
        transfer_parity=transfer_parity,
    )
    _write_report(args.output, result)
    _write_json(args.output / "audit.json", result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(args.output),
                "gates": gates,
            },
            allow_nan=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        metavar="301,302,303",
        help=(
            "comma-separated evaluation seeds "
            "(default: 301,302,303,304,305,306,307,308)"
        ),
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="skip saved-checkpoint rollouts; final and control audits still run",
    )
    args = parser.parse_args()
    raise SystemExit(0 if audit(args)["passed"] else 2)


if __name__ == "__main__":
    main()
