"""Continue the released recurrent basketball actor with local CPU PPO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from rlx.models.basketball import (
    LOCAL_CHECKPOINT_FORMAT,
    OBSERVATION_DIM,
    BasketballActor,
    BasketballCritic,
    compare_actor_to_onnx,
    export_recurrent_onnx,
    load_source_actor,
    sha256_file,
)

CTRL_DT = 0.02
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "microduck-playground"
    / "artifacts"
    / "basketball"
    / "checkpoint.pt"
)


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    bootstrap_values: torch.Tensor
    episode_starts: torch.Tensor
    initial_hidden: torch.Tensor
    initial_cell: torch.Tensor
    next_hidden: torch.Tensor
    next_cell: torch.Tensor
    next_episode_starts: torch.Tensor
    infos: list[list[dict[str, Any]]]


def gaussian_log_prob(
    actions: torch.Tensor,
    means: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    variance = std.square()
    return (
        -0.5
        * (
            (actions - means).square() / variance
            + 2.0 * torch.log(std)
            + math.log(2.0 * math.pi)
        )
    ).sum(dim=-1)


def gaussian_entropy(std: torch.Tensor) -> torch.Tensor:
    return (torch.log(std) + 0.5 * math.log(2.0 * math.pi * math.e)).sum()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (
        rewards.shape
        == values.shape
        == bootstrap_values.shape
        == terminated.shape
        == truncated.shape
    ):
        raise ValueError("GAE tensors must share [time, env] shape")
    advantages = torch.zeros_like(rewards)
    trace = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_mask = (~terminated[step]).to(rewards.dtype)
        continuation_mask = (~(terminated[step] | truncated[step])).to(
            rewards.dtype
        )
        delta = (
            rewards[step]
            + gamma * bootstrap_values[step] * bootstrap_mask
            - values[step]
        )
        trace = delta + gamma * gae_lambda * continuation_mask * trace
        advantages[step] = trace
    return advantages, advantages + values


def _validate_observations(observations: Any, num_envs: int) -> np.ndarray:
    array = np.asarray(observations, dtype=np.float32)
    expected = (num_envs, OBSERVATION_DIM)
    if array.shape != expected:
        raise ValueError(f"observations must have shape {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise RuntimeError("basketball observations contain non-finite values")
    if np.count_nonzero(array[:, 51:61]):
        raise RuntimeError("basketball actor head/body padding must remain zero")
    return array


def _validate_step(
    observation: Any,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: Any,
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape != (OBSERVATION_DIM,) or not np.isfinite(obs).all():
        raise RuntimeError("basketball step returned an invalid observation")
    if np.count_nonzero(obs[51:61]):
        raise RuntimeError("basketball actor received nonzero head/body padding")
    value = float(reward)
    if not math.isfinite(value):
        raise RuntimeError("basketball step returned a non-finite reward")
    if not isinstance(info, dict):
        raise TypeError("basketball info must be a dictionary")
    return obs, value, bool(terminated), bool(truncated), info


def make_environments(args: argparse.Namespace) -> list[Any]:
    from rlx.environments.basketball import BasketballEnv, basketball_model

    shared_model = (
        basketball_model(actuator=args.actuator) if args.actuator == "xml" else None
    )
    command = None if args.command is None else tuple(float(x) for x in args.command)
    environments = []
    for index in range(args.num_envs):
        kwargs = {
            "max_episode_s": 10.0,
            "hold": args.hold,
            "curriculum": args.curriculum,
            "command": command,
            "seed": args.seed + index,
            "actuator": args.actuator,
            "obs_noise": False,
            "domain_rand": False,
            "pushes": args.pushes,
        }
        if shared_model is not None:
            kwargs["model"] = shared_model
        environments.append(
            BasketballEnv(**kwargs)
        )
    return environments


def reset_environments(
    environments: Sequence[Any],
    *,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    observations: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for index, environment in enumerate(environments):
        observation, info = environment.reset(seed=seed + index)
        observations.append(np.asarray(observation, dtype=np.float32))
        if not isinstance(info, dict):
            raise TypeError("basketball reset info must be a dictionary")
        infos.append(info)
    return _validate_observations(observations, len(environments)), infos


def collect_rollout(
    actor: BasketballActor,
    critic: BasketballCritic,
    environments: Sequence[Any],
    observations: np.ndarray,
    hidden: torch.Tensor,
    cell: torch.Tensor,
    episode_starts: torch.Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    seed: int,
    reset_counts: np.ndarray,
) -> tuple[Rollout, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    if steps < 1:
        raise ValueError("steps must be positive")
    num_envs = len(environments)
    initial_hidden = hidden.detach().clone()
    initial_cell = cell.detach().clone()
    stored: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "observations",
            "actions",
            "old_log_probs",
            "old_values",
            "rewards",
            "terminated",
            "truncated",
            "bootstrap_values",
            "episode_starts",
        )
    }
    all_infos: list[list[dict[str, Any]]] = []
    actor.eval()
    critic.eval()
    for _ in range(steps):
        observation_tensor = torch.from_numpy(observations)
        with torch.inference_mode():
            carry = (~episode_starts.bool()).to(observation_tensor.dtype).view(
                1, num_envs, 1
            )
            hidden = hidden * carry
            cell = cell * carry
            means, next_hidden, next_cell = actor(
                observation_tensor,
                hidden,
                cell,
            )
            std = actor.action_std()
            actions = means + torch.randn(
                means.shape,
                generator=generator,
                dtype=means.dtype,
            ) * std
            log_probs = gaussian_log_prob(actions, means, std)
            values = critic(observation_tensor)
        next_observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminations: list[bool] = []
        truncations: list[bool] = []
        step_infos: list[dict[str, Any]] = []
        final_observations: list[np.ndarray] = []
        for index, environment in enumerate(environments):
            result = _validate_step(*environment.step(actions[index].numpy()))
            next_observation, reward, terminated, truncated, info = result
            final_observations.append(next_observation)
            rewards.append(reward)
            terminations.append(terminated)
            truncations.append(truncated)
            step_infos.append(info)
            if terminated or truncated:
                reset_counts[index] += 1
                reset_observation, reset_info = environment.reset(
                    seed=seed + index + int(reset_counts[index]) * num_envs
                )
                reset_observation = np.asarray(reset_observation, dtype=np.float32)
                if not isinstance(reset_info, dict):
                    raise TypeError("basketball reset info must be a dictionary")
                info = {**info, "reset_info": reset_info}
                step_infos[-1] = info
                next_observations.append(reset_observation)
            else:
                next_observations.append(next_observation)
        next_batch = _validate_observations(next_observations, num_envs)
        final_batch = _validate_observations(final_observations, num_envs)
        terminated_tensor = torch.tensor(terminations, dtype=torch.bool)
        truncated_tensor = torch.tensor(truncations, dtype=torch.bool)
        with torch.inference_mode():
            bootstrap_values = critic(torch.from_numpy(final_batch))
            bootstrap_values = torch.where(
                terminated_tensor,
                torch.zeros_like(bootstrap_values),
                bootstrap_values,
            )
        stored["observations"].append(observation_tensor)
        stored["actions"].append(actions)
        stored["old_log_probs"].append(log_probs)
        stored["old_values"].append(values)
        stored["rewards"].append(torch.tensor(rewards, dtype=torch.float32))
        stored["terminated"].append(terminated_tensor)
        stored["truncated"].append(truncated_tensor)
        stored["bootstrap_values"].append(bootstrap_values)
        stored["episode_starts"].append(episode_starts.clone())
        all_infos.append(step_infos)
        observations = next_batch
        hidden = next_hidden
        cell = next_cell
        episode_starts = terminated_tensor | truncated_tensor
    rollout = Rollout(
        **{name: torch.stack(values) for name, values in stored.items()},
        initial_hidden=initial_hidden,
        initial_cell=initial_cell,
        next_hidden=hidden.detach(),
        next_cell=cell.detach(),
        next_episode_starts=episode_starts,
        infos=all_infos,
    )
    return rollout, observations, hidden.detach(), cell.detach(), episode_starts


def ppo_update(
    actor: BasketballActor,
    critic: BasketballCritic,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    epochs: int = 5,
    clip_coefficient: float = 0.2,
    value_coefficient: float = 1.0,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 1.0,
    target_kl: float = 0.02,
) -> dict[str, float]:
    advantages, returns = compute_gae(
        rollout.rewards,
        rollout.old_values,
        rollout.bootstrap_values,
        rollout.terminated,
        rollout.truncated,
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    actor.train()
    critic.train()
    metrics: dict[str, float] = {}
    for epoch in range(epochs):
        means, _, _ = actor.forward_sequence(
            rollout.observations,
            rollout.initial_hidden,
            rollout.initial_cell,
            rollout.episode_starts,
        )
        std = actor.action_std()
        log_probs = gaussian_log_prob(rollout.actions, means, std)
        log_ratio = log_probs - rollout.old_log_probs
        ratio = torch.exp(log_ratio)
        approximate_kl = ((ratio - 1) - log_ratio).mean()
        if not torch.isfinite(approximate_kl):
            raise RuntimeError("PPO KL became non-finite")
        if float(approximate_kl.detach()) > target_kl:
            metrics["approx_kl"] = float(approximate_kl.detach())
            metrics["epochs_completed"] = epoch
            metrics["kl_early_stop"] = 1.0
            break
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - clip_coefficient,
            1.0 + clip_coefficient,
        ) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        values = critic(rollout.observations.reshape(-1, OBSERVATION_DIM)).reshape(
            rollout.rewards.shape
        )
        value_loss = 0.5 * (values - returns).square().mean()
        entropy = gaussian_entropy(std)
        loss = (
            policy_loss
            + value_coefficient * value_loss
            - entropy_coefficient * entropy
        )
        if not torch.isfinite(loss):
            raise RuntimeError("PPO loss became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            list(actor.parameters()) + list(critic.parameters()),
            max_grad_norm,
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError("PPO gradient norm became non-finite")
        optimizer.step()
        if not all(
            torch.isfinite(parameter).all()
            for parameter in list(actor.parameters()) + list(critic.parameters())
        ):
            raise RuntimeError("PPO parameters became non-finite")
        with torch.no_grad():
            updated_means, _, _ = actor.forward_sequence(
                rollout.observations, rollout.initial_hidden, rollout.initial_cell,
                rollout.episode_starts,
            )
            updated_log_probs = gaussian_log_prob(rollout.actions, updated_means, actor.action_std())
            updated_log_ratio = updated_log_probs - rollout.old_log_probs
            final_kl = (torch.exp(updated_log_ratio) - 1 - updated_log_ratio).mean()
        if not torch.isfinite(final_kl):
            raise RuntimeError("post-update PPO KL became non-finite")
        metrics = {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy.detach()),
            "gradient_norm": float(gradient_norm),
            "approx_kl": float(final_kl),
            "epochs_completed": epoch + 1,
            "kl_early_stop": 0.0,
        }
        if float(final_kl) > target_kl:
            metrics["kl_early_stop"] = 1.0
            break
    return metrics


def actor_weight_change(
    actor: BasketballActor,
    initial_state: dict[str, torch.Tensor],
) -> float:
    total = torch.tensor(0.0)
    current = actor.state_dict()
    for name, source in initial_state.items():
        if name.startswith("obs_normalizer.") or name == "distribution.std_param":
            continue
        total += (current[name].cpu() - source.cpu()).square().sum()
    return float(torch.sqrt(total))


def save_results(
    output: Path,
    actor: BasketballActor,
    critic: BasketballCritic,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    source_metadata: dict[str, Any],
    initial_actor_state: dict[str, torch.Tensor],
    local_steps: int,
    update_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    from rlx.environments import basketball as basketball_environment

    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.pt"
    onnx_path = output / "policy.onnx"
    checkpoint = {
        "format": LOCAL_CHECKPOINT_FORMAT,
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "local_steps": local_steps,
        "total_steps": source_metadata.get("source_training_steps", 0) + local_steps,
        "local_updates": int(update_metrics[-1]["update"]) if update_metrics else 0,
        "source": source_metadata,
        "normalization_frozen": True,
        "actor_observation_dim": OBSERVATION_DIM,
        "critic_observation_dim": OBSERVATION_DIM,
    }
    torch.save(checkpoint, checkpoint_path)
    export_recurrent_onnx(actor, onnx_path)
    parity = compare_actor_to_onnx(actor, onnx_path)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "continuation_kind": "actor warm-start with fresh local critic and optimizer",
        "full_upstream_resume": False,
        "source_kind": source_metadata.get("source_kind", "released_actor"),
        "source_training_steps": source_metadata.get("source_training_steps", 0),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": source_metadata["source_hashes"][
            "checkpoint_sha256"
        ],
        "source_onnx_sha256": source_metadata["source_hashes"].get("onnx_sha256"),
        "source_iteration": source_metadata["source_iteration"],
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_parity": parity,
        "local_steps": local_steps,
        "total_steps": source_metadata.get("source_training_steps", 0) + local_steps,
        "local_updates": int(update_metrics[-1]["update"]) if update_metrics else 0,
        "actor_weight_change_l2": actor_weight_change(actor, initial_actor_state),
        "normalization_frozen": True,
        "actor_observation_dim": OBSERVATION_DIM,
        "ball_state_actor": False,
        "environment": {
            "exact_upstream_recipe": False,
            "reward_weights": dict(basketball_environment.REWARD_WEIGHTS),
            "reward_scale": CTRL_DT,
            "omitted_upstream_terms": ["pose", "angular_momentum", "joint_limits", "self_collision"],
            "hold_levels": list(basketball_environment.HOLD_LEVELS),
            "source_sha256": sha256_file(basketball_environment.__file__),
        },
        "settings": {
            "num_envs": args.num_envs,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "initial_std": args.initial_std,
            "hold": args.hold,
            "curriculum": args.curriculum,
            "actuator": args.actuator,
            "seed": args.seed,
            "command": args.command,
            "pushes": args.pushes,
            "control_dt": CTRL_DT,
            "target_kl": args.target_kl,
            "obs_noise": False,
            "domain_rand": False,
            "randomized_commands": args.command is None,
        },
        "updates": update_metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return summary


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise FileExistsError(f"refusing to overwrite an existing run: {args.output}")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    actor, source_metadata = load_source_actor(
        args.checkpoint,
        initial_std=args.initial_std,
        verify_hash=True,
        allow_local=True,
    )
    initial_actor_state = {
        name: value.detach().clone() for name, value in actor.state_dict().items()
    }
    critic = BasketballCritic(actor.obs_normalizer)
    parameters = list(actor.parameters()) + [
        parameter
        for name, parameter in critic.named_parameters()
        if not name.startswith("normalizer.")
    ]
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    environments = make_environments(args)
    generator = torch.Generator().manual_seed(args.seed)
    reset_counts = np.zeros(args.num_envs, dtype=np.int64)
    update_metrics: list[dict[str, float]] = []
    local_steps = 0
    try:
        observations, _ = reset_environments(environments, seed=args.seed)
        hidden, cell = actor.initial_state(args.num_envs)
        episode_starts = torch.ones(args.num_envs, dtype=torch.bool)
        for update in range(args.updates):
            rollout, observations, hidden, cell, episode_starts = collect_rollout(
                actor,
                critic,
                environments,
                observations,
                hidden,
                cell,
                episode_starts,
                steps=args.steps,
                generator=generator,
                seed=args.seed,
                reset_counts=reset_counts,
            )
            metrics = ppo_update(actor, critic, optimizer, rollout, target_kl=args.target_kl)
            metrics["hold_min"] = min(environment.hold for environment in environments)
            metrics["hold_max"] = max(environment.hold for environment in environments)
            metrics["completed_episodes"] = int(reset_counts.sum())
            local_steps += args.num_envs * args.steps
            update_metrics.append({"update": update + 1, **metrics})
            print(
                json.dumps(
                    {
                        "event": "basketball_ppo_update",
                        "update": update + 1,
                        "updates": args.updates,
                        "local_steps": local_steps,
                        **metrics,
                    },
                    allow_nan=False,
                ),
                flush=True,
            )
            if (update + 1) % args.save_interval == 0 and update + 1 < args.updates:
                save_results(
                    args.output / f"update-{update + 1:04d}", actor, critic, optimizer,
                    args=args, source_metadata=source_metadata,
                    initial_actor_state=initial_actor_state,
                    local_steps=local_steps, update_metrics=update_metrics,
                )
    finally:
        for environment in environments:
            environment.close()
    return save_results(
        args.output,
        actor,
        critic,
        optimizer,
        args=args,
        source_metadata=source_metadata,
        initial_actor_state=initial_actor_state,
        local_steps=local_steps,
        update_metrics=update_metrics,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/basketball/local-recurrent-ppo"),
    )
    parser.add_argument("--num-envs", type=_positive_int, default=4)
    parser.add_argument("--updates", type=_positive_int, default=5)
    parser.add_argument("--steps", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=2e-5)
    parser.add_argument("--target-kl", type=_positive_float, default=0.02)
    parser.add_argument("--save-interval", type=_positive_int, default=25)
    parser.add_argument("--initial-std", type=_positive_float, default=0.03)
    parser.add_argument("--hold", type=_nonnegative_float, default=0.0)
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--actuator", default="bam")
    parser.add_argument("--seed", type=int, default=42)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=(0.08, 0.0, 0.0),
    )
    commands.add_argument("--randomized-commands", action="store_true")
    parser.add_argument(
        "--pushes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args(argv)
    if args.randomized_commands:
        args.command = None
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.command is not None and not all(math.isfinite(x) for x in args.command):
        parser.error("--command values must be finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = train(args)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
