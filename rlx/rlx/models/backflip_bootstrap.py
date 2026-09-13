"""Distill the named stand teacher on real assisted-release landing states."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def initialize_landing(output: Path, *, seed: int = 7, initial_std: float = 0.03,
                       gamma: float = 0.99, calibration_episodes: int = 16,
                       episode_seconds: float = 12, critic_epochs: int = 10,
                       normalize_rewards: bool = True, weight_overrides: dict | None = None) -> Path:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import onnx
    from onnx import numpy_helper
    from rlx.environments.backflip import STAND_POLICY, BackflipEnv, protocol_metadata, stand_action
    from rlx.models.microduck import create_actor_critic, normalize_observations, save_checkpoint
    from rlx.export.microduck_onnx import export_deterministic_actor

    metadata = protocol_metadata()
    if not 0 < gamma < 1 or calibration_episodes < 1 or critic_epochs < 1:
        raise ValueError("Calibration requires 0 < gamma < 1 and positive episode/epoch counts")
    source = onnx.load(STAND_POLICY)
    tensors = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    if [node.op_type for node in source.graph.node] != ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]:
        raise ValueError("Stand teacher graph changed; exact transfer needs a new parity review")
    mean = tensors["obs_normalizer._mean"].reshape(61)
    standard_deviation = tensors["onnx::Div_24"].reshape(61)
    if not np.all(np.isfinite(standard_deviation)) or not np.all(standard_deviation > 0):
        raise ValueError("Invalid stand normalizer")
    variance = np.square(standard_deviation)
    mx.random.seed(seed)
    network = create_actor_critic(initial_std=initial_std)
    network.load_weights([(f"actor_mean.layers.{index}.{kind}", mx.array(tensors[f"mlp.{index}.{kind}"].copy()))
                          for index in (0, 2, 4, 6) for kind in ("weight", "bias")], strict=False)
    mx.eval(network.parameters())
    env = BackflipEnv(backflip_mode="landing", seed=seed, actuator="xml", max_episode_s=episode_seconds,
                      domain_rand=False, obs_noise=False, action_delay=False, random_yaw=False,
                      weight_overrides=weight_overrides)
    observations, expected, discounted_returns, reward_episodes = [], [], [], []
    try:
        for episode in range(calibration_episodes):
            observation, _ = env.reset(seed=seed + episode)
            discounted = 0.0
            rewards = []
            for _ in range(round(episode_seconds * 50)):
                action = stand_action(observation)
                observations.append(observation.copy())
                expected.append(action.copy())
                observation, reward, _, _, _ = env.step(action)
                discounted = gamma * discounted + reward
                discounted_returns.append(discounted)
                rewards.append(reward)
            reward_episodes.append(rewards)
    finally:
        env.close()
    inputs = normalize_observations(mx.array(np.asarray(observations)), mean, variance, epsilon=0., clip=1e6)
    error = float(np.max(np.abs(np.asarray(network.deterministic(inputs)) - np.asarray(expected))))
    if error > 1e-4 or protocol_metadata() != metadata:
        raise RuntimeError(f"Exact stand transfer failed parity: {error}")
    return_mean = float(np.mean(discounted_returns))
    return_variance = float(np.var(discounted_returns))
    reward_scale = math.sqrt(return_variance)
    if not math.isfinite(reward_scale) or reward_scale <= 0:
        raise RuntimeError("Teacher reward calibration has no finite positive scale")
    targets = []
    for rewards in reward_episodes:
        future = float(np.mean(rewards[-50:])) / (1 - gamma)
        episode_targets = []
        for reward in reversed(rewards):
            future = reward + gamma * future
            episode_targets.append(future / reward_scale if normalize_rewards else future)
        targets.extend(reversed(episode_targets))
    targets = mx.array(targets, dtype=mx.float32)
    optimizer = optim.Adam(learning_rate=3e-4)
    rng = np.random.default_rng(seed)

    def value_loss(critic, observation_batch, target_batch):
        return mx.mean(mx.square(critic(observation_batch).squeeze(-1) - target_batch))

    value_and_grad = nn.value_and_grad(network.critic, value_loss)
    calibration_history = []
    for epoch in range(critic_epochs):
        losses = []
        for indices in np.array_split(rng.permutation(len(observations)), math.ceil(len(observations) / 256)):
            selected = mx.array(indices)
            loss, gradients = value_and_grad(network.critic, inputs[selected], targets[selected])
            optimizer.update(network.critic, gradients)
            mx.eval(network.critic.parameters(), optimizer.state, loss)
            losses.append(float(np.asarray(loss)))
        record = {"phase": "critic_calibration", "epoch": epoch + 1, "mean_loss": float(np.mean(losses))}
        calibration_history.append(record)
        print(json.dumps(record), flush=True)
    error = float(np.max(np.abs(np.asarray(network.deterministic(inputs)) - np.asarray(expected))))
    if error > 1e-4 or protocol_metadata() != metadata:
        raise RuntimeError("Critic calibration changed the transferred actor")
    save_checkpoint(output, network, mean, variance, 1., epsilon=0., clip=1e6,
                    return_mean=np.array(return_mean), return_variance=np.array(return_variance), return_count=len(discounted_returns),
                    metadata={"recipe": "backflip", "steps": 0, "teacher_assisted": True,
                              "bootstrap": "exact alpha_stand actor and normalizer transfer; no BC optimizer",
                              "reward_calibration": {"samples": len(discounted_returns), "gamma": gamma,
                                                     "standard_deviation": reward_scale, "critic_epochs": critic_epochs,
                                                     "normalize_rewards": normalize_rewards, "reward_overrides": weight_overrides or {}},
                              "initialization_method": "exact_weight_transfer", "seed": seed, **metadata})
    export_deterministic_actor(output, output.with_suffix(".onnx"))
    (output.parent / "critic-calibration-history.json").write_text(json.dumps(calibration_history, indent=2) + "\n")
    (output.parent / "transfer-parity.json").write_text(json.dumps({
        "samples": len(observations), "max_absolute_action_error": error,
        "threshold": 1e-4, "passed": True, "stand_policy_sha256": metadata["stand_policy_sha256"],
    }, indent=2) + "\n")
    print(json.dumps({"phase": "teacher_transfer", "parity_max_abs_error": error}), flush=True)
    return output


def bootstrap_landing(output: Path, *, seed: int = 7, episodes: int = 24, epochs: int = 40) -> Path:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rlx.environments.backflip import BackflipEnv, protocol_metadata, stand_action, stand_session
    from rlx.models.microduck import create_actor_critic, save_checkpoint
    from rlx.export.microduck_onnx import export_deterministic_actor

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = protocol_metadata()
    env = BackflipEnv(backflip_mode="landing", seed=seed, actuator="xml",
                      max_episode_s=6, domain_rand=False, obs_noise=False,
                      action_delay=False, random_yaw=False)
    observations, targets = [], []
    rng = np.random.default_rng(seed)
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            for _ in range(300):
                action = stand_action(observation)
                observations.append(observation.copy())
                targets.append(action.copy())
                applied = action + rng.normal(0, 0.03, 14).astype(np.float32)
                observation, _, terminated, truncated, _ = env.step(applied)
                if terminated or truncated:
                    break
    finally:
        env.close()
        stand_session.cache_clear()
    observations = np.asarray(observations, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    mean, variance = observations.mean(axis=0), observations.var(axis=0)
    normalized = np.clip((observations - mean) / np.sqrt(variance + 1e-8), -10, 10)
    mx.random.seed(seed)
    network = create_actor_critic(initial_std=0.1)
    optimizer = optim.Adam(learning_rate=3e-4)

    def imitation_loss(model, inputs, expected):
        return mx.mean(mx.square(model.deterministic(inputs) - expected))

    loss_and_grad = nn.value_and_grad(network, imitation_loss)
    history = []
    for epoch in range(epochs):
        losses = []
        for start in range(0, len(observations), 256):
            indices = rng.integers(0, len(observations), min(256, len(observations) - start))
            loss, gradients = loss_and_grad(network, mx.array(normalized[indices]), mx.array(targets[indices]))
            optimizer.update(network, gradients)
            mx.eval(network.parameters(), optimizer.state, loss)
            losses.append(float(np.asarray(loss)))
        record = {"phase": "behavior_cloning", "epoch": epoch + 1, "mean_imitation_loss": float(np.mean(losses))}
        history.append(record)
        print(json.dumps(record), flush=True)
    if protocol_metadata()["stand_policy_sha256"] != metadata["stand_policy_sha256"]:
        raise RuntimeError("Stand teacher changed during initialization")
    save_checkpoint(output, network, mean, variance, float(len(observations)),
                    return_mean=np.array(0.), return_variance=np.array(1.), return_count=1e-4,
                    metadata={"recipe": "backflip", "steps": 0, "teacher_assisted": True,
                              "bootstrap": "alpha_stand behavior cloning on assisted-release states",
                              "teacher_samples": len(observations), "seed": seed, **metadata})
    export_deterministic_actor(output, output.with_suffix(".onnx"))
    (output.parent / "imitation-history.json").write_text(json.dumps(history, indent=2) + "\n")
    return output
