"""Executable high-school-level PPO arithmetic, not a robot trainer."""

import json
import math


def gaussian_log_probability(action, mean, standard_deviation):
    if standard_deviation <= 0:
        raise ValueError("standard deviation must be positive")
    return -0.5 * (
        ((action - mean) / standard_deviation) ** 2
        + 2 * math.log(standard_deviation)
        + math.log(2 * math.pi)
    )


def clipped_objective(ratio, advantage, epsilon=0.2):
    clipped_ratio = min(max(ratio, 1 - epsilon), 1 + epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)


def advantages_and_returns(rewards, values, next_values, terminated, truncated,
                           gamma=0.99, gae_lambda=0.95):
    lengths = {len(series) for series in (rewards, values, next_values, terminated, truncated)}
    if len(lengths) != 1:
        raise ValueError("all series must have equal length")
    advantages = [0.0] * len(rewards)
    trace = 0.0
    for index in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[index] else next_values[index]
        delta = rewards[index] + gamma * bootstrap - values[index]
        continuation = 0.0 if terminated[index] or truncated[index] else trace
        trace = delta + gamma * gae_lambda * continuation
        advantages[index] = trace
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


def main():
    advantages, returns = advantages_and_returns(
        [1.0, 1.0, 1.0], [0.5, 0.6, 0.7], [0.6, 0.7, 0.0],
        [False, False, True], [False, False, False],
    )
    assert math.isclose(advantages[-1], 0.3)
    assert math.isclose(advantages[1], 1.37515)
    assert math.isclose(advantages[0], 2.387328575)
    terminal, _ = advantages_and_returns([1], [0.5], [10], [True], [False])
    timeout, _ = advantages_and_returns([1], [0.5], [10], [False], [True])
    assert math.isclose(terminal[0], 0.5)
    assert math.isclose(timeout[0], 10.4)
    assert math.isclose(clipped_objective(1.5, 2), 2.4)
    assert math.isclose(clipped_objective(0.5, -2), -1.6)
    assert math.isclose(clipped_objective(1.5, -2), -3)
    result = {
        "scope": "synthetic arithmetic fixture, not measured training data",
        "advantages": advantages,
        "returns": returns,
        "terminal_advantage": terminal[0],
        "timeout_advantage": timeout[0],
        "normal_density_at_mean_std_003": math.exp(gaussian_log_probability(0, 0, 0.03)),
        "positive_advantage_clipped": clipped_objective(1.5, 2),
        "negative_advantage_clipped": clipped_objective(0.5, -2),
        "passed": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
