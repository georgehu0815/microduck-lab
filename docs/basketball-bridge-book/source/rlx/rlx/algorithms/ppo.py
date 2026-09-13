from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from rlx.environments.environment import Environment
from rlx.buffers.rollout_buffer import RolloutBuffer
from rlx.utils import compute_generalized_advantage_estimate, flatten


LOG_RATIO_MIN = -20.0
LOG_RATIO_MAX = 20.0
VALUE_LOSS_DELTA = 10.0


@dataclass
class PPOConfig:
    num_envs: int = 4096
    num_steps: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 16
    update_epochs: int = 2
    normalize_advantages: bool = True
    clip_coefficient: float = 0.2
    clip_value_loss: bool = True
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class PPO:
    config: PPOConfig
    env: Environment
    network: nn.Module
    optimizer: optim.Optimizer
    buffer: RolloutBuffer
    key: mx.array
    step: int = 0

    def __post_init__(self):
        self._constrain_actor_log_std()
        mx.eval(self.network.state)
        self._assert_finite("model parameters", self.network.parameters())
        state = [self.network.state, self.optimizer.state]
        self.update_step = mx.compile(self.update_step, inputs=state, outputs=state)
        self._update_step_with_metrics = mx.compile(
            self._update_step_with_metrics,
            inputs=state,
            outputs=state,
        )

    def _constrain_actor_log_std(self):
        constrain = getattr(self.network, "constrain_actor_log_std", None)
        if constrain is not None:
            constrain()

    @staticmethod
    def _all_finite(tree):
        checks = [mx.all(mx.isfinite(value)) for _, value in tree_flatten(tree)]
        return mx.all(mx.stack(checks))

    @classmethod
    def _assert_finite(cls, label, tree):
        check = cls._all_finite(tree)
        mx.eval(check)
        if not bool(np.asarray(check)):
            raise RuntimeError(f"PPO {label} contains non-finite values")

    @staticmethod
    def _raise_if_non_finite(label, check):
        if not bool(np.asarray(check)):
            raise RuntimeError(f"PPO {label} contains non-finite values")

    def _gradient_failure_details(
        self,
        observations,
        actions,
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    ) -> str:
        """Recompute one failed minibatch eagerly and name unstable tensors."""
        loss, grads = nn.value_and_grad(self.network, self.loss_fn)(
            self.network,
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )
        flat_grads = tree_flatten(grads)
        mx.eval(loss, *(value for _, value in flat_grads))
        bad = [
            name for name, value in flat_grads
            if not np.isfinite(np.asarray(value)).all()
        ]

        def summary(name, value):
            array = np.asarray(value, dtype=np.float64)
            finite = array[np.isfinite(array)]
            if finite.size:
                bounds = f"{finite.min():.6g}..{finite.max():.6g}"
            else:
                bounds = "no finite values"
            return f"{name}[{bounds}; nonfinite={array.size - finite.size}]"

        inputs = ", ".join(
            summary(name, value)
            for name, value in (
                ("observations", observations),
                ("actions", actions),
                ("old_log_probabilities", old_log_probabilities),
                ("old_values", old_values),
                ("advantages", advantages),
                ("returns", returns),
            )
        )
        tensors = ", ".join(bad) if bad else "not reproduced during eager diagnosis"
        return f"gradient tensors={tensors}; {inputs}"

    def warmup(self, num_steps: int):
        pass

    def _truncation_values(self, terminated, truncated, info):
        """Evaluate only final timeout states, never autoreset observations."""
        selected = np.flatnonzero(np.asarray(truncated) & ~np.asarray(terminated))
        result = mx.zeros(self.config.num_envs)
        if not selected.size:
            return result
        if "terminal_observation" not in info or "_terminal_observation" not in info:
            raise ValueError("PPO timeout requires terminal_observation and its validity mask")
        terminals = mx.array(info["terminal_observation"])
        mask = np.asarray(info["_terminal_observation"])
        expected = (self.config.num_envs, *self.env.observation_space.shape)
        if terminals.shape != expected or mask.shape != (self.config.num_envs,) or mask.dtype != np.bool_:
            raise ValueError("PPO terminal observations or validity mask have invalid shape or dtype")
        if not np.all(mask[selected]):
            raise ValueError("PPO timeout is missing a valid terminal observation")
        indices = mx.array(selected.astype(np.int32))
        _, values = self.network(terminals[indices])
        self._assert_finite("timeout critic values", {"values": values})
        return result.at[indices].add(values.squeeze(-1))

    def train(
        self,
        num_steps: int,
        callback: Optional[Callable] = None,
        *,
        observer: Optional[Callable[[dict], None]] = None,
    ):
        """Train to the step target, optionally observing completed phases.

        The episode callback keeps its ``callback(info, step)`` convention.
        Observer dictionaries contain per-phase ``phase``, ``steps`` and wall
        ``seconds``; updates also contain ``optimizer_steps`` and ``mean_loss``
        plus the mean actual-minibatch ``policy_loss``, ``value_loss``,
        ``entropy``, ``approximate_kl``, ``clip_fraction`` and
        ``explained_variance``.
        Collection includes buffer reset and rollout; update includes last-value
        evaluation, GAE and deferred critic work. Observer time is excluded;
        these intervals do not isolate device execution or transfer costs.
        Observer exceptions propagate. A failed phase emits no completion event.
        Observers must not mutate training state or consume training RNGs.
        """
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        next_done = mx.zeros(self.config.num_envs)
        while self.step < num_steps:
            if observer is not None:
                collection_step = self.step
                collection_start = perf_counter()
            self.buffer.reset()
            timeout_values = []
            for _ in range(self.config.num_steps):
                self._constrain_actor_log_std()
                distribution, value = self.network(observation)
                action = distribution.sample()
                log_probability = distribution.log_prob(action)

                self.key, step_key = mx.random.split(self.key)
                keys = mx.random.split(step_key, self.config.num_envs)
                next_observation, state, reward, terminated, truncated, info = (
                    self.env.step(keys, state, action)
                )
                mx.eval(
                    next_observation, reward, terminated, truncated, *state.values()
                )

                timeout_values.append(self._truncation_values(terminated, truncated, info))
                if callback is not None and "episode" in info:
                    callback(info, self.step)

                self.buffer.add(
                    observation,
                    next_observation,
                    action,
                    reward,
                    terminated,
                    truncated,
                    value=value,
                    log_prob=log_probability,
                )

                observation = next_observation
                next_done = mx.logical_or(terminated, truncated).astype(
                    mx.float32
                )
                self.step += self.config.num_envs

            if observer is not None:
                collection_seconds = perf_counter() - collection_start
                phase_steps = self.step - collection_step
                observer({
                    "phase": "collection",
                    "steps": phase_steps,
                    "seconds": collection_seconds,
                })
                update_start = perf_counter()

            _, last_value = self.network(observation)
            advantages = compute_generalized_advantage_estimate(
                self.buffer.rewards,
                self.buffer.values,
                self.buffer.terminations,
                last_value.squeeze(-1),
                next_done,
                self.config.gamma,
                self.config.gae_lambda,
                truncations=self.buffer.truncations,
                truncation_values=mx.stack(timeout_values),
            )
            returns = advantages + self.buffer.values

            if observer is None:
                self.update(advantages, returns)
            else:
                metrics = self.update(advantages, returns, collect_metrics=True)
                update_seconds = perf_counter() - update_start
                observer({
                    "phase": "update",
                    "steps": phase_steps,
                    "seconds": update_seconds,
                    **metrics,
                })

    def loss_fn(
        self,
        network,
        observations,
        actions,
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    ):
        return self._loss_and_metrics(
            network,
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )[0]

    def _loss_and_metrics(
        self,
        network,
        observations,
        actions,
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    ):
        distribution, new_values = network(observations)
        new_log_probabilities = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # PPO's clipped objective does not need an unbounded probability ratio.
        # Large but finite log-probability differences can overflow exp() during
        # backpropagation before gradient clipping has a chance to act.
        log_ratio = mx.clip(
            new_log_probabilities - old_log_probabilities,
            LOG_RATIO_MIN,
            LOG_RATIO_MAX,
        )
        ratio = mx.exp(log_ratio)

        if self.config.normalize_advantages:
            advantages = (advantages - mx.mean(advantages)) / (
                mx.std(advantages) + 1e-8
            )

        policy_loss = mx.maximum(
            -advantages * ratio,
            -advantages
            * mx.clip(
                ratio,
                1 - self.config.clip_coefficient,
                1 + self.config.clip_coefficient,
            ),
        )
        policy_loss = mx.mean(policy_loss)

        new_values = new_values.squeeze(-1)

        def value_error_loss(predictions):
            errors = predictions - returns
            absolute_errors = mx.abs(errors)
            quadratic = mx.minimum(absolute_errors, VALUE_LOSS_DELTA)
            linear = absolute_errors - quadratic
            return 0.5 * mx.square(quadratic) + VALUE_LOSS_DELTA * linear

        if self.config.clip_value_loss:
            value_loss_unclipped = value_error_loss(new_values)
            clipped_values = old_values + mx.clip(
                new_values - old_values,
                -self.config.clip_coefficient,
                self.config.clip_coefficient,
            )
            value_loss_clipped = value_error_loss(clipped_values)
            value_loss = mx.mean(
                mx.maximum(value_loss_unclipped, value_loss_clipped)
            )
        else:
            value_loss = mx.mean(value_error_loss(new_values))

        entropy_loss = mx.mean(entropy)

        loss = (
            policy_loss
            - self.config.entropy_coefficient * entropy_loss
            + self.config.value_coefficient * value_loss
        )
        approximate_kl = mx.mean((ratio - 1) - log_ratio)
        clip_fraction = mx.mean(
            mx.abs(ratio - 1) > self.config.clip_coefficient
        )
        value_errors = returns - new_values
        return (
            loss,
            policy_loss,
            value_loss,
            entropy_loss,
            approximate_kl,
            clip_fraction,
            mx.sum(returns),
            mx.sum(mx.square(returns)),
            mx.sum(value_errors),
            mx.sum(mx.square(value_errors)),
        )

    def _apply_update(self, loss, grads):
        loss_finite = mx.all(mx.isfinite(loss))
        gradients_finite = self._all_finite(grads)
        update_finite = mx.logical_and(loss_finite, gradients_finite)
        grads = tree_map(
            lambda grad: mx.where(update_finite, grad, mx.zeros_like(grad)),
            grads,
        )
        grads, _ = optim.clip_grad_norm(grads, self.config.max_grad_norm)
        self.optimizer.update(self.network, grads)
        self._constrain_actor_log_std()
        parameters_finite = self._all_finite(self.network.parameters())
        return loss_finite, gradients_finite, parameters_finite

    def update_step(
        self,
        observations,
        actions,
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    ):
        loss, grads = nn.value_and_grad(self.network, self.loss_fn)(
            self.network,
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )
        loss_finite, gradients_finite, parameters_finite = self._apply_update(
            loss, grads
        )
        return loss, loss_finite, gradients_finite, parameters_finite

    def _update_step_with_metrics(
        self,
        observations,
        actions,
        old_log_probabilities,
        old_values,
        advantages,
        returns,
    ):
        metrics, grads = nn.value_and_grad(
            self.network, self._loss_and_metrics
        )(
            self.network,
            observations,
            actions,
            old_log_probabilities,
            old_values,
            advantages,
            returns,
        )
        loss_finite, gradients_finite, parameters_finite = self._apply_update(
            metrics[0], grads
        )
        return (
            *metrics,
            loss_finite,
            gradients_finite,
            parameters_finite,
        )

    def update(
        self,
        advantages: mx.array,
        returns: mx.array,
        *,
        collect_metrics: bool = False,
    ):
        """Update minibatches, returning metrics only when requested.

        Metrics count completed optimizer steps and average the values produced
        by their actual minibatch objective evaluations.
        Metrics require at least one optimizer step; exceptions propagate without
        returning partial metrics. The default return value remains ``None``.
        """
        if collect_metrics:
            optimizer_steps = 0
            totals = {
                "mean_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approximate_kl": 0.0,
                "clip_fraction": 0.0,
            }
            return_sum = 0.0
            return_square_sum = 0.0
            value_error_sum = 0.0
            value_error_square_sum = 0.0
            metric_samples = 0
        observations = flatten(self.buffer.observations)
        actions = flatten(self.buffer.actions)
        log_probabilities = flatten(self.buffer.log_probs)
        values = flatten(self.buffer.values)
        advantages = flatten(advantages)
        returns = flatten(returns)
        self._assert_finite(
            "update inputs",
            {
                "observations": observations,
                "actions": actions,
                "log_probabilities": log_probabilities,
                "values": values,
                "advantages": advantages,
                "returns": returns,
            },
        )

        batch_size = self.config.num_steps * self.config.num_envs
        minibatch_size = batch_size // self.config.num_minibatches

        for _ in range(self.config.update_epochs):
            permutation = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                minibatch_indices = mx.array(
                    permutation[start : start + minibatch_size]
                )
                if collect_metrics:
                    result = self._update_step_with_metrics(
                        observations[minibatch_indices],
                        actions[minibatch_indices],
                        log_probabilities[minibatch_indices],
                        values[minibatch_indices],
                        advantages[minibatch_indices],
                        returns[minibatch_indices],
                    )
                    metric_values = result[:6]
                    metric_moments = result[6:10]
                    loss_finite, gradients_finite, parameters_finite = result[10:]
                    loss = metric_values[0]
                else:
                    loss, loss_finite, gradients_finite, parameters_finite = (
                        self.update_step(
                            observations[minibatch_indices],
                            actions[minibatch_indices],
                            log_probabilities[minibatch_indices],
                            values[minibatch_indices],
                            advantages[minibatch_indices],
                            returns[minibatch_indices],
                        )
                    )
                mx.eval(
                    loss,
                    loss_finite,
                    gradients_finite,
                    parameters_finite,
                    self.network.state,
                    self.optimizer.state,
                )
                self._raise_if_non_finite("loss", loss_finite)
                if not bool(np.asarray(gradients_finite)):
                    details = self._gradient_failure_details(
                        observations[minibatch_indices],
                        actions[minibatch_indices],
                        log_probabilities[minibatch_indices],
                        values[minibatch_indices],
                        advantages[minibatch_indices],
                        returns[minibatch_indices],
                    )
                    raise RuntimeError(
                        f"PPO gradients contains non-finite values: {details}"
                    )
                self._raise_if_non_finite(
                    "model parameters", parameters_finite
                )
                if collect_metrics:
                    for name, value in zip(totals, metric_values):
                        totals[name] += float(value.item())
                    return_sum += float(metric_moments[0].item())
                    return_square_sum += float(metric_moments[1].item())
                    value_error_sum += float(metric_moments[2].item())
                    value_error_square_sum += float(metric_moments[3].item())
                    metric_samples += int(returns[minibatch_indices].size)
                    optimizer_steps += 1

        if collect_metrics:
            if optimizer_steps == 0:
                raise ValueError("PPO update metrics require at least one optimizer step")
            return_variance = (
                return_square_sum / metric_samples
                - (return_sum / metric_samples) ** 2
            )
            value_error_variance = (
                value_error_square_sum / metric_samples
                - (value_error_sum / metric_samples) ** 2
            )
            explained_variance = (
                1 - value_error_variance / return_variance
                if return_variance > 1e-8
                else 0.0
            )
            return {
                "optimizer_steps": optimizer_steps,
                **{
                    name: total / optimizer_steps
                    for name, total in totals.items()
                },
                "explained_variance": explained_variance,
            }

    def evaluate(self, num_steps: int, callback: Optional[Callable] = None):
        self.key, reset_key = mx.random.split(self.key)
        keys = mx.random.split(reset_key, self.config.num_envs)
        observation, state, _ = self.env.reset(keys)
        mx.eval(observation, *state.values())

        for _ in range(0, num_steps, self.config.num_envs):
            self._constrain_actor_log_std()
            distribution, _ = self.network(observation)
            action = distribution.sample()

            self.key, step_key = mx.random.split(self.key)
            keys = mx.random.split(step_key, self.config.num_envs)
            observation, state, reward, terminated, truncated, info = (
                self.env.step(keys, state, action)
            )
            mx.eval(observation, reward, terminated, truncated, *state.values())

            if callback is not None and "episode" in info:
                callback(info, self.step)
