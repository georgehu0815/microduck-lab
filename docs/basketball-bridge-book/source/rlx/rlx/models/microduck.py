"""Lazy MLX model and safe checkpoint helpers for MicroDuck."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

OBSERVATION_DIM = 61
ACTION_DIM = 14
HIDDEN_DIMS = (512, 256, 128)
CHECKPOINT_FORMAT = "rlx.microduck.actor_critic.v1"
ACTOR_LOG_STD_MIN = -5.0
ACTOR_LOG_STD_MAX = -0.5


def create_actor_critic(
    *,
    initial_std: float | None = None,
    action_limit: float | None = None,
) -> Any:
    """Create the 512-256-128 actor-critic, importing MLX only on demand."""
    if initial_std is not None and initial_std <= 0.0:
        raise ValueError("initial_std must be positive")
    if action_limit is not None and action_limit <= 0.0:
        raise ValueError("action_limit must be positive")
    import mlx.core as mx
    import mlx.nn as nn

    class MicroDuckActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor_mean = _mlp(nn, OBSERVATION_DIM, ACTION_DIM)
            self.actor_log_std = (
                mx.zeros((ACTION_DIM,))
                if initial_std is None
                else mx.full((ACTION_DIM,), math.log(initial_std))
            )
            self.critic = _mlp(nn, OBSERVATION_DIM, 1)

        def __call__(self, observation: Any) -> Any:
            from rlx.utils.distributions import Gaussian

            mean = self._actor_output(observation)
            log_std = mx.broadcast_to(
                mx.clip(
                    self.actor_log_std,
                    ACTOR_LOG_STD_MIN,
                    ACTOR_LOG_STD_MAX,
                ),
                mean.shape,
            )
            return Gaussian(mean, log_std), self.critic(observation)

        def deterministic(self, observation: Any) -> Any:
            return self._actor_output(observation)

        def _actor_output(self, observation: Any) -> Any:
            mean = self.actor_mean(observation)
            if action_limit is not None:
                mean = mx.tanh(mean) * action_limit
            return mean

        def constrain_actor_log_std(self) -> None:
            """Keep MicroDuck exploration inside its safe operating range."""
            self.actor_log_std = mx.clip(
                self.actor_log_std,
                ACTOR_LOG_STD_MIN,
                ACTOR_LOG_STD_MAX,
            )

    return MicroDuckActorCritic()


def _mlp(nn: Any, input_dim: int, output_dim: int) -> Any:
    class StableELU(nn.Module):
        def __call__(self, values: Any) -> Any:
            return stable_elu(values)

    layers: list[Any] = []
    previous = input_dim
    for width in HIDDEN_DIMS:
        layers.extend((nn.Linear(previous, width), StableELU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


def stable_elu(values: Any) -> Any:
    """ELU with a bounded inactive exponential branch and finite gradients."""
    import mlx.core as mx

    return mx.where(values >= 0, values, mx.exp(mx.minimum(values, 0)) - 1)


def normalize_observations(
    observations: Any,
    mean: Any,
    variance: Any,
    *,
    epsilon: float = 1e-8,
    clip: float = 10.0,
) -> Any:
    """Apply the adapter's observation-normalization contract in MLX."""
    import mlx.core as mx

    return mx.clip(
        (observations - mx.array(mean)) / mx.sqrt(mx.array(variance) + epsilon),
        -clip,
        clip,
    )


def save_checkpoint(
    path: str | Path,
    model: Any,
    observation_mean: Any,
    observation_variance: Any,
    observation_count: float,
    *,
    return_mean: Any | None = None,
    return_variance: Any | None = None,
    return_count: float | None = None,
    epsilon: float = 1e-8,
    clip: float = 10.0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save weights and normalizer without pickle."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.{name}": value for name, value in tree_flatten(model.parameters())}
    tensors["normalizer.mean"] = mx.array(np.asarray(observation_mean, np.float32))
    tensors["normalizer.variance"] = mx.array(
        np.asarray(observation_variance, np.float32)
    )
    tensors["normalizer.count"] = mx.array([observation_count], dtype=mx.float32)
    return_values = (return_mean, return_variance, return_count)
    if any(value is not None for value in return_values):
        if not all(value is not None for value in return_values):
            raise ValueError(
                "return_mean, return_variance, and return_count must be provided together"
            )
        tensors["return_normalizer.mean"] = mx.array(
            np.asarray(return_mean, np.float32)
        )
        tensors["return_normalizer.variance"] = mx.array(
            np.asarray(return_variance, np.float32)
        )
        tensors["return_normalizer.count"] = mx.array(
            [return_count], dtype=mx.float32
        )
    finite_checks = {
        name: mx.all(mx.isfinite(value)) for name, value in tensors.items()
    }
    mx.eval(*finite_checks.values())
    non_finite = [
        name for name, finite in finite_checks.items() if not bool(np.asarray(finite))
    ]
    if non_finite:
        raise RuntimeError(
            "refusing to save checkpoint with non-finite tensor(s): "
            + ", ".join(non_finite)
        )
    mx.save_safetensors(str(target), tensors, metadata={"format": CHECKPOINT_FORMAT})
    sidecar = {
        "format": CHECKPOINT_FORMAT,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "epsilon": float(epsilon),
        "clip": float(clip),
        "metadata": metadata or {},
    }
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    )
    return target


def load_checkpoint(path: str | Path, model: Any | None = None) -> dict[str, Any]:
    """Load a checkpoint and return its reconstructed model and normalizer."""
    import mlx.core as mx

    source = Path(path)
    sidecar = json.loads(source.with_suffix(source.suffix + ".json").read_text())
    if sidecar.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported MicroDuck checkpoint: {sidecar.get('format')!r}")
    tensors = mx.load(str(source))
    if model is None:
        model = create_actor_critic(
            action_limit=sidecar.get("metadata", {}).get("policy_action_limit")
        )
    model.load_weights(
        [(name.removeprefix("model."), value) for name, value in tensors.items()
         if name.startswith("model.")],
        strict=True,
    )
    mx.eval(model.parameters())
    has_return_normalizer = "return_normalizer.mean" in tensors
    return {
        "model": model,
        "mean": np.asarray(tensors["normalizer.mean"], dtype=np.float32),
        "variance": np.asarray(tensors["normalizer.variance"], dtype=np.float32),
        "count": float(np.asarray(tensors["normalizer.count"])[0]),
        "return_mean": (
            np.asarray(tensors["return_normalizer.mean"], dtype=np.float32)
            if has_return_normalizer
            else None
        ),
        "return_variance": (
            np.asarray(tensors["return_normalizer.variance"], dtype=np.float32)
            if has_return_normalizer
            else None
        ),
        "return_count": (
            float(np.asarray(tensors["return_normalizer.count"])[0])
            if has_return_normalizer
            else None
        ),
        "epsilon": float(sidecar["epsilon"]),
        "clip": float(sidecar["clip"]),
        "metadata": sidecar.get("metadata", {}),
    }
