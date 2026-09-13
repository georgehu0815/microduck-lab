"""Create a bridge-training warm start from the shipped walking actor."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


WALK_POLICY = (
    Path(__file__).resolve().parents[3] / "microduck" / "policies" / "alpha_walking.onnx"
)
EXPECTED_NODE_TYPES = (
    "Sub",
    "Div",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
    "Elu",
    "Gemm",
)
EXPECTED_INITIALIZERS = {
    "obs_normalizer._mean": (1, 61),
    "onnx::Div_24": (1, 61),
    "mlp.0.weight": (512, 61),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (256, 512),
    "mlp.2.bias": (256,),
    "mlp.4.weight": (128, 256),
    "mlp.4.bias": (128,),
    "mlp.6.weight": (14, 128),
    "mlp.6.bias": (14,),
}
PARITY_SAMPLES = 32
PARITY_THRESHOLD = 1e-4


def _fixed_shape(value_info: Any) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim)


def _validated_source() -> tuple[Any, dict[str, np.ndarray], str]:
    import onnx
    from onnx import numpy_helper

    if not WALK_POLICY.is_file():
        raise FileNotFoundError(f"walking source policy not found: {WALK_POLICY}")
    source_hash = hashlib.sha256(WALK_POLICY.read_bytes()).hexdigest()
    source = onnx.load(WALK_POLICY)
    onnx.checker.check_model(source)

    if len(source.graph.input) != 1 or source.graph.input[0].name != "obs":
        raise ValueError("Walking teacher input contract changed")
    if _fixed_shape(source.graph.input[0]) != (1, 61):
        raise ValueError("Walking teacher observation shape changed")
    if len(source.graph.output) != 1 or source.graph.output[0].name != "actions":
        raise ValueError("Walking teacher output contract changed")
    if _fixed_shape(source.graph.output[0]) != (1, 14):
        raise ValueError("Walking teacher action shape changed")
    if tuple(node.op_type for node in source.graph.node) != EXPECTED_NODE_TYPES:
        raise ValueError("Walking teacher graph changed; exact transfer needs a parity review")

    tensors = {
        value.name: np.asarray(numpy_helper.to_array(value))
        for value in source.graph.initializer
    }
    shapes = {name: value.shape for name, value in tensors.items()}
    if shapes != EXPECTED_INITIALIZERS:
        raise ValueError("Walking teacher initializer contract changed")
    if any(value.dtype != np.float32 for value in tensors.values()):
        raise ValueError("Walking teacher initializers must be float32")

    mean = tensors["obs_normalizer._mean"]
    standard_deviation = tensors["onnx::Div_24"]
    if not np.all(np.isfinite(mean)):
        raise ValueError("Walking teacher normalizer mean is invalid")
    if not np.all(np.isfinite(standard_deviation)) or not np.all(
        standard_deviation > 0
    ):
        raise ValueError("Walking teacher normalizer standard deviation is invalid")
    return source, tensors, source_hash


def _representative_observations(
    mean: np.ndarray, standard_deviation: np.ndarray, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    normalized = rng.uniform(-3.0, 3.0, size=(PARITY_SAMPLES - 1, 61)).astype(
        np.float32
    )
    normalized = np.concatenate(
        (np.zeros((1, 61), dtype=np.float32), normalized), axis=0
    )
    observations = mean.reshape(1, 61) + normalized * standard_deviation.reshape(
        1, 61
    )
    if not np.all(np.isfinite(observations)):
        raise RuntimeError("Seeded parity observations are not finite")
    return observations.astype(np.float32)


def initialize_bridge(
    output: Path, *, seed: int = 7, initial_std: float = 0.03
) -> Path:
    """Transfer alpha_walking exactly into an MLX actor with fresh training state."""
    import mlx.core as mx
    import onnxruntime as ort

    from rlx.models.microduck import (
        create_actor_critic,
        normalize_observations,
        save_checkpoint,
    )

    if not math.isfinite(initial_std) or initial_std <= 0:
        raise ValueError("initial_std must be finite and positive")

    _, tensors, source_hash = _validated_source()
    mean = tensors["obs_normalizer._mean"].reshape(61).copy()
    standard_deviation = tensors["onnx::Div_24"].reshape(61).copy()
    variance = np.square(standard_deviation, dtype=np.float32)

    mx.random.seed(seed)
    network = create_actor_critic(initial_std=initial_std)
    actor_weights = [
        (
            f"actor_mean.layers.{index}.{kind}",
            mx.array(tensors[f"mlp.{index}.{kind}"].copy()),
        )
        for index in (0, 2, 4, 6)
        for kind in ("weight", "bias")
    ]
    network.load_weights(actor_weights, strict=False)
    mx.eval(network.parameters())

    observations = _representative_observations(mean, standard_deviation, seed)
    session = ort.InferenceSession(
        str(WALK_POLICY), providers=["CPUExecutionProvider"]
    )
    expected = np.concatenate(
        [session.run(["actions"], {"obs": item[None]})[0] for item in observations],
        axis=0,
    )
    normalized = normalize_observations(
        mx.array(observations), mean, variance, epsilon=0.0, clip=1e6
    )
    actual = np.asarray(network.deterministic(normalized), dtype=np.float32)
    parity_error = float(np.max(np.abs(actual - expected)))
    if not math.isfinite(parity_error) or parity_error > PARITY_THRESHOLD:
        raise RuntimeError(
            f"Exact walking actor transfer failed parity: {parity_error}"
        )
    if hashlib.sha256(WALK_POLICY.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Walking source policy changed during initialization")

    return save_checkpoint(
        output,
        network,
        mean,
        variance,
        1.0,
        epsilon=0.0,
        clip=1e6,
        metadata={
            "recipe": "bridge",
            "steps": 0,
            "bootstrap": (
                "exact alpha_walking actor and normalizer transfer; "
                "fresh critic and optimizer"
            ),
            "initialization_method": "exact_weight_transfer",
            "continuation_kind": (
                "actor warm-start with fresh local critic and optimizer"
            ),
            "source_policy": str(WALK_POLICY.resolve()),
            "source_policy_sha256": source_hash,
            "walking_policy_sha256": source_hash,
            "parity_samples": len(observations),
            "parity_max_absolute_action_error": parity_error,
            "parity_threshold": PARITY_THRESHOLD,
            "critic_initialization": "fresh_seeded_mlx",
            "optimizer_initialization": "fresh_by_training_caller",
            "freeze_observation_normalization_by_caller": True,
            "seed": seed,
            "initial_std": initial_std,
            "trained": False,
        },
    )
