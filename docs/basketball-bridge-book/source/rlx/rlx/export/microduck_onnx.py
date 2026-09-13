"""Direct deterministic ONNX export for the lazy MicroDuck MLX actor."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

import numpy as np


def export_deterministic_actor(
    checkpoint: str | Path,
    output: str | Path,
    *,
    opset: int = 17,
) -> Path:
    """Export actor mean with baked normalization and a dynamic batch axis."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from mlx.utils import tree_flatten

    from rlx.models.microduck import load_checkpoint

    loaded = load_checkpoint(checkpoint)
    weights = {name: np.asarray(value, dtype=np.float32)
               for name, value in tree_flatten(loaded["model"].parameters())}
    initializers = [
        numpy_helper.from_array(loaded["mean"], "obs_mean"),
        numpy_helper.from_array(loaded["variance"], "obs_variance"),
        numpy_helper.from_array(np.array(loaded["epsilon"], np.float32), "epsilon"),
        numpy_helper.from_array(np.array(-loaded["clip"], np.float32), "clip_min"),
        numpy_helper.from_array(np.array(loaded["clip"], np.float32), "clip_max"),
    ]
    nodes = [
        helper.make_node("Sub", ["observations", "obs_mean"], ["centered"]),
        helper.make_node("Add", ["obs_variance", "epsilon"], ["variance_eps"]),
        helper.make_node("Sqrt", ["variance_eps"], ["std"]),
        helper.make_node("Div", ["centered", "std"], ["normalized_unclipped"]),
        helper.make_node(
            "Clip", ["normalized_unclipped", "clip_min", "clip_max"], ["hidden_0"]
        ),
    ]
    layer_indices = (0, 2, 4, 6)
    current = "hidden_0"
    for ordinal, layer_index in enumerate(layer_indices):
        prefix = f"actor_mean.layers.{layer_index}"
        weight_name = f"actor_weight_{ordinal}"
        bias_name = f"actor_bias_{ordinal}"
        initializers.extend(
            (
                numpy_helper.from_array(weights[f"{prefix}.weight"], weight_name),
                numpy_helper.from_array(weights[f"{prefix}.bias"], bias_name),
            )
        )
        linear = "actor_linear" if ordinal == len(layer_indices) - 1 else f"linear_{ordinal}"
        nodes.append(
            helper.make_node(
                "Gemm", [current, weight_name, bias_name], [linear], transB=1
            )
        )
        if linear != "actor_linear":
            current = f"hidden_{ordinal + 1}"
            nodes.append(helper.make_node("Elu", [linear], [current], alpha=1.0))

    output_name = "actor_linear"
    action_limit = loaded["metadata"].get("policy_action_limit")
    if action_limit is not None:
        nodes.append(helper.make_node("Tanh", ["actor_linear"], ["actor_tanh"]))
        initializers.append(
            numpy_helper.from_array(np.array(action_limit, np.float32), "action_limit")
        )
        nodes.append(
            helper.make_node("Mul", ["actor_tanh", "action_limit"], ["actions"])
        )
        output_name = "actions"
    elif loaded["metadata"].get("recipe") == "swing":
        initializers.extend(
            (
                numpy_helper.from_array(np.array(-1.0, np.float32), "action_min"),
                numpy_helper.from_array(np.array(1.0, np.float32), "action_max"),
            )
        )
        nodes.append(
            helper.make_node(
                "Clip",
                ["actor_linear", "action_min", "action_max"],
                ["bounded_actions"],
                name="swing_action_clip",
            )
        )
        output_name = "bounded_actions"

    graph = helper.make_graph(
        nodes,
        "RLX MicroDuck deterministic actor",
        [helper.make_tensor_value_info(
            "observations", TensorProto.FLOAT, ["batch", 61]
        )],
        [helper.make_tensor_value_info(
            output_name, TensorProto.FLOAT, ["batch", 14]
        )],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="rlx",
        opset_imports=[helper.make_opsetid("", opset)],
    )
    model.ir_version = min(model.ir_version, 10)
    if loaded["metadata"].get("recipe") == "backflip":
        helper.set_model_props(model, {
            "rlx_metadata": json.dumps(loaded["metadata"], sort_keys=True),
            "rlx_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "controller_scope": "landing policy only; requires spotter launch and pretrained stand handoff",
        })
    onnx.checker.check_model(model)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, target)
    return target


def parity_error(checkpoint: str | Path, onnx_path: str | Path, batch: Any) -> float:
    """Return maximum absolute MLX/ONNX deterministic-policy error."""
    import mlx.core as mx
    import onnxruntime as ort

    from rlx.models.microduck import load_checkpoint, normalize_observations

    loaded = load_checkpoint(checkpoint)
    values = np.asarray(batch, dtype=np.float32)
    normalized = normalize_observations(
        mx.array(values), loaded["mean"], loaded["variance"],
        epsilon=loaded["epsilon"], clip=loaded["clip"],
    )
    expected = np.asarray(loaded["model"].deterministic(normalized))
    if loaded["metadata"].get("recipe") == "swing":
        expected = np.clip(expected, -1.0, 1.0)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        [session.get_outputs()[0].name], {"observations": values}
    )[0]
    return float(np.max(np.abs(expected - actual)))
