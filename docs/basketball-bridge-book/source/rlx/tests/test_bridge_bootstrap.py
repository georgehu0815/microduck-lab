import hashlib
import json

import numpy as np
import pytest


def test_initialize_bridge_transfers_walking_actor_and_normalizer(tmp_path):
    mx = pytest.importorskip("mlx.core")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")

    from rlx.models.bridge_bootstrap import (
        PARITY_SAMPLES,
        PARITY_THRESHOLD,
        WALK_POLICY,
        initialize_bridge,
    )
    from rlx.models.microduck import load_checkpoint, normalize_observations

    output = tmp_path / "bridge.safetensors"
    assert initialize_bridge(output) == output

    loaded = load_checkpoint(output)
    metadata = loaded["metadata"]
    assert metadata["recipe"] == "bridge"
    assert metadata["steps"] == 0
    assert metadata["trained"] is False
    assert metadata["source_policy"].endswith(
        "microduck/policies/alpha_walking.onnx"
    )
    assert metadata["source_policy_sha256"] == hashlib.sha256(
        WALK_POLICY.read_bytes()
    ).hexdigest()
    assert metadata["walking_policy_sha256"] == metadata["source_policy_sha256"]
    assert metadata["continuation_kind"] == (
        "actor warm-start with fresh local critic and optimizer"
    )
    assert metadata["critic_initialization"] == "fresh_seeded_mlx"
    assert metadata["optimizer_initialization"] == "fresh_by_training_caller"
    assert metadata["freeze_observation_normalization_by_caller"] is True
    assert metadata["parity_samples"] == PARITY_SAMPLES
    assert metadata["parity_max_absolute_action_error"] <= PARITY_THRESHOLD
    assert loaded["epsilon"] == 0.0
    assert loaded["clip"] == 1e6

    session = ort.InferenceSession(
        str(WALK_POLICY), providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(19)
    normalized = rng.uniform(-2.5, 2.5, size=(8, 61)).astype(np.float32)
    observations = (
        loaded["mean"]
        + normalized * np.sqrt(loaded["variance"], dtype=np.float32)
    ).astype(np.float32)
    expected = np.concatenate(
        [session.run(["actions"], {"obs": item[None]})[0] for item in observations]
    )
    inputs = normalize_observations(
        mx.array(observations),
        loaded["mean"],
        loaded["variance"],
        epsilon=loaded["epsilon"],
        clip=loaded["clip"],
    )
    actual = np.asarray(loaded["model"].deterministic(inputs), dtype=np.float32)
    np.testing.assert_allclose(actual, expected, atol=PARITY_THRESHOLD, rtol=0)

    sidecar = json.loads(output.with_suffix(".safetensors.json").read_text())
    assert sidecar["metadata"] == metadata
