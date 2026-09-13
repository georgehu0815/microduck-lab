"""Contracts for the learned drawing and brush feedback policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlx.models.drawing_feedback import (
    ACTION_DIM,
    BODY_ACTION_IDS,
    BRUSH_OBSERVATION_DIM,
    DEFAULT_HEAD_STD,
    DEFAULT_JAW_OFFSET,
    DEFAULT_JAW_STD,
    DEFAULT_LEG_STD,
    DrawingFeedbackActor,
    DrawingFeedbackPolicy,
    HEAD_ACTION_IDS,
    HEAD_DELTA_LIMIT,
    JAW_ACTION_ID,
    RAW_OBSERVATION_DIM,
)


def _policy(observation_dim: int = BRUSH_OBSERVATION_DIM):
    import gymnasium as gym

    return DrawingFeedbackPolicy(
        gym.spaces.Box(
            -np.inf, np.inf, (observation_dim,), dtype=np.float32
        ),
        gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32),
        lambda _: 3e-4,
    )


def test_policy_uses_sb3_gaussian_forward_and_evaluate_actions_consistently():
    policy = _policy()
    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(
        8, BRUSH_OBSERVATION_DIM, generator=generator
    )

    actions, values, forward_log_prob = policy(
        observations, deterministic=False
    )
    evaluated_values, evaluated_log_prob, entropy = policy.evaluate_actions(
        observations, actions
    )

    assert actions.shape == (8, ACTION_DIM)
    assert values.shape == evaluated_values.shape == (8, 1)
    assert forward_log_prob.shape == evaluated_log_prob.shape == (8,)
    assert entropy is not None and entropy.shape == (8,)
    torch.testing.assert_close(values, evaluated_values)
    torch.testing.assert_close(forward_log_prob, evaluated_log_prob)
    expected_std = torch.full((ACTION_DIM,), DEFAULT_LEG_STD)
    expected_std[list(HEAD_ACTION_IDS)] = DEFAULT_HEAD_STD
    expected_std[JAW_ACTION_ID] = DEFAULT_JAW_STD
    torch.testing.assert_close(policy.log_std.exp(), expected_std)


@pytest.mark.parametrize(
    "observation_dim", [RAW_OBSERVATION_DIM, BRUSH_OBSERVATION_DIM]
)
def test_actor_skips_previous_head_actions_and_ignores_brush_tail(
    observation_dim,
):
    actor = DrawingFeedbackActor(observation_dim)
    observations = torch.zeros(3, observation_dim)
    previous_head = torch.tensor(
        [
            [-0.20, -0.10, 0.05, 0.12],
            [0.18, -0.04, -0.09, 0.02],
            [0.01, 0.02, 0.03, 0.04],
        ]
    )
    observations[:, [34 + index for index in HEAD_ACTION_IDS]] = previous_head

    actions = actor(observations)

    assert actions.shape == (3, ACTION_DIM)
    torch.testing.assert_close(actions[:, HEAD_ACTION_IDS], previous_head)
    torch.testing.assert_close(
        actions[:, JAW_ACTION_ID],
        torch.full((3,), DEFAULT_JAW_OFFSET),
    )
    if observation_dim == BRUSH_OBSERVATION_DIM:
        changed_tail = observations.clone()
        changed_tail[:, RAW_OBSERVATION_DIM:] = torch.randn(
            3, BRUSH_OBSERVATION_DIM - RAW_OBSERVATION_DIM
        )
        torch.testing.assert_close(actor(changed_tail), actions)


def test_head_feedback_is_linear_polynomial_then_clipped_to_teacher_limit():
    actor = DrawingFeedbackActor(BRUSH_OBSERVATION_DIM)
    with torch.no_grad():
        actor.head_net.bias.fill_(1.0)
    observation = torch.zeros(1, BRUSH_OBSERVATION_DIM)
    observation[:, [34 + index for index in HEAD_ACTION_IDS]] = 0.1

    actions = actor(observation)

    torch.testing.assert_close(
        actions[:, HEAD_ACTION_IDS],
        torch.full((1, len(HEAD_ACTION_IDS)), 0.1 + HEAD_DELTA_LIMIT),
    )


def test_supervised_fit_recovers_structured_absolute_teacher_actions():
    generator = torch.Generator().manual_seed(11)
    sample_count = 512
    observations = torch.zeros(sample_count, BRUSH_OBSERVATION_DIM)
    observations[:, :6] = 0.2 * torch.randn(
        sample_count, 6, generator=generator
    )
    observations[:, 6:20] = 0.15 * torch.randn(
        sample_count, 14, generator=generator
    )
    observations[:, 34:48] = 0.2 * torch.randn(
        sample_count, 14, generator=generator
    )
    observations[:, 67:73] = 0.04 * torch.randn(
        sample_count, 6, generator=generator
    )
    observations[:, 70:73] += torch.tensor([1.0, 0.0, 0.0])
    observations[:, 80:83] = 0.1 * torch.randn(
        sample_count, 3, generator=generator
    )

    source = DrawingFeedbackActor(BRUSH_OBSERVATION_DIM)
    with torch.no_grad():
        source.head_feature_mean.copy_(
            source._head_features(observations).mean(dim=0)
        )
        source.head_feature_std.copy_(
            source._head_features(observations)
            .std(dim=0, unbiased=False)
            .clamp_min(1e-4)
        )
        source.balance_feature_mean.copy_(
            source._balance_features(observations).mean(dim=0)
        )
        source.balance_feature_std.copy_(
            source._balance_features(observations)
            .std(dim=0, unbiased=False)
            .clamp_min(1e-4)
        )
        source.head_net.weight.normal_(0.0, 0.0003, generator=generator)
        source.head_net.bias.uniform_(-0.003, 0.003, generator=generator)
        source.ankle_net.weight.normal_(0.0, 0.015, generator=generator)
        source.ankle_net.bias.copy_(torch.tensor([0.012, -0.009]))
        source.body_offsets.copy_(
            torch.linspace(-0.025, 0.025, len(BODY_ACTION_IDS))
        )
        source.jaw_offset.fill_(-0.81)
        teacher_actions = source(observations)

    policy = _policy()
    before = torch.mean(
        torch.square(policy.feedback_actor(observations) - teacher_actions)
    )
    metrics = policy.fit_feedback(
        observations, teacher_actions, ridge=1e-6
    )
    after = torch.mean(
        torch.square(policy.feedback_actor(observations) - teacher_actions)
    )

    assert metrics["samples"] == sample_count
    assert metrics["normalized_increment_mse_after"] < (
        metrics["normalized_increment_mse_before"] * 1e-3
    )
    assert after < before * 1e-3
    assert metrics["head_increment_mse"] < 1e-8
    assert metrics["ankle_mse"] < 1e-8


def test_deterministic_actor_exports_to_onnx_and_matches_torch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    policy = _policy()
    actor = policy.feedback_actor.eval()
    observations = torch.randn(5, BRUSH_OBSERVATION_DIM)
    path = tmp_path / "drawing_feedback.onnx"

    torch.onnx.export(
        actor,
        torch.zeros(1, BRUSH_OBSERVATION_DIM),
        path,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    expected = actor(observations).detach().numpy()
    actual = session.run(
        ["actions"], {"obs": observations.numpy().astype(np.float32)}
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.shape == (5, ACTION_DIM)
