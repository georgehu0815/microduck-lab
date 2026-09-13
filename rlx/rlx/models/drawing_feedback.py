"""Structured SB3 policy for learned drawing feedback."""

from __future__ import annotations

import math
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from microduck_local import contract as C
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


RAW_OBSERVATION_DIM = 83
BRUSH_OBSERVATION_DIM = 93
ACTION_DIM = 15
JAW_ACTION_ID = 14
HEAD_DELTA_LIMIT = 0.025
DEFAULT_JAW_OFFSET = -0.8333333
DEFAULT_HEAD_STD = 0.0003
DEFAULT_LEG_STD = 0.0001
DEFAULT_JAW_STD = 0.0002

HEAD_ACTION_IDS = tuple(int(index) for index in C.HEAD_JOINT_IDS)
ANKLE_ACTION_IDS = (4, 13)
BODY_ACTION_IDS = tuple(
    index
    for index in range(C.NUM_JOINTS)
    if index not in HEAD_ACTION_IDS and index not in ANKLE_ACTION_IDS
)

_HEAD_POSITION_IDS = tuple(6 + index for index in HEAD_ACTION_IDS)
_HEAD_LAST_ACTION_IDS = tuple(34 + index for index in HEAD_ACTION_IDS)
_ANGULAR_VELOCITY_SLICE = slice(0, 3)
_GRAVITY_SLICE = slice(3, 6)
_TARGET_ERROR_SLICE = slice(67, 70)
_TOOL_DIRECTION_SLICE = slice(70, 73)
_CONTACT_SLICE = slice(80, 83)
_HEAD_FEATURE_DIM = 4 + 3 + 3 + 3 + 4 * 3
_BALANCE_FEATURE_DIM = 6


def _validate_standard_deviation(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_spaces(
    observation_space: gym.spaces.Space[Any],
    action_space: gym.spaces.Space[Any],
) -> int:
    if not isinstance(observation_space, gym.spaces.Box):
        raise TypeError("drawing feedback requires a Box observation space")
    if observation_space.shape not in (
        (RAW_OBSERVATION_DIM,),
        (BRUSH_OBSERVATION_DIM,),
    ):
        raise ValueError("drawing feedback observations must have 83 or 93 values")
    if not isinstance(action_space, gym.spaces.Box) or action_space.shape != (
        ACTION_DIM,
    ):
        raise ValueError("drawing feedback actions must have shape (15,)")
    return int(observation_space.shape[0])


def _ridge_linear_fit(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ones = torch.ones(
        (features.shape[0], 1), dtype=features.dtype, device=features.device
    )
    design = torch.cat((features, ones), dim=1).to(torch.float64)
    fit_targets = targets.to(torch.float64)
    if ridge:
        regularizer = torch.eye(
            design.shape[1], dtype=design.dtype, device=design.device
        )
        regularizer[-1, -1] = 0
        design = torch.cat((design, math.sqrt(ridge) * regularizer), dim=0)
        fit_targets = torch.cat(
            (
                fit_targets,
                torch.zeros(
                    design.shape[1],
                    fit_targets.shape[1],
                    dtype=fit_targets.dtype,
                    device=fit_targets.device,
                ),
            ),
            dim=0,
        )
    coefficients = torch.linalg.lstsq(design, fit_targets).solution
    return (
        coefficients[:-1].T.to(features.dtype),
        coefficients[-1].to(features.dtype),
    )


class DrawingFeedbackActor(nn.Module):
    """Deterministic actor with learned head increments and balance feedback."""

    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        if observation_dim not in (RAW_OBSERVATION_DIM, BRUSH_OBSERVATION_DIM):
            raise ValueError("observation_dim must be 83 or 93")
        self.observation_dim = observation_dim
        self.head_net = nn.Linear(_HEAD_FEATURE_DIM, len(HEAD_ACTION_IDS))
        self.ankle_net = nn.Linear(_BALANCE_FEATURE_DIM, len(ANKLE_ACTION_IDS))
        self.body_offsets = nn.Parameter(torch.zeros(len(BODY_ACTION_IDS)))
        self.jaw_offset = nn.Parameter(torch.tensor(DEFAULT_JAW_OFFSET))
        self.register_buffer("head_feature_mean", torch.zeros(_HEAD_FEATURE_DIM))
        self.register_buffer("head_feature_std", torch.ones(_HEAD_FEATURE_DIM))
        self.register_buffer(
            "balance_feature_mean", torch.zeros(_BALANCE_FEATURE_DIM)
        )
        self.register_buffer(
            "balance_feature_std", torch.ones(_BALANCE_FEATURE_DIM)
        )
        nn.init.zeros_(self.head_net.weight)
        nn.init.zeros_(self.head_net.bias)
        nn.init.zeros_(self.ankle_net.weight)
        nn.init.zeros_(self.ankle_net.bias)

    def _check_observations(self, observations: torch.Tensor) -> None:
        if (
            not torch.jit.is_tracing()
            and observations.shape[-1] != self.observation_dim
        ):
            raise ValueError(
                f"expected observations ending in {self.observation_dim} values"
            )

    def _head_features(self, observations: torch.Tensor) -> torch.Tensor:
        head_positions = observations[..., _HEAD_POSITION_IDS]
        target_error = observations[..., _TARGET_ERROR_SLICE]
        interactions = (
            head_positions.unsqueeze(-1) * target_error.unsqueeze(-2)
        ).flatten(start_dim=-2)
        return torch.cat(
            (
                head_positions,
                target_error,
                observations[..., _TOOL_DIRECTION_SLICE],
                observations[..., _CONTACT_SLICE],
                interactions,
            ),
            dim=-1,
        )

    def _balance_features(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                observations[..., _GRAVITY_SLICE],
                observations[..., _ANGULAR_VELOCITY_SLICE],
            ),
            dim=-1,
        )

    @staticmethod
    def _normalize(
        features: torch.Tensor, mean: torch.Tensor, standard_deviation: torch.Tensor
    ) -> torch.Tensor:
        return (features - mean) / standard_deviation.clamp_min(1e-6)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self._check_observations(observations)
        head_features = self._normalize(
            self._head_features(observations),
            self.head_feature_mean,
            self.head_feature_std,
        )
        balance_features = self._normalize(
            self._balance_features(observations),
            self.balance_feature_mean,
            self.balance_feature_std,
        )
        previous_head_actions = observations[..., _HEAD_LAST_ACTION_IDS]
        head_actions = previous_head_actions + torch.clamp(
            self.head_net(head_features),
            -HEAD_DELTA_LIMIT,
            HEAD_DELTA_LIMIT,
        )
        ankle_actions = self.ankle_net(balance_features)
        leading_shape = observations.shape[:-1]
        body_actions = self.body_offsets.expand(*leading_shape, -1)
        jaw_action = self.jaw_offset.expand(*leading_shape, 1)
        actions = torch.cat(
            (
                body_actions[..., :4],
                ankle_actions[..., :1],
                head_actions,
                body_actions[..., 4:],
                ankle_actions[..., 1:],
                jaw_action,
            ),
            dim=-1,
        )
        return torch.clamp(actions, -1.0, 1.0)

    def fit_feedback(
        self,
        observations: np.ndarray | torch.Tensor,
        teacher_actions: np.ndarray | torch.Tensor,
        *,
        ridge: float = 1e-4,
        head_scale: float = HEAD_DELTA_LIMIT,
        leg_scale: float = 0.1,
        jaw_scale: float = 0.2,
    ) -> dict[str, float | int]:
        return fit_feedback(
            self,
            observations,
            teacher_actions,
            ridge=ridge,
            head_scale=head_scale,
            leg_scale=leg_scale,
            jaw_scale=jaw_scale,
        )


class DrawingFeedbackPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy using a structured drawing-feedback mean actor."""

    def __init__(
        self,
        observation_space: gym.spaces.Space[Any],
        action_space: gym.spaces.Space[Any],
        lr_schedule: Any,
        *,
        head_std: float = DEFAULT_HEAD_STD,
        leg_std: float = DEFAULT_LEG_STD,
        jaw_std: float = DEFAULT_JAW_STD,
        value_net_arch: Sequence[int] = (64, 64),
        **kwargs: Any,
    ) -> None:
        self._drawing_observation_dim = _validate_spaces(
            observation_space, action_space
        )
        head_std = _validate_standard_deviation("head_std", head_std)
        leg_std = _validate_standard_deviation("leg_std", leg_std)
        jaw_std = _validate_standard_deviation("jaw_std", jaw_std)
        standard_deviations = np.full(ACTION_DIM, leg_std, dtype=np.float32)
        standard_deviations[list(HEAD_ACTION_IDS)] = head_std
        standard_deviations[JAW_ACTION_ID] = jaw_std
        self._drawing_initial_log_std = torch.log(
            torch.as_tensor(standard_deviations)
        )
        if kwargs.pop("use_sde", False):
            raise ValueError("drawing feedback supports standard Gaussian noise only")
        if kwargs.pop("squash_output", False):
            raise ValueError("drawing feedback does not use squashed distributions")
        kwargs.pop("net_arch", None)
        kwargs.pop("log_std_init", None)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch={"pi": [], "vf": [int(width) for width in value_net_arch]},
            ortho_init=False,
            use_sde=False,
            squash_output=False,
            log_std_init=0.0,
            **kwargs,
        )

    def _build(self, lr_schedule: Any) -> None:
        self._build_mlp_extractor()
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise TypeError("drawing feedback requires DiagGaussianDistribution")
        self.action_net = DrawingFeedbackActor(self._drawing_observation_dim)
        self.log_std = nn.Parameter(self._drawing_initial_log_std.clone())
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    @property
    def feedback_actor(self) -> DrawingFeedbackActor:
        return self.action_net

    def fit_feedback(
        self,
        observations: np.ndarray | torch.Tensor,
        teacher_actions: np.ndarray | torch.Tensor,
        *,
        ridge: float = 1e-4,
        head_scale: float = HEAD_DELTA_LIMIT,
        leg_scale: float = 0.1,
        jaw_scale: float = 0.2,
    ) -> dict[str, float | int]:
        return fit_feedback(
            self.feedback_actor,
            observations,
            teacher_actions,
            ridge=ridge,
            head_scale=head_scale,
            leg_scale=leg_scale,
            jaw_scale=jaw_scale,
        )


def fit_feedback(
    actor_or_policy: DrawingFeedbackActor | DrawingFeedbackPolicy,
    observations: np.ndarray | torch.Tensor,
    teacher_actions: np.ndarray | torch.Tensor,
    *,
    ridge: float = 1e-4,
    head_scale: float = HEAD_DELTA_LIMIT,
    leg_scale: float = 0.1,
    jaw_scale: float = 0.2,
) -> dict[str, float | int]:
    """Fit the structured actor to absolute teacher actions without rollouts."""
    actor = (
        actor_or_policy.feedback_actor
        if isinstance(actor_or_policy, DrawingFeedbackPolicy)
        else actor_or_policy
    )
    if not isinstance(actor, DrawingFeedbackActor):
        raise TypeError("fit_feedback requires a drawing feedback actor or policy")
    if next(actor.parameters()).device.type != "cpu":
        raise ValueError("drawing feedback fitting is CPU-only")
    if not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    scales = (
        _validate_standard_deviation("head_scale", head_scale),
        _validate_standard_deviation("leg_scale", leg_scale),
        _validate_standard_deviation("jaw_scale", jaw_scale),
    )
    obs = torch.as_tensor(observations, dtype=torch.float32, device="cpu")
    targets = torch.as_tensor(teacher_actions, dtype=torch.float32, device="cpu")
    if obs.ndim != 2 or obs.shape[1] != actor.observation_dim:
        raise ValueError(
            f"observations must have shape (N, {actor.observation_dim})"
        )
    if targets.shape != (obs.shape[0], ACTION_DIM):
        raise ValueError("teacher_actions must have shape (N, 15)")
    if obs.shape[0] < 1:
        raise ValueError("at least one teacher sample is required")
    if not torch.isfinite(obs).all() or not torch.isfinite(targets).all():
        raise ValueError("teacher data must be finite")

    loss_scales = torch.full((ACTION_DIM,), scales[1], dtype=torch.float32)
    loss_scales[list(HEAD_ACTION_IDS)] = scales[0]
    loss_scales[JAW_ACTION_ID] = scales[2]
    with torch.no_grad():
        before = actor(obs)
        before_loss = torch.mean(torch.square((before - targets) / loss_scales))

        head_features = actor._head_features(obs)
        head_mean = head_features.mean(dim=0)
        head_std = head_features.std(dim=0, unbiased=False).clamp_min(1e-4)
        actor.head_feature_mean.copy_(head_mean)
        actor.head_feature_std.copy_(head_std)
        normalized_head = (head_features - head_mean) / head_std
        previous_head = obs[:, _HEAD_LAST_ACTION_IDS]
        head_delta = (targets[:, HEAD_ACTION_IDS] - previous_head).clamp(
            -HEAD_DELTA_LIMIT, HEAD_DELTA_LIMIT
        )
        head_weight, head_bias = _ridge_linear_fit(
            normalized_head, head_delta, ridge
        )
        actor.head_net.weight.copy_(head_weight)
        actor.head_net.bias.copy_(head_bias)

        balance_features = actor._balance_features(obs)
        balance_mean = balance_features.mean(dim=0)
        balance_std = balance_features.std(dim=0, unbiased=False).clamp_min(1e-4)
        actor.balance_feature_mean.copy_(balance_mean)
        actor.balance_feature_std.copy_(balance_std)
        normalized_balance = (balance_features - balance_mean) / balance_std
        ankle_weight, ankle_bias = _ridge_linear_fit(
            normalized_balance, targets[:, ANKLE_ACTION_IDS], ridge
        )
        actor.ankle_net.weight.copy_(ankle_weight)
        actor.ankle_net.bias.copy_(ankle_bias)

        sample_count = float(obs.shape[0])
        shrinkage = sample_count / (sample_count + ridge)
        actor.body_offsets.copy_(
            shrinkage * targets[:, BODY_ACTION_IDS].mean(dim=0)
        )
        jaw_target = targets[:, JAW_ACTION_ID].sum()
        actor.jaw_offset.copy_(
            (jaw_target + ridge * DEFAULT_JAW_OFFSET) / (sample_count + ridge)
        )

        after = actor(obs)
        after_loss = torch.mean(torch.square((after - targets) / loss_scales))
        head_increment_error = (
            (after[:, HEAD_ACTION_IDS] - obs[:, _HEAD_LAST_ACTION_IDS])
            - (targets[:, HEAD_ACTION_IDS] - obs[:, _HEAD_LAST_ACTION_IDS])
        )
    return {
        "samples": int(obs.shape[0]),
        "normalized_increment_mse_before": float(before_loss),
        "normalized_increment_mse_after": float(after_loss),
        "head_increment_mse": float(torch.mean(torch.square(head_increment_error))),
        "ankle_mse": float(
            torch.mean(
                torch.square(
                    after[:, ANKLE_ACTION_IDS] - targets[:, ANKLE_ACTION_IDS]
                )
            )
        ),
    }


__all__ = [
    "ACTION_DIM",
    "ANKLE_ACTION_IDS",
    "BODY_ACTION_IDS",
    "BRUSH_OBSERVATION_DIM",
    "DEFAULT_HEAD_STD",
    "DEFAULT_JAW_OFFSET",
    "DEFAULT_JAW_STD",
    "DEFAULT_LEG_STD",
    "DrawingFeedbackActor",
    "DrawingFeedbackPolicy",
    "HEAD_ACTION_IDS",
    "HEAD_DELTA_LIMIT",
    "JAW_ACTION_ID",
    "RAW_OBSERVATION_DIM",
    "fit_feedback",
]
