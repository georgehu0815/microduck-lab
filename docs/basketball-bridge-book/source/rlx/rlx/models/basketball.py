"""Torch recurrent policy support for the released blind basketball actor."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

OBSERVATION_DIM = 61
ACTION_DIM = 14
HIDDEN_SIZE = 256
HIDDEN_DIMS = (512, 256, 128)
NORMALIZER_EPSILON = 0.01
SOURCE_CHECKPOINT_SHA256 = (
    "57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41"
)
SOURCE_ONNX_SHA256 = (
    "e105148b160b3a86170621648215be38dd5018e955930ce182394f4513c7569b"
)
LOCAL_CHECKPOINT_FORMAT = "rlx.basketball.local_recurrent_ppo.v1"


class ObservationNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, OBSERVATION_DIM))
        self.register_buffer("_var", torch.ones(1, OBSERVATION_DIM))
        self.register_buffer("_std", torch.ones(1, OBSERVATION_DIM))
        self.register_buffer("count", torch.tensor(0.0))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self._mean) / (self._std + NORMALIZER_EPSILON)


class RecurrentCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.LSTM(OBSERVATION_DIM, HIDDEN_SIZE, num_layers=1)


class GaussianDistribution(nn.Module):
    def __init__(self, initial_std: float) -> None:
        super().__init__()
        if not np.isfinite(initial_std) or initial_std <= 0.0:
            raise ValueError("initial_std must be finite and positive")
        self.std_param = nn.Parameter(torch.full((ACTION_DIM,), float(initial_std)))


class BasketballActor(nn.Module):
    def __init__(self, *, initial_std: float = 0.03) -> None:
        super().__init__()
        self.obs_normalizer = ObservationNormalizer()
        self.distribution = GaussianDistribution(initial_std)
        self.rnn = RecurrentCore()
        layers: list[nn.Module] = []
        previous = HIDDEN_SIZE
        for width in HIDDEN_DIMS:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(nn.Linear(previous, ACTION_DIM))
        self.mlp = nn.Sequential(*layers)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        target_device = parameter.device if device is None else device
        target_dtype = parameter.dtype if dtype is None else dtype
        shape = (1, batch_size, HIDDEN_SIZE)
        return (
            torch.zeros(shape, device=target_device, dtype=target_dtype),
            torch.zeros(shape, device=target_device, dtype=target_dtype),
        )

    def forward(
        self,
        observations: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.obs_normalizer(observations)
        recurrent, (h_out, c_out) = self.rnn.rnn(
            normalized.unsqueeze(0),
            (h_in, c_in),
        )
        actions = self.mlp(recurrent.squeeze(0))
        return actions, h_out, c_out

    def forward_sequence(
        self,
        observations: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
        episode_starts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 3 or observations.shape[-1] != OBSERVATION_DIM:
            raise ValueError("observations must have shape [time, batch, 61]")
        if episode_starts.shape != observations.shape[:2]:
            raise ValueError("episode_starts must have shape [time, batch]")
        hidden = h_in
        cell = c_in
        outputs: list[torch.Tensor] = []
        for step in range(observations.shape[0]):
            carry = (~episode_starts[step].bool()).to(observations.dtype)
            carry = carry.view(1, -1, 1)
            hidden = hidden * carry
            cell = cell * carry
            actions, hidden, cell = self(observations[step], hidden, cell)
            outputs.append(actions)
        return torch.stack(outputs), hidden, cell

    def action_std(self) -> torch.Tensor:
        return self.distribution.std_param.clamp_min(1e-6)


class BasketballCritic(nn.Module):
    def __init__(self, normalizer: ObservationNormalizer) -> None:
        super().__init__()
        self.normalizer = normalizer
        layers: list[nn.Module] = []
        previous = OBSERVATION_DIM
        for width in HIDDEN_DIMS:
            layers.extend((nn.Linear(previous, width), nn.ELU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            normalized = self.normalizer(observations)
        return self.mlp(normalized).squeeze(-1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_artifacts(
    checkpoint: str | Path,
    onnx_path: str | Path | None = None,
) -> dict[str, str]:
    checkpoint_path = Path(checkpoint)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != SOURCE_CHECKPOINT_SHA256:
        raise ValueError(
            f"unexpected basketball checkpoint SHA256: {checkpoint_hash}"
        )
    result = {"checkpoint_sha256": checkpoint_hash}
    candidate = Path(onnx_path) if onnx_path is not None else checkpoint_path.with_name(
        "policy.onnx"
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"required source ONNX is missing: {candidate}")
    onnx_hash = sha256_file(candidate)
    if onnx_hash != SOURCE_ONNX_SHA256:
        raise ValueError(f"unexpected basketball ONNX SHA256: {onnx_hash}")
    result["onnx_sha256"] = onnx_hash
    return result


def load_source_actor(
    checkpoint: str | Path,
    *,
    initial_std: float = 0.03,
    verify_hash: bool = True,
    allow_local: bool = False,
) -> tuple[BasketballActor, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    local_candidate = (
        allow_local and verify_hash
        and sha256_file(checkpoint_path) != SOURCE_CHECKPOINT_SHA256
    )
    if local_candidate:
        policy_path = checkpoint_path.with_name("policy.onnx")
        hashes = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_sha256": sha256_file(policy_path),
        }
    else:
        hashes = verify_source_artifacts(checkpoint_path) if verify_hash else {}
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError("basketball checkpoint is missing actor_state_dict")
    source_steps = 0
    if local_candidate:
        if payload.get("format") != LOCAL_CHECKPOINT_FORMAT:
            raise ValueError("unsupported local basketball checkpoint format")
        if (
            payload.get("normalization_frozen") is not True
            or payload.get("actor_observation_dim") != OBSERVATION_DIM
            or payload.get("critic_observation_dim") != OBSERVATION_DIM
        ):
            raise ValueError("invalid local basketball normalization/observation contract")
        source_steps = payload.get("total_steps", payload.get("local_steps"))
        if type(source_steps) is not int or source_steps < 0:
            raise ValueError("local basketball checkpoint requires nonnegative training steps")
    actor = BasketballActor(initial_std=initial_std)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    if any(not torch.isfinite(value).all() for value in actor.state_dict().values()):
        raise ValueError("basketball actor contains non-finite parameters")
    parity = None
    if local_candidate:
        parity = compare_actor_to_onnx(actor, policy_path)
        error = max(float(parity["max_abs_error"]), float(parity["max_state_abs_error"]))
        if not np.isfinite(error) or error > 3e-5:
            raise ValueError(f"local basketball checkpoint/ONNX parity failed: {error}")
        if hashes != {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_sha256": sha256_file(policy_path),
        }:
            raise RuntimeError("local basketball source changed during validation")
    with torch.no_grad():
        actor.distribution.std_param.fill_(float(initial_std))
    actor.obs_normalizer.requires_grad_(False)
    metadata = {
        "source_iteration": int(payload.get("iter", -1)),
        "source_hashes": hashes,
        "source_actor_keys": sorted(payload["actor_state_dict"]),
        "source_kind": "local_actor_warm_start" if local_candidate else "released_actor",
        "source_training_steps": source_steps,
        "source_onnx_parity": parity,
    }
    return actor, metadata


def export_recurrent_onnx(
    actor: BasketballActor,
    output: str | Path,
) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    actor = actor.cpu().eval()
    h_in, c_in = actor.initial_state(1, device="cpu", dtype=torch.float32)
    torch.onnx.export(
        actor,
        (torch.zeros(1, OBSERVATION_DIM), h_in, c_in),
        target,
        input_names=["obs", "h_in", "c_in"],
        output_names=["actions", "h_out", "c_out"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    return target


def compare_actor_to_onnx(
    actor: BasketballActor,
    onnx_path: str | Path,
    *,
    steps: int = 40,
    reset_at: int = 20,
    seed: int = 42,
) -> dict[str, float | int]:
    import onnxruntime as ort

    if steps < 1 or not 0 <= reset_at < steps:
        raise ValueError("reset_at must identify a step in the comparison")
    generator = torch.Generator().manual_seed(seed)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if [item.name for item in session.get_inputs()] != ["obs", "h_in", "c_in"]:
        raise ValueError("unexpected recurrent ONNX input contract")
    if [item.name for item in session.get_outputs()] != [
        "actions",
        "h_out",
        "c_out",
    ]:
        raise ValueError("unexpected recurrent ONNX output contract")
    actor = actor.cpu().eval()
    hidden, cell = actor.initial_state(1, device="cpu", dtype=torch.float32)
    onnx_hidden = hidden.numpy()
    onnx_cell = cell.numpy()
    max_error = 0.0
    max_state_error = 0.0
    with torch.inference_mode():
        for step in range(steps):
            if step == reset_at:
                hidden.zero_()
                cell.zero_()
                onnx_hidden.fill(0.0)
                onnx_cell.fill(0.0)
            observation = torch.randn(
                1,
                OBSERVATION_DIM,
                generator=generator,
            ) * 0.2
            observation[:, -10:] = 0.0
            expected, hidden, cell = actor(observation, hidden, cell)
            actual, onnx_hidden, onnx_cell = session.run(
                None,
                {
                    "obs": observation.numpy(),
                    "h_in": onnx_hidden,
                    "c_in": onnx_cell,
                },
            )
            for index, (torch_value, onnx_value) in enumerate((
                (expected, actual), (hidden, onnx_hidden), (cell, onnx_cell),
            )):
                expected_array = torch_value.numpy()
                if not np.isfinite(expected_array).all() or not np.isfinite(onnx_value).all():
                    raise ValueError("non-finite recurrent ONNX parity output")
                error = float(np.max(np.abs(onnx_value - expected_array)))
                if index == 0:
                    max_error = max(max_error, error)
                else:
                    max_state_error = max(max_state_error, error)
    return {
        "steps": steps, "reset_at": reset_at, "max_abs_error": max_error,
        "max_state_abs_error": max_state_error,
    }
