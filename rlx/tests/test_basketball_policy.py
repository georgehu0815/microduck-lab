"""Contracts for local recurrent basketball policy loading and PPO."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from rlx.models.basketball import (
    ACTION_DIM,
    HIDDEN_SIZE,
    NORMALIZER_EPSILON,
    OBSERVATION_DIM,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_ONNX_SHA256,
    BasketballActor,
    BasketballCritic,
    compare_actor_to_onnx,
    export_recurrent_onnx,
    load_source_actor,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "microduck-playground" / "artifacts" / "basketball"
SOURCE_CHECKPOINT = SOURCE_DIR / "checkpoint.pt"
SOURCE_ONNX = SOURCE_DIR / "policy.onnx"


@pytest.fixture(scope="module")
def basketball_script():
    path = ROOT / "rlx" / "examples" / "ppo_microduck_basketball.py"
    spec = importlib.util.spec_from_file_location("ppo_microduck_basketball_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_actor():
    actor, metadata = load_source_actor(SOURCE_CHECKPOINT)
    return actor, metadata


def test_source_hashes_and_checkpoint_architecture(source_actor):
    actor, metadata = source_actor
    assert sha256_file(SOURCE_CHECKPOINT) == SOURCE_CHECKPOINT_SHA256
    assert sha256_file(SOURCE_ONNX) == SOURCE_ONNX_SHA256
    assert metadata["source_iteration"] == 6999
    assert metadata["source_hashes"] == {
        "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "onnx_sha256": SOURCE_ONNX_SHA256,
    }
    assert actor.rnn.rnn.input_size == OBSERVATION_DIM
    assert actor.rnn.rnn.hidden_size == HIDDEN_SIZE
    assert actor.rnn.rnn.num_layers == 1
    assert actor.mlp[0].weight.shape == (512, 256)
    assert actor.mlp[2].weight.shape == (256, 512)
    assert actor.mlp[4].weight.shape == (128, 256)
    assert actor.mlp[6].weight.shape == (ACTION_DIM, 128)
    torch.testing.assert_close(
        actor.distribution.std_param,
        torch.full((ACTION_DIM,), 0.03),
    )
    assert not any(
        parameter.requires_grad for parameter in actor.obs_normalizer.parameters()
    )


def test_source_normalization_matches_export_divisor(source_actor):
    actor, _ = source_actor
    observations = torch.linspace(-0.4, 0.4, OBSERVATION_DIM).reshape(1, -1)
    expected = (
        observations - actor.obs_normalizer._mean
    ) / (actor.obs_normalizer._std + NORMALIZER_EPSILON)
    torch.testing.assert_close(actor.obs_normalizer(observations), expected)
    assert torch.count_nonzero(actor.obs_normalizer._std[:, 51:61]) == 0


def test_source_onnx_parity_with_carried_state_and_reset(source_actor):
    actor, _ = source_actor
    parity = compare_actor_to_onnx(actor, SOURCE_ONNX, steps=40, reset_at=20)
    assert parity["steps"] == 40
    assert parity["reset_at"] == 20
    assert parity["max_abs_error"] < 3e-6


def test_sequence_masks_reset_hidden_and_cell(source_actor):
    actor, _ = source_actor
    generator = torch.Generator().manual_seed(7)
    observations = torch.randn(
        4,
        2,
        OBSERVATION_DIM,
        generator=generator,
    )
    observations[:, :, 51:61] = 0
    hidden = torch.randn(1, 2, HIDDEN_SIZE, generator=generator)
    cell = torch.randn(1, 2, HIDDEN_SIZE, generator=generator)
    starts = torch.tensor(
        [[True, False], [False, False], [True, False], [False, False]]
    )
    sequence, _, _ = actor.forward_sequence(
        observations,
        hidden,
        cell,
        starts,
    )
    zero_hidden, zero_cell = actor.initial_state(1)
    first_reset, _, _ = actor(
        observations[0, :1],
        zero_hidden,
        zero_cell,
    )
    third_reset, _, _ = actor(
        observations[2, :1],
        zero_hidden,
        zero_cell,
    )
    torch.testing.assert_close(sequence[0, :1], first_reset)
    torch.testing.assert_close(sequence[2, :1], third_reset)
    carried, _, _ = actor.forward_sequence(
        observations[2:3, :1],
        torch.ones(1, 1, HIDDEN_SIZE),
        torch.ones(1, 1, HIDDEN_SIZE),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    assert not torch.allclose(carried[0], third_reset)


def test_exported_onnx_has_recurrent_contract_and_parity(source_actor, tmp_path):
    actor, _ = source_actor
    output = export_recurrent_onnx(actor, tmp_path / "policy.onnx")
    import onnxruntime as ort

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    assert [(item.name, item.shape) for item in session.get_inputs()] == [
        ("obs", [1, OBSERVATION_DIM]),
        ("h_in", [1, 1, HIDDEN_SIZE]),
        ("c_in", [1, 1, HIDDEN_SIZE]),
    ]
    assert [(item.name, item.shape) for item in session.get_outputs()] == [
        ("actions", [1, ACTION_DIM]),
        ("h_out", [1, 1, HIDDEN_SIZE]),
        ("c_out", [1, 1, HIDDEN_SIZE]),
    ]
    assert compare_actor_to_onnx(actor, output)["max_abs_error"] < 3e-5


def test_gae_bootstraps_truncation_but_not_termination(basketball_script):
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    bootstraps = torch.tensor([[10.0, 10.0], [3.0, 3.0]])
    terminated = torch.tensor([[False, True], [False, False]])
    truncated = torch.tensor([[True, False], [False, False]])
    advantages, returns = basketball_script.compute_gae(
        rewards,
        values,
        bootstraps,
        terminated,
        truncated,
        gamma=0.9,
        gae_lambda=0.8,
    )
    assert returns[0, 0] == pytest.approx(10.0)
    assert returns[0, 1] == pytest.approx(1.0)
    assert advantages[1, 0] == pytest.approx(4.2)
    assert advantages[1, 1] == pytest.approx(4.2)


class _TimeoutEnv:
    def __init__(self, *, terminate: bool) -> None:
        self.terminate = terminate
        self.steps = 0
        self.resets = 0

    def reset(self, *, seed):
        self.resets += 1
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[0] = self.resets
        return observation, {"seed": seed}

    def step(self, action):
        assert np.asarray(action).shape == (ACTION_DIM,)
        self.steps += 1
        observation = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        observation[0] = 7.0
        return (
            observation,
            1.0,
            self.terminate,
            not self.terminate,
            {"termination_reasons": ["test"] if self.terminate else []},
        )

    def close(self):
        pass


class _FirstValueCritic(torch.nn.Module):
    def forward(self, observations):
        return observations[:, 0]


def test_rollout_stores_timeout_final_value_and_episode_masks(
    basketball_script,
    source_actor,
):
    actor, _ = source_actor
    environments = [_TimeoutEnv(terminate=False), _TimeoutEnv(terminate=True)]
    observations = np.stack(
        [environment.reset(seed=index)[0] for index, environment in enumerate(environments)]
    )
    hidden, cell = actor.initial_state(2)
    rollout, _, _, _, starts = basketball_script.collect_rollout(
        actor,
        _FirstValueCritic(),
        environments,
        observations,
        hidden,
        cell,
        torch.ones(2, dtype=torch.bool),
        steps=2,
        generator=torch.Generator().manual_seed(1),
        seed=10,
        reset_counts=np.zeros(2, dtype=np.int64),
    )
    assert rollout.bootstrap_values[0, 0] == 7.0
    assert rollout.bootstrap_values[0, 1] == 0.0
    assert rollout.episode_starts[0].tolist() == [True, True]
    assert rollout.episode_starts[1].tolist() == [True, True]
    assert starts.tolist() == [True, True]
    assert environments[0].resets == 3
    assert environments[1].resets == 3


def test_ppo_update_recomputes_full_recurrent_sequence(basketball_script):
    torch.manual_seed(3)
    actor = BasketballActor(initial_std=0.03)
    critic = BasketballCritic(actor.obs_normalizer)
    time_steps = 3
    num_envs = 2
    observations = torch.randn(time_steps, num_envs, OBSERVATION_DIM)
    observations[:, :, 51:61] = 0
    hidden, cell = actor.initial_state(num_envs)
    starts = torch.tensor([[True, True], [False, False], [True, False]])
    with torch.no_grad():
        means, _, _ = actor.forward_sequence(
            observations,
            hidden,
            cell,
            starts,
        )
        actions = means + 0.01
        log_probs = basketball_script.gaussian_log_prob(
            actions,
            means,
            actor.action_std(),
        )
        values = critic(observations.reshape(-1, OBSERVATION_DIM)).reshape(
            time_steps,
            num_envs,
        )
    rollout = basketball_script.Rollout(
        observations=observations,
        actions=actions,
        old_log_probs=log_probs,
        old_values=values,
        rewards=torch.ones(time_steps, num_envs),
        terminated=torch.zeros(time_steps, num_envs, dtype=torch.bool),
        truncated=torch.zeros(time_steps, num_envs, dtype=torch.bool),
        bootstrap_values=torch.ones(time_steps, num_envs),
        episode_starts=starts,
        initial_hidden=hidden,
        initial_cell=cell,
        next_hidden=hidden,
        next_cell=cell,
        next_episode_starts=torch.zeros(num_envs, dtype=torch.bool),
        infos=[],
    )
    calls = 0
    original = actor.forward_sequence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    actor.forward_sequence = counted
    normalizer = {
        name: value.clone() for name, value in actor.obs_normalizer.state_dict().items()
    }
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.mlp.parameters()),
        lr=2e-5,
    )
    metrics = basketball_script.ppo_update(
        actor,
        critic,
        optimizer,
        rollout,
        epochs=2,
        target_kl=1e6,
    )
    assert calls == 4
    assert all(np.isfinite(value) for value in metrics.values())
    for name, value in actor.obs_normalizer.state_dict().items():
        torch.testing.assert_close(value, normalizer[name])


def test_ppo_kl_guard_rejects_stale_distribution_before_update(basketball_script):
    actor = BasketballActor()
    critic = BasketballCritic(actor.obs_normalizer)
    environments = [_TimeoutEnv(terminate=True), _TimeoutEnv(terminate=False)]
    observations = np.stack([environment.reset(seed=0)[0] for environment in environments])
    hidden, cell = actor.initial_state(2)
    rollout, *_ = basketball_script.collect_rollout(
        actor, critic, environments, observations, hidden, cell,
        torch.ones(2, dtype=torch.bool), steps=2,
        generator=torch.Generator().manual_seed(1), seed=0,
        reset_counts=np.zeros(2, dtype=np.int64),
    )
    rollout.old_log_probs = rollout.old_log_probs + 10
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.mlp.parameters()))
    metrics = basketball_script.ppo_update(actor, critic, optimizer, rollout, target_kl=.02)
    assert metrics["kl_early_stop"] == 1
    assert metrics["epochs_completed"] == 0
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_source_requires_adjacent_verified_onnx(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.symlink_to(SOURCE_CHECKPOINT)
    with pytest.raises(FileNotFoundError, match="source ONNX"):
        load_source_actor(checkpoint)


def test_cli_can_select_randomized_commands(basketball_script):
    args = basketball_script.parse_args(["--randomized-commands"])
    assert args.command is None


def test_training_cannot_overwrite_existing_artifacts(basketball_script, tmp_path):
    existing = tmp_path / "policy.onnx"
    existing.write_bytes(b"preserve me")
    args = basketball_script.parse_args(["--output", str(tmp_path)])
    with pytest.raises(FileExistsError, match="overwrite"):
        basketball_script.train(args)
    assert existing.read_bytes() == b"preserve me"


def _factory_args(actuator: str) -> argparse.Namespace:
    return argparse.Namespace(
        actuator=actuator,
        command=(0.08, 0.0, 0.0),
        num_envs=3,
        hold=0.0,
        curriculum=False,
        seed=9,
        pushes=False,
    )


def test_environment_factory_does_not_share_bam_models(
    basketball_script,
    monkeypatch,
):
    import rlx.environments.basketball as environment_module

    calls: list[dict] = []

    class FakeEnvironment:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(environment_module, "BasketballEnv", FakeEnvironment)
    monkeypatch.setattr(
        environment_module,
        "basketball_model",
        lambda **kwargs: pytest.fail("BAM models must not be shared"),
    )
    environments = basketball_script.make_environments(_factory_args("bam"))
    assert len(environments) == 3
    assert all("model" not in kwargs for kwargs in calls)
    assert [kwargs["seed"] for kwargs in calls] == [9, 10, 11]
    assert all(kwargs["max_episode_s"] == 10.0 for kwargs in calls)
    assert all(kwargs["obs_noise"] is False for kwargs in calls)
    assert all(kwargs["domain_rand"] is False for kwargs in calls)


def test_environment_factory_shares_only_xml_model(
    basketball_script,
    monkeypatch,
):
    import rlx.environments.basketball as environment_module

    model = object()
    model_calls = 0
    calls: list[dict] = []

    def fake_model(**kwargs):
        nonlocal model_calls
        model_calls += 1
        assert kwargs == {"actuator": "xml"}
        return model

    class FakeEnvironment:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(environment_module, "BasketballEnv", FakeEnvironment)
    monkeypatch.setattr(environment_module, "basketball_model", fake_model)
    basketball_script.make_environments(_factory_args("xml"))
    assert model_calls == 1
    assert all(kwargs["model"] is model for kwargs in calls)


def test_cli_defaults_are_bounded_smoke_settings(basketball_script):
    args = basketball_script.parse_args([])
    assert args.checkpoint == SOURCE_CHECKPOINT
    assert args.num_envs == 4
    assert args.steps == 32
    assert args.updates == 5
    assert args.learning_rate == 2e-5
    assert args.initial_std == 0.03
    assert args.hold == 0.0
    assert args.curriculum is False
    assert args.actuator == "bam"
    assert args.command == (0.08, 0.0, 0.0)
    assert args.pushes is False


def test_saved_artifacts_report_hashes_steps_and_actor_change(
    basketball_script,
    tmp_path,
):
    actor, source_metadata = load_source_actor(SOURCE_CHECKPOINT)
    initial_state = {
        name: value.detach().clone() for name, value in actor.state_dict().items()
    }
    with torch.no_grad():
        actor.mlp[6].bias.add_(1e-4)
    critic = BasketballCritic(actor.obs_normalizer)
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.mlp.parameters()),
        lr=2e-5,
    )
    args = argparse.Namespace(
        output=tmp_path,
        checkpoint=SOURCE_CHECKPOINT,
        updates=5,
        num_envs=4,
        steps=32,
        learning_rate=2e-5,
        target_kl=0.02,
        initial_std=0.03,
        hold=0.0,
        curriculum=False,
        actuator="bam",
        seed=42,
        command=(0.08, 0.0, 0.0),
        pushes=False,
    )
    summary = basketball_script.save_results(
        tmp_path,
        actor,
        critic,
        optimizer,
        args=args,
        source_metadata=source_metadata,
        initial_actor_state=initial_state,
        local_steps=640,
        update_metrics=[{"update": 5, "loss": 0.1}],
    )
    assert summary["full_upstream_resume"] is False
    assert summary["local_steps"] == 640
    assert summary["source_checkpoint_sha256"] == SOURCE_CHECKPOINT_SHA256
    assert summary["source_onnx_sha256"] == SOURCE_ONNX_SHA256
    assert summary["actor_weight_change_l2"] > 0
    assert summary["checkpoint_sha256"] == sha256_file(tmp_path / "checkpoint.pt")
    assert summary["onnx_sha256"] == sha256_file(tmp_path / "policy.onnx")
    assert summary["onnx_parity"]["max_abs_error"] < 3e-5
    saved = torch.load(
        tmp_path / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert saved["format"] == "rlx.basketball.local_recurrent_ppo.v1"
    assert saved["local_steps"] == 640
    assert saved["critic_observation_dim"] == OBSERVATION_DIM
