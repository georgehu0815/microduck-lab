from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.locomotion import (
    DEFAULT_LEG_TARGETS,
    LEGACY_HEAD_INDICES,
    LEGACY_LEG_INDICES,
    POLICY_SHA256,
    SCREENED_STAND_CONFIGURATION,
    LegacyLegPolicyAdapter,
    policy_sha256,
)
from microduck_local import contract as legacy


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "microduck_local" / "policies"


@pytest.mark.parametrize("name", ["alpha_stand.onnx", "alpha_walking.onnx"])
def test_shipped_policy_hashes_and_contract_are_pinned(name):
    path = POLICIES / name
    adapter = LegacyLegPolicyAdapter(path)

    assert policy_sha256(path) == POLICY_SHA256[name]
    assert adapter.source_sha256 == POLICY_SHA256[name]
    assert adapter.session.get_inputs()[0].shape == [1, 61]
    assert adapter.session.get_outputs()[0].shape == [1, 14]


def test_observation_uses_real_leg_state_and_virtual_missing_head():
    env = IntegratedArmEnv(
        case="stowed_arm_walking", mode="free", candidate="A", max_steps=2
    )
    env.reset(seed=3)
    adapter = LegacyLegPolicyAdapter(
        POLICIES / "alpha_stand.onnx", twist_command=(0.1, -0.02, 0.03)
    )

    env.data.qpos[env.qpos_indices[:10]] += np.linspace(-0.02, 0.02, 10)
    env.data.qvel[env.dof_indices[:10]] = np.linspace(-0.4, 0.4, 10)
    observation = adapter.observation(env)

    expected_q = np.zeros(14, dtype=np.float32)
    expected_dq = np.zeros(14, dtype=np.float32)
    expected_q[LEGACY_LEG_INDICES] = (
        env.data.qpos[env.qpos_indices[:10]] - env.home[:10]
    )
    expected_dq[LEGACY_LEG_INDICES] = env.data.qvel[env.dof_indices[:10]]
    np.testing.assert_allclose(observation[6:20], expected_q, atol=1e-7)
    np.testing.assert_allclose(observation[20:34], expected_dq, atol=1e-7)
    np.testing.assert_array_equal(
        observation[6:20][LEGACY_HEAD_INDICES], np.zeros(4, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        observation[20:34][LEGACY_HEAD_INDICES], np.zeros(4, dtype=np.float32)
    )
    np.testing.assert_array_equal(observation[34:48], np.zeros(14, np.float32))
    np.testing.assert_allclose(observation[48:51], [0.1, -0.02, 0.03])
    assert observation[3:6] == pytest.approx([0.0, 0.0, -1.0])


def test_adapter_owns_full_action_history_and_returns_only_leg_targets():
    env = IntegratedArmEnv(
        case="stowed_arm_walking", mode="free", candidate="A", max_steps=2
    )
    env.reset(seed=0)
    adapter = LegacyLegPolicyAdapter(POLICIES / "alpha_stand.onnx")

    targets = adapter.leg_targets(env)
    first_action = adapter.last_action.copy()
    assert targets.shape == (10,)
    assert np.isfinite(targets).all()
    assert first_action.shape == (14,)
    np.testing.assert_array_equal(adapter.last_observation[34:48], np.zeros(14))

    adapter.leg_targets(env)
    np.testing.assert_array_equal(adapter.last_observation[34:48], first_action)
    adapter.reset()
    np.testing.assert_array_equal(adapter.last_action, np.zeros(14))
    np.testing.assert_array_equal(adapter.virtual_head_q, legacy.DEFAULT_POSE[5:9])


def test_screened_stand_factory_has_named_simulation_only_provenance():
    adapter = LegacyLegPolicyAdapter.screened_stand(POLICIES)

    assert adapter.action_gain == pytest.approx(1.25)
    assert adapter.configuration_name == SCREENED_STAND_CONFIGURATION["name"]
    assert adapter.provenance["source_policy_sha256"] == POLICY_SHA256[
        "alpha_stand.onnx"
    ]
    assert adapter.provenance["simulation_only"] is True
    assert adapter.provenance["hardware_ready"] is False
    assert "domain_shift" in adapter.provenance


def test_free_model_stands_with_manual_leg_replacement_and_normalization():
    env = IntegratedArmEnv(
        case="stowed_arm_walking", mode="free", candidate="A", max_steps=251
    )
    env.reset(seed=0)
    adapter = LegacyLegPolicyAdapter.screened_stand(POLICIES)

    max_tilt = 0.0
    min_height = np.inf
    for _ in range(250):
        target = env.home.copy()
        target[:10] = adapter.leg_targets(env)
        action = env.normalize_targets(target)
        _, _, terminated, truncated, info = env.step(action)
        diagnostics = info["diagnostics"]
        max_tilt = max(max_tilt, diagnostics["tilt_rad"])
        min_height = min(min_height, diagnostics["trunk_height_m"])
        assert not terminated
        assert not truncated

    assert info["failure_reason"] is None
    assert max_tilt < 0.10
    assert min_height > 0.11
    assert np.max(np.abs(target[:10] - DEFAULT_LEG_TARGETS)) > 0.01
