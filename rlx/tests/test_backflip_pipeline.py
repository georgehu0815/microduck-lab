import numpy as np
import json
import pytest

from rlx.environments.backflip import BackflipEnv, stand_action, protocol_metadata, landing_reward_terms
from rlx.environments.backflip_evaluation import BackflipEvaluation
from rlx.environments.microduck_recipes import make_single_recipe_env


def environment(mode="showcase"):
    return BackflipEnv(backflip_mode=mode, actuator="xml", max_episode_s=12,
                       domain_rand=False, obs_noise=False, action_delay=False, random_yaw=False)


def test_landing_rewards_are_bounded_and_prefer_precision():
    def reward(offset=0.0, gyro=0.0, gravity=-1.0):
        return landing_reward_terms(np.array([0., 0., gravity]), np.full(14, offset),
                                    np.full(3, gyro), np.zeros(14), np.zeros(14), np.zeros(14))

    perfect, imprecise, inverted = reward(), reward(0.2), reward(gravity=1.)
    assert perfect["landing_pose"] > imprecise["landing_pose"] > inverted["landing_pose"]
    assert perfect["landing_settle"] > reward(gyro=2.)["landing_settle"]
    rng = np.random.default_rng(7)
    for _ in range(100):
        terms = landing_reward_terms(rng.normal(size=3), rng.normal(size=14), rng.normal(size=3),
                                     rng.normal(size=14) * 100, rng.normal(size=14) * 10, rng.normal(size=14))
        assert all(-1 <= value <= 0 if key.endswith("penalty") else 0 <= value <= 1
                   for key, value in terms.items())


def test_launch_reward_does_not_credit_assistance():
    env = environment()
    try:
        env.reset(seed=7)
        for _ in range(10):
            _, reward, _, _, _ = env.step(np.zeros(14, dtype=np.float32))
            assert reward == 0.0
    finally:
        env.close()


def test_handoff_requires_policy_rotation_and_stability():
    env = environment()
    try:
        env.reset(seed=7)
        env.landing_stable_steps = 10
        env._bf_rot = 5.2
        assert not env._handoff_ready()
        env._bf_rot = 5.8
        env.landing_stable_steps = 9
        assert not env._handoff_ready()
        env.landing_stable_steps = 10
        assert env._handoff_ready()
    finally:
        env.close()


def test_exact_initializer_exports_bound_controller_metadata(tmp_path):
    import onnx
    from rlx.models.backflip_bootstrap import initialize_landing
    from test_microduck_studio import load_studio_example

    checkpoint = initialize_landing(tmp_path / "backflip.safetensors", calibration_episodes=2, critic_epochs=2)
    parity = json.loads((tmp_path / "transfer-parity.json").read_text())
    assert parity["max_absolute_action_error"] < 1e-4
    module = load_studio_example()
    module._validate_backflip_source(checkpoint, "checkpoint")
    module._validate_backflip_source(checkpoint.with_suffix(".onnx"), "policy")
    exported = onnx.load(checkpoint.with_suffix(".onnx"))
    properties = {item.key: item.value for item in exported.metadata_props}
    assert len(properties["rlx_checkpoint_sha256"]) == 64
    sidecar = checkpoint.with_suffix(".safetensors.json")
    data = json.loads(sidecar.read_text())
    assert data["metadata"]["reward_calibration"]["samples"] == 1200
    assert data["metadata"]["reward_calibration"]["standard_deviation"] > 10
    data["metadata"]["stand_policy_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="stand_policy_sha256"):
        module._validate_backflip_source(checkpoint, "checkpoint")


def test_landing_training_never_uses_spotter_or_handoff():
    env = environment("landing")
    try:
        observation, _ = env.reset(seed=7)
        assert observation.shape == (61,)
        assert env._bf_rot >= 4.6
        assert env.stage == "landing"
        for _ in range(120):
            observation, _, _, _, _ = env.step(stand_action(observation))
            assert not env.spotter_active
            assert not np.any(env.data.qfrc_applied)
            assert env.stage == "landing"
        assert env.landing_policy_steps == 120
    finally:
        env.close()


def test_assisted_teacher_rehearsal_has_all_three_phases():
    env = environment()
    assessor = BackflipEvaluation(1, 600)
    try:
        observation, _ = env.reset(seed=7)
        assert env.stage == "launch"
        stages = set()
        for _ in range(600):
            observation, _, terminated, truncated, _ = env.step(stand_action(observation))
            stages.add(env.stage)
            assessor.observe(0, env.recipe_metrics(), terminated, truncated)
        report = assessor.report()
        assert stages == {"launch", "landing", "stand"}
        assert report["passed"], report
        assert report["episodes"][0]["assist_after_release_steps"] == 0
        env.reset(seed=8)
        assert env.stage == "launch"
        assert env.landing_policy_steps == 0
    finally:
        env.close()


def test_zero_landing_is_not_credited_with_teacher_success():
    env = environment()
    assessor = BackflipEvaluation(1, 600)
    try:
        env.reset(seed=7)
        for _ in range(600):
            _, _, terminated, truncated, _ = env.step(np.zeros(14, dtype=np.float32))
            assessor.observe(0, env.recipe_metrics(), terminated, truncated)
        assert not assessor.report()["passed"]
    finally:
        env.close()


def test_registry_factory_reports_provenance_and_metrics():
    env = make_single_recipe_env("backflip", domain_rand=False, obs_noise=False, action_delay=False, random_yaw=False)
    try:
        observation, _ = env.reset(seed=7)
        observation, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
        assert observation.shape == (61,)
        assert "rotation_rad" in info["recipe_metrics"]
        assert len(protocol_metadata()["stand_policy_sha256"]) == 64
    finally:
        env.close()
