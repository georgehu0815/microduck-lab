"""Pure per-episode checks for the assisted backflip showcase."""

import json
import math

import pytest

from rlx.environments.backflip_evaluation import (
    BackflipEvaluation,
    BackflipEvaluationCriteria,
    REQUIRED_METRICS,
)


def metrics_for_step(step: int) -> dict[str, float]:
    launch_complete = step >= 40
    handed_off = step >= 60
    landing_steps = min(max(step - 40, 0), 20)
    return {
        "rotation_rad": min(6.0, step * 0.15),
        "spotter_active": float(step < 40),
        "assist_torque_norm": 1.0 if step < 40 else 0.0,
        "launch_complete": float(launch_complete),
        "landing_policy_steps": float(landing_steps),
        "handed_off": float(handed_off),
        "upright": 1.0,
        "height_m": 0.12,
        "both_feet": 1.0,
        "nonfoot_contact": 0.0,
        "gyro_norm": 0.1,
    }


def observe_episode(
    evaluation: BackflipEvaluation,
    *,
    env_index: int = 0,
    steps: int = 300,
    terminated: bool = False,
    truncated: bool = True,
    mutate=None,
) -> None:
    for step in range(steps):
        metrics = metrics_for_step(step)
        if mutate is not None:
            mutate(step, metrics)
        last = step == steps - 1
        evaluation.observe(
            env_index,
            metrics,
            terminated=last and terminated,
            truncated=last and truncated,
        )


def assert_finite_json(report: dict) -> None:
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_fixed_criteria_describe_assisted_showcase_not_unassisted_skill():
    criteria = BackflipEvaluation(1, 300).criteria.to_dict()

    assert criteria == {
        "version": 2,
        "required_steps": 300,
        "control_hz": 50,
        "min_rotation_rad": 5.8,
        "max_rotation_rad": 2.0 * math.pi,
        "min_landing_policy_steps": 10,
        "min_pre_handoff_stable_steps": 10,
        "min_stand_hold_steps": 100,
        "min_stand_upright": 0.9,
        "min_stand_height_m": 0.105,
        "max_stand_gyro_norm": math.sqrt(2.0),
        "requires_launch_assistance": True,
        "credits_unassisted_skill": False,
        "all_episodes_required": True,
    }


@pytest.mark.parametrize(
    "num_envs,required_steps",
    [(0, 300), (True, 300), (1.5, 300), (1, 299), (1, 300.0)],
)
def test_invalid_dimensions_and_short_skill_horizons_are_rejected(
    num_envs, required_steps
):
    with pytest.raises(ValueError):
        BackflipEvaluation(num_envs, required_steps)


def test_complete_assisted_showcase_passes():
    evaluation = BackflipEvaluation(1, 300)
    observe_episode(evaluation)

    report = evaluation.report()

    assert report["passed"] is True
    assert report["completed_episodes"] == report["passed_episodes"] == 1
    episode = report["episodes"][0]
    assert episode["rotation_rad"] == pytest.approx(6.0)
    assert episode["landing_policy_steps"] == 20
    assert episode["pre_handoff_stable_steps"] == 19
    assert episode["rotation_at_handoff"] == pytest.approx(6.0)
    assert episode["stand_hold_seconds"] == pytest.approx(4.8)
    assert episode["min_stand_height_m"] == pytest.approx(0.12)
    assert episode["stand_upright_fraction"] == 1.0
    assert episode["assist_after_release_steps"] == 0
    assert episode["failures"] == []
    assert episode["passed"] is episode["complete"] is True
    assert "does not establish" in report["credit_scope"]
    assert "simulator" in report["credit_scope"]
    assert "PPO improvement" in report["credit_scope"]
    assert_finite_json(report)


def test_environment_timing_passes_with_release_at_one_and_handoff_at_ten():
    evaluation = BackflipEvaluation(1, 300)

    def exact_environment_timing(step, metrics):
        metrics["landing_policy_steps"] = float(
            min(max(step - 39, 0), 10)
        )
        metrics["handed_off"] = float(step >= 50)
        metrics["upright"] = 0.9
        metrics["height_m"] = 0.105
        metrics["gyro_norm"] = math.nextafter(math.sqrt(2.0), 0.0)

    observe_episode(evaluation, mutate=exact_environment_timing)
    episode = evaluation.report()["episodes"][0]

    assert episode["landing_policy_steps"] == 10
    assert episode["pre_handoff_stable_steps"] == 10
    assert episode["passed"] is True


@pytest.mark.parametrize("rotation_before_handoff", [5.2, 5.799, 5.8])
def test_rotation_must_reach_target_before_teacher_takes_control(
    rotation_before_handoff,
):
    evaluation = BackflipEvaluation(1, 300)

    def teacher_finishes_rotation(step, metrics):
        if step < 60:
            metrics["rotation_rad"] = min(
                metrics["rotation_rad"], rotation_before_handoff
            )

    observe_episode(evaluation, mutate=teacher_finishes_rotation)
    episode = evaluation.report()["episodes"][0]

    assert episode["rotation_rad"] == 6.0
    assert episode["rotation_at_handoff"] == rotation_before_handoff
    assert episode["passed"] is (rotation_before_handoff >= 5.8)
    if rotation_before_handoff < 5.8:
        assert "backward rotation below target before handoff" in episode["failures"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("upright", 0.899),
        ("height_m", 0.1049),
        ("both_feet", 0.0),
        ("nonfoot_contact", 1.0),
        ("gyro_norm", math.sqrt(2.0)),
        ("spotter_active", 1.0),
        ("assist_torque_norm", 0.001),
        ("landing_policy_steps", 9.0),
        ("upright", float("nan")),
        ("both_feet", 0.5),
        ("gyro_norm", -1.0),
    ],
)
def test_pre_handoff_stability_is_consecutive_and_cannot_be_repaired_by_teacher(
    key, value,
):
    evaluation = BackflipEvaluation(1, 300)

    def interrupt_landing(step, metrics):
        if step == 50:
            metrics[key] = value

    observe_episode(evaluation, mutate=interrupt_landing)
    episode = evaluation.report()["episodes"][0]

    assert episode["pre_handoff_stable_steps"] <= 9
    assert episode["stand_hold_seconds"] == pytest.approx(4.8)
    assert "insufficient consecutive stable landing-policy steps before handoff" in (
        episode["failures"]
    )
    assert episode["passed"] is False
    assert_finite_json(evaluation.report())


def test_single_stable_landing_sample_cannot_credit_handoff():
    evaluation = BackflipEvaluation(1, 300)

    def unstable_landing(step, metrics):
        if 40 <= step < 59:
            metrics["upright"] = 0.8

    observe_episode(evaluation, mutate=unstable_landing)
    episode = evaluation.report()["episodes"][0]

    assert episode["pre_handoff_stable_steps"] == 1
    assert episode["passed"] is False


def test_ten_stable_samples_after_landing_recovery_pass():
    evaluation = BackflipEvaluation(1, 300)

    def recover_landing(step, metrics):
        if step == 49:
            metrics["nonfoot_contact"] = 1.0

    observe_episode(evaluation, mutate=recover_landing)
    episode = evaluation.report()["episodes"][0]

    assert episode["pre_handoff_stable_steps"] == 10
    assert episode["passed"] is True


def test_first_handoff_evidence_is_not_replaced_by_a_later_handoff():
    evaluation = BackflipEvaluation(1, 300)

    def repeated_handoff(step, metrics):
        if step == 45 or step >= 60:
            metrics["handed_off"] = 1.0

    observe_episode(evaluation, mutate=repeated_handoff)
    episode = evaluation.report()["episodes"][0]

    assert episode["pre_handoff_stable_steps"] == 4
    assert episode["passed"] is False


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda step, metrics: metrics.update(
                spotter_active=0.0, assist_torque_norm=0.0
            ),
            "launch assistance was not observed",
        ),
        (
            lambda step, metrics: metrics.update(launch_complete=0.0),
            "assisted launch release was not recorded",
        ),
        (
            lambda step, metrics: metrics.update(handed_off=0.0),
            "landing-to-stand handoff was not recorded",
        ),
    ],
)
def test_missing_phases_fail_explicitly(mutate, expected):
    evaluation = BackflipEvaluation(1, 300)
    observe_episode(evaluation, mutate=mutate)

    assert expected in evaluation.report()["failures"]


def test_lingering_assistance_after_release_fails():
    evaluation = BackflipEvaluation(1, 300)

    def linger(step, metrics):
        if step == 45:
            metrics["spotter_active"] = 1.0
            metrics["assist_torque_norm"] = 0.2

    observe_episode(evaluation, mutate=linger)
    episode = evaluation.report()["episodes"][0]

    assert episode["assist_after_release_steps"] == 1
    assert "spotter or assist torque remained active after release" in episode[
        "failures"
    ]


def test_requires_ten_unassisted_landing_steps_before_handoff():
    evaluation = BackflipEvaluation(1, 300)

    def early_handoff(step, metrics):
        metrics["handed_off"] = float(step >= 49)

    observe_episode(evaluation, mutate=early_handoff)
    episode = evaluation.report()["episodes"][0]

    assert episode["landing_policy_steps"] == 20
    assert "insufficient unassisted landing-policy steps before handoff" in episode[
        "failures"
    ]


def test_landing_counter_must_reset_at_release():
    evaluation = BackflipEvaluation(1, 300)

    def preload_counter(step, metrics):
        metrics["landing_policy_steps"] = float(100 + min(max(step - 40, 0), 9))

    observe_episode(evaluation, mutate=preload_counter)
    episode = evaluation.report()["episodes"][0]

    assert episode["landing_policy_steps"] == 109
    assert "landing_policy_steps did not reset at release" in episode["failures"]


@pytest.mark.parametrize(
    "steps,terminated,truncated",
    [(299, False, True), (300, False, False), (300, True, False)],
)
def test_partial_or_prematurely_terminated_horizon_fails(
    steps, terminated, truncated
):
    evaluation = BackflipEvaluation(1, 300)
    observe_episode(
        evaluation,
        steps=steps,
        terminated=terminated,
        truncated=truncated,
    )

    episode = evaluation.report()["episodes"][0]
    assert episode["complete"] is False
    assert "episode did not complete the required 6-second horizon" in episode[
        "failures"
    ]
    assert (
        "episode terminated before showcase completion" in episode["failures"]
    ) is terminated


@pytest.mark.parametrize(
    "key,value",
    [
        ("upright", 0.89),
        ("height_m", 0.104),
        ("both_feet", 0.0),
        ("nonfoot_contact", 1.0),
        ("gyro_norm", math.sqrt(2.0)),
    ],
)
def test_unstable_final_stand_fails(key, value):
    evaluation = BackflipEvaluation(1, 300)

    def destabilize(step, metrics):
        if step == 250:
            metrics[key] = value

    observe_episode(evaluation, mutate=destabilize)
    episode = evaluation.report()["episodes"][0]

    assert episode["stand_hold_seconds"] == pytest.approx(49 / 50)
    assert "final stable stand hold below target" in episode["failures"]


def test_no_rotation_fails():
    evaluation = BackflipEvaluation(1, 300)
    observe_episode(
        evaluation,
        mutate=lambda step, metrics: metrics.update(rotation_rad=0.0),
    )

    assert "backward rotation below target" in evaluation.report()["failures"]


@pytest.mark.parametrize("key", REQUIRED_METRICS)
@pytest.mark.parametrize("invalid", ["missing", float("nan"), float("inf")])
def test_missing_or_nonfinite_metrics_fail_closed(key, invalid):
    evaluation = BackflipEvaluation(1, 300)

    def corrupt(step, metrics):
        if step == 17:
            if invalid == "missing":
                del metrics[key]
            else:
                metrics[key] = invalid

    observe_episode(evaluation, mutate=corrupt)
    report = evaluation.report()

    assert report["passed"] is False
    assert report["episodes"][0]["measured_steps"] == 299
    assert "missing, malformed, or non-finite backflip metrics" in report["failures"]
    assert "incomplete backflip metric coverage" in report["failures"]
    assert_finite_json(report)


def test_vector_lanes_and_reset_episodes_are_never_pooled():
    evaluation = BackflipEvaluation(2, 300)
    observe_episode(evaluation, env_index=0)
    observe_episode(
        evaluation,
        env_index=1,
        mutate=lambda step, metrics: metrics.update(rotation_rad=0.0),
    )
    observe_episode(
        evaluation,
        env_index=0,
        steps=20,
        truncated=False,
    )

    report = evaluation.report()

    assert report["passed"] is False
    assert report["completed_episodes"] == 2
    assert report["incomplete_episodes"] == 1
    assert report["passed_episodes"] == 1
    assert [
        (episode["env_index"], episode["episode_index"])
        for episode in report["episodes"]
    ] == [(0, 0), (1, 0), (0, 1)]
    assert [episode["pre_handoff_stable_steps"] for episode in report["episodes"]] == [
        19, 19, 0
    ]
    assert [episode["rotation_at_handoff"] for episode in report["episodes"]] == [
        6.0, 0.0, None
    ]


def test_empty_or_missing_lane_evidence_cannot_pass():
    empty = BackflipEvaluation(1, 300).report()
    assert empty["episodes"] == []
    assert empty["failures"] == ["missing evaluation lane coverage"]

    evaluation = BackflipEvaluation(2, 300)
    observe_episode(evaluation, env_index=1)
    report = evaluation.report()
    assert report["passed"] is False
    assert report["passed_episodes"] == 1
    assert report["failures"] == ["missing evaluation lane coverage"]
    assert_finite_json(report)


def test_invalid_cumulative_or_binary_metrics_fail_explicitly():
    evaluation = BackflipEvaluation(1, 300)

    def corrupt(step, metrics):
        if step == 17:
            metrics["both_feet"] = 0.5
        if step == 80:
            metrics["rotation_rad"] = 7.0

    observe_episode(evaluation, mutate=corrupt)
    report = evaluation.report()

    assert "invalid binary backflip metrics" in report["failures"]
    assert "invalid cumulative or physical backflip metrics" in report["failures"]
    assert_finite_json(report)


def test_public_criteria_type_serializes_directly():
    assert BackflipEvaluationCriteria(required_steps=350).to_dict()[
        "required_steps"
    ] == 350
