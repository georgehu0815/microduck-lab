"""Pure acceptance tests for unassisted suspended-bridge evaluation."""

import json

import pytest

from rlx.environments.bridge_evaluation import (
    BridgeEvaluation,
    REQUIRED_METRICS,
)


def metrics_for_step(step: int, crossing_step: int = 60) -> dict[str, float]:
    crossed = step >= crossing_step
    return {
        "bridge_crossed": float(crossed),
        "bridge_progress_m": min(1.65, step * 0.03),
        "bridge_assistance": 0.0,
        "robot_floor_contact": 0.0,
        "nonfoot_support": 0.0,
        "feet_support": 2.0 if step == 0 or crossed else 1.0,
        "bridge_contact_observed": float(step >= 20),
    }


def observe_episode(
    evaluation: BridgeEvaluation,
    *,
    env_index: int = 0,
    steps: int = 100,
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


def test_complete_unassisted_contact_validated_crossing_passes():
    evaluation = BridgeEvaluation(1, 100)
    observe_episode(evaluation)

    report = evaluation.report()

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["completed_episodes"] == report["passed_episodes"] == 1
    episode = report["episodes"][0]
    assert episode["crossed"] is True
    assert episode["crossing_step"] == 61
    assert episode["end_feet_at_crossing"] == 2
    assert episode["max_progress_m"] == pytest.approx(1.65)
    assert episode["assisted_steps"] == 0
    assert episode["floor_contact_steps"] == 0
    assert episode["nonfoot_support_steps"] == 0
    assert_finite_json(report)


@pytest.mark.parametrize(
    "mutation, failure",
    [
        (
            lambda step, metrics: metrics.update(bridge_assistance=0.1)
            if step == 20 else None,
            "bridge assistance was active during evaluation",
        ),
        (
            lambda step, metrics: metrics.update(robot_floor_contact=1.0)
            if step == 20 else None,
            "robot contacted the floor",
        ),
        (
            lambda step, metrics: metrics.update(nonfoot_support=1.0)
            if step == 20 else None,
            "non-foot robot support contact occurred",
        ),
        (
            lambda step, metrics: metrics.update(bridge_crossed=0.0),
            "contact-validated end-to-end crossing was not observed",
        ),
        (
            lambda step, metrics: metrics.update(feet_support=1.0)
            if step >= 60 else None,
            "crossing lacked two-foot landing support",
        ),
        (
            lambda step, metrics: metrics.update(
                bridge_contact_observed=0.0
            ),
            "suspended-plank foot contact was not observed",
        ),
    ],
)
def test_any_assistance_fall_or_dishonest_crossing_fails(mutation, failure):
    evaluation = BridgeEvaluation(1, 100)
    observe_episode(evaluation, mutate=mutation)

    report = evaluation.report()

    assert report["passed"] is False
    assert failure in report["failures"]
    assert failure in report["episodes"][0]["failures"]
    assert_finite_json(report)


def test_termination_and_partial_horizon_cannot_pass():
    terminated = BridgeEvaluation(1, 100)
    observe_episode(terminated, terminated=True, truncated=False)
    report = terminated.report()
    assert report["passed"] is False
    assert "episode terminated or fell before evaluation completion" in report["failures"]

    partial = BridgeEvaluation(1, 100)
    observe_episode(partial, steps=99, truncated=True)
    assert partial.report()["passed"] is False


def test_vector_lanes_and_episodes_are_not_pooled():
    evaluation = BridgeEvaluation(2, 100)
    observe_episode(evaluation, env_index=0)
    observe_episode(
        evaluation,
        env_index=1,
        mutate=lambda step, metrics: metrics.update(bridge_crossed=0.0),
    )
    report = evaluation.report()
    assert report["passed"] is False
    assert [episode["passed"] for episode in report["episodes"]] == [True, False]


@pytest.mark.parametrize("key", REQUIRED_METRICS)
@pytest.mark.parametrize("invalid", ["missing", float("nan"), float("inf")])
def test_missing_or_nonfinite_metrics_fail(key, invalid):
    evaluation = BridgeEvaluation(1, 1)
    metrics = metrics_for_step(100, crossing_step=0)
    if invalid == "missing":
        del metrics[key]
    else:
        metrics[key] = invalid
    evaluation.observe(0, metrics, truncated=True)
    report = evaluation.report()
    assert report["passed"] is False
    assert "missing, malformed, or non-finite bridge metrics" in report["failures"]
    assert_finite_json(report)


@pytest.mark.parametrize(
    "num_envs,required_steps",
    [(0, 100), (True, 100), (1.5, 100), (1, 0), (1, 100.0)],
)
def test_invalid_dimensions_are_rejected(num_envs, required_steps):
    with pytest.raises(ValueError):
        BridgeEvaluation(num_envs, required_steps)
