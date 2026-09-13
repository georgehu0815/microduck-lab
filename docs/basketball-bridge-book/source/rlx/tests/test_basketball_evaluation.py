"""Focused physical-metric tests for standalone basketball evaluation."""

from __future__ import annotations

import json

import pytest

from scripts.eval_basketball_local import (
    CTRL_DT,
    evaluate_trace,
    initial_heading_from_xmat,
    summarize_trials,
)


def basketball_samples(
    *,
    seconds: float = 4.0,
    velocity_mps: float = 0.08,
    world_direction: tuple[float, float] = (1.0, 0.0),
    hold: float = 0.0,
    body_ball_contact: bool = False,
    robot_floor_contact: bool = False,
) -> list[dict[str, object]]:
    position = [0.0, 0.0, 0.12]
    result = []
    for _ in range(round(seconds / CTRL_DT)):
        position[0] += world_direction[0] * velocity_mps * CTRL_DT
        position[1] += world_direction[1] * velocity_mps * CTRL_DT
        result.append(
            {
                "reward": 1_000_000.0,
                "tilt_deg": 4.0,
                "root_ball_offset_m": 0.02,
                "root_above_ball_m": 0.21,
                "ball_speed_mps": abs(velocity_mps),
                "ball_position": position.copy(),
                "ball_angular_speed_rad_s": abs(velocity_mps) / 0.12,
                "body_forward_mps": velocity_mps,
                "body_lateral_mps": 0.0,
                "body_yaw_rate_rad_s": 0.0,
                "foot_ball_contacts": 2,
                "body_ball_contact": body_ball_contact,
                "robot_floor_contact": robot_floor_contact,
                "hold": hold,
                "elapsed_s": len(result) * CTRL_DT,
                "termination_reasons": [],
            }
        )
    return result


def evaluate(
    samples,
    *,
    command=(0.08, 0.0, 0.0),
    seconds=4.0,
    initial_forward=(1.0, 0.0),
    seed=101,
    **kwargs,
):
    return evaluate_trace(
        samples,
        seed=seed,
        command=command,
        seconds=seconds,
        initial_forward_xy=initial_forward,
        initial_ball_position=(0.0, 0.0, 0.12),
        **kwargs,
    )


def test_commanded_success_requires_duration_progress_contact_and_no_exploit():
    record = evaluate(basketball_samples())

    assert record["duration_complete"] is True
    assert record["progress_complete"] is True
    assert record["contact_complete"] is True
    assert record["exploit_free"] is True
    assert record["controlled_rolling"] is True
    assert record["learned_success"] is True
    assert record["signed_commanded_progress_m"] == pytest.approx(0.32)
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_reward_only_cannot_pass_without_physical_progress():
    samples = basketball_samples(velocity_mps=0.0)
    for sample in samples:
        sample["reward"] = 1e30

    record = evaluate(samples)

    assert record["integrated_absolute_ball_travel_m"] == 0.0
    assert record["progress_complete"] is False
    assert record["learned_success"] is False
    assert "command-directed ball progress below target" in record["failures"]


def test_drift_with_progress_but_wrong_body_velocity_is_not_controlled():
    samples = basketball_samples()
    for sample in samples:
        sample["body_forward_mps"] = -.1
    record = evaluate(samples)
    assert record["progress_complete"]
    assert not record["tracking_complete"]
    assert not record["controlled_rolling"]


def test_sliding_without_ball_rotation_is_not_controlled_rolling():
    samples = basketball_samples()
    for sample in samples:
        sample["ball_angular_speed_rad_s"] = 0
    record = evaluate(samples)
    assert record["progress_complete"]
    assert not record["rotation_complete"]
    assert not record["controlled_rolling"]


def test_wrong_direction_travel_is_reported_as_random_not_controlled():
    record = evaluate(
        basketball_samples(velocity_mps=0.20, world_direction=(-1.0, 0.0))
    )

    assert record["integrated_absolute_ball_travel_m"] == pytest.approx(0.80)
    assert record["signed_commanded_progress_m"] == pytest.approx(-0.80)
    assert record["random_ball_movement"] is True
    assert record["controlled_rolling"] is False
    assert record["learned_success"] is False


def test_initial_yaw_projects_command_into_world_direction():
    xmat_yaw_90 = (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    initial_forward = initial_heading_from_xmat(xmat_yaw_90)
    record = evaluate(
        basketball_samples(world_direction=(0.0, 1.0)),
        initial_forward=initial_forward,
    )

    assert initial_forward == pytest.approx((0.0, 1.0))
    assert record["signed_commanded_progress_m"] == pytest.approx(0.32)
    assert record["controlled_rolling"] is True


def test_hidden_auto_reset_cannot_turn_two_partial_runs_into_survival():
    samples = basketball_samples()
    midpoint = len(samples) // 2
    for index in range(midpoint, len(samples)):
        samples[index]["ball_position"][0] -= samples[midpoint - 1]["ball_position"][0]

    record = evaluate(samples, automatic_resets=1)

    assert record["measured_steps"] == record["required_steps"]
    assert record["survived"] is False
    assert record["duration_complete"] is False
    assert record["learned_success"] is False
    assert "automatic reset observed" in record["failures"]


def test_terminated_first_fall_preserves_actual_survival_and_reason():
    samples = basketball_samples()[:30]
    record = evaluate(
        samples,
        terminated=True,
        fall_reasons=("robot_floor_contact",),
    )

    assert record["measured_steps"] == 30
    assert record["elapsed_s"] == pytest.approx(0.6)
    assert record["survived"] is False
    assert record["first_fall_reasons"] == ["robot_floor_contact"]
    assert record["learned_success"] is False


@pytest.mark.parametrize(
    "sample_kwargs,expected_failure",
    [
        ({"hold": 0.1}, "hold assistance observed"),
        ({"body_ball_contact": True}, "body-ball contact exploit observed"),
        ({"robot_floor_contact": True}, "robot-floor contact exploit observed"),
    ],
)
def test_assistance_and_contact_exploits_fail_closed(sample_kwargs, expected_failure):
    record = evaluate(basketball_samples(**sample_kwargs))

    assert record["exploit_free"] is False
    assert record["learned_success"] is False
    assert expected_failure in record["failures"]


def test_summary_reports_every_seed_and_separates_balance_from_rolling():
    balance = [
        evaluate(
            basketball_samples(velocity_mps=0.0),
            seed=seed,
            command=(0.0, 0.0, 0.0),
        )
        for seed in (101, 202, 303)
    ]
    rolling = [
        evaluate(basketball_samples(), seed=seed)
        for seed in (101, 202, 303)
    ]

    summary = summarize_trials([*balance, *rolling])

    assert summary["trial_count"] == 6
    assert summary["survival_count"] == 6
    assert summary["all_evaluated_seeds_reported"] is True
    assert [case["classification"] for case in summary["cases"]] == [
        "zero_command_balance",
        "commanded_rolling",
    ]
    assert summary["cases"][0]["sustained_balance_count"] == 3
    assert summary["cases"][0]["learned_success_count"] == 0
    assert summary["cases"][1]["controlled_rolling_count"] == 3
    assert summary["passed"] is True
