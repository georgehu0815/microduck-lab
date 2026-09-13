"""Tests for the authored swimming-duck drawing reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rlx.environments.drawing_reference import (
    REFERENCE_VERSION,
    assess_drawing,
    make_trajectory,
    reference_svg,
    swimming_duck_strokes,
)


def test_authored_reference_has_expected_parts_and_canvas_bounds():
    strokes = swimming_duck_strokes()

    assert isinstance(REFERENCE_VERSION, str) and REFERENCE_VERSION
    assert len(strokes) == 8
    assert all(stroke.ndim == 2 and stroke.shape[1] == 2 for stroke in strokes)
    points = np.concatenate(strokes)
    width, height = np.ptp(points, axis=0)
    assert width == pytest.approx(0.040)
    assert height == pytest.approx(0.028)
    assert np.max(np.abs(points.mean(axis=0))) < 0.004

    scaled = swimming_duck_strokes(0.5)
    for original, half_size in zip(strokes, scaled, strict=True):
        np.testing.assert_allclose(half_size, original * 0.5)
        assert not np.shares_memory(original, half_size)


def test_trajectory_is_deterministic_typed_and_splits_all_strokes():
    first = make_trajectory(steps=1600, offset=(0.003, -0.002))
    second = make_trajectory(steps=1600, offset=(0.003, -0.002))

    assert set(first) == {"points", "pen_down", "stroke_ids", "velocity"}
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert first["points"].shape == first["velocity"].shape == (1600, 2)
    assert first["points"].dtype == first["velocity"].dtype == np.float32
    assert first["pen_down"].shape == first["stroke_ids"].shape == (1600,)
    assert first["pen_down"].dtype == np.bool_
    assert np.issubdtype(first["stroke_ids"].dtype, np.integer)
    assert first["pen_down"][0] == first["pen_down"][-1] == np.False_
    assert first["stroke_ids"][0] == first["stroke_ids"][-1] == -1
    assert set(first["stroke_ids"][first["pen_down"]]) == set(range(8))
    assert np.all(first["stroke_ids"][~first["pen_down"]] == -1)

    transitions = np.flatnonzero(np.diff(first["stroke_ids"]) != 0)
    assert len(transitions) >= 15
    for stroke_id in range(7):
        end = np.flatnonzero(first["stroke_ids"] == stroke_id)[-1]
        start = np.flatnonzero(first["stroke_ids"] == stroke_id + 1)[0]
        assert np.any(~first["pen_down"][end + 1 : start])


def test_trajectory_is_finite_bounded_and_has_no_pen_up_teleports():
    trajectory = make_trajectory()
    points = trajectory["points"]

    assert np.isfinite(points).all()
    assert np.isfinite(trajectory["velocity"]).all()
    assert np.max(np.abs(points[:, 0])) <= 0.0201
    assert np.max(points[:, 1]) <= 0.0136
    assert np.min(points[:, 1]) >= -0.0146
    jumps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert jumps.max() < 0.001
    np.testing.assert_allclose(
        trajectory["velocity"][1:],
        np.diff(points, axis=0) / 0.02,
        atol=1e-7,
    )
    np.testing.assert_array_equal(trajectory["velocity"][0], np.zeros(2))


def test_reference_svg_declares_source_and_units():
    svg = reference_svg()

    assert svg.startswith("<svg")
    assert REFERENCE_VERSION in svg
    assert 'data-units="millimetres"' in svg
    assert svg.count("<path ") == 8
    assert "input units=metres" in svg


def test_correct_shape_and_mild_sampling_variation_pass():
    reference = swimming_duck_strokes()
    trajectory = make_trajectory(steps=800)
    recorded = [
        trajectory["points"][
            trajectory["pen_down"] & (trajectory["stroke_ids"] == stroke_id)
        ]
        for stroke_id in range(len(reference))
    ]

    exact = assess_drawing(reference, reference)
    sampled = assess_drawing(reference, recorded)

    assert exact["passed"] is True
    assert exact["coverage"] == pytest.approx(1.0)
    assert exact["precision"] == pytest.approx(1.0)
    assert exact["symmetric_chamfer_m"] < 0.0001
    assert exact["length_ratio"] == pytest.approx(1.0)
    assert sampled["passed"] is True


@pytest.mark.parametrize(
    "recorded",
    [
        [],
        [np.array([[0.0, 0.0]])],
        [stroke + np.array([0.012, 0.0]) for stroke in swimming_duck_strokes()],
        [
            np.tile(
                np.array(
                    [
                        [-0.020, -0.014],
                        [0.020, 0.014],
                        [-0.020, 0.014],
                        [0.020, -0.014],
                    ]
                ),
                (12, 1),
            )
        ],
    ],
    ids=["empty", "tiny-dot", "wrong-place", "repeated-scribble"],
)
def test_empty_dot_wrong_shape_and_scribble_fail(recorded):
    result = assess_drawing(swimming_duck_strokes(), recorded)

    assert result["passed"] is False
    assert 0.0 <= result["coverage"] <= 1.0
    assert 0.0 <= result["precision"] <= 1.0
    assert math.isfinite(result["recorded_length_m"])
    assert math.isfinite(result["reference_length_m"])
    assert result["symmetric_chamfer_m"] is None or math.isfinite(
        result["symmetric_chamfer_m"]
    )
    assert result["length_ratio"] is None or math.isfinite(result["length_ratio"])


def test_stationary_repetition_has_no_precision_or_length_credit():
    dot = np.repeat([[0.0, 0.0]], 10_000, axis=0)
    result = assess_drawing(swimming_duck_strokes(), [dot])

    assert result["precision"] == 0.0
    assert result["recorded_length_m"] == 0.0
    assert result["length_ratio"] == 0.0
    assert result["passed"] is False


@pytest.mark.parametrize("steps", [0, 31, 31.5, True])
def test_invalid_trajectory_steps_fail_closed(steps):
    with pytest.raises(ValueError):
        make_trajectory(steps=steps)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), True])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        swimming_duck_strokes(scale)
