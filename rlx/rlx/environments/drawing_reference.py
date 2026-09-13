"""Deterministic authored reference data for the swimming-duck drawing task."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Iterable

import numpy as np


REFERENCE_VERSION = "swimming-duck-pencil-v1"
_DT = 0.02


def _as_polyline(points: Iterable[tuple[float, float]]) -> np.ndarray:
    return np.asarray(tuple(points), dtype=np.float64)


_UNIT_STROKES = (
    # Body
    _as_polyline(
        (
            (-0.0160, 0.0010),
            (-0.0130, 0.0060),
            (-0.0070, 0.0085),
            (0.0000, 0.0080),
            (0.0060, 0.0050),
            (0.0090, 0.0010),
            (0.0060, -0.0045),
            (0.0000, -0.0070),
            (-0.0080, -0.0065),
            (-0.0140, -0.0030),
            (-0.0160, 0.0010),
        )
    ),
    # Head
    _as_polyline(
        (
            (0.0045, 0.0060),
            (0.0055, 0.0105),
            (0.0090, 0.0135),
            (0.0130, 0.0130),
            (0.0160, 0.0100),
            (0.0165, 0.0060),
            (0.0140, 0.0030),
            (0.0100, 0.0025),
            (0.0065, 0.0035),
            (0.0045, 0.0060),
        )
    ),
    # Beak
    _as_polyline(
        (
            (0.0155, 0.0090),
            (0.0200, 0.0070),
            (0.0155, 0.0055),
            (0.0155, 0.0090),
        )
    ),
    # Tail feathers
    _as_polyline(
        (
            (-0.0140, 0.0040),
            (-0.0195, 0.0080),
            (-0.0170, 0.0020),
            (-0.0200, -0.0010),
            (-0.0140, -0.0015),
        )
    ),
    # Wing
    _as_polyline(
        (
            (-0.0090, 0.0030),
            (-0.0040, 0.0055),
            (0.0020, 0.0035),
            (0.0040, 0.0000),
            (-0.0010, -0.0025),
            (-0.0070, -0.0010),
            (-0.0090, 0.0030),
        )
    ),
    # Eye
    _as_polyline(
        (
            (0.0115, 0.0100),
            (0.0120, 0.0105),
            (0.0125, 0.0100),
            (0.0120, 0.0095),
            (0.0115, 0.0100),
        )
    ),
    # Upper water wave
    _as_polyline(
        (
            (-0.0200, -0.0095),
            (-0.0150, -0.0110),
            (-0.0100, -0.0095),
            (-0.0050, -0.0080),
            (0.0000, -0.0095),
            (0.0050, -0.0110),
            (0.0100, -0.0095),
            (0.0150, -0.0080),
            (0.0200, -0.0095),
        )
    ),
    # Lower water wave
    _as_polyline(
        (
            (-0.0170, -0.0130),
            (-0.0120, -0.0145),
            (-0.0070, -0.0130),
            (-0.0020, -0.0115),
            (0.0030, -0.0130),
            (0.0080, -0.0145),
            (0.0130, -0.0130),
            (0.0180, -0.0115),
        )
    ),
)


def _positive_scale(scale: float) -> float:
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise ValueError("scale must be a positive finite number")
    value = float(scale)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("scale must be a positive finite number")
    return value


def swimming_duck_strokes(scale: float = 1.0) -> list[np.ndarray]:
    """Return authored swimming-duck pencil strokes in canvas metres."""

    factor = _positive_scale(scale)
    return [(stroke * factor).copy() for stroke in _UNIT_STROKES]


def _polyline_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        return np.empty((0, 2), dtype=np.float64)
    if len(points) == 0:
        raise ValueError("cannot sample an empty polyline")
    if len(points) == 1:
        return np.repeat(points, count, axis=0)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total == 0:
        return np.repeat(points[:1], count, axis=0)

    distances = np.linspace(0.0, total, count)
    segment_ids = np.searchsorted(cumulative, distances, side="right") - 1
    segment_ids = np.clip(segment_ids, 0, len(segment_lengths) - 1)
    local = distances - cumulative[segment_ids]
    fractions = np.divide(
        local,
        segment_lengths[segment_ids],
        out=np.zeros_like(local),
        where=segment_lengths[segment_ids] > 0,
    )
    return (
        points[segment_ids]
        + (points[segment_ids + 1] - points[segment_ids]) * fractions[:, None]
    )


def _proportional_counts(lengths: np.ndarray, total: int) -> np.ndarray:
    minimum = np.full(len(lengths), 2, dtype=np.int64)
    remaining = total - int(minimum.sum())
    if remaining < 0:
        raise ValueError(
            f"steps must be at least {int(minimum.sum()) + 2} "
            "to include every stroke and pen-up travel"
        )
    if remaining == 0:
        return minimum

    weights = lengths / lengths.sum()
    exact = weights * remaining
    additions = np.floor(exact).astype(np.int64)
    leftover = remaining - int(additions.sum())
    if leftover:
        order = np.argsort(-(exact - additions), kind="stable")
        additions[order[:leftover]] += 1
    return minimum + additions


def make_trajectory(
    steps: int = 1600,
    scale: float = 1.0,
    offset: tuple[float, float] = (0, 0),
) -> dict[str, np.ndarray]:
    """Sample the drawing and pen-up travel into a fixed 50 Hz trajectory."""

    if isinstance(steps, bool) or not isinstance(steps, Integral):
        raise ValueError("steps must be an integer")
    offset_array = np.asarray(offset, dtype=np.float64)
    if offset_array.shape != (2,) or not np.isfinite(offset_array).all():
        raise ValueError("offset must contain two finite coordinates")

    strokes = swimming_duck_strokes(scale)
    pieces: list[tuple[np.ndarray, bool, int]] = []
    for stroke_id, stroke in enumerate(strokes):
        if stroke_id:
            travel = np.stack((strokes[stroke_id - 1][-1], stroke[0]))
            pieces.append((travel, False, -1))
        pieces.append((stroke, True, stroke_id))

    available = int(steps) - 2
    lengths = np.asarray([_polyline_length(piece[0]) for piece in pieces])
    counts = _proportional_counts(lengths, available)

    sampled_points = [strokes[0][0][None, :]]
    sampled_pen = [np.zeros(1, dtype=np.bool_)]
    sampled_ids = [np.full(1, -1, dtype=np.int64)]
    for (piece, pen_down, stroke_id), count in zip(pieces, counts, strict=True):
        sampled_points.append(_resample_polyline(piece, int(count)))
        sampled_pen.append(np.full(int(count), pen_down, dtype=np.bool_))
        sampled_ids.append(np.full(int(count), stroke_id, dtype=np.int64))
    sampled_points.append(strokes[-1][-1][None, :])
    sampled_pen.append(np.zeros(1, dtype=np.bool_))
    sampled_ids.append(np.full(1, -1, dtype=np.int64))

    points = (np.concatenate(sampled_points) + offset_array).astype(np.float32)
    pen_down = np.concatenate(sampled_pen)
    stroke_ids = np.concatenate(sampled_ids)
    velocity = np.zeros_like(points)
    velocity[1:] = np.diff(points, axis=0) / _DT
    return {
        "points": points,
        "pen_down": pen_down,
        "stroke_ids": stroke_ids,
        "velocity": velocity,
    }


def _validated_strokes(
    strokes: list[np.ndarray],
    *,
    name: str,
) -> list[np.ndarray]:
    if not isinstance(strokes, list):
        raise ValueError(f"{name} must be a list of 2D stroke arrays")
    validated = []
    for stroke in strokes:
        array = np.asarray(stroke, dtype=np.float64)
        if array.ndim != 2 or array.shape[1:] != (2,):
            raise ValueError(f"{name} must contain arrays with shape (N, 2)")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} strokes must be finite")
        if len(array):
            validated.append(array)
    return validated


def reference_svg(strokes: list[np.ndarray] | None = None) -> str:
    """Render strokes as a standalone SVG whose coordinates are millimetres."""

    source = swimming_duck_strokes() if strokes is None else strokes
    validated = _validated_strokes(source, name="strokes")
    if not validated:
        raise ValueError("strokes must contain at least one point")
    all_points = np.concatenate(validated) * 1000.0
    minimum = all_points.min(axis=0) - 2.0
    maximum = all_points.max(axis=0) + 2.0
    width, height = maximum - minimum
    paths = []
    for stroke in validated:
        millimetres = stroke * 1000.0
        commands = [f"M {millimetres[0, 0]:.3f} {-millimetres[0, 1]:.3f}"]
        commands.extend(f"L {x:.3f} {-y:.3f}" for x, y in millimetres[1:])
        paths.append(f'  <path d="{" ".join(commands)}" />')
    view_y = -maximum[1]
    return "\n".join(
        (
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'     viewBox="{minimum[0]:.3f} {view_y:.3f} {width:.3f} {height:.3f}"',
            '     data-source="deterministic-authored-pencil-reference"',
            '     data-units="millimetres">',
            f"  <metadata>Swimming duck; source={REFERENCE_VERSION}; "
            "input units=metres; display units=millimetres</metadata>",
            '  <g fill="none" stroke="#202020" stroke-width="0.6" '
            'stroke-linecap="round" stroke-linejoin="round">',
            *paths,
            "  </g>",
            "</svg>",
        )
    )


def _curve_samples(
    strokes: list[np.ndarray],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    points = []
    weights = []
    total_length = 0.0
    for stroke in strokes:
        if len(stroke) == 1:
            points.append(stroke)
            weights.append(np.zeros(1, dtype=np.float64))
            continue
        for start, end in zip(stroke[:-1], stroke[1:], strict=True):
            delta = end - start
            length = float(np.linalg.norm(delta))
            if length == 0:
                continue
            count = max(1, math.ceil(length / spacing))
            fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
            points.append(start + fractions[:, None] * delta)
            weights.append(np.full(count, length / count))
            total_length += length
    if not points:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
        )
    return np.concatenate(points), np.concatenate(weights), total_length


def _nearest_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(target) == 0:
        return np.full(len(source), np.inf)
    result = np.empty(len(source), dtype=np.float64)
    chunk_size = max(1, 1_000_000 // len(target))
    for start in range(0, len(source), chunk_size):
        chunk = source[start : start + chunk_size]
        squared = np.sum((chunk[:, None, :] - target[None, :, :]) ** 2, axis=2)
        result[start : start + len(chunk)] = np.sqrt(squared.min(axis=1))
    return result


def _weighted_fraction(
    distances: np.ndarray,
    weights: np.ndarray,
    tolerance: float,
) -> float:
    total = float(weights.sum())
    if total == 0:
        return 0.0
    return float(weights[distances <= tolerance].sum() / total)


def _weighted_mean(distances: np.ndarray, weights: np.ndarray) -> float | None:
    total = float(weights.sum())
    if total == 0:
        return None
    return float(np.dot(distances, weights) / total)


def assess_drawing(
    reference_strokes: list[np.ndarray],
    recorded_strokes: list[np.ndarray],
    tolerance_m: float = 0.0015,
) -> dict[str, float | bool | None]:
    """Measure length-weighted bidirectional agreement between drawn curves."""

    if (
        isinstance(tolerance_m, bool)
        or not isinstance(tolerance_m, Real)
        or not math.isfinite(float(tolerance_m))
        or float(tolerance_m) <= 0
    ):
        raise ValueError("tolerance_m must be a positive finite number")
    tolerance = float(tolerance_m)
    reference = _validated_strokes(reference_strokes, name="reference_strokes")
    recorded = _validated_strokes(recorded_strokes, name="recorded_strokes")
    spacing = tolerance / 3.0

    reference_points, reference_weights, reference_length = _curve_samples(
        reference, spacing
    )
    recorded_points, recorded_weights, recorded_length = _curve_samples(
        recorded, spacing
    )
    reference_to_recorded = _nearest_distances(reference_points, recorded_points)
    recorded_to_reference = _nearest_distances(recorded_points, reference_points)
    coverage = _weighted_fraction(reference_to_recorded, reference_weights, tolerance)
    precision = _weighted_fraction(recorded_to_reference, recorded_weights, tolerance)
    reference_mean = _weighted_mean(reference_to_recorded, reference_weights)
    recorded_mean = _weighted_mean(recorded_to_reference, recorded_weights)
    chamfer = (
        (reference_mean + recorded_mean) / 2.0
        if reference_mean is not None and recorded_mean is not None
        else None
    )
    length_ratio = recorded_length / reference_length if reference_length > 0 else None
    passed = (
        coverage >= 0.85
        and precision >= 0.8
        and length_ratio is not None
        and 0.65 <= length_ratio <= 1.5
    )
    return {
        "coverage": coverage,
        "precision": precision,
        "symmetric_chamfer_m": chamfer,
        "recorded_length_m": recorded_length,
        "reference_length_m": reference_length,
        "length_ratio": length_ratio,
        "passed": passed,
    }
