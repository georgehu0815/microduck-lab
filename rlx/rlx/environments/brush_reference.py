"""Authored color strokes and approach/dip/lift instructions, never rendered as ink."""

from __future__ import annotations

import numpy as np

from rlx.environments.drawing_reference import _resample_polyline, swimming_duck_strokes

COLORS = ((0.95, 0.69, 0.08), (0.96, 0.32, 0.05), (0.13, 0.12, 0.10), (0.10, 0.49, 0.84))
COLOR_NAMES = ("gold", "orange", "charcoal", "blue")
PALETTE_X = 0.150
PALETTE_Y = -0.034
PALETTE_Z = (0.2425, 0.2305, 0.2185, 0.2065)
BRUSH_RADIUS = 0.0010


def hatch(polygon: np.ndarray, spacing: float = 0.0014) -> list[np.ndarray]:
    result = []
    for height in np.arange(polygon[:, 1].min() + spacing / 2, polygon[:, 1].max(), spacing):
        crossings = []
        for start, end in zip(polygon[:-1], polygon[1:]):
            if (start[1] <= height < end[1]) or (end[1] <= height < start[1]):
                crossings.append(start[0] + (height - start[1]) * (end[0] - start[0]) / (end[1] - start[1]))
        crossings.sort()
        for index in range(0, len(crossings) - 1, 2):
            line = np.array([[crossings[index], height], [crossings[index + 1], height]])
            result.append(line if len(result) % 2 == 0 else line[::-1])
    return result


def color_strokes(scale=1.0, offset=(0.0, 0.0)):
    strokes = swimming_duck_strokes()
    tail = np.concatenate((strokes[3], strokes[3][:1]))
    source = [(0, line) for polygon in (strokes[0], strokes[1], tail) for line in hatch(polygon)]
    source += [(1, line) for line in hatch(strokes[2], 0.001)]
    source += [(1, strokes[2]), (1, strokes[4])]
    source += [(2, strokes[index]) for index in (0, 1, 3, 5)]
    source += [(3, strokes[index]) for index in (6, 7)]
    return [(color, points * scale + np.asarray(offset)) for color, points in source]


def brush_trajectory(steps=6000, scale=1.0, offset=(0.0, 0.0)):
    strokes = color_strokes(scale, offset)
    pieces = []
    position = np.array([0.154, 0.0, 0.2245])
    loaded = -1

    def piece(end, mode, color, count):
        nonlocal position
        points = np.linspace(position, end, max(2, count))
        pieces.append((points, mode, color))
        position = np.asarray(end)

    draw_length = sum(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() for _, points in strokes)
    overhead = len(strokes) * 55 + 4 * 130 + 30
    if steps <= overhead + 200:
        raise ValueError(f"brush horizon needs > {overhead + 200} steps")
    draw_steps = steps - overhead
    for color, stroke in strokes:
        if color != loaded:
            well = np.array([PALETTE_X - 0.00085, PALETTE_Y, PALETTE_Z[color]])
            piece([position[0] - 0.007, position[1], position[2]], 0, color, 15)
            piece(well + [-0.006, 0, 0], 0, color, 60)
            piece(well, 1, color, 25)
            piece(well, 1, color, 30)
            loaded = color
        piece([position[0] - 0.005, position[1], position[2]], 0, color, 6)
        start = np.r_[0.15915, stroke[0] + [0, 0.2245]]
        piece(start + [-0.006, 0, 0], 0, color, 34)
        piece(start, 0, color, 15)
        count = max(2, round(draw_steps * np.linalg.norm(np.diff(stroke, axis=0), axis=1).sum() / draw_length))
        path = _resample_polyline(stroke, count)
        points = np.column_stack((np.full(count, 0.15915), path[:, 0], path[:, 1] + 0.2245))
        pieces.append((points, 2, color))
        position = points[-1]
    piece(position + [-0.007, 0, 0], 0, 3, 30)
    points = np.concatenate([part[0] for part in pieces])
    modes = np.concatenate([np.full(len(part[0]), part[1]) for part in pieces])
    colors = np.concatenate([np.full(len(part[0]), part[2]) for part in pieces])
    velocity = np.vstack((np.diff(points[:, 1:], axis=0) / 0.02, [0, 0]))
    pen_down = (points[:, 0] >= 0.160 - BRUSH_RADIUS - 0.00015) & (modes != 1)
    return {"points": points[:, 1:] - [0, 0.2245], "xyz": points, "mode": modes,
            "color": colors, "pen_down": pen_down, "velocity": velocity}
