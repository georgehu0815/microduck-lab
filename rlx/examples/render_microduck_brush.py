"""Render an actual-contact MicroDuck brush rollout from ONNX or the teacher."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from microduck_local import contract as C
from rlx.environments.brush import (
    ACTION_DIM,
    CONTRACT_VERSION,
    OBS_DIM,
    BrushEnv,
)
from rlx.environments.brush_reference import BRUSH_RADIUS, COLORS, COLOR_NAMES
from rlx.environments.drawing import CANVAS_Z


CANVAS_HALF_SPAN = 0.031
PAINT_SURFACE_X = 0.1597
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
DEFAULT_VIDEO_FPS = 10
SHEET_FRAMES = 6
FRAME_SHEET_NAME = "frame_sheet.png"
MAX_PAINT_GEOMS = 2_000
TRACE_COLUMNS = (
    "t",
    "x",
    "y",
    "z",
    "stroke_id",
    "force_n",
    "requested_down",
    "color",
    "color_name",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _paint_geometry(width: int, height: int) -> tuple[float, float, float, int]:
    if width < 1 or height < 1:
        raise ValueError("paint image dimensions must be positive")
    scale = min(width, height) / (2 * CANVAS_HALF_SPAN)
    center_x = width / 2
    center_y = height / 2
    radius = max(1, round(BRUSH_RADIUS * scale))
    return scale, center_x, center_y, radius


def _paint_point(
    row: Sequence[Any],
    *,
    scale: float,
    center_x: float,
    center_y: float,
) -> tuple[float, float]:
    if len(row) < 4:
        raise ValueError("trace rows must contain at least t, x, y, and z")
    y = float(row[2])
    z = float(row[3])
    if not math.isfinite(y) or not math.isfinite(z):
        raise ValueError("trace coordinates must be finite")
    return (
        center_x + y * scale,
        center_y - (z - CANVAS_Z) * scale,
    )


def _draw_paint(
    image: Any,
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
) -> None:
    from PIL import ImageDraw

    if len(trace) != len(colors):
        raise ValueError("trace and colors must contain the same number of records")
    scale, center_x, center_y, radius = _paint_geometry(*image.size)
    draw = ImageDraw.Draw(image)
    for row, color in zip(trace, colors, strict=True):
        color = int(color)
        if not 0 <= color < len(COLORS):
            raise ValueError(f"invalid paint color index: {color}")
        x, y = _paint_point(
            row, scale=scale, center_x=center_x, center_y=center_y
        )
        rgb = tuple(round(channel * 255) for channel in COLORS[color])
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=rgb,
        )


def paint_image(
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
    width: int = 720,
    height: int = 540,
) -> Any:
    """Rasterize only actual brush contacts in the fixed 62 mm YZ canvas."""
    from PIL import Image

    image = Image.new("RGB", (width, height), (250, 247, 237))
    _draw_paint(image, trace, colors)
    return image


def _continuous_runs(
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
) -> list[tuple[int, list[np.ndarray]]]:
    if len(trace) != len(colors):
        raise ValueError("trace and colors must contain the same number of records")
    runs: list[tuple[int, list[np.ndarray]]] = []
    active_key: tuple[int, int] | None = None
    for row, color_value in zip(trace, colors, strict=True):
        if len(row) < 5:
            raise ValueError("trace rows must contain a stroke id")
        color = int(color_value)
        if not 0 <= color < len(COLORS):
            raise ValueError(f"invalid paint color index: {color}")
        point = np.array(
            [PAINT_SURFACE_X, float(row[2]), float(row[3])],
            dtype=np.float64,
        )
        if not np.isfinite(point).all():
            raise ValueError("trace coordinates must be finite")
        key = (int(row[4]), color)
        if key != active_key:
            runs.append((color, [point]))
            active_key = key
        else:
            runs[-1][1].append(point)
    return runs


def _paint_segments(
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
    limit: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    if limit < 0:
        raise ValueError("paint segment limit must be nonnegative")
    runs = [
        (color, points)
        for color, points in _continuous_runs(trace, colors)
        if len(points) >= 2
    ]
    total = sum(len(points) - 1 for _, points in runs)
    if total <= limit:
        return [
            (start, end, color)
            for color, points in runs
            for start, end in zip(points, points[1:])
        ]
    if not runs or limit == 0:
        return []
    if limit < len(runs):
        selected = sorted(
            sorted(
                range(len(runs)),
                key=lambda index: len(runs[index][1]),
                reverse=True,
            )[:limit]
        )
        runs = [runs[index] for index in selected]
        allocations = [1] * len(runs)
    else:
        allocations = [
            max(1, limit * (len(points) - 1) // total)
            for _, points in runs
        ]
    while sum(allocations) > limit:
        index = max(range(len(allocations)), key=allocations.__getitem__)
        if allocations[index] == 1:
            break
        allocations[index] -= 1
    while sum(allocations) < limit:
        candidates = [
            index
            for index, (_, points) in enumerate(runs)
            if allocations[index] < len(points) - 1
        ]
        if not candidates:
            break
        index = max(
            candidates,
            key=lambda item: (len(runs[item][1]) - 1) / allocations[item],
        )
        allocations[index] += 1
    segments = []
    for (color, points), allocation in zip(runs, allocations, strict=True):
        indices = np.linspace(0, len(points) - 1, allocation + 1).astype(int)
        segments.extend(
            (points[start], points[end], color)
            for start, end in zip(indices, indices[1:])
            if end > start
        )
    return segments


def add_contact_geoms(
    scene: Any,
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
    *,
    max_segments: int = MAX_PAINT_GEOMS,
) -> int:
    """Add downsampled, color-preserving actual paint capsules to a scene."""
    import mujoco

    available = max(0, min(max_segments, scene.maxgeom - scene.ngeom))
    segments = _paint_segments(trace, colors, available)
    for start, end, color in segments:
        geom = scene.geoms[scene.ngeom]
        rgba = np.array((*COLORS[color], 1.0), dtype=np.float32)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            rgba,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            BRUSH_RADIUS,
            start,
            end,
        )
        scene.ngeom += 1
    return len(segments)


class OnnxPolicy:
    """CPU ONNX Runtime policy with a strict 93-observation/15-action contract."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("brush ONNX must have exactly one input and one output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        batch = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if batch.shape != (1, OBS_DIM):
            raise ValueError(f"brush ONNX observations must have shape ({OBS_DIM},)")
        result = self.session.run([self.output_name], {self.input_name: batch})[0]
        action = np.asarray(result, dtype=np.float32)
        if action.shape != (1, ACTION_DIM) or not np.isfinite(action).all():
            raise ValueError("brush ONNX returned an invalid action")
        return np.clip(action[0], -1.0, 1.0)


def _trace_rows(
    trace: Sequence[Sequence[Any]], colors: Sequence[int]
) -> list[list[Any]]:
    if len(trace) != len(colors):
        raise ValueError("trace and colors must contain the same number of records")
    rows = []
    for row, color_value in zip(trace, colors, strict=True):
        color = int(color_value)
        if not 0 <= color < len(COLORS):
            raise ValueError(f"invalid paint color index: {color}")
        rows.append([*list(row), color, COLOR_NAMES[color]])
    return rows


def write_raw_trace(
    output: Path,
    trace: Sequence[Sequence[Any]],
    colors: Sequence[int],
    *,
    controller: str,
    seed: int,
) -> dict[str, str]:
    rows = _trace_rows(trace, colors)
    csv_path = output / "raw_trace.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TRACE_COLUMNS)
        writer.writerows(rows)
    json_path = output / "raw_trace.json"
    write_json(
        json_path,
        {
            "controller": controller,
            "seed": seed,
            "columns": list(TRACE_COLUMNS),
            "rows": rows,
        },
    )
    return {"csv": str(csv_path.resolve()), "json": str(json_path.resolve())}


def _camera() -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.07, 0.0, 0.205]
    camera.distance = 0.58
    camera.azimuth = -45
    camera.elevation = -12
    return camera


def _frame_with_inset(frame: np.ndarray, painting: Any, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    inset_width, inset_height = 192, 144
    inset = painting.resize((inset_width, inset_height))
    left = image.width - inset_width - 12
    top = image.height - inset_height - 12
    image.paste(inset, (left, top))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (left - 1, top - 1, left + inset_width, top + inset_height),
        outline=(245, 245, 245),
        width=1,
    )
    draw.rectangle((0, 0, image.width, 34), fill=(12, 16, 20))
    draw.text((10, 9), label, fill=(255, 255, 255))
    return np.asarray(image)


def _write_contact_sheet(frames: Sequence[np.ndarray], path: Path) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise RuntimeError("cannot build a contact sheet without frames")
    retained = list(frames)
    while len(retained) < SHEET_FRAMES:
        retained.append(retained[-1])
    retained = retained[:SHEET_FRAMES]
    tile_width, tile_height = 320, 240
    sheet = Image.new("RGB", (tile_width * 3, tile_height * 2), (12, 16, 20))
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(retained):
        tile = Image.fromarray(frame).resize((tile_width, tile_height))
        x = (index % 3) * tile_width
        y = (index // 3) * tile_height
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + 30, y + 20), fill=(12, 16, 20))
        draw.text((x + 6, y + 4), str(index + 1), fill=(255, 255, 255))
    sheet.save(path)


def render_brush(
    *,
    onnx_path: Path | None,
    output: Path,
    seed: int = 501,
    seconds: float = 120.0,
    fps: float = DEFAULT_VIDEO_FPS,
    teacher: bool = False,
) -> dict[str, Any]:
    """Render one full BrushEnv episode while retaining only six sheet frames."""
    import imageio.v2 as imageio
    import mujoco

    if teacher == (onnx_path is not None):
        raise ValueError("select exactly one of an ONNX policy or --teacher")
    if not 70 <= seconds <= 180:
        raise ValueError("brush render seconds must be between 70 and 180")
    if not math.isfinite(fps) or not 0 < fps <= 50:
        raise ValueError("brush render fps must be finite and in (0, 50]")
    if onnx_path is not None and not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX policy does not exist: {onnx_path}")
    output.mkdir(parents=True, exist_ok=True)

    from rlx.environments import brush as brush_environment

    if teacher:
        source_path = Path(brush_environment.__file__).resolve()
        source_label = "TEACHER CONTROLLER (diagnostic only)"
        controller = "teacher"
        source_type = "teacher_controller"
        policy = None
    else:
        assert onnx_path is not None
        source_path = onnx_path.resolve()
        source_label = f"LEARNED ONNX POLICY | {source_path.name}"
        controller = "onnx"
        source_type = "policy"
        policy = OnnxPolicy(source_path)
    source_sha256 = sha256_file(source_path)

    env = BrushEnv(seed=seed, max_episode_s=seconds)
    env.model.vis.global_.offwidth = max(
        env.model.vis.global_.offwidth, VIDEO_WIDTH
    )
    env.model.vis.global_.offheight = max(
        env.model.vis.global_.offheight, VIDEO_HEIGHT
    )
    renderer = mujoco.Renderer(
        env.model,
        height=VIDEO_HEIGHT,
        width=VIDEO_WIDTH,
        max_geom=10_000,
    )
    camera = _camera()
    frame_stride = max(1, round((1 / C.CTRL_DT) / fps))
    actual_fps = (1 / C.CTRL_DT) / frame_stride
    expected_frames = math.ceil(env.max_steps / frame_stride)
    sheet_targets = set(
        np.linspace(0, max(0, expected_frames - 1), SHEET_FRAMES).astype(int)
    )
    retained_frames: list[np.ndarray] = []
    painting = paint_image([], [], width=720, height=540)
    painted_contacts = 0
    visible_segments = 0
    total_reward = 0.0
    control_steps = 0
    rendered_frames = 0
    observation, _ = env.reset(seed=seed)
    video_path = output / "rollout.mp4"
    try:
        with imageio.get_writer(
            video_path,
            fps=actual_fps,
            macro_block_size=None,
        ) as writer:
            for step in range(env.max_steps):
                if policy is None:
                    action = np.asarray(env.teacher_action(), dtype=np.float32)
                else:
                    action = policy(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                control_steps = step + 1
                if step % frame_stride == 0:
                    if painted_contacts < len(env.trace):
                        _draw_paint(
                            painting,
                            env.trace[painted_contacts:],
                            env.color_trace[painted_contacts:],
                        )
                        painted_contacts = len(env.trace)
                    renderer.update_scene(env.data, camera=camera)
                    visible_segments = add_contact_geoms(
                        renderer.scene, env.trace, env.color_trace
                    )
                    label = (
                        f"{source_label} | actual paint only | "
                        f"t={env.data.time:.2f}s"
                    )
                    frame = _frame_with_inset(
                        renderer.render().copy(), painting, label
                    )
                    writer.append_data(frame)
                    if rendered_frames in sheet_targets:
                        retained_frames.append(frame.copy())
                    rendered_frames += 1
                if terminated or truncated:
                    break
        trace = [list(row) for row in env.trace]
        colors = [int(color) for color in env.color_trace]
        assessment = env.assessment()
    finally:
        renderer.close()
        env.close()

    if rendered_frames == 0:
        raise RuntimeError("brush render produced no frames")
    if sha256_file(source_path) != source_sha256:
        raise RuntimeError("render source changed during rollout")

    painting_path = output / "painting.png"
    paint_image(trace, colors).save(painting_path)
    sheet_path = output / FRAME_SHEET_NAME
    _write_contact_sheet(retained_frames, sheet_path)
    raw_trace = write_raw_trace(
        output, trace, colors, controller=controller, seed=seed
    )
    report = {
        "created_at": utc_now(),
        "controller": controller,
        "source_type": source_type,
        "teacher_diagnostic": teacher,
        "source": str(source_path),
        "source_sha256": source_sha256,
        "contract_version": CONTRACT_VERSION,
        "recipe": "drawing",
        "seed": seed,
        "requested_seconds": seconds,
        "requested_fps": fps,
        "control_steps": control_steps,
        "elapsed_seconds": control_steps * C.CTRL_DT,
        "frames": rendered_frames,
        "fps": actual_fps,
        "return": total_reward,
        "assessment": assessment,
        "drawing_assessment": assessment,
        "environment": {
            "recipe": "drawing",
            "actuator": "xml",
            "max_episode_s": seconds,
            "assistance": 0,
            "recipe_options": {},
        },
        "paint_source": "actual MuJoCo brush-tip contact trace only",
        "visible_paint_segments": visible_segments,
        "outputs": {
            "video": str(video_path.resolve()),
            "painting": str(painting_path.resolve()),
            "contact_sheet": str(sheet_path.resolve()),
            "raw_trace": raw_trace,
        },
    }
    write_json(output / "render.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 93-observation/15-action MicroDuck brush policy."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--onnx", type=Path)
    source.add_argument(
        "--teacher",
        action="store_true",
        help="render the analytic teacher and label it as diagnostic",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = render_brush(
        onnx_path=args.onnx,
        output=args.out,
        seed=args.seed,
        seconds=args.seconds,
        fps=args.fps,
        teacher=args.teacher,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return report


if __name__ == "__main__":
    main()
