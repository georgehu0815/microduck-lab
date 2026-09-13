"""Focused tests for the standalone actual-paint brush renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from rlx.environments.brush_reference import BRUSH_RADIUS, COLORS
from rlx.environments.drawing import CANVAS_Z


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "render_microduck_brush.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_microduck_brush", MODULE_PATH
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_paint_image_uses_fixed_actual_contact_yz_and_color_disks():
    trace = [
        [0.0, 9.0, 0.0, CANVAS_Z, 1, 0.1, True],
        [0.1, -9.0, 0.031, CANVAS_Z + 0.031, 2, 0.1, True],
    ]
    image = module.paint_image(trace, [0, 3], width=720, height=540)
    pixels = np.asarray(image)
    gold = np.array([round(channel * 255) for channel in COLORS[0]])
    blue = np.array([round(channel * 255) for channel in COLORS[3]])

    np.testing.assert_array_equal(pixels[270, 360], gold)
    np.testing.assert_array_equal(pixels[0, 630], blue)
    metric_scale = 540 / 0.062
    radius_pixels = round(BRUSH_RADIUS * metric_scale)
    np.testing.assert_array_equal(pixels[270, 360 + radius_pixels], gold)
    assert not np.array_equal(pixels[270, 20], gold)


def test_paint_image_rejects_misaligned_or_invalid_actual_trace():
    with pytest.raises(ValueError, match="same number"):
        module.paint_image([[0, 0, 0, CANVAS_Z]], [])
    with pytest.raises(ValueError, match="color index"):
        module.paint_image([[0, 0, 0, CANVAS_Z]], [99])


class SimpleScene:
    def __init__(self, maxgeom=10):
        self.maxgeom = maxgeom
        self.ngeom = 0
        self.geoms = [SimpleGeom() for _ in range(maxgeom)]


class SimpleGeom:
    category = None


def test_contact_geoms_use_surface_x_colors_and_preserve_breaks(monkeypatch):
    calls = []

    class FakeMujoco:
        class mjtGeom:
            mjGEOM_CAPSULE = 3

        class mjtCatBit:
            mjCAT_DECOR = 4

        @staticmethod
        def mjv_initGeom(*args):
            calls.append(("init", args))

        @staticmethod
        def mjv_connector(*args):
            calls.append(("connector", args))

    monkeypatch.setitem(sys.modules, "mujoco", FakeMujoco)
    trace = [
        [0.00, 0.1, -0.01, CANVAS_Z, 1, 0.1, True],
        [0.02, 0.2, 0.00, CANVAS_Z + 0.001, 1, 0.1, True],
        [0.04, 0.3, 0.01, CANVAS_Z + 0.002, 2, 0.1, True],
        [0.06, 0.4, 0.02, CANVAS_Z + 0.003, 2, 0.1, True],
    ]
    scene = SimpleScene()

    added = module.add_contact_geoms(scene, trace, [0, 0, 3, 3])

    assert added == scene.ngeom == 2
    connectors = [args for kind, args in calls if kind == "connector"]
    assert len(connectors) == 2
    for args in connectors:
        assert args[2] == BRUSH_RADIUS
        assert args[3][0] == pytest.approx(module.PAINT_SURFACE_X)
        assert args[4][0] == pytest.approx(module.PAINT_SURFACE_X)
    rgba = [args[-1] for kind, args in calls if kind == "init"]
    np.testing.assert_allclose(rgba[0], [*COLORS[0], 1])
    np.testing.assert_allclose(rgba[1], [*COLORS[3], 1])
    assert all(geom.category == FakeMujoco.mjtCatBit.mjCAT_DECOR for geom in scene.geoms[:2])

    limited = SimpleScene()
    assert module.add_contact_geoms(
        limited, trace, [0, 0, 3, 3], max_segments=1
    ) == 1
    assert limited.ngeom == 1


def test_raw_trace_includes_aligned_color_fields(tmp_path):
    trace = [[0.02, 0.1597, 0.0, CANVAS_Z, 7, 0.2, True]]

    paths = module.write_raw_trace(
        tmp_path, trace, [2], controller="onnx_policy", seed=501
    )

    csv_lines = (tmp_path / "raw_trace.csv").read_text().splitlines()
    payload = json.loads((tmp_path / "raw_trace.json").read_text())
    assert csv_lines[0].endswith(",color,color_name")
    assert csv_lines[1].endswith(",2,charcoal")
    assert payload["rows"][0][-2:] == [2, "charcoal"]
    assert payload["controller"] == "onnx_policy"
    assert paths["csv"].endswith("raw_trace.csv")


def test_cli_requires_exactly_one_source_and_has_brush_defaults(tmp_path):
    args = module.parse_args(["--onnx", "policy.onnx", "--out", str(tmp_path)])
    assert args.onnx == Path("policy.onnx")
    assert args.teacher is False
    assert args.seed == 501
    assert args.seconds == 120.0
    assert args.fps == 10

    teacher = module.parse_args(["--teacher", "--out", str(tmp_path)])
    assert teacher.teacher is True
    assert teacher.onnx is None
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--teacher",
                "--onnx",
                "policy.onnx",
                "--out",
                str(tmp_path),
            ]
        )


def test_studio_frame_sheet_and_receipt_contract_constants():
    assert module.FRAME_SHEET_NAME == "frame_sheet.png"
    assert module.CONTRACT_VERSION == "microduck-brush-v2"
    assert module.DEFAULT_VIDEO_FPS == 10


@pytest.mark.parametrize("fps", [0, -1, 51, float("nan")])
def test_render_rejects_invalid_fps_before_loading_sources(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        module.render_brush(
            onnx_path=tmp_path / "missing.onnx",
            output=tmp_path,
            fps=fps,
        )


def test_onnx_policy_uses_cpu_runtime_and_validates_contract(monkeypatch):
    calls = {}

    class Value:
        def __init__(self, name):
            self.name = name

    class Session:
        def __init__(self, path, providers):
            calls["path"] = path
            calls["providers"] = providers

        def get_inputs(self):
            return [Value("obs")]

        def get_outputs(self):
            return [Value("actions")]

        def run(self, outputs, feed):
            calls["outputs"] = outputs
            calls["feed"] = feed
            return [np.zeros((1, 15), np.float32)]

    class FakeOrt:
        InferenceSession = Session

    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOrt)
    policy = module.OnnxPolicy(Path("learned.onnx"))
    action = policy(np.zeros(93, np.float32))

    assert calls["providers"] == ["CPUExecutionProvider"]
    assert calls["outputs"] == ["actions"]
    assert calls["feed"]["obs"].shape == (1, 93)
    assert action.shape == (15,)
