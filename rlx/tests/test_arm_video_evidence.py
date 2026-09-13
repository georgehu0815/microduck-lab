"""Regression tests for bounded arm video evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from rlx.environments import arm


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "render_microduck_arm_video_evidence",
    EXAMPLES / "render_microduck_arm.py",
)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


def sample_metrics(*, passed: bool = False) -> dict[str, object]:
    return {
        "passed": passed,
        "gates": {"position": passed, "no_drop": True},
        "object_position": [0.1, 0.2, 0.3],
        "goal": [0.2, 0.3, 0.4],
        "position_error_m": 0.02,
        "tip_error_m": 0.01,
        "peak_lift_m": 0.03,
        "peak_contact_force_n": 1.25,
        "peak_internal_force_n": 0.4,
        "invalid_contacts": 0,
        "drop_count": 0,
        "contact_flags": [[True, False]],
        "evaluator_stage": "lifted",
    }


def test_trace_cli_is_optional_and_defaults_off(tmp_path):
    base = [
        "--case",
        "arm-reach-v1",
        "--teacher",
        "--output",
        str(tmp_path),
    ]
    assert renderer.parse_args(base).trace is False
    assert renderer.parse_args([*base, "--trace"]).trace is True


def test_telemetry_record_has_step_time_action_reward_and_metrics():
    metrics = sample_metrics()
    record = renderer.telemetry_record(
        step=3,
        action=np.array([0.1, -0.2]),
        reward=0.75,
        cumulative_return=1.25,
        metrics=metrics,
    )
    assert record == {
        "step": 3,
        "time": 3 * arm.CONTROL_DT,
        "action": [0.1, -0.2],
        "reward": 0.75,
        "cumulative_return": 1.25,
        "metrics": metrics,
    }
    json.dumps(record, allow_nan=False)


def test_overlay_labels_live_and_terminal_evidence():
    assert renderer.OVERLAY_FONT_SIZE >= 17
    assert renderer.OVERLAY_LINE_HEIGHT == 22
    assert renderer.overlay_line_color("FINAL PASS") == (90, 230, 130)
    assert renderer.overlay_line_color("FINAL FAIL") == (255, 105, 105)
    assert renderer.overlay_line_color("LIVE") == (255, 255, 255)

    live = renderer.overlay_lines(
        controller="ppo_residual",
        seed=7,
        step=25,
        metrics=sample_metrics(),
        case_id="arm-pick-place-v1",
        cumulative_return=2.5,
    )
    joined_live = "\n".join(live)
    assert "SIMULATION ONLY - HARDWARE NOT VERIFIED" in joined_live
    assert "composite authored IK/FSM + PPO residual" in joined_live
    assert "case: arm-pick-place-v1" in joined_live
    assert "t: 0.50s" in joined_live
    assert "stage: lifted" in joined_live
    assert "return: 2.500" in joined_live
    assert "LIVE (not final verdict)" in joined_live
    assert "object xyz:" in joined_live
    assert "drop: 0" in joined_live
    assert "force: 1.25N" in joined_live
    assert "FINAL" not in joined_live

    terminal = renderer.overlay_lines(
        controller="teacher",
        seed=7,
        step=30,
        metrics=sample_metrics(),
        case_id="arm-pick-place-v1",
        cumulative_return=3.0,
        terminal=True,
    )
    joined_terminal = "\n".join(terminal)
    assert "SIMULATION ONLY - HARDWARE NOT VERIFIED" in joined_terminal
    assert "authored IK/FSM teacher" in joined_terminal
    assert "PPO residual" not in joined_terminal
    assert "FINAL FAIL" in joined_terminal
    assert "failed gates: position" in joined_terminal


def test_receipt_hashes_renderer_and_optional_telemetry(tmp_path):
    video = tmp_path / renderer.VIDEO_NAME
    sheet = tmp_path / renderer.CONTACT_SHEET_NAME
    video.write_bytes(b"video")
    sheet.write_bytes(b"sheet")
    common = {
        "output": tmp_path,
        "case_id": "arm-reach-v1",
        "controller": "teacher",
        "checkpoint": None,
        "checkpoint_provenance": None,
        "seed": 1,
        "options": {},
        "episode": {"contract": "md-arm-table-v1"},
        "export": None,
    }

    without_trace = renderer.build_receipt(**common)
    assert without_trace["telemetry_sha256"] is None
    assert without_trace["outputs"]["telemetry"] is None
    assert renderer.TELEMETRY_NAME not in without_trace["media_hashes"]
    assert without_trace["renderer_source_sha256"] == renderer.sha256_file(
        renderer.__file__
    )

    telemetry = tmp_path / renderer.TELEMETRY_NAME
    telemetry.write_text('{"step":1}\n')
    with_trace = renderer.build_receipt(**common, telemetry=telemetry)
    expected = renderer.sha256_file(telemetry)
    assert with_trace["telemetry_sha256"] == expected
    assert with_trace["media_hashes"][renderer.TELEMETRY_NAME] == expected
    assert with_trace["outputs"]["telemetry"] == str(telemetry.resolve())


def test_trace_and_odd_terminal_frame_are_recorded_without_hold_frames(
    tmp_path, monkeypatch
):
    import imageio.v2 as imageio

    class FakeEnv:
        contract = "md-arm-table-v1"
        observation_dim = 66
        action_dim = 6

        def __init__(self, case_id, render_mode=None):
            self.case_id = case_id
            self.render_mode = render_mode
            self.steps = 0
            self.terminated = False
            self.truncated = False
            self.controller = "manual"

        def reset(self, seed, options):
            return np.zeros(self.observation_dim), sample_metrics()

        def teacher_action(self):
            return np.full(self.action_dim, 0.25)

        def step(self, action):
            self.steps += 1
            self.terminated = self.steps == 3
            metrics = sample_metrics(passed=self.terminated)
            metrics["gates"] = {"position": self.terminated, "no_drop": True}
            return (
                np.zeros(self.observation_dim),
                float(self.steps),
                self.terminated,
                False,
                metrics,
            )

        def render(self):
            return np.zeros((240, 640, 3), dtype=np.uint8)

        def metrics(self):
            return sample_metrics(passed=True)

        def close(self):
            pass

    class FakeWriter:
        def __init__(self, path):
            self.path = Path(path)
            self.frames = []

        def __enter__(self):
            self.path.write_bytes(b"mock-mp4")
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def append_data(self, frame):
            self.frames.append(frame)

    writers = []

    def fake_get_writer(path, **kwargs):
        writer = FakeWriter(path)
        writers.append((writer, kwargs))
        return writer

    monkeypatch.setattr(renderer.arm, "ArmEnv", FakeEnv)
    monkeypatch.setattr(imageio, "get_writer", fake_get_writer)
    monkeypatch.setattr(
        renderer,
        "write_contact_sheet",
        lambda frames, path: path.write_bytes(b"mock-sheet"),
    )

    receipt = renderer.render_episode(
        case_id="arm-reach-v1",
        output=tmp_path,
        teacher=True,
        checkpoint=None,
        seed=80000,
        options={},
        fps=25.0,
        export=False,
        trace=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / renderer.TELEMETRY_NAME)
        .read_text()
        .splitlines()
    ]
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert set(rows[0]) == {
        "step",
        "time",
        "action",
        "reward",
        "cumulative_return",
        "metrics",
    }
    assert rows[-1]["time"] == 3 * arm.CONTROL_DT
    assert rows[-1]["cumulative_return"] == 6.0
    assert len(writers) == 1
    assert len(writers[0][0].frames) == 2

    timing = receipt["episode"]["video_timing"]
    assert timing["frame_count"] == receipt["episode"]["frames"] == 2
    assert timing["frame_times_seconds"] == [
        2 * arm.CONTROL_DT,
        3 * arm.CONTROL_DT,
    ]
    assert timing["terminal_frame_off_cadence"] is True
    assert np.isclose(
        timing["final_sample_duration_seconds"], arm.CONTROL_DT
    )
    assert timing["appended_hold_frames"] == 0
    assert receipt["telemetry_sha256"] == renderer.sha256_file(
        tmp_path / renderer.TELEMETRY_NAME
    )
    assert (
        receipt["media_hashes"][renderer.TELEMETRY_NAME]
        == receipt["telemetry_sha256"]
    )
