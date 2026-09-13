from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_local import studio_policies
from microduck_local import viz_server

DRAWING_REWARD_WEIGHTS = {
    "tracking": 2.0,
    "contact": 0.4,
    "grasp": 0.3,
    "upright": 0.3,
    "action_rate_penalty": 0.01,
    "force_penalty": 0.02,
}
BRUSH_REWARD_WEIGHTS = {**DRAWING_REWARD_WEIGHTS, "color": 0.4}


def _write_onnx(path: Path, metadata: dict, input_width: int, output_width: int) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["observations"], ["actions"])],
        "drawing-live-lab",
        [
            helper.make_tensor_value_info(
                "observations", TensorProto.FLOAT, ["batch", input_width]
            )
        ],
        [
            helper.make_tensor_value_info(
                "actions", TensorProto.FLOAT, ["batch", output_width]
            )
        ],
    )
    model = helper.make_model(graph)
    helper.set_model_props(model, {"rlx_metadata": json.dumps(metadata)})
    path.parent.mkdir(parents=True)
    onnx.save(model, path)


def _drawing_metadata() -> dict:
    return {
        "recipe": "drawing",
        "actuator": "xml",
        "recipe_options": {},
        "reward_weights": DRAWING_REWARD_WEIGHTS,
        "max_episode_s": 32.0,
        "contract_version": "microduck-drawing-v1",
        "observation_dim": 83,
        "action_dim": 15,
    }


def _brush_metadata() -> dict:
    return {
        "recipe": "drawing",
        "actuator": "xml",
        "recipe_options": {},
        "reward_weights": BRUSH_REWARD_WEIGHTS,
        "max_episode_s": 120.0,
        "contract_version": "microduck-brush-v2",
        "observation_dim": 93,
        "action_dim": 15,
    }


def test_old_seven_remain_61_by_14_and_drawing_contracts_are_exact_exceptions(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    assert studio_policies.TRUSTED_RECIPES == {
        "backflip", "dance", "running", "stilts", "swing", "basketball", "bridge"
    }
    drawing = tmp_path / "drawing" / "ink-run" / "policy.onnx"
    _write_onnx(drawing, _drawing_metadata(), 83, 15)
    brush = tmp_path / "drawing" / "brush-run" / "policy.onnx"
    _write_onnx(brush, _brush_metadata(), 93, 15)
    wrong_artifact = tmp_path / "drawing" / "wrong-artifact" / "live.onnx"
    _write_onnx(wrong_artifact, _drawing_metadata(), 83, 15)
    running = tmp_path / "running" / "run" / "policy.onnx"
    _write_onnx(
        running,
        {
            "recipe": "running",
            "actuator": "xml",
            "recipe_options": {},
            "reward_weights": {},
            "max_episode_s": 12.0,
        },
        83,
        15,
    )

    assert studio_policies.studio_metadata(drawing)["contract_version"] == (
        "microduck-drawing-v1"
    )
    assert studio_policies.studio_metadata(brush)["contract_version"] == (
        "microduck-brush-v2"
    )
    assert not studio_policies.is_studio_policy_path(wrong_artifact)
    with pytest.raises(ValueError, match="input width must be 61"):
        studio_policies.studio_metadata(running)


def test_drawing_discovery_fails_closed_and_selects_env_from_contract_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    valid = tmp_path / "drawing" / "valid" / "policy.onnx"
    stale = tmp_path / "drawing" / "stale" / "policy.onnx"
    _write_onnx(valid, _drawing_metadata(), 83, 15)
    stale_metadata = _drawing_metadata()
    stale_metadata["contract_version"] = "microduck-drawing-v0"
    _write_onnx(stale, stale_metadata, 83, 15)
    brush = tmp_path / "drawing" / "brush" / "policy.onnx"
    _write_onnx(brush, _brush_metadata(), 93, 15)
    captured = []

    class DrawingEnv:
        def __init__(self, **kwargs):
            captured.append(("pencil", kwargs))

    class BrushEnv:
        def __init__(self, **kwargs):
            captured.append(("brush", kwargs))

    def drawing_module(contract_version):
        if contract_version == "microduck-drawing-v1":
            return SimpleNamespace(
                REWARD_WEIGHTS=DRAWING_REWARD_WEIGHTS,
                DrawingEnv=DrawingEnv,
            )
        if contract_version == "microduck-brush-v2":
            return SimpleNamespace(
                REWARD_WEIGHTS=BRUSH_REWARD_WEIGHTS,
                BrushEnv=BrushEnv,
            )
        raise ValueError("unsupported drawing contract")

    monkeypatch.setattr(studio_policies, "_drawing_module", drawing_module)

    policies = studio_policies.discover_studio_policies()
    assert {policy["id"] for policy in policies} == {
        "studio:drawing/brush",
        "studio:drawing/valid",
    }
    env = studio_policies.make_studio_env(valid, seed=9)
    brush_env = studio_policies.make_studio_env(brush, seed=10)
    assert env.studio_recipe == "drawing"
    assert brush_env.studio_recipe == "drawing"
    assert captured == [
        ("pencil", {"seed": 9, "max_episode_s": 32.0, "assistance": 0}),
        ("brush", {"seed": 10, "max_episode_s": 120.0, "assistance": 0}),
    ]


@pytest.mark.parametrize(
    ("metadata", "input_width", "output_width", "message"),
    [
        (_drawing_metadata(), 93, 15, "input width must be 83"),
        (_brush_metadata(), 83, 15, "input width must be 93"),
        (_brush_metadata(), 93, 14, "output width must be 15"),
        (
            {**_brush_metadata(), "reward_weights": DRAWING_REWARD_WEIGHTS},
            93,
            15,
            "requires reward_weights=",
        ),
        (
            {**_brush_metadata(), "max_episode_s": 119.0},
            93,
            15,
            "requires max_episode_s=120.0",
        ),
        (
            {**_brush_metadata(), "contract_version": "microduck-drawing-v1"},
            93,
            15,
            "input width must be 83",
        ),
    ],
)
def test_drawing_contract_rejects_wrong_metadata_or_shape(
    tmp_path, monkeypatch, metadata, input_width, output_width, message
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy = tmp_path / "drawing" / "invalid" / "policy.onnx"
    _write_onnx(policy, metadata, input_width, output_width)
    with pytest.raises(ValueError, match=message):
        studio_policies.studio_metadata(policy)


class _PayloadEnv:
    studio_recipe = "drawing"

    def __init__(self):
        self.points = []
        self.twist_cmd = np.zeros(3, np.float32)

    def drawing_payload(self):
        return {
            "points": self.points,
            "contract": "microduck-drawing-v1",
            "assistance": 0,
        }

    def reset(self):
        self.points.clear()
        return np.zeros(83, np.float32), {}


def test_drawing_payload_is_per_duck_bounded_and_cleared_by_reset():
    first = object.__new__(viz_server.Duck)
    second = object.__new__(viz_server.Duck)
    first.env = _PayloadEnv()
    second.env = _PayloadEnv()
    first.env.points.extend([[0.16, 0.0, 0.2, 1]] * 6002)
    second.env.points.append([0.16, 0.1, 0.2, 1])

    payload = first.drawing_payload()
    assert len(payload["points"]) == 6000
    assert payload["points"] is not first.env.points
    assert second.drawing_payload()["points"] == [[0.16, 0.1, 0.2, 1]]

    first.env.reset()
    assert first.drawing_payload()["points"] == []
    assert not viz_server._uses_global_regroup(first)
    assert viz_server._uses_global_regroup(
        SimpleNamespace(env=SimpleNamespace(studio_recipe="running"))
    )


def test_live_brush_payload_preserves_aligned_color_and_palette_fields():
    data = {"points": [[.1597, 0, .22, 1]] * 6002, "colors": [0, 1] + [2] * 6000,
            "palette": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
            "brush_radius": .001, "contract": "microduck-brush-v2", "assistance": 0}
    duck = object.__new__(viz_server.Duck)
    duck.env = SimpleNamespace(drawing_payload=lambda: data)
    payload = duck.drawing_payload()
    assert len(payload["points"]) == len(payload["colors"]) == 6000
    assert payload["colors"] == [2] * 6000
    assert payload["palette"] == data["palette"]
    assert payload["brush_radius"] == .001
    data["colors"].pop()
    assert duck.drawing_payload() is None


def test_drawing_tick_does_not_require_walking_speed_helpers():
    env = _PayloadEnv()
    env.step_count = 0

    def step(action):
        assert np.asarray(action).shape == (15,)
        env.step_count += 1
        return np.zeros(83, np.float32), 0.0, False, False, {}

    env.step = step
    duck = object.__new__(viz_server.Duck)
    duck.env = env
    duck.obs = np.zeros(83, np.float32)
    duck.infer = lambda observation: np.zeros(15, np.float32)
    duck.handoff_infer = None
    duck.handed = False
    duck.reward_ema = 0.0
    duck.speed_hist = []

    duck.tick()

    assert env.step_count == 1
    assert duck.speed_hist == []
