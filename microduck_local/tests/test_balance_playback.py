from types import SimpleNamespace
import json

import numpy as np
import pytest

from microduck_local import studio_policies
from microduck_local import viz_server


class RecurrentSession:
    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in ("obs", "h_in", "c_in")]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("actions", "h_out", "c_out")]

    def run(self, outputs, feeds):
        assert feeds["obs"].shape == (1, 61)
        assert feeds["obs"].dtype == np.float32
        return np.full((1, 14), feeds["h_in"][0, 0, 0]), feeds["h_in"] + 1, feeds["c_in"] + 2


def test_recurrent_state_is_per_duck_and_resettable():
    first = viz_server._RecurrentOnnxInfer(RecurrentSession())
    second = viz_server._RecurrentOnnxInfer(RecurrentSession())
    observation = np.zeros(61)
    np.testing.assert_array_equal(first(observation), np.zeros(14))
    np.testing.assert_array_equal(first(observation), np.ones(14))
    np.testing.assert_array_equal(second(observation), np.zeros(14))
    first.reset()
    np.testing.assert_array_equal(first(observation), np.zeros(14))


def test_duck_reset_clears_recurrent_state_but_command_does_not():
    inference = viz_server._RecurrentOnnxInfer(RecurrentSession())
    inference(np.zeros(61))
    commands = []
    environment = SimpleNamespace(
        studio_recipe="basketball", twist_cmd=np.zeros(3),
        set_command=lambda command: commands.append(command),
        reset=lambda: (np.zeros(61), {}),
    )
    duck = object.__new__(viz_server.Duck)
    duck.env = environment
    duck.infer = inference
    duck.speed_hist = []
    duck.set_cmd(np.array([0.1, 0, 0]))
    assert len(commands) == 1
    assert inference.hidden[0, 0, 0] == 1
    duck.reset()
    assert inference.hidden[0, 0, 0] == 0


def test_seven_trusted_scene_recipes():
    assert studio_policies.TRUSTED_RECIPES == {
        "dance", "swing", "running", "stilts", "backflip", "basketball", "bridge",
    }
    for recipe in ("basketball", "bridge"):
        assert studio_policies._recipe_kwargs({"recipe": recipe, "recipe_options": {}}) == {}
        with pytest.raises(ValueError, match="unsupported"):
            studio_policies._recipe_kwargs({"recipe": recipe, "recipe_options": {"hold": 1}})
        assert not viz_server.is_trick_duck(SimpleNamespace(env=SimpleNamespace(studio_recipe=recipe)))


def test_cached_policy_allocates_independent_recurrent_states(tmp_path, monkeypatch):
    policy = tmp_path / "basketball.onnx"
    policy.write_bytes(b"fake-session")
    monkeypatch.setattr(viz_server, "_policy_entry", lambda _: {"path": str(policy)})
    monkeypatch.setattr(viz_server, "_onnx_infer", lambda _: viz_server._RecurrentOnnxInfer(RecurrentSession()))
    monkeypatch.setattr(viz_server, "_infer_cache", {})
    first = viz_server.load_policy_infer("basketball")
    second = viz_server.load_policy_infer("basketball")
    assert first is not second
    assert first.session is second.session
    first(np.zeros(61))
    assert second.hidden[0, 0, 0] == 0


def test_studio_accepts_bam_recurrent_contract_only_for_basketball(tmp_path, monkeypatch):
    import onnx
    from onnx import TensorProto, helper

    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy = tmp_path / "basketball" / "balance" / "policy.onnx"
    policy.parent.mkdir(parents=True)
    graph = helper.make_graph([], "recurrent-contract", [
        helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 61]),
        helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 1, 256]),
        helper.make_tensor_value_info("c_in", TensorProto.FLOAT, [1, 1, 256]),
    ], [
        helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14]),
        helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 1, 256]),
        helper.make_tensor_value_info("c_out", TensorProto.FLOAT, [1, 1, 256]),
    ])
    model = helper.make_model(graph)
    metadata = {"recipe": "basketball", "actuator": "bam", "recipe_options": {},
                "reward_weights": {}, "max_episode_s": 60}
    helper.set_model_props(model, {"rlx_metadata": json.dumps(metadata)})
    onnx.save(model, policy)
    assert studio_policies.studio_metadata(policy) == metadata
    model.graph.input[1].type.tensor_type.shape.dim[2].dim_value = 128
    with pytest.raises(ValueError, match="recurrent state"):
        studio_policies._validate_onnx_contract(model, policy)
    model.graph.input[1].type.tensor_type.shape.dim[2].dim_value = 256
    with pytest.raises(ValueError, match="exactly one input"):
        studio_policies._validate_onnx_contract(model, tmp_path / "bridge" / "run" / "bridge.onnx")


def test_cli_policy_startup_selects_its_custom_scene(monkeypatch):
    created = []
    monkeypatch.setattr(viz_server, "_onnx_infer", lambda _: lambda observation: observation)
    monkeypatch.setattr(viz_server, "env_kwargs_for_policy_path", lambda path: {"studio_policy_path": path})
    monkeypatch.setattr(viz_server, "Duck", lambda *args, **kwargs: created.append(kwargs))
    viz_server.build_ducks(SimpleNamespace(checkpoints=None, policies=["bridge.onnx", "policy.onnx"]))
    assert created[0]["env_kwargs"] == {"studio_policy_path": "bridge.onnx"}
    assert created[1]["env_kwargs"] == {"studio_policy_path": "policy.onnx"}


def test_new_support_geometry_is_visible_in_streamed_scenes():
    from rlx.environments.bridge import bridge_model
    from rlx.environments.basketball import basketball_model

    bridge = bridge_model()
    scene = viz_server.extract_scene(bridge)
    plank = bridge.body("bridge_plank").id
    assert any(primitive["body"] == plank and primitive["type"] == "box" for primitive in scene["primitives"])
    assert len(scene["primitives"]) >= 9
    assert len(scene["tendons"]) == 4
    ball = basketball_model()
    ball_scene = viz_server.extract_scene(ball)
    assert any(geom["body"] == ball.body("basketball").id and geom["mat"] == "basketball_mat" for geom in ball_scene["geoms"])
