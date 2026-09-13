from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from microduck_local import studio_policies


def write_onnx(
    path: Path,
    metadata: dict | None = None,
    *,
    input_width: int = 61,
    output_width: int = 14,
    tensor_type: int | None = None,
) -> None:
    import onnx
    from onnx import TensorProto, helper

    element_type = TensorProto.FLOAT if tensor_type is None else tensor_type
    graph = helper.make_graph(
        [helper.make_node("Identity", ["observations"], ["actions"])],
        "studio-test",
        [
            helper.make_tensor_value_info(
                "observations", element_type, ["batch", input_width]
            )
        ],
        [helper.make_tensor_value_info("actions", element_type, ["batch", output_width])],
    )
    model = helper.make_model(graph)
    if metadata is not None:
        helper.set_model_props(model, {"rlx_metadata": json.dumps(metadata)})
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def base_metadata(recipe: str, options: dict | None = None) -> dict:
    return {
        "recipe": recipe,
        "actuator": "xml",
        "max_episode_s": 10.0,
        "recipe_options": options or {},
        "reward_weights": {},
    }


def write_sidecar(policy_path: Path, metadata: dict) -> Path:
    checkpoint_path = policy_path.with_suffix(".safetensors")
    sidecar_path = policy_path.with_suffix(".safetensors.json")
    checkpoint_path.write_bytes(b"checkpoint")
    sidecar_path.write_text(json.dumps({"metadata": metadata}))
    policy_time = policy_path.stat().st_mtime_ns + 10_000_000
    os.utime(policy_path, ns=(policy_time, policy_time))
    return sidecar_path


def test_discover_studio_policies_supports_embedded_and_safetensors_metadata(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    embedded_path = tmp_path / "dance" / "dance-run" / "dance.onnx"
    embedded = base_metadata("dance")
    write_onnx(embedded_path, embedded)
    fallback_path = tmp_path / "running" / "fast-run" / "running.onnx"
    write_onnx(fallback_path)
    write_sidecar(
        fallback_path,
        base_metadata("running", {"locomotion_forward_command": 0.75}),
    )
    untrusted_path = tmp_path / "unknown" / "bad-run" / "unknown.onnx"
    write_onnx(untrusted_path, base_metadata("unknown"))
    unsupported_path = tmp_path / "running" / "unsupported-run" / "running.onnx"
    write_onnx(
        unsupported_path,
        base_metadata("running", {"unsupported_option": True}),
    )
    corrupt_path = tmp_path / "dance" / "corrupt-run" / "dance.onnx"
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not onnx")

    policies = studio_policies.discover_studio_policies()

    assert {policy["id"] for policy in policies} == {
        "studio:dance/dance-run",
        "studio:running/fast-run",
    }
    running = next(policy for policy in policies if policy["recipe"] == "running")
    assert running == {
        "group": "studio",
        "id": "studio:running/fast-run",
        "label": "fast-run",
        "path": str(fallback_path.resolve()),
        "mtime": fallback_path.stat().st_mtime,
        "artifact": "running.onnx",
        "recipe": "running",
    }
    assert studio_policies.is_studio_policy_path(fallback_path)
    assert not studio_policies.is_studio_policy_path(untrusted_path)


def test_studio_metadata_rejects_newer_checkpoint_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "running" / "run" / "running.onnx"
    write_onnx(policy_path)
    sidecar_path = write_sidecar(
        policy_path,
        base_metadata("running", {"locomotion_forward_command": 0.5}),
    )
    newer_time = policy_path.stat().st_mtime_ns + 10_000_000
    os.utime(sidecar_path, ns=(newer_time, newer_time))

    with pytest.raises(ValueError, match="re-export the ONNX"):
        studio_policies.studio_metadata(policy_path)


@pytest.mark.parametrize(
    ("input_width", "output_width", "tensor_type", "message"),
    [
        (60, 14, None, "input width must be 61"),
        (61, 13, None, "output width must be 14"),
        (61, 14, 11, "must use float32"),
    ],
)
def test_studio_metadata_rejects_wrong_onnx_contract(
    tmp_path,
    monkeypatch,
    input_width,
    output_width,
    tensor_type,
    message,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "running" / "run" / "running.onnx"
    write_onnx(
        policy_path,
        base_metadata("running"),
        input_width=input_width,
        output_width=output_width,
        tensor_type=tensor_type,
    )

    with pytest.raises(ValueError, match=message):
        studio_policies.studio_metadata(policy_path)
    assert studio_policies.discover_studio_policies() == []


def test_studio_policy_signature_tracks_sidecar_content(tmp_path, monkeypatch):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "running" / "run" / "running.onnx"
    write_onnx(policy_path)
    sidecar_path = write_sidecar(
        policy_path,
        base_metadata("running", {"locomotion_forward_command": 0.5}),
    )
    first = studio_policies.studio_policy_signature(policy_path)
    updated = base_metadata("running", {"locomotion_forward_command": 0.75})
    sidecar_path.write_text(json.dumps({"metadata": updated}))
    older_time = policy_path.stat().st_mtime_ns - 1
    os.utime(sidecar_path, ns=(older_time, older_time))

    second = studio_policies.studio_policy_signature(policy_path)

    assert first != second


def test_studio_metadata_cache_reloads_after_sidecar_replacement(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "running" / "run" / "running.onnx"
    write_onnx(policy_path)
    sidecar_path = write_sidecar(
        policy_path,
        base_metadata("running", {"locomotion_forward_command": 0.5}),
    )
    first = studio_policies.studio_metadata(policy_path)
    first["recipe_options"]["locomotion_forward_command"] = 99.0
    assert (
        studio_policies.studio_metadata(policy_path)["recipe_options"][
            "locomotion_forward_command"
        ]
        == 0.5
    )
    updated = base_metadata("running", {"locomotion_forward_command": 0.75})
    sidecar_path.write_text(json.dumps({"metadata": updated}) + " ")
    older_time = policy_path.stat().st_mtime_ns - 1
    os.utime(sidecar_path, ns=(older_time, older_time))

    reloaded = studio_policies.studio_metadata(policy_path)

    assert reloaded["recipe_options"]["locomotion_forward_command"] == 0.75
    policy_path.unlink()
    with pytest.raises(FileNotFoundError):
        studio_policies.studio_metadata(policy_path)


def test_make_studio_env_preserves_stilt_options_and_nominal_playback(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "stilts" / "tall-run" / "stilts.onnx"
    metadata = base_metadata(
        "stilts",
        {
            "stilt_height_cm": 12.0,
            "stilt_blend": 0.4,
            "stilt_mass_kg": 0.03,
            "locomotion_forward_command": 0.25,
        },
    )
    metadata["max_episode_s"] = 9.0
    metadata["reward_weights"] = {"track_lin_vel": 20.0}
    write_onnx(policy_path, metadata)
    base_env = SimpleNamespace()
    captured = {}

    def factory(recipe, **kwargs):
        captured["recipe"] = recipe
        captured["kwargs"] = kwargs
        return SimpleNamespace(unwrapped=base_env)

    monkeypatch.setattr(studio_policies, "_load_recipe_factory", lambda: factory)

    env = studio_policies.make_studio_env(policy_path, seed=17)

    assert env is base_env
    assert env.studio_recipe == "stilts"
    assert captured == {
        "recipe": "stilts",
        "kwargs": {
            "seed": 17,
            "actuator": "xml",
            "domain_rand": False,
            "obs_noise": False,
            "action_delay": False,
            "random_yaw": False,
            "max_episode_s": 9.0,
            "weight_overrides": {"track_lin_vel": 20.0},
            "stilt_height_cm": 12.0,
            "stilt_blend": 0.4,
            "stilt_mass_kg": 0.03,
            "locomotion_forward_command": 0.25,
        },
    }


def test_make_studio_env_validates_dance_clip_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    clip_path = tmp_path / "clips" / "dance.json"
    clip_path.parent.mkdir()
    clip_path.write_text('{"frames": []}')
    policy_path = tmp_path / "dance" / "dance-run" / "dance.onnx"
    metadata = base_metadata(
        "dance",
        {
            "dance_clip": str(clip_path),
            "dance_clip_sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
            "dance_pose_sigma": 0.2,
        },
    )
    write_onnx(policy_path, metadata)
    captured = {}

    def factory(recipe, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(unwrapped=SimpleNamespace())

    monkeypatch.setattr(studio_policies, "_load_recipe_factory", lambda: factory)
    studio_policies.make_studio_env(policy_path, seed=3)
    assert captured["dance_clip"] == clip_path.resolve()
    clip_path.write_text('{"frames": [1]}')

    with pytest.raises(ValueError, match="clip hash mismatch"):
        studio_policies.make_studio_env(policy_path, seed=3)


def test_make_studio_env_uses_repository_default_dance_clip(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "dance" / "default-dance" / "dance.onnx"
    write_onnx(policy_path, base_metadata("dance"))
    captured = {}

    def factory(recipe, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(unwrapped=SimpleNamespace())

    monkeypatch.setattr(studio_policies, "_load_recipe_factory", lambda: factory)

    env = studio_policies.make_studio_env(policy_path, seed=11)

    assert env.studio_recipe == "dance"
    assert "dance_clip" not in captured
    assert "dance_pose_sigma" not in captured


def test_make_studio_env_guards_backflip_protocol_and_uses_showcase(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(tmp_path))
    policy_path = tmp_path / "backflip" / "flip-run" / "backflip.onnx"
    metadata = base_metadata(
        "backflip",
        {
            "backflip_protocol_version": "protocol-v2",
            "stand_policy_sha256": "stand-hash",
        },
    )
    write_onnx(policy_path, metadata)
    base_env = SimpleNamespace()
    captured = {}

    def factory(recipe, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(unwrapped=base_env)

    monkeypatch.setattr(studio_policies, "_load_recipe_factory", lambda: factory)
    monkeypatch.setattr(
        studio_policies,
        "_validate_backflip_protocol",
        lambda: {
            "backflip_protocol_version": "protocol-v2",
            "stand_policy_sha256": "stand-hash",
        },
    )

    env = studio_policies.make_studio_env(policy_path, seed=5)

    assert env.studio_recipe == "backflip"
    assert captured["backflip_mode"] == "showcase"
    metadata["recipe_options"]["stand_policy_sha256"] = "stale-hash"
    write_onnx(policy_path, metadata)
    with pytest.raises(ValueError, match="stand_policy_sha256"):
        studio_policies.make_studio_env(policy_path, seed=5)


@pytest.mark.parametrize(
    "recipe",
    ["backflip", "dance", "running", "stilts", "swing"],
)
def test_real_exported_scenario_smoke(recipe, monkeypatch):
    root = Path(__file__).resolve().parents[2] / "rlx" / "runs" / "studio"
    if not root.is_dir():
        pytest.skip("real Studio exports are not present")
    monkeypatch.setenv("MICRODUCK_STUDIO_RUNS_DIR", str(root))
    policy = next(
        (
            item
            for item in studio_policies.discover_studio_policies()
            if item["recipe"] == recipe
        ),
        None,
    )
    if policy is None:
        pytest.skip(f"no valid real Studio {recipe} export is present")
    policy_path = Path(policy["path"])

    metadata = studio_policies.studio_metadata(policy_path)
    env = studio_policies.make_studio_env(policy_path, seed=7)
    try:
        observation, _ = env.reset(seed=7)
        assert metadata["recipe"] == recipe
        assert observation.shape == (61,)
        assert env.studio_recipe == recipe
    finally:
        env.close()
