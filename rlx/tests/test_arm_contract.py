from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from stable_baselines3 import PPO

from rlx.environments import arm


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "render_microduck_arm_tests", EXAMPLES / "render_microduck_arm.py"
)
renderer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(renderer)


@pytest.mark.parametrize("case_id", arm.CASES)
def test_reset_contract_is_finite(case_id):
    env = arm.ArmEnv(case_id)
    try:
        observation, info = env.reset(seed=17)
        expected = (116, 12, "md-dualarm-table-v1") if case_id.startswith(
            "arms-"
        ) else (66, 6, "md-arm-table-v1")
        assert observation.shape == (expected[0],)
        assert env.action_space.shape == (expected[1],)
        assert env.contract == expected[2]
        assert np.isfinite(observation).all()
        assert info["simulation_only"] is True
    finally:
        env.close()


@pytest.mark.parametrize("case_id", arm.CASES)
def test_reset_is_exact_for_seed_and_options(case_id):
    options = {
        "position_noise": 0.0015,
        "mass_scale": 0.9,
        "friction_scale": 1.1,
    }
    first = arm.ArmEnv(case_id)
    second = arm.ArmEnv(case_id)
    try:
        observation_a, _ = first.reset(seed=883, options=options)
        observation_b, _ = second.reset(seed=883, options=options)
        assert np.array_equal(observation_a, observation_b)
        assert first.state() == second.state()
        observation_c, _ = second.reset(seed=884, options=options)
        assert not np.array_equal(observation_a, observation_c)
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("bad_action", [
    np.zeros(5),
    np.zeros(7),
    np.array([0, 0, 0, 0, 0, np.nan]),
    np.array([0, 0, 0, 0, 0, np.inf]),
])
def test_invalid_actions_are_rejected(bad_action):
    env = arm.ArmEnv("arm-reach-v1")
    try:
        with pytest.raises(ValueError, match="finite.*contract"):
            env.step(bad_action)
    finally:
        env.close()


@pytest.mark.parametrize(
    "options",
    [
        {"mass_scale": 0.49},
        {"mass_scale": 1.51},
        {"friction_scale": 0.49},
        {"friction_scale": 1.51},
    ],
)
def test_invalid_declared_reset_envelope_is_rejected(options):
    env = arm.ArmEnv("arm-reach-v1")
    try:
        with pytest.raises(ValueError, match="outside declared"):
            env.reset(seed=1, options=options)
    finally:
        env.close()


@pytest.mark.parametrize("case_id", arm.CASES)
def test_object_is_free_and_never_equality_attached(case_id):
    env = arm.ArmEnv(case_id)
    try:
        object_joint = env.model.joint("object_free")
        assert object_joint.type == mujoco.mjtJoint.mjJNT_FREE
        for equality_id in range(env.model.neq):
            assert env.model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_JOINT
            assert not (
                env.model.eq_obj1id[equality_id] == object_joint.id
                or env.model.eq_obj2id[equality_id] == object_joint.id
            )
        assert all("object" not in name for name in env.xml.split("<equality>")[1].split("</equality>")[0].split())
    finally:
        env.close()


@pytest.mark.parametrize("case_id", arm.CASES)
def test_only_intentional_serial_wrist_collision_pairs_are_excluded(case_id):
    root = ET.fromstring(arm.make_xml(case_id))
    excluded = {
        (node.attrib["body1"], node.attrib["body2"])
        for node in root.findall("./contact/exclude")
    }
    expected = {("left_elbow", "left_hand")}
    if case_id.startswith("arms-"):
        expected.add(("right_elbow", "right_hand"))
    assert excluded == expected


@pytest.mark.parametrize(
    ("case_id", "expected_mass"),
    [
        ("arm-reach-v1", 0.020),
        ("arm-pick-place-v1", 0.020),
        ("arm-relocate-v1", 0.020),
        ("arm-carry-v1", 0.020),
        ("arms-handover-v1", 0.005),
        ("arms-co-carry-v1", 0.012),
    ],
)
def test_object_mass_matches_case_contract(case_id, expected_mass):
    env = arm.ArmEnv(case_id)
    try:
        assert env.model.body_mass[env.object_id] == pytest.approx(expected_mass)
        if case_id == "arms-co-carry-v1":
            payload = env.model.body("payload").id
            assert env.model.body_mass[payload] == pytest.approx(0.005)
            assert env.model.body_mass[env.object_id] + env.model.body_mass[
                payload
            ] == pytest.approx(0.017)
    finally:
        env.close()


def test_cli_uses_native_case_choices_and_exact_options(tmp_path):
    args = renderer.parse_args([
        "--case", arm.CASES[-1],
        "--teacher",
        "--output", str(tmp_path),
        "--seed", "73",
        "--position-noise", ".002",
        "--mass-scale", ".9",
        "--friction-scale", "1.1",
    ])
    assert args.case == arm.CASES[-1]
    assert args.seed == 73
    assert args.position_noise == pytest.approx(0.002)
    assert args.mass_scale == pytest.approx(0.9)
    assert args.friction_scale == pytest.approx(1.1)
    with pytest.raises(SystemExit):
        renderer.parse_args([
            "--case", "not-a-case", "--teacher", "--output", str(tmp_path)
        ])


def test_receipt_root_contract_and_hashes(tmp_path):
    video = tmp_path / renderer.VIDEO_NAME
    sheet = tmp_path / renderer.CONTACT_SHEET_NAME
    video.write_bytes(b"video")
    sheet.write_bytes(b"sheet")
    episode = {
        "contract": "md-arm-table-v1",
        "metrics": {"passed": True},
    }
    receipt = renderer.build_receipt(
        output=tmp_path,
        case_id="arm-reach-v1",
        controller="teacher",
        checkpoint=None,
        checkpoint_provenance=None,
        seed=7,
        options={"position_noise": 0.001},
        episode=episode,
        export=None,
    )
    assert receipt["video_sha256"] == renderer.sha256_file(video)
    assert receipt["source_hashes"] == {
        "environment": renderer.sha256_file(arm.__file__),
        "pipeline": renderer.sha256_file(renderer.pipeline.__file__),
    }
    assert receipt["controller"] == "teacher"
    assert receipt["checkpoint"] is None
    assert receipt["episode_sha256"] == renderer.canonical_sha256(episode)
    assert receipt["media_hashes"][renderer.CONTACT_SHEET_NAME] == renderer.sha256_file(sheet)


def test_ppo_receipt_names_composite_controller_and_checkpoint(tmp_path):
    video = tmp_path / renderer.VIDEO_NAME
    sheet = tmp_path / renderer.CONTACT_SHEET_NAME
    checkpoint = tmp_path / "ppo-residual.zip"
    video.write_bytes(b"video")
    sheet.write_bytes(b"sheet")
    checkpoint.write_bytes(b"checkpoint")
    provenance = {
        "path": str((tmp_path / "training.json").resolve()),
        "sha256": "metadata-digest",
        "case_id": "arm-reach-v1",
        "model_sha256": arm.model_hash("arm-reach-v1"),
        "checkpoint_sha256": renderer.sha256_file(checkpoint),
        "source_hashes": renderer.source_hashes(),
    }
    receipt = renderer.build_receipt(
        output=tmp_path,
        case_id="arm-reach-v1",
        controller="ppo_residual",
        checkpoint=checkpoint,
        checkpoint_provenance=provenance,
        seed=9,
        options={},
        episode={"contract": "md-arm-table-v1"},
        export=None,
    )
    assert receipt["controller"] == "ppo_residual"
    assert receipt["controller_description"] == (
        "composite authored IK/FSM + PPO residual"
    )
    assert receipt["checkpoint"] == str(checkpoint.resolve())
    assert receipt["checkpoint_sha256"] == renderer.sha256_file(checkpoint)
    assert receipt["checkpoint_metadata"] == provenance


def test_checkpoint_metadata_coordinates_case_model_sources_and_checkpoint(
    tmp_path,
):
    checkpoint = tmp_path / "ppo-residual.zip"
    metadata_path = tmp_path / "training.json"
    checkpoint.write_bytes(b"checkpoint")
    metadata = {
        "case_id": "arm-reach-v1",
        "model_sha256": arm.model_hash("arm-reach-v1"),
        "checkpoint_sha256": renderer.sha256_file(checkpoint),
        "source_hashes": renderer.source_hashes(),
    }
    metadata_path.write_text(json.dumps(metadata))
    verified = renderer.validate_checkpoint_metadata(
        checkpoint, "arm-reach-v1"
    )
    assert verified["case_id"] == "arm-reach-v1"
    assert verified["model_sha256"] == metadata["model_sha256"]
    assert verified["checkpoint_sha256"] == metadata["checkpoint_sha256"]
    assert verified["source_hashes"] == metadata["source_hashes"]
    for key, replacement in (
        ("case_id", "arm-pick-place-v1"),
        ("model_sha256", "0" * 64),
        ("checkpoint_sha256", "1" * 64),
        ("source_hashes", {"environment": "bad", "pipeline": "bad"}),
    ):
        damaged = metadata | {key: replacement}
        metadata_path.write_text(json.dumps(damaged))
        with pytest.raises(ValueError, match="provenance mismatch"):
            renderer.validate_checkpoint_metadata(
                checkpoint, "arm-reach-v1"
            )


def test_residual_onnx_contract_and_float32_parity(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    env = arm.ArmEnv("arm-reach-v1")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=8,
            policy_kwargs={"net_arch": [16]},
            seed=5,
            verbose=0,
        )
        output = tmp_path / renderer.ONNX_NAME
        result = renderer.export_residual_actor(
            model, "arm-reach-v1", output
        )
    finally:
        env.close()
    graph = onnx.load(output)
    properties = {item.key: json.loads(item.value) for item in graph.metadata_props}
    metadata = properties["rlx_metadata"]
    assert graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    assert graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value == 66
    assert graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value == 6
    assert metadata["action_semantics"] == "residual"
    assert metadata["residual_scale"] == pytest.approx(0.02)
    assert metadata["contract"].endswith("-ppo-residual-v1")
    assert metadata["stock_deployment_compatible"] is False
    assert result["native_onnx_float32_parity"]["max_abs_error"] <= 1e-4
