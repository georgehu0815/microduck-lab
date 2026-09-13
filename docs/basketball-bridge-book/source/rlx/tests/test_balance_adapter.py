"""Contracts for the Studio-compatible recurrent basketball adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/ppo_microduck_balance.py"


@pytest.fixture()
def adapter():
    name = "ppo_microduck_balance_tests"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_is_command_first_and_clamps_action_delay(adapter):
    args = adapter.parse_args(
        [
            "eval",
            "--recipe",
            "basketball",
            "--policy",
            "policy.onnx",
            "--no-action-delay",
        ]
    )

    assert args.command == "eval"
    assert args.recipe == "basketball"
    assert args.action_delay is False
    assert adapter._adjustments(args) == [
        {
            "option": "backend",
            "requested": "standalone",
            "applied": "standalone",
            "reason": "basketball uses its single-process MuJoCo evaluator",
        },
        {
            "option": "action_delay",
            "requested": False,
            "applied": True,
            "reason": "BasketballEnv and the source policy use inherited action delay",
        },
    ][1:]

    with pytest.raises(SystemExit):
        adapter.parse_args(["--recipe", "basketball", "eval"])


def test_train_routes_complete_recurrent_batches_and_writes_studio_metadata(
    adapter, monkeypatch, tmp_path
):
    source = tmp_path / "source.pt"
    source.write_bytes(b"source-checkpoint")
    source.with_name("policy.onnx").write_bytes(b"source-policy")
    output = tmp_path / "studio-run"
    captured = {}

    class FakeLearner:
        @staticmethod
        def train(args):
            captured["args"] = args
            args.output.mkdir(parents=True)
            checkpoint = args.output / "checkpoint.pt"
            policy = args.output / "policy.onnx"
            checkpoint.write_bytes(b"trained-checkpoint")
            policy.write_bytes(b"trained-policy")
            return {
                "checkpoint": str(checkpoint),
                "onnx": str(policy),
                "local_steps": args.num_envs * args.steps * args.updates,
                "onnx_parity": {"max_abs_error": 1e-6},
                "updates": [{"update": 1, "loss": 1.0}],
                "continuation_kind": "test",
                "full_upstream_resume": False,
            }

    def annotate(policy, checkpoint, metadata, parity):
        captured["metadata"] = metadata
        captured["parity"] = parity
        policy.write_bytes(policy.read_bytes() + b"-annotated")

    monkeypatch.setattr(adapter, "_learner", lambda: FakeLearner)
    monkeypatch.setattr(adapter, "_append_onnx_metadata", annotate)
    args = adapter.parse_args(
        [
            "train",
            "--recipe",
            "basketball",
            "--output-dir",
            str(output),
            "--init-from",
            str(source),
            "--num-envs",
            "2",
            "--num-steps",
            "2",
            "--total-timesteps",
            "5",
            "--max-episode-s",
            "1",
            "--no-action-delay",
        ]
    )

    result = adapter.train(args)

    assert captured["args"].updates == 2
    assert result["steps"] == 8
    assert result["requested_steps"] == 5
    assert captured["metadata"]["recipe"] == "basketball"
    assert captured["metadata"]["actuator"] == "bam"
    assert captured["metadata"]["max_episode_s"] == 10.0
    assert captured["metadata"]["reward_weights"] == {}
    assert captured["metadata"]["recipe_options"] == {}
    assert captured["metadata"]["randomization"]["action_delay"] is True
    assert Path(result["checkpoint"]).name == "checkpoint.pt"
    assert Path(result["onnx"]).name == "policy.onnx"
    assert Path(result["metadata"]).is_file()
    assert Path(result["result_json"]).is_file()
    assert {row["option"] for row in result["adapter_adjustments"]} == {
        "max_episode_s",
        "action_delay",
        "total_timesteps",
    }


def test_import_run_copies_then_annotates_without_mutating_source(
    adapter, monkeypatch, tmp_path
):
    source = tmp_path / "candidate"
    output = tmp_path / "studio"
    source.mkdir()
    (source / "checkpoint.pt").write_bytes(b"checkpoint")
    (source / "policy.onnx").write_bytes(b"policy")
    (source / "summary.json").write_text(
        json.dumps(
            {
                "local_steps": 102400,
                "settings": {"seed": 42, "num_envs": 8},
            }
        ),
        encoding="utf-8",
    )
    original = {
        path.name: path.read_bytes()
        for path in source.iterdir()
    }
    parity_calls = []

    def parity(checkpoint, policy):
        parity_calls.append(policy.read_bytes())
        return {"max_abs_error": 1e-6, "steps": 40, "reset_at": 20}

    def annotate(policy, checkpoint, metadata, parity_result):
        assert policy.parent == output
        assert metadata["randomization"]["action_delay"] is True
        assert parity_result["max_abs_error"] < adapter.PARITY_LIMIT
        policy.write_bytes(policy.read_bytes() + b"-metadata")

    monkeypatch.setattr(adapter, "_verify_local_parity", parity)
    monkeypatch.setattr(adapter, "_append_onnx_metadata", annotate)

    result = adapter.import_run(source, output)

    assert parity_calls == [b"policy", b"policy-metadata"]
    assert {
        path.name: path.read_bytes()
        for path in source.iterdir()
    } == original
    assert (output / "policy.onnx").read_bytes() == b"policy-metadata"
    sidecar = json.loads((output / "checkpoint.pt.json").read_text())
    metadata = sidecar["metadata"]
    assert metadata["recipe"] == "basketball"
    assert metadata["actuator"] == "bam"
    assert metadata["max_episode_s"] == 10.0
    assert metadata["recipe_options"] == {}
    assert metadata["reward_weights"] == {}
    assert metadata["randomization"]["action_delay"] is True
    assert result["steps"] == 102400
    assert Path(result["result_json"]).is_file()


def _trial(command, *, rolling):
    commanded = command[0] != 0
    passed = rolling if commanded else True
    return {
        "seed": 7,
        "command": list(command),
        "measured_steps": 10,
        "automatic_resets": 0,
        "hold_zero": True,
        "controlled_rolling": passed if commanded else False,
        "sustained_balance": True,
        "failures": [] if passed else ["command-directed ball progress below target"],
    }


def _fake_evaluator(tmp_path, *, rolling):
    evaluator_file = tmp_path / "evaluator.py"
    evaluator_file.write_text("# evaluator\n", encoding="utf-8")

    class FakePolicy:
        def __init__(self, path):
            self.path = path

    def run_trial(**kwargs):
        trial = _trial(kwargs["command"], rolling=rolling)
        trial["seed"] = kwargs["seed"]
        if kwargs["render_path"] is not None:
            video = kwargs["render_path"] / "rollout.mp4"
            sheet = kwargs["render_path"] / "contact-sheet.png"
            video.write_bytes(b"video")
            sheet.write_bytes(b"sheet")
            trial["render"] = {"video": str(video), "contact_sheet": str(sheet)}
        return trial

    def summarize(trials):
        balance = [row for row in trials if row["command"][0] == 0]
        rolling_rows = [row for row in trials if row["command"][0] != 0]
        cases = [
            {
                "classification": "zero_command_balance",
                "passed": all(row["sustained_balance"] for row in balance),
            },
            {
                "classification": "commanded_rolling",
                "passed": all(row["controlled_rolling"] for row in rolling_rows),
            },
        ]
        return {"passed": all(case["passed"] for case in cases), "cases": cases}

    return SimpleNamespace(
        __file__=str(evaluator_file),
        CTRL_DT=0.02,
        DEFAULT_SECONDS=60.0,
        RecurrentOnnxPolicy=FakePolicy,
        criteria_for=lambda seconds, command: None,
        run_trial=run_trial,
        summarize_trials=summarize,
    )


@pytest.mark.parametrize(
    ("rolling", "expected_success", "expected_status"),
    [(False, False, "failed"), (True, False, "failed")],
)
def test_eval_cannot_claim_full_steering_from_forward_only_trials(
    adapter, monkeypatch, tmp_path, rolling, expected_success, expected_status
):
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"policy")
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        adapter,
        "_evaluator",
        lambda: _fake_evaluator(tmp_path, rolling=rolling),
    )
    args = adapter.parse_args(
        [
            "eval",
            "--recipe",
            "basketball",
            "--policy",
            str(policy),
            "--output-dir",
            str(output),
            "--num-envs",
            "2",
            "--eval-steps",
            "10",
            "--no-action-delay",
        ]
    )

    result = adapter.evaluate(args)

    assert result["passed"] is expected_success
    assert result["success"] is expected_success
    assert result["skill_status"] == expected_status
    assert result["task_assessment"]["rolling_passed"] is expected_success
    assert result["task_assessment"]["forward_rolling_passed"] is rolling
    assert result["task_assessment"]["steering_coverage"] == {"forward": True, "lateral": False, "yaw": False}
    assert result["task_assessment"]["hold_zero"] is True
    assert result["report"]["protocol"]["hold"] == 0
    assert result["report"]["protocol"]["action_delay"] is True
    assert result["evaluation"]["environment"]["action_delay"] is True
    assert Path(result["evaluation_path"]).is_file()
    assert Path(result["result_json"]).is_file()


def test_render_writes_expected_evidence_and_provenance(
    adapter, monkeypatch, tmp_path
):
    policy = tmp_path / "policy.onnx"
    checkpoint = tmp_path / "checkpoint.pt"
    policy.write_bytes(b"policy")
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "render"
    monkeypatch.setattr(
        adapter,
        "_evaluator",
        lambda: _fake_evaluator(tmp_path, rolling=False),
    )
    args = adapter.parse_args(
        [
            "render",
            "--recipe",
            "basketball",
            "--policy",
            str(policy),
            "--output",
            str(output),
            "--render-seconds",
            "1",
            "--no-action-delay",
        ]
    )

    result = adapter.render(args)

    assert (output / "rollout.mp4").is_file()
    assert (output / "contact-sheet.png").is_file()
    assert (output / "render.json").is_file()
    assert (output / "result.json").is_file()
    assert result["source_sha256"] == adapter.sha256_file(policy)
    assert result["checkpoint_sha256"] == adapter.sha256_file(checkpoint)
    assert result["environment"]["action_delay"] is True
    assert result["task_assessment"]["success"] is False


@pytest.mark.parametrize("hold", ["-0.1", "1.1", "nan", "inf"])
def test_training_rejects_invalid_hold(adapter, hold):
    with pytest.raises(SystemExit):
        adapter.parse_args(["train", "--recipe", "basketball", "--hold", hold])


def test_hold_curriculum_is_training_only(adapter):
    pytest.importorskip("mujoco")
    pytest.importorskip("microduck_local")
    from rlx.environments.basketball import HOLD_LEVELS

    for hold in HOLD_LEVELS:
        args = adapter.parse_args([
            "train", "--recipe", "basketball", "--hold", str(hold), "--curriculum",
        ])
        assert args.hold == hold
        assert args.curriculum is True
    with pytest.raises(SystemExit):
        adapter.parse_args([
            "train", "--recipe", "basketball", "--hold", "0.4", "--curriculum",
        ])
    for command in ("eval", "render"):
        with pytest.raises(SystemExit):
            adapter.parse_args([command, "--recipe", "basketball", "--hold", "0.5"])


@pytest.fixture(scope="module")
def local_candidate(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from rlx.models.basketball import (
        BasketballActor, LOCAL_CHECKPOINT_FORMAT, export_recurrent_onnx,
    )

    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("local-candidate")
    actor = BasketballActor()
    torch.save({
        "format": LOCAL_CHECKPOINT_FORMAT,
        "actor_state_dict": actor.state_dict(),
        "normalization_frozen": True,
        "actor_observation_dim": 61,
        "critic_observation_dim": 61,
        "local_steps": 16,
    }, directory / "checkpoint.pt")
    export_recurrent_onnx(actor, directory / "policy.onnx")
    return directory


def test_local_loader_requires_opt_in_and_preserves_actor(local_candidate):
    import torch
    from rlx.models.basketball import load_source_actor, sha256_file

    checkpoint = local_candidate / "checkpoint.pt"
    with pytest.raises(ValueError, match="unexpected basketball checkpoint SHA256"):
        load_source_actor(checkpoint)
    actor, metadata = load_source_actor(checkpoint, allow_local=True, initial_std=0.02)
    original = torch.load(checkpoint, weights_only=True)["actor_state_dict"]
    for name, value in actor.state_dict().items():
        if name != "distribution.std_param":
            torch.testing.assert_close(value, original[name])
    torch.testing.assert_close(actor.action_std(), torch.full((14,), 0.02))
    assert metadata["source_kind"] == "local_actor_warm_start"
    assert metadata["source_training_steps"] == 16
    assert metadata["source_hashes"]["checkpoint_sha256"] == sha256_file(checkpoint)
    assert metadata["source_onnx_parity"]["max_abs_error"] < 3e-5


@pytest.mark.parametrize("fault", ["format", "normalizer", "dimensions", "steps", "nan", "mismatch", "missing_policy"])
def test_local_loader_rejects_invalid_candidates(local_candidate, tmp_path, fault):
    import shutil
    import torch
    from rlx.models.basketball import load_source_actor

    checkpoint = tmp_path / "checkpoint.pt"
    payload = torch.load(local_candidate / "checkpoint.pt", weights_only=True)
    if fault == "format":
        payload["format"] = "unrecognized"
    elif fault == "normalizer":
        payload["normalization_frozen"] = False
    elif fault == "dimensions":
        payload["actor_observation_dim"] = 62
    elif fault == "steps":
        payload["local_steps"] = -1
    elif fault == "nan":
        payload["actor_state_dict"]["mlp.6.bias"][0] = float("nan")
    elif fault == "mismatch":
        payload["actor_state_dict"]["mlp.6.bias"][0] += 10
    torch.save(payload, checkpoint)
    if fault != "missing_policy":
        shutil.copy2(local_candidate / "policy.onnx", tmp_path / "policy.onnx")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_source_actor(checkpoint, allow_local=True)


def test_real_train_in_prepopulated_studio_and_local_continuation(adapter, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("microduck_local")
    if not adapter.DEFAULT_SOURCE_CHECKPOINT.is_file():
        pytest.skip("released basketball artifacts are not installed")
    output = tmp_path / "studio"
    output.mkdir()
    ui_files = {"recipe.json": b'{"owner":"Studio"}', "state.json": b'{"phase":"running"}'}
    for name, contents in ui_files.items():
        (output / name).write_bytes(contents)
    source_hashes = {
        str(path): adapter.sha256_file(path)
        for path in (adapter.DEFAULT_SOURCE_CHECKPOINT, adapter.DEFAULT_SOURCE_CHECKPOINT.with_name("policy.onnx"))
    }
    base = [
        "train", "--recipe", "basketball", "--output-dir", str(output),
        "--checkpoint", str(output / "checkpoint.pt"),
        "--onnx-output", str(output / "policy.onnx"),
        "--num-envs", "1", "--num-steps", "4", "--total-timesteps", "4",
        "--hold", "0.5", "--curriculum",
    ]
    first = adapter.train(adapter.parse_args(base))
    first_checkpoint_hash = adapter.sha256_file(output / "checkpoint.pt")
    first_policy_hash = adapter.sha256_file(output / "policy.onnx")
    second = adapter.train(adapter.parse_args([
        *base, "--init-from", str(output / "checkpoint.pt"),
    ]))

    assert first["steps"] == first["trained_steps"] == 4
    assert second["trained_steps"] == 4
    assert second["steps"] == 8
    assert first["training_directory"] != second["training_directory"]
    assert second["source_sha256"] == first_checkpoint_hash
    assert second["source_files_sha256"][str(output / "policy.onnx")] == first_policy_hash
    assert second["resumed_from"] == str(output / "checkpoint.pt")
    assert second["continuation_kind"] == "actor warm-start with fresh local critic and optimizer"
    assert second["full_upstream_resume"] is False
    assert second["learner_summary"]["source_kind"] == "local_actor_warm_start"
    assert second["learner_summary"]["source_training_steps"] == 4
    assert second["training_assistance"] == {"hold": 0.5, "curriculum": True}
    assert second["learner_summary"]["settings"]["hold"] == 0.5
    assert second["learner_summary"]["settings"]["curriculum"] is True
    for name, contents in ui_files.items():
        assert (output / name).read_bytes() == contents
    for path, digest in source_hashes.items():
        assert adapter.sha256_file(Path(path)) == digest
    archived = Path(second["training_directory"]) / "source"
    assert adapter.sha256_file(archived / "checkpoint.pt") == first_checkpoint_hash
    assert adapter.sha256_file(archived / "policy.onnx") == first_policy_hash
    for result in (first, second):
        assert result["onnx_parity"]["max_abs_error"] < 3e-5
        assert Path(result["learner_summary"]["checkpoint"]).is_file()
        checkpoint = torch.load(result["learner_summary"]["checkpoint"], weights_only=True)
        assert checkpoint["local_steps"] == 4
        assert checkpoint["total_steps"] == result["steps"]
        assert all(float(state["step"]) <= 5 for state in checkpoint["optimizer_state_dict"]["state"].values())
    sidecar = json.loads((output / "checkpoint.pt.json").read_text())
    assert sidecar["metadata"]["steps"] == 8
    assert sidecar["metadata"]["training_assistance"] == {"hold": 0.5, "curriculum": True}
    assert adapter._verify_local_parity(output / "checkpoint.pt", output / "policy.onnx")["max_abs_error"] < 3e-5
