from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts/replay_tennis_evidence.py"
SPEC = importlib.util.spec_from_file_location("tennis_action_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _telemetry(actions=(0.0, 0.25), *, terminal_step=2):
    records = []
    for step, value in enumerate(actions, start=1):
        done = step == terminal_step
        records.append(
            {
                "step": step,
                "action": [value] * 15,
                "time_s": step * 0.02,
                "ball_position_m": [value, 0.0, 0.03],
                "joint_positions_rad": [value] * 15,
                "phase": "success" if done else "approach",
                "done": done,
                "success": done,
                "failure_reason": None,
            }
        )
    return records


def _result(steps=2):
    return {
        "variant": "nominal",
        "candidate": "A",
        "seed": 4,
        "controller": "teacher",
        "success": True,
        "failure_reason": None,
        "steps": steps,
        "max_steps": 10,
        "gripper": "wide_candidate",
        "release_mode": "half_height",
    }


def _source_context():
    return {
        "evaluation_sha256": "1" * 64,
        "result_sha256": "2" * 64,
        "telemetry_sha256": "3" * 64,
        "snapshot_hashes": {"tennis_return.py": "4" * 64},
    }


class FakeMonitor:
    def __init__(self):
        self.success = False
        self.failure = None
        self.phase = "approach"

    def advance(self, sample, duration):
        return None


class FakeRobot:
    def __init__(self):
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.substeps = 1


class FakeEnv:
    early_done = False

    def __init__(self, **kwargs):
        self.monitor = FakeMonitor()
        self.robot = FakeRobot()
        self.done = True
        self.steps = 0

    def reset(self, seed):
        self.done = False
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        self.steps += 1
        self.robot.previous_action = np.asarray(action, dtype=np.float32).copy()
        self.monitor.advance({}, 0.002)
        value = float(action[0])
        done = self.steps == (1 if self.early_done else 2)
        self.done = done
        self.monitor.success = done
        self.monitor.phase = "success" if done else "approach"
        return np.zeros(1), done, {
            "time_s": self.steps * 0.02,
            "ball_position_m": [value, 0.0, 0.03],
            "joint_positions_rad": [value] * 15,
            "phase": self.monitor.phase,
            "success": self.monitor.success,
            "failure_reason": None,
        }

    def close(self):
        return None


def _episode_directory(tmp_path, telemetry):
    directory = tmp_path / "nominal-4"
    directory.mkdir()
    (directory / "telemetry.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in telemetry),
        encoding="utf-8",
    )
    return directory


def test_replay_rejects_tampered_action(tmp_path):
    telemetry = _telemetry()
    tampered = copy.deepcopy(telemetry)
    tampered[0]["action"][0] = 0.5
    directory = _episode_directory(tmp_path, tampered)

    with pytest.raises(replay.ReplayError, match="ball_position_m"):
        replay.replay_episode(
            directory,
            _result(),
            tampered,
            _source_context(),
            render=False,
            env_factory=FakeEnv,
        )

    validation = json.loads(
        (directory / "replay-validation.json").read_text()
    )
    assert validation["incomplete"] is True
    assert validation["claimed_video_pass"] is False


def test_replay_rejects_tampered_physical_telemetry(tmp_path):
    telemetry = _telemetry()
    tampered = copy.deepcopy(telemetry)
    tampered[0]["ball_position_m"][0] += 1e-6
    directory = _episode_directory(tmp_path, tampered)

    with pytest.raises(replay.ReplayError, match="ball_position_m"):
        replay.replay_episode(
            directory,
            _result(),
            tampered,
            _source_context(),
            render=False,
            env_factory=FakeEnv,
        )


def test_replay_rejects_early_done(tmp_path):
    telemetry = _telemetry()
    directory = _episode_directory(tmp_path, telemetry)
    env_type = type("EarlyDoneEnv", (FakeEnv,), {"early_done": True})

    with pytest.raises(replay.ReplayError, match="done mismatch"):
        replay.replay_episode(
            directory,
            _result(),
            telemetry,
            _source_context(),
            render=False,
            env_factory=env_type,
        )


@pytest.mark.parametrize("timestamp", [float("nan"), float("inf"), True])
def test_replay_rejects_nonfinite_or_boolean_timestamp(tmp_path, timestamp):
    telemetry = _telemetry()
    telemetry[0]["time_s"] = timestamp
    directory = _episode_directory(tmp_path, telemetry)
    with pytest.raises(replay.ReplayError, match="finite timestamp"):
        replay.replay_episode(directory, _result(), telemetry, _source_context(),
                              render=False, env_factory=FakeEnv)
    validation = json.loads((directory / "replay-validation.json").read_text())
    assert not validation["complete"]
    assert not validation["claimed_video_pass"]


def test_result_hash_validation_rejects_missing_telemetry_hash(tmp_path):
    directory = tmp_path / "nominal-0"
    directory.mkdir()
    (directory / "telemetry.jsonl").write_text("{}\n")

    with pytest.raises(replay.ReplayError, match="telemetry hash"):
        replay._safe_file_hashes(directory, {"files": {}})


def test_replay_rejects_output_inside_immutable_source(tmp_path):
    source = tmp_path / "metrics"
    source.mkdir()
    with pytest.raises(replay.ReplayError, match="outside the immutable source"):
        replay.replay_bundle(source, source / "videos")
    assert not (source / "videos").exists()


@pytest.mark.parametrize("changed", [None, "evaluation", "result", "telemetry", "snapshot"])
def test_original_source_rechecked_before_video_publication(tmp_path, changed):
    directory = tmp_path / "nominal-0"
    directory.mkdir()
    snapshot = tmp_path / "source-snapshot"
    snapshot.mkdir()
    paths = {
        "evaluation": tmp_path / "evaluation.json",
        "result": directory / "result.json",
        "telemetry": directory / "telemetry.jsonl",
        "snapshot": snapshot / "tennis-return.json",
    }
    for path in paths.values():
        path.write_text("{}\n")
    result = {"files": {"telemetry.jsonl": replay.sha256(paths["telemetry"])}}
    evaluation = {"experiment_hashes": {
        "python_sources": {}, "task_spec": replay.sha256(paths["snapshot"]),
    }}
    preflight = {
        "source": tmp_path, "evaluation": evaluation,
        "evaluation_sha256": replay.sha256(paths["evaluation"]),
        "snapshot_hashes": {"tennis-return.json": replay.sha256(paths["snapshot"])},
        "episodes": {"nominal-0": {
            "result": result, "result_sha256": replay.sha256(paths["result"]),
        }},
    }
    if changed:
        paths[changed].write_text('{"changed": true}\n')
        with pytest.raises(replay.ReplayError):
            replay._verify_original_source(preflight)
    else:
        replay._verify_original_source(preflight)


def test_small_real_episode_replays_exactly_without_graphics(tmp_path):
    from microduck_arm_experiments.tennis_return import TennisReturnEnv

    environment = TennisReturnEnv(
        variant="small",
        gripper="wide_candidate",
        release_mode="half_height",
        max_steps=1,
    )
    try:
        environment.reset(seed=4)
        action = environment.teacher_action("hold")
        _, done, info = environment.step(action)
        assert done
        record = {
            "step": 1,
            "action": action.tolist(),
            "time_s": info["time_s"],
            "ball_position_m": info["ball_position_m"],
            "joint_positions_rad": info["joint_positions_rad"],
            "phase": info["phase"],
            "done": done,
            "success": info["success"],
            "failure_reason": info["failure_reason"],
        }
    finally:
        environment.close()

    directory = _episode_directory(tmp_path, [record])
    result = {
        **_result(steps=1),
        "variant": "small",
        "success": record["success"],
        "failure_reason": record["failure_reason"],
        "max_steps": 1,
    }
    validation = replay.replay_episode(
        directory,
        result,
        [record],
        _source_context(),
        render=False,
    )

    assert validation["complete"] is True
    assert validation["zero_error"] is True
    assert validation["steps_replayed"] == 1
    assert validation["substep_monitor_calls"] > 0
