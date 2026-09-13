"""Deterministic first-fall evaluation and rendering for local basketball policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


CTRL_DT = 0.02
DEFAULT_SECONDS = 60.0
DEFAULT_SEEDS = (101, 202, 303)
DEFAULT_COMMANDS = ((0.0, 0.0, 0.0), (0.08, 0.0, 0.0))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO_ROOT / "microduck-playground/artifacts/basketball/policy.onnx"
DEFAULT_OUTPUT = REPO_ROOT / "rlx/artifacts/basketball-local-evaluation"


@dataclass(frozen=True)
class BasketballCriteria:
    version: int
    control_dt_s: float
    required_steps: int
    min_foot_ball_contact_fraction: float
    max_tilt_deg: float
    max_root_ball_offset_m: float
    min_command_progress_ratio: float
    min_command_progress_m: float
    random_travel_m: float
    max_linear_tracking_error_mps: float
    max_yaw_tracking_error_rad_s: float
    min_ball_rotation_rad: float
    all_trials_required: bool = True


def criteria_for(seconds: float, command: Sequence[float]) -> BasketballCriteria:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and greater than zero")
    if len(command) != 3 or not all(math.isfinite(float(value)) for value in command):
        raise ValueError("command must contain three finite values")
    required_steps = round(seconds / CTRL_DT)
    if required_steps < 1 or not math.isclose(
        required_steps * CTRL_DT, seconds, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("seconds must be an exact multiple of CTRL_DT")
    command_speed = math.hypot(float(command[0]), float(command[1]))
    return BasketballCriteria(
        version=2,
        control_dt_s=CTRL_DT,
        required_steps=required_steps,
        min_foot_ball_contact_fraction=0.5,
        max_tilt_deg=50.0,
        max_root_ball_offset_m=0.16,
        min_command_progress_ratio=0.35,
        min_command_progress_m=max(0.25, command_speed * seconds * 0.35),
        random_travel_m=0.25,
        max_linear_tracking_error_mps=max(0.04, command_speed * 0.6),
        max_yaw_tracking_error_rad_s=0.5,
        min_ball_rotation_rad=max(0.25, command_speed * seconds * 0.35) / 0.12 * 0.5,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initial_heading_from_xmat(xmat: Sequence[float]) -> tuple[float, float]:
    matrix = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    forward = matrix[:2, 0]
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("root xmat does not define a horizontal forward direction")
    forward /= norm
    return float(forward[0]), float(forward[1])


def commanded_world_direction(
    command: Sequence[float], initial_forward_xy: Sequence[float]
) -> tuple[float, float] | None:
    local = np.asarray(command[:2], dtype=np.float64)
    speed = float(np.linalg.norm(local))
    if speed < 1e-9:
        return None
    forward = np.asarray(initial_forward_xy, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("initial forward direction must be finite and nonzero")
    forward /= norm
    left = np.array([-forward[1], forward[0]], dtype=np.float64)
    world = (forward * local[0] + left * local[1]) / speed
    return float(world[0]), float(world[1])


def first_fall_reasons(
    info: dict[str, Any], *, terminated: bool, truncated: bool
) -> list[str]:
    raw_reasons = info.get("termination_reasons", ())
    if isinstance(raw_reasons, str):
        reasons = [raw_reasons]
    else:
        reasons = [str(reason) for reason in raw_reasons]
    if bool(info.get("robot_floor_contact", False)):
        reasons.append("robot_floor_contact")
    if bool(info.get("body_ball_contact", False)):
        reasons.append("body_ball_contact")
    try:
        if float(info.get("tilt_deg", 0.0)) >= 50.0:
            reasons.append("tilt")
        if float(info.get("root_ball_offset_m", 0.0)) >= 0.30:
            reasons.append("root_ball_offset")
    except (TypeError, ValueError):
        reasons.append("invalid_terminal_metrics")
    if truncated:
        reasons.append("truncated")
    if terminated and not reasons:
        reasons.append("terminated")
    return reasons


def evaluate_trace(
    samples: Iterable[dict[str, Any]],
    *,
    seed: int,
    command: Sequence[float],
    seconds: float,
    initial_forward_xy: Sequence[float],
    initial_ball_position: Sequence[float],
    terminated: bool = False,
    truncated: bool = False,
    automatic_resets: int = 0,
    fall_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    criteria = criteria_for(seconds, command)
    direction = commanded_world_direction(command, initial_forward_xy)
    previous_position = np.asarray(initial_ball_position, dtype=np.float64)
    if previous_position.shape != (3,) or not np.isfinite(previous_position).all():
        raise ValueError("initial ball position must contain three finite values")

    measured_steps = 0
    tracking_error_sum = 0.0
    lateral_tracking_error_sum = 0.0
    yaw_tracking_error_sum = 0.0
    absolute_ball_travel_m = 0.0
    commanded_progress_m = 0.0
    ball_rotation_rad = 0.0
    contact_steps = 0
    max_tilt_deg = 0.0
    max_root_ball_offset_m = 0.0
    body_ball_contact_steps = 0
    robot_floor_contact_steps = 0
    nonzero_hold_steps = 0
    invalid_samples = 0

    for sample in samples:
        measured_steps += 1
        try:
            tilt_deg = float(sample["tilt_deg"])
            offset_m = float(sample["root_ball_offset_m"])
            angular_speed = abs(float(sample["ball_angular_speed_rad_s"]))
            body_forward_mps = float(sample["body_forward_mps"])
            body_lateral_mps = float(sample["body_lateral_mps"])
            body_yaw_rate_rad_s = float(sample["body_yaw_rate_rad_s"])
            foot_contacts = int(sample["foot_ball_contacts"])
            hold = float(sample["hold"])
            position = np.asarray(sample["ball_position"], dtype=np.float64)
            finite = (
                position.shape == (3,)
                and np.isfinite(position).all()
                and all(
                    math.isfinite(value)
                    for value in (
                        tilt_deg,
                        offset_m,
                        angular_speed,
                        body_forward_mps,
                        body_lateral_mps,
                        body_yaw_rate_rad_s,
                        hold,
                    )
                )
            )
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            invalid_samples += 1
            continue

        delta = position - previous_position
        horizontal_delta = delta[:2]
        absolute_ball_travel_m += float(np.linalg.norm(horizontal_delta))
        if direction is not None:
            commanded_progress_m += float(
                horizontal_delta @ np.asarray(direction, dtype=np.float64)
            )
        previous_position = position
        tracking_error_sum += abs(body_forward_mps - float(command[0]))
        lateral_tracking_error_sum += abs(body_lateral_mps - float(command[1]))
        yaw_tracking_error_sum += abs(body_yaw_rate_rad_s - float(command[2]))
        ball_rotation_rad += angular_speed * CTRL_DT
        contact_steps += int(foot_contacts > 0)
        body_ball_contact_steps += int(bool(sample["body_ball_contact"]))
        robot_floor_contact_steps += int(bool(sample["robot_floor_contact"]))
        nonzero_hold_steps += int(not math.isclose(hold, 0.0, abs_tol=1e-12))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)
        max_root_ball_offset_m = max(max_root_ball_offset_m, offset_m)

    valid_steps = measured_steps - invalid_samples
    contact_fraction = contact_steps / valid_steps if valid_steps else 0.0
    mean_tracking_error = tracking_error_sum / valid_steps if valid_steps else None
    mean_lateral_tracking_error = (
        lateral_tracking_error_sum / valid_steps if valid_steps else None
    )
    mean_yaw_tracking_error = (
        yaw_tracking_error_sum / valid_steps if valid_steps else None
    )
    duration_complete = (
        measured_steps == criteria.required_steps
        and invalid_samples == 0
        and not terminated
        and not truncated
        and automatic_resets == 0
    )
    contact_complete = contact_fraction >= criteria.min_foot_ball_contact_fraction
    exploit_free = (
        body_ball_contact_steps == 0
        and robot_floor_contact_steps == 0
        and nonzero_hold_steps == 0
    )
    stable_geometry = (
        max_tilt_deg < criteria.max_tilt_deg
        and max_root_ball_offset_m < criteria.max_root_ball_offset_m
    )
    commanded = direction is not None
    progress_complete = (
        commanded and commanded_progress_m >= criteria.min_command_progress_m
    )
    sustained_balance = (
        duration_complete and contact_complete and exploit_free and stable_geometry
    )
    tracking_complete = (
        mean_tracking_error is not None
        and mean_tracking_error <= criteria.max_linear_tracking_error_mps
        and mean_lateral_tracking_error is not None
        and mean_lateral_tracking_error <= criteria.max_linear_tracking_error_mps
        and mean_yaw_tracking_error is not None
        and mean_yaw_tracking_error <= criteria.max_yaw_tracking_error_rad_s
    )
    rotation_complete = ball_rotation_rad >= criteria.min_ball_rotation_rad
    controlled_rolling = (
        sustained_balance and progress_complete and tracking_complete and rotation_complete
    )
    learned_success = controlled_rolling
    random_ball_movement = (
        absolute_ball_travel_m >= criteria.random_travel_m
        and (
            not commanded
            or commanded_progress_m
            < min(
                criteria.min_command_progress_m,
                absolute_ball_travel_m * criteria.min_command_progress_ratio,
            )
        )
    )
    failures: list[str] = []
    if measured_steps != criteria.required_steps or terminated or truncated:
        failures.append("required first-fall duration not completed")
    if automatic_resets:
        failures.append("automatic reset observed")
    if invalid_samples:
        failures.append("missing or non-finite physical metrics")
    if not contact_complete:
        failures.append("foot-ball contact fraction below target")
    if body_ball_contact_steps:
        failures.append("body-ball contact exploit observed")
    if robot_floor_contact_steps:
        failures.append("robot-floor contact exploit observed")
    if nonzero_hold_steps:
        failures.append("hold assistance observed")
    if not stable_geometry:
        failures.append("tilt or root-ball offset exceeded limit")
    if commanded and not progress_complete:
        failures.append("command-directed ball progress below target")
    if commanded and not tracking_complete:
        failures.append("velocity command tracking error exceeds target")
    if commanded and not rotation_complete:
        failures.append("ball rotation below rolling target")

    survived = duration_complete
    return {
        "seed": int(seed),
        "command": [float(value) for value in command],
        "command_world_direction": list(direction) if direction is not None else None,
        "horizon_s": float(seconds),
        "required_steps": criteria.required_steps,
        "measured_steps": measured_steps,
        "elapsed_s": measured_steps * CTRL_DT,
        "survived": survived,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "automatic_resets": int(automatic_resets),
        "first_fall_reasons": list(dict.fromkeys(fall_reasons)),
        "mean_abs_body_velocity_tracking_error_mps": mean_tracking_error,
        "mean_abs_body_lateral_velocity_tracking_error_mps": mean_lateral_tracking_error,
        "mean_abs_body_yaw_rate_tracking_error_rad_s": mean_yaw_tracking_error,
        "mean_abs_body_velocity_tracking": {
            "forward_mps": mean_tracking_error,
            "lateral_mps": mean_lateral_tracking_error,
            "yaw_rad_s": mean_yaw_tracking_error,
        },
        "integrated_absolute_ball_travel_m": absolute_ball_travel_m,
        "signed_commanded_progress_m": commanded_progress_m,
        "integrated_ball_rotation_rad": ball_rotation_rad,
        "foot_ball_contact_fraction": contact_fraction,
        "foot_ball_contact_percent": 100.0 * contact_fraction,
        "max_tilt_deg": max_tilt_deg,
        "max_root_ball_offset_m": max_root_ball_offset_m,
        "body_ball_contact_steps": body_ball_contact_steps,
        "robot_floor_contact_steps": robot_floor_contact_steps,
        "nonzero_hold_steps": nonzero_hold_steps,
        "hold_zero": nonzero_hold_steps == 0,
        "duration_complete": duration_complete,
        "progress_complete": progress_complete,
        "tracking_complete": tracking_complete,
        "rotation_complete": rotation_complete,
        "contact_complete": contact_complete,
        "exploit_free": exploit_free,
        "sustained_balance": sustained_balance,
        "controlled_rolling": controlled_rolling,
        "random_ball_movement": random_ball_movement,
        "learned_success": learned_success,
        "failures": failures,
        "criteria": asdict(criteria),
    }


def summarize_trials(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for trial in trials:
        command = (float(trial["command"][0]), float(trial["command"][1]), float(trial["command"][2]))
        groups.setdefault(command, []).append(trial)
    cases: list[dict[str, Any]] = []
    for command, rows in groups.items():
        commanded = math.hypot(command[0], command[1]) > 1e-9
        expected_key = "controlled_rolling" if commanded else "sustained_balance"
        cases.append(
            {
                "command": list(command),
                "classification": (
                    "commanded_rolling" if commanded else "zero_command_balance"
                ),
                "evaluated_seeds": [int(row["seed"]) for row in rows],
                "trial_count": len(rows),
                "survival_count": sum(bool(row["survived"]) for row in rows),
                "sustained_balance_count": sum(
                    bool(row["sustained_balance"]) for row in rows
                ),
                "controlled_rolling_count": sum(
                    bool(row["controlled_rolling"]) for row in rows
                ),
                "random_ball_movement_count": sum(
                    bool(row["random_ball_movement"]) for row in rows
                ),
                "learned_success_count": sum(
                    bool(row["learned_success"]) for row in rows
                ),
                "passed": all(bool(row[expected_key]) for row in rows),
            }
        )
    return {
        "trial_count": len(trials),
        "survival_count": sum(bool(trial["survived"]) for trial in trials),
        "all_evaluated_seeds_reported": all(
            len(case["evaluated_seeds"]) == case["trial_count"] for case in cases
        ),
        "cases": cases,
        "passed": bool(cases) and all(bool(case["passed"]) for case in cases),
        "interpretation": {
            "sustained_balance": "full first-fall horizon with foot-ball contact and no assistance or contact exploit",
            "controlled_rolling": "sustained balance, signed ball progress, rotation, and velocity tracking",
            "random_ball_movement": "substantial ball travel without sufficient command-directed progress",
        },
    }


class RecurrentOnnxPolicy:
    def __init__(self, path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        input_names = [item.name for item in self.session.get_inputs()]
        output_names = [item.name for item in self.session.get_outputs()]
        if input_names != ["obs", "h_in", "c_in"]:
            raise ValueError(f"unexpected recurrent ONNX inputs: {input_names}")
        if output_names != ["actions", "h_out", "c_out"]:
            raise ValueError(f"unexpected recurrent ONNX outputs: {output_names}")
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 256), dtype=np.float32)
        self.c = np.zeros((1, 1, 256), dtype=np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(1, 61)
        actions, self.h, self.c = self.session.run(
            ["actions", "h_out", "c_out"],
            {"obs": obs, "h_in": self.h, "c_in": self.c},
        )
        return np.asarray(actions[0], dtype=np.float32)


def _sample_from_info(info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "tilt_deg",
        "root_ball_offset_m",
        "root_above_ball_m",
        "ball_speed_mps",
        "ball_position",
        "ball_angular_speed_rad_s",
        "body_forward_mps",
        "body_lateral_mps",
        "body_yaw_rate_rad_s",
        "foot_ball_contacts",
        "body_ball_contact",
        "robot_floor_contact",
        "hold",
        "elapsed_s",
        "termination_reasons",
    )
    return {key: info[key] for key in keys}


def _initial_ball_position(env: Any, info: dict[str, Any]) -> np.ndarray:
    if "ball_position" in info:
        position = np.asarray(info["ball_position"], dtype=np.float64)
    else:
        position = np.asarray(env.data.xpos[env.ball_body_id], dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("environment did not provide a valid initial ball position")
    return position.copy()


def _make_env(
    *,
    seed: int,
    command: Sequence[float],
    seconds: float,
    actuator: str,
    pushes: bool,
) -> Any:
    from rlx.environments.basketball import BasketballEnv

    return BasketballEnv(
        max_episode_s=seconds + 1.0,
        hold=0,
        curriculum=False,
        command=(float(command[0]), float(command[1]), float(command[2])),
        seed=int(seed),
        actuator=actuator,
        obs_noise=False,
        domain_rand=False,
        pushes=pushes,
        random_yaw=True,
    )


class RenderCapture:
    def __init__(
        self,
        env: Any,
        *,
        path: Path,
        width: int,
        height: int,
        command: Sequence[float],
        seed: int,
    ):
        import imageio.v2 as imageio
        import mujoco

        self.np = np
        self.path = path
        self.path.mkdir(parents=True, exist_ok=True)
        self.width = width - width % 2
        self.height = height - height % 2
        self.command = tuple(float(value) for value in command)
        self.seed = seed
        self.renderer = mujoco.Renderer(
            env.model, height=self.height, width=self.width
        )
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.azimuth = 125.0
        self.camera.elevation = -14.0
        self.camera.distance = 1.0
        env.model.vis.headlight.ambient[:] = [.5, .5, .5]
        self.video_path = self.path / "rollout.mp4"
        self.sheet_path = self.path / "contact-sheet.png"
        self.writer = imageio.get_writer(
            self.video_path, fps=25.0, macro_block_size=None
        )
        self.sheet_frames: list[Any] = []
        self.next_sheet_s = 0.0

    def capture(self, env: Any, info: dict[str, Any], step: int) -> None:
        if step % 2:
            return
        from PIL import Image, ImageDraw, ImageFont

        trunk = np.asarray(env.data.xpos[env.trunk_body_id], dtype=np.float64)
        ball = np.asarray(env.data.xpos[env.ball_body_id], dtype=np.float64)
        self.camera.lookat[:] = (trunk + ball) * 0.5
        self.camera.lookat[2] += .04
        self.renderer.update_scene(env.data, camera=self.camera)
        image = Image.fromarray(self.renderer.render().copy())
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(14, self.width // 60))
        elapsed = step * CTRL_DT
        text = (
            f"t={elapsed:05.2f}s seed={self.seed} "
            f"cmd=({self.command[0]:+.2f},{self.command[1]:+.2f},{self.command[2]:+.2f}) "
            f"tilt={float(info.get('tilt_deg', 0.0)):04.1f}deg "
            f"contact={int(info.get('foot_ball_contacts', 0))}"
        )
        box = draw.textbbox((0, 0), text, font=font)
        draw.rectangle((0, 0, self.width, box[3] + 10), fill=(12, 18, 24))
        draw.text((8, 5), text, fill=(245, 245, 245), font=font)
        frame = np.asarray(image)
        self.writer.append_data(frame)
        if elapsed + CTRL_DT >= self.next_sheet_s:
            self.sheet_frames.append(image.copy())
            self.next_sheet_s += 5.0

    def close(self, final_elapsed_s: float) -> dict[str, Any]:
        from PIL import Image, ImageDraw, ImageFont

        self.writer.close()
        self.renderer.close()
        if not self.sheet_frames:
            raise RuntimeError("rendering produced no contact-sheet frames")
        tile_width = min(self.width, 640)
        ratio = tile_width / self.width
        tile_height = round(self.height * ratio)
        columns = min(3, len(self.sheet_frames))
        rows = math.ceil(len(self.sheet_frames) / columns)
        header_height = 48
        sheet = Image.new(
            "RGB",
            (columns * tile_width, header_height + rows * tile_height),
            (12, 18, 24),
        )
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=18)
        draw.text(
            (10, 12),
            f"Basketball first-fall rollout | seed {self.seed} | {final_elapsed_s:.2f}s",
            fill=(245, 245, 245),
            font=font,
        )
        for index, image in enumerate(self.sheet_frames):
            tile = image.resize((tile_width, tile_height))
            x = index % columns * tile_width
            y = header_height + index // columns * tile_height
            sheet.paste(tile, (x, y))
        sheet.save(self.sheet_path)
        return {
            "video": str(self.video_path),
            "contact_sheet": str(self.sheet_path),
            "contact_sheet_interval_s": 5.0,
            "rendered_until_s": final_elapsed_s,
        }


def run_trial(
    *,
    policy: RecurrentOnnxPolicy,
    seed: int,
    command: Sequence[float],
    seconds: float,
    actuator: str,
    pushes: bool,
    render_path: Path | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    env = _make_env(
        seed=seed,
        command=command,
        seconds=seconds,
        actuator=actuator,
        pushes=pushes,
    )
    capture: RenderCapture | None = None
    samples: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    reasons: list[str] = []
    try:
        observation, reset_info = env.reset(seed=seed)
        policy.reset()
        initial_forward = initial_heading_from_xmat(
            env.data.xmat[env.trunk_body_id]
        )
        initial_ball = _initial_ball_position(env, reset_info)
        if render_path is not None:
            capture = RenderCapture(
                env,
                path=render_path,
                width=width,
                height=height,
                command=command,
                seed=seed,
            )
            capture.capture(env, reset_info, 0)
        required_steps = criteria_for(seconds, command).required_steps
        for step in range(required_steps):
            action = policy.act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            samples.append(_sample_from_info(info))
            if capture is not None:
                capture.capture(env, info, step + 1)
            if terminated or truncated:
                reasons = first_fall_reasons(
                    info, terminated=terminated, truncated=truncated
                )
                break
        record = evaluate_trace(
            samples,
            seed=seed,
            command=command,
            seconds=seconds,
            initial_forward_xy=initial_forward,
            initial_ball_position=initial_ball.tolist(),
            terminated=terminated,
            truncated=truncated,
            automatic_resets=0,
            fall_reasons=reasons,
        )
        if capture is not None:
            record["render"] = capture.close(record["elapsed_s"])
            capture = None
        return record
    finally:
        if capture is not None:
            capture.close(len(samples) * CTRL_DT)
        env.close()


def _command_slug(command: Sequence[float]) -> str:
    return "cmd-" + "-".join(
        f"{float(value):+.3f}".replace("+", "p").replace("-", "m").replace(".", "_")
        for value in command
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        action="append",
        dest="commands",
        metavar=("FORWARD", "LATERAL", "YAW"),
    )
    parser.add_argument("--actuator", choices=("bam", "xml"), default="bam")
    parser.add_argument("--pushes", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = args.commands or [list(command) for command in DEFAULT_COMMANDS]
    policy_path = args.policy.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"policy does not exist: {policy_path}")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    if args.width < 160 or args.height < 120:
        raise ValueError("render width and height are too small")
    for command in commands:
        criteria_for(args.seconds, command)

    output.mkdir(parents=True, exist_ok=True)
    policy_hash = sha256_file(policy_path)
    evaluator_path = Path(__file__).resolve()
    policy = RecurrentOnnxPolicy(policy_path)
    trials = []
    for command in commands:
        for seed_index, seed in enumerate(args.seeds):
            render_path = None
            if args.render and seed_index == 0:
                render_path = output / "render" / _command_slug(command) / f"seed-{seed}"
            trials.append(
                run_trial(
                    policy=policy,
                    seed=seed,
                    command=command,
                    seconds=args.seconds,
                    actuator=args.actuator,
                    pushes=args.pushes,
                    render_path=render_path,
                    width=args.width,
                    height=args.height,
                )
            )

    environment_module = __import__(
        "rlx.environments.basketball", fromlist=["BasketballEnv"]
    )
    if environment_module.__file__ is None:
        raise RuntimeError("basketball environment has no source file for provenance")
    environment_path = Path(environment_module.__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "policy": str(policy_path),
        "output": str(output),
        "hashes": {
            "policy_sha256": policy_hash,
            "evaluator_sha256": sha256_file(evaluator_path),
            "environment_sha256": sha256_file(environment_path),
        },
        "protocol": {
            "control_dt_s": CTRL_DT,
            "seconds": args.seconds,
            "seeds": list(args.seeds),
            "commands": [[float(value) for value in command] for command in commands],
            "actuator": args.actuator,
            "pushes": bool(args.pushes),
            "hold": 0,
            "curriculum": False,
            "obs_noise": False,
            "domain_rand": False,
            "random_yaw": True,
            "first_fall": True,
            "automatic_resets": False,
            "recurrent_state": "zeroed on trial reset and carried for every ordinary control step",
        },
        "summary": summarize_trials(trials),
        "trials": trials,
    }
    result_path = output / "evaluation.json"
    result_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"evaluation": str(result_path), "passed": result["summary"]["passed"]}))
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
