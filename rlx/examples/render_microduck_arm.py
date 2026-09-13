"""Render and optionally export the bounded Microduck arm sidecar."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import textwrap
from typing import Any, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO

from rlx.environments import arm

import ppo_microduck_arm as pipeline


VIDEO_NAME = "rollout.mp4"
CONTACT_SHEET_NAME = "contact-sheet.png"
RECEIPT_NAME = "render-receipt.json"
TELEMETRY_NAME = "telemetry.jsonl"
ONNX_NAME = "residual-actor.onnx"
ONNX_METADATA_NAME = "residual-actor.json"
SHEET_FRAMES = 6
OVERLAY_FONT_SIZE = 17
OVERLAY_LINE_HEIGHT = 22


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_hashes() -> dict[str, str]:
    return {
        "environment": sha256_file(arm.__file__),
        "pipeline": sha256_file(pipeline.__file__),
    }


def renderer_source_sha256() -> str:
    return sha256_file(__file__)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def controller_label(controller: str) -> str:
    if controller == "teacher":
        return "authored IK/FSM teacher"
    if controller == "ppo_residual":
        return "composite authored IK/FSM + PPO residual"
    raise ValueError(f"unsupported arm controller: {controller}")


def telemetry_record(
    *,
    step: int,
    action: Sequence[float],
    reward: float,
    cumulative_return: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "time": float(step * arm.CONTROL_DT),
        "action": np.asarray(action, dtype=np.float64).tolist(),
        "reward": float(reward),
        "cumulative_return": float(cumulative_return),
        "metrics": metrics,
    }


def failed_gate_names(metrics: dict[str, Any]) -> list[str]:
    gates = metrics.get("gates", {})
    if not isinstance(gates, dict):
        return []
    return [str(name) for name, passed in gates.items() if not passed]


def load_overlay_font() -> Any:
    from PIL import ImageFont

    for font_path in (
        "DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        try:
            return ImageFont.truetype(font_path, OVERLAY_FONT_SIZE)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=OVERLAY_FONT_SIZE)
    except TypeError:
        return ImageFont.load_default()


def overlay_line_color(line: str) -> tuple[int, int, int]:
    if line == "FINAL PASS":
        return (90, 230, 130)
    if line == "FINAL FAIL":
        return (255, 105, 105)
    return (255, 255, 255)


def overlay_lines(
    *,
    controller: str,
    seed: int,
    step: int,
    metrics: dict[str, Any],
    case_id: str | None = None,
    sim_time: float | None = None,
    cumulative_return: float | None = None,
    terminal: bool = False,
) -> tuple[str, ...]:
    contacts = sum(
        int(value) for row in metrics.get("contact_flags", []) for value in row
    )
    gates = metrics.get("gates", {})
    passed_gates = (
        sum(bool(value) for value in gates.values())
        if isinstance(gates, dict)
        else 0
    )
    total_gates = len(gates) if isinstance(gates, dict) else 0
    status = "PASS" if metrics.get("passed") else "FAIL"
    verdict = status if terminal else "LIVE (not final verdict)"
    elapsed = step * arm.CONTROL_DT if sim_time is None else sim_time
    cumulative = 0.0 if cumulative_return is None else cumulative_return
    object_position = metrics.get("object_position", [])
    goal = metrics.get("goal", [])

    def position_label(values: Any) -> str:
        if not isinstance(values, (list, tuple)) or len(values) < 3:
            return "n/a"
        return ",".join(f"{float(value):.3f}" for value in values[:3])

    lines = [
        "SIMULATION ONLY - HARDWARE NOT VERIFIED",
        (
            f"case: {case_id or 'n/a'}  seed: {seed}  "
            f"t: {elapsed:.2f}s  step: {step}  {verdict}"
        ),
        (
            f"controller: {controller_label(controller)}  "
            f"stage: {metrics.get('evaluator_stage', 'n/a')}"
        ),
        (
            f"return: {cumulative:.3f}  gates: {passed_gates}/{total_gates}  "
            f"contacts: {contacts}"
        ),
        (
            f"error: {float(metrics.get('position_error_m', 0.0)):.4f}m  "
            f"tip: {float(metrics.get('tip_error_m', 0.0)):.4f}m  "
            f"drop: {int(metrics.get('drop_count', 0))}"
        ),
        (
            f"peak force: {float(metrics.get('peak_contact_force_n', 0.0)):.2f}N  "
            f"peak internal: {float(metrics.get('peak_internal_force_n', 0.0)):.2f}N  "
            f"illegal: {int(metrics.get('invalid_contacts', 0))}"
        ),
        (
            f"object xyz: {position_label(object_position)}  "
            f"goal xyz: {position_label(goal)}"
        ),
    ]
    if terminal:
        failed = failed_gate_names(metrics)
        failed_text = ", ".join(failed) if failed else "none"
        lines.append(f"FINAL {status}")
        lines.extend(
            textwrap.wrap(
                f"failed gates: {failed_text}",
                width=78,
                subsequent_indent="  ",
            )
        )
    return tuple(lines)


def validate_checkpoint_metadata(
    checkpoint: Path, case_id: str
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    metadata_path = checkpoint.parent / "training.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"checkpoint provenance metadata is missing: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "case_id": case_id,
        "model_sha256": arm.model_hash(case_id),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_hashes": source_hashes(),
    }
    mismatches = {
        key: {"expected": value, "recorded": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint provenance mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "path": str(metadata_path.resolve()),
        "sha256": sha256_file(metadata_path),
        **expected,
    }


def overlay_frame(
    frame: np.ndarray,
    *,
    controller: str,
    seed: int,
    step: int,
    metrics: dict[str, Any],
    case_id: str | None = None,
    sim_time: float | None = None,
    cumulative_return: float | None = None,
    terminal: bool = False,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = load_overlay_font()
    lines = overlay_lines(
        controller=controller,
        seed=seed,
        step=step,
        metrics=metrics,
        case_id=case_id,
        sim_time=sim_time,
        cumulative_return=cumulative_return,
        terminal=terminal,
    )
    panel_height = OVERLAY_LINE_HEIGHT * len(lines) + 10
    draw.rectangle((0, 0, image.width, panel_height), fill=(10, 14, 18))
    for index, line in enumerate(lines):
        draw.text(
            (10, 7 + index * OVERLAY_LINE_HEIGHT),
            line,
            fill=overlay_line_color(line),
            font=font,
        )
    return np.asarray(image)


def write_contact_sheet(frames: Sequence[np.ndarray], path: Path) -> None:
    from PIL import Image, ImageDraw

    if not frames:
        raise RuntimeError("cannot create an arm contact sheet without frames")
    selected = list(frames[:SHEET_FRAMES])
    while len(selected) < SHEET_FRAMES:
        selected.append(selected[-1])
    tile_width, tile_height = 320, 180
    sheet = Image.new("RGB", (tile_width * 3, tile_height * 2), (10, 14, 18))
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(selected):
        tile = Image.fromarray(frame).resize((tile_width, tile_height))
        x = index % 3 * tile_width
        y = index // 3 * tile_height
        sheet.paste(tile, (x, y))
        draw.rectangle((x, y, x + 28, y + 19), fill=(10, 14, 18))
        draw.text((x + 7, y + 3), str(index + 1), fill=(255, 255, 255))
    sheet.save(path)


class ResidualActor(torch.nn.Module):
    """Deterministic PPO residual only; nominal IK/FSM is intentionally absent."""

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean = self.policy.get_distribution(observation).distribution.mean
        return torch.clamp(mean, -1.0, 1.0)


def residual_contract(case_id: str) -> str:
    env = arm.ArmEnv(case_id)
    try:
        return f"{env.contract}-ppo-residual-v1"
    finally:
        env.close()


def compare_onnx(
    actor: ResidualActor, path: Path, observation_dim: int
) -> dict[str, Any]:
    import onnxruntime as ort

    rng = np.random.default_rng(9127)
    observations = rng.normal(
        0.0, 0.1, size=(8, observation_dim)
    ).astype(np.float32)
    with torch.no_grad():
        native = actor(torch.from_numpy(observations)).cpu().numpy()
    session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    inferred = session.run(
        None, {session.get_inputs()[0].name: observations}
    )[0]
    error = float(np.max(np.abs(native - inferred)))
    return {"max_abs_error": error, "passed": error <= 1e-4}


def export_residual_actor(
    model: PPO, case_id: str, output: Path
) -> dict[str, Any]:
    import onnx

    env = arm.ArmEnv(case_id)
    try:
        observation_dim = env.observation_dim
        action_dim = env.action_dim
        contract = residual_contract(case_id)
    finally:
        env.close()
    actor = ResidualActor(model.policy).cpu().eval()
    sample = torch.zeros(1, observation_dim, dtype=torch.float32)
    torch.onnx.export(
        actor,
        sample,
        output,
        input_names=["observation"],
        output_names=["residual_action"],
        dynamic_axes={
            "observation": {0: "batch"},
            "residual_action": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "contract": contract,
        "case_id": case_id,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "action_semantics": "residual",
        "residual_scale": pipeline.RESIDUAL_SCALE,
        "action_clamp": [-1.0, 1.0],
        "controller": "composite authored IK/FSM + PPO residual",
        "simulation_only": True,
        "stock_deployment_compatible": False,
        "source_hashes": source_hashes(),
    }
    graph = onnx.load(output)
    graph.graph.output[0].type.tensor_type.shape.dim[1].dim_param = ""
    graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = action_dim
    del graph.metadata_props[:]
    properties = {
        key: json.dumps(value, sort_keys=True, separators=(",", ":"))
        for key, value in metadata.items()
    }
    properties["rlx_metadata"] = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    onnx.helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph)
    onnx.save(graph, output)
    parity = compare_onnx(actor, output, observation_dim)
    if not parity["passed"]:
        raise RuntimeError(
            f"native/ONNX residual parity exceeded 1e-4: {parity}"
        )
    result = {
        **metadata,
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "native_onnx_float32_parity": parity,
    }
    write_json(output.with_name(ONNX_METADATA_NAME), result)
    return result


def build_receipt(
    *,
    output: Path,
    case_id: str,
    controller: str,
    checkpoint: Path | None,
    checkpoint_provenance: dict[str, Any] | None,
    seed: int,
    options: dict[str, float],
    episode: dict[str, Any],
    export: dict[str, Any] | None,
    telemetry: Path | None = None,
) -> dict[str, Any]:
    video = output / VIDEO_NAME
    sheet = output / CONTACT_SHEET_NAME
    checkpoint_path = checkpoint.resolve() if checkpoint is not None else None
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulation_only": True,
        "hardware_verified": False,
        "case_id": case_id,
        "controller": controller,
        "controller_description": controller_label(controller),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "checkpoint_metadata": checkpoint_provenance,
        "seed": seed,
        "reset_options": options,
        "contract": episode["contract"],
        "model_sha256": arm.model_hash(case_id),
        "source_hashes": source_hashes(),
        "renderer_source_sha256": renderer_source_sha256(),
        "episode": episode,
        "episode_sha256": canonical_sha256(episode),
        "video_sha256": sha256_file(video),
        "telemetry_sha256": (
            sha256_file(telemetry) if telemetry is not None else None
        ),
        "media_hashes": {
            VIDEO_NAME: sha256_file(video),
            CONTACT_SHEET_NAME: sha256_file(sheet),
        },
        "outputs": {
            "video": str(video.resolve()),
            "contact_sheet": str(sheet.resolve()),
            "telemetry": (
                str(telemetry.resolve()) if telemetry is not None else None
            ),
        },
        "export": export,
    }
    if telemetry is not None:
        receipt["media_hashes"][TELEMETRY_NAME] = sha256_file(telemetry)
    return receipt


def render_episode(
    *,
    case_id: str,
    output: Path,
    teacher: bool,
    checkpoint: Path | None,
    seed: int,
    options: dict[str, float],
    fps: float,
    export: bool,
    trace: bool = False,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    if teacher == (checkpoint is not None):
        raise ValueError("select exactly one of --teacher or --checkpoint")
    if not math.isfinite(fps) or not 0 < fps <= 50:
        raise ValueError("fps must be finite and in (0, 50]")
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    start_sources = source_hashes()
    start_renderer_source = renderer_source_sha256()
    checkpoint_provenance = (
        validate_checkpoint_metadata(checkpoint, case_id)
        if checkpoint is not None
        else None
    )
    model = PPO.load(checkpoint) if checkpoint is not None else None
    controller = "teacher" if teacher else "ppo_residual"
    env = arm.ArmEnv(case_id, render_mode="rgb_array")
    observation, _ = env.reset(seed=seed, options=options)
    frame_stride = max(1, round((1 / arm.CONTROL_DT) / fps))
    actual_fps = (1 / arm.CONTROL_DT) / frame_stride
    max_steps = math.ceil(arm.SPECS[case_id].horizon / arm.CONTROL_DT)
    targets = set(
        np.linspace(
            0, max(0, math.ceil(max_steps / frame_stride) - 1), SHEET_FRAMES
        ).astype(int)
    )
    retained: list[np.ndarray] = []
    total_reward = 0.0
    rendered = 0
    frame_times: list[float] = []
    terminal_frame_off_cadence = False
    video = output / VIDEO_NAME
    telemetry = output / TELEMETRY_NAME if trace else None
    telemetry_stream = telemetry.open("w") if telemetry is not None else None
    try:
        with imageio.get_writer(
            video, fps=actual_fps, macro_block_size=None
        ) as writer:
            while not (env.terminated or env.truncated):
                nominal = env.teacher_action()
                if model is None:
                    action = nominal
                else:
                    residual, _ = model.predict(
                        observation, deterministic=True
                    )
                    action = np.clip(
                        nominal
                        + pipeline.RESIDUAL_SCALE
                        * np.asarray(residual, dtype=np.float32),
                        -1.0,
                        1.0,
                    )
                env.controller = controller
                observation, reward, _, _, metrics = env.step(action)
                total_reward += float(reward)
                if telemetry_stream is not None:
                    telemetry_stream.write(
                        json.dumps(
                            telemetry_record(
                                step=env.steps,
                                action=action,
                                reward=reward,
                                cumulative_return=total_reward,
                                metrics=metrics,
                            ),
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                terminal = env.terminated or env.truncated
                render_now = env.steps % frame_stride == 0 or terminal
                if render_now:
                    terminal_frame_off_cadence = bool(
                        terminal and env.steps % frame_stride != 0
                    )
                    frame = overlay_frame(
                        env.render().copy(),
                        controller=controller,
                        seed=seed,
                        step=env.steps,
                        metrics=metrics,
                        case_id=case_id,
                        sim_time=env.steps * arm.CONTROL_DT,
                        cumulative_return=total_reward,
                        terminal=terminal,
                    )
                    writer.append_data(frame)
                    frame_times.append(env.steps * arm.CONTROL_DT)
                    if rendered in targets or terminal:
                        if terminal and len(retained) >= SHEET_FRAMES:
                            retained[-1] = frame.copy()
                        else:
                            retained.append(frame.copy())
                    rendered += 1
        metrics = env.metrics()
        elapsed_seconds = env.steps * arm.CONTROL_DT
        final_sample_duration = (
            frame_times[-1] - frame_times[-2]
            if len(frame_times) > 1
            else frame_times[-1]
        )
        episode = {
            "case_id": case_id,
            "contract": env.contract,
            "observation_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "seed": seed,
            "reset_options": options,
            "steps": env.steps,
            "elapsed_seconds": elapsed_seconds,
            "frames": rendered,
            "fps": actual_fps,
            "return": total_reward,
            "metrics": metrics,
            "video_timing": {
                "requested_fps": fps,
                "frame_stride_steps": frame_stride,
                "nominal_frame_period_seconds": frame_stride
                * arm.CONTROL_DT,
                "frame_count": rendered,
                "frame_times_seconds": frame_times,
                "encoded_duration_seconds": rendered / actual_fps,
                "terminal_frame_time_seconds": elapsed_seconds,
                "terminal_frame_off_cadence": terminal_frame_off_cadence,
                "final_sample_duration_seconds": final_sample_duration,
                "appended_hold_frames": 0,
                "note": (
                    "Frames are real simulation samples. An off-cadence "
                    "terminal frame may have a shorter final sample interval "
                    "than the MP4 constant frame period; no hold/title frames "
                    "are appended."
                ),
            },
        }
    finally:
        if telemetry_stream is not None:
            telemetry_stream.close()
        env.close()
    if rendered == 0:
        raise RuntimeError("arm rollout produced no frames")
    if source_hashes() != start_sources:
        raise RuntimeError("arm environment or pipeline changed during render")
    if renderer_source_sha256() != start_renderer_source:
        raise RuntimeError("arm renderer changed during render")
    sheet = output / CONTACT_SHEET_NAME
    write_contact_sheet(retained, sheet)
    if export and model is None:
        raise ValueError("--export requires a PPO residual checkpoint")
    export_result = (
        export_residual_actor(model, case_id, output / ONNX_NAME)
        if export
        else None
    )
    receipt = build_receipt(
        output=output,
        case_id=case_id,
        controller=controller,
        checkpoint=checkpoint,
        checkpoint_provenance=checkpoint_provenance,
        seed=seed,
        options=options,
        episode=episode,
        export=export_result,
        telemetry=telemetry,
    )
    write_json(output / RECEIPT_NAME, receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Microduck arm simulation rollout."
    )
    parser.add_argument("--case", choices=arm.CASES, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher", action="store_true")
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--position-noise", type=float, default=0.001)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--friction-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args(argv)
    if args.export and args.checkpoint is None:
        parser.error("--export requires --checkpoint")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = render_episode(
        case_id=args.case,
        output=args.output,
        teacher=args.teacher,
        checkpoint=args.checkpoint,
        seed=args.seed,
        options={
            "position_noise": args.position_noise,
            "mass_scale": args.mass_scale,
            "friction_scale": args.friction_scale,
        },
        fps=args.fps,
        export=args.export,
        trace=args.trace,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return receipt


if __name__ == "__main__":
    main()
