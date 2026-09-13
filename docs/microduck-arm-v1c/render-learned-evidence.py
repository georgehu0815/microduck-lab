import json

import imageio.v2 as imageio
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from microduck_arm_v1c.config import ROOT, sha256
from microduck_arm_v1c.env import IntegratedArmEnv
from microduck_arm_v1c.evidence import verify_video
from microduck_arm_v1c.learning import compose_arm_action, compose_residual_action, _validate_provenance, source_provenance
from microduck_arm_v1c.validation import write_json


def main():
    training = ROOT / "artifacts/microduck-arm-v1c/learning/attempt-2"
    output = ROOT / "artifacts/microduck-arm-v1c/videos-learned"
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((training / "training-summary.json").read_text())
    initial_sources = source_provenance()
    _validate_provenance(summary["source_provenance"])
    dagger_records = summary["evaluation"]["dagger"]["records"]
    choices = [("bc-initial-arm", "bc", 101), ("ppo-residual-arm", "ppo", 101)]
    for success in (True, False):
        matching = [record for record in dagger_records if record["success"] == success]
        if matching:
            choices.append(("bc-dagger-merged-arm", "bc", matching[0]["seed"]))
    records = []
    for name, kind, seed in choices:
        model_path = training / "exports" / f"{name}.onnx"
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        env = IntegratedArmEnv(case="reach", mode="fixture")
        path = output / f"{name}-seed-{seed}.mp4"
        telemetry = []
        try:
            observation, info = env.reset(seed=seed)
            with imageio.get_writer(path, fps=25, codec="libx264") as writer:
                for step in range(env.max_steps):
                    teacher = env.teacher_action()
                    arm = session.run(None, {"observation": observation[None]})[0][0]
                    action = compose_arm_action(teacher, arm) if kind == "bc" else compose_residual_action(teacher, arm, env.action_scales)
                    if step % 2 == 0:
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 26), fill="black")
                        draw.text((8, 7), f"SIMULATION | ONNX {name} | fixture | seed {seed}", fill="white")
                        writer.append_data(np.asarray(frame))
                    observation, reward, terminated, truncated, info = env.step(action)
                    telemetry.append({"step": step, "reward": reward, "action": action.tolist(), "diagnostics": info["diagnostics"]})
                    if terminated or truncated:
                        break
            _validate_provenance(initial_sources)
            record = {"policy": name, "seed": seed, "onnx_sha256": sha256(model_path), "success": info["success"], "failure_reason": info["failure_reason"], "video": verify_video(path), "source_provenance": initial_sources}
            records.append(record)
            write_json(output / f"{name}-{seed}.json", {**record, "telemetry": telemetry})
            print(name, seed, info["success"], info["failure_reason"], flush=True)
        finally:
            env.close()
    write_json(output / "video-manifest.json", {"scope": "ONNX simulator rollouts including failures, not hardware deployment", "records": records})


if __name__ == "__main__":
    main()
