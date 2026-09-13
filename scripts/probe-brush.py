import argparse
import json
from pathlib import Path

import numpy as np

from rlx.environments.brush import BrushEnv

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="rlx/runs/brush-probes/teacher-01")
parser.add_argument("--seed", type=int, default=4)
parser.add_argument("--seconds", type=float, default=120)
args = parser.parse_args()
env = BrushEnv(seed=args.seed, max_episode_s=args.seconds)
observations, actions = [], []
for step in range(env.max_steps):
    observations.append(env._get_obs())
    action = env.teacher_action()
    actions.append(action)
    _, reward, done, truncated, info = env.step(action)
    if step % 500 == 0:
        print(step, info["distance_m"], env.loaded_color, len(env.trace), flush=True)
    if done or truncated:
        break
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
assessment = env.assessment()
(out / "assessment.json").write_text(json.dumps(assessment, indent=2))
np.savez_compressed(out / "teacher.npz", observations=observations, actions=actions, trace=np.asarray(env.trace), colors=env.color_trace)
print(json.dumps(assessment, indent=2), flush=True)
