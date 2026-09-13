# Microduck basketball: local simulation and training

This prototype uses the user-supplied
[`microduck-playground/experiments/basketball`](../../microduck-playground/experiments/basketball/README.md)
and [`artifacts/basketball`](../../microduck-playground/artifacts/basketball/README.md)
as its reference. The source b11 checkpoint and ONNX hashes are verified before
training. The reference's published evaluation is **not** a local training result.

## What is implemented

- A free sphere with radius 0.12 m and mass 0.62 kg, reference collision mesh,
  basketball texture, full-collision robot, and 50 Hz control in CPU MuJoCo.
- A blind 61-input, 14-action LSTM256 actor, source normalization baked into
  ONNX, and explicit hidden/cell state carried across control steps.
- A six-level physical assistance curriculum: `1, 0.5, 0.25, 0.1, 0.03, 0`.
  Completed episodes of at least 6 seconds promote a level; falls before
  1.5 seconds demote. Rewards never change with the curriculum.
- Fresh local recurrent PPO updates, exported ONNX parity checks, intermediate
  checkpoints, deterministic first-fall evaluation, MP4 and contact-sheet rendering.

The robot starts upright at the ball apex. Climbing onto the ball, recovering
from the floor, hardware operation, and Studio UI integration are not included.
The existing five Studio scenarios are left unchanged.

## Run it

Use the existing `rlx/.venv-microduck` environment. No extra packages are installed
by these commands. Run from the workspace root:

```bash
# Contracts, recurrent parity, and evaluation honesty checks.
rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_basketball.py rlx/tests/test_basketball_policy.py \
  rlx/tests/test_basketball_evaluation.py -q

# Five-update smoke, before any longer run.
bash scripts/train-basketball.sh \
  --output rlx/artifacts/basketball-smoke \
  --num-envs 4 --updates 5 --steps 32 --learning-rate 2e-6

# Adapt the reference actor locally, starting with a free ball.
bash scripts/train-basketball.sh \
  --output rlx/artifacts/basketball-training \
  --num-envs 8 --updates 100 --steps 128 \
  --learning-rate 2e-6 --save-interval 25 --command .15 0 0

# Evaluate both stationary and commanded-forward cases; inspect the renders.
rlx/.venv-microduck/bin/python rlx/scripts/eval_basketball_local.py \
  --policy rlx/artifacts/basketball-training/policy.onnx \
  --output rlx/artifacts/basketball-evaluation \
  --seconds 60 --seeds 101 202 303 \
  --command 0 0 0 --command .15 0 0 --render
```

Use `--hold 1 --curriculum` to exercise the assisted ladder. Reference warm-start
training defaults to `--hold 0`, because b11 already learned free-ball balance.
Use `--randomized-commands` instead of `--command` for the mixed-command sampler.
Evaluation always forces hold zero and never auto-resets a failed trial.
Use a new output directory for each experiment. Rendering requires a working
local graphics context; training and non-rendered evaluation are CPU-only.

## Important differences from the reference

The local trainer imports the reference **actor**, freezes its normalizer, and
creates a fresh critic and Adam optimizer. It does **not** resume the upstream
critic, optimizer, GPU curriculum counter, or exact mjlab dynamics. Its checkpoint
is a local artifact, not a claim of upstream-compatible full-run continuation.
The CLI starts from the hash-verified b11 source, not from arbitrary checkpoints.

The local task uses the existing CPU BAM actuator and a sealed bounded subset of
the reference rewards: velocity tracking, upright posture, centering, foot/ball
distance, height, ball speed, body angular velocity and action changes. It omits
the upstream angular-momentum and other inherited walking regularizers. The
actor sees no ball state; its memory supplies history, while simulator state is
used only for rewards, resets, assistance and evaluation.

The default local evaluation omits observation noise, mass/friction randomization,
and pushes. `--pushes` enables horizontal velocity disturbances every 1.5–3 seconds;
this is not the upstream 0.5–1 second stress battery. Local BAM voltage and bus
delay sampling still apply. Passing a few local seeds does not reproduce the
reference's 3,072-trial benchmark or establish hardware safety.

## Reading the evidence

`summary.json` records training settings, source/output hashes, actual environment
steps, parameter changes, PPO diagnostics and recurrent export parity.
`evaluation.json` records every seed, first-fall duration, foot-ball contacts,
tilt/offset, signed ball progress, ball rotation, tracking errors and failures.
The evaluator returns a nonzero status when the behavior criteria fail; a JSON
report and failure video are still useful evidence.

**Staying on the ball is not the same as standing still or controlled rolling.**
Always distinguish sustained balance, drift, and command-following. A rendered
rollout with moving basketball texture alone does not prove steering.

See [verified local results and reviewed videos](RESULTS.md). Balance passed;
controlled rolling remains unfinished.
