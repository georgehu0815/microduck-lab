# Verified local results — September 10, 2026

## Status

**Sustained free-ball balance is verified. Controlled walking/steering is not solved.**

The demonstration candidate stays upright on the moving basketball throughout
all six 60-second local trials: seeds 101, 202 and 303, each with a zero command
and with a 0.15 m/s forward command. There are no automatic resets, no hold
assistance, and no robot-floor or non-foot robot-ball contact. Minimum per-trial
foot-ball contact coverage is 99.93%; maximum tilt is 4.74 degrees.

However, **zero of three commanded-forward trials pass the steering gate**.
The ball moves and the robot adjusts its feet, but movement includes drift and
turning instead of reliable command-following. This is not a stationary-position
hold, nor a demonstrated straight-line walking controller. The aggregate
evaluation correctly reports `passed: false`.

## Watch the measured behavior

- [60-second zero-command balance video](evidence/render/cmd-p0_000-p0_000-p0_000/seed-101/rollout.mp4)
- [60-second commanded rolling attempt — steering gate fails](evidence/render/cmd-p0_150-p0_000-p0_000/seed-101/rollout.mp4)
- [Rolling-attempt contact sheet](evidence/render/cmd-p0_150-p0_000-p0_000/seed-101/contact-sheet.png)
- [Machine-readable final evaluation](evidence/evaluation.json)

The actual rendered contact sheets were inspected: the duck remains upright
above the ball, the ball texture rotates, and foot support changes between one
and two contacts. The camera was widened to retain the full robot at reset and
the lighting was increased for visibility. No animation clip or scripted robot
motion is used. Rendered clips show seed 101; the JSON includes all six trials.

## Training and comparison

All rows use the same 60-second local evaluation conditions: CPU BAM simulation,
free ball, no observation noise or mass/friction randomization, no pushes,
randomized initial yaw, and first-fall accounting. BAM voltage and bus-delay
sampling remain part of the local actuator. This is not the reference GPU stress
battery, and six local trials do not establish broad reliability.

| Candidate | New training steps | 60 s survival | Forward tracking MAE at 0.15 m/s | Commanded rolling |
|---|---:|---:|---:|---:|
| Supplied b11 source, local replay | None | 6/6 | 0.140 m/s | 0/3 |
| Fixed-command local adaptation | 102,400 | 6/6 | 0.138 m/s | 0/3 |
| Mixed-command local adaptation | 153,600 | 6/6 | 0.160 m/s | 0/3 |

These are **two independent warm-start runs from b11**, not a single cumulative
256,000-step lineage. Each uses a fresh local critic and optimizer. The
fixed-command candidate supplies the demonstration videos; neither candidate is
accepted as a completed walking policy, and the supplied source is not replaced.
The tiny fixed-command tracking difference is not evidence of a robust improvement.

The fixed-command run used 8 environments, 128-step rollouts, 100 updates,
learning rate 2e-6 and initial action standard deviation 0.03. The mixed run used
150 updates, learning rate 1e-5 and standard deviation 0.08, sampling stationary
and ranged forward/lateral/yaw commands. Both had hold zero throughout training.
PPO recomputes recurrent sequences and resets memory only at episode boundaries.
KL stopping limits repeated update epochs; it is not a guarantee that a single
optimizer step cannot exceed the configured KL target.

| Artifact | Location |
|---|---|
| Fixed-command checkpoint, ONNX, summary | `rlx/artifacts/basketball-local-20260910/` |
| Mixed-command checkpoint, ONNX, summary | `rlx/artifacts/basketball-mixed-20260910/` |
| Mixed-command evaluation | `rlx/artifacts/basketball-mixed-eval-20260910/evaluation.json` |
| Intermediate checkpoints | `update-0025/`, `update-0050/`, `update-0075/` in the fixed run; `update-0050/`, `update-0100/` in the mixed run |
| Curriculum training smoke | `rlx/artifacts/basketball-curriculum-smoke-20260910/` |

The fixed and mixed actors changed from the source by L2 parameter distances
0.04296 and 0.22950 respectively. Their 40-step PyTorch/ONNX parity checks,
including a recurrent-state reset, had maximum absolute action errors 1.43e-6
and 2.15e-6. Normalization remained frozen and baked into the exports.

## Curriculum evidence and limitation

A separate 7,168-step, two-environment training smoke exercised the assistance
ladder from hold 1. It recorded 39 completed episodes and both promotion and
demotion, finishing with holds between 0.25 and 0.5. **It did not graduate through
the whole ladder to zero assistance.** This tests the curriculum implementation,
not a claim of learning free balance from scratch. The main candidates start
from the supplied mature free-ball actor instead.

The curriculum changes physical support only, never reward weights. Ball
rotational assistance is transformed into world-frame torque; a regression test
locks this behavior. Every reported behavior evaluation uses hold zero, so none
of the demonstrated balance is supplied by curriculum forces.

## Verification

- 59 focused basketball and existing environment-contract tests passed.
- Ruff correctness lint, mypy over the four new Python implementation files,
  Python compilation, and shell syntax validation passed.
- Source checkpoint/ONNX SHA256 values matched the supplied release receipt.
- Full pre-training `microduck_local` suite: 411 passed, 1 skipped, 12 failed.
  The failures are two existing BAM golden fingerprints, nine existing step
  fingerprints, and one seed-11 symmetry drift assertion. Those files were not
  changed for this task; the failures were not hidden or repaired by changing
  golden data.
- Reference files and the existing five Studio scenarios remain untouched.
- No hardware deployment or hardware-safety claim is made.

## Remaining work

Reliable command-conditioned rolling needs a further training experiment; it is
not fixed by relaxing the evaluation thresholds. The current gates require
60-second balance, at least 3.15 m signed progress for the 0.15 m/s forward case,
linear tracking MAE at most 0.09 m/s, yaw tracking MAE at most 0.5 rad/s, genuine
ball rotation and no support/contact exploits.

Before another long run, test a rolling-state initialization curriculum to make
command-aligned motion occur in rollouts, then remove that initialization aid
and re-evaluate starts from rest. An exact continuation of the source GPU recipe
is a separate route requiring its mjlab/CUDA environment; the local prototype
does not impersonate that optimizer/critic continuation.
