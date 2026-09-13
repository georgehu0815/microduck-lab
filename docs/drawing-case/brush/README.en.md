# MicroDuck Color-Brush Upgrade: Accepted Learned Policy

Author: George Hu
Date: 2026-09-11
中文：[README.md](README.md)

This page documents the successful color-brush upgrade to the eighth Studio
scenario, `drawing`. It is not a ninth scenario and does not revise the failed
historical mouth-held-pencil result. The old `microduck-drawing-v1` contract
(83 observations / 15 actions) remains preserved; this upgrade uses the
separate `microduck-brush-v2` contract (93 observations / 15 actions).

## Verified result

### Two independently rendered videos

- [BC + DAgger: rollout_bc.mp4](media/rollout_bc.mp4), from `bc.onnx`, before PPO updates.
- [BC + DAgger + PPO: rollout.mp4](media/rollout.mp4), from `policy.onnx`.

Both use seed `10001`, the same camera, and 10fps, with an encoded duration of 120 seconds.
The BC video was separately rendered on 2026-09-12 and passed the complete painting assessment.
Its [source and decoding receipt](media/render_bc.json) and [contact sheet](media/frame_sheet_bc.png)
are retained for comparison; it is not a renamed copy of the PPO video.

The final evidence is
`rlx/runs/studio/drawing/brush-color-v2-03/eval.json` and the matching
`bc-eval.json`. The actual Studio API evaluation completed on 2026-09-11:

- the deterministic PPO ONNX passed `40` required episodes:
  `10` unique configurations x `4` seeds each;
- the matching BC ONNX passed the same 40 episodes;
- every episode completed `5,998` control steps, or `119.96 s`;
- PPO mean coverage was `1.0` and mean precision was `0.999531977503557`;
- BC mean coverage was `1.0` and mean precision was `0.9996346455645799`;
- PPO worst per-color coverage was `100%` and worst precision was `98.624%`,
  against `95%` thresholds at `1.5 mm` tolerance;
- worst grasp fraction was `99.983%`, maximum normal force was `0.0404 N`,
  maximum brush compression was `1.552 mm`, and maximum pen-up leakage was
  `4.144%`;
- every episode contacted the wells in the required
  gold -> orange -> charcoal -> blue order;
- both `null` and `open_jaw` negative controls failed;
- both reports record `passed=true`, `skill_status="passed"`,
  `provenance_errors=[]`, and `environment_source_match=true`.

This is a pass for the current simulation, fixed artwork, and measured
perturbation range. It is not evidence of hardware deployment, unknown-image
generalization, or learned visual planning. Because matching BC also passed
40/40 and had slightly higher precision, the supported conclusion is only that
anchored PPO **preserved** accepted BC behavior. There is no PPO improvement
claim.

`brush-color-v2-03` was trained by the older pipeline preserved in the run's
source archive. That version checked the teacher NPZ's adjacent JSON, dataset
SHA-256, and six then-current source hashes when loading the dataset, and it
saved all six source files. It did **not** contain the repeated lifecycle
`assert_sources` checks added after training, so final03 is not claimed to have
benefited from that new training-time guard.

The current hardened evaluator is an independent post-training verification
layer. It permits evaluator-only changes in `ppo_microduck_brush.py`, but the
archived older training script and the other five archived sources must all
match the six training hashes embedded in the ONNX metadata. The environment,
reference, actor, and two base sources must also match the evaluation source.
The evaluator verifies the actual checkpoint hash against both sidecar and
embedded ONNX metadata, checks the ONNX sidecar hash, reads the immutable ONNX
bytes once, and verifies file stability during execution. Reports use
`training_pipeline_source_match` to state whether the current evaluator is
byte-identical to the training script. It may be `false` when the archived
training package matches and every other provenance gate passes. ONNX export
also fixes the output's second dimension statically to `15` for the Lab/Viewer
action-contract check.

### Run history

| Run | Status | Evidence boundary |
|---|---|---|
| `brush-color-v2-01` | Preliminary success | PPO 20/20; later matching BC also 20/20. Its normal-duck painting was visually inspected, but it does not establish PPO improvement |
| `brush-color-v2-02` | Not promoted | Data seed `101`; BC validation terminated at 2,464 steps and failed, retained as training-data seed sensitivity |
| `brush-color-v2-03` | Final evaluation passed | Data seed `11`, training seed `101`; PPO 40/40, matching BC 40/40, strict provenance passed |

API evaluation and the source-bound final03 render are verified, and the
documentation media are published under this directory's `media/`. The PPO
and matching BC re-evaluations under the final hardened evaluator are
complete: `eval.json` was created at `2026-09-11T22:20:38Z` and `bc-eval.json`
at `2026-09-11T22:24:48Z`. Both passed 40/40 with
`training_pipeline_source_match=false` and `provenance_errors=[]`. These reports
include embedded-checkpoint hash and all-six-source-archive integrity gates;
their evaluator digest matches the current script. Independent browser verification passed:
`docs/drawing-case/brush/evidence/viewer.json` checks eight cards, eight live ducks,
colored contact data, the matching model's video playback, and a 390px mobile layout
without horizontal overflow or page errors. Live coverage is a bounded smoke check;
the complete 119.96-second painting is verified separately by the render and 40-episode gate.

## System boundary

The task has two distinct parts:

1. A **fixed authored plan** specifies the same swimming duck, color assignment,
   stroke order, pen-up travel, and four physical paint-well targets.
2. A **learned feedback actor plus PPO** executes 15 motor actions from the
   93-value state while maintaining grip, balance, contact, brush compression,
   and trajectory tracking.

This is not learned vision, image understanding, or task planning, and it is
not PPO from scratch. The exported ONNX has no runtime Jacobian teacher; policy
inference executes only the learned actor.

The loaded-color, planned-color, compression, and mode observation tail exists
for the environment contract and future extensions, but the current actor's
feature extraction stops at fields from the original 83-value layout and does
not consume `83:93`. Color changes and dip timing therefore come from the
authored sequence. If a dip is missed, the current actor has not learned to
detect and recover from it.

## Implementation

| File | Responsibility |
|---|---|
| `rlx/rlx/environments/brush.py` | Free brush, passive compression, physical wells, 93/15 contract, reward, and acceptance |
| `rlx/rlx/environments/brush_reference.py` | Fixed colored duck strokes, palette order, and approach/dip/lift trajectory |
| `rlx/rlx/models/drawing_feedback.py` | Structured feedback actor, ridge BC, and per-action exploration noise |
| `rlx/examples/ppo_microduck_brush.py` | Data, DAgger, BC, anchored PPO, ONNX export, and independent evaluation |
| `rlx/examples/render_microduck_brush.py` | Actual-contact rendering from ONNX or the diagnostic teacher |

## Simulation BOM and physics

These are educational simulation parameters, not a purchasing specification,
real material calibration, or hardware safety values.

| Item | Quantity | Simulation assumption |
|---|---:|---|
| Full-collision MicroDuck | 1 | Original 14 joints plus one simulated jaw action |
| Free brush | 1 | Held by upper/lower beak pads through `pencil_free`; not welded to the head |
| Brush shaft and ferrule | 1 each | Shaft replaces the pencil appearance; ferrule is non-contact decoration |
| Passive compliant tuft | 1 | `bristle_compression` slide over `-3.0..0.3 mm` |
| Decorative hair strands | 3 | Three capsules for appearance only; they do not contact |
| Contact tuft | 1 | One compliant sphere of radius `1.0 mm` represents the full tuft |
| Palette backing | 1 | Non-contact visual support |
| Physical paint wells | 4 | Gold, orange, charcoal, blue; tip contact changes pigment |
| Canvas and easel | 1 each | Inherited physical drawing scene |

The brush joint is an unactuated passive spring:

- stiffness: `25 N/m`
- damping: `0.32 N s/m`
- effective armature: `0.001 kg`
- default brush/beak-pad friction: `1.2`
- contact-tuft mass: `0.00003 kg`
- physics timestep: `0.002 s`, or `500 Hz`
- control timestep: `0.020 s`, or `50 Hz`
- `10` MuJoCo physics substeps per control action

The three visible hairs are not individual finite-element bristles. The
physical approximation is **three decorative hair strands plus one passive
compliant contact tuft**.

Pigment is also a discrete contact state, not fluid simulation:

- color loads only after actual tip/well contact above `0.0001 N`;
- canvas marks use the currently loaded color and actual MuJoCo tip contact;
- there is no fluid mixing, washing, residual-pigment ratio, absorption, or
  wash/pickup dynamics.

## 93 observations / 15 actions

The first 83 values preserve the pencil contract layout:

| Indices | Count | Value |
|---|---:|---|
| `0:3` | 3 | IMU angular velocity |
| `3:6` | 3 | Projected gravity in trunk frame |
| `6:20` | 14 | Original joint positions relative to defaults |
| `20:34` | 14 | Previous original-joint velocities |
| `34:48` | 14 | Previous original-joint action |
| `48:61` | 13 | Reserved twist/head/body command slots, fixed to zero |
| `61:64` | 3 | Mouth position, velocity, and previous action |
| `64:67` | 3 | Brush tip relative to trunk |
| `67:70` | 3 | Target minus tip in trunk frame |
| `70:73` | 3 | Brush-axis direction in trunk frame |
| `73:76` | 3 | Brush-tip velocity |
| `76:79` | 3 | Target trajectory velocity |
| `79` | 1 | Requested pen-down flag |
| `80:83` | 3 | Canvas, upper-pad, and lower-pad forces |

The ten brush-specific values are:

| Indices | Count | Value |
|---|---:|---|
| `83:87` | 4 | Loaded-color one-hot; all zero before loading |
| `87:91` | 4 | Planned-color one-hot |
| `91` | 1 | Tuft slide qpos in millimeters |
| `92` | 1 | Trajectory mode divided by two: travel, dip, or paint |

Actions remain 15 absolute normalized actuator offsets:

- `action[0:14]` controls the original 14 joints;
- `action[14]` controls the simulated mouth through
  `mouth_target = 0.24 + 0.36 * action[14]`;
- the head output is a bounded delta from the previous head action, limited to
  `0.025` per step;
- ridge head features contain head position, target error, brush direction,
  contact force, and quadratic head-position x target-error interactions;
- the previous head action is deliberately skipped as an input feature and is
  added only at the output, so the fit learns an action delta rather than
  copying the last action;
- ankles use linear gravity/angular-velocity feedback; remaining body joints
  and the jaw learn constant offsets.

## Data, BC, DAgger, and PPO

The final `brush-color-v2-03` pipeline was:

1. With data seed `11`, the analytic Jacobian teacher generated `47,984`
   records over eight training conditions.
2. Ridge BC fit the structured feedback actor with `ridge=0.01`.
3. Two DAgger rounds relabeled policy-visited states with the teacher and
   repeated the ridge fit.
4. With training seed `101`, PPO started from the BC/DAgger actor and collected
   `49,152` real environment steps.
5. PPO performed eight updates; all eight passed the anchor gate.
6. Deterministic 93x15 PPO and matching BC ONNX files were evaluated through
   the actual Studio API over 40 required episodes each.

Initial exploration standard deviations are separated by action type:

- head: `0.0003`
- leg/body: `0.0001`
- jaw: `0.0002`

PPO uses `learning_rate=1e-7`. After each update, the deterministic actor is
checked on up to approximately 4,096 BC/DAgger anchor observations. Maximum
action drift must remain `<=0.00025`; otherwise both policy parameters and
optimizer state are restored. All eight updates in this run were accepted.

No reward curriculum was changed to obtain this result. Brush training remains
unassisted and uses the tracking, contact, grasp, upright, action-rate, force,
and color reward terms. Anchored PPO limits movement away from BC/DAgger
behavior; it does not fabricate trajectories, contacts, pigment, or acceptance.

## Why the old drawing PPO failed

One historical pencil-run defect is confirmed: refinement configured
`initial_std=0.005`, but the post-BC/DAgger path unconditionally reset the PPO
starting `log_std` to `0.02`. Existing checkpoint and metadata evidence confirms
that configured-versus-effective mismatch.

Measurements also show that both old PPO runs drifted away from their BC
baselines: teacher-state action MSE, BC-to-PPO action drift, and cumulative KL
increased while drawing coverage fell. **That is measured degradation, not a
proven single root cause.** Missing BC anchoring, local rewards, short rollouts,
critic initialization, and recovery-state distribution shift remain hypotheses
that require controlled ablations.

The new brush run uses new physics, state, and actor structure plus low noise,
a very low learning rate, and update rollback. PPO and matching BC both passed
40/40, proving that the pipeline produced accepted BC behavior and PPO did not
destroy it. This does not prove PPO superiority or establish one historical
hypothesis as the sole cause.

## Acceptance contract

Every required episode must satisfy all of the following:

- coverage `>=95%` for each of gold, orange, charcoal, and blue;
- precision `>=95%` for each color;
- geometric tolerance of `1.5 mm`;
- complete all `119.96 s`;
- `grasp_fraction >=90%`;
- `max_normal_force_n <0.5 N`;
- `max_compression_m <3.5 mm`;
- `pen_up_leak_fraction <5%`;
- strict gold, orange, charcoal, blue palette order;
- no fall, drop, or assistance;
- both `null` and `open_jaw` negative controls must fail.

Required configurations:

| Condition | Parameters |
|---|---|
| nominal | Defaults |
| offset | Artwork offset `(0.002, 0.002) m` |
| scale | Artwork scale `0.95` |
| friction | Grip friction `0.9` |
| compliance | Brush stiffness `30 N/m` |
| combined_left | Offset `(-0.002, -0.001) m`, scale `1.04`, friction `0.95`, stiffness `23 N/m` |
| combined_small | Offset `(0.001, -0.002) m`, scale `0.92`, friction `1.1`, stiffness `32 N/m` |
| combined_right | Offset `(0.004, 0.001) m`, scale `1.06`, friction `0.9`, stiffness `28 N/m` |
| boundary_large | Offset `(0.005, -0.003) m`, scale `1.12`, friction `0.8`, stiffness `40 N/m` |
| boundary_soft | Offset `(-0.004, 0.003) m`, scale `0.88`, friction `1.0`, stiffness `18 N/m` |

## Exact commands

Run from the workspace root:

```bash
export PYTHONPATH=rlx:microduck_local/src
PY=rlx/.venv-microduck/bin/python
BRUSH=rlx/examples/ppo_microduck_brush.py
DATA=rlx/runs/brush-data/final-seed11
RUN=rlx/runs/studio/drawing/brush-color-v2-03
```

Generate the eight teacher episodes:

```bash
$PY $BRUSH data \
  --out "$DATA" \
  --seed 11
```

Train ridge BC, two DAgger rounds, and anchored PPO:

```bash
$PY $BRUSH train \
  --out "$RUN" \
  --data "$DATA/teacher.npz" \
  --seed 101 \
  --steps 49152 \
  --dagger 2 \
  --ridge 0.01 \
  --learning-rate 1e-7 \
  --anchor-limit 0.00025
```

Run the independent evaluation:

```bash
$PY $BRUSH eval \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/eval.json" \
  --seed 10001 \
  --episodes 4
```

Run the identical 40-episode gate on matching BC:

```bash
$PY $BRUSH eval \
  --onnx "$RUN/bc.onnx" \
  --out "$RUN/bc-eval.json" \
  --seed 10001 \
  --episodes 4
```

Render the learned ONNX:

```bash
$PY rlx/examples/render_microduck_brush.py \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/render" \
  --seed 501 \
  --seconds 120
```

Render the teacher only as a diagnostic:

```bash
$PY rlx/examples/render_microduck_brush.py \
  --teacher \
  --out /tmp/microduck-brush-teacher \
  --seed 501 \
  --seconds 120
```

Check current source hashes against `eval.json`:

```bash
shasum -a 256 \
  rlx/examples/ppo_microduck_brush.py \
  rlx/rlx/environments/brush.py \
  rlx/rlx/environments/brush_reference.py \
  rlx/rlx/models/drawing_feedback.py \
  rlx/rlx/environments/drawing.py \
  rlx/rlx/environments/drawing_reference.py

jq '.source_hashes, .training_source_hashes, .provenance_errors,
    .unique_conditions, .environment_source_match,
    .training_pipeline_source_match, .source_sha256,
    .source_files_sha256' "$RUN/eval.json"
```

If ONNX is missing or needs the static 15-action shape repair, it can be
re-exported from the retained checkpoint. The environment, reference, actor,
and base sources must still match checkpoint metadata, and
`sources/ppo_microduck_brush.py` must match the embedded training pipeline
hash. An evaluator-only change to the current `ppo_microduck_brush.py` does not
require retraining. Re-export creates new ONNX bytes, so rerun both PPO and BC
evaluations and record the new immutable hashes:

```bash
$PY $BRUSH export \
  --checkpoint "$RUN/policy.zip" \
  --onnx-output "$RUN/policy.onnx"

$PY $BRUSH eval \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/eval.json" \
  --seed 10001 \
  --episodes 4

$PY $BRUSH export \
  --checkpoint "$RUN/bc.zip" \
  --onnx-output "$RUN/bc.onnx"

$PY $BRUSH eval \
  --onnx "$RUN/bc.onnx" \
  --out "$RUN/bc-eval.json" \
  --seed 10001 \
  --episodes 4
```

If the environment, reference, actor, base sources, or archived training
pipeline do not match, the strict provenance gate rejects the artifact.
Re-export cannot repair that training-source mismatch; regenerate the teacher
dataset from the new source and train into a new immutable run directory. The
only permitted exception is a current evaluator-pipeline change with a valid
archived training script. In that case the report records
`training_pipeline_source_match=false` instead of pretending the training and
evaluation scripts were identical.

## Evidence and limits

- PPO evidence: `rlx/runs/studio/drawing/brush-color-v2-03/eval.json`
- Matching BC evidence: `rlx/runs/studio/drawing/brush-color-v2-03/bc-eval.json`
- Training record: `rlx/runs/studio/drawing/brush-color-v2-03/training.json`
- PPO history: `rlx/runs/studio/drawing/brush-color-v2-03/ppo_history.csv`
- ONNX: `rlx/runs/studio/drawing/brush-color-v2-03/policy.onnx`
- Source archive: `rlx/runs/studio/drawing/brush-color-v2-03/sources/`
- Studio API receipt: `docs/drawing-case/brush/evidence/workflow.json`
- Preliminary visual check: `brush-color-v2-01/render/painting.png` shows the
  normal swimming duck; it remains only as historical visual evidence
- Final03 source-bound render:
  `docs/drawing-case/brush/media/render.json`
- Documentation image:
  `docs/drawing-case/brush/media/painting.png`
- Documentation animation:
  `docs/drawing-case/brush/media/brush.gif`
- Browser check: `docs/drawing-case/brush/evidence/viewer.json`, `passed=true`, within the scope above
- Final verification checklist: `docs/drawing-case/brush/VERIFICATION.md`

The result covers only the current CPU MuJoCo model, fixed authored artwork,
and listed perturbations. It includes no camera input, unknown-image planning,
real fluid paint, individual finite-element bristles, hardware calibration,
purchasing recommendation, or deployment claim.
