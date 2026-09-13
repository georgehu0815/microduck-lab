# Ground tennis-ball return: 75 mm wide-gripper candidate

## User-approved half-height release profile — September 13, 2026

The user now permits opening the gripper inside the bin and allowing a gravity
drop, rather than requiring bottom support before opening. This is a separately
versioned task, `tennis-half-height-gravity-return-v2`; old supported-release
results remain historical evidence and are not relabeled.

Before opening, the complete sphere must be geometrically inside the bin and its
center no higher than 50 mm above the inner bottom (54 mm world height). A lower
release, including bottom-supported release, is also allowed. Measured velocity
must pass the recorded release gates; upward/lateral throwing is not permitted.
After opening, gravity-driven downward motion is allowed. Final acceptance still
requires bottom support, no robot contact, retreat clearance, low speed and a
continuous two-second settled interval. Robot/bin collisions still fail.

Post-release numerical side contact is distinct from escape: an overrun of at
most 0.5 mm requires the corresponding actual, nonpositive-distance ball/wall
contact pair, bounded penetration and unchanged vertical containment. This is
an explicit simulation tolerance, not a physical manufacturing allowance.
Initial release continues to require strict geometric containment.

The bounded simulator-truth planner screens complete trajectories before
selecting a walking warm-up, speed and gain profile. It uses no seed lookup,
root teleportation or external-force injection. **This is an offline teacher,
not real-time MPC or new whole-body PPO training.**

### Current unified result: v13, 30/30 simulated returns

The complete `half-height-planned-v13` matrix passed **30/30**, with three ball
sizes and seeds 0–9 per size. Both negative controls correctly timed out without
success. The six-case v11 screen remains 6/6 and historical v12 remains 29/30;
neither was combined with diagnostics to obtain this result.

| Check | Unified v13 result |
|---|---:|
| Nominal / small / large ball | 10/10 / 10/10 / 10/10 |
| Ground pickup and complete return | 30/30 |
| Open-jaw / hold negative controls | 2 expected failures; 0 false successes |
| Simulated task time: minimum / median / maximum | 46.208 / 55.299 / 79.998 s |
| Navigation candidate trials, including rejected forecasts | 107 |
| Verified complete videos | 32: 30 returns + 2 negative controls |
| Recorded control steps compared / physics substeps monitored | 94,765 / 947,532 |
| Related full regression | 465 passed, 10 warnings |
| New PPO training steps | 0 |
| Hardware release | No |

The last v12 failure, small-ball seed 5, now completes in 2,575 steps. The repair
adds a 3.25-second walking warm-up to the **common 15-profile candidate list**,
not a seed-specific choice. After pickup, each candidate is evaluated through
the remaining carry/docking/release/retreat sequence before selection. Rejected
forecasts remain in each episode's `navigation_prediction` evidence. A candidate
prediction cannot overwrite the live episode's success. Seeds 0–9 are regression
cases used during development, not an independent held-out generalization set.

**Release-height limitation:** measured authorization heights range from
36.493 to 38.290 mm above world zero, below the 54 mm allowed maximum. All 30
episodes record **zero post-release unsupported free-fall duration**. This proves
successful bin returns under the lower-or-half-height contract; it does **not**
demonstrate dropping the ball from half-height. Lower/bottom-supported release
was explicitly retained as a valid outcome.

The controller coordinates **15 actuators: 10 leg, 4 arm-pose, and 1 gripper**;
there is no waist actuator. It combines the existing leg ONNX policy, arm IK,
feedback and simulator-truth trajectory screening. No BC/DAgger/PPO training or
new ONNX export occurred. Reward/loss curves are therefore not applicable.

The inherited evaluator allows a 0.08 rad excursion beyond nominal joint limits;
the maximum observed excursion in v13 is 0.013041 rad. Passing this evaluator
is not strict nominal-limit qualification. The 80 s horizon does not
prove the original 15 s performance target. The tennis payload class remains
above the original 20–30 g design target. Electrical, thermal, structural,
real-time execution and physical robot validation remain open.

### Reproduction and evidence

Run from the workspace root with MuJoCo 3.10.0 and the existing environment:

```sh
export PYTHONPATH=.:microduck_local/src
export OMP_NUM_THREADS=1
PY=rlx/.venv-microduck/bin/python
BASE=artifacts/microduck-arm-v1c/tennis-return
$PY -m microduck_arm_experiments.tennis_return \
  --gripper wide_candidate --release-mode half_height \
  --plan-navigation --max-steps 4000 --workers 12 \
  --out "$BASE/half-height-planned-v13-reproduction"
```

Use a new output directory; do not overwrite historical evidence. Full planning
is offline and may be much slower than simulated time. The complete measured
matrix is `half-height-planned-v13/evaluation.json` beneath `BASE`, alongside all
32 episode directories and the source snapshot. Final regression output is
`repair-development/half-height-v13-release-regression.log`.

**All 32 videos are complete and verified** in `half-height-v13-videos/` beneath
`BASE`. Open its `videos.html` for a bilingual, individually playable video
gallery, or `README.md` for the per-case MP4 index. Each episode also contains
its raw telemetry, result, and `replay-validation.json`.

The videos are **recorded-action physics replays**, not rendered pose animation
or a second independent policy evaluation. Every recorded command is applied to
a freshly reset MuJoCo environment; the original telemetry is preserved
byte-for-byte. Over all 32 episodes, replay compared 94,765 control steps and
monitored 947,532 physics substeps. Maximum recorded-time, ball-position,
joint-position and action-round-trip differences were all **zero**.

The separate video verifier checked the sampled 50 Hz acceptance envelope,
including release, containment, safety, free-fall limits and final two-second
settling, then fully decoded every MP4. Videos are 640 × 480 at 25 fps, with
simulation/replay labels. This sampled verification and the replay's substep
monitoring are distinct evidence, not interchangeable claims.

Key files under the video bundle:

- `evaluation.json`: video-bound aggregate report.
- `evidence-summary.json`: measured summary and report hashes.
- `video-verification/manifest.json`: 32/32 decode and integrity checks.
- `video-verification/contact-sheet.jpg`: five sampled frames per episode.
- `video-verification/terminal-contact-sheet.jpg`: the actual final frame of all 32 episodes.

To reproduce videos from the new metrics directory:

```sh
$PY scripts/replay_tennis_evidence.py \
  --source "$BASE/half-height-planned-v13-reproduction" \
  --out "$BASE/half-height-v13-videos-reproduction" --workers 1
$PY scripts/verify_tennis_videos.py \
  --root "$BASE/half-height-v13-videos-reproduction"
```

On this Mac, one four-worker renderer stalled in a Metal/OpenGL cache file lock
after 31 videos completed. The missing episode was recovered serially, using
`--resume` only after verifying retained video hashes, original source bindings,
step counts and errors. Interrupted logs and the process stack are retained in
`repair-development/half-height-v13-render-recovery.md` and its companion files.
The final verifier includes the recovered video; no missing episode was omitted.
The resume checker also rejects aliased output files and requires the exact
source-model physics-substep count, including a shortened terminal control step.

## Historical supported-release v9 result

Date: 2026-09-13. **Historical complete metrics matrix: `wide-repair-v9-screen`, 11/30
complete returns under the existing evaluator tolerance and 30/30 ground lifts.
The task matrix still fails. Intermediate v7 reached 9/30; historical v6 reached 4/30.
The independent `wide-repair-v9` rerun now has 32/32 declared videos fully decoded
and verified; its positive-task labels reproduce 11/30, not 32 task successes.
This is not strict nominal joint-limit qualification, PPO training success,
real-time deployment validation, or hardware release.**

## v9 repair, verification, and remaining failures

| Check | Complete metrics matrix |
|---|---:|
| Positive tasks | 11/30 |
| Nominal / small / large | 5/10 / 5/10 / 1/10 |
| Real ground lifts | 30/30 |
| Open-jaw / hold negative controls | 0 false successes |
| Fully decoded videos | 32/32; 11 positive-task successes |
| Related code regressions | 372 passed, 10 warnings |
| New PPO updates | 0; reward/loss are not applicable, not fabricated |
| Hardware or manufacturing qualification | None |

Successful seeds: nominal **0, 2, 3, 5, 6**; small **0, 2, 3, 6, 8**; large **8**.
The other 19 episodes contain **10 robot/bin collisions, 4 timeouts, 4 premature
grasp losses, and 1 non-pad robot/ball collision**. Successes increased relative
to v6, but bin collisions increased from 3 to 10. **Higher completion rate is not
an across-the-board safety improvement.** This remains an unreleased candidate.

### Implemented changes

1. Preserve the original leg policy until a movement command produces less than
   5 mm of measured progress for 2.5 s. Recovery adds a bounded 0.06 rad, 1 Hz hip
   roll motion. The existing 6 rad/s leg command slew limit remains unchanged.
2. Before lowering, forecast 25, 30, 35, and 40 mm crouch candidates on independent
   simulator copies. Authorize a docking posture only after a copy completes the
   entire placement, release, retreat, and stable-hold acceptance sequence.
3. Compare legacy release with a held-arm, active-leg, 0.025 rad/s slow-opening
   alternative. If neither predicts success, do not authorize jaw opening; never
   label a least-bad failing candidate as successful.
4. Slow opening screens both measured arm pose and the effective held command
   pose, plus jaw sweep, bottom support, whole-ball containment and prohibited
   contacts. Forecast results cannot overwrite the live episode's success flag.

**This is a privileged simulator-truth clone teacher**, not newly learned PPO or
a validated real-time MPC deployment. Each docking candidate forecasts at most
2000 control steps, and each release candidate at most 800. Offline computation
time is not evidence that the robot can run this planner at 50 Hz. Metadata states
`forecast_source=simulator_truth_clone_teacher`, `ppo=false`, and
`hardware_eligible=false`. Forecasts omit the renderer and independently copy
model, dynamics state, and policy history. They never move live free roots or
inject external forces.

The coordinated controller still commands **15 actuators: 10 legs, 4 arm-pose
joints, and 1 gripper**, with no waist actuator. This is teacher/control
verification, not completed whole-body PPO training.

All three MJCF files are **byte-identical to v6**, and frozen v1/v1-C sources are
unchanged. Ball, bin, contacts, mass, actuator force limits and task gates were
not relaxed. The 80 s horizon does not satisfy the old 15 s target. The inherited
joint gate still permits 0.08 rad; the matrix measured a maximum excursion of
approximately **0.012744 rad** beyond nominal ranges. Strict nominal limits,
electrical/thermal/inertial measurements, the 58 g payload class, and real-time
deployment remain unqualified.

### v9 evidence locations

- Complete metrics: `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9-screen/evaluation.json`
- Per-episode `result.json`, `telemetry.jsonl`, and `source-snapshot/` share that root.
- Regression log: `artifacts/microduck-arm-v1c/tennis-return/repair-development/regression-final-v9.log`
- Full-video rerun: `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/`
- Video verification manifest:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/manifest.json`
- Video contact sheet:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/contact-sheet.jpg`
- Terminal-frame review sheet:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/final-frames-contact-sheet.jpg`
- Video status: **32/32 declared videos passed hash, terminal-telemetry,
  frame-count, and full-decode checks; 11/30 positive tasks are successes**.

The final-frame visual review agrees with the recorded labels: all 11 pass-labeled
positive episodes show the ball resting in the bin, while the remaining positives
and both controls end in visibly non-pass states. This qualitative review does not
relabel any episode or convert video integrity into task acceptance.

### Release-profile boundary after the v9 freeze

The verified v9 bundles remain historical evidence for the **strict bottom-supported
release** contract documented below. The separately identified `half_height`
profile and its new evidence are documented at the start of this guide. Its
results are not included in, or substituted for, the v9 11/30 result.
Because the active specification is changing, no working-tree specification hash is
used as a comparison claim against frozen v9; frozen source snapshots remain the
historical comparison boundary.

An empty `--video-seeds` deliberately produces a `metrics_only` bundle, which the
video verifier rejects. Explicit seeds produce `metrics_and_video` evidence; the
verifier requires every declared seed for all three variants plus both controls,
including hashes, terminal telemetry agreement, and full video decoding. Historical
reports without `video_seeds` retain the five-video seed-0 contract.

Foot-clearance projection, velocity-based braking, always-on hip sway, and short
horizon navigation prediction have separate diagnostic records. Their individual
successful trials are not promoted into this matrix or a general repair claim.

## Design decision

The old jaws are not permanently fixed. A separate, replaceable wide candidate
now has a 75 mm open inner-pad gap, while the original 19 mm jaws and all frozen
v1-C source, models and policy artifacts remain unchanged.

The experiment uses a 67 mm, 58 g nominal ball and two Type 1/2 boundary pairs:
65.4 mm / 56 g and 68.6 mm / 59.4 g. These pairs are not a full Cartesian tolerance
matrix and exclude larger Type 3 balls. The experiment specification records the
ITF Appendix I source. Actual felt friction, compliance and inertia are unmeasured.

| Parameter | Original | Wide candidate |
|---|---:|---:|
| Open inner-pad gap | 19 mm | 75 mm |
| Gear/finger shaft spacing | 24 mm | 24 mm, unchanged |
| Outward pad offset per side | 0 | 28 mm |
| Pad-center local x | 23 mm | 48 mm |
| TCP local x | 30 mm | 55 mm |
| Contact-pad size | 24 × 5 × 8 mm | unchanged |
| Arm actuators | 5 | 5 |
| Whole robot actuators | 15 | 15 |
| Link-plus-tool geometric length | 135 mm | 160 mm |

There are ten leg actuators, four arm-pose actuators and one gripper actuator;
there is no waist actuator. The original mirrored jaw coupling and motor
envelopes remain intact. The width comes from offset fingers, not an impossible
change in gear centers with unchanged gears.

Widening alone would let a ball centered at the old 30 mm TCP intersect the palm
or motor. Moving the TCP forward to 55 mm avoids that. The supporting stems were
also moved outward to ensure the pads contact the ball before the rigid stems.

Added structural blank volume corresponds to approximately 4.20 g at an assumed
PA12 density of 1100 kg/m$^3$. This excludes unresolved horn/shaft attachment,
retention and fasteners. The OpenSCAD and SVG/PDF are review-only geometry, not
manufacturing drawings with validated holes, fits, tolerances or strength.

## Closing-angle interpretation

For simplified symmetric straight fingers with 40 mm effective length:

$$\theta=\arcsin\frac{75-D}{2\times40}$$

A 66 mm ball gives approximately 6.46° per finger, or 12.92° relative closure.
This approximation must not be used directly as the offset-finger servo command.
For the actual candidate, a planar face-distance approximation in metres is:

$$0.0255+0.012\cos\theta-0.055\sin\theta=D/2$$

With the ball centered at the TCP and no deformation, a 66 mm ball gives about
4.65° per finger. A 0.001 rad MuJoCo kinematic sweep measured first bilateral pad
intersections at 4.98° / 4.18° / 3.32° for 65.4 / 67.0 / 68.6 mm balls.
All three open snapshots are clear of ball/robot intersections; only the pads
touch at first bilateral closure. **This is not dynamic grasp or force validation.**
The proposed ±33 mm contact locations describe an equatorial grasp on a 66 mm
ball, not the shaft spacing.

## Task and acceptance

The ball starts on the floor. Both ball and robot retain free joints; no weld,
suction, externally injected force or raised ball is used. The classroom bin is
anchored, with 180 × 180 mm inner dimensions, 100 mm walls and 4 mm bottom/wall
thickness. Its center is roughly 0.4 m from the robot's origin. This is not a
validated full-height household bin.

Sequence: approach → bilateral grasp → lift → carry → lower → bottom-supported
release → withdraw → remain stable.

- The initial ball must be on the floor outside the bin; spawning in the bin
  cannot count as a return.
- Lift the ball bottom at least 20 mm, maintaining bilateral grasp for 0.5 s.
- Carry at least 120 mm horizontally.
- The entire ball must fit inside the bin, not just its center; rim support does
  not count.
- Establish bottom support before releasing, with ball speed at most 0.03 m/s.
  Throwing and premature release fail.
- Withdraw beyond the ball radius plus 20 mm, with no robot/ball contact; retain
  the ball stably on the bottom for 2 s.
- Reject falls, self-collision, robot/bin collisions, joint violations and
  non-finite telemetry.

The ball uses analytical thin-shell inertia and uncalibrated rigid contact. Its
actual mass is modeled in this separate experiment; the old small-object
scenario's 50 g construction limit was not changed.

## Frozen v6 results and limitations

The evaluation is complete. These numbers describe only the frozen
`wide-repair-v6` bundle. Predictive-navigation / MPC and SOUTH-route experiments
are not integrated into this bundle and have not established a fully qualified
improvement. Source/controller review continues; this is a frozen historical
measurement, not a working-tree performance guarantee or release approval.

| Check | Frozen v6 result |
|---|---:|
| Wide-gripper geometry at three ball sizes | 3/3 |
| Complete returns under the existing tolerance gate | 4/30 |
| Ground lifts | 30/30 |
| Nominal / small / large complete returns | 4/10 / 0/10 / 0/10 |
| Open-jaw and hold negative controls | 0 false successes |
| Fresh verified videos | 5; only `nominal-0` is a task success |
| PPO updates for this repair | 0 steps |
| Hardware qualification | None |

The successful records are nominal seeds **0, 2, 3 and 5**, completing at
52.846, 51.650, 54.154 and 53.988 s respectively. Only seed 0 has a video;
the other three successes are result/telemetry records, not additional videos.
The evaluation horizon is 4000 control steps, or 80 s at 50 Hz; v2 used 750
steps, or 15 s. These completions **do not meet the old 15 s
deadline**. The physical acceptance rules above are unchanged: ground start,
bilateral grasp and lift, 120 mm carry, whole-ball containment, bottom-supported
slow release, withdrawal, two-second stability and all safety rejections still
apply. The bundle remains `all_tasks_passed: false`.

The 26 failures are 18 `timeout`, 3 `robot_bin_collision`,
3 `premature_release_or_throw`, 1 `non_pad_ball_collision`, and
1 `unsupported_or_fast_release`. Both negative controls time out without success.

**Joint-limit review caveat:** the evaluator already allowed a **0.08 rad**
joint-limit tolerance; neither v5 nor v6 introduced it. Review found that successful
nominal episodes exceed the nominal 1.8 rad elbow limit by approximately
0.009–0.011 rad under MuJoCo soft limits. Thus 4/30 passes the existing legacy
tolerance gate, **not strict nominal joint-limit qualification**. Do not infer
that every physical constraint is perfectly satisfied or that hardware is safe.
V6 now records `maximum_nominal_joint_excursion_rad` on each telemetry frame,
alongside `evaluator_joint_tolerance_rad`, making nominal-limit overshoot explicit
without tightening the legacy acceptance gate.

The robot has 15 actuators: 10 legs, 4 arm-pose joints and 1 gripper, with
0 waist actuators. `tennis_controller.py` adds a coordinated whole-body
kinematic teacher around the legacy `alpha_walking.onnx` leg policy. It uses
actual-state feedback for crouching, downward ground IK, a folded loaded lift,
loaded navigation and collision/torque-screened placement. This is not a newly
trained end-to-end policy and not new PPO.

V6 retains the v5 placement crouch of **25 mm** and caches
the last safe release IK target if the new target is unreachable or its shoulder
pitch is **>= 0.62 rad**. In the blocked-plan fallback, the gripper opens after
at least **1 s**, after ball speed has remained **< 0.001 m/s for 0.3 s**, and
only after the additional cached-pose release checks pass.

V6 formalizes the blocked-plan stage as `release_cached_pose`. Before opening
from the cached pose, it requires actual bottom support, whole-ball containment
inside the bin and no forbidden contacts, plus a **nine-position jaw-opening
collision sweep on scratch data**, not the live robot. The sweep allows intended
pad/ball contacts and rejects other penetrating arm contacts. This discrete
kinematic screen is not a continuous-motion or hardware safety proof.
`release_plan_blocked` and `cached_release_screen_passed` are now explicit
telemetry fields; the unrecorded fallback limitation belongs to historical v5.

The physical monitor still checks bottom support, release speed, containment,
withdrawal and stability. Its acceptance thresholds and the motor limits are
unchanged. The nominal, small and large scene XML files have identical SHA-256
hashes across v2, v5 and v6; this verifies unchanged scene files, not hardware
qualification or every source dependency. This frozen run is not a release or
a claim that all constraints are perfect.

The frozen regression log reports **347 passed, 10 warnings**. It covers the repair while
retaining the acceptance defenses, including rejection of false initial-bin
success and unsafe release behavior. Passing tests does not convert the 4/30
matrix into task acceptance or strict joint-limit qualification.

The ball alone produces approximately 0.091 N·m at a 160 mm horizontal lever arm,
before arm, fingers, acceleration and safety margin. The old 0.1 N·m simulation
screening limit is not a qualified continuous actuator rating.

Next engineering gates remain loaded wide-jaw bench testing, collision-free
wrist orientation and whole-body crouch/reach, reliable loaded navigation and
placement, then demonstrations and whole-body learning. This experiment has
**zero new PPO training steps** and no task-qualified tennis-return ONNX export.
Old 10 g task exports do not prove a 58 g tennis-ball capability. Nothing in v6
was run on hardware.

## Historical v5, v4 and v2 results

Keep earlier evidence for comparison, not as a description of v6:

- Frozen `wide-repair-v5`: 4/30 complete returns under the same legacy joint-limit
  tolerance, 30/30 ground lifts, 4000 steps / 80 s. Successes are nominal seeds
  0, 2, 3 and 5. Its five verified videos contain only one success (`nominal-0`);
  its regression log records 302 passed. The blocked-plan cached release was
  not explicitly recorded in telemetry and lacked v6's formal release screen.

- Frozen `wide-repair-v4`: 1/30 complete returns, 30/30 ground lifts, 4000 steps /
  80 s; nominal seed 0 completes at 56.228 s. Its five verified videos contain
  one success and four failures. Its regression log records 297 passed.

- `wide-screen-v2` completed 0/30 tasks.
- Its five videos (`nominal-0`, `small-0`, `large-0`, `open_jaw`, `hold`) are
  all failure videos.
- That run used the earlier 750-step / 15 s horizon and predates the coordinated
  v4 crouch, folded-lift, loaded-navigation and placement controller.

## Keep the old results separate

The original teacher matrix now passes 120/120 free-base and 80/80 fixture
episodes. However, the actual teacher-plus-whole-body-residual ONNX matrix passed
**100/120**, not 120/120. Remaining failures are primarily stowed walking and
walking carry. Those models use 69 observations and 15 residual outputs and
retain their teacher. This candidate does not modify those frozen artifacts or
claim independent learned competence.

## Artifacts and reproduction

- Sources: `microduck_arm_experiments/tennis_return.py` and
  `microduck_arm_experiments/tennis_controller.py`
- Specification: `hardware/microduck-arm-v1c/experiments/tennis-return.json`
- CAD/dimensions: `hardware/microduck-arm-v1c/experiments/tennis-wide-gripper/`
- Stock results: `artifacts/microduck-arm-v1c/tennis-return/stock-screen-v2/evaluation.json`
- Historical wide result:
  `artifacts/microduck-arm-v1c/tennis-return/wide-screen-v2/evaluation.json`
- Historical v4 result:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v4/evaluation.json`
- Historical v5 result:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v5/evaluation.json`
- Frozen v6 result:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/evaluation.json`
- Frozen v6 regression:
  `artifacts/microduck-arm-v1c/tennis-return/repair-development/regression-final-v6.log`
- Frozen v6 media verification and provenance hashes:
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/video-verification/manifest.json`

The five exact frozen v6 video paths are:

- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/nominal-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/small-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/large-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/open_jaw/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/hold/rollout.mp4`

The verifier fully decodes exactly those five videos, checks frame counts
against telemetry, requires each `result.json` to equal its evaluation record,
and requires the terminal telemetry outcome to equal the recorded success and
failure fields. It does not inspect a failed video and relabel it as successful.
The manifest records one verified success (`nominal-0`) and four verified
failures.

The manifest retains full SHA-256 hashes for the evaluation, episode results,
telemetry, videos, source snapshots, verifier and contact sheet. It establishes
internal hash consistency, not cryptographic attestation, independent physics
re-evaluation or strict joint-limit qualification.

Exact v6 SHA-256 values from that manifest:

- Evaluation: `91a6da825f7d73ec21b0daf1cb01d7659c16cf6bda3b4474cbe59e8031c0597c`
- Controller snapshot: `fa8cad9e55ac71fc8b99d8e179c62a08c266498b56f1d1ec7445f970f65e33a3`
- Recorded verifier: `15a2bb0c32e1b4fc64e5d84b4f91ad49fba376ff789b99301380ec01fc2e79c8`

\newpage

From the repository root, use a fresh output path. These commands run the
working-tree controller; they do not promise to reproduce frozen v6 after later
source changes. Compare the new source hashes with v6 before attributing results.

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
rlx/.venv-microduck/bin/python -m microduck_arm_experiments.tennis_return \
  --gripper wide_candidate --out artifacts/tennis-new-run \
  --max-steps 4000 --workers 4 --video-seeds 0

PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_microduck_arm_tennis_return.py -q

PYTHONPATH=.:microduck_local/src \
rlx/.venv-microduck/bin/python scripts/verify_tennis_videos.py \
  --root artifacts/tennis-new-run
```

The evaluation CLI now exits 1 when any positive task fails or a negative
control succeeds; the frozen v6 matrix therefore has a nonzero task status even
though it wrote complete evidence. Run the verifier separately after the
evaluation command; shell automation must account for the expected nonzero exit
when collecting a partially successful diagnostic matrix. Passing unit tests or
video-integrity checks is not task acceptance. On macOS, offscreen rendering
needs host graphics access; omitting `--video-seeds` runs without videos.
