# MicroDuck Single-Arm v1-B — implementation and evidence

## Decision: prototype delivered; integrated robot release blocked

This is an executable **engineering-validation prototype**, not a fabrication-ready robot or a claim that every task passed. The current reference evaluation is `artifacts/microduck-arm-v1/evaluation-attempt-3/evaluation.json`. Attempts 1 and 2 are historical, with older environment hashes; do not combine their results with the current revision.

Software tests, simulated task success, artifact integrity, and physical acceptance are four different claims. In particular, a fixed-root reach success does not establish walking, carrying, balance, electrical safety, or hardware readiness.

Final targeted software run: **137 passed** (42 new-package tests and 95 previous-arm regression tests), with seven ONNX-export/tracing warnings. Evidence: `artifacts/microduck-arm-v1/software-tests.log` and `software-tests.xml`. This does not mean the physical task suite passed.

## Implemented package

| Deliverable | Repository location | Qualification |
|---|---|---|
| Canonical dimensions, masses and joint contract | `hardware/microduck-arm-v1/spec/design.json` | Assumptions and unknowns are explicit |
| BOM and assembly/test procedure | `docs/microduck-arm-v1/BOM.md`, `SOP.md` | Exact motor candidate; other procurement interfaces remain TBD |
| Editable CAD and dimension drawings | `hardware/microduck-arm-v1/cad/` | SCAD, SVG and PDF review geometry; no released mating holes or STEP/STL |
| Circuit, wiring and netlist | `hardware/microduck-arm-v1/electrical/` | Logical topology; not a fabrication-ready PCB or qualified harness |
| Integrated dynamic model | `hardware/microduck-arm-v1/generated/microduck-arm-v1-free.xml` | Free robot root; physical task-object contacts |
| Explicit bench model | `hardware/microduck-arm-v1/generated/microduck-arm-v1-fixture.xml` | Fixed-root assistance, never a mobile validation substitute |
| URDF | `hardware/microduck-arm-v1/generated/microduck-arm-v1.urdf` | Kinematic interchange with documented backlash/collision approximations |
| Model, workspace, torque and task validation | `microduck_arm_v1/validation.py` | Executed; failures preserved |
| BC, residual PPO, ONNX and evaluation | `microduck_arm_v1/learning.py` | Simulation training tools, not an onboard controller |
| Evidence audit and video gallery | `microduck_arm_v1/evidence.py` | Hash, trace and video-frame integrity checks |
| Hardware acceptance | `docs/microduck-arm-v1/HARDWARE-ACCEPTANCE.md` | Not run; missing evidence blocks release |

## Design consistency and deliberate corrections

- **Four pose axes plus one gripper = five actuators**, not “5 pose DOF plus gripper.” Ten retained leg actuators plus five arm actuators give a new **15-action / 64-observation** contract. Existing 14-action / 61-observation policies must not be loaded into this model.
- v1-B uses five identical `XL330-M288-T` candidates. This retains one motor identity and interface definition throughout the new package. It does not prove torque adequacy or imply that an M077 substitution was validated.
- Nominal axis distances are **55 / 50 / 30 mm**, totaling 135 mm. This geometric sum is not a certified payload reach or balanced workspace.
- The gripper uses two rotary fingers and a -1 gear coupling. MJCF equality and URDF mimic describe this proposed transmission; real gears, backlash, bearings and strength still need engineering.
- Motor envelopes are 20 × 26 × 34 mm in the model's width/depth/height convention. The old visual mesh depth differs; neither visual geometry nor a bounding box supplies trustworthy mounting-hole coordinates.
- The original model's removed neck subtree weighs 0.2799274 kg, not merely an empty head shell. Electronics, battery, power hardware and sensor shell are therefore explicitly reintroduced as **unmeasured mass assumptions**, rather than silently deleted.
- The resulting modeled robot mass is **0.80631578 kg**, excluding the task object. This is not a weighed article. No mass/CoM claim is hardware-qualified.
- The arm positive power domain is kept separate from the body positive domain. Final pack/BMS/DC-DC/fuse/connector selections, regenerative-energy protection and thermal tests remain unresolved. No real actuation or onboard software change was performed.

## Actual simulation results

Teacher: IK plus a time-phased state machine. Seeds: 910000 and 910001. Limit: 1,500 control steps, 0.02 s per step; episodes end early on success or failure. Payload in the task suite: 10 g. It is a modeled task object, not a rated payload.

| New case | Free-base result | Fixture result | Observed limitation |
|---|---:|---:|---|
| Reach | 0/2 | 2/2 | Free-base fall |
| Pick/place | 0/2 | 0/2 | Environment contact / robot self-collision |
| Obstacle relocation | 0/2 | 0/2 | Environment contact / robot self-collision |
| Stationary waypoint carry | 0/2 | 0/2 | Environment contact / robot self-collision |
| Stowed-arm walking | 0/2 | Not allowed | No working gait teacher; environment contact |
| Walking while carrying | 0/2 | Not allowed | Environment contact; no validated whole-body controller |
| **Total** | **0/12** | **2/8** | **Integrated task release fails** |

The two fixture reach episodes end with TCP errors of **4.63 mm and 4.09 mm**, and full rotation errors of **0.0457 and 0.0427 rad**. The acceptance window also requires a one-second hold. These are two seeded simulation outcomes, not an accuracy guarantee or a 40/40 campaign.

The previous lab's single-arm reach/pick/relocate/carry tasks have corresponding new scenarios, but their old pass records cannot transfer to a different morphology. `arms-handover-v1` and `arms-co-carry-v1` require phase two and are not implemented on a single arm. Painting, DAgger, color coverage/precision, and real contact-force qualification are also not implemented here.

### Acceptance protections

Forbidden contacts are latched across physics substeps. Reach checks a full target rotation, not just one tool axis. Grasp loss after a valid lift is failure unless release is authorized in a valid placement phase; authorization is revocable and unavailable to walking/carrying. Walking requires recent ordered alternating support events while upright and making forward progress. Stowed walking checks arm posture; carrying checks current lift, bilateral grasp, absence of table/floor support, and a bounded body-relative object pose.

The default 30 s horizon includes the teacher's 21 s release phase and one-second hold. Shorter runs may be deliberate smoke tests, not complete manipulation evaluations. The current gait teacher merely commands the home pose; it is not a locomotion implementation.

### Workspace and torque screening

- 10,000 seeded target samples in the specified Cartesian box: **3,331** meet the constrained horizontal-tool IK test; **2,831** additionally pass the collision screen.
- Balance-safe count is **unknown**. These counts are not a usable-workspace rating and do not establish arbitrary six-dimensional reachability.
- A 500-pose quasistatic table with an assumed 20 g TCP payload gives a maximum required qualified continuous torque at the 3× screening factor of **0.33063 N·m**. The table includes infeasible poses, explicitly flagged.
- Actual qualified continuous motor capability is unknown. The model's ±0.10 N·m actuator limit is a screening assumption, not a manufacturer continuous-duty rating. Dynamic acceleration, contact loading, heating and battery behavior are not qualified.

Raw files: `artifacts/microduck-arm-v1/workspace.json` and `artifacts/microduck-arm-v1/mechanics/torque-com.csv`.

## Training and video evidence

The learning smoke uses successful fixture reach demonstrations, BC, residual PPO with an external IK teacher, and held-out evaluation. Exact checkpoint paths, losses, returns, normalization, source hashes and ONNX parity are in `artifacts/microduck-arm-v1/learning-smoke/`. Consult the final learning summary rather than historical attempts. A short training smoke is not convergence, a successful mobile policy, or proof that PPO improves the teacher.

The actor currently receives simulator-derived TCP/object/goal poses. Hardware state estimation is absent. Residual ONNX output requires the matching teacher, observation normalization and action mapping; it is not a drop-in Radxa or robotd policy.

### Recorded learning metrics

Seed 101 produced eight successful demonstration episodes and 643 state/action samples. BC ran 20 epochs; residual PPO ran 2,048 environment steps and 32 optimizer updates. The independent evaluation seeds were 201–205, with an explicit 250-step horizon for this short reach smoke.

| Controller | Held-out success | Mean return | Mean steps |
|---|---:|---:|---:|
| IK/FSM teacher | 5/5 | -0.033414 | 88.0 |
| BC | 0/5 | -10.173485 | 250.0 |
| Teacher + residual PPO | 5/5 | -0.020445 | 87.6 |

BC action MSE fell from 0.097145 to 0.00033247, **but every held-out BC episode timed out**. PPO's first/last logged training loss was 0.090142/0.048559; these totals need not decrease monotonically. Its 23 completed training episodes had mean return -0.102520. Raw JSONL also records value loss, policy-gradient loss, entropy, approximate KL, clipping and explained variance. Five small paired evaluations do not establish a statistically reliable PPO improvement over the teacher.

ONNX numerical parity passed on separate seeds 301–305: BC maximum absolute error 1.55e-7, raw PPO residual 2.24e-8. The residual is not a standalone action policy.

![Actual BC and PPO loss histories](../../artifacts/microduck-arm-v1/learning-smoke/training-curves.png)

Ten current IK/FSM MP4s cover all six free-base and four fixture cases, including failures. Open `artifacts/microduck-arm-v1/verified-evidence/videos.html`. Its index checks each MP4 hash, H.264/25fps encoding, terminal-inclusive sampled-frame count and the associated episode trace. The terminal frames of fixture reach and free reach were visually inspected: fixture success and free-base fall are visible. Video integrity is not task success.

The same gallery also includes two actual post-training checkpoint videos: held-out seed 201, BC timeout and teacher-plus-residual-PPO success. **Twelve current videos total**, with trace and checkpoint hashes. The checkpoint videos are under `artifacts/microduck-arm-v1/learned-videos/`; they do not reuse teacher footage as learned-policy evidence.

No-actuation and open-gripper controls are stored in `artifacts/microduck-arm-v1/negative-controls-final/`. Open gripper is not a meaningful negative for reach. When the teacher itself fails a manipulation case, control failures cannot establish a valid manipulation benchmark.

## Reproduction

Run from the workspace root using the existing environment; no dependency installation is required in this checkout. Tested runtime versions: MuJoCo 3.10.0, Stable Baselines3 2.9.0, PyTorch 2.9.1 and ONNX Runtime 1.29.0. PDF tools and ffmpeg/ffprobe are separate system tools. OpenSCAD CLI is not installed here, so no fabrication mesh export was verified.

```bash
export PYTHONPATH=.:rlx
PY=rlx/.venv-microduck/bin/python
$PY -m microduck_arm_v1 build
$PY -m microduck_arm_v1 check
$PY -m microduck_arm_v1 torque --out artifacts/my-v1/torque --samples 500
$PY -m microduck_arm_v1 workspace --out artifacts/my-v1/workspace.json --samples 10000
$PY -m microduck_arm_v1 evaluate --out artifacts/my-v1/eval --episodes 2 --max-steps 1500 --videos
$PY -m microduck_arm_v1.evidence controls --out artifacts/my-v1/controls
$PY -m microduck_arm_v1.evidence audit --evaluation artifacts/my-v1/eval/evaluation.json --out artifacts/my-v1/evidence
$PY -m microduck_arm_v1 demonstrate --case reach --mode fixture --episodes 8 --max-steps 250 --out artifacts/my-v1/demos
```

Use the dataset path printed by `demonstrate` with `train-bc --dataset ... --epochs 20 --out ...`. Use `train-ppo --case reach --mode fixture --steps 2048 --max-steps 250 --out ...` for the small residual smoke. Use the printed checkpoint with `export --checkpoint ... --out ...`. The API `learning.evaluate_policy` requires explicit held-out seeds. The final smoke script and result artifacts supply an exact runnable example.

`evaluate` intentionally exits **2** when free-base tasks fail. `hardware-check` also exits **2** without required measured evidence. `generate_engineering.py --manufacture-check` exits **3** while fabrication requirements are unresolved. Do not mask those statuses as passes. On macOS, offscreen MuJoCo may require permission to access graphical services.

## Remaining work before physical integration

1. Measure the actual duck, motors, mating interfaces, feet, battery and electronics; replace mass/inertia placeholders and resolve the motor-mesh discrepancy.
2. Revise the load-bearing mount and collision-free arm motion; current pick trajectories collide with the robot. Do not disable collisions to manufacture success.
3. Develop and validate a whole-body standing controller before locomotion; then implement a gait and repeat loaded free-base evaluations.
4. Complete actual mating CAD, tolerances, transmission and structural calculations; select and qualify electrical protection, energy storage and harnesses. The current logical diagrams cannot authorize fabrication.
5. Add sensor estimation, timing qualification, safe robotd/control-center integration and independent hardware acceptance. The viewer's earlier arm page does not establish support for this new 64/15 contract.
6. Retrain and compare teacher/BC/PPO across preregistered task and perturbation conditions. Only then advance through restrained bench, standing, contact and slow-carry hardware tests.

The deliverable stops at reproducible prototype artifacts and explicit failure evidence. **No full-skill pass, physical capability rating, fabrication release, or hardware safety certification is asserted.**
