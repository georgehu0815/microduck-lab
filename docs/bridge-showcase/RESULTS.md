# Bridge And Basketball Studio Integration — September 10, 2026

**Seven cases are available; seven mastered skills are not claimed.**

## Suspended Bridge

- Scene: 1.10 m × 0.13 m plank, four physical suspension tendons, two fixed
  platforms, visible gantries, separate floor/contact-failure detection.
- Final pilot: `rlx/runs/studio/bridge/bridge-studio-02/`, 32,768 new PPO
  transitions, four environments, seed 7, XML servos, frozen walking-actor
  normalizer, fresh critic and optimizer.
- Training used the physics/spawn-only assistance curriculum. Evaluation
  used assistance zero, no noise/randomization/delay, three lanes seeded
  101–103, 1,000 control steps each, and the declared navigation commander.
- **No successful crossing.** All 11 observed attempts, including trailing
  partial attempts after falls, failed the skill gate. The best measured
  forward progress from the launch position was **1.175 m**; the crossing
  target is at least 1.60 m plus validated two-foot destination contact.
- [Actual 20-second diagnostic video](evidence/rollout.mp4) and
  [contact sheet](evidence/contact-sheet.png) show two explicit resets and a
  later attempt walking partway onto the plank, not a continuous crossing.
- [Evaluation JSON](evidence/evaluation.json) retains the failed verdict.

The earlier `bridge-studio-01` is a separate 32,768-step pilot from the same
walking bootstrap. It preceded destination-stop navigation and is not the
final demonstration. These are independent warm starts, not one cumulative
65,536-step policy. Decorative non-colliding gantries and lighting do not
provide physical support. Curriculum completion is not established by the
reward history or by the number of PPO transitions.

## Basketball

`basketball-balance-01` imports the previously trained 102,400-step recurrent
candidate without modifying its source artifacts. ONNX metadata insertion
preserves action parity (maximum error 1.43e-6). The new Studio evaluation
uses seeds 101–103 for 60 seconds each under zero and 0.15 m/s forward commands:
**6/6 sustained-balance trials, 0/3 commanded-rolling passes**.

Only zero/forward commands are measured by this adapter. Lateral and yaw
steering coverage is explicitly absent, so the adapter and UI fail closed
for full-skill success even if a future forward-only trial passes. The
preview is labeled **BALANCE ONLY · STEERING NOT PASSED**. Live playback uses
a 60-second episode; the original training horizon was 10 seconds.

Basketball uses an independent LSTM state per duck, retained across commands
and reset on episode, policy, and environment changes. Local checkpoint
continuation is an actor warm-start with a fresh critic/optimizer, not an
exact optimizer resume. The supplied playground bootstrap and ball visual
assets are required; Studio reports missing references before launching.

## Verification Evidence

- [Real Studio eval/render API receipts](evidence/api/verification.json): both
  runs have current-policy-bound render evidence; both full-task verdicts fail.
- [Browser and live-scene verification](evidence/studio/verification.json):
  seven cards, explicit readiness, custom MuJoCo scenes, desktop/mobile layout.
- Targeted tests cover curriculum invariants, no-contact/teleport exploits,
  strict evaluation, recurrent lifecycle and cache isolation, safe local
  continuation, CLI scene selection, and Studio command/artifact routing.
- Final targeted Python suite: **306 passed, 1 skipped**. Viewer suite:
  **161 passed**; ESLint, TypeScript/build, targeted Python correctness lint,
  and four-module mypy checks pass.
- Both new cases also completed real four-step training through the Studio
  API. These smoke runs are pipeline checks, not behavioral evidence.
- Final live screenshots verify that the ball mesh, plank, platforms, four
  cables, and gantries are actually rendered, not just present in simulation.
- [Main-service readiness](evidence/service-readiness.json) checks all seven
  routes without claiming seven accepted policies. Use
  `bash restart-lab.sh --readiness-only` while pilot skill gates remain failed.

These are local simulation prototypes, not hardware-ready policies.
