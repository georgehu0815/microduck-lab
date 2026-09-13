# Co-carry evaluator audit

Date: 2026-09-12

## Scope and decision

This audit is limited to contact measurement, drop classification, internal-force
measurement, and their acceptance gates for `arms-co-carry-v1`.

The repair-v3 decision is:

- Preserve the original conservative evaluator and all acceptance thresholds.
- Do not relax `no_drop`, the 30 mm/s safe-placement speed, the 20 ms airborne
  grip-loss limit, or the strict `< 1.0 N` internal-force limit.
- Change only the authored co-carry TCP trajectory: use synchronized quintic
  Cartesian interpolation during the existing 3 s lift, 6 s transport, and 3 s
  descent phases.
- Do not claim success from the prior 18 aggregate statistical gates. Final
  acceptance requires actual `1800/1800`, an independent `1800/1800` holdout,
  and all 21 designated replays.

No product or evaluator source changes were made as part of this audit.

## Source identity

The prior `recheck-20260912-v2` evidence is bound to:

- Environment SHA-256:
  `617aa9270529e5226d353cd6df38eefbb5e352bee2a02ea6cf2252f89dbe28f1`
- Pipeline SHA-256:
  `22e4759f93223d32ba948169f7895c3eff0e07949246ca54cc262ddd2b87ff01`

At audit time, the repaired environment SHA-256 was:

- `ec708791af2180656231d91c91defaa77c6cd39d39e7db714faf199e6adb5c92`

A direct comparison with the archived frozen environment found one production
hunk only: the six-line co-carry quintic interpolation in
`rlx/rlx/environments/arm.py:375-380`. The phase durations remain unchanged at
`rlx/rlx/environments/arm.py:369-370`. Reward calculation, contact measurement,
drop logic, task gates, masses, and horizons were unchanged.

## Direct evidence

### Prior aggregate result

`rlx/runs/arm/recheck-20260912-v2/summary.json` records:

- 1,800 evaluated episodes.
- 1,798 successes.
- All 18 checkpoints met their statistical thresholds.
- `all_episodes_pass` was false.
- Both new failures belonged to co-carry training seed 202.

This is not an acceptable final result under the repair-v3 all-episode
requirement.

### Drop evaluator behavior

Contact state is refreshed at every 2 ms physics substep in
`rlx/rlx/environments/arm.py:535-537`. However,
`previous_contacts` is captured once at the beginning of each 20 ms control
step at `rlx/rlx/environments/arm.py:527`.

Before release, table support is classified as safe only when all of the
following are true:

- The object is within 10 mm of the goal.
- Absolute vertical speed is below 30 mm/s.
- A complete grasp exists now, or pad contact existed at the start of the
  current 20 ms control step.

The relevant logic is at `rlx/rlx/environments/arm.py:540-547`.

The drop counter is a conservative safety-event latch. A `drop_count` of one
does not by itself prove that the object physically fell out of both grippers
and hit the table.

### Seven known flagged episodes

Read-only deterministic replays against the frozen checkpoints reproduced all
seven prior verdicts exactly.

| Training seed | Evaluation seed | Measured trigger | Audit classification |
|---:|---:|---|---|
| 101 | 50705 | Supported near goal, 2.888 mm/s downward, no current or control-start pad contact | Conservative drop flag; not evidence of a physical drop |
| 202 | 50007 | First table support at 61.653 mm/s downward; internal force peaked at 1.221160 N | Genuine unsafe touchdown and genuine force violation |
| 202 | 50606 | Supported near goal, 2.917 mm/s downward, no current or control-start pad contact | Conservative drop flag; not evidence of a physical drop |
| 303 | 50704 | Supported near goal, 2.955 mm/s downward, incomplete current contact and no control-start contact | Conservative drop flag; not evidence of a physical drop |
| 303 | 50807 | Supported near goal, 2.817 mm/s downward, incomplete current contact and no control-start contact | Conservative drop flag; not evidence of a physical drop |
| 202 | 80307 | First table support at 81.827 mm/s downward; internal force peaked at 1.199345 N | Genuine unsafe touchdown and genuine force violation |
| 202 | 80609 | Supported 0.212 mm from goal at 2.906 mm/s downward, with no current or control-start pad contact | Conservative drop flag; not evidence of a physical drop |

Seeds 50007 and 80307 exceeded the unchanged 30 mm/s safe-placement threshold.
They were unsafe placements even though their final states later satisfied the
release and settling gates.

For seed 80307, the internal-force value was not an isolated solver spike. The
instantaneous value remained at or above 1.0 N for 87 physics samples
(`0.174 s`) between approximately 16.062 s and 16.272 s. At the maximum, the
arms applied opposing horizontal forces while both remained in contact with
the table-supported tray. The existing metric and strict gate therefore
correctly rejected this behavior.

## Evidence versus inference

### Evidence

- The two high-speed touchdown values and the five low-speed supported-state
  values above were measured at the physics substep that latched each drop
  flag.
- The old failure reports are deterministic: their historical replays retained
  the same verdicts and gates.
- The internal-force evaluator sums object-pad contact forces per arm and uses
  half the horizontal force difference at
  `rlx/rlx/environments/arm.py:455-496`.
- The gate remains strictly `peak_internal_force < 1.0` at
  `rlx/rlx/environments/arm.py:638-644`.
- The repaired target path is position-continuous at 5, 8, 14, and 17 seconds.
  A 20 ms boundary probe measured adjacent position increments below
  `0.1 micrometre`.

### Inference

- The phase target jumps were the root cause of the two genuinely unsafe
  touchdown/force episodes. This is supported by their timing, the unchanged
  evaluator, and the development result in which the same seven checkpoint and
  seed pairs pass after only the smooth trajectory change, with peak forces
  reduced to approximately 0.535-0.649 N.
- The five low-speed flags arise from the evaluator's intentionally
  conservative contact-history filter: a safely resting object can be flagged
  after pad separation because the filter does not retain a safe-touchdown
  state across control frames.
- The smooth trajectory avoids that latent filter path in the known episodes,
  but does not remove the filter itself.

The development seven-replay result is supporting evidence, not final
acceptance evidence. The full training, holdout, and replay gates remain
required.

## Preserved conservative filter

The contact-history behavior remains unchanged deliberately for repair-v3.
This avoids weakening acceptance while the trajectory defect is corrected.

Consequences:

- A future `no_drop` failure must be inspected at physics-substep resolution
  before it is described as a physical drop.
- High-speed or off-goal touchdown remains a valid failure.
- More than 20 ms of unsupported airborne grip loss remains a valid failure.
- A low-speed, on-goal, table-supported flag after pad separation may be the
  known conservative classification rather than a physical drop.

## Regression coverage

The following tests now protect the repair without changing acceptance:

- `rlx/tests/test_arm_trajectory.py:9-23` checks positional continuity at the
  5, 8, 14, and 17 second phase boundaries and preserves the intended release
  width transition.
- `rlx/tests/test_arm_trajectory.py:26-40` checks synchronized arm offsets,
  bounded Cartesian target speed, and the final goal target.
- `rlx/tests/test_arm_evaluation.py:187-197` checks that 0.999 N passes,
  1.0 N fails, `drop_count == 0` passes, and `drop_count == 1` fails.
- `rlx/tests/test_arm_evaluation.py:65-109` retains the unsupported-airborne
  and unsafe-table-support negative tests.

Recommended acceptance sequence:

1. Run the focused trajectory, evaluator, and contract tests.
2. Replay all seven known checkpoint/seed pairs under the repaired source.
3. Require the complete new training evaluation to report `1800/1800`.
4. Require an independently seeded holdout to report `1800/1800`.
5. Require all 21 designated regression replays to pass.
6. Bind every report and media artifact to the repaired environment and
   pipeline hashes; do not combine old-hash evidence with the new verdict.

## Audit conclusion

The quintic trajectory repair is narrow and preserves the original safety
contract. The old evidence contains two genuine unsafe touchdown/force
episodes and five conservative drop flags; it would be inaccurate to describe
all seven as physical drops. The conservative filter remains a known evaluator
limitation, not an acceptance relaxation. Final success must wait for the
specified `1800/1800 + 1800/1800 holdout + 21/21 replay` evidence.
