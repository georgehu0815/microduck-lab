# v1-C measured simulation results

**NOT ALL EXPERIMENTS PASS.** Software regression checks and the fixture task matrix pass; free-base manipulation/walking and physical release do not. Preserve those failures rather than relaxing acceptance.

## Test evidence / 测试证据

`298 passed, 11 warnings in 59.25s`

Ruff and Python compilation pass. Standalone mypy remains blocked by resolving
the workspace's `microduck_local.contract` import; this is not claimed as a passed
whole-project type check. Runtime policy-contract regression tests pass.

| Mode | Case | Success / total | Failure |
|---|---|---|---|
| fixture | Obstacle move | 20/20 | PASS |
| fixture | Pick/place | 20/20 | PASS |
| fixture | Reach | 20/20 | PASS |
| fixture | Waypoint carry | 20/20 | PASS |
| free | Obstacle move | 0/20 | arm_environment, object_drop, phase_timeout |
| free | Pick/place | 0/20 | arm_environment, object_drop, phase_timeout |
| free | Reach | 8/20 | timeout |
| free | Waypoint carry | 0/20 | arm_environment, object_drop, phase_timeout |
| free | Walk (stowed) | 0/20 | timeout |
| free | Walk + carry | 0/20 | object_drop, robot_self_collision |

Candidates A/B, seeds 0–9, 10 g object. Fixture results are **not** free-base qualification.
The table top is 145 mm; v1-B low-table results are not directly comparable.

## BC → DAgger → residual PPO

8 teacher episodes (seeds 0–7); 4 DAgger episodes (seeds 20–23).
20 BC epochs, 20 merged-data BC epochs, 2,048 PPO timesteps (seed 40).
Held-out seeds: 101–105. Case: reach, fixture, candidate A.
Five arm outputs with an explicit fifteen-action adapter; not stock 14-action ONNX.

| Policy | Held-out success | Failures |
|---|---|---|
| bc | 0/5 | {"arm_environment": 5} |
| dagger | 4/5 | {"timeout": 1} |
| ppo | 5/5 | {} |
| teacher | 5/5 | {} |

Final metrics (raw, not smoothed):

```json
{
  "bc": {
    "epoch": 20,
    "raw_train_arm_mse": 0.0018554778071120381,
    "raw_validation_arm_mse": 0.005228061694651842
  },
  "dagger_merged_bc": {
    "epoch": 20,
    "raw_train_arm_mse": 0.0008629434159956872,
    "raw_validation_arm_mse": 0.0036217200104147196
  },
  "ppo": {
    "approx_kl": 9.514857083559036e-06,
    "clip_fraction": 0.0,
    "entropy_loss": -7.093945741653442,
    "phase": "training_end",
    "raw_loss": 0.05093790963292122,
    "raw_reward_sum": -1.2966566238901578,
    "timesteps": 2048,
    "value_loss": 0.104794941842556
  }
}
```

PPO retains the IK teacher with a maximum 0.02 rad arm residual.
Matching the teacher's 5/5 is not evidence that PPO is better than the teacher.
BC loss reduction does not imply closed-loop success. DAgger still has a failure.

![Raw learning curves / 原始训练曲线](../../artifacts/microduck-arm-v1c/training-curves.png)

## Workspace / 工作空间

10000 sampled target positions; 3505 IK-reachable;
3255 reachable and collision-free.
Balance-qualified workspace: **NOT ESTABLISHED**.
Full mass/CoM/inertia tables are analytical model estimates, not measured CAD release.
Joint torque CSV contains 5,000 screening rows; qualified continuous torque is unknown.
The 3× torque release gate remains closed.

## Payload screening / 负载筛查

Candidate A, fixture, seed 0 only: these are not payload ratings or reliability estimates.

| Payload g | Case | Task success | Failure |
|---|---|---|---|
| 5 | Reach | True | PASS |
| 5 | Pick/place | True | PASS |
| 5 | Obstacle move | True | PASS |
| 5 | Waypoint carry | True | PASS |
| 10 | Reach | True | PASS |
| 10 | Pick/place | True | PASS |
| 10 | Obstacle move | True | PASS |
| 10 | Waypoint carry | True | PASS |
| 20 | Reach | True | PASS |
| 20 | Pick/place | False | object_drop |
| 20 | Obstacle move | False | object_drop |
| 20 | Waypoint carry | False | object_drop |
| 30 | Reach | True | PASS |
| 30 | Pick/place | False | object_drop |
| 30 | Obstacle move | False | object_drop |
| 30 | Waypoint carry | False | object_drop |
| 50 | Reach | True | PASS |
| 50 | Pick/place | False | robot_self_collision |
| 50 | Obstacle move | False | robot_self_collision |
| 50 | Waypoint carry | False | robot_self_collision |

## Videos / 视频

20 full IK/FSM episode videos, plus 4 actual ONNX policy videos.
All are simulation recordings; successful and failed episodes are retained.
Each video has full ffmpeg decode verification, ffprobe metadata and SHA-256.
See `artifacts/microduck-arm-v1c/videos-final/video-manifest.json` and
`artifacts/microduck-arm-v1c/videos-learned/video-manifest.json`.

## Release / 放行

Modeled cartridge: 166 g;
target: 160 g — **FAIL**.
Supervisor tests check emulated inputs, not real shorts/reversed wiring or device safety.
Hardware release: **FALSE**. Fabrication, battery, continuous torque, thermal,
sensor/estimator, mobile control and physical fault qualification remain blocked.

Final evidence directories: `evaluation-final`, `engineering-final`, `videos-final`,
`videos-learned`, `learning/attempt-2` under `artifacts/microduck-arm-v1c/`.
Earlier root-level learning outputs, `engineering`, and `videos-attempt-1-interrupted`
are superseded diagnostics, not current release evidence.
