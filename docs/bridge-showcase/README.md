# Suspended Bridge Showcase

`BridgeEnv` is a CPU MuJoCo prototype for testing whether the standard
Microduck stand/walk policy contract can cope with a narrow moving support.
It is integrated into the shared Studio recipe registry, CLI, policy catalog,
and `duck-viewer` as the seventh use case. See [measured results](RESULTS.md).

## Environment API

```python
from rlx.environments.bridge import BridgeEnv

env = BridgeEnv(
    actuator="xml",
    action_delay=False,
    command=(0.25, 0.0, 0.0),
    navigation=True,
    assistance=None,
    curriculum=False,
    obs_noise=False,
    domain_rand=False,
    pushes=False,
    random_yaw=False,
    seed=7,
)
obs, metrics = env.reset()
env.set_command((0.12, 0.0, 0.0))
obs, reward, terminated, truncated, metrics = env.step(action)
```

The Gymnasium observation and action shapes remain `61` and `14`. Existing
`alpha_stand.onnx` and `alpha_walking.onnx` actors accept the observation
without adaptation. The actor does not receive bridge pose, contact labels,
world position, or crossing state.

## Physical Setup

- A `1.10 m x 0.13 m` plank is a free body suspended by four tensioned MuJoCo
  tendons. Contacts can produce translation, swing, pitch, and roll.
- Fixed start and end platforms share the nominal plank top height.
- A visible gantry shows the four fixed suspension anchors. Its decorative
  posts have no collision response and cannot support the robot.
- The ground plane remains below the platforms and is classified separately
  from valid foot support.
- Evaluation starts on the launch platform. Curriculum resets use declared
  progressively longer starting distances; no control step teleports the
  robot, advances it, or writes forward velocity.
- Optional pushes apply a short lateral force only.

## Physics-Only Curriculum

`ASSISTANCE_LEVELS = (1.0, 0.6, 0.3, 0.1, 0.0)` scales restoring force,
leveling torque, and damping on the plank body. It never applies assistance to
the robot. `assistance=None` resolves to `0.0` normally and to `1.0` when
`curriculum=True`, so evaluation is unassisted by default while curriculum
training starts on the easiest rung. A successful crossing advances one level;
an episode ending before two seconds retreats one level. Early rungs start on
the plank, then move progressively back to the launch platform; the final rung
uses the exact unassisted evaluation spawn. These are physics/spawn changes
only. The inherited `MicroduckWalkEnv` locomotion reward is unchanged at every
level. Crossing is an evaluation metric, not a world-position reward that the
actor cannot observe.

With `navigation=True`, the environment converts the nominal forward command,
world lateral displacement, and bridge-axis heading error into the standard
body-frame `twist_cmd` observation slots every control step. This commander is
part of the protocol, not a reward or physical assist: it cannot move the robot
and the policy still produces every joint action. Set `navigation=False` to
pass the fixed command through unchanged. Under navigation, `set_command()`
primarily sets the nominal forward scalar; lateral and yaw slots are generated
by the bridge-axis correction. Once the trunk reaches
`END_PLATFORM_X - 0.05`, the commander sets desired world-forward velocity to
zero so a successful policy can hold the landing platform for the remainder of
the 20-second evaluation horizon.

## Acceptance And Metrics

`crossing_accepted` becomes true only after all of the following are observed
from MuJoCo state and contacts:

1. The trunk reaches the landing-platform region.
2. Both feet contact the end platform.
3. Neither foot contacts the ground.
4. No robot body contacts the ground.
5. Prior foot contact with the suspended plank exists, with no non-foot support.

Metrics also report support counts, bridge contacts, ground contacts, plank
position/speed/roll/pitch, robot tilt, lateral offset, command, assistance, and
push force.

`recipe_metrics()` is the numeric-only trainer/evaluator surface. Its stable
core fields are `bridge_crossed`, `bridge_progress_m`, `bridge_assistance`,
`robot_floor_contact`, `nonfoot_support`, `feet_support`, and
`bridge_contact_observed`.

`BridgeEvaluation(num_envs, required_steps)` consumes those metrics lane by
lane. A passing episode must run the full requested horizon without
termination, remain unassisted, start with valid foot support, cover the full
bridge distance, finish through the contact-validated two-foot crossing flag,
have prior suspended-plank foot-contact history, and never use the floor or a
non-foot body contact for support. Repositioning directly onto the end platform
therefore cannot earn crossing credit.

Skill evaluations require complete horizon multiples (for example 1,000 or
2,000 control steps for a 20-second episode), not a passing episode followed
by an arbitrarily cut-off fragment.

## Train And Inspect

```bash
./scripts/train-bridge.sh train --bridge-curriculum --total-timesteps 32768 \
  --num-envs 4 --num-steps 128 --num-minibatches 4 --backend dummy \
  --no-domain-rand --no-obs-noise --no-action-delay --no-random-yaw \
  --max-episode-s 20 --seed 7 --output-dir rlx/runs/studio/bridge/my-bridge
./scripts/train-bridge.sh eval --backend dummy --num-envs 3 --seed 101 \
  --policy rlx/runs/studio/bridge/my-bridge/bridge.onnx --eval-steps 1000 \
  --max-episode-s 20 --no-domain-rand --no-obs-noise --no-action-delay --no-random-yaw
```

Full training transfers the shipped walking actor and baked normalizer after
an action-parity check, then starts a fresh critic and optimizer with frozen
observation normalization. This is not an exact upstream training resume.
Rendering and evaluation never enable bridge assistance.

## Measured Limitations

- This module proves environment mechanics and ONNX interface compatibility;
  it does not claim that the shipped walking policy completes the bridge.
- Deterministic `alpha_walking.onnx` probes on September 10, 2026, with the
  final unassisted defaults reached `x=0.128`, `-0.231`, and `-0.003 m` for
  seeds 101, 202, and 303 before tilt termination at 9.24, 6.34, and 8.10
  seconds. None crossed. `alpha_stand.onnx` held the launch platform for the
  full 20-second seed-202 episode.
- The first curriculum rung did sample one contact-valid crossing from three
  deterministic `alpha_walking.onnx` probes: seed 101 crossed after starting
  at `x=0` with assistance `1.0`; seeds 202 and 303 did not. All three
  eventually fell, so this is exploration evidence, not a passing evaluation.
- The actor is blind to plank state. Adaptation must come from IMU and joint
  feedback already present in the 61-value deployment observation.
- The local BAM path is available for prototype stress tests, but final
  sim-to-real work still requires porting the design to the official
  `microduck_rl` GPU stack.
- The cable and plank parameters are plausible local-training values, not
  measurements from a physical bridge.
