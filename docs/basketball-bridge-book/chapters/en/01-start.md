# Two balance laboratories: what this book actually reproduces

## A promise with a boundary

Imagine standing on a basketball. Moving your feet changes the ball's motion; the ball's motion changes the surface under your feet. Now imagine stepping onto a narrow plank suspended by four cables. Every footstep changes the support that the next footstep depends on. These are **coupled balance problems**, not simply walking animations placed on unusual scenery.

This book follows the working Microduck repository from physical assumptions to simulation, observations, actions, PPO updates, exported policies, evaluation, and the seven-case Duck Viewer. It assumes algebra, graphs, and elementary probability, not university mechanics or calculus. Each symbol is introduced before use. Code in the source appendix is actual implementation, not a fictional miniature presented as a complete trainer.

The evidence snapshot concerns local experiments recorded on **September 10, 2026**. The edition is **1.0, September 11, 2026**. A version number identifies a document snapshot; it does not mean that every skill is solved.

| Question | Evidence-backed answer |
|---|---|
| Can a duck balance on a free basketball? | Yes, the fixed-command local candidate survived all six recorded 60-second nominal trials. |
| Can it drive the ball reliably in the requested direction? | Not established: zero of three forward-command trials passed the rolling gate. |
| Does the suspended bridge exist as a working physical simulation? | Yes: a free plank, four physical cable tendons, platforms, and visible gantries. |
| Did the final bridge pilot complete an accepted crossing? | No. It has 32,768 new PPO transitions and zero accepted crossings. |
| Are both examples available in Duck Viewer? | Yes. Seven cases means five existing cases plus these two, not seven mastered skills. |
| Was either behavior verified on hardware? | No. All results here are simulation-only. |

The successful basketball behavior began from a **supplied trained recurrent actor**. The successful parts of this local experiment are source import, accurate actor transfer, further PPO training, export parity, sustained balance, and visible playback. This is not evidence that a randomly initialized local network learned ball balance in 102,400 steps. Likewise, a successful bridge training command means the pipeline ran; it does not mean the duck crossed.

“From scratch” therefore has two meanings. You can reconstruct the **software experiment** from source and required assets. You cannot reconstruct an undocumented upstream training history or replace a missing mature checkpoint with an assertion. This edition gives the reproducible local path in full and marks the original policy as an external prerequisite. The book's `source/` snapshot and ZIP are supporting source, not a redistribution of all third-party robot meshes, software dependencies, or policy weights.

## Choose a learning route

1. **First afternoon:** read the assumptions, inspect the scene diagrams, run preflight, run the pure-Python PPO arithmetic, and inspect real contact sheets. No training is necessary to learn the data flow.
2. **First experiment:** preserve the supplied artifacts, create a fresh run directory, execute a tiny smoke run, and inspect every output. A four-step smoke is an integration check only.
3. **Measured reproduction:** run the documented fixed-command basketball adaptation or the bridge pilot, export, evaluate deterministic policies with zero assistance, then inspect the complete rendered episode.
4. **Research extension:** change one physics or curriculum parameter, keep the evaluation fixed, run independent seeds, and document failures as carefully as successes.

Read the PPO chapter before tuning a learning rate. Read the evaluation chapter before deciding that a curve proves learning. Read the source appendix alongside the case chapters rather than starting with thousands of lines without context.

## A vocabulary we will keep consistent

| Term | Meaning in this book |
|---|---|
| State | All simulated physical variables, including information the policy cannot see. |
| Observation | The 61 numbers provided to the actor at one control step. |
| Action | Fourteen policy outputs interpreted by the robot control contract. |
| Policy / actor | A function producing an action distribution during training, and its mean during deterministic evaluation. |
| Critic / value | A learned estimate of future discounted reward, used for training. |
| Episode | One attempt between reset and termination/time limit. |
| Transition | One observation–action–reward–next-observation interaction in one environment. |
| Rollout | A block of transitions collected before updating the network. |
| Checkpoint | Training weights plus whatever metadata that checkpoint format actually stores. |
| ONNX export | Portable inference graph with the observation normalizer baked in. |
| Curriculum | A sequence of easier-to-harder physics, spawning, or judging conditions; not permission to change rewards silently. |
| Parity | Numerical agreement between two implementations on the same inputs and recurrent-state lifecycle. |
| Skill gate | A measurable acceptance condition, distinct from process exit or file creation. |

## Keep an experiment notebook

Record the source hashes, command, Python executable, package versions, actuator mode, seed, spawn distribution, assistance level, randomization settings, command distribution, step count, and evaluation horizon. Save the full JSON result, not only a screenshot of its green region.

An example notebook entry is:

```json
{
  "experiment": "my-independent-basketball-run",
  "initialization": "supplied actor; fresh critic and optimizer",
  "control_dt_s": 0.02,
  "actor_observations": 61,
  "actions": 14,
  "claim_to_test": "60-second free-ball balance",
  "claim_not_yet_supported": "commanded rolling and steering",
  "hardware_tested": false
}
```

This is a template, not an additional measured run. The actual run settings live beside the checkpoints. Filenames such as `final`, `full`, or `success` are not evidence by themselves.

## Where everything lives

All shell commands in this book start at the **workspace root**, unless a block explicitly changes directory. A path beginning `rlx/` is relative to that root. Book assets and source snapshots are relative to `docs/basketball-bridge-book/`.

```text
microduck-lab/
  microduck_local/       CPU MuJoCo harness, robot contract, lab server
  microduck_rl/          upstream robot/training reference checkout
  microduck/             upstream runtime and supplied walking policies
  microduck-playground/  supplied basketball experiment/assets/checkpoint
  rlx/                  local PPO, task environments, ONNX exports
  duck-viewer/           browser Studio and real scene playback
  scripts/              scenario command wrappers and API verification
  docs/basketball-bridge-book/
    chapters/en/        English chapter sources
    chapters/zh/        Chinese chapter sources
    assets/             measured plots, screenshots, schematic diagrams
    examples/           executable arithmetic and environment probes
    tools/              build, preflight, validation
    source/             exact selected implementation snapshot
```

The full workspace is the execution unit. Copying one environment file to an empty directory will not provide its robot contract, meshes, actuator code, or training framework.

# Mathematical and physical foundations

## Units before equations

Use metres for length, seconds for time, kilograms for mass, radians for angles, newtons for force, and newton-metres for torque. A value without a unit can look reasonable while being wrong by a factor of a thousand.

At a 50 Hz control rate the actor is consulted 50 times each second:

$$\Delta t = 1/50 = 0.02\ \mathrm{s}.$$

A 60-second basketball test contains 3,000 control steps. A 20-second bridge episode contains 1,000. These are control steps; the physics engine can perform multiple smaller integration steps inside each one. More environments increase data per rollout, not the duration of one simulated second.

Velocity is change of position divided by elapsed time. If the duck moves 0.03 m in 0.2 s, its average speed is 0.15 m/s. Acceleration is change of velocity divided by elapsed time. These finite differences are enough to understand the simulator's job before studying continuous derivatives.

Forces change momentum. In one dimension, a simple teaching update is

$$v_{t+1}=v_t+(F/m)\Delta t,\qquad x_{t+1}=x_t+v_{t+1}\Delta t.$$

This is an explanatory approximation, **not** a replacement for MuJoCo's integration, articulated-body dynamics, and contact solver. The actual simulator also resolves joints, contacts, friction and constraints. A torque is a turning effect: force times its perpendicular lever arm. A foot force displaced from the plank's centre can rotate it even when the duck remains upright.

## Why balance and motion interact

On a floor, the support under a planted foot is nearly fixed. On the ball, a foot tangential force can rotate the ball and move the support. If an ideal rigid sphere rolls without slip, distance and angle satisfy

$$d=r\theta.$$

With radius 0.12 m, a 90-degree rotation is $\theta=\pi/2$ radians, giving $d\approx0.1885$ m. At 0.15 m/s the ideal angular speed is $v/r=1.25$ rad/s. These are scale checks, not guarantees: slipping, contact deformation and turning break the simple straight-line relation.

For the suspended bridge, gravity pulls the combined masses down, while cables support and constrain the plank. A cable is not a rigid beam: its tension acts along its length. The implementation's tendon and limit settings approximate the desired cable behavior; their exact meaning is determined by the model, not by how the rendered cable looks. A gantry can be visible while having no collision role. The visual scene is evidence of appearance, not a complete description of the dynamics.

A material name such as “wood” or “rubber” does not automatically create correct physics. The simulator needs mass, dimensions, inertia, friction/contact parameters, joint and tendon definitions. The wooden appearance is a visual material. The 0.45 kg plank mass is a dynamic parameter. They must be documented separately.

## Coordinate frames: forward for whom?

The **world frame** belongs to the scene. The **body frame** turns with the duck. If the duck rotates 90 degrees, body-forward no longer points along world-x. Rewarding world-x velocity while issuing body-forward commands creates a misleading task.

For a planar heading angle $\psi$, a body-frame velocity can be expressed in world coordinates as

$$v_x^w=\cos\psi\,v_x^b-\sin\psi\,v_y^b,$$
$$v_y^w=\sin\psi\,v_x^b+\cos\psi\,v_y^b.$$

Superscripts $w$ and $b$ mean world and body, not exponentiation. With body velocity $(0.15,0)$ and heading 90 degrees, world velocity is approximately $(0,0.15)$. The basketball evaluator uses an initial-heading reference for directional progress; the bridge commander translates navigation information into ordinary body-frame commands. Neither fact makes world position an actor input.

## Feedback, memory, and partial observation

An **open-loop** script repeats preset joint targets regardless of what happens. A **feedback policy** chooses the next targets using current sensed information. If a foot slips, the next observation changes and the policy can respond.

The robot does not receive the entire simulator state. It gets angular velocity, projected gravity, commands, joint information, previous actions and reserved command slots. It does not directly receive ball centre or plank pose. This is partial observation. Two identical current observations may have different motion histories. The basketball actor therefore carries LSTM memory; the bridge actor is a transferred feedforward walking actor. A memory network is not clairvoyance: it can only summarize evidence that entered its observations.

During learning, random sampled actions help explore nearby behavior. During deterministic evaluation, the exported actor's mean is used. A stochastic policy that occasionally catches itself may export to a mean policy that falls. That is why training reward and exported-policy testing are different experiments.

## Arithmetic workshop

```python
import math

control_dt = 0.02
radius = 0.12
command_speed = 0.15
steps = round(60 / control_dt)
quarter_turn_distance = radius * math.pi / 2
ideal_angular_speed = command_speed / radius
print(steps, round(quarter_turn_distance, 4), ideal_angular_speed)
assert steps == 3000
assert abs(ideal_angular_speed - 1.25) < 1e-12
```

Expected output: `3000 0.1885 1.25`. This is a unit test of arithmetic, not a balance test.

**Exercise.** Four environments collect 128 steps before each update. How many transitions are collected? How much simulated time elapses in each lane? How many such updates make 32,768 transitions?

**Answer.** $4\times128=512$ transitions; each lane advances $128\times0.02=2.56$ seconds; $32768/512=64$ updates. Summing lane durations gives aggregate simulated experience, not one continuous 655.36-second bridge crossing.

**Exercise.** Why can a high average reward fail to prove a 60-second hold?

**Answer.** Short successful segments separated by falls/resets can have high average reward. A sustained hold requires one unbroken episode, first-fall accounting, valid foot support, and the entire requested duration.
