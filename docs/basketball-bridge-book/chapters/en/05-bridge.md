# Suspended Bridge: Learning to Walk on a Moving Support

## Learning objectives

After this chapter, you should be able to:

1. describe the simulated bridge geometry, four tendon cables, free plank, platforms, and non-colliding gantry with exact dimensions;
2. distinguish a simulated bill of materials from a certified physical construction plan;
3. trace the unchanged 61-observation and 14-action deployment contract;
4. explain why the navigation commander may use world state while the learned actor receives only the standard command slots;
5. derive every inherited locomotion reward term and its actual weight;
6. explain why the curriculum changes physics and spawn position, but never reward weights;
7. describe the walking-policy transfer: transferred actor and normalizer, frozen normalization statistics, fresh critic, and fresh optimizer;
8. reproduce the current build, train, export, evaluation, and render commands; and
9. apply the strict full-20-second crossing gate without turning partial progress into a successful result.

The most important result comes first:

> The recorded `bridge-studio-02` pilot trained for 32,768 PPO transitions and did **not** cross the suspended bridge. Its best measured progress was 1.175 m from the launch position, below the 1.60 m progress threshold and without a contact-validated two-foot landing. This chapter documents a working simulation and training protocol, not a mastered skill.

## 1. Evidence and claim boundary

This chapter is grounded in the current repository implementation:

| Question | Source |
| --- | --- |
| Bridge geometry, commander, curriculum, contacts, and metrics | `rlx/rlx/environments/bridge.py` |
| Strict crossing gate | `rlx/rlx/environments/bridge_evaluation.py` |
| Walking actor transfer | `rlx/rlx/models/bridge_bootstrap.py` |
| Studio train/eval/render/export parser | `rlx/examples/ppo_microduck_studio.py` |
| Shared observation and action order | `microduck_local/src/microduck_local/contract.py` |
| Inherited reward equations | `microduck_local/src/microduck_local/walk_env.py` |
| Recorded pilot and deterministic evaluation | `docs/bridge-showcase/` and `rlx/runs/studio/bridge/bridge-studio-02/` |

The local training playbook adds three interpretation rules:

- never change the 61/14 interface for one task;
- do not treat reward or PPO completion as proof of behavior; and
- inspect a deterministic exported rollout before claiming a skill.

The bridge model is a **local MuJoCo prototype**. Its dimensions, masses,
friction coefficients, tendon stiffness, damping, and limits are simulator
parameters. They are not measurements from a constructed bridge, cable ratings,
load calculations, a safety factor, or a hardware certification. The render
material names such as `bridge_plank_mat` describe color and appearance, not a
specified species of wood or grade of metal.

## 2. What is physically simulated?

The task places Microduck on a raised launch platform. A narrow plank spans the
gap to a second platform. The plank is not welded, hinged, or kinematically
animated. It has a MuJoCo free joint, so contact forces can translate it and
rotate it in roll, pitch, and yaw. Four spatial tendons connect moving
attachment sites on the plank to fixed world anchors.

### 2.1 Simulated bill of materials

This table is a simulator inventory, not a shopping or construction list.

| Count | Simulated item | Exact model values |
| ---: | --- | --- |
| 1 | Free plank | `1.10 x 0.13 x 0.03 m`, mass `0.45 kg`, center initially at `(0, 0, 0.225) m` |
| 2 | Fixed platforms | each `0.60 x 0.48 x 0.24 m`, centers at `x=-0.85 m` and `x=+0.85 m` |
| 4 | Spatial tendon cables | stiffness `320`, damping `2.0`, render width `0.0015 m`, length-limited |
| 4 | Fixed cable anchors | `(x,y,z)=(+/-0.45,+/-0.17,0.77) m` |
| 4 | Plank attachment sites | local `(x,y,z)=(+/-0.45,+/-0.057,0) m` |
| 2 | Gantry crossbars | `0.028 x 0.48 x 0.028 m`, centered at `x=+/-0.45 m`, `z=0.784 m` |
| 4 | Gantry posts | `0.028 x 0.028 x 0.77 m`, at `x=+/-0.45 m`, `y=+/-0.225 m` |
| 1 | Ground plane | top at world `z=0`, below the platform tops |

MuJoCo box `size` values are half-extents. For example, the plank source uses
half-extents `(0.55, 0.065, 0.015)`, which gives the full dimensions in the
table.

The start platform spans `x=-1.15` to `-0.55 m`. The plank spans `x=-0.55` to
`+0.55 m`. The end platform spans `x=+0.55` to `+1.15 m`. Their boundaries
touch exactly in the nominal model. All three top surfaces are at `z=0.24 m`;
the plank center is therefore `0.24-0.015=0.225 m`.

### 2.2 The four cables

At each of the two longitudinal positions, `x=-0.45 m` and `x=+0.45 m`, one
left and one right cable connect the plank to the gantry. The lateral
anchor-to-attachment offset is

$$
\!0.17-0.057=0.113\ \text{m}
$$

on the left and the same magnitude on the right. The vertical offset is

$$
0.77-0.225=0.545\ \text{m}.
$$

The nominal straight-line length is therefore

$$
L=\sqrt{0.113^2+0.545^2}
  =0.556591412079\ \text{m}.
$$

The source sets the upper spring length to `L-0.0015`, or about
`0.555091 m`, and the maximum limited length to `L+0.008`, or about
`0.564591 m`. The tendons are massless MuJoCo constraints with elastic and
damping parameters; the code does not model cable strands, knots, connectors,
fatigue, breaking strength, or anchor pull-out.

A simplified linear-spring estimate gives an initial extension of `0.0015 m`
and approximately

$$
F=k\Delta L=320(0.0015)=0.48
$$

in the simulator's force units before damping and constraint effects. This is
an instructional estimate, not a physical load rating.

### 2.3 The gantry is visible but cannot catch the robot

The crossbars and posts use `contype=0` and `conaffinity=0`. MuJoCo renders
them, but collision detection does not let the robot stand on them, lean on
them, or receive a rescue contact. The four tendon constraints are physical
simulation elements; the rectangular gantry geometry is decorative,
non-colliding context.

The essential model construction is shown in this abridged source excerpt. It
omits the surrounding loop and is not a standalone program:

```python
plank = spec.worldbody.add_body(
    name="bridge_plank",
    pos=(0, 0, 0.225),
)
plank.add_freejoint(name="bridge_freejoint")
plank.add_geom(
    name="bridge_surface",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.55, 0.065, 0.015),
    mass=0.45,
    friction=(1.15, 0.006, 0.0002),
    condim=4,
)

tendon = spec.add_tendon(
    stiffness=320.0,
    damping=2.0,
    springlength=(0.0, cable_length - 0.0015),
    limited=True,
    range=(0.0, cable_length + 0.008),
    width=0.0015,
)
tendon.wrap_site(f"{name}_anchor")
tendon.wrap_site(f"{name}_attach")
```

The floor friction is `(1.0, 0.005, 0.0001)`, platform friction is
`(1.1, 0.005, 0.0001)`, and plank friction is
`(1.15, 0.006, 0.0002)`. These coefficients are chosen local-training values.

## 3. Contact, motion, and failure

The simulation runs with a `0.005 s` physics step and four physics substeps per
policy action:

$$
\Delta t_{\text{control}}=4(0.005)=0.02\ \text{s},
$$

so the actor runs at 50 Hz. A default 20-second episode contains 1,000 control
steps.

The action has 14 values in the shared joint order. It is a joint-target
offset:

$$
\mathbf q^{\text{target}}
=\mathbf q^{\text{default}}+\operatorname{clip}(\mathbf a,-4,4).
$$

The plank moves only through simulated dynamics: gravity, tendon forces,
contacts, and optional curriculum forces. No control step teleports the plank
or robot forward. The reset function does place the robot at a declared
curriculum spawn and gives the plank a small random initial roll, pitch, and
angular velocity.

The bridge episode terminates when any of these conditions occurs:

- any robot body contacts the ground plane;
- trunk tilt exceeds 70 degrees;
- trunk height falls below `0.10 m`;
- the trunk moves beyond `|y| > 0.36 m` while neither foot has valid support;
- physics becomes non-finite.

Reaching the ordinary time limit is a truncation, not a fall. Valid foot support
may come from the start platform, plank, or end platform. A hand, head, trunk,
or other non-foot body touching one of those support surfaces is tracked as
`nonfoot_support`.

## 4. The observation contract: privileged commander, ordinary actor

### 4.1 The actor still receives exactly 61 values

The bridge actor has no special bridge-state input:

| Slice | Width | Meaning |
| --- | ---: | --- |
| `0:3` | 3 | trunk angular velocity |
| `3:6` | 3 | gravity projected into the trunk frame |
| `6:20` | 14 | joint position relative to `DEFAULT_POSE` |
| `20:34` | 14 | joint velocity delayed by one control step |
| `34:48` | 14 | previous raw actor action |
| `48:51` | 3 | body-frame forward, lateral, and yaw-rate command |
| `51:55` | 4 | head command, zero for Bridge |
| `55:61` | 6 | body command, zero for Bridge |

Bridge position, bridge velocity, cable length, plank roll and pitch, world
position, contact labels, crossing state, and assistance level are absent. A
test moves the plank laterally by `0.03 m` without changing the robot and
confirms that the 61-value observation remains byte-for-byte unchanged.

The actor and critic are separate `61 -> 512 -> 256 -> 128` ELU networks, but
both receive the same normalized 61-vector. This implementation does **not**
provide an asymmetric privileged critic. Privileged simulator quantities are
used by reward calculation, curriculum decisions, logs, and evaluation.

### 4.2 Why the commander is privileged

The navigation commander runs outside the learned actor. It reads the trunk's
world `x`, world lateral displacement `y`, and world heading. It converts those
quantities into the normal three command slots. The actor sees the resulting
command, not the world measurements that produced it.

Let the horizontal forward heading be

$$
\mathbf h=(h_x,h_y,0),
$$

and the body-left direction be

$$
\mathbf s=(-h_y,h_x,0).
$$

Before the destination stop, the desired world velocity is

$$
\mathbf d=
\left(
0.25,\
\operatorname{clip}(-1.2y,-0.12,0.12),\
0
\right).
$$

The heading error is

$$
\psi=\operatorname{atan2}(h_y,h_x).
$$

The published body-frame command is

$$
\mathbf c=
\left(
\mathbf d\cdot\mathbf h,\
\mathbf d\cdot\mathbf s,\
\operatorname{clip}(-1.5\psi,-0.6,0.6)
\right).
$$

The implementation is:

```python
reached_destination = (
    trunk_world_x >= END_PLATFORM_X - 0.05
)  # 0.85 - 0.05 = 0.80 m

desired_world = np.array((
    0.0 if reached_destination else fixed_command[0],
    np.clip(-1.2 * trunk_world_y, -0.12, 0.12),
    0.0,
))

twist_cmd[:] = (
    desired_world @ heading,
    desired_world @ side,
    np.clip(-1.5 * yaw_error, -0.6, 0.6),
)
```

This commander is not a physical assist: it applies no force and writes no
joint action. However, it is privileged protocol logic because it uses
world-frame localization unavailable inside the 61-vector. A complete system
claim must therefore include the commander, not attribute all navigation to the
actor.

When trunk `x >= 0.80 m`, the current commander sets desired world-forward
speed to zero while retaining lateral and yaw corrections. This destination
stop distinguishes the final `bridge-studio-02` protocol from
`bridge-studio-01`, which preceded that stop behavior. The two runs are
independent 32,768-transition warm starts; they are not one 65,536-transition
training chain.

## 5. Reward: inherited walking, not a crossing jackpot

`BridgeEnv` does not override `_compute_reward`. The bridge recipe also declares
an empty set of bridge reward override keys. Therefore all curriculum stages
use the exact base walking reward:

$$
r_t=
r_{\text{lin}}+
r_{\text{ang}}+
r_{\text{upright}}+
r_{\text{pose}}+
r_{\text{head}}+
r_{\text{air}}+
r_{\Delta a}+
r_{\omega xy}.
$$

There is no reward for world `x`, bridge progress, plank angle, cable tension,
end-platform contact, or `bridge_crossed`.

Let:

- $\mathbf c=(c_x,c_y,c_\omega)$ be `twist_cmd`;
- $\mathbf v_b=(v_x,v_y,v_z)$ be true trunk linear velocity in the body frame;
- $\boldsymbol\omega=(\omega_x,\omega_y,\omega_z)$ be true trunk angular velocity;
- $\mathbf g_b=(g_x,g_y,g_z)$ be projected gravity;
- $\delta q_j=q_j-q_j^{\text{default}}$;
- $h_j$ be the head command, zero in Bridge.

The exact implemented terms and weights are:

### Linear velocity tracking

$$
e_v=(c_x-v_x)^2+(c_y-v_y)^2+v_z^2,
$$

$$
r_{\text{lin}}=2\exp(-e_v/0.1).
$$

### Angular velocity tracking

$$
e_\omega=(c_\omega-\omega_z)^2+\omega_x^2+\omega_y^2,
$$

$$
r_{\text{ang}}=2\exp(-e_\omega/0.5).
$$

### Uprightness

$$
r_{\text{upright}}
=2\exp[-(g_x^2+g_y^2)/0.05].
$$

### Default leg pose

For the ten leg joints,

$$
r_{\text{pose}}
=\exp\left[
-\frac{\sum_{j\in\text{legs}}\delta q_j^2}{0.5}
\right].
$$

### Head pose

For the four head joints,

$$
r_{\text{head}}
=2\left[
\frac14\sum_{j\in\text{head}}
\exp\left(-\left(\frac{\delta q_j-h_j}{0.5}\right)^2\right)
\right].
$$

### Dense foot air time

Each unsupported foot accumulates air time in `0.02 s` increments. If the
motion command magnitude exceeds `0.01`, a foot contributes one while

$$
0.125<\tau_f<0.300\ \text{s}.
$$

Thus

$$
r_{\text{air}}
=3\sum_{f\in\{L,R\}}
\mathbb 1(0.125<\tau_f<0.300).
$$

Bridge overrides foot-contact detection so the start platform, plank, and end
platform all count as valid support for this inherited term.

### Action-rate penalty

$$
r_{\Delta a}
=-w_{\Delta a}
\sum_{j=1}^{14}(a_{t,j}-a_{t-1,j})^2.
$$

The per-environment lifetime schedule is:

```text
environment steps:  0     12,000  18,000  24,000  30,000  36,000
weight:             0.1   0.2     0.4     0.6     0.8     1.0
```

The recorded pilot used four environments and 32,768 total transitions, or
8,192 control steps per environment. It therefore remained on weight `0.1`
throughout that pilot.

### Roll/pitch angular-velocity penalty

$$
r_{\omega xy}
=-0.05(\omega_x^2+\omega_y^2).
$$

Both penalty terms are non-positive by construction.

### Worked numerical example

Suppose one step has:

```text
command             = (0.25, 0, 0)
body velocity       = (0.15, 0, 0.05) m/s
angular velocity    = (0.10, 0.05, 0.10) rad/s
projected gravity   = (0.10, 0, approximately -0.995)
leg squared error   = 0.20
head errors         = (0.10, -0.10, 0, 0.20) rad
one foot air time   = 0.20 s; the other foot is supported
action delta sumsq  = 0.08
action-rate weight  = 0.1
```

Then:

| Term | Value |
| --- | ---: |
| `track_lin_vel` | `1.764994` |
| `track_ang_vel` | `1.911995` |
| `upright` | `1.637462` |
| `pose` | `0.670320` |
| `head_pose` | `1.886861` |
| `feet_air_time` | `3.000000` |
| `action_rate_penalty` | `-0.008000` |
| `ang_vel_xy_penalty` | `-0.000625` |
| **total** | **`10.863007`** |

This high reward still says nothing about crossing. The robot could earn it
while walking on an early assisted spawn or before reaching the end platform.

## 6. Physics-only curriculum

The curriculum has five stages:

| Stage | Assistance $\alpha$ | Robot spawn `x` | Interpretation |
| ---: | ---: | ---: | --- |
| 0 | `1.0` | `0.00 m` | starts near plank center with full plank stabilization |
| 1 | `0.6` | `-0.30 m` | longer plank approach, reduced stabilization |
| 2 | `0.3` | `-0.55 m` | begins at the plank/start-platform boundary |
| 3 | `0.1` | `-0.75 m` | begins on the launch platform |
| 4 | `0.0` | `-0.90 m` | exact unassisted evaluation spawn |

Assistance acts only on the plank. For plank displacement
$\Delta\mathbf p=\mathbf p-\mathbf p_0$ and linear velocity $\mathbf v$:

$$
\mathbf F_{xy}
=\alpha(-45\Delta\mathbf p_{xy}-4\mathbf v_{xy}),
$$

$$
F_z=\alpha(-25\Delta p_z-2.5v_z).
$$

For plank up-axis $\mathbf z_p$, world up $\mathbf z_w$, and world angular
velocity $\boldsymbol\omega_w$:

$$
\boldsymbol\tau
=\alpha\left[
3.5(\mathbf z_p\times\mathbf z_w)
-0.18\boldsymbol\omega_w
\right].
$$

At `alpha=0`, the applied force array is exactly zero. The curriculum never
pushes the robot forward, changes an action, modifies a reward coefficient, or
grants crossing credit.

After an episode:

```python
if crossed:
    stage = min(stage + 1, 4)
elif elapsed_s < 2.0:
    stage = max(stage - 1, 0)
# Otherwise keep the same stage.
```

This follows the playbook's exploration rule: if the target state never occurs,
change physics or starts so it can be sampled; do not invent a sparse jackpot.
The easiest rung did produce one contact-valid crossing in a diagnostic
`alpha_walking.onnx` probe, but that attempt later fell. It is exploration
evidence, not a strict skill pass.

## 7. Transfer from the shipped walking actor

A full Bridge run without `--init-from` automatically creates an initializer
from `microduck/policies/alpha_walking.onnx` when
`total_timesteps > 4`.

The bootstrap validates:

- input name `obs`, shape `(1,61)`;
- output name `actions`, shape `(1,14)`;
- exact ONNX node sequence;
- expected tensor names, shapes, and `float32` types;
- finite normalizer mean and positive standard deviation; and
- unchanged source hash during transfer.

It creates a new MLX actor-critic. The four actor linear layers are copied from
the ONNX actor. The observation mean and variance are copied from the ONNX
normalizer. The critic is freshly seeded. `actor_log_std` is newly initialized
from `initial_std=0.03`. The training caller creates a new Adam optimizer.

The actor is **not frozen**. It is a warm start and remains trainable. The
observation normalization statistics are frozen so the transferred actor keeps
the coordinate system in which the walking weights were trained.

Before saving, the code compares the transferred MLX actor with the source ONNX
on 32 representative observations. The allowed maximum absolute action error
is `1e-4`. The recorded `bridge-studio-02` initializer measured
`1.0728836e-6`.

The resulting continuation is accurately summarized as:

```text
actor mean weights:       transferred, then trainable
observation normalizer:   transferred and frozen
critic weights:           fresh seeded MLX initialization
actor exploration std:    fresh, initial value 0.03
optimizer state:          fresh Adam
```

It is not an exact PPO resume from the upstream walking run.

## 8. Reproduce the workflow

Run these commands from the workspace root. They use the existing
`rlx/.venv-microduck` environment through `scripts/train-bridge.sh`.

### 8.1 Build and inspect the simulated scene

The Studio parser has no `build` subcommand. `bridge_model()` builds the MuJoCo
model at runtime. This checked command constructs one environment and verifies
the public dimensions:

```bash
rlx/.venv-microduck/bin/python - <<'PY'
from rlx.environments.bridge import BridgeEnv

env = BridgeEnv(
    actuator="xml",
    curriculum=False,
    obs_noise=False,
    domain_rand=False,
    action_delay=False,
    random_yaw=False,
    seed=7,
)
obs, info = env.reset(seed=7)
print(obs.shape, env.action_space.shape, env.model.ntendon)
print(round(info["bridge_position"][2], 3), info["assistance"])
env.close()
PY
```

Expected output:

```text
(61,) (14,) 4
0.225 0.0
```

### 8.2 Train the bounded pilot recipe

```bash
./scripts/train-bridge.sh train \
  --bridge-curriculum \
  --total-timesteps 32768 \
  --num-envs 4 \
  --num-steps 128 \
  --num-minibatches 4 \
  --update-epochs 2 \
  --learning-rate 1e-5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-coefficient 0.1 \
  --entropy-coefficient 0 \
  --max-grad-norm 0.5 \
  --initial-std 0.03 \
  --backend dummy \
  --actuator xml \
  --max-episode-s 20 \
  --seed 7 \
  --no-normalize-rewards \
  --no-domain-rand \
  --no-obs-noise \
  --no-action-delay \
  --no-random-yaw \
  --output-dir rlx/runs/studio/bridge/my-bridge
```

Training automatically writes `bridge.safetensors`, its JSON sidecar,
`bridge.onnx`, an initial snapshot, and `training-metrics.jsonl`. Periodic
snapshots additionally depend on `--checkpoint-interval`.

The batch accounting is exact:

$$
4\text{ envs}\times128\text{ steps}=512
\text{ transitions per rollout}.
$$

$$
32768/512=64\text{ rollouts}.
$$

Each rollout uses four minibatches and two epochs:

$$
64\times4\times2=512
\text{ optimizer minibatch steps}.
$$

### 8.3 Export explicitly

Training exports ONNX by default. To rebuild only the deterministic ONNX actor:

```bash
./scripts/train-bridge.sh export \
  --checkpoint rlx/runs/studio/bridge/my-bridge/bridge.safetensors \
  --output rlx/runs/studio/bridge/my-bridge/bridge.onnx
```

The export contains the observation normalizer and actor mean. It does not
export the critic or stochastic action sampling.

### 8.4 Evaluate the strict skill gate

```bash
./scripts/train-bridge.sh eval \
  --backend dummy \
  --num-envs 3 \
  --seed 101 \
  --policy rlx/runs/studio/bridge/my-bridge/bridge.onnx \
  --evaluation-mode skill \
  --eval-steps 1000 \
  --max-episode-s 20 \
  --actuator xml \
  --no-domain-rand \
  --no-obs-noise \
  --no-action-delay \
  --no-random-yaw
```

Bridge skill evaluation rejects the `fork` backend because it needs
control-step metrics. Use `dummy` or `subproc`. Skill mode also requires
complete episode-horizon multiples and at least 1,000 steps per 20-second
episode. The command exits with status 2 when the skill result fails, even if
the inference pipeline itself stayed finite.

### 8.5 Render and look

```bash
./scripts/train-bridge.sh render \
  --policy rlx/runs/studio/bridge/my-bridge/bridge.onnx \
  --output rlx/runs/studio/bridge/my-bridge/render \
  --render-seconds 20 \
  --episodes 1 \
  --seed 101 \
  --actuator xml \
  --camera side \
  --width 640 \
  --height 360 \
  --fps 25 \
  --sheet-frames 12
```

Rendering forces domain randomization, observation noise, action delay, random
yaw, and bridge assistance off. If the robot falls before 20 seconds, the
renderer resets and continues recording; the overlay reports attempt and reset
counts. A 20-second video may therefore contain several failed attempts rather
than one complete episode.

## 9. The strict full-20-second crossing rule

Environment-level `crossing_accepted` becomes sticky only when all of these are
true at one control step:

1. trunk `x >= END_PLATFORM_X-0.08 = 0.77 m`;
2. both feet contact the end platform;
3. neither foot contacts the ground;
4. no robot body contacts the ground;
5. no non-foot body supports the robot; and
6. at least one earlier foot contact with the suspended plank was observed.

The separate `BridgeEvaluation` then requires more:

- the episode starts with launch support near progress zero;
- maximum progress is at least `1.60 m`;
- assistance is zero on every measured step;
- no floor or non-foot support occurs;
- the contact-validated crossing flag is observed;
- the episode continues to a 1,000-step truncation without termination; and
- every evaluated lane and episode passes.

Progress is measured from the final launch spawn:

$$
p=x_{\text{trunk}}-(-0.90)=x_{\text{trunk}}+0.90.
$$

Thus the `1.60 m` progress threshold alone corresponds to
`x_trunk >= 0.70 m`. It is deliberately insufficient: the stricter crossing
flag still requires `x >= 0.77 m`, two end-platform feet, prior plank contact,
and clean support history.

Crossing early and falling later fails because the required 20-second horizon
did not complete. Teleporting directly to the end platform fails because plank
contact history is missing. Walking on the floor fails even if the final `x`
coordinate is large.

## 10. What the current pilot actually achieved

The latest recorded final pilot is
`rlx/runs/studio/bridge/bridge-studio-02/`, produced on September 10, 2026.
Its checkpoint records:

| Item | Value |
| --- | ---: |
| New PPO transitions | `32,768` |
| Environments | `4` |
| Rollout length | `128` |
| Minibatches | `4` |
| Update epochs | `2` |
| Seed | `7` |
| Actuator | XML |
| Observation normalizer | transferred and frozen |
| Reward normalization | off |
| Curriculum | physics/spawn only |

The deterministic skill evaluation used three lanes seeded 101, 102, and 103,
1,000 control steps per lane, no randomization/noise/delay/yaw, XML actuators,
zero assistance, and the destination-stop commander.

The report contains 11 attempts because terminated lanes reset while the
3,000-transition evaluation continued. None completed the full 20-second
horizon. None crossed. Eight attempts reached the plank. The best attempt made
`1.174915875 m` of progress from the launch spawn, about

$$
1.60-1.174915875=0.425084125\ \text{m}
$$

short of the evaluator's progress threshold. No attempt recorded feet on the
end platform. One attempt contacted the floor. The pipeline was finite, but the
skill status was `failed`.

The correct summary is:

> The pilot learned or retained partial approach and plank-walking behavior,
> but no successful unassisted crossing was observed.

It is incorrect to say that 32,768 steps "completed the curriculum," that a
high return proves crossing, or that the render shows one continuous
20-second crossing. The retained video explicitly includes resets and partial
attempts.

## 11. Exercises

### Exercise 1: Cable geometry

Using anchor `(0.45,0.17,0.77)` and nominal attachment
`(0.45,0.057,0.225)`, calculate the cable length.

### Exercise 2: Commander correction

Assume the trunk is aligned with world `+x`, at `y=0.08 m`, and has not reached
the destination. What command is published?

### Exercise 3: Observation privilege

The plank rolls five degrees while the robot state and command remain exactly
unchanged. Which observation indices change because of the plank roll?

### Exercise 4: Curriculum transition

An environment is at stage 3. Determine the next stage after:

1. a clean crossing at 8 seconds;
2. a failure at 1.5 seconds;
3. a failure at 6 seconds.

### Exercise 5: PPO accounting

For 32,768 transitions, four environments, 128 rollout steps, four minibatches,
and two epochs, calculate rollouts, transitions per minibatch, and optimizer
minibatch steps.

### Exercise 6: Honest crossing

For each case, state pass or fail:

1. progress 1.65 m, crossed flag true at step 700, no invalid contacts, then
   truncation at step 1,000;
2. the same crossing, followed by a fall at step 850;
3. two feet on the end platform with no earlier plank contact;
4. progress 1.59 m with a true crossing flag and full horizon;
5. a perfect 20-second episode with assistance `0.1` for one step.

### Exercise 7: Result interpretation

The policy earns a larger mean return than its initializer but has
`bridge_crossed.max = 0`. What may be claimed?

## 12. Answers

### Answer 1

The longitudinal difference is zero, lateral difference is `0.113 m`, and
vertical difference is `0.545 m`:

$$
L=\sqrt{0^2+0.113^2+0.545^2}=0.556591412079\ \text{m}.
$$

### Answer 2

With heading `(1,0,0)`, side `(0,1,0)`, and `y=0.08`,

$$
d_y=\operatorname{clip}(-1.2(0.08),-0.12,0.12)=-0.096.
$$

The yaw error is zero, so the command is

```text
(0.25, -0.096, 0.0)
```

before floating-point conversion.

### Answer 3

None change directly. There is no plank-state block in the 61-vector. Future
robot motion caused by contact may change IMU, joints, or previous action, but
the isolated plank pose is privileged.

### Answer 4

1. crossing: stage 4;
2. failure before 2 seconds: stage 2;
3. later failure: remain at stage 3.

### Answer 5

There are `4*128=512` transitions per rollout and `32768/512=64` rollouts.
Each minibatch contains `512/4=128` transitions. Two epochs over four
minibatches give `8` optimizer steps per rollout and `64*8=512` total.

### Answer 6

1. pass, assuming launch support and metric coverage are valid;
2. fail: the episode terminated before the required horizon;
3. fail: suspended-plank foot-contact history is required;
4. fail: progress is below `1.60 m`;
5. fail: any assisted step invalidates unassisted evaluation.

### Answer 7

You may claim improved return under the inherited locomotion objective, if the
comparison is otherwise controlled. You may not claim a bridge crossing or a
mastered bridge skill. Deterministic physical evaluation remains the deciding
evidence.
