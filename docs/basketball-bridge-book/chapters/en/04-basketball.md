# Basketball: Blind Recurrent Balance on a Free Ball

## Learning objectives

After this chapter, you should be able to:

- reconstruct the basketball scene from the numbers in the repository;
- explain every slice of the shared 61-value observation and all 14 actions;
- describe why the actor needs recurrent memory even though the ball is absent from its input;
- follow hidden and cell state through reset, rollout collection, PPO replay, ONNX export, evaluation, and Studio playback;
- calculate every local reward term from the implemented formulas and coefficients;
- explain the six-level physical-assistance curriculum without calling it a reward curriculum;
- distinguish a released-actor warm start from a full optimizer resume and from training truly from scratch;
- run the parser-verified training, evaluation, rendering, and Studio-adapter commands;
- apply the physical evaluation gate and report the retained result honestly: **6/6 sustained balance, 0/3 commanded rolling**;
- state the limits: steering is not solved, lateral and yaw steering are not assessed, bridge behavior is not established, and no hardware claim is made.

## Evidence and claim boundary

This chapter uses repository code and retained repository artifacts only. The main sources are:

| Question | Repository source |
|---|---|
| Shared observation, joint order, timing, and default pose | `microduck_local/src/microduck_local/contract.py` |
| Action application, one-step joint-velocity lag, and observation assembly | `microduck_local/src/microduck_local/walk_env.py` |
| Basketball scene, assistance, termination, rewards, and metrics | `rlx/rlx/environments/basketball.py` |
| LSTM actor, local critic, source verification, ONNX export, and parity | `rlx/rlx/models/basketball.py` |
| Local recurrent PPO and direct CLI | `rlx/examples/ppo_microduck_basketball.py` |
| First-fall physical evaluator | `rlx/scripts/eval_basketball_local.py` |
| Studio train/eval/render/import adapter | `rlx/examples/ppo_microduck_balance.py` |
| Studio recurrent-policy loader | `microduck_local/src/microduck_local/studio_policies.py` |
| Human-readable result summary | `docs/basketball-showcase/RESULTS.md` |
| Retained direct evidence | `docs/basketball-showcase/evidence/evaluation.json` |
| Latest retained Studio evidence | `rlx/runs/studio/basketball/basketball-balance-01/evaluation.json` |

The retained artifacts were produced on September 10, 2026. The latest Studio report evaluates seeds 101, 102, and 103 for 60 seconds under zero and `0.15 m/s` forward commands. All six trials sustained balance, but none of the three commanded trials passed controlled rolling. The adapter also states that lateral and yaw command trials are not implemented. Therefore:

> The demonstrated result is unassisted sustained balance on a free basketball. It is not controlled rolling, general steering, bridge traversal, climbing onto the ball, recovery from the floor, or hardware validation.

The source basketball release in `microduck-playground` reports a much larger upstream evaluation, but that published evaluation is not a result of this local trainer. This chapter keeps source evidence, local evidence, and Studio evidence separate.

**Block labels used in this chapter**

- **Source excerpt:** an actual repository excerpt shown for study. Surrounding imports or class context may be omitted, so it is not a standalone executable program.
- **Executable command:** a shell command runnable from the workspace root when the documented environment and assets are available. Training commands create new artifacts.
- **Recorded text/data:** a literal path, sequence, or metadata value from repository evidence. It is not a command.

## 1. Build the physical scene from numbers

### 1.1 Units

The implementation uses the usual MuJoCo metric quantities, made explicit by variable and metric names:

| Quantity | Unit | Example |
|---|---|---|
| length and position | metre, `m` | ball radius `0.12 m` |
| mass | kilogram, `kg` | ball mass `0.62 kg` |
| time | second, `s` | control interval `0.02 s` |
| linear velocity | metres per second, `m/s` | command `0.15 m/s` |
| angle | radian internally | joint offsets and yaw command |
| angular velocity | radians per second, `rad/s` | yaw and ball rotation rates |
| reported tilt | degree | evaluator limits tilt to less than `50 deg` |
| force | newton, `N` | assistance spring and damping forces |
| torque | newton-metre, `N m` | assistance torques |

A radian is an angle measured by arc length divided by radius. For orientation, `pi rad = 180 deg`. The policy's 14 action values are joint-angle offsets in radians, not motor torques.

### 1.2 Solver, floor, ball, and visual mesh

The scene starts from the full-collision Microduck robot and adds a floor and a free body:

![Schematic basketball geometry using the implemented radius and spawn heights](../../assets/diagrams/schematic-basketball-geometry.png)

**Source excerpt (not standalone):** `rlx/rlx/environments/basketball.py`

```python
# rlx/rlx/environments/basketball.py
BALL_RADIUS = 0.12
BALL_MASS = 0.62

spec.option.timestep = C.PHYSICS_DT
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
spec.option.iterations = 10
spec.option.ls_iterations = 20

spec.worldbody.add_geom(
    name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[0, 0, 0.05], material="basketball_ground_mat",
    friction=[1, 0.005, 0.0001],
)

ball = spec.worldbody.add_body(
    name="basketball", pos=[0, 0, BALL_RADIUS + 0.001]
)
ball.add_freejoint(name="basketball_freejoint")
ball.add_geom(
    name="ball_sphere", type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[BALL_RADIUS, 0, 0], mass=BALL_MASS,
    friction=[1.2, 0.01, 0.001], priority=1, condim=4,
)
```

The sphere's centre begins at `z = 0.121 m`, one millimetre above a radius of `0.12 m`. Its physical geom has mass and contact. A second mesh geom supplies the basketball appearance but has `mass=0`, `contype=0`, and `conaffinity=0`; it cannot secretly support the robot. The texture and mesh come from `microduck-playground/src/mjlab_microduck/robot/assets/basketball/`.

The floor uses checkerboard texture colours and friction `[1, 0.005, 0.0001]`. The ball uses `[1.2, 0.01, 0.001]`, contact dimension 4, and priority 1. These are simulator contact parameters, not measured certificates for a particular real floor or ball.

The free joint gives the basketball seven configuration values: three position coordinates plus a four-value quaternion. It also gives six velocity values: three linear and three angular. Tests lock the complete model at `nq=28`, `nv=26`, and `nu=14`.

### 1.3 Timing

The physics step is

$$
\Delta t_{\text{physics}}=0.005\text{ s}.
$$

Each policy action is held for four physics steps:

$$
\Delta t_{\text{control}}
=4(0.005)
=0.02\text{ s},
$$

so the policy runs at

$$
f_{\text{control}}=\frac{1}{0.02}=50\text{ Hz}.
$$

A 60-second evaluation therefore contains exactly

$$
\frac{60}{0.02}=3000
$$

control steps per trial.

### 1.4 Spawn geometry

The ball centre is at `0.121 m`. The robot root is placed at

$$
z_{\text{root}}
=2R+0.128
=2(0.12)+0.128
=0.368\text{ m}.
$$

The root is therefore

$$
0.368-0.121=0.247\text{ m}
$$

above the ball centre and about

$$
0.368-(0.121+0.12)=0.127\text{ m}
$$

above the ball's top. The reset test locks `root_above_ball_m` to `0.247`.

The robot starts upright near the apex. Its horizontal root position is sampled within `+-0.01 m`, roll and pitch within `+-2 deg`, yaw across `[-pi, pi]` when random yaw is enabled, and every joint within `+-0.05 rad` of the default pose. The ball starts at rest. This task does not teach mounting the ball or recovering from the ground.

## 2. The exact 61-observation and 14-action contract

### 2.1 Observation slices

The actor always receives a `float32[61]` vector:

| Python slice | Indices | Width | Meaning |
|---|---:|---:|---|
| `0:3` | 0-2 | 3 | trunk angular velocity |
| `3:6` | 3-5 | 3 | gravity projected into the trunk frame |
| `6:20` | 6-19 | 14 | joint position minus the default pose |
| `20:34` | 20-33 | 14 | joint velocity, delayed one control step |
| `34:48` | 34-47 | 14 | previous raw actor action |
| `48:51` | 48-50 | 3 | forward, lateral, and yaw velocity command |
| `51:55` | 51-54 | 4 | head-command padding, forced to zero |
| `55:61` | 55-60 | 6 | body-command padding, forced to zero |

The assembly is direct:

**Source excerpt (not standalone):** `microduck_local/src/microduck_local/walk_env.py`

```python
# microduck_local/src/microduck_local/walk_env.py
obs = np.empty(C.OBS_DIM, np.float32)
obs[0:3] = gyro
obs[3:6] = gravity
obs[6:20] = joint_pos
obs[20:34] = joint_vel
obs[34:48] = self.last_action
obs[48:51] = self.twist_cmd
obs[51:55] = self.head_cmd
obs[55:61] = self.body_cmd
```

Basketball's command sampler sets the last ten values to zero. Both the trainer and tests reject nonzero `51:61` values. Moving the ball without changing robot proprioception leaves the observation unchanged in the focused test. The actor is therefore **blind to explicit ball state**: no ball position, velocity, rotation, radius, contact label, or world position appears in the 61 values.

The word "blind" does not mean information-free. Contact with a moving ball changes body angular velocity, projected gravity, joint state, and the action-response history. The LSTM can use that time sequence to infer useful hidden dynamics.

### 2.2 Joint and action order

The 14 action entries use this fixed servo order:

| Action index | Joint | Default pose, rad |
|---:|---|---:|
| 0 | left hip yaw | 0.0000 |
| 1 | left hip roll | -0.0873 |
| 2 | left hip pitch | -0.4579 |
| 3 | left knee | -0.0049 |
| 4 | left ankle | 0.4530 |
| 5 | neck pitch | 0.3491 |
| 6 | head pitch | 0.3491 |
| 7 | head yaw | 0.0000 |
| 8 | head roll | 0.0000 |
| 9 | right hip yaw | 0.0000 |
| 10 | right hip roll | 0.0873 |
| 11 | right hip pitch | 0.4579 |
| 12 | right knee | 0.0049 |
| 13 | right ankle | -0.4530 |

For raw actor output $\mathbf a_t\in\mathbb R^{14}$, the applied position target is

$$
\mathbf q^{\text{target}}_t
=\mathbf q^{\text{default}}
+\operatorname{clip}(\mathbf a_t,-4,4).
$$

The observation stores the **raw** action, not the clipped action. This keeps the next observation and action-change penalty aware of what the network requested. Under BAM actuation, the position target drives the voltage-based servo model, including its inherited 3-6 physics-step bus delay. Four physics steps make one control step, so that internal delay is shorter than two control intervals.

## 3. The recurrent actor and its state lifecycle

### 3.1 Network

The local actor is:

$$
61
\rightarrow \operatorname{LSTM}(256)
\rightarrow 512
\rightarrow 256
\rightarrow 128
\rightarrow 14,
$$

with ELU activations in the multilayer perceptron. The diagonal Gaussian exploration scale has 14 trainable values and starts at the requested `initial_std`, usually `0.03`. It is a direct standard-deviation parameter, clamped to at least `1e-6`, not a log-standard-deviation parameter.

The observation normalizer computes

$$
\widehat{\mathbf o}
=\frac{\mathbf o-\boldsymbol\mu}
{\boldsymbol\sigma+0.01}.
$$

The imported source normalizer is frozen. Its stored standard deviation for the ten zero-padded head/body slots is zero, so the `+0.01` denominator keeps those divisions finite. The exported ONNX contains the normalizer.

The local critic is different from the released upstream training critic described in the source model card. The local `BasketballCritic` takes only the same normalized 61 values and is a fresh feedforward `61 -> 512 -> 256 -> 128 -> 1` network. It has no ball-state input.

### 3.2 One-step inference

**Source excerpt (not standalone):** `rlx/rlx/models/basketball.py`

```python
# rlx/rlx/models/basketball.py
def forward(self, observations, h_in, c_in):
    normalized = self.obs_normalizer(observations)
    recurrent, (h_out, c_out) = self.rnn.rnn(
        normalized.unsqueeze(0),
        (h_in, c_in),
    )
    actions = self.mlp(recurrent.squeeze(0))
    return actions, h_out, c_out
```

The ONNX interface is exact:

- inputs: `obs [1,61]`, `h_in [1,1,256]`, `c_in [1,1,256]`;
- outputs: `actions [1,14]`, `h_out [1,1,256]`, `c_out [1,1,256]`;
- data type: `float32`.

### 3.3 When memory is carried and reset

At the beginning of a trial or episode, hidden state $\mathbf h$ and cell state $\mathbf c$ are zero. At every ordinary 50 Hz step:

1. send `obs`, `h_in`, and `c_in` to the actor;
2. apply `actions`;
3. carry `h_out` and `c_out` into the next call.

An ordinary command change does not reset memory. A trial reset, episode boundary, policy activation/switch, or recovery boundary must reset both tensors.

Training stores an `episode_starts[time, env]` mask. During collection and during every PPO replay, the code multiplies hidden and cell state by zero where an episode starts:

**Source excerpt (not standalone):** `rlx/rlx/models/basketball.py`

```python
# rlx/rlx/models/basketball.py
for step in range(observations.shape[0]):
    carry = (~episode_starts[step].bool()).to(observations.dtype)
    carry = carry.view(1, -1, 1)
    hidden = hidden * carry
    cell = cell * carry
    actions, hidden, cell = self(observations[step], hidden, cell)
    outputs.append(actions)
```

This matters because recurrent PPO cannot treat stored observations as unrelated rows. It recomputes the full time sequence from the rollout's initial hidden and cell tensors, applying the same episode masks. The evaluator similarly calls `policy.reset()` once at each trial start and carries state until first fall or the exact horizon.

The exporter runs a 40-step PyTorch/ONNX parity test with a state reset at step 20. The retained fixed-command actor's maximum absolute action error was `1.43e-6`; the mixed-command actor's was `2.15e-6`.

## 4. The exact local reward

### 4.1 Symbols

Let:

- $\mathbf v_{xy}$ be body-frame forward and lateral trunk velocity;
- $\mathbf c_{xy}$ be the commanded forward and lateral velocity;
- $\boldsymbol\omega$ be trunk gyroscope values;
- $c_\omega$ be commanded yaw rate;
- $\mathbf g$ be gravity projected into the trunk frame;
- $\mathbf d_{xy}$ be trunk position minus ball position in the horizontal plane;
- $g_i$ be foot $i$'s radial gap from `BALL_RADIUS + 0.012`;
- $h$ be trunk height above the ball centre;
- $\mathbf v_b$ be horizontal ball velocity;
- $\mathbf a_t-\mathbf a_{t-1}$ be the raw action change.

Every weighted term is multiplied by `CTRL_DT = 0.02`. The reward is:

$$
r_t=0.02\sum_k w_k f_k.
$$

### 4.2 Implemented formulas and weights

| Term | Unweighted formula $f_k$ | Weight $w_k$ |
|---|---|---:|
| linear tracking | $\exp(-\|\mathbf v_{xy}-\mathbf c_{xy}\|^2/0.1)$ | 1.0 |
| yaw tracking | $\exp(-(\omega_z-c_\omega)^2/0.5)$ | 0.5 |
| upright | $\exp(-\|(g_x,g_y)\|^2/0.09)$ | 1.0 |
| centred | $\exp(-\|\mathbf d_{xy}\|^2/0.0016)$ | 2.0 |
| feet on ball | $\frac12\sum_i\exp(-g_i^2/0.0004)$ | 1.0 |
| height | $\exp(-(h-0.245)^2/0.0009)$ | 1.0 |
| ball-speed penalty | $-\min(\|\mathbf v_b\|^2,100)$ | 0.05 |
| angular-velocity penalty | $-\min(\omega_x^2+\omega_y^2,100)$ | 0.05 |
| action-rate penalty | $-\min(\|\mathbf a_t-\mathbf a_{t-1}\|^2,100)$ | 0.2 |

The source is compact enough to compare directly:

**Source excerpt (not standalone):** `rlx/rlx/environments/basketball.py`

```python
# rlx/rlx/environments/basketball.py
terms = {
    "linear_tracking": math.exp(
        -float(np.sum((self.body_lin_vel()[:2] - self.twist_cmd[:2]) ** 2)) / .1
    ),
    "yaw_tracking": math.exp(
        -float((self._gyro[2] - self.twist_cmd[2]) ** 2) / .5
    ),
    "upright": math.exp(-float(np.sum(gravity[:2] ** 2)) / .09),
    "centered": math.exp(-float(np.sum(relative[:2] ** 2)) / .0016),
    "feet_on_ball": float(np.mean(np.exp(-gaps ** 2 / .0004))),
    "height": math.exp(-float((relative[2] - BALL_RADIUS - .125) ** 2) / .0009),
    "ball_speed_penalty": -float(min(np.sum(ball_velocity ** 2), 100)),
    "angular_velocity_penalty": -float(min(np.sum(self._gyro[:2] ** 2), 100)),
    "action_rate_penalty": -float(
        min(np.sum((self.last_action - self.prev_action) ** 2), 100)
    ),
}
weighted = {
    key: value * REWARD_WEIGHTS[key] * CTRL_DT
    for key, value in terms.items()
}
```

The reward uses privileged simulator state for the ball, contacts, and geometry. That is allowed for training rewards, but none of it enters the actor observation.

### 4.3 Worked reward example

Suppose one step has:

- command `0.15 m/s` forward and measured forward speed `0.01 m/s`;
- no lateral error;
- horizontal root-ball offset `0.02 m`;
- reset-like height `h=0.247 m`;
- raw action-change squared sum `0.04`;
- all other positive terms equal to 1 and other penalties zero.

Linear tracking:

$$
f_{\text{lin}}
=\exp\left(-\frac{(0.01-0.15)^2}{0.1}\right)
=\exp(-0.196)
\approx0.8220.
$$

Its weighted contribution is

$$
0.8220(1.0)(0.02)\approx0.01644.
$$

Centred:

$$
f_{\text{center}}
=\exp\left(-\frac{0.02^2}{0.0016}\right)
=e^{-0.25}
\approx0.7788,
$$

so the contribution is

$$
0.7788(2.0)(0.02)\approx0.03115.
$$

Height:

$$
f_{\text{height}}
=\exp\left(-\frac{(0.247-0.245)^2}{0.0009}\right)
\approx0.9956,
$$

giving approximately `0.01991`.

Action-rate penalty:

$$
(-0.04)(0.2)(0.02)=-0.00016.
$$

If every positive term were exactly 1 and every penalty were zero, the maximum per-step total would be

$$
0.02(1+0.5+1+2+1+1)=0.13.
$$

This is a dense shaping score, not the success test. The evaluator deliberately ignores reward totals.

## 5. Physical-assistance curriculum

### 5.1 Ladder and transition rule

The only curriculum levels are:

**Recorded text/data (not executable):** the implemented assistance ladder.

```text
1.0 -> 0.5 -> 0.25 -> 0.1 -> 0.03 -> 0.0
```

After a completed episode:

- duration at least `6 s`: move one level toward zero assistance;
- duration below `1.5 s`: move one level toward stronger assistance;
- otherwise: remain at the same level.

The transition is applied on the next reset and only if the previous episode actually finished. Reward weights never change.

### 5.2 What assistance does

The assistance method first clears every external force. At `hold=0`, it returns immediately. For nonzero hold, it applies:

**Source excerpt (not standalone):** `rlx/rlx/environments/basketball.py`

```python
# rlx/rlx/environments/basketball.py
ball_force = -400 * (ball_pos - self._ball_anchor) - 20 * ball_vel[:3]
ball_force[2] = -20 * ball_vel[2]
self.data.xfrc_applied[self.ball_body_id, :3] = self.hold * ball_force

self.data.xfrc_applied[self.ball_body_id, 3:] = (
    -self.hold * .2 * (ball_rotation @ ball_vel[3:])
)

duck_force = -40 * (
    self.data.xpos[self.trunk_body_id] - ball_pos
) - 4 * self.data.qvel[:3]
duck_force[2] = 0
self.data.xfrc_applied[self.trunk_body_id, :3] = self.hold * duck_force

self.data.xfrc_applied[self.trunk_body_id, 3:] = self.hold * (
    3 * np.cross(up, [0, 0, 1]) - .08 * angular_world
)
```

The ball receives horizontal spring-damper support toward its anchor, vertical damping without a vertical position spring, and rotational damping. The trunk receives horizontal centring force plus an upright-restoring and angular-damping torque.

**Worked assistance example.** If the ball is displaced `0.02 m` in positive x with zero velocity, its unscaled x force is

$$
-400(0.02)=-8\text{ N}.
$$

At `hold=0.25`, the applied x force is `-2 N`. At `hold=0`, it is exactly zero.

The retained curriculum smoke ran 7,168 environment steps with two environments, completed 39 episodes, and exercised both promotion and demotion. It ended with environments between `hold=0.25` and `hold=0.5`. It did **not** graduate to zero and does not demonstrate learning free-ball balance from scratch.

## 6. Training: warm start, PPO, and the scratch limit

### 6.1 What is loaded

The direct trainer defaults to:

**Recorded text/data (not executable):** the default source-checkpoint path.

```text
microduck-playground/artifacts/basketball/checkpoint.pt
```

It verifies the released source checkpoint and adjacent ONNX with fixed SHA256 values. It then loads `actor_state_dict`, freezes the source observation normalizer, and overwrites the actor's exploration standard deviation with `--initial-std`.

It does **not** restore the source critic, Adam optimizer, GPU curriculum counter, or exact mjlab simulator state. Instead:

**Source excerpt (not standalone):** `rlx/examples/ppo_microduck_basketball.py`

```python
# rlx/examples/ppo_microduck_basketball.py
actor, source_metadata = load_source_actor(...)
critic = BasketballCritic(actor.obs_normalizer)
optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
```

The retained summaries correctly label this:

**Recorded text/data (not executable):** retained run metadata.

```text
actor warm-start with fresh local critic and optimizer
full_upstream_resume: false
```

A local checkpoint may be used as another actor warm start, but it must have the local format, frozen-normalization contract, an adjacent ONNX, and recurrent parity. This is still not a full upstream resume.

### 6.2 What “scratch” would require

The local CLI has no `--scratch` or random-actor option. `train()` always calls `load_source_actor()`. The Studio adapter also defaults to the released checkpoint or an `--init-from` checkpoint. Therefore the repository contains no retained evidence that this local reward, curriculum, and CPU PPO implementation can discover basketball balance from a random actor.

The mature b11 actor is the source of the demonstrated balance. Local runs adapt it. The assistance smoke also begins from that actor. A truly scratch experiment would need a deliberate code path, a declared random initialization, a smoke test, and an unassisted first-fall evaluation. None is claimed here.

### 6.3 Rollout and PPO details

For `N` environments and `T` rollout steps, one update collects

$$
N\times T
$$

transitions. The retained fixed-command run used `N=8`, `T=128`, and 100 updates:

$$
8\times128\times100=102{,}400
$$

new local transitions.

The actor samples

$$
\mathbf a_t
=\boldsymbol\mu_t+\boldsymbol\epsilon_t\odot\boldsymbol\sigma,
\qquad
\boldsymbol\epsilon_t\sim\mathcal N(\mathbf0,\mathbf I).
$$

GAE uses `gamma=0.99` and `lambda=0.95`. A true termination does not bootstrap. A time-limit truncation uses the final state's critic value in the one-step delta but stops the multi-step trace across the reset. Advantages are normalized.

The PPO update uses one full recurrent sequence minibatch, five epochs, clip coefficient `0.2`, value coefficient `1.0`, entropy coefficient `0.01`, maximum gradient norm `1.0`, and target KL `0.02`. It recomputes the complete LSTM sequence each epoch. KL checking can stop later epochs, but a completed optimizer step can still leave post-update KL above the target.

The fixed-command run used learning rate `2e-6`, initial standard deviation `0.03`, and command `[0.15,0,0]`. The mixed-command run separately used 153,600 steps, learning rate `1e-5`, standard deviation `0.08`, and the direct trainer's randomized sampler: exact zero 25% of the time, otherwise uniform forward `[-0.15,0.15] m/s`, lateral `[-0.1,0.1] m/s`, and yaw `[-0.5,0.5] rad/s`. They are independent warm starts, not a single 256,000-step lineage.

![Observed PPO diagnostics for the two independent basketball continuation runs; the empty reward panel records that per-update reward history was not saved](../../assets/evidence/basketball-training-curves.png)

## 7. Parser-verified command line workflow

All commands below run from the workspace root. Their option spellings were checked against the current parsers. Training commands create new artifacts and can take substantial time.

### 7.1 Focused contract tests

**Executable command (workspace root):**

```bash
rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_basketball.py \
  rlx/tests/test_basketball_policy.py \
  rlx/tests/test_basketball_evaluation.py -q
```

### 7.2 Direct five-update smoke

Use a new output directory; the trainer refuses to overwrite a nonempty run.

**Executable command (workspace root):**

```bash
bash scripts/train-basketball.sh \
  --output rlx/artifacts/basketball-book-smoke \
  --num-envs 4 \
  --updates 5 \
  --steps 32 \
  --learning-rate 2e-6 \
  --initial-std 0.03 \
  --hold 0 \
  --no-curriculum \
  --command 0.15 0 0
```

This requests `4 x 32 x 5 = 640` new transitions. It is a wiring check, not learning evidence.

### 7.3 Reproduce the fixed-command local adaptation shape

**Executable command (workspace root):**

```bash
bash scripts/train-basketball.sh \
  --output rlx/artifacts/basketball-book-fixed \
  --num-envs 8 \
  --updates 100 \
  --steps 128 \
  --learning-rate 2e-6 \
  --target-kl 0.02 \
  --save-interval 25 \
  --initial-std 0.03 \
  --hold 0 \
  --no-curriculum \
  --command 0.15 0 0
```

### 7.4 Honest direct evaluation

The repeated `--command` option is intentional.

**Executable command (workspace root):**

```bash
rlx/.venv-microduck/bin/python rlx/scripts/eval_basketball_local.py \
  --policy rlx/artifacts/basketball-book-fixed/policy.onnx \
  --output rlx/artifacts/basketball-book-fixed-eval \
  --seconds 60 \
  --seeds 101 202 303 \
  --command 0 0 0 \
  --command 0.15 0 0 \
  --actuator bam \
  --render
```

The evaluator returns exit status 1 when the aggregate physical gate fails, while still writing `evaluation.json` and failure media. A nonzero status is expected when commanded rolling fails.

### 7.5 Studio-compatible training

The adapter rounds requested timesteps up to a whole `num_envs x num_steps` recurrent batch. This command divides exactly:

**Executable command (workspace root):**

```bash
rlx/.venv-microduck/bin/python rlx/examples/ppo_microduck_balance.py train \
  --recipe basketball \
  --output-dir rlx/runs/studio/basketball/basketball-book-fixed \
  --total-timesteps 102400 \
  --num-envs 8 \
  --num-steps 128 \
  --num-minibatches 1 \
  --learning-rate 2e-6 \
  --initial-std 0.03 \
  --target-kl 0.02 \
  --hold 0 \
  --no-curriculum
```

The adapter fixes the command to `[0.15,0,0]`, BAM actuation, a 10-second training horizon, frozen observation normalization, one full-sequence minibatch, and the standalone learner's PPO constants.

### 7.6 Studio evaluation and rendering

**Executable command (workspace root):**

```bash
rlx/.venv-microduck/bin/python rlx/examples/ppo_microduck_balance.py eval \
  --recipe basketball \
  --policy rlx/runs/studio/basketball/basketball-book-fixed/policy.onnx \
  --output-dir rlx/runs/studio/basketball/basketball-book-fixed \
  --num-envs 3 \
  --seed 101 \
  --max-episode-s 60 \
  --evaluation-mode skill
```

The adapter returns exit status 2 while full steering is unassessed or rolling fails. It evaluates both zero command and `0.15 m/s` forward command for seeds 101, 102, and 103.

**Executable command (workspace root):**

```bash
rlx/.venv-microduck/bin/python rlx/examples/ppo_microduck_balance.py render \
  --recipe basketball \
  --policy rlx/runs/studio/basketball/basketball-book-fixed/policy.onnx \
  --output-dir rlx/runs/studio/basketball/basketball-book-fixed \
  --output rlx/runs/studio/basketball/basketball-book-fixed/render \
  --episodes 1 \
  --seed 101 \
  --render-seconds 60 \
  --width 1280 \
  --height 720
```

Studio's renderer currently records the commanded-forward case only. A failed rolling render can still be valuable evidence of sustained balance and drift.

## 8. Evaluation: balance is not rolling

### 8.1 First-fall accounting

Evaluation does not auto-reset. It stops a trial at the first terminal state or at the exact horizon. A complete trial requires:

- exactly the required number of samples;
- no invalid physical metrics;
- no termination or truncation;
- zero automatic resets.

The environment terminates for low root height, root-ball offset above `0.16 m`, tilt above `55 deg`, robot-floor contact, non-foot robot-ball contact, or nonfinite physics. The evaluator is stricter on accepted geometry: maximum tilt must remain below `50 deg`, and maximum root-ball offset below `0.16 m`.

### 8.2 Sustained balance

A trial counts as sustained balance only when all are true:

$$
\begin{aligned}
&\text{duration complete}\ \land\ \text{foot-contact fraction}\ge0.5\\
&\land\ \text{no body-ball contact}\ \land\ \text{no robot-floor contact}\\
&\land\ \text{hold always zero}\ \land\ \text{stable geometry}.
\end{aligned}
$$

Balance does not require the ball to remain at one world position. A policy may balance while the ball drifts or turns.

### 8.3 Controlled rolling

For a nonzero planar command, controlled rolling additionally requires:

1. enough signed ball progress along the initial command direction;
2. low forward and lateral body-velocity error;
3. low yaw-rate error;
4. enough integrated ball rotation.

For command speed $s$, duration $T$, and ball radius $R=0.12$:

$$
d_{\min}=\max(0.25,\;0.35sT),
$$

$$
e_{\text{linear,max}}=\max(0.04,\;0.6s),
$$

$$
e_{\text{yaw,max}}=0.5,
$$

$$
\theta_{\min}
=\frac{\max(0.25,\;0.35sT)}{R}(0.5).
$$

**Worked 60-second gate.** With `s=0.15 m/s`:

$$
sT=0.15(60)=9.0\text{ m},
$$

$$
d_{\min}=0.35(9.0)=3.15\text{ m},
$$

$$
e_{\text{linear,max}}=0.6(0.15)=0.09\text{ m/s},
$$

$$
\theta_{\min}=\frac{3.15}{0.12}(0.5)=13.125\text{ rad}.
$$

Ball travel alone is insufficient. The evaluator labels substantial travel as random movement when signed command progress is too small relative to the required distance or to total travel.

### 8.4 Latest retained Studio result

The latest retained Studio evaluation uses BAM, action delay, random initial yaw, no pushes, no observation noise, no mass/friction domain randomization, no assistance, and no automatic reset.

| Case | Trials | Sustained balance | Controlled rolling | Case verdict |
|---|---:|---:|---:|---|
| command `[0,0,0]` | 3 | 3/3 | not the case criterion | pass |
| command `[0.15,0,0]` | 3 | 3/3 | 0/3 | fail |
| all trials | 6 | **6/6** | **0/3 commanded** | full task fail |

Every commanded trial failed both command-directed progress and velocity tracking. The adapter reports:

![Observed zero-command 60-second rollout, seed 101](../../assets/evidence/observed-basketball-zero-command-contact-sheet.png)

![Observed 0.15 m/s forward-command attempt, seed 101; balance continues but the rolling gate fails](../../assets/evidence/observed-basketball-forward-command-contact-sheet.png)

**Source excerpt (not standalone):** `rlx/examples/ppo_microduck_balance.py`

```python
# rlx/examples/ppo_microduck_balance.py
forward_rolling_passed = (
    bool(rolling_cases) and all(case["passed"] for case in rolling_cases)
)
rolling_passed = False
success = False
failures.append(
    "full steering is not assessed: lateral and yaw command trials are not implemented"
)
```

The hard-coded fail-closed result is deliberate. Even a future forward-only pass would not prove full steering. The current evidence does not pass forward rolling either.

## 9. Reading artifacts without overclaiming

Use the files for different questions:

| Artifact | What it can establish |
|---|---|
| `summary.json` | source hashes, settings, local steps, actor change, parity, PPO diagnostics |
| `checkpoint.pt` | local actor, fresh local critic, optimizer, and run metadata |
| `policy.onnx` | deterministic recurrent deployment graph with frozen normalizer |
| `evaluation.json` | every physical trial, criteria, failures, and aggregate verdict |
| `rollout.mp4` | visible motion for one rendered seed and command |
| `contact-sheet.png` | spaced visual inspection across the rollout |
| Studio `result.json` | adapter-level pass/fail and provenance |

Do not infer:

- steering from rotating ball texture;
- straight commanded travel from total path length;
- scratch learning from a curriculum smoke;
- exact upstream continuation from actor warm-start training;
- hardware safety from CPU MuJoCo;
- bridge traversal from basketball balance.

The correct stopping statement for the retained policy is:

> It sustains unassisted balance for all six retained 60-second Studio trials. It fails all three commanded-forward rolling trials. Lateral and yaw steering are unassessed. Bridge behavior and hardware behavior are outside this evidence.

## Exercises

1. The ball radius is `0.12 m`. What is the ball diameter?
2. At 50 Hz, how many control steps occur in 12 seconds?
3. Which observation indices hold the yaw command?
4. Does observation index 55 contain ball position?
5. A raw action at joint 4 is `0.20 rad`. Ignoring clipping, what ankle target is applied when the default is `0.4530 rad`?
6. Compute the centred reward's unweighted value for horizontal offset `0.04 m`.
7. At `hold=0.5`, a stationary ball is displaced `0.01 m` in x. What x assistance force is applied to the ball?
8. A curriculum environment at `hold=0.25` completes a `6.4 s` episode. What is the next hold?
9. How many new transitions are collected by 8 environments, 128 steps, and 25 updates?
10. For `0.15 m/s` over 60 seconds, why is `1.30 m` signed progress a failure even if the robot remains upright?
11. A trial has 60-second duration, 90% foot contact, zero hold, no forbidden contacts, `4 deg` maximum tilt, and `0.02 m` maximum offset. The ball travels `4 m` sideways under a forward command. Does it pass sustained balance? Does it pass controlled rolling?
12. Why is the retained 7,168-step curriculum smoke not evidence of scratch learning?

## Answers

1. $2R=2(0.12)=0.24\text{ m}$.
2. $12/0.02=600$ steps.
3. Index 50, the third value in slice `48:51`.
4. No. Slice `55:61` is zero body-command padding. The actor receives no explicit ball state.
5. $0.4530+0.20=0.6530\text{ rad}$.
6. $\exp(-0.04^2/0.0016)=\exp(-1)\approx0.3679$.
7. Unscaled force is $-400(0.01)=-4\text{ N}$; half hold applies `-2 N`.
8. The ladder moves one step toward zero: `0.25 -> 0.1`.
9. $8\times128\times25=25{,}600$ transitions.
10. The gate requires at least `3.15 m` signed progress, so `1.30 m` is below target. Upright survival alone is balance, not controlled rolling.
11. It passes sustained balance because those conditions satisfy the balance gate. It fails controlled rolling because sideways travel does not provide enough signed forward progress and likely fails velocity tracking.
12. The smoke started from the mature released actor, used physical assistance, and ended above zero assistance. The local CLI did not initialize a random actor, and no unassisted scratch policy was evaluated.
