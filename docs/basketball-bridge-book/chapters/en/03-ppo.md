# PPO: Learning Carefully from a Short Batch of Experience

Proximal Policy Optimization, usually shortened to **PPO**, is the learning algorithm used by the local RLX trainer. PPO does not directly teach a robot what basketball means. The environment supplies observations, actions, rewards, and episode boundaries. PPO uses those records to answer a narrower question:

> Given the actions the current policy just tried, which actions should become a little more likely, which should become a little less likely, and how large a change is safe enough to attempt?

This chapter develops that answer from high-school algebra, probability density, and averages. It then maps each idea to the repository's current implementation:

- `rlx/examples/ppo_microduck_basketball.py`: the recurrent PyTorch continuation trainer for the released basketball actor;
- `rlx/rlx/algorithms/ppo.py`: the generic feedforward MLX PPO implementation;
- `rlx/rlx/buffers/rollout_buffer.py`: the fixed-size rollout storage used by the MLX implementation.

The code is the authority for implementation details. PPO papers explain the general method, but this repository adds choices such as timeout-aware bootstrapping, a robust critic loss, numerical log-ratio clipping, and a basketball-specific KL stopping rule.

## The learning loop in plain language

Imagine coaching a player who must act every 20 milliseconds. At each tick:

1. The policy receives an observation, such as joint state, orientation, command, and recent motion information.
2. The actor produces a probability distribution over 14 continuous joint commands.
3. One action is sampled from that distribution.
4. The simulator advances and returns a reward plus boundary flags.
5. The critic predicts how much discounted future reward was expected from the old observation.

PPO saves a short block of these transitions. It then replays the saved actions through the updated policy and compares:

$$
\text{old log probability}
\quad\text{with}\quad
\text{new log probability}.
$$

If an action turned out better than the critic expected, PPO tries to increase its probability. If it turned out worse, PPO tries to decrease its probability. The word **proximal** means that the update is deliberately conservative: PPO limits how much benefit it can claim from moving the probability ratio too far in the helpful direction.

After several optimizer passes, the rollout is discarded. New data must be collected from the new policy. PPO is therefore **on-policy**: it does not keep an unlimited replay database containing actions from many old policies.

The two implementations in this repository organize that same idea differently:

| Path | Actor memory | Update organization | Main numerical framework |
|---|---|---|---|
| Generic `rlx/rlx/algorithms/ppo.py` | Feedforward | Flatten time and environments, shuffle, use minibatches | MLX |
| Basketball `ppo_microduck_basketball.py` | One-layer LSTM | Replay the entire time-by-environment sequence each epoch | PyTorch |

That difference is not cosmetic. A feedforward sample can be trained as an independent row. An LSTM action depends on the hidden state produced by earlier observations, so its history must be reconstructed in the correct order.

## Actor, critic, and the meaning of advantage

PPO trains two prediction systems.

The **actor** chooses actions. For continuous control, it produces a mean action $\mu(o)$ and a standard deviation $\sigma$. The basketball actor has 14 action dimensions. Its recurrent core first processes a 61-dimensional observation with an LSTM of hidden size 256, then an MLP maps the recurrent output through widths 512, 256, and 128 to 14 means.

The **critic** predicts a scalar value:

$$
V(o_t)\approx
\mathbb E[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots].
$$

The discount $\gamma$ makes distant rewards count less. With $\gamma=0.99$, a reward one step away is multiplied by $0.99$, two steps away by $0.99^2$, and so on.

The actor does not train directly from raw reward. It trains from an **advantage estimate**:

$$
\widehat A_t \approx
\text{outcome after action }a_t
-
\text{critic's expectation before }a_t.
$$

- $\widehat A_t>0$: the sampled action did better than the baseline; make it more likely.
- $\widehat A_t<0$: it did worse than the baseline; make it less likely.
- $\widehat A_t\approx0$: the sample provides little evidence for changing its probability.

“Positive advantage” does not mean “positive reward.” A reward of 2 can have negative advantage if the critic expected 5. A reward of $-1$ can have positive advantage if the critic expected $-4$. Advantage is a comparison with a baseline.

The basketball critic is feedforward even though the actor is recurrent. It reads the current normalized 61-vector and predicts one value. The actor and critic share the actor's frozen observation normalizer, but the basketball continuation creates a **fresh critic and fresh Adam optimizer**. It is an actor warm start, not a full optimizer-and-critic resume.

## Continuous actions use probability density

For one action coordinate, a Gaussian policy has density

$$
p(a\mid\mu,\sigma)
=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(a-\mu)^2}{2\sigma^2}
\right).
$$

The corresponding log density is

$$
\log p(a\mid\mu,\sigma)
=
-\frac12\left[
\frac{(a-\mu)^2}{\sigma^2}
+2\log\sigma
+\log(2\pi)
\right].
$$

For 14 independent Gaussian coordinates, the implementation sums 14 log densities:

$$
\log\pi(a\mid o)
=
\sum_{j=1}^{14}\log p(a_j\mid\mu_j(o),\sigma_j).
$$

This produces one joint log probability per transition.

### Density can exceed 1

Students often learn that a probability cannot exceed 1. That statement is true for the probability of an event, but a continuous Gaussian formula returns a **density**, not the probability of one exact point. A narrow density can be taller than 1 as long as the total area under the curve remains 1.

For $\mu=0$ and $\sigma=0.1$, the density at the mean is

$$
p(0)=\frac{1}{0.1\sqrt{2\pi}}\approx3.989.
$$

The probability of landing in an interval is the area over that interval. For a tiny interval of width $0.01$ near the mean, the probability is roughly density times width, about $3.989\times0.01=0.03989$, still below 1.

The log density can therefore be positive. That is legal. A positive continuous log density does not mean a probability greater than 1.

The following complete standard-library program reproduces the calculation:

```python
import math


def gaussian_log_density(x: float, mean: float, std: float) -> float:
    if std <= 0.0:
        raise ValueError("std must be positive")
    variance = std * std
    return -0.5 * (
        (x - mean) ** 2 / variance
        + 2.0 * math.log(std)
        + math.log(2.0 * math.pi)
    )


def main() -> None:
    log_density = gaussian_log_density(0.0, 0.0, 0.1)
    density = math.exp(log_density)
    approximate_interval_probability = density * 0.01

    print(f"log density: {log_density:.6f}")
    print(f"density: {density:.6f}")
    print(f"approximate probability in width 0.01: "
          f"{approximate_interval_probability:.6f}")

    assert density > 1.0
    assert 0.0 < approximate_interval_probability < 1.0


if __name__ == "__main__":
    main()
```

Expected output, up to final-digit rounding:

```text
log density: 1.383647
density: 3.989423
approximate probability in width 0.01: 0.039894
```

In `ppo_microduck_basketball.py`, `gaussian_log_prob()` implements the same expression with Torch tensors and sums over the action dimension. `gaussian_entropy()` also sums over dimensions.

## From log probabilities to the PPO ratio

The stored action was sampled under the old policy. PPO asks how its density changes under the new policy:

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}.
$$

Direct division of tiny densities is numerically awkward, so code uses log probabilities:

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid o_t)
-
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)
\right).
$$

The ratio is always positive because an exponential is positive.

- $r=1$: the saved action has the same density.
- $r=1.2$: its density is 20% larger.
- $r=0.7$: its density is 30% smaller.

The ratio is not “the probability of the action.” It is a comparison of new and old densities for the same recorded action.

The generic MLX code clips the **log ratio** to `[-20, 20]` before exponentiating. This is a numerical overflow guard. It is separate from PPO's much narrower policy clip such as $[0.8,1.2]$. The basketball trainer currently exponentiates the unbounded log-ratio directly, then rejects non-finite KL values or losses.

## The clipped policy objective

PPO's sample-level surrogate is

$$
L_t^{\mathrm{clip}}
=
\min\left(
r_t\widehat A_t,\;
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right).
$$

The algorithm wants to maximize this quantity. The implementation minimizes its negative. The generic MLX source writes the negative form as a `maximum`; the basketball source writes the positive form with `minimum` and negates the mean. They are algebraically equivalent.

With $\epsilon=0.2$, the clipped interval is $[0.8,1.2]$.

### Positive advantage

Suppose $\widehat A=+2$. Increasing the ratio is helpful because it makes a good action more likely.

- At $r=1.1$, both terms equal $2.2$.
- At $r=1.3$, the ordinary product is $2.6$, but the clipped product is $1.2\times2=2.4$. PPO uses $2.4$.
- At $r=0.7$, the ordinary product is $1.4$ and the clipped product is $1.6$. PPO uses the smaller value, $1.4$.

The objective stops rewarding an overly large helpful increase, but it does not erase a harmful decrease.

### Negative advantage

Suppose $\widehat A=-2$. Decreasing the ratio is helpful because it makes a bad action less likely.

- At $r=0.7$, the ordinary product is $-1.4$, while the clipped product is $0.8\times(-2)=-1.6$. PPO uses the smaller value, $-1.6$.
- At $r=1.3$, the ordinary product is $-2.6$ and the clipped product is $-2.4$. PPO uses $-2.6$.

For negative advantage, the helpful direction is downward. The plateau appears below $0.8$, not above $1.2$.

This standalone program prints both cases:

```python
def clipped_surrogate(ratio: float, advantage: float, epsilon: float = 0.2) -> float:
    clipped_ratio = min(max(ratio, 1.0 - epsilon), 1.0 + epsilon)
    return min(ratio * advantage, clipped_ratio * advantage)


def main() -> None:
    for advantage in (2.0, -2.0):
        print(f"advantage = {advantage:+.1f}")
        for ratio in (0.7, 1.0, 1.3):
            value = clipped_surrogate(ratio, advantage)
            print(f"  ratio={ratio:.1f}, surrogate={value:+.1f}")

    assert clipped_surrogate(1.3, 2.0) == 2.4
    assert clipped_surrogate(0.7, -2.0) == -1.6


if __name__ == "__main__":
    main()
```

Clipping is not a guarantee that the new policy is better. It only changes the training objective for the sampled batch. The network can still change on observations absent from that batch, multiple parameters interact, the critic can be wrong, and the reward can fail to represent the intended basketball skill.

![Schematic PPO clipping curves. This explanatory diagram is not measured training evidence.](assets/diagrams/schematic-ppo-clipping.png)

## Generalized Advantage Estimation

The one-step temporal-difference residual is

$$
\delta_t
=
r_t+\gamma b_t-V(o_t),
$$

where $b_t$ is the permitted bootstrap value after the transition. Generalized Advantage Estimation, or GAE, combines residuals backward:

$$
\widehat A_t
=
\delta_t+\gamma\lambda m_t\widehat A_{t+1}.
$$

Here:

- $\gamma$ discounts future return;
- $\lambda$ controls how strongly later residuals affect the current estimate;
- $m_t$ is a continuation mask that prevents a trace from crossing an episode reset.

With the repository defaults $\gamma=0.99$ and $\lambda=0.95$, the trace multiplier is

$$
\gamma\lambda=0.9405.
$$

A smaller $\lambda$ relies more on the critic's one-step predictions. A larger $\lambda$ carries more sampled future information backward, usually with more sampling variance. Neither value is universally best.

The critic target is

$$
\widehat R_t=\widehat A_t+V(o_t).
$$

The actor uses $\widehat A_t$; the critic fits $\widehat R_t$.

## Terminal and time-limit masks are different

Gymnasium separates two boundaries:

- **termination**: the modeled task truly ended, such as a failure condition;
- **truncation**: collection stopped for an external reason, commonly a time limit, although the underlying task could continue.

Both boundaries reset the environment. They do not receive the same value bootstrap.

For transition $t$, define:

$$
b_t=
\begin{cases}
0,&\text{if terminated},\\
V(o_{t+1}^{\mathrm{final}}),&\text{if truncated but not terminated},\\
V(o_{t+1}),&\text{if the episode continues}.
\end{cases}
$$

The GAE trace mask is zero for either termination or truncation:

$$
m_t=1-\mathbf 1[\text{terminated}\lor\text{truncated}].
$$

This gives two separate decisions:

1. **May the current residual bootstrap a value?** A pure timeout may; a true terminal may not.
2. **May advantage flow into the next stored transition?** Neither boundary may, because the next stored observation can belong to a reset episode.

The basketball implementation stores `bootstrap_values` for every transition. During collection, it evaluates the critic on the final pre-reset observation, then explicitly replaces the value with zero where `terminated` is true. Its `compute_gae()` uses:

- `~terminated` as the bootstrap mask;
- `~(terminated | truncated)` as the trace-continuation mask.

The generic MLX implementation reaches the same semantics differently. `_truncation_values()` selects only `truncated & ~terminated` environments and evaluates `info["terminal_observation"]`. The shared GAE helper uses the timeout value for a truncation, zero for termination, and the ordinary next value otherwise. It never bootstraps from an autoreset observation.

### Worked boundary example

Suppose:

$$
r_t=1,\quad V(o_t)=2,\quad
V(o_{t+1}^{\mathrm{final}})=5,\quad \gamma=0.99.
$$

If the transition is a true terminal:

$$
\delta_t=1+0-2=-1.
$$

If it is only a time limit:

$$
\delta_t=1+0.99(5)-2=3.95.
$$

If an autoreset observation has value 99, using it would give

$$
1+0.99(99)-2=97.01,
$$

which incorrectly assigns value from the next episode to the previous one.

The next program computes a three-step trace and proves that a reward in the reset episode cannot leak backward:

```python
from typing import Sequence


def gae(
    rewards: Sequence[float],
    values: Sequence[float],
    bootstraps: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float]]:
    size = len(rewards)
    if not all(len(x) == size for x in (
        values, bootstraps, terminated, truncated
    )):
        raise ValueError("all inputs must have equal length")

    advantages = [0.0] * size
    trace = 0.0
    for step in range(size - 1, -1, -1):
        bootstrap_mask = 0.0 if terminated[step] else 1.0
        continuation_mask = (
            0.0 if terminated[step] or truncated[step] else 1.0
        )
        delta = (
            rewards[step]
            + gamma * bootstraps[step] * bootstrap_mask
            - values[step]
        )
        trace = delta + gamma * gae_lambda * continuation_mask * trace
        advantages[step] = trace

    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def main() -> None:
    advantages, returns = gae(
        rewards=[0.5, 0.2, 0.0],
        values=[1.0, 1.2, 1.1],
        bootstraps=[1.2, 1.1, 0.9],
        terminated=[False, False, False],
        truncated=[False, False, True],
    )
    print([round(x, 6) for x in advantages])
    print([round(x, 6) for x in returns])

    changed_next_episode_reward = 1000.0
    assert changed_next_episode_reward > 0.0  # It is deliberately unused.
    assert round(advantages[-1], 6) == -0.209
    assert [round(x, 6) for x in advantages] == [
        0.586836, -0.107565, -0.209
    ]


if __name__ == "__main__":
    main()
```

The expected advantages are:

```text
[0.586836, -0.107565, -0.209]
```

The timeout transition bootstraps from 0.9, but its continuation mask is zero, so no later reset-episode advantage enters the trace.

## Advantage normalization

Raw advantages can have very different scales across tasks or batches. Both implementations normalize advantages:

$$
\widehat A'_t
=
\frac{\widehat A_t-\operatorname{mean}(\widehat A)}
{\operatorname{std}(\widehat A)+10^{-8}}.
$$

The basketball trainer normalizes once over the complete rollout before its epoch loop. The generic MLX trainer normalizes inside `_loss_and_metrics()` after selecting a minibatch, so each actual minibatch uses its own mean and standard deviation.

Normalization changes scale and can change a sample's sign relative to the selected group's mean. It does not change rewards, returns, or old log probabilities. It also does not mean that an advantage of `+2` in a teaching table will literally remain `+2` in training.

## The critic loss

The basketball continuation uses ordinary half squared error:

$$
L_V
=
\frac12\operatorname{mean}\left[
(V_{\mathrm{new}}-\widehat R)^2
\right].
$$

It does not clip the value update.

The generic MLX PPO uses a Huber-style error with threshold 10:

$$
h(e)=
\begin{cases}
\frac12e^2,&|e|\le10,\\
10|e|-50,&|e|>10.
\end{cases}
$$

When `clip_value_loss=True`, it also creates a clipped prediction:

$$
V_{\mathrm{clip}}
=
V_{\mathrm{old}}+
\operatorname{clip}(V_{\mathrm{new}}-V_{\mathrm{old}},-\epsilon,\epsilon),
$$

then uses the larger of the unclipped and clipped Huber errors. Choosing the larger prevents the clipped branch from making an excessively changed value prediction look artificially good.

This is an important implementation difference. “RLX PPO uses value clipping” is true for `rlx/rlx/algorithms/ppo.py`, but false for the basketball-specific `ppo_update()`.

## Entropy and why it can be negative

Entropy encourages exploration. For one Gaussian coordinate,

$$
H
=
\log\sigma+\frac12\log(2\pi e).
$$

For independent coordinates, the implementation sums this value.

This is **differential entropy**, not discrete Shannon entropy. Differential entropy may be negative. For example, with $\sigma=0.03$:

$$
H_{\text{one dimension}}\approx-2.087,
$$

so 14 equal dimensions have total entropy near $-29.2$.

Nothing is wrong with a negative value. It means the continuous density is concentrated in a narrow region. It does not mean negative probability.

Both loss functions subtract the entropy bonus:

$$
L_{\mathrm{total}}
=
L_{\mathrm{policy}}
+c_VL_V
-c_HH.
$$

If $H$ is negative, the term $-c_HH$ is positive. The optimizer still follows the exact differential-entropy objective; the sign should not be interpreted with discrete-entropy intuition.

The basketball actor stores a directly trained standard-deviation parameter and clamps it only to a minimum of `1e-6` when used. Loading a source actor then fills this parameter with the requested `initial_std`, whose command-line default is `0.03`. Thus the continuation deliberately resets exploration scale instead of retaining the source checkpoint's saved standard deviation.

## Total loss is not reward, and neither proves mastery

The basketball total loss is

$$
L
=
L_{\mathrm{policy}}
+1.0L_V
-0.01H
\quad\text{under its defaults}.
$$

The generic PPO defaults use value coefficient `0.5` and entropy coefficient `0.01`, although example programs can override them.

A falling training loss is not the same as a rising episode reward:

- policy loss measures a clipped probability objective on the latest batch;
- value loss measures critic prediction error;
- entropy measures distribution spread;
- reward comes from the environment;
- mastery is an external behavioral claim.

The total loss can rise while behavior improves because the critic is fitting larger return targets. It can fall while the robot learns a reward loophole. A high reward can still represent holding the ball incorrectly, exploiting assistance, or failing after the logged horizon. PPO only optimizes the reward and data it receives.

Therefore a basketball policy requires independent evaluation: complete episodes, physical task metrics, deterministic exported-policy tests, and rendered inspection. No scalar PPO metric alone establishes mastery.

## Recurrent sequence replay in basketball

The basketball actor is an LSTM. Its action mean at time $t$ depends on:

$$
\mu_t=f(o_t,h_{t-1},c_{t-1}).
$$

If training shuffled individual time steps, the hidden state would no longer match the one that generated the action. The basketball rollout therefore stores:

- observations with shape `[time, environments, 61]`;
- sampled actions and old log probabilities;
- old critic values, rewards, terminations, truncations, and bootstrap values;
- `episode_starts` for every time step;
- the initial LSTM hidden and cell states for the rollout.

During each PPO epoch, `actor.forward_sequence()` starts from the saved initial hidden state and walks forward in time. Before processing a step whose `episode_starts` flag is true, it multiplies the hidden and cell state by zero for that environment. This reconstructs the recurrent state while preventing memory from crossing resets.

The update uses the full sequence each epoch. There is no basketball minibatch loop in `ppo_update()`. With defaults of 32 steps and 4 environments, one rollout contains:

$$
32\times4=128\text{ transitions}.
$$

Five epochs present those same 128 transitions to the optimizer up to five times, unless KL stopping ends earlier. This is 128 new environment transitions, not 640 new transitions.

The critic is evaluated on a flattened `[time*environment, 61]` tensor because it is feedforward. Its outputs are reshaped back to `[time, environment]`.

## Feedforward minibatches in generic MLX PPO

`RolloutBuffer.reset()` allocates fixed arrays shaped `[num_steps, num_envs, ...]`. It stores:

| Field | Purpose |
|---|---|
| `observations` | input used to choose each action |
| `next_observations` | returned next input |
| `actions` | sampled action |
| `rewards` | scalar reward |
| `terminations`, `truncations` | separate boundary types |
| `values` | old critic predictions |
| `log_probs` | old-policy joint log probabilities |
| `advantages`, `returns` | allocated fields, although the current PPO path passes computed arrays directly to `update()` |

The generic update flattens the first two dimensions. For the library defaults:

$$
16\text{ steps}\times4096\text{ environments}
=65{,}536\text{ samples}.
$$

With 16 minibatches, each minibatch has 4,096 samples. Two epochs produce 32 optimizer steps per rollout.

The generic Microduck example does **not** use those large library defaults. `rlx/examples/ppo_microduck.py` defines 16 environments, 24 steps, 4 minibatches, and 5 epochs, then passes those values into `PPOConfig`. Its rollout has 384 transitions, each minibatch has 96, and one rollout produces 20 optimizer steps.

The MLX update creates a fresh random permutation each epoch. This is appropriate for its feedforward network because every saved observation contains the complete input expected by the network. `RolloutBuffer` stores no recurrent hidden state and no sequence-start mask; using it unchanged for an LSTM would lose necessary history.

## KL divergence: diagnostic in MLX, stopping rule in basketball

Both implementations calculate

$$
\widehat D_{\mathrm{KL}}
=
\operatorname{mean}\left[(r-1)-\log r\right].
$$

Because samples come from the old policy, this is a common sample approximation related to
$D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})$. It should be nonnegative in expectation, although a finite sample and floating-point arithmetic can produce small irregularities.

The implementations use it differently.

### Generic MLX behavior

`rlx/rlx/algorithms/ppo.py`:

- clips log ratios to `[-20,20]` before exponentiation;
- reports `approximate_kl` when metrics are requested;
- reports `clip_fraction`;
- does **not** define `target_kl`;
- does **not** stop epochs or minibatches based on KL.

Its KL is a diagnostic, not a gate.

### Basketball behavior

`ppo_microduck_basketball.py`:

- defaults `target_kl` to `0.02`;
- computes KL before each epoch's optimizer step;
- can stop with `epochs_completed == 0` if saved old log probabilities are already inconsistent with the current actor;
- recomputes the full recurrent sequence after each optimizer step;
- computes post-update KL and stops further epochs when it exceeds the target;
- records `kl_early_stop`.

The pre-update guard matters for stale or mismatched rollout data. The post-update guard limits repeated reuse of the same short recurrent rollout. It is still not a mathematical guarantee that every state changed by less than a fixed amount.

## Exact defaults and overrides

Do not describe “the PPO defaults” without naming the layer.

### Generic `PPOConfig` defaults

| Setting | Value |
|---|---:|
| `num_envs` | 4096 |
| `num_steps` | 16 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `num_minibatches` | 16 |
| `update_epochs` | 2 |
| `normalize_advantages` | true |
| `clip_coefficient` | 0.2 |
| `clip_value_loss` | true |
| `entropy_coefficient` | 0.01 |
| `value_coefficient` | 0.5 |
| `max_grad_norm` | 0.5 |

### Generic Microduck example overrides

`rlx/examples/ppo_microduck.py` replaces `num_envs`, `num_steps`, `num_minibatches`, and `update_epochs` with `16`, `24`, `4`, and `5`. It keeps gamma `0.99`, lambda `0.95`, clip `0.2`, and entropy coefficient `0.01`. Its optimizer learning-rate default is `1e-3`, which belongs to the example arguments rather than `PPOConfig`.

### Basketball trainer defaults

The basketball trainer does not instantiate `PPOConfig`. Its command line and function defaults produce:

| Setting | Value |
|---|---:|
| environments | 4 |
| rollout steps | 32 |
| updates | 5 |
| PPO epochs per rollout | 5 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| clip coefficient | 0.2 |
| value coefficient | 1.0 |
| entropy coefficient | 0.01 |
| gradient norm limit | 1.0 |
| Adam learning rate | `2e-5` |
| target KL | 0.02 |
| reset exploration standard deviation | 0.03 |

Its environment construction also overrides training conditions explicitly:

- episode limit: 10 seconds;
- actuator: `"bam"` by default;
- observation noise: off;
- domain randomization: off;
- pushes: off;
- curriculum: off;
- hold assistance: `0.0`;
- command: fixed `(0.08, 0.0, 0.0)` unless `--randomized-commands` is selected.

These choices describe the default local continuation experiment. They are not universal PPO recommendations and do not reproduce the complete upstream training recipe.

## One complete update, end to end

For the default basketball run:

1. Reset four environments with deterministic seed offsets.
2. Initialize LSTM hidden and cell tensors with shape `[1, 4, 256]`.
3. Collect 32 control steps, giving 128 transitions.
4. At every step, zero recurrent state for environments whose current observation starts a new episode.
5. Sample 14-dimensional Gaussian actions and save their joint old log probabilities.
6. Step each environment independently.
7. Evaluate final pre-reset observations for bootstrap values; force terminal bootstraps to zero.
8. Reset ended environments and mark the next observations as episode starts.
9. Run backward GAE with termination and truncation masks.
10. Normalize the 128 advantages together.
11. Replay the entire actor sequence from the saved initial LSTM state.
12. Check pre-update approximate KL.
13. Compute clipped policy loss, squared critic loss, and Gaussian entropy.
14. Backpropagate, clip the combined actor-and-critic gradient norm to 1.0, and step Adam.
15. Replay the sequence again, measure post-update KL, and decide whether another epoch is allowed.

The command-line defaults request five such rollout/update cycles:

$$
4\times32\times5=640
\text{ local environment transitions}.
$$

That small default is suitable as a short continuation or smoke-scale run, not evidence of basketball mastery.

## Common interpretation mistakes

**Mistake: “The Gaussian returned probability 3.9, which is impossible.”**  
Correction: it returned density. Probability is area over an interval.

**Mistake: “A positive log probability is invalid.”**  
Correction: continuous log density may be positive when the density exceeds 1.

**Mistake: “Clipping keeps every ratio between 0.8 and 1.2.”**  
Correction: the objective uses a clipped comparison. The actual ratio can lie outside the interval.

**Mistake: “A timeout gets zero future value because the environment reset.”**  
Correction: a pure time limit bootstraps from the final pre-reset observation, then stops the trace.

**Mistake: “Negative entropy means negative probability.”**  
Correction: continuous differential entropy can be negative.

**Mistake: “Lower total loss means higher reward.”**  
Correction: total loss mixes policy, critic, and entropy terms on one batch.

**Mistake: “High reward means the skill is mastered.”**  
Correction: reward is a designed training signal. Mastery requires independent physical criteria and inspection.

**Mistake: “The basketball updater uses MLX minibatches.”**  
Correction: it is a separate PyTorch recurrent updater that replays full sequences.

**Mistake: “Both PPO implementations stop at KL 0.02.”**  
Correction: only the basketball path has that target and early stopping. Generic MLX PPO reports KL without stopping.

## Exercises

1. A Gaussian has mean 0 and standard deviation 0.2. Compute its density at the mean. Can the answer exceed 1?
2. An action has old log density `-3.0` and new log density `-2.8`. Compute the PPO ratio.
3. With advantage `+3`, ratio `1.4`, and clip coefficient `0.2`, compute the unclipped product, clipped product, and selected surrogate.
4. Repeat Exercise 3 with advantage `-3`.
5. A transition has reward 2, current value 4, final-observation value 5, and gamma 0.9. Compute its TD residual for a true terminal and for a pure timeout.
6. Why must an LSTM rollout save its initial hidden state and episode-start masks?
7. Under generic `PPOConfig` defaults, how many optimizer steps occur per rollout?
8. Under basketball defaults, how many new transitions are collected in one rollout, and how many times can they be presented to the optimizer?
9. A training log shows total loss falling and reward rising. Give two reasons this still does not prove basketball mastery.
10. State the KL behavior difference between the generic MLX and basketball implementations in one sentence.

## Answers

1. At the mean, density is $1/(0.2\sqrt{2\pi})\approx1.995$. Yes. It is density, and its total area is still 1.
2. $r=\exp(-2.8-(-3.0))=\exp(0.2)\approx1.2214$.
3. Unclipped: $1.4\times3=4.2$. Clipped ratio: 1.2. Clipped product: 3.6. PPO selects the smaller surrogate, 3.6.
4. Unclipped: $1.4\times(-3)=-4.2$. Clipped product: $1.2\times(-3)=-3.6$. PPO selects $-4.2$ because it is smaller. Clipping does not protect movement in the harmful direction.
5. Terminal: $2-4=-2$. Timeout: $2+0.9(5)-4=2.5$.
6. The same observation can produce a different action mean under a different hidden state. Sequence replay must reconstruct the state that corresponds to the stored action, and reset masks must prevent memory from crossing episode boundaries.
7. `update_epochs × num_minibatches = 2 × 16 = 32` optimizer steps.
8. `4 × 32 = 128` new transitions. Up to five full-sequence optimizer presentations occur, unless KL early stopping ends the loop sooner.
9. The reward may omit important physical requirements or contain an exploitable shortcut; evaluation may cover only short or assisted episodes. In addition, loss and reward are not the same quantity.
10. Generic MLX PPO reports approximate KL as a metric but never stops on it, while basketball checks KL before and after each full-sequence epoch and stops further epochs when it exceeds the default target `0.02`.

## References

- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. **Proximal Policy Optimization Algorithms**. arXiv:1707.06347, 2017. <https://arxiv.org/abs/1707.06347>
- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. arXiv:1506.02438, 2015. <https://arxiv.org/abs/1506.02438>

The paper metadata and abstracts were checked against locally retrieved arXiv pages. The repository implementation statements in this chapter were checked separately against the local source files named at the beginning.
