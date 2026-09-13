# PPO：从一小批经验中谨慎学习

近端策略优化（Proximal Policy Optimization，通常简称 **PPO**）是 RLX 本地训练器使用的学习算法。PPO 不会直接告诉机器人“篮球”是什么意思。环境负责提供观测、动作、奖励和回合边界。PPO 回答的是一个范围更窄的问题：

> 根据当前策略刚刚尝试的动作，哪些动作应当稍微提高概率，哪些动作应当稍微降低概率，以及一次更新最多应该改变多大？

本章从高中阶段的代数、概率密度和平均数出发解释这个问题，并把每个概念对应到仓库中的当前实现：

- `rlx/examples/ppo_microduck_basketball.py`：用于已发布篮球 actor 的循环 PyTorch 续训器；
- `rlx/rlx/algorithms/ppo.py`：通用的前馈 MLX PPO 实现；
- `rlx/rlx/buffers/rollout_buffer.py`：MLX 实现使用的定长 rollout 缓冲区。

实现细节以代码为准。PPO 论文解释通用方法，而本仓库还加入了超时感知 bootstrap、稳健 critic 损失、log-ratio 数值裁剪，以及篮球训练器专用的 KL 提前停止规则。

## 用普通语言理解学习循环

设想一位教练每 20 毫秒必须指导一次运动员。每个控制时刻都会发生以下过程：

1. 策略接收一条观测，例如关节状态、身体朝向、运动命令和近期运动信息。
2. Actor 输出 14 个连续关节命令上的概率分布。
3. 从该分布中采样一个动作。
4. 模拟器向前推进，并返回奖励和边界标志。
5. Critic 预测执行动作之前的观测预计能获得多少折扣未来奖励。

PPO 保存一小段这样的 transition。随后，它把已保存的动作放入更新后的策略，比较：

$$
\text{旧 log 概率}
\quad\text{和}\quad
\text{新 log 概率}.
$$

如果某个动作的结果比 critic 原先预期更好，PPO 就尝试提高该动作的概率；如果结果更差，就尝试降低其概率。**近端**的含义是更新应当保守：当概率比率在有利方向上变化过大时，PPO 不允许目标函数继续获得同样的收益。

执行若干次优化后，这批 rollout 会被丢弃。系统必须使用新策略重新收集数据。因此 PPO 属于 **on-policy（同策略）** 算法：它不会无限保留来自许多旧策略的动作作为长期 replay 数据库。

本仓库中的两个实现用不同方式组织同一个思想：

| 路径 | Actor 记忆 | 更新组织方式 | 主要数值框架 |
|---|---|---|---|
| 通用 `rlx/rlx/algorithms/ppo.py` | 前馈 | 展平时间和环境维，打乱后使用 minibatch | MLX |
| 篮球 `ppo_microduck_basketball.py` | 单层 LSTM | 每个 epoch 按完整的时间×环境序列重放 | PyTorch |

这个差异不是表面上的代码风格。前馈样本可以作为独立的数据行训练，而 LSTM 在当前时刻的动作依赖先前观测产生的隐藏状态，所以必须按正确顺序重建历史。

## Actor、critic 与 advantage 的含义

PPO 训练两个预测系统。

**Actor** 负责选择动作。在连续控制中，它输出动作均值 $\mu(o)$ 和标准差 $\sigma$。篮球 actor 有 14 个动作维度。它先用隐藏维度为 256 的 LSTM 处理 61 维观测，再通过宽度依次为 512、256、128 的 MLP，把循环网络输出映射成 14 个均值。

**Critic** 预测一个标量 value：

$$
V(o_t)\approx
\mathbb E[r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots].
$$

折扣因子 $\gamma$ 让更远的奖励权重更小。当 $\gamma=0.99$ 时，下一步奖励乘以 $0.99$，下两步奖励乘以 $0.99^2$，依此类推。

Actor 并不直接使用原始奖励训练，而是使用 **advantage（优势）估计**：

$$
\widehat A_t \approx
\text{动作 }a_t\text{ 之后的结果}
-
\text{执行 }a_t\text{ 前 critic 的预期}.
$$

- $\widehat A_t>0$：采样动作优于基线，应提高其概率；
- $\widehat A_t<0$：采样动作劣于基线，应降低其概率；
- $\widehat A_t\approx0$：该样本几乎没有提供改变其概率的证据。

“正 advantage”并不等于“正奖励”。如果 critic 预计得到 5，而实际奖励相关结果只有 2，那么 advantage 可以为负。如果 critic 预计为 $-4$，而结果为 $-1$，advantage 反而可以为正。Advantage 表示与基线的比较。

虽然篮球 actor 是循环网络，但篮球 critic 是前馈网络。它读取当前经过归一化的 61 维向量并预测一个 value。Actor 和 critic 共享 actor 中冻结的观测归一化器，但篮球续训会创建一个**全新的 critic 和全新的 Adam 优化器**。这属于 actor warm start，而不是对 critic 和优化器状态的完整恢复。

## 连续动作使用概率密度

对于一个动作坐标，高斯策略的密度为

$$
p(a\mid\mu,\sigma)
=
\frac{1}{\sigma\sqrt{2\pi}}
\exp\left(
-\frac{(a-\mu)^2}{2\sigma^2}
\right).
$$

对应的 log 密度为

$$
\log p(a\mid\mu,\sigma)
=
-\frac12\left[
\frac{(a-\mu)^2}{\sigma^2}
+2\log\sigma
+\log(2\pi)
\right].
$$

对于 14 个相互独立的高斯坐标，实现会把 14 个 log 密度相加：

$$
\log\pi(a\mid o)
=
\sum_{j=1}^{14}\log p(a_j\mid\mu_j(o),\sigma_j).
$$

因此每条 transition 最终只得到一个联合 log 概率。

### 密度可以大于 1

学生常常学过“概率不能大于 1”。这对事件的概率是正确的，但连续高斯公式返回的是**密度**，不是某个精确点本身的概率。窄分布的曲线高度可以超过 1，只要整条曲线下面积仍然等于 1。

当 $\mu=0$、$\sigma=0.1$ 时，均值位置的密度为

$$
p(0)=\frac{1}{0.1\sqrt{2\pi}}\approx3.989.
$$

落入某个区间的概率是该区间下的面积。在均值附近取宽度为 $0.01$ 的很小区间，概率可近似为密度乘宽度，即 $3.989\times0.01=0.03989$，仍然小于 1。

因此 log 密度也可以为正。这完全合法。连续分布中的正 log 密度不代表概率大于 1。

下面这个完整的纯标准库程序可以复现该计算：

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

除最后一位浮点舍入外，预期输出为：

```text
log density: 1.383647
density: 3.989423
approximate probability in width 0.01: 0.039894
```

在 `ppo_microduck_basketball.py` 中，`gaussian_log_prob()` 使用 Torch 张量实现同一表达式，并沿动作维求和。`gaussian_entropy()` 同样沿动作维求和。

## 从 log 概率得到 PPO ratio

缓冲区中的动作由旧策略采样。PPO 要比较该动作在新策略下的密度发生了怎样的变化：

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid o_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)}.
$$

直接除以极小的密度在数值上不方便，因此代码使用 log 概率：

$$
r_t(\theta)
=
\exp\left(
\log\pi_\theta(a_t\mid o_t)
-
\log\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t)
\right).
$$

由于指数函数始终为正，ratio 也始终为正。

- $r=1$：已保存动作的密度没有变化；
- $r=1.2$：密度提高了 20%；
- $r=0.7$：密度降低了 30%。

Ratio 不是“这个动作的概率”，而是同一个已记录动作在新旧密度下的比值。

通用 MLX 代码在求指数之前把 **log ratio** 裁剪到 `[-20,20]`。这是防止数值溢出的保护措施，与 PPO 通常使用的 $[0.8,1.2]$ 等较窄策略裁剪区间不是同一件事。篮球训练器目前直接对未限制的 log-ratio 求指数，然后检查 KL 或损失是否出现非有限值。

## 裁剪后的策略目标

PPO 对单个样本使用的 surrogate 为

$$
L_t^{\mathrm{clip}}
=
\min\left(
r_t\widehat A_t,\;
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right).
$$

算法希望最大化该值，而实现通常最小化它的负数。通用 MLX 源码用两个负项的 `maximum` 表达；篮球源码先对两个正项取 `minimum`，再对平均值取负。两种写法在代数上等价。

当 $\epsilon=0.2$ 时，裁剪区间为 $[0.8,1.2]$。

### 正 advantage

设 $\widehat A=+2$。提高 ratio 是有利方向，因为它让好动作更可能被采样。

- 当 $r=1.1$ 时，两项都等于 $2.2$；
- 当 $r=1.3$ 时，普通乘积为 $2.6$，裁剪乘积为 $1.2\times2=2.4$，PPO 选择 $2.4$；
- 当 $r=0.7$ 时，普通乘积为 $1.4$，裁剪乘积为 $1.6$，PPO 选择较小的 $1.4$。

目标函数不再奖励过大的有利增幅，但不会抹掉有害的概率下降。

### 负 advantage

设 $\widehat A=-2$。降低 ratio 是有利方向，因为它让坏动作更不容易被采样。

- 当 $r=0.7$ 时，普通乘积为 $-1.4$，裁剪乘积为 $0.8\times(-2)=-1.6$，PPO 选择较小的 $-1.6$；
- 当 $r=1.3$ 时，普通乘积为 $-2.6$，裁剪乘积为 $-2.4$，PPO 选择 $-2.6$。

对于负 advantage，有利方向是向下，因此平台出现在 $0.8$ 以下，而不是 $1.2$ 以上。

下面这个独立程序打印正负 advantage 的结果：

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

裁剪并不保证新策略一定更好。它只改变当前采样 batch 上的训练目标。网络仍可能在本批次没有覆盖的观测上发生变化；多个参数会相互作用；critic 可能不准确；奖励也可能没有正确描述目标篮球技能。

![PPO 裁剪曲线示意图。该图用于解释概念，不是实测训练证据。](assets/diagrams/schematic-ppo-clipping.png)

## 广义优势估计

一步 temporal-difference residual 为

$$
\delta_t
=
r_t+\gamma b_t-V(o_t),
$$

其中 $b_t$ 表示该 transition 之后允许使用的 bootstrap value。广义优势估计（Generalized Advantage Estimation，GAE）按时间反向组合 residual：

$$
\widehat A_t
=
\delta_t+\gamma\lambda m_t\widehat A_{t+1}.
$$

其中：

- $\gamma$ 对未来 return 进行折扣；
- $\lambda$ 控制后续 residual 对当前估计的影响强度；
- $m_t$ 是 continuation mask，用于阻止 trace 穿过回合重置。

使用仓库默认值 $\gamma=0.99$、$\lambda=0.95$ 时，trace 乘数为

$$
\gamma\lambda=0.9405.
$$

较小的 $\lambda$ 更依赖 critic 的一步预测；较大的 $\lambda$ 会把更多采样到的未来信息向前传播，通常也带来更大的采样方差。不存在对所有任务都最佳的固定取值。

Critic 的训练目标为

$$
\widehat R_t=\widehat A_t+V(o_t).
$$

Actor 使用 $\widehat A_t$，critic 拟合 $\widehat R_t$。

## Terminal 与 time limit 使用不同 mask

Gymnasium 区分两类边界：

- **termination**：所建模的任务真正结束，例如定义中的失败条件；
- **truncation**：由于外部原因停止收集，常见情况是时间上限，但底层任务本来仍可继续。

两类边界都会导致环境 reset，但它们不应使用相同的 value bootstrap。

对 transition $t$，定义：

$$
b_t=
\begin{cases}
0,&\text{terminated},\\
V(o_{t+1}^{\mathrm{final}}),&\text{truncated 但未 terminated},\\
V(o_{t+1}),&\text{回合继续}.
\end{cases}
$$

对于 termination 或 truncation，GAE trace mask 都为零：

$$
m_t=1-\mathbf 1[\text{terminated}\lor\text{truncated}].
$$

这里存在两个不同的问题：

1. **当前 residual 能否 bootstrap 一个 value？** 纯超时可以，真正 terminal 不可以。
2. **Advantage 能否继续传播到下一条已保存 transition？** 两种边界都不可以，因为下一条观测可能来自 reset 后的新回合。

篮球实现为每条 transition 保存 `bootstrap_values`。收集过程中，它在 reset 前的 final observation 上调用 critic，然后在 `terminated` 为 true 的位置显式把 value 替换成零。它的 `compute_gae()` 使用：

- `~terminated` 作为 bootstrap mask；
- `~(terminated | truncated)` 作为 trace continuation mask。

通用 MLX 实现通过另一种方式得到相同语义。`_truncation_values()` 只选择 `truncated & ~terminated` 的环境，并计算 `info["terminal_observation"]` 的 value。共享 GAE helper 在 truncation 时使用 timeout value，在 termination 时使用零，其余情况使用普通 next value。它绝不会从 autoreset observation 进行 bootstrap。

### 边界计算示例

假设：

$$
r_t=1,\quad V(o_t)=2,\quad
V(o_{t+1}^{\mathrm{final}})=5,\quad \gamma=0.99.
$$

如果 transition 是真正 terminal：

$$
\delta_t=1+0-2=-1.
$$

如果它只是时间上限：

$$
\delta_t=1+0.99(5)-2=3.95.
$$

如果 autoreset observation 的 value 为 99，错误使用它会得到

$$
1+0.99(99)-2=97.01,
$$

这会把新回合的 value 错误分配给旧回合。

下面的程序计算三步 trace，并证明 reset 后回合中的奖励不能向前泄漏：

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

预期 advantage 为：

```text
[0.586836, -0.107565, -0.209]
```

超时 transition 从 0.9 bootstrap，但其 continuation mask 为零，因此 reset 后的新回合 advantage 不会进入该 trace。

## Advantage 归一化

不同任务或 batch 中的原始 advantage 可能具有非常不同的尺度。两个实现都会归一化 advantage：

$$
\widehat A'_t
=
\frac{\widehat A_t-\operatorname{mean}(\widehat A)}
{\operatorname{std}(\widehat A)+10^{-8}}.
$$

篮球训练器在进入 epoch 循环前，对完整 rollout 统一归一化一次。通用 MLX 训练器在选择 minibatch 后，于 `_loss_and_metrics()` 内归一化，因此每个实际 minibatch 使用自己的均值和标准差。

归一化会改变尺度，也可能改变某个样本相对于所选数据组均值的正负号。它不会修改奖励、return 或旧 log 概率。因此教学表格中的 `+2` advantage 并不意味着训练时一定仍然是 `+2`。

## Critic 损失

篮球续训使用普通的二分之一平方误差：

$$
L_V
=
\frac12\operatorname{mean}\left[
(V_{\mathrm{new}}-\widehat R)^2
\right].
$$

它不会裁剪 value 更新。

通用 MLX PPO 使用阈值为 10 的 Huber 风格误差：

$$
h(e)=
\begin{cases}
\frac12e^2,&|e|\le10,\\
10|e|-50,&|e|>10.
\end{cases}
$$

当 `clip_value_loss=True` 时，它还会构造一个裁剪后的预测：

$$
V_{\mathrm{clip}}
=
V_{\mathrm{old}}+
\operatorname{clip}(V_{\mathrm{new}}-V_{\mathrm{old}},-\epsilon,\epsilon),
$$

随后取未裁剪 Huber 误差与裁剪分支 Huber 误差中的较大值。选择较大值可以避免裁剪分支把变化过大的 value 预测伪装成较好的结果。

这是一个重要的实现差异。“RLX PPO 使用 value clipping”对 `rlx/rlx/algorithms/ppo.py` 成立，但对篮球专用的 `ppo_update()` 不成立。

## Entropy 以及它为何可以为负

Entropy 用于鼓励探索。对一个高斯坐标，

$$
H
=
\log\sigma+\frac12\log(2\pi e).
$$

对于相互独立的多个坐标，实现会把这些值相加。

这里使用的是**微分熵（differential entropy）**，不是离散 Shannon entropy。微分熵可以为负。例如当 $\sigma=0.03$ 时：

$$
H_{\text{单维}}\approx-2.087,
$$

因此 14 个相同维度的总 entropy 约为 $-29.2$。

负值并不表示出现错误。它表示连续密度集中在较窄区域内，不代表负概率。

两个损失函数都减去 entropy bonus：

$$
L_{\mathrm{total}}
=
L_{\mathrm{policy}}
+c_VL_V
-c_HH.
$$

如果 $H$ 为负，那么 $-c_HH$ 为正。优化器仍然按照微分熵的准确目标工作，不能用离散 entropy 的直觉简单判断这个符号。

篮球 actor 保存一个直接训练的标准差参数，使用时只把它下限裁剪到 `1e-6`。加载源 actor 后，代码会把该参数填充为请求的 `initial_std`，命令行默认值为 `0.03`。因此续训会有意重置探索尺度，而不是保留源 checkpoint 中保存的标准差。

## 总 loss 不是 reward，两者都不能证明掌握技能

篮球默认总损失为

$$
L
=
L_{\mathrm{policy}}
+1.0L_V
-0.01H.
$$

通用 PPO 默认使用 value 系数 `0.5` 和 entropy 系数 `0.01`，但具体示例程序可以覆盖这些值。

训练 loss 下降并不等于 episode reward 上升：

- policy loss 衡量最新 batch 上的裁剪概率目标；
- value loss 衡量 critic 预测误差；
- entropy 衡量分布的扩散程度；
- reward 由环境提供；
- mastery 是外部行为结论。

总 loss 可能在行为改善时上升，因为 critic 正在拟合幅度更大的 return target。总 loss 也可能下降，而机器人只是学会利用奖励漏洞。高 reward 仍可能表示错误持球、利用辅助物理，或者在日志时间范围之后失败。PPO 只会优化它收到的奖励和数据。

因此篮球策略需要独立评估：完整回合、物理任务指标、确定性导出策略测试，以及渲染后的人工检查。任何一个 PPO 标量指标都不能单独证明技能已经掌握。

## 篮球中的循环序列重放

篮球 actor 是 LSTM。时刻 $t$ 的动作均值依赖：

$$
\mu_t=f(o_t,h_{t-1},c_{t-1}).
$$

如果训练时打乱单个时间步，隐藏状态将不再与生成该动作时的隐藏状态一致。因此篮球 rollout 保存：

- 形状为 `[time, environments, 61]` 的观测；
- 采样动作和旧 log 概率；
- 旧 critic value、奖励、termination、truncation 和 bootstrap value；
- 每个时间步的 `episode_starts`；
- rollout 开始时的 LSTM hidden state 和 cell state。

每个 PPO epoch 中，`actor.forward_sequence()` 从保存的初始隐藏状态开始，并按时间顺序向前执行。在处理 `episode_starts` 为 true 的时间步之前，它会把对应环境的 hidden state 和 cell state 乘以零。这样既能重建循环状态，又能防止记忆跨越 reset 边界。

更新每个 epoch 使用完整序列。`ppo_update()` 中没有篮球 minibatch 循环。使用默认的 32 步和 4 个环境时，一个 rollout 包含：

$$
32\times4=128\text{ 条 transition}.
$$

五个 epoch 最多会把同样的 128 条 transition 提交给优化器五次，除非 KL 提前停止先结束循环。这仍然只收集了 128 条新环境 transition，而不是 640 条。

Critic 是前馈网络，所以代码把输入展平为 `[time*environment, 61]` 进行计算，再把输出恢复成 `[time, environment]`。

## 通用 MLX PPO 的前馈 minibatch

`RolloutBuffer.reset()` 分配形状为 `[num_steps, num_envs, ...]` 的定长数组。它保存：

| 字段 | 用途 |
|---|---|
| `observations` | 选择每个动作时使用的输入 |
| `next_observations` | 环境返回的下一个输入 |
| `actions` | 采样动作 |
| `rewards` | 标量奖励 |
| `terminations`、`truncations` | 两类独立边界 |
| `values` | 旧 critic 预测 |
| `log_probs` | 旧策略的联合 log 概率 |
| `advantages`、`returns` | 已分配字段，但当前 PPO 路径把计算结果直接作为参数传给 `update()` |

通用更新会展平前两个维度。使用库级默认值时：

$$
16\text{ 步}\times4096\text{ 个环境}
=65{,}536\text{ 个样本}.
$$

分成 16 个 minibatch 后，每个 minibatch 有 4,096 个样本。两个 epoch 会在每个 rollout 上产生 32 次优化器更新。

通用 Microduck 示例并不使用这些巨大的库级默认值。`rlx/examples/ppo_microduck.py` 定义 16 个环境、24 步、4 个 minibatch 和 5 个 epoch，再把这些值传给 `PPOConfig`。因此它的 rollout 有 384 条 transition，每个 minibatch 有 96 条，每个 rollout 产生 20 次优化器更新。

MLX 更新在每个 epoch 创建新的随机排列。对于前馈网络，这是合理的，因为每条保存的 observation 已经包含网络所需的完整输入。`RolloutBuffer` 不保存循环 hidden state，也不保存 sequence-start mask；如果不加修改就用于 LSTM，会丢失必要历史。

## KL divergence：MLX 中是诊断，篮球中是停止规则

两个实现都计算

$$
\widehat D_{\mathrm{KL}}
=
\operatorname{mean}\left[(r-1)-\log r\right].
$$

因为样本来自旧策略，这是一种常用的样本近似，与
$D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_{\mathrm{new}})$ 相关。其期望应当非负，但有限样本和浮点运算可能产生很小的不规则值。

两个实现对它的用途不同。

### 通用 MLX 行为

`rlx/rlx/algorithms/ppo.py`：

- 求指数前把 log ratio 裁剪到 `[-20,20]`；
- 请求 metrics 时报告 `approximate_kl`；
- 报告 `clip_fraction`；
- 不定义 `target_kl`；
- 不根据 KL 停止 epoch 或 minibatch。

这里的 KL 是诊断指标，不是训练门控。

### 篮球行为

`ppo_microduck_basketball.py`：

- `target_kl` 默认值为 `0.02`；
- 在每个 epoch 的优化器更新前计算 KL；
- 如果保存的旧 log 概率已经与当前 actor 不一致，可以在 `epochs_completed == 0` 时停止；
- 每次优化器更新后重新计算完整循环序列；
- 计算更新后的 KL，并在超过目标时停止后续 epoch；
- 记录 `kl_early_stop`。

更新前的检查可以发现陈旧或不匹配的 rollout 数据。更新后的检查限制同一小批循环 rollout 被重复使用的次数。即使如此，它仍不保证每个状态上的策略变化都小于某个严格上限。

## 准确区分默认值与覆盖值

描述“PPO 默认值”时，必须说明具体是哪一层。

### 通用 `PPOConfig` 默认值

| 设置 | 值 |
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

### 通用 Microduck 示例覆盖值

`rlx/examples/ppo_microduck.py` 把 `num_envs`、`num_steps`、`num_minibatches` 和 `update_epochs` 覆盖为 `16`、`24`、`4`、`5`。它保留 gamma `0.99`、lambda `0.95`、clip `0.2` 和 entropy 系数 `0.01`。优化器学习率默认值为 `1e-3`，它属于示例参数，不属于 `PPOConfig`。

### 篮球训练器默认值

篮球训练器不会实例化 `PPOConfig`。其命令行与函数默认值共同得到：

| 设置 | 值 |
|---|---:|
| 环境数 | 4 |
| rollout 步数 | 32 |
| update 次数 | 5 |
| 每个 rollout 的 PPO epoch | 5 |
| gamma | 0.99 |
| GAE lambda | 0.95 |
| clip 系数 | 0.2 |
| value 系数 | 1.0 |
| entropy 系数 | 0.01 |
| 梯度范数上限 | 1.0 |
| Adam 学习率 | `2e-5` |
| target KL | 0.02 |
| 重置后的探索标准差 | 0.03 |

环境构造还显式覆盖了训练条件：

- 回合上限：10 秒；
- actuator：默认 `"bam"`；
- observation noise：关闭；
- domain randomization：关闭；
- pushes：关闭；
- curriculum：关闭；
- hold assistance：`0.0`；
- command：默认固定为 `(0.08, 0.0, 0.0)`，除非选择 `--randomized-commands`。

这些选择描述的是默认本地续训实验，不是通用 PPO 建议，也不等于完整复现上游训练配方。

## 一次完整更新的全过程

对于默认篮球运行：

1. 使用确定性的 seed 偏移重置四个环境。
2. 创建形状为 `[1,4,256]` 的 LSTM hidden tensor 和 cell tensor。
3. 收集 32 个控制步，共得到 128 条 transition。
4. 在每一步，对当前 observation 属于新回合的环境清零循环状态。
5. 采样 14 维高斯动作，并保存其联合旧 log 概率。
6. 分别推进每个环境。
7. 在 reset 前的 final observation 上计算 bootstrap value，并把 terminal 位置强制设为零。
8. 重置已结束环境，并把下一条 observation 标记为 episode start。
9. 使用 termination 和 truncation mask 反向计算 GAE。
10. 对 128 个 advantage 统一归一化。
11. 从已保存的初始 LSTM 状态重放整个 actor 序列。
12. 检查更新前的 approximate KL。
13. 计算裁剪 policy loss、平方 critic loss 和高斯 entropy。
14. 反向传播，把 actor 与 critic 的联合梯度范数裁剪到 1.0，再执行 Adam 更新。
15. 再次重放序列，计算更新后的 KL，并判断是否允许进入下一 epoch。

命令行默认请求五次这样的 rollout/update 循环：

$$
4\times32\times5=640
\text{ 条本地环境 transition}.
$$

这个很小的默认规模适合短续训或 smoke-scale 运行，不能作为篮球技能已经掌握的证据。

## 常见解释错误

**错误：“高斯返回了 3.9 的概率，不可能。”**  
纠正：返回的是密度。概率是区间下的面积。

**错误：“正 log 概率无效。”**  
纠正：连续 log 密度可以为正，因为密度可以大于 1。

**错误：“裁剪会把所有 ratio 都限制在 0.8 到 1.2。”**  
纠正：目标函数使用裁剪后的比较项，实际 ratio 仍可位于区间外。

**错误：“环境 reset 了，所以 timeout 的未来 value 应当是零。”**  
纠正：纯时间上限从 reset 前的 final observation bootstrap，然后停止 trace。

**错误：“负 entropy 表示负概率。”**  
纠正：连续分布的微分熵可以为负。

**错误：“总 loss 更低就表示 reward 更高。”**  
纠正：总 loss 混合了一个 batch 上的 policy、critic 和 entropy 项。

**错误：“reward 高就表示技能已经掌握。”**  
纠正：reward 是人为设计的训练信号，mastery 需要独立物理标准和可视检查。

**错误：“篮球 updater 使用 MLX minibatch。”**  
纠正：它是独立的 PyTorch 循环 updater，每次重放完整序列。

**错误：“两个 PPO 实现都会在 KL 0.02 时停止。”**  
纠正：只有篮球路径具有该目标和提前停止；通用 MLX PPO 只报告 KL。

## 练习

1. 一个高斯分布的均值为 0、标准差为 0.2。计算均值位置的密度。结果可以大于 1 吗？
2. 某动作的旧 log 密度为 `-3.0`，新 log 密度为 `-2.8`。计算 PPO ratio。
3. Advantage 为 `+3`、ratio 为 `1.4`、clip 系数为 `0.2`。计算未裁剪乘积、裁剪乘积和最终选中的 surrogate。
4. 把第 3 题的 advantage 改为 `-3`，重新计算。
5. 某 transition 的 reward 为 2，当前 value 为 4，final-observation value 为 5，gamma 为 0.9。分别计算真正 terminal 与纯 timeout 的 TD residual。
6. 为什么 LSTM rollout 必须保存初始 hidden state 和 episode-start mask？
7. 使用通用 `PPOConfig` 默认值时，每个 rollout 会产生多少次优化器更新？
8. 使用篮球默认值时，每个 rollout 收集多少条新 transition？这些数据最多会提交给优化器多少次？
9. 训练日志显示总 loss 下降、reward 上升。给出两个仍然不能证明篮球 mastery 的理由。
10. 用一句话说明通用 MLX 与篮球实现的 KL 行为差异。

## 答案

1. 均值位置密度为 $1/(0.2\sqrt{2\pi})\approx1.995$。可以大于 1，因为它是密度，总面积仍然为 1。
2. $r=\exp(-2.8-(-3.0))=\exp(0.2)\approx1.2214$。
3. 未裁剪乘积为 $1.4\times3=4.2$；裁剪 ratio 为 1.2；裁剪乘积为 3.6；PPO 选择较小的 3.6。
4. 未裁剪乘积为 $1.4\times(-3)=-4.2$；裁剪乘积为 $1.2\times(-3)=-3.6$；PPO 选择更小的 $-4.2$。裁剪不会保护有害方向上的变化。
5. Terminal：$2-4=-2$。Timeout：$2+0.9(5)-4=2.5$。
6. 同一个 observation 在不同 hidden state 下可以产生不同动作均值。序列重放必须重建与已保存动作对应的状态，reset mask 必须阻止记忆跨越回合边界。
7. `update_epochs × num_minibatches = 2 × 16 = 32` 次优化器更新。
8. `4 × 32 = 128` 条新 transition。最多执行五次完整序列优化，但 KL 提前停止可能更早结束。
9. Reward 可能漏掉重要物理要求或存在可利用的捷径；评估也可能只覆盖短回合或带辅助的回合。此外，loss 与 reward 本来就不是同一个量。
10. 通用 MLX PPO 把 approximate KL 作为指标报告但不会据此停止，而篮球实现会在每个完整序列 epoch 前后检查 KL，并在超过默认目标 `0.02` 时停止后续 epoch。

## 参考文献

- John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford、Oleg Klimov，**Proximal Policy Optimization Algorithms**，arXiv:1707.06347，2017。<https://arxiv.org/abs/1707.06347>
- John Schulman、Philipp Moritz、Sergey Levine、Michael Jordan、Pieter Abbeel，**High-Dimensional Continuous Control Using Generalized Advantage Estimation**，arXiv:1506.02438，2015。<https://arxiv.org/abs/1506.02438>

论文元数据和摘要已经通过本地保存的 arXiv 页面核对。本章关于仓库实现的陈述则单独来自对开头所列本地源码的核对。
