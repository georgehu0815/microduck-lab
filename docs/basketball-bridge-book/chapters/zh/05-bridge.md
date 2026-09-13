# 悬索桥：在会移动的支撑面上学习行走

## 学习目标

学完本章后，你应当能够：

1. 用准确尺寸说明仿真桥面的自由木板、四根肌腱缆索、两端平台和无碰撞门架；
2. 区分“仿真物料表”和“经过认证的实体建造方案”；
3. 追踪不变的 61 维观测、14 维动作部署契约；
4. 解释为什么导航指挥器可以读取世界坐标，而学习到的 actor 仍只接收标准命令槽；
5. 推导继承自行走环境的全部奖励公式及真实权重；
6. 解释课程学习为何只改变物理辅助和出生位置，而不改变奖励；
7. 说明行走策略迁移过程：迁移 actor 与归一化器、冻结观测统计量、重新初始化 critic 和优化器；
8. 使用当前解析器支持的准确命令完成场景构建检查、训练、导出、评估和渲染；以及
9. 按严格的完整 20 秒规则判断过桥，不能把“走了一部分”写成“成功过桥”。

先给出本章最重要的结论：

> 已记录的 `bridge-studio-02` 试验进行了 32,768 个 PPO transition，但**没有**穿越悬索桥。它从起点算起的最佳进度为 1.175 m，低于 1.60 m 的进度门槛，也没有出现经过接触验证的双脚终点落地。本章记录的是可运行的仿真与训练协议，不是已经掌握的技能。

## 1. 证据范围与结论边界

本章以当前仓库中的以下实现为准：

| 问题 | 权威来源 |
| --- | --- |
| 桥梁几何、指挥器、课程、接触与指标 | `rlx/rlx/environments/bridge.py` |
| 严格过桥判据 | `rlx/rlx/environments/bridge_evaluation.py` |
| 行走 actor 迁移 | `rlx/rlx/models/bridge_bootstrap.py` |
| Studio 的 train/eval/render/export 参数解析 | `rlx/examples/ppo_microduck_studio.py` |
| 共享观测与动作顺序 | `microduck_local/src/microduck_local/contract.py` |
| 继承的奖励公式 | `microduck_local/src/microduck_local/walk_env.py` |
| 已记录试验与确定性评估 | `docs/bridge-showcase/` 与 `rlx/runs/studio/bridge/bridge-studio-02/` |

本地训练手册还规定了三条解释规则：

- 不能为了某个任务改变 61/14 接口；
- 不能把奖励值或 PPO 跑完当作行为成功的证据；
- 声称技能成功前，必须检查导出 ONNX 的确定性 rollout。

桥梁模型是一个**本地 MuJoCo 原型**。尺寸、质量、摩擦系数、肌腱刚度、阻尼和长度限制都是仿真参数。它们不是实体桥梁的测量结果，也不是缆索额定载荷、结构计算、安全系数或硬件认证。`bridge_plank_mat` 等材质名称只定义渲染颜色和外观，并没有指定木材树种、金属牌号或制造工艺。

## 2. 仿真中真正存在什么？

任务把 Microduck 放在抬高的起点平台上。一块窄板跨过间隙，连接到终点平台。窄板不是焊死的、铰接的，也不是用动画脚本移动的。它带有 MuJoCo 自由关节，因此接触力能够让它平移，并产生滚转、俯仰和偏航。四条空间肌腱把木板上的移动连接点连到世界坐标中的固定锚点。

### 2.1 仿真物料表

下表是仿真对象清单，不是采购清单或施工图。

| 数量 | 仿真对象 | 准确模型参数 |
| ---: | --- | --- |
| 1 | 自由木板 | `1.10 x 0.13 x 0.03 m`，质量 `0.45 kg`，初始中心 `(0,0,0.225) m` |
| 2 | 固定平台 | 每个 `0.60 x 0.48 x 0.24 m`，中心 `x=-0.85 m` 与 `x=+0.85 m` |
| 4 | 空间肌腱缆索 | 刚度 `320`，阻尼 `2.0`，渲染宽度 `0.0015 m`，带长度限制 |
| 4 | 固定锚点 | `(x,y,z)=(+/-0.45,+/-0.17,0.77) m` |
| 4 | 木板连接点 | 木板局部坐标 `(x,y,z)=(+/-0.45,+/-0.057,0) m` |
| 2 | 门架横梁 | `0.028 x 0.48 x 0.028 m`，中心位于 `x=+/-0.45 m`、`z=0.784 m` |
| 4 | 门架立柱 | `0.028 x 0.028 x 0.77 m`，位于 `x=+/-0.45 m`、`y=+/-0.225 m` |
| 1 | 地面平面 | 世界坐标顶部 `z=0`，低于平台顶面 |

MuJoCo 的 box `size` 使用半尺寸。例如木板源码中的
`(0.55,0.065,0.015)` 是半长、半宽和半厚，因此完整尺寸正好是表中的
`1.10 x 0.13 x 0.03 m`。

起点平台覆盖 `x=-1.15` 到 `-0.55 m`；木板覆盖 `x=-0.55` 到
`+0.55 m`；终点平台覆盖 `x=+0.55` 到 `+1.15 m`。在标称模型中，它们的边界恰好相接。三个表面的顶面都位于 `z=0.24 m`，所以木板中心高度为

$$
0.24-0.015=0.225\ \text{m}.
$$

### 2.2 四根缆索

在 `x=-0.45 m` 和 `x=+0.45 m` 两个纵向位置，各有左、右两根缆索。左侧锚点与连接点的横向距离为

$$
+\!0.17-0.057=0.113\ \text{m},
$$

右侧的距离大小相同。竖直距离为

$$
0.77-0.225=0.545\ \text{m}.
$$

因此标称直线长度为

$$
L=\sqrt{0.113^2+0.545^2}
  =0.556591412079\ \text{m}.
$$

源码把弹簧长度上界设为 `L-0.0015`，约为 `0.555091 m`；把受限最大长度设为 `L+0.008`，约为 `0.564591 m`。这些肌腱是 MuJoCo 中无质量、带弹性和阻尼参数的约束。代码没有模拟钢丝股、绳结、连接件、疲劳、断裂强度或锚点拔出。

若只做简化的线性弹簧估算，初始伸长量是 `0.0015 m`，对应

$$
F=k\Delta L=320(0.0015)=0.48
$$

个仿真力单位，尚未考虑阻尼和约束求解。这只是教学估算，不是实体缆索载荷等级。

### 2.3 门架可见，但不能接住机器人

横梁和立柱都使用 `contype=0`、`conaffinity=0`。MuJoCo 会把它们画出来，但碰撞系统不允许机器人站在门架上、靠在门架上或借门架获得支撑。四条 tendon 约束属于物理仿真；矩形门架几何只是无碰撞的视觉背景。

下面是经过删节的核心源码片段。它省略了外层循环，不能作为独立程序直接运行：

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

地面摩擦参数为 `(1.0,0.005,0.0001)`，平台为
`(1.1,0.005,0.0001)`，木板为 `(1.15,0.006,0.0002)`。它们是本地训练选择的参数，不是实体材料测试结果。

## 3. 接触、运动与失败条件

仿真物理步长为 `0.005 s`，每个策略动作保持四个物理子步：

$$
\Delta t_{\text{control}}=4(0.005)=0.02\ \text{s}.
$$

因此 actor 的控制频率是 50 Hz。默认 20 秒 episode 包含 1,000 个控制步。

动作向量有 14 个值，顺序与共享关节契约一致。动作表示关节目标偏移：

$$
\mathbf q^{\text{target}}
=\mathbf q^{\text{default}}+\operatorname{clip}(\mathbf a,-4,4).
$$

木板只通过仿真动力学运动，包括重力、肌腱力、接触力和可选课程辅助。任何控制步都不会把木板或机器人向前瞬移。`reset()` 会按照课程阶段把机器人放到声明过的出生点，并给木板很小的随机滚转、俯仰和角速度。

出现以下任一条件时，桥梁 episode 会终止：

- 任意机器人部位接触地面；
- 躯干倾斜超过 70 度；
- 躯干高度低于 `0.10 m`；
- 躯干横向位置超过 `|y|>0.36 m`，并且双脚都没有有效支撑；
- 物理状态出现非有限数值。

达到普通时间上限属于 truncation，不等于跌倒。有效脚部支撑可以来自起点平台、木板或终点平台。如果手、头、躯干或其他非脚部位接触这些支撑面，系统会记录 `nonfoot_support`。

## 4. 观测契约：特权指挥器与普通 actor

### 4.1 Actor 仍然只接收 61 个值

桥梁 actor 没有专门的桥面状态输入：

| 切片 | 宽度 | 含义 |
| --- | ---: | --- |
| `0:3` | 3 | 躯干角速度 |
| `3:6` | 3 | 投影到躯干坐标系的重力 |
| `6:20` | 14 | 相对 `DEFAULT_POSE` 的关节位置 |
| `20:34` | 14 | 延迟一个控制步的关节速度 |
| `34:48` | 14 | 上一个原始 actor 动作 |
| `48:51` | 3 | 机体坐标系前进、横移和偏航角速度命令 |
| `51:55` | 4 | 头部命令，Bridge 中为零 |
| `55:61` | 6 | 身体命令，Bridge 中为零 |

桥面位置、桥面速度、缆索长度、木板滚转/俯仰、世界位置、接触标签、过桥状态和辅助等级都不在观测中。测试会在机器人不动时把木板横向移动 `0.03 m`，然后确认 61 维观测逐字节不变。

Actor 和 critic 是两个独立的 `61 -> 512 -> 256 -> 128` ELU 网络，但二者接收相同的归一化 61 维向量。本实现**没有**给 critic 增加非对称的特权观测。特权仿真量只用于奖励计算、课程决策、日志和评估。

### 4.2 为什么指挥器是特权模块

导航指挥器运行在学习到的 actor 外部。它读取躯干世界坐标 `x`、世界横向偏移 `y` 和世界航向，再把结果写入标准的三个 `twist_cmd` 槽。Actor 看到的是转换后的命令，而不是生成命令所用的世界测量值。

设水平前向单位向量为

$$
\mathbf h=(h_x,h_y,0),
$$

机体左向为

$$
\mathbf s=(-h_y,h_x,0).
$$

到达终点停止区域之前，期望世界速度是

$$
\mathbf d=
\left(
0.25,\
\operatorname{clip}(-1.2y,-0.12,0.12),\
0
\right).
$$

航向误差为

$$
\psi=\operatorname{atan2}(h_y,h_x).
$$

写入观测的机体坐标命令为

$$
\mathbf c=
\left(
\mathbf d\cdot\mathbf h,\
\mathbf d\cdot\mathbf s,\
\operatorname{clip}(-1.5\psi,-0.6,0.6)
\right).
$$

实现代码为：

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

这个指挥器不是物理辅助：它不施加力，也不写关节动作。但它属于特权协议逻辑，因为它使用了 61 维观测中不存在的世界坐标定位。完整系统的能力声明必须把指挥器算进去，不能把全部导航能力都归功于 actor。

当躯干 `x>=0.80 m` 时，当前指挥器把期望世界前进速度设为零，同时保留横向和偏航修正。这个“终点停止”行为区分了最终的 `bridge-studio-02` 协议和更早的 `bridge-studio-01`；后者早于该停止逻辑。两个试验都是独立地从行走策略热启动并训练 32,768 个 transition，不能合并成一个 65,536 transition 的连续训练。

## 5. 奖励：继承行走奖励，没有过桥大奖

`BridgeEnv` 没有覆盖 `_compute_reward`。桥梁 recipe 的可覆盖奖励键集合也是空集。因此所有课程阶段都使用完全相同的基础行走奖励：

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

奖励中没有世界 `x`、桥梁进度、木板角度、缆索张力、终点平台接触或 `bridge_crossed`。

记：

- $\mathbf c=(c_x,c_y,c_\omega)$ 为 `twist_cmd`；
- $\mathbf v_b=(v_x,v_y,v_z)$ 为躯干在机体坐标系中的真实线速度；
- $\boldsymbol\omega=(\omega_x,\omega_y,\omega_z)$ 为真实躯干角速度；
- $\mathbf g_b=(g_x,g_y,g_z)$ 为投影重力；
- $\delta q_j=q_j-q_j^{\text{default}}$；
- $h_j$ 为头部命令，在 Bridge 中为零。

真实实现的奖励项和权重如下。

### 线速度跟踪

$$
e_v=(c_x-v_x)^2+(c_y-v_y)^2+v_z^2,
$$

$$
r_{\text{lin}}=2\exp(-e_v/0.1).
$$

### 角速度跟踪

$$
e_\omega=(c_\omega-\omega_z)^2+\omega_x^2+\omega_y^2,
$$

$$
r_{\text{ang}}=2\exp(-e_\omega/0.5).
$$

### 直立

$$
r_{\text{upright}}
=2\exp[-(g_x^2+g_y^2)/0.05].
$$

### 默认腿部姿态

对十个腿部关节：

$$
r_{\text{pose}}
=\exp\left[
-\frac{\sum_{j\in\text{legs}}\delta q_j^2}{0.5}
\right].
$$

### 头部姿态

对四个头部关节：

$$
r_{\text{head}}
=2\left[
\frac14\sum_{j\in\text{head}}
\exp\left(-\left(\frac{\delta q_j-h_j}{0.5}\right)^2\right)
\right].
$$

### 稠密腾空时间

每只没有支撑的脚每个控制步累加 `0.02 s` 腾空时间。如果运动命令幅值大于 `0.01`，且

$$
0.125<\tau_f<0.300\ \text{s},
$$

该脚在当前步贡献 1。因此

$$
r_{\text{air}}
=3\sum_{f\in\{L,R\}}
\mathbb 1(0.125<\tau_f<0.300).
$$

Bridge 覆盖了脚部接触检测，因此起点平台、木板和终点平台都属于该继承奖励项中的有效支撑。

### 动作变化惩罚

$$
r_{\Delta a}
=-w_{\Delta a}
\sum_{j=1}^{14}(a_{t,j}-a_{t-1,j})^2.
$$

权重按照每个环境实例的累计控制步数变化：

```text
环境步数:  0     12,000  18,000  24,000  30,000  36,000
权重:      0.1   0.2     0.4     0.6     0.8     1.0
```

已记录试验共有四个环境和 32,768 个总 transition，即每个环境 8,192 个控制步，所以整个试验都处于 `0.1` 权重阶段。

### 滚转/俯仰角速度惩罚

$$
r_{\omega xy}
=-0.05(\omega_x^2+\omega_y^2).
$$

两个 penalty 项按构造都不大于零。

### 数值例题

假设某一步的状态为：

```text
命令                 = (0.25, 0, 0)
机体速度             = (0.15, 0, 0.05) m/s
角速度               = (0.10, 0.05, 0.10) rad/s
投影重力             = (0.10, 0, 约 -0.995)
腿部误差平方和       = 0.20
头部误差             = (0.10, -0.10, 0, 0.20) rad
一只脚腾空时间       = 0.20 s，另一只脚有支撑
动作变化平方和       = 0.08
动作变化惩罚权重     = 0.1
```

则：

| 奖励项 | 数值 |
| --- | ---: |
| `track_lin_vel` | `1.764994` |
| `track_ang_vel` | `1.911995` |
| `upright` | `1.637462` |
| `pose` | `0.670320` |
| `head_pose` | `1.886861` |
| `feet_air_time` | `3.000000` |
| `action_rate_penalty` | `-0.008000` |
| `ang_vel_xy_penalty` | `-0.000625` |
| **总和** | **`10.863007`** |

这个奖励很高，但仍不能证明过桥。机器人可能在带辅助的早期出生点上获得它，也可能在到达终点前获得它。

## 6. 只改变物理的课程学习

课程共有五个阶段：

| 阶段 | 辅助 $\alpha$ | 机器人出生 `x` | 含义 |
| ---: | ---: | ---: | --- |
| 0 | `1.0` | `0.00 m` | 从木板中心附近开始，木板稳定辅助最大 |
| 1 | `0.6` | `-0.30 m` | 更长的木板路径，辅助减弱 |
| 2 | `0.3` | `-0.55 m` | 从木板与起点平台边界开始 |
| 3 | `0.1` | `-0.75 m` | 从起点平台内部开始 |
| 4 | `0.0` | `-0.90 m` | 与无辅助评估完全相同的出生点 |

辅助只作用于木板。设木板位移
$\Delta\mathbf p=\mathbf p-\mathbf p_0$，线速度为 $\mathbf v$：

$$
\mathbf F_{xy}
=\alpha(-45\Delta\mathbf p_{xy}-4\mathbf v_{xy}),
$$

$$
F_z=\alpha(-25\Delta p_z-2.5v_z).
$$

设木板向上轴为 $\mathbf z_p$，世界向上轴为 $\mathbf z_w$，世界角速度为
$\boldsymbol\omega_w$：

$$
\boldsymbol\tau
=\alpha\left[
3.5(\mathbf z_p\times\mathbf z_w)
-0.18\boldsymbol\omega_w
\right].
$$

当 `alpha=0` 时，外加力数组严格为零。课程不会推着机器人向前，不会修改动作，不会修改奖励系数，也不会直接给予过桥资格。

每个 episode 结束后：

```python
if crossed:
    stage = min(stage + 1, 4)
elif elapsed_s < 2.0:
    stage = max(stage - 1, 0)
# 其他情况保持原阶段。
```

这符合训练手册中的探索原则：如果目标状态从未出现，应修改物理或出生条件，让它能够被采样，而不是增加稀疏大奖。最容易的阶段曾在确定性 `alpha_walking.onnx` 诊断中出现一次接触有效的过桥，但该尝试随后跌倒，因此它只是探索证据，不是严格技能通过。

## 7. 从已发布行走 actor 迁移

当 Bridge 的完整训练没有指定 `--init-from` 且
`total_timesteps>4` 时，训练器会自动从
`microduck/policies/alpha_walking.onnx` 创建初始化 checkpoint。

Bootstrap 会验证：

- 输入名为 `obs`，形状 `(1,61)`；
- 输出名为 `actions`，形状 `(1,14)`；
- ONNX 节点序列完全匹配；
- tensor 名称、形状和 `float32` 类型匹配；
- 归一化均值有限、标准差为正；
- 迁移期间源策略哈希没有变化。

随后创建新的 MLX actor-critic。四个 actor 线性层从 ONNX actor 精确复制；观测均值与方差从 ONNX 归一化器复制；critic 按种子重新初始化；`actor_log_std` 根据 `initial_std=0.03` 新建；训练调用方创建全新的 Adam 优化器。

Actor **没有被冻结**。它只是热启动，之后仍会参与 PPO 更新。被冻结的是观测归一化统计量，这样迁移后的 actor 继续使用原行走权重熟悉的输入坐标系。

保存前，代码用 32 个代表性观测比较 MLX actor 和源 ONNX。最大允许动作绝对误差为 `1e-4`。`bridge-studio-02` 初始化器记录的实际最大误差为 `1.0728836e-6`。

准确的迁移总结是：

```text
actor 均值网络权重:    迁移后继续训练
观测归一化器:          迁移并冻结
critic 权重:           全新、按种子初始化
actor 探索标准差:      全新，初值 0.03
优化器状态:            全新的 Adam
```

这不是从上游行走 PPO 训练中恢复完整优化器状态。

## 8. 复现实验流程

以下命令都从工作区根目录运行。`scripts/train-bridge.sh` 默认使用已有的
`rlx/.venv-microduck` 环境。

### 8.1 构建并检查仿真场景

Studio 解析器没有 `build` 子命令。MuJoCo 模型由 `bridge_model()` 在运行时构建。下面的已检查命令创建一个环境并验证公开契约：

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

预期输出：

```text
(61,) (14,) 4
0.225 0.0
```

### 8.2 训练有界试验 recipe

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

训练会自动写入 `bridge.safetensors`、JSON sidecar、`bridge.onnx`、初始 snapshot 和 `training-metrics.jsonl`。是否额外生成周期 snapshot 由
`--checkpoint-interval` 决定。

Batch 数量可以准确计算：

$$
4\text{ 个环境}\times128\text{ 步}
=512\text{ 个 transition/rollout}.
$$

$$
32768/512=64\text{ 个 rollout}.
$$

每个 rollout 有四个 minibatch，并重复两个 epoch：

$$
64\times4\times2
=512\text{ 个优化器 minibatch 更新}.
$$

### 8.3 单独导出

训练默认会导出 ONNX。若只想重新生成确定性 ONNX actor：

```bash
./scripts/train-bridge.sh export \
  --checkpoint rlx/runs/studio/bridge/my-bridge/bridge.safetensors \
  --output rlx/runs/studio/bridge/my-bridge/bridge.onnx
```

导出文件包含观测归一化器和 actor 均值网络，不包含 critic，也不包含随机动作采样。

### 8.4 运行严格技能评估

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

Bridge 技能评估拒绝 `fork` backend，因为它需要逐控制步物理指标。应使用
`dummy` 或 `subproc`。Skill 模式还要求完整 episode 时长的整数倍，并且每个 20 秒 episode 至少 1,000 步。如果技能失败，即使推理管线全程有限，命令也会以状态码 2 退出。

### 8.5 渲染并观察

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

渲染会强制关闭域随机化、观测噪声、动作延迟、随机偏航和桥面辅助。如果机器人在 20 秒结束前跌倒，渲染器会 reset 后继续录制；画面叠加层会显示尝试编号和 reset 次数。因此一个 20 秒视频可能包含多次失败尝试，而不是一个完整 episode。

## 9. 严格的完整 20 秒过桥规则

只有当某一个控制步同时满足以下条件时，环境级
`crossing_accepted` 才会变为真并保持为真：

1. 躯干 `x>=END_PLATFORM_X-0.08=0.77 m`；
2. 双脚都接触终点平台；
3. 两只脚都没有接触地面；
4. 机器人任何部位都没有接触地面；
5. 没有非脚部位为机器人提供支撑；
6. 之前至少观察到一次脚与悬空木板的接触。

独立的 `BridgeEvaluation` 还要求：

- episode 在进度接近零的位置观察到起点平台脚部支撑；
- 最大进度至少 `1.60 m`；
- 每一个被测控制步的辅助都为零；
- 全程没有地面接触或非脚支撑；
- 观察到经过接触验证的 crossing flag；
- episode 一直运行到 1,000 步 truncation，中途不能 termination；
- 每个评估 lane、每个 episode 都通过。

进度从最终出生点计算：

$$
p=x_{\text{trunk}}-(-0.90)
=x_{\text{trunk}}+0.90.
$$

因此 `1.60 m` 进度门槛本身只对应 `x_trunk>=0.70 m`。它故意不是充分条件：更严格的 crossing flag 仍要求 `x>=0.77 m`、双脚终点平台接触、之前接触木板并且支撑历史干净。

提前过桥但之后跌倒仍然失败，因为没有完成要求的 20 秒时长。直接把机器人放到终点平台会因为缺少木板接触历史而失败。沿地面走到很大的 `x` 坐标也会失败。

## 10. 当前试验实际完成了什么？

最新记录的最终试验是
`rlx/runs/studio/bridge/bridge-studio-02/`，生成于 2026 年 9 月 10 日。Checkpoint 中记录：

| 项目 | 数值 |
| --- | ---: |
| 新 PPO transition | `32,768` |
| 环境数 | `4` |
| Rollout 长度 | `128` |
| Minibatch 数 | `4` |
| Update epoch | `2` |
| 种子 | `7` |
| 执行器 | XML |
| 观测归一化器 | 迁移并冻结 |
| 奖励归一化 | 关闭 |
| 课程 | 只改变物理与出生点 |

确定性技能评估使用三个 lane，种子分别为 101、102、103；每个 lane 运行
1,000 个控制步；关闭随机化、噪声、延迟和随机偏航；使用 XML 执行器、零辅助和带终点停止的导航指挥器。

报告中有 11 次尝试，因为某些 lane termination 后会 reset，但整个
3,000 transition 评估继续运行。没有一次尝试完成完整 20 秒，也没有一次过桥。八次尝试接触了木板。最佳尝试从起点前进
`1.174915875 m`，距离评估进度门槛还差

$$
1.60-1.174915875
=0.425084125\ \text{m}.
$$

没有任何尝试记录到脚站上终点平台。有一次尝试接触了地面。推理管线保持有限，但技能状态为 `failed`。

正确总结是：

> 该试验学习到或保留了部分接近木板和在木板上行走的行为，但没有观察到成功的无辅助完整过桥。

不能说 32,768 步“完成了课程”，不能用高 return 证明过桥，也不能说保留的视频是一段连续的 20 秒成功过桥。视频明确包含 reset 和不完整尝试。

## 11. 练习

### 练习 1：缆索几何

使用锚点 `(0.45,0.17,0.77)` 和标称连接点
`(0.45,0.057,0.225)`，计算缆索长度。

### 练习 2：指挥器修正

假设躯干朝向世界 `+x`，位于 `y=0.08 m`，尚未到达终点。指挥器会发布什么命令？

### 练习 3：观测中的特权信息

木板滚转 5 度，但机器人状态和命令完全不变。61 维观测中哪些索引会因为木板滚转而改变？

### 练习 4：课程阶段转移

某环境当前处于阶段 3。分别判断以下情况后的下一阶段：

1. 8 秒时干净过桥；
2. 1.5 秒时失败；
3. 6 秒时失败。

### 练习 5：PPO 数量计算

总 transition 为 32,768，环境数 4，rollout 步数 128，minibatch 数 4，
epoch 数 2。计算 rollout 数、每个 minibatch 的 transition 数和优化器 minibatch 更新总数。

### 练习 6：诚实的过桥判断

判断下列情况通过还是失败：

1. 进度 1.65 m，第 700 步 crossing flag 变真，无非法接触，第 1,000 步 truncation；
2. 同样的 crossing，但第 850 步跌倒；
3. 双脚在终点平台，但之前没有接触木板；
4. 完整运行到时限，crossing flag 为真，但最大进度只有 1.59 m；
5. 完美运行 20 秒，但其中一个控制步的辅助为 `0.1`。

### 练习 7：解释结果

策略的平均 return 高于初始化策略，但 `bridge_crossed.max=0`。可以声称什么？

## 12. 参考答案

### 答案 1

纵向差为零，横向差为 `0.113 m`，竖直差为 `0.545 m`：

$$
L=\sqrt{0^2+0.113^2+0.545^2}
=0.556591412079\ \text{m}.
$$

### 答案 2

航向为 `(1,0,0)`，侧向为 `(0,1,0)`，并且 `y=0.08`：

$$
d_y=\operatorname{clip}(-1.2(0.08),-0.12,0.12)
=-0.096.
$$

偏航误差为零，所以命令是：

```text
(0.25, -0.096, 0.0)
```

### 答案 3

没有任何索引会直接变化。61 维向量中没有木板状态块。接触造成的后续机器人运动可能改变 IMU、关节或上一动作，但孤立的木板姿态属于特权信息。

### 答案 4

1. 过桥：进入阶段 4；
2. 2 秒前失败：退回阶段 2；
3. 较晚失败：保持阶段 3。

### 答案 5

每个 rollout 有 `4*128=512` 个 transition，共有
`32768/512=64` 个 rollout。每个 minibatch 有 `512/4=128` 个
transition。每个 rollout 的两个 epoch 共执行 `4*2=8` 次优化器更新，合计
`64*8=512` 次。

### 答案 6

1. 通过，前提是起点支撑和指标覆盖也有效；
2. 失败：在要求时长结束前 termination；
3. 失败：必须有悬空木板脚部接触历史；
4. 失败：进度低于 `1.60 m`；
5. 失败：任何一个带辅助的控制步都会使无辅助评估失效。

### 答案 7

如果比较条件受控，可以声称继承的行走目标 return 提高。不能声称完成过桥或掌握桥梁技能。最终结论仍由确定性物理评估决定。
