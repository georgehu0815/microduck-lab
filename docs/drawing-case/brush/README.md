# MicroDuck 彩色画笔升级：通过验收的学习策略

作者：George Hu
日期：2026-09-11
English: [README.en.md](README.en.md)

本页记录第八个 Studio 场景 `drawing` 的彩色画笔升级。它不是第九个场景，也不改写
此前嘴持铅笔实验的失败结论。旧铅笔合同 `microduck-drawing-v1`（83 observation /
15 action）仍保留；本升级使用独立的 `microduck-brush-v2`（93 observation /
15 action）。

## 已验证结果

### 两份独立录像

- [BC + DAgger：rollout_bc.mp4](media/rollout_bc.mp4)，来自 `bc.onnx`，不包含 PPO 更新。
- [BC + DAgger + PPO：rollout.mp4](media/rollout.mp4)，来自 `policy.onnx`。

两份录像均使用 seed `10001`、相同相机和 10fps，编码时长 120 秒。
BC 录像于 2026-09-12 单独生成，完整绘画验收通过；[来源和解码验证](media/render_bc.json)、
[BC 接触图集](media/frame_sheet_bc.png) 保留用于比较，不是 PPO 视频的重命名副本。

最终证据是
`rlx/runs/studio/drawing/brush-color-v2-03/eval.json` 和同目录
`bc-eval.json`。2026-09-11 的 actual Studio API 评估报告：

- deterministic PPO ONNX 共 `40` 个必测 episode：
  `10` 个 unique configuration x 每个 `4` 个 seed；
- matching BC ONNX 在完全相同的 40 个 episode 中也全部通过；
- 每个 episode 都完成 `5,998` 个 control step，即 `119.96 s`；
- PPO mean coverage 为 `1.0`，mean precision 为 `0.999531977503557`；
- BC mean coverage 为 `1.0`，mean precision 为 `0.9996346455645799`；
- PPO 所有颜色的最差 coverage 为 `100%`，最差 precision 为 `98.624%`
  （验收门槛均为 `95%`，距离容差 `1.5 mm`）；
- PPO 最差 grasp fraction 为 `99.983%`，最大法向力 `0.0404 N`，
  最大画笔压缩 `1.552 mm`，最大抬笔漏画率 `4.144%`；
- 四个颜料井均按 gold -> orange -> charcoal -> blue 的规定顺序接触；
- `null` 与 `open_jaw` 两个负控制都失败，因此评分器没有把不动作或张嘴掉笔误判为成功；
- `passed=true`、`skill_status="passed"`、`provenance_errors=[]`、
  `environment_source_match=true`。

这是当前仿真、当前固定图案和所测扰动范围内的通过结果。它不证明硬件可部署、未知
图案泛化或视觉规划能力。由于 matching BC 同样 40/40 通过且 precision 略高，唯一
支持的结论是 anchored PPO **保留**了 BC 的已通过行为；没有 PPO improvement claim。

`brush-color-v2-03` 由 run 中归档的旧 pipeline 训练。该版本在加载 teacher NPZ 时
检查相邻 JSON、dataset SHA-256 和当时的六项 source hash，并保存六份 source
archive；它**没有**训练后才加入当前代码的全生命周期重复 `assert_sources`，因此不能
声称 final03 训练过程享受了该新增保护。

当前 hardened evaluator 是训练后的独立验证层。它允许 evaluator-only 的
`ppo_microduck_brush.py` 版本变化，但要求归档的旧训练脚本和其余五份 source archive
全部匹配 ONNX 内嵌的六项训练 hash；environment、reference、actor 和两份 base
source 还必须与当前评估 source 一致。它同时验证实际 checkpoint hash、sidecar 与
ONNX 内嵌 checkpoint hash 一致、ONNX sidecar hash，并一次性读取 immutable ONNX
bytes 后检查评估期间文件稳定性。报告用 `training_pipeline_source_match` 明确记录
当前 evaluator 是否仍等于训练脚本；该值可以为 `false`，只要归档训练 package 匹配
且其余 provenance gate 全部通过。ONNX export 还把输出第二维静态固定为 `15`，用于
Lab/Viewer 的 15-action 合同检查。

### Run 历史

| run | 状态 | 说明 |
|---|---|---|
| `brush-color-v2-01` | 初步成功 | PPO 20/20；后补 matching BC 也是 20/20。其 normal-duck painting 已人工查看，但不用于 PPO 优于 BC 的结论 |
| `brush-color-v2-02` | 未提升 | data seed `101`；BC validation 在 2,464 step 提前终止并失败，保留为 training-data seed sensitivity |
| `brush-color-v2-03` | 最终评估通过 | data seed `11`、train seed `101`；PPO 40/40，matching BC 40/40，严格 provenance 通过 |

API evaluation 和 source-bound final03 render 已验证，文档媒体已经发布到本目录的
`media/`。最终 hardened evaluator 的 PPO 与 matching BC 重评估均已完成：
`eval.json` 创建于 `2026-09-11T22:20:38Z`，`bc-eval.json` 创建于
`2026-09-11T22:24:48Z`；两者均为 40/40、`training_pipeline_source_match=false`、
`provenance_errors=[]`。这些报告已包含 checkpoint 内嵌 hash 与六份 source archive
完整性检查，记录的 evaluator digest 与当前脚本一致。独立浏览器检查已通过：
`docs/drawing-case/brush/evidence/viewer.json` 记录八个案例、八只 live duck、
彩色接触数据流、同一模型的录像播放，以及 390px 手机布局无横向溢出和无页面错误。
live 检查是有界 smoke；完整 119.96 秒绘画由独立录像和 40 回合验收证明，不能混为一次完整 live 验证。

## 系统边界

完整任务由两部分组成：

1. **人工编写的固定计划**决定同一只游泳鸭、颜色分配、笔画顺序、抬笔移动和四个
   颜料井目标。
2. **学习到的反馈 actor + PPO**根据 93 维状态执行 15 个电机 action，维持夹持、
   平衡、接触、画笔压缩和轨迹跟踪。

因此它不是 learned vision、learned image understanding 或 learned task planning，
也不是 PPO from scratch。导出的 ONNX 内没有运行时 Jacobian teacher；推理只执行
学习到的 actor。

新增的 loaded-color、planned-color、compression 和 mode observation tail 用于
环境合同与未来扩展，但当前 actor 的特征提取仍停在原 83 维相关字段，不读取
`83:93` color tail。因此颜色切换和蘸色时机来自 authored sequence；如果漏蘸，
当前 actor 没有学习到自主识别并恢复的策略。

## 实现入口

| 文件 | 作用 |
|---|---|
| `rlx/rlx/environments/brush.py` | 自由画笔、被动画笔压缩、实体颜料井、93/15 合同、奖励与验收 |
| `rlx/rlx/environments/brush_reference.py` | 固定彩色游泳鸭笔画、调色板顺序和 approach/dip/lift 轨迹 |
| `rlx/rlx/models/drawing_feedback.py` | 结构化反馈 actor、ridge BC、分 action 探索噪声 |
| `rlx/examples/ppo_microduck_brush.py` | 数据、DAgger、BC、anchored PPO、ONNX export 和独立评估 |
| `rlx/examples/render_microduck_brush.py` | 从 ONNX 或教师渲染真实 MuJoCo 接触颜料 |

## 仿真 BOM 与物理

这些是教育仿真的建模参数，不是采购清单、真实材料标定或硬件安全值。

| 项目 | 数量 | 仿真假设 |
|---|---:|---|
| MicroDuck 全碰撞模型 | 1 | 保留 14 个原关节，加 1 个仿真嘴部 action |
| 自由画笔 | 1 | 通过 `pencil_free` free joint 被上下喙垫夹持，不焊接到头部 |
| 画笔杆与金属箍 | 各 1 | 画笔杆替换铅笔外观；金属箍为非接触装饰几何 |
| 被动柔顺笔簇 | 1 | `bristle_compression` slide，范围 `-3.0..0.3 mm` |
| 装饰性笔毛 | 3 | 三条 capsule 仅用于可视化，不参与接触 |
| 接触笔簇 | 1 | 一个半径 `1.0 mm` 的柔顺球形接触体代表整束笔毛 |
| 调色板 | 1 | 非接触底板 |
| 实体颜料井 | 4 | gold、orange、charcoal、blue；必须由笔尖实际接触后才改变 pigment |
| 画布与画架 | 各 1 | 继承 drawing 场景的实体画布和画架 |

画笔柔顺关节是未驱动的被动弹簧：

- stiffness：`25 N/m`
- damping：`0.32 N s/m`
- effective armature：`0.001 kg`
- 默认画笔/喙垫摩擦：`1.2`
- 画笔接触球质量：`0.00003 kg`
- physics timestep：`0.002 s`，即 `500 Hz`
- control timestep：`0.020 s`，即 `50 Hz`
- 每个 control action 执行 `10` 个 MuJoCo physics substep

三条可见笔毛不分别计算有限元弯曲、毛束间流体或单根接触；物理近似是
**三条装饰性 hair strand + 一个被动柔顺 contact tuft**。

颜料也采用离散接触状态，而不是流体仿真：

- 只有画笔接触某个实体颜料井且法向力超过 `0.0001 N` 才加载该颜色；
- 接触画布时，记录的是当时已加载颜色的真实 MuJoCo 笔尖接触；
- 没有颜料混合、洗笔、残留颜料比例、吸液或再次蘸取动力学。

## 93 observation / 15 action 合同

前 83 维与保留的铅笔合同布局一致：

| 索引 | 数量 | 内容 |
|---|---:|---|
| `0:3` | 3 | IMU angular velocity |
| `3:6` | 3 | trunk frame projected gravity |
| `6:20` | 14 | 原关节相对默认姿态的位置 |
| `20:34` | 14 | 上一步原关节速度 |
| `34:48` | 14 | 上一步原关节 action |
| `48:61` | 13 | 保留的 twist/head/body command 槽，本任务为 0 |
| `61:64` | 3 | mouth position、velocity、last action |
| `64:67` | 3 | trunk frame 中画笔尖相对位置 |
| `67:70` | 3 | trunk frame 中 target minus tip |
| `70:73` | 3 | trunk frame 中画笔轴方向 |
| `73:76` | 3 | 画笔尖速度 |
| `76:79` | 3 | 目标轨迹速度 |
| `79` | 1 | requested pen-down |
| `80:83` | 3 | 画布、上喙垫、下喙垫接触力 |

新增 10 维：

| 索引 | 数量 | 内容 |
|---|---:|---|
| `83:87` | 4 | 当前已加载颜色 one-hot；未蘸色时全 0 |
| `87:91` | 4 | 当前计划颜色 one-hot |
| `91` | 1 | 笔簇 slide qpos，单位 mm |
| `92` | 1 | 轨迹 mode 除以 2：移动、蘸色或绘画阶段 |

Action 仍为 15 维 absolute normalized actuator offset：

- `action[0:14]` 控制原 14 个关节；
- `action[14]` 控制仿真嘴部，`mouth_target = 0.24 + 0.36 * action[14]`；
- actor 的头部输出是相对上一 head action 的有界 delta，每步最大 `0.025`；
- 头部 ridge 特征包括 head position、target error、画笔方向、接触力及
  head-position x target-error 的二次交互项；
- 特征刻意不输入上一 head action，上一 action 只在输出端加回，因此拟合的是
  action delta，不是复制 last action；
- 脚踝由 gravity + angular velocity 的线性反馈控制；其他身体关节和嘴部学习常量
  offset。

## 数据、BC、DAgger 与 PPO

最终 `brush-color-v2-03` 的实际流水线：

1. data seed `11` 的解析式 Jacobian teacher 在 8 个训练条件中产生
   `47,984` 条记录；
2. 对结构化反馈 actor 做 ridge BC，`ridge=0.01`；
3. 执行 2 轮 DAgger，在当前 policy 访问状态上重新调用 teacher 标注，再重新 ridge fit；
4. train seed `101` 从 BC/DAgger actor 开始执行 `49,152` 个真实 PPO
   environment step；
5. PPO 共 8 次 update，8 次都通过 anchor gate；
6. 导出 deterministic 93x15 ONNX，再通过 actual Studio API 分别对 PPO 和
   matching BC 运行 40-episode acceptance。

探索噪声按 action 类型固定初始化：

- head：`0.0003`
- leg/body：`0.0001`
- jaw：`0.0002`

PPO 使用 `learning_rate=1e-7`，每次 update 后在最多约 4,096 个 BC/DAgger
anchor observation 上检查 deterministic actor。最大 action drift 必须
`<=0.00025`；否则同时回滚 policy 参数和 optimizer state。本 run 的 8 次 update
全部被接受。

没有修改 reward curriculum 来取得该结果。画笔任务一直无 assistance，奖励仍是
tracking、contact、grasp、upright、action-rate、force 和 color 项。anchored PPO
限制的是相对 BC/DAgger 行为的更新幅度，不伪造轨迹、接触、颜料或验收结果。

## 为什么旧 drawing PPO 失败

历史铅笔 run 的已确认问题是：refinement 配置了 `initial_std=0.005`，但 BC/DAgger
之后的代码无条件把 PPO 起始 `log_std` 重置为 `0.02`。现有 checkpoint 与 metadata
已经确认该配置/实际值不一致。

另外，测量显示两个旧 run 都从各自 BC baseline 发生 cumulative policy drift，
teacher-state action MSE、BC-to-PPO action drift 和累计 KL 都上升，同时绘图 coverage
下降。**这是测得的退化现象，不是已证明的单一根因。** 缺少 BC anchor、局部 reward、
短 rollout、critic 初始化和 recovery-state 分布偏移仍只是待消融的假设。

新画笔 run 采用新的物理、状态和结构化 actor，并加入低噪声、极低 learning rate 与
anchor 回滚。PPO 与 matching BC 都 40/40 通过，因此它证明这套训练流水线产生了
被接受的 BC，并且 PPO 没有破坏该行为；它不能当作“PPO 优于 BC”或某个旧假设已经
被单因素证明的证据。

## 验收合同

每个 required episode 必须同时满足：

- gold、orange、charcoal、blue 每色 coverage `>=95%`；
- 每色 precision `>=95%`；
- 几何距离 tolerance 为 `1.5 mm`；
- 完成全部 `119.96 s`；
- `grasp_fraction >=90%`；
- `max_normal_force_n <0.5 N`；
- `max_compression_m <3.5 mm`；
- `pen_up_leak_fraction <5%`；
- 调色板接触顺序严格为 gold、orange、charcoal、blue；
- 未跌倒、未掉笔、无 assistance；
- `null` 和 `open_jaw` 负控制必须失败。

十组 required configuration：

| condition | 参数 |
|---|---|
| nominal | 默认参数 |
| offset | 图案 offset `(0.002, 0.002) m` |
| scale | 图案 scale `0.95` |
| friction | 夹持摩擦 `0.9` |
| compliance | 画笔 stiffness `30 N/m` |
| combined_left | offset `(-0.002, -0.001) m`、scale `1.04`、friction `0.95`、stiffness `23 N/m` |
| combined_small | offset `(0.001, -0.002) m`、scale `0.92`、friction `1.1`、stiffness `32 N/m` |
| combined_right | offset `(0.004, 0.001) m`、scale `1.06`、friction `0.9`、stiffness `28 N/m` |
| boundary_large | offset `(0.005, -0.003) m`、scale `1.12`、friction `0.8`、stiffness `40 N/m` |
| boundary_soft | offset `(-0.004, 0.003) m`、scale `0.88`、friction `1.0`、stiffness `18 N/m` |

## 精确命令

从 workspace 根目录运行：

```bash
export PYTHONPATH=rlx:microduck_local/src
PY=rlx/.venv-microduck/bin/python
BRUSH=rlx/examples/ppo_microduck_brush.py
DATA=rlx/runs/brush-data/final-seed11
RUN=rlx/runs/studio/drawing/brush-color-v2-03
```

生成 8 个 teacher episode：

```bash
$PY $BRUSH data \
  --out "$DATA" \
  --seed 11
```

训练 ridge BC + 2 轮 DAgger + anchored PPO：

```bash
$PY $BRUSH train \
  --out "$RUN" \
  --data "$DATA/teacher.npz" \
  --seed 101 \
  --steps 49152 \
  --dagger 2 \
  --ridge 0.01 \
  --learning-rate 1e-7 \
  --anchor-limit 0.00025
```

独立评估：

```bash
$PY $BRUSH eval \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/eval.json" \
  --seed 10001 \
  --episodes 4
```

对 matching BC 运行完全相同的 40-episode gate：

```bash
$PY $BRUSH eval \
  --onnx "$RUN/bc.onnx" \
  --out "$RUN/bc-eval.json" \
  --seed 10001 \
  --episodes 4
```

渲染学习到的 ONNX：

```bash
$PY rlx/examples/render_microduck_brush.py \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/render" \
  --seed 501 \
  --seconds 120
```

教师只用于诊断渲染：

```bash
$PY rlx/examples/render_microduck_brush.py \
  --teacher \
  --out /tmp/microduck-brush-teacher \
  --seed 501 \
  --seconds 120
```

检查当前 source hash 是否与 `eval.json` 一致：

```bash
shasum -a 256 \
  rlx/examples/ppo_microduck_brush.py \
  rlx/rlx/environments/brush.py \
  rlx/rlx/environments/brush_reference.py \
  rlx/rlx/models/drawing_feedback.py \
  rlx/rlx/environments/drawing.py \
  rlx/rlx/environments/drawing_reference.py

jq '.source_hashes, .training_source_hashes, .provenance_errors,
    .unique_conditions, .environment_source_match,
    .training_pipeline_source_match, .source_sha256,
    .source_files_sha256' "$RUN/eval.json"
```

如果 ONNX 缺失或需要重新写入静态 15-action shape，可以从保留 checkpoint 重新
导出。此时 environment、reference、actor 和 base source 必须仍与 checkpoint
metadata 一致，`sources/ppo_microduck_brush.py` 必须与内嵌训练 pipeline hash
一致；当前 `ppo_microduck_brush.py` 仅含 evaluator 改动时不要求重新训练。重新导出
会产生新的 ONNX bytes，必须重新运行 PPO 与 BC 评估并记录新的 immutable hash：

```bash
$PY $BRUSH export \
  --checkpoint "$RUN/policy.zip" \
  --onnx-output "$RUN/policy.onnx"

$PY $BRUSH eval \
  --onnx "$RUN/policy.onnx" \
  --out "$RUN/eval.json" \
  --seed 10001 \
  --episodes 4

$PY $BRUSH export \
  --checkpoint "$RUN/bc.zip" \
  --onnx-output "$RUN/bc.onnx"

$PY $BRUSH eval \
  --onnx "$RUN/bc.onnx" \
  --out "$RUN/bc-eval.json" \
  --seed 10001 \
  --episodes 4
```

如果 environment、reference、actor、base source 或归档训练 pipeline 不匹配，
strict provenance gate 会拒绝该 artifact；re-export 不能修复这种训练来源不匹配，
必须用新 source 重新生成 teacher dataset，并训练到新的 immutable run directory。
只有当前 evaluator pipeline 版本不同、且归档训练脚本仍匹配内嵌 hash 的情况可以
继续评估；报告会把 `training_pipeline_source_match` 记为 `false`，而不是伪装成训练
与评估脚本相同。

## 证据与限制

- PPO 通过证据：`rlx/runs/studio/drawing/brush-color-v2-03/eval.json`
- matching BC 证据：`rlx/runs/studio/drawing/brush-color-v2-03/bc-eval.json`
- 训练记录：`rlx/runs/studio/drawing/brush-color-v2-03/training.json`
- PPO 历史：`rlx/runs/studio/drawing/brush-color-v2-03/ppo_history.csv`
- ONNX：`rlx/runs/studio/drawing/brush-color-v2-03/policy.onnx`
- source archive：`rlx/runs/studio/drawing/brush-color-v2-03/sources/`
- Studio API receipt：`docs/drawing-case/brush/evidence/workflow.json`
- preliminary visual check：`brush-color-v2-01/render/painting.png` 显示正常游泳鸭；
  它只保留为历史视觉检查
- final03 source-bound render：`docs/drawing-case/brush/media/render.json`
- 文档静态图：`docs/drawing-case/brush/media/painting.png`
- 文档动画：`docs/drawing-case/brush/media/brush.gif`
- 浏览器检查：`docs/drawing-case/brush/evidence/viewer.json`，`passed=true`；范围见上述边界
- 最终验证清单：`docs/drawing-case/brush/VERIFICATION.md`

本结果只覆盖当前 CPU MuJoCo 模型、固定 authored artwork 和列出的扰动条件。没有
相机输入、未知图片规划、真实流体颜料、单根笔毛有限元、硬件标定、采购建议或部署
声明。
