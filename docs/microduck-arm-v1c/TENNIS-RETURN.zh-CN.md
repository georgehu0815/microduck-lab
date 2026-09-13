# 网球拾取并放回垃圾桶：75 mm 宽口夹爪候选

## 用户确认的半高释放规则——2026 年 9 月 13 日

用户允许在桶内松爪，让球自然落下，不再要求松爪前必须接触桶底。
该任务独立编号为 `tennis-half-height-gravity-return-v2`；历史桶底释放实验不改标签、
不追溯改判成功。

松爪前，整个球必须严格处于桶内，球心不高于桶内底面上方 50 mm
（世界坐标高度 54 mm）。更低位置、包括接触桶底后释放，也符合新规则。
实测释放速度必须满足配置中的限制，禁止向上或横向抛射；松爪后允许重力下落。
最终仍要求桶底支撑、无机器人接触、机械臂退出到安全距离、低球速，连续稳定两秒。
机器人撞桶依然失败。

释放后将数值接触与出桶区分：至多 0.5 mm 的侧向越界必须同时有对应球—桶壁的
真实非正距离碰撞对、受限穿透量和不变的竖向完整容纳条件。这是明确记录的仿真容差，
不是实体制造公差。松爪初始位置仍采用严格几何容纳条件。

离线规划器在仿真副本中完整预演后，才选择步行预备时间、速度和增益；不按 seed 查表，
不移动自由基座或球，不施加外力。**这是使用仿真真值的离线教师，不是实时 MPC，
也不是新一轮全身 PPO 训练。**

### 当前统一结果：v13，30/30 仿真入桶通过

完整 `half-height-planned-v13` 矩阵已通过 **30/30**，覆盖三种球型、每种种子 0–9。
两个负对照均按预期超时，未误判成功。v11 小矩阵仍是 6/6，历史 v12 仍是 29/30；
本轮结果来自统一版本的完整重跑，没有把单独诊断拼成 30/30。

| 项目 | 统一 v13 结果 |
|---|---:|
| 标称 / 小球 / 大球 | 10/10 / 10/10 / 10/10 |
| 地面拾取并完整放回 | 30/30 |
| 开爪 / 保持负对照 | 2 个预期失败，0 个误成功 |
| 仿真任务时长：最短 / 中位数 / 最长 | 46.208 / 55.299 / 79.998 秒 |
| 导航候选预测次数，包含被拒绝的预测 | 107 |
| 已核验的完整视频 | 32 段：30 个入桶任务 + 2 个负对照 |
| 对比的控制步数 / 监测的物理子步数 | 94,765 / 947,532 |
| 相关全套回归 | 465 项通过，10 个警告 |
| 新增 PPO 训练步数 | 0 |
| 真机放行 | 否 |

v12 最后失败的小球种子 5，本轮用 2,575 步完成。修复是把 3.25 秒步行预备时序
加入**通用的 15 组候选集**，不是按种子选择动作。拾取完成后，每个候选均依据后续
搬运、停靠、松爪、撤离全过程的仿真结果筛选；被拒绝的预测也保留在每个回合的
`navigation_prediction` 中。预测不能覆盖实际运行回合的成功判定。
种子 0–9 是开发中使用过的回归案例，不是独立留出的泛化测试集。

**释放高度边界：**实测授权松爪时球心世界高度为 36.493–38.290 mm，低于允许上限
54 mm。全部 30 回合记录的**松爪后无支撑自由落体时长均为零**。因此证据证明的是
“半桶高度或更低位置释放”规则下完整放回桶内，**不是从半桶高度丢下的能力**。
新规则明确允许更低位置、包括桶底支撑后释放。

控制器协同使用 **15 个执行器：10 个腿、4 个机械臂姿态、1 个夹爪**，没有腰电机。
本轮组合了既有腿部 ONNX 策略、机械臂 IK、反馈和仿真真值轨迹筛选；没有新增
BC/DAgger/PPO 训练或 ONNX 导出，因此不存在本轮 reward/loss 训练曲线。

继承的评估器允许超出标称关节范围 0.08 rad；本轮实测最大超出量为 0.013041 rad，
不能据此宣称严格标称限位合格。
80 秒回合期限不证明原 15 秒性能目标。网球载荷仍超过原设计的 20–30 g 目标。
电气、热、结构、实时运行和真实机器人验证仍未完成。

### 复现与证据

从工作区根目录运行，使用 MuJoCo 3.10.0 与现有环境：

```sh
export PYTHONPATH=.:microduck_local/src
export OMP_NUM_THREADS=1
PY=rlx/.venv-microduck/bin/python
BASE=artifacts/microduck-arm-v1c/tennis-return
$PY -m microduck_arm_experiments.tennis_return \
  --gripper wide_candidate --release-mode half_height \
  --plan-navigation --max-steps 4000 --workers 12 \
  --out "$BASE/half-height-planned-v13-reproduction"
```

复现时使用新目录，不覆盖历史证据。规划是离线过程，实际计算可能明显慢于仿真时间。
完整实测矩阵为 `BASE` 下的 `half-height-planned-v13/evaluation.json`，同级保留 32 个
回合目录和源码快照。最终回归记录为 `repair-development/half-height-v13-release-regression.log`。

**32 段视频均已生成并核验**，位于 `BASE` 下的 `half-height-v13-videos/`。
打开其中的 `videos.html` 可查看中英文视频集，逐段播放；`README.md` 提供逐案例 MP4
索引。每个回合还保留原始遥测、结果和 `replay-validation.json`。

视频是**记录动作的物理重放**，不是直接摆放姿态的动画，也不是第二次独立策略评估。
每条记录指令都在重新初始化的 MuJoCo 环境中执行，原始遥测逐字节保留。
32 回合共对比 94,765 个控制步、监测 947,532 个物理子步；记录时刻、球位置、关节
位置和动作往返转换的最大误差均为**零**。

独立视频核验器另外检查 50 Hz 采样的验收条件，包括释放、容纳、安全、自由落体限制
和最后两秒稳定保持，并完整解码所有 MP4。视频为 640 × 480、25 fps，明确标注仿真与
动作重放。采样验收检查和重放中的物理子步监测是两种不同证据，不互相冒充。

视频目录中的主要文件：

- `evaluation.json`：绑定视频文件的完整评估报告。
- `evidence-summary.json`：实测汇总与报告哈希。
- `video-verification/manifest.json`：32/32 段完整解码与一致性检查。
- `video-verification/contact-sheet.jpg`：每个回合五张采样图。
- `video-verification/terminal-contact-sheet.jpg`：全部 32 回合的真实最后一帧。

从新生成的指标目录复现视频：

```sh
$PY scripts/replay_tennis_evidence.py \
  --source "$BASE/half-height-planned-v13-reproduction" \
  --out "$BASE/half-height-v13-videos-reproduction" --workers 1
$PY scripts/verify_tennis_videos.py \
  --root "$BASE/half-height-v13-videos-reproduction"
```

本机四进程渲染在完成 31 段后，有一个进程卡在 Metal/OpenGL 缓存文件锁。保留并校验
已完成视频的哈希、原始来源绑定、步数和误差后，通过单进程 `--resume` 恢复缺失回合。
中断日志与堆栈保留在 `repair-development/half-height-v13-render-recovery.md` 及配套文件中。
最终核验包含恢复的视频，没有省略缺失案例。恢复检查还拒绝输出文件别名，并按
来源模型验证精确物理子步数，包含提前结束的最后一个控制步。

## 历史桶底释放 v9 结果

日期：2026-09-13。状态：**历史完整指标矩阵 `wide-repair-v9-screen` 在既有评估容差下
完整放回 11/30，地面抬球 30/30；仍未全部通过。中间版本 v7 为 9/30，历史 v6 为 4/30。
独立的 `wide-repair-v9` 重跑现已完成 32/32 段声明视频的完整解码与核验；
其正任务标签复现 11/30，并不是 32 个任务成功。
这不是严格标称关节限位合格、PPO 训练成功、实时控制验证或硬件放行。**

## v9 修复、验证与未通过项

| 项目 | 完整指标矩阵结果 |
|---|---:|
| 正任务 | 11/30 |
| 标称 / 小球 / 大球 | 5/10 / 5/10 / 1/10 |
| 从地面真实抬球 | 30/30 |
| 开爪 / 保持负对照 | 0 个误成功 |
| 完整解码视频 | 32/32；其中正任务成功 11 个 |
| 相关代码回归 | 372 项通过，10 个警告 |
| 新 PPO 更新 | 0 步；reward/loss 不适用，未伪造训练曲线 |
| 硬件验证与制造放行 | 无 |

成功种子：标称 **0、2、3、5、6**；小球 **0、2、3、6、8**；大球 **8**。
剩余 19 次失败为：**撞桶 10、超时 4、提前失握 4、非夹垫撞球 1**。
与 v6 相比成功数增加，但撞桶次数从 3 增至 10，**不能把成功率提高等同于安全性全面改善**。
本版本仍是未放行的实验候选，而不是可上真机的控制器。

### 实际修改

1. 保留原腿策略；仅当有运动指令但实测位置 2.5 秒未产生 5 mm 进展时，才启动
   0.06 rad、1 Hz 的双髋滚转恢复动作。腿部原有 6 rad/s 指令限速没有取消。
2. 降入桶前，分别预测 25、30、35、40 mm 下蹲候选。只有独立仿真副本完成整个
   放置、释放、撤离与稳定保持流程，才授权该停靠姿态和下蹲量。
3. 释放比较原控制器与“保持手臂指令、保持腿部反馈、按 0.025 rad/s 慢开爪”方案。
   两者都未预测成功时不授权开爪；并不把“最不坏”的失败方案当作通过。
4. 慢开爪同时检查实测手臂姿态和实际保持的关节指令姿态，保留桶底支撑、整球在桶内、
   禁止接触及夹爪扫掠检查。预测不能改写真实回合的成功字段。

**这是使用模拟器完整真实状态的克隆预测教师**，不是新学到的 PPO 策略，也不是已经
验证的实时 MPC 部署。停靠每个候选最多预测 2000 控制步，释放每个候选最多 800 步；
预测耗费的真实计算时间不会被伪装成机器人已经达到 50 Hz 实时推理。
元数据明确记录 `forecast_source=simulator_truth_clone_teacher`、`ppo=false`、
`hardware_eligible=false`。预测中省略渲染器，独立复制模型、动力学状态和策略历史，
不移动真实自由基座或球，也不向真实场景注入外力。

整机协同输出仍为 **15 个执行器：10 个腿 + 4 个机械臂姿态 + 1 个夹爪**，没有腰电机。
本轮是控制与教师验证，不是“机械臂单独完成了全身 PPO 训练”。

三个球型 MJCF 与 v6 **逐字节相同**，冻结 v1/v1-C 源码也未改变。球、桶、碰撞、质量、
电机力矩限制与验收条件没有放宽。80 秒回合期限不等于原 15 秒目标已通过。
继承的关节验收容差仍为 0.08 rad；本矩阵测到最大标称关节超限约 **0.012744 rad**。
因此严格标称关节限位、真实电气/热/惯量、58 g 量级载荷及实时部署仍未合格。

### v9 证据入口

- 完整指标：`artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9-screen/evaluation.json`
- 逐回合 `result.json`、`telemetry.jsonl` 与 `source-snapshot/` 位于同一目录。
- 回归：`artifacts/microduck-arm-v1c/tennis-return/repair-development/regression-final-v9.log`
- 全视频重跑目录：`artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/`
- 视频核验清单：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/manifest.json`
- 视频联系图：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/contact-sheet.jpg`
- 终帧目视检查图：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v9/video-verification/final-frames-contact-sheet.jpg`
- 视频状态：**32/32 段声明视频通过哈希、终止遥测、帧数和完整解码检查；
  正任务成功仍为 11/30**。

终帧目视检查与记录标签一致：11 个标记通过的正任务终帧都显示球停在桶内；
其余正任务及两个负对照终止于可见的未通过状态。该定性检查不重新标注回合，
也不把视频完整性改写为任务验收通过。

### v9 冻结后的释放配置边界

已核验的 v9 证据包继续作为下文**桶底承托后释放**严格合同的历史证据。
另行标识的 `half_height` 释放配置及其新证据见本文开头；新结果不计入、
也不替换 v9 的 11/30。
由于活动规格正在变化，不使用工作树中的规格哈希对冻结 v9 作比较性结论；
历史比较边界仅限冻结证据包中的源码快照。

`--video-seeds` 留空会明确生成 `metrics_only` 指标包，视频核验器会拒绝该无视频包。
指定种子后生成 `metrics_and_video` 包；验证器要求所声明种子的三个球型视频及两个
负对照视频全部存在并完成哈希、遥测一致性和完整解码验证。历史未声明 `video_seeds`
的包仍按种子 0 的五段视频合同核验。

脚部几何投影、线速度预制动、全程髋摆和短时域导航预测另有诊断记录；它们没有作为
本矩阵的成功证据。未经完整矩阵回归，不把其中单个成功回合推广为通用修复。

## 直接结论

夹爪没有定死。本次保留旧夹爪，另外建立可替换宽口候选；不修改已存档的 v1-C
源码、质量假设、原任务门槛或原 ONNX。宽口机构是一个新的实验构型，不继承旧构型
的任务通过结论。

采用 75 mm 张开净间距。网球按真实量级建模：标称直径 67 mm、质量 58 g；另外
测试 65.4 mm / 56 g 和 68.6 mm / 59.4 g 两个 Type 1/2 边界组合。
这些组合不是所有直径/质量的笛卡尔积，也不包含更大的 Type 3 球。
尺寸出处登记在实验 JSON 的 ITF Appendix I 链接中；球的实际摩擦、弹性和惯量仍未测量。

## 机构尺寸与修改原因

| 项目 | 旧夹爪 | 宽口候选 |
|---|---:|---:|
| 张开时两夹垫内表面间距 | 19 mm | 75 mm |
| 两齿轮/手指转轴间距 | 24 mm | 24 mm，不变 |
| 单侧夹垫向外偏置 | 0 mm | 28 mm |
| 夹垫中心距腕关节的局部 x | 23 mm | 48 mm |
| TCP 局部 x | 30 mm | 55 mm |
| 夹垫尺寸 | 24 × 5 × 8 mm | 不变 |
| 机械臂执行器 | 5 | 5，不增加 |
| 整机执行器 | 15 | 15，不增加 |
| 理论连杆加 TCP 长度 | 135 mm | 160 mm，必须重新做负载预算 |

整机仍是十个腿关节、四个机械臂姿态关节、一个夹爪驱动；**没有腰部执行器**。
不改变原电机外形、安装轴距和镜像开合约束。宽口通过外扩手指实现，不是假装
24 mm 齿轮中心距可以直接拉大，也没有给被动手指增加一个虚拟电机。

只扩大开口还不够：直径约 67 mm 的球若仍以旧 30 mm TCP 为中心，会侵入掌部/
电机外形。因此候选同时把 TCP 前移至 55 mm。承力连接条也向外避让，确保几何闭合
首次接触来自两侧夹垫，而不是硬质支撑条抢先撞球。

新增承力桥和连接条以 PA12 密度 1100 kg/m$^3$ 假设计算，合计约 4.20 g。
这是未开孔毛坯体积，不含尚未定义的轮毂、轴保持件、紧固件和公差；不是最终重量。
CAD 输出为概念 OpenSCAD 与尺寸 SVG/PDF，**没有完成轴孔/轮毂配合或制造放行**。

## 夹持角度不能直接固定

对“有效指长 40 mm、初始净开口 75 mm”的简化对称直指：

$$\theta=\arcsin\frac{75-D}{2\times40}$$

当球径 $D=66$ mm，每指约 6.46°，两指相对角度变化约 12.92°。
这不是当前外扩指的关节指令。当前模型有偏置、夹垫厚度和 55 mm TCP，其平面近似为：

$$0.0255+0.012\cos\theta-0.055\sin\theta=D/2$$

这里长度用米，球心假定在 TCP，忽略变形。66 mm 球约为每指 4.65°。
MuJoCo 按 0.001 rad 步进的刚体接触扫描得到：

| 球径 | 每指首次双侧夹垫接触角 | 张开穿模 | 非夹垫抢先接触 |
|---|---:|---|---|
| 65.4 mm | 4.98° | 无 | 无 |
| 67.0 mm | 4.18° | 无 | 无 |
| 68.6 mm | 3.32° | 无 | 无 |

这三项是**运动学接触筛查，不是抓持力、抗滑、动态负载或真实抓取成功**。
不能用角度硬压网球；夹持力闭环、接触材料、传动效率及电流限制必须另外验证。
“接触点约 ±33 mm”仅是 66 mm 球赤道处对称夹持的球面几何描述，不是转轴间距。

## 场景与不可放宽的验收

球从地面开始；鸭身保留自由基座，球是自由体，不使用吸附、焊接或外力搬运。
垃圾桶为固定在地面的低矮课堂测试桶：内部 180 × 180 mm，侧壁 100 mm，底板/壁厚
4 mm，中心距起点约 0.4 m。这**不是**已验证的家用高垃圾桶。

流程：接近 → 双侧夹持 → 离地 → 搬运 → 降入桶内 → 底部支撑后松爪 → 撤离 → 保持。

- 开始时必须确实在桶外地面，不能在桶内出生而计作“放回”。
- 球底离地至少 20 mm，保持双侧接触 0.5 s。
- 水平搬运至少 120 mm，不是把桶挪到球下方。
- 球整体进入桶内，不能只判断球心进入，不能算挂在桶沿。
- **松爪之前必须由桶底承托**；放置速度最多 0.03 m/s，不允许投掷或提前掉落。
- 松爪后机器人不再接触球，TCP 撤离至球半径加 20 mm 以外，球在桶底稳定 2 s。
- 全程拒绝跌倒、自碰、与桶碰撞、关节越界及非有限遥测。

球使用解析薄球壳惯量和未标定刚体接触参数；其来源/假设与硬件放行分开记录。
新场景使用真实球质量，并保持原 v1-C 小物体场景的 50 g 创建限制原封不动。

## 冻结 v6 实际结果与局限

评估已经完成。以下数字只描述 `wide-repair-v6` 冻结证据包。预测导航 / MPC 与
SOUTH 路线实验没有集成进该证据包，也没有建立完整合格的改进结论。源码/控制器
审查仍在继续；这是冻结的历史测量，不代表工作树性能保证或发布批准。

| 检查 | 冻结 v6 结果 |
|---|---:|
| 宽口几何三种球径 | 3/3 |
| 既有容差验收下完整放回 | 4/30 |
| 地面抬球 | 30/30 |
| 标称 / 小球 / 大球完整放回 | 4/10 / 0/10 / 0/10 |
| 开爪与保持负对照 | 0 个误成功 |
| 新生成并验证的视频 | 5 个；仅 `nominal-0` 为任务成功 |
| 本轮 PPO 更新 | 0 步 |
| 硬件验证 | 无 |

成功记录是标称球种子 **0、2、3、5**，分别在 52.846、51.650、54.154、53.988 秒完成。
仅种子 0 有视频；其余三个成功由结果/遥测记录支持，不是额外视频。
评估时域为 4000 个控制步，即 50 Hz 下 80 秒；v2 是 750 步，即 15 秒。
这些完成记录**不满足原 15 秒期限**。上文物理验收没有放宽：地面起球、双侧夹持
并离地、水平搬运 120 mm、整球入桶、桶底承托后的低速松爪、撤离、稳定 2 秒及全部
安全拒绝条件仍然有效。证据包的 `all_tasks_passed` 仍为 `false`。

26 个失败分别为：18 个 `timeout`、3 个 `robot_bin_collision`、
3 个 `premature_release_or_throw`、1 个 `non_pad_ball_collision`、
1 个 `unsupported_or_fast_release`。两个负对照均超时且没有误成功。

**关节限位审查注意事项：**评估器原有 **0.08 rad** 的关节限位容差，并非 v5 或 v6 新增。
审查发现，标称球成功回合在 MuJoCo 软限位下，肘关节超过标称 1.8 rad 上限约
0.009–0.011 rad。因此 4/30 是通过既有容差门槛，**不是严格标称关节限位合格**。
不能据此宣称所有物理约束完美满足或硬件安全。
v6 现在逐帧记录 `maximum_nominal_joint_excursion_rad`，并记录
`evaluator_joint_tolerance_rad`，显式显示标称限位超调，但没有收紧既有验收容差。

整机实际为 15 个执行器：10 个腿执行器、4 个机械臂姿态执行器、1 个夹爪执行器，
腰部执行器为 0。`tennis_controller.py` 在旧
`alpha_walking.onnx` 腿策略上增加全身协调运动学教师，使用实际状态反馈完成下蹲、
向下地面 IK、折臂带载抬升、带载导航，以及带碰撞/力矩筛查的放置。这不是新训练的
端到端策略，也不是新 PPO。

v6 保留 v5 的 **25 mm** 放置下蹲；新的释放 IK 目标不可达，或其
肩俯仰 **>= 0.62 rad** 时，保留缓存的最后安全 IK 目标。规划受阻的回退路径仅在
经过至少 **1 秒**、球速持续 **< 0.001 m/s 达 0.3 秒**，并通过新增缓存姿态
释放检查后开爪。

v6 将规划受阻阶段正式标记为 `release_cached_pose`。从缓存姿态开爪前，必须实际
由桶底承托、整球在桶内、没有禁止接触，并通过**临时数据上的九个开爪位置碰撞扫描**，
不会移动实时机器人。扫描允许预期的夹垫/球接触，拒绝其他机械臂穿透接触；这种离散
运动学筛查不是连续运动或硬件安全证明。遥测现在显式记录 `release_plan_blocked`
与 `cached_release_screen_passed`；回退路径未显式记录的局限属于历史 v5。

物理监视器仍检查桶底承托、释放速度、整球入桶、撤离和稳定；其验收阈值与电机限值
没有改变。标称、小球和大球三个场景 XML 在 v2、v5、v6 间的 SHA-256 均分别相同。
这证明场景文件没有变化，不等于硬件合格或全部源码依赖一致。该冻结运行不是发布
批准，也不是所有约束完美满足的证明。

冻结回归日志为 **347 passed, 10 warnings**。它覆盖本轮修复并保留验收防线，包括
拒绝“球在桶内出生”的误成功以及不安全释放。测试通过不能把 4/30 矩阵改写为任务
全部通过，也不能代替严格关节限位合格验证。

真实负载不能由“电机堵转力矩大”推导：58 g 球在 160 mm 水平力臂上的球体单独
静态力矩已约 0.091 N·m，尚未计臂体、手指、加速度及安全裕量。原 0.1 N·m 仿真
筛查上限不是经过认证的连续额定值。

下一轮门槛仍是宽口夹爪台架负载/抗滑验证、腕部朝向和全身下蹲的无碰撞 IK、
可靠的带载导航与放置，再建立示教数据和全身残差学习。本轮**新 PPO 步数为 0**，
没有任务合格的网球放回 ONNX，也不能把旧 10 g ONNX 用作 58 g 网球能力证明。
v6 没有在硬件上运行。

## 历史 v5、v4 与 v2 结果

旧证据只用于历史对照，不能描述 v6：

- 冻结 `wide-repair-v5`：同一既有关节容差下完整放回 4/30、地面抬球 30/30，
  时域 4000 步 / 80 秒。成功为标称种子 0、2、3、5；五个已验证视频中仅
  `nominal-0` 成功，回归日志为 302 passed。规划受阻时的缓存释放没有显式遥测，
  也没有 v6 的正式释放筛查。

- 冻结 `wide-repair-v4`：完整放回 1/30、地面抬球 30/30，时域 4000 步 / 80 秒；
  标称种子 0 在 56.228 秒完成。其五个已验证视频为一个成功、四个失败；回归日志
  为 297 passed。

- `wide-screen-v2` 完整任务为 0/30。
- 其中五个视频（`nominal-0`、`small-0`、`large-0`、`open_jaw`、`hold`）
  全部是失败视频。
- 该运行使用旧的 750 步 / 15 秒时域，早于 v4 的协调下蹲、折臂抬升、带载导航和
  放置控制器。

## 与旧任务的状态分离

另一个已完成的原任务复核：教师自由基座 120/120、固定基座 80/80；但实际全身
残差 ONNX 矩阵为 **100/120**，不是 120/120。主要剩余失败是收臂行走与携物行走。
该批模型是 69 维观测、15 维残差，仍依赖教师；不代表独立学会全部技能。
本次宽口候选没有修改这些原模型、证据、硬件参数或旧策略接口。

## 文件与复现

- 实验源：`microduck_arm_experiments/tennis_return.py` 与
  `microduck_arm_experiments/tennis_controller.py`
- 参数：`hardware/microduck-arm-v1c/experiments/tennis-return.json`
- CAD/尺寸图：`hardware/microduck-arm-v1c/experiments/tennis-wide-gripper/`
- 原夹爪结果：`artifacts/microduck-arm-v1c/tennis-return/stock-screen-v2/evaluation.json`
- 历史宽口结果：
  `artifacts/microduck-arm-v1c/tennis-return/wide-screen-v2/evaluation.json`
- 历史 v4 结果：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v4/evaluation.json`
- 历史 v5 结果：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v5/evaluation.json`
- 冻结 v6 结果：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/evaluation.json`
- 冻结 v6 回归日志：
  `artifacts/microduck-arm-v1c/tennis-return/repair-development/regression-final-v6.log`
- 冻结 v6 视频验证与来源哈希：
  `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/video-verification/manifest.json`

冻结 v6 的五个视频准确路径为：

- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/nominal-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/small-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/large-0/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/open_jaw/rollout.mp4`
- `artifacts/microduck-arm-v1c/tennis-return/wide-repair-v6/hold/rollout.mp4`

对未声明视频种子的历史 v6，验证器完整解码且只接受这五个视频，核对视频帧数与遥测，要求每个 `result.json`
与评估记录完全一致，并要求末帧遥测的成功/失败字段与结果一致。它不会查看失败视频
后把标签改成成功。清单记录为一个已验证成功（`nominal-0`）和四个已验证失败。

清单保留评估、各回合结果、遥测、视频、源码快照、验证器和联系图的完整 SHA-256。
它只证明内部哈希一致性，不是密码学认证、独立物理重评估或严格关节限位合格验证。

清单中的 v6 精确 SHA-256：

- 评估：`91a6da825f7d73ec21b0daf1cb01d7659c16cf6bda3b4474cbe59e8031c0597c`
- 控制器快照：`fa8cad9e55ac71fc8b99d8e179c62a08c266498b56f1d1ec7445f970f65e33a3`
- 已记录验证器：`15a2bb0c32e1b4fc64e5d84b4f91ad49fba376ff789b99301380ec01fc2e79c8`

从仓库根目录运行，使用新输出目录，不覆盖历史证据。这些命令运行工作树中的控制器，
源码更新后不保证复现冻结 v6；归因前应将新运行的源码哈希与 v6 对照。

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
rlx/.venv-microduck/bin/python -m microduck_arm_experiments.tennis_return \
  --gripper wide_candidate --out artifacts/tennis-new-run \
  --max-steps 4000 --workers 4 --video-seeds 0

PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_microduck_arm_tennis_return.py -q

PYTHONPATH=.:microduck_local/src \
rlx/.venv-microduck/bin/python scripts/verify_tennis_videos.py \
  --root artifacts/tennis-new-run
```

现在只要任一正任务失败或任一负对照成功，评估 CLI 就退出 1。因此冻结 v6 虽然写出
完整证据，任务进程状态仍为非零；自动化收集部分成功诊断矩阵时必须处理该预期退出码，
然后单独运行验证脚本。视频完整性通过和单元测试通过都不能替代任务验收。

Mac 离屏渲染需要主机图形服务；无图形服务时可省略 `--video-seeds`，但不会产生视频。
