# MicroDuck Single-Arm v1-B：实现与验证报告

## 结论：工程验证原型已落地，整机放行未通过

本交付是可执行的**工程验证原型**，不是可直接加工装配的成品，也不代表全部任务已经通过。当前评估入口为 `artifacts/microduck-arm-v1/evaluation-attempt-3/evaluation.json`。第1、2次评估属于旧源码历史记录，不可与本次结果混算。

必须区分四种证据：软件测试通过、仿真任务成功、文件/视频完整性通过、实物验收通过。固定根部台架的到达任务成功，不代表整机能够行走、搬运、平衡，也不代表电气或硬件已通过验证。

最终定向软件测试为**137项通过**：新工程包42项、旧机械臂回归95项；有7条ONNX导出/追踪警告。原始记录为 `artifacts/microduck-arm-v1/software-tests.log` 和 `software-tests.xml`。这不代表物理任务套件通过。

## 已实现的工程包

| 交付项 | 仓库路径 | 真实状态 |
|---|---|---|
| 唯一规格源、尺寸、质量和关节顺序 | `hardware/microduck-arm-v1/spec/design.json` | 明确标注假设和未知项 |
| BOM、装配/指令式测试规程 | `docs/microduck-arm-v1/BOM.md`、`SOP.md` | 电机型号候选确定，其余采购接口仍有待定项 |
| 可编辑 CAD、机械尺寸图 | `hardware/microduck-arm-v1/cad/` | SCAD、SVG、PDF 评审图；无已放行的孔位或 STEP/STL |
| 原理图、接线图、网表 | `hardware/microduck-arm-v1/electrical/` | 逻辑拓扑，不是生产 PCB 或已验证线束 |
| 整机 MJCF | `hardware/microduck-arm-v1/generated/microduck-arm-v1-free.xml` | 自由根部、真实物体接触 |
| 台架 MJCF | `hardware/microduck-arm-v1/generated/microduck-arm-v1-fixture.xml` | 明确固定根部辅助，不冒充移动验证 |
| URDF | `hardware/microduck-arm-v1/generated/microduck-arm-v1.urdf` | 运动学交换，回差/碰撞近似有记录 |
| 力矩、重心、工作区、任务验证 | `microduck_arm_v1/validation.py` | 实际运行，失败完整保留 |
| BC、残差 PPO、ONNX、策略评估 | `microduck_arm_v1/learning.py` | 仿真训练工具，不是上机控制器 |
| 证据审计与视频索引 | `microduck_arm_v1/evidence.py` | 哈希、轨迹、视频帧完整性检查 |
| 硬件验收规程 | `docs/microduck-arm-v1/HARDWARE-ACCEPTANCE.md` | 未执行，缺失实测证据禁止放行 |

## 设计一致性与必要调整

- **4个姿态轴＋1个夹爪＝5颗执行器**。不能写成“5个姿态自由度再加夹爪”。保留10颗腿部执行器，总计15维动作、64维观测；旧的14动作/61观测策略不能直接加载。
- v1-B 统一使用五颗 `XL330-M288-T` 候选，在新工程包内保持电机型号和接口定义一致。这不证明持续扭矩足够，也不表示 M077 替代方案已验证。
- 肩肘、肘腕、腕端标称轴距为55/50/30 mm，合计135 mm；这是几何长度，不是承重半径或平衡工作区。
- 夹爪采用旋转双指、-1齿轮耦合。MJCF equality 和 URDF mimic 描述的是拟议传动；实际齿轮、回差、轴承和强度仍需设计。
- 模型电机包络采用宽/深/高20×26×34 mm。旧可视网格深度与其不同，不能根据网格或包络猜测安装孔。
- 原模型被移除的颈部子树质量为0.2799274 kg，并非只有空外壳。因此重新加入电子件、电池、电源器件和传感器壳的**未实测质量占位**，而不是把它们从重量账中删除。
- 新机器人模型质量为0.80631578 kg，不含任务物体。这不是实物称重结果；新增质量、重心和包装尚未实测。
- 手臂电源正极与原机身正极保持分域。电池/BMS/DC-DC/保险丝/连接器、再生能量保护、热测试仍未完成。本轮未给实物发送运动指令，也未修改机载控制软件。

## 实际仿真结果

教师为 IK＋分阶段状态机。评估种子910000、910001；每回合最多1500步、控制周期0.02 s，成功或失败时提前结束。任务物体为仿真10 g载荷，不代表已经取得10 g额定承重能力。

| 新案例 | 自由底座 | 固定台架 | 实际失败原因或限制 |
|---|---:|---:|---|
| 到达目标 | 0/2 | 2/2 | 自由底座跌倒 |
| 取放物体 | 0/2 | 0/2 | 环境碰撞／机器人自碰撞 |
| 避障移动物体 | 0/2 | 0/2 | 环境碰撞／机器人自碰撞 |
| 原地经过路点搬运 | 0/2 | 0/2 | 环境碰撞／机器人自碰撞 |
| 收臂行走 | 0/2 | 不允许 | 尚无可工作步态，环境碰撞 |
| 搬物行走 | 0/2 | 不允许 | 环境碰撞，整机控制器未验证 |
| **合计** | **0/12** | **2/8** | **整机任务放行失败** |

两次台架到达成功的末端位置误差分别为4.63 mm、4.09 mm，完整旋转误差分别为0.0457 rad、0.0427 rad，均经过一秒保持窗口。这只是两个固定种子的仿真结果，不是精度承诺，也不是40/40验收。

旧实验室的单臂到达、取放、移动、搬运在新模型中有对应场景，但旧形态的通过记录不能转移给新机器人。`arms-handover-v1` 和 `arms-co-carry-v1` 属于第二阶段双臂任务，单臂版本不声称实现。绘画、DAgger、颜色覆盖率/精确率、真实接触力资格验证也未在此工程包实现。

### 验收防作弊约束

每个物理子步出现的禁止接触都会记录。到达任务检查完整目标旋转，不能仅凭一根工具轴方向判断成功。有效抬升后失去抓握视为掉落，只有合法放置阶段才能释放；释放许可可撤销，搬物行走不具备此许可。行走要求近期、直立状态下、有前向进展的交替单足支撑事件。收臂行走检查实际臂姿；搬运检查当前抬升高度、双侧抓握、没有地面/桌面支撑，以及受约束的相对机身物体位置。

默认回合改为30秒，覆盖教师21秒时的释放阶段和1秒保持。更短回合只能明确作为冒烟测试，不能冒充完整搬运评估。目前行走教师仅保持 home 姿态，尚未实现步态。

### 工作区和力矩筛选

- 在定义的笛卡尔空间盒内采样10000个目标，其中3331个满足受约束的水平工具 IK 条件，2831个同时通过碰撞筛选。
- 平衡安全点数量未知。上述数字不能当作可用工作区，也不证明任意六维姿态可达。
- 500姿态准静态计算表采用20 g TCP载荷假设；3倍筛选因子下，所需合格持续扭矩的采样最大值为0.33063 N·m。表中包含明确标记的不可行姿态。
- 电机实际合格持续能力未知。模型±0.10 N·m限幅是筛选假设，不是厂家持续工况额定值。加速度、工具接触、发热和电池动态均未验证。

原始数据在 `artifacts/microduck-arm-v1/workspace.json` 和 `artifacts/microduck-arm-v1/mechanics/torque-com.csv`。

## 训练与视频证据

训练冒烟流程使用成功的台架到达示范、BC、带外部 IK 教师的残差 PPO，以及独立种子评估。精确检查点、loss、reward、归一化、源码哈希和 ONNX 一致性记录在 `artifacts/microduck-arm-v1/learning-smoke/`，应读取最终汇总而不是旧 attempts。短训练不等于收敛，也不证明 PPO 超过教师或整机移动策略已成功。

当前策略输入的 TCP、物体和目标位姿来自仿真真值；硬件状态估计尚未实现。残差 ONNX 必须配合相同教师、归一化和动作映射，不能直接交给 Radxa/robotd 驱动实物。

### 已记录的训练指标

训练种子101，采集8个成功示范回合、643条状态/动作样本。BC训练20轮；残差PPO训练2048个环境步、32次优化更新。独立评估种子为201至205；此到达任务冒烟显式采用250步上限。

| 控制器 | 独立评估成功数 | 平均回报 | 平均步数 |
|---|---:|---:|---:|
| IK/FSM教师 | 5/5 | -0.033414 | 88.0 |
| BC | 0/5 | -10.173485 | 250.0 |
| 教师＋残差PPO | 5/5 | -0.020445 | 87.6 |

BC动作MSE从0.097145下降到0.00033247，**但独立评估全部超时失败**。PPO首/末记录总loss为0.090142/0.048559，不要求单调下降；23个已结束训练回合平均回报-0.102520。原始JSONL同时记录value loss、policy-gradient loss、entropy、近似KL、clip fraction、explained variance。5个小样本配对评估不足以证明PPO在统计上稳定优于教师。

ONNX在另外的301至305种子观测上通过数值一致性检查：BC最大绝对误差1.55e-7，原始PPO残差2.24e-8；残差不是独立动作策略。

![实际BC和PPO损失曲线](../../artifacts/microduck-arm-v1/learning-smoke/training-curves.png)

本轮10段 IK/FSM MP4 覆盖全部6个自由底座和4个台架案例，失败视频也保留。打开 `artifacts/microduck-arm-v1/verified-evidence/videos.html` 浏览。索引检查每段视频哈希、H.264/25fps格式、包含终止帧的采样帧数及对应轨迹。已查看台架到达与自由底座到达终止画面，分别可见成功和跌倒；视频完整性不等于任务成功。

同一视频库还包含独立种子201的两段实际训练后检查点视频：BC超时、教师＋残差PPO成功。**当前共12段视频**，均有轨迹和检查点/文件哈希。训练后视频在 `artifacts/microduck-arm-v1/learned-videos/`，没有把教师视频冒充学习策略结果。

关闭执行器和张开夹爪对照保存在 `artifacts/microduck-arm-v1/negative-controls-final/`。张开夹爪不是到达任务的有效负对照；如果教师本身无法完成操作，对照失败也不能证明该操作基准有效。

## 复现方法

在工作区根目录使用已有环境。当前验证版本为 MuJoCo 3.10.0、Stable Baselines3 2.9.0、PyTorch 2.9.1、ONNX Runtime 1.29.0。PDF工具和 ffmpeg/ffprobe 为独立系统工具。本机没有 OpenSCAD CLI，因此未验证加工网格导出。

```bash
export PYTHONPATH=.:rlx
PY=rlx/.venv-microduck/bin/python
$PY -m microduck_arm_v1 build
$PY -m microduck_arm_v1 check
$PY -m microduck_arm_v1 torque --out artifacts/my-v1/torque --samples 500
$PY -m microduck_arm_v1 workspace --out artifacts/my-v1/workspace.json --samples 10000
$PY -m microduck_arm_v1 evaluate --out artifacts/my-v1/eval --episodes 2 --max-steps 1500 --videos
$PY -m microduck_arm_v1.evidence controls --out artifacts/my-v1/controls
$PY -m microduck_arm_v1.evidence audit --evaluation artifacts/my-v1/eval/evaluation.json --out artifacts/my-v1/evidence
$PY -m microduck_arm_v1 demonstrate --case reach --mode fixture --episodes 8 --max-steps 250 --out artifacts/my-v1/demos
```

把 `demonstrate` 打印的数据集路径交给 `train-bc --dataset ... --epochs 20 --out ...`。小型残差训练使用 `train-ppo --case reach --mode fixture --steps 2048 --max-steps 250 --out ...`。使用输出的检查点调用 `export --checkpoint ... --out ...`。策略评估 API `learning.evaluate_policy` 要求明确独立种子；最终训练冒烟脚本提供可执行实例。

自由底座任务失败时，`evaluate` 返回2是预期结果，不能隐去；缺少实测硬件证据时 `hardware-check` 返回2；制造要求未解决时 `generate_engineering.py --manufacture-check` 返回3。macOS 离屏渲染可能需要图形服务授权。

## 下一步工程门槛

1. 实测原机、舵机、安装界面、脚底、电池和电子件，替换质量/惯量占位，解决电机网格尺寸冲突。
2. 修改承力安装件和无碰撞运动路径。当前取物轨迹发生自碰撞，不可通过关闭碰撞制造通过结果。
3. 先实现整机站立平衡，再实现步态并重跑带载自由底座任务。
4. 完成孔位、公差、传动和结构计算，选定并验证电源保护、储能和线束；当前逻辑图不能作为加工授权。
5. 增加传感器估计、时序资格、安全的 robotd/控制中心集成及独立实物验收。旧 viewer 机械臂页面不代表已支持本次64/15契约。
6. 按预登记任务/扰动条件重新训练并比较教师、BC、PPO，再进入受约束台架、站立、接触和低速搬运实测。

本交付终点是可复现原型和明确的失败证据，**不声明全部技能通过、实体能力额定值、加工放行或硬件安全认证**。
