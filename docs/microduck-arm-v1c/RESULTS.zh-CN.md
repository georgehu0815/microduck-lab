# v1-C 仿真验证结果

**尚未达到全部实验通过。** 软件回归测试和台架任务矩阵通过；自由基座操作、行走和实物放行仍未通过。禁止降低标准或把夹具支撑当作机器人自身平衡。

改进已落实：J1–J5 映射、A/B 电机候选、独立电源域研究、夹具和承力件 BOM、惯量和扭矩表、实际开爪判据、上电前新鲜心跳检查、源码变更中止以及端到端视频证据。混合电机没有重量优势；166 g 模型超出 160 g 目标。

学习结果仅覆盖台架 reach：BC 未通过，DAgger 仍有一例失败，残差 PPO 与教师均为 5/5，并不能证明 PPO 优于教师。机械臂负载表仅为种子 0 的台架筛查，不能宣称为实物承重能力。

## Test evidence / 测试证据

`298 passed, 11 warnings in 59.25s`

Ruff 和 Python 编译检查通过。独立 mypy 检查仍无法解析工作区的 `microduck_local.contract` 导入，因此不宣称全项目类型检查通过；实际运行的策略接口回归测试通过。

| 模式 | 案例 | 成功 / 总数 | 失败原因 |
|---|---|---|---|
| fixture | 绕障搬移 | 20/20 | PASS |
| fixture | 抓取放置 | 20/20 | PASS |
| fixture | 伸手定位 | 20/20 | PASS |
| fixture | 定点搬运 | 20/20 | PASS |
| free | 绕障搬移 | 0/20 | arm_environment, object_drop, phase_timeout |
| free | 抓取放置 | 0/20 | arm_environment, object_drop, phase_timeout |
| free | 伸手定位 | 8/20 | timeout |
| free | 定点搬运 | 0/20 | arm_environment, object_drop, phase_timeout |
| free | 收臂行走 | 0/20 | timeout |
| free | 行走搬运 | 0/20 | object_drop, robot_self_collision |

候选 A/B，种子 0–9，物体 10 g。fixture 是固定台架，free 是自由基座；台架结果**不能**用于整机平衡放行。桌面高 145 mm，不能与 v1-B 的低桌面任务直接比较。

## BC → DAgger → residual PPO

教师示范 8 回合（种子 0–7）；DAgger 采集 4 回合（种子 20–23）。初始 BC 和合并数据 BC 各训练 20 轮；PPO 训练 2,048 步（种子 40）。留出评估种子为 101–105，任务为候选 A 的台架 reach。ONNX 输出 5 个手臂动作，必须经过显式 15 动作适配器，不能直接替换原有 14 动作策略。

| 策略 | 留出集成功数 | 失败原因 |
|---|---|---|
| bc | 0/5 | {"arm_environment": 5} |
| dagger | 4/5 | {"timeout": 1} |
| ppo | 5/5 | {} |
| teacher | 5/5 | {} |

最后一轮原始指标（未平滑；字段名保留日志原文）：

```json
{
  "bc": {
    "epoch": 20,
    "raw_train_arm_mse": 0.0018554778071120381,
    "raw_validation_arm_mse": 0.005228061694651842
  },
  "dagger_merged_bc": {
    "epoch": 20,
    "raw_train_arm_mse": 0.0008629434159956872,
    "raw_validation_arm_mse": 0.0036217200104147196
  },
  "ppo": {
    "approx_kl": 9.514857083559036e-06,
    "clip_fraction": 0.0,
    "entropy_loss": -7.093945741653442,
    "phase": "training_end",
    "raw_loss": 0.05093790963292122,
    "raw_reward_sum": -1.2966566238901578,
    "timesteps": 2048,
    "value_loss": 0.104794941842556
  }
}
```

PPO 仍保留 IK 教师，手臂残差上限为 0.02 rad。与教师同为 5/5 不代表优于教师。BC 损失下降不等于闭环任务成功；DAgger 仍有一例失败。

![Raw learning curves / 原始训练曲线](../../artifacts/microduck-arm-v1c/training-curves.png)

## Workspace / 工作空间

10000 个采样目标位置； 3505 个 IK 可达目标；
3255 个同时满足可达和无碰撞条件。
通过平衡验证的工作空间：**尚未建立**。
质量、重心和完整惯量张量表来自解析模型估计，不是实测或加工放行的 CAD 数据。关节扭矩 CSV 包含 5,000 行筛查记录；合格连续扭矩仍未知，3 倍裕量门槛保持未通过。

## Payload screening / 负载筛查

仅测试候选 A、固定台架、种子 0，不能据此发布额定载荷或可靠性指标。

| 载荷 g | 案例 | 任务成功 | 失败原因 |
|---|---|---|---|
| 5 | 伸手定位 | True | PASS |
| 5 | 抓取放置 | True | PASS |
| 5 | 绕障搬移 | True | PASS |
| 5 | 定点搬运 | True | PASS |
| 10 | 伸手定位 | True | PASS |
| 10 | 抓取放置 | True | PASS |
| 10 | 绕障搬移 | True | PASS |
| 10 | 定点搬运 | True | PASS |
| 20 | 伸手定位 | True | PASS |
| 20 | 抓取放置 | False | object_drop |
| 20 | 绕障搬移 | False | object_drop |
| 20 | 定点搬运 | False | object_drop |
| 30 | 伸手定位 | True | PASS |
| 30 | 抓取放置 | False | object_drop |
| 30 | 绕障搬移 | False | object_drop |
| 30 | 定点搬运 | False | object_drop |
| 50 | 伸手定位 | True | PASS |
| 50 | 抓取放置 | False | robot_self_collision |
| 50 | 绕障搬移 | False | robot_self_collision |
| 50 | 定点搬运 | False | robot_self_collision |

## Videos / 视频

20 段完整 IK/状态机回合视频，另有 4 段实际运行 ONNX 的策略视频。
全部为仿真录像，同时保留成功和失败回合。每段均经过 ffmpeg 全片解码验证，并记录 ffprobe 元数据和 SHA-256。索引文件： `artifacts/microduck-arm-v1c/videos-final/video-manifest.json` and
`artifacts/microduck-arm-v1c/videos-learned/video-manifest.json`.

## Release / 放行

模型中的机械臂模块质量： 166 g;
目标： 160 g — **FAIL**.
监督器测试仅检查模拟输入，不是真实短路、反接或器件安全试验。硬件放行：**否**。加工、电池、连续扭矩、热性能、传感器/状态估计、移动控制和实体故障验证均仍待完成。

最终证据目录： `evaluation-final`, `engineering-final`, `videos-final`,
`videos-learned`, `learning/attempt-2` ，均位于 `artifacts/microduck-arm-v1c/` 下。
较早的根目录训练产物、`engineering` 和 `videos-attempt-1-interrupted` 仅保留为历史诊断，不属于当前放行证据。
