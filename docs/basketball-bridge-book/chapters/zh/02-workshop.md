# 搭建可复现的实验室

## 物料：仿真资产不是实体采购清单

下面是**仿真物料表**，说明复现软件实验所需的对象与文件，不是经过认证的机械图纸、采购建议，更不是把昂贵机器人放到晃动木板上的施工规范。

| 项目 | 数量 | 用途与限制 |
|---|---:|---|
| Microduck 全碰撞 MJCF 和引用网格 | 每场景一个模型 | 14 个驱动关节；使用配套 XML，不随意换机器人。 |
| 自由篮球 | 一个 | 半径 0.12 米、质量 0.62 千克；碰撞球体与贴图网格分开。 |
| 篮球 OBJ、PNG | 各一份 | 使用 playground 资产；外观不同于接触物理。 |
| 悬挂木板 | 一块 | 长 1.10、宽 0.13、厚 0.03 米，质量 0.45 千克。 |
| 绳索 tendon | 四根 | 物理约束和支撑，不只是装饰线。 |
| 起点、终点平台 | 两个 | 固定支撑，明确出发与到达区域。 |
| 可见支架 | 场景组件 | 表示锚点位置；自身不参与碰撞。 |
| 地面、灯光、相机 | 一组 | 地面接触需要被判失败，不能被镜头遮住。 |
| 已训练篮球 actor 与参考 ONNX | 一对 | 实测本地路线所需的成熟初始化策略。 |
| 现成 walking ONNX | 一份 | 用于精确迁移独木桥 actor 和归一化。 |

本书没有测量真实木材等级、绳索断裂载荷、紧固件规格、电池限制、舵机热限制和安全系数。不能从仿真尺寸推导出这些结论。实体测试还需要工程安全绳、隔离区、急停、力矩／电流限制、专业监督和独立 sim-to-real 验证。本地训练器是原型工具，不是部署许可。

## 计算机和软件前提

记录中的路线使用 Apple Silicon Mac。MuJoCo 在 CPU 上推进物理场景；篮球 PPO 使用 PyTorch CPU；独木桥 PPO 使用 MLX／Metal。“不需要 CUDA”**不等于**“所有环节在任何纯 CPU 操作系统上都可原样执行”。本版不声称完成 Linux 上独木桥 learner 的全新复现。

使用 Python 3.12，而不是 PATH 中恰好排第一的 `python`。Node/npm 运行 viewer；`uv` 管理 Python 环境。Pandoc、XeLaTeX、`pdfinfo`、`pdftotext`、`pdftoppm` 用于生成和检查教材，训练不需要它们。Pillow 和 Matplotlib 生成图表，不生成机器人动作。

实测 Python 包版本自动记录在 `environment.json`。它是**环境清单**，不是经过全平台测试的依赖锁。`source-manifest.json` 用 SHA-256 绑定每个源文件，因为仅靠 Git commit 无法包含尚未提交的新技能实现。

## 获取工作区，而不只是 PDF

向项目维护者取得完整工作区，包括 `rlx`、上游兄弟仓库，以及用户提供的 playground 参考。不要编造权重的公开下载地址。安装前检查：

```bash
test -f microduck_local/src/microduck_local/contract.py
test -f rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions.xml
test -f microduck/policies/alpha_walking.onnx
test -f microduck-playground/artifacts/basketball/checkpoint.pt
test -f microduck-playground/artifacts/basketball/policy.onnx
test -f microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.obj
test -f microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.png
command -v uv node npm
```

检查失败就停在缺失资产处。改 actor 架构或把归一化换成零，不会修复缺失权重。独木桥本身使用配套机器人资产；篮球额外需要 playground 的 OBJ／PNG。walking 基线及其他 harness 路径还可能需要上游模型仓库。

新环境采用仓库文档中的安装路线：

```bash
cd rlx
uv sync --python 3.12
uv pip install --python .venv/bin/python -e ../microduck_local
.venv/bin/python examples/ppo_microduck_studio.py --help
cd ..
cd duck-viewer
npm ci
cd ..
```

安装需要网络和可用的软件包。本书验证使用已有的记录环境，没有声称在另一台离线机器上完成全新安装。保留 lockfile；改变解析后的依赖前，比较版本并运行测试。不要跨机器复制虚拟环境。

实测解释器是 `rlx/.venv-microduck/bin/python`；上述安装生成 `rlx/.venv/bin/python`。如果使用新环境，请一致地替换各章命令路径：

```bash
PYTHON="$PWD/rlx/.venv-microduck/bin/python"
test -x "$PYTHON"
"$PYTHON" docs/basketball-bridge-book/tools/preflight.py
"$PYTHON" docs/basketball-bridge-book/examples/ppo_arithmetic.py
```

预检查文件和包版本，不导入 Metal，因此不能证明后续 Metal／图形上下文一定可用。受限沙箱可能允许读文件却拒绝 Metal 或 CoreGraphics。这是运行权限问题，不等于策略错误。

## 分清预检、smoke、训练和评估

| 阶段 | 最小有效证据 | 不能证明什么 |
|---|---|---|
| 预检 | 必要文件和包可定位。 | 动力学、绘图和策略表现。 |
| 合同测试 | 维度、reset、课程和评估规则在测试中正确。 | 训练技能可靠。 |
| Smoke 训练 | 少量 transition 能进入更新、checkpoint 和 ONNX。 | 技能习得或收敛。 |
| Pilot 试训 | 有界真实实验改变参数并记录指标。 | 未经评估的过桥或转向。 |
| 确定性评估 | 导出策略满足或不满足明确条件。 | 测试条件以外的鲁棒性。 |
| 视觉检查 | 接触、姿态、真实运动与指标一致。 | 未检查的种子或实体场景。 |

长期训练前运行针对性合同测试：

```bash
rlx/.venv-microduck/bin/python -m pytest -q \
  rlx/tests/test_basketball.py \
  rlx/tests/test_basketball_policy.py \
  rlx/tests/test_basketball_evaluation.py \
  rlx/tests/test_balance_adapter.py \
  rlx/tests/test_bridge.py \
  rlx/tests/test_bridge_evaluation.py \
  rlx/tests/test_bridge_bootstrap.py
```

这些测试覆盖所列边界，不代表仓库所有无关测试都通过。本次教材的当前测试与出版检查另存于 `verification/`。

## 第一步动作前，先看场景

本书环境探针编译两个模型，检查输入／输出维度，并各采四个有限数值 transition。它使用零动作，只验证构造，不加载已训练策略，也不证明平衡：

```bash
rlx/.venv-microduck/bin/python \
  docs/basketball-bridge-book/examples/probe_scenes.py
```

每个环境有私有的 MuJoCo 可变数据；共享不可变模型可以省内存，但不能共享循环记忆或机器人状态。viewer 可以让两只鸭子复用同一 ONNX session，但每只鸭子必须拥有独立 LSTM 记忆。

## 新实验必须使用新输出目录

例如 `rlx/runs/book/basketball-YYYYMMDD-HHMMSS`，把日期占位符换成新的运行标识。检查目录不存在再训练，绝不覆盖本书证据目录。

每次至少保存：完整命令、软件清单、源码与源策略哈希；checkpoint 及元数据；最终 ONNX 和一致性结果；完整 update／episode 日志；包含失败、种子、时长、辅助设置的评估 JSON；与被测策略绑定的视频、连续帧和渲染元数据。

Checkpoint 不自动等于完整优化器恢复。这两条路线都是迁移 actor，但新建 critic 和 optimizer。重跑可以复现协议，不一定得到逐位相同的浮点权重。数值库、版本、调度、仿真器变化都可能改变轨迹。正确承诺是可追溯实验与一致的验收规则，而不是无依据的“训练结果逐位相同”。

## 排错时不要悄悄改题

| 现象 | 首先检查 | 不应该做什么 |
|---|---|---|
| 篮球不可见 | OBJ／PNG 路径与 group-2 视觉几何。 | 用装饰动画代替真实碰撞球体。 |
| 模型能编译但 viewer 是平地 | 每策略环境参数和 lab 场景提取。 | 只凭策略名称声称桥已经显示。 |
| 导出动作不一致 | 归一化分母、观测顺序、h/c reset、ONNX 元数据。 | 因“看起来正常”忽略 parity。 |
| `No Metal device available` | 在获准的本地运行时执行 MLX。 | 暗中换算法，仍称同一实验。 |
| CoreGraphics 错误 | 图形上下文权限和渲染后端。 | 拿示意图冒充实测截图。 |
| Reward 高但不成功 | 完整时长、合法接触和视频。 | 没查探索缺口就加奖励系数。 |
| stdout JSON 格式不匹配 | 优先结构化日志和结果文件。 | 把解析不到的 loss 画成零。 |

**练习：**为什么既保留策略哈希又保留视频？**答案：**好看的视频可能来自另一个策略。绑定哈希的元数据把结论、checkpoint、确定性导出、评估和可见轨迹连成证据链。
