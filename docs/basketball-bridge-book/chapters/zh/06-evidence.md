# 看证据，不看庆祝语

## 建立证据链

![证据制作示意图：保留的运行记录经过结构检查与 CSV 提取，生成实测曲线或复制的连续帧，最终成为绑定哈希的教材资源。这不是实测轨迹。](assets/diagrams/schematic-evidence-pipeline.png)

训练程序完全正确地执行，也可能学错行为；reward 上升，平衡可能变差；ONNX 可能不同于训练 actor；viewer 可能加载错误场景。因此最终结论要经过多层独立检查：模型和输入正确、参数确实更新、导出一致、无辅助评估诚实、实际观看画面。

**独立**很重要。如果把“成功”定义成 reward 高，再展示同一个 reward 证明成功，只是把一个定义检查了两次。接触合法性、无摔倒时长、方向位移和完整过桥，才是对用户物理目标的独立验证。

## 阅读真实 reward 和 loss 曲线

![真实篮球训练 loss，明确标出历史 reward 缺失。两个策略是独立 warm start，不是一条连续训练曲线。](assets/evidence/basketball-training-curves.png)

篮球固定指令 `summary.json` 有 100 条更新，混合指令有 150 条。包含 total loss、policy loss、value loss、entropy、approximate KL 和 gradient norm，**没有保存历史每次更新 reward 或 episode return**。这是证据缺口，不能补画一条漂亮上升曲线。图中的 reward 缺失提示是有意保留的。

两次训练都从同一个成熟 actor 起步，并新建 critic 与 optimizer。如果把固定指令后面接上混合指令，画成累计曲线，就伪造了连续训练关系。正确横轴分别到 102,400 与 153,600 个 transition。固定指令每次更新对应 $8\times128=1024$ 步新数据。

本版另附的篮球 reward 诊断图记录现有策略确定性执行时的逐步奖励。它是**新增诊断轨迹**，不是找回的历史训练曲线，也不是进一步优化的证据。CSV 与凭据记录策略、条件与实际时长，不能把它拼接到旧训练历史里。

![新增确定性篮球 reward 诊断，不是历史 PPO 训练奖励；源策略没有改变。](assets/evidence/basketball-reward-trace.png)

![真实独木桥 pilot 日志：每步平均 reward、完整回合 return 与 PPO loss 分开呈现。](assets/evidence/bridge-training-curves.png)

独木桥日志包含 collection、update、completed episode 事件。collection 的 `mean_reward` 是每个 transition 的平均奖励；`mean_raw_return` 是一个回合奖励之和，再对该事件中结束的回合求均值。回合更长，即使每步表现没变，return 也可能更高。update 的 `mean_loss` 是优化器小批次目标的均值。这三者不可互换。

独木桥 total／value loss 图使用对数纵轴：相同纵向距离表示相同倍数变化，而不是相同差值。1,000 降到 100，与 100 降到 10，在对数轴上距离相同。负 policy loss 不能放到普通对数轴，应单独显示。

Policy loss 很小不等于技能学会。优势归一化使均值接近零，而概率比接近一时，平均 surrogate 本来就可能接近零。早期 value loss 较大也不意外，因为 actor 成熟但 critic 是新建的。连续分布的 entropy 可以为负，见 PPO 章。Target KL 也不能保证单次优化绝不超限。

**好图注说明五件事：**哪次运行、哪个原始字段、横轴单位、是否变换／平滑、不能证明什么。本书 CSV 保留未经平滑的值，重新生成图不需要重新训练。

## 确定性评估：究竟通过了什么？

![记录中的评估结果，而非训练分数。篮球持续平衡通过；指令滚动与合格过桥没有通过。](assets/evidence/evaluation-outcomes.png)

| 实测实验 | 结果 | 正确理解 |
|---|---|---|
| 篮球固定指令微调，六次 60 秒 | 6/6 持续存活 | 名义条件下自由球平衡已验证。 |
| 同候选，三次前进指令 | 0/3 滚动通过 | 有运动不等于可靠受控滚动。 |
| 篮球混合指令微调 | 6/6 存活、0/3 滚动 | 独立候选，不是固定指令的后续步数。 |
| 最终独木桥 pilot | 0 次合格过桥 | 训练和导出工作正常；物理任务尚未完成。 |
| 七案例 API／Studio 检查 | 七场景可访问和显示 | 产品集成，不是七张技能证书。 |

直接篮球报告用种子 101、202、303；后来的 Studio 评估用 101、102、103。两者都有零指令和前进指令，但属于不同证据集。不能用另一个报告的截屏重标种子。直接报告的最小脚—球接触覆盖率约 99.93%，最大倾斜约 4.74 度；只描述该报告对应策略和条件。

**历史源码边界：**保留的直接篮球报告早于当前源码快照。报告中 evaluator／environment 哈希前缀为 `fcfa65`／`2eda2a`，本版提供的对应文件为 `bbe735`／`6bff54`。我们保留旧报告，但不把它说成用当前这些源码字节重新执行的结果。新增 10 秒 reward 诊断绑定当前环境哈希，却不能替代六次 60 秒验收。验证当前快照时，应按书中命令输出到新目录，保留新的哈希。复现协议和逐字节重建历史，是两种不同承诺。

固定指令 run 的前进跟踪平均绝对误差约 0.138 米／秒，目标为 0.15 米／秒；混合指令约 0.160 米／秒。即使平衡良好，这仍是较差的指令跟踪。与源策略约 0.140 米／秒相比的一点差异，不能证明稳健改善。

独木桥最大进度 1.174916 米相对于出发点计算，**不能说“桥只有 1.10 米，已经走了 1.174916 米，所以完成了”**。鸭子必须离开起点平台，产生合法木板接触，双脚到终点，避免非法支撑，并满足完整时长。最终评估含三个并行环境，总计 3,000 transition；reset 产生多次尝试和末尾片段。这些片段不是新增的合格 20 秒回合。

名义条件下三次或六次成功，不代表广泛可靠性。当前测试没有穷尽摩擦、质量、绳索刚度、指令方向、推力、传感器噪声、执行器温度、制造差异或硬件条件。左右侧向和 yaw 指令评估还没有实现。应该报告实际测了什么，而不是编造总体成功率。

## 视觉验证：看连续画面

![种子 101 的实测零指令篮球序列。这证明该次持续平衡，不证明受控行走。](assets/evidence/observed-basketball-zero-command-contact-sheet.png)

不要只看鸭子是否在橙色球上方。躯干是否直立？脚是否真正支撑？地面或身体其他部位是否在提供支撑？纹理转动是否与运动一致？时钟是否连续、没有 reset？相机是否跟随得太紧，以致漂移被隐藏？

![实测前进指令尝试。鸭子保持平衡、球在动，但方向滚动门槛未通过。](assets/evidence/observed-basketball-forward-command-contact-sheet.png)

连续帧拼图仍是视频采样，可能漏掉快速抖脚或帧间事件。不确定时，按原帧率看完整 MP4，或以 50 Hz 控制频率渲染。结合接触与终止数据。接受技能前，“看起来成功”应与“数值满足规则”一致。

![最终独木桥 pilot 诊断序列。20 秒视频含 reset，不是一次不间断成功过桥。](assets/evidence/observed-bridge-rollout-contact-sheet.png)

保留的独木桥视频长 20 秒，包含两次 reset，后续一次尝试走到木板中途。Reset 将机器人放回新出生状态，不能隐藏后算成连续进度。本书保留失败序列，是因为看清在哪个接触转换处失去平衡，对课程设计很有价值。

视频保留在工作区，未嵌入 PDF：

- `docs/basketball-showcase/evidence/render/cmd-p0_000-p0_000-p0_000/seed-101/rollout.mp4`
- `docs/basketball-showcase/evidence/render/cmd-p0_150-p0_000-p0_000/seed-101/rollout.mp4`
- `rlx/runs/studio/bridge/bridge-studio-02/render/ep0.mp4`

## 在 Duck Viewer 运行两个场景

![实测七案例 Studio。卡片可用与技能验收是不同概念。](assets/evidence/observed-seven-cases.png)

七案例是 **Dance、Swing、Running、Stilts、Backflip、Basketball、Bridge**。篮球在滚动未通过时应标明 balance-only；独木桥不能把试训视频变成成功过桥预览。

已有合适服务时优先使用，不要为了占一个端口停掉别人的服务。需要新服务时，每个长期进程放一个独立终端，使用空闲端口：

```bash
export MICRODUCK_STUDIO_PYTHON_DIRECT="$PWD/rlx/.venv-microduck/bin/python"
cd duck-viewer
node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 -p 63317
```

第二个终端从工作区根目录执行：

```bash
cd microduck_local
MICRODUCK_ACTUATOR=bam .venv/bin/duck-lab --port 8788 \
  runs/first-gait ../microduck/policies/alpha_walking.onnx
```

第二条命令需要 harness 自己的虚拟环境和基线路径。新仓库若没有 `runs/first-gait`，省略该位置参数，只使用现成 walking ONNX。必要时在 `microduck_local` 内运行 `uv sync --python 3.12` 建立环境。每策略场景参数决定篮球或独木桥；默认执行器环境变量不能替代场景路由。

打开 `http://127.0.0.1:63317`，选择实验和导出策略，检查 live 场景。浏览器能打开还不够：保存的预览视频与实时物理场景是两种不同界面。

![真实 live 场景：左侧篮球，右侧独木桥、平台和支架。](assets/evidence/observed-live-scenes.png)

此前两个集成问题很有教学价值。其一，lab 提取 group-2 视觉几何，物理对象若只在 group 0，就会“存在但不可见”。其二，CLI 加载策略时必须传入该策略的自定义环境参数，否则名字叫 bridge 的策略也可能运行在错误场景。修复应是正确路由和视觉提取，而不是添加 CSS 装饰。LSTM 缓存也必须保证每只鸭子的 h/c 私有。

服务运行时可以使用 readiness 和 API 检查：

```bash
bash restart-lab.sh --readiness-only
node scripts/verify-seven-cases.mjs
node scripts/verify-balance-studio-api.mjs
```

API 验证器调用真实评估／渲染并可能创建证据；可选 `--smoke-train` 会运行真实微型训练，不是只读检查。这些与静态教材构建不同，可能需要浏览器／图形权限和时间。

## 尚未完成技能的研究计划

先问：是否有任何 rollout 曾出现目标行为？如果完全没有，换奖励系数无法教会从未访问过的状态。独木桥应研究早期出生练习、木板稳定辅助、落脚位置、起点离台过渡；篮球则分别研究保留平衡、前进跟踪、侧向／yaw 覆盖。

冻结验收规则，每次只改一种训练条件。记录每次尝试的辅助程度，评估时关闭辅助。比较源策略、零动作基线和新策略。站在起点平台的基线只证明站平台，不证明平衡于木板；中途出生的过桥只证明探索中出现行为，不证明从起点完整成功。

不要承诺“再训练多少步一定成功”。训练时间取决于探索、动力学、初始化和优化，而不仅是总步数。有价值的下一次实验也可能失败，却准确暴露课程必须解决的接触转换。

## 重建并检查教材

```bash
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/prepare_assets.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/build_book.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/validate_book.py
```

构建使用已有依赖，不安装软件，不重新训练策略。Markdown 是可复制源码版；PDF 增加代码换行、页码、公式和目录。`supporting-source.zip` 与 `source-manifest.json` 绑定实现快照。验证报告记录语法、源码／图像哈希、PDF 页数／文本检查和残留告警。

## 综合实践考核

1. 为什么 61 维观测不等于 61 个独立传感器？
2. 导出归一化在哪里执行？改分母为什么破坏 parity？
3. 算出一批 rollout 的 transition 数，区分仿真时间与现实用时。
4. 为什么 timeout 需要 value bootstrap，却不能让 GAE 跨 reset 传播？
5. 指出一张实测 loss、一张实测 reward、一张示意图。
6. 分开报告篮球平衡和滚动，并列出种子和时长。
7. 为什么独木桥最大进度不能证明过桥？
8. 不修改他人策略地启动两个 viewer 场景。
9. 哪些外部资产阻止从空文件夹离线复现？
10. 提出一个不改验收门槛的有界后续实验。

**参考答案：**观测含派生重力、指令和历史动作；归一化属于 actor 合同；总步数等于环境数乘每环境步数；超时不代表物理失败，但 reset 必须截断优势链；图表需来源；平衡通过而滚动失败；进度含进桥前接近距离；按策略路由场景且独立保留循环记忆；现成权重、网格和依赖仍为前提；改变一个课程或物理条件，保持确定性完整时长评估不变。

# 参考文献与来源

本地实现快照决定本书案例实际执行什么。算法论文和仿真器文档解释原理，但不证明两个策略学会技能。

1. John Schulman、Filip Wolski、Prafulla Dhariwal、Alec Radford、Oleg Klimov，**Proximal Policy Optimization Algorithms**，2017，arXiv:1707.06347。PPO 原论文，解释裁剪 surrogate 和重复利用采样。本版核对官方记录：`https://arxiv.org/abs/1707.06347`。
2. John Schulman、Philipp Moritz、Sergey Levine、Michael Jordan、Pieter Abbeel，**High-Dimensional Continuous Control Using Generalized Advantage Estimation**，arXiv:1506.02438。GAE 参考；本书另外解释仓库中 termination／truncation 掩码。`https://arxiv.org/abs/1506.02438`。
3. **MuJoCo XML Reference**，官方文档：body、free joint、geom、material、spatial tendon、springlength 与 limit。应对照所装版本；本快照是 MuJoCo 3.10.0。`https://mujoco.readthedocs.io/en/stable/XMLreference.html`。
4. `microduck_local/AGENTS.md`：固定观测合同、物理课程、导出—评估—观看的本地训练纪律。
5. `microduck-playground/experiments/basketball/`、`microduck-playground/artifacts/basketball/`：用户提供的设计和成熟策略，含来源与许可说明。
6. `microduck-playground/experiments/swing/README.md`：悬挂支撑参考；独木桥不是简单改名的 swing。
7. `docs/basketball-showcase/RESULTS.md`、`docs/bridge-showcase/RESULTS.md`：保留的本地结果及限制。
8. `assets/provenance.json`、`assets/SHA256SUMS`、`environment.json`、`source-manifest.json`：图像／数据、运行环境和代码的机器可读指纹。

本书没有整篇复制外部论文。公式、说明和代码讨论以本地实现为中心。复制的现有源码保留原文和相应许可说明；教材快照不改变第三方许可。
