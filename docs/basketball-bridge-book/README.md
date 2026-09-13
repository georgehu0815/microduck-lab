# Microduck 平衡实验室 / The Microduck Balance Laboratory

**版本 / Edition:** 1.0, 2026-09-11. 主要训练证据 / Main training evidence: 2026-09-10.

**Author: George Hu.** 两版封面均使用篮球与悬挂独木桥的真实 live 场景截图；PDF 作者元数据同步署名。

面向只有高中数学基础的读者，完整解释篮球与悬挂独木桥的建模、仿真物料、软件环境、61 维观测／14 维动作、PPO／GAE／LSTM、固定奖励、物理课程、训练、ONNX、验收和 Duck Viewer。中文与英文为独立完整版本，不是互相夹杂的摘要。

For readers with high-school mathematics: modeling, simulation materials, setup, observations/actions, PPO/GAE/LSTM, rewards, physics-only curricula, training, export, evaluation and seven-case viewer integration. Both language editions contain the full core-source appendix.

## 阅读 / Read

| 版本 / Edition | PDF | Markdown |
|---|---|---|
| 中文 | [中文 PDF](microduck-basketball-bridge-zh.pdf) | [中文 Markdown](microduck-basketball-bridge-zh.md) |
| English | [English PDF](microduck-basketball-bridge-en.pdf) | [English Markdown](microduck-basketball-bridge-en.md) |

- [配套源码 ZIP / Supporting source ZIP](supporting-source.zip): 41 selected files, seven printed in full; not the complete runnable workspace or all external assets.
- [代码指纹 / Source fingerprints](source-manifest.json), [环境版本 / Environment inventory](environment.json).
- [图像、数据与来源 / Figures, data and provenance](assets/README.md).
- [出版检查 / Publication checks](verification/book-validation.json), [119 项合同测试 / Contract-test receipt](verification/contracts-junit.xml).

## 必须保留的事实边界 / Evidence boundaries

- 篮球：保留报告中 **6/6 次 60 秒持续平衡**；**0/3 指令滚动通过**。Basketball balance is demonstrated; command-following rolling and general steering are not mastered.
- 独木桥：最终 pilot 新增 **32,768 transitions，零次合格全程过桥**。Bridge simulation and training work, but accepted crossing remains unsolved.
- 成功篮球策略来自成熟源 actor 的本地 warm start，不是随机初始化训练成功。Both local routes use existing actors, new critics/optimizers, and explicit transfer/normalization contracts.
- “物料表”是仿真 BOM，不是实体桥承重施工规范；没有硬件验证。No hardware safety or deployment certification is implied.
- 篮球旧训练没有逐步 reward 历史，本书明确标注缺失；新增 500 步确定性 reward 图不是历史训练曲线。Historical losses and new diagnostic rewards are kept separate.
- 旧篮球验收报告早于当前源码快照；教材明确列出哈希差异。Historical evidence is preserved, not relabeled as a fresh acceptance battery on current source bytes.
- 七案例表示产品集成：Dance, Swing, Running, Stilts, Backflip, Basketball, Bridge；不等于七技能全部成功。

## 从哪里开始 / Starting point

从工作区根目录运行；如果按新环境文档安装，请把 `.venv-microduck` 替换为实际虚拟环境。

Run from the workspace root, using the actual environment path on your machine:

```bash
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/preflight.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/examples/ppo_arithmetic.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/examples/probe_scenes.py
```

然后依次读环境准备、PPO、篮球、独木桥、证据章节。案例章节给出完整训练、评估、渲染命令以及输出解释。训练必须用新目录，不要覆盖教材引用的保留结果。

The case chapters give complete train/eval/render commands and output interpretation. Training must use fresh output directories. The measured path requires the supplied basketball checkpoint, ONNX and mesh/texture assets, plus the walking policy; the book does not invent a public download URL for them.

## 重建 / Rebuild

The installed toolchain is Python 3.12 with Pillow/Matplotlib, Pandoc, XeLaTeX, macOS PingFang SC/Menlo fonts, and Poppler inspection tools. No packages are installed by these scripts. XeLaTeX may require ordinary macOS font-cache access outside a restricted sandbox; MLX contract tests require Metal access.

```bash
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/prepare_assets.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/build_book.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/validate_book.py
```

`build_book.py --markdown-only` skips PDF generation. `--language en` or `--language zh` builds one edition. `prepare_assets.py` uses saved raw evidence and does not rerun training. The optional `tools/collect_reward_trace.py` reruns the separately labeled ten-second deterministic diagnostic; rebuilding plots does not require it.

出版检查包括代码块语法、真实源文件和资源哈希、图片可读性、两种语言的 PDF 文本、全部页的文字边界、排版告警，以及可执行 PPO 算术。另有实际场景探针与 Metal 环境下 119 项合同测试凭据。并未重新执行全部历史训练，也未在新机器上从零安装依赖；这些边界在正文中明确说明。

Publication validation checks code-fence syntax, source/asset hashes, image readability, PDF text and page bounds, layout warnings, and executable PPO arithmetic. Separate receipts cover actual scene probes and 119 contract tests with Metal access. It does not imply historical retraining or a clean-room dependency installation.
