# MicroDuck single arm v1-C engineering baseline

> **ENGINEERING CONFIGURATION BASELINE - NOT FABRICATION READY**
>• 当前 MicroDuck＋单臂模型一共使用 15 个舵机（servo）。

   部位      数量    关节
  ━━━━━━━━  ━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   左腿         5    髋 yaw、髋 roll、髋 pitch、膝、踝
  ────────  ──────  ────────────────────────────────────────
   右腿         5    髋 yaw、髋 roll、髋 pitch、膝、踝
  ────────  ──────  ────────────────────────────────────────
   机械臂       4    底座 yaw、肩 pitch、肘 pitch、腕 pitch
  ────────  ──────  ────────────────────────────────────────
   夹爪         1    开合驱动
  ────────  ──────  ────────────────────────────────────────
   合计        15    10 个腿部＋5 个手臂/夹爪
   
> **工程配置基线 - 不可用于加工**

This directory is the controlled documentation package for the v1-C single-arm
engineering baseline. It describes what must be reviewed, measured, selected,
and tested before any fabrication or powered robot integration. It does not
claim that hardware exists or that any physical evaluation passed.

本目录是 v1-C 单臂工程配置基线的受控文档包。它说明在加工或机器人带电集成之前
必须完成的评审、测量、选型和测试。本文件包不声明实物已经存在，也不声明任何实体
评估已经通过。

## Scope

### Additional tennis-ball experiment / 新增网球实验

The replaceable **75 mm wide-gripper candidate** is documented in
`TENNIS-RETURN.en.md` and `TENNIS-RETURN.zh-CN.md`. The three-size geometric fit
screen passed; ground pickup and controlled placement back into the bin have
**not** passed (0/30 per morphology). Its CAD is conceptual, not fabrication-ready.
The original teacher free-base matrix now passes 120/120, but the exported
whole-body-residual ONNX matrix is 100/120; do not merge these different claims.
See `artifacts/microduck-arm-v1c/free-base-repair-v2/` and
`artifacts/microduck-arm-v1c/tennis-return/` at the repository root for raw evidence.

- v1-C has four pose joints and one gripper actuator: five arm actuators total.
- Candidate A is five `XL330-M288-T` units.
- Candidate B is three `XL330-M288-T` units plus `XL330-M077-T` at the wrist
  and gripper.
- Both candidates use the same documented 18 g mass per motor, so the five
  motors total 90 g in either candidate. No inertia, energy, or performance
  advantage is inferred from equal mass.
- Bus IDs `21` through `25` are provisional only.
- No mobile controller, pack, regulator, BMS, fuse, connector, harness, or
  E-stop is approved.
- Bench qualification defaults to an external regulated 5 V source. Shared
  robot-pack and separate arm-pack topologies remain studies.
- The user-provided category sum is 170 g; the controlled cartridge target is
  corrected to 160 g. The 10 g difference is not hidden: all categories must
  be reweighed and reconciled before release.
- The current simulation scene is `raised_tabletop_v1_c`, with tabletop top at
  145 mm and shoulder pivot at local +18 mm. It is distinct from the v1-B
  floor-level scene; v1-B task results do not transfer.
- Rectangular modeled contact pads are 24 x 5 x 8 mm. Their physical material,
  friction, compliance, contact stiffness, attachment, and wear are unmeasured.

## Controlled files

- `BOOK.en.md`: English engineering baseline.
- `BOOK.zh-CN.md`: Simplified Chinese engineering baseline.
- `ENGINEERING-BOM.csv`: traceable engineering BOM and release gates.
- `HARDWARE-SOP.md`: bilingual staged inspection, power, motion, load, and
  fault-test procedure.
- `TEST-MATRIX.csv`: planned `S0-S6` by `P0-P50` evidence matrix. Every entry is
  `NOT RUN` or blocked; it is not a result record.
- `RESULTS.en.md` and `RESULTS.zh-CN.md`: leader-owned result records. The
  baseline builder does not convert missing evidence into a pass.
- `build-book.sh`: builds the two baseline PDFs with local Pandoc/XeLaTeX.

Review diagrams are under `hardware/microduck-arm-v1c/engineering/`:

- `load-path.svg`
- `wiring-architecture.svg`
- `mechanical-dimension-overview.svg`

Every diagram is explicitly a concept review artifact, not a scale-controlled
manufacturing drawing or released schematic.

## Reproduce simulation evidence / 重现仿真证据

Use the existing workspace Python environment (MuJoCo, NumPy, PyTorch,
Stable-Baselines3, ONNX Runtime, Gymnasium, imageio and Pillow).
No hardware is accessed by these commands.

```bash
PYTHONPATH=.:rlx rlx/.venv-microduck/bin/python -m microduck_arm_v1c model --out hardware/microduck-arm-v1c/generated
PYTHONPATH=.:rlx rlx/.venv-microduck/bin/python -m microduck_arm_v1c evaluate --out artifacts/microduck-arm-v1c/evaluation-rerun --seeds 10
PYTHONPATH=.:rlx rlx/.venv-microduck/bin/python -m microduck_arm_v1c workspace --out artifacts/microduck-arm-v1c/workspace-rerun --samples 10000
PYTHONPATH=.:rlx rlx/.venv-microduck/bin/python -m microduck_arm_v1c.learning --help
PYTHONPATH=.:rlx rlx/.venv-microduck/bin/python -m microduck_arm_v1c.evidence --out artifacts/microduck-arm-v1c/videos-rerun
```

MuJoCo offscreen rendering requires access to macOS graphics services. Results
contain independent task-success and video-decode flags: a playable video is not
a passed task. Read `RESULTS.zh-CN.md` / `RESULTS.en.md` for current limitations.

MuJoCo 离屏渲染需要访问 macOS 图形服务。视频可播放不代表任务通过；台架通过不代表
自由站立、行走或真机通过。当前源模型为 166 g，超出 160 g 机械臂目标，尚未放行。

## Build books / 构建双语手册

From the repository root:

```bash
bash docs/microduck-arm-v1c/build-book.sh
```

Expected outputs:

- `MicroDuck-Arm-v1-C-Engineering.en.pdf`
- `MicroDuck-Arm-v1-C-Engineering.zh-CN.pdf`

The build uses the existing `pandoc`, `xelatex`, and `rsvg-convert` tools and
does not install dependencies.

## Authority and evidence boundary

The current design source is
`hardware/microduck-arm-v1c/spec/design.json`. Runtime interpretation is in
`microduck_arm_v1c/config.py`, `env.py`, `model.py`, and `safety.py`. Vendor
files and retrieval notes are in
`hardware/microduck-arm-v1c/vendor/SOURCE-MANIFEST.md`.

The older `docs/microduck-arm-v1/` package is v1-B reference material only.
Its dimensions were reused where the v1-C specification explicitly inherits
them. Its test outcomes, power assumptions, controller assumptions, and release
state are not v1-C evidence.
