# WingPod Camera v2 — Engineering BOM / 双摄眼睛工程物料表

Revision: 2026-09-13. **Candidate configuration, not fabrication or procurement release.**

Soft appearance update: the generated BOM now names the cream face and sage-gray
lens trims, with peach cheeks and a honey beak as cosmetic finish proposals.
See `../CAMERA-V2-SOFT.md`. No camera module, physical mass, or finish is thereby
qualified; the part count remains 110. Historical case footage uses graphite v2.

柔和配色版：BOM 已同步奶油面板、灰绿镜框、桃色腮红与小黄嘴；仍为外观候选，
不代表相机选型、材料或真机已验证，物料行数仍为 110。历史案例视频保留深色版本。

## 1. Deliverables / 交付物

- `BOM.csv`: spreadsheet-ready complete parts inventory, including source, status and missing evidence.
- `BOM.json`: same inventory plus control order, candidate motor configurations and source hashes.
- `PARTS.md`: complete readable parts table.
- `consistency-check.json`: generated quantities and configuration checks.

The list covers one converted existing MicroDuck: base components, electronics,
five-actuator arm, wide tennis gripper, appearance covers, camera eyes, protection
functions and test fixtures. It is not Pollen's commercial manufacturing BOM.
Reuse rows, assembly-included functions, bench equipment and alternative designs
must not all be summed into one purchase order. Unknown screw counts and exact
supplier part numbers remain unknown; an STL is not a fabrication drawing.

覆盖一台改装MicroDuck的底座、电子器件、五执行器单臂、宽夹爪、外壳、摄像头眼睛、
保护功能与测试夹具。不是Pollen商业整机制造BOM。复用件、已集成功能、台架设备和可选架构
不可重复采购。未知螺钉数量和型号保留待定，不从STL猜测可制造图纸。

## 2. Main assemblies / 主要总成

| Assembly / 总成 | Qty / 数量 | Configuration / 配置 |
| --- | ---: | --- |
| Existing leg servos / 原腿部电机 | 10 | Five per leg: hip yaw/roll/pitch, knee, ankle; inspect actual XL330 gearbox labels |
| J1–J3 arm motors / 基座、肩、肘 | 3 | XL330-M288-T candidates |
| J4 wrist + J5 grip / 腕部及夹爪 | 2 | A: 2 × M288; B: 2 × M077; alternatives, not four motors |
| Load-bearing trunk interface / 承力机身接口 | 1 | Reuse chassis datums; custom interface pending measured mounting drawing |
| Upper / forearm links / 上臂、前臂 | 1 each | 55 / 50 mm joint-axis distances |
| Structural arm bearings / 机械臂承力轴承 | 8 | Two per J1–J4 shaft; sizes/ratings not selected |
| Jaw bearings / 夹爪轴承 | 4 | Two per left/right shaft; sizes/ratings not selected |
| Gripper gears / 夹爪齿轮 | 2 | Candidate module 1, 24 teeth each, 24 mm axis spacing; full tooth profile unresolved |
| Wide jaws + pads / 宽指及接触垫 | 2 + 2 | 75 mm modeled opening; wide jaws replace narrow v1-C jaws |
| Joint shafts / 转轴 | 4 + 2 | Four main arm shafts and two jaw shafts; exact fits/retention unresolved |
| Horns/couplers / 舵盘及联轴件 | 4 + 1 function | J1–J4 horns and J5 transmission; do not count J5 coupler twice |
| WingPod cover envelopes / 外观罩壳 | 10 | Source-derived envelopes; not ten added solid mass blocks |
| Active camera modules / 实际摄像头模块 | 2 target | Exact miniature modules unselected; two real optical channels required |
| Eye bezel / 双眼面罩 | 1 | 28 × 17 × 5 mm proposed outer envelope |
| Lens trim / 镜头装饰圈 | 2 | Visual rings, not two extra purchased optical lenses |
| Amber LED + driver + diffuser / 状态灯组件 | 1 each | Hardware part numbers and electrical values TBD |
| Radxa compute / 计算主板 | 1 reuse | ZERO 3W local reference; installed board/storage revision to inspect |
| IMU / 惯性传感器 | 1 function reuse | Usually assembly-owned; no assumed new separate sensor board |
| Main pack / 主电池 | 1 reuse | Read installed label, measure mass, verify charger/protection |
| Arm regulated supply / 机械臂稳压电源 | 1 architecture | Protected independent domain; shared-pack branch vs separate pack unresolved |

The simulated robot has **15 actuators = 10 legs + 5 arm**, not 15 plus the old
neck/head motors. Four arm pose joints plus one gripper actuator is **4-DOF pose
control + gripper**, not a general 5-DOF pose arm plus a sixth gripper motor.
Inherited actuator family geometry does not establish exact installed M288/M077
SKUs. Keep A as the simulated default; B is an A/B qualification candidate.
Both motor variants have documented 18 g mass; swapping variants does not by
itself reduce the five-motor 90 g nominal mass. [R3, R4]

仿真共15个执行器，不额外叠加已被替换的旧头颈电机。机械臂是4个姿态关节加1个夹爪执行器。
A/B电机方案不能重复采购；两种电机同为18g，混用型号本身不减轻五电机90g标称质量。

## 3. Camera fit decision / 摄像头适配结论

**Do not buy two full Pi Camera Module 2 boards as a claimed drop-in eye kit.**
The official Camera Module 2 reference uses Sony IMX219, 8 megapixels,
3280 × 2464 pixels, approximately 25 × 24 × 9 mm module dimensions, 3 g and
a standard 15-pin camera connector. Its complete board/lens assembly cannot fit
inside a 28 × 17 × 5 mm outer enclosure. The rendered lens spacing is 13 mm;
it is not proof two complete boards can sit side by side at that spacing. [R1]

Radxa ZERO 3W documents one four-lane CSI interface on a 22-pin connector and
supports Camera V2. A physically compatible, correctly wired cable is required;
15-pin and 22-pin FFCs cannot simply be swapped. One direct CSI input does not
provide two simultaneous camera feeds. A passive Y cable is not a solution. [R2]

Local MicroDuck bring-up records IMX219 with the board's Pi Camera V2 overlay.
This makes it a useful **single-camera bench/software reference**, not a
qualified miniature stereo module for the WingPod chest. [L3]

Two paths remain design-review options:

1. Preserve the small face: select real miniature camera/lens modules with
   remotely mounted electronics, then verify signal integrity, cable routing,
   capture support, synchronization and actual assembly fit. No matching module
   has been qualified here; do not invent a vendor part number.
2. Preserve a standard ready-made camera assembly: redesign/enlarge the enclosure
   around its real drawings. This would be a new mechanical revision, not the
   unchanged v2 render. Recheck occlusion, mass/CoM/inertia and balance.

不应将两块完整Pi Camera Module 2作为现有小眼睛外壳的即插即用采购方案。
该模块尺寸超过外壳高度和厚度；Radxa也只有一个直接CSI入口，不能用被动分线实现双路同步。
目前可先复用MicroDuck的单摄软件链路做台架验证，但最终双摄仍需小型模块/桥接方案，
或增大并重新验证外壳。**没有把第二只眼睛偷偷改成装饰假镜头。**

## 4. Wiring concept / 接线概念

```text
Qualified main pack + pack protection + main disconnect
  ├── Existing protected leg-power/control system (10 servos)
  ├── Qualified logic supply → Radxa + retained IMU/control electronics
  │                            └── selected camera interface → LEFT + RIGHT RGB
  │                                 (camera power per actual module specification)
  └── Protected arm feeder → reviewed disconnect/regulator → regulated ARM supply
                                                            └── J1 … J5, parallel power

Radxa → qualified USB/UART-to-DXL interface → TTL multidrop J1 … J5
LED → selected current-limit/driver → qualified logic rail
```

The five XL330 candidates specify 3.7–6.0 V input, with 5 V recommended. This does
not rate the entire wiring system or converter. U2D2 provides communications,
not servo power. [R3–R5] No raw battery voltage goes to an XL330 or an unqualified
camera input. Do not power cameras from the noisy servo domain by default.
Camera and servo connector pin numbering, wire lengths/gauges, fuse curves,
regulator capacities and LED resistor values require selected real parts and
measured loads; no final PCB/netlist is released by this document.

Separate arm domain does not necessarily mean another battery. A shared-pack
arm branch remains a candidate until simultaneous leg/arm/compute/camera load,
brownout, regulator thermal behavior and fault containment are qualified.
The inherited `np_f970` mesh name is **not a battery purchase specification**.
The dedicated-pack visual placeholder is not approval to add that pack's mass.
Existing protection/IMU functions may already be on a board; do not double count.

相同电机沿用同一官方外形与接口基准；供电域分离不等于增加独立电池。
摄像头不接原始电池或未经验证的电机电源，保险丝、稳压器、线规和针脚需实物选型后确定。
U2D2只负责通讯，不负责给舵机供电。电池名称不能从网格文件名推导。

## 5. Mechanical and acceptance gates / 机械及验收门槛

1. Inventory existing robot: all labels, connectors, board revisions, battery,
   servo IDs, baseline measured mass/CoM and gait. No new procurement from filenames.
2. Keep arm load path structural: jaws → supported shafts/bearings → wrist →
   links → shoulder/base support → trunk → legs. Cosmetic covers carry no arm loads.
3. Use the **tennis wide** geometry, not the narrow tabletop gripper: 75 mm open
   pad gap; wrist-to-TCP 55 mm; 55 + 50 + 55 = 160 mm summed geometric reach.
   This is not a qualified safe reach/payload rating. [L1, L2]
4. Fit actual sensor boards, lenses and cables in CAD before printing final shells.
   Include connector height, walls, screw access, cable bend radii and thermal path.
   Cosmetic honey feathers/caps/eye highlights are styling features, not an extra
   set of load-bearing links or optical parts.
5. Measure complete mass, CoM and inertia including wide jaws, covers and eyes.
   The inherited 160 g arm budget is not proof this expanded assembly meets 160 g.
6. Bench-check single camera first, then both real streams: enumeration, sustained
   capture/encode, timestamps, calibration if stereo is required, thermal and power.
   Start at a bounded low capture mode; do not assume sensor maximum resolution
   equals simultaneous real-time encode performance.
7. Validate camera visibility across stand, ground reach, carry and release; the
   home pose already shows some arm occlusion. Gate full sensor coverage explicitly.
8. Repeat free-base payload/balance/contact tests with the physical camera/shell
   mass and collision model. Then validate on a restrained real robot under a
   reviewed stop/fall/drop procedure. A bus timeout is not a complete safety system.

先盘点实物，再确定结构及选型；先台架单摄，再验证真实双路采集及功耗。
将相机和外壳真实质量、惯量及碰撞几何回填模型后重测平衡。当前视频仅增加渲染外观，
并不验证真实附加重量、制造强度、双目深度或视觉抓取。

## 6. Sources / 参考资料

Local source files are bound by SHA-256 in `BOM.json`. External references were
checked September 13, 2026; recheck vendor drawings against the purchased revision.

- [L1] `docs/microduck-arm-v1c/ENGINEERING-BOM.csv`: inherited arm hardware requirements.
- [L2] `hardware/microduck-arm-v1c/experiments/tennis-return.json`: wide jaws and low-bin fixture.
- [L3] `microduck/docs/project/media-bringup.md`: local IMX219/Radxa camera bring-up.
- [L4] `microduck_arm_design/wingpod.py` and `wingpod_camera.py`: appearance envelopes only.
- [R1] Raspberry Pi official camera documentation: `https://www.raspberrypi.com/documentation/accessories/camera.html`
- [R2] Radxa board and camera references: `https://docs.radxa.com/en/zero/zero3` and `https://docs.radxa.com/en/zero/zero3/accessories/camera`
- [R3] XL330-M288: `https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/` and `https://e-shop.robotis.co.jp/product.php?id=417`
- [R4] XL330-M077: `https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/`
- [R5] U2D2: `https://emanual.robotis.com/docs/en/parts/interface/u2d2/`
- [R6] Pollen commercial-product scope: `https://pollen-robotics.com/microduck/press-kit/`.
  Its commercial 15-DoF description does not establish this project's 10-leg + 5-arm layout.
- [R7] Separate Pollen Robot HAT board BOM, **not the entire MicroDuck BOM nor proof
  of this robot's installed board**:
  `https://raw.githubusercontent.com/pollen-robotics/elec_RPI_Robot_HAT/23eab11927f95ceca0dfa35bf182caeb7db39ea0/production/ASE01187-C1_elec_RPI_Robot_HAT_BOM.csv`

Regenerate from workspace root:

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python scripts/build_wingpod_camera_bom.py
```

**No hardware purchase, BOM substitution, circuit release, or fabrication was performed.**
