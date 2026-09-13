# MicroDuck arm v1-B BOM / 物料清单

> **ENGINEERING PROTOTYPE NOT FABRICATION RELEASE**  
> **No purchasing or substitution approval / 不代表采购或替代批准**

Quantities are for one arm. A named candidate is not an approved assembly until
all compatibility and release gates pass.

数量按单臂计算。候选型号只有在兼容性和放行门槛全部通过后才可成为批准组件。

| ID | Qty | Candidate or requirement / 候选或要求 | Status / 状态 | Required evidence / 必需证据 |
|---|---:|---|---|---|
| M1-M5 | 5 | ROBOTIS `XL330-M288-T`, 18 g each | Candidate exact variant / 精确型号候选 | Purchased label, readable identity/firmware, unique bus ID, connector inspection, measured fit |
| TRUNK-BLANK | 1 | Central trunk mount, 20 g target | Design target, unmeasured / 设计目标，未测 | Released interface, weighed article, fixed-root fixture load review |
| ARM-CARTRIDGE | 1 | Complete cartridge, 160 g target | Design target, unmeasured / 设计目标，未测 | Complete weighed assembly and center-of-mass record |
| LINK-BLANKS | 1 set | 55 mm and 50 mm axis-distance solid blanks | Concept only / 仅概念 | Official mating definition, coupon fit, material/process/load review |
| WRIST-GRIPPER | 1 | 30 mm wrist-TCP; equal gears ratio -1; jaw centers `y=+/-12 mm`; 30 mm jaws; left `[-0.32,0] rad`, right `[0,0.32] rad`, `right=-left` | Concept only / 仅概念 | Gear module/teeth/backlash/material, shaft support, retention, pinch review, grip calibration |
| U2D2 | 1 | ROBOTIS U2D2, bench TTL candidate | Bench candidate; mobile TBD / 台架候选，移动方案待定 | Correct TTL port, stable enumeration, communication test, mobile architecture review |
| HUB1 | 1 | Compatible TTL/power hub | Unselected / 未选型 | Exact part, topology, current path, branch protection, connector ratings |
| BAT1 | 1 | Separate arm battery source for regulated 5 V domain | Unselected / 未选型 | Chemistry, cells, capacity, fault current, enclosure, charger and transport controls |
| BMS1 | 1 | Pack-compatible BMS/protection | Unselected / 未选型 | Exact pack compatibility and voltage/current/fault/temperature ratings |
| F_MAIN | 1 | Main arm feeder protection | Unselected / 未选型 | Part number, rating/curve, conductor/connector coordination, measured transients |
| F_M1-F_M5 | 5 | One protected VDD branch per actuator | Unselected / 未选型 | Part numbers, ratings/curves, branch fault verification |
| SW_DC | 1 | Manual DC disconnect or appropriate power pole | Unselected / 未选型 | DC interrupt rating at measured source fault conditions; reset behavior |
| HARNESS | 1 set | GND/VDD/DATA harness with strain relief | Unselected / 未选型 | Pin-by-pin continuity, wire/insulation/flex ratings, polarity keying, bend/strain review |
| FIXTURE | 1 | Fixed-root bench fixture | Outside released CAD / 不在放行 CAD 内 | Table/fixture dimensions, fasteners/clamp, overturning/load test, guarded workspace |

## Provisional simulation mass assumptions / 临时仿真质量假设

The following values may be retained in a model only as **unmeasured
placeholders**. They are not BOM facts, procurement specifications, measured
mass, payload evidence, or physical validation:

下列数值只能作为模型中的**未测量占位值**保留。它们不是 BOM 事实、采购规格、
实测质量、载荷证据或实体验证：

| Modeling placeholder / 建模占位项 | Provisional mass / 临时质量 | Provisional center (m) / 临时中心 |
|---|---:|---|
| Retained electronics / 保留电子件 | 70 g | `[-0.010, 0, 0.018]` |
| Separate arm pack / 独立手臂电池包 | 80 g | `[-0.040, 0, 0.015]` |
| Power hardware / 电源硬件 | 25 g | `[-0.020, 0, 0.034]` |
| Sensor shell / 传感器外壳 | 8 g | `[-0.020, 0, 0.059]` |

Every model and report using these values must label them
`unmeasured_placeholder`. Floating-mode conclusions must be rerun after actual
component selection, weighing, center-of-mass measurement, and inertia update.

所有使用这些数值的模型和报告必须标记 `unmeasured_placeholder`。实际选型、称重、
重心测量和惯量更新后，必须重新执行浮动模式分析。

## Non-substitution and prohibited connections / 禁止替代与连接

- Do not substitute M077, XL320, XL430, XC330, or another voltage/protocol
  variant without a new design review.
- Never connect `ARM_5V` to original MicroDuck battery positive or body VDD.
- Never connect arm TTL DATA to the original body actuator DATA bus.
- U2D2 is a communication candidate, not proof of servo power delivery.
- No fuse, connector, BMS, pack, or disconnect rating is inferred from a
  marketplace title or arithmetic stall-current sum.

- 未经新设计评审，不得以 M077、XL320、XL430、XC330 或其它电压/协议型号替代。
- 禁止将 `ARM_5V` 接到原 MicroDuck 电池正极或机身 VDD。
- 禁止将手臂 TTL DATA 接到原机身执行器 DATA 总线。
- U2D2 只是通信候选，不证明能够给舵机供电。
- 不得根据网购标题或堵转电流算术和推断保险丝、连接器、BMS、电池包或断电器额定值。
