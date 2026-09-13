# MicroDuck arm v1-B hardware acceptance / 硬件验收

> **ENGINEERING PROTOTYPE NOT FABRICATION RELEASE**  
> **Current state / 当前状态: NOT RUN - FAIL CLOSED / 未执行 - 失败关闭**

This is an evidence template. Empty evidence means failure to release, not an
implicit pass. It contains no claim that hardware exists, was operated, or
passed.

本文件是证据模板。证据为空表示不能放行，不代表默认通过。本文不声明硬件已经
存在、运行或通过测试。

| Gate / 门槛 | Required evidence / 必需证据 | Current result / 当前结果 |
|---|---|---|
| Canonical configuration | Spec revision and generated artifact hashes match | NOT RUN |
| Exact actuator identity | Five purchased `XL330-M288-T` units identified; firmware and unique IDs recorded | NOT RUN |
| Mechanical mating release | Official revision-controlled geometry plus physical measurements, fit coupon, tolerance review | BLOCKED - TBD |
| Motor depth discrepancy | 29 mm visual mesh and 26 mm body envelope reconciled without rescaling/inference | BLOCKED - TBD |
| Structural load path | Material, fasteners, dual-side support/bearings, retention, fixture load evidence | BLOCKED - TBD |
| Gripper mechanism | Equal external gears, ratio -1, centers `y=+/-12 mm`, 30 mm rotary jaws; tooth/backlash/retention verified | BLOCKED - TBD |
| Packaging and clearance | Canonical motor centers checked against real envelopes, cables, fasteners, table, fixture and full motion | BLOCKED - UNVERIFIED |
| Mass targets | Trunk mount measured against 20 g target; complete cartridge measured against 160 g target | NOT RUN |
| Shoulder gravity margin | Measured worst-case gravity torque and actual continuous-duty evidence `>=3x` | BLOCKED - EVIDENCE MISSING |
| Separate 5 V arm domain | Verified no connection to original MicroDuck positive/body VDD/body DATA/USB VBUS | NOT RUN |
| Pack and BMS | Exact compatible parts, ratings, enclosure, charger and fault review | BLOCKED - TBD |
| Protection | Main and five branch protective devices coordinated with measured loads and conductors | BLOCKED - TBD |
| Harness/connectors | Pinout, continuity, polarity keying, ratings, strain relief, flex and routing evidence | BLOCKED - TBD |
| Disconnect behavior | Correct DC-rated device; open/reset behavior tested without automatic motion | BLOCKED - TBD |
| U2D2/hub bench interface | Stable communication and verified topology; no assumption that U2D2 powers motors | NOT RUN |
| Mobile interface | Compute, communication, power and mounting architecture | BLOCKED - TBD |
| Fixed-root fixture | Geometry, fasteners/clamp, overturning/load, workspace and catch review | BLOCKED - OUTSIDE CAD |
| Software limits | Joint signs, zeros, ranges, speed/acceleration/current/temperature responses recorded | NOT RUN |
| Incremental motion | Unloaded and reduced-range commanded tests completed per SOP | NOT RUN |
| Payload target | Incremental test through 20 g with thermal/tracking/grip/drop evidence | NOT RUN |
| Floating mode | Measured mass/inertia/attachment/power/balance evidence and separate authorization | BLOCKED |

## Acceptance rule / 验收规则

Release requires every applicable row to contain traceable evidence and a
reviewed `PASS`. `NOT RUN`, `TBD`, `BLOCKED`, missing data, estimates, simulation
only, or fixed-root evidence reused for floating mode all fail closed.

放行要求每个适用条目都包含可追溯证据并经评审标记为 `PASS`。`NOT RUN`、`TBD`、
`BLOCKED`、缺失数据、估算、仅仿真，或把固定根部证据复用于浮动模式，均按失败关闭。

## Sign-off / 签署

| Role / 角色 | Name / 姓名 | Revision / 修订 | Date / 日期 | Decision / 决定 |
|---|---|---|---|---|
| Mechanical reviewer / 机械评审 |  |  |  | NOT REVIEWED |
| Electrical reviewer / 电气评审 |  |  |  | NOT REVIEWED |
| Controls reviewer / 控制评审 |  |  |  | NOT REVIEWED |
| Test lead / 测试负责人 |  |  |  | NOT REVIEWED |
