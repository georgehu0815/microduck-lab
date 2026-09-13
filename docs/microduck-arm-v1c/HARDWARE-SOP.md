# MicroDuck arm v1-C hardware SOP / 硬件测试规程

> **PROCEDURE BASELINE ONLY - NO TEST HAS BEEN RUN OR PASSED**
>
> **仅为流程基线 - 尚无测试执行或通过**

This SOP applies only after an independent mechanical, electrical, controls,
and safety review authorizes the named stage. Use a guarded fixed-root fixture,
catch provision, calibrated instruments, and an enclosed current-limited
low-voltage source. Never test directly from mains wiring, short a source,
reverse polarity, backfeed USB, or reverse-power the robot.

本规程仅在机械、电气、控制和安全评审分别批准相应阶段后适用。必须使用带防护的固定
根部夹具、承接装置、校准仪器和封闭式限流低压电源。禁止直接操作市电接线、短接电源、
反接极性、向 USB 倒灌电流或反向给机器人供电。

## Evidence rules / 证据规则

1. Record specimen serials, actual actuator SKUs, firmware, actual bus IDs,
   source/regulator/protection part numbers, fixture revision, software commit,
   operator, observer, date, ambient temperature, and calibration identifiers.
2. Mark every unexecuted cell `NOT RUN`. `TBD`, estimates, simulations, and a
   prior v1-B result are not passes.
3. Keep Candidate A and Candidate B datasets separate until a preregistered
   comparison is complete.
4. Keep fixed-root, restrained-body, and free/mobile evidence separate.
5. Stop on unexpected motion, odor, noise, looseness, contact damage, cable
   strain, communication loss, overcurrent, undervoltage, excessive
   temperature, or any disagreement between watchdog and heartbeat.

1. 记录样机序列号、实际执行器型号、固件、实际总线 ID、电源/变换器/保护器件型号、
   夹具修订、软件提交、操作员、观察员、日期、环境温度和仪器校准编号。
2. 未执行单元必须标记 `NOT RUN`。`TBD`、估算、仿真和 v1-B 历史结果均不算通过。
3. 候选 A 与候选 B 的数据必须分开，直到完成预先登记的对比。
4. 固定根部、机身约束和自由/移动证据必须分开。
5. 出现意外运动、异味、异响、松动、接触损伤、线缆受力、通信丢失、过流、欠压、
   温升过高，或看门狗与心跳状态不一致时立即停止。

## Stage definitions / 阶段定义

| Stage | Required activity | Advance condition |
| --- | --- | --- |
| S0 | Configuration audit: spec hash, BOM closure, drawings, risk review, test plan, calibrated tools | All documents identified; unresolved gates remain visibly blocked |
| S1 | Dead-system mechanical inspection: measured interfaces, shafts, bearings, horns, spacers, fasteners, jaw gears, pads, cable path, fixture | No envelope-derived hole pattern; load path and retention inspected |
| S2 | Isolated power-domain test with motors disconnected, then one protected branch at a time | Correct polarity; no backfeed; disconnect removes arm VDD; protection behavior recorded |
| S3 | Torque-disabled communication: one actuator at a time, IDs, firmware, telemetry, watchdog, independent heartbeat | Unique actual IDs; stale/replayed traffic removes motion authority; no real-I/O shortcut |
| S4 | Fixed-root single-axis dynamic motion at reduced range, speed, acceleration, and current | Direction, zero, limits, tracking, temperature, cable motion, and stop response acceptable |
| S5 | Fixed-root coordinated motion and gripper cycles with payload ladder | No forbidden contact, looseness, excessive backlash, thermal excursion, or dropped payload |
| S6 | Restrained-system disturbance and fault testing; only then separately authorized mobile study | Measured inertia/CoM and >=3x continuous torque gate closed; E-stop/watchdog/heartbeat validated |

`S0-S6` is not a linear claim that simulation automatically advances to
hardware. Each stage has independent evidence and review authority.

`S0-S6` 不是“仿真通过即可自动进入硬件”的线性流程。每个阶段都需要独立证据和评审
授权。

## Payload levels / 载荷等级

| Level | Payload | Purpose | Initial authorization |
| --- | ---:| --- | --- |
| P0 | 0 g | Unloaded commissioning | Eligible after S0-S3 gates |
| P5 | 5 g | First loaded commissioning point | Blocked until P0 dynamic evidence |
| P10 | 10 g | Second commissioning point | Blocked until P5 review |
| P20 | 20 g | Stationary payload target | Blocked until >=3x continuous-torque and thermal evidence |
| P30 | 30 g | Engineering overload point, not a product target | Not authorized |
| P40 | 40 g | Engineering overload point, not a product target | Not authorized |
| P50 | 50 g | Maximum matrix point, not a rating | Not authorized |

The authoritative planned cells are in `TEST-MATRIX.csv`. No cell currently
contains an actual pass.

权威计划单元见 `TEST-MATRIX.csv`。当前没有任何单元包含实体通过结果。

## S0-S1: document and dead-system inspection

1. Verify the v1-C design ID and source hashes. Confirm that the 145 mm
   tabletop and +18 mm shoulder pivot are labeled as the changed scenario.
2. Reconcile the 170 g user category sum against the controlled 160 g target.
   Weigh each installed item; do not erase over-target mass with a negative
   reserve.
3. Inspect the real load path: payload -> pads -> jaws -> jaw shafts -> jaw
   bearings -> gripper housing -> independent wrist shaft/bearings -> forearm
   -> elbow shaft/bearings -> upper link -> shoulder shaft/bearings -> base
   shaft/bearings -> trunk interface -> fixture or robot structure.
4. Confirm motor output bearings are not treated as the only structural
   support unless the manufacturer explicitly approves the measured external
   loads.
5. Record all horn pilots, spline identities, screws, bolt circles, bearing
   fits, shaft shoulders, spacers, retainers, fastener grades, thread
   engagement, locking methods, and cable exits. Unknown part numbers remain
   `TBD`; do not machine from the 20 x 26 x 34 mm clearance envelope.
6. Inspect the independent wrist and jaw transmission. The wrist shaft must
   not double as an undocumented jaw shaft. Both jaw shafts require explicit
   support and retention.
7. Measure both 24 x 5 x 8 mm pad articles. Record material, durometer,
   friction method, compression curve/contact stiffness, retention, and wear.

## S2: isolated electrical qualification

1. Default topology is an external regulated 5 V bench source. Start with all
   motor branches physically open and torque disabled.
2. Verify source polarity, current limit, main protection, disconnect, ground
   reference, and absence of voltage on robot battery positive, body VDD, body
   DATA, and USB VBUS.
3. Close one protected branch at a time. Record idle and inrush voltage/current
   at source and load. Confirm no branch can energize another disabled branch
   through an unintended path.
4. Open the DC disconnect and verify actuator VDD removal. Controller USB
   enumeration is not proof of actuator power removal.
5. Shared-pack study: measure the robot pack's full operating range and
   transients before choosing a converter. Because pack voltage is not yet
   qualified, a buck-boost topology may be required; buck-only is not assumed.
6. Separate-pack study: require a qualified pack, BMS, charger, enclosure,
   connector, fault-current review, and reverse-current/regenerative-energy
   analysis.
7. Never apply reverse polarity, intentionally short the source, or connect
   any arm source so it can reverse-power the robot.

## S3: communications and safety logic

1. With torque disabled and only one motor connected, read model identity,
   firmware, voltage, temperature, error state, and current bus ID.
2. Assign and verify unique actual IDs only under a controlled commissioning
   record. IDs `21-25` are provisional labels, not evidence of physical IDs.
3. Add one motor at a time and confirm deterministic mapping J1/M1 through
   J5/M5.
4. Verify the command watchdog and the independent software heartbeat.
   Repeated read traffic must not refresh command age. A bus watchdog request
   is not guaranteed torque-off.
5. Inject stale command, stale heartbeat, replayed sequence, controller loss,
   overcurrent input, undervoltage input, and E-stop input. Record physical
   behavior only after the safety architecture is released.
6. Reset must require an explicit safe state and must not restore torque or
   command motion automatically.

## S4-S5: dynamic motion and load ladder

1. Start at P0 with one axis, reduced range, low speed, low acceleration, and
   a low-energy pose above the catch provision.
2. For each move record command/observed position, velocity, tracking error,
   source and branch voltage/current, temperature, fixture deflection, cable
   clearance, backlash, bearing motion, fastener witness marks, and stop time.
3. Advance J1 through J5 individually before coordinated motion. Verify signs,
   zeros, software ranges, hard-clearance margins, and no collision with the
   raised tabletop.
4. For M5, verify the driven jaw and passive jaw counter-rotate at ratio `-1`.
   Reject sliding-jaw behavior, same-direction rotation, gear disengagement,
   shaft walk, or bearing movement.
5. Run the payload ladder only after the previous level is reviewed. At every
   level include static holds, slow reversals, representative acceleration,
   grasp cycles, transfer waypoints, and controlled release.
6. The 20 g target requires continuous-duty evidence at least three times the
   measured worst-case gravity torque. Stall torque and the simulation
   `0.1 N*m` force limit are not continuous ratings.
7. P30-P50 are not ratings. They require a separate overload plan and may be
   omitted if structural or actuator limits make them unsafe.

## S6: restrained system and mobile prohibition

1. Replace all mass, CoM, and inertia estimates with measured values and rerun
   balance, collision, thermal, power, and controls analyses.
2. Validate the current raised-table scenario; do not import v1-B floor-level
   success.
3. Verify disturbance response and all stop paths with a restrained body
   before any free/mobile trial.
4. No mobile trial is authorized until a mobile controller is selected and
   reviewed. Current documents approve none.
5. Free/mobile operation requires a separate signed authorization covering
   attachment, balance, estimator latency, power, communication, dropped-load,
   tip-over, and human-access hazards.

## Torque and inertia release calculation

For each joint, calculate measured gravity torque from every distal component:

```text
tau_g,j = sum_i(m_i * g * horizontal_lever_arm_i_about_joint_j)
tau_required_continuous,j >= 3 * max(tau_g,j over released poses)
```

Then add dynamic torque, friction, cable, contact, and disturbance terms using
the released motion profile. The `3x` gate applies to qualified continuous
capability at the actual voltage and thermal condition, not stall torque.

As a deliberately conservative bounding check only, placing the full 160 g
cartridge plus 20 g payload at the full 135 mm geometric reach gives
`3 * 0.18 kg * 9.80665 m/s^2 * 0.135 m = 0.715 N*m`. This is not the actual
shoulder requirement, but it demonstrates why measured mass distribution and
continuous-duty data are release-critical.

对每个关节必须使用所有远端零件的实测质量和水平力臂计算重力矩，再叠加已放行运动
曲线中的动态、摩擦、线缆、接触和扰动项。`3 倍`门槛针对实际电压和热条件下的合格
持续能力，不是堵转力矩。将 160 g 卡匣和 20 g 载荷全部放在 135 mm 末端得到的
`0.715 N*m` 仅是保守边界检查，不是实际肩部需求。

