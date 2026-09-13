# MicroDuck arm v1-B commanded test SOP / 指令式测试规程

> **ENGINEERING PROTOTYPE NOT FABRICATION RELEASE**  
> **Procedure definition only; no test in this document has been run or passed.**  
> **仅定义流程；本文档不声明任何测试已执行或通过。**

This SOP is for a qualified operator after component selection and an
independent safety review. It gives commanded test steps, not instructions for
assembling mains wiring. Use only a suitably enclosed, protected low-voltage
source. Stop on unexpected motion, heating, odor, noise, current, voltage drop,
communication loss, looseness, or damage.

本 SOP 供完成选型并通过独立安全评审后的合格操作人员使用。它规定指令式测试步骤，
不提供市电装配说明。只使用合适的封闭式、受保护低压电源。出现意外运动、发热、
异味、异响、异常电流、压降、通信中断、松动或损伤时立即停止。

## 0. Record control / 记录控制

1. Record the canonical spec revision, CAD/netlist hashes, operator, observer,
   date, fixture ID, source/BMS/fuse/connector part numbers, firmware, and
   software commit.
2. Mark every unperformed item `NOT RUN`; never infer a pass.
3. Keep fixed-root and floating-mode evidence separate.

1. 记录规格修订、CAD/网表哈希、操作员、观察员、日期、夹具 ID、电源/BMS/保险丝/
   连接器型号、固件和软件提交。
2. 未执行项目标记 `NOT RUN`，不得推断通过。
3. 固定根部和浮动模式证据必须分开。

## 1. Document and dead-system inspection / 文档与断电检查

1. Command no motion. Isolate the arm source and verify the documented
   isolation method before touching the harness.
2. Confirm exactly five M288 candidates and unique planned IDs.
3. Inspect solid support blanks and verify no one machined TBD hole patterns
   from the envelope or visual mesh.
4. Measure and record actual motor, horn, fastener, link, gear, jaw, cable, and
   fixture interfaces. Compare against released drawings only.
5. Verify pin-by-pin harness continuity and all explicit no-connects.

1. 不发送运动指令。隔离手臂电源，并在接触线束前核实文件化的隔离方法。
2. 确认正好五颗 M288 候选及计划中的唯一 ID。
3. 检查实心支撑毛坯，确认无人依据包络或可视网格加工 TBD 孔位。
4. 测量并记录电机、舵盘、紧固件、连杆、齿轮、夹指、线缆和夹具实际接口。
5. 按针脚检查线束连续性和全部明确断开点。

## 2. Power-domain command sequence / 电源域指令顺序

1. Keep all servo torque disabled in software.
2. Command the source on with current limiting/protection set by the approved
   electrical test plan, not by this document.
3. Record source voltage/current and each protected branch at idle.
4. Command the disconnect open; verify arm VDD is removed while noting that
   USB may keep U2D2 enumerated.
5. Reset only after an explicit software-safe state; reset must not command
   motion or automatically restore torque.

1. 软件保持全部舵机扭矩关闭。
2. 按已批准的电气测试计划设置限流/保护后上电，不使用本文档猜测额定值。
3. 记录电源和各保护支路的空闲电压、电流。
4. 指令断电器断开，验证手臂 VDD 被切断；注意 USB 可能仍使 U2D2 保持枚举。
5. 仅在明确的软件安全状态下复位；复位不得触发运动或自动恢复扭矩。

## 3. Communications and direction / 通信与方向

1. With torque disabled, command a ping/read of one isolated motor at a time.
2. Record model identity, firmware, ID, voltage, temperature, and error state.
3. Command a very small, low-speed motion for one unloaded axis at a time in
   the fixed-root fixture and low workspace.
4. Verify sign, limit, zero reference, cable motion, and disconnect response.
5. For the gripper, command jaw A through a small angular increment and verify
   jaw B rotates oppositely through the equal-gear coupling. Reject translation
   or independent same-direction rotation.

1. 扭矩关闭时逐颗隔离执行 ping/读取。
2. 记录型号、固件、ID、电压、温度和错误状态。
3. 在固定根部夹具和低位工作区内，每次只对一个空载轴发送极小、低速运动指令。
4. 验证方向符号、限位、零位、线缆运动和断电响应。
5. 对夹爪，指令夹指 A 小角度转动，验证夹指 B 通过等齿轮反向转动。若出现平移或
   同向独立旋转则拒收。

## 4. Incremental fixed-root motion / 固定根部递增运动

1. Command one joint at a time over a reduced range, then coordinated motion
   at reduced speed/acceleration.
2. At each step record commanded/observed position, tracking error, voltage,
   current, temperature, branch protection state, fixture motion, and cable
   clearance.
3. Command the disconnect during a preplanned low-energy motion and record the
   stop behavior; do not claim a safety category from this test.
4. Repeat only within the approved thermal/rest plan.

1. 先逐关节在缩小范围内运动，再以降低速度/加速度执行协调运动。
2. 每步记录指令/观测位置、跟踪误差、电压、电流、温度、支路保护状态、夹具位移和
   线缆间隙。
3. 在预先规划的低能量运动中指令断电并记录停止行为；不得据此声明安全等级。
4. 只在已批准的温升/休息计划内重复。

## 5. Load and shoulder evidence / 载荷与肩部证据

1. Measure the complete cartridge mass and center of mass before any payload.
2. Determine worst-case shoulder gravity torque from measured mass geometry.
3. Obtain continuous-duty capability evidence for the actual actuator,
   thermal environment, voltage, motion profile, and support arrangement.
4. Require documented continuous capability of at least `3 x` the measured
   worst-case gravity torque before advancing. Stall torque is not acceptable
   evidence.
5. Add test mass incrementally above a catch provision; the 20 g payload target
   remains unverified until all acceptance criteria pass.

1. 加载前测量完整卡匣质量和重心。
2. 根据实测质量几何计算肩部最坏重力矩。
3. 对实际执行器、热环境、电压、运动曲线和支撑结构取得连续工作能力证据。
4. 继续测试前，文件化连续能力必须至少为实测最坏重力矩的 `3 倍`。堵转力矩不能
   作为证据。
5. 在承接措施上方逐步增加测试质量；20 g 载荷目标在全部验收通过前仍未验证。

## 6. Floating-mode restriction / 浮动模式限制

Do not transfer fixed-root results to floating mode. Floating simulation uses
unmeasured mass placeholders and requires updated measured mass, center of
mass, inertia, attachment compliance, power architecture, balance/control
analysis, and a separate test authorization before any physical trial.

不得把固定根部结果直接用于浮动模式。浮动仿真使用未测量质量占位值；任何实体
试验前必须更新实测质量、重心、惯量、连接柔度、供电架构和平衡/控制分析，并取得
单独测试授权。

