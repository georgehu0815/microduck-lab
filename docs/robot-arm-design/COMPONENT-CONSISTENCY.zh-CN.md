# Microduck 与机械臂：共享电机和部件一致性核对

日期：2026-09-12。范围：用户指定的 `rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions_backlash.xml`、其资产、机械臂 BOM/CAD 和厂家规格。**未接实体硬件，未确认身体电机完整 SKU，未放行制造。**

## 1. 核对结论

- 该 XML 有 **15 个 `xl330` 电机外观实例**，全部引用同一个 `assets/xl330.stl`；有 **14 个主动执行器、14 个 backlash 关节**。嘴部不在原有 14 动作策略内，不能把 14 动作等同于 14 台实物电机。
- XML 和 `xl330.part` 标明 XL330 家族，**没有 M288/M077 后缀**。机械臂仍是 **XL330-M288-T 候选、每臂 6 个**，不能据此声称身体必定是相同减速箱版本。
- 已将机械臂 OpenSCAD 的电机显示由另画的方盒改为**直接导入身体的原始 STL**；BOM 和规格文件记录同一资产路径，生成器记录 SHA256、Onshape part 元数据及每个身体实例的位姿。
- 原 STL 坐标单位按源 MuJoCo 模型为米。CAD 仅做统一 **×1000** 转毫米、居中和绕 Z 轴旋转 90°，不改变比例、不重绘安装孔。
- 本次不修改身体 XML、原 STL、机械臂训练物理模型或评估标准。共享**外观来源**已实现；机械安装、动力学和实机兼容仍须分别验证。

## 2. 同源三视图与尺寸差异

![身体与机械臂使用同一网格的三视图](../../hardware/md-arm-t1/generated/shared-xl330-comparison.svg)

| 表示 | 尺寸 | 使用范围 |
|---|---|---|
| ROBOTIS 发布的本体包络 | W20 × H34 × D26 mm | 规格、初步本体空间预算 |
| 当前身体原始 STL 的轴对齐边界 | X29 × Y20 × Z34 mm，4,126 个三角形 | 本次共享外观/几何比较 |
| CAD 旋转后的网格边界 | W20 × H34 × D29 mm | 保留源网格，不强压成 D26 |
| 厂家精密装配图 | 未取得可验证的 STEP/PDF 二进制 | 孔位、轴线基准、公差仍然阻断 |

**29 mm 与 26 mm 不是可随意修正的舍入误差。** 是否来自输出端凸起或模型简化，当前没有足够证据，报告不猜测。旧 `xl330-body-envelope.stl` 仍只表示厂家本体包络，不是安装模型，也不是整个电机所有凸出部分的碰撞包络。

ROBOTIS 的 M288 与 M077 手册指向相同的通用 `XL330.pdf / XL330.dwg / XL330.stp` 下载入口，支持使用同一外部零件 CAD；但仍需检查厂家图纸与实物修订后才能确定精密配合。

## 3. 相同外形不等于相同马达参数

| 厂家参数（5 V 条件） | XL330-M288-T 候选 | XL330-M077-T 对比 |
|---|---:|---:|
| 减速比 | 288.4:1 | 77.5:1 |
| 堵转力矩 | 0.52 N·m | 0.215 N·m |
| 空载速度 | 103 rpm | 383 rpm |
| 堵转电流 | 1.47 A | 1.47 A |
| 本体质量 | 18 g | 18 g |

两者推荐 5 V、输入范围 3.7–6 V。**堵转指标不是连续工作额定值**；不同减速箱的控制参数、惯量、摩擦、回差不能因为 CAD 相同就混用。当前机械臂 BOM 不允许用 M077 无审查替换 M288。

用户 XML 的 `chosen_actuator` 为 `kp=0.55`、`kv=0`、`forcerange=-0.96…0.96`；还有 ±1° 的 backlash 模型。这些是该身体模型的仿真设置，**不是 0.96 N·m 的厂家承载认证，也不是机械臂校准结果**。本次没有把它们复制进机械臂训练模型。

## 4. 其他部件逐类核对

| 部件/接口 | 身体模型或软件中证据 | 机械臂处理 |
|---|---|---|
| 电机外观 | 15 个同源 `xl330` mesh | 共享原 STL 和坐标约定；实物 SKU 仍待读取/铭牌核实 |
| 电机支架 | 存在 `motor_support` mesh | 不假设能直装到肩/肘；孔位、受力、线缆余量需独立设计 |
| 轴承 | `seeed_bearing__configuration__22x16x4` 及 `configuration_default` | 名字不是轴承额定载荷证书；不直接复用未核实的规格/公差 |
| 电池 | XML 有 `np_f970` 外观；身体软件电量估算 6.6–8.2 V | 电池侧数值不是舵机轨电压；保持机械臂独立电源，不连接身体电池正极 |
| 控制板/计算机外观 | XML 包含 `pcb__raspberry_pi_zero_2_w` 与 `elec_rpi_robot_hat_pcb` | 旧 CAD 名称不代表当前机载硬件实物 BOM，不能据此开 PCB 或认定引脚 |
| TTL/连接器 | 身体代码使用 XL330 总线；具体物理线束仍需核实 | 机械臂保留独立 U2D2；舵机 1GND/2VDD/3DATA，U2D2 的 2 为 N/C |
| 夹爪 | 身体嘴部零件不是已验证机械臂夹爪 | 不把嘴部连杆替作承重夹持器；夹爪传动和开口标定仍独立验证 |

完整网格实例清单和位姿见 `hardware/md-arm-t1/generated/component-consistency.json`，不只核对名称而忽略父 body 和变换。

## 5. 可打开的文件及复现

- `hardware/md-arm-t1/cad/md_arm_t1.scad`：设 `render_part="shared_motor"` 查看同源电机。原 `assembly` 仍是三个电机参考体的连杆草图，**不是六电机完整装配**。
- `hardware/md-arm-t1/generated/shared-xl330.scad`：由现有规格和源 STL 生成，禁止手工维护第二套电机外形尺寸。
- `hardware/md-arm-t1/generated/shared-xl330-comparison.svg`：两排共用相同图元的三视图，可直接浏览。
- `hardware/md-arm-t1/generated/shared-xl330-preview.xml`：独立的 MuJoCo 双电机视觉比较场景，两边共用 mesh ID 0；**不替换训练场景，不证明碰撞安全**。
- `hardware/md-arm-t1/generated/component-consistency.json`：来源、哈希、尺寸、位姿、型号未知状态、动力学隔离声明。

```bash
rlx/.venv-microduck/bin/python hardware/md-arm-t1/scripts/generate.py
rlx/.venv-microduck/bin/python -m unittest discover -s hardware/md-arm-t1/tests -p 'test_*.py'
```

当前硬件测试 **10/10 通过**，包含生成器、共享图元、坐标/尺度、15/14 计数和损坏 STL 拒绝。MuJoCo 已实际载入比较场景：`nmesh=1`、`ngeom=2`、两几何共用 `[0,0]`。测试通过不等于 OpenSCAD 编译验证：本机没有 OpenSCAD CLI，SCAD 只完成源码与引用检查。

## 6. 下一阶段放行条件

1. 逐个记录身体/机械臂电机完整 SKU、硬件/固件修订和型号寄存器；身体当前未知，不填假值。
2. 获取有效厂家 CAD、核对安装孔/轴线/舵盘/螺纹啮合及连接器插拔空间；建立一个共同零件库，不分别画两份相同电机。
3. 完成六电机实际装配、双侧支撑和干涉/线缆/重量校核；共享 mesh 不能替代装配设计。
4. 实测不同关节负载下的限流、温升、摩擦、回差，再决定机械臂参数，不能照搬身体拟合值。
5. 若改变训练模型的质量、碰撞体、传动或马达约束，创建新模型版本并重新训练/验证；不得把当前旧模型的通过率移作新硬件结论。

## 官方来源

检索日期：2026-09-12。

- [ROBOTIS XL330-M288-T 手册](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
- [ROBOTIS XL330-M077-T 手册](https://emanual.robotis.com/docs/en/dxl/x/xl330-m077/)
- [ROBOTIS U2D2 手册](https://emanual.robotis.com/docs/en/parts/interface/u2d2/)
- 厂家 CAD 下载状态保留在 `hardware/md-arm-t1/vendor/manifest.json`；不能把返回的 HTML 错误页当 STEP/PDF。
- 本地源：用户指定 XML、同目录 `assets/xl330.stl` 与 `assets/xl330.part`、`microduck/duck-control/src/model.rs`。
