# MicroDuck Arm v1-C WingPod Appearance Brief
# MicroDuck 单臂 v1-C WingPod 外观设计简报

## Status / 状态

- **Draft, visual sidecar only; refreshed 2026-09-13. / 草案，仅限外观侧车；更新于 2026-09-13。**
- Purpose: make the single arm feel like one cute, integrated MicroDuck wing without changing its function. / 目标：让单臂像从 MicroDuck 身体自然长出的可爱完整翅膀，同时不改变功能。
- This brief governs a separate render overlay and parametric review-only shell CAD. It does not release fabrication or guarantee hardware performance. / 本简报仅约束独立渲染覆盖层与参数化评审壳体 CAD；不构成加工放行，也不保证真机性能。

## Evidence Reviewed / 已审查依据

- Render: `artifacts/microduck-arm-v1c/tennis-return/half-height-v13-videos/video-verification/terminal-frames/small-5.png`. Observed: the current arm reads as exposed graphite motor boxes, thin honey links, orange offset gripper stems, green pads, and several stacked rectangular trunk modules. The forms are functionally legible but visually fragmented beside MicroDuck's rounded body. / 渲染观察：当前机械臂由外露石墨色电机盒、细黄连杆、橙色外扩夹爪支杆、绿色夹垫和多层矩形躯干模块组成；功能可读，但与圆润机身相比显得零散。
- Model: `artifacts/microduck-arm-v1c/tennis-return/half-height-planned-v13/nominal.xml`. The values below are **observed frozen model facts**, not new proposals. / 模型：以下数值为**已读取的冻结模型事实**，不是新提案。

| Frozen item / 冻结项 | Observed value / 读取值 |
|---|---:|
| Arm pose joints / 手臂姿态关节 | base yaw, shoulder pitch, elbow pitch, wrist pitch / 底座偏航、肩俯仰、肘俯仰、腕俯仰 |
| Gripper actuation / 夹爪驱动 | one driven joint plus opposite follower / 单驱动关节加反向随动关节 |
| Mount position / 安装位置 | local `[15, 0, 40]` mm |
| Shoulder pivot / 肩轴 | local `+18` mm within arm-yaw body / 位于偏航体局部 `+18` mm |
| Shoulder-to-elbow / 肩至肘轴距 | `55 mm` |
| Elbow-to-wrist / 肘至腕轴距 | `50 mm` |
| Wide-candidate TCP / 宽夹爪 TCP | local `x = 55 mm` |
| Open inner-pad gap / 张开时夹垫内距 | `75 mm` |
| Each contact pad / 单个夹垫 | `24 x 5 x 8 mm` |
| Motor review envelope / 电机评审包络 | `20 x 26 x 34 mm`; not a mounting drawing / 非安装图 |

## Design Direction / 设计方向

### Camera v2 Soft revision / 相机柔和版

The latest camera-face reference is `duck-soft-frame-finished.png`: prefer a
cream rounded panel, sage-gray rings, navy lens centers, peach cheeks and a small
honey beak instead of the previous graphite rectangle. Apply only through
`wingpod_camera_soft.py`; preserve the original optical centers and physical
model. Full palette and evidence boundaries: `CAMERA-V2-SOFT.md`.

相机脸按最新参考图采用奶油底色、灰绿眼圈、深蓝镜心、淡桃色腮红和小黄嘴；
不改运动模型、胸部位置或光轴。页面主图和精选视频用新配色，历史 32 案例明确
标注原深色外观，不冒充新配色训练或硬件验收。

**A small porcelain wing with honey feathers, growing from a friendly central saddle. / 一只从友好中央肩鞍自然长出的瓷白小翅膀，带蜂蜜黄色羽片。**

- Warm porcelain/cream white is the primary shell color; honey yellow marks wing surfaces and intentional action areas; graphite marks joints, gaps, pads, and service hardware. / 暖瓷白或奶油白为主壳色；蜂蜜黄用于翅膀表面和主动区域；石墨色用于关节、间隙、夹垫与维护件。
- Prefer soft, inflated, asymmetric feather segments over tubes or rectangular armor. Keep the silhouette compact at the root and gently tapered toward the wrist. / 采用柔和饱满、略带不对称的羽片分段，不做管状或方盒装甲；根部紧凑，向手腕逐渐收细。
- The memorable feature is the **central shoulder saddle**: one calm cream bridge visually unifies the trunk interface, yaw root, and arm origin. / 核心识别点是**中央肩鞍**：一块安静的奶油白桥形体，将躯干接口、偏航根部和手臂起点统一起来。
- Add a passive sensor-face detail to the saddle front: two small graphite lens-like dots and one shallow honey notch. It is decorative, fixed, and must not imply a qualified sensor. / 肩鞍正面增加被动“传感器脸”：两个小石墨色镜点加一个浅蜂蜜黄凹口；固定、纯装饰，不得暗示已验证传感器。

### Visual hierarchy / 视觉层级

1. **Character:** saddle face and porcelain root establish “MicroDuck first.” / **角色感：** 肩鞍小脸与瓷白根部首先表达 MicroDuck。
2. **Gesture:** one continuous honey wing line connects the 55 mm upper arm and 50 mm forearm. / **动作线：** 连续蜂蜜黄翅膀线串联 55 mm 上臂与 50 mm 前臂。
3. **Mechanism:** graphite joint reveals clearly show base, shoulder, elbow, and wrist articulation. / **机构：** 石墨色关节留缝清楚显示底座、肩、肘、腕的转动关系。
4. **Tool:** the gripper remains visibly a tool; its 75 mm opening and both pads must read unobstructed. / **工具：** 夹爪仍应明显是工具；75 mm 开口和两块夹垫必须完整可见。

## Form and Part Intent / 形体与部件意图

| ID | Appearance part / 外观件 | Intent / 意图 |
|---|---|---|
| WP-01 | Central shoulder saddle / 中央肩鞍 | Porcelain bridge around the existing mount; removable underside service hatch / 围绕现有安装位的瓷白桥体；底部可拆维护盖 |
| WP-02 | Passive face insert / 被动小脸嵌件 | Fixed graphite “eyes” plus honey notch; no added joint or sensor claim / 固定石墨“眼”与蜂蜜黄凹口；不增加关节，不宣称传感器 |
| WP-03 | Yaw root collar / 偏航根部环 | Graphite rotational reveal separating saddle from moving arm / 用石墨色转动留缝区分固定肩鞍与运动手臂 |
| WP-04 | Upper wing shell / 上臂翼壳 | Rounded two-feather clamshell following the frozen 55 mm axis distance / 沿冻结 55 mm 轴距布置的双羽片圆润对合壳 |
| WP-05 | Elbow knuckle / 肘节罩 | Compact graphite joint ring; never bridge fixed and moving bodies / 紧凑石墨关节环；不得跨接两个相对运动体 |
| WP-06 | Forearm wing shell / 前臂翼壳 | Slim single-feather taper following the frozen 50 mm axis distance / 沿冻结 50 mm 轴距布置的单羽片收尖壳 |
| WP-07 | Wrist cuff / 腕环 | Short graphite cuff preserving wrist sweep and gripper service access / 短石墨腕环，保留腕部扫掠与夹爪维护空间 |
| WP-08 | Palm fairing / 掌部整流壳 | Cream palm cap behind, not around, the contact region / 位于接触区之后而非包围接触区的奶油白掌盖 |
| WP-09L/R | Wide-jaw finish / 宽夹爪表面 | Current implementation is honey recoloring of existing bridges/stems only, not added geometry; pads unchanged / 当前仅将已有桥架与支杆改为蜂蜜黄，不新增几何；接触垫不变 |
| WP-10 | Cable guides / 线缆导向件 | Future intent only, not included in current render/CAD: graphite clips/troughs aligned to the real neutral cable path / 仅为后续意图，尚未进入当前渲染或 CAD：石墨色卡扣与线槽须服从真实线缆中性路径 |

The current face detail is on the existing sensor shell's +x surface and may be occluded by
the central arm in some poses. The render does not relocate the head to conceal this
packaging limitation. / 当前小脸位于已有传感器壳的 +x 面，某些姿态下可能被中央机械臂遮挡。
渲染不通过迁移头部来掩盖这一布置限制。

## Frozen Functional Contract / 冻结功能契约

- No new moving joint, linkage, flap, head motion, or head relocation. / 不新增运动关节、连杆、活动翼片、头部动作或头部迁移。
- Do not move or rotate any existing joint axis, body transform, actuator, TCP, site, collision geometry, or control limit. / 不移动或旋转任何现有关节轴、刚体变换、执行器、TCP、site、碰撞几何或控制限位。
- Keep shoulder-to-elbow `55 mm` and elbow-to-wrist `50 mm` exactly as modeled. / 肩肘 `55 mm` 与肘腕 `50 mm` 轴距保持模型原值。
- Do not shrink the `75 mm` open inner-pad gap or move the `55 mm` wide-candidate TCP. / 不缩小 `75 mm` 张开夹垫内距，不移动宽夹爪 `55 mm` TCP。
- Do not cover, recess, shorten, soften, recolor out of legibility, or replace either `24 x 5 x 8 mm` modeled contact pad. / 不覆盖、内陷、缩短、软化、弱化辨识度或替换任一 `24 x 5 x 8 mm` 模型夹垫。
- Shell pieces must terminate before every joint reveal and pad working face. Cute form is subordinate to motion, contact, and service access. / 壳体必须在各关节留缝和夹垫工作面之前结束；可爱造型服从运动、接触与维护。

## Render-Only Equivalence / 仅渲染功能等价

A render may be called **functionally equivalent** only when the WingPod is renderer-side or otherwise excluded from mass, inertia, collision, contact, sensing, and control. The frozen model must produce identical body transforms, controls, contacts, and task result with the overlay hidden or shown. / 只有当 WingPod 位于渲染器侧，或明确不参与质量、惯量、碰撞、接触、传感与控制时，才可称为**功能等价渲染**。显示或隐藏覆盖层时，冻结模型的刚体变换、控制、接触与任务结果必须一致。

This equivalence applies only to the visualization. It is **not evidence** that a real shell fits, survives motion, cools the motors, routes cables, or preserves balance. / 此等价仅适用于可视化，**不能证明**实体壳体可装配、可承受运动、可散热、可布线或不影响平衡。

## Seams, Service, and Cable Treatment / 接缝、维护与线缆处理

- Put primary seams on the underside/rear shadow line; keep the top wing surface visually continuous. / 主接缝放在底部或后侧阴影线，上表面保持连续。
- Use separate clamshells per moving body. Never make one cosmetic shell span a joint. / 每个运动刚体使用独立对合壳，禁止用单一外壳跨越关节。
- Keep graphite reveal rings around axes and visible tool access at every motor, horn, shaft, bearing, fastener, and gripper transmission interface. / 各轴周围保留石墨色留缝，并确保电机、舵盘、轴、轴承、紧固件与夹爪传动均可维护。
- Route cables in a shallow dorsal trough with short graphite flexible bridges at joints; preserve bend radius, strain relief, and full-range slack. / 线缆沿背侧浅槽布置，在关节处用短石墨柔性桥过渡；保留弯曲半径、应力释放与全行程余量。
- **Review seeds, not released dimensions:** `0.8 mm` nominal visual seam, `1.5 mm` nominal joint reveal, `1.2 mm` starting shell wall, and `2.0 mm` starting hard-part clearance. All require tolerance, motion, thermal, and process validation. / **仅为评审起点，非放行尺寸：** 名义视觉缝 `0.8 mm`、关节留缝 `1.5 mm`、起始壁厚 `1.2 mm`、起始硬件净距 `2.0 mm`；均须公差、运动、热与工艺验证。

## Physical Release Gates / 实体放行门槛

No shell may be released for fabrication or hardware installation until all gates pass. / 以下门槛全部通过前，不得放行加工或真机安装。

- **Datum lock:** CAD references the frozen mount, axes, `55/50 mm` link distances, `55 mm` TCP, `75 mm` opening, and pad envelopes. / **基准锁定：** CAD 引用冻结安装位、轴线、`55/50 mm` 轴距、`55 mm` TCP、`75 mm` 开口和夹垫包络。
- **Sweep clearance:** measured/tolerance-aware full joint and gripper sweeps show no shell collision, pad obstruction, pinch intrusion, or task-envelope loss. / **扫掠净空：** 含公差的全关节与夹爪扫掠无壳体碰撞、夹垫遮挡、夹伤侵入或任务包络损失。
- **Mass properties:** complete shell, retainers, fasteners, and cable hardware are weighed; CoM and inertia are updated and accepted against robot stability and actuator margins. / **质量特性：** 完整壳体、保持件、紧固件与线缆硬件实测称重，并更新质心与惯量，确认稳定性及执行器余量。
- **Cooling:** vent area, airflow, motor/driver temperature, and thermal-soak behavior pass at representative duty cycles. / **散热：** 通风面积、气流、电机或驱动温度及热浸表现通过代表性工况测试。
- **Cable routing:** real connector sizes, bend radii, strain relief, flex life, abrasion, snag, and full-range slack are verified. / **布线：** 实际连接器尺寸、弯曲半径、应力释放、弯折寿命、磨损、勾挂与全行程余量完成验证。
- **Serviceability:** shell removal does not require disturbing joint calibration; all critical fasteners, pads, gears, and connectors remain accessible and inspectable. / **可维护性：** 拆壳无需破坏关节标定；关键紧固件、夹垫、齿轮与连接器均可接近和检查。
- **Structure and safety:** material/process, retention, edge radii, flammability, impact/drop behavior, proof load, and fragment containment are qualified. / **结构与安全：** 材料与工艺、保持方式、边缘圆角、阻燃、冲击或跌落、证明载荷与碎片约束完成验证。

## Risk Checklist / 风险清单

- [ ] Added shell mass or shifted CoM reduces balance or torque margin. / 壳体增重或质心偏移降低平衡与扭矩余量。
- [ ] Shell enters the robot, bin, ball, floor, or self-collision envelope. / 壳体进入机身、桶、球、地面或自碰撞包络。
- [ ] Fairing narrows the 75 mm opening or contacts the ball before the pads. / 饰壳缩小 75 mm 开口，或先于夹垫接触球。
- [ ] Joint seam becomes a pinch point or hides axis motion. / 关节缝形成夹点或掩盖轴运动。
- [ ] Enclosure traps heat or blocks inspection. / 封闭壳体积热或阻碍检查。
- [ ] Cable bridge snags, rubs, over-bends, or changes joint load. / 线缆过渡件勾挂、磨损、过弯或改变关节负载。
- [ ] Decorative face is mistaken for a working safety/perception sensor. / 装饰小脸被误认为有效安全或感知传感器。
- [ ] Cosmetic fasteners obstruct structural fasteners or calibration access. / 外观紧固件妨碍结构紧固件或标定入口。
- [ ] Render beauty is presented as physical clearance, durability, or hardware evidence. / 将渲染美观误作实体净空、耐久或真机证据。

## Stop Condition / 完成条件

Appearance work is complete when the render overlay clearly reads as one porcelain-and-honey wing, all frozen functional features remain visible and unchanged, and the review CAD remains explicitly **UNVERIFIED** until every physical gate above has evidence. / 当渲染覆盖层清楚呈现为一体化瓷白蜂蜜黄翅膀、所有冻结功能特征保持可见且不变，并且评审 CAD 在上述实体门槛取得证据前始终明确标注为 **UNVERIFIED（未验证）** 时，外观设计工作才算完成。
