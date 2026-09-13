# Amazon 美国站硬件候选：先核型号，再采购

核查日期：2026-09-12 · 状态：**选型建议，不是已下单清单或实时库存确认**。

用户允许在 Amazon 搜寻更合适硬件，因此不再把不明版本的拍卖套件作为必须采用的器件。主方案仍是“同族 XL330 小型桌面臂 + 独立电源/总线 + Microduck 统一控制”。

## 1. 本次可确认到什么程度


- 除单列的 RoArm-M2-S 具体商品外，以下 Amazon 链接是**明确标注的关键词搜索入口**，不是已确认商品页面、ASIN 或购买保证。不会编造商品编号或推荐未经验证的低价替代型号。
- XL330、U2D2、SO-101 与电源候选的规格依据是厂商/项目官方页面，不是 Amazon 标题。拍卖的供电和型号依旧未确认。
- 本次没有购物车操作、购买、账户登录或实物接线。当前设计不能因为找到电商入口就解锁硬件。

你随后提供的 [Amazon robot arm 搜索页](https://www.amazon.com/s?k=robot+arm&crid=28TV47GUK3DL2&sprefix=robot+arm%2Caps%2C175&ref=nb_sb_noss_1) 也已尝试读取，仍返回 503；不能声称看到了该页面当前排序或全部商品。补充检索获得下面一个可核对身份的具体商品页，和一般关键词入口分开列出。

### 可核对的具体商品：Waveshare RoArm-M2-S（条件备选）

- Amazon 美国站：[RoArm-M2-S，ASIN B0CMLNPH8V](https://www.amazon.com/dp/B0CMLNPH8V)。该具体页面已成功直接读取，商品表的 Manufacturer Part Number 为 `RoArm-M2-S`，ASIN 匹配；搜索页仍无法读取。没有登录/配送地址/购物车复核，不保证结算时卖家、库存、价格或附件。
- 厂商产品页：https://www.waveshare.com/product/roarm-m2-s.htm
- 厂商接口/教程：https://www.waveshare.com/wiki/RoArm-M2-S
- 官方资料给出独立控制器和 USB/UART/HTTP JSON 控制与反馈路径，可作为“Microduck 发任务 → 外部机械臂网关”的桌面适配候选；首选 USB 台架。厂商注明 ESP-NOW 路径不能获取反馈，因此不用于本方案闭环臂状态读取。仍须验证具体固件、限位、时序和失联行为。
- 官方供电范围 7–12.6 V，推荐 12 V / 5 A；**它不是本方案 XL330 的 5 V 电源分支，禁止混接。** 官方重量约 826±15 g（不含夹具），不作为挂在 Microduck 身上的候选。
- 教程列 base、shoulder、elbow、end 四个控制通道；end 选 gripper 模式时是 **3 个姿态关节 + 夹爪**，不是六姿态轴。它不能完整代替腕部姿态实验。为它创建自己的 MJCF、动作合同和缩减案例，不复用 `md-arm-table-v1` 的六动作映射。
- 资料中的上电回位行为是集成审查重点：必须预留清空的启动空间，并验证可控的上电/使能顺序；未通过上电运动和急停检查，不接入无人值守任务。
- 结论：**适合作为有独立接口的桌面入门备选，不是 Microduck 电气/机械直接兼容臂。** 若首要目标是未来复用原机舵机生态，仍优先 XL330 定制路线。

## 2. 推荐主线：兼容性优先

| 优先级 | 搜索型号 / 数量 | Amazon 入口 | 已核规格 / 选择理由 | 必须排除或进一步确认 |
|---|---|---|---|---|
| 1 | **ROBOTIS DYNAMIXEL XL330-M288-T**，单臂 6 / 双臂 12 | [搜索 XL330-M288-T](https://www.amazon.com/s?k=ROBOTIS+DYNAMIXEL+XL330-M288-T) | 同 XL330 技术族，TTL 半双工；18 g；推荐 5 V；有位置/电流/温度反馈 | 不能静默替换成 XL430、XL320、XC330 或 M077；原机实际型号也需核对；肩部载荷仍有灰区 |
| 2 | **ROBOTIS U2D2**，每臂 1 | [搜索 ROBOTIS U2D2](https://www.amazon.com/s?k=ROBOTIS+U2D2) | USB→Dynamixel；本方案使用 TTL 口，分离原身体 UART | 不给舵机供电；普通 USB-TTL 模块不能因便宜而直接替代半双工接口 |
| 3 | **5 V / 10 A 级电源候选**，每臂 1 | [搜索 ALITOVE 5V 10A](https://www.amazon.com/s?k=ALITOVE+5V+10A+power+supply) | 官方候选 SKU `5V-10A-BK` 为 5 V / max 10 A / 50 W，中心正极圆孔连接器 | 只是候选，不等于适合机器人脉冲负载或已获课堂电气安全批准；不要误选同店 12 V/24 V |
| 4 | **XL330 专用线束/连接器**，按实际端口和分路 | [搜索 XL330 cable](https://www.amazon.com/s?k=ROBOTIS+XL330+cable) | 按 XL330 官方 pinout 和连接器采购，保持数据/地正确 | 3-pin、JST、servo cable 等通用标题不足以证明间距/线序/电流能力匹配 |
| 5 | **USB UVC 桌面相机**，初始 1 | [搜索 UVC manual focus camera](https://www.amazon.com/s?k=USB+UVC+camera+manual+focus+robotics) | 用于物体定位/标定；优先可固定焦距、可控制曝光、Linux UVC 可识别的具体 SKU | 还未选定型号；需核机载 USB、视场、工作距离、遮挡和真实延迟；不是“4K 就一定更准” |
| 6 | **桌面双点夹具/固定板/软接物盘**，一套 | [搜索 robot arm table clamp](https://www.amazon.com/s?k=robot+arm+table+clamp+mount) | 桌面路线的稳定与防坠比装饰外观重要 | 开口、孔距、承载、侧向保持力必须匹配 CAD；未指定的通用夹具不直接算验收通过 |
| 7 | **急停、分路保护和配电线束**，一套 | [搜索 DC emergency stop switch](https://www.amazon.com/s?k=DC+emergency+stop+switch) | 需要正确 DC 分断额定值、手动复位、覆盖全部臂电源分路 | 该入口仅供找元件；不是推荐任意红色按钮。AC 标称额定值不能直接当 DC 分断能力 |

### 电源候选特别说明

ALITOVE 官方页面：
https://alitove.com/products/alitove-5v-10a-power-supply

官网列 SKU `5V-10A-BK`，输入 AC 100–240 V，输出 DC 5 V / 最大 10 A / 最大 50 W，圆孔 5.5×2.5 mm、说明兼容 5.5×2.1 mm、中心正极。这些是页面规格，不是我们完成的电气测试。

**不会只因为写着“10 A”就推荐直接接六颗电机。** 电源输出线、圆孔接插件、转接头、分配板和最细分支都要核持续/峰值电流与温升；没有足够额定证明就换可验证的电源分配方案，不让一个小端子承担整臂电流。该页的 FCC/CE/RoHS 描述不代替课堂所需的市电安全认证核验；优先向供应商索取适用地区认证/文件。禁止学生接裸露市电开关电源端子。

仅 5 V 电压“看起来匹配”不够：还须记录欠压、纹波、保护触发、启动浪涌和断电后重力坠落行为。必要时先用带限流、经过学校认可的实验室电源做单关节台架，再确定整臂电源 SKU。

## 3. 如果希望先买一套能上课的成品臂

| 候选路线 | Amazon 搜索入口 | 适合的课堂用途 | 与 Microduck 的边界 |
|---|---|---|---|
| **Hiwonder / LewanSoul LeArm 当前明确 SKU 套件** | [搜索 Hiwonder LeArm robotic arm kit](https://www.amazon.com/s?k=Hiwonder+LeArm+robotic+arm+kit) | 桌面抓取、关节链、手柄/PC 示教入门，先熟悉机构 | 比旧拍卖更容易向卖家索取资料，但不能默认页面一定是同一版本；确认姿态轴/夹爪、舵机、原配电源和反馈后再适配 |
| **SO-101 follower 套件** | [搜索 SO-101 follower robot arm kit](https://www.amazon.com/s?k=SO-101+follower+robot+arm+kit) | 开源装配/标定/示教；后续复用案例组织方式 | 官方 follower 为 6×STS3215，需自己的供电/控制适配；不是 Dynamixel 电气直兼容，也不自动获得 RLX/PPO 接口 |
| **SO-101 leader + follower 组合** | [搜索 SO-101 leader follower kit](https://www.amazon.com/s?k=SO-101+leader+follower+kit) | 用一只手示教另一只，采 BC/DAgger 数据 | **leader+follower 不等于两只执行任务的协作臂**；双臂协作通常需要两只 follower，各自反馈/供电，再决定是否另加示教设备 |
| **Waveshare RoArm-M2-S** | [具体商品 B0CMLNPH8V](https://www.amazon.com/dp/B0CMLNPH8V) | 桌面抓取与 JSON 任务网关适配 | 低自由度、独立 12 V 供电，约 826 g；不适合直接挂载原机，需新合同/模型 |

只追求“6DOF”“APP 控制”不够：购买前要求卖家提供可编程接口文档、读取实测状态的方式、控制更新率、丢线/急停行为。只能播放动作组而读不到关节反馈的套件，可以做示教课程，但不作为闭环 PPO 真机验证主平台。

## 4. 我的采购次序建议

1. **先不买双臂整套。** 把需求按“只做仿真课堂”与“要做硬件台架”分开，仿真部分不依赖实物立即到货。
2. 若选兼容主线，先采购/借用一颗已核对的 XL330 + U2D2 + 受控 5 V 电源，验证机载 USB、反馈、标定、单关节热/负载和保护；然后才冻结六颗方案。
3. 若课堂更急，选卖家能给出完整型号/资料的 LeArm 成套包，使用原厂匹配电源与控制板，不接鸭子舵机总线；通过单独网关做任务级集成。
4. 单臂抓取和热/供电通过后，再买第二只执行臂；两臂的新增对象观察、碰撞和负载分担检查不能省略。
5. 本阶段采购目标始终是**桌面固定**；任何机身挂载零件都等 P6 质量/惯量/稳定性评审后再定。

## 5. 每个商品页必须补全的记录

商品完整标题、ASIN、链接、卖家、SKU/修订、关节数与实际执行器数、单臂/双臂/leader/follower 包装、每颗舵机型号、额定电压、原配电源、反馈协议、USB/串口支持、公开 SDK 文档、连接器与线序、整机质量/尺寸、附带夹具、退换条件、当时价格/运费/税费、核验日期。

当前这些字段不齐的条目均为“候选/待核”，不能进“可直接接 Microduck”名单。Amazon 补货或链接变化不改变兼容性标准；遇到缺货优先找精确原型号的授权渠道，而不是自动购买搜索结果第一项。
