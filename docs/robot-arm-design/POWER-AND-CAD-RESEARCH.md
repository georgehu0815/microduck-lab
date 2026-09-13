# 供电与官方 CAD 补充核验

核验日期：2026-09-12。**不改变 BOM 的“电源/保护/制造未放行”状态。**

## 已查到的候选，不是课堂可直接接线套件

MEAN WELL `LRS-75-5` 官方数据表列出输出5 V、14 A、70 W，输入85–264 VAC。它是需要装入最终设备的电源组件，端子处有暴露市电，不是供学生直接使用的封闭桌面电源适配器。

官方数据表第4页端子图必须按图看：**1 AC/L、2 AC/N、3 FG、4 −V、5 +V**。不能使用跨列错位的 PDF 文本提取结果接线。本项目不提供该市电组件的学生装配教程，也没有将它加入已批准物料。

官方资料：

- MEAN WELL `LRS-75-SPEC.PDF`：https://www.meanwell.com/Upload/PDF/LRS-75/LRS-75-SPEC.PDF
- ROBOTIS XL330-M288-T：https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- ROBOTIS U2D2：https://emanual.robotis.com/docs/en/parts/interface/u2d2/
- U2D2 Power Hub：https://emanual.robotis.com/docs/en/parts/interface/u2d2_power_hub/
- JST EH：https://www.jst-mfg.com/product/index.php?lang=2&series=58

U2D2 Power Hub 官方最大10 A不是六路保险保护证明，也不能自动承接14 A电源的全部故障电流。JST EH系列公布3 A（AWG22条件），不能把6颗舵机的8.82 A堵转算术总和串在一个入口连接器上。真实分支电流、温升、保护动作与供电跌落均需实测。

仍未核准：可用于此负载且输出连接器额定明确的封闭桌面5 V电源、支路保险、直流急停触点/接触器、线径、外壳、拉力固定及接地实现。未知型号、参数保持空白和 `blocked`，不以网购标题补全安全设计。

## 官方制造接口文件

ROBOTIS e-Manual 列出 XL330 PDF/DWG/STEP。此次请求 STEP `download.php?no=1987` 与 PDF `download.php?no=1986`，返回1272字节HTML错误页，而非STEP/PDF；没有把错误页面保存成“CAD”。见 [检索清单](../../hardware/md-arm-t1/vendor/manifest.json)。

工作区另有上游 `microduck_rl/src/mjlab_microduck/robot/microduck/assets/xl330.stl` 和舵机测试台模型可作视觉/仿真参照。它们不是经供应商修订与购入实物核准的安装孔制造图，不能从网格猜公差或批准螺钉长度。

当前交付的是可编辑 SCAD、尺寸 SVG、概念 STL 与独立 MuJoCo模型。**没有输出已放行的精密机械加工图、已审核 PCB Gerber 或安全认证电路。** 下一阶段必须获得有效官方修订图并测量实物，才能冻结制造接口。
