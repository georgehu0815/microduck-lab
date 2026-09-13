# WingPod / 奶油小鸭：协调外观评审包

这次只重新设计**外观层**，不重新训练、不移动电机，也不改已有网球实验的控制器。
新外观采用暖瓷白机身与关节舱、蜂蜜黄翼形连杆和脚掌、石墨色关节环。
小脸固定在已有传感器包络位置；夹爪接触垫保持外露。

## 查看成果

- 统一入口：`../../../artifacts/microduck-arm-v1c/appearance/wingpod-v1/index.html`
- 前后对比：`../../../artifacts/microduck-arm-v1c/appearance/wingpod-v1/before-after.jpg`
- 中英双语图册：`../../../artifacts/microduck-arm-v1c/appearance/wingpod-v1/wingpod-review.pdf`
- 旋转视频：`../../../artifacts/microduck-arm-v1c/appearance/wingpod-v1/turntable.mp4`
- 原理与实体放行清单：`DESIGN.md`

## 保持什么、改变什么

| 项目 | 本轮处理 |
|---|---|
| 肩—肘 / 肘—腕轴距 | 保留 55 / 50 mm |
| 宽夹爪开口、TCP、接触垫 | 保留 75 mm 开口、55 mm TCP 与原接触面 |
| 执行器与控制 | 保留 10 个腿部 + 4 个臂姿态 + 1 个夹爪执行器；没有新增腰关节 |
| 基座、惯量、质量、碰撞、关节限位 | 不修改编译模型物理数组 |
| 承力路径、电池、电源、线束 | 不在本轮改动，不冒充重新验证 |
| 电机舱与底座外观 | 增加圆润分段饰壳及肩鞍的渲染候选 |
| 配色 | 机身、腿部和机械臂使用统一色系；接触垫颜色保持原样 |

外观饰件通过 MuJoCo **可视化场景**添加，而不是物理 `geom`。
它们不是“零重量实体零件”。保持功能的证据仅针对原仿真机械结构和控制器。

## 设计交付物

- `appearance-spec.json`：圆润舱体、羽片、关节环和脸部细节的统一参数。
- `shell-dimensions.csv`：10 个候选圆润舱体的局部坐标、外包络及圆角，单位 mm。
- `wingpod-review-shells.scad`：按刚体局部坐标分组的**评审用爆炸排版**；不是装配坐标中的整机模型。
- SCAD 壳体以 1.2 mm 壁厚起点建模；脸部、羽片、环形细节为实心外观包络。
- 手指支杆只是现有几何重新配色，没有制造新的夹爪零件。
- SCAD 尚未通过 OpenSCAD 编译、加工工艺或实体安装验证；不可直接当生产文件。
- 装配孔、卡扣、分型面、通风孔、布线槽仍需与真实硬件联调后设计，不能从渲染图推断已有。

## 功能证据的边界

`replay-summary.json` 记录原 v13 的 30 个正例、2 个负对照的逐动作物理复放。
`replays/*/replay-validation.json` 保存每回合球位置、记录关节角、时间误差、源文件哈希和物理子步计数。
`zero_error` 仅针对这三类数值；不表示接触力、基座全部状态或其他遥测均逐字段对比。
新版完整视频为 `nominal-0`、`small-5`、`large-6`；其余回合为无视频的物理复放。
`media-verification.json` 保存 MP4 全帧解码及文件哈希；旋转展示不是任务成功证据。

这些结果不能扩展为“新增实体壳体通过全部机械臂技能”，也不能扩展为真机、PPO、
自由落体投球或新载重验证。沿用任务的实际释放仍为靠近桶底的支承释放。

## 实体落地前的顺序

1. 先执行不增重的色彩与表面处理评审，保护接触面及散热界面。
2. 测量舵机、连接器、螺钉、转轴及线缆真实包络；调整圆角，不能隐藏硬件穿壳。
3. 完成分壳、紧固、拆装与热设计，壳体不能作为原设计之外的承力替代件。
4. CAD 导出实际质量、质心、惯量；给饰壳增加真实碰撞体，而不是沿用渲染覆盖层。
5. 对全关节范围、张合夹爪、下探、载球走动、桶口释放进行含公差净空扫描。
6. 重跑扭矩、动态平衡、碰撞、热、电气、全部已有任务及负对照，合格后才放行。

**尚无实体饰壳质量预算闭环或净空通过证据，所以本包不是制造放行。**

## 复现

从工作区根目录运行；Mac 离屏渲染需要可用的图形连接：

```bash
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python scripts/build_wingpod_design.py \
  --output artifacts/microduck-arm-v1c/appearance/wingpod-new-run \
  --artwork --turntable --replay
PYTHONPATH=.:microduck_local/src rlx/.venv-microduck/bin/python \
  scripts/publish_wingpod_design.py \
  --output artifacts/microduck-arm-v1c/appearance/wingpod-new-run
PYTHONPATH=.:microduck_local/src rlx/.venv-microduck/bin/python \
  -m pytest rlx/tests/test_wingpod_appearance.py -q
```

使用新的输出目录。复放不覆盖已有回合，避免将旧视频混入新结果。

## English summary

WingPod is an isolated visual redesign, not a new physical robot release. Porcelain pods,
honey feather accents, dark joint rings and matching feet unify the existing single-arm duck.
The frozen 55/50 mm links, 75 mm open grip, 55 mm TCP, 15 actuators and physical/control
parameters are retained. Original contact pads remain uncovered and unchanged.

The review CAD and JSON share the decorative geometry. CAD is a local-frame exploded study,
not an assembled or fabrication-ready robot. Shell weight, inertia, tolerance-aware clearance,
cable routing, cooling, fastening and hardware safety remain unverified. The original v13
actions are replayed in real simulation dynamics; selected complete videos show the new
render layer. This is not new PPO training, physical-shell qualification or a hardware test.
