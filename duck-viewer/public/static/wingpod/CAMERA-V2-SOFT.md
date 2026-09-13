# WingPod Camera v2 Soft / 奶油相机小鸭

## Appearance revision / 外观改版

Matches the user-supplied `duck-soft-frame-finished.png` reference: cream rounded
chest face, muted sage-gray lens rims, navy optical centers, tiny peach cheeks
and a honey-yellow duck beak. The camera remains on the fixed chest, not the arm.

按参考图将深色胸部面板改为奶油白，保留圆角；采用灰绿镜头边框、深蓝镜心、
淡桃色腮红和小黄嘴。相机仍安装在固定胸部，不随机械臂旋转。

| Surface / 部位 | Renderer RGBA / 渲染颜色 |
| --- | --- |
| Cream face / 奶油面板 | 1.00, 0.97, 0.87, 1 |
| Sage rim / 灰绿外圈 | 0.53, 0.60, 0.54, 1 |
| Inner rim / 内圈 | 0.31, 0.40, 0.39, 1 |
| Navy lens / 深蓝镜心 | 0.035, 0.12, 0.19, 1 |
| Peach cheeks / 桃色腮红 | 0.94, 0.68, 0.52, 1 |
| Honey beak / 小黄嘴 | 1.00, 0.76, 0.29, 1 |

These are render colors, not qualified paint/material specifications. Cheeks and
beak are decorative scene geometry, not camera optics or collision geometry.
The housing envelope and 13 mm optical-center spacing are unchanged proposals.
Mass, physical clearance, power and actual camera modules remain unqualified;
see `camera-v2-bom/README.md`. The 110-line candidate BOM has not gained any
qualified physical parts from this appearance change.

以上是渲染配色，不是已验收的涂料或材料规格。腮红和嘴仅为显示层装饰，不改变
相机光轴、碰撞模型、惯量、控制器或原有 15 个执行器。外壳与 13 mm 镜心间距
仍为设计候选；不能据此声称真实相机已选型、安装无干涉或真机验收通过。

## Evidence boundary / 证据边界

- Soft revision: fresh hero/detail images and a 20-second static-pose design video.
- Soft task video: one complete `nominal-0` action replay, compared against the
  existing recorded state trajectory; exact-match and full MP4 decode receipts.
- The 32-case gallery retains **original graphite-camera footage**: 30 positive
  tennis-return cases and two expected-failure controls. It is not a new 32-case
  soft-style training or evaluation run. No other arm skills are implied.
- No new PPO training, camera-based control, or hardware validation. The demonstrated
  release is low supported placement, not an airborne half-bin-height drop.

新配色单独生成预览和完整 nominal-0 捡球归桶回放；32 段历史视频继续保留原深色
相机版本，并在页面明确标注。负向对照预期失败不能计作捡球成功。所有证据均为
仿真或外观回放，不代表视觉策略训练或真机验证。

## Reproduce / 复现

Run from the workspace root; use a fresh output directory when rerunning the task
replay so existing receipts are not overwritten. Rendering jobs must run serially.

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python scripts/build_wingpod_camera_soft.py
rlx/.venv-microduck/bin/python scripts/publish_wingpod_camera.py --soft
```

Outputs: `artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2-soft/`.
Pages: `/wingpod/` and `/wingpod-v2/index.html`, English default with 中文 toggle.
