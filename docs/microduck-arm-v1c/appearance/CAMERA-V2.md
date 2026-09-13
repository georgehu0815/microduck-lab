# WingPod Camera v2 / 双摄眼睛版

## Scope / 范围

A separate appearance version matching the supplied chest-camera reference. Two
graphite/teal optical eyes and an amber indicator sit on the fixed chest below
the shoulder saddle. The original WingPod v1 is preserved. This is a MuJoCo
render-only design study, not fabricated hardware or a vision-trained robot.

独立外观版本：按照参考图，在肩部承台下方的固定胸部安装深色双摄“眼睛”和琥珀色指示灯。
保留 WingPod v1，不改变原有腿、机械臂和夹爪运动模型。本版本是仿真外观设计，
不代表摄像头已采购、装机或完成视觉策略训练。

## Candidate dimensions / 候选尺寸

| Item / 项目 | Proposal / 建议值 |
| --- | --- |
| Housing width × height × depth / 外壳宽高深 | 28 × 17 × 5 mm |
| Lens center spacing / 镜头中心距 | 13 mm |
| Local body / 固定父节点 | `arm_mount` |
| Lens direction / 光轴方向 | local +X; up +Z |
| Synthetic vertical field of view / 仿真垂直视场 | 70° |
| Hardware model, mass, power / 实物型号、质量、功耗 | Not selected / 未选定 |

Dimensions and FOV are design proposals, not manufacturer specifications. The
two RGB views are synthetic; stereo calibration and depth sensing are unverified.
The home-pose views show some arm occlusion, so camera tilt, occlusion throughout
motion, and ground-object visibility require further review before hardware selection.

以上尺寸和视场不是产品规格。左右 RGB 图是仿真视角，不代表双目深度功能已验证。
初始姿态中机械臂会部分遮挡视野；实物选型前需进一步检查俯角、全行程遮挡和地面目标可见性。

## Output / 输出

Directory: `artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/`

- `wingpod-camera-eyes.mp4`: 20-second, 960×720, 25 fps design video; overview,
  orbit and chest close-up. Static robot pose; camera moves, not a task rollout.
- `hero.png`, `camera-eyes-detail.png`: overall design and lens detail.
- `left-eye-rgb.png`, `right-eye-rgb.png`: actual simulated optical viewpoints.
- `camera-spec.json`: machine-readable candidate geometry and limitations.
- `appearance-validation.json`, `video-verification.json`: model invariance and
  full video-decode evidence.

视频是外观展示，不是捡球任务验证或真实机器视频。既有任务通过率不作为本版本新增验证结果。

## Pickup-and-bin task video / 捡球归桶任务视频

A separate full-episode action replay uses the camera-eye appearance:

`artifacts/microduck-arm-v1c/appearance/wingpod-camera-v2/tennis-return-nominal-0/rollout.mp4`

It reproduces the source `half-height-planned-v13/nominal-0` sequence: approach,
grasp the ground tennis ball, lift, carry, lower into the bin, release, and retreat.
The task camera stays fixed so the complete trajectory remains visible. Phase,
simulation time and outcome are overlaid on the video. This is not the static
20-second design-preview movie described above.

这个独立视频展示完整捡球归桶过程：靠近、抓取地面网球、抬起、搬运、下降入桶、松爪及后退。
画面标注任务阶段、仿真时间和结果，不是前述20秒静态外观展示视频。

The accompanying `camera-task-verification.json` records decoded frame count,
video/source hashes, monitored physics substeps, and replay errors for ball
position, recorded joint positions, and simulation time. Inspect
`sequence-contact-sheet.jpg` for representative stages. This verifies one
simulation replay, not all-case performance or new PPO/vision training. The
camera module remains visual-only; the controller does not use its RGB images.
The observed release is low, supported placement, not a half-bin airborne drop.

验证范围是单条仿真回放；摄像头不参与控制，没有新增PPO训练或全部案例评估。
实际松爪为低位支撑放置，不应描述为半桶高度自由落体投放。

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python scripts/render_wingpod_camera_task.py
```

The command refuses to overwrite existing task evidence. Use `--output` with a
fresh directory to reproduce it again. / 不覆盖已有证据；重复生成请用 `--output` 指定新目录。

## Reproduce / 重现

From the workspace root, using the installed local environment:

```sh
PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python scripts/build_wingpod_camera.py

PYTHONPATH=.:microduck_local/src OMP_NUM_THREADS=1 \
  rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_wingpod_camera.py rlx/tests/test_wingpod_appearance.py -q
```

macOS rendering requires an accessible graphics session. No new training is run.
Before fabrication, select a real sensor and verify its envelope, mounting,
cable routing, voltage/interface, bandwidth, mass/inertia, thermal limits and
whole-body balance. Render-only geometry does not validate physical clearance.

制造前必须落实真实传感器型号、安装尺寸、线束、电压与接口、带宽、质量惯量、散热及整机平衡；
仅渲染外观不会验证真实外壳碰撞间隙。本版本不改变既有15个执行器的物理模型。
