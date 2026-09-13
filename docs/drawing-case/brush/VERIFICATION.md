# Final brush verification / 彩色画刷最终验证

Author: George Hu · Experiment date: 2026-09-11

Release: `brush-color-v2-03`; case 8 tool variant, not a ninth case.
ONNX SHA-256: `6e083a2a1a0e3114016be9ef0d377d9e9245648ca174303c5f8d0340bdf4500e`.

## Learned skill / 学习结果

| Check | Evidence | Result |
|---|---|---|
| Final PPO ONNX, 10 conditions × 4 seeds | [eval.json](eval.json) | 40/40 accepted |
| Matching BC/DAgger ONNX | [bc-eval.json](bc-eval.json) | 40/40 accepted |
| Separate BC/DAgger video, generated 2026-09-12 | [rollout_bc.mp4](media/rollout_bc.mp4), [receipt](media/render_bc.json) | Full painting passed; 1,200 decoded frames, 10fps, 120 seconds |
| Null and open-jaw controls | Both evaluation reports | Both fail as required |
| Full physical-contact painting | [render receipt](media/render.json), [raw trace](media/raw_trace.json) | 119.96 simulated seconds, all four colors, unassisted |
| Actual artwork inspected | [painting](media/painting.png), [contact sheet](media/frame_sheet.png) | Recognizable colored swimming duck |
| Real training history | [training.json](training.json), [reward/loss plot](media/reward-loss.png) | 49,152 PPO transitions; eight drift-bounded updates |

每种颜色均须通过至少 95% 的覆盖率和精度门槛，距离容差 1.5mm；同时满足完整回合、
不摔倒、不掉笔、夹持比例、力、压缩、抬笔漏画和顺序蘸色检查。不是只看累计 reward。
BC 已经成功，PPO 保留了成功行为；没有证据证明 PPO 比 BC 更好。

Release 03 was trained with its archived earlier pipeline. The final hardened evaluator
checks immutable ONNX bytes, embedded/checkpoint/sidecar agreement, all six archived source
files, matching current physics/reference/actor, and unchanged files during evaluation.
Both final reports have `provenance_errors=[]` and `training_pipeline_source_match=false`.
This is hardened evaluation of an older-source training run, not retroactive hardened training.

## Viewer and regression checks / 网页及回归检查

| Check | Evidence | Result |
|---|---|---|
| Real Studio eval, render, ranged MP4 delivery | [workflow.json](evidence/workflow.json) | Passed; HTTP 206 |
| Eight cards and eight live ducks | [viewer.json](evidence/viewer.json) | Passed |
| Brush live contact-color payload | Same browser report | Correct 93/15 contract, physical palette, zero assistance |
| Selected brush video plays | [video screenshot](evidence/viewer-video.png) | Matching model hash; playback time advances |
| 390px mobile layout and browser errors | [mobile screenshot](evidence/viewer-mobile.png) | No horizontal overflow; no page errors |
| Main Lab catalog and saved roster | [main-lab.json](evidence/main-lab.json) | Release available; original roster unchanged |
| Python drawing/brush/live/catalog regressions | [python-tests.log](evidence/python-tests.log) | 120 passed |
| Focused pipeline provenance regressions | [provenance-tests.log](evidence/provenance-tests.log) | 9 passed, included in Python total |
| Viewer tests | [viewer-tests.log](evidence/viewer-tests.log) | 181 passed |
| ESLint / TypeScript | [lint.log](evidence/lint.log), [typecheck.log](evidence/typecheck.log) | Passed |
| Production build | [build.log](evidence/build.log) | Passed |

Browser live verification is a bounded stream/UI smoke test, not a complete live 120-second
multi-policy performance test. The full painting is independently proven by the source-bound
render and 40-episode gate. The Studio API receipt predates the final provenance-only CLI
refresh; it refers to the same unchanged policy and render, not a different training run.

Verification also corrected a dropped live-color payload and a stale blanket “no accepted
drawing result” label. The browser audit uses the actual Review result link and scopes playback
to the selected run; it does not confuse autoplaying Dance previews with the Brush rollout.

## Reproduce / 复验

From the workspace root, with the documented Python environment and running Studio/Lab:

```bash
bash scripts/train-brush.sh eval --onnx rlx/runs/studio/drawing/brush-color-v2-03/policy.onnx --out /tmp/brush-eval.json --episodes 4 --seed 10001
bash scripts/train-brush.sh eval --onnx rlx/runs/studio/drawing/brush-color-v2-03/bc.onnx --out /tmp/brush-bc-eval.json --episodes 4 --seed 10001
STUDIO_URL=http://127.0.0.1:63317 BRUSH_LAB=127.0.0.1:8792 node scripts/verify-brush-viewer.mjs
cd duck-viewer && npm test && npm run lint && npm run build
```

The browser script requires the existing Playwright installation; override
`PLAYWRIGHT_REQUIRE_ROOT` with a package.json whose dependency tree contains Playwright.
See the [Chinese](README.md) and [English](README.en.md) guides for data generation and full training.

## Limits / 边界

- Fixed authored artwork, stroke targets and dip order; learned motor feedback, not learned vision or arbitrary composition.
- One passive spring-contact tuft and decorative hair geometry; no individual-bristle finite-element model or fluid mixing.
- Preloaded free brush held by frictional jaw contact; no pickup, washing, runtime teacher, weld or tool teleportation.
- Seed-sensitive training; failed runs are retained. Forty accepted test episodes do not guarantee every retraining seed succeeds.
- Simulation only. The independent 93-observation/15-action brush model is not the stock 61/14 hardware contract.
