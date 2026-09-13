# Verification / 验证记录

2026-09-11 · Author: George Hu

## Application and artifact gates

- `evidence/verification.json`: **passed**. Eight scenario cards, saved runs,
  recipe dialogs, source-bound videos, eight mounted live duck components,
  progressing physics, pencil/canvas/lower-beak geometry, isolated drawing
  payload, reset/ink regeneration, working WebGL, and 390px mobile layout.
- Drawing preview time advanced from 0.140 to 0.218 seconds in the browser.
  Its badge remains **DIAGNOSTIC · DRAWING NOT ACCEPTED**.
- `evidence/workflow.json`: **passed**. Both trained runs evaluated on 32
  held-out episodes plus matched BC baselines, rendered through the actual
  Studio API, and returned source-bound video evidence. Byte-range video
  requests returned HTTP 206. A separate real 768-transition PPO smoke job
  emitted live training progress and produced checkpoint/ONNX files.
- `evidence/main-lab.json`: default Lab 8788 exposes drawing policies after
  restart. Its saved roster is byte-for-byte unchanged.
- `evidence/readme-media.json`: all 19 animated GIF links across the two
  READMEs resolve to multi-frame images, including all eight cases and the
  separately labeled teacher reference.

The eight-scene overview camera shows only part of the spatially separated
roster at once. The receipt verifies all eight components and environments;
use camera navigation for individual close-ups. Failed learned policies may
fall and reset normally: this is not a transport failure. The no-global-reset
exception after 30 seconds is tested at the Lab contract level, not falsely
inferred from a failing policy surviving that long.

## Automated tests

- Focused Python environment/reference/training/export/Studio playback suite:
  **83 passed**; full output in `evidence/python-tests.txt`.
- `duck-viewer`: **171 tests passed**; ESLint and TypeScript passed after the final
  live-roster contract-label fix. Four added regressions cover empty, original,
  drawing-only, and mixed rosters. Production build passed again after this fix.
  Logs: `evidence/viewer-tests.txt` and `evidence/viewer-build.txt`.
  Drawing labels identify its actual SB3/PyTorch CPU trainer, separate
  256 → 256 Tanh actor/critic networks, and `policy.onnx` filename.
- Python compile checks, Ruff on the new environment/report/tests, shell syntax,
  Node script syntax, and `git diff --check` passed.
- Non-blocking warnings: legacy TorchScript ONNX exporter deprecation and the
  existing Basketball dynamic-filesystem tracing warning in the Next build.
- This is a focused regression result, not a claim that every unrelated
  historical test in the entire multi-checkout workspace passes.

## Physical and learned-skill evidence

The current contact-only teacher completes a 32-second unassisted drawing.
Its actual ink, full video, raw contact CSV/JSON and contact sheet were inspected.
The two learned PPO trials **do not** pass drawing acceptance. See `RESULTS.md`
for measured BC/PPO differences and limitations; software integration does not
imply learned mastery or hardware readiness.

## Re-run

From the workspace root, with Studio 63317 and an isolated eight-policy Lab on
8791 running:

```bash
node scripts/verify-eight-cases.mjs
node scripts/verify-drawing-workflow.mjs
SMOKE_RUN="drawing-smoke-$(date +%s)" node scripts/verify-drawing-workflow.mjs --smoke
PYTHONPATH=rlx:microduck_local/src rlx/.venv-microduck/bin/python -m pytest \
  rlx/tests/test_drawing.py rlx/tests/test_drawing_reference.py \
  rlx/tests/test_microduck_drawing_pipeline.py \
  microduck_local/tests/test_drawing_live_lab.py \
  microduck_local/tests/test_studio_policies.py \
  microduck_local/tests/test_balance_playback.py -q
```

Use a new smoke run name: the trainer intentionally refuses to overwrite an
existing non-empty run. The browser harness resets only the isolated Lab, not
the default roster. Training, evaluation and rendering remain simulation-only.
