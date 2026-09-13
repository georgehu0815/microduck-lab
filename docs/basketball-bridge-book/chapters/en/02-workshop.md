# Set up a reproducible workshop

## Materials: simulation assets are not a hardware shopping list

The following is the **simulation bill of materials**. It identifies objects and files you need to reproduce the software experiment. It is not a certified mechanical design, purchasing recommendation, or construction specification for putting an expensive robot above a moving plank.

| Item | Quantity | Role and limitation |
|---|---:|---|
| Microduck full-collision robot MJCF and referenced meshes | One model per scene | Fourteen actuated joints; inspect the packaged XML, do not substitute an unrelated robot. |
| Free basketball | One | Radius 0.12 m, mass 0.62 kg; collision sphere and separate textured visual mesh. |
| Basketball OBJ and PNG | One each | Supplied playground assets; appearance is separate from sphere contact physics. |
| Suspended plank | One | Length 1.10 m, width 0.13 m, thickness 0.03 m, mass 0.45 kg. |
| Cable tendons | Four | Physical constraints/support, not only decorative lines. |
| Launch and destination platforms | Two | Static support and unambiguous start/end regions. |
| Visible gantries | Scene components | Show anchor placement; non-colliding visual structure. |
| Ground plane, lighting, camera | One scene setup | Ground contact is detected as failure, not hidden by the camera. |
| Supplied basketball actor and reference ONNX | One pair | Required mature initializer for the measured local experiment. |
| Supplied walking ONNX | One | Exact bridge actor/normalizer initializer. |

No physical timber grade, cable breaking load, fastening specification, battery limit, servo thermal envelope or safety factor was measured for this book. Do not infer those from simulated dimensions. Real tests would need engineered restraints, an exclusion zone, an emergency stop, torque/current limits, qualified supervision, and separate sim-to-real validation. The local harness is a prototype, not deployment approval.

## Computer and software assumptions

The recorded path uses an Apple-Silicon Mac. MuJoCo steps the physical scene on CPU; basketball PPO uses PyTorch CPU; bridge PPO uses MLX/Metal. “No CUDA required” does **not** mean “every part runs without a GPU runtime on every operating system.” This edition does not claim a clean Linux reproduction of the bridge learner.

Use Python 3.12, not whichever `python` happens to be first in your shell. Node/npm run the viewer. `uv` manages Python environments. Pandoc, XeLaTeX, `pdfinfo`, `pdftotext`, and `pdftoppm` build and inspect the book; they are not needed for training. Pillow and Matplotlib generate plots, not policy actions.

The measured Python package versions are recorded automatically in `environment.json`. This is a **recorded environment inventory**, not a tested universal resolver lock. Source changes outside a Git commit are bound by file SHA-256 values in `source-manifest.json`; a commit hash alone would miss the new uncommitted skill implementations.

## Acquire the workspace, not just the PDF

Obtain this workspace checkout from its maintainer, including its `rlx` checkout, upstream sibling repositories and the user-supplied playground references. Do not invent a public download URL for policy weights. Before installing anything, inspect the local prerequisites:

```bash
test -f microduck_local/src/microduck_local/contract.py
test -f rlx/rlx/mjlab_microduck/robot/microduck/robot_allcollisions.xml
test -f microduck/policies/alpha_walking.onnx
test -f microduck-playground/artifacts/basketball/checkpoint.pt
test -f microduck-playground/artifacts/basketball/policy.onnx
test -f microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.obj
test -f microduck-playground/src/mjlab_microduck/robot/assets/basketball/basketball.png
command -v uv node npm
```

If a test fails, stop at that missing asset. Changing the actor architecture or replacing its normalizer with zeros does not repair a missing checkpoint. The bridge scene itself uses the packaged robot assets; basketball additionally requires the supplied playground OBJ/PNG. Walking baselines and other harness paths may require the upstream sibling model checkout.

For a new environment, the repo's documented setup route is:

```bash
cd rlx
uv sync --python 3.12
uv pip install --python .venv/bin/python -e ../microduck_local
.venv/bin/python examples/ppo_microduck_studio.py --help
cd ..
cd duck-viewer
npm ci
cd ..
```

These installation commands require network/package availability. The book's validation uses the existing recorded environment and does not claim to have recreated a fresh machine offline. Preserve lockfiles; compare package versions and test before changing a resolved dependency. Do not copy a virtual environment between computers.

The recorded executable is `rlx/.venv-microduck/bin/python`. The clean setup above creates `rlx/.venv/bin/python`. Substitute the latter consistently in the case chapters if that is your environment. The following variable makes the distinction explicit:

```bash
PYTHON="$PWD/rlx/.venv-microduck/bin/python"
test -x "$PYTHON"
"$PYTHON" docs/basketball-bridge-book/tools/preflight.py
"$PYTHON" docs/basketball-bridge-book/examples/ppo_arithmetic.py
```

Preflight checks required local assets and reports package versions without importing Metal. It does not prove that a subsequent Metal or graphics context can be opened. A restricted sandbox may permit file reads but reject Metal or CoreGraphics. That is a runtime permission problem, not proof of a broken actor.

## Separate preflight, smoke, training, and evaluation

| Stage | Smallest useful proof | What it cannot establish |
|---|---|---|
| Preflight | Assets exist, expected files and packages can be located. | Dynamics, graphics, or policy quality. |
| Contract test | Shapes, resets, curriculum and evaluator behavior are correct on fixtures/probes. | A trained skill is reliable. |
| Smoke training | A few transitions reach update, checkpoint and ONNX output. | Skill acquisition or convergence. |
| Pilot training | A bounded real experiment changes weights and records metrics. | Crossing/steering unless separately evaluated. |
| Deterministic evaluation | Exported policy meets or fails explicit conditions. | Robustness outside tested conditions. |
| Visual review | Contact, body posture and actual motion agree with metrics. | An unseen seed or hardware scenario. |

Run targeted contract tests before spending a long time training:

```bash
rlx/.venv-microduck/bin/python -m pytest -q \
  rlx/tests/test_basketball.py \
  rlx/tests/test_basketball_policy.py \
  rlx/tests/test_basketball_evaluation.py \
  rlx/tests/test_balance_adapter.py \
  rlx/tests/test_bridge.py \
  rlx/tests/test_bridge_evaluation.py \
  rlx/tests/test_bridge_bootstrap.py
```

These tests exercise the named implementation boundary. They do not assert that every unrelated test in the workspace passes. For this publication, current test results and book-only checks are saved separately under `verification/`.

## Before the first step: inspect the scene

Run the book's environment probe to compile both models, check the observation/action shape and sample four finite transitions. It applies zero action and records a construction check; it does not load a learned policy or prove balance:

```bash
rlx/.venv-microduck/bin/python \
  docs/basketball-bridge-book/examples/probe_scenes.py
```

Each environment owns mutable MuJoCo data. Shared immutable model resources can save memory, but recurrent hidden states and per-duck simulation states must never be shared. A viewer can display two ducks using the same ONNX session while each maintains independent LSTM memory.

## Preserve outputs before a new run

Use new output directories, such as `rlx/runs/book/basketball-YYYYMMDD-HHMMSS`, replacing the date token with a fresh run identifier. Check the destination does not already exist. Never train into the evidence directories printed in this book.

For each run, preserve at least:

- the executed command, software inventory, source hash and source-policy hash;
- checkpoint and associated metadata, final ONNX and parity result;
- full update/episode logs, not only a smoothed reward graph;
- evaluation JSON including failures, seeds, horizon and assistance settings;
- video, contact sheet and render metadata bound to the evaluated policy.

A checkpoint is not automatically a complete optimizer resume. In these two pathways, source actors are transferred but critics/optimizers are new. Re-running a command may reproduce the protocol without reproducing identical floating-point weights. Different numerical libraries, package versions, scheduling and simulator versions can change trajectories. The honest goal is a traceable experiment and matching acceptance criteria, not an unjustified promise of bitwise retraining.

## Troubleshooting without changing the question

| Symptom | First check | Do not do this |
|---|---|---|
| Missing ball appearance | OBJ/PNG paths and group-2 visual geoms. | Replace the physical sphere with decorative animation. |
| Model compiles but viewer is flat ground | Per-policy environment kwargs and lab scene extraction. | Claim bridge playback from the policy name alone. |
| Actor output diverges after export | Normalizer divisor, observation order, h/c reset and ONNX metadata. | Ignore parity because the duck looks plausible. |
| `No Metal device available` | Execute MLX in an authorized local runtime. | Change algorithms silently and call it the same experiment. |
| CoreGraphics rendering failure | Authorized graphics context and installed rendering backend. | Substitute a schematic as a measured rollout screenshot. |
| High reward, no crossing | Contact-valid full-horizon evaluation and rollout inspection. | Raise a reward coefficient before checking exploration. |
| JSON/stdout parsing mismatch | Prefer structured `training-metrics.jsonl` and result files. | Treat incomplete log parsing as a flat zero loss curve. |

**Exercise.** Why preserve both policy hashes and video evidence? **Answer:** a good-looking video can come from a different policy. Hash-bound metadata connects the claim, checkpoint, deterministic export, evaluation and visible trajectory.
