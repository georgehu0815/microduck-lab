# Independent scenario training pipelines

Each entrypoint runs its own input checks, initialization, PPO training, ONNX
export and verification, deterministic ONNX evaluation, and MP4/contact-sheet
generation. **No web server, browser, or other scenario script is required.**

From the repository root, run whichever scenario you want:

```bash
./scripts/train-dance.sh
./scripts/train-swing.sh
./scripts/train-running.sh
./scripts/train-stilts.sh
./scripts/train-backflip.sh
```

These commands default to **Full**, which can take substantial time. They
create fresh timestamped runs; they do not find and silently reuse old policies.
Run them separately rather than concurrently on a memory-constrained Mac.

## Additional balance scenarios

Studio also includes Basketball and Suspended bridge (seven cases in total).
Basketball uses a separate PyTorch recurrent-policy adapter rather than the
five-scenario pipeline above. Bridge uses the shared MLX Studio trainer:

```bash
./scripts/train-bridge.sh train --bridge-curriculum --total-timesteps 32768 \
  --num-envs 4 --num-steps 128 --num-minibatches 4 --backend dummy \
  --no-domain-rand --no-obs-noise --no-action-delay --no-random-yaw \
  --output-dir rlx/runs/studio/bridge/bridge-local
```

The bridge starts from the shipped walking actor with fresh local critic and
optimizer, not a previously mastered bridge policy. Curriculum changes only
physics. Evaluation and rendering disable assistance. See
`docs/bridge-showcase/README.md` and `docs/basketball-showcase/RESULTS.md` for
contracts and limitations. These two entrypoints do not accept the legacy
pipeline's `--profile` or `--dry-run` options.

## Requirements

Use the workspace's installed Python environment containing RLX, MLX,
`microduck_local`, MuJoCo, ONNX, ONNX Runtime, Pillow, ImageIO, and its FFmpeg
binary. The default interpreter is `rlx/.venv-microduck/bin/python`.

For another installed environment:

```bash
export MICRODUCK_STUDIO_PYTHON_DIRECT="/absolute/path/to/venv/bin/python"
```

See [RLX setup](../../rlx/README.md#train-and-render-the-microduck-dance) for
installation. These scripts do not install packages or download dependencies.
On macOS, MLX needs an Apple Silicon GPU accessible to the process; a sandbox
or headless session that denies Metal access fails the runtime check before
training. The same robot-model dependencies as the native Studio CLI must be
available. Backflip additionally requires the sibling
`microduck/policies/alpha_stand.onnx` from the upstream Microduck checkout.

## Check the pipeline before a long run

```bash
./scripts/train-dance.sh --dry-run
./scripts/train-dance.sh --profile smoke --run-name dance-shell-smoke
./scripts/train-swing.sh --profile smoke --run-name swing-shell-smoke
./scripts/train-running.sh --profile smoke --run-name running-shell-smoke
./scripts/train-stilts.sh --profile smoke --run-name stilts-shell-smoke
./scripts/train-backflip.sh --profile smoke --run-name backflip-shell-smoke
```

`--dry-run` prints every command and input hash, creates no output directory,
and does not import MLX or start training. It also works with system Python
when the training environment is not installed.

**Smoke is not a skill test.** It defaults to 32 PPO transitions, one update,
32 evaluation steps, and a one-second video. Swing still exercises a bounded
teacher initialization; Backflip still exercises its stand-policy transfer and
critic calibration. Successful smoke reports `smoke_passed` / `not_assessed`,
never a learned-skill claim.

## Scenario-specific Full inputs and stages

| Script | Inputs and training stages | Default Full PPO transitions | Evaluation/video horizon |
|---|---|---:|---:|
| `train-dance.sh` | Tracked `rlx/assets/clips/dance-120bpm.json`; fresh PPO actor | 4,001,792 | 8 s |
| `train-swing.sh` | Tracked `swing-teacher.json`; BC/DAgger initialization, then PPO with frozen observation normalization | 524,288, plus separately recorded teacher initialization | 24 s |
| `train-running.sh` | Tracked Running base recipe, then its refinement recipe; forward command 0.75 m/s | 6,000,640 + 2,097,152 | 12 s |
| `train-stilts.sh` | Tracked Stilt base recipe, then its refinement recipe; 2 cm stilts and forward command 0.25 m/s | 6,000,640 + 1,048,576 | 10 s |
| `train-backflip.sh` | Native stand actor/normalizer transfer and critic calibration, then PPO landing refinement | 401,408, plus separately recorded initialization | 12 s |

These are local XML-actuator simulation recipes. Full evaluation applies the
existing scenario-specific skill gates; the command does **not** guarantee
that training will pass them. The default Dance input is the bundled looping
clip, not the historical Bachata excerpt in the README gallery.

Backflip is an **assisted launch → learned landing → pretrained stand handoff**
controller. The ONNX is its landing actor, not an unassisted whole-flip policy.
Swing's initializer uses privileged simulator state to label demonstrations;
the exported actor itself accepts only the standard 61 observations. Teacher
initialization must not be represented as PPO-discovered behavior.

## Supply your own inputs

All five scripts accept the same options; `--help` lists them.

```bash
./scripts/train-dance.sh --dance-clip /absolute/path/to/motion.json \
  --run-name my-dance

./scripts/train-running.sh \
  --recipe-json docs/remaining-scenarios-e2e/recipes/running-base.json \
  --run-name my-running

./scripts/train-running.sh \
  --recipe-json docs/remaining-scenarios-e2e/recipes/running.json \
  --init-from /absolute/path/to/base/running.safetensors \
  --run-name my-running-refinement

./scripts/train-stilts.sh --profile smoke \
  --output-dir "/tmp/my stilt pipeline" --seed 7 --eval-seed 101
```

- `--recipe-json` accepts one Studio-style recipe and replaces the default
  multistage recipe. A continuation recipe requires an explicit `--init-from`.
  Unknown fields and mismatched scenario IDs are rejected rather than ignored.
  The command-line `--profile` controls Full versus Smoke, not the JSON profile.
- `--init-from` requires the checkpoint and adjacent `.safetensors.json`
  metadata. It skips the default Running/Stilt base stage or Swing bootstrap;
  it always writes a **new** run and does not overwrite the input checkpoint.
- `--dance-clip` takes a native motion clip JSON. Its full duration is covered
  by Full evaluation/rendering; long clips therefore increase those costs.
- `--steps N` overrides the transition budget **per PPO stage**; `--num-envs N`
  changes training parallelism. The budget must cover a complete rollout and
  minibatches must divide it. Training rounds up to complete rollouts when
  necessary. Shortening a Full run does not disable skill gates.
- `--eval-envs N` changes evaluation lanes (default 8 Full / 1 Smoke).
- `--seed` preserves the recipe's seed (7 for the defaults); `--eval-seed`
  defaults to 101. Multiple lanes
  do not imply robustness when the nominal environment has no randomization.
- Relative input/output paths are resolved from your calling directory; the
  shell entrypoint itself works from any directory, including paths with spaces.

## Outputs and verification

Default location:

```text
rlx/runs/studio/<scenario>/<scenario>-pipeline-<UTC timestamp>/
  plan.json                  exact inputs, hashes, commands, and training recipes
  pipeline.json              stage results, final status, and any failure
  initialization/            teacher artifacts, when required
  stages/stage-1/            base checkpoint and telemetry for multistage runs
  <scenario>.safetensors     final actor/critic checkpoint
  <scenario>.safetensors.json
  <scenario>.onnx            actor with baked observation normalization
  training-metrics.jsonl     final stage's measured PPO updates
  verification.json          checkpoint, telemetry, contract and ONNX parity checks
  evaluation.json            evaluation of this exported ONNX, with its hash
  render.json                render settings, source hash, and reset count
  render/ep0.mp4
  render/ep0_sheet.png
  *.log                      per-operation logs, including media decode checks
```

Verification requires real optimizer updates and the requested number of
collected transitions, finite PPO metrics, a valid float32 **61 → 14** ONNX
contract, finite runtime outputs, and MLX/ONNX maximum absolute error no greater
than `1e-4` on deterministic sample observations. This proves artifact/runtime
consistency, not behavior quality. The native ONNX evaluator judges the skill.
The MP4 is decoded completely and its duration is checked; the contact-sheet
image must also be readable.

The scripts use Studio's run-directory convention, but do not fabricate its
API-managed render-review/provenance receipts. Use the generated files and
`pipeline.json` directly, or render/review the run through Studio to populate
that UI's evidence state.

## Failures and exit codes

- **0:** Full skill evaluation passed, or Smoke wiring passed. Read `profile`
  and `status` in `pipeline.json` to distinguish them.
- **2:** Evaluation failed. The pipeline still generates the MP4/contact sheet
  for diagnosis, then exits unsuccessfully. Rendering cannot turn a failed
  evaluation into a pass.
- **1:** Invalid input, missing runtime/asset, training/export/verification
  error, or media-generation failure. Downstream dependent steps do not run.
- **130:** Interrupted. The running child process is terminated and the
  interruption is recorded.

Existing output directories are rejected, including empty ones, so reruns
cannot overwrite saved evidence. Keep the generated logs on failure and choose
a fresh output directory after fixing the cause. Watch the **entire MP4** and
inspect the contact sheet before claiming that the duck performs the skill.
Neither a video nor a Full simulation pass authorizes physical-robot deployment.

## Regression tests

```bash
python3 -m unittest discover -s scripts/scenarios -p 'test_*.py'
for script in scripts/scenarios/run.sh scripts/train-*.sh; do
  bash -n "$script" || exit 1
done
```

The unit tests use synthetic inputs and mocked execution; they do not start
training. Use each script's Smoke profile for real training/export/eval/render
validation on a configured machine.
