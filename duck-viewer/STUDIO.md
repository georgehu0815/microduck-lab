# Microduck Studio

The root route is the local control plane for exactly seven macOS RLX Microduck recipes:

```text
dance | swing | running | stilts | backflip | basketball | bridge
    -> train -> deterministic evaluation -> render review -> ONNX handoff
```

Run it with:

```bash
cd duck-viewer
npm run dev
```

Open the printed local URL. Studio is one application: the Three.js simulation,
HUD, policy assignment, teaching, animation, recording, training, evaluation,
and artifact workflow all run on `/`. There is no iframe or separate viewer
product; legacy `/viewer` requests return to the Studio.

## Local execution

The Studio API runs the shared command surface in
`../rlx/examples/ppo_microduck_studio.py`, except Basketball train/eval/render,
which route to the recurrent `../rlx/examples/ppo_microduck_balance.py`
adapter. Run names are sanitized and artifact downloads are restricted to known
files under `rlx/runs/studio/<experiment>/<run>/`. Basketball training emits
its recurrent ONNX directly, so generic export is unavailable.

All seven recipes use RLX PPO on macOS. MuJoCo simulation runs on CPU and PPO
updates run through MLX on Apple Silicon; CUDA is not required. The bundled
Swing, Running, and Stilts videos and JSON files are labeled reference evidence
and remain separate from the selected run's artifacts.

Set `MICRODUCK_STUDIO_PYTHON` when the framework-linked Python 3.12 interpreter
is not `/usr/local/bin/python3.12`.

The smoke profile trains four timesteps and proves pipeline wiring only. It does
not prove that the policy learned the selected behavior. The deployment handoff remains
locked until an ONNX policy exists, deterministic evaluation passes, a rollout
is rendered, and a person confirms the visual review. Direct hardware upload is
intentionally unavailable.

## Basketball contract

Basketball train/eval/render uses the standalone recurrent adapter. Hidden and
cell state carry between ordinary control steps and reset with the episode.
Studio training defaults to an unassisted fine-tune (`--hold 0`, curriculum
disabled); the adapter's hold/curriculum ladder is available only through
explicit CLI training flags. Full evaluation currently covers zero-command
balance and `0.15` forward-command tracking. It does not yet evaluate lateral
or yaw steering. A verified 60-second balance rollout may be shown as
**BALANCE ONLY · STEERING NOT PASSED**, but Basketball never passes the full
task without an explicit steering verdict.

## Bridge pilot contract

Bridge Default Full mirrors the bounded `bridge-studio-02` pilot: 32,768 steps,
4 environments, 128 rollout steps, 4 minibatches, seed 7, initial standard
deviation `0.03`, learning rate `1e-5`, gamma `0.99`, clip `0.1`, 2 epochs,
zero entropy, max gradient norm `0.5`, and no randomization, noise, delay, or
random yaw. Observation normalization is frozen. Training alone receives
`--bridge-curriculum`; Full evaluation remains unassisted for at least 1,000
steps and 20 seconds on the final suspended narrow bridge.

Automatic bootstrap and checkpoint continuation are actor warm-starts with a
fresh critic and optimizer, not exact optimizer resume. The training pilot is
available for saved-run review even while its task evaluation or render fails.
It does not become a verified preview until the explicit bridge assessment and
source-bound render both pass.

## Backflip controller contract

Backflip is always **spotter-assisted launch → PPO landing → pretrained
stand-policy handoff**. The fixed Python protocol
`spotter-launch-landing-stand-v2` uses
`microduck/policies/alpha_stand.onnx` for an exact actor and observation
normalizer transfer into a fresh Full landing student, during the assisted
launch, and after the learned landing satisfies the handoff gate. Pipeline smoke
remains four steps and skips teacher initialization.

Default Full requests 400,000 PPO steps and normalizes to 401,408 complete
16-environment × 128-step rollout batches. It uses 4 minibatches, 2 epochs,
learning rate `3e-6`, initial standard deviation `0.03`, frozen observation
normalization, zero entropy, normalized rewards, seed 7, and
randomization/noise/delay/yaw disabled. Full evaluation runs 600 control steps
and rendering runs 12 seconds. The launch releases at the 5.3 rad threshold
(measured 5.376–5.380 rad) with angular rate in the 3.5–6.5 rad/s window;
external lift and pitch are launch-only.

Before Backflip PPO, Python collects 16 pretrained-teacher landing episodes of
600 steps (9,600 calibration steps), fixes the discounted-return RMS, and runs
10 critic-only calibration epochs. The transferred actor remains unchanged,
calibrated reward normalization is frozen for Backflip, and PPO starts with a
fresh optimizer. This prevents warm-start reward-scale shock; it does not by
itself establish positive PPO improvement.

Version 2 acceptance requires at least 5.8 rad of credited rotation before
handoff, followed by 10 consecutive stable PPO landing steps with no spotter,
stand override, or body support, and at least 2 seconds of stable standing. The
stand policy cannot finish credited rotation.
The exported `backflip.onnx` is the PPO landing stage only, not a complete or
hardware-ready controller.

Backflip reward shaping is active only during PPO landing. Its observable dense
terms reward upright orientation, target pose, and low-gyro settling, with
bounded penalties for joint speed, action changes, and action size. Every term
is zero during the assisted launch and pretrained stand-policy handoff.

## Swing recovery recipe

Selecting **Self-pumped swing** opens the preserved `swing-studio-01` baseline.
Play its latest-run MP4 inside the Evaluation gate. Its `2.516°` mean and
`5.661°` best span are valid rollout measurements, but the policy is rejected
because the target is approximately `163°` median span.

1. Select **Apply exact Discovery settings**. Studio creates
   `swing-curriculum-01` with 250,000 new PPO steps, 16 environments, seed 2,
   `1e-4` learning rate, `0.995` gamma, `0.10` clip, 3 update epochs, `0.002`
   entropy, and `1.0` max gradient norm. Initial-motion assistance is `12°`
   and `0.35 rad/s`; domain randomization, observation noise, action delay,
   and random yaw are off.
2. Select **Start RLX**. This is a new run; `swing-studio-01` is not changed.
3. Select **Run evaluation**. Studio evaluates 16 environments for 1,200
   control steps each, or 24 seconds at 50 Hz. Evaluation starts from rest:
   `0°` initial angle and `0 rad/s` initial rate.
4. Select **Render rollout**, then play the MP4 in the Evaluation gate.
5. Continue only when mean span is at least `10°` or best span is at least
   `20°`. Otherwise keep the 250k checkpoint as evidence and revise the
   curriculum or reward; do not spend another 500k steps automatically.
6. When the Discovery gate passes, select **Apply exact Consolidation
   settings**. Studio resumes the same checkpoint for 500,000 additional PPO
   steps, reduces assistance to `4°` and `0.12 rad/s`, and enables domain
   randomization, observation noise, and action delay. The PPO optimizer values
   remain unchanged.
7. Evaluate and render again. Continue toward 1M additional steps only when
   mean span reaches `30°` or best span reaches `60°`.
8. Do not deploy until the still-start evaluation reaches approximately
   `163°` median span, geometry remains valid, and the full video has been
   reviewed. A finite rollout pass alone does not satisfy this task gate.

The detailed guide is available at:

```text
docs/microduck-studio-experiments-guide/microduck-studio-experiments-guide.md
docs/microduck-studio-experiments-guide/microduck-studio-experiments-guide.pdf
```

## Verification

```bash
cd duck-viewer
npm run lint
npm run build

cd ..
node .omx/artifacts/visual-ralph/microduck-studio/verify.cjs
```

The browser script captures 1536 px and 390 px screenshots, checks horizontal
overflow, exercises recipe controls, verifies the read-only RLX status endpoint,
and confirms that the integrated WebGL workspace and its tool panels are mounted
directly in the Studio.
