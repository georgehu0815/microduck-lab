# Read the evidence, not the celebration

## The evidence chain

![Evidence-preparation schematic: retained run records pass through schema checks and CSV extraction, then become observed plots or copied frames, and finally hash-bound book assets. This is not a measured rollout.](assets/diagrams/schematic-evidence-pipeline.png)

A training program can execute perfectly while learning the wrong behavior. A reward can improve while balance deteriorates. An exported model can differ from its training actor. A viewer can show the wrong scene. Therefore the final claim must pass several independent checks: correct model and inputs, real parameter updates, export parity, honest unassisted evaluation, and actual visual inspection.

The word **independent** matters. If you define “success” as high reward and then show the same reward as proof, you have checked one definition twice. Contact validity, no-fall duration, directional displacement and complete bridge crossing are separate observable tests of the requested physical task.

## Read real reward and loss curves

![Real recorded basketball training losses, with an explicit missing-history panel for reward. The two policies are independent warm starts, not a single continuous training run.](assets/evidence/basketball-training-curves.png)

The fixed-command basketball `summary.json` has 100 update records; the mixed-command run has 150. They include total loss, policy loss, value loss, entropy, approximate KL and gradient norm. **They do not include historical per-update reward or episode return.** This is an evidence gap, not permission to draw an attractive rising curve. The missing-reward panel is intentional.

Both runs begin from the supplied actor with fresh critics and optimizers. Plotting the fixed run followed by the mixed run on one cumulative axis would falsely imply continuation. The correct horizontal axes end at 102,400 and 153,600 transitions separately. Each fixed-command update corresponds to $8\times128=1024$ new transitions.

The separate basketball reward diagnostic supplied with this edition records deterministic per-step rewards from the existing policy. It is a **new diagnostic trajectory**, not a recovered historical training curve and not evidence of further optimization. Its CSV and receipt identify the policy, conditions and actual horizon. Do not splice it into the old training history.

![New deterministic basketball reward diagnostic, not historical PPO training reward. The source policy is unchanged.](assets/evidence/basketball-reward-trace.png)

![Real bridge pilot telemetry: collection mean reward and completed-episode returns are distinct from PPO loss components.](assets/evidence/bridge-training-curves.png)

The bridge log contains collection events, update events and completed-episode events. Collection `mean_reward` is a per-transition average. `mean_raw_return` is the reward summed over an episode, then averaged over completed episodes in that event. An episode that lasts longer can have a larger return even without better per-step behavior. An update `mean_loss` averages optimizer minibatch objectives. These three numbers do not have interchangeable meanings.

The bridge's total/value-loss panel uses a logarithmic vertical axis. Equal vertical distances represent equal multiplicative changes, not equal additive changes. A reduction from 1,000 to 100 has the same log-axis distance as 100 to 10. Negative policy losses cannot be placed on the same ordinary log axis and belong in their own panel.

Small policy loss is not proof of a solved policy. Advantage normalization deliberately centres advantages near zero; ratios close to one can make their average surrogate close to zero. A large value loss early in these runs is also unsurprising because the actor is mature but the critic is newly initialized. Entropy can be negative for continuous densities, as explained in the PPO chapter. A target KL is not a guarantee that a single optimizer step never overshoots.

**A good curve caption names five things:** the run, exact recorded field, horizontal-axis unit, any transformation/smoothing, and the claim it does not prove. The book's CSV extracts preserve the unsmoothed numbers. You can regenerate figures without running training.

## Deterministic evaluation: what passed?

![Recorded evaluation outcomes, not training scores. Basketball balance succeeds in the tested trials while controlled rolling and accepted bridge crossing do not.](assets/evidence/evaluation-outcomes.png)

| Recorded experiment | Result | Correct interpretation |
|---|---|---|
| Basketball fixed adaptation, six 60-second trials | 6/6 sustained survival | Verified local nominal-condition free-ball balance. |
| Same candidate, three forward-command trials | 0/3 rolling passes | Movement is not reliable commanded rolling. |
| Basketball mixed adaptation | 6/6 survival, 0/3 rolling passes | A separate candidate, not further steps of the fixed candidate. |
| Bridge final pilot | 0 accepted crossings | Training/export work; the requested physical task remains unsolved. |
| Seven-case API/Studio checks | Seven scenarios reachable and rendered | Product integration, not seven skill certificates. |

The direct basketball report uses seeds 101, 202 and 303. The later Studio evaluation uses 101, 102 and 103. Both contain zero/forward command pairs, but they are separate evidence sets. Do not relabel one report's seeds using another report's screenshot. The minimum direct-trial foot-ball support coverage was about 99.93%, and maximum tilt about 4.74 degrees; these describe that report's candidate and conditions, not an untested distribution.

**Historical source boundary.** The retained direct basketball report predates the current source snapshot: its evaluator/environment hashes begin `fcfa65`/`2eda2a`, while this edition's supplied evaluator/environment begin `bbe735`/`6bff54`. The old report is preserved, not presented as a fresh execution of these exact source bytes. The new 10-second reward diagnostic binds the current environment hash, but is not a replacement for the six-trial 60-second acceptance battery. To test the current snapshot, run the documented evaluator into a new directory and retain its new hashes. Protocol reproduction and byte-identical historical reconstruction are different claims.

The direct fixed run's forward tracking mean absolute error is about 0.138 m/s against a 0.15 m/s command. The mixed run's is about 0.160 m/s. These are poor command-following results even though balance is good. A small fixed-versus-source change near 0.140 m/s does not establish a robust improvement.

For bridge, 1.174916 m maximum progress is measured relative to the launch position. It is **not** “1.174916 m of a 1.10 m bridge, therefore done.” The duck must leave its platform, make valid plank contact, reach the destination with both feet, avoid forbidden support and survive the required full episode. The final evaluation has three vector lanes and 3,000 total transitions, with resets producing multiple attempts and trailing fragments. Those fragments are not extra successful 20-second trials.

Three or six successes in nominal tests do not establish broad reliability. The current tests do not exhaust friction, mass, cable stiffness, command directions, pushes, noisy sensors, actuator temperature, real manufacturing variation or hardware conditions. General steering coverage is explicitly absent for lateral/yaw commands. Report the tested set, not an invented population success percentage.

## Visual review: inspect consecutive frames

![Measured zero-command basketball rollout, seed 101. This is evidence of sustained balance in the recorded trial, not commanded locomotion.](assets/evidence/observed-basketball-zero-command-contact-sheet.png)

Check more than whether the duck is above the orange ball. Is its trunk upright? Are feet actually supporting it? Is the ground or another body part supporting the robot? Are the textures rotating consistently with physical motion? Did the clock continue without a reset? Was the camera following so tightly that drift became invisible?

![Measured forward-command attempt. The duck balances and the ball moves, but the directional rolling gate fails.](assets/evidence/observed-basketball-forward-command-contact-sheet.png)

A contact sheet samples a video. It can miss fast foot chatter or an event between frames. When uncertain, inspect the MP4 at the original frame rate or render at the 50 Hz control rate. Use the numerical contact/termination record alongside images. “Looks successful” and “metrics look successful” should agree before accepting a skill.

![Bridge diagnostic sequence from the final pilot. The 20-second video contains resets; it is not one uninterrupted successful crossing.](assets/evidence/observed-bridge-rollout-contact-sheet.png)

The retained bridge video spans 20 seconds and contains two resets. A later attempt reaches partway onto the plank. A reset returns the robot to a fresh spawn; it cannot be hidden and counted as continuous progress. The book includes this failed sequence because seeing where a policy loses contact is useful for curriculum design.

Videos remain in the workspace rather than being embedded in the PDF:

- `docs/basketball-showcase/evidence/render/cmd-p0_000-p0_000-p0_000/seed-101/rollout.mp4`
- `docs/basketball-showcase/evidence/render/cmd-p0_150-p0_000-p0_000/seed-101/rollout.mp4`
- `rlx/runs/studio/bridge/bridge-studio-02/render/ep0.mp4`

## Run both scenes in Duck Viewer

![Observed seven-case Studio screen. Card availability is separate from learned-skill acceptance.](assets/evidence/observed-seven-cases.png)

The seven cases are **Dance, Swing, Running, Stilts, Backflip, Basketball, Bridge**. The basketball card must say balance-only when rolling is unaccepted. The bridge card must not turn a pilot video into an accepted-crossing preview.

Start a new instance only if there is no appropriate existing service. Keep each server in its own terminal; these are long-running processes. Use available ports instead of stopping someone else's server.

```bash
export MICRODUCK_STUDIO_PYTHON_DIRECT="$PWD/rlx/.venv-microduck/bin/python"
cd duck-viewer
node node_modules/next/dist/bin/next dev --hostname 127.0.0.1 -p 63317
```

In a second terminal, from the workspace root:

```bash
cd microduck_local
MICRODUCK_ACTUATOR=bam .venv/bin/duck-lab --port 8788 \
  runs/first-gait ../microduck/policies/alpha_walking.onnx
```

This second command requires the local harness's own virtual environment and the named baseline paths. If `runs/first-gait` is not available on a clean checkout, omit that positional baseline and use the supplied walking ONNX only. Create the harness environment with `uv sync --python 3.12` inside `microduck_local` if necessary. Per-policy custom-scene settings select basketball or bridge; the default actuator environment variable is not a substitute for those settings.

Open `http://127.0.0.1:63317`, select the experiment and its exported policy, and check the live scene. Browser availability alone is not enough: a saved preview and a live physics scene are different surfaces.

![Observed live custom scenes: basketball at left and suspended bridge with platforms and gantries at right.](assets/evidence/observed-live-scenes.png)

Two earlier integration defects are instructive. First, the lab extracted group-2 visual geometry, so a correct physics object placed only in group 0 was invisible. Second, the CLI policy loader needed each policy's custom environment arguments; otherwise a named bridge policy ran on the wrong scene. The fix was correct scene routing and visual extraction, not adding decorative CSS objects. LSTM cache sharing also had to preserve private h/c state for every duck.

Use the repository's readiness and API checks with the services running:

```bash
bash restart-lab.sh --readiness-only
node scripts/verify-seven-cases.mjs
node scripts/verify-balance-studio-api.mjs
```

The API verifier invokes real evaluation/render operations and may create evidence. The optional `--smoke-train` variant performs real tiny training jobs, not a read-only check. Those integration checks are separate from the static book build. They may need browser/graphics access and sufficient time.

## A research plan for the unresolved skills

First ask whether any rollout contains the missing behavior. If not, changing a reward coefficient cannot teach a state the learner never visits. For bridge, investigate early-spawn practice, plank stabilization, step placement and transition off the launch platform. For basketball, separate balance retention, forward command tracking and lateral/yaw steering coverage.

Keep the nominal accepted gate frozen while changing one training condition. Log assistance on every attempt and evaluate with it off. Compare a transferred walking or source actor, a zero-action baseline, and the newly trained actor. A platform-standing baseline proves platform standing, not plank balance. A partial-spawn crossing is exploration evidence, not a full launch-to-destination solution.

Do not promise a number of additional steps sufficient to solve the task. Training time depends on exploration, dynamics, initialization and optimization, not only the requested total. A scientifically useful next run may fail while revealing the exact contact transition the curriculum must make learnable.

## Rebuild and inspect this book

```bash
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/prepare_assets.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/build_book.py
rlx/.venv-microduck/bin/python docs/basketball-bridge-book/tools/validate_book.py
```

The build uses existing dependencies; it does not install software or retrain either policy. Markdown is the copyable source edition. PDF adds wrapped code, page numbers, equations and a table of contents. `supporting-source.zip` and `source-manifest.json` bind the implementation snapshot. The validation report lists syntax checks, source/asset hashes, PDF page/text checks and any remaining warnings.

## Final practical examination

1. Explain why 61 observations do not mean 61 independent sensors.
2. Show where the exported normalizer is applied and why changing its denominator breaks parity.
3. Compute the transitions in one rollout and distinguish simulated time from wall time.
4. Explain why a timeout bootstraps value but stops the GAE trace across reset.
5. Identify one observed loss curve, one observed reward curve, and one schematic.
6. Report basketball balance and rolling separately, with seeds and horizon.
7. Explain why the bridge's best progress number does not prove a crossing.
8. Start both viewer scenes without changing another user's running policies.
9. State which assets prevent an empty-folder, offline reproduction.
10. Propose one bounded next experiment without altering the acceptance gate.

**Answer guide.** Observations include derived gravity, commands and past actions; normalization is part of the actor contract; transitions equal lanes times rollout length; timeout is not physical failure but reset must break trace propagation; plots require provenance; the measured balance passes while rolling fails; launch-relative progress includes approach; route per-policy scenes and preserve recurrent memory; supplied weights/meshes and dependencies remain prerequisites; change one curriculum or physical condition and keep deterministic full-horizon evaluation fixed.

# References and provenance

The implementation snapshot is the authority for what these examples actually execute. General algorithm and simulator references explain the ideas; they are not evidence of either trained skill.

1. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford and Oleg Klimov. **Proximal Policy Optimization Algorithms**. 2017. arXiv:1707.06347. Original PPO paper; clipped surrogate and repeated sample reuse. Official record checked for this edition: `https://arxiv.org/abs/1707.06347`.
2. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan and Pieter Abbeel. **High-Dimensional Continuous Control Using Generalized Advantage Estimation**. arXiv:1506.02438. The GAE reference; this book separately explains the repository's termination/truncation masks. `https://arxiv.org/abs/1506.02438`.
3. **MuJoCo XML Reference**, official documentation. Body, free joint, geom, material, spatial tendon, springlength and limit semantics. Consult documentation matching the installed MuJoCo version; this snapshot records MuJoCo 3.10.0. `https://mujoco.readthedocs.io/en/stable/XMLreference.html`.
4. `microduck_local/AGENTS.md`: the local training playbook, fixed observation contract, physics-only curricula, and export–evaluate–look discipline.
5. `microduck-playground/experiments/basketball/` and `microduck-playground/artifacts/basketball/`: user-supplied reference design and mature actor, including provenance/license notices.
6. `microduck-playground/experiments/swing/README.md`: suspended-support reference; a bridge is not a renamed swing task.
7. `docs/basketball-showcase/RESULTS.md` and `docs/bridge-showcase/RESULTS.md`: retained local results and limitations.
8. `assets/provenance.json`, `assets/SHA256SUMS`, `environment.json`, `source-manifest.json`: machine-readable image/data, runtime and implementation fingerprints.

No external paper's text is reproduced wholesale. Equations, explanations and code discussions are tied to the local implementation. Existing copied source retains its original content and accompanying applicable notices; this educational snapshot does not change third-party licensing.
