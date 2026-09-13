# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-09-12
- Primary product surface: one Microduck Studio application containing an eight-experiment catalog, the live simulation workspace, policy tools, teaching controls, PPO recipe configuration, training telemetry, deterministic evaluation, render review, ONNX export, and deployment readiness.
- Evidence reviewed: `rlx/mylab.md`, `rlx/docs/ppo_microduck_dance_notebook.ipynb`, `rlx/docs/ppo-microduck-dance-guide.md`, `rlx/examples/ppo_microduck_dance.py`, `rlx/rlx/models/microduck.py`, `duck-viewer/README.md`, `duck-viewer/AGENTS.md`, and `/tmp/microduck-studio-design/index.html`.

## Brand
- Personality: precise, curious, calm, technically credible, and approachable.
- Trust signals: explicit local/offline status, source artifact names, fixed contract dimensions, deterministic evaluation labels, safety gates, and honest distinctions between smoke checks and learned behavior.
- Avoid: marketing-first layouts, decorative gradients, fake hardware status, opaque scores, unqualified claims that a policy is ready, and playful copy that obscures risk.

## Product goals
- Goals: provide one coherent local workflow for Dance, Swing, Running, Stilts, Backflip, Basketball, Suspended bridge, and Swimming-duck drawing; configuring a macOS recipe; running training; inspecting telemetry; evaluating deterministically; reviewing rendered motion; and exporting the appropriate ONNX interface without implying hardware readiness.
- Non-goals: maintaining a separate viewer product, silently deploying to physical hardware, claiming sim-to-real readiness from local training, or hiding command and artifact provenance.
- Success signals: a user can identify the current stage and next action, recover the exact command and artifacts, distinguish live and offline data, and understand why deployment is enabled or blocked.

## Personas and jobs
- Primary personas: robotics learners, reward-design experimenters, and engineers training and evaluating Microduck behaviors locally on Apple Silicon.
- User jobs: start from a reliable recipe, tune a small set of meaningful controls, monitor progress, compare evaluation evidence, inspect motion, export a compatible policy, and perform a deliberate deployment handoff.
- Key contexts of use: desktop development on Apple Silicon, responsive monitoring on a smaller screen, and local operation alongside `duck-lab`.

## Information architecture
- Primary navigation: Overview, Projects, Experiments, Policies, Evaluations, Robot fleet, Deployments, Settings.
- Core routes/screens: `/` is the complete Studio. Legacy `/viewer` requests return to `/` rather than opening a second application.
- Content hierarchy: compact experiment catalog with inline reference previews, selected-experiment workflow stage, integrated live simulation workspace and training status, policy/teach/animation/record tools, evaluation evidence, artifacts and deployment gates, then supporting system details.

## Design principles
- Show truth before confidence: every metric states whether it is live, persisted, or sample data.
- Make the next safe action obvious: one primary action per workflow stage, with blocked actions explaining the missing gate.
- Preserve expert depth without burdening beginners: Microduck-aligned defaults remain prominent; advanced PPO and environment controls are progressively disclosed.
- Treat visual review as evidence: render artifacts and human review are required before deployment readiness.
- Distinguish reference evidence from the current run: published Playground videos and metrics are labeled `REFERENCE`; newly rendered Studio output is labeled `LATEST RUN`.
- Unify lifecycle, not task semantics: all recipes use Train, Evaluate, Render, and Artifacts, while each keeps its own physics, rewards, metrics, and morphology controls.
- Tradeoffs: the Studio keeps a large operational scene while surrounding it with dense workflow controls; the simulation can expand inside the same application but does not navigate to a separate product.

## Visual language
- Color: near-black neutral canvas; cool graphite panels; mint for active/safe states; amber for review/warning; red only for destructive or failed states; restrained violet for assistant guidance.
- Typography: compact sans-serif UI with a slightly geometric display face when available; tabular numerals for metrics; no viewport-scaled type.
- Spacing/layout rhythm: 4 px base, dense 8/12/16/24/32 px increments, strong alignment, and full-width workflow bands.
- Shape/radius/elevation: 6-8 px control and panel radii, one-pixel borders, minimal shadows, no nested decorative cards.
- Motion: short state transitions, progress interpolation, and optional live-scene motion; respect reduced-motion preferences.
- Imagery/iconography: the real WebGL Microduck scene is the primary visual asset; use familiar compact symbols for controls and status.

## Components
- Existing components to reuse: the Three.js scene engine inside `Viewer`, `Hud`, `PolicyPanel`, `TeachPanel`, `AnimPanel`, `RecordPanel`, `LabClient`, policy and training payload types, and the current duck-lab HTTP/WebSocket contracts.
- New/changed components: `Viewer` remains a reusable in-process workspace with Studio/fullscreen layouts; `Studio` owns a four-recipe catalog, selected-experiment workflow shell, training telemetry, evaluation matrix, artifact list, and deployment gate around that workspace. Each recipe card includes an information icon that opens recipe-specific input, Train, Evaluate, Render, PPO reward-function, concept, and output guidance in an in-page dialog. Advanced settings expose recipe-specific absolute reward coefficients with a reset-to-default action. Reward history includes a sample-count indicator that opens the exact available step/reward samples in a copyable table. Start RLX provides immediate button and in-panel status feedback, reports launch failures beside the control, and identifies a different active local run because the Studio supports one RLX process at a time. The render view includes a compact camera navigator with press-and-hold orbit, truck, elevation, zoom, home-view reset, and a visually distinct simulation restart control.
- Swing recovery: selecting Swing opens the preserved `swing-studio-01` baseline and its render. The training surface provides executable Discovery and Consolidation presets. Discovery creates `swing-curriculum-01` for 250,000 assisted steps; Consolidation resumes that checkpoint for 500,000 additional steps only after the 10° mean or 20° best discovery gate passes. Evaluation and rendering always start the swing at rest.
- Render review: the latest run MP4 plays inline inside the Evaluation gate with native controls and byte-range seeking, with explicit full-video and contact-sheet actions.
- Variants and states: live/offline, idle/running/succeeded/failed/cancelled, pass/review/blocked, expanded/collapsed advanced settings, and desktop/mobile navigation.
- Token/component ownership: Studio tokens live in `duck-viewer/components/Studio.module.css`; existing viewer components retain their current inline visual system.

## Accessibility
- Target standard: WCAG 2.2 AA for the Studio surface.
- Keyboard/focus behavior: visible focus rings, native buttons/inputs/details, logical document order, and a native recipe-guidance dialog that traps focus while open and closes with its close control, backdrop click, or Escape.
- Contrast/readability: muted text remains readable on dark panels; state is never communicated by color alone.
- Screen-reader semantics: landmark navigation, labeled progress, status regions, tables with headers, and descriptive control labels.
- Reduced motion and sensory considerations: disable decorative transitions and animations when requested.

## Responsive behavior
- Supported breakpoints/devices: wide desktop, compact desktop/tablet, and 390 px mobile.
- Layout adaptations: collapse the sidebar below tablet width, stack simulation/training and supporting panels, keep workflow steps horizontally legible, and prevent horizontal overflow.
- Touch/hover differences: all actions remain available without hover; touch targets are at least 36 px where layout permits.

## Interaction states
- Loading: show preflight and connection checks without inventing telemetry.
- Empty: explain which artifact or action is needed next.
- Error: preserve logs and display the failed stage plus a concrete recovery action.
- Success: name the produced artifact and unlock only the next valid stage.
- Disabled: state the missing prerequisite in adjacent copy or the button title.
- Offline/slow network, if applicable: switch to labeled reference data and command preview; never imply that the lab or trainer is running.

## Content voice
- Tone: direct, instructional, and evidence-led.
- Terminology: use “Microduck PPO Recipe,” “checkpoint,” “deterministic evaluation,” “render review,” “ONNX policy,” and “deployment rehearsal.”
- Microcopy rules: avoid “AI magic,” avoid calling a smoke policy trained, and qualify local output as prototyping evidence rather than hardware readiness.

## Implementation constraints
- Framework/styling system: Next.js 16, React 19, TypeScript, and CSS Modules; no new dependencies.
- Design-token constraints: extend CSS custom properties within the Studio module; do not replace the existing viewer styling system.
- Performance constraints: keep the existing lightweight Three.js rendering path, no shadow maps or GPU text, and mount exactly one viewer canvas and one Studio-visible lab connection path.
- Compatibility constraints: preserve `?lab=host:port`, current WebSocket behavior, localStorage behavior, camera/selection keyboard controls, policy assignment, teaching, animation, recording, and HUD behavior inside Studio.
- Training compatibility: the original Studio recipes retain their existing macOS trainers; drawing uses CPU Stable Baselines 3 PPO with a BC/DAgger warm-start. The UI must not require or imply CUDA. Fixed drawing controls must reflect the standalone trainer rather than expose ignored options.
- Experiment contracts: the original seven retain their 61-observation / 14-action / 50 Hz interfaces, including Basketball's additional recurrent state. The eighth case has two explicitly selected tools: `microduck-drawing-v1` pencil (83/15) and `microduck-brush-v2` color brush (93/15). Both require exact versioned ONNX metadata and are incompatible with stock hardware. Stilt artifacts remain morphology-specific; Swing keeps its mechanism-specific contract.
- Drawing evidence: only measured tool-tip contacts produce visible paint or graphite, never target artwork. Tools are preloaded and free, with an added simulated mouth actuator. Brush color changes require physical palette contact; its hair bundle uses passive normal compliance. The stroke/color planner is authored, not learned visual planning. Teacher feasibility, BC baseline and PPO evaluation are separate evidence categories. Unsupported or failed learned rollouts remain diagnostic even when another controller succeeds. No deployment handoff is enabled for drawing.
- Reward customization contract: editable reward coefficients are finite and non-negative because penalty measurements already carry their negative sign. The selected weights must be used consistently for Train, Evaluate, and Render, recorded in checkpoint metadata, and treated as a new experiment that requires retraining.
- Swing optimizer contract: Studio defaults mirror the repository's stable Swing profile (`1e-4` learning rate, `0.1` PPO clip, three update epochs, `0.002` entropy coefficient, `0.995` gamma, and `1.0` gradient norm) instead of inheriting generic recipe settings.
- Swing acceptance contract: finite rollout checks are necessary but not sufficient. Discovery may continue at 10° mean or 20° best span; consolidation may continue toward 1M at 30° mean or 60° best span; deployment remains blocked until approximately 163° median span plus valid geometry and human video review.
- Test/screenshot expectations: `npm run lint`, `npm run build`, desktop screenshot at 1536 x 1100, mobile screenshot at 390 x 900, overflow checks, and interaction checks for stage changes and command actions.

## Open questions
- [ ] Define the production robot transport and authentication contract before enabling direct hardware upload.
- [ ] Decide whether long-running RLX jobs should move from in-memory Next.js process state to a durable local daemon.
- [ ] Define quantitative dance-quality thresholds; current numeric evaluation proves rollout validity, while motion quality still requires visual review.

## Robot arm extension — tabletop simulation implemented, 2026-09-12
- Canonical first-stage specification: `docs/robot-arm-design/README.md`; classroom experiments, policy contracts, metrics and verification: `docs/robot-arm-design/EXPERIMENTS.md`; source ledger: `docs/robot-arm-design/SOURCES.md`.
- `docs/robot-arm-design/index.html` is an offline Chinese design prototype, not an implemented Studio route, MuJoCo rollout, training service, or hardware controller. Existing eight experiments remain unchanged.
- The implemented `/arm` route uses the real loopback `arm_lab` backend on port 8812, separate from existing duck-lab services. Actual scope, commands and results are in `docs/robot-arm-design/IMPLEMENTATION.md` and `RESULTS.md`; the original design goals are not automatically fulfilled by the reduced simulation curriculum.
- Proposed progression: fixed tabletop single arm → fixed dual arms → Microduck-coordinated docked manipulation → separately reviewed mobile integration. The initial design does not attach a load to the biped or assume the original walking policy remains valid with new mass.
- Reuse control conventions, Dynamixel-compatible actuator family, 50 Hz timing, evidence-bound lifecycle and Studio components; preserve the stock 61-observation / 14-action policy. Actual onboard alpha has 15 physical joints, with the mouth deliberately excluded from the policy (`microduck/duck-control/src/model.rs`, `obs.rs`).
- Proposed arm services and protocols are independent, versioned extensions: one owner per motor bus, independent arm power, Microduck-hosted manipulation coordination, fail-closed hardware gates. Neither raw browser joint commands nor direct attachment of unidentified kit servos to the stock bus is allowed.
- Navigation adds an Arm Lab family alongside the existing catalog, with six cases, BOM, live scene, evaluation, video and integration readiness. Training is CLI-driven, not a pretend browser training action. Design/teacher/BC/PPO/simulation/hardware evidence remain distinct. Missing metrics display “未测”, not zeros or invented learning curves.
- The learned controller is a bounded PPO residual over authored IK/FSM, not end-to-end planning. Single/dual contracts are 66/6 and 116/12; handover uses 5 g and co-carry uses 17 g total. The hardware coordinator and serial factory unconditionally deny hardware in this phase; simulated HTTP and Unix coordination are explicitly separate paths.
- Initial mechanism is five pose joints plus one coupled gripper actuator, not arbitrary six-DOF end-effector pose control. Reach and payload are design targets requiring mechanical, electrical and thermal validation.
- Accessibility and visual language inherit Studio: keyboard operation, descriptive SVGs, labeled state badges, responsive layout and no color-only verdicts. Hardware execution stays locked in this phase.
- Arm classroom navigation reuses `/arm` with a recorded-video workspace as the default and a separate live-simulator workspace. `/arm#videos` is the Studio shortcut. The video library is served locally by Next.js independently of the live simulator, so offline hardware/backend status never hides saved evidence.
- Include every arm MP4, including current episodes, regression replays, historical failures, and compilations. Deduplicate identical bytes without discarding source aliases. Filter by case, provenance, outcome, and recording type; label those dimensions independently. A successful historical episode is not current validation, an unverified file is not a pass, and a compilation is not a new experiment.
- Use one native video player with keyboard-accessible playlist buttons, seek/download controls, source receipts, and explicit load/error/empty states. No simultaneous auto-playing thumbnails or additional WebGL canvases in the video workspace. Preserve six simulation cases and the existing eight duck skills.
- Studio and Arm Lab share an explicit English / 中文 interface toggle. Default to English, persist the preference locally across navigation/reloads, update the document language for accessibility, and keep switching functional when storage is unavailable. Translate rendered interface copy rather than mutating the DOM or protocol data. Switching must not reset experiments, alter video selection/filters/playback, issue simulator commands, or relabel raw logs and evidence artifacts as translated evidence.
