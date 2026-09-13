export type ExperimentId =
  | "dance"
  | "swing"
  | "running"
  | "stilts"
  | "backflip"
  | "basketball"
  | "bridge"
  | "drawing";

export type DrawingTool = "pencil" | "brush";

export interface DrawingToolDefinition {
  id: DrawingTool;
  label: string;
  contractVersion: "microduck-drawing-v1" | "microduck-brush-v2";
  observations: 83 | 93;
  actions: 15;
  fullTimesteps: number;
  smokeTimesteps: number;
  fullNumSteps: number;
  fullNumMinibatches: number;
  batchSize: number;
  evalEpisodes: number;
  evaluationConfigs: number;
  maxEpisodeSeconds: number;
  initialStd: number;
  explorationLabel: string;
  trainerLabel: string;
  actorLabel: string;
  ppo: {
    learningRate: number;
    gamma: number;
    clipCoefficient: number;
    updateEpochs: number;
    entropyCoefficient: number;
    maxGradNorm: number;
  };
}

export const DRAWING_TOOLS: Record<DrawingTool, DrawingToolDefinition> = {
  pencil: {
    id: "pencil",
    label: "Pencil",
    contractVersion: "microduck-drawing-v1",
    observations: 83,
    actions: 15,
    fullTimesteps: 32_768,
    smokeTimesteps: 768,
    fullNumSteps: 256,
    fullNumMinibatches: 1,
    batchSize: 256,
    evalEpisodes: 8,
    evaluationConfigs: 1,
    maxEpisodeSeconds: 32,
    initialStd: 0.02,
    explorationLabel: "Uniform initial std 0.02",
    trainerLabel: "SB3 PPO · PyTorch CPU",
    actorLabel: "Actor / critic: 256 → 256 Tanh",
    ppo: {
      learningRate: 0.00003,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 5,
      entropyCoefficient: 0,
      maxGradNorm: 0.5,
    },
  },
  brush: {
    id: "brush",
    label: "Brush",
    contractVersion: "microduck-brush-v2",
    observations: 93,
    actions: 15,
    fullTimesteps: 49_152,
    smokeTimesteps: 6_144,
    fullNumSteps: 6_144,
    fullNumMinibatches: 12,
    batchSize: 512,
    evalEpisodes: 4,
    evaluationConfigs: 10,
    maxEpisodeSeconds: 120,
    initialStd: 0.0003,
    explorationLabel: "Head std 0.0003 · legs 0.0001 · jaw 0.0002",
    trainerLabel: "SB3 PPO · PyTorch CPU",
    actorLabel: "Learned-feedback actor · per-joint exploration",
    ppo: {
      learningRate: 0.0000001,
      gamma: 0.995,
      clipCoefficient: 0.05,
      updateEpochs: 2,
      entropyCoefficient: 0,
      maxGradNorm: 0.3,
    },
  },
};

export function getDrawingTool(tool: DrawingTool = "brush"): DrawingToolDefinition {
  return DRAWING_TOOLS[tool];
}

export function drawingToolFromContract(value: unknown): DrawingTool | null {
  if (value === DRAWING_TOOLS.pencil.contractVersion) return "pencil";
  if (value === DRAWING_TOOLS.brush.contractVersion) return "brush";
  return null;
}

export interface RewardTermDefinition {
  key: string;
  label: string;
  description: string;
  defaultWeight: number;
  penalty?: boolean;
  editable?: boolean;
}

export interface RewardDefinition {
  summary: string;
  formula: string;
  terms: RewardTermDefinition[];
}

export interface ExperimentDefinition {
  id: ExperimentId;
  title: string;
  shortTitle: string;
  goal: string;
  description: string;
  task: string;
  result: string;
  metricLabel: string;
  metricKey: string;
  metricSuffix: string;
  preview: string;
  poster: string;
  video: string;
  evidence: string[];
  artifactStem: string;
  artifactFiles?: {
    checkpoint: string;
    metadata: string;
    onnx: string;
    evaluation: string;
    renderSheet?: string;
    renderVideo?: string;
  };
  commandAdapter?: "studio" | "basketball" | "drawing";
  defaultRunName: string;
  fullTimesteps: number;
  fullEnvs: number;
  fullNumSteps: number;
  fullNumMinibatches: number;
  maxEpisodeSeconds: number;
  seed: number;
  initialStd: number;
  normalizeRewards: boolean;
  fullEnvironment: {
    domainRand: boolean;
    obsNoise: boolean;
    actionDelay: boolean;
    randomYaw: boolean;
  };
  ppo: {
    learningRate: number;
    gamma: number;
    clipCoefficient: number;
    updateEpochs: number;
    entropyCoefficient: number;
    maxGradNorm: number;
  };
  guideAnchor: string;
  readiness: string;
  policyContract: string;
  guidance: {
    requiredInput: string;
    steps: string[];
    distinctions: string[];
    output: string;
  };
  reward: RewardDefinition;
  controls?: {
    stiltHeightCm?: number;
    stiltBlend?: number;
    stiltMassKg?: number;
  };
}

const REWARDS: Record<ExperimentId, RewardDefinition> = {
  dance: {
    summary:
      "PPO is paid for matching the authored pose and timing while remaining upright, travelling, and avoiding unsafe contacts or wasteful actuator use.",
    formula:
      "reward = pose tracking + rotation tracking + balance + travel - slip - spin - impacts - joint limits - motor effort",
    terms: [
      { key: "pose_match", label: "Pose match", defaultWeight: 4, description: "Match the clip's joint pose at the current frame." },
      { key: "rotation_match", label: "Rotation timing", defaultWeight: 4, description: "Match the clip's body rotation at the current time." },
      { key: "stick_it", label: "Stick the landing", defaultWeight: 5, description: "Finish a one-shot trick standing on both feet." },
      { key: "on_feet", label: "Stay on feet", defaultWeight: 5, description: "Remain upright, tall, and in foot contact during a looping dance." },
      { key: "travel", label: "Forward travel", defaultWeight: 3, description: "Move forward instead of performing the loop in place." },
      { key: "no_slip", label: "Foot slip", defaultWeight: 0.5, penalty: true, description: "Charge planted feet that skate across the floor." },
      { key: "no_spin", label: "Unwanted spin", defaultWeight: 0.3, penalty: true, description: "Charge yaw motion when the dance should travel straight." },
      { key: "gentle_head", label: "Head impacts", defaultWeight: 1, penalty: true, description: "Charge head-floor impacts by severity." },
      { key: "soft_landings", label: "Hard landings", defaultWeight: 0.75, penalty: true, description: "Charge excessive landing impact." },
      { key: "no_limit_parking", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge joints held against their end stops." },
      { key: "save_energy", label: "Motor effort", defaultWeight: 0.5, penalty: true, description: "Charge excessive actuator torque." },
    ],
  },
  swing: {
    summary:
      "PPO is paid for expanding the bidirectional swing frontier and gaining controlled height and energy, while geometry and actuator penalties keep the motion planar and physically valid.",
    formula:
      "reward = frontier progress + height + late height + energy - lateral motion - misalignment - string violations - unsafe actions",
    terms: [
      { key: "swing_peak_progress", label: "New arc frontier", defaultWeight: 224, description: "Pay only for reaching a new peak on either side of the swing." },
      { key: "swing_height", label: "Swing height", defaultWeight: 8, description: "Reward height throughout the episode." },
      { key: "swing_late_height", label: "Late height", defaultWeight: 24, description: "Increase the value of height later in the attempt." },
      { key: "swing_energy", label: "Pendulum energy", defaultWeight: 0.5, description: "Reward useful height and angular speed." },
      { key: "swing_lateral_penalty", label: "Sideways offset", defaultWeight: 3, penalty: true, description: "Charge lateral displacement from the swing plane." },
      { key: "swing_lateral_barrier_penalty", label: "Lateral barrier", defaultWeight: 8, penalty: true, description: "Strongly charge motion beyond the safe lateral band." },
      { key: "swing_lateral_velocity_penalty", label: "Sideways velocity", defaultWeight: 3, penalty: true, description: "Charge fast sideways movement." },
      { key: "swing_out_of_plane_penalty", label: "Out-of-plane rotation", defaultWeight: 1, penalty: true, description: "Charge roll and yaw angular velocity." },
      { key: "swing_alignment_penalty", label: "Attachment alignment", defaultWeight: 18, penalty: true, description: "Keep the two attachment axes aligned." },
      { key: "swing_alignment_barrier_penalty", label: "Alignment barrier", defaultWeight: 4, penalty: true, description: "Strongly charge alignment beyond the valid range." },
      { key: "string_slack_penalty", label: "String slack", defaultWeight: 8, penalty: true, description: "Charge slack or unequal string lengths." },
      { key: "string_extension_penalty", label: "String extension", defaultWeight: 12, penalty: true, description: "Charge strings stretched beyond their intended length." },
      { key: "invalid_episode_penalty", label: "Invalid geometry", defaultWeight: 10, penalty: true, description: "Charge an episode after persistent mechanism invalidity." },
      { key: "action_rate_penalty", label: "Action changes", defaultWeight: 0.03, penalty: true, description: "Charge abrupt changes in policy actions." },
      { key: "joint_torque_penalty", label: "Joint torque", defaultWeight: 0.001, penalty: true, description: "Charge excessive actuator force." },
      { key: "joint_limit_penalty", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge motion beyond joint limits." },
    ],
  },
  running: {
    summary:
      "PPO follows a forward-speed command, earns controlled flight and posture rewards, and pays for foot, actuator, and body-motion errors.",
    formula:
      "reward = speed tracking + turn tracking + air time + posture - clearance error - slip - jerk - joint limits - body roll",
    terms: [
      { key: "keep_pace", label: "Match speed", defaultWeight: 4, description: "Match the commanded body-frame velocity." },
      { key: "track_turn", label: "Match turn rate", defaultWeight: 2, description: "Match the commanded yaw rate." },
      { key: "air_time", label: "Running flight", defaultWeight: 3, description: "Reward feet spending a useful duration in the air." },
      { key: "flight", label: "Both-feet flight", defaultWeight: 0, description: "Optional bounded reward for both feet off the ground, gated by upright posture and command-directed body speed; disabled by default." },
      { key: "yaw_tracking", label: "Precise yaw tracking", defaultWeight: 0, description: "Optional yaw-only tracking reward independent of roll and pitch, using observed gyro and commanded turn rate; disabled by default." },
      { key: "stay_upright", label: "Stay upright", defaultWeight: 2, description: "Keep the body upright while permitting running lean." },
      { key: "pose", label: "Running pose", defaultWeight: 1, description: "Track a speed-appropriate leg posture." },
      { key: "head_up", label: "Head posture", defaultWeight: 3.5, description: "Keep the head aligned with its command." },
      { key: "foot_clearance", label: "Foot clearance", defaultWeight: 2, penalty: true, description: "Charge a moving foot at the wrong height." },
      { key: "plant_the_foot", label: "Planted-foot slip", defaultWeight: 0.1, penalty: true, description: "Charge a planted foot that skids." },
      { key: "smooth_moves", label: "Action smoothness", defaultWeight: 1, penalty: true, description: "Charge jerky actions as the gait curriculum advances." },
      { key: "no_limit_parking", label: "Joint limits", defaultWeight: 1, penalty: true, description: "Charge joints held against their end stops." },
      { key: "calm_roll", label: "Body roll rate", defaultWeight: 0.025, penalty: true, description: "Charge excessive trunk roll and pitch velocity." },
    ],
  },
  stilts: {
    summary:
      "PPO follows walking commands while prioritizing upright balance on the selected stilts, controlled foot flight, head posture, and smooth body motion.",
    formula:
      "reward = linear tracking + turn tracking + upright balance + foot air time + head pose + optional leg pose - action changes - body angular velocity",
    terms: [
      { key: "track_lin_vel", label: "Match walking speed", defaultWeight: 2.5, description: "Match the requested linear velocity." },
      { key: "track_ang_vel", label: "Match turn rate", defaultWeight: 1, description: "Match the requested yaw rate." },
      { key: "yaw_tracking", label: "Precise yaw tracking", defaultWeight: 0, description: "Optional yaw-only tracking reward independent of roll and pitch, using observed gyro and commanded turn rate; disabled by default." },
      { key: "upright", label: "Upright balance", defaultWeight: 3, description: "Prioritize a level trunk on the raised contacts." },
      { key: "pose", label: "Leg pose", defaultWeight: 0, description: "Optional pull toward the default leg pose; disabled by default." },
      { key: "head_pose", label: "Head pose", defaultWeight: 0.25, description: "Track the requested head posture without dominating balance." },
      { key: "feet_air_time", label: "Foot air time", defaultWeight: 2, description: "Reward controlled swing phases while walking." },
      { key: "action_rate_penalty", label: "Action changes", defaultWeight: 0.2, penalty: true, editable: false, description: "Apply 20% of the walking curriculum's dynamic action-rate penalty." },
      { key: "ang_vel_xy_penalty", label: "Body angular velocity", defaultWeight: 0.05, penalty: true, description: "Charge excessive roll and pitch velocity." },
    ],
  },
  backflip: {
    summary:
      "During PPO landing only, dense observable terms reward upright pose and settling while bounded penalties charge joint speed, action changes, and action size. All terms are zero during assisted launch and pretrained stand-policy handoff.",
    formula:
      "reward = 3 upright + 2 landing pose + settle - 0.05 joint speed - 0.02 action rate - 0.01 action size",
    terms: [
      { key: "landing_upright", label: "Landing upright", defaultWeight: 3, description: "Reward normalized trunk orientation during PPO landing: (1 - gravity z) / 2, bounded to [0, 1]." },
      { key: "landing_pose", label: "Landing pose", defaultWeight: 2, description: "Reward upright posture times the per-joint average of broad 0.6 rad and precise 0.15 rad squared-exponential pose kernels, bounded to [0, 1]." },
      { key: "landing_settle", label: "Landing settle", defaultWeight: 1, description: "Reward upright pose settling with exp(-||gyro||² / 2), bounded to [0, 1]." },
      { key: "landing_joint_speed_penalty", label: "Landing joint speed", defaultWeight: 0.05, penalty: true, description: "Charge clipped mean joint velocity squared divided by 100; the term is bounded to [-1, 0]." },
      { key: "landing_action_rate_penalty", label: "Landing action changes", defaultWeight: 0.02, penalty: true, description: "Charge clipped mean squared action delta during PPO landing; the term is bounded to [-1, 0]." },
      { key: "landing_action_size_penalty", label: "Landing action size", defaultWeight: 0.01, penalty: true, description: "Charge clipped mean squared action divided by 16 during PPO landing; the term is bounded to [-1, 0]." },
    ],
  },
  basketball: {
    summary:
      "The standalone recurrent basketball trainer owns its fixed reward; Studio does not expose misleading reward sliders.",
    formula:
      "fixed recurrent basketball recipe: unassisted balance and command tracking",
    terms: [],
  },
  bridge: {
    summary:
      "The shared bridge recipe owns its fixed reward and staged suspended-bridge curriculum; Studio does not expose reward sliders.",
    formula:
      "fixed bridge recipe: curriculum training, then unassisted narrow suspended-bridge evaluation",
    terms: [],
  },
  drawing: {
    summary:
      "The selected independent drawing environment owns a fixed reward for contact-based ink, target-curve coverage, precision, tool control, and stable robot motion.",
    formula:
      "fixed drawing recipe: real pencil or brush contact produces ink while the lower jaw traces the authored swimming-duck target",
    terms: [],
  },
};

export const EXPERIMENTS: readonly ExperimentDefinition[] = [
  {
    id: "dance",
    title: "Dance imitation",
    shortTitle: "Dance",
    goal: "Learn a smooth, repeatable two-minute dance from an authored motion loop.",
    description:
      "A clip-conditioned PPO policy learns the 120 BPM choreography while preserving balance and the deployable observation contract.",
    task: "RLX imitate · dance-120bpm",
    result: "Local Studio baseline with deterministic rollout and visual review.",
    metricLabel: "Mean return",
    metricKey: "mean_return",
    metricSuffix: "",
    preview: "/experiments/dance/preview.mp4",
    poster: "/experiments/dance/poster.png",
    video: "/experiments/dance/preview.mp4",
    evidence: ["61 observations", "14 actions", "50 Hz", "2-minute render supported"],
    artifactStem: "dance",
    defaultRunName: "dance-studio",
    fullTimesteps: 1_000_000,
    fullEnvs: 16,
    fullNumSteps: 24,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 4,
    seed: 1,
    initialStd: Math.exp(-0.5),
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: true,
    },
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "dance-imitation",
    readiness: "Verified Studio workflow; simulation evidence is required per run.",
    policyContract: "Feedforward ONNX · obs[batch,61] → actions[batch,14]",
    guidance: {
      requiredInput:
        "One saved animation clip containing the poses and timing for the choreography. A velocity command is not required.",
      steps: [
        "Open Dance imitation.",
        "Use Animate to create or edit the dance clip.",
        "Save the clip.",
        "Select that clip for Dance training.",
        "Run Pipeline smoke first.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history.",
        "Run Evaluate.",
        "Run Render to generate the two-minute video.",
        "Review the video before accepting the policy.",
      ],
      distinctions: [
        "Clip means “perform these poses at this timing.”",
        "Command means “move at this speed or turn rate.” Dance imitation does not require a command.",
        "PPO is the learning algorithm. It learns from the selected recipe’s observations and reward.",
        "The clip supplies the choreography, while PPO learns balance, contacts, momentum, and actuator actions.",
      ],
      output:
        "A checkpoint, deterministic evaluation record, two-minute rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.dance,
  },
  {
    id: "swing",
    title: "Self-pumped swing",
    shortTitle: "Swing",
    goal: "Start still and discover coordinated body motion that pumps the swing.",
    description:
      "A mechanism-aware RLX task rewards bidirectional arc growth while guarding string tension, lateral motion, and attachment alignment.",
    task: "RLX swing · tension-only strings",
    result: "Published reference reaches a 173.20° strict full span.",
    metricLabel: "Peak-to-peak span",
    metricKey: "swing_span_deg",
    metricSuffix: "°",
    preview: "/experiments/swing/preview.gif",
    poster: "/experiments/swing/poster.jpg",
    video: "/experiments/swing/full.mp4",
    evidence: ["173.20° reference span", "71/100 strict seeds", "0.7 action scale"],
    artifactStem: "swing",
    defaultRunName: "swing-studio-01",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    fullNumSteps: 64,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 24,
    seed: 1,
    initialStd: 0.1,
    normalizeRewards: true,
    fullEnvironment: {
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: true,
    },
    ppo: {
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
    },
    guideAnchor: "self-pumped-swing",
    readiness: "Verified reference exists; each Studio run still requires bound evaluation and render evidence.",
    policyContract: "Feedforward ONNX · obs[batch,61] → actions[batch,14]",
    guidance: {
      requiredInput:
        "No clip and no velocity command. The swing mechanism, initial still state, and pumping rewards define what PPO must discover.",
      steps: [
        "Open Self-pumped swing.",
        "Confirm the swing mechanism recipe is selected; do not add a dance clip or movement command.",
        "Run Pipeline smoke first to verify the mechanism, observations, rewards, and artifact path.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history and the measured swing span as training progresses.",
        "Run Full evaluation: the default skill target is a symmetric 150° span (75° each side), all 24 seconds, valid geometry, and tensioned strings in every episode. Pipeline smoke does not assess skill.",
        "Run Render to generate the complete swing rollout.",
        "Review the video for a full controlled arc, tensioned strings, and low sideways drift.",
        "Accept the policy only after the evaluation and visual review both pass.",
      ],
      distinctions: [
        "Clip is not used because PPO must discover the pumping motion.",
        "Command is not used because the objective is mechanism motion, not a requested speed or turn rate.",
        "PPO learns from swing angle growth and safety rewards while choosing the 14 actuator actions.",
      ],
      output:
        "A swing checkpoint, strict-span evaluation, full rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.swing,
  },
  {
    id: "running",
    title: "Fast running",
    shortTitle: "Running",
    goal: "Discover a fast forward gait, then harden it against delay and disturbances.",
    description:
      "A speed curriculum prioritizes forward progress, controlled flight, low drift, and stable recovery under local MuJoCo simulation.",
    task: "RLX run · forward speed curriculum",
    result: "Published reference: 1.651 m/s nominal and 1.612 m/s under stress.",
    metricLabel: "Forward speed",
    metricKey: "forward_speed_m_s",
    metricSuffix: " m/s",
    preview: "/experiments/running/preview.gif",
    poster: "/experiments/running/poster.jpg",
    video: "/experiments/running/full.mp4",
    evidence: ["1.651 m/s nominal", "1.612 m/s stressed", "98.44% stressed survival"],
    artifactStem: "running",
    defaultRunName: "running-studio",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    fullNumSteps: 24,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 12,
    seed: 1,
    initialStd: Math.exp(-0.5),
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: true,
    },
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "fast-running",
    readiness: "Verified reference exists; each Studio run still requires bound evaluation and render evidence.",
    policyContract: "Feedforward ONNX · obs[batch,61] → actions[batch,14]",
    guidance: {
      requiredInput:
        "No animation clip. The environment supplies forward-speed commands and a speed curriculum for PPO to follow.",
      steps: [
        "Open Fast running.",
        "Confirm the forward-speed recipe is selected; no dance clip is needed.",
        "Run Pipeline smoke first to verify commands, observations, rewards, and artifact output.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history, forward speed, survival, and drift.",
        "Run Evaluate under nominal conditions and under backlash plus disturbance stress.",
        "Run Render to generate a sustained running rollout.",
        "Review the video for forward progress, controlled flight, low sideways drift, and recovery without falls.",
        "Accept the policy only after the speed, stability, and visual checks pass.",
      ],
      distinctions: [
        "Clip is not used because running is command-driven rather than pose-timed choreography.",
        "Command means the target forward speed sampled by the environment.",
        "PPO learns the gait, balance, contacts, momentum, and actuator actions that satisfy those commands.",
      ],
      output:
        "A running checkpoint, nominal and stressed evaluation metrics, rollout video, contact sheet, metadata, and deployable ONNX policy.",
    },
    reward: REWARDS.running,
  },
  {
    id: "stilts",
    title: "Stilt walking",
    shortTitle: "Stilts",
    goal: "Learn stable locomotion on a selected stilt height and support shape.",
    description:
      "A morphology-specific RLX recipe trains balance and forward motion on explicit stilt contact geometry, beginning with the 10 cm blend-0.50 setup.",
    task: "RLX stilts · fixed morphology",
    result: "Published 10 cm reference: 0.14055 m/s, 100% survival, 3.28° median max tilt.",
    metricLabel: "Forward speed",
    metricKey: "forward_speed_m_s",
    metricSuffix: " m/s",
    preview: "/experiments/stilts/preview.gif",
    poster: "/experiments/stilts/poster.jpg",
    video: "/experiments/stilts/full.mp4",
    evidence: ["10 cm shown", "0.50 support blend", "8 released heights"],
    artifactStem: "stilts",
    defaultRunName: "stilts-10cm-studio",
    fullTimesteps: 4_000_000,
    fullEnvs: 16,
    fullNumSteps: 24,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 12,
    seed: 1,
    initialStd: Math.exp(-0.5),
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: true,
    },
    ppo: {
      learningRate: 0.0003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 0.5,
    },
    guideAnchor: "stilt-walking",
    readiness: "Verified reference exists for specific morphologies only.",
    policyContract: "Feedforward ONNX · obs[batch,61] → actions[batch,14]",
    guidance: {
      requiredInput:
        "No animation clip. Choose the stilt height, support blend, and mass; the environment then supplies walking velocity commands.",
      steps: [
        "Open Stilt walking.",
        "Choose the stilt height, support blend, and mass you want to train.",
        "Run Pipeline smoke first to verify that exact morphology, commands, rewards, and artifact path.",
        "Switch to Full training and select Start RLX.",
        "Watch Reward history, forward speed, survival, tilt, and contact stability.",
        "Run Evaluate with the same stilt height, blend, and mass used for training.",
        "Run Render to generate the stilt-walking rollout.",
        "Review the video for stable contacts, controlled tilt, forward progress, and recovery without falls.",
        "Accept the policy only for the morphology recorded in its metadata.",
      ],
      distinctions: [
        "Clip is not used because stilt walking is command-driven locomotion.",
        "Command means the requested walking speed or turn rate supplied by the environment.",
        "PPO learns balance and actuator actions for one explicit stilt morphology; changing the hardware shape requires matching training and evaluation.",
      ],
      output:
        "A morphology-specific checkpoint, evaluation record, rollout video, contact sheet, metadata with stilt dimensions, and deployable ONNX policy.",
    },
    reward: REWARDS.stilts,
    controls: {
      stiltHeightCm: 10,
      stiltBlend: 0.5,
      stiltMassKg: 0.029,
    },
  },
  {
    id: "backflip",
    title: "Backflip showcase",
    shortTitle: "Backflip",
    goal: "spotter-assisted launch → PPO landing → pretrained stand-policy handoff, with full rotation and at least two seconds of stable standing.",
    description:
      "spotter-assisted launch → PPO landing → pretrained stand-policy handoff. The fixed, versioned assistance protocol is owned by Python; the PPO-updated landing policy is only one stage. Showcase acceptance does not establish PPO improvement.",
    task: "RLX backflip · spotter-assisted launch → PPO landing → pretrained stand-policy handoff",
    result: "spotter-assisted launch → PPO landing → pretrained stand-policy handoff; acceptance requires at least 5.8 rad of credited rotation before handoff, 10 consecutive stable PPO landing steps with no override or body support, and at least 2 seconds of stable standing.",
    metricLabel: "Rotation",
    metricKey: "rotation_rad",
    metricSuffix: " rad",
    preview: "/experiments/backflip/preview.gif",
    poster: "/experiments/backflip/poster.jpg",
    video: "/experiments/backflip/full.mp4",
    evidence: ["Version 2 assistance protocol", "Support-free policy landing", "≥2 s stand hold"],
    artifactStem: "backflip",
    defaultRunName: "backflip-studio",
    fullTimesteps: 400_000,
    fullEnvs: 16,
    fullNumSteps: 128,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 12,
    seed: 7,
    initialStd: 0.03,
    normalizeRewards: true,
    fullEnvironment: {
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
    },
    ppo: {
      learningRate: 0.000003,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 2,
      entropyCoefficient: 0,
      maxGradNorm: 0.5,
    },
    guideAnchor: "backflip-showcase",
    readiness: "Showcase verified; PPO improvement and a complete hardware controller are not established.",
    policyContract: "Feedforward landing ONNX · external launch and stand handoff required",
    guidance: {
      requiredInput:
        "No animation clip or velocity command. Python owns spotter-assisted launch → PPO landing → pretrained stand-policy handoff through protocol spotter-launch-landing-stand-v2.",
      steps: [
        "Open Backflip showcase.",
        "Keep spotter-assisted launch → PPO landing → pretrained stand-policy handoff unchanged.",
        "Run Pipeline smoke first to verify the four-step API, observations, rewards, and artifact path.",
        "Switch to Default full and select Start RLX.",
        "Fresh Full initializes the landing student by exactly transferring the pretrained stand actor and observation normalizer, then runs PPO; Pipeline smoke skips teacher initialization.",
        "Before Backflip PPO, Python collects 16 pretrained-teacher landing episodes of 600 steps (9,600 calibration steps), fixes the discounted-return RMS, and runs 10 critic-only calibration epochs. The actor remains unchanged, calibrated reward normalization is frozen, and PPO starts with a fresh optimizer.",
        "The launch releases at the 5.3 rad threshold with measured release rotation 5.376–5.380 rad and angular rate between 3.5 and 6.5 rad/s; external lift and pitch act only before release.",
        "Watch for at least 5.8 rad of credited rotation before handoff, followed by 10 consecutive stable PPO landing steps with no spotter, stand override, or body support.",
        "Run Full evaluation for spotter-assisted launch → PPO landing → pretrained stand-policy handoff, requiring the support-free prehandoff gate and at least 2 seconds of stable standing.",
        "Run Render to review the complete assisted launch, PPO-updated landing, and pretrained stand-policy handoff.",
        "Accept only when the evaluator reports skill_status passed and the source-bound render matches.",
      ],
      distinctions: [
        "spotter-assisted launch → PPO landing → pretrained stand-policy handoff.",
        "The complete controller always includes Python-owned launch assistance and the pretrained stand policy for the final hold.",
        "Backflip-only warm calibration prevents a reward-scale shock at PPO startup; it does not by itself establish positive PPO improvement.",
        "Measured reference backflip-e2e-20260908-v5 passes 24/24 showcase episodes across two evaluation sets, but its paired landing-only return regresses 12.63% versus initialization, with 7/8 standalone landings passing. Its extended learning audit FAILS; fast, precise PPO improvement is not established.",
        "The stand-policy phase cannot finish credited rotation; at least 5.8 rad must be complete before handoff.",
        "The exported backflip ONNX contains the PPO-updated landing policy only, not the assistance protocol or pretrained stand-policy handoff.",
      ],
      output:
        "A landing checkpoint and ONNX, backflip_assessment evaluation, rollout video, contact sheet, and metadata. The landing ONNX alone is not the complete backflip controller.",
    },
    reward: REWARDS.backflip,
  },
  {
    id: "basketball",
    title: "Basketball balancing",
    shortTitle: "Basketball",
    goal: "Balance on a free basketball, then learn command-directed rolling without assistance.",
    description:
      "A standalone recurrent policy carries LSTM state while balancing on the ball. The installed preview has verified 60-second balance, but steering has not passed.",
    task: "RLX basketball · recurrent balance and steering curriculum",
    result: "Balance-only preview available after 60-second unassisted evidence; full task remains failed until steering passes.",
    metricLabel: "Unassisted balance",
    metricKey: "balance_seconds",
    metricSuffix: " s",
    preview: "",
    poster: "",
    video: "",
    evidence: ["60 s balance verified", "Steering not passed", "Recurrent state required"],
    artifactStem: "basketball",
    artifactFiles: {
      checkpoint: "checkpoint.pt",
      metadata: "checkpoint.pt.json",
      onnx: "policy.onnx",
      evaluation: "eval.json",
      renderSheet: "contact-sheet.png",
      renderVideo: "rollout.mp4",
    },
    commandAdapter: "basketball",
    defaultRunName: "basketball-balance-01",
    fullTimesteps: 1_000_000,
    fullEnvs: 16,
    fullNumSteps: 64,
    fullNumMinibatches: 1,
    maxEpisodeSeconds: 60,
    seed: 42,
    initialStd: 0.03,
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: false,
      obsNoise: false,
      actionDelay: true,
      randomYaw: true,
    },
    ppo: {
      learningRate: 0.00002,
      gamma: 0.99,
      clipCoefficient: 0.2,
      updateEpochs: 5,
      entropyCoefficient: 0.01,
      maxGradNorm: 1,
    },
    guideAnchor: "basketball-balancing",
    readiness: "BALANCE ONLY · 60-second balance verified; steering not passed.",
    policyContract: "Recurrent ONNX · obs[batch,61] + h/c → actions[batch,14] + h/c",
    guidance: {
      requiredInput:
        "No clip. Use the released recurrent basketball bootstrap; hidden and cell state carry across control steps and reset only with the episode.",
      steps: [
        "Open Basketball balancing.",
        "Keep the recurrent policy contract and fixed reward recipe unchanged.",
        "Run Pipeline smoke to verify the standalone adapter, recurrent ONNX, and run-owned artifacts.",
        "Studio starts unassisted fine-tuning with hold 0 and curriculum disabled. The adapter's hold/curriculum ladder is an explicit CLI-only training option.",
        "Verify balance in place for 60 seconds before treating the run as a balance-only preview.",
        "Evaluate the current zero-command balance case and 0.15 forward-command tracking case. Lateral and yaw steering are not evaluated yet.",
        "Do not mark the full task passed until both balance and steering criteria explicitly pass.",
        "Render and review the complete unassisted rollout with recurrent state carried continuously.",
      ],
      distinctions: [
        "The installed basketball-balance-01 preview is balance-only evidence, not a steering success.",
        "The actor remains blind to ball state and uses the shared 61-observation robot and command input plus recurrent hidden state.",
        "A finite rollout, a high reward, or 60-second balance alone cannot establish command-directed steering.",
      ],
      output:
        "checkpoint.pt, policy.onnx, result.json, eval.json, and source-bound render artifacts. Full readiness remains gated until steering passes.",
    },
    reward: REWARDS.basketball,
  },
  {
    id: "bridge",
    title: "Suspended bridge",
    shortTitle: "Bridge",
    goal: "Cross a narrow suspended bridge without assistance, falls, or side exits.",
    description:
      "A shared Studio recipe has completed a bounded transferred-walker training pilot, then evaluates the final narrow bridge unassisted.",
    task: "RLX bridge · suspended narrow-bridge curriculum",
    result: "A 32,768-step training pilot exists; a complete unassisted crossing has not yet been verified.",
    metricLabel: "Bridge progress",
    metricKey: "bridge_progress_m",
    metricSuffix: " m",
    preview: "",
    poster: "",
    video: "",
    evidence: ["New suspended bridge", "Training curriculum available", "Not verified yet"],
    artifactStem: "bridge",
    commandAdapter: "studio",
    defaultRunName: "bridge-studio-01",
    fullTimesteps: 32_768,
    fullEnvs: 4,
    fullNumSteps: 128,
    fullNumMinibatches: 4,
    maxEpisodeSeconds: 20,
    seed: 7,
    initialStd: 0.03,
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
    },
    ppo: {
      learningRate: 0.00001,
      gamma: 0.99,
      clipCoefficient: 0.1,
      updateEpochs: 2,
      entropyCoefficient: 0,
      maxGradNorm: 0.5,
    },
    guideAnchor: "suspended-bridge",
    readiness: "TRAINING PILOT · 32,768-step transferred-walker run exists; crossing not yet verified.",
    policyContract: "Feedforward ONNX · obs[batch,61] → actions[batch,14]",
    guidance: {
      requiredInput:
        "No clip. The shared bridge registry supplies the suspended narrow bridge and automatic walking bootstrap.",
      steps: [
        "Open Suspended bridge.",
        "Run Pipeline smoke first to verify registry, observations, fixed rewards, and artifact paths.",
        "Use Default full for the bounded 32,768-step transferred-walker pilot and training-only bridge curriculum.",
        "Let the automatic walking bootstrap transfer the walking actor and observation normalizer before narrowing and suspending the bridge.",
        "Evaluate unassisted for at least 1,000 control steps over the full 20-second horizon.",
        "Require explicit bridge assessment success; pipeline validity and partial progress are not acceptance.",
        "Render the same unassisted final bridge and review the complete crossing before marking it verified.",
      ],
      distinctions: [
        "Bridge curriculum is training-only. Evaluation and rendering use the final suspended narrow bridge without assistance.",
        "Automatic bootstrap and checkpoint continuation are actor warm-starts with a fresh critic and optimizer, not exact optimizer resume.",
        "A training pilot exists, but the task has no verified crossing preview yet.",
        "Empty reward sliders are intentional because the registered bridge recipe owns fixed reward keys.",
      ],
      output:
        "A standard Studio checkpoint and ONNX, explicit bridge_assessment evaluation, and source-bound unassisted render artifacts.",
    },
    reward: REWARDS.bridge,
  },
  {
    id: "drawing",
    title: "Swimming-duck drawing",
    shortTitle: "Drawing",
    goal: "Use a pencil or four-color brush held by the active lower jaw to draw the swimming-duck target with real contact-based ink.",
    description:
      "An independent simulation-only Microduck task supports the microduck-drawing-v1 pencil and microduck-brush-v2 four-color brush contracts. Ink appears only from the selected tool's physical surface contact.",
    task: "RLX drawing · pencil or brush",
    result: "Diagnostic until drawing_assessment.passed is true and the evaluator confirms the rollout is unassisted.",
    metricLabel: "Drawing coverage",
    metricKey: "coverage",
    metricSuffix: "",
    preview: "/experiments/drawing/preview.svg",
    poster: "/experiments/drawing/poster.jpg",
    video: "/experiments/drawing/rollout.mp4",
    evidence: ["83/93 observations", "15 actions", "Contact-based ink", "Simulation only"],
    artifactStem: "drawing",
    artifactFiles: {
      checkpoint: "policy.zip",
      metadata: "policy.zip.json",
      onnx: "policy.onnx",
      evaluation: "eval.json",
      renderSheet: "frame_sheet.png",
      renderVideo: "rollout.mp4",
    },
    commandAdapter: "drawing",
    defaultRunName: "drawing-pilot-01",
    fullTimesteps: DRAWING_TOOLS.brush.fullTimesteps,
    fullEnvs: 1,
    fullNumSteps: DRAWING_TOOLS.brush.fullNumSteps,
    fullNumMinibatches: DRAWING_TOOLS.brush.fullNumMinibatches,
    maxEpisodeSeconds: DRAWING_TOOLS.brush.maxEpisodeSeconds,
    seed: 7,
    initialStd: DRAWING_TOOLS.brush.initialStd,
    normalizeRewards: false,
    fullEnvironment: {
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
    },
    ppo: {
      ...DRAWING_TOOLS.brush.ppo,
    },
    guideAnchor: "swimming-duck-drawing",
    readiness: "RUN-SPECIFIC · use the saved evaluation and matching rollout to verify the selected Pencil or Brush policy. Simulation only.",
    policyContract: "Independent simulation ONNX · microduck-brush-v2 · obs[batch,93] → actions[batch,15]",
    guidance: {
      requiredInput:
        "No clip or velocity command. Select Pencil or Brush; the environment supplies the authored swimming-duck target and preloads the selected free tool in the active lower jaw.",
      steps: [
        "Open Swimming-duck drawing.",
        "Run Pipeline smoke to invoke the selected independent drawing adapter with its bounded minimum budget.",
        "Switch to Default full for the selected tool's complete training preset.",
        "Keep the selected tool's fixed reward and trainer recipe unchanged.",
        "Run evaluation and require drawing_assessment.passed true with explicit unassisted evidence.",
        "Render the same policy.onnx source and inspect frame_sheet.png plus rollout.mp4.",
        "Treat every run as diagnostic until both the source-bound evaluation and render receipt match.",
      ],
      distinctions: [
        "Pencil is an independent 83/15 contract and Brush is an independent 93/15 contract; neither is the shared deployable 61/14 robot policy contract.",
        "The lower jaw actively holds and moves the selected free tool; it is not welded to the robot.",
        "Ink is generated by physical tool-surface contact, not by projecting the target or drawing directly from policy coordinates.",
        "A finite rollout, reward increase, or visually plausible trace is not an accepted learned result.",
      ],
      output:
        "policy.zip, policy.zip.json, policy.onnx, eval.json, render/frame_sheet.png, render/rollout.mp4, and a source-bound render evidence receipt.",
    },
    reward: REWARDS.drawing,
  },
] as const;

export function getExperiment(id: unknown): ExperimentDefinition {
  return (
    EXPERIMENTS.find((experiment) => experiment.id === id) ?? EXPERIMENTS[0]
  );
}

export function isExperimentId(value: unknown): value is ExperimentId {
  return EXPERIMENTS.some((experiment) => experiment.id === value);
}

export function defaultRewardWeights(
  experimentId: ExperimentId
): Record<string, number> {
  return Object.fromEntries(
    getExperiment(experimentId).reward.terms
      .filter((term) => term.editable !== false)
      .map((term) => [term.key, term.defaultWeight])
  );
}
