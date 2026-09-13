import type {
  ExperimentDefinition,
  ExperimentId,
  RewardTermDefinition,
} from "./experiments";

type Translate = (english: string, chinese: string) => string;

interface GuidanceTranslation {
  requiredInput: string;
  steps: string[];
  distinctions: string[];
  output: string;
  rewardSummary: string;
  rewardTerms: Record<string, { label: string; description: string }>;
}

const GUIDANCE_ZH: Record<ExperimentId, GuidanceTranslation> = {
  dance: {
    requiredInput: "一个已保存的动画片段，其中包含编舞所需的姿态和时间信息。不需要速度指令。",
    steps: [
      "打开“舞蹈模仿”。",
      "使用“动画”创建或编辑舞蹈片段。",
      "保存片段。",
      "为舞蹈训练选择该片段。",
      "先运行流水线冒烟测试。",
      "切换到完整训练并选择“启动 RLX”。",
      "查看奖励历史。",
      "运行评估。",
      "运行渲染以生成两分钟视频。",
      "接受策略前检查视频。",
    ],
    distinctions: [
      "片段表示“按此时间执行这些姿态”。",
      "指令表示“以此速度移动或按此速率转向”。舞蹈模仿不需要指令。",
      "PPO 是学习算法。它从所选配方的观测和奖励中学习。",
      "片段提供编舞，而 PPO 学习平衡、接触、动量和执行器动作。",
    ],
    output: "检查点、确定性评估记录、两分钟回放视频、帧预览图、元数据和可部署 ONNX 策略。",
    rewardSummary: "PPO 因匹配编排姿态和时间而获得奖励，同时需要保持直立、向前移动，并避免不安全接触或浪费执行器输出。",
    rewardTerms: {
      pose_match: { label: "姿态匹配", description: "匹配片段当前帧的关节姿态。" },
      rotation_match: { label: "旋转时序", description: "匹配片段当前时刻的身体旋转。" },
      stick_it: { label: "稳住落地", description: "以双脚站立完成一次性动作。" },
      on_feet: { label: "保持站立", description: "循环舞蹈期间保持直立、伸展并维持足部接触。" },
      travel: { label: "向前移动", description: "向前移动，而不是在原地完成循环。" },
      no_slip: { label: "脚部打滑", description: "惩罚支撑脚在地面上滑动。" },
      no_spin: { label: "非预期旋转", description: "当舞蹈应直线移动时惩罚偏航运动。" },
      gentle_head: { label: "头部撞击", description: "按严重程度惩罚头部与地面撞击。" },
      soft_landings: { label: "重落地", description: "惩罚过大的落地冲击。" },
      no_limit_parking: { label: "关节限位", description: "惩罚关节长时间顶住行程端点。" },
      save_energy: { label: "电机用力", description: "惩罚过大的执行器扭矩。" },
    },
  },
  swing: {
    requiredInput: "不需要片段或速度指令。秋千机构、初始静止状态和摆动奖励定义了 PPO 必须探索的行为。",
    steps: [
      "打开“自主荡秋千”。",
      "确认已选择秋千机构配方；不要添加舞蹈片段或移动指令。",
      "先运行流水线冒烟测试，验证机构、观测、奖励和产物路径。",
      "切换到完整训练并选择“启动 RLX”。",
      "训练过程中查看奖励历史和测得的秋千摆幅。",
      "运行完整评估：默认技能目标为对称 150° 摆幅（每侧 75°）、完整 24 秒、有效几何结构，并且每个回合的绳索保持张紧。流水线冒烟不评估技能。",
      "运行渲染以生成完整秋千回放。",
      "检查视频中的完整受控摆幅、绳索张力和较小横向漂移。",
      "只有评估和视觉检查都通过后才接受策略。",
    ],
    distinctions: [
      "不使用片段，因为 PPO 必须自行探索摆动动作。",
      "不使用指令，因为目标是机构运动，而不是指定速度或转向率。",
      "PPO 根据摆角增长和安全奖励学习，并选择 14 个执行器动作。",
    ],
    output: "秋千检查点、严格摆幅评估、完整回放视频、帧预览图、元数据和可部署 ONNX 策略。",
    rewardSummary: "PPO 因扩展双向摆幅边界、获得可控高度和能量而得到奖励，同时由几何与执行器惩罚约束动作保持在平面内并符合物理条件。",
    rewardTerms: {
      swing_peak_progress: { label: "新摆幅边界", description: "仅在秋千任一侧达到新峰值时奖励。" },
      swing_height: { label: "秋千高度", description: "在整个回合中奖励高度。" },
      swing_late_height: { label: "后期高度", description: "提高尝试后期高度的价值。" },
      swing_energy: { label: "摆动能量", description: "奖励有效高度和角速度。" },
      swing_lateral_penalty: { label: "横向偏移", description: "惩罚偏离秋千平面的横向位移。" },
      swing_lateral_barrier_penalty: { label: "横向边界", description: "强烈惩罚超出安全横向范围的运动。" },
      swing_lateral_velocity_penalty: { label: "横向速度", description: "惩罚快速横向运动。" },
      swing_out_of_plane_penalty: { label: "平面外旋转", description: "惩罚滚转和偏航角速度。" },
      swing_alignment_penalty: { label: "连接轴对齐", description: "保持两个连接轴对齐。" },
      swing_alignment_barrier_penalty: { label: "对齐边界", description: "强烈惩罚超出有效范围的对齐误差。" },
      string_slack_penalty: { label: "绳索松弛", description: "惩罚绳索松弛或长度不一致。" },
      string_extension_penalty: { label: "绳索伸长", description: "惩罚绳索超出设计长度的拉伸。" },
      invalid_episode_penalty: { label: "无效几何", description: "机构持续无效后惩罚整个回合。" },
      action_rate_penalty: { label: "动作变化", description: "惩罚策略动作的突变。" },
      joint_torque_penalty: { label: "关节扭矩", description: "惩罚过大的执行器力。" },
      joint_limit_penalty: { label: "关节限位", description: "惩罚超出关节限位的运动。" },
    },
  },
  running: {
    requiredInput: "不需要动画片段。环境向 PPO 提供前进速度指令和速度课程。",
    steps: [
      "打开“高速奔跑”。",
      "确认已选择前进速度配方；不需要舞蹈片段。",
      "先运行流水线冒烟测试，验证指令、观测、奖励和产物输出。",
      "切换到完整训练并选择“启动 RLX”。",
      "查看奖励历史、前进速度、存活率和漂移。",
      "在标称条件以及齿隙与扰动压力条件下运行评估。",
      "运行渲染以生成持续奔跑回放。",
      "检查视频中的前进进度、受控腾空、较小横向漂移以及无跌倒恢复。",
      "只有速度、稳定性和视觉检查都通过后才接受策略。",
    ],
    distinctions: [
      "不使用片段，因为奔跑由指令驱动，而不是按时间编排姿态。",
      "指令表示环境采样的目标前进速度。",
      "PPO 学习满足这些指令所需的步态、平衡、接触、动量和执行器动作。",
    ],
    output: "奔跑检查点、标称和压力评估指标、回放视频、帧预览图、元数据和可部署 ONNX 策略。",
    rewardSummary: "PPO 跟随前进速度指令，因受控腾空和姿态获得奖励，并为脚部、执行器和身体运动误差付出代价。",
    rewardTerms: {
      keep_pace: { label: "速度匹配", description: "匹配指令要求的机体坐标系速度。" },
      track_turn: { label: "转向率匹配", description: "匹配指令要求的偏航率。" },
      air_time: { label: "奔跑腾空", description: "奖励双脚在空中保持有效时长。" },
      flight: { label: "双脚腾空", description: "可选的有界双脚离地奖励，由直立姿态和指令方向身体速度门控；默认禁用。" },
      yaw_tracking: { label: "精确偏航跟踪", description: "独立于滚转和俯仰的可选偏航跟踪奖励，使用观测陀螺仪和指令转向率；默认禁用。" },
      stay_upright: { label: "保持直立", description: "在允许奔跑前倾的同时保持身体直立。" },
      pose: { label: "奔跑姿态", description: "跟踪与速度相匹配的腿部姿态。" },
      head_up: { label: "头部姿态", description: "保持头部与其指令对齐。" },
      foot_clearance: { label: "脚部离地高度", description: "惩罚运动脚处于错误高度。" },
      plant_the_foot: { label: "支撑脚打滑", description: "惩罚支撑脚滑动。" },
      smooth_moves: { label: "动作平滑度", description: "随着步态课程推进，惩罚突兀动作。" },
      no_limit_parking: { label: "关节限位", description: "惩罚关节长时间顶住行程端点。" },
      calm_roll: { label: "身体滚转速率", description: "惩罚过大的躯干滚转和俯仰角速度。" },
    },
  },
  stilts: {
    requiredInput: "不需要动画片段。选择高跷高度、支撑混合和质量；环境随后提供行走速度指令。",
    steps: [
      "打开“高跷行走”。",
      "选择要训练的高跷高度、支撑混合和质量。",
      "先运行流水线冒烟测试，验证该精确形态、指令、奖励和产物路径。",
      "先运行流水线冒烟测试，验证该精确形态、指令、奖励和产物路径。",
      "切换到完整训练并选择“启动 RLX”。",
      "查看奖励历史、前进速度、存活率、倾斜和接触稳定性。",
      "使用与训练相同的高跷高度、混合和质量运行评估。",
      "运行渲染以生成高跷行走回放。",
      "检查视频中的稳定接触、受控倾斜、前进进度以及无跌倒恢复。",
      "仅针对元数据中记录的形态接受该策略。",
    ],
    distinctions: [
      "不使用片段，因为高跷行走是指令驱动的运动。",
      "指令表示环境提供的目标行走速度或转向率。",
      "PPO 针对一种明确的高跷形态学习平衡和执行器动作；改变硬件形状需要匹配的训练与评估。",
    ],
    output: "形态专用检查点、评估记录、回放视频、帧预览图、包含高跷尺寸的元数据和可部署 ONNX 策略。",
    rewardSummary: "PPO 跟随行走指令，同时优先保持所选高跷上的直立平衡、受控脚部摆动、头部姿态和身体运动平滑。",
    rewardTerms: {
      track_lin_vel: { label: "行走速度匹配", description: "匹配请求的线速度。" },
      track_ang_vel: { label: "转向率匹配", description: "匹配请求的偏航率。" },
      yaw_tracking: { label: "精确偏航跟踪", description: "独立于滚转和俯仰的可选偏航跟踪奖励，使用观测陀螺仪和指令转向率；默认禁用。" },
      upright: { label: "直立平衡", description: "优先保持抬高接触点上的躯干水平。" },
      pose: { label: "腿部姿态", description: "可选地拉向默认腿部姿态；默认禁用。" },
      head_pose: { label: "头部姿态", description: "跟踪请求的头部姿态，同时不压制平衡目标。" },
      feet_air_time: { label: "脚部腾空时间", description: "奖励行走时受控的摆动阶段。" },
      action_rate_penalty: { label: "动作变化", description: "应用行走课程动态动作变化惩罚的 20%。" },
      ang_vel_xy_penalty: { label: "身体角速度", description: "惩罚过大的滚转和俯仰角速度。" },
    },
  },
  backflip: {
    requiredInput: "不需要动画片段或速度指令。Python 通过协议 spotter-launch-landing-stand-v2 管理辅助起跳 → PPO 落地 → 预训练站立策略接管。",
    steps: [
      "打开“后空翻展示”。",
      "保持辅助起跳 → PPO 落地 → 预训练站立策略接管不变。",
      "先运行流水线冒烟测试，验证四步 API、观测、奖励和产物路径。",
      "切换到默认完整训练并选择“启动 RLX”。",
      "新的完整训练会精确迁移预训练站立 actor 和观测归一化器来初始化落地学生策略，然后运行 PPO；流水线冒烟会跳过教师初始化。",
      "后空翻 PPO 开始前，Python 收集 16 个预训练教师落地回合，每个 600 步（共 9,600 个校准步），固定折扣回报 RMS，并执行 10 个仅 critic 的校准 epoch。actor 保持不变，校准后的奖励归一化被冻结，PPO 使用新优化器开始。",
      "起跳在 5.3 rad 阈值释放，测得释放旋转为 5.376–5.380 rad，角速度介于 3.5 和 6.5 rad/s；外部抬升和俯仰只在释放前作用。",
      "检查接管前至少 5.8 rad 的计入旋转，随后连续 10 个稳定 PPO 落地步，期间无辅助者、站立覆盖或身体支撑。",
      "运行完整评估，验证辅助起跳 → PPO 落地 → 预训练站立策略接管，要求通过无支撑接管前门槛并稳定站立至少 2 秒。",
      "运行渲染，检查完整辅助起跳、PPO 更新后的落地和预训练站立策略接管。",
      "仅当评估器报告 skill_status passed 且绑定源文件的渲染匹配时接受。",
    ],
    distinctions: [
      "辅助起跳 → PPO 落地 → 预训练站立策略接管。",
      "完整控制器始终包含 Python 管理的起跳辅助和用于最终保持的预训练站立策略。",
      "后空翻专用热校准可避免 PPO 启动时的奖励尺度冲击；它本身不能证明 PPO 有正向提升。",
      "测得的参考 backflip-e2e-20260908-v5 在两组评估中通过 24/24 个展示回合，但其配对的仅落地回报相较初始化下降 12.63%，独立落地通过 7/8。扩展学习审计失败；尚未证明快速、精确的 PPO 提升。",
      "站立策略阶段不能补足计入旋转；接管前必须完成至少 5.8 rad。",
      "导出的后空翻 ONNX 仅包含 PPO 更新后的落地策略，不包含辅助协议或预训练站立策略接管。",
    ],
    output: "落地检查点和 ONNX、backflip_assessment 评估、回放视频、帧预览图和元数据。仅落地 ONNX 并不是完整后空翻控制器。",
    rewardSummary: "仅在 PPO 落地阶段，密集可观测项奖励直立姿态和稳定过程，有界惩罚项约束关节速度、动作变化和动作幅度。辅助起跳和预训练站立策略接管期间所有项均为零。",
    rewardTerms: {
      landing_upright: { label: "落地直立", description: "在 PPO 落地期间奖励归一化躯干方向：(1 - gravity z) / 2，限制在 [0, 1]。" },
      landing_pose: { label: "落地姿态", description: "奖励直立姿态乘以逐关节宽 0.6 rad 与精确 0.15 rad 平方指数姿态核的平均值，限制在 [0, 1]。" },
      landing_settle: { label: "落地稳定", description: "使用 exp(-||gyro||² / 2) 奖励直立姿态稳定，限制在 [0, 1]。" },
      landing_joint_speed_penalty: { label: "落地关节速度", description: "惩罚裁剪后的平均关节速度平方除以 100；该项限制在 [-1, 0]。" },
      landing_action_rate_penalty: { label: "落地动作变化", description: "惩罚 PPO 落地期间裁剪后的平均动作差平方；该项限制在 [-1, 0]。" },
      landing_action_size_penalty: { label: "落地动作幅度", description: "惩罚裁剪后的平均动作平方除以 16；该项限制在 [-1, 0]。" },
    },
  },
  basketball: {
    requiredInput: "不需要片段。使用已发布的循环篮球引导策略；隐藏状态和单元状态在控制步之间延续，仅随回合重置。",
    steps: [
      "打开“篮球平衡”。",
      "保持循环策略合约和固定奖励配方不变。",
      "运行流水线冒烟测试，验证独立适配器、循环 ONNX 和运行自有产物。",
      "Studio 以保持时间 0、课程关闭的方式开始无辅助微调。适配器的保持/课程阶梯是仅能通过显式 CLI 启用的训练选项。",
      "将运行视为仅平衡预览前，先验证原地无辅助平衡 60 秒。",
      "评估当前零指令平衡和 0.15 前进指令跟踪。横向和偏航转向尚未评估。",
      "在平衡和转向标准都明确通过前，不要将完整任务标记为通过。",
      "渲染并检查完整无辅助回放，确保循环状态持续传递。",
    ],
    distinctions: [
      "已安装的 basketball-balance-01 预览仅是平衡证据，不代表转向成功。",
      "actor 不观察球体状态，而是使用共享的 61 维机器人与指令输入，加上循环隐藏状态。",
      "有限值回放、高奖励或仅 60 秒平衡都不能证明指令驱动转向。",
    ],
    output: "checkpoint.pt、policy.onnx、result.json、eval.json 和绑定源文件的渲染产物。在转向通过前，完整就绪状态仍受门槛限制。",
    rewardSummary: "独立循环篮球训练器拥有固定奖励；Studio 不展示会造成误导的奖励滑块。",
    rewardTerms: {},
  },
  bridge: {
    requiredInput: "不需要片段。共享桥梁注册表提供狭窄悬索桥和自动行走引导。",
    steps: [
      "打开“悬索桥”。",
      "先运行流水线冒烟测试，验证注册表、观测、固定奖励和产物路径。",
      "使用默认完整训练执行有界的 32,768 步迁移行走试验和仅用于训练的桥梁课程。",
      "在桥面逐步变窄并悬空前，让自动行走引导迁移行走 actor 和观测归一化器。",
      "在完整 20 秒时长内无辅助评估至少 1,000 个控制步。",
      "要求明确的桥梁评估成功；仅流水线有效和部分进度不能通过验收。",
      "渲染相同的无辅助最终桥梁，并在标记已验证前检查完整过桥过程。",
    ],
    distinctions: [
      "桥梁课程仅用于训练。评估和渲染在最终狭窄悬索桥上无辅助运行。",
      "自动引导和检查点续训是使用全新 critic 与优化器的 actor 热启动，并非精确恢复优化器。",
      "已有训练试验，但任务尚无已验证的过桥预览。",
      "奖励滑块为空是有意设计，因为注册的桥梁配方拥有固定奖励键。",
    ],
    output: "标准 Studio 检查点和 ONNX、明确的 bridge_assessment 评估，以及绑定源文件的无辅助渲染产物。",
    rewardSummary: "共享桥梁配方拥有固定奖励和分阶段悬索桥课程；Studio 不展示奖励滑块。",
    rewardTerms: {},
  },
  drawing: {
    requiredInput: "不需要片段或速度指令。选择铅笔或画笔；环境提供编排好的游泳鸭目标，并将所选自由工具预装到活动下颚中。",
    steps: [
      "打开“游泳鸭绘画”。",
      "运行流水线冒烟测试，以有界最小预算调用所选独立绘画适配器。",
      "切换到默认完整训练，使用所选工具的完整训练预设。",
      "保持所选工具的固定奖励和训练器配方不变。",
      "运行评估，并要求 drawing_assessment.passed 为 true，且有明确的无辅助证据。",
      "渲染同一 policy.onnx 源文件，并检查 frame_sheet.png 和 rollout.mp4。",
      "在绑定源文件的评估和渲染回执都匹配前，将每次运行视为诊断结果。",
    ],
    distinctions: [
      "铅笔使用独立 83/15 合约，画笔使用独立 93/15 合约；两者都不是共享的可部署 61/14 机器人策略合约。",
      "下颚主动夹持并移动所选自由工具；工具没有焊接到机器人上。",
      "墨迹由工具与表面的物理接触生成，而不是投影目标或直接根据策略坐标绘制。",
      "有限值回放、奖励上升或视觉上合理的轨迹都不代表已验收的学习结果。",
    ],
    output: "policy.zip、policy.zip.json、policy.onnx、eval.json、render/frame_sheet.png、render/rollout.mp4，以及绑定源文件的渲染证据回执。",
    rewardSummary: "所选独立绘画环境拥有固定奖励，用于接触墨迹、目标曲线覆盖、精度、工具控制和机器人稳定运动。",
    rewardTerms: {},
  },
};

export function experimentGuidanceDisplay(
  experiment: ExperimentDefinition,
  t: Translate
) {
  const translated = GUIDANCE_ZH[experiment.id];
  return {
    requiredInput: t(experiment.guidance.requiredInput, translated.requiredInput),
    steps: experiment.guidance.steps.map((step, index) =>
      t(step, translated.steps[index] ?? step)
    ),
    distinctions: experiment.guidance.distinctions.map((item, index) =>
      t(item, translated.distinctions[index] ?? item)
    ),
    output: t(experiment.guidance.output, translated.output),
    rewardSummary: t(experiment.reward.summary, translated.rewardSummary),
  };
}

export function rewardTermDisplay(
  experimentId: ExperimentId,
  term: RewardTermDefinition,
  t: Translate
) {
  const translated = GUIDANCE_ZH[experimentId].rewardTerms[term.key];
  return {
    label: translated ? t(term.label, translated.label) : term.label,
    description: translated
      ? t(term.description, translated.description)
      : term.description,
  };
}
