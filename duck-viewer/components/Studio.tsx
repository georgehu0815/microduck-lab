"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import Link from "next/link";

import {
  LAB_HTTP,
  LabClient,
  fetchPolicies,
  type Frame,
  type Policy,
} from "@/lib/lab";
import {
  defaultRewardWeights,
  EXPERIMENTS,
  getExperiment,
  getDrawingTool,
  type DrawingTool,
  type ExperimentDefinition,
  type ExperimentId,
} from "@/lib/experiments";
import { evaluationVerdict } from "@/lib/evaluation";
import { experimentGuidanceDisplay, rewardTermDisplay } from "@/lib/experiment-labels";
import { liveContractLabels } from "@/lib/live-contract";
import { evidenceLabel, rolloutMetricRanges, skillEvidence } from "@/lib/studio-evidence";
import { availableProfileRunName, fetchSavedRuns, latestDiagnosticRun, latestReviewRun, latestVerifiedRun, type SavedRun } from "@/lib/studio-run";
import { IS_STATIC_EXPORT, publicAssetUrl, staticStudioArtifactUrl } from "@/lib/static-assets";
import type { RlxRecipe } from "@/lib/rlx-job";
import type { RlxTrainingHistory } from "@/lib/rlx-history";
import { AnimPanel } from "./AnimPanel";
import { LanguageToggle, useLanguage } from "./LanguageProvider";
import { PolicyPanel } from "./PolicyPanel";
import { TeachPanel } from "./TeachPanel";
import Viewer from "./Viewer";
import styles from "./Studio.module.css";

type Operation = "train" | "eval" | "render" | "export";
type Phase = "idle" | "running" | "succeeded" | "failed" | "cancelled";

type Recipe = RlxRecipe;

interface JobState {
  renderVerified?: boolean;
  renderEvidenceId?: string | null;
  phase: Phase;
  operation: Operation | null;
  activeJob?: {
    operation: Operation;
    experimentId: ExperimentId;
    runName: string;
    startedAt: string | null;
  } | null;
  experimentId: ExperimentId;
  drawingTool?: DrawingTool;
  runName: string;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  evaluation: Record<string, unknown> | null;
  rewardHistory: RewardPoint[];
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  savedRecipe?: Recipe | null;
  trainingHistory?: RlxTrainingHistory;
  artifacts: {
    checkpoint: boolean;
    metadata: boolean;
    onnx: boolean;
    renderSheet: boolean;
    renderVideo: boolean;
    evaluation: boolean;
    checkpointPath: string;
    onnxPath: string;
    renderDirectory: string;
  };
}

interface RewardPoint {
  step: number;
  reward: number;
}

type Translate = (english: string, chinese: string) => string;

const EXPERIMENT_ZH: Record<ExperimentId, {
  title: string;
  shortTitle: string;
  goal: string;
  description: string;
  task: string;
  metricLabel: string;
  readiness: string;
  evidence: string[];
}> = {
  dance: {
    title: "舞蹈模仿",
    shortTitle: "舞蹈",
    goal: "从编排好的动作循环中学习流畅、可重复的两分钟舞蹈。",
    description: "使用关节角参考训练策略，并通过确定性评估和完整视频检查动作质量。",
    task: "RLX 模仿 · dance-120bpm",
    metricLabel: "平均回报",
    readiness: "Studio 工作流已验证；每次运行仍需仿真证据。",
    evidence: ["61 维观测", "14 维动作", "50 Hz", "支持两分钟渲染"],
  },
  swing: {
    title: "自主荡秋千",
    shortTitle: "秋千",
    goal: "从静止开始，探索协调的身体动作来逐步荡高。",
    description: "通过教师初始化和 PPO 优化学习受绳索约束的周期动作，并严格检查几何与张力。",
    task: "RLX 秋千 · 仅受拉绳索",
    metricLabel: "峰峰摆幅",
    readiness: "已有验证参考；每次 Studio 运行仍需绑定评估与渲染证据。",
    evidence: ["173.20° 参考摆幅", "严格种子 71/100", "0.7 动作缩放"],
  },
  running: {
    title: "高速奔跑",
    shortTitle: "奔跑",
    goal: "先学习快速前进步态，再增强其对延迟和扰动的鲁棒性。",
    description: "用速度课程学习前进步态，并在随机化环境中检查速度、存活率和控制稳定性。",
    task: "RLX 奔跑 · 前进速度课程",
    metricLabel: "前进速度",
    readiness: "已有验证参考；每次 Studio 运行仍需绑定评估与渲染证据。",
    evidence: ["标称 1.651 m/s", "压力测试 1.612 m/s", "压力测试存活率 98.44%"],
  },
  stilts: {
    title: "高跷行走",
    shortTitle: "高跷",
    goal: "在指定高跷高度和支撑形状下学习稳定行走。",
    description: "针对固定形态训练步态，并分别验证每种高度与支撑配置。",
    task: "RLX 高跷 · 固定形态",
    metricLabel: "前进速度",
    readiness: "仅特定形态已有验证参考。",
    evidence: ["展示 10 cm", "0.50 支撑混合", "已发布 8 种高度"],
  },
  backflip: {
    title: "后空翻展示",
    shortTitle: "后空翻",
    goal: "辅助起跳 → PPO 落地 → 预训练站立策略接管，完成整周旋转并稳定站立至少两秒。",
    description: "训练可导出的 PPO 落地阶段，并将辅助起跳与站立接管作为外部控制协议进行验证。",
    task: "RLX 后空翻 · 辅助起跳 → PPO 落地 → 预训练站立策略接管",
    metricLabel: "旋转量",
    readiness: "展示已验证；尚未证明 PPO 有正向提升，也不是完整硬件控制器。",
    evidence: ["第 2 版辅助协议", "无支撑策略落地", "稳定站立 ≥2 秒"],
  },
  basketball: {
    title: "篮球平衡",
    shortTitle: "篮球",
    goal: "先在自由篮球上保持平衡，再学习无辅助的指令滚动。",
    description: "使用循环策略学习球上平衡与转向，并将仅平衡证据与完整任务通过明确区分。",
    task: "RLX 篮球 · 循环平衡与转向课程",
    metricLabel: "无辅助平衡",
    readiness: "仅平衡 · 已验证 60 秒平衡；转向未通过。",
    evidence: ["已验证 60 秒平衡", "转向未通过", "需要循环状态"],
  },
  bridge: {
    title: "悬索桥",
    shortTitle: "桥梁",
    goal: "无辅助跨越狭窄悬索桥，不跌落，也不从侧边退出。",
    description: "使用训练课程学习桥面行走，并在最终狭窄悬索桥上执行完整无辅助评估。",
    task: "RLX 桥梁 · 狭窄悬索桥课程",
    metricLabel: "过桥进度",
    readiness: "训练试验 · 已有 32,768 步迁移行走试验；尚未验证成功过桥。",
    evidence: ["新悬索桥", "可用训练课程", "尚未验证"],
  },
  drawing: {
    title: "游泳鸭绘画",
    shortTitle: "绘画",
    goal: "用下颚夹持铅笔或四色画笔，通过真实接触墨迹绘制游泳鸭目标。",
    description: "在独立仿真策略合约中训练工具控制，并用实际工具轨迹生成的墨迹验证绘画结果。",
    task: "RLX 绘画 · 铅笔或画笔",
    metricLabel: "绘画覆盖率",
    readiness: "按运行验证 · 使用保存的评估和匹配回放验证所选铅笔或画笔策略。仅限仿真。",
    evidence: ["83/93 维观测", "15 维动作", "接触生成墨迹", "仅限仿真"],
  },
};

function experimentDisplay(experiment: ExperimentDefinition, t: Translate) {
  const translated = EXPERIMENT_ZH[experiment.id];
  return {
    ...experiment,
    title: t(experiment.title, translated.title),
    shortTitle: t(experiment.shortTitle, translated.shortTitle),
    goal: t(experiment.goal, translated.goal),
    description: t(experiment.description, translated.description),
    task: t(experiment.task, translated.task),
    metricLabel: t(experiment.metricLabel, translated.metricLabel),
    readiness: t(experiment.readiness, translated.readiness),
    evidence: experiment.evidence.map((item, index) =>
      t(item, translated.evidence[index] ?? item)
    ),
  };
}

function drawingToolLabel(tool: DrawingTool, t: Translate) {
  return tool === "pencil" ? t("Pencil", "铅笔") : t("Brush", "画笔");
}

function operationDisplay(operation: Operation | null, t: Translate) {
  if (operation === "train") return t("training", "训练");
  if (operation === "eval") return t("evaluation", "评估");
  if (operation === "render") return t("rendering", "渲染");
  if (operation === "export") return t("export", "导出");
  return t("operation", "操作");
}

function recipeDefaults(
  experimentId: ExperimentId,
  drawingTool: DrawingTool = "brush"
): Recipe {
  const experiment = getExperiment(experimentId);
  const drawing = experimentId === "drawing";
  const drawingDefinition = drawing ? getDrawingTool(drawingTool) : null;
  return {
    experimentId,
    drawingTool: drawing ? drawingTool : undefined,
    runName: experiment.defaultRunName,
    profile: "smoke",
    totalTimesteps: drawing ? drawingDefinition!.smokeTimesteps : 4,
    numEnvs: drawing ? experiment.fullEnvs : 2,
    numSteps: drawing ? drawingDefinition!.fullNumSteps : 2,
    numMinibatches: drawing ? drawingDefinition!.fullNumMinibatches : 1,
    maxEpisodeS: drawing ? drawingDefinition!.maxEpisodeSeconds : 1,
    evalSteps: 4,
    evalEpisodes: drawingTool === "brush" && drawing ? drawingDefinition!.evalEpisodes : 1,
    renderSeconds: drawing
      ? drawingDefinition!.maxEpisodeSeconds
      : experiment.maxEpisodeSeconds,
    danceClip: null,
    dancePoseSigma: null,
    locomotionForwardCommand: null,
    initialStd: drawing ? drawingDefinition!.initialStd : experiment.initialStd,
    normalizeRewards: experiment.normalizeRewards,
    freezeObservationNormalization: experimentId === "basketball",
    checkpointInterval: 0,
    seed: experiment.seed,
    learningRate: drawing ? drawingDefinition!.ppo.learningRate : experiment.ppo.learningRate,
    gamma: drawing ? drawingDefinition!.ppo.gamma : experiment.ppo.gamma,
    clipCoefficient: drawing ? drawingDefinition!.ppo.clipCoefficient : experiment.ppo.clipCoefficient,
    updateEpochs: drawing ? drawingDefinition!.ppo.updateEpochs : experiment.ppo.updateEpochs,
    entropyCoefficient: drawing ? drawingDefinition!.ppo.entropyCoefficient : experiment.ppo.entropyCoefficient,
    maxGradNorm: drawing ? drawingDefinition!.ppo.maxGradNorm : experiment.ppo.maxGradNorm,
    domainRand: false,
    obsNoise: false,
    actionDelay: false,
    randomYaw: false,
    bridgeCurriculum: false,
    stiltHeightCm: experiment.controls?.stiltHeightCm ?? 10,
    stiltBlend: experiment.controls?.stiltBlend ?? 0.5,
    stiltMassKg: experiment.controls?.stiltMassKg ?? 0.029,
    swingInitialAngleDeg: 0,
    swingInitialRateRadS: 0,
    swingPlanarActions: experimentId === "swing",
    swingMinSpanDeg: 150,
    resumeFromCheckpoint: false,
    rewardWeights: defaultRewardWeights(experimentId),
  };
}

function fullHorizon(
  experimentId: ExperimentId,
  danceDuration?: number,
  drawingTool: DrawingTool = "brush"
) {
  const experiment = getExperiment(experimentId);
  const drawing = experimentId === "drawing" ? getDrawingTool(drawingTool) : null;
  const seconds = drawing?.maxEpisodeSeconds ?? (
    experimentId === "dance" && danceDuration !== undefined
      ? danceDuration
      : Math.max(experimentId === "running" ? 12 : 0, experiment.maxEpisodeSeconds)
  );
  return {
    numSteps: drawing?.fullNumSteps ?? experiment.fullNumSteps,
    numMinibatches: drawing?.fullNumMinibatches ?? experiment.fullNumMinibatches,
    maxEpisodeS: seconds,
    evalSteps: experimentId === "dance"
      ? Math.ceil(seconds * 50)
      : Math.max(experimentId === "bridge" ? 1_000 : 500, Math.ceil(seconds * 50)),
    evalEpisodes: drawing?.evalEpisodes ?? 1,
    renderSeconds: seconds,
  };
}

function hasDefaultFullEnvironment(recipe: Recipe, experiment: ExperimentDefinition) {
  return recipe.domainRand === experiment.fullEnvironment.domainRand &&
    recipe.obsNoise === experiment.fullEnvironment.obsNoise &&
    recipe.actionDelay === experiment.fullEnvironment.actionDelay &&
    recipe.randomYaw === experiment.fullEnvironment.randomYaw;
}

const DEFAULT_RECIPE = recipeDefaults("dance");

const EMPTY_JOB: JobState = {
  phase: "idle",
  operation: null,
  activeJob: null,
  experimentId: "dance",
  runName: DEFAULT_RECIPE.runName,
  startedAt: null,
  finishedAt: null,
  exitCode: null,
  logs: [],
  result: null,
  evaluation: null,
  rewardHistory: [],
  normalizeRewards: false,
  trainingSteps: 0,
  trainingTotal: 0,
  artifacts: {
    checkpoint: false,
    metadata: false,
    onnx: false,
    renderSheet: false,
    renderVideo: false,
    evaluation: false,
    checkpointPath: "runs/studio/dance/dance-studio/dance.safetensors",
    onnxPath: "runs/studio/dance/dance-studio/dance.onnx",
    renderDirectory: "runs/studio/dance/dance-studio/render",
  },
};

const STEPS = [
  ["01", "Train", "Learn in simulation"],
  ["02", "Evaluate", "Measure the policy"],
  ["03", "Render", "Review the motion"],
  ["04", "Artifacts", "Export the policy"],
] as const;

function Icon({ children }: { children: React.ReactNode }) {
  return <span className={styles.icon} aria-hidden="true">{children}</span>;
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function formatAxisNumber(value: number) {
  const absolute = Math.abs(value);
  if (absolute > 0 && absolute < 0.01) return value.toExponential(1);
  if (absolute >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (absolute >= 1_000) return `${Math.round(value / 1_000)}k`;
  if (absolute >= 100) return Math.round(value).toString();
  if (absolute >= 10) return value.toFixed(0);
  return value.toFixed(absolute < 1 ? 3 : 1);
}

function rewardChart(points: RewardPoint[], totalSteps: number) {
  if (!points.length) return null;
  const left = 34;
  const right = 414;
  const top = 14;
  const bottom = 116;
  const rewards = points.map((point) => point.reward);
  let minimum = Math.min(...rewards);
  let maximum = Math.max(...rewards);
  if (minimum === maximum) {
    const padding = Math.max(1, Math.abs(minimum) * 0.1);
    minimum -= padding;
    maximum += padding;
  } else {
    const padding = (maximum - minimum) * 0.12;
    minimum -= padding;
    maximum += padding;
  }
  const xMaximum = Math.max(totalSteps, points.at(-1)?.step ?? 0, 1);
  const coordinates = points.map((point) => {
    const x = left + (Math.max(0, point.step) / xMaximum) * (right - left);
    const y = bottom - ((point.reward - minimum) / (maximum - minimum)) * (bottom - top);
    return [x, y] as const;
  });
  const line = coordinates
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`)
    .join(" ");
  const [firstX] = coordinates[0];
  const [lastX, lastY] = coordinates[coordinates.length - 1];
  return {
    line,
    area: `${line} L${lastX.toFixed(1)} ${bottom} L${firstX.toFixed(1)} ${bottom} Z`,
    lastX,
    lastY,
    yLabels: [maximum, (maximum + minimum) / 2, minimum],
    xLabels: [0, xMaximum / 3, (xMaximum * 2) / 3, xMaximum],
  };
}

function HistoryPlot({ title, points, start, end }: { title: string; points: RewardPoint[]; start: number; end: number }) {
  const { t } = useLanguage();
  const plot = rewardChart(points.map((point) => ({ ...point, step: point.step - start })), end - start);
  return <div>
    <h4>{title} · {points.length} {t("samples", "个样本")}</h4>
    {plot ? <svg viewBox="0 0 450 145" role="img" aria-label={t(`${title}, complete segment from ${start} to ${end} transitions`, `${title}，从 ${start} 到 ${end} 次转换的完整区段`)}>
      <path d={plot.line} />
      {plot.yLabels.map((label, index) => <text key={`y-${index}`} x="0" y={18 + index * 49}>{formatAxisNumber(label)}</text>)}
      {plot.xLabels.map((label, index) => <text key={`x-${index}`} x={34 + index * 126} y="132">{formatAxisNumber(label + start)}</text>)}
      <circle cx={plot.lastX} cy={plot.lastY} r="2" fill="currentColor" />
    </svg> : <p>{t("No measurements recorded for this series.", "此序列尚无测量记录。")}</p>}
  </div>;
}

function TrainingLifecycle({ history }: { history: RlxTrainingHistory | undefined }) {
  const { t } = useLanguage();
  if (!history?.segments.length) return null;
  const steps = history.segments.reduce((total, segment) => total + segment.trainedSteps, 0);
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(history, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "ppo-training-lifecycle.json";
    anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className={styles.lifecycle} aria-label={t("Full training lifecycle", "完整训练生命周期")}>
    <h3>{t("Full training lifecycle", "完整训练生命周期")}</h3>
    <p>{t(`${formatNumber(steps)} recorded PPO transitions · ${history.segments.length} stage${history.segments.length === 1 ? "" : "s"}. All recorded samples, from the first rollout through the final update; no 240-sample tail window.`, `已记录 ${formatNumber(steps)} 次 PPO 转换 · ${history.segments.length} 个阶段。包含从首次 rollout 到最终更新的全部样本，不限于末尾 240 个样本。`)}</p>
    <p>{t("Stages remain separate: running reward normalization changes the scale. Rollout reward is not deterministic evaluation return, and PPO losses need not decrease monotonically. Missing history is not reconstructed as invented data.", "各阶段保持分离：启用奖励归一化会改变数值尺度。Rollout 奖励不等于确定性评估回报，PPO 损失也不必单调下降。缺失历史不会用虚构数据补齐。")}</p>
    <button type="button" className={styles.button} onClick={download}>{t("Download full reward and loss history", "下载完整奖励与损失历史")}</button>
    {history.segments.map((segment) => <article key={segment.id} data-history-segment={segment.id}>
      <h4 title={segment.id}>{segment.kind === "bc-initializer" ? t("Teacher initialization (not PPO)", "教师初始化（非 PPO）") : t("PPO training stage", "PPO 训练阶段")} · {segment.id.split("/").at(-1)}</h4>
      {segment.initializer ? <p>{segment.initializer.label} · {segment.initializer.teacherSamples ?? t("Unknown", "未知")} {t("teacher samples. Skill acquisition can precede PPO refinement.", "个教师样本。技能习得可以先于 PPO 优化。")}</p> : <>
        <p>{formatNumber(segment.startStep)}–{formatNumber(segment.endStep)} {t("cumulative transitions", "累计转换")} · {segment.normalizationLabel} · {segment.status === "active" ? t("recording", "记录中") : t("recorded history (not a skill verdict)", "已记录历史（并非技能判定）")}</p>
        {segment.episodes.length > 0 && <HistoryPlot title={t("Raw training episode return", "原始训练回合回报")} points={segment.episodes.map((sample) => ({ step: sample.step, reward: sample.meanRawReturn }))} start={segment.startStep} end={segment.endStep} />}
        {segment.episodes.length > 0 && <p>{t("Raw episode returns mix episode lengths and training seeds/resets. Increasing returns can reflect longer survival; physical skill is checked independently below.", "原始回合回报混合了不同回合长度、训练种子和重置。回报上升可能仅表示存活更久；物理技能将在下方独立检查。")}</p>}
        <HistoryPlot title={segment.normalizationLabel} points={segment.collections.map((sample) => ({ step: sample.step, reward: sample.meanReward }))} start={segment.startStep} end={segment.endStep} />
        <HistoryPlot title={t("Policy loss", "策略损失")} points={segment.updates.flatMap((sample) => sample.policyLoss === null ? [] : [{ step: sample.step, reward: sample.policyLoss }])} start={segment.startStep} end={segment.endStep} />
        <HistoryPlot title={t("Value loss", "价值损失")} points={segment.updates.flatMap((sample) => sample.valueLoss === null ? [] : [{ step: sample.step, reward: sample.valueLoss }])} start={segment.startStep} end={segment.endStep} />
      </>}
    </article>)}
  </section>;
}

function elapsed(startedAt: string | null, endedAt: string | null, now: number) {
  if (!startedAt) return "00:00:00";
  const end = endedAt ? Date.parse(endedAt) : now;
  const seconds = Math.max(0, Math.floor((end - Date.parse(startedAt)) / 1000));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return [hours, minutes, seconds % 60]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function resultNumber(result: Record<string, unknown> | null, key: string) {
  const value = result?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatReward(value: number | null) {
  if (value === null) return "—";
  if (value !== 0 && Math.abs(value) < 0.05) return value.toExponential(2);
  return value.toFixed(2);
}

function recipeMetricNumber(
  result: Record<string, unknown> | null,
  experiment: ExperimentDefinition
) {
  const direct = resultNumber(result, experiment.metricKey);
  if (direct !== null) return direct;

  const recipeMetrics = result?.recipe_metrics;
  if (!recipeMetrics || typeof recipeMetrics !== "object") return null;
  const summary = (recipeMetrics as Record<string, unknown>)[experiment.metricKey];
  if (!summary || typeof summary !== "object") return null;
  const values = summary as Record<string, unknown>;

  return resultNumber(values, "mean");
}

function recipeMetricSummaryNumber(
  result: Record<string, unknown> | null,
  experiment: ExperimentDefinition,
  statistic: "mean" | "median" | "max"
) {
  const recipeMetrics = result?.recipe_metrics;
  if (!recipeMetrics || typeof recipeMetrics !== "object") return null;
  const summary = (recipeMetrics as Record<string, unknown>)[experiment.metricKey];
  if (!summary || typeof summary !== "object") return null;
  return resultNumber(summary as Record<string, unknown>, statistic);
}

function StatusPill({
  tone,
  children,
}: {
  tone: "live" | "warn" | "muted";
  children: React.ReactNode;
}) {
  return <span className={`${styles.statusPill} ${styles[tone]}`}>{children}</span>;
}

function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <label className={styles.toggle}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span aria-hidden="true" />
      {label}
    </label>
  );
}

export default function Studio() {
  const { t } = useLanguage();
  const [selectedExperimentId, setSelectedExperimentId] =
    useState<ExperimentId>("dance");
  const [recipe, setRecipe] = useState(DEFAULT_RECIPE);
  const [receivedJob, setJob] = useState<JobState>(EMPTY_JOB);
  const job = receivedJob.experimentId === recipe.experimentId && receivedJob.runName === recipe.runName
    ? receivedJob : { ...EMPTY_JOB, experimentId: recipe.experimentId, runName: recipe.runName };
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [danceClips, setDanceClips] = useState<{ path: string; name: string; durationSeconds: number }[]>([]);
  const [clipError, setClipError] = useState<string | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const liveContract = liveContractLabels(connected ? frame?.ducks : []);
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [advanced, setAdvanced] = useState(false);
  const [logsOpen, setLogsOpen] = useState(false);
  const [reviewedEvidence, setReviewedEvidence] = useState<string | null>(null);
  const reviewIdentity = `${recipe.experimentId}/${recipe.drawingTool ?? "standard"}/${recipe.runName}/${job.evaluation?.source_sha256 ?? "unverified"}/${job.renderEvidenceId ?? "unbound"}`;
  const visualReviewed = reviewedEvidence === reviewIdentity;
  function setVisualReviewed(reviewed: boolean) { setReviewedEvidence(reviewed ? reviewIdentity : null); }
  const [notice, setNotice] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [pendingAction, setPendingAction] = useState<Operation | "cancel" | null>(null);
  const [now, setNow] = useState(Date.now());
  const [labRewardHistory, setLabRewardHistory] = useState<RewardPoint[]>([]);
  const [viewerExpanded, setViewerExpanded] = useState(false);
  const [rewardHistoryOpen, setRewardHistoryOpen] = useState(false);
  const [rewardHistoryCopied, setRewardHistoryCopied] = useState(false);
  const [guidanceExperimentId, setGuidanceExperimentId] =
    useState<ExperimentId | null>(null);
  const [sessionTab, setSessionTab] = useState<
    "training" | "animate" | "policies" | "teach"
  >("training");
  const clientRef = useRef<LabClient | null>(null);
  const labRunRef = useRef<string | null>(null);
  const guidanceDialogRef = useRef<HTMLDialogElement | null>(null);
  const rewardHistoryDialogRef = useRef<HTMLDialogElement | null>(null);
  const selectedExperiment = getExperiment(selectedExperimentId);
  const selectedExperimentText = experimentDisplay(selectedExperiment, t);
  const selectedDrawingTool = selectedExperimentId === "drawing"
    ? getDrawingTool(recipe.drawingTool ?? "brush")
    : null;
  const selectedDrawingToolLabel = selectedDrawingTool
    ? drawingToolLabel(selectedDrawingTool.id, t)
    : null;
  const selectedPolicyContract = selectedDrawingTool
    ? `Independent simulation ONNX · ${selectedDrawingTool.contractVersion} · obs[batch,${selectedDrawingTool.observations}] → actions[batch,${selectedDrawingTool.actions}]`
    : selectedExperiment.policyContract;
  const guidanceExperiment = guidanceExperimentId
    ? getExperiment(guidanceExperimentId)
    : null;
  const guidanceExperimentText = guidanceExperiment
    ? experimentDisplay(guidanceExperiment, t)
    : null;
  const guidanceDetails = guidanceExperiment
    ? experimentGuidanceDisplay(guidanceExperiment, t)
    : null;
  const selectedDanceDuration = danceClips.find((clip) => recipe.danceClip === clip.path || recipe.danceClip?.endsWith(`/${clip.path}`))?.durationSeconds;

  useEffect(() => {
    window.localStorage.setItem(
      "microduck-studio-experiment",
      selectedExperimentId
    );
  }, [selectedExperimentId]);

  useEffect(() => {
    if (IS_STATIC_EXPORT) return;
    fetchPolicies().then(setPolicies).catch(() => setPolicies([]));
  }, []);

  useEffect(() => {
    if (IS_STATIC_EXPORT) return;
    let active = true;
    fetch("/api/rlx/clips", { cache: "no-store" }).then(async (response) => {
      if (!response.ok) throw new Error("Dance clip catalog unavailable.");
      const data = await response.json();
      if (active) setDanceClips(data.clips);
    }).catch((error) => { if (active) setClipError(String(error)); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    let active = true;
    fetchSavedRuns().then((runs) => {
      if (active) { setSavedRuns(runs); setCatalogError(null); }
    }).catch((error) => { if (active) setCatalogError(String(error)); });
    return () => { active = false; };
  }, [job.phase]);

  useEffect(() => {
    if (!viewerExpanded) return;
    const previousOverflow = document.documentElement.style.overflow;
    document.documentElement.style.overflow = "hidden";
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setViewerExpanded(false);
    };
    window.addEventListener("keydown", close);
    return () => {
      document.documentElement.style.overflow = previousOverflow;
      window.removeEventListener("keydown", close);
    };
  }, [viewerExpanded]);

  useEffect(() => {
    const dialog = guidanceDialogRef.current;
    if (!dialog) return;
    if (guidanceExperiment && !dialog.open) {
      dialog.showModal();
    } else if (!guidanceExperiment && dialog.open) {
      dialog.close();
    }
  }, [guidanceExperiment]);

  useEffect(() => {
    const dialog = rewardHistoryDialogRef.current;
    if (!dialog) return;
    if (rewardHistoryOpen && !dialog.open) {
      dialog.showModal();
    } else if (!rewardHistoryOpen && dialog.open) {
      dialog.close();
    }
  }, [rewardHistoryOpen]);

  function handleViewerFrame(nextFrame: Frame | null) {
    setFrame(nextFrame);
    const training = nextFrame?.training;
    if (!training) return;
    const step =
      training.progress.overallSteps ?? training.progress.steps ?? 0;
    const reward = training.progress.ep_rew;
    setLabRewardHistory((history) => {
      const existingHistory =
        labRunRef.current === training.runName ? history : [];
      labRunRef.current = training.runName;
      if (!Number.isFinite(reward) || step <= 0) return existingHistory;
      const point = { step, reward: reward as number };
      const previous = existingHistory.at(-1);
      if (previous?.step === point.step && previous.reward === point.reward) {
        return existingHistory;
      }
      if (previous?.step === point.step) {
        return [...existingHistory.slice(0, -1), point];
      }
      return [...existingHistory, point].slice(-240);
    });
  }

  useEffect(() => {
    if (IS_STATIC_EXPORT) return;
    let active = true;
    async function poll() {
      try {
        const response = await fetch(
          `/api/rlx?experiment=${recipe.experimentId}&run=${encodeURIComponent(recipe.runName)}`,
          { cache: "no-store" }
        );
        const data = (await response.json()) as JobState & { error?: string };
        if (active && response.ok) setJob(data);
      } catch {
        if (active) setNotice(t("The local Studio API is unavailable.", "本地 Studio API 不可用。"));
      }
    }
    poll();
    const id = window.setInterval(poll, job.phase === "running" ? 1000 : 2500);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, [recipe.experimentId, recipe.runName, job.phase, t]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const liveTraining = frame?.training;
  const labSteps = liveTraining?.progress.overallSteps ?? liveTraining?.progress.steps ?? 0;
  const labTotal = liveTraining?.progress.overallTotal ?? liveTraining?.progress.total ?? 0;

  const labTrainingActive =
    selectedExperimentId === "dance" && liveTraining?.status === "training";
  const rlxTrainingActive = job.phase === "running" && job.operation === "train";
  const useRlxTelemetry =
    rlxTrainingActive || (!labTrainingActive && job.rewardHistory.length > 0);
  const chartPoints = useRlxTelemetry
    ? job.rewardHistory
    : selectedExperimentId === "dance"
      ? labRewardHistory
      : [];
  const trainingSteps = useRlxTelemetry
    ? job.trainingSteps
    : selectedExperimentId === "dance"
      ? labSteps
      : 0;
  const trainingTotal = useRlxTelemetry
    ? job.trainingTotal
    : selectedExperimentId === "dance"
      ? labTotal
      : 0;
  const progress = trainingTotal > 0
    ? Math.min(100, (trainingSteps / trainingTotal) * 100)
    : 0;
  const chart = rewardChart(chartPoints, trainingTotal);
  const hasTrainingTelemetry =
    rlxTrainingActive || labTrainingActive || chartPoints.length > 0;
  const rewardIsNormalized = useRlxTelemetry && job.normalizeRewards;
  const rewardHistoryLabel = rewardIsNormalized
    ? t("Normalized rollout reward history", "归一化 rollout 奖励历史")
    : t("Rollout reward history", "Rollout 奖励历史");
  const evaluation = job.evaluation;
  const evidence = skillEvidence(evaluation, selectedExperimentId);
  const rolloutRanges = selectedExperimentId === "backflip"
    ? rolloutMetricRanges(evaluation, ["assist_force_n", "assist_torque_nm"])
    : [];
  const recipeMetric = recipeMetricNumber(evaluation, selectedExperiment);
  const verdict = evaluationVerdict(evaluation, selectedExperimentId);
  const evalPassed = verdict.pipelinePassed;
  const swingMeanSpan =
    selectedExperimentId === "swing"
      ? recipeMetricSummaryNumber(evaluation, selectedExperiment, "mean")
      : null;
  const swingBestSpan =
    selectedExperimentId === "swing"
      ? recipeMetricSummaryNumber(evaluation, selectedExperiment, "max")
      : null;
  const swingDiscoveryPassed =
    verdict.scope === "skill" && evalPassed &&
    ((swingMeanSpan ?? 0) >= 10 || (swingBestSpan ?? 0) >= 20);
  const swingConsolidationPassed =
    verdict.scope === "skill" && evalPassed &&
    ((swingMeanSpan ?? 0) >= 30 || (swingBestSpan ?? 0) >= 60);
  const swingTargetPassed = selectedExperimentId === "swing" && verdict.taskPassed;
  const taskPassed = verdict.taskPassed;
  const evaluationAndRenderPassed = taskPassed && job.renderVerified;
  const sourceReady = selectedExperimentId === "drawing"
    ? job.artifacts.checkpoint && job.artifacts.metadata && job.artifacts.onnx
    : job.artifacts.checkpoint || job.artifacts.onnx;
  const operationTime = elapsed(job.startedAt, job.finishedAt, now);
  const operationLabel = job.operation
    ? t(
        `${job.operation.charAt(0).toUpperCase()}${job.operation.slice(1)} time`,
        `${operationDisplay(job.operation, t)}用时`
      )
    : t("Operation time", "操作用时");
  const stage = !sourceReady
    ? 0
    : !job.artifacts.evaluation
      ? 1
      : !job.artifacts.renderVideo || !visualReviewed
        ? 2
        : 3;
  const deployReady =
    job.artifacts.onnx && taskPassed && job.renderVerified && job.artifacts.renderSheet && visualReviewed;
  const reward = chartPoints.at(-1)?.reward ?? null;
  const activeJob = receivedJob.activeJob;
  const anotherRunActive = Boolean(
    activeJob &&
      (activeJob.experimentId !== recipe.experimentId ||
        activeJob.runName !== recipe.runName)
  );
  const trainButtonLabel =
    pendingAction === "train"
      ? t("Starting RLX...", "正在启动 RLX...")
      : job.phase === "running" && job.operation === "train"
        ? t("RLX training", "RLX 训练中")
        : anotherRunActive
          ? t("RLX busy", "RLX 忙碌")
          : t("Start RLX", "启动 RLX");
  const recipeActionStatus = actionError ?? (
    pendingAction === "train"
      ? t(
          `Sending ${recipe.profile === "smoke" ? "pipeline smoke" : "full training"} request for ${recipe.runName}...`,
          `正在为 ${recipe.runName} 发送${recipe.profile === "smoke" ? "流水线冒烟" : "完整训练"}请求...`
        )
      : anotherRunActive && activeJob
        ? t(
            `RLX is already running ${activeJob.operation} for ${activeJob.experimentId}/${activeJob.runName}. Only one local RLX job can run at a time.`,
            `RLX 正在为 ${activeJob.experimentId}/${activeJob.runName} 执行${operationDisplay(activeJob.operation, t)}。本地一次只能运行一个 RLX 作业。`
          )
        : job.phase === "running" && job.operation === "train"
          ? t(
              `Training ${job.runName}. Reward history will update after each PPO rollout.`,
              `正在训练 ${job.runName}。每次 PPO rollout 后会更新奖励历史。`
            )
          : job.phase === "succeeded" && job.operation === "train"
            ? t(
                `Training completed for ${job.runName} with ${job.rewardHistory.length} reward sample${job.rewardHistory.length === 1 ? "" : "s"}.`,
                `${job.runName} 训练完成，共 ${job.rewardHistory.length} 个奖励样本。`
              )
          : job.phase === "failed" && job.operation === "train"
            ? t(
                `Training failed for ${job.runName}. Open View logs for the exact error.`,
                `${job.runName} 训练失败。打开“查看日志”以查看具体错误。`
              )
            : job.phase === "cancelled" && job.operation === "train"
              ? t(`Training was stopped for ${job.runName}.`, `${job.runName} 的训练已停止。`)
            : notice);
  const rewardHistoryText = [
    `step\t${rewardIsNormalized ? "normalized_rollout_reward" : "rollout_reward"}`,
    ...chartPoints.map((point) => `${point.step}\t${point.reward}`),
  ].join("\n");
  const activePolicies = policies.filter((policy) => policy.group === "runs").length;
  const workflowSteps = STEPS.map(([number, label, detail]) => [
    number,
    label === "Train"
      ? t(label, "训练")
      : label === "Evaluate"
        ? t(label, "评估")
        : label === "Render"
          ? t(label, "渲染")
          : t(label, "产物"),
    detail === "Learn in simulation"
      ? t(detail, "在仿真中学习")
      : detail === "Measure the policy"
        ? t(detail, "测量策略表现")
        : detail === "Review the motion"
          ? t(detail, "检查动作")
          : t(detail, "导出策略"),
  ] as const);

  const recipeCommand = `POST /api/rlx\n${JSON.stringify({ action: "train", recipe }, null, 2)}`;

  async function runAction(action: Operation | "cancel", recipeOverride?: Recipe) {
    setActionError(null);
    setBusy(true);
    setPendingAction(action);
    let activeRecipe = recipeOverride ?? recipe;
    const actionLabel =
      action === "train"
        ? t("Training", "训练")
        : action === "eval"
          ? t("Evaluation", "评估")
          : action === "render"
            ? t("Rendering", "渲染")
            : action === "export"
              ? t("Export", "导出")
              : t("Cancellation", "取消");
    setNotice(
      action === "cancel"
        ? t("Stopping the active RLX job...", "正在停止活动的 RLX 作业...")
        : t(
            `Starting ${actionLabel.toLowerCase()} for ${activeRecipe.runName}...`,
            `正在为 ${activeRecipe.runName} 启动${actionLabel}...`
          )
    );
    try {
      const response = await fetch("/api/rlx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, recipe: activeRecipe }),
      });
      const data = (await response.json()) as { error?: string; recipe?: Recipe };
      if (!response.ok) throw new Error(data.error || t("The action could not start.", "无法启动该操作。"));
      if (data.recipe) {
        activeRecipe = data.recipe;
        setRecipe(activeRecipe);
      }
      if (action !== "cancel") {
        setVisualReviewed(false);
        setJob((current) => ({
          ...current,
          phase: "running",
          operation: action,
          activeJob: {
            operation: action,
            experimentId: activeRecipe.experimentId,
            runName: activeRecipe.runName,
            startedAt: new Date().toISOString(),
          },
          experimentId: activeRecipe.experimentId,
          runName: activeRecipe.runName,
          startedAt: new Date().toISOString(),
          finishedAt: null,
          exitCode: null,
          logs: [
            `[studio] ${action} request accepted`,
            `[studio] experiment: ${activeRecipe.experimentId}`,
            `[studio] run: ${activeRecipe.runName}`,
          ],
          ...(action === "train"
            ? {
                rewardHistory: [],
                trainingSteps: 0,
                trainingTotal: activeRecipe.totalTimesteps,
              }
            : {}),
        }));
      }
      setNotice(
        action === "cancel"
          ? t("Cancellation requested.", "已请求取消。")
          : t(`${actionLabel} started.`, `${actionLabel}已启动。`)
      );
      try {
        const status = await fetch(
          `/api/rlx?experiment=${activeRecipe.experimentId}&run=${encodeURIComponent(activeRecipe.runName)}`,
          { cache: "no-store" }
        );
        if (status.ok) setJob(await status.json());
      } catch {
        setNotice(t(`${actionLabel} started. Status will refresh automatically.`, `${actionLabel}已启动，状态将自动刷新。`));
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : t("The action failed.", "操作失败。");
      setActionError(message);
      setNotice(message);
    } finally {
      setPendingAction(null);
      setBusy(false);
    }
  }

  function selectExperiment(experimentId: ExperimentId) {
    setActionError(null);
    setSelectedExperimentId(experimentId);
    setRecipe(recipeDefaults(experimentId));
    setVisualReviewed(false);
    setNotice(null);
  }

  async function loadSavedRun(experimentId: ExperimentId, runName: string) {
    setActionError(null);
    setBusy(true);
    try {
      const href = IS_STATIC_EXPORT
        ? staticStudioArtifactUrl(experimentId, runName, "evaluation")
        : `/api/rlx?experiment=${experimentId}&run=${encodeURIComponent(runName)}`;
      const response = await fetch(href, { cache: "no-store" });
      if (!response.ok) throw new Error(t("Cannot load saved run.", "无法加载已保存的运行。"));
      const data = await response.json() as JobState;
      setSelectedExperimentId(experimentId);
      setRecipe(data.savedRecipe ?? {
        ...recipeDefaults(experimentId, data.drawingTool ?? "brush"),
        runName,
      });
      setJob(data);
      setVisualReviewed(false);
      setNotice(data.savedRecipe
        ? t("Loaded the saved evaluation recipe. Changes affect future jobs only.", "已加载保存的评估配方。修改只影响后续作业。")
        : t("Artifacts loaded; historical recipe unavailable. Current controls are defaults, not training provenance.", "已加载产物，但历史配方不可用。当前控件显示默认值，并非训练溯源。"));
      document.getElementById("evaluation")?.scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      setNotice(String(error));
    } finally {
      setBusy(false);
    }
  }

  function previewSelectedExperiment() {
    if (selectedExperimentId === "dance") {
      setSessionTab("animate");
      document
        .getElementById("training")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
      window.requestAnimationFrame(() => {
        window.dispatchEvent(new Event("microduck:play-choreography"));
      });
      return;
    }
    const previewRun = latestVerifiedRun(savedRuns, selectedExperimentId);
    if (previewRun) {
      window.open(
        IS_STATIC_EXPORT
          ? staticStudioArtifactUrl(selectedExperimentId, previewRun.runName, "video")
          : `/api/rlx/artifact?experiment=${selectedExperimentId}&run=${encodeURIComponent(previewRun.runName)}&kind=video&inline=1`,
        "_blank",
        "noopener,noreferrer"
      );
      return;
    }
    if (selectedExperiment.video) {
      window.open(publicAssetUrl(selectedExperiment.video), "_blank", "noopener,noreferrer");
      return;
    }
    setNotice(t(
      `${selectedExperiment.title} has no verified rollout yet. ${selectedExperiment.readiness}`,
      `${selectedExperimentText.title} 尚无已验证的回放。${selectedExperimentText.readiness}`
    ));
    document.getElementById("experiments")?.scrollIntoView({ behavior: "smooth" });
  }

  function recipeForProfile(profile: Recipe["profile"]): Recipe {
    const reserved = savedRuns.filter((run) => run.experimentId === recipe.experimentId).map((run) => run.runName);
    if (job.artifacts.checkpoint) reserved.push(recipe.runName);
    return {
      ...recipe,
      profile,
      runName: availableProfileRunName(recipe.runName, profile, reserved),
      ...(profile === "full" ? fullHorizon(
        recipe.experimentId,
        selectedDanceDuration,
        recipe.drawingTool ?? "brush"
      ) : {
        numSteps: selectedDrawingTool?.fullNumSteps ?? 2,
        numMinibatches: selectedDrawingTool?.fullNumMinibatches ?? 1,
        maxEpisodeS: selectedDrawingTool?.maxEpisodeSeconds ?? 1,
        evalSteps: 4,
        evalEpisodes: selectedDrawingTool?.id === "brush"
          ? selectedDrawingTool.evalEpisodes
          : 1,
      }),
      totalTimesteps: profile === "full"
        ? selectedDrawingTool?.fullTimesteps ?? selectedExperiment.fullTimesteps
        : selectedDrawingTool?.smokeTimesteps ?? 4,
      numEnvs: profile === "full" || recipe.experimentId === "drawing"
        ? selectedExperiment.fullEnvs
        : 2,
      resumeFromCheckpoint: false,
      freezeObservationNormalization: profile === "full" &&
        (recipe.experimentId === "backflip" || recipe.experimentId === "bridge"),
      domainRand: profile === "full" ? selectedExperiment.fullEnvironment.domainRand : false,
      obsNoise: profile === "full" ? selectedExperiment.fullEnvironment.obsNoise : false,
      actionDelay: profile === "full" ? selectedExperiment.fullEnvironment.actionDelay : false,
      randomYaw: profile === "full" ? selectedExperiment.fullEnvironment.randomYaw : false,
      bridgeCurriculum: profile === "full" && recipe.experimentId === "bridge",
      ...(profile === "full" && recipe.experimentId === "backflip" ? {
        seed: selectedExperiment.seed,
        initialStd: selectedExperiment.initialStd,
        normalizeRewards: selectedExperiment.normalizeRewards,
        learningRate: selectedExperiment.ppo.learningRate,
        gamma: selectedExperiment.ppo.gamma,
        clipCoefficient: selectedExperiment.ppo.clipCoefficient,
        updateEpochs: selectedExperiment.ppo.updateEpochs,
        entropyCoefficient: selectedExperiment.ppo.entropyCoefficient,
        maxGradNorm: selectedExperiment.ppo.maxGradNorm,
        rewardWeights: defaultRewardWeights("backflip"),
      } : {}),
    };
  }

  function selectTrainingProfile(profile: Recipe["profile"]) {
    const next = recipeForProfile(profile);
    setRecipe(next);
    setActionError(null);
    setNotice(t(
      `${profile === "full" ? "Full training" : "Pipeline smoke"} selected for ${next.runName}. ${next.runName !== recipe.runName ? `Saved run ${recipe.runName} is preserved. ` : ""}Click Start RLX to begin.`,
      `已为 ${next.runName} 选择${profile === "full" ? "完整训练" : "流水线冒烟"}。${next.runName !== recipe.runName ? `已保留运行 ${recipe.runName}。` : ""}点击“启动 RLX”开始。`
    ));
  }

  function prepareFreshRun() {
    const reserved = savedRuns.filter((run) => run.experimentId === recipe.experimentId).map((run) => run.runName);
    const runName = availableProfileRunName(recipe.runName, recipe.profile, [...reserved, recipe.runName]);
    setRecipe({ ...recipe, runName, resumeFromCheckpoint: false });
    setActionError(null);
    setNotice(t(
      `New run ${runName} is ready. Previous checkpoints are preserved. Click Start RLX to begin.`,
      `新运行 ${runName} 已就绪。旧检查点已保留。点击“启动 RLX”开始。`
    ));
  }

  function selectDrawingTool(drawingTool: DrawingTool) {
    const definition = getDrawingTool(drawingTool);
    const base = recipeDefaults("drawing", drawingTool);
    setRecipe({
      ...base,
      runName: recipe.runName,
      profile: recipe.profile,
      ...(recipe.profile === "full"
        ? {
            ...fullHorizon("drawing", undefined, drawingTool),
            totalTimesteps: definition.fullTimesteps,
          }
        : {}),
    });
    setVisualReviewed(false);
    setActionError(null);
    setNotice(t(
      `${definition.label} selected. Future jobs will use ${definition.contractVersion}; saved artifacts and verdicts are unchanged.`,
      `已选择${drawingToolLabel(definition.id, t)}。后续作业将使用 ${definition.contractVersion}；已保存的产物和判定不变。`
    ));
  }

  function trainRecommendedRecipe() {
    const fullRecipe = {
      ...recipeForProfile("full"),
      totalTimesteps: Math.max(
        recipe.totalTimesteps,
        selectedDrawingTool?.fullTimesteps ?? selectedExperiment.fullTimesteps
      ),
      numEnvs: Math.max(recipe.numEnvs, selectedExperiment.fullEnvs),
    };
    setRecipe(fullRecipe);
    void runAction("train", fullRecipe);
  }

  function applySwingDiscovery() {
    setAdvanced(true);
    setRecipe((current) => ({
      ...current,
      runName: "swing-curriculum-01",
      profile: "full",
      ...fullHorizon("swing"),
      totalTimesteps: 250_000,
      numEnvs: 16,
      seed: 2,
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
      swingInitialAngleDeg: 12,
      swingInitialRateRadS: 0.35,
      swingPlanarActions: true,
      resumeFromCheckpoint: false,
      rewardWeights: defaultRewardWeights("swing"),
    }));
    document
      .getElementById("recipe")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function applySwingConsolidation() {
    setAdvanced(true);
    setRecipe((current) => ({
      ...current,
      profile: "full",
      ...fullHorizon("swing"),
      totalTimesteps: 500_000,
      numEnvs: 16,
      seed: 2,
      learningRate: 0.0001,
      gamma: 0.995,
      clipCoefficient: 0.1,
      updateEpochs: 3,
      entropyCoefficient: 0.002,
      maxGradNorm: 1,
      domainRand: true,
      obsNoise: true,
      actionDelay: true,
      randomYaw: false,
      swingInitialAngleDeg: 4,
      swingInitialRateRadS: 0.12,
      swingPlanarActions: true,
      resumeFromCheckpoint: true,
      rewardWeights: defaultRewardWeights("swing"),
    }));
    document
      .getElementById("recipe")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function startLabTraining(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      const response = await fetch(`${LAB_HTTP}/teach`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: selectedExperiment.goal,
          steps: recipe.profile === "smoke" ? 100_000 : recipe.totalTimesteps,
        }),
      });
      if (!response.ok) throw new Error(`duck-lab returned ${response.status}`);
      setNotice(t("Duck Lab training request accepted.", "Duck Lab 已接受训练请求。"));
    } catch {
      setNotice(
        t(
          `Duck Lab is offline. Use Run RLX training for the local ${selectedExperiment.shortTitle.toLowerCase()} pipeline.`,
          `Duck Lab 离线。请使用“运行 RLX 训练”执行本地${selectedExperimentText.shortTitle}流水线。`
        )
      );
    } finally {
      setBusy(false);
    }
  }

  function artifactHref(kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video") {
    if (IS_STATIC_EXPORT && (kind === "sheet" || kind === "video")) {
      return staticStudioArtifactUrl(recipe.experimentId, recipe.runName, kind);
    }
    return `/api/rlx/artifact?experiment=${recipe.experimentId}&run=${encodeURIComponent(recipe.runName)}&kind=${kind}`;
  }

  async function copyRewardHistory() {
    try {
      await navigator.clipboard.writeText(rewardHistoryText);
    } catch {
      const textArea = document.createElement("textarea");
      textArea.value = rewardHistoryText;
      textArea.style.position = "fixed";
      textArea.style.opacity = "0";
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      textArea.remove();
    }
    setRewardHistoryCopied(true);
    window.setTimeout(() => setRewardHistoryCopied(false), 1800);
  }

  return (
    <div className={styles.app}>
      <aside className={styles.sidebar}>
        <Link className={styles.brand} href="/" aria-label={t("Microduck Studio home", "Microduck Studio 首页")}>
          <span className={styles.brandMark}><Icon>⌁</Icon></span>
          <span>microduck<small>STUDIO</small></span>
        </Link>
        <div className={styles.workspace}>
          <span className={styles.avatar}>ML</span>
          <span>{t("Local workspace", "本地工作区")}<small>{t("Apple Silicon lab", "Apple Silicon 实验室")}</small></span>
          <span className={styles.chevron}>⌄</span>
        </div>
        <div className={styles.languageToggle}><LanguageToggle /></div>
        <p className={styles.navHeading}>{t("WORKSPACE", "工作区")}</p>
        <nav aria-label={t("Workspace", "工作区")}>
          <a className={styles.navItem} href="#overview" aria-label={t("Overview", "概览")}><Icon>▦</Icon><span>{t("Overview", "概览")}</span></a>
          <a className={styles.navItem} href="#recipe" aria-label={t("Projects", "项目")}><Icon>◇</Icon><span>{t("Projects", "项目")}</span><b>1</b></a>
          <a className={`${styles.navItem} ${styles.active}`} href="#experiments" aria-label={t("Experiments", "实验")}><Icon>♙</Icon><span>{t("Experiments", "实验")}</span><b>{EXPERIMENTS.length}</b></a>
          <Link className={styles.navItem} href="/arm" aria-label={t("Arm Lab", "机械臂实验室")}><Icon>⌁</Icon><span>{t("Arm Lab", "机械臂实验室")}</span><b>6</b></Link>
          <Link className={styles.navItem} href="/arm#videos" aria-label={t("Arm videos", "机械臂视频")}><Icon>▷</Icon><span>{t("Arm videos", "机械臂视频")}</span></Link>
          <a className={styles.navItem} href="#artifacts" aria-label={t("Policies", "策略")}><Icon>▱</Icon><span>{t("Policies", "策略")}</span><b>{activePolicies}</b></a>
          <a className={styles.navItem} href="#evaluation" aria-label={t("Evaluations", "评估")}><Icon>⌁</Icon><span>{t("Evaluations", "评估")}</span></a>
        </nav>
        <p className={styles.navHeading}>{t("CONTROL CENTER", "控制中心")}</p>
        <nav aria-label={t("Control center", "控制中心")}>
          <a className={styles.navItem} href="#deployment" aria-label={t("Robot fleet", "机器人群组")}><Icon>⌾</Icon><span>{t("Robot fleet", "机器人群组")}</span></a>
          <a className={styles.navItem} href="#deployment" aria-label={t("Deployments", "部署")}><Icon>♢</Icon><span>{t("Deployments", "部署")}</span></a>
          <a className={styles.navItem} href="#system" aria-label={t("Settings", "设置")}><Icon>⚙</Icon><span>{t("Settings", "设置")}</span></a>
          <a className={styles.navItem} href="mailto:bochuxt7@gmail.com" aria-label={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}><Icon>✉</Icon><span>{t("Contact", "联系")}</span></a>
        </nav>
        <div className={styles.sidebarBottom}>
          <a
            className={styles.contactLink}
            href="mailto:bochuxt7@gmail.com"
            aria-label={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}
          >
            <Icon>✉</Icon>
            <span>{t("Contact", "联系")}<small>bochuxt7@gmail.com</small></span>
          </a>
          <div className={styles.localStatus}>
            <strong><i className={connected ? styles.onlineDot : styles.offlineDot} />{connected ? t("Duck Lab connected", "Duck Lab 已连接") : t("Local studio ready", "本地 Studio 已就绪")}</strong>
            <p>{connected ? t("Live frames on :8788", "实时帧来自 :8788") : t("RLX jobs run on this Mac", "RLX 作业在本机运行")}</p>
          </div>
          <div className={styles.profile}>
            <span className={styles.profileAvatar}>GH</span>
            <span>{t("Local operator", "本地操作员")}<small>{t("No cloud required", "无需云端")}</small></span>
          </div>
        </div>
      </aside>

      <main className={styles.main} id="overview">
        <header className={styles.topbar}>
          <div className={styles.breadcrumbs}>
            <span>{t("Workspace", "工作区")}</span><b>/</b><span>{t("Experiments", "实验")}</span><b>/</b><strong>{selectedExperimentText.shortTitle}</strong>
          </div>
          <div className={styles.topActions}>
            <div className={styles.mobileTopActions}>
              <Link
                className={styles.mobileArmVideos}
                href="/arm#videos"
                aria-label={t("Arm videos", "机械臂视频")}
              >
                <Icon>▷</Icon>
                <span>{t("Arm videos", "机械臂视频")}</span>
              </Link>
              <a
                className={styles.mobileContact}
                href="mailto:bochuxt7@gmail.com"
                title={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}
                aria-label={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}
              >
                <Icon>✉</Icon>
                <span>{t("Contact", "联系")}</span>
              </a>
              <div className={styles.mobileLanguageToggle}>
                <LanguageToggle />
              </div>
            </div>
            <StatusPill tone={connected ? "live" : "muted"}>
              {connected ? t("LIVE LAB", "实验室在线") : t("LOCAL MODE", "本地模式")}
            </StatusPill>
            <button
              className={styles.iconButton}
              onClick={() => setViewerExpanded(true)}
              title={t("Expand the simulation workspace", "展开仿真工作区")}
              aria-label={t("Expand the simulation workspace", "展开仿真工作区")}
            >
              ⛶
            </button>
            <span className={styles.avatar}>GH</span>
          </div>
        </header>

        <section className={styles.heading}>
          <div>
            <p className={styles.eyebrow}>{t("SMALL ROBOT. COMPLETE POLICY PIPELINE.", "小型机器人，完整策略流水线。")}</p>
            <h1>{t("Learning in motion.", "在运动中学习。")}</h1>
            <p>{selectedExperimentId === "backflip"
              ? t("spotter-assisted launch → PPO landing → pretrained stand-policy handoff. Export the learned landing stage for controller integration.", "辅助起跳 → PPO 落地 → 预训练站立策略接管。导出学到的落地阶段，用于控制器集成。")
              : t("Train a behavior. Prove it in simulation. Export what the robot can run.", "训练一种行为，在仿真中验证，再导出机器人可运行的策略。")}</p>
          </div>
          <div className={styles.headingActions}>
            {job.artifacts.metadata && (
              <a className={styles.button} href={artifactHref("metadata")}>
                <Icon>↓</Icon> {t("Metadata", "元数据")}
              </a>
            )}
            <button
              className={`${styles.button} ${styles.primary}`}
              onClick={() => {
                const suffix = new Date().toISOString().slice(11, 19).replaceAll(":", "");
                setRecipe((current) => ({
                  ...current,
                  runName: `${selectedExperiment.id}-${suffix}`,
                }));
                setVisualReviewed(false);
                document.getElementById("recipe")?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              <Icon>＋</Icon> {t("New experiment", "新建实验")}
            </button>
          </div>
        </section>

        <section className={styles.experimentCatalog} id="experiments" aria-labelledby="experiment-catalog-title">
          <div className={styles.catalogHead}>
            <div>
              <p className={styles.eyebrow}>{t("MACOS RLX RECIPES", "MACOS RLX 配方")}</p>
              <h2 id="experiment-catalog-title">{t("Choose what the duck should learn", "选择小鸭要学习的技能")}</h2>
            </div>
            <span>{t("Trained rollouts and diagnostic pilots · simulation only.", "训练回放与诊断试验 · 仅限仿真。")}</span>
          </div>
          <div className={styles.experimentTable} role="radiogroup" aria-label={t("Training experiments", "训练实验")}>
            <div className={styles.experimentHeader} aria-hidden="true">
              <span>{t("Experiment", "实验")}</span><span>{t("Preview", "预览")}</span><span>{t("Result and artifacts", "结果与产物")}</span>
            </div>
            {EXPERIMENTS.map((experiment) => {
              const experimentText = experimentDisplay(experiment, t);
              const selected = experiment.id === selectedExperimentId;
              const verifiedRun = latestVerifiedRun(savedRuns, experiment.id);
              const diagnosticRun = latestDiagnosticRun(savedRuns, experiment.id);
              const previewRun = verifiedRun ?? diagnosticRun;
              const balanceOnlyPreview = experiment.id === "basketball" &&
                previewRun?.balanceOnly === true;
              const failedDiagnosticPreview = Boolean(diagnosticRun && previewRun === diagnosticRun);
              const runQuery = previewRun ? new URLSearchParams({ experiment: experiment.id, run: previewRun.runName }).toString() : null;
              const videoHref = previewRun
                ? IS_STATIC_EXPORT
                  ? staticStudioArtifactUrl(experiment.id, previewRun.runName, "video")
                  : `/api/rlx/artifact?${runQuery}&kind=video&inline=1&v=${encodeURIComponent(previewRun.renderEvidenceId!)}`
                : null;
              const sheetHref = previewRun
                ? IS_STATIC_EXPORT
                  ? staticStudioArtifactUrl(experiment.id, previewRun.runName, "sheet")
                  : `/api/rlx/artifact?${runQuery}&kind=sheet&v=${encodeURIComponent(previewRun.renderEvidenceId!)}`
                : null;
              const evaluationHref = previewRun
                ? IS_STATIC_EXPORT
                  ? staticStudioArtifactUrl(experiment.id, previewRun.runName, "evaluation")
                  : `/api/rlx?${runQuery}`
                : null;
              return (
                <div
                  role="radio"
                  aria-checked={selected}
                  tabIndex={0}
                  className={`${styles.experimentRow} ${selected ? styles.selectedExperiment : ""}`}
                  key={experiment.id}
                  onClick={() => selectExperiment(experiment.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      selectExperiment(experiment.id);
                    }
                  }}
                >
                  <span className={styles.experimentIdentity}>
                    <span className={styles.recipeCardTopline}>
                      <i>{selected ? t("SELECTED", "已选择") : t("RLX RECIPE", "RLX 配方")}</i>
                      <button
                        type="button"
                        className={styles.recipeInfoButton}
                        aria-label={t(`Show ${experiment.title} guidance`, `显示${experimentText.title}指南`)}
                        title={t(`How to train ${experiment.title.toLowerCase()}`, `如何训练${experimentText.title}`)}
                        onClick={(event) => {
                          event.stopPropagation();
                          setGuidanceExperimentId(experiment.id);
                        }}
                        onKeyDown={(event) => event.stopPropagation()}
                      >
                        i
                      </button>
                    </span>
                    <strong>{experimentText.title}</strong>
                    <small>{experimentText.goal}</small>
                  </span>
                  <span className={styles.previewFrame}>
                    {previewRun && videoHref ? (
                      <>
                        <video
                          key={videoHref}
                          src={videoHref}
                          aria-label={t(`${experiment.shortTitle} trained rollout: ${previewRun.runName}`, `${experimentText.shortTitle}训练回放：${previewRun.runName}`)}
                          autoPlay
                          muted
                          loop
                          playsInline
                        />
                        <picture className={styles.previewStill}><img src={sheetHref ?? ""} alt={t(`${experiment.shortTitle} trained rollout contact sheet`, `${experimentText.shortTitle}训练回放帧预览图`)} /></picture>
                      </>
                    ) : (
                      <span className={styles.previewEmpty}>{t("No verified rollout yet", "尚无已验证回放")}</span>
                    )}
                    {previewRun && <b>{`${previewRun.drawingTool ? `${drawingToolLabel(previewRun.drawingTool, t)} · ` : ""}${balanceOnlyPreview
                      ? t("BALANCE ONLY · STEERING NOT PASSED", "仅平衡 · 转向未通过")
                        : failedDiagnosticPreview
                          ? experiment.id === "drawing"
                            ? t("DIAGNOSTIC · DRAWING NOT ACCEPTED", "诊断 · 绘画未通过")
                            : t("TRAINING PILOT · CROSSING FAILED", "训练试验 · 过桥失败")
                          : t("TRAINED · PASSED", "已训练 · 已通过")}`}</b>}
                  </span>
                  <span className={styles.experimentResult}>
                    <strong>{previewRun
                      ? balanceOnlyPreview
                        ? t("60-second balance preview", "60 秒平衡预览")
                        : failedDiagnosticPreview
                          ? experiment.id === "drawing"
                            ? t("Latest diagnostic drawing pilot", "最新诊断绘画试验")
                            : t("Latest diagnostic training pilot", "最新诊断训练试验")
                          : t("Latest successful trained policy", "最新成功训练策略")
                      : experimentText.readiness}</strong>
                    {previewRun && <small className={styles.previewRunName}>{previewRun.runName}</small>}
                    <small>{previewRun
                      ? balanceOnlyPreview
                        ? t("Matched balance evidence and rollout · steering remains failed", "平衡证据与回放匹配 · 转向仍未通过")
                        : failedDiagnosticPreview
                          ? experiment.id === "drawing"
                            ? t("Source-bound evaluation and rollout · drawing acceptance failed", "评估与回放已绑定源文件 · 绘画验收失败")
                            : t("Source-bound evaluation and rollout · bridge crossing failed", "评估与回放已绑定源文件 · 过桥失败")
                          : t("Matched evaluation and rollout · simulation only", "评估与回放匹配 · 仅限仿真")
                      : experimentText.task}</small>
                    <span className={styles.referenceLinks}>
                      {previewRun && videoHref && (
                        <>
                          <a
                            href={videoHref}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(event) => event.stopPropagation()}
                          >
                            {t("Full video ↗", "完整视频 ↗")}
                          </a>
                          <a
                            href={evaluationHref ?? undefined}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(event) => event.stopPropagation()}
                          >
                            {t("Evaluation ↗", "评估 ↗")}
                          </a>
                        </>
                      )}
                      <a
                        href={`${publicAssetUrl("/guides/microduck-studio-experiments-guide.pdf")}#${experiment.guideAnchor}`}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(event) => event.stopPropagation()}
                      >
                        {t("Guide ↗", "指南 ↗")}
                      </a>
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        </section>

        <section className={styles.savedEvidence} aria-label={t("Saved scenario evidence", "已保存的场景证据")}>
          <h2>{t("Review trained policies", "检查已训练策略")}</h2>
          <p>{t("Current checkpoint-bound evaluations, not preview animations. Nominal MuJoCo simulation only; not hardware certification.", "这里展示与当前检查点绑定的评估，而非预览动画。仅为标称 MuJoCo 仿真，不代表硬件认证。")}</p>
          {catalogError && <p role="alert">{catalogError}</p>}
          <div className={styles.evidenceCards}>
            {EXPERIMENTS.map((experiment) => {
              const experimentText = experimentDisplay(experiment, t);
              const run = latestReviewRun(savedRuns, experiment.id);
              return <article key={experiment.id}>
                <strong>{experimentText.title}</strong>
                <span>{run
                  ? run.taskPassed
                    ? t("Skill passed · saved evaluation", "技能通过 · 已保存评估")
                    : run.balanceOnly
                      ? t("Balance only · steering not passed", "仅平衡 · 转向未通过")
                      : run.skillAssessed
                        ? t("Skill failed", "技能失败")
                        : t("Skill not assessed", "技能未评估")
                  : experimentText.readiness}</span>
                <small>{run?.runName ?? t("Train and evaluate to create evidence", "训练并评估以生成证据")}</small>
                <button className={styles.button} disabled={!run || busy} onClick={() => run && void loadSavedRun(experiment.id, run.runName)}>{t(`Review ${experiment.shortTitle}`, `检查${experimentText.shortTitle}`)}</button>
              </article>;
            })}
          </div>
        </section>

        <section className={styles.steps} aria-label={t(`${selectedExperiment.title} workflow`, `${selectedExperimentText.title}工作流`)}>
          {workflowSteps.map(([number, label, detail], index) => (
            <a
              key={label}
              className={`${styles.step} ${index < stage ? styles.done : ""} ${index === stage ? styles.current : ""}`}
              href={index === 0 ? "#recipe" : index === 1 ? "#evaluation" : index === 2 ? "#evaluation" : "#artifacts"}
            >
              <span className={styles.stepNumber}>{index < stage ? "✓" : number}</span>
              <span><strong>{label}</strong><small>{detail}</small></span>
            </a>
          ))}
        </section>

        {notice && (
          <div className={styles.notice} role="status">
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} aria-label={t("Dismiss message", "关闭消息")}>×</button>
          </div>
        )}

        <section className={styles.coach} aria-labelledby="experiment-coach-title">
          <div className={styles.coachIntro}>
            <p className={styles.eyebrow}>{t("SELECTED RLX WORKFLOW", "已选择的 RLX 工作流")}</p>
            <h2 id="experiment-coach-title">{selectedExperimentText.title}</h2>
            <p>{selectedExperimentText.description}</p>
            <div className={styles.evidenceList}>
              {selectedExperimentText.evidence.map((item) => (
                <span key={item}>{item}</span>
              ))}
            </div>
            <div className={styles.coachActions}>
              <button className={styles.button} onClick={previewSelectedExperiment}>
                <Icon>▶</Icon> {t("Preview reference", "预览参考")}
              </button>
              {!sourceReady ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={trainRecommendedRecipe}
                  disabled={busy || job.phase === "running" || Boolean(activeJob)}
                >
                  <Icon>✦</Icon> {t(`Train ${selectedExperiment.shortTitle.toLowerCase()}`, `训练${selectedExperimentText.shortTitle}`)}
                </button>
              ) : !job.artifacts.evaluation ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("eval")}
                  disabled={busy || job.phase === "running" || Boolean(activeJob)}
                >
                  <Icon>✓</Icon> {t("Evaluate policy", "评估策略")}
                </button>
              ) : !job.artifacts.renderVideo ? (
                <button
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("render")}
                  disabled={busy || job.phase === "running" || Boolean(activeJob)}
                >
                  <Icon>◫</Icon> {t("Render rollout", "渲染回放")}
                </button>
              ) : (
                <a className={`${styles.button} ${styles.primary}`} href="#evaluation">
                  <Icon>▶</Icon> {t("Review result", "检查结果")}
                </a>
              )}
            </div>
          </div>
          <ol className={styles.coachSteps}>
            <li className={styles.coachDone}>
              <span>1</span>
              <div><strong>{t("Plan", "计划")}</strong><small>{selectedExperimentText.task}</small></div>
            </li>
            <li className={sourceReady ? styles.coachDone : styles.coachCurrent}>
              <span>2</span>
              <div><strong>{t("Train", "训练")}</strong><small>{sourceReady ? job.artifacts.checkpoint ? t("Checkpoint ready", "检查点已就绪") : t("ONNX policy ready", "ONNX 策略已就绪") : t(`${formatNumber(recipe.totalTimesteps)} planned steps · ${recipe.numEnvs} environments`, `计划 ${formatNumber(recipe.totalTimesteps)} 步 · ${recipe.numEnvs} 个环境`)}</small></div>
            </li>
            <li className={job.artifacts.evaluation ? styles.coachDone : sourceReady ? styles.coachCurrent : ""}>
              <span>3</span>
              <div><strong>{t("Evaluate", "评估")}</strong><small>{selectedExperimentText.metricLabel} · {t("finite actions", "有限动作值")}</small></div>
            </li>
            <li className={visualReviewed ? styles.coachDone : job.artifacts.renderVideo ? styles.coachCurrent : ""}>
              <span>4</span>
              <div><strong>{t("Review", "检查")}</strong><small>{t("Watch motion · compare against null controls", "观看动作 · 与空控制对比")}</small></div>
            </li>
          </ol>
        </section>

        <div className={styles.dashboard}>
          <section
            className={`${styles.panel} ${styles.simulationPanel} ${viewerExpanded ? styles.workspaceExpanded : ""}`}
            id="simulation"
          >
            {viewerExpanded && (
              <button
                className={styles.fullscreenClose}
                onClick={() => setViewerExpanded(false)}
                title={t("Restore Studio layout", "恢复 Studio 布局")}
                aria-label={t("Restore Studio layout", "恢复 Studio 布局")}
              >
                ×
              </button>
            )}
            <div className={styles.panelHead}>
              <div>
                <h2><Icon>◇</Icon>{t("Simulation workspace", "仿真工作区")}</h2>
                <p>{t("Policies, teaching, animation, recording, and live MuJoCo controls", "策略、教学、动画、录制和实时 MuJoCo 控制")}</p>
              </div>
              <div className={styles.panelTools}>
                <StatusPill tone={connected ? "live" : "muted"}>
                  {IS_STATIC_EXPORT
                    ? t("STATIC EVIDENCE", "静态证据")
                    : connected
                      ? t("LIVE MUJOCO", "MUJOCO 在线")
                      : t("WAITING FOR LAB", "等待实验室")}
                </StatusPill>
                <button
                  className={styles.iconButton}
                  onClick={() => setViewerExpanded((expanded) => !expanded)}
                  title={viewerExpanded ? t("Restore Studio layout", "恢复 Studio 布局") : t("Expand the simulation workspace", "展开仿真工作区")}
                  aria-label={viewerExpanded ? t("Restore Studio layout", "恢复 Studio 布局") : t("Expand the simulation workspace", "展开仿真工作区")}
                >
                  {viewerExpanded ? "×" : "⛶"}
                </button>
              </div>
            </div>
            <div className={styles.viewer}>
              <Viewer
                offline={IS_STATIC_EXPORT}
                layout="studio"
                showAnimationTools={false}
                showPolicyTools={false}
                showTeachTools={false}
                onClientReady={(client) => {
                  clientRef.current = client;
                }}
                onConnectionChange={setConnected}
                onFrame={handleViewerFrame}
              />
            </div>
            <div className={styles.viewerControls}>
              <button onClick={() => clientRef.current?.sendReset()} disabled={!connected} title={t("Reset every live simulation", "重置所有实时仿真")}>
                ↺
              </button>
              <span>{connected ? t(`${frame?.ducks.length ?? 0} live duck${frame?.ducks.length === 1 ? "" : "s"}`, `${frame?.ducks.length ?? 0} 只实时小鸭`) : t("Lab offline", "实验室离线")}</span>
              <span>{liveContract.observations} {t("observations", "维观测")}</span>
              <span>{liveContract.actions} {t("actions", "维动作")}</span>
              <span>{t("50 Hz control", "50 Hz 控制")}</span>
              <button
                className={styles.workspaceButton}
                onClick={() => setViewerExpanded((expanded) => !expanded)}
              >
                {viewerExpanded ? t("Restore layout", "恢复布局") : t("Expand workspace", "展开工作区")}
              </button>
            </div>
          </section>

          <section className={styles.panel} id="training">
            <div className={styles.sessionHead}>
              <div className={styles.sessionTabs} role="tablist" aria-label={t(`${selectedExperiment.title} workspace`, `${selectedExperimentText.title}工作区`)}>
                <button
                  role="tab"
                  aria-selected={sessionTab === "training"}
                  className={sessionTab === "training" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("training")}
                >
                  <Icon>♙</Icon> {t("Training", "训练")}
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "animate"}
                  className={sessionTab === "animate" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("animate")}
                >
                  <Icon>▶</Icon> {t("Animate", "动画")}
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "policies"}
                  className={sessionTab === "policies" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("policies")}
                >
                  <Icon>◇</Icon> {t("Policies", "策略")}
                </button>
                <button
                  role="tab"
                  aria-selected={sessionTab === "teach"}
                  className={sessionTab === "teach" ? styles.activeTab : ""}
                  onClick={() => setSessionTab("teach")}
                >
                  <Icon>✦</Icon> {t("Teach", "教学")}
                </button>
              </div>
            </div>
            <div
              className={styles.trainingBody}
              role="tabpanel"
              hidden={sessionTab !== "training"}
            >
              <div className={styles.experimentTop}>
                <div>
                  <h3>{recipe.runName}</h3>
                  <p>{selectedExperimentText.shortTitle}{selectedDrawingTool ? ` · ${selectedDrawingToolLabel}` : ""} · {selectedDrawingTool?.trainerLabel ?? "RLX PPO · MLX Metal"} · {t("Seed", "种子")} {recipe.seed}</p>
                </div>
                <StatusPill tone={hasTrainingTelemetry ? "live" : "muted"}>
                  {hasTrainingTelemetry ? t("LIVE DATA", "实时数据") : t("WAITING FOR DATA", "等待数据")}
                </StatusPill>
              </div>
              <div className={styles.metricGrid}>
                <div>
                  <span>{rewardIsNormalized ? t("Normalized training reward", "归一化训练奖励") : t("Training reward", "训练奖励")}</span>
                  <strong>{formatReward(reward)}</strong>
                  <small>
                    {reward == null
                      ? t("Waiting for the first rollout", "等待首次 rollout")
                      : rewardIsNormalized
                        ? t("Latest normalized rollout-buffer mean", "最新归一化 rollout 缓冲区均值")
                        : t("Latest rollout mean", "最新 rollout 均值")}
                  </small>
                </div>
                <div>
                  <span>{t("Evaluation score", "评估分数")}</span>
                  <strong>
                    {recipeMetric == null
                      ? t("Not measured", "未测量")
                      : `${recipeMetric.toFixed(2)}${selectedExperiment.metricSuffix}`}
                  </strong>
                  <small>
                    {evaluation
                      ? evalPassed
                      ? evaluationAndRenderPassed
                        ? t("Evaluator and render verified", "评估器与渲染均已验证")
                        : taskPassed
                          ? t("Evaluator criteria met; render pending", "已满足评估标准；等待渲染")
                        : t("Pipeline passed; skill not accepted", "流水线通过；技能未验收")
                        : t("Evaluation needs review", "评估需要检查")
                      : t("Run evaluation after export", "导出后运行评估")}
                  </small>
                </div>
                <div className={styles.timerMetric}>
                  <span>{operationLabel}</span>
                  <strong>{operationTime}</strong>
                  <small>
                    {job.phase === "running"
                      ? t(`${job.operation} in progress`, `${operationDisplay(job.operation, t)}进行中`)
                      : job.startedAt
                        ? t(`Last ${job.operation} ${job.phase}`, `上次${operationDisplay(job.operation, t)}状态：${job.phase}`)
                        : t("Starts with training or evaluation", "从训练或评估开始")}
                  </small>
                </div>
              </div>
              <div className={styles.progressLabel}>
                <span>{job.phase === "running" ? t(`${job.operation} process`, `${operationDisplay(job.operation, t)}进度`) : t("Training progress", "训练进度")}</span>
                <b>{trainingTotal > 0 ? `${formatNumber(trainingSteps)} / ${formatNumber(trainingTotal)}` : operationTime}</b>
              </div>
              <div
                className={`${styles.progress} ${job.phase === "running" && job.operation !== "train" ? styles.indeterminate : ""}`}
                role="progressbar"
                aria-label={t("Training progress", "训练进度")}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={trainingTotal > 0 ? Math.round(progress) : undefined}
              >
                <span style={{ width: `${trainingTotal > 0 ? progress : sourceReady ? 100 : 0}%` }} />
              </div>
              <div className={styles.chartHead}>
                <div className={styles.chartTitle}>
                  <strong>{rewardHistoryLabel} · {useRlxTelemetry ? t("current PPO stage", "当前 PPO 阶段") : t("live Duck Lab stream (up to 240 samples)", "Duck Lab 实时流（最多 240 个样本）")}</strong>
                  <button
                    type="button"
                    className={styles.historyIndicator}
                    onClick={() => setRewardHistoryOpen(true)}
                    aria-label={t(`Open reward history table with ${chartPoints.length} samples`, `打开包含 ${chartPoints.length} 个样本的奖励历史表`)}
                    title={t("Open reward history table", "打开奖励历史表")}
                  >
                    <Icon>▦</Icon>
                    <b>{chartPoints.length}</b>
                  </button>
                </div>
                <span>
                  <i />
                  {rewardIsNormalized
                    ? t("Normalized PPO rollout curve · evaluate exported ONNX separately", "归一化 PPO rollout 曲线 · 导出的 ONNX 需单独评估")
                    : t("PPO rollout curve · evaluate exported ONNX separately", "PPO rollout 曲线 · 导出的 ONNX 需单独评估")}
                </span>
              </div>
              <svg
                className={styles.chart}
                viewBox="0 0 420 128"
                preserveAspectRatio="none"
                role="img"
                aria-label={rewardHistoryLabel}
                data-reward-points={chartPoints.length}
              >
                <g className={styles.gridLines}>
                  <path d="M34 14H414M34 48H414M34 82H414M34 116H414" />
                </g>
                {chart ? (
                  <>
                    <g className={styles.chartLabels}>
                      {chart.yLabels.map((label, index) => (
                        <text key={`y-${index}`} x="2" y={18 + index * 49}>{formatAxisNumber(label)}</text>
                      ))}
                      {chart.xLabels.map((label, index) => (
                        <text key={`x-${index}`} x={34 + index * 126} y="127">{formatAxisNumber(label)}</text>
                      ))}
                    </g>
                    <path className={styles.chartArea} d={chart.area} />
                    <path className={styles.chartLine} d={chart.line} data-testid="reward-history-line" />
                    <circle cx={chart.lastX} cy={chart.lastY} r="3" className={styles.chartPoint} />
                  </>
                ) : (
                  <text x="224" y="67" textAnchor="middle" className={styles.chartEmpty}>
                    {hasTrainingTelemetry ? t("Waiting for first PPO rollout", "等待首次 PPO rollout") : t("No training telemetry yet", "尚无训练遥测")}
                  </text>
                )}
              </svg>
              <TrainingLifecycle history={job.trainingHistory} />
              <div className={styles.trainingActions}>
                {job.phase === "running" ? (
                  <button className={styles.button} onClick={() => runAction("cancel")} disabled={busy}>
                    <Icon>■</Icon> {t(`Stop ${job.operation}`, `停止${operationDisplay(job.operation, t)}`)}
                  </button>
                ) : (
                  <button
                    className={styles.button}
                    onClick={() =>
                      document
                        .getElementById("recipe")
                        ?.scrollIntoView({ behavior: "smooth", block: "start" })
                    }
                  >
                    <Icon>⚙</Icon> {t("Configure training", "配置训练")}
                  </button>
                )}
                <button className={styles.button} onClick={() => setLogsOpen((open) => !open)}>
                  {logsOpen ? t("Hide logs", "隐藏日志") : t("View logs", "查看日志")} <Icon>→</Icon>
                </button>
              </div>
              {logsOpen && (
                <pre className={styles.logs}>{job.logs.length ? job.logs.join("\n") : recipeCommand}</pre>
              )}
            </div>
            <div
              className={styles.animationBody}
              role="tabpanel"
              hidden={sessionTab !== "animate"}
            >
              <AnimPanel variant="embedded" active={sessionTab === "animate"} />
            </div>
            <div
              className={styles.toolBody}
              role="tabpanel"
              hidden={sessionTab !== "policies"}
            >
              <PolicyPanel
                clientRef={clientRef}
                variant="embedded"
                active={sessionTab === "policies"}
                onPoliciesChange={setPolicies}
                onStudioRunsChange={setSavedRuns}
                onReviewRun={loadSavedRun}
                reviewDisabled={busy || job.phase === "running"}
              />
            </div>
            <div
              className={styles.toolBody}
              role="tabpanel"
              hidden={sessionTab !== "teach"}
            >
              <TeachPanel clientRef={clientRef} variant="embedded" />
            </div>
          </section>
        </div>

        <div className={styles.lower}>
          <section className={styles.panel} id="recipe">
            <div className={styles.panelHead}>
              <h2><Icon>✦</Icon>{t("Microduck PPO Recipe", "Microduck PPO 配方")}</h2>
              <StatusPill tone="live">{t("RECOMMENDED", "推荐")}</StatusPill>
            </div>
            <form className={styles.recipeBody} onSubmit={(event) => { event.preventDefault(); if (!busy) void runAction("train"); }}>
              <div className={styles.recipeIntro}>
                <div className={styles.recipeMark}>PPO</div>
                <div>
                  <strong>{selectedExperimentText.title}</strong>
                  <span>{selectedExperimentText.task} · {selectedDrawingTool?.actorLabel ?? "512 → 256 → 128 ELU"}</span>
                </div>
              </div>
              <label className={styles.field}>
                <span>{t("Run name", "运行名称")}</span>
                <input
                  value={recipe.runName}
                  onChange={(event) => { setActionError(null); setRecipe((current) => ({ ...current, runName: event.target.value })); }}
                  pattern="[A-Za-z0-9_-]+"
                  maxLength={48}
                  required
                />
              </label>
              <button type="button" className={styles.button} onClick={prepareFreshRun} disabled={busy || Boolean(activeJob)}>{t("New run", "新运行")}</button>
              <label className={styles.field}>
                <span>{t("Saved runs", "已保存运行")}</span>
                <select aria-label={t("Saved runs", "已保存运行")} value="" disabled={busy} onChange={(event) => { if (event.target.value) void loadSavedRun(recipe.experimentId, event.target.value); }}>
                  <option value="">{t("Load a run and its saved evaluation recipe…", "加载运行及其保存的评估配方…")}</option>
                  {savedRuns.filter((run) => run.experimentId === recipe.experimentId).map((run) => <option key={run.runName} value={run.runName}>{run.taskPassed && run.renderVerified ? t("Passed", "已通过") : run.taskPassed ? t("Evaluator passed · render pending", "评估器通过 · 等待渲染") : run.skillAssessed ? t("Failed", "失败") : t("Not assessed", "未评估")}{run.drawingTool ? ` · ${drawingToolLabel(run.drawingTool, t)}` : ""} · {run.runName}</option>)}
                </select>
              </label>
              <p className={styles.evidenceNote}>{t("Typing a run name selects artifacts only. Use Saved runs to restore the exact saved evaluation settings, including clip, command, horizon, and PPO parameters. The checkpoint metadata records training provenance.", "直接输入运行名称只会选择产物。请使用“已保存运行”恢复准确的评估设置，包括片段、指令、时长和 PPO 参数。检查点元数据记录训练溯源。")}</p>
              {job.artifacts.checkpoint && <p className={styles.evidenceNote}>{t("This run already has a checkpoint. Choose a new run name for fresh training, or explicitly enable Continue current checkpoint in Advanced settings. Timesteps on continuation are additional, not the lifetime total.", "此运行已有检查点。若要重新训练，请选择新运行名称；若要继续训练，请在高级设置中明确启用“继续当前检查点”。续训步数是新增步数，不是生命周期总步数。")}</p>}
              {recipe.experimentId === "basketball" && job.artifacts.onnx && !job.artifacts.checkpoint && <p className={styles.evidenceNote}>{t("This balance run has a recurrent ONNX policy without a resumable checkpoint. Evaluation and rendering remain available; start training under a new run name.", "此平衡运行包含循环 ONNX 策略，但没有可续训检查点。仍可评估和渲染；请使用新运行名称开始训练。")}</p>}
              {recipe.experimentId === "dance" && <section className={styles.clipSelection}>
                <label className={styles.field}>
                  <span>{t("Dance reference clip", "舞蹈参考片段")}</span>
                  <select aria-label={t("Dance reference clip", "舞蹈参考片段")} value={recipe.danceClip ?? ""} onChange={(event) => {
                    const selected = danceClips.find((clip) => clip.path === event.target.value);
                    setRecipe((current) => ({ ...current, danceClip: selected?.path ?? null, ...(current.profile === "full" ? {
                      maxEpisodeS: selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds,
                      evalSteps: Math.ceil((selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds) * 50),
                      renderSeconds: selected?.durationSeconds ?? getExperiment("dance").maxEpisodeSeconds,
                    } : {}) }));
                  }}>
                    <option value="">{t("Built-in default choreography", "内置默认编舞")}</option>
                    {recipe.danceClip && !danceClips.some((clip) => clip.path === recipe.danceClip) && <option value={recipe.danceClip}>{t("Saved reference", "已保存参考")} · {recipe.danceClip.split("/").at(-1)}</option>}
                    {danceClips.map((clip) => <option key={clip.path} value={clip.path}>{clip.name} · {clip.durationSeconds.toFixed(2)} s</option>)}
                  </select>
                </label>
                <p>{t("Joint-angle reference for PPO, evaluation, and rendering—not an animation-only preview. In Full mode, selection sets the episode, evaluation, and video horizons to one complete clip. Changing the clip does not retrain an existing checkpoint or change its saved verdict. Catalog clips are not guaranteed physically learnable; train and verify each one.", "这是 PPO、评估和渲染使用的关节角参考，不只是动画预览。完整模式会将回合、评估和视频时长设为一个完整片段。更换片段不会重新训练已有检查点，也不会改变已保存判定。目录中的片段不保证在物理上可学习；每个片段都必须单独训练和验证。")}</p>
                <code>{recipe.danceClip ?? t("Built-in reference", "内置参考")}</code>
                {clipError && <p role="alert">{clipError}</p>}
              </section>}
              {recipe.experimentId === "drawing" && (
                <div className={styles.segmented} aria-label={t("Drawing tool", "绘画工具")}>
                  {(["pencil", "brush"] as const).map((drawingTool) => {
                    const definition = getDrawingTool(drawingTool);
                    return (
                      <button
                        key={drawingTool}
                        type="button"
                        className={recipe.drawingTool === drawingTool ? styles.selected : ""}
                        onClick={() => selectDrawingTool(drawingTool)}
                        disabled={busy || Boolean(activeJob)}
                      >
                        {drawingToolLabel(definition.id, t)}
                        <small>{definition.contractVersion} · {definition.observations}/{definition.actions}</small>
                      </button>
                    );
                  })}
                </div>
              )}
              <div className={styles.segmented} aria-label={t("Training profile", "训练配置")}>
                <button
                  type="button"
                  className={recipe.profile === "smoke" ? styles.selected : ""}
                  onClick={() => selectTrainingProfile("smoke")}
                  disabled={busy || Boolean(activeJob)}
                >
                  {t("Pipeline smoke", "流水线冒烟")}<small>{selectedDrawingTool ? t(`${formatNumber(selectedDrawingTool.smokeTimesteps)} steps · bounded pipeline`, `${formatNumber(selectedDrawingTool.smokeTimesteps)} 步 · 有界流水线`) : t("4 steps · wiring only", "4 步 · 仅验证连通性")}</small>
                </button>
                <button
                  type="button"
                  className={
                    recipe.profile === "full" &&
                    recipe.totalTimesteps === (selectedDrawingTool?.fullTimesteps ?? selectedExperiment.fullTimesteps) &&
                    hasDefaultFullEnvironment(recipe, selectedExperiment)
                      ? styles.selected
                      : ""
                  }
                  onClick={() => selectTrainingProfile("full")}
                  disabled={busy || Boolean(activeJob)}
                >
                  {t("Default full", "默认完整训练")}<small>{formatNumber(selectedDrawingTool?.fullTimesteps ?? selectedExperiment.fullTimesteps)} {t("steps", "步")} · {
                    selectedExperimentId === "backflip"
                      ? t("fixed assisted protocol", "固定辅助协议")
                      : selectedExperimentId === "basketball"
                        ? t("recurrent balance → steering", "循环平衡 → 转向")
                        : selectedExperimentId === "bridge"
                          ? t("suspended-bridge curriculum", "悬索桥课程")
                          : selectedExperimentId === "drawing"
                            ? t(`${selectedDrawingTool?.label.toLowerCase()} · fixed drawing recipe`, `${selectedDrawingToolLabel} · 固定绘画配方`)
                          : t("randomized", "随机化")
                  }</small>
                </button>
              </div>
              <p className={styles.evidenceNote}>{t("Choose a profile first, then click Start RLX. Profile presets prepare fresh training and preserve existing checkpoints under their original run names.", "先选择训练配置，再点击“启动 RLX”。配置预设会准备新的训练，并按原运行名称保留现有检查点。")}</p>
              {(recipe.experimentId === "basketball" || recipe.experimentId === "bridge" || recipe.experimentId === "drawing") && <p className={styles.evidenceNote}>{selectedExperimentText.readiness}</p>}
              {recipe.experimentId === "basketball" && <p className={styles.evidenceNote}>{t("Studio training is an unassisted fine-tune by default (hold 0, curriculum off). The adapter's assistance ladder is available only through its explicit CLI flags. Current Full evaluation covers zero-command balance and 0.15 forward tracking; lateral and yaw steering remain unevaluated.", "Studio 训练默认执行无辅助微调（保持时间 0，课程关闭）。适配器的辅助阶梯只能通过明确的 CLI 参数启用。当前完整评估覆盖零指令平衡和 0.15 前进跟踪；横向与偏航转向尚未评估。")}</p>}
              {recipe.experimentId === "bridge" && <p className={styles.evidenceNote}>{t("Default Full uses the 32,768-step transferred-walker pilot preset and freezes observation normalization. Automatic bootstrap and continuation warm-start the actor with a fresh critic and optimizer; this is not an exact optimizer resume.", "默认完整训练使用 32,768 步迁移行走试验预设，并冻结观测归一化。自动引导和续训会用全新的 critic 与优化器热启动 actor；这不是精确恢复优化器状态。")}</p>}
              {selectedDrawingTool?.id === "pencil" && <p className={styles.evidenceNote}>{t("Pencil uses microduck-drawing-v1: 83 observations, 15 actions, 32-second episodes, 256-step rollouts, and the original fixed SB3 recipe. A free pencil is held by the active lower jaw, and ink must come from physical pencil-surface contact.", "铅笔使用 microduck-drawing-v1：83 维观测、15 维动作、32 秒回合、256 步 rollout，以及原始固定 SB3 配方。自由铅笔由活动下颚夹持，墨迹必须来自铅笔与表面的物理接触。")}</p>}
              {selectedDrawingTool?.id === "brush" && <p className={styles.evidenceNote}>{t(`Brush uses microduck-brush-v2: 93 observations, 15 actions, 120-second episodes, 6,144-step rollouts, 512-sample batches (12 minibatches), two update epochs, clip 0.05, gamma 0.995, learning rate 1e-7, entropy 0, and max gradient norm 0.3. Exploration is per joint, not uniform: ${selectedDrawingTool.explorationLabel.toLowerCase()}. Final evaluation covers ${selectedDrawingTool.evaluationConfigs} distinct configurations × ${selectedDrawingTool.evalEpisodes} seeds = ${selectedDrawingTool.evaluationConfigs * selectedDrawingTool.evalEpisodes} rollouts. Its four RGB palette colors and visible ink must come from the actual brush point trace.`, `画笔使用 microduck-brush-v2：93 维观测、15 维动作、120 秒回合、6,144 步 rollout、512 样本批次（12 个 minibatch）、2 个更新 epoch、clip 0.05、gamma 0.995、学习率 1e-7、熵 0、最大梯度范数 0.3。探索按关节设置而非均匀分布：${selectedDrawingTool.explorationLabel.toLowerCase()}。最终评估覆盖 ${selectedDrawingTool.evaluationConfigs} 种不同配置 × ${selectedDrawingTool.evalEpisodes} 个种子 = ${selectedDrawingTool.evaluationConfigs * selectedDrawingTool.evalEpisodes} 次 rollout。四色 RGB 调色板与可见墨迹必须来自真实画笔点轨迹。`)}</p>}
              {recipe.experimentId === "backflip" && <p className={styles.evidenceNote}>{t("spotter-assisted launch → PPO landing → pretrained stand-policy handoff. Fresh Full exactly transfers the pretrained stand actor and observation normalizer, then calibrates reward scale from 16 teacher landing episodes × 600 steps with fixed discounted-return RMS and 10 critic-only epochs. The actor stays unchanged; calibrated reward normalization is frozen before PPO starts with a fresh optimizer, std 0.03, LR 3e-6, and 2 update epochs. This prevents warm-start scale shock but does not establish positive PPO improvement. Pipeline smoke remains unfrozen and skips teacher initialization and calibration.", "辅助起跳 → PPO 落地 → 预训练站立策略接管。新的完整训练会精确迁移预训练站立 actor 与观测归一化器，然后使用 16 个教师落地回合 × 600 步、固定折扣回报 RMS 和 10 个仅 critic epoch 校准奖励尺度。actor 保持不变；校准后的奖励归一化在 PPO 开始前冻结，并使用新优化器、标准差 0.03、学习率 3e-6 和 2 个更新 epoch。这可避免热启动尺度冲击，但不能证明 PPO 有正向提升。流水线冒烟不冻结归一化，并跳过教师初始化与校准。")}</p>}
              {recipe.experimentId === "swing" && (
                <section className={styles.swingPlan} aria-labelledby="swing-plan-title">
                  <div className={styles.swingPlanHead}>
                    <div>
                      <p className={styles.eyebrow}>{t("RECOVERY PLAN", "恢复计划")}</p>
                      <h3 id="swing-plan-title">{t("Build motion before spending 4M steps", "先建立有效动作，再投入 400 万步")}</h3>
                    </div>
                    <span className={swingTargetPassed ? styles.planPass : styles.planReject}>
                      {recipeMetric == null
                        ? t("No evaluation yet", "尚无评估")
                        : !verdict.skillAssessed
                          ? t("Skill not assessed", "技能未评估")
                          : swingTargetPassed
                            ? t(`${recipeMetric.toFixed(2)}° accepted`, `${recipeMetric.toFixed(2)}° 已接受`)
                            : t(`${recipeMetric.toFixed(2)}° rejected`, `${recipeMetric.toFixed(2)}° 已拒绝`)}
                    </span>
                  </div>
                  <div className={styles.swingSettings} aria-label={t("Exact Swing recovery settings", "秋千恢复的精确设置")}>
                    <div><strong>{t("Setting", "设置")}</strong><b>{t("Discovery", "探索")}</b><b>{t("Consolidation", "巩固")}</b></div>
                    <div><span>{t("Run", "运行")}</span><code>swing-curriculum-01</code><code>{t("same checkpoint", "同一检查点")}</code></div>
                    <div><span>{t("New PPO steps", "新增 PPO 步数")}</span><code>250,000</code><code>500,000</code></div>
                    <div><span>{t("Parallel envs / seed", "并行环境 / 种子")}</span><code>16 / 2</code><code>16 / 2</code></div>
                    <div><span>{t("Initial angle / rate", "初始角度 / 角速度")}</span><code>12° / 0.35 rad/s</code><code>4° / 0.12 rad/s</code></div>
                    <div><span>{t("Randomization / noise / delay", "随机化 / 噪声 / 延迟")}</span><code>{t("off / off / off", "关 / 关 / 关")}</code><code>{t("on / on / on", "开 / 开 / 开")}</code></div>
                    <div><span>{t("LR / gamma / clip", "学习率 / gamma / clip")}</span><code>1e-4 / .995 / .10</code><code>{t("same", "相同")}</code></div>
                    <div><span>{t("Epochs / entropy / grad norm", "Epoch / 熵 / 梯度范数")}</span><code>3 / .002 / 1.0</code><code>{t("same", "相同")}</code></div>
                  </div>
                  <ol className={styles.swingPlanSteps}>
                    <li>
                      <b>1</b>
                      <div>
                        <strong>{t("Preserve the failed baseline", "保留失败基线")}</strong>
                        <p>{t("Keep", "保留")} <code>swing-studio-01</code>。{t("Use a new run named", "使用新的运行名称")} <code>swing-curriculum-01</code>。</p>
                      </div>
                    </li>
                    <li>
                      <b>2</b>
                      <div>
                        <strong>{t("Discovery: 250k assisted steps", "探索：25 万辅助步")}</strong>
                        <p>{t("16 envs, seed 2, 12° initial angle, 0.35 rad/s initial rate, randomization/noise/delay off, default Swing rewards.", "16 个环境，种子 2，初始角度 12°，初始角速度 0.35 rad/s，关闭随机化、噪声和延迟，使用默认秋千奖励。")}</p>
                        <button type="button" className={styles.textButton} onClick={applySwingDiscovery}>{t("Apply exact Discovery settings", "应用精确探索设置")}</button>
                      </div>
                    </li>
                    <li>
                      <b>3</b>
                      <div>
                        <strong>{t("Evaluate from a still start", "从静止状态评估")}</strong>
                        <p>{t("Run Evaluate, then Render. Evaluation uses 16 envs × 1,200 control steps (24 s), with 0° initial angle and 0 rad/s. Continue only if mean span is at least 10° or best span is at least 20°.", "先运行评估，再运行渲染。评估使用 16 个环境 × 1,200 个控制步（24 秒），初始角度 0°、角速度 0 rad/s。仅当平均摆幅至少 10° 或最佳摆幅至少 20° 时继续。")}</p>
                        {evaluation && (
                          <p className={swingDiscoveryPassed ? styles.decisionPass : styles.decisionReject}>
                            {t("Current gate", "当前门槛")}：{t("mean", "平均")} {swingMeanSpan?.toFixed(2) ?? "—"}°，{t("best", "最佳")} {swingBestSpan?.toFixed(2) ?? "—"}° · {swingDiscoveryPassed ? t("continue", "继续") : t("reject and revise curriculum", "拒绝并修改课程")}
                          </p>
                        )}
                      </div>
                    </li>
                    <li>
                      <b>4</b>
                      <div>
                        <strong>{t("Consolidate: 500k additional steps", "巩固：新增 50 万步")}</strong>
                        <p>{t("Resume the same checkpoint, reduce assistance to 4° and 0.12 rad/s, then enable domain randomization, observation noise, and action delay.", "继续同一检查点，将辅助降至 4° 和 0.12 rad/s，然后启用域随机化、观测噪声和动作延迟。")}</p>
                        <button
                          type="button"
                          className={styles.textButton}
                          onClick={applySwingConsolidation}
                          disabled={
                            !job.artifacts.checkpoint ||
                            recipe.runName !== "swing-curriculum-01" ||
                            !swingDiscoveryPassed
                          }
                          title={
                            !job.artifacts.checkpoint
                              ? t("Run Discovery first", "先运行探索阶段")
                              : !swingDiscoveryPassed
                                ? t("Evaluate Discovery and pass the 10° mean or 20° best gate first", "先评估探索阶段，并通过平均 10° 或最佳 20° 的门槛")
                                : undefined
                          }
                        >
                          {t("Apply exact Consolidation settings", "应用精确巩固设置")}
                        </button>
                      </div>
                    </li>
                    <li>
                      <b>5</b>
                      <div>
                        <strong>{t("Decide with evidence", "依据证据决策")}</strong>
                        <p>{t("After 500k, continue toward 1M only if mean span reaches 30° or best span reaches 60°. Default skill acceptance requires a symmetric 150° span (at least 75° each side), a complete 24-second episode, valid geometry, and tensioned strings in every episode. Review the video separately; pipeline smoke is not a skill assessment.", "完成 50 万步后，只有平均摆幅达到 30° 或最佳摆幅达到 60° 才继续到 100 万步。默认技能验收要求对称总摆幅 150°（每侧至少 75°）、完整 24 秒回合、有效几何结构，并且每个回合的绳索保持张紧。视频需单独检查；流水线冒烟不属于技能评估。")}</p>
                        {evaluation && (
                          <p className={swingConsolidationPassed ? styles.decisionPass : styles.decisionReject}>
                            {t("1M gate", "100 万步门槛")}：{swingConsolidationPassed ? t("eligible to continue", "可以继续") : t("do not extend yet", "暂不延长")}
                          </p>
                        )}
                      </div>
                    </li>
                  </ol>
                </section>
              )}
              {recipe.experimentId !== "drawing" && <button type="button" className={styles.advancedButton} onClick={() => setAdvanced((open) => !open)}>
                <span>{t("Advanced PPO and environment settings", "高级 PPO 与环境设置")}</span><b>{advanced ? "−" : "+"}</b>
              </button>}
              {advanced && recipe.experimentId !== "drawing" && (
                <div className={styles.advancedSettings}>
                  <div className={styles.advancedGrid}>
                    {([
                      ["numSteps", t("Rollout steps per environment", "每个环境的 rollout 步数"), 1, 100000, 1],
                      ["numMinibatches", t("Minibatches per PPO epoch", "每个 PPO epoch 的 minibatch 数"), 1, 100000, 1],
                      ["maxEpisodeS", t("Episode horizon (seconds)", "回合时长（秒）"), 0.02, 3600, 0.02],
                      ["evalSteps", t("Evaluation steps per environment", "每个环境的评估步数"), 1, 10000000, 1],
                      ["renderSeconds", t("Video horizon (seconds)", "视频时长（秒）"), 0.02, 3600, 0.02],
                      ["initialStd", t("Initial exploration standard deviation", "初始探索标准差"), 0.001, 10, 0.01],
                      ["checkpointInterval", t("Checkpoint interval (transitions; 0 disables)", "检查点间隔（转换数；0 表示禁用）"), 0, 40000000, 1],
                    ] as const).map(([key, label, minimum, maximum, step]) => <label className={styles.field} key={key}><span>{label}</span><input type="number" min={minimum} max={maximum} step={step} value={recipe[key]} onChange={(event) => setRecipe((current) => ({ ...current, [key]: Number(event.target.value) }))} /></label>)}
                    <Toggle label={t("Normalize rewards", "归一化奖励")} checked={recipe.normalizeRewards} onChange={(checked) => setRecipe((current) => ({ ...current, normalizeRewards: checked }))} />
                    <Toggle label={t("Freeze observation normalization", "冻结观测归一化")} checked={recipe.freezeObservationNormalization} onChange={(checked) => setRecipe((current) => ({ ...current, freezeObservationNormalization: checked }))} />
                    <Toggle label={t("Continue current checkpoint", "继续当前检查点")} checked={recipe.resumeFromCheckpoint} onChange={(checked) => setRecipe((current) => ({ ...current, resumeFromCheckpoint: checked }))} />
                    {recipe.experimentId === "dance" && <label className={styles.field}><span>{t("Dance pose sigma (blank uses default)", "舞蹈姿态 sigma（留空使用默认值）")}</span><input type="number" min="0.001" step="0.01" value={recipe.dancePoseSigma ?? ""} onChange={(event) => setRecipe((current) => ({ ...current, dancePoseSigma: event.target.value === "" ? null : Number(event.target.value) }))} /></label>}
                    {(recipe.experimentId === "running" || recipe.experimentId === "stilts") && <label className={styles.field}><span>{t("Fixed forward command (m/s; blank samples commands)", "固定前进指令（m/s；留空则采样指令）")}</span><input type="number" min="0.01" max="1.5" step="0.01" value={recipe.locomotionForwardCommand ?? ""} onChange={(event) => setRecipe((current) => ({ ...current, locomotionForwardCommand: event.target.value === "" ? null : Number(event.target.value) }))} /></label>}
                    <label className={styles.field}><span>{t("Total timesteps", "总时间步")}</span><input type="number" min="4" max="40000000" value={recipe.totalTimesteps} onChange={(event) => setRecipe((current) => ({ ...current, totalTimesteps: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Parallel envs", "并行环境数")}</span><input type="number" min="1" max="64" value={recipe.numEnvs} onChange={(event) => setRecipe((current) => ({ ...current, numEnvs: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Learning rate", "学习率")}</span><input type="number" min="0.000001" max="0.1" step="0.0001" value={recipe.learningRate} onChange={(event) => setRecipe((current) => ({ ...current, learningRate: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Gamma", "Gamma")}</span><input type="number" min="0.8" max="1" step="0.01" value={recipe.gamma} onChange={(event) => setRecipe((current) => ({ ...current, gamma: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("PPO clip", "PPO 裁剪系数")}</span><input type="number" min="0.01" max="1" step="0.01" value={recipe.clipCoefficient} onChange={(event) => setRecipe((current) => ({ ...current, clipCoefficient: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Update epochs", "更新 epoch 数")}</span><input type="number" min="1" max="20" step="1" value={recipe.updateEpochs} onChange={(event) => setRecipe((current) => ({ ...current, updateEpochs: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Entropy coefficient", "熵系数")}</span><input type="number" min="0" max="1" step="0.001" value={recipe.entropyCoefficient} onChange={(event) => setRecipe((current) => ({ ...current, entropyCoefficient: Number(event.target.value) }))} /></label>
                    <label className={styles.field}><span>{t("Max gradient norm", "最大梯度范数")}</span><input type="number" min="0.01" max="10" step="0.1" value={recipe.maxGradNorm} onChange={(event) => setRecipe((current) => ({ ...current, maxGradNorm: Number(event.target.value) }))} /></label>
                    <Toggle label={t("Domain randomization", "域随机化")} checked={recipe.domainRand} onChange={(checked) => setRecipe((current) => ({ ...current, domainRand: checked }))} />
                    <Toggle label={t("Observation noise", "观测噪声")} checked={recipe.obsNoise} onChange={(checked) => setRecipe((current) => ({ ...current, obsNoise: checked }))} />
                    <Toggle label={t("Action delay", "动作延迟")} checked={recipe.actionDelay} onChange={(checked) => setRecipe((current) => ({ ...current, actionDelay: checked }))} />
                    <Toggle label={t("Random yaw", "随机偏航")} checked={recipe.randomYaw} onChange={(checked) => setRecipe((current) => ({ ...current, randomYaw: checked }))} />
                    {recipe.experimentId === "stilts" && (
                      <>
                        <label className={styles.field}><span>{t("Stilt height (cm)", "高跷高度（cm）")}</span><input type="number" min="0.8" max="300" step="0.5" value={recipe.stiltHeightCm} onChange={(event) => setRecipe((current) => ({ ...current, stiltHeightCm: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>{t("Support blend", "支撑混合")}</span><input type="number" min="0" max="1" step="0.05" value={recipe.stiltBlend} onChange={(event) => setRecipe((current) => ({ ...current, stiltBlend: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>{t("Mass per stilt (kg)", "每根高跷质量（kg）")}</span><input type="number" min="0.001" max="2" step="0.001" value={recipe.stiltMassKg} onChange={(event) => setRecipe((current) => ({ ...current, stiltMassKg: Number(event.target.value) }))} /></label>
                      </>
                    )}
                    {recipe.experimentId === "swing" && (
                      <>
                        <label className={styles.field}><span>{t("Initial angle assistance (degrees)", "初始角度辅助（度）")}</span><input type="number" min="0" max="30" step="1" value={recipe.swingInitialAngleDeg} onChange={(event) => setRecipe((current) => ({ ...current, swingInitialAngleDeg: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>{t("Initial rate assistance (rad/s)", "初始角速度辅助（rad/s）")}</span><input type="number" min="0" max="1" step="0.01" value={recipe.swingInitialRateRadS} onChange={(event) => setRecipe((current) => ({ ...current, swingInitialRateRadS: Number(event.target.value) }))} /></label>
                        <label className={styles.field}><span>{t("Skill target: symmetric total span (degrees)", "技能目标：对称总摆幅（度）")}</span><input type="number" min="1" max="180" step="1" value={recipe.swingMinSpanDeg} onChange={(event) => setRecipe((current) => ({ ...current, swingMinSpanDeg: Number(event.target.value) }))} /></label>
                        <Toggle label={t("Planar discovery actions", "平面探索动作")} checked={recipe.swingPlanarActions} onChange={(checked) => setRecipe((current) => ({ ...current, swingPlanarActions: checked }))} />
                      </>
                    )}
                  </div>
                  <section className={styles.rewardEditor} aria-labelledby="reward-weights-title">
                    <div className={styles.rewardEditorHead}>
                      <div>
                        <h3 id="reward-weights-title">{t("Reward weights", "奖励权重")}</h3>
                        <p>{t("Absolute non-negative coefficients. Penalty measurements are already negative.", "使用非负绝对系数。惩罚测量值本身已经为负。")}</p>
                      </div>
                      <button
                        type="button"
                        className={styles.textButton}
                        onClick={() => setRecipe((current) => ({
                          ...current,
                          rewardWeights: defaultRewardWeights(current.experimentId),
                        }))}
                      >
                        {t("Reset defaults", "恢复默认值")}
                      </button>
                    </div>
                    <div className={styles.rewardWeightGrid}>
                      {selectedExperiment.reward.terms
                        .filter((term) => term.editable !== false)
                        .map((term) => {
                          const termText = rewardTermDisplay(selectedExperiment.id, term, t);
                          return <label className={styles.rewardWeightRow} key={term.key}>
                            <span>
                              <strong>{termText.label}</strong>
                              <small>{term.penalty ? t("Penalty", "惩罚") : t("Reward", "奖励")} · {term.key}</small>
                            </span>
                            <input
                              type="number"
                              min="0"
                              max="10000"
                              step="any"
                              value={recipe.rewardWeights[term.key] ?? term.defaultWeight}
                              onChange={(event) => {
                                const value = Math.max(0, Number(event.target.value));
                                setRecipe((current) => ({
                                  ...current,
                                  rewardWeights: {
                                    ...current.rewardWeights,
                                    [term.key]: Number.isFinite(value) ? value : term.defaultWeight,
                                  },
                                }));
                              }}
                              aria-label={t(`${term.label} reward weight`, `${termText.label}奖励权重`)}
                            />
                          </label>;
                        })}
                    </div>
                    <p className={styles.rewardRetrainNote}>{t("Changing a reward weight defines a new experiment. Start a new training run before evaluating or rendering it.", "修改奖励权重会定义一个新实验。请先开始新的训练运行，再进行评估或渲染。")}</p>
                  </section>
                </div>
              )}
              <div className={styles.recipeActions}>
                <button
                  type="button"
                  className={`${styles.button} ${styles.primary}`}
                  onClick={() => runAction("train")}
                  disabled={busy || job.phase === "running" || anotherRunActive}
                  aria-describedby="rlx-action-status"
                >
                  <Icon>{pendingAction === "train" ? "…" : "▶"}</Icon> {trainButtonLabel}
                </button>
                {recipe.experimentId === "dance" ? (
                  <button type="button" onClick={startLabTraining} className={styles.button} disabled={busy || !connected} title={t("Separate free-text teaching workflow; does not use this RLX clip or recipe", "独立的自由文本教学流程；不使用此 RLX 片段或配方")}>
                    <Icon>⌁</Icon> {t("Separate Duck Lab teaching", "独立 Duck Lab 教学")}
                  </button>
                ) : (
                  <a
                    className={styles.button}
                    href={`${publicAssetUrl("/guides/microduck-studio-experiments-guide.pdf")}#${selectedExperiment.guideAnchor}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    <Icon>?</Icon> {t("Open recipe guide", "打开配方指南")}
                  </a>
                )}
              </div>
              <p
                id="rlx-action-status"
                className={`${styles.recipeActionStatus} ${
                  actionError || (job.phase === "failed" && job.operation === "train")
                    ? styles.recipeActionError
                    : ""
                }`}
                role={actionError ? "alert" : "status"}
                aria-live="polite"
              >
                <i
                  className={actionError ? styles.statusError :
                    pendingAction === "train" || job.phase === "running"
                      ? styles.statusWorking
                      : job.phase === "failed"
                        ? styles.statusError
                        : styles.statusReady
                  }
                />
                <span>
                  {recipeActionStatus ??
                    t("Ready to launch this recipe with the selected training settings.", "已准备使用所选训练设置启动此配方。")}
                </span>
              </p>
              <p className={styles.honesty}>{t(`Smoke mode validates the pipeline only. A learned ${selectedExperiment.shortTitle.toLowerCase()} policy requires full training plus deterministic and visual review.`, `冒烟模式只验证流水线。学到的${selectedExperimentText.shortTitle}策略需要完整训练、确定性评估和视觉检查。`)}</p>
            </form>
          </section>

          <section className={styles.panel} id="evaluation">
            <div className={styles.panelHead}>
              <div>
                <h2><Icon>⌁</Icon>{t("Evaluation gate", "评估门槛")}</h2>
                <p>{t("Smoke checks the pipeline (4 steps); full evaluates the skill. Swing requires 1,200 steps (24 s); Backflip 600 steps (12 s); Basketball 3,000 steps (60 s); Bridge 1,000 steps (20 s).", "冒烟模式检查流水线（4 步）；完整模式评估技能。秋千需要 1,200 步（24 秒）；后空翻 600 步（12 秒）；篮球 3,000 步（60 秒）；桥梁 1,000 步（20 秒）。")}</p>
                {evaluation && <p>{t("Saved evaluation scope", "已保存评估范围")}：{verdict.scope ?? t("unknown (legacy report)", "未知（旧版报告）")}{verdict.swingMinSpanDeg == null ? "" : t(` · target ${verdict.swingMinSpanDeg}° total / ${verdict.swingMinSpanDeg / 2}° each side`, ` · 目标总摆幅 ${verdict.swingMinSpanDeg}° / 每侧 ${verdict.swingMinSpanDeg / 2}°`)}。{t("Current recipe edits do not change saved results.", "当前配方修改不会改变已保存结果。")}</p>}
              </div>
              <button className={styles.textButton} onClick={() => runAction("eval")} disabled={busy || job.phase === "running" || Boolean(activeJob) || !sourceReady}>
                {t("Run evaluation ↗", "运行评估 ↗")}
              </button>
            </div>
            <div className={styles.evaluationBody}>
              <section className={styles.skillSummary} aria-label={t("Skill verification summary", "技能验证摘要")}>
                <h3>{evaluationAndRenderPassed ? t("Skill verified in simulation", "技能已在仿真中验证") : taskPassed ? t("Evaluator passed; render verification pending", "评估器已通过；等待渲染验证") : verdict.skillAssessed ? t("Skill criteria failed", "技能标准未通过") : t("Skill not yet verified", "技能尚未验证")}</h3>
                <p><strong>{evidence.passed} / {evidence.episodes.length} {t("episodes passed", "个回合通过")}</strong> · {recipe.runName}</p>
                <p>{t("Acceptance requires every complete episode, not just a high average reward. The verdict is bound to current policy bytes; preview clips are not training evidence.", "验收要求每个完整回合都通过，而不是仅有较高平均奖励。判定绑定到当前策略字节；预览片段不是训练证据。")}</p>
                {selectedExperimentId === "swing" && <p>{t("Swing acquisition used teacher BC/DAgger initialization followed by PPO refinement. This is not a demonstrated pure-PPO-from-scratch success.", "秋千技能使用教师 BC/DAgger 初始化，再由 PPO 优化。这并不是已证明的纯 PPO 从零训练成功。")}</p>}
                {selectedExperimentId === "dance" && <p>{t("Dance verifies the saved reference horizon and joint-angle tracking, not automatic imitation of the entire source video.", "舞蹈验证保存的参考时长与关节角跟踪，而不是自动模仿完整源视频。")}</p>}
                {selectedExperimentId === "backflip" && <p>{t("spotter-assisted launch → PPO landing → pretrained stand-policy handoff. Fresh Full exactly transfers the pretrained stand actor and observation normalizer into the landing student. Acceptance requires at least 5.8 rad of credited rotation before handoff, then 10 consecutive stable PPO landing steps with no spotter, stand override, or body support, followed by at least 2 seconds of stable standing. The stand policy cannot finish credited rotation.", "辅助起跳 → PPO 落地 → 预训练站立策略接管。新的完整训练会将预训练站立 actor 和观测归一化器精确迁移到落地学生策略。验收要求接管前至少完成 5.8 rad 的计入旋转，然后连续 10 个稳定 PPO 落地步，期间没有辅助者、站立覆盖或身体支撑，之后稳定站立至少 2 秒。站立策略不能补足计入旋转。")}</p>}
                {selectedExperimentId === "basketball" && <p>{t("Balance-only evidence is intentionally previewable after 60 unassisted seconds, but the task remains failed until the evaluator explicitly passes steering. Recurrent state must carry across ordinary steps and reset with the episode.", "仅平衡证据在无辅助 60 秒后可供预览，但在评估器明确通过转向之前，任务仍视为失败。循环状态必须在普通步骤间延续，并随回合重置。")}</p>}
                {selectedExperimentId === "bridge" && <p>{t("The bridge curriculum is training-only. Evaluation and rendering must use the final suspended narrow bridge unassisted for all 1,000 steps.", "桥梁课程仅用于训练。评估和渲染必须在最终狭窄悬索桥上无辅助运行全部 1,000 步。")}</p>}
                {selectedExperimentId === "drawing" && <p>{t(`${selectedDrawingTool?.label} drawing remains diagnostic until the saved evaluator reports`, `${selectedDrawingToolLabel}绘画仍属于诊断状态，直到保存的评估器报告`)} <code>drawing_assessment.passed: true</code> {t("and", "且")} <code>drawing_assessment.unassisted: true</code>，{t(`and the render receipt matches the same ${selectedDrawingTool?.contractVersion} ONNX bytes. The rendered ink must come from the actual tool trace with no reference overlay.`, `并且渲染回执匹配同一 ${selectedDrawingTool?.contractVersion} ONNX 字节。渲染墨迹必须来自真实工具轨迹，不得叠加参考图。`)}</p>}
                {typeof evaluation?.source_sha256 === "string" && <p>{t("Evaluated source SHA-256", "已评估源文件 SHA-256")}：<code>{evaluation.source_sha256}</code></p>}
                {evidence.ranges.length > 0 && <table aria-label={t("Measured skill ranges", "已测技能范围")}><thead><tr><th>{t("Measurement", "测量项")}</th><th>{t("Episode minimum", "回合最小值")}</th><th>{t("Episode maximum", "回合最大值")}</th></tr></thead><tbody>
                  {evidence.ranges.map((range) => <tr key={range.metric}><td>{evidenceLabel(range.metric)}<small> · {range.measured}/{evidence.episodes.length} {t("measured", "已测量")}</small></td><td>{range.minimum.toFixed(4)}</td><td>{range.maximum.toFixed(4)}</td></tr>)}
                </tbody></table>}
                {rolloutRanges.length > 0 && <table aria-label={t("Rollout assistance channel ranges", "Rollout 辅助通道范围")}><thead><tr><th>{t("Actual-unit channel", "实际单位通道")}</th><th>{t("Rollout minimum", "Rollout 最小值")}</th><th>{t("Rollout maximum", "Rollout 最大值")}</th></tr></thead><tbody>
                  {rolloutRanges.map((range) => <tr key={range.metric}><td>{evidenceLabel(range.metric)}</td><td>{range.minimum.toFixed(4)}</td><td>{range.maximum.toFixed(4)}</td></tr>)}
                </tbody></table>}
                <details><summary>{t("Exact acceptance criteria and per-episode results", "精确验收标准与逐回合结果")}</summary>
                  <table><tbody>{Object.entries(evidence.criteria).map(([key, value]) => <tr key={key}><th>{evidenceLabel(key)}</th><td>{JSON.stringify(value)}</td></tr>)}</tbody></table>
                  {evidence.episodes.map((episode, index) => <details key={index}><summary>{t("Environment", "环境")} {String(episode.env_index)} · {t("episode", "回合")} {String(episode.episode_index)} · {episode.passed === true ? t("Passed", "通过") : t("Failed", "失败")} · {String(episode.measured_steps)} {t("measured steps", "个测量步")}</summary><pre>{JSON.stringify(episode, null, 2)}</pre></details>)}
                </details>
                <details><summary>{t("Saved evaluation recipe and raw report", "已保存评估配方与原始报告")}</summary><pre>{JSON.stringify(evaluation, null, 2)}</pre></details>
              </section>
              {job.artifacts.renderVideo && (
                <section className={styles.rolloutPlayer} aria-labelledby="rollout-player-title">
                  <div>
                    <p className={styles.eyebrow}>{t("SELECTED RUN · SAVED VIDEO", "已选择运行 · 已保存视频")}</p>
                    <h3 id="rollout-player-title">{recipe.runName} {t("rollout", "回放")}</h3>
                    <p>{job.renderVerified ? t("Policy, external dependencies, and media hashes match. Video settings are the nominal projection of the saved evaluation recipe (randomization disabled); this does not visualize every evaluation domain. Watch the complete rollout before marking reviewed.", "策略、外部依赖和媒体哈希匹配。视频设置是保存评估配方在关闭随机化时的标称投影；它不会展示每个评估域。标记已检查前请观看完整回放。") : t("Unbound or stale video: re-render using the saved evaluation recipe before marking reviewed. Legacy videos cannot satisfy the package gate.", "视频未绑定或已过期：请使用保存的评估配方重新渲染，再标记为已检查。旧版视频不能满足产物门槛。")}</p>
                  </div>
                  <video
                    key={reviewIdentity}
                    controls
                    playsInline
                    preload="metadata"
                    src={`${artifactHref("video")}&inline=1&source=${encodeURIComponent(String(evaluation?.source_sha256 ?? "unknown"))}`}
                  >
                    {t("Your browser cannot play this MP4.", "你的浏览器无法播放此 MP4。")}
                  </video>
                  <div className={styles.rolloutPlayerActions}>
                    <a className={styles.button} href={`${artifactHref("video")}&inline=1`} target="_blank" rel="noreferrer">
                      <Icon>▶</Icon> {t("Open full video", "打开完整视频")}
                    </a>
                    <a className={styles.button} href={artifactHref("sheet")} target="_blank" rel="noreferrer">
                      <Icon>▦</Icon> {t("Open contact sheet", "打开帧预览图")}
                    </a>
                  </div>
                </section>
              )}
              <table>
                <thead><tr><th>{t("Check", "检查项")}</th><th>{t("Evidence", "证据")}</th><th>{t("Result", "结果")}</th></tr></thead>
                <tbody>
                  <tr><td><i className={job.artifacts.checkpoint ? styles.passDot : styles.mutedDot} />{t("Checkpoint", "检查点")}</td><td>{selectedExperimentId === "basketball" ? "Recurrent checkpoint.pt + result.json" : selectedExperimentId === "drawing" ? "policy.zip + policy.zip.json" : "Safetensors + sidecar"}</td><td>{job.artifacts.checkpoint ? t("Ready", "就绪") : t("Missing", "缺失")}</td></tr>
                  <tr><td><i className={job.artifacts.onnx ? styles.passDot : styles.mutedDot} />{t("ONNX contract", "ONNX 合约")}</td><td>{selectedPolicyContract}</td><td>{job.artifacts.onnx ? t("Ready", "就绪") : t("Missing", "缺失")}</td></tr>
                  <tr><td><i className={evalPassed ? styles.passDot : styles.warnDot} />{t("Rollout validity", "Rollout 有效性")}</td><td>{t("Finite observations, rewards, and actions", "观测、奖励和动作均为有限值")}</td><td>{typeof evaluation?.pipeline_passed === "boolean" ? (evalPassed ? t("Passed", "通过") : t("Failed", "失败")) : t("Unavailable", "不可用")}</td></tr>
                  <tr><td><i className={evaluationAndRenderPassed ? styles.passDot : styles.warnDot} />{selectedExperimentText.metricLabel}</td><td>{recipeMetric == null ? t("Not measured", "未测量") : `${recipeMetric.toFixed(3)}${selectedExperiment.metricSuffix}`}</td><td>{!verdict.skillAssessed ? t("Not assessed", "未评估") : evaluationAndRenderPassed ? t("Verified", "已验证") : taskPassed ? t("Render pending", "等待渲染") : t("Failed", "失败")}</td></tr>
                  <tr><td><i className={job.artifacts.renderSheet && job.artifacts.renderVideo ? styles.passDot : styles.warnDot} />{t("Visual artifact", "视觉产物")}</td><td>{t("MP4 + contact sheet (separate visual review)", "MP4 + 帧预览图（单独视觉检查）")}</td><td>{job.artifacts.renderSheet && job.artifacts.renderVideo ? t("Ready", "就绪") : t("Required", "必需")}</td></tr>
                  <tr><td><i className={job.renderVerified ? styles.passDot : styles.warnDot} />{t("Video provenance", "视频溯源")}</td><td>{t("Source, clip, settings and media hashes", "源文件、片段、设置与媒体哈希")}</td><td>{job.renderVerified ? t("Matched", "匹配") : t("Re-render required", "需要重新渲染")}</td></tr>
                </tbody>
              </table>
              <div className={styles.evaluationActions}>
                <button className={styles.button} onClick={() => runAction("render")} disabled={busy || job.phase === "running" || Boolean(activeJob) || !sourceReady}>
                  <Icon>◫</Icon> {t("Render rollout", "渲染回放")}
                </button>
                {!job.artifacts.onnx && job.artifacts.checkpoint && selectedExperimentId !== "basketball" && (
                  <button className={styles.button} onClick={() => runAction("export")} disabled={busy || job.phase === "running" || Boolean(activeJob)}>
                    <Icon>↓</Icon> {t("Export ONNX", "导出 ONNX")}
                  </button>
                )}
                {job.artifacts.renderVideo && <a className={styles.button} href={`${artifactHref("video")}&inline=1`} target="_blank" rel="noreferrer"><Icon>▶</Icon> {t("Play rollout", "播放回放")}</a>}
                {job.artifacts.renderSheet && <a className={styles.button} href={artifactHref("sheet")} target="_blank" rel="noreferrer"><Icon>▦</Icon> {t("Contact sheet", "帧预览图")}</a>}
              </div>
              <label className={styles.reviewCheck}>
                <input type="checkbox" checked={visualReviewed && Boolean(job.renderVerified)} onChange={(event) => setVisualReviewed(event.target.checked)} disabled={!job.renderVerified} />
                <span><strong>{t("I reviewed the rendered motion", "我已检查渲染动作")}</strong><small>{
                  selectedExperimentId === "backflip"
                    ? t("Confirm spotter-assisted launch → PPO landing → pretrained stand-policy handoff, including credited rotation before handoff and 10 support-free landing steps.", "确认辅助起跳 → PPO 落地 → 预训练站立策略接管，包括接管前计入旋转以及 10 个无支撑落地步。")
                    : selectedExperimentId === "basketball"
                      ? t("Confirm unassisted balance and whether the commanded ball motion actually steers; balance alone is not full success.", "确认无辅助平衡，以及指令球体运动是否真正实现转向；仅平衡不代表完整成功。")
                      : selectedExperimentId === "bridge"
                        ? t("Confirm the complete unassisted suspended narrow-bridge crossing without falls or side exits.", "确认完整无辅助通过狭窄悬索桥，过程中没有跌落或侧边退出。")
                        : selectedExperimentId === "drawing"
                          ? t(`Confirm the free ${selectedDrawingTool?.label.toLowerCase()} remains held by the active lower jaw and that visible ink uses the actual ${selectedDrawingTool?.label.toLowerCase()} point trace with no reference overlay.`, `确认自由${selectedDrawingToolLabel}始终由活动下颚夹持，并且可见墨迹来自真实${selectedDrawingToolLabel}点轨迹，没有参考图叠加。`)
                        : t("Confirm the policy moves as intended and outperforms null controls.", "确认策略按预期运动，并优于空控制。")
                }</small></span>
              </label>
            </div>
          </section>

          <section className={styles.panel} id="deployment">
            <div className={styles.panelHead}>
              <h2><Icon>♢</Icon>{selectedExperimentId === "backflip" ? t("Controller handoff", "控制器交接") : selectedExperimentId === "drawing" ? t("Simulation export", "仿真导出") : t("Deployment handoff", "部署交接")}</h2>
              <StatusPill tone={deployReady ? "live" : "warn"}>{deployReady ? selectedExperimentId === "backflip" ? t("LANDING POLICY READY", "落地策略已就绪") : selectedExperimentId === "drawing" ? t("DIAGNOSTIC EXPORT READY", "诊断导出已就绪") : t("PACKAGE READY", "包已就绪") : t("GATED", "受门槛限制")}</StatusPill>
            </div>
            <div className={styles.deployBody}>
              <div className={styles.deployArtifact}>
                <span className={styles.fileIcon}>ONNX</span>
                <div><strong>{selectedExperimentId === "drawing" ? "policy" : selectedExperiment.artifactStem}.onnx</strong><small>{selectedExperimentId === "backflip" ? t("PPO landing stage only; launch assistance and stand handoff are external", "仅包含 PPO 落地阶段；起跳辅助和站立接管位于外部") : t("Normalizer baked into the graph", "归一化器已烘焙进计算图")}</small></div>
                <span>{job.artifacts.onnx ? t("Ready", "就绪") : t("Pending", "等待中")}</span>
              </div>
              <ul className={styles.gateList}>
                <li className={job.artifacts.onnx ? styles.complete : ""}>{selectedPolicyContract}</li>
                <li className={evaluationAndRenderPassed ? styles.complete : ""}>
                  {selectedExperimentId === "swing"
                    ? t("Swing skill passed: saved symmetric target, 24 s, geometry and string tension", "秋千技能通过：保存的对称目标、24 秒、几何与绳索张力")
                    : selectedExperimentId === "backflip"
                      ? t("Backflip skill passed: ≥5.8 rad before handoff, ≥10 consecutive support-free PPO landing steps, ≥2 s stand hold", "后空翻技能通过：接管前 ≥5.8 rad、连续 ≥10 个无支撑 PPO 落地步、站立保持 ≥2 秒")
                    : selectedExperimentId === "basketball"
                      ? t("Basketball task passed: 60 s unassisted balance and explicit steering acceptance", "篮球任务通过：无辅助平衡 60 秒，并明确通过转向验收")
                    : selectedExperimentId === "bridge"
                      ? t("Bridge task passed: explicit unassisted suspended-bridge assessment", "桥梁任务通过：明确通过无辅助悬索桥评估")
                    : selectedExperimentId === "drawing"
                      ? t("Drawing passed: explicit unassisted drawing_assessment acceptance", "绘画通过：drawing_assessment 明确通过无辅助验收")
                    : t("Deterministic evaluation passed", "确定性评估通过")}
                </li>
                <li className={job.artifacts.renderSheet && visualReviewed ? styles.complete : ""}>{t("Rendered motion reviewed", "已检查渲染动作")}</li>
                <li>{selectedExperimentId === "backflip" ? t("Integrate spotter-assisted launch → PPO landing → pretrained stand-policy handoff; landing ONNX alone is not the controller", "集成辅助起跳 → PPO 落地 → 预训练站立策略接管；仅有落地 ONNX 并不构成完整控制器") : selectedExperimentId === "drawing" ? t("Simulation-only independent contract; do not present this ONNX as a robot deployment policy", "仅限仿真的独立合约；不要将此 ONNX 作为机器人部署策略") : t("Complete hardware-specific validation before deployment", "部署前完成硬件专项验证")}</li>
              </ul>
              <div className={styles.deployActions}>
                <a
                  className={`${styles.button} ${deployReady ? styles.primary : styles.disabled}`}
                  href={deployReady ? artifactHref("onnx") : undefined}
                  aria-disabled={!deployReady}
                  onClick={(event) => {
                    if (!deployReady) event.preventDefault();
                  }}
                >
                  <Icon>↓</Icon> {selectedExperimentId === "backflip" ? t("Download PPO landing policy", "下载 PPO 落地策略") : selectedExperimentId === "drawing" ? t("Download simulation policy", "下载仿真策略") : t("Download policy", "下载策略")}
                </a>
                <button className={styles.button} disabled title={t("Physical robot transport is not configured", "尚未配置实体机器人传输")}>
                  <Icon>⌾</Icon> {t("Send to robot", "发送到机器人")}
                </button>
              </div>
              <p className={styles.safetyNote}><Icon>✓</Icon><span>{selectedExperimentId === "backflip"
                ? t(
                    "spotter-assisted launch → PPO landing → pretrained stand-policy handoff. This download is the learned landing stage only, depends on the source-bound alpha_stand policy and Python-owned protocol, and is not a hardware-ready controller.",
                    "辅助起跳 → PPO 落地 → 预训练站立策略接管。此下载仅包含学到的落地阶段，依赖绑定源文件的 alpha_stand 策略和 Python 管理的协议，并非硬件就绪控制器。"
                  )
                : selectedExperimentId === "drawing"
                  ? t(
                      `${selectedDrawingTool?.contractVersion} is an independent ${selectedDrawingTool?.observations}-observation, ${selectedDrawingTool?.actions}-action simulation contract. Its ONNX is diagnostic and is not compatible with the shared robot deployment contract.`,
                      `${selectedDrawingTool?.contractVersion} 是独立的 ${selectedDrawingTool?.observations} 维观测、${selectedDrawingTool?.actions} 维动作仿真合约。其 ONNX 仅用于诊断，与共享机器人部署合约不兼容。`
                    )
                  : t(
                      "Local RLX output is a prototyping artifact. Hardware deployment remains disabled until an authenticated robot transport and hardware-specific validation are configured.",
                      "本地 RLX 输出是原型产物。在配置经过认证的机器人传输和硬件专项验证之前，硬件部署保持禁用。"
                    )}</span></p>
            </div>
          </section>
        </div>

        <section className={`${styles.panel} ${styles.artifactPanel}`} id="artifacts">
          <div className={styles.panelHead}>
            <h2><Icon>▱</Icon>{t("Run artifacts", "运行产物")}</h2>
            <span>{job.artifacts.checkpointPath}</span>
          </div>
          <div className={styles.artifactGrid}>
            {[
              [t("Checkpoint", "检查点"), "checkpoint", job.artifacts.checkpoint, t("Training weights", "训练权重")],
              [t("Metadata", "元数据"), "metadata", job.artifacts.metadata, t("Recipe and contract", "配方与合约")],
              [t("ONNX policy", "ONNX 策略"), "onnx", deployReady, selectedExperimentId === "backflip" ? t("PPO landing stage only · requires assisted launch and pretrained stand-policy handoff", "仅 PPO 落地阶段 · 需要辅助起跳与预训练站立策略接管") : t("Simulation graph · requires matched visual review", "仿真计算图 · 需要匹配的视觉检查")],
              [t("Contact sheet", "帧预览图"), "sheet", job.artifacts.renderSheet, t("Visual verification", "视觉验证")],
              [t("Rollout video", "回放视频"), "video", job.artifacts.renderVideo, t("Motion review", "动作检查")],
            ].map(([label, kind, ready, detail]) => (
              <a
                key={String(kind)}
                className={`${styles.artifact} ${ready ? styles.ready : ""}`}
                href={ready ? artifactHref(kind as "checkpoint" | "metadata" | "onnx" | "sheet" | "video") : undefined}
                aria-disabled={!ready}
                onClick={(event) => {
                  if (!ready) event.preventDefault();
                }}
              >
                <Icon>{ready ? "✓" : "·"}</Icon>
                <span><strong>{label}</strong><small>{detail}</small></span>
                <b>{ready ? "↓" : t("Pending", "等待中")}</b>
              </a>
            ))}
          </div>
        </section>

        <section className={styles.systemStrip} id="system">
          <div><Icon>⌘</Icon><span><strong>Apple Silicon</strong><small>{t("MLX · Metal learning path", "MLX · Metal 学习路径")}</small></span><b>{t("Local", "本地")}</b></div>
          <div><Icon>◇</Icon><span><strong>{t("MuJoCo physics", "MuJoCo 物理")}</strong><small>{connected ? t("Duck Lab streaming", "Duck Lab 流式传输") : t("Waiting on :8788", "等待 :8788")}</small></span><b>{connected ? t("Live", "在线") : t("Offline", "离线")}</b></div>
          <div><Icon>⌾</Icon><span><strong>{t("Physical robot", "实体机器人")}</strong><small>{t("Authenticated transport", "已认证传输")}</small></span><b className={styles.mutedText}>{t("Not configured", "未配置")}</b></div>
          <div><Icon>▱</Icon><span><strong>{t("Policy library", "策略库")}</strong><small>{t("Assignable local runs", "可分配的本地运行")}</small></span><b>{activePolicies}</b></div>
        </section>

        <footer className={styles.footer}>
          <span><i className={styles.onlineDot} />Microduck Studio</span>
          <span>{selectedDrawingTool ? t(`RLX PPO · ${selectedDrawingTool.observations} observations · ${selectedDrawingTool.actions} actions`, `RLX PPO · ${selectedDrawingTool.observations} 维观测 · ${selectedDrawingTool.actions} 维动作`) : t("RLX PPO · 61 observations · 14 actions", "RLX PPO · 61 维观测 · 14 维动作")}</span>
          <span>{t("Local-first · Hardware gated", "本地优先 · 硬件受门槛限制")}</span>
        </footer>
      </main>

      <dialog
        ref={rewardHistoryDialogRef}
        className={styles.rewardHistoryDialog}
        aria-labelledby="reward-history-title"
        onClose={() => {
          setRewardHistoryOpen(false);
          setRewardHistoryCopied(false);
        }}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            event.currentTarget.close();
          }
        }}
      >
        <div className={styles.rewardHistoryContent}>
          <header className={styles.rewardHistoryHeader}>
            <div>
              <p className={styles.eyebrow}>{t("TRAINING TELEMETRY", "训练遥测")}</p>
              <h2 id="reward-history-title">{rewardHistoryLabel}</h2>
              <p>
                {chartPoints.length
                  ? t(`${chartPoints.length} available ${rewardIsNormalized ? "normalized " : ""}rollout samples for ${recipe.runName}.`, `${recipe.runName} 有 ${chartPoints.length} 个可用的${rewardIsNormalized ? "归一化" : ""} rollout 样本。`)
                  : t(`No rollout samples are available for ${recipe.runName} yet.`, `${recipe.runName} 尚无可用 rollout 样本。`)}
              </p>
            </div>
            <button
              type="button"
              className={styles.dialogClose}
              aria-label={t("Close reward history", "关闭奖励历史")}
              onClick={() => rewardHistoryDialogRef.current?.close()}
            >
              ×
            </button>
          </header>
          <div className={styles.rewardHistoryActions}>
            <span>{t("Tab-separated text · paste directly into a spreadsheet", "制表符分隔文本 · 可直接粘贴到电子表格")}</span>
            <button
              type="button"
              className={`${styles.button} ${chartPoints.length ? styles.primary : ""}`}
              disabled={!chartPoints.length}
              onClick={copyRewardHistory}
            >
              <Icon>{rewardHistoryCopied ? "✓" : "▣"}</Icon>
              {rewardHistoryCopied ? t("Copied", "已复制") : t("Copy table", "复制表格")}
            </button>
          </div>
          <div className={styles.rewardHistoryTableWrap}>
            <table className={styles.rewardHistoryTable}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>{t("Training step", "训练步数")}</th>
                  <th>{rewardIsNormalized ? t("Normalized mean rollout reward", "归一化平均 rollout 奖励") : t("Mean rollout reward", "平均 rollout 奖励")}</th>
                </tr>
              </thead>
              <tbody>
                {chartPoints.length ? (
                  chartPoints.map((point, index) => (
                    <tr key={`${point.step}-${index}`}>
                      <td>{index + 1}</td>
                      <td>{formatNumber(point.step)}</td>
                      <td>{point.reward}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={3}>{t("Start training to collect reward samples.", "开始训练以收集奖励样本。")}</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </dialog>

      <dialog
        ref={guidanceDialogRef}
        className={styles.recipeGuidanceDialog}
        aria-labelledby="recipe-guidance-title"
        onClose={() => setGuidanceExperimentId(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) {
            event.currentTarget.close();
          }
        }}
      >
        {guidanceExperiment && (
          <div className={styles.recipeGuidanceContent}>
            <header className={styles.recipeGuidanceHeader}>
              <div>
                <p className={styles.eyebrow}>{t("RECIPE GUIDANCE", "配方指南")}</p>
                <h2 id="recipe-guidance-title">{guidanceExperimentText?.title}</h2>
                <p>{guidanceExperimentText?.goal}</p>
              </div>
              <button
                type="button"
                className={styles.dialogClose}
                aria-label={t("Close recipe guidance", "关闭配方指南")}
                onClick={() => guidanceDialogRef.current?.close()}
              >
                ×
              </button>
            </header>

            <section className={styles.guidanceInput}>
              <strong>{t("Required input", "必需输入")}</strong>
              <p>{guidanceDetails?.requiredInput}</p>
            </section>

            <section className={styles.guidanceSteps}>
              <h3>{t("Train, evaluate, and render", "训练、评估与渲染")}</h3>
              <ol>
                {guidanceDetails?.steps.map((step) => (
                  <li key={step}>{step}</li>
                ))}
              </ol>
            </section>

            <section className={styles.guidanceConcepts}>
              <h3>{t("What clip, command, and PPO mean here", "此处片段、指令与 PPO 的含义")}</h3>
              <ul>
                {guidanceDetails?.distinctions.map((distinction) => (
                  <li key={distinction}>{distinction}</li>
                ))}
              </ul>
            </section>

            <section className={styles.guidanceReward}>
              <h3>{t("PPO reward function", "PPO 奖励函数")}</h3>
              <p>{guidanceDetails?.rewardSummary}</p>
              <code>{guidanceExperiment.reward.formula}</code>
              <p className={styles.rewardRule}>
                {t("Each measured term is multiplied by its weight and summed every control step. Penalty terms already return negative values, so their editable weights stay non-negative.", "每个测量项乘以其权重，并在每个控制步求和。惩罚项已经返回负值，因此其可编辑权重保持非负。")}
              </p>
              <div className={styles.guidanceRewardTerms}>
                {guidanceExperiment.reward.terms.map((term) => {
                  const termText = rewardTermDisplay(guidanceExperiment.id, term, t);
                  return <div key={term.key}>
                    <span>
                      <strong>{termText.label}</strong>
                      <small>{termText.description}</small>
                    </span>
                    <b>
                      {term.editable === false ? t("dynamic x ", "动态 × ") : ""}
                      {term.defaultWeight}
                    </b>
                  </div>;
                })}
              </div>
              <p className={styles.rewardRule}>
                {guidanceExperiment.id === "drawing"
                  ? t("Drawing rewards are fixed by the independent Python environment and are not editable in Studio.", "绘画奖励由独立 Python 环境固定，无法在 Studio 中编辑。")
                  : t("Customize editable weights in Advanced PPO and environment settings. Retrain after any change; the checkpoint metadata records the chosen weights.", "在高级 PPO 与环境设置中调整可编辑权重。任何修改后都需要重新训练；检查点元数据会记录所选权重。")}
              </p>
            </section>

            <footer className={styles.guidanceOutput}>
              <strong>{t("Outputs", "输出")}</strong>
              <p>{guidanceDetails?.output}</p>
            </footer>
          </div>
        )}
      </dialog>
    </div>
  );
}
