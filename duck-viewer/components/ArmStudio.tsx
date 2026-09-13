"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  ARM_CASES,
  parseArmRuns,
  parseArmState,
  type ArmCaseId,
  type ArmGeom,
  type ArmRunsResponse,
  type ArmState,
} from "@/lib/arm-contract";
import { armCaseTitle } from "@/lib/arm-labels";
import { IS_STATIC_EXPORT } from "@/lib/static-assets";
import styles from "./ArmStudio.module.css";
import ArmVideoLibrary from "./ArmVideoLibrary";
import { LanguageToggle, useLanguage } from "./LanguageProvider";

const DEFAULT_CASE: ArmCaseId = "arm-reach-v1";
type Translate = (english: string, chinese: string) => string;

const ARM_CASE_COPY: Record<ArmCaseId, {
  summary: { en: string; zh: string };
}> = {
  "arm-reach-v1": {
    summary: {
      en: "Observe coordinates, joints, and target error to complete an unloaded reach.",
      zh: "观察坐标、关节与目标误差，完成无载 reach。",
    },
  },
  "arm-pick-place-v1": {
    summary: {
      en: "Use real contact, lift, move, and release; pushing into the target does not count as a grasp.",
      zh: "真实接触、抬升、移动、释放，不把推入目标算抓取。",
    },
  },
  "arm-relocate-v1": {
    summary: {
      en: "Lift and reposition an object in a fixed obstacle scene.",
      zh: "在固定障碍场景中抬高并重新放置物体。",
    },
  },
  "arm-carry-v1": {
    summary: {
      en: "Move a fixed-base arm through waypoints; this is not a walking carry task.",
      zh: "固定底座机械臂依次经过航点；不是鸭子行走搬运。",
    },
  },
  "arms-handover-v1": {
    summary: {
      en: "The sending arm releases only after the receiving arm establishes a grasp.",
      zh: "接收臂建立抓握后，发送臂才释放。",
    },
  },
  "arms-co-carry-v1": {
    summary: {
      en: "Both arms grasp, lift, carry, and stably release the tray together.",
      zh: "两臂共同抓持、抬升、搬运与稳定释放。",
    },
  },
};

const JOINT_NAMES = [
  "base_yaw",
  "shoulder_pitch",
  "elbow_pitch",
  "wrist_pitch",
  "wrist_roll",
  "gripper",
];

interface ApiError extends Error {
  status?: number;
}

function Icon({ children }: { children: React.ReactNode }) {
  return <span className={styles.icon} aria-hidden="true">{children}</span>;
}

async function requestJson(path: string, init?: RequestInit) {
  const response = await fetch(`/api/arm/${path}`, {
    ...init,
    cache: "no-store",
    headers: {
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  const payload = await response.json().catch(() => null) as
    | { error?: string }
    | null;
  if (!response.ok) {
    const error = new Error(
      payload?.error || `Arm API request failed (${response.status}).`
    ) as ApiError;
    error.status = response.status;
    throw error;
  }
  return payload;
}

function localizedCase(caseId: ArmCaseId, language: "en" | "zh") {
  const copy = ARM_CASE_COPY[caseId];
  return {
    title: armCaseTitle(caseId, language),
    summary: copy.summary[language],
    group: language === "zh"
      ? (caseId.startsWith("arms-") ? "双臂" : "单臂")
      : (caseId.startsWith("arms-") ? "Dual arm" : "Single arm"),
  };
}

function formatMetric(value: unknown, t: Translate): string {
  if (typeof value === "number") {
    const absolute = Math.abs(value);
    if (absolute >= 1_000) return value.toLocaleString("en-US");
    if (absolute > 0 && absolute < 0.001) return value.toExponential(2);
    return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return t("Not measured", "未测");
  return JSON.stringify(value);
}

function controllerLabel(controller: string, t: Translate) {
  const normalized = controller.toLowerCase();
  if (normalized.includes("teacher")) {
    return t("Teacher reference controller (not PPO)", "Teacher 参考控制器（非 PPO）");
  }
  if (normalized.includes("ppo")) return `PPO · ${controller}`;
  if (normalized.includes("bc")) return `BC · ${controller}`;
  return controller || t("Not reported", "未报告");
}

function statusLabel(state: ArmState | null, t: Translate) {
  if (!state) return t("No live state", "无实时状态");
  if (state.terminated) return t("Terminated", "已终止");
  if (state.truncated) return t("Truncated", "已截断");
  return t("Simulation running", "仿真运行中");
}

function localizedError(message: string, t: Translate) {
  const requestFailure = message.match(/^Arm API request failed \((\d+)\)\.$/);
  if (requestFailure) {
    return t(
      message,
      `机械臂 API 请求失败（${requestFailure[1]}）。`
    );
  }
  const translations: Record<string, string> = {
    "Saved runs response failed contract validation.": "已保存运行记录未通过契约验证。",
    "Live state failed contract validation.": "实时状态未通过契约验证。",
    "Arm simulation backend unavailable on localhost:8812.": "localhost:8812 上的机械臂仿真后端不可用。",
    "Arm backend unavailable.": "机械臂后端不可用。",
    "Arm backend returned an invalid state.": "机械臂后端返回了无效状态。",
    "Cannot load runs.": "无法加载运行记录。",
  };
  return t(message, translations[message] ?? message);
}

function GeometryShape({ geom }: { geom: ArmGeom }) {
  const quaternion: [number, number, number, number] = [
    geom.quat[1],
    geom.quat[2],
    geom.quat[3],
    geom.quat[0],
  ];
  const color = `rgb(${Math.round(geom.rgba[0] * 255)}, ${Math.round(geom.rgba[1] * 255)}, ${Math.round(geom.rgba[2] * 255)})`;
  const opacity = Math.max(0, Math.min(1, geom.rgba[3]));
  const material = (
    <meshStandardMaterial
      color={color}
      transparent={opacity < 1}
      opacity={opacity}
      roughness={0.72}
      metalness={0.04}
    />
  );

  if (geom.type === "box") {
    return (
      <mesh position={geom.pos} quaternion={quaternion}>
        <boxGeometry args={[geom.size[0] * 2, geom.size[1] * 2, geom.size[2] * 2]} />
        {material}
      </mesh>
    );
  }
  if (geom.type === "sphere") {
    return (
      <mesh position={geom.pos} quaternion={quaternion}>
        <sphereGeometry args={[geom.size[0], 20, 14]} />
        {material}
      </mesh>
    );
  }
  if (geom.type === "capsule") {
    return (
      <group position={geom.pos} quaternion={quaternion}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <capsuleGeometry args={[geom.size[0], geom.size[1] * 2, 8, 16]} />
          {material}
        </mesh>
      </group>
    );
  }
  return (
    <group position={geom.pos} quaternion={quaternion}>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[geom.size[0], geom.size[0], geom.size[1] * 2, 20]} />
        {material}
      </mesh>
    </group>
  );
}

function ArmScene({ state }: { state: ArmState | null }) {
  return (
    <Canvas
      dpr={[1, 1.5]}
      gl={{ antialias: true, powerPreference: "high-performance" }}
      camera={{ position: [0.32, 0.26, 0.38], fov: 42, near: 0.01, far: 20 }}
    >
      <color attach="background" args={["#f4f7f5"]} />
      <hemisphereLight intensity={1.1} groundColor="#c8d0cc" color="#ffffff" />
      <directionalLight position={[2, 4, 3]} intensity={1.8} />
      <directionalLight position={[-2, 2, -1]} intensity={0.45} color="#b9c8d7" />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]}>
        <planeGeometry args={[4, 4]} />
        <meshStandardMaterial color="#edf1ef" roughness={1} />
      </mesh>
      <gridHelper args={[3, 30, "#9eaaa5", "#d4dbd7"]} position={[0, -0.001, 0]} />
      <axesHelper args={[0.12]} position={[-0.42, 0.002, 0.36]} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {state?.geoms.map((geom, index) => (
          <GeometryShape key={`${geom.name}-${index}`} geom={geom} />
        ))}
      </group>
      <OrbitControls
        makeDefault
        target={[0.04, 0.05, 0]}
        minDistance={0.25}
        maxDistance={3}
        maxPolarAngle={Math.PI / 2 - 0.02}
        zoomSpeed={0.55}
      />
    </Canvas>
  );
}

function Metrics({
  metrics,
  t,
  compact = false,
}: {
  metrics: Record<string, unknown>;
  t: Translate;
  compact?: boolean;
}) {
  const entries = Object.entries(metrics);
  if (!entries.length) return <p className={styles.emptyText}>{t("Not measured", "未测")}</p>;
  return (
    <dl className={compact ? styles.metricListCompact : styles.metricList}>
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{formatMetric(value, t)}</dd>
        </div>
      ))}
    </dl>
  );
}

const PRIMARY_METRIC_KEYS = [
  "position_error_m",
  "tip_error_m",
  "grasp_hold_s",
  "receiver_hold_s",
  "settled_s",
  "drop_count",
  "invalid_contacts",
  "arm_collisions",
  "self_collisions",
] as const;

function PrimaryMetrics({
  metrics,
  t,
}: {
  metrics: Record<string, unknown>;
  t: Translate;
}) {
  const gates = metrics.gates;
  const gateEntries = gates && typeof gates === "object" && !Array.isArray(gates)
    ? Object.entries(gates as Record<string, unknown>)
      .filter((entry): entry is [string, boolean] => typeof entry[1] === "boolean")
    : [];
  const failedGates = gateEntries.filter(([, passed]) => !passed).map(([key]) => key);
  const entries = PRIMARY_METRIC_KEYS
    .filter((key) => key in metrics)
    .map((key) => [key, metrics[key]] as const);

  if (!gateEntries.length && !entries.length) {
    return <p className={styles.emptyText}>{t("Not measured", "未测")}</p>;
  }

  return (
    <dl className={styles.metricList}>
      {gateEntries.length > 0 && (
        <>
          <div>
            <dt>gates</dt>
            <dd>
              {gateEntries.filter(([, passed]) => passed).length}/{gateEntries.length}{" "}
              {t("passed", "通过")}
            </dd>
          </div>
          <div>
            <dt>failed_gates</dt>
            <dd>{failedGates.length ? failedGates.join(", ") : t("none", "无")}</dd>
          </div>
        </>
      )}
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{formatMetric(value, t)}</dd>
        </div>
      ))}
    </dl>
  );
}

function hasCurrentProvenance(metrics: Record<string, unknown>) {
  return metrics.sources_match === true &&
    metrics.model_match === true &&
    metrics.checkpoint_match === true &&
    metrics.context_match === true;
}

function runVerdict(run: ArmRunsResponse["runs"][number], t: Translate) {
  if (!hasCurrentProvenance(run.metrics)) {
    return { label: t("Evidence unverified", "证据未核实"), className: styles.unmeasured };
  }
  if (run.passed === true) return { label: t("Passed", "通过"), className: styles.pass };
  if (run.passed === false) return { label: t("Failed", "未通过"), className: styles.fail };
  return { label: t("Not measured", "未测"), className: styles.unmeasured };
}

function runSeed(runId: string) {
  return runId.match(/(?:^|:)seed-(\d+)(?::|$)/)?.[1] ?? "—";
}

export default function ArmStudio() {
  const { language, t } = useLanguage();
  const [workspaceView, setWorkspaceView] = useState<"videos" | "live">("videos");
  const [videoCase, setVideoCase] = useState<ArmCaseId | "all">("all");
  const [selectedCase, setSelectedCase] = useState<ArmCaseId>(DEFAULT_CASE);
  const selectedDefinition = useMemo(
    () => ARM_CASES.find((entry) => entry.id === selectedCase) ?? ARM_CASES[0],
    [selectedCase]
  );
  const selectedCopy = localizedCase(selectedCase, language);
  const [seed, setSeed] = useState(7);
  const [state, setState] = useState<ArmState | null>(null);
  const [runs, setRuns] = useState<ArmRunsResponse["runs"]>([]);
  const [available, setAvailable] = useState(false);
  const [loading, setLoading] = useState(!IS_STATIC_EXPORT);
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [teacherTicks, setTeacherTicks] = useState(5);
  const [action, setAction] = useState<number[]>(() => new Array(6).fill(0));
  const [error, setError] = useState<string | null>(null);
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null);
  const [showRunHistory, setShowRunHistory] = useState(false);
  const commandLock = useRef(false);

  useEffect(() => {
    const readHash = () => {
      if (IS_STATIC_EXPORT) {
        setWorkspaceView("videos");
        return;
      }
      if (window.location.hash === "#live" || window.location.hash === "#evidence" || window.location.hash === "#bom") setWorkspaceView("live");
      else if (window.location.hash === "#videos") setWorkspaceView("videos");
    };
    readHash();
    window.addEventListener("hashchange", readHash);
    return () => window.removeEventListener("hashchange", readHash);
  }, []);

  function showWorkspace(view: "videos" | "live") {
    if (IS_STATIC_EXPORT && view === "live") return;
    setWorkspaceView(view);
    if (view === "videos") setPlaying(false);
    window.history.replaceState(window.history.state, "", `#${view}`);
  }

  const markOffline = useCallback((message: string) => {
    setAvailable(false);
    setPlaying(false);
    setState(null);
    setRuns([]);
    setError(message);
  }, []);

  const loadRuns = useCallback(async () => {
    const payload = parseArmRuns(await requestJson("runs"));
    if (!payload) throw new Error("Saved runs response failed contract validation.");
    setRuns(payload.runs);
  }, []);

  const loadState = useCallback(async (syncSelection = false) => {
    const payload = parseArmState(await requestJson("state"));
    if (!payload) throw new Error("Live state failed contract validation.");
    setState(payload);
    if (syncSelection) {
      setSelectedCase(payload.case_id);
      setSeed(payload.seed);
      setAction(new Array(payload.action_dim).fill(0));
    }
    setAvailable(true);
    setError(null);
    return payload;
  }, []);

  const connect = useCallback(async () => {
    if (IS_STATIC_EXPORT) return;
    setLoading(true);
    try {
      await requestJson("health");
      await Promise.all([loadState(true), loadRuns()]);
      setAvailable(true);
      setError(null);
    } catch (cause) {
      markOffline(
        cause instanceof Error
          ? cause.message
          : "Arm simulation backend unavailable on localhost:8812."
      );
    } finally {
      setLoading(false);
    }
  }, [loadRuns, loadState, markOffline]);

  useEffect(() => {
    if (IS_STATIC_EXPORT) return;
    const timer = window.setTimeout(() => void connect(), 0);
    return () => window.clearTimeout(timer);
  }, [connect]);

  useEffect(() => {
    if (!available || playing) return;
    const interval = window.setInterval(() => {
      void loadState().catch((cause) => {
        markOffline(cause instanceof Error ? cause.message : "Arm backend unavailable.");
      });
    }, 1_500);
    return () => window.clearInterval(interval);
  }, [available, loadState, markOffline, playing]);

  const runCommand = useCallback(async (
    path: "reset" | "step" | "teacher",
    body: Record<string, unknown>
  ) => {
    if (IS_STATIC_EXPORT) return null;
    if (commandLock.current) return null;
    commandLock.current = true;
    setBusy(true);
    try {
      const payload = parseArmState(await requestJson(path, {
        method: "POST",
        body: JSON.stringify(body),
      }));
      if (!payload) throw new Error("Arm backend returned an invalid state.");
      setState(payload);
      setAvailable(true);
      setError(null);
      return payload;
    } catch (cause) {
      markOffline(
        cause instanceof Error
          ? cause.message
          : "Arm simulation backend unavailable on localhost:8812."
      );
      return null;
    } finally {
      commandLock.current = false;
      setBusy(false);
    }
  }, [markOffline]);

  useEffect(() => {
    if (!playing || !available) return;
    let cancelled = false;
    let timer = 0;
    const advance = async () => {
      const next = await runCommand("teacher", { ticks: teacherTicks });
      if (cancelled || !next || next.terminated || next.truncated) {
        if (!cancelled) setPlaying(false);
        return;
      }
      timer = window.setTimeout(advance, 40);
    };
    void advance();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [available, playing, runCommand, teacherTicks]);

  const caseMismatch = state !== null && state.case_id !== selectedCase;
  const currentActionDim = selectedDefinition.actionDim;
  const visibleAction = action.length === currentActionDim
    ? action
    : new Array(currentActionDim).fill(0);
  const visibleRuns = useMemo(
    () => showRunHistory
      ? runs
      : runs.filter((run) =>
        run.case_id === selectedCase && hasCurrentProvenance(run.metrics)
      ),
    [runs, selectedCase, showRunHistory]
  );

  async function reset() {
    setPlaying(false);
    const next = await runCommand("reset", { case_id: selectedCase, seed });
    if (next) {
      setAction(new Array(next.action_dim).fill(0));
      void loadRuns().catch(() => undefined);
    }
  }

  function updateAction(index: number, value: number) {
    setAction((current) => {
      const next = current.length === currentActionDim
        ? [...current]
        : new Array(currentActionDim).fill(0);
      next[index] = value;
      return next;
    });
  }

  function selectCase(caseId: ArmCaseId) {
    const definition = ARM_CASES.find((entry) => entry.id === caseId);
    setSelectedCase(caseId);
    setVideoCase(caseId);
    setPlaying(false);
    setAction(new Array(definition?.actionDim ?? 6).fill(0));
  }

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.brandBlock}>
          <Link
            className={styles.backLink}
            href="/"
            aria-label={t("Back to Microduck Studio", "返回 Microduck Studio")}
          >
            <Icon>←</Icon><span>Microduck Studio</span>
          </Link>
          <div>
            <p className={styles.eyebrow}>
              {t("SIMULATION HTTP · LOCALHOST:8812", "仿真 HTTP · LOCALHOST:8812")}
            </p>
            <h1>{t("Arm Lab", "机械臂课堂")}</h1>
          </div>
        </div>
        <div className={styles.headerActions}>
          <LanguageToggle />
          <a
            className={styles.contactButton}
            href="mailto:bochuxt7@gmail.com"
            title={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}
            aria-label={t("Email bochuxt7@gmail.com", "发送邮件至 bochuxt7@gmail.com")}
          >
            <Icon>✉</Icon>
            <span>{t("Contact", "联系")}</span>
          </a>
          <span className={styles.lockBadge}>
            <Icon>▣</Icon>
            {t("HARDWARE LOCKED · NO UNIX TRANSPORT", "硬件已锁定 · 无 UNIX 传输")}
          </span>
          <span className={available ? styles.onlineBadge : styles.offlineBadge}>
            <i />
            {IS_STATIC_EXPORT
              ? t("Static evidence", "静态证据")
              : loading
              ? t("Connecting", "连接中")
              : available
                ? t("Simulation online", "仿真在线")
                : t("Simulation unavailable", "仿真不可用")}
          </span>
          <button
            className={styles.iconButton}
            onClick={() => void connect()}
            disabled={IS_STATIC_EXPORT}
            title={IS_STATIC_EXPORT ? t("Static evidence mode", "静态证据模式") : t("Reconnect", "重新连接")}
            aria-label={IS_STATIC_EXPORT ? t("Static evidence mode", "静态证据模式") : t("Reconnect", "重新连接")}
          >
            ↻
          </button>
        </div>
      </header>

      <main className={styles.main}>
        <section className={styles.introBand}>
          <div>
            <p className={styles.eyebrow}>{t("SIX ARM EXPERIMENTS", "六个机械臂案例")}</p>
            <h2>{t("From joint actions to verifiable tabletop tasks", "从关节动作到可验证的桌面操作")}</h2>
            <p>
              {t(
                "Review videos and failure analysis before entering the live MuJoCo experiment. Recordings are independent of the simulation service; physics, rewards, and training remain on the Python side.",
                "先看视频与失败复盘，再进入实时 MuJoCo 实验。录像独立于仿真服务；物理、奖励和训练仍由 Python 端负责。"
              )}
            </p>
          </div>
          <nav className={styles.docLinks} aria-label={t("Arm Lab documents", "机械臂实验室文档")}>
            <a href="#videos" onClick={() => showWorkspace("videos")}>
              <Icon>▷</Icon><span>{t("All case videos", "全部案例视频")}</span>
            </a>
            <a href="#bom"><Icon>▤</Icon><span>{t("BOM and safety", "BOM 与安全")}</span></a>
            <a href="#evidence"><Icon>▱</Icon><span>{t("Evidence and run history", "证据与运行记录")}</span></a>
          </nav>
        </section>

        <nav className={styles.workspaceNav} aria-label={t("Arm workspaces", "机械臂工作区")}>
          <button aria-pressed={workspaceView === "videos"} onClick={() => showWorkspace("videos")}>
            <Icon>▷</Icon>{t("Video library", "视频证据")}
          </button>
          {!IS_STATIC_EXPORT && <button aria-pressed={workspaceView === "live"} onClick={() => showWorkspace("live")}>
            <Icon>⌁</Icon>{t("Live simulator", "实时实验")}
          </button>}
          <span>{t("SIMULATION ONLY · HARDWARE LOCKED", "仅限仿真 · 硬件已锁定")}</span>
        </nav>

        {error && workspaceView === "live" && (
          <div className={styles.errorBanner} role="alert">
            <strong>{t("Arm backend unavailable", "机械臂后端不可用")}</strong>
            <span>{localizedError(error, t)}</span>
            <code>python HTTP server · 127.0.0.1:8812</code>
          </div>
        )}

        <section className={styles.caseStrip} aria-label={t("Arm experiment cases", "机械臂实验案例")}>
          {ARM_CASES.map((entry, index) => {
            const caseCopy = localizedCase(entry.id, language);
            return (
              <button
                key={entry.id}
                className={entry.id === (workspaceView === "videos" ? videoCase : selectedCase) ? styles.caseActive : styles.caseButton}
                onClick={() => selectCase(entry.id)}
                aria-label={language === "zh"
                  ? `${caseCopy.title}。${caseCopy.summary}`
                  : `${caseCopy.title}. ${caseCopy.summary}`}
                aria-pressed={entry.id === (workspaceView === "videos" ? videoCase : selectedCase)}
              >
                <span className={styles.caseNumber}>{String(index + 1).padStart(2, "0")}</span>
                <span>
                  <strong>{caseCopy.title}</strong>
                  <code>{entry.id}</code>
                </span>
                <small>{entry.observationDim}/{entry.actionDim}</small>
              </button>
            );
          })}
        </section>

        {workspaceView === "videos" && <div id="videos"><ArmVideoLibrary caseFilter={videoCase} onCaseFilterChange={setVideoCase} /></div>}

        {workspaceView === "live" && <>
        <section className={styles.workspace} id="live">
          <div className={styles.viewerPanel}>
            <div className={styles.panelHeading}>
              <div>
                <p className={styles.eyebrow}>{t("WORLD-SPACE GEOMETRY", "世界空间几何")}</p>
                <h3 title={selectedCopy.summary}>{selectedCopy.title}</h3>
              </div>
              <div className={styles.stateBadges}>
                <span>{selectedCopy.group}</span>
                <span>{statusLabel(state, t)}</span>
                {caseMismatch && (
                  <span className={styles.warningBadge}>
                    {t("Reset to switch cases", "重置后切换案例")}
                  </span>
                )}
              </div>
            </div>
            <div className={styles.canvasWrap}>
              <ArmScene state={state} />
              {!state && (
                <div className={styles.canvasEmpty}>
                  <strong>{t("No live geometry", "没有实时几何")}</strong>
                  <span>
                    {t(
                      "Start the local simulation and retry. This page does not generate substitute telemetry.",
                      "启动本地模拟后重试；页面不会生成替代遥测。"
                    )}
                  </span>
                </div>
              )}
              <div className={styles.canvasLegend}>
                <span>{t("X red", "X 红")}</span>
                <span>{t("Y green", "Y 绿")}</span>
                <span>{t("Z blue", "Z 蓝")}</span>
              </div>
            </div>
            <div className={styles.liveFooter}>
              <div><span>{t("case", "案例")}</span><strong>{state?.case_id ?? "—"}</strong></div>
              <div><span>{t("step", "步数")}</span><strong>{state?.step ?? "—"}</strong></div>
              <div><span>{t("sim time", "仿真时间")}</span><strong>{state ? `${state.time.toFixed(2)} s` : "—"}</strong></div>
              <div><span>{t("stage", "阶段")}</span><strong>{state?.stage || "—"}</strong></div>
            </div>
          </div>

          <aside className={styles.controlPanel}>
            <div className={styles.panelHeading}>
              <div>
                <p className={styles.eyebrow}>{t("EPISODE CONTROL", "回合控制")}</p>
                <h3>{t("Simulation controls", "仿真控制")}</h3>
              </div>
              <div className={styles.controlBadges}>
                <span className={styles.contractBadge}>
                  {state?.observation_dim ?? selectedDefinition.observationDim} obs /
                  {" "}{state?.action_dim ?? selectedDefinition.actionDim} act
                </span>
                <span className={styles.hardwareLockInline}>
                  {t("HARDWARE · LOCKED", "硬件 · 已锁定")}
                </span>
              </div>
            </div>

            <label className={styles.seedField}>
              <span>{t("Reset seed", "重置种子")}</span>
              <input
                type="number"
                min="0"
                max="2147483647"
                step="1"
                value={seed}
                onChange={(event) => setSeed(Math.max(0, Math.min(2_147_483_647, Number(event.target.value) || 0)))}
              />
            </label>
            <button className={styles.primaryButton} disabled={!available || busy} onClick={() => void reset()}>
              <Icon>↺</Icon>{t("Reset selected case", "重置所选案例")}
            </button>

            <div className={styles.controlSection}>
              <div className={styles.sectionTitle}>
                <div>
                  <span>{t("Manual action", "手动动作")}</span>
                  <small>{t("Normalized joint velocity · −1 to 1", "归一化关节速度 · −1 到 1")}</small>
                </div>
                <button
                  className={styles.smallButton}
                  onClick={() => setAction(new Array(currentActionDim).fill(0))}
                >
                  {t("Zero", "归零")}
                </button>
              </div>
              <div className={styles.sliders}>
                {visibleAction.map((value, index) => (
                  <label key={index}>
                    <span>{index < 6 ? JOINT_NAMES[index] : `right_${JOINT_NAMES[index - 6]}`}</span>
                    <input
                      type="range"
                      min="-1"
                      max="1"
                      step="0.01"
                      value={value}
                      onChange={(event) => updateAction(index, Number(event.target.value))}
                    />
                    <output>{value.toFixed(2)}</output>
                  </label>
                ))}
              </div>
              <button
                className={styles.secondaryButton}
                disabled={!available || busy || caseMismatch}
                onClick={() => void runCommand("step", { action: visibleAction })}
              >
                <Icon>›</Icon>{t("Step", "单步执行")}
              </button>
            </div>

            <div className={styles.teacherBox}>
              <div className={styles.teacherHeader}>
                <div>
                  <strong>{t("Teacher reference controller", "Teacher 参考控制器")}</strong>
                  <span>
                    {t(
                      "Deterministic IK / FSM; not PPO and does not start training",
                      "确定性 IK / FSM，不是 PPO，也不会启动训练"
                    )}
                  </span>
                </div>
                <span className={styles.teacherBadge}>{t("NOT PPO", "非 PPO")}</span>
              </div>
              <label className={styles.ticksField}>
                <span>{t("Advance by", "每次推进")}</span>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={teacherTicks}
                  onChange={(event) => setTeacherTicks(Math.max(1, Math.min(50, Number(event.target.value) || 1)))}
                />
                <span>{t("ticks", "步")}</span>
              </label>
              <div className={styles.teacherActions}>
                <button
                  className={styles.secondaryButton}
                  disabled={!available || busy || caseMismatch}
                  onClick={() => void runCommand("teacher", { ticks: teacherTicks })}
                >
                  <Icon>›</Icon>{t("Advance Teacher", "推进 Teacher")}
                </button>
                <button
                  className={playing ? styles.stopButton : styles.playButton}
                  disabled={!available || caseMismatch || Boolean(state?.terminated || state?.truncated)}
                  onClick={() => setPlaying((current) => !current)}
                >
                  <Icon>{playing ? "■" : "▶"}</Icon>
                  {playing ? t("Pause", "暂停") : t("Play continuously", "连续播放")}
                </button>
              </div>
            </div>
          </aside>
        </section>

        <section className={styles.infoGrid}>
          <article>
            <p className={styles.eyebrow}>{t("LIVE CONTRACT", "实时契约")}</p>
            <h3>{t("State and control sources", "状态与控制来源")}</h3>
            <dl className={styles.factList}>
              <div><dt>{t("contract", "契约")}</dt><dd>{state?.contract || t("Not connected", "未连接")}</dd></div>
              <div><dt>{t("controller", "控制器")}</dt><dd>{state ? controllerLabel(state.controller, t) : t("Not reported", "未报告")}</dd></div>
              <div><dt>{t("joints", "关节")}</dt><dd>{state ? state.joints.map((value) => value.toFixed(3)).join(", ") : t("Not reported", "未报告")}</dd></div>
              <div>
                <dt>{t("hardware", "硬件")}</dt>
                <dd>
                  {t(
                    "LOCKED · simulation HTTP only · Unix transport not integrated",
                    "已锁定 · 仅限仿真 HTTP · 未集成 Unix 传输"
                  )}
                </dd>
              </div>
            </dl>
          </article>
          <article>
            <p className={styles.eyebrow}>{t("MEASURED ONLY", "仅实测数据")}</p>
            <h3>{t("Live metrics", "实时指标")}</h3>
            <PrimaryMetrics metrics={state?.metrics ?? {}} t={t} />
            {state && Object.keys(state.metrics).length > 0 && (
              <details className={styles.rawDetails}>
                <summary>{t("Raw live metrics", "完整实时指标")}</summary>
                <Metrics metrics={state.metrics} t={t} />
              </details>
            )}
          </article>
          <article id="bom">
            <p className={styles.eyebrow}>{t("DESIGN REFERENCES", "设计参考")}</p>
            <h3>{t("BOM and classroom boundaries", "BOM 与课堂边界")}</h3>
            <p className={styles.bodyCopy}>
              {t(
                "A single arm uses five pose joints plus one linked gripper; dual arms use 12 actions. Independent power, an independent bus, and a hardware emergency stop remain requirements for a future bench setup.",
                "单臂设计为 5 个姿态关节加 1 个联动夹爪；双臂为 12 个动作。独立供电、独立总线与硬件急停仍是未来台架门槛。"
              )}
            </p>
            <div className={styles.pathList}>
              <span><code>docs/robot-arm-design/IMPLEMENTATION.md</code><small>{t("Actual implementation", "实际实现")}</small></span>
              <span><code>docs/robot-arm-design/repair-v3/RESULTS.zh-CN.md</code><small>{t("Strict verification results", "严格验证结果")}</small></span>
              <span><code>docs/robot-arm-design/CLASSROOM.zh-CN.pdf</code><small>{t("Classroom materials", "课堂资料")}</small></span>
              <span><code>docs/robot-arm-design/EXPERIMENTS.md</code><small>{t("Design notes, not run evidence", "设计说明，不是运行证据")}</small></span>
            </div>
          </article>
        </section>

        <section className={styles.runsSection} id="evidence">
          <div className={styles.sectionHeading}>
            <div>
              <p className={styles.eyebrow}>{t("SAVED EVIDENCE", "已保存证据")}</p>
              <h2>{t("Live service run history", "实时服务运行记录")}</h2>
              <p>
                {t(
                  "By default, only records for the selected case with matching sources, model, checkpoint, and context are shown.",
                  "默认显示所选案例且来源、模型、检查点与上下文均匹配的记录。"
                )}
              </p>
            </div>
            <div className={styles.runFilters}>
              <label>
                <input
                  type="checkbox"
                  checked={showRunHistory}
                  onChange={(event) => setShowRunHistory(event.target.checked)}
                />
                <span>{t("Show history and other cases", "显示历史/其他案例")}</span>
              </label>
              <button
                className={styles.secondaryButton}
                disabled={!available || busy}
                onClick={() => void loadRuns().catch((cause) => markOffline(cause instanceof Error ? cause.message : "Cannot load runs."))}
              >
                <Icon>↻</Icon>{t("Refresh records", "刷新记录")}
              </button>
            </div>
          </div>
          {!runs.length ? (
            <div className={styles.emptyRuns}>
              <strong>
                {available ? t("No run records yet", "暂无运行记录") : t("Backend unavailable", "后端不可用")}
              </strong>
              <span>
                {available
                  ? t(
                    "Evidence generated by Teacher, BC, or PPO will appear here.",
                    "Teacher、BC 或 PPO 生成证据后会出现在这里。"
                  )
                  : t(
                    "No sample data or fabricated success history is used.",
                    "没有使用示例数据或虚构成功历史。"
                  )}
              </span>
            </div>
          ) : !visibleRuns.length ? (
            <div className={styles.emptyRuns}>
              <strong>{t("No current matching records for the selected case", "所选案例没有当前匹配记录")}</strong>
              <span>
                {t(
                  "Show history and other cases to view all evidence returned by the backend.",
                  "可显示历史/其他案例查看后端返回的全部证据。"
                )}
              </span>
            </div>
          ) : (
            <div className={styles.runsTableWrap}>
              <table>
                <thead>
                  <tr>
                    <th>{t("Controller", "控制器")}</th>
                    <th>{t("Seed", "种子")}</th>
                    <th>{t("Success", "成功率")}</th>
                    <th>{t("Verdict / Provenance", "结论 / 来源")}</th>
                    <th>{t("Video", "视频")}</th>
                    <th>{t("Details", "详情")}</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleRuns.map((run) => {
                    const verdict = runVerdict(run, t);
                    const provenanceMatched = hasCurrentProvenance(run.metrics);
                    return (
                    <tr key={`${run.case_id}-${run.run_id}-${run.path}`}>
                      <td><strong>{controllerLabel(run.controller, t)}</strong><code>{run.case_id}</code></td>
                      <td><code>{runSeed(run.run_id)}</code></td>
                      <td>
                        <strong>
                          {typeof run.metrics.successes === "number" &&
                          typeof run.metrics.episodes === "number"
                            ? `${run.metrics.successes}/${run.metrics.episodes}`
                            : t("Not measured", "未测")}
                        </strong>
                      </td>
                      <td>
                        <span className={verdict.className}>{verdict.label}</span>
                        <span className={provenanceMatched ? styles.provenanceMatched : styles.provenanceStale}>
                          {provenanceMatched
                            ? t("PROVENANCE MATCHED", "来源匹配")
                            : t("PROVENANCE UNMATCHED", "来源不匹配")}
                        </span>
                      </td>
                      <td>
                        {run.video_path ? (
                          <button className={styles.videoButton} onClick={() => setSelectedVideo(run.video_path ?? null)}>
                            <Icon>▶</Icon>{t("View", "查看")}
                          </button>
                        ) : <span className={styles.muted}>{t("No video", "无视频")}</span>}
                      </td>
                      <td>
                        <details className={styles.rawDetails}>
                          <summary>{t("Full metrics", "完整指标")}</summary>
                          <div className={styles.runIdentity}>
                            <code>{run.run_id}</code>
                            <small>{run.path}</small>
                          </div>
                          <Metrics metrics={run.metrics} t={t} compact />
                        </details>
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {selectedVideo && (
            <div className={styles.videoPanel}>
              <div className={styles.videoHeading}>
                <div><strong>{t("Source-bound artifact", "来源绑定制品")}</strong><code>{selectedVideo}</code></div>
                <button
                  className={styles.iconButton}
                  onClick={() => setSelectedVideo(null)}
                  aria-label={t("Close video", "关闭视频")}
                >
                  ×
                </button>
              </div>
              <video
                controls
                preload="metadata"
                src={`/api/arm/artifact?path=${encodeURIComponent(selectedVideo)}`}
                aria-label={t("Selected run video", "所选运行视频")}
              />
            </div>
          )}
        </section>
        </>}
      </main>
    </div>
  );
}
