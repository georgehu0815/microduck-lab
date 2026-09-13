"use client";

import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useCallback, useEffect, useState } from "react";
import { IS_STATIC_EXPORT } from "@/lib/static-assets";
import {
  parseWingPodLiveState,
  type WingPodLiveGeom,
  type WingPodLiveState,
} from "@/lib/wingpod-live";
import { useLanguage } from "./LanguageProvider";
import styles from "./WingPodPage.module.css";

function Shape({ geom }: { geom: WingPodLiveGeom }) {
  const quaternion: [number, number, number, number] = [geom.quat[1], geom.quat[2], geom.quat[3], geom.quat[0]];
  const material = (
    <meshStandardMaterial
      color={`rgb(${geom.rgba.slice(0, 3).map((value) => Math.round(value * 255)).join(",")})`}
      transparent={geom.rgba[3] < 1}
      opacity={geom.rgba[3]}
      roughness={0.72}
      metalness={0.03}
    />
  );
  if (geom.type === "box") {
    return <mesh position={geom.pos} quaternion={quaternion}><boxGeometry args={geom.size.map((value) => value * 2) as [number, number, number]} />{material}</mesh>;
  }
  if (geom.type === "sphere" || geom.type === "ellipsoid") {
    return <mesh position={geom.pos} quaternion={quaternion} scale={geom.type === "ellipsoid" ? geom.size : undefined}><sphereGeometry args={[geom.type === "sphere" ? geom.size[0] : 1, 16, 12]} />{material}</mesh>;
  }
  if (geom.type === "capsule") {
    return <group position={geom.pos} quaternion={quaternion}><mesh rotation={[Math.PI / 2, 0, 0]}><capsuleGeometry args={[geom.size[0], geom.size[1] * 2, 6, 12]} />{material}</mesh></group>;
  }
  return <group position={geom.pos} quaternion={quaternion}><mesh rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[geom.size[0], geom.size[0], geom.size[1] * 2, 16]} />{material}</mesh></group>;
}

function Scene({ state }: { state: WingPodLiveState | null }) {
  return (
    <Canvas dpr={[1, 1.5]} gl={{ antialias: true, powerPreference: "high-performance" }} camera={{ position: [0.52, 0.34, 0.38], fov: 40, near: 0.01, far: 10 }}>
      <color attach="background" args={["#e9eee8"]} />
      <hemisphereLight intensity={1.15} groundColor="#adb8ad" color="#ffffff" />
      <directionalLight position={[2, 4, 3]} intensity={1.7} />
      <gridHelper args={[2, 24, "#9ca9a0", "#d3d9d3"]} position={[0, -0.002, 0]} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {state?.geoms.map((geom, index) => <Shape key={`${geom.name}-${index}`} geom={geom} />)}
      </group>
      <OrbitControls makeDefault target={[0.12, 0.1, 0.04]} minDistance={0.25} maxDistance={2.5} maxPolarAngle={Math.PI / 2 - 0.02} />
    </Canvas>
  );
}

async function requestState(path: string, init?: RequestInit) {
  const response = await fetch(`/api/arm/${path}`, { ...init, cache: "no-store" });
  if (!response.ok) throw new Error(`WingPod live API failed (${response.status})`);
  const state = parseWingPodLiveState(await response.json());
  if (!state) throw new Error("WingPod live state failed contract validation");
  return state;
}

export default function WingPodLive() {
  const { t } = useLanguage();
  const [state, setState] = useState<WingPodLiveState | null>(null);
  const [seed, setSeed] = useState(0);
  const [running, setRunning] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const send = useCallback(async (path: "wingpod-reset" | "wingpod-teacher", body: object) => {
    const next = await requestState(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setState(next);
    setError(null);
    return next;
  }, []);

  useEffect(() => {
    if (IS_STATIC_EXPORT) return;
    const controller = new AbortController();
    requestState("wingpod-state", { signal: controller.signal })
      .then(setState)
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "WingPod live backend unavailable");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!running || IS_STATIC_EXPORT) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const advance = async () => {
      try {
        const next = await send("wingpod-teacher", { ticks: 2 });
        if (cancelled) return;
        if (next.terminated) setRunning(false);
        else timer = setTimeout(advance, 0);
      } catch (reason) {
        if (!cancelled) {
          setRunning(false);
          setError(reason instanceof Error ? reason.message : "WingPod live backend unavailable");
        }
      }
    };
    void advance();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [running, send]);

  const command = async (path: "wingpod-reset" | "wingpod-teacher", body: object) => {
    setBusy(true);
    try {
      await send(path, body);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "WingPod live backend unavailable");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={styles.liveSection} id="live" aria-labelledby="wingpod-live-title">
      <div className={styles.liveHeading}>
        <div>
          <p className={styles.kicker}>{t("LOCAL MUJOCO · 50 HZ", "本地 MUJOCO · 50 HZ")}</p>
          <h2 id="wingpod-live-title">{t("Run the tennis return live", "实时运行网球归桶")}</h2>
        </div>
        <span className={styles.liveStatus} data-running={running}>
          {IS_STATIC_EXPORT
            ? t("Recorded mode on GitHub Pages", "GitHub Pages 录制模式")
            : error
              ? t("Backend offline", "后端离线")
              : running
                ? t("MuJoCo running", "MuJoCo 运行中")
                : t("MuJoCo ready", "MuJoCo 已就绪")}
        </span>
      </div>
      <p className={styles.liveDisclosure}>
        {t(
          "The local Arm Lab advances the real WingPod MuJoCo model with the authored IK/FSM tennis controller. This is live simulation, not a trained or vision-guided policy.",
          "本地机械臂实验室使用编写的 IK/FSM 网球控制器推进真实 WingPod MuJoCo 模型。这是实时仿真，并非训练或视觉引导策略。",
        )}
      </p>
      {IS_STATIC_EXPORT ? (
        <div className={styles.liveStatic}>
          <strong>{t("Live MuJoCo requires the local Arm Lab", "实时 MuJoCo 需要本地机械臂实验室")}</strong>
          <span>{t("The verified 50.84-second replay remains available below.", "下方仍可查看已验证的 50.84 秒回放。")}</span>
        </div>
      ) : (
        <div className={styles.liveWorkspace}>
          <div className={styles.liveCanvas} data-testid="wingpod-live-canvas"><Scene state={state} /></div>
          <aside className={styles.liveControls}>
            <div className={styles.liveButtons}>
              <button type="button" onClick={() => setRunning((value) => !value)} disabled={!state || busy || state.terminated}>
                <span aria-hidden="true">{running ? "Ⅱ" : "▶"}</span>{running ? t("Pause", "暂停") : t("Run", "运行")}
              </button>
              <button type="button" onClick={() => void command("wingpod-teacher", { ticks: 1 })} disabled={!state || busy || running || state.terminated}>
                <span aria-hidden="true">›</span>{t("Step", "单步")}
              </button>
            </div>
            <label className={styles.seedControl}>
              <span>{t("Reset seed", "重置种子")}</span>
              <input type="number" min="0" max="2147483647" step="1" value={seed} onChange={(event) => setSeed(Number(event.target.value))} />
              <button type="button" onClick={() => { setRunning(false); void command("wingpod-reset", { seed }); }} disabled={busy || !Number.isSafeInteger(seed) || seed < 0}>
                ↻ <span>{t("Reset", "重置")}</span>
              </button>
            </label>
            {error && <p className={styles.liveError}>{error}</p>}
            <dl className={styles.liveMetrics}>
              <div><dt>{t("Phase", "阶段")}</dt><dd>{state?.stage ?? "—"}</dd></div>
              <div><dt>{t("Simulation time", "仿真时间")}</dt><dd>{state ? `${state.time.toFixed(2)} s` : "—"}</dd></div>
              <div><dt>{t("Ball in bin", "球在桶内")}</dt><dd>{state ? (state.metrics.ball_inside_bin ? t("Yes", "是") : t("No", "否")) : "—"}</dd></div>
              <div><dt>{t("TCP distance", "末端距离")}</dt><dd>{state ? `${(state.metrics.tcp_ball_distance_m * 1000).toFixed(1)} mm` : "—"}</dd></div>
            </dl>
            <small>{t("Controller: authored IK/FSM · 15 actuators · hardware locked", "控制器：编写的 IK/FSM · 15 执行器 · 硬件锁定")}</small>
          </aside>
        </div>
      )}
    </section>
  );
}