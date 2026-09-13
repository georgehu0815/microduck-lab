"use client";

// 🎥 capture panel (top-center): select a duck, hit record, get content.
// The flow: camera glides to a ¾ shot of the duck (RecordCamera in
// Viewer.tsx), MediaRecorder captures the WebGL canvas — DOM labels and
// panels are not part of the canvas, so takes come out clean — then the lab
// server converts the upload to mp4 + gif (POST /captures) and the panel
// offers both as downloads.

import { useEffect, useRef, useState } from "react";
import { LAB_HTTP, type LabClient } from "@/lib/lab";
import { useSelectedDuck } from "@/lib/select";
import { useHudRight } from "@/lib/ui";
import {
  captureDone,
  captureError,
  captureFraming,
  captureProcessing,
  captureRecording,
  captureReset,
  getCapture,
  getCaptureCanvas,
  getFramesPushed,
  hasCaptureTrack,
  setCaptureTrack,
  snapshotNow,
  useCapture,
} from "@/lib/record";
import { useLanguage } from "./LanguageProvider";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** Camera glide before the recorder rolls (matches RecordCamera's damping —
 *  the shot has settled by then, so takes don't open with a swish). */
const FRAMING_MS = 1200;

/** Duck names carry emoji/spaces — reduce to a safe filename stem (mirrors
 *  the server's capture_slug, so shots and takes sort together). */
function slug(s: string): string {
  return (
    s
      .replace(/[^A-Za-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .toLowerCase()
      .slice(0, 40)
      // Mirror capture_slug exactly: it strips again AFTER truncating, and
      // strips leading _ as well (a stem starting with _ is one the server's
      // own /captures route then refuses to serve). Without both, a 📷 shot
      // and a 🎥 take of the same duck get different stems.
      .replace(/-+$/, "")
      .replace(/^[_-]+/, "") || "duck"
  );
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}` +
    `-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`
  );
}

function pickMime(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  const prefs = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4", // Safari
  ];
  return prefs.find((m) => MediaRecorder.isTypeSupported(m)) ?? null;
}

// No `left` here — the component computes it per render: centered, but never
// under the top-left HUD panel (see the layout block in RecordPanel).
const panelStyle: React.CSSProperties = {
  position: "absolute",
  top: 10,
  zIndex: 20,
  display: "flex",
  alignItems: "center",
  gap: 8,
  background: "rgba(14, 16, 20, 0.86)",
  border: "1px solid rgba(255,255,255,0.12)",
  borderRadius: 8,
  padding: "6px 10px",
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  backdropFilter: "blur(6px)",
};

const btnStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 6,
  color: "#d8dee8",
  fontFamily: mono,
  fontSize: 12,
  padding: "3px 10px",
  cursor: "pointer",
};

const linkStyle: React.CSSProperties = {
  ...btnStyle,
  textDecoration: "none",
  color: "#7db8d8",
};

export function RecordPanel({
  clientRef,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
}) {
  const { t } = useLanguage();
  const selected = useSelectedDuck();
  const cap = useCapture();
  const recRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timersRef = useRef<number[]>([]);
  const [, bump] = useState(0); // re-render tick for the elapsed timer

  // Layout inputs for the HUD-dodging `left` computed at the bottom. HUD
  // geometry is viewport-relative, while this panel is viewer-relative.
  const hudRight = useHudRight();
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [panelW, setPanelW] = useState(220);
  const [hostRect, setHostRect] = useState({ left: 0, width: 1200 });
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const host = el.parentElement;
    const measure = () => {
      setPanelW(el.offsetWidth);
      const rect = host?.getBoundingClientRect();
      if (rect) setHostRect({ left: rect.left, width: rect.width });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    if (host) ro.observe(host);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (cap.phase !== "recording") return;
    const id = setInterval(() => bump((t) => t + 1), 250);
    return () => clearInterval(id);
  }, [cap.phase]);

  const clearTimers = () => {
    timersRef.current.forEach((t) => clearTimeout(t));
    timersRef.current = [];
  };

  const fail = (msg: string) => {
    captureError(msg);
    pushToast(`🎥 ${msg}`);
  };

  const start = () => {
    if (["framing", "recording", "processing"].includes(getCapture().phase)) return;
    const duckName =
      clientRef.current?.frame?.ducks.find((d) => d.id === selected)?.name ??
      "duck-lab";
    captureFraming(selected);
    if (selected) {
      timersRef.current.push(
        window.setTimeout(() => beginRecording(duckName), FRAMING_MS)
      );
    } else {
      beginRecording(duckName);
    }
  };

  const beginRecording = (duckName: string) => {
    if (getCapture().phase !== "framing") return; // cancelled during the glide
    const canvas = getCaptureCanvas();
    const mime = pickMime();
    if (!canvas) return fail(t("no canvas to record", "没有可录制的画布"));
    if (!mime)
      return fail(
        t(
          "this browser can't record video (no MediaRecorder)",
          "此浏览器无法录制视频（不支持 MediaRecorder）"
        )
      );
    // captureStream(0) + an explicit requestFrame() per rendered frame (the
    // pump lives in RecordCamera's useFrame) — automatic capture rides the
    // compositor and delivered near-empty webms whenever the tab was
    // throttled. Browsers without requestFrame (Safari) fall back to auto.
    let stream: MediaStream;
    let pumpTrack: { requestFrame: () => void } | null = null;
    try {
      stream = canvas.captureStream(0);
      const t = stream.getVideoTracks()[0] as unknown as {
        requestFrame?: () => void;
      };
      if (typeof t?.requestFrame === "function") {
        pumpTrack = t as { requestFrame: () => void };
      } else {
        stream = canvas.captureStream(30);
      }
    } catch (e) {
      return fail(t(`canvas capture failed: ${e}`, `画布捕获失败：${e}`));
    }
    setCaptureTrack(pumpTrack);
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, {
        mimeType: mime,
        videoBitsPerSecond: 12_000_000,
      });
    } catch (e) {
      return fail(t(`recorder failed to start: ${e}`, `录制器启动失败：${e}`));
    }
    chunksRef.current = [];
    rec.ondataavailable = (e) => {
      if (e.data.size) chunksRef.current.push(e.data);
    };
    rec.onstop = () =>
      upload(duckName, new Blob(chunksRef.current, { type: mime }));
    rec.onerror = () => fail(t("recorder error mid-take", "录制过程中发生错误"));
    recRef.current = rec;
    rec.start(250);
    captureRecording();
  };

  const stop = () => {
    const rec = recRef.current;
    if (rec?.state !== "recording") return;
    clearTimers();
    captureProcessing(); // before .stop(): onstop checks the phase
    rec.stop();
  };

  const upload = async (duckName: string, blob: Blob) => {
    recRef.current = null;
    const pushed = getFramesPushed();
    const pumped = hasCaptureTrack();
    setCaptureTrack(null);
    if (getCapture().phase !== "processing") return; // cancelled
    if (!blob.size)
      return fail(t("empty recording — nothing captured", "录制为空，未捕获任何内容"));
    // A handful of frames means the scene never rendered during the take
    // (hidden/throttled tab) — a 0.1 s "video" out of ffmpeg would only
    // confuse; say what actually happened instead.
    if (pumped && pushed < 5)
      return fail(
        t(
          "scene barely rendered during the take — keep the tab visible while recording",
          "录制期间场景几乎没有渲染，请在录制时保持此标签页可见"
        )
      );
    try {
      const res = await fetch(
        `${LAB_HTTP}/captures?name=${encodeURIComponent(duckName)}`,
        { method: "POST", body: blob }
      );
      if (!res.ok) {
        const detail = (await res.json().catch(() => null))?.detail;
        throw new Error(detail ?? `HTTP ${res.status}`);
      }
      const result = await res.json();
      captureDone(result);
      pushToast(
        t(
          `🎥 saved ${result.name} (mp4 + gif) in captures/`,
          `🎥 已将 ${result.name}（mp4 + gif）保存到 captures/`
        )
      );
    } catch (e) {
      const detail = e instanceof Error ? e.message : e;
      fail(t(`capture failed: ${detail}`, `捕获失败：${detail}`));
    }
  };

  const cancel = () => {
    clearTimers();
    const rec = recRef.current;
    if (rec && rec.state !== "inactive") {
      rec.onstop = null; // discard, don't upload
      rec.stop();
    }
    recRef.current = null;
    setCaptureTrack(null);
    captureReset();
  };

  // Unmount: drop timers and a still-rolling recorder without uploading.
  useEffect(() => () => cancel(), []); // eslint-disable-line react-hooks/exhaustive-deps

  // 📷 is instant and client-side: name the file after the selected duck (or
  // the whole lab) and capture SYNCHRONOUSLY — the download must fire inside
  // this click's user gesture (see lib/record.ts).
  const snap = () => {
    const duckName = selected
      ? clientRef.current?.frame?.ducks.find((d) => d.id === selected)?.name
      : null;
    if (!snapshotNow(`${slug(duckName ?? "duck-lab")}-${stamp()}`))
      pushToast(t("📷 scene still loading — try again in a moment", "📷 场景仍在加载，请稍后重试"));
  };

  let content: React.ReactNode = null;
  if (cap.phase === "framing" || cap.phase === "recording") {
    const secs =
      cap.recordingSince > 0
        ? Math.floor((Date.now() - cap.recordingSince) / 1000)
        : 0;
    content = (
      <>
        {cap.phase === "framing" ? (
          <span>{t("🎥 framing…", "🎥 正在取景…")}</span>
        ) : (
          <>
            <span style={{ color: "#e07a5f" }}>●</span>
            <span>{secs}s</span>
          </>
        )}
        <button style={btnStyle} onClick={cancel} title={t("discard the take", "丢弃本次录制")}>
          ✕
        </button>
      </>
    );
  } else if (cap.phase === "processing") {
    content = <span>{t("⏳ making mp4 + gif…", "⏳ 正在生成 mp4 + gif…")}</span>;
  } else if (cap.phase === "done" && cap.result) {
    content = (
      <>
        <span>🎥 {cap.result.name}</span>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.mp4}`}>
          ⬇ mp4 {Math.max(1, Math.round(cap.result.mp4Kb / 1024))}MB
        </a>
        <a style={linkStyle} href={`${LAB_HTTP}${cap.result.gif}`}>
          ⬇ gif {Math.max(1, Math.round(cap.result.gifKb / 1024))}MB
        </a>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  } else if (cap.phase === "error") {
    content = (
      <>
        <span style={{ color: "#e07a5f" }}>
          ⚠ {cap.error ?? t("capture failed", "捕获失败")}
        </span>
        <button style={btnStyle} onClick={captureReset}>
          ✕
        </button>
      </>
    );
  }

  // Centered at the top — but never UNDER the HUD: a wide duck-lab panel
  // (long duck names) used to reach right beneath these buttons. The HUD
  // publishes its right edge (useHudRight); slide right of it when centering
  // would collide, and keep a margin from the right edge as a backstop.
  const centered = (hostRect.width - panelW) / 2;
  const localHudRight = Math.max(0, hudRight - hostRect.left);
  const left = Math.round(
    Math.min(
      Math.max(centered, localHudRight + 12),
      Math.max(12, hostRect.width - panelW - 12)
    )
  );
  return (
    <div ref={wrapRef} data-policy-ui style={{ ...panelStyle, left }}>
      <button
        style={btnStyle}
        onClick={snap}
        title={t(
          "download a PNG of the current view (selection ring hidden for the shot)",
          "下载当前视图的 PNG（截图时隐藏选择环）"
        )}
        aria-label={t("Take screenshot", "截取屏幕")}
      >
        {t("📷 shot", "📷 截图")}
      </button>
      <button
        style={{ ...btnStyle, opacity: cap.phase === "processing" ? 0.5 : 1 }}
        onClick={cap.phase === "recording" ? stop : cap.phase === "framing" ? cancel : start}
        disabled={cap.phase === "processing"}
        aria-label={
          cap.phase === "recording" || cap.phase === "framing"
            ? t("Stop recording", "停止录制")
            : t("Start video recording", "开始视频录制")
        }
        aria-pressed={cap.phase === "recording" || cap.phase === "framing"}
        title={
          cap.phase === "recording"
            ? t("stop recording and save mp4 + gif", "停止录制并保存 mp4 + gif")
            : cap.phase === "framing"
              ? t("cancel camera framing", "取消相机取景")
              : selected
                ? t("record the selected duck until you click stop", "录制选中的小鸭，直到点击停止")
                : t("record the current MuJoCo view until you click stop", "录制当前 MuJoCo 视图，直到点击停止")
        }
      >
        {cap.phase === "recording" || cap.phase === "framing"
          ? t("■ stop", "■ 停止")
          : t("🎥 record", "🎥 录制")}
      </button>
      {content}
    </div>
  );
}
