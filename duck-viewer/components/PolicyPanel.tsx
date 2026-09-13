"use client";

// Policy palette (top-right): every brain the lab server knows about, as
// draggable chips grouped by origin. Drop a chip on a duck — or click to arm
// it, then click a duck — to hot-swap that duck's policy over the WebSocket.
// Drop (or armed-click) on EMPTY floor instead — farther than the assign
// radius from every duck — to spawn a fresh duck running that policy.
// Pointer events only (no HTML5 DnD): a ghost chip follows the cursor and the
// Canvas-side AssignTargets helper (Viewer.tsx) picks/highlights the nearest
// duck via lib/assign's shared store. Styling mirrors Hud.tsx.
// Hovering one of OUR runs reveals a ✕ that deletes that run's training data
// from disk (a chain deletes all its stages at once). It is irreversible, so
// it always goes through the confirm dialog below — never a bare click.

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  deleteRun,
  fetchPolicies,
  formatBytes,
  isRunPolicy,
  LAB_HTTP,
  loadTeachRun,
  type LabClient,
  type Policy,
} from "@/lib/lab";
import { assignDrag, clearAssignDrag, isCanvasAt, nearestDuck } from "@/lib/assign";
import { loadJSON, saveJSON } from "@/lib/persist";
import { modalIsOpen, setPolicyOpen, useTeachHeight } from "@/lib/ui";
import { Tip } from "./TeachPanel";
import { pushToast } from "./Toasts";
import { fetchSavedRuns, type SavedRun } from "@/lib/studio-run";
import { useLanguage } from "./LanguageProvider";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

const GROUPS: { key: Policy["group"]; title: string; zh: string }[] = [
  { key: "studio", title: "Studio policies", zh: "Studio 策略" },
  { key: "runs", title: "Our runs", zh: "我们的训练" },
  { key: "checkpoints", title: "Checkpoints", zh: "检查点" },
  { key: "pollen", title: "Pollen (shipped)", zh: "Pollen（随附）" },
];

const DRAG_THRESHOLD_PX = 5; // less movement than this counts as a click

/** "2h ago"-style label from an epoch-seconds timestamp. Computed at render;
 *  the panel re-renders once a minute so it can't go stale. */
function relTime(epochS: number, t: (english: string, chinese: string) => string): string {
  const s = Math.max(0, Date.now() / 1000 - epochS);
  if (s < 60) return t("just now", "刚刚");
  if (s < 3600) return t(`${Math.floor(s / 60)}m ago`, `${Math.floor(s / 60)} 分钟前`);
  if (s < 86400) return t(`${Math.floor(s / 3600)}h ago`, `${Math.floor(s / 3600)} 小时前`);
  return t(`${Math.floor(s / 86400)}d ago`, `${Math.floor(s / 86400)} 天前`);
}

/** Free-text filter match. Case-insensitive AND over whitespace-separated
 *  terms, tested against every name a user might type: the chip label, the
 *  curriculum chain it belongs to and the full policy id. Because `chain`
 *  lives on each stage policy, a query that hits the chain name keeps all
 *  of that chain's stages — "back" shows the whole backflip family. */
function matchesQuery(p: Policy, terms: string[]): boolean {
  if (!terms.length) return true;
  const hay = `${p.label} ${p.chain ?? ""} ${p.id}`.toLowerCase();
  return terms.every((t) => hay.includes(t));
}

/** "Our runs" rows: curriculum chains (same teach-…-<hash> prefix, -sN
 *  suffixes) fold into one family of compact stage chips, positioned by
 *  their NEWEST stage (the list arrives newest-first from the server, so
 *  first-seen = newest); everything else stays a single chip + time. */
type RunRow =
  | { kind: "single"; p: Policy }
  | { kind: "chain"; chain: string; newest: number; stages: Policy[] };

function runRows(list: Policy[]): RunRow[] {
  const rows: RunRow[] = [];
  const chains = new Map<string, Extract<RunRow, { kind: "chain" }>>();
  for (const p of list) {
    if (p.chain) {
      let row = chains.get(p.chain);
      if (!row) {
        row = { kind: "chain", chain: p.chain, newest: p.mtime ?? 0, stages: [] };
        chains.set(p.chain, row);
        rows.push(row);
      }
      row.stages.push(p);
      row.newest = Math.max(row.newest, p.mtime ?? 0);
    } else {
      rows.push({ kind: "single", p });
    }
  }
  for (const row of chains.values())
    row.stages.sort((a, b) => (a.stage ?? 0) - (b.stage ?? 0));
  return rows;
}

function Chip({
  policy,
  display,
  armed,
  accent,
  title,
  onDown,
  onMove,
  onUp,
  onDouble,
}: {
  policy: Policy;
  /** Chip text override (stage chips show "s2"); drag ghost, toasts and
   *  tooltips keep the full policy.label so nothing ambiguous ships over
   *  the wire. */
  display?: string;
  armed: boolean;
  /** Warm-tinted styling for the chain-level "whole trick" chip, so it
   *  stands out from the cool single-stage chips beside it. */
  accent?: boolean;
  /** Native-tooltip override — "" suppresses it entirely, for chips whose
   *  explainer is a styled <Tip> wrapper instead. */
  title?: string;
  onDown: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onMove: (e: React.PointerEvent<HTMLButtonElement>) => void;
  onUp: (e: React.PointerEvent<HTMLButtonElement>) => void;
  /** Double-click shortcut: spawn a fresh duck running this policy — same as
   *  dragging the chip to empty floor. The two single clicks inside the
   *  double-click arm then disarm (toggle), so nothing stays armed after. */
  onDouble: () => void;
}) {
  const { t } = useLanguage();
  const [hover, setHover] = useState(false);
  return (
    <button
      type="button"
      title={
        title ??
        (armed
          ? t(
              `armed — click a duck to assign ${policy.label}, or empty floor to spawn`,
              `已选中 — 点击鸭子分配 ${policy.label}，或点击空地生成新鸭子`
            )
          : t(
              `drag onto a duck to assign — double-click (or drop on empty floor) to spawn — click to arm: ${policy.label}`,
              `拖到鸭子上进行分配；双击或拖到空地可生成新鸭子；单击可选中：${policy.label}`
            ))
      }
      onPointerDown={onDown}
      onPointerMove={onMove}
      onPointerUp={onUp}
      onDoubleClick={onDouble}
      onPointerEnter={() => setHover(true)}
      onPointerLeave={() => setHover(false)}
      style={{
        background: armed ? "#2a3548" : hover ? "#232a3a" : "#1c2230",
        color: armed ? "#cfe4f5" : accent ? "#d8cfa0" : "#9fb4d8",
        border: `1px solid ${
          armed ? "#7db8d8" : accent ? "rgba(216, 198, 125, 0.45)" : "rgba(255,255,255,0.08)"
        }`,
        borderRadius: 12,
        padding: "3px 9px",
        fontFamily: mono,
        fontSize: 11,
        lineHeight: 1.4,
        cursor: "grab",
        userSelect: "none",
        touchAction: "none", // pointer-event drags must not turn into scrolls
        // Long run/checkpoint names must truncate, not spill over neighbors:
        // flex items refuse to shrink below content unless minWidth is 0.
        minWidth: 0,
        maxWidth: 186,
        boxSizing: "border-box",
        overflow: "hidden",
        textOverflow: "ellipsis",
        whiteSpace: "nowrap",
      }}
    >
      {display ?? policy.label}
    </button>
  );
}

/** What a pending delete would erase — everything the confirm dialog needs
 *  to describe it, captured when the ✕ is clicked so a background refresh
 *  can't swap the target out from under the dialog. */
interface DeleteTarget {
  /** Name sent to the server: a run dir, or a chain prefix when `chain`. */
  name: string;
  chain: boolean;
  /** Run dirs the palette knows about (a chain's exported stages). */
  runs: string[];
  bytes: number;
}

/** The hover-revealed ✕ on a run row. Kept mounted (not conditionally
 *  rendered) so it can also be reached by keyboard — focus reveals it the
 *  same way hover does. */
function DeleteBtn({
  show,
  label,
  onFocus,
  onBlur,
  onClick,
}: {
  show: boolean;
  label: string;
  onFocus: () => void;
  onBlur: () => void;
  onClick: () => void;
}) {
  const { t } = useLanguage();
  const [hover, setHover] = useState(false);
  return (
    <button
      type="button"
      aria-label={t(`delete ${label}`, `删除 ${label}`)}
      title={t(
        `delete ${label} — removes its training data from disk`,
        `删除 ${label} — 从磁盘移除其训练数据`
      )}
      onFocus={onFocus}
      onBlur={onBlur}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      // Click, not pointerdown: a stray press while dragging a chip must not
      // fire a destructive action.
      onClick={onClick}
      style={{
        flexShrink: 0,
        opacity: show ? 1 : 0,
        pointerEvents: show ? "auto" : "none",
        transition: "opacity 90ms ease-out",
        background: "none",
        border: "none",
        color: hover ? "#e07a5f" : "#8b93a3",
        cursor: "pointer",
        fontFamily: mono,
        fontSize: 11,
        lineHeight: 1,
        padding: "2px 3px",
      }}
    >
      ✕
    </button>
  );
}

/** Hover-revealed ⤓ next to a run: downloads the trained brain as .onnx.
 *  Serves the run's baked export (policy.onnx, obs normalizer included —
 *  the deployable artifact; raw checkpoints are never handed out), falling
 *  back to the live snapshot while a run is still training. A plain anchor,
 *  not fetch: the browser streams the file straight from the lab. */
function DownloadBtn({
  show,
  run,
  label,
  onFocus,
  onBlur,
}: {
  show: boolean;
  run: string;
  label: string;
  onFocus: () => void;
  onBlur: () => void;
}) {
  const { t } = useLanguage();
  const [hover, setHover] = useState(false);
  return (
    <a
      aria-label={t(`download ${label}`, `下载 ${label}`)}
      title={t(
        `download ${label} — the trained .onnx brain, ready to run`,
        `下载 ${label} — 可直接运行的已训练 .onnx 策略`
      )}
      href={`${LAB_HTTP}/runs/${encodeURIComponent(run)}/policy.onnx`}
      download
      onFocus={onFocus}
      onBlur={onBlur}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        flexShrink: 0,
        opacity: show ? 1 : 0,
        pointerEvents: show ? "auto" : "none",
        transition: "opacity 90ms ease-out",
        background: "none",
        border: "none",
        color: hover ? "#7ab87a" : "#8b93a3",
        cursor: "pointer",
        fontFamily: mono,
        fontSize: 11,
        lineHeight: 1,
        padding: "2px 3px",
        textDecoration: "none",
      }}
    >
      ⤓
    </a>
  );
}

/** Modal confirmation for an irreversible delete. Deliberately awkward to
 *  dismiss by accident: the destructive button is NOT focused (cancel is),
 *  Escape and a backdrop click both cancel, and the exact run dirs and the
 *  bytes freed are spelled out before anything is touched. */
function DeleteDialog({
  target,
  busy,
  error,
  onCancel,
  onConfirm,
}: {
  target: DeleteTarget;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const { t } = useLanguage();
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  // Focus runs ONCE per dialog: this effect must not depend on onCancel, which
  // the parent passes as an inline arrow and so re-creates on every render —
  // that re-ran this body continuously and snapped focus back to Cancel while
  // the user was tabbing to Delete.
  useEffect(() => {
    cancelRef.current?.focus();
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      // No stopPropagation: it cannot reach the scene's listener (same
      // element, registered earlier), and the data-modal backdrop below
      // already made the scene stand down — including its own
      // Escape-deselects handler.
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [onCancel]);

  const shown = target.runs.slice(0, 6);
  const btn = {
    fontFamily: mono,
    fontSize: 11,
    borderRadius: 8,
    padding: "5px 11px",
    cursor: busy ? "wait" : "pointer",
  } as const;

  return createPortal(
    // data-policy-ui: the armed-chip global pointerdown handler skips panel
    // UI — without it, clicking this dialog while a chip is armed would
    // assign/spawn a duck behind the modal.
    //
    // data-modal: the whole-keyboard gate the scene reads (lib/ui.ts). It has
    // to sit on a node that exists only while this dialog does, so it can never
    // be left armed or disarmed by mistake — Backspace would otherwise still
    // delete the selected duck behind this dialog.
    <div
      data-policy-ui
      data-modal
      role="dialog"
      aria-modal="true"
      aria-label={t(`delete ${target.name}`, `删除 ${target.name}`)}
      onClick={() => !busy && onCancel()}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1100, // above the drag ghost (100) and the tooltips (1000)
        background: "rgba(0,0,0,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backdropFilter: "blur(1.5px)",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 340,
          maxWidth: "calc(100vw - 32px)",
          background: "rgba(14, 16, 20, 0.97)",
          border: "1px solid rgba(224, 122, 95, 0.4)",
          borderRadius: 10,
          padding: "13px 15px 12px",
          color: "#e8e6e1",
          fontFamily: mono,
          fontSize: 12,
          lineHeight: 1.5,
          boxShadow: "0 10px 34px rgba(0,0,0,0.6)",
        }}
      >
        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>
          {target.chain
            ? t(
                `🗑 delete the whole ${target.name.replace(/^teach-/, "")} chain?`,
                `🗑 删除整个 ${target.name.replace(/^teach-/, "")} 训练链？`
              )
            : t(
                `🗑 delete ${target.name.replace(/^teach-/, "")}?`,
                `🗑 删除 ${target.name.replace(/^teach-/, "")}？`
              )}
        </div>
        <div style={{ color: "#aab3c0", fontSize: 11 }}>
          {t(
            "This erases the training data on disk — the exported policy, every checkpoint and the progress log. It cannot be undone, and no policy here can be retrained back into exactly the same brain.",
            "这会删除磁盘上的训练数据，包括导出的策略、所有检查点和进度日志。此操作无法撤销，也无法通过重新训练得到完全相同的策略。"
          )}
        </div>
        <div
          style={{
            margin: "9px 0",
            padding: "6px 8px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: 7,
            fontSize: 10,
            color: "#9fb4d8",
          }}
        >
          {shown.map((r) => (
            <div
              key={r}
              style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
            >
              runs/{r}
            </div>
          ))}
          {target.runs.length > shown.length && (
            <div style={{ color: "#8b93a3" }}>
              {t(
                `…and ${target.runs.length - shown.length} more`,
                `…以及另外 ${target.runs.length - shown.length} 项`
              )}
            </div>
          )}
          <div style={{ color: "#8b93a3", marginTop: 3 }}>
            {t(
              `${formatBytes(target.bytes)} frees up`,
              `将释放 ${formatBytes(target.bytes)}`
            )}
            {target.chain
              ? t(
                  " — plus any stage of this chain that never exported a policy",
                  " — 以及此训练链中从未导出策略的阶段"
                )
              : ""}
          </div>
        </div>
        {error && (
          <div style={{ color: "#e07a5f", fontSize: 11, marginBottom: 7 }}>⚠ {error}</div>
        )}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            disabled={busy}
            style={{
              ...btn,
              background: "#1c2230",
              color: "#cfe4f5",
              border: "1px solid rgba(255,255,255,0.12)",
            }}
          >
            {t("keep it", "保留")}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={busy}
            style={{
              ...btn,
              background: busy ? "#3a2a26" : "#4a2620",
              color: "#f0c4b6",
              border: "1px solid rgba(224, 122, 95, 0.55)",
            }}
          >
            {busy ? t("deleting…", "正在删除…") : t("delete forever", "永久删除")}
          </button>
        </div>
      </div>
    </div>,
    document.body
  );
}

export function PolicyPanel({
  clientRef,
  variant = "overlay",
  active = true,
  onPoliciesChange,
  onStudioRunsChange,
  onReviewRun,
  reviewDisabled = false,
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
  variant?: "overlay" | "embedded";
  active?: boolean;
  onPoliciesChange?: (policies: Policy[]) => void;
  onStudioRunsChange?: (runs: SavedRun[]) => void;
  onReviewRun?: (experimentId: SavedRun["experimentId"], runName: string) => Promise<void>;
  reviewDisabled?: boolean;
}) {
  const { t } = useLanguage();
  const embedded = variant === "embedded";
  // Starts collapsed to its pill: the scene, not the roster, is the first
  // thing to see. The choice is persisted, so a user who opens it keeps it.
  const [open, setOpen] = useState(() => embedded || loadJSON("policyOpen", false));
  // Per-section collapse, toggled by clicking a group heading. Persisted like
  // the panel itself. An active filter overrides it (all matches stay
  // visible) — a hit hiding inside a collapsed section would look like the
  // filter found nothing.
  const [folded, setFolded] = useState<Record<string, boolean>>(() =>
    loadJSON("policyGroupsFolded", {})
  );
  const toggleFold = (key: string) =>
    setFolded((f) => {
      const next = { ...f, [key]: !f[key] };
      saveJSON("policyGroupsFolded", next);
      return next;
    });
  // Measured height of the teach panel below (0 until it reports in) — this
  // panel's list grows into whatever teach leaves free.
  const teachHeight = useTeachHeight();
  const [policies, setPolicies] = useState<Policy[]>([]);
  const [err, setErr] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);
  const refreshRequest = useRef<AbortController | null>(null);
  const [studioRuns, setStudioRuns] = useState<SavedRun[]>([]);
  const [studioError, setStudioError] = useState(false);
  const includeStudio = Boolean(onReviewRun);
  // Free-text filter over every group — typing "head" leaves only the
  // headstand chips. Deliberately NOT persisted: a hidden filter surviving a
  // reload would look like policies had vanished.
  const [query, setQuery] = useState("");
  // Bumped by the "/" shortcut; the effect that focuses the box watches it,
  // so one press both expands the panel and lands the caret in the input.
  const [focusTick, setFocusTick] = useState(0);
  // Armed chip. `showcase` disambiguates the chain-level "whole trick" chip
  // from the final-stage chip — they share a policy id (the whole trick IS
  // the final stage's policy), but must arm independently.
  const [armed, setArmed] = useState<{ id: string; showcase: boolean } | null>(null);
  const [dragging, setDragging] = useState<Policy | null>(null);
  // Run row the ✕ is showing on (hover OR keyboard focus), keyed by row key.
  const [hoverRow, setHoverRow] = useState<string | null>(null);
  // A delete waiting for confirmation — the dialog is the ONLY path to
  // deleteRun(); nothing here erases anything before the user says so.
  const [pendingDelete, setPendingDelete] = useState<DeleteTarget | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [deleteErr, setDeleteErr] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const ghostRef = useRef<HTMLDivElement | null>(null);
  const ghostSubRef = useRef<HTMLDivElement | null>(null);
  const moved = useRef(false);
  const start = useRef({ x: 0, y: 0 });

  useEffect(() => {
    if (embedded) return;
    saveJSON("policyOpen", open);
    setPolicyOpen(open); // lets the TeachPanel reclaim the vertical space
  }, [embedded, open]);

  // "/" anywhere jumps to the filter box — expanding the panel first when
  // it's collapsed, since there is nothing to focus otherwise. Ignored while
  // the user is already typing somewhere (the teach prompt, this box) or
  // holding a modifier, so it can't eat a real "/" keystroke.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (!active) return;
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey) return;
      const t = e.target as HTMLElement | null;
      if (t?.closest("input, textarea, select, [contenteditable='true']")) return;
      // A dialog owns the keyboard: "/" used to expand this panel behind the
      // overlay and pull focus into a filter box hidden by the backdrop.
      if (modalIsOpen()) return;
      e.preventDefault();
      setOpen(true);
      // The input doesn't exist yet on this tick when the panel was
      // collapsed, so hand the focus to a commit-time effect (below) rather
      // than a rAF — rAF never fires while the tab is hidden.
      setFocusTick((n) => n + 1);
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [active]);

  useEffect(() => {
    if (!focusTick) return; // nothing asked for focus yet
    searchRef.current?.focus();
    searchRef.current?.select();
  }, [focusTick]);

  const refresh = useCallback(async () => {
    refreshRequest.current?.abort();
    const request = new AbortController();
    refreshRequest.current = request;
    setRefreshing(true);
    setErr(false);
    setStudioError(false);
    try {
      const [labResult, studioResult] = await Promise.allSettled([
        fetchPolicies(request.signal),
        includeStudio ? fetchSavedRuns(request.signal) : Promise.resolve([] as SavedRun[]),
      ]);
      if (request.signal.aborted) return;
      if (labResult.status === "fulfilled") {
        setPolicies(labResult.value);
        onPoliciesChange?.(labResult.value);
      } else {
        setErr(true);
      }
      if (studioResult.status === "fulfilled") {
        setStudioRuns(studioResult.value);
        if (includeStudio) onStudioRunsChange?.(studioResult.value);
      } else {
        setStudioError(true);
      }
      setRefreshedAt(new Date());
    } catch {
      if (!request.signal.aborted) setErr(true);
    } finally {
      if (!request.signal.aborted) setRefreshing(false);
    }
  }, [includeStudio, onPoliciesChange, onStudioRunsChange]);

  // (Re)fetch whenever the panel is expanded — new training runs appear over time.
  useEffect(() => {
    if (!open || !active) return;
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => {
      window.clearTimeout(timer);
      refreshRequest.current?.abort();
    };
  }, [open, active, refresh]);

  useEffect(() => () => refreshRequest.current?.abort(), []);

  // Relative "2h ago" labels are computed at render — tick once a minute so
  // an open panel's labels age along with the runs instead of going stale.
  const [, setTimeTick] = useState(0);
  useEffect(() => {
    if (!open) return;
    const id = setInterval(() => setTimeTick((n) => n + 1), 60_000);
    return () => clearInterval(id);
  }, [open]);

  // A finished training run drops a new "Our runs" policy on disk: watch the
  // streamed training status and refetch on the training → done transition.
  const prevStatus = useRef<string | null>(null);
  useEffect(() => {
    const id = setInterval(() => {
      const status = clientRef.current?.frame?.training?.status ?? null;
      if (prevStatus.current === "training" && status === "done") refresh();
      prevStatus.current = status;
    }, 1000);
    return () => clearInterval(id);
  }, [clientRef, refresh]);

  const disarm = () => {
    clearAssignDrag();
    setArmed(null);
  };

  /** True when this exact chip (id + whole-trick-ness) is the armed one. */
  const chipArmed = (id: string, showcase = false) =>
    armed !== null && armed.id === id && armed.showcase === showcase;

  const assign = (duckId: string, policy: { id: string; label: string }, showcase = false) => {
    clientRef.current?.sendAssign(duckId, policy.id, showcase);
    pushToast(t(
      `⚡ assigning ${policy.label} → ${duckId}`,
      `⚡ 正在将 ${policy.label} 分配给 ${duckId}`
    ));
  };

  // Empty-floor drop/click: spawn a fresh duck running the policy. The server
  // confirms (or refuses, e.g. at the 20-duck cap) with its own event toast.
  const spawn = (policy: { id: string; label: string }, showcase = false) => {
    clientRef.current?.sendSpawnDuck(policy.id, showcase);
    pushToast(t(`⚡ spawning ${policy.label}…`, `⚡ 正在生成使用 ${policy.label} 的鸭子…`));
  };

  // Drop (or armed-click) on the TEACH panel: pull the run's recipe up there
  // for refinement — POST /teach/load seats it in "done" state, sliders
  // unlocked, fine-tune targeting that run. Success feedback is the server's
  // own 📋 event toast; refusals (training in flight, shipped policy) toast
  // here.
  const isTeachAt = (x: number, y: number): boolean =>
    !!document.elementFromPoint(x, y)?.closest("[data-teach-ui]");

  const loadIntoTeach = (policy: { id: string; label: string }) => {
    if (!isRunPolicy(policy.id)) {
      pushToast(t(
        `🧠 ${policy.label} is not a local teach run — use Studio review for Studio policies`,
        `🧠 ${policy.label} 不是本地教学运行 — Studio 策略请使用 Studio 评审`
      ));
      return;
    }
    if (clientRef.current?.frame?.training?.status === "training") {
      pushToast(t(
        "🎓 already teaching — stop the run before loading another",
        "🎓 正在教学 — 请先停止当前运行再加载其他策略"
      ));
      return;
    }
    loadTeachRun(policy.id)
      .then((r) => {
        if (!r.ok && r.message) pushToast(`⚠ ${r.message}`);
      })
      .catch(() => pushToast(t(
        "⚠ can't reach the lab server on :8788",
        "⚠ 无法连接 :8788 上的实验室服务器"
      )));
  };

  // Ask before erasing: the ✕ only ever OPENS the dialog. Disarms first, so a
  // chip armed a moment ago can't fire an assign at the dialog behind it.
  const askDelete = (target: DeleteTarget) => {
    disarm();
    setDeleteErr(null);
    setPendingDelete(target);
  };

  const confirmDelete = async () => {
    if (!pendingDelete || deleting) return;
    const { name, chain } = pendingDelete;
    setDeleting(true);
    setDeleteErr(null);
    try {
      await deleteRun(name, chain);
      // No local toast: the lab narrates the delete on its own event stream
      // (every viewer sees it), and two near-identical lines just look broken.
      setPendingDelete(null);
      refresh();
    } catch (e) {
      // Stay open with the server's reason (most often: it's still training),
      // so the user sees WHY nothing was deleted instead of a vanished dialog.
      setDeleteErr(e instanceof Error ? e.message : String(e));
    } finally {
      setDeleting(false);
    }
  };

  // Double-click on a chip = drag it to empty floor: spawn a duck running the
  // policy. The double-click's own single clicks armed then disarmed the chip
  // (toggle), so only the leftover shared-store fields need clearing.
  const chipDouble =
    (p: { id: string; label: string }, showcase = false) =>
    () => {
      spawn(p, showcase);
      disarm();
    };

  // --- drag gesture (pointer capture keeps every move/up on the chip) ---

  const chipDown =
    (p: Policy, showcase = false) =>
    (e: React.PointerEvent<HTMLButtonElement>) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        // synthetic/stale pointer ids can't be captured — drag still works
        // through bubbling for clicks, so don't let this abort the gesture
      }
      moved.current = false;
      start.current = { x: e.clientX, y: e.clientY };
      assignDrag.mode = "drag";
      assignDrag.policyId = p.id;
      assignDrag.policyLabel = p.label;
      assignDrag.showcase = showcase;
      assignDrag.px = e.clientX;
      assignDrag.py = e.clientY;
      setDragging(p);
    };

  const chipMove = (e: React.PointerEvent<HTMLButtonElement>) => {
    if (assignDrag.mode !== "drag") return;
    assignDrag.px = e.clientX;
    assignDrag.py = e.clientY;
    if (
      !moved.current &&
      Math.hypot(e.clientX - start.current.x, e.clientY - start.current.y) >
        DRAG_THRESHOLD_PX
    )
      moved.current = true;
    const g = ghostRef.current;
    if (g && moved.current) {
      g.style.display = "block";
      g.style.left = `${e.clientX}px`;
      g.style.top = `${e.clientY}px`;
      // Sublabel tells the drop outcome apart: duck in range → assign, open
      // canvas → spawn, anything else (panels/buttons) → cancel. Same checks
      // the chipUp drop handler runs, so the hint can't lie.
      const sub = ghostSubRef.current;
      if (sub) {
        // Teach panel first: it overlays the stage, so a duck projected
        // behind it must not win the hint the drop handler won't honour.
        sub.textContent = isTeachAt(e.clientX, e.clientY)
          ? t("drop to load into 🎓 teach", "松开放入 🎓 教学")
          : nearestDuck(e.clientX, e.clientY)
            ? t("drop to assign", "松开以分配")
            : isCanvasAt(e.clientX, e.clientY)
              ? t("drop to spawn", "松开以生成")
              : t("release to cancel", "松开以取消");
      }
    }
  };

  // The ghost is shown/positioned imperatively (chipMove), so it must be
  // hidden imperatively too — React never rewrites display since the ghost's
  // style prop is unchanged between renders. Without this the pill lingered
  // at the drop point after every completed drag.
  const hideGhost = () => {
    const g = ghostRef.current;
    if (g) g.style.display = "none";
  };

  const chipUp =
    (p: Policy, showcase = false) =>
    (e: React.PointerEvent<HTMLButtonElement>) => {
      setDragging(null);
      hideGhost();
      if (assignDrag.mode !== "drag") return; // canceled (Escape)
      if (!moved.current) {
        // No movement → click: toggle armed mode instead.
        if (chipArmed(p.id, showcase)) {
          disarm();
        } else {
          setArmed({ id: p.id, showcase });
          assignDrag.mode = "armed";
          assignDrag.policyId = p.id;
          assignDrag.policyLabel = p.label;
          assignDrag.showcase = showcase;
        }
        return;
      }
      // Real drag → drop. Compute the target from the event position directly
      // (not hoverDuck) so a fast drag can't outrun the render loop. Teach
      // panel → load the run's recipe there; no duck in range but open canvas
      // under the cursor → spawn; dropped on other UI → cancel.
      if (isTeachAt(e.clientX, e.clientY)) {
        loadIntoTeach(p);
      } else {
        const duck = nearestDuck(e.clientX, e.clientY);
        if (duck) assign(duck, p, showcase);
        else if (isCanvasAt(e.clientX, e.clientY)) spawn(p, showcase);
      }
      disarm();
    };

  // Escape cancels an in-flight drag.
  useEffect(() => {
    if (!dragging) return;
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        clearAssignDrag();
        setArmed(null);
        setDragging(null);
        hideGhost();
      }
    };
    window.addEventListener("keydown", key, true);
    return () => window.removeEventListener("keydown", key, true);
  }, [dragging]);

  // Armed mode: track the pointer for the highlight, assign on duck click,
  // disarm on click-elsewhere or Escape.
  useEffect(() => {
    if (!armed) return;
    const move = (e: PointerEvent) => {
      assignDrag.px = e.clientX;
      assignDrag.py = e.clientY;
    };
    const down = (e: PointerEvent) => {
      const t = e.target as HTMLElement | null;
      if (t?.closest("[data-policy-ui]")) return; // panel clicks handled above
      if (assignDrag.policyId) {
        const policy = {
          id: assignDrag.policyId,
          label: assignDrag.policyLabel ?? assignDrag.policyId,
        };
        if (isTeachAt(e.clientX, e.clientY)) {
          // Armed click on the teach panel = same as dropping the chip there.
          loadIntoTeach(policy);
        } else {
          const duck = nearestDuck(e.clientX, e.clientY);
          if (duck) assign(duck, policy, assignDrag.showcase);
          // Armed click on open canvas away from every duck → spawn there;
          // clicks on other UI (HUD, pads) just disarm as before.
          else if (isCanvasAt(e.clientX, e.clientY)) spawn(policy, assignDrag.showcase);
        }
      }
      disarm();
    };
    const key = (e: KeyboardEvent) => {
      // Escape belongs to the dialog on top; without this it also silently
      // threw away the user's armed chip.
      if (modalIsOpen()) return;
      if (e.key === "Escape") disarm();
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerdown", down, true);
    window.addEventListener("keydown", key, true);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerdown", down, true);
      window.removeEventListener("keydown", key, true);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [armed]);

  // Applied filter, recomputed per render (the list is tens of items — no
  // memo needed). Groups that filter down to nothing render nothing, so the
  // headings disappear along with their chips.
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const shown = terms.length ? policies.filter((p) => matchesQuery(p, terms)) : policies;
  const shownStudio = studioRuns.filter((run) => terms.every((term) =>
    `${run.experimentId} ${run.runName}`.toLowerCase().includes(term)));

  const ghost = (
    <div
      ref={ghostRef}
      style={{
        display: "none",
        position: "fixed",
        left: 0,
        top: 0,
        transform: "translate(-50%, -140%)",
        background: "rgba(42, 53, 72, 0.92)",
        color: "#cfe4f5",
        border: "1px solid #7db8d8",
        borderRadius: 12,
        padding: "3px 9px",
        fontFamily: mono,
        fontSize: 11,
        whiteSpace: "nowrap",
        pointerEvents: "none",
        zIndex: 100,
        boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
      }}
    >
      {dragging?.label}
      {/* outcome hint, retargeted per pointer move (assign/spawn/cancel) */}
      <div
        ref={ghostSubRef}
        style={{
          fontSize: 9,
          color: "#8fb8d0",
          textAlign: "center",
          marginTop: 1,
        }}
      >
        {t("drop to spawn", "松开以生成")}
      </div>
    </div>
  );

  if (!open && !embedded)
    return (
      <button
        data-policy-ui
        onClick={() => setOpen(true)}
        style={{
          position: "absolute",
          zIndex: 20,
          right: 14,
          top: 14,
          background: "rgba(14,16,20,0.86)",
          color: "#e8e6e1",
          border: "1px solid rgba(255,255,255,0.12)",
          borderRadius: 10,
          padding: "8px 12px",
          fontFamily: mono,
          fontSize: 12,
          cursor: "pointer",
          backdropFilter: "blur(6px)",
        }}
      >
        {t("🧠 policies", "🧠 策略")}
      </button>
    );

  return (
    <>
      <div
        data-policy-ui
        style={{
          position: embedded ? "relative" : "absolute",
          // Above the ducks' floating DOM labels (drei Html, zIndexRange
          // [10, 0]) — labels must never scribble over the chip list.
          zIndex: 20,
          right: embedded ? "auto" : 14,
          top: embedded ? "auto" : 14,
          width: embedded ? "100%" : "min(230px, calc(100cqw - 28px))",
          // Grow to fill the column: take everything the TeachPanel below
          // isn't using. Chrome to subtract = 14px top inset + 14px gap +
          // teach's 14px bottom inset + our own 2px of border (maxHeight is
          // content-box here) = 44px. So a collapsed or short teach panel
          // hands the chip list its space instead of leaving a dead gap.
          // Teach derives its own maxHeight from the min(40vh, 380px) cap
          // below and never from our measured height, so this stays a
          // one-way dependency — that cap is also the fallback until teach
          // reports in (SSR and first paint).
          maxHeight: embedded
            ? "none"
            : teachHeight
              ? `calc(100cqh - ${Math.round(teachHeight) + 44}px)`
              : "min(40cqh, 380px)",
          height: embedded ? "100%" : "auto",
          display: "flex",
          flexDirection: "column",
          background: embedded ? "transparent" : "rgba(14, 16, 20, 0.82)",
          border: embedded ? "0" : "1px solid rgba(255,255,255,0.09)",
          borderRadius: embedded ? 0 : 10,
          color: "#e8e6e1",
          fontFamily: mono,
          fontSize: 12,
          lineHeight: 1.55,
          backdropFilter: embedded ? "none" : "blur(6px)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            padding: "8px 12px",
            fontWeight: 700,
            fontSize: 13,
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
          }}
        >
          <span style={{ flex: 1 }}>{t("🧠 policies", "🧠 策略")}</span>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={refreshing}
            aria-label={t("Refresh policies", "刷新策略")}
            title={t(
              "Scan training runs and checkpoints for the latest policies",
              "扫描训练运行和检查点以获取最新策略"
            )}
            style={{
              background: "rgba(125,184,216,0.1)",
              border: "1px solid rgba(125,184,216,0.3)",
              borderRadius: 6,
              color: "#cfe4f5",
              cursor: refreshing ? "wait" : "pointer",
              fontFamily: mono,
              fontSize: 11,
              minHeight: 36,
              padding: "4px 8px",
            }}
          >
            {refreshing ? t("↻ Scanning…", "↻ 扫描中…") : t("↻ Refresh", "↻ 刷新")}
          </button>
          {!embedded && (
            <button
              onClick={() => setOpen(false)}
              title={t("collapse", "收起")}
              style={{
                // keep clear of the Next.js dev-tools badge that floats in
                // this corner during development
                marginRight: 30,
                background: "none",
                border: "none",
                color: "#8b93a3",
                cursor: "pointer",
                fontFamily: mono,
                fontSize: 12,
                padding: "0 4px",
              }}
            >
              —
            </button>
          )}
        </div>

        {/* Filter box. Sits between the header and the scrolling list so it
            stays put while the chips scroll under it. */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            padding: "5px 10px",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            flexShrink: 0,
          }}
        >
          <span style={{ color: "#8b93a3", fontSize: 11, flexShrink: 0 }}>⌕</span>
          <input
            ref={searchRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            // Escape clears the filter instead of bubbling out to the
            // drag/armed cancel handlers — but only when there is something
            // to clear, so an empty box still lets Escape disarm a chip.
            onKeyDown={(e) => {
              if (e.key === "Escape" && query) {
                e.stopPropagation();
                setQuery("");
              }
            }}
            placeholder={t("filter", "筛选")}
            spellCheck={false}
            autoComplete="off"
            aria-label={t("filter policies", "筛选策略")}
            style={{
              flex: 1,
              minWidth: 0,
              background: "none",
              border: "none",
              outline: "none",
              color: "#cfe4f5",
              fontFamily: mono,
              fontSize: 11,
              padding: 0,
            }}
          />
          {query && (
            <>
              <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                {shown.length + shownStudio.length}
              </span>
              <button
                type="button"
                onClick={() => setQuery("")}
                aria-label={t("clear filter", "清除筛选")}
                title={t("clear filter", "清除筛选")}
                style={{
                  flexShrink: 0,
                  background: "none",
                  border: "none",
                  color: "#8b93a3",
                  cursor: "pointer",
                  fontFamily: mono,
                  fontSize: 11,
                  lineHeight: 1,
                  padding: "2px 3px",
                }}
              >
                ✕
              </button>
            </>
          )}
        </div>

        <div role="status" aria-live="polite" style={{ padding: "5px 10px", color: err ? "#e07a5f" : "#8b93a3", fontSize: 10 }}>
          {refreshing
            ? t("Scanning training runs and checkpoints…", "正在扫描训练运行和检查点…")
            : err
              ? t(
                  `Could not refresh from ${LAB_HTTP}. ${policies.length ? "Previous list retained. " : ""}Check the lab connection and retry.`,
                  `无法从 ${LAB_HTTP} 刷新。${policies.length ? "已保留之前的列表。" : ""}请检查实验室连接后重试。`
                )
              : refreshedAt
                ? t(
                    `${policies.length} live policies${includeStudio ? ` · ${studioRuns.length} Studio runs` : ""} · scanned ${refreshedAt.toLocaleTimeString()} · newest runs first`,
                    `${policies.length} 个实时策略${includeStudio ? ` · ${studioRuns.length} 个 Studio 运行` : ""} · 扫描于 ${refreshedAt.toLocaleTimeString()} · 最新运行优先`
                  )
                : t("Refresh to discover available policies.", "刷新以查找可用策略。")}
        </div>
        <div aria-busy={refreshing} style={{ overflowY: "auto", padding: "4px 10px 10px" }}>
          {includeStudio && (
            <section aria-label={t("Studio trained policies", "Studio 训练策略")} style={{ marginBottom: 12 }}>
              <div style={{ color: "#cfe4f5", fontWeight: 700, margin: "7px 0 4px" }}>
                {t("Studio runs · latest training / export first", "Studio 运行 · 最新训练或导出优先")}
              </div>
              <div style={{ color: "#8b93a3", fontSize: 10, marginBottom: 6 }}>
                {t(
                  "Drag a policy onto a duck in the left viewer to apply it. Double-click to add a duck. Review opens evaluation and export controls.",
                  "将策略拖到左侧查看器中的鸭子上即可应用。双击可添加鸭子。评审会打开评估和导出控制。"
                )}
              </div>
              {studioError && <div role="alert" style={{ color: "#e07a5f" }}>
                {t(
                  "Studio scan failed. Previous Studio list retained; refresh to retry.",
                  "Studio 扫描失败。已保留之前的 Studio 列表；请刷新重试。"
                )}
              </div>}
              {!refreshing && !studioError && !studioRuns.length && <div style={{ color: "#8b93a3" }}>
                {t("No saved Studio runs yet.", "尚无已保存的 Studio 运行。")}
              </div>}
              <div style={{ maxHeight: 240, overflowY: "auto" }}>
              {shownStudio.map((run) => {
                const policy = policies.find((entry) => entry.id === `studio:${run.experimentId}/${run.runName}`);
                return (
                <div key={`${run.experimentId}:${run.runName}`} style={{ borderBottom: "1px solid rgba(255,255,255,0.08)", padding: "6px 0" }}>
                  {policy ? (
                    <Chip
                      policy={policy}
                      armed={chipArmed(policy.id)}
                      onDown={chipDown(policy)}
                      onMove={chipMove}
                      onUp={chipUp(policy)}
                      onDouble={chipDouble(policy)}
                    />
                  ) : <div style={{ overflowWrap: "anywhere", fontSize: 11 }}>{run.runName}</div>}
                  <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 6, marginTop: 3 }}>
                    <span style={{ color: "#8b93a3", fontSize: 10 }}>
                      {run.experimentId} · {run.balanceOnly
                        ? t("balance-only preview · steering not passed", "仅平衡预览 · 未通过转向")
                        : policy
                          ? t("drag to apply", "拖动以应用")
                          : run.onnx
                            ? t("not available in Lab · review export", "实验室中不可用 · 检查导出")
                            : run.checkpoint
                              ? t("export ONNX to enable drag", "导出 ONNX 后可拖动")
                              : t("evaluation only", "仅评估")}
                    </span>
                    <button type="button" disabled={reviewDisabled} onClick={() => void onReviewRun?.(run.experimentId, run.runName)}
                      aria-label={t(`Review ${run.runName}`, `评审 ${run.runName}`)}
                      title={reviewDisabled
                        ? t(
                            "Wait for the current Studio operation to finish",
                            "请等待当前 Studio 操作完成"
                          )
                        : t(
                            "Load this run's scenario, artifacts, and evaluation controls",
                            "加载此运行的场景、产物和评估控制"
                          )}
                      style={{ background: "rgba(125,184,216,0.1)", color: "#cfe4f5", border: "1px solid rgba(125,184,216,0.3)", borderRadius: 6, padding: "4px 8px", minHeight: 36, fontFamily: mono, cursor: reviewDisabled ? "not-allowed" : "pointer" }}>
                      {t("Review / render", "评审 / 渲染")}
                    </button>
                  </div>
                </div>
                );
              })}
              </div>
            </section>
          )}
          {!refreshing && !err && refreshedAt && !policies.length && (
            <div style={{ color: "#8b93a3", margin: "6px 0" }}>
              {t(
                "No policies found. Save a training checkpoint or export an ONNX policy, then refresh.",
                "未找到策略。请保存训练检查点或导出 ONNX 策略，然后刷新。"
              )}
            </div>
          )}
          {GROUPS.map(({ key, title, zh }) => {
            if (key === "studio" && includeStudio) return null;
            const list = shown.filter((p) => p.group === key);
            if (!list.length) return null;
            const isFolded = !!folded[key] && !terms.length;
            return (
              <div key={key}>
                <button
                  type="button"
                  onClick={() => toggleFold(key)}
                  aria-expanded={!isFolded}
                  title={isFolded
                    ? t(`expand ${title}`, `展开${zh}`)
                    : t(`collapse ${title}`, `收起${zh}`)}
                  style={{
                    display: "flex",
                    alignItems: "baseline",
                    gap: 5,
                    width: "100%",
                    background: "none",
                    border: "none",
                    padding: 0,
                    margin: "7px 0 4px",
                    color: "#8b93a3",
                    fontFamily: mono,
                    fontSize: 10,
                    textAlign: "left",
                    cursor: "pointer",
                  }}
                >
                  <span style={{ fontSize: 8 }}>{isFolded ? "▸" : "▾"}</span>
                  <span>{t(title, zh)}</span>
                  {/* Folded count matches what the section shows unfolded:
                      chains collapse to one row each in "Our runs". */}
                  {isFolded && (
                    <span style={{ fontSize: 9 }}>
                      ({key === "runs" ? runRows(list).length : list.length})
                    </span>
                  )}
                </button>
                {isFolded ? null : key === "runs" ? (
                  // Newest-first (server-sorted by mtime), curriculum chains
                  // folded into one family row of stage chips — every chip,
                  // stage chips included, drags/arms exactly like before.
                  runRows(list).map((row) => {
                    if (row.kind === "single")
                      return (
                        <div
                          key={row.p.id}
                          onPointerEnter={() => setHoverRow(row.p.id)}
                          onPointerLeave={() => setHoverRow((h) => (h === row.p.id ? null : h))}
                          style={{ display: "flex", alignItems: "center", gap: 6, margin: "3px 0" }}
                        >
                          <Chip
                            policy={row.p}
                            armed={chipArmed(row.p.id)}
                            onDown={chipDown(row.p)}
                            onMove={chipMove}
                            onUp={chipUp(row.p)}
                            onDouble={chipDouble(row.p)}
                          />
                          {row.p.mtime != null && (
                            <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                              {relTime(row.p.mtime, t)}
                            </span>
                          )}
                          <span style={{ flex: 1 }} />
                          {row.p.artifact !== "live.onnx" && <DownloadBtn
                            show={hoverRow === row.p.id}
                            run={row.p.label}
                            label={row.p.label}
                            onFocus={() => setHoverRow(row.p.id)}
                            onBlur={() => setHoverRow((h) => (h === row.p.id ? null : h))}
                          />}
                          {row.p.artifact === "live.onnx" && <span style={{ color: "#8b93a3", fontSize: 9 }}>
                            {t("live snapshot", "实时快照")}
                          </span>}
                          <DeleteBtn
                            show={hoverRow === row.p.id}
                            label={row.p.label}
                            onFocus={() => setHoverRow(row.p.id)}
                            onBlur={() => setHoverRow((h) => (h === row.p.id ? null : h))}
                            onClick={() =>
                              askDelete({
                                name: row.p.label,
                                chain: false,
                                runs: [row.p.label],
                                bytes: row.p.sizeBytes ?? 0,
                              })
                            }
                          />
                        </div>
                      );
                    // The chain-level "whole trick" chip assigns the FINAL
                    // stage's policy — each stage fine-tunes the same network,
                    // so the last one carries the entire curriculum — flagged
                    // showcase so the duck's env rehearses the whole trick arc
                    // instead of only a standing start. Ghost/toast label is
                    // the chain's name (✨), matching the server's roster label.
                    const last = row.stages[row.stages.length - 1];
                    const whole: Policy = {
                      ...last,
                      label: `${row.chain.replace(/^teach-/, "")} ✨`,
                    };
                    return (
                      <div
                        key={row.chain}
                        onPointerEnter={() => setHoverRow(row.chain)}
                        onPointerLeave={() => setHoverRow((h) => (h === row.chain ? null : h))}
                        style={{ margin: "4px 0" }}
                      >
                        <div
                          style={{
                            color: "#aab3c0",
                            fontSize: 10,
                            display: "flex",
                            gap: 6,
                            alignItems: "baseline",
                          }}
                        >
                          <span
                            style={{
                              minWidth: 0,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {row.chain.replace(/^teach-/, "")}
                          </span>
                          <span style={{ color: "#8b93a3", fontSize: 9, flexShrink: 0 }}>
                            {t(
                              `${row.stages.length} stages`,
                              `${row.stages.length} 个阶段`
                            )} · {relTime(row.newest, t)}
                          </span>
                          <span style={{ flex: 1 }} />
                          {/* ⤓ downloads the FINAL stage's brain — each stage
                              fine-tunes the same network, so the last one IS
                              the whole trick. */}
                          {last.artifact !== "live.onnx" && <DownloadBtn
                            show={hoverRow === row.chain}
                            run={last.label}
                            label={t(
                              `${row.chain.replace(/^teach-/, "")} (final stage)`,
                              `${row.chain.replace(/^teach-/, "")}（最终阶段）`
                            )}
                            onFocus={() => setHoverRow(row.chain)}
                            onBlur={() => setHoverRow((h) => (h === row.chain ? null : h))}
                          />}
                          {/* One ✕ for the family: the stages of a chain are
                              one trick's training data, and a half-deleted
                              chain can't be resumed or fine-tuned from. */}
                          <DeleteBtn
                            show={hoverRow === row.chain}
                            label={t(
                              `the whole ${row.chain.replace(/^teach-/, "")} chain`,
                              `整个 ${row.chain.replace(/^teach-/, "")} 训练链`
                            )}
                            onFocus={() => setHoverRow(row.chain)}
                            onBlur={() => setHoverRow((h) => (h === row.chain ? null : h))}
                            onClick={() =>
                              askDelete({
                                name: row.chain,
                                chain: true,
                                runs: row.stages.map((p) => p.label),
                                bytes: row.stages.reduce((n, p) => n + (p.sizeBytes ?? 0), 0),
                              })
                            }
                          />
                        </div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 2 }}>
                          <Tip
                            tip={
                              <>
                                {t(
                                  "each stage trains the same brain — the final stage carries all of it. ▶ assigns that finished policy in showcase mode: the duck rehearses spawns across the whole trick arc, so every section gets performed.",
                                  "每个阶段都训练同一个策略，最终阶段包含全部成果。▶ 会以展示模式分配完成的策略，让鸭子从整个动作流程的不同位置开始练习，从而演示所有部分。"
                                )}
                              </>
                            }
                          >
                            <Chip
                              policy={whole}
                              display={t("▶ whole trick", "▶ 完整动作")}
                              accent
                              title=""
                              armed={chipArmed(last.id, true)}
                              onDown={chipDown(whole, true)}
                              onMove={chipMove}
                              onUp={chipUp(whole, true)}
                              onDouble={chipDouble(whole, true)}
                            />
                          </Tip>
                          {row.stages.map((p) => (
                            <Chip
                              key={p.id}
                              policy={p}
                              display={`s${p.stage}`}
                              armed={chipArmed(p.id)}
                              onDown={chipDown(p)}
                              onMove={chipMove}
                              onUp={chipUp(p)}
                              onDouble={chipDouble(p)}
                            />
                          ))}
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {list.map((p) => (
                      <Chip
                        key={p.id}
                        policy={p}
                        armed={chipArmed(p.id)}
                        onDown={chipDown(p)}
                        onMove={chipMove}
                        onUp={chipUp(p)}
                        onDouble={chipDouble(p)}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {terms.length > 0 && shown.length === 0 && shownStudio.length === 0 && !err && !studioError && (
            <div style={{ color: "#8b93a3", fontSize: 11, margin: "8px 2px" }}>
              {t(
                `no policy matches “${query.trim()}”`,
                `没有策略匹配“${query.trim()}”`
              )}
            </div>
          )}
        </div>

        {armed && (
          <div
            style={{
              padding: "6px 12px",
              borderTop: "1px solid rgba(255,255,255,0.08)",
              color: "#7db8d8",
              fontSize: 10,
              flexShrink: 0,
            }}
          >
            {t(
              "armed — click a duck to assign, or empty floor to spawn · esc cancels",
              "已选中 — 点击鸭子进行分配，或点击空地生成 · Esc 取消"
            )}
          </div>
        )}
      </div>
      {ghost}
      {pendingDelete && (
        <DeleteDialog
          target={pendingDelete}
          busy={deleting}
          error={deleteErr}
          onCancel={() => {
            if (deleting) return; // a delete in flight can't be called back
            setPendingDelete(null);
            setDeleteErr(null);
          }}
          onConfirm={confirmDelete}
        />
      )}
    </>
  );
}
