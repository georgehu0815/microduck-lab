"use client";

// The teach panel: a chat-like stream where you ask the duck for a trick
// ("stand on one leg"), see the reward recipe it will be scored on in plain
// English, then watch the training run live — score curve, per-term reward
// bars, helper count, and snapshot updates landing on the 🎓 trainee duck in
// the scene. Once a run ends, the recipe becomes editable: drag the weight
// sliders and either retrain from scratch or fine-tune the finished result.
// Self-contained: talks to the lab server directly and reads the streamed
// frame via clientRef (same polling pattern as Hud).

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  LAB_HTTP,
  MAX_STEP_BUDGET,
  MIN_STEP_BUDGET,
  clampStepBudget,
  isRunPolicy,
  loadTeachRun,
  resolveStageSteps,
  runNameOfPolicy,
  type BehaviorCard,
  type LabClient,
  type TermCard,
  type TrainingPayload,
} from "@/lib/lab";
import { loadJSON, saveJSON } from "@/lib/persist";
import { useSelectedDuck } from "@/lib/select";
import { modalIsOpen, setTeachHeight, usePolicyOpen } from "@/lib/ui";
import { useLanguage } from "./LanguageProvider";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

// Terms arrive in recipe order, which interleaves the ones that pay points
// with the ones that charge — a green/red zebra you have to re-read row by
// row. Every list of them groups: earners first, then penalties, recipe
// order preserved inside each group.
function byPolarity<T extends { isPenalty: boolean }>(terms: T[]): T[] {
  return [...terms.filter((t) => !t.isPenalty), ...terms.filter((t) => t.isPenalty)];
}

type Msg =
  | { kind: "user"; text: string }
  | { kind: "note"; text: string }
  // stageSteps/stepBudget are what THAT launch actually got — the card
  // itself carries the recipe's declared numbers, so without them the chat
  // log would keep advertising a plan the run never trained under.
  | { kind: "card"; card: BehaviorCard; stageSteps?: number[]; stepBudget?: number };

const MSG_CAP = 50;

const SUGGESTIONS = [
  { value: "stand still", zh: "原地站立" },
  { value: "stand on one leg", zh: "单腿站立" },
  { value: "crouch down", zh: "蹲下" },
  { value: "spin in place", zh: "原地旋转" },
  { value: "do a headstand", zh: "倒立" },
];

// --- instant hover tooltip ---------------------------------------------------
// Native `title` attrs take ~1 s to appear and are easy to miss; this shows a
// styled tooltip on mouseenter. Rendered through a portal at position:fixed
// coordinates (from the hovered element's rect) so the panel's overflow
// scroll — and its backdrop-filter, which would hijack fixed positioning —
// can never clip it.

const TIP_HALF_W = 138; // max-width 260 / 2 + 8px viewport margin

// Exported: PolicyPanel's "whole trick" chip uses the same styled tooltip so
// its explainer matches the teach panel's instead of a laggy native title.
export function Tip({ tip, children }: { tip: React.ReactNode; children: React.ReactNode }) {
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);
  return (
    <div
      onMouseEnter={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        setAnchor({ x: r.left + r.width / 2, y: r.top });
      }}
      onMouseLeave={() => setAnchor(null)}
    >
      {children}
      {anchor &&
        createPortal(
          <div
            style={{
              position: "fixed",
              left: Math.max(TIP_HALF_W, Math.min(anchor.x, window.innerWidth - TIP_HALF_W)),
              bottom: window.innerHeight - anchor.y + 6,
              transform: "translateX(-50%)",
              maxWidth: 260,
              background: "rgba(14, 16, 20, 0.96)",
              border: "1px solid rgba(255,255,255,0.14)",
              borderRadius: 7,
              padding: "6px 8px",
              color: "#e8e6e1",
              fontFamily: mono,
              fontSize: 11,
              lineHeight: 1.45,
              pointerEvents: "none",
              zIndex: 1000,
              boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
              animation: "ducklab-tip-in 70ms ease-out",
            }}
          >
            <style>{"@keyframes ducklab-tip-in { from { opacity: 0 } }"}</style>
            {tip}
          </div>,
          document.body
        )}
    </div>
  );
}

function RecipeRows({ terms }: { terms: BehaviorCard["terms"] }) {
  const { t } = useLanguage();
  const max = Math.max(...terms.map((t) => t.weight));
  return (
    <div style={{ marginTop: 6 }}>
      {byPolarity(terms).map((t) => (
        <Tip key={t.key} tip={t.friendly}>
          <div style={{ display: "flex", alignItems: "flex-start", gap: 6, margin: "3px 0" }}>
            <div
              style={{
                width: 46,
                height: 6,
                borderRadius: 3,
                background: "#262a33",
                overflow: "hidden",
                flexShrink: 0,
                marginTop: 4, // centers the bar on the first text line
              }}
            >
              <div
                style={{
                  width: `${(t.weight / max) * 100}%`,
                  height: "100%",
                  background: t.isPenalty ? "#e07a5f" : "#7dd87d",
                }}
              />
            </div>
            <span
              style={{
                flex: 1,
                minWidth: 0,
                color: t.isPenalty ? "#e0a08f" : "#c9d4c9",
                fontSize: 11,
                lineHeight: 1.4,
                overflowWrap: "anywhere", // full sentence, wrapped — never ellipsized
              }}
            >
              {t.friendly}
            </span>
          </div>
        </Tip>
      ))}
      <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 4 }}>
        {t(
          "green = points to win · red = points lost · bar = how much it matters",
          "绿色 = 得分 · 红色 = 扣分 · 条形长度 = 重要程度"
        )}
      </div>
    </div>
  );
}

/** The recipe's own practice budget: a staged curriculum sums its stages, a
 *  single-run behavior trains defaultSteps — the baseline a user-chosen
 *  budget replaces. */
function cardSteps(card: BehaviorCard): number {
  return card.curriculum?.length
    ? card.curriculum.reduce((sum, s) => sum + s.steps, 0)
    : card.defaultSteps;
}

/** Per-stage declared steps, as a flat list (one entry for a single run). */
function declaredStages(card: BehaviorCard): number[] {
  return card.curriculum?.length
    ? card.curriculum.map((s) => s.steps)
    : [card.defaultSteps];
}

/** A step count in the panel's voice. Millions once there are millions;
 *  thousands below that, because a scaled-down stage rendered "0.0M" is the
 *  same lie the budget control exists to remove — and a second decimal under
 *  1M, so a 0.75M stage doesn't advertise itself as 0.8M. */
function fmtSteps(n: number): string {
  if (n < 1e5) return `${Math.round(n / 1e3)}k`;
  if (n < 1e6) return `${(n / 1e6).toFixed(2).replace(/0$/, "")}M`;
  return `${(n / 1e6).toFixed(1)}M`;
}

function Sparkline({ points }: { points: { x: number; y: number }[] }) {
  if (points.length < 2) return null;
  const w = 250, h = 36;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const path = points
    .map((p, i) => {
      const px = ((p.x - x0) / Math.max(x1 - x0, 1)) * w;
      const py = h - ((p.y - y0) / Math.max(y1 - y0, 1e-6)) * (h - 4) - 2;
      return `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={w} height={h} style={{ display: "block" }}>
      <path d={path} fill="none" stroke="#7db8d8" strokeWidth={1.5} />
    </svg>
  );
}

// --- practice budget --------------------------------------------------------
// "How long should it practice?" — the one number that used to live only in
// behaviors.py. Typed entry is the point: a slider can't land on 3.5 on
// purpose, and the user has asked for real number fields before. Presets sit
// beside it for the common answers.

const BUDGET_PRESETS_M = [1, 2, 4, 8];

/** A "millions of steps" field. `value` and `onCommit` speak STEPS; the box
 *  speaks millions, and keeps its own text so half-typed numbers ("3.",
 *  ".75") survive. null = nothing chosen, show the placeholder. */
function MStepsInput({
  value,
  placeholder,
  disabled,
  width = 54,
  onCommit,
}: {
  value: number | null;
  placeholder?: string;
  disabled?: boolean;
  width?: number;
  onCommit: (steps: number | null) => void;
}) {
  const toText = (v: number | null) =>
    v == null ? "" : String(Math.round(v / 1e4) / 100);
  const [draft, setDraft] = useState(() => toText(value));
  // What this box last emitted — an incoming `value` it did not cause (a
  // preset, a reset, a new run's numbers) resyncs the text; its own echo
  // must not, or typing "3." would be rewritten to "3" mid-keystroke.
  const emitted = useRef<number | null>(value);
  useEffect(() => {
    if (value === emitted.current) return;
    emitted.current = value;
    setDraft(toText(value));
  }, [value]);
  const commit = (text: string) => {
    setDraft(text);
    const n = parseFloat(text);
    const steps = Number.isFinite(n) && n > 0 ? Math.round(n * 1e6) : null;
    emitted.current = steps;
    onCommit(steps);
  };
  return (
    <input
      type="number"
      min={0}
      step={0.5}
      inputMode="decimal"
      value={draft}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => commit(e.target.value)}
      // Clamp when they leave the field, not while typing — fighting a "4"
      // on its way to "40" is worse than a late correction.
      onBlur={() => {
        if (emitted.current == null) return;
        const c = clampStepBudget(emitted.current);
        if (c !== emitted.current) {
          emitted.current = c;
          setDraft(toText(c));
          onCommit(c);
        }
      }}
      style={{
        width,
        background: "#12151b",
        border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: 5,
        color: disabled ? "#8b93a3" : "#e8e6e1",
        padding: "2px 4px",
        fontFamily: mono,
        fontSize: 11,
        textAlign: "right",
        outline: "none",
      }}
    />
  );
}

// --- the "edit the recipe" section -----------------------------------------
// While training runs the sliders are the read-only truth of the live
// scorecard; once the run ends they unlock, and the two buttons resubmit the
// behavior with only the weights the user actually moved.

function RecipeEditor({
  t,
  wide,
  onSubmit,
}: {
  t: TrainingPayload;
  wide: boolean;
  onSubmit: (weights: Record<string, number>, fineTune: boolean) => void;
}) {
  const { t: tr } = useLanguage();
  // Only touched sliders live here; everything else displays the effective
  // weight straight from the stream. Keyed by run so a new run resets dirt.
  const [edited, setEdited] = useState<Record<string, number>>({});
  // Catalog terms pulled into the recipe locally (key → weight). They ride to
  // the server inside the same weights dict as slider overrides; the next run
  // then owns them (they arrive back in behavior.terms), so they reset with
  // `edited` when the run changes.
  const [added, setAdded] = useState<Record<string, number>>({});
  const [pickerOpen, setPickerOpen] = useState(false);
  const runRef = useRef(t.runName);
  useEffect(() => {
    if (runRef.current !== t.runName) {
      runRef.current = t.runName;
      setEdited({});
      setAdded({});
      setPickerOpen(false);
    }
  }, [t.runName]);
  useEffect(() => {
    if (!pickerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (modalIsOpen()) return;   // a dialog on top owns Escape
      if (e.key === "Escape") setPickerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pickerOpen]);

  const live = t.status === "training";
  const effective = (key: string, fallback: number) => t.weights[key] ?? fallback;
  const moved: Record<string, number> = {};
  for (const term of t.behavior.terms) {
    const v = edited[term.key];
    if (v != null && Math.abs(v - effective(term.key, term.weight)) > 1e-9)
      moved[term.key] = v;
  }
  const movedN = Object.keys(moved).length;

  // Catalog terms: already-pulled ones render as regular slider rows and fold
  // into the submitted weights (adding at the default weight is itself the
  // change — the key's presence is what makes the server adopt the term);
  // the rest wait in the picker.
  const catalog = t.behavior.availableTerms ?? [];
  const inRecipe = new Set(t.behavior.terms.map((term) => term.key));
  const addedTerms = catalog.filter((a) => !inRecipe.has(a.key) && added[a.key] != null);
  const pickable = catalog.filter((a) => !inRecipe.has(a.key) && added[a.key] == null);
  const addedN = addedTerms.length;
  for (const a of addedTerms) moved[a.key] = added[a.key];

  const btn: React.CSSProperties = {
    background: "#1c2230",
    color: "#9fb4d8",
    border: "1px solid rgba(255,255,255,0.14)",
    borderRadius: 6,
    padding: "3px 8px",
    fontFamily: mono,
    fontSize: 10,
    cursor: "pointer",
  };

  // One recipe row, in either layout. Added rows differ only in where their
  // weight lives (`added`, not `edited`) and in growing a tiny ✕ that puts
  // the term back in the picker.
  const termRow = (term: TermCard, isAdded: boolean) => {
    const def = term.weight;
    const max = def > 0 ? def * 2.5 : 1;
    const value = isAdded
      ? (added[term.key] ?? def)
      : (edited[term.key] ?? effective(term.key, def));
    const color = term.isPenalty ? "#e0a08f" : "#9fd89f";
    const slider = (
      <input
        type="range"
        min={0}
        max={max}
        step={0.05}
        value={value}
        disabled={live}
        onChange={(e) => {
          const v = Number(e.target.value);
          (isAdded ? setAdded : setEdited)((w) => ({ ...w, [term.key]: v }));
        }}
        style={{ flex: 1, minWidth: 0, height: 14, accentColor: term.isPenalty ? "#e07a5f" : "#7dd87d" }}
      />
    );
    const valueLabel = (
      <span
        style={{
          width: 34,
          flexShrink: 0,
          textAlign: "right",
          fontSize: 10,
          color: isAdded || edited[term.key] != null ? "#e8e6e1" : "#8b93a3",
        }}
      >
        {value.toFixed(2)}
      </span>
    );
    const removeBtn = isAdded ? (
      <button
        aria-label={tr(`remove ${term.key} from the recipe`, `从配方中移除 ${term.key}`)}
        onClick={() =>
          setAdded((w) => {
            const rest = { ...w };
            delete rest[term.key];
            return rest;
          })
        }
        style={{
          background: "none",
          border: "none",
          color: "#8b93a3",
          cursor: "pointer",
          fontFamily: mono,
          fontSize: 10,
          padding: "0 2px",
          flexShrink: 0,
          lineHeight: 1,
        }}
      >
        ✕
      </button>
    ) : null;
    const tip = (
      <>
        <div>{term.friendly}</div>
        <div style={{ color: "#8b93a3", marginTop: 3 }}>
          {tr("default weight", "默认权重")} {def} · {tr("currently", "当前")} {value.toFixed(2)}
          {isAdded && tr(" · added — ✕ puts it back in the picker", " · 已添加 — 点击 ✕ 可放回选择器")}
        </div>
      </>
    );
    return (
      <Tip key={term.key} tip={tip}>
        {wide ? (
          // Wide panel: the classic single-line row, but with a label
          // column wide enough (~55%) to hold a whole sentence.
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              margin: "3px 0",
              opacity: live ? 0.55 : 1,
            }}
          >
            <span
              style={{
                width: "55%",
                flexShrink: 0,
                color,
                fontSize: 10,
                lineHeight: 1.35,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {term.friendly}
            </span>
            {slider}
            {valueLabel}
            {removeBtn}
          </div>
        ) : (
          // Normal 320px panel: sentence on its own line (wrapping, up to
          // two lines, no single-line ellipsis), slider + value below.
          <div style={{ margin: "5px 0", opacity: live ? 0.55 : 1 }}>
            <div
              style={{
                color,
                fontSize: 10,
                lineHeight: 1.35,
                overflowWrap: "anywhere",
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {term.friendly}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 1 }}>
              {slider}
              {valueLabel}
              {removeBtn}
            </div>
          </div>
        )}
      </Tip>
    );
  };

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ color: "#8b93a3", fontSize: 10 }}>
        {tr("the recipe", "训练配方")}
        {live ? tr(" (read-only while training)", "（训练期间只读）") : tr(" — edit it:", " — 可编辑：")}
      </div>
      <div style={{ color: "#8b93a3", fontSize: 10, margin: "2px 0 4px" }}>
        {tr(
          "green terms pay points, red ones charge — drag to change how much each matters, then retrain.",
          "绿色项目加分，红色项目扣分。拖动滑块调整各项权重，然后重新训练。"
        )}
      </div>
      {(
        [
          { label: tr("what pays points", "加分项"), penalty: false },
          { label: tr("what costs points", "扣分项"), penalty: true },
        ] as const
      ).map(({ label, penalty }) => {
        // Added terms join the group they belong to instead of trailing the
        // whole list, so a freshly added penalty sits with the penalties.
        const rows = [
          ...t.behavior.terms.map((term) => [term, false] as const),
          ...(live ? [] : addedTerms.map((term) => [term, true] as const)),
        ].filter(([term]) => term.isPenalty === penalty);
        if (rows.length === 0) return null;
        return (
          <div key={label} style={{ marginTop: 4 }}>
            <div style={{ color: penalty ? "#e0a08f" : "#9fd89f", fontSize: 9, opacity: 0.7 }}>
              {label}
            </div>
            {rows.map(([term, isAdded]) => termRow(term, isAdded))}
          </div>
        );
      })}
      {!live && (pickable.length > 0 || pickerOpen) && (
        <div style={{ marginTop: 5 }}>
          <button
            onClick={() => setPickerOpen((o) => !o)}
            style={{
              background: "none",
              color: pickerOpen ? "#9fb4d8" : "#8b93a3",
              border: "1px dashed rgba(255,255,255,0.22)",
              borderRadius: 6,
              padding: "2px 8px",
              fontFamily: mono,
              fontSize: 10,
              cursor: "pointer",
            }}
          >
            {tr("＋ add a term", "＋ 添加项目")}
          </button>
          {pickerOpen && (
            <div
              style={{
                marginTop: 4,
                background: "#12151b",
                border: "1px solid rgba(255,255,255,0.10)",
                borderRadius: 6,
                padding: 3,
              }}
            >
              {pickable.length === 0 ? (
                <div style={{ color: "#8b93a3", fontSize: 10, padding: "2px 5px" }}>
                  {tr("every catalog term is already in the recipe", "目录中的所有项目都已加入配方")}
                </div>
              ) : (
                byPolarity(pickable).map((a) => (
                  <button
                    key={a.key}
                    onClick={() => setAdded((w) => ({ ...w, [a.key]: a.weight }))}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      width: "100%",
                      background: "none",
                      border: "none",
                      borderRadius: 4,
                      padding: "3px 5px",
                      fontFamily: mono,
                      fontSize: 10,
                      cursor: "pointer",
                      textAlign: "left",
                    }}
                  >
                    <span
                      style={{
                        flex: 1,
                        minWidth: 0,
                        color: a.isPenalty ? "#e0a08f" : "#9fd89f",
                        lineHeight: 1.35,
                        overflowWrap: "anywhere",
                      }}
                    >
                      {a.friendly}
                    </span>
                    <span style={{ flexShrink: 0, color: "#8b93a3" }}>{a.weight}</span>
                  </button>
                ))
              )}
            </div>
          )}
        </div>
      )}
      {!live && (
        <div style={{ marginTop: 6 }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button style={btn} onClick={() => onSubmit(moved, false)}>
              {tr("↻ retrain with edited recipe", "↻ 使用修改后的配方重新训练")}
            </button>
            <button
              style={{ ...btn, color: "#d8c97d", borderColor: "#5a5233" }}
              onClick={() => onSubmit(moved, true)}
            >
              {tr("✨ fine-tune the result", "✨ 微调当前结果")}
            </button>
          </div>
          <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 3 }}>
            {tr(
              "fine-tune keeps what it learned and adjusts; retrain starts fresh",
              "微调会保留已有能力并继续调整；重新训练会从头开始"
            )}
            {movedN > 0 && tr(
              ` · ${movedN} weight${movedN > 1 ? "s" : ""} changed`,
              ` · 已修改 ${movedN} 个权重`
            )}
            {addedN > 0 && tr(
              ` · ${addedN} term${addedN > 1 ? "s" : ""} added`,
              ` · 已添加 ${addedN} 个项目`
            )}
          </div>
        </div>
      )}
    </div>
  );
}

type StageWeightsMap = Record<string, Record<string, number>>;

function LiveTraining({
  t,
  traineeSpeed,
  wide,
  rewHistory,
  plan,
  pinned,
  onPinStage,
  onStop,
  onRecipeSubmit,
  onStageWeights,
  onStartStage,
}: {
  t: TrainingPayload;
  /** The trainee duck's live forward speed in m/s (null before its first
   *  snapshot, or right after an episode reset) — watching this climb is the
   *  point of a walking run, and the reward curve alone does not show it. */
  traineeSpeed: number | null;
  wide: boolean;
  rewHistory: { x: number; y: number }[];
  /** Practice steps per stage for the NEXT launch — the run's own numbers
   *  until the budget is edited, the previewed split after. */
  plan: number[];
  /** Stages the user pinned to a step count of their own (1-based keys);
   *  everything else takes its proportional share. */
  pinned: Record<string, number>;
  onPinStage: (stage: number, steps: number | null) => void;
  onStop: () => void;
  onRecipeSubmit: (
    weights: Record<string, number>,
    fineTune: boolean,
    stageWeights: StageWeightsMap | null
  ) => void;
  onStageWeights: (stageWeights: StageWeightsMap) => void;
  onStartStage: (idx: number, stageWeights: StageWeightsMap | null) => void;
}) {
  const { t: tr } = useLanguage();
  const p = t.progress;
  const stage = t.stage ?? null;
  // Curriculum jobs count the WHOLE chain in the headline numbers and main
  // bar (per-stage progress gets the thin bar below); overall* fall back to
  // the per-stage fields, so single-run jobs render exactly as before.
  const overallSteps = p.overallSteps ?? p.steps ?? 0;
  const overallTotal = Math.max(p.overallTotal ?? p.total ?? 1, 1);
  const pct = Math.min(100, (overallSteps / overallTotal) * 100);
  const stagePct = Math.min(100, ((p.steps ?? 0) / Math.max(p.total ?? 1, 1)) * 100);
  // The trainer only tallies term earnings from episodes that END inside a
  // reporting window, so windows with no episode boundary arrive with an
  // empty terms dict — rendering that directly made the whole bars section
  // blink in and out. Hold the last real breakdown; reset when a new teach
  // job starts (stage handoffs within one chain keep the bars).
  const lastTerms = useRef<{ job: string; terms: [string, number][] }>({ job: "", terms: [] });
  const jobKey = (t.runName ?? "").replace(/-s\d+$/, "");
  const incoming = Object.entries(p.terms ?? {});
  if (jobKey !== lastTerms.current.job || incoming.length > 0) {
    lastTerms.current = { job: jobKey, terms: incoming };
  }
  const terms = lastTerms.current.terms;

  // --- stage inspector state -----------------------------------------------
  // Clicking a strip segment selects a stage; null = follow the ACTIVE stage
  // (the default, so a handoff moves the inspector along until the user picks
  // one). Edits are per-stage slider moves layered over the streamed
  // stageWeights; both reset when a NEW job starts (jobKey strips the -sN
  // suffix, so handoffs within one chain keep them).
  const [stageSel, setStageSel] = useState<number | null>(null);
  const [stageEdits, setStageEdits] = useState<Record<number, Record<string, number>>>({});
  const stageJobRef = useRef(jobKey);
  useEffect(() => {
    if (stageJobRef.current !== jobKey) {
      stageJobRef.current = jobKey;
      setStageSel(null);
      setStageEdits({});
    }
  }, [jobKey]);
  const curriculum = t.behavior.curriculum ?? [];
  const stageStart = stage?.start ?? 1;
  const selStage = stageSel ?? stage?.idx ?? 1;
  const live = t.status === "training";
  /** Behavior-level weight for a key (what a stage inherits sans override). */
  const baseWeight = (key: string) =>
    t.weights[key] ?? t.behavior.terms.find((term) => term.key === key)?.weight ?? 0;
  const stageOverrides = (i: number) => t.stageWeights?.[String(i)] ?? {};
  const stageWeight = (i: number, key: string) =>
    stageEdits[i]?.[key] ?? stageOverrides(i)[key] ?? baseWeight(key);
  const stageDirty = Object.values(stageEdits).some((m) => Object.keys(m).length > 0);
  /** The FULL per-stage override map the server expects: streamed overrides
   *  with local edits folded in, dropping keys dragged back to the
   *  behavior-level value (that's how a stage override is removed). */
  const fullStageWeights = (): StageWeightsMap => {
    const out: StageWeightsMap = {};
    const count = stage?.count ?? curriculum.length;
    for (let i = 1; i <= count; i++) {
      const merged = { ...stageOverrides(i), ...(stageEdits[i] ?? {}) };
      const kept: Record<string, number> = {};
      for (const [k, v] of Object.entries(merged)) {
        if (Math.abs(v - baseWeight(k)) > 1e-9) kept[k] = v;
      }
      if (Object.keys(kept).length) out[String(i)] = kept;
    }
    return out;
  };
  const stageWeightsOrNull = () => {
    const sw = fullStageWeights();
    return Object.keys(sw).length ? sw : null;
  };
  const maxAbs = Math.max(0.01, ...terms.map(([, v]) => Math.abs(v)));
  const statusLine = {
    training: tr(
      `training… ${overallSteps.toLocaleString()} / ${overallTotal.toLocaleString()} practice steps`,
      `训练中… ${overallSteps.toLocaleString()} / ${overallTotal.toLocaleString()} 个练习步`
    ),
    done: tr("✔ finished — the trainee duck runs the final result", "✔ 已完成 — 学员鸭正在运行最终结果"),
    stopped: tr("■ stopped — trainee keeps the last snapshot", "■ 已停止 — 学员鸭保留最后一个快照"),
    failed: tr("✗ training crashed (see runs/…/train.log)", "✗ 训练崩溃（请查看 runs/…/train.log）"),
  }[t.status];

  return (
    <div style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: 8, marginTop: 8 }}>
      <div style={{ fontWeight: 700 }}>
        {t.behavior.emoji} {t.behavior.title}
        {t.status === "training" && (
          <button
            onClick={onStop}
            style={{
              float: "right", background: "#3a2622", color: "#e0a08f",
              border: "1px solid #5a3a33", borderRadius: 5, padding: "1px 8px",
              fontFamily: mono, fontSize: 11, cursor: "pointer",
            }}
          >
            {tr("stop", "停止")}
          </button>
        )}
      </div>
      <div style={{ color: "#8b93a3", margin: "4px 0" }}>{statusLine}</div>
      {t.status === "training" && (
        <div style={{ color: "#8b93a3", fontSize: 10, marginBottom: 4 }}>
          {tr(
            `practicing on ${t.envs} parallel ducks (${t.helpers} helper${t.helpers === 1 ? "" : "s"})`,
            `正在用 ${t.envs} 只并行鸭练习（${t.helpers} 个辅助实例）`
          )}
          {traineeSpeed != null && (
            <span title={tr(
              "how fast the trainee duck is actually walking right now, forward, in metres per second",
              "学员鸭当前实际向前行走速度，单位为米/秒"
            )}>
              {tr(" · now going ", " · 当前速度 ")}
              <span style={{ color: "#7db8d8" }}>
                {traineeSpeed.toFixed(2)} m/s
              </span>
            </span>
          )}
          {t.restarting && (
            <span style={{ color: "#d8c97d" }}>
              {tr(" · restarting the trainer…", " · 正在重启训练器…")}
            </span>
          )}
        </div>
      )}
      {/* Curriculum strip: one CLICKABLE segment per stage — done filled,
          current pulsing, future dim, pre-startStage stages muted (they were
          skipped, trained in an earlier run). Clicking opens the stage in the
          inspector below; the active stage is inspected by default. */}
      {stage && (
        <div style={{ margin: "2px 0 5px" }}>
          <style>{"@keyframes ducklab-stage-pulse { 50% { opacity: 0.35 } }"}</style>
          <div style={{ display: "flex", gap: 3 }}>
            {Array.from({ length: stage.count }, (_, i) => (
              <div
                key={i}
                onClick={() => setStageSel(i + 1)}
                style={{
                  flex: 1,
                  height: 5,
                  borderRadius: 2,
                  cursor: "pointer",
                  outline: i + 1 === selStage ? "1px solid #d8c97d" : undefined,
                  outlineOffset: 1,
                  background:
                    i + 1 < stageStart
                      ? "#31414f"
                      : i + 1 < stage.idx || (i + 1 === stage.idx && t.status === "done")
                        ? "#7db8d8"
                        : i + 1 === stage.idx
                          ? "#9fb4d8"
                          : "#262a33",
                  animation:
                    i + 1 === stage.idx && t.status === "training"
                      ? "ducklab-stage-pulse 1.6s ease-in-out infinite"
                      : undefined,
                }}
              />
            ))}
          </div>
          <div style={{ color: "#9fb4d8", fontSize: 10, marginTop: 2 }}>
            {tr(`stage ${stage.idx} of ${stage.count}`, `第 ${stage.idx}/${stage.count} 阶段`)} · {stage.label}
            {stageStart > 1 && (
              <span style={{ color: "#8b93a3" }}>
                {tr(` · started at stage ${stageStart}`, ` · 从第 ${stageStart} 阶段开始`)}
              </span>
            )}
          </div>
          {/* Stage inspector: what the selected stage rehearses + its merged
              weights. Editing the ACTIVE stage warm-restarts it on apply;
              editing a FUTURE stage records for its launch. */}
          <div
            style={{
              marginTop: 4,
              background: "#12151b",
              border: "1px solid rgba(255,255,255,0.10)",
              borderRadius: 6,
              padding: "5px 7px",
            }}
          >
            <div style={{ color: "#9fb4d8", fontSize: 10, fontWeight: 700 }}>
              {tr(`stage ${selStage}`, `第 ${selStage} 阶段`)}
              {selStage === stage.idx && tr(" (active)", "（当前）")} ·{" "}
              {curriculum[selStage - 1]?.label ?? ""}
            </div>
            {selStage < stageStart && (
              <div style={{ color: "#d8c97d", fontSize: 10, marginTop: 2 }}>
                {tr(
                  "skipped this run — its result came from an earlier training",
                  "本次运行已跳过 — 结果来自更早的训练"
                )}
              </div>
            )}
            {curriculum[selStage - 1]?.detail && (
              <div style={{ color: "#aab3c0", fontSize: 10, marginTop: 3, lineHeight: 1.45 }}>
                {curriculum[selStage - 1].detail}
              </div>
            )}
            {/* Per-stage practice budget. The chain's total is split across
                the stages in the recipe's proportions; this pins ONE stage to
                a number of its own — the per-stage weight overrides' shape,
                for time instead of points. */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 5,
                flexWrap: "wrap",
                marginTop: 5,
              }}
            >
              <Tip
                tip={
                  <>
                    <div>{tr("How long this stage practices for.", "此阶段的练习时长。")}</div>
                    <div style={{ color: "#8b93a3", marginTop: 3 }}>
                      {pinned[String(selStage)] != null
                        ? tr(
                            "you set this one by hand — ✕ hands it back its share of the total",
                            "此数值由你手动设置 — 点击 ✕ 可恢复为总预算中的比例份额"
                          )
                        : tr(
                            "its share of the total below, kept in the recipe's proportions",
                            "按配方比例分配下方总预算"
                          )}
                      {live && tr(
                        " · takes effect when you retrain or start from a stage",
                        " · 重新训练或从某阶段开始时生效"
                      )}
                    </div>
                  </>
                }
              >
                <span style={{ color: "#8b93a3", fontSize: 10 }}>
                  {tr("practices for", "练习时长")}
                </span>
              </Tip>
              <MStepsInput
                value={plan[selStage - 1] ?? null}
                disabled={live}
                width={50}
                onCommit={(steps) => onPinStage(selStage, steps)}
              />
              <span style={{ color: "#8b93a3", fontSize: 10 }}>
                {tr("M steps", "百万步")}
              </span>
              {!live && pinned[String(selStage)] != null && (
                <button
                  aria-label={tr(
                    `give stage ${selStage} its share of the total back`,
                    `将第 ${selStage} 阶段恢复为总预算中的比例份额`
                  )}
                  onClick={() => onPinStage(selStage, null)}
                  style={{
                    background: "none",
                    border: "none",
                    color: "#8b93a3",
                    cursor: "pointer",
                    fontFamily: mono,
                    fontSize: 10,
                    padding: "0 2px",
                    lineHeight: 1,
                  }}
                >
                  ✕
                </button>
              )}
              <span style={{ color: "#8b93a3", fontSize: 10 }}>
                {pinned[String(selStage)] != null
                  ? tr("· set by you", "· 由你设置")
                  : tr("· its share", "· 按比例分配")}
              </span>
            </div>
            <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 4 }}>
              {tr(
                "this stage's weights (stage overrides win over the chain sliders):",
                "此阶段的权重（阶段覆盖值优先于整条训练链的滑块）："
              )}
            </div>
            {byPolarity(t.behavior.terms).map((term) => {
              const def = term.weight;
              const max = def > 0 ? def * 2.5 : 1;
              const value = stageWeight(selStage, term.key);
              const overridden =
                stageEdits[selStage]?.[term.key] != null ||
                stageOverrides(selStage)[term.key] != null;
              return (
                <div
                  key={term.key}
                  style={{ display: "flex", alignItems: "center", gap: 5, margin: "2px 0" }}
                >
                  <span
                    style={{
                      width: 92,
                      flexShrink: 0,
                      color: term.isPenalty ? "#e0a08f" : "#9fd89f",
                      fontSize: 9,
                      textAlign: "right",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={term.friendly}
                  >
                    {term.key.replace(/_/g, " ")}
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={max}
                    step={0.05}
                    value={value}
                    onChange={(e) => {
                      const v = Number(e.target.value);
                      setStageEdits((prev) => ({
                        ...prev,
                        [selStage]: { ...prev[selStage], [term.key]: v },
                      }));
                    }}
                    style={{
                      flex: 1,
                      minWidth: 0,
                      height: 12,
                      accentColor: term.isPenalty ? "#e07a5f" : "#7dd87d",
                    }}
                  />
                  <span
                    style={{
                      width: 30,
                      flexShrink: 0,
                      textAlign: "right",
                      fontSize: 9,
                      color: overridden ? "#e8e6e1" : "#8b93a3",
                    }}
                  >
                    {value.toFixed(2)}
                  </span>
                </div>
              );
            })}
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 5 }}>
              {live && stageDirty && (
                <button
                  onClick={() => {
                    onStageWeights(fullStageWeights());
                    setStageEdits({});
                  }}
                  style={{
                    background: "#1c2230",
                    color: "#9fb4d8",
                    border: "1px solid rgba(255,255,255,0.14)",
                    borderRadius: 6,
                    padding: "2px 8px",
                    fontFamily: mono,
                    fontSize: 10,
                    cursor: "pointer",
                  }}
                >
                  {tr("✓ apply stage weights", "✓ 应用阶段权重")}
                </button>
              )}
              <Tip
                tip={
                  live
                    ? tr(
                        "stop the current run first — starting a stage launches a new training chain",
                        "请先停止当前运行 — 从某阶段开始会启动新的训练链"
                      )
                    : selStage === 1
                      ? tr(
                          "trains the whole chain from the beginning (same as retrain)",
                          "从头训练整条训练链（与重新训练相同）"
                        )
                      : tr(
                          `skips stages 1–${selStage - 1}: the chain warm-starts from your newest trained stage ${selStage - 1} run — if none exists, the server explains that in the chat`,
                          `跳过第 1–${selStage - 1} 阶段：训练链会从最近完成的第 ${selStage - 1} 阶段运行热启动；如果不存在，服务器会在对话中说明`
                        )
                }
              >
                <button
                  disabled={live}
                  onClick={() => {
                    onStartStage(selStage, stageWeightsOrNull());
                    setStageEdits({});
                  }}
                  style={{
                    background: live ? "#171b23" : "#1c2230",
                    color: live ? "#57627a" : "#d8c97d",
                    border: "1px solid rgba(255,255,255,0.14)",
                    borderRadius: 6,
                    padding: "2px 8px",
                    fontFamily: mono,
                    fontSize: 10,
                    cursor: live ? "default" : "pointer",
                  }}
                >
                  {tr("▶ start from this stage", "▶ 从此阶段开始")}
                </button>
              </Tip>
            </div>
            {live && stageDirty && (
              <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 3 }}>
                {tr(
                  "applying to the active stage restarts it warm; future-stage edits wait for their launch",
                  "应用到当前阶段会进行热重启；未来阶段的修改将在启动时生效"
                )}
              </div>
            )}
            {!live && stageDirty && (
              <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 3 }}>
                {tr(
                  "stage weights ride along when you retrain or start from a stage",
                  "重新训练或从某阶段开始时会一并使用这些阶段权重"
                )}
              </div>
            )}
          </div>
        </div>
      )}
      <div style={{ height: 6, background: "#262a33", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: "#7db8d8" }} />
      </div>
      {/* Thin secondary bar: THIS stage's progress (the main bar is the
          whole chain — without this the handoffs look like a stall). */}
      {stage && (
        <div
          style={{
            height: 3, background: "#20242c", borderRadius: 2,
            overflow: "hidden", marginTop: 2,
          }}
        >
          <div style={{ width: `${stagePct}%`, height: "100%", background: "#57627a" }} />
        </div>
      )}
      {rewHistory.length > 1 && (
        <div style={{ marginTop: 6 }}>
          <div style={{ color: "#8b93a3", fontSize: 10 }}>
            {tr(
              "score per practice run (higher = doing the trick better)",
              "每次练习的得分（越高表示动作完成得越好）"
            )}
          </div>
          <Sparkline points={rewHistory} />
        </div>
      )}
      {terms.length > 0 && (
        <div style={{ marginTop: 4 }}>
          <div style={{ color: "#8b93a3", fontSize: 10, marginBottom: 2 }}>
            {tr("where the points come from right now", "当前得分来源")}
          </div>
          {terms.map(([k, v]) => (
            <div key={k} style={{ display: "flex", alignItems: "center", gap: 6, margin: "2px 0" }}>
              <span style={{ width: 120, color: "#aab3c0", fontSize: 10, textAlign: "right" }}>
                {k.replace(/_penalty$/, "").replace(/_/g, " ")}
              </span>
              <div style={{ flex: 1, height: 5, background: "#20242c", borderRadius: 2, position: "relative" }}>
                <div
                  style={{
                    position: "absolute",
                    left: v >= 0 ? "50%" : `${50 - (Math.abs(v) / maxAbs) * 50}%`,
                    width: `${(Math.abs(v) / maxAbs) * 50}%`,
                    height: "100%",
                    background: v >= 0 ? "#7dd87d" : "#e07a5f",
                    borderRadius: 2,
                  }}
                />
              </div>
              <span style={{ width: 44, fontSize: 10, color: v >= 0 ? "#7dd87d" : "#e07a5f" }}>
                {v >= 0 ? "+" : ""}{v.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}
      {(p.snapshots ?? 0) > 0 && (
        <div style={{ color: "#d8c97d", fontSize: 10, marginTop: 4 }}>
          {tr(
            `📸 ${p.snapshots} snapshot${(p.snapshots ?? 0) > 1 ? "s" : ""} sent to the 🎓 duck — watch it improve in the scene`,
            `📸 已向 🎓 学员鸭发送 ${p.snapshots} 个快照 — 可在场景中观察它进步`
          )}
        </div>
      )}
      <RecipeEditor
        t={t}
        wide={wide}
        // Pending stage edits ride the retrain (the fine-tune path is a
        // single run — the server drops them there).
        onSubmit={(w, ft) => onRecipeSubmit(w, ft, stageWeightsOrNull())}
      />
    </div>
  );
}

export function TeachPanel({
  clientRef,
  variant = "overlay",
}: {
  clientRef: React.MutableRefObject<LabClient | null>;
  variant?: "overlay" | "embedded";
}) {
  const { t } = useLanguage();
  const embedded = variant === "embedded";
  // Collapsed by default, like the PolicyPanel above it — persisted after
  // the first open.
  const [open, setOpen] = useState(() => embedded || loadJSON("teachOpen", false));
  const [wide, setWide] = useState(() => loadJSON("teachWide", false));
  const policyOpen = usePolicyOpen();
  const [msgs, setMsgs] = useState<Msg[]>(() => {
    const stored = loadJSON<Msg[] | null>("teachMsgs", null);
    return Array.isArray(stored) && stored.length
      ? stored.slice(-MSG_CAP)
      : [{
          kind: "note",
          text: t(
            "Ask me to teach the duck a trick — try one of the suggestions below.",
            "告诉我想让鸭子学什么动作，也可以试试下面的建议。"
          ),
        }];
  });
  const [input, setInput] = useState("");
  const [training, setTraining] = useState<TrainingPayload | null>(null);
  // Read off the streamed roster rather than the trainer: the trainee duck in
  // the scene runs the newest snapshot, so this is the speed the user is
  // actually watching.
  const [traineeSpeed, setTraineeSpeed] = useState<number | null>(null);
  // How long the NEXT run should practice, in steps. A PENDING edit only —
  // null means "leave it to the trick", and the server remembers each
  // behavior's last choice exactly like it remembers the weight sliders, so
  // after a launch this goes back to null and the stream is the truth.
  const [budgetSteps, setBudgetSteps] = useState<number | null>(null);
  // Un-launched per-stage pins, keyed by 1-based stage; null = "hand this
  // stage back its proportional share".
  const [stagePins, setStagePins] = useState<Record<number, number | null>>({});
  const behaviorRef = useRef<string | null>(null);
  const rewHistory = useRef<{ x: number; y: number }[]>([]);
  const lastSteps = useRef(-1);
  // Which JOB the sparkline's points belong to (runName sans the -sN stage
  // suffix). The history must live and die with the job: points are keyed by
  // its cumulative step counter, so mixing jobs — or keeping points a warm
  // restart is about to re-earn — folds the path back on itself and draws
  // stray lines across the chart.
  const histJob = useRef("");
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!embedded) saveJSON("teachOpen", open);
  }, [embedded, open]);
  useEffect(() => {
    if (!embedded) saveJSON("teachWide", wide);
  }, [embedded, wide]);
  useEffect(() => saveJSON("teachMsgs", msgs.slice(-MSG_CAP)), [msgs]);

  // Poll the streamed frame for training progress + one-shot events.
  useEffect(() => {
    const id = setInterval(() => {
      const frame = clientRef.current?.frame;
      if (!frame) return;
      const t = frame.training ?? null;
      setTraining(t);
      setTraineeSpeed(frame.ducks.find((d) => d.id === "trainee")?.speed ?? null);
      // Pending budget edits belong to the trick on screen: a different one
      // has different stages, and its own remembered budget.
      if (t && t.behavior.id !== behaviorRef.current) {
        behaviorRef.current = t.behavior.id;
        setBudgetSteps(null);
        setStagePins({});
      }
      // Overall steps (falling back to per-stage for single runs): a
      // stage-local x would rewind to 0 at every curriculum handoff and fold
      // the sparkline back on itself.
      const steps = t?.progress?.overallSteps ?? t?.progress?.steps;
      if (t != null && steps != null) {
        const job = (t.runName ?? "").replace(/-s\d+$/, "");
        if (job !== histJob.current) {
          // A job this panel didn't launch (another tab, a relaunch) — its
          // counter starts over, so the old points can't share an axis.
          histJob.current = job;
          rewHistory.current = [];
          lastSteps.current = -1;
        }
        if (steps !== lastSteps.current) {
          if (steps < lastSteps.current) {
            // Same job, counter rewound: a warm restart (stage-weight apply,
            // helper rescale, crash recovery) is replaying this stretch —
            // drop the stale tail so the new attempt redraws it.
            rewHistory.current = rewHistory.current.filter((p) => p.x < steps);
          }
          lastSteps.current = steps;
          if (t.progress.ep_rew != null)
            rewHistory.current.push({ x: steps, y: t.progress.ep_rew });
          if (rewHistory.current.length > 400) rewHistory.current.shift();
        }
      }
      for (const ev of frame.events ?? []) {
        if (/Trainee|Training/.test(ev))
          setMsgs((m) =>
            m[m.length - 1]?.kind === "note" && (m[m.length - 1] as { text?: string }).text === ev
              ? m
              : [...m, { kind: "note", text: ev }]
          );
      }
    }, 500);
    return () => clearInterval(id);
  }, [clientRef]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [msgs, training?.status]);

  // Selecting a duck (stage click or HUD row) pulls ITS run up in this panel
  // — recipe card + sliders in "done" state, ✨ fine-tune targeting that run —
  // so "click the duck, keep refining it" is one gesture. Quiet no-op for
  // shipped policies (nothing to refine), while actively training (never
  // yank a live job), and when that run is already up. The server streams
  // the loaded payload back, so the panel updates through the normal poll.
  const selectedDuck = useSelectedDuck();
  useEffect(() => {
    if (!selectedDuck) return;
    const frame = clientRef.current?.frame;
    const pid = frame?.ducks.find((d) => d.id === selectedDuck)?.policy;
    if (!pid || !isRunPolicy(pid)) return;
    const t = frame?.training ?? null;
    if (t?.status === "training" || t?.restarting) return;
    if (t && runNameOfPolicy(pid) === t.runName) return;
    loadTeachRun(pid)
      .then((r) => {
        // Refusals surface in the chat (the success toast is the server's).
        if (!r.ok && r.message)
          setMsgs((m) => [...m, { kind: "note", text: `⚠ ${r.message}` }]);
      })
      .catch(() => {});
  }, [selectedDuck, clientRef]);

  /** POST /teach and fold the response into the chat. The practice budget
   *  rides along on every launch path (typed trick, suggestion, retrain,
   *  fine-tune, start-from-stage) — it's one control, so it applies to
   *  whatever you start next. Omitted when untouched, which is what makes
   *  the server's per-behavior memory the fallback. */
  async function postTeach(body: {
    text: string;
    weights?: Record<string, number>;
    stageWeights?: Record<string, Record<string, number>>;
    stageSteps?: Record<string, number>;
    startStage?: number;
    initFrom?: string;
  }) {
    try {
      const res = await fetch(`${LAB_HTTP}/teach`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...body,
          ...(budgetSteps != null ? { steps: budgetSteps } : {}),
        }),
      });
      const data = await res.json();
      if (data.matched) {
        // The sparkline history is NOT cleared here: the poll loop resets it
        // when the new job's frames actually arrive — clearing early would
        // let straggler frames from the old job repopulate it.
        // The job's own numbers are the truth from here on; the pending
        // edits have been spent.
        setTraining(data.job);
        behaviorRef.current = data.job.behavior.id;
        setBudgetSteps(null);
        setStagePins({});
        setMsgs((m) => [
          ...m,
          {
            kind: "note",
            text: t(
              "On it! Here's the deal I'm offering the duck:",
              "开始！这是我给鸭子安排的训练方案："
            ),
          },
          {
            kind: "card",
            card: data.job.behavior,
            stageSteps: data.job.stageSteps,
            stepBudget: data.job.stepBudget,
          },
        ]);
      } else {
        setMsgs((m) => [
          ...m,
          { kind: "note", text: data.message },
          ...(data.behaviors ?? []).map((b: BehaviorCard) => ({
            kind: "note" as const,
            text: `${b.emoji} “${b.title}” — ${b.description}`,
          })),
        ]);
      }
    } catch {
      setMsgs((m) => [
        ...m,
        {
          kind: "note",
          text: t(
            "⚠ can't reach the lab server on :8788",
            "⚠ 无法连接 :8788 上的实验室服务器"
          ),
        },
      ]);
    }
  }

  async function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed) return;
    setInput("");
    setMsgs((m) => [...m, { kind: "user", text: trimmed }]);
    await postTeach({ text: trimmed });
  }

  /** Recipe buttons: resubmit the current behavior with the moved sliders,
   *  fresh (retrain) or warm-started from the finished run (fine-tune).
   *  Pending per-stage edits ride the retrain path (fine-tunes are single
   *  runs — the server ignores stage weights there, so don't send them). */
  async function submitRecipe(
    weights: Record<string, number>,
    fineTune: boolean,
    stageWeights: Record<string, Record<string, number>> | null
  ) {
    if (!training) return;
    const n = Object.keys(weights).length;
    const tweak = n
      ? t(
          ` with ${n} adjusted weight${n > 1 ? "s" : ""}`,
          `（已调整 ${n} 个权重）`
        )
      : "";
    setMsgs((m) => [
      ...m,
      {
        kind: "note",
        text: fineTune
          ? t(
              `✨ fine-tuning ${training.runName}${tweak} — keeping what it learned`,
              `✨ 正在微调 ${training.runName}${tweak} — 保留已有能力`
            )
          : t(
              `↻ retraining “${training.behavior.title}” from scratch${tweak}`,
              `↻ 正在从头重新训练“${training.behavior.title}”${tweak}`
            ),
      },
    ]);
    await postTeach({
      text: training.behavior.title,
      ...(n ? { weights } : {}),
      ...(fineTune ? { initFrom: training.runName } : {}),
      ...(!fineTune && stageWeights ? { stageWeights } : {}),
      // Per-stage pins are a chain thing, like per-stage weights — a
      // fine-tune is one run and the server drops them there anyway.
      ...(!fineTune && pinsDirty ? { stageSteps: pins } : {}),
    });
  }

  /** Stage inspector: live per-stage weight edit on the active chain. The
   *  server records future stages and warm-restarts the active one when its
   *  merged weights changed. */
  async function applyStageWeights(stageWeights: Record<string, Record<string, number>>) {
    try {
      const res = await fetch(`${LAB_HTTP}/teach/weights`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stageWeights }),
      });
      const data = await res.json();
      setMsgs((m) => [
        ...m,
        {
          kind: "note",
          text: data.ok
            ? data.restarted
              ? t(
                  "⚖ stage weights applied — restarting the current stage warm",
                  "⚖ 已应用阶段权重 — 正在热重启当前阶段"
                )
              : t(
                  "⚖ stage weights recorded — future stages launch with them",
                  "⚖ 已记录阶段权重 — 后续阶段启动时将使用它们"
                )
            : `⚠ ${data.message}`,
        },
      ]);
    } catch {
      setMsgs((m) => [
        ...m,
        {
          kind: "note",
          text: t(
            "⚠ can't reach the lab server on :8788",
            "⚠ 无法连接 :8788 上的实验室服务器"
          ),
        },
      ]);
    }
  }

  /** Stage inspector: relaunch the behavior's chain starting at `idx`. The
   *  server resolves the warm start from the newest trained prior-stage run
   *  and REFUSES with a plain message when none exists — that message lands
   *  in this chat log (chosen over plumbing a per-stage "trainable" flag). */
  async function startFromStage(
    idx: number,
    stageWeights: Record<string, Record<string, number>> | null
  ) {
    if (!training) return;
    setMsgs((m) => [
      ...m,
      {
        kind: "note",
        text: t(
          `▶ starting “${training.behavior.title}” from stage ${idx}`,
          `▶ 正在从第 ${idx} 阶段开始“${training.behavior.title}”`
        ),
      },
    ]);
    await postTeach({
      text: training.behavior.title,
      startStage: idx,
      ...(stageWeights ? { stageWeights } : {}),
      ...(pinsDirty ? { stageSteps: pins } : {}),
    });
  }

  // --- what the next run will practice for ---------------------------------
  // The pins in force: what the server streams back, with the user's
  // un-launched edits layered on (null removes one).
  const pins: Record<string, number> = { ...(training?.stageBudgets ?? {}) };
  for (const [k, v] of Object.entries(stagePins)) {
    if (v == null) delete pins[k];
    else pins[k] = v;
  }
  const pinsDirty = Object.keys(stagePins).length > 0;
  const declared = training ? declaredStages(training.behavior) : [];
  const declaredTotal = declared.reduce((s, v) => s + v, 0);
  // Untouched, the run's OWN per-stage numbers are the truth; once the user
  // edits anything, preview the split the server will compute (re-splitting
  // around the chosen total, not the pinned sum).
  const plan: number[] =
    budgetSteps == null && !pinsDirty && training?.stageSteps?.length
      ? training.stageSteps
      : resolveStageSteps(declared, budgetSteps ?? training?.chosenBudget ?? null, pins);
  // Before the first trick there are no stages to split — the typed number
  // is the whole story.
  const planTotal = training ? plan.reduce((s, v) => s + v, 0) : (budgetSteps ?? 0);
  const budgetShown = budgetSteps ?? training?.chosenBudget ?? null;
  const offRecipe = training != null && (planTotal !== declaredTotal || Object.keys(pins).length > 0);
  const resetToRecipe = () => {
    setBudgetSteps(declaredTotal);
    const cleared: Record<number, number | null> = {};
    for (const k of Object.keys(pins)) cleared[Number(k)] = null;
    setStagePins(cleared);
  };

  // Publish this panel's measured height into the shared ui store so the
  // PolicyPanel above can grow into the space teach isn't using. One callback
  // ref serves both the open panel and the collapsed pill — only one of them
  // is mounted at a time.
  const teachRO = useRef<ResizeObserver | null>(null);
  const teachSizeRef = useCallback((el: HTMLElement | null) => {
    teachRO.current?.disconnect();
    teachRO.current = null;
    if (!el) {
      setTeachHeight(0);
      return;
    }
    const publish = () => setTeachHeight(el.getBoundingClientRect().height);
    publish();
    teachRO.current = new ResizeObserver(publish);
    teachRO.current.observe(el);
  }, []);

  const panel: React.CSSProperties = {
    position: embedded ? "relative" : "absolute",
    // Above the ducks' floating DOM labels (drei Html, zIndexRange [10, 0]).
    zIndex: 20,
    right: embedded ? "auto" : 14,
    bottom: embedded ? "auto" : 14,
    // Wide mode makes room for full recipe sentences beside the sliders. The
    // extra min() terms cap it responsively: the bottom-center Controls pad is
    // 118px wide, so its right edge sits at 50vw + 59px — our left edge
    // (100vw - 14px - width) stays right of it for any viewport width.
    width: embedded
      ? "100%"
      : wide
        ? "min(560px, calc(100cqw - 28px))"
        : "min(320px, calc(100cqw - 28px))",
    // Complementary to the PolicyPanel's NOMINAL cap (min(40vh, 380px)) plus
    // margins, so the two right-column panels can never overlap — but when
    // that panel is collapsed to its pill, reclaim the space and grow tall.
    // Stays keyed to that constant, never to the policy panel's measured
    // height: policies sizes itself off OUR measured height (lib/ui.ts), and
    // measuring each other both ways would make the pair oscillate.
    maxHeight: embedded
      ? "none"
      : policyOpen
        ? "calc(100cqh - min(40cqh, 380px) - 56px)"
        : "calc(100cqh - 100px)",
    height: embedded ? "100%" : "auto",
    display: "flex",
    flexDirection: "column",
    background: embedded ? "transparent" : "rgba(14, 16, 20, 0.86)",
    border: embedded ? "0" : "1px solid rgba(255,255,255,0.09)",
    borderRadius: embedded ? 0 : 10,
    color: "#e8e6e1",
    fontFamily: mono,
    fontSize: 12,
    lineHeight: 1.5,
    backdropFilter: embedded ? "none" : "blur(6px)",
  };

  if (!open && !embedded)
    return (
      <button
        ref={teachSizeRef}
        data-teach-ui
        onClick={() => setOpen(true)}
        style={{
          position: "absolute", zIndex: 20, right: 14, bottom: 14,
          background: "rgba(14,16,20,0.86)", color: "#e8e6e1",
          border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10,
          padding: "8px 12px", fontFamily: mono, fontSize: 12, cursor: "pointer",
          backdropFilter: "blur(6px)",
        }}
      >
        {t("🎓 teach", "🎓 教学")}
      </button>
    );

  return (
    // data-teach-ui doubles as the PolicyPanel's drop target: a policy chip
    // dropped (or armed-clicked) anywhere on this panel loads its run here.
    <div ref={embedded ? undefined : teachSizeRef} style={panel} data-teach-ui>
      <div
        style={{
          padding: "8px 12px", fontWeight: 700, fontSize: 13,
          borderBottom: "1px solid rgba(255,255,255,0.08)",
          display: "flex", alignItems: "center", flexShrink: 0,
        }}
      >
        <span style={{ flex: 1 }}>{t("🎓 teach", "🎓 教学")}</span>
        <button
          onClick={() => {
            setMsgs([{
              kind: "note",
              text: t(
                "Ask me to teach the duck a trick — try one of the suggestions below.",
                "告诉我想让鸭子学什么动作，也可以试试下面的建议。"
              ),
            }]);
            // Also dismiss a FINISHED training card — the farm keeps
            // broadcasting the job payload until told to let go (a running
            // job is protected server-side; stop it first).
            fetch(`${LAB_HTTP}/teach/clear`, { method: "POST" }).catch(() => {});
          }}
          title={t(
            "clear the conversation and any finished training card",
            "清除对话和已完成的训练卡片"
          )}
          style={{
            background: "none", border: "none", color: "#8b93a3",
            cursor: "pointer", fontFamily: mono, fontSize: 12, padding: "0 4px",
          }}
        >
          🗑
        </button>
        {!embedded && (
          <>
            <button
              onClick={() => setWide((w) => !w)}
              title={wide
                ? t("back to the narrow panel", "恢复窄面板")
                : t("widen the panel — full recipe sentences", "加宽面板以显示完整配方说明")}
              style={{
                background: "none", border: "none", color: "#8b93a3",
                cursor: "pointer", fontFamily: mono, fontSize: 12, padding: "0 4px",
              }}
            >
              {wide ? "⤡" : "⤢"}
            </button>
            <button
              onClick={() => setOpen(false)}
              title={t("collapse", "收起")}
              style={{
                background: "none", border: "none", color: "#8b93a3",
                cursor: "pointer", fontFamily: mono, fontSize: 12, padding: "0 4px",
              }}
            >
              —
            </button>
          </>
        )}
      </div>

      <div ref={logRef} style={{ overflowY: "auto", padding: "8px 12px", flex: 1 }}>
        {msgs.map((m, i) =>
          m.kind === "user" ? (
            <div key={i} style={{ textAlign: "right", margin: "6px 0" }}>
              <span style={{ background: "#2a3548", borderRadius: 8, padding: "3px 8px" }}>
                {m.text}
              </span>
            </div>
          ) : m.kind === "note" ? (
            <div key={i} style={{ color: "#aab3c0", margin: "6px 0" }}>
              {m.text}
            </div>
          ) : (
            <div
              key={i}
              style={{
                background: "#181c24", borderRadius: 8, padding: "8px 10px",
                margin: "6px 0", border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <div style={{ fontWeight: 700 }}>
                {m.card.emoji} {m.card.title}
              </div>
              <div style={{ color: "#aab3c0", margin: "3px 0" }}>{m.card.description}</div>
              {/* Staged tricks get their training plan spelled out up front —
                  the chain is part of the story, not trainer plumbing. */}
              {m.card.curriculum && m.card.curriculum.length > 0 && (
                <div style={{ margin: "4px 0" }}>
                  <div style={{ color: "#8b93a3", fontSize: 10 }}>
                    {t(
                      `how it trains — ${m.card.curriculum.length} stages, each building on the last:`,
                      `训练方式 — 共 ${m.card.curriculum.length} 个阶段，每个阶段都建立在上一阶段之上：`
                    )}
                  </div>
                  {m.card.curriculum.map((s, i) => (
                    // Hovering a stage shows its detail — what the practice
                    // actually looks like (the stage inspector's text).
                    <Tip key={i} tip={s.detail || s.label}>
                      <div style={{ color: "#aab3c0", fontSize: 11, margin: "2px 0 0 4px" }}>
                        {i + 1}. {s.label}
                        <span style={{ color: "#8b93a3" }}>
                          {/* The steps this run actually got, not the
                              recipe's — the two differ the moment a budget
                              is chosen. */}
                          {" "}· {fmtSteps(m.stageSteps?.[i] ?? s.steps)} {t("steps", "步")}
                        </span>
                      </div>
                    </Tip>
                  ))}
                </div>
              )}
              <details style={{ margin: "4px 0" }}>
                <summary style={{ cursor: "pointer", color: "#7db8d8" }}>
                  {t("how will it learn this?", "它会怎样学习？")}
                </summary>
                <div style={{ color: "#aab3c0", marginTop: 4 }}>{m.card.howItLearns}</div>
                <div style={{ color: "#8b93a3", marginTop: 4, fontSize: 10 }}>
                  {t(
                    `The sim runs far faster than real life, so ${fmtSteps(m.stepBudget ?? cardSteps(m.card))} practice steps run on this Mac without you waiting on a real robot.`,
                    `仿真速度远快于现实，因此可直接在这台 Mac 上完成 ${fmtSteps(m.stepBudget ?? cardSteps(m.card))} 个练习步，无需等待真实机器人。`
                  )}
                </div>
              </details>
              <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 2 }}>
                {t("the scorecard (checked 50× per second):", "评分表（每秒检查 50 次）：")}
              </div>
              <RecipeRows terms={m.card.terms} />
            </div>
          )
        )}
        {training && (
          <LiveTraining
            t={training}
            traineeSpeed={traineeSpeed}
            wide={wide}
            rewHistory={rewHistory.current}
            plan={plan}
            pinned={pins}
            onPinStage={(stage, steps) =>
              setStagePins((p) => ({ ...p, [stage]: steps }))
            }
            onStop={() => fetch(`${LAB_HTTP}/teach/stop`, { method: "POST" })}
            onRecipeSubmit={submitRecipe}
            onStageWeights={applyStageWeights}
            onStartStage={startFromStage}
          />
        )}
      </div>

      <div style={{ padding: "8px 12px", borderTop: "1px solid rgba(255,255,255,0.08)" }}>
        {/* How long should it practice? Sits with the ask, because that's
            when the question comes up — and it applies to whatever you start
            next: a typed trick, a suggestion, retrain, fine-tune, or a
            start-from-stage. */}
        <div style={{ marginBottom: 6 }}>
          <div
            style={{ display: "flex", alignItems: "center", gap: 5, flexWrap: "wrap" }}
          >
            <Tip
              tip={
                <>
                  <div>{t(
                    "How long the duck gets to practice, in millions of tries.",
                    "设置鸭子的练习时长，单位为百万次尝试。"
                  )}</div>
                  <div style={{ color: "#8b93a3", marginTop: 3 }}>
                    {t(
                      `More practice usually means a better trick and a longer wait. Type any number between ${fmtSteps(MIN_STEP_BUDGET)} and ${fmtSteps(MAX_STEP_BUDGET)}, or tap a preset. A trick with stages splits this across them, keeping the recipe's proportions.`,
                      `练习越多通常动作越好，但等待也更久。可输入 ${fmtSteps(MIN_STEP_BUDGET)} 到 ${fmtSteps(MAX_STEP_BUDGET)} 之间的任意数值，或选择预设。多阶段动作会按配方比例分配这些步数。`
                    )}
                  </div>
                </>
              }
            >
              <span style={{ color: "#8b93a3", fontSize: 10 }}>
                {t("practice for", "练习时长")}
              </span>
            </Tip>
            <MStepsInput
              value={budgetShown}
              placeholder={t("recipe", "配方")}
              onCommit={setBudgetSteps}
            />
            <span style={{ color: "#8b93a3", fontSize: 10 }}>
              {t("M steps", "百万步")}
            </span>
            {offRecipe && (
              <Tip tip={t(
                "back to the practice plan the recipe ships with",
                "恢复配方自带的练习计划"
              )}>
                <button
                  onClick={resetToRecipe}
                  style={{
                    background: "none",
                    color: "#8b93a3",
                    border: "1px dashed rgba(255,255,255,0.22)",
                    borderRadius: 12,
                    padding: "1px 7px",
                    fontFamily: mono,
                    fontSize: 10,
                    cursor: "pointer",
                  }}
                >
                  {t("↺ recipe", "↺ 配方")}
                </button>
              </Tip>
            )}
          </div>
          <div
            style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap", marginTop: 3 }}
          >
            {BUDGET_PRESETS_M.map((m) => {
              const on = budgetShown === m * 1e6;
              return (
                <button
                  key={m}
                  onClick={() => setBudgetSteps(m * 1e6)}
                  style={{
                    background: on ? "#2a3548" : "#1c2230",
                    color: on ? "#e8e6e1" : "#9fb4d8",
                    border: `1px solid ${on ? "#7db8d8" : "rgba(255,255,255,0.08)"}`,
                    borderRadius: 12,
                    padding: "1px 7px",
                    fontFamily: mono,
                    fontSize: 10,
                    cursor: "pointer",
                  }}
                >
                  {m}M
                </button>
              );
            })}
          </div>
          <div style={{ color: "#8b93a3", fontSize: 10, marginTop: 2 }}>
            {planTotal > 0
              ? t(
                  `${fmtSteps(planTotal)} practice steps in total`,
                  `总计 ${fmtSteps(planTotal)} 个练习步`
                )
              : t(
                  "each trick practices for as long as its own recipe says — set a number to change that",
                  "每个动作默认按自身配方练习；设置数值可覆盖默认时长"
                )}
          </div>
          {plan.length > 1 && (
            <div style={{ color: "#8b93a3", fontSize: 10 }}>
              {t(`${plan.length} stages`, `${plan.length} 个阶段`)}:{" "}
              {plan.map((s) => fmtSteps(s)).join(" / ")}
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 6 }}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s.value}
              onClick={() => submit(s.value)}
              style={{
                background: "#1c2230", color: "#9fb4d8",
                border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12,
                padding: "2px 8px", fontFamily: mono, fontSize: 10, cursor: "pointer",
              }}
            >
              {t(s.value, s.zh)}
            </button>
          ))}
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={t("teach the duck a new policy…", "教鸭子一个新动作…")}
            style={{
              width: "100%", boxSizing: "border-box", background: "#12151b",
              border: "1px solid rgba(255,255,255,0.12)", borderRadius: 6,
              color: "#e8e6e1", padding: "6px 8px", fontFamily: mono, fontSize: 12,
              outline: "none",
            }}
          />
        </form>
      </div>
    </div>
  );
}
