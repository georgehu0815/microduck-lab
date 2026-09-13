"use client";

// In-app operating guide for Duck Viewer. The policy notes are intentionally
// specific to the nine vendored Pollen policies: this is a verification guide,
// not a promise that every robotd command sequence is reproduced by the lab.

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Frame, LabClient } from "@/lib/lab";
import { useLanguage } from "./LanguageProvider";
import { pushToast } from "./Toasts";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

export const GUIDE_TABS = ["Quick start", "Policies", "Verify", "Controls"] as const;
type GuideTab = (typeof GUIDE_TABS)[number];
type Support = "full" | "partial" | "smoke-test only";

export interface ShippedPolicyGuide {
  name: string;
  purpose: string;
  semantics: string;
  support: Support;
  steps: readonly string[];
  expected: string;
  verify: readonly string[];
  limitation: string;
}

/** One card per ONNX file in microduck/policies. Keep this explicit: adding a
 * policy to the shipped bundle should force a deliberate guide update. */
export const SHIPPED_POLICY_GUIDES: readonly ShippedPolicyGuide[] = [
  {
    name: "alpha_walking",
    purpose: "Velocity-command walking / velstand gait.",
    semantics:
      "Uses the shared 61-value observation and 14 joint actions. Duck Lab's automatic runway asks vx = 0.9 m/s for 27 s, then zero for 3 s; the 30 s episode repeats.",
    support: "full",
    steps: [
      "Spawn it, or drag its policy chip onto an existing duck.",
      "Select the duck so its roster row and amber floor ring agree.",
      "Press R to restart every duck at the beginning of the same runway cycle.",
      "Watch the HUD m/s pair (achieved / asked) and falls through at least 30 s.",
    ],
    expected:
      "It should start from rest, walk forward during the 0.9 m/s request, then settle during the 3 s zero-command tail.",
    verify: [
      "Achieved speed remains meaningful beside the 0.9 m/s request.",
      "No fall is hidden by an unsynchronised start.",
      "Direction and recovery remain visually stable for a full cycle.",
    ],
    limitation:
      "This is deterministic local MuJoCo inspection, not the official GPU sim2real stack or hardware validation.",
  },
  {
    name: "alpha_stand",
    purpose: "Standing balance with trained head and body-pose control.",
    semantics:
      "robotd can drive head/body command fields. The Viewer sends zero head and body commands while still sharing the runway twist, so only the zero-pose slice is represented faithfully.",
    support: "partial",
    steps: [
      "Spawn it and select its roster row.",
      "Use Zero command for 6 seconds below to remove the shared runway twist temporarily.",
      "Press R, then inspect trunk, feet, head, and recovery from the reset pose.",
    ],
    expected: "At zero command it should seek a stable standing/body-neutral pose.",
    verify: [
      "It loads and produces a controlled, non-limp body response.",
      "Balance can be inspected without treating walking reward as its score.",
    ],
    limitation:
      "The Viewer cannot sweep or verify the policy's commanded head/body-pose range.",
  },
  {
    name: "alpha_sitstand",
    purpose: "Sit-to-stand posture transitions.",
    semantics:
      "The posture flag is encoded in twist vx: 1 means sit; 0 means rise/stand. The current 0.9-then-0 runway roughly exercises sit then rise, but it is not robotd’s exact sit/rise scheduler.",
    support: "partial",
    steps: [
      "Spawn it and press R to align the 30 s command cycle.",
      "Observe the long nonzero segment as the sit request.",
      "Observe the final 3 s zero segment for the rise/stand response.",
    ],
    expected: "A visible posture change toward sitting, followed by a rise/stand response when vx returns to zero.",
    verify: [
      "The two command phases produce distinct postures.",
      "The duck remains controlled across the transition.",
    ],
    limitation:
      "0.9 is only an approximate nonzero posture flag here, and the Viewer's 27 s / 3 s timing does not reproduce robotd’s transition timing.",
  },
  {
    name: "alpha_ground_pick",
    purpose: "Phase-driven ground-pick motion.",
    semantics:
      "robotd supplies twist = [cos(2πp), sin(2πp), 0], advances p over a nominal 4 s cycle, and hands back at phase 0.7.",
    support: "smoke-test only",
    steps: [
      "Spawn it to confirm the ONNX loads into a Viewer duck.",
      "Press R and check that the stream, roster, and body outputs remain alive.",
      "Do not grade the resulting movement as a ground pick.",
    ],
    expected: "Only a loaded policy and finite body response are expected in this Viewer path.",
    verify: [
      "The policy appears in the roster and produces renderable output.",
      "Treat any apparent trick motion as incidental without the phase command.",
    ],
    limitation:
      "The Viewer does not generate the cosine/sine phase command, so it cannot validate ground-pick behavior.",
  },
  {
    name: "ball_kick_left",
    purpose: "One-shot kick with the left leg.",
    semantics:
      "Expected twist is zero. robotd runs the kick network for a 0.5 s window, then hands control back to the normal policy scheduler.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and watch the left leg closely from a clear side/front angle.",
      "Capture a short video if frame-by-frame review is needed.",
    ],
    expected: "A brief left-leg kick-like response may be inspected near the start.",
    verify: ["Confirm the left leg is the active leg.", "Judge only the initial response, not a repeated raw-policy loop."],
    limitation:
      "A raw Viewer duck has no automatic 0.5 s robotd handoff, so repetition and recovery are not end-to-end kick validation.",
  },
  {
    name: "ball_kick_right",
    purpose: "One-shot kick with the right leg.",
    semantics:
      "Expected twist is zero. robotd runs the kick network for a 0.5 s window, then hands control back to the normal policy scheduler.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and watch the right leg closely from a clear side/front angle.",
      "Capture a short video if frame-by-frame review is needed.",
    ],
    expected: "A brief right-leg kick-like response may be inspected near the start.",
    verify: ["Confirm the right leg is the active leg.", "Judge only the initial response, not a repeated raw-policy loop."],
    limitation:
      "A raw Viewer duck has no automatic 0.5 s robotd handoff, so repetition and recovery are not end-to-end kick validation.",
  },
  {
    name: "roller",
    purpose: "Roller-mode locomotion.",
    semantics:
      "In robotd roller mode this replaces the walk network and uses the roller tuning preset, including action scale 0.8.",
    support: "partial",
    steps: [
      "Spawn it and select it to confirm policy loading.",
      "Press R and inspect the commanded body/joint response through one cycle.",
      "Record what the body does, but do not score wheel travel or traction.",
    ],
    expected: "A coherent roller-trained body response can be visually inspected.",
    verify: ["The interface loads and outputs remain controlled.", "Label captures as Viewer body-response inspection."],
    limitation:
      "The current lab scene and physics are not roller-mode hardware validation; wheel locomotion claims are unsupported.",
  },
  {
    name: "roller_crouch",
    purpose: "Roller-mode crouch occupying robotd’s ground-pick slot.",
    semantics:
      "Robot configuration identifies a nominal 5 s phase cycle and action scale 0.8 for the roller crouch.",
    support: "smoke-test only",
    steps: [
      "Spawn it to confirm loading and 61→14 interface compatibility.",
      "Press R and inspect only for finite, renderable joint/body output.",
      "Do not infer a valid crouch cycle from the runway command.",
    ],
    expected: "Only policy loading and a body response are expected.",
    verify: ["The duck remains in the roster and stream.", "No wheel or phase behavior is claimed."],
    limitation:
      "The Viewer neither validates roller physics nor supplies the phase drive; this is not crouch behavior validation.",
  },
  {
    name: "roulade",
    purpose: "One-shot forward roll.",
    semantics:
      "Expected command is zero. robotd runs a 1.0 s skill window, then hands off; a held request can chain another window.",
    support: "partial",
    steps: [
      "Spawn it, select it, and apply Zero command for 6 seconds.",
      "Press R and inspect the first second from a useful side angle.",
      "Record the attempt to inspect rotation, contact, and the state after the nominal window.",
    ],
    expected: "A forward-roll response may be visible during the initial one-second interval.",
    verify: ["Check forward rather than sideways/backward rotation.", "Separate the first window from later raw-policy motion."],
    limitation:
      "The raw Viewer lacks robotd’s timed handoff and request-based chaining, so full sequencing and recovery are not validated.",
  },
] as const;

const SHIPPED_POLICY_GUIDES_ZH: Record<
  string,
  Pick<
    ShippedPolicyGuide,
    "purpose" | "semantics" | "steps" | "expected" | "verify" | "limitation"
  >
> = {
  alpha_walking: {
    purpose: "速度指令行走 / velstand 步态。",
    semantics:
      "使用共享的 61 维观测和 14 个关节动作。Duck Lab 的自动跑道先请求 vx = 0.9 m/s 持续 27 秒，再归零 3 秒；30 秒回合循环重复。",
    steps: [
      "生成该策略，或将策略标签拖到现有小鸭上。",
      "选择小鸭，确认列表行与琥珀色地面环一致。",
      "按 R 让所有小鸭从同一跑道周期起点重新开始。",
      "观察 HUD 的 m/s 对（实际 / 请求）和跌倒次数，至少持续 30 秒。",
    ],
    expected:
      "它应从静止开始，在 0.9 m/s 请求期间向前行走，并在最后 3 秒零指令阶段稳定下来。",
    verify: [
      "实际速度与 0.9 m/s 请求并列时仍有意义。",
      "同步起点不会掩盖跌倒。",
      "整个周期内方向和恢复在视觉上保持稳定。",
    ],
    limitation:
      "这是确定性的本地 MuJoCo 检查，不是官方 GPU sim2real 流程或硬件验证。",
  },
  alpha_stand: {
    purpose: "带已训练头部和身体姿态控制的站立平衡。",
    semantics:
      "robotd 可驱动头部和身体指令字段。Viewer 发送零头部和身体指令，同时仍共享跑道 twist，因此只有零姿态切片得到忠实呈现。",
    steps: [
      "生成该策略并选择其列表行。",
      "使用下方“零指令 6 秒”暂时移除共享跑道 twist。",
      "按 R，然后检查躯干、脚、头部以及从重置姿态恢复的过程。",
    ],
    expected: "在零指令下，它应尝试保持稳定、身体中立的站立姿态。",
    verify: [
      "策略能够加载，并产生受控而非松软的身体响应。",
      "可以检查平衡，而不把行走奖励当作它的评分。",
    ],
    limitation: "Viewer 无法扫描或验证该策略的头部/身体姿态指令范围。",
  },
  alpha_sitstand: {
    purpose: "坐下到站立的姿态转换。",
    semantics:
      "姿态标志编码在 twist vx 中：1 表示坐下，0 表示起身/站立。当前先 0.9 后 0 的跑道大致覆盖坐下再起身，但并非 robotd 的精确调度。",
    steps: [
      "生成该策略并按 R 对齐 30 秒指令周期。",
      "将较长的非零阶段观察为坐下请求。",
      "观察最后 3 秒零指令阶段的起身/站立响应。",
    ],
    expected: "应出现向坐姿变化的明显动作，并在 vx 归零后出现起身/站立响应。",
    verify: ["两个指令阶段产生不同姿态。", "小鸭在转换过程中保持受控。"],
    limitation:
      "这里的 0.9 只是近似非零姿态标志，Viewer 的 27 秒 / 3 秒时序也不复现 robotd 的转换时序。",
  },
  alpha_ground_pick: {
    purpose: "由相位驱动的地面拾取动作。",
    semantics:
      "robotd 提供 twist = [cos(2πp), sin(2πp), 0]，在名义 4 秒周期内推进 p，并在相位 0.7 时交回控制。",
    steps: [
      "生成该策略，确认 ONNX 能加载到 Viewer 小鸭。",
      "按 R，确认数据流、列表和身体输出保持活动。",
      "不要把产生的动作评定为有效地面拾取。",
    ],
    expected: "此 Viewer 路径只要求策略已加载且身体响应为有限值。",
    verify: ["策略出现在列表中并产生可渲染输出。", "缺少相位指令时，任何类似技巧的动作都视为偶然。"],
    limitation: "Viewer 不生成余弦/正弦相位指令，因此无法验证地面拾取行为。",
  },
  ball_kick_left: {
    purpose: "左腿单次踢球。",
    semantics:
      "预期 twist 为零。robotd 运行踢球网络 0.5 秒，然后将控制交回正常策略调度器。",
    steps: [
      "生成并选择该策略，然后应用“零指令 6 秒”。",
      "按 R，从清晰的侧面/正面角度仔细观察左腿。",
      "如需逐帧检查，请录制短视频。",
    ],
    expected: "开始附近可能看到短暂的左腿踢球响应。",
    verify: ["确认左腿是主动腿。", "只评估初始响应，不评估重复的原始策略循环。"],
    limitation:
      "Viewer 原始小鸭没有 robotd 的自动 0.5 秒交接，因此重复和恢复不构成端到端踢球验证。",
  },
  ball_kick_right: {
    purpose: "右腿单次踢球。",
    semantics:
      "预期 twist 为零。robotd 运行踢球网络 0.5 秒，然后将控制交回正常策略调度器。",
    steps: [
      "生成并选择该策略，然后应用“零指令 6 秒”。",
      "按 R，从清晰的侧面/正面角度仔细观察右腿。",
      "如需逐帧检查，请录制短视频。",
    ],
    expected: "开始附近可能看到短暂的右腿踢球响应。",
    verify: ["确认右腿是主动腿。", "只评估初始响应，不评估重复的原始策略循环。"],
    limitation:
      "Viewer 原始小鸭没有 robotd 的自动 0.5 秒交接，因此重复和恢复不构成端到端踢球验证。",
  },
  roller: {
    purpose: "轮式模式运动。",
    semantics:
      "在 robotd 轮式模式中，此策略替代行走网络，并使用轮式调参预设，包括 0.8 的动作缩放。",
    steps: [
      "生成并选择该策略，确认策略加载。",
      "按 R，在一个周期内检查受指令驱动的身体/关节响应。",
      "记录身体动作，但不要评估轮子行程或牵引力。",
    ],
    expected: "可以目视检查连贯的轮式训练身体响应。",
    verify: ["界面成功加载且输出保持受控。", "将捕获内容标注为 Viewer 身体响应检查。"],
    limitation: "当前实验室场景和物理并非轮式硬件验证，不支持轮式运动结论。",
  },
  roller_crouch: {
    purpose: "占用 robotd 地面拾取槽位的轮式蹲伏。",
    semantics: "机器人配置为轮式蹲伏指定了名义 5 秒相位周期和 0.8 的动作缩放。",
    steps: [
      "生成该策略，确认加载以及 61→14 接口兼容性。",
      "按 R，只检查关节/身体输出是否为有限值且可渲染。",
      "不要根据跑道指令推断有效的蹲伏周期。",
    ],
    expected: "只要求策略加载并产生身体响应。",
    verify: ["小鸭保持在列表和数据流中。", "不声称验证了轮子或相位行为。"],
    limitation: "Viewer 既不验证轮式物理，也不提供相位驱动；这不是蹲伏行为验证。",
  },
  roulade: {
    purpose: "单次前滚翻。",
    semantics:
      "预期指令为零。robotd 运行 1.0 秒技巧窗口后交回控制；持续请求可串联另一个窗口。",
    steps: [
      "生成并选择该策略，然后应用“零指令 6 秒”。",
      "按 R，从合适的侧面角度检查第一秒。",
      "录制尝试，以检查旋转、接触以及名义窗口后的状态。",
    ],
    expected: "初始一秒内可能看到前滚响应。",
    verify: ["确认是向前旋转，而非侧向/向后。", "将第一个窗口与后续原始策略动作分开判断。"],
    limitation:
      "原始 Viewer 缺少 robotd 的定时交接和按请求串联，因此无法验证完整时序和恢复。",
  },
};

export function countPolicyDucks(frame: Frame | null, name: string): number {
  const id = `pollen:${name}`;
  return frame?.ducks.filter((duck) => duck.policy === id).length ?? 0;
}

const sectionTitle: React.CSSProperties = {
  color: "#e8e6e1",
  fontSize: 14,
  fontWeight: 700,
  margin: "0 0 8px",
};

const cardStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.035)",
  border: "1px solid rgba(255,255,255,0.09)",
  borderRadius: 9,
  padding: "11px 12px",
};

const actionStyle: React.CSSProperties = {
  background: "#1c2230",
  border: "1px solid rgba(255,255,255,0.14)",
  borderRadius: 7,
  color: "#cfe4f5",
  cursor: "pointer",
  fontFamily: mono,
  fontSize: 11,
  padding: "5px 9px",
};

function Numbered({ items }: { items: readonly string[] }) {
  return (
    <ol style={{ margin: "5px 0 0", paddingLeft: 20 }}>
      {items.map((item) => <li key={item} style={{ marginBottom: 3 }}>{item}</li>)}
    </ol>
  );
}

function Checks({ items }: { items: readonly string[] }) {
  return (
    <ul style={{ listStyle: "none", margin: "5px 0 0", padding: 0 }}>
      {items.map((item) => <li key={item} style={{ marginBottom: 3 }}>□ {item}</li>)}
    </ul>
  );
}

function SupportBadge({ support }: { support: Support }) {
  const { t } = useLanguage();
  const color = support === "full" ? "#7dd87d" : support === "partial" ? "#d8c97d" : "#e0a08f";
  const label =
    support === "full"
      ? t("full", "完整")
      : support === "partial"
        ? t("partial", "部分")
        : t("smoke-test only", "仅冒烟测试");
  return (
    <span style={{ color, border: `1px solid ${color}66`, borderRadius: 10, padding: "1px 7px", fontSize: 10, whiteSpace: "nowrap" }}>
      {label}
    </span>
  );
}

function QuickStart() {
  const { t } = useLanguage();
  return (
    <div>
      <h2 style={sectionTitle}>{t("Quick start", "快速开始")}</h2>
      <p style={{ marginTop: 0 }}>
        {t(
          "From the workspace root, run this command in a terminal. It is reference text, not a GUI action:",
          "在工作区根目录的终端中运行此命令。它是参考文本，不是 GUI 操作："
        )}
      </p>
      <div style={{ ...cardStyle, color: "#9fb4d8", userSelect: "text", marginBottom: 10 }}>./restart-lab.sh</div>
      <p>
        {t("It starts Duck Lab at", "它会启动 Duck Lab：")}{" "}
        <span style={{ color: "#7db8d8" }}>http://127.0.0.1:8788</span>{" "}
        {t("(frames on", "（帧数据位于")}{" "}
        <span style={{ color: "#7db8d8" }}>ws://127.0.0.1:8788/ws</span>
        {t(") and Duck Viewer at", "），并启动 Duck Viewer：")}{" "}
        <span style={{ color: "#7db8d8" }}>http://127.0.0.1:63317</span>.
      </p>
      <Numbered items={[
        t(
          "Wait for the Duck Lab header to show a green live connection and a non-empty roster.",
          "等待 Duck Lab 标题显示绿色实时连接和非空列表。"
        ),
        t(
          "Open 🧠 policies. Drag a policy onto a duck to assign it, or double-click a chip / use Spawn here to add a duck.",
          "打开 🧠 策略。将策略标签拖到小鸭上进行分配，或双击标签 / 使用此处的“生成”添加小鸭。"
        ),
        t(
          "Click the duck or its HUD row; confirm the amber floor ring and highlighted row.",
          "点击小鸭或其 HUD 行，确认琥珀色地面环和高亮行一致。"
        ),
        t(
          "Press R to synchronise/reset every duck to episode step zero.",
          "按 R 将所有小鸭同步/重置到回合第零步。"
        ),
        t(
          "Compare side by side for at least one full 30 s cycle.",
          "并排比较至少一个完整的 30 秒周期。"
        ),
        t(
          "Use 📷 shot or select a duck and use 🎥 record to capture evidence.",
          "使用 📷 截图，或选择一只小鸭并使用 🎥 录制来保存证据。"
        ),
      ]} />
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <strong style={{ color: "#d8c97d" }}>
          {t("Read claims narrowly.", "谨慎解读结论。")}
        </strong>{" "}
        {t(
          "A policy loading is a smoke test. A visible response without robotd’s command encoding or handoff is partial inspection. Only the walking runway is fully represented here.",
          "策略成功加载只是冒烟测试。缺少 robotd 指令编码或交接时，可见响应仅属于部分检查。这里只完整呈现行走跑道。"
        )}
      </div>
    </div>
  );
}

function Policies({ frame, online, clientRef }: { frame: Frame | null; online: boolean; clientRef: React.MutableRefObject<LabClient | null> }) {
  const { language, t } = useLanguage();
  const spawn = (name: string) => {
    clientRef.current?.sendSpawnDuck(`pollen:${name}`);
    pushToast(t(`＋ spawning ${name}`, `＋ 正在生成 ${name}`));
  };
  return (
    <div>
      <h2 style={sectionTitle}>{t("Shipped policies", "随附策略")}</h2>
      <p style={{ marginTop: 0 }}>
        {t("All nine vendored policies are ONNX", "九个随附策略均为 ONNX")}{" "}
        <strong style={{ color: "#dfe5ee" }}>
          {t("61 inputs → 14 actions", "61 个输入 → 14 个动作")}
        </strong>
        {t(
          ". Their names are stable robot roles copied from the upstream runtime, not training-run names.",
          "。它们的名称是从上游运行时复制的稳定机器人角色，而不是训练运行名称。"
        )}
      </p>
      <div style={{ display: "grid", gap: 10 }}>
        {SHIPPED_POLICY_GUIDES.map((policy) => {
          const count = countPolicyDucks(frame, policy.name);
          const copy =
            language === "zh"
              ? SHIPPED_POLICY_GUIDES_ZH[policy.name] ?? policy
              : policy;
          return (
            <article key={policy.name} style={cardStyle}>
              <div style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 7 }}>
                <strong style={{ color: "#dfe5ee", fontSize: 12 }}>{policy.name}</strong>
                <SupportBadge support={policy.support} />
                <span style={{ color: "#566072", fontSize: 10 }}>
                  {t(`${count} in scene`, `场景中 ${count} 只`)}
                </span>
                <span style={{ flex: 1 }} />
                <button
                  type="button"
                  disabled={!online}
                  title={
                    online
                      ? t(`spawn pollen:${policy.name}`, `生成 pollen:${policy.name}`)
                      : t("Duck Lab is offline", "Duck Lab 已离线")
                  }
                  onClick={() => spawn(policy.name)}
                  style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}
                >
                  {t("＋ Spawn", "＋ 生成")}
                </button>
              </div>
              <p style={{ color: "#c4cad4", margin: "7px 0 4px" }}>
                <strong>{t("Purpose:", "用途：")}</strong> {copy.purpose}
              </p>
              <p style={{ margin: "4px 0" }}>
                <strong style={{ color: "#aab3c0" }}>
                  {t("Input/runtime:", "输入/运行时：")}
                </strong>{" "}
                {copy.semantics}
              </p>
              <div style={{ color: "#aab3c0", marginTop: 7 }}>
                <strong>{t("In the Viewer", "在 Viewer 中")}</strong>
                <Numbered items={copy.steps} />
              </div>
              <p style={{ margin: "7px 0 4px" }}>
                <strong style={{ color: "#aab3c0" }}>{t("Expected:", "预期：")}</strong>{" "}
                {copy.expected}
              </p>
              <div>
                <strong style={{ color: "#aab3c0" }}>
                  {t("Verification checklist", "验证清单")}
                </strong>
                <Checks items={copy.verify} />
              </div>
              <p style={{ color: "#e0a08f", margin: "7px 0 0" }}>⚠ {copy.limitation}</p>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function Verify() {
  const { t } = useLanguage();
  return (
    <div>
      <h2 style={sectionTitle}>
        {t("Repeatable verification protocol", "可重复的验证流程")}
      </h2>
      <Numbered items={[
        t(
          "Spawn the candidate and a relevant reference side by side; use matching viewpoints and do not compare unsynchronised episodes.",
          "并排生成候选策略和相关参考策略；使用一致视角，不要比较未同步的回合。"
        ),
        t(
          "Select the candidate. Confirm its amber ring, highlighted HUD row, exact policy id/name, episode time, and fall count.",
          "选择候选策略。确认琥珀色环、高亮 HUD 行、准确策略 ID/名称、回合时间和跌倒次数。"
        ),
        t("Press R once to synchronise every duck at step zero.", "按一次 R，将所有小鸭同步到第零步。"),
        t(
          "Observe at least one full 30 s cycle. Note falls, resets, contacts, posture, direction, and recovery—not only a pleasing instant.",
          "观察至少一个完整的 30 秒周期。记录跌倒、重置、接触、姿态、方向和恢复，不要只看某个好看的瞬间。"
        ),
        t(
          "For locomotion only, compare HUD achieved / asked speed. Do not apply that speed test to a zero-command or phase-driven skill.",
          "仅对运动策略比较 HUD 的实际 / 请求速度。不要把该速度测试用于零指令或相位驱动技巧。"
        ),
        t(
          "Ignore walking reward r̄ for trick policies; the HUD deliberately dims it because it scores the walking recipe.",
          "对技巧策略忽略行走奖励 r̄；HUD 会刻意将其变暗，因为它评估的是行走配方。"
        ),
        t(
          "Capture a PNG for pose evidence or a video for timing/contact evidence. Keep the tab visible while recording.",
          "捕获 PNG 作为姿态证据，或录制视频作为时序/接触证据。录制时保持标签页可见。"
        ),
        t(
          "State the verdict as smoke test, partial inspection, or behavior validation, using each policy card’s support badge.",
          "根据每张策略卡的支持徽章，将结论表述为冒烟测试、部分检查或行为验证。"
        ),
      ]} />
      <div style={{ ...cardStyle, marginTop: 12 }}>
        <strong style={{ color: "#dfe5ee" }}>{t("Compatibility:", "兼容性：")}</strong>{" "}
        {t(
          "loading confirms the shared ONNX shape contract (61 observation values to 14 actions) and that the local interface can run the model. It does not prove the commands match the skill’s training distribution.",
          "加载成功确认共享 ONNX 形状契约（61 个观测值到 14 个动作），以及本地接口能够运行模型；它不能证明指令符合该技巧的训练分布。"
        )}
      </div>
      <div style={{ ...cardStyle, marginTop: 10 }}>
        <strong style={{ color: "#dfe5ee" }}>
          {t("Deterministic visual check:", "确定性视觉检查：")}
        </strong>{" "}
        {t(
          "the exported ONNX path is deterministic for the same observation sequence, but resets, simulation state, camera angle, command encoding, physics, and robotd handoffs still affect what is seen. Preserve the reset point and capture the whole relevant interval.",
          "对相同观测序列，导出的 ONNX 路径是确定性的，但重置、仿真状态、相机角度、指令编码、物理和 robotd 交接仍会影响所见结果。请保留重置点并捕获完整相关区间。"
        )}
      </div>
    </div>
  );
}

function Controls() {
  const { t } = useLanguage();
  return (
    <div>
      <h2 style={sectionTitle}>{t("Controls and panels", "控制与面板")}</h2>
      <div style={{ display: "grid", gap: 9 }}>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("Camera and stage", "相机与场景")}</strong>
          <p>
            {t(
              "Drag to orbit; wheel or two-finger vertical scroll zooms; two-finger horizontal swipe slides. A/D slide, W/S or ↑/↓ dolly, ←/→ orbit, Q/E rise/fall, Shift+R resets the view.",
              "拖动可环绕；滚轮或双指纵向滚动可缩放；双指横向滑动可平移。A/D 平移，W/S 或 ↑/↓ 推拉，←/→ 环绕，Q/E 升降，Shift+R 重置视角。"
            )}
          </p>
          <p style={{ color: "#d8c97d", marginBottom: 0 }}>
            {t(
              "WASD moves the camera—not the duck. Policies drive ducks; there is no keyboard teleop.",
              "WASD 控制相机，而不是小鸭。小鸭由策略驱动，不提供键盘遥控。"
            )}
          </p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("Selection and episode", "选择与回合")}</strong>
          <p>
            {t(
              "Click a duck or HUD row to select it (amber ring). Click empty floor or press Esc to deselect. Delete/Backspace removes the selected duck. R restarts every duck’s simulation together; Shift+R is camera reset.",
              "点击小鸭或 HUD 行进行选择（琥珀色环）。点击空地或按 Esc 取消选择。Delete/Backspace 移除所选小鸭。R 同时重启所有小鸭仿真；Shift+R 重置相机。"
            )}
          </p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("🧠 Policies", "🧠 策略")}</strong>
          <p>
            {t(
              "Drag a chip onto a duck to assign it. Drag to empty floor, double-click the chip, or arm it then click empty floor to spawn. Click a chip to arm, then click a duck to assign. Drop a run onto 🎓 teach to load it. Press / to open/focus policy search; Esc cancels an armed or dragged chip.",
              "将标签拖到小鸭上进行分配。拖到空地、双击标签，或先激活再点击空地即可生成。点击标签激活后，再点击小鸭即可分配。将运行拖到 🎓 教学面板可加载。按 / 打开或聚焦策略搜索；Esc 取消已激活或正在拖动的标签。"
            )}
          </p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("🎓 Teach", "🎓 教学")}</strong>
          <p>
            {t(
              "Describe a trick, inspect its reward recipe and live score, add helpers with ＋, then adjust unlocked term/stage weights after a run. Retrain starts fresh; fine-tune continues from the selected finished run. Reward charts are training evidence, not visual proof.",
              "描述一个技巧，检查奖励配方和实时分数，使用 ＋ 添加辅助小鸭，然后在运行后调整已解锁的项/阶段权重。重新训练会从头开始；微调会从选中的已完成运行继续。奖励图表是训练证据，不是视觉证明。"
            )}
          </p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("🎬 Animate", "🎬 动画")}</strong>
          <p>
            {t(
              "Choose 🦴 joint or 🎮 rig mode, use sliders or drag duck parts/⇕ handle, add and retime key poses, scrub/play the timeline, save a clip, then train a policy to track it. Shift gives fine dragging.",
              "选择 🦴 关节或 🎮 绑定模式，使用滑块或拖动小鸭部件/⇕ 手柄，添加并调整关键姿态时间，拖动/播放时间轴，保存片段，再训练策略跟踪它。按住 Shift 可精细拖动。"
            )}
          </p>
        </section>
        <section style={cardStyle}>
          <strong style={{ color: "#dfe5ee" }}>{t("📷 / 🎥 Capture", "📷 / 🎥 捕获")}</strong>
          <p>
            {t(
              "📷 shot downloads the current view as PNG. Select a duck to reveal 🎥 record; the camera frames it and the lab produces MP4 + GIF downloads. Takes stop at 60 s; keep the tab visible so rendered frames are recorded.",
              "📷 截图会将当前视图下载为 PNG。选择小鸭后可使用 🎥 录制；相机会为其取景，实验室会生成可下载的 MP4 + GIF。录制在 60 秒时停止；请保持标签页可见以记录渲染帧。"
            )}
          </p>
        </section>
      </div>
    </div>
  );
}

export function GuidePanel({ clientRef, connected, onClose }: { clientRef: React.MutableRefObject<LabClient | null>; connected: boolean; onClose: () => void }) {
  const { t } = useLanguage();
  const [tab, setTab] = useState<GuideTab>("Quick start");
  const [frame, setFrame] = useState<Frame | null>(() => clientRef.current?.frame ?? null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeRef.current?.focus();
    return () => previous?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const id = window.setInterval(() => setFrame(clientRef.current?.frame ?? null), 500);
    return () => window.clearInterval(id);
  }, [clientRef]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;
      const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      );
      if (!focusable?.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [onClose]);

  const online = connected && !!frame;
  const reset = () => {
    clientRef.current?.sendReset();
    pushToast(t("↺ sim restarted — every duck from zero", "↺ 仿真已重启，所有小鸭从零开始"));
  };
  const zero = () => {
    clientRef.current?.sendCmd([0, 0, 0]);
    pushToast(
      t(
        "zero command shared by all ducks for 6 seconds",
        "所有小鸭共享零指令 6 秒"
      )
    );
  };
  const tabLabel = (item: GuideTab) => {
    if (item === "Quick start") return t("Quick start", "快速开始");
    if (item === "Policies") return t("Policies", "策略");
    if (item === "Verify") return t("Verify", "验证");
    return t("Controls", "控制");
  };

  return createPortal(
    <div
      data-policy-ui
      data-modal
      role="dialog"
      aria-modal="true"
      aria-labelledby="duck-guide-title"
      onClick={onClose}
      style={{ position: "fixed", inset: 0, zIndex: 1100, background: "rgba(0,0,0,0.62)", display: "flex", alignItems: "center", justifyContent: "center", padding: 12, boxSizing: "border-box", backdropFilter: "blur(2px)" }}
    >
      <div
        ref={dialogRef}
        onClick={(event) => event.stopPropagation()}
        style={{ width: 820, maxWidth: "calc(100vw - 24px)", maxHeight: "calc(100dvh - 24px)", background: "rgba(14,16,20,0.98)", border: "1px solid rgba(255,255,255,0.14)", borderRadius: 11, color: "#aab3c0", fontFamily: mono, fontSize: 11, lineHeight: 1.55, boxShadow: "0 16px 50px rgba(0,0,0,0.7)", display: "flex", flexDirection: "column", overflow: "hidden" }}
      >
        <header style={{ padding: "11px 13px 9px", borderBottom: "1px solid rgba(255,255,255,0.09)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <h1 id="duck-guide-title" style={{ margin: 0, color: "#e8e6e1", fontSize: 14 }}>
              {t("? Duck Viewer guide", "? Duck Viewer 指南")}
            </h1>
            <span style={{ flex: 1 }} />
            <span aria-live="polite" style={{ color: online ? "#7dd87d" : "#e07a5f", whiteSpace: "nowrap" }}>
              {online
                ? t(
                    `● live · ${frame?.ducks.length ?? 0} ducks · ${frame?.mode ?? "auto"}`,
                    `● 实时 · ${frame?.ducks.length ?? 0} 只小鸭 · ${frame?.mode ?? "auto"}`
                  )
                : `○ ${
                    connected
                      ? t("waiting for frames", "等待帧")
                      : t("offline", "离线")
                  }`}
            </span>
            <button
              ref={closeRef}
              type="button"
              onClick={onClose}
              aria-label={t("Close Duck Viewer guide", "关闭 Duck Viewer 指南")}
              title={t("close guide (Esc)", "关闭指南 (Esc)")}
              style={{ ...actionStyle, padding: "2px 7px", color: "#aab3c0" }}
            >
              ✕
            </button>
          </div>
          <div
            style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 9 }}
            role="tablist"
            aria-label={t("Guide sections", "指南章节")}
          >
            {GUIDE_TABS.map((item) => (
              <button
                key={item}
                type="button"
                role="tab"
                aria-selected={tab === item}
                onClick={() => setTab(item)}
                style={{ ...actionStyle, background: tab === item ? "#2a3548" : "#161b26", color: tab === item ? "#cfe4f5" : "#8b93a3", borderColor: tab === item ? "#7db8d8" : "rgba(255,255,255,0.10)" }}
              >
                {tabLabel(item)}
              </button>
            ))}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 7, marginTop: 9 }}>
            <button type="button" disabled={!online} onClick={reset} style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}>
              {t("↺ Restart all ducks", "↺ 重启所有小鸭")}
            </button>
            <button type="button" disabled={!online} onClick={zero} style={{ ...actionStyle, opacity: online ? 1 : 0.42, cursor: online ? "pointer" : "default" }}>
              {t("0 Zero command for 6 seconds", "0 零指令 6 秒")}
            </button>
            <span style={{ color: "#566072", alignSelf: "center" }}>
              {t(
                "Command override is shared/global and temporary; auto resumes after 6 s.",
                "指令覆盖是共享、全局且临时的；6 秒后自动恢复。"
              )}
            </span>
          </div>
        </header>
        <main role="tabpanel" tabIndex={0} style={{ padding: "13px", overflowY: "auto", overscrollBehavior: "contain" }}>
          {tab === "Quick start" && <QuickStart />}
          {tab === "Policies" && <Policies frame={frame} online={online} clientRef={clientRef} />}
          {tab === "Verify" && <Verify />}
          {tab === "Controls" && <Controls />}
        </main>
      </div>
    </div>,
    document.body
  );
}
