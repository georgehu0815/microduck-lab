"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { ARM_CASES, type ArmCaseId } from "@/lib/arm-contract";
import type { ArmVideo, ArmVideoLibrary as VideoLibrary } from "@/lib/arm-videos";
import { armCaseTitle } from "@/lib/arm-labels";
import { useLanguage } from "./LanguageProvider";
import type { Language } from "@/lib/language";
import { IS_STATIC_EXPORT, publicAssetUrl } from "@/lib/static-assets";
import {
  compareArmVideosNewestFirst,
  missingCurrentSuccessCaseIds,
  selectLatestCurrentSuccesses,
} from "@/lib/arm-video-selection";
import styles from "./ArmVideoLibrary.module.css";

function durationLabel(seconds: number | null, unknownLabel: string) {
  if (seconds === null || !Number.isFinite(seconds)) return unknownLabel;
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
}

function recordedAtLabel(value: string | null, language: Language, unknownLabel: string) {
  if (!value || !Number.isFinite(Date.parse(value))) return unknownLabel;
  return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en-US", {
    dateStyle: "medium",
    timeStyle: "medium",
    timeZone: "UTC",
  }).format(new Date(value));
}

function caseTitle(video: ArmVideo, language: Language) {
  return video.caseIds.map((id) => armCaseTitle(id, language)).join(" / ");
}

function videoTitle(video: ArmVideo, language: Language) {
  let title = video.title;
  for (const entry of ARM_CASES) title = title.replace(entry.title, armCaseTitle(entry.id, language));
  if (language === "zh") title = title.replace(/\btrain (\d+)/g, "训练 $1").replace(/\beval (\d+)/g, "评估 $1");
  return title;
}

export default function ArmVideoLibrary({ caseFilter, onCaseFilterChange }: {
  caseFilter: ArmCaseId | "all";
  onCaseFilterChange: (value: ArmCaseId | "all") => void;
}) {
  const { language, t } = useLanguage();
  const provenanceLabels = { current: t("Current source", "当前来源"), historical: t("Historical source", "历史来源"), unverified: t("Unverified source", "来源未核实") };
  const outcomeLabels = { passed: t("PASS", "通过"), failed: t("FAIL", "失败"), mixed: t("MIXED", "混合结果"), unverified: t("UNVERIFIED", "未核实") };
  const [library, setLibrary] = useState<VideoLibrary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<{ reason: "http" | "invalid" | "network"; status?: number } | null>(null);
  const [mediaError, setMediaError] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [view, setView] = useState<"latest" | "history">("latest");
  const [query, setQuery] = useState("");
  const [provenance, setProvenance] = useState<ArmVideo["provenance"] | "all">("all");
  const [outcome, setOutcome] = useState<ArmVideo["outcome"] | "all">("all");
  const [kind, setKind] = useState<ArmVideo["kind"] | "all">("all");

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const catalogUrl = IS_STATIC_EXPORT
        ? publicAssetUrl("/static/arm-videos/catalog.json")
        : "/api/arm/videos";
      const response = await fetch(catalogUrl, { cache: "no-store", signal });
      if (!response.ok) { setError({ reason: "http", status: response.status }); return; }
      const payload = await response.json() as VideoLibrary;
      if (!Array.isArray(payload.videos) || payload.hardwareEnabled !== false) { setError({ reason: "invalid" }); return; }
      setLibrary({
        ...payload,
        videos: payload.videos.map((video) => ({
          ...video,
          videoHash: video.videoHash ?? "",
          evidenceRecordedAt: video.evidenceRecordedAt ?? null,
          receiptRecordedAt: video.receiptRecordedAt ?? null,
          videoUrl: publicAssetUrl(video.videoUrl),
          evidenceUrl: video.evidenceUrl ? publicAssetUrl(video.evidenceUrl) : null,
        })),
      });
    } catch {
      if (signal?.aborted) return;
      setError({ reason: "network" });
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => void load(controller.signal), 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [load]);

  const latestVideos = useMemo(
    () => selectLatestCurrentSuccesses(library?.videos ?? []),
    [library]
  );
  const missingLatestCases = useMemo(
    () => missingCurrentSuccessCaseIds(library?.videos ?? []),
    [library]
  );
  const visibleVideos = useMemo(() => {
    const viewVideos = view === "latest"
      ? latestVideos
      : [...(library?.videos ?? [])].sort(compareArmVideosNewestFirst);
    return viewVideos.filter((video) => {
    const text = [video.title, caseTitle(video, "en"), caseTitle(video, "zh"), video.batch, video.path, video.trainingSeed, video.evaluationSeed].join(" ").toLowerCase();
    return (caseFilter === "all" || video.caseIds.includes(caseFilter)) &&
      (provenance === "all" || video.provenance === provenance) &&
      (outcome === "all" || video.outcome === outcome) &&
      (kind === "all" || video.kind === kind) && text.includes(query.trim().toLowerCase());
    });
  }, [library, latestVideos, view, caseFilter, provenance, outcome, kind, query]);
  const selected = visibleVideos.find((video) => video.id === selectedId) ?? visibleVideos[0] ?? null;
  const selectedIndex = selected ? visibleVideos.indexOf(selected) : -1;
  const failedCount = library?.videos.filter((video) => video.outcome === "failed").length ?? 0;

  function selectVideo(video: ArmVideo) {
    setSelectedId(video.id);
    setMediaError(false);
  }

  function clearFilters() {
    onCaseFilterChange("all");
    setQuery("");
    setProvenance("all");
    setOutcome("all");
    setKind("all");
    setMediaError(false);
  }

  function selectView(nextView: "latest" | "history") {
    setView(nextView);
    setSelectedId(null);
    setMediaError(false);
  }

  return (
    <section className={styles.library} aria-labelledby="video-library-heading" data-testid="arm-video-library">
      <div className={styles.heading}>
        <div>
          <p className={styles.eyebrow}>{t("RECORDED EXPERIMENTS · LOCAL EVIDENCE", "实验录像 · 本地证据")}</p>
          <h2 id="video-library-heading">{t("Arm video library", "机械臂视频证据库")}</h2>
          <p>{t("All six cases, regression replays, historical failures, and compilations. Watch every saved video without starting the live simulator.", "覆盖六个案例、修复回归、历史失败与合集。无需启动实时仿真，即可回看全部已保存视频。")}</p>
        </div>
        <button className={styles.button} onClick={() => void load()} disabled={loading}>↻ {t("Refresh videos", "刷新视频")}</button>
      </div>

      <div className={styles.stats} aria-label={t("Video inventory", "视频目录统计")}>
        <div><strong data-testid="arm-video-current-success-count">{library ? `${latestVideos.length} / ${ARM_CASES.length}` : "—"}</strong><span>{t("Cases with a current-source success", "具有当前来源成功录像的案例")}</span></div>
        <div><strong>{library ? library.uniqueVideos : "—"}</strong><span>{t("All recordings retained", "保留的全部录像")}</span></div>
        <div><strong>{library ? failedCount : "—"}</strong><span>{t("Failed videos retained", "失败视频保留")}</span></div>
        <div><strong>{library ? library.totalFiles : "—"}</strong><span>{t("Source files (including identical copies)", "原始文件（包含重复副本）")}</span></div>
      </div>

      <div className={styles.viewSelector} aria-label={t("Recording view", "录像视图")}>
        <button type="button" data-testid="arm-video-view-latest" className={view === "latest" ? styles.activeView : styles.viewButton} aria-pressed={view === "latest"} onClick={() => selectView("latest")}>
          {t("Latest current successes", "最新当前来源成功录像")} <span>{latestVideos.length} / {ARM_CASES.length}</span>
        </button>
        <button type="button" data-testid="arm-video-view-history" className={view === "history" ? styles.activeView : styles.viewButton} aria-pressed={view === "history"} onClick={() => selectView("history")}>
          {t("All recordings (history)", "全部录像（历史）")} <span>{library?.uniqueVideos ?? 0}</span>
        </button>
      </div>
      {view === "latest" && missingLatestCases.length > 0 && (
        <div className={styles.latestNotice} data-testid="arm-video-missing-current-successes">
          <strong>{t("No current-source success is available for:", "以下案例没有当前来源成功录像：")}</strong>{" "}
          {missingLatestCases.map((caseId) => armCaseTitle(caseId, language)).join(" · ")}
          <p>{t("Historical or unverified recordings are retained in history and are not substituted as the latest success.", "历史或未核实录像仍保留在历史视图中，但不会替代最新成功录像。")}</p>
        </div>
      )}

      <div className={styles.filters}>
        <label>{t("Search videos", "搜索视频")}<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={t("Case, seed, batch…", "案例、种子、运行批次…")} /></label>
        <label>{t("Case", "案例")}<select aria-label={t("Case", "案例")} value={caseFilter} onChange={(event) => onCaseFilterChange(event.target.value as ArmCaseId | "all")}>
          <option value="all">{t("All six cases", "全部六个案例")}</option>{ARM_CASES.map((entry) => <option key={entry.id} value={entry.id}>{armCaseTitle(entry.id, language)}</option>)}
        </select></label>
        <label>{t("Provenance", "来源")}<select aria-label={t("Provenance", "来源")} value={provenance} onChange={(event) => setProvenance(event.target.value as typeof provenance)}>
          <option value="all">{t("All sources (including historical)", "全部来源（含历史）")}</option>{Object.entries(provenanceLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
        <label>{t("Outcome", "结果")}<select aria-label={t("Outcome", "结果")} value={outcome} onChange={(event) => setOutcome(event.target.value as typeof outcome)}>
          <option value="all">{t("All outcomes (including failures)", "全部结果（含失败）")}</option>{Object.entries(outcomeLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select></label>
        <label>{t("Type", "类型")}<select aria-label={t("Type", "类型")} value={kind} onChange={(event) => setKind(event.target.value as typeof kind)}>
          <option value="all">{t("Episodes and compilations", "单回合与合集")}</option><option value="episode">{t("Episode", "单回合")}</option><option value="compilation">{t("Compilation", "合集")}</option>
        </select></label>
      </div>

      <div className={styles.resultsLine}>
        <span role="status">{loading ? t("Loading local videos…", "正在读取本地视频…") : view === "latest" ? t(`${visibleVideos.length} latest current successes shown`, `显示 ${visibleVideos.length} 个最新当前来源成功录像`) : t(`${visibleVideos.length} / ${library?.uniqueVideos ?? 0} recordings`, `${visibleVideos.length} / ${library?.uniqueVideos ?? 0} 个录像`)}</span>
        <button className={styles.textButton} onClick={clearFilters}>{t("Clear filters", "清除筛选")}</button>
      </div>
      {error && <div className={styles.error} role="alert">{error.reason === "http" ? t(`Video catalog request failed (${error.status}).`, `视频目录加载失败 (${error.status})。`) : error.reason === "invalid" ? t("Invalid video catalog.", "视频目录格式无效。") : t("Unable to read the local video catalog.", "无法读取本地视频目录。")} {library ? t("The previous catalog is retained below. Refresh to retry.", "下面保留上次加载的目录，刷新后重试。") : t("No demo data is used.", "没有使用演示数据。")}</div>}
      {library && library.warnings.length > 0 && <details className={styles.notice}><summary>{t("Catalog warnings", "目录读取提示")} ({library.warnings.length})</summary><ul>{library.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}

      {selected ? (
        <div className={styles.workspace}>
          <article className={styles.playerPanel}>
            <div className={styles.playerHeading}>
              <div><p className={styles.eyebrow}>{selected.kind === "compilation" ? t("COMPILATION · NOT A NEW INDEPENDENT EXPERIMENT", "合集 · 非独立新实验") : view === "latest" ? t("LATEST CURRENT-SOURCE SUCCESS", "最新当前来源成功录像") : t("RECORDED ROLLOUT", "回合录像")}</p><h3>{videoTitle(selected, language)}</h3>
                <p className={styles.selectionContext} data-testid="arm-video-current-selection">{selected.batch} · {t("Evidence", "证据")} <time dateTime={selected.evidenceRecordedAt ?? selected.receiptRecordedAt ?? undefined} title={selected.evidenceRecordedAt ?? selected.receiptRecordedAt ?? undefined}>{recordedAtLabel(selected.evidenceRecordedAt ?? selected.receiptRecordedAt, language, t("date unavailable", "日期不可用"))}</time></p>
              </div>
              <span className={`${styles.badge} ${styles[selected.outcome]}`}>{outcomeLabels[selected.outcome]}</span>
            </div>
            <video key={selected.id} className={styles.player} controls playsInline preload="metadata"
              src={selected.videoUrl} aria-label={t(`Arm evidence video: ${videoTitle(selected, language)}`, `机械臂证据视频：${videoTitle(selected, language)}`)}
              onLoadStart={() => setMediaError(false)} onError={() => setMediaError(true)}>{t("Your browser cannot play this video. Download the MP4 instead.", "浏览器不支持视频播放，请下载 MP4。")}</video>
            {mediaError && <p className={styles.error} role="alert">{t("Unable to play this video. Download the original MP4 or refresh the catalog to check whether the file still exists.", "视频无法播放。请尝试下载原始 MP4，或刷新目录检查文件是否仍存在。")}</p>}
            <div className={styles.transport}>
              <button className={styles.button} disabled={selectedIndex <= 0} onClick={() => selectVideo(visibleVideos[selectedIndex - 1])}>← {t("Previous", "上一个")}</button>
              <span>{selectedIndex + 1} / {visibleVideos.length}</span>
              <button className={styles.button} disabled={selectedIndex >= visibleVideos.length - 1} onClick={() => selectVideo(visibleVideos[selectedIndex + 1])}>{t("Next", "下一个")} →</button>
            </div>
            <dl className={styles.metadata}>
              <div><dt>{t("Case", "案例")}</dt><dd>{caseTitle(selected, language) || t("Case not specified", "案例未标注")}</dd></div>
              <div><dt>{t("Provenance", "来源")}</dt><dd>{provenanceLabels[selected.provenance]}</dd></div>
              <div><dt>{t("Training / evaluation seed", "训练 / 评估种子")}</dt><dd>{selected.trainingSeed ?? "—"} / {selected.evaluationSeed ?? "—"}</dd></div>
              <div><dt>{t("Duration", "时长")}</dt><dd>{durationLabel(selected.durationSeconds, t("Duration not reported", "时长未报告"))}</dd></div>
              <div><dt>{t("Evidence recorded", "证据记录时间")}</dt><dd>{recordedAtLabel(selected.evidenceRecordedAt, language, t("Not reported", "未报告"))}</dd></div>
              <div><dt>{t("Video receipt recorded", "视频凭据时间")}</dt><dd>{recordedAtLabel(selected.receiptRecordedAt, language, t("Not reported", "未报告"))}</dd></div>
              <div className={styles.wide}><dt>{t("Video SHA-256", "视频 SHA-256")}</dt><dd><code>{selected.videoHash || t("Not reported", "未报告")}</code></dd></div>
              <div className={styles.wide}><dt>{t("Batch", "批次")}</dt><dd>{selected.batch}</dd></div>
            </dl>
            {selected.failedGates.length > 0 && <div className={styles.error}><strong>{t("Failed gates", "未通过门槛")}</strong><p>{selected.failedGates.join(" · ")}</p></div>}
            <div className={styles.links}>
              <a className={styles.button} href={selected.videoUrl} download>{t("Download original MP4", "下载原始 MP4")} ↓</a>
              {selected.evidenceUrl && <a className={styles.button} href={selected.evidenceUrl} target="_blank" rel="noreferrer">{t("View source receipt", "查看来源凭据")} ↗</a>}
            </div>
            <details className={styles.paths}><summary>{t("Source files and duplicate copies", "文件来源与重复副本")} ({selected.aliases.length})</summary><code>{selected.path}</code>{selected.aliases.filter((alias) => alias !== selected.path).map((alias) => <code key={alias}>{alias}</code>)}</details>
          </article>

          <aside className={styles.playlist} aria-label={t("Arm video playlist", "机械臂视频播放列表")} data-testid="arm-video-playlist">
            <div className={styles.playlistHeading}><strong>{view === "latest" ? t("Latest success by case", "各案例最新成功录像") : t("All recordings (history)", "全部录像（历史）")}</strong><span>{visibleVideos.length} {t("videos", "个视频")}</span></div>
            <ol>{visibleVideos.map((video, index) => <li key={video.id}>
              <button className={selected.id === video.id ? styles.selectedCard : styles.card} aria-pressed={selected.id === video.id}
                data-video-id={video.id}
                data-arm-case-id={video.caseIds[0] ?? ""}
                aria-label={`${t("Watch", "观看")} ${videoTitle(video, language)} ${video.evaluationSeed ?? t("compilation", "合集")} ${video.batch}`} onClick={() => selectVideo(video)}>
                <span className={styles.number}>{String(index + 1).padStart(2, "0")}</span>
                <span className={styles.cardContent}><strong>{videoTitle(video, language)}</strong><small>{provenanceLabels[video.provenance]} · {video.kind === "compilation" ? t("Compilation", "合集") : `${t("seed", "种子")} ${video.evaluationSeed ?? t("not reported", "未报告")}`}</small>
                  <span className={styles.cardBadges}><span className={`${styles.badge} ${styles[video.outcome]}`}>{outcomeLabels[video.outcome]}</span><small>{durationLabel(video.durationSeconds, t("Duration not reported", "时长未报告"))}</small></span></span>
              </button>
            </li>)}</ol>
          </aside>
        </div>
      ) : !loading && <div className={styles.empty}><strong>{view === "latest" && library?.uniqueVideos ? t("No matching current-source success", "没有匹配的当前来源成功录像") : library?.uniqueVideos ? t("No matching recordings", "没有匹配的录像") : t("No arm videos available", "暂无可读取的机械臂视频")}</strong><p>{view === "latest" && library?.uniqueVideos ? t("Adjust the filters or open All recordings (history). Historical evidence is not substituted here.", "调整筛选条件，或打开“全部录像（历史）”。此处不会用历史证据替代最新成功录像。") : library?.uniqueVideos ? t("Adjust the filters to inspect retained current, historical, failed, and unverified evidence.", "调整筛选条件，查看保留的当前、历史、失败和未核实证据。") : t("Videos are read from rlx/runs/arm. Generated MP4s appear automatically; no results are fabricated.", "视频从 rlx/runs/arm 读取；生成的 MP4 会自动进入目录，不会显示虚构结果。")}</p></div>}
      <p className={styles.caveat}>{t("Simulation evidence, not physical hardware certification. PPO videos use authored IK/FSM + bounded PPO residual, not end-to-end learning. A passing video does not establish a passing full validation matrix; historical and unverified files are not current success evidence.", "仿真证据，不是实体硬件认证。PPO 视频采用人工编写的 IK/FSM + 有界 PPO 残差，不是端到端学习。单个视频通过不等于完整验证矩阵通过；历史和未核实文件不计作当前成功证据。")}</p>
    </section>
  );
}
