"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  WINGPOD_DOCUMENTS,
  WINGPOD_DOWNLOADS,
  WINGPOD_SCOPE_BADGES,
  WINGPOD_VIDEOS,
  createWingPodMediaModel,
  filterWingPodCases,
  parseWingPodCasesManifest,
  wingPodAssetUrl,
  wingPodCaseAssetUrl,
  type WingPodCase,
  type WingPodCaseFilter,
  type WingPodCaseKind,
  type WingPodCasesManifest,
  type WingPodVideoId,
} from "@/lib/wingpod";
import { publicAssetUrl } from "@/lib/static-assets";
import { LanguageToggle, useLanguage } from "./LanguageProvider";
import WingPodLive from "./WingPodLive";
import styles from "./WingPodPage.module.css";

function ArrowIcon() {
  return <span aria-hidden="true">↗</span>;
}

function DownloadIcon() {
  return <span aria-hidden="true">↓</span>;
}

export default function WingPodPage() {
  const { language, t } = useLanguage();
  const [selectedVideo, setSelectedVideo] = useState<WingPodVideoId>("tennis");
  const [casesManifest, setCasesManifest] = useState<WingPodCasesManifest | null>(null);
  const [casesError, setCasesError] = useState(false);
  const [caseVariant, setCaseVariant] = useState<WingPodCaseFilter>("all");
  const [caseKind, setCaseKind] = useState<WingPodCaseKind | "all">("all");
  const [selectedCase, setSelectedCase] = useState<WingPodCase | null>(null);
  const media = createWingPodMediaModel(selectedVideo);
  const shownCases = useMemo(
    () => filterWingPodCases(casesManifest?.cases ?? [], caseVariant, caseKind),
    [caseKind, caseVariant, casesManifest],
  );
  const activeVideo = selectedCase
    ? {
        src: wingPodCaseAssetUrl(selectedCase.video),
        poster: wingPodAssetUrl("sequence-contact-sheet.jpg"),
        downloadName: selectedCase.video.split("/").at(-1) ?? `${selectedCase.id}.mp4`,
        title: selectedCase.id,
        summary: selectedCase.kind === "positive"
          ? t(
              `Historical original-graphite palette · positive ${selectedCase.variant} case · seed ${selectedCase.seed}`,
              `历史原版石墨灰相机配色 · 正向 ${selectedCase.variant} 案例 · 种子 ${selectedCase.seed}`,
            )
          : t(
              `Historical original-graphite palette · negative control · expected outcome ${selectedCase.expectedOutcomeMatched ? "matched" : "did not match"}`,
              `历史原版石墨灰相机配色 · 负向对照 · 预期结果${selectedCase.expectedOutcomeMatched ? "匹配" : "未匹配"}`,
            ),
      }
    : {
        src: media.videoSrc,
        poster: media.posterSrc,
        downloadName: media.downloadName,
        title: media.selected.title[language],
        summary: media.selected.summary[language],
      };

  useEffect(() => {
    const controller = new AbortController();
    fetch(wingPodAssetUrl("cases.json"), { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`WingPod cases request failed: ${response.status}`);
        return response.json();
      })
      .then((value) => {
        setCasesManifest(parseWingPodCasesManifest(value));
        setCasesError(false);
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setCasesError(true);
      });
    return () => controller.abort();
  }, []);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <Link className={styles.brand} href="/" aria-label={t("Back to Microduck Studio", "返回 Microduck Studio")}>
          <span className={styles.brandMark} aria-hidden="true">W</span>
          <span>WingPod <small>CAMERA V2</small></span>
        </Link>
        <nav className={styles.primaryNav} aria-label={t("WingPod navigation", "WingPod 导航")}>
          <Link href="/">{t("Studio", "Studio")}</Link>
          <Link href="/arm">{t("Arm Lab", "机械臂实验室")}</Link>
          <Link href="/classroom">{t("Classroom PPT", "课堂课件")}</Link>
          <a href="#live">{t("Live", "实时")}</a>
          <a href="#videos">{t("Videos", "视频")}</a>
          <a href="#cases">{t("Cases", "案例")}</a>
          <a href="#downloads">{t("Downloads", "下载")}</a>
        </nav>
        <Link className={styles.mobileClassroom} href="/classroom">{t("Classroom", "课堂课件")}</Link>
        <LanguageToggle />
      </header>

      <main>
        <section className={styles.hero} aria-labelledby="wingpod-title">
          <Image
            className={styles.heroImage}
            src={wingPodAssetUrl("hero.png")}
            alt={t(
              "Soft WingPod Camera v2 with a cream face, sage gray lens rings, navy centers, peach cheeks, and honey beak",
              "柔和版 WingPod Camera v2：奶油白面部、鼠尾草灰镜圈、海军蓝镜心、桃色脸颊与蜂蜜色喙",
            )}
            fill
            priority
            sizes="100vw"
          />
          <div className={styles.heroShade} aria-hidden="true" />
          <div className={styles.heroContent}>
            <p className={styles.kicker}>{t("MICRODUCK ARM PERCEPTION CONCEPT", "MICRODUCK 机械臂感知概念")}</p>
            <h1 id="wingpod-title">WingPod Camera v2</h1>
            <p>
              {t(
                "A softer cream, sage, navy, peach, and honey camera-face revision for reviewing the same tennis-return simulation, with evidence boundaries kept explicit.",
                "以奶油白、鼠尾草灰、海军蓝、桃色与蜂蜜色组成更柔和的相机面部改版，用于检查同一捡球归桶仿真，并明确保留证据边界。",
              )}
            </p>
            <div className={styles.heroActions}>
              <a className={styles.primaryAction} href="#videos">
                <span aria-hidden="true">▶</span>{t("Review footage", "查看视频")}
              </a>
              <a className={styles.secondaryAction} href={wingPodAssetUrl("CAMERA-V2.md")} download>
                <DownloadIcon />{t("Camera notes", "相机说明")}
              </a>
            </div>
          </div>
        </section>

        <section className={styles.scopeBand} aria-label={t("Evidence scope", "证据范围")}>
          {WINGPOD_SCOPE_BADGES.map((badge) => (
            <span key={badge.id} data-scope={badge.id}>
              <i aria-hidden="true" />
              {badge.label[language]}
            </span>
          ))}
        </section>

        <WingPodLive />

        <section className={styles.metrics} aria-label={t("WingPod key metrics", "WingPod 关键指标")}>
          <div><strong>15</strong><span>{t("servos total", "舵机总数")}</span><small>{t("10 legs + 5 arm", "10 腿部 + 5 机械臂")}</small></div>
          <div><strong>28 × 17 × 5 mm</strong><span>{t("proposed eye housing", "拟议眼部外壳")}</span><small>{t("compact envelope", "紧凑包络尺寸")}</small></div>
          <div><strong>13 mm</strong><span>{t("lens separation", "镜头间距")}</span><small>{t("proposed center-to-center", "拟议中心距")}</small></div>
          <div><strong>{t("Unknown", "未知")}</strong><span>{t("camera SKU", "相机 SKU")}</span><small>{t("selection remains open", "型号仍待确定")}</small></div>
        </section>

        <section className={styles.detailSection}>
          <div className={styles.sectionCopy}>
            <p className={styles.kicker}>{t("PROPOSED INDUSTRIAL DESIGN", "拟议工业设计")}</p>
            <h2>{t("A softer camera face, without overstating autonomy", "更柔和的相机面部，但不夸大自主能力")}</h2>
            <p>
              {t(
                "The cream face adds sage gray lens rings, navy centers, peach cheeks, and a honey beak around the same proposed compact camera volume. This remains an appearance and integration study, not a released enclosure or confirmed optical stack.",
                "奶油白面部在同一拟议紧凑相机体积周围加入鼠尾草灰镜圈、海军蓝镜心、桃色脸颊与蜂蜜色喙。它仍是外观与集成研究，并非已发布外壳或已确认的光学方案。",
              )}
            </p>
            <dl className={styles.specList}>
              <div><dt>{t("Housing", "外壳")}</dt><dd>28 × 17 × 5 mm</dd></div>
              <div><dt>{t("Lens spacing", "镜头间距")}</dt><dd>13 mm</dd></div>
              <div><dt>{t("Camera module", "相机模组")}</dt><dd>{t("SKU unknown", "SKU 未知")}</dd></div>
              <div><dt>{t("Control role", "控制作用")}</dt><dd>{t("None in this replay", "本回放中无控制作用")}</dd></div>
            </dl>
          </div>
          <figure className={styles.detailFigure}>
            <Image
              src={wingPodAssetUrl("camera-eyes-detail.png")}
              alt={t("Close detail of the soft WingPod camera face and paired eye housing", "柔和版 WingPod 相机面部与双眼外壳细节")}
              width={1200}
              height={900}
              sizes="(max-width: 900px) 100vw, 52vw"
            />
            <figcaption>
              {t(
                "Soft appearance revision with documented reference-match and physics checks. Dimensions remain proposed; camera SKU and final mounting stack are unresolved.",
                "柔和外观改版已记录参考外观匹配与物理检查。尺寸仍为拟议值；相机 SKU 与最终安装结构尚未确定。",
              )}
            </figcaption>
          </figure>
        </section>

        <section className={styles.videoSection} id="videos" aria-labelledby="videos-title">
          <div className={styles.sectionHeading}>
            <div>
              <p className={styles.kicker}>{t("RECORDED EVIDENCE", "已录制证据")}</p>
              <h2 id="videos-title">{t("Choose what to inspect", "选择检查内容")}</h2>
            </div>
            <a className={styles.downloadCurrent} href={activeVideo.src} download={activeVideo.downloadName}>
              <DownloadIcon />{t("Download selected MP4", "下载所选 MP4")}
            </a>
          </div>

          <div className={styles.videoChooser} role="group" aria-label={t("Choose a WingPod video", "选择 WingPod 视频")}>
            {WINGPOD_VIDEOS.map((video) => (
              <button
                key={video.id}
                type="button"
                aria-pressed={!selectedCase && selectedVideo === video.id}
                onClick={() => {
                  setSelectedCase(null);
                  setSelectedVideo(video.id);
                }}
              >
                <span aria-hidden="true">{video.id === "appearance" ? "01" : "02"}</span>
                <strong>{video.title[language]}</strong>
                <small>{video.durationSeconds.toFixed(video.durationSeconds % 1 ? 2 : 0)}s</small>
              </button>
            ))}
          </div>

          <div className={styles.playerWrap}>
            <video
              key={activeVideo.src}
              controls
              playsInline
              preload="metadata"
              poster={activeVideo.poster}
              aria-label={activeVideo.title}
            >
              <source src={activeVideo.src} type="video/mp4" />
              {t("Your browser does not support MP4 video.", "您的浏览器不支持 MP4 视频。")}
            </video>
            <div className={styles.playerCaption}>
              <div>
                <strong>{activeVideo.title}</strong>
                <span>{activeVideo.summary}</span>
              </div>
              <span className={styles.replayBadge}>
                {selectedCase
                  ? selectedCase.kind === "negative_control"
                    ? t("Historical graphite · negative control", "历史石墨灰 · 负向对照")
                    : t("Historical graphite · recorded case", "历史石墨灰 · 已录案例")
                  : selectedVideo === "tennis"
                    ? t("Soft view · source nominal-0", "柔和外观 · 源案例 nominal-0")
                    : t("Soft appearance revision", "柔和外观改版")}
              </span>
            </div>
          </div>

          <section className={styles.caseLibrary} id="cases" aria-labelledby="cases-title">
            <div className={styles.caseHeader}>
              <div>
                <p className={styles.kicker}>{t("SOURCE-BOUND V2 MATRIX", "源绑定 V2 矩阵")}</p>
                <h3 id="cases-title">{t("Historical 32-case catalog", "历史 32 案例目录")}</h3>
              </div>
              {casesManifest && (
                <div className={styles.caseSummary} aria-label={t("Case manifest summary", "案例清单摘要")}>
                  <span>
                    <strong>{casesManifest.summary.positivePassed}/{casesManifest.summary.positiveTotal}</strong>
                    {t("positive task passes", "正向任务通过")}
                  </span>
                  <span>
                    <strong>{casesManifest.summary.controlsMatched}/{casesManifest.summary.controlsTotal}</strong>
                    {t("controls matched", "对照符合预期")}
                  </span>
                </div>
              )}
            </div>

            <div className={styles.historicalNotice}>
              <strong>{t("Original graphite camera palette", "原版石墨灰相机配色")}</strong>
              <span>
                {t(
                  "These 32 recorded cases predate the soft camera revision. They document the same simulated controller and evidence matrix, not 32 fresh soft-style evaluations.",
                  "这 32 个已录案例早于柔和相机改版。它们记录的是同一仿真控制器与证据矩阵，并非 32 次全新的柔和配色评估。",
                )}
              </span>
            </div>

            <div className={styles.caseFilters}>
              <div role="group" aria-label={t("Filter cases by variant", "按变体筛选案例")}>
                {(["all", "nominal", "small", "large", "control"] as const).map((variant) => (
                  <button
                    key={variant}
                    type="button"
                    aria-pressed={caseVariant === variant}
                    onClick={() => setCaseVariant(variant)}
                  >
                    {variant === "all" ? t("All variants", "全部变体") : variant}
                  </button>
                ))}
              </div>
              <div role="group" aria-label={t("Filter cases by evidence kind", "按证据类型筛选案例")}>
                {(["all", "positive", "negative_control"] as const).map((kind) => (
                  <button
                    key={kind}
                    type="button"
                    aria-pressed={caseKind === kind}
                    onClick={() => setCaseKind(kind)}
                  >
                    {kind === "all"
                      ? t("All evidence", "全部证据")
                      : kind === "positive"
                        ? t("Positive", "正向")
                        : t("Negative controls", "负向对照")}
                  </button>
                ))}
              </div>
            </div>

            {casesError ? (
              <p className={styles.caseStatus} role="status">
                {t(
                  "The static case manifest is not available yet. Main replay videos remain available above.",
                  "静态案例清单尚不可用。上方主回放视频仍可使用。",
                )}
              </p>
            ) : !casesManifest ? (
              <p className={styles.caseStatus} role="status">{t("Loading case manifest…", "正在加载案例清单…")}</p>
            ) : shownCases.length === 0 ? (
              <p className={styles.caseStatus}>{t("No cases match these filters.", "没有案例符合这些筛选条件。")}</p>
            ) : (
              <div className={styles.caseGrid}>
                {shownCases.map((entry) => {
                  const isControl = entry.kind === "negative_control";
                  const outcomeLabel = isControl
                    ? entry.expectedOutcomeMatched
                      ? t("Control matched", "对照符合预期")
                      : t("Control mismatch", "对照不符合预期")
                    : entry.taskSuccess
                      ? t("Task passed", "任务通过")
                      : t("Task not passed", "任务未通过");
                  return (
                    <article key={entry.id} className={selectedCase?.id === entry.id ? styles.caseSelected : undefined}>
                      <button
                        type="button"
                        className={styles.caseSelect}
                        aria-pressed={selectedCase?.id === entry.id}
                        onClick={() => setSelectedCase(entry)}
                      >
                        <span className={isControl ? styles.controlKind : styles.positiveKind}>
                          {isControl ? t("Negative control", "负向对照") : t("Positive", "正向")}
                        </span>
                        <strong>{entry.id}</strong>
                        <small>{entry.variant} · seed {entry.seed}</small>
                        <span className={entry.expectedOutcomeMatched ? styles.outcomeGood : styles.outcomeWarn}>{outcomeLabel}</span>
                      </button>
                      <div className={styles.caseMeta}>
                        <span>{entry.durationSeconds.toFixed(2)}s</span>
                        <span>{entry.frames} {t("frames", "帧")}</span>
                        <span>{entry.zeroError ? t("recorded-state match", "已记录状态一致") : t("replay discrepancy", "回放存在偏差")}</span>
                      </div>
                      <div className={styles.caseDownloads}>
                        <a href={wingPodCaseAssetUrl(entry.video)} download>
                          <DownloadIcon />MP4
                        </a>
                        <a href={wingPodCaseAssetUrl(entry.receipt)} download>
                          <DownloadIcon />JSON
                        </a>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>
        </section>

        <section className={styles.sequenceSection}>
          <div className={styles.sectionCopy}>
            <p className={styles.kicker}>{t("TASK SEQUENCE", "任务序列")}</p>
            <h2>{t("Contact sheet for frame-by-frame review", "用于逐帧检查的序列图")}</h2>
            <p>
              {t(
                "Use the sequence sheet to inspect positioning, contact timing, and recovery from source case nominal-0. The soft primary video replays that exact motion; neither view establishes new training or vision-guided control.",
                "使用序列图检查源案例 nominal-0 的定位、接触时机与恢复。柔和版主视频原样重放该动作；两者都不证明新训练或视觉引导控制。",
              )}
            </p>
          </div>
          <a className={styles.sequenceImage} href={wingPodAssetUrl("sequence-contact-sheet.jpg")} download>
            <Image
              src={wingPodAssetUrl("sequence-contact-sheet.jpg")}
              alt={t("Contact sheet from the WingPod tennis-return simulation replay", "WingPod 捡球归桶仿真回放序列图")}
              width={1280}
              height={1530}
              sizes="(max-width: 900px) 100vw, 60vw"
            />
            <span><DownloadIcon />{t("Download contact sheet", "下载序列图")}</span>
          </a>
        </section>

        <section className={styles.notesSection} aria-labelledby="notes-title">
          <div className={styles.sectionHeading}>
            <div>
              <p className={styles.kicker}>{t("CAMERA DESIGN NOTES", "相机设计说明")}</p>
              <h2 id="notes-title">{t("Known geometry, open hardware decisions", "已知几何与待定硬件决策")}</h2>
            </div>
          </div>
          <div className={styles.notesGrid}>
            <article>
              <span>01</span>
              <h3>{t("Optics remain unspecified", "光学器件尚未指定")}</h3>
              <p>{t("The camera SKU, sensor, field of view, focus distance, and interface are unknown.", "相机 SKU、传感器、视场角、对焦距离与接口均未知。")}</p>
            </article>
            <article>
              <span>02</span>
              <h3>{t("Dimensions are a proposal", "尺寸属于方案值")}</h3>
              <p>{t("The 28 × 17 × 5 mm housing and 13 mm lens separation require physical fit and tolerance checks.", "28 × 17 × 5 mm 外壳与 13 mm 镜头间距仍需实体装配和公差检查。")}</p>
            </article>
            <article>
              <span>03</span>
              <h3>{t("Replay is not perception", "回放不等于感知")}</h3>
              <p>{t("The soft tennis video exactly replays source nominal-0. Camera frames do not drive the unchanged simulated controller.", "柔和版网球视频原样重放源案例 nominal-0；未改变的仿真控制器不使用相机帧。")}</p>
            </article>
          </div>
        </section>

        <section className={styles.downloadSection} id="downloads" aria-labelledby="downloads-title">
          <div className={styles.downloadIntro}>
            <p className={styles.kicker}>{t("DESIGN PACKAGE", "设计包")}</p>
            <h2 id="downloads-title">{t("BOM and evidence downloads", "BOM 与证据下载")}</h2>
            <p>
              {t(
                "Download the generated inventory in the format that fits your review workflow. The interface intentionally reads no fixed row count.",
                "按适合审查流程的格式下载生成的物料清单。界面不会写死任何行数。",
              )}
            </p>
          </div>
          <div className={styles.downloadGrid}>
            {WINGPOD_DOWNLOADS.map((item) => (
              <a key={item.file} href={wingPodAssetUrl(item.file)} download>
                <span className={styles.fileType}>{item.format}</span>
                <span><strong>{item.label[language]}</strong><small>{item.file}</small></span>
                <DownloadIcon />
              </a>
            ))}
          </div>
          <nav className={styles.documentLinks} aria-label={t("WingPod supporting documents", "WingPod 支持文档")}>
            {WINGPOD_DOCUMENTS.map((item) => (
              <a key={item.file} href={wingPodAssetUrl(item.file)} download>
                <span>{item.label[language]}</span><code>{item.file}</code><ArrowIcon />
              </a>
            ))}
            <a href={publicAssetUrl("/wingpod-v2/index.html")}>
              <span>{t("Standalone WingPod v2 page", "独立 WingPod v2 页面")}</span><code>/wingpod-v2/index.html</code><ArrowIcon />
            </a>
          </nav>
        </section>
      </main>

      <footer className={styles.footer}>
        <span>WingPod Camera v2</span>
        <span>{t("Simulation evidence · hardware decisions open", "仿真证据 · 硬件决策待定")}</span>
        <Link href="/">{t("Return to Studio", "返回 Studio")} <ArrowIcon /></Link>
      </footer>
    </div>
  );
}
