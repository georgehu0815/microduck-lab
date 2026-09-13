"use client";

import Image from "next/image";
import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import {
  classroomAssetUrl,
  parseClassroomCatalog,
  type ClassroomCatalog,
  type ClassroomLesson,
} from "@/lib/classroom";
import { publicAssetUrl } from "@/lib/static-assets";
import { LanguageToggle, useLanguage } from "./LanguageProvider";
import styles from "./ClassroomPage.module.css";

function isInteractiveTarget(target: EventTarget | null) {
  return target instanceof HTMLElement
    && Boolean(target.closest("a, button, input, select, textarea, video"));
}

export default function ClassroomPage() {
  const { language, setLanguage, t } = useLanguage();
  const [catalog, setCatalog] = useState<ClassroomCatalog | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<"http" | "invalid" | "network" | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [selectedLessonId, setSelectedLessonId] = useState<string | null>(null);
  const [slideIndex, setSlideIndex] = useState(0);
  const [fullscreenMessage, setFullscreenMessage] = useState("");
  const playerRef = useRef<HTMLElement>(null);

  const loadCatalog = useCallback(async (signal?: AbortSignal) => {
    setLoading(true);
    setError(null);
    setErrorStatus(null);
    try {
      const response = await fetch(publicAssetUrl("/classroom-assets/catalog.json"), {
        cache: "no-store",
        signal,
      });
      if (!response.ok) {
        setError("http");
        setErrorStatus(response.status);
        return;
      }
      try {
        setCatalog(parseClassroomCatalog(await response.json()));
      } catch {
        setError("invalid");
      }
    } catch {
      if (signal?.aborted) return;
      setError("network");
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(() => void loadCatalog(controller.signal), 0);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [loadCatalog]);

  const lessons = useMemo(
    () => [...(catalog?.lessons ?? [])].sort((left, right) => left.order - right.order),
    [catalog],
  );
  const selectedLesson = lessons.find((lesson) => lesson.id === selectedLessonId) ?? null;
  const selectedEdition = selectedLesson?.editions[language] ?? null;
  const slides = selectedEdition?.slides ?? [];
  const activeSlideIndex = Math.min(slideIndex, Math.max(slides.length - 1, 0));
  const activeSlide = slides[activeSlideIndex] ?? null;

  function openLesson(lesson: ClassroomLesson) {
    setSelectedLessonId(lesson.id);
    setSlideIndex(0);
    setFullscreenMessage("");
    window.requestAnimationFrame(() => {
      playerRef.current?.scrollIntoView({ behavior: "instant", block: "start" });
      playerRef.current?.focus({ preventScroll: true });
    });
  }

  function closeLesson() {
    document.querySelector<HTMLButtonElement>(`[data-lesson-id="${selectedLessonId}"]`)?.focus();
    setSelectedLessonId(null);
    setSlideIndex(0);
    setFullscreenMessage("");
  }

  function selectEdition(nextEdition: "en" | "zh") {
    setLanguage(nextEdition);
    setFullscreenMessage("");
  }

  function showPreviousSlide() {
    setSlideIndex(Math.max(0, activeSlideIndex - 1));
  }

  function showNextSlide() {
    setSlideIndex(Math.min(slides.length - 1, activeSlideIndex + 1));
  }

  function handlePlayerKeyDown(event: KeyboardEvent<HTMLElement>) {
    if (isInteractiveTarget(event.target) || slides.length === 0) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      showPreviousSlide();
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      showNextSlide();
    } else if (event.key === "Home") {
      event.preventDefault();
      setSlideIndex(0);
    } else if (event.key === "End") {
      event.preventDefault();
      setSlideIndex(slides.length - 1);
    }
  }

  async function enterFullscreen() {
    const target = playerRef.current;
    if (!target?.requestFullscreen) {
      setFullscreenMessage(t(
        "Fullscreen is not available in this browser. The presentation remains available inline.",
        "此浏览器不支持全屏。演示仍可在页面内查看。",
      ));
      return;
    }
    try {
      await target.requestFullscreen();
      setFullscreenMessage("");
    } catch {
      setFullscreenMessage(t(
        "Fullscreen could not be opened. The presentation remains available inline.",
        "无法打开全屏。演示仍可在页面内查看。",
      ));
    }
  }

  return (
    <main className={styles.page}>
      <nav className={styles.navigation} aria-label={t("Workspace navigation", "工作区导航")}>
        <Link href="/">{t("Studio", "工作台")}</Link>
        <Link href="/arm">{t("Arm Lab", "机械臂实验室")}</Link>
        <Link href="/wingpod">WingPod</Link>
        <Link href="/classroom" aria-current="page">{t("Classroom PPT", "课堂课件")}</Link>
      </nav>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>{t("MICRODUCK CLASSROOM", "MICRODUCK 课堂")}</p>
          <h1>{t("Eight skills, ready to teach", "八项技能，随时开课")}</h1>
          <p className={styles.intro}>
            {t(
              "Open a lesson, present its static PDF slides, watch the recorded examples, or download the original teaching files.",
              "打开课程，展示静态 PDF 幻灯片，观看录制示例，或下载原始教学文件。",
            )}
          </p>
        </div>
        <LanguageToggle />
      </header>

      <section className={styles.catalogSection} aria-labelledby="catalog-heading">
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.eyebrow}>{t("LESSON CATALOG", "课程目录")}</p>
            <h2 id="catalog-heading">{t("Choose a lesson", "选择课程")}</h2>
          </div>
          <p role="status">
            {loading
              ? t("Loading lessons…", "正在加载课程…")
              : catalog
                ? t(`${lessons.length} lessons available`, `共 ${lessons.length} 节课程`)
                : t("Catalog unavailable", "目录不可用")}
          </p>
        </div>

        {error && (
          <div className={styles.error} role="alert">
            <p>
              {error === "http"
                ? t(
                    `The classroom catalog could not be loaded (${errorStatus}).`,
                    `课堂目录无法加载（${errorStatus}）。`,
                  )
                : error === "invalid"
                  ? t("The classroom catalog is invalid.", "课堂目录格式无效。")
                  : t("The classroom catalog is unavailable.", "课堂目录暂时不可用。")}
            </p>
            <button type="button" onClick={() => void loadCatalog()}>
              {t("Retry", "重试")}
            </button>
          </div>
        )}

        {loading && !catalog && (
          <div className={styles.loadingGrid} aria-hidden="true">
            {Array.from({ length: 8 }, (_, index) => <span key={index} />)}
          </div>
        )}

        {catalog && (
          <div className={styles.lessonGrid}>
            {lessons.map((lesson) => (
              <button
                type="button"
                className={selectedLessonId === lesson.id ? styles.selectedCard : styles.lessonCard}
                key={lesson.id}
                data-testid="lesson-card"
                data-lesson-id={lesson.id}
                aria-pressed={selectedLessonId === lesson.id}
                onClick={() => openLesson(lesson)}
              >
                <span className={styles.lessonNumber}>{String(lesson.order).padStart(2, "0")}</span>
                <span className={styles.cardCopy}>
                  <strong lang="en">{lesson.title.en}</strong>
                  <span lang="zh-CN">{lesson.title.zh}</span>
                </span>
                <span className={styles.cardFooter}>
                  <span>English</span>
                  <span lang="zh-CN">中文</span>
                  <span>{lesson.editions[language].slides.length} {t("slides", "页")}</span>
                </span>
              </button>
            ))}
          </div>
        )}
      </section>

      {selectedLesson && selectedEdition && (
        <section
          ref={playerRef}
          className={styles.player}
          aria-labelledby="presentation-heading"
          data-testid="slide-player"
          tabIndex={0}
          onKeyDown={handlePlayerKeyDown}
        >
          <div className={styles.playerTopbar}>
            <button
              type="button"
              className={styles.backButton}
              data-testid="close-presentation"
              onClick={closeLesson}
            >
              <span aria-hidden="true">←</span> {t("Back to catalog", "返回目录")}
            </button>
            <div className={styles.editionToggle} role="group" aria-label={t("Lesson edition", "课程版本")}>
              <button type="button" lang="en" aria-pressed={language === "en"} onClick={() => selectEdition("en")}>
                English
              </button>
              <button type="button" lang="zh-CN" aria-pressed={language === "zh"} onClick={() => selectEdition("zh")}>
                中文
              </button>
            </div>
          </div>

          <div className={styles.presentationHeading}>
            <div>
              <p className={styles.eyebrow}>
                {t("STATIC PDF PREVIEW · NO ANIMATIONS OR EMBEDDED PPT PLAYBACK", "静态 PDF 预览 · 不含动画或嵌入式 PPT 播放")}
              </p>
              <h2 id="presentation-heading">{selectedLesson.title[language]}</h2>
              <p>
                {t(
                  `Lesson ${selectedLesson.order} of ${lessons.length}`,
                  `第 ${selectedLesson.order} 课，共 ${lessons.length} 课`,
                )}
              </p>
            </div>
            <div className={styles.downloads}>
              <a href={classroomAssetUrl(selectedEdition.pptx)} download>
                <span aria-hidden="true">↓</span> PPTX
              </a>
              <a href={classroomAssetUrl(selectedEdition.pdf)} download>
                <span aria-hidden="true">↓</span> PDF
              </a>
            </div>
          </div>

          {activeSlide ? (
            <>
              <div className={styles.slideStage}>
                <Image
                  className={styles.slideImage}
                  data-testid="slide-image"
                  src={classroomAssetUrl(activeSlide.image)}
                  alt={activeSlide.title}
                  width={1200}
                  height={675}
                  unoptimized
                  priority={activeSlideIndex === 0}
                  sizes="(max-width: 900px) 100vw, 1200px"
                />
              </div>

              <div className={styles.transport}>
                <button
                  type="button"
                  data-testid="previous-slide"
                  disabled={activeSlideIndex === 0}
                  onClick={showPreviousSlide}
                  aria-label={t("Previous slide", "上一页")}
                >
                  <span aria-hidden="true">←</span>
                  <span>{t("Previous", "上一页")}</span>
                </button>
                <label>
                  <span>{t("Slide", "幻灯片")}</span>
                  <select
                    data-testid="slide-select"
                    value={activeSlideIndex}
                    onChange={(event) => setSlideIndex(Number(event.target.value))}
                    aria-label={t("Choose slide", "选择幻灯片")}
                  >
                    {slides.map((slide, index) => (
                      <option key={`${slide.image}-${index}`} value={index}>
                        {index + 1}. {slide.title}
                      </option>
                    ))}
                  </select>
                </label>
                <p aria-live="polite">
                  <strong>{activeSlideIndex + 1}</strong> / {slides.length}
                </p>
                <button
                  type="button"
                  data-testid="next-slide"
                  disabled={activeSlideIndex === slides.length - 1}
                  onClick={showNextSlide}
                  aria-label={t("Next slide", "下一页")}
                >
                  <span>{t("Next", "下一页")}</span>
                  <span aria-hidden="true">→</span>
                </button>
                <button type="button" className={styles.fullscreenButton} onClick={() => void enterFullscreen()}>
                  <span aria-hidden="true">⛶</span>
                  <span>{t("Fullscreen", "全屏")}</span>
                </button>
              </div>

              {fullscreenMessage && <p className={styles.fullscreenMessage} role="status">{fullscreenMessage}</p>}

              <article className={styles.slideDetails}>
                <p className={styles.eyebrow}>{t("SLIDE NOTES", "幻灯片内容")}</p>
                <h3>{activeSlide.title}</h3>
                <p>{activeSlide.text}</p>
              </article>
            </>
          ) : (
            <div className={styles.empty} role="status">
              {t("This edition has no slide previews.", "此版本没有幻灯片预览。")}
            </div>
          )}

          <section className={styles.videoSection} aria-labelledby="lesson-videos-heading">
            <div className={styles.videoHeading}>
              <div>
                <p className={styles.eyebrow}>{t("RECORDED EXAMPLES", "录制示例")}</p>
                <h3 id="lesson-videos-heading">{t("Lesson videos", "课程视频")}</h3>
              </div>
              <p>{selectedEdition.videos.length} {t("videos", "个视频")}</p>
            </div>
            {selectedEdition.videos.length > 0 ? (
              <div className={styles.videoGrid}>
                {selectedEdition.videos.map((video) => (
                  <figure key={`${video.file}-${video.label}`}>
                    <video
                      controls
                      playsInline
                      preload="metadata"
                      src={classroomAssetUrl(video.file)}
                      aria-label={video.label}
                    >
                      {t("Your browser cannot play this video.", "浏览器无法播放此视频。")}
                    </video>
                    <figcaption>{video.label}</figcaption>
                  </figure>
                ))}
              </div>
            ) : (
              <p className={styles.empty}>{t("No separate videos are included with this edition.", "此版本没有单独的视频。")}</p>
            )}
          </section>
        </section>
      )}
    </main>
  );
}
