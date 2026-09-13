import { publicAssetUrl } from "./static-assets";

export type WingPodVideoId = "appearance" | "tennis";
export type WingPodScopeId = "simulation" | "no-vision" | "not-release";
export type WingPodCaseVariant = "nominal" | "small" | "large" | "control";
export type WingPodCaseKind = "positive" | "negative_control";
export type WingPodCaseFilter = WingPodCaseVariant | "all";

export type WingPodVideo = {
  id: WingPodVideoId;
  file: string;
  durationSeconds: number;
  poster: string;
  title: { en: string; zh: string };
  summary: { en: string; zh: string };
};

export type WingPodScopeBadge = {
  id: WingPodScopeId;
  label: { en: string; zh: string };
};

export type WingPodCase = {
  id: string;
  variant: WingPodCaseVariant;
  seed: number;
  kind: WingPodCaseKind;
  taskSuccess: boolean;
  expectedOutcomeMatched: boolean;
  video: string;
  receipt: string;
  durationSeconds: number;
  frames: number;
  zeroError: boolean;
};

export type WingPodCasesManifest = {
  cases: WingPodCase[];
  summary: {
    positivePassed: number;
    positiveTotal: number;
    controlsMatched: number;
    controlsTotal: number;
  };
};

export const WINGPOD_VIDEOS: readonly WingPodVideo[] = [
  {
    id: "appearance",
    file: "wingpod-camera-eyes.mp4",
    durationSeconds: 20,
    poster: "hero.png",
    title: { en: "Camera appearance · soft revision", zh: "相机外观 · 柔和改版" },
    summary: {
      en: "A 20-second appearance review of the cream face, sage lens rings, navy centers, peach cheeks, and honey beak.",
      zh: "20 秒外观检查，展示奶油白面部、鼠尾草灰镜圈、海军蓝镜心、桃色脸颊与蜂蜜色喙。",
    },
  },
  {
    id: "tennis",
    file: "tennis-return.mp4",
    durationSeconds: 50.84,
    poster: "sequence-contact-sheet.jpg",
    title: { en: "Tennis ball pick to bin", zh: "网球捡取归桶" },
    summary: {
      en: "Approach, grasp, lift, carry, lower, and release a tennis ball into the low supported bin placement in the exact 50.84-second nominal-0 replay. The simulated controller is unchanged.",
      zh: "在 50.84 秒 nominal-0 精确回放中完成接近、抓取、抬升、搬运、下降，并将网球释放到低位承托式桶中。仿真控制器未改变。",
    },
  },
] as const;

export const WINGPOD_SCOPE_BADGES: readonly WingPodScopeBadge[] = [
  {
    id: "simulation",
    label: { en: "Simulation replay · same controller", zh: "仿真回放 · 控制器未变" },
  },
  {
    id: "no-vision",
    label: { en: "No vision control", zh: "无视觉控制" },
  },
  {
    id: "not-release",
    label: { en: "Not new training or hardware validation", zh: "非新训练或硬件验证" },
  },
] as const;

export const WINGPOD_DOWNLOADS = [
  { file: "BOM.csv", format: "CSV", label: { en: "Bill of materials", zh: "物料清单" } },
  { file: "BOM.json", format: "JSON", label: { en: "Structured BOM", zh: "结构化 BOM" } },
  { file: "PARTS.md", format: "Markdown", label: { en: "Parts notes", zh: "零件说明" } },
] as const;

export const WINGPOD_DOCUMENTS = [
  { file: "cases.json", label: { en: "All case records", zh: "全部案例记录" } },
  { file: "CAMERA-V2.md", label: { en: "Camera v2 design notes", zh: "Camera v2 设计说明" } },
  { file: "CAMERA-V2-SOFT.md", label: { en: "Soft camera revision guide", zh: "柔和相机改版指南" } },
  { file: "camera-spec.json", label: { en: "Soft camera specification", zh: "柔和相机规格" } },
  { file: "appearance-validation.json", label: { en: "Appearance validation", zh: "外观验证记录" } },
  { file: "README.md", label: { en: "WingPod package README", zh: "WingPod 包说明" } },
  { file: "camera-task-verification.json", label: { en: "Task verification", zh: "任务验证记录" } },
  { file: "video-verification.json", label: { en: "Video verification", zh: "视频验证记录" } },
  { file: "manifest.json", label: { en: "Asset manifest", zh: "资源清单" } },
] as const;

export const WINGPOD_IMAGES: Readonly<Record<string, string>> = {
  "hero.png": "hero-soft-2fe5dab4cf70.png",
  "camera-eyes-detail.png": "camera-eyes-detail-soft-45582996692a.png",
  "sequence-contact-sheet.jpg": "sequence-contact-sheet-soft-956664897334.jpg",
};

export function wingPodAssetUrl(
  file: string,
  resolve: (path: string) => string = publicAssetUrl,
): string {
  return resolve(`/static/wingpod/${Object.hasOwn(WINGPOD_IMAGES, file) ? WINGPOD_IMAGES[file] : file}`);
}

export function createWingPodMediaModel(
  selectedId: WingPodVideoId | string,
  resolve: (path: string) => string = publicAssetUrl,
) {
  const selected = WINGPOD_VIDEOS.find((video) => video.id === selectedId) ?? WINGPOD_VIDEOS[0];
  return {
    selected,
    videoSrc: wingPodAssetUrl(selected.file, resolve),
    posterSrc: wingPodAssetUrl(selected.poster, resolve),
    downloadName: selected.file,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isSafeRelativeFile(value: unknown): value is string {
  return typeof value === "string"
    && value.length > 0
    && !value.startsWith("/")
    && /^[a-zA-Z0-9_/-]+\.(?:mp4|json)$/.test(value)
    && value.split("/").every((segment) => segment.length > 0 && segment !== "..");
}

function isCount(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0;
}

export function parseWingPodCasesManifest(value: unknown): WingPodCasesManifest {
  if (!isRecord(value) || !Array.isArray(value.cases) || !isRecord(value.summary)) {
    throw new Error("Invalid WingPod cases manifest");
  }
  const variants = new Set<WingPodCaseVariant>(["nominal", "small", "large", "control"]);
  const kinds = new Set<WingPodCaseKind>(["positive", "negative_control"]);
  const cases = value.cases.map((entry, index) => {
    if (
      !isRecord(entry)
      || typeof entry.id !== "string"
      || !entry.id
      || !variants.has(entry.variant as WingPodCaseVariant)
      || !kinds.has(entry.kind as WingPodCaseKind)
      || !isCount(entry.seed)
      || typeof entry.taskSuccess !== "boolean"
      || typeof entry.expectedOutcomeMatched !== "boolean"
      || !isSafeRelativeFile(entry.video)
      || !entry.video.endsWith(".mp4")
      || !isSafeRelativeFile(entry.receipt)
      || !entry.receipt.endsWith(".json")
      || typeof entry.durationSeconds !== "number"
      || !Number.isFinite(entry.durationSeconds)
      || entry.durationSeconds <= 0
      || !isCount(entry.frames)
      || typeof entry.zeroError !== "boolean"
    ) {
      throw new Error(`Invalid WingPod case at index ${index}`);
    }
    return {
      id: entry.id,
      variant: entry.variant as WingPodCaseVariant,
      seed: entry.seed,
      kind: entry.kind as WingPodCaseKind,
      taskSuccess: entry.taskSuccess,
      expectedOutcomeMatched: entry.expectedOutcomeMatched,
      video: entry.video,
      receipt: entry.receipt,
      durationSeconds: entry.durationSeconds,
      frames: entry.frames,
      zeroError: entry.zeroError,
    };
  });
  const { summary } = value;
  if (
    !isCount(summary.positivePassed)
    || !isCount(summary.positiveTotal)
    || !isCount(summary.controlsMatched)
    || !isCount(summary.controlsTotal)
    || summary.positivePassed > summary.positiveTotal
    || summary.controlsMatched > summary.controlsTotal
  ) {
    throw new Error("Invalid WingPod cases summary");
  }
  const positives = cases.filter((entry) => entry.kind === "positive");
  const controls = cases.filter((entry) => entry.kind === "negative_control");
  if (
    new Set(cases.map((entry) => entry.id)).size !== cases.length
    || cases.some((entry) => (entry.variant === "control") !== (entry.kind === "negative_control"))
    || cases.some((entry) => entry.expectedOutcomeMatched !== (entry.kind === "positive" ? entry.taskSuccess : !entry.taskSuccess))
    || summary.positiveTotal !== positives.length
    || summary.positivePassed !== positives.filter((entry) => entry.taskSuccess && entry.zeroError).length
    || summary.controlsTotal !== controls.length
    || summary.controlsMatched !== controls.filter((entry) => entry.expectedOutcomeMatched && entry.zeroError).length
  ) {
    throw new Error("WingPod summary does not match case evidence");
  }
  return {
    cases,
    summary: {
      positivePassed: summary.positivePassed,
      positiveTotal: summary.positiveTotal,
      controlsMatched: summary.controlsMatched,
      controlsTotal: summary.controlsTotal,
    },
  };
}

export function filterWingPodCases(
  cases: readonly WingPodCase[],
  variant: WingPodCaseFilter,
  kind: WingPodCaseKind | "all",
): WingPodCase[] {
  return cases.filter((entry) =>
    (variant === "all" || entry.variant === variant)
    && (kind === "all" || entry.kind === kind)
  );
}

export function wingPodCaseAssetUrl(
  file: string,
  resolve: (path: string) => string = publicAssetUrl,
): string {
  if (!isSafeRelativeFile(file)) throw new Error("Unsafe WingPod case asset path");
  return wingPodAssetUrl(file, resolve);
}
