import { publicAssetUrl } from "./static-assets";

export type ClassroomSlide = { image: string; title: string; text: string };
export type ClassroomEdition = {
  pptx: string;
  pdf: string;
  slides: ClassroomSlide[];
  videos: { file: string; label: string }[];
};
export type ClassroomLesson = {
  id: string;
  order: number;
  title: { en: string; zh: string };
  editions: { en: ClassroomEdition; zh: ClassroomEdition };
};
export type ClassroomCatalog = { schemaVersion: 1; lessons: ClassroomLesson[] };

function validPath(value: unknown): value is string {
  return typeof value === "string" && /^[a-zA-Z0-9._/-]+$/.test(value)
    && value.split("/").every((segment) => segment && segment !== "." && segment !== "..");
}

export function classroomAssetUrl(file: string, resolve: (path: string) => string = publicAssetUrl): string {
  if (!validPath(file)) throw new Error("Invalid classroom asset path");
  return resolve(`/classroom-assets/${file}`);
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function edition(value: unknown): value is ClassroomEdition {
  if (!record(value) || !validPath(value.pptx) || !value.pptx.endsWith(".pptx")
    || !validPath(value.pdf) || !value.pdf.endsWith(".pdf")
    || !Array.isArray(value.slides) || value.slides.length === 0 || !Array.isArray(value.videos)) return false;
  return value.slides.every((slide) => record(slide) && validPath(slide.image)
    && slide.image.endsWith(".jpg") && typeof slide.title === "string" && typeof slide.text === "string")
    && value.videos.every((video) => record(video) && validPath(video.file)
      && video.file.endsWith(".mp4") && typeof video.label === "string");
}

export function parseClassroomCatalog(value: unknown): ClassroomCatalog {
  if (!record(value) || value.schemaVersion !== 1 || !Array.isArray(value.lessons) || value.lessons.length === 0)
    throw new Error("Invalid classroom catalog");
  const identities = new Set<string>();
  const orders = new Set<number>();
  for (const lesson of value.lessons) {
    if (!record(lesson) || typeof lesson.id !== "string" || !/^[a-z0-9-]+$/.test(lesson.id)
      || identities.has(lesson.id) || !Number.isSafeInteger(lesson.order) || (lesson.order as number) < 1
      || orders.has(lesson.order as number) || !record(lesson.title)
      || typeof lesson.title.en !== "string" || typeof lesson.title.zh !== "string"
      || !record(lesson.editions) || !edition(lesson.editions.en) || !edition(lesson.editions.zh))
      throw new Error("Invalid classroom lesson");
    identities.add(lesson.id);
    orders.add(lesson.order as number);
  }
  return value as ClassroomCatalog;
}
