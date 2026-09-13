export const IS_STATIC_EXPORT =
  process.env.NEXT_PUBLIC_DUCK_VIEWER_STATIC_EXPORT === "1";

const BASE_PATH = process.env.NEXT_PUBLIC_DUCK_VIEWER_BASE_PATH ?? "";

export function publicAssetUrl(path: string): string {
  if (!path || /^(?:https?:|data:|blob:)/.test(path)) return path;
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return `${BASE_PATH}${normalized}`;
}

export function staticStudioArtifactUrl(
  experimentId: string,
  runName: string,
  kind: "video" | "sheet" | "evaluation",
  revision?: string | null,
): string {
  const extension = kind === "video" ? "mp4" : kind === "sheet" ? "png" : "json";
  const asset = publicAssetUrl(
    `/static/studio-runs/${encodeURIComponent(experimentId)}/${encodeURIComponent(runName)}/${kind}.${extension}`,
  );
  return revision ? `${asset}?v=${encodeURIComponent(revision)}` : asset;
}
