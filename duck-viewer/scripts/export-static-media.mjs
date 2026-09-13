import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const publicRoot = path.join(root, "public", "static");
const endpoint = process.env.DUCK_VIEWER_CATALOG_URL ?? "http://127.0.0.1:63317";

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

async function download(url, destination) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, Buffer.from(await response.arrayBuffer()));
}

await rm(publicRoot, { recursive: true, force: true });

const armCatalog = await getJson(`${endpoint}/api/arm/videos`);
for (const [index, video] of armCatalog.videos.entries()) {
  const stem = String(index + 1).padStart(3, "0");
  const videoPath = `/static/arm-videos/media/${stem}.mp4`;
  await download(new URL(video.videoUrl, endpoint), path.join(root, "public", videoPath));
  video.videoUrl = videoPath;
  if (video.evidenceUrl) {
    const evidencePath = `/static/arm-videos/media/${stem}.json`;
    await download(new URL(video.evidenceUrl, endpoint), path.join(root, "public", evidencePath));
    video.evidenceUrl = evidencePath;
  }
}
await mkdir(path.join(publicRoot, "arm-videos"), { recursive: true });
await writeFile(
  path.join(publicRoot, "arm-videos", "catalog.json"),
  `${JSON.stringify(armCatalog, null, 2)}\n`,
);

const studioCatalog = await getJson(`${endpoint}/api/rlx/runs`);
const exportedRuns = [];
for (const run of studioCatalog.runs) {
  if (!run.video || !run.renderVerified || !run.renderEvidenceId) continue;
  const directory = path.join(publicRoot, "studio-runs", run.experimentId, run.runName);
  const query = new URLSearchParams({ experiment: run.experimentId, run: run.runName });
  const artifactQuery = new URLSearchParams(query);
  artifactQuery.set("v", run.renderEvidenceId);
  try {
    artifactQuery.set("kind", "video");
    artifactQuery.set("inline", "1");
    await download(`${endpoint}/api/rlx/artifact?${artifactQuery}`, path.join(directory, "video.mp4"));
    artifactQuery.set("kind", "sheet");
    artifactQuery.delete("inline");
    await download(`${endpoint}/api/rlx/artifact?${artifactQuery}`, path.join(directory, "sheet.png"));
    await download(`${endpoint}/api/rlx?${query}`, path.join(directory, "evaluation.json"));
    exportedRuns.push(run);
  } catch (error) {
    console.warn(`Skipping ${run.experimentId}/${run.runName}: ${error}`);
  }
}
await mkdir(path.join(publicRoot, "studio-runs"), { recursive: true });
await writeFile(
  path.join(publicRoot, "studio-runs", "catalog.json"),
  `${JSON.stringify({ runs: exportedRuns }, null, 2)}\n`,
);

console.log(`exported ${armCatalog.videos.length} arm videos`);
console.log(`exported ${exportedRuns.length} verified Studio runs`);