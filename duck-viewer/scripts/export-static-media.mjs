import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const publicRoot = path.join(root, "public", "static");
const endpoint = process.env.DUCK_VIEWER_CATALOG_URL ?? "http://127.0.0.1:63317";
const generatedAt = new Date().toISOString();
const backups = path.resolve(root, "../.omx/artifacts/media-export-backups");
await mkdir(backups, { recursive: true });
const staging = await mkdtemp(path.join(backups, "pending-"));
const media = [];

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

async function download(url, relativePath, expectedHash) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  assert.ok(bytes.length, `Empty asset: ${url}`);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (expectedHash) assert.equal(sha256, expectedHash, `Changed media: ${url}`);
  const filename = relativePath.replace("{hash}", sha256);
  const destination = path.join(staging, filename);
  await mkdir(path.dirname(destination), { recursive: true });
  await writeFile(destination, bytes);
  media.push({ path: `/static/${filename}`, bytes: bytes.length, sha256 });
  return `/static/${filename}`;
}

const armCatalog = await getJson(`${endpoint}/api/arm/videos`);
assert.ok(Array.isArray(armCatalog.videos) && armCatalog.videos.length > 0, "Arm catalog is empty");
assert.equal(armCatalog.hardwareEnabled, false);
for (const video of armCatalog.videos) {
  assert.match(video.videoHash, /^[a-f0-9]{64}$/);
  video.videoUrl = await download(new URL(video.videoUrl, endpoint), "arm-videos/media/{hash}.mp4", video.videoHash);
  if (video.evidenceUrl) {
    video.evidenceUrl = await download(new URL(video.evidenceUrl, endpoint), "arm-videos/media/{hash}.json");
  }
}
await mkdir(path.join(staging, "arm-videos"), { recursive: true });
await writeFile(
  path.join(staging, "arm-videos", "catalog.json"),
  `${JSON.stringify({ ...armCatalog, generatedAt }, null, 2)}\n`,
);

const studioCatalog = await getJson(`${endpoint}/api/rlx/runs`);
const exportedRuns = [];
for (const run of studioCatalog.runs) {
  if (!run.video || !run.renderVerified || !run.renderEvidenceId) continue;
  assert.match(run.experimentId, /^[a-z0-9-]+$/);
  assert.match(run.runName, /^[a-zA-Z0-9_-]+$/);
  const directory = path.join("studio-runs", run.experimentId, run.runName);
  const query = new URLSearchParams({ experiment: run.experimentId, run: run.runName });
  const artifactQuery = new URLSearchParams(query);
  artifactQuery.set("v", run.renderEvidenceId);
  const before = await getJson(`${endpoint}/api/rlx?${query}`);
  assert.equal(before.renderVerified, true, `Stale render: ${run.runName}`);
  assert.equal(before.renderEvidenceId, run.renderEvidenceId, `Changed render: ${run.runName}`);
  artifactQuery.set("kind", "video");
  artifactQuery.set("inline", "1");
  await download(`${endpoint}/api/rlx/artifact?${artifactQuery}`, path.join(directory, "video.mp4"), run.renderEvidenceId);
  artifactQuery.set("kind", "sheet");
  artifactQuery.delete("inline");
  await download(`${endpoint}/api/rlx/artifact?${artifactQuery}`, path.join(directory, "sheet.png"));
  const after = await getJson(`${endpoint}/api/rlx?${query}`);
  assert.equal(after.renderVerified, true, `Render invalidated during export: ${run.runName}`);
  assert.equal(after.renderEvidenceId, before.renderEvidenceId, `Render changed during export: ${run.runName}`);
  assert.equal(after.evaluation?.source_sha256, before.evaluation?.source_sha256, `Policy changed during export: ${run.runName}`);
  const evaluationBytes = Buffer.from(JSON.stringify(after, null, 2) + "\n");
  await writeFile(path.join(staging, directory, "evaluation.json"), evaluationBytes);
  media.push({ path: `/static/${directory}/evaluation.json`, bytes: evaluationBytes.length, sha256: createHash("sha256").update(evaluationBytes).digest("hex") });
  exportedRuns.push(run);
}
assert.ok(exportedRuns.length > 0, "No verified Studio runs; previous export retained");
await mkdir(path.join(staging, "studio-runs"), { recursive: true });
await writeFile(
  path.join(staging, "studio-runs", "catalog.json"),
  `${JSON.stringify({ generatedAt, runs: exportedRuns }, null, 2)}\n`,
);
await writeFile(path.join(staging, "manifest.json"), JSON.stringify({ generatedAt, files: media }, null, 2) + "\n");
const backup = path.join(backups, `previous-${generatedAt.replace(/[:.]/g, "-")}`);
let previousMoved = false;
try {
  await rename(publicRoot, backup);
  previousMoved = true;
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}
try {
  await rename(staging, publicRoot);
} catch (error) {
  if (previousMoved) await rename(backup, publicRoot);
  throw error;
}

console.log(`exported ${armCatalog.videos.length} arm videos`);
console.log(`exported ${exportedRuns.length} verified Studio runs`);
