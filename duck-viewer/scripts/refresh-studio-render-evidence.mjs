import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

if (!process.argv.includes("--execute")) throw new Error("Use --execute to refresh saved rollout artifacts. Existing MP4s and sheets will be backed up first.");
const base = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const directory = path.resolve(process.env.STUDIO_EVIDENCE_DIR || "../rlx/artifacts/studio-ui-20260908/render-refresh");
await mkdir(directory, { recursive: true });
async function json(url, init) {
  const response = await fetch(new URL(url, base), { ...init, signal: AbortSignal.timeout(60000) });
  const data = await response.json();
  if (!response.ok) throw new Error(JSON.stringify(data));
  return data;
}
const results = [];
const skips = [];
const verifiedExperiments = ["dance", "swing", "running", "stilts", "backflip"];
const unverifiedExperiments = new Map([
  ["basketball", "Balance-only evidence is not a passed steering task."],
  ["bridge", "Training pilot exists, but a complete crossing is not verified."],
]);
try {
  const { runs } = await json("/api/rlx/runs");
  for (const [experiment, reason] of unverifiedExperiments) {
    const candidates = runs.filter((candidate) => candidate.experimentId === experiment);
    skips.push({
      experiment,
      reason,
      savedRuns: candidates.map((candidate) => ({
        runName: candidate.runName,
        taskPassed: candidate.taskPassed === true,
        balanceOnly: candidate.balanceOnly === true,
      })),
    });
    console.log(`${experiment}: skipped - ${reason}`);
  }
  for (const experiment of verifiedExperiments) {
    const run = runs.find((candidate) => candidate.experimentId === experiment && candidate.taskPassed);
    assert.ok(run, `No verified ${experiment} run available`);
    const query = `experiment=${experiment}&run=${encodeURIComponent(run.runName)}`;
    const before = await json(`/api/rlx?${query}`);
    assert.ok(before.savedRecipe);
    const backupDirectory = path.join(directory, run.runName);
    await mkdir(backupDirectory, { recursive: true });
    const backupHashes = {};
    for (const kind of ["video", "sheet"]) {
      const response = await fetch(`${base}/api/rlx/artifact?${query}&kind=${kind}`);
      assert.ok(response.ok);
      const bytes = Buffer.from(await response.arrayBuffer());
      backupHashes[kind] = createHash("sha256").update(bytes).digest("hex");
      await writeFile(path.join(backupDirectory, `original-${backupHashes[kind]}.${kind === "video" ? "mp4" : "png"}`), bytes, { flag: "wx" }).catch((error) => { if (error.code !== "EEXIST") throw error; });
    }
    console.log(`${experiment}: backed up saved video and sheet; rendering ${run.runName}`);
    await json("/api/rlx", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "render", recipe: before.savedRecipe }) });
    const deadline = Date.now() + 300000;
    let after;
    do {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      after = await json(`/api/rlx?${query}`);
      if (after.phase !== "running") break;
    } while (Date.now() < deadline);
    assert.equal(after.phase, "succeeded", JSON.stringify(after.logs));
    assert.equal(after.evaluation?.source_sha256, before.evaluation.source_sha256);
    assert.equal(after.renderVerified, true, "New video receipt must match evaluated policy and settings");
    results.push({ experiment, run: run.runName, sourceSha256: before.evaluation.source_sha256, backupHashes, videoSha256: after.renderEvidenceId, videoByteIdentical: after.renderEvidenceId === backupHashes.video, renderVerified: after.renderVerified });
    console.log(`${experiment}: provenance matched`);
  }
} finally {
  await writeFile(path.join(directory, "verification.json"), JSON.stringify({
    passed: results.length === verifiedExperiments.length &&
      skips.length === unverifiedExperiments.size,
    results,
    skips,
  }, null, 2) + "\n");
}
