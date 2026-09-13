import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve("docs/drawing-case/brush/evidence");
const runName = process.env.BRUSH_RUN || "brush-color-v2-03";
const recipe = { experimentId: "drawing", drawingTool: "brush", runName, profile: "full",
  seed: 10001, maxEpisodeS: 120, renderSeconds: 120, evalEpisodes: 4 };
const report = { checkedAt: new Date().toISOString(), runName, passed: false, jobs: [] };
await mkdir(output, { recursive: true });

async function operation(action) {
  const response = await fetch(`${studio}/api/rlx`, { method: "POST",
    headers: { "content-type": "application/json" }, body: JSON.stringify({ action, recipe }) });
  const launch = await response.json();
  assert.equal(response.status, 202, JSON.stringify(launch));
  const deadline = Date.now() + 1_200_000;
  let snapshot;
  while (Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 2000));
    const status = await fetch(`${studio}/api/rlx?experiment=drawing&run=${runName}`);
    snapshot = await status.json();
    if (snapshot.phase !== "running") break;
  }
  assert.equal(snapshot?.phase, "succeeded", JSON.stringify({ phase: snapshot?.phase, logs: snapshot?.logs?.slice(-5) }));
  assert.equal(snapshot.exitCode, 0);
  report.jobs.push({ action, recipe: launch.recipe, exitCode: snapshot.exitCode,
    renderVerified: snapshot.renderVerified, artifacts: snapshot.artifacts,
    evaluationPassed: snapshot.evaluation?.passed });
  await writeFile(path.join(output, "workflow.json"), JSON.stringify(report, null, 2));
  console.log(`${runName}: ${action} succeeded`);
  return snapshot;
}

try {
  const evaluated = await operation("eval");
  assert.equal(evaluated.evaluation?.contract_version, "microduck-brush-v2");
  assert.equal(evaluated.evaluation?.passed, true);
  assert.equal(evaluated.evaluation?.drawing_assessment.episodes.length, 40);
  assert.equal(evaluated.evaluation?.unique_conditions, 10);
  assert.deepEqual(evaluated.evaluation?.provenance_errors, []);
  const rendered = await operation("render");
  assert.equal(rendered.renderVerified, true);
  const video = await fetch(`${studio}/api/rlx/artifact?experiment=drawing&run=${runName}&kind=video`, { headers: { Range: "bytes=0-1023" } });
  assert.equal(video.status, 206);
  assert.equal((await video.arrayBuffer()).byteLength, 1024);
  report.passed = true;
} finally {
  await writeFile(path.join(output, "workflow.json"), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, output }));
