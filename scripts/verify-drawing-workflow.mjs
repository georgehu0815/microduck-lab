import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve("docs/drawing-case/evidence");
const report = { checkedAt: new Date().toISOString(), passed: false, jobs: [] };
await mkdir(output, { recursive: true });

async function operation(action, recipe) {
  const response = await fetch(`${studio}/api/rlx`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ action, recipe }),
  });
  const launch = await response.json();
  assert.equal(response.status, 202, JSON.stringify(launch));
  const deadline = Date.now() + 900_000;
  let snapshot;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    const status = await fetch(`${studio}/api/rlx?experiment=drawing&run=${recipe.runName}`);
    assert.equal(status.status, 200);
    snapshot = await status.json();
    if (snapshot.phase !== "running") break;
  }
  assert.equal(snapshot?.phase, "succeeded", JSON.stringify({ phase: snapshot?.phase, logs: snapshot?.logs?.slice(-3) }));
  assert.equal(snapshot.exitCode, 0);
  const entry = { action, recipe: launch.recipe, phase: snapshot.phase, exitCode: snapshot.exitCode,
    artifacts: snapshot.artifacts, renderVerified: snapshot.renderVerified,
    trainingSteps: snapshot.trainingSteps, rewardHistoryPoints: snapshot.rewardHistory.length,
    evaluationPresent: Boolean(snapshot.evaluation), evaluationPassed: snapshot.evaluation?.passed ?? null };
  report.jobs.push(entry);
  await writeFile(path.join(output, "workflow.json"), JSON.stringify(report, null, 2));
  console.log(`${recipe.runName}: ${action} succeeded`);
  return snapshot;
}

try {
  for (const runName of ["drawing-pilot-01", "drawing-refinement-02"]) {
    const recipe = { experimentId: "drawing", runName, profile: "full", seed: 101,
      maxEpisodeS: 32, renderSeconds: 32, evalEpisodes: 8 };
    const evaluated = await operation("eval", recipe);
    assert.equal(evaluated.evaluation?.drawing_assessment.episodes.length, 32);
    assert.equal(evaluated.evaluation?.drawing_assessment.unassisted, true);
    assert.equal(evaluated.evaluation?.drawing_assessment.passed, false);
    assert.ok(evaluated.evaluation?.bc_baseline, "matched BC comparison missing");
    const rendered = await operation("render", recipe);
    assert.equal(rendered.renderVerified, true, "source-bound render verification missing");
    const video = await fetch(`${studio}/api/rlx/artifact?experiment=drawing&run=${runName}&kind=video`, {
      headers: { Range: "bytes=0-1023" },
    });
    assert.equal(video.status, 206);
    assert.equal((await video.arrayBuffer()).byteLength, 1024);
  }
  if (process.argv.includes("--smoke")) {
    const recipe = { experimentId: "drawing", runName: process.env.SMOKE_RUN || "drawing-api-smoke-20260911",
      profile: "smoke", totalTimesteps: 768, seed: 11 };
    const trained = await operation("train", recipe);
    assert.equal(trained.trainingSteps, 768);
    assert.ok(trained.rewardHistory.length >= 3, "live PPO progress not received");
    assert.equal(trained.artifacts.checkpoint, true);
    assert.equal(trained.artifacts.onnx, true);
  }
  report.passed = true;
} finally {
  await writeFile(path.join(output, "workflow.json"), JSON.stringify(report, null, 2));
}
console.log(JSON.stringify({ passed: report.passed, output }));
