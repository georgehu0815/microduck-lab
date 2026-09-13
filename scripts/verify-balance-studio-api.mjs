import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const base = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve("docs/bridge-showcase/evidence/api");
await mkdir(output, { recursive: true });
async function request(url, body) {
  const response = await fetch(`${base}${url}`, {
    ...(body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
    signal: AbortSignal.timeout(60000),
  });
  const payload = await response.json();
  assert.ok(response.ok, JSON.stringify(payload));
  return payload;
}
const results = [];
const smokeOnly = process.argv.includes("--smoke-train");
if (smokeOnly) {
  for (const experimentId of ["basketball", "bridge"]) {
    const runName = `${experimentId}-api-smoke-${Date.now()}`;
    await request("/api/rlx", { action: "train", recipe: {
      experimentId, runName, profile: "smoke", totalTimesteps: 4,
      numEnvs: 2, numSteps: 2, numMinibatches: 1,
    } });
    let state;
    const deadline = Date.now() + 180000;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      state = await request(`/api/rlx?experiment=${experimentId}&run=${runName}`);
    } while (state.phase === "running" && Date.now() < deadline);
    assert.equal(state.phase, "succeeded", JSON.stringify(state.logs));
    assert.equal(state.artifacts.onnx, true);
    results.push({ experimentId, runName, phase: state.phase, artifacts: state.artifacts, result: state.result });
    console.log(`${experimentId}: real API training smoke succeeded`);
  }
}
for (const [experimentId, runName, seconds] of smokeOnly ? [] : [
  ["basketball", "basketball-balance-01", 60],
  ["bridge", "bridge-studio-02", 20],
]) {
  const basketball = experimentId === "basketball";
  const recipe = {
    experimentId, runName, profile: "full", totalTimesteps: basketball ? 102400 : 32768,
    numEnvs: 3, numSteps: 128, numMinibatches: basketball ? 1 : 4,
    maxEpisodeS: seconds, evalSteps: seconds * 50, renderSeconds: seconds,
    seed: 101, actuator: basketball ? "bam" : "xml", domainRand: false, obsNoise: false,
    actionDelay: basketball, randomYaw: basketball, initialStd: 0.03,
    rewardWeights: {}, resumeFromCheckpoint: true, freezeObservationNormalization: true,
  };
  let state;
  for (const action of ["eval", "render"]) {
    await request("/api/rlx", { action, recipe });
    const deadline = Date.now() + 300000;
    do {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      state = await request(`/api/rlx?experiment=${experimentId}&run=${runName}`);
    } while (state.phase === "running" && Date.now() < deadline);
    assert.notEqual(state.phase, "running", `${experimentId} ${action} timed out`);
    if (action === "eval") {
      assert.equal(state.evaluation?.pipeline_passed, true, JSON.stringify(state.logs));
      assert.equal(state.evaluation?.passed, false, "These pilot policies must not be promoted to verified skills");
    } else {
      assert.equal(state.phase, "succeeded", JSON.stringify(state.logs));
      assert.equal(state.renderVerified, true, "render must match current evaluation source and settings");
    }
    console.log(`${experimentId} ${action}: ${state.phase}, renderVerified=${state.renderVerified}`);
  }
  results.push({ experimentId, runName, evaluation: state.evaluation,
    renderVerified: state.renderVerified, renderEvidenceId: state.renderEvidenceId,
    artifacts: state.artifacts });
}
await writeFile(path.join(output, smokeOnly ? "smoke-verification.json" : "verification.json"), JSON.stringify({ passed: true, results }, null, 2));
console.log(JSON.stringify({ passed: true, output }));
