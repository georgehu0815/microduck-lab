import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const runName = process.env.BACKFLIP_RUN || "backflip-e2e-20260908-v5";
const protocol = "spotter-launch-landing-stand-v2";
const standPolicySha256 =
  "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b";
const output = path.resolve(process.env.STUDIO_EVIDENCE_DIR || "../rlx/artifacts/backflip-browser");

await mkdir(output, { recursive: true });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1600, height: 1100 },
  deviceScaleFactor: 2,
  reducedMotion: "reduce",
});
const page = await context.newPage();
page.setDefaultTimeout(30_000);

const receipt = {
  passed: false,
  checkedAt: new Date().toISOString(),
  baseUrl,
  runName,
  postRequestsBlocked: 0,
  pageErrors: [],
  checks: {},
};

page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
await page.route("**/api/rlx**", async (route) => {
  if (route.request().method() !== "POST") {
    await route.continue();
    return;
  }
  receipt.postRequestsBlocked += 1;
  await route.fulfill({
    status: 409,
    contentType: "application/json",
    body: '{"error":"read-only backflip browser audit"}',
  });
});

async function layout(locator) {
  return locator.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
  }));
}

try {
  await page.goto(`${baseUrl}/#experiments`, { waitUntil: "domcontentloaded" });

  const response = await context.request.get(
    `${baseUrl}/api/rlx?experiment=backflip&run=${runName}`,
  );
  assert.equal(response.status(), 200);
  const snapshot = await response.json();
  const evaluation = snapshot.evaluation;
  const options = evaluation.evaluation.environment.recipe_options;
  const episodes = evaluation.backflip_assessment.episodes;

  assert.equal(snapshot.activeJob, null);
  assert.equal(snapshot.trainingSteps, 401_408);
  assert.equal(snapshot.trainingTotal, 401_408);
  assert.equal(snapshot.renderVerified, true);
  assert.ok(snapshot.renderEvidenceId);
  assert.equal(evaluation.evaluation_mode, "skill");
  assert.equal(evaluation.skill_status, "passed");
  assert.equal(evaluation.pipeline_passed, true);
  assert.notEqual(evaluation.evaluation_settings_match, false);
  assert.equal(evaluation.backflip_assessment.passed, true);
  assert.equal(evaluation.backflip_assessment.passed_episodes, 16);
  assert.equal(evaluation.backflip_assessment.completed_episodes, 16);
  assert.equal(options.backflip_protocol_version, protocol);
  assert.equal(options.stand_policy_sha256, standPolicySha256);
  assert.equal(episodes.length, 16);
  assert.ok(episodes.every((episode) => episode.assist_after_release_steps === 0));
  assert.ok(episodes.every((episode) => episode.landing_policy_steps >= 10));
  assert.ok(episodes.every((episode) => episode.pre_handoff_stable_steps >= 10));
  assert.ok(episodes.every((episode) => episode.rotation_at_handoff >= 5.8));
  assert.ok(episodes.every((episode) => episode.stand_hold_seconds >= 2));

  receipt.checks.api = {
    normalizedBudget: snapshot.trainingTotal,
    renderVerified: snapshot.renderVerified,
    renderEvidenceId: snapshot.renderEvidenceId,
    evaluationMode: evaluation.evaluation_mode,
    skillStatus: evaluation.skill_status,
    pipelinePassed: evaluation.pipeline_passed,
    evaluationSettingsMatch: evaluation.evaluation_settings_match ?? "implicit-match",
    protocol: options.backflip_protocol_version,
    standPolicySha256: options.stand_policy_sha256,
    sourceSha256: evaluation.source_sha256,
    passedEpisodes: evaluation.backflip_assessment.passed_episodes,
    completedEpisodes: evaluation.backflip_assessment.completed_episodes,
    landingPolicySteps: {
      min: Math.min(...episodes.map((episode) => episode.landing_policy_steps)),
      max: Math.max(...episodes.map((episode) => episode.landing_policy_steps)),
    },
    standHoldSeconds: {
      min: Math.min(...episodes.map((episode) => episode.stand_hold_seconds)),
      max: Math.max(...episodes.map((episode) => episode.stand_hold_seconds)),
    },
    assistAfterReleaseSteps: {
      min: Math.min(...episodes.map((episode) => episode.assist_after_release_steps)),
      max: Math.max(...episodes.map((episode) => episode.assist_after_release_steps)),
    },
  };

  const rows = page.locator('#experiments [role="radio"]');
  await rows.nth(4).waitFor();
  assert.equal(await rows.count(), 5);
  const backflip = page.getByRole("radio").filter({ hasText: "Backflip showcase" });
  await backflip.getByText(runName, { exact: true }).waitFor();
  await backflip.getByText("TRAINED · PASSED", { exact: true }).waitFor();
  await backflip
    .getByText("Matched evaluation and rollout · simulation only", { exact: true })
    .waitFor();
  await backflip
    .getByText(
      /spotter-assisted launch → PPO landing → pretrained stand-policy handoff/,
    )
    .first()
    .waitFor();
  receipt.checks.catalog = {
    cards: await rows.count(),
    title: "Backflip showcase",
    runName,
    trainedPassed: true,
    matchedEvaluationAndRollout: true,
  };
  await page.locator("#experiments").screenshot({
    path: path.join(output, "backflip-catalog-desktop.png"),
    animations: "disabled",
    timeout: 60_000,
  });

  await backflip.getByRole("button", {
    name: "Show Backflip showcase guidance",
  }).click();
  const dialog = page.getByRole("dialog");
  for (const pattern of [
    /spotter-launch-landing-stand-v2/,
    /exactly transferring the pretrained stand actor and observation normalizer/,
    /16 pretrained-teacher landing episodes of 600 steps \(9,600 calibration steps\)/,
    /fixes the discounted-return RMS/,
    /10 critic-only calibration epochs/,
    /actor remains unchanged/,
    /calibrated reward normalization is frozen/,
    /fresh optimizer/,
    /does not by itself establish positive PPO improvement/,
    /return regresses 12\.63% versus initialization/,
    /5\.3 rad threshold/,
    /5\.376–5\.380 rad/,
    /3\.5 and 6\.5 rad\/s/,
    /at least 5\.8 rad of credited rotation before handoff/,
    /10 consecutive stable PPO landing steps/,
    /no spotter, stand override, or body support/,
    /stand-policy phase cannot finish credited rotation/,
    /landing ONNX alone is not the complete backflip controller/i,
  ]) {
    await dialog.getByText(pattern).waitFor();
  }
  receipt.checks.guidance = {
    protocol,
    exactActorNormalizerTransfer: true,
    calibrationSteps: 9600,
    criticOnlyCalibrationEpochs: 10,
    actorUnchangedDuringCalibration: true,
    rewardNormalizationFrozenAfterCalibration: true,
    freshPpoOptimizer: true,
    positivePpoImprovementClaimed: false,
    launchReleaseThresholdRad: 5.3,
    measuredReleaseRotationRad: [5.376, 5.38],
    launchReleaseRateRadS: [3.5, 6.5],
    minRotationBeforeHandoffRad: 5.8,
    consecutiveStableLandingSteps: 10,
    bodySupportForbidden: true,
    standCannotFinishRotation: true,
    landingOnnxBoundaryExplicit: true,
  };
  await dialog.screenshot({
    path: path.join(output, "backflip-guidance-desktop.png"),
    animations: "disabled",
    timeout: 60_000,
  });
  await dialog.getByRole("button", { name: "Close recipe guidance" }).click();

  await backflip.click();
  const savedRuns = page.getByLabel("Saved runs", { exact: true });
  await savedRuns.selectOption(runName);
  await page.waitForFunction(
    (expected) =>
      document.querySelector('#recipe input[pattern]')?.value === expected,
    runName,
  );

  const evaluationPanel = page.locator("#evaluation");
  await evaluationPanel
    .getByRole("heading", { name: "Skill verified in simulation" })
    .waitFor();
  await evaluationPanel.getByText(`16 / 16 episodes passed · ${runName}`).waitFor();
  await evaluationPanel.getByText("Saved evaluation scope: skill", {
    exact: false,
  }).waitFor();
  await evaluationPanel.getByText("Accepted", { exact: true }).waitFor();
  await evaluationPanel.getByText("Matched", { exact: true }).waitFor();
  await evaluationPanel
    .getByText(/Policy, external dependencies, and media hashes match/)
    .waitFor();
  for (const metric of [
    "rotation rad",
    "landing policy steps",
    "stand hold seconds",
    "min stand height m",
    "stand upright fraction",
    "assist after release steps",
    "assist force n",
    "assist torque nm",
  ]) {
    await evaluationPanel.getByText(metric, { exact: false }).first().waitFor();
  }
  receipt.checks.evaluationUi = {
    skillVerified: true,
    episodeResult: "16 / 16",
    accepted: true,
    renderProvenance: "Matched",
    evidenceMetrics: [
      "rotation_rad",
      "landing_policy_steps",
      "stand_hold_seconds",
      "min_stand_height_m",
      "stand_upright_fraction",
      "assist_after_release_steps",
      "assist_force_n",
      "assist_torque_nm",
    ],
  };
  await evaluationPanel.screenshot({
    path: path.join(output, "backflip-evaluation-render-desktop.png"),
    animations: "disabled",
    timeout: 60_000,
  });

  await page.setViewportSize({ width: 390, height: 844 });
  const catalogLayout = await layout(page.locator("#experiments"));
  const evaluationLayout = await layout(evaluationPanel);
  const documentLayout = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  assert.ok(catalogLayout.scrollWidth <= catalogLayout.clientWidth + 1);
  assert.ok(evaluationLayout.scrollWidth <= evaluationLayout.clientWidth + 1);
  assert.ok(documentLayout.scrollWidth <= documentLayout.clientWidth + 1);
  receipt.checks.mobileLayout = {
    viewport: { width: 390, height: 844 },
    catalog: catalogLayout,
    evaluation: evaluationLayout,
    document: documentLayout,
  };
  await page.locator("#experiments").screenshot({
    path: path.join(output, "backflip-catalog-mobile.png"),
    animations: "disabled",
    timeout: 60_000,
  });
  await evaluationPanel.screenshot({
    path: path.join(output, "backflip-evaluation-render-mobile.png"),
    animations: "disabled",
    timeout: 60_000,
  });

  await backflip.getByRole("button", {
    name: "Show Backflip showcase guidance",
  }).click();
  await dialog.getByRole("heading", { name: "Backflip showcase" }).waitFor();
  const guidanceLayout = await layout(dialog);
  assert.ok(guidanceLayout.scrollWidth <= guidanceLayout.clientWidth + 1);
  receipt.checks.mobileLayout.guidance = guidanceLayout;
  await dialog.screenshot({
    path: path.join(output, "backflip-guidance-mobile.png"),
    animations: "disabled",
    timeout: 60_000,
  });

  assert.deepEqual(receipt.pageErrors, []);
  assert.equal(receipt.postRequestsBlocked, 0);
  receipt.passed = true;
} catch (error) {
  receipt.error = error instanceof Error ? error.stack : String(error);
  await page
    .screenshot({ path: path.join(output, "failure.png"), fullPage: true })
    .catch(() => {});
  await writeFile(path.join(output, "failure.html"), await page.content());
  throw error;
} finally {
  await writeFile(
    path.join(output, "verification.json"),
    `${JSON.stringify(receipt, null, 2)}\n`,
  );
  await context.close();
  await browser.close();
}

console.log(JSON.stringify({ passed: receipt.passed, output, checks: receipt.checks }));
