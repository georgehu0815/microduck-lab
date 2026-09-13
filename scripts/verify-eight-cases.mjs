import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(
  process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json"
)("playwright");

const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const lab = process.env.LAB_ADDRESS || "127.0.0.1:8791";
const output = path.resolve(
  process.env.EVIDENCE_DIR || "docs/drawing-case/evidence"
);
const skipRender = process.argv.includes("--skip-render");
const studioOnly = process.argv.includes("--studio-only");
const cases = [
  {
    id: "dance",
    title: "Dance imitation",
    run: "dance-e2e-20260907-low-noise",
    passed: true,
  },
  {
    id: "swing",
    title: "Self-pumped swing",
    run: "swing-e2e-20260907-v3",
    passed: true,
  },
  {
    id: "running",
    title: "Fast running",
    run: "running-e2e-20260907-v4",
    passed: true,
  },
  {
    id: "stilts",
    title: "Stilt walking",
    run: "stilts-e2e-20260907-v3",
    passed: true,
  },
  {
    id: "backflip",
    title: "Backflip showcase",
    run: "backflip-e2e-20260908-v5",
    passed: true,
  },
  {
    id: "basketball",
    title: "Basketball balancing",
    run: "basketball-balance-01",
    passed: false,
  },
  {
    id: "bridge",
    title: "Suspended bridge",
    run: "bridge-studio-02",
    passed: false,
  },
  {
    id: "drawing",
    title: "Swimming-duck drawing",
    run: "drawing-pilot-01",
    passed: false,
  },
];

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitForJson(url, validate, timeoutMs = 360_000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const value = await response.json();
      if (validate(value)) return value;
      lastError = new Error("response was not ready");
    } catch (error) {
      lastError = error;
    }
    await sleep(2_000);
  }
  throw new Error(
    `Timed out waiting for ${url}: ${lastError?.message || "unknown error"}`
  );
}

function closeEnough(actual, expected, tolerance = 0.002) {
  return Math.abs(Number(actual) - expected) <= tolerance;
}

function drawingReceiptPath() {
  return path.resolve(
    "rlx/runs/studio/drawing/drawing-pilot-01/render/evidence.json"
  );
}

await mkdir(output, { recursive: true });
const readiness = {};
readiness.studio = await waitForJson(
  `${studio}/api/rlx?experiment=drawing&run=drawing-pilot-01`,
  (snapshot) =>
    snapshot?.experimentId === "drawing" &&
    snapshot?.artifacts?.onnx === true &&
    snapshot?.evaluation?.contract_version === "microduck-drawing-v1"
);
readiness.lab = studioOnly
  ? null
  : await waitForJson(
      `http://${lab}/policies`,
      (payload) =>
        Array.isArray(payload?.policies) &&
        payload.policies.some(
          (policy) =>
            policy?.id?.includes("drawing") ||
            policy?.path?.includes("drawing-pilot-01")
        )
    );

const browser = await chromium.launch({
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage({ viewport: { width: 1536, height: 1000 } });
page.setDefaultTimeout(45_000);
await page.addInitScript(() => {
  localStorage.setItem(
    "ducklab.camera",
    JSON.stringify({ p: [0, 5, 1.8], t: [0, 0.3, -2.6] })
  );
});

const errors = [];
const contextMessages = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (/WebGL.*Context Lost/i.test(message.text())) {
    contextMessages.push(message.text());
  }
});

const report = {
  checkedAt: new Date().toISOString(),
  studio,
  lab,
  skipRender,
  studioOnly,
  passed: false,
  errors,
  contextMessages,
  readiness: {
    labPolicies: readiness.lab?.policies?.length ?? null,
    drawingRenderVerified: readiness.studio.renderVerified,
  },
  checks: {},
};

try {
  await page.goto(studio, { waitUntil: "domcontentloaded" });
  await page.addStyleTag({
    content: "html, body, * { scroll-behavior: auto !important; }",
  });
  const rows = page.locator('#experiments [role="radio"]');
  await rows.nth(7).waitFor();
  assert.equal(await rows.count(), 8);
  report.checks.cards = await rows.allTextContents();
  for (const scenario of cases) {
    assert.ok(
      report.checks.cards.some((text) => text.includes(scenario.title)),
      `${scenario.id}: experiment card missing`
    );
  }
  await page
    .locator("#experiments")
    .screenshot({ path: path.join(output, "eight-cases-desktop.png") });

  report.checks.scenarios = [];
  for (const scenario of cases) {
    const row = rows.filter({ hasText: scenario.title }).first();
    await row.click();
    assert.equal(await row.getAttribute("aria-checked"), "true");
    await row
      .getByRole("button", { name: `Show ${scenario.title} guidance` })
      .click();
    await page.getByRole("dialog").waitFor();
    await page.getByRole("button", { name: "Close recipe guidance" }).click();
    await page
      .getByLabel("Saved runs", { exact: true })
      .selectOption(scenario.run);
    await page.waitForFunction(
      (runName) =>
        document.querySelector('input[aria-label="Run name"]')?.value ===
          runName ||
        [...document.querySelectorAll("label")].some(
          (label) =>
            label.textContent?.trim() === "Run name" &&
            (label.control?.value === runName ||
              label.querySelector("input")?.value === runName)
        ),
      scenario.run
    );
    assert.equal(
      await page.getByLabel("Run name", { exact: true }).inputValue(),
      scenario.run
    );

    const response = await page.request.get(
      `${studio}/api/rlx?experiment=${scenario.id}&run=${scenario.run}`
    );
    assert.equal(response.status(), 200);
    const snapshot = await response.json();
    assert.equal(snapshot.artifacts.onnx, true, `${scenario.id}: ONNX missing`);
    assert.ok(snapshot.evaluation, `${scenario.id}: evaluation missing`);
    assert.equal(
      snapshot.evaluation.passed,
      scenario.passed,
      `${scenario.id}: historical verdict changed`
    );
    if (scenario.id !== "drawing" || !skipRender) {
      assert.equal(
        snapshot.renderVerified,
        true,
        `${scenario.id}: source-bound render missing`
      );
      assert.ok(
        snapshot.renderEvidenceId,
        `${scenario.id}: render evidence id missing`
      );
      assert.equal(
        snapshot.artifacts.renderVideo,
        true,
        `${scenario.id}: render video missing`
      );
      assert.equal(
        snapshot.artifacts.renderSheet,
        true,
        `${scenario.id}: render sheet missing`
      );
    }
    if (scenario.id === "drawing") {
      assert.equal(snapshot.evaluation.contract_version, "microduck-drawing-v1");
      assert.equal(snapshot.evaluation.drawing_assessment?.unassisted, true);
      assert.equal(snapshot.evaluation.drawing_assessment?.passed, false);
      assert.equal(
        snapshot.evaluation.evaluation?.environment?.max_episode_s,
        32
      );
      assert.equal(
        snapshot.evaluation.evaluation?.environment?.assistance,
        0
      );
      assert.equal(
        snapshot.evaluation.evaluation?.environment?.recipe,
        "drawing"
      );
      assert.deepEqual(
        snapshot.evaluation.evaluation?.environment?.recipe_options,
        {}
      );
      report.checks.drawingSnapshot = snapshot;
    }
    report.checks.scenarios.push({
      ...scenario,
      artifacts: snapshot.artifacts,
      taskPassed: snapshot.evaluation.passed,
      renderVerified: snapshot.renderVerified,
      renderEvidenceId: snapshot.renderEvidenceId,
    });
  }

  await page.goto(studio, { waitUntil: "domcontentloaded" });
  const drawingRow = page.locator('#experiments [role="radio"]').filter({ hasText: "Swimming-duck drawing" }).first();
  await drawingRow.waitFor();
  if (!skipRender || report.checks.drawingSnapshot.renderVerified) {
    await drawingRow.locator("video").waitFor();
  }
  const drawingPreviewText = await drawingRow.textContent();
  if (skipRender && !report.checks.drawingSnapshot.renderVerified) {
    assert.match(drawingPreviewText, /No verified rollout yet/);
    assert.doesNotMatch(drawingPreviewText, /TRAINED · PASSED/);
    report.checks.drawingPreview = {
      state: "no-render",
      label: "No verified rollout yet",
    };
  } else {
    assert.match(drawingPreviewText, /DIAGNOSTIC · DRAWING NOT ACCEPTED/);
    assert.doesNotMatch(drawingPreviewText, /TRAINED · PASSED/);
    const video = drawingRow.locator("video");
    await video.waitFor();
    await video.scrollIntoViewIfNeeded();
    await page.waitForFunction(
      (element) => element.readyState >= 2,
      await video.elementHandle()
    );
    const media = await video.evaluate(async (element) => {
      element.muted = true;
      await element.play();
      return {
        duration: element.duration,
        paused: element.paused,
        readyState: element.readyState,
      };
    });
    assert.ok(Number.isFinite(media.duration) && media.duration > 0.1);
    assert.equal(media.paused, false);
    const before = await video.evaluate((element) => element.currentTime);
    await page.waitForFunction(
      (start) => {
        const current = document.querySelector('video[aria-label^="Drawing trained rollout:"]');
        return current && !current.paused && Math.abs(current.currentTime - start) > 0.05;
      },
      before,
      { timeout: 10_000 }
    );
    const after = await video.evaluate((element) => element.currentTime);
    assert.ok(Math.abs(after - before) > 0.05, `drawing preview did not advance: ${before} -> ${after}`);
    report.checks.drawingPreview = {
      state: "diagnostic-video",
      label: "DIAGNOSTIC · DRAWING NOT ACCEPTED",
      before,
      after,
    };
  }

  if (!skipRender) {
    const receipt = JSON.parse(await readFile(drawingReceiptPath(), "utf8"));
    const evaluation = report.checks.drawingSnapshot.evaluation;
    assert.equal(receipt.version, 1);
    assert.equal(receipt.sourceSha256, evaluation.source_sha256);
    assert.equal(receipt.source, evaluation.source);
    assert.equal(
      receipt.sourceFilesSha256?.[receipt.source],
      evaluation.source_sha256
    );
    assert.ok(receipt.video?.size > 0);
    assert.ok(receipt.sheet?.size > 0);
    assert.equal(
      receipt.video.sha256,
      report.checks.drawingSnapshot.renderEvidenceId
    );
    report.checks.drawingReceipt = receipt;
  }

  if (studioOnly) {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    const dimensions = await page.evaluate(() => ({
      width: innerWidth,
      scroll: document.documentElement.scrollWidth,
    }));
    assert.ok(dimensions.scroll <= dimensions.width + 1, JSON.stringify(dimensions));
    report.checks.studioMobile = dimensions;
    await page
      .locator("#experiments")
      .screenshot({ path: path.join(output, "eight-cases-mobile.png") });
  } else {
    await page.goto(`${studio}/?lab=${lab}`, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => document.querySelectorAll("[data-duck-id]").length === 8,
    null,
    { timeout: 180_000 }
  );
  const canvas = page.locator("canvas").first();
  await canvas.scrollIntoViewIfNeeded();
  await page.waitForTimeout(1_000);

  report.checks.live = await page.evaluate(async (address) => {
    const frames = await new Promise((resolve, reject) => {
      const socket = new WebSocket(`ws://${address}/ws`);
      const received = [];
      const timeout = setTimeout(() => {
        socket.close();
        reject(new Error("no live frames"));
      }, 60_000);
      socket.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("lab websocket failed"));
      };
      socket.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.ducks?.length !== 8) return;
        received.push(frame.ducks);
        if (received.length >= 40) {
          clearTimeout(timeout);
          socket.close();
          resolve(received);
        }
      };
    });
    const first = frames[0];
    const last = frames.at(-1);
    const scenes = await Promise.all(
      last.map(async (duck) => {
        const response = await fetch(
          `http://${address}/scene?duck=${encodeURIComponent(
            duck.id
          )}&version=${encodeURIComponent(duck.sceneKey || "")}`
        );
        if (!response.ok) throw new Error(`scene status ${response.status}`);
        const scene = await response.json();
        return {
          duck: duck.id,
          name: duck.name,
          policy: duck.policy,
          bodies: scene.bodies,
          tendons: scene.tendons?.length || 0,
          primitives: scene.primitives || [],
          ballVisible: scene.geoms.some(
            (geom) => scene.bodies[geom.body] === "basketball"
          ),
          advanced:
            first.find((candidate) => candidate.id === duck.id)?.step !==
            duck.step,
          drawing: duck.drawing || null,
        };
      })
    );
    return { first, last, scenes };
  }, lab);

  assert.equal(report.checks.live.scenes.length, 8);
  for (const scenario of cases) {
    assert.ok(
      report.checks.live.scenes.some(
        (scene) =>
          scene.name === scenario.id ||
          scene.name.includes(scenario.run) ||
          scene.policy?.includes(scenario.run)
      ),
      `${scenario.id}: live policy missing`
    );
  }
  assert.ok(report.checks.live.scenes.every((scene) => scene.advanced));
  assert.ok(
    report.checks.live.scenes.some(
      (scene) => scene.bodies.includes("basketball") && scene.ballVisible
    )
  );
  assert.ok(
    report.checks.live.scenes.some(
      (scene) =>
        scene.bodies.includes("bridge_plank") &&
        scene.tendons === 4 &&
        scene.primitives.length >= 9
    )
  );

  const drawingScene = report.checks.live.scenes.find(
    (scene) =>
      scene.policy?.includes("drawing-pilot-01") ||
      scene.name.includes("drawing")
  );
  assert.ok(drawingScene, "drawing live scene missing");
  assert.ok(drawingScene.bodies.includes("drawing_pencil"));
  assert.ok(drawingScene.bodies.includes("drawing_lower_beak"));
  const pencilBody = drawingScene.bodies.indexOf("drawing_pencil");
  assert.ok(
    drawingScene.primitives.some((primitive) => primitive.body === pencilBody),
    "drawing pencil primitive missing"
  );
  assert.ok(
    drawingScene.primitives.some(
      (primitive) =>
        primitive.type === "box" &&
        closeEnough(primitive.pos?.[0], 0.161) &&
        closeEnough(primitive.pos?.[1], 0) &&
        closeEnough(primitive.pos?.[2], 0.2245) &&
        closeEnough(primitive.size?.[0], 0.001, 0.0002) &&
        closeEnough(primitive.size?.[1], 0.095) &&
        closeEnough(primitive.size?.[2], 0.073)
    ),
    "drawing canvas primitive missing"
  );
  assert.equal(
    report.checks.live.scenes.filter((scene) => scene.drawing !== null).length,
    1,
    "drawing payload leaked to another duck"
  );
  assert.equal(drawingScene.drawing?.contract, "microduck-drawing-v1");
  assert.equal(drawingScene.drawing?.assistance, 0);

  report.checks.renderedDucks = await page
    .locator("[data-duck-id]")
    .evaluateAll((labels) =>
      labels.map((label) => ({
        id: label.dataset.duckId,
        name: label.textContent,
      }))
    );
  assert.equal(report.checks.renderedDucks.length, 8);
  report.checks.webglDesktop = await page.evaluate(() =>
    Array.from(document.querySelectorAll("canvas"), (element) => {
      const context =
        element.getContext("webgl2") || element.getContext("webgl");
      return context
        ? {
            lost: context.isContextLost(),
            width: element.width,
            height: element.height,
          }
        : null;
    }).filter(Boolean)
  );
  assert.ok(report.checks.webglDesktop.length > 0);
  assert.ok(
    report.checks.webglDesktop.every(
      (context) =>
        !context.lost && context.width > 0 && context.height > 0
    )
  );
  await canvas.screenshot({
    path: path.join(output, "eight-live-scenes-desktop.png"),
  });

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(500);
  report.checks.mobile = await page.evaluate(() => ({
    width: innerWidth,
    scroll: document.documentElement.scrollWidth,
    mountedDucks: document.querySelectorAll("[data-duck-id]").length,
  }));
  assert.ok(
    report.checks.mobile.scroll <= report.checks.mobile.width + 1,
    JSON.stringify(report.checks.mobile)
  );
  assert.equal(report.checks.mobile.mountedDucks, 8);
  report.checks.webglMobile = await page.evaluate(() =>
    Array.from(document.querySelectorAll("canvas"), (element) => {
      const context =
        element.getContext("webgl2") || element.getContext("webgl");
      return context
        ? {
            lost: context.isContextLost(),
            width: element.width,
            height: element.height,
          }
        : null;
    }).filter(Boolean)
  );
  assert.ok(
    report.checks.webglMobile.every(
      (context) =>
        !context.lost && context.width > 0 && context.height > 0
    )
  );
  await canvas.screenshot({
    path: path.join(output, "eight-live-scenes-mobile.png"),
  });

  report.checks.drawingEpisode = await page.evaluate(async (address) => {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(`ws://${address}/ws`);
      const timeout = setTimeout(() => {
        socket.close();
        reject(new Error("drawing reset did not clear and regenerate contact ink"));
      }, 90_000);
      let resetObserved = false;
      let cleared = false;
      let maxStep = -1;
      let payload = null;
      socket.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("drawing reset websocket failed"));
      };
      socket.onopen = () => socket.send(JSON.stringify({ reset: true }));
      socket.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (frame.ducks?.length !== 8) return;
        const drawing = frame.ducks.find(
          (duck) =>
            duck.policy?.includes("drawing-pilot-01") ||
            duck.name?.includes("drawing")
        );
        if (!drawing) return;
        const points = drawing.drawing?.points || [];
        if (!resetObserved) {
          if (drawing.step <= 10) resetObserved = true;
          else return;
        }
        if (drawing.step <= 10 && points.length <= drawing.step) cleared = true;
        if (points.length > 0) payload = drawing.drawing;
        if (drawing.step + 25 < maxStep) {
          clearTimeout(timeout);
          socket.close();
          reject(
            new Error(
              `drawing unexpectedly reset before ink verification: ${maxStep} -> ${drawing.step}`
            )
          );
          return;
        }
        maxStep = Math.max(maxStep, drawing.step);
        if (drawing.step > 50 && points.length > 10) {
          clearTimeout(timeout);
          socket.close();
          resolve({
            resetObserved,
            cleared,
            maxStep,
            payload,
          });
        }
      };
    });
  }, lab);
  assert.equal(report.checks.drawingEpisode.resetObserved, true);
  assert.equal(report.checks.drawingEpisode.cleared, true);
  const drawingPayload = report.checks.drawingEpisode.payload;
  assert.equal(drawingPayload?.contract, "microduck-drawing-v1");
  assert.equal(drawingPayload?.assistance, 0);
  assert.ok(drawingPayload.points.length > 0, "drawing emitted no contact ink");
  assert.ok(drawingPayload.points.length <= 6000);
  assert.ok(
    drawingPayload.points.every(
      (point) =>
        Array.isArray(point) &&
        point.length === 4 &&
        point.every(Number.isFinite) &&
        point[0] >= 0.155 &&
        point[0] <= 0.165 &&
        Number.isInteger(point[3]) &&
        point[3] > 0
    ),
    "drawing payload was not finite world-space contact ink"
  );
  }

  assert.deepEqual(errors, []);
  assert.deepEqual(contextMessages, []);
  report.passed = true;
} catch (error) {
  report.failure = {
    name: error?.name || "Error",
    message: error?.message || String(error),
    stack: error?.stack || null,
  };
  throw error;
} finally {
  await writeFile(
    path.join(output, "verification.json"),
    `${JSON.stringify(report, null, 2)}\n`
  );
  await browser.close();
}

console.log(JSON.stringify({ passed: true, output, skipRender }));
