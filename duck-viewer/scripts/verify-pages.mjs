import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || import.meta.url)("playwright");
const root = path.resolve(import.meta.dirname, "..");
const output = path.join(root, "out");
const evidence = path.resolve(root, "../.omx/artifacts/github-pages");
const prefix = "/microduck-lab/";
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".txt": "text/plain", ".mp4": "video/mp4", ".pdf": "application/pdf", ".png": "image/png", ".svg": "image/svg+xml", ".woff2": "font/woff2" };
let server;
let browser;
const report = { success: false, url: "", assets: [], pageErrors: [], failedResponses: [], backendRequests: [], screenshots: [], latestCases: [], studioPreviews: [], contactChecks: 0, hashesVerified: 0 };
await mkdir(evidence, { recursive: true });

try {
  let site = process.argv[2];
  if (!site) {
    server = createServer(async (request, response) => {
      try {
        const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
        if (!pathname.startsWith(prefix)) { response.writeHead(404).end(); return; }
        let filename = path.resolve(output, pathname.slice(prefix.length));
        if (filename !== output && !filename.startsWith(`${output}${path.sep}`)) { response.writeHead(403).end(); return; }
        if ((await stat(filename)).isDirectory()) filename = path.join(filename, "index.html");
        const size = (await stat(filename)).size;
        const range = request.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
        const start = range ? Number(range[1]) : 0;
        const end = range?.[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
        if (range && (start > end || start >= size)) { response.writeHead(416).end(); return; }
        response.writeHead(range ? 206 : 200, {
          "content-type": mime[path.extname(filename)] || "application/octet-stream",
          "accept-ranges": "bytes",
          "content-length": String(Math.max(0, end - start + 1)),
          ...(range ? { "content-range": `bytes ${start}-${end}/${size}` } : {}),
        });
        if (request.method === "HEAD" || !size) response.end();
        else createReadStream(filename, { start, end }).pipe(response);
      } catch { response.writeHead(404).end(); }
    });
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    site = `http://127.0.0.1:${server.address().port}${prefix}`;
  }
  site = site.endsWith("/") ? site : `${site}/`;
  report.url = site;
  const arm = JSON.parse(await readFile(path.join(root, "public/static/arm-videos/catalog.json"), "utf8"));
  const studio = JSON.parse(await readFile(path.join(root, "public/static/studio-runs/catalog.json"), "utf8"));
  const assets = new Set(["static/arm-videos/catalog.json", "static/studio-runs/catalog.json", "guides/microduck-studio-experiments-guide.pdf", "guides/microduck-studio-dance-guide.pdf"]);
  for (const video of arm.videos) {
    assets.add(video.videoUrl.replace(/^\//, ""));
    if (video.evidenceUrl) assets.add(video.evidenceUrl.replace(/^\//, ""));
  }
  for (const run of studio.runs) {
    const directory = `static/studio-runs/${encodeURIComponent(run.experimentId)}/${encodeURIComponent(run.runName)}`;
    for (const file of ["video.mp4", "sheet.png", "evaluation.json"]) assets.add(`${directory}/${file}`);
  }
  for (const asset of assets) {
    const response = await fetch(new URL(asset, site), { method: "HEAD" });
    assert.equal(response.status, 200, asset);
    assert.ok(Number(response.headers.get("content-length")) > 0, `${asset} is not empty`);
    report.assets.push(asset);
  }
  const manifest = JSON.parse(await readFile(path.join(root, "public/static/manifest.json"), "utf8"));
  const publishedManifest = await (await fetch(new URL("static/manifest.json", site))).json();
  assert.equal(publishedManifest.generatedAt, manifest.generatedAt, "The new media snapshot is published");
  report.mediaGeneratedAt = manifest.generatedAt;
  for (const file of manifest.files) {
    const response = await fetch(new URL(file.path.replace(/^\//, ""), site));
    assert.equal(response.status, 200, file.path);
    const bytes = Buffer.from(await response.arrayBuffer());
    assert.equal(bytes.length, file.bytes, file.path);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), file.sha256, file.path);
    report.hashesVerified += 1;
  }
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: "reduce" });
  const page = await context.newPage();
  page.on("pageerror", (error) => report.pageErrors.push(error.message));
  page.on("response", (response) => { if (response.status() >= 400) report.failedResponses.push(`${response.status()} ${response.url()}`); });
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.includes("/api/") || ["8788", "8812"].includes(url.port)) report.backendRequests.push(request.url());
  });
  page.on("websocket", (socket) => report.backendRequests.push(socket.url()));
  assert.equal((await page.goto(site)).status(), 200);
  await page.getByRole("radio").first().waitFor();
  assert.equal(await page.getByRole("radio").count(), 8);
  await page.waitForFunction(() => [...document.querySelectorAll("video")].some((video) => video.readyState >= 2));
  await verifyContact(page);
  for (const experimentId of [...new Set(studio.runs.map((run) => run.experimentId))]) {
    const candidates = studio.runs.filter((run) => run.experimentId === experimentId && run.video && run.renderVerified && run.taskPassed)
      .sort((left, right) => Date.parse(right.trainedAt ?? right.policyModifiedAt ?? right.modifiedAt) - Date.parse(left.trainedAt ?? left.policyModifiedAt ?? left.modifiedAt));
    const expected = candidates[0];
    if (!expected) continue;
    const preview = page.locator(`video[src*="/studio-runs/${experimentId}/${expected.runName}/"]`).first();
    assert.equal(await preview.count(), 1, `${experimentId} uses its latest successful verified video`);
    await preview.evaluate((video) => new Promise((resolve, reject) => {
      if (video.readyState >= 2) { resolve(); return; }
      video.addEventListener("loadeddata", () => resolve(), { once: true });
      video.addEventListener("error", () => reject(new Error("Preview decode failed")), { once: true });
      setTimeout(() => reject(new Error("Preview load timed out")), 15000);
    }));
    report.studioPreviews.push({ experimentId, runName: expected.runName });
  }
  for (const radio of await page.getByRole("radio").all()) {
    await radio.click();
    assert.equal(await radio.getAttribute("aria-checked"), "true");
  }
  for (const button of await page.getByRole("button", { name: /^Train(?: |$)/ }).all()) assert.equal(await button.isDisabled(), true);
  await page.screenshot({ path: path.join(evidence, "studio-desktop.png"), fullPage: true });
  report.screenshots.push("studio-desktop.png");
  await page.getByRole("link", { name: "Arm videos", exact: true }).first().click();
  await page.getByTestId("arm-video-library").waitFor();
  assert.ok(page.url().startsWith(new URL("arm/", site).href));
  await page.waitForFunction(() => document.querySelectorAll("[data-video-id]").length === 6);
  const latestIds = await page.locator("[data-video-id]").evaluateAll((items) => items.map((item) => item.dataset.videoId));
  const latestCases = new Set();
  for (const id of latestIds) {
    const candidate = arm.videos.find((video) => video.id === id);
    assert.equal(candidate?.provenance, "current");
    assert.equal(candidate?.outcome, "passed");
    assert.equal(candidate?.kind, "episode");
    for (const caseId of candidate.caseIds) {
      const selectedTime = Date.parse(candidate.receiptRecordedAt ?? candidate.evidenceRecordedAt);
      assert.ok(Number.isFinite(selectedTime));
      const newer = arm.videos.some((video) => video.caseIds.includes(caseId) && video.kind === "episode" && video.provenance === "current" && video.outcome === "passed" && Date.parse(video.receiptRecordedAt ?? video.evidenceRecordedAt) > selectedTime);
      assert.equal(newer, false, `${caseId} uses its most recent successful recording`);
      latestCases.add(caseId);
    }
    await page.locator(`[data-video-id="${id}"]`).click();
    await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
    await page.locator("video").evaluate(async (video) => { video.muted = true; await video.play(); });
    await page.waitForFunction(() => document.querySelector("video")?.currentTime > 0);
    await page.locator("video").evaluate((video) => video.pause());
    report.latestCases.push({ id, caseIds: candidate.caseIds, batch: candidate.batch, sha256: candidate.videoHash });
  }
  assert.equal(latestCases.size, 6);
  await verifyContact(page);
  await page.getByTestId("arm-video-view-history").click();
  assert.equal(await page.locator("[data-video-id]").count(), arm.videos.length);
  await page.waitForFunction(() => document.querySelector("video")?.readyState >= 2);
  await page.locator("video").evaluate(async (video) => { video.muted = true; await video.play(); });
  await page.waitForFunction(() => document.querySelector("video")?.currentTime > 0);
  await page.locator("video").evaluate((video) => { video.pause(); video.currentTime = Math.min(2, video.duration / 2); });
  const search = page.getByRole("searchbox");
  await search.fill("no-such-video-verification");
  assert.equal(await page.locator("[data-video-id]").count(), 0);
  await search.fill("");
  await page.getByRole("button", { name: "中文", exact: true }).click();
  await page.getByRole("button", { name: "打开联系选项", exact: true }).click();
  await page.getByRole("dialog", { name: "联系", exact: true }).waitFor();
  assert.equal(await page.getByLabel("邮箱地址", { exact: true }).inputValue(), "bochuxt7@gmail.com");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "English", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await verifyContact(page);
  await page.screenshot({ path: path.join(evidence, "arm-mobile.png"), fullPage: true });
  report.screenshots.push("arm-mobile.png");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.getByRole("link", { name: "Back to Microduck Studio", exact: true }).click();
  await page.getByRole("radio").first().waitFor();
  await verifyContact(page);
  assert.equal(new URL(page.url()).pathname, new URL(site).pathname);
  await page.screenshot({ path: path.join(evidence, "studio-mobile.png"), fullPage: true });
  report.screenshots.push("studio-mobile.png");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  assert.deepEqual(report.pageErrors, []);
  assert.deepEqual(report.failedResponses, []);
  assert.deepEqual(report.backendRequests, []);
  report.success = true;
  report.armVideos = arm.videos.length;
  report.studioRuns = studio.runs.length;
  console.log(JSON.stringify({ success: true, site, assets: report.assets.length, armVideos: report.armVideos, studioRuns: report.studioRuns, latestCases: report.latestCases.length, hashesVerified: report.hashesVerified, contactChecks: report.contactChecks }));
} catch (error) {
  report.error = String(error);
  throw error;
} finally {
  report.checkedAt = new Date().toISOString();
  await writeFile(path.join(evidence, "verification.json"), JSON.stringify(report, null, 2) + "\n");
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}

async function verifyContact(page) {
  const trigger = page.getByRole("button", { name: "Open contact options", exact: true });
  let visibleTriggers = 0;
  for (const button of await trigger.all()) {
    if (!await button.isVisible()) continue;
    visibleTriggers += 1;
    await button.click();
    const dialog = page.getByRole("dialog", { name: "Contact", exact: true });
    await dialog.waitFor();
    assert.equal(await dialog.getByLabel("Email address", { exact: true }).inputValue(), "bochuxt7@gmail.com");
    assert.equal(await dialog.getByRole("link", { name: "Write email", exact: true }).getAttribute("href"), "mailto:bochuxt7@gmail.com");
    assert.ok((await dialog.getByRole("link", { name: "Open Gmail", exact: true }).getAttribute("href")).startsWith("https://mail.google.com/mail/"));
    await page.evaluate(() => Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: async () => {} } }));
    await dialog.getByRole("button", { name: "Copy address", exact: true }).click();
    await dialog.getByRole("button", { name: "Copied", exact: true }).waitFor();
    await page.keyboard.press("Escape");
    await dialog.waitFor({ state: "hidden" });
    assert.equal(await button.evaluate((element) => element === document.activeElement), true);
    await button.click();
    await page.evaluate(() => Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText: async () => { throw new DOMException("Denied", "NotAllowedError"); } } }));
    await dialog.getByRole("button", { name: "Copy address", exact: true }).click();
    await dialog.getByText("Clipboard access was denied.", { exact: false }).waitFor();
    assert.equal(await dialog.getByLabel("Email address", { exact: true }).evaluate((element) => element.selectionEnd - element.selectionStart), "bochuxt7@gmail.com".length);
    await dialog.getByRole("button", { name: "Close contact dialog", exact: true }).click();
    await dialog.waitFor({ state: "hidden" });
    report.contactChecks += 1;
  }
  assert.ok(visibleTriggers > 0, "At least one Contact trigger must be visible");
}
