import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import path from "node:path";
import { pathToFileURL } from "node:url";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || import.meta.url)("playwright");
const root = path.resolve(import.meta.dirname, "..");
const output = path.join(root, "out");
const evidence = path.resolve(root, "../.omx/artifacts/wingpod-camera-viewer");
const prefix = "/microduck-lab/";
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".txt": "text/plain", ".md": "text/plain", ".csv": "text/csv", ".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".woff2": "font/woff2", ".ico": "image/x-icon" };
const report = { success: false, verifiedAssets: 0, playedCases: [], pageErrors: [], failedResponses: [], backendRequests: [], screenshots: [] };
await mkdir(evidence, { recursive: true });
const server = createServer(async (request, response) => {
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
    response.writeHead(range ? 206 : 200, { "content-type": mime[path.extname(filename)] || "application/octet-stream",
      "accept-ranges": "bytes", "content-length": String(size ? end - start + 1 : 0),
      ...(range ? { "content-range": `bytes ${start}-${end}/${size}` } : {}) });
    if (request.method === "HEAD" || !size) response.end();
    else createReadStream(filename, { start, end }).pipe(response);
  } catch { response.writeHead(404).end(); }
});
let browser;
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const site = `http://127.0.0.1:${server.address().port}${prefix}`;
  report.site = site;
  const manifest = JSON.parse(await readFile(path.join(output, "static/wingpod/manifest.json")));
  assert.equal(manifest.all_cases_included, true);
  assert.equal(manifest.design, "WingPod Camera v2 Soft");
  assert.equal(manifest.primary_palette, "cream-sage-peach");
  assert.equal(manifest.case_library_palette, "original graphite v2");
  for (const [name, expected] of Object.entries(manifest.files)) {
    const response = await fetch(new URL(`static/wingpod/${name}`, site));
    assert.equal(response.status, 200, name);
    const bytes = Buffer.from(await response.arrayBuffer());
    assert.equal(bytes.length, expected.bytes, name);
    assert.equal(createHash("sha256").update(bytes).digest("hex"), expected.sha256, name);
    report.verifiedAssets += 1;
  }
  const cases = JSON.parse(await readFile(path.join(output, "static/wingpod/cases.json")));
  assert.deepEqual(cases.summary, { positivePassed: 30, positiveTotal: 30, controlsMatched: 2, controlsTotal: 2 });
  const bom = JSON.parse(await readFile(path.join(output, "static/wingpod/BOM.json")));
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, locale: "en-US" });
  const page = await context.newPage();
  page.on("pageerror", (error) => report.pageErrors.push(error.message));
  page.on("response", (response) => { if (response.status() >= 400) report.failedResponses.push(`${response.status()} ${response.url()}`); });
  page.on("request", (request) => { if (new URL(request.url()).pathname.includes("/api/")) report.backendRequests.push(request.url()); });
  async function playAndSeek(expectedFile) {
    await page.waitForFunction((filename) => {
      const video = document.querySelector("video");
      return video && video.currentSrc.endsWith(filename) && video.readyState >= 2 && video.duration > 0;
    }, expectedFile);
    const duration = await page.locator("video").evaluate(async (video) => {
      video.muted = true; video.currentTime = 0;
      await video.play();
      return video.duration;
    });
    await page.waitForFunction(() => document.querySelector("video").currentTime > 0.08);
    await page.locator("video").evaluate((video) => { video.pause(); video.currentTime = Math.max(0, video.duration - 0.2); });
    await page.waitForFunction(() => {
      const video = document.querySelector("video");
      return !video.seeking && video.readyState >= 2 && video.currentTime > video.duration - 0.5;
    });
    assert.equal(await page.locator("video").evaluate((video) => video.error), null);
    return duration;
  }
  await page.goto(new URL("wingpod/", site).href);
  await page.waitForFunction(() => document.querySelectorAll("#cases article").length === 32);
  assert.equal(await page.locator("html").getAttribute("lang"), "en");
  await playAndSeek("tennis-return.mp4");
  await page.locator("video").screenshot({ path: path.join(evidence, "soft-task-final.png") });
  report.screenshots.push("soft-task-final.png");
  for (const record of cases.cases) {
    const card = page.locator("#cases article").filter({ has: page.getByText(record.id, { exact: true }) });
    await card.getByRole("button").click();
    const duration = await playAndSeek(record.video);
    assert.ok(Math.abs(duration - record.durationSeconds) < 0.1, record.id);
    report.playedCases.push(record.id);
    if (["nominal-0", "small-0", "large-0", "hold", "open_jaw"].includes(record.id)) {
      const filename = `historical-${record.id}-final.png`;
      await page.locator("video").screenshot({ path: path.join(evidence, filename) });
      report.screenshots.push(filename);
    }
  }
  await page.getByRole("button", { name: "Negative controls", exact: true }).click();
  assert.equal(await page.locator("#cases article").count(), 2);
  await page.getByRole("button", { name: "All evidence", exact: true }).click();
  await page.getByRole("button", { name: /Camera appearance/ }).click();
  await playAndSeek("wingpod-camera-eyes.mp4");
  const time = await page.locator("video").evaluate((video) => video.currentTime);
  await page.getByRole("button", { name: "中文", exact: true }).click();
  assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
  assert.ok(Math.abs(await page.locator("video").evaluate((video) => video.currentTime) - time) < 0.1);
  await page.screenshot({ path: path.join(evidence, "react-zh.png") });
  report.screenshots.push("react-zh.png");
  await page.getByRole("button", { name: "English", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(evidence, "react-mobile.png") });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  report.screenshots.push("react-mobile.png");
  await page.goto(new URL("wingpod-v2/index.html", site).href);
  await page.waitForFunction((rows) => document.querySelectorAll("#bomrows tr").length === rows
    && document.querySelectorAll("#caselist article").length === 32, bom.parts.length);
  assert.equal(await page.locator("#passcount").textContent(), "30/30");
  await playAndSeek("tennis-return.mp4");
  await page.selectOption("#filter", "control");
  assert.equal(await page.locator("#caselist article").count(), 2);
  assert.match(await page.locator("#caselist").textContent(), /expected failure/);
  await page.locator("#design").click();
  await playAndSeek("wingpod-camera-eyes.mp4");
  await page.locator("#chinese").click();
  assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
  assert.match(await page.locator("h1").textContent(), /小鸭子/);
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(evidence, "standalone-mobile-zh.png") });
  report.screenshots.push("standalone-mobile-zh.png");
  await page.reload();
  assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
  await page.locator("#english").click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const selector of [".hero img", ".detail"]) {
    const ratio = await page.locator(selector).evaluate((image) => {
      const bounds = image.getBoundingClientRect();
      return bounds.width / bounds.height;
    });
    assert.ok(Math.abs(ratio - 960 / 720) < 0.01, `${selector} must retain its natural aspect ratio`);
  }
  await page.screenshot({ path: path.join(evidence, "standalone-desktop.png") });
  report.screenshots.push("standalone-desktop.png");
  const downloads = await page.locator("a[download]").evaluateAll((links) => links.map((link) => link.href));
  for (const href of downloads) assert.equal((await fetch(href, { method: "HEAD" })).status, 200, href);
  const localPage = await context.newPage();
  const localErrors = [];
  localPage.on("pageerror", (error) => localErrors.push(error.message));
  await localPage.goto(pathToFileURL(path.resolve(root,
    "../artifacts/microduck-arm-v1c/appearance/wingpod-v2/index.html")).href);
  await localPage.waitForFunction((rows) => document.querySelectorAll("#bomrows tr").length === rows
    && document.querySelectorAll("#caselist article").length === 32, bom.parts.length);
  assert.deepEqual(localErrors, []);
  report.localArtifactLoads = true;
  await localPage.close();
  assert.equal(report.playedCases.length, 32);
  assert.deepEqual(report.pageErrors, []);
  assert.deepEqual(report.failedResponses, []);
  assert.deepEqual(report.backendRequests, []);
  report.success = true;
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
  await writeFile(path.join(evidence, "verification.json"), JSON.stringify(report, null, 2) + "\n");
}
console.log(JSON.stringify(report, null, 2));
