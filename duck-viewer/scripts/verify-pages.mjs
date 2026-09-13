import assert from "node:assert/strict";
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
const report = { success: false, url: "", assets: [], pageErrors: [], failedResponses: [], backendRequests: [], screenshots: [] };
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
  await page.getByRole("button", { name: "English", exact: true }).click();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: path.join(evidence, "arm-mobile.png"), fullPage: true });
  report.screenshots.push("arm-mobile.png");
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
  await page.getByRole("link", { name: "Back to Microduck Studio", exact: true }).click();
  await page.getByRole("radio").first().waitFor();
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
  console.log(JSON.stringify({ success: true, site, assets: report.assets.length, armVideos: report.armVideos, studioRuns: report.studioRuns }));
} catch (error) {
  report.error = String(error);
  throw error;
} finally {
  report.checkedAt = new Date().toISOString();
  await writeFile(path.join(evidence, "verification.json"), JSON.stringify(report, null, 2) + "\n");
  await browser?.close();
  if (server) await new Promise((resolve) => server.close(resolve));
}
