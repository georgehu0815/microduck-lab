import assert from "node:assert/strict";
import { createReadStream } from "node:fs";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || import.meta.url)("playwright");
const root = path.resolve(import.meta.dirname, "..");
const output = path.join(root, "out");
const evidence = path.resolve(root, "../.omx/artifacts/classroom-viewer");
const prefix = "/microduck-lab/";
const mime = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
  ".txt": "text/plain", ".jpg": "image/jpeg", ".png": "image/png", ".svg": "image/svg+xml", ".woff2": "font/woff2",
  ".pdf": "application/pdf", ".mp4": "video/mp4", ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation" };
const server = createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    if (!pathname.startsWith(prefix)) { response.writeHead(404).end(); return; }
    let filename = path.resolve(output, pathname.slice(prefix.length));
    if (!filename.startsWith(`${output}${path.sep}`) && filename !== output) { response.writeHead(403).end(); return; }
    if ((await stat(filename)).isDirectory()) filename = path.join(filename, "index.html");
    const size = (await stat(filename)).size;
    const range = request.headers.range?.match(/^bytes=(\d+)-(\d*)$/);
    const start = range ? Number(range[1]) : 0;
    const end = range?.[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (range && (start > end || start >= size)) { response.writeHead(416).end(); return; }
    response.writeHead(range ? 206 : 200, { "content-type": mime[path.extname(filename)] || "application/octet-stream",
      "content-length": String(size ? end - start + 1 : 0), "accept-ranges": "bytes",
      ...(range ? { "content-range": `bytes ${start}-${end}/${size}` } : {}) });
    if (request.method === "HEAD" || !size) response.end();
    else createReadStream(filename, { start, end }).pipe(response);
  } catch { response.writeHead(404).end(); }
});
const report = { success: false, editionsOpened: [], slidesDecoded: 0, downloadsVerified: 0, videosPlayed: 0, errors: [], screenshots: [] };
let browser;
await mkdir(evidence, { recursive: true });
try {
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const site = process.argv[2] || `http://127.0.0.1:${server.address().port}${prefix}`;
  const catalog = await (await fetch(new URL("classroom-assets/catalog.json", site))).json();
  browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  page.on("pageerror", (error) => report.errors.push(error.message));
  await page.goto(new URL("classroom/", site).href);
  await page.getByTestId("lesson-card").last().waitFor();
  assert.equal(await page.getByTestId("lesson-card").count(), 8);
  assert.equal(await page.locator("html").getAttribute("lang"), "en");
  await page.screenshot({ path: path.join(evidence, "catalog-desktop.png"), fullPage: true });
  report.screenshots.push("catalog-desktop.png");
  for (const language of ["en", "zh"]) {
    await page.getByRole("button", { name: language === "en" ? "English" : "中文", exact: true }).click();
    for (const lesson of catalog.lessons) {
      const edition = lesson.editions[language];
      await page.locator(`[data-lesson-id="${lesson.id}"]`).click();
      const player = page.getByTestId("slide-player");
      const image = page.getByTestId("slide-image");
      await image.evaluate(async (element) => { await element.decode(); });
      assert.ok((await image.getAttribute("src")).endsWith(edition.slides[0].image));
      assert.equal(await page.getByTestId("previous-slide").isDisabled(), true);
      await page.getByTestId("next-slide").click();
      await image.evaluate(async (element) => { await element.decode(); });
      assert.ok((await image.getAttribute("src")).endsWith(edition.slides[1].image));
      await player.focus();
      await page.keyboard.press("End");
      await image.evaluate(async (element) => { await element.decode(); });
      assert.ok((await image.getAttribute("src")).endsWith(edition.slides.at(-1).image));
      assert.equal(await page.getByTestId("next-slide").isDisabled(), true);
      await page.keyboard.press("Home");
      assert.ok((await image.getAttribute("src")).endsWith(edition.slides[0].image));
      for (const file of [edition.pptx, edition.pdf]) {
        const href = new URL(`classroom-assets/${file}`, site).href;
        assert.equal((await fetch(href, { method: "HEAD" })).status, 200);
        assert.ok(await player.locator(`a[download][href$="${file}"]`).count());
        report.downloadsVerified += 1;
      }
      report.slidesDecoded += await page.evaluate(async (urls) => {
        for (const url of urls) {
          const image = new Image();
          image.src = url;
          await image.decode();
          if (!image.naturalWidth || !image.naturalHeight) throw new Error("Empty slide image");
        }
        return urls.length;
      }, edition.slides.map((slide) => new URL(`classroom-assets/${slide.image}`, site).href));
      report.editionsOpened.push(`${lesson.id}/${language}`);
      if (language === "en") {
        for (const video of await player.locator("video").all()) {
          await video.evaluate(async (element) => { element.muted = true; await element.play(); });
          await page.waitForFunction((source) => [...document.querySelectorAll("video")]
            .some((element) => element.src === source && element.currentTime > 0.08), await video.getAttribute("src").then((src) => new URL(src, site).href));
          await video.evaluate((element) => element.pause());
          report.videosPlayed += 1;
        }
      }
      await page.getByTestId("close-presentation").click();
    }
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('[data-lesson-id="drawing"]').click();
  await page.getByTestId("slide-image").evaluate(async (image) => { await image.decode(); });
  await page.waitForFunction(() => {
    const player = document.querySelector('[data-testid="slide-player"]');
    const image = document.querySelector('[data-testid="slide-image"]').getBoundingClientRect();
    return player.getBoundingClientRect().top <= parseFloat(getComputedStyle(player).scrollMarginTop) + 1
      && image.top < innerHeight && image.bottom > 0;
  });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
  await page.screenshot({ path: path.join(evidence, "player-mobile-zh.png") });
  report.screenshots.push("player-mobile-zh.png");
  await page.reload();
  assert.equal(await page.locator("html").getAttribute("lang"), "zh-CN");
  for (const route of ["", "arm/", "wingpod/"]) {
    await page.goto(new URL(route, site).href);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${route} mobile overflow`);
    const link = page.locator('a[href$="/classroom"]:visible, a[href$="/classroom/"]:visible').first();
    await link.waitFor({ state: "visible" });
    await link.click();
    await page.getByTestId("lesson-card").last().waitFor();
  }
  assert.equal(report.editionsOpened.length, 16);
  assert.equal(report.slidesDecoded, 644);
  assert.equal(report.downloadsVerified, 32);
  assert.equal(report.videosPlayed, 9);
  assert.deepEqual(report.errors, []);
  report.success = true;
} finally {
  await browser?.close();
  await new Promise((resolve) => server.close(resolve));
  await writeFile(path.join(evidence, "verification.json"), JSON.stringify(report, null, 2) + "\n");
}
console.log(JSON.stringify(report, null, 2));
