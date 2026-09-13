import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || import.meta.url)("playwright");
const base = process.env.WINGPOD_SITE || "http://127.0.0.1:63317/";
const root = path.resolve(import.meta.dirname, "..");
const output = path.resolve(root, "../.omx/artifacts/wingpod-image-refresh");
const manifest = JSON.parse(await readFile(path.join(root, "public/static/wingpod/manifest.json")));
const expected = ["hero.png", "camera-eyes-detail.png"].map((filename) => {
  const entry = manifest.files[filename];
  const extension = path.extname(filename);
  return `${path.basename(filename, extension)}-soft-${entry.sha256.slice(0, 12)}${extension}`;
});
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const report = { success: false, pages: [], errors: [] };
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  page.on("pageerror", (error) => report.errors.push(error.message));
  for (const [route, selectors] of [
    ["wingpod/", ["main > section:first-child img", "figure img"]],
    ["wingpod-v2/index.html", [".hero img", ".detail"]],
  ]) {
    await page.goto(new URL(route, base).href);
    const images = [];
    for (const [index, selector] of selectors.entries()) {
      const image = page.locator(selector).first();
      await image.scrollIntoViewIfNeeded();
      await image.evaluate(async (element) => { await element.decode(); });
      const src = await image.evaluate((element) => element.currentSrc);
      const url = new URL(src);
      const underlying = url.searchParams.get("url") || url.pathname;
      assert.equal(path.basename(underlying), expected[index]);
      const response = await fetch(new URL(underlying, base));
      assert.equal(response.status, 200);
      assert.equal(createHash("sha256").update(Buffer.from(await response.arrayBuffer())).digest("hex"),
        manifest.files[expected[index]].sha256);
      const screenshot = `${route.startsWith("wingpod/") ? "app" : "standalone"}-${index ? "content" : "header"}.png`;
      await image.screenshot({ path: path.join(output, screenshot) });
      images.push({ src, underlying, screenshot });
    }
    report.pages.push({ route, images });
  }
  assert.deepEqual(report.errors, []);
  report.success = true;
} finally {
  await browser.close();
  await writeFile(path.join(output, "verification.json"), JSON.stringify(report, null, 2) + "\n");
}
console.log(JSON.stringify(report, null, 2));
