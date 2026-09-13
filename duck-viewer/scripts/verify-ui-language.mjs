import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve(process.env.LANGUAGE_EVIDENCE_DIR || path.join(root, "rlx/artifacts/viewer-language"));
const receipt = { passed: false, checkedAt: new Date().toISOString(), baseUrl, checks: [], pageErrors: [] };
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });

function languageToggle(page) {
  return page.locator('[data-testid="language-toggle"]:visible');
}

async function ready(page, language) {
  await page.waitForFunction((expected) => document.documentElement.lang === expected, language === "zh" ? "zh-CN" : "en");
  await languageToggle(page).waitFor();
  assert.equal(await languageToggle(page).getByRole("button", { name: language === "zh" ? "中文" : "English", exact: true }).getAttribute("aria-pressed"), "true");
}

async function switchLanguage(page, language) {
  await languageToggle(page).getByRole("button", { name: language === "zh" ? "中文" : "English", exact: true }).click();
  await ready(page, language);
}

async function loadedVideo(page) {
  await page.waitForFunction(() => {
    const video = document.querySelector('[data-testid="arm-video-library"] video');
    return video && video.readyState >= 2 && video.videoWidth > 0;
  });
}

try {
  const context = await browser.newContext({ viewport: { width: 1536, height: 1100 }, locale: "zh-CN" });
  const page = await context.newPage();
  page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
  await page.goto(`${baseUrl}/arm#videos`, { waitUntil: "domcontentloaded" });
  await ready(page, "en");
  await page.getByRole("heading", { name: "Arm video library", exact: true }).waitFor();
  receipt.checks.push("English default, even with a Chinese browser locale");
  await loadedVideo(page);
  await page.getByLabel("Case", { exact: true }).selectOption("arms-co-carry-v1");
  await page.getByLabel("Provenance", { exact: true }).selectOption("current");
  await loadedVideo(page);
  const videoHandle = await page.locator("video").elementHandle();
  await videoHandle.evaluate(async (video) => { video.muted = true; await video.play(); });
  await page.waitForFunction(() => document.querySelector("video").currentTime > 0.4);
  await videoHandle.evaluate((video) => video.pause());
  const before = await videoHandle.evaluate((video) => ({ time: video.currentTime, src: video.currentSrc }));
  const selected = await page.locator('[data-video-id][aria-pressed="true"]').getAttribute("data-video-id");
  await page.screenshot({ path: path.join(output, "arm-english.png"), fullPage: true });

  await switchLanguage(page, "zh");
  await page.getByRole("heading", { name: "机械臂视频证据库", exact: true }).waitFor();
  assert.equal(await page.getByLabel("案例", { exact: true }).inputValue(), "arms-co-carry-v1");
  assert.equal(await page.getByLabel("来源", { exact: true }).inputValue(), "current");
  assert.equal(await page.locator('[data-video-id][aria-pressed="true"]').getAttribute("data-video-id"), selected);
  assert.equal(await videoHandle.evaluate((video) => video === document.querySelector("video")), true);
  const after = await videoHandle.evaluate((video) => ({ time: video.currentTime, src: video.currentSrc }));
  assert.equal(after.src, before.src);
  assert(Math.abs(after.time - before.time) < 0.05);
  receipt.checks.push("Chinese translation preserves filters, selection, player DOM and playback position");
  await page.screenshot({ path: path.join(output, "arm-chinese.png"), fullPage: true });

  await page.getByRole("button", { name: "清除筛选", exact: true }).click();
  await page.getByLabel("搜索视频", { exact: true }).fill("Pick and place");
  assert(await page.locator("[data-video-id]").count() > 0);
  await page.getByLabel("搜索视频", { exact: true }).fill("抓取与放置");
  assert(await page.locator("[data-video-id]").count() > 0);
  receipt.checks.push("Video search supports both case-name languages");

  await page.reload({ waitUntil: "domcontentloaded" });
  await ready(page, "zh");
  await page.getByRole("heading", { name: "机械臂视频证据库", exact: true }).waitFor();
  receipt.checks.push("Chinese preference survives reload");

  const sibling = await context.newPage();
  sibling.on("pageerror", (error) => receipt.pageErrors.push(error.message));
  await sibling.goto(`${baseUrl}/arm#videos`, { waitUntil: "domcontentloaded" });
  await ready(sibling, "zh");
  await switchLanguage(page, "en");
  await ready(sibling, "en");
  await sibling.close();
  receipt.checks.push("Preference synchronizes across tabs");

  for (const language of ["en", "zh"]) {
    await switchLanguage(page, language);
    await page.setViewportSize({ width: 390, height: 900 });
    await loadedVideo(page);
    await page.locator("video").evaluate((video) => { video.currentTime = Math.min(1, video.duration / 2); });
    await page.screenshot({ path: path.join(output, `arm-mobile-${language}.png`), fullPage: true });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
    assert(await languageToggle(page).isVisible());
  }
  receipt.checks.push("English and Chinese mobile layouts have no horizontal overflow");

  await page.setViewportSize({ width: 1536, height: 1100 });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await ready(page, "zh");
  await page.getByRole("link", { name: "机械臂视频", exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, "studio-chinese.png"), fullPage: true });
  await switchLanguage(page, "en");
  await page.getByRole("link", { name: "Arm videos", exact: true }).waitFor();
  await page.screenshot({ path: path.join(output, "studio-english.png"), fullPage: true });
  for (const language of ["en", "zh"]) {
    await switchLanguage(page, language);
    for (const width of [800, 390]) {
      await page.setViewportSize({ width, height: 900 });
      const toggleBounds = await languageToggle(page).boundingBox();
      assert(toggleBounds, "Language toggle is hidden");
      if (width > 560) {
        const sidebarBounds = await page.locator("aside").first().boundingBox();
        assert(sidebarBounds);
        assert(toggleBounds.x >= sidebarBounds.x && toggleBounds.x + toggleBounds.width <= sidebarBounds.x + sidebarBounds.width + 1, "Tablet language toggle exceeds sidebar");
      } else {
        assert(toggleBounds.x >= 0 && toggleBounds.x + toggleBounds.width <= width, "Mobile language toggle exceeds viewport");
      }
      assert(await page.getByRole("link", { name: language === "zh" ? "机械臂视频" : "Arm videos", exact: true }).isVisible());
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
      await page.screenshot({ path: path.join(output, `studio-${width}-${language}.png`), fullPage: true });
    }
  }
  await page.setViewportSize({ width: 1536, height: 1100 });
  await switchLanguage(page, "en");
  receipt.checks.push("Studio tablet/mobile toggle fits its container and arm navigation remains accessible in both languages");
  await page.getByRole("link", { name: "Arm videos", exact: true }).click();
  await ready(page, "en");
  await page.getByRole("heading", { name: "Arm video library", exact: true }).waitFor();
  receipt.checks.push("Studio/Arm navigation shares the selected language");

  await page.getByRole("button", { name: "Live simulator", exact: true }).click();
  await page.getByText("Simulation online", { exact: true }).waitFor();
  const reachCase = page.getByRole("region", { name: "Arm experiment cases" }).getByRole("button").filter({ hasText: "arm-reach-v1" });
  await reachCase.click();
  await page.getByLabel("Reset seed", { exact: true }).fill("42");
  for (let tick = 0; tick < 12; tick += 1) await page.getByRole("slider").first().press("ArrowRight");
  const simulatorRequests = [];
  const recordRequest = (request) => {
    if (/\/api\/arm\/(reset|step|teacher|health)(?:\?|$)/.test(request.url())) simulatorRequests.push(`${request.method()} ${request.url()}`);
  };
  page.on("request", recordRequest);
  await switchLanguage(page, "zh");
  await page.getByRole("heading", { name: "仿真控制", exact: true }).waitFor();
  assert.equal(await page.getByLabel("重置种子", { exact: true }).inputValue(), "42");
  assert.equal(await page.getByRole("slider").first().inputValue(), "0.12");
  assert.equal(await page.getByRole("region", { name: "机械臂实验案例" }).getByRole("button").filter({ hasText: "arm-reach-v1" }).getAttribute("aria-pressed"), "true");
  await page.waitForTimeout(500);
  assert.deepEqual(simulatorRequests, []);
  page.off("request", recordRequest);
  const resetResponse = page.waitForResponse((response) => response.url().endsWith("/api/arm/reset") && response.request().method() === "POST");
  await page.getByRole("button", { name: "重置所选案例", exact: true }).click();
  const reset = await (await resetResponse).json();
  assert.equal(reset.case_id, "arm-reach-v1");
  assert.equal(reset.seed, 42);
  assert.equal(reset.hardware_enabled, false);
  const teacherResponse = page.waitForResponse((response) => response.url().endsWith("/api/arm/teacher") && response.request().method() === "POST");
  await page.getByRole("button", { name: "推进 Teacher", exact: true }).click();
  const stepped = await (await teacherResponse).json();
  assert(stepped.step > reset.step);
  await page.screenshot({ path: path.join(output, "arm-live-chinese.png"), fullPage: true });
  receipt.checks.push("Live language switch preserves case, seed and manual action without reconnecting or sending commands; Chinese reset/teacher controls work");

  await page.evaluate(() => localStorage.setItem("microduck.ui.language", "unsupported"));
  await page.reload({ waitUntil: "domcontentloaded" });
  await ready(page, "en");
  receipt.checks.push("Invalid saved language falls back to English");

  const blocked = await browser.newContext();
  await blocked.addInitScript(() => {
    Object.defineProperty(window, "localStorage", { configurable: true, get() { throw new DOMException("Storage blocked", "SecurityError"); } });
  });
  const blockedPage = await blocked.newPage();
  blockedPage.on("pageerror", (error) => receipt.pageErrors.push(error.message));
  await blockedPage.goto(`${baseUrl}/arm#videos`, { waitUntil: "domcontentloaded" });
  await ready(blockedPage, "en");
  await switchLanguage(blockedPage, "zh");
  await blockedPage.getByRole("heading", { name: "机械臂视频证据库", exact: true }).waitFor();
  await blocked.close();
  receipt.checks.push("Toggle works with browser storage blocked");
  assert.deepEqual(receipt.pageErrors, []);
  receipt.passed = true;
  console.log(JSON.stringify(receipt, null, 2));
} catch (error) {
  receipt.error = error.stack || String(error);
  throw error;
} finally {
  await writeFile(path.join(output, "verification.json"), `${JSON.stringify(receipt, null, 2)}\n`);
  await browser.close();
}
