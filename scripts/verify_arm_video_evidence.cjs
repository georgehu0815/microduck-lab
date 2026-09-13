const fs = require('node:fs/promises');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('/Users/ghu/community/node_modules/playwright');

async function main() {
  const root = path.resolve(process.argv[2] || 'rlx/runs/arm/recheck-20260912-v2');
  const manifest = JSON.parse(await fs.readFile(path.join(root, 'video-evidence.json'), 'utf8'));
  const browser = await chromium.launch({ headless: true });
  const errors = [];
  const evidence = [];
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 1000 } });
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.goto(pathToFileURL(path.join(root, 'index.zh-CN.html')).href);
    await page.evaluate(() => document.fonts.ready);
    const count = await page.locator('video').count();
    if (count !== manifest.video_count) throw new Error(`Video count mismatch: ${count}`);
    await page.screenshot({ path: path.join(root, 'browser-desktop.png') });
    for (let index = 0; index < count; index += 1) {
      const item = manifest.videos[index];
      const video = page.locator('video').nth(index);
      await video.scrollIntoViewIfNeeded();
      await video.evaluate(element => { element.muted = true; element.load(); });
      await page.waitForFunction(index => {
        const element = document.querySelectorAll('video')[index];
        return element.readyState >= 2 && element.duration > 0;
      }, index);
      await video.evaluate(element => element.play());
      await page.waitForFunction(index => document.querySelectorAll('video')[index].currentTime > 0.08, index);
      await video.evaluate(element => { element.pause(); });
      for (const [name, position] of [['middle', 0.5], ['terminal', 1]]) {
        await video.evaluate((element, position) => new Promise(resolve => {
          element.addEventListener('seeked', () => requestAnimationFrame(() => requestAnimationFrame(resolve)), { once: true });
          element.currentTime = position === 1 ? Math.max(0, element.duration - 0.02) : element.duration * position;
        }), position);
        await video.screenshot({ path: path.join(root, 'videos', item.id, `browser-${name}.png`) });
      }
      evidence.push({
        id: item.id,
        actual_playback_passed: true,
        ...await video.evaluate(element => ({
          duration: element.duration, width: element.videoWidth, height: element.videoHeight,
          terminal_seek_time: element.currentTime, media_error: element.error?.message || null,
        })),
      });
      console.log(JSON.stringify(evidence.at(-1)));
    }
    await page.setViewportSize({ width: 390, height: 844 });
    await page.evaluate(() => window.scrollTo(0, 0));
    const mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    if (mobileOverflow) throw new Error('Evidence page has horizontal mobile overflow');
    await page.screenshot({ path: path.join(root, 'browser-mobile.png') });
    await page.evaluate(() => {
      document.querySelectorAll('video').forEach(video => {
        const image = document.createElement('img');
        image.src = video.poster;
        image.style.width = '100%';
        video.after(image);
      });
    });
    await page.evaluate(() => Promise.all([...document.images].map(image => image.decode())));
    await page.pdf({ path: path.join(root, 'evidence.zh-CN.pdf'), format: 'A4', printBackground: true,
      margin: { top: '14mm', right: '12mm', bottom: '14mm', left: '12mm' } });
    if (errors.length) throw new Error(`Browser errors: ${errors.join('; ')}`);
    await fs.writeFile(path.join(root, 'browser-verification.json'), JSON.stringify({
      checked_at: new Date().toISOString(), actual_playbacks: evidence.length,
      all_passed: true, mobile_overflow: mobileOverflow, console_errors: errors, videos: evidence,
      pdf_contains_stills_not_embedded_video: true,
    }, null, 2) + '\n');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
