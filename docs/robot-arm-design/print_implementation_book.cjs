const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('/Users/ghu/community/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    await page.goto(pathToFileURL(path.join(__dirname, 'CLASSROOM.zh-CN.html')).href);
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    await page.screenshot({ path: path.join(__dirname, 'classroom-book-preview.png') });
    await page.pdf({
      path: path.join(__dirname, 'CLASSROOM.zh-CN.pdf'), format: 'A4',
      printBackground: true, margin: { top: '16mm', bottom: '18mm', left: '14mm', right: '14mm' },
      displayHeaderFooter: true, headerTemplate: '<span></span>',
      footerTemplate: '<div style="width:100%;font-size:8px;text-align:center;color:#52666f">MD-Arm-T1 · 2026-09-12 · Simulation only · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    });
    console.log('Rendered Chinese classroom PDF with all images decoded.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
