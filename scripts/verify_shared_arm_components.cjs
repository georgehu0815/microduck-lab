const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { chromium } = require('/Users/ghu/community/node_modules/playwright');

const root = path.resolve(__dirname, '..');
const generated = path.join(root, 'hardware/md-arm-t1/generated');
const docs = path.join(root, 'docs/robot-arm-design');

(async () => {
  const tests = spawnSync(path.join(root, 'rlx/.venv-microduck/bin/python'),
    ['-m', 'unittest', 'discover', '-s', 'hardware/md-arm-t1/tests', '-p', 'test_*.py'],
    { cwd: root, encoding: 'utf8' });
  const testLog = (tests.stdout || '') + (tests.stderr || '');
  fs.writeFileSync(path.join(docs, 'component-hardware-tests.log'), testLog);
  assert.equal(tests.status, 0, testLog);
  const count = /Ran (\d+) tests/.exec(testLog);
  assert(count, 'Hardware test result missing');
  function hash(filename) {
    return createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
  }
  const audit = JSON.parse(fs.readFileSync(path.join(generated, 'component-consistency.json'), 'utf8'));
  assert.equal(audit.source_xml_sha256, hash(path.join(root, audit.source_xml)));
  assert.equal(audit.mesh.sha256, hash(path.join(root, audit.mesh.source)));
  assert.equal(audit.spec_sha256, hash(path.join(root, 'hardware/md-arm-t1/spec/md-arm-t1.json')));
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 960, height: 800 }, deviceScaleFactor: 1 });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const svg = fs.readFileSync(path.join(generated, 'shared-xl330-comparison.svg'), 'utf8');
    await page.setContent(`<html><body style="margin:0">${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: path.join(docs, 'shared-xl330-comparison.png') });
    const transform = await page.locator('svg').evaluate(element => {
      const matrix = element.getScreenCTM();
      return { scale: matrix.a, x: matrix.e, y: matrix.f };
    });
    assert.equal(transform.scale, 1, 'SVG must render at native scale');
    const pairs = [];
    for (let column = 0; column < 3; column++) {
      const body = await page.screenshot({ clip: { x: transform.x + 60 + column * 295, y: transform.y + 120, width: 200, height: 185 } });
      const arm = await page.screenshot({ clip: { x: transform.x + 60 + column * 295, y: transform.y + 390, width: 200, height: 185 } });
      fs.writeFileSync(path.join(docs, `shared-motor-body-${column}.png`), body);
      fs.writeFileSync(path.join(docs, `shared-motor-arm-${column}.png`), arm);
      const difference = await page.evaluate(async ({ body, arm }) => {
        async function pixels(encoded) {
          const image = new Image();
          image.src = `data:image/png;base64,${encoded}`;
          await image.decode();
          const canvas = document.createElement('canvas');
          canvas.width = image.width;
          canvas.height = image.height;
          const context = canvas.getContext('2d');
          context.drawImage(image, 0, 0);
          return context.getImageData(0, 0, image.width, image.height).data;
        }
        const bodyPixels = await pixels(body);
        const armPixels = await pixels(arm);
        let total = 0;
        let maximum = 0;
        for (let offset = 0; offset < bodyPixels.length; offset++) {
          const delta = Math.abs(bodyPixels[offset] - armPixels[offset]);
          total += delta;
          maximum = Math.max(maximum, delta);
        }
        return { mean_absolute_rgba: total / bodyPixels.length, max_channel_difference: maximum };
      }, { body: body.toString('base64'), arm: arm.toString('base64') });
      assert(difference.mean_absolute_rgba <= 0.5 && difference.max_channel_difference <= 32,
        `Shared projection ${column} differs beyond rasterization tolerance: ${JSON.stringify(difference)}`);
      pairs.push({ projection: column, identical_png_bytes: body.equals(arm), ...difference });
    }
    await page.goto(pathToFileURL(path.join(docs, 'COMPONENT-CONSISTENCY.zh-CN.html')).href);
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    assert.equal(await page.locator('img').count(), 1);
    await page.pdf({ path: path.join(docs, 'COMPONENT-CONSISTENCY.zh-CN.pdf'), format: 'A4', printBackground: true,
      margin: { top: '14mm', bottom: '14mm', left: '14mm', right: '14mm' } });
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(docs, 'component-browser-verification.json'), JSON.stringify({
      status: 'passed', projections: pairs, page_errors: errors,
      raster_tolerance: { mean_absolute_rgba: 0.5, max_channel_difference: 32,
        reason: 'Subpixel antialiasing differs across vertical translations; source geometry equality is tested separately.' },
      html_source_images_decoded: true, hardware_certified: false,
      source_equality_tests: { passed: true, tests: Number(count[1]), log: 'component-hardware-tests.log',
        log_sha256: hash(path.join(docs, 'component-hardware-tests.log')) },
      source_hashes: Object.fromEntries([
        audit.source_xml, audit.mesh.source,
        'hardware/md-arm-t1/spec/md-arm-t1.json',
        'hardware/md-arm-t1/scripts/shared_components.py',
        'hardware/md-arm-t1/tests/test_shared_components.py',
        'hardware/md-arm-t1/cad/md_arm_t1.scad',
        'hardware/md-arm-t1/generated/component-consistency.json',
        'hardware/md-arm-t1/generated/shared-xl330.scad',
        'hardware/md-arm-t1/generated/shared-xl330-comparison.svg',
        'docs/robot-arm-design/COMPONENT-CONSISTENCY.zh-CN.md',
        'docs/robot-arm-design/COMPONENT-CONSISTENCY.zh-CN.html',
        'docs/robot-arm-design/COMPONENT-CONSISTENCY.zh-CN.pdf',
        'scripts/verify_shared_arm_components.cjs',
      ].map(filename => [filename, hash(path.join(root, filename))])),
    }, null, 2) + '\n');
    console.log('Three shared-motor pairs match within recorded rasterization tolerance; Chinese PDF rendered.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
