/* Optional browser regression suite.
   npm install --no-save playwright && npx playwright install chromium
   python3 tools/serve.py . 8000
   node tests/browser_turnaround.cjs
   Env: BASE_URL, PLAYWRIGHT_PATH, CHROMIUM_EXECUTABLE, SCREENSHOT_DIR.
*/
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_EXECUTABLE || undefined,
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-zygote', '--single-process'],
  });
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1040}});
    const errors = [], badResponses = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => { if (response.status() >= 400) badResponses.push(response.url()); });
    const base = process.env.BASE_URL || 'http://127.0.0.1:8000';
    const goto = async (suffix = '/turnaround/') => {
      await page.goto(base + suffix);
      await page.waitForSelector('#main-image:not([hidden])');
    };
    await goto('/');
    assert.match(page.url(), /\/turnaround\//);
    assert.equal(await page.locator('.frame-card').count(), 8);
    assert.equal(await page.locator('#detail-name').textContent(), 'Спереди');
    assert.equal(await page.locator('#error').isVisible(), false);
    await page.locator('#prev').click();
    assert.equal(await page.locator('#detail-index').textContent(), '08 / 08');
    await page.locator('#next').click();
    assert.equal(await page.locator('#detail-index').textContent(), '01 / 08');
    await page.locator('.frame-card').nth(1).click();
    assert.equal(await page.locator('#status-badge').textContent(), 'Зеркальный черновик');
    assert.match(await page.locator('#download-frame').getAttribute('href'), /02_045/);
    // The visible label also toggles the visually hidden, accessible switch.
    await page.locator('.toggle-row').click();
    assert.equal(await page.locator('#onion-image').isVisible(), true);
    assert.match(await page.locator('#onion-image').getAttribute('src'), /01_000/);
    await page.locator('#grid-toggle').click();
    assert.equal(await page.locator('#grid-toggle').getAttribute('aria-pressed'), 'false');
    await page.locator('#bg-toggle').click();
    assert.equal(await page.locator('#stage').getAttribute('data-bg'), 'white');
    await page.locator('#frame-range').fill('4');
    assert.equal(await page.locator('#detail-name').textContent(), 'Сзади');
    await page.locator('#play').click();
    await page.waitForFunction(() => document.querySelector('#detail-index').textContent !== '05 / 08');
    await page.locator('#play').click();
    assert.equal(await page.locator('#play').getAttribute('aria-pressed'), 'false');
    await page.locator('[data-mode="head"]').click();
    assert.equal(await page.locator('.frame-card').count(), 9);
    assert.equal(await page.locator('#detail-index').textContent(), '05 / 09');
    assert.match(await page.locator('#detail-note').textContent(), /целевые ключи/);
    await page.locator('.frame-card').nth(8).click();
    assert.equal(await page.locator('#detail-angle').textContent(), 'X +30 / Y -30');
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#download-frame').click();
    assert.equal((await downloadPromise).suggestedFilename(), '09_x+30_y-30.png');
    await page.locator('#play').click();
    await page.locator('[data-tab="storyboard"]').click();
    assert.equal(await page.locator('#play').getAttribute('aria-pressed'), 'false');
    assert.equal(await page.locator('#storyboard').isVisible(), true);
    assert.equal(await page.locator('#views').isVisible(), false);
    const pdf = await page.request.get(base + '/turnaround/pack/reference_boards.pdf');
    assert.equal(pdf.status(), 200);
    assert.equal((await pdf.body()).subarray(0, 4).toString(), '%PDF');
    await page.locator('[data-tab="prepare"]').click();
    assert.equal(await page.locator('#checklist input').count(), 26);
    await page.locator('#checklist input').first().check();
    assert.equal(await page.locator('#progress-number').textContent(), '4%');
    await page.reload();
    await page.waitForSelector('#main-image:not([hidden])', {state: 'attached'});
    assert.equal(await page.locator('#prepare').isVisible(), true);
    assert.equal(await page.locator('#checklist input').first().isChecked(), true);
    assert.equal(await page.locator('#progress-number').textContent(), '4%');
    const exportPromise = page.waitForEvent('download');
    await page.locator('#export-checklist').click();
    const exported = await exportPromise;
    const result = JSON.parse(fs.readFileSync(await exported.path(), 'utf8'));
    assert.equal(result.items.length, 26);
    assert.equal(result.items[0].done, true);
    await page.locator('[data-tab="views"]').click();
    await page.locator('body').click({position: {x: 240, y: 95}});
    await page.keyboard.press('ArrowLeft');
    assert.equal(await page.locator('#detail-index').textContent(), '08 / 08');
    await page.keyboard.press('ArrowRight');
    assert.equal(await page.locator('#detail-index').textContent(), '01 / 08');
    if (process.env.SCREENSHOT_DIR) {
      fs.mkdirSync(process.env.SCREENSHOT_DIR, {recursive: true});
      await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, 'desktop.png'), fullPage: true});
    }
    for (const width of [768, 390, 320]) {
      await page.setViewportSize({width, height: 844});
      await page.locator('[data-tab="views"]').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `No horizontal page overflow at ${width}`);
      await page.locator('[data-mode="head"]').click();
      assert.equal(await page.locator('.frame-card').count(), 9);
      await page.locator('[data-mode="body"]').click();
      await page.locator('#next').click();
      assert.match(await page.locator('#detail-index').textContent(), /\/ 08/);
      if (width === 390 && process.env.SCREENSHOT_DIR) await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, 'mobile.png'), fullPage: true});
      await page.locator('[data-tab="prepare"]').click();
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `Checklist fits at ${width}`);
    }
    // File failures have a readable recovery path and disabled playback.
    await page.route('**/pack/manifest.json', route => route.fulfill({status: 404, body: 'not found'}));
    await page.goto(base + '/turnaround/');
    await page.waitForSelector('#error:not([hidden])');
    assert.equal(await page.locator('#play').isDisabled(), true);
    assert.match(await page.locator('#error').textContent(), /build_turnaround/);
    assert.deepEqual(errors, []);
    assert.deepEqual(badResponses.filter(url => !url.endsWith('/pack/manifest.json')), []);
    console.log('PASS: views, looping, player, head keys, downloads, tabs, persistence, keyboard, mobile layouts and error recovery');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
