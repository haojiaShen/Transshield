import { firefox } from 'playwright';
const browser = await firefox.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 1700 } });
await page.goto('http://127.0.0.1:7860/', { waitUntil: 'networkidle', timeout: 60000 });
await page.locator('#innovation').scrollIntoViewIfNeeded();
await page.waitForTimeout(1200);
await page.screenshot({ path: '/home/yclcg/Transshield_final/tmp/playwright_shots/innovation-real.png' });
await browser.close();
