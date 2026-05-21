#!/usr/bin/env node

const path = require('path');
const { chromium } = require('playwright');

const repoRoot = path.resolve(__dirname, '..');
const sourceDir = path.join(repoRoot, 'docs', 'report_evidence', 'figure_sources');
const assetDir = path.join(repoRoot, 'docs', 'report_evidence', 'assets');

const figures = [
  {
    url: 'http://127.0.0.1:8123/figure2_1_topology.html',
    output: path.join(assetDir, 'system_trust_boundary_topology.png'),
    width: 1800,
    height: 1080,
  },
  {
    url: 'http://127.0.0.1:8123/figure2_2_sequence.html',
    output: path.join(assetDir, 'software_flow_sequence.png'),
    width: 1820,
    height: 1240,
  },
];

async function renderFigure(browser, figure) {
  const page = await browser.newPage({
    viewport: { width: figure.width, height: figure.height },
    deviceScaleFactor: 2,
  });
  await page.goto(figure.url, { waitUntil: 'networkidle' });
  await page.screenshot({
    path: figure.output,
    fullPage: false,
    type: 'png',
  });
  await page.close();
  console.log(`[ok] rendered ${path.basename(figure.output)}`);
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
  });
  try {
    for (const figure of figures) {
      await renderFigure(browser, figure);
    }
  } finally {
    await browser.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
