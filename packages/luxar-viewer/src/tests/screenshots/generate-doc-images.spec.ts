/**
 * Documentation Image Generator
 *
 * Generates screenshots for Sphinx documentation (tutorials, guides).
 * Separate from README images — these capture UI panels, nD navigation, etc.
 *
 * Usage:
 *   pnpm doc-images
 *   # or: npx playwright test --config playwright.screenshots.config.ts generate-doc-images
 *
 * Prerequisites:
 *   - Run demo generators first: make generate-readme-demos
 *   - Servers started automatically by playwright config
 */

import { test } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

// Output directory for documentation images
const OUTPUT_DIR = path.resolve(__dirname, '../../../../../docs/images/docs');

// Server URLs
const DATA_SERVER = 'http://localhost:9876';
const VIEWER_URL = 'http://localhost:5173';

/**
 * Wait for Luxar to initialize and load data.
 * @param requireElements - If true (default), wait for totalElements > 0.
 *   Set to false for datasets where no geometry is expected initially (e.g., toggles off).
 */
async function waitForReady(
  page: any,
  { timeout = 90000, requireElements = true } = {}
): Promise<void> {
  await page.waitForFunction(
    (req: boolean) => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.getState) return false;
      const state = debug.getState();
      if (!state || !state.initialized) return false;
      // totalElements includes both points and gsplats; fall back to totalPoints for compat
      if (req && ((state.totalElements ?? state.totalPoints) || 0) === 0) return false;
      return true;
    },
    requireElements,
    { timeout }
  );
}

/**
 * Center camera and wait
 */
async function centerCamera(page: any): Promise<void> {
  await page.keyboard.press('f');
  await page.waitForTimeout(800);
}

/**
 * Hide all UI panels
 */
async function hideUI(page: any): Promise<void> {
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
}

/**
 * Render frames to settle
 */
async function renderFrames(page: any, count = 10): Promise<void> {
  for (let i = 0; i < count; i++) {
    await page.evaluate(() => {
      (window as any).__luxarDebug?.renderOnce?.();
    });
    await page.waitForTimeout(50);
  }
  await page.waitForTimeout(300);
}

// Ensure output directory exists
test.beforeAll(async () => {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  console.log(`\n[Doc Images] Output directory: ${OUTPUT_DIR}\n`);
});

/**
 * 1. Basic 3D point cloud — clean view for landing page and basic tutorial
 */
test('Basic 3D point cloud', async ({ page }, testInfo) => {
  // First test bears Vite dev server cold-start compilation cost — warm up first
  testInfo.setTimeout(240000);
  const dataUrl = `${DATA_SERVER}/datasets/demos/lorenz.luxar.zarr`;

  // Warm up the Vite dev server by loading the page once (first load triggers module compilation)
  await page.goto(`${VIEWER_URL}/?debug`, { waitUntil: 'networkidle', timeout: 120000 });
  await page.waitForTimeout(2000);

  // Now load with data
  await page.goto(`${VIEWER_URL}/?src=${dataUrl}&debug`, { waitUntil: 'networkidle' });
  await waitForReady(page);
  await centerCamera(page);
  await hideUI(page);
  await renderFrames(page, 15);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'basic-3d-pointcloud.png'),
    type: 'png',
  });
  console.log('[Doc Images] basic-3d-pointcloud.png saved');
});

/**
 * 2. Viewer UI overview — all panels visible
 */
test('Viewer UI overview', async ({ page }) => {
  const dataUrl = `${DATA_SERVER}/datasets/demos/lorenz.luxar.zarr`;
  await page.goto(`${VIEWER_URL}/?src=${dataUrl}&debug`, { waitUntil: 'networkidle' });
  await waitForReady(page);
  await centerCamera(page);

  // Open panels: performance stats, rendering controls
  await page.keyboard.press('p');
  await page.waitForTimeout(300);
  await page.keyboard.press('r');
  await page.waitForTimeout(300);
  await page.keyboard.press('b');
  await page.waitForTimeout(300);

  await renderFrames(page, 10);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'viewer-ui-overview.png'),
    type: 'png',
  });
  console.log('[Doc Images] viewer-ui-overview.png saved');
});

/**
 * 3. nD navigation with dimension sliders — using a 5D dataset
 */
test('nD navigation with sliders', async ({ page }) => {
  // Use the 5D nD-transform bench (X,Y,Z + Frame + Channel) — lightweight
  // Points-based, renders reliably in headless Chrome without blocking the
  // main thread. Produced by `luxar demo run nd_transforms`.
  const dataUrl = `${DATA_SERVER}/datasets/demos/nd_transforms_bench.luxar.zarr`;
  await page.goto(`${VIEWER_URL}/?src=${dataUrl}&debug`, { waitUntil: 'networkidle' });
  await waitForReady(page);
  await centerCamera(page);

  // Open dimension sliders
  await page.keyboard.press('n');
  await page.waitForTimeout(500);

  // Navigate a few steps to show the slider in action
  await page.keyboard.press('1');
  await page.waitForTimeout(200);
  await page.keyboard.press(']');
  await page.waitForTimeout(300);
  await page.keyboard.press(']');
  await page.waitForTimeout(300);

  await renderFrames(page, 10);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'nd-navigation-sliders.png'),
    type: 'png',
  });
  console.log('[Doc Images] nd-navigation-sliders.png saved');
});

/**
 * 4. Gaussian splats scene
 */
test('Gaussian splats scene', async ({ page }, testInfo) => {
  testInfo.setTimeout(180000); // GSplat datasets need more time in headless Chrome
  const dataUrl = `${DATA_SERVER}/datasets/demos/gsplats_3d_tribolium_embryo.luxar.zarr`;
  await page.goto(`${VIEWER_URL}/?src=${dataUrl}&debug`, { waitUntil: 'networkidle' });
  await waitForReady(page);
  await centerCamera(page);
  await hideUI(page);

  // GSplats can block the main thread in headless Chrome, so avoid page.evaluate
  // and let the animation loop settle naturally
  await page.waitForTimeout(2000);

  await page.screenshot({
    path: path.join(OUTPUT_DIR, 'gsplats-scene.png'),
    type: 'png',
  });
  console.log('[Doc Images] gsplats-scene.png saved');
});

// Summary
test('Summary', async () => {
  const expected = [
    'basic-3d-pointcloud.png',
    'viewer-ui-overview.png',
    'nd-navigation-sliders.png',
    'gsplats-scene.png',
  ];

  console.log('\n========================================');
  console.log('Documentation Image Generation Complete');
  console.log('========================================');
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('\nImages:');
  for (const img of expected) {
    const exists = fs.existsSync(path.join(OUTPUT_DIR, img));
    console.log(`  ${exists ? '[OK]' : '[--]'} ${img}`);
  }
  console.log('');
});
