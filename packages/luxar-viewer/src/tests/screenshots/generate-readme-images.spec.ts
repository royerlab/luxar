/**
 * README Image Generator
 *
 * Generates screenshots of Luxar demos for README documentation.
 * Uses keyboard commands (F to center, scroll to zoom) for camera control.
 *
 * Usage:
 *   pnpm readme-images
 *
 * Prerequisites:
 *   - Run demo generators first: make run-demos
 *   - Servers started automatically by playwright config
 */

import { test } from '@playwright/test';
import * as path from 'path';
import * as fs from 'fs';

// Output directory for README images
const OUTPUT_DIR = path.resolve(__dirname, '../../../../../docs/images/readme');

// Server URLs
const DATA_SERVER = 'http://localhost:9876';
const VIEWER_URL = 'http://localhost:5173';

// Demo configurations
interface DemoConfig {
  name: string;
  datasetPath: string;
  filename: string;
  // Zoom adjustment: positive = zoom out, negative = zoom in
  zoomClicks?: number;
  // Extra wait time after loading
  extraWaitMs?: number;
  // For 4D+ demos: dimension key and steps to navigate
  dimensionNav?: { key: string; steps: number };
  // Exposure in log2 stops (default: 1.0 for all screenshots)
  exposure?: number;
}

// Default exposure boost for better screenshot visibility (log2 stops)
const DEFAULT_EXPOSURE = 1.0;

const DEMOS: DemoConfig[] = [
  {
    name: 'Lorenz Attractor',
    datasetPath: 'datasets/demos/lorenz.luxar.zarr',
    filename: 'lorenz-demo.png',
    zoomClicks: 3, // Zoom out
    extraWaitMs: 1000,
    exposure: 1.3, // ~2.5x boost for vibrant colors
  },
  {
    name: 'Mandelbulb',
    datasetPath: 'datasets/demos/mandelbulb.luxar.zarr',
    filename: 'mandelbulb-demo.png',
    zoomClicks: -22, // Zoom IN 2x more
    extraWaitMs: 2000,
    exposure: 3.6, // ~12x boost
  },
  {
    name: 'Spiral Galaxy',
    datasetPath: 'datasets/demos/spiral_galaxy.luxar.zarr',
    filename: 'spiral-galaxy-demo.png',
    zoomClicks: 4, // Zoom out
    extraWaitMs: 1500,
    exposure: 1.6, // ~3x boost for star visibility
  },
  {
    name: 'Zebrahub Multiome UMAP',
    datasetPath: 'datasets/demos/zebrahub_multiome_peak_umap.luxar.zarr',
    filename: 'zebrahub-umap-demo.png',
    zoomClicks: 0,
    extraWaitMs: 2000,
    exposure: 4.6, // ~25x boost
  },
  {
    name: 'Rainbow Sphere',
    datasetPath: 'datasets/demos/rainbow_sphere.luxar.zarr',
    filename: 'rainbow-sphere-demo.png',
    zoomClicks: 17, // Zoom out more to see full sphere
    extraWaitMs: 1500,
    exposure: 4.6, // ~25x boost
  },
];

/**
 * Wait for Luxar to initialize
 */
async function waitForLuxarReady(page: any, timeout = 60000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      return debug && debug.getState && debug.getState().initialized;
    },
    { timeout }
  );
}

/**
 * Wait for data to load (with points)
 */
async function waitForDataLoaded(page: any, timeout = 45000): Promise<void> {
  await page.waitForFunction(
    () => {
      const debug = (window as any).__luxarDebug;
      if (!debug || !debug.getState) return false;
      const state = debug.getState();
      return state && !state.isLoading && state.totalPoints > 0;
    },
    { timeout }
  );
}

/**
 * Center camera on scene (press F)
 */
async function centerCamera(page: any): Promise<void> {
  await page.keyboard.press('f');
  await page.waitForTimeout(800);
}

/**
 * Zoom using scroll wheel
 * @param clicks - positive = zoom out, negative = zoom in
 */
async function zoom(page: any, clicks: number): Promise<void> {
  if (clicks === 0) return;

  const canvas = await page.$('canvas');
  if (!canvas) return;

  const box = await canvas.boundingBox();
  if (!box) return;

  // Center of canvas
  const x = box.x + box.width / 2;
  const y = box.y + box.height / 2;

  const absClicks = Math.abs(clicks);
  const direction = clicks > 0 ? 100 : -100; // positive scroll = zoom out, negative = zoom in

  for (let i = 0; i < absClicks; i++) {
    await page.mouse.move(x, y);
    await page.mouse.wheel(0, direction);
    await page.waitForTimeout(150);
  }
  await page.waitForTimeout(300);
}

/**
 * Navigate dimension (for nD demos)
 */
async function navigateDimension(page: any, key: string, steps: number): Promise<void> {
  await page.keyboard.press(key);
  await page.waitForTimeout(300);

  for (let i = 0; i < steps; i++) {
    await page.keyboard.press(']');
    await page.waitForTimeout(400);
  }
  await page.waitForTimeout(500);
}

/**
 * Hide UI panels
 */
async function hideUI(page: any): Promise<void> {
  await page.keyboard.press('Escape');
  await page.waitForTimeout(100);
  await page.keyboard.press('Escape');
  await page.waitForTimeout(200);
}

/**
 * Render frames and wait
 */
async function renderAndWait(page: any, frames = 10): Promise<void> {
  for (let i = 0; i < frames; i++) {
    await page.evaluate(() => {
      (window as any).__luxarDebug?.renderOnce?.();
    });
    await page.waitForTimeout(50);
  }
  await page.waitForTimeout(300);
}

/**
 * Set exposure for brighter screenshots (log2 stops)
 */
async function setExposure(page: any, exposureStops: number): Promise<void> {
  await page.evaluate((stops: number) => {
    const debug = (window as any).__luxarDebug;

    // Access sceneManager via app (debug.sceneManager is not exposed directly)
    const sceneManager = debug?.app?.sceneManager;
    if (sceneManager?.updateExposure) {
      sceneManager.updateExposure(stops);
    }

    // Trigger render to apply exposure changes
    if (debug?.renderOnce) {
      debug.renderOnce();
    }
  }, exposureStops);

  // Wait for exposure changes to propagate
  await page.waitForTimeout(1000);
  // Render multiple frames to ensure exposure is fully applied
  await renderAndWait(page, 10);
}

// Ensure output directory exists
test.beforeAll(async () => {
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }
  console.log(`\n[README Images] Output directory: ${OUTPUT_DIR}\n`);
});

// Generate screenshots for each demo
for (const demo of DEMOS) {
  test(`Generate ${demo.name} screenshot`, async ({ page }) => {
    const dataUrl = `${DATA_SERVER}/${demo.datasetPath}`;
    const viewerUrl = `${VIEWER_URL}/?src=${dataUrl}&debug`;

    console.log(`[${demo.name}] Loading from ${dataUrl}`);

    // Navigate to viewer
    await page.goto(viewerUrl, { waitUntil: 'networkidle' });

    // Wait for initialization
    await waitForLuxarReady(page);
    console.log(`[${demo.name}] Viewer initialized`);

    // Wait for data
    await waitForDataLoaded(page);
    console.log(`[${demo.name}] Data loaded`);

    // Navigate dimension if needed
    if (demo.dimensionNav) {
      console.log(`[${demo.name}] Navigating dimension...`);
      await navigateDimension(page, demo.dimensionNav.key, demo.dimensionNav.steps);
    }

    // Center camera (F key)
    await centerCamera(page);
    console.log(`[${demo.name}] Camera centered`);

    // Zoom if needed (positive = out, negative = in)
    if (demo.zoomClicks && demo.zoomClicks !== 0) {
      await zoom(page, demo.zoomClicks);
      const direction = demo.zoomClicks > 0 ? 'out' : 'in';
      console.log(`[${demo.name}] Zoomed ${direction} (${Math.abs(demo.zoomClicks)} clicks)`);
    }

    // Hide UI
    await hideUI(page);

    // Extra wait
    if (demo.extraWaitMs) {
      await page.waitForTimeout(demo.extraWaitMs);
    }

    // Boost exposure for brighter screenshots - do this LAST before screenshot
    const exposure = demo.exposure ?? DEFAULT_EXPOSURE;
    await setExposure(page, exposure);
    console.log(`[${demo.name}] Exposure set to ${exposure} stops`);

    // Render frames after HDR change
    await renderAndWait(page, 15);

    // Take screenshot
    const outputPath = path.join(OUTPUT_DIR, demo.filename);
    await page.screenshot({
      path: outputPath,
      type: 'png',
    });

    console.log(`[${demo.name}] Screenshot saved: ${outputPath}`);
  });
}

// Summary
test('Summary', async () => {
  console.log('\n========================================');
  console.log('README Image Generation Complete');
  console.log('========================================');
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('\nImages:');
  for (const demo of DEMOS) {
    const exists = fs.existsSync(path.join(OUTPUT_DIR, demo.filename));
    console.log(`  ${exists ? '[OK]' : '[--]'} ${demo.filename}`);
  }
  console.log('');
});
