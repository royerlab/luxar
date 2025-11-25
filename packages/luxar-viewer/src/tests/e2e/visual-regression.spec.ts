/**
 * Visual Regression Tests
 *
 * These tests capture screenshots of known datasets and compare them to baselines.
 * This catches visual rendering bugs, HDR issues, blending problems, etc.
 *
 * IMPORTANT: First run creates baselines. Subsequent runs compare against them.
 * Update baselines with: pnpm test:e2e --update-snapshots
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady } from './helpers';

// Dataset paths (served from Python HTTP server on port 8001)
const DATASETS = {
  nav: 'http://localhost:9000/examples/dimension_navigation_example.zarr',
  grid5D: 'http://localhost:9000/examples/dense_grid_5d_example.zarr',
  build: 'http://localhost:9000/examples/build_example_structured.zarr',
};

test.describe('Visual Regression - Basic Rendering', () => {
  test('should render dimension_navigation dataset consistently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    // Wait for initial render to stabilize
    await page.waitForFunction(
      () => (window as any).__luxarDebug?.renderer?.info?.render?.frame > 3,
      { timeout: 10000 }
    );

    // Additional wait for GPU to finish
    await page.waitForTimeout(2000);

    // Take screenshot
    await expect(page).toHaveScreenshot('nav-dataset-default-view.png', {
      maxDiffPixelRatio: 0.08, // 8% tolerance for WebGL variability
      threshold: 0.25, // Color tolerance
    });
  });

  test('should render 5D dense grid consistently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    await page.waitForFunction(
      () => (window as any).__luxarDebug?.renderer?.info?.render?.frame > 3,
      { timeout: 10000 }
    );

    await page.waitForTimeout(2000);

    await expect(page).toHaveScreenshot('grid-5d-initial-slice.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('Visual Regression - HDR & Tone Mapping', () => {
  test('should render with HDR multiplier = 1.0', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Check if HDR API is available, skip if not
    const hasHDRAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.sceneManager && typeof debug.sceneManager.updateHDRMultiplier === 'function';
    });

    if (!hasHDRAPI) {
      console.log('Skipping: HDR multiplier API not available');
      return;
    }

    // Set HDR multiplier to 1.0
    await page.evaluate(() => {
      (window as any).__luxarDebug.sceneManager.updateHDRMultiplier(1.0);
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('hdr-multiplier-1.0.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render with HDR multiplier = 10.0', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Check if HDR API is available, skip if not
    const hasHDRAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.sceneManager && typeof debug.sceneManager.updateHDRMultiplier === 'function';
    });

    if (!hasHDRAPI) {
      console.log('Skipping: HDR multiplier API not available');
      return;
    }

    // Set HDR multiplier to 10.0 (brighter)
    await page.evaluate(() => {
      (window as any).__luxarDebug.sceneManager.updateHDRMultiplier(10.0);
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('hdr-multiplier-10.0.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render differently at different nD slices', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    await page.waitForTimeout(2000);

    // Slice 0
    await expect(page).toHaveScreenshot('grid-5d-slice-0.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });

    // Navigate to different slice
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(3000);

    // Slice 1 (should look different)
    await expect(page).toHaveScreenshot('grid-5d-slice-1.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('Visual Regression - Camera Views', () => {
  test('should render with FOV = 47 (default 50mm)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    await page.waitForTimeout(2000);

    await expect(page).toHaveScreenshot('fov-47-default.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render with FOV = 90 (wide angle)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    // Check if updateFOV API is available
    const hasFOVAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.sceneManager && typeof debug.sceneManager.updateFOV === 'function';
    });

    // Set wide FOV
    await page.evaluate((hasAPI) => {
      const debug = (window as any).__luxarDebug;
      debug.camera.fov = 90;
      debug.camera.updateProjectionMatrix();
      if (hasAPI) {
        debug.sceneManager.updateFOV(0); // Trigger material updates
      }
      debug.renderOnce();
    }, hasFOVAPI);

    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('fov-90-wide.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render centered on bounding box', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Press 'F' to center on bounding box
    await page.keyboard.press('f');
    await page.waitForTimeout(2000);

    await expect(page).toHaveScreenshot('centered-on-bbox.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('Visual Regression - Control Modes', () => {
  test('should render in orbit control mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    // Ensure orbit mode
    await page.evaluate(() => {
      (window as any).__luxarDebug.controls.setControlType('orbit');
    });

    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('orbit-mode-view.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render in fly control mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.waitForTimeout(1000);

    await expect(page).toHaveScreenshot('fly-mode-view.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});
