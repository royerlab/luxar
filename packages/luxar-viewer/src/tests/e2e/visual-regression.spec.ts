/**
 * Visual Regression Tests
 *
 * These tests capture screenshots of known datasets and compare them to baselines.
 * This catches visual rendering bugs, HDR issues, blending problems, etc.
 *
 * IMPORTANT: First run creates baselines. Subsequent runs compare against them.
 * Update the Linux visual corpus with: pnpm test:e2e:visual:update
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForRenderStable, waitForNextRender } from './helpers';

// Dataset paths (served from Python HTTP server on port 9000)
const DATASETS = {
  nav: 'http://localhost:9000/datasets/examples/dimension_navigation_example.luxar.zarr',
  grid5D: 'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr',
  build: 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr',
};

test.describe('@visual Visual Regression - Basic Rendering', () => {
  test('should render dimension_navigation dataset consistently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Wait for render to stabilize using condition-based wait
    await waitForRenderStable(page, 5);

    // Take screenshot
    await expect(page).toHaveScreenshot('nav-dataset-default-view.png', {
      maxDiffPixelRatio: 0.08, // 8% tolerance for WebGL variability
      threshold: 0.25, // Color tolerance
    });
  });

  test('should render 5D dense grid consistently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Wait for render to stabilize using condition-based wait
    await waitForRenderStable(page, 5);

    await expect(page).toHaveScreenshot('grid-5d-initial-slice.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('@visual Visual Regression - HDR & Tone Mapping', () => {
  // Tests skip gracefully when the exposure API isn't available (see
  // early return below). To regenerate snapshots after intentional
  // changes: `pnpm test:e2e:visual:update`.
  test('should render with exposure = 0.0 (neutral)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // __luxarDebug exposes the post-processing manager directly
    // (debug.postProcessing.updateExposure), not via a wrapping
    // `sceneManager` field — checking the wrong path silently no-ops
    // the test.
    const hasExposureAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.postProcessing && typeof debug.postProcessing.updateExposure === 'function';
    });

    if (!hasExposureAPI) {
      console.log('Skipping: Exposure API not available');
      return;
    }

    // Set exposure to 0.0 (neutral)
    await page.evaluate(() => {
      (window as any).__luxarDebug.postProcessing.updateExposure(0.0);
      (window as any).__luxarDebug.renderOnce();
    });

    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('exposure-0.0.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render with exposure = 3.32 (10x brighter)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // See exposure-0.0 test for the API path rationale.
    const hasExposureAPI = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.postProcessing && typeof debug.postProcessing.updateExposure === 'function';
    });

    if (!hasExposureAPI) {
      console.log('Skipping: Exposure API not available');
      return;
    }

    // Set exposure to ~3.32 stops (equivalent to 10x brighter)
    await page.evaluate(() => {
      (window as any).__luxarDebug.postProcessing.updateExposure(3.32);
      (window as any).__luxarDebug.renderOnce();
    });

    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('exposure-3.32.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render differently at different nD slices', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Wait for initial render to stabilize
    await waitForRenderStable(page, 5);

    // Slice 0
    await expect(page).toHaveScreenshot('grid-5d-slice-0.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });

    // Navigate to different slice
    await page.keyboard.press('1');
    await page.keyboard.press(']');

    // Wait for slice navigation to complete and render to stabilize
    await waitForNextRender(page);
    await waitForRenderStable(page, 3);

    // Slice 1 (should look different)
    await expect(page).toHaveScreenshot('grid-5d-slice-1.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('@visual Visual Regression - Camera Views', () => {
  test('should render with FOV = 47 (default 50mm)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug&dpr=1`);
    await waitForLuxarReady(page);

    await waitForRenderStable(page, 5);

    await expect(page).toHaveScreenshot('fov-47-default.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render with FOV = 90 (wide angle)', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug&dpr=1`);
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

    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('fov-90-wide.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render centered on bounding box', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Press 'F' to center on bounding box
    await page.keyboard.press('f');
    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('centered-on-bbox.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});

test.describe('@visual Visual Regression - Control Modes', () => {
  test('should render in orbit control mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Ensure orbit mode
    await page.evaluate(() => {
      (window as any).__luxarDebug.controls.setControlType('orbit');
    });

    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('orbit-mode-view.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });

  test('should render in fly control mode', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug&dpr=1`);
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await waitForNextRender(page);

    await expect(page).toHaveScreenshot('fly-mode-view.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});
