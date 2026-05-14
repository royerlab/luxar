/**
 * Basic Rendering Tests for Luxar Viewer
 *
 * These tests verify that the Luxar viewer can:
 * - Load without errors
 * - Initialize Three.js scene correctly
 * - Handle missing datasets gracefully
 * - Render basic demo datasets
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady } from './helpers';

test.describe('Luxar Basic Rendering', () => {
  test('should load viewer without errors', async ({ page }) => {
    // Track console errors
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Navigate to viewer with debug mode
    await page.goto('/?debug');

    // Wait for Luxar to initialize
    await waitForLuxarReady(page);

    // Should have no critical errors
    expect(errors).toEqual([]);

    // Verify debug interface is available
    const hasDebugInterface = await page.evaluate(() => {
      return typeof (window as any).__luxarDebug !== 'undefined';
    });
    expect(hasDebugInterface).toBe(true);

    // Verify basic state
    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug.getState();
    });

    expect(state).toBeDefined();
    expect(state.initialized).toBe(true);
    expect(state.cameraFov).toBeGreaterThan(0);
  });

  test('should handle missing dataset gracefully', async ({ page }) => {
    test.info().annotations.push({
      type: 'allow-console-errors',
      description: 'Bad-URL recovery path intentionally produces 404/502 responses.',
    });
    // Navigate with non-existent dataset
    await page.goto('/?src=/data/nonexistent.zarr&debug');

    // Wait for error handling (either error message or dataset browser appears)
    await page
      .waitForFunction(
        () => {
          const errorEl = document.querySelector('.error-message');
          const browserEl = document.querySelector('.dataset-browser');
          return (errorEl && errorEl.isConnected) || (browserEl && browserEl.isConnected);
        },
        { timeout: 10000 }
      )
      .catch(() => {
        // If neither appears, that's also a valid test result (app initialized)
      });

    // Should show error message (not crash)
    const errorVisible = await page
      .locator('.error-message')
      .isVisible()
      .catch(() => false);

    // Either error is shown OR we're in dataset browser
    const browserVisible = await page
      .locator('.dataset-browser')
      .isVisible()
      .catch(() => false);

    // At least one should be true (graceful handling)
    expect(errorVisible || browserVisible).toBe(true);
  });

  test('should initialize Three.js scene correctly', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Query scene state
    const sceneState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        hasScene: !!debug.scene,
        hasCamera: !!debug.camera,
        hasRenderer: !!debug.renderer,
        hasControls: !!debug.controls,
        cameraPosition: debug.camera
          ? { x: debug.camera.position.x, y: debug.camera.position.y, z: debug.camera.position.z }
          : null,
      };
    });

    // Verify Three.js components are initialized
    expect(sceneState.hasScene).toBe(true);
    expect(sceneState.hasCamera).toBe(true);
    expect(sceneState.hasRenderer).toBe(true);
    expect(sceneState.hasControls).toBe(true);

    // Camera should have a position
    expect(sceneState.cameraPosition).toBeDefined();
    expect(sceneState.cameraPosition?.z).toBeGreaterThan(0);
  });

  test('should render canvas element', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Check for canvas element (ID is 'app' per index.html and config)
    const canvas = page.locator('canvas#app');
    await expect(canvas).toBeVisible();

    // Canvas should have dimensions
    const dimensions = await canvas.evaluate((el) => ({
      width: (el as HTMLCanvasElement).width,
      height: (el as HTMLCanvasElement).height,
    }));

    expect(dimensions.width).toBeGreaterThan(0);
    expect(dimensions.height).toBeGreaterThan(0);
  });

  test('@visual should take screenshot without crashing', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Wait for initial render to complete. WebGLRenderer reports the
    // frame counter at `info.render.frame`; the unified Renderer base
    // (the WebGPURenderer path) reports it at `info.frame`. Check
    // both so the test is renderer-agnostic.
    await page.waitForFunction(
      () => {
        const info = (window as any).__luxarDebug?.renderer?.info;
        if (!info) return false;
        const frame = info.render?.frame ?? info.frame ?? 0;
        return frame > 0;
      },
      { timeout: 10000 }
    );

    // Take screenshot (this tests WebGL rendering stability)
    await expect(page).toHaveScreenshot('viewer-initial-state.png', {
      maxDiffPixelRatio: 0.1, // Allow 10% difference for first baseline
      threshold: 0.3, // Relaxed for WebGL variability
    });
  });
});
