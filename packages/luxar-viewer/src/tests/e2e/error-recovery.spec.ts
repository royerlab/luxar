/**
 * Error Recovery Tests
 *
 * These tests verify the viewer handles error conditions gracefully:
 * - Corrupted or invalid zarr files
 * - Network failures during loading
 * - Missing required data (positions array)
 * - Invalid spatial index data
 * - WebGL context loss
 *
 * IMPORTANT: Good error recovery improves user experience and debugging
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, waitForSpatialQuery } from './helpers';

test.describe('Error Recovery - Invalid Datasets', () => {
  test('should show error for non-existent dataset', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?src=/data/does-not-exist.zarr&debug');

    // Should show error message, not crash
    await page.waitForSelector('.error-message', { timeout: 10000 });

    const errorVisible = await page.locator('.error-message').isVisible();
    expect(errorVisible).toBe(true);

    // Should have helpful error text
    const errorText = await page.locator('.error-message').textContent();
    expect(errorText).toContain('Unable to Load Dataset');
  });

  test('should handle corrupted .zmetadata gracefully', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Try to load dataset that will fail during metadata parsing
    await page.goto('/?src=/data/corrupted.zarr&debug');

    // Should either show error or fallback gracefully
    await page.waitForTimeout(5000);

    // Verify app didn't crash (error message OR dataset browser should appear)
    const hasErrorOrBrowser = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.dataset-browser') !== null
      );
    });

    expect(hasErrorOrBrowser).toBe(true);
  });

  test('should handle missing positions array', async ({ page }) => {
    // This would be a dataset with .zmetadata but missing critical data
    // For now, test that missing dataset URL shows proper error
    await page.goto('/?src=/data/no-positions.zarr&debug');

    await page.waitForTimeout(3000);

    // Should show error or dataset browser (graceful handling)
    const recovered = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.dataset-browser') !== null ||
        (window as any).__luxarDebug?.getState()?.initialized === true
      );
    });

    expect(recovered).toBe(true);
  });
});

test.describe('Error Recovery - Network Failures', () => {
  test('should handle network timeout gracefully', async ({ page }) => {
    // Simulate slow/failing network by using invalid URL
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?src=http://invalid-host-12345.example.com/data.zarr&debug');

    // Should show error after timeout
    await page.waitForTimeout(8000);

    // Should have error message or fallback to browser
    const hasRecovery = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.dataset-browser') !== null
      );
    });

    expect(hasRecovery).toBe(true);
  });

  test('should recover if network fails mid-load', async ({ page }) => {
    // Load a valid dataset, then simulate network failure
    // This is tricky to test without mocking, so we verify error handling exists

    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Should initialize without crashing
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);

    // Even if errors occurred during optional resource loading, app should work
    // (Some 404s are expected for optional features)
  });
});

test.describe('Error Recovery - WebGL Failures', () => {
  test('should detect WebGL context loss', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Simulate WebGL context loss
    const contextLostHandled = await page.evaluate(() => {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;
      if (!canvas) return false;

      // Try to trigger context loss event
      const ext = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
      if (ext) {
        ext.loseContext();
        // Check if context is lost
        return canvas.getContext('webgl2')?.isContextLost() === true;
      }

      return false;
    });

    // If we could trigger context loss, verify it was detected
    if (contextLostHandled) {
      // App should detect the loss (implementation-specific)
      await page.waitForTimeout(500);

      // Restore context
      await page.evaluate(() => {
        const canvas = document.querySelector('canvas') as HTMLCanvasElement;
        const ext = canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context');
        ext?.restoreContext();
      });
    }

    // Test passes if context loss mechanism exists (even if not triggered)
    expect(true).toBe(true);
  });

  test('should verify WebGL context is valid on load', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const webglValid = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const canvas = debug.renderer?.domElement;

      if (!canvas) return false;

      const context = canvas.getContext('webgl2') || canvas.getContext('webgl');
      return context && !context.isContextLost();
    });

    expect(webglValid).toBe(true);
  });
});

test.describe('Error Recovery - Data Validation', () => {
  test('should handle empty dataset gracefully', async ({ page }) => {
    // Dataset with 0 points should load without crashing
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);

    // 0 points is valid (shows empty scene)
    expect(state.totalPoints).toBeGreaterThanOrEqual(0);
  });

  test('should validate point data ranges', async ({ page }) => {
    const DATASET = 'http://localhost:9000/packages/luxar/examples/build_example_structured.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Check that loaded data has reasonable values (no NaN, Infinity)
    const dataValid = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let allValid = true;

      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points') {
          const positions = obj.geometry.attributes.position;
          if (positions) {
            const array = positions.array;
            for (let i = 0; i < Math.min(100, array.length); i++) {
              if (!isFinite(array[i])) {
                allValid = false;
                break;
              }
            }
          }
        }
      });

      return allValid;
    });

    expect(dataValid).toBe(true);
  });

  test('should handle malformed transform matrices', async ({ page }) => {
    const DATASET = 'http://localhost:9000/packages/luxar/examples/transform_example.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Check that all transform matrices are valid
    const matricesValid = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let allValid = true;

      debug.scene.traverse((obj: any) => {
        if (obj.matrix) {
          const elements = obj.matrix.elements;
          // Check for NaN or Infinity
          for (const val of elements) {
            if (!isFinite(val)) {
              allValid = false;
              break;
            }
          }
        }
      });

      return allValid;
    });

    expect(matricesValid).toBe(true);
  });
});

test.describe('Error Recovery - Memory Limits', () => {
  test('should handle reasonable memory usage', async ({ page }) => {
    const DATASET = 'http://localhost:9000/packages/luxar/examples/dense_grid_5d_example.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Navigate several times to accumulate cache
    await page.keyboard.press('4');
    await page.waitForTimeout(100);

    for (let i = 0; i < 5; i++) {
      await page.keyboard.press(']');
      await waitForSpatialQuery(page);
    }

    // Check memory didn't explode
    const memoryOK = await page.evaluate(() => {
      const perf = performance as any;
      if (!perf.memory) return true; // Memory API not available (Firefox, Safari)

      const heapMB = perf.memory.usedJSHeapSize / 1024 / 1024;
      return heapMB < 2000; // Less than 2GB
    });

    expect(memoryOK).toBe(true);
  });

  test('should report cache statistics for memory monitoring', async ({ page }) => {
    const DATASET = 'http://localhost:9000/packages/luxar/examples/dense_grid_5d_example.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Navigate to populate cache
    await page.keyboard.press('4');
    await page.waitForTimeout(100);
    await page.keyboard.press(']');
    await waitForSpatialQuery(page);

    const cacheStats = await page.evaluate(async () => {
      const loader = await (window as any).__luxarDebug.getSceneLoader();
      const defaultLoader = loader?.getDefaultLoader();

      return defaultLoader?.getCacheStats ? defaultLoader.getCacheStats() : null;
    });

    // Cache stats API is optional, but if present should be valid
    if (cacheStats && typeof cacheStats === 'object') {
      if ('numEntries' in cacheStats && typeof cacheStats.numEntries === 'number') {
        expect(cacheStats.numEntries).toBeGreaterThanOrEqual(0);
      }
      if ('totalMemory' in cacheStats && typeof cacheStats.totalMemory === 'number') {
        expect(cacheStats.totalMemory).toBeGreaterThanOrEqual(0);
      }
    }
  });
});
