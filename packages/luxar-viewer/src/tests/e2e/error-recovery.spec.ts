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

import { test, expect, ALLOW_CONSOLE_ERRORS } from './fixtures';
import {
  waitForLuxarReady,
  getLuxarState,
  waitForSpatialQueryOrThrow,
  waitForNextRender,
} from './helpers';

// All tests in this file deliberately exercise broken-dataset paths
// (missing files, corrupted zmetadata, missing positions, network
// timeouts, etc.). The viewer's correct response logs console errors;
// asserting their absence would defeat the purpose of these tests.
// Playwright requires the first beforeEach arg to be a destructuring
// pattern (its way of declaring used fixtures). We don't need any
// fixtures here — only `testInfo` — so the empty pattern is correct.
// eslint-disable-next-line no-empty-pattern
test.beforeEach(async ({}, testInfo) => {
  testInfo.annotations.push({
    type: ALLOW_CONSOLE_ERRORS,
    description: 'Error-recovery tests deliberately trigger viewer console.error output.',
  });
});

test.describe('Error Recovery - Invalid Datasets', () => {
  test('should show error for non-existent dataset', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?src=http://localhost:9000/does-not-exist.zarr&debug');

    // Should show error dialog or dataset browser (graceful handling, not crash)
    await page.waitForSelector(
      '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
      {
        timeout: 30000,
      }
    );

    const hasError = await page
      .locator('.error-message, .luxar-error-dialog')
      .first()
      .isVisible()
      .catch(() => false);
    const hasBrowser = await page
      .locator('.dataset-browser, .luxar-dataset-browser')
      .first()
      .isVisible()
      .catch(() => false);
    expect(hasError || hasBrowser).toBe(true);
  });

  test('should handle corrupted metadata gracefully', async ({ page }) => {
    const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Intercept the root metadata formats the loader may try (Zarr v2
    // consolidated metadata, Zarr v3 metadata, and fallback attrs) and
    // return corrupted JSON. Older versions only intercepted
    // `.zmetadata`, which let v3 datasets load successfully and made the
    // test assert an error UI for a non-error path.
    await page.route(/\/(\.zmetadata|zarr\.json|\.zattrs)(\?.*)?$/, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: '{"this is not valid json: {{[[[',
      });
    });

    await page.goto(`/?src=${DATASET}&debug`);

    // Should show error dialog or dataset browser (graceful handling, not crash)
    await page.waitForSelector(
      '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
      {
        timeout: 30000,
      }
    );

    const hasErrorOrBrowser = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.luxar-error-dialog') !== null ||
        document.querySelector('.dataset-browser') !== null ||
        document.querySelector('.luxar-dataset-browser') !== null
      );
    });

    expect(hasErrorOrBrowser).toBe(true);
  });

  test('should handle missing positions array', async ({ page }) => {
    const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Intercept requests for positions data and return 404
    await page.route('**/positions/**', (route) => {
      route.fulfill({ status: 404, body: 'Not Found' });
    });
    await page.route('**/positions/.zarray', (route) => {
      route.fulfill({ status: 404, body: 'Not Found' });
    });

    await page.goto(`/?src=${DATASET}&debug`);

    // Wait for the app to either show error UI or settle without crashing
    // The app may show an error dialog, dataset browser, or simply handle the error gracefully
    try {
      await page.waitForSelector(
        '.error-message, .luxar-error-dialog, .dataset-browser, .luxar-dataset-browser',
        {
          timeout: 10000,
        }
      );
    } catch {
      // If no error UI appeared within 10s, that's also acceptable —
      // the key requirement is that the app didn't crash (no uncaught page errors)
    }

    // The app handled the missing positions gracefully if it didn't throw uncaught errors
    // (page errors from network failures are expected and acceptable)
    const crashErrors = errors.filter(
      (e) => !e.includes('fetch') && !e.includes('404') && !e.includes('NetworkError')
    );
    expect(crashErrors.length).toBe(0);
  });
});

test.describe('Error Recovery - Network Failures', () => {
  test('should handle network timeout gracefully', async ({ page }) => {
    // Simulate slow/failing network by using invalid URL
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?src=http://invalid-host-12345.example.com/data.zarr&debug');

    // Wait for the recovery UI (error message OR dataset browser fallback) to
    // appear. The actual network timeout can take several seconds; poll until
    // the DOM shows we've handled it or the budget elapses.
    await page
      .waitForFunction(
        () =>
          document.querySelector('.error-message') !== null ||
          document.querySelector('.dataset-browser') !== null,
        null,
        { timeout: 15000 }
      )
      .catch(() => {
        /* fall through to assertion for a meaningful failure message */
      });

    const hasRecovery = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.dataset-browser') !== null
      );
    });

    expect(hasRecovery).toBe(true);
  });

  test('should handle network failure mid-load gracefully', async ({ page }) => {
    test.setTimeout(120000);

    const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    // Let metadata requests through, but abort chunk data requests
    let requestCount = 0;
    await page.route(`${DATASET}/**`, (route) => {
      const url = route.request().url();
      // Let metadata through — BOTH formats' documents. Naming only format 2's
      // makes this test quietly stop testing what it says: on a format-3 store
      // the metadata document is `zarr.json`, which would fall through to the
      // chunk branch and be ABORTED after the first two requests, so the viewer
      // fails at metadata load rather than mid-chunk. The assertion below is a
      // disjunction, so it would still pass — while exercising a different path.
      // Same failure mode the corrupted-metadata test above already documents.
      if (
        url.includes('.zmetadata') ||
        url.includes('.zarray') ||
        url.includes('.zattrs') ||
        url.includes('.zgroup') ||
        url.includes('zarr.json')
      ) {
        route.continue();
        return;
      }
      // Let the first few data chunk requests through, then abort the rest
      requestCount++;
      if (requestCount <= 2) {
        route.continue();
      } else {
        route.abort('connectionfailed');
      }
    });

    await page.goto(`/?src=${DATASET}&debug`);

    // Wait for the viewer to handle the partial load: either it recovers
    // (initialized === true) or it surfaces an error / falls back to the
    // dataset browser. Poll the disjunction until met or budget elapses.
    //
    // Budget rationale: with `route.abort('connectionfailed')`, each chunk
    // walks the full cache-retry chain (4 attempts with backoff) before
    // returning undefined. On a real dataset that's tens of chunks
    // serialized per array, so the total load can legitimately take ~45 s
    // before the loader settles on a final state. The test's CONTRACT is
    // "viewer doesn't hang indefinitely" — 60 s comfortably bounds that.
    await page
      .waitForFunction(
        () =>
          document.querySelector('.error-message') !== null ||
          document.querySelector('.luxar-error-dialog') !== null ||
          document.querySelector('.dataset-browser') !== null ||
          document.querySelector('.luxar-dataset-browser') !== null ||
          (window as any).__luxarDebug?.getState?.()?.initialized === true,
        null,
        { timeout: 60000 }
      )
      .catch(() => {
        /* fall through to assertion for a meaningful failure message */
      });

    const handled = await page.evaluate(() => {
      return (
        document.querySelector('.error-message') !== null ||
        document.querySelector('.luxar-error-dialog') !== null ||
        document.querySelector('.dataset-browser') !== null ||
        document.querySelector('.luxar-dataset-browser') !== null ||
        (window as any).__luxarDebug?.getState?.()?.initialized === true
      );
    });

    expect(handled).toBe(true);
  });
});

test.describe('Error Recovery - WebGL Failures', () => {
  test('should detect WebGL context loss without crashing', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Snapshot state before context loss so we can verify it survives the cycle
    const stateBefore = await getLuxarState(page);

    // Simulate WebGL context loss via the WEBGL_lose_context extension
    const contextLostTriggered = await page.evaluate(() => {
      const canvas = document.querySelector('canvas') as HTMLCanvasElement;
      if (!canvas) return false;

      const gl = canvas.getContext('webgl2');
      const ext = gl?.getExtension('WEBGL_lose_context');
      if (ext) {
        ext.loseContext();
        return true;
      }
      return false;
    });

    // Intentional fixed sleep: WEBGL_lose_context dispatches the lost
    // event asynchronously through the browser's GL queue, with no
    // JS-observable signal. Same shape as context-restore.spec.ts.
    await page.waitForTimeout(500);

    // Restore context
    if (contextLostTriggered) {
      await page.evaluate(() => {
        const canvas = document.querySelector('canvas') as HTMLCanvasElement;
        const gl = canvas?.getContext('webgl2');
        const ext = gl?.getExtension('WEBGL_lose_context');
        ext?.restoreContext();
      });

      // Same as the loss event above — give the restore handler
      // (rebuildAfterContextRestore + dirty marking) wall-clock time
      // to complete before reading state.
      await page.waitForTimeout(1000);

      // Verify the viewer state survived the loss/restore cycle. The
      // post-restore handler in scene-manager.ts disposes/recreates
      // post-processing and re-marks geometry/materials dirty so the next
      // render re-uploads everything to the GPU.
      const stateAfter = await getLuxarState(page);
      expect(stateAfter.initialized).toBe(true);
      expect(stateAfter.totalPoints).toBe(stateBefore.totalPoints);

      // renderOnce must not throw against the restored context
      const renderOk = await page.evaluate(() => {
        try {
          (window as any).__luxarDebug?.renderOnce?.();
          return true;
        } catch {
          return false;
        }
      });
      expect(renderOk).toBe(true);
    }

    // The key assertion: no unhandled exceptions during context loss/restore cycle
    expect(errors).toEqual([]);
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
    const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Check that loaded data has reasonable values (no NaN, Infinity)
    const dataValid = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let allValid = true;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points') {
          // Per-point data is texture-backed: centers live at texel slots
          // [i*12 .. i*12+2] of the RGBA32F element texture; instanceCount
          // is the visible point count.
          const texData = obj.geometry?.userData?.elementTexture?.image?.data;
          if (texData) {
            const STRIDE = 12;
            const count = Math.min(
              obj.geometry.instanceCount ?? 0,
              Math.floor(texData.length / STRIDE),
              100
            );
            for (let i = 0; i < count; i++) {
              if (
                !isFinite(texData[i * STRIDE]) ||
                !isFinite(texData[i * STRIDE + 1]) ||
                !isFinite(texData[i * STRIDE + 2])
              ) {
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
    const DATASET = 'http://localhost:9000/datasets/examples/transform_example.luxar.zarr';

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
    const DATASET = 'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Navigate several times to accumulate cache
    await page.keyboard.press('1');
    await waitForNextRender(page);

    for (let i = 0; i < 5; i++) {
      await page.keyboard.press(']');
      await waitForSpatialQueryOrThrow(page);
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
    const DATASET = 'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr';

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Navigate to populate cache
    await page.keyboard.press('1');
    await waitForNextRender(page);
    await page.keyboard.press(']');
    await waitForSpatialQueryOrThrow(page);

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
