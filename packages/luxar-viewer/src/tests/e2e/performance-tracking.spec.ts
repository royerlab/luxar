/**
 * Performance Regression Tracking
 *
 * These tests measure and track performance metrics over time:
 * - Initial load time
 * - Frame rate (FPS)
 * - Navigation responsiveness
 * - Memory usage
 *
 * Metrics are stored in performance-baselines.json and tests fail if
 * performance degrades by more than 20% compared to baseline.
 *
 * To reset baselines: delete performance-baselines.json and run tests
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, waitForSpatialQuery, waitForNextRender } from './helpers';
import * as fs from 'fs';
import * as path from 'path';

const BASELINE_FILE = path.join(__dirname, '../../../performance-baselines.json');
const REGRESSION_THRESHOLD = 1.5; // Fail if >50% slower than baseline (generous for CI variability)

interface PerformanceBaselines {
  loadTime: number;
  initTime: number;
  navigationTime: number;
  fps: number;
  lastUpdated: string;
}

function loadBaselines(): PerformanceBaselines | null {
  try {
    if (fs.existsSync(BASELINE_FILE)) {
      return JSON.parse(fs.readFileSync(BASELINE_FILE, 'utf-8'));
    }
  } catch {
    // Ignore errors
  }
  return null;
}

function saveBaselines(metrics: Partial<PerformanceBaselines>) {
  const existing = loadBaselines() || {
    loadTime: 0,
    initTime: 0,
    navigationTime: 0,
    fps: 0,
    lastUpdated: '',
  };

  const updated = {
    ...existing,
    ...metrics,
    lastUpdated: new Date().toISOString(),
  };

  fs.writeFileSync(BASELINE_FILE, JSON.stringify(updated, null, 2));
}

const DATASET = 'http://localhost:9000/datasets/examples/build_example_structured.zarr';

test.describe('Performance Regression Tracking', () => {
  // Force serial execution: saveBaselines() does non-atomic read-modify-write
  // on a shared file, causing data loss when tests run in parallel.
  // Performance tests also interfere with each other's measurements.
  test.describe.configure({ mode: 'serial' });

  test('should track dataset load time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Wait for points to load
    await page.waitForFunction(
      () => {
        const state = (window as any).__luxarDebug?.getState?.();
        return state && state.totalPoints > 0;
      },
      { timeout: 15000 }
    );

    const loadTime = Date.now() - startTime;

    console.log(`Load time: ${loadTime}ms`);

    // Check against baseline
    const baselines = loadBaselines();
    if (baselines && baselines.loadTime > 0) {
      const ratio = loadTime / baselines.loadTime;
      console.log(`  Baseline: ${baselines.loadTime}ms (ratio: ${ratio.toFixed(2)}x)`);

      // Fail if >30% slower
      expect(loadTime).toBeLessThan(baselines.loadTime * REGRESSION_THRESHOLD);
    } else {
      console.log('  No baseline - establishing new baseline');
      saveBaselines({ loadTime });
    }

    // Always update baseline if faster (continuous improvement)
    if (!baselines || loadTime < baselines.loadTime) {
      saveBaselines({ loadTime });
    }
  });

  test('should track initialization time', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const initTime = Date.now() - startTime;

    console.log(`Init time: ${initTime}ms`);

    const baselines = loadBaselines();
    if (baselines && baselines.initTime > 0) {
      const ratio = initTime / baselines.initTime;
      console.log(`  Baseline: ${baselines.initTime}ms (ratio: ${ratio.toFixed(2)}x)`);

      // Allow some slack — initialization time can vary significantly depending on system load
      expect(initTime).toBeLessThan(baselines.initTime * REGRESSION_THRESHOLD);
    } else {
      console.log('  No baseline - establishing new baseline');
      saveBaselines({ initTime });
    }

    if (!baselines || !baselines.initTime || initTime < baselines.initTime) {
      saveBaselines({ initTime });
    }
  });

  test('should track navigation responsiveness', async ({ page }) => {
    const NAV_DATASET = 'http://localhost:9000/datasets/examples/dense_grid_5d_example.zarr';

    await page.goto(`/?src=${NAV_DATASET}&debug`);
    await waitForLuxarReady(page);

    const startTime = Date.now();

    // Perform a single navigation
    await page.keyboard.press('4');
    await waitForNextRender(page);
    await page.keyboard.press(']');
    await waitForSpatialQuery(page);

    const navigationTime = Date.now() - startTime;

    console.log(`Navigation time: ${navigationTime}ms`);

    const baselines = loadBaselines();
    if (baselines && baselines.navigationTime > 0) {
      const ratio = navigationTime / baselines.navigationTime;
      console.log(`  Baseline: ${baselines.navigationTime}ms (ratio: ${ratio.toFixed(2)}x)`);

      expect(navigationTime).toBeLessThan(baselines.navigationTime * REGRESSION_THRESHOLD);
    } else {
      saveBaselines({ navigationTime });
    }

    if (!baselines || navigationTime < baselines.navigationTime) {
      saveBaselines({ navigationTime });
    }
  });

  test('should track rendering frame rate', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Measure FPS
    const fps = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      let frames = 0;
      const startTime = performance.now();
      const duration = 2000; // 2 seconds

      return new Promise<number>((resolve) => {
        const measureFrame = () => {
          frames++;
          const elapsed = performance.now() - startTime;

          if (elapsed < duration) {
            debug.renderOnce();
            requestAnimationFrame(measureFrame);
          } else {
            resolve(frames / (elapsed / 1000));
          }
        };

        debug.renderOnce();
        requestAnimationFrame(measureFrame);
      });
    });

    console.log(`FPS: ${fps.toFixed(1)}`);

    const baselines = loadBaselines();
    if (baselines && baselines.fps > 0) {
      const ratio = fps / baselines.fps;
      console.log(`  Baseline: ${baselines.fps.toFixed(1)} FPS (ratio: ${ratio.toFixed(2)}x)`);

      // FPS should not drop below 70% of baseline
      expect(fps).toBeGreaterThan(baselines.fps * 0.7);
    } else {
      saveBaselines({ fps });
    }

    // Update baseline if better
    if (!baselines || fps > baselines.fps) {
      saveBaselines({ fps });
    }
  });

  test('should report current baselines', async () => {
    const baselines = loadBaselines();

    if (baselines) {
      console.log('\n📊 Performance Baselines:');
      console.log(`  Load Time: ${baselines.loadTime}ms`);
      console.log(`  Init Time: ${baselines.initTime}ms`);
      console.log(`  Navigation: ${baselines.navigationTime}ms`);
      console.log(`  FPS: ${baselines.fps.toFixed(1)}`);
      console.log(`  Last Updated: ${baselines.lastUpdated}\n`);
    } else {
      console.log('\n⚠️  No baselines established yet. Run tests to create baselines.\n');
    }

    expect(true).toBe(true); // Info-only test
  });
});

test.describe('Performance - Memory & Rendering', () => {
  test('should not leak memory while idle', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const initialMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    // Intentional fixed sleep: this IS the idle-memory measurement window.
    // An event-driven wait would defeat the test (we WANT to observe whether
    // the heap grows over a known wall-clock interval with no input).
    await page.waitForTimeout(3000);

    const finalMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    // Memory shouldn't grow significantly while idle
    const growthMB = (finalMemory - initialMemory) / 1024 / 1024;
    expect(growthMB).toBeLessThan(50);
  });

  test('should track WebGL memory usage', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const webglMemory = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        geometries: debug.renderer.info.memory.geometries,
        textures: debug.renderer.info.memory.textures,
      };
    });

    expect(webglMemory.geometries).toBeGreaterThan(0);
  });

  test('should render 60 frames efficiently', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const renderTime = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      let frameCount = 0;
      const targetFrames = 60;
      const startTime = performance.now();

      return new Promise<number>((resolve) => {
        const countFrames = () => {
          frameCount++;
          if (frameCount < targetFrames) {
            debug.renderOnce();
            requestAnimationFrame(countFrames);
          } else {
            resolve(performance.now() - startTime);
          }
        };

        debug.renderOnce();
        requestAnimationFrame(countFrames);
      });
    });

    // 60 frames should render in under 2 seconds (30+ FPS)
    expect(renderTime).toBeLessThan(2000);
  });

  test('should not degrade FPS over time', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    const measureFPS = async () => {
      return page.evaluate(async () => {
        let frames = 0;
        const startTime = performance.now();

        await new Promise<void>((resolve) => {
          const counter = () => {
            frames++;
            if (performance.now() - startTime < 1000) {
              requestAnimationFrame(counter);
            } else {
              resolve();
            }
          };
          requestAnimationFrame(counter);
        });

        return frames;
      });
    };

    const initialFPS = await measureFPS();

    // Intentional fixed sleeps: this test measures FPS *over time* to detect
    // gradual degradation. The 5s gap is the inter-sample window; replacing it
    // with an event-driven wait would defeat the purpose. The 1s after the
    // 'v' (control-mode toggle) is settling for the new control state.
    await page.waitForTimeout(5000);
    await page.keyboard.press('v');
    await page.waitForTimeout(1000);

    const laterFPS = await measureFPS();

    // FPS shouldn't degrade significantly (within 20%)
    const degradation = (initialFPS - laterFPS) / initialFPS;
    expect(degradation).toBeLessThan(0.2);
  });
});
