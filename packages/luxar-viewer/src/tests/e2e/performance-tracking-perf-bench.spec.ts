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
 * performance degrades by more than 2× compared to the per-machine
 * baseline (or, for the hard-coded "60 frames in N ms" gate, the
 * absolute threshold).
 *
 * **Suite routing.** The filename ends in `-perf-bench.spec.ts` so the
 * default `playwright.config.ts` excludes this spec via its
 * `testIgnore: /.*perf-bench\.spec\.ts$/` rule. The spec runs under
 * the opt-in perf config instead:
 *
 *   pnpm test:perf:e2e
 *
 * (which uses `playwright.perf.config.ts`, serial workers, headed
 * Chrome by default, headless with `LUXAR_PERF_HEADLESS=1`).
 *
 * Rationale: the perf-tracking thresholds are inherently env-sensitive
 * — `pnpm test:e2e` runs in parallel on whatever machine load happens
 * to be present, so a 2× threshold or a 2000ms hard cap will trip on
 * a busy run and gate the merge for reasons unrelated to a real
 * regression. Treating perf-tracking as an opt-in suite keeps the
 * regression signal accessible without holding up the default merge
 * gate.
 *
 * To reset baselines: delete performance-baselines.json and run tests.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForSpatialQueryOrThrow, waitForNextRender } from './helpers';
import { PERF_DATA_BASE } from './perf-data-base';
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';

// `package.json` declares `"type": "module"`, so the CommonJS `__dirname`
// global is undefined at module load. Reconstruct it from `import.meta.url`.
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Per-machine baseline (gitignored). Each successful run rewrites it so the
// file tracks the rolling minimum/typical timing for the local hardware
// rather than a hardcoded reference. CI regression detection should emit
// metrics to a dashboard rather than depending on this file.
const BASELINE_FILE = path.join(__dirname, '../../../performance-baselines.json');
// Threshold is intentionally loose: timings drift across hardware, system
// load, and minor app changes that aren't real regressions. The file
// updates on every passing run, so a one-off drift won't permanently fail
// future runs.
const REGRESSION_THRESHOLD = 2.0;

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

const DATASET = `${PERF_DATA_BASE}/datasets/examples/build_example_structured.luxar.zarr`;

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

      expect(loadTime).toBeLessThan(baselines.loadTime * REGRESSION_THRESHOLD);
    } else {
      console.log('  No baseline - establishing new baseline');
    }

    // Always update on a passing run so the baseline tracks current hardware
    // rather than a stale committed value.
    saveBaselines({ loadTime });
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

      expect(initTime).toBeLessThan(baselines.initTime * REGRESSION_THRESHOLD);
    } else {
      console.log('  No baseline - establishing new baseline');
    }

    saveBaselines({ initTime });
  });

  test('should track navigation responsiveness', async ({ page }) => {
    const NAV_DATASET = `${PERF_DATA_BASE}/datasets/examples/dense_grid_5d_example.luxar.zarr`;

    await page.goto(`/?src=${NAV_DATASET}&debug`);
    await waitForLuxarReady(page);

    const startTime = Date.now();

    // Perform a single navigation
    await page.keyboard.press('4');
    await waitForNextRender(page);
    await page.keyboard.press(']');
    await waitForSpatialQueryOrThrow(page);

    const navigationTime = Date.now() - startTime;

    console.log(`Navigation time: ${navigationTime}ms`);

    const baselines = loadBaselines();
    if (baselines && baselines.navigationTime > 0) {
      const ratio = navigationTime / baselines.navigationTime;
      console.log(`  Baseline: ${baselines.navigationTime}ms (ratio: ${ratio.toFixed(2)}x)`);

      expect(navigationTime).toBeLessThan(baselines.navigationTime * REGRESSION_THRESHOLD);
    }

    saveBaselines({ navigationTime });
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
    }

    saveBaselines({ fps });
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
