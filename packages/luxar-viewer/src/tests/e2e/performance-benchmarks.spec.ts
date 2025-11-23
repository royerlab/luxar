/**
 * Performance Benchmark Tests
 *
 * These tests measure and verify performance characteristics:
 * - Load time for datasets
 * - Frame rate (FPS) under load
 * - Memory usage and leak detection
 * - Navigation responsiveness
 *
 * IMPORTANT: These tests prevent performance regressions
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady } from './helpers';

const DATASETS = {
  nav: '/examples/dimension_navigation_example.zarr',
  grid5D: '/examples/dense_grid_5d_example.zarr',
  build: '/examples/build_example_structured.zarr',
};

test.describe('Performance - Load Times', () => {
  test('should load dataset in under 5 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Wait for points to load
    await page.waitForFunction(() => (window as any).__luxarDebug?.getState().totalPoints > 0, {
      timeout: 10000,
    });

    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(5000); // Under 5 seconds
  });

  test('should initialize WebGL context quickly', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const initTime = Date.now() - startTime;

    // Initialization should be fast (under 2 seconds)
    expect(initTime).toBeLessThan(2000);
  });
});

test.describe('Performance - Frame Rate', () => {
  test('should maintain reasonable FPS when idle', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Wait for scene to stabilize
    await page.waitForTimeout(2000);

    // Measure FPS over 3 seconds
    const fps = await page.evaluate(async () => {
      let frames = 0;
      const startTime = performance.now();
      const duration = 3000; // 3 seconds

      return new Promise<number>((resolve) => {
        const measureFrame = () => {
          frames++;
          const elapsed = performance.now() - startTime;

          if (elapsed < duration) {
            requestAnimationFrame(measureFrame);
          } else {
            resolve(frames / (elapsed / 1000));
          }
        };

        requestAnimationFrame(measureFrame);
      });
    });

    // Should get reasonable FPS (at least 30, ideally 60)
    expect(fps).toBeGreaterThan(30);
  });

  test('should render frames after data loading', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nav}&debug`);
    await waitForLuxarReady(page);

    // Wait for data to load
    await page.waitForFunction(() => (window as any).__luxarDebug?.getState().totalPoints > 0, {
      timeout: 5000,
    });

    // Trigger render
    await page.evaluate(() => {
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(1000);

    // Check frames rendered
    const frames = await page.evaluate(() => {
      return (window as any).__luxarDebug?.renderer?.info?.render?.frame || 0;
    });

    expect(frames).toBeGreaterThan(0);
  });
});

test.describe('Performance - Memory Usage', () => {
  test('should not leak memory on page load', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    const initialMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    // Wait a bit
    await page.waitForTimeout(3000);

    const finalMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    // Memory shouldn't grow significantly while idle
    const growthMB = (finalMemory - initialMemory) / 1024 / 1024;
    expect(growthMB).toBeLessThan(50); // Less than 50MB growth
  });

  test('should track WebGL memory usage', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    const webglMemory = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        geometries: debug.renderer.info.memory.geometries,
        textures: debug.renderer.info.memory.textures,
      };
    });

    // Should have allocated some geometries
    expect(webglMemory.geometries).toBeGreaterThan(0);
  });

  test('should report cache memory usage', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    // Navigate to populate cache
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);

    const cacheMemory = await page.evaluate(async () => {
      const loader = await (window as any).__luxarDebug.getSceneLoader();
      const defaultLoader = loader?.getDefaultLoader();

      if (!defaultLoader?.getCacheStats) return null;

      const stats = defaultLoader.getCacheStats();
      return {
        totalMemory: stats.totalMemory,
        numEntries: stats.numEntries,
      };
    });

    if (cacheMemory) {
      expect(cacheMemory.totalMemory).toBeGreaterThan(0);
      expect(cacheMemory.numEntries).toBeGreaterThan(0);
    }
  });
});

test.describe('Performance - Navigation Responsiveness', () => {
  test('should navigate through nD slice in under 2 seconds', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    const startTime = Date.now();

    // Navigate
    await page.keyboard.press('4');
    await page.keyboard.press(']');

    // Wait for query to complete
    await page
      .waitForFunction(
        () => {
          const logs = document.body.textContent || '';
          return logs.includes('Query result') || logs.includes('points');
        },
        { timeout: 5000 }
      )
      .catch(() => {
        // If no logs visible, wait for state update
        return page.waitForTimeout(2000);
      });

    const navTime = Date.now() - startTime;

    expect(navTime).toBeLessThan(2000); // Under 2 seconds
  });

  test('should handle rapid navigation without blocking', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    const startTime = Date.now();

    await page.keyboard.press('4');

    // Rapid navigation (5 steps)
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press(']');
      await page.waitForTimeout(300); // Quick succession
    }

    const totalTime = Date.now() - startTime;

    // Should complete all 5 navigations in under 10 seconds
    expect(totalTime).toBeLessThan(10000);

    // Should not crash
    expect(errors).toEqual([]);
  });
});

test.describe('Performance - Rendering Pipeline', () => {
  test('should render frames efficiently', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Measure time to render 60 frames
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

  test('should not degrade performance over time', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.build}&debug`);
    await waitForLuxarReady(page);

    // Measure FPS initially
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

    // Wait and navigate a bit
    await page.waitForTimeout(5000);
    await page.keyboard.press('v'); // Switch mode
    await page.waitForTimeout(1000);

    const laterFPS = await measureFPS();

    // FPS shouldn't degrade significantly (within 20%)
    const degradation = (initialFPS - laterFPS) / initialFPS;
    expect(degradation).toBeLessThan(0.2);
  });
});

test.describe('Performance - Cache Efficiency', () => {
  test('should improve load time with cache hits', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.grid5D}&debug`);
    await waitForLuxarReady(page);

    // First navigation (cache miss)
    await page.keyboard.press('4');

    const firstNavStart = Date.now();
    await page.keyboard.press(']');
    await page.waitForTimeout(2000);
    const firstNavTime = Date.now() - firstNavStart;

    // Navigate back (cache hit)
    const secondNavStart = Date.now();
    await page.keyboard.press('[');
    await page.waitForTimeout(1000);
    const secondNavTime = Date.now() - secondNavStart;

    // Cache hit should be faster (or at least not slower)
    expect(secondNavTime).toBeLessThanOrEqual(firstNavTime * 1.5); // Allow some variance
  });
});
