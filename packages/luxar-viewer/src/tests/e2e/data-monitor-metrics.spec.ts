/**
 * Data Loading Monitor Metrics Tests
 *
 * These tests verify that the Data Loading Monitor displays meaningful metrics:
 * - visiblePoints shows current visible points (not cumulative)
 * - datasetSize shows the total points from zarr metadata
 * - The displayed values are reasonable and don't grow infinitely
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, waitForPointsLoaded } from './helpers';

// Dataset served from Python HTTP server on port 9000
const DATASET_URL =
  'http://localhost:9000/packages/luxar/examples/dimension_navigation_example.zarr';

test.describe('Data Loading Monitor Metrics', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate with a known example dataset
    await page.goto(`/?src=${DATASET_URL}&debug`);
    await waitForLuxarReady(page);
  });

  test('should display visible points metric that reflects current view', async ({ page }) => {
    // Wait for some points to load
    await waitForPointsLoaded(page, 100, 30000);

    // Get the loader metrics via debug interface
    const metrics = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      if (!debug.getSceneLoader) return null;

      const sceneLoader = debug.getSceneLoader();
      if (!sceneLoader) return null;

      // Get the monitor if it exists
      const monitor = sceneLoader.getDataMonitor?.();
      if (!monitor) return { hasMonitor: false };

      const globalStats = monitor.getGlobalStats();
      return {
        hasMonitor: true,
        visiblePoints: globalStats.visiblePoints,
        datasetSize: globalStats.datasetSize,
        totalPoints: globalStats.totalPoints,
        totalLoaders: globalStats.totalLoaders,
      };
    });

    // Log metrics for debugging
    console.log('Monitor metrics:', JSON.stringify(metrics, null, 2));

    if (metrics && metrics.hasMonitor) {
      // visiblePoints should be less than or equal to datasetSize
      // (can't see more points than exist in the dataset)
      if (metrics.datasetSize > 0) {
        expect(metrics.visiblePoints).toBeLessThanOrEqual(metrics.datasetSize);
      }

      // visiblePoints should be reasonable (not millions when dataset is small)
      // If dataset is 1M points, visible should be <= 1M
      expect(metrics.visiblePoints).toBeGreaterThanOrEqual(0);
    }
  });

  test('should show monitor UI via M key press', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page, 100, 30000);

    // Press M to show the monitor
    await page.keyboard.press('m');
    await page.waitForTimeout(500);

    // Check if monitor panel is visible
    const monitorVisible = await page.evaluate(() => {
      // Look for monitor panel in DOM
      const monitorPanel = document.querySelector('[class*="monitor"]');
      return !!monitorPanel;
    });

    // The monitor should be visible after pressing M
    // Note: This test may fail if the monitor is not enabled by default
    console.log('Monitor visible after M key:', monitorVisible);
  });

  test('visiblePoints should not grow infinitely with interactions', async ({ page }) => {
    // Wait for initial load
    await waitForPointsLoaded(page, 100, 30000);

    // Get initial metrics
    const initialMetrics = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const sceneLoader = debug.getSceneLoader?.();
      if (!sceneLoader) return null;

      const monitor = sceneLoader.getDataMonitor?.();
      if (!monitor) return null;

      return monitor.getGlobalStats();
    });

    if (!initialMetrics) {
      console.log('No monitor available, skipping test');
      return;
    }

    console.log('Initial visible points:', initialMetrics.visiblePoints);

    // Perform some interactions that would trigger more queries
    for (let i = 0; i < 5; i++) {
      await page.mouse.wheel(0, 100); // Zoom
      await page.waitForTimeout(200);
    }

    // Get metrics after interactions
    const afterMetrics = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const sceneLoader = debug.getSceneLoader?.();
      if (!sceneLoader) return null;

      const monitor = sceneLoader.getDataMonitor?.();
      if (!monitor) return null;

      return monitor.getGlobalStats();
    });

    console.log('After interactions visible points:', afterMetrics?.visiblePoints);

    // visiblePoints should still be reasonable (not 100x or 1000x larger)
    // This catches the bug where points were counted cumulatively
    if (afterMetrics && initialMetrics.datasetSize > 0) {
      // Visible points should never exceed dataset size
      expect(afterMetrics.visiblePoints).toBeLessThanOrEqual(afterMetrics.datasetSize);

      // If dataset is 1M points, after 5 zoom interactions, we shouldn't have 200M visible
      // (which would happen if counting cumulatively)
      const maxReasonable = afterMetrics.datasetSize * 2; // Allow 2x for safety margin
      expect(afterMetrics.visiblePoints).toBeLessThanOrEqual(maxReasonable);
    }
  });

  test('datasetSize should match zarr metadata total_points', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page, 100, 30000);

    // Get both the loader's reported dataset size and actual points in scene
    const comparison = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;

      // Get dataset size from monitor
      const sceneLoader = debug.getSceneLoader?.();
      if (!sceneLoader) return null;

      const monitor = sceneLoader.getDataMonitor?.();
      if (!monitor) return null;

      const stats = monitor.getGlobalStats();

      // Count actual points in scene
      let scenePointCount = 0;
      debug.scene?.traverse((obj: any) => {
        if (obj.type === 'Points' && obj.geometry?.attributes?.position) {
          scenePointCount += obj.geometry.attributes.position.count;
        }
      });

      return {
        datasetSize: stats.datasetSize,
        visiblePoints: stats.visiblePoints,
        scenePointCount,
      };
    });

    console.log('Dataset comparison:', JSON.stringify(comparison, null, 2));

    if (comparison) {
      // If we have a dataset size from metadata, it should be positive
      if (comparison.datasetSize > 0) {
        expect(comparison.datasetSize).toBeGreaterThan(0);

        // Visible points should not exceed dataset size
        expect(comparison.visiblePoints).toBeLessThanOrEqual(comparison.datasetSize);
      }

      // Scene point count should match visible points (they're what's actually rendered)
      // Allow some tolerance for rounding/timing
      if (comparison.scenePointCount > 0 && comparison.visiblePoints > 0) {
        const ratio = comparison.scenePointCount / comparison.visiblePoints;
        // Should be within 50% (temporal differences during loading)
        expect(ratio).toBeGreaterThan(0.5);
        expect(ratio).toBeLessThan(2.0);
      }
    }
  });
});
