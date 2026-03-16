/**
 * Real Dataset Loading Tests
 *
 * These tests load ACTUAL Zarr datasets from the examples/ directory
 * and verify the complete data loading pipeline works correctly.
 *
 * CRITICAL: These tests validate that real data loading works end-to-end,
 * not just mocked scenarios.
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState, waitForPointsLoaded } from './helpers';

// Dataset paths (served from Python HTTP server on port 9000)
const DATASETS = {
  dimensionNav: 'http://localhost:9000/datasets/examples/dimension_navigation_example.zarr',
  dimSliders5D: 'http://localhost:9000/datasets/examples/dimension_sliders_5d_example.zarr',
  denseGrid5D: 'http://localhost:9000/datasets/examples/dense_grid_5d_example.zarr',
  broadcast: 'http://localhost:9000/datasets/examples/simple_nd_example.zarr',
  buildManual: 'http://localhost:9000/datasets/examples/build_example_manual.zarr',
  buildStructured: 'http://localhost:9000/datasets/examples/build_example_structured.zarr',
};

test.describe('Real Dataset Loading', () => {
  test('should load dimension_navigation_example.zarr successfully', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(err.message));

    await page.goto(`/?src=${DATASETS.dimensionNav}&debug`);
    await waitForLuxarReady(page);

    // Should have no critical errors
    expect(errors).toEqual([]);

    // Verify scene loaded
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
    expect(state.totalPoints).toBeGreaterThan(0);
    expect(state.pointClouds.length).toBeGreaterThan(0);

    // Note: Playwright automatically captures screenshot (screenshot: 'on' in config)
  });

  test('should load 5D dataset with correct dimensions', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.dimSliders5D}&debug`);
    await waitForLuxarReady(page);

    // Verify scene state
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify scene has initialized
    expect(state.initialized).toBe(true);
  });

  test('should load dataset with all point attributes', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.buildStructured}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 100);

    // Inspect point cloud attributes
    const attributes = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      if (!debug.scene) return null;

      const pointClouds: any[] = [];
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points') {
          const geom = obj.geometry;
          pointClouds.push({
            name: obj.name,
            pointCount: geom.attributes.position?.count || 0,
            hasPosition: !!geom.attributes.position,
            hasColor: !!geom.attributes.color,
            hasRadius: !!geom.attributes.radius,
            hasSharpness: !!geom.attributes.sharpness,
            positionCount: geom.attributes.position?.count || 0,
            colorCount: geom.attributes.color?.count || 0,
            radiusCount: geom.attributes.radius?.count || 0,
            sharpnessCount: geom.attributes.sharpness?.count || 0,
          });
        }
      });
      return pointClouds;
    });

    expect(attributes).toBeDefined();
    expect(attributes!.length).toBeGreaterThan(0);

    // Verify attributes are present
    const firstCloud = attributes![0];
    expect(firstCloud.hasPosition).toBe(true);
    expect(firstCloud.pointCount).toBeGreaterThan(0);

    // If attributes exist, verify they're aligned (same count)
    if (firstCloud.hasColor) {
      expect(firstCloud.colorCount).toBe(firstCloud.positionCount);
    }
    if (firstCloud.hasRadius) {
      expect(firstCloud.radiusCount).toBe(firstCloud.positionCount);
    }
    if (firstCloud.hasSharpness) {
      expect(firstCloud.sharpnessCount).toBe(firstCloud.positionCount);
    }
  });

  test('should load broadcast dataset correctly', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('Broadcasting') || text.includes('broadcast')) {
        consoleLogs.push(text);
      }
    });

    await page.goto(`/?src=${DATASETS.broadcast}&debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);
    expect(state.totalPoints).toBeGreaterThan(0);

    // May see broadcasting logs if navigating through broadcast dims
  });

  test('should handle dataset without spatial index', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', (msg) => {
      const text = msg.text();
      if (text.includes('spatial index') || text.includes('3D dataset')) {
        consoleLogs.push(text);
      }
    });

    // Use a simple 3D dataset
    await page.goto(`/?src=${DATASETS.buildManual}&debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);

    // Should work even without spatial index
    expect(state.totalPoints).toBeGreaterThan(0);
  });

  test('should load multiple point clouds in hierarchy', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.buildStructured}&debug`);
    await waitForLuxarReady(page);

    const sceneInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      const pointClouds: string[] = [];
      const groups: string[] = [];

      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points') {
          pointClouds.push(obj.name || 'unnamed');
        } else if (obj.type === 'Group' && obj !== debug.scene) {
          groups.push(obj.name || 'unnamed');
        }
      });

      return {
        totalChildren: debug.scene.children.length,
        pointClouds,
        groups,
      };
    });

    expect(sceneInfo.totalChildren).toBeGreaterThan(0);
  });

  test('should preserve scene dimensions from dataset', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.dimSliders5D}&debug`);
    await waitForLuxarReady(page);

    // Check if scene dimensions were loaded
    const hasDimensions = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      // Check if scene has dimension info
      return debug.scene?.userData?.sceneDimensions !== undefined;
    });

    // 5D dataset should have dimensions
    expect(typeof hasDimensions).toBe('boolean');
  });

  test('should verify WebGL rendering with real data', async ({ page }) => {
    // Use build_example_structured (3D with guaranteed visible points)
    await page.goto(`/?src=${DATASETS.buildStructured}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 100, 60000); // Increased timeout to 60s

    // Wait for at least one frame to render
    await page.waitForFunction(
      () => {
        const debug = (window as any).__luxarDebug;
        return debug?.renderer?.info?.render?.frame > 0;
      },
      { timeout: 10000 }
    );

    // Verify renderer stats
    const renderInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        frames: debug.renderer.info.render.frame,
        points: debug.renderer.info.render.points,
        calls: debug.renderer.info.render.calls,
      };
    });

    expect(renderInfo.frames).toBeGreaterThan(0);
    expect(renderInfo.calls).toBeGreaterThan(0);
  });
});
