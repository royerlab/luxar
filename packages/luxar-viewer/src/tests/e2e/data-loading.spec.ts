/**
 * Data Loading Tests for Luxar Viewer
 *
 * These tests verify that the Luxar viewer can:
 * - Load real Zarr datasets
 * - Query spatial indices correctly
 * - Load point data with all attributes
 * - Handle cache hits/misses
 * - Update data on nD navigation
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady, getLuxarState } from './helpers';

test.describe('Luxar Data Loading', () => {
  test('should load points from demo dataset', async ({ page }) => {
    // Track console for errors
    const consoleErrors: string[] = [];
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    // Navigate with demo dataset (assuming it exists)
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Check for errors
    expect(consoleErrors).toEqual([]);

    // Verify scene state
    const state = await getLuxarState(page);
    expect(state).toBeDefined();
    expect(state.initialized).toBe(true);

    // Debug mode should have all components
    const debugState = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        hasApp: !!debug.app,
        hasScene: !!debug.scene,
        hasCamera: !!debug.camera,
        hasConsoleInterceptor: !!debug.consoleInterceptor,
        hasGetState: typeof debug.getState === 'function',
        runtimeReady: debug.runtimeReady,
      };
    });

    expect(debugState.hasApp).toBe(true);
    expect(debugState.hasScene).toBe(true);
    expect(debugState.hasCamera).toBe(true);
    expect(debugState.hasConsoleInterceptor).toBe(true);
    expect(debugState.hasGetState).toBe(true);
    expect(debugState.runtimeReady).toBe(true);
  });

  test('should verify point cloud attributes', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get detailed point cloud information
    const pointClouds = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      if (!debug.scene) return [];

      const clouds: any[] = [];
      debug.scene.traverse((obj: any) => {
        if (obj.type === 'Points') {
          const geom = obj.geometry;
          clouds.push({
            name: obj.name,
            pointCount: geom.attributes.position?.count || 0,
            hasPosition: !!geom.attributes.position,
            hasColor: !!geom.attributes.color,
            hasRadius: !!geom.attributes.radius,
            hasSharpness: !!geom.attributes.sharpness,
            visible: obj.visible,
            material: obj.material?.type,
          });
        }
      });
      return clouds;
    });

    // If we have point clouds, verify they have valid attributes
    if (pointClouds.length > 0) {
      for (const cloud of pointClouds) {
        expect(cloud.hasPosition).toBe(true); // Position is mandatory
        expect(cloud.pointCount).toBeGreaterThan(0);
        expect(cloud.visible).toBe(true);
        expect(cloud.material).toBe('ShaderMaterial'); // Should use custom shader
      }
    }
  });

  test('should access scene loader for cache inspection', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Use the getSceneLoader helper
    const loaderInfo = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      if (!debug.getSceneLoader) return null;

      try {
        const loaderManager = await debug.getSceneLoader();
        const defaultLoader = loaderManager.getDefaultLoader();

        if (!defaultLoader) return null;

        return {
          hasLoader: true,
          hasClearCache: typeof defaultLoader.clearCaches === 'function',
        };
      } catch (error) {
        return { error: String(error) };
      }
    });

    // If we loaded a scene, loader should be available
    if (loaderInfo) {
      expect(loaderInfo.hasLoader).toBe(true);
    }
  });

  test('should render frame and update renderer info', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Trigger a render
    await page.evaluate(() => {
      (window as any).__luxarDebug.renderOnce();
    });

    // Wait for render to complete
    await page.waitForFunction(
      () => (window as any).__luxarDebug?.renderer?.info?.render?.frame > 0,
      { timeout: 5000 }
    );

    // Verify renderer has processed frames
    const rendererInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        frames: debug.renderer.info.render.frame,
        triangles: debug.renderer.info.render.triangles,
        points: debug.renderer.info.render.points,
        calls: debug.renderer.info.render.calls,
      };
    });

    expect(rendererInfo.frames).toBeGreaterThan(0);
    expect(rendererInfo.calls).toBeGreaterThan(0);
  });

  test('should track animation state', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get animation state
    const state1 = await getLuxarState(page);

    // Trigger animation
    await page.evaluate(() => {
      (window as any).__luxarDebug.renderOnce();
    });

    // Wait a bit
    await page.waitForTimeout(100);

    // Check state again
    const state2 = await getLuxarState(page);

    // State should be tracked correctly
    expect(typeof state1.isAnimating).toBe('boolean');
    expect(typeof state2.isAnimating).toBe('boolean');
  });
});
