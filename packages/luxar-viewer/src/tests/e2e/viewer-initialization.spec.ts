/**
 * Viewer Initialization Tests
 *
 * Tests the viewer startup without loading an external dataset (/?debug only).
 * Verifies core Three.js initialization, debug interface, and renderer.
 *
 * For tests with real datasets, see:
 * - `real-dataset-loading.spec.ts` - Real dataset loading
 * - `all-examples-smoke-test.spec.ts` - All examples smoke test
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState, waitForNextRender } from './helpers';

test.describe('Viewer Initialization', () => {
  test('should initialize viewer without dataset', async ({ page }) => {
    // Track console for errors
    const consoleErrors: string[] = [];
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    // Navigate without any dataset (just debug mode)
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
        if (obj.userData?.nodeType === 'points') {
          const geom = obj.geometry;
          clouds.push({
            name: obj.name,
            pointCount: geom.attributes.aCenter?.count || 0,
            hasPosition: !!geom.attributes.aCenter,
            hasColor: !!geom.attributes.aColor,
            hasRadius: !!geom.attributes.aRadius,
            hasSharpness: !!geom.attributes.aSharpness,
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
      { timeout: 10000 }
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

    // Wait for render
    await waitForNextRender(page);

    // Check state again
    const state2 = await getLuxarState(page);

    // State should be tracked correctly
    expect(typeof state1.isAnimating).toBe('boolean');
    expect(typeof state2.isAnimating).toBe('boolean');
  });
});
