/**
 * Data Loading Monitor Metrics Tests
 *
 * These tests verify that the viewer tracks data metrics correctly:
 * - totalPoints reflects actually loaded/visible points
 * - Points count doesn't grow infinitely with interactions
 * - Scene point count matches state-reported totals
 * - Monitor UI can be toggled with M key
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  focusCanvas,
  waitForNextRender,
} from './helpers';

// Dataset served from Python HTTP server on port 9000
// Use build_example_structured - it's 3D with guaranteed visible points
const DATASET_URL = 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr';
const LADDER_URL =
  'http://localhost:9000/packages/luxar-viewer/tests/fixtures/test_mesh_reveal_ladder.luxar.zarr';

test.describe('Data Loading Monitor Metrics', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate with a known example dataset
    await page.goto(`/?src=${DATASET_URL}&debug`);
    await waitForLuxarReady(page);
  });

  test('should display visible points metric that reflects current view', async ({ page }) => {
    // Wait for some points to load
    await waitForPointsLoaded(page, 100, 30000);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Count actual visible point instances in the scene to verify consistency.
    const scenePointCount = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let count = 0;
      debug.scene?.traverse((obj: any) => {
        // Per-point data is texture-backed: the RGBA32F element texture
        // holds 12 floats per point and instanceCount is the visible point
        // count (the texel buffer may be over-allocated by the GPU pool).
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (obj.userData?.nodeType === 'points' && texData) {
          const texelCapacity = Math.floor(texData.length / 12);
          const instanceCount = obj.geometry.isInstancedBufferGeometry
            ? obj.geometry.instanceCount
            : texelCapacity;
          count += Math.min(instanceCount, texelCapacity);
        }
      });
      return count;
    });

    // Scene point count should be positive and match state roughly
    expect(scenePointCount).toBeGreaterThan(0);
    // State totalPoints and scene count should agree closely. Both are
    // single-frame reads of the same (non-LOD) scene, so they should match
    // within loading-timing jitter. Substitutive-LOD double-counting (which
    // would push these apart by a ~K× factor) is covered by the unit suite
    // and by lod-group.spec.ts; here a tight band catches gross divergence.
    if (state.totalPoints > 0 && scenePointCount > 0) {
      const ratio = scenePointCount / state.totalPoints;
      expect(ratio).toBeGreaterThan(0.8);
      expect(ratio).toBeLessThan(1.25);
    }
  });

  test('debug surface wires live FPS sampling and additive ladder state', async ({ page }) => {
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.app.adaptiveDPRManager.setEnabled(true);
      debug.renderOnce();
    });
    await page.waitForFunction(
      () => {
        const debug = (window as unknown as { __luxarDebug?: Record<string, unknown> })
          .__luxarDebug;
        return debug?.fpsSamplingEnabled === true && typeof debug.fps === 'number';
      },
      undefined,
      { timeout: 15000 }
    );

    await page.goto(`/?src=${LADDER_URL}&debug`);
    await waitForLuxarReady(page);
    await page.waitForFunction(
      () => {
        const debug = (
          window as unknown as {
            __luxarDebug?: { getState?: () => { meshNodes?: Array<Record<string, unknown>> } };
          }
        ).__luxarDebug;
        return debug
          ?.getState?.()
          .meshNodes?.some(
            (node) =>
              typeof node.totalLODCount === 'number' &&
              node.totalLODCount > 1 &&
              typeof node.committedLadderComplete === 'boolean'
          );
      },
      undefined,
      { timeout: 60000 }
    );

    const ladder = await page.evaluate(() => {
      const state = (window as any).__luxarDebug.getState();
      return state.meshNodes.find((node: any) => node.totalLODCount > 1);
    });
    expect(ladder.loadedLODCount).toBeGreaterThan(0);
    expect(ladder.totalLODCount).toBe(4);
    expect(typeof ladder.lastAllResident).toBe('boolean');
  });

  test('should show monitor UI via M key press', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page, 100, 30000);

    // Focus canvas so M key reaches the app
    await focusCanvas(page);

    // Press M to show the monitor
    await page.keyboard.press('m');
    await waitForNextRender(page);

    // Check if monitor panel is visible
    const monitorVisible = await page.evaluate(() => {
      // Look for monitor panel in DOM (various class patterns)
      const monitorPanel =
        document.querySelector('[class*="monitor"]') ||
        document.querySelector('[class*="data-monitor"]');
      return !!monitorPanel;
    });

    // The monitor should be visible after pressing M
    expect(monitorVisible).toBe(true);
  });

  test('totalPoints should not grow infinitely with interactions', async ({ page }) => {
    // Wait for initial load
    await waitForPointsLoaded(page, 100, 30000);

    // Get initial point count
    const initialState = await getLuxarState(page);
    const initialPoints = initialState.totalPoints;
    expect(initialPoints).toBeGreaterThan(0);

    // Perform some interactions that trigger re-renders. The pacing
    // sleep between wheel events is intentional: the test simulates
    // discrete user-driven zoom events rather than a single tight burst,
    // so the viewer's per-event damping/render path runs each time.
    await focusCanvas(page);
    for (let i = 0; i < 5; i++) {
      await page.mouse.wheel(0, 100); // Zoom
      await page.waitForTimeout(200);
    }

    // Get point count after interactions
    const afterState = await getLuxarState(page);

    // Points should not have grown unboundedly (cumulative counting bug)
    // For a 3D dataset without nD slicing, zoom doesn't change point count
    // Allow generous 3x margin for timing/loading variance
    expect(afterState.totalPoints).toBeLessThanOrEqual(initialPoints * 3);
    expect(afterState.totalPoints).toBeGreaterThanOrEqual(0);
  });

  test('scene point count should match state totalPoints', async ({ page }) => {
    // Wait for points to load
    await waitForPointsLoaded(page, 100, 30000);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Count actual visible point instances in the Three.js scene.
    const scenePointCount = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let count = 0;
      debug.scene?.traverse((obj: any) => {
        // Per-point data is texture-backed: the RGBA32F element texture
        // holds 12 floats per point and instanceCount is the visible point
        // count (the texel buffer may be over-allocated by the GPU pool).
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (obj.userData?.nodeType === 'points' && texData) {
          const texelCapacity = Math.floor(texData.length / 12);
          const instanceCount = obj.geometry.isInstancedBufferGeometry
            ? obj.geometry.instanceCount
            : texelCapacity;
          count += Math.min(instanceCount, texelCapacity);
        }
      });
      return count;
    });

    expect(scenePointCount).toBeGreaterThan(0);

    // The scene point count and state totalPoints should match closely for
    // this non-LOD dataset (small async-loading jitter only). A loose band
    // previously let multiplicative reporting bugs slip through; LOD-specific
    // divergence is covered by the unit suite + lod-group.spec.ts.
    if (scenePointCount > 0 && state.totalPoints > 0) {
      const ratio = scenePointCount / state.totalPoints;
      expect(ratio).toBeGreaterThan(0.8);
      expect(ratio).toBeLessThan(1.25);
    }
  });
});
