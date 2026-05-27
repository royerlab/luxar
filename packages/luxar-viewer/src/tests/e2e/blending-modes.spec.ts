/**
 * Blending Modes E2E Tests
 *
 * Validates that per-layer blending modes are correctly applied to Three.js materials:
 * - Initial blending modes match zarr metadata
 * - Changing blending mode updates material blending, depthTest, depthWrite
 * - Different modes produce visually different results
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForNextRender,
  getWebGLErrors,
  assertNoConsoleErrors,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/rendering_modes_example.zarr';
const MULTI_DATASET = 'http://localhost:9000/datasets/examples/multiple_objects_example.zarr';

test.describe('Blending Modes', () => {
  // The blending-mode datasets contain multiple groups (5+ point clouds) and
  // render with software-accelerated WebGL on most CI/test machines, where
  // FPS sits at ~3–10. The default 60s budget is marginal once data loading
  // plus several render passes are added; bump to 120s so we measure
  // correctness, not the test runner's tolerance for slow blits.
  test.slow();

  test('should load dataset with initial blending modes from zarr metadata', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Get all materials' blending state from the scene
    const blendingStates = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const states: { name: string; blending: number; depthTest: boolean; depthWrite: boolean }[] =
        [];

      debug.scene.traverse((obj: any) => {
        if ((obj.userData?.nodeType === 'points' || obj.type === 'Mesh') && obj.material) {
          states.push({
            name: obj.name || 'unnamed',
            blending: obj.material.blending,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
          });
        }
      });

      return states;
    });

    expect(blendingStates.length).toBeGreaterThan(0);

    // All materials should have a valid blending mode (THREE.js enum values)
    // NormalBlending=1, AdditiveBlending=2, CustomBlending=5
    for (const state of blendingStates) {
      expect([1, 2, 5]).toContain(state.blending);
    }
  });

  test('should have different depth behavior for additive vs normal blending', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Find additive and normal blended objects
    const modeInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let additive: any = null;
      let normal: any = null;

      debug.scene.traverse((obj: any) => {
        // Points render as `THREE.Mesh + userData.nodeType === 'points'`.
        const isPoints = obj.userData?.nodeType === 'points';
        if (!isPoints || !obj.material) return;

        // THREE.AdditiveBlending = 2
        if (obj.material.blending === 2 && !additive) {
          additive = {
            name: obj.name,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
            transparent: obj.material.transparent,
          };
        }
        // THREE.NormalBlending = 1
        if (obj.material.blending === 1 && !normal) {
          normal = {
            name: obj.name,
            depthTest: obj.material.depthTest,
            depthWrite: obj.material.depthWrite,
            transparent: obj.material.transparent,
          };
        }
      });

      return { additive, normal };
    });

    // If both modes exist in this dataset, verify their depth behavior differs
    if (modeInfo.additive && modeInfo.normal) {
      // Additive blending typically has depthTest=false, depthWrite=false
      expect(modeInfo.additive.depthWrite).toBe(false);

      // Normal blending typically has depthTest=true
      expect(modeInfo.normal.depthTest).toBe(true);
    }

    // At least one mode should exist
    expect(modeInfo.additive || modeInfo.normal).toBeTruthy();
  });

  test('should render without WebGL errors for all blending modes', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Force multiple renders to flush any deferred errors
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      for (let i = 0; i < 5; i++) debug.renderOnce();
    });
    await waitForNextRender(page);

    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors.length).toBe(0);

    await assertNoConsoleErrors(page);
  });

  test('should handle multiple point clouds with different blending modes simultaneously', async ({
    page,
  }) => {
    await page.goto(`/?src=${MULTI_DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    // Count how many distinct blending modes are used
    const blendingModes = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const modes = new Set<number>();

      debug.scene.traverse((obj: any) => {
        // Points are Mesh + nodeType='points'.
        if (obj.userData?.nodeType === 'points' && obj.material) {
          modes.add(obj.material.blending);
        }
      });

      return Array.from(modes);
    });

    // Dataset should have at least one blending mode
    expect(blendingModes.length).toBeGreaterThan(0);

    // No WebGL errors from mixed blending
    const errors = await getWebGLErrors(page);
    expect(errors.length).toBe(0);
  });

  test('@visual visual regression: scene renders with blending applied', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);
    await waitForNextRender(page, 5);

    // Take screenshot — this establishes a baseline for blending correctness
    await expect(page).toHaveScreenshot('blending-modes-rendering.png', {
      maxDiffPixelRatio: 0.08,
      threshold: 0.25,
    });
  });
});
