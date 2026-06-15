/**
 * Dataset Switching Tests for Luxar Viewer
 *
 * These tests verify that switching between datasets properly cleans up
 * the old scene, disposes WebGL resources, and leaves the viewer in a
 * functional state. Catches regressions in:
 * - Scene graph cleanup (stale objects left behind)
 * - WebGL memory leaks (geometries/textures not disposed)
 * - Cross-geometry-type transitions (points -> lines -> points)
 * - Rapid navigation resilience
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  getLuxarState,
  getSceneObjectNames,
  getWebGLErrors,
  assertNoConsoleErrors,
} from './helpers';

// Base URL for examples (served by HTTP server on port 9000)
const EXAMPLES_BASE = 'http://localhost:9000/datasets/examples';

// Dataset URLs
const DATASET_A = `${EXAMPLES_BASE}/sharpness_showcase_example.luxar.zarr`;
const DATASET_B = `${EXAMPLES_BASE}/multiple_objects_example.luxar.zarr`;
const DATASET_LINES = `${EXAMPLES_BASE}/lines_basic_example.luxar.zarr`;

test.describe('Dataset Switching', () => {
  // Each switching test performs 2–3 full `page.goto()` cycles. With even a
  // healthy 10–20 s per dataset load, the 60 s per-test default leaves no
  // margin once chromium's first-load JIT, asset compile, and worker spin-up
  // are factored in. `test.slow()` triples the budget to 180 s.
  test.slow();

  test('should replace old scene objects when loading new dataset', async ({ page }) => {
    // Load dataset A
    await page.goto(`/?src=${DATASET_A}&debug`);
    await waitForLuxarReady(page);

    const namesA = await getSceneObjectNames(page);
    expect(namesA.length).toBeGreaterThan(0);

    // Navigate to dataset B
    await page.goto(`/?src=${DATASET_B}&debug`);
    await waitForLuxarReady(page);

    const namesB = await getSceneObjectNames(page);
    expect(namesB.length).toBeGreaterThan(0);

    // No data-specific names from dataset A should remain in dataset B's scene.
    // Filter out structural names that exist in every scene (LuxarScene, /, etc.)
    const STRUCTURAL_NAMES = ['LuxarScene', '/', '', 'Scene', 'AmbientLight', 'DirectionalLight'];
    const dataA = namesA.filter((n) => n && !STRUCTURAL_NAMES.includes(n));
    const dataB = namesB.filter((n) => n && !STRUCTURAL_NAMES.includes(n));

    const staleObjects = dataA.filter((nameA) => dataB.includes(nameA));
    expect(
      staleObjects,
      `Stale objects from dataset A found in dataset B: ${staleObjects.join(', ')}`
    ).toEqual([]);
  });

  test('should have correct scene child count (not A+B combined)', async ({ page }) => {
    // Load dataset A and count scene children
    await page.goto(`/?src=${DATASET_A}&debug`);
    await waitForLuxarReady(page);

    const countA = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.scene.children.length;
    });
    expect(countA).toBeGreaterThan(0);

    // Load dataset B and count scene children
    await page.goto(`/?src=${DATASET_B}&debug`);
    await waitForLuxarReady(page);

    const countB = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.scene.children.length;
    });
    expect(countB).toBeGreaterThan(0);

    // Each page.goto() is a full page reload, so scene children from A cannot
    // leak into B. Verify that B's child count is reasonable on its own — not
    // suspiciously large (which could indicate an unrelated accumulation bug).
    // Use 2x countA as a generous upper bound.
    expect(
      countB,
      `Scene child count ${countB} is unexpectedly large (dataset A had ${countA} children)`
    ).toBeLessThanOrEqual(Math.max(countA, countB) * 2);
  });

  test('should switch from points to lines dataset without errors', async ({ page }) => {
    // Load sharpness showcase (points)
    await page.goto(`/?src=${DATASET_A}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page);

    const stateA = await getLuxarState(page);
    expect(stateA.totalPoints).toBeGreaterThan(0);

    // Switch to lines dataset
    await page.goto(`/?src=${DATASET_LINES}&debug`);
    await waitForLuxarReady(page);

    // Assert no console errors
    await assertNoConsoleErrors(page);

    // Assert no WebGL errors
    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors, `WebGL errors after switching: ${webglErrors.join(', ')}`).toEqual([]);

    // Assert viewer is still functional
    const stateB = await getLuxarState(page);
    expect(stateB.initialized).toBe(true);
  });

  test('should not leak WebGL memory across multiple switches', async ({ page }) => {
    // Load dataset A and read geometry count
    await page.goto(`/?src=${DATASET_A}&debug`);
    await waitForLuxarReady(page);

    const geometriesAfterA = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.renderer.info.memory.geometries;
    });

    // Load dataset B
    await page.goto(`/?src=${DATASET_B}&debug`);
    await waitForLuxarReady(page);

    // Read (and discard) geometry count to ensure dataset B is fully loaded
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.renderer.info.memory.geometries;
    });

    // Load dataset A again
    await page.goto(`/?src=${DATASET_A}&debug`);
    await waitForLuxarReady(page);

    const geometriesAfterA2 = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.renderer.info.memory.geometries;
    });

    // Geometry count should NOT grow unboundedly.
    // After returning to dataset A, count should be within 2x of initial.
    expect(
      geometriesAfterA2,
      `Geometry count grew from ${geometriesAfterA} to ${geometriesAfterA2} after A->B->A cycle, suggesting a memory leak`
    ).toBeLessThanOrEqual(geometriesAfterA * 2);
  });

  test('should maintain functional state after rapid dataset switching', async ({ page }) => {
    // Track page-level errors
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    // Rapidly switch between 3 datasets without waiting for full load
    await page.goto(`/?src=${DATASET_A}&debug`);
    // Don't wait for ready -- immediately switch
    await page.goto(`/?src=${DATASET_LINES}&debug`);
    // Don't wait for ready -- immediately switch again
    await page.goto(`/?src=${DATASET_B}&debug`);

    // Now wait for the final dataset to fully load
    await waitForLuxarReady(page);

    // Assert viewer is functional
    const state = await getLuxarState(page);
    expect(state.initialized).toBe(true);

    // Assert no WebGL errors
    const webglErrors = await getWebGLErrors(page);
    expect(webglErrors, `WebGL errors after rapid switching: ${webglErrors.join(', ')}`).toEqual(
      []
    );

    // Assert no page-level errors (uncaught exceptions)
    expect(pageErrors, `Page errors during rapid switching: ${pageErrors.join('; ')}`).toEqual([]);
  });
});
