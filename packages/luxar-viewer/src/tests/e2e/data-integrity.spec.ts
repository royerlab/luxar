/**
 * Data Integrity E2E Tests
 *
 * Validates that loaded geometry data is internally consistent:
 * - Position, color, radius, sharpness attribute arrays are aligned (same count)
 * - Position values contain no NaN or Infinity
 * - Color values are finite and non-negative
 * - Radius values are non-negative
 * - DrawRange never exceeds buffer size
 * - Integrity is maintained after nD navigation
 *
 * These tests catch the most dangerous class of silent bug: misaligned
 * attribute arrays that cause users to see garbage data with no error message.
 */

import { test, expect } from './fixtures';
import {
  waitForLuxarReady,
  waitForPointsLoaded,
  waitForSpatialQueryOrThrow,
  validateSceneAttributes,
  focusCanvas,
  waitForNextRender,
} from './helpers';

const DATASET = 'http://localhost:9000/datasets/examples/sharpness_showcase_example.luxar.zarr';
const DATASET_5D = 'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr';

test.describe('Data Integrity - Attribute Alignment', () => {
  test('should have matching attribute counts across position/color/radius/sharpness', async ({
    page,
  }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const results = await validateSceneAttributes(page);
    expect(results.length).toBeGreaterThan(0);

    for (const cloud of results) {
      // All present attributes must have the same count
      expect(cloud.aligned).toBe(true);
    }
  });

  test('should have valid position values (no NaN, no Infinity)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const results = await validateSceneAttributes(page);
    expect(results.length).toBeGreaterThan(0);

    for (const cloud of results) {
      expect(cloud.hasNaN).toBe(false);
      expect(cloud.hasInfinity).toBe(false);
    }
  });

  test('should have valid color values (finite, non-negative)', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const colorCheck = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const issues: string[] = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'points' || !obj.geometry?.attributes?.aColor) return;
        const col = obj.geometry.attributes.aColor;
        // aColor is an InterleavedBufferAttribute; .array is the shared
        // interleaved buffer (positions/radii/colors/sharpness all live in
        // it). Read per-instance components with getX/getY/getZ so we
        // validate actual colors instead of a stride-misaligned mix.
        const sampleCount = Math.min(col.count, 1000);
        for (let i = 0; i < sampleCount; i++) {
          const r = col.getX(i);
          const g = col.getY(i);
          const b = col.getZ(i);
          if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b))
            issues.push(`${obj.name}: color[${i}] = [${r}, ${g}, ${b}] (not finite)`);
          if (r < 0 || g < 0 || b < 0)
            issues.push(`${obj.name}: color[${i}] = [${r}, ${g}, ${b}] (negative)`);
        }
      });

      return issues;
    });

    expect(colorCheck).toEqual([]);
  });

  test('should have non-negative radius values', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const radiusCheck = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const issues: string[] = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'points' || !obj.geometry?.attributes?.aRadius) return;
        const rad = obj.geometry.attributes.aRadius;
        // aRadius is an InterleavedBufferAttribute view; reading .array[i]
        // would hit unrelated attributes (positions, colors, sharpness).
        // Use getX(i) so we validate the actual per-instance radius.
        const instanceCount = rad.count;
        for (let i = 0; i < instanceCount; i++) {
          const v = rad.getX(i);
          if (v < 0) issues.push(`${obj.name}: radius[${i}] = ${v} (negative)`);
          if (!Number.isFinite(v)) issues.push(`${obj.name}: radius[${i}] = ${v} (not finite)`);
        }
      });

      return issues;
    });

    expect(radiusCheck).toEqual([]);
  });

  test('should have visible instance count <= attribute count for all points geometry', async ({
    page,
  }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10);

    const results = await validateSceneAttributes(page);
    expect(results.length).toBeGreaterThan(0);

    for (const cloud of results) {
      expect(cloud.visibleInstanceCount).toBeLessThanOrEqual(cloud.positionCount);
      expect(cloud.drawRangeCount).toBeLessThanOrEqual(cloud.positionCount);
    }
  });

  test('should maintain data integrity after nD navigation', async ({ page }) => {
    await page.goto(`/?src=${DATASET_5D}&debug`);
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1);

    // Verify initial integrity
    const before = await validateSceneAttributes(page);
    expect(before.length).toBeGreaterThan(0);
    for (const cloud of before) {
      expect(cloud.aligned).toBe(true);
      expect(cloud.hasNaN).toBe(false);
    }

    // Navigate to a different slice
    await focusCanvas(page);
    await page.keyboard.press('4');
    await waitForNextRender(page);
    await page.keyboard.press(']');
    await waitForSpatialQueryOrThrow(page);

    // Verify integrity after navigation
    const after = await validateSceneAttributes(page);
    expect(after.length).toBeGreaterThan(0);
    for (const cloud of after) {
      expect(cloud.aligned).toBe(true);
      expect(cloud.hasNaN).toBe(false);
      expect(cloud.visibleInstanceCount).toBeLessThanOrEqual(cloud.positionCount);
      expect(cloud.drawRangeCount).toBeLessThanOrEqual(cloud.positionCount);
    }
  });
});
