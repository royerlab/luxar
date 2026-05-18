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

const DATASET = 'http://localhost:9000/datasets/examples/sharpness_showcase_example.zarr';
const DATASET_5D = 'http://localhost:9000/datasets/examples/dense_grid_5d_example.zarr';

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
        const count = Math.min(col.count * col.itemSize, 3000);
        for (let i = 0; i < count; i++) {
          const v = col.array[i];
          if (!Number.isFinite(v)) issues.push(`${obj.name}: color[${i}] = ${v} (not finite)`);
          if (v < 0) issues.push(`${obj.name}: color[${i}] = ${v} (negative)`);
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
        const dr = obj.geometry.drawRange;
        const count = dr.count < Infinity ? Math.min(dr.count, rad.count) : rad.count;
        for (let i = 0; i < count; i++) {
          const v = rad.array[i];
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
