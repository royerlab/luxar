/**
 * Data Integrity E2E Tests
 *
 * Validates that loaded geometry data is internally consistent. Points
 * store per-point data in an RGBA32F element texture (12 floats per point:
 * center xyz, radius, color rgb, sharpness, scalar, alpha) with a single
 * aSortedIndex per-instance attribute:
 * - The texel buffer and aSortedIndex cover every visible instance
 * - Position values contain no NaN or Infinity
 * - Color values are finite and non-negative
 * - Radius values are non-negative
 * - DrawRange never exceeds buffer size
 * - Integrity is maintained after nD navigation
 *
 * These tests catch the most dangerous class of silent bug: under-allocated
 * or misaligned point storage that shows garbage data with no error message.
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
      // The texel buffer and aSortedIndex must cover every visible instance
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
        if (obj.userData?.nodeType !== 'points' || !obj.geometry?.userData?.hasColors) return;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!texData) return;
        // Per-point data is texture-backed: the RGBA32F element texture
        // holds 12 floats per point, with color rgb at slots
        // [i*12+4 .. i*12+6] (already normalized floats). Validate the
        // committed (visible) instances.
        const STRIDE = 12;
        const count = Math.min(
          obj.geometry.instanceCount ?? 0,
          Math.floor(texData.length / STRIDE)
        );
        const sampleCount = Math.min(count, 1000);
        for (let i = 0; i < sampleCount; i++) {
          const r = texData[i * STRIDE + 4];
          const g = texData[i * STRIDE + 5];
          const b = texData[i * STRIDE + 6];
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
        if (obj.userData?.nodeType !== 'points' || !obj.geometry?.userData?.hasRadii) return;
        const texData = obj.geometry?.userData?.elementTexture?.image?.data;
        if (!texData) return;
        // Per-point data is texture-backed: the radius lives at texel slot
        // [i*12+3] of the RGBA32F element texture (a dtype-normalized
        // float). Validate the committed (visible) instances.
        const STRIDE = 12;
        const instanceCount = Math.min(
          obj.geometry.instanceCount ?? 0,
          Math.floor(texData.length / STRIDE)
        );
        for (let i = 0; i < instanceCount; i++) {
          const v = texData[i * STRIDE + 3];
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
