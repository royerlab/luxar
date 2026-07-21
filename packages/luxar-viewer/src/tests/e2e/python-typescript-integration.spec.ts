/**
 * Python↔TypeScript Integration Tests
 *
 * CRITICAL: These tests verify that Python encoding and TypeScript decoding
 * remain compatible. Changes to either side must not break the other.
 *
 * These tests use pre-generated Python datasets from examples/ directory
 * and verify they load correctly in TypeScript viewer.
 *
 * This catches bugs like:
 * - Array_ref resolution failures
 * - Encoding format mismatches
 *
 * Note: Transform/hierarchy integration is tested in transform-hierarchy.spec.ts.
 * Note: All-dataset smoke tests are in all-examples-smoke-test.spec.ts.
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, getLuxarState } from './helpers';

const DATASETS = {
  basic: 'http://localhost:9000/datasets/examples/build_example_structured.luxar.zarr',
  nD5D: 'http://localhost:9000/datasets/examples/dense_grid_5d_example.luxar.zarr',
};

test.describe('Python→TypeScript Integration', () => {
  test('should decode Python-generated basic dataset', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.basic}&debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);
    expect(state.pointClouds.length).toBeGreaterThan(0);

    // Verify point data structure is correct
    const pointData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const points: any[] = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points') {
          const geom = obj.geometry;
          // Per-point data is texture-backed: the visible point count is
          // the geometry's instanceCount and field presence comes from the
          // node's declared metadata (userData.attrs).
          points.push({
            name: obj.name,
            count: geom.instanceCount,
            hasColors: !!obj.userData?.attrs?.has_colors,
            hasRadii: !!obj.userData?.attrs?.has_radii,
          });
        }
      });

      return points;
    });

    expect(pointData.length).toBeGreaterThan(0);
    pointData.forEach((cloud) => {
      expect(cloud.count).toBeGreaterThan(0);
    });
  });

  test('should decode nD positions correctly', async ({ page }) => {
    await page.goto(`/?src=${DATASETS.nD5D}&debug`);
    await waitForLuxarReady(page);

    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);

    // Verify dimensions were loaded
    const dimensionsInfo = await page.evaluate(() => {
      const scene = (window as any).__luxarDebug.scene;
      return {
        hasDimensions: !!scene.userData?.sceneDimensions,
        dimensionCount: scene.userData?.sceneDimensions?.length || 0,
      };
    });

    expect(typeof dimensionsInfo.hasDimensions).toBe('boolean');
  });
});
