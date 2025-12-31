/**
 * Position Bounds and Clipping Planes E2E Tests
 *
 * These tests verify the end-to-end flow of:
 * 1. Python compiler computing and storing position_bounds in zarr metadata
 * 2. TypeScript viewer loading position_bounds from metadata
 * 3. Clipping planes being automatically set based on scene bounds
 *
 * This is critical for ensuring optimal Z-buffer precision regardless of
 * whether point data has been loaded yet (important for large nD datasets).
 */

import { test, expect } from '@playwright/test';
import { waitForLuxarReady } from './helpers';

// Use a basic example dataset which should have position_bounds set
const DATASET_WITH_BOUNDS = 'http://localhost:9000/datasets/examples/build_example_structured.zarr';

test.describe('Position Bounds and Clipping Planes', () => {
  test('should load scene with position_bounds from metadata', async ({ page }) => {
    await page.goto(`/?src=${DATASET_WITH_BOUNDS}&debug`);
    await waitForLuxarReady(page);

    // Check if position bounds were loaded from metadata
    const boundsInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let positionBounds: { min: number[]; max: number[] } | null = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !positionBounds) {
          positionBounds = obj.userData.positionBounds;
        }
      });

      return positionBounds;
    });

    // Position bounds MUST be present - this is the core feature we're testing
    expect(boundsInfo).not.toBeNull();
    expect(boundsInfo).toBeDefined();

    // TypeScript narrowing - we know it's not null now
    const bounds = boundsInfo as unknown as { min: number[]; max: number[] };

    // Validate structure
    expect(Array.isArray(bounds.min)).toBe(true);
    expect(Array.isArray(bounds.max)).toBe(true);
    expect(bounds.min.length).toBeGreaterThan(0);
    expect(bounds.max.length).toBe(bounds.min.length);

    // Min should be less than or equal to max for each dimension
    for (let i = 0; i < bounds.min.length; i++) {
      expect(bounds.min[i]).toBeLessThanOrEqual(bounds.max[i]);
    }

    // Log actual values for debugging
    console.log(`Position bounds: min=[${bounds.min.join(', ')}], max=[${bounds.max.join(', ')}]`);
  });

  test('should set clipping planes from metadata bounds (not defaults)', async ({ page }) => {
    await page.goto(`/?src=${DATASET_WITH_BOUNDS}&debug`);
    await waitForLuxarReady(page);

    // Get both bounds and camera clipping planes
    const info = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let positionBounds: { min: number[]; max: number[] } | null = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !positionBounds) {
          positionBounds = obj.userData.positionBounds;
        }
      });

      return {
        bounds: positionBounds,
        camera: {
          near: debug.camera.near,
          far: debug.camera.far,
        },
        // Default values from config
        defaults: {
          near: 0.1,
          far: 1000,
        },
      };
    });

    // Bounds MUST be present for this test to be meaningful
    expect(info.bounds).not.toBeNull();

    // Clipping planes should be valid
    expect(info.camera.near).toBeGreaterThan(0);
    expect(info.camera.far).toBeGreaterThan(info.camera.near);

    // Near/far ratio should be reasonable (less than 10000:1)
    const ratio = info.camera.far / info.camera.near;
    expect(ratio).toBeLessThan(10000);

    // CRITICAL: Clipping planes should NOT be defaults if bounds exist
    // This verifies that autoAdjustClippingPlanes actually ran with metadata bounds
    // At least one of near/far should differ from defaults
    // (they might coincidentally match defaults, but unlikely for real data)
    const isNotDefault =
      Math.abs(info.camera.near - info.defaults.near) > 0.001 ||
      Math.abs(info.camera.far - info.defaults.far) > 1;

    // Log for debugging
    console.log(
      `Clipping: near=${info.camera.near.toFixed(4)}, far=${info.camera.far.toFixed(1)}, ` +
        `defaults: near=${info.defaults.near}, far=${info.defaults.far}`
    );

    expect(isNotDefault).toBe(true);
  });

  test('should have clipping planes that encompass scene bounds', async ({ page }) => {
    await page.goto(`/?src=${DATASET_WITH_BOUNDS}&debug`);
    await waitForLuxarReady(page);

    // Get both bounds and camera info
    const sceneInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let foundBounds: { min: number[]; max: number[] } | null = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !foundBounds) {
          foundBounds = obj.userData.positionBounds as { min: number[]; max: number[] };
        }
      });

      // Calculate scene diagonal from bounds (first 3 dims = X, Y, Z)
      let sceneDiagonal = 0;
      let sceneCenter = { x: 0, y: 0, z: 0 };
      if (foundBounds !== null) {
        const bounds = foundBounds as { min: number[]; max: number[] };
        if (bounds.min.length >= 3) {
          const bMin = bounds.min;
          const bMax = bounds.max;
          const dx = bMax[0] - bMin[0];
          const dy = bMax[1] - bMin[1];
          const dz = bMax[2] - bMin[2];
          sceneDiagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);
          sceneCenter = {
            x: (bMin[0] + bMax[0]) / 2,
            y: (bMin[1] + bMax[1]) / 2,
            z: (bMin[2] + bMax[2]) / 2,
          };
        }
      }

      // Calculate camera distance to scene center
      const cameraPos = debug.camera.position;
      const distanceToCenter = Math.sqrt(
        Math.pow(cameraPos.x - sceneCenter.x, 2) +
          Math.pow(cameraPos.y - sceneCenter.y, 2) +
          Math.pow(cameraPos.z - sceneCenter.z, 2)
      );

      return {
        positionBounds: foundBounds,
        sceneDiagonal,
        sceneCenter,
        distanceToCenter,
        camera: {
          near: debug.camera.near,
          far: debug.camera.far,
          position: { x: cameraPos.x, y: cameraPos.y, z: cameraPos.z },
        },
      };
    });

    // Must have bounds for this test to be meaningful
    expect(sceneInfo.positionBounds).not.toBeNull();
    expect(sceneInfo.sceneDiagonal).toBeGreaterThan(0);

    // Far plane should be large enough to see the entire scene from current camera position
    // Far plane should at least cover: camera distance + half scene diagonal
    const minimumFar = sceneInfo.distanceToCenter + sceneInfo.sceneDiagonal / 2;
    expect(sceneInfo.camera.far).toBeGreaterThanOrEqual(minimumFar * 0.9); // 10% tolerance

    // Log for debugging
    console.log(
      `Scene diagonal: ${sceneInfo.sceneDiagonal.toFixed(2)}, ` +
        `Camera distance: ${sceneInfo.distanceToCenter.toFixed(2)}, ` +
        `Far plane: ${sceneInfo.camera.far.toFixed(2)}, ` +
        `Minimum needed: ${minimumFar.toFixed(2)}`
    );
  });

  test('should update clipping planes when autoAdjust is called', async ({ page }) => {
    await page.goto(`/?src=${DATASET_WITH_BOUNDS}&debug`);
    await waitForLuxarReady(page);

    // Disable dynamic clipping so we can test static auto-adjust in isolation
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const app = debug.app;
      if (app?.components?.sceneManager) {
        app.components.sceneManager.setDynamicClipping(false);
      }
    });

    // Manually set clipping planes to obviously wrong values
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.camera.near = 0.001;
      debug.camera.far = 1;
      debug.camera.updateProjectionMatrix();
    });

    // Verify they were changed
    const wrongClipping = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return { near: debug.camera.near, far: debug.camera.far };
    });
    expect(wrongClipping.near).toBeCloseTo(0.001, 4);
    expect(wrongClipping.far).toBeCloseTo(1, 1);

    // Call autoAdjustClippingPlanes via the scene manager
    const result = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const app = debug.app;

      // Verify sceneManager exists
      if (!app?.components?.sceneManager) {
        return { error: 'sceneManager not found' };
      }

      const adjusted = app.components.sceneManager.autoAdjustClippingPlanes();
      return {
        adjusted,
        cameraAfter: {
          near: debug.camera.near,
          far: debug.camera.far,
        },
      };
    });

    // Must not have error
    expect(result).not.toHaveProperty('error');

    // Must have adjusted values
    const adjustedResult = result as {
      adjusted: { near: number; far: number };
      cameraAfter: { near: number; far: number };
    };

    expect(adjustedResult.adjusted).toBeDefined();
    expect(adjustedResult.adjusted.near).toBeGreaterThan(0.001);
    expect(adjustedResult.adjusted.far).toBeGreaterThan(1);

    // Camera should now have the adjusted values
    expect(adjustedResult.cameraAfter.near).toBe(adjustedResult.adjusted.near);
    expect(adjustedResult.cameraAfter.far).toBe(adjustedResult.adjusted.far);

    // Log for debugging
    console.log(
      `Auto-adjust: near ${wrongClipping.near} -> ${adjustedResult.adjusted.near.toFixed(4)}, ` +
        `far ${wrongClipping.far} -> ${adjustedResult.adjusted.far.toFixed(1)}`
    );
  });

  test('should use 50% safety margin on calculated clipping planes', async ({ page }) => {
    await page.goto(`/?src=${DATASET_WITH_BOUNDS}&debug`);
    await waitForLuxarReady(page);

    // Disable dynamic clipping and call autoAdjust to test static calculation
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const app = debug.app;
      if (app?.components?.sceneManager) {
        app.components.sceneManager.setDynamicClipping(false);
        app.components.sceneManager.autoAdjustClippingPlanes();
      }
    });

    // Get bounds and camera info to verify margin is applied
    // The actual implementation in scene-manager-utils.ts uses:
    // - calculateDistancesToBoundingBox to find nearDist/farDist (to corners AND face centers)
    // - near = nearDist * (1 - CLIPPING_SAFETY_MARGIN) = nearDist * 0.5
    // - far = farDist * (1 + CLIPPING_SAFETY_MARGIN) = farDist * 1.5
    // where CLIPPING_SAFETY_MARGIN = 0.5 (50%)
    const info = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let foundBounds: { min: number[]; max: number[] } | null = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !foundBounds) {
          foundBounds = obj.userData.positionBounds as { min: number[]; max: number[] };
        }
      });

      if (!foundBounds) return null;

      // Extract arrays for TypeScript type narrowing
      const bounds = foundBounds as { min: number[]; max: number[] };
      const bMin = bounds.min;
      const bMax = bounds.max;

      // Convert to 3D bounding box format
      const box = {
        min: { x: bMin[0], y: bMin[1], z: bMin[2] },
        max: { x: bMax[0], y: bMax[1], z: bMax[2] },
      };

      const cameraPos = debug.camera.position;
      const point = { x: cameraPos.x, y: cameraPos.y, z: cameraPos.z };

      // Calculate distances to all 8 corners AND 6 face centers (matching scene-manager-utils.ts)
      const cx = (box.min.x + box.max.x) / 2;
      const cy = (box.min.y + box.max.y) / 2;
      const cz = (box.min.z + box.max.z) / 2;

      const corners = [
        { x: box.min.x, y: box.min.y, z: box.min.z },
        { x: box.max.x, y: box.min.y, z: box.min.z },
        { x: box.min.x, y: box.max.y, z: box.min.z },
        { x: box.max.x, y: box.max.y, z: box.min.z },
        { x: box.min.x, y: box.min.y, z: box.max.z },
        { x: box.max.x, y: box.min.y, z: box.max.z },
        { x: box.min.x, y: box.max.y, z: box.max.z },
        { x: box.max.x, y: box.max.y, z: box.max.z },
      ];

      const faceCenters = [
        { x: box.min.x, y: cy, z: cz },
        { x: box.max.x, y: cy, z: cz },
        { x: cx, y: box.min.y, z: cz },
        { x: cx, y: box.max.y, z: cz },
        { x: cx, y: cy, z: box.min.z },
        { x: cx, y: cy, z: box.max.z },
      ];

      const testPoints = [...corners, ...faceCenters];

      let nearDist = Infinity;
      let farDist = 0;

      for (const p of testPoints) {
        const dx = point.x - p.x;
        const dy = point.y - p.y;
        const dz = point.z - p.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        nearDist = Math.min(nearDist, dist);
        farDist = Math.max(farDist, dist);
      }

      // Apply 50% safety margin (CLIPPING_SAFETY_MARGIN = 0.5)
      const MIN_NEAR_PLANE = 0.0001;
      const MARGIN = 0.5;
      const expectedNear = Math.max(MIN_NEAR_PLANE, nearDist * (1 - MARGIN));
      const expectedFar = farDist * (1 + MARGIN);

      return {
        actual: {
          near: debug.camera.near,
          far: debug.camera.far,
        },
        expected: {
          near: expectedNear,
          far: expectedFar,
        },
        nearDist,
        farDist,
      };
    });

    expect(info).not.toBeNull();
    if (!info) return;

    // Verify near plane is approximately correct (with some tolerance for floating point)
    expect(info.actual.near).toBeCloseTo(info.expected.near, 2);

    // Verify far plane is approximately correct
    expect(info.actual.far).toBeCloseTo(info.expected.far, 0);

    console.log(
      `Margin test: near actual=${info.actual.near.toFixed(4)} expected=${info.expected.near.toFixed(4)}, ` +
        `far actual=${info.actual.far.toFixed(1)} expected=${info.expected.far.toFixed(1)}`
    );
  });
});
