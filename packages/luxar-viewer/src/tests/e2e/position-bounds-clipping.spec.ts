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
const DATASET = 'http://localhost:9000/packages/luxar/examples/build_example_structured.zarr';

test.describe('Position Bounds and Clipping Planes', () => {
  test('should load scene with position_bounds from metadata', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Check if position bounds were loaded from metadata
    const boundsInfo: { min: number[]; max: number[] } | null = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let positionBounds: any = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !positionBounds) {
          positionBounds = obj.userData.positionBounds;
        }
      });

      return positionBounds;
    });

    // Position bounds should be present in scene userData
    expect(boundsInfo).not.toBeNull();
    if (boundsInfo) {
      expect(Array.isArray(boundsInfo.min)).toBe(true);
      expect(Array.isArray(boundsInfo.max)).toBe(true);
      expect(boundsInfo.min.length).toBeGreaterThan(0);
      expect(boundsInfo.max.length).toBe(boundsInfo.min.length);

      // Min should be less than or equal to max for each dimension
      for (let i = 0; i < boundsInfo.min.length; i++) {
        expect(boundsInfo.min[i]).toBeLessThanOrEqual(boundsInfo.max[i]);
      }
    }
  });

  test('should set clipping planes based on scene bounds', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get camera clipping planes
    const cameraInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return {
        near: debug.camera.near,
        far: debug.camera.far,
        position: {
          x: debug.camera.position.x,
          y: debug.camera.position.y,
          z: debug.camera.position.z,
        },
      };
    });

    // Clipping planes should be valid
    expect(cameraInfo.near).toBeGreaterThan(0);
    expect(cameraInfo.far).toBeGreaterThan(cameraInfo.near);

    // Near/far ratio should be reasonable (less than 10000:1)
    const ratio = cameraInfo.far / cameraInfo.near;
    expect(ratio).toBeLessThan(10000);
  });

  test('should have clipping planes that encompass scene bounds', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get both bounds and camera info
    const sceneInfo = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let positionBounds: { min: number[]; max: number[] } | null = null;

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.positionBounds && !positionBounds) {
          positionBounds = obj.userData.positionBounds;
        }
      });

      // Calculate scene size from bounds (first 3 dims = X, Y, Z)
      let sceneSize = 0;
      if (positionBounds && (positionBounds as any).min && (positionBounds as any).min.length >= 3) {
        const bounds = positionBounds as { min: number[]; max: number[] };
        const dx = bounds.max[0] - bounds.min[0];
        const dy = bounds.max[1] - bounds.min[1];
        const dz = bounds.max[2] - bounds.min[2];
        sceneSize = Math.sqrt(dx * dx + dy * dy + dz * dz);
      }

      return {
        positionBounds,
        sceneSize,
        camera: {
          near: debug.camera.near,
          far: debug.camera.far,
          position: debug.camera.position.clone(),
        },
      };
    });

    // If we have scene bounds, verify clipping planes accommodate them
    if (sceneInfo.positionBounds && sceneInfo.sceneSize > 0) {
      // Far plane should be large enough to see the entire scene
      // (scene size + some margin for camera distance)
      expect(sceneInfo.camera.far).toBeGreaterThan(sceneInfo.sceneSize);
    }
  });

  test('should update clipping planes when autoAdjust is called', async ({ page }) => {
    await page.goto(`/?src=${DATASET}&debug`);
    await waitForLuxarReady(page);

    // Manually set clipping planes to invalid values
    // (No need to store initial values - just test adjustment works)
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.camera.near = 0.001;
      debug.camera.far = 1;
      debug.camera.updateProjectionMatrix();
    });

    // Call autoAdjustClippingPlanes via the scene manager
    // Note: We access it through the app components
    const updatedClipping = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const app = debug.app;
      if (app && app.components && app.components.sceneManager) {
        return app.components.sceneManager.autoAdjustClippingPlanes();
      }
      return null;
    });

    // Should have adjusted clipping planes
    if (updatedClipping) {
      expect(updatedClipping.near).toBeGreaterThan(0.001);
      expect(updatedClipping.far).toBeGreaterThan(1);
    }
  });
});
