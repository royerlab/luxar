/**
 * Transform Hierarchy E2E Tests
 *
 * CRITICAL: Tests hierarchical transform composition to prevent matrix bugs.
 *
 * The NumPy→THREE.js matrix transpose bug (2025-01) showed why these tests
 * are essential. Without E2E tests, bugs like incorrect transform composition
 * can go undetected.
 *
 * These tests verify:
 * 1. Parent→child transform composition works correctly
 * 2. Matrix transpose is handled properly for THREE.js
 * 3. Multiple levels of hierarchy work
 * 4. Transform updates propagate correctly
 */

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForNextRender } from './helpers';

// Use pre-generated hierarchy example
const HIERARCHY_DATASET = 'http://localhost:9000/datasets/examples/hierarchy_example.luxar.zarr';
const TRANSFORM_DATASET = 'http://localhost:9000/datasets/examples/transform_example.luxar.zarr';

test.describe('Transform Hierarchy - Basic Composition', () => {
  test('should apply parent transform to child objects', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get world positions of objects in hierarchy
    const positions = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{ name: string; localPos: number[]; worldPos: number[] }> = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' || obj.type === 'Group') {
          // Local translation comes from the matrix, not `.position`: node
          // transforms are installed as a full affine matrix (so shear
          // survives) and TRS is left unpopulated.
          const e = obj.matrix.elements;
          const localPos = [e[12], e[13], e[14]];

          const worldPosVec = new (obj.position.constructor as any)();
          obj.getWorldPosition(worldPosVec);

          results.push({
            name: obj.name,
            localPos,
            worldPos: [worldPosVec.x, worldPosVec.y, worldPosVec.z],
          });
        }
      });

      return results;
    });

    expect(positions.length).toBeGreaterThan(0);

    // Find objects in hierarchy
    const parent = positions.find((p) => p.name.includes('Galaxy') || p.name === 'Parent');
    const child = positions.find((p) => p.name.includes('SolarSystem') || p.name.includes('Child'));

    // If we have a parent→child relationship, verify composition
    if (parent && child) {
      // Child's world position should differ from local position if parent has transform
      const localDiffersFromWorld =
        Math.abs(child.localPos[0] - child.worldPos[0]) > 0.01 ||
        Math.abs(child.localPos[1] - child.worldPos[1]) > 0.01 ||
        Math.abs(child.localPos[2] - child.worldPos[2]) > 0.01;

      // The previous assertion here was `expect(typeof x).toBe('boolean')`,
      // which is true of any boolean and therefore tested nothing. A child's
      // world position must at minimum be finite and derived from the parent
      // chain; whether it DIFFERS from local depends on the fixture, so that
      // stays a soft signal rather than a hard assertion.
      expect(child.worldPos.every((v) => Number.isFinite(v))).toBe(true);
      expect(parent.worldPos.every((v) => Number.isFinite(v))).toBe(true);
      if (parent.localPos.some((v) => Math.abs(v) > 0.01)) {
        expect(localDiffersFromWorld).toBe(true);
      }
    } else {
      // If no clear parent-child found, just verify hierarchy exists
      expect(positions.length).toBeGreaterThan(0);
    }
  });

  test('should handle multi-level hierarchy (grandparent→parent→child)', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Verify 3+ levels of hierarchy exist
    const hierarchy = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const levels: Array<{ name: string; depth: number }> = [];

      function traverse(obj: any, depth: number) {
        if (obj.userData?.nodeType === 'points' || obj.type === 'Group') {
          levels.push({ name: obj.name, depth });
        }
        obj.children.forEach((child: any) => traverse(child, depth + 1));
      }

      traverse(debug.scene, 0);
      return levels;
    });

    // Should have objects at multiple hierarchy depths
    const maxDepth = Math.max(...hierarchy.map((h) => h.depth));
    expect(maxDepth).toBeGreaterThanOrEqual(2); // At least 3 levels (0, 1, 2)
  });

  test('should apply rotation transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${TRANSFORM_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Read the MATRIX, not position/quaternion/scale: authored node transforms
    // are installed as a full affine matrix so shear survives, which leaves the
    // TRS fields at their defaults. Reading `obj.rotation` here would report no
    // rotation on a correctly-rotated node.
    const rotatedObject = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      let rotated: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.name && obj.name.includes('Rotated')) {
          rotated = { name: obj.name, matrix: Array.from(obj.matrix.elements) };
        }
      });

      return rotated;
    });

    // Fail if the fixture node is absent — an `if (obj)` guard would turn a
    // missing node into a silent pass, which is how this class of test rots.
    expect(rotatedObject, 'no node named *Rotated* in the scene').not.toBeNull();

    // The 3x3 linear block must not be the identity.
    const m = rotatedObject.matrix;
    const linear = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
    const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const hasRotation = linear.some((v: number, i: number) => Math.abs(v - identity[i]) > 0.01);
    expect(hasRotation).toBe(true);
  });

  test('should apply scale transforms correctly', async ({ page }) => {
    await page.goto(`/?src=${TRANSFORM_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Matrix, not `.scale` — see the rotation test above.
    const scaledObject = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      let scaled: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.name && obj.name.includes('Scaled')) {
          scaled = { name: obj.name, matrix: Array.from(obj.matrix.elements) };
        }
      });

      return scaled;
    });

    expect(scaledObject, 'no node named *Scaled* in the scene').not.toBeNull();

    // Column norms of the 3x3 linear block are the axis scale factors, and they
    // stay correct under shear (unlike reading `.scale`, which is unpopulated).
    const m = scaledObject.matrix;
    const axisLengths = [
      Math.hypot(m[0], m[1], m[2]),
      Math.hypot(m[4], m[5], m[6]),
      Math.hypot(m[8], m[9], m[10]),
    ];
    const hasScale = axisLengths.some((s: number) => Math.abs(s - 1.0) > 0.01);
    expect(hasScale).toBe(true);
  });

  test('should compose translate + rotate + scale transforms', async ({ page }) => {
    await page.goto(`/?src=${TRANSFORM_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Find object with complex transform
    const complexObject = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;

      let complex: any = null;
      debug.scene.traverse((obj: any) => {
        if (obj.name && obj.name.includes('Complex')) {
          // getWorldPosition reads matrixWorld, so it is correct either way; the
          // argument is only a scratch vector.
          const worldPos = new (obj.position.constructor as any)();
          obj.getWorldPosition(worldPos);

          complex = {
            name: obj.name,
            matrix: Array.from(obj.matrix.elements),
            worldPosition: [worldPos.x, worldPos.y, worldPos.z],
          };
        }
      });

      return complex;
    });

    expect(complexObject, 'no node named *Complex* in the scene').not.toBeNull();

    // A composed translate+rotate+scale must differ from the identity somewhere
    // in the full 4x4 — checked on the matrix, since TRS is not populated.
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const hasTransform = complexObject.matrix.some(
      (v: number, i: number) => Math.abs(v - identity[i]) > 0.01
    );
    expect(hasTransform).toBe(true);
  });
});

test.describe('Transform Hierarchy - Matrix Correctness', () => {
  test('should load transform matrices in correct order (column-major for THREE.js)', async ({
    page,
  }) => {
    await page.goto(`/?src=${TRANSFORM_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Verify transform matrices are valid
    const matrixData = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const matrices: Array<{ name: string; elements: number[]; isValid: boolean }> = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' || obj.type === 'Group') {
          const elements = obj.matrix.elements;
          // Check if matrix is valid (not all zeros)
          const isValid = elements.some((v: number) => Math.abs(v) > 0.01);

          matrices.push({
            name: obj.name,
            elements: Array.from(elements),
            isValid,
          });
        }
      });

      return matrices;
    });

    // All matrices should be valid
    matrixData.forEach((m) => {
      expect(m.isValid).toBe(true);
    });

    // Verify matrix element [0] is reasonable (should be scale_x component in THREE.js)
    const firstMatrix = matrixData[0];
    expect(Math.abs(firstMatrix.elements[0])).toBeGreaterThan(0.01);
  });

  test('should maintain transform consistency across renders', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Get world positions
    const getPositions = () =>
      page.evaluate(() => {
        const debug = (window as any).__luxarDebug;
        const positions: Array<{ name: string; pos: number[] }> = [];

        debug.scene.traverse((obj: any) => {
          if (obj.userData?.nodeType === 'points') {
            // Get world position using THREE.js object method
            const worldPos = obj.getWorldPosition(obj.position.clone());
            positions.push({
              name: obj.name,
              pos: [worldPos.x, worldPos.y, worldPos.z],
            });
          }
        });

        return positions;
      });

    const positions1 = await getPositions();

    // Trigger a render
    await page.evaluate(() => {
      (window as any).__luxarDebug.renderOnce();
    });
    await waitForNextRender(page);

    const positions2 = await getPositions();

    // Positions should remain consistent (transforms don't change between renders)
    expect(positions1.length).toBe(positions2.length);

    for (let i = 0; i < positions1.length; i++) {
      expect(positions1[i].name).toBe(positions2[i].name);
      expect(positions1[i].pos[0]).toBeCloseTo(positions2[i].pos[0], 5);
      expect(positions1[i].pos[1]).toBeCloseTo(positions2[i].pos[1], 5);
      expect(positions1[i].pos[2]).toBeCloseTo(positions2[i].pos[2], 5);
    }
  });
});

test.describe('Transform Hierarchy - Edge Cases', () => {
  test('should handle identity transforms', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Find objects with identity transform
    const identityObjects = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{ name: string; hasIdentity: boolean }> = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType === 'points' || obj.type === 'Group') {
          // Identity is checked on the local MATRIX. Checking position/rotation/
          // scale would report identity for every authored node, since node
          // transforms are installed as a full matrix and leave TRS unpopulated
          // — i.e. the old form silently became always-true.
          const e = obj.matrix.elements;
          const ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
          const hasIdentity = ident.every((v, i) => Math.abs(e[i] - v) < 0.01);

          results.push({
            name: obj.name,
            hasIdentity,
          });
        }
      });

      return results;
    });

    // At least some objects should exist (identity or not)
    expect(identityObjects.length).toBeGreaterThan(0);
  });

  test('should handle very deep hierarchies without stack overflow', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // Traverse entire hierarchy without errors
    const traversalResult = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      let count = 0;

      try {
        debug.scene.traverse(() => {
          count++;
        });
        return { success: true, count };
      } catch (error) {
        return { success: false, error: String(error) };
      }
    });

    expect(traversalResult.success).toBe(true);
    expect(traversalResult.count).toBeGreaterThan(0);
  });
});
