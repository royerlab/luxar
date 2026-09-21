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

import * as THREE from 'three';

import { test, expect } from './fixtures';
import { waitForLuxarReady, waitForNextRender } from './helpers';

// Use pre-generated hierarchy example
const HIERARCHY_DATASET = 'http://localhost:9000/datasets/examples/hierarchy_example.luxar.zarr';
const TRANSFORM_DATASET = 'http://localhost:9000/datasets/examples/transform_example.luxar.zarr';

test.describe('Transform Hierarchy - Basic Composition', () => {
  test('should apply parent transform to child objects', async ({ page }) => {
    await page.goto(`/?src=${HIERARCHY_DATASET}&debug`);
    await waitForLuxarReady(page);

    // For every node, recompute the world origin from its PARENT's world matrix
    // and its own local matrix, and compare against what three.js reports. That
    // is the composition this test is named for, and it catches the failure
    // mode that matters here: a child whose `matrixWorld` went stale and no
    // longer reflects its parent. Reading `.position` would not work — node
    // transforms are installed as a full affine matrix so shear survives, which
    // leaves the TRS fields at their defaults.
    const nodes = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const results: Array<{
        name: string;
        parentName: string | null;
        parentWorldPos: number[] | null;
        worldPos: number[];
        expectedWorldPos: number[] | null;
      }> = [];

      debug.scene.traverse((obj: any) => {
        if (obj.userData?.nodeType !== 'points' && obj.type !== 'Group') return;

        const worldPosVec = new (obj.position.constructor as any)();
        obj.getWorldPosition(worldPosVec);

        let parentWorldPos: number[] | null = null;
        let expectedWorldPos: number[] | null = null;
        if (obj.parent) {
          const pw = new (obj.position.constructor as any)();
          pw.setFromMatrixPosition(obj.parent.matrixWorld);
          parentWorldPos = [pw.x, pw.y, pw.z];

          // parent.matrixWorld ∘ (local translation)
          const expected = new (obj.position.constructor as any)();
          expected.setFromMatrixPosition(obj.matrix);
          expected.applyMatrix4(obj.parent.matrixWorld);
          expectedWorldPos = [expected.x, expected.y, expected.z];
        }

        results.push({
          name: obj.name,
          parentName: obj.parent?.name ?? null,
          parentWorldPos,
          worldPos: [worldPosVec.x, worldPosVec.y, worldPosVec.z],
          expectedWorldPos,
        });
      });

      return results;
    });

    expect(nodes.length).toBeGreaterThan(0);

    // Every parented node must agree with its own parent.
    const inconsistent = nodes
      .filter((n) => n.expectedWorldPos !== null)
      .filter((n) =>
        n.worldPos.some((v, i) => Math.abs(v - (n.expectedWorldPos as number[])[i]) > 1e-4)
      )
      .map((n) => `${n.name}: world ${n.worldPos} != parent∘local ${n.expectedWorldPos}`);
    expect(inconsistent, 'nodes whose world matrix disagrees with their parent').toEqual([]);

    // Guard against the check above passing vacuously on a flat scene: the
    // fixture must actually contain a node sitting under a displaced ancestor.
    const displaced = nodes.filter(
      (n) => n.parentWorldPos !== null && n.parentWorldPos.some((v) => Math.abs(v) > 0.01)
    );
    expect(
      displaced.length,
      'fixture has no node under a translated ancestor, so composition was never exercised'
    ).toBeGreaterThan(0);
    for (const n of displaced) {
      expect(n.worldPos.every((v) => Number.isFinite(v))).toBe(true);
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

    // The assertion that actually discriminates, and the only one in this file
    // that does. `transform_example.py` authors this node as
    // `rotate_y(30)` THEN a non-uniform scale — i.e. S·R, which TRS cannot
    // represent. So decomposing the delivered matrix and recomposing it must
    // NOT round-trip: if it did, the matrix reaching the renderer would already
    // have been flattened to position/quaternion/scale, which is exactly the
    // corruption this file's fix removes.
    //
    // Everything else here is a "not the identity" check and stays green
    // against the decomposing implementation — verified by reverting
    // `transforms.ts` and watching all nine tests pass. Without this block the
    // suite is a regression net, not evidence.
    const authored = new THREE.Matrix4().fromArray(complexObject.matrix);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3();
    authored.decompose(position, quaternion, scale);
    const roundTripped = new THREE.Matrix4().compose(position, quaternion, scale);

    const worstElement = Math.max(
      ...authored.elements.map((v, i) => Math.abs(v - roundTripped.elements[i]))
    );
    expect(
      worstElement,
      'the delivered matrix survives a TRS round trip, so shear was already lost'
    ).toBeGreaterThan(1e-3);
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
