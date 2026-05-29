/**
 * Lock-in tests for gsplat geometry bounding-box expansion.
 *
 * `createInstancedGSplatsMesh` expands `geometry.boundingBox` outward by
 * `maxRowNorm × truncationRadius` (`gsplat-geometry.ts`). This margin is what
 * keeps the GPU-picking ray-AABB cull from clipping the pick footprint of a
 * splat sitting on the scene's bounding-box surface — the pick cull
 * (`picking/picking-system/ray-aabb.ts`) trusts `boundingBox` directly for
 * lines/gsplats (it only adds its own margin for points, whose mesh has
 * frustum culling disabled). The visual truncation (3σ) is ≥ the pick
 * footprint (1.5σ), so the cull box always covers the pick footprint. If the
 * expansion here is ever removed, splat hover-picking silently regresses for
 * edge splats, so pin it.
 *
 * Pure buffer + Box3 math — no GL context needed (jsdom-safe).
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  createInstancedGSplatsMesh,
  packCholeskyForShader,
  type InstancedGSplatsMeshConfig,
} from '../../../rendering/gsplat-geometry';

/**
 * One splat at the origin with a diagonal Cholesky factor
 * [L00,L10,L11,L20,L21,L22] = [2,0,2,0,0,2] → every row norm is 2, so
 * `maxRowNorm === 2`.
 */
function makeSingleSplatConfig(): InstancedGSplatsMeshConfig {
  const factors = new Float32Array([2, 0, 2, 0, 0, 2]);
  const { cholesky01, cholesky23, cholesky45 } = packCholeskyForShader(factors, 1);
  return {
    centers: new Float32Array([0, 0, 0]),
    cholesky01,
    cholesky23,
    cholesky45,
    amplitudes: new Float32Array([1]),
    colors: new Float32Array([1, 1, 1]),
    splatCount: 1,
  };
}

/** Minimal material stub exposing the `uTruncate` uniform the builder reads. */
function materialWithTruncate(value: number): THREE.Material {
  return { uniforms: { uTruncate: { value } } } as unknown as THREE.Material;
}

describe('createInstancedGSplatsMesh — bounding-box footprint expansion', () => {
  it('expands boundingBox by maxRowNorm × truncationRadius', () => {
    const maxRowNorm = 2; // by construction (see makeSingleSplatConfig)
    const truncate = 3.0;
    const mesh = createInstancedGSplatsMesh(
      makeSingleSplatConfig(),
      materialWithTruncate(truncate)
    );
    const box = mesh.geometry.boundingBox!;
    const margin = maxRowNorm * truncate; // 6.0

    expect(box).not.toBeNull();
    // Single center at origin → box collapses to a point, then expands by margin.
    expect(box.min.x).toBeCloseTo(-margin, 5);
    expect(box.max.x).toBeCloseTo(margin, 5);
    expect(box.min.y).toBeCloseTo(-margin, 5);
    expect(box.max.y).toBeCloseTo(margin, 5);
    expect(box.min.z).toBeCloseTo(-margin, 5);
    expect(box.max.z).toBeCloseTo(margin, 5);
    // Mesh is frustum-culled, so this box doubles as the frustum-cull box.
    expect(mesh.frustumCulled).toBe(true);
  });

  it('falls back to a 3.0 truncation when the material has no uTruncate uniform', () => {
    const mesh = createInstancedGSplatsMesh(makeSingleSplatConfig(), new THREE.MeshBasicMaterial());
    const box = mesh.geometry.boundingBox!;
    const margin = 2 * 3.0; // maxRowNorm 2 × default truncation 3.0
    expect(box.max.x).toBeCloseTo(margin, 5);
    expect(box.min.x).toBeCloseTo(-margin, 5);
  });
});
