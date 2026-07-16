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

import { describe, it, expect, afterEach } from 'vitest';
import * as THREE from 'three';
import {
  createInstancedGSplatsMesh,
  updateInstancedGSplatsMesh,
  packCholeskyForShader,
  writeSortedIndexOrdering,
  type InstancedGSplatsMeshConfig,
} from '../../../rendering/gsplat-geometry';
import {
  configureSplatTextureLayout,
  resetSplatTextureLayoutForTests,
} from '../../../rendering/splat-texture-layout';

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

/** A zero-filled config with `count` splats (clamp tests only need sizes). */
function makeConfig(count: number): InstancedGSplatsMeshConfig {
  return {
    centers: new Float32Array(count * 3),
    cholesky01: new Float32Array(count * 2),
    cholesky23: new Float32Array(count * 2),
    cholesky45: new Float32Array(count * 2),
    amplitudes: new Float32Array(count),
    colors: new Float32Array(count * 3),
    splatCount: count,
  };
}

describe('non-pool writers — capacity clamp self-consistency', () => {
  afterEach(() => {
    resetSplatTextureLayoutForTests();
  });

  it('createInstancedGSplatsMesh clamps instanceCount to the texture bound', () => {
    // maxTextureSize 8 → width 8, per-node bound = 8×8/4 = 16 splats.
    configureSplatTextureLayout(8);
    const mesh = createInstancedGSplatsMesh(makeConfig(100), materialWithTruncate(3.0));
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    // Instances, ordering, and texels all sized to the clamped count —
    // an over-bound node renders its clamped prefix instead of going black
    // (pre-fix, the uncapped texture height blew past maxTextureSize).
    expect(geometry.instanceCount).toBe(16);
    expect((geometry.getAttribute('aSortedIndex').array as Uint32Array).length).toBe(16);
  });

  it('updateInstancedGSplatsMesh does NOT rebuild on a same-count-over-bound recommit', () => {
    configureSplatTextureLayout(8); // bound = 16 splats
    const mesh = createInstancedGSplatsMesh(makeConfig(100), materialWithTruncate(3.0));
    const geometryBefore = mesh.geometry;
    // Same over-bound request again: both sides of the rebuild check are
    // clamped, so this must take the in-place path — comparing the raw
    // request against the clamped instanceCount would rebuild forever.
    const rebuilt = updateInstancedGSplatsMesh(mesh, makeConfig(100));
    expect(rebuilt).toBe(false);
    expect(mesh.geometry).toBe(geometryBefore);
    expect((mesh.geometry as THREE.InstancedBufferGeometry).instanceCount).toBe(16);
  });
});

describe('updateInstancedGSplatsMesh — preserveOrdering (same-node same-count recommit)', () => {
  const orderingOf = (mesh: THREE.Mesh, n: number): number[] =>
    Array.from((mesh.geometry.getAttribute('aSortedIndex').array as Uint32Array).subarray(0, n));

  it('same-size branch keeps the sort permutation when the flag is set', () => {
    const mesh = createInstancedGSplatsMesh(makeConfig(4), materialWithTruncate(3.0));
    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    // The SortWorker landed a depth-sort permutation between commits.
    writeSortedIndexOrdering(geometry, new Uint32Array([3, 2, 1, 0]), 4);

    const rebuilt = updateInstancedGSplatsMesh(mesh, makeConfig(4), { preserveOrdering: true });
    expect(rebuilt).toBe(false);
    expect(mesh.geometry).toBe(geometry);
    expect(orderingOf(mesh, 4)).toEqual([3, 2, 1, 0]);

    // Without the flag the same-size branch restores identity (default).
    updateInstancedGSplatsMesh(mesh, makeConfig(4));
    expect(orderingOf(mesh, 4)).toEqual([0, 1, 2, 3]);
  });

  it('rebuild branch IGNORES the flag — the fresh geometry gets identity', () => {
    const mesh = createInstancedGSplatsMesh(makeConfig(4), materialWithTruncate(3.0));
    writeSortedIndexOrdering(
      mesh.geometry as THREE.InstancedBufferGeometry,
      new Uint32Array([3, 2, 1, 0]),
      4
    );

    // Count change: fresh geometry+texture pair. Fresh aSortedIndex arrays
    // are zero-filled, not identity, so the identity write is structurally
    // required regardless of preserveOrdering.
    const rebuilt = updateInstancedGSplatsMesh(mesh, makeConfig(6), { preserveOrdering: true });
    expect(rebuilt).toBe(true);
    expect(orderingOf(mesh, 6)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});
