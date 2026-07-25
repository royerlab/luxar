/**
 * End-to-end commit→material pin for the RGBA alpha-presence chain,
 * three-geometry symmetric.
 *
 * The per-geometry commit tests (`commit-{gsplats,points,lines}-geometry.
 * test.ts`) vi.mock the GPU writers and/or the material sync helpers, so
 * no checked-in test pinned the FULL chain:
 *
 *   commit* → REAL GPUBufferPool adapter → texel writer →
 *   stamp*PresenceFlags (geometry.userData.hasElementAlpha) →
 *   sync*MaterialWithGeometry → material.uniforms.uHasElementAlpha
 *
 * These tests drive the real commit functions with a REAL pool and REAL
 * render materials (nothing along the chain is mocked) and assert the
 * uniform value the volumetric w(a) optical-depth map gates on:
 *
 *   1. RGBA commit (colorComponents: 4) → uniform becomes 1.
 *   2. RGB recommit on the same mesh → uniform returns to 0 (the pool
 *      tenant refresh through the real presence stamp).
 *   3. A stamp-only no-op commit (memoized same-reference data /
 *      `staged.noop`) → uniform untouched (sentinel-poked to prove the
 *      sync helper did not run).
 *
 * All three geometries exercise the POOL branch of their commit (the
 * production default); the points nodeFactory stub throws to pin that.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';

import { commitGSplatsGeometry } from '../../../../data/scene-loader/commit/commit-gsplats-geometry';
import { commitPointsGeometry } from '../../../../data/scene-loader/commit/commit-points-geometry';
import { commitLinesGeometry } from '../../../../data/scene-loader/commit/commit-lines-geometry';
import { GPUBufferPool } from '../../../../rendering/gpu-buffer-pool';
import { GSplatMaterial } from '../../../../rendering/materials/gsplat/material-glsl';
import { PointMaterial } from '../../../../rendering/materials/point/material-glsl';
import { LineMaterial } from '../../../../rendering/materials/line/material-glsl';
import type { StagedGSplatsCommit } from '../../../../data/scene-loader/process/data-processor-gsplats';
import type { StagedLinesCommit } from '../../../../data/scene-loader/process/data-processor-lines';
import type { LoadedGSplatsData } from '../../../../types/gsplats';
import type { LoadedLinesData, ProcessedLinesData } from '../../../../types/lines';
import type { LoadedPointsData } from '../../../../types/points';
import type { NodeFactory } from '../../../../rendering/node-factory';

/** Read the alpha-presence uniform off a real material. */
function alphaUniform(mesh: THREE.Mesh): number {
  return (mesh.material as GSplatMaterial | PointMaterial | LineMaterial).uniforms.uHasElementAlpha
    .value;
}

/** Poke a sentinel so a no-op commit's untouched uniform is observable. */
function pokeAlphaUniform(mesh: THREE.Mesh, value: number): void {
  (mesh.material as GSplatMaterial | PointMaterial | LineMaterial).uniforms.uHasElementAlpha.value =
    value;
}

// =============================================================================
// GSplats
// =============================================================================

function makeGSplatsSource(splatCount: number, rgba: boolean): LoadedGSplatsData {
  return {
    positions: new Float32Array(splatCount * 3),
    amplitudes: new Float32Array(splatCount).fill(1),
    choleskyFactors: new Float32Array(splatCount * 6).fill(0.1),
    colors: rgba ? new Float32Array(splatCount * 4).fill(1) : null,
    colorComponents: rgba ? 4 : undefined,
    splatCount,
    ndim: 3,
  };
}

function makeGSplatsStaged(splatCount: number, rgba: boolean): StagedGSplatsCommit {
  const colorK = rgba ? 4 : 3;
  return {
    path: '/g',
    sourceData: makeGSplatsSource(splatCount, rgba),
    processed: {
      centers3D: new Float32Array(splatCount * 3),
      choleskyFactors3D: new Float32Array(splatCount * 6).fill(0.1),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: new Float32Array(splatCount * colorK).fill(1),
      colorComponents: rgba ? 4 : undefined,
      splatCount,
    },
    cholesky01: new Float32Array(splatCount * 2).fill(0.1),
    cholesky23: new Float32Array(splatCount * 2).fill(0.1),
    cholesky45: new Float32Array(splatCount * 2).fill(0.1),
  };
}

function makeGSplatsMesh(): { root: THREE.Group; mesh: THREE.Mesh } {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new GSplatMaterial());
  mesh.name = '/g';
  mesh.userData = { nodeType: 'gsplats', attrs: {}, visibleSplatCount: 0 };
  const root = new THREE.Group();
  root.add(mesh);
  return { root, mesh };
}

describe('commit alpha end-to-end — gsplats (real pool + real GSplatMaterial)', () => {
  it('RGBA commit drives uHasElementAlpha to 1 through the full unmocked chain', () => {
    const { root, mesh } = makeGSplatsMesh();
    const pool = new GPUBufferPool(20, 300);
    expect(alphaUniform(mesh)).toBe(0); // constructor default

    commitGSplatsGeometry(makeGSplatsStaged(4, /*rgba=*/ true), root, pool, undefined, 1);

    expect(alphaUniform(mesh)).toBe(1);
    expect((mesh.userData as { visibleSplatCount: number }).visibleSplatCount).toBe(4);
  });

  it('RGB recommit on the same mesh returns the uniform to 0 (pool tenant refresh)', () => {
    const { root, mesh } = makeGSplatsMesh();
    const pool = new GPUBufferPool(20, 300);
    commitGSplatsGeometry(makeGSplatsStaged(4, /*rgba=*/ true), root, pool, undefined, 1);
    expect(alphaUniform(mesh)).toBe(1);

    commitGSplatsGeometry(makeGSplatsStaged(4, /*rgba=*/ false), root, pool, undefined, 2);

    expect(alphaUniform(mesh)).toBe(0);
  });

  it('stamp-only noop commit leaves the uniform untouched', () => {
    const { root, mesh } = makeGSplatsMesh();
    const pool = new GPUBufferPool(20, 300);
    const staged = makeGSplatsStaged(4, /*rgba=*/ true);
    commitGSplatsGeometry(staged, root, pool, undefined, 1);
    expect(alphaUniform(mesh)).toBe(1);

    // Sentinel: if the noop path wrongly re-ran the sync helper, the
    // geometry's hasElementAlpha stamp (true) would restore the uniform to 1.
    pokeAlphaUniform(mesh, 0);
    commitGSplatsGeometry(
      { path: '/g', noop: true, sourceData: staged.sourceData },
      root,
      pool,
      undefined,
      7
    );

    expect(alphaUniform(mesh)).toBe(0); // untouched
    // The stamp-only path DID run (not an early bail on a missing mesh).
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(7);
  });
});

// =============================================================================
// Points
// =============================================================================

function makePointsData(pointCount: number, rgba: boolean): LoadedPointsData {
  const colorK = rgba ? 4 : 3;
  return {
    positions: new Float32Array(pointCount * 3),
    colors: new Uint8Array(pointCount * colorK).fill(255),
    colorComponents: rgba ? 4 : undefined,
    pointCount,
    ndim: 3,
    metadata: {
      totalPoints: pointCount,
      loadedPoints: pointCount,
      bounds: new THREE.Box3(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1)),
      usedSpatialIndex: false,
    },
  };
}

function makePointsMesh(): { root: THREE.Group; mesh: THREE.Mesh } {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('aCenter', new THREE.InstancedBufferAttribute(new Float32Array(0), 3));
  const mesh = new THREE.Mesh(geometry, new PointMaterial());
  mesh.name = '/p';
  mesh.userData = { nodeType: 'points', visiblePointCount: 0 };
  const root = new THREE.Group();
  root.add(mesh);
  return { root, mesh };
}

// The pool branch never touches the factory — a throwing stub pins that
// these tests exercise the pool path, not the dispose+recreate fallback.
const throwingNodeFactory = {
  createPointsGeometry: () => {
    throw new Error('unexpected non-pool commit path');
  },
} as unknown as NodeFactory;

describe('commit alpha end-to-end — points (real pool + real PointMaterial)', () => {
  it('RGBA commit drives uHasElementAlpha to 1 through the full unmocked chain', () => {
    const { root, mesh } = makePointsMesh();
    const pool = new GPUBufferPool(20, 300);
    expect(alphaUniform(mesh)).toBe(0); // constructor default

    commitPointsGeometry(
      '/p',
      makePointsData(3, /*rgba=*/ true),
      root,
      pool,
      throwingNodeFactory,
      undefined,
      1
    );

    expect(alphaUniform(mesh)).toBe(1);
    expect((mesh.userData as { visiblePointCount: number }).visiblePointCount).toBe(3);
  });

  it('RGB recommit on the same mesh returns the uniform to 0 (pool tenant refresh)', () => {
    const { root, mesh } = makePointsMesh();
    const pool = new GPUBufferPool(20, 300);
    commitPointsGeometry(
      '/p',
      makePointsData(3, /*rgba=*/ true),
      root,
      pool,
      throwingNodeFactory,
      undefined,
      1
    );
    expect(alphaUniform(mesh)).toBe(1);

    commitPointsGeometry(
      '/p',
      makePointsData(3, /*rgba=*/ false),
      root,
      pool,
      throwingNodeFactory,
      undefined,
      2
    );

    expect(alphaUniform(mesh)).toBe(0);
  });

  it('memoized same-reference recommit is stamp-only: uniform untouched', () => {
    const { root, mesh } = makePointsMesh();
    const pool = new GPUBufferPool(20, 300);
    const data = makePointsData(3, /*rgba=*/ true);
    commitPointsGeometry('/p', data, root, pool, throwingNodeFactory, undefined, 1);
    expect(alphaUniform(mesh)).toBe(1);

    // Sentinel — a wrongly re-run sync would restore 1 from the stamp.
    pokeAlphaUniform(mesh, 0);
    commitPointsGeometry('/p', data, root, pool, throwingNodeFactory, undefined, 7);

    expect(alphaUniform(mesh)).toBe(0); // untouched
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(7);
  });
});

// =============================================================================
// Lines
// =============================================================================

function makeLinesSource(segmentCount: number, rgba: boolean): LoadedLinesData {
  const vertexCount = segmentCount * 2;
  return {
    positions: new Float32Array(vertexCount * 3),
    segments: new Uint32Array(vertexCount),
    widths: new Float32Array(vertexCount).fill(1),
    colors: rgba ? new Uint8Array(vertexCount * 4).fill(255) : null,
    colorComponents: rgba ? 4 : undefined,
    sharpness: null,
    segmentCount,
    vertexCount,
    ndim: 3,
  };
}

function makeLinesProcessed(segmentCount: number, rgba: boolean): ProcessedLinesData {
  return {
    startPositions: new Float32Array(segmentCount * 3),
    endPositions: new Float32Array(segmentCount * 3),
    startColors: new Float32Array(segmentCount * 3).fill(1),
    endColors: new Float32Array(segmentCount * 3).fill(1),
    startWidths: new Float32Array(segmentCount).fill(1),
    endWidths: new Float32Array(segmentCount).fill(1),
    startSharpness: new Float32Array(segmentCount).fill(0.5),
    endSharpness: new Float32Array(segmentCount).fill(0.5),
    segmentLengths: new Float32Array(segmentCount).fill(1),
    startClipped: new Uint8Array(segmentCount),
    endClipped: new Uint8Array(segmentCount),
    // RGBA alpha columns: presence of BOTH drives the hasElementAlpha stamp.
    startAlphas: rgba ? new Float32Array(segmentCount).fill(0.5) : undefined,
    endAlphas: rgba ? new Float32Array(segmentCount).fill(0.5) : undefined,
    segmentCount,
  };
}

function makeLinesStaged(segmentCount: number, rgba: boolean): StagedLinesCommit {
  return {
    path: '/l',
    sourceData: makeLinesSource(segmentCount, rgba),
    processed: makeLinesProcessed(segmentCount, rgba),
  };
}

function makeLinesMesh(): { root: THREE.Group; mesh: THREE.Mesh } {
  const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new LineMaterial());
  mesh.name = '/l';
  mesh.userData = { nodeType: 'lines', attrs: {}, visibleSegmentCount: 0 };
  const root = new THREE.Group();
  root.add(mesh);
  return { root, mesh };
}

describe('commit alpha end-to-end — lines (real pool + real LineMaterial)', () => {
  it('RGBA commit drives uHasElementAlpha to 1 through the full unmocked chain', () => {
    const { root, mesh } = makeLinesMesh();
    const pool = new GPUBufferPool(20, 300);
    expect(alphaUniform(mesh)).toBe(0); // constructor default

    commitLinesGeometry(makeLinesStaged(2, /*rgba=*/ true), root, pool, undefined, 1);

    expect(alphaUniform(mesh)).toBe(1);
    expect((mesh.userData as { visibleSegmentCount: number }).visibleSegmentCount).toBe(2);
  });

  it('RGB recommit on the same mesh returns the uniform to 0 (pool tenant refresh)', () => {
    const { root, mesh } = makeLinesMesh();
    const pool = new GPUBufferPool(20, 300);
    commitLinesGeometry(makeLinesStaged(2, /*rgba=*/ true), root, pool, undefined, 1);
    expect(alphaUniform(mesh)).toBe(1);

    commitLinesGeometry(makeLinesStaged(2, /*rgba=*/ false), root, pool, undefined, 2);

    expect(alphaUniform(mesh)).toBe(0);
  });

  it('stamp-only noop commit leaves the uniform untouched', () => {
    const { root, mesh } = makeLinesMesh();
    const pool = new GPUBufferPool(20, 300);
    const staged = makeLinesStaged(2, /*rgba=*/ true);
    commitLinesGeometry(staged, root, pool, undefined, 1);
    expect(alphaUniform(mesh)).toBe(1);

    // Sentinel — a wrongly re-run sync would restore 1 from the stamp.
    pokeAlphaUniform(mesh, 0);
    commitLinesGeometry(
      { path: '/l', noop: true, sourceData: staged.sourceData },
      root,
      pool,
      undefined,
      7
    );

    expect(alphaUniform(mesh)).toBe(0); // untouched
    expect((mesh.userData as { loadedViewVersion?: number }).loadedViewVersion).toBe(7);
  });
});
