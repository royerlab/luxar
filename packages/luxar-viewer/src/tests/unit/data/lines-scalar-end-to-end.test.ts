/**
 * Lines scalar plumbing end-to-end tests.
 *
 * Covers:
 *  - LinesDataAccumulator: scalar buffer init/grow/fill/getData.
 *  - projectLinesTo3D (TS path): scalars interpolated at clipped
 *    endpoints; output omits scalars when input has none.
 *  - GPU pool updateLinesGeometry: scalars land in texel5.xy of the
 *    fixed 6-texel line-texture layout; non-scalar updates write the
 *    0.0 identity there and presence rides `userData.hasScalars`
 *    (refreshed every write — pool geometries are reused across
 *    tenants).
 *  - createInstancedLinesMesh + updateInstancedLinesMesh: write /
 *    update the scalar texels in place — a scalar toggle never
 *    rebuilds (the fixed layout always carries the slots).
 *  - End-to-end: a LoadedLinesData with scalars produces an
 *    InstancedLinesMesh whose geometry carries the presence stamp →
 *    `supportsScalarColormap('lines', geometry)` returns true.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { LinesDataAccumulator } from '../../../data/accumulators/lines';
// The main-thread lines projection copy was deleted in W4; drive the live
// worker dispatcher in-process via the adapter, backed by the TypeScript
// reference (always available without a compiled WASM build).
import { projectLinesViaDispatcher } from '../../helpers/projection-adapters';
import { TypeScriptFallback } from '../../../wasm/typescript';
import { GPUBufferPool } from '../../../rendering/gpu-buffer-pool';
import {
  createInstancedLinesMesh,
  getLineTexture,
  updateInstancedLinesMesh,
  type InstancedLinesMeshConfig,
} from '../../../rendering/line-geometry';
import { LINE_FLOATS_PER_SEGMENT } from '../../../rendering/element-texture-layout';
import { LineMaterial } from '../../../rendering/materials/line/material-glsl';
import { supportsScalarColormap } from '../../../rendering/material-colormap-helpers';
import type { LoadedLinesData, ProcessedLinesData } from '../../../types/lines';

/** TypeScript-reference backend for the in-process dispatcher adapter. */
const lineBackend = new TypeScriptFallback();

function loadedLines({
  positions,
  segments,
  widths,
  scalars,
}: {
  positions: Float32Array;
  segments: Uint32Array;
  widths: Float32Array;
  scalars?: Float32Array;
}): LoadedLinesData {
  const ndim = 3;
  const vertexCount = positions.length / ndim;
  return {
    positions,
    segments,
    widths,
    colors: null,
    sharpness: null,
    ...(scalars ? { scalars } : {}),
    segmentCount: segments.length / 2,
    vertexCount,
    ndim,
  };
}

function meshConfig(data: ProcessedLinesData, withScalars: boolean): InstancedLinesMeshConfig {
  const base: InstancedLinesMeshConfig = {
    startPositions: data.startPositions,
    endPositions: data.endPositions,
    startColors: data.startColors,
    endColors: data.endColors,
    startWidths: data.startWidths,
    endWidths: data.endWidths,
    startSharpness: data.startSharpness,
    endSharpness: data.endSharpness,
    segmentLengths: data.segmentLengths,
    startClipped: data.startClipped,
    endClipped: data.endClipped,
    segmentCount: data.segmentCount,
  };
  if (withScalars && data.startScalars && data.endScalars) {
    return {
      ...base,
      startScalars: data.startScalars,
      endScalars: data.endScalars,
    };
  }
  return base;
}

describe('LinesDataAccumulator scalar buffer', () => {
  it('flips hasScalars on markScalarsLoaded()', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
    });
    let data = acc.getData(1, 1);
    expect(data.scalars).toBeUndefined();

    // Direct buffer write + markScalarsLoaded mirrors the loader's path.
    acc.getScalarBuffer()[0] = 0.5;
    acc.markScalarsLoaded();
    data = acc.getData(1, 1);
    expect(data.scalars).toBeInstanceOf(Float32Array);
    expect(data.scalars![0]).toBeCloseTo(0.5, 5);
  });

  it('flips hasScalars on fill() with scalars', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
      scalars: new Float32Array([0.7]),
    });
    const data = acc.getData(1, 1);
    expect(data.scalars).toBeInstanceOf(Float32Array);
    expect(data.scalars![0]).toBeCloseTo(0.7, 5);
  });

  it('returns scalars: undefined when no scalars were filled', () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
    });
    const data = acc.getData(1, 1);
    expect(data.scalars).toBeUndefined();
  });

  it('grows scalar buffer with vertex capacity', () => {
    const acc = new LinesDataAccumulator(2, 2, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
      scalars: new Float32Array([0.25]),
    });
    acc.ensureCapacity(100);
    expect(acc.getScalarBuffer().length).toBeGreaterThanOrEqual(100);
    expect(acc.getScalarBuffer()[0]).toBeCloseTo(0.25, 5);
  });

  it('dispose() resets scalar state', () => {
    const acc = new LinesDataAccumulator(4, 4, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0]),
      segments: new Uint32Array([0, 0]),
      widths: new Float32Array([0.1]),
      scalars: new Float32Array([0.9]),
    });
    acc.dispose();
    expect(acc.getScalarBuffer().length).toBe(0);
    const data = acc.getData(0, 0);
    expect(data.scalars).toBeUndefined();
  });
});

describe('projectLinesTo3D scalar interpolation', () => {
  it('passes scalars through unclipped segments unchanged', async () => {
    const data = loadedLines({
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: new Float32Array([0.0, 1.0]),
    });
    // No clipping — slice covers full range.
    const out = await projectLinesViaDispatcher(
      lineBackend,
      data,
      [0, 0, 0, 0],
      [Infinity, Infinity, Infinity, Infinity],
      [0, 1, 2]
    );
    expect(out.startScalars).toBeDefined();
    expect(out.endScalars).toBeDefined();
    expect(out.startScalars![0]).toBeCloseTo(0.0, 5);
    expect(out.endScalars![0]).toBeCloseTo(1.0, 5);
  });

  it('omits scalars from output when input has no scalars', async () => {
    const data = loadedLines({
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: undefined,
    });
    const out = await projectLinesViaDispatcher(
      lineBackend,
      data,
      [0, 0, 0, 0],
      [Infinity, Infinity, Infinity, Infinity],
      [0, 1, 2]
    );
    expect(out.startScalars).toBeUndefined();
    expect(out.endScalars).toBeUndefined();
  });

  it('rejects when scalar length mismatches vertex count', async () => {
    // 2 vertices, but only 1 scalar — too short. The worker dispatcher
    // validates per-vertex scalar length and throws (fail-hard), matching
    // the Points/GSplats validators (three-geometry symmetry). The deleted
    // main-thread copy fail-soft-suppressed instead; W4 makes Lines
    // consistent with the other geometries and the large-data worker path.
    const data = loadedLines({
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: new Float32Array([0.5]) as Float32Array,
    });
    await expect(
      projectLinesViaDispatcher(
        lineBackend,
        data,
        [0, 0, 0, 0],
        [Infinity, Infinity, Infinity, Infinity],
        [0, 1, 2]
      )
    ).rejects.toThrow(/scalars too short/);
  });

  it('roundtrips Uint8 scalars through accumulator + projection', async () => {
    const acc = new LinesDataAccumulator(64, 32, 3);
    acc.fill(0, 0, {
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: new Uint8Array([64, 192]),
    });
    // Accumulator preserves Uint8 dtype natively.
    const buf = acc.getScalarBuffer();
    expect(buf).toBeInstanceOf(Uint8Array);
    expect((buf as Uint8Array)[0]).toBe(64);
    expect((buf as Uint8Array)[1]).toBe(192);

    // Output of accumulator carries Uint8 scalars; the dispatcher
    // coerces them to Float32 normalized by 1/255 (colormap-shader [0,1]
    // contract). This also unifies behavior with the large-data worker
    // path, which always normalized — the deleted main-thread copy
    // raw-widened, an inconsistency W4 removes.
    const data = acc.getData(1, 2);
    expect(data.scalars).toBeInstanceOf(Uint8Array);

    const out = await projectLinesViaDispatcher(
      lineBackend,
      data,
      [0, 0, 0, 0],
      [Infinity, Infinity, Infinity, Infinity],
      [0, 1, 2]
    );
    expect(out.startScalars).toBeInstanceOf(Float32Array);
    expect(out.endScalars).toBeInstanceOf(Float32Array);
    // Unclipped, t1=0 ⇒ start = scalar[0], t2=1 ⇒ end = scalar[1], each
    // normalized by 1/255.
    expect(out.startScalars![0]).toBeCloseTo(64 / 255, 5);
    expect(out.endScalars![0]).toBeCloseTo(192 / 255, 5);
  });
});

/**
 * A minimal ProcessedLinesData for pool-update tests; scalars ride
 * along when provided (per-segment start/end pairs).
 */
function processedLines(
  segmentCount: number,
  scalars?: { start: number[]; end: number[] }
): ProcessedLinesData {
  return {
    startPositions: new Float32Array(segmentCount * 3),
    endPositions: new Float32Array(segmentCount * 3),
    startColors: new Float32Array(segmentCount * 3),
    endColors: new Float32Array(segmentCount * 3),
    startWidths: new Float32Array(segmentCount),
    endWidths: new Float32Array(segmentCount),
    startSharpness: new Float32Array(segmentCount),
    endSharpness: new Float32Array(segmentCount),
    segmentLengths: new Float32Array(segmentCount),
    startClipped: new Uint8Array(segmentCount),
    endClipped: new Uint8Array(segmentCount),
    ...(scalars
      ? {
          startScalars: new Float32Array(scalars.start),
          endScalars: new Float32Array(scalars.end),
        }
      : {}),
    segmentCount,
  };
}

/** texel5 offsets of segment `i`: [startScalar, endScalar, alpha, alpha]. */
function scalarTexels(geometry: THREE.BufferGeometry, i: number): number[] {
  const arr = getLineTexture(geometry)!.image.data as Float32Array;
  const o = i * LINE_FLOATS_PER_SEGMENT + 20;
  return [arr[o], arr[o + 1], arr[o + 2], arr[o + 3]];
}

describe('GPU pool updateLinesGeometry scalar texels', () => {
  it('writes the texel5 identity (0.0 scalars, 1.0 alphas) and stamps hasScalars=false when data has no scalars', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l1', 4);
    pool.updateLinesGeometry(g, processedLines(4), 4);
    expect(g.userData.hasScalars).toBe(false);
    // Identity fills are written UNCONDITIONALLY — a reused pool texture
    // must never leak a previous tenant's scalars/alphas.
    for (let i = 0; i < 4; i++) {
      expect(scalarTexels(g, i)).toEqual([0.0, 0.0, 1.0, 1.0]);
    }
  });

  it('lands scalars at texel5.xy and stamps hasScalars=true (fixed layout — no acquire-time declaration)', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l2', 2);
    pool.updateLinesGeometry(g, processedLines(2, { start: [0.1, 0.9], end: [0.2, 0.8] }), 2);
    expect(g.userData.hasScalars).toBe(true);
    // Byte-level layout check: startScalar/endScalar per segment at the
    // documented texel5 offsets, opacity alphas at the 1.0 identity.
    expect(scalarTexels(g, 0)[0]).toBeCloseTo(0.1, 5);
    expect(scalarTexels(g, 0)[1]).toBeCloseTo(0.2, 5);
    expect(scalarTexels(g, 1)[0]).toBeCloseTo(0.9, 5);
    expect(scalarTexels(g, 1)[1]).toBeCloseTo(0.8, 5);
    expect(scalarTexels(g, 0).slice(2)).toEqual([1.0, 1.0]);
  });

  it('grow = release + reacquire: a larger acquire returns a FRESH geometry and pools the old one', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l-grow', 2);

    // Growth is NEVER an in-place rebuild (that strands the old GPU
    // resources in the renderer caches — permanent leak under the WebGPU
    // renderer). The undersized geometry is released to the pool intact
    // and a fresh one is allocated; content carry-forward is not needed
    // because every commit rewrites the texels for the full count right
    // after acquire.
    const before = pool.getStats();
    const grown = pool.acquireLinesGeometry('l-grow', 200);
    expect(grown).not.toBe(g);
    expect(getLineTexture(grown)).not.toBeNull();
    expect(pool.didLastAcquireRebuildAttributes()).toBe(true);
    expect(pool.getStats().capacityGrowths).toBe(before.capacityGrowths + 1);

    // The old geometry went back to the pool with its texture intact.
    expect(getLineTexture(g)).not.toBeNull();
    expect(pool.getStats().pooledBuffers).toBeGreaterThan(0);
  });

  it('presence stamp flips across pool-reuse tenants and stale scalars are identity-refilled', () => {
    // The interleaved era bucketed pool geometries by scalar spec-set
    // (and THREW when scalar data hit a base-only geometry). The fixed
    // texel layout retires both: ANY pooled lines geometry fits ANY
    // lines node, so the presence stamp + identity refill are the sole
    // guards against a previous tenant leaking through.
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('tenant-a', 2);
    pool.updateLinesGeometry(g, processedLines(2, { start: [0.3, 0.7], end: [0.4, 0.6] }), 2);
    expect(g.userData.hasScalars).toBe(true);

    pool.releaseLinesGeometry('tenant-a');
    const adopted = pool.acquireLinesGeometry('tenant-b', 2);
    expect(adopted).toBe(g); // best-fit reuse hands back the same geometry

    pool.updateLinesGeometry(adopted, processedLines(2), 2);
    expect(adopted.userData.hasScalars).toBe(false);
    // tenant-a's scalars must not survive in texel5.xy.
    expect(scalarTexels(adopted, 0)).toEqual([0.0, 0.0, 1.0, 1.0]);
    expect(scalarTexels(adopted, 1)).toEqual([0.0, 0.0, 1.0, 1.0]);
  });

  it('overwrites scalar texels in place on subsequent commits (same texture instance)', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l3', 1);
    pool.updateLinesGeometry(g, processedLines(1, { start: [0.1], end: [0.2] }), 1);
    const texture1 = getLineTexture(g);
    pool.updateLinesGeometry(g, processedLines(1, { start: [0.5], end: [0.6] }), 1);
    expect(getLineTexture(g)).toBe(texture1); // same texture — no storage rebuild
    expect(scalarTexels(g, 0)[0]).toBeCloseTo(0.5, 5);
    expect(scalarTexels(g, 0)[1]).toBeCloseTo(0.6, 5);
  });
});

describe('line-geometry mesh creation/update', () => {
  const baseConfig = (): InstancedLinesMeshConfig => ({
    startPositions: new Float32Array([0, 0, 0]),
    endPositions: new Float32Array([1, 0, 0]),
    startColors: new Float32Array([1, 1, 1]),
    endColors: new Float32Array([1, 1, 1]),
    startWidths: new Float32Array([0.1]),
    endWidths: new Float32Array([0.1]),
    startSharpness: new Float32Array([2.0]),
    endSharpness: new Float32Array([2.0]),
    segmentLengths: new Float32Array([1.0]),
    startClipped: new Uint8Array([0]),
    endClipped: new Uint8Array([0]),
    segmentCount: 1,
  });

  it('createInstancedLinesMesh + updateInstancedLinesMesh keep the scalar texels in sync', () => {
    const initial: InstancedLinesMeshConfig = {
      ...baseConfig(),
      startScalars: new Float32Array([0.0]),
      endScalars: new Float32Array([1.0]),
    };
    const mesh = createInstancedLinesMesh(initial, new LineMaterial());
    expect(mesh.geometry.userData.hasScalars).toBe(true);
    expect(scalarTexels(mesh.geometry, 0)).toEqual([0.0, 1.0, 1.0, 1.0]);

    // Update with new scalars
    const updated: InstancedLinesMeshConfig = {
      ...initial,
      startScalars: new Float32Array([0.25]),
      endScalars: new Float32Array([0.75]),
    };
    const rebuilt = updateInstancedLinesMesh(mesh, updated);
    // Same count → in-place texel write, no storage rebuild — the commit
    // layer must NOT invalidate the cached RenderObject.
    expect(rebuilt).toBe(false);
    expect(scalarTexels(mesh.geometry, 0)[0]).toBeCloseTo(0.25, 5);
    expect(scalarTexels(mesh.geometry, 0)[1]).toBeCloseTo(0.75, 5);
  });

  it('a scalar toggle does NOT rebuild — the fixed layout always has the texel5 slots', () => {
    const base = baseConfig();
    const mesh = createInstancedLinesMesh(base, new LineMaterial());
    const geometryBefore = mesh.geometry;
    expect(mesh.geometry.userData.hasScalars).toBe(false);
    expect(scalarTexels(mesh.geometry, 0)).toEqual([0.0, 0.0, 1.0, 1.0]);

    // Toggle scalars ON: same count → in-place write; only the presence
    // stamp and texel5.xy change (the interleaved era rebuilt here).
    const withScalars: InstancedLinesMeshConfig = {
      ...base,
      startScalars: new Float32Array([0.5]),
      endScalars: new Float32Array([0.5]),
    };
    expect(updateInstancedLinesMesh(mesh, withScalars)).toBe(false);
    expect(mesh.geometry).toBe(geometryBefore);
    expect(mesh.geometry.userData.hasScalars).toBe(true);
    expect(scalarTexels(mesh.geometry, 0)).toEqual([0.5, 0.5, 1.0, 1.0]);

    // Toggle scalars OFF again: stamp flips back and the identity fill
    // scrubs the stale scalar values.
    expect(updateInstancedLinesMesh(mesh, baseConfig())).toBe(false);
    expect(mesh.geometry.userData.hasScalars).toBe(false);
    expect(scalarTexels(mesh.geometry, 0)).toEqual([0.0, 0.0, 1.0, 1.0]);
  });
});

describe('end-to-end scalar binding for Lines', () => {
  it('processed → mesh → supportsScalarColormap returns true', async () => {
    const data = loadedLines({
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: new Float32Array([0.0, 1.0]),
    });
    const processed = await projectLinesViaDispatcher(
      lineBackend,
      data,
      [0, 0, 0, 0],
      [Infinity, Infinity, Infinity, Infinity],
      [0, 1, 2]
    );
    const mesh = createInstancedLinesMesh(meshConfig(processed, true), new LineMaterial());
    expect(supportsScalarColormap('lines', mesh.geometry)).toBe(true);
  });

  it('without scalars, supportsScalarColormap returns false', async () => {
    const data = loadedLines({
      positions: new Float32Array([0, 0, 0, 1, 0, 0]),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([0.1, 0.1]),
      scalars: undefined,
    });
    const processed = await projectLinesViaDispatcher(
      lineBackend,
      data,
      [0, 0, 0, 0],
      [Infinity, Infinity, Infinity, Infinity],
      [0, 1, 2]
    );
    const mesh = createInstancedLinesMesh(meshConfig(processed, false), new LineMaterial());
    expect(supportsScalarColormap('lines', mesh.geometry)).toBe(false);
  });
});
