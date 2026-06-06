/**
 * Lines scalar plumbing end-to-end tests.
 *
 * Covers:
 *  - LinesDataAccumulator: scalar buffer init/grow/fill/getData.
 *  - projectLinesTo3D (TS path): scalars interpolated at clipped
 *    endpoints; output omits scalars when input has none.
 *  - GPU pool updateLinesGeometry: lazily allocates aStartScalar /
 *    aEndScalar when present; non-scalar updates leave geometry without
 *    those attributes.
 *  - createInstancedLinesMesh + updateInstancedLinesMesh: bind / update
 *    the scalar attributes via the shared `attrSpecs` path.
 *  - End-to-end: a LoadedLinesData with scalars produces an
 *    InstancedLinesMesh whose geometry has aStartScalar/aEndScalar →
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
  updateInstancedLinesMesh,
  type InstancedLinesMeshConfig,
} from '../../../rendering/line-geometry';
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

describe('GPU pool updateLinesGeometry scalar attribute', () => {
  it('does NOT create scalar attributes when data has no scalars', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l1', 4);
    const processed: ProcessedLinesData = {
      startPositions: new Float32Array(12),
      endPositions: new Float32Array(12),
      startColors: new Float32Array(12),
      endColors: new Float32Array(12),
      startWidths: new Float32Array(4),
      endWidths: new Float32Array(4),
      startSharpness: new Float32Array(4),
      endSharpness: new Float32Array(4),
      segmentLengths: new Float32Array(4),
      startClipped: new Uint8Array(4),
      endClipped: new Uint8Array(4),
      segmentCount: 4,
    };
    pool.updateLinesGeometry(g, processed, 4);
    expect(g.hasAttribute('aStartScalar')).toBe(false);
    expect(g.hasAttribute('aEndScalar')).toBe(false);
  });

  it('lazily creates aStartScalar/aEndScalar on first scalar commit', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l2', 2);
    const processed: ProcessedLinesData = {
      startPositions: new Float32Array(6),
      endPositions: new Float32Array(6),
      startColors: new Float32Array(6),
      endColors: new Float32Array(6),
      startWidths: new Float32Array(2),
      endWidths: new Float32Array(2),
      startSharpness: new Float32Array(2),
      endSharpness: new Float32Array(2),
      segmentLengths: new Float32Array(2),
      startClipped: new Uint8Array(2),
      endClipped: new Uint8Array(2),
      startScalars: new Float32Array([0.1, 0.9]),
      endScalars: new Float32Array([0.2, 0.8]),
      segmentCount: 2,
    };
    pool.updateLinesGeometry(g, processed, 2);
    expect(g.hasAttribute('aStartScalar')).toBe(true);
    expect(g.hasAttribute('aEndScalar')).toBe(true);
    // Pooled attributes are now `InterleavedBufferAttribute` views
    // over a shared `InstancedInterleavedBuffer` — `.array[0]` reads
    // the first float of the stride (not necessarily this attribute's
    // first value). Use the semantic `getX(i)` API instead.
    const startAttr = g.getAttribute('aStartScalar');
    expect(startAttr.getX(0)).toBeCloseTo(0.1, 5);
    expect(startAttr.getX(1)).toBeCloseTo(0.9, 5);
  });

  it('growLinesGeometry preserves scalar attribute contents on resize', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    // Acquire a small capacity, write scalars, then grow.
    const g = pool.acquireLinesGeometry('l-grow', 2);
    const small: ProcessedLinesData = {
      startPositions: new Float32Array(6),
      endPositions: new Float32Array(6),
      startColors: new Float32Array(6),
      endColors: new Float32Array(6),
      startWidths: new Float32Array(2),
      endWidths: new Float32Array(2),
      startSharpness: new Float32Array(2),
      endSharpness: new Float32Array(2),
      segmentLengths: new Float32Array(2),
      startClipped: new Uint8Array(2),
      endClipped: new Uint8Array(2),
      startScalars: new Float32Array([0.25, 0.75]),
      endScalars: new Float32Array([0.5, 1.0]),
      segmentCount: 2,
    };
    pool.updateLinesGeometry(g, small, 2);
    // Pre-grow capacity = floor(buffer.array.length / stride), reads
    // through the interleaved view (every per-instance attribute on a
    // pooled line geometry shares one buffer).
    const startBefore = g.getAttribute('aStartScalar') as THREE.InterleavedBufferAttribute;
    const endBefore = g.getAttribute('aEndScalar') as THREE.InterleavedBufferAttribute;
    const capacityBefore = Math.floor(
      (startBefore.data.array as Float32Array).length / startBefore.data.stride
    );
    const endCapacityBefore = Math.floor(
      (endBefore.data.array as Float32Array).length / endBefore.data.stride
    );

    // Force growth by acquiring with a much larger count for the same nodeId.
    pool.acquireLinesGeometry('l-grow', 200);
    const startAfter = g.getAttribute('aStartScalar') as THREE.InterleavedBufferAttribute;
    const endAfter = g.getAttribute('aEndScalar') as THREE.InterleavedBufferAttribute;
    const capacityAfter = Math.floor(
      (startAfter.data.array as Float32Array).length / startAfter.data.stride
    );
    const endCapacityAfter = Math.floor(
      (endAfter.data.array as Float32Array).length / endAfter.data.stride
    );
    expect(capacityAfter).toBeGreaterThan(capacityBefore);
    expect(endCapacityAfter).toBeGreaterThan(endCapacityBefore);
    // Preserved contents at indices [0, 1] — semantic accessors handle
    // the new strided storage transparently.
    expect(startAfter.getX(0)).toBeCloseTo(0.25, 5);
    expect(startAfter.getX(1)).toBeCloseTo(0.75, 5);
    expect(endAfter.getX(0)).toBeCloseTo(0.5, 5);
    expect(endAfter.getX(1)).toBeCloseTo(1.0, 5);
  });

  it('reuses scalar attributes on subsequent commits', () => {
    const pool = new GPUBufferPool(20, 300, 5, () => 0);
    const g = pool.acquireLinesGeometry('l3', 1);
    const make = (s: number, e: number): ProcessedLinesData => ({
      startPositions: new Float32Array(3),
      endPositions: new Float32Array(3),
      startColors: new Float32Array(3),
      endColors: new Float32Array(3),
      startWidths: new Float32Array(1),
      endWidths: new Float32Array(1),
      startSharpness: new Float32Array(1),
      endSharpness: new Float32Array(1),
      segmentLengths: new Float32Array(1),
      startClipped: new Uint8Array(1),
      endClipped: new Uint8Array(1),
      startScalars: new Float32Array([s]),
      endScalars: new Float32Array([e]),
      segmentCount: 1,
    });
    pool.updateLinesGeometry(g, make(0.1, 0.2), 1);
    const attr1 = g.getAttribute('aStartScalar') as THREE.InterleavedBufferAttribute;
    pool.updateLinesGeometry(g, make(0.5, 0.6), 1);
    const attr2 = g.getAttribute('aStartScalar') as THREE.InterleavedBufferAttribute;
    expect(attr2).toBe(attr1); // same view instance — interleaved buffer reused
    expect(attr2.getX(0)).toBeCloseTo(0.5, 5);
  });
});

describe('line-geometry mesh creation/update', () => {
  it('createInstancedLinesMesh + updateInstancedLinesMesh keep aStartScalar in sync', () => {
    const initial: InstancedLinesMeshConfig = {
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
      startScalars: new Float32Array([0.0]),
      endScalars: new Float32Array([1.0]),
      segmentCount: 1,
    };
    const mesh = createInstancedLinesMesh(initial, new LineMaterial());
    expect(mesh.geometry.hasAttribute('aStartScalar')).toBe(true);
    expect(mesh.geometry.hasAttribute('aEndScalar')).toBe(true);

    // Update with new scalars
    const updated: InstancedLinesMeshConfig = {
      ...initial,
      startScalars: new Float32Array([0.25]),
      endScalars: new Float32Array([0.75]),
    };
    updateInstancedLinesMesh(mesh, updated);
    const startAttr = mesh.geometry.getAttribute('aStartScalar');
    // Pooled / standalone line attributes are now interleaved views —
    // use `getX(i)` for semantic per-instance reads.
    expect(startAttr.getX(0)).toBeCloseTo(0.25, 5);
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
