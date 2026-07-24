/**
 * Unit tests for the point-loader projection helpers.
 *
 * The full projection paths (effective-radius compaction, worker
 * fallback) are exercised by the integration tests in
 * `points-spatial-index-loader.test.ts`; the cases below pin the small
 * pure-helper contracts so the loader can rely on them:
 *   - `createEmptyPointsData` returns the right shape with the right
 *     dtype passthrough and the right `ndim` fallback (chunkIndex → 3).
 *   - `projectPointsTo3D` extracts displayed dims into the first 3
 *     position slots, fills the rest with zero, throws on missing
 *     positions, and recomputes ndim from the actual buffer length.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  createEmptyPointsData,
  projectPointsTo3D,
  type ProjectionContext,
  type ProjectionTargetBuffers,
} from '../../../../data/points/projection';
import type { ViewState, PointRange } from '../../../../data/data-loader-types';
import type { PointsMetadata, EffectiveRadiusConfig } from '../../../../types/points';
import type { PointsChunkIndex } from '../../../../data/points/chunk-index-loader';
import { LoadedPointsDataAccumulator } from '../../../../data/accumulators/points';
import { TypeScriptFallback } from '../../../../wasm/typescript';

// projectPointsTo3D is WASM-accelerated; drive it with the TS-reference
// backend (always available without a compiled build). The extraction +
// effective-radius math under test is the kernel's.
const wasm = new TypeScriptFallback();

function makeAttrs(overrides: Partial<PointsMetadata> = {}): PointsMetadata {
  return {
    type: 'points',
    n_points: 0,
    ndim: 3,
    position_dtype: 'float32',
    color_dtype: 'uint8',
    radius_dtype: 'float32',
    sharpness_dtype: 'uint8',
    ...overrides,
  } as PointsMetadata;
}

function makeChunkIndex(ndim: number): PointsChunkIndex {
  return {
    chunkBounds: new Float32Array(0),
    chunkCount: 0,
    metadata: {
      ordering: 'morton',
      ordering_dims: [],
      slice_dims: [],
      ordering_bits_per_dim: 21,
      chunk_size: 0,
      total_points: 0,
      total_chunks: 0,
      ndim,
    },
  };
}

function makeCtx(overrides: Partial<ProjectionContext> = {}): ProjectionContext {
  return {
    chunkIndex: null,
    effectiveRadiusConfig: null,
    accumulator: null,
    nodeAttrs: makeAttrs(),
    ...overrides,
  };
}

function makeViewState(overrides: Partial<ViewState> = {}): ViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0],
    tolerance: [0, 0, 0],
    ...overrides,
  };
}

describe('createEmptyPointsData', () => {
  it('returns ndim=3 when no chunkIndex is set (3D fallback)', () => {
    const data = createEmptyPointsData(makeCtx(), makeViewState());
    expect(data.ndim).toBe(3);
  });

  it('returns ndim from chunkIndex.metadata when present', () => {
    const data = createEmptyPointsData(makeCtx({ chunkIndex: makeChunkIndex(7) }), makeViewState());
    expect(data.ndim).toBe(7);
  });

  it('emits a zero-length positions buffer and pointCount=0', () => {
    const data = createEmptyPointsData(makeCtx(), makeViewState());
    expect(data.pointCount).toBe(0);
    expect(data.positions.length).toBe(0);
  });

  it('sets metadata.usedSpatialIndex=true and bounds is an empty Box3', () => {
    const data = createEmptyPointsData(makeCtx(), makeViewState());
    expect(data.metadata.usedSpatialIndex).toBe(true);
    expect((data.metadata.bounds as THREE.Box3).isEmpty()).toBe(true);
  });

  it('passes through dtypes from nodeAttrs', () => {
    const data = createEmptyPointsData(
      makeCtx({
        nodeAttrs: makeAttrs({
          position_dtype: 'float64',
          color_dtype: 'uint16',
        }),
      }),
      makeViewState()
    );
    expect(data.metadata.dtypes?.positions).toBe('float64');
    expect(data.metadata.dtypes?.colors).toBe('uint16');
  });

  it('uses nodeAttrs.n_points for totalPoints (or 0 when absent)', () => {
    const dataWithN = createEmptyPointsData(
      makeCtx({ nodeAttrs: makeAttrs({ n_points: 1234 }) }),
      makeViewState()
    );
    expect(dataWithN.metadata.totalPoints).toBe(1234);

    const dataDefault = createEmptyPointsData(makeCtx(), makeViewState());
    expect(dataDefault.metadata.totalPoints).toBe(0);
  });
});

describe('projectPointsTo3D', () => {
  it('throws when positions is null', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        null,
        null,
        null,
        null,
        makeViewState(),
        [{ start: 0, end: 1 }] as PointRange[],
        makeCtx()
      )
    ).toThrow(/Positions data is required/);
  });

  it('extracts the three displayed dimensions for a 5D buffer', () => {
    // 2 points × 5 dims; pick dims 1, 3, 4
    // Point 0: (10, 11, 12, 13, 14)  → projected (11, 13, 14)
    // Point 1: (20, 21, 22, 23, 24)  → projected (21, 23, 24)
    const positions = new Float32Array([10, 11, 12, 13, 14, 20, 21, 22, 23, 24]);
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({
        displayDims: [1, 3, 4],
        slicePosition: [0, 0, 0, 0, 0],
        tolerance: [0, 0, 0, 0, 0],
      }),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.pointCount).toBe(2);
    expect(Array.from(result.positions)).toEqual([11, 13, 14, 21, 23, 24]);
  });

  it('zero-fills positions[1..2] when only one displayed dimension is set', () => {
    // 2 points × 2 dims; only display dim 0
    const positions = new Float32Array([5, 99, 6, 99]);
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({
        displayDims: [0],
        slicePosition: [0, 0],
        tolerance: [0, 0],
      }),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(Array.from(result.positions)).toEqual([5, 0, 0, 6, 0, 0]);
  });

  it('infers ndim from positions.length / numPoints (not chunkIndex)', () => {
    // 1 point × 4 dims, but chunkIndex says ndim=99 — actual ndim wins.
    const positions = new Float32Array([1, 2, 3, 4]);
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ chunkIndex: makeChunkIndex(99) })
    );
    expect(result.ndim).toBe(4);
  });

  it('returns metadata.usedSpatialIndex=true and totalPoints from nodeAttrs.n_points', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([1, 2, 3]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ nodeAttrs: makeAttrs({ n_points: 999 }) })
    );
    expect(result.metadata.usedSpatialIndex).toBe(true);
    expect(result.metadata.totalPoints).toBe(999);
  });

  it('falls back to range total for totalPoints when n_points is 0', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([1, 2, 3, 4, 5, 6]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx({ nodeAttrs: makeAttrs({ n_points: 0 }) })
    );
    expect(result.metadata.totalPoints).toBe(2);
  });
});

describe('projectPointsTo3D — fallback path (no accumulator, no targetBuffers)', () => {
  // The accumulator path is the supported hot path used in production.
  // The fallback path runs when targetBuffers is null/undefined — used
  // for tests and the explicit no-accumulator opt-out.

  it('allocates a fresh Float32Array for positions3D (length = numPoints * 3)', () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]); // 2 points × 3 dims
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.positions).toBeInstanceOf(Float32Array);
    expect(result.positions.length).toBe(6);
    // Fresh allocation: not the same buffer reference as the input.
    expect(result.positions).not.toBe(positions);
  });

  it('passes Float32 colors through to the result (same reference)', () => {
    const colors = new Float32Array([1.0, 0.0, 0.0, 0.5, 0.5, 0.5]);
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1]),
      colors,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.colors).toBeInstanceOf(Float32Array);
    expect(result.colors).toBe(colors);
  });

  it('passes Uint8 colors through to the result (same reference, dtype preserved)', () => {
    const colors = new Uint8Array([255, 0, 0, 0, 128, 255]);
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1]),
      colors,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.colors).toBeInstanceOf(Uint8Array);
    expect(result.colors).toBe(colors);
  });

  it('passes Uint16 colors through to the result (HDR dtype preserved)', () => {
    const colors = new Uint16Array([65535, 0, 0, 0, 32768, 65535]);
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1]),
      colors,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.colors).toBeInstanceOf(Uint16Array);
    expect(result.colors).toBe(colors);
  });

  it('passes Float32 radii through (same reference) when no effectiveRadiusConfig', () => {
    const radii = new Float32Array([0.5, 1.0]);
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1]),
      null,
      radii,
      null,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.radii).toBeInstanceOf(Float32Array);
    expect(result.radii).toBe(radii);
  });

  it('passes sharpness through to the result (same reference)', () => {
    const sharpness = new Uint8Array([200, 100]);
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1]),
      null,
      null,
      sharpness,
      makeViewState(),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx()
    );
    expect(result.sharpness).toBe(sharpness);
  });

  it('metadata.dtypes mirrors the nodeAttrs dtype declarations', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({
        nodeAttrs: makeAttrs({
          position_dtype: 'float32',
          color_dtype: 'uint16',
          radius_dtype: 'float32',
          sharpness_dtype: 'uint8',
        }),
      })
    );
    expect(result.metadata.dtypes).toEqual({
      positions: 'float32',
      colors: 'uint16',
      radii: 'float32',
      sharpness: 'uint8',
    });
  });

  it('metadata.usedEffectiveRadius is false when effectiveRadiusConfig is null', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0]),
      null,
      new Float32Array([1.0]),
      null,
      makeViewState(),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: null })
    );
    expect(result.metadata.usedEffectiveRadius).toBe(false);
  });

  it('loadedPoints reflects post-projection count (matches range total when no filtering)', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx()
    );
    expect(result.metadata.loadedPoints).toBe(3);
    expect(result.pointCount).toBe(3);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// [P5] BOUNDARY: Uint8 radius normalization (÷255) before effective-radius use.
//
// Verified against the REAL source: the `/255` normalization (projection.ts
// ~L241-262) only runs when `finalRadii instanceof Uint8Array`, which in turn
// only happens on the targetBuffers branch where BOTH the input radii AND the
// accumulator radii buffer are Uint8Array. The normalized radii are NOT
// surfaced verbatim in `result.radii` (the Uint8 accumulator buffer is what
// getData() returns), so we pin the normalization via its observable
// CONSEQUENCE: with a hidden-dim distance large vs the *normalized* radius the
// point's effective radius (√(R²−D²)) clamps to 0 and the point is filtered
// out — which only happens if R was scaled 255→1.0 first. Without the ÷255
// the un-normalized R=255 would dwarf D=100 and the point would survive.
// ════════════════════════════════════════════════════════════════════════════

describe('Uint8 radius normalization (÷255) before effective-radius use', () => {
  function makeErConfig(overrides: Partial<EffectiveRadiusConfig> = {}): EffectiveRadiusConfig {
    return {
      // dims 0..2 displayed (in-plane); dim 3 is a non-displayed SPATIAL dim
      // so shouldApplyEffectiveRadius() returns true and the Pythagorean path runs.
      spatialExtendDims: [true, true, true, true],
      maxRadius: 255,
      ...overrides,
    };
  }

  it('normalizes Uint8 radius so √(R²−D²) clamps to 0 and the far point is filtered out', () => {
    // 1 point at dim3 = 100 (far in the hidden spatial dim), Uint8 radius 255.
    // Normalized radius = 1.0; D = 100 → R_eff = √(1 − 10000) < 0 → 0 → filtered.
    const accumulator = new LoadedPointsDataAccumulator(8, 4, 1);
    // Pin radius buffer type to Uint8 by filling once with a Uint8 radius.
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      radii: new Uint8Array([255]),
    });

    const targetBuffers: ProjectionTargetBuffers = {
      positions3D: accumulator.getPositionBuffer(),
      colors: accumulator.getColorBuffer(),
      radii: accumulator.getRadiiBuffer(),
      sharpness: accumulator.getSharpnessBuffer(),
    };
    expect(accumulator.getRadiiBuffer()).toBeInstanceOf(Uint8Array);

    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 100]), // 1 point × 4 dims; dim3 = 100
      null,
      new Uint8Array([255]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0], // slice at dim3 = 0 → distance 100
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: makeErConfig(), accumulator }),
      targetBuffers
    );

    // Normalized R=1.0 ≪ D=100 → effective radius 0 → point filtered out.
    // (The all-filtered path returns createEmptyPointsData, whose metadata
    // does not carry usedEffectiveRadius — pointCount 0 is the observable.)
    expect(result.pointCount).toBe(0);
  });

  it('on-slice Uint8 radius (D=0) yields effective radius ≈ normalized R (1.0), point kept', () => {
    // Same setup but the point sits ON the slice (dim3 = 0). D = 0 →
    // R_eff = √(R²) = R = 1.0 (the normalized radius). The point survives.
    const accumulator = new LoadedPointsDataAccumulator(8, 4, 1);
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0]),
      radii: new Uint8Array([255]),
    });
    const targetBuffers: ProjectionTargetBuffers = {
      positions3D: accumulator.getPositionBuffer(),
      colors: accumulator.getColorBuffer(),
      radii: accumulator.getRadiiBuffer(),
      sharpness: accumulator.getSharpnessBuffer(),
    };

    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 0]), // dim3 = 0 → on slice
      null,
      new Uint8Array([255]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: makeErConfig(), accumulator }),
      targetBuffers
    );

    // Point survives (accumulator path returns getData(), so metadata does
    // not carry usedEffectiveRadius — pointCount 1 is the observable).
    expect(result.pointCount).toBe(1);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// [P5] BOUNDARY: multi-range projection. `projectPointsTo3D` derives the point
// COUNT from `Σ(range.end − range.start)` and reads positions as a contiguous
// `totalPoints × ndim` buffer (it does NOT index into positions by range
// bounds). Pin that the per-range count sum drives the output point count and
// that positions are projected in buffer order.
// ════════════════════════════════════════════════════════════════════════════

describe('multi-range projection', () => {
  it('sums non-contiguous range lengths into the total count and projects all positions', () => {
    // ranges [0,2) + [5,7) → totalPoints = 2 + 2 = 4. Positions buffer holds
    // exactly 4 points × 3 dims, projected in order.
    const positions = new Float32Array([
      0,
      0,
      0, // pt 0
      1,
      1,
      1, // pt 1
      2,
      2,
      2, // pt 2
      3,
      3,
      3, // pt 3
    ]);
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({ displayDims: [0, 1, 2] }),
      [
        { start: 0, end: 2 },
        { start: 5, end: 7 },
      ] as PointRange[],
      makeCtx()
    );

    expect(result.pointCount).toBe(4);
    expect(Array.from(result.positions)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
  });

  it('multi-range with non-standard displayDims projects each range point correctly', () => {
    // 4 points × 4 dims; display dims [3, 0, 1] (out-of-order). totalPoints = 4.
    const positions = new Float32Array([
      10,
      11,
      12,
      13, // pt 0 → (13, 10, 11)
      20,
      21,
      22,
      23, // pt 1 → (23, 20, 21)
      30,
      31,
      32,
      33, // pt 2 → (33, 30, 31)
      40,
      41,
      42,
      43, // pt 3 → (43, 40, 41)
    ]);
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({
        displayDims: [3, 0, 1],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [
        { start: 0, end: 1 },
        { start: 10, end: 13 },
      ] as PointRange[], // lengths 1 + 3 = 4
      makeCtx()
    );

    expect(result.pointCount).toBe(4);
    expect(Array.from(result.positions)).toEqual([13, 10, 11, 23, 20, 21, 33, 30, 31, 43, 40, 41]);
  });
});

describe('scalar length validation', () => {
  it('suppresses scalars when scalars.length !== point count', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx(),
      null,
      // 2 scalars for 3 points — mismatch should drop the scalar field.
      new Float32Array([0.1, 0.9])
    );
    expect(result.scalars).toBeUndefined();
  });

  it('keeps scalars when length matches point count', () => {
    const result = projectPointsTo3D(
      wasm,
      new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      null,
      null,
      null,
      makeViewState(),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx(),
      null,
      new Float32Array([0.1, 0.5, 0.9])
    );
    expect(result.scalars).toBeDefined();
    expect(result.scalars!.length).toBe(3);
    expect(result.scalars![0]).toBeCloseTo(0.1, 5);
    expect(result.scalars![2]).toBeCloseTo(0.9, 5);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// MED-14 (production-bug worklist): when `positions.length !== totalPoints *
// ndim` after rounding (encoding-metadata corruption), the function used to
// log an error and continue with the wrong `ndim` — producing silently wrong
// 3D projections downstream. The fix throws an Error to fail fast.
// ════════════════════════════════════════════════════════════════════════════

describe('projectPointsTo3D — size-mismatch guard (MED-14)', () => {
  it('throws when positions.length is not an exact multiple of totalPoints * ndim', () => {
    // The "39 elements / 10 points → ndim=4" case the worklist calls out:
    // 39 / 10 rounds to 4, but 10 * 4 = 40 ≠ 39. Round-up should fail fast.
    const positions = new Float32Array(39); // garbage size
    expect(() =>
      projectPointsTo3D(
        wasm,
        positions,
        null,
        null,
        null,
        makeViewState({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 0],
          tolerance: [0, 0, 0, 0],
        }),
        [{ start: 0, end: 10 }] as PointRange[],
        makeCtx()
      )
    ).toThrow(/size mismatch/i);
  });

  it('does NOT throw when positions.length == totalPoints * ndim (happy path stays valid)', () => {
    // 12 elements / 3 points → ndim=4 exactly. Must not regress.
    const positions = new Float32Array(12);
    expect(() =>
      projectPointsTo3D(
        wasm,
        positions,
        null,
        null,
        null,
        makeViewState({
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 0],
          tolerance: [0, 0, 0, 0],
        }),
        [{ start: 0, end: 3 }] as PointRange[],
        makeCtx()
      )
    ).not.toThrow();
  });
});

// W4b: Points projection moved to the main thread (WASM-accelerated). copy-C
// now feeds the WASM kernels directly, so the WASM-boundary guards the former
// worker dispatcher applied live here. These cases migrated from
// tests/unit/workers/data-worker/validation.test.ts.
describe('projectPointsTo3D — WASM-boundary validation guards', () => {
  const ranges10: PointRange[] = [{ start: 0, end: 10 }];

  it('rejects displayDims with out-of-range entries', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30), // 10 points × 3
        null,
        null,
        null,
        makeViewState({ displayDims: [0, 1, 7], slicePosition: [0, 0, 0] }), // 7 ≥ ndim=3
        ranges10,
        makeCtx()
      )
    ).toThrow(/displayDims\[2\]=7 out of range/);
  });

  it('rejects radii shorter than numPoints', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        null,
        new Float32Array(3), // need 10
        null,
        makeViewState(),
        ranges10,
        makeCtx()
      )
    ).toThrow(/radii too short/);
  });

  it('rejects colors shorter than 3 × numPoints', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        new Uint8Array(15), // need 30
        null,
        null,
        makeViewState(),
        ranges10,
        makeCtx()
      )
    ).toThrow(/colors too short/);
  });

  it('rejects sharpness shorter than numPoints (symmetry with lines/gsplats)', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        null,
        null,
        new Float32Array(3), // need 10
        makeViewState(),
        ranges10,
        makeCtx()
      )
    ).toThrow(/sharpness too short/);
  });

  it('rejects effectiveRadiusConfig.spatialExtendDims shorter than ndim', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        null,
        new Float32Array(10),
        null,
        makeViewState({ tolerance: [0, 0, 0] }),
        ranges10,
        makeCtx({ effectiveRadiusConfig: { spatialExtendDims: [false, false], maxRadius: 1.0 } })
      )
    ).toThrow(/spatialExtendDims too short/);
  });

  it('rejects effectiveRadiusConfig.maxRadius non-finite', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        null,
        new Float32Array(10),
        null,
        makeViewState({ tolerance: [0, 0, 0] }),
        ranges10,
        makeCtx({
          effectiveRadiusConfig: {
            spatialExtendDims: [false, false, false],
            maxRadius: Number.POSITIVE_INFINITY,
          },
        })
      )
    ).toThrow(/maxRadius=Infinity must be a finite number/);
  });

  it('rejects effectiveRadiusConfig with tolerance shorter than ndim', () => {
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        null,
        new Float32Array(10),
        null,
        makeViewState({ tolerance: [0, 0] }), // len 2 < ndim 3
        ranges10,
        makeCtx({
          effectiveRadiusConfig: { spatialExtendDims: [false, false, false], maxRadius: 1.0 },
        })
      )
    ).toThrow(/viewState.tolerance too short/);
  });

  it('supports ndim > 16 via the uncapped TS reference backend (not rejected)', () => {
    // 1 point × 17 dims. The compiled WASM kernels cap at 16; the TS-reference
    // backend (used here, and selected by getPointsBackend for ndim>16) is
    // uncapped, so projection succeeds.
    const positions = new Float32Array(17);
    for (let d = 0; d < 17; d++) positions[d] = d;
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      null,
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: new Array(17).fill(0),
        tolerance: new Array(17).fill(0),
      }),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx()
    );
    expect(result.ndim).toBe(17);
    expect(Array.from(result.positions)).toEqual([0, 1, 2]);
  });
});

describe('projectPointsTo3D — RGBA color compaction (colorComponents=4, fallback path)', () => {
  // The zero-radius filter compacts colors strided by `colorComponents`
  // (3 = RGB, 4 = RGBA — alpha is per-point opacity). A hardcoded stride 3
  // would truncate the output and hand every survivor after a removed point
  // a misaligned tuple (the gsplat colorK lesson). Mirrors the
  // Uint8-radius filter tests above: a hidden-spatial-dim distance drives
  // the effective radius to 0 for the middle point.
  it('surviving points keep their OWN RGBA tuples when a middle point is filtered out', () => {
    // 3 points × 4 dims: points 0 and 2 sit ON the slice (dim3 = 0, kept);
    // point 1 is far in the hidden dim (dim3 = 100 ≫ R=1 → R_eff 0, filtered).
    const positions = new Float32Array([0, 0, 0, 0, 1, 1, 1, 100, 2, 2, 2, 0]);
    const colors = new Float32Array([
      0.1,
      0.2,
      0.3,
      0.9, // point 0 (kept)
      0.4,
      0.5,
      0.6,
      0.5, // point 1 (filtered)
      0.7,
      0.8,
      0.9,
      0.25, // point 2 (kept)
    ]);

    const result = projectPointsTo3D(
      wasm,
      positions,
      colors,
      new Float32Array([1, 1, 1]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0], // slice at dim3 = 0
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({
        effectiveRadiusConfig: {
          spatialExtendDims: [true, true, true, true],
          maxRadius: 1.0,
        },
      }),
      null, // no targetBuffers / accumulator → fallback (allocating) path
      undefined,
      4 // colorComponents: RGBA
    );

    expect(result.pointCount).toBe(2);
    expect(result.colorComponents).toBe(4);
    expect(result.colors!.length).toBe(2 * 4); // NOT 2 * 3 (a stride-3 compaction)
    // Survivor 0 keeps point 0's tuple; survivor 1 keeps point 2's tuple —
    // a stride-3 compaction would hand survivor 1 [0.6, 0.7, 0.8] instead.
    const expected = [0.1, 0.2, 0.3, 0.9, 0.7, 0.8, 0.9, 0.25];
    for (let i = 0; i < expected.length; i++) {
      expect(result.colors![i]).toBeCloseTo(expected[i], 6);
    }
  });
});
