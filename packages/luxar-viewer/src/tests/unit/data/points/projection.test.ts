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
// [P5] BOUNDARY: Uint8 radius decode to WORLD units before effective-radius use.
//
// Verified against the REAL source: the uint8 decode (projection.ts) only runs
// when `finalRadii instanceof Uint8Array`, which in turn only happens on the
// targetBuffers branch where BOTH the input radii AND the accumulator radii
// buffer are Uint8Array. The on-disk uint8 is a normalized [0,255] encoding of
// world-unit radii `(u8/255)·max_radius`, so it is decoded to WORLD units
// before the kernel (issue #740 revision) — radius and slice distance D must
// share units (D stays world-unit). We pin the decode via its observable
// CONSEQUENCE: with a hidden-dim distance large vs the *world* radius the
// point's effective radius (√(R²−D²)) clamps to 0 and the point is filtered
// out. Here max_radius=255, so u8=255 → world R=255; a distance > 255 clamps.
// ════════════════════════════════════════════════════════════════════════════

describe('Uint8 radius decode to world units before effective-radius use', () => {
  function makeErConfig(overrides: Partial<EffectiveRadiusConfig> = {}): EffectiveRadiusConfig {
    return {
      // dims 0..2 displayed (in-plane); dim 3 is a non-displayed SPATIAL dim
      // so shouldApplyEffectiveRadius() returns true and the Pythagorean path runs.
      spatialExtendDims: [true, true, true, true],
      maxRadius: 255,
      ...overrides,
    };
  }

  it('decodes Uint8 radius to world units so √(R²−D²) clamps to 0 and the far point is filtered out', () => {
    // 1 point at dim3 = 300 (far in the hidden spatial dim), Uint8 radius 255.
    // World radius = (255/255)·255 = 255; D = 300 → R_eff = √(255² − 300²) < 0
    // → 0 → filtered. (D must exceed the WORLD radius 255 to clamp — under the
    // old normalized-scale bug R was 1.0 and any D>1 would clamp.)
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
      new Float32Array([0, 0, 0, 300]), // 1 point × 4 dims; dim3 = 300
      null,
      new Uint8Array([255]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0], // slice at dim3 = 0 → distance 300
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 1 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: makeErConfig(), accumulator }),
      targetBuffers
    );

    // World R=255 ≪ D=300 → effective radius 0 → point filtered out.
    // (The all-filtered path returns createEmptyPointsData, whose metadata
    // does not carry usedEffectiveRadius — pointCount 0 is the observable.)
    expect(result.pointCount).toBe(0);
  });

  it('on-slice Uint8 radius (D=0) yields effective radius = world R (255), point kept', () => {
    // Same setup but the point sits ON the slice (dim3 = 0). D = 0 →
    // R_eff = √(R²) = R = 255 (the WORLD radius). The point survives.
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

  it('rejects colors whose length does not match numPoints × colorComponents (STRICT)', () => {
    // Strict equality since the volumetric double-check: a minimum
    // check (4N ≥ 3N) let an RGBA array with an undeclared
    // colorComponents silently mis-stride every point.
    expect(() =>
      projectPointsTo3D(
        wasm,
        new Float32Array(30),
        new Uint8Array(15), // need exactly 30
        null,
        null,
        makeViewState(),
        ranges10,
        makeCtx()
      )
    ).toThrow(/colors length 15 does not match count 10 × colorComponents 3/);
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

// ════════════════════════════════════════════════════════════════════════════
// Regression: issue #751 — in-place sharpness/scalars compaction gated on the
// SOURCE dtype. The shuffle only touches the accumulator TARGET buffer (already
// Float32-pinned for any non-uint8 attribute), so a Float16/Uint16 source
// matched NO `instanceof` branch and the `[writeIdx] = [readIdx]` move was
// skipped — leaving stale values at the survivor slots. The fix gates only on
// the target buffer existing.
// ════════════════════════════════════════════════════════════════════════════

describe('projectPointsTo3D — sharpness/scalars compaction with a non-Float32 source (issue #751)', () => {
  // 3 points × 4 dims: points 0 and 2 sit ON the slice (dim3 = 0, kept);
  // point 1 is far in the hidden dim (dim3 = 100 ≫ R=1 → R_eff 0, filtered).
  // Survivors [0, 2] compact to slots [0, 1]: slot 1 must receive point 2's
  // value (readIdx 2), NOT keep point 1's stale value.
  const positions = new Float32Array([0, 0, 0, 0, 1, 1, 1, 100, 2, 2, 2, 0]);
  const erConfig: EffectiveRadiusConfig = {
    spatialExtendDims: [true, true, true, true],
    maxRadius: 1.0,
  };
  const fillPositions = new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]);

  function makeTargetBuffers(accumulator: LoadedPointsDataAccumulator): ProjectionTargetBuffers {
    return {
      positions3D: accumulator.getPositionBuffer(),
      colors: accumulator.getColorBuffer(),
      radii: accumulator.getRadiiBuffer(),
      sharpness: accumulator.getSharpnessBuffer(),
      scalars: accumulator.getScalarBuffer(),
    };
  }

  it('compacts a NATIVE Uint16 sharpness/scalars buffer (dtype preserved, not zeroed)', () => {
    // Native Uint16 accumulator buffers pre-populated with distinct per-point
    // values. The in-place shuffle must move survivor 2 into slot 1 while
    // keeping the buffers Uint16 (the upload site normalizes ÷65535). Pre-fix
    // the source-dtype `instanceof` gate skipped the move → slot 1 kept 20 / 2.
    const accumulator = new LoadedPointsDataAccumulator(8, 4, 3);
    accumulator.fill(0, {
      positions: fillPositions,
      radii: new Float32Array([1, 1, 1]),
      sharpness: new Uint16Array([10, 20, 30]),
      scalars: new Uint16Array([1, 2, 3]),
    });
    const targetBuffers = makeTargetBuffers(accumulator);
    expect(accumulator.getSharpnessBuffer()).toBeInstanceOf(Uint16Array);
    expect(accumulator.getScalarBuffer()).toBeInstanceOf(Uint16Array);

    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      new Float32Array([1, 1, 1]),
      new Uint16Array([0, 0, 0]),
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig, accumulator }),
      targetBuffers,
      new Uint16Array([0, 0, 0])
    );

    expect(result.pointCount).toBe(2);
    // Native dtype preserved (RAW values, upload normalizes later).
    expect(result.sharpness).toBeInstanceOf(Uint16Array);
    expect(result.scalars).toBeInstanceOf(Uint16Array);
    // Survivor slot 1 must hold point 2's value (30 / 3), not point 1's (20 / 2).
    expect(Array.from(result.sharpness!)).toEqual([10, 30]);
    expect(Array.from(result.scalars!)).toEqual([1, 3]);
  });

  it('compacts a Float16Array source into the Float32 accumulator buffer', () => {
    // Skip on engines without a Float16Array global (older Node / JSDOM).
    const F16 = (globalThis as unknown as { Float16Array?: Float16ArrayConstructor }).Float16Array;
    if (typeof F16 === 'undefined') return;

    // Float16 sources pin the accumulator to Float32 (value-preserving).
    const accumulator = new LoadedPointsDataAccumulator(8, 4, 3);
    accumulator.fill(0, {
      positions: fillPositions,
      radii: new Float32Array([1, 1, 1]),
      sharpness: new F16([10, 20, 30]),
      scalars: new F16([1, 2, 3]),
    });
    const targetBuffers = makeTargetBuffers(accumulator);
    expect(accumulator.getSharpnessBuffer()).toBeInstanceOf(Float32Array);

    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      new Float32Array([1, 1, 1]),
      new F16([0, 0, 0]),
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig, accumulator }),
      targetBuffers,
      new F16([0, 0, 0])
    );

    expect(result.pointCount).toBe(2);
    expect(Array.from(result.sharpness!)).toEqual([10, 30]);
    expect(Array.from(result.scalars!)).toEqual([1, 3]);
  });

  it('fallback path (no accumulator) aligns a Float16 sharpness/scalars source (Finding 2)', () => {
    // Skip on engines without a Float16Array global (older Node / JSDOM).
    const F16 = (globalThis as unknown as { Float16Array?: Float16ArrayConstructor }).Float16Array;
    if (typeof F16 === 'undefined') return;

    // No targetBuffers → the allocating fallback. A Float16 source matched no
    // `filteredX` allocation branch, so the copy loop was skipped and the
    // ORIGINAL full-length array survived while pointCount=filteredCount →
    // misaligned attributes. The fix widens Float16 to Float32(filteredCount).
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      new Float32Array([1, 1, 1]),
      new F16([10, 20, 30]),
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig }),
      null, // no accumulator → fallback allocating path
      new F16([1, 2, 3])
    );

    expect(result.pointCount).toBe(2);
    // Length MUST match the filtered count (2), not the original 3, and the
    // survivor values must be points 0 and 2 (not the stale mid point).
    expect(result.sharpness!.length).toBe(2);
    expect(result.scalars!.length).toBe(2);
    expect(Array.from(result.sharpness!)).toEqual([10, 30]);
    expect(Array.from(result.scalars!)).toEqual([1, 3]);
  });

  it('fallback path (no accumulator) aligns a Float64 sharpness/scalars source (Finding 2, else-branch)', () => {
    // Covers the SAME else-branch as the Float16 test above, but with a
    // Float64Array source so it needs NO Float16 global and runs in CI on
    // every engine. Float64 is neither Float32/Uint8/Uint16, so it falls
    // into the else → widens value-preserving to Float32(filteredCount).
    // Pre-fix (no else) it left `filteredSharpness` undefined → copy loop
    // skipped → the ORIGINAL length-3 array survived → length assertion fails.
    const result = projectPointsTo3D(
      wasm,
      positions,
      null,
      new Float32Array([1, 1, 1]),
      // Float64Array is not in the param union; the cast is honest — the test
      // verifies the else-branch's "any other source → widen to Float32".
      new Float64Array([10, 20, 30]) as unknown as Float32Array,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig }),
      null, // no accumulator → fallback allocating path
      new Float64Array([1, 2, 3]) as unknown as Float32Array
    );

    expect(result.pointCount).toBe(2);
    // Length MUST match the filtered count (2), not the original 3, and the
    // survivor values must be points 0 and 2 (not the stale mid point).
    expect(result.sharpness!.length).toBe(2);
    expect(result.scalars!.length).toBe(2);
    expect(Array.from(result.sharpness!)).toEqual([10, 30]);
    expect(Array.from(result.scalars!)).toEqual([1, 3]);
    // The else-branch widened the "any other" source to Float32.
    expect(result.sharpness).toBeInstanceOf(Float32Array);
    expect(result.scalars).toBeInstanceOf(Float32Array);
  });
});

// ════════════════════════════════════════════════════════════════════════════
// Regression: issue #740 — effective radii on the UINT8-radii ACCUMULATOR path.
//
// When a points node stores radii as native Uint8Array, the projection used to
// compute the normalized/effective radii into a TEMP Float32Array (the
// write-back guards only fired for `targetBuffers.radii instanceof
// Float32Array`). Two consequences on the hot accumulator path:
//   (1) A partial zero-radius cull compacted positions/colors/sharpness/scalars
//       in the ACCUMULATOR buffers but radii only in the temp — and getData()
//       returns the raw UNCOMPACTED uint8 radii, so every survivor after the
//       first removed point rendered with another point's radius (misalignment).
//   (2) With NO filtering the slice-attenuated effective radii never reached
//       the output (the temp was discarded; getData returned raw uint8).
// The fix decodes uint8 radii to WORLD units for the kernel (so radius and
// slice distance D share units), then re-encodes the world-unit effective
// radii back into the uint8 accumulator buffer with the SAME /max_radius·255
// divisor the renderer inverts (so getData()/compaction operate on the SAME
// buffer), while the cull decision reads the float (world-unit) effective
// radii so the threshold semantics stay consistent.
// ════════════════════════════════════════════════════════════════════════════

describe('projectPointsTo3D — effective radii on the uint8 accumulator path (issue #740)', () => {
  const erConfig: EffectiveRadiusConfig = {
    // dims 0..2 displayed; dim 3 is a non-displayed SPATIAL dim so the
    // Pythagorean effective-radius path runs. maxRadius 255 is the WORLD-unit
    // max radius: uint8 u decodes to world radius (u/255)·255 = u.
    spatialExtendDims: [true, true, true, true],
    maxRadius: 255,
  };

  function makeTargetBuffers(accumulator: LoadedPointsDataAccumulator): ProjectionTargetBuffers {
    return {
      positions3D: accumulator.getPositionBuffer(),
      colors: accumulator.getColorBuffer(),
      radii: accumulator.getRadiiBuffer(),
      sharpness: accumulator.getSharpnessBuffer(),
    };
  }

  it('consequence (1): a PARTIAL cull returns each survivor its OWN uint8 radius (aligned)', () => {
    // 4 points × 4 dims. With maxRadius=255 each uint8 u decodes to world
    // radius u. Point 0 is far in the hidden dim (dim3 = 100 > its world R=40 →
    // R_eff 0, filtered); points 1..3 sit ON the slice (dim3 = 0), so on-slice
    // R_eff == world R → re-encodes back to the same uint8 value. Survivors
    // [1,2,3] compact to slots [0,1,2].
    //
    // This pins alignment: pre-fix, getData() returned the raw uncompacted
    // uint8 buffer → slots [0,1,2] held points [0,1,2]'s radii [40, 100, 150]
    // (the removed point's radius leaks in and every survivor is shifted). The
    // fix returns the compacted survivors' radii [100, 150, 200].
    const accumulator = new LoadedPointsDataAccumulator(8, 4, 4);
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
      radii: new Uint8Array([40, 100, 150, 200]),
    });
    const targetBuffers = makeTargetBuffers(accumulator);
    expect(accumulator.getRadiiBuffer()).toBeInstanceOf(Uint8Array);

    const result = projectPointsTo3D(
      wasm,
      // point 0 far (dim3=100); points 1..3 on slice (dim3=0)
      new Float32Array([0, 0, 0, 100, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0]),
      null,
      new Uint8Array([40, 100, 150, 200]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 4 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig, accumulator }),
      targetBuffers
    );

    expect(result.pointCount).toBe(3);
    expect(result.radii).toBeInstanceOf(Uint8Array);
    // Survivors are the SOURCE points 1,2,3 — radii aligned, not shifted.
    expect(Array.from(result.radii!)).toEqual([100, 150, 200]);
    // Positions confirm the same survivors in the same order.
    expect(Array.from(result.positions)).toEqual([1, 1, 1, 2, 2, 2, 3, 3, 3]);
  });

  it('consequence (2): with NO cull the returned uint8 radii carry the ATTENUATED (effective) values', () => {
    // 2 points × 4 dims, both survive (non-empty effective radius). Point 0
    // sits ON the slice (dim3 = 0 → R_eff == world R, unchanged). Point 1 is a
    // non-trivial hidden-dim distance away (dim3 = 120, comparable to its world
    // radius 200) so its effective radius is ATTENUATED below its raw value.
    // Pre-fix the uint8 path discarded the effective radii (getData returned
    // the raw uint8), so point 1 rendered at full size; the fix re-encodes the
    // attenuated WORLD-unit radius into the buffer.
    const maxRadius = 255;
    const rawRadius = 200;
    const D = 120;
    // Attenuation is computed in WORLD units: decode u8 → world R, apply
    // R_eff = sqrt(R_world² − D²), then re-encode round(R_eff / max · 255).
    const rWorld = (rawRadius / 255) * maxRadius; // = 200 here
    const expectedAttenuated = Math.round((Math.sqrt(rWorld * rWorld - D * D) / maxRadius) * 255);
    expect(expectedAttenuated).toBeLessThan(rawRadius); // attenuation actually happened
    expect(expectedAttenuated).toBeGreaterThan(0); // …but the point still survives

    const accumulator = new LoadedPointsDataAccumulator(8, 4, 2);
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1]),
      radii: new Uint8Array([rawRadius, rawRadius]),
    });
    const targetBuffers = makeTargetBuffers(accumulator);

    const result = projectPointsTo3D(
      wasm,
      // point 0 on slice (dim3=0); point 1 at dim3=120 (attenuated, kept)
      new Float32Array([0, 0, 0, 0, 1, 1, 1, D]),
      null,
      new Uint8Array([rawRadius, rawRadius]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 2 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig, accumulator }),
      targetBuffers
    );

    // No points removed — the effective radii themselves reach the output.
    expect(result.pointCount).toBe(2);
    expect(result.radii).toBeInstanceOf(Uint8Array);
    // Point 0 (D=0) keeps its raw radius; point 1 is attenuated below it.
    expect(Array.from(result.radii!)).toEqual([rawRadius, expectedAttenuated]);
  });

  it('combined: a PARTIAL cull where a SURVIVOR is also attenuated (attenuation × compaction)', () => {
    // 3 points × 4 dims, maxRadius=255 (world R = u8). Point 0 is far in the
    // hidden dim (dim3 = 200 > world R=100 → R_eff 0, filtered). Point 1 sits
    // ON the slice (dim3 = 0 → unchanged, u8=200). Point 2 is attenuated
    // (dim3 = 120, comparable to its world R=200). This pins the interaction
    // the two isolated tests above miss: the SURVIVING attenuated point (2)
    // must both (a) carry its attenuated radius AND (b) land in the correct
    // compacted slot. Survivors [1,2] → slots [0,1].
    const maxRadius = 255;
    const D2 = 120;
    const rWorld2 = 200; // (200/255)·255
    const expected2 = Math.round((Math.sqrt(rWorld2 * rWorld2 - D2 * D2) / maxRadius) * 255); // 160
    expect(expected2).toBe(160);

    const accumulator = new LoadedPointsDataAccumulator(8, 4, 3);
    accumulator.fill(0, {
      positions: new Float32Array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
      radii: new Uint8Array([100, 200, 200]),
    });
    const targetBuffers = makeTargetBuffers(accumulator);
    expect(accumulator.getRadiiBuffer()).toBeInstanceOf(Uint8Array);

    const result = projectPointsTo3D(
      wasm,
      // point 0 far (dim3=200, filtered); point 1 on slice (dim3=0);
      // point 2 attenuated (dim3=120, kept)
      new Float32Array([0, 0, 0, 200, 1, 1, 1, 0, 2, 2, 2, D2]),
      null,
      new Uint8Array([100, 200, 200]),
      null,
      makeViewState({
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      }),
      [{ start: 0, end: 3 }] as PointRange[],
      makeCtx({ effectiveRadiusConfig: erConfig, accumulator }),
      targetBuffers
    );

    expect(result.pointCount).toBe(2);
    expect(result.radii).toBeInstanceOf(Uint8Array);
    // Slot 0 = survivor 1 (on-slice, 200); slot 1 = survivor 2 (attenuated 160).
    expect(Array.from(result.radii!)).toEqual([200, expected2]);
    // Positions confirm the same survivors in the same order.
    expect(Array.from(result.positions)).toEqual([1, 1, 1, 2, 2, 2]);
  });
});
