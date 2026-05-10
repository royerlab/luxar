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
} from '../../../../data/points/projection';
import type { ViewState, PointRange } from '../../../../data/data-loader-types';
import type { PointsMetadata } from '../../../../types/points';
import type { PointsChunkIndex } from '../../../../data/points/chunk-index-loader';

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
    const data = createEmptyPointsData(
      makeCtx({ chunkIndex: makeChunkIndex(7) }),
      makeViewState()
    );
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
      positions,
      null,
      null,
      null,
      makeViewState({ displayDims: [1, 3, 4] }),
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
  // for tests, the no-accumulator opt-out, and the worker-error rescue
  // route in projectPointsTo3DUsingWorker.

  it('allocates a fresh Float32Array for positions3D (length = numPoints * 3)', () => {
    const positions = new Float32Array([1, 2, 3, 4, 5, 6]); // 2 points × 3 dims
    const result = projectPointsTo3D(
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

describe('A.3 — scalar length validation', () => {
  it('suppresses scalars when scalars.length !== point count', () => {
    const result = projectPointsTo3D(
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
    expect(result.scalars![0]).toBeCloseTo(0.1);
    expect(result.scalars![2]).toBeCloseTo(0.9);
  });
});
