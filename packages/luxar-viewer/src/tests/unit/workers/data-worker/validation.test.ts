/**
 * Validate that data-worker entry points reject malformed inputs at
 * the JS boundary instead of letting them reach WASM.
 *
 * Each test loads the worker module fresh via `vi.resetModules` with a
 * stubbed `wasm` module, calls `initialize()`, then exercises one
 * entry point with a deliberately bad payload.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface WorkerModule {
  workerAPI: Record<string, (...args: unknown[]) => Promise<unknown>>;
}

async function loadWorker(): Promise<WorkerModule> {
  vi.resetModules();
  const wasmStub = {
    extract_3d_positions: vi.fn(),
    calculate_effective_radii: vi.fn(),
    compute_nd_visibility_points: vi.fn(() => 0),
    compute_nd_visibility_lines: vi.fn(() => 0),
    compute_nd_visibility_gsplats: vi.fn(() => 0),
    clip_segments_batch: vi.fn(() => 0),
    interpolate_clipped_positions: vi.fn(),
    interpolate_colors_batch: vi.fn(),
    interpolate_scalars_batch: vi.fn(),
    calculate_segment_lengths: vi.fn(),
    mark_clipped_endpoints: vi.fn(),
    compact_by_mask: vi.fn(),
    calculate_bounds_3d: vi.fn(() => 0),
    count_visible: vi.fn(() => 0),
    radii_to_visibility_mask: vi.fn(() => 0),
    extract_visible_cholesky_3d: vi.fn(),
    compute_gsplats_attenuation: vi.fn(() => 0),
    compact_attenuated_amplitudes: vi.fn(() => 0),
    decode_quantized_u8: vi.fn(),
    decode_quantized_u16: vi.fn(),
    decode_log_scalar_u8: vi.fn(),
    decode_log_scalar_u16: vi.fn(),
    decode_lut_scalar_u8: vi.fn(),
    decode_lut_scalar_u16: vi.fn(),
    decode_lut_row_u8: vi.fn(),
    decode_lut_row_u16: vi.fn(),
    decode_broadcasted: vi.fn(),
    query_chunks_for_view: vi.fn(() => 0),
  };
  vi.doMock('../../../../wasm', () => ({
    initWasm: vi.fn(async () => wasmStub),
    // wasmStub stands in for the compiled backend, not the fallback.
    isWasmFallback: vi.fn(() => false),
    // getFallback supplies the uncapped TS backend used for ndim>16 routing;
    // we reuse the same stub so >16D calls land on the same spies.
    getFallback: vi.fn(() => wasmStub),
  }));
  vi.doMock('../../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn(), success: vi.fn() },
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));
  vi.doMock('comlink', () => ({
    expose: vi.fn(),
    transfer: vi.fn((obj) => obj),
  }));

  const mod = (await import('../../../../workers/data-worker')) as unknown as WorkerModule;
  await mod.workerAPI.initialize();
  return mod;
}

describe('data-worker validation — projection entry points', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  // Points projection moved to the main thread (WASM-accelerated) in W4b,
  // so it's no longer a worker entry point. Its validation guards (ndim,
  // positions/radii length, displayDims range, effectiveRadiusConfig) are
  // now exercised against `projectPointsTo3D` directly in
  // `tests/unit/data/points/projection.test.ts`.

  it('projectLinesTo3D rejects negative segmentCount', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(30),
        segments: new Uint32Array(0),
        widths: new Float32Array(0),
        colors: null,
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: -1,
      })
    ).rejects.toThrow(/numSegments=-1 must be a non-negative integer/);
  });

  it('projectLinesTo3D rejects segments array shorter than 2 × segmentCount', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(30),
        segments: new Uint32Array(3), // need 4 for 2 segments
        widths: new Float32Array(2),
        colors: null,
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: 2,
      })
    ).rejects.toThrow(/segments too short/);
  });

  it('projectGSplatsTo3D rejects choleskyFactors shorter than packed-lower-triangular size', async () => {
    const mod = await loadWorker();
    // 3D × 5 splats: each Cholesky packed = 3*4/2 = 6, total = 30.
    await expect(
      mod.workerAPI.projectGSplatsTo3D({
        positions: new Float32Array(15),
        choleskyFactors: new Float32Array(20), // need 30
        amplitudes: new Float32Array(5),
        colors: null,
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        splatCount: 5,
      })
    ).rejects.toThrow(/choleskyFactors too short/);
  });

  it('projectGSplatsTo3D rejects amplitudes shorter than splatCount', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectGSplatsTo3D({
        positions: new Float32Array(15),
        choleskyFactors: new Float32Array(30),
        amplitudes: new Float32Array(2), // need 5
        colors: null,
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        splatCount: 5,
      })
    ).rejects.toThrow(/amplitudes too short/);
  });
});

describe('data-worker validation — decode entry points', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('decodeQuantized rejects bounds where max ≤ min', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeQuantized({
        data: new Uint8Array([0, 128, 255]),
        bounds: [10, 5],
        dtype: 'uint8',
      })
    ).rejects.toThrow(/max \(5\) must be greater than min \(10\)/);
  });

  it('decodeQuantized rejects non-finite bounds', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeQuantized({
        data: new Uint8Array([0, 128, 255]),
        bounds: [0, Number.POSITIVE_INFINITY],
        dtype: 'uint8',
      })
    ).rejects.toThrow(/must be finite/);
  });

  it('decodeQuantized accepts the happy path', async () => {
    const mod = await loadWorker();
    const out = (await mod.workerAPI.decodeQuantized({
      data: new Uint8Array([0, 128, 255]),
      bounds: [0, 1],
      dtype: 'uint8',
    })) as Float32Array;
    expect(out).toBeInstanceOf(Float32Array);
    expect(out.length).toBe(3);
  });

  it('decodeLogScalar rejects non-finite maxLog', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLogScalar({
        data: new Uint8Array([0, 128, 255]),
        maxLog: Number.NaN,
        dtype: 'uint8',
      })
    ).rejects.toThrow(/maxLog=NaN must be a finite number/);
  });

  it('decodeLUT rejects k=0 (positive integer required)', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0, 1]),
        lut: [0.1, 0.2, 0.3],
        k: 0,
        lutMode: 'scalar',
      })
    ).rejects.toThrow(/k=0 must be a positive integer/);
  });

  it('decodeLUT rejects empty lut', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0, 1]),
        lut: [],
        k: 1,
        lutMode: 'scalar',
      })
    ).rejects.toThrow(/lut must be non-empty/);
  });

  it('decodeLUT rejects row-mode lut shorter than k', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0]),
        lut: [0.1, 0.2], // need ≥ 3 for k=3 row mode
        k: 3,
        lutMode: 'row',
      })
    ).rejects.toThrow(/lut too short/);
  });

  it('decodeLUT rejects unknown lutMode', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0]),
        lut: [0.1],
        k: 1,
        lutMode: 'cubic' as 'row',
      })
    ).rejects.toThrow(/lutMode='cubic' must be 'row' or 'scalar'/);
  });

  // out-of-range index validation. Pre-fix, an
  // index ≥ entry-count would reach Rust/WASM (`lut[indices[i]]`)
  // and panic. JS-side rejection at the worker boundary is the
  // intended contract.
  it('decodeLUT rejects scalar-mode index >= lut.length (Uint8 indices)', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0, 5]), // 5 ≥ lut.length=3
        lut: [0.1, 0.2, 0.3],
        k: 1,
        lutMode: 'scalar',
      })
    ).rejects.toThrow(/indices\[1\]=5 out of range for 3 LUT entries/);
  });

  it('decodeLUT rejects scalar-mode index >= lut.length (Uint16 indices)', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint16Array([0, 1000]),
        lut: [0.1, 0.2, 0.3],
        k: 1,
        lutMode: 'scalar',
      })
    ).rejects.toThrow(/indices\[1\]=1000 out of range/);
  });

  it('decodeLUT rejects row-mode index >= lut.length / k', async () => {
    const mod = await loadWorker();
    // 6 lut values / k=3 = 2 entries. Index 2 is out of range.
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0, 1, 2]),
        lut: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        k: 3,
        lutMode: 'row',
      })
    ).rejects.toThrow(/indices\[2\]=2 out of range for 2 LUT entries/);
  });

  it('decodeLUT rejects row-mode lut length not divisible by k', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeLUT({
        indices: new Uint8Array([0]),
        lut: [0.1, 0.2, 0.3, 0.4], // 4 not divisible by k=3
        k: 3,
        lutMode: 'row',
      })
    ).rejects.toThrow(/lut length 4 is not divisible by k=3/);
  });

  // MED-23: empty indices arrays must skip the O(n) out-of-range scan
  // and return a zero-length Float32Array without invoking WASM.
  it('decodeLUT short-circuits on empty scalar-mode indices', async () => {
    const mod = await loadWorker();
    const result = (await mod.workerAPI.decodeLUT({
      indices: new Uint8Array(0),
      lut: [0.1, 0.2, 0.3],
      k: 1,
      lutMode: 'scalar',
    })) as Float32Array;
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(0);
  });

  it('decodeLUT short-circuits on empty row-mode indices', async () => {
    const mod = await loadWorker();
    const result = (await mod.workerAPI.decodeLUT({
      indices: new Uint8Array(0),
      lut: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      k: 3,
      lutMode: 'row',
    })) as Float32Array;
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(0);
  });

  it('decodeBroadcasted rejects non-positive elementsPerPoint', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeBroadcasted({
        value: new Float32Array([1, 2, 3]),
        numPoints: 10,
        elementsPerPoint: 0,
      })
    ).rejects.toThrow(/elementsPerPoint=0 must be a positive integer/);
  });

  it('decodeBroadcasted rejects ambiguous value length (not 1 and not elementsPerPoint)', async () => {
    const mod = await loadWorker();
    // Strict contract: value must be a scalar (length 1) or exactly
    // elementsPerPoint. length 2 with elementsPerPoint 4 is rejected.
    await expect(
      mod.workerAPI.decodeBroadcasted({
        value: new Float32Array([1, 2]),
        numPoints: 10,
        elementsPerPoint: 4,
      })
    ).rejects.toThrow(/value\.length must be 1/);
  });

  it('decodeBroadcasted also rejects an over-long value (length > elementsPerPoint)', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.decodeBroadcasted({
        value: new Float32Array([1, 2, 3, 4]), // 4 > elementsPerPoint 3
        numPoints: 10,
        elementsPerPoint: 3,
      })
    ).rejects.toThrow(/value\.length must be 1/);
  });
});

describe('data-worker validation — segment/color/query gaps', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('projectLinesTo3D rejects segments referencing past the end of positions', async () => {
    const mod = await loadWorker();
    // ndim=3, segmentCount=2, segments references vertex index 7 but
    // positions only has 5 vertices worth of data (5*3=15).
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(15),
        segments: new Uint32Array([0, 1, 2, 7]),
        widths: new Float32Array(8),
        colors: null,
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: 2,
      })
    ).rejects.toThrow(/positions too short for max segment vertex 7/);
  });

  it('projectLinesTo3D rejects colors shorter than 3 × max-vertex-count', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(30),
        segments: new Uint32Array([0, 1, 2, 3]),
        widths: new Float32Array(4),
        colors: new Float32Array(6), // need 4 vertices × 3 = 12
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: 2,
      })
    ).rejects.toThrow(/colors too short for max segment vertex 3/);
  });

  // (Points colors-length guard moved to data/points/projection.test.ts — W4b.)

  it('projectGSplatsTo3D rejects colors shorter than 3 × splatCount', async () => {
    const mod = await loadWorker();
    // 5 splats × 3 dims; cholesky packed lower = 5 × 6 = 30
    await expect(
      mod.workerAPI.projectGSplatsTo3D({
        positions: new Float32Array(15),
        choleskyFactors: new Float32Array(30),
        amplitudes: new Float32Array(5),
        colors: new Float32Array(10), // need 5 × 3 = 15
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        splatCount: 5,
        discreteDims: [],
        discreteSteps: {},
        extendToAllDims: [],
        truncate: 3.0,
      })
    ).rejects.toThrow(/colors too short/);
  });

  it('projectGSplatsTo3D rejects sharpness shorter than splatCount (three-geometry symmetry)', async () => {
    // [workers OOS] GSplats accepts sharpness for API parity but never
    // reads it. Pre-fix, a caller bug producing a wrong-sized sharpness
    // array flowed through silently. Now it surfaces with the same
    // clear error the Points/Lines validation produces.
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectGSplatsTo3D({
        positions: new Float32Array(15),
        choleskyFactors: new Float32Array(30),
        amplitudes: new Float32Array(5),
        colors: null,
        sharpness: new Float32Array(3), // need 5 (splatCount)
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        splatCount: 5,
        discreteDims: [],
        discreteSteps: {},
        extendToAllDims: [],
        truncate: 3.0,
      })
    ).rejects.toThrow(/sharpness too short/);
  });

  it('querySpatialIndex rejects non-positive / non-integer ndim', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.querySpatialIndex({
        chunkBounds: new Float32Array(0),
        slicePosition: new Float32Array(20),
        tolerance: new Float32Array(20),
        numChunks: 0,
        ndim: 0,
      })
    ).rejects.toThrow(/must be a positive integer/);
  });

  it('querySpatialIndex accepts ndim>16 (chunk AABB query is dimension-agnostic)', async () => {
    const mod = await loadWorker();
    // query_chunks_for_view uses dynamic indexing (no fixed 16-dim cap), so >16D
    // is handled natively — it must not be rejected.
    await expect(
      mod.workerAPI.querySpatialIndex({
        chunkBounds: new Float32Array(0),
        slicePosition: new Float32Array(17),
        tolerance: new Float32Array(17),
        numChunks: 0,
        ndim: 17,
      })
    ).resolves.toBeDefined();
  });

  it('querySpatialIndex rejects chunkBounds shorter than numChunks × ndim × 2', async () => {
    const mod = await loadWorker();
    // 5 chunks × 3 dims × 2 = 30 bounds entries needed
    await expect(
      mod.workerAPI.querySpatialIndex({
        chunkBounds: new Float32Array(20),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        numChunks: 5,
        ndim: 3,
      })
    ).rejects.toThrow(/chunkBounds too short/);
  });

  it('querySpatialIndex rejects slicePosition shorter than ndim', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.querySpatialIndex({
        chunkBounds: new Float32Array(30),
        slicePosition: new Float32Array(2), // need 3
        tolerance: new Float32Array(3),
        numChunks: 5,
        ndim: 3,
      })
    ).rejects.toThrow(/slicePosition too short/);
  });

  // gaps the reviewer identified that weren't yet covered.

  it('querySpatialIndex rejects tolerance shorter than ndim', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.querySpatialIndex({
        chunkBounds: new Float32Array(30),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(2), // need 3
        numChunks: 5,
        ndim: 3,
      })
    ).rejects.toThrow(/tolerance too short/);
  });

  it('computeNDVisibilityLines rejects segment indices past end of vertices', async () => {
    const mod = await loadWorker();
    // ndim=3, 2 segments, segments[3]=8 references vertex 8; vertices
    // has only 5 vertices (15 floats). The line-segment validator must
    // catch this before WASM clip_segments_batch.
    await expect(
      mod.workerAPI.computeNDVisibilityLines({
        vertices: new Float32Array(15),
        segments: new Uint32Array([0, 1, 2, 8]),
        widths: new Float32Array(9),
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 2,
      })
    ).rejects.toThrow(/vertices too short|positions too short for max segment vertex 8/);
  });

  it('computeNDVisibilityLines rejects widths shorter than max-vertex+1', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.computeNDVisibilityLines({
        vertices: new Float32Array(30),
        segments: new Uint32Array([0, 1, 2, 3]),
        widths: new Float32Array(2), // need at least max-vertex+1 = 4
        slicePosition: new Float32Array(3),
        tolerance: new Float32Array(3),
        ndim: 3,
        numSegments: 2,
      })
    ).rejects.toThrow(/widths too short/);
  });

  it('projectLinesTo3D rejects sharpness shorter than max-vertex+1', async () => {
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(30),
        segments: new Uint32Array([0, 1, 2, 3]),
        widths: new Float32Array(4),
        colors: null,
        sharpness: new Float32Array(2), // need 4
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: 2,
      })
    ).rejects.toThrow(/sharpness too short/);
  });

  it('projectLinesTo3D rejects scalars shorter than max-vertex+1', async () => {
    // [workers OOS] Three-geometry symmetry: Points + GSplats validate
    // their own scalars; this pins Lines must too. Pre-fix, a short
    // scalars array reached `interpolate_scalars_batch` and panicked WASM
    // (or returned garbage in the TS fallback).
    const mod = await loadWorker();
    await expect(
      mod.workerAPI.projectLinesTo3D({
        positions: new Float32Array(30),
        segments: new Uint32Array([0, 1, 2, 3]),
        widths: new Float32Array(4),
        colors: null,
        sharpness: null,
        scalars: new Float32Array(2), // need 4 (max-vertex=3 → minVertices=4)
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0],
          tolerance: [0, 0, 0],
        },
        ndim: 3,
        segmentCount: 2,
      })
    ).rejects.toThrow(/scalars too short/);
  });

  // (Points effectiveRadiusConfig guards — spatialExtendDims length,
  // finite maxRadius, tolerance length — moved to
  // data/points/projection.test.ts in W4b, where they run against the
  // main-thread WASM-accelerated projectPointsTo3D.)
});
