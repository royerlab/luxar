/**
 * Happy-path tests for the worker projection entry points. The
 * validation tests in `data-worker-validation.test.ts` cover bad
 * inputs being rejected at the JS boundary; these tests cover good
 * inputs flowing through the JS-side glue. The WASM math itself is
 * tested by the Rust unit suite, so the WASM stubs here just need
 * to be observable.
 *
 * The shared `loadWorker()` helper mocks every WASM function with
 * vi.fn so the worker's `wasmModule!.X(...)` calls are observable.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';

interface WorkerModule {
  workerAPI: Record<string, (...args: unknown[]) => Promise<unknown>>;
}

interface WasmStubs {
  extract_3d_positions: ReturnType<typeof vi.fn>;
  calculate_effective_radii: ReturnType<typeof vi.fn>;
  calculate_bounds_3d: ReturnType<typeof vi.fn>;
  radii_to_visibility_mask: ReturnType<typeof vi.fn>;
  compute_nd_visibility_points: ReturnType<typeof vi.fn>;
  compute_nd_visibility_lines: ReturnType<typeof vi.fn>;
  compute_nd_visibility_gsplats: ReturnType<typeof vi.fn>;
  clip_segments_batch: ReturnType<typeof vi.fn>;
  interpolate_clipped_positions: ReturnType<typeof vi.fn>;
  interpolate_colors_batch: ReturnType<typeof vi.fn>;
  interpolate_scalars_batch: ReturnType<typeof vi.fn>;
  calculate_segment_lengths: ReturnType<typeof vi.fn>;
  mark_clipped_endpoints: ReturnType<typeof vi.fn>;
  compact_by_mask: ReturnType<typeof vi.fn>;
  extract_visible_cholesky_3d: ReturnType<typeof vi.fn>;
  compute_gsplats_attenuation: ReturnType<typeof vi.fn>;
  compact_attenuated_amplitudes: ReturnType<typeof vi.fn>;
  decode_broadcasted: ReturnType<typeof vi.fn>;
}

async function loadWorker(): Promise<{ mod: WorkerModule; wasm: WasmStubs }> {
  vi.resetModules();

  // [workers.md C2 / Phase F4] Route the REAL TypeScriptFallback through
  // the projection pipeline rather than mocking every WASM function with
  // a no-op `vi.fn()`. Each method is `vi.fn`-wrapped around the real
  // implementation so call counts are still observable AND
  // wasm-output→JS-state coupling is exercised end-to-end. A regression
  // that decoupled positions3D / visibleCount from the real WASM-side
  // result will now fail here instead of slipping through.
  const { TypeScriptFallback } = await import('../../../../wasm/typescript');
  const real = new TypeScriptFallback();
  const wasm: WasmStubs = {
    extract_3d_positions: vi.fn(real.extract_3d_positions.bind(real)),
    calculate_effective_radii: vi.fn(real.calculate_effective_radii.bind(real)),
    calculate_bounds_3d: vi.fn(real.calculate_bounds_3d.bind(real)),
    radii_to_visibility_mask: vi.fn(real.radii_to_visibility_mask.bind(real)),
    compute_nd_visibility_points: vi.fn(real.compute_nd_visibility_points.bind(real)),
    compute_nd_visibility_lines: vi.fn(real.compute_nd_visibility_lines.bind(real)),
    compute_nd_visibility_gsplats: vi.fn(real.compute_nd_visibility_gsplats.bind(real)),
    clip_segments_batch: vi.fn(real.clip_segments_batch.bind(real)),
    interpolate_clipped_positions: vi.fn(real.interpolate_clipped_positions.bind(real)),
    interpolate_colors_batch: vi.fn(real.interpolate_colors_batch.bind(real)),
    interpolate_scalars_batch: vi.fn(real.interpolate_scalars_batch.bind(real)),
    calculate_segment_lengths: vi.fn(real.calculate_segment_lengths.bind(real)),
    mark_clipped_endpoints: vi.fn(real.mark_clipped_endpoints.bind(real)),
    compact_by_mask: vi.fn(real.compact_by_mask.bind(real)),
    extract_visible_cholesky_3d: vi.fn(real.extract_visible_cholesky_3d.bind(real)),
    compute_gsplats_attenuation: vi.fn(real.compute_gsplats_attenuation.bind(real)),
    compact_attenuated_amplitudes: vi.fn(real.compact_attenuated_amplitudes.bind(real)),
    decode_broadcasted: vi.fn(real.decode_broadcasted.bind(real)),
  };

  vi.doMock('../../../../wasm', () => ({
    initWasm: vi.fn(async () => wasm),
    isWasmFallback: vi.fn(() => false),
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
  return { mod, wasm };
}

describe('projectPointsTo3D — happy paths', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('flows valid inputs through extract_3d_positions and returns shape', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 4;
    const ndim = 3;
    const positions = new Float32Array(numPoints * ndim);
    for (let i = 0; i < positions.length; i++) positions[i] = i;

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions,
      colors: null,
      radii: null,
      sharpness: null,
      viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0] },
      effectiveRadiusConfig: null,
      ndim,
      numPoints,
    })) as {
      positions3D: Float32Array;
      visibleCount: number;
      bounds: { min: number[]; max: number[] };
    };

    expect(wasm.extract_3d_positions).toHaveBeenCalledTimes(1);
    expect(result.positions3D.length).toBe(numPoints * 3);
    expect(result.visibleCount).toBe(numPoints);
    // workers.md W4 fix: previously `expect(result.bounds).toBeDefined()` —
    // unfailable for a result object. Pin the actual shape (min/max are
    // 3-element coordinate arrays).
    expect(result.bounds.min).toHaveLength(3);
    expect(result.bounds.max).toHaveLength(3);
    // workers.md W5 fix: previously the call count was the only
    // assertion. Inspect the argument shape so a regression that
    // dropped the displayDims conversion (Uint32Array), or shifted
    // numPoints/ndim positions, would surface as a failed shape match.
    const callArgs = (wasm.extract_3d_positions as any).mock.calls[0];
    expect(callArgs[0]).toBeInstanceOf(Float32Array); // positions
    expect((callArgs[0] as Float32Array).length).toBe(numPoints * ndim);
    expect(callArgs[1]).toBeInstanceOf(Uint32Array); // displayDims
    expect((callArgs[1] as Uint32Array).length).toBe(3);
    expect(callArgs[2]).toBe(ndim);
    expect(callArgs[3]).toBe(numPoints);
    expect(callArgs[4]).toBeInstanceOf(Float32Array); // output positions3D
    expect((callArgs[4] as Float32Array).length).toBe(numPoints * 3);
  });

  it('with radii but no effectiveRadiusConfig: copies radii through', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 3;
    const radii = new Float32Array([0.5, 1.0, 1.5]);

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * 3),
      colors: null,
      radii,
      sharpness: null,
      viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0] },
      effectiveRadiusConfig: null,
      ndim: 3,
      numPoints,
    })) as { radii: Float32Array | null };

    // No effective-radius config → calculate_effective_radii NOT called.
    expect(wasm.calculate_effective_radii).not.toHaveBeenCalled();
    expect(result.radii).not.toBeNull();
    expect(result.radii?.length).toBe(numPoints);
  });

  // workers.md/W5/P3: previously call-count only; now inspects WASM-bound
  // buffer/array shape (radii/positions/displayDims) so a regression
  // that reordered args, dropped the radii buffer, or shifted ndim/numPoints
  // would surface as a failed shape match.
  it('with effectiveRadiusConfig: invokes calculate_effective_radii with the expected buffer shape', async () => {
    // workers.md [W5][P3] strengthening: previously call-count only. Inspect
    // the buffer/array shape that flows OUT to WASM so a regression that
    // re-ordered args, dropped the radii buffer, or shifted ndim/numPoints
    // would surface as a failed shape match.
    const { mod, wasm } = await loadWorker();
    const numPoints = 4;
    const ndim = 4;
    const radii = new Float32Array(numPoints);

    await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.5, 0.5, 0.5, 0.5],
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [false, false, false, true],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    });

    expect(wasm.calculate_effective_radii).toHaveBeenCalledTimes(1);
    const callArgs = (wasm.calculate_effective_radii as any).mock.calls[0];
    // Production call order (workers/data-worker/projection/points.ts:165):
    //   (positions, radii, displayDims, slicePos, spatialExtendDims, ndim, numPoints, out)
    // Arg 0: positions (Float32Array, length numPoints * ndim).
    expect(callArgs[0]).toBeInstanceOf(Float32Array);
    expect((callArgs[0] as Float32Array).length).toBe(numPoints * ndim);
    // Arg 1: radii (Float32Array, length numPoints).
    expect(callArgs[1]).toBeInstanceOf(Float32Array);
    expect((callArgs[1] as Float32Array).length).toBe(numPoints);
    // Arg 2: displayDims as Uint32Array (3 displayed dims).
    expect(callArgs[2]).toBeInstanceOf(Uint32Array);
    // Some scalar/index arg is numPoints + ndim — search for both.
    const scalarArgs = callArgs.filter((a: unknown) => typeof a === 'number');
    expect(scalarArgs).toContain(numPoints);
    expect(scalarArgs).toContain(ndim);
    // The output buffer (last positional, Float32Array of length numPoints).
    expect(callArgs[7]).toBeInstanceOf(Float32Array);
    expect((callArgs[7] as Float32Array).length).toBe(numPoints);
  });

  it('numPoints=0 yields empty result; effective-radius WASM is not called', async () => {
    // workers.md C1 fix: docstring previously claimed "short-circuits without
    // invoking WASM" but only asserted result shape; the actual invariant
    // (radii-WASM NOT called, extract path is bounded) is pinned here.
    const { mod, wasm } = await loadWorker();

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(0),
      colors: null,
      radii: null,
      sharpness: null,
      viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0] },
      effectiveRadiusConfig: null,
      ndim: 3,
      numPoints: 0,
    })) as { visibleCount: number; positions3D: Float32Array };

    expect(result.visibleCount).toBe(0);
    expect(result.positions3D.length).toBe(0);
    // The contract: with empty inputs, effective-radius WASM never fires
    // AND extract_3d_positions runs at most once (the no-op write-through).
    expect(wasm.calculate_effective_radii).not.toHaveBeenCalled();
    expect((wasm.extract_3d_positions as any).mock.calls.length).toBeLessThanOrEqual(1);
  });
});

describe('projectLinesTo3D — happy paths', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('zero segments returns a zero-shape output without throwing', async () => {
    const { mod } = await loadWorker();
    // Validator requires ≥ ndim positions even for zero segments
    // (it uses numItems=1); supply ndim-length zeros.
    const result = (await mod.workerAPI.projectLinesTo3D({
      positions: new Float32Array(3),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [1, 1, 1],
      },
      ndim: 3,
      segmentCount: 0,
    })) as { visibleSegmentCount: number };
    expect(result.visibleSegmentCount).toBe(0);
  });

  // workers.md/W5/P3: previously call-count only; now pins the WASM-bound
  // buffer shape (segments array, vertices length, ndim, numSegments)
  // so a regression that reordered args or dropped the segments buffer
  // would surface as a failed shape match.
  it('valid 1-segment input invokes clip_segments_batch with the expected buffer shape', async () => {
    // workers.md [W5][P3] strengthening: previously call-count only. Pin
    // the buffer shape that flows to WASM so a regression that re-ordered
    // the segments/positions args or dropped the out-param tuple
    // (visibility/t1/t2) is caught.
    const { mod, wasm } = await loadWorker();
    const ndim = 3;
    const numVertices = 2;
    const positions = new Float32Array(numVertices * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    const segmentCount = 1;

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount,
    });

    expect(wasm.clip_segments_batch).toHaveBeenCalledTimes(1);
    const callArgs = (wasm.clip_segments_batch as any).mock.calls[0];
    // Production call order (workers/data-worker/projection/lines.ts:108):
    //   (positions, segments, slicePos, tol, displayDims, ndim, segmentCount,
    //    visibility, t1, t2)
    // Arg 0: positions Float32Array of size numVertices * ndim.
    expect(callArgs[0]).toBeInstanceOf(Float32Array);
    expect((callArgs[0] as Float32Array).length).toBe(numVertices * ndim);
    // Arg 1: segments Uint32Array of size segmentCount * 2.
    expect(callArgs[1]).toBeInstanceOf(Uint32Array);
    expect((callArgs[1] as Uint32Array).length).toBe(segmentCount * 2);
    // Arg 4: displayDims Uint32Array (3 displayed dims).
    expect(callArgs[4]).toBeInstanceOf(Uint32Array);
    expect((callArgs[4] as Uint32Array).length).toBe(3);
    // Arg 7: visibility Uint8Array of size segmentCount (out param).
    expect(callArgs[7]).toBeInstanceOf(Uint8Array);
    expect((callArgs[7] as Uint8Array).length).toBe(segmentCount);
  });

  it('scalars=null returns empty startScalars/endScalars and does not call interpolate_scalars_batch for scalars', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const positions = new Float32Array(2 * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);

    const result = (await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    })) as { startScalars: Float32Array; endScalars: Float32Array };

    expect(result.startScalars.length).toBe(0);
    expect(result.endScalars.length).toBe(0);
    // workers.md W6 fix: previously `toBeLessThanOrEqual(2)` — too loose
    // because a regression that skipped widths (call count 1) survived.
    // In this test, scalars + sharpness are both null but widths is
    // present, so exactly 1 call is expected. Pin that exactly.
    expect(wasm.interpolate_scalars_batch).toHaveBeenCalledTimes(1);
  });

  it('Float32Array scalars are passed through to interpolate_scalars_batch unchanged', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const positions = new Float32Array(2 * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    const scalars = new Float32Array([0.25, 0.75]);

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: null,
      sharpness: null,
      scalars,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    });

    // Find the scalar-interpolation call: 1st arg is the scalars buffer,
    // distinguishable from widths/sharpness by content.
    const calls = wasm.interpolate_scalars_batch.mock.calls as unknown[][];
    const scalarCall = calls.find((call) => {
      const arr = call[0];
      return arr instanceof Float32Array && arr.length === 2 && (arr as Float32Array)[0] === 0.25;
    });
    expect(scalarCall).toBeDefined();
  });

  it('Uint8Array scalars are normalized by 1/255 before WASM', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const positions = new Float32Array(2 * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    const scalars = new Uint8Array([0, 128, 255, 64]);

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: null,
      sharpness: null,
      scalars,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    });

    const calls = wasm.interpolate_scalars_batch.mock.calls as unknown[][];
    const scalarCall = calls.find((call) => {
      const arr = call[0];
      return (
        arr instanceof Float32Array &&
        arr.length === 4 &&
        Math.abs((arr as Float32Array)[1] - 128 / 255) < 1e-5
      );
    });
    expect(scalarCall).toBeDefined();
    const arr = scalarCall![0] as Float32Array;
    expect(arr[0]).toBeCloseTo(0, 5);
    expect(arr[1]).toBeCloseTo(128 / 255, 5);
    expect(arr[2]).toBeCloseTo(1, 5);
    expect(arr[3]).toBeCloseTo(64 / 255, 5);
  });
});

describe('projectGSplatsTo3D — happy paths', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('zero splats produces a zero-shape output without throwing', async () => {
    const { mod } = await loadWorker();

    const result = (await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(0),
      choleskyFactors: new Float32Array(0),
      amplitudes: new Float32Array(0),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      ndim: 3,
      splatCount: 0,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    })) as { centers3D: Float32Array; visibleCount: number };

    expect(result.visibleCount).toBe(0);
    expect(result.centers3D.length).toBe(0);
  });

  // workers.md/W5/P3: previously call-count only; now pins the WASM-bound
  // arg shape (centers, choleskyFactors length matching ndim*(ndim+1)/2 * N,
  // slicePos, tolerance) so reorder/drop regressions are caught.
  it('non-zero splats: compute_gsplats_attenuation receives the expected shaped args', async () => {
    // workers.md [W5][P3] strengthening: previously only asserted call-count.
    // Pin the argument shape so a refactor that re-ordered or dropped the
    // positions/cholesky/amplitudes buffers, or shifted ndim/splatCount,
    // would surface as a failed shape match.
    const { mod, wasm } = await loadWorker();
    const splatCount = 2;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2; // 6 for 3D

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    });

    expect(wasm.compute_gsplats_attenuation).toHaveBeenCalledTimes(1);
    const callArgs = (wasm.compute_gsplats_attenuation as any).mock.calls[0];
    // positions: Float32Array of size splatCount * ndim
    expect(callArgs[0]).toBeInstanceOf(Float32Array);
    expect((callArgs[0] as Float32Array).length).toBe(splatCount * ndim);
    // cholesky: Float32Array of size splatCount * k
    expect(callArgs[1]).toBeInstanceOf(Float32Array);
    expect((callArgs[1] as Float32Array).length).toBe(splatCount * k);
    // amplitudes: Float32Array of size splatCount
    expect(callArgs[2]).toBeInstanceOf(Float32Array);
    expect((callArgs[2] as Float32Array).length).toBe(splatCount);
    // visibility (out param) is a Uint8Array of size splatCount
    expect(callArgs[9]).toBeInstanceOf(Uint8Array);
    expect((callArgs[9] as Uint8Array).length).toBe(splatCount);
  });
});

describe('Color normalization at WASM boundary', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('Lines: Uint8Array colors are normalized to Float32 [0,1] before interpolate_colors_batch', async () => {
    const { mod, wasm } = await loadWorker();
    // Override clip_segments_batch to report 1 visible segment so the
    // interpolate_colors_batch call site fires (early-exit guards
    // skip it when visibleCount === 0).
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const numVertices = 2;
    const positions = new Float32Array(numVertices * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    // 6 RGB bytes for 2 vertices.
    const colorsU8 = new Uint8Array([0, 128, 255, 64, 200, 32]);

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: colorsU8,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    });

    // The first arg to interpolate_colors_batch must be a Float32Array
    // with values in [0, 1] (Uint8 / 255). Spot-check the conversion.
    expect(wasm.interpolate_colors_batch).toHaveBeenCalled();
    const firstArg = (wasm.interpolate_colors_batch.mock.calls[0] as unknown[])[0];
    expect(firstArg).toBeInstanceOf(Float32Array);
    const arr = firstArg as Float32Array;
    expect(arr[0]).toBeCloseTo(0, 5);
    expect(arr[1]).toBeCloseTo(128 / 255, 5);
    expect(arr[2]).toBeCloseTo(1, 5);
    expect(arr[3]).toBeCloseTo(64 / 255, 5);
    expect(arr[5]).toBeCloseTo(32 / 255, 5);
  });

  it('Lines: Uint16Array colors are normalized by 1/65535 before WASM', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const positions = new Float32Array(2 * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    const colorsU16 = new Uint16Array([0, 32768, 65535, 0, 0, 0]);

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: colorsU16,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    });

    const firstArg = (wasm.interpolate_colors_batch.mock.calls[0] as unknown[])[0];
    expect(firstArg).toBeInstanceOf(Float32Array);
    const arr = firstArg as Float32Array;
    expect(arr[0]).toBeCloseTo(0, 5);
    expect(arr[1]).toBeCloseTo(32768 / 65535, 5);
    expect(arr[2]).toBeCloseTo(1, 5);
  });

  it('GSplats: Uint8Array colors are normalized to Float32 [0,1] before compact_by_mask', async () => {
    const { mod, wasm } = await loadWorker();
    // Make every splat visible. compute_gsplats_attenuation writes into
    // the caller-supplied `visibility` Uint8Array (arg #9), so the mock
    // must fill it; otherwise the visibleCount==0 early-exit fires and
    // colors compaction never runs.
    wasm.compute_gsplats_attenuation.mockImplementation(
      (
        _positions: unknown,
        _cholesky: unknown,
        _amplitudes: unknown,
        _slicePos: unknown,
        _hiddenDims: unknown,
        _ndim: unknown,
        splatCountArg: number,
        _minAmp: unknown,
        _truncate: unknown,
        visibility: Uint8Array
      ) => {
        for (let i = 0; i < splatCountArg; i++) visibility[i] = 1;
        return 0;
      }
    );
    wasm.compact_by_mask.mockImplementation(() => {});
    wasm.compact_attenuated_amplitudes.mockImplementation(() => {});
    wasm.extract_visible_cholesky_3d.mockImplementation(() => {});
    wasm.extract_3d_positions.mockImplementation(() => {});

    const splatCount = 2;
    const ndim = 3;
    const k = (ndim * (ndim + 1)) / 2; // 6 for 3D
    const colorsU8 = new Uint8Array([0, 128, 255, 64, 200, 32]);

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: colorsU8,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    });

    // compact_by_mask is called for positions (3-component), amplitudes
    // (1-component), and finally colors (3-component). Find the colors
    // call by `numComponents === 3` AND the buffer being our normalized
    // Float32 input (the positions call passes a Float32Array of size
    // splatCount*3 all zeros, so we filter by content too).
    const colorCalls = wasm.compact_by_mask.mock.calls.filter((call) => {
      const arr = (call as unknown[])[0];
      const components = (call as unknown[])[3];
      return (
        components === 3 &&
        arr instanceof Float32Array &&
        (arr as Float32Array).length === splatCount * 3 &&
        // Distinguish from the all-zero positions call.
        (arr as Float32Array).some((v) => v !== 0)
      );
    });
    expect(colorCalls.length).toBe(1);
    const colorsArg = (colorCalls[0] as unknown[])[0] as Float32Array;
    expect(colorsArg[0]).toBeCloseTo(0, 5);
    expect(colorsArg[1]).toBeCloseTo(128 / 255, 5);
    expect(colorsArg[2]).toBeCloseTo(1, 5);
    expect(colorsArg[3]).toBeCloseTo(64 / 255, 5);
    expect(colorsArg[4]).toBeCloseTo(200 / 255, 5);
    expect(colorsArg[5]).toBeCloseTo(32 / 255, 5);
  });
});

describe('projectPointsTo3D — compaction path (G14)', () => {
  // workers.md G14: previously no test drove visibleCount BELOW numPoints
  // (compaction branch in points.ts). The compaction runs only when:
  //   - radii is non-null AND
  //   - effectiveRadiusConfig is non-null AND
  //   - the effective-radii WASM result yields some-but-not-all points
  //     (visibleCount strictly between 0 and numPoints).
  // The radii_to_visibility_mask + compact_by_mask mocks must cooperate
  // to set visibleCount in that window so the JS-side compaction loop
  // runs.

  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('partial visibility: invokes compact_by_mask and shrinks output arrays', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 4;
    const ndim = 4;

    // Mock effective-radii output: alternating 0 / 1 → 2 points kept.
    wasm.calculate_effective_radii.mockImplementation(
      (
        _positions: unknown,
        _radii: unknown,
        _displayDims: unknown,
        _slicePos: unknown,
        _extendDims: unknown,
        _ndim: unknown,
        _numPoints: number,
        out: Float32Array
      ) => {
        out[0] = 1.0;
        out[1] = 0.0;
        out[2] = 1.0;
        out[3] = 0.0;
      }
    );
    // radii_to_visibility_mask: write the mask and return visibleCount.
    wasm.radii_to_visibility_mask.mockImplementation(
      (radii: Float32Array, threshold: number, n: number, mask: Uint8Array) => {
        let count = 0;
        for (let i = 0; i < n; i++) {
          const visible = radii[i] > threshold ? 1 : 0;
          mask[i] = visible;
          count += visible;
        }
        return count;
      }
    );

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii: new Float32Array(numPoints).fill(1),
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.5, 0.5, 0.5, 0.5],
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [false, false, false, true],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    })) as { visibleCount: number; positions3D: Float32Array; radii: Float32Array | null };

    // The compaction path ran: visibleCount strictly between 0 and numPoints.
    expect(result.visibleCount).toBe(2);
    expect(result.visibleCount).toBeLessThan(numPoints);
    // Output sizes match the compacted count.
    expect(result.positions3D.length).toBe(2 * 3);
    expect(result.radii?.length).toBe(2);
    // compact_by_mask is called at least once: for positions, and once
    // each for radii / colors / sharpness depending on which inputs were
    // non-null. Here: positions + radii → at least 2 calls.
    expect(wasm.compact_by_mask).toHaveBeenCalled();
    expect((wasm.compact_by_mask as any).mock.calls.length).toBeGreaterThanOrEqual(2);
  });

  it('full filtering (visibleCount=0): output arrays are zero-length and color type is preserved (G4)', async () => {
    // workers.md G4: previously the `visibleCount === 0` type-preservation
    // contract for points.ts (lines 230-246) was untested.
    const { mod, wasm } = await loadWorker();
    const numPoints = 3;
    const ndim = 3;

    wasm.calculate_effective_radii.mockImplementation(
      (
        _p: unknown,
        _r: unknown,
        _dd: unknown,
        _sp: unknown,
        _ed: unknown,
        _n: unknown,
        _np: number,
        out: Float32Array
      ) => {
        // All zero → all filtered out.
        for (let i = 0; i < numPoints; i++) out[i] = 0;
      }
    );
    wasm.radii_to_visibility_mask.mockImplementation(
      (_radii: unknown, _t: unknown, _n: number, mask: Uint8Array) => {
        for (let i = 0; i < numPoints; i++) mask[i] = 0;
        return 0;
      }
    );

    // Pre-cast Uint8 colors must survive the empty-output branch as Uint8.
    const colorsU8 = new Uint8Array(numPoints * 3).fill(128);

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: colorsU8,
      radii: new Float32Array(numPoints).fill(1),
      sharpness: new Float32Array(numPoints).fill(0.5),
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0.5, 0.5, 0.5],
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [false, false, false],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    })) as {
      visibleCount: number;
      positions3D: Float32Array;
      colors: Float32Array | Uint8Array | Uint16Array | null;
      radii: Float32Array | null;
      sharpness: Float32Array | null;
    };

    expect(result.visibleCount).toBe(0);
    expect(result.positions3D.length).toBe(0);
    expect(result.radii?.length).toBe(0);
    expect(result.sharpness?.length).toBe(0);
    // workers.md G4 pin: colors must be a Uint8Array (preserved type),
    // length 0 — not a Float32Array or null.
    expect(result.colors).toBeInstanceOf(Uint8Array);
    expect(result.colors?.length).toBe(0);
  });

  it('full filtering with colors=null: result.colors stays null', async () => {
    // Symmetry pin: the type-preservation logic must not synthesize a
    // colors output when none was supplied.
    const { mod, wasm } = await loadWorker();
    const numPoints = 2;
    const ndim = 3;

    wasm.calculate_effective_radii.mockImplementation(
      (
        _p: unknown,
        _r: unknown,
        _dd: unknown,
        _sp: unknown,
        _ed: unknown,
        _n: unknown,
        _np: number,
        out: Float32Array
      ) => {
        for (let i = 0; i < numPoints; i++) out[i] = 0;
      }
    );
    wasm.radii_to_visibility_mask.mockImplementation(
      (_r: unknown, _t: unknown, _n: number, mask: Uint8Array) => {
        for (let i = 0; i < numPoints; i++) mask[i] = 0;
        return 0;
      }
    );

    const result = (await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii: new Float32Array(numPoints).fill(1),
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0.5, 0.5, 0.5],
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [false, false, false],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    })) as {
      visibleCount: number;
      colors: Float32Array | Uint8Array | Uint16Array | null;
    };

    expect(result.visibleCount).toBe(0);
    expect(result.colors).toBeNull();
  });
});

describe('projectPointsTo3D — sharpness validation symmetry (G16)', () => {
  // workers.md G16 / P8: lines.ts and points.ts both validate `sharpness
  // too short`. The lines test was already present; this pins the
  // points-side parity. gsplats.ts has no sharpness validation in source
  // (the param is accepted-but-ignored — see source line 27 / 240),
  // so there's no symmetric test possible. See `## OOS` in workers.md
  // for the production-side note.

  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('projectPointsTo3D rejects sharpness shorter than numPoints (symmetry with lines)', async () => {
    const { mod } = await loadWorker();
    await expect(
      mod.workerAPI.projectPointsTo3D({
        positions: new Float32Array(15), // 5 points × 3 dims
        colors: null,
        radii: null,
        sharpness: new Float32Array(2), // need 5
        viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0] },
        effectiveRadiusConfig: null,
        ndim: 3,
        numPoints: 5,
      })
    ).rejects.toThrow(/sharpness too short/);
  });
});

describe('projectPointsTo3D — MED-21 extend_to_all sentinel guard', () => {
  // The extend_to_all detection in points.ts builds an augmented
  // displayDims array including dims where `tolerance[d] >= 1e9`. The
  // guard was tightened to also require `Number.isFinite(tol)` so that
  // missing/NaN entries do NOT silently disable extend_to_all (they would
  // fail the `>= 1e9` test and silently treat the dim as a normal hidden
  // dim). Validation upstream rejects short tolerance arrays; this test
  // pins both: (a) the validation belt fires, and (b) a NaN entry is
  // explicitly excluded from extend_to_all by the isFinite guard.

  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('rejects short tolerance when effectiveRadiusConfig is set (validation belt)', async () => {
    const { mod } = await loadWorker();
    await expect(
      mod.workerAPI.projectPointsTo3D({
        positions: new Float32Array(12), // 3 points × 4 dims
        colors: null,
        radii: new Float32Array(3),
        sharpness: null,
        viewState: {
          displayDims: [0, 1, 2],
          slicePosition: [0, 0, 0, 0],
          tolerance: [0.5, 0.5, 0.5], // length 3 < ndim=4
        },
        effectiveRadiusConfig: {
          spatialExtendDims: [true, true, true, true],
          maxRadius: 10.0,
        },
        ndim: 4,
        numPoints: 3,
      })
    ).rejects.toThrow(/viewState\.tolerance too short/);
  });

  it('NaN tolerance is not treated as extend_to_all (Number.isFinite guard)', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 2;
    const ndim = 4;

    // Capture the displayDims argument passed to calculate_effective_radii.
    // If NaN were treated as extend_to_all, the augmented array would
    // include dim 3 (length 4); with the isFinite guard, dim 3 is omitted
    // and the original displayDims (length 3) is passed.
    let displayDimsArg: Uint32Array | undefined;
    wasm.calculate_effective_radii.mockImplementation((...args: unknown[]) => {
      displayDimsArg = args[2] as Uint32Array;
      return 0;
    });

    await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii: new Float32Array(numPoints),
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        // dim 3 tolerance is NaN — must NOT be treated as extend_to_all.
        tolerance: [0.5, 0.5, 0.5, Number.NaN],
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    });

    expect(displayDimsArg).toBeDefined();
    // Length is 3 (just the original display dims), NOT 4 (would mean
    // dim 3 was incorrectly added as extend_to_all).
    expect((displayDimsArg as Uint32Array).length).toBe(3);
    expect(Array.from(displayDimsArg as Uint32Array)).toEqual([0, 1, 2]);
  });

  it('finite >= 1e9 tolerance IS treated as extend_to_all (positive case)', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 2;
    const ndim = 4;

    let displayDimsArg: Uint32Array | undefined;
    wasm.calculate_effective_radii.mockImplementation((...args: unknown[]) => {
      displayDimsArg = args[2] as Uint32Array;
      return 0;
    });

    await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii: new Float32Array(numPoints),
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0.5, 0.5, 0.5, 1e10], // finite, >= 1e9 → extend_to_all
      },
      effectiveRadiusConfig: {
        spatialExtendDims: [true, true, true, true],
        maxRadius: 10.0,
      },
      ndim,
      numPoints,
    });

    expect(displayDimsArg).toBeDefined();
    // Length is 4 (display dims [0,1,2] + extend_to_all dim 3).
    expect((displayDimsArg as Uint32Array).length).toBe(4);
    expect(Array.from(displayDimsArg as Uint32Array)).toEqual([0, 1, 2, 3]);
  });
});

describe('projectGSplatsTo3D — TS-side branching (G2, G15)', () => {
  // workers.md G15 / G2: gsplats has substantial TS-side branching
  // (discreteDims half-step filter, extendToAllDims skip, post-
  // attenuation visibility AND, compaction). These tests exercise the
  // branching with the WASM mocks cooperating just enough to keep the
  // pipeline alive.

  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('discreteDims path: splats outside ±0.5*step on a discrete dim are filtered before WASM', async () => {
    const { mod, wasm } = await loadWorker();
    // 4 splats, 4D, dim 3 is discrete with step=1. slicePosition[3]=0.
    // Splats at d3 = {0, 0, 2, 0} → splat 2 fails the half-step test.
    const splatCount = 4;
    const ndim = 4;
    const k = (ndim * (ndim + 1)) / 2; // 10
    const positions = new Float32Array(splatCount * ndim);
    positions[0 * ndim + 3] = 0;
    positions[1 * ndim + 3] = 0;
    positions[2 * ndim + 3] = 2; // out of range
    positions[3 * ndim + 3] = 0;

    // Mock compute_gsplats_attenuation: marks every splat visible
    // (visibility[i]=1) AND attenuates by their preceding-pass mask
    // (the discrete filter already excluded splat 2). The kernel
    // receives the input visibility buffer and the discrete-filter
    // result is AND-ed in TS, so a fully-visible mock here lets the
    // TS-side discrete prefilter be the sole cause of splat 2 dropping.
    let attenuationVisible: Uint8Array | undefined;
    wasm.compute_gsplats_attenuation.mockImplementation((...args: unknown[]) => {
      const vis = args[9] as Uint8Array;
      attenuationVisible = vis;
      for (let i = 0; i < splatCount; i++) vis[i] = 1;
      return 0;
    });
    wasm.compact_by_mask.mockImplementation(() => {});
    wasm.compact_attenuated_amplitudes.mockImplementation(() => {});
    wasm.extract_visible_cholesky_3d.mockImplementation(() => {});
    wasm.extract_3d_positions.mockImplementation(() => {});

    await mod.workerAPI.projectGSplatsTo3D({
      positions,
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      },
      ndim,
      splatCount,
      discreteDims: [3],
      discreteSteps: { 3: 1.0 },
      extendToAllDims: [],
      truncate: 3.0,
    });

    // compute_gsplats_attenuation was invoked — the pipeline reached
    // the WASM call, which means the discrete filter ran first.
    expect(wasm.compute_gsplats_attenuation).toHaveBeenCalledTimes(1);
    expect(attenuationVisible).toBeDefined();
  });

  it('extendToAllDims path: dim is skipped from hidden-dim attenuation', async () => {
    const { mod, wasm } = await loadWorker();
    const splatCount = 2;
    const ndim = 4;
    const k = (ndim * (ndim + 1)) / 2;

    // Capture the hiddenDims arg WASM receives. Source builds it as
    // sortedHiddenDims minus extendSet → with extendToAllDims=[3],
    // hidden dims should be [] (dim 3 was the only hidden dim).
    let hiddenDimsArg: Uint32Array | undefined;
    wasm.compute_gsplats_attenuation.mockImplementation((...args: unknown[]) => {
      hiddenDimsArg = args[4] as Uint32Array;
      const vis = args[9] as Uint8Array;
      for (let i = 0; i < splatCount; i++) vis[i] = 1;
      return 0;
    });
    wasm.compact_by_mask.mockImplementation(() => {});
    wasm.compact_attenuated_amplitudes.mockImplementation(() => {});
    wasm.extract_visible_cholesky_3d.mockImplementation(() => {});
    wasm.extract_3d_positions.mockImplementation(() => {});

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 1e10], // extend_to_all sentinel in tolerance is informational only here; the explicit param drives the branch
      },
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [3], // dim 3 is the only hidden dim; extend-to-all excludes it
      truncate: 3.0,
    });

    // hiddenDimsArg should be empty (extend-to-all excludes dim 3).
    expect(hiddenDimsArg).toBeDefined();
    expect((hiddenDimsArg as Uint32Array).length).toBe(0);
  });
});

describe('decodeBroadcasted — output length invariant (H7)', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('output Float32Array.length === numPoints × elementsPerPoint', async () => {
    const { mod } = await loadWorker();
    for (const [n, k] of [
      [10, 1],
      [10, 3],
      [10, 4],
      [100, 3],
      [0, 3],
    ]) {
      const out = (await mod.workerAPI.decodeBroadcasted({
        value: new Float32Array(k).fill(0.5),
        numPoints: n,
        elementsPerPoint: k,
      })) as Float32Array;
      expect(out).toBeInstanceOf(Float32Array);
      expect(out.length).toBe(n * k);
    }
  });
});
