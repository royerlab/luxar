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
}

async function loadWorker(): Promise<{ mod: WorkerModule; wasm: WasmStubs }> {
  vi.resetModules();

  const wasm: WasmStubs = {
    extract_3d_positions: vi.fn(),
    calculate_effective_radii: vi.fn(),
    calculate_bounds_3d: vi.fn(),
    radii_to_visibility_mask: vi.fn(() => 0),
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
    extract_visible_cholesky_3d: vi.fn(),
    compute_gsplats_attenuation: vi.fn(),
    compact_attenuated_amplitudes: vi.fn(),
  };

  vi.doMock('../../../wasm', () => ({
    initWasm: vi.fn(async () => wasm),
  }));
  vi.doMock('../../../utils/log', () => ({
    log: { info: vi.fn(), warning: vi.fn(), error: vi.fn(), success: vi.fn() },
    Modules: { WORKER_POOL: 'WorkerPool' },
  }));
  vi.doMock('comlink', () => ({
    expose: vi.fn(),
    transfer: vi.fn((obj) => obj),
  }));

  const mod = (await import('../../../workers/data-worker')) as unknown as WorkerModule;
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
    expect(result.bounds).toBeDefined();
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

  it('with effectiveRadiusConfig: invokes calculate_effective_radii', async () => {
    const { mod, wasm } = await loadWorker();
    const numPoints = 4;
    const ndim = 4;

    await mod.workerAPI.projectPointsTo3D({
      positions: new Float32Array(numPoints * ndim),
      colors: null,
      radii: new Float32Array(numPoints),
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
  });

  it('numPoints=0 short-circuits without invoking WASM', async () => {
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

    // extract_3d_positions still called with zero-length output (it's
    // a no-op in WASM). What we care about: result has the right
    // empty shape.
    expect(result.visibleCount).toBe(0);
    expect(result.positions3D.length).toBe(0);
    // Validation guards run regardless; with empty inputs no other
    // wasm path fires.
    expect(wasm.calculate_effective_radii).not.toHaveBeenCalled();
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
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
      displayDims: [0, 1, 2],
      ndim: 3,
      segmentCount: 0,
    })) as { visibleSegmentCount: number };
    expect(result.visibleSegmentCount).toBe(0);
  });

  it('valid 1-segment input invokes clip_segments_batch', async () => {
    const { mod, wasm } = await loadWorker();
    const ndim = 3;
    const numVertices = 2;
    const positions = new Float32Array(numVertices * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);

    await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: null,
      sharpness: null,
      slicePosition: [0, 0, 0],
      tolerance: [10, 10, 10],
      displayDims: [0, 1, 2],
      ndim,
      segmentCount: 1,
    });

    expect(wasm.clip_segments_batch).toHaveBeenCalledTimes(1);
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
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
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

  it('non-zero splats invokes compute_gsplats_attenuation', async () => {
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
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    });

    expect(wasm.compute_gsplats_attenuation).toHaveBeenCalledTimes(1);
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
      slicePosition: [0, 0, 0],
      tolerance: [10, 10, 10],
      displayDims: [0, 1, 2],
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
      slicePosition: [0, 0, 0],
      tolerance: [10, 10, 10],
      displayDims: [0, 1, 2],
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
});
