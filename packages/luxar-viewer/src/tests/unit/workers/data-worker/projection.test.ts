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
  clip_segments_batch: ReturnType<typeof vi.fn>;
  interpolate_clipped_positions: ReturnType<typeof vi.fn>;
  interpolate_colors_batch: ReturnType<typeof vi.fn>;
  interpolate_scalars_batch: ReturnType<typeof vi.fn>;
  calculate_segment_lengths: ReturnType<typeof vi.fn>;
  compute_cap_suppression: ReturnType<typeof vi.fn>;
  compact_by_mask: ReturnType<typeof vi.fn>;
  extract_visible_cholesky_3d: ReturnType<typeof vi.fn>;
  compute_gsplats_attenuation: ReturnType<typeof vi.fn>;
  compact_attenuated_amplitudes: ReturnType<typeof vi.fn>;
  project_gsplats_nd_to_3d: ReturnType<typeof vi.fn>;
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
    clip_segments_batch: vi.fn(real.clip_segments_batch.bind(real)),
    interpolate_clipped_positions: vi.fn(real.interpolate_clipped_positions.bind(real)),
    interpolate_colors_batch: vi.fn(real.interpolate_colors_batch.bind(real)),
    interpolate_scalars_batch: vi.fn(real.interpolate_scalars_batch.bind(real)),
    calculate_segment_lengths: vi.fn(real.calculate_segment_lengths.bind(real)),
    compute_cap_suppression: vi.fn(real.compute_cap_suppression.bind(real)),
    compact_by_mask: vi.fn(real.compact_by_mask.bind(real)),
    extract_visible_cholesky_3d: vi.fn(real.extract_visible_cholesky_3d.bind(real)),
    compute_gsplats_attenuation: vi.fn(real.compute_gsplats_attenuation.bind(real)),
    compact_attenuated_amplitudes: vi.fn(real.compact_attenuated_amplitudes.bind(real)),
    project_gsplats_nd_to_3d: vi.fn(real.project_gsplats_nd_to_3d.bind(real)),
    decode_broadcasted: vi.fn(real.decode_broadcasted.bind(real)),
  };

  vi.doMock('../../../../wasm', () => ({
    initWasm: vi.fn(async () => wasm),
    isWasmFallback: vi.fn(() => false),
    // Uncapped TS backend for >16D routing (pickBackend); reuse the same stub.
    getFallback: vi.fn(() => wasm),
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

describe('projectLinesTo3D — happy paths', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('zero segments returns a zero-shape output without throwing', async () => {
    const { mod } = await loadWorker();
    // The canonical empty payload (createEmptyLinesData: ALL arrays
    // zero-length, matching the zero-splat gsplats test below). A
    // regression here silently freezes stale Lines geometry: the loader
    // returns this payload when a node's data falls outside the current
    // slice, and only a successful empty projection lets the commit step
    // clear the mesh.
    const result = (await mod.workerAPI.projectLinesTo3D({
      positions: new Float32Array(0),
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

  it('canonical empty payload with a non-displayed dim projects to empty (nD scrub re-cull regression)', async () => {
    // Regression: scrubbing a non-displayed dim to a slot where a Lines
    // node has no data made the loader return the canonical empty payload,
    // which projectLinesTo3D REJECTED (hardcoded numItems=1 required ≥ 1
    // vertex). The throw was swallowed as a failed loader update, so the
    // node's previous-slot geometry stayed rendered forever — Lines
    // accumulated (A ∪ B) where Points/GSplats swapped (A xor B).
    const { mod } = await loadWorker();
    const result = (await mod.workerAPI.projectLinesTo3D({
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 1], // hidden dim 3 scrubbed to slot 1
        tolerance: [1, 1, 1, 0.5], // half-cell membership slab
      },
      ndim: 4,
      segmentCount: 0,
    })) as { visibleSegmentCount: number; startPositions: Float32Array };
    expect(result.visibleSegmentCount).toBe(0);
    expect(result.startPositions.length).toBe(0);
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

  it('#780 a straight polyline suppresses the cap at every INTERIOR joint and keeps it at the two free ends', async () => {
    // End-to-end through the real projection pipeline: a 4-segment straight
    // polyline along +x. Before the fix every interior joint kept the 0.5
    // endpoint dip with no overlapping neighbour to add the missing half
    // back, so thick lines rendered as a chain of beads.
    const { mod } = await loadWorker();
    const ndim = 3;
    const nVerts = 5;
    const positions = new Float32Array(nVerts * ndim);
    for (let i = 0; i < nVerts; i++) positions[i * ndim] = i; // (i, 0, 0)
    const segments = new Uint32Array([0, 1, 1, 2, 2, 3, 3, 4]);

    const result = (await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths: new Float32Array(nVerts).fill(1),
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [1e6, 1e6, 1e6],
      },
      ndim,
      segmentCount: 4,
    })) as {
      visibleSegmentCount: number;
      startCapSuppression: Float32Array;
      endCapSuppression: Float32Array;
    };

    expect(result.visibleSegmentCount).toBe(4);
    // Free start of the first segment and free end of the last keep the cap;
    // all six interior endpoints are fully suppressed.
    expect(Array.from(result.startCapSuppression)).toEqual([0, 1, 1, 1]);
    expect(Array.from(result.endCapSuppression)).toEqual([1, 1, 1, 0]);
  });

  it('#780 a right-angle polyline keeps the cap at the corner (quads genuinely overlap there)', async () => {
    const { mod } = await loadWorker();
    const ndim = 3;
    // v0 (0,0,0) -> v1 (1,0,0) -> v2 (1,1,0): one 90-degree turn at v1.
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0]);

    const result = (await mod.workerAPI.projectLinesTo3D({
      positions,
      segments: new Uint32Array([0, 1, 1, 2]),
      widths: new Float32Array(3).fill(1),
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [1e6, 1e6, 1e6],
      },
      ndim,
      segmentCount: 2,
    })) as { startCapSuppression: Float32Array; endCapSuppression: Float32Array };

    // Corner keeps the full cap — behaviour identical to before the fix.
    expect(result.endCapSuppression[0]).toBe(0);
    expect(result.startCapSuppression[1]).toBe(0);
  });

  it('#780 an nD-clipped endpoint still reports full suppression', async () => {
    // A 4D polyline where the slice cuts the second segment mid-way: the cut
    // end is not a real endpoint, so it must stay fully suppressed (the
    // pre-existing clipped-flag contract, preserved by the new kernel).
    const { mod } = await loadWorker();
    const ndim = 4;
    // dim 3 is the hidden/slicing axis. v0,v1 at w=0; v2 at w=10.
    const positions = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 10]);

    const result = (await mod.workerAPI.projectLinesTo3D({
      positions,
      segments: new Uint32Array([0, 1, 1, 2]),
      widths: new Float32Array(3).fill(1),
      colors: null,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [1e6, 1e6, 1e6, 0.5],
      },
      ndim,
      segmentCount: 2,
    })) as {
      visibleSegmentCount: number;
      startCapSuppression: Float32Array;
      endCapSuppression: Float32Array;
    };

    expect(result.visibleSegmentCount).toBe(2);
    // Segment 1 is cut by the slice → its end is clipped → suppression 1.
    expect(result.endCapSuppression[1]).toBe(1);
    // v1 is still a genuine straight-through joint reached by both segments.
    expect(result.endCapSuppression[0]).toBeCloseTo(1, 6);
    expect(result.startCapSuppression[1]).toBeCloseTo(1, 6);
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
  it('non-zero splats: the fused project_gsplats_nd_to_3d receives the expected shaped args', async () => {
    // [W5] The gsplat worker path now makes a SINGLE fused WASM call instead of
    // the former 6-call pipeline. Pin the fused kernel's argument shape so a
    // refactor that reordered/dropped the positions/cholesky/amplitudes/colors
    // buffers, the discrete-visibility mask, or the worst-case output buffers
    // would surface as a failed shape match.
    const { mod, wasm } = await loadWorker();
    // [W4] ndim=4: ndim=3 with standard displayDims now takes the copy
    // fast path that bypasses the fused kernel. Use an nD projection so
    // the fused-kernel argument shape is actually exercised.
    const splatCount = 2;
    const ndim = 4;
    const k = (ndim * (ndim + 1)) / 2; // 10 for 4D

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
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
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    });

    // The granular kernels are no longer called for the gsplat path.
    expect(wasm.compute_gsplats_attenuation).not.toHaveBeenCalled();
    expect(wasm.extract_visible_cholesky_3d).not.toHaveBeenCalled();

    expect(wasm.project_gsplats_nd_to_3d).toHaveBeenCalledTimes(1);
    const a = (wasm.project_gsplats_nd_to_3d as any).mock.calls[0];
    // positions, cholesky, amplitudes, colors (white-filled f32), discreteVisibility
    expect(a[0]).toBeInstanceOf(Float32Array);
    expect((a[0] as Float32Array).length).toBe(splatCount * ndim);
    expect((a[1] as Float32Array).length).toBe(splatCount * k);
    expect((a[2] as Float32Array).length).toBe(splatCount);
    expect(a[3]).toBeInstanceOf(Float32Array); // colors (coerced/white) length splatCount*3
    expect((a[3] as Float32Array).length).toBe(splatCount * 3);
    expect(a[4]).toBeInstanceOf(Uint8Array); // discreteVisibility
    expect((a[4] as Uint8Array).length).toBe(splatCount);
    // colorComponents (3 = RGB here; the null-colors path white-fills RGB)
    expect(a[10]).toBe(3);
    // worst-case output buffers sized to splatCount (indices shifted +1 by
    // the colorComponents arg at index 10)
    expect((a[13] as Float32Array).length).toBe(splatCount * 3); // outCenters
    expect((a[14] as Float32Array).length).toBe(splatCount * 6); // outCholesky
    expect((a[15] as Float32Array).length).toBe(splatCount); // outAmplitudes
    expect((a[16] as Float32Array).length).toBe(splatCount * 3); // outColors
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

  it('Lines: RGBA colors are de-interleaved — RGB stride-3 to interpolate_colors_batch, the alpha column through interpolate_scalars_batch (volumetric phase 4)', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const positions = new Float32Array(2 * ndim);
    const segments = new Uint32Array([0, 1]);
    const widths = new Float32Array([1.0, 1.0]);
    // 2 RGBA vertices; distinct alphas 217/255 and 128/255.
    const colorsU8 = new Uint8Array([0, 128, 255, 217, 64, 200, 32, 128]);

    const result = (await mod.workerAPI.projectLinesTo3D({
      positions,
      segments,
      widths,
      colors: colorsU8,
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
      colorComponents: 4,
    })) as { startAlphas: Float32Array; endAlphas: Float32Array };

    // The RGB kernel is stride-3 by contract: it must receive the
    // de-interleaved RGB array, NOT the 4-strided input (which would
    // read vertex 1's color as [alpha0, r1, g1]).
    const colorsArg = (wasm.interpolate_colors_batch.mock.calls[0] as unknown[])[0];
    expect(colorsArg).toBeInstanceOf(Float32Array);
    const rgb = colorsArg as Float32Array;
    expect(rgb.length).toBe(2 * 3);
    expect(rgb[0]).toBeCloseTo(0, 5);
    expect(rgb[1]).toBeCloseTo(128 / 255, 5);
    expect(rgb[2]).toBeCloseTo(1, 5);
    expect(rgb[3]).toBeCloseTo(64 / 255, 5); // vertex 1 red — NOT alpha 0
    expect(rgb[5]).toBeCloseTo(32 / 255, 5);

    // Alpha rides the scalar kernel (the widths/sharpness path). The
    // color step runs before the widths step, so the alpha column is
    // the FIRST interpolate_scalars_batch call; the column is the
    // normalized [0, 1] alpha, one entry per vertex.
    const alphaArg = (wasm.interpolate_scalars_batch.mock.calls[0] as unknown[])[0];
    expect(alphaArg).toBeInstanceOf(Float32Array);
    const alphas = alphaArg as Float32Array;
    expect(alphas.length).toBe(2);
    expect(alphas[0]).toBeCloseTo(217 / 255, 5);
    expect(alphas[1]).toBeCloseTo(128 / 255, 5);

    // Per-segment alpha outputs are allocated (one per visible segment).
    expect(result.startAlphas).toBeInstanceOf(Float32Array);
    expect(result.startAlphas.length).toBe(1);
    expect(result.endAlphas.length).toBe(1);
  });

  it('Lines: RGB colors emit EMPTY alpha arrays and no extra scalar-interp call', async () => {
    const { mod, wasm } = await loadWorker();
    wasm.clip_segments_batch.mockImplementation(() => 1);
    const ndim = 3;
    const result = (await mod.workerAPI.projectLinesTo3D({
      positions: new Float32Array(2 * ndim),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([1.0, 1.0]),
      colors: new Uint8Array([0, 128, 255, 64, 200, 32]),
      sharpness: null,
      scalars: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [10, 10, 10],
      },
      ndim,
      segmentCount: 1,
    })) as { startAlphas: Float32Array; endAlphas: Float32Array };
    // Only widths ride the scalar kernel on the RGB path.
    expect(wasm.interpolate_scalars_batch).toHaveBeenCalledTimes(1);
    expect(result.startAlphas.length).toBe(0);
    expect(result.endAlphas.length).toBe(0);
  });

  it('GSplats: Uint8Array colors are normalized to Float32 [0,1] before the fused kernel', async () => {
    const { mod, wasm } = await loadWorker();
    // [W5] Colors are coerced to normalized f32 once on the worker side and
    // passed as the `colors` arg (index 3) of the single fused kernel — the
    // /255 contract stays in color-utils, not in WASM.
    // [W4] Use ndim=4 (an actual nD projection): ndim=3 with standard
    // displayDims=[0,1,2] now takes the copy fast path that bypasses the
    // fused kernel, so it wouldn't exercise the boundary under test.
    const splatCount = 2;
    const ndim = 4;
    const k = (ndim * (ndim + 1)) / 2; // 10 for 4D
    const colorsU8 = new Uint8Array([0, 128, 255, 64, 200, 32]);

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: colorsU8,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
      },
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [],
      truncate: 3.0,
    });

    expect(wasm.project_gsplats_nd_to_3d).toHaveBeenCalledTimes(1);
    const colorsArg = (wasm.project_gsplats_nd_to_3d as any).mock.calls[0][3] as Float32Array;
    expect(colorsArg).toBeInstanceOf(Float32Array);
    expect(colorsArg[0]).toBeCloseTo(0, 5);
    expect(colorsArg[1]).toBeCloseTo(128 / 255, 5);
    expect(colorsArg[2]).toBeCloseTo(1, 5);
    expect(colorsArg[3]).toBeCloseTo(64 / 255, 5);
    expect(colorsArg[4]).toBeCloseTo(200 / 255, 5);
    expect(colorsArg[5]).toBeCloseTo(32 / 255, 5);
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

    // [W5] The discrete prefilter runs in TS and is passed to the fused kernel
    // as the `discreteVisibility` mask (arg index 4). Splat 2 (d3=2, > 0.5*step)
    // must be gated out there, independent of continuous attenuation.
    expect(wasm.project_gsplats_nd_to_3d).toHaveBeenCalledTimes(1);
    const discreteVis = (wasm.project_gsplats_nd_to_3d as any).mock.calls[0][4] as Uint8Array;
    expect(discreteVis).toBeInstanceOf(Uint8Array);
    expect(Array.from(discreteVis)).toEqual([1, 1, 0, 1]);
  });

  it('extendToAllDims path: dim is excluded from the continuous hidden dims passed to WASM', async () => {
    const { mod, wasm } = await loadWorker();
    const splatCount = 2;
    const ndim = 4;
    const k = (ndim * (ndim + 1)) / 2;

    await mod.workerAPI.projectGSplatsTo3D({
      positions: new Float32Array(splatCount * ndim),
      choleskyFactors: new Float32Array(splatCount * k),
      amplitudes: new Float32Array(splatCount).fill(1),
      colors: null,
      sharpness: null,
      viewState: {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 1e10], // sentinel informational; explicit param drives the branch
      },
      ndim,
      splatCount,
      discreteDims: [],
      discreteSteps: {},
      extendToAllDims: [3], // dim 3 is the only hidden dim; extend-to-all excludes it
      truncate: 3.0,
    });

    // [W5] continuousHiddenDims is arg index 7 of the fused kernel; with dim 3
    // extend-to-all it should be empty (the only hidden dim was excluded).
    expect(wasm.project_gsplats_nd_to_3d).toHaveBeenCalledTimes(1);
    const continuousHidden = (wasm.project_gsplats_nd_to_3d as any).mock.calls[0][6] as Uint32Array;
    expect(continuousHidden).toBeInstanceOf(Uint32Array);
    expect(continuousHidden.length).toBe(0);
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
