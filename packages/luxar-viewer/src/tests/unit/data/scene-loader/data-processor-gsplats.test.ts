/**
 * Unit tests for the gsplats data-processor concern.
 *
 * Strategy mirrors data-processor-lines.test.ts:
 *   - Real `THREE.Group` + `THREE.Mesh` for `rootGroup` so the
 *     getObjectByName + nodeType guard is exercised in full.
 *   - Mock `projectGSplatsInProcess` (the in-process dispatcher that
 *     replaced the deleted main-thread `projectGSplats` copy) so tests
 *     aren't coupled to the real Mahalanobis math; we only verify
 *     orchestration.
 *   - Mock the worker pool so we can control success / failure /
 *     fallback paths.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// The non-worker and worker-failure paths now run the shared dispatcher
// in-process (`workers/data-worker/projection/in-process`), not a
// separate main-thread copy. Mock it to detect those paths.
const mockProcessGSplats = vi.fn();
vi.mock('../../../../workers/data-worker/projection/in-process', () => ({
  projectGSplatsInProcess: (...args: unknown[]) => mockProcessGSplats(...args),
}));

const mockUpdateInstancedMesh = vi.fn();
vi.mock('../../../../rendering/gsplat-geometry', () => ({
  updateInstancedGSplatsMesh: (...args: unknown[]) => mockUpdateInstancedMesh(...args),
}));

const mockGetWorkerPool = vi.fn();
vi.mock('../../../../workers/worker-pool', () => ({
  getWorkerPool: () => mockGetWorkerPool(),
}));

vi.mock('../../../../config', () => ({
  config: {
    dataLoading: {
      performance: {
        useWebWorkers: true,
      },
    },
  },
}));

import {
  processGSplatsData,
  WORKER_MIN_SPLATS_3D,
  projectGSplatsTo3DUsingWorker,
} from '../../../../data/scene-loader/process/data-processor-gsplats';
import type { LoadedGSplatsData, GSplatsViewState } from '../../../../types/gsplats';
import { WorkerTimeoutError, WorkerUnavailableError } from '../../../../workers/worker-pool/errors';

/**
 * Dispatcher-shaped result (keyed by `visibleCount`, as the worker /
 * in-process dispatcher returns it). `toProcessed` in the data-processor
 * maps `visibleCount → splatCount`.
 */
function makeDispatcherResult(visibleCount = 2) {
  return {
    centers3D: new Float32Array(visibleCount * 3),
    choleskyFactors3D: new Float32Array(visibleCount * 6),
    amplitudes: new Float32Array(visibleCount),
    colors: new Float32Array(visibleCount * 3),
    visibleCount,
    bounds: {
      min: [0, 0, 0] as [number, number, number],
      max: [1, 1, 1] as [number, number, number],
      maxRowNorm: 0.5,
    },
  };
}

function makeData(splatCount = 100, ndim = 4): LoadedGSplatsData {
  return {
    positions: new Float32Array(splatCount * ndim),
    choleskyFactors: new Float32Array(splatCount * 21),
    amplitudes: new Float32Array(splatCount),
    colors: new Float32Array(splatCount * 3),
    splatCount,
    ndim,
  };
}

function makeViewState(): GSplatsViewState {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, 0],
    tolerance: [1, 1, 1, 1],
    dimensions: [
      { name: 'x', unit: 'px', scale: 1 },
      { name: 'y', unit: 'px', scale: 1 },
      { name: 'z', unit: 'px', scale: 1 },
      { name: 't', unit: 's', scale: 1 },
    ],
  };
}

function makeMesh(name: string, attrs: Record<string, unknown> = {}): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType: 'gsplats',
    attrs,
    visibleSplatCount: 0,
  };
  return mesh;
}

beforeEach(() => {
  mockProcessGSplats.mockReset().mockImplementation(() => makeDispatcherResult());
  mockUpdateInstancedMesh.mockReset();
  mockGetWorkerPool.mockReset();
});

describe('processGSplatsData', () => {
  it('returns null when rootGroup is null', async () => {
    const result = await processGSplatsData('/foo', makeData(), makeViewState(), null, 1);
    expect(result).toBeNull();
  });

  it('returns null when no mesh with the path is found', async () => {
    const root = new THREE.Group();
    const result = await processGSplatsData('/missing', makeData(), makeViewState(), root, 1);
    expect(result).toBeNull();
  });

  it('returns null when mesh exists but is not a gsplats node', async () => {
    const root = new THREE.Group();
    const mesh = new THREE.Mesh();
    mesh.name = '/foo';
    mesh.userData = { nodeType: 'points' };
    root.add(mesh);
    const result = await processGSplatsData('/foo', makeData(), makeViewState(), root, 1);
    expect(result).toBeNull();
  });

  it('runs in-process dispatcher for small datasets (splatCount <= 1000)', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const result = await processGSplatsData('/g', makeData(500), makeViewState(), root, 1);
    expect(result).not.toBeNull();
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
    expect(mockGetWorkerPool).not.toHaveBeenCalled();
  });

  it('rejects a label channel whose length does not match the splat count', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const data = makeData(3);
    data.labelIndices = new Uint32Array([1, 2]);
    data.labelVocabulary = [
      { id: '1', name: 'one' },
      { id: '2', name: 'two' },
    ];

    await expect(processGSplatsData('/g', data, makeViewState(), root, 1)).rejects.toThrow(
      /labelIndices length 2.*splat count 3/
    );
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  it('keeps 3D data in-process below WORKER_MIN_SPLATS_3D even when splatCount > 1000', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    // ndim=3 → the nD threshold does not apply; the fast path is cheap per splat
    const result = await processGSplatsData(
      '/g',
      makeData(2000, 3),
      { ...makeViewState(), slicePosition: [0, 0, 0] },
      root,
      1
    );
    expect(result).not.toBeNull();
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
    expect(mockGetWorkerPool).not.toHaveBeenCalled();
  });

  it('uses worker for large nD datasets', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const projectGSplatsTo3D = vi.fn(async () => ({
      centers3D: new Float32Array(),
      choleskyFactors3D: new Float32Array(),
      amplitudes: new Float32Array(),
      colors: new Float32Array(),
      visibleCount: 0,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectGSplatsTo3D })),
    });

    const result = await processGSplatsData('/g', makeData(2000, 4), makeViewState(), root, 1);
    expect(result).not.toBeNull();
    expect(projectGSplatsTo3D).toHaveBeenCalledTimes(1);
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  // Worker is used iff: useWebWorkers && splatCount > (ndim > 3 ? 1000 : 100_000).
  // These cases pin the exact boundary on both axes. A 3-D node crosses only at
  // WORKER_MIN_SPLATS_3D: its fast path is a copy, so a small node is cheaper
  // in-process, while a multi-million-splat slide blocked the main thread for
  // seconds per load (2026-09 audit).
  it.each([
    { splatCount: 1000, ndim: 4, expectWorker: false, label: '[1000, 4] → in-process' },
    { splatCount: 1001, ndim: 4, expectWorker: true, label: '[1001, 4] → worker' },
    { splatCount: 2000, ndim: 3, expectWorker: false, label: '[2000, 3] → in-process' },
    { splatCount: 2000, ndim: 4, expectWorker: true, label: '[2000, 4] → worker' },
    {
      splatCount: WORKER_MIN_SPLATS_3D,
      ndim: 3,
      expectWorker: false,
      label: '[100_000, 3] → in-process',
    },
    {
      splatCount: WORKER_MIN_SPLATS_3D + 1,
      ndim: 3,
      expectWorker: true,
      label: '[100_001, 3] → worker',
    },
  ])('threshold boundary $label', async ({ splatCount, ndim, expectWorker }) => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    const projectGSplatsTo3D = vi.fn(async () => ({
      centers3D: new Float32Array(),
      choleskyFactors3D: new Float32Array(),
      amplitudes: new Float32Array(),
      colors: new Float32Array(),
      visibleCount: 0,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectGSplatsTo3D })),
    });

    // ndim=3 needs a length-3 slicePosition so the projection inputs are
    // self-consistent; the worker-path cases use the default 4D viewState.
    const viewState =
      ndim === 3 ? { ...makeViewState(), slicePosition: [0, 0, 0] } : makeViewState();

    const result = await processGSplatsData('/g', makeData(splatCount, ndim), viewState, root, 1);
    expect(result).not.toBeNull();

    if (expectWorker) {
      expect(projectGSplatsTo3D).toHaveBeenCalledTimes(1);
      expect(mockProcessGSplats).not.toHaveBeenCalled();
    } else {
      expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
      expect(mockGetWorkerPool).not.toHaveBeenCalled();
    }
  });

  it('stages the projection output directly (6-stride cholesky + fused-scan bounds)', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const result = await processGSplatsData('/g', makeData(50), makeViewState(), root, 1);
    expect(result).not.toBeNull();
    if (!result || result.noop) throw new Error('expected a geometry staged commit');
    // No split/re-interleave pass: the dispatcher's choleskyFactors3D is
    // the staged commit's cholesky source, and the fused-scan bounds
    // metadata rides through toProcessed untouched.
    expect(result.processed.choleskyFactors3D).toBeInstanceOf(Float32Array);
    expect(result.processed.choleskyFactors3D.length).toBe(2 * 6);
    expect(result.processed.bounds).toEqual({
      min: [0, 0, 0],
      max: [1, 1, 1],
      maxRowNorm: 0.5,
    });
  });

  it('derives extendToAllDims from the 1e10 tolerance sentinel (extend_to_all symptom 3, #1157)', async () => {
    // A fully-extended node's derived view state carries the 1e10 sentinel on
    // its non-displayed dim(s). buildGSplatsParams must translate that into
    // `extendToAllDims`, so the projector skips those dims (`hidden-dims.ts`)
    // instead of filtering every splat out on the hidden axis. Regression guard
    // for `isExtendToAll` — with an empty extendToAllDims, gsplat LOD levels of
    // a fully-extended node vanish.
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const viewState: GSplatsViewState = {
      ...makeViewState(),
      // 't' (index 3) is the only non-displayed dim; mark it extend-to-all.
      tolerance: [1, 1, 1, 1e10],
    };
    await processGSplatsData('/g', makeData(50), viewState, root, 1);
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
    const params = mockProcessGSplats.mock.calls[0][0] as {
      extendToAllDims: number[];
      discreteDims: number[];
    };
    expect(params.extendToAllDims).toEqual([3]);
    // An extend-to-all dim is NOT also treated as a discrete slice dim.
    expect(params.discreteDims).not.toContain(3);
  });

  // -----------------------------------------------------------------------
  // Slot → on-disk element-ID map (issue #1423).
  //
  // The gsplats pick shader reports the visible-buffer slot, but the label CSR
  // is keyed by the on-disk splat index. The loader publishes its visible
  // `ranges` (labelled nodes only) and the kernel records which source splat
  // each slot came from; `toProcessed` composes the two.
  // -----------------------------------------------------------------------

  it('composes elementIds from the loader ranges + the kernel source indices', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    // Two visible chunks — concat [0, 2048) ↦ on-disk [2048, 4096) and concat
    // [2048, 4096) ↦ on-disk [6144, 8192) — with the projection keeping three
    // splats straddling the boundary.
    mockProcessGSplats.mockImplementation(() => ({
      ...makeDispatcherResult(3),
      sourceIndices: new Uint32Array([0, 2047, 2048]),
    }));

    // Under the 1000-splat worker threshold so the in-process mock runs.
    const data: LoadedGSplatsData = {
      ...makeData(500),
      ranges: [
        { start: 2048, end: 4096 },
        { start: 6144, end: 8192 },
      ],
    };

    const result = await processGSplatsData('/g', data, makeViewState(), root, 1);
    if (!result || result.noop) throw new Error('expected a geometry staged commit');

    // Without the fix `elementIds` doesn't exist and hover reads the raw slot,
    // which for slot 0 would report on-disk splat 0 instead of 2048.
    expect(result.processed.elementIds).toBeInstanceOf(Uint32Array);
    expect(Array.from(result.processed.elementIds!)).toEqual([2048, 4095, 6144]);
  });

  it('asks the kernel to record source indices only for a ranges-publishing nD node', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    const data: LoadedGSplatsData = {
      ...makeData(50),
      ranges: [{ start: 0, end: 50 }],
    };
    await processGSplatsData('/g', data, makeViewState(), root, 1);

    const params = mockProcessGSplats.mock.calls[0][0] as { emitSourceIndices?: boolean };
    expect(params.emitSourceIndices).toBe(true);
  });

  it('does not ask for source indices for a node that publishes no ranges', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    // No label CSR ⇒ the loader publishes no `ranges` ⇒ no map can be composed
    // from the recorded indices, so recording them would be 4 B/splat spent on
    // an array nothing reads. The nD/compacting half of the gate is satisfied
    // here (makeViewState is the nD path), so this pins the `ranges` half.
    await processGSplatsData('/g', makeData(50), makeViewState(), root, 1);

    const params = mockProcessGSplats.mock.calls[0][0] as { emitSourceIndices?: boolean };
    expect(params.emitSourceIndices).toBe(false);
  });

  it('does not ask for source indices on the standard-3D fast path', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    // ndim 3 with displayDims [0,1,2]: every splat is emitted in order, so the
    // dispatcher's fast path returns none and the composer's range-offset path
    // is exactly right. Asking would allocate 4 B/splat for nothing.
    const data: LoadedGSplatsData = {
      ...makeData(50, 3),
      choleskyFactors: new Float32Array(50 * 6),
      ranges: [{ start: 2048, end: 2098 }],
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };
    await processGSplatsData('/g', data, viewState, root, 1);

    const params = mockProcessGSplats.mock.calls[0][0] as { emitSourceIndices?: boolean };
    expect(params.emitSourceIndices).toBe(false);
  });

  it('composes elementIds as pure range offsets on the standard-3D fast path', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));

    // The most common labelled-gsplats hover case: a plain 3D node whose
    // `ranges` were chunk-culled, so nothing is compacted (the fast path emits
    // every splat in order and records no source indices) but the slots are
    // still offset — and shifted again across the gap between the two ranges.
    mockProcessGSplats.mockImplementation(() => makeDispatcherResult(5));

    const data: LoadedGSplatsData = {
      ...makeData(5, 3),
      choleskyFactors: new Float32Array(5 * 6),
      ranges: [
        { start: 2048, end: 2051 },
        { start: 6144, end: 6146 },
      ],
    };
    const viewState: GSplatsViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0],
      tolerance: [1, 1, 1],
    };

    const result = await processGSplatsData('/g', data, viewState, root, 1);
    if (!result || result.noop) throw new Error('expected a geometry staged commit');

    // `null` source indices ⇒ the composer's range-offset path: slot 0 is the
    // first range's start, and slot 3 lands in the second range, skipping the
    // 3093-splat gap rather than reading a wrong-but-plausible neighbour.
    expect(result.processed.elementIds).toBeInstanceOf(Uint32Array);
    expect(Array.from(result.processed.elementIds!)).toEqual([2048, 2049, 2050, 6144, 6145]);
  });

  it('leaves elementIds undefined for a node that publishes no ranges', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    mockProcessGSplats.mockImplementation(() => ({
      ...makeDispatcherResult(2),
      sourceIndices: new Uint32Array([0, 5]),
    }));

    const result = await processGSplatsData('/g', makeData(50), makeViewState(), root, 1);
    if (!result || result.noop) throw new Error('expected a geometry staged commit');

    // No label CSR on the node ⇒ no ranges ⇒ no map, even though the kernel
    // result happens to carry source indices.
    expect(result.processed.elementIds).toBeUndefined();
  });
});

describe('projectGSplatsTo3DUsingWorker', () => {
  it('returns mapped worker result on success', async () => {
    const projectGSplatsTo3D = vi.fn(async () => ({
      centers3D: new Float32Array([1, 2, 3]),
      choleskyFactors3D: new Float32Array(),
      amplitudes: new Float32Array(),
      colors: new Float32Array(),
      visibleCount: 1,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectGSplatsTo3D })),
    });

    const result = await projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1);
    expect(result.splatCount).toBe(1);
    expect(Array.from(result.centers3D)).toEqual([1, 2, 3]);
  });

  // Worker UNAVAILABILITY — the pool never got the work to a worker at all, so
  // the in-process dispatcher is the only executor left.
  it('falls back to the in-process dispatcher on WorkerUnavailableError', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new WorkerUnavailableError('[WorkerPool] No workers available after initialization');
      }),
    });
    mockProcessGSplats.mockReturnValue(makeDispatcherResult(7));

    const result = await projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1);
    expect(result.splatCount).toBe(7);
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
  });

  // A timeout does NOT establish infrastructure failure: the worker may be hung
  // inside a data-dependent kernel, or the projection may genuinely exceed the
  // budget — either way a main-thread rerun blocks the frame at least as long
  // again. The pool has already evicted the worker, so propagating leaves the
  // node failed-but-retryable against a fresh one.
  it('propagates a worker timeout instead of re-running the projection on the main thread', async () => {
    const timeout = new WorkerTimeoutError('projectGSplatsTo3D', 60000);
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw timeout;
      }),
    });

    await expect(projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1)).rejects.toBe(
      timeout
    );
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  // The regression this guards: `projectGSplatsInProcess` runs the SAME kernel
  // through the same `pickBackend`, so re-running a rejected input reproduces the
  // fault on the UI thread — a WASM trap there blocks the frame. A rejection that
  // came back FROM the worker must propagate untouched.
  it('re-throws a kernel fault instead of re-running it on the main thread', async () => {
    // Shape of a WASM trap crossing the Comlink boundary.
    const trap = new Error('unreachable');
    trap.name = 'RuntimeError';
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw trap;
      }),
    });

    await expect(projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1)).rejects.toBe(
      trap
    );
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  // A worker-RETURNED error reconstructed across Comlink loses its prototype but
  // may still carry the name string. It must NOT be treated as infrastructure —
  // otherwise the rejected kernel re-runs on the UI thread, the exact fault this
  // guards. `instanceof` is what makes the name insufficient.
  it('does not fall back for a non-instance error that merely spoofs the infra name', async () => {
    const spoof = new Error('No workers available');
    spoof.name = 'WorkerUnavailableError';
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw spoof;
      }),
    });

    await expect(projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1)).rejects.toBe(
      spoof
    );
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  // Fail closed: an unrecognized error is NOT assumed to be infrastructure.
  it('re-throws an unrecognized error rather than assuming infrastructure failure', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new Error('boom');
      }),
    });

    await expect(
      projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1)
    ).rejects.toThrow('boom');
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });

  // A dataset-switch abort must still short-circuit before the infra check.
  it('re-throws a dataset-switch abort without falling back', async () => {
    const abort = new Error('aborted by caller signal');
    abort.name = 'WorkerAbortError';
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw abort;
      }),
    });

    await expect(projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1)).rejects.toBe(
      abort
    );
    expect(mockProcessGSplats).not.toHaveBeenCalled();
  });
});
