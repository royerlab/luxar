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

  it('runs in-process dispatcher for 3D data even when splatCount > 1000', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    // ndim=3 → worker NOT used regardless of count
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

  // Worker is used iff: useWebWorkers && splatCount > 1000 && ndim > 3.
  // These cases pin the exact boundary on both axes.
  it.each([
    { splatCount: 1000, ndim: 4, expectWorker: false, label: '[1000, 4] → in-process' },
    { splatCount: 1001, ndim: 4, expectWorker: true, label: '[1001, 4] → worker' },
    { splatCount: 2000, ndim: 3, expectWorker: false, label: '[2000, 3] → in-process' },
    { splatCount: 2000, ndim: 4, expectWorker: true, label: '[2000, 4] → worker' },
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
