/**
 * Unit tests for the lines data-processor concern.
 *
 * Strategy:
 *   - Use a real `THREE.Group` + `THREE.Mesh` for `rootGroup` so the
 *     getObjectByName lookup matches the inline original.
 *   - Mock the worker pool through the existing module path so we can
 *     control whether `projectLinesTo3DUsingWorker` succeeds, throws,
 *     or returns canned data.
 *   - Mock `appConfig` indirectly: the threshold (`segmentCount > 1000`)
 *     is exercised by simply varying `data.segmentCount` in tests.
 *   - Mock `projectLinesInProcess` (the in-process dispatcher that
 *     replaced the deleted main-thread `projectLinesTo3D` copy) so we can
 *     detect the non-worker / worker-failure paths without running the
 *     real clipping math.
 *
 * The class isLinesUserData / mesh.userData.attrs / nodeType plumbing
 * is real — that's the boundary the helper guards on.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

// The non-worker and worker-failure paths now run the shared dispatcher
// in-process. It takes a SINGLE params object (not positional args), so
// `mock.calls[0][0]` is the params; `params.viewState.tolerance` carries
// the per-dim tolerance the old positional `calls[0][2]` exposed.
const mockBuildInstanceBuffers = vi.fn();
vi.mock('../../../../workers/data-worker/projection/in-process', () => ({
  projectLinesInProcess: (...args: unknown[]) => mockBuildInstanceBuffers(...args),
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
  processLinesData,
  projectLinesTo3DUsingWorker,
} from '../../../../data/scene-loader/process/data-processor-lines';
import type { LoadedLinesData } from '../../../../types/lines';
import { WorkerTimeoutError, WorkerUnavailableError } from '../../../../workers/worker-pool/errors';

/**
 * Dispatcher-shaped result (keyed by `visibleSegmentCount`, plus empty
 * `startScalars`/`endScalars`, as the worker / in-process dispatcher
 * returns it). `toProcessedLines` in the data-processor maps
 * `visibleSegmentCount → segmentCount`.
 */
function makeDispatcherLinesResult(visibleSegmentCount = 2) {
  return {
    startPositions: new Float32Array(visibleSegmentCount * 3),
    endPositions: new Float32Array(visibleSegmentCount * 3),
    startColors: new Float32Array(visibleSegmentCount * 3),
    endColors: new Float32Array(visibleSegmentCount * 3),
    startWidths: new Float32Array(visibleSegmentCount),
    endWidths: new Float32Array(visibleSegmentCount),
    startSharpness: new Float32Array(visibleSegmentCount),
    endSharpness: new Float32Array(visibleSegmentCount),
    startScalars: new Float32Array(0),
    endScalars: new Float32Array(0),
    segmentLengths: new Float32Array(visibleSegmentCount),
    startJointCode: new Float32Array(visibleSegmentCount),
    endJointCode: new Float32Array(visibleSegmentCount),
    visibleSegmentCount,
  };
}

function makeData(segmentCount = 100): LoadedLinesData {
  return {
    positions: new Float32Array(segmentCount * 6),
    segments: new Uint32Array(segmentCount * 2),
    widths: new Float32Array(segmentCount * 2),
    colors: new Float32Array(segmentCount * 6),
    sharpness: new Float32Array(segmentCount * 2),
    scalars: undefined,
    segmentCount,
    vertexCount: segmentCount * 2,
    ndim: 3,
  };
}

function makeMesh(name: string, attrs: Record<string, unknown> = {}): THREE.Mesh {
  const mesh = new THREE.Mesh();
  mesh.name = name;
  mesh.userData = {
    nodeType: 'lines',
    attrs,
    visibleSegmentCount: 0,
  };
  return mesh;
}

beforeEach(() => {
  mockBuildInstanceBuffers.mockReset().mockImplementation(() => makeDispatcherLinesResult());
  mockGetWorkerPool.mockReset();
});

describe('processLinesData', () => {
  it('returns null when rootGroup is null', async () => {
    const result = await processLinesData(
      '/foo',
      makeData(),
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      null,
      1
    );
    expect(result).toBeNull();
  });

  it('returns null when no mesh with the path is found', async () => {
    const root = new THREE.Group();
    const result = await processLinesData(
      '/missing',
      makeData(),
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      root,
      1
    );
    expect(result).toBeNull();
  });

  it('returns null when the mesh exists but is not a lines node', async () => {
    const root = new THREE.Group();
    const mesh = new THREE.Mesh();
    mesh.name = '/foo';
    mesh.userData = { nodeType: 'points', attrs: {} };
    root.add(mesh);
    const result = await processLinesData(
      '/foo',
      makeData(),
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0],
        tolerance: [0, 0, 0],
      },
      root,
      1
    );
    expect(result).toBeNull();
  });

  it('stages an empty commit (segmentCount 0) for the canonical empty payload', async () => {
    // Regression: an empty payload (loader found no data at the current
    // slice of a non-displayed dim) must stage a REAL commit with
    // segmentCount 0 — that commit is what clears the previous slice's
    // geometry. Pre-fix, projection rejected the empty payload and the
    // throw left stale geometry rendered forever (Lines accumulated
    // across scrubs instead of swapping like Points/GSplats).
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const empty: LoadedLinesData = {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      segmentCount: 0,
      vertexCount: 0,
      ndim: 4,
    };
    mockBuildInstanceBuffers.mockReturnValue(makeDispatcherLinesResult(0));

    const result = await processLinesData(
      '/lines',
      empty,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0, 1], tolerance: [0, 0, 0, 0] },
      root,
      1
    );

    expect(result).not.toBeNull();
    expect(result && !result.noop && result.processed.segmentCount).toBe(0);
    // Empty data (0 ≤ worker threshold) runs the in-process dispatcher.
    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
  });

  it('uses the in-process dispatcher for small datasets', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const result = await processLinesData(
      '/lines',
      makeData(500),
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      root,
      1
    );
    expect(result).not.toBeNull();
    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
    expect(mockGetWorkerPool).not.toHaveBeenCalled();
  });

  it('keeps scalar PRESENCE at a zero-visible-segment slice (empty-but-defined fields)', async () => {
    // Presence follows the SOURCE, not the visible count: at a slice
    // with 0 visible segments the dispatcher's scalar arrays are empty,
    // but the node still HAS scalars — dropping the fields here used to
    // flip the geometry's `hasScalars` stamp false on the next commit,
    // silently suppressing a colormap picked while the slice was empty
    // (nothing re-applies it when segments return). Points keep an
    // empty-but-defined subarray; lines must match.
    mockBuildInstanceBuffers.mockImplementation(() => makeDispatcherLinesResult(0));
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const data = makeData(500);
    data.scalars = new Float32Array(data.vertexCount);
    const result = await processLinesData(
      '/lines',
      data,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      root,
      1
    );
    expect(result).not.toBeNull();
    const staged = result as {
      processed: { startScalars?: Float32Array; endScalars?: Float32Array; segmentCount: number };
    };
    expect(staged.processed.segmentCount).toBe(0);
    expect(staged.processed.startScalars).toBeInstanceOf(Float32Array);
    expect(staged.processed.endScalars).toBeInstanceOf(Float32Array);

    // The no-source-scalars signal is preserved: absent source ⇒
    // absent fields, regardless of what the dispatcher returned.
    mockBuildInstanceBuffers.mockImplementation(() => makeDispatcherLinesResult(0));
    const noScalars = await processLinesData(
      '/lines',
      makeData(500),
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      root,
      1
    );
    const stagedNo = noScalars as { processed: { startScalars?: Float32Array } };
    expect(stagedNo.processed.startScalars).toBeUndefined();
  });

  it('uses worker projection for large datasets', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines'));
    const projectLinesTo3D = vi.fn(async () => ({
      startPositions: new Float32Array(),
      endPositions: new Float32Array(),
      startColors: new Float32Array(),
      endColors: new Float32Array(),
      startWidths: new Float32Array(),
      endWidths: new Float32Array(),
      startSharpness: new Float32Array(),
      endSharpness: new Float32Array(),
      segmentLengths: new Float32Array(),
      startJointCode: new Float32Array(),
      endJointCode: new Float32Array(),
      visibleSegmentCount: 0,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectLinesTo3D })),
    });

    const result = await processLinesData(
      '/lines',
      makeData(2000), // > 1000 → worker path
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      root,
      1
    );
    expect(result).not.toBeNull();
    expect(projectLinesTo3D).toHaveBeenCalledTimes(1);
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
  });

  // Regression (deep-double-check round 5): pins that processLinesData
  // requests the MEMBERSHIP role at the call site — dropping the
  // { discreteRole: 'membership' } options object (the exact regression
  // class ee0971f3 introduced) would silently revert the lines clipping
  // slab from 0.5 x step to the 0.25 x step query reach with every other
  // test still green.
  it('passes the half-cell MEMBERSHIP tolerance for hidden discrete dims to projection', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines', {}));
    const data = makeData(100);
    data.ndim = 4;

    await processLinesData(
      '/lines',
      data,
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
        dimensions: [
          { name: 'x', unit: 'px', scale: 1 },
          { name: 'y', unit: 'px', scale: 1 },
          { name: 'z', unit: 'px', scale: 1 },
          { name: 't', unit: 's', scale: 1, discrete: true, step: 1 },
        ],
      },
      root,
      1
    );

    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
    const params = mockBuildInstanceBuffers.mock.calls[0][0] as {
      viewState: { tolerance: number[] };
    };
    // 0.5 x step (membership slab), NOT 0.25 x step (query reach).
    expect(params.viewState.tolerance[3]).toBe(0.5);
  });

  it('extends tolerance to 1e10 for extend_to_all dimensions', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/lines', { extend_to_all: ['t'] }));
    // ndim=4 so the computed tolerance has 4 entries and dim "t" at index 3 fits.
    const data = makeData(100);
    data.ndim = 4;

    await processLinesData(
      '/lines',
      data,
      {
        displayDims: [0, 1, 2],
        slicePosition: [0, 0, 0, 0],
        tolerance: [0, 0, 0, 0],
        dimensions: [
          { name: 'x', unit: 'px', scale: 1 },
          { name: 'y', unit: 'px', scale: 1 },
          { name: 'z', unit: 'px', scale: 1 },
          { name: 't', unit: 's', scale: 1 },
        ],
      },
      root,
      1
    );

    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
    // In-process dispatcher takes one params object; tolerance lives in
    // params.viewState.tolerance.
    const params = mockBuildInstanceBuffers.mock.calls[0][0] as {
      viewState: { tolerance: number[] };
    };
    expect(params.viewState.tolerance[3]).toBe(1e10);
  });
});

describe('projectLinesTo3DUsingWorker', () => {
  it('returns the worker result mapped to ProcessedLinesData on success', async () => {
    const projectLinesTo3D = vi.fn(async () => ({
      startPositions: new Float32Array([1, 2, 3]),
      endPositions: new Float32Array([4, 5, 6]),
      startColors: new Float32Array(),
      endColors: new Float32Array(),
      startWidths: new Float32Array(),
      endWidths: new Float32Array(),
      startSharpness: new Float32Array(),
      endSharpness: new Float32Array(),
      segmentLengths: new Float32Array(),
      startJointCode: new Float32Array(),
      endJointCode: new Float32Array(),
      visibleSegmentCount: 1,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectLinesTo3D })),
    });

    const result = await projectLinesTo3DUsingWorker(
      makeData(),
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    );
    expect(result.segmentCount).toBe(1);
    expect(Array.from(result.startPositions)).toEqual([1, 2, 3]);
  });

  it('EMPTY slice of an RGBA source keeps startAlphas/endAlphas DEFINED (presence follows SOURCE)', () => {
    // The stamp chain's linchpin (volumetric phase 4): 0 visible
    // segments returns empty alpha arrays from the worker, and
    // `toProcessedLines` gates alpha presence on the SOURCE layout
    // (`colorComponents === 4`) — NOT on `length > 0` — so the
    // geometry's `hasElementAlpha` stamp survives an empty slice
    // exactly like the scalars presence fix this mirrors. A future
    // edit gating on length would resurrect the empty-slice flip.
    const projectLinesTo3D = vi.fn(async () => ({
      startPositions: new Float32Array(),
      endPositions: new Float32Array(),
      startColors: new Float32Array(),
      endColors: new Float32Array(),
      startWidths: new Float32Array(),
      endWidths: new Float32Array(),
      startSharpness: new Float32Array(),
      endSharpness: new Float32Array(),
      startAlphas: new Float32Array(),
      endAlphas: new Float32Array(),
      segmentLengths: new Float32Array(),
      startJointCode: new Float32Array(),
      endJointCode: new Float32Array(),
      visibleSegmentCount: 0,
    }));
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectLinesTo3D })),
    });
    const rgbaData: LoadedLinesData = {
      ...makeData(4),
      colors: new Float32Array(4 * 2 * 4),
      colorComponents: 4,
    };
    return projectLinesTo3DUsingWorker(
      rgbaData,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    ).then((result) => {
      expect(result.segmentCount).toBe(0);
      expect(result.startAlphas).toBeDefined();
      expect(result.endAlphas).toBeDefined();
    });
  });

  const callProjectLines = () =>
    projectLinesTo3DUsingWorker(
      makeData(),
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    );

  // Worker UNAVAILABILITY — the pool never got the work to a worker at all, so
  // the in-process dispatcher is the only executor left.
  it('falls back to the in-process dispatcher on WorkerUnavailableError', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new WorkerUnavailableError('[WorkerPool] No workers available after initialization');
      }),
    });
    mockBuildInstanceBuffers.mockReturnValue(makeDispatcherLinesResult(3));

    const result = await callProjectLines();
    expect(result.segmentCount).toBe(3);
    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
  });

  // A timeout does NOT establish infrastructure failure: a hung kernel or a
  // genuinely-slow projection re-run on the main thread blocks the frame at
  // least as long again. Mirrors data-processor-gsplats.
  it('propagates a worker timeout instead of re-running the projection on the main thread', async () => {
    const timeout = new WorkerTimeoutError('projectLinesTo3D', 60000);
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw timeout;
      }),
    });

    await expect(callProjectLines()).rejects.toBe(timeout);
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
  });

  // Mirrors data-processor-gsplats: the in-process dispatcher runs the SAME
  // kernel, so a rejection that came back FROM the worker must propagate rather
  // than reproduce the fault on the UI thread.
  it('re-throws a kernel fault instead of re-running it on the main thread', async () => {
    const trap = new Error('unreachable');
    trap.name = 'RuntimeError';
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw trap;
      }),
    });

    await expect(callProjectLines()).rejects.toBe(trap);
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
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

    await expect(callProjectLines()).rejects.toBe(spoof);
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
  });

  // Fail closed on an unrecognized error.
  it('re-throws an unrecognized error rather than assuming infrastructure failure', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new Error('boom');
      }),
    });

    await expect(callProjectLines()).rejects.toThrow('boom');
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
  });

  it('forwards per-vertex scalars along the WORKER path (no main-thread fallback)', async () => {
    // This test previously claimed "scalars force main-thread fallback" and
    // asserted buildInstanceBuffers ran. That contract no longer exists —
    // scalars ride the worker payload and are interpolated there
    // (`interpolate_scalars_batch`, see the note above `buildLinesParams`).
    // It only passed because it never mocked the pool: `getWorkerPool()`
    // returned undefined, the TypeError hit the catch, and the old blind
    // fallback swallowed it into the in-process dispatcher. Now that only
    // infrastructure failures fall back, that accident is gone, so the test
    // asserts what the code actually does.
    const dataWithScalars: LoadedLinesData = {
      ...makeData(2000),
      scalars: new Float32Array(2 * 2000), // 2 vertices per segment
    };
    const projectLinesTo3D = vi.fn(async (_params: { scalars?: Float32Array | null }) =>
      makeDispatcherLinesResult(2000)
    );
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async (_op, _kind, fn) => fn({ projectLinesTo3D })),
    });

    await projectLinesTo3DUsingWorker(
      dataWithScalars,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    );

    expect(projectLinesTo3D).toHaveBeenCalledTimes(1);
    expect(projectLinesTo3D.mock.calls[0][0]).toMatchObject({
      scalars: dataWithScalars.scalars,
    });
    // The whole point: the main-thread dispatcher is NOT involved.
    expect(mockBuildInstanceBuffers).not.toHaveBeenCalled();
  });
});
