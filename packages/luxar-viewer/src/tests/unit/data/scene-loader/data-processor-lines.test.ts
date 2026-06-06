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
    startClipped: new Uint8Array(visibleSegmentCount),
    endClipped: new Uint8Array(visibleSegmentCount),
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
      startClipped: new Uint8Array(),
      endClipped: new Uint8Array(),
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
      startClipped: new Uint8Array(),
      endClipped: new Uint8Array(),
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

  it('falls back to the in-process dispatcher on worker failure', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new Error('boom');
      }),
    });
    mockBuildInstanceBuffers.mockReturnValue(makeDispatcherLinesResult(3));

    const result = await projectLinesTo3DUsingWorker(
      makeData(),
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    );
    expect(result.segmentCount).toBe(3);
    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(1);
  });

  it('takes the main-thread fallback path when scalars are present (no throw)', async () => {
    // data.md C1[P2] fix: prior test name "emits a one-shot warning when
    // scalars force main-thread fallback" was misleading — the body
    // explicitly admits the module-scoped warned-flag cannot be reliably
    // reset, so the warning emission is NOT asserted. The behavioural
    // contract the test actually verifies is "scalars → main-thread
    // fallback (buildInstanceBuffers runs) — no throw, no crash."
    // Renamed accordingly; the warning-emission assertion is left as
    // future work (would require a public reset hook on the
    // module-private flag).
    const dataWithScalars: LoadedLinesData = {
      ...makeData(2000),
      scalars: new Float32Array(2 * 2000), // 2 vertices per segment
    };
    mockBuildInstanceBuffers.mockReturnValue(makeDispatcherLinesResult(2000));

    await projectLinesTo3DUsingWorker(
      dataWithScalars,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      1
    );
    await projectLinesTo3DUsingWorker(
      dataWithScalars,
      { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      [1, 1, 1],
      2
    );
    // Main-thread fallback runs once per call → 2 invocations total.
    expect(mockBuildInstanceBuffers).toHaveBeenCalledTimes(2);
  });
});
