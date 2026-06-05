/**
 * Unit tests for the gsplats data-processor concern.
 *
 * Strategy mirrors data-processor-lines.test.ts:
 *   - Real `THREE.Group` + `THREE.Mesh` for `rootGroup` so the
 *     getObjectByName + nodeType guard is exercised in full.
 *   - Mock `projectGSplats` and `packCholeskyForShader` so tests aren't
 *     coupled to the real Mahalanobis math; we only verify orchestration.
 *   - Mock the worker pool so we can control success / failure /
 *     fallback paths.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const mockProcessGSplats = vi.fn();
vi.mock('../../../../data/gsplats/projection', () => ({
  projectGSplats: (...args: unknown[]) => mockProcessGSplats(...args),
}));

const mockPackCholesky = vi.fn();
const mockUpdateInstancedMesh = vi.fn();
vi.mock('../../../../rendering/gsplat-geometry', () => ({
  packCholeskyForShader: (...args: unknown[]) => mockPackCholesky(...args),
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

function makeProcessed(splatCount = 2) {
  return {
    centers3D: new Float32Array(splatCount * 3),
    choleskyFactors3D: new Float32Array(splatCount * 6),
    amplitudes: new Float32Array(splatCount),
    colors: new Float32Array(splatCount * 3),
    splatCount,
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
  mockProcessGSplats.mockReset().mockImplementation(() => makeProcessed());
  mockPackCholesky.mockReset().mockReturnValue({
    cholesky01: new Float32Array(),
    cholesky23: new Float32Array(),
    cholesky45: new Float32Array(),
  });
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

  it('runs main thread for small datasets (splatCount <= 1000)', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const result = await processGSplatsData('/g', makeData(500), makeViewState(), root, 1);
    expect(result).not.toBeNull();
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
    expect(mockGetWorkerPool).not.toHaveBeenCalled();
  });

  it('runs main thread for 3D data even when splatCount > 1000', async () => {
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
    { splatCount: 1000, ndim: 4, expectWorker: false, label: '[1000, 4] → main thread' },
    { splatCount: 1001, ndim: 4, expectWorker: true, label: '[1001, 4] → worker' },
    { splatCount: 2000, ndim: 3, expectWorker: false, label: '[2000, 3] → main thread' },
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

  it('packs cholesky factors after projection', async () => {
    const root = new THREE.Group();
    root.add(makeMesh('/g'));
    const result = await processGSplatsData('/g', makeData(50), makeViewState(), root, 1);
    expect(result).not.toBeNull();
    expect(mockPackCholesky).toHaveBeenCalledTimes(1);
    expect(result?.cholesky01).toBeDefined();
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

  it('falls back to main thread on worker failure', async () => {
    mockGetWorkerPool.mockReturnValue({
      runWithTimeout: vi.fn(async () => {
        throw new Error('boom');
      }),
    });
    mockProcessGSplats.mockReturnValue(makeProcessed(7));

    const result = await projectGSplatsTo3DUsingWorker(makeData(), makeViewState(), 3.0, 1);
    expect(result.splatCount).toBe(7);
    expect(mockProcessGSplats).toHaveBeenCalledTimes(1);
  });
});
