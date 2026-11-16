// @vitest-environment jsdom
/**
 * `SceneLoader.releasePartitionPart` (#3332): what a partition part that left
 * the committed slice gives back — every sweep-registered loader AT or UNDER
 * the part path (never a sibling sharing a name prefix) returns its GPU
 * geometry, drops its decoded ladder, and loses its `committedData` stamp so
 * the commit on re-entry is a full write — and the refinement residency seed
 * no longer charges loaders outside the committed slice.
 *
 * Harness: the mocked-zarrita `SceneLoader` scaffolding of
 * `scene-graph-lookups.test.ts`, with loaders and placeholders attached by hand.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import * as THREE from 'three';
import * as zarr from 'zarrita';
import { SceneLoader } from '../../../../data';
import { hasCommittedData, setCommittedData } from '../../../../types/committed-data';
import type { LadderResidency } from '../../../../data/scene-loader/progressive/residency-budget';
import type { ViewState } from '../../../../data/data-loader-types';

vi.mock('zarrita', () => ({
  FetchStore: vi.fn(),
  withMaybeConsolidatedMetadata: vi.fn(),
  registry: {},
  root: vi.fn(),
  open: (() => {
    const openMock = vi.fn();
    return Object.assign(openMock, { v2: openMock, v3: openMock });
  })(),
  NotFoundError: class NotFoundError extends Error {},
  InvalidMetadataError: class InvalidMetadataError extends Error {},
  get: vi.fn(),
  slice: vi.fn((start, end) => ({ start, end })),
}));

vi.mock('../../../../rendering/material-manager', () => ({
  materialManager: {
    getMeshMaterial: vi.fn(() => ({
      uniforms: {},
      userData: {},
      defines: {},
      side: 0,
      needsUpdate: false,
      updateCameraParams: vi.fn(),
      updateShading: vi.fn(),
      updateColormapTexture: vi.fn(),
      updateScalarRange: vi.fn(),
      updateIntensity: vi.fn(),
      updateOffset: vi.fn(),
    })),
  },
  SOFT_DISPOSE_FLAG: Symbol.for('luxar.material.softDispose.test-mock'),
}));

vi.mock('../../../../utils/cross-layer/notifier', () => ({
  notifier: {
    toast: vi.fn(),
    error: vi.fn(),
    showHelp: vi.fn(),
    hideHelp: vi.fn(),
    showLoading: vi.fn(),
    hideLoading: vi.fn(),
    clearError: vi.fn(),
    showSceneIdentityBanner: vi.fn(),
    hideSceneIdentityBanner: vi.fn(),
  },
}));

interface Internals {
  rootGroup: THREE.Group;
  registry: { register(kind: string, path: string, loader: unknown): void };
  _committedViewState: ViewState | null;
  progressiveLadderResidencies(): Map<string, LadderResidency>;
}

const RESIDENCY: LadderResidency = {
  residentBytes: 1000,
  loadedRungs: 2,
  elementCount: 10,
  bytesPerElement: 4,
};

function fakeLoader() {
  return {
    updateView: vi.fn(),
    dispose: vi.fn(),
    releaseResidentLadder: vi.fn(),
    ladderResidency: vi.fn(() => RESIDENCY),
  };
}

/** Attach a placeholder and a sweep loader for `path`, stamped as committed. */
function attach(sceneLoader: SceneLoader, kind: string, path: string) {
  const internals = sceneLoader as unknown as Internals;
  const object = new THREE.Mesh(new THREE.BufferGeometry());
  object.name = path;
  internals.rootGroup.add(object);
  setCommittedData(object, {});
  const loader = fakeLoader();
  internals.registry.register(kind, path, loader);
  return { object, loader };
}

describe('SceneLoader.releasePartitionPart (#3332)', () => {
  let sceneLoader: SceneLoader;

  beforeEach(() => {
    vi.clearAllMocks();
    const mockStore = { contents: vi.fn().mockResolvedValue([{ path: '/', kind: 'group' }]) };
    const mockRootLoc = {
      resolve: vi.fn().mockImplementation(() => ({
        resolve: vi.fn().mockImplementation(() => ({ resolve: vi.fn() })),
      })),
    };
    const mockZarrGroup = {
      attrs: {
        scene_dimensions: {
          dimensions: [
            { name: 'x', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'y', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'z', unit: 'um', range: [0, 100], display: true, step: 1 },
            { name: 'w', unit: 'um', range: [0, 100], display: false, step: 1 },
          ],
        },
      },
    };
    vi.mocked(zarr.FetchStore).mockImplementation(() => mockStore as unknown as zarr.FetchStore);
    vi.mocked(zarr.withMaybeConsolidatedMetadata).mockResolvedValue(mockStore as never);
    vi.mocked(zarr.root).mockReturnValue(mockRootLoc as never);
    vi.mocked(zarr.open).mockResolvedValue(mockZarrGroup as never);
    sceneLoader = new SceneLoader();
  });

  afterEach(async () => {
    await sceneLoader.dispose();
    vi.restoreAllMocks();
  });

  it('releases every loader at or under the part, of every kind, and nothing else', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    const pool = sceneLoader.gpuBufferPool;
    expect(pool).not.toBeNull();
    const releaseGSplats = vi.spyOn(pool!, 'releaseGSplatsGeometry');
    const releasePoints = vi.spyOn(pool!, 'releasePointsGeometry');
    const releaseLines = vi.spyOn(pool!, 'releaseLinesGeometry');

    const splats = attach(sceneLoader, 'gsplats', '/p/part_3');
    const points = attach(sceneLoader, 'points', '/p/part_3/nested/leaf');
    const lines = attach(sceneLoader, 'lines', '/p/part_3/lines');
    const sibling = attach(sceneLoader, 'gsplats', '/p/part_30');

    expect(sceneLoader.releasePartitionPart('/p/part_3')).toBe(true);

    expect(releaseGSplats).toHaveBeenCalledWith('/p/part_3');
    expect(releaseGSplats).not.toHaveBeenCalledWith('/p/part_30');
    expect(releasePoints).toHaveBeenCalledWith('/p/part_3/nested/leaf');
    expect(releaseLines).toHaveBeenCalledWith('/p/part_3/lines');
    for (const released of [splats, points, lines]) {
      expect(released.loader.releaseResidentLadder).toHaveBeenCalledTimes(1);
      // The re-entry commit must full-write, not take the identity no-op.
      expect(hasCommittedData(released.object)).toBe(false);
    }
    expect(sibling.loader.releaseResidentLadder).not.toHaveBeenCalled();
    expect(hasCommittedData(sibling.object)).toBe(true);
  });

  it('reports false when nothing is under the part, and never matches a pathless part', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    const leaf = attach(sceneLoader, 'gsplats', '/p/part_0');
    expect(sceneLoader.releasePartitionPart('/p/part_9')).toBe(false);
    expect(sceneLoader.releasePartitionPart('')).toBe(false);
    expect(leaf.loader.releaseResidentLadder).not.toHaveBeenCalled();
  });

  it('the refinement residency seed skips loaders outside the committed slice', async () => {
    await sceneLoader.loadScene('http://localhost:8000/test.zarr');
    attach(sceneLoader, 'gsplats', '/p/part_0');
    attach(sceneLoader, 'gsplats', '/p/part_1');
    const internals = sceneLoader as unknown as Internals;
    const committed: ViewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 0],
    };
    internals._committedViewState = committed;
    const inSlice = vi.fn((path: string, _view: unknown) => path === '/p/part_0');
    Object.defineProperty(sceneLoader, 'lodGroupRegistry', {
      // `clear` because the harness's dispose() clears the registry.
      value: { isPathInPartitionSlice: inSlice, clear: vi.fn() },
    });

    const seeded = internals.progressiveLadderResidencies();
    expect([...seeded.keys()]).toEqual(['/p/part_0']);
    expect(inSlice).toHaveBeenCalledWith('/p/part_1', committed);
  });
});
