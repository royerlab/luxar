/**
 * Direct tests for runPointsRefinement — the progressive Points LOD
 * refinement loop. Mirrors `data/gsplats/lod-refinement.test.ts`.
 *
 * Pins the unit semantics: cancellation hand-off when viewStateQueue
 * has pending state, normal completion when no loaders have more LODs,
 * error-handling that doesn't terminate the loop, and skip-path
 * handling.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { runPointsRefinement } from '../../../../data/points/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { PointsDataLoader } from '../../../../types/points';
import type { ViewState } from '../../../../data/data-loader-types';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

function makeLoader(stages: Array<{ hasMoreLODs: boolean }>): PointsDataLoader {
  let i = 0;
  const loader: Partial<PointsDataLoader> & { hasMoreLODs?: boolean } = {
    get hasMoreLODs() {
      const stage = stages[Math.min(i, stages.length - 1)];
      return stage.hasMoreLODs;
    },
    updateView: vi.fn().mockImplementation(async () => {
      i++;
      return null;
    }),
  };
  return loader as PointsDataLoader;
}

describe('runPointsRefinement', () => {
  it('returns immediately and releases lock when no loaders have more LODs', async () => {
    const loader = makeLoader([{ hasMoreLODs: false }]);
    const pointsLoaders = new Map([['/p', loader]]);
    const releaseLock = vi.fn();
    const updateVisibleCountsInMonitor = vi.fn();

    await runPointsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      pointsLoaders,
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      updatePointsGeometry: vi.fn(),
      updateVisibleCountsInMonitor,
      releaseLock,
      retriggerUpdate: vi.fn(),
    });

    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect(updateVisibleCountsInMonitor).toHaveBeenCalledTimes(1);
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  it('hands off lock to retriggerUpdate when pending view-state is observed', async () => {
    const queue = new ViewStateQueue();
    queue.setPending({ slicePosition: [1, 2, 3, 4] });

    const releaseLock = vi.fn();
    const retriggerUpdate = vi.fn();
    const loader = makeLoader([{ hasMoreLODs: true }]);

    await runPointsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      pointsLoaders: new Map([['/p', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      updatePointsGeometry: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate,
    });

    expect(retriggerUpdate).toHaveBeenCalledTimes(1);
    expect(retriggerUpdate).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
    expect(releaseLock).not.toHaveBeenCalled();
  });

  it('catches per-loader errors so other loaders continue refining', async () => {
    const loaderA: PointsDataLoader = {
      updateView: vi.fn().mockRejectedValue(new Error('synthetic')),
    } as unknown as PointsDataLoader;
    Object.defineProperty(loaderA, 'hasMoreLODs', {
      get: vi
        .fn()
        .mockReturnValueOnce(true)
        .mockReturnValue(false),
    });

    await expect(
      runPointsRefinement({
        rootGroup: new THREE.Group(),
        viewStateQueue: new ViewStateQueue(),
        pointsLoaders: new Map([['/a', loaderA]]),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        updatePointsGeometry: vi.fn(),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
      })
    ).resolves.not.toThrow();
  });

  it('skips loaders whose derived view-state is fully-extended', async () => {
    let calls = 0;
    const loader: PointsDataLoader = {
      get hasMoreLODs() {
        calls += 1;
        return calls === 1;
      },
      updateView: vi.fn(),
    } as unknown as PointsDataLoader;
    const updatePointsGeometry = vi.fn();

    await runPointsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      pointsLoaders: new Map([['/p', loader]]),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
      updatePointsGeometry,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(loader.updateView).not.toHaveBeenCalled();
    expect(updatePointsGeometry).not.toHaveBeenCalled();
  });

  it('skips loaders that do not expose hasMoreLODs (non-progressive loaders)', async () => {
    // Single-shot PointsSpatialIndexLoader doesn't have `hasMoreLODs`;
    // the refinement loop must skip it via the `progressiveLoader.hasMoreLODs !== true`
    // guard rather than treating absence as "true" and trying to refine.
    const singleShot: PointsDataLoader = {
      // No hasMoreLODs property at all.
      updateView: vi.fn(),
    } as unknown as PointsDataLoader;

    const updatePointsGeometry = vi.fn();

    await runPointsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      pointsLoaders: new Map([['/p', singleShot]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      updatePointsGeometry,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(singleShot.updateView).not.toHaveBeenCalled();
    expect(updatePointsGeometry).not.toHaveBeenCalled();
  });

  it('commits freshly loaded data via updatePointsGeometry on successful refinement', async () => {
    // Loader reports hasMoreLODs=true, returns data, then anyHasMoreLODs=false so loop exits.
    let hasMore = true;
    const refinedData = {
      positions: new Float32Array(9),
      pointCount: 3,
      ndim: 3,
      metadata: {
        totalPoints: 3,
        loadedPoints: 3,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };
    const loader: PointsDataLoader = {
      get hasMoreLODs() {
        return hasMore;
      },
      updateView: vi.fn().mockImplementation(async () => {
        hasMore = false;
        return refinedData;
      }),
    } as unknown as PointsDataLoader;

    const updatePointsGeometry = vi.fn();

    await runPointsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      pointsLoaders: new Map([['/p', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      updatePointsGeometry,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(updatePointsGeometry).toHaveBeenCalledTimes(1);
    expect(updatePointsGeometry).toHaveBeenCalledWith('/p', refinedData);
  });
});
