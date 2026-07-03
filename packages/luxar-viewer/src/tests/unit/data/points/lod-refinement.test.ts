/**
 * Direct tests for runPointsRefinement — the progressive Points LOD
 * refinement loop.
 *
 * The shared loop semantics (completion, lock release, cancellation hand-off,
 * error isolation, skip-path) live in `../_shared/refinement-loop-contract`
 * and are bound to `runPointsRefinement` below. Only the Points-specific
 * behaviour — non-progressive (single-shot) loader handling and the direct
 * `updatePointsGeometry` commit — is tested inline here.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  runPointsRefinement,
  type PointsRefinementCtx,
} from '../../../../data/points/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { PointsDataLoader } from '../../../../types/points';
import type { ViewState } from '../../../../data/data-loader-types';
import { defineRefinementLoopContract } from '../_shared/refinement-loop-contract';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

defineRefinementLoopContract('runPointsRefinement', (w) =>
  runPointsRefinement({
    rootGroup: new THREE.Group(),
    viewStateQueue: w.viewStateQueue,
    pointsLoaders: w.loaders as PointsRefinementCtx['pointsLoaders'],
    deriveNodeViewState: w.deriveNodeViewState as PointsRefinementCtx['deriveNodeViewState'],
    updatePointsGeometry: w.processSpy as PointsRefinementCtx['updatePointsGeometry'],
    updateVisibleCountsInMonitor: w.updateVisibleCountsInMonitor,
    releaseLock: w.releaseLock,
    retriggerUpdate: w.retriggerUpdate,
  })
);

describe('runPointsRefinement — Points-specific behaviour', () => {
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
    // Third arg is the profiler pass session — undefined when no profiler is wired.
    expect(updatePointsGeometry).toHaveBeenCalledWith('/p', refinedData, undefined);
  });
});
