/**
 * Direct tests for runMeshRefinement — the progressive Mesh LOD
 * refinement loop.
 *
 * The shared loop semantics live in `../_shared/refinement-loop-contract`
 * and are bound to `runMeshRefinement` below. The Mesh-specific no-progress
 * warning is tested inline here.
 */

import { describe, expect, it, vi } from 'vitest';
import * as THREE from 'three';
import { runMeshRefinement, type MeshRefinementCtx } from '../../../../data/mesh/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { MeshDataLoader } from '../../../../types/mesh';
import type { ViewState } from '../../../../data/data-loader-types';
import { log, Modules } from '../../../../utils/log';
import { defineRefinementLoopContract } from '../_shared/refinement-loop-contract';

const viewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

defineRefinementLoopContract('runMeshRefinement', 'Mesh', (w) =>
  runMeshRefinement({
    rootGroup: new THREE.Group(),
    viewStateQueue: w.viewStateQueue,
    meshLoaders: w.loaders as MeshRefinementCtx['meshLoaders'],
    deriveNodeViewState: w.deriveNodeViewState as MeshRefinementCtx['deriveNodeViewState'],
    processMesh: w.processSpy as MeshRefinementCtx['processMesh'],
    commitMesh: vi.fn(),
    updateVisibleCountsInMonitor: w.updateVisibleCountsInMonitor,
    releaseLock: w.releaseLock,
    retriggerUpdate: w.retriggerUpdate,
    signal: w.signal,
  })
);

describe('runMeshRefinement — Mesh-specific behaviour', () => {
  it('stops and logs when a successful mesh refinement pass advances no rung', async () => {
    const loader = {
      hasMoreLODs: true,
      loadedLODCount: 2,
      totalLODCount: 4,
      updateView: vi.fn().mockResolvedValue(null),
    } as unknown as MeshDataLoader;
    const releaseLock = vi.fn();
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});

    await runMeshRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      meshLoaders: new Map([['/mesh', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState }),
      processMesh: vi.fn(),
      commitMesh: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate: vi.fn(),
    });

    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(releaseLock).toHaveBeenCalledTimes(1);
    expect(warning).toHaveBeenCalledWith(
      Modules.SCENE_LOADER,
      'Mesh refinement stopped for /mesh: no progress at LOD 2/4'
    );
  });
});
