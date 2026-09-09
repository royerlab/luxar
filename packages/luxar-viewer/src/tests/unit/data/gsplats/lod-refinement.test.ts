/**
 * Direct tests for runGSplatsRefinement — the progressive GSplats LOD
 * refinement loop.
 *
 * End-to-end behaviour (rAF integration with real browser paint timing) is
 * covered by the e2e suite. The unit loop semantics — completion, lock
 * release, cancellation hand-off, error isolation, skip-path — are shared
 * across all three geometry types and live in
 * `../_shared/refinement-loop-contract`. GSplats has no behaviour beyond that
 * contract, so this file just binds the contract to `runGSplatsRefinement`.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import {
  runGSplatsRefinement,
  type GSplatsRefinementCtx,
} from '../../../../data/gsplats/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { GSplatsDataLoader } from '../../../../types/gsplats';
import type { ViewState } from '../../../../data/data-loader-types';
import { defineRefinementLoopContract } from '../_shared/refinement-loop-contract';

defineRefinementLoopContract('runGSplatsRefinement', 'GSplats', 'showing reduced detail', (w) =>
  runGSplatsRefinement({
    rootGroup: w.rootGroup ?? new THREE.Group(),
    objects: w.objects,
    viewStateQueue: w.viewStateQueue,
    loaders: w.loaders as GSplatsRefinementCtx['loaders'],
    deriveNodeViewState: w.deriveNodeViewState as GSplatsRefinementCtx['deriveNodeViewState'],
    processGSplats: w.processSpy as GSplatsRefinementCtx['processGSplats'],
    commitGSplats: vi.fn(),
    updateVisibleCountsInMonitor: w.updateVisibleCountsInMonitor,
    releaseLock: w.releaseLock,
    retriggerUpdate: w.retriggerUpdate,
    signal: w.signal,
    residencyBudget: w.residencyBudget,
  })
);

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

describe('runGSplatsRefinement — abort-gated commit', () => {
  it('does NOT commit when the signal aborts during the async process step', async () => {
    // The core failure mode: loader.updateView resolves cleanly (no throw), so
    // the AbortError catch never fires; the supersede then aborts the run's
    // controller WHILE the async processGSplats round-trip is in flight. The
    // commit must be skipped so no stale-slice geometry reaches the GPU —
    // mirroring runAtomicCommit's signal.aborted guard on the main path.
    const controller = new AbortController();
    let hasMore = true;
    const loader = {
      get hasMoreLODs() {
        return hasMore;
      },
      updateView: vi.fn().mockImplementation(async () => {
        hasMore = false;
        return { splatCount: 3 };
      }),
    } as unknown as GSplatsDataLoader;

    const processGSplats = vi.fn().mockImplementation(async () => {
      // A newer view-state superseded us mid-process.
      controller.abort();
      return { path: '/g' };
    });
    const commitGSplats = vi.fn();

    await runGSplatsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      loaders: new Map([['/g', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processGSplats: processGSplats as unknown as GSplatsRefinementCtx['processGSplats'],
      commitGSplats: commitGSplats as unknown as GSplatsRefinementCtx['commitGSplats'],
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
      signal: controller.signal,
    });

    expect(processGSplats).toHaveBeenCalledTimes(1);
    expect(commitGSplats).not.toHaveBeenCalled();
  });
});
