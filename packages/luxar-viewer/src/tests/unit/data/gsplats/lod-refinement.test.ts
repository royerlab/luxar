/**
 * Direct tests for runGSplatsRefinement — the progressive GSplats LOD
 * refinement loop.
 *
 * End-to-end behaviour (rAF integration with real browser paint timing)
 * is covered by the e2e suite; these tests pin the unit semantics:
 * cancellation hand-off when viewStateQueue has pending state, normal
 * completion when no loaders have more LODs, error-handling that
 * doesn't terminate the loop, and skip-path handling.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { runGSplatsRefinement } from '../../../../data/gsplats/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { GSplatsDataLoader } from '../../../../types/gsplats';
import type { ViewState } from '../../../../data/data-loader-types';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

function makeLoader(stages: Array<{ hasMoreLODs: boolean }>): GSplatsDataLoader {
  let i = 0;
  const loader: Partial<GSplatsDataLoader> & { hasMoreLODs?: boolean } = {
    get hasMoreLODs() {
      const stage = stages[Math.min(i, stages.length - 1)];
      return stage.hasMoreLODs;
    },
    updateView: vi.fn().mockImplementation(async () => {
      i++;
      return null;
    }),
  };
  return loader as GSplatsDataLoader;
}

describe('runGSplatsRefinement', () => {
  it('returns immediately and releases lock when no loaders have more LODs', async () => {
    const loader = makeLoader([{ hasMoreLODs: false }]);
    const gsplatLoaders = new Map([['/g', loader]]);
    const releaseLock = vi.fn();
    const updateVisibleCountsInMonitor = vi.fn();

    await runGSplatsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      gsplatLoaders,
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processGSplats: vi.fn(),
      commitGSplats: vi.fn(),
      updateVisibleCountsInMonitor,
      releaseLock,
      retriggerUpdate: vi.fn(),
    });

    // Loop ran once, observed no more LODs, called releaseLock.
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

    await runGSplatsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      gsplatLoaders: new Map([['/g', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processGSplats: vi.fn(),
      commitGSplats: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate,
    });

    expect(retriggerUpdate).toHaveBeenCalledTimes(1);
    expect(retriggerUpdate).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
    // Lock NOT released — handed off to retriggerUpdate's rAF callback.
    expect(releaseLock).not.toHaveBeenCalled();
  });

  it('catches per-loader errors so other loaders continue refining', async () => {
    const loaderA: GSplatsDataLoader = {
      get hasMoreLODs() {
        return false;
      },
      updateView: vi.fn().mockRejectedValue(new Error('synthetic')),
    } as unknown as GSplatsDataLoader;
    // loaderA marked hasMoreLODs=false so the loop exits after one pass —
    // but during that pass, the error must not propagate out of the loop.
    Object.defineProperty(loaderA, 'hasMoreLODs', {
      get: vi
        .fn()
        .mockReturnValueOnce(true) // first iteration: enter try block
        .mockReturnValue(false), // post-error: loop exits
    });

    await expect(
      runGSplatsRefinement({
        rootGroup: new THREE.Group(),
        viewStateQueue: new ViewStateQueue(),
        gsplatLoaders: new Map([['/a', loaderA]]),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        processGSplats: vi.fn(),
        commitGSplats: vi.fn(),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
      })
    ).resolves.not.toThrow();
  });

  it('skips loaders whose derived view-state is fully-extended', async () => {
    // Loader reports hasMoreLODs=true on first read (enters skip branch in
    // the per-loader inner loop), then false on the post-pass `anyMore`
    // check so the outer while-loop exits. Without this transition the
    // skip-path would loop forever in the test environment.
    let calls = 0;
    const loader: GSplatsDataLoader = {
      get hasMoreLODs() {
        calls += 1;
        return calls === 1;
      },
      updateView: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const processGSplats = vi.fn();

    await runGSplatsRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      gsplatLoaders: new Map([['/g', loader]]),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
      processGSplats,
      commitGSplats: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    // Skip path: loader.updateView never called, processGSplats never called.
    expect(loader.updateView).not.toHaveBeenCalled();
    expect(processGSplats).not.toHaveBeenCalled();
  });
});
