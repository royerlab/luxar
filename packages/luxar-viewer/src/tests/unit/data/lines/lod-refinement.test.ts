/**
 * Direct tests for runLinesRefinement — the progressive Lines LOD
 * refinement loop. Mirrors `data/gsplats/lod-refinement.test.ts` and
 * `data/points/lod-refinement.test.ts`.
 *
 * Pins the unit semantics: cancellation hand-off when viewStateQueue
 * has pending state, normal completion when no loaders have more LODs,
 * error-handling that doesn't terminate the loop, skip-path handling,
 * and the process+commit two-step that distinguishes Lines from Points.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { runLinesRefinement } from '../../../../data/lines/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { LinesDataLoader } from '../../../../types/lines';
import type { ViewState } from '../../../../data/data-loader-types';
import type { StagedLinesCommit } from '../../../../data/scene-loader/process/data-processor-lines';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

function makeLoader(stages: Array<{ hasMoreLODs: boolean }>): LinesDataLoader {
  let i = 0;
  const loader: Partial<LinesDataLoader> & { hasMoreLODs?: boolean } = {
    get hasMoreLODs() {
      const stage = stages[Math.min(i, stages.length - 1)];
      return stage.hasMoreLODs;
    },
    updateView: vi.fn().mockImplementation(async () => {
      i++;
      return null;
    }),
  };
  return loader as LinesDataLoader;
}

describe('runLinesRefinement', () => {
  it('returns immediately and releases lock when no loaders have more LODs', async () => {
    const loader = makeLoader([{ hasMoreLODs: false }]);
    const linesLoaders = new Map([['/l', loader]]);
    const releaseLock = vi.fn();
    const updateVisibleCountsInMonitor = vi.fn();

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      linesLoaders,
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processLines: vi.fn(),
      commitLines: vi.fn(),
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

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      linesLoaders: new Map([['/l', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processLines: vi.fn(),
      commitLines: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock,
      retriggerUpdate,
    });

    expect(retriggerUpdate).toHaveBeenCalledTimes(1);
    expect(retriggerUpdate).toHaveBeenCalledWith({ slicePosition: [1, 2, 3, 4] });
    expect(releaseLock).not.toHaveBeenCalled();
  });

  it('catches per-loader errors so other loaders continue refining', async () => {
    const loaderA: LinesDataLoader = {
      updateView: vi.fn().mockRejectedValue(new Error('synthetic')),
    } as unknown as LinesDataLoader;
    Object.defineProperty(loaderA, 'hasMoreLODs', {
      get: vi
        .fn()
        .mockReturnValueOnce(true)
        .mockReturnValue(false),
    });

    await expect(
      runLinesRefinement({
        rootGroup: new THREE.Group(),
        viewStateQueue: new ViewStateQueue(),
        linesLoaders: new Map([['/a', loaderA]]),
        deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
        processLines: vi.fn(),
        commitLines: vi.fn(),
        updateVisibleCountsInMonitor: vi.fn(),
        releaseLock: vi.fn(),
        retriggerUpdate: vi.fn(),
      })
    ).resolves.not.toThrow();
  });

  it('skips loaders whose derived view-state is fully-extended', async () => {
    let calls = 0;
    const loader: LinesDataLoader = {
      get hasMoreLODs() {
        calls += 1;
        return calls === 1;
      },
      updateView: vi.fn(),
    } as unknown as LinesDataLoader;
    const processLines = vi.fn();

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      linesLoaders: new Map([['/l', loader]]),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
      processLines,
      commitLines: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(loader.updateView).not.toHaveBeenCalled();
    expect(processLines).not.toHaveBeenCalled();
  });

  it('skips loaders that do not expose hasMoreLODs (non-progressive loaders)', async () => {
    const singleShot: LinesDataLoader = {
      updateView: vi.fn(),
    } as unknown as LinesDataLoader;

    const processLines = vi.fn();

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      linesLoaders: new Map([['/l', singleShot]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processLines,
      commitLines: vi.fn(),
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(singleShot.updateView).not.toHaveBeenCalled();
    expect(processLines).not.toHaveBeenCalled();
  });

  it('processes and commits via processLines + commitLines on successful refinement', async () => {
    let hasMore = true;
    const refinedData = {
      positions: new Float32Array(6),
      segments: new Uint32Array([0, 1]),
      widths: new Float32Array([1, 1]),
      colors: null,
      sharpness: null,
      segmentCount: 1,
      vertexCount: 2,
      ndim: 3,
    };
    const loader: LinesDataLoader = {
      get hasMoreLODs() {
        return hasMore;
      },
      updateView: vi.fn().mockImplementation(async () => {
        hasMore = false;
        return refinedData;
      }),
    } as unknown as LinesDataLoader;

    const staged = { foo: 'commit' } as unknown as StagedLinesCommit;
    const processLines = vi.fn().mockResolvedValue(staged);
    const commitLines = vi.fn();

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      linesLoaders: new Map([['/l', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processLines,
      commitLines,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(processLines).toHaveBeenCalledTimes(1);
    expect(processLines).toHaveBeenCalledWith('/l', refinedData, expect.any(Object));
    expect(commitLines).toHaveBeenCalledTimes(1);
    expect(commitLines).toHaveBeenCalledWith(staged);
  });

  it('skips commitLines when processLines returns null', async () => {
    // processLines returns null when nothing needs committing (e.g. empty
    // intersection); the loop must not call commitLines with null.
    let hasMore = true;
    const refinedData = {
      positions: new Float32Array(0),
      segments: new Uint32Array(0),
      widths: new Float32Array(0),
      colors: null,
      sharpness: null,
      segmentCount: 0,
      vertexCount: 0,
      ndim: 3,
    };
    const loader: LinesDataLoader = {
      get hasMoreLODs() {
        return hasMore;
      },
      updateView: vi.fn().mockImplementation(async () => {
        hasMore = false;
        return refinedData;
      }),
    } as unknown as LinesDataLoader;

    const processLines = vi.fn().mockResolvedValue(null);
    const commitLines = vi.fn();

    await runLinesRefinement({
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      linesLoaders: new Map([['/l', loader]]),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
      processLines,
      commitLines,
      updateVisibleCountsInMonitor: vi.fn(),
      releaseLock: vi.fn(),
      retriggerUpdate: vi.fn(),
    });

    expect(processLines).toHaveBeenCalledTimes(1);
    expect(commitLines).not.toHaveBeenCalled();
  });
});
