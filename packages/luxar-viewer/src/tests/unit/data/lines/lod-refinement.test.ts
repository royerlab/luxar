/**
 * Direct tests for runLinesRefinement — the progressive Lines LOD
 * refinement loop.
 *
 * The shared loop semantics (completion, lock release, cancellation hand-off,
 * error isolation, skip-path) live in `../_shared/refinement-loop-contract`
 * and are bound to `runLinesRefinement` below. Only the Lines-specific
 * behaviour — non-progressive (single-shot) loader handling and the
 * process+commit two-step that distinguishes Lines from Points — is tested
 * inline here.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { runLinesRefinement, type LinesRefinementCtx } from '../../../../data/lines/lod-refinement';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { LinesDataLoader } from '../../../../types/lines';
import type { ViewState } from '../../../../data/data-loader-types';
import type { StagedLinesCommit } from '../../../../data/scene-loader/process/data-processor-lines';
import { defineRefinementLoopContract } from '../_shared/refinement-loop-contract';

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

defineRefinementLoopContract('runLinesRefinement', (w) =>
  runLinesRefinement({
    rootGroup: new THREE.Group(),
    viewStateQueue: w.viewStateQueue,
    linesLoaders: w.loaders as LinesRefinementCtx['linesLoaders'],
    deriveNodeViewState: w.deriveNodeViewState as LinesRefinementCtx['deriveNodeViewState'],
    processLines: w.processSpy as LinesRefinementCtx['processLines'],
    commitLines: vi.fn(),
    updateVisibleCountsInMonitor: w.updateVisibleCountsInMonitor,
    releaseLock: w.releaseLock,
    retriggerUpdate: w.retriggerUpdate,
  })
);

describe('runLinesRefinement — Lines-specific behaviour', () => {
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
