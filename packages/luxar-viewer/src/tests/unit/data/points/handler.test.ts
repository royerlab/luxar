/**
 * Smoke tests for the Points handler — mirrors
 * data/lines/handler.test.ts and data/gsplats/handler.test.ts so all
 * geometry handlers share the same basic contract.
 *
 * End-to-end exercise of load + stage runs through the SceneLoader
 * suite; these tests just pin the kind / label discriminants and the
 * basic skip-path behaviour in isolation.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/points/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { DataLoader, ViewState } from '../../../../data/data-loader-types';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

const baseViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1],
};

describe('points handler', () => {
  it('discriminates as kind="points" with label="Points"', () => {
    expect(kind).toBe('points');
    expect(label).toBe('Points');
  });

  it('returns null on derived.skip without calling the loader', async () => {
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    };
    const queue = new ViewStateQueue();
    const result = await loadAndStage('/p', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: 'extend_to_all', viewState: baseViewState }),
    });
    expect(result).toBeNull();
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  it('forgets the path on skip so the next non-skip update re-baselines', async () => {
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    };
    const queue = new ViewStateQueue();
    // Spy on forgetPath so we can pin the actual contract (data.md, C3 fixed:
    // previously the test had no expect() at all — the comment said
    // "Indirect assertion: prev-state map is empty for /p" but never asserted).
    const forgetPathSpy = vi.spyOn(queue, 'forgetPath');
    queue.dispatchPrefetch('/p', baseViewState, loader); // seed prev
    await loadAndStage('/p', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: 'extend_to_all', viewState: baseViewState }),
    });
    // The skip path MUST call forgetPath('/p') exactly once.
    expect(forgetPathSpy).toHaveBeenCalledTimes(1);
    expect(forgetPathSpy).toHaveBeenCalledWith('/p');
    // updateView must NOT have been called (skip path short-circuits before).
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  it('returns the staged commit on a successful load', async () => {
    const fakeData = {
      positions: new Float32Array(),
      colors: new Float32Array(),
      radii: new Float32Array(),
      sharpness: new Float32Array(),
      pointCount: 3,
      ndim: 3,
      metadata: {
        totalPoints: 3,
        loadedPoints: 3,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn().mockResolvedValue(fakeData),
      dispose: vi.fn(),
    };
    const result = await loadAndStage('/p', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 5,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
    });
    expect(result).toEqual({ path: '/p', data: fakeData });
  });

  it('skips predictive prefetch when the update was superseded (signal aborted)', async () => {
    const fakeData = {
      positions: new Float32Array(),
      pointCount: 1,
      ndim: 3,
      metadata: {
        totalPoints: 1,
        loadedPoints: 1,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn().mockResolvedValue(fakeData),
      dispose: vi.fn(),
    };
    const queue = new ViewStateQueue();
    const prefetchSpy = vi.spyOn(queue, 'dispatchPrefetch');
    const controller = new AbortController();
    controller.abort(); // superseded before the stage step finished

    await loadAndStage('/p', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 5,
      extendedToleranceCache: new Map(),
      signal: controller.signal,
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
    });

    // Warming chunks for an abandoned view-state wastes bandwidth and
    // pollutes the per-path prefetch baseline.
    expect(prefetchSpy).not.toHaveBeenCalled();
  });

  it('forwards the per-update abort signal to loader.updateView', async () => {
    const fakeData = {
      positions: new Float32Array(),
      colors: new Float32Array(),
      radii: new Float32Array(),
      sharpness: new Float32Array(),
      pointCount: 0,
      ndim: 3,
      metadata: {
        totalPoints: 0,
        loadedPoints: 0,
        bounds: new THREE.Box3(),
        usedSpatialIndex: false,
      },
    };
    const updateView = vi.fn().mockResolvedValue(fakeData);
    const loader: DataLoader = { loadPoints: vi.fn(), updateView, dispose: vi.fn() };
    const ac = new AbortController();
    await loadAndStage('/p', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 5,
      extendedToleranceCache: new Map(),
      signal: ac.signal,
      deriveNodeViewState: () => ({ skip: false, viewState: baseViewState }),
    });
    expect(updateView).toHaveBeenCalledWith(baseViewState, expect.anything(), ac.signal);
  });
});
