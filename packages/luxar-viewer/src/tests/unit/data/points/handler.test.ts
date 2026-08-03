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

// A fully-extended node's derived view state: the extend-to-all sentinel on the
// (single) non-displayed dim + that dim's slicePosition pinned to 0 — a
// slice-INVARIANT query. It flows through the handler as a normal node (#1157).
const extendedViewState: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1e10],
};

const fakePointsData = {
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

/** Root group containing a Points mesh stamped as already-committed. */
function rootWithMesh(path: string): THREE.Group {
  const root = new THREE.Group();
  const mesh = new THREE.Mesh();
  mesh.name = path;
  mesh.userData = { loadedViewVersion: 1 };
  root.add(mesh);
  return root;
}

describe('points handler', () => {
  it('discriminates as kind="points" with label="Points"', () => {
    expect(kind).toBe('points');
    expect(label).toBe('Points');
  });

  it('loads a fully-extended node with the derived extended+pinned view state on FIRST paint (#1157)', async () => {
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn().mockResolvedValue(fakePointsData),
      dispose: vi.fn(),
    };
    const session = makeSession();
    // A fully-extended node is a normal node — the handler queries the loader
    // with the slice-invariant extended view state, never a skip shortcut.
    const result = await loadAndStage('/p', loader, session, {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(result).toEqual({ path: '/p', data: fakePointsData });
    // No skip path is taken for a fully-extended node under the new design.
    expect(session.markSkipped).not.toHaveBeenCalled();
  });

  it('still queries the loader on a later sweep even with a committed mesh (no skip shortcut)', async () => {
    // Regression guard for the reverted "skip once loaded" design: a
    // fully-extended node with an already-committed, non-progressive mesh must
    // STILL call updateView (the loader's viewStatesEqual no-op handles the
    // cheapness), so playback-budget / abort masking can never freeze it.
    const loader: DataLoader = {
      loadPoints: vi.fn(),
      updateView: vi.fn().mockResolvedValue(fakePointsData),
      dispose: vi.fn(),
    };
    const session = makeSession();
    await loadAndStage('/p', loader, session, {
      rootGroup: rootWithMesh('/p'),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 2,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(session.markSkipped).not.toHaveBeenCalled();
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
