/**
 * Smoke tests for the GSplats handler — mirrors points/handler.test.ts
 * and lines/handler.test.ts so all geometry handlers share the same
 * basic contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';

// Spy-wrapped with the REAL implementation as the default: one test below stages
// a genuine commit through it. Individual tests override the mock to simulate a
// failure that happens AFTER the fetch (the projection step).
vi.mock('../../../../data/scene-loader/process/data-processor-gsplats', async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import('../../../../data/scene-loader/process/data-processor-gsplats')
    >();
  return { ...actual, processGSplatsData: vi.fn(actual.processGSplatsData) };
});

import { processGSplatsData } from '../../../../data/scene-loader/process/data-processor-gsplats';
import { kind, label, loadAndStage } from '../../../../data/gsplats/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state/view-state-queue';
import type { GSplatsDataLoader } from '../../../../types/gsplats';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

// A fully-extended node's derived view state: extend-to-all sentinel on the
// (single) non-displayed dim + that dim's slicePosition pinned to 0 — a
// slice-INVARIANT query flowing through the handler as a normal node (#1157).
const extendedViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 0],
  tolerance: [0, 0, 0, 1e10],
};

/** Root group with a gsplats mesh stamped as already-committed. */
function rootWithGSplatsMesh(path: string): THREE.Group {
  const root = new THREE.Group();
  const mesh = new THREE.Mesh();
  mesh.name = path;
  mesh.userData = { nodeType: 'gsplats', attrs: {}, loadedViewVersion: 1 };
  root.add(mesh);
  return root;
}

describe('gsplats handler', () => {
  it('discriminates as kind="gsplats" with label="GSplats"', () => {
    expect(kind).toBe('gsplats');
    expect(label).toBe('GSplats');
  });

  it('loads a fully-extended node with the derived extended+pinned view state on FIRST paint (#1157)', async () => {
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null), // empty slice: exits before process
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const session = makeSession();
    await loadAndStage('/g', loader, session, {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      // A fully-extended node is a normal node — no skip shortcut.
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledTimes(1);
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(session.markSkipped).not.toHaveBeenCalled();
  });

  it('still queries the loader on a later sweep even with a committed mesh (no skip shortcut)', async () => {
    // Regression guard for the reverted "skip once loaded" design.
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const session = makeSession();
    await loadAndStage('/g', loader, session, {
      rootGroup: rootWithGSplatsMesh('/g'),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 2,
      updateVersion: 2,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState: extendedViewState }),
    });
    expect(loader.updateView).toHaveBeenCalledWith(extendedViewState, expect.anything(), undefined);
    expect(session.markSkipped).not.toHaveBeenCalled();
  });

  it('passes the derived viewState through to loader.updateView on the non-skip path', async () => {
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1],
    };
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    await loadAndStage('/g', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });
    expect(loader.updateView).toHaveBeenCalledTimes(1);
    // 3rd arg is the per-update signal (undefined when none is supplied).
    expect(loader.updateView).toHaveBeenCalledWith(viewState, expect.anything(), undefined);
  });

  it('forwards the per-update abort signal to loader.updateView', async () => {
    const viewState = {
      displayDims: [0, 1, 2],
      slicePosition: [0, 0, 0, 0],
      tolerance: [0, 0, 0, 1],
    };
    const updateView = vi.fn().mockResolvedValue(null);
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView,
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const ac = new AbortController();
    await loadAndStage('/g', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      signal: ac.signal,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });
    expect(updateView).toHaveBeenCalledWith(viewState, expect.anything(), ac.signal);
  });
});

describe('gsplats handler — no-op commit skip', () => {
  function makeGSplatsMesh(name: string): THREE.Mesh {
    const mesh = new THREE.Mesh();
    mesh.name = name;
    mesh.userData = { nodeType: 'gsplats', attrs: {} };
    return mesh;
  }

  const viewState = {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0],
    tolerance: [0, 0, 0],
  };

  const data = {
    positions: new Float32Array(9),
    amplitudes: new Float32Array(3),
    choleskyFactors: new Float32Array(18),
    colors: null,
    splatCount: 3,
    ndim: 3,
  };

  it('returns a noop staged commit when the mesh already committed this exact data reference', async () => {
    const root = new THREE.Group();
    const mesh = makeGSplatsMesh('/g');
    (mesh.userData as { committedData?: unknown }).committedData = data;
    root.add(mesh);

    const loader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(data),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;

    const session = makeSession();
    const staged = await loadAndStage('/g', loader, session, {
      rootGroup: root,
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 2,
      updateVersion: 2,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });

    expect(staged).toEqual({ path: '/g', noop: true, sourceData: data });
    expect(session.setMetadata).toHaveBeenCalledWith(
      expect.objectContaining({ info: 'unchanged' })
    );
  });

  it('noop fast path skips predictive prefetch when the update was superseded', async () => {
    // P8 symmetry with the Points/Lines gating tests, on the dispatch site
    // unique to the gsplats/lines handlers: the stamp-only noop path also
    // fires prefetch, and it too must be gated for a superseded update.
    const root = new THREE.Group();
    const mesh = makeGSplatsMesh('/g');
    (mesh.userData as { committedData?: unknown }).committedData = data;
    root.add(mesh);

    const loader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(data),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const queue = new ViewStateQueue();
    const prefetchSpy = vi.spyOn(queue, 'dispatchPrefetch');
    const controller = new AbortController();
    controller.abort();

    const staged = await loadAndStage('/g', loader, makeSession(), {
      rootGroup: root,
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 2,
      updateVersion: 2,
      extendedToleranceCache: new Map(),
      signal: controller.signal,
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });

    expect(staged).toEqual({ path: '/g', noop: true, sourceData: data }); // still stages the noop
    expect(prefetchSpy).not.toHaveBeenCalled(); // but never prefetches
  });

  it('stages a real commit when the data reference differs from committedData', async () => {
    const root = new THREE.Group();
    const mesh = makeGSplatsMesh('/g');
    (mesh.userData as { committedData?: unknown }).committedData = { other: true };
    root.add(mesh);

    const loader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn().mockResolvedValue(data),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;

    const staged = await loadAndStage('/g', loader, makeSession(), {
      rootGroup: root,
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 2,
      updateVersion: 2,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: false, viewState }),
    });

    expect(staged).not.toBeNull();
    expect(staged?.noop).toBeUndefined();
  });

  // The failure record's scope is the WHOLE loadAndStage step, so it must not be
  // cleared until everything fallible after the fetch has also succeeded.
  // Clearing right after the fetch meant a post-fetch failure re-recorded with
  // retryCount 0 — pinning the log at "(attempt 1)" — and left hasFailures()
  // transiently reporting clean.
  describe('failure-record lifetime', () => {
    const makeLoaderReturning = (value: unknown): GSplatsDataLoader =>
      ({
        loadGSplats: vi.fn(),
        updateView: vi.fn().mockResolvedValue(value),
        dispose: vi.fn(),
      }) as unknown as GSplatsDataLoader;

    const makeCtx = (clearFailure: () => void, root: THREE.Group) => ({
      rootGroup: root,
      viewStateQueue: new ViewStateQueue(),
      clearFailure,
      currentVersion: 2,
      updateVersion: 2,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({
        skip: false as const,
        viewState: { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] },
      }),
    });

    it('does NOT clear the failure record when projection fails after the fetch', async () => {
      vi.mocked(processGSplatsData).mockRejectedValueOnce(new Error('projection boom'));
      const root = new THREE.Group();
      const mesh = makeGSplatsMesh('/g');
      (mesh.userData as { committedData?: unknown }).committedData = { other: true };
      root.add(mesh);
      const clearFailure = vi.fn();

      await expect(
        loadAndStage('/g', makeLoaderReturning(data), makeSession(), makeCtx(clearFailure, root))
      ).rejects.toThrow('projection boom');

      expect(clearFailure).not.toHaveBeenCalled();
    });

    it('clears the failure record once fetch AND projection both succeed', async () => {
      const root = new THREE.Group();
      const mesh = makeGSplatsMesh('/g');
      (mesh.userData as { committedData?: unknown }).committedData = { other: true };
      root.add(mesh);
      const clearFailure = vi.fn();

      await loadAndStage(
        '/g',
        makeLoaderReturning(data),
        makeSession(),
        makeCtx(clearFailure, root)
      );

      expect(clearFailure).toHaveBeenCalledWith('/g');
    });

    it('clears the failure record when the loader legitimately returns no data', async () => {
      // An empty slice is a healthy outcome: a formerly-failing node that now
      // yields nothing must not keep a permanent failure record.
      const clearFailure = vi.fn();

      const result = await loadAndStage(
        '/g',
        makeLoaderReturning(null),
        makeSession(),
        makeCtx(clearFailure, new THREE.Group())
      );

      expect(result).toBeNull();
      expect(clearFailure).toHaveBeenCalledWith('/g');
    });
  });
});
