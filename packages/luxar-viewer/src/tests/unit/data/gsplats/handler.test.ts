/**
 * Smoke tests for the GSplats handler — mirrors points/handler.test.ts
 * and lines/handler.test.ts so all geometry handlers share the same
 * basic contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
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

describe('gsplats handler', () => {
  it('discriminates as kind="gsplats" with label="GSplats"', () => {
    expect(kind).toBe('gsplats');
    expect(label).toBe('GSplats');
  });

  it('returns null on derived.skip without calling the loader', async () => {
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const result = await loadAndStage('/g', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(result).toBeNull();
    expect(loader.updateView).not.toHaveBeenCalled();
  });

  // data.md G2 fix: parallel coverage to points/handler.test.ts.
  it('forgets the path on skip so the next non-skip update re-baselines', async () => {
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const queue = new ViewStateQueue();
    const forgetPathSpy = vi.spyOn(queue, 'forgetPath');
    await loadAndStage('/g', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: queue,
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(forgetPathSpy).toHaveBeenCalledTimes(1);
    expect(forgetPathSpy).toHaveBeenCalledWith('/g');
    expect(loader.updateView).not.toHaveBeenCalled();
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
});
