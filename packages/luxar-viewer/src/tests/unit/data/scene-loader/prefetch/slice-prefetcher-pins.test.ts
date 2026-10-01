/**
 * SlicePrefetcher ↔ SliceCache handoff contracts (real SliceCache):
 *
 *   - A4c pin leak: every entry a shadow pass pinned (its final store AND its
 *     departure store of the previous target) is unpinned by releaseShadows()
 *     (playback stop) and dispose(); the `scache.pinnedEntries` gauge returns
 *     to 0. Measured before the fix: 513 entries / 490 MB (29% of the S-cache)
 *     stayed pinned after h2afva playback stopped.
 *   - A4b adoption: while a shadow pass for key K is running, the S-cache
 *     advertises an in-flight store for K (so a foreground pass for K can wait
 *     for it instead of re-assembling the same slice); it settles when the
 *     shadow pass ends, including on failure.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';
import type { SceneNode, ViewState, DataLoader } from '../../../../../data/data-loader-types';
import { SliceCache } from '../../../../../cache/slice-cache';
import {
  awaitShadowStore,
  storeLadder,
} from '../../../../../data/loaders/progressive/slice-cache-helper';
import { perfCounters } from '../../../../../profiling/perf-counters';

type ShadowImpl = (path: string, vs: ViewState) => Promise<unknown>;
const shadowLoaders = new Map<string, ReturnType<typeof makeShadowLoader>>();
let shadowImpl: ShadowImpl = () => Promise.resolve({});

function makeShadowLoader(path: string) {
  return {
    path,
    updateView: vi.fn((vs: ViewState) => shadowImpl(path, vs)),
    dispose: vi.fn(),
  };
}

vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => {
  const build = (node: SceneNode) => {
    const loader = makeShadowLoader(node.path);
    shadowLoaders.set(node.path, loader);
    return loader;
  };
  return {
    createPointsLoader: vi.fn(build),
    createLinesLoader: vi.fn(build),
    createGSplatsLoader: vi.fn(build),
    createProgressivePointsLoader: vi.fn((n: SceneNode) => Promise.resolve(build(n))),
    createProgressiveLinesLoader: vi.fn((n: SceneNode) => Promise.resolve(build(n))),
    createProgressiveGSplatsLoader: vi.fn((n: SceneNode) => Promise.resolve(build(n))),
    createMeshLoader: vi.fn(),
    createProgressiveMeshLoader: vi.fn(),
  };
});

vi.mock('../../../../../data/zarr', () => ({
  root: vi.fn(() => ({ resolve: vi.fn() })),
}));

import { SlicePrefetcher } from '../../../../../data/scene-loader/prefetch/slice-prefetcher';
import { SceneNodeIndex } from '../../../../../data/scene-loader/view-state/scene-node-index';

function makeNode(path: string): SceneNode {
  return { path, type: 'gsplats', attrs: {}, hasSpatialIndex: true, children: [] };
}

const view: ViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0, 7],
  tolerance: [0, 0, 0, 0.25],
};
const at = (t: number): ViewState => ({ ...view, slicePosition: [0, 0, 0, t] });

function flushAsync(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

const pinnedEntries = (): number => perfCounters.get('scache.pinnedEntries');

/**
 * A shadow that behaves like a progressive shadow loader's store discipline:
 * on a view change it DEPARTURE-stores its previous target (pinned — the
 * incoming pass is a prefetch pass), then stores the new target (pinned).
 */
function storingShadow(sc: SliceCache): ShadowImpl {
  const last = new Map<string, ViewState>();
  return (path, vs) => {
    const prev = last.get(path);
    if (prev) {
      storeLadder(sc, path, prev, [{ a: new Float32Array(8) }, { a: new Float32Array(8) }], {
        pin: vs.prefetch === true,
      });
    }
    storeLadder(sc, path, vs, [{ a: new Float32Array(4) }], { pin: vs.prefetch === true });
    last.set(path, vs);
    return Promise.resolve({});
  };
}

describe('SlicePrefetcher ↔ SliceCache pins and in-flight stores', () => {
  let sc: SliceCache;
  let prefetcher: SlicePrefetcher;

  beforeEach(() => {
    shadowLoaders.clear();
    sc = new SliceCache({ maxSize: 10 * 1024 * 1024 });
    shadowImpl = storingShadow(sc);
    const graph: SceneNode = { ...makeNode('/'), children: [makeNode('/a'), makeNode('/b')] };
    const fg = { updateView: vi.fn() } as unknown as DataLoader;
    const objects = new Map([
      ['/a', new THREE.Group()],
      ['/b', new THREE.Group()],
    ]);
    prefetcher = new SlicePrefetcher({
      getSceneGraph: () => graph,
      getSceneNodeIndex: () => new SceneNodeIndex(graph),
      factoryDeps: () => ({ zarrStore: {}, sliceCache: sc }) as never,
      registry: {
        loaders: new Map(),
        linesLoaders: new Map(),
        gsplatLoaders: new Map([
          ['/a', fg],
          ['/b', fg],
        ]),
      } as never,
      applyEffectiveAttrs: (node) => node.attrs,
      resolveObject: (path) => objects.get(path),
    });
  });

  describe('A4c pin leak', () => {
    it('releaseShadows unpins every entry its shadow passes pinned', async () => {
      prefetcher.prefetch(at(7), 10);
      await flushAsync();
      expect(pinnedEntries()).toBe(2); // one t=7 entry per node

      prefetcher.releaseShadows();
      expect(pinnedEntries()).toBe(0);
    });

    it('releases DEPARTURE pins (the previous target, re-stored deeper) when playback stops', async () => {
      prefetcher.prefetch(at(7), 10);
      await flushAsync();
      prefetcher.prefetch(at(8), 10); // departure store re-pins t=7 deeper
      await flushAsync();
      expect(pinnedEntries()).toBe(4);

      prefetcher.releaseShadows();
      expect(pinnedEntries()).toBe(0);
      // The entries themselves stay cached — only the eviction protection goes.
      expect(sc.getStats().count).toBe(4);
    });

    it('dispose unpins too', async () => {
      prefetcher.prefetch(at(7), 10);
      await flushAsync();
      prefetcher.dispose();
      expect(pinnedEntries()).toBe(0);
    });

    it('does not unpin on a mere stall-guard abort (the t+1 entries are still wanted)', async () => {
      prefetcher.prefetch(at(7), 10);
      await flushAsync();
      prefetcher.abortInFlight();
      expect(pinnedEntries()).toBe(2);
      prefetcher.releaseShadows();
      expect(pinnedEntries()).toBe(0);
    });
  });

  describe('A4b in-flight shadow store', () => {
    it('advertises an in-flight store for the shadow key until the shadow pass ends', async () => {
      const finishers = new Map<string, () => void>();
      shadowImpl = (path) => new Promise<void>((resolve) => finishers.set(path, resolve));
      prefetcher.prefetch(at(7), 10);
      await flushAsync();

      const shadowVs = shadowLoaders.get('/a')!.updateView.mock.calls[0][0];
      const pending = awaitShadowStore(sc, '/a', shadowVs);
      expect(pending).not.toBeNull();
      expect(awaitShadowStore(sc, '/a', at(8))).toBeNull(); // other key: nothing in flight

      let settled = false;
      void pending!.then(() => (settled = true));
      await flushAsync();
      expect(settled).toBe(false);

      finishers.get('/a')!();
      await flushAsync();
      expect(settled).toBe(true);
      expect(awaitShadowStore(sc, '/a', shadowVs)).toBeNull();
    });

    it('settles the in-flight store when the shadow pass fails', async () => {
      shadowImpl = () => Promise.reject(new Error('boom'));
      prefetcher.prefetch(at(7), 10);
      await flushAsync();
      const shadowVs = shadowLoaders.get('/a')!.updateView.mock.calls[0][0];
      expect(awaitShadowStore(sc, '/a', shadowVs)).toBeNull();
    });
  });
});
