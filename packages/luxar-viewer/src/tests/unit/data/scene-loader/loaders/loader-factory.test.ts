/**
 * Unit tests for the loader-factory helpers.
 *
 * Strategy:
 *   - Mock each spatial-index loader class so we can detect *which*
 *     loader was constructed and capture its constructor args.
 *   - Mock zarrita just enough to produce a stub `loc` with
 *     `.resolve(path)` and a stub `zarr.open` for the LOD subgroup
 *     reads in the progressive helper.
 *
 * The factory functions are pure given the dependency snapshot, so
 * verifying constructor args + return type is sufficient.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// Capture constructor args for each loader class
const pointsCtorArgs: unknown[][] = [];
const linesCtorArgs: unknown[][] = [];
const gsplatsCtorArgs: unknown[][] = [];
const progressiveCtorArgs: unknown[][] = [];
const pointsProgressiveCtorArgs: unknown[][] = [];
const linesProgressiveCtorArgs: unknown[][] = [];

vi.mock('../../../../../data/points/points-spatial-index-loader', () => ({
  PointsSpatialIndexLoader: vi.fn(function (...args: unknown[]) {
    pointsCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/points/points-progressive-loader', () => ({
  PointsProgressiveLoader: vi.fn(function (...args: unknown[]) {
    pointsProgressiveCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/lines/lines-spatial-index-loader', () => ({
  LinesSpatialIndexLoader: vi.fn(function (...args: unknown[]) {
    linesCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/lines/lines-progressive-loader', () => ({
  LinesProgressiveLoader: vi.fn(function (...args: unknown[]) {
    linesProgressiveCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/gsplats/gsplats-spatial-index-loader', () => ({
  GSplatsSpatialIndexLoader: vi.fn(function (...args: unknown[]) {
    gsplatsCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/gsplats/gsplats-progressive-loader', () => ({
  GSplatsProgressiveLoader: vi.fn(function (...args: unknown[]) {
    progressiveCtorArgs.push(args);
  }),
}));

// Stub zarrita: every resolve() returns another stub with its own
// resolve() so chained calls (parent.resolve('lod_0')) still work.
function makeStubLoc(path: string): {
  kind: 'resolved';
  path: string;
  resolve: (s: string) => { kind: 'resolved'; path: string; resolve: (s: string) => unknown };
} {
  return {
    kind: 'resolved',
    path,
    resolve: (s: string) => makeStubLoc(path === '' ? s : `${path}/${s}`),
  };
}
const zarrOpenMock = vi.fn(async (_loc: unknown, _opts: unknown) => ({ attrs: { foo: 'bar' } }));
vi.mock('zarrita', () => ({
  registry: {},
  root: (_store: unknown) => makeStubLoc(''),
  open: (loc: unknown, opts: unknown) => zarrOpenMock(loc, opts),
  // Stubbed for vitest strict-mock compatibility; loader-factory tests
  // don't reach the consolidation probe but mock isolation can leak.
  withMaybeConsolidatedMetadata: undefined,
}));

import {
  createPointsLoader,
  createLinesLoader,
  createGSplatsLoader,
  createProgressiveGSplatsLoader,
  createProgressivePointsLoader,
  createProgressiveLinesLoader,
  type LoaderFactoryDeps,
} from '../../../../../data/scene-loader/loaders/loader-factory';
import type { SceneNode } from '../../../../../data/data-loader-types';

function makeNode(path: string, type: SceneNode['type'] = 'points'): SceneNode {
  return {
    path,
    type,
    attrs: {},
    hasSpatialIndex: false,
    children: [],
  };
}

function makeDeps(): LoaderFactoryDeps {
  return {
    zarrStore: {} as never,
    arrayRefRegistry: {} as never,
    profiler: null,
    l0Cache: null,
    sliceCache: null,
    cachingStore: null,
  };
}

beforeEach(() => {
  pointsCtorArgs.length = 0;
  linesCtorArgs.length = 0;
  gsplatsCtorArgs.length = 0;
  progressiveCtorArgs.length = 0;
  pointsProgressiveCtorArgs.length = 0;
  linesProgressiveCtorArgs.length = 0;
  zarrOpenMock.mockClear();
});

describe('createPointsLoader', () => {
  it('uses the supplied loc directly for the root node', () => {
    const sentinel = { kind: 'origin' } as unknown;
    createPointsLoader(makeNode('/'), sentinel as never, makeDeps());
    expect(pointsCtorArgs[0][0]).toBe(sentinel);
  });

  it('resolves nested node paths via zarr.root().resolve', () => {
    createPointsLoader(makeNode('/group/points'), {} as never, makeDeps());
    const loc = pointsCtorArgs[0][0] as { kind: string; path: string };
    expect(loc.kind).toBe('resolved');
    expect(loc.path).toBe('group/points');
  });

  it('passes registry and store through to PointsSpatialIndexLoader', () => {
    const deps = makeDeps();
    createPointsLoader(makeNode('/p'), {} as never, deps);
    // Constructor signature:
    //   (loc, node, registry, store, profiler?, l0?, prefetcher?)
    expect(pointsCtorArgs[0][2]).toBe(deps.arrayRefRegistry);
    expect(pointsCtorArgs[0][3]).toBe(deps.zarrStore);
  });
});

describe('createLinesLoader', () => {
  it('constructs LinesSpatialIndexLoader with the resolved location', () => {
    createLinesLoader(makeNode('/lines'), {} as never, makeDeps());
    expect(linesCtorArgs).toHaveLength(1);
    const loc = linesCtorArgs[0][0] as { kind: string; path: string };
    expect(loc.path).toBe('lines');
  });
});

describe('createGSplatsLoader', () => {
  it('constructs GSplatsSpatialIndexLoader with the resolved location', () => {
    createGSplatsLoader(makeNode('/g', 'gsplats'), {} as never, makeDeps());
    expect(gsplatsCtorArgs).toHaveLength(1);
    const loc = gsplatsCtorArgs[0][0] as { kind: string; path: string };
    expect(loc.path).toBe('g');
  });
});

describe('sliceCache wiring (plain-leaf S-cache)', () => {
  const sentinelCache = { kind: 'slice-cache' } as never;

  it('plain factories forward deps.sliceCache as the 8th ctor arg (×3 symmetric)', () => {
    const deps = { ...makeDeps(), sliceCache: sentinelCache };
    createPointsLoader(makeNode('/p'), {} as never, deps);
    createLinesLoader(makeNode('/l', 'lines'), {} as never, deps);
    createGSplatsLoader(makeNode('/g', 'gsplats'), {} as never, deps);
    // (loc, node, registry, store, profiler, l0, prefetcher, sliceCache)
    expect(pointsCtorArgs[0][7]).toBe(sentinelCache);
    expect(linesCtorArgs[0][7]).toBe(sentinelCache);
    expect(gsplatsCtorArgs[0][7]).toBe(sentinelCache);
  });

  it('progressive factories do NOT hand the sliceCache to sub-LOD loaders (their wrapper owns the whole-ladder entry)', async () => {
    const deps = { ...makeDeps(), sliceCache: sentinelCache };
    await createProgressiveGSplatsLoader(
      makeNode('/g', 'gsplats'),
      2,
      {} as SceneNode['attrs'],
      deps
    );
    expect(gsplatsCtorArgs).toHaveLength(2);
    for (const args of gsplatsCtorArgs) {
      expect(args[7]).toBeUndefined();
    }
    // The WRAPPER still receives it (existing behavior, 5th ctor arg).
    expect(progressiveCtorArgs[0][4]).toBe(sentinelCache);
  });
});

describe('createProgressiveGSplatsLoader', () => {
  it('opens N additive sub-LOD subgroups and constructs N GSplatsSpatialIndexLoaders', async () => {
    const node = makeNode('/g', 'gsplats');
    const parentEffectiveAttrs = {
      opacity: 0.5,
      gamma: 1.2,
      intensity: 0.8,
      offset: 0.0,
      blending_mode: 'add',
    } as SceneNode['attrs'];

    await createProgressiveGSplatsLoader(node, 3, parentEffectiveAttrs, makeDeps());

    expect(zarrOpenMock).toHaveBeenCalledTimes(3);
    expect(gsplatsCtorArgs).toHaveLength(3);
    expect(progressiveCtorArgs).toHaveLength(1);
    // Wrapped loader receives the array of LOD loaders + the LOD count.
    expect(progressiveCtorArgs[0][1]).toBe(3);
  });

  it('synthesizes additive sub-LOD nodes that inherit parent effective rendering attrs', async () => {
    const node = makeNode('/g', 'gsplats');
    node.attrs = { extend_to_all: ['t'] };
    const parentEffectiveAttrs = {
      opacity: 0.7,
      gamma: 1.5,
      intensity: 1.1,
      offset: 0.05,
      blending_mode: 'normal',
    } as SceneNode['attrs'];

    await createProgressiveGSplatsLoader(node, 1, parentEffectiveAttrs, makeDeps());

    // Constructor signature: (loc, node, registry, store, profiler?, l0?, prefetcher?)
    const lodNode = gsplatsCtorArgs[0][1] as SceneNode;
    expect(lodNode.path).toBe('/g/additive_0');
    expect(lodNode.type).toBe('gsplats');
    expect(lodNode.attrs.opacity).toBe(0.7);
    expect(lodNode.attrs.gamma).toBe(1.5);
    expect(lodNode.attrs.intensity).toBe(1.1);
    expect(lodNode.attrs.offset).toBe(0.05);
    expect(lodNode.attrs.blending_mode).toBe('normal');
    expect(lodNode.attrs.extend_to_all).toEqual(['t']);
    // Plus the per-additive-sub-LOD raw zarr attrs (foo from our mock).
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
  });

  it('synthesizes a clean additive path for a bare-node ROOT (no "//additive_0")', async () => {
    // A standalone additive-ladder .gsplats.zarr opened directly has node.path "/".
    const node = makeNode('/', 'gsplats');
    await createProgressiveGSplatsLoader(node, 1, {} as SceneNode['attrs'], makeDeps());
    const lodNode = gsplatsCtorArgs[0][1] as SceneNode;
    expect(lodNode.path).toBe('/additive_0'); // not "//additive_0"
  });
});

// Symmetry mirror of the gsplats progressive tests for Points + Lines.
// All three factories share the same shape: open N additive_<i>
// subgroups, build one spatial-index loader per sub-LOD, then wrap them
// in the geometry's progressive loader with (loaders, nAdditive, path).
describe('createProgressivePointsLoader', () => {
  it('opens N additive sub-LOD subgroups and constructs N PointsSpatialIndexLoaders', async () => {
    const node = makeNode('/p', 'points');
    const parentEffectiveAttrs = {
      opacity: 0.5,
      gamma: 1.2,
      intensity: 0.8,
      offset: 0.0,
      blending_mode: 'add',
    } as SceneNode['attrs'];

    await createProgressivePointsLoader(node, 3, parentEffectiveAttrs, makeDeps());

    expect(zarrOpenMock).toHaveBeenCalledTimes(3);
    expect(pointsCtorArgs).toHaveLength(3);
    expect(pointsProgressiveCtorArgs).toHaveLength(1);
    // Progressive wrapper args: (lodLoaders, nAdditive, node.path).
    expect(pointsProgressiveCtorArgs[0][1]).toBe(3);
    expect(pointsProgressiveCtorArgs[0][2]).toBe('/p');
  });

  it('synthesizes additive sub-LOD nodes that inherit parent effective rendering attrs', async () => {
    const node = makeNode('/p', 'points');
    node.attrs = { extend_to_all: ['t'] };
    const parentEffectiveAttrs = {
      opacity: 0.7,
      gamma: 1.5,
      intensity: 1.1,
      offset: 0.05,
      blending_mode: 'normal',
    } as SceneNode['attrs'];

    await createProgressivePointsLoader(node, 1, parentEffectiveAttrs, makeDeps());

    // Constructor signature: (loc, node, registry, store, profiler?, l0?, prefetcher?)
    const lodNode = pointsCtorArgs[0][1] as SceneNode;
    expect(lodNode.path).toBe('/p/additive_0');
    expect(lodNode.type).toBe('points');
    expect(lodNode.attrs.opacity).toBe(0.7);
    expect(lodNode.attrs.gamma).toBe(1.5);
    expect(lodNode.attrs.intensity).toBe(1.1);
    expect(lodNode.attrs.offset).toBe(0.05);
    expect(lodNode.attrs.blending_mode).toBe('normal');
    expect(lodNode.attrs.extend_to_all).toEqual(['t']);
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
    // Registry + store threaded through to each sub-LOD loader.
    const deps = makeDeps();
    pointsCtorArgs.length = 0;
    await createProgressivePointsLoader(node, 1, parentEffectiveAttrs, deps);
    expect(pointsCtorArgs[0][2]).toBe(deps.arrayRefRegistry);
    expect(pointsCtorArgs[0][3]).toBe(deps.zarrStore);
  });
});

describe('createProgressiveLinesLoader', () => {
  it('opens N additive sub-LOD subgroups and constructs N LinesSpatialIndexLoaders', async () => {
    const node = makeNode('/l', 'lines');
    const parentEffectiveAttrs = {
      opacity: 0.5,
      gamma: 1.2,
      intensity: 0.8,
      offset: 0.0,
      blending_mode: 'add',
    } as SceneNode['attrs'];

    await createProgressiveLinesLoader(node, 3, parentEffectiveAttrs, makeDeps());

    expect(zarrOpenMock).toHaveBeenCalledTimes(3);
    expect(linesCtorArgs).toHaveLength(3);
    expect(linesProgressiveCtorArgs).toHaveLength(1);
    // Progressive wrapper args: (lodLoaders, nAdditive, node.path).
    expect(linesProgressiveCtorArgs[0][1]).toBe(3);
    expect(linesProgressiveCtorArgs[0][2]).toBe('/l');
  });

  it('synthesizes additive sub-LOD nodes that inherit parent effective rendering attrs', async () => {
    const node = makeNode('/l', 'lines');
    node.attrs = { extend_to_all: ['t'] };
    const parentEffectiveAttrs = {
      opacity: 0.7,
      gamma: 1.5,
      intensity: 1.1,
      offset: 0.05,
      blending_mode: 'normal',
    } as SceneNode['attrs'];

    await createProgressiveLinesLoader(node, 1, parentEffectiveAttrs, makeDeps());

    const lodNode = linesCtorArgs[0][1] as SceneNode;
    expect(lodNode.path).toBe('/l/additive_0');
    expect(lodNode.type).toBe('lines');
    expect(lodNode.attrs.opacity).toBe(0.7);
    expect(lodNode.attrs.gamma).toBe(1.5);
    expect(lodNode.attrs.intensity).toBe(1.1);
    expect(lodNode.attrs.offset).toBe(0.05);
    expect(lodNode.attrs.blending_mode).toBe('normal');
    expect(lodNode.attrs.extend_to_all).toEqual(['t']);
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
    const deps = makeDeps();
    linesCtorArgs.length = 0;
    await createProgressiveLinesLoader(node, 1, parentEffectiveAttrs, deps);
    expect(linesCtorArgs[0][2]).toBe(deps.arrayRefRegistry);
    expect(linesCtorArgs[0][3]).toBe(deps.zarrStore);
  });
});
// The energy-table (quality stamps) plumbing: each additive_<i> subgroup's
// `lod_stats.energy_fraction_cum` is collected into a table and passed as the
// progressive loaders' 4th constructor arg (→ committedEnergyFraction, the
// display gate's energy-threshold release). Unstamped subgroups yield null
// entries — the loader then reads as unstamped.
describe('progressive loader energy tables (quality stamps)', () => {
  afterEach(() => {
    // mockClear() in the global beforeEach does NOT reset implementations —
    // restore the file-wide default so these stamps never leak elsewhere.
    zarrOpenMock.mockImplementation(async () => ({ attrs: { foo: 'bar' } }));
  });

  it('collects e(k) from each sub-LOD attrs and passes the table to the wrapper (all three geometries)', async () => {
    zarrOpenMock.mockImplementation((async () => ({
      attrs: { lod_stats: { energy_fraction_cum: 0.75 } },
    })) as never);
    await createProgressiveGSplatsLoader(
      makeNode('/g', 'gsplats'),
      2,
      {} as SceneNode['attrs'],
      makeDeps()
    );
    await createProgressivePointsLoader(makeNode('/p'), 2, {} as SceneNode['attrs'], makeDeps());
    await createProgressiveLinesLoader(
      makeNode('/l', 'lines'),
      2,
      {} as SceneNode['attrs'],
      makeDeps()
    );
    expect(progressiveCtorArgs[0][3]).toEqual([0.75, 0.75]);
    expect(pointsProgressiveCtorArgs[0][3]).toEqual([0.75, 0.75]);
    expect(linesProgressiveCtorArgs[0][3]).toEqual([0.75, 0.75]);
  });

  it('yields null entries for unstamped (legacy) sub-LODs', async () => {
    zarrOpenMock
      .mockImplementationOnce((async () => ({
        attrs: { lod_stats: { energy_fraction_cum: 0.4 } },
      })) as never)
      .mockImplementationOnce((async () => ({ attrs: { lod_stats: {} } })) as never)
      .mockImplementationOnce((async () => ({ attrs: {} })) as never);
    await createProgressiveGSplatsLoader(
      makeNode('/g', 'gsplats'),
      3,
      {} as SceneNode['attrs'],
      makeDeps()
    );
    expect(progressiveCtorArgs[0][3]).toEqual([0.4, null, null]);
  });
});
