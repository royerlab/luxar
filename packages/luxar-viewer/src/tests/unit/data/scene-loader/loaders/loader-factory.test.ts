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

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Capture constructor args for each loader class
const pointsCtorArgs: unknown[][] = [];
const linesCtorArgs: unknown[][] = [];
const gsplatsCtorArgs: unknown[][] = [];
const progressiveCtorArgs: unknown[][] = [];

vi.mock('../../../../../data/points/points-spatial-index-loader', () => ({
  PointsSpatialIndexLoader: vi.fn(function (...args: unknown[]) {
    pointsCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/lines/lines-spatial-index-loader', () => ({
  LinesSpatialIndexLoader: vi.fn(function (...args: unknown[]) {
    linesCtorArgs.push(args);
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
    cachingStore: null,
  };
}

beforeEach(() => {
  pointsCtorArgs.length = 0;
  linesCtorArgs.length = 0;
  gsplatsCtorArgs.length = 0;
  progressiveCtorArgs.length = 0;
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
});
