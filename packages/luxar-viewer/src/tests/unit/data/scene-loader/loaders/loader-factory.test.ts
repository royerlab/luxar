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
const meshCtorArgs: unknown[][] = [];
const meshProgressiveCtorArgs: unknown[][] = [];

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
vi.mock('../../../../../data/mesh/mesh-whole-node-loader', () => ({
  MeshWholeNodeLoader: vi.fn(function (...args: unknown[]) {
    meshCtorArgs.push(args);
  }),
}));
vi.mock('../../../../../data/mesh/mesh-progressive-loader', () => ({
  MeshProgressiveLoader: vi.fn(function (...args: unknown[]) {
    meshProgressiveCtorArgs.push(args);
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
// Return type widened to arbitrary attrs: the mesh cases below re-resolve it
// with per-case sub-LOD attrs (label flags, energy stamps), which the narrow
// inferred `{ foo: string }` would reject.
const zarrOpenMock = vi.fn(
  async (_loc: unknown, _opts: unknown): Promise<{ attrs: Record<string, unknown> }> => ({
    attrs: { foo: 'bar' },
  })
);
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
  createMeshLoader,
  createProgressiveMeshLoader,
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
  meshCtorArgs.length = 0;
  meshProgressiveCtorArgs.length = 0;
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
    //   (loc, node, registry, store, l0?, prefetcher?)
    expect(pointsCtorArgs[0][2]).toBe(deps.arrayRefRegistry);
    expect(pointsCtorArgs[0][3]).toBe(deps.zarrStore);
  });
});

describe('createMeshLoader', () => {
  it('forwards the renderer-owned KTX2 decoder to the whole-node loader', () => {
    const decodeKTX2 = Object.assign(vi.fn(), { dispose: vi.fn() });

    createMeshLoader(makeNode('/mesh', 'mesh'), {} as never, {
      ...makeDeps(),
      decodeKTX2,
    });

    expect(meshCtorArgs).toHaveLength(1);
    expect(meshCtorArgs[0][3]).toMatchObject({ decodeKTX2 });
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

  it('plain factories forward deps.sliceCache as the 7th ctor arg (×3 symmetric)', () => {
    const deps = { ...makeDeps(), sliceCache: sentinelCache };
    createPointsLoader(makeNode('/p'), {} as never, deps);
    createLinesLoader(makeNode('/l', 'lines'), {} as never, deps);
    createGSplatsLoader(makeNode('/g', 'gsplats'), {} as never, deps);
    // (loc, node, registry, store, l0, prefetcher, sliceCache)
    expect(pointsCtorArgs[0][6]).toBe(sentinelCache);
    expect(linesCtorArgs[0][6]).toBe(sentinelCache);
    expect(gsplatsCtorArgs[0][6]).toBe(sentinelCache);
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
      expect(args[6]).toBeUndefined();
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

    // Constructor signature: (loc, node, registry, store, l0?, prefetcher?)
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

  it('clears the label flags on each sub-LOD node, overriding the stored attrs', async () => {
    // A gsplat ladder carries no labels at any level — the authoring path has
    // no `labels` channel — and the pick path only ever resolves labels against
    // the PARENT path anyway, so the clearing here is defensive against
    // whatever attrs a sub-LOD group happens to carry: a truthy flag would make
    // the spatial-index loader publish per-level `ranges` that the ladder concat
    // then has to throw away — the projection runs DOWNSTREAM of the concat
    // (`data-processor-gsplats.ts` gates `emitSourceIndices` on the already-
    // concatenated payload's `ranges`), so no per-level slot → on-disk map is
    // ever composed in the first place (#1423). mockImplementationOnce (not
    // mockImplementation) so the default stub is restored for the next tests.
    zarrOpenMock.mockImplementationOnce((async () => ({
      attrs: { foo: 'bar', has_labels: true, has_image_labels: true },
    })) as never);

    await createProgressiveGSplatsLoader(
      makeNode('/g', 'gsplats'),
      1,
      {} as SceneNode['attrs'],
      makeDeps()
    );

    const lodNode = gsplatsCtorArgs[0][1] as SceneNode;
    // Non-label attrs still spread through from the store…
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
    // …but the two label flags are overridden, not merely absent.
    expect(lodNode.attrs.has_labels).toBe(false);
    expect(lodNode.attrs.has_image_labels).toBe(false);
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

    // Constructor signature: (loc, node, registry, store, l0?, prefetcher?)
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

  it('overrides each sub-LOD node’s label flags with the PARENT’s (here: absent)', async () => {
    // Since #1422 `write_points_multi_lod` stamps the label flags (and the union
    // CSR) on the ladder PARENT and leaves every `additive_<i>` group bare —
    // the parent is also the only path the pick path resolves labels against.
    // A sub-LOD's stored flags are therefore never trusted: the fixture below
    // hand-stamps `has_labels` / `has_image_labels` on the level anyway, to
    // prove the override is unconditional and always takes the PARENT node's
    // declaration (#1439). This parent declares nothing, so both read false and
    // no per-level map is built only for the ladder concat to throw away
    // (#1421).
    // mockImplementationOnce (not mockImplementation) so the default stub is
    // restored for the following tests.
    // `n_points` is VALID here, so the null offsets below can only come from
    // the parent gate — not from the unusable-counts branch.
    zarrOpenMock.mockImplementationOnce((async () => ({
      attrs: { foo: 'bar', n_points: 100, has_labels: true, has_image_labels: true },
    })) as never);

    await createProgressivePointsLoader(
      makeNode('/p', 'points'),
      1,
      {} as SceneNode['attrs'],
      makeDeps()
    );

    const lodNode = pointsCtorArgs[0][1] as SceneNode;
    // Non-label attrs still spread through from the store…
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
    // …but the two label flags are overridden, not merely absent.
    expect(lodNode.attrs.has_labels).toBe(false);
    expect(lodNode.attrs.has_image_labels).toBe(false);
    // No parent CSR ⇒ no level offsets (6th ctor arg) ⇒ the ladder publishes
    // no picking map at all.
    expect(pointsProgressiveCtorArgs[0][5]).toBeNull();
  });

  it('propagates a PARENT label declaration and the on-disk CSR-style levelOffsets', async () => {
    // The parent carries the ladder's UNION CSR (#1422), so every level must
    // build its own level-space map and the concat offsets level `i` by the
    // preceding levels' ON-DISK `n_points` (#1439).
    const node = makeNode('/p', 'points');
    node.attrs = { has_labels: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 100 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 250 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 400 } })) as never);

    await createProgressivePointsLoader(node, 3, {} as SceneNode['attrs'], makeDeps());

    for (const args of pointsCtorArgs) {
      const lodNode = args[1] as SceneNode;
      expect(lodNode.attrs.has_labels).toBe(true);
      expect(lodNode.attrs.has_image_labels).toBe(false);
    }
    // CSR-style bounds: level 0 always starts at 0 and the CLOSING entry is
    // the union row count, so every level (the last one included) has an end
    // the composer can bound its ids against.
    expect(pointsProgressiveCtorArgs[0][5]).toEqual([0, 100, 350, 750]);
  });

  it('propagates a has_image_labels-only parent (the other half of the gate)', async () => {
    // Defensive: pins the `has_image_labels` half of the reader's
    // `parentDeclaresLabels` gate. No producer can currently write this shape —
    // `add_points` refuses to build a ladder when `image_labels` is set (it
    // falls through to a single leaf with a warning) and
    // `write_points_multi_lod` has no image-labels channel at all.
    const node = makeNode('/p', 'points');
    node.attrs = { has_image_labels: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 7 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 11 } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_image_labels).toBe(true);
    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_labels).toBe(false);
    expect(pointsProgressiveCtorArgs[0][5]).toEqual([0, 7, 18]);
  });

  it('propagates a has_keys-only parent and builds level offsets', async () => {
    const node = makeNode('/p', 'points');
    node.attrs = { has_keys: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 7 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 11 } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_keys).toBe(true);
    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_labels).toBe(false);
    expect(pointsProgressiveCtorArgs[0][5]).toEqual([0, 7, 18]);
  });

  it('passes null levelOffsets (no throw) when a labelled ladder lacks n_points', async () => {
    const node = makeNode('/p', 'points');
    node.attrs = { has_image_labels: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 100 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: {} })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect(pointsProgressiveCtorArgs[0][5]).toBeNull();
    // …and with no offsets to place them in, the per-level maps are NOT built:
    // they would be allocated once per level per view change and discarded.
    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_image_labels).toBe(false);
    expect((pointsCtorArgs[1][1] as SceneNode).attrs.has_image_labels).toBe(false);
  });

  it('rejects a NaN n_points (typeof NaN === "number")', async () => {
    const node = makeNode('/p', 'points');
    node.attrs = { has_labels: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 100 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: Number.NaN } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    // A weaker `typeof === 'number'` check would pass NaN through and poison
    // every later offset.
    expect(pointsProgressiveCtorArgs[0][5]).toBeNull();
  });

  it('fails closed when the parent n_points disagrees with the levels’ sum', async () => {
    // The writer's parent `n_points` IS the sum of the levels' row counts, so a
    // mismatch means the CSR and the levels are from different builds —
    // composing would land in someone else's row.
    const node = makeNode('/p', 'points');
    node.attrs = { has_labels: true, n_points: 999 };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 100 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 250 } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect(pointsProgressiveCtorArgs[0][5]).toBeNull();
    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_labels).toBe(false);
  });

  it('accepts a parent n_points that MATCHES the levels’ sum', async () => {
    const node = makeNode('/p', 'points');
    node.attrs = { has_labels: true, n_points: 350 };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 100 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 250 } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect(pointsProgressiveCtorArgs[0][5]).toEqual([0, 100, 350]);
  });

  it('fails closed when the levels sum past the 2^32 picking-map index range', async () => {
    // Each count is individually a safe integer, but the composed map is a
    // Uint32Array of UNION indices — a wider union would wrap into another
    // CSR row, so it is the SUM that has to be range-checked.
    const node = makeNode('/p', 'points');
    node.attrs = { has_labels: true };
    zarrOpenMock
      .mockImplementationOnce((async () => ({ attrs: { n_points: 3_000_000_000 } })) as never)
      .mockImplementationOnce((async () => ({ attrs: { n_points: 3_000_000_000 } })) as never);

    await createProgressivePointsLoader(node, 2, {} as SceneNode['attrs'], makeDeps());

    expect(pointsProgressiveCtorArgs[0][5]).toBeNull();
    expect((pointsCtorArgs[0][1] as SceneNode).attrs.has_labels).toBe(false);
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

  it('clears the label flags on each sub-LOD node, overriding the stored attrs', async () => {
    // Since #1422 `write_lines_multi_lod` stamps the label flags (and the
    // per-vertex union CSR) on the ladder PARENT and leaves every
    // `additive_<i>` group bare, which is also the only path the pick path
    // resolves labels against. The clearing here is therefore defensive against
    // any attrs a sub-LOD may carry — a truthy flag would make the
    // spatial-index loader publish per-level `vertexRangeBounds` that the ladder
    // concat then has to throw away; the projection runs DOWNSTREAM of the concat
    // (`data-processor-lines.ts:95` gates `emitSourceIndices` on the already-
    // concatenated payload's `vertexRangeBounds`), so no per-level segment-slot →
    // on-disk start-vertex map is ever composed (#1424). mockImplementationOnce
    // (not mockImplementation) so the default stub is restored for the
    // following tests.
    zarrOpenMock.mockImplementationOnce((async () => ({
      attrs: { foo: 'bar', has_labels: true, has_image_labels: true },
    })) as never);

    await createProgressiveLinesLoader(
      makeNode('/l', 'lines'),
      1,
      {} as SceneNode['attrs'],
      makeDeps()
    );

    const lodNode = linesCtorArgs[0][1] as SceneNode;
    // Non-label attrs still spread through from the store…
    expect((lodNode.attrs as { foo?: string }).foo).toBe('bar');
    // …but the two label flags are overridden, not merely absent.
    expect(lodNode.attrs.has_labels).toBe(false);
    expect(lodNode.attrs.has_image_labels).toBe(false);
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

  it('synthesizes sub-LOD paths under a NESTED lod-group child (composed points ladder)', async () => {
    // The composed shape: a points node that is BOTH a level of a substitutive
    // lod group AND itself additively laddered, so its path is nested
    // (`/cloud/child_3`), not top-level. Every other case here uses `/p`, which
    // cannot catch a leading-slash / `path.slice(1)` slip — `//additive_0` or
    // `cloud/child_3/additive_0` would both look fine at depth 1.
    await createProgressivePointsLoader(
      makeNode('/cloud/child_3'),
      2,
      {} as SceneNode['attrs'],
      makeDeps()
    );

    expect(pointsCtorArgs).toHaveLength(2);
    expect((pointsCtorArgs[0][1] as SceneNode).path).toBe('/cloud/child_3/additive_0');
    expect((pointsCtorArgs[1][1] as SceneNode).path).toBe('/cloud/child_3/additive_1');
    // The zarr locations resolve WITHOUT the leading slash (store-relative).
    expect((pointsCtorArgs[0][0] as { path: string }).path).toBe('cloud/child_3/additive_0');
    // The progressive wrapper still gets the node's own (slash-prefixed) path.
    expect(pointsProgressiveCtorArgs[0][2]).toBe('/cloud/child_3');
  });
});

describe('createProgressiveMeshLoader', () => {
  /** A laddered mesh node, with whatever parent attrs the case needs. */
  function meshLadderNode(attrs: Record<string, unknown>): SceneNode {
    return {
      path: '/surf',
      type: 'mesh',
      attrs: { type: 'mesh', n_additive_sublods: 4, ...attrs },
      hasSpatialIndex: false,
      children: [],
    };
  }

  it('builds one whole-node loader per additive subgroup', async () => {
    const decodeKTX2 = Object.assign(vi.fn(), { dispose: vi.fn() });
    await createProgressiveMeshLoader(
      meshLadderNode({}),
      3,
      {},
      {
        ...makeDeps(),
        decodeKTX2,
      }
    );

    expect(meshCtorArgs).toHaveLength(3);
    expect(meshCtorArgs.map((args) => args[0])).toEqual([
      '/surf/additive_0',
      '/surf/additive_1',
      '/surf/additive_2',
    ]);
    expect(meshProgressiveCtorArgs).toHaveLength(1);
    expect(meshProgressiveCtorArgs[0][1]).toBe(3);
    expect(meshProgressiveCtorArgs[0][2]).toBe('/surf');
    for (const args of meshCtorArgs) {
      expect(args[3]).toMatchObject({ decodeKTX2 });
    }
  });

  it('clears the label flags on every sub-LOD', async () => {
    // A laddered mesh has no labels — a face-partition duplicates boundary
    // vertices, so the union index space a parent CSR would need does not
    // exist. Cleared here as well as refused by the writer, so a hand-written
    // store cannot make a level publish ranges the concat would discard.
    zarrOpenMock.mockResolvedValue({ attrs: { has_labels: true, has_image_labels: true } });
    await createProgressiveMeshLoader(meshLadderNode({}), 2, {}, makeDeps());

    for (const args of meshCtorArgs) {
      const attrs = args[1] as { has_labels?: boolean; has_image_labels?: boolean };
      expect(attrs.has_labels).toBe(false);
      expect(attrs.has_image_labels).toBe(false);
    }
  });

  it('passes NO energy table — a reveal prefix carries no energy stamps', async () => {
    // Third of the three latches (the writer refuses the keys, this factory
    // never reads them, the loader's getter returns null). If the ladder ever
    // grew an energy-table argument, `energyCompensation` would brighten a
    // partial surface by 1/e(k).
    zarrOpenMock.mockResolvedValue({
      attrs: { lod_stats: { energy_fraction_cum: 0.25 } },
    });
    await createProgressiveMeshLoader(meshLadderNode({}), 2, {}, makeDeps());

    // ctor is (lodLoaders, nLods, path) — exactly three arguments, so there is
    // no slot an energy table could arrive in.
    expect(meshProgressiveCtorArgs[0]).toHaveLength(3);
  });

  it('refuses a ladder whose LEVELS SUM past the vertex cap', async () => {
    // Each level passes its own preflight; the concatenated buffer is what the
    // pick vote key strides over, so the cap binds on the total (spec §6.5).
    await expect(
      createProgressiveMeshLoader(meshLadderNode({ n_vertices: 200_000_000 }), 4, {}, makeDeps())
    ).rejects.toThrow(/vertices across its 4 levels/);
    // Refused before a single subgroup is opened — the refusal must cost no fetch.
    expect(zarrOpenMock).not.toHaveBeenCalled();
    expect(meshCtorArgs).toHaveLength(0);
  });

  it('SENSITIVITY: a ladder just under the cap is built, not refused', async () => {
    // Without this the test above would pass against a guard that refused every
    // ladder, or one whose comparison was inverted. Level counts summing to the
    // parent's totals, because the parent-vs-levels cross-check runs too.
    zarrOpenMock.mockResolvedValue({ attrs: { n_vertices: 67_108_864, n_faces: 4 } });
    await createProgressiveMeshLoader(
      meshLadderNode({ n_vertices: 134_217_728, n_faces: 8 }),
      2,
      {},
      makeDeps()
    );
    expect(meshProgressiveCtorArgs).toHaveLength(1);
  });

  it('does not second-guess a store that omits the parent totals', async () => {
    // Absence is not evidence of a violation: the per-level preflights still
    // run, so the cap binds per level exactly as it did before this guard, and
    // the buffer capacity falls back to the committed counts.
    zarrOpenMock.mockResolvedValue({ attrs: {} });
    await createProgressiveMeshLoader(meshLadderNode({ n_vertices: undefined }), 2, {}, makeDeps());
    expect(meshProgressiveCtorArgs).toHaveLength(1);
  });

  it('refuses a parent FACE total its own levels do not hold', async () => {
    // `n_faces` has no cap of its own and the parent group has no `faces` array
    // for a preflight to check it against, yet the commit sizes the index buffer
    // from it — so a tiny `additive_0` behind a huge parent declaration would
    // allocate for geometry that never arrives.
    zarrOpenMock.mockResolvedValue({ attrs: { n_vertices: 4, n_faces: 2 } });
    await expect(
      createProgressiveMeshLoader(
        meshLadderNode({ n_vertices: 8, n_faces: 2_000_000_000 }),
        2,
        {},
        makeDeps()
      )
    ).rejects.toThrow(/levels hold 4/);
    expect(meshProgressiveCtorArgs).toHaveLength(0);
  });

  it('refuses a declared total no level vouches for', async () => {
    // Fail-closed: a level that declares no counts cannot back the parent's
    // totals, and it would fail its own preflight only after the allocation.
    zarrOpenMock.mockResolvedValue({ attrs: { n_vertices: 4 } });
    await expect(
      createProgressiveMeshLoader(
        meshLadderNode({ n_vertices: 8, n_faces: 100 }),
        2,
        {},
        makeDeps()
      )
    ).rejects.toThrow(/does not declare its own n_faces/);
  });

  it('accepts a parent that UNDER-declares its totals', async () => {
    // The dangerous direction is the parent claiming more than its levels hold;
    // claiming less only costs the allocate-once property (`resolveCapacity`
    // takes the max against the live count), so it must still load.
    zarrOpenMock.mockResolvedValue({ attrs: { n_vertices: 10, n_faces: 20 } });
    await createProgressiveMeshLoader(
      meshLadderNode({ n_vertices: 12, n_faces: 30 }),
      2,
      {},
      makeDeps()
    );
    expect(meshProgressiveCtorArgs).toHaveLength(1);
  });
});
