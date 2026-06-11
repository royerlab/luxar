/**
 * Unit tests for `buildSceneGraph` in scene-loader/nodes/build-scene-graph.ts.
 *
 * Two invariants under test:
 *   1. Hierarchical assembly — sorted-by-depth processing puts each
 *      group under its zarr-path parent.
 *   2. Internal-subtree skipping — `additive_<i>/` subgroups under a
 *      gsplats node carry `type: "gsplats"` themselves (the per-LOD
 *      attrs) and would otherwise surface as spurious child nodes in
 *      the Scene Graph monitor. buildSceneGraph must mark the subtree
 *      as internal once it sees a gsplats node and skip everything
 *      beneath it. Overlay groups are similarly skipped.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

// Per-test attrs lookup the zarr.open mock walks at runtime.
const attrsByPath: Record<string, Record<string, unknown>> = {};
const openCalls: string[] = [];

// Stub zarr location: resolve() composes paths so the mock can echo
// the resolved path back as the attrs-lookup key.
function makeStubLoc(path: string): {
  kind: 'resolved';
  path: string;
  resolve: (s: string) => ReturnType<typeof makeStubLoc>;
} {
  return {
    kind: 'resolved',
    path,
    resolve: (s: string) => makeStubLoc(path === '' ? s : `${path}/${s}`),
  };
}

vi.mock('zarrita', () => ({
  registry: {},
  root: (_store: unknown) => makeStubLoc(''),
  open: vi.fn(async (loc: { path: string }, _opts: unknown) => {
    openCalls.push(loc.path);
    return { attrs: attrsByPath[`/${loc.path}`] ?? {} };
  }),
  withMaybeConsolidatedMetadata: undefined,
}));

// Mock the enumerate-store dependency so we can feed buildSceneGraph
// arbitrary fixture listings without needing a real store.
const enumerateStoreMock = vi.fn();
vi.mock('../../../../../data/scene-loader/nodes/enumerate-store', () => ({
  enumerateStore: (...args: unknown[]) => enumerateStoreMock(...args),
}));

import { buildSceneGraph } from '../../../../../data/scene-loader/nodes/build-scene-graph';
import type { ZarrSceneAttrs } from '../../../../../types/zarr';

function makeRootAttrs(): ZarrSceneAttrs {
  return { scene_dimensions: { dimensions: [] } } as unknown as ZarrSceneAttrs;
}

beforeEach(() => {
  for (const k of Object.keys(attrsByPath)) delete attrsByPath[k];
  openCalls.length = 0;
  enumerateStoreMock.mockReset();
});

describe('buildSceneGraph — hierarchy', () => {
  it('places groups under their zarr-path parents', async () => {
    enumerateStoreMock.mockResolvedValue([
      { path: '/a', kind: 'group' },
      { path: '/a/b', kind: 'group' },
      { path: '/a/b/leaf', kind: 'group' },
    ]);
    attrsByPath['/a'] = { type: 'group' };
    attrsByPath['/a/b'] = { type: 'group' };
    attrsByPath['/a/b/leaf'] = { type: 'points' };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children?.[0].path).toBe('/a');
    expect(root.children?.[0].children?.[0].path).toBe('/a/b');
    expect(root.children?.[0].children?.[0].children?.[0].path).toBe('/a/b/leaf');
    expect(root.children?.[0].children?.[0].children?.[0].type).toBe('points');
  });
});

describe('buildSceneGraph — overlay skip', () => {
  it('omits /overlays and its descendants from the scene graph', async () => {
    enumerateStoreMock.mockResolvedValue([
      { path: '/group_a', kind: 'group' },
      { path: '/overlays', kind: 'group' },
      { path: '/overlays/text_1', kind: 'group' },
    ]);
    attrsByPath['/group_a'] = { type: 'group' };
    attrsByPath['/overlays'] = {};
    attrsByPath['/overlays/text_1'] = { type: 'overlay_text' };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children?.map((c) => c.path)).toEqual(['/group_a']);
    expect(openCalls).not.toContain('overlays');
    expect(openCalls).not.toContain('overlays/text_1');
  });
});

describe('buildSceneGraph — gsplats internal-subtree skip', () => {
  it('does not surface additive_<i> LOD subgroups as scene-graph children of a gsplats node', async () => {
    // Mimic a scene with a single multi-additive gsplats node:
    //   /splats               type=gsplats, n_additive_sublods=3
    //   /splats/additive_0    type=gsplats   (per-LOD attrs)
    //   /splats/additive_1    type=gsplats
    //   /splats/additive_2    type=gsplats
    enumerateStoreMock.mockResolvedValue([
      { path: '/splats', kind: 'group' },
      { path: '/splats/additive_0', kind: 'group' },
      { path: '/splats/additive_1', kind: 'group' },
      { path: '/splats/additive_2', kind: 'group' },
    ]);
    attrsByPath['/splats'] = { type: 'gsplats', n_additive_sublods: 3 };
    attrsByPath['/splats/additive_0'] = { type: 'gsplats', n_splats: 5 };
    attrsByPath['/splats/additive_1'] = { type: 'gsplats', n_splats: 10 };
    attrsByPath['/splats/additive_2'] = { type: 'gsplats', n_splats: 20 };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children).toHaveLength(1);
    const splatsNode = root.children?.[0];
    expect(splatsNode?.path).toBe('/splats');
    expect(splatsNode?.type).toBe('gsplats');
    // The additive_<i> subgroups must NOT appear as scene-graph
    // children — they're internal to the gsplats loader.
    expect(splatsNode?.children).toEqual([]);
    // And we should never have opened them (avoids unnecessary fetches).
    expect(openCalls).toEqual(['splats']);
  });

  it('also skips arbitrarily deeper internal subgroups under a gsplats node', async () => {
    // Defensive: even if a future per-LOD subgroup adds its own
    // descendants, they stay internal.
    enumerateStoreMock.mockResolvedValue([
      { path: '/splats', kind: 'group' },
      { path: '/splats/additive_0', kind: 'group' },
      { path: '/splats/additive_0/inner', kind: 'group' },
    ]);
    attrsByPath['/splats'] = { type: 'gsplats', n_additive_sublods: 1 };
    attrsByPath['/splats/additive_0'] = { type: 'gsplats' };
    attrsByPath['/splats/additive_0/inner'] = { type: 'gsplats' };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children?.[0].children).toEqual([]);
    expect(openCalls).toEqual(['splats']);
  });

  it('does not skip siblings of a gsplats node', async () => {
    // A gsplats node next to a regular group must not bleed its
    // internal-subtree skip across siblings.
    enumerateStoreMock.mockResolvedValue([
      { path: '/splats', kind: 'group' },
      { path: '/splats/additive_0', kind: 'group' },
      { path: '/points', kind: 'group' },
    ]);
    attrsByPath['/splats'] = { type: 'gsplats', n_additive_sublods: 1 };
    attrsByPath['/splats/additive_0'] = { type: 'gsplats' };
    attrsByPath['/points'] = { type: 'points' };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children?.map((c) => c.path).sort()).toEqual(['/points', '/splats']);
    expect(openCalls.sort()).toEqual(['points', 'splats']);
  });

  it('skips deeper subgroups under a gsplats nested inside a group', async () => {
    // The prefix-match logic must work for gsplats at any depth.
    enumerateStoreMock.mockResolvedValue([
      { path: '/grp', kind: 'group' },
      { path: '/grp/splats', kind: 'group' },
      { path: '/grp/splats/additive_0', kind: 'group' },
      { path: '/grp/splats/additive_1', kind: 'group' },
    ]);
    attrsByPath['/grp'] = { type: 'group' };
    attrsByPath['/grp/splats'] = { type: 'gsplats', n_additive_sublods: 2 };
    attrsByPath['/grp/splats/additive_0'] = { type: 'gsplats' };
    attrsByPath['/grp/splats/additive_1'] = { type: 'gsplats' };

    const root = await buildSceneGraph(makeStubLoc('') as never, makeRootAttrs(), {} as never);

    expect(root.children?.[0].path).toBe('/grp');
    expect(root.children?.[0].children?.[0].path).toBe('/grp/splats');
    expect(root.children?.[0].children?.[0].children).toEqual([]);
  });
});

describe('buildSceneGraph — bare node root (standalone .gsplats.zarr)', () => {
  it('a bare gsplats leaf root becomes a childless gsplats node', async () => {
    // A single-set v3.0 standalone file: arrays live directly under root,
    // there are no child GROUPS. The root IS the gsplats leaf.
    enumerateStoreMock.mockResolvedValue([]);
    const rootAttrs = {
      type: 'gsplats',
      format_type: 'gsplats_zarr',
      format_version: '3.0',
      n_splats: 100,
      ndim: 3,
    } as unknown as ZarrSceneAttrs;

    const root = await buildSceneGraph(makeStubLoc('') as never, rootAttrs, {} as never);

    expect(root.type).toBe('gsplats'); // not 'scene'
    expect(root.path).toBe('/');
    expect(root.children).toEqual([]);
  });

  it('a bare gsplats leaf root with an additive ladder hides additive_<i>', async () => {
    // A v3.0 ladder leaf at the root: additive_<i>/ subgroups are internal
    // to the gsplats loader and must NOT become scene-graph children.
    enumerateStoreMock.mockResolvedValue([
      { path: '/additive_0', kind: 'group' },
      { path: '/additive_1', kind: 'group' },
    ]);
    attrsByPath['/additive_0'] = { type: 'gsplats', n_splats: 30 };
    attrsByPath['/additive_1'] = { type: 'gsplats', n_splats: 10 };
    const rootAttrs = {
      type: 'gsplats',
      n_additive_sublods: 2,
      ndim: 3,
    } as unknown as ZarrSceneAttrs;

    const root = await buildSceneGraph(makeStubLoc('') as never, rootAttrs, {} as never);

    expect(root.type).toBe('gsplats');
    expect(root.children).toEqual([]);
    // additive subgroups never opened (the leaf loader walks them itself)
    expect(openCalls).toEqual([]);
  });

  it('a bare kind=lod root keeps its child_<i> as lod-group children', async () => {
    enumerateStoreMock.mockResolvedValue([
      { path: '/child_0', kind: 'group' },
      { path: '/child_1', kind: 'group' },
    ]);
    attrsByPath['/child_0'] = { type: 'gsplats', n_splats: 8 }; // coarsest
    attrsByPath['/child_1'] = { type: 'gsplats', n_splats: 100 }; // finest
    const rootAttrs = {
      type: 'group',
      kind: 'lod',
      selector: 'pixel_size',
      default_level: 1,
    } as unknown as ZarrSceneAttrs;

    const root = await buildSceneGraph(makeStubLoc('') as never, rootAttrs, {} as never);

    expect(root.type).toBe('group');
    expect((root.attrs as Record<string, unknown>).kind).toBe('lod');
    expect(root.children?.map((c) => c.path).sort()).toEqual(['/child_0', '/child_1']);
    expect(root.children?.every((c) => c.type === 'gsplats')).toBe(true);
  });

  it('a bare kind=partition root keeps its part_<i> as children', async () => {
    enumerateStoreMock.mockResolvedValue([
      { path: '/part_0', kind: 'group' },
      { path: '/part_1', kind: 'group' },
    ]);
    attrsByPath['/part_0'] = { type: 'gsplats', n_splats: 20 };
    attrsByPath['/part_1'] = { type: 'gsplats', n_splats: 20 };
    const rootAttrs = {
      type: 'group',
      kind: 'partition',
      display_type: 'gsplats',
      max_elements: 25,
    } as unknown as ZarrSceneAttrs;

    const root = await buildSceneGraph(makeStubLoc('') as never, rootAttrs, {} as never);

    expect(root.type).toBe('group');
    expect((root.attrs as Record<string, unknown>).kind).toBe('partition');
    expect(root.children?.map((c) => c.path).sort()).toEqual(['/part_0', '/part_1']);
  });

  it('a true scene root is still a scene container', async () => {
    enumerateStoreMock.mockResolvedValue([{ path: '/g', kind: 'group' }]);
    attrsByPath['/g'] = { type: 'gsplats', n_splats: 5 };
    const rootAttrs = {
      type: 'scene',
      scene_dimensions: { dimensions: [] },
    } as unknown as ZarrSceneAttrs;

    const root = await buildSceneGraph(makeStubLoc('') as never, rootAttrs, {} as never);

    expect(root.type).toBe('scene');
    expect(root.children?.[0].path).toBe('/g');
  });
});
