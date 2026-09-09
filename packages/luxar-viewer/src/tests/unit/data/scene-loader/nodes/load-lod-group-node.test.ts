/**
 * Tests for ``loadLodGroupNode``.
 *
 * Strategy: the lod_group loader is mostly a thin wrapper that
 * recurses children through ``loadSceneNodes`` and pushes the
 * resulting THREE nodes + per-child attrs (``coverage_fraction``,
 * ``position_bounds``, optional ``lod_bounds``) into the
 * :class:`LODGroupRegistry`. We mock
 * ``loadSceneNodes`` to attach a stub mesh per child, then assert on
 * what landed in the registry.
 *
 * The registry's per-frame selector logic is exercised separately in
 * ``../../scene/lod-group-registry.test.ts``.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const loadSceneNodesMock = vi.fn();

// loadLodGroupNode imports loadGSplatsNodeCheap/Expensive directly (not
// injected like loadSceneNodes), so mock that module to exercise the
// lazy deferral path without the real gsplats loader machinery.
const { loadGSplatsNodeCheapMock, loadGSplatsNodeExpensiveMock } = vi.hoisted(() => ({
  loadGSplatsNodeCheapMock: vi.fn(),
  loadGSplatsNodeExpensiveMock: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/nodes/load-gsplats-node', () => ({
  loadGSplatsNodeCheap: loadGSplatsNodeCheapMock,
  loadGSplatsNodeExpensive: loadGSplatsNodeExpensiveMock,
}));

// Points deferral mirrors gsplats: mock the points cheap/expensive split so the
// lazy-points path can be exercised without the real spatial-index loader.
const { loadPointsNodeCheapMock, loadPointsNodeExpensiveMock } = vi.hoisted(() => ({
  loadPointsNodeCheapMock: vi.fn(),
  loadPointsNodeExpensiveMock: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/nodes/load-points-node', () => ({
  loadPointsNodeCheap: loadPointsNodeCheapMock,
  loadPointsNodeExpensive: loadPointsNodeExpensiveMock,
}));

// Lines deferral mirrors points/gsplats.
const { loadLinesNodeCheapMock, loadLinesNodeExpensiveMock } = vi.hoisted(() => ({
  loadLinesNodeCheapMock: vi.fn(),
  loadLinesNodeExpensiveMock: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/nodes/load-lines-node', () => ({
  loadLinesNodeCheap: loadLinesNodeCheapMock,
  loadLinesNodeExpensive: loadLinesNodeExpensiveMock,
}));

// Mesh deferral mirrors the three above — the fourth branch of the defer
// dispatch, which throws for any type it does not name.
const { loadMeshNodeCheapMock, loadMeshNodeExpensiveMock } = vi.hoisted(() => ({
  loadMeshNodeCheapMock: vi.fn(),
  loadMeshNodeExpensiveMock: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/nodes/load-mesh-node', () => ({
  loadMeshNodeCheap: loadMeshNodeCheapMock,
  loadMeshNodeExpensive: loadMeshNodeExpensiveMock,
}));

import { loadLodGroupNode } from '../../../../../data/scene-loader/nodes/load-lod-group-node';
import { LoaderError } from '../../../../../data/scene-loader/nodes/load-leaf-error-dispatch';
import { ArchiveFaultError } from '../../../../../cache/chunk-source';
import { LODGroupRegistry } from '../../../../../scene/lod-group-registry';
import { log } from '../../../../../utils/log';
import { makeTestNodeBuildCtx } from '../../../../helpers/make-test-node-build-ctx';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

beforeEach(() => {
  loadSceneNodesMock.mockReset();
  loadGSplatsNodeCheapMock.mockReset();
  loadGSplatsNodeExpensiveMock.mockReset();
  loadPointsNodeCheapMock.mockReset();
  loadPointsNodeExpensiveMock.mockReset();
  loadLinesNodeCheapMock.mockReset();
  loadLinesNodeExpensiveMock.mockReset();
  loadMeshNodeCheapMock.mockReset();
  loadMeshNodeExpensiveMock.mockReset();
  // Default cheap-attach: attach a stub mesh named after the node path
  // (so getObjectByName / visibility toggles work) and return a
  // placeholder + dummy loader. Expensive defaults to a no-op resolve.
  const cheapImpl = async (node: SceneNode, parent: THREE.Object3D) => {
    const mesh = new THREE.Mesh();
    mesh.name = node.path;
    parent.add(mesh);
    return { placeholder: mesh, loader: {} as never };
  };
  loadGSplatsNodeCheapMock.mockImplementation(cheapImpl);
  loadGSplatsNodeExpensiveMock.mockResolvedValue(undefined);
  loadPointsNodeCheapMock.mockImplementation(cheapImpl);
  loadPointsNodeExpensiveMock.mockResolvedValue(undefined);
  loadLinesNodeCheapMock.mockImplementation(cheapImpl);
  loadLinesNodeExpensiveMock.mockResolvedValue(undefined);
  loadMeshNodeCheapMock.mockImplementation(cheapImpl);
  loadMeshNodeExpensiveMock.mockResolvedValue(undefined);
});

function makeChildNode(
  path: string,
  coverageFraction: number,
  positionBounds: { min: number[]; max: number[] } | undefined = {
    min: [0, 0, 0],
    max: [1, 1, 1],
  },
  lodBounds?: { min: number[]; max: number[] }
): SceneNode {
  return {
    path,
    type: 'gsplats',
    attrs: {
      type: 'gsplats',
      coverage_fraction: coverageFraction,
      ...(positionBounds ? { position_bounds: positionBounds } : {}),
      ...(lodBounds ? { lod_bounds: lodBounds } : {}),
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

/** A ``points`` leaf child (the finest level of a points-substitutive ladder). */
function makePointsChildNode(
  path: string,
  coverageFraction: number,
  positionBounds: { min: number[]; max: number[] } = { min: [0, 0, 0], max: [1, 1, 1] }
): SceneNode {
  return {
    path,
    type: 'points',
    attrs: {
      type: 'points',
      coverage_fraction: coverageFraction,
      position_bounds: positionBounds,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

/** A mesh lod-group child (the fourth deferrable type). */
function makeMeshChildNode(path: string, coverageFraction: number): SceneNode {
  return {
    path,
    type: 'mesh',
    attrs: {
      type: 'mesh',
      coverage_fraction: coverageFraction,
      position_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

/** A ``lines`` leaf child (the finest level of a lines-substitutive ladder). */
function makeLinesChildNode(
  path: string,
  coverageFraction: number,
  positionBounds: { min: number[]; max: number[] } = { min: [0, 0, 0], max: [1, 1, 1] }
): SceneNode {
  return {
    path,
    type: 'lines',
    attrs: {
      type: 'lines',
      coverage_fraction: coverageFraction,
      position_bounds: positionBounds,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

/**
 * A nested group child (e.g. multiscale's fine kind=partition branch). The
 * ``displayType`` lets a single test prove the deferral is geometry-agnostic —
 * a partition/lod wrapper of points, lines or mesh defers identically to
 * gsplats.
 */
function makeGroupChildNode(
  path: string,
  coverageFraction: number,
  displayType: 'gsplats' | 'points' | 'lines' | 'mesh' = 'gsplats',
  kind: 'partition' | 'lod' = 'partition',
  positionBounds: { min: number[]; max: number[] } = { min: [0, 0, 0], max: [1, 1, 1] }
): SceneNode {
  return {
    path,
    type: 'group',
    attrs: {
      type: 'group',
      kind,
      display_type: displayType,
      coverage_fraction: coverageFraction,
      position_bounds: positionBounds,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function makeLodGroupNode(
  children: SceneNode[],
  extraAttrs: Record<string, unknown> = {}
): SceneNode {
  return {
    path: '/lod',
    type: 'group',
    attrs: {
      type: 'group',
      kind: 'lod',
      display_type: 'gsplats',
      selector: 'coverage',
      ...extraAttrs,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

function makeCtx(registry?: LODGroupRegistry, overrides: Partial<NodeBuildCtx> = {}): NodeBuildCtx {
  // `applyTransform` is the only nodeFactory member loadLodGroupNode reaches —
  // placeholder construction happens in the mocked loadSceneNodes recursion.
  const nodeFactory = {
    applyTransform: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  // Only the members loadLodGroupNode actually reads are named here; the rest
  // come from the shared factory (see make-test-node-build-ctx.ts).
  return makeTestNodeBuildCtx({
    registry: {
      registerGSplatsLoader: vi.fn(),
      registerPointsLoader: vi.fn(),
      registerLinesLoader: vi.fn(),
      registerMeshLoader: vi.fn(),
      unregisterPointsLoader: vi.fn(),
      unregisterLinesLoader: vi.fn(),
    } as never,
    lodGroupRegistry: registry,
    nodeFactory,
    viewState: { displayDims: [0, 1, 2], slicePosition: [], tolerance: [] },
    ...overrides,
  });
}

/**
 * Mock loadSceneNodes implementation that attaches a stub mesh to
 * parentThree for each invocation, named after the SceneNode path —
 * loadLodGroupNode looks the child object up by name.
 */
/** Minimal stub for the zarr Location surface: only ``resolve()`` is called. */
function makeStubLoc(): never {
  return { resolve: () => makeStubLoc() } as never;
}

function attachStubChildren(): void {
  loadSceneNodesMock.mockImplementation(async (node: SceneNode, parent: THREE.Object3D) => {
    const mesh = new THREE.Mesh();
    mesh.name = node.path;
    parent.add(mesh);
  });
}

// ────────────────────────────────────────────────────────────────────────
// Happy-path: registry populated with per-child attrs
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — registry registration', () => {
  it('registers an entry with one LODGroupChild per scene-graph child', async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0, { min: [0, 0, 0], max: [1, 1, 1] }),
        makeChildNode('/lod/child_1', 0.5, { min: [0, 0, 0], max: [2, 2, 2] }),
        makeChildNode('/lod/child_2', 1.0, { min: [0, 0, 0], max: [4, 4, 4] }),
      ],
      { default_level: 0 }
    );

    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(reg.size()).toBe(1);
    const entry = reg.get('/lod')!;
    expect(entry.children).toHaveLength(3);
    expect(entry.children.map((c) => c.coverageFraction)).toEqual([0, 0.5, 1.0]);
    expect(entry.children[1].positionBounds).toEqual({
      min: [0, 0, 0],
      max: [2, 2, 2],
    });
    expect(entry.activeChildIndex).toBe(0);
    expect(entry.selectorMode).toBe('auto');
    // makeLodGroupNode stamps the legacy attr; the entry must carry it so the
    // registry keeps the diagonal metric for this group.
    expect(entry.selector).toBe('coverage');
  });

  it("passes selector='screen-area' through to the registry entry", async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0, { min: [0, 0, 0], max: [1, 1, 1] }),
        makeChildNode('/lod/child_1', 0.5, { min: [0, 0, 0], max: [1, 1, 1] }),
      ],
      { selector: 'screen-area' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    expect(reg.get('/lod')!.selector).toBe('screen-area');
  });

  it('whitelists unknown selector spellings to the legacy diagonal metric', async () => {
    // A stale pre-v3.2 'pixel_size' (or any future/unknown value) must not
    // switch the metric: those groups' thresholds are in diagonal units
    // (min_pixel_size adaptation included), so anything but the literal
    // 'screen-area' falls back to 'coverage'.
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0, { min: [0, 0, 0], max: [1, 1, 1] }),
        makeChildNode('/lod/child_1', 0.5, { min: [0, 0, 0], max: [1, 1, 1] }),
      ],
      { selector: 'pixel_size' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    expect(reg.get('/lod')!.selector).toBe('coverage');
  });

  it('clamps default_level out of range to the nearest valid index', async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 99 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    expect(reg.get('/lod')!.activeChildIndex).toBe(1);
  });

  it('registers every loaded child and warns when lod_bounds only stamp part of the ladder', async () => {
    // Atomic-swap on initial load is enforced by the loader's
    // sequential-await + visible=false-after-attach pattern + atomic
    // ``register()`` at the end (see the no-stacked-LOD-flash test
    // below). The selector itself no longer carries a per-child
    // readiness flag.
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode([
      makeChildNode(
        '/lod/child_0',
        0,
        { min: [-100, -100, -100], max: [100, 100, 100] },
        { min: [-1, -1, -1], max: [1, 1, 1] }
      ),
      makeGroupChildNode('/lod/fine_partition', 0.5),
    ]);
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
      const entry = reg.get('/lod')!;
      expect(entry.children).toHaveLength(2);
      expect(entry.children.map((c) => c.coverageFraction)).toEqual([0, 0.5]);
      expect(entry.children[0].lodBounds).toEqual({
        min: [-1, -1, -1],
        max: [1, 1, 1],
      });
      expect(entry.children[1].lodBounds).toBeUndefined();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.anything(),
        expect.stringContaining(
          'lod_group /lod: lod_bounds are only usable on part of the ladder; missing /lod/fine_partition'
        )
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('falls back to center_bounds when position_bounds is absent', async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode([
      {
        path: '/lod/child_0',
        type: 'gsplats',
        attrs: {
          type: 'gsplats',
          coverage_fraction: 0,
          center_bounds: { min: [1, 1, 1], max: [3, 3, 3] },
        } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [],
      },
    ]);
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    expect(reg.get('/lod')!.children[0].positionBounds).toEqual({
      min: [1, 1, 1],
      max: [3, 3, 3],
    });
  });

  it.each([
    ['an object with min/max arrays', 'bad'],
    ['non-empty', { min: [], max: [] }],
    ['equal min/max lengths', { min: [-1, -1, -1], max: [1, 1] }],
    ['3 values per bound', { min: [-1, -1], max: [1, 1] }],
    ['numeric entries', { min: [0, 0, 'bad'], max: [1, 1, 1] }],
    ['finite numbers', { min: [0, 0, Number.NaN], max: [1, 1, 1] }],
    ['ordered bounds', { min: [0, 0.8, 0], max: [1, 0.2, 1] }],
    ['bounds contained in position_bounds', { min: [-1, 0, 0], max: [1, 1, 1] }],
    [
      'child has no usable position_bounds to validate against',
      { min: [0, 0, 0], max: [1, 1, 1] },
      { min: [0, 0, 0], max: [1, 1] },
    ],
  ])(
    'rejects malformed lod_bounds with a diagnostic requiring %s',
    async (
      reason: string,
      lodBounds: unknown,
      positionBounds?: { min: number[]; max: number[] }
    ) => {
      attachStubChildren();
      const reg = new LODGroupRegistry({
        getCamera: () => new THREE.Camera(),
        getViewportSize: () => ({ width: 100, height: 100 }),
        getDisplayDims: () => [0, 1, 2],
      });
      const ctx = makeCtx(reg);
      const child = makeChildNode('/lod/child_0', 0);
      (child.attrs as Record<string, unknown>).lod_bounds = lodBounds;
      if (positionBounds) {
        (child.attrs as Record<string, unknown>).position_bounds = positionBounds;
      }
      const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

      try {
        await loadLodGroupNode(
          makeLodGroupNode([child]),
          new THREE.Group(),
          makeStubLoc(),
          ctx,
          loadSceneNodesMock
        );

        expect(reg.get('/lod')!.children[0].lodBounds).toBeUndefined();
        expect(warnSpy).toHaveBeenCalledWith(
          expect.anything(),
          expect.stringContaining(
            `/lod/child_0: rejected lod_bounds; ${positionBounds ? '' : 'expected '}${reason}`
          )
        );
      } finally {
        warnSpy.mockRestore();
      }
    }
  );

  it('reports a dimension mismatch without duplicating the expected shape', async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);
    const child = makeChildNode('/lod/child_0', 0);
    (child.attrs as Record<string, unknown>).lod_bounds = { min: [0, 0], max: [1, 1] };
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    try {
      await loadLodGroupNode(
        makeLodGroupNode([child]),
        new THREE.Group(),
        makeStubLoc(),
        ctx,
        loadSceneNodesMock
      );

      expect(warnSpy).toHaveBeenCalledWith(
        expect.anything(),
        'lod_group child /lod/child_0: rejected lod_bounds; expected 3 values per bound, ' +
          'falling back to position_bounds'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rejects lod_bounds when the child has no position_bounds to validate against', async () => {
    attachStubChildren();
    const reg = new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
    const ctx = makeCtx(reg);
    const child = makeChildNode('/lod/child_0', 0);
    delete (child.attrs as Record<string, unknown>).position_bounds;
    (child.attrs as Record<string, unknown>).lod_bounds = { min: [0, 0, 0], max: [1, 1, 1] };
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    try {
      await loadLodGroupNode(
        makeLodGroupNode([child]),
        new THREE.Group(),
        makeStubLoc(),
        ctx,
        loadSceneNodesMock
      );

      expect(reg.get('/lod')!.children[0].lodBounds).toBeUndefined();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.anything(),
        'lod_group child /lod/child_0: rejected lod_bounds; ' +
          'child has no usable position_bounds to validate against'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });
});

// ────────────────────────────────────────────────────────────────────────
// Lazy loading: only the default level loads eagerly; the rest defer
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — lazy level loading', () => {
  function makeReg(): LODGroupRegistry {
    return new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
  }

  it('eager-loads only the default level and defers the other gsplats levels', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0),
        makeChildNode('/lod/child_1', 0.5),
        makeChildNode('/lod/child_2', 1.0),
      ],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    // Default level (index 0) goes through the eager recursion; the
    // other two are cheap-attached and deferred.
    expect(loadSceneNodesMock).toHaveBeenCalledTimes(1);
    expect(loadGSplatsNodeCheapMock).toHaveBeenCalledTimes(2);
    // No geometry fetched for deferred levels until the selector wants them.
    expect(loadGSplatsNodeExpensiveMock).not.toHaveBeenCalled();

    const entry = reg.get('/lod')!;
    expect(entry.children[0].ready).not.toBe(false); // eager → ready
    expect(entry.children[1].ready).toBe(false);
    expect(entry.children[2].ready).toBe(false);
    expect(typeof entry.children[1].ensureLoaded).toBe('function');
    expect(typeof entry.children[2].ensureLoaded).toBe('function');
  });

  it('ensureLoaded runs the expensive load once and marks the child ready', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    expect(deferred.ready).toBe(false);

    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.ready).toBe(true));

    expect(loadGSplatsNodeExpensiveMock).toHaveBeenCalledTimes(1);
    expect(deferred.loading).toBe(false);
    expect(deferred.failed).toBeUndefined();
    // A lazy LEAF level is registry-driven (hasMoreLODs thunk) — it must NOT
    // kick the sweep refinement orchestrator (that's the deferred-GROUP path).
    expect(ctx.kickRefinementIfIdle).not.toHaveBeenCalled();
  });

  // Four-way symmetry: a nested kind=partition / kind=lod wrapper defers
  // identically regardless of the inner geometry. loadLodGroupNode never reads
  // `display_type` here at all — the deferral branches on the wrapper node's
  // `type === 'group'` and its `kind`, never on the leaf type underneath — and
  // that geometry-agnosticism is the invariant this parametrization guards.
  // The mesh row is a real authored shape, not a hypothetical:
  // `add_lod_group(…).add_partition_group(display_type='mesh',
  // coverage_fraction=…)` filled with `add_mesh` parts writes exactly the node
  // below — a kind=partition wrapper carrying the threshold, under a kind=lod
  // parent the writer back-fills to display_type='mesh'. (The shorthand
  // `lod.add_mesh(…, partition=…)` writes the same tree but puts
  // `coverage_fraction` on the PARTS, not the wrapper, so it is the explicit
  // route above that produces the shape this test models.)
  it.each(['gsplats', 'points', 'lines', 'mesh'] as const)(
    'defers a non-leaf group child (display_type=%s) and loads its subtree on activation',
    async (displayType) => {
      attachStubChildren();
      const reg = makeReg();
      const ctx = makeCtx(reg);

      // child_0 = eager leaf (default); child_1 = a kind=partition group.
      const node = makeLodGroupNode(
        [makeChildNode('/lod/child_0', 0), makeGroupChildNode('/lod/child_1', 0.5, displayType)],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      const entry = reg.get('/lod')!;
      expect(entry.children).toHaveLength(2);
      const groupChild = entry.children[1];
      // Registered with its threshold/bounds, but NOT loaded.
      expect(groupChild.coverageFraction).toBe(0.5);
      expect(groupChild.positionBounds).toEqual({ min: [0, 0, 0], max: [1, 1, 1] });
      expect(groupChild.ready).toBe(false);
      expect(typeof groupChild.ensureLoaded).toBe('function');
      // Grouped subtrees aren't evictable (no leaf-style pool) → no release thunk.
      expect(groupChild.release).toBeUndefined();
      // The transparent wrapper must not steal the child's identity — only the
      // real node (attached on load) owns the path/kind.
      expect(groupChild.object.name).toBe('');
      expect(groupChild.object.userData.kind).toBeUndefined();
      // At init, loadChildren ran ONLY for the eager default — NOT the group child.
      const initPaths = loadSceneNodesMock.mock.calls.map((c) => (c[0] as SceneNode).path);
      expect(initPaths).toContain('/lod/child_0');
      expect(initPaths).not.toContain('/lod/child_1');

      // Activation loads the whole subtree (loadChildren on the group), marks ready.
      expect(ctx.kickRefinementIfIdle).not.toHaveBeenCalled(); // not during init
      groupChild.ensureLoaded!();
      await vi.waitFor(() => expect(groupChild.ready).toBe(true));
      const afterPaths = loadSceneNodesMock.mock.calls.map((c) => (c[0] as SceneNode).path);
      expect(afterPaths).toContain('/lod/child_1');
      // The subtree's leaves registered into the sweep maps mid-session;
      // refinement is only scheduled at update-view tails, so the activation
      // must kick the orchestrator or the branch stalls at chunk-1 per part.
      expect(ctx.kickRefinementIfIdle).toHaveBeenCalledTimes(1);
    }
  );

  it('activating a deferred GROUP child a second time loads nothing (no duplicate subtree)', async () => {
    // `loadChildren` attaches a fresh THREE.Group on every call, so a second
    // `ensureLoaded` (a stale re-kick, an explicit retry after success) would
    // hang a second full copy of the subtree under the placeholder: doubled
    // geometry, duplicate names, re-registered loaders, leaked buffers (#2632).
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeGroupChildNode('/lod/child_1', 0.5, 'gsplats')],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    const groupChild = reg.get('/lod')!.children[1];

    groupChild.ensureLoaded!();
    await vi.waitFor(() => expect(groupChild.ready).toBe(true));
    const loadsAfterFirst = loadSceneNodesMock.mock.calls.filter(
      (c) => (c[0] as SceneNode).path === '/lod/child_1'
    ).length;
    expect(loadsAfterFirst).toBe(1);

    groupChild.loading = false; // what the registry's finally does
    groupChild.ensureLoaded!();
    await vi.waitFor(() => expect(groupChild.loading).toBe(false));
    const loadsAfterSecond = loadSceneNodesMock.mock.calls.filter(
      (c) => (c[0] as SceneNode).path === '/lod/child_1'
    ).length;
    expect(loadsAfterSecond).toBe(1);
    expect(groupChild.ready).toBe(true);
    // No second refinement kick either: nothing new registered.
    expect(ctx.kickRefinementIfIdle).toHaveBeenCalledTimes(1);
  });

  it('does not retry a deferred GROUP child after a failed load partially attached its subtree', async () => {
    const archiveFault = new ArchiveFaultError('archive is no longer readable', '/scene.zip');
    let groupLoads = 0;
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
      if (child.path === '/lod/child_1' && groupLoads++ === 0) throw archiveFault;
    });
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [makeChildNode('/lod/child_0', 0), makeGroupChildNode('/lod/child_1', 0.5, 'gsplats')],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
      const groupChild = reg.get('/lod')!.children[1];

      groupChild.ensureLoaded!();
      await vi.waitFor(() => expect(groupChild.permanentlyFailed).toBe(true));
      expect(groupChild.object.children).toHaveLength(1);
      expect(groupLoads).toBe(1);

      expect(reg.retryLazyChildByNodePath('/lod/child_1')).toBe(true);
      await vi.waitFor(() => expect(groupChild.loading).toBe(false));
      expect(groupLoads).toBe(1);
      expect(groupChild.object.children).toHaveLength(1);
      expect(groupChild.ready).toBe(false);
      expect(groupChild.permanentlyFailed).toBe(true);
      expect(groupChild.failureReason).toBeTruthy();
      expect(reg.getFailedLazyChildPaths()).toEqual(['/lod/child_1']);
      expect(ctx.kickRefinementIfIdle).not.toHaveBeenCalled();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.anything(),
        expect.stringContaining('already has attached children')
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('re-sorts children to ascending coverage_fraction (and warns) when the order is wrong', async () => {
    // Defense-in-depth for malformed / hand-authored scenes: the selector assumes
    // ascending thresholds. Given out-of-order thresholds (0, 1.0, 0.5), the loader
    // must repair to ascending so the selector works, keep the eager default active,
    // and warn so the producer bug is surfaced.
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [
          makeChildNode('/lod/child_0', 0), // eager default + coarsest
          makeChildNode('/lod/child_1', 1.0),
          makeChildNode('/lod/child_2', 0.5), // out of order (< 1.0)
        ],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      const entry = reg.get('/lod')!;
      // Repaired to strictly ascending so pickChildWithHysteresis is well-defined.
      expect(entry.children.map((c) => c.coverageFraction)).toEqual([0, 0.5, 1.0]);
      // The eager default (coverage_fraction=0) is still the active level after the re-sort.
      expect(entry.children[entry.activeChildIndex].coverageFraction).toBe(0);
      // The violation was surfaced.
      expect(warnSpy.mock.calls.some((c) => String(c[1]).includes('not strictly ascending'))).toBe(
        true
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does not warn or reorder when child thresholds are already ascending', async () => {
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [
          makeChildNode('/lod/child_0', 0),
          makeChildNode('/lod/child_1', 0.5),
          makeChildNode('/lod/child_2', 1.0),
        ],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      expect(reg.get('/lod')!.children.map((c) => c.coverageFraction)).toEqual([0, 0.5, 1.0]);
      expect(warnSpy.mock.calls.some((c) => String(c[1]).includes('not strictly ascending'))).toBe(
        false
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('warns on EQUAL adjacent thresholds (not strictly ascending), matching the Python invariant', async () => {
    // Python's `_assert_strict_ascending` forbids equal thresholds (zero-width
    // hysteresis band between two levels). The viewer's order check is therefore
    // strict (`<`, not `<=`): equal adjacent thresholds must surface the same
    // producer-bug warning. The stable sort leaves the equal pair in place.
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [
          makeChildNode('/lod/child_0', 0),
          makeChildNode('/lod/child_1', 0.5),
          makeChildNode('/lod/child_2', 0.5), // equal to previous → not strictly ascending
        ],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      expect(warnSpy.mock.calls.some((c) => String(c[1]).includes('not strictly ascending'))).toBe(
        true
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('keeps a deferred GROUP child correct through a threshold re-sort (FIX A × FIX B)', async () => {
    // Interaction of the two new behaviours: a nested-group child is deferred
    // (FIX A) AND the input thresholds are out of order (FIX B). The sort must
    // reorder the deferred group child by its threshold WITHOUT loading it or
    // losing its deferred state, and the eager default must stay active.
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      // child_0 leaf (eager default, coverage_fraction=0); child_1 a deferred
      // PARTITION group (cf=1.0); child_2 leaf (cf=0.5) — out of order (0, 1.0, 0.5).
      const node = makeLodGroupNode(
        [
          makeChildNode('/lod/child_0', 0),
          makeGroupChildNode('/lod/child_1', 1.0),
          makeChildNode('/lod/child_2', 0.5),
        ],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      const entry = reg.get('/lod')!;
      // Re-sorted ascending; the deferred group child lands at its threshold slot.
      expect(entry.children.map((c) => c.coverageFraction)).toEqual([0, 0.5, 1.0]);
      const groupChild = entry.children[2];
      expect(groupChild.coverageFraction).toBe(1.0);
      // Sort must NOT have loaded or lost the deferred state of the group child.
      expect(groupChild.ready).toBe(false);
      expect(typeof groupChild.ensureLoaded).toBe('function');
      expect(groupChild.release).toBeUndefined();
      // The eager default (coverage_fraction=0) is still the active level after the re-sort.
      expect(entry.children[entry.activeChildIndex].coverageFraction).toBe(0);
      // The group child was NOT eager-loaded at init...
      expect(loadSceneNodesMock.mock.calls.map((c) => (c[0] as SceneNode).path)).not.toContain(
        '/lod/child_1'
      );
      // ...but loads its subtree on activation (post-sort object identity intact).
      groupChild.ensureLoaded!();
      await vi.waitFor(() => expect(groupChild.ready).toBe(true));
      expect(loadSceneNodesMock.mock.calls.map((c) => (c[0] as SceneNode).path)).toContain(
        '/lod/child_1'
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('does not register or mark ready when the dataset is switched mid-load', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    expect(deferred.ready).toBe(false);

    // Simulate a dataset switch completing while the deferred load is in
    // flight: the expensive load resolves (commit skipped internally) but the
    // thunk must NOT re-register the loader or mark the geometry-less level
    // ready.
    ctx.isDatasetLive = () => false;
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.loading).toBe(false));

    expect(ctx.registry.registerGSplatsLoader).not.toHaveBeenCalled();
    expect(deferred.ready).toBe(false);
    expect(deferred.failed).toBeUndefined();
  });

  it('gives deferred levels a release thunk that frees the buffer and resets readiness', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ready = true; // simulate a completed load
    deferred.failed = true; // simulate a stale failure flag from a prior cycle
    deferred.failedTick = 42;
    expect(typeof deferred.release).toBe('function');

    deferred.release!();
    expect(vi.mocked(ctx.releaseLazyGSplats)).toHaveBeenCalledWith('/lod/child_1');
    expect(deferred.ready).toBe(false);
    expect(deferred.loading).toBe(false);
    expect(deferred.failed).toBe(false);
    // The failure cooldown is also cleared so a reload starts fresh.
    expect(deferred.failedTick).toBeUndefined();
  });

  it('does not give the eager default level a release thunk', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(reg.get('/lod')!.children[0].release).toBeUndefined();
  });

  it('marks the child failed (not ready) when the deferred load throws', async () => {
    attachStubChildren();
    loadGSplatsNodeExpensiveMock.mockRejectedValue(
      new Error('outer load failure', { cause: new Error('boom') })
    );
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.failed).toBe(true));

    expect(deferred.ready).toBe(false);
    expect(deferred.loading).toBe(false);
    expect(deferred.permanentlyFailed).not.toBe(true);
  });

  it('latches a wrapped archive fault as a permanent lazy-level failure', async () => {
    attachStubChildren();
    const archiveFault = new ArchiveFaultError('archive is no longer readable', '/scene.zip');
    loadGSplatsNodeExpensiveMock.mockRejectedValue(
      new Error('outer load failure', {
        cause: new LoaderError('Network', '/lod/child_1', archiveFault),
      })
    );
    const reg = makeReg();
    const reportArchiveFault = vi.fn();
    const ctx = makeCtx(reg, { reportArchiveFault });

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.failed).toBe(true));

    expect(deferred.permanentlyFailed).toBe(true);
    expect(deferred.ready).toBe(false);
    expect(deferred.loading).toBe(false);

    deferred.ready = true;
    deferred.release!();
    expect(deferred.ready).toBe(false);
    expect(deferred.failed).toBe(true);
    expect(deferred.permanentlyFailed).toBe(true);
    expect(reportArchiveFault).toHaveBeenCalledOnce();
    expect(reportArchiveFault).toHaveBeenCalledWith(archiveFault);
  });

  it('does not report a lazy-level archive fault after its dataset is replaced', async () => {
    attachStubChildren();
    const archiveFault = new ArchiveFaultError('archive is no longer readable', '/scene.zip');
    loadGSplatsNodeExpensiveMock.mockRejectedValue(archiveFault);
    const reg = makeReg();
    const reportArchiveFault = vi.fn();
    const ctx = makeCtx(reg, { isDatasetLive: () => false, reportArchiveFault });

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.permanentlyFailed).toBe(true));

    expect(reportArchiveFault).not.toHaveBeenCalled();
  });

  it('latches and reports an archive fault from an anonymous deferred GROUP', async () => {
    attachStubChildren();
    const archiveFault = new ArchiveFaultError('archive open failed', '/scene.zip');
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      if (child.path === '/lod/child_1') throw archiveFault;
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
    });
    const reg = makeReg();
    const reportArchiveFault = vi.fn();
    const ctx = makeCtx(reg, { reportArchiveFault });
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeGroupChildNode('/lod/child_1', 0.5)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.failed).toBe(true));

    expect(deferred.object.name).toBe('');
    expect(deferred.nodePath).toBe('/lod/child_1');
    expect(deferred.permanentlyFailed).toBe(true);
    expect(reportArchiveFault).toHaveBeenCalledOnce();
    expect(reportArchiveFault).toHaveBeenCalledWith(archiveFault);
    expect(reg.getFailedLazyChildPaths()).toEqual(['/lod/child_1']);
  });

  it('keeps an anonymous deferred GROUP recoverable after an ordinary error', async () => {
    const failure = new Error('nested subtree failed');
    attachStubChildren();
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      if (child.path === '/lod/child_1') throw failure;
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
    });
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [makeChildNode('/lod/child_0', 0), makeGroupChildNode('/lod/child_1', 0.5)],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      const deferred = reg.get('/lod')!.children[1];
      deferred.ensureLoaded!();
      await vi.waitFor(() => expect(deferred.failed).toBe(true));

      expect(deferred.object.name).toBe('');
      expect(deferred.release).toBeUndefined();
      expect(deferred.loading).toBe(false);
      expect(deferred.permanentlyFailed).not.toBe(true);
      expect(reg.retryLazyChildByNodePath('')).toBe(false);

      reg.setSelectorMode('/lod', { lockLevel: 1 });
      reg.evaluatePerFrame();
      for (let frame = 0; frame < 121; frame++) reg.evaluatePerFrame();
      await vi.waitFor(() => expect(deferred.failed).toBe(true));
      expect(deferred.loading).toBe(false);

      const attempts = loadSceneNodesMock.mock.calls.filter(
        ([loadedChild]) => (loadedChild as SceneNode).path === '/lod/child_1'
      );
      expect(attempts).toHaveLength(2);
    } finally {
      warningSpy.mockRestore();
    }
  });
});

// ────────────────────────────────────────────────────────────────────────
// Lazy loading: a points finest child (points-substitutive ladder) defers too
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — lazy points level loading', () => {
  function makeReg(): LODGroupRegistry {
    return new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
  }

  it('defers a non-default points child via the points cheap split (not eager)', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    // A points-substitutive ladder: coarse gsplat default + finest points child.
    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0), // coarsest gsplat (default/eager)
        makeChildNode('/lod/child_1', 0.5), // mid gsplat
        makePointsChildNode('/lod/child_2', 1.0), // finest = points
      ],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    // The eager default (child_0) recurses; child_1 (gsplats) + child_2 (points)
    // both cheap-attach. The points child must NOT take the eager loadSceneNodes
    // path (which would fetch the full cloud up front).
    expect(loadSceneNodesMock).toHaveBeenCalledTimes(1);
    expect(loadGSplatsNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadPointsNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadPointsNodeExpensiveMock).not.toHaveBeenCalled();

    const pts = reg.get('/lod')!.children[2];
    expect(pts.ready).toBe(false);
    expect(typeof pts.ensureLoaded).toBe('function');
  });

  it('ensureLoaded loads the points level WITHOUT joining the per-slice sweep (decoupled)', async () => {
    // Decoupling (B2): a lazy fine level commits independently in its expensive
    // loader and is NOT registered into the per-slice sweep — so the cheap
    // coarse level can commit a new timepoint without being gated behind it.
    // The registry reloads the fine level on a settled slice change instead.
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const pts = reg.get('/lod')!.children[1];
    pts.ensureLoaded!();
    await vi.waitFor(() => expect(pts.ready).toBe(true));

    expect(loadPointsNodeExpensiveMock).toHaveBeenCalledTimes(1);
    expect(ctx.registry.registerPointsLoader).not.toHaveBeenCalled();
    expect(pts.loading).toBe(false);
  });

  it('does not register the points level when the dataset is switched mid-load', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const pts = reg.get('/lod')!.children[1];
    ctx.isDatasetLive = () => false;
    pts.ensureLoaded!();
    await vi.waitFor(() => expect(pts.loading).toBe(false));

    expect(ctx.registry.registerPointsLoader).not.toHaveBeenCalled();
    expect(pts.ready).toBe(false);
  });

  it('gives a deferred points level a release thunk that evicts + resets readiness', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ready = true; // simulate a completed load
    deferred.failed = true;
    deferred.failedTick = 42;
    expect(typeof deferred.release).toBe('function');

    deferred.release!();
    expect(vi.mocked(ctx.releaseLazyPoints)).toHaveBeenCalledWith('/lod/child_1');
    expect(deferred.ready).toBe(false);
    expect(deferred.loading).toBe(false);
    expect(deferred.failed).toBe(false);
    expect(deferred.failedTick).toBeUndefined();
  });

  it('surfaces the progressive points loader hasMoreLODs on the lazy level (composed ladder advances)', async () => {
    // The composed shape: the finest lod level is a points node that is ALSO
    // additively laddered (`n_additive_sublods: 4`) — a substitutive level that
    // is itself progressive. A lazy lod child is deliberately never registered
    // into the per-slice update sweep, so the registry's `hasMoreLODs` probe is
    // the ONLY thing that re-fires `ensureLoaded` to walk the remaining
    // sub-LODs. Without this wiring the ladder stalls at 2 of 4 levels forever.
    const loaderStub = { hasMoreLODs: true };
    loadPointsNodeCheapMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
      return { placeholder: mesh, loader: loaderStub as never };
    });
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const finest = makePointsChildNode('/lod/child_2', 1.0);
    finest.attrs.n_additive_sublods = 4;
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5), finest],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    // Still deferred: an additive ladder does not make the level eager.
    expect(loadPointsNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadPointsNodeExpensiveMock).not.toHaveBeenCalled();

    const pts = reg.get('/lod')!.children[2];
    expect(typeof pts.hasMoreLODs).toBe('function');
    // Live probe (not a snapshot taken at attach time): it must track the
    // loader as the ladder streams.
    expect(pts.hasMoreLODs!()).toBe(true);
    loaderStub.hasMoreLODs = false;
    expect(pts.hasMoreLODs!()).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Lazy loading: a lines finest child (lines-substitutive ladder) defers too
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — lazy lines level loading', () => {
  function makeReg(): LODGroupRegistry {
    return new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
  }

  it('shares eager line admission across sibling substitutive ladders', async () => {
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const started: string[] = [];
    const releases: Array<() => void> = [];
    loadSceneNodesMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      started.push(child.path);
      await new Promise<void>((resolve) => releases.push(resolve));
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
    });

    const nodes = Array.from({ length: 2 }, (_, index) => {
      const child = makeLinesChildNode(`/lod_${index}/child_0`, 0);
      child.attrs.n_vertices = 1_630_000;
      child.attrs.n_segments = 1_280_000;
      child.attrs.ndim = 3;
      const node = makeLodGroupNode([child], { default_level: 0, display_type: 'lines' });
      node.path = `/lod_${index}`;
      return node;
    });

    const loadPromise = Promise.all(
      nodes.map((node) =>
        loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock)
      )
    );

    await vi.waitFor(() => expect(started).toHaveLength(1));
    releases.shift()?.();
    await vi.waitFor(() => expect(started).toHaveLength(2));
    releases.shift()?.();
    await loadPromise;
  });

  it('releases eager line admission when a ladder load rejects', async () => {
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const makeOversizedLadder = (index: number) => {
      const child = makeLinesChildNode(`/lod_${index}/child_0`, 0);
      child.attrs.n_vertices = 1_630_000;
      child.attrs.n_segments = 1_280_000;
      child.attrs.ndim = 3;
      const node = makeLodGroupNode([child], { default_level: 0, display_type: 'lines' });
      node.path = `/lod_${index}`;
      return node;
    };

    loadSceneNodesMock.mockRejectedValueOnce(new Error('line load failed'));
    await expect(
      loadLodGroupNode(
        makeOversizedLadder(0),
        new THREE.Group(),
        makeStubLoc(),
        ctx,
        loadSceneNodesMock
      )
    ).rejects.toThrow('line load failed');

    let secondStarted = false;
    loadSceneNodesMock.mockImplementationOnce(async (child: SceneNode, parent: THREE.Object3D) => {
      secondStarted = true;
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
    });
    const secondLoad = loadLodGroupNode(
      makeOversizedLadder(1),
      new THREE.Group(),
      makeStubLoc(),
      ctx,
      loadSceneNodesMock
    );

    await vi.waitFor(() => expect(secondStarted).toBe(true));
    await secondLoad;
  });

  it('defers a non-default lines child via the lines cheap split (not eager)', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0), // coarsest gsplat (default/eager)
        makeLinesChildNode('/lod/child_1', 0.5), // finest = lines
      ],
      { default_level: 0, display_type: 'lines' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(loadSceneNodesMock).toHaveBeenCalledTimes(1); // only the eager default
    expect(loadLinesNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadLinesNodeExpensiveMock).not.toHaveBeenCalled();
    expect(reg.get('/lod')!.children[1].ready).toBe(false);
  });

  it('ensureLoaded loads the lines level WITHOUT joining the per-slice sweep (decoupled)', async () => {
    // See the points peer above — lazy fine levels commit independently and are
    // not registered into the per-slice sweep (the coarse/fine commit decoupling).
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeLinesChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'lines' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const ln = reg.get('/lod')!.children[1];
    ln.ensureLoaded!();
    await vi.waitFor(() => expect(ln.ready).toBe(true));
    expect(loadLinesNodeExpensiveMock).toHaveBeenCalledTimes(1);
    expect(ctx.registry.registerLinesLoader).not.toHaveBeenCalled();
    expect(ln.loading).toBe(false);
  });

  it('gives a deferred lines level a release thunk that evicts + resets readiness', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeLinesChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'lines' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const ln = reg.get('/lod')!.children[1];
    ln.ready = true; // simulate a completed load
    ln.failed = true; // simulate a stale failure flag from a prior cycle
    ln.failedTick = 42;
    expect(typeof ln.release).toBe('function');
    ln.release!();
    expect(vi.mocked(ctx.releaseLazyLines)).toHaveBeenCalledWith('/lod/child_1');
    expect(ln.ready).toBe(false);
    expect(ln.loading).toBe(false);
    expect(ln.failed).toBe(false);
    expect(ln.failedTick).toBeUndefined();
  });

  it('does not register the lines level when the dataset is switched mid-load', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeLinesChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'lines' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const ln = reg.get('/lod')!.children[1];
    ctx.isDatasetLive = () => false;
    ln.ensureLoaded!();
    await vi.waitFor(() => expect(ln.loading).toBe(false));
    expect(ctx.registry.registerLinesLoader).not.toHaveBeenCalled();
    expect(ln.ready).toBe(false);
  });

  it('surfaces the progressive lines loader hasMoreLODs on the lazy level (composed ladder advances)', async () => {
    // Symmetry mirror of the points peer above — a lines finest level that is
    // also additively laddered advances only via the registry's live
    // `hasMoreLODs` probe (lazy levels never join the per-slice sweep).
    const loaderStub = { hasMoreLODs: true };
    loadLinesNodeCheapMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
      return { placeholder: mesh, loader: loaderStub as never };
    });
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const finest = makeLinesChildNode('/lod/child_2', 1.0);
    finest.attrs.n_additive_sublods = 4;
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 0.5), finest],
      { default_level: 0, display_type: 'lines' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(loadLinesNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadLinesNodeExpensiveMock).not.toHaveBeenCalled();

    const ln2 = reg.get('/lod')!.children[2];
    expect(typeof ln2.hasMoreLODs).toBe('function');
    expect(ln2.hasMoreLODs!()).toBe(true);
    loaderStub.hasMoreLODs = false;
    expect(ln2.hasMoreLODs!()).toBe(false);
  });

  it('surfaces the progressive MESH loader hasMoreLODs on the lazy level (reveal advances)', async () => {
    // The composed shape for mesh: a substitutive level that is ITSELF a reveal
    // ladder. A lazy lod child is deliberately never registered into the
    // per-slice sweep, so the registry's `hasMoreLODs` probe is the only thing
    // that re-fires `ensureLoaded` to walk the remaining sub-LODs. This branch
    // used to pass `undefined` — correct while a mesh level was whole-node
    // resident and therefore complete the moment it was ready, and silently
    // wrong the day mesh gained a ladder: the level would freeze at its first
    // patch with nothing in the logs to say why.
    const loaderStub = { hasMoreLODs: true };
    loadMeshNodeCheapMock.mockImplementation(async (child: SceneNode, parent: THREE.Object3D) => {
      const mesh = new THREE.Mesh();
      mesh.name = child.path;
      parent.add(mesh);
      return { placeholder: mesh, loader: loaderStub as never };
    });
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const finest = makeMeshChildNode('/lod/child_1', 1.0);
    finest.attrs.n_additive_sublods = 4;
    const node = makeLodGroupNode([makeChildNode('/lod/child_0', 0), finest], {
      default_level: 0,
      display_type: 'mesh',
    });
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const meshChild = reg.get('/lod')!.children[1];
    expect(typeof meshChild.hasMoreLODs).toBe('function');
    // A LIVE probe, not a snapshot taken at attach time: it has to track the
    // loader as the reveal streams, or the registry stops re-firing one pass early.
    expect(meshChild.hasMoreLODs!()).toBe(true);
    loaderStub.hasMoreLODs = false;
    expect(meshChild.hasMoreLODs!()).toBe(false);
  });

  it('defers a mesh level through the cheap/expensive split (the fourth dispatch branch)', async () => {
    // The dispatch throws for any deferrable type it does not name, so this is
    // the positive control the release test below needs — without it, a broken
    // dispatch and a missing release are indistinguishable.
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeMeshChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'mesh' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(loadMeshNodeCheapMock).toHaveBeenCalledTimes(1);
    expect(loadMeshNodeExpensiveMock).not.toHaveBeenCalled();
    // Sibling-symmetric with the gsplats/points/lines defer tests above: a lazy
    // mesh level must never land in `registry.meshLoaders`. One that does joins
    // the scene-wide per-slice sweep, which re-projects and re-commits the whole
    // full-resolution surface on every scrub — including while the level is
    // hidden — and gates the cheap coarse level's commit behind it (#1356).
    // SCOPE, so this is not read for more than it is: load-mesh-node is
    // module-mocked here, so these two assertions pin only that the LOD-GROUP
    // side never registers. The loader side — `loadMeshNodeExpensive` does not
    // register, `loadMeshNode` does — is what actually guards #1356, and it is
    // pinned against the real functions and a real registry in
    // load-mesh-node.test.ts.
    expect(ctx.registry.registerMeshLoader).not.toHaveBeenCalled();

    const ln = reg.get('/lod')!.children[1];
    ln.ensureLoaded!();
    await vi.waitFor(() => expect(ln.ready).toBe(true));
    expect(loadMeshNodeExpensiveMock).toHaveBeenCalledTimes(1);
    // Re-checked after a full activation to `ready`, and not redundant: the
    // pre-activation assertion cannot see a register call made from inside the
    // `attachLazyChild` thunk, which is where the wrapper would most plausibly
    // regress. The invariant spans the level's whole lifetime.
    expect(ctx.registry.registerMeshLoader).not.toHaveBeenCalled();
  });

  it('gives a deferred mesh level a release thunk, even though it has no pooled buffer', async () => {
    // The asymmetry worth pinning: a mesh is `pooled: false`, so this release
    // hands nothing back to the evictable pool — which is exactly why it was
    // originally omitted. It is needed anyway because mesh is `depthSortable`
    // (#1347): `releaseLazyMesh` drops the demoted level's depth-sort state and
    // its worker-side centroids. Assert the thunk EXISTS and is wired, so the
    // "no pooled buffer" reasoning cannot re-delete it.
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeMeshChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'mesh' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const ln = reg.get('/lod')!.children[1];
    ln.ready = true; // simulate a completed load
    ln.failed = true; // stale failure flag from a prior cycle
    ln.failedTick = 42;
    expect(typeof ln.release).toBe('function');
    ln.release!();
    expect(vi.mocked(ctx.releaseLazyMesh)).toHaveBeenCalledWith('/lod/child_1');
    expect(ln.ready).toBe(false);
    expect(ln.loading).toBe(false);
    expect(ln.failed).toBe(false);
    expect(ln.failedTick).toBeUndefined();
  });

  it('a deferred mesh level with NO ladder reports hasMoreLODs false, not undefined', async () => {
    // NARROWED, not deleted (#1476). This used to assert `undefined` — the mesh
    // row's asymmetry, on the grounds that a surface could not have an additive
    // ladder at all. Mesh has one now (a reveal, not a level of detail), so the
    // probe is passed like every other type and what survives is the weaker,
    // still-meaningful claim: an UNLADDERED level reports no further work, so the
    // registry does not re-fire `ensureLoaded` for a level that is already whole.
    //
    // The default `loadMeshNodeCheapMock` returns a loader with no `hasMoreLODs`
    // property at all — a plain `MeshWholeNodeLoader` — which is exactly the
    // structural probe's negative case.
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);
    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeMeshChildNode('/lod/child_1', 0.5)],
      { default_level: 0, display_type: 'mesh' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const meshChild = reg.get('/lod')!.children[1];
    expect(typeof meshChild.hasMoreLODs).toBe('function');
    expect(meshChild.hasMoreLODs!()).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Legacy (pre-v3.2) selector attrs: min_pixel_size → derived coverage
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — legacy min_pixel_size adaptation', () => {
  function makeReg(): LODGroupRegistry {
    return new LODGroupRegistry({
      getCamera: () => new THREE.Camera(),
      getViewportSize: () => ({ width: 100, height: 100 }),
      getDisplayDims: () => [0, 1, 2],
    });
  }

  /** A pre-v3.2 child: carries ``min_pixel_size``, no ``coverage_fraction``. */
  function makeLegacyChildNode(path: string, minPixelSize: number): SceneNode {
    return {
      path,
      type: 'gsplats',
      attrs: {
        type: 'gsplats',
        min_pixel_size: minPixelSize,
        position_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
      } as SceneNode['attrs'],
      hasSpatialIndex: false,
      children: [],
    };
  }

  it('derives coverage fractions from legacy min_pixel_size (normalized by finest) and warns once', async () => {
    // A pre-v3.2 dataset (selector='pixel_size', per-child min_pixel_size).
    // Silently defaulting every threshold to 0 would pin the selector to the
    // FINEST child (eager full-res download, progressive LOD defeated) — the
    // loader must instead derive coverage fractions by normalizing the legacy
    // pixel ladder by its finest value: [100, 200, 400] → [0.25, 0.5, 1.0].
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const node = makeLodGroupNode(
        [
          makeLegacyChildNode('/lod/child_0', 100),
          makeLegacyChildNode('/lod/child_1', 200),
          makeLegacyChildNode('/lod/child_2', 400),
        ],
        { selector: 'pixel_size', default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      const entry = reg.get('/lod')!;
      expect(entry.children.map((c) => c.coverageFraction)).toEqual([0.25, 0.5, 1.0]);
      // The eager default (coarsest) is still the active level.
      expect(entry.activeChildIndex).toBe(0);
      // One clear adaptation warning naming the migration path…
      const adaptWarnings = warnSpy.mock.calls.filter((c) =>
        String(c[1]).includes('migrate-format')
      );
      expect(adaptWarnings).toHaveLength(1);
      expect(String(adaptWarnings[0][1])).toContain('min_pixel_size');
      // …and NOT the misleading producer-blaming order warning (the derived
      // ladder is strictly ascending, so the defense-in-depth check passes).
      expect(warnSpy.mock.calls.some((c) => String(c[1]).includes('not strictly ascending'))).toBe(
        false
      );
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('prefers an authored coverage_fraction over min_pixel_size in a mixed ladder', async () => {
    attachStubChildren();
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      // child_1 was re-authored with the current attr; the two legacy siblings
      // still normalize by the finest LEGACY value (400 → finest 1.0).
      const node = makeLodGroupNode(
        [
          makeLegacyChildNode('/lod/child_0', 100),
          makeChildNode('/lod/child_1', 0.5),
          makeLegacyChildNode('/lod/child_2', 400),
        ],
        { default_level: 0 }
      );
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      expect(reg.get('/lod')!.children.map((c) => c.coverageFraction)).toEqual([0.25, 0.5, 1.0]);
      expect(warnSpy.mock.calls.some((c) => String(c[1]).includes('migrate-format'))).toBe(true);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('logs an actionable error (naming migrate-format) when children carry no threshold at all', async () => {
    // Neither coverage_fraction nor min_pixel_size: a malformed producer
    // output. Previously this silently defaulted to 0 and then mis-blamed the
    // producer with only a 'not strictly ascending' warning.
    attachStubChildren();
    const errorSpy = vi.spyOn(log, 'error').mockImplementation(() => {});
    const warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    try {
      const reg = makeReg();
      const ctx = makeCtx(reg);
      const bareChild = (path: string): SceneNode => ({
        path,
        type: 'gsplats',
        attrs: {
          type: 'gsplats',
          position_bounds: { min: [0, 0, 0], max: [1, 1, 1] },
        } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [],
      });
      const node = makeLodGroupNode([bareChild('/lod/child_0'), bareChild('/lod/child_1')], {
        default_level: 0,
      });
      await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

      // Still registers (renders something) with the 0-defaults…
      expect(reg.get('/lod')!.children.map((c) => c.coverageFraction)).toEqual([0, 0]);
      // …but the failure is loud and actionable.
      const errors = errorSpy.mock.calls.filter((c) => String(c[1]).includes('migrate-format'));
      expect(errors).toHaveLength(1);
      expect(String(errors[0][1])).toContain('/lod/child_0');
      expect(String(errors[0][1])).toContain('/lod/child_1');
    } finally {
      errorSpy.mockRestore();
      warnSpy.mockRestore();
    }
  });
});

// ────────────────────────────────────────────────────────────────────────
// Fallback behaviour when no registry is wired
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — without a registry', () => {
  it('still loads children but only the default level is visible', async () => {
    attachStubChildren();
    const ctx = makeCtx(/* no registry */);

    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0),
        makeChildNode('/lod/child_1', 0.5),
        makeChildNode('/lod/child_2', 1.0),
      ],
      { default_level: 1 }
    );

    const parent = new THREE.Group();
    const lodGroup = await loadLodGroupNode(node, parent, makeStubLoc(), ctx, loadSceneNodesMock);

    // Without a registry we fall back to visibility set inline.
    expect(lodGroup.children[0].visible).toBe(false);
    expect(lodGroup.children[1].visible).toBe(true);
    expect(lodGroup.children[2].visible).toBe(false);
  });

  it('keeps the eager default active when an earlier child fails to attach', async () => {
    // child_0 fails to attach → dropped from the registry list, shifting
    // indices. The eager default (child_1) must remain the active/visible
    // level, not the index-1 survivor (child_2). Regression: previously the
    // default level was recomputed from `default_level` over the shortened
    // list, mis-pointing it after a drop.
    loadSceneNodesMock.mockImplementation(async (n: SceneNode, parent: THREE.Object3D) => {
      if (n.path === '/lod/child_0') return; // simulate attach failure
      const mesh = new THREE.Mesh();
      mesh.name = n.path;
      parent.add(mesh);
    });
    const ctx = makeCtx(/* no registry → all children eager */);

    const node = makeLodGroupNode(
      [
        makeChildNode('/lod/child_0', 0),
        makeChildNode('/lod/child_1', 0.5),
        makeChildNode('/lod/child_2', 1.0),
      ],
      { default_level: 1 }
    );

    const parent = new THREE.Group();
    const lodGroup = await loadLodGroupNode(node, parent, makeStubLoc(), ctx, loadSceneNodesMock);

    // child_0 dropped → survivors [child_1, child_2]; the eager default
    // (child_1) sits at index 0 and is the visible one.
    expect(lodGroup.children.map((c) => c.name)).toEqual(['/lod/child_1', '/lod/child_2']);
    expect(lodGroup.children[0].visible).toBe(true);
    expect(lodGroup.children[1].visible).toBe(false);
  });
});

// ────────────────────────────────────────────────────────────────────────
// Regression: no transient "stacked LOD levels" flash during load
// ────────────────────────────────────────────────────────────────────────

describe('loadLodGroupNode — transient visibility', () => {
  it('hides each child immediately upon attach (no stacked-LOD flash)', async () => {
    // Instrument the loadSceneNodes mock to snapshot every sibling's
    // visibility at the moment each new child is attached. Without the
    // loader-side ``childObject.visible = false`` line, previously-loaded
    // children would still be visible=true here — the user briefly sees
    // every loaded LOD level rendered simultaneously.
    const visibilitySnapshots: { childPath: string; siblings: Record<string, boolean> }[] = [];
    loadSceneNodesMock.mockImplementation(async (node: SceneNode, parent: THREE.Object3D) => {
      // Snapshot sibling state BEFORE attaching this new child, so we
      // see what the user would have momentarily rendered.
      const siblings: Record<string, boolean> = {};
      for (const c of parent.children) {
        if (c.name) siblings[c.name] = c.visible;
      }
      visibilitySnapshots.push({ childPath: node.path, siblings });
      // Now attach the new child (default visible=true per THREE).
      const mesh = new THREE.Mesh();
      mesh.name = node.path;
      parent.add(mesh);
    });

    const ctx = makeCtx(/* no registry — exercises the fallback path */);
    const node = makeLodGroupNode([
      makeChildNode('/lod/child_0', 0),
      makeChildNode('/lod/child_1', 0.5),
      makeChildNode('/lod/child_2', 1.0),
    ]);
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    // For every load-iteration AFTER the first, every already-attached
    // sibling must be hidden. The first iteration has no siblings to
    // check.
    for (let i = 1; i < visibilitySnapshots.length; i++) {
      const snap = visibilitySnapshots[i];
      for (const [siblingPath, visible] of Object.entries(snap.siblings)) {
        expect(
          visible,
          `at the moment ${snap.childPath} was attached, sibling ${siblingPath} was visible — stacked-LOD flash`
        ).toBe(false);
      }
    }
  });
});
