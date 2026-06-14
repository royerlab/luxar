/**
 * Tests for ``loadLodGroupNode``.
 *
 * Strategy: the lod_group loader is mostly a thin wrapper that
 * recurses children through ``loadSceneNodes`` and pushes the
 * resulting THREE nodes + per-child attrs (``min_pixel_size``,
 * ``position_bounds``) into the :class:`LODGroupRegistry`. We mock
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

import { loadLodGroupNode } from '../../../../../data/scene-loader/nodes/load-lod-group-node';
import { LODGroupRegistry } from '../../../../../scene/lod-group-registry';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

beforeEach(() => {
  loadSceneNodesMock.mockReset();
  loadGSplatsNodeCheapMock.mockReset();
  loadGSplatsNodeExpensiveMock.mockReset();
  loadPointsNodeCheapMock.mockReset();
  loadPointsNodeExpensiveMock.mockReset();
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
});

function makeChildNode(
  path: string,
  minPixelSize: number,
  positionBounds: { min: number[]; max: number[] } | undefined = {
    min: [0, 0, 0],
    max: [1, 1, 1],
  }
): SceneNode {
  return {
    path,
    type: 'gsplats',
    attrs: {
      type: 'gsplats',
      min_pixel_size: minPixelSize,
      ...(positionBounds ? { position_bounds: positionBounds } : {}),
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

/** A ``points`` leaf child (the finest level of a points-substitutive ladder). */
function makePointsChildNode(
  path: string,
  minPixelSize: number,
  positionBounds: { min: number[]; max: number[] } = { min: [0, 0, 0], max: [1, 1, 1] }
): SceneNode {
  return {
    path,
    type: 'points',
    attrs: {
      type: 'points',
      min_pixel_size: minPixelSize,
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
      selector: 'pixel_size',
      ...extraAttrs,
    } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children,
  };
}

function makeCtx(registry?: LODGroupRegistry): NodeBuildCtx {
  // Build the minimum NodeBuildCtx surface that loadLodGroupNode
  // actually reads — the rest is forwarded to the mocked
  // loadSceneNodes child recursion.
  const nodeFactory = {
    applyTransform: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  return {
    registry: { registerGSplatsLoader: vi.fn(), registerPointsLoader: vi.fn() } as never,
    lodGroupRegistry: registry,
    nodeFactory,
    viewState: { displayDims: [0, 1, 2], slicePosition: [], tolerance: [] },
    factoryDeps: {} as never,
    isDatasetLive: () => true,
    releaseLazyGSplats: vi.fn(),
    applyEffectiveAttrs: (n) => n.attrs,
    deriveNodeViewState: vi.fn() as never,
    connectLoaderToMonitor: vi.fn(),
    updatePointsGeometry: vi.fn(),
    processLinesData: vi.fn() as never,
    commitLinesGeometry: vi.fn(),
    processGSplatsData: vi.fn() as never,
    commitGSplatsGeometry: vi.fn(),
  };
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
        makeChildNode('/lod/child_1', 100, { min: [0, 0, 0], max: [2, 2, 2] }),
        makeChildNode('/lod/child_2', 500, { min: [0, 0, 0], max: [4, 4, 4] }),
      ],
      { default_level: 0 }
    );

    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(reg.size()).toBe(1);
    const entry = reg.get('/lod')!;
    expect(entry.children).toHaveLength(3);
    expect(entry.children.map((c) => c.minPixelSize)).toEqual([0, 100, 500]);
    expect(entry.children[1].positionBounds).toEqual({
      min: [0, 0, 0],
      max: [2, 2, 2],
    });
    expect(entry.activeChildIndex).toBe(0);
    expect(entry.selectorMode).toBe('auto');
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
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
      { default_level: 99 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    expect(reg.get('/lod')!.activeChildIndex).toBe(1);
  });

  it('registers every loaded child with its threshold and bounds', async () => {
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
      makeChildNode('/lod/child_0', 0),
      makeChildNode('/lod/child_1', 100),
    ]);
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);
    const entry = reg.get('/lod')!;
    expect(entry.children).toHaveLength(2);
    expect(entry.children.map((c) => c.minPixelSize)).toEqual([0, 100]);
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
          min_pixel_size: 0,
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
        makeChildNode('/lod/child_1', 100),
        makeChildNode('/lod/child_2', 500),
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
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
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
  });

  it('does not register or mark ready when the dataset is switched mid-load', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
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
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
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
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(reg.get('/lod')!.children[0].release).toBeUndefined();
  });

  it('marks the child failed (not ready) when the deferred load throws', async () => {
    attachStubChildren();
    loadGSplatsNodeExpensiveMock.mockRejectedValue(new Error('boom'));
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makeChildNode('/lod/child_1', 100)],
      { default_level: 0 }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const deferred = reg.get('/lod')!.children[1];
    deferred.ensureLoaded!();
    await vi.waitFor(() => expect(deferred.failed).toBe(true));

    expect(deferred.ready).toBe(false);
    expect(deferred.loading).toBe(false);
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
        makeChildNode('/lod/child_1', 100), // mid gsplat
        makePointsChildNode('/lod/child_2', 500), // finest = points
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

  it('ensureLoaded loads the points level and registers the points loader', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 100)],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    const pts = reg.get('/lod')!.children[1];
    pts.ensureLoaded!();
    await vi.waitFor(() => expect(pts.ready).toBe(true));

    expect(loadPointsNodeExpensiveMock).toHaveBeenCalledTimes(1);
    expect(ctx.registry.registerPointsLoader).toHaveBeenCalledWith(
      '/lod/child_1',
      expect.anything()
    );
    expect(pts.loading).toBe(false);
  });

  it('does not register the points level when the dataset is switched mid-load', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 100)],
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

  it('gives a deferred points level NO release thunk (no points eviction pool)', async () => {
    attachStubChildren();
    const reg = makeReg();
    const ctx = makeCtx(reg);

    const node = makeLodGroupNode(
      [makeChildNode('/lod/child_0', 0), makePointsChildNode('/lod/child_1', 100)],
      { default_level: 0, display_type: 'points' }
    );
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx, loadSceneNodesMock);

    expect(reg.get('/lod')!.children[1].release).toBeUndefined();
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
        makeChildNode('/lod/child_1', 100),
        makeChildNode('/lod/child_2', 500),
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
        makeChildNode('/lod/child_1', 100),
        makeChildNode('/lod/child_2', 500),
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
      makeChildNode('/lod/child_1', 100),
      makeChildNode('/lod/child_2', 500),
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
