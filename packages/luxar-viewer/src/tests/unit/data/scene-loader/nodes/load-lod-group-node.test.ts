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
vi.mock('../../../../../data/scene-loader/nodes/load-scene-nodes', () => ({
  loadSceneNodes: (...args: unknown[]) => loadSceneNodesMock(...args),
}));

import { loadLodGroupNode } from '../../../../../data/scene-loader/nodes/load-lod-group-node';
import { LODGroupRegistry } from '../../../../../scene/lod-group-registry';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../../../../data/data-loader-types';

beforeEach(() => {
  loadSceneNodesMock.mockReset();
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

function makeLodGroupNode(children: SceneNode[], extraAttrs: Record<string, unknown> = {}): SceneNode {
  return {
    path: '/lod',
    type: 'lod_group',
    attrs: {
      type: 'lod_group',
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
    registry: {} as never,
    lodGroupRegistry: registry,
    nodeFactory,
    viewState: { displayDims: [0, 1, 2], slicePosition: [], tolerance: [] },
    factoryDeps: {} as never,
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
  loadSceneNodesMock.mockImplementation(
    async (node: SceneNode, parent: THREE.Object3D) => {
      const mesh = new THREE.Mesh();
      mesh.name = node.path;
      parent.add(mesh);
    }
  );
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

    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);

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
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);
    expect(reg.get('/lod')!.activeChildIndex).toBe(1);
  });

  it('children load with ready=true so the selector can swap immediately', async () => {
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
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);
    const entry = reg.get('/lod')!;
    for (const child of entry.children) {
      expect(child.ready).toBe(true);
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
          min_pixel_size: 0,
          center_bounds: { min: [1, 1, 1], max: [3, 3, 3] },
        } as SceneNode['attrs'],
        hasSpatialIndex: false,
        children: [],
      },
    ]);
    await loadLodGroupNode(node, new THREE.Group(), makeStubLoc(), ctx);
    expect(reg.get('/lod')!.children[0].positionBounds).toEqual({
      min: [1, 1, 1],
      max: [3, 3, 3],
    });
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
    const lodGroup = await loadLodGroupNode(node, parent, makeStubLoc(), ctx);

    // Without a registry we fall back to visibility set inline.
    expect(lodGroup.children[0].visible).toBe(false);
    expect(lodGroup.children[1].visible).toBe(true);
    expect(lodGroup.children[2].visible).toBe(false);
  });
});
