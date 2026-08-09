/**
 * Unit tests for `load-mesh-node.ts`, covering the one invariant its
 * cheap/expensive split can silently break: WHERE the loader is registered.
 *
 * A mesh became a `kind=lod` level when `luxar.mesh.decimate` landed, so a
 * non-default level is now cheap-attached and its expensive half runs only on
 * first activation. Registration must therefore stay on the EAGER path — a lazy
 * level that lands in `registry.meshLoaders` joins the per-slice update sweep,
 * which re-fetches and re-commits it on every scrub (even while hidden, and
 * concurrently with the registry's own `ensureLoaded`), gating each slice change
 * on projecting the full-resolution surface. That is exactly what deferring it
 * exists to avoid, and it is what the sibling loaders' identical structure
 * prevents. Mirrors `load-lines-node.test.ts`'s registration-timing block.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as THREE from 'three';

const createMeshLoaderMock = vi.fn();
vi.mock('../../../../../data/scene-loader/loaders/loader-factory', () => ({
  createMeshLoader: (...args: unknown[]) => createMeshLoaderMock(...args),
}));

import {
  loadMeshNode,
  loadMeshNodeCheap,
  loadMeshNodeExpensive,
} from '../../../../../data/scene-loader/nodes/load-mesh-node';
import type { NodeBuildCtx } from '../../../../../data/scene-loader/nodes/build-ctx';
import { nodeBuildCtxDefaults } from '../../../../helpers/node-build-ctx';
import type { SceneNode, ViewState } from '../../../../../data/data-loader-types';
import type { LoadedMeshData, MeshDataLoader } from '../../../../../types/mesh';

function makeViewState(): ViewState {
  return { displayDims: [0, 1, 2], slicePosition: [0, 0, 0], tolerance: [0, 0, 0] };
}

function makeSceneNode(): SceneNode {
  return {
    path: '/scene/m',
    type: 'mesh',
    attrs: { type: 'mesh', n_vertices: 3, n_faces: 1 } as SceneNode['attrs'],
    hasSpatialIndex: false,
    children: [],
  };
}

function makeMeshLoader(loadMesh: () => Promise<LoadedMeshData>): MeshDataLoader {
  return { loadMesh } as unknown as MeshDataLoader;
}

function makeCtx(): NodeBuildCtx {
  const viewState = makeViewState();
  const nodeFactory = {
    createEmptyMeshNode: vi.fn((path: string) => {
      const m = new THREE.Mesh();
      m.name = path;
      return m;
    }),
    applyTransform: vi.fn(),
    markPickingDirty: vi.fn(),
  } as unknown as NodeBuildCtx['nodeFactory'];

  return {
    ...nodeBuildCtxDefaults(viewState),
    nodeFactory,
    // The mesh path is what this file exercises, so its process step returns a
    // real staged result instead of the shared no-op.
    processMeshData: vi.fn().mockResolvedValue({
      path: '/scene/m',
      projected: { visibleFaceCount: 1 },
    }) as never,
  };
}

beforeEach(() => {
  createMeshLoaderMock.mockReset();
});

describe('load-mesh-node — the loader joins the sweep only on the eager path', () => {
  it('does NOT register when only the expensive half runs (a lazy LOD level)', async () => {
    createMeshLoaderMock.mockReturnValue(
      makeMeshLoader(vi.fn().mockResolvedValue({ faceCount: 1 } as LoadedMeshData))
    );
    const ctx = makeCtx();
    const node = makeSceneNode();

    // Exactly what the lod_group defer path does: cheap-attach now, run the
    // expensive tail later from `ensureLoaded`.
    const { loader } = await loadMeshNodeCheap(node, new THREE.Group(), {} as never, ctx);
    await loadMeshNodeExpensive(node, ctx, loader);

    expect(ctx.registry.meshLoaders.has('/scene/m')).toBe(false);
  });

  it('does NOT register while the eager initial load is in flight; registers on success', async () => {
    let resolveLoad!: (d: LoadedMeshData) => void;
    const pending = new Promise<LoadedMeshData>((res) => (resolveLoad = res));
    createMeshLoaderMock.mockReturnValue(makeMeshLoader(() => pending));
    const ctx = makeCtx();

    const promise = loadMeshNode(makeSceneNode(), new THREE.Group(), {} as never, ctx);
    await Promise.resolve();
    expect(ctx.registry.meshLoaders.has('/scene/m')).toBe(false);

    resolveLoad({ faceCount: 1 } as LoadedMeshData);
    await promise;
    expect(ctx.registry.meshLoaders.has('/scene/m')).toBe(true);
  });

  it('registers even when the eager initial load FAILS (loader stays retryable)', async () => {
    createMeshLoaderMock.mockReturnValue(
      makeMeshLoader(vi.fn().mockRejectedValue(new Error('network down')))
    );
    const ctx = makeCtx();

    await expect(
      loadMeshNode(makeSceneNode(), new THREE.Group(), {} as never, ctx)
    ).rejects.toThrow();

    expect(ctx.registry.meshLoaders.has('/scene/m')).toBe(true);
    expect(ctx.registry.failedLoaders.has('/scene/m')).toBe(true);
  });
});
