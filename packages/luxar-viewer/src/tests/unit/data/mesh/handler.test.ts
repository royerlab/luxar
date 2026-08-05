/**
 * Smoke tests for the Mesh handler — mirrors `points/handler.test.ts`,
 * `lines/handler.test.ts` and `gsplats/handler.test.ts` so all four geometry
 * handlers share the same basic contract.
 *
 * Mesh was the one kind with no handler test, which is how `handler.ts`'s `kind`
 * export came to be flagged as dead: the three siblings are each kept alive by
 * their own test importing it. The right repair is the missing test, not dropping
 * the export — the file deliberately mirrors its siblings, and asymmetry there is
 * what the geometry-symmetry rule exists to prevent.
 *
 * The distinctive behaviour pinned here is `extend_to_all` threading. Mesh does not
 * merely pass the derived view state through: `processMeshData` RECOMPUTES the
 * membership tolerance from dimension metadata, discarding the derived state's
 * extended tolerance, so the handler has to re-supply `extend_to_all` from the
 * node's attrs. Miss it and an extended mesh commits on first load, then culls on
 * the first slice move.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/mesh/handler';
import type { MeshDataLoader, LoadedMeshData, MeshMetadata } from '../../../../types/mesh';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

/** One triangle in 4D, all three vertices on the hidden axis at `w`. */
function loadedAtW(w: number): LoadedMeshData {
  return {
    vertices: new Float32Array([0, 0, 0, w, 1, 0, 0, w, 0, 1, 0, w]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 3,
    faceCount: 1,
    ndim: 4,
  };
}

/**
 * A 4D view with per-dim metadata, so `processMeshData` has something to recompute
 * the membership tolerance FROM. Hidden dim 3 is discrete with step 1.
 */
function viewWith(slicePositionW: number) {
  return {
    displayDims: [0, 1, 2],
    slicePosition: [0, 0, 0, slicePositionW],
    tolerance: [1e10, 1e10, 1e10, 0.5],
    dimensions: [
      { name: 'x', unit: '', scale: 1 },
      { name: 'y', unit: '', scale: 1 },
      { name: 'z', unit: '', scale: 1 },
      { name: 't', unit: '', scale: 1, discrete: true, step: 1 },
    ],
  };
}

/** Root group holding a mesh node whose `attrs` the handler will read. */
function rootWithMesh(path: string, attrs: Partial<MeshMetadata>): THREE.Group {
  const root = new THREE.Group();
  const mesh = new THREE.Mesh();
  mesh.name = path;
  mesh.userData = { nodeType: 'mesh', attrs, loadedViewVersion: 1 };
  root.add(mesh);
  return root;
}

function ctxFor(root: THREE.Group, viewState: unknown, extra: Record<string, unknown> = {}) {
  return {
    rootGroup: root,
    clearFailure: vi.fn(),
    currentVersion: 1,
    deriveNodeViewState: () => ({ skip: false, viewState }),
    extendedToleranceCache: new Map<string, number[]>(),
    ...extra,
  } as unknown as Parameters<typeof loadAndStage>[3];
}

describe('mesh handler', () => {
  it('discriminates as kind="mesh" with label="Mesh"', () => {
    expect(kind).toBe('mesh');
    expect(label).toBe('Mesh');
  });

  it('returns null and marks the path healthy when the loader yields no data', async () => {
    const loader = {
      loadMesh: vi.fn(),
      updateView: vi.fn().mockResolvedValue(null),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;
    const clearFailure = vi.fn();
    const view = viewWith(0);

    const staged = await loadAndStage(
      '/surface',
      loader,
      makeSession(),
      ctxFor(rootWithMesh('/surface', {}), view, { clearFailure })
    );

    expect(staged).toBeNull();
    // Healthy at every terminal success, including this one — otherwise a later
    // failure re-records with retryCount 0 and the log is pinned at "attempt 1".
    expect(clearFailure).toHaveBeenCalledWith('/surface');
  });

  it('forwards the per-update abort signal to loader.updateView', async () => {
    const updateView = vi.fn().mockResolvedValue(null);
    const loader = { loadMesh: vi.fn(), updateView, dispose: vi.fn() } as unknown as MeshDataLoader;
    const ac = new AbortController();
    const view = viewWith(0);

    await loadAndStage(
      '/surface',
      loader,
      makeSession(),
      ctxFor(rootWithMesh('/surface', {}), view, { signal: ac.signal })
    );

    expect(updateView).toHaveBeenCalledWith(view, expect.anything(), ac.signal);
  });

  it('rides the playback frame budget on the derived per-node view state', async () => {
    const updateView = vi.fn().mockResolvedValue(null);
    const loader = { loadMesh: vi.fn(), updateView, dispose: vi.fn() } as unknown as MeshDataLoader;
    const view = viewWith(0);

    await loadAndStage(
      '/surface',
      loader,
      makeSession(),
      ctxFor(rootWithMesh('/surface', {}), view, { frameBudgetMs: 7 })
    );

    expect(updateView.mock.calls[0][0]).toMatchObject({ frameBudgetMs: 7 });
  });

  it('stages the projection and reports the face count', async () => {
    const loader = {
      loadMesh: vi.fn(),
      updateView: vi.fn().mockResolvedValue(loadedAtW(0)),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;
    const session = makeSession();

    const staged = await loadAndStage(
      '/surface',
      loader,
      session,
      ctxFor(rootWithMesh('/surface', { double_sided: true }), viewWith(0))
    );

    expect(staged?.path).toBe('/surface');
    expect(staged?.projected.visibleFaceCount).toBe(1);
    expect(session.setMetadata).toHaveBeenCalledWith({ info: '1 faces' });
  });

  it('threads extend_to_all so an extended mesh survives an off-slice position', async () => {
    // The behaviour the handler's own comment calls out as required rather than a
    // ride-along. The mesh sits at w = 0 and the slice is at w = 40, far outside any
    // membership slab, so WITHOUT extend_to_all reaching the projection every triangle
    // culls. This is the first-slice-move regression, asserted end to end rather than
    // by spying on the call.
    const off = viewWith(40);
    const loaderPlain = {
      loadMesh: vi.fn(),
      updateView: vi.fn().mockResolvedValue(loadedAtW(0)),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;

    const culled = await loadAndStage(
      '/surface',
      loaderPlain,
      makeSession(),
      ctxFor(rootWithMesh('/surface', { double_sided: true }), off)
    );
    expect(culled?.projected.visibleFaceCount).toBe(0);

    const loaderExtended = {
      loadMesh: vi.fn(),
      updateView: vi.fn().mockResolvedValue(loadedAtW(0)),
      dispose: vi.fn(),
    } as unknown as MeshDataLoader;

    const kept = await loadAndStage(
      '/surface',
      loaderExtended,
      makeSession(),
      ctxFor(rootWithMesh('/surface', { double_sided: true, extend_to_all: ['t'] }), off)
    );
    expect(kept?.projected.visibleFaceCount).toBe(1);
  });
});
