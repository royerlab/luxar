/**
 * Unit tests for `MeshProgressiveLoader` — the reveal-ladder composite.
 *
 * Two things here are worth more than the rest, and both are silent when wrong:
 *
 *  - **Face-index offsetting.** Each level's indices are local to its own vertex
 *    array, so a verbatim copy does not crash — it draws a plausible-looking
 *    garbage surface stitched out of the wrong vertices. Every assertion about
 *    it is therefore on VALUES, not lengths.
 *  - **`committedEnergyFraction` existing and returning `null`.** Absent, the
 *    commit stamps a half-revealed mesh as carrying all its energy; numeric, the
 *    LOD fade brightens it by `1/e`. Only the middle case is right, and the two
 *    wrong ones are invisible to any test that just reads the getter — so the
 *    consequence is asserted through `stampLadderComplete` itself, with a
 *    sensitivity control that pins the OTHER branch.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  MeshProgressiveLoader,
  concatenateMeshData,
} from '../../../data/mesh/mesh-progressive-loader';
import { stampLadderComplete } from '../../../data/scene-loader/commit/stamp-view-version';
import type { LoadedMeshData, MeshViewState } from '../../../types/mesh';
import type { MeshWholeNodeLoader } from '../../../data/mesh/mesh-whole-node-loader';

// ============================================================================
// Fixtures
// ============================================================================

/**
 * One ladder level: `nVertices` vertices at ascending coordinates and
 * `faces` given as LOCAL indices, exactly as the writer's `split_mesh_by_faces`
 * emits them.
 */
function level(
  nVertices: number,
  faces: number[],
  opts: {
    ndim?: number;
    normals?: boolean;
    colors?: 3 | 4 | null;
    scalars?: boolean;
    base?: number;
  } = {}
): LoadedMeshData {
  const ndim = opts.ndim ?? 3;
  const base = opts.base ?? 0;
  const vertices = new Float32Array(nVertices * ndim);
  for (let v = 0; v < nVertices; v++) {
    for (let d = 0; d < ndim; d++) vertices[v * ndim + d] = base + v + d / 10;
  }
  const colorK = opts.colors ?? null;
  return {
    vertices,
    faces: new Uint32Array(faces),
    normals: opts.normals ? new Float32Array(nVertices * 3).fill(base + 1) : null,
    colors: colorK ? new Uint8Array(nVertices * colorK).fill(base + 2) : null,
    ...(colorK ? { colorComponents: colorK } : {}),
    ...(opts.scalars ? { scalars: new Float32Array(nVertices).fill(base + 3) } : {}),
    vertexCount: nVertices,
    faceCount: faces.length / 3,
    ndim,
  };
}

/** A sub-loader stub with the one method the ladder loop calls. */
function subLoader(
  data: LoadedMeshData,
  opts: { resident?: boolean; elapsed?: number } = {}
): MeshWholeNodeLoader & { calls: number; disposed: boolean } {
  const stub = {
    calls: 0,
    disposed: false,
    updateViewWithResidency: vi.fn(async () => {
      stub.calls++;
      return { data, allResident: opts.resident ?? true };
    }),
    dispose: vi.fn(() => {
      stub.disposed = true;
    }),
  };
  return stub as unknown as MeshWholeNodeLoader & { calls: number; disposed: boolean };
}

const VIEW: MeshViewState = {
  displayDims: [0, 1, 2],
  slicePosition: [0, 0, 0],
  tolerance: [0, 0, 0],
};

// ============================================================================
// concatenateMeshData
// ============================================================================

describe('concatenateMeshData', () => {
  it('offsets each level face index by the preceding levels VERTEX count', () => {
    // Level 0: 4 vertices, one triangle. Level 1: 3 vertices, one triangle whose
    // indices are 0,1,2 IN ITS OWN array — i.e. global 4,5,6.
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2], { base: 100 })]);

    expect(merged.vertexCount).toBe(7);
    expect(merged.faceCount).toBe(2);
    // The whole point: a verbatim copy would give [0,1,2, 0,1,2] — the same
    // length, the same face count, and a surface stitched from level 0's
    // vertices twice. Assert the VALUES.
    expect(Array.from(merged.faces)).toEqual([0, 1, 2, 4, 5, 6]);
    // And every index must address a real vertex of the merged buffer.
    for (const index of merged.faces) expect(index).toBeLessThan(merged.vertexCount);
  });

  it('keeps the second level vertices reachable through the offset indices', () => {
    // The end-to-end version of the check above: follow an index back to the
    // coordinates it lands on, so an offset that is merely *present* but wrong
    // (off by a face count rather than a vertex count, say) still fails.
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2], { base: 100 })]);
    const firstOfSecondTriangle = merged.faces[3];
    expect(merged.vertices[firstOfSecondTriangle * 3]).toBeCloseTo(100);
  });

  it('strides vertices by ndim, not by 3', () => {
    const merged = concatenateMeshData([
      level(2, [0, 1, 1], { ndim: 4 }),
      level(2, [0, 1, 1], { ndim: 4, base: 50 }),
    ]);
    expect(merged.ndim).toBe(4);
    expect(merged.vertices.length).toBe(4 * 4);
    // Level 1's first vertex starts at element 2*4 = 8, not 2*3 = 6.
    expect(merged.vertices[8]).toBeCloseTo(50);
  });

  it('throws on levels that disagree about ndim', () => {
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1]), level(2, [0, 1, 1], { ndim: 4 })])
    ).toThrow(/mixed dimensionality/);
  });

  it('throws when one level carries normals and another omits them', () => {
    // Not a fill-with-default case, unlike the sibling ladders' colours: the
    // `normal` attribute is bound once from the node's metadata and can never be
    // added or removed later, so a level without normals would leave the previous
    // level's bound at the wrong length.
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1], { normals: true }), level(2, [0, 1, 1])])
    ).toThrow(/'normals'/);
  });

  it('throws when levels disagree about the colour layout', () => {
    expect(() =>
      concatenateMeshData([level(2, [0, 1, 1], { colors: 3 }), level(2, [0, 1, 1], { colors: 4 })])
    ).toThrow(/mixed color layouts|'colors'/);
  });

  it('concatenates the optional per-vertex arrays alongside the vertices', () => {
    const merged = concatenateMeshData([
      level(2, [0, 1, 1], { normals: true, colors: 3, scalars: true }),
      level(3, [0, 1, 2], { normals: true, colors: 3, scalars: true, base: 10 }),
    ]);
    expect(merged.normals?.length).toBe(5 * 3);
    expect(merged.colors?.length).toBe(5 * 3);
    expect(merged.scalars?.length).toBe(5);
    // Level 1's values start where level 0's end — the same offset the faces use.
    expect(merged.scalars?.[2]).toBeCloseTo(13);
  });

  it('sizes the projection scratch to the REVEALED prefix', () => {
    const merged = concatenateMeshData([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
    expect(merged.projection?.position.length).toBe(7 * 3);
    expect(merged.projection?.mask.length).toBe(7);
    expect(merged.projection?.faceScratch.length).toBe(2 * 3);
    // A fresh scratch has not been extracted into yet, so the epoch key must be
    // null — otherwise the projection would skip the re-extract and draw a stale
    // frame into a buffer that has never held one.
    expect(merged.projection?.displayDimsKey).toBeNull();
  });

  it('hands back the single level unchanged, scratch included', () => {
    const only = level(4, [0, 1, 2]);
    expect(concatenateMeshData([only])).toBe(only);
  });
});

// ============================================================================
// MeshProgressiveLoader
// ============================================================================

describe('MeshProgressiveLoader', () => {
  function makeLadder(
    levels: LoadedMeshData[],
    opts: { resident?: boolean } = {}
  ): {
    loader: MeshProgressiveLoader;
    subs: ReturnType<typeof subLoader>[];
  } {
    const subs = levels.map((d) => subLoader(d, opts));
    return { loader: new MeshProgressiveLoader(subs, levels.length, '/surf'), subs };
  }

  it('streams every level on a refine pass and reports completion', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])]);

    expect(loader.hasMoreLODs).toBe(true);
    const data = await loader.updateView(VIEW);

    expect(loader.loadedLODCount).toBe(3);
    expect(loader.totalLODCount).toBe(3);
    expect(loader.hasMoreLODs).toBe(false);
    expect(data.faceCount).toBe(3);
    expect(data.vertexCount).toBe(10);
  });

  it('stops after the first COLD level on a refine pass', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])], {
      resident: false,
    });

    await loader.updateView(VIEW);

    // The first-paint floor always loads (level 0), and the next level's cache
    // miss ends the pass so the frame can render. A later pass continues.
    expect(loader.loadedLODCount).toBe(2);
    expect(loader.hasMoreLODs).toBe(true);
  });

  it('loads only the first level under a playback frame budget', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    await loader.updateView({ ...VIEW, frameBudgetMs: 8 });

    expect(loader.loadedLODCount).toBe(1);
    // The budgeted prefix IS the target while playback is running, so nothing
    // should schedule background refinement between animation ticks.
    expect(loader.hasMoreLODs).toBe(false);
  });

  it('returns the SAME object across view changes once the ladder is complete', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    const first = await loader.updateView(VIEW);
    const afterSliceMove = await loader.updateView({
      ...VIEW,
      slicePosition: [5, 5, 5],
    });

    // A mesh is whole-node resident, so a slice move changes nothing about what
    // is LOADED. Identity is what keeps the loader-owned projection scratch (and
    // with it the uploaded vertex buffer) alive across a scrub — the sibling
    // loaders cannot do this because their levels answer a range query.
    expect(afterSliceMove).toBe(first);
    expect(first.projection).toBe(afterSliceMove.projection);
  });

  it('returns a NEW object once a further level lands', async () => {
    const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])], {
      resident: false,
    });

    const partial = await loader.updateView(VIEW);
    const deeper = await loader.updateView(VIEW);

    expect(deeper).not.toBe(partial);
    expect(deeper.faceCount).toBeGreaterThan(partial.faceCount);
  });

  it('re-serves each level from its own decode rather than re-fetching', async () => {
    const { loader, subs } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);

    await loader.updateView(VIEW);
    await loader.updateView({ ...VIEW, slicePosition: [1, 1, 1] });

    // Once a level is in the ladder it is never asked again — the ladder is
    // view-independent. A per-view reset (what the three siblings do) would show
    // up here as a second call per level.
    for (const sub of subs) expect(sub.calls).toBe(1);
  });

  it('disposes every sub-loader and reports no further work', async () => {
    const { loader, subs } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
    await loader.updateView(VIEW);

    loader.dispose();

    for (const sub of subs) expect(sub.disposed).toBe(true);
    expect(loader.hasMoreLODs).toBe(false);
  });

  describe('energy stamps', () => {
    it('reports a null committed energy fraction', () => {
      const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
      expect(loader.committedEnergyFraction).toBeNull();
    });

    it('makes the commit leave `committedEnergyFraction` ABSENT', () => {
      // The consequence, not the getter. `stampLadderComplete` probes
      // `'committedEnergyFraction' in loader`, so this is what distinguishes a
      // null-returning getter from no getter at all.
      const { loader } = makeLadder([level(4, [0, 1, 2]), level(3, [0, 1, 2])]);
      const userData: { loader: unknown; committedEnergyFraction?: number } = { loader };

      stampLadderComplete(userData);

      expect('committedEnergyFraction' in userData).toBe(false);
    });

    it('SENSITIVITY: a loader without the getter is stamped as fully committed', () => {
      // The control for the assertion above. Without it, "the key is absent"
      // would pass against a loader that simply has no ladder at all — and the
      // real hazard is exactly that misreading: a half-revealed mesh stamped
      // `committedEnergyFraction: 1`, i.e. "all of it is on screen".
      const userData: { loader: unknown; committedEnergyFraction?: number } = {
        loader: { hasMoreLODs: true },
      };

      stampLadderComplete(userData);

      expect(userData.committedEnergyFraction).toBe(1);
    });

    it('stamps the ladder INCOMPLETE while levels remain', async () => {
      const { loader } = makeLadder(
        [level(4, [0, 1, 2]), level(3, [0, 1, 2]), level(3, [0, 1, 2])],
        {
          resident: false,
        }
      );
      await loader.updateView(VIEW);
      const userData: { loader: unknown; committedLadderComplete?: boolean } = { loader };

      stampLadderComplete(userData);

      expect(userData.committedLadderComplete).toBe(false);
    });
  });
});
