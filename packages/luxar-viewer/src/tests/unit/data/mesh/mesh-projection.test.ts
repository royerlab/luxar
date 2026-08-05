/**
 * Mesh display-space projection: the whole-triangle cull, the no-hidden-dims
 * fast path, and winding parity.
 *
 * Run against `TypeScriptFallback` — not a mock. It is the real reference
 * implementation of both cull kernels AND the production backend for
 * `ndim > 16`, so exercising it here tests shipping code rather than a stub's
 * idea of the contract.
 */

import { describe, it, expect } from 'vitest';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import {
  projectMesh,
  resolveWinding,
  noticeUndecidableWinding,
} from '../../../../data/mesh/projection';
import type { LoadedMeshData, MeshViewState } from '../../../../types/mesh';

const backend = new TypeScriptFallback();

/** A view state with everything the projection reads. */
function viewState(
  displayDims: number[],
  slicePosition: number[],
  tolerance: number[]
): MeshViewState {
  return { displayDims, slicePosition, tolerance } as MeshViewState;
}

/**
 * Two independent triangles at distinct positions on a hidden 4th axis:
 * vertices 0-2 sit at `w = 0`, vertices 3-5 at `w = 10`.
 *
 * Independent (not welded) on purpose — the whole-triangle cull is an AND over a
 * face's three vertices, so a fixture where every face shares vertices could not
 * show one face surviving while another is culled.
 */
function twoTrianglesIn4D(): LoadedMeshData {
  const vertices = new Float32Array([
    // x, y, z, w
    0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
    // second triangle, offset on w
    0, 0, 1, 10, 1, 0, 1, 10, 0, 1, 1, 10,
  ]);
  return {
    vertices,
    faces: new Uint32Array([0, 1, 2, 3, 4, 5]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 6,
    faceCount: 2,
    ndim: 4,
  };
}

/** A single 3D triangle, for the fast path and winding tests. */
function oneTriangleIn3D(): LoadedMeshData {
  return {
    vertices: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    faces: new Uint32Array([0, 1, 2]),
    normals: null,
    colors: null,
    scalars: undefined,
    vertexCount: 3,
    faceCount: 1,
    ndim: 3,
  };
}

describe('projectMesh — the whole-triangle cull', () => {
  it('keeps only the triangle whose vertices are all inside the slab', () => {
    const data = twoTrianglesIn4D();
    // Slice at w = 0 with a half-unit slab: triangle A (w=0) is in, B (w=10) out.
    const result = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(result.visibleFaceCount).toBe(1);
    expect(Array.from(result.indices)).toEqual([0, 1, 2]);
    expect(result.visibleVertexCount).toBe(3);
    expect(result.usedFastPath).toBe(false);
  });

  it('keeps the OTHER triangle when the slice moves', () => {
    // The anti-vacuity half: a cull that always returned the first face would
    // pass the test above.
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 1, 2], [0, 0, 0, 10], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(Array.from(result.indices)).toEqual([3, 4, 5]);
  });

  it('keeps both when the slab spans both', () => {
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 1, 2], [0, 0, 0, 5], [1e10, 1e10, 1e10, 6]),
      undefined,
      true,
      backend
    );
    expect(result.visibleFaceCount).toBe(2);
    expect(Array.from(result.indices)).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('keeps none when the slab misses both, and emits an empty index buffer', () => {
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 1, 2], [0, 0, 0, 100], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(result.visibleFaceCount).toBe(0);
    expect(result.indices.length).toBe(0);
  });

  it('drops a whole triangle when ONE vertex falls outside — no clipping', () => {
    // The defining consequence of the design (spec §5.1/§5.3): a partially
    // straddling triangle is not cut, it is dropped. A build that clipped, or
    // that OR-ed instead of AND-ing the three vertices, keeps it.
    const data = twoTrianglesIn4D();
    data.vertices[3] = 10; // move vertex 0 of triangle A off the slice
    const result = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(result.visibleFaceCount).toBe(0);
  });

  it('hides a vertex with a NON-FINITE hidden coordinate (#806)', () => {
    for (const bad of [NaN, Infinity, -Infinity]) {
      const data = twoTrianglesIn4D();
      data.vertices[3] = bad;
      const result = projectMesh(
        data,
        // A slab wide enough to admit everything finite — so only the
        // non-finite rule can remove this triangle.
        viewState([0, 1, 2], [0, 0, 0, 5], [1e10, 1e10, 1e10, 1e9]),
        undefined,
        true,
        backend
      );
      expect(result.visibleFaceCount, `w = ${String(bad)}`).toBe(1);
      expect(Array.from(result.indices)).toEqual([3, 4, 5]);
    }
  });

  it('sizes the index buffer to exactly the visible faces', () => {
    // The kernel writes into a worst-case-sized scratch buffer. Handing over an
    // oversized buffer (or a view onto it) would upload the untouched tail and
    // draw stale triangles.
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(result.indices.length).toBe(result.visibleFaceCount * 3);
    // And it must be a standalone buffer, not a subarray view: THREE uploads
    // `array.buffer`, so a view would carry the whole worst-case allocation.
    expect(result.indices.byteOffset).toBe(0);
    expect(result.indices.buffer.byteLength).toBe(result.indices.byteLength);
  });

  it('extracts display-space positions for the displayed triple', () => {
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    expect(result.position.length).toBe(6 * 3);
    // Vertex 1 is (1, 0, 0, 0) in nD -> (1, 0, 0) in display space.
    expect(Array.from(result.position.slice(3, 6))).toEqual([1, 0, 0]);
  });

  it('honours a permuted displayDims when extracting positions', () => {
    const result = projectMesh(
      twoTrianglesIn4D(),
      // Display (w, x, y): vertex 3 is (0, 0, 1, 10) -> (10, 0, 0).
      viewState([3, 0, 1], [0, 0, 0, 0], [1e10, 1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    expect(Array.from(result.position.slice(9, 12))).toEqual([10, 0, 0]);
  });
});

describe('projectMesh — the no-hidden-dims fast path', () => {
  it('skips the cull and indexes every face when displayDims covers ndim', () => {
    const result = projectMesh(
      oneTriangleIn3D(),
      viewState([0, 1, 2], [0, 0, 0], [1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    expect(result.usedFastPath).toBe(true);
    expect(result.visibleFaceCount).toBe(1);
    expect(result.visibleVertexCount).toBe(3);
    expect(Array.from(result.indices)).toEqual([0, 1, 2]);
  });

  it('takes the fast path even under a slice position that would cull nothing', () => {
    // With no hidden dims there is no slab to fail, so slicePosition is
    // irrelevant here. Asserted so a future change that starts consulting it on
    // this path is caught.
    const result = projectMesh(
      oneTriangleIn3D(),
      viewState([0, 1, 2], [999, 999, 999], [0, 0, 0]),
      undefined,
      true,
      backend
    );
    expect(result.usedFastPath).toBe(true);
    expect(result.visibleFaceCount).toBe(1);
  });

  it('does NOT hand back the loader’s own faces array', () => {
    // The fast path has no compaction to do, so copying is easy to forget — and
    // aliasing the loader's cached `faces` would let the winding post-pass mutate
    // the source of truth, corrupting every later rebuild.
    const data = oneTriangleIn3D();
    const result = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0], [1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    expect(result.indices).not.toBe(data.faces);
    result.indices[0] = 2;
    expect(data.faces[0]).toBe(0);
  });

  it('still re-extracts positions on the fast path', () => {
    // A displayDims change here leaves the mask all-ones but MUST re-derive
    // position — "fast path" means the cull is elided, not that no work happens.
    const data = oneTriangleIn3D();
    const asIs = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0], [1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    const swapped = projectMesh(
      data,
      viewState([1, 0, 2], [0, 0, 0], [1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    // Vertex 1 is (1, 0, 0): as (x,y,z) it is [1,0,0], as (y,x,z) it is [0,1,0].
    expect(Array.from(asIs.position.slice(3, 6))).toEqual([1, 0, 0]);
    expect(Array.from(swapped.position.slice(3, 6))).toEqual([0, 1, 0]);
  });
});

describe('projectMesh — view-state guards', () => {
  // The cull kernel indexes slicePosition[dim] / tolerance[dim] for every hidden
  // dim < ndim, and the Rust crate is `panic = "abort"`: a short array does not
  // fail as a node-scoped error, it traps and takes down the whole WASM module.
  // Points guards slicePosition and Lines guards tolerance; mesh reads both.

  it('rejects a slicePosition shorter than ndim', () => {
    expect(() =>
      projectMesh(
        twoTrianglesIn4D(),
        viewState([0, 1, 2], [0, 0, 0], [1e10, 1e10, 1e10, 1]),
        undefined,
        true,
        backend
      )
    ).toThrow(/slicePosition too short \(got 3, expected ≥ 4\)/);
  });

  it('rejects a tolerance shorter than ndim', () => {
    expect(() =>
      projectMesh(
        twoTrianglesIn4D(),
        viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10]),
        undefined,
        true,
        backend
      )
    ).toThrow(/tolerance too short \(got 3, expected ≥ 4\)/);
  });

  it('rejects a displayDims entry outside [0, ndim)', () => {
    expect(() =>
      projectMesh(
        oneTriangleIn3D(),
        viewState([0, 1, 7], [0, 0, 0], [1e10, 1e10, 1e10]),
        undefined,
        true,
        backend
      )
    ).toThrow(/displayDims\[2\]=7 out of range/);
  });

  it('rejects a vertices array shorter than vertexCount x ndim', () => {
    const data = oneTriangleIn3D();
    data.vertices = new Float32Array(6); // 2 vertices' worth, not 3
    expect(() =>
      projectMesh(
        data,
        viewState([0, 1, 2], [0, 0, 0], [1e10, 1e10, 1e10]),
        undefined,
        true,
        backend
      )
    ).toThrow(/positions array too short/);
  });
});

describe('resolveWinding', () => {
  it('needs no decision for an authored double-sided mesh', () => {
    // Both orientations draw, so parity is unobservable — and the notice must
    // stay quiet for what is the overwhelmingly common case.
    expect(resolveWinding([0, 2, 1], [0, 1, 2], true)).toEqual({
      reverse: false,
      side: 'double',
    });
  });

  it.each([
    [[0, 1, 2], false, 'identity'],
    [[1, 2, 0], false, 'cycle'],
    [[2, 0, 1], false, 'cycle'],
    [[0, 2, 1], true, 'swap'],
    [[1, 0, 2], true, 'swap'],
    [[2, 1, 0], true, 'swap'],
  ])('gets parity right for %j (%s)', (displayDims, reverse) => {
    // All six permutations, exhaustively: the three even ones must NOT reverse
    // and the three odd ones must. A build that keyed off "is it sorted?" would
    // get the two cycles wrong.
    const result = resolveWinding(displayDims as number[], [0, 1, 2], false);
    expect(result).toEqual({ reverse, side: 'front' });
  });

  it('compares against the SORTED frame, so normal_dims order does not shift parity', () => {
    // The frame is sorted(normal_dims) (spec §3.2). Authoring normals as
    // (2, 0, 1) describes the same frame {0,1,2} as (0, 1, 2).
    expect(resolveWinding([0, 1, 2], [2, 0, 1], false).reverse).toBe(false);
    expect(resolveWinding([0, 2, 1], [2, 0, 1], false).reverse).toBe(true);
  });

  it('falls back to double-sided with no winding frame', () => {
    const result = resolveWinding([0, 1, 2], undefined, false);
    expect(result.side).toBe('double');
    expect(result.reverse).toBe(false);
    expect(result.undecidableReason).toMatch(/no stored normals/);
  });

  it('falls back to double-sided for a DIFFERENT axis triple', () => {
    // [0,1,2] -> [1,2,3]: projected orientation is per-triangle data-dependent,
    // so no index post-pass can correct it. Reversing anyway would be worse than
    // doing nothing — it would flip the triangles that were already correct.
    const result = resolveWinding([1, 2, 3], [0, 1, 2], false);
    expect(result.side).toBe('double');
    expect(result.reverse).toBe(false);
    expect(result.undecidableReason).toMatch(/different triple/);
  });

  it('falls back to double-sided with fewer than 3 displayed dimensions', () => {
    const result = resolveWinding([0, 1], [0, 1, 2], false);
    expect(result.side).toBe('double');
    expect(result.undecidableReason).toMatch(/2 displayed dimensions/);
  });
});

describe('projectMesh — the winding post-pass', () => {
  it('swaps exactly two indices per face in an odd-parity epoch', () => {
    const data = oneTriangleIn3D();
    const result = projectMesh(
      data,
      viewState([0, 2, 1], [0, 0, 0], [1e10, 1e10, 1e10]),
      [0, 1, 2],
      false,
      backend
    );
    // (0,1,2) -> (0,2,1): orientation reversed. A rotation like (1,2,0) would
    // leave winding unchanged, which is the easy mistake here.
    expect(Array.from(result.indices)).toEqual([0, 2, 1]);
    expect(result.side).toBe('front');
  });

  it('leaves winding alone in an even-parity epoch', () => {
    const result = projectMesh(
      oneTriangleIn3D(),
      viewState([1, 2, 0], [0, 0, 0], [1e10, 1e10, 1e10]),
      [0, 1, 2],
      false,
      backend
    );
    expect(Array.from(result.indices)).toEqual([0, 1, 2]);
  });

  it('runs on the CULL path too, not only the fast path', () => {
    // The reversal is keyed to the current displayDims parity, so it applies to
    // every index build in an odd-parity epoch — including a pure slice move on
    // a 4D mesh, where only the index buffer is rebuilt.
    const result = projectMesh(
      twoTrianglesIn4D(),
      viewState([0, 2, 1], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      [0, 1, 2],
      false,
      backend
    );
    expect(result.visibleFaceCount).toBe(1);
    expect(Array.from(result.indices)).toEqual([0, 2, 1]);
  });

  it('reports side "double" and does not reverse when winding is undecidable', () => {
    const result = projectMesh(
      twoTrianglesIn4D(),
      // Displaying (x, w, y) — a different triple than the frame {0,1,2}.
      viewState([0, 3, 1], [0, 0, 0, 0], [1e10, 1e10, 0.5, 1e10]),
      [0, 1, 2],
      false,
      backend
    );
    expect(result.side).toBe('double');
    expect(Array.from(result.indices)).toEqual([0, 1, 2]);
  });
});

describe('noticeUndecidableWinding', () => {
  it('logs once per node, not once per rebuild', () => {
    // The notice is separate from resolveWinding precisely so it cannot fire on
    // every index build; this pins that.
    const seen = new Set<string>();
    noticeUndecidableWinding('/a', 'reason', seen);
    noticeUndecidableWinding('/a', 'reason', seen);
    noticeUndecidableWinding('/b', 'reason', seen);
    expect(Array.from(seen)).toEqual(['/a', '/b']);
  });
});
