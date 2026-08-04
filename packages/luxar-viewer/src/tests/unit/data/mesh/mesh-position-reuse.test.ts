/**
 * Position-buffer reuse across a pure slice move (#1245).
 *
 * `projectMesh` memoizes the display-space projection per `displayDims`, keyed by
 * the whole-node `LoadedMeshData`. The point is array IDENTITY: a pure slice move
 * (same displayed triple, different slice position) must hand back the SAME
 * `position` array, because `updateMeshGeometry` decides whether to re-upload the
 * position buffer and recompute bounds by array identity. A `displayDims` change
 * is a cache miss and re-extracts into a fresh array.
 *
 * Run against `TypeScriptFallback` — not a mock — mirroring `mesh-projection.test.ts`:
 * it is the real reference implementation of the cull kernels and the production
 * `ndim > 16` backend, so this exercises shipping code.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { TypeScriptFallback } from '../../../../wasm/typescript';
import { projectMesh } from '../../../../data/mesh/projection';
import { buildMeshGeometry, updateMeshGeometry } from '../../../../rendering/mesh-geometry';
import { log } from '../../../../utils/log';
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
 * vertices 0-2 sit at `w = 0`, vertices 3-5 at `w = 10`. Same fixture as
 * `mesh-projection.test.ts` — a slice at w=0 keeps face A, at w=10 keeps face B.
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

describe('projectMesh — position memoization (#1245)', () => {
  it('reuses the SAME position array across a pure slice move (same displayDims)', () => {
    const data = twoTrianglesIn4D();
    // Same displayed triple [0,1,2], different slice on the hidden w axis: this is
    // exactly a pure slice move, so the display-space positions are identical and
    // the memoized array must be handed back by identity.
    const atW0 = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    const atW10 = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 10], [1e10, 1e10, 1e10, 0.5]),
      undefined,
      true,
      backend
    );
    // Identity, not just equality — that is what lets updateMeshGeometry skip the
    // re-upload and bounds recompute.
    expect(atW10.position).toBe(atW0.position);
    // The index buffers still track the two slices, so the reuse is not hiding a
    // stale-projection bug: each slice indexes its own triangle.
    expect(Array.from(atW0.indices)).toEqual([0, 1, 2]);
    expect(Array.from(atW10.indices)).toEqual([3, 4, 5]);
  });

  it('re-extracts into a DIFFERENT array on a displayDims change', () => {
    const data = twoTrianglesIn4D();
    const asIs = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    // Display (w, x, y): a different triple is a cache miss and must re-extract
    // into a fresh array holding the permuted coordinates.
    const permuted = projectMesh(
      data,
      viewState([3, 0, 1], [0, 0, 0, 0], [1e10, 1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    expect(permuted.position).not.toBe(asIs.position);
    // Vertex 3 is (0, 0, 1, 10) in nD -> (w, x, y) = (10, 0, 0) in display space.
    expect(Array.from(permuted.position.slice(9, 12))).toEqual([10, 0, 0]);
  });

  it('does not re-extract into the previously-cached array (no in-place clobber)', () => {
    // The old array may still be bound to the live geometry from the previous
    // displayDims epoch, so a miss must allocate FRESH rather than overwrite it.
    const data = twoTrianglesIn4D();
    const first = projectMesh(
      data,
      viewState([0, 1, 2], [0, 0, 0, 0], [1e10, 1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    const firstCopy = Float32Array.from(first.position);
    projectMesh(
      data,
      viewState([3, 0, 1], [0, 0, 0, 0], [1e10, 1e10, 1e10, 1e10]),
      undefined,
      true,
      backend
    );
    // The first array's contents are untouched by the second (permuted) extraction.
    expect(Array.from(first.position)).toEqual(Array.from(firstCopy));
  });
});

describe('updateMeshGeometry — pure-slice-move reuse honours the memoized identity', () => {
  it('reuses the position attribute and its bounds when handed the same array', () => {
    // Build a real 3-vertex geometry, then commit two "slice moves" with the SAME
    // position array (as projectMesh now hands back) but different indices.
    const position = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const geom = buildMeshGeometry({
      position,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
    });
    const positionBefore = geom.getAttribute('position');

    // Box3 identity proves nothing here: three r184's computeBoundingBox() mutates
    // the existing Box3 in place (it only allocates when boundingBox === null), so
    // the object survives a recompute. Spy on the recompute methods instead — the
    // pure slice move must not call either.
    const boxSpy = vi.spyOn(geom, 'computeBoundingBox');
    const sphereSpy = vi.spyOn(geom, 'computeBoundingSphere');
    try {
      updateMeshGeometry(geom, {
        position, // same array — the memoised pure-slice-move projection
        indices: new Uint32Array([2, 1, 0]),
        colors: null,
        vertexCount: 3,
      });
      const rebuilt = updateMeshGeometry(geom, {
        position, // still the same array
        indices: new Uint32Array([0, 1, 2]),
        colors: null,
        vertexCount: 3,
      });

      // Same attribute object → the position buffer was neither re-uploaded nor rebound.
      expect(geom.getAttribute('position')).toBe(positionBefore);
      // The recompute is guarded behind the position rebind, so a pure slice move
      // never touches bounds.
      expect(boxSpy).not.toHaveBeenCalled();
      expect(sphereSpy).not.toHaveBeenCalled();
      // No vertex attribute rebound → the commit skips the WebGPU RenderObject eviction.
      expect(rebuilt).toBe(false);
    } finally {
      boxSpy.mockRestore();
      sphereSpy.mockRestore();
    }
  });

  it('does not warn on the expected first commit from the 1-vertex placeholder', () => {
    // The placeholder from createEmptyMeshNode is 1-vertex (length-3 position). The
    // first real commit grows it to N vertices — expected growth, gated by
    // `count > 1`, so it must stay quiet. Only a genuine mid-life vertex-count
    // change on an already-populated node warns.
    const placeholder = buildMeshGeometry({
      position: new Float32Array(3),
      indices: new Uint32Array(0),
      colors: null,
      vertexCount: 1,
    });
    const warnSpy = vi.spyOn(log, 'warning');
    try {
      updateMeshGeometry(placeholder, {
        position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices: new Uint32Array([0, 1, 2]),
        colors: null,
        vertexCount: 3,
      });
      const lengthWarnings = warnSpy.mock.calls.filter(([, message]) =>
        String(message).includes('position length changed')
      );
      expect(lengthWarnings).toHaveLength(0);
    } finally {
      warnSpy.mockRestore();
    }
  });

  it('rebinds and adopts the new array identity on a displayDims-change re-extraction', () => {
    // A displayDims change re-extracts the projection into a FRESH array. The update
    // must rebind (adopt B's identity) rather than copy into the old buffer, so that
    // the next same-`displayDims` epoch compares equal and skips the rebind.
    const a = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const geom = buildMeshGeometry({
      position: a,
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
    });

    const b = new Float32Array([10, 0, 0, 10, 1, 0, 10, 0, 1]);
    const rebuilt = updateMeshGeometry(geom, {
      position: b, // different array, same length — a displayDims-change re-extraction
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
    });

    // The geometry adopted B's identity (rebind, not copy-in-place).
    expect((geom.getAttribute('position') as THREE.BufferAttribute).array).toBe(b);
    // A vertex attribute was rebound → the commit must evict the WebGPU RenderObject.
    expect(rebuilt).toBe(true);

    // Handing back the SAME B is now a pure slice move: identity matches, nothing
    // rebinds. This is the regression guard against reverting to copy-in-place.
    const rebuiltAgain = updateMeshGeometry(geom, {
      position: b,
      indices: new Uint32Array([2, 1, 0]),
      colors: null,
      vertexCount: 3,
    });
    expect(rebuiltAgain).toBe(false);
  });

  it('warns on a genuine mid-life vertex-count change on a populated node', () => {
    // Growth from the 1-vertex placeholder stays quiet (count > 1 gate), but a real
    // vertex-count change on an already-populated node must still warn — the gate
    // must not regress to never-warn.
    const geom = buildMeshGeometry({
      position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      indices: new Uint32Array([0, 1, 2]),
      colors: null,
      vertexCount: 3,
    });
    const warnSpy = vi.spyOn(log, 'warning');
    try {
      updateMeshGeometry(geom, {
        position: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        colors: null,
        vertexCount: 4,
      });
      const lengthWarnings = warnSpy.mock.calls.filter(([, message]) =>
        String(message).includes('position length changed')
      );
      expect(lengthWarnings).toHaveLength(1);
    } finally {
      warnSpy.mockRestore();
    }
  });
});
