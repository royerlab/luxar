/**
 * Tests for `src/wasm/typescript/mesh-culling.ts`
 * (whole-triangle nD visibility culling: mesh_vertex_visibility_mask,
 * compact_visible_faces).
 *
 * Mirrors the Rust `#[cfg(test)] mod tests` in `wasm/rust/src/mesh_culling.rs`
 * case for case. This backend is not merely a WASM-missing fallback — it is the
 * production backend for `ndim > 16` (`pickBackend`), so the >16D cases below
 * exercise a shipping path, not a hypothetical one.
 */

import { describe, it, expect } from 'vitest';
import { mesh_vertex_visibility_mask, compact_visible_faces } from '../../../../wasm/typescript';

/** 4D positions with dim 3 hidden — the common (x, y, z, t) shape. */
function mask4d(
  positions: number[],
  sliceW: number,
  tolW: number,
  numVertices: number
): { mask: Uint8Array; visible: number } {
  const output = new Uint8Array(numVertices);
  const visible = mesh_vertex_visibility_mask(
    new Float32Array(positions),
    new Float32Array([0, 0, 0, sliceW]),
    new Float32Array([1e10, 1e10, 1e10, tolW]),
    new Uint32Array([0, 1, 2]),
    4,
    numVertices,
    output
  );
  return { mask: output, visible };
}

// ============================================================================
// mesh_vertex_visibility_mask
// ============================================================================

describe('mesh_culling: mesh_vertex_visibility_mask', () => {
  it('marks every vertex visible when there are no hidden dimensions', () => {
    // §5.5 fast path: displayDims.length === ndim.
    const positions = new Float32Array([0, 0, 0, 1, 2, 3, -5, 9, 100]);
    const output = new Uint8Array(3);

    const visible = mesh_vertex_visibility_mask(
      positions,
      new Float32Array([50, 50, 50]), // slice far from everything
      new Float32Array([0, 0, 0]), // zero tolerance
      new Uint32Array([0, 1, 2]),
      3,
      3,
      output
    );

    expect(visible).toBe(3);
    expect(Array.from(output)).toEqual([1, 1, 1]);
  });

  it('includes a vertex inside the slab and excludes one outside', () => {
    const { mask, visible } = mask4d([1, 2, 3, 5, 1, 2, 3, 9], 5.0, 0.5, 2);

    expect(visible).toBe(1);
    expect(Array.from(mask)).toEqual([1, 0]);
  });

  it('treats the slab bounds as CLOSED (inclusive)', () => {
    // Half-cell discrete tolerances put on-grid geometry exactly on the bound,
    // so an exclusive comparison here would drop whole timepoints.
    const { mask, visible } = mask4d([1, 2, 3, 4.5, 1, 2, 3, 5.5], 5.0, 0.5, 2);

    expect(visible).toBe(2);
    expect(Array.from(mask)).toEqual([1, 1]);
  });

  it('excludes a vertex one ulp outside either bound', () => {
    // One ulp at 4.5/5.5, computed rather than assumed: `Number.EPSILON` is the
    // f64 ulp at 1.0 and would be swallowed entirely by the f32 round-trip.
    const below = new Float32Array([4.5]);
    below[0] = Math.fround(4.5 - Math.pow(2, -21));
    const above = new Float32Array([5.5]);
    above[0] = Math.fround(5.5 + Math.pow(2, -21));
    expect(below[0]).not.toBe(4.5);
    expect(above[0]).not.toBe(5.5);

    const { mask, visible } = mask4d([1, 2, 3, below[0], 1, 2, 3, above[0]], 5.0, 0.5, 2);

    expect(visible).toBe(0);
    expect(Array.from(mask)).toEqual([0, 0]);
  });

  it('degenerates to exact equality at zero tolerance', () => {
    // This is WHY mesh cannot reuse the Lines spatial tolerance of 0 (§5.2.1):
    // with no interpolation a zero-thickness slab renders essentially nothing.
    const { mask, visible } = mask4d([1, 2, 3, 5, 1, 2, 3, 5.000001], 5.0, 0.0, 2);

    expect(visible).toBe(1);
    expect(Array.from(mask)).toEqual([1, 0]);
  });

  it('fails OPEN on a NaN slab parameter — the opposite of a NaN coordinate', () => {
    // Both `value < NaN` and `value > NaN` are false, so the slab test
    // degenerates to "not non-finite". Pinned so nobody adds a one-sided guard
    // in one backend and silently breaks parity; the Rust kernel asserts the same.
    const far = [1, 2, 3, 999]; // far outside any sane slab
    expect(mask4d(far, NaN, 0.5, 1).visible).toBe(1); // NaN slice position
    expect(mask4d(far, 5.0, NaN, 1).visible).toBe(1); // NaN tolerance
  });

  it('culls everything when the tolerance is negative (inverted slab)', () => {
    // min > max, so even a vertex exactly on the slice is out.
    const { mask, visible } = mask4d([1, 2, 3, 5], 5.0, -1.0, 1);

    expect(visible).toBe(0);
    expect(Array.from(mask)).toEqual([0]);
  });

  it('culls a NaN coordinate on a hidden dimension (#806)', () => {
    const { mask, visible } = mask4d([1, 2, 3, NaN], 5.0, 1e10, 1);

    expect(visible).toBe(0);
    expect(Array.from(mask)).toEqual([0]);
  });

  it.each([
    ['+Infinity', Infinity],
    ['-Infinity', -Infinity],
  ])('culls %s on a hidden dimension (#806)', (_label, w) => {
    const { mask, visible } = mask4d([1, 2, 3, w], 5.0, 1e10, 1);

    expect(visible).toBe(0);
    expect(Array.from(mask)).toEqual([0]);
  });

  it('still culls an infinite coordinate when the tolerance is infinite', () => {
    // The case a bare range test gets WRONG: an infinite tolerance makes
    // sliceMax === +Infinity, and `+Infinity <= +Infinity` is true. Only the
    // explicit Number.isFinite test keeps this vertex culled.
    const { mask, visible } = mask4d([1, 2, 3, Infinity], 5.0, Infinity, 1);

    expect(visible).toBe(0);
    expect(Array.from(mask)).toEqual([0]);
  });

  it('does NOT cull a non-finite coordinate on a DISPLAYED dimension', () => {
    // The rule is scoped to hidden dims: a non-finite displayed coordinate is a
    // rendering problem, not a slicing one, and no sibling loader finite-scans
    // its decoded positions either (§3.5).
    const { mask, visible } = mask4d([NaN, 2, Infinity, 5], 5.0, 0.5, 1);

    expect(visible).toBe(1);
    expect(Array.from(mask)).toEqual([1]);
  });

  it('ANDs membership across every hidden dimension', () => {
    // 5D, display [0,1,2] → dims 3 and 4 hidden. One vertex per row.
    // prettier-ignore
    const positions = new Float32Array([
      0, 0, 0, 5, 7, // v0: both in
      0, 0, 0, 5, 9, // v1: dim 4 out
      0, 0, 0, 1, 7, // v2: dim 3 out
      0, 0, 0, 1, 9, // v3: both out
    ]);
    const output = new Uint8Array(4);

    const visible = mesh_vertex_visibility_mask(
      positions,
      new Float32Array([0, 0, 0, 5, 7]),
      new Float32Array([1e10, 1e10, 1e10, 0.5, 0.5]),
      new Uint32Array([0, 1, 2]),
      5,
      4,
      output
    );

    expect(visible).toBe(1);
    expect(Array.from(output)).toEqual([1, 0, 0, 0]);
  });

  it('treats displayDims as a SET — a permuted order selects the same hidden dims', () => {
    // Which display dim maps to renderer X/Y/Z is the projection's business.
    const positions = new Float32Array([1, 2, 3, 9]);
    const slicePos = new Float32Array([0, 0, 0, 5]);
    const tolerance = new Float32Array([1e10, 1e10, 1e10, 0.5]);
    const ascending = new Uint8Array(1);
    const permuted = new Uint8Array(1);

    mesh_vertex_visibility_mask(
      positions,
      slicePos,
      tolerance,
      new Uint32Array([0, 1, 2]),
      4,
      1,
      ascending
    );
    mesh_vertex_visibility_mask(
      positions,
      slicePos,
      tolerance,
      new Uint32Array([2, 0, 1]),
      4,
      1,
      permuted
    );

    expect(Array.from(ascending)).toEqual(Array.from(permuted));
    expect(Array.from(ascending)).toEqual([0]);
  });

  it('handles a LEADING hidden dimension', () => {
    // 4D, display [1,2,3] → dim 0 hidden.
    const output = new Uint8Array(2);
    const visible = mesh_vertex_visibility_mask(
      new Float32Array([5, 1, 2, 3, 9, 1, 2, 3]),
      new Float32Array([5, 0, 0, 0]),
      new Float32Array([0.5, 1e10, 1e10, 1e10]),
      new Uint32Array([1, 2, 3]),
      4,
      2,
      output
    );

    expect(visible).toBe(1);
    expect(Array.from(output)).toEqual([1, 0]);
  });

  it.each([
    ['2D data, both dims shown (no hidden)', [0, 0, 5, 5], [5, 5], [0.5, 0.5], [0, 1], 2, [1, 1]],
    ['3D data, 2 dims shown', [0, 0, 5, 0, 0, 9], [0, 0, 5], [1e10, 1e10, 0.5], [0, 1], 3, [1, 0]],
    ['1D shown of 2D', [0, 5, 0, 9], [0, 5], [1e10, 0.5], [0], 2, [1, 0]],
    ['ZERO dims shown (all hidden)', [5, 5, 9, 9], [5, 5], [0.5, 0.5], [], 2, [1, 0]],
  ])('handles fewer than 3 display dims: %s', (_label, pos, slice, tol, dd, ndim, expected) => {
    // The dimension hazard is two-sided: >16D panics in WASM, and a hardcoded
    // sub-ndim of 3 has crashed a sibling kernel on 2D data (#881). This kernel
    // reads only hidden dims and never builds a display marginal, so it is
    // structurally immune — these cases keep it that way.
    const numVertices = (pos as number[]).length / (ndim as number);
    const output = new Uint8Array(numVertices);
    const visible = mesh_vertex_visibility_mask(
      new Float32Array(pos as number[]),
      new Float32Array(slice as number[]),
      new Float32Array(tol as number[]),
      new Uint32Array(dd as number[]),
      ndim as number,
      numVertices,
      output
    );

    expect(Array.from(output)).toEqual(expected);
    expect(visible).toBe((expected as number[]).filter((v) => v === 1).length);
  });

  it('admits the whole finite axis at EXTEND_TO_ALL_TOLERANCE (1e10)', () => {
    const { mask, visible } = mask4d([1, 2, 3, -1e6, 1, 2, 3, 1e6], 0.0, 1e10, 2);

    expect(visible).toBe(2);
    expect(Array.from(mask)).toEqual([1, 1]);
  });

  it('OVERWRITES the output mask rather than OR-ing into it', () => {
    // A reused buffer must not leak stale 1s.
    const output = new Uint8Array([1]);
    const visible = mesh_vertex_visibility_mask(
      new Float32Array([1, 2, 3, 9]),
      new Float32Array([0, 0, 0, 5]),
      new Float32Array([1e10, 1e10, 1e10, 0.5]),
      new Uint32Array([0, 1, 2]),
      4,
      1,
      output
    );

    expect(visible).toBe(0);
    expect(Array.from(output)).toEqual([0]);
  });

  it('handles zero vertices', () => {
    const visible = mesh_vertex_visibility_mask(
      new Float32Array(0),
      new Float32Array([0, 0, 0, 5]),
      new Float32Array([1e10, 1e10, 1e10, 0.5]),
      new Uint32Array([0, 1, 2]),
      4,
      0,
      new Uint8Array(0)
    );

    expect(visible).toBe(0);
  });

  it('handles ndim > 16, which the WASM kernel cannot (this is the production path)', () => {
    // 20D: display [0,1,2], dims 3..19 hidden. The Rust kernel panics here via
    // validate_ndim, so pickBackend routes to this implementation.
    const ndim = 20;
    const positions = new Float32Array(2 * ndim);
    const slicePos = new Float32Array(ndim);
    const tolerance = new Float32Array(ndim).fill(0.5);
    for (const d of [0, 1, 2]) tolerance[d] = 1e10;
    // v0 sits on every hidden slice; v1 is off on the LAST dimension only.
    positions[ndim + (ndim - 1)] = 9;
    const output = new Uint8Array(2);

    const visible = mesh_vertex_visibility_mask(
      positions,
      slicePos,
      tolerance,
      new Uint32Array([0, 1, 2]),
      ndim,
      2,
      output
    );

    expect(visible).toBe(1);
    expect(Array.from(output)).toEqual([1, 0]);
  });
});

// ============================================================================
// compact_visible_faces
// ============================================================================

describe('mesh_culling: compact_visible_faces', () => {
  it('keeps every face when all vertices are visible', () => {
    const faces = new Uint32Array([0, 1, 2, 1, 2, 3]);
    const output = new Uint32Array(6);

    const kept = compact_visible_faces(faces, new Uint8Array([1, 1, 1, 1]), 2, output);

    expect(kept).toBe(2);
    expect(Array.from(output)).toEqual([0, 1, 2, 1, 2, 3]);
  });

  it('drops a face with any invisible vertex', () => {
    const faces = new Uint32Array([0, 1, 2, 1, 2, 3]);
    const output = new Uint32Array(6);

    const kept = compact_visible_faces(faces, new Uint8Array([1, 1, 1, 0]), 2, output);

    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([0, 1, 2]);
  });

  it.each([7, 8, 9])('drops the face when vertex %i alone is hidden', (hidden) => {
    // Each of the three slots is checked individually, so a partial predicate
    // (e.g. testing only i0 and i1) cannot pass.
    const mask = new Uint8Array(10).fill(1);
    mask[hidden] = 0;
    const output = new Uint32Array(3);

    const kept = compact_visible_faces(new Uint32Array([7, 8, 9]), mask, 1, output);

    expect(kept).toBe(0);
  });

  it('keeps nothing when no vertex is visible', () => {
    const output = new Uint32Array(3);
    const kept = compact_visible_faces(
      new Uint32Array([0, 1, 2]),
      new Uint8Array([0, 0, 0]),
      1,
      output
    );

    expect(kept).toBe(0);
  });

  it('writes ORIGINAL vertex indices, not indices remapped to a compacted array', () => {
    const faces = new Uint32Array([0, 1, 2, 5, 6, 7]);
    const mask = new Uint8Array([0, 0, 0, 0, 0, 1, 1, 1]);
    const output = new Uint32Array(6);

    const kept = compact_visible_faces(faces, mask, 2, output);

    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([5, 6, 7]);
  });

  it('preserves the authored per-face index order (winding-agnostic)', () => {
    const output = new Uint32Array(3);
    const kept = compact_visible_faces(
      new Uint32Array([2, 0, 1]), // deliberately not sorted
      new Uint8Array([1, 1, 1]),
      1,
      output
    );

    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([2, 0, 1]);
  });

  it('is order-stable across faces', () => {
    const faces = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const mask = new Uint8Array([1, 1, 1, 0, 1, 1, 1, 1, 1]);
    const output = new Uint32Array(9);

    const kept = compact_visible_faces(faces, mask, 3, output);

    expect(kept).toBe(2);
    expect(Array.from(output.subarray(0, 6))).toEqual([0, 1, 2, 6, 7, 8]);
  });

  it('drops out-of-range face indices instead of reading past the mask', () => {
    // The Rust sibling would abort the whole WASM module on this input; here an
    // unguarded read would yield `undefined`. Both must simply drop the face.
    // The trailing VALID face is the point of the layout: a bad index must skip
    // only that face. Were it a `break`, one corrupt index early in the array
    // would silently discard every valid face after it, and a test whose only
    // valid face came first could not tell the two apart.
    // prettier-ignore
    const faces = new Uint32Array([
      0, 1, 2,  // valid
      0, 1, 3,  // index 3 out of range (mask has 3 entries)
      99, 0, 1, // wildly out of range
      2, 1, 0,  // valid, AFTER the bad ones
    ]);
    const output = new Uint32Array(12);

    const kept = compact_visible_faces(faces, new Uint8Array([1, 1, 1]), 4, output);

    expect(kept).toBe(2);
    expect(Array.from(output.subarray(0, 6))).toEqual([0, 1, 2, 2, 1, 0]);
  });

  it('coerces indices with ToUint32 so a SIGNED source matches WASM', () => {
    // The declared `Uint32Array` is not a runtime guarantee, and the spec
    // contemplates externally produced signed stores (§3.5 Stage 2). Without the
    // `>>> 0`, `-1` passed this upper-bound-only guard and `vertexMask[-1]` was
    // `undefined` (`!== 0`, so "visible") — emitting a face with a negative index
    // here while WASM, seeing wasm-bindgen's `0xffffffff`, dropped it. That is the
    // #806 divergence pattern, and it ships because this is the >16D backend.
    const mask = new Uint8Array([1, 1, 1]);
    const output = new Uint32Array(3);
    const signed = new Int32Array([0, 1, -1]) as unknown as Uint32Array;

    expect(compact_visible_faces(signed, mask, 1, output)).toBe(0);

    // Fractional indices truncate exactly as wasm-bindgen's coercion does.
    const fractional = [0.9, 1.2, 2.7] as unknown as Uint32Array;
    expect(compact_visible_faces(fractional, mask, 1, output)).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([0, 1, 2]);
  });

  it('drops a 0xffffffff index (a signed -1 reinterpreted during coercion)', () => {
    const output = new Uint32Array(3);
    const kept = compact_visible_faces(
      new Uint32Array([0, 1, 0xffffffff]),
      new Uint8Array([1, 1, 1]),
      1,
      output
    );

    expect(kept).toBe(0);
  });

  it('treats vertexMask.length as the valid-vertex range, not a separate count', () => {
    // 4 real vertices in a 16-slot scratch buffer whose tail is stale 1s. Passed
    // WHOLE, the stale tail counts as real; sliced to the true count, it doesn't.
    // Pinned because the doc comment promises exactly this, and a caller reusing
    // one buffer across meshes must subarray it.
    const scratch = new Uint8Array(16).fill(1);
    scratch.fill(0, 0, 4);
    scratch[1] = 1; // the only genuinely visible real vertex
    const faces = new Uint32Array([1, 1, 1, 9, 10, 11]); // 2nd face is in the stale tail
    const output = new Uint32Array(6);

    expect(compact_visible_faces(faces, scratch, 2, output)).toBe(2);

    const kept = compact_visible_faces(faces, scratch.subarray(0, 4), 2, output);
    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([1, 1, 1]);
  });

  it('keeps a degenerate (repeated-index) face when its vertices are visible', () => {
    // Culling is not the place to fix topology.
    const output = new Uint32Array(3);
    const kept = compact_visible_faces(
      new Uint32Array([1, 1, 1]),
      new Uint8Array([1, 1]),
      1,
      output
    );

    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([1, 1, 1]);
  });

  it('handles zero faces', () => {
    const kept = compact_visible_faces(
      new Uint32Array(0),
      new Uint8Array([1]),
      0,
      new Uint32Array(0)
    );

    expect(kept).toBe(0);
  });

  it('treats any non-zero mask value as visible', () => {
    // Matches every sibling mask consumer (`compact_by_mask`, `count_visible`).
    const output = new Uint32Array(3);
    const kept = compact_visible_faces(
      new Uint32Array([0, 1, 2]),
      new Uint8Array([2, 255, 7]),
      1,
      output
    );

    expect(kept).toBe(1);
  });
});

// ============================================================================
// The two kernels composed — the shape the loader actually uses
// ============================================================================

describe('mesh_culling: mask then compact', () => {
  it('culls a triangle that straddles two timepoints, keeping original indices', () => {
    // 4D quad (2 triangles): v0..v2 at t=0, v3 at t=1. Slicing to t=0 keeps
    // only the triangle entirely at t=0.
    // prettier-ignore
    const positions = new Float32Array([
      0, 0, 0, 0, // v0 t=0
      1, 0, 0, 0, // v1 t=0
      0, 1, 0, 0, // v2 t=0
      1, 1, 0, 1, // v3 t=1
    ]);
    const faces = new Uint32Array([0, 1, 2, 1, 3, 2]);
    const mask = new Uint8Array(4);

    const visibleVertices = mesh_vertex_visibility_mask(
      positions,
      new Float32Array([0, 0, 0, 0]),
      // Half-cell membership tolerance for a discrete axis of step 1.
      new Float32Array([1e10, 1e10, 1e10, 0.5]),
      new Uint32Array([0, 1, 2]),
      4,
      4,
      mask
    );
    expect(visibleVertices).toBe(3);
    expect(Array.from(mask)).toEqual([1, 1, 1, 0]);

    const output = new Uint32Array(6);
    const kept = compact_visible_faces(faces, mask, 2, output);

    expect(kept).toBe(1);
    expect(Array.from(output.subarray(0, 3))).toEqual([0, 1, 2]);
  });
});
