/**
 * Mesh Culling for nD → 3D Slicing
 *
 * TypeScript reference implementation matching `mesh_culling.rs`.
 *
 * ## Whole-triangle cull, not clipping
 *
 * A triangle is rendered **iff all three of its vertices pass the nD slab
 * membership test**. Unlike lines, nothing is interpolated and no new vertices
 * are created, so a cut boundary is ragged and triangle-quantized rather than a
 * clean planar section — a documented v1 trade (`docs/specs/MESH_NODE_SPEC.md`
 * §5.3).
 *
 * ## Buffer-shape contract
 *
 * `ndim`, `numVertices` and `numFaces` are trusted; the loader reconciles declared
 * shapes against materialized array lengths before calling (`MESH_NODE_SPEC.md`
 * §3.5 Stage 2). The two backends fail *differently* when that is violated, which
 * matters when reading a bug report: this implementation reads `undefined`, fails
 * the finite test and silently culls, whereas Rust bounds-checks even in release
 * and **traps** — taking down the whole WASM module, since the crate is
 * `panic = "abort"` (surfaced as `RuntimeError: unreachable`).
 *
 * ## This file is not only a fallback
 *
 * `mesh_vertex_visibility_mask` in the Rust crate calls `validate_ndim` and the
 * crate is built `panic = "abort"`, so it must never run above 16 dimensions.
 * `pickBackend(ctx, ndim)` (`workers/data-worker/state.ts`) therefore routes
 * every `ndim > 16` operation here. These implementations are uncapped and are
 * the **production** backend for high-dimensional data — keep them in 1:1
 * parity with the Rust kernels.
 */

/**
 * Compute per-vertex nD slab membership.
 *
 * For each non-displayed ("hidden") dimension `d`, with
 * `sliceMin = slicePosition[d] - tolerance[d]` and
 * `sliceMax = slicePosition[d] + tolerance[d]`, a vertex is **in** iff
 * `v[d] >= sliceMin && v[d] <= sliceMax` for *every* such `d`.
 *
 * A `NaN` or `±Inf` coordinate on any hidden dimension makes the vertex
 * invisible (the #806 rule). The explicit `Number.isFinite` test is
 * load-bearing rather than decorative: an infinite `tolerance` makes
 * `sliceMax = +Infinity`, and `+Infinity <= +Infinity` is **true**, so the bare
 * range test alone would report an infinite coordinate as visible.
 *
 * That rule covers the *coordinate* only, and the slab parameters behave the
 * OPPOSITE way: a `NaN` in `slicePosition` or `tolerance` makes every **finite**
 * vertex visible (both `value < NaN` and `value > NaN` are false, so the test
 * degenerates to "not non-finite" — it fails OPEN), while a **negative**
 * `tolerance` inverts the slab and culls everything. Both are caller bugs —
 * these are viewer-computed, not store-supplied — and both backends agree
 * exactly, so neither is guarded. Don't let a `NaN` tolerance derived from
 * absent dimension metadata reach here expecting it to be culled.
 *
 * @param positions - Vertex positions [numVertices * ndim]
 * @param slicePosition - Current slice position [ndim]
 * @param tolerance - Per-dimension tolerance [ndim]
 * @param displayDims - Which dimensions are displayed [numDisplayDims]
 * @param ndim - Number of dimensions (uncapped here)
 * @param numVertices - Number of vertices
 * @param output - Output visibility mask [numVertices] (1 = in, 0 = out)
 * @returns Number of visible vertices
 */
export function mesh_vertex_visibility_mask(
  positions: Float32Array,
  slicePosition: Float32Array,
  tolerance: Float32Array,
  displayDims: Uint32Array,
  ndim: number,
  numVertices: number,
  output: Uint8Array
): number {
  // No MAX_SUPPORTED_DIMS cap: this backend serves ndim > 16 in production.
  const isDisplayDim = new Uint8Array(ndim);
  for (let i = 0; i < displayDims.length; i++) {
    isDisplayDim[displayDims[i]] = 1;
  }

  // Hoist the slab bounds out of the per-vertex loop (they are
  // vertex-independent) and keep only the hidden dimensions, so the inner loop
  // runs over `numHidden` — usually 1, a timepoint or channel. Load-bearing:
  // measured 2.6x against the same kernel with the bounds recomputed per vertex
  // (1M vertices, 8D, 5 hidden dims). Don't fold it back in for brevity.
  const hiddenDims: number[] = [];
  const slabMin: number[] = [];
  const slabMax: number[] = [];
  for (let d = 0; d < ndim; d++) {
    if (isDisplayDim[d]) continue;
    hiddenDims.push(d);
    // `Math.fround` is REQUIRED for WASM parity, not a stylistic flourish.
    // Reading a Float32Array yields the exact f32 value widened to f64, and the
    // f64 difference of two f32 values is EXACT — whereas Rust computes the same
    // subtraction in f32 and rounds it. The gap is under half an ulp, so it is
    // invisible for almost every vertex; it becomes OBSERVABLE precisely when
    // the f32 rounding goes DOWN, because the rounded bound is then itself a
    // legal f32 vertex coordinate sitting inside the gap:
    //
    //   slice = 1.0, tolerance = 0.1 (both f32)
    //     exact f64 bound   = 0.8999999985098839
    //     f32 rounded bound = 0.8999999761581421   <- rounds DOWN
    //   A vertex at exactly 0.8999999761581421 is then `>=` the f32 bound
    //   (Rust: visible) but `<` the exact f64 bound (unfrounded JS: culled).
    //
    // Rounding to f32 here reproduces the Rust result bit-for-bit. Same
    // convention as `depth-sort.ts`, which frounds every step in the Rust
    // operation order to keep its permutation exact.
    slabMin.push(Math.fround(slicePosition[d] - tolerance[d]));
    slabMax.push(Math.fround(slicePosition[d] + tolerance[d]));
  }
  const numHidden = hiddenDims.length;

  // Fast path (§5.5): no hidden dimensions — the common plain-3D case — means
  // every vertex trivially passes.
  if (numHidden === 0) {
    for (let i = 0; i < numVertices; i++) output[i] = 1;
    return numVertices;
  }

  let visible = 0;

  for (let v = 0; v < numVertices; v++) {
    const base = v * ndim;
    let isIn = true;

    for (let h = 0; h < numHidden; h++) {
      const value = positions[base + hiddenDims[h]];
      // See the `Number.isFinite` note above — it is not subsumed by the range
      // test when the tolerance is infinite.
      if (!Number.isFinite(value) || value < slabMin[h] || value > slabMax[h]) {
        isIn = false;
        break;
      }
    }

    if (isIn) {
      output[v] = 1;
      visible++;
    } else {
      output[v] = 0;
    }
  }

  return visible;
}

/**
 * Compact `faces` to those whose three vertices are all visible.
 *
 * Writes **original (un-remapped)** vertex indices: on a slice change only the
 * index buffer is rebuilt, while the vertex attribute buffers stay uploaded in
 * full. `drawElements` never fetches an unreferenced vertex, so culled vertices
 * cost nothing to draw.
 *
 * The authored per-face index order is preserved exactly, so this is
 * winding-agnostic; restoring front-facing winding under a reflected display
 * permutation is a separate caller-owned post-pass (§5.4).
 *
 * A face index `>= vertexMask.length` drops the whole face. The values come
 * from the *store* — the viewer loads arbitrary, possibly corrupted datasets —
 * and the two backends fail differently without this guard: Rust would read out
 * of bounds and, being `panic = "abort"`, take down the entire WASM module,
 * while here the read would yield `undefined` and silently diverge. The loader
 * range-checks face indices up front and fails the node with a `LoaderError`
 * before reaching either backend (§3.5 Stage 2), so on the sanctioned path this
 * is unreachable; it is defense in depth, not a substitute for that gate.
 *
 * @param faces - Triangle vertex indices [numFaces * 3]
 * @param vertexMask - Per-vertex visibility from
 *   {@link mesh_vertex_visibility_mask}; its **length defines the valid vertex
 *   range**, so pass a view sized exactly `numVertices` rather than a larger
 *   reused scratch buffer
 * @param numFaces - Number of triangles
 * @param output - Output indices [numFaces * 3] worst case
 * @returns Number of visible faces written. **Slice `output` to `3 ×` this before
 *   use.** Everything past that point is left untouched — deliberately, to avoid a
 *   second pass — so a reused buffer still holds the previous frame's indices and a
 *   fresh one holds zeros. Uploading the whole buffer as an index range draws
 *   stale or degenerate triangles rather than nothing, which is the failure this
 *   return value exists to prevent.
 */
export function compact_visible_faces(
  faces: Uint32Array,
  vertexMask: Uint8Array,
  numFaces: number,
  output: Uint32Array
): number {
  const numVertices = vertexMask.length;
  let outFaces = 0;

  for (let f = 0; f < numFaces; f++) {
    const base = f * 3;
    // `>>> 0` (ToUint32) is a PARITY requirement, not defensive noise. On a real
    // `Uint32Array` it is the identity, so the sanctioned path is unchanged — but
    // the declared type is not a runtime guarantee, and the spec explicitly
    // contemplates externally produced SIGNED stores (§3.5 Stage 2). Given an
    // `Int32Array` holding `-1`, wasm-bindgen copies through `Uint32Array.set`,
    // so the WASM kernel sees `0xffffffff` and drops the face — whereas a raw
    // read here yields `-1`, which passes an upper-bound-only check and then
    // makes `vertexMask[-1]` `undefined`, i.e. `!== 0`, i.e. VISIBLE. That
    // emitted a face with a negative index in TS and none in WASM: the #806
    // divergence pattern, and it ships because this is the >16D production
    // backend. ToUint32 reproduces wasm-bindgen's coercion exactly — including
    // truncating a fractional index the same way — so both backends agree for
    // any numeric input.
    const i0 = faces[base] >>> 0;
    const i1 = faces[base + 1] >>> 0;
    const i2 = faces[base + 2] >>> 0;

    // Range-check before indexing the mask — see the note above. With the
    // ToUint32 normalisation this single upper-bound test also covers negatives.
    if (i0 >= numVertices || i1 >= numVertices || i2 >= numVertices) {
      continue;
    }

    if (vertexMask[i0] !== 0 && vertexMask[i1] !== 0 && vertexMask[i2] !== 0) {
      const dst = outFaces * 3;
      // Authored order preserved (winding-agnostic).
      output[dst] = i0;
      output[dst + 1] = i1;
      output[dst + 2] = i2;
      outFaces++;
    }
  }

  return outFaces;
}
