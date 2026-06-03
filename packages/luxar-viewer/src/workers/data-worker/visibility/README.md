# Worker-side nD Visibility Kernels

Per-geometry nD-visibility task functions invoked by `data-worker.ts` over
Comlink. Each kernel receives **already-decoded** geometry arrays plus the
current slice `(slicePosition, tolerance, ndim)` and returns a packed
`Uint8Array` visibility mask along with the visible count.

All three kernels share the same shape on purpose — Points, Lines, and
GSplats are sibling first-class geometries and the worker API surface
mirrors that symmetry (same signature, same buffer-reuse strategy, same
WASM dispatch contract).

## Files

```
visibility/
├── points.ts   — computeNDVisibilityPoints  (hypersphere ∩ slice)
├── lines.ts    — computeNDVisibilityLines   (either endpoint visible)
└── gsplats.ts  — computeNDVisibilityGSplats (Cholesky ellipsoid extent ∩ slice)
```

## Common contract

Every kernel:

1. Calls `requireWasm(ctx)` to obtain the WASM module (or its TypeScript
   fallback — see `../../../wasm/`).
2. Validates inputs against `ndim` via helpers in `../validation.ts`
   (`validateNDArrays` for points/gsplats, `validateLineSegmentReferences`
   for lines). Lines additionally enforce `ndim ∈ [1, MAX_WASM_DIMS]` and
   that `slicePosition` / `tolerance` are long enough.
3. Grows the pooled `ctx.visibilityMaskBuffer` (allocating
   `ceil(n * 1.5)` headroom on miss) and writes the new size back so the
   next call reuses the same backing store. This is the only state these
   kernels mutate on `ctx`.
4. Dispatches to the matching WASM entry point
   (`compute_nd_visibility_{points,lines,gsplats}`), which writes the
   per-element mask into the pooled buffer and returns the visible count.
5. Returns `{ visibilityMask: buf.subarray(0, n), visibleCount }`. The
   subarray is a zero-copy view into the pooled buffer — callers must
   consume or copy it before the next visibility call on the same worker.

## Geometry-specific notes

| Kernel    | Semantics                                                                                                                      | Extra inputs vs. points           |
| --------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------- |
| `points`  | Per-point hypersphere of radius `radii[i]` intersects slice.                                                                   | `radii: Float32Array`             |
| `lines`   | Segment is visible iff EITHER endpoint is visible.                                                                             | `segments: Uint32Array`, `widths` |
| `gsplats` | Ellipsoid extent derived from packed lower-triangular Cholesky factors (`ndim*(ndim+1)/2` entries per splat) intersects slice. | `choleskyFactors: Float32Array`   |

## Dependencies

- Internal: `../state` (`WasmCtx`, `requireWasm`, `visibilityMaskBuffer`
  pool), `../validation` (`validateNDArrays`,
  `validateLineSegmentReferences`, `MAX_WASM_DIMS`).
- External: none — input arrays are pre-decoded transferables and
  results are pooled buffers, so no allocator or copy happens inside
  these files.

## See Also

- [`../../README.md`](../../README.md) — worker pool, Comlink API surface,
  and the `runWithTimeout('...', 'visibility', ...)` dispatch path.
- [`../projection/`](../projection/) — the projection kernels that
  consume these visibility masks to build final transferable buffers.
- [`../../../wasm/`](../../../wasm/) — WASM module and TS fallback that
  implement the `compute_nd_visibility_*` entry points.
