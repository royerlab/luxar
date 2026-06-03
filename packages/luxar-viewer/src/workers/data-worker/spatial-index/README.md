# data-worker / spatial-index

Spatial-index chunk queries running inside the data worker.

This leaf holds the worker-side implementation of `querySpatialIndex`,
which finds the set of zarr chunks intersecting the current nD view
frustum. The kernel is geometry-agnostic — Points, Lines, and GSplats
all encode their chunk bounding boxes the same way, so the same WASM
entry point (`query_chunks_for_view`) serves all three geometry types.

## Files

```
spatial-index/
└── query.ts   — querySpatialIndex(ctx, params) → Uint32Array of matching chunk indices
```

## API

`querySpatialIndex(ctx: WasmCtx, params)` takes:

- `chunkBounds: Float32Array` — packed `[min, max]` per dimension per chunk.
- `slicePosition: Float32Array` — current nD slice center.
- `tolerance: Float32Array` — per-dimension half-extent of the query
  hyperbox (effectively infinite on displayed dims; geometry-aware on
  hidden dims — see `data/loaders/tolerance-computer.ts`).
- `numChunks: number`, `ndim: number`.

Returns a `Uint32Array` of matching chunk indices (a subarray view into
a pre-allocated max-size buffer; do not retain across worker calls).

## Boundary contract

Inputs are validated by `validation.ts::validateChunkQueryInputs`
before crossing the JS→WASM boundary. The compiled Rust kernel reads
buffers using lengths the JS side promised — an off-by-one here would
become a wild read inside WASM. Callers must populate `chunkBounds`
with exactly `2 * ndim * numChunks` floats and respect
`MAX_WASM_DIMS = 16`.

`requireWasm(ctx)` will throw if `initialize.ts` has not run; the
worker entry (`data-worker.ts`) guarantees this ordering via Comlink.

## See Also

- [../../README.md](../../README.md) — worker pool, `runWithTimeout`,
  and the full Comlink surface.
- [../../../data/loaders/README.md](../../../data/loaders/README.md) —
  `SpatialQueryBuilder`, the main-thread caller that constructs the
  `slicePosition` / `tolerance` arrays consumed here.
- [../state.ts](../state.ts) — `WasmCtx` definition and `requireWasm`.
- [../validation.ts](../validation.ts) — boundary checks.
  </content>
  </invoke>
