# Mesh data path

Mesh-specific loading, admission validation, and display-space projection. The
wider data pipeline lives in `src/data/`; this folder holds the parts specific to
the Mesh node type.

Spec: `docs/specs/MESH_NODE_SPEC.md` (§3.5 validation, §5 nD semantics, §7
loading).

## Files

| File                        | Role                                                                                                                                                                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mesh-whole-node-loader.ts` | `MeshWholeNodeLoader` — the whole-node loader. Opens metadata handles, runs Stage 1, fetches and decodes every array in full, runs Stage 2, then caches the result for the loader's lifetime. Implements `MeshDataLoader`.                           |
| `preflight.ts`              | Stage 1: the **metadata preflight**. Vertex cap, byte budget, shape/dtype cross-checks, `normal_dims` well-formedness — all decided from `.zarray`/`.zattrs` alone, with **no chunk fetched**. Also exports `parseDtype`.                            |
| `validate.ts`               | Stage 2: post-decode value checks. Materialized lengths, and the two-sided face-index range check that runs on the **source-typed** values before the u32 coercion.                                                                                  |
| `projection.ts`             | `projectMeshTo3D` — display-space `position` extraction, the whole-triangle nD cull (via the B2 kernels), the no-hidden-dims fast path, and the winding post-pass. Also `resolveWinding` (pure, exhaustively tested) and `noticeUndecidableWinding`. |

## Why this folder is so much smaller than `data/lines/`

Because a mesh has no per-slice working set to exploit.
`lines-spatial-index-loader.ts` runs to ~1400 LOC because line datasets reach
tens of millions of vertices and a slice change genuinely needs only a fraction
of them, so a dual chunk index earns its complexity. Mesh faces share vertices
across any cut, so the working set after a `displayDims` change is the whole mesh
regardless — an index would add machinery and skip nothing.

That is a property of the geometry, not a v1 shortcut. If it ever stops holding,
`MeshWholeNodeLoader` implements the full `MeshDataLoader` interface, so a spatial-index
implementation drops in behind it with no caller change.

## The two-stage admission gate

The writer's `validate_*_for_writing` family protects only stores Luxar produced.
The viewer loads arbitrary `?src=` URLs, so the loader validates independently —
and it has to do so in two stages, because the whole-node design means a check
that runs after decode arrives too late for the quantities that gate admission. A
hostile store can declare enormous arrays and exhaust tab memory before the
per-node `LoaderError` containment is ever reachable.

| Stage | Runs on                    | Catches                                                                                                                                                            |
| ----- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1     | attrs + `.zarray` metadata | `n_vertices > 2^27`; over-budget stored, decoded or per-chunk bytes; wrong logical shapes/dtypes; malformed `normal_dims`; a presence flag with no array behind it |
| 2     | the materialized arrays    | short arrays; face indices outside `[0, V)`                                                                                                                        |

Both fail the node with a `LoaderError` of kind `Validation` — one node lost, not
the scene. The kind is load-bearing: it is persisted on the failure record, so the
retry policy treats a malformed store as deterministic instead of re-fetching it
on every reconnect.

Three details in here are easy to get wrong and are pinned by tests:

- **`faces` is read RAW, never through `ArrayDecoder`.** The decoder returns
  `Float32Array`, whose 24-bit mantissa cannot represent every index a
  2^27-vertex mesh may carry — it would silently round anything above 16,777,216.
- **The face-index check is two-sided and pre-coercion.** A signed store's `-1`
  passes a one-sided `< V` test and wraps to `0xffffffff`; a 64-bit store's
  `2^32 + 1` survives a post-cast check because it wraps to `1`, lands inside
  range, and rewrites topology instead of trapping. An out-of-range index
  **panics** the Rust kernel (`panic = "abort"`, so it takes down the whole WASM
  module, not one node) and silently corrupts the TypeScript one.
- **The budget fails CLOSED on an unknown encoding.** This one is structural rather
  than a single bug fix: the gate was bypassed four times by _one_ category — the bytes
  the loader fetches are not the bytes the handle declares. `broadcasted` expands a
  `(1, k)` stub to `n_elements` rows; a narrow dtype widens 4x on decode; an unbounded
  `ndim` inflates the logical count; `array_ref` reads a different array entirely. Each
  was fixed as an instance and a fourth still appeared, because an unrecognised encoding
  fell through to "use the stored shape". The set is now closed against the contract's
  `ENCODING_NAMES` via a `Record<EncodingName, ...>`, so a new contract encoding is a
  **compile error here** and an unknown one at runtime is refused. It earned its keep
  immediately: it caught a missing `uint64` entry on first compile.
- **An `array_ref` is budgeted at its TARGET.** `ArrayDecoder` resolves
  `encoding.target` against the store root and reads _that_ array in full, while the
  referring array is a stub the writer emits at `(0, k)` — so charging the handle the
  loader opened budgets ~48 bytes for a read that can pull gigabytes. Stage 1 follows
  the chain metadata-only (`zarr.open` reads `.zarray`/`.zattrs` and no chunk), with a
  hop limit and a seen-set so a cyclic store is refused rather than hung. Rejecting
  `array_ref` outright would not do: `normals`/`colors`/`scalars` are written with
  dedup ON, so two meshes sharing a colour array legitimately produce a ref the viewer
  must still load.
- **The byte budget adds three terms rather than checking them separately.** Stored
  bytes, decoded bytes, and the largest single chunk buffer are SUMMED, because a
  chunk buffer exists during decode alongside the arrays. Checking the chunk term
  independently lets a store sit just under budget on both and peak near 2x — and
  zarr v2's `chunks > shape` allowance makes that directly constructible. (An
  oversized chunk is _also_ rejected on its own, so one array can never exceed the
  ceiling even where the sum would fit.)
- **The budget must charge what arrays DECODE to, not just what they store.**
  The stored side can be arbitrarily smaller than the allocation: a broadcast array
  stores one row and expands to `n_elements` rows, so a ~12-byte declaration can
  materialize gigabytes; every decoder-routed array yields a `Float32Array`, so a
  `uint8` store decodes at 4x; and `faces` widens to u32 whatever narrow dtype the
  `INDEX` encoder chose. The decoded term is also the only thing bounding `ndim`,
  which has no cap of its own. The stored term still reads the **declared** dtype
  (never a canonical one — `int64` costs 8 bytes per index), and the per-chunk term
  is separate because zarr v2 does not require `chunks <= shape`, so a 100-triangle
  array can declare a 268M-triangle chunk.
- **Shapes are checked LOGICALLY, and "logical" means three sources.**
  `encoding.n_elements` (broadcast) wins first, then `encoding.original_shape`
  (LUT / per-channel quantization), then the stored shape. Consulting only
  `original_shape` rejects a uniform colour: the broadcast encoder stamps
  `n_elements` and _not_ `original_shape`, so `add_mesh(..., colors=(1, 0, 0))` —
  and any incidentally-uniform colour array — lands as `shape: [1, 3]` and looks
  like a 1-row array.

At the default 512 MiB budget the **budget binds long before the vertex cap**: a
3D float32 mesh runs out of bytes at ~22.4M vertices, well under 2^27 (134.2M) —
half the stored-only arithmetic, because the decoded term is charged as well.
The cap is still checked, and checked first, so a nonsensical declaration gets the
message that names the real problem (pick-key aliasing) rather than blaming bytes.

## nD semantics: whole-triangle cull

A triangle renders **iff all three of its vertices pass the nD slab test**. No
clipping, no re-triangulation, no attribute interpolation — and no vertex
compaction either: only the _index_ buffer is rebuilt on a slice change, so
culled vertices cost nothing to draw and the loader gets to keep native-dtype
colours (a generic `&[f32]`-only mask compaction, as the former `compact_by_mask`
was, could not).

The price is a **ragged, triangle-quantized cut boundary** rather than a clean
planar section. On a well-tessellated mesh with a tolerance comparable to the edge
length that reads as a slightly jagged edge; on a coarse mesh with a thin
tolerance it can drop whole regions. Exact nD clipping is explicitly out of scope
for v1 (spec §9).

`projectMeshTo3D` takes its backend as a parameter; the caller selects it through
`pickBackend`, so `ndim > 16` runs the TypeScript reference. Note that
`pickBackend` swaps the **whole module**, which is why both cull kernels must
exist on `WasmModule` and in both backends.

## Winding

Winding is decidable only against the authored winding frame, which is
`sorted(normal_dims)`. Three cases, and they are genuinely different:

| Displayed triple vs frame              | Action                                                           |
| -------------------------------------- | ---------------------------------------------------------------- |
| Same triple, even parity               | Draw as authored                                                 |
| Same triple, odd parity                | Swap two of each triangle's three indices (`side: 'front'` kept) |
| A different triple, or no frame at all | `side: 'double'` for the epoch + a one-time notice               |

The reversal is keyed to the _current_ `displayDims` parity, not to the event of
`displayDims` changing, so it runs on **every** index build in an odd-parity epoch
— initial load and slice moves included. Nothing restricts the opening view to
ascending order, so the very first build can already need it. Without it a
`double_sided: false` mesh renders inside-out, which for an open surface means it
vanishes.

Two traps: swapping all three indices is a rotation and leaves winding
_unchanged_; and reversing in the undecidable case is worse than doing nothing,
because it flips the triangles that were already correct.

## The in-flight latch is identity-guarded

`load()` collapses concurrent callers onto one fetch. Clearing that latch on
completion has to check that the latch is still _the completing promise's own_:
`dispose()` nulls `inFlight` while the old promise may still be pending, so an
unconditional clear lets a stale completion wipe a REPLACEMENT load's latch. From
then until the new fetch publishes, every `updateView` — a slice scrub, precisely
what the latch exists for — starts another whole-mesh fetch. Dispose-then-reload is
a designed, tested path here, so this is a live race rather than a theoretical one.

Note the two guards are separate and both needed: a generation token stops a stale
completion _publishing_ into a disposed loader, and the identity check stops it
_erasing the latch_. Fixing only the first leaves the duplicate fetches.

## Public surface

`MeshWholeNodeLoader` implements `MeshDataLoader` (`types/mesh.ts`) — the same shape as the
Points/Lines/GSplats facades. Scene-loader code goes through
`scene-loader/loaders/loader-factory.ts` rather than importing the concrete class.

`updateView` never re-fetches. The mesh is resident in full, so a view change has
no subset to fetch; it returns the same cached `LoadedMeshData` object, and what
varies with the view is handled downstream by `projectMeshTo3D`. This is the one place
a reader might expect a re-fetch and find none.

## Not here yet

Label / image-label CSR arrays are **budgeted and pair-checked** by Stage 1 but not
fetched — picking arrives in a later phase. Counting them from the start means the
ceiling does not silently loosen when the label loader lands, and the pair check
means a `has_labels` with one array missing fails now, while the error can still
name the real problem. Stage 2's CSR offset-monotonicity check arrives with the
fetch.

Note what the presence-flag check does _not_ cover: the converse direction, an
array the flags disown, is unreachable through the loader (it opens an optional
array only when its flag is set), so it is deliberately not asserted rather than
being enforcement that can never fire.
