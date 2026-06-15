# luxar.io._compiler

Private implementation of the Luxar zarr compiler. The public orchestrator —
`LuxarZarrCompiler` in `luxar/io/compiler.py` — owns the session state (open
zarr store, encoder, compressor, scene bounds, warn flags) and sequences the
calls; the bodies of those steps live here, split into focused, stateless
helpers that read only the orchestrator state they actually need.

There is no public API in this package: it is imported only by
`compiler.py`. The `__init__.py` is intentionally empty (helpers are imported
by submodule path, e.g. `from ._compiler.bounds import expand_bounds_with_transforms`).

## Why it is split out

Every helper here is a free function (no back-pointer to the orchestrator
instance). When a helper needs configuration it takes a narrow, frozen `Ctx`
dataclass (see `context.py`) carrying only its read-set. State that the
orchestrator must keep across calls — most notably the one-shot LUT
tone-mapping warning flag — is **threaded by value**: the helper takes the
current flag in and returns the (possibly updated) flag for the caller to
store back, rather than mutating the orchestrator directly.

## Layout

```
_compiler/
├── __init__.py            # empty (helpers imported by submodule path)
├── context.py             # DatasetCtx / OrderingCtx — narrow config dataclasses
├── chunking.py            # calculate_intelligent_chunks — shared chunk-shape heuristic
├── colormap.py            # custom-colormap LUT resolution + zarr write
├── bounds.py              # scene-bounds compute / union / world-space expansion
├── gsplat_assembly.py     # gsplat validate → order → write arrays → stamp attrs
├── dataset_writers/       # per-attribute zarr array serializers (positions, colors, scalars)
├── spatial_ordering/      # points/lines space-filling-curve ordering glue
├── labels/                # CSR serialization of string + image labels
└── finalize/              # post-write tree passes (back-fill, validate, hash)
```

## Loose helper modules

### `context.py`
Two frozen dataclasses passed into the helper families instead of the
orchestrator instance:

| Ctx | Carries | Used by |
|-----|---------|---------|
| `DatasetCtx` | `encoder`, `encoding_mode`, `compressor` | every `dataset_writers/` serializer + the gsplat array writer |
| `OrderingCtx` | `enable_spatial_index`, `ordering_method` (`"morton"`/`"hilbert"`) | gsplat ordering + the points/lines ordering glue |

Built by `_make_*_ctx()` methods on the orchestrator.

### `chunking.py`
`calculate_intelligent_chunks(shape, target_chunk_bytes=TARGET_CHUNK_BYTES,
spatial_index_data=None, *, dtype=float32)` — the single chunk-shape heuristic
shared by every dataset serializer. When spatial-ordering data is present it
aligns chunks to the ordering's `chunk_size`; otherwise it picks a chunk that
hits the byte target given the array's `dtype.itemsize`. The `dtype=`
keyword is mandatory in spirit (defaults to float32 only for compatibility) so
non-float32 callers — e.g. a `(N, 3)` uint8 colors array — are chunked
correctly rather than silently under-chunked.

### `colormap.py`
`write_colormap_lut_if_needed(group, attrs, scene_tone_mapping,
lut_tone_mapping_warned) -> bool` — if `attrs["colormap"]` is a numpy array or
a non-built-in name (matplotlib/colorcet), resolves it to a `(256, 3)` uint8
LUT, writes a `colormap_lut` dataset, and rewrites the attr value to
`"custom"`. Built-in names are left untouched (the viewer resolves them
directly). Emits an at-most-once `UserWarning` advising authors to pin
`tone_mapping="Neutral"` when the scene uses a LUT but the viewer's default
ACES tone-mapping would shift hues — skipped for the implicit `"gray"` default
and when the author already chose `"Neutral"`. The warn flag is threaded by
value.

### `bounds.py`
Scene-level position-bounds machinery:

- `compute_position_bounds(positions)` — per-node nD axis-aligned box from an
  `(N, D)` array (`{"min": [...], "max": [...]}`); empty arrays yield zeros.
- `update_scene_bounds(scene_bounds, node_bounds)` — running union across nodes,
  extending dimensionality when a later node has more dims.
- `expand_bounds_with_transforms(store, scene_bounds)` — finalize-time pass that
  walks the written zarr tree, composes **two** transform families down the
  hierarchy, and overwrites the scene's `position_bounds` with the world-space
  union. The 4x4 spatial `transform` (translate/rotate/scale) acts on the
  displayed dims and is applied by transforming all 8 box corners (correct under
  rotation); the per-dimension `nd_transform` (affine scale/offset) acts on the
  non-displayed slider dims. Getting the 4x4 into the scene bounds is
  load-bearing: the viewer derives per-frame near/far clipping from a bounding
  sphere built off this metadata, so a node translated far from the origin would
  otherwise be clipped as the camera rotates.

### `gsplat_assembly.py`
The gsplat-specific pipeline sequenced by the shared walker
`gsplat_tree.write_gsplat_node` (called from both the standalone
`write_gsplats_tree` and the scene compiler's `write_gsplat_leaf_subtree`):

| Function | Step |
|----------|------|
| `validate_gsplat_inputs(...)` | shape/sign checks; normalizes a 1-D Cholesky (uniform covariance) to `(1, K)` and flags `cholesky_is_uniform`; returns `n_splats`, `n_dims` (`K = D(D+1)/2`) |
| `apply_gsplat_spatial_ordering(..., ctx: OrderingCtx)` | reorders centers/amplitudes/cholesky/colors along a space-filling curve, computes per-chunk bounds, returns `ordering_data` (or `None`) |
| `write_gsplat_arrays(..., ctx: DatasetCtx)` | encodes `centers` (COORDINATE), `amplitudes` (via the shared positive-scalar writer), `cholesky_factors` (CHOLESKY), optional `colors` (shared color writer), and `chunk_bounds`; returns a metadata dict |
| `apply_gsplat_group_attrs(...)` | resolves the colormap LUT, prepares/validates `transform` + `nd_transform`, fills rendering defaults (`opacity`, `gamma`, `intensity`, `offset`, `blending_mode`, `truncation_radius`), then stamps authoritative `type="gsplats"` attrs and `position_bounds` |

`amplitudes` and `cholesky_factors` go through `dataset_writers/scalars.py` and
the gsplat-specific paths so default-precision selection stays symmetric with
Points `radii` and Lines `widths`.

## Subpackages

- **`dataset_writers/`** — Per-attribute zarr array serializers (`write_positions`,
  `write_colors`, `write_positive_scalar`, ...): the canonical writers shared
  across all three geometry types, each routing one attribute through the shared
  `ArrayEncoder`.
- **`spatial_ordering/`** — Compiler glue to the space-filling-curve primitives in
  `luxar.io.ordering`. Each geometry gets a matched `build_*_ordering()` /
  `write_*_ordering_to_zarr()` pair for Points and Lines.
- **`labels/`** — Serialization of per-element string and image labels into the
  archive using CSR-style storage.
- **`finalize/`** — Finalize-time tree passes that walk the fully-written scene
  tree once (post-order) to back-fill aggregate metadata, validate authored
  metadata against what was written, and stamp content hashes.

## See Also

- `luxar/io/compiler.py` — the `LuxarZarrCompiler` orchestrator that imports and
  sequences everything here.
- `luxar/io/ordering.py` — the space-filling-curve and chunk-bounds primitives
  the gsplat assembly and `spatial_ordering/` glue call into.
- `docs/guides/user/LUXAR_ZARR_FORMAT.md` — the on-disk format these helpers write.
