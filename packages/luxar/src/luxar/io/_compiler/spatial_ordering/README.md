# spatial_ordering

Compiler glue that connects the `LuxarZarrCompiler` orchestrator to the
spatial-ordering primitives in `luxar.io.ordering`. Each geometry type gets a
matched **build** / **write** pair: `build_*_ordering()` reorders the data along
a space-filling curve and computes per-chunk bounds; `write_*_ordering_to_zarr()`
persists the resulting metadata and `chunk_bounds` arrays into the zarr group.

These functions hold no orchestrator state of their own — they receive a frozen
`OrderingCtx` (`enable_spatial_index`, `ordering_method`) and the root zarr
`store` (read for `scene_dimensions`), keeping the compiler decoupled from the
ordering math.

## Overview

Spatial ordering sorts elements so that nearby elements in space are nearby in
storage, then records the spatial extent of each storage chunk. The viewer uses
those `chunk_bounds` to skip chunks that fall outside the current slice/tolerance
query, so large datasets stream only the chunks they need.

The ordering itself (Morton / Hilbert curve construction, discrete "slice" vs.
continuous "ordering" dimension partitioning, chunk-bounds computation) lives in
`luxar.io.ordering`. This package is the thin adapter the compiler calls during a
write.

```
LuxarZarrCompiler (io/compiler.py)
        │  build_*_ordering(ctx, store, ...)
        ▼
spatial_ordering/  ── reads scene_dimensions, calls ──▶  luxar.io.ordering
        │  write_*_ordering_to_zarr(group, ordering_data, compressor)
        ▼
   zarr group (chunk_bounds array + ordering metadata in attrs)
```

## File Structure

```
spatial_ordering/
├── __init__.py   # empty (functions imported directly by the orchestrator)
├── points.py     # build_points_ordering, write_points_ordering_to_zarr
└── lines.py      # build_lines_ordering, write_lines_ordering_to_zarr
```

GSplat spatial ordering is not handled here; it lives alongside the gsplat
assembly path (`_compiler/gsplat_assembly.py`). Points and Lines share the same
`OrderingCtx` and the same `luxar.io.ordering` backend.

## API

### Points (`points.py`)

```python
build_points_ordering(
    positions: NDArray[np.float32],
    n_points: int,
    n_dims: int,
    radii: Optional[Union[NDArray[np.float32], float]],
    ctx: OrderingCtx,
    store: zarr.Group,
) -> Optional[Dict[str, Any]]
```

Applies `sort_points_compound()` from `luxar.io.ordering` along the configured
curve. Returns `None` when ordering is disabled (`not ctx.enable_spatial_index`),
when `n_points == 0`, or when the store has no `scene_dimensions` attribute.

The returned dict carries:

- `sorted_positions` — positions reordered by `sort_order`
- `sort_order` — index array to apply the same reordering to colors/radii/etc.
- `chunk_bounds` — `(num_chunks, n_dims, 2)` float32 min/max box per chunk
- `chunk_size` — elements per chunk, derived from `TARGET_CHUNK_BYTES`
- the flattened `ordering_metadata` (`ordering`, `slice_dims`, `ordering_dims`,
  `ordering_min`, `ordering_max`, `ordering_bits_per_dim`, ...)

Radii are normalised before chunk-bounds computation: broadcast arrays (shape
`(1,)` / `(1, k)`) and scalars collapse to a single float to avoid materialising
a full-length array; per-element radii are reordered by `sort_order`.

```python
write_points_ordering_to_zarr(group, ordering_data, compressor) -> None
```

Writes the ordering metadata into `group.attrs` and creates a `chunk_bounds`
dataset (chunked as a single chunk per axis, scene compressor applied). The
array is only created when there is at least one chunk.

### Lines (`lines.py`)

Lines carry **two** index spaces — vertices (D-dimensional positions) and
segments (index pairs) — so ordering is *dual*: both are sorted and each gets its
own chunk bounds.

```python
build_lines_ordering(
    vertices: NDArray[np.float32],
    segments: NDArray[np.uint32],
    widths: Union[NDArray[np.float32], float],
    n_vertices: int,
    n_dims: int,
    n_segments: int,
    ctx: OrderingCtx,
    store: zarr.Group,
) -> Optional[Dict[str, Any]]
```

Delegates the dual sort to `order_lines_spatial()` and the bounds to
`compute_vertex_chunk_bounds()` / `compute_segment_chunk_bounds()` (all from
`luxar.io.ordering`). Returns `None` under the same disabled / empty / missing-
dimensions conditions as points.

The returned dict carries `sorted_vertices`, `sorted_segments`,
`vertex_sort_indices`, `segment_sort_indices`, `vertex_chunk_bounds`,
`segment_chunk_bounds`, `ordering`, plus the nested `ordering_metadata`
(`vertex_ordering` / `segment_ordering`, each with its own `slice_dims` and a
`chunk_size` injected here). Vertex and segment chunk sizes are computed
separately from `TARGET_CHUNK_BYTES`. Segment bounds use the vertex `slice_dims`
(D-space) and require expanded per-vertex widths, so scalar/broadcast widths are
materialised to a full array and per-element widths are reordered by
`vertex_sort_indices`.

```python
write_lines_ordering_to_zarr(group, ordering_data, compressor) -> None
```

Creates two datasets — `vertex_chunk_bounds` and `segment_chunk_bounds` — each
`(num_chunks, n_dims, 2)` float32, written with the scene compressor and skipped
when empty.

## Invariants

- **Reordering is index-based.** `build_*` returns sort indices; the orchestrator
  must apply the *same* indices to every parallel attribute array (colors,
  radii, widths, sharpness) so they stay aligned with `sorted_positions` /
  `sorted_vertices`.
- **Graceful skip.** Missing `scene_dimensions`, an empty dataset, or a disabled
  `OrderingCtx` all return `None` (with a warning for the missing-dimensions
  case) rather than raising — the compiler then writes data in input order.
- **`chunk_bounds` shape is `(num_chunks, n_dims, 2)`** with `[..., 0]` = min and
  `[..., 1]` = max per dimension. The viewer's chunk-bounds loader reads this
  layout to cull chunks against the slice query.
- **Chunk size targets `TARGET_CHUNK_BYTES`** (see
  `typing_utils/constants.py`), floored at 1024 elements and capped at the
  element count.

## See Also

- [io/ordering.py](../../ordering.py) — the Morton/Hilbert sort and chunk-bounds
  primitives these glue functions call (`sort_points_compound`,
  `order_lines_spatial`, `compute_*_chunk_bounds`)
- [context.py](../context.py) — `OrderingCtx` definition
- [core/README.md](../../../core/README.md) — `Dimensions` and the
  displayed/non-displayed split that drives slice vs. ordering dimensions
