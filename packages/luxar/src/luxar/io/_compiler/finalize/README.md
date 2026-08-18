# luxar.io._compiler.finalize

Finalize-time tree passes for the Luxar zarr compiler. Each function in this
package walks the fully-written zarr scene tree once (post-order) and either
**back-fills** missing aggregate metadata, **validates** authored metadata
against the data that was actually written, or **stamps** a content hash. They
run during `LuxarZarrCompiler.finalize()` — after every node's arrays and
attrs are on disk, before the consolidated metadata is written.

These are deliberately thin, store-only helpers: the compiler exposes each one
as a private `_…` method that supplies the extra context (e.g. scene bounds)
and the bodies live here so `compiler.py` stays a thin orchestration layer.

## File Structure

```
finalize/
├── __init__.py        (empty — functions imported directly by module)
├── hashing.py         compute_content_hashes()
├── lod_backfill.py    finalize_lod_position_bounds(), finalize_lod_display_types(),
│                      warn_one_part_partition_anchors()
└── validation.py      validate_discrete_dimension_ranges()
```

## API

### `hashing.compute_content_hashes(store) -> str`

Post-order xxhash64 over the whole zarr tree. For each group it hashes, in a
deterministic order:

1. for every array (`array_keys()` sorted), its **storage identity** as sorted
   JSON, then its **decoded values** (`dataset[:].tobytes()`). The identity is
   `name`/`shape`/`chunks`/`dtype`/shard shape, the array's own `attrs`, and its
   codec **ids** (`codec_ids`, derived at either on-disk format) — plus the full
   codec pipeline (`codecs`) when the array is sharded. `_storage_identity` is
   where the reasoning lives: why layout and codec identity count as identity,
   why the per-array `encoding` attrs do, and why codec settings deliberately do
   not.
2. the group's attrs as sorted JSON, **excluding** any existing `content_hash` to
   avoid self-reference.
3. each child group's recursively-computed hash.

The resulting hex digest is written back into the group's `content_hash`
attribute, and the root digest is returned. xxhash64 is chosen for speed over
cryptographic strength.

### `lod_backfill.finalize_lod_position_bounds(store) -> None`

Back-fills a missing `position_bounds` on every `kind == "lod"` group by taking
the **union** of its children's bounds (recursing through nested
`kind="lod"` / `kind="partition"` wrappers and plain groups down to leaves
that carry their own bounds). The convenience-builder path
(`_add_gsplats_as_lod_group`) leaves the LOD parent without bounds because each
leaf carries its own; `kind="partition"` wrappers already persist their union
at write time, so only `kind="lod"` wrappers needed this aggregate. Without it
the viewer's `loadLodGroupNode` saw empty bounds for a nested LOD-of-LOD
construction and skipped that level in projection.

- **Never overwrites** an authored `position_bounds` — fills missing values only.
- Children with empty or mismatched-dimensionality bounds are skipped in the
  union, mirroring the viewer registry's defensive fallback.

### `lod_backfill.finalize_lod_display_types(store) -> None`

Back-fills a missing `display_type` on every `kind == "lod"` group by resolving
it from the **finest** child's own type (recursing through nested
`kind="lod"` / `kind="partition"` groups). LOD children are stored
coarsest→finest, so the finest is the last sorted child. The convenience
builder already sets `display_type` explicitly; this pass covers
explicit-builder constructions where the user called `add_lod_group(...)` with
a mix of leaf types and never set the parent's `display_type`.

- **Never overwrites** an authored `display_type` — fills missing values only.

### `lod_backfill.warn_one_part_partition_anchors(store) -> None`

The one pass here that **reports without writing anything**. A per-TILE
(fills-screen) LOD ladder — one whose `coverage_fraction` thresholds reach
`MAX_COVERAGE_FRACTION` = 4.0 — is correct only under a real tiling of **two or
more** parts, because a tile's projected bbox diagonal is intrinsically a
fraction of the whole object's. Under a **one-part** `kind=partition` that part's
bbox _is_ the whole object, so the ladder holds its finest level back until the
object overfills the viewport.

Every producer that can see the final sibling count already excludes that shape
(`gsplats/lod/recipes.py::build_adaptive` and both gsplat tree writers). The
scene adders cannot: `core/group/lod/group.py::derive_coverage_fractions` is
handed only the insertion point, and part 0's ladder is derived before part 1 has
been added. Finalize is the first moment the count exists.

- One `aprint` warning per offending `kind=lod` group. A ladder is only
  reported when a one-part partition encloses it **and no partition further up
  is a real tiling** — the same rule the writers thread down (`under_partition
  or len(children) > 1`), so neither `partition(2) → partition(1) → lod` nor a
  genuine multi-part partition nested inside a one-part wrapper is blamed. When
  it does fire, the offender named is the **nearest** enclosing partition.
- Never raises and never re-anchors: an authored
  `coverage_fractions=[0, …, 4.0]` list is indistinguishable on disk from a
  derived one, so a silent rewrite would override a deliberate choice.

### `validation.validate_discrete_dimension_ranges(store, scene_bounds) -> None`

Emits `UserWarning`s when a discrete, non-displayed dimension's declared
`range` falls outside the actual data extent (read from `scene_bounds`), or
when the data itself sits off the viewer's navigation grid. Catches three
authoring mistakes: a range that starts before any data exists (e.g. range
starts at frame 0 but data starts at frame 1), a range that extends beyond the
data — either of which lets the viewer initialise or navigate to a slice with
nothing in it — and discrete data more than a quarter-step off the `k·step`
grid (the viewer snaps navigation to that grid and its chunk query reaches
only a quarter-step around it, so off-grid data can silently never display).
Comparisons use a quarter-step tolerance (`dim.step / 4`, default `0.25`),
mirroring the viewer's `DISCRETE_TOLERANCE_FRACTION`. A dimension without a
declared step is checked against the integer grid (the viewer defaults a
missing step to `1.0`). The on-grid check inspects the data min/max only —
interior off-grid values on an otherwise on-grid extent are not scanned.
No-ops when `scene_bounds` is `None` or the store has no `scene_dimensions`
attr.

## How the compiler wires these

`LuxarZarrCompiler.finalize()` calls each pass against the open store. The
display-type pass runs before the position-bounds pass (LOD-of-LOD constructions
need a resolved type before bounds aggregation), the one-part-anchor warning runs
after both (it only reads), and `compute_content_hashes` runs last so the stamped
hashes cover the back-filled attrs.

```python
# packages/luxar/src/luxar/io/compiler.py (finalize-time)
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
    warn_one_part_partition_anchors,
)
from ._compiler.finalize.validation import validate_discrete_dimension_ranges
```

## Dependencies

**Internal:**
- `luxar.core.dimensions.Dimensions` — imported lazily in `validation.py` to
  rebuild dimensions from the store's `scene_dimensions` attr.
- `luxar.core.group.lod.group.MAX_COVERAGE_FRACTION` — the single definition of
  the fills-screen anchor, read by `warn_one_part_partition_anchors` so the
  warning cannot drift from what the producers derive.

**External:**
- `zarr` — tree traversal and attribute storage
- `xxhash` — fast non-cryptographic content hashing
- `arbol` — structured `aprint` logging

## See Also

- [io/README.md](../../README.md) — I/O operations, writers, and the compiler
- [core/README.md](../../../core/README.md) — Scene graph, Dimensions, and `position_bounds`
