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
├── lod_backfill.py    finalize_lod_position_bounds(), finalize_lod_display_types()
└── validation.py      validate_discrete_dimension_ranges()
```

## API

### `hashing.compute_content_hashes(store) -> str`

Post-order xxhash64 over the whole zarr tree. For each group it hashes, in a
deterministic order: (1) its own arrays (`array_keys()` sorted, raw bytes),
(2) its attrs as sorted JSON — **excluding** any existing `content_hash` to
avoid self-reference, then (3) each child group's recursively-computed hash.
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

### `validation.validate_discrete_dimension_ranges(store, scene_bounds) -> None`

Emits `UserWarning`s when a discrete, non-displayed dimension's declared
`range` falls outside the actual data extent (read from `scene_bounds`). Catches
two common authoring mistakes: a range that starts before any data exists
(e.g. range starts at frame 0 but data starts at frame 1) and a range that
extends beyond the data, either of which lets the viewer initialise or navigate
to a slice with nothing in it. Comparisons use a half-step tolerance
(`dim.step / 2`, default `0.5`). No-ops when `scene_bounds` is `None` or the
store has no `scene_dimensions` attr.

## How the compiler wires these

`LuxarZarrCompiler.finalize()` calls each pass against the open store. The
display-type pass runs before the position-bounds pass (LOD-of-LOD constructions
need a resolved type before bounds aggregation), and `compute_content_hashes`
runs last so the stamped hashes cover the back-filled attrs.

```python
# packages/luxar/src/luxar/io/compiler.py (finalize-time)
from ._compiler.finalize.hashing import compute_content_hashes
from ._compiler.finalize.lod_backfill import (
    finalize_lod_display_types,
    finalize_lod_position_bounds,
)
from ._compiler.finalize.validation import validate_discrete_dimension_ranges
```

## Dependencies

**Internal:**
- `luxar.core.dimensions.Dimensions` — imported lazily in `validation.py` to
  rebuild dimensions from the store's `scene_dimensions` attr.

**External:**
- `zarr` — tree traversal and attribute storage
- `xxhash` — fast non-cryptographic content hashing
- `arbol` — structured `aprint` logging

## See Also

- [io/README.md](../../README.md) — I/O operations, writers, and the compiler
- [core/README.md](../../../core/README.md) — Scene graph, Dimensions, and `position_bounds`
