# luxar.core.scene

The scene-graph root node. `Scene` is the top of every Luxar hierarchy: it owns
the coordinate system (`Dimensions`), the progressive writer, screen-space
overlays, and the export path to a finalized Zarr store.

## Overview

`Scene` subclasses [`Group`](../group/README.md), so all data-adding methods
(`add_points`, `add_lines`, `add_gsplats`, `add_group`, …) are inherited and
work identically on the root. What the `Scene` class adds on top of `Group` is:

- **Dimensions ownership** — scene dimensions are REQUIRED at construction and
  serve as the single source of truth for the coordinate system. `Scene` is its
  own root (`_find_scene()` returns `self`), so all child nodes resolve their
  dimensions and writer through it.
- **Viewer config** — optional `ViewerConfig` hints persisted into the zarr file
  as scene-specific viewer defaults.
- **Screen-space overlays** — `add_text()`, `add_image()`, `add_video()`, and
  `add_html()` for annotations anchored to the viewport (delegated to the
  `overlays/` subpackage).
- **Export** — `to_zarr()` finalizes the progressive writer, copying a
  directory store or publishing an archive at its selected output path.

`Scene` is created through `LuxarZarrCompiler`, never instantiated directly.

```python
from luxar import LuxarZarrCompiler, Dimensions, Dimension
import numpy as np

dims = Dimensions([
    Dimension("X", display=True),
    Dimension("Y", display=True),
    Dimension("Z", display=True),
])

with LuxarZarrCompiler("output.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dims)
    scene.add_points("points", np.random.randn(10000, 3).astype(np.float32))

    # Group methods are inherited; nested writes work the same way
    group = scene.add_group("markers")
    group.add_points("nested", np.random.randn(100, 3).astype(np.float32))
```

## File Structure

```
scene/
├── __init__.py      # re-exports Scene (from luxar.core.scene import Scene)
├── scene.py         # Scene class: construction, properties, overlays, export
├── validation.py    # extend_to_all + data-dimension validation (free functions)
├── dim_order.py     # dim_order column remapping (free function)
└── overlays/        # screen-space text/image/video/HTML overlay implementations
```

The `Scene` class keeps thin method stubs (`_resolve_extend_to_all`,
`_validate_data_dimensions`, `_apply_dim_order`, `_write_overlay`, …) that
forward to the free functions in the sibling modules and the `overlays/`
subpackage. This keeps `scene.py` focused on construction, properties, and
export while the heavier logic lives in dedicated, individually testable units.

## Scene API

### Construction

`Scene(writer, dimensions, viewer_config=None)` — both `writer` and
`dimensions` are required (a `ValueError` is raised if either is `None`).
Construction writes `scene_dimensions` (and optionally `viewer_config`) into the
root zarr group immediately. Reassigning `scene.dimensions` inside the active
compiler context updates both the live scene and the stored root metadata; as with
other node property setters, assignment after writer finalization updates only the
live object and emits a warning. Same-dimensionality changes (names, units, ranges,
displayed axes) are allowed, but changing the dimension count (`ndim`) after any
geometry has been added is rejected with a `ValueError` — existing data arrays would
no longer match the scene's coordinate system.

### Properties

| Property | Description |
|----------|-------------|
| `dimensions` | Get/set the scene `Dimensions`. Setter rejects `None` (and an `ndim` change once geometry exists) and persists changes to the root zarr attributes while the writer is active. |
| `viewer_config` | Get/set `ViewerConfig` hints; lazily read back from zarr attrs. |
| `overlays` | List of `Overlay` objects added to the scene. |
| `get_store_path()` | Path to the backing Zarr store (requires a writer). |

### Overlays

Screen-space HTML annotations positioned in normalized viewport coordinates
`(x, y) ∈ [0, 1]` with a configurable `anchor`. The `Scene` methods are thin
delegates; the implementations live in [`overlays/`](overlays/README.md).

- `add_text(text, position, ...)` — text rendered as an HTML element, with
  optional wrapping (`width`), background, stroke, blend mode, and hover
  templating (`{hover_label}`, `{hover_node}`, `{hover_index}`).
- `add_image(image, position, ...)` — image (path, bytes, numpy array, or PIL
  image) stored in the zarr directory and rendered as an `<img>`.
- `add_html(html, position, ...)` — sanitized HTML (script tags and event
  handlers stripped) rendered over the canvas.

All three accept `visible_range` (dimension-based visibility filter),
`transition` / `transition_duration`, `interactive`, and `blend_mode`.

The top-level scene-node name `overlays` is reserved for this internal storage
namespace. Use another name for user-created groups or geometry nodes; nested
nodes named `overlays` remain valid because they do not collide with the root.

A default hover tooltip overlay is auto-injected at compiler finalization when
nodes carry labels but no hover overlay was defined (`_auto_inject_hover_overlay`,
driven by `_notify_labels_added` / `_notify_image_labels_added`).

### Export

```python
scene.to_zarr("export.luxar.zarr")
```

`to_zarr(path)` finalizes the backing writer. For a directory-backed scene it
atomically copies the on-disk Zarr store to a destination that must not already
exist unless it is the current backing store, which is an explicit
finalize-in-place; the destination also cannot live inside the source store.
For an archive-backed scene, `path` must be the selected archive path;
finalization publishes there and replaces an existing archive. A
directory-backed scene cannot be copied directly to a `.zip` destination —
create it with `LuxarZarrCompiler` or use `luxar optimise` instead. Because
finalization closes the writer, **do not add more nodes after calling
`to_zarr()`**; create a new compiler for further writes.

## Validation helpers (`validation.py`)

Five free functions invoked by `Group`'s leaf adders through `Scene`'s stubs:

- `resolve_extend_to_all(scene, extend_to_all, positions, data_type)` —
  interprets the `extend_to_all=` kwarg. Accepts `None` (no extension, but warns
  when single-value candidate dimensions are detected), `"all"` (every
  non-displayed dimension), an explicit list of dimension names (unknown names
  raise `ValueError`), or `[]` (explicit no-extension that silences the warning).
- `analyze_extend_candidates(scene, positions)` — flags non-displayed dimensions
  that have exactly one unique value in the data yet declare a wider range —
  likely candidates for `extend_to_all`.
- `validate_array_rank(positions, data_type)` — the 2-D `(N, D)` shape raise on
  its own, so a split path can rank-check without also count-checking (under a
  `dim_order` the incoming width is legitimately not the scene's).
- `validate_dimension_count(scene, positions, node_name, data_type)` — the hard
  `ValueError` on a column-count vs scene-dimension mismatch, on its own. It also
  rejects any non-2-D array, wording that like the rank message of the three
  adders that can reach it (`… must have shape (N, D)`; mesh says `(V, D)` but
  rank-checks before calling here). The in-memory split paths call it on the
  caller's SOURCE array before they create a wrapper group, so a mismatch is
  refused against the caller's own node with nothing written (#1446), and without
  re-firing the range warning below once per part or level: `partition=`,
  `additive_lod=` and `substitutive_lod=` on `add_points`/`add_lines`/
  `add_gsplats` check it at the top of the adder, above every structural branch;
  `add_gsplats_from_data`'s `lod_group=` checks it just inside the
  multi-substitutive branch instead, below the `coverage_fraction` refusal so that
  kwarg fault keeps precedence, and still above `add_lod_group`. That gate also
  runs the `dim_order`/`fill`/`fill_sigma` spec checks and the colours/colormap
  exclusion, in the flat path's own statement order (rank, then the `dim_order`
  spec — which the adder runs while applying the transform — then colours, then
  the width), so no in-memory fault it can see leaves a childless wrapper behind
  and a multi-fault call is told about the same one on both paths. `add_gsplats_from_file` is covered on
  both of its doors: a matrix-shaped store (leaf / additive ladder / `kind=lod` of
  leaves) is dispatched down `add_gsplats_from_data` and inherits that gate, while a
  genuinely nested one — a `kind=partition` root, or a lod group with non-leaf
  children — is checked against the STORED tree at the `graft_gsplat_node` entry,
  since the graft builds the whole wrapper chain from the on-disk tree before the
  first leaf is added. One leaf answers for the subtree there: a graft applies no
  `dim_order` (it refuses the kwarg), and both container node types reject
  mixed-`ndim` children at construction.
- `validate_data_dimensions(scene, positions, node_name, data_type)` — the full
  check the flat write runs: `validate_dimension_count` first, then a
  `UserWarning` per dimension whose values fall outside its declared `range`.

## dim_order remapping (`dim_order.py`)

`apply_dim_order(scene, positions, dim_order, fill=None)` reorders and pads
lower-dimensional data to match the scene's full dimension set. `dim_order` maps
each data column to a scene dimension by name; its length must equal the number
of data columns and names must be unique and exist in the scene. Dimensions not
named in `dim_order` are padded with `fill[name]` (default `0.0`) and returned in
the `unmapped` list — the caller uses that list to infer `extend_to_all`.
`fill` keys must be valid scene dimensions and must not also appear in
`dim_order`.

`validate_dim_order_spec(scene, dim_order, data_ndim, fill=None)` is that whole
validation preamble on its own — length, duplicate names, names in the scene,
`fill` keys, in that order. `apply_dim_order` calls it first, so the messages are
unchanged; it exists separately because none of those refusals need the data,
which lets a split path run them before it creates a wrapper group (#1446). Its
gsplats-only companion for `fill_sigma` is `validate_fill_sigma_keys` in
`core/group/dim_order.py`.

```python
# Map 3D data columns into a 4D scene, holding Time at 0.0
remapped, unmapped = apply_dim_order(
    scene, positions_3d,
    dim_order=["Z", "Y", "X"],
    fill={"Time": 0.0},
)
# unmapped == ["Time"]  → candidate for extend_to_all
```

## Subpackages

- [`overlays/`](overlays/README.md) — pure-function implementations behind
  `Scene.add_text()` / `add_image()` / `add_video()` / `add_html()`, plus naming, zarr
  persistence, and label-driven auto-injection of a default hover overlay.

## See Also

- [core/README.md](../README.md) — full scene-graph overview (Node, Group, DataNode, Dimensions, Transforms, ViewerConfig)
- [group/README.md](../group/README.md) — the `Group` base class that supplies the `add_*` data methods
- [dimensions.py](../dimensions.py) — `Dimension` / `Dimensions` coordinate system
- [overlay.py](../overlay.py) — the `Overlay` metadata dataclass returned by the overlay adders
- `docs/guides/specs/ND_TRANSFORMS_SPEC.md` — nD transform specification
