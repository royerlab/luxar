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
- **Screen-space overlays** — `add_text()`, `add_image()`, `add_html()` for HTML
  annotations anchored to the viewport (delegated to the `overlays/` subpackage).
- **Export** — `to_zarr()` finalizes the progressive writer and atomically
  copies the backing store to a destination.

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
└── overlays/        # screen-space text/image/HTML overlay implementations
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

A default hover tooltip overlay is auto-injected at compiler finalization when
nodes carry labels but no hover overlay was defined (`_auto_inject_hover_overlay`,
driven by `_notify_labels_added` / `_notify_image_labels_added`).

### Export

```python
scene.to_zarr("export.luxar.zarr")
```

`to_zarr(path)` finalizes the backing writer and atomically copies the on-disk
Zarr store to `path`. Because finalization closes the writer, **do not add more
nodes after calling `to_zarr()`** — create a new `LuxarZarrCompiler` for further
writes. Passing the current backing-store path is an explicit finalize-in-place;
otherwise the destination must not already exist and must not live inside the
source store (raises `FileExistsError` / `ValueError`).

## Validation helpers (`validation.py`)

Three free functions invoked by `Group`'s leaf adders through `Scene`'s stubs:

- `resolve_extend_to_all(scene, extend_to_all, positions, data_type)` —
  interprets the `extend_to_all=` kwarg. Accepts `None` (no extension, but warns
  when single-value candidate dimensions are detected), `"all"` (every
  non-displayed dimension), an explicit list of dimension names (unknown names
  raise `ValueError`), or `[]` (explicit no-extension that silences the warning).
- `analyze_extend_candidates(scene, positions)` — flags non-displayed dimensions
  that have exactly one unique value in the data yet declare a wider range —
  likely candidates for `extend_to_all`.
- `validate_data_dimensions(scene, positions, node_name, data_type)` — hard
  `ValueError` on a column-count vs scene-dimension mismatch; a `UserWarning`
  when values fall outside a dimension's declared `range`.

## dim_order remapping (`dim_order.py`)

`apply_dim_order(scene, positions, dim_order, fill=None)` reorders and pads
lower-dimensional data to match the scene's full dimension set. `dim_order` maps
each data column to a scene dimension by name; its length must equal the number
of data columns and names must be unique and exist in the scene. Dimensions not
named in `dim_order` are padded with `fill[name]` (default `0.0`) and returned in
the `unmapped` list — the caller uses that list to infer `extend_to_all`.
`fill` keys must be valid scene dimensions and must not also appear in
`dim_order`.

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
  `Scene.add_text()` / `add_image()` / `add_html()`, plus naming, zarr
  persistence, and label-driven auto-injection of a default hover overlay.

## See Also

- [core/README.md](../README.md) — full scene-graph overview (Node, Group, DataNode, Dimensions, Transforms, ViewerConfig)
- [group/README.md](../group/README.md) — the `Group` base class that supplies the `add_*` data methods
- [dimensions.py](../dimensions.py) — `Dimension` / `Dimensions` coordinate system
- [overlay.py](../overlay.py) — the `Overlay` metadata dataclass returned by the overlay adders
- `docs/guides/specs/ND_TRANSFORMS_SPEC.md` — nD transform specification
