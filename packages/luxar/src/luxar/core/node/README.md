# luxar.core.node

The base class of Luxar's scene graph: `Node`. Every node in a scene — the
root `Scene`, intermediate `Group`s, and the data-bearing `Points` / `Lines` /
`GSplats` — ultimately derives from `Node`. A node is a lightweight metadata
container: it holds its name, parent/child links, and a cache of attributes,
and it writes those attributes straight through a writer interface rather than
keeping any Zarr group in memory.

`Node` is re-exported at the subpackage root, so both of these resolve to the
same class:

```python
from luxar import Node
from luxar.core.node import Node
```

## File structure

```
node/
├── __init__.py            # Re-exports Node
├── node.py                # The Node class (hierarchy, attrs, transforms, rendering)
└── specialized_groups.py  # add_lod_group / add_partition_group bodies
```

## Node

A node is constructed with a `name`, an optional `parent`, an optional `writer`
(`ZarrWriterProtocol`), and arbitrary keyword `attrs`. On construction it:

- rejects names containing `/` (use `add_group()` to nest instead),
- links itself into `parent.children`, rejecting duplicate sibling names,
- computes its `path` (`parent_path/name`, or `""` for a root),
- validates any recognized attributes (`transform`, `nd_transform`, `opacity`,
  `absorption`, `gamma`, `intensity`, `offset`, `blending_mode`, `layer`,
  `visible`, `colormap`) and, if a writer is present, writes the group attrs immediately
  and caches them.

When no writer is supplied the node runs in **metadata-only mode**: attributes
are cached but nothing is written to disk.

### Hierarchy and traversal

| Member | Description |
|--------|-------------|
| `name`, `parent`, `children`, `path` | Scene-graph wiring |
| `add_group(name, **attrs)` | Create a child `Group` (carries `type="group"`) |
| `add_lod_group(name, *, selector="coverage", default_level=0, **attrs)` | Create a child `kind="lod"` `Group` |
| `add_partition_group(name, *, display_type, max_elements, **attrs)` | Create a child `kind="partition"` `Group` |
| `walk(depth=0)` | Depth-first generator yielding `(depth, node)` tuples |
| `num_children`, `is_leaf`, `is_root` | Convenience predicates |

`add_lod_group` and `add_partition_group` delegate their bodies to
`specialized_groups.py` (see below); both are thin validating wrappers over
`add_group`.

### Attributes and persistence

The `attrs` property exposes the cached attribute dict. The private
`_persist_attr(key, value)` helper updates both the cache and the on-disk Zarr
store via the writer. If the writer has already been finalized (after the
`LuxarZarrCompiler` context exits, or after `Scene.to_zarr`), the on-disk
attribute can no longer be updated through it — the in-memory cache is still
updated, but a `UserWarning` is raised so the disk/memory drift is not silent.
Set attributes inside the compiler context (or before `to_zarr`) for
persistence.

### Transforms

| Property | Description |
|----------|-------------|
| `transform` | This node's local 4x4 matrix. Getter reads back via `read_transform_from_zarr`; setter runs `prepare_transform_for_zarr` and persists immediately. Set to `None` to delete. |
| `world_transform` | Composes local transforms up the parent chain root-outermost (this node's transform applied first/innermost, root last/outermost — `world = root @ ... @ leaf`). Returns identity if none are set. |

Matrices are stored in THREE.js-compatible (column-major) form — see the
matrix-storage gotcha in the project `CLAUDE.md`.

### nD transforms

Separate from the spatial `transform`, a node may carry an `nd_transform` dict
that applies per-dimension affine (`{"scale": float, "offset": float}`) or
permutation (`{"permutation": [int, ...]}`) transforms on **non-displayed**
dimensions (e.g. time, channel).

| Property | Description |
|----------|-------------|
| `nd_transform` | Local per-dimension transform dict. Setter validates via `validation.nd_transforms.validate_nd_transform` and persists; `None` deletes. |
| `world_nd_transform` | Composes nd_transforms up the parent chain via `compose_nd_transforms` (root outermost). Empty dict means identity. |

See `docs/guides/specs/ND_TRANSFORMS_SPEC.md` for the full specification.

### Rendering attributes

All of these are validated on assignment and persisted immediately. Each has a
matching `set_*` method that returns `self` for chaining.

| Property | Default | Range / values |
|----------|---------|----------------|
| `opacity` | `1.0` | 0.0–1.0 |
| `absorption` | `1.0` | ≥ 0.0 (volumetric mode's κ; multiplicative) |
| `gamma` | `1.0` | 0.1–10.0 |
| `intensity` | `1.0` | 0.0–100.0 |
| `offset` | `0.0` | -10.0–10.0 |
| `blending_mode` | `"additive"` | `normal`, `additive`, `max`, `opaque`, `luminous`, `volumetric` |
| `colormap` | `None` | colormap name (string only via setter) |
| `layer` | `False` | whether the node appears in the viewer's Layers panel |
| `visible` | `True` | initial visibility when the scene loads |

```python
group.set_opacity(0.5).set_gamma(1.0).set_intensity(2.0).set_blending_mode("additive")
```

The `colormap` setter accepts only string names; custom LUT arrays must be set
at node-creation time (e.g. `add_points(..., colormap=array)`).

### Identity

`__eq__` / `__hash__` key on `(path, root identity)`, so two nodes with the same
path in different scenes are not equal and hash differently. Root nodes (empty
path) fall back to object identity.

## specialized_groups

Free-function bodies for the two specialized-`Group` builders on `Node`,
extracted to keep `node.py` readable.

- **`add_lod_group_impl`** — validates `selector` (only `"coverage"` is
  currently supported) and `default_level >= 0`, then creates a child group
  with `kind="lod"`. A `kind=lod` group picks one of N alternative children at
  runtime by projecting the group's bbox diagonal to screen pixels and
  comparing against each child's `coverage_fraction` threshold (a
  dimensionless, viewport-relative value in `[0, 1]`, multiplied by the
  current viewport diagonal to get the pixel comparison).
- **`add_partition_group_impl`** — validates `display_type` ∈
  {`points`, `lines`, `gsplats`} and `max_elements` (int ≥ 1), then creates a
  child group with `kind="partition"`. A `kind=partition` group is a
  compile-time decomposition of one large geometry node into homogeneous
  children for per-child frustum culling and LOD; the user still sees one
  logical layer.

```python
lod = scene.add_lod_group("multires")
lod.add_gsplats_from_data("c", coarse, coverage_fraction=0.0)
lod.add_gsplats_from_data("m", medium, coverage_fraction=0.5)
lod.add_gsplats_from_data("f", fine,   coverage_fraction=1.0)
```

## See Also

- [core/README.md](../README.md) — the scene graph (`Scene`, `Group`, data nodes, transforms, dimensions)
- [validation/README.md](../../validation/README.md) — attribute and nd_transform validation
- `docs/guides/specs/ND_TRANSFORMS_SPEC.md` — nD transform specification
