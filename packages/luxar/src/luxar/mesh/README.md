# `luxar.mesh`

Mesh-specific tooling that sits *beside* the `Mesh` node type rather than inside it.

The scene-graph object lives in `luxar.core.mesh`; everything else about meshes that is
not that object lives here. This mirrors `luxar.gsplats`, which is likewise separate
from `luxar.core.gsplats`.

## Contents

| Module | Purpose |
|---|---|
| [`decimate.py`](decimate.py) | Produce a genuinely coarser SURFACE — the producer a substitutive LOD ladder needs. `cluster` snaps vertices to a grid, collapses each occupied cell to its centroid, reindexes the faces and drops the triangles that collapsed. Vectorized NumPy, no new dependencies. |
| [`interop/`](interop/README.md) | Import classical mesh files — PLY, OBJ, STL, glTF/GLB — into a `TriangleMesh`, the intermediate the CLI writes into a scene. NumPy + stdlib only, no new dependencies. |

## Why this is not `luxar.core.mesh`

`luxar.core` is the scene graph. A format decoder that produces plain NumPy arrays and
touches no zarr does not belong there, and putting it in `luxar.io` — the
compiler/reader layer — would muddy that layer's direction of travel.

`luxar/mesh/__init__.py` deliberately does **not** re-export `Mesh`. Pulling the node
class up here would make this package import `luxar.core`, an edge that buys nothing
and risks a cycle, since `luxar.core` already reaches most of the package. The two
names are distinct fully-qualified modules and stay that way.

## Quick Start

```python
from luxar.mesh import import_mesh

mesh = import_mesh("bunny.ply")  # -> TriangleMesh (welded, triangulated)
scene.add_mesh(
    "Bunny",
    mesh.vertices,
    mesh.faces,
    normals=mesh.normals,
    normal_dims=[0, 1, 2],
    colors=mesh.colors,
)
```

```bash
luxar mesh import bunny.ply bunny.luxar.zarr
```

## Not here

- **The `Mesh` node class, the writer, and the reader** — `luxar.core.mesh`,
  `luxar.io._compiler.geometry_writers.mesh`, `luxar.io.reader`.
- **An *additive* (prefix) LOD ladder.** That flavour cannot apply to a surface at all —
  a prefix of an index buffer is a surface with holes in it, not a coarser one — so it is
  refused on principle. *Substitutive* levels, which were missing only a producer, now
  work: `decimate.py` above is that producer (vertex clustering; a Garland-Heckbert
  `qem` tier is the one this is shaped to admit next, issue #1348), and
  `add_mesh(substitutive_lod=…)` writes the resulting `kind=lod` group. See
  `docs/specs/MESH_NODE_SPEC.md` §9.
- **Export.** The inverse direction (`.luxar.zarr` → PLY/OBJ/STL) has no consumer yet;
  `gsplats/interop/inria_export.py` is the shape it would take.
