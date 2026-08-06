# `luxar.mesh`

Mesh-specific tooling that sits *beside* the `Mesh` node type rather than inside it.

The scene-graph object lives in `luxar.core.mesh`; everything else about meshes that is
not that object lives here. This mirrors `luxar.gsplats`, which is likewise separate
from `luxar.core.gsplats`.

## Contents

| Module | Purpose |
|---|---|
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

mesh = import_mesh("bunny.ply")   # -> TriangleMesh (welded, triangulated)
scene.add_mesh("Bunny", mesh.vertices, mesh.faces,
               normals=mesh.normals, normal_dims=[0, 1, 2],
               colors=mesh.colors)
```

```bash
luxar mesh import bunny.ply bunny.luxar.zarr
```

## Not here

- **The `Mesh` node class, the writer, and the reader** — `luxar.core.mesh`,
  `luxar.io._compiler.geometry_writers.mesh`, `luxar.io.reader`.
- **Decimation / LOD.** Mesh has no LOD ladder yet: the additive prefix flavour cannot
  apply to a surface at all, and substitutive levels are missing only a producer (QEM).
  See `docs/specs/MESH_NODE_SPEC.md` §9. When that producer lands, `luxar/mesh/simplify/`
  is where it belongs.
- **Export.** The inverse direction (`.luxar.zarr` → PLY/OBJ/STL) has no consumer yet;
  `gsplats/interop/inria_export.py` is the shape it would take.
