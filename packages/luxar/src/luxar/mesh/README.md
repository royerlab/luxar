# `luxar.mesh`

Mesh-specific tooling that sits *beside* the `Mesh` node type rather than inside it.

The scene-graph object lives in `luxar.core.mesh`; everything else about meshes that is
not that object lives here. This mirrors `luxar.gsplats`, which is likewise separate
from `luxar.core.gsplats`.

## Contents

| Module | Purpose |
|---|---|
| [`decimate.py`](decimate.py) | Dispatch mesh decimation and provide the vectorized `cluster` tier, which snaps vertices to a grid, collapses each occupied cell to its centroid, reindexes faces, and drops collapsed triangles. |
| [`qem.py`](qem.py) | Topology-preserving Garland-Heckbert edge-collapse decimation with a link-condition veto. |
| [`interop/`](interop/README.md) | Import classical mesh files — PLY, OBJ, STL, VTK XML PolyData (`.vtp`), glTF/GLB — or indexed file directories into a `TriangleMesh`, the intermediate the CLI writes into a scene. NumPy + stdlib only, no new dependencies. |
| `split.py` | Split a mesh into spatially disjoint, independently drawable parts by face. The bookkeeping behind `add_mesh(partition=…)` and behind the per-level re-indexing of `add_mesh(additive_lod=…)`. |
| [`primitives.py`](primitives.py) | Analytic primitives — today the welded, closed `icosphere()` with outward unit normals — for demos, tests and marker geometry that want to *show* a material rather than reconstruct a dataset. |

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
from luxar.mesh import import_mesh, import_mesh_directory

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

Use `import_mesh_directory(...)` to stack `T<number>`-indexed files, or supply
`index_regex=` for other filename conventions.

The decimation API is public on the package too — `from luxar.mesh import decimate,
decimate_ladder, decimate_cluster, DecimatedMesh, DECIMATION_METHODS,
resolve_decimation_method` — so `luxar.mesh.decimate(...)` reads the way the
`add_mesh(substitutive_lod=…)` docs describe it. (`import luxar.mesh.decimate as m`
still reaches the submodule.)

```bash
luxar mesh import bunny.ply bunny.luxar.zarr
```

## Splitting a mesh into parts

Unlike Points / Lines / GSplats, a mesh cannot be partitioned by slicing its element
arrays: a triangle is not an independent row, it is three references into a shared vertex
table. `split.py` does the re-indexing instead — faces are assigned whole to a part (the
BSP splits on face centroids, so no triangle is ever cut), each part gathers the vertices
its own faces use, and those faces are renumbered against the gathered table. A vertex on
the cut is **duplicated** into both parts, which is what lets each part stand alone as a
drawable leaf.

It is used through `add_mesh(partition=…)` rather than directly, and it returns index
bookkeeping only — the caller gathers per-vertex attributes through `MeshPart.vertex_index`,
which keeps the splitter ignorant of the attribute set. The reveal ladder is its second
caller: a ladder's levels are a true partition of the faces too, so
`add_mesh(additive_lod=…)` re-indexes each level through the same `split_mesh_by_faces`
(and orders the faces with `face_centroids`) instead of needing a re-indexer of its own.

## Not here

- **The `Mesh` node class, the writer, and the reader** — `luxar.core.mesh`,
  `luxar.io._compiler.geometry_writers.mesh`, `luxar.io.reader`.
- **An additive (prefix) LOD ladder over an *arbitrary* order.** A prefix of an
  arbitrarily ordered index buffer is a surface with holes in it, not a coarser one — so
  that flavour stays refused. A spatially coherent *reveal* is accepted instead:
  `add_mesh(additive_lod=...)` takes `method="radial"` and nothing else, which grows the
  faces best-first through adjacency, keyed on radius from the surface's own bbox centre,
  so every prefix is a connected patch per connected component (spec §9.1 — deliberately
  *not* a plain radius sort, whose prefixes interleave on a closed surface). The ladder's
  *ordering* lives in `core/group/lod/mesh.py`; the re-indexing it uses is `split.py`,
  above. *Substitutive*
  levels, which were missing only a producer, now work: `decimate.py` dispatches
  between vectorized vertex clustering and the `qem.py` Garland-Heckbert edge-collapse
  tier. QEM applies the link-condition veto that preserves manifoldness and is selected
  by `auto` through 10,000 vertices. Multi-level QEM ladders reuse one collapse
  sequence and snapshot each requested target; `add_mesh(substitutive_lod=…)` writes the
  resulting `kind=lod` group. Partitioning (`split.py`, above) is a *different* axis and
  also ships — it divides one surface in space rather than approximating it at lower detail.
  No two of the three can be combined in one `add_mesh` call. See
  `docs/specs/MESH_NODE_SPEC.md` §9.
- **Export.** The inverse direction (`.luxar.zarr` → PLY/OBJ/STL) has no consumer yet;
  `gsplats/interop/inria_export.py` is the shape it would take.
