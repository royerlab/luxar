# `luxar.mesh.interop` — classical mesh formats → Luxar

Reads PLY, OBJ, STL and glTF/GLB into a `TriangleMesh`, which the CLI writes into a
`.luxar.zarr` scene. The structural sibling of `luxar.gsplats.interop`, and shaped the
same way: one frozen intermediate, one hand-written sniffer, a `_READERS` dispatch dict,
and an `import_mesh()` that does exists-check → sniff → validate → read → normalize.

**No new dependencies.** Everything is NumPy + stdlib, matching the gsplat importer, so
`luxar mesh import` works on a bare `pip install luxar`.

## Dialects

| Dialect | Files | Notes |
|---|---|---|
| PLY | `.ply` | ascii, binary LE **and** binary BE; faces as `property list`; optional `nx/ny/nz` normals and `red/green/blue[/alpha]` colours |
| OBJ | `.obj` | 1-based **and** negative indices; polygons fan-triangulated; `v x y z r g b` vertex colours; materials ignored |
| STL | `.stl` | ascii and binary; always welded (STL is a triangle soup); per-facet normals dropped |
| glTF 2.0 | `.gltf`, `.glb` | GLB chunks, external and data-URI buffers, interleaved accessors (`byteStride`), full node-transform composition, `COLOR_0` |

## What the readers normalize, and why

- **Welding** (`--no-weld` to skip). STL always, and glTF sometimes, arrive as triangle
  soups with no shared vertices. Left that way, per-vertex normals cannot be averaged,
  the writer's authoring lint flags the node, and picking — which returns a *vertex*
  ordinal — reports a different id for the same corner depending on which triangle was
  hit. Two vertices merge only when their position **and every per-vertex attribute**
  agree: the indexed formats express a hard edge as coincident positions carrying
  different normals, so a position-only key would weld every crease in a CAD model
  flat. STL supplies no normals at all, so its key degenerates to position and the soup
  still collapses — no per-format special case is needed. The comparison is on rounded
  values; the surviving rows keep full precision.
- **Fan triangulation.** PLY and OBJ both allow polygons; quads are the common case
  from any modelling package. Taking the first three indices would drop half of each.
- **Degenerate-face removal.** Welding can collapse a sliver triangle to a line.

## Per-format traps that are handled here

- **`.ply` is shared with the gsplat importer.** A splat PLY has `scale_0`/`rot_0`/
  `opacity` and no `face` element. The sniffer recognises it and names
  `luxar gsplat import` rather than failing thirty lines deeper in a parser.
- **A binary STL's 80-byte header may begin with `solid`.** Magic-word sniffing
  misclassifies real files; the discriminator is arithmetic — a binary STL is exactly
  `84 + 50n` bytes.
- **A PLY row is laid out in property DECLARATION order.** A `face` element may carry a
  scalar before `vertex_indices` (a per-face flag or colour) and a second list after it
  (`texcoord`, from any exporter that carries UVs). Assuming list-first-scalars-after
  consumes the wrong bytes from row two onward, which silently decodes garbage faces
  rather than failing, so every property is walked in order and only the first list's
  rows are kept.
- **OBJ indices are 1-based, and may be negative** (end-relative). Reading them as
  0-based shifts the whole surface by one vertex and still produces a valid-looking mesh.
- **OBJ `vn` is indexed per CORNER, independently of `v`.** Every mainstream exporter
  deduplicates the pool, so it is generally neither the same length as the positions nor
  parallel to them. Requiring parallel indexing would throw away the normals of
  essentially every smooth-shaded export, so vertices are split per distinct
  (position, normal) pair instead — and welding, which keys on both, merges back the
  pairs that agree, leaving only genuine creases split.
- **OBJ `vn` applies only where a face references it.** A pool no `f` corner names is
  dropped even when its length matches the vertex count: matching counts are a
  coincidence, and honouring them shades the surface with normals the exporter never
  bound to a vertex. A *partial* binding is dropped whole for the same reason.
- **A glTF buffer URI is data from the file.** It must resolve inside the `.gltf`'s own
  directory; an absolute path or a `../` climb is refused rather than read.
- **glTF node transforms are correctness, not polish.** Skip the graph and every part of
  a multi-part model stacks at the origin.
- **glTF `byteStride`.** Interleaved POSITION+NORMAL is common; ignoring the stride
  decodes garbage rather than raising.
- **A glTF accessor is bounded by its own `bufferView`, and indices by their own
  primitive.** Views sit back to back in one buffer and primitives are concatenated into
  one vertex array, so both over-runs land on *valid-looking* neighbouring data — one
  attribute read as another, or a triangle stitched to the next primitive's geometry —
  and neither raises if the bound is only the whole buffer or the assembled mesh.

## Deliberate omissions

- **`gsplats/interop/_ply.py` is not reused.** It rejects `property list` (how a `face`
  element declares its indices) and everything but binary-LE. Generalizing it would put
  list handling and three format branches into a module whose only other consumer is a
  fixed-layout float table. Two parsers, on purpose.
- **STL per-facet normals are discarded.** `add_mesh` takes per-*vertex* normals; after
  welding, one vertex has several conflicting facet normals and no non-arbitrary choice
  exists. Dropping them puts the mesh on the shader's derivative flat-normal path, which
  reproduces exactly the faceted look STL describes without inventing data.
- **Draco / meshopt glTF is refused by name.** Not hand-rollable at reasonable cost;
  mirrors `read_spz`'s SPZ-v4 refusal. Run `gltf-transform` first.
- **OBJ materials, glTF textures/animations/skins/morph targets.** Luxar meshes carry
  geometry plus per-vertex colour; a per-*face* material model would need vertex
  splitting at material boundaries, which is a different feature.

## Usage

```bash
luxar mesh import bunny.ply bunny.luxar.zarr
luxar mesh import scan.stl scan.luxar.zarr --unit mm --name Skull
luxar mesh import model.glb model.luxar.zarr --no-center
```

```python
from luxar.mesh.interop import import_mesh

mesh = import_mesh("bunny.ply")  # -> TriangleMesh
scene.add_mesh(
    "Bunny",
    mesh.vertices,
    mesh.faces,
    normals=mesh.normals,
    normal_dims=[0, 1, 2],
    colors=mesh.colors,
)
```

## Testing

`tests/_synthetic.py` writes byte-exact miniature files of every dialect from one shared
ground truth — a **unit tetrahedron**, the smallest closed surface, so every vertex is
shared by three faces and welding is genuinely load-bearing (a cube would hide index
bugs behind axis-aligned symmetry). No binary fixtures are committed.

Parity is asserted on the *surface*, not on indices: welding renumbers vertices, so each
face is compared as its three sorted corner positions.
