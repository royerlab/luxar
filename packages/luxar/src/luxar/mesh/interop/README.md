# `luxar.mesh.interop` — classical mesh formats → Luxar

Reads PLY, OBJ, STL, VTK XML PolyData and glTF/GLB into a `TriangleMesh`, which the CLI
writes into a `.luxar.zarr` scene. The structural sibling of `luxar.gsplats.interop`, and
shaped the same way: one frozen intermediate, one hand-written sniffer, a `_READERS`
dispatch dict, and an `import_mesh()` that does exists-check → sniff → validate → read →
normalize.

**No new dependencies.** Everything is NumPy + stdlib, matching the gsplat importer, so
`luxar mesh import` works on a bare `pip install luxar`. The VTP reader holds to the same
bar with `xml.etree` + `base64` + `zlib`.

## Dialects

| Dialect | Files | Notes |
|---|---|---|
| PLY | `.ply` | ascii, binary LE **and** binary BE; faces as `property list`; optional `nx/ny/nz` normals and `red/green/blue[/alpha]` colours |
| OBJ | `.obj` | 1-based **and** negative indices; polygons fan-triangulated; `v x y z r g b` vertex colours; materials ignored |
| STL | `.stl` | ascii and binary; always welded (STL is a triangle soup); per-facet normals dropped |
| VTK XML PolyData | `.vtp` | `ascii` / inline base64 (`binary`) / appended `raw` **and** `base64`; `vtkZLibDataCompressor`; `UInt32` and `UInt64` headers; either byte order; `Polys` (fan-triangulated) and `Strips`; `Verts`/`Lines` and per-cell `CellData` dropped, along with any points only they referenced; `PointData` normals and colours; several `<Piece>`s concatenated into one surface; XML namespaces, prefixed or default |
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
- **Fan triangulation.** PLY, OBJ and VTP all allow polygons; quads are the common case
  from any modelling package. Taking the first three indices would drop half of each.
  VTP triangle *strips* are triangulated separately, with the alternate-winding flip a
  strip requires, and merged in as ready-made triangles.
- **Degenerate-face removal.** Welding can collapse a sliver triangle to a line.
- **Vertex compaction.** After degenerate faces are removed, the default welded import
  drops vertices no surviving triangle references and remaps every per-vertex attribute
  with the index buffer. `--no-weld` preserves the reader-produced vertex list instead.

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
- **A `.vtp` with `<AppendedData encoding="raw">` is not well-formed XML.** The bytes
  after the `_` marker are arbitrary binary — NUL bytes, `<`, `&`, invalid UTF-8 — so
  `ElementTree` refuses the *whole* document, header included. The byte stream is split
  at the `<AppendedData` start tag first (prefix-tolerant, so a fully namespace-prefixed
  document still splits), the leading portion is fed to a *pull* parser — whose first
  `start` event already carries the root with every element that completed before the cut
  hanging off it, so nothing has to be synthesised to replace the closing tags that never
  arrive — and each appended `DataArray` is indexed into the tail by its own `offset=`.
  The pull parser's event queue is then drained to the end: `feed()` *stores* a markup
  error rather than raising it, so stopping at the first event would import whatever the
  parser managed before the error and silently drop the rest. That is ParaView's default output, so it is the common case rather than
  an exotic one. `encoding=` itself is *required*, not defaulted: raw bytes and base64
  text cannot be told apart from the payload, and a wrong guess reports a valid file as
  corrupt instead of failing honestly.
- **A `.vtp` may put every tag in an XML namespace**, by default declaration or by
  prefix, so tags are matched by local name throughout. One URI may legally have more
  than one in-scope prefix at once, which is why the header parser reads the tree the
  parser already built rather than trying to rebuild each tag's source spelling: the
  spelling is not recoverable from ElementTree's Clark notation, and a uri → prefix map
  inverts a relation that is not one-to-one.
- **A `.vtp` may carry several `<Piece>`s, each numbering its own points from 0.** They
  are concatenated into one vertex array, so every index shifts by the running base —
  bounded against its own piece *first*, since after the shift an over-running index
  lands inside a neighbouring piece's geometry instead of raising. `PointData` is
  all-or-nothing across the pieces: a normals array present on only some of them would
  otherwise mean either a short attribute or invented rows.
- **A compressed inline VTP block is TWO concatenated base64 streams.** VTK encodes the
  block header (`nblocks`, block size, last partial size, then one compressed size per
  block) separately from the compressed payload and writes the two encodings back to
  back. A single `b64decode` of the element text succeeds and yields bytes that are not
  the file's data — the misparse never raises. The header's own length depends on
  `nblocks`, so it is read in three steps: decode enough characters for the first word,
  decode the full header, then decode the payload as a second stream starting at the
  character the first ended on.
- **VTP `offsets` are cumulative END offsets** with no leading `0`. Read as start
  offsets, uniform triangles merely rotate the cell list by one; mixed cell lengths also
  mis-size every cell. The per-array `type=` varies too (`Int64` from modern VTK,
  `Int32` from older writers, and `connectivity` need not match `offsets`), so every
  width is read from the array that declares it.
- **A VTP compressor other than `vtkZLibDataCompressor` is refused by name.** LZ4 and
  LZMA need codecs outside the standard library, and inflating their blocks with zlib
  would be a silent misparse rather than an error.
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

- **Unreferenced vertices are pruned during the default welded import.** Only points
  referenced by the retained triangle index buffer belong to the imported surface;
  keeping any others would inflate bounds, camera framing, and the picking ordinal
  range with geometry nothing draws. `--no-weld` preserves the reader-produced vertex
  list, including unreferenced points, as its explicit opt-out contract requires.
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
- **A `.vtu` (UnstructuredGrid) is not converted to a surface.** It is a volume mesh;
  extracting its boundary is a filter, not a read. The sniffer names the actual type and
  points at ParaView's *Extract Surface* rather than failing inside the PolyData parser.
- **A nameless float 3-vector in VTP `PointData` is not taken as colour.** In a VTK
  surface that is far more often a displacement or velocity field. Only 3- or
  4-component arrays are candidates: colour is the 3/4-component array
  `<PointData Scalars="…">` names, one carrying a conventional colour name, or a `UInt8`
  one — VTK's own unsigned-char colour convention. A `Scalars=` naming a 1-component
  field (a label, a curvature scalar) is ignored rather than painted on.
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
import numpy as np

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.mesh.interop import import_mesh, import_mesh_directory

mesh = import_mesh("bunny.ply")  # -> TriangleMesh
scene.add_mesh(
    "Bunny",
    mesh.vertices,
    mesh.faces,
    normals=mesh.normals,
    normal_dims=[0, 1, 2],
    colors=mesh.colors,
)

# P12_Ch0-registered-T0001.vtp, ... → vertices shaped (V, 5): x, y, z, t, c
timelapse = import_mesh_directory("000_deconv.ome.zarr/meshes/cells")

# Override filename parsing when frames do not use T<number>/Ch<number> tokens.
frames = import_mesh_directory(
    "exported_frames",
    index_regex=r"frame_(?P<t>\d+)",
)
time_range = (
    float(timelapse.vertices[:, 3].min()),
    float(timelapse.vertices[:, 3].max()),
)
channel_range = (
    float(timelapse.vertices[:, 4].min()),
    float(timelapse.vertices[:, 4].max()),
)


def discrete_step(values):
    unique_values = np.unique(values).astype(np.int64)
    return (
        1.0
        if unique_values.size < 2
        else float(np.gcd.reduce(np.diff(unique_values)))
    )


dimensions = Dimensions(
    [
        Dimension("x", unit="um"),
        Dimension("y", unit="um"),
        Dimension("z", unit="um"),
        Dimension(
            "t",
            unit="frame",
            range=time_range,
            step=discrete_step(timelapse.vertices[:, 3]),
            display=False,
            discrete=True,
        ),
        Dimension(
            "c",
            unit="index",
            range=channel_range,
            step=discrete_step(timelapse.vertices[:, 4]),
            display=False,
            discrete=True,
        ),
    ]
)
with LuxarZarrCompiler("cells.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=dimensions)
    scene.add_mesh(
        "Cells",
        timelapse.vertices,
        timelapse.faces,
        normals=timelapse.normals,
        normal_dims=[0, 1, 2],
        colors=timelapse.colors,
    )
```

## Testing

`tests/_synthetic.py` writes byte-exact miniature files of every dialect from one shared
ground truth — a **unit tetrahedron**, the smallest closed surface, so every vertex is
shared by three faces and welding is genuinely load-bearing (a cube would hide index
bugs behind axis-aligned symmetry). No binary fixtures are committed.

Parity is asserted on the *surface*, not on indices: welding renumbers vertices, so each
face is compared as its three sorted corner positions.

**The VTP fixtures are entirely self-attesting, and that is a real limit.** Neither VTK,
meshio nor PyVista is a dependency of this package or of its test environment, so every
`.vtp` the suite reads starts life as `_synthetic.py`'s own `_VtpWriter` output. (A few
tests then edit those bytes by hand — respelling an attribute, or corrupting one field to
check the error names the file — and one hand-writes a stub document outright, but none
of that is an independent encoder.) **No arm carries external corroboration** — not one
file from a real writer is checked against, in any encoding. Where the reader and that
writer share a misreading of the format, the tests agree with themselves and say nothing.

Two things stand in for a real file. First, the format claims the whole decoder rests on
were reasoned out against the VTK XML specification and re-derived independently, rather
than being read off the writer: `offsets` are cumulative *end* offsets, a *compressed*
inline block is two concatenated base64 streams while an uncompressed one is a single
stream, and an appended-base64 `offset=` counts *characters* and not bytes. Second, the
two most consequential of those are pinned by tests that do not go through the reader at
all — the two-stream layout by a test that reconstructs the positions **by hand** from the
raw element text, and the cumulative-ends rule by a fixture (a quad beside a triangle) on
which the wrong reading changes the face *count* rather than merely rotating the cell
list.

Neither substitutes for a byte from ParaView. In practice the highest-risk arms are the
ones no writer in common use emits, since a misreading there would also be the last to
surface in the field: big-endian (`byte_order="BigEndian"`), which no mainstream writer
produces today, and appended base64, which VTK writes far less often than appended raw.
Checking `_VtpWriter` against real ParaView output remains worth doing the first time a
`.vtp` in the wild misparses.
