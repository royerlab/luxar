# Mesh Node Specification

**Status:** Delivered — Phases 0–6 landed (writer, cull kernels, drawable, shaded, picking + panel + stats, docs; §11); real WebGPU verified pixel-equivalent to WebGL (§11 row 6). `kind=partition` (§9.2), SUBSTITUTIVE LOD levels (§9) and the §9.1 reveal ladder (authoring AND the viewer half) all now ship. The additive prefix ladder AS A LOD, spatial indexing, exact nD triangle clipping, `volumetric` blending and worker projection remain deliberate non-goals (§9).
**Scope:** A fourth first-class geometry type — `mesh` — symmetric to Points, Lines and GSplats.
**Non-goals:** the additive prefix ladder as a LOD (the §9.1 reveal, which reuses its subgroup layout, has landed), spatial indexing, exact nD triangle clipping, `volumetric` blending, worker projection. (Substitutive LOD levels — decimation — and `kind=partition` were non-goals and have since landed.) See [§9](#9-explicitly-out-of-scope).
**Target data:** isosurfaces and segmentation boundaries — 3D geometry whose hidden dimensions are
discrete (time, channel). This is a deliberate narrowing; it is what makes §5, §7 and §9 defensible.

> ## The groundwork this spec called for has already landed
>
> An earlier draft proposed a preparatory track — single-source the geometry vocabulary, collapse the
> loader registry, table-drive the hand-written dispatch — as a precondition for adding a fourth type
> cheaply. That work shipped **before** any mesh code, in three PRs: #1079 (vocabulary + registry),
> #1099 (uniform per-type pipeline + descriptor table) and #1150 (metadata symmetry + dead-code
> removal).
>
> This document is therefore now **only** the mesh node, written against the post-consolidation
> architecture. §10 records what that consolidation means for a fourth type instead of arguing for it.

---

## 1. Motivation

Luxar had three first-class geometry types, all of which are *soft, emissive, per-element* primitives
rendered as instanced quads. None of them can represent a **surface**: an isosurface from a volume, a
segmentation boundary, a cortical/organ mesh, a CAD or simulation domain, or a molecular solvent
surface. These are routine outputs in the same scientific pipelines Luxar already serves, and until
this spec landed they could only be approximated by dense point clouds.

A `mesh` node closes that gap with indexed triangles and real surface shading.

### 1.1 Existing forward-declaration

The viewer reserved the name ahead of the data model. One of those placeholders has since been folded
into the contract:

- `packages/luxar-viewer/src/types/data-monitor-types.ts` — `SceneGraphNodeType` was
  `NodeTypeName | 'mesh'` while `mesh` was a viewer-only forward declaration; #1220 put `mesh` in the
  contract and deleted the local extension, so it is now the plain alias
  `SceneGraphNodeType = NodeTypeName`.
- `packages/luxar-viewer/src/data/scene-loader/monitor/scene-graph-converter.ts:26` — `'mesh'` is in
  the display-type whitelist, and that entry is now backed by a real `GraphNodeType` rather than by a
  reservation.

These were display-only placeholders; there was no data model, loader, geometry, or material behind
them. This spec makes the name real end to end.

---

## 2. Where mesh is symmetric, and where it is not

The three-geometry symmetry rule (same attribute names, same decomposition, same shared helpers,
parallel tests) applies at the **node, attribute, writer and loader** layers. It does **not** apply at
the storage layer, nor to the ADDITIVE LOD ladder (the substitutive flavour is symmetric — see the
table), and pretending otherwise would produce a worse design.

| Layer | Symmetric with Points/Lines/GSplats? | Notes |
|---|---|---|
| `DataNode` subclass, metadata, `n_elements` | ✅ Yes | Direct mirror of `core/lines.py` |
| Scene adder (`add_mesh`) | ✅ Yes | Mirror of `add_lines` |
| Zarr writer, encoders, shared dataset helpers | ✅ Yes | Reuses `write_colors` / `write_scalars` / `SemanticType.{COORDINATE,INDEX}` |
| Render attrs (opacity/gamma/intensity/offset/blending/colormap/transform/nd_transform) | ✅ Yes | Reuses `apply_default_render_attrs`, `prepare_transform_attrs` |
| nD slicing | ⚠️ Partial | Reuses the slab *semantics*, **not** the clipping algorithm, and needs **its own tolerance strategy** (the Lines one is derived from segment interpolation and would render nothing) — see §5, §5.2.1 |
| GPU storage | ❌ No | Indexed triangles, not instanced quads — see §2.1 |
| Per-element extent | ❌ No | A mesh has no `radii`/`widths`/`amplitudes` analog — see §2.2 |
| Depth sorting | ⚠️ Partial | Registration/worker/kernel reused; the APPLY is per-triangle index permutation, not per-instance — see §6.3 |
| LOD (substitutive) | ✅ Yes | Levels are decimated surfaces — see §9. The additive ladder stays excluded *as a LOD*; the §9.1 reveal ships |
| `kind=partition` | ⚠️ Partial | Supported via `add_mesh(partition=…)`, but a part is a **re-indexing, not a slice** of the parent's arrays (with vertices duplicated across the cut) — see §9.2 |

### 2.1 The storage layer does not transfer

Points, Lines and GSplats all render as `THREE.Mesh` + `InstancedBufferGeometry` over a shared
4-vertex base quad, with per-element data packed into an RGBA32F element texture and addressed via
`aSortedIndex` (`rendering/element-texture-layout.ts`, `rendering/element-storage.ts`; 4 texels/splat,
3/point, 6/segment). See `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §4/§8.

A mesh is not a collection of independent elements — it is a *connected* indexed structure. It renders
as a plain `THREE.BufferGeometry` with `position` / `normal` / `color` / `scalar` attributes plus an
index buffer, drawn once. Consequently the mesh vertical does **not** use:

- `element-texture-layout.ts` / `element-storage.ts`
- the `gpu-buffer-pool` per-geometry adapters
- `aSortedIndex` addressing
- the depth-sort coordinator's per-instance APPLY (its registration, worker and
  kernel *are* shared — see §6.3)

This is not a gap to be closed later; it is the correct shape for the primitive.

### 2.2 There is no primary size scalar

Points→`radii`, Lines→`widths`, GSplats→`amplitudes` are each a required per-element extent, and each
feeds cull-box expansion (`LinesProjectionBounds.maxWidth`, `LinesUserData.maxWidth`, and the
equivalents for points/gsplats). A mesh vertex has no extent. Mesh bounds are therefore the raw
position AABB with **zero** expansion, and `MeshUserData` carries no `maxWidth`-analog field.

This also removes an entire class of writer work: no `write_positive_scalar` call, no `max_*` attr, no
radius-scale uniform.

---

## 3. Data model

### 3.1 Node type

`mesh` joins the canonical node-type vocabulary at its single source of truth:

```yaml
# format-contract/contract.yaml
node_types:     ["scene", "group", "points", "lines", "gsplats", "mesh"]  # done (#1220)
geometry_types: ["points", "lines", "gsplats", "mesh"]  # writable leaf subset — done (#1220)
loader_types:   ["points", "lines", "gsplats", "mesh"]  # viewer-drawable subset — done (#1241)
```

There are **three** lists, and they answer three different questions (the split landed in #1220).
`node_types` is the full vocabulary; `geometry_types` (added in #1079) is the writable leaf subset that
keys the Python writer side (bounds, `luxar info`, LOD backfill); `loader_types` (added in #1220) is the
viewer-drawable subset that keys per-geometry dispatch. The generator enforces the nesting
`loader_types ⊆ geometry_types ⊆ node_types`, that `geometry_types` and `loader_types` are each
non-empty and duplicate-free, and that `geometry_types` is free of the container types — so a half-done
addition fails `check-contract` with a named error rather than drifting silently.

`mesh` is **already in `node_types` and `geometry_types`** (#1220) — and #1220 also landed the whole
Python writer vertical behind those entries (`core/mesh.py`, `add_mesh`, validators, compiler writer,
reader, `info`; the Phase-1 checklist in §8) — so a mesh leaf is writable today.
`mesh` is now also in `loader_types` (the Phase-3 switch-on landed in #1241),
which turned on the three §10.2 compile errors that mapped out the viewer work — all
since resolved. The contract still names the writable and drawable sets separately
because a type becomes authorable before it becomes drawable; today both sets include
`mesh`.

`make gen-contract` regenerated both projections
(`packages/luxar/src/luxar/typing_utils/_format_contract.py` and
`packages/luxar-viewer/src/types/format-contract.ts`) for the node-type addition, and `hatch run
check-contract` gates drift. As part of #1220 the `'mesh'` local extension in `data-monitor-types.ts`
was deleted — `SceneGraphNodeType` is now plain `NodeTypeName` — and `NodeType.MESH = "mesh"` /
`NODE_TYPE_MESH = "mesh"` were added to `luxar/typing_utils/enums.py` and `constants.py`. Adding `mesh`
to `loader_types` (the Phase-3 switch-on, #1241) re-ran `gen-contract` for the viewer projection.

### 3.2 Arrays

| Array | dtype | Shape | Required | Semantic type | Notes |
|---|---|---|---|---|---|
| `vertices` | float32 | `(V, D)` | **yes** | `COORDINATE` | nD, exactly like `Lines.vertices` |
| `faces` | uint32 | `(F, 3)` | **yes** | `INDEX` | Triangle vertex indices |
| `normals` | float32 | `(V, 3)` | no | `COORDINATE` | Per-vertex; paired with a required `normal_dims` attr — see §3.4 |
| `colors` | uint8/uint16/float32 | `(V, 3\|4)` | no | color helpers | RGB or RGBA; the 4th component is a **load-bearing** per-vertex opacity — see §6.2 |
| `scalars` | float32/float16/uint8 | `(V,)` | no | scalar helpers | Colormap lookup |
| `uvs` | float32 | `(V, 2)` | no | `COORDINATE` | Required iff `texture`; values outside `[0, 1]` are legal |
| `texture` | raw numeric or encoded uint8 bytes | `(H, W, C)` for `raw`; `(B,)` for codec payloads including `ktx2` | no | `COLOR` for raw | Per-node base colour; mutually exclusive with other base-colour sources |
| `label_offsets`/`label_bytes` | — | CSR | no | — | Per-vertex hover tooltips |
| `image_label_*` | — | CSR | no | — | Per-vertex hover thumbnails |

`faces` uses `SemanticType.INDEX` with `deduplicate=False` and `allow_lut=False`, for exactly the
reason `Lines.segments` does: the loader reads it as raw chunked zarr and does not resolve `array_ref`,
so dedup would silently drop geometry for a byte-identical sibling, and LUT encoding of grid-snapped
values would decode as garbage topology.

KTX2 is authored from uint8 `(H, W, 3|4)` pixels but stored as an opaque `(B,)`
container. Authoring requires `toktx` 4.1.0 or newer and rejects a produced
container whose KTX2 identifier, supercompression scheme, or DFD colour model
does not match the requested codec. Admission charges
`ceil(width * height * 4 / 3)` bytes for the native compressed surface plus its
full mip tail. A renderer with no native ASTC, ETC1/2, S3TC/BC or PVRTC target
rejects the node; an uncompressed RGBA8 transcode fallback is not permitted
because it would exceed that device-independent charge.

**Winding convention:** faces are wound counter-clockwise as seen with the mesh's authored spatial
triple in ascending index order (front-facing under `FrontSide`, §6.1). For a 3D mesh that frame is
trivially `[0,1,2]`; for an nD mesh it is `sorted(normal_dims)` when normals are present — there is no
other signal for which three axes the author wound against, and no winding can be counter-clockwise
under *every* 3D projection of an nD mesh (orientation under a different axis triple is per-triangle
data-dependent). The viewer restores front-facing winding only when the displayed *set* equals that
frame and its order is an odd permutation of it (§5.4/§7); for any other displayed triple — or an nD
mesh with no stored normals — `double_sided: false` falls back to `DoubleSide` for the epoch (§5.4).
The frame depends only on the *set* of `normal_dims`, not its order (§3.4).

**Naming.** `faces` (not `triangles`, not `indices`) — it parallels `segments` as the topology array,
reads correctly in the mesh domain, and leaves `indices` free for its existing meaning in the Lines
`line_type='indexed'` authoring API.

### 3.3 Metadata attrs

Mirrors `LinesMetadata` minus the extent/line-type fields:

```
type: "mesh"
n_vertices: int
n_faces: int
ndim: int
has_normals: bool
normal_dims: [int, int, int]  # required iff has_normals; see 3.4
has_colors: bool
has_scalars: bool
has_uvs: bool
has_texture: bool
texture_encoding: "raw" | "png" | "webp" | "jpeg" | "ktx2"
texture_width: int
texture_height: int
texture_channels: 1 | 3 | 4
texture_color_space: "srgb" | "linear"
texture_data_range: [float, float]  # HDR raw textures only
has_labels: bool
has_image_labels: bool
shading: "smooth" | "flat" | "none"  # "none" is explicit unlit; never the default
double_sided: bool              # default true
position_bounds: {"min": [...], "max": [...]}  # nD vertex bbox, per io/_compiler/bounds.py
ordering: "none"                # v1 always; reserved for a future spatial index
```

plus the standard render attrs already handled by `apply_default_render_attrs` and
`prepare_transform_attrs` (`opacity`, `gamma`, `intensity`, `offset`, `absorption`, `blending_mode`,
`colormap`, `scalar_data_range`, `layer`, `transform`, `nd_transform`, `extend_to_all`).

`MESH_RESERVED_ATTRS` is added to `io/_compiler/node_common.py` alongside the other three frozensets:

```python
MESH_RESERVED_ATTRS = frozenset({
    "type", "n_vertices", "n_faces", "ndim",
    "has_normals", "normal_dims", "has_colors", "has_scalars", "has_uvs",
    "has_texture", "texture_encoding", "texture_width", "texture_height",
    "texture_channels", "texture_color_space", "texture_data_range", "has_labels",
    "has_image_labels", "has_keys", "shading", "double_sided", "position_bounds",
    "ordering",  # mesh-only: no spatial index, so a supplied ordering can't be honoured
})
```

All four sets — `POINTS_/LINES_/GSPLATS_RESERVED_ATTRS` and `MESH_RESERVED_ATTRS` — now reserve
`has_image_labels`; #1220 aligned the three sibling sets with the mesh set, so every set covers every
presence flag its writer stamps.

### 3.4 Normals are 3D, positions are nD

Positions live in nD like every other geometry type. Normals are a **display-space** quantity: they are
only meaningful for the three displayed dimensions, so re-deriving them when `displayDims` rotates (not
on a plain slice move) is the correct behaviour — the `displayDims`-change rebuild path in §7.

Therefore `normals` is stored as `(V, 3)`, accompanied by a **required companion attr** recording which
three dimension indices those components correspond to:

```
normals:     (V, 3) float32
normal_dims: [i, j, k]      # center-column indices, e.g. [0,1,2] or [1,2,3]
```

Stored normals are used **iff `shading == "smooth"` and `normal_dims` equals the active `displayDims`**.
Otherwise — when `shading == "flat"`, whenever `normals` is absent, or whenever `normal_dims` no longer
matches the active `displayDims` — the viewer computes **flat face normals** from the projected triangle
via a cross product (§6.2), which is exactly why the derivative fallback is not optional. `shading` is thus
a first-class input to this decision, not inert metadata: an explicit `"flat"` overrides otherwise-valid
stored normals to give a faceted surface.

⚠️ **Do not store normals against an implicit "first three dimensions".** For a `(t, x, y, z)` mesh the
first three dims are `(t, x, y)` and such a normal is meaningless. This is a bug class the codebase has
already been burned by and documented: `rendering/depth-sort-coordinator/render-order.ts:79,110` warns
that the serialized BSP `axis` is a *center-column* index which must be mapped through `displayDims`,
and that "the two coincide only for `displayDims == [0, 1, 2]`". An explicit index list is also the
established convention on the Python side (`gsplat transform --spatial-dims`,
`cli/gsplat_ops/transforms/commands.py:717`).

Making `normal_dims` explicit turns an invisible wrong-orientation render into a cheap, checkable
equality — and costs one attr.

An alternative — storing a full `(V, D, 3)` normal frame so any `displayDims` has true smooth normals —
was rejected as over-engineering for v1; the flat-normal fallback covers it correctly, just without
smoothing.

### 3.5 Validation

New shared validators in `luxar/validation/base.py`, following the existing `validate_*_for_writing`
convention (fail-fast, before any zarr group is created):

- `validate_vertices_for_writing(vertices)` — enforces `n_vertices = vertices.shape[0] <= 2^27`, the
  **same** alias-free bound as the §6.5 pick vote-key stride and the §3.5 loader gate below. This mirrors
  the loader gate at write time so the public `add_mesh` path cannot emit a store that Luxar's own loader
  then rejects — restoring the fail-fast contract this section opens with, and honouring the §6.5 house
  rule that an unenforced bound is not a bound. It also restores a secondary guarantee: with
  `n_vertices <= 2^27` pinned at write time, `validate_faces_for_writing`'s `max < n_vertices` check again
  guarantees every admitted face index (max `< 2^27`, well under `2^32`) survives the `.astype(np.uint32)`
  cast. Raise `ValidationError(message, hint)` with a remediation hint, matching the shared-validator half
  of the split below; the cap is a pure function of the `vertices` array and independently testable, so it
  belongs on the shared-validator side, not among the cheap structural gates that need writer context and
  stay inline. The validator's sole job is this vertex-count cap; generic coordinate finiteness/shape stays
  in the shared coordinate-writing path.
- `validate_faces_for_writing(faces, n_vertices)` — shape `(F, 3)` or flat `(3F,)`; integer dtype
  (reject float, which `.astype(np.uint32)` would silently truncate); `min >= 0`; `max < n_vertices`;
  `F >= 1`. Mirrors the `line_type='indexed'` index gate at `geometry_writers/lines.py:134-170`, which
  is the closest precedent and already encodes each of these traps.
- `validate_uvs_for_writing(uvs, n_vertices)` — shape `(V, 2)`, exactly one finite `(u, v)` pair per
  vertex. Values outside `[0, 1]` remain legal because `texture_wrap="repeat"` intentionally tiles them.
- `validate_normals_for_writing(normals, n_vertices)` — shape `(V, 3)`, finite. Zero-length normals are
  **warned**, not rejected (degenerate triangles legitimately produce them). Render-time handling is
  **pointwise, not per-face**: on a shared-vertex indexed mesh the interpolated normal blends toward the
  neighbouring vertices' directions, so the stored-normal fragment variant (§6.2) simply epsilon-guards
  its `normalize` — when the interpolated normal is not affirmatively valid (`!(dot(N, N) >= ε)` before
  normalization — the *negated* form on purpose: `NaN` fails every comparison, so a corrupt store's
  `NaN` normal takes the same fallback instead of slipping past a `dot(N, N) < ε` test and normalizing
  into `NaN` shading) it
  falls back to the §6.2 screen-space-derivative flat normal rather than normalizing a zero vector into
  NaN shading. That substituted normal also **bypasses** §6.2's `gl_FrontFacing` two-sided flip, which
  would otherwise negate an already-viewer-facing normal — see the two-sided-normal bullet there.
  Shading near a degenerate vertex is therefore locally distorted rather than cleanly flat;
  the warning exists so authors fix the normals instead of relying on the guard.
- `normal_dims` (§3.4) — exactly 3 entries, integers, distinct, each `0 <= i < ndim`. Required when
  `normals` is supplied; rejected when it is not. It is an explicit `add_mesh` parameter (§4) — as
  writer-reserved metadata (§3.3) it cannot ride in through `**attrs`.

**Raise `ValidationError`, not `ValueError`.** Note the precedent cited above is split: the shared
`validate_*_for_writing` family in `validation/base.py` raises `ValidationError(message, hint)` — a
two-arg form that gives the user a remediation hint — whereas the Lines indexed-index checks are
**inline in the writer** and raise bare `ValueError`. Mesh should follow the *shared validator* half of
that precedent: vertex/face/normal validation belongs in `validation/base.py` as reusable, independently
testable functions, matching `validate_widths_for_writing` / `validate_radii_for_writing`. Only the
cheap structural gates that need writer context stay inline.

Tests mirror `validation/tests/test_lines_validation.py`, which is already parametrized over
`(factory, error_pattern, test_id)` triples — reuse that shape so each rejection in this section gets
its own named case, and **verify each fails before the validator exists** (a test that passes against
a no-op validator is vacuous).

**Loader-side validation (viewer).** The validators above run at write time and protect only stores this
writer produced; the viewer loads arbitrary — externally produced or corrupted — stores and hands `faces`
straight to the §5.4 kernels. An out-of-range face index **panics** the Rust kernel (the crate is
`panic = "abort"`, so the trap escapes as an opaque, uncatchable `RuntimeError: unreachable` rather
than a node-scoped error) and silently corrupts the TS backend
(out-of-bounds reads yield `undefined`), so the loader must structurally validate before either backend
is invoked. But the whole-node loader (§7) fetches and decodes every array in full up front, so a check
that runs only *after* decode arrives too late for the quantities that gate admission: a corrupt or
hostile store (the viewer loads arbitrary `?src=` URLs) can declare enormous arrays and exhaust tab
memory before the `LoaderError` containment ("one node lost, not the scene") is ever reachable. The gate
therefore runs in **two stages**, and everything decidable from metadata is checked *first*, before a
single chunk is fetched.

**Stage 1 — metadata preflight (before any array materialization).** Runs purely on the node attrs
(`n_vertices`, `n_faces`, `ndim`, `has_normals`/`normal_dims`, the presence flags §3.3) and each array's
zarr `.zarray` metadata (declared shape, `chunks`, and dtype), touching no chunk data. It (a) rejects
`n_vertices > 2^27` — the pick vote-key stride bound (§6.5) — so a giant vertex count is refused before
allocation, not after a multi-gigabyte fetch; (b) bounds `n_faces`, which the writer floors at `F >= 1`
but never caps, against a viewer-side per-node ceiling `MESH_DECODE_BUDGET_BYTES` (a viewer `src/config/`
constant, default 512 MiB — a few-million-triangle mesh's `vertices`+`faces` run to tens–hundreds of MB,
so the default admits the §7 workload expectation with several-fold headroom; why it must sit well
*under* what a tab survives is the transient-peak multiplier below): both the summed declared
footprint — each array's declared shape × its *declared-dtype* itemsize: `vertices` `V·D·4`, `faces` `F·3·itemsize` (8 bytes per index for
an external int64 store, not the canonical uint32's 4 — budgeting the canonical dtype instead of the
declared one would let a 64-bit store fetch twice the audited bytes), every present optional array and
the CSR arrays included — and each array's *per-chunk* decode
allocation (`chunks × itemsize`, edge chunks padded to the full chunk shape) must fall under it — the
per-chunk term because zarr allocates chunk-shaped buffers, not shape-shaped ones, and zarr v2 does not
require `chunks <= shape`, so a `faces` `"shape": [100, 3], "chunks": [268435456, 3]` declaration would
otherwise slip a ~3 GB first-chunk allocation past a shape-only budget; (c) cross-checks every declared
`.zarray` shape and dtype against `n_vertices`/`n_faces` and the §3.2 array table — `vertices` `(V, D)`
float32 with `D == ndim` (a `vertices` width that disagrees with the `ndim` attr would otherwise index
out of slice bounds in the §5.4 slab kernel — a `panic = "abort"` trap, the exact class this gate
exists to stop), `faces` `(F, 3)` of an **integer** dtype (so `faces` materializes to a multiple of 3; float is
rejected because it truncates in the u32 cast, mirroring the write-side `validate_faces_for_writing`
rule — §3.2's uint32 is this writer's canonical dtype, but an external integer store is coerced to u32,
a coercion Stage 2 makes value-preserving by range-checking the *source* values first), and each
present optional
`normals` `(V, 3)`, `colors` `(V, 3|4)`, `scalars` `(V,)` — because these optionals bind as enabled
vertex attributes on an **indexed** draw (§6.1): an undersized attribute doesn't trap, it makes
`drawElements` read past the buffer (an invalid-operation draw or silent zeros, backend-dependent) and
mis-shades every vertex it covers, and a declared-undersized array is caught here from its shape alone;
and (d) checks `normal_dims` well-formed whenever normals are present — exactly 3 entries, distinct
integers, each `0 <= i < ndim` — decidable from the attrs alone, so it never forces a fetch. The decode
layer must in turn allocate from the declared chunk `nbytes` and reject any stream that decompresses to
a different size, so a blosc header claiming gigabytes cannot win either. Be clear about what the
ceiling bounds: the *declared source* footprint plus any single decode buffer — not the loader's whole
transient peak. On the admission path the decoded sources coexist with derived copies — the u32-coerced
`faces`, the extracted display-space `position` (§6.1), the driver-side GPU upload — each itself bounded
by the source footprint, so the worst-case transient peak is a small known multiple (≈ 3–4×) of the
ceiling. The default prices that multiplier in: 512 MiB of admitted declaration keeps the worst-case
transient around 2 GiB, comfortably inside a 64-bit tab — which is also why the ceiling must never be
raised toward "what a tab survives"; the tab has to survive the *multiple*, not the ceiling. The ceiling
is **per node** —
N nodes can still sum to N×budget, so the "one node lost, not the scene" guarantee is per-node; v1
imposes no aggregate cap. A §9.1 reveal ladder is a single node for this purpose: its levels are summed
and charged once against the same ceiling, on the ladder's first load — before any level's chunks are
fetched, and at the same point a leaf's own budget is enforced, so a refusal gets the same failure
containment (recorded, banner entry, siblings unaffected) as a leaf's — retryable for a level's own
preflight rejection, while the aggregate over-budget verdict is cached and re-thrown rather than
re-derived, no retry being able to make the sum fit. Any failure fails
the node with a `LoaderError` (one node lost, not the scene)
**without fetching a single chunk**, preserving the blast radius before allocation.

The `n_vertices <= 2^27` cap belongs at this preflight because mesh's pick `elementId` is `gl_VertexID`
(§6.5) — the one type not bounded by the element-texture capacity — and once a vertex ordinal reaches
the pick vote-key stride (`2^27`) the vote key silently aliases across nodes (the largest ordinal is
`n_vertices - 1`, so `n_vertices <= 2^27` is the exact alias-free bound: every admitted ordinal stays
strictly under the stride), so the bound must be enforced here, not assumed from the §7 whole-load
workload.

**Stage 2 — post-decode value checks (after fetch + decode).** The remaining checks genuinely need the
materialized arrays. First, each materialized array's length/shape must equal the shape Stage 1 admitted
(`vertices`/`normals`/`colors`/`scalars` length `V`, `faces` `3F`) — Stage 1 vets only the *declared*
`.zarray` shape, so a store that declares correctly but materializes a short array (a raw or mis-sized
chunk, a non-compliant decoder) would otherwise resurrect the undersized-attribute `drawElements`
over-read Stage 1(c) closes. Then every face index in `[0, V)` — a **two-sided check on the
source-typed values, before the integer→u32 coercion**, because each side of the cast hides its own
wrap-around: an externally produced *signed* store's `-1` passes a one-sided pre-cast `< V` check and
wraps to `0xffffffff`, while a 64-bit store's `2^32 + 1` survives a check run only *after* the cast —
it wraps to `1`, lands inside `[0, V)`, and silently rewrites topology instead of trapping. The
two-sided source-value check rejects both, and because Stage 1 admits only `V <= 2^27`, every index it
passes is preserved bit-for-bit by the u32 cast — so the values checked are exactly the values the
kernels receive. Finally, the label and image-label CSR offsets monotone and in-bounds (§3.2). Same
`LoaderError`, same one-node
blast radius; these run only once Stage 1 has admitted the declared shapes and budget, so the
fetch+decode they gate is already bounded.

Stage 2 deliberately does **not** finite-scan the float arrays (`vertices`, `normals`, `colors`,
`scalars`). A non-finite value can neither trap a kernel nor over-read a buffer, and its blast radius
is already per-node without a gate: a `NaN`/`±Inf` coordinate on a hidden dimension hides the vertex
(§5.2's #806 rule), a non-finite *displayed* coordinate corrupts at most that node's rasterization and
bounding sphere (which the depth-sort coordinator already refuses to sort by —
`depth-sort-coordinator/render-order.ts` checks `Number.isFinite` on every sphere it uses), non-finite
colors/alpha are clamped by the shared shader sanitizers (§6.2's `sanitizeAlpha` and the
`materials/_shared` helpers), and a non-finite stored normal degrades only that node's shading —
contained because the §3.5 normalize guard is written in its NaN-robust negated form (above). No
sibling loader finite-scans its decoded positions either; mesh matches that policy rather than
inventing a stricter one here.

### 3.6 Authoring lint

Reusing the pattern of `_exploded_chain_fraction` in the lines writer (a warn-only heuristic that
catches a common authoring mistake), the mesh writer emits one warn-only lint:

> **Unwelded vertices.** If `V == 3F` and no two faces share a vertex index, the mesh was authored as
> independent triangles rather than a welded indexed surface. Smooth shading is impossible, per-vertex
> normals are meaningless, and the vertex array is ~3× larger than needed. Suggests welding.

Gated behind a minimum face count so tiny test meshes stay quiet, and routed through
`ctx.claim_*_warning` so a partition's leaves collapse to one message.

---

### 3.7 `dim_order`, `normal_dims` and face winding

`add_mesh(dim_order=[...])` renumbers the vertex **columns** into the scene's
dimension order (widening to the scene's `ndim`, filling any dimension the caller
did not name). This section exists because that interaction was unspecified until
#2141 — and the omission cost a reviewer a wrong diagnosis before it cost anyone a
bug, which is the more useful thing to record.

**`normal_dims` names SCENE dimension indices**, i.e. the layout *after*
`dim_order`. This is the contract, and it is worth stating baldly because the
alternative reading is superficially more natural — the array is "authored", so
surely its companion attr indexes authored columns? It does not. Three
independent pieces of evidence:

* §3.4's own example, "for a `(t, x, y, z)` mesh those are `(t, x, y)`", describes
  the *stored* layout;
* the writer range-checks `normal_dims` against the post-`dim_order` `ndim`;
* `demo_lsystem_forest` authors four columns
  (`dim_order=["season", "x", "y", "z"]`) and passes `normal_dims=[2, 3, 4]` — one
  of which exceeds every authored column index — with the inline comment "The
  three scene dims the normals describe".

**So `normals` needs no companion transform, and this is where mesh differs from
gsplats.** GSplats' Cholesky factors are authored in the *source* frame, so
`apply_dim_order_cholesky` must carry them through the map alongside the centers.
A mesh's normal components are already expressed in the destination frame by
contract, so there is nothing to remap: an `apply_dim_order_normals` would be a
double transform, and would corrupt exactly the callers who read the contract
correctly. The asymmetry between the two types is real, and it is in the
*contract*, not in the adders' completeness.

**Face winding is the part `dim_order` can invalidate, and Luxar deliberately
does not repair it.** Since `cross(Ra, Rb) = det(R)·R·cross(a, b)`, an
orientation-reversing column permutation negates a triangle's geometric normal
while leaving its stored corner order untouched. Whether that makes the store
*wrong* depends on which frame the caller wound in:

| The caller wound faces CCW in… | After an orientation-reversing `dim_order` |
|---|---|
| the SCENE column order (what §3.2 asks for, read literally, since the frame is `sorted(normal_dims)` and those are scene indices) | already correct — nothing to fix |
| their own AUTHORED column order | now clockwise in the scene frame; violates §3.2 |

Nothing in the store distinguishes the two, so an automatic flip would fix the
second caller by breaking the first. The writer therefore **warns** — the same
warn-only posture as §3.6's unwelded-vertices lint — naming both consequences:
with `double_sided: false` the surface renders inside-out and an open surface can
vanish; with stored normals, `gl_FrontFacing` chooses the wrong sign and flips the
shading gradient even when `double_sided` is true. The one-line winding remedy is
`faces[:, [0, 2, 1]]`. It is silent only with no `normals`, because
`sorted(normal_dims)` is the only declared winding frame there is (§3.2) and the
viewer already renders such a mesh `DoubleSide` regardless.

Handedness is judged on the frame's **preimage**: walk `sorted(normal_dims)` in
ascending scene order, record which authored column each axis came from, and test
whether that sequence is an odd permutation. A frame axis with no preimage (an
unmapped, constant-filled scene dimension) makes the restricted map not a
permutation at all, and is skipped rather than guessed at.

`faces` is still never reindexed, for the reason the `add_mesh` docstring gives:
it addresses vertex *rows*, and `dim_order` permutes *columns*, so a row index
names the same physical vertex afterwards. The lesson of #2141 is that this was
the *whole* of the recorded reasoning — `faces` was considered as indices and
never as orientation — and that the missing sentence was about winding, not about
the arrays.

---

## 4. Python API

```python
scene.add_mesh(
    name: str,
    vertices: NDArray[np.float32],          # (V, D)
    faces: NDArray[np.uint32],              # (F, 3)
    normals: NDArray[np.float32] | None = None,
    normal_dims: Sequence[int] | None = None,   # required iff normals is given — §3.4/§3.5
    colors: NDArray | Sequence[float] | None = None,
    scalars: NDArray[np.float32] | float | None = None,
    uvs: NDArray[np.float32] | None = None,
    texture: NDArray | None = None,
    *,
    texture_encoding: Literal["raw", "png", "webp", "jpeg", "ktx2"] = "raw",
    texture_width: int | None = None,
    texture_height: int | None = None,
    texture_channels: int | None = None,
    texture_color_space: Literal["srgb", "linear"] = "srgb",
    texture_ktx2_mode: Literal["uastc", "etc1s"] = "uastc",
    texture_ktx2_quality: int | None = None,
    shading: Literal["smooth", "flat", "none"] | None = None,
    double_sided: bool = True,
    labels: Sequence[str] | None = None,
    image_labels: Any | None = None,
    **attrs,                                 # opacity, colormap, transform, blending_mode, ...
) -> Mesh
```

`shading` resolves as: `None` (the default) → `"smooth"` when `normals` is supplied, else `"flat"`; an explicit
`"smooth"` with no stored `normals` has nothing to smooth — the writer stamps the value as given and the
viewer's §6.2 rule falls back to the flat derivative normal at render time (no write-time rewrite); an
explicit `"flat"` is always honored and renders the faceted derivative-normal surface even when `normals`
is present (§3.4, §6.2). Explicit `"none"` is unlit and computes no normal.

`normal_dims` is §3.4's required companion attr, surfaced as an **explicit keyword** because it has no
other way in: it is a member of `MESH_RESERVED_ATTRS` (§3.3), so passing it through `**attrs` fails the
write as a reserved-key collision. The adder forwards it to the writer alongside `normals`, and §3.5
validates the pair (required when `normals` is supplied, rejected when it is not).

Placement mirrors the other three exactly:

| Concern | File |
|---|---|
| Node class | `luxar/core/mesh.py` (mirror of `core/lines.py`) |
| Adder | `luxar/core/group/adders/mesh.py` (mirror of `adders/lines.py`) |
| Wiring | `core/group/group.py`, `core/scene/scene.py`, `core/__init__.py`, `luxar/__init__.py` |
| Writer | `luxar/io/_compiler/geometry_writers/mesh.py` |
| Facade | `luxar/io/compiler.py::write_mesh` |
| Reader | `luxar/io/reader.py` — `MeshData`, `get_mesh()`, `list_meshes()` |
| CLI | `luxar/cli/info_command.py` — vertex/face counts in `luxar info --stats` |

`add_mesh` accepts a broadcast scalar color/scalar exactly as `add_lines` does, and routes through the
same `validate_broadcast_color` / `validate_scalars_preflight` gate.

---

## 5. nD semantics: per-vertex slab test, whole-triangle cull

### 5.1 The decision

Lines clip a segment against the nD slab and interpolate every attribute at the clip parameter
(`wasm/rust/src/lines_clipping.rs`, 1381 LOC + a 591-LOC TypeScript parity backend). The exact
equivalent for a triangle is nD polygon clipping: a triangle cut by the slab becomes a convex polygon
of up to `3 + k` vertices, requiring fan re-triangulation and attribute interpolation at each new
vertex, every frame the slice moves.

**v1 does not do this.** Instead:

> A triangle is rendered **iff all three of its vertices pass the nD slab membership test.**

### 5.2 Semantics, reused verbatim from the lines kernel

For each non-displayed dimension `d`, with `slice_min = slice_position[d] - tolerance[d]` and
`slice_max = slice_position[d] + tolerance[d]`:

- vertex is **in** iff `v[d] >= slice_min && v[d] <= slice_max` for every such `d`;
- a **non-finite** (`NaN` or `±Inf`) coordinate on any non-displayed dimension makes the vertex
  **invisible** — matching the `#806` rule enforced identically in both lines backends
  (`lines_clipping.rs:75-80`, `lines-clipping.ts`);
- `extend_to_all` dimensions get infinite tolerance via the existing
  `EXTEND_TO_ALL_TOLERANCE` path (`= 1e10`), unchanged.

This is precisely the `p1_in` branch of `clip_segment_single`, applied per vertex and AND-ed across the
three vertices of a face.

#### 5.2.1 Tolerance — mesh needs its own strategy, and cannot reuse the Lines one

`computeTolerance(geometryType, …)` switches per type, and **each existing strategy is derived from
that type's per-element extent**:

| Type | Hidden *spatial* dim | Hidden *discrete* dim |
|---|---|---|
| Points | `maxRadius` | quarter-cell (`discreteDimTolerance`) |
| Lines | **`0`** — "bounds already include width" | quarter-cell, or **half-cell** when `discreteRole: 'membership'` |
| GSplats | float-safety epsilon `max(1e-3 × step, T × 1e-5)` where `T` is the node's own `truncation_radius` (2.75 by default, so `max(1e-3 × step, 2.75e-5)` for a typical node) — the chunk bounds already carry the `truncation_radius · σ` extent | quarter-cell |

⚠️ **Mesh must NOT copy the Lines row.** Lines can use `0` because segment clipping *interpolates
through* the slab — a segment crossing the slice yields a clipped intersection even at zero thickness.
Mesh has whole-triangle cull (no interpolation) and no per-element extent (§2.2), so a spatial
tolerance of `0` reduces the membership test to **exact float equality with the slice plane** and the
node renders **nothing**.

Mesh therefore adds a fourth arm to `computeHiddenDimTolerance`:

- **Discrete hidden dims** → `discreteDimMembershipTolerance` (half-cell). Mesh's slab test is a
  MEMBERSHIP gate, exactly like the lines projection-clipping slab, but unlike lines it does not
  *request* that role: mesh has no spatial index and issues no range query, so membership is the
  only rule it has and `computeMeshHiddenTolerance` deliberately IGNORES `discreteRole`
  (`data-processor-mesh.ts` passes only the authored `meshSlabTolerance`, never a role). Honouring a `'query'` role here
  would hand back the *fetch reach* (deliberately `< 0.5 × step`) to the one caller that is asking
  about visibility, and drop on-grid geometry. **This is the dominant real case** —
  a mesh's hidden dimensions are almost always time or channel.
- **Continuous hidden spatial dims** → `step × meshSlabTolerance`, default `1.0` (one cell). This is
  a half-width: the slab spans `slice ± step × slab_tolerance`. It is **authored**, not viewer-internal:
  the mesh-only `slab_tolerance` node attr is measured in cells of the hidden dimension's own `step`,
  strictly positive (zero would reduce membership to exact float equality with the slice plane and
  render nothing), and rejected on every non-mesh node. The viewer reads it off `MeshMetadata` and
  forwards it as `ToleranceOptions.meshSlabTolerance`; see the format guide's *nD slicing:
  whole-triangle cull* for the user-facing account.
  Mesh is the only type with a *tunable* continuous arm: a mesh has no
  per-element extent, so the slab thickness is invented rather than measured. GSplats exposes no
  equivalent slab knob — its chunk bounds already carry the real `truncation_radius · σ` extent, so
  its continuous arm is a float-safety epsilon (`gsplatsContinuousDimTolerance`); the two inputs that
  do vary it are read off the store, not authored — the node's own `truncation_radius`
  (`ToleranceOptions.truncationRadius`) and the writer's published barrier set
  (`ToleranceOptions.barrierDims`, the `slice_dims` attr, which decides whether a dim takes this arm
  at all). Neither reaches a mesh: it publishes no `slice_dims` (no spatial index) and has no
  truncation radius, and `isBarrierDim` honours a published set for gsplats ONLY — precisely because
  both of a mesh's arms are membership gates, so narrowing one would change what the user sees. So
  `meshSlabTolerance` really is the only control a mesh has.

Be honest about what the second bullet means: with per-vertex cull there is no such thing as a true
cut, so a continuous hidden dimension renders a **thick slab** ("the surface near this slice"), not a
planar section, and the slab thickness is the only control. Exact nD clipping (§9) is the fix; until
then a mesh whose hidden dims are continuous and spatial is a poor fit for this node type, and the
loader says so once, by name — `noticeContinuousHiddenDim` in
`data/scene-loader/process/data-processor-mesh.ts` logs an `info` line naming every such dimension
and its unit, deduplicated per node **and** dimension (a node's not-yet-reported dimensions are
named together in one line, and an axis that becomes hidden later still gets its own). One
carve-out, so silence is not over-read: a dimension listed in `extend_to_all` is slice-invariant, so
its membership slab is infinite and the approximation cannot bite — it is deliberately NOT reported.
That line is also the evidence gate on §9's deferral of exact clipping.

### 5.3 Consequence, stated plainly

A surface cut by a slice shows a **ragged, triangle-quantized boundary** rather than a clean planar
cut. For a well-tessellated mesh sliced with a tolerance comparable to the edge length this reads as a
slightly jagged edge; for a coarse mesh with a thin tolerance it can drop whole regions. This is a real
visual limitation and must be documented in the user guide, not glossed.

**Discharged in #2144**, and worth recording how long it took: this sentence carried no §8 checklist
item, so for six phases it was neither ticked nor missed. `docs/guides/user/LUXAR_ZARR_FORMAT.md` §5
now carries an *nD slicing: whole-triangle cull* subsection (the rule, the ragged edge, the thick slab,
`slab_tolerance`, and when to reach for gsplats instead) and `VIEWER_GUIDE.md`'s nD Navigation section
cross-references it — every other rule there describes per-element visibility, which is exactly what a
mesh does not do. A "must" in this document with no checklist item behind it is a "maybe".

It is the right v1 trade: it costs **~140 LOC of new kernel** instead of ~1500, requires no
re-triangulation, no new vertices, and no attribute interpolation machinery.

### 5.4 Kernel

Two new functions, one per backend, kept in 1:1 parity like every other kernel pair
(the TypeScript reference is also the production `ndim > 16` backend — see the WASM 16-dimension note
in `CLAUDE.md`):

```rust
// wasm/rust/src/mesh_culling.rs   (~90 LOC)

/// Per-vertex nD slab membership. Output mask [num_vertices], 1 = in.
pub fn mesh_vertex_visibility_mask(
    positions: &[f32], slice_position: &[f32], tolerance: &[f32],
    display_dims: &[u32], ndim: usize, num_vertices: usize,
    output: &mut [u8],
) -> u32;

/// Compact `faces` to those whose three vertices are all visible.
/// Writes ORIGINAL (un-remapped) vertex indices into `output`.
pub fn compact_visible_faces(
    faces: &[u32], vertex_mask: &[u8],
    num_faces: usize, output: &mut [u32],
) -> u32;
```

**No vertex compaction.** On a per-*slice* change (`displayDims` unchanged) only the *index buffer* is
rebuilt; the vertex attribute buffers are uploaded once, in full, and left alone. The one exception is a
`displayDims` change: because `position` and `normal` are both `displayDims`-derived (§6.1, §3.4), it
re-extracts and re-uploads the `position` buffer and re-decides the `normal` attribute (§7). This is
re-extraction of the display-space projection, **not** compaction — compaction is still never done.
A reveal-ladder level is a second exception: each level writes its grown vertex/face prefix into the SAME
capacity-sized buffers (sized to the ladder's lifetime totals, never resized), rather than uploading
"once, in full" the way an unladdered mesh does.
`drawElements` never fetches an unreferenced vertex, so culled vertices cost nothing to draw, and the
mesh is resident in full anyway (§7). This deliberately avoids:

- a generic mask-compaction helper over the vertex buffers, which would be **`&[f32]`-only** and could
  not compact the native `uint8`/`uint16` colors §3.2 permits without a widening pass (the `uint8`/`float16`
  scalars are uploaded as `f32` on the attribute path anyway, §6.1.1, but the colors stay native, only
  padded RGB→RGBA);
- a `vertex_remap` array and the index remapping that goes with it;
- re-uploading every attribute buffer on each slice change (on a slice change the index buffer alone is
  re-uploaded; a `displayDims` change additionally re-uploads `position`/`normal`, §7).

The only cost is VRAM for vertices that are currently invisible — bounded by the mesh size, which is
already the resident working set.

**Winding.** `compact_visible_faces` preserves the authored order, so it is winding-agnostic. Parity is
decidable only against the authored winding frame (§3.2): when the displayed set equals that frame and
the `displayDims` (x,y,z) column order is an **odd permutation** of it — a reflection of display
space — a post-pass swaps two of each triangle's three indices to restore front-facing winding. This is
keyed to the *current* `displayDims` parity, so it runs on **every** index build in an odd-parity epoch
(initial load, slice move, and `displayDims` change alike), not only at the moment `displayDims`
changes. Equivalently, render the opposite material `side` for the duration of the odd-parity epoch — a
persistent form that needs no per-rebuild post-pass. This equivalence holds only while nothing consumes
`gl_FrontFacing`: the stored-normal shading flip (§6.2) requires the index post-pass form, since the
opposite side of a `DoubleSide` mesh is `DoubleSide` and leaves projected winding (and thus
`gl_FrontFacing`) reversed. When the displayed set is a **different triple**
than the frame (e.g. `[0,1,2]` → `[1,2,3]`), or an nD mesh declares no frame (no stored normals),
projected orientation varies per triangle and no index post-pass can fix it — the viewer renders
`DoubleSide` for that epoch regardless of `double_sided: false`, and logs a one-time notice naming the
node.

`mesh_vertex_visibility_mask` calls `validate_ndim` like its siblings and therefore panics above 16D;
`pickBackend(ctx, ndim)` (`workers/data-worker/state.ts:60`) returns `ctx.tsFallback` — the **whole
module** — for `ndim > 16`. No change to the routing logic is needed, but because it swaps modules
wholesale, both new kernels must be declared on the `WasmModule` interface (`wasm/types.ts:7`) and
implemented by **both** backends, or the TS module will not structurally satisfy the interface.

### 5.5 Fast path

When `displayDims.length === ndim` (no hidden dimensions — the common 3D case), the mask is trivially
all-ones and the *cull* is skipped: with `displayDims` unchanged, positions are extracted once via the
existing `extract_3d_positions` and the index buffer is uploaded verbatim (no compaction) — after the §5.4 parity
post-pass, which reverses the winding if the initial `displayDims` is odd-parity (nothing restricts the
opening/restored view to ascending order). Skipping the cull is **not**
the same as doing no work, though: a `displayDims` change on this fast path still re-extracts positions
via `extract_3d_positions`, re-decides the `normal` attribute per §3.4, recomputes bounds, and — for an
odd-parity permutation — reverses the index winding (§7); only the visibility-mask recompute is elided.
A non-3D dataset triggers the full cull path.

Combined with §5.4's no-compaction rule, this means a plain 3D mesh uploads every buffer once per
`displayDims` epoch (once until `displayDims` changes), and only a pure slicePosition/tolerance move
with unchanged `displayDims` is truly zero-work — `updateView` returns early. A `displayDims` change is
never zero-work, even here (it rebuilds `position`/`normal`).

---

## 6. Rendering

### 6.1 Geometry

`rendering/mesh-geometry.ts` builds a plain `THREE.BufferGeometry`:

| Attribute | Size | Source |
|---|---|---|
| `position` | 3 | `extract_3d_positions(vertices, displayDims)` |
| `normal` | 3 | stored normals when valid (§3.4), else omitted |
| `color` | 3 or 4 | **always bound** (never left to the GL default `(0,0,0,1)` black): when `colors` present → from `colors`, native dtype kept, with `uint8`/`uint16` RGB **padded to RGBA** (opaque alpha) so it binds as a valid vertex format on the WebGPU backend (§6.1.1); when `colors` absent → filled opaque white `(1,1,1)` as a size-3 `float32` attribute (`float32x3` is valid on both backends, so `vAlpha == 1.0` comes via the size-3 `w = 1.0` default, §6.1.1), mirroring `create-points-node.ts:91`. A 4th component is a per-vertex opacity carried through an interpolated `vAlpha` (§6.2) |
| `aScalar` | 1 | `scalars`, when `has_scalars`; `uint8`/`float16` uploaded as `float32` on the attribute path (§6.1.1) |
| index | — | `compact_visible_faces` output — **the only buffer rewritten on a slice change** (§5.4); `position` (and `normal`) are additionally rewritten on a `displayDims` change (§7, §3.4) |

Drawn as `THREE.Mesh` with `side: DoubleSide` when `double_sided`, else `FrontSide`.

#### 6.1.1 Color/scalar vertex-attribute dtypes

Bind `uint8`/`uint16` colors with `new THREE.BufferAttribute(u8, 4, /* normalized */ true)`, keeping the
native dtype (the GPU normalizes to `[0,1]` for free) rather than widening to `Float32Array` — but always
at **4 components**. When `colors` is RGB (§3.2 permits `(V, 3)`), pad it to RGBA at geometry-assembly
time with a fully-opaque alpha (`255` for `uint8`, `65535` for `uint16`, both normalizing to `1.0`); RGBA
input binds as-is. This is load-bearing, not cosmetic: the TSL materials run on a real `WebGPURenderer`
(`?renderer=webgpu` negotiates an adapter and passes the device; `forceWebGL` is a diagnostic mode only),
and three r184's `WebGPURenderer` exposes **no 3-component 8/16-bit vertex format** for that family — its
`GPUVertexFormat` table lists only `unorm8x2`/`unorm8x4` and `unorm16x2`/`unorm16x4` — and WebGPU requires
`arrayStride` to be a multiple of 4. A tightly-packed size-3 `uint8`/`uint16` attribute has a 3-byte /
6-byte stride (three r184 uploads it packed and sets
`arrayStride = itemSize · BYTES_PER_ELEMENT`); neither stride is a multiple of 4, so `createRenderPipeline`
fails validation and an RGB `uint8`/`uint16` mesh renders **nothing** on the WebGPU backend — violating
§6.4's "both backends must produce matching output". Padding to `unorm8x4` (4-byte stride) / `unorm16x4`
(8-byte stride) fixes both the format and the stride, and the memory win over float32 widening survives:
4 bytes/vertex for `uint8` RGBA vs 12 for `f32×3`. `float32` colors need no padding — `float32x3` is a
valid WebGPU format with a 12-byte (4-multiple) stride — so they bind at their native `3` or `4`
components. This still mirrors the loader doctrine — `LoadedLinesData` and `LoadedPointsData` keep
`Uint8Array | Uint16Array | Float32Array` colors — and §5.4 removed the one place mesh would have needed a
full `f32` widen of the color array (the compaction pass); the RGBA pad is a 3→4 component copy in the
native dtype, not a dtype widen.

`uint8`/`float16` **scalars** are uploaded as **`float32`** on the attribute path for the same root cause:
three r184 has no itemSize-1 vertex format for a `Uint8Array` or a native `Float16Array`
(`typeArraysToVertexFormatPrefixForItemSize1` maps neither → "Vertex format not supported yet" and a
broken pipeline). The partial escape hatches r184 does ship rescue nothing here. A **non-normalized**
`Uint8Array` is caught by the buffer-side "patch for INT16 and UINT16" widen, which rebuilds the whole
buffer as `Uint32Array` and binds an **integer** `uint32` attribute — four bytes per scalar, the exact
memory of the `float32` upload, but integer-typed in the shader graph (`getTypeFromArray` → `uint`)
instead of the `float` the material reads; a **normalized** one skips that widen and dies on the format
lookup miss, with a 1-byte stride that violates WebGPU's multiple-of-4 rule anyway. Three's own
`Float16BufferAttribute` (which is `Uint16Array`-backed) fails just as loudly, only more confusingly:
`getTypeFromAttribute` special-cases it to a `float` shader input while the attribute path binds a
`uint32` vertex format, so `createRenderPipeline` rejects the input/format class mismatch. Widening to
`float32` (4-byte stride, valid on both backends, `float`-typed everywhere) sidesteps the whole minefield
at zero memory cost over the only native path that even binds. `float32` scalars bind directly as
`float32`. Mesh is the first geometry type to feed these dtypes to *vertex attributes* — the siblings
route colors/scalars through the RGBA32F element texture — which is why nothing in the shipped tree has
hit this before.

There is **one** `color` attribute regardless of the source component count, and hence **one** shader that
reads it as a `vec4`. RGB input still carries a per-vertex opacity of `1.0` for free — the same *"1.0 for
RGB data"* contract the gsplat/line shaders document (`materials/gsplat/shader-glsl.ts`,
`materials/line/shader-glsl.ts`) — but mesh obtains that `1.0` two ways depending on the format: for
`uint8`/`uint16` RGB it is the **CPU-side pad alpha** (exactly as the siblings write `1.0` into their
element texels), and only for a format WebGPU already accepts (`float32x3` — the absent-colors white fill
in §6.1, and any `float32` RGB) is it left to the size-3 `w = 1.0` attribute default. That default holds
on **both** backends, but *only for a valid format*; it cannot rescue the invalid `uint8`/`uint16` size-3
layout, which is exactly why those are padded rather than left size-3. No separate RGB-vs-RGBA material
variant is needed; the alpha handling in §6.2 is unconditional.

### 6.2 Shading

The other three geometry types are purely emissive and have no lighting whatsoever. A mesh without
shading is a flat silhouette and effectively unreadable, so mesh is the first type to shade. The v1
model is deliberately minimal and light-free:

> ⚠️ **Two refinements the implementation forced, both now shipped.** (a) The derivative fallback's
> `normalize(cross(dFdx(vViewPos), dFdy(vViewPos)))` is **explicitly forced** to `z >= 0` in both
> backends. The cross product carries the sign of the fragment-space y axis, and GLSL's `dFdy` is
> bottom-up while WGSL's `dpdy` is top-down — so "orientation-defined by the rasterized fragment, so it
> always faces the viewer" is only true once enforced. That an unenforced flat variant would collapse to
> `uAmbient` on WebGPU alone is a spec-derived RISK, not an observed behaviour: the real-WebGPU A/B (§11
> row 6) MEASURED a face-on flat quad rendering identically with the flip REMOVED, so on Chrome + Apple
> Silicon the two conventions coincide and the flip is currently inert there. It is kept as insurance —
> one instruction, correct under either convention, and neither shading-language spec promises the two
> conventions agree. (b) The stored-normal view transform is written out as
> `viewMatrix · (modelNormalMatrix · n)` rather than through three's
> `transformNormalToView`, whose `transformDirection` **normalizes**: since the writer accepts
> zero-length normals with a warning (§3.5), that normalize would produce `NaN` and interpolate it
> across every triangle touching the vertex, flat-shading all of them instead of distorting locally.
> Also: because the `normal` attribute must not be bound/unbound per epoch (that grows or shrinks a live
> geometry's WebGPU vertex layout), it is bound for the node's lifetime whenever `has_normals`; only the
> shader variant swaps.

- **Normal source:** the stored `normal` attribute is used **iff `shading == "smooth"` and the stored
  normals are valid for the active view** (present and `normal_dims == displayDims`, §3.4); otherwise —
  an explicit `shading == "flat"`, absent normals, or a `normal_dims`/`displayDims` mismatch — a flat
  normal is derived in the fragment shader from screen-space derivatives of the view position
  (`normalize(cross(dFdx(vViewPos), dFdy(vViewPos)))`). The derivative fallback means a mesh with no
  stored normals still shades correctly, and it is what makes §3.4's "recompute on non-default
  displayDims" cheap: when `normal_dims` no longer matches `displayDims`, the mesh drops the stored
  normals and shades from derivatives. Because a declared-but-unbound `normal` reads `(0,0,0,1)` rather
  than "absent", this stored↔flat choice is a **compile-time shader variant** (§6.4's `mesh.fragment` vs
  `mesh-flat-normal.fragment`), selected identically by both the GLSL and TSL backends — so `shading`
  drives the variant, it is not inert metadata. The selection is computed **once** per node (in
  `createMeshNode`, from `shading`, the stored normals' presence and `normal_dims`, and the active
  `displayDims`) and handed to both material factories, so
  the two backends never re-derive it independently. Its two conjuncts differ in stability: `shading` is
  view-independent, so a `shading == "flat"` node is statically the flat variant and never swaps; the
  `normal_dims == displayDims` conjunct is view-dependent, so a `shading == "smooth"` node re-evaluates the
  rule — swapping variant and binding/omitting the `normal` attribute — when a `displayDims` change flips
  its validity, the same event that re-extracts the display-space `position` (§6.1, the §7
  `displayDims`-change rebuild).
- **View-space normals (stored-normal variants).** The `normal` attribute binds in the node's local
  display frame, but every other input to the shade term is **view-space** by construction — `vViewPos`,
  the fixed view axis `V`, and the derivative fallback's `cross(dFdx(vViewPos), dFdy(vViewPos))`. The
  stored-normal vertex stage must therefore carry the normal into view space before interpolation:
  `vNormal = normalMatrix * normal` in GLSL (the built-in `mat3 normalMatrix`, the inverse-transpose of
  the model-view matrix), and the TSL twin via `transformNormalToView`. The inverse-transpose is
  load-bearing, not pedantry: a mesh node carries the standard 4×4 `transform` attr (§3.3), and
  anisotropic scaling is routine in this domain (voxel size z ≠ xy), under which the plain model-view
  linear map skews normals off-perpendicular — while a raw untransformed normal mislights any *rotated*
  node (the same mesh shades correctly under the flat variant and wrongly under the smooth one, since
  only the latter reads the attribute). The fragment-stage renormalization (two-sided bullet below)
  absorbs the length change `normalMatrix` introduces under scaling, so no vertex-stage normalize is
  needed. The TSL backend's generated `.vertex` codegen snapshots (§6.4) pin this transform for that
  backend; the hand-written GLSL twin has no codegen snapshot and is instead pinned by the §8
  rotated/anisotropically-scaled shading test.
- **Shade term:** a camera-anchored offset key light,
  `L = normalize(vec3(-0.35, 0.55, 0.75))`, above and slightly left of the view axis. It remains
  light-free in the scene sense: `L` is a view-space constant that rides with the camera, so there is no
  light object, scene-graph state, or per-frame light uniform. The diffuse term keeps the existing wrap,
  `shade = mix(uAmbient, 1.0, pow(saturate(dot(N, L) * 0.5 + 0.5), uShadeExponent))`, preserving the
  readable ambient floor on away-facing surfaces. A subtle additive Blinn–Phong highlight supplies a
  second curvature cue: with `V = vec3(0, 0, 1)`, the half-vector is also constant,
  `H = normalize(L + V)`, and
  `spec = uSpecular * pow(max(dot(N, H), 0.0), uShininess)`. The final lit RGB is
  `finalColor * shade + vec3(spec)`. Defaults are `uAmbient = 0.25`, `uShadeExponent = 1.5`,
  `uSpecular = 0.12`, and `uShininess = 24`; `uAmbient = 1.0` removes the diffuse gradient and
  `uSpecular = 0.0` disables the highlight.
- **Two-sided normal (stored-normal variants).** Every non-flat fragment build — `mesh.fragment` and its
  `mesh-additive`/`mesh-max`/`mesh-colormap` siblings (§6.4) — renormalizes the interpolated normal in
  the fragment stage and then flips it to face the camera BEFORE the lighting term:
  `N = gl_FrontFacing ? N : -N` in GLSL, and the TSL twin via the `frontFacing` node. Without it, a
  back-facing fragment keeps an unflipped normal, so the wrap term typically lands on the dimmer side
  of its range and the back side shades with an inverted gradient collapsing toward `uAmbient` (dark at the
  head-on interior, rising to a mid value at the silhouette) instead of the front-facing gradient —
  visible immediately because `double_sided` defaults **true** (§3.3) and §5's whole-triangle cull
  exposes the interior back faces of a sliced closed isosurface, exactly the target data. The
  **derivative fallback** (`mesh-flat-normal.fragment`) needs **no** such flip:
  `normalize(cross(dFdx(vViewPos), dFdy(vViewPos)))` is orientation-defined by the rasterized fragment,
  not the winding, so it always faces the viewer (§7's "always faces the camera regardless of winding" is
  correct for that variant) — the flip is a stored-normal-variant-only concern.

  That exemption is **per fragment, not per variant**, which matters because §3.5's epsilon guard
  substitutes the same derivative normal *inside* these stored-normal builds whenever the interpolated
  normal is not affirmatively valid. Such a fragment must take the fallback's rule, not the variant's:
  the substituted normal already faces the viewer, so applying `gl_FrontFacing ? N : -N` to it would
  negate a viewer-facing normal on every back-facing fragment and reintroduce exactly the inverted,
  `uAmbient`-collapsing shade the flip exists to remove — worst precisely at the degenerate and corrupt
  vertices the guard is there to rescue. So the flip is gated on whether the stored normal survived the
  guard; only normals that did are flipped. One structural constraint on the fragment build follows: the
  guard's condition reads an interpolated varying, so a branch on it is **non-uniform control flow**,
  where GLSL leaves `dFdx`/`dFdy` undefined (normal validity can differ between fragments of the same
  2×2 quad). The derivative fallback normal is therefore computed **unconditionally**, before any
  guard-dependent branching, and the guard *selects* per fragment between the flipped stored normal and
  that precomputed fallback — only the `gl_FrontFacing` flip, never the derivative evaluation, sits
  behind the guard.

  The flip is well-defined wherever it applies: stored normals are only active when
  `normal_dims == displayDims` (§3.4), where
  §5.4's parity post-pass keeps winding coherent — specifically its **index post-pass** form, since the
  flip consumes `gl_FrontFacing` and needs it to correlate with the authored orientation — so the flip
  puts the back face on the lit side of its own view-anchored gradient instead of the inverted,
  `uAmbient`-collapsing side.
- **Base color:** the colormap LUT applied to `aScalar` under `USE_COLORMAP`, else the vertex `color`
  attribute — which is opaque white when `colors` is absent (§6.1, filled CPU-side exactly as
  `create-points-node.ts:91` does for points). So the minimal `add_mesh(vertices, faces)` call (no
  colors, no scalars) renders a readable opaque-white surface, not the GL-default black that an unbound
  `color` attribute would give. The colormap path is the same `getColormapTexture` / `updateScalarRange`
  path `createLinesNode` uses, including the same fail-closed guard when `colormap` is set without
  `has_scalars`. This default is identical in both the GLSL and TSL backends (§6.4).
- **Per-vertex alpha (load-bearing).** The `color` attribute's 4th component is a per-vertex opacity and
  is carried through a **smoothly-interpolated** varying `vAlpha` — the vertex stage writes
  `vAlpha = sanitizeAlpha(color.a)` and the rasterizer interpolates it across the triangle. Contrast the
  gsplat shader, whose per-splat data is a per-instance constant and is therefore `flat`-qualified
  (`flat out mediump vec3 vColor;`, `flat out mediump float vAlpha;` in
  `materials/gsplat/shader-glsl.ts`, whose comment states "alpha is load-bearing in EVERY mode"): a mesh
  vertex is **not** an instance constant, so its alpha must interpolate across the face exactly as the
  line shader's `vAlpha` interpolates along a segment
  (`vAlpha = mix(sanitizeAlpha(lineT5.z), sanitizeAlpha(lineT5.w), tEff);` in
  `materials/line/shader-glsl.ts`). RGB data supplies `vAlpha = 1.0` for free (§6.1.1), so no
  RGB-vs-RGBA variant is needed. This rule holds identically in **both** the GLSL and TSL material
  backends (§6.4).
- **Tail.** The shaded RGB then goes through the standard `intensity` → `offset` → `gamma` chain, after
  which the coverage that drives blending is formed and the fragment is emitted per blending mode:

  Mesh has **no** per-element `intensity`/amplitude/falloff scalar (§2.2) — it is a solid shaded
  surface — so the coverage entering the blend is simply the per-vertex alpha times node opacity:

  ```glsl
  float a = vAlpha * uOpacity;   // the single coverage term; NOT intensity * uOpacity
  ```

  The §6.2 diffuse **shade** factor and additive specular are lighting terms applied to RGB only; neither
  enters `a`. Emission then branches on the blending mode's `shaderOutputMode` (`blending-state.ts`),
  mirroring the line shader's fragment tail (`materials/line/shader-glsl.ts`):

  - `additive` / `luminous` / `normal` → **alpha-weighted** (`SrcAlpha/One` or `SrcAlpha/OneMinusSrcAlpha`
    apply `a` at composite): emit `fragColor = vec4(shadedColor, a);`.
  - `max` → **rgb-contribution**: `MaxEquation + OneFactor/OneFactor` does **not** weight source RGB by
    alpha at composite, so premultiply by coverage — emit `fragColor = vec4(shadedColor * a, a);` — exactly
    the line shader's `LUXAR_MAX_RGB_CONTRIBUTION` branch (`vec4(gammaColor * a, a)`).
  - `opaque` (mesh default) → a hard alpha **cutout**, see below.

  For `normal`, per-triangle depth sorting orders the triangles back-to-front (§6.3), so per-vertex
  alpha composites in a sane order rather than in authoring order. It is an approximation, not an
  exact solve — see §6.3 for the two ways a centroid key falls short — and that is the reason
  `opaque`, which is depth-correct per fragment, remains the mesh default.

  The max-premultiply and the opaque-cutout emissions are distinct per-mode shader variants — a GLSL
  `#define` exactly like the siblings' `LUXAR_MAX_RGB_CONTRIBUTION` branch (and a graph-baked TSL twin) —
  so each is a separately compiled shader that carries its **own** codegen snapshot (§6.4), not one free
  runtime branch. (Compile-time `#define` and runtime-uniform branches coexist in the shipped materials —
  e.g. gsplat's opaque/peak split is a runtime `uProjectionMode` branch that TSL bakes per graph — but
  either way the harness snapshots each mode separately, which is the point here.)
- **`opaque` (the mesh default) → alpha is a hard cutout, not smooth transparency.** Decision, stated
  rather than left silent: `opaque` is depth-writing and order-independent (`shaderOutputMode: 'opaque'`,
  `blending-state.ts`), which is precisely why it is the only mode correct with no sorting at all —
  including when depth sorting is switched off (§6.3) — and smooth partial transparency is
  contradictory there. So under `opaque`
  the coverage `a = vAlpha · uOpacity` acts as a **hard, order-independent cutout**:

  ```glsl
  float a = vAlpha * uOpacity;
  if (a < uAlphaCutoff) discard;   // masks / holes; order-independent
  fragColor = vec4(shadedColor, 1.0);   // survivors are fully opaque, depth written normally
  ```

  `uAlphaCutoff` is a material uniform with a sane default (`0.5`). This keeps `opaque` correct without
  sorting while giving authored alpha a defined, useful meaning (masks, holes, alpha-tested detail); the
  pick pass applies the **same** cutout so holes are neither pickable nor depth-occluding (§6.5).

  Stated plainly: because node `opacity` is folded into the cutoff, `opacity` does **not** dim a default
  (`opaque`) mesh — it sweeps the cutout threshold. On an RGB mesh (`vAlpha ≡ 1`) that is a hard **step**:
  `opacity < uAlphaCutoff` dissolves the whole surface at once and any `opacity` above it produces no
  change. With authored per-vertex RGBA alpha it instead **erodes** — as `opacity` drops, more vertices
  fall below the cutoff and the surface eats away — never a uniform fade. Either way, animating opacity on
  a default mesh does not cross-fade; a user who wants a *smooth* opacity fade selects `normal` instead
  (which is depth sorted, §6.3).

  ⚠️ **This chain is *not* currently shared.** `materials/_shared/` provides only sanitizers,
  near-fade, sorted-index addressing (`glsl-lib.ts`, `tsl-helpers.ts`) and the
  `clampGamma`/`isGammaOne`/`isNoGOG` **defines** helpers (`uniform-helpers.ts`). The
  `vColor * uIntensity + uOffset` / gamma / blend chain itself is written out independently in each of
  the six `materials/{point,line,gsplat}/shader-{glsl,tsl}.ts` files.

  Mesh therefore has two options: (a) copy the chain a fourth time, consistent with the current
  codebase, or (b) lift it into `_shared` first as a prerequisite refactor. **(a) is proposed for v1** —
  (b) touches all six existing shader files and their codegen snapshots, and mixing that into the mesh
  PR would make the diff unreviewable. Lifting the tail is worth a separate follow-up once there are
  four copies to justify it.

- **Perspective near fade — the shared `perspectiveNearFade`, evaluated PER FRAGMENT.** The other
  three types already suppress geometry approaching the near plane with the one helper in
  `materials/_shared/glsl-lib.ts` / `tsl-helpers.ts` (0 behind the camera, `smoothstep` across
  `[nearCull, 2·nearCull]`, 1.0 under ortho), in both their visual and their pick shaders. Mesh
  does the same, so a surface fades out as the camera flies into it instead of clipping hard —
  the last near-plane asymmetry between the four types.

  The **stage** differs from the sibling types and is forced, not chosen. Points and gsplats
  evaluate the fade per VERTEX, which is exact because an instanced quad has one center depth. A
  triangle spans depth, so a per-vertex value would interpolate the *ramp* across the face and a
  large triangle straddling the band would smear a linear gradient over the smoothstep. Mesh
  therefore evaluates it in the fragment stage, off the `vViewPos` varying the shade term already
  carries:

  ```glsl
  float nearFade = perspectiveNearFade(uIsOrtho, vViewPos.z, max(uNearCull, 1e-20));
  if (nearFade < 0.01) discard;   // every mode — see below
  ```

  Three consequences worth stating:

  - The `< 0.01` reject applies in **every** blending mode, not just the translucent ones.
    A mesh may WRITE depth — `opaque` always does, and `normal` does at opacity ≥ 0.99
    (`rendering/blending-state.ts::normalModeDepthWrite`, which mesh feeds its real opacity) — and
    a fully-faded but still-rasterized fragment would then sit in the depth buffer occluding
    everything behind it while contributing nothing visible. Unconditional rather than gated on
    that predicate: gating would buy a runtime uniform in order to save a `discard`.
  - Under `opaque` the emission is `vec4(shadedColor, 1.0)` — there is no alpha to fade — so the
    fade ramps the **shaded RGB** instead: `vec4(shadedColor * nearFade, 1.0)`. Every other mode
    folds it into the coverage `a` before the emission branch, exactly as points and lines do,
    which is also how `max`'s RGB premultiply picks it up. The cutout comparison itself reads the
    **unfaded** coverage: the fade is a distance effect, not an authored mask, and letting it move
    the comparison would dissolve the holes open as the camera approached.

    So in `opaque` — the mesh default — the near fade **darkens rather than dissolves**, and that
    is an accepted trade rather than an oversight. The fragment keeps writing depth and keeps
    emitting alpha 1.0, so over a non-black background the near shell goes visibly BLACK for the
    width of the band before the `< 0.01` reject removes it; over a black background it reads as a
    dissolve. Every other mode dissolves properly, via coverage. Two things make the trade the
    right one. The band is `[nearCull, 2·nearCull]` with `nearCull = 1e-3 · diagonal`, so the
    darkened shell *sits* 0.1–0.2% of the scene diagonal in front of the eye and is ~0.1% thick — a
    distance a real approach crosses in a frame or two. And each alternative is worse in its own
    way. The only DETERMINISTIC way to dissolve — moving the cutout comparison onto the faded
    coverage (`a * nearFade < uAlphaCutoff`) — was rejected above for a stronger reason than the
    fade: it would let a distance effect rewrite an authored mask, opening the surface's holes as
    the camera closed in. A **stochastic** reject (`discard` when `nearFade < hash(gl_FragCoord.xy)`)
    would dissolve an order-independent depth-writing surface properly and without touching the
    mask; it is declined rather than overlooked, because it costs a hash plus a codegen variant and
    because a non-deterministic fragment would downgrade the parity harness's exact-factor lock
    (`mesh-near-fade` is pinned at precisely `0.15625 ×` its un-faded reference, per pixel) to a
    coverage-fraction test. Keeping only the `< 0.01` reject and dropping the RGB ramp is the fourth
    option, and is the hard near-plane clip this section exists to remove. An order-independent
    depth-writing mode drawn without per-triangle sorting (§6.3) has no honest partial coverage to
    fade; the answer for a user who wants a dissolve is `normal`. The viewer answers the same
    structural question identically elsewhere: `scene/lod-fade.ts`'s `BLENDABLE_MODES` is
    `additive`/`luminous`/`volumetric` only, so an `opaque`/`normal`/`max` layer keeps a hard LOD
    swap rather than a cross-fade, on the same premise that a mode with no linear opacity knob does
    not get a fake one.
  - `uIsOrtho` and `uNearCull` are the two camera inputs mesh consumes, which is why all four mesh
    material wrappers implement `CameraAwareMaterial` and join the material manager's camera
    broadcast — ignoring `fov` and `resolution`, which size a screen-space sprite a mesh does not
    have. Both are **runtime uniforms**, so an ortho-mode toggle is a uniform write and never
    recompiles a mesh program (the TSL graphs therefore take `perspectiveNearFadeTSL`, not the
    compile-time-ortho `…StaticTSL` variant the line graphs use).

  Under ORTHO the fade returns 1.0 for mesh exactly as for the other three, so nothing changes
  there; NDC near/far clipping stays the sole cull authority. The per-type stage table lives in
  `rendering/materials/_shared/README.md`, and the near-plane floor derivation that depends on it
  in `scene/scene-manager/clipping/bounds-math.ts`.

### 6.3 Blending and depth

v1 supports `opaque`, `normal`, `additive`, `luminous`, and `max`.

`volumetric` is **not supported**: it is an emission–absorption model over per-element optical depth and
has no meaning for an opaque surface. The observable behavior is a **one-time warning naming the node,
then an `opaque` fallback**, applied in `createMeshNode` — a warning rather than a load failure, for the
inheritance reason at the end of this section. An **explicitly-authored** per-node
`blending_mode='volumetric'` is additionally refused at authoring time (`add_mesh` and the
`Mesh.blending_mode` setter raise) — that case is never an inheritance, so the fail-fast rejection and
the viewer fallback coexist: the fallback remains the handler for a mode inherited from an ancestor and
for pre-existing stores.

`normal` on a mesh **is** per-triangle depth sorted. The registration, the SortWorker and the sort
kernel are shared with the other three types — a triangle's "center" is its vertex centroid, which is
3 floats per element exactly like a splat center or a segment midpoint. Only the APPLY differs, and it
differs structurally: the instanced types permute the `aSortedIndex` draw-slot indirection, while a
mesh has no indirection to permute and its ordering is written into `geometry.index` itself
(`rendering/depth-sort-coordinator/triangle-ordering.ts`).

Two consequences of that difference are worth stating, because they are not free choices:

- **The write is atomic, never chunked.** The instanced path streams a new ordering into an inactive
  twin attribute and flips a `uSortedIndexSlot` uniform on completion, so no frame ever samples a
  half-applied permutation. `geometry.index` is BOUND state, not sampled state — a shader cannot select
  between two index buffers, and reassigning `geometry.index` is the drawn-geometry rebind
  `applyMeshIndices` exists to avoid. So the whole visible prefix is written in one pass, because a
  partially-permuted index buffer is **not a permutation**: some triangles would be drawn twice and
  others not at all, a wrong picture rather than a stale one. The cost is bounded by a quantity the
  mesh path already pays — `applyMeshIndices` re-uploads the same prefix on every slice move.
- **The permutation is applied to the canonical triples, not to the live buffer.** The index buffer
  already holds the previous permutation, so permuting it again would compose the two. The coordinator
  retains the commit's `ProjectedMeshData.indices` for exactly as long as the node is being sorted.

**Sorting by centroid is an approximation, and it is worth being precise about how.** The kernel ranks
each element by the view-space z of ONE point, so:

- Two triangles that never intersect can still be ordered wrongly: they may overlap in screen space
  with one consistently in front across the shared region while their centroids — possibly both
  outside that region — rank the other way. This is a consequence of reducing a triangle to a single
  depth sample; an exact answer needs a per-fragment method (depth peeling, OIT) or a BSP split.
- **Interpenetrating** triangles have no correct order at all, since which one is in front changes
  across the shared region. No primitive-granularity sort can fix that with any key.

Both residuals are shared with the other three geometry types, whose quads sort by a single center for
the same reason. And both are why `opaque` stays the mesh default: it depth-tests and depth-writes, so
it is exactly correct per fragment whatever the index order is — and it stays correct when depth
sorting is switched off (`?depthSort=0`) or the SortWorker is unavailable.

Making `opaque` the mesh default is a deliberate asymmetry — it is the only mode that is unconditionally
correct without sorting, and it is what a surface should look like.

That asymmetry also makes Mesh the common trigger for cross-geometry depth hazards: a default-additive
Points/Lines/GSplats node ignores the depth written by an opaque mesh, while two overlapping
order-dependent nodes cannot be globally interleaved. The authoring rules and finalize-time diagnostics
are specified in `docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md` §1 under
"Overlapping-node authoring rule".

#### Where the default lives — and where it must NOT

⚠️ The mesh default must be applied **viewer-side only**, in `createMeshNode`:

```ts
blendingMode: (nodeAttrs.blending_mode as BlendingMode) ?? 'opaque',   // mesh
// cf. create-{points,lines,gsplats}-node.ts, all `?? 'additive'`
```

It must **not** be stamped by the writer. `apply_default_render_attrs` deliberately omits
`blending_mode` from the attrs it defaults, and says why: unlike the identity-valued compositing attrs
(`opacity`/`gamma`/`intensity`/`offset`/`absorption`, all no-ops under hierarchical composition), a
stamped `blending_mode` would **override an ancestor's** setting under the viewer's nearest-setter-wins
rule. Writing `blending_mode="opaque"` into every mesh node would silently break
`group(blending_mode="additive")` for its mesh children.

So: mesh joins the other three in *not* stamping the attr, and diverges only in the viewer-side `??`
fallback. Ancestor inheritance is preserved exactly.

⚠️ The `??` only fires if **composition preserves the unset state**. `nodeAttrs` here is the composed
record (`applyEffectiveAttrs` → `composeAttrs`, `data/attrs-composer.ts`), and
`normalizeBlendingMode(undefined)` returns `'additive'` — so composing unconditionally through it hands
`createMeshNode` an `'additive'` indistinguishable from an authored one and the `?? 'opaque'` becomes
dead code (it was, until the first end-to-end render caught it — §11 row 6). `composeAttrs` therefore
normalizes only a value some level actually set, and leaves a fully-unset chain `undefined`.

`volumetric` handling belongs in the same place — `createMeshNode`, warn once and fall back to `opaque`
(§6.3 above) — **not** in the writer, for the same reason: the mode may be inherited from an ancestor the
mesh node knows nothing about at write time.

### 6.4 Materials

Per the repo's dual-backend rule, mesh ships **two** visual materials and **two** picking materials,
registered in the existing tables at `rendering/material-manager/factories.ts`:

```ts
export const VISUAL_FACTORIES = {
  point:  { glsl: PointMaterial,  tsl: PointTSLMaterial  },
  line:   { glsl: LineMaterial,   tsl: LineTSLMaterial   },
  gsplat: { glsl: GSplatMaterial, tsl: GSplatTSLMaterial },
  mesh:   { glsl: MeshMaterial,   tsl: MeshTSLMaterial   },   // NEW
} as const;
// PICKING_FACTORIES gains the matching mesh entry.
```

with `MeshMaterialProperties` alongside the existing three property interfaces. Materials are **per
node** (the convention for all three types since the texture-storage migration), so there is no cache
key and no LRU.

There are no per-type material cache maps — every material is per-node, so `getCacheStats()` reports
only registry size and create-time, never a cache size. Mesh follows the same convention: no cache map,
`createMeshMaterial` constructs directly.

Both backends must produce matching output; the existing codegen snapshot harness
(`src/tests/__codegen__/`) gates the **TSL-generated** shaders (the hand-written GLSL twins are pinned
by behavior tests instead — e.g. §8's stored-normal view-space-transform check), and keys one snapshot variant per blend-mode build — whether a GLSL
`#define` (the sibling `line-max`, `point-max`, `gsplat-normal-premult`) or a runtime-uniform branch the
TSL path bakes per graph (`gsplat-opaque`, from gsplat's runtime `uProjectionMode` split). Note the
harness (`tsl-codegen-snapshot.spec.ts`) asserts **both stages** of every variant unconditionally, so
each variant is a `.vertex` + `.fragment` snapshot pair — the shipped inventory is 39 such pairs, i.e.
78 files under `src/tests/__codegen__/`. (This count drifts as OTHER types gain variants; mesh's own six
pairs are the part this section is responsible for. It read 37/74 until #2144 corrected it.)
Mesh's per-mode emissions (§6.2) are therefore separately snapshotted — and note the mesh **default is
`opaque`**, unlike the siblings whose default is the alpha-weighted `additive`. New variants — six, i.e.
twelve snapshot files: `mesh` (the `opaque` default — alpha cutout, §6.2), `mesh-additive` (the
alpha-weighted emission shared by `additive`/`luminous`/`normal`, §6.2), `mesh-max` (the max
premultiply, §6.2), `mesh-flat-normal`, `mesh-colormap`, `mesh-pick`.

> **TSL house rule** (from the depth-sorting spec's remediation): both vertex stages must trace inside
> `Fn()` with explicit `.toVar()` statements; the fragment must reconstruct the bottom-left
> fragcoord as `vec2(x, screenSize.y - y)` if it reads screen coordinates at all; and a value shared
> between two fragment entry points (`colorNode` and `depthNode`) must be ASSIGNED in an unconditional
> prologue that both of them call first. The third rule is the one the mesh pick shader learned the hard
> way (#1683): the order in which three builds the two entry points is not part of its API — it flipped
> from colour-first to depth-first between r184 and r185 — and a `.toVar()` is assigned wherever three
> first builds it, so a branching `depthNode` buried the shared coverage/fade/brightness chain inside one
> `if` arm and every top-level reader saw 0, discarding every fragment of the pick pass. The mesh
> fragment shaders sidestep the fragcoord trap (neither the visual nor the pick one needs fragcoord),
> but the pick fragment sits squarely in the third rule and follows it via `fragmentPrologue` in
> `picking/mesh/pick.tsl.ts`.

### 6.5 Picking

Standard mechanism: `pickingSystem.allocatePickId()`, a shadow `THREE.Mesh` sharing the same
`BufferGeometry` with the pick material, registered via `pickingSystem.registerNode` — exactly as
points / lines / gsplats do it.

v1 picks at **vertex granularity**. Mesh has no sorted-index indirection — its depth-sort ordering
permutes `geometry.index` itself (§6.3) — so unlike the other three types the mesh pick vertex shader
does **not** bind `aSortedIndex` and does **not** call the shared `luxarElementIdParts()` helper (which
reads it). That is not a gap the sort has to close: for an indexed draw `gl_VertexID` (WGSL
`@builtin(vertex_index)`) IS the value fetched from the index buffer, so the element ordinal is
**invariant under any permutation of the triples**. Depth sorting a mesh therefore cannot desynchronise
picking from rendering, which is the failure the instanced types' slot-syncing exists to prevent. Instead it splits `uint(gl_VertexID)` into low/high 16-bit halves exactly as that
helper does and writes them into the same `flat out highp vec2 vElementId` varying the readback already
understands:

```glsl
uint i = uint(gl_VertexID);
vElementId = vec2(float(i & 0xFFFFu), float(i >> 16u));  // split like luxarElementIdParts()
```

The mesh is drawn **indexed** (`faces` is the index buffer, §6.1), so under `drawElements`
`gl_VertexID` is the ordinal of the vertex in the `vertices` array — a stable per-vertex id, **not** a
triangle ordinal (`gl_VertexID / 3` would be meaningless: shared vertices break it, and WebGL2 has no
`gl_PrimitiveID`). This pick vertex shader is the one place mesh diverges from the shared helper; the
`mesh-pick.{vertex,fragment}` codegen snapshots of §6.4 cover it. The TSL twin (§6.4) must declare the
`vElementId` (and `vNodeId`) varyings with `.setInterpolation('flat')`: TSL `varying()` interpolates
linearly by default, and the point TSL pick (`rendering/picking/point/pick.tsl.ts`) only escapes without
`flat` because a single-instance quad's four corners all carry the same id (interpolation is the
identity). That identity fails for a mesh — `gl_VertexID` differs at every triangle corner, so a
linearly-interpolated `vElementId` would arrive fractional and `Math.round` in the readback would resolve
to arbitrary wrong vertices. Follow the LINE pick precedent (`rendering/picking/line/pick.tsl.ts`), not
the point pick, whose implicit-identity interpolation is unsafe for a shared-vertex indexed draw.

**Provoking-vertex convention (which corner wins).** A `flat` varying is sourced from **one** corner of
the triangle, and the two backends do not default to the same one: OpenGL ES 3.0 (the GLSL backend) fixes
the provoking vertex to the **last** vertex of the primitive, while the WGSL `@interpolate( flat )` three
emits for `.setInterpolation('flat')` defaults to `first`-vertex sampling. Left there, the same click on
the same triangle `(i0, i1, i2)` reports `i2` on WebGL and `i0` on WebGPU. The pick **contract** at
vertex granularity is therefore "a corner vertex of the front-most triangle under the cursor" — the
cursor is over the face, not a vertex, so every corner is an equally valid answer, and no consumer may
assume a specific one. To keep the backends bit-identical where the platform allows, the GLSL pick path
enables the `WEBGL_provoking_vertex` extension when present and sets
`provokingVertexWEBGL(FIRST_VERTEX_CONVENTION_WEBGL)`, aligning WebGL with WebGPU's first-vertex rule —
context-wide state, but safe: every other `flat` varying in the shipped materials is a per-**instance**
constant (point/gsplat quads, line segments), identical at all corners, so the convention flip is
observable only by mesh. Where the extension is unavailable the last-vs-first divergence stands as a
**documented exception** to §6.4's matching-output rule, and pick parity tests must assert the returned
id is *a corner of the expected face* (membership), not one exact corner. (Making the id
corner-independent outright would need per-corner pick data — a de-indexed shadow geometry, 3× pick
memory — abandoning the shared-`BufferGeometry` design above for the pick pass alone; rejected for v1.)

**Pick fragment output.** The `mesh-pick.fragment` writes the same shared vec4 the readback decodes —
`vec4(vNodeId, vElementId.x, brightness, vElementId.y)` — with `brightness` the fragment's
coverage/opacity (1.0 for a fully opaque mesh), so the cross-node brightness-weighted vote still has a
value. Depth is keyed on the blending mode, mirroring the gsplat pick wrapper's
`setSurfacePickDepth(isNormalMode(mode) || isOpaqueMode(mode))` sync (`rendering/picking/README.md`):
under the depth-ordered surface modes — `opaque` and `normal` — the mesh writes real projected depth
(`gl_FragDepth = gl_FragCoord.z`, i.e. leaves the default), matching gsplat's surface-mode branch
(`uSurfaceDepth == 1` in `rendering/picking/gsplat/shaders.ts`), so the front-most surface wins;
otherwise every fully-opaque fragment collapses to depth 0 and the mesh neither self-occludes nor
occludes other nodes correctly in the shared pick buffer. Under the commutative modes — `additive`,
`luminous`, `max` — it writes the `1.0 - clamp(brightness, 0, 1)` brightness-as-depth that points /
lines / commutative-mode gsplats use: real surface depth there would let a dim, barely-visible mesh in
front depth-occlude a brighter node behind it, contradicting the brightest-wins vote. See
`rendering/picking/README.md`; the `mesh-pick.fragment` codegen snapshot (§6.4) is the final authority.

**Alpha in the pick pass — the cutout must match.** The pick material computes the **same** coverage
`a = vAlpha · uOpacity` (§6.2), so its vertex shader binds the `color` attribute and carries an
interpolated `vAlpha` varying — the only vertex attribute it needs beyond `position` (element ids come
from the `gl_VertexID` built-in, not an attribute). In `opaque` mode it
applies the **identical** `if (a < uAlphaCutoff) discard;` before writing, so a cutout hole is neither
pickable nor depth-occluding; without this a discarded-in-visual hole would still rasterize in the pick
pass at true surface depth, becoming pickable **and** occluding picks of nodes visible through it. In the
translucent modes `a` is the `brightness` coverage term the readback already votes on (replacing the
"1.0 for a fully opaque mesh" placeholder above whenever alpha is authored). This is a **runtime-uniform
branch** in the single pick fragment (keyed on the blending mode, like the `uSurfaceDepth` split above),
**not** a separate `#define` — so `mesh-pick` stays a single snapshot variant and §6.4's count is
unchanged.

**The near fade must match too.** For the same reason the cutout must: the pick pass reproduces the
visual shader's `perspectiveNearFade` per fragment, off a `vViewZ` varying the pick vertex stage adds
(just the z — the visual stage's whole `vViewPos` exists to be differentiated for the flat-normal
fallback, which the pick pass has none of). It folds into `brightness` the way the point and gsplat
pick shaders fold theirs, so pick salience tracks visible salience, and a fragment below the same 0.01
threshold is discarded so it writes neither an id nor depth. Without it a surface the user can barely
see would stay fully pickable and keep depth-occluding the nodes behind it. This is what makes the
mesh pick materials `CameraAwareMaterial`s — `uIsOrtho` / `uNearCull` only; there is still no
`uResolution` and no focal length, since a mesh has no screen-space footprint to size.

**Stability.** §5.4 rewrites only the index buffer per slice (`compact_visible_faces`) and never remaps
vertex attributes ("No vertex compaction"). A face ordinal would be renumbered on every slice change; a
vertex ordinal is invariant across slices. That is why vertex — not face — granularity is chosen for the
compacted draw.

**Label mapping.** The returned vertex ordinal indexes the per-vertex `label_offsets` / `label_bytes`
CSR (§3.2) directly, precisely as a point/line element ordinal indexes its own per-element label CSR, so
hover tooltips resolve with no extra mapping.

**Capacity.** As with the other three types, the two-half 16-bit split keeps vertex counts exact past the
float32 24-bit mantissa (the readback recombines the halves); vertex count `V` uses the same split (see
`rendering/picking/README.md` for the rationale). The pick readback also packs
`nodeId * VOTE_KEY_STRIDE + elementId` for brightness-weighted voting, whose alias-free stride is `2^27`
(`rendering/picking/picking-system/pick-render.ts`). The §7 ≤-few-million-triangles figure is a workload
*expectation*, not an invariant: for the three texture-fed types the alias-free condition is
*structural* — `elementId` is bounded by `getMaxElementCapacityPerNode`, well under `2^27` — whereas
mesh's `gl_VertexID` source is bounded only by `V`. The cap is therefore *enforced*: `n_vertices <=
2^27` at the §3.5 loader gate's **metadata preflight** (Stage 1, before any chunk is fetched — fail the
node with a `LoaderError`; the largest admitted ordinal
`2^27 - 1` is the last alias-free one, and its worst-case vote key
`(2^24 - 1) * 2^27 + 2^27 - 1 = 2^51 - 1` stays exactly representable), invoking the stride's own house
rule that an unenforced bound is not a bound (the reason `MAX_PICK_NODE_ID` is checked at allocation,
`rendering/picking/picking-system/pick-render.ts`). Note that `pick-render.test.ts`'s existing headroom
test only pins the texture-*layout* maxima, so this vertex cap needs its own pin — a dedicated test
that the vote key stays exact up to the largest admitted vertex ordinal and that a mesh with
`n_vertices > 2^27` is rejected with a `LoaderError`. The writer **also** rejects `n_vertices > 2^27` at
write time (`ValidationError`, via `validate_vertices_for_writing`, §3.5) — the fail-fast twin of this
loader-side `LoaderError` pin.

**Accepted v1 limitation.** `vElementId` is a `flat` varying, so within a triangle it resolves to that
triangle's **provoking vertex**, not the cursor's barycentric-nearest vertex. Hovering a triangle
therefore reports a well-defined vertex *of* that triangle. Barycentric-nearest-vertex resolution would
need a de-indexed pick geometry or per-corner attributes plus barycentrics — a follow-up, not v1.

**FACE granularity** (e.g. highlighting a whole triangle) is deferred: it needs either a de-indexed pick
geometry or a per-corner face-id attribute, **plus** a compacted→original face map to stay stable under
§5.4 compaction. It is a natural follow-up, pairing with the §9 partition work.

---

## 7. Loading

v1 uses a **whole-node loader**, and the fetch is explicitly gated by §3.5's two-stage loader-side
validation. **First** the metadata preflight (§3.5 Stage 1) runs on the node attrs and every array's
`.zarray` shape, `chunks`, and dtype — the `n_vertices <= 2^27` cap, the `n_faces`/`MESH_DECODE_BUDGET_BYTES`
byte budget, the §3.2 shape cross-checks, and `normal_dims` well-formedness — **before any chunk is
fetched**, so an oversized or malformed declaration fails with a `LoaderError` without allocating.
**Then**, on a store that clears preflight, it fetches `vertices`, `faces` and the optional attribute
arrays in full, decodes, and holds them. No spatial index and no chunk-bounds query: one leaf loads
whole. The only progressive path is §9.1's reveal ladder, which is one such whole-node load per
`additive_<i>` level (`data/mesh/mesh-progressive-loader.ts`, drained by `data/mesh/lod-refinement.ts`)
rather than a partial load of any one of them. Immediately after decode the arrays get §3.5's Stage 2 post-decode value checks (materialized
lengths, face indices in `[0, V)`, CSR offsets), before anything reaches the §5.4 kernels.

Justification: meshes in this domain are typically ≤ a few million triangles and fit comfortably; the
dual-index machinery in `lines-spatial-index-loader.ts` (1010 LOC) exists because line datasets reach
tens of millions of vertices with a meaningful per-slice working set. A mesh's working set after a
`displayDims` change is the whole mesh regardless.

The loader still implements the standard `MeshDataLoader` interface (`loadMesh` / `updateView` /
`dispose` + the optional monitor surface), so a spatial-index implementation can be swapped in behind
it later with no caller change.

It DOES implement that monitor surface (`addEventListener` / `removeEventListener` / `getMetrics` /
`getActiveQueries`, reporting `type: 'mesh-whole-node'`), and the shape is load-bearing rather than
decorative: `connect-loader-to-monitor.ts` duck-types the complete set of four and silently skips a
loader that lacks any of them. What it reports is a whole-node loader's honest telemetry — `loads`,
`bytesLoaded` (decoded), `avgLoadTime`, `elementsLoaded` in TRIANGLES, and `memoryUsed` (payload plus
the per-node projection scratch) — with `queries` / `avgQueryTime` / `spatialIndex` deliberately at
zero/absent, since there is no index and a view change re-serves the resident mesh. `visibleElements`
is pushed IN by `commit-mesh-geometry.ts` (`recordVisibleElements`), because the count is produced
downstream of the loader: projection, not the loader, decides which faces reach the index buffer.

`updateView` distinguishes two kinds of view change:

- **slicePosition/tolerance change only** (`displayDims` unchanged): recompute the visibility mask and
  index buffer (§5) and return (the index build still applies the current-parity winding post-pass §5.4,
  so an odd-parity epoch stays correct across slice moves); on the fast path (§5.5) it is a no-op
  returning the cached data.
- **`displayDims` change**: because both `position` and `normal` are `displayDims`-derived (§6.1, §3.4),
  the rebuild runs these steps, in order:
  1. re-run `extract_3d_positions(vertices, displayDims)` and re-upload the `position` buffer;
  2. **recompute `geometry.boundingBox`/`boundingSphere`** — re-uploading `position` does *not*
     invalidate Three.js's cached bounds, which `frustumCulled` and the raycaster/picking broad phase
     consult, and the display-space AABB (§2.2) changes under a permutation. (A defect this path
     introduces; the per-slice path never touched bounds.)
  3. re-decide the `normal` attribute per §3.4 — attach stored normals iff `shading == "smooth"` and
     `normal_dims` equals the active `displayDims` element-wise (same length, same order — **not** a JS
     `===`, which compares array identity and would silently force flat shading on every rebuild), else
     omit so the shader's flat-normal fallback (§6.2) takes over.
     (For a `shading == "flat"` node the first conjunct is always false, so it stays the flat variant
     and this step never swaps it — §6.2.) Stored↔flat is a **compile-time
     shader variant** (§6.4's separate `mesh.fragment` / `mesh-flat-normal.fragment`), not mere attribute
     presence — a declared-but-unbound `normal` reads `(0,0,0,1)`, not "absent" — so flipping the choice
     must switch the material variant through each backend's existing variant path (§6.4), not by
     attaching/detaching the buffer alone;
  4. recompute the mask and index buffer; when the displayed set equals the winding frame (§3.2), the
     index build applies the current-`displayDims` winding post-pass (§5.4) — an odd-parity selection
     reflects display space, so without it a `double_sided: false` mesh (§6.1, `FrontSide`) renders
     **inside-out** (an open surface vanishes). When the new `displayDims` is a **different triple**
     than the frame, the material instead falls back to `DoubleSide` for the epoch (§5.4) — projected
     orientation is per-triangle data-dependent there and no post-pass can correct it. The §6.2
     derivative-normal fallback is unaffected either way — it always faces the camera regardless of
     winding.

  This holds **even on the §5.5 fast path**: a 3D axis permutation leaves the mask all-ones but still
  requires `position`/`normal` re-derivation, bounds recompute, and (for an odd permutation) winding
  reversal.

Files:

| Concern | File |
|---|---|
| Loader | `data/mesh/mesh-whole-node-loader.ts` |
| Node load | `data/scene-loader/nodes/load-mesh-node.ts` |
| Projection/process | `data/scene-loader/process/data-processor-mesh.ts` |
| Commit | `data/scene-loader/commit/commit-mesh-geometry.ts` |
| Types | `types/mesh.ts` |

Projection runs **in-process** in v1 (`workers/data-worker/projection/in-process.ts` gains a mesh
dispatch). Worker offload is a follow-up: the payload is a single large transfer rather than lines' many
small ones, so the worker's benefit profile is different and should be measured before being built —
consistent with the measure-first performance doctrine.

---

## 8. Integration checklist

Most of the *dispatch* plumbing is now table-driven (§10), so this list is dominated by genuinely
new mesh code rather than by edits to existing branches. The items that remain hand-written are
called out as such.

⚠️ This is the **planning** checklist, written before the phases ran. Every box is now ticked — the
Python vertical in #1220 (Phase 1), the cull kernels in #1232 (Phase 2), and the viewer loader,
materials, picking, fixtures and demo across Phases 3–6 — so read it as the original work breakdown
and as the record of *what* each subsystem had to change, not as current status. **§11 is the
authoritative delivery record.** Paths are as they stood when each phase landed; later refactors
have moved or removed a few of them (the `types/index.ts` barrel is gone, for instance), so treat a
name here as a pointer to the subsystem rather than to a live file.

**Python**

- [x] `format-contract/contract.yaml` → `make gen-contract` — the writable-side edit (`node_types` +
      `geometry_types`) **landed in #1220**; the Phase-3 `loader_types` switch-on re-ran it (#1241, §3.1)
- [x] `typing_utils/enums.py` — `NodeType.MESH` and `typing_utils/constants.py` — `NODE_TYPE_MESH`
      — **both landed in #1220.** They must match `contract.yaml`: `test_node_type_enum_matches_contract`
      guards the enum and `test_named_node_type_constants_match_contract` the `NODE_TYPE_*` constants —
      that pair is how the missing constant was caught in #1220
- [x] `core/mesh.py`, `core/group/adders/mesh.py` — **landed in #1220**
- [x] `core/group/group.py`, `core/__init__.py`, `luxar/__init__.py` — **landed in #1220**;
      `core/scene/scene.py` needed no edit (`Scene` subclasses `Group`, so it inherits `add_mesh`)
- [x] `core/group/dim_order.py` — **landed in #1220** as anticipated: `add_mesh` calls
      `apply_dim_order_positions` for `vertices` like the other three adders (no `dim_order.py` edit
      was needed); `faces` is index data and is **not** reordered
- [x] `io/_compiler/geometry_writers/mesh.py`, `io/_compiler/node_common.py` (`MESH_RESERVED_ATTRS`)
      — **landed in #1220**
- [x] `io/compiler.py` (`write_mesh` facade), `io/reader.py` (`MeshData`/`get_mesh`/`list_meshes`)
      — **landed in #1220**
- [x] `validation/base.py` (`validate_vertices_for_writing`, `validate_faces_for_writing`,
      `validate_normals_for_writing`) — **landed in #1220**
- [x] `cli/info_command.py` — **landed in #1220**
- [x] **Partition rejection** — **landed in #1220**, in a stronger form than this item planned:
      `add_partition_group_impl` asks the shared capability table (`supports_partition` in
      `typing_utils/geometry_capabilities.py`) instead of hardcoding a tuple.
      **Superseded:** the exclusion itself has since been lifted (§9.2) — mesh's
      `partition` capability is now `true`, so the guard passes for mesh and the
      test that pinned the refusal is now
      `test_mesh_under_a_mesh_partition_group_is_allowed`. The capability-table
      indirection is what made lifting it a one-row change.
- [x] **LOD rejection** — **landed in #1220.** The hole this item flagged was real:
      `compute_lod_display_type` simply returned `resolve_display_type(children[-1])`, which falls
      through to `node.attrs.get("type", "group")` for a plain leaf, so a mesh child would have been
      silently accepted into a `kind=lod` group with `display_type="mesh"` that no viewer path can
      load. It now calls `require_lod_display_type` (capability-driven, matching the partition guard),
      and the explicit `display_type=` route through `add_lod_group_impl` is gated the same way.
      **Superseded:** the exclusion itself has since been lifted (§9) — mesh's `lod`
      capability is now `true` for the SUBSTITUTIVE flavour, so the capability-driven guard
      passes for mesh and the test that pinned the refusal is now
      `test_mesh_under_a_lod_group_is_accepted`. An ADDITIVE ladder over an *arbitrary* order is
      still refused — not by this guard, but by `MESH_ADDITIVE_METHODS` naming `radial` as the
      only accepted method (a reveal ladder is accepted; see §9.1).
- [x] `io/_compiler/finalize/lod_backfill.py` — **already handled by #1079.** `resolve()` now tests
      `t in GEOMETRY_TYPES` instead of a hardcoded tuple, so `mesh` is recognised as a leaf the moment
      it enters the contract, with no edit here. All four child-iteration sites also moved to
      `group_keys()`, which removes the failure this spec previously described: a leaf group's
      `keys()` lists its *arrays* (`['faces', 'vertices']`), so the old code could recurse into a zarr
      `Array` and raise a bare `AttributeError` mid-finalize.

**TypeScript**

- [x] `types/mesh.ts`, `types/index.ts`, `types/window.d.ts`, `data/data-loader-types.ts`
- [x] `types/data-monitor-types.ts` — the local `| 'mesh'` extension is **already deleted** (#1220), so `SceneGraphNodeType` is now plain `NodeTypeName`
- [x] `data/scene-loader/loaders/loader-registry.ts` — one line in `LoaderByKind`. The three parallel
      maps became a single kind-keyed store in #1079; `getLoaderType` / `disposeAll` / the counters are
      one implementation each. Omitting the entry is a **compile error**
      (`TS2339: Property 'mesh' does not exist on type 'LoaderByKind'`), not a silent gap
- [x] `data/scene-loader/geometry-descriptors.ts` — one row in `GEOMETRY_DESCRIPTORS`, carrying
      `loadNode`, `applyPartialExtendTolerance`, `retryCommit` and the two loader factories. This is
      the row that `load-scene-nodes`, `lifecycle/retry` and `prefetch/slice-prefetcher` all read, so
      those three files need **no mesh edit at all**. A missing row fails the build with
      `TS2741: Property 'mesh' is missing … required in type 'Record<GeometryKind, …>'`
- [x] `data/scene-loader/loaders/loader-factory.ts`, `nodes/build-scene-graph.ts`
      (bare-leaf-root union), `nodes/build-ctx.ts` (the `processMeshData` / `commitMeshGeometry`
      pair, matching the uniform shape #1099 gave all three existing types)
- [x] `data/scene-loader/monitor/monitor-wiring.ts`, `scene-graph-converter.ts`,
      `data/scene-loader-monitor-port.ts`
- [x] `data/scene-loader/lifecycle/retry.ts` — **no edit needed.** The `else if (gsplatsLoader)` chain
      became a descriptor lookup in #1099. The hazard this spec flagged — a missing arm meaning a
      failed mesh load could never be retried, with nothing in the type system to say so — is now a
      build failure at the descriptor table instead
- [x] `data/scene-loader/prefetch/slice-prefetcher.ts` — **no dispatch edit**, but mesh has no
      meaningful slice prefetch in v1 (whole-node resident, §7). The prefetcher is driven by three
      hardcoded `prefetchNode(path, 'points'|'lines'|'gsplats', …)` call sites, so mesh is excluded by
      simply not adding a fourth — confirm that stays true rather than assuming it
- [x] `data/loaders/spatial-query/tolerance-computer.ts` — add the `mesh` arm to
      `computeHiddenDimTolerance` and `meshSlabTolerance` to `ToleranceOptions` (§5.2.1). Callers must
      pass `discreteRole: 'membership'`. **Do not** default the spatial arm to Lines' `0`
- [x] `wasm/types.ts` — declare both new kernels on the `WasmModule` interface (§5.4); `pickBackend`
      swaps the module wholesale, so a kernel missing from either backend breaks the `>16D` path
- [x] `data/attrs-composer.ts`, `data/stats/{aggregator,scene-stats}.ts`
- [x] `rendering/mesh-geometry.ts`, `rendering/node-factory.ts`,
      `rendering/node-factory/create-mesh-node.ts`
- [x] `rendering/materials/mesh/{material,shader}-{glsl,tsl}.ts`, `rendering/picking/mesh/*`
- [x] `rendering/material-manager/factories.ts`, `rendering/material-colormap-helpers.ts`
- [x] `scene/scene-manager/camera/camera-framing.ts` — its own arm, since a mesh contributes DRAWN
      triangles (`drawRange.count / 3`) rather than an `instanceCount`. `scene/lod-freshness.ts`
      needed no literal *while mesh was LOD-excluded*, since it discriminates on
      `supportsLod(nodeType)`. **Superseded:** the substitutive path ships (§9), so
      `GEOMETRY_CAPABILITIES.mesh.lod` is `true`, mesh levels ARE freshness-tracked, and
      `countFromUserData` carries its own `case 'mesh':` arm — without it the empty-level display
      guard silently no-ops for meshes. `scene/synthetic-scene.ts` is deliberately **not** extended —
      the `?debug` perf-bench injector still builds points/lines/gsplats only, mesh having no pooled
      instanced path to bench
- [x] `ui/layers/{layer-apply,layer-state,layers-panel}.ts`
- [x] `ui/data-loading-monitor.ts` + `data-loading-monitor/{templates,advisor}.ts`
- [x] `core/app/debug/{debug-interface,debug-state}.ts`

**Verified type-agnostic — no mesh change needed.** Swept for geometry-type literals and found clean,
so mesh rides these subsystems for free. Recorded so an implementer doesn't re-derive it:

| Subsystem | Evidence |
|---|---|
| `cache/` (slice cache, multi-level store, decompressed-chunk cache) | No geometry-type literals; keyed by path + chunk |
| `rendering/picking/picking-system.ts` core | No type literals; `registerNode(mesh, pickNode, pickId)` is generic (only the *material* is per type, §6.5) |
| `data/loaders/overlays/` | No type literals |
| `cli/export.py`, `cli/native_app.py` | No type literals; the offline/native bundlers copy the store wholesale |
| `encoding/` | `SemanticType.COORDINATE` and `.INDEX` already exist — **no new semantic type**. `COORDINATE` selects per-axis `linear_perchannel_u16`, which quantizes each normal axis over its own `[-1, 1]` range for a free 2× over float32, and correctly **blocks broadcasting** (a normal is always per-vertex) |
| `colormaps/` | Mesh reuses the scalar→LUT path unchanged (§6.2) |

**Deliberately NOT touched** — each of these enumerates `'points' | 'lines' | 'gsplats'` and must
**keep** doing so. Adding `mesh` to any of them silently re-enables something §9 excludes, with no
error to catch it:

| Site | Why mesh stays out |
|---|---|
| `types/lod-group.ts` — `display_type` union | Includes `'mesh'`; the SUBSTITUTIVE LOD path ships (§9) |
| `types/partition-group.ts` — `display_type` union | Includes `'mesh'`; the partition path ships (§9.2) |
| `rendering/gpu-buffer-pool/pool-stats.ts:22` — `type` union | Mesh doesn't use the buffer pool (§2.1) |
| `data/loaders/spatial-query/spatial-query-builder.ts` — chunk-query construction | No spatial index in v1 (§7). **Note:** this is the *query builder* only — `tolerance-computer.ts` in the same folder **does** need a mesh arm (§5.2.1); don't let the shared folder mislead you |
| `ui/layers/absorption-range.ts:90,98` | Mesh doesn't support `volumetric` blending — warn + `opaque` fallback (§6.3) |

A reviewer should treat a `| 'mesh'` appearing in any of the three remaining rows (pool stats,
spatial-query builder, absorption range) as a defect. The two `display_type` unions have since
been widened deliberately, each behind its capability flag.

**Rust / WASM**

- [x] `wasm/rust/src/mesh_culling.rs` + `lib.rs` registration
- [x] `wasm/typescript/mesh-culling.ts` + `index.ts` registration (also the production `>16D` backend)
- [x] Rust↔TS parity tests

**Docs**

- [x] `docs/guides/user/LUXAR_ZARR_FORMAT.md` — mesh node layout — **landed in #1220**
- [x] `CLAUDE.md` — geometry-types line flipped to "four first-class geometry types" once mesh
      rendered (Phase 3+); #1220 had named it writable-but-not-renderable
- [x] Package READMEs: `core/` and `io/_compiler/geometry_writers/` were updated in #1220; the viewer
      ones (`data/mesh/`, `rendering/materials/mesh/`, `rendering/picking/mesh/`) arrived with their
      packages
- [x] `CHANGELOG.md` — mesh entries landed with each phase (loader, material pair, picking + panel)

**Tests**

- [x] Python — **landed in #1220** (`core/tests/test_mesh.py`,
      `validation/tests/test_mesh_validation.py`): writer round-trip, validators (incl. every rejection in §3.5 — among them `n_vertices >
      2^27` rejected at write time with `ValidationError` via `validate_vertices_for_writing`, the
      fail-fast twin of the §6.5 loader-side `n_vertices > 2^27 → LoaderError` pin, verified to fail
      before the validator exists per this section's rule), authoring lint, broadcast color/scalar,
      `extend_to_all`, reader
- [x] Rust: `mesh_vertex_visibility_mask` / `compact_visible_faces` unit tests incl. the non-finite rule
- [x] TS unit: loader, geometry assembly, cull correctness, colormap fail-closed guard,
      Rust↔TS kernel parity, corrupt-store rejection (out-of-range face index → `LoaderError`, not a
      WASM trap — including a 64-bit index like `2^32 + 1` whose bare u32 cast would wrap into range,
      pinning §3.5 Stage 2's source-value check; an undersized `normals`/`colors`/`scalars` array →
      `LoaderError`, §3.5; `n_vertices >
      2^27` → `LoaderError`, its own pin since `pick-render.test.ts` only covers the texture-layout
      maxima, §6.5; an **oversized declaration** — `n_vertices > 2^27`, or an `n_faces`/`.zarray`
      shape-or-`chunks` footprint exceeding `MESH_DECODE_BUDGET_BYTES` — rejected by the §3.5 Stage 1
      metadata preflight, asserting **no chunk-key request** (an `<array>/<chunk-coords>` key) is ever
      issued — only the `.zarray`/`.zattrs` metadata keys the preflight legitimately reads — verified to
      fail before the preflight exists), and the `volumetric`→`opaque` fallback warning (§6.3)
- [x] TS unit (alpha chain, §6.2): an **RGBA** mesh produces **different** fragment output than the same
      mesh RGB-only (goes red if `vAlpha` is dropped — the exact "(V,4) renders like (V,3)" defect); under
      `opaque`, fragments with `a < uAlphaCutoff` are **discarded** (cutout) and survivors write alpha 1.0;
      under `max`, the emitted RGB is **premultiplied** by `a` (`vec4(shadedColor * a, a)`). Each must be
      verified to fail before the fix.
- [x] TS unit (default base color, §6.1/§6.2): a bare `add_mesh(vertices, faces)` mesh (no `colors`, no
      `scalars`) renders opaque **white**, not black — the `color` attribute is filled `(1,1,1)` rather
      than left unbound. Verified to go **red** if the white fill is dropped (an unbound `color` reads the
      GL default `(0,0,0,1)` and, times the multiplicative §6.2 shade term, the surface comes out solid
      black).
- [x] TS unit (native vertex-attr dtypes, §6.1.1): geometry assembly binds a `uint8`/`uint16` **RGB** mesh
      as a **4-component** normalized `color` attribute (padded RGBA, opaque alpha) and a `uint8`/`float16`
      **scalar** as a `float32` `aScalar` — never a size-3 `uint8`/`uint16` color nor an itemSize-1
      `uint8`/`float16` scalar, both of which fail `createRenderPipeline` validation on the WebGPU backend.
      Assert the resulting `BufferAttribute` `itemSize`/array-type and that the padded alpha is opaque
      (`vAlpha == 1.0`); pairs with the §6.4 codegen-snapshot harness that exercises the TSL/WebGPU path.
- [x] TS unit: on a **`double_sided: false`** mesh, `updateView` for a `displayDims` change `[0,1,2]`→`[0,2,1]`
      re-extracts positions, re-decides the `normal` attribute, recomputes bounds, and reverses the index
      winding so front faces stay visible (goes red without the reversal precisely because `FrontSide`
      culls the flipped triangles) — rebuilds `position`/`normal`/index, not just the index (§7).
      Separately, on a **`double_sided: false`, ≥4D** mesh carrying normals (so `sorted(normal_dims)`
      declares the winding frame, §3.2) displayed with an **odd-parity** ordering of that frame, a pure
      slicePosition move (unchanged `displayDims`) rebuilds the index only BUT still preserves
      front-facing winding (front faces stay visible), proving the winding post-pass persists across
      slice moves (§5.4), not just the displayDims-change event. And on the same mesh, a `displayDims`
      change to a **different axis triple** than the frame falls back to `DoubleSide` for the epoch —
      both orientations render (§5.4/§7)
- [x] TS unit: **flat-vs-smooth on the same normal-bearing mesh** — one mesh with valid stored normals
      (`normal_dims == displayDims`; use vertex-averaged normals that differ from the geometric face
      normals, so smooth and flat genuinely disagree — a faceted face-normal fixture would render the
      two variants identically even in a correct build) selects the stored-normal `mesh.fragment` variant under
      `shading="smooth"`, and the SAME mesh under `shading="flat"` selects `mesh-flat-normal.fragment`
      and shades from screen-space derivatives (§3.4, §6.2). Verified to differ — a build that ignores
      `shading` renders both identically and the test goes red. And under `shading="smooth"` on a
      `double_sided` mesh, a **back-viewed** face shades on the lit side of its own view-anchored
      gradient, NOT collapsed toward flat `uAmbient` — verified to go **red** without the §6.2
      `gl_FrontFacing` normal flip (the back face shades the unflipped, inverted gradient instead). And
      on the same back-viewed mesh
      with one vertex's stored normal zeroed, the fragments where §3.5's epsilon guard fires shade from
      the substituted viewer-facing derivative normal **without** the flip — verified to go **red**
      against a build that applies `gl_FrontFacing ? N : -N` to the fallback normal (§6.2's
      per-fragment exemption)
- [x] TS unit (**both backends** — GLSL and TSL): **stored-normal view-space transform** — a
      smooth-shaded mesh whose stored per-vertex normals equal its geometric face normals, with at least
      one face normal that **mixes the differently-scaled axes** — a nonzero component both along z and
      within the xy-plane (a tetrahedron, whose four face normals positively span R³, guarantees this;
      an axis-aligned box, or a quad tilted only within the equal-scale xy-plane, does not: for any
      normal that is an eigenvector of the scale `S·n` stays parallel to `S⁻¹·n`, so the
      inverse-transpose error hides), placed on a node whose §3.3 4×4 `transform` both rotates and
      non-uniformly scales (voxel z ≠ xy).
      Under `shading="smooth"` it shades from the `normalMatrix`-transformed (inverse-transpose of
      model-view) view-space normal (§6.2): the gradient stays head-on / camera-anchored and — on this
      faceted fixture — matches the flat variant's derivative-normal response on the same geometry. Run
      on the hand-written GLSL backend as well as TSL, since §6.2 designates this behavior test (not a
      codegen snapshot) as the GLSL twin's pin. Verified to go **red** against a build that binds the raw
      untransformed normal, or uses the plain model-view instead of the inverse-transpose — either
      mislights the rotated node / skews normals off-perpendicular under the anisotropic scale
- [x] TS unit / E2E (pick corner contract, §6.5): a pick on a mesh triangle resolves to **a corner
      vertex of that face** — assert membership in `(i0, i1, i2)`, never one exact corner (the provoking
      vertex is backend-dependent: last on WebGL, first on WebGPU; `WEBGL_provoking_vertex` aligns them
      only where available)
- [x] Codegen snapshots: 6 new variants = 12 files (§6.4) — incl. the per-blend-mode
      `mesh-additive`/`mesh-max` variants
- [x] Fixture: `tests/fixtures/generate_test_data.py` gains a mesh fixture (auto-picked up by
      `vitest.config.ts` globalSetup)
- [x] E2E: one `mesh-rendering.spec.ts`, plus extend the existing multi-geometry
      `tests/e2e/geometry-types.spec.ts` (it asserts `userData.nodeType` per type and already covers
      lines + gsplats) with a mesh case
- [x] One demo exercising the type end to end

> Every test must be verified to **fail before the fix** — mutate the implementation and confirm the
> test goes red. A cull test that passes against an all-ones mask is vacuous.

---

## 9. Explicitly out of scope

Each of these is a deliberate exclusion, not an oversight. Each should surface clearly — an error, or
for `volumetric` the named one-time warning + `opaque` fallback of §6.3 — rather than silently misbehave.

> This section carried two code follow-ups it OVERSTATED its own compliance on, and the
> §6.3 exclusion one of them mitigated has since been **lifted**:
>
> - The `add_mesh` refusal message no longer justifies refusing substitutive LOD with
>   additive's reason. It used to read "the additive/substitutive ladder reduces independent
>   elements (a surface is connected)" — true of the additive flavour, false of the
>   substitutive one. Substitutive levels have since **landed**, so only an additive ladder
>   over an *arbitrary* order is still refused, with its own reason (`MESH_ADDITIVE_METHODS`
>   naming `radial` as the sole accepted method, in
>   `packages/luxar/src/luxar/core/group/lod/mesh.py`).
> - §6.3's translucent-`normal` warning was implemented in `commit-mesh-geometry.ts`, and has
>   since been **removed** — not regressed. It existed as the named mitigation for the
>   per-triangle-depth-sort exclusion, and that exclusion no longer exists (§6.3, and the note
>   under the table below). What is left to say about an unsorted `normal` mesh is exactly what
>   is true of the other three types with depth sorting switched off, and none of them warn.

| Excluded | Why | Natural follow-up |
|---|---|---|
| **Additive LOD ladder** | A prefix of an index buffer is a **holed** surface, not a coarse one. That is the difference from a splat prefix, which genuinely is a sparser approximation of the same field — so the ladder degrades gracefully there and produces a *wrong picture* here. The legitimate refinement scheme is a progressive mesh (base mesh + vertex-split records), which cannot use the prefix-count ladder at all: different data structure, not a widening. | Not a LOD in any form. A deliberate progressive-draw effect IS worth exposing, and has shipped for all four types as the `radial` ordering — see **§9.1**. It uses the `additive` code path deliberately (only the ordering differs); what it must not inherit is the energy stamps, which one shared `REVEAL_METHODS` predicate handles |
| **Exact nD triangle clipping** | ~1500 LOC across two backends. §5 covers the dominant real case (hidden dims are discrete — time/channel) for ~10% of the cost, but gives only a **thick slab**, never a true cut, when a hidden dim is continuous and spatial (§5.2.1). | Slot in behind the same `MeshDataLoader.updateView`; the mask kernel becomes the fast pre-pass. **Promote this if continuous hidden spatial dims turn out to be a real use case** — a condition that is now *measured* rather than asserted: `processMeshData` emits a `log.info` for a node whose hidden dims include a continuous one (`noticeContinuousHiddenDim` in `data/scene-loader/process/data-processor-mesh.ts`), naming each such dimension and its unit. Promote when that line starts appearing against real datasets; see TODO item 29 under "Future / Exploratory" for why the deferral is a decision rather than a backlog entry |
| **Spatial index** | Not merely "see §7": a chunk of faces is not independently meaningful, because the index buffer references vertices anywhere in the array — so a face chunk draws only with the whole vertex buffer resident, or after the same remap/duplicate bookkeeping the partition row describes. An efficiency cliff, not an impossibility: partial loading is achievable, it just forfeits most of the bandwidth win a chunk index exists to buy. Moot in practice as well, since the 512 MiB per-node byte budget binds first (≈22.4M vertices for a 3D float32 mesh, measured), well under §7's ≤-few-million-triangle expectation. | Mirror the lines dual-index loader over faces |
| **`volumetric` blending** | Not about opacity — about **path length**. Emission–absorption integrates κ over the distance a ray spends inside a participating medium, and a triangle is zero-thickness, so τ = 0 however translucent the surface is. The adjacent feature that DOES make sense — volume rendering bounded by a mesh's front and back faces — is a different thing entirely and is not what this excludes. | — |
| **Worker projection** | Measure first (§7). | — |
| ~~**Mesh import formats** (PLY/OBJ/STL/VTP/glTF)~~ — **landed** | Independent of the node type, which is why it could ship on its own afterwards. | Shipped as `luxar mesh import` (`luxar/mesh/interop/`), mirroring `gsplat import`. VTK XML PolyData (`.vtp`) joined later, on the same NumPy + stdlib terms |
| ~~**`kind=partition`**~~ — **landed** | The exclusion was bookkeeping, not correctness, and the bookkeeping is now written. | Shipped as `add_mesh(partition=…)`; see §9.2 |

Two rows have been **lifted since this table was written**, for opposite reasons: one was
over-estimated, the other under-estimated.

**Per-triangle depth sorting.** It was the weakest exclusion here, and the only one with a live
user-visible consequence. It turned out to be almost entirely reuse — the registration, the
SortWorker and the sort kernel are geometry-agnostic and a triangle's centroid is 3 floats like
any other center — with one genuinely different piece: the apply permutes `geometry.index`
atomically instead of streaming an `aSortedIndex` indirection, because a half-written index
buffer is not a permutation. See §6.3 and
`rendering/depth-sort-coordinator/triangle-ordering.ts`.
`GEOMETRY_CAPABILITIES.mesh.depthSortable` is now `true`.

**Substitutive LOD levels.** The row was right that
nothing structural blocked them — a level is an independently-authored `(vertices, faces)`
pair selected by `coverage_fraction`, and the machinery makes no independence assumption —
but its cost estimate was wrong twice over, in the same way in two places.

It called the viewer side "a two-line widening: flip `GEOMETRY_CAPABILITIES.mesh.lod` and
extend `LODGroupMetadata.display_type`". Both edits are real and necessary, but flipping
the capability makes `canDefer` admit mesh children, and the defer dispatch then needs a
`loadMeshNodeCheap`/`loadMeshNodeExpensive` split the mesh loader did not have. Without it
the dispatch throws; with the flip but no split, every level loads eagerly at scene open —
measured 4/4 levels resident at first paint, against 1/4 once the split landed.
`lod-freshness.ts::countFromUserData` needed a mesh arm too, or the empty-level display
guard silently no-ops for meshes.

The **Python** side was assumed free and is not, for a reason that only shows up on reading
`resolve_substitutive_axis`: Points and Lines share one resolver because both coarsen by
LIFTING to gsplats, so its vocabulary carries `truncation_radius`, `max_aspect`, `device`
and `seed` — four keys that exist only because of that lift — plus a `method` set of
Gaussian-mixture reducers. A mesh is decimated instead, so it needs its own resolver
(`core/group/lod/mesh.py`) with `method` in `{auto, cluster, qem}` and each lift-only key
refused by name. Widening the shared one would have accepted five words that quietly do
nothing.

The producer is `luxar.mesh.decimate`: vectorized vertex clustering for very large
surfaces, or Garland-Heckbert QEM edge collapse with a link-condition veto when
manifoldness must survive. `method="auto"` selects QEM through 10,000 vertices and
clustering above that measured worst-case open-surface envelope. QEM ladders reuse one
collapse sequence and snapshot each requested level rather than restarting from the
original mesh. Each coarse level is therefore a strict collapse-subsequence of the finer
one, so a LOD swap cannot reshuffle the surface between independent approximations. On
an open near-planar surface the orientation veto can stop well above the requested count
and therefore shorten the ladder; `cluster` is the tier to use when closely hitting the
count matters more than topology preservation. A QEM quadric also requires a
normal direction outside the triangle span, so fewer than three coarsening dimensions
make `auto` fall back to clustering and make explicit `qem` invalid. It is reached from
`add_mesh(substitutive_lod=…)` or `luxar mesh lod`. Per-level picking needed no work,
exactly as the row predicted: the LOD registry hides inactive levels and the picking system
skips hidden nodes.

The **ADDITIVE** ladder row above is untouched and still correct — *as a row about LOD*.
The `lod` capability flag gates `kind=lod` groups, whose levels REPLACE one another; an
additive ladder is `additive_<i>/` subgroups inside a leaf, and as a **level of detail** it
stays refused permanently, for the reason the row gives.

What has since shipped is the other thing the row points at: the **reveal effect** of §9.1.
`add_mesh(additive_lod=…)` writes `additive_<i>/` subgroups for `method="radial"` and for
nothing else — the method set is a one-element frozenset, so the refusal of every
arbitrary-order prefix is enforced by the vocabulary rather than by a separate guard. So the
distinction this section draws is now load-bearing in code: same subgroup layout, admitted
for the effect, still refused for the LOD.

### 9.1 Reveal ladders — an additive prefix as an EFFECT, never as a LOD

An additive prefix over a mesh is a legitimate thing to *offer*, as a deliberate
progressive-draw effect: a surface that grows in. It is not a level of detail, and the
distinction has to survive into the naming, because the existing additive machinery
assumes *prefix ≈ approximation* and that assumption is false for a surface.

> **Superseded mechanism (2026-08-09/10).** This section previously specified the reveal
> as successive `drawRange` extents over a reordered index buffer. That is **not** how it
> is built. The reveal is **authoring only**: an additive ladder whose ordering is
> concentric shells around the node's own bounding-box centre. The existing streaming
> machinery then loads the prefixes progressively and the object visibly grows outward.
> **No viewer changes, no new display machinery — nothing about how the data is
> DISPLAYED, only how it is LOADED.** §9.1's *reasoning* below was right about the hazard
> and is preserved and generalised; only the mechanism claim was wrong.

**The mechanism is one new ordering option per additive implementation.** Ordering was
already a pluggable choice, so the method — named **`radial`** — is a single new member
of each method registry plus a scorer. It has **shipped for GSplats, Points, Lines and
Mesh** (`-m radial` on `gsplat lod`, `additive_lod={"method": "radial"}` on `add_points` /
`add_lines` / `add_mesh`), with `reveal_centre` / `spatial_dims` overrides. Two properties are worth
stating because they are what make it read as a reveal:

- the centre is the **node's own bbox centre, not the scene origin**, so a dataset far
  from the origin grows from its own middle rather than in from one corner;
- the distance spans the **spatial axes only**, so a stacked time or channel column
  cannot become a shell dimension — otherwise the elements furthest in time land at the
  end of the ladder and an off-centre timepoint's slice paints last instead of growing
  outward. How that set is found differs by geometry, and it is worth knowing which you
  get: gsplats use the axes with real **covariance** extent (`_nondegenerate_axes`),
  which a stacked axis fails by construction (it is built with `sigma=0`); element
  geometries have no covariance, so `add_points` / `add_lines` take the scene's
  **displayed** dims, falling back to non-zero positional extent when the positions are
  not scene-aligned. Either way `spatial_dims` overrides it.

On Lines it orders **whole polylines** by their own centre, so every prefix keeps valid
segment topology. On **mesh** it orders whole **faces** by their centroid, and each level
re-indexes its own gathered vertex table through `luxar.mesh.split.split_mesh_by_faces` —
whose contract is "these face groups are a true partition", which is exactly what an
additive ladder's levels are, so the ladder needed no new re-indexer.

The mesh half is where the reveal restriction becomes *the whole vocabulary* rather than one
option among several: `MESH_ADDITIVE_METHODS` is `{"radial"}`. Points and Lines also accept
`random`, `salience` and the two samplers, because a prefix of an element cloud is a sparser
SAMPLE of the same object. A random half of a mesh's triangles is not a coarser surface, it
is confetti — so for mesh, "reveal" is not a mode, it is the only thing a prefix can honestly
be. Two things fall out of that, each independently worth the restriction: the energy stamps
below cannot arise (every reveal method is excluded from them), and vertex duplication at the
level boundaries stays far off the 3x unwelded ceiling a scattered order approaches.

The ordering is **best-first growth through face adjacency**, keyed on radius from the reveal
centre — not a radius sort. A radius sort delivers the contiguity claim on a convex blob and
breaks it on a closed surface, which is what §Target data names: every centroid sits at nearly
the same radius, so the order is decided by noise over the whole shell. Measured edge-connected
components of each cumulative prefix, `n_lods=4`, under a plain radius sort: `20 / 20 / 1 / 1`
on a 1280-face icosphere. Growing through shared edges makes it structural — the frontier only
admits a face touching one already admitted — so every prefix is one connected patch on a
sphere, a torus or a non-convex dumbbell, and the duplication argument gets its single boundary
curve. Measured at 4 levels, reveal vs random over the same faces: 288-face plane 1.66 vs 2.95,
320-face icosphere 2.67 vs 3.31, 1280-face icosphere 1.69 vs 3.25 (2.81 / 2.17 for the reveal
under the radius sort — the connectivity fix is what moved them).

Authoring landed with `add_mesh(additive_lod=…)` and `write_mesh_multi_lod`; the viewer's
second half landed with it — `createProgressiveMeshLoader` opens a mesh node declaring
`n_additive_sublods > 1` and `MeshProgressiveLoader` fetches its levels in order,
concatenating each revealed prefix into the buffers the node was sized for.

The **CLI** reaches it through `luxar mesh lod --recipe reveal`, whose `-m/--add-method`,
`--n-lods`, `--counts`, `--reveal-centre` and `--spatial-dims` map onto the `additive_lod=`
keys above. `--recipe` selects rather than the knobs composing, because `add_mesh` refuses
an additive ladder alongside a substitutive one — a mesh has no coarse prefix, so the two
are different products rather than two axes of one. `--reveal-centre` / `--spatial-dims`
share their parser with `gsplat lod` (`cli/reveal_options.py`), so the two commands cannot
drift on what a centre or an axis order means.
Labels are cleared on a laddered mesh — one source vertex maps into every level that touches
it, so a union CSR spanning levels has no well-defined index space; `substitutive_lod=` and
`partition=` both keep theirs.

**One argument FOR the ladder route that the `drawRange` design missed:** depth sorting
permutes `geometry.index`, so a reveal expressed as a `drawRange` over a reveal-ordered
index buffer would be scrambled by the first sort. A ladder of separate subgroups is
immune, because membership and draw order are independent concerns.

**The hard rule: a reveal ladder must NOT carry `energy_fraction_cum` stamps.**

This is now **enforced by construction** for all four shipped types, not merely
specified: `luxar.utils.lod_methods.REVEAL_METHODS` names which orderings are reveals, and
both the gsplat ladder (`gsplats/lod/additive.py`) and the element ladder
(`core/group/lod/group.py::additive_level_stats`) consult it and omit the stamps. The
element path gets it from the single flag that already governed the per-level
`energy_fraction_cum` and the parent `reference_energy` together, so the both-or-neither
contract holds without a second coordinated edit. `lod_method` and the count fields are
still written — they are provenance, and nothing keys brightness off them.

Accepted consequence, stated rather than buried: cross-fade and the `e >= 0.6`
early-upgrade release are gated on the same stamps, so **a reveal hard-switches between
shells**. That is the right trade — a wrong brightness is far more visible than a missing
fade — but it is a real loss.

The reasoning, which generalises to every geometry:

`energyCompensation` (`scene/lod-blend.ts`) multiplies a leaf's brightness by `1/e(k)`
while an additive ladder is incomplete, and it is gated on
`BLENDABLE_MODES = {additive, luminous, volumetric}` — **not on geometry type**. Mesh
supports `additive` and `luminous` (§6.3), so a stamped mesh reveal in either mode would
be brightened. That is right for a splat prefix, which genuinely is dimmer than the whole,
and backwards for a mesh prefix, which is not dimmer but *holed*: the result is a surface
with gaps that also glows. Mesh's `opaque` default escapes this today by luck, not design.

Omitting the stamp is both the cheapest fix and the honest encoding, because the stamp's
meaning — "this prefix is dim, compensate for it" — is a false statement about a holed
surface. `energyCompensation` already returns exactly `1` for an absent `e`, leaving the
leaf byte-identical. LOD **cross-fade** is gated on the same set and follows the same rule.

**Measured scope of the hazard (2026-08-10).** The compensation is applied in exactly one
place — `applyLodFade` (`scene/lod-fade.ts`), whose only caller is the `kind=lod` group
registry (`scene/lod-group-registry.ts`). So the stamps are consulted for a ladder *inside*
a lod group and are inert on a bare `stream`/`flat` leaf. Verified by rendering the same
radial ladder twice, stamped and unstamped, under a throttled server: as a bare leaf the
two are byte-identical frame for frame, while inside a `levels` group the stamped arm is
**1.87× brighter in mean luma** (p99 luma 109 → 155) for as long as the ladder is
incomplete, converging to identical once it completes.

The rule stays unconditional, and the reason is not that a bare ladder might drift into a
lod group — it cannot do so silently. Every ladder-producing path rebuilds through
`make_additive_lod` / `additive_level_stats` and therefore re-consults `REVEAL_METHODS`:
both `gsplat additive` and `lod --recipe levels` were run on an already-radial ladder and
both **discarded** it, re-deriving from the method they were given (default → a `greedy`
ladder, correctly stamped; `-m radial` → still unstamped). The real reason is simpler: the
method is the only thing an authoring call keys on, and it already covers both cases —
`--recipe levels|adaptive|overview -m radial` writes reveal ladders *directly inside* a lod
group, where the stamps bite, and `--recipe stream -m radial` writes a bare one, where they
are inert. One predicate, both cases, no scope test needed.

The rule is **enforced at write time**: `add_mesh` raises if `level_stats` or `lod_stats` is
supplied (`_reject_energy_stamps` in `packages/luxar/src/luxar/core/group/adders/mesh.py`) — on key
presence, deliberately broader than the energy fields themselves, since neither attribute has any
meaning on a mesh today. Substitutive mesh levels are the one thing that would change that
(`level_stats.quality` is a legitimate non-energy stamp): the decimator has since landed, so
narrowing the guard to the energy keys — rather than routing around it — is the outstanding
follow-up. It has to be the adder that refuses,
because the write path's allow-list `_ALLOWED_NODE_ATTRS` in
`packages/luxar/src/luxar/io/_compiler/node_common.py` is geometry-blind and would let either key
through on any node type. The refusal is prophylactic rather than a fix for a live bug: one latch
still holds — the mesh commit never stamps `committedEnergyFraction`, so the factor is 1. The other
(mesh could not sit in a `kind=lod` group) is **gone**: substitutive LOD landed, so the fade pass
does visit a mesh level now. A reveal ladder would remove the remaining one.

**Not a concern:** `coverage_fraction` auto-selection. That selector chooses between
*substitutive levels*; an additive ladder inside a leaf streams to completion and is never
distance-selected. A reveal cannot be picked as a distant stand-in because nothing picks it.

**Consequences to state plainly rather than bury:**

- **Every intermediate frame is a wrong picture, not an approximate one.** Under `opaque`
  with depth writes you see interior back faces through the gaps — correctly lit by the
  `gl_FrontFacing` flip, so it reads as a hollow shell. For a deliberate reveal that is
  arguably the appeal; as a quality ladder it is indefensible.
- **It must converge to 100%.** A splat ladder may legitimately stop early against a
  bandwidth budget. A mesh reveal that stops early leaves a permanently broken model.
- **The reveal ORDER is the whole effect.** Area-descending reads as blocky-then-refined;
  region-growing from a seed reads as the surface growing; contribution-ordered — the
  splat metric — reads as confetti, because a mesh triangle has no "contribution" to
  order by. Whoever ships this is choosing an aesthetic, not an error metric, which is a
  different kind of decision from QEM's and should not reuse its vocabulary. `radial` is
  the shipped choice, and for mesh it has a second, non-aesthetic advantage: concentric
  shells keep vertex duplication near the lower bound (each shell boundary is one closed
  surface), where a confetti order approaches the 3× worst case.

**Naming.** The method is `radial`, and it sits *inside* the additive vocabulary rather
than beside it — the earlier plan to keep it off that code path entirely
(`reveal` / `progressive_draw`) was dropped once the ordering turned out to be the only
thing that differs. What it must NOT inherit is the energy-stamp machinery, and that is
handled by `REVEAL_METHODS` above, which is a sharper boundary than a separate code path
would have been: one predicate, consulted by every implementation, rather than a parallel
set of builders that could drift.

**Pre-existing gap noticed during this spec's review, since closed by #1220:**
`POINTS_/LINES_/GSPLATS_RESERVED_ATTRS` had omitted `has_image_labels` even though all three writers
stamp it. No clobber ever resulted — `validate_render_attrs`'s reject-unknown gate already failed such a
write, just with the *unknown-attr* message instead of the *reserved* one — so it was only an
error-message gap. #1220 added the key to all three sibling sets, matching `MESH_RESERVED_ATTRS` (§3.3).

### 9.2 `kind=partition` — lifted

**Shipped.** `add_mesh(partition=True | {"max_elements": N, "rule": …})` writes a
`kind=partition` wrapper with one independently-drawable `Mesh` child per BSP part, and
`GEOMETRY_CAPABILITIES.mesh.partition` is `true` on both sides of the contract.

This row was always the weakest of the structural exclusions, for the reason the old table
gave: it was **bookkeeping, not correctness**. What the bookkeeping had to do:

* **Split on face CENTROIDS, never through a face.** A triangle is the indivisible unit, so
  `max_elements` counts FACES. No geometry is cut and no vertex is invented.
* **Re-index, do not slice.** The sibling adders hand each part a slice of their element
  arrays because their elements are independent rows. A triangle is three *references* into
  a shared vertex table, so each part gathers the vertices its own faces use and renumbers
  those faces against the gathered table (`luxar/mesh/split.py`).
* **Duplicate across the cut.** A vertex referenced from both sides appears in both parts.
  That is the cost, and it is what makes each part stand alone as a drawable leaf. It is
  bounded by `3F` in the pathological case and is a few percent in practice — only the cut
  surface duplicates. The writer reports the measured factor.
* **Gather every per-vertex attribute**, including the per-vertex label CSR the old table
  called out: `normals`, `colors`, `scalars` and `labels` all travel with their vertices
  (via the shared `slice_optional_array`, which gathers on length and so leaves a uniform
  RGB triple or a colormap name alone).
* **Stamp ONE scalar display window on every part.** Derived from the whole field before
  the split, unioned with an explicit `_scalar_data_range` when one is given, because the
  viewer windows each node's colormap on that node's own stamped range: per-part min/max
  recoloured the
  same value either side of a cut, and a part whose subset is constant landed on the LUT
  midpoint. Same rule §9's substitutive ladder applies to its levels, via the same helper.
  The pair is also each part's quantization range, so a field with one extreme outlier now
  spends its codes on the global span rather than per part — the display the file is meant
  to be viewed at is unchanged, and the ladder path already made that trade.

**The seam question resolves the way the old row predicted.** With stored normals split
verbatim, a duplicated boundary vertex carries an identical position AND an identical normal
in both parts, so nothing seams under §6.2's shading; the derivative variant is per-fragment
off the rasterized triangle and is part-agnostic by construction. **Revisit this the moment
shading gains anything RECOMPUTED per part** — area-averaged normals, tangent frames, UVs,
baked AO — because each of those is computed from a part's own contents and would differ
across the cut.

**The wrapper records the recursive split planes as `bsp_tree`.** The tree is built from the
same face-centroid BSP whose leaves become the mesh parts, then pruned and renumbered against
the parts actually written. Because faces are assigned by centroid while `position_bounds`
cover all three vertices, a triangle may cross its assigning plane; the resulting traversal is
stable and localizes ambiguity to the overlap, but is approximate rather than an exact painter's
order. It still avoids the coarser whole-part centroid fallback for translucent mesh.
`render-order.ts::traverseBspBackToFront` also orders opaque mesh parts back-to-front, which is
correct but forfeits the front-to-back early-Z order an opaque pass would prefer. That is a known
performance tradeoff, not a correctness defect; keeping one partition metadata contract across
all four geometry types is more important than special-casing opaque mesh authoring.

**One refusal survives the lift.** A mesh may go under a `kind=partition` group whose
`display_type` is `'mesh'` — nothing else. A partition is homogeneous by definition, and
`validate_partition_group`'s homogeneity check has no production caller, so declaring a
`points` partition and dropping a mesh into it would write clean and load as a layer
claiming to be points. `reject_mismatched_partition_parent` (in `core/group/partition.py`)
refuses that pairing fail-fast, before any array reaches disk — and it is called by all
four leaf adders, so the rule is symmetric: a points leaf under a `display_type='mesh'`
partition is refused the same way. That direction only became reachable here, since a mesh
partition could not be built at all before mesh became partition-capable.

Still excluded, and unaffected by this: a partition **of** a mesh LOD ladder. A substitutive
ladder now exists (§9), but `partition=` and `substitutive_lod=` cannot be combined in one
`add_mesh` call — the same refusal `add_lines` carries. Points composes that pair into a
global-coarse overview above partitioned fine detail; Mesh does not write that topology yet.

---

## 10. Architecture: what the consolidation changed for mesh

Mesh was the forcing function for a question the codebase had been deferring — whether a fourth type
should be added to ~30 hand-written dispatch sites, or whether the dispatch should be table-driven
first. An earlier draft of this section argued for the second answer. It shipped, before any mesh
code, so this section now records the result rather than the argument.

### 10.1 What landed

| PR | Change |
|---|---|
| #1079 | `geometry_types` added to `contract.yaml` and `GeometryKind` derived from it; the duplicate `GeometryType` union repointed; `LoaderRegistry`'s three parallel maps collapsed to one kind-keyed store; `lod_backfill` moved to `group_keys()` + `GEOMETRY_TYPES` |
| #1099 | Points given the same `process`/`commit` pair as lines and gsplats, so `NodeBuildCtx` carries one uniform pair per kind; the three hand-written dispatch switches replaced by a single `Record<GeometryKind, GeometryDescriptor>` |
| #1150 | Points metadata brought level with lines (`ndim`, `ordering`, `max_radius`, `has_spatial_index`); vestigial material caches and unused exports removed; the `GeometryType` alias #1079 left behind finally deleted |

### 10.2 What that means for a fourth type

The vocabulary is now single-sourced from the format contract, so **adding `mesh` to
`contract.yaml` propagates to every consumer**, and the places that must still be taught about it
fail the build rather than going quiet. Note the trigger: these three errors fire on adding `mesh` to
**`loader_types`** (the Phase-3 switch-on, §3.1), not to `node_types`/`geometry_types` — those two
already contain `mesh` (#1220) and compile clean. The errors are the map of what Phase 3 must implement:

- **Missing loader entry** → `TS2339: Property 'mesh' does not exist on type 'LoaderByKind'`
  at `loader-registry.ts`, plus three `TS2536` follow-ons inside the generic accessors
- **Missing descriptor row** → `TS2741: Property 'mesh' is missing … required in type
  'Record<GeometryKind, GeometryDescriptor>'`
- **Missing tolerance arm** → `TS2366: Function lacks ending return statement` in
  `tolerance-computer.ts`

The descriptor row is the one that matters most. Before the consolidation, `lifecycle/retry.ts`
expressed a per-type capability as *the presence of an `if`*: a kind with no arm there was not a
compile error, it was a load that could never be retried, manually or on reconnect. Nothing in the
type system said so, and it was found only by reading. It is now a declared field on a row that
cannot be omitted.

**Three dispatch sites therefore need no mesh edit at all:** `load-scene-nodes.ts`,
`lifecycle/retry.ts` and `prefetch/slice-prefetcher.ts` all read the descriptor table.

### 10.3 What deliberately stayed specialized

Not everything should be unified, and the consolidation did not try. Materials, geometry assembly,
projection kernels and storage layout are genuinely different per type — §2.1 and §6 argue mesh
differs from the other three *more* than they differ from each other. Forcing those behind one
interface would be worse architecture, not better.

The descriptor table holds only capabilities with a real consumer. It deliberately carries no
"supports X" flag that every kind currently answers identically, on the grounds that a field no
branch reads is indistinguishable from a field that is wrong. Mesh should respect that when it adds
its row: declare `applyPartialExtendTolerance` and the factories because those are read, and express
"no slice prefetch" (§7) by not registering a prefetch call site, not by adding an unread flag.

### 10.4 One safety rule mesh must follow

`SceneNode.type` is an unvalidated string — `build-scene-graph.ts` copies `attrs.type` verbatim out of
the store. A bare `GEOMETRY_DESCRIPTORS[node.type]` therefore resolves through `Object.prototype` for
values like `constructor` or `toString` and returns a truthy non-descriptor; the resulting `TypeError`
is not a `LoaderError`, and `loadLeafNode` re-throws anything else, so a single such node sinks the
whole scene load. This was a real regression caught in review of #1099.

Use `geometryDescriptorFor(node.type)`, which gates the lookup with `Object.hasOwn`. Code holding a
`GeometryKind` from a trusted source (the loader registry, a literal) may index the table directly.

---

## 11. Phased delivery

| Phase | Contents | Verifiable outcome |
|---|---|---|
| **0** ✅ *(done — §10.1)* | Single-source `GeometryKind` from the contract; collapse `LoaderRegistry`; unify the per-type pipeline; table-drive the dispatch switches | Landed as #1079 / #1099 / #1150, all behaviour-preserving. The `SceneGraphNodeType` local extension has since been deleted (#1220) — it is now plain `NodeTypeName` — so Phase 0 is fully landed |
| **1** ✅ *(done — #1220)* | Writable contract (`node_types`/`geometry_types` + `NodeType.MESH`/`NODE_TYPE_MESH`) + `core/mesh.py` + adder + writer + validators + reader + `info` + the LOD/partition rejections (§8) | Landed as #1220, all in one PR: `scene.add_mesh(...)` writes a `.luxar.zarr`; `luxar info --stats` reports it; round-trip tests green; a mesh child of a lod/partition group **raises** (both rejections pinned by tests). The partition half has since been **lifted** (§9.2): a mesh may now go under a `display_type='mesh'` partition group (`test_mesh_under_a_mesh_partition_group_is_allowed`), and only the mismatched-`display_type` refusal survives. **Superseded:** the lod half has since been lifted too (§9) — `_reject_specialized_parent` is deleted, mesh's `lod` capability is `true` for the SUBSTITUTIVE flavour, and the test that pinned the refusal is now `test_mesh_under_a_lod_group_is_accepted`. What survives of this item is the refusal of an ADDITIVE ladder over an *arbitrary* order, which never came from this guard: it is `MESH_ADDITIVE_METHODS` naming `radial` as the sole accepted method |
| **2** ✅ *(done — #1232)* | Rust + TS cull kernels with parity tests | Kernels green in isolation, no viewer changes |
| **3** ✅ *(done)* | `mesh` → `loader_types` in `contract.yaml` (the switch-on — fires the three §10.2 compile errors) + `types/mesh.ts` + loader + node load + `mesh-geometry.ts` + one `LoaderByKind` entry + one `GEOMETRY_DESCRIPTORS` row + the `computeHiddenDimTolerance` arm | Mesh loads and renders **unshaded** (flat vertex color); E2E smoke green |
| **4** ✅ *(done)* | GLSL + TSL material pair + codegen snapshots + shading model | Shaded surface, both backends pixel-equivalent. Landed as the `mesh/` material stack (4 files + `appearance.ts`), 5 codegen variants (10 snapshot files — the sixth, `mesh-pick`, arrives with picking in Phase 5, which is why §6.4 and §8 count six), and the original §6.2 headlight with its compile-time stored-normal/derivative variant. Two spec refinements were forced by the implementation and are folded back into §6.2: the derivative normal is **forced** viewer-facing rather than assumed so (`cross(dFdx, dFdy)` has the sign of the fragment-space y axis, and WGSL's `dpdy` is top-down where GLSL's `dFdy` is bottom-up), and the stored normal is transformed WITHOUT three's `transformNormalToView` (whose internal `normalize` turns a legitimately zero-length normal into a whole-triangle NaN, contradicting §3.5's locally-distorted contract). The `normal`/`aScalar` attributes are bound for the node's lifetime from the metadata rather than bound/unbound per epoch — the shader variant alone stops reading them, which keeps the attribute set (and hence the WebGPU vertex layout) fixed |
| **5** ✅ *(done)* | Picking pair, layers panel, monitor, stats, camera framing, debug | Full parity with the other three at the UI level. The pick pair landed as `rendering/picking/mesh/` — six files rather than the siblings' four, the extra two being `pick-mode.ts` (the blending mode's TWO pick-pass consequences derived in one place, so "cutout on, brightness-as-depth" is unrepresentable) and `provoking-vertex.ts` (WebGL's `flat` provoking vertex aligned with WebGPU's). Two §6.5 refinements were forced by the implementation and are folded back above: the pick material tracks the visual material's `side` per epoch (the siblings' quads are view-facing, so theirs can pin `DoubleSide`; a mesh's culled back faces must not rasterize into the pick buffer at true surface depth), and the surface-depth VALUE differs between backends (`gl_FragCoord.z` vs three's `depth` node, which expands to a linear view-space depth) — benign, since both are monotone in distance and the pick depth only orders fragments within one render, and now documented rather than latent. The registration itself went into `NodeFactory.registerExistingSceneNodes`, NOT only the node factory: `initPicking` traverses the finished scene before constructing the `PickingSystem`, so on a first load the factory has no system to register with and that retro pass is the one production runs. It was a three-way `else if` chain and is now a `Record<GeometryTypeName, …>`. Stats / camera framing needed no work (Phase 3 had already made them four-way); the debug surface did — `getState()` gained `meshNodes` + `totalTriangles`, and `getDrawOrder()` was reporting **0 elements for every mesh** because its element-count fallback was a partial copy of the shared per-type reader with `visibleTriangleCount` missing. **The monitor was recorded here as needing no work, and that was wrong** — a later audit found mesh present in its scene-graph tree and absent from every other surface: no headline card (a mesh-only scene showed a permanent "LOADING …"), no compact-badge entry, no per-node visible-triangle suffix, no loader row / bytes / resident memory (the four-method surface above did not exist, so `connectLoaderToMonitor` dropped every mesh node without a log line), and no aggregated `Mesh` row in the Performance tab (its count rode as a free-text `info` string, which is last-write on merge). The lesson generalizes: "already four-way" was inferred from the type VOCABULARY being four-way (`GEOMETRY_TYPES`, `visibleByType.mesh`) while the display layer's named per-type fields — `GlobalStats.visiblePoints`/`visibleSegments`/`visibleSplats` — silently stopped at three, so the counts were aggregated and then dropped |
| **6** ✅ *(done — real-WebGPU A/B run; two claims remain reasoned, see the cell)* | Fixture + E2E spec + demo + docs + CHANGELOG | Shippable. `test_mesh.luxar.zarr` / `test_mesh_nd.luxar.zarr` + `mesh-rendering.spec.ts` landed first, and earned their keep immediately: the FIRST end-to-end render of a written mesh found three defects, including that §6.3's `opaque` default had never fired on the production load path (`normalizeBlendingMode(undefined)` returns `'additive'`, so the `?? 'opaque'` was dead code — see the §6.3 note). No unit test could have caught it: they all hand the resolver an attrs object rather than one that has been through composition. The fixture is a WELDED, CLOSED icosphere on purpose — welded so `gl_VertexID` is a genuine many-to-one pick target (162 vertices, not the 960 a de-indexed mesh would have), closed so an nD slab cull exposes interior back faces, and smooth non-axis-aligned normals so a build ignoring `shading` renders visibly differently. The reference demo is `demo_mesh_isosurface_cells3d` — marching-cubes isosurfaces of the same volume its gsplat twin fits, so the two representations can be compared directly on one dataset. The docs pass covered the two package READMEs that had omitted mesh entirely (`rendering/` gained a Mesh Material section naming the two backend hazards; `data/` gained the whole-node-loader section explaining why mesh does NOT stream and why its tolerance arm is its own), the Layers-panel README, and the stale "picking and the panel controls land in later phases" claim in `CLAUDE.md`. Two `README.md` claims that still say "all three geometry types" were checked and LEFT: one is about volumetric blending physics and the other about the chunk-bounds spatial query, and mesh genuinely participates in neither (§9). On the remaining gap: a **real-WebGPU A/B was run** (system Chrome channel, `?renderer=webgpu` vs the same URL without it, screenshot-then-decode with the WebGL arm as the control — the recipe matters, see the note below), and it establishes the substance of what was open. Native WebGPU reports `apiSurface: 'webgpu'`, commits all three fixture nodes with identical triangle/vertex counts and identical shader variants, and renders **pixel-equivalent** output to WebGL: 105,822 lit pixels on both, mean lit channel differing by 0.14% (sub-quantization dithering — equivalent to the eye, not byte-identical). So the WGSL path is no longer unexercised, and it agrees with GLSL on a real mesh.

The A/B also **corrected an overstatement** in the §6.2 notes, which is the more useful half. Those notes claimed an unforced derivative normal "would collapse to `uAmbient` everywhere on WebGPU". That is a spec-derived RISK, not an observed behaviour: `test_mesh.luxar.zarr` gained a `flat_facing` node — a flat-shaded quad FACE-ON, where `N.z ≈ ±1` makes the flip the difference between full brightness and the ambient floor — and with the `z >= 0` flip REMOVED, real WebGPU still renders it identically to WebGL. The metric is demonstrably sensitive (a control run with every mesh hidden drops from 93,851 lit pixels to 56,700, so the quad contributes 37,151 and lifts the mean from 40 to 116), so this is a measurement rather than a blind test. Conclusion: on Chrome + Apple Silicon the two derivative conventions COINCIDE and the flip is inert. It is kept because it costs one instruction, is correct under either convention, and neither shading-language spec promises they agree — insurance, not a fix for an observed bug. The shader comments now say exactly that.

(The earlier edge-on-only `flat_patch` could not have shown this either way: `N.z ≈ 0` there, and flipping the sign of ~0 leaves `wrap = clamp(0 · 0.5 + 0.5) = 0.5` unchanged. That is why the fixture needed the face-on node.)

Still arguments rather than measurements: the provoking-vertex convention and the surface-depth VALUE, both because reading the pick buffer's ids and depth from outside the app is not cheaply reachable |

**Post-Phase-6: the near-plane fade (#1431).** The Phase 6 audit above swept the "all three geometry
types" claims and left the two that were genuinely about volumetric physics and chunk-bounds queries.
It missed a third asymmetry that was real: mesh had no `perspectiveNearFade` in either backend, so a
triangle clipped hard against the near plane while the other three faded out. That is now closed —
§6.2 for the visual pair, §6.5 for the pick pair — and the change reaches further than the shaders:
all four mesh material wrappers became `CameraAwareMaterial`s (consuming `isOrtho` / `nearCull` and
ignoring `fov` / `resolution`), mesh joined the material manager's camera broadcast, and the
near-plane floor derivation in `scene/scene-manager/clipping/bounds-math.ts` — which called mesh "the
one type the floor can clip" — now holds for all four. Parity is covered by the harness's two
perspective entries (`mesh-near-fade`, `mesh-pick-near-fade`); the ORTHO default camera every other
entry uses makes the fade the identity, which is why those two exist.

Phase 0 landed alone, with no mesh code, so any regression it caused would have been unambiguous.
Phase 1 (the Python writable side) landed next, as #1220; Phase 2 (the cull kernels) was independent
of the remaining viewer phases and landed on its own as #1232. The *drawable* half of the contract
edit — adding `mesh` to `loader_types` — was deliberately **not** part of Phase 1: it fires the three
§10.2 compile errors, which can only be cleared by real Phase-3 code, so it belonged with Phase 3.
Phase 3 would once have been the widest diff — a sweep across every dispatch site — and turned out to
be one of the narrower ones: the three dispatch files needed no mesh edit, leaving the `loader_types`
switch-on, the loader, the geometry builder, and two table entries. Phase 4 was the deepest single
piece of work.

**Estimate (as written up front):** 5–6 PRs, roughly 4–5.5K LOC including tests — against ~14K LOC
for the full Lines vertical, the difference being everything in §9 plus the dispatch work Phase 0
already absorbed.
