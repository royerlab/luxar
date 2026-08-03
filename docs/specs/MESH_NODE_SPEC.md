# Mesh Node Specification

**Status:** Proposed
**Scope:** A fourth first-class geometry type — `mesh` — symmetric to Points, Lines and GSplats.
**Non-goals:** LOD/decimation, spatial indexing, exact nD triangle clipping. See [§9](#9-explicitly-out-of-scope).
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

Luxar has three first-class geometry types, all of which are *soft, emissive, per-element* primitives
rendered as instanced quads. None of them can represent a **surface**: an isosurface from a volume, a
segmentation boundary, a cortical/organ mesh, a CAD or simulation domain, or a molecular solvent
surface. These are routine outputs in the same scientific pipelines Luxar already serves, and today
they can only be approximated by dense point clouds.

A `mesh` node closes that gap with indexed triangles and real surface shading.

### 1.1 Existing forward-declaration

The viewer already reserves the name in one place:

- `packages/luxar-viewer/src/types/data-monitor-types.ts:473` —
  `export type SceneGraphNodeType = NodeTypeName | 'mesh';`
- `packages/luxar-viewer/src/data/scene-loader/monitor/scene-graph-converter.ts:25` — `'mesh'` in the
  display-type whitelist.

`format-contract/contract.yaml` documents this explicitly as *"a TS-only forward-looking member … the
viewer extends this base union locally"*. Both are display-only placeholders; there is no data model,
loader, geometry, or material behind them. This spec makes the name real and removes the local
extension in favour of the contract.

---

## 2. Where mesh is symmetric, and where it is not

The three-geometry symmetry rule (same attribute names, same decomposition, same shared helpers,
parallel tests) applies at the **node, attribute, writer and loader** layers. It does **not** apply at
the storage or LOD layers, and pretending otherwise would produce a worse design.

| Layer | Symmetric with Points/Lines/GSplats? | Notes |
|---|---|---|
| `DataNode` subclass, metadata, `n_elements` | ✅ Yes | Direct mirror of `core/lines.py` |
| Scene adder (`add_mesh`) | ✅ Yes | Mirror of `add_lines` |
| Zarr writer, encoders, shared dataset helpers | ✅ Yes | Reuses `write_colors` / `write_scalars` / `SemanticType.{COORDINATE,INDEX}` |
| Render attrs (opacity/gamma/intensity/offset/blending/colormap/transform/nd_transform) | ✅ Yes | Reuses `apply_default_render_attrs`, `prepare_transform_attrs` |
| nD slicing | ⚠️ Partial | Reuses the slab *semantics*, **not** the clipping algorithm, and needs **its own tolerance strategy** (the Lines one is derived from segment interpolation and would render nothing) — see §5, §5.2.1 |
| GPU storage | ❌ No | Indexed triangles, not instanced quads — see §2.1 |
| Per-element extent | ❌ No | A mesh has no `radii`/`widths`/`amplitudes` analog — see §2.2 |
| Depth sorting | ⚠️ Deferred | Per-triangle, not per-instance — see §6.3 |
| LOD / partition groups | ❌ Excluded | See §9 |

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
- the depth-sort coordinator's per-instance permutation

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
node_types:     ["scene", "group", "points", "lines", "gsplats", "mesh"]
geometry_types: ["points", "lines", "gsplats", "mesh"]   # the leaf subset
```

**Both** lists must be updated. `geometry_types` (added in #1079) is the leaf subset that keys
per-geometry dispatch, and the generator enforces that it is non-empty, a subset of `node_types`, free
of the container types, and duplicate-free — so adding `mesh` to only one of the two fails
`check-contract` with a named error rather than drifting silently.

Then `make gen-contract` regenerates both projections
(`packages/luxar/src/luxar/typing_utils/_format_contract.py` and
`packages/luxar-viewer/src/types/format-contract.ts`), and `hatch run check-contract` gates drift. The
`'mesh'` local extension in `data-monitor-types.ts` is deleted at the same time — `SceneGraphNodeType`
becomes plain `NodeTypeName`.

`NodeType.MESH = "mesh"` is added to `luxar/typing_utils/enums.py`.

### 3.2 Arrays

| Array | dtype | Shape | Required | Semantic type | Notes |
|---|---|---|---|---|---|
| `vertices` | float32 | `(V, D)` | **yes** | `COORDINATE` | nD, exactly like `Lines.vertices` |
| `faces` | uint32 | `(F, 3)` | **yes** | `INDEX` | Triangle vertex indices |
| `normals` | float32 | `(V, 3)` | no | `COORDINATE` | Per-vertex; paired with a required `normal_dims` attr — see §3.4 |
| `colors` | uint8/uint16/float32 | `(V, 3\|4)` | no | color helpers | RGB or RGBA; the 4th component is a **load-bearing** per-vertex opacity — see §6.2 |
| `scalars` | float32/float16/uint8 | `(V,)` | no | scalar helpers | Colormap lookup |
| `label_offsets`/`label_bytes` | — | CSR | no | — | Per-vertex hover tooltips |
| `image_label_*` | — | CSR | no | — | Per-vertex hover thumbnails |

`faces` uses `SemanticType.INDEX` with `deduplicate=False` and `allow_lut=False`, for exactly the
reason `Lines.segments` does: the loader reads it as raw chunked zarr and does not resolve `array_ref`,
so dedup would silently drop geometry for a byte-identical sibling, and LUT encoding of grid-snapped
values would decode as garbage topology.

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
has_labels: bool
has_image_labels: bool
shading: "smooth" | "flat"      # default "smooth" when normals present, else "flat"; consumed by §3.4/§6.2
double_sided: bool              # default true
position_bounds: [[min...], [max...]]
ordering: "none"                # v1 always; reserved for a future spatial index
```

plus the standard render attrs already handled by `apply_default_render_attrs` and
`prepare_transform_attrs` (`opacity`, `gamma`, `intensity`, `offset`, `absorption`, `blending_mode`,
`colormap`, `scalar_data_range`, `layer`, `transform`, `nd_transform`, `extend_to_all`).

`MESH_RESERVED_ATTRS` is added to `io/_compiler/node_common.py` alongside the other three frozensets:

```python
MESH_RESERVED_ATTRS = frozenset({
    "type", "n_vertices", "n_faces", "ndim",
    "has_normals", "normal_dims", "has_colors", "has_scalars", "has_labels",
    "has_image_labels", "shading", "double_sided", "position_bounds",
})
```

Note the sibling sets reserve `has_labels` but **not** `has_image_labels`, even though all three
writers stamp it. No clobber is actually possible — `validate_render_attrs` rejects the key as
*unknown* when it appears in no set at all — so the omission only costs the accurate "reserved"
error message. Mesh reserves it anyway, so its set covers every presence flag it stamps; aligning
the three sibling sets is a separate sweep, noted in §9.

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
`cli/gsplat_ops/transforms_commands.py:300`).

Making `normal_dims` explicit turns an invisible wrong-orientation render into a cheap, checkable
equality — and costs one attr.

An alternative — storing a full `(V, D, 3)` normal frame so any `displayDims` has true smooth normals —
was rejected as over-engineering for v1; the flat-normal fallback covers it correctly, just without
smoothing.

### 3.5 Validation

New shared validators in `luxar/validation/base.py`, following the existing `validate_*_for_writing`
convention (fail-fast, before any zarr group is created):

- `validate_faces_for_writing(faces, n_vertices)` — shape `(F, 3)` or flat `(3F,)`; integer dtype
  (reject float, which `.astype(np.uint32)` would silently truncate); `min >= 0`; `max < n_vertices`;
  `F >= 1`. Mirrors the `line_type='indexed'` index gate at `geometry_writers/lines.py:134-170`, which
  is the closest precedent and already encodes each of these traps.
- `validate_normals_for_writing(normals, n_vertices)` — shape `(V, 3)`, finite. Zero-length normals are
  **warned**, not rejected (degenerate triangles legitimately produce them). Render-time handling is
  **pointwise, not per-face**: on a shared-vertex indexed mesh the interpolated normal blends toward the
  neighbouring vertices' directions, so the stored-normal fragment variant (§6.2) simply epsilon-guards
  its `normalize` — when the interpolated normal is near-zero (`dot(N, N) < ε` before normalization) it
  falls back to the §6.2 screen-space-derivative flat normal rather than normalizing a zero vector into
  NaN shading. Shading near a degenerate vertex is therefore locally distorted rather than cleanly flat;
  the warning exists so authors fix the normals instead of relying on the guard.
- `normal_dims` (§3.4) — exactly 3 entries, integers, distinct, each `0 <= i < ndim`. Required when
  `normals` is supplied; rejected when it is not. It is an explicit `add_mesh` parameter (§4) — as
  writer-reserved metadata (§3.3) it cannot ride in through `**attrs`.

**Raise `ValidationError`, not `ValueError`.** Note the precedent cited above is split: the shared
`validate_*_for_writing` family in `validation/base.py` raises `ValidationError(message, hint)` — a
two-arg form that gives the user a remediation hint — whereas the Lines indexed-index checks are
**inline in the writer** and raise bare `ValueError`. Mesh should follow the *shared validator* half of
that precedent: face/normal validation belongs in `validation/base.py` as reusable, independently
testable functions, matching `validate_widths_for_writing` / `validate_radii_for_writing`. Only the
cheap structural gates that need writer context stay inline.

Tests mirror `validation/tests/test_lines_validation.py`, which is already parametrized over
`(factory, error_pattern, test_id)` triples — reuse that shape so each rejection in this section gets
its own named case, and **verify each fails before the validator exists** (a test that passes against
a no-op validator is vacuous).

**Loader-side validation (viewer).** The validators above run at write time and protect only stores this
writer produced; the viewer loads arbitrary — externally produced or corrupted — stores and hands `faces`
straight to the §5.4 kernels. An out-of-range face index **panics** the Rust kernel (the crate is
`panic = "abort"`, so the trap takes down the whole WASM module) and silently corrupts the TS backend
(out-of-bounds reads yield `undefined`). The mesh loader must therefore structurally validate after
decode, before either backend is invoked: `vertices`/`faces` shapes against `n_vertices`/`n_faces`,
`faces` length a multiple of 3, every face index `< V`, `n_vertices < 2^27`, and `normal_dims`
well-formed whenever normals are present — failing the node with a `LoaderError` (one node lost, not the
scene) instead of trapping. The `n_vertices < 2^27` check belongs at this gate because mesh's pick
`elementId` is `gl_VertexID` (§6.5) — the one type not bounded by the element-texture capacity — and
once a vertex ordinal reaches the pick vote-key stride (`2^27`) the vote key silently aliases across
nodes (the largest ordinal is `n_vertices - 1`, so `n_vertices < 2^27` keeps every ordinal strictly
under the stride), so the bound must be enforced here, not assumed from the §7 whole-load workload.
The optional arrays get the same structural gate whenever present — `normals` shape `(V, 3)`, `colors`
shape `(V, 3|4)`, `scalars` length `V`, and the label/image-label CSR offsets monotone and in-bounds
(§3.2) — because they bind as enabled vertex attributes on an **indexed** draw (§6.1): an undersized
attribute doesn't trap, it makes `drawElements` read past the buffer (an invalid-operation draw or
silent zeros, backend-dependent) and mis-shades every vertex it covers. Same `LoaderError`, same
one-node blast radius.

### 3.6 Authoring lint

Reusing the pattern of `_exploded_chain_fraction` in the lines writer (a warn-only heuristic that
catches a common authoring mistake), the mesh writer emits one warn-only lint:

> **Unwelded vertices.** If `V == 3F` and no two faces share a vertex index, the mesh was authored as
> independent triangles rather than a welded indexed surface. Smooth shading is impossible, per-vertex
> normals are meaningless, and the vertex array is ~3× larger than needed. Suggests welding.

Gated behind a minimum face count so tiny test meshes stay quiet, and routed through
`ctx.claim_*_warning` so a partition's leaves collapse to one message.

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
    *,
    shading: Literal["smooth", "flat"] | None = None,
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
is present (§3.4, §6.2).

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
| GSplats | `step × 3.0` (3σ) | quarter-cell |

⚠️ **Mesh must NOT copy the Lines row.** Lines can use `0` because segment clipping *interpolates
through* the slab — a segment crossing the slice yields a clipped intersection even at zero thickness.
Mesh has whole-triangle cull (no interpolation) and no per-element extent (§2.2), so a spatial
tolerance of `0` reduces the membership test to **exact float equality with the slice plane** and the
node renders **nothing**.

Mesh therefore adds a fourth arm to `computeHiddenDimTolerance`:

- **Discrete hidden dims** → `discreteDimMembershipTolerance` (half-cell). Mesh's slab test is a
  MEMBERSHIP gate, exactly like the lines projection-clipping slab, so it must request
  `discreteRole: 'membership'`; the default `'query'` role returns the *fetch reach*
  (deliberately `< 0.5 × step`) and would drop on-grid geometry. **This is the dominant real case** —
  a mesh's hidden dimensions are almost always time or channel.
- **Continuous hidden spatial dims** → `step × meshSlabTolerance`, default `1.0` (one cell), exposed
  via `ToleranceOptions` as the mesh sibling of `gsplatsDefaultTolerance`.

Be honest about what the second bullet means: with per-vertex cull there is no such thing as a true
cut, so a continuous hidden dimension renders a **thick slab** ("the surface near this slice"), not a
planar section, and the slab thickness is the only control. Exact nD clipping (§9) is the fix; until
then a mesh whose hidden dims are continuous and spatial is a poor fit for this node type, and the
loader should say so once, by name.

### 5.3 Consequence, stated plainly

A surface cut by a slice shows a **ragged, triangle-quantized boundary** rather than a clean planar
cut. For a well-tessellated mesh sliced with a tolerance comparable to the edge length this reads as a
slightly jagged edge; for a coarse mesh with a thin tolerance it can drop whole regions. This is a real
visual limitation and must be documented in the user guide, not glossed.

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
`drawElements` never fetches an unreferenced vertex, so culled vertices cost nothing to draw, and the
mesh is resident in full anyway (§7). This deliberately avoids:

- `compact_by_mask` (`wasm/rust/src/projection.rs:122`), which is **`&[f32]`-only** and could not
  compact the `uint8`/`uint16` colors or `float16`/`uint8` scalars §3.2 permits without a widening pass;
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
persistent form that needs no per-rebuild post-pass. When the displayed set is a **different triple**
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
| `color` | 3 or 4 | `colors` — RGB or RGBA, keep the native dtype (§6.1.1); a 4th component is a per-vertex opacity carried through an interpolated `vAlpha` (§6.2) |
| `aScalar` | 1 | `scalars`, when `has_scalars` |
| index | — | `compact_visible_faces` output — **the only buffer rewritten on a slice change** (§5.4); `position` (and `normal`) are additionally rewritten on a `displayDims` change (§7, §3.4) |

Drawn as `THREE.Mesh` with `side: DoubleSide` when `double_sided`, else `FrontSide`.

#### 6.1.1 Keep color/scalar dtypes native

Bind `uint8`/`uint16` colors with `new THREE.BufferAttribute(u8, itemSize, /* normalized */ true)` —
`itemSize` is the native `3` (RGB) or `4` (RGBA) — rather than widening to `Float32Array`. The GPU
normalizes to `[0,1]` for free, and this avoids a 4× memory blow-up on the single largest optional
attribute. This mirrors the existing loader doctrine — `LoadedLinesData` and `LoadedPointsData` both keep
`Uint8Array | Uint16Array | Float32Array` colors and pay a single widening only where a kernel demands
`f32` — and §5.4 removed the one place mesh would have needed `f32` (the compaction pass).

There is **one** `color` attribute regardless of component count, and hence **one** shader that reads it
as a `vec4`: a size-3 (RGB) attribute read as a `vec4` yields `w = 1.0` by the GL attribute default, so
RGB data carries a per-vertex opacity of `1.0` for free — the same *"1.0 for RGB data"* contract the
gsplat/line shaders document (`materials/gsplat/shader-glsl.ts`, `materials/line/shader-glsl.ts`), though
mesh is the first shader to obtain that `1.0` from the **GL size-3-attribute default** rather than by
packing it CPU-side (the siblings write `1.0` into their element texels). No separate RGB-vs-RGBA material
variant is needed; the alpha handling in §6.2 is unconditional.

### 6.2 Shading

The other three geometry types are purely emissive and have no lighting whatsoever. A mesh without
shading is a flat silhouette and effectively unreadable, so mesh is the first type to shade. The v1
model is deliberately minimal and light-free:

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
- **Shade term:** a camera-anchored headlight with a wrap term,
  `shade = mix(uAmbient, 1.0, pow(saturate(dot(N, V) * 0.5 + 0.5), uShadeExponent))`. View-anchored, so
  it needs no light in the scene graph and no scene-graph API change. `uAmbient` and `uShadeExponent`
  are material uniforms with sane defaults; a fully-flat `uAmbient = 1.0` reproduces the emissive look
  of the other types.
- **Base color:** vertex `color`, or the colormap LUT applied to `aScalar` under `USE_COLORMAP` — the
  same `getColormapTexture` / `updateScalarRange` path `createLinesNode` uses, including the same
  fail-closed guard when `colormap` is set without `has_scalars`.
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

  The §6.2 **shade** factor is a lighting term that multiplies the RGB base color only; it must **not**
  enter `a`. Emission then branches on the blending mode's `shaderOutputMode` (`blending-state.ts`),
  mirroring the line shader's fragment tail (`materials/line/shader-glsl.ts`):

  - `additive` / `luminous` / `normal` → **alpha-weighted** (`SrcAlpha/One` or `SrcAlpha/OneMinusSrcAlpha`
    apply `a` at composite): emit `fragColor = vec4(shadedColor, a);`.
  - `max` → **rgb-contribution**: `MaxEquation + OneFactor/OneFactor` does **not** weight source RGB by
    alpha at composite, so premultiply by coverage — emit `fragColor = vec4(shadedColor * a, a);` — exactly
    the line shader's `LUXAR_MAX_RGB_CONTRIBUTION` branch (`vec4(gammaColor * a, a)`).
  - `opaque` (mesh default) → a hard alpha **cutout**, see below.

  ⚠️ For `normal`, the unsorted-translucency caveat of §6.3 (no per-triangle depth sort in v1) **compounds**
  with per-vertex alpha: partially-transparent authored vertices make the missing sort visible, not just a
  uniform `opacity < 1`.

  The max-premultiply and the opaque-cutout emissions are distinct per-mode shader variants — a GLSL
  `#define` exactly like the siblings' `LUXAR_MAX_RGB_CONTRIBUTION` branch (and a graph-baked TSL twin) —
  so each is a separately compiled shader that carries its **own** codegen snapshot (§6.4), not one free
  runtime branch. (Compile-time `#define` and runtime-uniform branches coexist in the shipped materials —
  e.g. gsplat's opaque/peak split is a runtime `uProjectionMode` branch that TSL bakes per graph — but
  either way the harness snapshots each mode separately, which is the point here.)
- **`opaque` (the mesh default) → alpha is a hard cutout, not smooth transparency.** Decision, stated
  rather than left silent: `opaque` is depth-writing and order-independent (`shaderOutputMode: 'opaque'`,
  `blending-state.ts`), which is precisely why it is the only mode unconditionally correct without
  per-triangle sorting (§6.3) — and smooth partial transparency is contradictory there. So under `opaque`
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
  (and accepts its §6.3 unsorted caveat).

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

### 6.3 Blending and depth

v1 supports `opaque`, `normal`, `additive`, `luminous`, and `max`.

`volumetric` is **not supported**: it is an emission–absorption model over per-element optical depth and
has no meaning for an opaque surface. The observable behavior is a **one-time warning naming the node,
then an `opaque` fallback**, applied in `createMeshNode` — a warning rather than a load failure, for the
inheritance reason at the end of this section.

`normal` on a mesh is drawn **without per-triangle depth sorting** in v1. The depth-sort coordinator
sorts *instances* via `aSortedIndex`; the mesh analog is permuting triangle triples in the index
buffer, which is a natural but separate extension (§9). Until then:

- `opaque` (the default for mesh, unlike the other types) depth-tests and depth-writes, and is
  therefore correct;
- `normal` with `opacity < 1` **or per-vertex RGBA alpha present** (either makes the surface
  translucent, §6.2) may show incorrect inter-triangle ordering, and the loader logs a one-time warning
  naming the node.

Making `opaque` the mesh default is a deliberate asymmetry — it is the only mode that is unconditionally
correct without sorting, and it is what a surface should look like.

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

Note that `material-manager.ts` still declares `pointMaterialCache` / `lineMaterialCache` /
`gsplatMaterialCache`; these are **vestigial and permanently empty**, kept only so `getCacheStats()`
keeps its shape. Mesh must **not** add a fourth empty map — `createMeshMaterial` constructs directly.

Both backends must produce matching output and are gated by the existing codegen snapshot harness
(`src/tests/__codegen__/`), which keys one snapshot variant per blend-mode build — whether a GLSL
`#define` (the sibling `line-max`, `point-max`, `gsplat-normal-premult`) or a runtime-uniform branch the
TSL path bakes per graph (`gsplat-opaque`, from gsplat's runtime `uProjectionMode` split). Note the
harness (`tsl-codegen-snapshot.spec.ts`) asserts **both stages** of every variant unconditionally, so
each variant is a `.vertex` + `.fragment` snapshot pair — the shipped inventory is exactly 24 such pairs.
Mesh's per-mode emissions (§6.2) are therefore separately snapshotted — and note the mesh **default is
`opaque`**, unlike the siblings whose default is the alpha-weighted `additive`. New variants — six, i.e.
twelve snapshot files: `mesh` (the `opaque` default — alpha cutout, §6.2), `mesh-additive` (the
alpha-weighted emission shared by `additive`/`luminous`/`normal`, §6.2), `mesh-max` (the max
premultiply, §6.2), `mesh-flat-normal`, `mesh-colormap`, `mesh-pick`.

> **TSL house rule** (from the depth-sorting spec's remediation): both vertex stages must trace inside
> `Fn()` with explicit `.toVar()` statements, and the fragment must reconstruct the bottom-left
> fragcoord as `vec2(x, screenSize.y - y)` if it reads screen coordinates at all. The mesh fragment
> shader does **not** need fragcoord, which sidesteps that trap entirely.

### 6.5 Picking

Standard mechanism: `pickingSystem.allocatePickId()`, a shadow `THREE.Mesh` sharing the same
`BufferGeometry` with the pick material, registered via `pickingSystem.registerNode` — exactly as
points / lines / gsplats do it.

v1 picks at **vertex granularity**. Mesh has no depth sort and therefore no sorted-index indirection
(§9 defers per-triangle sorting), so — unlike the other three types — the mesh pick vertex shader does
**not** bind `aSortedIndex` and does **not** call the shared `luxarElementIdParts()` helper (which reads
`aSortedIndex`). Instead it splits `uint(gl_VertexID)` into low/high 16-bit halves exactly as that
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

**Pick fragment output.** The `mesh-pick.fragment` writes the same shared vec4 the readback decodes —
`vec4(vNodeId, vElementId.x, brightness, vElementId.y)` — with `brightness` the fragment's
coverage/opacity (1.0 for a fully opaque mesh), so the cross-node brightness-weighted vote still has a
value. Depth is the **opaque-surface** case, not the brightness-as-depth one: an opaque mesh writes real
projected depth (`gl_FragDepth = gl_FragCoord.z`, i.e. leaves the default), matching gsplat's
surface-mode branch (`uSurfaceDepth == 1` in `rendering/picking/gsplat/shaders.ts`) rather than the
`1.0 - clamp(brightness, 0, 1)` branch that points / lines / translucent gsplats use — otherwise every
fully-opaque fragment collapses to depth 0 and the mesh neither self-occludes nor occludes other nodes
correctly in the shared pick buffer. See `rendering/picking/README.md`; the `mesh-pick.fragment` codegen
snapshot (§6.4) is the final authority.

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
mesh's `gl_VertexID` source is bounded only by `V`. The cap is therefore *enforced*: `n_vertices < 2^27`
at the §3.5 loader gate (fail the node with a `LoaderError`), invoking the stride's own house rule that
an unenforced bound is not a bound (the reason `MAX_PICK_NODE_ID` is checked at allocation,
`rendering/picking/picking-system/pick-render.ts`). Note that `pick-render.test.ts`'s existing headroom
test only pins the texture-*layout* maxima, so this vertex cap needs its own pin — a dedicated test
that the vote key stays exact up to the largest admitted vertex ordinal and that a mesh with
`n_vertices ≥ 2^27` is rejected with a `LoaderError`.

**Accepted v1 limitation.** `vElementId` is a `flat` varying, so within a triangle it resolves to that
triangle's **provoking vertex**, not the cursor's barycentric-nearest vertex. Hovering a triangle
therefore reports a well-defined vertex *of* that triangle. Barycentric-nearest-vertex resolution would
need a de-indexed pick geometry or per-corner attributes plus barycentrics — a follow-up, not v1.

**FACE granularity** (e.g. highlighting a whole triangle) is deferred: it needs either a de-indexed pick
geometry or a per-corner face-id attribute, **plus** a compacted→original face map to stay stable under
§5.4 compaction. It is a natural follow-up, pairing with the §9 per-triangle-sort / partition work.

---

## 7. Loading

v1 uses a **whole-node loader**: fetch `vertices`, `faces` and the optional attribute arrays in full,
decode, and hold them. No spatial index, no progressive refinement, no chunk-bounds query. Immediately
after decode the arrays get §3.5's loader-side structural validation, before anything reaches the §5.4
kernels.

Justification: meshes in this domain are typically ≤ a few million triangles and fit comfortably; the
dual-index machinery in `lines-spatial-index-loader.ts` (1010 LOC) exists because line datasets reach
tens of millions of vertices with a meaningful per-slice working set. A mesh's working set after a
`displayDims` change is the whole mesh regardless.

The loader still implements the standard `MeshDataLoader` interface (`loadMesh` / `updateView` /
`dispose` + the optional monitor surface), so a spatial-index implementation can be swapped in behind
it later with no caller change.

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
     `normal_dims === displayDims`, else omit so the shader's flat-normal fallback (§6.2) takes over.
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
| Loader | `data/mesh/mesh-loader.ts` |
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

**Python**

- [ ] `format-contract/contract.yaml` → `make gen-contract` (regenerates both projections)
- [ ] `typing_utils/enums.py` — `NodeType.MESH`
- [ ] `core/mesh.py`, `core/group/adders/mesh.py`
- [ ] `core/group/group.py`, `core/scene/scene.py`, `core/__init__.py`, `luxar/__init__.py`
- [ ] `core/group/dim_order.py` — `add_mesh` calls `apply_dim_order_positions` for `vertices` like the
      other three adders; `faces` is index data and is **not** reordered
- [ ] `io/_compiler/geometry_writers/mesh.py`, `io/_compiler/node_common.py` (`MESH_RESERVED_ATTRS`)
- [ ] `io/compiler.py` (`write_mesh` facade), `io/reader.py` (`MeshData`/`get_mesh`/`list_meshes`)
- [ ] `validation/base.py` (`validate_faces_for_writing`, `validate_normals_for_writing`)
- [ ] `cli/info_command.py`
- [ ] **Partition rejection** — `core/node/specialized_groups.py:62`: the `display_type` guard admits
      exactly `("points", "lines", "gsplats")`. Leave `mesh` **out** of it, and extend the error message
      to say why (mesh has no LOD/partition support yet) rather than just listing valid values.
- [ ] **LOD rejection** — `core/group/lod/group.py`: ⚠️ **there is no equivalent whitelist.**
      `compute_lod_display_type` simply returns `resolve_display_type(children[-1])`, and
      `resolve_display_type` falls through to `node.attrs.get("type", "group")` for a plain leaf. A mesh
      child would therefore be **silently accepted** and produce a `kind=lod` group with
      `display_type="mesh"` that no viewer path can load. An explicit reject must be **added** to
      `compute_lod_display_type` (or to `Node.add_lod_group`) — this is a new guard, not a
      leave-mesh-out-of-an-existing-list edit. Cover it with a test asserting the raise.
- [x] `io/_compiler/finalize/lod_backfill.py` — **already handled by #1079.** `resolve()` now tests
      `t in GEOMETRY_TYPES` instead of a hardcoded tuple, so `mesh` is recognised as a leaf the moment
      it enters the contract, with no edit here. All four child-iteration sites also moved to
      `group_keys()`, which removes the failure this spec previously described: a leaf group's
      `keys()` lists its *arrays* (`['faces', 'vertices']`), so the old code could recurse into a zarr
      `Array` and raise a bare `AttributeError` mid-finalize.

**TypeScript**

- [ ] `types/mesh.ts`, `types/index.ts`, `types/window.d.ts`, `data/data-loader-types.ts`
- [ ] `types/data-monitor-types.ts` — **delete** the local `| 'mesh'` extension (now in the contract)
- [ ] `data/scene-loader/loaders/loader-registry.ts` — one line in `LoaderByKind`. The three parallel
      maps became a single kind-keyed store in #1079; `getLoaderType` / `disposeAll` / the counters are
      one implementation each. Omitting the entry is a **compile error**
      (`TS2339: Property 'mesh' does not exist on type 'LoaderByKind'`), not a silent gap
- [ ] `data/scene-loader/geometry-descriptors.ts` — one row in `GEOMETRY_DESCRIPTORS`, carrying
      `loadNode`, `applyPartialExtendTolerance`, `retryCommit` and the two loader factories. This is
      the row that `load-scene-nodes`, `lifecycle/retry` and `prefetch/slice-prefetcher` all read, so
      those three files need **no mesh edit at all**. A missing row fails the build with
      `TS2741: Property 'mesh' is missing … required in type 'Record<GeometryTypeName, …>'`
- [ ] `data/scene-loader/loaders/loader-factory.ts`, `nodes/build-scene-graph.ts`
      (bare-leaf-root union), `nodes/build-ctx.ts` (the `processMeshData` / `commitMeshGeometry`
      pair, matching the uniform shape #1099 gave all three existing types)
- [ ] `data/scene-loader/monitor/monitor-wiring.ts`, `scene-graph-converter.ts`,
      `data/scene-loader-monitor-port.ts`
- [x] `data/scene-loader/lifecycle/retry.ts` — **no edit needed.** The `else if (gsplatsLoader)` chain
      became a descriptor lookup in #1099. The hazard this spec flagged — a missing arm meaning a
      failed mesh load could never be retried, with nothing in the type system to say so — is now a
      build failure at the descriptor table instead
- [ ] `data/scene-loader/prefetch/slice-prefetcher.ts` — **no dispatch edit**, but mesh has no
      meaningful slice prefetch in v1 (whole-node resident, §7). The prefetcher is driven by three
      hardcoded `prefetchNode(path, 'points'|'lines'|'gsplats', …)` call sites, so mesh is excluded by
      simply not adding a fourth — confirm that stays true rather than assuming it
- [ ] `data/loaders/spatial-query/tolerance-computer.ts` — add the `mesh` arm to
      `computeHiddenDimTolerance` and `meshSlabTolerance` to `ToleranceOptions` (§5.2.1). Callers must
      pass `discreteRole: 'membership'`. **Do not** default the spatial arm to Lines' `0`
- [ ] `wasm/types.ts` — declare both new kernels on the `WasmModule` interface (§5.4); `pickBackend`
      swaps the module wholesale, so a kernel missing from either backend breaks the `>16D` path
- [ ] `data/attrs-composer.ts`, `data/stats/{aggregator,scene-stats}.ts`
- [ ] `rendering/mesh-geometry.ts`, `rendering/node-factory.ts`,
      `rendering/node-factory/create-mesh-node.ts`
- [ ] `rendering/materials/mesh/{material,shader}-{glsl,tsl}.ts`, `rendering/picking/mesh/*`
- [ ] `rendering/material-manager/factories.ts`, `rendering/material-colormap-helpers.ts`
- [ ] `scene/scene-manager/camera/camera-framing.ts`, `scene/lod-freshness.ts`,
      `scene/synthetic-scene.ts`
- [ ] `ui/layers/{layer-apply,layer-state,layers-panel}.ts`
- [ ] `ui/data-loading-monitor.ts` + `data-loading-monitor/{templates,advisor}.ts`
- [ ] `core/app/debug/{debug-interface,debug-state}.ts`

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
| `types/lod-group.ts:31` — `display_type` union | Mesh is excluded from `kind=lod` (§9) |
| `types/partition-group.ts:48` — `display_type` union | Mesh is excluded from `kind=partition` (§9) |
| `rendering/gpu-buffer-pool/pool-stats.ts:22` — `type` union | Mesh doesn't use the buffer pool (§2.1) |
| `data/loaders/spatial-query/spatial-query-builder.ts` — chunk-query construction | No spatial index in v1 (§7). **Note:** this is the *query builder* only — `tolerance-computer.ts` in the same folder **does** need a mesh arm (§5.2.1); don't let the shared folder mislead you |
| `ui/layers/absorption-range.ts:90,98` | Mesh doesn't support `volumetric` blending — warn + `opaque` fallback (§6.3) |

A reviewer should treat a `| 'mesh'` appearing in any of those five as a defect.

**Rust / WASM**

- [ ] `wasm/rust/src/mesh_culling.rs` + `lib.rs` registration
- [ ] `wasm/typescript/mesh-culling.ts` + `index.ts` registration (also the production `>16D` backend)
- [ ] Rust↔TS parity tests

**Docs**

- [ ] `docs/guides/user/LUXAR_ZARR_FORMAT.md` — mesh node layout
- [ ] `CLAUDE.md` — geometry-types line, "three first-class geometry types" → four
- [ ] Package READMEs: `core/`, `io/_compiler/geometry_writers/`, `data/mesh/`,
      `rendering/materials/mesh/`, `rendering/picking/mesh/`
- [ ] `CHANGELOG.md`

**Tests**

- [ ] Python: writer round-trip, validators (incl. every rejection in §3.5), authoring lint,
      broadcast color/scalar, `extend_to_all`, reader
- [ ] Rust: `mesh_vertex_visibility_mask` / `compact_visible_faces` unit tests incl. the non-finite rule
- [ ] TS unit: loader, geometry assembly, cull correctness, colormap fail-closed guard,
      Rust↔TS kernel parity, corrupt-store rejection (out-of-range face index → `LoaderError`, not a
      WASM trap; an undersized `normals`/`colors`/`scalars` array → `LoaderError`, §3.5; `n_vertices ≥
      2^27` → `LoaderError`, its own pin since `pick-render.test.ts` only covers the texture-layout
      maxima, §6.5), and the `volumetric`→`opaque` fallback warning (§6.3)
- [ ] TS unit (alpha chain, §6.2): an **RGBA** mesh produces **different** fragment output than the same
      mesh RGB-only (goes red if `vAlpha` is dropped — the exact "(V,4) renders like (V,3)" defect); under
      `opaque`, fragments with `a < uAlphaCutoff` are **discarded** (cutout) and survivors write alpha 1.0;
      under `max`, the emitted RGB is **premultiplied** by `a` (`vec4(shadedColor * a, a)`). Each must be
      verified to fail before the fix.
- [ ] TS unit: on a **`double_sided: false`** mesh, `updateView` for a `displayDims` change `[0,1,2]`→`[0,2,1]`
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
- [ ] TS unit: **flat-vs-smooth on the same normal-bearing mesh** — one mesh with valid stored normals
      (`normal_dims == displayDims`) selects the stored-normal `mesh.fragment` variant under
      `shading="smooth"`, and the SAME mesh under `shading="flat"` selects `mesh-flat-normal.fragment`
      and shades from screen-space derivatives (§3.4, §6.2). Verified to differ — a build that ignores
      `shading` renders both identically and the test goes red
- [ ] Codegen snapshots: 6 new variants = 12 files (§6.4) — incl. the per-blend-mode
      `mesh-additive`/`mesh-max` variants
- [ ] Fixture: `tests/fixtures/generate_test_data.py` gains a mesh fixture (auto-picked up by
      `vitest.config.ts` globalSetup)
- [ ] E2E: one `mesh-rendering.spec.ts`, plus extend the existing multi-geometry
      `tests/e2e/geometry-types.spec.ts` (it asserts `userData.nodeType` per type and already covers
      lines + gsplats) with a mesh case
- [ ] One demo exercising the type end to end

> Every test must be verified to **fail before the fix** — mutate the implementation and confirm the
> test goes red. A cull test that passes against an all-ones mask is vacuous.

---

## 9. Explicitly out of scope

Each of these is a deliberate exclusion, not an oversight. Each should surface clearly — an error, or
for `volumetric` the named one-time warning + `opaque` fallback of §6.3 — rather than silently misbehave.

| Excluded | Why | Natural follow-up |
|---|---|---|
| **LOD / decimation** | The additive/substitutive ladder machinery assumes independent elements. The mesh analog is QEM decimation — a project, not a line item. | `luxar mesh lod` with QEM levels feeding the existing `kind=lod` group |
| **`kind=partition`** | Cheap in principle (BSP over face centroids) but needs vertex duplication at part boundaries. | The **first** follow-up — highest value for large meshes |
| **Exact nD triangle clipping** | ~1500 LOC across two backends. §5 covers the dominant real case (hidden dims are discrete — time/channel) for ~10% of the cost, but gives only a **thick slab**, never a true cut, when a hidden dim is continuous and spatial (§5.2.1). | Slot in behind the same `MeshDataLoader.updateView`; the mask kernel becomes the fast pre-pass. **Promote this if continuous hidden spatial dims turn out to be a real use case** |
| **Per-triangle depth sorting** | Index-buffer permutation, not instance permutation. | Extend the depth-sort coordinator with an index-permutation path |
| **Spatial index** | See §7. | Mirror the lines dual-index loader over faces |
| **`volumetric` blending** | No meaning for an opaque surface. | — |
| **Worker projection** | Measure first (§7). | — |
| **Mesh import formats** (PLY/OBJ/STL/glTF) | Independent of the node type. | `luxar mesh import`, mirroring `gsplat import` |

**Pre-existing gap noticed during this spec's review, not introduced by mesh:** none of
`POINTS_/LINES_/GSPLATS_RESERVED_ATTRS` includes `has_image_labels`, though all three writers stamp it.
No clobber results — `validate_render_attrs`'s reject-unknown gate already fails such a write, just with
the *unknown-attr* message instead of the *reserved* one — so this is an error-message gap.
`MESH_RESERVED_ATTRS` includes the key from the start (§3.3); aligning the three sibling sets is a
separate change so it isn't buried in the mesh diff.

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
fail the build rather than going quiet:

- **Missing loader entry** → `TS2339: Property 'mesh' does not exist on type 'LoaderByKind'`
  at `loader-registry.ts`, plus three `TS2536` follow-ons inside the generic accessors
- **Missing descriptor row** → `TS2741: Property 'mesh' is missing … required in type
  'Record<GeometryTypeName, GeometryDescriptor>'`
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
| **0** ✅ *(done — §10.1)* | Single-source `GeometryKind` from the contract; collapse `LoaderRegistry`; unify the per-type pipeline; table-drive the dispatch switches | Landed as #1079 / #1099 / #1150, all behaviour-preserving. Deleting the `SceneGraphNodeType` local extension is the one piece left, and belongs with Phase 1's contract edit |
| **1** | Contract + `NodeType.MESH` + `core/mesh.py` + adder + writer + validators + reader + `info` + the LOD/partition rejections (§8) | `scene.add_mesh(...)` writes a `.luxar.zarr`; `luxar info --stats` reports it; round-trip test green; a mesh child of a lod/partition group **raises** |
| **2** | Rust + TS cull kernels with parity tests | Kernels green in isolation, no viewer changes |
| **3** | `types/mesh.ts` + loader + node load + `mesh-geometry.ts` + one `LoaderByKind` entry + one `GEOMETRY_DESCRIPTORS` row | Mesh loads and renders **unshaded** (flat vertex color); E2E smoke green |
| **4** | GLSL + TSL material pair + codegen snapshots + shading model | Shaded surface, both backends pixel-equivalent |
| **5** | Picking pair, layers panel, monitor, stats, camera framing, debug | Full parity with the other three at the UI level |
| **6** | Fixture + E2E spec + demo + docs + CHANGELOG | Shippable |

Phase 0 landed alone, with no mesh code, so any regression it caused would have been unambiguous.
Phases 1–2 are independent and can land in parallel. Phase 3 used to be the widest diff — a sweep
across every dispatch site — and is now one of the narrower ones: the three dispatch files need no
mesh edit, leaving the loader, the geometry builder, and two table entries. Phase 4 is the deepest
single piece of work.

**Estimate:** 5–6 PRs, roughly 4–5.5K LOC including tests — against ~14K LOC for the full Lines
vertical, the difference being everything in §9 plus the dispatch work Phase 0 already absorbed.
