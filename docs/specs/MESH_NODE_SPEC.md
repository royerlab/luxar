# Mesh Node Specification

**Status:** Proposed
**Scope:** A fourth first-class geometry type — `mesh` — symmetric to Points, Lines and GSplats.
**Non-goals:** LOD/decimation, spatial indexing, exact nD triangle clipping. See [§9](#9-explicitly-out-of-scope).

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
| nD slicing | ⚠️ Partial | Reuses the slab semantics, **not** the clipping algorithm — see §5 |
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
node_types: ["scene", "group", "points", "lines", "gsplats", "mesh"]
```

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
| `normals` | float32 | `(V, D')` | no | `COORDINATE` | Per-vertex, over the 3 display dims only — see §3.4 |
| `colors` | uint8/uint16/float32 | `(V, 3\|4)` | no | color helpers | RGB or RGBA (alpha = per-vertex opacity) |
| `scalars` | float32/float16/uint8 | `(V,)` | no | scalar helpers | Colormap lookup |
| `label_offsets`/`label_bytes` | — | CSR | no | — | Per-vertex hover tooltips |
| `image_label_*` | — | CSR | no | — | Per-vertex hover thumbnails |

`faces` uses `SemanticType.INDEX` with `deduplicate=False` and `allow_lut=False`, for exactly the
reason `Lines.segments` does: the loader reads it as raw chunked zarr and does not resolve `array_ref`,
so dedup would silently drop geometry for a byte-identical sibling, and LUT encoding of grid-snapped
values would decode as garbage topology.

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
has_colors: bool
has_scalars: bool
has_labels: bool
has_image_labels: bool
shading: "smooth" | "flat"      # default "smooth" when normals present, else "flat"
double_sided: bool              # default true
position_bounds: [[min...], [max...]]
ordering: "none"                # v1 always; reserved for a future spatial index
```

plus the standard render attrs already handled by `apply_default_render_attrs` and
`prepare_transform_attrs` (`opacity`, `gamma`, `intensity`, `offset`, `absorption`, `blending_mode`,
`colormap`, `scalar_data_range`, `layer`, `transform`, `nd_transform`, `extend_to_all`).

`MESH_RESERVED_ATTRS` is added to `io/_compiler/node_common.py` alongside the other three frozensets.

### 3.4 Normals are 3D, positions are nD

Positions live in nD like every other geometry type. Normals are a **display-space** quantity: they are
only meaningful for the three displayed dimensions, and re-deriving them per slice change is the
correct behaviour when `displayDims` rotates.

Therefore: `normals` is stored as `(V, 3)` in the order of the node's **first three dimensions**, and is
treated as valid only while `displayDims == [0, 1, 2]`. For any other `displayDims`, or when `normals`
is absent, the viewer computes **flat face normals** from the projected triangle via a cross product
(§6.2). This is a deliberate simplification, documented in the writer's docstring and warned about at
load time when a non-default `displayDims` is combined with stored normals.

An alternative — storing a full `(V, D, 3)` normal frame — was rejected as over-engineering for v1.

### 3.5 Validation

New shared validators in `luxar/validation/base.py`, following the existing `validate_*_for_writing`
convention (fail-fast, before any zarr group is created):

- `validate_faces_for_writing(faces, n_vertices)` — shape `(F, 3)` or flat `(3F,)`; integer dtype
  (reject float, which `.astype(np.uint32)` would silently truncate); `min >= 0`; `max < n_vertices`;
  `F >= 1`. Mirrors the `line_type='indexed'` index gate at `geometry_writers/lines.py:134-170`, which
  is the closest precedent and already encodes each of these traps.
- `validate_normals_for_writing(normals, n_vertices)` — shape `(V, 3)`, finite. Zero-length normals are
  **warned**, not rejected (degenerate triangles legitimately produce them), and are renormalized to the
  flat face normal at render time.

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
  `EXTEND_TO_ALL_TOLERANCE` path, unchanged;
- tolerance itself comes from the existing `computeTolerance` / `tolerance-computer.ts`, unchanged.

This is precisely the `p1_in` branch of `clip_segment_single`, applied per vertex and AND-ed across the
three vertices of a face.

### 5.3 Consequence, stated plainly

A surface cut by a slice shows a **ragged, triangle-quantized boundary** rather than a clean planar
cut. For a well-tessellated mesh sliced with a tolerance comparable to the edge length this reads as a
slightly jagged edge; for a coarse mesh with a thin tolerance it can drop whole regions. This is a real
visual limitation and must be documented in the user guide, not glossed.

It is the right v1 trade: it costs **~140 LOC of new kernel** instead of ~1500, requires no
re-triangulation, no per-frame index rebuild, and no attribute interpolation machinery.

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
/// Writes remapped (compacted-vertex-space) indices into `output`.
pub fn compact_visible_faces(
    faces: &[u32], vertex_mask: &[u8], vertex_remap: &[u32],
    num_faces: usize, output: &mut [u32],
) -> u32;
```

`mesh_vertex_visibility_mask` calls `validate_ndim` like its siblings and therefore panics above 16D;
`pickBackend(ctx, ndim)` in `workers/data-worker/state.ts` routes `ndim > 16` to the TypeScript
implementation, as it already does for every other kernel. No change to the routing logic is needed —
only registration of the two new names.

Vertex compaction reuses the existing `compact_by_mask` (`wasm/rust/src/projection.rs:122`) for the
per-vertex attribute arrays.

### 5.5 Fast path

When `displayDims.length === ndim` (no hidden dimensions — the common 3D case), the mask is trivially
all-ones and the whole cull is skipped: positions are extracted once via the existing
`extract_3d_positions` and the index buffer is uploaded verbatim. Only a `displayDims` change or a
non-3D dataset triggers the cull path.

---

## 6. Rendering

### 6.1 Geometry

`rendering/mesh-geometry.ts` builds a plain `THREE.BufferGeometry`:

| Attribute | Size | Source |
|---|---|---|
| `position` | 3 | `extract_3d_positions(vertices, displayDims)` |
| `normal` | 3 | stored normals when valid (§3.4), else omitted |
| `color` | 3 or 4 | `colors`, normalized to float |
| `aScalar` | 1 | `scalars`, when `has_scalars` |
| index | — | `compact_visible_faces` output |

Drawn as `THREE.Mesh` with `side: DoubleSide` when `double_sided`, else `FrontSide`.

### 6.2 Shading

The other three geometry types are purely emissive and have no lighting whatsoever. A mesh without
shading is a flat silhouette and effectively unreadable, so mesh is the first type to shade. The v1
model is deliberately minimal and light-free:

- **Normal source:** the `normal` attribute when present and valid; otherwise a flat normal derived in
  the fragment shader from screen-space derivatives of the view position
  (`normalize(cross(dFdx(vViewPos), dFdy(vViewPos)))`). The derivative fallback means a mesh with no
  stored normals still shades correctly, and it is what makes §3.4's "recompute on non-default
  displayDims" cheap.
- **Shade term:** a camera-anchored headlight with a wrap term,
  `shade = mix(uAmbient, 1.0, pow(saturate(dot(N, V) * 0.5 + 0.5), uShadeExponent))`. View-anchored, so
  it needs no light in the scene graph and no scene-graph API change. `uAmbient` and `uShadeExponent`
  are material uniforms with sane defaults; a fully-flat `uAmbient = 1.0` reproduces the emissive look
  of the other types.
- **Base color:** vertex `color`, or the colormap LUT applied to `aScalar` under `USE_COLORMAP` — the
  same `getColormapTexture` / `updateScalarRange` path `createLinesNode` uses, including the same
  fail-closed guard when `colormap` is set without `has_scalars`.
- **Tail:** the shaded color then goes through the standard `intensity` → `offset` → `gamma` →
  `opacity` → blending-mode output chain.

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

`volumetric` is **rejected at load** with a clear message: it is an emission–absorption model over
per-element optical depth and has no meaning for an opaque surface.

`normal` on a mesh is drawn **without per-triangle depth sorting** in v1. The depth-sort coordinator
sorts *instances* via `aSortedIndex`; the mesh analog is permuting triangle triples in the index
buffer, which is a natural but separate extension (§9). Until then:

- `opaque` (the default for mesh, unlike the other types) depth-tests and depth-writes, and is
  therefore correct;
- `normal` with `opacity < 1` may show incorrect inter-triangle ordering, and the loader logs a
  one-time warning naming the node.

Making `opaque` the mesh default is a deliberate asymmetry — it is the only mode that is unconditionally
correct without sorting, and it is what a surface should look like.

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
(`src/tests/__codegen__/`) — new snapshots: `mesh.vertex`, `mesh.fragment`, `mesh-flat-normal.fragment`,
`mesh-colormap.fragment`, `mesh-pick.{vertex,fragment}`.

> **TSL house rule** (from the depth-sorting spec's remediation): both vertex stages must trace inside
> `Fn()` with explicit `.toVar()` statements, and the fragment must reconstruct the bottom-left
> fragcoord as `vec2(x, screenSize.y - y)` if it reads screen coordinates at all. The mesh fragment
> shader does **not** need fragcoord, which sidesteps that trap entirely.

### 6.5 Picking

Standard: `pickingSystem.allocatePickId()`, a shadow `THREE.Mesh` sharing the same `BufferGeometry` with
the pick material, registered via `pickingSystem.registerNode`. Simpler than lines/points — no
`aSortedIndex` indirection, so `vElementId` is just `gl_VertexID / 3` for the face id (or the vertex id,
depending on the picking granularity chosen; **face granularity** is proposed, matching the mesh's
natural element).

---

## 7. Loading

v1 uses a **whole-node loader**: fetch `vertices`, `faces` and the optional attribute arrays in full,
decode, and hold them. No spatial index, no progressive refinement, no chunk-bounds query.

Justification: meshes in this domain are typically ≤ a few million triangles and fit comfortably; the
dual-index machinery in `lines-spatial-index-loader.ts` (1010 LOC) exists because line datasets reach
tens of millions of vertices with a meaningful per-slice working set. A mesh's working set after a
`displayDims` change is the whole mesh regardless.

The loader still implements the standard `MeshDataLoader` interface (`loadMesh` / `updateView` /
`dispose` + the optional monitor surface), so a spatial-index implementation can be swapped in behind
it later with no caller change.

`updateView` recomputes the visibility mask and index buffer (§5) and returns; on the fast path (§5.5)
it is a no-op returning the cached data.

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

Adding a fourth type means touching every hardcoded dispatch site. There is no plugin registry — the
registry holds three explicit `Map`s and `build-scene-graph` hardcodes the type union.

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
- [ ] `core/node/specialized_groups.py:62` — the `display_type` guard currently admits exactly
      `("points", "lines", "gsplats")`. Leave `mesh` **out** of it, and extend the error message to say
      why (mesh has no LOD/partition support yet) rather than just listing valid values

**TypeScript**

- [ ] `types/mesh.ts`, `types/index.ts`, `types/window.d.ts`, `data/data-loader-types.ts`
- [ ] `types/data-monitor-types.ts` — **delete** the local `| 'mesh'` extension (now in the contract)
- [ ] `data/scene-loader/loaders/loader-registry.ts` — 4th map + `getLoaderType` + `disposeAll`
- [ ] `data/scene-loader/loaders/loader-factory.ts`, `nodes/load-scene-nodes.ts`,
      `nodes/build-scene-graph.ts` (bare-leaf-root union), `nodes/build-ctx.ts`
- [ ] `data/scene-loader/monitor/monitor-wiring.ts`, `scene-graph-converter.ts`,
      `data/scene-loader-monitor-port.ts`
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
      Rust↔TS kernel parity
- [ ] Codegen snapshots: 6 new (§6.4)
- [ ] Fixture: `tests/fixtures/generate_test_data.py` gains a mesh fixture (auto-picked up by
      `vitest.config.ts` globalSetup)
- [ ] E2E: one `mesh-rendering.spec.ts`
- [ ] One demo exercising the type end to end

> Every test must be verified to **fail before the fix** — mutate the implementation and confirm the
> test goes red. A cull test that passes against an all-ones mask is vacuous.

---

## 9. Explicitly out of scope

Each of these is a deliberate exclusion, not an oversight. Each should error clearly rather than
silently misbehave.

| Excluded | Why | Natural follow-up |
|---|---|---|
| **LOD / decimation** | The additive/substitutive ladder machinery assumes independent elements. The mesh analog is QEM decimation — a project, not a line item. | `luxar mesh lod` with QEM levels feeding the existing `kind=lod` group |
| **`kind=partition`** | Cheap in principle (BSP over face centroids) but needs vertex duplication at part boundaries. | The **first** follow-up — highest value for large meshes |
| **Exact nD triangle clipping** | ~1500 LOC across two backends. §5 buys 90% of the value for 10% of the cost. | Slot in behind the same `MeshDataLoader.updateView`; the mask kernel becomes the fast pre-pass |
| **Per-triangle depth sorting** | Index-buffer permutation, not instance permutation. | Extend the depth-sort coordinator with an index-permutation path |
| **Spatial index** | See §7. | Mirror the lines dual-index loader over faces |
| **`volumetric` blending** | No meaning for an opaque surface. | — |
| **Worker projection** | Measure first (§7). | — |
| **Mesh import formats** (PLY/OBJ/STL/glTF) | Independent of the node type. | `luxar mesh import`, mirroring `gsplat import` |

---

## 10. Phased delivery

| Phase | Contents | Verifiable outcome |
|---|---|---|
| **1** | Contract + `NodeType.MESH` + `core/mesh.py` + adder + writer + validators + reader + `info` | `scene.add_mesh(...)` writes a `.luxar.zarr`; `luxar info --stats` reports it; round-trip test green |
| **2** | Rust + TS cull kernels with parity tests | Kernels green in isolation, no viewer changes |
| **3** | `types/mesh.ts` + loader + node load + process + commit + `mesh-geometry.ts` + dispatch sweep | Mesh loads and renders **unshaded** (flat vertex color); E2E smoke green |
| **4** | GLSL + TSL material pair + codegen snapshots + shading model | Shaded surface, both backends pixel-equivalent |
| **5** | Picking pair, layers panel, monitor, stats, camera framing, debug | Full parity with the other three at the UI level |
| **6** | Fixture + E2E spec + demo + docs + CHANGELOG | Shippable |

Phases 1–2 are independent and can land in parallel. Phase 3 is the widest diff (the dispatch sweep) but
the shallowest per-file. Phase 4 is the deepest single piece of work.

**Estimate:** 5–7 PRs, roughly 4.5–6K LOC including tests — against ~14K LOC for the full Lines
vertical, the difference being everything in §9.
