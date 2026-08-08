# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### August 2026

#### CytoSelf hover no longer strands its tooltip in an empty slot

The CytoSelf demo's hover layout is a bespoke pair: an image thumbnail anchored
top-right at `0.98`, and the text label at `x = 0.82` so it sits immediately to
the panel's LEFT. Defining any `hover=True` overlay suppresses the compiler's
auto-injected default, so that pair owns the whole hover experience. But the two
halves were guarded independently, and the image half is the fragile one — a
single failed `Image_data*.npy` download, or the count-mismatch guard rejecting a
stale thumbnail bundle, dropped it while the text label stayed pinned at `0.82`.
The tooltip then rendered into the gap reserved for a panel that did not exist,
which read as hovering doing nothing at all.

The two shapes are now both spelled out. With thumbnails, the two-panel layout is
unchanged. Without them the label moves to the centre-left slot (`(0.02, 0.5)`,
`center-left`), which nothing else in this scene occupies — the legend is
center-RIGHT. The parameters are copied from `demo_chromatrace_choir_umap`, whose
overlay layout is otherwise identical and which is one of seven siblings already
using that slot for a text-only tooltip; the viewer's control rail is docked at
that same edge house-wide, and matching the siblings beats diverging from them.
Falling back to auto-injection would have been the smaller diff and the wrong
answer: that overlay is the same corner, only 16% of the viewport further into it.

The reporting around the loss got honest too, and it differs per path because the
remedies do. One silent skip became loud — no thumbnails at all used to say
nothing — and the terse count-mismatch line gained a remediation: delete the
cached `.npz` (the message prints its full path), since a plain re-run
short-circuits on that file before any network call and reproduces the mismatch
forever. `--recompute` now reaches `load_cytoself_images` as well, which is the
same rebuild from the CLI, but it also discards the cached UMAP for a 10-30 minute
recompute, so it is offered second and with that caveat attached. A download
failure names the `Image_data*.npy` file that failed, and says plainly that
re-running skips whole completed files rather than resuming a partial one. A
deliberate `--without-images` run stays quiet, and `main()`'s navigation hint no
longer promises fluorescence images the scene does not contain.

#### Mesh is per-triangle depth sorted

`normal`-mode meshes composited in index order: whichever triangle the writer
emitted last drew last, so faces showed through each other and — with per-vertex
RGBA at full node opacity, where `depthWrite` is on while `transparent` is true —
whatever was behind a translucent fragment was depth-REJECTED outright. The
mitigation shipped in #1328 was a one-time warning naming the node. This replaces
the warning with the fix, and deletes it.

Almost all of it is reuse. A triangle's "center" is its vertex centroid: 3 floats
per element, exactly like a splat center or a line segment midpoint. So the
registration (`noteDepthSortCommit`), the SortWorker, the sort kernel, the
generation/stale-drop bookkeeping and the per-frame camera-motion re-sort
scheduler all serve mesh unchanged, and `GEOMETRY_CAPABILITIES.mesh.depthSortable`
is now `true` — which is also what makes a Layers-panel switch INTO `normal`
reprocess the node so it registers.

The apply is the part that genuinely differs, and it differs structurally rather
than by degree. The three instanced types permute `aSortedIndex`, a per-instance
draw-slot indirection, streaming a new ordering into an inactive twin attribute
and flipping a uniform when it completes. A mesh has no indirection: it is one
indexed `drawElements`, and the draw order of its triangles IS the order of the
index buffer. `geometry.index` is BOUND state, so no uniform can select between
two of them, and reassigning it is the drawn-geometry rebind `applyMeshIndices`
exists to avoid. The new `depth-sort-coordinator/triangle-ordering.ts` therefore
writes the whole visible prefix ATOMICALLY — one `set`, one update range — because
a half-written index buffer is not a permutation: some triangles would draw twice
and others not at all, a wrong picture rather than a stale one. The cost is
bounded by a quantity the mesh path already pays, since `applyMeshIndices`
re-uploads that same prefix on every slice move.

Two things that took finding:

- **The permutation must be applied to the CANONICAL triples, not to the live
  buffer.** The buffer already holds the previous permutation, so permuting it
  again composes the two — invisible on the first sort, a scrambled surface on the
  second. The coordinator retains the commit's `ProjectedMeshData.indices` for
  exactly as long as the node is being sorted, and drops it on every release
  branch so an opaque mesh never carries a second copy of its index.
- **`commitMeshGeometry` has to stamp `committedData`.** Mesh has no
  memoized-concat noop path of its own, so it never set the stamp — and the
  coordinator's resolve path, per-frame scheduler and capture drain all gate on it.
  Unset, every sort still dispatched, still resolved, and was then silently
  dropped: the surface would have rendered unsorted with nothing anywhere
  reporting a problem.

Picking needed no work, and the reason is worth recording: for an indexed draw
`gl_VertexID` (WGSL `@builtin(vertex_index)`) IS the value fetched from the index
buffer, so a mesh's element ordinal is invariant under any permutation of the
triples. The slot-syncing the instanced types need to keep the pick pass reading
the same permutation has no analogue here.

Verified on a real written scene rather than in unit tests alone: two overlapping
half-opaque quads, green at z = +2 and red at z = −2, authored NEAR-first so the
index order is deliberately the wrong one. Sampling the centre of the frame,
`?depthSort=0` gives (r 199, g 153) — red wins, the far quad composited last, as
authored. Sorted gives (r 162, g 196) — green wins. Orbiting to the far side with
real mouse drags flips it back to (r 184, g 136). What sorting still cannot fix is
interpenetrating triangles, a residual shared with the other three types and the
reason `opaque` remains the mesh default.
#### Documentation — pull-request quality gate and warning ratchets (#776)

Documentation-relevant pull requests now report a stable `docs-quality` check.
It combines the existing README/docstring/JSDoc completeness baseline with a
message-level TypeDoc warning baseline and a warning-fatal Sphinx build. Broken
internal references fail deterministically; external HTTP link checking remains
an explicit opt-in audit with narrow reviewed exceptions. The developer guide
records measurable phases for reducing the remaining completeness and TypeDoc
debt to zero.

#### Tests — the line-joint artifact gets an automated acceptance measurement (#790)

The outer-side miter wedge existed only as a scratch script, and nothing
guarded the already-fixed joint defects (#780 bead chain, #785 straight-joint
notch) while the vertex stage is rewritten to close it. Two pure metrics now
score a rendered frame (`src/tests/helpers/line-join-metrics.ts`): a
**local-median outlier count** for narrow wedge ticks, and an **axial flux
profile** (cross-section sum along the tube, normalised by its own median) for
smooth per-joint dips. Both are needed — a 50% flux ramp at every joint scores
zero outliers, and a 1-pixel tick leaves the flux profile flat.

The new `test_line_joins.luxar.zarr` fixture puts five joint cases in separate
world-Y bands (120-segment smooth curve, 90° zigzag, thin and thick straights,
and a nine-ray indexed hub) under a pinned photometry-grade viewer config, and
`line-join-artifact.spec.ts` measures every band on one frame, locating each by
projecting its world AABB through the live camera. Every band is asserted to
have a gapless flux profile — a torn tube is a defect at any turn angle — and
the straight bands additionally at zero outliers and a flat profile. The two
bending cases are **recorded** under documented ceilings rather than fixed:
measured 2026-08-06, with the device pixel ratio pinned, at 4.94% dark /
3.53% bright on the curve. Those ceilings drop to zero when the join geometry
lands.

Scope note: the E2E job is currently disabled in CI, so the spec runs only
under `make test-e2e` locally. What runs on every PR is the unit suite, and it
pins the metrics themselves — their arithmetic, their degenerate-input
behaviour and their sensitivity envelope — not the renderer.

The local-median metric's sensitivity envelope is documented and pinned by unit
tests: it is non-monotone in defect width (1 px and 2 px counted, ≥ 3 px
invisible, because the defect poisons its own median), so its fraction is a
detector, not a severity measure, and not comparable between bands of different
turn angle.

#### Four geometry types, said consistently

`grep -ci mesh README.md` returned 0. The lead paragraph, the capabilities table
and the architecture diagram all still described a three-geometry system, and
`docs/concepts/architecture.rst` went further, listing "Triangle meshes" under
*consider alternatives — use three.js*. All now name mesh; the alternatives entry
is rewritten as the caveat that is actually true (Luxar renders surfaces but does
not author them, and a mesh gets no LOD or spatial partitioning, so one very
large surface loads whole). `README.md` gains a Mesh section under Geometry
Types covering `add_mesh`, the no-per-element-size and `opaque`-by-default
differences, optional normals, and the §9 exclusions; `docs/api/core.rst` gains
the missing `luxar.core.Mesh` autoclass entry; `CITATION.cff`'s abstract names
four types.

`MESH_NODE_SPEC.md`'s §8 integration checklist still showed most of the
TypeScript / Rust / test half unchecked, long after those phases shipped — every
box verified against the tree and ticked, with the header saying plainly that §11
is the delivery record and that the paths are historical. Its shader-variant
inventory was also two counts stale.

The rest is the same drift a layer down: three demo docstrings saying "two of
Luxar's three geometry types", the Python subpackage READMEs and module
docstrings, the viewer package READMEs, shared-helper and source comments
(picking system, data-loading monitor, layers material), the architecture
diagrams, the gsplats format spec, the nD-transform and intensity/gamma developer
specs, and the visualization skill. `INTENSITY_GAMMA_DESIGN.md` needed more than
a count: mesh does carry the GOG chain, but it deliberately omits the
zero-contribution early discard, because an opaque surface still has to write
depth for a fragment that ends up black.

Claims that count the *instanced-quad, depth-sorted, volumetric* families rather
than the type vocabulary were checked and deliberately left at three — mesh takes
part in none of those, and the `?debug` synthetic-scene bench injector genuinely
still builds only points, lines and gsplats.

#### Removed — unreachable accelerated gsplat code paths

Three optimized paths existed and were maintained but could not be selected by
any production call, so they read as capabilities the project did not actually
have. All three are removed:

- The Metal **raw splat-centric** kernel path (`MetalRawSplatFunction`, the
  `_uses_raw_custom_metal` gate, the `forward_raw_splat_3d` / `backward_raw_splat_3d`
  bindings and their `rasterize_*_raw_splat_centric_3d` kernels). It was gated on
  `amp_max is None`, but the production fit pipeline always sets `amp_max` (auto
  default `1.0`), and the raw kernel applies softplus to `raw_a` in-shader with no
  amplitude clamp, so it could never honour the bound. The non-raw splat-centric
  path still serves production and shares the same threadgroup-reduction backward
  (the raw-parameter reparameterization VJP is handled by PyTorch autograd); the
  earlier fwd+bwd micro-optimization measured on the raw variant is the only thing
  lost. The already-dead `validate_splat_L_tensors_3d` validator in the same
  bindings file went with it.
- The GPU seeding helpers `local_maxima_gpu` / `soft_blur_nd_gpu` in
  `gsplats/seeds/gpu_ops.py`, which had only test callers — the CPU siblings have
  no `device=` dispatch and `seed_from_peaks` does its own GPU intensity sampling
  rather than a maxima filter.
- The stale `gsplats/models/gsplats/cuda/setup.py` (and its `LUXAR_CUDA_ALL_ARCHS`
  env var), a twin of the real `cuda/build.py` build entry point that nothing
  referenced.

The Metal README performance table, the seeds GPU-operations docs, and the
`CLAUDE.md` GPU-support notes are updated to describe only the reachable paths.

#### Mesh renders shaded — the fourth geometry type gets its material pair

Until now a mesh loaded and drew with flat per-vertex colour, which for a surface
is an unreadable silhouette. Mesh is the first geometry type in Luxar that
**shades** — the other three are purely emissive per-element sprites with no
notion of a surface orientation — so this adds the GLSL + TSL material pair and
the light-free shading model of `docs/specs/MESH_NODE_SPEC.md` §6.2:

```glsl
shade = mix(uAmbient, 1.0, pow(saturate(dot(N, V) * 0.5 + 0.5), uShadeExponent));
```

`V` is the fixed view-space axis `(0, 0, 1)` — a camera headlight — so nothing is
added to the scene graph and no light node exists. `uAmbient = 1.0` collapses the
term and reproduces the emissive look of the other three.

The normal comes from the stored `normal` attribute when `shading == "smooth"`
**and** `normal_dims` equals the displayed axes, and from screen-space
derivatives of the view position otherwise. That is a compile-time shader
variant, not a runtime branch, because a declared-but-unbound `normal` reads
`(0, 0, 0, 1)` rather than "absent" — there is no runtime value meaning "no
normals". It is decided once per node and re-applied per epoch, since the
`normal_dims == displayDims` half is view-dependent.

Two things had to be enforced rather than assumed, and both would have failed on
one backend only:

- **The derivative normal is forced viewer-facing.** `cross(dFdx(P), dFdy(P))`
  carries the sign of the fragment-space y axis, and GLSL's `dFdy` is bottom-up
  where WGSL's `dpdy` is top-down. Unforced, the flat variant could collapse to
  `uAmbient` on WebGPU while shading correctly on WebGL — and the parity harness
  compiles TSL *to GLSL*, so it could never see it. (Later measured inert on
  Chrome — see the WebGPU A/B entry below; the flip is kept as insurance.)
- **The stored normal is transformed without three's `transformNormalToView`**,
  whose `transformDirection` normalizes. The writer accepts zero-length normals
  with a warning (degenerate triangles legitimately produce them), and
  `normalize(vec3(0))` is `NaN`, which interpolates across every triangle
  touching that vertex and flat-shades all of them. Spelled out as
  `viewMatrix · (modelNormalMatrix · n)`, shading near a degenerate vertex stays
  locally distorted, which is what the spec promises.

Mesh's blending defaults to **`opaque`**, unlike the siblings' `additive`: it is
the only mode unconditionally correct without per-triangle depth sorting, and it
is what a surface should look like. The default is applied viewer-side and never
stamped by the writer, so an ancestor group's mode still wins. `volumetric`
degrades to `opaque` with a one-time warning naming the node — a warning rather
than a failure, because the mode can arrive by inheritance from a group the mesh
knows nothing about. Under `opaque`, node `opacity` sweeps a hard alpha **cutout**
rather than dimming the surface; a smooth fade means selecting `normal`.

Also here: per-vertex `scalars` + `colormap` now work on a mesh (including the
#936 rule where an authored gain becomes the LUT window instead of double-applying),
and the mesh material is deliberately **not** camera-aware — a mesh has no
screen-space size to recompute, so it is tracked for disposal in a separate
registry instead of taking a per-frame no-op broadcast per node.

That last point turned out to expose a latent mismatch worth its own note. The layers
panel's `LuxarMaterial` contract required `CameraAwareMaterial`, which made a mesh
material — a perfectly ordinary leaf material with the full layer-control surface —
unrepresentable in the panel. The panel never called `updateCameraParams`, and its own
`isLuxarMaterial` guard never checked for it, so the type had been over-claiming
relative to the check that produces it. The requirement came off the interface, and
`MaterialManager.register` now takes a plain `THREE.Material` and dispatches on
`isCameraAwareMaterial` (the pattern the picking system already used) — which let three
accreted `as Parameters<typeof register>[0]` casts go, restoring type checking at
those call sites.

The three shade knobs (`ambient`, `shade_exponent`, `alpha_cutoff`) are read from the
node's composed attrs and clamped. They were already reachable through
`add_mesh(**attrs)` and were being silently dropped; and they are fractions and an
exponent rather than gains, so out-of-range values are meaningless rather than merely
odd — `ambient = 1e9` whites out the surface, `alpha_cutoff = 1e9` discards every
fragment, and `shade_exponent = 0` makes `pow(0, 0)` (undefined GLSL) at every
face-away fragment, the same hazard `clampGamma` already exists for.

Still to come: the Layers-panel appearance controls.

#### Mesh is pickable — the fourth pick material pair

Hovering a mesh now resolves to a **vertex**, whose ordinal indexes the per-vertex
label CSR directly. Mesh could not reuse a sibling's pick material, because its
element id is a built-in rather than an attribute: it has no per-triangle depth
sort, so there is no `aSortedIndex` to indirect through and the id is
`gl_VertexID`. Under an indexed draw that is the ordinal of the vertex in the
`vertices` array — which is also why vertex, not face, is the granularity: a slice
move rewrites only the index buffer, so a face ordinal would be renumbered on every
slice change while a vertex ordinal is invariant.

Three things the pick pass has to copy from the visual material, each of which is a
real defect if it drifts:

- **The alpha cutout, identically.** Without it a hole you can see through still
  rasterizes at true surface depth — becoming pickable _and_ occluding picks of the
  nodes visible through it.
- **The face culling.** The siblings' quads are view-facing, so their pick
  materials can pin `DoubleSide`. A mesh's back faces may be culled on screen, and
  `side` is a property of the current `displayDims` epoch (an undecidable frame
  forces double-sided), so it is synced per epoch and per pick render.
- **The node opacity.** It is half the coverage term, so dragging the layers-panel
  opacity slider below the cutoff must dissolve the surface in the pick buffer too,
  not just on screen.

Both mode-dependent behaviours — the cutout, and real projected depth vs
brightness-as-depth — are **runtime uniforms rather than shader defines**, so a
blending-mode switch from the layers panel is a uniform write instead of a
mid-hover recompile. That is why `mesh-pick` is one codegen snapshot variant
covering every mode, and a second snapshot appearing would mean a build flag had
crept back in.

Two backend divergences are handled rather than hoped away. A `flat` varying is
sourced from ONE triangle corner, and OpenGL ES fixes that to the **last** vertex
where the WGSL `@interpolate(flat)` three emits samples the **first** — so the same
click would report different vertices on the two backends. `WEBGL_provoking_vertex`
aligns them where it exists; where it does not, the contract stands as "_a_ corner
of the front-most triangle under the cursor", which is the honest answer anyway
since the cursor is over the face. (The context-wide flip is safe because every
other `flat` varying in the shipped materials is a per-_instance_ constant,
identical at all corners.) Separately, the two backends write different
surface-depth _values_ — `gl_FragCoord.z` versus a linear view-space depth — which
is benign because both are monotone in distance and the pick buffer's depth only
ever orders fragments within one render; that pairing already shipped undocumented
in the gsplat pick shaders and is now written down.

The 16-bit element-id split is now single-sourced as `luxarElementIdSplit(uint)`:
`luxarElementIdParts()` became a one-line wrapper over it and mesh calls it
directly off `gl_VertexID`. Two copies of that mask-and-shift could drift, and the
only symptom would be picks resolving to the wrong vertex past 65,536 — silent, and
only on large meshes. The `2^27` vertex cap that keeps the pick vote key alias-free
(already enforced at write time and in the loader's metadata preflight) now has the
arithmetic pinned next to the stride it constrains, since mesh is the one type whose
bound is _enforced_ rather than structural.

Registration went where production actually registers picks, which turned out not to
be the node factory: `initPicking` traverses the finished scene to decide whether any
node declares labels and only then constructs the `PickingSystem`, so on a first load
the factory has nothing to register with and `NodeFactory.registerExistingSceneNodes`
is the pass that runs. That pass was a three-way `else if` chain and is now a
`Record<GeometryTypeName, …>` — a fifth geometry type becomes a compile error at one
table instead of a branch someone forgets, which is exactly the failure this would
otherwise have shipped: a mesh unpickable in every real scene, appearing to work only
on a second dataset load. It also had no direct test at all; it has seven now.

The hover/label path needed no changes, which is worth stating because it was checked
rather than assumed: the writer validates labels per-VERTEX — the granularity mesh
picks at — `LabelLoader` reads `<path>/label_offsets` lazily with no type dispatch,
and `buildPickResultHandler` takes `(nodePath, elementId)`.

#### Mesh gets its Layers-panel shading controls, and the debug surface learns to count it

**Ambient** and **Shade falloff** parameterize the §6.2 headlight; **Alpha cutoff** is
the `opaque` cutout threshold. These are the first controls in the panel gated on the
geometry TYPE rather than the blending mode — mesh is the only type that shades, so on
a points layer they would be controls that visibly do nothing. Alpha cutoff carries a
mode gate on top (the cutout exists only in `opaque`), and its drag also reaches the
mesh's pick material, since the pick pass applies the identical cutout.

They are the one control group that does not compose along the ancestry, and are
applied through their own path rather than through `applyComposed`: a shade floor is a
per-surface appearance choice with no composition rule — multiplying two ambients
would mean nothing — so a group layer over meshes does not offer them.

`window.__luxarDebug.getState()` gained `meshNodes` and `totalTriangles`, which had
been simply absent: three hand-written arms counted points, gsplats and lines, each
selecting on `InstancedBufferGeometry`, and a mesh is the one type that draws from a
plain indexed `BufferGeometry`. The triangle count comes from the **draw range**, not
the index length, because that is what the nD slice compaction narrows — reading
`index.count` would report the whole surface no matter where the slice sits, which is
the number a debug driver most needs to be honest about.

And a real defect next door: `getDrawOrder()` was reporting **0 elements for every
mesh**. Mesh reached that walk fine, but the element count fell through to a local
`visiblePointCount ?? visibleSplatCount ?? visibleSegmentCount ?? 0` chain — a partial
copy of the shared per-type reader with `visibleTriangleCount` missing, so the count
read as "absent" rather than as an error. Present, plausible and wrong is the worst
shape for a diagnostic. Both now go through one reader.

#### The mesh `opaque` default now actually fires (#1272)

Spec §6.3's central asymmetry — mesh defaults to `opaque` where the three emissive
types default to `additive` — had never worked on the production load path. The chain
is `applyEffectiveAttrs` → `composeAttrs` → `normalizeBlendingMode`, and
`normalizeBlendingMode(undefined)` returns `'additive'`, so a composed `blending_mode`
was **never** undefined by the time a material saw it. `createMeshNode`'s
`(attrs.blending_mode as BlendingMode) ?? 'opaque'` was dead code: every mesh in every
real scene rendered additive, the alpha cutout never compiled, and — once the panel
controls landed — the Alpha-cutoff slider was hidden for a default-config mesh because
its layer reported `additive`.

**Found by rendering a written mesh end to end for the first time** — no unit test
could have seen it, because they all hand `resolveRequestedMeshMode` an attrs object
directly rather than one that has been through composition. The fixture and E2E spec
that found it are the entry below.

**Fixed in #1274**, which landed independently while this work was in review: rather
than defaulting inside `composeAttrs`, it keeps `blending_mode` **undefined** through
composition when no level sets one and lets each consumer apply its own per-type default
via `defaultBlendingMode(nodeType)`. That is the better shape, and it covers a case the
alternative did not: it tracks whether a layer's mode is EXPLICIT, so a group layer
merely *displaying* a neutral default does not push it onto mesh descendants — which
would otherwise flip a mesh under a plain `layer=true` group back to additive at panel
init. Nearest-setter-wins is untouched either way, which is what keeps
`group(blending_mode="additive")` working for its mesh children and why §6.3 forbids the
writer from stamping the mode at all.

#### Blend-mode ownership finishes the job: groups stop overriding what they never set (#1275)

The explicit flag above stopped a plain group layer from *emitting* its displayed
default, but two paths still let a non-owning layer overwrite a descendant's mode:

- **The subtree-drop in `composeEffective` fired unconditionally.** The drop exists so
  an owning wrapper's Blend control wins over its parts' stamped modes (the
  `graft_gsplat_node` case). But a wrapper that owns no mode has no control value to
  impose, so dropping was pure loss: a mesh authored `additive` under a plain
  `layer=true` group snapped to its `opaque` type-default on any non-blend group edit.
  The drop is now gated on the same ownership flag.
- **Ownership was initialized from the COMPOSED ancestry, not the node's own attr.** A
  layer that merely inherits an ancestor's mode must not re-emit it as its own setter:
  the re-emitted copy is a snapshot of disk state, sits nearer the leaf, and would
  shadow the ancestor layer's next live pick. Ownership now reads the node's own
  `blending_mode` (or a user pick) — an inherited mode still displays, but the layer is
  not a setter. The same rule means an unauthored geometry leaf does not own its
  per-type default, so a group layer's Blend pick actually reaches it.

The Absorption and Alpha-cutoff gates also read the **mesh-resolved** mode through
`resolveLayerBlendingMode`: a mesh resolves `volumetric` → `opaque`, so Absorption (a
control no mesh shader reads) stays hidden and Alpha cutoff shows exactly when the
cutout is compiled — even if a stored layer mode reaches the gates unresolved.

#### Mesh gets its first real fixtures, and an end-to-end render spec

`test_mesh.luxar.zarr` and `test_mesh_nd.luxar.zarr`, plus
`src/tests/e2e/mesh-rendering.spec.ts`. Everything about mesh had been tested one layer
down — cull kernels by parity, the material pair by codegen snapshot and a GLSL↔TSL
pixel harness, the loader/commit/panel/debug paths by unit test with stub materials —
and none of that covers the wiring: that a mesh authored by the Python writer arrives
through the loader, commits, gets the right shader variant, and puts pixels on screen.
The first run of that spec found the `opaque` bug above and two more.

The fixture is a **welded, closed icosphere** rather than a cube or a grid, and each
property is load-bearing: welded so `gl_VertexID` is a genuine many-to-one pick target
(de-indexed it would be 960 vertices instead of 162, making the shared-vertex pick
semantics untestable); closed so an nD slab cull exposes interior back faces, which is
what the `gl_FrontFacing` flip exists for; and smooth non-axis-aligned normals so a
build that ignored `shading` would render visibly differently — on a cube the stored and
derivative normals agree per face.

Two smaller defects fell out of the same run. The debug surface read shader variants
with `!!defines?.FLAG`, and a GLSL define's conventional value is the **empty string**
(three emits a bare `#define`), so `!!''` reported every variant as off while the shader
was compiled with it. The **existing lines arm had the same bug** — `hasColormap` had
always been false in production — and its unit test passed only because the fixture used
`1` where production uses `''`: a vacuous assertion, now fixed on both sides. Separately,
the mesh pick material is now seeded from the visual material's **live uniforms** rather
than the authored attrs, so a WebGL context restore after a layers-panel drag no longer
reverts pick coverage to the load-time values.

#### Mesh gets its reference demo: isosurfaces of a real fluorescence volume

`luxar demo run mesh_isosurface_cells3d` — marching-cubes isosurfaces of the
two-channel scikit-image `cells3d` volume (membranes + nuclei) as two shaded,
toggleable mesh layers. Isosurfaces and segmentation boundaries are the named target
data for Mesh: routine outputs of the pipelines Luxar already serves, which before
Mesh could only be approximated by a dense point cloud.

Deliberately the **same dataset** as `gsplats_3d_cells3d_multichannel`, because the
pairing is the lesson. Splats approximate the whole intensity field and need no
threshold; an isosurface picks one level set and renders it as an opaque surface with
real occlusion and silhouettes. Neither is the better answer — they answer different
questions, and seeing the same nuclei both ways is the fastest way to feel the
difference.

No GPU and no fitting step: marching cubes is CPU-only and takes about two seconds,
which makes this the cheapest end-to-end demo of any Luxar geometry type. ~537K
vertices / 1.07M triangles across the two surfaces.

`GEOMETRY_VALUES` in the demo registry gained `mesh` — a vocabulary that had never
needed a fourth entry.

#### A mesh layer now reports the mode it actually renders

`volumetric` has no meaning for a zero-thickness surface, so the mesh material maps it
to `opaque` and stamps the RESOLVED mode. The layers panel was storing the composed,
UNresolved value — which made it disagree with the render in two visible ways at once: it
showed the **Absorption** slider (which no mesh shader reads) and hid **Alpha cutoff**
precisely when the cutout was active. The pick pass reads the material's resolved mode,
so it was correct and only the UI was wrong.

Resolved at the point of STORAGE rather than at each display gate, so every consumer —
the Blend dropdown's own displayed value included — sees the mode that renders. A user
who explicitly picks `volumetric` on a mesh sees it snap back to `opaque`, which is
honest: it is what the surface is doing, and it matches the one-time warning the loader
already emits. The three types that DO implement volumetric are untouched.

#### Docs catch up with mesh

Two viewer package READMEs had omitted mesh entirely. `rendering/README.md` gains a
Mesh Material section — the odd one out on purpose (a plain indexed `BufferGeometry`
rather than an instanced quad, not camera-aware, `opaque` by default) — naming the two
hazards that fail on exactly one backend and that the parity harness structurally cannot
see, since it compiles TSL *to* GLSL: the `dFdy`/`dpdy` sign, and
`transformNormalToView`'s internal normalize turning a legal zero-length normal into a
whole-triangle NaN. `data/README.md` gains the whole-node-loader section: why mesh does
NOT stream (a surface is connected, so a chunk of triangles is not independently
meaningful), why its admission gate has to run before any chunk is fetched, and why its
nD tolerance arm cannot be inherited from Lines.

Also corrected the stale "picking and the Layers-panel appearance controls land in later
phases" line in `CLAUDE.md`, and the "three geometry types" scene-graph and architecture
summaries there.

Two `README.md` claims that still read "all three geometry types" were checked and
deliberately LEFT: one is about volumetric blending physics and the other about the
chunk-bounds spatial query, and mesh participates in neither by design.

#### Mesh on real WebGPU: verified, with the remaining gap named precisely

The mesh vertical shipped with a stated gap — the GLSL↔TSL parity harness drives
`WebGPURenderer({ forceWebGL: true })`, so the real-WGSL path was never exercised. It
is now: an A/B against native WebGPU (system Chrome channel, `?renderer=webgpu`,
screenshot-then-decode with the WebGL arm as a control) shows `apiSurface: 'webgpu'`,
all three fixture nodes committing with identical triangle/vertex counts and identical
shader variants, and **pixel-equivalent output** — 105,822 lit pixels on both backends,
mean lit channel differing by 0.14% (sub-quantization dithering — equivalent to the eye,
not byte-identical).

It also **corrected an overstatement of our own**, which is the more useful half. The
§6.2 notes claimed an unforced derivative normal "would collapse to `uAmbient`
everywhere on WebGPU". That is a spec-derived RISK, not an observed behaviour. The
fixture gained a `flat_facing` node — a flat-shaded quad FACE-ON, where `N.z ≈ ±1` makes
the sign flip the difference between full brightness and the ambient floor — and with
the `z >= 0` flip REMOVED, real WebGPU still renders it identically to WebGL.

The metric is demonstrably sensitive rather than blind: a control run with every mesh
hidden drops from 93,851 lit pixels to 56,700, so the quad contributes 37,151 and lifts
the mean lit channel from 40 to 116. (The original edge-on `flat_patch` could not have
shown this either way — `N.z ≈ 0` there, and flipping the sign of ~0 leaves
`wrap = 0.5` unchanged. That is precisely why the fixture needed a face-on node.)

So: on Chrome + Apple Silicon the two derivative conventions **coincide** and the flip
is inert. It stays, because it costs one instruction, is correct under either
convention, and neither shading-language spec promises they agree — insurance, not a fix
for an observed bug. The shader comments, the `rendering/` README and the spec now say
exactly that instead of asserting a failure nobody has seen.

Still arguments rather than measurements: the provoking-vertex convention and the
surface-depth value, both because reading the pick buffer's ids and depth from outside
the app is not cheaply reachable.

#### Demos — the biodiversity globe is `opaque`, so it stops painting over its own data (#1227)

The globe was `volumetric` with a heavy absorption, which read well in isolation
but made the layers unreadable together. Measured draw order in the live scene
(hooking `onBeforeRender`, so this is THREE's real sequence rather than an
inference from `renderOrder`):

```
 3-10. All life  (8 tiles)  transparent  depthWrite=0
 11.   Earth                transparent  depthWrite=0   <- backdrop drawn LAST
 12.   Migration highways   transparent  depthWrite=0
```

Two consequences. The globe composited **on top of** the 15M-record layer,
multiplying it by the shell's transmittance — at absorption 10, most of the way
to erasing it. And because no mode except `opaque` writes depth, nothing occluded
anything, so far-side records and track ribbons showed straight through the
planet.

The ordering is a containment rule in
`rendering/depth-sort-coordinator/render-order.ts` firing on inverted geometry: it
hoists a group whose bounding sphere contains another's so that "embedded content
composites on top", which assumes *container = background*. Here the data sits on
a shell **outside** the globe and its 8-tile group sphere is a deliberately loose
upper bound, so the data was classified as the container and the backdrop as
embedded content. Filed as #1227.

`opaque` is the only mode with `transparent: false`, so THREE draws it in the
opaque bucket ahead of every transparent layer, and the only one that
unconditionally sets `depthWrite: true`. The globe now draws first (verified:
`/Earth` at step 1 with `depthWrite=1`) and occludes correctly. The cost is that
the shell no longer self-shades as a participating medium.

The brightness was re-tuned with it: the display window went from 0–0.041 (a
24.39x gain, chosen against `absorption=10`) to 0–0.205 (a 4.88x gain). The gains
land almost exactly 5x apart (24.39 / 4.88 = **5.0**) — a neat near-exact
coincidence, though the switch to `opaque` also changed the compositing (now
unblended and opacity-independent), not only the absorption term, so read it as a
mnemonic rather than a proof that the whole gain was absorption. Either way,
carrying the old gain over to `opaque` left the planet blown out.

#### Demos — 4D NEXRAD weather-radar supercell (atmosphere/geoscience gap)

New `demo_gsplats_4d_nexrad_supercell`: 82 WSR-88D Level II volume scans of the
2013-05-31 Oklahoma convective evening as a 4D Gaussian-splat timelapse, 21:00
UTC through 03:00 UTC — initiation, the El Reno tornado (touchdown 23:03,
dissipation 23:43), and the overnight growth into the mesoscale system that
flooded Oklahoma City. First atmosphere demo in the suite and the first gsplats
demo in the `geoscience` category.

Radar data arrives as nested *cones* — 14 discrete elevation tilts, not a
volume — so the demo regrids polar gates onto a fixed Cartesian storm box with a
Barnes-weighted `cKDTree` interpolation (scipy only, no Py-ART), geolocating
every gate through the 4/3-effective-earth beam model and masking cells the beam
could never reach rather than hiding the extrapolation with a threshold. Grid
spacing (750 m) is matched to the real beam width at the storm's range rather
than to gate spacing. Timepoints are stacked with `combine_as_new_dimension`
into one 4D node, and a single GLOBAL intensity scale is used instead of the
usual per-frame normalisation so the storm's intensification and decay survive.
The splat budget is adaptive (constant occupied-voxels-per-splat) because the
system grows ~6x across the window, and the scene bakes `CameraConfig(up=(0,0,1))`
— a geographic scene left on the viewer's default up-vector renders its altitude
axis sideways.

Two Lines nodes accompany the splats: a marker at the surveyed tornado position
on the frames when it was on the ground (NWS damage survey via the SPC tornado
database — it lands on the radar's hook echo, an independent cross-check of the
geolocation), and a faint wireframe cube outlining the analysis domain.

Adds `metpy>=1.6.3,<2.0` to the `demos` extra (pure-Python Level II decoder; the
floor is the NumPy-2 release). The default path loads precomputed splats from
Git LFS and needs no network, GPU or decoder.

#### Fixed — a caller `ordering=` no longer desyncs a node's on-disk sort order (#1221)

`add_points` / `add_lines` / `add_gsplats` used to accept an `ordering=` keyword
that nothing on the write side read, yet it was persisted over the value the
geometry writer had already stamped — leaving the `ordering` attr disagreeing
with how the arrays are actually sorted. The viewer trusts that attr to decode
the space-filling-curve chunk index, so a stale value silently decoded the wrong
curve (or, with `ordering="none"`, threw the spatial index away and loaded every
element). `ordering` is now a reserved, writer-stamped attr on all three
geometry types: supplying it is rejected up front, and the stamp always reflects
the compiler's `ordering_method`. The standalone `save_gsplats` /
`write_gsplats_tree` API, where `ordering` is a real honoured parameter, is
unchanged.

#### Added — targeted demo dependency installs and consistent install status (#915)

`luxar demo deps --only MODULE` now narrows the report to one import module and,
with `--install`, installs that row's exact constrained requirement instead of
pulling an entire Luxar extra. This also gives dependencies outside every extra
(currently `gdown`) an explicit managed install path. `--only` is
case-insensitive and mutually exclusive with `--extra`.

Generic `--install` continues to manage Luxar extras and now treats an
orphan-only report consistently with an orphan left beside a successfully
installed extra: it explains that no command was attempted and exits 0 rather
than turning an unattempted requirement into an install failure. Report-only
mode remains the CI/setup gate and still exits 1 for every missing or outdated
row. The obsolete editable-install `mkdir packages/luxar-viewer/dist`
workarounds were removed from the Makefile and CI/docs workflows; editable
installs have not required that wheel-only artifact since the custom Hatch build
hook gained `force_include_editable`.

#### Added — `mesh`, a writable triangle-surface geometry type

`scene.add_mesh(name, vertices, faces, ...)` writes a triangle surface into a
`.luxar.zarr`: nD `vertices` plus a `faces` triangle-index array, with optional
per-vertex `normals`, `colors`, `scalars`, labels and image labels. `luxar info`
reports vertex and face counts, `LuxarScene.get_mesh()` / `list_meshes()` read it
back, and a mesh leaf contributes to scene bounds like any other geometry.

**Writable first, renderable in the phases that followed.** At this point the
viewer could not draw a mesh — its loader, shaded material and picking landed in
the later phases described in the entries above (`docs/specs/MESH_NODE_SPEC.md`
§11 tracks the whole sequence). The format contract names the two sets separately
so neither side has to answer the other's question: `geometry_types` is the
writable leaf vocabulary (which `mesh` joined here) and the new `loader_types` is
the viewer-drawable subset (which it joined with the Phase-3 switch-on). Adding a
type to `loader_types` without its viewer code is still a compile error at
`LoaderByKind`, `GEOMETRY_DESCRIPTORS` and `computeHiddenDimTolerance`, exactly as
before.

Unlike the other three types a mesh has no per-element size — a triangle's extent
comes from its own vertices — so it adds no extent padding to bounds. `normals`
are stored `(V, 3)` with a **required** `normal_dims` attr naming which three
dimensions they describe, because normals are a display-space quantity and an
implicit "first three dimensions" is wrong for any mesh whose leading dimension
is not spatial (for a `(t, x, y, z)` mesh those are `(t, x, y)`).

Out of scope for this phase, each refused with an explanation rather than
silently degraded: LOD/decimation, `kind=partition`, per-triangle depth sorting,
spatial indexing, `volumetric` blending, and mesh import formats.

#### Added — `mesh` nD culling kernels, in both backends

The nD visibility half of the mesh vertical: `mesh_vertex_visibility_mask` and
`compact_visible_faces`, implemented in the Rust/WASM crate and its TypeScript
reference, kept in exact parity. Nothing calls them yet — the loader that will is
the next phase (`docs/specs/MESH_NODE_SPEC.md` §11) — so a mesh still writes but
does not render.

**Whole-triangle cull, not clipping.** A triangle is drawn iff *all three* of its
vertices pass the nD slab test. Lines clip a segment against the slab and
interpolate every attribute at the crossing; the exact triangle equivalent is nD
polygon clipping with fan re-triangulation and per-new-vertex interpolation on
every slice move. Culling whole triangles instead costs 183 code lines across both
backends against 803 for segment clipping (the spec puts full nD polygon clipping
at roughly ~1500), at the price of a ragged, triangle-quantized cut boundary
instead of a clean planar section. Exact clipping stays out of scope.

**Vertices are never compacted.** `compact_visible_faces` writes *original*
vertex indices, so a slice change rebuilds only the index buffer while the vertex
attribute buffers stay uploaded in full. `drawElements` never fetches an
unreferenced vertex, so culled vertices cost nothing to draw, and this avoids a
vertex-remap array as well as `compact_by_mask`, which is `f32`-only and could
not compact native `uint8`/`uint16` vertex colors.

Two details are load-bearing and both are pinned by mutation-tested cases:

- The TypeScript backend `Math.fround`s its slab bounds. JS computes
  `slice - tolerance` in f64, where the difference of two f32 values is *exact*,
  while Rust rounds it to f32. The gap is under half an ulp — but when the
  rounding goes down, the rounded bound is itself a legal f32 vertex coordinate,
  and a vertex sitting exactly there was visible in WASM and culled in
  TypeScript. `slice = 1.0, tolerance = 0.1` is such a case and is now a parity
  test.
- An out-of-range face index drops that face in both backends instead of reading
  past the mask. Face indices come from the store, and the two backends fail
  asymmetrically without the check: because the crate is `panic = "abort"`, a
  Rust out-of-bounds read traps with an opaque, uncatchable
  `RuntimeError: unreachable` rather than failing as a node-scoped error, while
  TypeScript would read `undefined` and diverge silently. The loader
  rejects such a store up front; this is defense in depth behind that gate.

Both kernels are declared on the `WasmModule` interface and listed in the
`REQUIRED_WASM_EXPORTS` staleness check, so an out-of-date `public/wasm/` build is
reported as stale (and falls back to TypeScript) instead of failing later with an
opaque "not a function".

#### Added — the mesh data path: whole-node loader, two-stage validation, nD cull

The viewer half of the mesh vertical that turns the kernels above into loaded
geometry: `types/mesh.ts`, and `data/mesh/` with a whole-node `MeshWholeNodeLoader`, the
admission gate that guards it, and the display-space projection that drives the
cull. `mesh` is still absent from `loader_types`, so nothing in the render path
reaches this yet and a mesh still writes without drawing; wiring it up is the next
step.

**Whole-node, deliberately.** `lines-spatial-index-loader.ts` runs to ~1400 lines
because line datasets reach tens of millions of vertices and a slice change
genuinely needs only a fraction of them, so a dual chunk index earns its
complexity. Mesh faces share vertices across any cut, so the working set after a
`displayDims` change is the whole mesh regardless — an index would add machinery
and skip nothing. Consequently `updateView` never re-fetches: it returns the same
cached mesh, and only the index buffer downstream changes with the view.

**Admission runs in two stages, and the split is what makes it work.** The writer's
validators protect only stores Luxar produced, while the viewer loads arbitrary
`?src=` URLs. Because the loader fetches everything up front, a check that runs
after decode arrives too late — a hostile store can declare enormous arrays and
exhaust tab memory before the per-node `LoaderError` containment is reachable. So
Stage 1 decides everything it can from `.zarray`/`.zattrs` alone, with no chunk
fetched: the `2^27` vertex cap, a 512 MiB `MESH_DECODE_BUDGET_BYTES` ceiling,
shape/dtype cross-checks, `normal_dims` well-formedness, and presence flags that
must agree with the store. Stage 2 then checks what needs materialized arrays.
Tests assert the no-fetch property against a store that records every key it is
asked for, rather than against a proxy for it.

Four things in there are easy to get wrong, and each is pinned:

- **`faces` is read raw, never through `ArrayDecoder`.** The decoder yields
  `Float32Array`, whose 24-bit mantissa cannot represent every index a
  2^27-vertex mesh may carry — it would silently round anything above 16,777,216.
- **The face-index range check is two-sided and runs pre-coercion**, because each
  side of the integer→u32 cast hides its own wrap-around: a signed store's `-1`
  passes a one-sided `< V` test and becomes `0xffffffff`, while a 64-bit store's
  `2^32 + 1` survives a post-cast check by wrapping to `1`, landing inside range,
  and rewriting topology instead of trapping. 64-bit values are compared as
  `BigInt` so nothing above 2^53 can round into range first.
- **The byte budget charges what arrays DECODE to, not only what they store.** The
  stored side can be arbitrarily smaller than the allocation, so a stored-only
  budget is wrong in the dangerous direction: a broadcast array stores one row and
  expands to `n_elements` rows (a ~12-byte declaration can materialize gigabytes),
  every decoder-routed array yields a `Float32Array` (a `uint8` store decodes at
  4x), and `faces` widens to u32 whatever narrow dtype the `INDEX` encoder chose.
  The decoded term is also the only thing bounding `ndim`, which has no cap of its
  own — `n_vertices: 4, ndim: 2^26` passes every count check on a trivial stored
  footprint. The stored term still reads the *declared* dtype (an external `int64`
  costs 8 bytes per index), and the largest single chunk buffer is SUMMED in rather
  than checked alone: a chunk buffer exists during decode alongside the arrays, and
  zarr v2's `chunks > shape` allowance makes "just under budget on both terms
  separately, near 2x together" directly constructible. An oversized chunk is *also*
  rejected on its own, so one array can never exceed the ceiling even where the sum
  would fit.
- **The budget fails CLOSED on an encoding it does not recognise.** Four separate
  bypasses turned out to be one category — *the bytes the loader fetches are not the
  bytes the handle declares* — and fixing them one at a time kept yielding a fifth,
  because an unrecognised encoding fell through to "use the stored shape". The budget
  now keys a `Record<EncodingName, ...>` off the contract's `ENCODING_NAMES`, so adding
  a contract encoding is a compile error at the mesh preflight and an unknown one at
  runtime is refused rather than admitted. It paid for itself on first compile by
  catching a missing `uint64` entry.
- **An `array_ref` is budgeted at its TARGET, not at the stub pointing to it.**
  `ArrayDecoder` resolves `encoding.target` against the store root and reads that array
  in full, while the referring array is a `(0, k)` stub — so a 48-byte declaration could
  pull an unbounded array, and the throw only came from Stage 2, after the allocation.
  Stage 1 now follows the chain metadata-only, with a hop limit and a seen-set so a
  cyclic store is refused rather than hung. Refusing `array_ref` outright was not an
  option: `normals`/`colors`/`scalars` are written with dedup ON, so meshes sharing a
  colour array legitimately produce a ref. This is the same class as the broadcast
  bypass above and was missed by that fix — which is why the budget is now expressed as
  "charge the array whose bytes are fetched" rather than as a list of encodings.
- **The in-flight latch is identity-guarded, and that is a separate fix from the
  generation token.** `load()` collapses concurrent callers onto one fetch, and
  `dispose()` nulls the latch while the old promise may still be pending — so an
  unconditional clear on completion lets a stale settle wipe a REPLACEMENT load's
  latch, after which every `updateView` (a slice scrub, precisely what the latch
  exists for) starts another whole-mesh fetch. The generation token stops a stale
  completion *publishing* into a disposed loader; the identity check stops it
  *erasing the latch*. Fixing only the first leaves the duplicate fetches.
- **An open failure is classified, not assumed deterministic.** Wrapping any
  `vertices`/`faces` open error as `kind: 'Validation'` would have the failure record
  treat a transient network fault as permanent and never retry it. Routed through
  `classifyLoaderError`, matching the sibling node loaders.
- **Shapes are checked logically, and "logical" has three sources.**
  `encoding.n_elements` (broadcast) first, then `encoding.original_shape` (LUT /
  per-channel quantization), then the stored shape. Consulting only
  `original_shape` rejects a uniform colour, because the broadcast encoder stamps
  `n_elements` and *not* `original_shape`: `add_mesh(..., colors=(1, 0, 0))` and any
  incidentally-uniform colour array both land as `shape: [1, 3]` and look like a
  1-row array. The broadcast branch is gated on the encoding NAME as well as the
  count — stricter than needed for writer-produced stores (only the two broadcast
  encoders stamp `n_elements`), but it stops a hostile store covering V vertices with
  a bare `n_elements`, and stops the branch hijacking an array carrying both keys.

At the default budget the ceiling binds long before the vertex cap: a 3D float32
mesh runs out of bytes at ~22.4M vertices against a cap of 134.2M — half the
stored-only arithmetic, since the decoded term is charged too. The cap is still
checked, and checked first, so a nonsensical declaration is told about pick-key
aliasing rather than blamed for bytes.

**Winding is resolved against the authored frame.** `sorted(normal_dims)` is the
axis triple the stored face order is front-facing in. When the displayed triple
equals that frame with odd parity, display space is a reflection and every
projected triangle is uniformly reversed, so two of each triangle's three indices
are swapped — without it a `double_sided: false` mesh renders inside-out, and an
open surface vanishes. That reversal is keyed to the current `displayDims` parity
rather than to the *event* of it changing, so it runs on every index build in an
odd-parity epoch, initial load included. When the displayed triple is a *different*
triple than the frame, or the mesh declares no frame at all, projected orientation
is per-triangle data-dependent and no index post-pass can fix it: the epoch renders
double-sided with a one-time notice naming the node.

#### Added — mesh renders: the `loader_types` switch-on

`mesh` joins `loader_types` in the format contract, which is the moment it becomes
viewer-drawable. A mesh now loads and draws — **unshaded**, with flat per-vertex
colour. The shading model, picking, and Layers-panel appearance controls land in
later phases (`docs/specs/MESH_NODE_SPEC.md` §11), so in this phase a mesh ignores
`blending_mode`, `opacity`, `intensity`, `gamma` and `offset`; that is stated in
`create-mesh-node.ts` because it otherwise reads as a bug.

Flipping the contract fires exactly the three compile errors the split was designed
to fire, and no others — measured before writing any code: 6 `tsc` errors across
`tolerance-computer.ts`, `geometry-descriptors.ts` and `loader-registry.ts`, plus one
vitest failure and one pytest failure, both deliberate pins. The Python pin
`test_mesh_is_writable_but_not_yet_drawable` asked in its own docstring to be MOVED
rather than deleted when this happened, and it was.

**Mesh needed its own tolerance arm, and could not borrow any of the other three.**
Every existing strategy derives from that type's per-element extent, and a mesh has
none. Two halves, each with a specific failure mode if copied:

- *Not Lines' `0` for hidden continuous dims.* Lines get away with zero because
  segment clipping interpolates through the slab — a segment crossing the slice
  yields an intersection even at zero thickness. Mesh culls whole triangles with no
  interpolation, so `0` reduces membership to exact float equality with the slice
  plane and the node renders **nothing**. This is the most tempting wrong answer,
  because Lines is the nearest structural sibling; a test asserts mesh and lines
  disagree here rather than merely checking mesh's value.
- *Not the quarter-cell query reach for hidden discrete dims.* Mesh's slab is a
  membership gate applied after fetch, not a chunk-fetch reach, so it takes the
  half-cell. It is also the only arm that ignores `discreteRole`: mesh issues no
  range query, so membership is the only rule it has, and honouring a `'query'` role
  would hand a fetch reach to the one caller asking about visibility.

Be honest about the continuous arm: with per-vertex cull there is no true planar
cut, so a continuous hidden spatial dimension renders a **thick slab** and the
thickness is the only control. The dominant real case is discrete — a mesh's hidden
dimensions are almost always time or channel.

**Mesh names now follow the sibling geometry conventions, everywhere they diverged.**
Four-fold symmetry is the whole point of the Track A groundwork, so the divergences were
worth paying off rather than documenting:

- `build*` is a `ui/` verb — it appears nowhere in `rendering/` or `data/`. The geometry
  factories are `createMeshGeometry` / `createMeshColorAttribute` /
  `createMeshDefaultColorAttribute` / `createMeshIndexAttribute` (after
  `createPointsGeometry`), and three of them were also missing the type prefix that every
  export in `point-geometry.ts` / `line-geometry.ts` / `gsplat-geometry.ts` carries.
  `applyIndices` → `applyMeshIndices` for the same reason.
- `MeshGeometryInput` → `MeshGeometryConfig`, after `InstancedLinesMeshConfig` /
  `InstancedGSplatsMeshConfig` (minus the `Instanced` those two carry because a mesh is
  not an instanced quad).
- `projectMesh` → `projectMeshTo3D`, after `projectPointsTo3D` / `projectLinesTo3D` /
  `projectGSplatsTo3D`; the private bounds helper → `computeMeshProjectionBounds`, after
  `computeLinesProjectionBounds` / `computeGSplatsProjectionBounds`.
- `MeshLoader` → `MeshWholeNodeLoader` in `mesh-whole-node-loader.ts`. The sibling loader
  classes name their STRATEGY (`PointsSpatialIndexLoader`), and "whole-node" is already
  this codebase's term for mesh's (spec §7). Naming it `MeshSpatialIndexLoader` would be
  a lie — mesh deliberately has no index — but saying nothing was the asymmetry.
- Inside `data/<type>/`, only loader files carry a type prefix; `projection.ts` and
  `handler.ts` do not. So `mesh-preflight.ts` → `preflight.ts` and `mesh-validate.ts` →
  `validate.ts`, and the test files drop the redundant prefix to match their siblings
  (`spatial-index-loader.test.ts`, `projection.test.ts`).

Two prefixes were checked and deliberately KEPT: `MeshDataLoader` (symmetric with
`LinesDataLoader`) and `wasm/typescript/mesh-culling.ts` (which mirrors its Rust module
`mesh_culling.rs`). And `LoaderType` still has no mesh member — mesh emits no monitor
events until the metrics phase, so adding one now would be a slot with no producer.

**Rendering is a plain indexed `BufferGeometry`, not the instanced-quad stack.** The
other three render per-element sprites whose size and orientation are computed in the
shader, so they need an `InstancedBufferGeometry` and an RGBA32F element texture. A
triangle is already geometry. Consequences: no GPU buffer pool (nothing churns —
vertex buffers are uploaded once per `displayDims` epoch), no depth-sort registration
(an opaque surface gets correct occlusion from the depth buffer), and no capacity
clamp (mesh is bounded by `MAX_MESH_VERTICES` at the loader instead of by texture
dimensions).

Mesh is also the first type to feed colours to **vertex attributes** rather than an
element texture, which is why the dtype rules matter: three r184's WebGPU backend
exposes no 3-component 8/16-bit vertex format and requires `arrayStride` to be a
multiple of 4, so a size-3 `uint8` colour attribute (3-byte stride) fails
`createRenderPipeline` and the mesh renders nothing on WebGPU while looking correct
on WebGL. RGB `uint8`/`uint16` colours are therefore padded to RGBA with an opaque
alpha; `float32` binds natively. The index dtype keys on `vertexCount`, not the
largest index present, because `vertexCount` is fixed for the node while the largest
index drawn changes with the slice, and a dtype that differs between epochs is the
attribute-identity change WebGPU does not tolerate.

**The projected position buffer is allocated once per node too, and re-extracted
only when the displayed axes change.** Positions depend on `displayDims` alone, yet
`projectMeshTo3D` allocated a fresh `vertexCount * 3` array on every call — so the
geometry's array-identity check never matched and every slice scrub copied and
re-uploaded the whole vertex buffer and recomputed both bounds, defeating the "only the
index changes on a pure slice move" design outright (#1245). The buffer now lives on
`LoadedMeshData.projection`, allocated by the loader: `updateView` returns that same
object for the node's whole life and drops it on dispose, so the buffer inherits exactly
the right lifetime with no separate cache to invalidate. This is the Mesh counterpart of
the Points accumulator's reusable target buffers, keeping the four geometry types
symmetric on where reuse lives.

Reuse makes array identity useless as a change signal, and dangerously so in both
directions: with a reused buffer the identity is stable while the contents change on a
`displayDims` change (so the upload is suppressed and the mesh stays in the stale frame),
and with a freshly allocated one it always differs (so the whole mesh re-uploads every
slice move). The projection therefore reports `positionChanged` explicitly and the
geometry gates on that. Worth knowing when reading the tests: after the first commit the
geometry's `position` attribute IS the loader's buffer, so only an integration test
through process -> commit can exercise the identity trap — a unit test that constructs
its own geometry cannot reach it.

**The index buffer is allocated once per node and drawn through `drawRange`.**
Replacing `geometry.index` on every slice move leaks its GPU buffer: three caches
attribute buffers in a `WeakMap` keyed by the attribute object and only calls
`gl.deleteBuffer` from `WebGLAttributes.remove()`, which runs on geometry disposal
(for whichever index is current then) and when the *wireframe* attribute is replaced —
never when `index` itself is swapped. The orphaned attribute's `WeakMap` entry is
collected and its GPU buffer is never freed, so a timelapse scrub orphaned one index
buffer per move. Mesh is the only type that rewrites its index per epoch (the other
three update pooled attributes in place), so nothing in the tree had hit this.

The buffer is now sized from the node's total `faceCount` and the visible prefix drawn
with `setDrawRange`. Reuse requires the buffer to already be at that FULL capacity
rather than merely large enough for the current epoch — the placeholder is born with a
zero-length index, so a first epoch that happens to be fully culled would otherwise
fit in it and force a reallocation on the next epoch that reveals a triangle. The
upload is bounded to the rewritten prefix via update ranges, so reuse does not trade
the leak for a per-move bandwidth regression (the classic WebGL backend honours them;
the WebGPU ones re-upload in full regardless). Because the attribute object is then
stable, a slice move also rebinds nothing, so it leaves three's cached `RenderObject`
alone and does not participate in the `attributesRebuilt` eviction contract.

One consequence for readers of counts: `index.count` is the capacity and
`drawRange.count` is what is drawn, which is why `camera-framing.ts` reads the latter —
taking the former would give a fully-culled mesh a non-zero primitive count and frame a
scene that draws nothing.

**Camera framing was silently blind to a mesh.** `computeSceneBoundingBox` gates on
`InstancedBufferGeometry`, which a mesh never is — so a mesh-only scene returned an
empty box and zero primitives, and the camera framed nothing. That was a live bug
introduced by making mesh drawable, and the file's own docstring had predicted it
verbatim ("a geometry type rendered from a plain `BufferGeometry` still contributes
nothing to the bounds — and therefore frames the camera wrongly, silently"). Mesh now
has its own arm, counting DRAWN triangles from the index rather than the authored
`n_faces`, so a culled mesh reports what is on screen. A prose warning was not enough
to stop this happening once; the lesson is that only a compile error is.

**Framing a mesh reads the drawn vertices, not the whole position buffer.** Counting
drawn triangles while taking bounds from `computeBoundingBox()` was internally
inconsistent: under the no-compaction design the position buffer always holds every
vertex of the whole nD mesh, so the box spanned vertices whose triangles the slab cull
removed — and vertices no triangle references at all. A 4D surface that translates over
time framed its ENTIRE trajectory, pulling the camera out (and inflating the derived
scene scale and zoom limits) while the drawn slice sat small and off-centre. The
projection now returns the AABB of the vertices its emitted index references, as a
`MeshProjectionBounds` — the Mesh member of the `LinesProjectionBounds` /
`GSplatsProjectionBounds` family — and `computeMeshBounds` sets the geometry's box and
sphere from it, exactly as `computeLineBounds` consumes the lines projection's
precomputed bounds. Camera framing therefore needs no mesh-specific bounds path at all:
it reads `geometry.boundingBox` as it always has. Mesh's bounds carry no per-element
extent term, and that absence is the point — lines expand by `maxWidth` and gsplats by
`maxRowNorm` because their elements are sprites larger than their centers, whereas a
triangle's extent IS its vertices.

These bounds are TIGHTER than a whole-buffer scan and correct for every consumer, not
just framing: frustum culling and the raycast broad phase become exact over what is
actually drawn. The two states the first round of tests pinned — fully visible and
fully culled — are exactly the two where the defect cannot show, so the partial-cull
case is now covered where the mechanism lives.

The monitor's scene tree gained its mesh arm too — it rendered a blank count while the
other three showed one, even though the converter was already populating `faceCount`.
And `loaderDisplay`'s `default:` became a `satisfies never` guard: it previously shared
an arm with points, so a future `LoaderType` member would have been rendered as
"points / pts" in silence. Mesh is the concrete case waiting on that, since
`MeshWholeNodeLoader` has no `getMetrics` yet.

Two exclusions confirmed rather than assumed. Mesh stays out of slice prefetch
because `prefetch()` has exactly three hardcoded call sites and no fourth was added —
now pinned by a test, since "it works because nobody wrote the line" is what a later
refactor table-drives away. And `createProgressiveMeshLoader` **rejects** with an
explanation instead of falling back to the single-LOD loader: mesh has no LOD path at
all, and the descriptor table requires the factory for every drawable kind.

#### Fixed — a `kind=lod` group could be given a display type nothing can load

The LOD path only ever *derived* `display_type` from its finest child, with no
check that the result was a type the ladder supports — the partition sibling has
always had that guard. A geometry type with no LOD ladder was therefore accepted
and stamped, producing a `kind=lod` group the viewer cannot load, written with no
error at all. The guard is now applied at all three routes the value can reach
zarr by: the explicit `add_lod_group(display_type=...)` kwarg, the finalize-time
back-fill (the route that actually runs), and `compute_lod_display_type`.

It is deliberately not a homogeneity check and not an allowlist: `kind=lod` stays
intentionally heterogeneous-tolerant (a coarse points level under a fine gsplats
level is supported), and nested specialized groups keep carrying non-geometry
marker strings. Only a known geometry type whose capability row says it has no
LOD support is refused.

Relatedly, `validate_node_attrs` no longer checks node `type` against a
hand-copied set. That one failed the opposite way from the geometry-leaf checks
fixed in #1203 — a newly added node type would make every store containing one
report as "invalid" — but had the same root cause.

#### Fixed — writer-stamped presence flags are now reserved on every geometry type

`has_image_labels` was missing from `POINTS_/LINES_/GSPLATS_RESERVED_ATTRS` even
though all three writers stamp it. Passing it explicitly was already an error, but
reported as an *unknown* attribute rather than a *reserved* one, and the asymmetry
made "which flags does this writer own?" unanswerable from the sets alone. All four
sets now cover every flag their writer stamps.

Mesh additionally reserves `ordering`, which the sibling types deliberately leave
open. A measurement while adding it corrected a stale claim in that module: the old
comment said a caller's `ordering=` is "stamped-over" by the writer, and it is not —
`Node.__init__` re-persists the caller's attrs through `write_group` *after* the
geometry writer has stamped the group, so the caller's value is what lands on disk
(verified for points, lines and gsplats). Those three keep accepting it because
`ordering=` is a real request parameter there (`add_gsplats(ordering="hilbert")`
selects the method). Mesh has no spatial index at all, so a supplied value could
only write a lie the viewer would later read — hence reserved.

**Behaviour change:** `add_points(has_image_labels=...)` /
`add_lines(has_image_labels=...)` / `add_gsplats(has_image_labels=...)` and
`add_mesh(ordering=...)` now raise a "reserved attribute" error. All four already
failed; only the message changes, except `add_mesh(ordering=...)` which is new.

#### Fixed — the bandit pre-commit hook was stricter than the gate CI runs

The hook ran bandit unfiltered while `hatch run security` (what CI gates on) passes
`--severity-level medium --confidence-level medium`, and the CI step's own comment
says low findings such as developer-tooling subprocess calls are out of scope. The
mismatch meant editing any file that happened to contain an accepted low finding
was blocked by something CI would pass, leaving only `# nosec` churn on unrelated
lines or `--no-verify`. The hook now carries the same thresholds. Verified it still
fails on a medium/medium finding.

#### CI — the PyPI wheel viewer build is now a required release check (#688)

CI now builds the standalone viewer application with the same `pnpm build`
command used by the PyPI publish workflow, in a dedicated `wheel-viewer` job
separate from the npm library bundle. The check verifies the application entry
bundle, production WASM binary, and worker assets rather than accepting an
empty or partial `dist/` directory. CI, docs, and both publish workflows also
read the exact pnpm version from the viewer package's `packageManager` field, so
release and pull-request builds cannot drift between pnpm patches.
#### Changed — one canonical GSplat truncation radius, 2.75 (#1179, #1181, #1182)

The fitter has stamped `truncation_radius = 2.75` since the truncation
experiment landed, but that value was only ever applied to the fit config.
The format spec, the three model classes, the LOD spec, the spatial-ordering
`coverage_sigma` alias, and every read-side fallback in both Python and the
viewer independently defaulted to `3.0` — around thirty sites in total, with
no single definition and no test relating them.

There is now one constant per language, mirrored and pinned by tests that name
each other: `DEFAULT_TRUNCATION_RADIUS` in
`luxar.typing_utils.constants` and `GSPLAT_DEFAULT_TRUNCATION_RADIUS` in
`packages/luxar-viewer/src/config/constants.ts`. The viewer's former
`SHIFTED_GAUSSIAN_DEFAULT_TRUNCATE` (a worker-internal module) is gone in
favour of the `config/` one, so materials no longer depend on worker internals.

**Behaviour change.** A store that carries *no* `truncation_radius` attribute
now renders and culls at 2.75 rather than 3.0. Measured on a 400-splat test
volume, the tighter kernel integrates ~7% less total mass and differs from the
3.0 render by ~3% of peak (42 dB PSNR) — visible as slightly dimmer, slightly
smaller splats rather than a subtle change. Every dataset produced by
`gsplat fit` already stamps its own value and is unaffected, so this only
reaches hand-written or pre-attr stores. `gsplat migrate-format` likewise now
writes 2.75 where it previously baked in 3.0.

`luxar.gsplats.lift` is the one deliberate exemption and keeps 3.0, as the new
`LIFT_TRUNCATION_RADIUS` constant. Its `T` is a profile-matching parameter, not
a render default: the point super-Gaussian sprite and the gsplat kernel
coincide exactly at `T* = sqrt(2 ln 100) = 3.0349`, and moving the lift to 2.75
degrades that match by roughly 8.5x.

`truncation_radius` is also now validated on write. It arrives unvalidated from
dataset attrs and sets both the kernel support and the `1/(1-C)` normalization,
so a zero or non-finite value poisons every derived quantity — and Python had
no guard at all. `validate_truncation_radius` joins the existing per-attr
validator family and runs in the scene compiler and in every
`AdditiveSubLOD.__post_init__` (the tree writer reads each sub-LOD's own value,
so validating only the first would let a bad radius on a later rung reach disk).

Its lower bound is derived rather than magic, and is checked in **float32**:
the CUDA and Metal kernels and the GPU shaders evaluate the shift in single
precision, where `exp(-T^2/2)` saturates to 1.0 around `T = 3e-4` — four orders
of magnitude before float64's ~1.5e-8. A float64 check would admit radii that
are finite in Python and infinite on the GPU.

On the viewer side the attr is now also clamped at `createGSplatsNode`, the
single earliest read. This is defence in depth rather than a bug fix: every
current consumer (`gsplat-geometry`, `gsplats-adapter`, the two `readTruncate`
helpers) sizes itself from `material.uniforms.uTruncate`, which the material
constructors already clamp. It matters if a future consumer reads the raw attr,
or a material is built outside those constructors. The clamp itself now uses
the same float32 bounds as the writer: a radius whose value or square would
narrow to Infinity in a float32 uniform (`uTruncate` / `uTruncateSq`; e.g.
`1e308` or `1e30`, both finite in JS) falls back to the default,
and the lower floor is the bisected float32 degeneracy bound (~2.4e-4) rather
than the former 0.1 — so small-but-valid radii the validator accepts render at
their stored value, consistent with the chunk bounds computed from them.

Separately, the Points/Lines sprite falloff constants (`K = ln 100`, the 1%
iso-contour floor, and the renormalization) were duplicated across eight
GLSL/TSL/picking shader sites with no shared symbol. They now come from
`rendering/materials/_shared/falloff.ts`. The emitted shader text is
byte-identical — the checked-in codegen snapshots are unchanged, which is the
proof — and new tests pin the serialization plus a GLSL3-vs-TSL drift guard
that the codegen snapshots (TSL-only) could not provide.

#### Fixed — points nodes now record `ndim`, and expose their spatial metadata (#1150)

Points was the only geometry type whose writer never stamped `ndim` on the
group, even though the attr was already reserved against user override — so it
could be neither written nor supplied. The viewer's points chunk-index loader
cross-checks `chunk_bounds` dimensionality against it and skips the check when
absent, so for points that check had never once run. It is now armed;
dimensionality is verified to agree for 2D through 5D, including compound
(spatial + discrete) ordering. Scenes written before this change still load —
the loader's `undefined` guard is unchanged.

`Points` also gains the three metadata properties `Lines` already had:
`max_radius`, `has_spatial_index` and `ordering`. Adding the last of these
exposed a second gap — the writer put `ordering` on disk but not in the
metadata backing the property, so it would have reported `"none"` for a node
that is hilbert-ordered. The writer now returns it, matching Lines and GSplats.

`docs/guides/user/LUXAR_ZARR_FORMAT.md` records the rule the three types follow:
one ordering per type means flat keys, several means namespaced objects (Lines
is the only type with two), and an absent `ordering` attr means `"none"`.

#### Demos — Biodiversity at Planetary Scale (GBIF + Movebank)

New `biodiversity_planetary_scale` demo, and the first one in the ecology
problem space: a Blue Marble globe carrying a 15M-record sample of GBIF's 3.7
billion georeferenced species occurrences as Points, plus CC0 Movebank animal
tracks as Lines, with `taxon` (categorical, 10 categories: `All life` plus 9
groups) and `period` (categorical, 14 categories: `All years` plus 13 decades,
1900s-2020s) as non-displayed dimensions. Reads the GBIF AWS Open Data parquet
snapshot directly and anonymously (250 random parts, 9 of 50 columns projected —
97.9M rows scanned, 76.1M kept, in 55 s at 48 threads).

**Lines whose time coordinate advances along the chain** — new to the repo;
every previous 4D Lines demo holds `t` constant per polyline. Migration
worldlines vary it, and the Liang-Barsky clipper handles it: a segment straddling
the slab is drawn *clipped*, so a boundary segment reads as a whisker that grows
and shrinks as you scrub. Verified arithmetically on a 40-track prototype (520
segments = 440 within-slice + 80 straddlers), then on real data. Track time is
binned to a whole decade deliberately: off-grid discrete values are only fetched
within a quarter-step of the grid, so fractional values would work on a small
scene and fail silently on a large one — and an A/B showed they render
identically anyway.

**Context layers use `extend_to_all`; selection layers use real coordinate
slots.** The globe must survive every scrub or the selected records are left
floating in black — which is what `extend_to_all` is for, and it was broken
(#1157: a fully-extended node was never queried at all).
That defect was found while building this demo, filed with a self-contained
repro, and **is now fixed**; verified here before the workaround (a 25k globe
replicated into all 139 slots, 3.5M elements) was removed:

- the issue's control scene loads its extended layer to the full 1,200,000
  points and holds it byte-identically across every scrub, where it previously
  fetched nothing;
- an `extend_to_all` + `substitutive_lod` partition-of-LOD holds 318,686
  elements identically at every `(taxon, period)` combination with the coarse
  gsplat level intact and no "filtered out during nD→3D processing" warnings.

The globe is now one `extend_to_all` layer at a fixed resolution (see the
textured-shell note below for why it carries no LOD of its own).

The scrubbable layers do the opposite — real `(taxon, period)` coordinates, so
scrubbing isolates — and **every reachable slot is materialised**: 9 taxon
marginals, 13 period marginals, 117 joint cells. That is not an `extend_to_all`
matter but a consequence of the viewer showing the *intersection* of the
non-displayed slices, so a joint-only layer leaves one-slider moves on an empty
slot. Two density findings, both measured: time is binned by **decade** (at year
granularity the median populated cell held 244 points and 688 held under 500;
by decade, 108/108 cells at median 2,390), and the reservoirs are **stratified
over the cross product during the read** — sampling joint cells from a per-taxon
reservoir gave `Birds`/`1960s` just 258 points, because bird records are
overwhelmingly recent eBird.

Scaling is a `kind=partition` wrapper whose every child is a per-tile `kind=lod`
ladder — the Points counterpart of the gsplat `adaptive` recipe, and the first
use of that shape for Points in the repo. Tiling alone does not bound cost (a
partition renders every part, and a `stream:` ladder is progressive so it
converges to 100% regardless of distance); built that way, whole-globe framing
held all 18.1M elements resident. With per-tile LOD and measured
`coverage_fractions` it holds **~0.5M**, and zooming into a tile walks that tile
up to its full 1,875,000-point finest level while neighbours stay coarse.

Two calibration notes, both measured in-browser: the default
`coverage_fraction = sqrt(N_i/N_finest)` is calibrated for a single lod group
filling the screen, so with T tiles (each ~0.6 of the viewport diagonal at
whole-globe) it still selects a mid level; and a threshold placed *on* that 0.6
metric makes the tiles flap, leaving two levels cross-faded and resident at once
(1.07M instead of 186k) because the selector's hysteresis is 10% and
downgrade-only.

On the `extend_to_all` fix itself: #1157 was that a fully-extended context node
was never queried — `deriveNodeViewState` returned its full-extend `skip` before
applying the tolerance override that implements the extension — so its arrays
never loaded and its ladder froze. That is fixed on main (#1167), which is what
lets the globe be a single extended layer. The scrubbable layers stay on real
`(taxon, period)` slots for the intersection reason above, not because of this
bug.

Also worth knowing when reusing the recipe: `partition=` must be *omitted* from
the per-tile calls (even the documented `partition=False` bypass trips the
mutual-exclusion guard, which tests `partition is not None`), and only `opacity`
propagates from a `kind=lod` group to its children's materials — `intensity` and
`gamma` leave the child uniforms at 1, so appearance trims must be baked into the
per-element colours.

Rendering settings were tuned interactively in the Layers panel and then baked,
not guessed. Two things generalise beyond this demo. **Absorption and brightness
are a coupled pair**: raising `absorption` is what makes a point cloud read as an
opaque material (one that can still be made slightly transparent, which a truly
opaque mode cannot), but it also drives the layer nearly black — so push
brightness up in the same move by lowering the display-range max, which raises
`intensity` (the panel's DISPLAY RANGE is a window, `intensity = 1/(hi-lo)`). And
**a textured shell cannot survive Gaussian merging**: giving the globe a
substitutive LOD turned its coarsest level into 46k merged splats per 750k-point
tile, which under volumetric absorption rendered as huge dark ellipsoids. Coarse
levels read as *density* — meaningful for the diffuse occurrence cloud, wrong for
a continuous surface — so the globe is a fixed-resolution backdrop instead.

Also fixed while transcribing: `intensity` is capped at 100 by
`validate_intensity` (a 250 build fails outright, and costs nothing because both
saturate); only `opacity` propagates from a `kind=lod` group to its children, so
compositing attrs ride on the `layer=True` partition wrapper; and deriving that
wrapper's `position_bounds` from a 3-column array in a 5-D scene dropped a whole
BSP tile, rendering the globe with a wedge missing.

Data handling is documented too, including why the measured 73.0% bird share of
the filtered sample is *not* GBIF's ~60% (the filters are not taxon-neutral), the
per-record `coordinateuncertaintyinmeters` jitter that breaks up
rounded-coordinate lattices, and the CC-BY/CC0-only license filter.

#### Tooling — documentation checker is now a baseline-driven ratchet (#776)

`scripts/check_documentation.py` no longer fails all-or-nothing on pre-existing
debt. Existing missing READMEs/docstrings/JSDoc are captured in a checked-in
baseline (`scripts/docs_baseline.json`); a flagless run tolerates baselined
findings and fails only on NEW ones. A `--json` mode emits a deterministic,
machine-readable report with a `ratchet` block (new/fixed/still-present), and a
malformed baseline now reports a clean one-line error instead of a traceback.
Regenerate or tighten the baseline with `--update-baseline`; see
`docs/guides/developer/DOCUMENTATION_QUALITY.md`.

#### Tests — demo dimension ranges are checked against generated scenes (#799)

Small end-to-end builds now pin the dimension declarations in
`network_performance` and `particle_collision_animated`. The tests fail on any
`outside declared range` warning and compare the persisted declarations with
the stored scene extent, so a fixed-but-too-narrow range and an unnecessarily
wide range are both caught. The network test protects the data-derived x/y/z
ranges for its Gaussian clusters; the animated collision test protects frame 0,
the final frame, and the discrete time step.

#### Performance — indexed-Lines partitioning scales to ribbon-heavy datasets (#1103)

The shared `add_lines(partition=...)` path identified indexed connected
components with a Python union-find and then, for every component, scanned the
full vertex-root array again. That grouping was `O(components × vertices)`:
the ocean-currents shape of 220,000 ribbons / 11.66M vertices turned scene
assembly into a minutes-long CPU and transient-memory stall. Edge partitioning
then compounded it with one Python tuple per edge and a global→local dictionary
per part.

Indexed components now use vectorized root hooking (`np.minimum.at`) with
pointer jumping, followed by one stable root-label sort. Components and their
members have an explicit deterministic order by smallest vertex index. The
partition writer likewise builds one stable edge permutation and reuses a
single vertex map for part-local remapping, preserving exact topology and edge
order without the tuple/dictionary expansion. On a 20,000 × 53-vertex ribbon
benchmark the component pass drops from 5.09 s to 0.024 s (212×); the full
220,000-ribbon component + edge-grouping preparation completes in 0.44 s on the
same machine, before BSP and zarr writes.

Along the way, a partition part containing only isolated vertices (an
indexed graph never draws a vertex no segment references) is now skipped
instead of degraded to ``segments``: the degrade fabricated visible edges
between distinct isolated vertices, desynced per-vertex attributes on
odd-sized parts, and crashed outright on one-vertex parts. An indexed
partition with no edges at all is refused with the same error as the
single-leaf writer.

#### Fixed — colormapped Points/Lines apply authored intensity once; solar-system demo re-tuned (#1082)

`#1081` stopped a colormapped node applying an authored `intensity`/`offset`
twice (once as the scalar LUT window, once as a post-LUT color gain), but only
the **gsplats** node factory was updated. Its layers-panel half
(`applyColorAdjustments` / `LayerApplyEngine`) is node-kind agnostic, so Points
and Lines were left meaning two different things depending on the `layer` flag:
with `layer=true` the panel reset the gain and windowed the value, while with
`layer=false` the factory kept it as a gain and left the window at the bare data
range.

The Points and Lines factories now match gsplats: identity color GOG when a
colormap takes over, with the authored value re-expressed as the scalar window.
All three factories use the shared
`rendering/display-range.ts::resolveColormapWindow`; the gsplat inline copy was
folded into it without changing behavior. The identity-vs-window decision keys
on the raw leaf gain so an ancestor-only gain still folds onto the data range
rather than replacing it. Points therefore threads optional `leafAttrs`
alongside composed `attrs`, as Lines and gsplats already do.

The semantics correction also exposed a demo tuning dependency:
`asteroids_solar_system` authored `intensity=0.09` on its 1.55M-point
colormapped asteroid node. The full catalog's scalar range is approximately
`[0.46, 14510]` AU, so replacing that authored window with the identity range
would collapse the main belt and Trojan clouds into the first tiny fraction of
the LUT. The demo keeps `0.09` as the `[0, ~11 AU]` scientific display window
and moves brightness to `opacity`. Because additive blending accumulates,
`asteroid_opacity(n_visible)` holds `opacity × N` constant across the full
catalog, `--max-asteroids` subsets, animated per-frame subsamples, and future
catalog growth. The independently dim orbit ellipses are also brightened to
read as continuous reference lines.

#### Fixed — pnpm security pins single-sourced in `pnpm-workspace.yaml` (#1030)

`packages/luxar-viewer/` declared pnpm `overrides` in two places at once:
`package.json` carried the current advisory pins (`ws`, `brace-expansion` 5.x,
`esbuild`, `form-data`, `linkify-it`, `markdown-it`, `js-yaml`), while
`pnpm-workspace.yaml` still carried the set they replaced back in June
(`postcss`, `rollup`, `minimatch` ×2, `brace-expansion` 1.x/2.x, `picomatch`,
`flatted`, `ajv`, `diff`). pnpm silently prefers `package.json`, so the
workspace block had been inert on `main` for two months and nothing looked
wrong — but Dependabot's updater reads `pnpm-workspace.yaml`, so every viewer
bump it opened regenerated the lockfile around the *stale* set and then failed
`typescript-tests` and `release-readiness` at the first
`pnpm install --frozen-lockfile` with `ERR_PNPM_LOCKFILE_CONFIG_MISMATCH`. That
took out #1024, #1025, #1026, #1027 and #1028 together, and would have taken out
every future one.

The pins now live only in `pnpm-workspace.yaml`, which is the file both
consumers agree on. The move is resolution-neutral: `pnpm install
--frozen-lockfile` passes against the existing lockfile with zero churn, because
the applied override set never changes. The ten stale entries are dropped rather
than merged — none of them binds anything in the current graph (`postcss`
8.5.23, `picomatch` 4.0.4/4.0.5, `flatted` 3.4.2/3.4.3, `ajv` 6.15.0,
`minimatch` 10.2.5, `brace-expansion` 5.0.8 all sit outside the advisory ranges,
and `rollup` and `diff` left the graph entirely with the rolldown-vite
migration), so removing them changes no resolved version.

A new `pnpm run check:overrides` gate fails the build if `package.json` ever
regains a `pnpm.overrides` block, if the lockfile's recorded overrides drift
from `pnpm-workspace.yaml`, or if the pin block vanishes entirely — the second
reporting both blocks side by side instead of pnpm's opaque mismatch code, and
the third closing a hole the first two leave open (both hold trivially at zero
pins, and `pnpm audit` is `continue-on-error`, so a total pin loss had no gate
at all). It runs inside `check:ci` and, in `ci.yml`, as its own step *before*
`pnpm install --frozen-lockfile` in both `typescript-tests` and
`release-readiness` — otherwise the frozen install aborts first and the PRs that
need the explanation never see it.

Because the pins now live in `pnpm-workspace.yaml`, the pnpm floor became
load-bearing. Measured against that file: 9 and ≤10.4 abort with "packages field
missing or empty" (it has no `packages:` key), 10.5.0/10.5.1 install *silently
without the pins* — the lockfile records zero overrides — and 10.5.2+ read them
correctly. So ≥10.6 is a conservative floor, chosen because the 10.5.0 window is
the one mode that drops the pins without saying so; `engines.pnpm` closes it,
since pnpm enforces that field itself (`ERR_PNPM_UNSUPPORTED_ENGINE`, no
`engine-strict` required) and so it is a real gate rather than documentation.
Three places disagreed with the floor and were
corrected — `publish.yml` and `publish-npm.yml` pinned pnpm **9** (they have
never run, being tag-triggered pre-launch, so this was a red job waiting to
happen rather than a silently unpinned release), `engines.pnpm` said
`>=9.0.0`, and the Makefile's `MIN_PNPM_MAJOR`,
which documents itself as mirroring `engines.pnpm`, still said `9`. The Makefile
check now compares major *and* minor, matching the existing Node check, since a
major-only test cannot express the 10.6 boundary. The now-obsolete half of the
pnpm-pinning rationale in `docs.yml` was rewritten to match.

#### Fixed — finalize no longer mistakes a zarr array for a child node (#1079)

`lod_backfill`'s display-type resolver returned early only for the leaf `type`
values it hardcoded; anything else fell through to "recurse into the finest
child". A zarr group's `keys()` lists its **arrays** as well as its sub-groups,
so a node reaching that branch while holding datasets could pick a `zarr.Array`
as its finest child. Two failure modes, both reproducible against the previous
release: an untyped array crashed the compiler with a bare
`AttributeError: 'Array' object has no attribute 'keys'`, and — quieter, and
worse — an array carrying a recognised `type` attr resolved *early* and wrote
the wrong `display_type` to the wrapper with no error at all. All four
child-iteration sites in the module now use `group_keys()`.

#### Fixed — Ctrl-C now stops batch/tiled fitting instead of draining the queue (#736)

The three parallel subprocess pools — `batch-fit run` (local multi-GPU) and
`gsplat fit -j N` for both uniform and content tiling — submitted every task up
front to a `ThreadPoolExecutor` and iterated `as_completed(...)` inside a bare
`with` block. On Ctrl-C the `KeyboardInterrupt` could not escape until
`Executor.__exit__` ran `shutdown(wait=True)` with the default
`cancel_futures=False`, so every still-queued task ran to completion first,
each freed worker thread spawning a fresh `luxar gsplat fit` subprocess — a
single Ctrl-C on a 500-tile run kept fitting for hours. All three sites now
catch the interrupt around the completion loop, set a stop flag (so a worker
that already dequeued its task bails before launching), shut the executor down
with `cancel_futures=True`, and re-raise. In-flight children still die on the
terminal's process-group SIGINT; the queue just no longer respawns behind them.

#### Fixed — npm library build no longer inlines a second THREE runtime (#743)

The publishable library build externalized only the exact module id `three`,
but the entry graph statically imports the `three/webgpu` and `three/tsl`
subpaths (TSL materials, WebGPU renderer). Array externals match ids exactly,
so those subpaths — and the `three.core.js` they pull in — were inlined into
`dist/lib/luxar-viewer.js`, shipping a duplicate THREE core next to the host's
peer `three` and breaking the single-runtime contract (`instanceof` checks,
texture interop), at ~2.5 MB of unminified bloat. The build now externalizes
`three` and every `three/*` subpath (all subpath exports of the peer package),
and the release-readiness guard (`scripts/check-lib-exports.mjs`) — which
previously grepped for `class WebGLRenderer`, a marker absent from the
webgpu/tsl/core bundles — now scans for markers that actually appear when
THREE source is inlined (`EventDispatcher`/`WebGLRenderer` class definitions,
the `REVISION` constant), in both classic Rollup and rolldown codegen forms.

#### Fixed — cache stores no longer mutate the shared OPFS directory when disposed mid-init (#1058)

A dispose that raced `MultiLevelCachingStore.init()` / `OPFSStore.init()`
could leak an undisposed OPFS store, run `?clear-cache`'s `clearAll()` on a
dead instance (wiping a newer same-URL store's directory), probe-write or
orphan-clean the shared per-dataset directory after teardown, or overwrite
good `_cache_meta.json` with an empty snapshot. `disposed` is now set
synchronously at `dispose()` entry and re-checked across every init await
(including inside the write probe and the orphan-cleanup crawl), the final
dispose-time metadata save is gated on a fully-completed init, and
`clear()`/`delete()`/validation setters are no-ops on a disposed store.
`dispose()` additionally awaits any in-flight `init()`/`clear()` before
resolving (an already-initiated OPFS operation cannot be cancelled),
concurrent `dispose()` callers all share that one completion (a second
caller no longer resolves early while the first is still draining), and
`clear()` re-checks `disposed` at each resumption point, so once `dispose()`
resolves no straggling wipe or write from the old store can touch a
directory a newer same-URL store has taken over.

### July 2026

#### Fixed — demo install hints now name the constrained requirement (#915)

The `--show-roundtrip` matplotlib guards in the eleven gsplat demos, the
`from umap import UMAP` guards in the network demos, and the CytoSelf demo's
Pillow thumbnail guard printed a bare `pip install <pkg>` — the unbounded form
`demos/README.md` forbids. All of them now go through `require_module`, so the
message names the pinned requirement (`matplotlib>=3.5.0`, `umap-learn>=0.5.0`,
`Pillow>=9.0.0`) with `luxar[demos]` offered as the alternative — rather than
sending someone after the whole heavyweight extra to draw a diagnostic plot —
and the wording comes from `INSTALL_SPECS` instead of being hand-copied into a
dozen-plus files.

The networkx guards in the CAIDA, HuRI and PPI demos went the same way — they
were hand-copying `networkx>=3.0` next to a duplicated import, and two of the
Louvain paths had no gate at all and died with a bare `ModuleNotFoundError`.
The two napari hints kept their `pip install` form (napari is a soft optional
outside every demo extra) but gained the `>=0.4.18,<0.8` cap, without which the
suggested command resolves napari 0.8 and drags `zarr>=3` past Luxar's pin.

A new guard in `test_demos_dependencies.py` scans every demo source — runtime
messages and docstrings alike — for a `pip install` naming a package the table
bounds, and fails if the bound is missing. It reports 23 offenders against the
tree before this change and none after, so the class cannot quietly grow back
the way the eleven matplotlib copies did.

#### Fixed — bound the HuRI demo's CORUM download and extraction

The HuRI demo's optional CORUM fallback fetched a remote ZIP with `stream=True`
but then buffered the whole compressed body via `r.content` and the whole
extracted member via `src.read()` — neither bounded. Since these third-party
CORUM URLs shift across releases (they are not controlled Luxar assets), an
unexpectedly large response or a highly-compressed member could exhaust memory.
The archive now streams to a temp file under a hard compressed-byte cap and the
member is extracted in bounded chunks under an uncompressed cap (the declared
size is the effective decompression-bomb guard, CPython clamps extraction to it),
staging both through a private per-invocation directory beside the destination
— so concurrent demo runs sharing the cache cannot clobber each other — and
atomically renaming the member into place only on success, so a failed attempt
can never leave a partial file behind (#684).

#### Fixed — clearing a node transform after finalize no longer desyncs metadata (#677)

Setting `Node.transform` / `Node.nd_transform` to `None` after the
`LuxarZarrCompiler` context exited (or after `Scene.to_zarr`) edited the raw
`.zattrs` while the consolidated `.zmetadata` stayed stale, leaving two
conflicting views. Post-finalize clears now warn and leave the sealed store
untouched — matching the assignment path — so raw and consolidated metadata
stay in agreement.

#### Fixed — authored colours are no longer contrast-stretched at load

The layers panel pushed an automatic display window into every layer material on
load, derived from `scalar_data_range || color_data_range || amplitude_data_range`.
For a **direct-colour** layer that windowed the authored RGB by its own spread —
a contrast stretch nobody asked for. A uniform grey `(0.72, 0.74, 0.78)` has
`color_data_range` `[0.72, 0.78]`, which maps to gain 16.7 / offset −12 and
renders as **saturated blue**. Replaying both rules over a generated corpus of
166 scenes found 111 layers whose starting window moves — **every one of them
direct-colour** (asteroid planets 8.3×, `collision/detector_geometry` 10×), and
not a single colormapped layer.

The window maps the *rendered value*, so it now follows what that value is: a
colormapped layer still windows on its scalar range (a linear `[0, 1]` window on
right-skewed gsplat amplitudes renders near-black), while a direct-colour layer
starts at the identity. `color_data_range` still sets the slider bounds, so
stretching authored colours remains one drag away, and toggling the colormap
re-defaults both window and bounds to the new mode.

#### Fixed — the Blend control now reaches every part of a partitioned layer

`add_gsplats_from_file` re-stamped `blending_mode` onto every child when grafting
a nested (`kind=partition` / `kind=lod`) `.gsplats.zarr`. The attr is
nearest-setter-wins, so each part shadowed the layer wrapper and the layers
panel's single Blend control did nothing — `tiles`, `overview` and `adaptive`
layers ignored it while flat/stream/levels layers responded. Compositing attrs
now ride on the wrapper only, matching every other writer, and within a layer's
subtree the panel treats the layer's mode as authoritative so scenes already on
disk are fixed too.

#### Fixed — picking resolved the wrong element on very large nodes

The pick pass carried the element index through a single `float` channel of
the RGBA32F pick buffer. float32 has a 24-bit mantissa, so it stops
representing consecutive integers past 16,777,216 — while a node's capacity
reaches 2^25 on a device reporting `maxTextureSize` 32768. Hovering such a
node resolved to a neighbouring element, silently.

The index is now split into two 16-bit halves in INT space
(`luxarElementIdParts` in `glsl-lib.ts` and its TSL twin), carried in the
`g` (low) and `a` (high) channels — `a` was an unused constant 1.0, and pick
materials are `NoBlending`, so the blend stage cannot touch it. `NoBlending`
alone is not sufficient, though: THREE's NodeMaterial appends
`DiffuseColor.w *= material.opacity` inside the fragment body, which would
scale the high half on the TSL path (the hand-written GLSL twins have no such
tail). The pick factories therefore pin `.opacity = 1` so that multiply is
provably identity. Both halves are ≤ 65535 and therefore exact, so the
round-trip is exact for every index a node can hold (the element-texture
layout caps a node at 44,728,320 elements). The vote key stride moves
from 2^24 to 2^27 to match: with the old multiplier, `(nodeId 1, element
2^24)` and `(nodeId 2, element 0)` hashed to the same bucket and merged their
votes. 2^27 clears the largest reachable element index (44,728,319, for points
on a 32768-texel device) while keeping every key exactly representable — a
larger stride such as 2^32 would push keys past 2^53 and start merging
adjacent elements instead.

Below 65536 the DECODE is unchanged — the high half is 0 and `g` still
holds the whole index, which is exactly what the old decoder read — so no
resolved id moves and ordinary scenes are unaffected. (The buffer bytes do
differ there: alpha went from a constant 1.0 to the zero high half.) Pinned
by a test.

#### Fixed — a scaling `nd_transform` no longer draws the wrong slice on a discrete dimension

An `nd_transform` with a non-unit `scale` on a discrete non-displayed dimension
rendered content from world slices the user had not selected. The inverse-query
design maps the world slice into the node's local space, and a scale makes that
mapping land **between** categories: `scale: 2` at world frame 7 inverts to local
3.5. The per-element membership window is a half-step (`|value − target| ≤ 0.5 ×
step`), so it admitted local 3 **and** local 4 — two frames belonging to world 6
and world 8, drawn together while the slider read 7. With `scale: 3` at world 7
(local 2.333) it silently admitted local 2.

The window itself was fine; its documented premise was not. It is calibrated for
slightly off-grid *data* against an on-grid *target*, and the whole stack
guarantees on-grid targets because discrete navigation snaps to the `k · step`
grid. A scaling `nd_transform` is the one thing that breaks that guarantee, and
it breaks it on the query side where no amount of window tuning helps (a strict
boundary would fix `scale: 2` and still get `scale: 3` wrong).

The fix applies the spec's own forward rule for discrete ordinals
(`effective = round(scale · original + offset)`): a world value that is the image
of no local grid point has no preimage and must display nothing.
`invertNdTransformForQuery` now walks the local grid candidates bracketing the
exact inverse, keeps the one whose forward image rounds to the queried world
value, and **snaps the query to it** — which makes the query exactly on-grid and
so removes the midpoint tie that caused the double-draw in the first place. When
no candidate qualifies it reports `noPreimage`, which rides the derived per-node
`ViewState` and makes each geometry's range query return an empty range list —
reusing every loader's existing "no visible elements → clear" path. One rule,
applied once per node per slice, so Points, Lines and GSplats are all fixed
together with no change to the WASM/TS projection kernels.

Testing the forward rule matters rather than testing whether the exact inverse
lands on the grid: those agree only for integer `scale`/`offset`, and fractional
scale on a discrete dimension is explicitly valid (spec §11.3). Under
`offset: 0.4`, `round(k + 0.4) = k` gives every world value a preimage, so an
inverse-on-grid test would have blanked such a node permanently.

Exemptions: categorical permutations (a bijection always has a preimage) and
`extend_to_all` dimensions — keyed off the node's `extend_to_all` name list, not
the 1e10 tolerance sentinel, since every Lines call site derives with
`applyPartialExtendTolerance: false` and never carries it.

#### Fixed — `extend_to_all` survives a large `nd_transform` scale

The inverse query rescaled every tolerance by `1 / |scale|`, including the 1e10
`extend_to_all` sentinel. Past `|scale| > 10` that lands under the `>= 1e9` floor
which every downstream extend check uses (`effective-radius-calculator`'s
`isExtendToAll`, `calculateSpatialQueryTolerance`, `fallbackQueryTolerance`), so a
dimension the node had extended silently went back to being sliced — under
ordinary unit-conversion scales (`{"scale": 1000}`, s → ms; the spec's flagship
ms → s example is the same conversion run the other way). An infinite tolerance
now passes through unscaled; only
finite tolerances carry a meaningful world→local conversion.

#### Changed — the `nd_transforms` demo is now a calibrated test bench

`demo_nd_transforms.py` was a "Multi-Instrument Observatory": three jittered
Gaussian blob clouds that looked identical at every time slice, so there was no
way to see whether `nd_transform` had done anything at all. It is now an
instrument. A ruler along X (one tick = one frame index), a cyan cursor column of
plain untransformed geometry marking the WORLD index, and one labelled row per
transform whose markers are 3D point-font digits printing their own LOCAL index
— so the gap between digit and cursor, read in ticks, *is* the transform. Faint
always-on ghosts mark every slot a row could light (a dark row means "no
preimage", not "failed to load"), and a `visible_range`-gated readout prints the
expected local index per row for the current slice. It covers affine
offset/scale/negative-scale/scale+offset, categorical permutations, nested-group
composition **order**, a 4x4 transform and an `nd_transform` on one group, and
doubles as the visual regression harness for the no-preimage fix above.

#### Added — `luxar demo deps` and `make install-demo-deps`

Demos deliberately keep heavyweight packages out of the core install, so a fresh
checkout lists every demo but cannot run them all. `luxar demo deps` reports
which optional demo dependencies are missing (exit 1 if any are) and, with
`--install`, installs the Luxar extras that provide them; `--extra
demos|io|gsplats` narrows the report to one extra. `make install-demo-deps`
installs all three demo extras in one step. One tabled dependency, `gdown`, is
deliberately in no extra (it serves only the Google-Drive download path), so
neither covers it — the report and `--install` both name it for an individual
`pip install` instead. The report and the runtime `require_module` gate are
both driven by `luxar.demos._dependencies.INSTALL_SPECS`, so a package cannot
be advertised without being installable. The report is version-aware: an
installed package whose version is below its pinned floor is flagged `OUTDATED`
(not `ok`) and counted toward the exit-1 gate, so `deps` no longer passes an
environment that would still crash a demo (e.g. `scipy` old enough to lack
`scipy.special.sph_harm_y`). Newly tabled pins: `pooch`, `scikit-learn`,
`matplotlib`.

#### Fixed — the volumetric Absorption slider did nothing on thin geometry

κ is a physical coefficient with units of 1/length: the volumetric shaders build
optical depth as `τ = κ · density · through-thickness`, where the thickness is
the geometry's own world size (`width · √(π/ln 100)` for lines, `radius · …` for
points, the ray integral through Σ for gsplats). The layers panel offered a fixed
**0–10** track, so on the 3D-Hilbert-curve demo's 1.5e-3-wide lines the WHOLE
slider spanned τ ≤ 0.012 — a sub-1/255 change, i.e. a knob that visibly did
nothing. (Switching to `max` mode appeared to "make absorption work"; that was
the mode change itself — κ is not read in `max` at all.)

The track is now **logarithmic with bounds re-derived per layer** from the
thickness the writer already records (`max_width` / `max_radius`; the thinnest
descendant sets the top, since one κ drives the whole subtree, and the thickest
anchors the floor so a mixed-thickness group can still reach near-transparency
for its fattest geometry), so its top lands near
τ = 5 — opaque — whatever the scene's units. That 1.5e-3-wide line now reaches
κ ≈ 4.0e3; sweeping the track moves mean luminance 53 → 21 where it used to move
one 8-bit level. Gsplats carry no comparable thickness stat and their
`τ = κ·opacity·rayMass` is already O(1)-calibrated for fitted volumes, so they
keep the historical 0.001–10 span — also the floor of every derived bound, so an
authored κ ≤ 10 stays reachable. Position 0 is a dedicated stop for exactly
κ = 0, the additive limit, and the floor lowers onto a smaller authored κ so the
value the readout shows is always the value the thumb represents.

#### Fixed — nD scenes were framed around a non-displayed axis on load

Auto-framing, scene scale, clipping planes and the near-cull margin all project
the nD `position_bounds` through `sceneDimsManager`'s displayed dims, which fall
back to `[0, 1, 2]` when it is uninitialised — and the dimension-navigation UI
only initialised it *after* the scene load resolved. So any scene whose displayed
dims are not the first three (a leading non-displayed time / channel / order
axis — the common nD shape) was framed around the wrong axes: that axis' extent
landed on world X, putting the look-at target off to one side of the geometry and
inflating the fit distance by its range. The Hilbert demo opened at target
`(2.50, 0, 0)` with diagonal 5.19 instead of `(0, 0, 0)` and 1.73 — off-centre at
3× over-zoom, which pressing `F` then "fixed" (that path measures loaded
geometry instead of metadata). The dims are now resolved from the freshly loaded
scene before anything reads bounds, and stale dims are dropped when a scene
carries no dimension metadata so a 3D scene loaded after an nD one cannot
inherit its axes.

#### Fixed — overlay HTML sanitizer: attribute allowlist + reverse-tabnabbing (#767)

`OverlayManager.sanitizeHtml` allowlisted tags but only denylisted attributes,
so everything the earlier pass did not explicitly name survived — `id`/`name`
(DOM clobbering), `data-*`, `ping`, `srcset`, `download`, and the `vbscript:`,
`data:` and `style: url(javascript:...)` vectors the #720 note had flagged as
still uncovered. The scrub is now an attribute **allowlist**: only `style`,
`href`, `src`, `alt`, `class`, `target`, `title`, `rel` and the inert
presentational `colspan`/`rowspan`/`width`/`height` survive, and every
other attribute (including `on*` handlers) is dropped. The value-bearing
survivors then face a per-attribute guard: `href`/`src` block the
`javascript:`, `vbscript:` and `data:` schemes; `style` is dropped if it carries
`javascript:`, `vbscript:` or `expression(` (which also catches
`url(javascript:...)` after whitespace/C0 normalization), or any CSS escape
(`\`) or comment opener (`/*`) — a substring check cannot see through CSS
tokenization (`\6a avascript:` decodes to `javascript:`), so escape/comment
syntax is rejected wholesale rather than parsed. Reverse tabnabbing is
neutralized on both fronts: `rel` is dropped when it carries a bare `opener`
token, and `target` is restricted to `_blank`/`_self` so a named target can no
longer open a top-level window with a live `window.opener` able to
cross-origin-navigate the viewer tab. Over-blocking is the deliberate
preference: this sanitizer is the only XSS control on the `?src=<url>` path,
where a hand-crafted zarr never meets the Python compiler.

#### Changed — the two PDB structure demos render as surfaces, not emissive media

`nuclear_pore_complex` and `atp_synthase` shipped on the default `additive`
blending, which sums every atom along the view ray. A dense atomic shell washes
toward pastel white that way, and both demos held it back with an intensity
anti-blowout workaround (0.125 and 0.0625) that left them dim. An atomic
structure is a *surface*: both nodes now use depth-sorted `normal` blending at
full exposure (opacity 1.0, intensity 1.0), so the nearest atom wins the pixel.
Measured at the opening framing as mean CIELAB chroma over the covered pixels,
the NPC goes 6.2 → 13.1; on ATP the chain hues go 44.8 → 57.8 at lightness
L\* 40.2 → 76.4, i.e. the subunits stop reading as a dark wash. Volumetric was
tried across kappa 2–20 and loses the colours at every setting (3.9–8.3), so it
is not the default — but both nodes are now `layer=True`, so blending, opacity
and the display range (and absorption, once you pick `volumetric`) are live in
the Layers panel. `docs/images/readme/gallery/atp_synthase.{webp,webm}` were
recaptured through the gallery harness. Regenerate the demo datasets to pick up
the new look.

#### Fixed — warnings now display through arbol instead of raw stderr lines

Python's default warning display wrote `path/to/file.py:299: UserWarning: ...`
straight to stderr, landing out of place in the middle of arbol's hierarchical
console output (e.g. the Cholesky covariance-certificate escalation warning
during gsplat scene compiles). Warning *display* is now routed through
`aprint` as `⚠️ UserWarning: ... [file.py:299]` tree lines: process-wide in
every `luxar` CLI run, and scoped around the arbol-tree-producing Python API
entry points (`LuxarZarrCompiler` write methods, `fit_gaussian_splats`,
`generate_seeds`, `save_gsplats`). Display-only by design — warning semantics
(filters, `-W error`, `catch_warnings`, `pytest.warns`) are unchanged, and the
override steps aside whenever a recorder or custom `showwarning` hook owns
warning display. New module: `luxar.utils.arbol_warnings`.

#### Fixed — depth-sort orderings swap atomically (no more mid-rotation flicker)

Rotating a large `normal`/`volumetric` node drew a **corrupt permutation**:
some elements twice, an equal number not at all. The chunked ordering apply
(perf lever L8) streamed slices into the LIVE `aSortedIndex` attribute and
accepted the intermediate `new[0,cursor) ∪ old[cursor,n)` as bounded transient
shimmer. The bound was real; the premise that it stays transient was not —
under a continuous orbit a new sort arrives about as fast as a stream drains,
so the mix is the steady state. Measured with a browser probe that validates
the drawn index buffer every sampled frame: **27–33% of frames** on the 1.9M
`visible_human_head` (27,613 double-drawn) and **70–80%** on the 8M
`global_rivers_earth` terrain (up to 1,022,162 double-drawn, 12.8% of the
node). It surfaced now because the bioimaging demos moved to `volumetric`,
which is order-dependent where `additive` was not.

Orderings now stream into the **inactive** buffer of an `aSortedIndex` /
`aSortedIndexB` pair and a runtime `uSortedIndexSlot` uniform flips once that
buffer holds the whole permutation — the A/B design
`GSPLAT_DEPTH_SORTING_SPEC.md` §2.1 tier 3 specced and deferred. Both buffers
are allocated at attach, so every node pays +4 B/element whether it sorts or
not. Materialising the second one lazily (the obvious saving, and how this
first landed) is unsafe on the **native WebGPU** backend: three keys a
pipeline's vertex-buffer layout by BufferAttribute identity but rebuilds the
pipeline only on a name-level cache-key change, so growing the attribute set
after first render shifted every later attribute down a vertex-buffer slot —
the quad-corner attribute read the ordering buffer's `u32`s as `vec2<f32>` and
the scene rendered black, with no validation error and no console warning.
WebGL binds by program location and never saw it. The per-frame upload bound L8
bought is unchanged. Sorting and applying now run concurrently (the dispatch
apply-gate is gone).

Trade, measured on the 10M orbit bench: sort-adjacent frame p99 ~77 → ~92 ms
(median and p95 unchanged, still far under the 119–563 ms chunking prevents),
and waiting for a whole ordering instead of showing a partly-applied one costs
some freshness (8M fast-orbit sort-axis lag 36.5° → 44.3° mean). Both are the
deliberate price of never drawing a corrupt permutation.

#### Added — spatial partitioning (BSP tiling) now works on 2D data

`luxar gsplat partition`, `lod --recipe tiles|overview|adaptive`, and
`GSplatData.to_spatial_partition()` all crashed on planar input with
"needs at least 3 spatial dimensions", and `scene.add_gsplats(partition=…)` /
`add_points` / `add_lines` **silently** dropped the request and wrote one
un-partitioned leaf instead — including when a compiler-level
`auto_partition_max_elements` asked for it. The victim was exactly the data that
most needs tiling: whole-slide 2D imagery like the 46K×32K `cmu1` pathology demo.

The limit turned out to be entry-point-only. The recursive splitters already pick
`argmax(extents)` over whatever columns they are handed, so median/midpoint were
always dimension-generic; only the guards (`shape[1] < 3`), the `positions[:, :3]`
slice, and two spots in SAH — its `range(3)` axis loop and its 3D
surface-area cost proxy — assumed three axes. SAH now uses the box **perimeter**
in 2D, which is the correct boundary measure there (a random ray's hit
probability scales with surface area in 3D, perimeter in 2D). Only 1D is still
rejected, and the three adders now warn via a shared
`warn_if_partition_needs_more_dims` helper — sibling of the existing
`warn_if_oversized_single_part` — rather than dropping the request in silence.

#### Fixed — BSP part ordering read split axes as x/y/z regardless of displayDims

A serialized BSP `axis` is a center-column index, but the depth-sort
coordinator's `eyeLocal` is in display space, where x/y/z are `displayDims[0..2]`.
The two coincide only for `displayDims == [0, 1, 2]`, so a 4D scene displaying
`[1, 2, 3]` ordered its partition parts along the wrong axis — silently, since the
result is still a valid permutation. The axis is now mapped through the **live**
display dims (not a load-time snapshot, which nD navigation would invalidate),
and a split on a currently-undisplayed column makes the coordinator decline the
tree and fall back to the documented centroid heuristic rather than order along
an axis the viewer isn't showing.

#### Fixed — marginal Cholesky inflated splats in small-unit scenes

The degenerate-variance floor in `compute_marginal_cholesky` was an absolute
constant (1e-10), but a variance carries world-units², so the threshold silently
answered "does this axis have extent?" by scene scale: a splat with σ = 1e-7
(nm-unit data) has variance 1e-14, tripped the floor, and was regularized up to
σ = 1e-5 — 100× larger than authored, and 10⁴× at σ = 1e-9. A 3D scene displaying
`[0, 1, 2]` escaped through the standard-3D fast path, which copies the factor
verbatim, but a 2D scene — or any nD scene with hidden dims — always goes through
the marginal.

The floor is now anchored to the largest diagonal of `Σ_S`
(`CHOLESKY_RELATIVE_EPSILON = 1e-12`), making it a pure condition-number check
that behaves identically at every scene scale, with the absolute constant kept as
a backstop for a genuinely scaleless (all-zero) covariance. Same
scale-free-conditioning reasoning as the shader's trace-normalized covariance
inverse. Rank-deficient axes are still regularized, now as a fixed _fraction_ of
the real axis.

#### Changed — ACES is now the recommended tone mapping, and choosing it explicitly no longer warns

`ACES` is the right tone mapping for almost every scene — its filmic highlight
rolloff is what keeps bright, dense structure from clipping flat — and it is
already the viewer's default. The tree was built around the opposite
assumption: seventeen demos pinned `Neutral`, and the compiler warned authors
away from ACES.

The LUT tone-mapping warning in `io/_compiler/colormap.py` now fires **only
when the author set no `tone_mapping` at all**. Its predicate was
`!= "Neutral"`, so an explicit `"ACES"` tripped it too — nagging about a
deliberate decision, while the message itself speaks of "the viewer's
*default*", which is only what you get by saying nothing. Any explicit value,
`"ACES"` included, now silences it. Fifteen demos move from `Neutral` to an
explicit `ACES`. Two keep `Neutral` as verified exceptions:
`demo_gsplats_3d_tribolium_embryo`, whose pairing with `exposure=1.97` was tuned
deliberately, and `demo_flywire_connectome`, where ACES blew its luminous
connection glow out into a white wash. The classical-capture interop demos also
stay on `Neutral`, via the `build_interop_scene` default — their baked per-splat
RGB is already display-referred, so ACES would distort it. `CLAUDE.md`, the HDR
guide, the `gsplat convert` CLI help and the `ViewerConfig.tone_mapping`
docstring all now recommend ACES, keeping `Neutral` for the narrower case where
a colormap LUT carries an exact scientific colour encoding. Regenerate the demo
datasets to pick up the new look.

#### Changed — five gsplat demos bake their preferred viewer appearance

The blanket `volumetric` + kappa 1.0 default from the bioimaging demo sweep was
wrong for scenes whose layers are *superimposed over the same specimen*: there,
emission-absorption makes whichever layer draws first occlude the other, so
channel overlap reads as one channel hiding the rest instead of the colours
mixing. The multi-channel organoid now composites `additive` (a pure sum, no
attenuation) and the 4D neuromast timelapse drops to absorption 0.05 — the
absorption slider's smallest non-zero step, which keeps volumetric's bounded
accumulation without the occlusion. At kappa 1.0 the neuromast rendered as a
dim blue haze with its hair-cell cluster and membrane filaments lost.

The Tribolium embryo switches to `normal`. That light-sheet volume carries a
heavy diffuse background, and integrating it along every ray saturates into a
solid slab with the embryo buried inside; `normal` composites the projected
2D-Gaussian peak with alpha-over instead, so the background stops accumulating
and the surface nuclei stay crisp. Because nothing sums any more the scene
needs `exposure=1.97` to sit at a normal level.

Two appearance tweaks round it out: the Milky Way dust cube moves from
`additive` to a light `volumetric` (absorption 0.3, so near dust softly
occludes far dust) under ACES at `exposure=-0.17`, with a `[0, 0.095]` display
window that holds its faint diffuse filaments just below clipping; and the
organoid DAPI nuclei demo gains a `plasma` colormap (it previously fell back to
the implicit grayscale default). Regenerate the demo datasets to pick up the new
look.

#### Fixed — errors logged as trailing arguments rendered as `{}` in the in-app console

A bug report contained the line `OPFSStore metadata save failed {}` — the cause
entirely absent. `log.*` is a pass-through to `console.*`, so an error passed as a
trailing argument reaches the in-app debug console as an object, and both of that
console's renderers `JSON.stringify` it. An `Error`'s `name`, `message` and `stack`
are non-enumerable, so the result is `{}` — and the existing `String(arg)` fallback
never fires, because stringify *succeeds* at producing that empty object. Browser
devtools renders it correctly, which is why this went unnoticed; the in-app console
is what a user copies into an issue.

Fixed with an `Error` branch in both renderers, which covers all ~45 trailing-error
log sites with no call-site changes. The branch must precede the object branch: an
`Error` subclass that assigns own enumerable fields stringifies to a non-empty but
still message-less object.

Stack capture was worse than missing. It only ever inspected the first argument,
which is always the formatted message *string*, so it never found the error behind
it — and when it failed it FABRICATED `new Error().stack` whenever the message
merely contained the word "error", producing a plausible trace rooted inside the
interceptor. It now scans arguments for a real error, the fabricator is gone,
warnings get stacks too, and the clipboard export includes the stack it was
silently dropping.

Also: the OPFS write path retried errors it was never meant to. Its own comment says
"one retry on stale bucket handle", but the catch only continued for that case and
then fell through with no `break`, so any out-of-space, quota or timeout error
immediately re-ran the whole write chain — with no backoff and no space reclaimed —
costing a duplicate warning per chunk and up to twice the operation timeout in
caller stall on a hung handle.

New `utils/format-error.ts` promotes an idiom that was inlined roughly 31 times.

#### Changed — volumetric joins the LOD anti-popping blendable set

`volumetric` is now in `BLENDABLE_MODES` (`packages/luxar-viewer/src/scene/lod-fade.ts`),
the predicate gating both LOD anti-popping mechanisms — the coverage-band
cross-fade between substitutive levels and the streaming `1/e(k)` brightness
compensation. Volumetric LOD scenes previously kept the hard visibility swap at
every level boundary and re-acquired the streaming brightening pop, a
documented phase-1 deferral of the volumetric mode
(`VOLUMETRIC_BLENDING_SPEC.md` §6); since the bioimaging demo sweep switched
the LOD showcase demos (tribolium ×2, embryo line) to volumetric, both
artifacts were user-visible there. The enabling physics: opacity linearly
scales optical depth (`τ = κ·opacity·intensity`), so a `w`/`1−w` cross-fade
composites to `1 − exp(−(w·τ_fine + (1−w)·τ_coarse))` — exact at the endpoints
and a monotone, always-bracketed log-space interpolation in between, which is
precisely the ghost-free dissolve anti-popping wants (it collapses to constant
absorption only where the two levels are per-ray mass-matched, which the
total-mass build invariant does not guarantee — so the code and spec explicitly
warn against assuming mid-fade invariance). The `1/e(k)` boost restores a
partial ladder's τ in **aggregate**, not per ray, since `e(k)` is a global
energy fraction over a subset of splats — the same structural approximation the
additive/luminous path has shipped since the compensation landed. On
individually optically-thick splats (`κ·splat-mass ≳ 1`) the boost saturates
emission instead of brightening — bounded by the shared 10× `ENERGY_FLOOR` cap
and transient, accepted as a single-set/shared-cap policy (spec §6 updated).
Also fixed en route: a layers-panel opacity edit during an in-flight LOD fade
was clobbered by the fade's next frame (the panel now rebases the fade's
snapshot, all modes), and stale per-node-material / single-visible-child
comments in `lod-fade.ts` / `lod-group-registry.ts`. New fixture
`test_lod_group_volumetric.luxar.zarr` covers the volumetric cross-fade
contract end-to-end.

#### Fixed — a rejected projection kernel was re-run on the UI thread, and load failures were reported as success

Four defects in the loader's failure handling, all pre-existing, all surfaced while
reviewing the 2D-gsplat WASM trap.

The worst turned a data error into a UI freeze: the gsplats and lines processors
fell back to the in-process dispatcher on ANY worker rejection except a
dataset-switch abort. That dispatcher runs the _same_ kernel through the same
`pickBackend`, so a WASM trap coming back from the worker trapped again on the main
thread, blocking the frame. Only worker UNAVAILABILITY now falls back, via
an `isWorkerInfrastructureError` allow-list matching the single pool-internal
error type (`WorkerUnavailableError`) by `instanceof` — it is constructed on
the main thread and never crosses the Comlink boundary, so its prototype stays
intact. A worker timeout propagates too: it cannot distinguish a wedged worker
from a data-dependent kernel hang or a projection genuinely slower than the
budget, and re-running those in-process blocks the frame at least as long
again — the pool evicts the timed-out worker, so the node stays retryable
against a fresh one. Unknown errors, and any rejection reconstructed from a
worker (which loses its prototype), fail closed and propagate. Points is
deliberately not included: its projection is main-thread-only, so it has no
worker path.

`loadScene` also logged "Scene loaded successfully" unconditionally. Because
`loadLeafNode` swallows every `LoaderError` so surviving siblings still render,
`loadScene` structurally cannot throw — so a scene whose every node failed produced
a green log over an empty viewport. A new end-of-load report grades the outcome:
clean keeps the historical message, partial logs one aggregate warning, and total
logs an error plus a long toast. The per-node failure toast is gone; with N failures
it showed one toast naming the last path, and never fired at all for network
failures.

All three handlers cleared a path's failure record immediately after the fetch,
though the record's scope is the whole load-and-stage step — so any post-fetch
failure re-recorded with a zero counter, pinning the log at "(attempt 1)" however
many times it failed, and `hasFailures()` briefly reported clean. And retries had no
notion of a permanent failure: the classified error kind was computed for logging
then discarded, and the retry counter was written in three places and never
compared, so a WASM trap was re-fetched on every reconnect forever. The kind is now
persisted and automatic retries are gated on a transient cause under an attempt cap
— while a manual Retry still forces every path, since the user pressing it is new
information.

#### Changed — Default viewer background is now pitch black (`0x000000`)

The scene background default was `0x111111` (dark gray, matched to the dark
theme's UI chrome) since the first commit. That color is rendered into the HDR
buffer, so the post-processing exposure chain treats it as scene light: at high
exposure an "empty" background lifted to gray and eventually white, and it sat
only ~1.8× under the default bloom threshold. The default is now pure black —
zero radiance, exposure- and bloom-invariant, and cleaner premultiplied-alpha
edges for transparent screenshots. Scenes can still author a tinted background
via `viewer_config.background_color` (the handful of demos that do are
unchanged). The dark theme's `#111111` UI panels are a separate token and keep
their color. Committed gallery/README media still show the old background until
regenerated (tracked as a follow-up).

#### Added — Points and Lines LOD levels now stream progressively (#811, #808)

`additive_lod` and `substitutive_lod` used to be mutually exclusive for Points
and Lines, which left the finest level of a substitutive ladder as the one node
in the LOD system that could not paint progressively: it committed
all-at-once however large it was. On the 9.75M-point DESI demo that single
commit froze the main thread for ~85 s. GSplats have always composed the two
axes, so this closes a three-geometry asymmetry as much as it fixes a stall.

The axes now compose — substitutive chooses _which_ level renders at the current
zoom, additive describes _how_ each level streams in — and every level is
laddered by default (`additive_lod=False` opts out), with the sibling-aware
first chunk on all but the coarsest. A level smaller than one stream chunk stays
a flat leaf automatically.

Supporting changes:

- `stream:<c>` breakpoints on Points and Lines, sharing the GSplats cut geometry
  via the new `luxar/utils/lod_breakpoints.py`. This matters: an equal-count
  split into 4 levels still ends with an N/4-sized commit, and the one existing
  Points ladder in the repo (`global_rivers_earth/terrain`) put 99.98% of its
  8M points in the final level — it streamed in name only.
- Energy quality stamps (`lod_stats.energy_fraction_cum` per sub-LOD,
  `level_stats.reference_energy` per leaf) so the never-downgrade gate can
  release a swap on committed energy rather than raw element count.
- On the GSplats additive ladder (shared here via `lod_breakpoints.py`),
  energy-fraction breakpoints (a `list[float]` of cumulative fractions) now
  place their cuts on the same O(N) self-energy cumulative the viewer reads
  back as `lod_stats.energy_fraction_cum`, for every score-ordered method
  (`self_energy` / `mass` / `amplitude` / `random`, and `method=auto` above
  N=5000). Cuts land where the on-disk energy stamp reports the requested
  fraction, and the coarse maximally-overlapping levels skip the O(nnz)
  sparse-Gram build entirely; only `greedy` / spectral orderings still cut on
  the residual-energy curve they already build a Gram for. Previously the
  score-ordered path built a Gram just to cut on a residual curve the viewer
  never sees.
- Hidden (`visible=false`) layers no longer fetch, decode and commit their LOD
  levels, and no longer escape eviction.
- `scripts/check_demo_ladders.py` — a structural gate that fails a leaf whose
  largest level is more than 60% of the data (the `--max-share` default) or
  exceeds the `--max-level-elements` absolute per-commit cap, which is exactly
  the degeneracy a level count alone cannot see.

#### Fixed — every 2D gsplats scene failed to load with a WASM `unreachable` trap

Loading a gsplats scene with fewer than 3 displayed dimensions failed with
`RuntimeError: unreachable` thrown from `project_gsplats_nd_to_3d`, leaving the
node unrendered (`Total gsplats loaded: 0`). Every 2D gsplats node in a scene
failed the same way, each after paying its full multi-million-splat fetch.

Root cause: the display-dims marginal Cholesky was computed with a hardcoded
sub-dimension count of 3 in both the fused `project_gsplats_nd_to_3d` and
`extract_visible_cholesky_3d`. A 2D scene supplies `display_dims.len() == 2`, so
Rust indexed `display_dims[2]` out of bounds — a panic, and the crate is
`panic = "abort"`, hence the wasm trap. The TypeScript reference had the same
bug but read `undefined` at that index and produced NaN-driven garbage instead of
trapping, so the production >16D backend was silently wrong rather than loud.
A regression: the pre-May-2026 main-thread path sized the loop from the data
(`subNdim = keepDims.length`), and the hardcoded 3 only became reachable when the
WASM kernel replaced it.

The marginal is now computed over however many display dims exist, and the
renderer's fixed 6-element packed-3D output is padded: zero off-diagonals (the
phantom axis is uncorrelated, leaving the in-plane profile untouched) plus a
phantom diagonal equal to the **geometric mean of the real Cholesky pivots** —
which is `(det Σ_S)^(1/2n)` and therefore rotation-invariant, so a 2D splat renders
as a round blob at its own in-plane scale. The phantom is deliberately **not** an
epsilon: in sum projection (additive, luminous, volumetric) the shader scales
amplitude by the Gaussian's extent along the view ray, `sigmaRay = 1/√(rᵀΣ⁻¹r)`, so
an ε-thin splat is scaled by ~1e-5 and discarded — the scene renders black. This
was settled by A/B-ing an ε-padded **3D twin** scene, which reaches the renderer
via the standard-3D fast path and so bypasses the marginal code entirely.
`luxar.gsplats.lift` depends on the scale-matched choice too: its
`opacity / (rayIntegralFactor · σ)` calibration holds for a 2D lift only because
`√(σ·σ) == σ`.

Pinned by 9 new Rust tests (2D/1D padding, linear scaling with splat size,
rotation invariance, the isotropic-2D-lift contract, degenerate/NaN and
empty-display-dims fallbacks), 3 cross-language parity tests, and 6
**unconditional** TypeScript-only tests — the parity suite is
`skipIf(!wasmFilesExist)`, so without those a TS-side regression in the >16D
backend would ship silently.

#### Fixed — the CI Python version matrix tested one version three times (#839)

`python-tests` declared a `['3.10', '3.11', '3.12']` matrix, but every leg ran the
tests under Python 3.12: `pipx install hatch` put Hatch on the runner's default
interpreter, and Hatch builds an environment that declares no `python` with whatever
interpreter Hatch itself runs under. `actions/setup-python` installed the requested
version and nothing downstream consumed it, so two of the three versions advertised by
`requires-python` and the PyPI classifiers had never once been executed.

Hatch is now installed onto the matrix interpreter, and a new step asserts the
environment's version **equals** the matrix leg before the tests run — the pre-existing
floor check (`>= 3.10`) passed happily while every leg ran 3.12, which is how this
stayed hidden. `fail-fast: false` means all three verdicts now come back from one run.

The interpreter-invariant gates (ruff, mypy, import-linter, bandit, version and contract
drift) now run once, on the 3.12 leg only, guarded by `if: matrix.python-version ==
'3.12'`. They judge the source, not the runtime — ruff is pinned to `target-version =
"py310"` and mypy to `python_version = "3.10"` — so a single run is enough, and keeping
them inside `python-tests` keeps them under the required `python-tests (3.12)` context: a
lint, type, security, or contract failure still blocks the merge. pip-audit runs on every
leg (advisory, `continue-on-error`) so each interpreter's dependency resolution is audited.

Also made `stats/generate_stats.py` import `tomllib` with a `tomli` fallback: it was the
one place in the tree that genuinely required 3.11+.

#### Changed — the bioimaging gsplat demos now bake `volumetric` blending

All 20 microscopy / bioimaging gsplat demos previously composited with
unbounded `additive` blending, so the front of a dense specimen never occluded
the back and a bright channel washed out the others rather than sitting in
front of them. On the 3-channel mouse embryo heart the SYTOX nuclear stain
covered the vasculature and cardiac-tissue channels almost entirely.

Eighteen 3D/4D demos now bake `blending_mode="volumetric"`
(emission-absorption, Max 1995) with absorption kappa 1.0. The two strictly 2D
slide reconstructions remain additive because every splat shares one depth
plane, so depth-ordered volumetric compositing would reduce to storage order.
Two converted demos carry tuned values: the acto3d heart drops to
`opacity=0.48` so its three channels read through one another, and the cryo-EM
capsid uses kappa 5.0 so the near side of the shell occludes the far side and
it reads as a hollow icosahedron. Regenerate the demo datasets to pick up the
new look. The astronomy gsplat demos and the classical-interop demos are
unchanged.

#### Fixed — camera-plane-crossing line segments rendered as razor-edged bands (one-sided cross-profile at close zoom)

Zooming very close to a thick line painted huge screen-filling bands with a
razor-sharp bright edge on one side and the smooth Gaussian falloff on the
other. Root cause: a segment with exactly ONE endpoint behind (or within
`uNearCull` of) the camera plane kept its full quad; the behind endpoint's
`clip.w ≤ 0` made the hardware rasterize the quad as an external (wrapped)
primitive whose near-clip boundary sliced mid-profile — a bright razor edge
running along the line's side.

The vertex stage (all four backends: visual + picking, GLSL + TSL) now
**clips the segment to the nearCull plane** before any screen-space math:
the offending endpoint is moved along the segment onto the plane (view-space
depth is linear, so the intersection is exact) and `t` is remapped so
per-endpoint attributes (width, color, sharpness, alpha, colormap scalars)
and the fragment cap math keep the original parameterization. Every vertex
then has `viewZ ≥ nearCull`: quads stay true trapezoids, the `wGuard` clamp
no longer disagrees with the rasterized geometry, and the cut end lands
exactly where the per-fragment near fade reaches zero — the approach to the
camera fades out smoothly instead of tearing. New `line-crossing` parity
harness entry pins the behavior with a content assertion (per-column
centroid on the projected centerline) that fails pre-fix on both backends.

#### Fixed — nodes embedded inside a larger node blinked out on orbit in order-dependent blending modes

Cross-node draw order sorted whole meshes by the view-depth of their
bounds centers — a single number that cannot express "this tiny node is
_inside_ that huge node." For a node embedded in another (the galaxy
demo's Sun/Betelgeuse/Rigel markers inside the 3M-star cloud), the
container's center sorts nearer for roughly half of all camera
orientations, making the container draw last; in `volumetric` (or
`normal`) blending its fragments then multiply the embedded node's
pixels by the container's whole transmittance — erasing it until the
camera orbits back past the flip point. A strict bounding-sphere
container now always draws before its contents (a priority topological
pass over the depth-sorted groups; provably acyclic), so embedded
content composites on top and stays visible from every angle.
Non-contained nodes keep the exact farthest-first order. The Gaia
galaxy demo now bakes `volumetric` blending (absorption 1.3) on all
four layers, with intensity retuned for volumetric's bounded
accumulation — regenerate the demo dataset to pick up the new look.

#### Fixed — demos authored continuous curves as exploded `segments`, defeating joint continuity (bead-chain gaps)

Nine demo line nodes (across six demos) built genuinely continuous curves (helix particle
tracks, detector rings, chromosome paths, jellyfish tentacles, L-system
tree skeletons, cell tracks and trails) and then exploded them into
duplicated start/end vertex pairs with `line_type="segments"`. The
viewer's joint-cap suppression matches joints by shared vertex INDEX, so
exploded authoring hides every joint — thick lines rendered as bead
chains (visible gaps between segments) even after the shader-side joint
fix. All nine nodes are now authored as `line_type="indexed"`: unique
per-vertex arrays + explicit per-curve edge lists, which also roughly
halves their vertex data. Converted: `collision` (particle tracks +
detector rings; neutral-particle tracks are now solid, dropping the
accidental bead-dashing), `collision_animated` (same, 4D), `dipc_3d_genome`
(chromosome paths), `bioluminescent_ocean` (tentacles/oral arms — the
per-segment 5%/2% end-of-segment tapers became smooth per-vertex tapers),
`lsystem_forest` (tree skeletons — a branching topology, deduplicated
exactly during turtle interpretation with branch points shared by 3+
edges), and `gsplats_4d_celegans_tracking` (cell tracks + fading trails —
the per-hop discrete fade became a smooth per-vertex fade). The other
seven `segments` call sites (connectome/interactome/AS-graph edges, velocity
comets, earthquake spikes, grid lines) are genuinely disconnected and
stay as-is. Regenerate demo datasets to pick up the fix.

- **Writer authoring lint**: `write_lines` now warns when
  `line_type="segments"` input looks like exploded continuous polylines
  (most consecutive segments sharing an endpoint coordinate), pointing at
  `polyline`/`indexed` authoring — the trap class is now self-diagnosing.
- **Indexed validation fix**: `(E, 2)` edge arrays with an ODD number of
  edges were wrongly rejected ("Indices must have even length") — the
  validator counted rows via `len()`, not elements; it now uses
  `indices.size`. Odd-edge-count geometry (e.g. most L-system trees)
  previously could not be written as pairs at all.
- **Mutable scene-data responses now send `Cache-Control: no-cache`** in
  `luxar serve`, native export launchers, and the standalone `serve.py` that
  `luxar export` generates. Responses previously carried only
  ETag/Last-Modified (or just Last-Modified for exports), so browsers used
  HEURISTIC freshness and silently served stale chunks after a dataset was
  regenerated — or a folder re-exported — in place (same URLs, new bytes);
  no viewer-side cache clearing could fix it. `no-cache` forces
  revalidation; unchanged files still return as cheap 304s. Content-hashed
  viewer assets remain cacheable without per-load revalidation.

#### Fixed — thick polylines rendered as bead chains: interior joint caps now suppressed per-endpoint (#780)

The line fragment shader dims every segment towards 0.5 at its own endpoints,
on the assumption that a neighbouring quad overlaps the endpoint and adds the
missing half back. But each quad spans exactly `[start, end]` — no
longitudinal extension — so collinear neighbours **tile** rather than overlap:
the halves never sum, and every interior joint of a thick polyline was a dark
notch of axial length `2 × width` bottoming out at 50%. (PR #785; follow-ups
#793/#795/#796.)

- The boolean "clipped" flag became a continuous per-endpoint **cap
  suppression scalar** in `[0, 1]` (`compute_cap_suppression` — Rust kernel +
  its TypeScript mirror, computed once per commit off the main thread, riding
  texel4.yz): `1.0` at a straight-through interior joint or a slice-clipped
  end (nothing will arrive to sum with), `0.0` at a ≥ 90° bend or a branch
  hub (there the quads genuinely do overlap — keep the cap) or a free
  polyline end (no neighbour at all — keep the soft cap), `cos θ` in
  between. Consumed by all four shader backends (visual + picking, GLSL +
  TSL). Joints are matched by vertex **index**, not position:
  `line_type="segments"` chains with per-segment duplicate points keep the
  cap (and the notch) — author connected geometry as `line_type="polyline"`.
- **Per-endpoint cap factor `min(startCap, endCap)`** (#796): the initial
  nearest-endpoint pick was discontinuous at the midpoint of segments shorter
  than `2 × width` whenever the two suppressions differ — the routine case
  for a polyline's first/last segment (one free end, one suppressed joint).
  Each endpoint's ramp is now lifted by its own suppression and the two
  combine with `min()`. Honest accounting: for sub-width segments this
  relocates a strictly smaller step to the joint seam (worst case
  `0.5 × (1 − L/w)`, zero for `L ≥ width`); polyline-wide C⁰ continuity needs
  join geometry, which is tracked separately.
- **`calculate_segment_lengths` now accumulates in f64** (#793,
  pre-existing): the Rust path overflowed the f32 squared-length to Infinity
  above a component delta of ~1.8e19, while the TypeScript mirror — the
  production backend above 16 dimensions — returned the true value, so the
  same scene disagreed across backends.
- Known limitations, documented in `materials/line/README.md` rather than
  fixed: the suppression angle is measured in data space once per commit
  while quad tiling/overlap is a screen-space, per-camera fact (#795 tracks a
  real screen-space suppression), and the outer-side miter wedge at sharp
  bends remains.

#### Added — manifest-driven demo-data fetch (R17 step 1)

`demos/data_manifest.json` is now the single source of truth for how every demo
dataset is obtained: its bucket (`zenodo` fetch-on-demand / `local-compute` /
`regenerate`), its true per-dataset license, and a sha256 per file.
`luxar.utils.data_fetch.ensure_dataset()` resolves a dataset through
cache → in-repo Git LFS → Zenodo, and `load_dataset_gsplats()` mirrors
`load_precomputed_gsplats`' contract so migrating a demo later is a one-line
swap. No demo is migrated yet and no Zenodo record exists yet, so the in-repo
Git LFS copy still serves every dataset.

The manifest is generated by `scripts/gen_data_manifest.py` (`make
gen-data-manifest`) and guarded in CI by `hatch run check-data-manifest`.
Checksums are read from the Git LFS pointers or the file bytes, so the generator
needs neither the `git-lfs` binary nor a git work tree — which is what lets the
drift gate run in CI, where the repo is deliberately checked out without LFS.

#### Fixed — a cached demo file that failed its checksum was used anyway

`ensure_dataset` logged `SHA256 mismatch` and then returned the corrupt file: the
fallback step asked a size+mtime staleness test whether to re-copy, and an
in-place corruption changes neither. The manifest checksum is now authoritative
at every step, and a failing cache entry is quarantined (`.corrupt`) instead of
being silently reused.

`cached_download` had the same flaw for an unpulled Git LFS pointer, which was
handed to the downloader as if it were real data; it now quarantines the LFS
pointer and the sha256-mismatch cases up front. A file whose size merely differs
from `expected_size` — _longer_ or _shorter_ — is left in place, since
`expected_size` is only a skip-if-matches hint (it can be a stale client-side
guess) and must never destroy a complete cached file; `robust_download`
reconciles it against the true remote size, staging every fetch into a sibling
`.part` so the stale bytes at the destination are never appended to.

#### Fixed — the demo-data manifest was excluded from the wheel and sdist

It was written inside `demos/data/`, which packaging excludes wholesale (~450 MB
of Git LFS payload), so `load_manifest()` raised `FileNotFoundError` for every
pip-installed user. Moved to `demos/data_manifest.json`, with a regression test
that re-runs hatchling's own matcher over the committed exclude globs.

#### Fixed — `gsplat transform --rotate-*` rotated the wrong center dims on stacked nD data (#722)

The 3x3 rotation was embedded in the **last** three center dims, a convention
that exists nowhere else in the codebase and is backwards: the repo appends
time/channel **last** (`embed_dimension`, `combine_as_new_dimension`,
`merge --as-dimension`, the batch-fit merge) and treats the **first** three
dims as spatial (the partitioner's `positions[:, :3]`). On a 4D `(x, y, z, t)`
dataset `--rotate-x 90` therefore mixed z with t — collapsing all timepoints
onto one value — and, because covariance transforms as A·Σ·Aᵀ, gave the
zero-variance time axis a spatial sigma, silently breaking the
degenerate-axis auto-detection the scale/eccentricity/isolation filters
depend on.

- The rotation now acts on the **first three** center dims by default
  (bit-identical for 3D data).
- New `--spatial-dims i,j,k` option selects which three center dims the
  `--rotate-*` matrix acts on — the escape hatch for a **direct nD fit**
  (e.g. a TZYX volume fitted as-is, where the spatial dims are `1,2,3`).
  Exactly 3 distinct in-range integer indices; passing it without a
  `--rotate-*` flag is an error.
- Rotating >3D data without `--spatial-dims` now warns, naming which dims
  are rotated and which are left alone — sharper when more than three axes
  carry real extent (the direct-nD-fit signature), but never an error, since
  a legitimate stacked dataset may carry a continuous stacked axis.

#### Fixed — overlay HTML sanitizer: unsanitized nested subtrees + obfuscated `javascript:` URLs (#720)

`OverlayManager.sanitizeHtml` _unwraps_ a tag outside its allowlist — lifting
the children into the parent — rather than dropping it. Two bugs followed.

- **Descendants of a disallowed tag were never sanitized.** The walk
  `continue`d past such an element without scrubbing it or descending into it,
  then lifted its subtree into the output verbatim, so the handlers on
  `<x><img src=x onerror=…></x>` and `<form><p onclick=…>` survived, as did a
  `javascript:` href under any unknown wrapper. Reachable through
  `scene.add_html(...)`: Python's handler regex wants whitespace before the
  attribute name, so the HTML-legal `<img/onerror=…>` separator walks past it.
  And reachable without any evasion through `?src=<url>`, since a hand-crafted
  zarr never meets the Python compiler at all — on that path this function is
  the only control. Now two flat passes: scrub every element's attributes
  regardless of its tag, then unwrap the disallowed ones.
- **`javascript:` URLs survived trivial obfuscation.** The scheme test used
  `value.trim()`, so it missed both interior tab/LF/CR (which the URL parser
  strips from anywhere in a URL) and leading C0 controls (which it also
  strips): `javascript&Tab;:`, `java&NewLine;script:` and `&#1;javascript:`
  all reached the browser and fired. Reachable through `scene.add_html(...)`
  too — `sanitize_html` matches the literal `javascript:` and never
  HTML-decodes entities, so
  `scene.add_html('<a href="javascript&Tab;:alert(1)">x</a>')` produced live
  XSS in the viewer. ASCII whitespace and C0 controls are now stripped before
  the scheme comparison. Still uncovered: `vbscript:`, `data:` and
  `style: url(...)`. (`xlink:href` is inert only because SVG tags are not
  allowlisted.)

Design note: the unwrap pass runs outermost-first, so each node moves exactly
once; the naive bottom-up ordering re-lifts the same payload once per enclosing
wrapper.

**Rendering change:** `ALLOWED_TAGS` and the attribute policy are unchanged,
but a tag outside the allowlist nested inside another such tag no longer
survives verbatim. `<section><hr></section>` rendered `<hr>` and now renders
nothing; `<figure><figcaption>Cap</figcaption><img>` collapses to `Cap<img>`.
The old walk stopped at the first disallowed tag on each path, so which
wrappers survived depended on nesting depth; the new behaviour is uniform.

#### Changed — three-geometry material-surface symmetry (deep-campaign flag closure)

- **GSplat materials render single-pass** (`forceSinglePass: true`, both
  backends): splat quads are screen-space billboards, and THREE's
  transparent+DoubleSide guard was rendering a redundant back-face pass
  per splat layer — measured live at ~2× the rasterized triangles.
  Line materials already did this; points avoid the guard via FrontSide.
  All three invariants are now unit-pinned.
- **Material surface completion**: `getSplatTexture()` (gsplat),
  `getAbsorption()` (point + line), and the `hasElementAlpha`
  constructor-config field (point + line) — every wrapper now exposes the
  same volumetric surface; clones round-trip the flag via config on all
  three geometry types.
- **Point uniform names u-prefixed**: `opacity`/`invGamma` →
  `uOpacity`/`uInvGamma` (the last cross-geometry naming drift; no
  fallback aliases).
- **`LUXAR_NO_GOG` fast path extended to point + gsplat** (was
  line-only): the identity gain/offset chain is skipped on all three
  geometry types at the default intensity=1/offset=0; `isNoGOG` moved to
  `materials/_shared/uniform-helpers`.
- **GSplat TSL stamps `LUXAR_VOLUMETRIC` as an inert introspection
  tracker** (rebuild boundaries unchanged) so define introspection is
  uniform across backends and geometry types.
- **`stampGSplatPresenceFlags`**: gsplat `hasElementAlpha` stamping
  consolidated beside the texel writer (all four write paths) and pushed
  by `syncGSplatMaterialWithGeometry` — the same chokepoint decomposition
  as points/lines; the commit's duck-typed direct call is gone.
- Ctor-default drift aligned (`uNearCull` 0.1 everywhere, incl. the line
  picking materials); stale docstrings fixed (`enums.py`,
  `render-order.ts`, `alpha.py`); ~30 sibling-symmetry tests added
  (convergence loops, gsplat vAlpha-fold pin, volumetric ctor-survival,
  clone/texture/override coverage, sync-gsplat suite).

#### Changed — volumetric flag closure (double-check follow-ups)

- **Strict color-layout guard** (`assertColorLayout`, all three geometry
  types): the per-geometry data processors / projection entry now require
  `colors.length === count × colorComponents` exactly. An RGBA array whose
  producer forgot to declare `colorComponents: 4` used to satisfy the old
  `≥ count·3` minimum checks and silently mis-stride every element; it now
  throws immediately, naming the mismatch.
- **`stampPointPresenceFlags`**: the two duplicated five-flag inline
  presence-stamp blocks (points node factory + pool adapter) extracted
  into one chokepoint beside the texel writer, adopting the lines
  pattern (`stampLinePresenceFlags`) — plus stamp/sync unit pins that a
  severed `hasElementAlpha` chain previously survived.
- **Combined `USE_COLORMAP` + `LUXAR_VOLUMETRIC` coverage** (points +
  lines): new parity-harness entries, codegen snapshots, and pixel-parity
  tests for the colormap-with-volumetric combination (both branches read
  the same single texel fetch). Also: a tripwire pinning that THREE's TSL
  `material.opacity` tail stays inert (Luxar opacity rides the custom
  uniform), and spec notes for the two deliberate per-element-alpha
  semantics (invisible-but-pickable; `normal`-mode depthWrite keys on
  node opacity only).

#### Added — volumetric blending for Lines (Phase 4) + lines RGBA colors

- **Lines now render the real `volumetric` emission–absorption math**
  (VOLUMETRIC_BLENDING_SPEC.md Phase 4 — the plan is complete: all three
  geometry types) on both shader backends: the fragment computes the
  TRANSVERSE chord through the Gaussian-profile ribbon —
  `rayMass = perpFalloff · width · √(π/ln 100)` (`LINE_CHORD_SCALE`,
  derivation in `rendering/materials/line/math.ts`; the value equals
  `POINT_CHORD_SCALE`, keeping the point/line/gsplat κ scales aligned) —
  with `τ = κ · density · rayMass` (density = the remaining intensity
  chain: cap factor, AA coverage, sub-pixel energy, width-clamp fade,
  near fade, node opacity — the profile enters once, via `rayMass`),
  self-screened emission `S(τ)`, and the
  physical absorption alpha `1 − e^(−τ)` over the premultiplied
  One/OneMinusSrcAlpha state. κ = 0 renders exactly like `additive`.
  Both line materials gained `absorption` config, `uAbsorption` +
  `updateAbsorption`, and `uHasElementAlpha` + `updateHasElementAlpha`
  (all clone-carried); the layers-panel κ slider now appears for
  volumetric lines layers too.
- **Lines accept RGBA colors** (`(N, 4)`; the alpha column is
  per-vertex opacity in `[0, 1]`), mirroring points and gsplats: the
  Python writer validates `channels=(3, 4)`, the loader threads
  `colorComponents`, and the worker de-interleaves RGBA into RGB plus an
  alpha column interpolated through the existing
  `interpolate_scalars_batch` scalar kernel — no WASM change. The
  per-endpoint alphas land in line-texture texel5.zw (the slots the
  texture migration reserved), are read through `sanitizeAlpha`, and mix
  along the segment parameter `t`; alpha scales a segment's contribution
  linearly in every blending mode and maps into optical depth
  `w(a) = −ln(1 − a)` under `volumetric` (gated by `uHasElementAlpha`,
  so RGB datasets are unaffected).
- **`effectiveGeometryMode` removed**: with lines implementing the real
  math, the interim geometry-mode downgrade helper became identity and
  was deleted from `rendering/blending-state.ts` — order dependence is
  judged directly on `needsDepthSort(mode)`, so lines `volumetric` now
  depth-sorts back-to-front through the existing lazy segment-midpoint
  provider with zero coordinator change.

#### Added — volumetric blending for Points (Phase 3) + points RGBA colors

- **Points now render the real `volumetric` emission–absorption math**
  (VOLUMETRIC_BLENDING_SPEC.md Phase 3) on both shader backends: the
  fragment computes the isotropic special case of the gsplat ray
  integral — `rayMass = falloff · radius · √(π/ln 100)`, the line
  integral through the Gaussian-profile point ball — with
  `τ = κ · density · rayMass`, self-screened emission `S(τ)`, and the
  physical absorption alpha `1 − e^(−τ)` over the premultiplied
  One/OneMinusSrcAlpha state. κ = 0 renders pixel-identical to
  `additive`; the depth sort engages automatically through the existing
  `needsDepthSort(effectiveGeometryMode(...))` gates (the lines-only
  additive fallback remains until Phase 4). The layers-panel κ slider
  now appears for volumetric points layers, and both point materials
  gained `uAbsorption` (composed node `absorption`) and
  `updateAbsorption`.
- **Points accept RGBA colors** (`(N, 4)`; the alpha column is per-point
  opacity in `[0, 1]`), mirroring gsplats: alpha rides texel2.y of the
  point texture, scales a point's contribution linearly in every
  blending mode, and maps into optical depth `w(a) = −ln(1 − a)` under
  `volumetric` (gated by `uHasElementAlpha`, so RGB datasets are
  unaffected). The whole pipeline — Python validation
  (`channels=(3, 4)`), accumulator, progressive-LOD concat (with a
  fail-fast mixed RGB/RGBA ladder guard), nD projection compaction, GPU
  adapters, and the texel writer — is stride-aware.
- **Mandelbulb demo showcases volumetric points**: full-strength colors
  with real depth cueing — the historical `colors *= 0.1` +
  `intensity=0.0625` anti-blowout dimming under additive (a combined
  ×0.00625) is gone; the demo now runs κ = 8 at intensity 0.5 (80× the
  old brightness), tuned live at the demo's full 1.24M-point resolution
  so the dense core sits just under saturation.

#### Added — lines texture storage + depth sorting (three-geometry symmetry complete)

- **Lines migrated to texture-backed element storage**: per-segment data
  now lives in an RGBA32F line texture (6 texels/segment — endpoints +
  widths, per-endpoint colors + sharpness, segment length + the two
  per-endpoint cap-suppression scalars,
  colormap scalars + reserved per-endpoint alphas for volumetric
  Phase 4), fetched in the vertex stage via `texelFetch` and indexed by
  the sole per-instance `aSortedIndex` attribute — the same storage
  model gsplats and points already use.
- **Lines in `normal` blending mode are now back-to-front depth-sorted**
  by segment midpoint, joining the SortWorker, the camera-motion re-sort
  scheduler, the cross-node `renderOrder` scale, the `preserveOrdering`
  same-count-recommit prior, and the suffix-only append fast path.
  Picking reads the storage slot (`aSortedIndex`), staying correct under
  any permutation. Lines `volumetric` keeps its additive fallback
  (unsorted) until volumetric Phase 4 flips the shared policy helper.
- **Three lines-only mechanisms retired by the fixed texel layout**: the
  interleaved-attribute packing, the colormap attribute-set toggle
  (scalar presence now rides the `userData.hasScalars` stamp — no more
  geometry rebuild or pool re-bucketing on a colormap switch), and the
  line-material LRU cache (line materials are per node, carrying the
  node's own `uLineTex`, like points and gsplats).

#### Added — points depth sorting (normal mode now correctly ordered)

- **Points in `normal` blending mode are now back-to-front depth-sorted**,
  exactly like gsplats: every points commit registers its projected 3D
  centers with the persistent SortWorker (via a lazy provider so the
  common additive path never pays the copy), the resulting permutation is
  applied through the existing `aSortedIndex` indirection, and points
  join the camera-motion re-sort scheduler and the cross-node global
  `renderOrder` scale. Same-count recommits (timepoint scrubs) keep the
  previous permutation as a no-worse prior until the re-sort lands
  (`preserveOrdering`, mirroring gsplats).
- The depth-sort coordinator went geometry-neutral: `noteGSplatsCommit` /
  `noteGSplatsBlendingModeSwitch` → `noteDepthSortCommit` /
  `noteDepthSortBlendingModeSwitch`; order-dependence is judged on the
  EFFECTIVE mode (`effectiveGeometryMode`), so points `volumetric` —
  rendered as additive until volumetric phase 3 — is deliberately not
  sorted, and phase 3's policy flip upgrades it automatically. LayersPanel
  mode switches, per-mesh disposal, and lazy-LOD demotion all
  register/release points sort state symmetrically with gsplats.
- Docs: Points `radii/` and Lines `widths/` shader contract documented in
  LUXAR_ZARR_FORMAT.md — size attributes do not scale with the node
  `transform` (centers/vertices do); gsplats differ (covariances
  transform with the node).

#### Added — `luxar demo` sub-app + DEMO_META registry + CLI dedup (#633–#638)

- **`luxar demo` sub-app** (#637): `luxar demo` lists all 75 bundled demos in a
  table; `demo info <key>`, `demo run <key|index>` (forwards `--` args, exit
  codes propagate), `demo run-all` (batch `--no-serve` generation with
  `--skip-existing/--force`, `--keep-going/--fail-fast`), and
  `demo cache list/clear` for the `~/.cache/luxar/` inventory.
- **DEMO_META registry** (#635): every `demo_*.py` carries a machine-readable
  `DEMO_META` literal, AST-parsed (never imported) by `demos/registry.py`;
  schema-validated for all demos by `tests/test_demo_meta.py`, single source
  for the CLI and the gallery manifest.
- **CLI refactor** (#638): shared option definitions in `common_options.py`
  (no duplicated `typer.Option` help/defaults), honest port API (`pick_port` /
  `find_available_port` return the actually-bound port), polled server
  readiness (`wait_for_server`, replaces fixed sleeps, fail-fast on a dead
  server thread), and one `ensure_viewer_built()` policy for every
  serve-family command.
- **Fixed** (#633): subcommands no longer swallow `typer.Exit`; zarr stores
  are mounted at the data-server root.

#### Fixed — review-campaign hardening riding the points-sorting PR

- **Runtime blending-mode switches now reach hidden LOD levels**: switching
  a layer TO an order-dependent mode also marks resident lazy LOD levels
  stale, so the LOD registry's settle-gated reload re-commits (and
  depth-sort registers) them — previously they rendered the sorted mode
  unsorted until an unrelated slice change (pre-existing, gsplats + points).
- **High-gain gsplats no longer lose dim splats**: the early fragment
  discard is gain-aware (`intensity × max(gain, 1)`), so dim fluorescence
  channels amplified with the intensity control keep their splats instead
  of showing hard clipped rims (identical output at gain ≤ 1).
- **Sort worker robustness**: a worker that dies during startup or crashes
  mid-session can no longer accumulate per-commit memory (init settle
  guard) or permanently stop a node's re-sorts (per-sort RPC deadline) —
  both degrade to the documented unsorted-normal fallback.
- Dataset-switch teardown no longer logs spurious node-failure errors when
  an in-flight load crosses the dispose (expected-abort classification).

#### Fixed — scale-correctness + LOD display + compiler validation campaign

- **Tiny/huge-unit scenes now render correctly in every geometry**: all
  absolute view/world-space epsilon floors (nearCull/invDistance/clip-w/
  cap-ramp/gsplat coverage-fade + determinant) replaced with scale-free
  guards; the gsplat sum-mode ray integral inverts a trace-normalized
  covariance (float32 determinant underflow at tiny scales); orbit
  re-init/reset distances are scene-relative. Verified on real GPU at
  ×1e-6 and ×1e6 with exact coverage parity for points/lines/gsplats.
- **LOD display bugs** (1.15M-frame property harness): a fresh-but-empty
  level can no longer blank a group by redirecting to a not-ready
  placeholder; the slice-aware fallback no longer shows a stale slice
  from a ready-but-stale branch; byte-budget eviction never releases the
  on-screen cross-fade partner; disabling fade flags mid-fade restores
  authored opacity.
- **Python writer validation** (440-case fuzz, 7 root causes): empty and
  zarr-reserved node names rejected everywhere (an empty name previously
  made the store unloadable by stamping the root); array lengths
  validated before spatial reorder (no silent truncation / raw
  IndexError); negative line indices and non-string labels rejected
  cleanly; render attrs, attr collisions, duplicate names, and transform
  prep validate BEFORE any zarr writes; scalar-broadcast validation
  matches array validation.
- Pool/TSL hardening: evictors splice-before-dispose; allocation
  counters can no longer desync on a throwing sweep; TSL materials
  persist explicit depthTest/transparent overrides across graph rebuilds.

#### Fixed — robustness campaign: pool throw-paths, picking lifecycle, lines catch-up (post-#630/#632)

- **Pool adapters**: a throwing allocation during a grow no longer strands a
  mesh on a free-pooled/disposed geometry — the released buffer is re-claimed
  on failure (all three geometry types); lines multi-attribute writes are now
  all-or-nothing (pre-flight length sweep).
- **Picking**: hidden/demoted LOD levels no longer render into the pick
  buffer (or resurrect released pool geometries); the pick material gets the
  RenderObject soft-dispose after pool swaps (WebGPU stale-buffer class); a
  failed dataset switch disposes the previous picking session up-front; an
  in-flight pick readback that resolves after dispose is dropped.
- **Lines**: materials now build from COMPOSED effective attrs (ancestor
  opacity/intensity contributions were silently dropped until a panel
  interaction); the colormap clone no longer detaches the cached original
  from camera updates (`detachFromGlobalUpdates` deleted — its founding
  rationale never held).
- **Commit pipeline**: one throwing geometry commit no longer starves the
  pass's sibling commits; errors aggregate and re-surface at the same call
  site.
- **Python**: the points writer stamps `has_colors`/`has_radii`/
  `has_sharpness` into node attrs like the lines/gsplat writers.
- Verified by fuzz/property/state-machine passes, a production build, the
  WebGPU-path E2E battery, and GPU-resource stability probes.

#### Changed — Points migrate to texture-backed storage + `aSortedIndex` (depth-sorting §8, PR-A of the points/lines→volumetric arc)

- **Points now render like gsplats**: per-point data lives in an RGBA32F
  element texture (fixed 3 texels/point — center+radius / rgb+sharpness /
  scalar+alpha-reserved) fetched in the vertex stage through the
  `aSortedIndex` indirection, decoupling draw order from storage order so
  points can be depth-sorted (next PR) without rewriting point data.
  Behavior-preserving: identity ordering, pixel-identical visual baselines,
  full parity suite green on both backends.
- The storage layer generalized into `element-texture-layout.ts`
  (parameterized layouts; gsplat width math unchanged) and
  `element-storage.ts` (attach, ranged texel uploads, sorted-index
  writers) shared by gsplats and points (lines follow).
- Point materials are now **per-node** (each carries its node's
  `uPointTex`); the colormap clone-on-divergence dance is gone; pick
  element ids read `aSortedIndex` (storage indices — permutation-stable).
- Pool simplification: dtype/scalar bucketing deleted — capacity is the
  only points pool match criterion; the append fast path (suffix-only
  texel writes) and context-restore full-dirty recovery carry over.
- Cost note: ~52 B/point GPU (48 texture + 4 ordering) vs 32–36 B
  interleaved; the reserved alpha slot pre-positions volumetric Phase 3's
  per-point RGBA opacity.

#### Fixed — RGBA per-element opacity: double-check follow-up (post-#618/#620)

- **Rust↔TypeScript parity test for `color_components = 4`.** The gsplat
  projection's RGBA color compaction (`copy_from_slice` in Rust, the channel
  loop in the TS twin) had no cross-language parity test — the RGB-only golden
  cases never exercised the 4th (alpha) channel, the exact drift the 1:1-parity
  rule guards. Added an RGBA case to both the Rust `test_fused_matches_multicall`
  sibling and the `wasm-vs-typescript` harness (alpha set distinct from RGB so a
  stride/drop bug misaligns the output); both pass against the shipped kernel.
- **Integer-dtype guard in `_merge_lod_colors`.** The additive-LOD color merge
  now works in float32 and normalizes any integer part by its full-scale before
  concatenating, so a mixed uint8-RGB + float-RGBA merge can no longer promote a
  widened `alpha = 255` into an out-of-`[0, 1]` opacity (latent; no live
  producer feeds integer RGBA today).
- **`_merge_lod_colors` no longer force-normalizes a uniform integer merge**
  (regression fix for the over-reaching float32 pin in the bullet above). A
  uniform-dtype integer merge now PRESERVES its native dtype (full-scale =
  opaque) — only a white-fill, a dtype mismatch, or a float part promotes the
  result to float32 `[0, 1]`. The float32 pin had diverged a multi-sub-LOD
  uint8 dataset (which normalized to float `[0, 1]`) from the single-sub-LOD
  path (`self.colors = lod0.colors`, which keeps uint8), silently changing the
  stored color encoding; both now agree.
- **Validator contract alignment.** `validation/types.py::validate_colors` now
  applies the alpha `[0, 1]` bound to floating dtypes only (integer storage is
  SDR in its native range), matching `validate_colors_for_writing` — the two
  previously contradicted each other on a uint8 `alpha = 255` array.
- **Docs.** Softened the INRIA-export "losslessly / bit-faithfully" wording to
  "verbatim (no rescale; float-precise in opacity, not bit-exact)" — the PLY
  stores logits; fixed the `wasm/types.ts` fused-kernel JSDoc (`* 3` → `*
colorComponents`, added the missing `@param`) and a stale `colors[i*3]`
  comment.
- **Non-finite opacity hardening (interop).** A corrupt classical source (e.g.
  a malformed INRIA float opacity field → `sigmoid(NaN)=NaN`) could ride a
  non-finite value into the color alpha channel and poison the alpha-aware
  `effective_amplitudes` ranking (LOD/cull) and the INRIA re-export logit —
  neither of which passes the write-time finiteness validator. All five import
  dialects funnel through `classical_to_gsplat_data`, which now maps non-finite
  opacity to a finite alpha (NaN/+inf → opaque 1.0, −inf → 0.0) before the
  `[0, 1]` clip.
- **RGB-column finiteness in `validate_colors` (types.py).** The lightweight
  type-guard checked finiteness only on the alpha column, letting NaN/Inf RGB
  slip through (the write validator already rejected them). It now rejects
  NaN/Inf in any channel while still accepting arbitrarily large finite HDR
  emission.

#### Changed — one shared `viewStatesEqual` for the progressive loaders

- The points/lines/gsplats progressive loaders' three byte-identical local
  `viewStatesEqual` copies (their `*ViewState` types are all aliases of the
  same `ViewState`) are consolidated into
  `data/loaders/progressive/view-state-equal.ts`. The equality is the
  linchpin of the memoized-noop commit skip AND the append fast path's
  generation reset, so the triplication was a drift hazard: a new
  query-affecting field added to only two copies would silently serve stale
  data for the third geometry. The consolidation itself is behavior-
  preserving (200k-trial old-vs-new fuzz equivalence, zero mismatches);
  direct unit tests added for the comparison semantics.
- **Fixed (pre-existing): the startup dimensions-metadata refresh reset every
  progressive loader once per dataset load.** The scene rebuilds the
  dimensions metadata right after the first data load (drops the
  `range: null` key, derives `step: null → 1` on displayed dims, reorders
  object keys); the old raw-JSON dimensions compare flagged that as a view
  change, bumping every loader's reset generation — discarding the ladder
  prefix (re-streamed from warm cache) and the append-fast-path lineage for
  a query-identical view. The dimensions compare is now a canonical
  QUERY-DETERMINANT projection mirroring the slice-cache key
  (`buildSliceViewSig`): name everywhere (extend_to_all matching) plus
  discrete/spatial/step/cyclic on non-displayed dims; display/navigation
  metadata (`range`, displayed-dim `step`, `unit`, `display`,
  `description`) and object key order no longer matter. Verified in-browser:
  the mid-load churn reset is gone on 3D and 4D datasets (the one remaining
  4D reset is a genuine tolerance change and must reset).

#### Added — per-element opacity via RGBA colors, volumetric Phase 2 (gsplats)

- **`colors` widens from `(N, 3)` RGB to optionally `(N, 4)` RGBA** for
  GSplats. The alpha column is **per-element opacity α ∈ [0, 1]** — one new
  concept, no new parameter. Every blending mode consumes it the way it
  consumes node opacity: additive/luminous/max/opaque scale contribution by α,
  `normal` gets true per-element alpha compositing, and `volumetric` maps it
  into optical depth `w = −ln(1 − α)` so a splat's peak alpha reproduces α
  (3DGS-faithful). Absent ⇒ α = 1; RGB datasets are unchanged and pay nothing.
- **Classical import now stores learned 3DGS opacity in the alpha channel**
  (amplitudes := 1), so imported photogrammetric scenes render with correct
  per-splat occlusion in `normal`/`volumetric` (dark solid surfaces hide the
  background) while additive stays visually identical. INRIA PLY export reads
  alpha back verbatim — lossless round-trip. Mass-ranked ops (LOD ladders,
  culling, `gsplat info`) use the alpha-effective amplitude `A·α`.
- No `.gsplats.zarr` format-version bump (codecs are channel-agnostic; readers
  key off the array shape). LOD substitutive merges aggregate α in optical-depth
  (`w`) space. Only direct-color splats get per-element opacity; intensity/
  colormap splats fall back to the node dials. Points/lines RGBA + volumetric
  are phases 3–4. See `docs/guides/specs/VOLUMETRIC_BLENDING_SPEC.md` §5.4.1.

#### Performance — Points & Lines append fast path (depth-sorting Phase 4 Stage 2, three-geometry symmetry)

- **Stage 2 extended to Points and Lines:** a progressive-LOD commit that merely
  extends an already-committed prefix now writes & uploads only the new
  `[prevCount, count)` instance suffix for Points and Lines too, via
  `writeInterleavedAttribute(…, { fromInstance })` threaded through
  `writePooledAttribute` and both pool adapters (Lines in segment units). The
  prefix-lineage helper is now geometry-agnostic (`types/gsplats-lineage.ts` →
  `types/prefix-lineage.ts`, shared by all three loaders/commits).
  Order-preservation was verified for both projections (Points' zero-radius drop
  is a pre-concat forward scan; Lines clipping drops/clips in place, never
  splits or reorders). The gates mirror the gsplat one minus
  `preserveOrdering`/`committedTruncate`, plus a new conjunct: optional-field
  presence must match the committed parent (the all-or-nothing
  `concatOptionalField` means a Float32↔absent flip re-fills a column with a
  constant fill the committed prefix need not match). The context-restore
  full-dirty hook now covers points/lines interleaved instance buffers as well.
- **Fixed (pre-existing, shipped with gsplats in #613): the prefix-lineage
  WeakMap retained every intermediate concat of a generation.** A WeakMap
  holds its value strongly while the key is reachable, so the forward chain
  (newest → … → first) pinned ~(n−1)/2 × the final CPU arrays on an n-level
  ladder for as long as the newest concat stayed committed — indefinitely on
  a static view (hundreds of MB on 10M-element datasets). `setPrefixParent`
  now caps the chain at depth 1 (linking a child deletes the superseded
  parent's own entry) and all three commits consume-and-clear the entry
  right after the gate check; an in-flight or retried commit degrades to a
  full rewrite, the safe direction.
- **Fixed (pre-existing): mixed-sharpness Lines ladders popped razor-sharp.**
  `concatenateLinesData` zero-filled sharpness for parts that lack it, while the
  worker projection substitutes the 0.5 default (β=2, Gaussian) for a null
  sharpness array — so the moment a sharpness-carrying level joined the ladder,
  every sharpness-less part flipped from soft default to razor-sharp 0.0. The
  concat now fills 0.5 (mirroring the white color fill), which is also what
  makes the append fast path's prefix-identity contract hold for Lines.

#### Added — `volumetric` blending mode, Phase 1 (gsplats + node-level κ)

- **Sixth blending mode `volumetric`** — emission–absorption compositing
  (Max 1995) per `VOLUMETRIC_BLENDING_SPEC.md`: each gsplat adds its
  ray-integrated, self-screened emission (S(τ) = (1−e^(−τ))/τ) and
  attenuates everything behind it by the physical absorption
  α = 1 − e^(−τ), τ = κ·opacity·rayMass, composited back-to-front on the
  depth-sort infrastructure (new `needsDepthSort` predicate = normal ∪
  volumetric — the first sum-projected _sorted_ mode). κ = 0 renders
  pixel-identical to `additive`; opacity scales density (emission AND τ),
  so layer fades leave no ghost occlusion; never depth-writes (no 0.99
  opacity cliff).
- **New node-level composable attr `absorption` (κ ≥ 0, default 1.0)** —
  follows the opacity path everywhere: Python `Node.absorption` property /
  `set_absorption` / `validate_absorption`, default-stamped by the writers,
  in `COMPOSITING_ATTRS`; viewer multiplicative composition, `uAbsorption`
  uniform (GLSL + TSL), material `updateAbsorption`/clone round-trip;
  layers-panel "Absorption" slider shown only for volumetric gsplat/group layers;
  `luxar gsplat convert --absorption`.
- **Phase-1 scope**: gsplats implement the fragment math
  (`LUXAR_VOLUMETRIC` GLSL define + TSL build-time branch, with the TSL
  rebuild predicate generalized so additive↔volumetric switches rebuild
  the graph); points/lines intercept the mode and render its exact κ = 0
  additive fallback until phases 3–4; picking stays brightness-as-depth.
  Black splats still absorb (the zero-color discard is bypassed when τ is
  significant). Codegen snapshot `gsplat-volumetric` + GLSL/TSL pixel
  parity, split-splat invariant (I2) and S(τ) series/seam unit tests,
  κ=0≡additive and absorption-darkening E2E, volumetric depth-sort E2E on
  a reversed-order fixture, 6-mode fixtures for points/lines.

#### Docs — volumetric blending mode spec (proposed)

- New `docs/guides/specs/VOLUMETRIC_BLENDING_SPEC.md`: design for a 6th
  blending mode `volumetric` — the emission–absorption model of volume
  rendering (Max 1995). Each element adds its ray-integrated emission AND
  exponentially attenuates what's behind it (α = 1 − e^(−τ),
  τ = κ·opacity·ray-mass), composited back-to-front on the existing
  depth-sort infrastructure under the same One/OneMinusSrcAlpha state
  gsplat-normal already uses. A node-level composable `absorption` (κ)
  attr spans additive glow (κ = 0, pixel-identical to `additive`) through
  attenuated projection to a dense self-occluding medium; opacity scales
  density (emission and absorption together), so layer fades leave no
  ghost occlusion. Spec includes the self-screening closed form and its
  split-splat invariant, optional per-splat `absorption_weights` (with a
  3DGS opacity mapping) deferred to phase 2, and gsplats → points → lines
  phasing. Design only — no code change.

#### Performance — partial splat-texture uploads (depth-sorting Phase 4, Stages 1–2)

- **Stage 1 (slack elimination):** GSplat commits no longer re-upload the entire
  capacity-sized RGBA32F splat texture on every frame. `writeSplatTexels`
  registers per-row `updateRanges` covering only the live `[0, count)` rows, so
  the GPU buffer pool's 1.5× growth headroom and best-fit slack rows stop riding
  every commit to the GPU. Measured 33–59% less upload per commit on
  `gsplats_4d_neuromast_2ch` (classic WebGL); pixel-identical (texel content
  unchanged, shaders only read `[0, count)`). Above 75% of rows dirty it falls
  back to a single full-image upload. WebGPU backends re-upload the whole image
  as before (Stage 3 follow-up).
- **Stage 2 (append fast path):** a progressive-LOD commit that merely extends an
  already-committed prefix (the common 4D-timelapse streaming case) now writes &
  uploads only the new `[prevCount, count)` suffix. Because the nD→3D projection
  is per-splat-independent and order-preserving and the loader appends LOD levels
  in order, the projected prefix is byte-identical to what the GPU holds under an
  unchanged view state — so no projection-kernel change was needed. A
  forward-chained prefix-lineage `WeakMap` (now `types/prefix-lineage.ts`) proves
  the extension by identity against the committed data; the append gate
  additionally requires in-place pool reuse, an intact GPU prefix, a strict count
  increase, and an unchanged `uTruncate`. A WebGL context-restore hook re-marks
  all splat buffers full-dirty and disables the fast path for the next commit so
  a restore never leaves a stale prefix. Points/Lines symmetry landed in the
  follow-up above. See `GSPLAT_DEPTH_SORTING_SPEC.md` §7.

#### Fixed — blending-modes correctness campaign (#601, #602, #603, #604)

- A full review of the five blending modes (`normal` / `additive` / `max` /
  `opaque` / `luminous`) across Python → zarr → viewer → shaders fixed two
  silent wrong-render bugs sharing one root cause — `blending_mode` has no
  identity value, unlike opacity/gamma/intensity/offset: (1) the Python
  writers stamped a default `"additive"` on every geometry leaf, which
  shadowed any ancestor-set mode under the viewer's nearest-setter-wins
  composition (leaves now omit the attr when unset and inherit); (2) the
  layers panel initialized a layer's blending mode from the node's raw attr
  while the material renders the composed effective mode, silently
  overriding ancestor-authored modes at panel init (now initialized from
  the same composed value the renderer uses).
- Invalid blending modes now fail BEFORE any zarr group is created (no more
  partial nodes on disk), and unknown mode strings from foreign scenes are
  normalized once at composition (`→ 'normal'`, warn-once) so they render
  AND depth-sort consistently instead of alpha-over-unsorted.
- `opaque` gsplats moved from the emissive sum ray-integral to peak
  projection, completing the #561 surface-vs-emissive taxonomy via a shared
  `usesPeakProjection` predicate (also fixing a stale-TSL-graph trap on
  `additive→opaque` switches). Opaque gsplat scenes render slightly
  dimmer/tighter.
- Coverage: per-mode E2E for all five modes × points and lines, an
  inherited-mode cross-stack fixture, `point-max`/`line-max` TSL
  parity + codegen variants, and panel blending unit tests; the mode set is
  now a single runtime tuple (`types/blending.ts`) from which the TypeScript
  union and per-node attr types derive, the Python validator derives from
  the `BlendingMode` enum, and `opaque` gsplats surface-pick front-most
  (matching what the user sees) instead of brightest-wins.

#### Added — `luxar_delta_v1` delta pre-filter (format v3.3, 12-16% smaller stores)

- Quantized code arrays (coordinates `linear_perchannel_u16`, Cholesky
  `log`/`signed_log_perchannel` halves, `bounded`/`geolog_scalar` amplitudes,
  and colors — SDR `rgb_uint8`, HDR `geolog_perchannel`, integer passthrough)
  can now carry the Luxar-owned zarr v2 filter `luxar_delta_v1`: per-axis
  modular delta + zigzag residuals, column-major within each chunk, under the
  unchanged width-aware Blosc policy. Hilbert ordering makes consecutive codes
  a smooth ramp; the residuals compress 12-16% smaller whole-store on real fits
  (14.9-15.7% measured end-to-end on real h2afva light-sheet fits)
  — lossless, and **probe-gated** at encode time (one representative chunk
  compressed both ways; the filter applies only where it wins, so output is
  never larger than before). `.gsplats.zarr` format v3.2 → v3.3; stores where
  the probe declines everywhere remain byte-identical to v3.2.
- A pure storage transform below the `encoding` layer (origin: the PlayCanvas
  SOG comparison — its size edge was WebP's spatial prediction): `encoding`
  attrs, the WASM/TS decode kernels, and the sub-chunk range-loader are all
  untouched; zarr/zarrita undoes the filter during whole-chunk reconstruction.
  Python codec: `luxar/encoding/_encoders/delta_codec.py` (auto-registered
  via the numcodecs `numcodecs.codecs` entry point whenever luxar is
  installed — vanilla `zarr.open` needs no import); viewer twin:
  `luxar-viewer/src/data/codecs/luxar-delta.ts` (registered as
  `numcodecs.luxar_delta_v1` in the zarr facade, so main thread and workers
  both resolve it). Wire format locked by identical hand-computed byte
  vectors in both languages' unit tests.

#### Fixed — depth-sorting correctness hardening (review campaign)

- A deep review of the depth-sorting + texture-based splat rendering
  subsystem fixed six lifecycle/consistency bugs (#568, #569, #571, #587,
  #596): a node whose first sort raced a null camera recovers instead of
  staying unsorted until the next commit; splat counts are clamped
  consistently across the texture, the sorted ordering, and the sort
  worker (previously a `normal`-mode node above the per-node texture bound
  — ~4.19M splats on a 4096-class GPU — lost an arbitrary subset of splats
  the moment its first sort landed, and the non-pool path could allocate
  an over-tall texture → black node); LOD demotion releases the sort
  worker's transferred centers; non-finite bounds can no longer scramble
  the cross-node draw order; and the cross-node ordering keeps working
  when the sort worker itself cannot be constructed (CSP-blocked script).

#### Fixed — one global back-to-front order across gsplat nodes (`normal` mode)

- Mixed scenes — several partitions, or partitions plus single-leaf gsplat
  nodes — now composite in true back-to-front order (#575): all visible
  `normal`-mode gsplat meshes share ONE sequential `renderOrder` scale
  (wrapper groups by mean view-depth, exact BSP ranks within a wrapper).
  Previously the per-wrapper painter ranks and the raw view-z fallback
  were mutually incomparable, so every single-leaf layer drew before every
  partition tile regardless of actual depth.

#### Fixed — Intensity/Offset controls work on colormapped Points and Lines

- Dragging a layer's Intensity/Offset sliders now affects colormapped
  Points and Lines layers exactly as it always did colormapped GSplats
  (#570): gain/offset apply post-LUT to the mapped color (gamma still
  shapes the scalar value pre-LUT). Defaults are pixel-identical.

#### Changed — front-most-wins picking for depth-sorted (`normal`) gsplats

- Clicking a `normal`-mode gsplat layer now picks the FRONT-MOST splat at
  the cursor — matching the occluding surface the user sees — instead of
  the brightest splat, which could sit behind the visible surface (#572).
  Commutative modes (additive/luminous/max) keep brightest-wins.

#### Changed — timepoint scrubbing keeps the previous sort order (`normal` mode)

- Re-committing a gsplat node at the same splat count (per-timepoint
  navigation) now retains the previous depth-sort permutation instead of
  flashing storage order for a frame until the fresh sort lands (#577).

#### Fixed — exact back-to-front ordering of gsplat partition tiles (classical-3DGS imports)

- Classical Gaussian-splat imports (Mip-NeRF `.splat`, INRIA PLY) built as a
  `kind=partition` (the `tiles` recipe) now composite correctly in `normal`
  (alpha-over) mode. Previously the co-located tile meshes drew in creation
  order, and the per-part centroid `renderOrder` heuristic degenerated when the
  camera was inside the volume — visible seams.
- The spatial partition now persists its **BSP split planes** (a `bsp_tree`
  attr on the `kind=partition` group; see `docs/specs/GSPLATS_ZARR_FORMAT.md`),
  and the viewer traverses them to order the tiles back-to-front **exactly**
  (Fuchs–Kedem–Naylor painter's algorithm) — correct for any camera pose,
  including inside the volume. Partitions without a stored tree (streamed
  grid/content merges) fall back to the centroid heuristic.
- The Mip-NeRF and INRIA garden interop demos return to the `tiles` recipe
  (per-tile frustum culling + streaming ladder), reversing the earlier
  single-leaf workaround now that tiles sort correctly.

#### Fixed — classical-splat imports no longer render washed-out (sRGB → linear)

- Importing a classical Gaussian-splat file (INRIA/`.splat`/SPZ/SuperSplat/SOG)
  now renders with the same colors a reference viewer (SuperSplat/PlayCanvas)
  shows, instead of washing toward white (#599). The DC band's baked color
  (`0.5 + C₀·f_dc`) is **display-referred sRGB**, but Luxar's viewer treats
  per-splat color as linear light and applies the sRGB OETF once at output — so
  importing it untouched double-encoded it. The import boundary now converts
  sRGB → linear (new `gsplats/interop/_color.py`, applied at the single
  `classical_to_gsplat_data` chokepoint all dialects funnel through), and the
  INRIA exporter inverts it (linear → sRGB) so round-trips and reference-viewer
  colors match. The transfer curve is the exact IEC 61966-2-1 piecewise sRGB,
  bit-for-bit the inverse of the viewer's output encode.

#### Fixed — responsive timelapse playback (decode/caching), symmetric across Points/Lines/GSplats

- Playing/scrubbing a dimension (e.g. a 4D gsplat timelapse) is now
  responsive: the visible content updates smoothly instead of stalling
  hundreds of ms per timepoint. The bottleneck was never the texture- or
  sort-based rendering (texture upload ~1 ms; depth sort doesn't run for
  additive blending) — it was per-timepoint decode latency plus a cache
  that missed on revisit.
- **Foreground never blocks on fine levels during playback.** A budgeted
  (playing) foreground pass now commits the restored cached prefix plus a
  LOD-0 first-paint floor and returns immediately, rather than
  synchronously decoding cold LOD levels mid-tick. The new shared
  `data/loaders/progressive/streaming-policy.ts` encodes the three
  disciplines (`playback` / `prefetch` / `refine`) so all three geometry
  loaders stay identical by construction.
- **Background prefetch deepens the cache across loops.** The `t+1`
  `SlicePrefetcher` now persists across ticks (a cold LOD level outlives
  one frame, so it is no longer aborted every foreground tick) and deepens
  each slice's cached ladder toward full. Result: the first loop is
  fast-but-coarse and each subsequent loop is higher-quality, still fast —
  and it skips the wasted concat on shadow passes so background work never
  stalls a frame.
- **SliceCache key canonicalization.** The per-slice cache key is now a
  canonical projection of the query determinants, so the two viewState
  builders (navigation vs. init/reprocess) produce identical keys for the
  same slice. Previously they disagreed on a query-irrelevant discrete-dim
  tolerance (`0` vs `0.5`), `step` (`null` vs `1`), and JSON property order,
  so every timepoint was stored under two keys and revisits missed forever.

#### Fixed — `?lod-finest` registered as a real URL param + app option

- The gallery harness's force-finest LOD override was read directly from
  `location.search` at module scope deep in the scene layer, unregistered
  in `UrlParams` (the last direct `window.location` feature-flag read).
  Now `UrlParams.lodFinest` → `LuxarAppOptions.lodFinest` →
  `LODGroupRegistryDeps.getForceFinestLOD` (read live, like the sibling
  cross-fade/energy-comp deps); embedders can force capture-quality LOD
  programmatically. Behavior under `?lod-finest` URLs is unchanged.

#### Fixed — depth-sort teardown sweep + embedder control of feature flags

- Dataset switches now drop ALL depth-sort registrations wholesale
  (`releaseAllDepthSortNodes` wired into the scene teardown) instead of
  relying solely on the per-mesh scene walk — registrations whose mesh
  was never attached no longer outlive their dataset.
- `lodFade`, `lodEnergyComp`, and `depthSort` are now real
  `LuxarAppOptions` threaded from the bootstrap's `urlParams`: an
  embedder-supplied `urlParams` object (or direct option) controls them,
  and the init pipeline no longer reads `window.location` itself.

#### Fixed — view-coherent Lines/Points substitutive LOD + additive-LOD custom-colormap crash (PR #549)

- `substitutive_lod=` coarse levels on Lines (and Points) no longer pop
  haphazardly in brightness/hue between LOD levels: the merge's elongated
  representatives (aspect up to ~25× by level 3) flared view-dependently and
  ACES re-hued the over-brightness. Coarse splats are now anisotropy-capped
  (`max_aspect`, default 3, mass-preserving, coarsened dims only) and merged
  with per-bin mass-preserving amplitudes (`amplitude="mass"`, lift path only)
  so per-channel colored light is conserved exactly per bin. Fitted-gsplat
  pipelines keep the L²-optimal default.
- `additive_lod=` + `colormap=<ndarray LUT>` no longer crashes with
  "ndarray is not JSON serializable" (Lines and Points): the multi-LOD writers
  now resolve custom colormaps to a `colormap_lut` dataset + `colormap='custom'`
  like the flat writers, and the returned node objects mirror the substitution.

#### Added — GSplat depth sorting: correct `normal`-mode transparency (PRs #511, #535, #540, #553)

- Gaussian splats with `blending_mode="normal"` now composite in true
  back-to-front order. Phase 0 (PR #511) fixed the premultiplied coverage
  alpha; Phase 1 (PR #535) moved per-splat data into an RGBA32F splat
  texture indexed by an `aSortedIndex` ordering attribute; Phase 2
  (PR #540) added the WASM depth-sort kernel (`sort_splats_by_depth`,
  scale-invariant normalized-key counting sort with an exact-parity
  TypeScript twin), a persistent SortWorker, and commit-time wiring with a
  per-node generation guard. Ordering refreshes on every data commit and
  on blending-mode switches. Phase 3 completed the feature with live
  camera tracking: a per-frame scheduler re-sorts a node when the
  node-relative view axis rotates past `config.depthSort.angleThresholdDeg`
  (default 3°) or the camera translates along it past
  `config.depthSort.translationFraction` (default 0.05) of the node's
  bounding radius; `?depthSort=0` pins the identity ordering for
  deterministic runs, and sort round-trips surface as the data-loading
  monitor's 'Depth Sort' timing line
  (`docs/guides/specs/GSPLAT_DEPTH_SORTING_SPEC.md`).

#### Fixed — LUT-on-COORDINATE line-vertex corruption + remaining doc-audit flags (PR #517)

- Grid-snapped line vertices (few unique coordinate values) could store as
  `lut_uint8` indices while the viewer's lines spatial-index loader reads
  `vertices` as raw chunked zarr — corrupted geometry. New `allow_lut` knob on
  `ArrayEncoder.encode`; line vertices now always materialise
  (`linear_perchannel_u16`), with a regression test.
- `format_gsplats_info` prints the real ordering metadata (`bits_per_dim=21`)
  instead of `resolution=unknown`; a vacuous CLI port-conflict test (passing a
  nonexistent `--no-viewer` flag) now genuinely exercises the conflict path;
  assorted stale docstrings/READMEs corrected (worker responsibilities, GSplats
  sharpness, cache tiers, per-part recipe names).

#### Fixed — Python completeness-audit sweep (PR #515)

- Deleted dead/unreachable surfaces, synced stale docs, and shared the widths
  validator across geometry writers (net −32 lines; behavior-preserving except
  where noted in the PR).

#### Documentation — full documentation-sync audit (PRs #513, #516)

- Audited every documentation surface against the code (CLI help, format
  writers/readers, viewer source, Sphinx targets, executed tutorial snippets)
  and fixed ~185 findings: stale LOD recipe names, wrong compression/encoding
  claims in `LUXAR_ZARR_FORMAT.md`, a new Lines on-disk format section, broken
  tutorial snippets, a fictional CI section in the E2E guide, missing skills
  coverage for `--floor`/`reencode`/`annotate-quality`/`additive`, and Sphinx
  autodoc for previously undocumented public modules.

#### Changed — gsplat demo LFS baselines upgraded to format v3.2 + AUTO quantization; cells3d demo resurrected (PR #505)

- All 19 precomputed `.gsplats.zarr` demo baselines shipped via Git LFS were
  rewritten from format v3.0 (float32) to v3.2 with AUTO (uint16 Cholesky)
  quantization — a lossless write-time `reencode` (no refit), shrinking the
  bundled demo data ~2.7× (573 MB → 212 MB).
- `demo_gsplats_3d_cells3d_multichannel.py` is resurrected as a BOP-LUT
  layers demo (3D multi-channel cells3d).

#### Fixed — script tidying: `reencode_gsplat_demos.py` and `calibrate_gsplat_demos.py` (PRs #506, #508)

- `reencode_gsplat_demos.py`: docstring, mypy-clean, robustness improvements.
- `calibrate_gsplat_demos.py`: cells3d calibration parity fix; mypy-clean.

#### Added — new spectacular-science gsplat + Lines demos (PRs #496, #497, #500, #502)

- **Cryo-EM giant virus capsid** (`demo_gsplats_3d_cryoem_virus.py`): Gaussian
  splats a real EMDB density map (EMD-5384, PBCV-1) — the microscopy splat
  pipeline applied to structural biology.
- **Visible Human head** (`demo_gsplats_3d_visible_human_head.py`): true-color
  anatomy from NLM cryosection photographs; luminance fit + per-splat RGB
  sampled from the original color volume.
- **3D interstellar dust** (`demo_gsplats_3d_milky_way_dust.py`): the Leike &
  Enßlin (2020) solar-neighborhood dust cube fit to Gaussian splats.
- **4D two-channel neuromast timelapse** (`demo_gsplats_4d_neuromast_2ch.py`):
  membranes + nuclei as layers.
- **Single-cell 3D genome (Dip-C)** (`demo_dipc_3d_genome.py`): chromosomes as
  3D Lines, with a non-displayed `haplotype` dimension to scrub maternal /
  paternal copies.

#### Fixed — viewer works over plain HTTP (PR #501)

- The viewer crashed on any plain-HTTP / non-localhost origin because
  `crypto.subtle` is undefined outside a secure context (used for content-hash
  cache keys). A vendored SHA-256 fallback is now used when `crypto.subtle` is
  unavailable, so the viewer loads over LAN / Tailscale HTTP.

#### Fixed — seven viewer rendering-engine bugs from a WebGL-path deep read (PRs #503, #507)

A full read of the viewer's Three.js WebGL rendering path surfaced seven
confirmed bugs; each fix shipped with a failing-first regression test.

- **Non-pool geometry commit was broken (all three geometry types)**. With
  `useGPUBufferPool: false`, the points same-count commit cast a plain
  `InstancedBufferAttribute` to an interleaved view and threw
  `Cannot read properties of undefined (reading 'stride')` on the _second_
  same-count commit (routine while scrubbing a constant-count dimension); the
  same branch also normalized `Uint16` colors with ÷255 instead of ÷65535
  (~257× too bright). Points now dispose+recreate via `createPointsGeometry`
  (the single owner of the dtype/bounds logic). `updateInstanced{Lines,GSplats}Mesh`
  now return whether they rebuilt the buffer so all three non-pool commit paths
  evict Three's cached `RenderObject` on rebuild (WebGPU stale-`vertexBuffers`
  parity with the pool path).
- **`nearCull` dropped for newly created materials**. The three `getXMaterial`
  factories omitted the stored `currentNearCull`, so a gsplat/line material
  created after camera setup kept its constructor default (0.1 / 0.05 world
  units) until the next resize/FOV event — geometry near the camera was
  wrongly faded/culled on first paint ("splats missing until the camera
  moves") on scenes whose world scale differs from those defaults.
- **LRU material eviction disposed materials still attached to live meshes**.
  Cache churn could dispose a shared material still on a mesh; Three
  auto-recompiled it (so it kept rendering) but its dispose listener had
  unregistered it, so it permanently stopped receiving `updateCameraParams`
  and rendered with stale resolution/FOV/nearCull after the next resize.
  Eviction is now defer-dispose: the entry leaves the cache but stays
  registered for camera updates and is disposed at teardown. (`dispose()` also
  now covers the `registeredMaterials ∪ ownedMaterials` union so a
  colormap-clone original that was detached then evicted is still freed.)
- **GSplat picking was displaced on wide / HiDPI displays**. `computePickBufferSize`
  clamped each axis independently to 1024 px, so any drawing buffer wider than
  2048 px (Retina fullscreen, 4K, ultrawide) produced a pick buffer whose
  aspect no longer matched the camera. The gsplat pick shader assumes square
  pixels (`uFx == uFy`), so hover/selection landed up to ~1.5–1.8× off
  horizontally away from screen center (points/lines pick through the
  aspect-aware projection matrix and were unaffected). The cap is now a single
  uniform scale on both axes, preserving aspect.
- **Progressive-refinement / retry commits never woke the render loop**.
  Refinement passes, failed-load retries, and the online auto-retry commit
  geometry _after_ the sweep that started them; with the rAF loop idle-paused
  (2 s), LOD chunks were fetched, decoded, and uploaded invisibly until the
  next input. A `requestRender` callback now funnels through the three
  SceneLoader commit methods (`SceneLoaderManager.setRequestRender` →
  `AnimationController.startAnimation`, idempotent).
- **The SceneManager `change` event had no subscriber → blank canvas after an
  idle-time context restore**. The WebGL context-restore path ends with
  `triggerChange()` ("trigger a render"), but nothing listened. A context
  restored while the loop was idle-paused (GPU driver reset with the page
  visible but untouched) rebuilt resources and resized the renderer (clearing
  the canvas), then never painted — blank viewer until the next input. The
  init pipeline now wires `change → startAnimation`.
- **Every window resize reallocated the HDR render target twice, unconditionally**.
  `reallocateForSize()` disposed + recreated the full-screen half-float HDR
  target on every call, and every resize reaches it through _two_ paths (the
  window `resize` listener via `ResizeOrchestrator` **and** the canvas-parent
  `ResizeObserver`). The allocation is now memoized on display size, effective
  logical size (SSAA), physical size (DPR), and MSAA sample count — an
  identical request is a complete no-op, while DPR/MSAA/SSAA changes still
  reallocate and `rebuildAfterContextRestore` resets the memo.

#### Removed — dead `computeNDVisibility{Points,Lines,GSplats}` worker kernels

- Deleted the three standalone nD-visibility worker tasks, their TS wrappers
  (`workers/data-worker/visibility/`), TypeScript reference kernels
  (`wasm/typescript/{points,lines,gsplats}.ts`), Rust WASM kernels
  (`wasm/rust/src/{points,lines,gsplats}.rs`), the pooled
  `visibilityMaskBuffer` worker state, and their tests/benchmarks.
- **Why**: they were speculative Phase-2 (2025-12) infrastructure that never
  gained a production caller — two days after they were built, projection
  moved to the worker (`projectXTo3D`) and per-element nD visibility/culling
  was fused INTO projection (Lines `clip_segments_batch` mask, Points
  effective radius, GSplats attenuation), making a standalone visibility
  round-trip redundant. Parity tests and benchmarks kept the dead kernels
  looking alive for six months.
- **Also removed — the dead `querySpatialIndex` worker task and its chain**:
  the sibling Phase-2 orphan. Chunk-AABB spatial queries run on the main
  thread (`SpatialQueryBuilder`) and never used the worker round-trip, so the
  task, its `query_chunks_for_view` WASM kernel (Rust `spatial.rs` + TS
  `spatial.ts`), `validateChunkQueryInputs`, the `'visibility'` `TimeoutKind`,
  and the `workerVisibilityTimeoutMs` config knob are gone (projection/decode
  keep `workerProjectionTimeoutMs`). Archived design docs
  (`docs/archive/implementation-notes/{WASM_ANALYSIS,WORKER_INFRASTRUCTURE_STATUS}.md`)
  now carry a historical-status note.

#### Fixed — Lines are re-culled when scrubbing a non-displayed dimension

- **Symptom**: scrubbing a non-displayed dimension (categorical toggle, time
  slider) only ever _added_ Lines geometry — every visited slot stayed
  rendered (A ∪ B) while Points/GSplats correctly swapped (A xor B). A
  categorical or time dimension was therefore unusable for slicing Lines.
- **Root cause**: the Lines projection dispatcher
  (`workers/data-worker/projection/lines.ts`) hardcoded `numItems = 1` in its
  input validation, rejecting the canonical _empty_ payload the loader
  returns when a node has no data at the current slice
  (`positions array too short (got 0, expected ≥ ndim)`). The throw was
  swallowed as a failed loader update (`staged: null`), so the empty commit
  that clears the previous slot's mesh never ran. The same throw fired on
  the initial load of any Lines node that starts out-of-slice (error log +
  spurious failure/retry bookkeeping).
- **Fix**: validate with `numItems = 0` — the real positions invariant
  ("covers the max vertex referenced by segments") is fully enforced by
  `validateLineSegmentReferences`, and the empty payload now flows through
  the existing zero-visible early-exit to a mesh-clearing commit.
  Regression coverage: dispatcher golden test (TS + WASM backends), worker
  happy-path tests, data-processor staging test, and a new E2E spec
  (`lines-nd-dimension-visibility.spec.ts`) driving a hidden categorical
  scrub over the new `test_lines_categorical.luxar.zarr` fixture with a
  Points pair as the in-frame control.

#### Fixed — `center_at_centroid` / `gsplat convert --center` no longer centers categorical axes

- **Why**: `center_at_centroid()` subtracted the amplitude-weighted centroid from
  **every** dimension, including a non-spatial categorical axis (a per-timepoint
  time axis, a channel axis). On an nD timelapse that pushed the integer
  timepoints (0..T-1) to fractional offsets, so the viewer's slice navigator —
  which steps in integer voxels — landed _between_ timepoints and showed a
  partial/sparse splat set (looked like corruption). `gsplat convert` centers by
  default, so every converted timelapse scene was affected.
- **Fix**: `center_at_centroid()` now shifts only the **non-degenerate (spatial)**
  axes, leaving categorical axes at their coordinates. Pure spatial data (no
  degenerate axis) is centered on every axis exactly as before. The same fix
  covers the partition/nested `gsplat transform --center` graft path (via a new
  node-tree `nondegenerate_axes`). Affects `gsplat convert --center`,
  `gsplat transform --center` (flat + partitioned), and the Python API.
- The spatial-vs-categorical axis rule now lives in one place
  (`gsplats/utils/spatial_axes.py`: `spatial_axes_from_max_sigma`,
  `spatial_only_shift`), shared by the flat and node-tree paths and the
  scale/eccentricity/isolation filters — no duplicated threshold logic.

#### Security & architecture — external review remediation

- **Archive extraction hardened + de-duplicated**: consolidated the two
  drifted `_extract_compressed_zarr` helpers into one safe
  `luxar.gsplats.io._archive.extract_compressed_zarr`. It rejects symlinks,
  hardlinks, devices and FIFOs (the migrate path previously did not — a
  malicious `.tar.gz` could escape the temp dir), validates every member before
  extracting, caps member count / total uncompressed size (archive-bomb guard),
  and removes the temp dir on any failure. The symlink-escape test now also
  covers the `migrate-format` path.
- **Security & layer checks now gate CI**: `bandit` (medium+), `import-linter`
  (domain layers `gsplats`/`io`/`core`/`encoding` must not import `luxar.cli`),
  and a Python/viewer version-consistency check run in `hatch run check`,
  `make check-all`, and CI. Rust `cargo test` and a new Go `go vet`/`build`/
  `test` job also gate PRs (browser E2E stays opt-in).
- **Volume / OME-Zarr loading moved out of the CLI**: `load_volume`, the
  zarr/OME-Zarr discovery helpers, and dimension inference now live in
  `luxar.io.volume`, `luxar.io.ome_zarr`, and `luxar.core.dimension_inference`;
  the domain denoise pipeline no longer imports upward into `luxar.cli`.
  `load_volume` raises a domain `ImportError` (the CLI converts it to a clean
  exit) instead of leaking `typer.Exit` to programmatic callers.
- **`import luxar` is lightweight again**: GSplat re-exports resolve lazily
  (PEP 562), so importing the base package no longer eagerly loads torch/scipy.
  The gsplats optional-import guard now re-raises internal import errors instead
  of masking them as "optional dependency missing".
- **CORS removed from same-origin bundles**: the exported `serve.py` and the
  native Go launcher no longer emit a wildcard `Access-Control-Allow-Origin` —
  the viewer and its data are same-origin, so it only widened exposure. The
  `luxar serve` LAN-exposure warning now correctly fires when binding
  `0.0.0.0` (all interfaces) with wildcard CORS.
- **Docs**: reconciled remaining `v3.1` references to the current `v3.2`
  `.gsplats.zarr` format (writer output, migration targets).

#### Added — GSIP: `gsplat filter` toolbox + `gsplat convert` appearance options

- **Why**: exporting background-suppression experiments from the neuromast
  gsplat timelapse exposed two gaps. `gsplat convert` hardcoded `colormap=gray`
  and wrote no `viewer_config`, so scenes silently rendered gray + ACES
  tone-mapping and couldn't be compared to a plasma/Neutral reference. And
  `gsplat filter` only cut on volume/amplitude/eccentricity/mass/sigma with
  absolute or linear-normalized thresholds — useless on a timelapse, where the
  near-zero time axis makes `eccentricities()` a constant 1.0 and linear
  normalization is meaningless on heavy-tailed splat attributes.
- **`gsplat convert` appearance**: `--colormap` (validated builtin /
  matplotlib / colorcet), `--tone-mapping` (validated, written to scene
  `viewer_config`), `--gamma`, `--intensity`, `--layer/--no-layer`. Pair a
  scientific colormap with `--tone-mapping Neutral` for faithful colors (the
  viewer default, ACES, shifts hues). NOTE: `--layer` now defaults on (the
  gsplats node is listed in the viewer Layers panel).
- **`gsplat filter` (GSIP)**:
    - Percentile value-syntax on any threshold: `pNN` / `NN%` (e.g.
      `--scale-max p90`), robust on heavy-tailed attributes.
    - `--scale-min/max`: characteristic size = geometric-mean **spatial** sigma;
      auto-ignores zero-variance axes (timelapse-safe). `--eccentricity` is now
      spatial-by-default too. `--spatial-dims` overrides the axis auto-detection.
    - `--isolation-max` (nearest-neighbour distance) and
      `--min-neighbors`+`--neighbor-radius` remove spatially-isolated noise
      splats, grouped by the non-spatial axis (timepoints never count as
      neighbours; reuses `BatchedSpatialHashGrid`).
    - `--soft-highpass`/`--soft-lowpass`+`--soft-width`: soft reweighting that
      attenuates amplitude by a smooth function of scale instead of deleting (no
      popping; splat count unchanged).
    - `--dry-run`: report impact (splats / mass / amplitude removed) without
      writing.
    - New `GSplatData` API: `scale()`, axes-aware `eccentricities()`,
      `nearest_neighbor_distances()`, `neighbor_counts()`, `reweight_amplitude()`,
      `soft_scale_filter()`, percentile mode on `_resolve_threshold`, and the new
      `filter_by` criteria (all forwarded per-level for substitutive pyramids).

#### Fixed — L2 OPFS cache: write-probe at init (WKWebView/Safari error storm)

- **Why**: in the native macOS app (WKWebView) the cache monitor showed the L2
  OPFS cache "healthy" but permanently empty — 0 entries, 0 B, 0 % hit rate —
  with tens of thousands of write errors. WebKit implements
  `navigator.storage.getDirectory()` and file handles but NOT the main-thread
  `FileSystemFileHandle.createWritable()` (OPFS writes there require
  worker-side `createSyncAccessHandle`), so the store mounted successfully and
  then failed every single put.
- **Fix**: `OPFSStore.init()` now runs a tiny timeout-wrapped write probe
  (create → write → close → remove). If the environment cannot actually write,
  the store degrades to the ordinary OPFS-unavailable path — L1-only operation,
  the `opfs-unavailable` badge, and one clear warning — instead of an error
  storm. Transient mid-session I/O failures still increment `writeFailures`
  as before.

#### Changed — three-geometry loader symmetry + geometry-neutral monitor naming (viewer + Python)

- **Why**: the Points spatial-index loader had drifted from the Lines/GSplats
  facade shape (a monolithic `loadPoints` with inlined S-cache/query-tracking),
  ~300 lines of byte-identical private helpers were duplicated across the three
  loaders, and several monitor names said "points" while counting vertices,
  segments, or splats — the kind of compat debt this project explicitly rejects.
- **Refactor**: `loadPoints` is now a thin wrapper around `loadPointsInternal`
  (mirroring Lines/GSplats); the shared facade orchestration lives in
  `data/loaders/spatial-facade.ts` (`loadSliceWithCache`, `recordLoadMetrics`,
  `runWithActiveSignal`, `runWithResidencyProbe`, one per-loader
  `SpatialFacadeCtx`) and `loader-metrics.ts` (`finishQueryTracking`,
  `makeInitialLoaderMetrics`, `buildSpatialIndexMetrics`).
- **Renames (no compat shims)**: `LoaderMetrics.pointsLoaded → elementsLoaded`
  and `LoaderMetrics.visiblePoints → visibleElements` (loader-level,
  geometry-neutral; the per-geometry `GlobalStats` / scene-graph trio
  deliberately keeps `visiblePoints` / `visibleSegments` / `visibleSplats`),
  `GlobalStats.totalPoints → totalElementsLoaded` (it always summed all
  three geometries), `MonitorEvent.data.points` / `QueryInfo.points →
elements`, `PointSpatialIndexMetrics → SpatialIndexMetrics`
  (+ `avgElementsPerCell`), and a shared `ElementRange` replaces the
  `as unknown as PointRange[]` casts. Python scene-node alias properties
  `n_points` / `n_vertices` / `n_splats` removed (`n_elements` is the one
  count property; on-disk metadata keys unchanged).
- **Fixed**: the Lines loader never wrote `visibleElements` (the monitor
  permanently showed 0 for lines layers — now the visible segment count,
  labeled `segs`); Points `dispose()` leaked its active-query map;
  the monitor's `avgCellsPerQuery` /
  `queryEfficiency` decayed ~1/n with session length (last-query cells over
  cumulative queries — now a true cumulative mean, so long sessions no
  longer drift into spurious low-efficiency recommendations); dead monitor
  fields (`totalCacheHits`, `totalPointsLoaded`, `totalMemoryUsed`,
  `globalCacheHitRate`, `activeFallbackLoaders`) and the unused `profiler`
  loader-constructor param deleted.
- **Added**: chunk-index `spatialIndex` telemetry is now reported by all three
  loaders (the monitor advisor's query-efficiency recommendations previously
  worked only for points); test suites brought to full three-way parity, with
  regression pins for both fixed bugs and direct unit coverage for the new
  shared facade helpers.

#### Added — first-class background/floor suppression (`--floor`, on by default) + companion whole-volume auto-tiling

- **Why**: a constant background pedestal / DC offset (camera offset,
  autofluorescence, scattered light — ubiquitous in real microscopy) is the
  worst case for a localized-Gaussian basis: the optimizer wastes splat capacity
  tiling the background with low-amplitude "haze" or under-fits the real signal.
  On the neuromast iSIM dataset a ~110-count pedestal held a naive fit at
  18.5 dB PSNR; subtracting the floor lifted it to ~47 dB (+28.5 dB).
- **`--floor auto|none|pN|<float>`** on `gsplat fit`, `cal`, and `batch-fit`,
  **default `auto`** — histogram-mode estimate of the low-intensity bulk, capped
  at the median so it can never eat real signal (a no-op on clean data with no
  pedestal). The floor raises the effective `image_min` used in normalization,
  so the pedestal clips to 0; output amplitudes are background-relative (the
  subtracted level is recorded in fit stats / `gsplat info`, not added back).
- **`cal`** subtracts the floor ONCE up front (not per-fit) so the fit target,
  render, and held-out reference stay on one scale; K\* is thus measured the same
  way you will fit. `--floor none` reproduces the legacy hard-min numbers.
- **Progressive fits** subtract the floor once up front and run every pass with
  floor `none` (the residual chain is built against the subtracted volume, so
  the pedestal is never reintroduced). Threaded through the parallel-tiled and
  content-tiled worker commands too, so an explicit `--floor` is honored at scale.
- **Companion auto-tiling fix**: `--tiling auto` now decides whole-volume vs
  tiled from a total voxel budget (not any single dim > tile size), so small and
  medium stacks fit whole-volume — eliminating the background tile seams that
  independent per-tile pedestal fits produced.

#### Added — k_knee "operating point" (point of diminishing returns) on the cal held-out curve

- **Why**: `k_star` is the splat-budget anchor (the max K for a signal-limited
  curve), but users also want the earlier knee where returns level off.
- `HeldOutPeak` now reports `k_knee` (argmax for a clear peak, else the smallest
  K within 0.3 dB of the max) plus supporting per-regime metadata; additive and
  defaulted so older `cal.json` files hydrate cleanly (`k_knee` falls back to
  `k_star`). `find_k_star`'s per-regime `k_star` selection is unchanged; a
  last-step floor guards a flat-topped plateau from inflating a signal-limited
  budget. The cal CLI prints the operating point when it differs from `k_star`.

#### Fixed — adaptive DPR: sub-throttle distress verdict (heavy scenes no longer latch at native)

- **Why**: on a heavy scene pinned at a uniform ~10 fps, the viewer's adaptive
  resolution refused to lower the DPR — it showed `DPR 2.00 @ 10 FPS`
  permanently. The refresh-rate estimator's throttle detector misread the
  GPU-bound plateau as a browser rAF throttle (the display had proven ~120 Hz
  earlier), reseeded its cap onto the loaded FPS, and the cap-relative
  thresholds then read 10 fps as "at the display cap = healthy": scale-up
  walked the DPR back to native and parked it there, with an exit line
  (plateau ×1.25) unreachable while GPU-bound.
- **Fix**: no real display/rAF mode runs below ~23.976 Hz (film/TV) while the
  user interacts, so a sustained
  plateau below `MIN_THROTTLE_PLATEAU` (22 fps — safely under the slowest real
  display mode, 23.976 Hz film/TV) is now classified as content
  **distress**, never a throttle: the cap stays fallback-floored (scale-down
  stays armed) and the estimator raises a one-shot verdict that the manager
  answers with a DPR-ceiling demotion to exactly 1.0 in one step
  (`BoundsLedger.demoteCeiling`, same TTL/backoff ladder as the
  punished-ascent demotion). Genuine throttles (≥ 22 fps plateaus, e.g.
  120→30 low-power) keep the existing downshift behavior.
- **Verified live**: the same repro (90 ms rAF stall) now demotes to DPR 1.0
  within ~10 s of sustained distress, the cap stays ~144, and the normal
  probe walk continues below 1.0 once the rejected-probe floor expires.

#### Fixed — barrier-aware GSplat chunk ordering (per-timepoint load locality)

- **Why**: navigating to a fresh timepoint in a dense timelapse loaded high-res
  data far slower than expected. GSplats ordered chunks by a Hilbert curve over
  **all** center axes including time, so chunks straddled timepoints and a
  single-timepoint query over-fetched (~2.5–3.5× the ideal, chaotically). Points
  and Lines already avoid this via compound ordering (discrete/barrier dims
  first); GSplats had diverged.
- **Fix**: GSplat ordering (`sort_splats_spatial`) and chunk bounds
  (`compute_chunk_bounds_gsplats`) are now barrier-aware, sharing a single
  `_compound_sort` core with Points/Lines. Categorical/barrier axes (time,
  channel) are grouped first and get tight ±0.5 chunk bounds (no σ expansion).
  The barrier is taken from an explicit `barrier_dims`, else the persisted LOD
  `coarsen_dims` complement, else a conservative auto-detect
  (`detect_barrier_dims`). Threaded through every gsplat write path
  (`write_gsplats_tree` / `write_partition_streaming` / scene compiler / batch
  merge / `reencode`); 3D data keeps identical splat ordering and `chunk_bounds`
  (leaf `.zattrs` gain informational `slice_dims=[]` / `ordering_dims` keys).
- **Measured** (51-timepoint h2afva, 127 M splats): per-timepoint chunk hit-rate
  went from a chaotic 0.01–7 % to a uniform ~1.96 % (= ideal 1/51) at every
  timepoint. Viewer needs no change (dimension-agnostic AABB chunk selection);
  re-order an existing dataset with `luxar gsplat reencode`.

#### Added — uint16 LUT encoding tier (exact few-color storage up to 65,536 uniques)

- **Why**: the LUT strategy is lossless (exact original values + integer
  indices) and beats any quantized encoding when it fires, but the producer
  hard-capped it at 256 uniques with uint8 indices — the 257..65,536 band
  (color-by-track/lineage: thousands of distinct colors) fell through to
  approximate quantized encodings at 3× the size.
- The encoder now emits `lut_uint16` for 257..65,536 unique ROWS (colors;
  scalar mode is structurally excluded — u16 indices cost exactly what
  quantized scalars cost, so the LUT JSON would be pure overhead), gated by
  a byte-modeled benefit rule: the LUT lives as JSON in
  `.zattrs` and is duplicated by consolidated `.zmetadata` (parsed at
  scene-open for every node), so the doubled JSON must cost at most half the
  raw savings over the cheapest realistic alternative AND stay under a new
  `ArrayEncoder(lut_json_max_bytes=...)` cap (default 512 KiB). Accepted
  uint16 LUTs are always strictly smaller than even the lossy alternative —
  while being exact.
- The uint8 tier's output is **byte-identical** to before (legacy rules
  preserved verbatim); eligibility and encoding now share ONE `np.unique`
  pass (was two). 64-bit integers beyond ±2^53 no longer LUT-encode (JSON
  fidelity guard, compared in the INTEGER domain so ±(2^53+1) can't slip
  through float rounding — pre-existing silent-corruption latent bug in the
  u8 tier). INDEX arrays never LUT-encode: the viewer reads line segments
  raw with no decode dispatch, so a LUT would silently corrupt connectivity
  (also a latent pre-existing hazard at K≤256, now closed).
- Decode needed no changes anywhere (Python decoder, viewer, worker, WASM
  all shipped `lut_uint16` support long ago); coverage is now organic via
  encoder-emitted fixtures + an E2E spec exercising the Uint16 row kernel
  in-browser. Viewer robustness: an unknown `lut_mode` now fails loud on the
  main thread too (the worker already threw; absent still defaults for 1-D).

#### Added — measured LOD quality (Q·e stamps): early upgrade swaps, sibling-aware ladders, annotate-quality

- **Why**: the never-downgrade display gate released LOD upgrades on
  committed-COUNT crossover, which on shared-base stream ladders lands
  structurally ~2 chunks from the ladder END (measured on real microscopy:
  crossovers at chunk 5/7, 7/9, 9/11, 11/13) — upgrades felt like "waits
  until fully loaded". Counts also compare apples to oranges across
  substitutive levels.
- **Quality stamps at build time** (format-additive, no version bump):
  `lod_stats.energy_fraction_cum` (cumulative committed self-energy fraction
  e(k) per additive sub-LOD), `level_stats.reference_energy` (the leaf's
  self-energy weight w for partition aggregation; group-consistent finest
  total inside lod groups), and `level_stats.quality` (measured mixture-L²
  Q of each level vs its group's finest content — new constant-cost sampled
  estimator in `luxar.gsplats.lod.quality`). On by default in every recipe
  (`--no-quality-stamps` to skip the Q measurement).
- **Sibling-aware stream ladders**: inside a lod group, every level with a
  coarser sibling starts its `stream:C` ladder at `max(C, ceil(n/(2·K)))`,
  so the upgrade catch-up fires at chunk 1-2 by construction (the coarsest
  level keeps the small user base — fast first paint unchanged). Applied by
  the levels/adaptive recipes and overview's fine partition.
- **`luxar gsplat annotate-quality <store>`**: retrofits the stamps onto an
  existing `.gsplats.zarr` IN PLACE (no refit, no re-ladder) — e(k)/w are a
  cheap O(N) pass over amplitudes + the Cholesky diagonal; `--with-quality`
  adds the measured Q. Re-stamps the root `content_hash` so viewer caches
  invalidate automatically.
- **Viewer**: the LOD display gate now releases an upgrade swap once the
  streaming candidate's committed energy reaches 0.6 of its total
  (`ENERGY_RELEASE_THRESHOLD`; w-weighted aggregate over partition subtrees,
  known-empty parts excluded) — chunks earlier than the count crossover,
  which remains the fallback for unstamped legacy datasets. Layers panel
  shows the on-screen level's quality estimate (`L2/3 · ~60%`); the data
  monitor's additive chip shows `LOD k/n ~NN%` with a didactic tooltip.

#### Changed — HDR colors quantize to per-channel true-log uint16 (~4× smaller, visually lossless)

- New `geolog_perchannel_u8/u16` encoding — the per-channel member of the
  geolog family (completes the scalar↔per-channel matrix:
  `bounded`↔`linear_perchannel`, `geolog`↔`geolog_perchannel`): each column
  quantized on its own min/max-anchored TRUE-log grid (ln-domain
  `col_lo`/`col_hi`), uniform relative precision across the column's whole
  dynamic range, code 0 reserved for exact zeros (no positive value can
  quantize to zero; the reserved level is the name's contract).
- HDR COLOR policy: AUTO → `geolog_perchannel_u16`, MEMORY → u8, PRECISION →
  float32 (previously float32 in ALL modes — the last unquantized hot
  attribute class). Applies to all three geometry types through the shared
  COLOR semantic type.
- Grounded in a 6-dataset spike (five h2afva timepoints + a 2.54M-splat fit;
  realistic volume-sampled colors + 12 synthetic distributions, 2–12.6
  realized decades): true-log dominated linear fixed-point AND log1p
  per-channel at EVERY dynamic range (realistic data: rel-err p95 1.8e-4
  uniform vs ~100% for both alternatives; faint-exposure renders ≥147 dB vs
  88–128 dB; ~4× smaller than float32 — and no linear→log rail needed).
- Full decode stack: Python decoder, viewer `makePerChannelDequant`,
  range-loader worker path, WASM Rust kernels + TS reference (bit-exact
  three-way parity), fixtures with u8+u16 coverage.

#### Fixed — adaptive DPR overhaul: correct indicator, sharp resting frames, no blur metronome, monitor-change safety

- The resolution indicator now shows percent-of-native: on a retina display
  the first reduction reads "Resolution Scaled to 90%" instead of the absolute
  "180%", the percentage tracks later steps while the toast is visible, and
  the target FPS reflects the refresh-relative rule.
- The resting frame is always sharp: when the render loop idle-pauses, DPR
  snaps back to native and one full-quality frame is rendered (guarded off
  during recordings and context loss); resuming interaction snaps back to the
  remembered operating DPR in ONE step instead of reactively re-walking the
  reduction ladder. Previously a reduced-DPR static image stayed blurry
  indefinitely (recovery needed ~50 s of uninterrupted high-FPS interaction).
- No more 30-second blur metronome: rejected U-shape probes back off
  exponentially (30 s → 60 s → 2 min → capped 5 min) on scenes where DPR
  reduction never helps; genuine content changes (dataset/layer/LOD swaps)
  re-arm probing within 5 s. Probes now settle only on clean samples (min
  frames/span, no data-loading jank) and void as inconclusive otherwise;
  frame-gap detection stops GC/decode stalls and idle-resume gaps from
  ratcheting spurious scale-downs.
- FPS thresholds are refresh-rate-relative (75%/90% of the display's
  estimated rAF cap) instead of fixed 50/58: recovery no longer stalls on
  60 Hz displays that drop a couple of frames (plus a mid-band grace sample),
  120 Hz displays scale down at 80 fps as they should, and a 30 Hz-throttled
  tab neither probe-loops forever nor is barred from scaling up.
- The native DPR is read live and rebases learned state on change: dragging
  the window to a lower-DPI monitor (or browser zoom) can no longer leave a
  supersampling override behind — including via the screenshot/recording
  save-restore round-trip, which also no longer runs a redundant second
  render-target reallocation pass.
- Stale-data hygiene: the Performance popover FPS row reads `idle` while the
  loop is paused instead of freezing a stale number; WebGPU device loss now
  latches the shared context-lost predicate so cheap no-op frames can't drive
  bogus scale-ups; frames during WebGL context loss are ignored too.
- The `AdaptiveDPRManager` is decomposed into pure timestamp-driven modules
  (`rendering/adaptive-dpr/`), all tuning knobs live in a validated
  `adaptiveDPR` config section, and the previously hardcoded probe constants
  are configurable.

#### Added — evidence-based DPR ceiling + `?dpr=` URL param

- On HiDPI displays, repeated "punished ascents" (a scale-up above DPR 1.0
  followed promptly by an FPS collapse) demote the operating ceiling from
  native to exactly 1.0 for the session — cutting the structural up/down
  oscillation on heavy scenes instead of merely slowing it. The demotion is
  TTL-decayed with backoff, softened by content changes, and never affects
  the idle resting frame (still full native).
- `?dpr=<value>` pins a fixed pixel ratio and locks adaptive resolution off
  for the session (clamped to [0.25, native]) — used by the visual-regression
  suites for deterministic baselines and handy for bug repros.

#### Changed — geolog amplitude quantization (rescale-first, zero-safe) + per-dtype compressors

- New `geolog_scalar_u8/u16` encoding: wide-dynamic-range positive scalars
  (gsplat amplitudes; shared path with points radii / lines widths) are
  quantized AFTER rescaling to the array's own nonzero `[min, max]` on a true
  log grid — uniform ~0.013% relative error across 7 decades at u16, with
  **level 0 reserved for exact zeros** so no nonzero splat can quantize to
  zero by construction. Replaces the float32 fallback the old heuristic used
  (and the legacy 0-anchored `log_scalar` family, which zeroed 3,026 splats
  on the real t252 amplitudes; still decodable, no longer produced).
  Validated on 2.54M real splats: 133.8 dB render vs float32, 121.7 dB on the
  faint-structure metric; float16 measured and refuted (4x worse, TS-banned).
  Full decode support: Python, viewer, range-loader, WASM (Rust + TS parity).
- Width-aware per-dtype compressor policy (from the manuscript
  `codec_selection` supplementary): u16 codes → zstd-l9/byte-shuffle, u8 and
  float arrays → zstd-l9/unshuffled, replacing the uniform zstd-l3/bitshuffle
  default that Blosc silently neutralises at 64 KiB chunks. Applied through a
  single encoder chokepoint plus the direct-write sites (chunk bounds,
  labels); fixes two arrays that silently used zarr's lz4 default
  (colormap LUTs, batch denoise intermediates). Decode is level-independent —
  read cost unchanged, stores ~20% smaller before the amplitude win.
- Viewer: the per-channel decode family (`linear_perchannel_*` centers,
  `log_perchannel_*` / `signed_log_perchannel_*` Cholesky factors) now decodes
  in the worker on new Rust/WASM kernels (`decode_{linear,log,signed_log}_
perchannel_{u8,u16}`) above the same threshold as the other encodings —
  previously the only hot decode path still running per-element on the main
  thread (an `expm1` per element for the Cholesky pair). f64 scales cross the
  boundary as Float64Array, so worker, TS-fallback, and main-thread decodes
  are bit-identical (three-way parity tests); sub-threshold ranges and worker
  failures keep the main-thread `makePerChannelDequant` path.
- Rescale-first generalised to the sibling encodings: `bounded_scalar_u8/u16`
  now anchor the linear grid at the array's own `[min, max]` instead of
  `[0, max]` (encoder-only — decoders already honoured the stored min), and
  the per-channel Cholesky pair (`log_perchannel_*` /
  `signed_log_perchannel_*`) gains **`zero_level: true`**: per-column scales
  from each column's nonzero min/max with code 0 reserved for exact zeros,
  so axis-aligned splats keep exactly-zero off-diagonal correlations instead
  of tiny spurious ones, and zeros stop dragging the scale anchor down.
  The covariance certificate round-trips through the identical transform.
  Legacy arrays (no flag) keep their original decode in Python and viewer.

#### Changed — AUTO covariance quantization: uint8 with an encode-time certificate (~3.3× smaller)

- Gsplat Cholesky factors at AUTO now default to the uint8 per-channel log
  encodings (formerly the MEMORY tier) instead of uint16 — measured on a real
  light-sheet fit at 94.5 dB vs the float32 render (~46 dB below the fit-error
  floor, end-to-end invisible) and 2.48 B/splat compressed vs 8.25 (~3.3×).
- AUTO keeps a _measured_ reason to go richer: the new
  `ArrayEncoder.encode_cholesky_split` joint entry point round-trips both
  halves through the exact quantization transform, rebuilds Σ = L·Lᵀ, and
  escalates to uint16 (then float32, practically unreachable) when the p95
  per-splat relative Frobenius error exceeds 0.05 — e.g. merged heterogeneous
  stores whose σ columns span many decades. The measurement is recorded as
  provenance in each array's own `encoding.certificate`; decode never needs it
  and both halves always share one tier. MEMORY stays uint8 unconditionally;
  PRECISION stays float32. No format change — readers were already
  bit-width-agnostic.
- VQ/codebook covariance compression was evaluated and refuted for this
  storage stack (2026-07 spike): codebook index streams are entropy-dense and
  defeat zstd+bitshuffle, losing to plain u8 scalar codes on compressed bytes
  at equal PSNR.
- The certificate measures a bounded evenly-spaced row sample above 262 144
  splats (recorded as `certificate.sample`; quantization scales always come
  from the full columns), keeping the float64 Σ scratch capped on large flat
  fits instead of scaling with N.

#### Fixed — LOD level switches no longer dip to chunk-1 quality (never-downgrade display gate)

- A lazy substitutive level becomes displayable after its **first** additive
  chunk commits, so switching to a cold level (zoom in, zoom out, or after a
  scrub settles) popped displayed quality down to chunk-1 and climbed back
  over the following refinement passes. The LOD registry now holds the
  previously-displayed level while the streaming target's committed element
  count is strictly below it, releasing on ladder completion (committed, not
  just fetched — every geometry commit now stamps `committedLadderComplete`
  next to the count, so the release is race-free by construction), count
  crossover (the ladder tail then streams visibly), ladder failure, or the
  held level losing freshness. Fast first paint is unchanged — a group with
  nothing better on screen still swaps immediately — and explicit level
  locks / off-screen groups bypass the gate.
- The gate covers **arbitrary nested hierarchies**: a lod_group child that is
  a whole subtree (the `overview` recipe's fine `kind=partition` branch,
  adaptive/hand-authored lod-of-partition nestings) participates through a
  visibility-respecting subtree aggregate of its leaves' commit stamps —
  inner lod_groups' own level toggling shapes the aggregate to exactly what
  would render, and a stale-content subtree is never held over fresh data.
- Deferred lod_group subtree activation now kicks the progressive-refinement
  orchestrator: previously nothing scheduled refinement outside update-view
  tails, so an `overview` fine branch activated by zooming in sat at its
  first additive chunk per part until the next slice change.

#### Fixed — stale-cache black screen on regenerated `.gsplats.zarr` + viewer LOD loading/scheduling

- Every `.gsplats.zarr` save now stamps a root `content_hash` (metadata-only
  xxhash64; the per-save `timestamp` folds in), and the viewer validates
  datasets without one via an implicit `zattrs-hash` token (SHA-256 of the raw
  root `.zattrs` bytes) — so a dataset regenerated in place at the same URL
  invalidates the persistent (OPFS) cache instead of serving a stale mix of
  old metadata and zero-filled chunks (the black-screen failure). Caches
  poisoned before the fix need one `?clear-cache` reload.
- The LOD registry never displays a fresh-but-empty level over a populated
  coarser fresh one (warns once, pointing at `?clear-cache`).
- Failed loads are now recoverable without a reload: the Data Loading
  Monitor's Overview tab shows a warning banner with a Retry action while
  failures are recorded, and failed loads are retried automatically when
  the connection comes back online.
- Viewer LOD loading/scheduling fixes: post-update refinement now covers
  points/lines additive ladders (not just gsplats); >3D progressive points
  concatenate at the correct stride; refinement retries are capped at 3
  consecutive failures per run and in-flight refinement reads abort when a
  new view-state arrives; eager loaders join the update sweep only after
  their initial load settles; lazy LOD levels are retryable via
  `retryFailedLoader`; pending updates progress in hidden tabs; abort
  classification is realm-proof (`.name`-based, not `instanceof Error`).

#### Changed (breaking) — recipe vocabulary renamed (plain-language names)

- The `--recipe` vocabulary on `gsplat lod`, `gsplat fit --recipe`, and
  `batch-fit submit/merge/run --merge-recipe/--recipe` is renamed to
  plain-language names: `additive`→**`stream`**, `substitutive` &
  `pyramid`→**`levels`**, `partitioned`→**`tiles`**,
  `multiscale`→**`overview`**, `mosaic`→**`adaptive`** (`flat` unchanged).
  Old names are rejected with a pointer to the new spelling; stored batch
  manifests from pre-rename runs are translated silently (including the
  generated Slurm merge command). Python math-layer names
  (`make_substitutive_lod` etc.) are unchanged. In the persisted `pipeline/`
  provenance, `recipe` (the build instruction) is recorded distinctly from
  `lod_kind` (the viewer-facing reduction mechanism).

#### Added — `--refine volume` warm-start volume re-fit of coarse levels

- **`refine="volume"`** on `make_substitutive_lod` / `make_lod_pyramid` /
  `RecipeParams` (CLI: `luxar gsplat lod --recipe levels --target <volume>
--refine volume [--refine-iters N]`): each merged level is warm-start
  re-fitted against the source volume itself (`fit_gaussian_splats` seeded by
  the merge; identity-preserving — no cull/dynamic-ops, colors carried over).
  Benchmarked on real microscopy: +5–6 dB full-res / +10–12 dB at viewing
  scale over the merge, with the warm start beating a cold fit and drifting
  ~2× less across levels. Never worse than the merge: each level keeps
  whichever of {merge seed, re-fit} renders closer to the volume (MSE); the
  re-fit's DC is pinned to the seed's (following `conserve_mass`, so no
  cross-level brightness pop), and a seed in a non-voxel coordinate frame — enlarged
  (bbox check) or shrunk/rotated/axis-swapped (post-fit per-splat relocation
  check) — keeps the seed with a warning rather than storing a misplaced
  level; batch merge rejects refine='volume' up front (volume-free by design). `refine_iters` omitted
  resolves to 300 (the volume default) in both the API and the CLI. New `gsplats/lod/volume_refit.py`
  engine. Scope: `levels`/`overview` recipes, no barrier dims (`adaptive` and
  `coarsen_dims` are rejected loudly); needs the volume in hand, so exposed on
  `gsplat lod` via `--target` (fit-time and batch-merge are follow-ups).
  Fitting a blurred/downscaled volume proxy was benchmarked and rejected.

#### Added — `--refine l2` post-merge refinement of substitutive levels

- **`refine="l2"`** on `make_substitutive_lod` (CLI: `--refine l2` /
  `--refine-iters`, token "substitutive"): each merged level is post-optimized
  with Adam against the closed-form mixture-to-mixture L² residual over sparse
  neighbour pair lists, starting from the coverage-inflated moment-matched
  seed. Unlike the per-bin merge (structurally blind to cross-bin overlap),
  this objective sees coverage gaps and over-blur, so the refit widens splats
  where the field is flat and keeps isolated structure tight (measured:
  rel-L² 0.089 vs 0.151 for the β=3 merge on flat fields; peak preservation
  0.99 vs 0.91 on isolated blobs). Trusted-checkpoint discipline guarantees
  the returned iterate is never worse than the seed in the trusted metric.
  Default remains `none`.

#### Added — mass conservation + proportional ridge (LOD brightness-pop fix)

- **`conserve_mass=True`** (CLI `--conserve-mass/--no-conserve-mass`): each
  reduced substitutive level's amplitudes are rescaled per barrier group so
  total mass over the coarsened dims matches the fine input's — fixing the
  up-to-−11.5 % per-time-slice DC drift (visible as a brightness pop at LOD
  switches) caused by the per-bin merge not being mass-preserving.
- The merge's absolute `1e-6` Cholesky ridge is now **proportional**
  (`1e-9·diag`, retry `1e-6·diag`), so near-delta sigmas on barrier axes
  (e.g. a time axis) are no longer inflated ~300 000×.

#### Fixed — fitting/pipeline stats survive the `.gsplats.zarr` round-trip

- Pipeline-level fitting info (lod kind, method, `compression_factor`,
  `coverage_inflation`, `refine`, …) is now written to a new optional
  `pipeline/` root group and merged back on load (previously silently
  dropped). `refine_stats` and other nested `level_stats` dicts round-trip
  via JSON-safe encoding (`json_safe_value`). Spec updated
  (`docs/specs/GSPLATS_ZARR_FORMAT.md`); the group is optional, so existing
  v3.1 stores remain loadable unchanged.

#### Changed — `.gsplats.zarr` format v3.2 (versioned `coverage` selector attrs + migration)

- The `kind=lod` selector attr rename (`selector: "pixel_size"` → `"coverage"`,
  per-child `min_pixel_size` → `coverage_fraction`) is now a **versioned**
  format change: the writer stamps `format_version: "3.2"`; readers accept
  3.0–3.2. Pre-v3.2 datasets are **auto-adapted by the viewer** (legacy
  `min_pixel_size` ladders normalized to coverage fractions, with one warning
  naming `migrate-format`) instead of silently pinning every LOD group to the
  finest level, and `luxar gsplat migrate-format` now also upgrades v3.0/v3.1
  stores that still carry the legacy `pixel_size` attrs (detected as
  `v3.x-lod-pixel-size`) to v3.2 with derived `coverage_fraction` thresholds.

#### Changed — viewport-relative `coverage_fraction` LOD switch threshold (replaces absolute-pixel `min_pixel_size`)

- Substitutive/LOD switch thresholds are now a single **viewport-relative**
  `coverage_fraction` per child: `sqrt(N_i / N_finest)` (`N_i` = level i's
  total splat count), strictly ascending coarsest→finest (coarsest `0.0`,
  finest `1.0`). Being a count _ratio_, it is immune to non-displayed-dimension
  multiplicity (e.g. a stacked time axis inflates every level equally and
  cancels). The viewer multiplies each `coverage_fraction` by the live
  viewport diagonal to get the pixel comparison, so the finest level shows
  when the object fills the screen and coarser levels step in as it shrinks
  — self-calibrating identically on any monitor/viewport.
- **Removed** the `extent` (`T·W/r`) and legacy `count` (`√N`) threshold
  methods and their tuning knobs — `--lod-method` / `--extent-percentile` /
  `--extent-anisotropy` / `--base-pixel-size` (and the `--merge-lod-method` /
  `--recipe-lod-method` forms) — from `gsplat lod`, `gsplat fit --recipe`, and
  `batch-fit`. There is no replacement knob; the derivation is automatic.
- **Removed** the matching Python API knobs: `RecipeParams.lod_method` /
  `extent_percentile` / `extent_anisotropy` / `base_pixel_size`;
  `GSplatData.save()` dropped the same kwargs; `add_lod_group()` dropped
  `base_pixel_size` and its `selector` default is now `"coverage"`; the
  `substitutive_lod=`/`lod_group=` explicit-override key `min_pixel_sizes=[...]`
  is now `coverage_fractions=[...]` (values in `[0, 1]`).
- On disk, the child zarr attr `min_pixel_size` is now `coverage_fraction`,
  and the `kind=lod` group's `selector` attr is now `"coverage"` (was
  `"pixel_size"`).

#### Added — bandwidth-aware streaming additive LODs + `gsplat additive` (per-leaf laddering)

- **`stream:<c>` breakpoints** — a new additive-ladder breakpoint form: geometric
  cumulative cuts `[c, 2c, 4c, …, N]` (first chunk `c` splats, then doubling),
  resolved against each call's own N so ONE spec adapts per part / per
  substitutive level (silently clamped for small parts, capped at 16 levels,
  sliver tails folded). Works everywhere additive ladders are built: `gsplat
lod`, `gsplat additive`, `fit --recipe additive`, `batch-fit merge/submit/run`.
- **`--target-ms` / `--bandwidth-mbps` / `--bytes-per-splat`** on all those
  surfaces: derive `stream:<c>` from a download budget — e.g. `--target-ms 200`
  at the default 25 Mbps sizes the first chunk to ~200 ms of download (fast
  first paint; the viewer streams additive sub-LODs progressively). Bytes/splat
  is measured from the input store when one exists, else an analytic estimate;
  always logged; `--bytes-per-splat` overrides. At `batch-fit submit/run` the
  trio resolves at plan time into the stored breakpoints string (no manifest
  schema change).
- **NEW `gsplat additive <in> <out>`** — give every leaf of an EXISTING tree an
  additive ladder, structure-preservingly (substitutive `kind=lod` levels,
  partition parts, mosaic groups keep their shape) WITHOUT recomputing the
  expensive substitutive/partition structure. The per-leaf counterpart of
  `lod --recipe additive` and the inverse companion of `gsplat flatten`.
  E.g. `gsplat additive sub.gsplats.zarr pyr.gsplats.zarr --target-ms 200`
  turns a 23 M-splat substitutive pyramid into a substitutive × streaming-additive
  pyramid (~200 ms first paint per level) in one command.

#### Added — coverage inflation for substitutive coarsening (grid-ripple fix)

- **`coverage_inflation` (default 3.0)** on every substitutive reduction
  (`substitutive` / `pyramid` / `multiscale` / `mosaic` recipes, all methods):
  merged representatives get their inter-center spread widened
  ×`coverage_inflation` (mass-preserving) so neighbouring coarse splats sum
  flat — suppressing the axis-aligned grid ripple pure moment matching shows
  at coarse levels. CLI: `--coverage-inflation` on `gsplat lod`, `fit
--recipe substitutive`, and `batch-fit --merge-recipe`/`merge --recipe`;
  pass `1.0` for the historical pure moment match.

#### Added — data-loading monitor: streaming-LOD visibility (viewer)

- **Additive-chip glyphs** in the loading-monitor tree: `LOD x/N` detail
  levels loaded, `●` last refinement fully cache-resident vs `◌` still
  streaming, `⏳` refinement in progress — with every glyph spelled out in
  the chip tooltip.
- **Active substitutive level highlighting**: the tree now marks which
  `kind=lod` level actually renders (inactive levels dimmed), re-marked per
  tick from the group's `activeLevel`.
- **Per-node visible counts**: badge tooltips read "N elements (M visible
  after slicing)", symmetric across points / lines / gsplats, pushed from
  the SceneLoader's visible-counts walk.

#### Fixed — additive-breakpoints edge cases

- **`counts:` on small parts no longer aborts the build**: `partitioned` /
  `pyramid` / per-part merge with explicit `counts:` breakpoints used to raise
  "largest breakpoint exceeds N" on any part/level smaller than the largest
  count, aborting the whole build. Per-part ladders now clamp the counts to
  each part's own size (`clamp_counts_breakpoints`), but the spec is still
  strictly validated ONCE against the full dataset N
  (`validate_counts_breakpoints`) so a dataset-scale typo (e.g.
  `counts:1000000` on a 50 k dataset) aborts loudly as before; direct
  whole-dataset builds keep the strict validation.
- The empty-leaf (n==0) fast path now labels its breakpoints kind `"none"`
  instead of mislabeling the requested spec as `"equal-count"`; boolean values
  are no longer accepted as integer count breakpoints.

### June 2026

#### Changed (breaking) — `.gsplats.zarr` format v3.0 → v3.1 (split Cholesky factors, per-channel differential quantization)

The on-disk packed `cholesky_factors` array is split into two independently
encoded arrays: `cholesky_factors_diag` (N, d) and `cholesky_factors_offdiag`
(N, k−d, where k = d·(d+1)/2; absent when d == 1). Each is quantized with a
per-channel ("differential") scheme — the diagonal uses `log_perchannel_u16`
(AUTO) / `log_perchannel_u8` (MEMORY) / `float32` (PRECISION); the off-diagonal
uses `signed_log_perchannel_u16` / `signed_log_perchannel_u8` / `float32`,
selected by the new `CHOLESKY_DIAG` / `CHOLESKY_OFFDIAG` semantic types. uint8
is measured visually lossless (≥93 dB vs the float32 render); decode is always
to float32 so GPU/shader/WASM paths are unchanged. `FORMAT_VERSION` is bumped to
`"3.1"` and `SUPPORTED_FORMAT_VERSIONS` is now `("3.0", "3.1")` — v3.0
single-array files are still read transparently (loaders fall back to the packed
`cholesky_factors` when `cholesky_factors_diag` is absent). The change is on-disk
only: the in-memory packed `GSplatData.cholesky_factors` (shape (N, d·(d+1)/2))
is unchanged. Both the Python scene reader and the gsplat-tree decoder share one
`recombine_cholesky` helper. `migrate-format` gains `--lossless` (PRECISION) for
exact archival float32 migration; the default (AUTO) re-encodes legacy Cholesky.

#### Changed (breaking) — `gsplat slurm-fit` renamed to `gsplat batch-fit`, plus new local multi-GPU `batch-fit run`

- **RENAME (breaking, no alias):** the `gsplat slurm-fit` command group is now
  `gsplat batch-fit`, making whole-timelapse fitting scheduler-agnostic. Verbs:
  `submit` (Slurm cluster array job, unchanged behavior) and the new `run`
  below; `status` / `validate` / `merge` / `cancel` are shared across both
  backends. `luxar gsplat slurm-fit` no longer exists.
- **NEW `gsplat batch-fit run`** — local multi-GPU whole-timelapse fit (no
  Slurm; the local sibling of `batch-fit submit`). Plans once (uniform tiles or
  a shared content box plan over T×C), fits every (t, c) task with a multi-GPU
  subprocess pool (one worker pinned per GPU, per-GPU concurrency sized from free
  VRAM), then stream-merges to one `kind=partition`. Resumable (re-running skips
  tiles already on disk) and writes `manifest.json` so `status` / `validate`
  work locally. Options mirror `submit` minus Slurm, plus
  `--gpus auto|all|cpu|'0,1,3'`, `--jobs-per-gpu`, `--no-resume`, `--dry-run`.

#### Added — GSplat fitting follow-ups (transform partitions, fit/merge per-part LOD, fitted density exponent, cluster content fan-out)

Four gsplat additions, each deep-double-checked:

- **`gsplat transform` is tree-aware** — it now preserves a `kind=partition`
  (and `mosaic`/`multiscale` trees) instead of flattening/rejecting it, walking
  the tree leaf-by-leaf with global `--center` / `--normalize-intensity` stats
  and re-deriving the extent-based `min_pixel_size` LOD thresholds.
- **`fit --recipe additive|substitutive`** — a tiled fit can emit per-part LOD
  (the `partitioned` / `mosaic` topology) at fit time, without a separate `lod`
  pass (which rejects a partition). Mirrors the `lod` knobs. **Any** per-part
  recipe on `--tiling uniform` (Hann-apodized, overlapping) tiles now warns: the
  halos form a partition of unity that holds only at the finest level, so per-part
  coarsening is approximate at coarse levels — `additive` drops the low-amplitude
  halo splats (overlap dims to a seam), `substitutive` merges them per-part
  (overlap smears). `--tiling content` (disjoint core-keep parts) stays exact.
  The warning fires from all three entry points (`fit --recipe`,
  `batch-fit submit --merge-recipe`, `batch-fit merge --recipe`).
- **`cal --fit-exponent` / `--exponent-scales`** — measures the saturation
  exponent α in `K ~ features^α` (instead of assuming 0.44) by regressing K\* over
  several region scales; the fitted α flows into `fit --tiling content` budgets.
- **`batch-fit submit --tiling content`** — the cluster sibling of
  `fit --tiling content`: one shared content-balanced box plan fanned across the
  Slurm array (`--plan-timepoint`, density knobs), merged into a `kind=partition`.
  The shared plan is now built from a **temporal max-projection** over up to
  `--plan-samples` (default 16) evenly-spaced timepoints, so boxes cover any
  region with signal at _any_ timepoint — fixing silent spatial holes where
  content moved over time and a single representative timepoint missed it.
  `--plan-timepoint` still pins a single timepoint when desired and is now
  range-checked at submit.

#### Fixed — review follow-ups (per-part-LOD-on-uniform warning, content 4D holes, cal exponent validation)

- Broadened the uniform per-part-LOD warning to cover `additive` (not just
  `substitutive`) — both break the halo partition-of-unity at coarse levels.
- `batch-fit submit --tiling content` plans from a capped temporal max-projection
  (see above) instead of one timepoint, and range-checks `--plan-timepoint`.
- `cal --fit-exponent` validates `--exponent-scales` (positive, deduplicated,
  dropped when larger than the volume, ≥2 distinct required, friendly parse
  errors) and the log-log regression now rejects a near-zero feature-count spread
  (ill-conditioned slope) rather than emitting a garbage exponent.

#### Fixed — Large LOD scenes exhausted the browser (unbounded chunk fetches)

A `kind=lod` scene whose finest level holds many millions of points (e.g. a
10M-cell UMAP, ~30M elements across stacked colorings) flooded the browser with
`net::ERR_INSUFFICIENT_RESOURCES`: zarrita's `get` fans out one `fetch()` per
chunk via an internal `Promise.all`, so a single visible range over the finest
level fired thousands of simultaneous requests. Added a shared
bounded-concurrency gate (`utils/fetch-concurrency.ts`, cap 64) funnelling both
data-fetch paths — the multi-level caching store's network tier
(`cache/multi-level-caching-store/fetch-retry.ts`, the default) and the
no-cache `FetchStore` (`data/zarr.ts`). HTTP/2 multiplexes happily at this
width, so throughput is unchanged while the browser's socket/memory budget is
respected. The retry path starts its per-attempt timeout _inside_ the gate (once
a slot is acquired), so time spent waiting in the concurrency queue is not
charged against the fetch budget — otherwise the tail of a large queued
selection would spuriously time out and, at scale, burn the retry budget into
dropped chunks. A 30M-element scene that previously error-stormed now loads
cleanly.

#### Added — Barrier-aware substitutive LOD (`coarsen_dims`)

Substitutive-LOD coarsening can now be restricted to a subset of dimensions; the
remaining dims become **hard grouping barriers** that coarse Gaussian splats never
merge across. This fixes blended/averaged coarse splats when slices are stacked
along a categorical / timepoint / channel axis (e.g. a `coloring` selector). The
engine partitions splats by their barrier-dim value and runs the **unchanged**
per-group reduction, so the merge stays barrier-pure.

- Engine: `make_substitutive_lod(..., coarsen_dims=)` and `make_lod_pyramid(...,
coarsen_dims=)` (`gsplats/lod/substitutive.py`, `pyramid.py`). `None` = coarsen
  all dims (historical behavior); passing every dim normalises to `None`.
- Scene API: `add_points`/`add_lines`/`add_gsplats_from_data(substitutive_lod=
dict(coarsen_dims=...))` accepts dim **names** or column indices, the sentinel
  `"display"`, or `"all"`. **Default is Auto** — coarsen the scene's _displayed_
  dims, group by the _non-displayed_ dims — so nD scenes are correct automatically
  (pure-3D scenes are unchanged: no barrier → identical output).
- CLI: `luxar gsplat lod ... --coarsen-dims i,j,k` (indices; standalone gsplats
  carry no display metadata, so the CLI takes explicit indices and warns on >3D
  input without the flag) across the substitutive/pyramid/multiscale recipes.
- Implied floor: the coarsest level has ≥ one splat per barrier group.

#### Fixed — Layers panel "Active level" readout was stale and off-by-one

The Layers panel's **Active level** status only refreshed inside
`renderControls()` (fired on layer-state changes), so under `auto` mode it went
stale as the camera moved while the per-frame selector
(`LODGroupRegistry.evaluatePerFrame`) swapped levels — disagreeing with the
data-monitor chip, which polls the live level. It also showed the raw 0-based
child index (`rendering: 2`) while the monitor showed 1-based (`L3/5`). Fixed in
`ui/layers/layers-panel.ts`: a lightweight per-frame callback
(`layers-lod-status`, registered in `buildPanel`, torn down in `clear`) keeps the
readout live (string-compare gated — no DOM write unless the level changed), a
shared `computeLodStatusText()` helper feeds both the per-frame path and
`renderControls()`, and all three LOD surfaces (readout, dropdown labels,
data-monitor chip) now use 1-based `L{i}/{n}` numbering (dropdown `value`s stay
0-based for the `lockLevel` API). Broadcast partitions show the first nested
group's live level as `L{i}/{n} · {N} groups`.

#### Fixed — Substitutive LOD pops to coarse when the camera enters the bounding box

The viewer selects which substitutive LOD level to show from the screen-space
pixel-diagonal of the group's bounding box (`projectBoxDiagonalPx` in
`scene/lod-group-registry.ts`). It projected the 8 corners with an unguarded
perspective divide, so when the camera was inside or straddling the box (any
corner at/behind the near plane, clip `w ≤ 0`) the NDC flipped/exploded and the
diagonal **collapsed** — dropping to a _coarse_ level exactly on close approach,
the inverse of the intended behaviour. The projection is now `w`-aware and
**saturates to `+Infinity`** (→ finest level) when any corner has `w ≤ 1e-6`,
reusing the per-frame `projectionMatrix × matrixWorldInverse` product. Also:

- Added a behind-camera reject (`uIsOrtho == 0 && view-z ≥ 0` → off-screen) to
  the **Points** vertex shaders — visual _and_ picking, GLSL _and_ TSL — matching
  the existing gsplat guard; the sprite-quad expansion multiplies by
  `projCenter.w` (`≤ 0` behind the camera) and could otherwise emit a
  degenerate/flipped sprite (and spurious pick hits). The generated-TSL codegen
  snapshots pin the guard; a perspective behind-camera parity case was added.
- Hardened `coarsestReadyIndex` (warn-once instead of per-frame) and documented
  the multi-level downgrade behaviour of `pickChildWithHysteresis`.

#### Fixed — LOD threshold-derivation robustness on degenerate input

Hardened the `min_pixel_size` derivation against degenerate LOD ladders so a
malformed/degenerate input yields a usable result (or a clear error) instead of
a cryptic crash:

- `_apply_monotonicity_guard` now falls back to a small absolute floor when the
  previous threshold is `0` (the relative ×1.1 bump is a no-op at `0`). A
  zero-count _intermediate_ level — or a zero-extent level whose neighbour
  derives to `0` — previously tripped the strict-ascending assertion and aborted
  the build; the guard is now _total_ (never raises).
- `derive_min_pixel_sizes`' empty-coarsest error is now actionable (names the
  likely cause: a substitutive reduction that culled every representative, e.g.
  all-non-positive input amplitudes) instead of "coarsest child must have at
  least 1 element".
- The viewer's LOD child-order check is now strict (`<`, was `<=`), matching the
  Python writer's `_assert_strict_ascending`: _equal_ adjacent `min_pixel_size`
  thresholds (a zero-width hysteresis band) now surface the same producer-bug
  warning rather than passing silently.

These paths are not reachable from a normal `fit → lod` workflow (a real fit
produces positive amplitudes; the builders clamp level depth) — confirmed
empirically — but the hardening turns hand-authored / externally-produced
degenerate `.gsplats.zarr` inputs from crashes into graceful handling. Also
documented (in `adders/points.py` / `lines.py`) why the Points/Lines node-extent
`W` (finest-level bbox) equals the gsplat path's union-over-levels bbox exactly
— so the apparent asymmetry is not "fixed" into a behavior-neutral churn.

#### Fixed — Layers panel now follows scene add-order (napari-style), not alphabetical

The viewer rebuilds the scene graph from zarr **consolidated metadata**, whose
group enumeration is alphabetical — so the Layers panel listed layers
alphabetically regardless of the order they were added in Python. In the LOD
recipe-gallery demo this made the panel (`additive, flat, mosaic, multiscale,
partitioned`) disagree with the left→right spatial placement and the numbered
overlay legend (both `flat → additive → partitioned → multiscale → mosaic`).

Each `Node` now stamps its insertion order among siblings as a `child_index`
attr on add, and the loader (`build-scene-graph.ts`) sorts every sibling list by
it — restoring napari-style add-order across the scene graph, Layers panel, and
any order-sensitive consumer. Siblings without `child_index` (legacy data) keep
their relative enumeration order. As a bonus this fixes a latent misordering of
≥10 `part_<i>`/`child_<i>` subgroups (alphabetical put `part_10` before `part_2`).

`child_index` is stamped on **both** scene-authored nodes (`Node.__init__`) and
the bare-root standalone `.gsplats.zarr` writer (`gsplat_tree.write_gsplat_node`),
so grafted and standalone trees order identically. The finalize back-fill that
resolves a kind=lod group's `display_type` from its finest child now selects by
`child_index` rather than alphabetical name order (name-sort mis-picked the
finest for ≥10-level ladders).

#### Changed — LOD switching thresholds are now extent-based (physically anchored)

The substitutive-LOD `min_pixel_size` selector thresholds — the on-screen sizes
at which the viewer swaps levels — were derived purely from element _counts_
(`base_pixel_size · √(nᵢ/n₀)`). That scene-relative proxy is biased for
substitutive levels (a coarse level has _fewer but larger_ elements), so it
switched at the wrong zoom and needed per-dataset `base_pixel_size` tuning.

The derivation is now **selectable** with a new default. `lod_method="extent"`
(default) anchors each threshold in physical element size — mipmap-style
`threshold_i = T·W/rᵢ` (W = node world-bbox diagonal, `rᵢ` = the level's
element radius, T = a ~1.5 px target). Because it is anchored in pixels it is
**self-calibrating** (no per-dataset tuning) and captures the substitutive
"larger coarse elements" effect that counts cannot. The element radius is an
anisotropy-aware p90 of the splat semi-axes (`GSplatData.principal_radii`), the
point `radii`, or the line `widths` — symmetric across all three geometries.
`lod_method="count"` keeps the legacy √N method. New knobs `extent_percentile`
(90), `extent_anisotropy` (True) and the re-anchored `base_pixel_size` (target px
in extent mode) are exposed on the `lod_group=`/`substitutive_lod=` Python specs,
`RecipeParams`, and the CLI (`gsplat lod --lod-method/--extent-percentile/
--extent-anisotropy`, multiscale). The viewer is unchanged (it reads the authored
`min_pixel_size`). The render-measured error factor and empirical SSE calibration
remain future work.

#### Changed — `gsplat lod` is now a single `--recipe` command

`luxar gsplat lod` is now one command driven by a required `--recipe` flag
instead of three subcommands. Recipes are scale-ordered: **flat**, **additive**,
**partitioned**, **multiscale**, **mosaic**, plus the **substitutive** and
**pyramid** primitives (which absorb the former `lod substitutive` / `lod pyramid`
subcommands; `lod additive` becomes `--recipe additive`). Three topologies are new
and were previously unbuildable from the CLI:

- **partitioned** — a spatial BSP `kind=partition` where _each part carries its
  own additive ladder_ (the old `partition` collapsed parts to a single level).
- **multiscale** — an unbalanced-by-design `kind=lod`: a single coarse
  substitutive cap for the far view above a `partitioned` fine branch, so detail
  structure exists only where you look closely.
- **mosaic** — a spatial BSP `kind=partition` where _each part is its own
  substitutive lod group_ (per-part coarse↔fine replacement): every cell
  frustum-culls AND picks its own level by its own on-screen size — locally
  adaptive detail, the per-part-substitutive sibling of `partitioned`.

The recipe builders are pure functions in `luxar.gsplats.lod.recipes`
(`build_recipe`); the CLI wrapper lives in `luxar/cli/lod.py` (keeping the
already-large `gsplat_commands.py` from growing). Options irrelevant to the
chosen recipe are rejected with a clear error. The `.gsplats.zarr` format and the
underlying `make_additive_lod` / `make_substitutive_lod` / `make_lod_pyramid`
Python builders are unchanged; output stays a standalone v3.0 `.gsplats.zarr` to
graft into a scene via `add_gsplats_from_file` / `gsplat convert`.

The niche `lod additive --substitutive-level N` flag (build an additive ladder on
one substitutive level of an existing pyramid) is dropped from the CLI — recipes
take a fitted/flat input. The capability remains in the Python API
(`make_additive_lod(..., substitutive_level=N)`).

Flag-name note for scripted users: under `--recipe`, `-m`/`--method` is the
**additive ordering** method (greedy/self_energy/…); the **substitutive
algorithm** (kmeans_lloyd/greedy/…) — formerly `lod substitutive -m`/`--method`
— is now the long-only `--substitutive-method` for `--recipe substitutive` /
`pyramid`.

#### Fixed — grafted `multiscale` LOD was stuck on its fine branch

Grafting a `multiscale` recipe into a scene (`add_gsplats_from_file` /
`gsplat convert`) dropped the per-child `min_pixel_size` selector threshold on
the `kind=partition` fine branch of the `kind=lod` group. The viewer reads an
absent threshold as `0` — identical to the coarse cap's `0` — so the LOD
selector always picked the finest eligible child and never switched to the
coarse cap (the embryo stayed stuck on the fine partition at every zoom). The
scene-graft path now stamps the threshold on the lod/partition **wrapper** group
(matching the standalone writer `gsplat_tree.write_gsplat_node`), so the coarse
far-view cap and the fine near-view branch switch as designed.

#### Fixed — viewer now streams nested-group LOD children (multiscale fine branch)

The viewer's lazy-loader (`load-lod-group-node.ts`) only deferred **leaf**
(gsplats/points/lines) lod-group children; a nested `kind=lod` / `kind=partition`
child loaded eagerly at scene-init. So `multiscale`'s fine `kind=partition`
branch was fully resident even while the coarse cap was the visible level,
contradicting its "detail only where you look closely" design. Such non-leaf
children are now cheap-attached as a transparent placeholder and their subtree
loads lazily on first activation (the selector only needs the child's
`min_pixel_size` + `position_bounds`, not geometry). Geometry-agnostic — a
partition/lod nesting of points or lines defers identically to gsplats. (Once
loaded, grouped subtrees stay resident until scene teardown — they have no
leaf-style evictable buffer pool yet.) As defense-in-depth, `loadLodGroupNode`
now validates that a group's child `min_pixel_size` thresholds are ascending
(coarsest→finest) — the selector's monotonic assumption — and gracefully
re-sorts + warns if a malformed / hand-authored scene violates it, rather than
silently mis-selecting levels.

Two follow-ups make the switch actually _visible_: grafted `kind=partition`
wrappers are now back-filled with `position_bounds` at scene finalization (the
graft, unlike the standalone writer, didn't compute the children union — needed
for partition-unit frustum culling), and `multiscale` gained a `base_pixel_size`
anchor (`RecipeParams.base_pixel_size` / the new `gsplat lod --base-pixel-size`
flag). The coarse cap is a _substitutive_ level — fewer but larger splats — so the
count-derived threshold (~10 px) switched too early and left the fine branch
eligible at every practical zoom; raising the anchor (e.g. `200`) pushes the
coarse cap across a wider/farther zoom range so it is actually seen.

#### Changed — scenes use the canonical `.luxar.zarr` extension

Full Luxar **scenes** now adopt a self-identifying `.luxar.zarr` extension
(previously the bare `.zarr`, which is indistinguishable from generic / OME-Zarr
stores). Standalone gsplat files are **unchanged** (`.gsplats.zarr`). The scene
compiler (`LuxarZarrCompiler`) auto-normalizes its output path —
`foo` → `foo.luxar.zarr`, `foo.zarr` → `foo.luxar.zarr`, `foo.luxar.zarr`
unchanged — and reports the final path via `store_path`; the CLI `luxar demo`
default output and `luxar export`/`serve` examples follow suit. Reading is
unaffected: format detection is attribute-based and every path check matches the
`.zarr` suffix, so plain `.zarr` scenes still load. A new shared helper
`luxar.utils.paths.normalize_zarr_path` enforces the canonical suffix for both
scenes (`.luxar.zarr`) and standalone gsplats (`.gsplats.zarr`).

### May 2026

#### Changed — `.gsplats.zarr` format v3.0 (node-tree, unified with the scene)

The standalone `.gsplats.zarr` is now a **detached scene-node subtree** — the
exact same thing a Luxar scene already contains for a gsplats node — rather than
a bespoke 2-D `substitutive × additive` matrix. Embedding into a scene becomes a
**graft** (in the identity case, a byte copy) instead of a lowering, the viewer
opens a standalone file **directly** (`?src=<file>.gsplats.zarr`), and any
combination of additive LOD / substitutive LOD / partition is expressed by
nesting three primitives. This is a hard cutover to **format v3.0**; convert any
legacy file (v1.0 single-LOD, v1.1 multi-additive, the pre-v2.0 substitutive
directory, or an interim v2.0 matrix) with `luxar gsplat migrate-format <in>
<out>`. Luxar is pre-1.0; no historical reading path is retained outside the
migrate tool.

**On-disk grammar (the node tree).** The file root **is** the node. Three
nestable primitives:

- **leaf** (`type=gsplats`) — a single splat set writes `centers` / `amplitudes`
  / `cholesky_factors` / `colors` directly; an additive ladder writes
  `additive_<i>/` subgroups + `n_additive_sublods`.
- **lod group** (`type=group, kind=lod`) — substitutive levels as `child_<i>/`
  (coarsest→finest on disk) each with a `min_pixel_size` selector threshold.
- **partition group** (`type=group, kind=partition`) — spatial parts as
  `part_<i>/`, each carrying its own `position_bounds`; `max_elements`.

Every node carries a `position_bounds` attr (groups = union of children) so the
viewer frames a bare-node file on load. Root header: `format_version:"3.0"`,
`format_type:"gsplats_zarr"`, `timestamp`, `luxar_gsplats_version`.

**One writer.** Scene and standalone gsplats now share a single root-agnostic
serializer (`io/_compiler/gsplat_tree.write_gsplat_node`) layered on the same
`gsplat_assembly` leaf functions the scene compiler uses, so a scene leaf /
additive ladder / kind=lod / kind=partition subtree is **byte-identical** to a
standalone one (locked by parity tests). `LuxarZarrCompiler.write_gsplats_multi_lod`
(the duplicate additive-ladder writer) is deleted; the scene additive-ladder
write flows through the shared walker via `write_gsplat_leaf_subtree`.

**Python model.** `GSplatData` bridges to a tree of
`GSplatLeaf` / `GSplatLodGroup` / `GSplatPartition` (`.tree` / `from_tree`).
`GSplatLOD` → `AdditiveSubLOD`; `SubstitutiveLevel` holds per-level metadata;
accessors `additive_sublods`, `n_additive_sublods`, `n_substitutive`,
`at_substitutive(s)`, `cell(s, a)`, `from_substitutive_levels(...)`,
`to_spatial_partition(...)`. `filter()` / `cull` apply per substitutive level
(the full pyramid is preserved, never silently flattened).

**CLI.** `luxar gsplat lod substitutive` / `pyramid` / `additive` write v3.0
trees; `migrate-format` converts any legacy layout to v3.0. `luxar gsplat
partition` now produces a single `kind=partition` file via a spatial BSP
(`--parts` / `--max-elements` / `--rule`); the index-based `--indices` flag is
**removed**. `luxar gsplat view` serves the node tree directly to the viewer (no
scene round-trip) so partition / nested files open framed. `luxar gsplat info`
reports the tree shape for kind=lod / kind=partition / nested roots.

**Viewer.** A bare gsplats node (leaf / kind=lod / kind=partition) loads as a
scene root: `buildSceneGraph` derives the root `SceneNode.type`/attrs from the
real root `.zattrs`, and `load-scene` frames on the root `position_bounds` (with
a `center_bounds`/union fallback). A `format_type==='gsplats_zarr' &&
format_version!=='3.0'` file surfaces a migrate-format toast. `GSplatsMetadata`
gains an optional `position_bounds`.

#### Removed — Per-package `SPECIFICATIONS.md` files (2026-05-19)

Deleted every `SPECIFICATIONS*.md` across the repository — per-package
specs, the `docs/templates/SPECIFICATIONS_TEMPLATE.md`, and stragglers
under `gsplats/models/gsplats/cuda/`. The files had drifted from the
code and were not pulling their weight. References were cleaned from
`CLAUDE.md`, `AGENTS.md`, all package READMEs, `scripts/check_documentation.py`,
`scripts/README.md`, `packages/luxar-viewer/CONVENTIONS.md`, the
gsplats `GLOSSARY.md`, `docs/concepts/architecture.rst`,
`docs/guides/user/HDR_GUIDE.md`, `docs/guides/developer/BUILD_SYSTEM_SPEC.md`,
`docs/specs/GSPLATS_ZARR_FORMAT.md`, and inline code comments. READMEs
remain the canonical per-package documentation; `docs/guides/specs/` is
unaffected.

#### Changed — Production default renderer flipped back to WebGL (2026-05-16)

The viewer's production rendering path now defaults to
`THREE.WebGLRenderer` (GLSL `ShaderMaterial`) again. Per-scene
performance measurements on the WebGPU path landed below the WebGL
baseline, so WebGL stays the safe choice until those gaps close.
`WebGPURenderer` (TSL `NodeMaterial`) remains a fully-supported
second backend behind `?renderer=webgpu` URL flag or
`VITE_LUXAR_USE_WEBGPU=1` env var; the TSL ↔ GLSL parity harness
keeps both stacks in sync and per-shader GLSL3 sources are retained
as the reference.

The compatibility `VITE_LUXAR_USE_LEGACY_WEBGL=1` env var is now a
no-op alias (WebGL is the default), and
`VITE_LUXAR_USE_WEBGPU_RENDERER=1` is accepted as a synonym for
`VITE_LUXAR_USE_WEBGPU=1` so existing CI invocations keep working.

#### Added — WebGPURenderer WebGL-backend diagnostic flag (2026-05-16)

- Added `?webgpu-force-webgl` for line-rendering performance triage.
  When combined with `?renderer=webgpu`, Luxar still constructs
  Three.js `WebGPURenderer` and dispatches TSL `NodeMaterial` shaders,
  but passes `{ forceWebGL: true }` so Three.js uses its internal
  WebGL2 backend instead of a native WebGPU adapter. This isolates
  TSL/generated-shader overhead from native WebGPU/Dawn/backend costs.

#### Fixed — Multi-agent review fixes for WebGPU/r184 renderer work (2026-05-16)

Landed a batch of correctness, performance, and architecture fixes
surfaced by a multi-agent review of the dual-stack rendering branch.
The batch ships with full unit/lint/type/layer checks green.

- **WebGPU readback row padding.** Added
  `compactWebGPUReadbackRows` to `hdr-pixel-utils.ts` and wired it
  into `PostProcessingManager.readTarget` (RGBA16F + RGBA32F),
  `PostProcessingManager.renderToImageData` (RGBA8), and
  `PickingSystem.readbackAndVote`. Previously the post-processing
  capture and screenshot paths assumed compact rows under WebGPU
  and produced corrupt / slanted output whenever
  `width × bytesPerTexel` wasn't a multiple of 256 (the WebGPU
  spec-mandated `bytesPerRow` alignment).
- **TSL HDR/LDR capture toggles.** `MegaShaderTSLMaterial.toggleRawHdrCapture`
  / `toggleLinearLdrCapture` now flip state and rebuild the TSL
  graph, threading `captureRawHDR` / `captureLinearLDR` flags into
  `megaWebGPUFactory` (which already had the early-exit support).
  EXR exports under WebGPU now produce the documented
  `hdr-effects-pre-tone` and `visible-ldr` outputs instead of
  silently falling through to the full pipeline.
- **GSplat picking `nearFade`.** Changed from
  `depthFade.mul(coverageFade)` to `min(depthFade, coverageFade)`
  in `gsplat-pick.tsl.ts` — matches visual TSL/GLSL gsplat and
  GLSL gsplat picking. Restores correct splat picking near
  coverage limits.
- **`material-manager.ts` ↔ picking-material cycles.** Removed the
  back-import of the `materialManager` singleton from all six
  picking-material classes and the manual `unregister(this)` calls
  in their `dispose()` methods. Cleanup now flows through the
  existing `subscribeToDispose` listener that `MaterialManager.register`
  attaches. `pnpm run check:layers` now reports 0 violations (was 6).
- **Picking sharpness/radius sanitization parity.** Point picking
  shaders (GLSL + TSL) now use the same `sanitizePositive` /
  `sanitizeNonNegative` helpers as the visual path, so malformed
  NaN/Inf sharpness can no longer make the pick footprint diverge
  from the visible footprint.
- **GSplat TSL invalid-value guards.** `gsplat.tsl.ts` and
  `gsplat-pick.tsl.ts` now reject NaN/Inf 2D covariance elements
  and amplitudes before the Cholesky / eigendecomposition,
  matching the GLSL `invalidCov2D || invalidFloat` guard. New
  `invalidFloatTSL` helper in `tsl-helpers.ts`.
- **`RendererCapabilities.api` semantics clarified.** JSDoc now
  states explicitly that `api` reports the **renderer API surface**
  (which signatures to call), not the physical GPU backend.
  Exported `isWebGLRenderer` and added a symmetric `isWebGPURenderer`
  type guard. `BROWSER_SUPPORT_POLICY.md` rewrites the
  previously-contradicting "honestly reports the backend"
  paragraph.
- **WebGPU device-loss signal.** `SceneManager.setupContextLossHandling`
  now observes `renderer.backend.device.lost` and dispatches a
  `webgpu-device-lost` event so host applications can prompt for
  a reload. WebGPU device loss is treated as **unrecoverable** in
  this release; a full rebuild path mirroring WebGL2's
  `WebGLContextRecovery` is deferred until there's
  WebGPU-native test infrastructure.
- **GSplat TSL cofactor gating.** `gsplatWebGPUFactory` now
  JS-conditionally emits the Σ_cam⁻¹ cofactor / ray-integration
  block only in sum-projection mode. TSL `.select()` does not
  short-circuit, so the previous factory paid the ~25-op cofactor
  expansion on every vertex even in max projection. The wrapper
  rebuilds the graph on sum↔max boundary crossings (same one-time
  cost as a bloom/vignette toggle).
- **GSplat picking Mahalanobis dedup.** Materialised the per-fragment
  Mahalanobis forward-substitution + intensity via `.toVar()`
  outside both `colorNode` and `depthNode` Fn bodies so the TSL
  builder can fold the shared subexpression into a single local
  if it supports CSE. Worst case is equivalent to before.
- **Docs refresh.** Rendering docs and `picking/PICKING_DESIGN.md`
  updated to describe the shipped dual-stack pipeline, including renderer
  dispatch, readback signatures, WebGPU row-padding, interleaved attributes,
  and the device-loss policy. Line cap-factor pseudocode rewritten to
  match the fragment-shader implementation (the original
  vertex-side `vCapFactor` design didn't survive the
  4-vertex-quad layout).

#### Changed — WebGPU renderer work: TSL ports + point container update (2026-05-13)

Infrastructure step toward the WebGPU rendering backend. What
landed at this commit (production rendering path was still on
`WebGLRenderer` here):

- **All 12 shaders ported to TSL / NodeMaterial**: scene materials
  (`point`, `line`, `gsplat`), picking variants (`point-pick`,
  `line-pick`, `gsplat-pick`), and post-processing (`fxaa`,
  `bloom-{threshold,downsample,upsample}`, `mega` including
  detector noise). Each lives in a `*.tsl.ts` file alongside the
  GLSL3 source, sharing the same `ShaderSource` registry and
  blending-state helper. Pixel parity vs. GLSL3 verified by
  `tsl-shader-parity.spec.ts` under
  `WebGPURenderer({ forceWebGL: true })`.
- **Point container update**: points are now rendered as a
  `THREE.Mesh + InstancedBufferGeometry` (matching the existing
  line/gsplat container shape) instead of `THREE.Points`. Required
  because r184's `GLSLNodeBuilder` hardcodes `gl_PointSize = 1.0`
  for `THREE.Points`, blocking the TSL point port. Per-instance
  attributes renamed to a shared convention (`aCenter`, `aRadius`,
  `aSharpness`, `aColor`, `aScalar`).
- **Async picking readback**: `picking-system.ts::readbackAndVote`
  is now `async` and uses `readRenderTargetPixelsAsync` (works on
  both WebGLRenderer and WebGPURenderer in r184). Stale-tooltip
  suppression added in `core/app.ts` so an in-flight readback
  doesn't blank the tooltip prematurely.
- **`renderToImageData` readback path**: capture now renders through an
  offscreen `WebGLRenderTarget` and reads via
  `readRenderTargetPixelsAsync`. Backbuffer readback fallback is
  retained on `RendererCapabilities` for WebGL2-only tests.
- **Renderer union widening**: `RendererCapabilities.Renderer` is
  now `WebGLRenderer | WebGPURenderer`; `setupRenderer` is
  `async` and selects between the two via
  `VITE_LUXAR_USE_WEBGPU_RENDERER=1`.
- **Shared TSL helpers**: new `tsl-helpers.ts` deduplicates the
  `sanitizePositive` / `sanitizeNonNegative` / `TSLNode` definitions
  that the per-shader files had each carried locally. Each visual
  TSL factory now accepts a `blendingMode` config field and applies
  the matching THREE blending state via the existing
  `blending-state.ts` helper.

#### Changed — Three.js r184 and custom post-processing pipeline (2026-05-12)

- **Breaking viewer dependency change**: the TypeScript viewer now targets
  `three@~0.184.x` and removes the `postprocessing` package dependency.
- Replaced the old pmndrs `EffectComposer` chain with Luxar's custom
  post-processing pipeline: scene → HDR half-float target → BloomChain →
  fused mega-shader → optional FXAA.
- Preserved core effects in the new pipeline: bloom, global exposure/offset/gamma,
  tone mapping, detector noise, vignette, chromatic lens distortion, FXAA, MSAA,
  and SSAA.
- Removed SMAA, Depth of Field, and SSAO controls. SMAA's 3-pass algorithm does
  not fit the fused pipeline; DoF needs depth-aware multi-pass blur; SSAO requires
  surface normals that Luxar's point/line/gsplat primitives do not provide.
- Restored HDR/EXR capture semantics with explicit modes (`hdr-effects-pre-tone`,
  `visible-ldr`, and `raw-scene-hdr`) and browser shader smoke coverage.

#### Changed — Cache final polish after S1–S7 (2026-05-10)

- Added browser screenshot artifact coverage for the Cache tab's status
  badge / Cache Health layout in `cache-hardening.spec.ts`.
- Cache disabled/fallback views now render any available cache-status
  badges (for example `no-cache`, `disabled-config`, or
  `provider-missing`) above the explanatory panel instead of only
  showing badges in the full L1/L2 view.
- SceneLoader clears the per-node predictive-prefetch baseline when a
  loader update throws, so the next successful update re-baselines
  rather than extrapolating across a stale/error gap.

#### Changed — Cache recheck polish S1–S7 (2026-05-10)

Follow-up cache polish for UI styling, predictive prefetch behavior,
cache metrics cleanup, and additional edge-case coverage.

- **S1 — CSS for the new cache UI classes.** `data-loading-monitor.css`
  gains rules for `.luxar-cache-section__metrics--cols-4` (the L2
  ERRORS card row, with a 2-column fallback under 480px),
  `.luxar-cache-status` (badge pill row), `.luxar-badge` (pill shape,
  reuses `.luxar-color--*` text-color modifiers for tint), and
  `.luxar-cache-health` / `.luxar-cache-health__{header,row,label,value}`
  (validation-mode + last-validated panel).
- **S2 — `opfs-unavailable` badge is wired and no longer dead.**
  `OPFSStore.getStats()` exposes `available: boolean` (true when
  `opfsRoot !== null && !disposed`). `MultiLevelCachingStore.getStats().health`
  surfaces `opfsAvailable: boolean`, treating caching-disabled modes
  as `true` (no L2 expected). `aggregateCacheMetrics` emits the badge
  only when explicitly `false`. Older providers that omit the field
  stay silent (treat-as-true).
- **S3 — L0/L1/L2 hit-rate no-data color is dimmed (not red).** New
  `getCacheHitRateColorClassWithGuard(rate, totalAccesses)` returns
  dimmed when `totalAccesses === 0`; otherwise delegates to the
  existing thresholds. Used by both the initial render in
  `renderCacheContent` and (for L0/L1 parity) the incremental
  `updateCacheTab`. Fixes the first-paint red-flash on empty caches.
- **S4 — `clearOnInitCount` telemetry + a real `?clear-cache` E2E
  assertion.** `MultiLevelCachingStore` counts each `?clear-cache`
  invocation in `init()`. Surfaced via `getStats().clearOnInitCount`
  and the cache-API snapshot. `cache-persistence.spec.ts` now asserts
  `clearOnInitCount > 0` after navigating with `?clear-cache` —
  proves the clear path executed, replacing the trivially-true
  `l2.size >= 0` assertion.
- **S5 — Bandwidth compaction test now exercises the production
  path.** The R5 test stub (seeded stale entries, never asserted
  compaction) now drives a real `getResult()` fetch so the
  production push site's `start > length/2` check fires; asserts
  `bandwidthWindowStart` resets to 0 and the array shrinks.
- **S6 — Predictive prefetch uses per-loader derived view-state.**
  `SceneLoader._dispatchPredictivePrefetch` no longer fans one
  global view-state to every loader. The dispatch now lives inside
  each loader-task branch (Points / Lines / GSplats) and uses the
  per-node `derived.viewState` computed by `deriveNodeViewState`.
  `extend_to_all`-skipped nodes no longer receive prefetch hints
  they would have skipped on demand. A per-loader `Map<path,
ViewState>` tracks prev state; cleared on `loadScene` / `dispose`
  and on skip transitions.
- **S7 — `evictionsPerMin` API break recorded.** As part of the R1–R7
  batch (commit `2f7feaaf`), the deprecated `CacheMetrics.evictionsPerMin`
  alias was removed in favor of `evictionsTotal`. External consumers
  reading `evictionsPerMin` now see `undefined`; this CHANGELOG note
  records the removal explicitly so the API break is discoverable.

**Result**: cache UI is fully styled, every documented status badge
is actually emitted, the predictor honors per-node tolerance/skip
semantics, the `?clear-cache` E2E proves real observable behavior,
and the bandwidth-compaction test exercises the real production
path. Tests: 89 cache unit + 86 monitor unit + 1 strengthened E2E.

#### Changed — Cache re-review remediation R1–R7 (2026-05-10)

Cache hardening pass for lifecycle/dispose, OPFS races, in-flight
coalescing, content-hash/TTL validation, Points/Lines prefetch parity,
UI/observability gaps, and edge-case test coverage.

- **R1 — config validation for new cache fields.**
  `validateConfig` rejects bad `cache.opfsOperationTimeoutMs` (NaN
  / zero / negative / Infinity) and `cache.externalDatasetTtlMs`
  (NaN / negative; `null` stays valid). Prevents cryptic OPFS
  timeouts or always-expired TTLs.
- **R2 — L2 hit-rate now patches live in the cache tab.**
  `updateCacheTab` patches `l2-hitrate` + `l2-hitrate-sub` and the
  color class alongside `l2-size` / `l2-io`. The L2 hit-rate card
  no longer freezes after the initial render.
- **R3 — Status badges, Cache Health, L2 error counters.**
  The cache tab now shows a pill row of `CacheStatusBadge` chips
  (`cache-enabled`, `quota-constrained`,
  `unvalidated-external-dataset`, …), a dedicated **Cache Health**
  section with validation mode + last-validated timestamp, and an
  inline `ERRORS` card on the L2 section summing the four OPFS
  health counters. `CacheMetrics.l2` carries those counters so
  debug snapshots and E2E can assert on them without reaching into
  the provider. `cache/README.md` gains a "Cache Status Badges"
  section.
- **R4 — Direction-aware predictive prefetch wired into
  `updateView`.** `SceneLoader.updateView` extrapolates one step
  from the previous → current view-state delta and fires
  `prefetchChunks(predicted)` on every loader (Points, Lines,
  GSplats) for the predicted state, in a microtask so it never
  blocks commit. Pure helper `predictNextViewState` is unit-tested
  in isolation; `dispatchPredictivePrefetch` handles iteration +
  error swallowing. State resets on dataset switch / dispose.
- **R5 — Bandwidth window start-index pruning.**
  `MultiLevelCachingStore.bandwidthWindow` no longer uses
  `Array.shift()` (O(n)); a `bandwidthWindowStart` index advances
  forward with amortized O(n) compaction when the dead prefix
  exceeds half the array. Numeric output unchanged.
- **R6 — Test gap closure.**
    - **R6a**: Unicode OPFS key roundtrip (µ, 通, é, base64-special
      characters, slash-traversing keys).
    - **R6b**: Demand-while-prefetch-in-flight — one underlying fetch
      shared via `pendingGets`; demand counter increments exactly once;
      L1-warm path lets demand hit L1.
    - **R6c**: 4D node-type cache parity — cache stats populated, slice
      navigation produces fresh L1 activity.
    - **R6d**: OPFS quota-clears-then-write-succeeds + concurrent
      quota-skipped writes don't corrupt the index.
    - **R6e**: Negative `totalSize` in persisted metadata is clamped /
      recomputed from `entries[]` (defensive hardening in
      `OPFSStore.loadMetadata`).
- **R7 — Real-browser L2 persistence E2E.**
  New `cache-persistence.spec.ts` asserts L2 entries survive a full
  page reload in Chromium and that `?clear-cache` wipes L2 across a
  reload. Skipped on Firefox/WebKit (OPFS persistence semantics
  across Playwright contexts are unreliable there).

**Result**: 4813 unit tests pass (up from 4662 — +151 new tests
across R1–R7). 9 cache test files, 272 cache unit tests. Bandwidth
pruning, metadata corruption recovery, and config validation become
regression-locked.

Deliberately deferred (re-review §1.6 "future work"): cross-browser
OPFS comparison harness, sustained-load cache performance benchmarks,
runtime cache profiles, per-array memory breakdowns, origin-wide
cache manager UI.

#### Changed — Viewer code-review recheck hardening (2026-05-10)

Follow-up hardening pass for findings still open after the prior viewer
code review:

- **Constants**: `MAX_SUPPORTED_DIMS` consolidated into
  `src/config/constants.ts`; `wasm/typescript/gsplats-processing.ts`,
  `data/gsplats/gsplats-processor.ts`, and `workers/validation.ts`
  (`MAX_WASM_DIMS` alias) now share the canonical value.
- **Worker init guard**: `projectPointsTo3D` in `workers/data-worker.ts`
  rejects with `NOT_INITIALIZED_MSG` when called before `initialize()`,
  matching the existing `projectLinesTo3D` / `projectGSplatsTo3D`
  guards.
- **Material registration idempotency**:
  `MaterialManager.subscribeToDispose` now tracks already-subscribed
  materials in a `WeakSet` so re-registering the same instance no
  longer stacks `dispose` listeners on the THREE EventDispatcher.
- **Post-processing rebuild safety**:
  `PostProcessingManager` replaces the boolean `deferRebuild` flag with
  a depth counter; `setQualityPreset` wraps its batch in `try/finally`
  so a thrown sub-setter cannot strand the counter above zero (which
  previously made all subsequent effect changes silent no-ops).
- **Shared clamp util**: `clampDPRScale` in
  `rendering/post-processing/visual-effects-handler.ts` calls the
  general `utils/clamp` helper instead of an inline `Math.min(max,
Math.max(min, x))`.
- **Lines TS fallback note**: `data/lines/projection.ts` documents the
  TS fallback path's allocation profile as an accepted trade-off.
- **Docs**: data-loader docs rewritten to show the
  `runWithTimeout(name, kind, fn)` pattern
  instead of raw `getWorker()` access. `CONVENTIONS.md` gains §§12-14
  for dependency-inversion ports, error-handling discipline, and the
  disposal pattern. `packages/luxar-viewer/README.md` documents
  `LUXAR_LAUNCHER_NO_WEBVIEW=1`.

#### Changed — Viewer code-review rerun hardening pass (2026-05-10)

Hardening of the scalar-colormap + GPU pool + blending-state feature
work, addressing actionable findings from a six-agent code review rerun.

**Type-safety:**

- `PointsAttributeTypes.scalar` is now optional (`undefined` when absent)
  instead of a `'none'` sentinel; Float16Array gets a first-class dtype tag.
- `LoadedLinesData.scalars` aligned with `ScalarArray` (Float32/Float16/
  Uint8). Lines accumulator preserves Uint8 dtype natively.
- Fail-closed scalar length validation in `projectPointsTo3D` and
  `buildInstanceBuffers` — mismatch logs a warning + suppresses colormap.
- `growLinesGeometry` preserves optional `aStartScalar`/`aEndScalar`
  attributes on resize.

**Lifecycle:**

- Material clone-vs-pool registration bug: pooled materials are
  detached from global updates before NodeFactory clone sites; clones
  take the global slot, pooled stays in the LRU cache.
- Custom colormap LUT cache: bounded LRU (16 entries), disposed
  per scene unload via `disposeCustomColormapTextures()`. Built-ins
  survive scene switches.
- Post-processing in-place effect swaps (AO/SMAA/Bloom-levels) now
  dispose the OLD effect AFTER `rebuildEffectPass`, eliminating the
  use-after-free window.
- `rebuildEffectPass` rolls back Pass A if Pass B construction fails.
- Data accumulators expose `isDisposed()`; `fill`/`ensureCapacity`
  throw with a descriptive error post-dispose.

**Performance:**

- Lines worker scalar fallback now emits a one-shot warning so users
  notice the main-thread cliff.
- Accumulator growth copies only the live prefix
  (`usedCount * stride`) instead of full capacity.
- Worker-projection fallback routes through the accumulator's
  pre-sized buffers when present.
- Scalar buffers in Points + Lines accumulators are lazily allocated
  on first scalar fill or first `getScalarBuffer()` call.
- GPU byte-budget eviction is single-pass sort + walk (replaces the
  per-iteration O(N²) re-scan). Pure helper `selectBuffersToEvict`
  exposed for tests. Loop bounded by `maxPoolSize * 3` iterations.
- `estimateGeometryBytes` cached on `geometry.userData.cachedByteSize`;
  invalidated on grow.

**Shaders:**

- Line shader: pathological near-camera wide-quad segments now early-
  discard instead of rasterizing a half-viewport quad at reduced
  intensity.
- Shared GLSL sanitize helpers (`isInvalidFloat`, `sanitizePositive`,
  `sanitizeNonNegative`) extracted to `glsl-lib.ts` and injected into
  the Points/Lines/GSplats vertex shaders.
- Documented `LUXAR_MAX_RGB_CONTRIBUTION` and `USE_COLORMAP` defines
  in the line shader header.

**API surface:**

- Public exports for `getCompleteBlendingState`,
  `applyBlendingStateToMaterial`, `supportsScalarColormap`,
  `applyColormapTextureToMaterial`, `applyScalarRangeToMaterial`,
  `BlendingMode`, `CompleteBlendingState`.
- Predicates (`isAdditiveMode`, `isOpaqueMode`, `isMaxMode`, etc.)
  centralize the mode discriminators across materials.
- `syncPointMaterialWithGeometry` moved to `rendering/material-sync-helpers.ts`.

**Diagnostics:**

- Array decoder broadcast-encoding error includes zarr path + encoding shape.
- `captureHDRPixels` validates the mode at runtime, falls back with a warning.
- `updateView` log differentiates supersede vs first-queue.
- Custom LUT cache validates content fingerprint on hit (defends
  against DJB2 collisions).

**Tests + Docs:**

- New `blending-state.test.ts` with predicate + canonical-state tests
  including max-mode round-trip lock-in.
- Accumulator dispose/usedCount tests, scalar buffer lazy-alloc tests,
  Float16 detection tests, Lines Uint8 roundtrip test.
- Byte-budget eviction tests for the pure selector + pool integration.
- `LUXAR_ZARR_FORMAT.md` documents `has_scalars`, `scalar_data_range`,
  `colormap`, and the `colormap_lut` sibling array.
- Rendering docs describe the byte-cache + bounded eviction policy.

#### Changed — Viewer shader/material follow-through (2026-05-10)

Completes the remaining shader/material work in this branch, excluding
conditional Gaussian slicing for correlated nD splats.

- **Custom Python-authored colormaps now display.** When a node's
  metadata declares `colormap='custom'`, the scene loader opens its
  `colormap_lut` zarr array, validates byte length (768 RGB or 1024
  RGBA), and passes the bytes through `getColormapTexture('custom', lut)`
  for Points / Lines / GSplats. Invalid LUTs log a warning and fall
  back to viridis. Custom-LUT cache is disposed on `SceneManager.dispose()`.
- **GPU buffer pool byte-budget eviction.** New `gpuPoolMaxBytes` config
  (default 512 MB). Pooled buffers are evicted (largest-first) once
  total pooled bytes exceed the budget — independent of the count cap.
  `getStats()` now reports activeBytes / pooledBytes / totalBytes /
  largestPooledBytes, surfaced via `__luxarDebug.getState().gpuPool`.
- **EXR export modes.** `captureHDRPixels(mode)` and `captureHDRAsEXR({mode})`
  accept `'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr'`.
  `'raw-scene-hdr'` disables every post-processing effect (incl. bloom/
  DOF/AO). `'hdr-effects-pre-tone'` keeps HDR-space effects but disables
  LDR display effects.
  `'visible-ldr'` keeps everything.
- **Points scalar colormaps end-to-end.** Loader (`loadPoints`) opens
  the `scalars` zarr array when `has_scalars=true`, populates
  `LoadedPointsData.scalars` through every load path; projection
  carries scalars through compaction; accumulator gains a scalar
  buffer (`getScalarBuffer()`); GPU pool detects scalar dtype
  (`'none' | 'Float32Array' | 'Uint8Array'`) and lazily binds the
  `scalar` attribute. The fail-closed guard naturally passes for any
  properly authored scalar dataset.
- **Lines scalar colormaps end-to-end.** Same pattern as Points.
  Loader opens `scalars`; `buildInstanceBuffers` interpolates scalars
  at clipped endpoints exactly like colors/widths/sharpness; WASM
  fallback path mirrors the TS path; GPU pool lazily allocates
  `aStartScalar`/`aEndScalar` instanced attributes; `updateInstancedLinesMesh`
  threads them through the shared `attrSpecs` path.
  Worker projection falls back to main-thread when scalars are
  present (worker payload doesn't carry scalars yet — separate change).
- **Browser-real shader compile + visual smoke tests.** New E2E specs:
  `shader-material-compile.spec.ts` (every material variant compiles +
  produces non-black pixels in a real browser),
  `line-rendering-visual.spec.ts` (cap factor and near-plane safety via
  sampled-pixel assertions), and `gsplat-rendering-visual.spec.ts` (ray
  integral and displayDims order). New helpers `assertNoShaderErrors(page)` and
  `samplePixelAt(page, sel, fx, fy)` in `tests/e2e/helpers.ts`. Visual
  screenshot baselines are deferred (need committed PNGs per
  platform); sampled-pixel assertions are platform-independent.

#### Changed — Viewer shader/material remediation (2026-05-10)

User-visible behavior changes in the Luxar viewer:

- **Point/Line scalar colormaps fail-closed when scalar attributes aren't bound.** Previously, metadata-authored `colormap`+`has_scalars` would activate `USE_COLORMAP` even though the geometry had no `scalar`/`aStartScalar`/`aEndScalar` attribute. Now the viewer logs a warning and falls back to vertex-color rendering.
- **Positions-only Points render visibly.** The GPU buffer pool now fills white/0.5/2.0 defaults for absent color/radius/sharpness arrays instead of leaving zeros (which the fragment shader discards as zero-contribution).
- **Point material parity for max blending.** `PointMaterial.applyBlendingMode` is the new single source of truth; runtime UI mode changes now produce the same `OneFactor/OneFactor` blend factors as creation-time max. The fragment shader gains a `LUXAR_MAX_RGB_CONTRIBUTION` define so `max` mode premultiplies RGB by intensity\*opacity.
- **Normal-mode depth writes.** Fully-opaque (opacity ≥ 0.99) `normal` layers now write depth so additive layers behind them are correctly occluded.
- **Line body intensity reaches 1.0 as documented.** Cap factor moved from vertex to fragment shader.
- **Line near-plane safety.** Lines whose endpoints cross or sit very close to the camera no longer produce screen-filling artifacts.
- **Line bounds include rendered footprint** so thick lines aren't culled prematurely.
- **Line max blending RGB contribution** (same `LUXAR_MAX_RGB_CONTRIBUTION` define as Points).
- **GSplat `displayDims` order is preserved** to match Points/Lines. Previously dims were silently sorted.
- **GSplat sum-projection ray integral uses precision** (`1/sqrt(rᵀΣ⁻¹r)`), not covariance variance (`sqrt(rᵀΣr)`). Fixes physically incorrect brightness for rotated anisotropic splats viewed off-eigenaxis.
- **Conditional Gaussian slicing for correlated nD splats remains marginal+attenuation** (documented inline). Conditional `μ_D|H` / `Σ_D|H` math is scheduled as a follow-up.
- **Disabling a colormap clears uniform state**; clones of disabled materials no longer resurrect the disabled texture.
- **`LineMaterial.clone` copies `uIsOrtho`**.
- **GPU buffer pool is disposed** in `SceneLoader.dispose()`. Dataset switches no longer leak `InstancedBufferGeometry` references.
- **No-op blending changes don't recompile shaders.**
- **AO Quality control is no longer a no-op when AO is already enabled.**
- **Effective AA reporting via `getEffectiveAA(smaa, fxaa)`.**
- **Lines counted in `__luxarDebug.getState()`** (`totalLines` + `lineMeshes`).

Documentation: `E2E_TESTING_GUIDE.md` now references `getBufferedMessages()` (was stale `getMessages()`); `HDR_GUIDE.md` reads tone-mapping/exposure from `renderingControls.settings` rather than the nonexistent `state.rendering`.

#### Added — `luxar gsplat lod substitutive` (PR #109)

- New CLI subcommand `luxar gsplat lod substitutive <in.gsplats.zarr> <out_dir/>`
  that builds a coarse-to-fine _substitutive_ LOD hierarchy: each level
  _replaces_ the previous one with `M = N / K^ℓ` synthesised representative
  splats. Output is a directory of per-level `.gsplats.zarr` files plus a
  `manifest.json`.
- New module `luxar.gsplats.lod.substitutive` with `make_substitutive_lod(data,
*, compression_factor=4, levels=3, method="kmeans_lloyd", ...)`. Three
  partition algorithms supported: `kmeans_lloyd` (k-means warm-start +
  cost-increment Lloyd refinement, the recommended workhorse), `greedy`
  (hierarchical pairwise greedy), and `kmeans` (spatial-only).
- Per-bin merge is a moment-matched single Gaussian with L²-optimal
  amplitude (closed-form per `manuscript/supp_doc/substitutive_lod`).
- New module `luxar.utils.spatial_hash` with `SpatialHashGrid` (online,
  CPU) elevated from `gsplats/seeds/utils.py` and a new GPU-capable
  `BatchedSpatialHashGrid.query_knn` that's behaviourally equivalent to
  `scipy.spatial.cKDTree.query` (covered by
  `TestBatchedNumpy::test_knn_matches_cKDTree`). The old
  `gsplats.seeds.utils.SpatialHashGrid` symbol is re-exported for back-compat.
- Seed-generation paths inside `fit_gaussian_splats`
  (`_subsample_seeds_spatially_diverse`, `_add_grid_fallback_seeds`) now
  use `BatchedSpatialHashGrid.query_knn` instead of `cKDTree.query`.
  Behaviour preserved by the equivalence test; cal smoke test passes
  end-to-end.
- 27 new tests in `packages/luxar/src/luxar/utils/tests/test_spatial_hash.py`
    - ~440 lines in `gsplats/tests/test_substitutive_lod.py`.

#### Changed — Type-ignore tightening + ruff format propagation (PR #108)

- Reverted decorator-targeted `# type: ignore[misc]` ignores on `numba.njit`
  and `torch.jit.ignore` back to bare `# type: ignore` to match what mypy
  actually reports on those decorators across the supported Python
  versions, after a brief over-tightening from PR #107's format sweep.
- Re-applied the `ruff format` sweep across the merged tree
  (`packages/luxar/src/luxar/`) so newly-landed code (calibration,
  additive LOD, progressive fitting demos) follows the same style as the
  rest of the package.

#### Changed — Massive viewer refactor and fixes (PR #107)

- Extensive TypeScript refactor of the viewer (`packages/luxar-viewer/`):
  modularised the `app.ts` lifecycle, decoupled overlay disposal from
  dataset switching, hardened cache-fetch retries and async-cleanup
  observability, tightened `extend_to_all` dimension validation, fixed
  canonical `sharpnesses` loading, and improved WebGL context-restoration
  resource recreation.
- New per-worker docs and tests under `packages/luxar-viewer/src/workers/`
  (SPECIFICATIONS.md, validation.ts, color-utils.ts).
- `LuxarApp.init({ updateBrowserUrl })` default flipped from `true` to
  `false` so embedded callers no longer have host-page URLs silently
  rewritten on dataset selection; the standalone bootstrap (`bootstrap.ts`)
  explicitly opts in.

#### Added — `luxar gsplat lod additive` and progressive-fitting decoupling (PR #106)

- New CLI subcommand `luxar gsplat lod additive <in.gsplats.zarr>
<out.gsplats.zarr>` that _reorders_ the splats of a fitted dataset into
  a multi-LOD `GSplatData` whose prefix sum at any `k` splats is the
  best L² approximation of the full scene. Output is a single multi-LOD
  `.gsplats.zarr` (each level _extends_ the previous one).
- New module `luxar.gsplats.lod.additive` with `make_additive_lod(data,
n_lods=4, method=..., breakpoints=..., ...)` and
  `compute_additive_order(...)`. Four ordering methods supported:
  `greedy` (residual-correlation matching pursuit; default),
  `self_energy` (cheap O(N log N) baseline within 2-10% AUC of greedy
  on real datasets), `mass` (peak-amplitude × covariance volume), and
  `amplitude` (peak height alone). Algorithms documented in
  `manuscript/supp_doc/additive_lod`.
- **Progressive fitting decoupled from LOD construction.**
  `fit_progressive_gaussian_splats` (and the `luxar gsplat fit
--progressive` CLI flag) now returns a single flattened
  `GSplatData` rather than a multi-LOD container. To build an LOD
  ladder, run `luxar gsplat lod additive` on the flat output. Progressive
  multi-pass fitting remains a valid alternative fitting flow — it just
  no longer overloads the LOD concept.
- New tests in `gsplats/tests/test_additive_lod.py`.

#### Added — `luxar gsplat cal` (blind-spot CV calibration) (PR #105)

- New CLI command `luxar gsplat cal <volume> <out.json>` that sweeps splat
  count `K` and reports the recommended `K*` via blind-spot
  cross-validation: 5%-donut-median masking (Noise2Self protocol from
  Batson & Royer 2019), fit at each `K` against the masked volume, evaluate
  PSNR at the held-out positions against the _original_ values. Hybrid
  peak/plateau/signal-limited detection rule from the manuscript's
  `splat_count_vs_quality §4.2`.
- Free byproduct: per-dataset noise-floor estimate via a three-estimator
  ensemble (discrete-Laplacian MAD, Haar HH-subband MAD, background-region
  MAD), giving an absolute PSNR ceiling.
- Configurable K grid: `--k-grid '1000,4000,...'` (explicit) or parametric
  via `--n-grid`, `--k-min`, `--k-max`, `--progression exp|power`,
  `--power`. Volume loader pass-through (`--channel`, `--timepoint`,
  `--array-key`) and full preset/config layering. Optional `--keep-fits`
  persists each per-K `.gsplats.zarr`; optional `--pdf` produces a 3-page
  matplotlib report (rate-distortion, blind-spot CV curves, slice
  montages).
- New module `luxar.gsplats.calibration`: `cv_mask`, `donut_median_fill`,
  `held_out_psnr`, `estimate_noise_floor` (returns `NoiseFloor`),
  `build_k_grid`, `find_k_star` (returns `HeldOutPeak`),
  `CalibrationResult` (JSON round-trip), and the top-level `calibrate`
  driver. New module `luxar.gsplats.calibration_report` for the optional
  PDF.
- Purely additive: no changes to `fit_gaussian_splats`, the optimisation
  loop, the loss module, the metrics module, or `GSplatData`. Held-out
  PSNR is fundamentally a _capacity_-selection criterion (across `K`),
  not an _iteration_-selection one — the existing patience-based early
  stop already covers the within-fit regime.
- Tests: 37 unit tests in `gsplats/tests/test_calibration.py` covering
  mask determinism, donut fill (2D/3D/4D), held-out PSNR, K-grid
  construction, peak detection rule, noise-floor recovery on synthetic
  Gaussian noise, JSON round-trip, and a CPU smoke test of the full
  driver. Plus 4 CLI smoke tests in
  `cli/tests/test_gsplat_cli_extended.py::TestCalibrateCommand`.
- Docs: `gsplats/README.md` Calibration section, Sphinx page
  `docs/api/gsplats.rst`, and CLAUDE.md examples block.

#### Changed — GSplats default device on macOS

- `GaussianSplatModel` (and the gsplat fitting API) now auto-selects MPS on macOS when no explicit device is provided and `use_metal=True` (the default). Pass `use_metal=False` to keep CPU as the default on Macs that prefer it.
- Centralized device selection in `luxar.gsplats.utils.device.resolve_torch_device(...)`; remaining hand-rolled `torch.cuda.is_available()` / `torch.backends.mps.is_available()` ternaries in `utils/demos.py`, `gsplats/multiscale/decompose.py`, `gsplats/seeds/gpu_ops.py`, `gsplats/preprocessing/denoise_pipeline.py`, and `gsplats/fitting/preprocessing.py` (FPS GPU gate) now route through it. The denoise and FPS paths consequently honor MPS where they previously ignored it.

#### Fixed — Encoding, viewer, and GSplats hardening

- Tightened Luxar encoding metadata validation in Python and TypeScript: present `encoding` objects must declare a known `name`, direct dtype names are no longer treated as quantization, `array_ref` targets must be explicit, and malformed LUT/quantization metadata now fails loudly.
- Preserved Python-written integer color arrays (`uint8`/`uint16`) as direct SDR storage in the viewer and added cross-language fixture coverage.
- Hardened viewer loading/cache paths with bounded cache fetch retries, observable async cleanup failures, stricter `extend_to_all` dimension validation, canonical `sharpnesses` loading, and improved WebGL context-restoration resource recreation.
- **Embedder-visible**: `LuxarApp.init({ updateBrowserUrl })` default flipped from `true` to `false` so embedded callers no longer have host-page URL silently rewritten on dataset selection. The standalone bootstrap (`bootstrap.ts`) explicitly opts in. Existing embedded callers that depended on URL reflection must now pass `updateBrowserUrl: true` explicitly. (000bf00a)
- Fixed GSplats batch planning/loading for folded channel-like axes in >5D Zarr arrays, persisting channel-axis metadata and using real selected timepoint/channel indices in batch jobs.
- Bounded the PyTorch GSplat support-grid cache by adaptive per-device byte budgets instead of entry count, with HPC-tunable environment variables and oversized-entry skip behavior to prevent unbounded CPU/GPU memory growth.
- Avoided runtime `torch.compile` CPU toolchain failures by using eager PyTorch on CPU/MPS loss kernels and compiling opportunistically only for CUDA with fallback.

#### Changed — Metal gsplat backend parity, reliability, and performance

- Refactored `GaussianSplatModelMetal` into an MPS-only `GaussianSplatModel` subclass that mirrors the CUDA/base model interface for parameter management (`current_params`, `append_`, `prune_`, `replace_with`, state dicts, constraints) while using custom Metal kernels for 3D MPS tensors and PyTorch rendering for other supported 2D-8D MPS shapes.
- Rewrote the custom 3D Metal renderer from a tile-binned voxel-centric pipeline to a CUDA-style splat-centric pipeline: one Metal threadgroup owns one splat in forward and one splat gradient row in backward.
- Removed the old Metal hot-path tile machinery (`preprocess_3d`, `bin_3d`, tile counts/offsets/content, PyTorch prefix sum, CPU `.item()` allocation sync) and removed the packed-conic `[Z,Y,X]`↔`[X,Y,Z]` reorder; kernels now use native `[Z,Y,X]` Cholesky factors and compute packed conics internally per splat.
- Replaced voxel-centric global parameter-gradient atomics in backward with threadgroup reductions, an inline native `d_conic -> d_L` VJP, and one write per splat gradient. The unconstrained 3D Metal training path now consumes raw model parameters directly, applies sigmoid/softplus transforms and their VJPs inside Metal, and accepts scalar-expanded loss gradients without materializing dense `grad_output`. On the `128³ @ 32k splats` M4 Max benchmark this reduced forward+backward from roughly 133 ms to roughly 2.8-3.0 ms while keeping CPU-reference error around `max_abs_diff≈1.8e-5`.
- Fixed native Metal loading by passing the absolute `default.metallib` path into the extension, rebuilding stale Metal artifacts automatically when sources change, and touching generated artifacts after no-op distutils rebuilds so imports do not repeatedly auto-compile.
- Added/updated Metal tests covering 2D/4D PyTorch rendering, CPU/device-transfer rejection, explicit FP16/dtype rejection, dynamic splat operations, clean nested state dict compatibility, native `[Z,Y,X]` conic ordering, and splat-centric gradient behavior.

#### Added — HuRI PPI flow-field demo

- New `demo_ppi_flow_field.py` turns the HuRI protein-protein interaction graph into a signed-flow UMAP landscape: PageRank orients interactions low→high, a signed sparse adjacency/flow-profile matrix drives the 3D embedding, and protein nodes are forward-advected through a smoothed vector field.
- Vector field construction uses KD-tree edge-sample candidate lookup, exact point-to-segment distances for candidate edges, regularized inverse-cubic weighting, Gaussian component smoothing, and vectorized RK4 integration. The default `full` preset builds the requested 256³ grid; `--preset preview` builds a faster 128³ grid.
- Added `networkx>=3.0` to demo development dependencies for HuRI/CAIDA/PPI graph analysis demos.

### April 2026

#### Added — Cosmicflows-4 Laniakea demo

- New `demo_cosmicflows_laniakea.py` recreates the Cosmicflows-4 / Laniakea flow visualization with 55,486 local-universe galaxies and a full preset that reproduces 29,555 RK4 streamlines from the public `manlius/laniakea` data pipeline.
- Downloads and caches the EDD galaxy table, CF4 velocity field, and basin-of-attraction grid; writes galaxies as Points and each basin's flow as toggleable indexed Lines with HDR additive rendering.

#### Added — Native bundles for distributable scenes

**`luxar export --native macos|linux-amd64|linux-arm64`**

- New flag on `luxar export` that wraps the viewer + zarr around a Go-compiled launcher binary instead of emitting the Python `serve.py` folder. End users double-click and get a real native window — no Python or browser-tab involvement on their machine.
- macOS produces a standard `.app` bundle (`Contents/Info.plist`, `MacOS/launcher`, `Resources/{viewer, data, AppIcon.icns}`) with Cmd+Q / Dock close-button → graceful HTTP server shutdown via `webview.Terminate`. No `LSUIElement`, so the app is fully Dock- and Cmd+Tab-visible.
- Linux produces a portable folder (`<App>-linux-<arch>/{luxar-launcher, viewer/, data/, <App>.png, README.txt}`); the `<App>.png` follows the FreeDesktop icon convention.
- `--name NAME` overrides the default bundle name (which defaults to the zarr stem).

**Embedded launcher**

- New `packages/luxar-launcher/` Go module (~150 lines, `github.com/webview/webview_go` + Go stdlib). Single source compiled to per-OS native binaries.
- Native window via system WebView (WKWebView on macOS, WebKitGTK on Linux). Static HTTP server on a free localhost port, CORS-enabled to mirror `luxar serve`.
- Runtime fallback: `LUXAR_LAUNCHER_NO_WEBVIEW=1` opens the user's default browser instead — useful for headless smoke tests and minimal Linux installs without `libwebkit2gtk`.
- 🌌 emoji icon (matching the viewer's favicon) rendered to PNG via Apple Color Emoji + Pillow, packed to `.icns` via `iconutil`. Source assets committed under `packages/luxar/src/luxar/cli/_launcher_assets/` so wheel installs ship a working icon.

**Build pipeline**

- New Make targets: `make install-go` (Homebrew on macOS, official tarball into `~/.local/go` on Linux — no sudo), `make build-launchers` (CGO=1 host-only build; emits `darwin-universal` via `lipo` on macOS, `linux-<arch>` on Linux), `make clean-launchers`.
- Wired into `check-deps`, `clean-all`, `clean-setup`, `help`, and the `setup-dev` "optional accelerators" footer — same treatment as `make install-rust` and `make setup-cuda`.
- Wheel packaging: launcher binaries (`cli/_launchers/`) and icon assets (`cli/_launcher_assets/`) live inside the Python package and ride along into wheel builds automatically when present at build time.

**Docs**

- New `docs/tutorials/distributing_scenes.rst` covering folder export, native bundles, multi-platform sharing, Gatekeeper / `xattr -cr` recovery, and lifecycle; the tutorial is linked from the main Sphinx docs index.
- New `Native Launcher Setup Details` section in `docs/guides/developer/BUILD_SYSTEM_SPEC.md`.
- New module entry `luxar.cli.native_app` in `docs/api/cli.rst`.
- New per-package READMEs at `packages/luxar-launcher/`, `cli/_launchers/`, `cli/_launcher_assets/`.

**Tests** — 14 new tests in `packages/luxar/src/luxar/cli/tests/test_native_app.py` covering bundle layout, plist correctness (no `LSUIElement`, `CFBundleIconFile = AppIcon`), icon distribution via the package, missing-launcher CLI error handling, name defaulting from `source.stem`, and the `--overwrite`-must-not-wipe-on-failure invariant.

#### Changed — Layer Semantics

**Rendering attribute composition (was: override)**

- Rendering attributes (`opacity`, `gamma`, `intensity`, `offset`, `blending_mode`) now compose along the scene graph root-to-leaf per the Luxar spec, rather than the previous override-only behavior
- `opacity`/`gamma`/`intensity` multiply through the chain; `offset` adds; `blending_mode` uses the nearest ancestor that sets it
- Wired via new `packages/luxar-viewer/src/data/attrs-composer.ts` and `SceneLoader.applyEffectiveAttrs()`; also applied in the progressive GSplats LOD pass-through
- The Layers panel recomposes per-leaf on every slider change so edits to group/ancestor layers flow into every descendant

**Group layers**

- A `group` node with `layer=True` is now exposed in the Layers panel as a composite layer whose controls fan out to every data descendant (points/lines/gsplats)
- Viewer's `LayerStateManager` was previously silently ignoring groups — fixed

#### Changed — GSplats API

**API default loss flipped from MSE to L1.** `fit_gaussian_splats()`,
`GaussianSplatFitter.fit()`, and `LossConfig` now default to
`loss_type="l1"` (was `"mse"`). The change is motivated by the
loss-comparison study (Supp. Doc. 5; `analysis/loss_comparison/`),
which shows L1 reaches equal-or-higher held-out PSNR than MSE on every
microscopy dataset tested, by up to +1.03 dB on the cleanest data and
+0.78 dB on the noisiest. Callers that explicitly pass `loss_type="mse"`
or `"poisson"` are unaffected. Test `LossConfig.test_default_values`
updated to assert `"l1"`.

#### Added — Layer API

- `Node.layer` is now a settable property (previously creation-only): `points.layer = True`
- New `Node.visible` authoring property (default `True`); `add_points(..., layer=True, visible=False)` starts the layer hidden in the panel
- `validate_colormap` now fails fast on unknown names (catches typos like `colormap='viridus'` at authoring time instead of downstream)
- Layers panel gained an **opacity** slider alongside gamma/range/blend/colormap
- Pressing **L** while focus is inside the Layers panel no longer closes it (previously a slider-drag could accidentally dismiss the panel)
- Docs: `LUXAR_ZARR_FORMAT.md` now has a Layers section; `data/README.md` documents composition; `layers/README.md` reflects groups + opacity

#### Major Features

**Screen-Space Overlay System**

- New `scene.add_text()`, `scene.add_image()`, `scene.add_html()` Python API for screen-space annotations
- Overlays rendered as HTML elements over the 3D canvas (below controls)
- Normalized screen coordinates `[0, 1]` with top-left origin, 9-point anchoring
- Viewport-relative sizing (fractions of viewport height/width)
- **Dimension-aware visibility**: `visible_range` parameter shows/hides overlays based on slider positions
- Image overlays: accept file paths, bytes, numpy arrays, PIL Images; stored as raw PNG/JPEG/WebP in zarr
- HTML overlays: restricted safe tag subset with inline styles, XSS sanitization
- Per-overlay configurable fade transitions
- Optional pointer interactivity (`interactive=True`)
- Blend modes for images: normal, multiply, screen, overlay, additive
- Text features: font presets (sans/serif/mono), background boxes, stroke outlines
- 93 new Python tests covering all overlay types, validation, and edge cases

### March 2026

#### Breaking Changes

**Removed sharpness from GSplats**

- Removed `sharpness` attribute from Gaussian Splats (GSplats) geometry type
- Points and Lines retain their sharpness attribute
- GSplats now use the standard Gaussian falloff (equivalent to sharpness=2.0) without per-splat configurability
- Affected formats: `.gsplats.zarr` standalone format and embedded Luxar scene format
- Compatibility: existing `.gsplats.zarr` files with sharpness arrays ignore the sharpness data on load

#### Major Features

**Per-Node GOG Color Model & Global EOG Controls**

- **Per-node Gain-Offset-Gamma (GOG)**: Added `intensity` (linear gain), `offset` (black level subtraction), and `gamma` (tonal curve) to all node types (Points, Lines, GSplats)
- **Use case**: Microscopy background subtraction — negative offset suppresses fluorescence floor per channel
- **Shader model**: `adjusted = color * intensity + offset; clip; pow(adjusted, 1/gamma)` with early discard for zero-contribution fragments
- **Global Exposure-Offset-Gamma (EOG)**: Replaced per-shader `hdrMultiplier` with `exposure` (log2 stops), `global_offset`, `global_gamma` applied in post-processing
- **Vendored tone mapping**: `LuxarToneMappingEffect` extends pmndrs ToneMappingEffect with EOG in a single shader pass (zero extra bandwidth cost)
- **Full config propagation**: Python `ViewerConfig` -> zarr -> TypeScript -> UI sliders -> post-processing uniforms
- **UI**: HDR folder now has Exposure (-5 to +5 stops), Offset (-1 to +1), Gamma (0.1 to 10.0), and Tone Mapping selector

**nD Transforms on Non-Displayed Dimensions**

- Per-dimension affine (`scale`, `offset`) and categorical (`permutation`) transforms for non-displayed dimensions
- Python validation in `nd_transform` property with full round-trip zarr serialization
- Viewer uses inverse-query approach: transforms the query (slicePosition + tolerance) from world to local space O(1), rather than transforming all point coordinates O(N)
- No loader internals changed; works transparently with existing spatial indexing

**Recording Panel with Screenshot and Video Export**

- New recording panel UI (`src/ui/recording-panel.ts`) for capturing viewer output
- Screenshot export: single-frame capture (PNG, WebP, JPEG) with configurable resolution
- Video export: record viewport as video for supplementary materials and demos
- Accessible from viewer UI controls

**Scale Bar Overlay with Physical Units**

- Scale bar component (`src/ui/components/scale-bar.ts`) rendered as an overlay on the viewport
- Supports all Luxar physical units (nm, um, mm, cm, m, km, inch, foot, px, au)
- Automatically adapts to current zoom level and camera projection
- Essential for microscopy figure generation

**Tiled Fitting for Large Volumes**

- Cosine-apodized (Hann window) overlapping tiles for fitting arbitrarily large volumes
- Removes GPU memory ceiling: each tile is fit independently, then splats are concatenated
- Enables parallel fitting across tiles (and potentially across GPUs)
- Implementation in `gsplats/fit_tiled_gsplats.py` with tiling utilities in `gsplats/tiling.py`

**GSplat CLI: filter, split, slice, compare Commands**

- `luxar gsplat filter`: Filter splats by amplitude, eccentricity, bounding box, volume, and more
- `luxar gsplat split`: Split gsplat datasets into parts by count or explicit indices
- `luxar gsplat slice`: Slice by coordinate ranges using numpy-style syntax (e.g., `"0:50, :, 10:90"`)
- `luxar gsplat compare`: PSNR/SSIM quality metrics comparing gsplat reconstruction to original volume

**Camera Utilities**

- New `camera-utils.ts` module with `PerspectiveCamera | OrthographicCamera` union type
- Shared utilities for camera setup across perspective and orthographic projections

#### Maintenance

- Added `luxar[gsplats]` optional dependency group and lazy imports for gsplats tooling
- Standardized Python console output on `arbol` and viewer output on `utils/log`
- Filled missing package documentation across Python and viewer packages

### December 2025

#### Major Features

**Modular Theming System** 🎨

- **What**: Complete theming system with runtime theme switching, CSS-based architecture, and modern aesthetics
- **Themes**: 3 production-ready themes (Dark, Light, Frosted Glass)
- **Components**: All 8 UI components fully themed (error dialogs, help overlay, dimension sliders, debug console, dataset browser, data loading monitor, rendering controls)
- **Architecture**: Three-layer system (Theme definitions -> CSS files -> Component logic)
- **Features**:
    - Instant theme switching via UI dropdown (R key -> 🎨 Theme)
    - URL parameter support (`?theme=light`)
    - Automatic persistence via localStorage
    - 130+ CSS utility classes with BEM naming
    - 80+ CSS custom properties (`--luxar-*`)
    - Zero hardcoded colors in components
    - Consistent 0.15s fade-in animation across all UI panels
- **Benefits**:
    - 400+ inline styles removed (-90% inline styling)
    - CSS bundle: 48KB (6.77KB gzipped) - cacheable separately
    - JS bundle: -13KB reduction
    - Better accessibility (WCAG AAA high-contrast theme)
    - Easier maintenance (single source of truth for colors)
- **Testing**: 1074 unit tests + 21 E2E visual regression tests
- **Polish**: Frosted Glass theme with Apple-inspired frosted glass design, HDR default 1.0
- **Implementation**: 22 commits, 4,200+ lines added, 100% complete with final polish
- **Location**: `src/themes/`, `src/styles/`, UI components
- **Documentation**: Complete implementation plan in `docs/archive/developer-archive/THEMING_IMPLEMENTATION_PLAN.md`

### January 2025

#### Critical Bug Fixes

**Transform Composition Bug**

- **Issue**: Matrix multiplication order was reversed in `compose()` function (left-multiply instead of right-multiply)
- **Impact**: `compose(T1, T2, T3)` was applying T3 first instead of T1 first
- **Fix**: Changed `result = transform @ result` to `result = result @ transform`
- **Location**: `core/transforms.py:271`
- **Lesson**: Order matters for non-commutative transforms (rotate+translate). Always test with order-sensitive operations.

**Points Metadata Loss Bug**

- **Issue**: `Points.__init__()` set `self._metadata` before calling `super().__init__()`, then `Node.__init__()` overwrote it with empty dict
- **Impact**: ALL Points objects lost their metadata (has_colors, has_radii, max_radius, etc.)
- **Fix**: Call `super().__init__()` BEFORE setting `self._metadata` in Points class
- **Location**: `core/points.py:50-58`
- **Lesson**: When subclass and parent both initialize the same attribute, parent must initialize first

**Fullscreen Resize Bug**

- **Issue**: Point sizes changed incorrectly on first fullscreen toggle or window resize
- **Root Cause**: Scene initialization didn't call `updateSize()`, causing different behavior on first resize
- **Solution**: Make initialization call `this.updateSize()` in `scene-manager.ts` init() method
- **Lesson**: Ensure initialization and resize paths are identical to avoid first-time-only bugs

#### Code Cleanup

**Dead Code Removed**

- Eliminated unused mode handling from Node class (~40 lines dead code)
- Removed `group` parameter and all `if self._group is not None:` branches
- Node now only supports progressive writing mode (simpler, clearer)

**Deprecated Parameters Removed**

- Removed `units` parameter from `LuxarZarrCompiler` (use Dimensions instead)
- Removed `DimensionMetadata` class (use full-featured `Dimension` instead)
- Removed unused version constants (LEGACY, PREVIOUS, FUTURE)
- Total: ~160 lines of dead/deprecated code removed

#### Quality Improvements

**Compiler Cleanup**

- Reduced `write_points()` from 432 to 117 lines (73% reduction)
- Extracted 8 focused helper methods with single responsibilities

**Validation Consolidation**

- Created `validation/types.py` centralizing all validation
- Eliminated ~350 lines of duplication between `protocols.py` and `validation/base.py`
- Clear organization: types.py (basic), base.py (detailed for writing), nd.py (dimensional)

**Transform Handling Centralized**

- Added `read_transform_from_zarr()` companion function
- Single source of truth for NumPy to THREE.js transform conversion
- Eliminated ~50 lines of duplicate transpose logic

**Magic Numbers Extracted**

- Created 18 named constants for spatial index tuning
- All grid sizing heuristics now configurable via constants

**Performance**

- Vectorized HSV to RGB conversion in demos (30-100x faster)

**Documentation**

- Added 80+ inline comments explaining complex algorithms

#### New Features

**Debug Console & Console Logging**

- In-App Debug Console: Press Ctrl+L to toggle debug console
- Ring Buffer Implementation: Console interceptor uses 10,000 message ring buffer
- Early Message Capture: Console messages captured from app initialization
- Standardized Logging: All console logs use format: `[emoji] [Luxar] message`
- Debug Interface: Debug tools available at `window.__luxarDebug` when `?debug` URL param is present

**HDR Color Pipeline**

- Float32 Colors: Changed from Uint8Array to Float32Array for HDR support
- nD Slicing Fix: Updated slicing algorithms to use `sliceColorsFloat32()` for proper HDR colors
- HDR Detection: Added comprehensive HDR capability detection in `utils/hdr-detection.ts`
- Note: WebGL canvas doesn't support true HDR output (limited to 8-bit)

**World-Space Point Sizing**

- Physical Accuracy: Points now use world-space sizing instead of screen-space
- Key Property: Two points with radius r at distance 2r will just touch
- FOV Independence: Points maintain physical size regardless of field of view changes
- Formula: `angularSize = 2 * atan(radius/distance)`, then converted to pixels

**nD Visualization**

- Slicing Tolerance: Use point radius for visibility, not fixed tolerance
- Scene Dimensions: Always define at scene level for consistency
- Keyboard Navigation: Simple 2-step: select dimension (1-9), navigate ([/])
- TypeScript Integration: Scene dimensions loaded from zarr attrs, used for step sizes

**Data Loading Architecture Cleanup**

- Removed Lazy Loading: Eliminated LazyDataManager in favor of spatial index-based loading
- Spatial Index Required: All datasets now require spatial indices for efficient loading
- Range-Based Caching: New RangeCache system for intelligent memory management
- Improved Monitoring: Enhanced DataLoadingMonitor with better error handling and disposal
- Cleaner Architecture: Removed intermediate abstractions for simpler, more maintainable code

---

## Earlier History

See git history for changes prior to January 2025.
