# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### June 2026

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
respected. A 30M-element scene that previously error-stormed now loads cleanly.

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
  `"display"`, or `"all"`. **Default is Auto** — coarsen the scene's *displayed*
  dims, group by the *non-displayed* dims — so nD scenes are correct automatically
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
diagonal **collapsed** — dropping to a *coarse* level exactly on close approach,
the inverse of the intended behaviour. The projection is now `w`-aware and
**saturates to `+Infinity`** (→ finest level) when any corner has `w ≤ 1e-6`,
reusing the per-frame `projectionMatrix × matrixWorldInverse` product. Also:

- Added a behind-camera reject (`uIsOrtho == 0 && view-z ≥ 0` → off-screen) to
  the **Points** vertex shaders — visual *and* picking, GLSL *and* TSL — matching
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
  zero-count *intermediate* level — or a zero-extent level whose neighbour
  derives to `0` — previously tripped the strict-ascending assertion and aborted
  the build; the guard is now *total* (never raises).
- `derive_min_pixel_sizes`' empty-coarsest error is now actionable (names the
  likely cause: a substitutive reduction that culled every representative, e.g.
  all-non-positive input amplitudes) instead of "coarsest child must have at
  least 1 element".
- The viewer's LOD child-order check is now strict (`<`, was `<=`), matching the
  Python writer's `_assert_strict_ascending`: *equal* adjacent `min_pixel_size`
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
at which the viewer swaps levels — were derived purely from element *counts*
(`base_pixel_size · √(nᵢ/n₀)`). That scene-relative proxy is biased for
substitutive levels (a coarse level has *fewer but larger* elements), so it
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

- **partitioned** — a spatial BSP `kind=partition` where *each part carries its
  own additive ladder* (the old `partition` collapsed parts to a single level).
- **multiscale** — an unbalanced-by-design `kind=lod`: a single coarse
  substitutive cap for the far view above a `partitioned` fine branch, so detail
  structure exists only where you look closely.
- **mosaic** — a spatial BSP `kind=partition` where *each part is its own
  substitutive lod group* (per-part coarse↔fine replacement): every cell
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

Two follow-ups make the switch actually *visible*: grafted `kind=partition`
wrappers are now back-filled with `position_bounds` at scene finalization (the
graft, unlike the standalone writer, didn't compute the children union — needed
for partition-unit frustum culling), and `multiscale` gained a `base_pixel_size`
anchor (`RecipeParams.base_pixel_size` / the new `gsplat lod --base-pixel-size`
flag). The coarse cap is a *substitutive* level — fewer but larger splats — so the
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
- **Point material parity for max blending.** `PointMaterial.applyBlendingMode` is the new single source of truth; runtime UI mode changes now produce the same `OneFactor/OneFactor` blend factors as creation-time max. The fragment shader gains a `LUXAR_MAX_RGB_CONTRIBUTION` define so `max` mode premultiplies RGB by intensity*opacity.
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
  that builds a coarse-to-fine *substitutive* LOD hierarchy: each level
  *replaces* the previous one with ``M = N / K^ℓ`` synthesised representative
  splats. Output is a directory of per-level ``.gsplats.zarr`` files plus a
  ``manifest.json``.
- New module `luxar.gsplats.lod.substitutive` with `make_substitutive_lod(data,
  *, compression_factor=4, levels=3, method="kmeans_lloyd", ...)`. Three
  partition algorithms supported: ``kmeans_lloyd`` (k-means warm-start +
  cost-increment Lloyd refinement, the recommended workhorse), ``greedy``
  (hierarchical pairwise greedy), and ``kmeans`` (spatial-only).
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
  + ~440 lines in `gsplats/tests/test_substitutive_lod.py`.

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
  <out.gsplats.zarr>` that *reorders* the splats of a fitted dataset into
  a multi-LOD `GSplatData` whose prefix sum at any ``k`` splats is the
  best L² approximation of the full scene. Output is a single multi-LOD
  ``.gsplats.zarr`` (each level *extends* the previous one).
- New module `luxar.gsplats.lod.additive` with `make_additive_lod(data,
  n_lods=4, method=..., breakpoints=..., ...)` and
  `compute_additive_order(...)`. Four ordering methods supported:
  ``greedy`` (residual-correlation matching pursuit; default),
  ``self_energy`` (cheap O(N log N) baseline within 2-10% AUC of greedy
  on real datasets), ``mass`` (peak-amplitude × covariance volume), and
  ``amplitude`` (peak height alone). Algorithms documented in
  `manuscript/supp_doc/additive_lod`.
- **Progressive fitting decoupled from LOD construction.**
  `fit_progressive_gaussian_splats` (and the `luxar gsplat fit
  --progressive` CLI flag) now returns a single flattened
  ``GSplatData`` rather than a multi-LOD container. To build an LOD
  ladder, run `luxar gsplat lod additive` on the flat output. Progressive
  multi-pass fitting remains a valid alternative fitting flow — it just
  no longer overloads the LOD concept.
- New tests in `gsplats/tests/test_additive_lod.py`.

#### Added — `luxar gsplat cal` (blind-spot CV calibration) (PR #105)

- New CLI command `luxar gsplat cal <volume> <out.json>` that sweeps splat
  count `K` and reports the recommended `K*` via blind-spot
  cross-validation: 5%-donut-median masking (Noise2Self protocol from
  Batson & Royer 2019), fit at each `K` against the masked volume, evaluate
  PSNR at the held-out positions against the *original* values. Hybrid
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
  PSNR is fundamentally a *capacity*-selection criterion (across `K`),
  not an *iteration*-selection one — the existing patience-based early
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
