# Changelog

All notable changes to Luxar are documented in this file.

## [Unreleased]

### July 2026

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
  widths, per-endpoint colors + sharpness, segment length + clip flags,
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
