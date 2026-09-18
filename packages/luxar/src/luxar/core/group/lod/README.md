# luxar.core.group.lod

Helpers for the **LOD-kind `Group`** — a scene-graph group that selects one of
N alternative children at runtime based on a view-driven metric. Which metric is
named by the group's own `selector`: the fraction of the viewport the projected
bbox covers by AREA under `screen-area` (what every derived ladder stamps), or
the projected bbox diagonal in pixels under the legacy `coverage` — see "The
selector rule" below. The LOD group is geometry-agnostic: its
children can be `points`, `lines`, `gsplats`, `mesh`, or themselves a specialized
group (`kind=lod` / `kind=partition`).

This package splits cleanly into two layers:

- **Geometry-agnostic machinery** (`group.py`) — threshold derivation, the
  `coverage_fraction` monotonicity invariant, the `kind=lod` validator, and the
  shared display-type resolver. Shared by every leaf geometry and by the
  Partition kind.
- **Per-geometry axis resolvers** (`points.py`, `lines.py`, `gsplats.py`,
  `mesh.py`) — one peer per leaf type, interpreting the `additive_lod=` /
  `substitutive_lod=` / `additive_lod=` convenience kwargs that the four geometry
  adders accept. Gsplats also retain `lod_group=` as a backward-compatible alias
  for `substitutive_lod=`. Mesh's vocabulary is the shortest — decimation rather
  than a lift, and a **reveal-only** additive axis.

The two sampler modules (`spatial_uniform.py`, `poisson_disk.py`) are NumPy
ordering primitives shared by the Points and Lines resolvers (`poisson_disk` also
runs an interpreted rejection loop — Bridson is sequential by construction), and
`reveal.py` is a third such primitive — the concentric-shell scorer behind
`method="radial"`, plus the resolvers that decide which columns may be shell
dimensions. It depends on nothing in this package (`group.py` imports *it*), which
is what let it come out of `group.py` cleanly.

## File structure

```
lod/
├── group.py            # Geometry-agnostic LOD-group machinery (thresholds, validator, display-type)
├── points.py           # Points additive-LOD ordering + ladder construction + resolver
├── lines.py            # Lines additive-LOD (per-polyline) ordering + ladder + resolver
├── gsplats.py           # GSplats substitutive + additive axis resolvers
├── mesh.py             # Mesh axes: substitutive (decimation, not lift-to-gsplats) + additive (reveal only)
├── reveal.py           # Concentric-shell reveal: radial scorer + spatial-dims resolvers
├── spatial_uniform.py  # Stratified-grid sampler (default spatial-uniform ordering)
└── poisson_disk.py     # Bridson blue-noise sampler (opt-in alternative)
```

## Two LOD axes

Luxar's gsplat LOD model has two orthogonal axes; the helpers here build both:

- **Substitutive** — coarser levels *replace* finer ones (fewer, larger
  elements). A multi-substitutive dataset becomes a `kind=lod` `Group` whose
  children are the levels in coarsest→finest order. GSplats carry a stored
  substitutive pyramid (`gsplats.py::resolve_substitutive_axis_gsplats` resolves
  the `lod_group=` kwarg). **Points** synthesise one on demand via the
  `substitutive_lod=` kwarg: each point is *lifted* to an isotropic Gaussian and
  the gsplat substitutive pipeline coarsens it — so the coarse levels are
  mass-preserving gsplats while the finest child stays the original Points node
  (a heterogeneous `kind=lod` group). See "Points substitutive" below.
- **Additive** — finer levels *add to* coarser ones (a streaming prefix of
  elements). All three geometries support an additive ladder via the
  `additive_lod=` kwarg. For gsplats it is applied per substitutive level
  (`resolve_additive_axis_gsplats`).

See `docs/specs/GSPLATS_ZARR_FORMAT.md` for the v3.4 node-tree grammar
(leaf / kind=lod / kind=partition) the gsplat format stores.

## Geometry-agnostic machinery (`group.py`)

Each LOD-group child carries a `coverage_fraction` attribute — a **dimensionless,
viewport-relative** threshold, strictly monotonic increasing in coarsest→finest
order (coarsest = `0.0`; a derived WHOLE-OBJECT ladder anchors its finest at
`WHOLE_OBJECT_FINEST_ANCHOR` = `0.5`, a derived ladder bound to a spatial
partition at `PARTITION_FINEST_AREA` = `1.0`, and an explicitly authored list —
being in the legacy units, see "The selector rule" below — may go up to
`MAX_COVERAGE_FRACTION` = `4.0`). Under the legacy `selector="coverage"` the
viewer multiplies each child's `coverage_fraction` by the current
viewport's **fitted screen axis** (`min(viewport.width, viewport.height)` in
pixels, times a fill-factor constant of `0.5` — the extent the camera framing
actually fits, so the comparison holds across aspect ratio and not just
viewport size) and picks the finest child whose resulting pixel threshold is
satisfied by the group's on-screen size. A legacy finest value is
AUTHOR-chosen, so where it activates is the author's call
(`core/tests/test_mesh_substitutive_lod.py::test_an_explicit_list_may_reach_the_legacy_ceiling`
writes `[0.0, 2.0, 4.0]` verbatim, whose finest waits for *twice* the fitted
axis); at the legacy finest of `1.0` — what the *retired* derivation
produced, and what no live derivation produces any more — the finest level
activates once the object's projected bbox diagonal reaches half of the fitted
screen axis, i.e. at any normal full-frame view, with coarser levels stepping
in as it shrinks below that, identically on any monitor or viewport size.
Across aspect ratio
the switch point is *exact* for a landscape viewport (aspect >= 1) and within
~25% of that value for a portrait one, where the camera fit distance itself
varies with aspect.

| Symbol | Purpose |
|--------|---------|
| `coverage_fractions(element_counts)` | **Auto-derivation, by SCREEN-OCCUPANCY HALVING.** Coarsest = `0.0` (the always-eligible floor), finest = `WHOLE_OBJECT_FINEST_ANCHOR` = `0.5` — full detail while the object occupies at least half the screen — and each level between halves once more: `[0, …, 1/8, 1/4, 1/2]`. Count-**INDEPENDENT**: `element_counts` sets only the ladder's LENGTH (the one per-level check is that the finest count is > 0), so the retired `sqrt(N_i / N_finest)` count-ratio derivation and its multiplicity-cancels-out rationale no longer apply — counts do not enter the thresholds at all. |
| `validate_lod_group(group)` | Free-function validator for any `Group` with `attrs["kind"] == "lod"`. Raises on no children, out-of-range `default_level`, missing `coverage_fraction`, or non-monotonic thresholds. |
| `resolve_display_type(node)` | The geometry type a node appears as to the user. For `kind in (lod, partition)` returns the recorded `display_type`; else the node's own `type`. Shared with the Partition kind's validator. |
| `compute_lod_display_type(children)` | Derive an LOD group's `display_type` from its finest (last) child, recursing through nested specialized groups. |
| `_assert_strict_ascending(thresholds, source)` | The shared monotonicity guard, applied by both the explicit-`coverage_fractions=` resolver paths and `coverage_fractions()`. |
| `_apply_monotonicity_guard(thresholds, source, cap=1.0)` | Defensive relative (×1.1) *downward* nudge so near-equal/degenerate levels still separate strictly, before the trailing `_assert_strict_ascending` check — downward so nothing is ever pushed past the anchor, with one exception the docstring spells out: a resulting ZERO PREFIX is lifted back UPWARD onto a geometric ramp between the `0.0` floor and the first positive threshold, since a run of zeros is not strictly ascending either (`[0, 0, 0, 0.5]` → `[0, 0.413…, 0.4545…, 0.5]`). Also caps the finest at `cap` — supplied by the CALLER, not fixed at the `1.0` default: `coverage_fractions` passes `cap=WHOLE_OBJECT_FINEST_ANCHOR`, so derived whole-object output stays in `[0, 0.5]`, and `partitioned_coverage_fractions` rescales ×2 *after* the guard has run, landing its finest on `PARTITION_FINEST_AREA` = `1.0`. |
| `partitioned_coverage_fractions(element_counts)` | **The partition-bound anchor.** `coverage_fractions(...)` scaled by `PARTITION_FINEST_AREA / WHOLE_OBJECT_FINEST_ANCHOR` (×2 in area units), so the finest lands on `PARTITION_FINEST_AREA` = `1.0`: the tile alone fills the screen. Used wherever a spatial partition is part of the switch: the `adaptive` recipe's per-tile lod groups, and the `overview` recipe's coarse-cap/fine-partition pair. A tile's projected rect is intrinsically a fraction of the whole object's (the metric is an AREA fraction — the wording the function's own docstring uses), so the whole-object anchor would put every tile on its finest level while the object is merely full-frame. **A ONE-part partition is excluded** — its single part covers the whole object, so `coverage_fractions` applies (`build_adaptive` and both tree writers' topology fallbacks special-case it). |
| `is_partition_bound(node)` | Does `node` (a ladder's **insertion point** — the future lod group's parent) sit inside a `kind=partition`? Walks `parent` links up, so the partition wrapper itself counts and a plain `add_group` in between still counts (the ladder is still inside one tile). The scene-graph mirror of the `under_partition` recursion flag the two gsplat writers thread down a detached tree. |
| `derive_coverage_fractions(element_counts, insertion_point, *, name=..., partition_bound=False)` | **The scene-side anchor chokepoint.** `partitioned_coverage_fractions` when the insertion point has a partition ancestor, or when `partition_bound=True` marks the verified multi-part finest branch of a Points overview; otherwise `coverage_fractions`. The ancestor route is geometric (one tile's bbox); the overview route is contractual (the group's bbox is the whole object, but fine detail stays deferred until zoom). Both log the anchor switch. The four scene adders use this through `resolve_lod_ladder`; explicit thresholds still win. Only the ancestor route cannot verify the final sibling count, so finalize warns on a one-part hand-built partition. |
| `resolve_lod_ladder(explicit, element_counts, insertion_point, *, name, partition_bound=False, length_error)` | **The explicit-vs-derived rule, decided once for the four scene adders.** Returns `(coverage_fractions, selector)` — thresholds *and* the `selector` naming their units, because those are one decision, not two. Explicit list → used verbatim with `LEGACY_LOD_SELECTOR`; otherwise `derive_coverage_fractions(...)` with `DERIVED_LOD_SELECTOR`, forwarding the verified-partition context. All four scene adders call it and none names a selector itself; the AST guard in `test_lod_selector_contract.py` keeps it that way. Detached-tree paths use `gsplats.tree.gate_authored_selector` instead. |
| `DERIVED_LOD_SELECTOR` / `LEGACY_LOD_SELECTOR` | `"screen-area"` / `"coverage"` (in `typing_utils/constants.py`, and together they ARE `LOD_SELECTORS`). The units a group's thresholds are in — see the rule below. `LEGACY_LOD_SELECTOR` is also the `add_lod_group` default, since a hand-built ladder is authored rather than derived. |
| `MAX_COVERAGE_FRACTION` | `4.0` — the ceiling on a **legacy** (`selector="coverage"`) ladder, so in practice on an explicitly authored `coverage_fractions=[...]` list and on a hand-written per-child `coverage_fraction=`; a *derived* screen-area ladder is capped at `PARTITION_FINEST_AREA` = `1.0` instead, which is the split `validate_lod_group` makes. It is `SCREEN_FILL_DIAGONAL_RATIO / FILL_FACTOR`: approximately the coverage metric a *screen-filling* object produces (exact only near aspect ratio √3 ≈ 1.73 — see the `FILL_FACTOR` doc in `luxar-viewer/src/scene/lod-group-registry.ts`), i.e. "a level may be required to fill the screen, at most" — the same numeric value `1.0` meant before the viewer's anchor moved from screen-filling to half the fitted screen axis, though no longer an exact identity at every aspect ratio the way it was under that older anchor. |

### The selector rule (stated once)

A `kind=lod` group's `selector` attr names the **units** its children's
`coverage_fraction` thresholds are in, so the two must always be decided
together:

- **explicit** `coverage_fractions=[...]` ⇒ the values are used verbatim and the
  group is stamped `LEGACY_LOD_SELECTOR` (`"coverage"`, the legacy diagonal
  metric bounded by `MAX_COVERAGE_FRACTION`). An authored list was tuned against
  that metric — it is what `add_lod_group` has always defaulted to and what every
  existing dataset means — so re-labelling it would silently move every switch
  point the author chose;
- **derived** (no explicit list) ⇒ `derive_coverage_fractions(...)` and
  `DERIVED_LOD_SELECTOR` (`"screen-area"`, literal screen-area fractions:
  finest at the whole-object anchor, or the fills-screen anchor when the
  insertion point is partition-bound).

`resolve_lod_ladder` is the one implementation the four `substitutive_lod=` /
`lod_group=` **scene adders** share — they delegate to it rather than restating
it, and `core/tests/group/lod/test_lod_selector_contract.py` pins both halves plus
the consistency invariant (a stamped selector must agree with its own stamped
thresholds — mislabelling raises nothing, it just puts the viewer's metric on the
wrong scale).

It is **not** the only place the pairing is decided, and the others are
deliberately separate rather than missed. A detached `.gsplats.zarr` tree has no
`explicit` argument to branch on; the question there is *what does a stored tree
already claim about its own thresholds*, answered by
[`luxar.gsplats.tree.gate_authored_selector`](../../../gsplats/tree.py) — a fully
authored ladder KEEPS its own stored selector verbatim (validated against that
selector's contract), falling back to legacy only when it carries none, while a
partially- or un-authored one is re-derived and stamped screen-area (an
out-of-vocabulary selector raises before anything is written). So a stored
`screen-area` ladder does not lose its stamp on re-save. That gate is shared by
the two detached-tree serializers
(`io/_compiler/gsplat_tree.py::write_gsplat_node` and
`gsplats_pipeline/from_io.py::graft_gsplat_node`), and
`gsplats.tree.tree_from_substitutive_levels` keys its own selector default on
whether a `coverage` callable was supplied. `graft_gsplat_node` is the one scene
door on that side of the split: `add_gsplats_from_file` of a *non*-matrix-shaped
subtree builds its `kind=lod` group there, calling `coverage_fractions` /
`partitioned_coverage_fractions` directly and taking its selector from the gate.
The AST guard in the contract suite exempts it by name for exactly that reason.

**No tunable anchor.** There is no method selector or per-dataset knob (the
former `extent`/`count` methods and `base_pixel_size`/`extent_percentile`/
`extent_anisotropy` are gone) — the anchor is a fraction of the LIVE viewport,
so the switch point self-calibrates to whatever monitor/window/aspect ratio the
viewer runs in: a derived (`selector="screen-area"`) ladder anchors its finest
level at half the screen AREA, and under the legacy `selector="coverage"` the
equivalent anchor is half of the live fitted screen axis.

## Per-geometry resolvers

All four resolvers share a `None | bool | dict` value vocabulary for the
convenience kwargs, but the semantics differ per geometry. Mesh's differs most —
it decimates instead of lifting to gsplats, so it rejects the four lift-only keys,
and its additive axis accepts one method (`radial`) instead of five; see the
`mesh.py` module docstring and the Mesh section below.

### Points (`points.py`)

`resolve_additive_axis_points(spec)` normalizes the `additive_lod=` kwarg into a
dict (`method` / `n_lods` / `counts` / `seed` / `salience_kind`) or `None` — a
thin wrapper over the shared `group.resolve_additive_axis(spec, "Points")` (one
body, shared with Lines, so the two can't drift).
The compiler (`_write_points_multi_lod`) then calls
`make_additive_lod_points(...)`, which returns per-LOD-level index arrays.

Ordering methods (`PointsMethodName`):

- `random` — uniform-random permutation (optional `seed`).
- `salience` — sort by radius descending (`salience_kind='size'`, default) or by
  `luminance × radius³` (`salience_kind='energy'`).
- `spatial-uniform` — stratified-grid sampling (see `spatial_uniform.py`).
- `poisson-disk` — Bridson blue-noise sampling.

Breakpoint vocabulary for `counts` / `breakpoints`:

- `None` → equal-count split into `n_lods` levels (`DEFAULT_N_LODS = 4`).
- `List[int]` → cumulative element-count breakpoints, e.g. `[1500, 8000, 40000]`.
- `"energy:0.5,0.9,0.99,1.0"` → cumulative perceptual-energy fractions; element
  energy is `luminance_i × radius_i³`. Requires `colors` or `scalars` for the
  luminance term.
- `"equi-energy:4"` → 4 rungs at equal shares of cumulative perceptual energy
  along the ordering (same energy as `energy:`), commit-capped; with
  `method="salience", salience_kind="energy"` the first rung is few heavy
  elements and the late rungs fat. For Lines the cap is counted in vertices.
- `"stream:40000"` → bandwidth-derived geometric ladder `[c, 2c, 4c, …, N]`, so
  first paint costs `c` elements and each refinement doubles. Resolved against
  the actual N, so one spec adapts to every level of a tree. For Lines, `c` is
  counted in **vertices** (the payload currency, symmetric with Points and
  GSplats) and converted internally to a polyline count, so cuts still land on
  whole-polyline boundaries. Cut geometry is shared with the GSplats ladder via
  `luxar/utils/lod_breakpoints.py`.

Prefer `stream:` over `n_lods` for large leaves: an equal-count split into 4
levels still ends with an N/4-sized commit, which is not a progressive paint.

`DEFAULT_METHOD` is `random`; `DEFAULT_N_LODS` is `4`.

#### Points substitutive (gsplat or points coarse levels)

`resolve_substitutive_axis_points(spec)` normalizes the `substitutive_lod=`
kwarg (`None`/`False` no-op; `True`/`dict()` defaults `K=4, levels=3,
method="auto"`; dict keys `compression_factor` (`K`), `levels` (`n_lods`),
`coarse`, `brightness_compensation`, `method`, `truncation_radius`, `device`,
`seed`, `coverage_fractions`
(explicit per-level viewport-relative thresholds, strict-ascending in
`[0, MAX_COVERAGE_FRACTION]` = `[0, 4]`), `coarsen_dims`, `max_aspect`
(per-splat anisotropy cap on the
coarse levels, default 3.0; `None` disables)).
`add_points_substitutive_lod_wrapper_impl` (`adders/points.py`) then:

With the default `coarse="gsplats"`, it:

1. **Lifts** each point to an isotropic Gaussian
   (`gsplats.lift.lift_points_to_gsplats`): `σ = 2R/T`, `a = opacity/(uRIF·σ)`
   — calibrated against the viewer shaders so a single lifted splat renders like
   its point (peak ratio 1.0, profile rel-L2 0.45% at the default `T=3`).
2. **Coarsens** via `gsplats.lift.coarse_substitutive_levels` →
   `make_substitutive_lod` with per-bin **mass-preserving amplitudes**
   (`amplitude="mass"` — per-channel colored light is conserved bin-by-bin, so
   hue stays coherent across levels), drops the 1:1 level 0 (the Points node is
   the finest level), **caps** each merged splat's anisotropy at `max_aspect`
   (default 3, mass-preserving — the merge would otherwise elongate the
   isotropic lifted splats level over level, whose view-dependent ray integrals
   flare end-on and pop between levels), and **rescales** each coarse level's
   amplitudes to conserve render-light (`Σ a·σ³`) so the LOD seam does not dim
   on zoom-out.
3. **Assembles** a `kind=lod` group: coarse gsplat children (coarsest-first) +
   the original Points node as the finest child; `display_type="points"`. With
   an explicit `partition=`, the finest child is instead a spatial partition and
   the ladder uses the overview fills-screen anchor. A one-part split falls back
   to the ordinary Points finest child and whole-object anchor. The per-tile
   `adaptive` shape used by `demo_biodiversity_planetary_scale` remains
   hand-built; this composition is the global-coarse `overview` shape.

With `coarse="points"`, it instead takes exact `N/K^level` prefixes of a
spatially stratified ordering and writes those rows as Points children. On a
stacked node, each discrete hidden coordinate is spatially ordered independently
and the orders are round-robin interleaved so a coarse level does not starve
individual slices. Dimensions named by `extend_to_all` are excluded because the
node is not sliced there. The coarsest level must have room for every remaining
discrete hidden coordinate or authoring raises; continuous hidden axes are not
treated as slices and emit a warning. Radii and all selected point channels stay
attached to the original rows. Under the effective nearest-setter-wins `additive`
or `luminous` mode,
`brightness_compensation="auto"` scales RGB by the finest/subsampled
`compute_points_energy` ratio, preserving summed light at the finest radius
rather than inflating screen coverage; a non-identity gain widens colours to
float32 HDR. Other blending modes default to a gain of 1 without widening the
input colour dtype; a numeric compensation overrides the per-level gain through
the same RGB path. The expected HDR warning is suppressed
for these synthesized compensated children. The gain conserves the summed light
over the whole node, not within each neighbourhood, so sparse and dense regions
can shift relative brightness. `truncation_radius`, `max_aspect`, `method`,
`device`, and `coarsen_dims` are refused because they do not affect the written
same-type geometry. By default each child carries a `level_stats.quality` stamp
measured from a fixed isotropic lift on CPU; for Points this scores the geometric
subsample and radii, not RGB/alpha or the brightness-compensation gain. Set
`quality_stamps=False` to skip that measurement.

Lines keeps the lifted-GSplat default too. With `coarse="lines"`, it instead
takes exact `P/K^level` prefixes of a seeded salience ordering and writes every
prefix as a Lines child, preserving whole-polyline topology and selected
per-vertex channels. Discrete hidden coordinates are ordered independently and
round-robin interleaved; dimensions named by `extend_to_all` are excluded because
the node is not sliced there. Authoring refuses a ladder whose coarsest polyline
count cannot represent every remaining occupied slice, or a polyline that crosses
one. Under the effective `additive` or `luminous` mode,
`brightness_compensation="auto"` measures `Σ(length × width × luminance)` and
conserves it at each level. Width growth is capped near the shader's pixel floor
at the level transition and the residual gain is carried by float32 HDR color,
keeping coarse fibers fiber-shaped; other modes keep unit gain, and a numeric
override applies per reduction level through the same capped width/HDR-color
split. The Gaussian lift controls (`truncation_radius`, `max_aspect`, `method`,
`device`, and `coarsen_dims`) are refused because they do not affect the written
same-type geometry. By default each child carries a `level_stats.quality` stamp
measured from a fixed bead lift on CPU; it captures the subsample and width
compensation, but not RGB/alpha or any residual HDR color gain. Set
`quality_stamps=False` to skip that measurement.

Composes with `additive_lod`: substitutive chooses WHICH level renders at the
current zoom, additive describes HOW each level streams in. Every level is given
a `stream:` ladder by default (`additive_lod=False` opts out), except that
`image_labels` are stored only on the original finest Points child and suppress
only that child's ladder; synthesized coarse levels still stream — see "Composed
axes" in `group.py`.
The lift is strictly isotropic (brightness stays view-independent). Scalar +
colormap points are supported by **baking** `scalars`→RGB through the colormap
LUT (`luxar.colormaps.scalars_to_colors`, same normalisation the viewer uses)
and lifting with those colours; the finest Points child keeps `scalars`+`colormap`
(native). Caveats: a *live* colormap change in the viewer re-colours only the
finest child, not the baked coarse gsplat levels; and a node `gamma` ≠ 1 is not
reproduced on the coarse levels (colormap mode applies gamma to the scalar
*pre-LUT* on the finest child, whereas the baked-colour gsplats get gamma applied
to RGB — fundamentally different, so they diverge at `gamma` ≠ 1). (`scalars`
without a `colormap` still raises.) A **uniform** `colors` — an RGB(A) tuple or
a `(1, c)` row — is broadcast onto every coarse level, alpha included, so it
renders the same at every LOD level (#1444); the one inexactness is an alpha
ABOVE `ALPHA_CLAMP = 511/512`, which the merge's optical-depth round-trip caps —
an authored 1.0 (or 0.999) reaches the coarse levels as 0.998, a ≤0.2% step the
finest child does not have. At or below the clamp the round-trip is exact.
A **per-element** `(N, 4)` RGBA is still refused by the lift, because
the substitutive merge is untested on a varying alpha (drop the alpha column, or
use `partition=` / `additive_lod=`). Colour dtype follows the leaf's rule —
floating, uint8 or uint16 — and any other (an `int64` array, say) is refused
before anything is built, rather than baking a near-black coarse level the
encoder then rejects at the finest child. A scene-door caller now hits that in
the shared pre-write validator (`validation.base.validate_color_dtype`, #1489),
which every geometry and every structural path runs; the lift keeps its own copy
of the rule for direct callers of `lift_points_to_gsplats` /
`lift_lines_to_gsplats`, which bypass the scene entirely.

### Lines (`lines.py`)

Mirrors Points in shape but operates **per-polyline**: each LOD level carries
whole polylines (vertices + their segments) so segment topology stays valid
during partial loads.

- `identify_polylines(n_vertices, line_type, indices=None)` splits vertices into
  per-polyline index arrays. `segments` → N/2 length-2 polylines; `indexed` →
  connected components found by vectorized root hooking + pointer jumping,
  followed by one stable root-label sort (components ordered by their smallest
  vertex); `polyline` / `loop` → one polyline spanning all vertices (a multi-LOD
  ladder is then a no-op — a warning is logged and a single level emitted).
- `compute_additive_order_lines(...)` orders polylines (not vertices); for
  `spatial-uniform` / `poisson-disk` the representative point is each polyline's
  bbox center.
- `make_additive_lod_lines(...)` returns per-LOD-level lists of per-polyline
  index arrays for `_write_lines_multi_lod` to gather and rewrite with
  subgroup-local segment indices. For `indexed` Lines, a multi-level ladder is
  refused unless each component's authored undirected edge multiset is exactly its
  consecutive-vertex chain; flat writes preserve the authored edge multiset.
- `salience_kind='energy'` uses the tube-volume score
  `mean_luminance × Σ(seg_length × width²)`.
- `resolve_additive_axis_lines(spec)` is the `add_lines(..., additive_lod=...)`
  resolver — a thin wrapper over the shared
  `group.resolve_additive_axis(spec, "Lines")` (one body, shared with Points).

#### Lines substitutive (lift to gsplats)

`resolve_substitutive_axis_lines(spec)` is the `add_lines(..., substitutive_lod=...)`
resolver — a thin wrapper over the shared
`group.resolve_substitutive_axis(spec, "Lines")` (one body, shared with Points,
so the two can't drift). `add_lines_substitutive_lod_wrapper_impl`
(`adders/lines.py`) then:

1. **Lifts** each segment to a string of isotropic **bead** Gaussians
   (`gsplats.lift.lift_lines_to_gsplats`): beads spaced `σ_perp = 2w/T` along the
   segment, each isotropic. Beads (not one elongated anisotropic Gaussian) because
   the gsplat ray-integral is *view-dependent* for anisotropic covariances (a
   single elongated Gaussian is ~`L/(4w)` brighter end-on than broadside);
   isotropic beads are view-independent and sum to a smooth tube. Per-bead
   amplitude divides by the Gaussian-comb overlap `√(2π)` so the tube centreline
   = `opacity`.
2. **Coarsens** via `coarse_substitutive_levels` (drop level 0, per-bin mass
   amplitudes, `max_aspect` anisotropy cap, render-light rescale) — identical
   to Points. The cap matters most here: Morton bins chunk a 1D bead string,
   so uncapped representatives elongate ~K× more per level and flare when
   viewed end-on (the "haphazard brightness/hue pops between levels" bug).
3. **Assembles** a `kind=lod` group: coarse gsplat children (coarsest-first) +
   the original Lines node as the finest child; `display_type="lines"`.
   Thresholds are auto-derived through `group.derive_coverage_fractions`
   (screen-occupancy halving to the whole-object anchor `0.5`, or that × 2 in
   area units — finest `1.0` — when the insertion point is inside a
   `kind=partition`; see the table above). The per-level counts handed to it are
   the full lifted **bead** counts, not vertex counts, so the two agree on one
   currency — though the derivation reads only their NUMBER, not their values —
   and there is no method selector or per-dataset anchor knob; pass explicit
   `coverage_fractions=[...]` in the `substitutive_lod=` spec to override.

Composes with `additive_lod` (laddered per level by default, as for Points);
mutually exclusive with `partition`. The synthesized coarse gsplat children keep
their additive ladders for every `line_type` — none of the suppression reasons is
a property of theirs. The original finest Lines child skips its ladder whenever
`image_labels` is set, for a single-polyline `line_type` (`polyline`/`loop`,
which cannot be split without breaking its segment topology), and for `indexed`
**only when its topology does not permit one**, which
`indexed_components_are_chains` decides. The additive multi-LOD writer carries no
edge list — it rebuilds one by chaining each connected component in ascending
vertex order — so a chain is faithful exactly when every component's undirected
edge multiset, including duplicate multiplicity, equals its consecutive-vertex
pairs. Real tractography and streamline
sets satisfy that and are laddered; a branching, cyclic,
out-of-ascending-order, or duplicate-edge component is refused (a `UserWarning`
when the ladder was explicit, an info line when it was the default), and only
that finest child then loads all-at-once. Note a gap in a producer's vertex
numbering is NOT a problem:
two index-contiguous but unconnected runs are two components, chained
separately, so no edge is invented across the gap. On the DIRECT
`add_lines(additive_lod=…)` path, a non-qualifying set raises when the resolved
ladder has multiple levels; a one-level result falls through to the flat writer
and preserves the authored edges. Partition preflight remains deliberately eager
so invalid indexed input fails before any part is written. Before these checks,
the writer fabricated edges silently. `scalars`+`colormap` are
mapped per bead (scalar interpolated along each segment, *then* the LUT — matching
the line shader's interpolate-then-LUT order; same colormap/gamma caveats as
Points, and the same uniform-vs-per-element RGBA rule: a uniform colour is
broadcast to the beads with its alpha, a per-element `(N, 4)` is refused —
uniformity is judged once, per VERTEX, so a line set that collapses to a single
bead cannot re-present a per-element colour as a uniform row). All `line_type`s
(segments/polyline/loop/indexed) are supported for the substitutive pyramid
itself, and all four receive composed additive ladders on their synthesized
coarse children; on the finest child, `segments` always receives one and
`indexed` does whenever its components verify as ascending chains.
Degenerate-width segments are dropped; bead allocation is bounded both
per-segment (`lift.MAX_BEADS_PER_SEGMENT`) and in aggregate
(`lift.MAX_TOTAL_BEADS`, spacing widened to fit with a `UserWarning`), so a
zero/tiny-width line — or a large set of long thin ones — degrades the tube
rather than OOMing. The viewer
lazily defers + evicts the finest Lines lod child (like `gsplats`/`points`) via
`loadLinesNodeCheap`/`loadLinesNodeExpensive` + `releaseLazyLines`.

### GSplats (`gsplats.py`)

GSplats are the only geometry with a stored substitutive pyramid, so it has two
resolvers:

- `resolve_substitutive_axis_gsplats(data, spec)` — the `lod_group=` axis.
  Returns `(resolved_data, explicit_coverage_fractions_or_None)`.
  `explicit_coverage_fractions` is non-None only when the user passed
  `dict(coverage_fractions=[...])` (strict-ascending, in
  `[0, MAX_COVERAGE_FRACTION]` = `[0, 4]`) —
  otherwise downstream code auto-derives per-level thresholds via
  `group.derive_coverage_fractions` (screen-occupancy halving to the whole-object
  anchor `0.5`, re-anchored at fills-screen `1.0` when the insertion point is
  inside a `kind=partition`; the per-level splat counts set only the ladder's
  length). There is
  no method selector or per-dataset anchor knob: a whole-object ladder shows its
  finest level while the object occupies at least half the SCREEN AREA — any
  normal full-frame view. `None` auto-keeps a
  multi-substitutive pyramid (routes to the `kind=lod` builder, no work
  discarded); `False` collapses to the finest level (index 0); `True`
  requires a stored pyramid; `dict(...)` reuses a stored pyramid or computes
  one via `gsplats.lod.substitutive.make_substitutive_lod` (canonical default
  `levels=3`), with `recompute=True` forcing recomputation.
- `resolve_additive_axis_gsplats(data, spec)` — the `additive_lod=` axis,
  applied independently per substitutive level. `dict(...)` computes a ladder on
  any level missing one via `gsplats.lod.additive.make_additive_lod` (default
  `n_lods=4`); `False` flattens each level to a single additive sub-LOD.
  **Trap:** "missing one" means `n_additive_lods <= 1`, and
  `GSplatData.combine_as_new_dimension` MERGES its sources' ladders (rung *i* of
  every source becomes rung *i* of the stack) rather than dropping them — so on a
  stacked dataset built from already-laddered per-timepoint fits the spec is
  silently a NO-OP and the merged per-source ladder is what ships. Pass
  `recompute=True`. The store's tell for the shadowed case is an
  `additive_0/lod_stats` carrying `n_sources` and no `lod_method` (#2485).
- `resolve_additive_rungs(spec, *, stored_rungs, n_splats)` — the same vocabulary
  stated a second time, ordering-free: it answers only "how many rungs would this
  leave on one leaf?", without building anything (#1632). `None` means UNKNOWN
  (energy-fraction breakpoints, or a spec the resolver will reject on its own
  terms), and it is a statement about the SPEC, never about the input — a caller
  that cannot read the kwarg must fall back to what it already knows, not assume
  the ladder went away. It exists for gates that must know whether a ladder will
  exist before the data does; the live one is the file/graft partition-vs-ladder
  gate, which runs above `graft_gsplat_node`'s wrapper build. The two functions
  are one contract stated twice — change either and you change both.

Convention throughout: the **finest** substitutive level is index 0; LOD-group
children are stored coarsest→finest (finest last).

### Mesh (`mesh.py`)

The one geometry whose resolvers are NOT wrappers over the shared ones, because a
surface coarsens by decimation rather than by reducing a Gaussian mixture. Both
axes therefore have a narrower vocabulary, and the keys they refuse are refused
**by name with the reason** — every one of them is a reasonable thing to have
tried after reading the Points docs.

- `resolve_substitutive_axis_mesh(spec)` — the `substitutive_lod=` axis. Keys:
  `compression_factor` (`K`), `levels` (`n_lods`), `method`
  (`{'auto', 'cluster', 'qem'}`; `auto` uses QEM through 10,000 vertices), `coverage_fractions`,
  `coarsen_dims`. Refuses `truncation_radius` / `max_aspect` / `device` / `seed` —
  all four exist only for a lift to gsplats. `coarsen_dims` is the authoring name
  for the decimator's `spatial_dims`. `add_mesh_substitutive_lod_wrapper_impl`
  (`adders/mesh.py`) then decimates via `luxar.mesh.decimate.decimate_ladder`
  (sharing one QEM collapse sequence for a multi-level ladder) and
  assembles a `kind=lod` group whose finest child is the original surface.
- `resolve_additive_axis_mesh(spec)` — the `additive_lod=` axis. Keys: `method`,
  `n_lods`, `counts` (alias `breakpoints`), `reveal_center`, `spatial_dims`.
  `MESH_ADDITIVE_METHODS` is `{"radial"}` and that is the whole design: a prefix of
  an arbitrarily ordered index buffer is a surface with **holes**, not a coarser
  one, so `random` / `salience` and the two samplers are refused with that
  argument, as are `salience_kind` (a triangle has no independent energy) and
  `seed` (a reveal is deterministic). `'energy:'` breakpoints are refused for the
  same absence — left alone they would silently degrade to equal-count splits.
  `make_additive_lod_mesh(...)` returns DISJOINT **face**-index groups, coarsest
  (innermost shell) first, whose union is every face exactly once; the adder
  re-indexes each through `luxar.mesh.split.split_mesh_by_faces` and
  `add_mesh_multi_lod_wrapper_impl` writes them as `additive_<i>/` levels.

Two consequences of the reveal-only restriction, which is the tell that it is the
right cut rather than a convenient one:

1. **No energy stamps, by construction.** `additive_level_stats` already suppresses
   `energy_fraction_cum` / `reference_energy` for every reveal method, so the
   viewer's `1/e(k)` brightness compensation — gated on the BLENDING MODE and never
   on geometry type — cannot reach a mesh ladder and blow out its innermost shell.
   Restricting the method set IS the enforcement; nothing has to remember to
   suppress anything.
2. **Vertex duplication stays far below the unwelded worst case.** Each level
   re-indexes its own vertices, so a boundary vertex is stored once per level that
   touches it. A connected patch has ONE boundary curve, so duplication scales with
   that curve; a random order duplicates nearly every interior vertex. Measured at
   4 levels, reveal vs random over the same faces: 288-face plane **1.66 vs 2.95**,
   320-face icosphere **2.67 vs 3.31**, 1280-face icosphere **1.69 vs 3.25**.

**How the ordering earns that.** It is not `argsort(radius)`. A radius sort keeps a
prefix connected on a convex blob and fails on a closed surface — the shape this
library actually targets — because every centroid sits at nearly the same radius, so
the order is decided by noise spread over the whole shell. Measured edge-connected
components of each cumulative prefix under a plain radius sort, `n_lods=4`: `20 / 20
/ 1 / 1` on a 1280-face icosphere, i.e. a half-loaded sphere as twenty patches of
lace. `compute_additive_order_mesh` therefore grows **best-first through face
adjacency**, keyed on radius: the frontier only admits a face touching one already
admitted, so every prefix is a connected patch on any topology, and the radius key
is what makes it a reveal rather than an arbitrary flood. It degenerates to the
radius sort exactly on the convex case the sort already handled. Cost, stated: a
Python heap loop, order a second per 100k faces, at authoring time only.

Neither axis composes with the other or with `partition=` yet — each pairing is
refused by name in `adders/mesh.py`, where Points and Lines compose both. Labels
are refused on the additive axis only (the adder degrades to a plain leaf with a
`UserWarning`): a level re-indexes its own vertices, so there is no single index
space for the union label CSR the sibling ladders write on their parent.

## Samplers

### `spatial_uniform.py` — `stratified_grid_order(positions, n_lods)`

Assigns elements to LOD levels by which grid resolution first "covers" their
position. Level `i` uses a `2^(i+1)`-per-axis grid (LOD 0 = 2×2×2, LOD 1 =
4×4×4, …); the first unassigned element in each newly-occupied cell joins that
level. Any cumulative prefix of the resulting permutation has approximately
uniform spatial density. Returns `(permutation, per_level_counts)`. Pure NumPy,
`O(N · n_lods)`. Uses only the first 3 spatial dims; the exponent is clamped at
31 to avoid int64 overflow.

### `poisson_disk.py` — `poisson_disk_order(positions, n_lods, seed=0)`

Opt-in blue-noise alternative (Bridson, SIGGRAPH 2007). Runs progressively
finer radii (`r_i = (diag/2) · 0.5^i`); each level keeps the points its radius
selects that no coarser level already took. A cell grid **of accepted samples**
sized at `r/√3` keeps the rejection test to a local 5×5×5 neighborhood, which is
what makes it `O(N)` per level — measured flat in cost-per-point from 10K to 1M
(25 µs/point at `n_lods=6` on an Apple M-series core, ~100 µs/point on a slower
x86 one, or roughly 25 to 100 seconds at 1M; the walk is interpreted, so the
constant is hardware-bound).
Bucketing every input index there instead is quadratic and was the shipped
behaviour until #2530 (~4.3 h at 1M points); `test_cost_grows_linearly_with_n`
is the regression gate.
Same `(permutation, per_level_counts)` contract as the stratified sampler, so
`make_additive_lod_*` stays symmetric across methods; the last level absorbs any
points the finest pass rejected.

## Usage

```python
# Points: 4-level energy-weighted additive ladder
scene.add_points(
    "cloud",
    positions,
    colors=colors,
    radii=radii,
    additive_lod=dict(method="salience", salience_kind="energy", n_lods=4),
)

# Points: explicit cumulative-count breakpoints, spatial-uniform ordering
scene.add_points(
    "cloud",
    positions,
    additive_lod=dict(method="spatial-uniform", counts=[1500, 8000, 40000]),
)

# GSplats: keep the stored substitutive pyramid as a kind=lod Group,
# and add a 4-level additive ladder per level
scene.add_gsplats_from_data(
    "splats",
    gsplat_data,
    substitutive_lod=True,  # require/use stored substitutive levels
    additive_lod=dict(n_lods=4),
)
```

## See Also

- [../../README.md](../../README.md) — `luxar.core` overview (Group, Node, Points, Lines, GSplats)
- `../partition.py` — the sibling `kind=partition` group, which shares
  `resolve_display_type` from `group.py`
- `luxar.gsplats.lod` — `make_substitutive_lod` / `make_additive_lod` builders
  the gsplats resolvers delegate to
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — the v3.4 node-tree format (leaf / kind=lod / kind=partition)
