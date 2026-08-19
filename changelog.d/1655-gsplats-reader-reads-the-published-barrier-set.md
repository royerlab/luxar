#### The gsplats reader now reads the barrier set the writer published, instead of re-deriving it (#1655 items 2 and 3)

`compute_chunk_bounds_gsplats` expands a chunk's bounds by `truncation_radius · σ`
on every dimension _except_ the ones it was told are barriers (time, channel), which
get only a tight `_BARRIER_BOUND_EPS` float-boundary pad. The viewer's chunk-fetch
tolerance depends on that split in opposite directions — a σ-expanded dimension wants
a near-zero float-safety epsilon (anything wider is pure over-fetch), a barrier-padded
one wants the quarter-cell reach — and the reader was deciding which was which on its
own, from `dimensions[d].discrete`.

The two agree whenever the scene declares its dimensions, because
`geometry_writers/gsplats.py::scene_barrier_dims` uses exactly `discrete and not
display`. They need not agree otherwise: with `scene_dimensions` absent (a standalone
`.gsplats.zarr` grafted into a scene) the writer falls back to the value-based
`detect_barrier_dims`, whose own docstring calls a false positive a correctness bug.
A really-spatial axis with integer, low-cardinality values then gets barrier-tight
bounds while the reader, seeing it as continuous, queries with a ~1e-3 epsilon — and
a spatially extended splat near a chunk edge is simply never fetched. The old
`step × 3.0` tolerance was wide enough to mask that; the epsilon that replaced it in
#1183 is not, which turned a latent disagreement into a documented precondition.

The writer already publishes the exact set it used, as the `slice_dims` group attr,
on every ordered gsplats leaf and every lightweight `additive_<i>` sub-LOD alike. The
loader now reads it and hands it to the tolerance computer as the authoritative
barrier set, so barrier-ness is settled by the party that made the decision rather
than inferred twice. An empty list is a real answer ("I ordered purely spatially,
nothing is a barrier") and is honoured as one. Barrier-ness is resolved in one shared
place, `isBarrierDim`, which honours the published set for GSPLATS ONLY. That scoping
is deliberate rather than an accident of who calls it: reading the set picks the rule
matching the chunk BOUNDS, which can narrow a window the renderer still draws, so each
arm owes its own floor first and only the gsplats arm has one. Lines' continuous arm is
a literal `0` (while its projection path keeps clipping against a half-cell slab) and
both of mesh's arms are membership gates, where narrowing changes what the user sees;
wiring either up is therefore a deliberate edit that has to add that floor. Points
classify from their own `spatialExtendDims` and are untouched.

Reading the published set decides which rule matches the chunk BOUNDS, and that is
all it decides. The per-splat gate for a hidden dimension is still built from the
scene's `discrete` flag (`data-processor-gsplats.ts` → the projection kernel's
`|Δ| ≤ step × 0.5` test), and a dimension in that set is routed to the binary
half-cell gate _only_ — `classifyHiddenDims` keeps it out of the continuous group, so
the Gaussian attenuation never runs on it. A dimension the writer OMITTED while the
scene declares it discrete therefore gets the half-cell as its FETCH window too, the
one place where a membership window and a chunk-fetch window have to be the same
number. Not the quarter-cell reach the barrier arm uses: that fraction is `0.25`
rather than `0.5` only because legacy stores pad barrier bounds by ±0.5 step and pad
plus reach must stay under a full step, and a dimension the writer never
barrier-padded has no such pad — a quarter would still have missed splats sitting 0.3
off a `step: 1` grid (what `gsplat merge --as-dimension --values 0.3,1.3,2.3` writes),
which are inside the render gate and outside a 0.25-cell window. The BARRIER arm keeps
that `(0.25, 0.5]` gap, deliberately: the pad budget is real there, and off-grid values
on a scene-declared discrete axis are a write-time authoring fault the compiler already
reports (`io/_compiler/finalize/validation.py::validate_discrete_dimension_ranges`
warns when such an axis's declared range sits more than a quarter step outside its
data).

Without that case, the dimension's fetch reach would have narrowed ~500× at `step = 1`
while the renderer still drew a half cell, and most of the node would have
disappeared — not all of it: with `slice_dims: []` the compound sort is a pure
spatial curve over every center column, so a chunk straddling two adjacent stacked
values still spans the query and matches, and what is lost is every chunk sitting
entirely inside one value. It also needs the scene's declared grid to be misaligned
with the stored values, since the query snaps to a multiple of the step while the
splats sit where they were written; an axis whose values are multiples of its step has
offset 0 and matches even on the bare epsilon.

The path that actually reaches the arm is the STANDALONE OPEN, not a graft. Serving a
`.gsplats.zarr` directly (`?src=….gsplats.zarr`, which the writer enables by stamping
`layer` on the root) leaves the scene with no `scene_dimensions`, so
`load-scene.ts::synthesizeSceneDimensionsFromNode` marks every axis ≥ 3
`discrete: true, step: 1` regardless of the stored values, while the store publishes
whatever its own value-based `detect_barrier_dims` found — `[]` for a stacked axis whose
values are not near-integers. Scene-discrete, writer-omitted: demote. GRAFTING such a
store into a scene does not reach it, because `add_gsplats_from_file_impl` routes
through `add_gsplats_from_data_impl` / `graft_gsplat_node`, both of which write through
the scene gsplats writer, which re-sorts and RE-STAMPS `slice_dims` from
`scene_barrier_dims` (`discrete and not display`) — so an axis the scene declares
discrete and non-displayed comes back inside the set and lands on the barrier arm.

The promote direction needs no such window, and not because it only widens: it
replaces `max(1e-3 × step, T × 1e-5)` with `0.25 × step`, which is narrower whenever
`step < 4e-5 × T` (≈1.1e-4 at the default radius — a 110× narrowing at
`step = 1e-6`). It is safe because `_BARRIER_BOUND_EPS = 1e-3` is an ABSOLUTE pad, not
a step fraction: the writer, having listed the dimension, already extended its bounds
by 1e-3 either side, which dwarfs the `T × 1e-5` band the renderer can draw there for
any `T < 100`, so even a zero tolerance would match. Being absolute is also why that
argument cannot be ported to a geometry whose bounds are not barrier-padded.

Be precise about what this fixes. The writer can still pick the _wrong_ barrier set —
that is a write-side bug and it still produces bounds tighter than the splats' true
extent. What is gone is the reader independently disagreeing with whatever the writer
chose, which is the half that was silent and unfixable from either side alone. A
legacy store that publishes no `slice_dims` still falls back to `discrete`, and the
attr is validated all-or-nothing: anything that is not an array of integers in
`[0, ndim)` is discarded whole. Filtering it element-wise would drop a genuine barrier
dimension and _narrow_ the fetch window — the exact failure the plumbing exists to
prevent — whereas falling all the way back reproduces the previous behaviour exactly,
so no store ends up worse off than before the attr was read.

The same call site closes the second, smaller gap. The tolerance's degenerate-band
term is `truncation_radius × 1e-5`: the read side regularizes an all-zero hidden
covariance block to a pivot of `sqrt(CHOLESKY_EPSILON)`, so such a splat still renders
out to that distance, and the fetch window has to cover it. The term was pinned to the
default radius of 2.75 because the computer only ever saw a `DimensionInfo`, leaving
`(2.75e-5, T × 1e-5]` uncovered for a node stamping a larger `T` — little visible was
lost in practice (the old window edge sits at 2.75 σ of the regularized pivot, where
attenuation is ≈2.3%), but the limit was real. The node's
own radius now travels with the query, sanitized through the same
`clampTruncationRadius` the material path uses, so the band the query covers is by
construction the band the renderer draws rather than a second, independently-derived
number. That clamp gained a non-number branch to make the identity hold for an attr
that is not a number at all: `truncation_radius: "6"` passes every numeric test in it
by coercion (`"6" * "6" === 36`), so the string used to be handed straight to the
`uTruncate` uniform — a 6σ material band against a 2.75σ fetch band, the very gap
being closed. A non-number now falls back to the default on both paths.

Tying the band to the node's radius also gives it no ceiling, and that is documented
rather than capped. A node stamping `truncation_radius: 1e18` — finite, and a legal
authored value on both sides of the contract, since `clampTruncationRadius` and the
write-side `MAX_TRUNCATION_RADIUS_FLOAT32` share the same `sqrt(float32.max)` ≈ 1.84e19
bound — yields a fetch window of ~1e13, i.e. the whole node. Capping it would be the
wrong fix, because the material draws that same band: a cap would re-create exactly the
fetch-narrower-than-render under-fetch this change removes. The remedy is not to author
such a node.
