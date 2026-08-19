#### A rewritten `.gsplats.zarr` no longer publishes the fit's PSNR as its own

`gsplat info` reads `psnr_db` as *the* reconstruction quality of the dataset it is
pointed at, and every command that rewrote a store carried the fit's measured
metrics over unchanged. On a small synthetic fit, `cull -m cumulative -r 0.5` kept
13 of 40 splats and still reported `psnr_db: 44.4587` / `foreground_psnr_db:
32.8791` — a score measured on a splat set that no longer exists. `n_original` and
`n_culled` were recorded beside it, but nothing marked the metric stale, so the
reading was simply wrong rather than merely unqualified. `filter`, `slice` and
`transform --scale-intensity` / `--normalize-intensity` had the same hole.

The root cause was a missing category. `_data/filtering.py` had exactly one
pass-through rule — the *region*-scoped source-grid stamps, dropped when a
bbox/slice crop actually excluded splats — and the measured metrics were in
neither category. The crop predicate is the wrong one to reuse for them: an
amplitude-threshold cull leaves the represented region untouched while changing
the reconstruction completely. There are now two categories with two predicates.
**Content-scoped** covers every number MEASURED against the source volume, in
both scopes it can live in: the top-level scores (`psnr_db`, `ssim`, `mse`,
`rel_l2`, `max_abs_error`, the `foreground_*` trio, the optimizer's `final_loss` /
`final_rel_l2` / `final_max_abs_error`, the error-budget cull's `error_budget` /
`max_joint_error` and its search counters) and the per-sub-LOD `lod_stats` a
progressive fit stamps (`cumulative_psnr_db` / `delta_psnr_db`). It also covers
the record of the reduction that produced the artifact — `culled`,
`culling_method`, `n_original`, `n_culled`, `amplitude_retention` — which is true
of the op that stamped it and false of anything downstream: a `decimate` of a
culled store published the input's `amplitude_retention: 0.95` beside a prefix
reduction that had just discarded ~75% of the amplitude mass. Every op that stamps
one of those keys does so after its own filter, so a rewrite publishes the
reduction it actually performed. **Region-scoped** is unchanged. Neither subsumes
the other: a non-spatial cull loses the scores and keeps the source grid, a
whole-volume bbox that excluded nothing keeps both, and a real crop loses both.

The LOD **Q·e ladder stamps** are a known separate case, deliberately left alone
here: `lod_stats.energy_fraction_cum` (a rung's prefix energy e(k)),
`level_stats.reference_energy` (its weight w) and `level_stats.quality` (a level's
measured Q against its group's finest). Unlike the scores above they are measured
on the artifact's *own* content rather than against a source volume, so a coarse
level's stamps are statements about that coarse level — and the scene-authoring
path reaches them through the same `at_substitutive` accessor a reduction uses,
copying them onto every coarse child of a `kind=lod` group, so scrubbing there
silently stripped the viewer's `e(k) >= 0.6` early-upgrade release and its
`1/e(k)` brightness compensation from every coarse level. Deleting w is not free
either: `annotate-quality` writes a leaf-local `reference_energy` only when none is
present, so removing it licenses a fabricated, group-inconsistent value that then
looks correctly stamped. A reduction does make these stale, and the likely right
answer is to *recompute* them (cheap, O(N), no volume — exactly what
`annotate-quality` already does) rather than to drop them; that gets its own
change.

The scrub is wired at the chokepoints rather than at one writer, since some
commands save through `GSplatData.save` and others through `write_gsplats_tree`:
`GSplatData.filter` (which `filter_by`, `slice_by` and every `cull` strategy
funnel through), the multi-substitutive rebuild branches that construct their own
top-level stats, `_with_new_amplitudes` (the amplitude-edit chokepoint — PSNR is
an absolute-error metric, so a global `x0.5` invalidates it), the reduced-LOD
*views* `additive_prefix` and `at_substitutive` (a strict prefix or a coarser,
merged level is a reduction like any other — `lod --recipe overview` builds its
coarse cap as `at_substitutive(n-1).flattened()` and was publishing the input fit's
`psnr_db` on merged representatives), `lod.decimate`, and the `transform`
command's node-tree path, which writes the root `fitting/` group straight from the
stats it loaded. That last one uses the same value-based predicate as the dataset
methods: gating it on the flag's mere presence made `transform --scale-intensity
1.0` destroy a partition's scores while the flat path correctly kept them, so one
store had two answers depending on its shape.

Two commands also stopped throwing provenance away. `gsplat decimate` loaded with
`include_stats=False` and called the tree writer without `fitting_info=`, so its
output had no `fitting/` group at all — it dropped `fitter_name` / `iterations` /
the source grid along with the score, and left the scrub nothing to do; it now
threads the descriptive half through and the rule decides the rest. `gsplat
partition` had the same omission (a known gap) and now publishes the stamp too —
a BSP partition is content-preserving, so its measured scores stay.

Two exemptions are deliberate. On the producing side, every fitter ends with a
high-retention cumulative trim (`cull_retention`, 0.95 by default for the flat and
tiled fitters, 0.98 for the progressive one) applied *after* it scored the
reconstruction. Scrubbing there would have left every default fit with no
`psnr_db` at all — the fit's own console summary would have disagreed with the
store it just wrote — and re-scoring costs a second full render of the volume. The
three fitters therefore snapshot their measurement across that trim and put it
back, in every scope, including the nested per-pass `pass_stats` a progressive fit
publishes. A later `gsplat cull` at a retention the user picked gets no such pass;
that is the bug. In the other direction, a content-planned box fit
(`fit --tiling content`) now publishes *no* score for a part whose crop was
halo-padded or whose core mask dropped splats: that measurement was taken on the
padded crop, with the neighbour splats contributing and against a larger target
region, so it described neither this splat set nor this part's region.

Descriptive provenance is untouched: `iterations`, `best_iteration`, `converged`,
`time_seconds`, `fitter_name` and `filtered` / `filter_criteria` describe a run or
an edit that happened. The nested per-pass
`pass_stats` list is scrubbed key-by-key rather than dropped whole, so the pass
counts survive without the stale per-pass PSNR — and it is scrubbed into a *fresh*
list rather than edited in place, because every call site reaches it through a
shallow `dict(source.stats)` that shares it with the input, and editing the entries
deleted the caller's own ladder scores (`GSplatData` is documented as
conceptually immutable). A **geometry-only** transform (scale / rotate / translate
/ center) keeps the scores because the splat set is identical and only its frame
moved — a weaker claim than reproducibility, which the spec now says explicitly:
`gsplat transform --scale` records no factor and does not update `fitted_shape`, so
the number cannot be re-derived from the artifact afterwards. `flatten`,
`additive`, `annotate-quality`, `partition` and `reencode -e precision` keep them
because they change nothing at all; `reencode -e auto|memory` (and
`migrate-format`) re-quantize the Cholesky factors and the centers to fixed point,
which is a deliberate bounded loss (~93 dB at `memory`, far under the
reconstruction error any of these scores report), so the scores are kept as valid
to well within their own precision. The format spec states the whole rule.
