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
**Content-scoped** covers every MEASURED number, in all three scopes it can live
in: the top-level scores (`psnr_db`, `ssim`, `mse`, `rel_l2`, `max_abs_error`, the
`foreground_*` trio, the optimizer's `final_loss` / `final_rel_l2` /
`final_max_abs_error`, the error-budget cull's `error_budget` / `max_joint_error`
and its search counters), the per-sub-LOD `lod_stats` (`cumulative_psnr_db` /
`delta_psnr_db` from a progressive fit, and `energy_fraction_cum`), and the
per-level `level_stats` (`quality`, `reference_energy`). It is dropped whenever the
splat set changes, spatially or not. **Region-scoped** is unchanged. Neither
subsumes the other: a non-spatial cull loses the scores and keeps the source grid,
a whole-volume bbox that excluded nothing keeps both, and a real crop loses both.

`energy_fraction_cum` was the one stale stamp that was *rendering*-visible. The
viewer brightens an incomplete ladder inside a `kind=lod` group by `1/e(k)`, so a
rung still claiming its pre-cull `e(k)` after the cull got the wrong compensation —
in the worst case (a rung that ends up holding *everything* while still claiming
0.69) a fully-loaded level rendered ~1.44x too bright. On the reproduction,
`cull -r 0.9` over a 3-rung `stream` ladder left rung 0 stamped 0.681 where the
truth for the culled ladder is 0.7035 (`gsplat annotate-quality` re-measures it as
exactly that). It is dropped together with the `level_stats.reference_energy`
weight it is aggregated by — `lod/additive.py` calls that pair a contract — and
both degrade correctly when absent: the viewer's energy compensation is exactly 1
without a stamp, its display gate falls back to committed-count comparison, and
`annotate-quality` restamps e(k)/w (and Q with `--with-quality`) on the rewritten
store in place. The structural per-rung counts beside them (`lod_n_splats`,
`lod_cumulative_n`, `n_splats_total`) are *re-stamped* from the result rather than
dropped: a reduction makes them wrong, not unknown, and the writer stores them
verbatim.

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
`time_seconds`, `fitter_name` and each operation's own record (`culled`,
`culling_method`, `n_original`, `n_culled`, `amplitude_retention`,
`filter_criteria`) describe a run or an edit that happened. The nested per-pass
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
