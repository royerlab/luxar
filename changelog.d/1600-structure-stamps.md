#### `flatten` / `decimate` / `partition` no longer advertise a topology they don't have

The other half of #1600. Every rewriting command threads the input's `stats`
through to the output's `fitting/` / `provenance/` / `pipeline/` groups, which is
right for the fit provenance and wrong for the *topology* record when the rewrite
changes what shape the artifact is. On a 400-splat fit put through `gsplat lod
--recipe levels -K 4 -L 3`, all three of `flatten`, `decimate` and `partition`
published their input's `pipeline/` group verbatim: `lod_kind: substitutive`,
`n_substitutive_levels: 4`, `compression_factor: 4`, `coverage_inflation: 3.0`,
`lod_n_lods: 4`, `lod_cutpoints: [2, 4, 5, 7]`, `recipe: levels`. The `flatten`
and `decimate` outputs are one flat leaf; the `partition` output is four bare
parts. Every one of those keys is false of them, and `gsplat info` reads them as
statements about the store it is pointed at.

`_data/filtering.py` now carries a third hygiene axis beside the region- and
content-scoped ones, because the three are independent and no predicate covers
more than one: a crop invalidates the source grid, a cull invalidates the measured
scores, and only a change of *structure kind* invalidates the topology.
`_STRUCTURE_SCOPED_STATS_KEYS` is enumerated from the producers rather than from a
reader's expectations — `make_substitutive_lod` (`lod_kind`,
`compression_factor`, `method`, `n_substitutive_levels`, `coverage_inflation`,
`conserve_mass`, `refine`, `refine_iters`), `make_additive_lod`'s ladder summary
(`lod_method`, `lod_n_lods`, `lod_breakpoints_kind`, `lod_cutpoints`,
`lod_substitutive_level`), the `recipe` stamp on the `gsplat lod` write path, and
`_recipe_pipeline_info`, the only site that stamps the `batch-fit merge` per-part
knobs (`per_part`, `n_lods`, `breakpoints`, `levels`, `additive_ladders`). A
completeness test closes the registry against those producers in both directions,
measuring what they actually stamp rather than restating the constant, so a new
stamp that is neither classified nor exempt goes red.

`flatten` and `partition` apply the rule in the command, because the domain
methods they build on (`flattened()`, `concatenate`, `to_spatial_partition`) have
other callers for which the record is still true — a recipe flattens a level it is
about to re-wrap in a `kind=lod` group. `decimate` is the opposite case and scrubs
inside `luxar.gsplats.lod.decimate` itself: its contract is one flat leaf whatever
it was handed, so no caller can want the topology kept, and `CLAUDE.md` advertises
that function as public API — a CLI-only scrub left
`decimate(GSplatData.load(pyramid), target=50).save(out)` publishing all nineteen
keys. The no-op path (`target >= n_splats`, which returns the input verbatim) is
deliberately untouched: nothing changed, so nothing is invalidated.

It is a deny-list of key names rather than "drop the `pipeline/` group", because
that group is shared and two of its tenants must survive. The normalization block
(`floor`, `image_min`, `image_max`, `intensity_range`) describes the *input
volume's* intensity scale, which regrouping splats cannot change. And
`coarsen_dims` is load-bearing: `write_gsplats_tree` derives its chunk-ordering
barrier axes from that key's complement, so a naive scrub would not merely delete
a stamp — it would silently change the output's chunk layout, dropping the barrier
on a stacked time/channel axis and smearing every chunk across timepoints. That
one is pinned by a test asserting on the written ordering attrs, with a
half-integer time grid so auto-detection cannot supply the barrier by accident.

Structure-*preserving* rewrites are deliberately untouched: `cull`, `filter`,
`slice`, `transform`, `reencode` and `additive` all leave a substitutive pyramid a
substitutive pyramid, so the *kind* stays true of their output and the scrub
correctly does not apply to any of them. `cull` and `filter` also re-stamp the
counts that moved: on a four-rung `[2, 4, 5, 7]` root ladder both rewrote the
cutpoints, and a reduction that emptied rungs took `lod_n_lods` down with them
(4 → 1 for `cull -r 0.20`, 4 → 2 for `filter --amplitude-min 0.95`).
`additive` does not: re-laddering the same pyramid with `--n-lods 6`
writes `lod_n_lods: 6` into every level's `level_stats` while the root `pipeline/`
group still publishes the input's `lod_n_lods: 4` / `lod_cutpoints: [2, 4, 5, 7]`
/ `lod_method: greedy`. That is a stale-*value* defect within a preserved kind,
which this change does not address — a different rule from the one added here,
tracked separately on #1600. `merge` publishes no inherited provenance at all,
and `batch-fit merge` stamps its own record fresh.
