#### `flatten` / `partition` / `lod` / `decimate` no longer advertise a topology they don't have

The other half of #1600. Every rewriting command threads the input's `stats`
through to the output's `fitting/` / `provenance/` / `pipeline/` groups, which is
right for the fit provenance and wrong for the *topology* record when the rewrite
changes what shape the artifact is. On a 400-splat fit put through `gsplat lod
--recipe levels -K 4 -L 3`, all four of `flatten`, `partition`, `lod` and
`decimate` republished their input's `pipeline/` group — all fourteen topology
keys it carried, `lod` alone overwriting one of them (`recipe`): `lod_kind:
substitutive`, `n_substitutive_levels: 4`, `compression_factor: 4`,
`coverage_inflation: 3.0`, `lod_n_lods: 4`, `lod_cutpoints: [2, 4, 5, 7]`,
`recipe: levels`. The `flatten` and `decimate` outputs are one flat leaf; the
`partition` output is four bare parts. Every one of those keys is false of them,
and `gsplat info` reads them as statements about the store it is pointed at.

`lod` is the same defect one level up, and the one that is easiest to miss
because the command *does* publish a topology record — just not only its own. It
loads its input with stats and refuses nothing but a `kind=partition`, so an
already-LOD store is a legal input, while every recipe begins with
`data.flattened()`; no `lod` output preserves its input's shape. Measured on the
same store, `--recipe flat` — literally `flatten`'s twin, since `build_flat` is
`return data.flattened()` — emitted a single flat leaf publishing thirteen
inherited topology keys one line above its own `recipe: flat`. `--recipe
stream` re-stamped the ladder half correctly (`lod_n_lods`, `lod_cutpoints`,
`lod_breakpoints_kind`, `lod_method`, `lod_substitutive_level`) and inherited the
substitutive half unchanged, which is exactly what made it look handled. And
`tiles` / `overview` / `adaptive` pushed the whole block through the composed
node-tree write path onto a `kind=partition`. The scrub goes in once, on the
loaded input, which fixes both write paths at their common source: the matrix
builders derive `result.stats` from `dict(src.stats)` and then stamp their own
record over it, and the composed path hands the same dict to
`split_fitting_info`. After it each recipe publishes exactly what its own builder
stamped — `recipe` alone for `flat` / `tiles` / `overview` / `adaptive`, plus the
ladder summary for `stream`, plus the substitutive block as well for `levels` —
and nothing positive is invented to fill the gap, because an absent `lod_kind` is
the format's "this artifact does not know" and stamping `lod_kind: additive` on a
`stream` output would be a new claim rather than a scrub. Safe because nothing in
the build path reads a topology key to decide anything: the only reads are two
per-sub-LOD `lod_stats.lod_method` lookups (`lod/annotate.py`, `lod/restamp.py` —
a different dict) and `lod_substitutive_level` inside `_map_substitutive`, which
no recipe reaches; `flattened()` copies stats without reading them; and
`coarsen_dims` is exempt, so a `levels` build keeps its barrier provenance.

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

A second guard closes the *command* table the same way, and it is why `lod` is in
this entry at all: the first pass listed the three commands someone thought of,
with nothing comparing that list against the commands that exist —
the same hole through which `merge` slipped four passes of #1600's
appearance-carry audit. `test_every_gsplat_command_is_classified` now reuses that
audit's command enumerator (imported, not copied) and requires every registered
`gsplat` command to be either a `_KIND_CHANGING` row — which means *fixed*, since
each row also pins the exact topology keys that command stamps itself — or a
`_NOT_KIND_CHANGING` entry with a one-line reason. There is deliberately no
"known-broken" bucket: offering one is how a command gets to stay broken. The
`lod` rows are per *recipe*, because the recipe is the axis the defect hid on.

`flatten`, `partition` and `lod` apply the rule in the command, because the domain
methods they build on (`flattened()`, `concatenate`, `to_spatial_partition`, the
recipe builders) have other callers for which the record is still true — a recipe
flattens a level it is about to re-wrap in a `kind=lod` group. `decimate` is the
opposite case and scrubs inside `luxar.gsplats.lod.decimate` itself: its contract
is one flat leaf whatever it was handed, so no caller can want the topology kept,
and `CLAUDE.md` advertises that function as public API — a CLI-only scrub left
`decimate(GSplatData.load(pyramid), target=50).save(out)` republishing every one
of its input's topology keys, fourteen of them for a `levels` pyramid built at the
defaults. (The registry names nineteen; the remaining five — `per_part`,
`n_lods`, `breakpoints`, `levels`, `additive_ladders` — are stamped only by
`_recipe_pipeline_info` on a `batch-fit merge` output, so a `lod`-built store
never carries them and the count you see on disk depends on what wrote it.) The
no-op path (`target >= n_splats`, which returns the input object verbatim rather
than a flat leaf) is deliberately untouched: nothing changed, so nothing is
invalidated.

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
