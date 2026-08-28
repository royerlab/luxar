#### Six embedding demos drop their substitutive LOD for a streaming ladder

`esm3_protein_landscape`, `mouse_multiome_peak_umap`, `zebrahub_multiome_peak_umap`,
`cellxgene_census_umap`, `human_multiome_peak_umap` and `arxiv_papers_kaggle` each
carried three or four coarse replacement levels. None of them needed any.

Every one of these nodes is stacked over its colouring/attribute views on a
hidden dimension, so the slice the viewer makes resident is a fraction of the
stored total — and the total is the wrong number to compare against a ceiling.
Measured resident counts against the **5,591,040-point** Points cap (the cap is
per geometry type; the gsplat 4,194,304 does not apply here):

| demo | stored | resident | headroom |
|---|---|---|---|
| `mouse_multiome_peak_umap` | 1,153,506 | 192,251 | 29x |
| `esm3_protein_landscape` | 1,151,006 | ~575,503 | ~10x |
| `zebrahub_multiome_peak_umap` | 4,485,810 | 640,830 | 8.7x |
| `cellxgene_census_umap` | 3,000,000 | ~1,000,000 | ~5.6x |
| `human_multiome_peak_umap` | 6,248,730 | 1,041,455 | 5.4x |
| `arxiv_papers_kaggle` | 6,572,730 | ~3,286,365 | 1.7x |

And the coarse levels were never selected anyway. All six use
`selector="screen-area"` with the finest level anchored at `coverage_fraction=0.5`
— half-screen occupancy — while opening in cinematic mode with an auto-fit
camera. So the finest level is what the opening pose shows, and the coarse levels
were bytes nobody fetched: 19.7% of `human_multiome`'s 111 MB store (196K + 2.1M
+ 16M), 17% of `mouse_multiome`'s 20 MB.

They are replaced by an additive ladder via a new `_lod_policy.stream_ladder`, so
all six agree on one schedule. Group counts (wrapper + rungs, which is roughly
the bootstrap request count) fall 13→7, 13→7, 17→11, 18→9, 17→13, 18→13.

Two details that were easy to get wrong, both now pinned:

**An additive ladder on a plain leaf is opt-in.** Under `substitutive_lod=` one is
composed in by default, so simply deleting the substitutive spec would have
silently deleted the ladder with it. `scripts/check_demo_ladders.py` is the
backstop.

**The first rung is the ~200 ms download budget (39,062 elements), not desi's
2,000.** desi's figure is sized to land in a single zarr chunk because its *eager
coarsest substitutive level* is what paints first; an additive-only leaf has no
coarse level, so its first rung IS first paint. At 2,000 the group counts come out
12/12/15/14/17/18 — no reduction at all, because every extra rung is another node.
Increments stay capped via `capped_stream_cuts`.

A side effect worth having: `substitutive_lod_or_flat` is gone from these six.
That gate existed because the coarsening write path imports torch and scipy, and
a warm-cache machine without them got a degraded, ladderless scene plus a notice.
An additive ladder imports neither, so the "complete cache runs anywhere" contract
now holds with the ladder intact — the tests assert the structure is *identical*
with either module blocked, rather than asserting a degradation notice.
`cellxgene_census_umap.build_scene` loses its `device`/`compression_factor`/
`levels` parameters and `CENSUS_UMAP_DEVICE` along with them; the build is a write
now, not a compute.

Unchanged, deliberately: `desi_galaxies` (9,751,955 resident with no hidden axis —
genuinely over the cap, and its coarse levels are what bound residency, since
`partition=` alone loads every part), `gaia_milky_way` (orbited at range as well
as inspected close up, the case `_lod_policy` already documents for
`milky_way_dust`), `nuclear_pore_complex`, `ocean_currents_earth` and
`dmri_tractography`.
