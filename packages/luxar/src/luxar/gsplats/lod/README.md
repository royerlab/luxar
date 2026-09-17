# Levels-of-Detail for Gaussian Splats

Post-processing operators that turn a fitted `GSplatData` into a
multi-resolution representation. Two complementary axes are implemented:

- **Additive** (`additive.py`) — same `N` splats, ordered so that the
  prefix sum at any `k` splats is the best $L^2$ approximation. Streaming
  / progressive-refinement use case.
- **Substitutive** (`substitutive.py`) — each coarser level synthesises
  $\lceil N/K^\ell \rceil$ representative splats that *replace* the
  finer level. Real geometry / memory compression. Built on the
  $K$-wise moment-matched merge with $L^2$-optimal amplitude (supp doc
  `substitutive_lod.tex` Prop. 2.1–2.2) plus cost-increment Lloyd
  refinement (Algorithm 4.3).

This module is a **pure post-process**. Fitting (single-pass or
progressive) returns a single flattened `GSplatData`; an LOD hierarchy
is built only on demand.

> **API terms versus CLI recipes.** `additive` and `substitutive` are the
> algorithm/data-model axes used by the Python API and on-disk metadata. The
> `luxar gsplat lod --recipe` CLI uses intent-first names: `stream` for one
> additive prefix ladder, `levels` for a substitutive replacement hierarchy,
> `tiles` for spatial parts with per-part stream ladders, `overview` for a
> coarse global level over fine tiles, and `adaptive` for per-part replacement
> levels. The former recipe names (`additive`, `substitutive`, `partitioned`,
> `multiscale`, `mosaic`) are not accepted by the current CLI.

### Module layout

| File | Role |
|------|------|
| `additive.py` | additive axis: ordering + ladder (`make_additive_lod`, `compute_additive_order`) |
| `substitutive.py` | substitutive axis orchestrator (`make_substitutive_lod`, `merge_to_count` — one level, an exact representative count rather than an integer factor, `_reduce_one_level`, `_pack_level`) |
| `decimate.py` | `decimate` — reduce a dataset to a TARGET SPLAT COUNT, returning one flat `GSplatData` rather than a structure. Wraps the two builders above behind the blunt question "make this smaller": `merge` (a one-level substitutive reduction) vs `prefix` (a prefix of the additive ordering), with `auto` following the measured crossover at 50% kept. Merging leads by 3–4 dB below that, the prefix above it — numbers and method in the module docstring. Stamps the `coarsen_dims` its `merge` actually resolved, always as an EXPLICIT list — coarsen-everything included, because the writer reads a null the same way it reads an absent key (no provenance → auto-detect), which re-imposes a barrier on an axis the merge just blended whenever the reduction leaves that axis' grid intact (widely-separated timepoints, never clustered together) and is merely redundant when the merge averages the grid away — measured both ways, and the explicit list is honest on either. `make_substitutive_lod` (and the `batch-fit merge` per-part record) resolve their own stamp through the SAME `resolved_merge_coarsen_dims`, which lives beside `_normalise_coarsen_dims` in `substitutive.py` so the three paths that WRITE this key cannot spell one choice two ways; `lod --recipe adaptive` / `overview` and `fit --recipe levels` coarsen too but publish no stamp at all (an absent key reads exactly like a `null`), which is still open on #1600. Because the key is exempt from the structure scrub, the explicit list is also inherited by every downstream rewrite of a `levels` store — safe in direction, and true of those outputs, but the finest level (the input unreduced) loses a barrier that would have been legitimate. A `prefix` coarsens no axis, keeps the input's stamp, and raises a `UserWarning` (not a print — a `verbose=False` call stays off stdout, and the CLI still shows it) saying it ignored the request; the request is range-validated for both families before the family is chosen. CLI: `luxar gsplat decimate` |
| `pyramid.py` | `make_lod_pyramid` — chains substitutive (outer) × additive (inner) |
| `recipes.py` | intent-first recipe layer (`build_recipe`, `RecipeParams`): composes the builders above into the `flat`/`stream`/`levels`/`tiles`/`overview`/`adaptive` topologies (CLI `gsplat lod --recipe`) |
| `volume_refit.py` | `refine="volume"`: warm-start re-fit of a coarse level against the source volume (thin orchestration over `fit_gaussian_splats`) |
| `energy.py` | torch-free artifact-local self-energy (`total_self_energy`), the reference weight w in the viewer's recursive Q·e quality algebra |
| `quality.py` | measured approximation quality: `mixture_quality` (constant-cost sampled mixture-L² → Q ∈ [0,1]) |
| `restamp.py` | refresh inherited additive energy/count stamps after a content-changing rewrite; stale `quality` / source-volume `refine_stats` are dropped. Also `refresh_root_ladder_summary` — the ROOT ladder block after a structure-preserving re-ladder (`gsplat additive`), summarising the leaf / the level `lod_substitutive_level` names, and dropped where no single leaf can be the summary. Both spellings of that block: the five `lod_*` keys `make_additive_lod` stamps, and the un-prefixed `n_lods` / `method` / `breakpoints` `batch-fit merge --recipe stream` stamps for the same ladder (`breakpoints` is dropped rather than refreshed — a build SPEC is not recoverable from a ladder) |
| `annotate.py` | `annotate_quality_store` — retrofit the Q·e stamps (`energy_fraction_cum` / `reference_energy` / `quality`) onto an existing `.gsplats.zarr` IN PLACE, no refit (CLI: `luxar gsplat annotate-quality`) |
| `_kernels.py` | shared closed-form Gaussian-mixture math (numpy + torch) |
| `_substitutive/` | private support subpackage for `substitutive.py`: `warm_start.py` (Morton partition), `kmeans_lloyd.py` (cost-increment Lloyd), `greedy.py` (Runnalls lazy-heap merge), `refine.py` (L2 mixture-to-mixture refit) |

The public import paths (`luxar.gsplats.lod`, `lod.additive`,
`lod.substitutive`, `lod._kernels`) are unchanged; the three substitutive
algorithms are grouped under `_substitutive/` (leading underscore because
a module `substitutive.py` and a package `substitutive/` cannot coexist
in CPython).

> **Upstream step**: use `luxar gsplat cal` to pick a principled splat
> budget K\* before fitting. The canonical end-to-end pipeline is
> **`cal` → `fit --seeds K*` → `lod --recipe stream` (or `tiles` / `overview` / `adaptive`)**.
> See `gsplats/calibration/README.md` and the "Calibration (Blind-Spot CV)"
> section in the parent `gsplats/README.md`.

## Quick start

```python
from luxar.gsplats.lod import (
    make_additive_lod, make_substitutive_lod, make_lod_pyramid,
)
from luxar.gsplats.gsplat_data import GSplatData

data = GSplatData.load("fit.gsplats.zarr")          # bare leaf

# ── Additive: same N splats, prefix-monotone
ladder = make_additive_lod(data, n_lods=4)          # GSplatData with 4-sublod additive ladder
ladder.save("additive_lod.gsplats.zarr")            # v3.4 leaf with additive_<i>/ subgroups

# ── Substitutive: ceil(N/K^L) splats per level, replacement hierarchy
pyramid = make_substitutive_lod(data, compression_factor=4, levels=3)
# pyramid is a GSplatData with 4 substitutive levels (index 0 = finest)
pyramid.save("substitutive_pyramid.gsplats.zarr")   # v3.4 kind=lod group
for s, lev in enumerate(pyramid.substitutive_levels):
    print(f"level {s}: {lev.n_splats_total} splats, K={lev.compression_factor}")

# ── Barrier-aware coarsening: restrict merging to a subset of dims.
# `coarsen_dims` lists the center-column indices coarsening may merge over;
# the complement becomes hard grouping barriers (a categorical / timepoint /
# channel axis), so coarse splats never blend across them. None = all dims.
barrier = make_substitutive_lod(data4d, compression_factor=4, levels=3,
                                coarsen_dims=[1, 2, 3])   # group by dim 0
# In the scene API (add_points/add_lines/add_gsplats_from_data) the default is
# Auto: coarsen the displayed dims, group by the non-displayed dims — so nD
# scenes are barrier-correct without specifying anything. The standalone
# `luxar gsplat lod --coarsen-dims i,j,k` takes explicit indices (no display
# metadata exists standalone).

# ── Full pyramid: substitutive (outer) × additive (inner) in one call
full = make_lod_pyramid(
    data,
    compression_factor=4, levels=3,     # substitutive axis (4 levels)
    n_additive_lods=4,                  # additive axis (4 sublods per level)
)
full.save("pyramid.gsplats.zarr")       # v3.4 kind=lod group of additive-ladder leaves
```

## API

```python
make_additive_lod(
    data: GSplatData,
    n_lods: int = 4,
    *,
    method: str = "auto",         # see "Methods" below
    breakpoints = "equal-count",   # or list[int] | list[float]
    truncation_sigmas: float | None = None,   # None = data.truncation_radius
    max_n_dense: int = 2_000,
    seed: int | None = None,
    slice_dims: Sequence[int] | None = None,   # see "slice_dims" below
) -> GSplatData
```

`truncation_sigmas=None` (the default) means the σ support the **dataset itself**
was fitted and is rendered at (`data.truncation_radius`, canonically 2.75) — not a
hardcoded 3.0. It is read off the object actually being pruned, so with
`substitutive_level=` it is that level's radius, not the finest leaf's. Pass a
float to prune at a different support.

```python
compute_additive_order(
    data: GSplatData,
    method: str = "auto",
    *,
    truncation_sigmas: float | None = None,   # None = data.truncation_radius
    max_n_dense: int = 2_000,
    seed: int | None = None,
    slice_dims: Sequence[int] | None = None,   # see "slice_dims" below
) -> np.ndarray   # length-N permutation
```

## Methods

| Method        | Score / rule                                             | Precompute    | Optimal at `k=1` | Submodular guarantee |
|---------------|----------------------------------------------------------|---------------|------------------|----------------------|
| `random`      | uniform permutation                                       | none          | no                | no                  |
| `amplitude`   | sort by $a_i$ desc                                        | $O(N\log N)$  | no                | no                  |
| `mass`        | sort by $a_i \, |\Sigma_i|^{1/2}$ desc                    | $O(N\log N)$  | no                | no                  |
| `self_energy` | sort by $\|\phi_i\|^2 \propto a_i^2 |\Sigma_i|^{1/2}$ desc | $O(N\log N)$  | sometimes         | no                  |
| `spectral`    | sort by $|u_1[i]|$ desc                                   | sparse Gram   | no                | no                  |
| `greedy`      | matching pursuit; sparse / dense fallback                 | sparse Gram   | yes               | yes ($1-1/e$)       |
| `radial`      | sort by distance from the bbox centre **asc**              | $O(N\log N)$  | n/a — a reveal    | n/a — a reveal      |
| `auto`        | size-adaptive: `greedy` if `N ≤ 5000` else `self_energy`  | (see chosen)  | (see chosen)      | (see chosen)        |

### `radial` — the reveal

Every other method above ranks splats by how much they *contribute*, so a prefix
is a low-quality approximation of the whole scene. `radial` is categorically
different: it orders by distance from the object's own bounding-box centre, so a
prefix is a **complete rendering of the inner part of the object** and successive
prefixes grow outward as concentric shells. Streaming a `radial` ladder makes a
scene appear to grow from its middle. That is purely an authoring choice — the
viewer needs no changes and does nothing special with it.

Three consequences, each load-bearing:

- **It is the only ASCENDING sort.** Its score is a distance, not a contribution
  to maximise. Every neighbouring branch in `compute_additive_order` sorts
  `-score`.
- **The centre is the bounding-box centre, not the scene origin**, so a dataset
  sitting far from the origin still reveals from its own middle rather than from
  one corner. Override with `reveal_center`.
- **A `radial` ladder carries NO energy stamps** (`energy_fraction_cum` per
  sub-LOD, `reference_energy` on the leaf), and this is enforced at authoring
  time. The viewer multiplies brightness by `1/e(k)` while a ladder is
  incomplete — correct for an approximation, backwards for a reveal, where an
  inner shell is blown out and then *dims* as the object completes. The boost is
  capped at 10× by `ENERGY_FLOOR`. The compensation is gated on the blending mode
  and never on geometry type, so omitting the stamps is the only place to stop it.
  **Where it actually bites:** `applyLodFade` is the sole consumer and the
  `kind=lod` group registry is its sole caller, so the stamps matter for a ladder
  *inside* a lod group (`levels`/`adaptive`/`overview`, which carry stream ladders
  by default) and are inert on a bare `stream`/`flat` leaf. Measured on a
  throttled server: a bare leaf renders byte-identically stamped or not, while
  inside a `levels` group the stamped arm is 1.87× brighter in mean luma until
  the ladder completes. The rule is unconditional regardless, because the method
  already distinguishes both cases: `--recipe levels|adaptive|overview -m radial`
  writes reveal ladders *inside* a lod group (stamps bite) and `--recipe stream -m
  radial` writes a bare one (stamps inert). Nothing can smuggle stamps back in
  either — `gsplat additive` and `lod --recipe levels` both DISCARD an input
  ladder and re-derive from the method they are given (verified).
  Consequence to know about: cross-fade and the `e >= 0.6` early-upgrade release
  are therefore also inactive for a reveal, so shells hard-switch.

Distance is measured only over the axes with non-zero **covariance** extent
(`GSplatData._nondegenerate_axes`), so a stacked time/channel axis — built with
`sigma=0` — cannot become a shell dimension (the splats furthest in time would
otherwise land at the end of the ladder). Override with `spatial_dims`.

#### On a PARTITIONED recipe, pass `--reveal-center` explicitly

With `tiles` / `overview` / `adaptive`, the ladder is built **per part**, and the
default centre is each part's *own* bounding box. So `-m radial` without a centre
gives **N independent local reveals** — the object appears to grow from every tile's
middle at once — not one reveal growing from the object's middle. Measured on a
4-part BSP of a ball (mean distance of the first shell, per part):

| | → own part centre | → global centre |
|---|---|---|
| no `--reveal-center` | **16.3–17.6** (part avg 27–28) | 35–36 (part avg 37) |
| `--reveal-center 0,0,0` | 24–25 (part avg 27–28) | **24–26** (part avg 37) |

Both behaviours are useful and this is a deliberate default, not an oversight:
per-part centres make each *visible* tile paint its own middle first, which is the
right thing when tiles are frustum-culled and streamed independently. But if you
want the whole object to grow from one point, **pass the centre**:

```bash
luxar gsplat lod in.gsplats.zarr out.gsplats.zarr \
    --recipe tiles -m radial --reveal-center 0,0,0
```

The `batch-fit` path (`--merge-add-method radial`) accepts the method but has
**no** centre override — its manifest is string-keyed and the two knobs are not
plumbed — so a batch merge always produces per-part reveals. Use `gsplat lod` if you
need a global one.

The same `radial` method, with the same two knobs and the same no-stamps rule, is
available on Points and Lines — see `core/group/lod/`. On Lines it orders whole
polylines by their own centre, so every prefix keeps valid segment topology.
Those two have no covariance to read, so their shell axes come from the scene's
displayed dims (`resolve_reveal_spatial_dims`), falling back to non-zero
positional extent when the positions are not scene-aligned.

`auto` is the default. It resolves (via `resolve_additive_method`) to `greedy`
for `N ≤ _AUTO_ADDITIVE_MAX_N` (= 5000) and to `self_energy` above it. Rationale:
`greedy`/`spectral` need a sparse Gram, whose construction (`_build_sparse_gram`,
a pure-Python per-pair loop) is `O(nnz)` and scales with *overlap density* (avg
neighbours per splat), not `N` alone — so it becomes the bottleneck on large or
dense fits (the lazy-greedy heap pass itself is cheap). Empirically (`additive_lod`
Experiment C) `self_energy` trails greedy by only 2–10% AUC on real Luxar datasets
and is the right `O(N log N)` choice above the threshold. Pass an explicit method
to override `auto`; `greedy` remains the quality reference at small `N`.

### `slice_dims` — the slice-even interleave

A **modifier**, not a method: it composes with every row of the table above, and
is deliberately absent from `luxar.utils.lod_methods.GSPLAT_ADDITIVE_METHODS` for
that reason. `interleave_order_across_slices(data, order, slice_dims)` groups the
elements of an already-computed `order` by their distinct combination of the named
centre columns and re-emits them round-robin, one per group per pass, ties inside
a pass broken by position in `order`.

Why it exists: on a node the viewer **slices** (any hidden time/channel
dimension) a rung is sized against the whole node, but only one coordinate is on
screen — so a global contribution-ordered prefix piles onto the busy coordinates
and starves the sparse ones. The NEXRAD supercell (82 scans, 817,989 splats;
per-scan min 562, p05 774, median 10,499, max 19,237) shipped an absolute
`breakpoints="stream:20000"` first rung whose 5th-percentile scan held **4
splats**, failing both arms of `scripts/check_demo_ladders.py` (#2485).

What it guarantees: after `R` completed passes a prefix holds `min(n_i, R)`
elements of every slice `i` — an equal **absolute** budget per slice, with any
slice smaller than the budget carried WHOLE. That is exactly the shape
`check_demo_ladders.py`'s absolute first-paint arm asks for, and it is
**deterministic** — no seed. Sizing alone cannot get there: the gate's
250-element floor needs ~32% of that stack's sparsest scan, and a uniform
`method="random"` permutation is only proportional *in expectation* (measured p05
over seeds 0–7, `n_lods=3` cleared the floor 5 times in 8 and `n_lods=4` never
did). It is also **idempotent**, so a caller need not track whether it has
already been applied.

`slice_dims` indexes **raw, pre-`dim_order` centre columns** — the columns of the
array you handed in, not the scene's post-`dim_order` dimension positions. The
ladder is built in `core/group/gsplats_pipeline/from_data.py` above the
`apply_dim_order_*` pass `lod_dispatch` runs, so the two frames of reference
differ whenever `dim_order` permutes; they coincide for NEXRAD only because its
`dim_order` leaves time last in both. Reading the index off the scene's
`Dimensions` list is the likeliest way to get this wrong, and per the next
paragraph it fails *quietly*.

Two preconditions on the guarantee. A rung must be **at least as large as the
slice count**, or it cannot reach every slice at all — measured, 500 slices of 40
splats with `breakpoints=[200]` leaves 300 coordinates on zero, and since
within-pass ties break by position in `order` the ones left out are the faintest.
And the columns must be **genuinely discrete**: pointed at a continuous one,
nearly every key is distinct, nearly every rank is 0, and the `lexsort` reproduces
`order` — functionally a no-op, but **not** a bit-identical one, so do not use it
as an equality assertion. Real centres collide, and each collision demotes one
element by a pass, shifting the whole tail behind it: on 8 cached NEXRAD frames
(5,937 splats, 5,907 distinct values in column 0) `slice_dims=[0]` left the first
20 positions untouched yet moved 5,901 of 5,937 overall, and one deliberate
collision among 2,000 float32 samples moved 48. That mis-aim is reachable by
composition too, not only by typo: `lod_group=dict(coarsen_dims=[0, 1, 2, 3])`
coarsens *over* the stacked axis and turned 3 exact time coordinates into 35
fractional ones on the coarse level, while `resolve_additive_axis_gsplats` applies
one `slice_dims` to every level — a slice-even finest level and silently uneven
coarse ones. The default `Auto` coarsening (hidden axis as a hard barrier) is safe.

Two things it does *not* change. The within-slice order stays whatever the base
method produced, so with `method="auto"` each coordinate still paints
bright-core-first rather than evenly thin. And there is **no default column set**
— not because the columns are undiscoverable (the writer auto-detects them and
`save_gsplats.py` stamps the result as each rung's `slice_dims` attr, which reads
`[3]` on the built NEXRAD store) but because that detection runs at *save* time,
after the ladder has been ordered and cut. So the caller names them, exactly as
the CLI does for `--coarsen-dims`. Authoring spelling:

```python
scene.add_gsplats_from_data(
    ..., additive_lod=dict(n_lods=4, slice_dims=[3], recompute=True)
)
```

`recompute=True` is not optional boilerplate on a **stacked** dataset:
`combine_as_new_dimension` merges its sources' ladders instead of dropping them,
so a stack of already-laddered per-timepoint fits arrives with rungs to spare and
`resolve_additive_axis_gsplats` passes the whole spec through untouched. See the
trap note in `core/group/lod/README.md`.

Not exposed on the CLI: `gsplat lod` has no scene to tell it which columns the
viewer hides, and the demo authoring path is what #2485 is about.

## Breakpoints

The `breakpoints` parameter selects how the ordered splats are sliced
into LOD levels:

- `"equal-count"` (default) — `n_lods` levels of (nearly-)equal size.
- `"stream:<c>"` — bandwidth-derived **streaming ladder**: geometric
  cumulative cuts `[c, 2c, 4c, …, N]` (first chunk `c` splats, then
  doubling), resolved against each call's own `N` — so the same spec
  adapts per part / per substitutive level, silently clamping for small
  `N` (capped at `DEFAULT_STREAM_MAX_LEVELS` = 16 levels; a sliver tail
  `< c/2` folds into the previous cut). Size `c` from a download budget
  with `streaming_chunk_splats(target_ms, bandwidth_mbps,
  bytes_per_splat)` — e.g. 200 ms @ 25 Mbps @ 45 B/splat → ~14 k. The
  CLI's `--target-ms`/`--bandwidth-mbps` do this for you.
- `"equi-energy:<n>"` — `n` rungs at **equal shares of cumulative
  self-energy** along the ordering, then any increment above
  `DEFAULT_MAX_ADDITIVE_COMMIT` split into capped steps. Under a
  contribution-first ordering (`self_energy`) the first rung is the few
  heaviest splats and each later rung is fatter in count for the same light:
  fast first paint, and the slow rungs are the ones whose absence shows
  least. The `e(k)` stamps read ≈ `k/n` at the requested cuts by
  construction. Shared with Points/Lines (`equi_energy_cuts`).
- `list[int]` — explicit cumulative splat counts. The list length sets
  the number of levels. Example: `[1000, 5000, 25000]`. In per-part /
  per-level contexts (partitioned parts, pyramid levels) the counts are
  clamped to each part's own `N` via `clamp_counts_breakpoints` (a fixed
  list would otherwise abort on small parts).
- `list[float]` in $(0, 1]$ — cumulative *energy fractions*. Cutpoints
  are placed at the smallest `k` whose cumulative-utility curve crosses
  each target. Example: `[0.5, 0.9, 0.99, 1.0]`.

The result of `make_additive_lod` is a multi-LOD `GSplatData` whose
`additive_prefix(k)` returns a valid additive prefix.

`additive_rung_count(n, n_lods, breakpoints)` answers "how many rungs would this
spec leave on a leaf of `n` splats?" without ordering anything. The exactness
comes from SHARING the builder's own `_resolve_breakpoints`: the three
count-based kinds above are counted off the very cuts the build would consume,
never re-derived. The loop also mirrors the builder's `if end <= prev: continue`
de-duplication, but as a mirror only — against a future cut resolver that emits a
duplicate — since `_resolve_breakpoints` returns strictly-increasing positive
cuts on every non-energy path, making that branch unreachable today. It
returns `None` (UNKNOWN, never a raise) for energy fractions — those need the
ordering and the energy curve, the expensive half this query exists to avoid —
and for any spec `_resolve_breakpoints` would reject, leaving the real build to
report the fault at its own site. `None` is a statement about the SPEC, not about
the leaf, so a caller that cannot read it should fall back to what it already
knows rather than assume "no ladder". Callers are gates that must know whether a
ladder will EXIST before paying to build one: `partition=` and a multi-rung
ladder are mutually exclusive, and the scene's file/graft door has to settle that
before it writes a `kind=partition` wrapper it would otherwise strand (#1632).
Note the converse too — a fault past the cut resolver (a bad `method`, a stray
`substitutive_level` key) does not move the count, so a gate refusing on it masks
that fault instead of letting the builder report it.

## Complexity

| Step                      | Cost                                          |
|---------------------------|-----------------------------------------------|
| Sparse Gram (σ-trunc. pruning) | $O(N \log N + N \cdot \bar{c})$ where $\bar{c}$ = avg neighbourhood size |
| Dense Gram                | $O(N^2 D^3)$ — used at $N \le 2000$            |
| Lazy greedy (sparse)      | $O(N \cdot \mathrm{nnz}(G))$ amortised        |
| Scan greedy (dense)       | $O(N^2)$                                      |
| Self-energy / mass / amp  | $O(N \log N)$                                 |

The reference Luxar dataset benchmarks from `additive_lod` Experiment C:

- `dapi_nuclei` ($N = 1{,}274$): full sparse-Gram + greedy in well under a second.
- `acto3d_heart` ($N = 79{,}104$): 20 s sparse-Gram + 13 s greedy ≈ 33 s
  total, ~50 GB savings versus a dense Gram.

## Limits

- Pure NumPy / SciPy. No GPU acceleration. Greedy on large/dense fits is
  expensive (the `O(nnz)` sparse-Gram build) — the default `method="auto"`
  switches to `self_energy` above `N = 5000` to avoid it.
- `method="auto"` (the default) IS a size-adaptive fallback: `greedy` at small
  `N`, `self_energy` above the threshold (see "Methods"). Pass an explicit
  method to pin one and keep the choice loud.
- Output is written as a v3.4 `.gsplats.zarr` node tree (v3.0 still readable): a leaf with
  `additive_<i>/` subgroups for an additive ladder, or a `kind=lod` group
  of children for a substitutive hierarchy. See
  `docs/specs/GSPLATS_ZARR_FORMAT.md` for the full on-disk grammar.

## Substitutive LOD (`substitutive.py`)

Each coarser level synthesises $M = \lceil N/K \rceil$ representative
splats per the supp doc `substitutive_lod.tex`:

```python
make_substitutive_lod(
    data: GSplatData,
    *,
    compression_factor: int = 4,        # K
    levels: int = 3,                    # L (coarser levels to produce)
    method: str = "auto",               # see "Substitutive methods" below
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    coverage_inflation: float = 3.0,    # anti-grid inter-spread widening (1.0 = off)
    conserve_mass: bool = True,         # per-level (per-barrier-group) DC conservation
    refine: str = "none",               # "l2" = post-merge L2 refit per level
    refine_iters: Optional[int] = None, # Adam steps per refined level (None → 120 for l2 / 300 for volume)
    device: str = "auto",               # auto | cpu | cuda | mps
    seed: int | None = None,
    verbose: bool = False,              # per-level Arbol logging
) -> GSplatData
```

Returns a single `GSplatData` with `n_substitutive = levels + 1`, one
additive sub-LOD per substitutive level, and splat counts
`[N, ⌈N/K⌉, ⌈N/K²⌉, …, ⌈N/K^L⌉]`. The on-disk container is a v3.4
`kind=lod` group (`child_<i>/` per level, coarsest→finest on disk)
— no `splats/substitutive_<s>/` wrapper.

### Coverage inflation (anti-grid widening)

Every method finishes with a moment-matched merge, which gives balanced
spatial bins of pitch $d$ a representative with $\sigma \approx d/\sqrt{12}
\approx 0.29\,d$ — well below the $\sigma \gtrsim d/2$ a lattice of
Gaussians needs to sum flat. Because the Morton warm start quantises bin
boundaries onto a *global dyadic grid*, the resulting coverage dips align
into coherent axis-aligned planes: a very visible grid pattern at every
coarse level. `coverage_inflation` (default **3.0**) widens only the
*inter-center* spread term of each merge ($\Sigma_\text{out} =
\text{intra} + \beta\,\text{inter}$; $\beta = 3$ turns $d^2/12$ into
$(d/2)^2$) with a mass-preserving amplitude rescale, so each splat's
integral — hence the additive-blend X-ray projection — is unchanged.
$\beta = 3$ is the exact fixed point of the level recurrence, so the
calibration holds at every depth. Set `coverage_inflation=1.0` (or CLI
`--coverage-inflation 1.0`) for the historical pure moment match.

### L2 refinement (`refine="l2"`)

Opt-in post-merge refinement (`_substitutive/refine.py`): each merged level is
Adam-optimized against its fine input under the **closed-form mixture L²**
`‖f−g‖² = ‖f‖² − 2⟨f,g⟩ + ‖g‖²` (pairwise Gaussian inner products over sparse
neighbour pair lists). The per-bin merge objective is structurally blind to
cross-bin overlap; the global L² objective *contains* the coverage gaps and the
over-blur, so the refit widens splats exactly where the field is flat and keeps
isolated structure tight. Measured: rel-L² 0.089 vs 0.151 for the β=3 merge on
flat fields (equal flatness), 0.141 vs 0.233 on isolated blobs with peak
preservation 0.99 vs 0.91.

Engineering guarantees:

- **Trusted checkpoints** — pair lists are rebuilt from the current geometry
  every `rebuild_every` steps and the objective is only compared/snapshotted
  right after a rebuild (a stale/truncated pair list is exploitable: the
  optimizer inflates mass to harvest cross terms a capped `‖g‖²` cannot see).
  The raw merge seed is the first trusted candidate, so the refit is **never
  worse than the merge** in the trusted metric.
- **Mass manifold** — amplitudes are renormalized so the coarse mixture's total
  mass equals the fine mixture's at every iterate: the additive-render DC is
  pinned (no brightness pop across levels) and the mass-inflation exploit
  direction is closed outright.
- **Chain semantics** — the refined level feeds the next reduction, so each
  level fits its immediate predecessor; with `refine="l2"` the β=3 coverage
  inflation is demoted from final answer to *optimizer seed*.
- **Barrier freezing** — under `coarsen_dims` grouping, barrier center
  coordinates and every Σ row/column touching a barrier dim stay at the seed
  values (no sliced-dim bleed).
- Minibatched per-step pair sampling above `step_pair_budget` (unbiased);
  PD-by-construction Cholesky parameterization plus a defensive NaN-grad step
  skip; deterministic given `seed` (a local `torch.Generator`).

Only `refine_iters` is exposed on the public builders; the remaining constants
live in `L2RefineConfig` (stability-critical, not a tuning surface).

### Volume re-fit (`refine="volume"`)

The third quality rung, above the merge and the L2 mixture refit
(`volume_refit.py`): each merged level is **warm-start re-fitted against the
source volume itself** via `fit_gaussian_splats(volume, seeds=<merge level>)`
— the full rasterizer + Adam stack, identity-preserving (no cull, no dynamic
ops, colors carried over). Unlike `l2`, whose target is the fine *mixture*
(which already carries the fine fit's own error), this optimizes the true
render-fidelity objective at the coarse budget. Benchmarked on real microscopy
(skimage cells3d nuclei): **+5–6 dB** full-res and
**+10–12 dB** at viewing scale over the merge, with unchanged splat count, and
the warm start beats a cold fit while drifting ~2× less across levels (less
LOD popping). (Those figures were measured before the #1172 amplitude-convention
fix. The never-worse guard bounds the outcome at the merge, so the *sign* of the
reported gain is safe, but the magnitudes have not been re-measured since.)
Fitting a blurred/downscaled volume proxy was benchmarked and
rejected — it discards positional detail the merge inherits from the sharp
fine fit.

Engineering guarantees and scope:

- **Never worse than the merge** — the seed and the re-fit candidate are both
  rendered to the volume grid and the lower-MSE one is kept.
- **Mass pinning** (follows the ladder-wide `conserve_mass`, default on) — the
  re-fit's amplitudes are rescaled so its rendered DC equals the seed's (already
  pinned to the fine chain). The free fit otherwise tracks the volume's true DC,
  which the finest level may under-explain — stored unpinned that is a visible
  cross-level brightness pop (measured ~14 %). `--no-conserve-mass` opts into the
  raw volume-accurate DC.
- **Frame checks (both directions)** — an ENLARGED physical frame (center bbox
  outside the volume's padded voxel index range) skips the re-fit up front; a
  SHRUNK, ROTATED, or AXIS-SWAPPED frame fits inside that box, so it is
  caught after the fit by the per-splat relocation check (rows are 1:1 in the
  identity-preserving fit; a fit whose median per-splat displacement exceeds
  the seed's own spread/footprint slack wholesale relocated the splats — a
  frame mismatch, whatever its shape). Either way the seed is kept with a warning; a voxel-frame re-fit
  would *win* the MSE guard while being misplaced relative to the ladder.
- **Chain semantics** — unlike `l2`, the merge chain continues from the
  *unrefined* merge output; only the stored level is replaced by the re-fit
  (each level's re-fit is independently seeded from its own merge).
- **Requires the volume in hand** — exposed at all three entry points:
  - `gsplat lod --target <volume> --refine volume` (+ `--target-axes` for a
    stacked target);
  - `gsplat fit --recipe levels --refine volume` — no `--target` needed, since
    the volume being fitted is already in hand and the fit emits splats in its
    voxel frame, so the identity axis map is correct by construction;
  - `batch-fit merge --recipe levels --refine volume` (and the same thing spelled
    `--merge-refine volume` on `batch-fit run` / `batch-fit submit`, which record
    it into the manifest so the local auto-merge and the Slurm merge job both
    pick it up; validated at PLAN time so a typo does not surface after every
    tile has been fitted) — the streaming merge
    re-opens the source the manifest recorded and crops it to each tile as that
    tile streams. It validates up front (source readable, `--axes` recorded, no
    folded channel axis) because a per-part failure mid-stream would leave a
    half-written store. This path composes BOTH mechanisms: each tile-part is
    cropped spatially AND split per stacked timepoint, which is what a tiled
    timelapse needs.
- **Barrier dims and per-tile crops are supported** — both once meant "one
  volume cannot serve every seed", and both are answered the same way: each
  re-fit is handed the sub-volume it is actually responsible for
  (`volume_regions.py`). A barrier group owns one index along the barrier axes
  and re-fits against that slice; a partition part owns a box of the spatial
  axes and re-fits against that crop. They compose, which is what a tiled
  timelapse needs. So `levels`, the `overview` cap, and per-part `adaptive` are
  all supported.
  - The barrier axis is **sliced out of the fit**, not held still, because it
    cannot be held still: the fitting stack has no freeze mechanism, it
    re-parameterises centers as `sigmoid(raw)·(shape−1)` over every axis and
    floors every per-axis sigma, so a 4D re-fit would drag splats off their
    timepoint and widen them along time whatever penalty it was given. The
    barrier coordinate and covariance rows come back verbatim from the seed,
    recombined in factor space (`L = [[A,0],[B,C]]`, substitute only `A`) so
    positive-definiteness is structural.
  - A stacked target needs `--target-axes` (e.g. `time,z,y,x`): fitted splats
    order their centers spatial-first with the stacked axis LAST, while a source
    array is usually time-FIRST, so the identity map would target the wrong
    axis. Distinct from `--timepoint`, which slices one timepoint out instead.
  - **Tile containment** — a per-part re-fit that moves a centre out of its own
    cell is discarded in favour of the merge (stat: `tile_escape`). The
    never-worse MSE guard is structurally blind to this, since an escapee can
    still lower the crop's MSE, while the viewer frustum-culls by part bounds,
    so an escapee would simply stop being drawn.
  - **The volume is only ever sliced**, never coerced whole, so a lazy zarr
    store stays lazy: a 253-timepoint 407×2048×2048 uint16 timelapse is 431 GB
    while one timepoint is 3.4 GB.
  - Level stats are **aggregated** once a level is refined in pieces
    (`n_pieces`, `*_frac` fractions instead of one verdict that would hide 252
    of 253 outcomes) plus `mse_stored`, which reads the kept verdict rather than
    `min(seed, refit)` — a re-fit rejected for leaving its tile can hold the
    lower MSE without being what was stored.

Only `refine_iters` (default 300 — omitted resolves to `VolumeRefitConfig`'s
value in both the API and CLI) and the ladder-wide `conserve_mass` are exposed;
the remaining constants live in `VolumeRefitConfig`.

### Mass conservation (`conserve_mass=True`)

The per-bin L²-optimal amplitude is **not** mass-preserving (3–17 % total-mass
drift per level, content-dependent), and total mass over the displayed dims is
exactly the DC an additive render integrates — uncorrected it shows as a
visible **brightness pop at every LOD switch** (measured up to −11.5 % per time
slice at the coarsest level on real 4D data). Each reduced level's amplitudes
are therefore rescaled by one global factor so its mass over the *coarsened*
dims equals its fine input's; under `coarsen_dims` grouping this runs per
barrier group, so every time/channel slice keeps its exact brightness at every
level. Related numerics fix: the merge's Cholesky ridge is proportional per
dim (an absolute `1e-6·I` ridge inflated a near-delta barrier width, e.g. a
lifted time σ of ~1e-9, by ×300,000). `--no-conserve-mass` restores the raw
per-bin amplitudes.

For the **lifted points/lines LOD path** (and any caller passing
`amplitude="mass"` to `make_substitutive_lod`), the merge goes one step
further: each bin's amplitude is set to exactly its members' summed
`a·|det L|` mass on the final (inflated, ridged) covariance instead of the
L²-optimal projection. Per-bin per-channel colored light is then conserved
together with the mass-weighted mean colors (hue coherence across levels),
and the global `conserve_mass` rescale becomes a near-no-op safety net.
The default stays `amplitude="l2"` for fitted volumetric gsplats.

### Substitutive methods

| Method          | Warm start        | Refinement        | Recommended for                                |
|-----------------|-------------------|-------------------|------------------------------------------------|
| `auto`          | per level         | per level         | **default**. Picks `greedy` for levels ≤ 5000 splats (best quality, fast there) and `kmeans_lloyd` above. Large datasets get fast coarse-level reductions + greedy-quality fine levels |
| `kmeans`        | Morton chunk      | none              | fast and already high quality — the raw spatially-coherent partition |
| `kmeans_lloyd`  | Morton chunk      | vectorised cost-increment Lloyd | large-N workhorse. Adds a few dB of PSNR over `kmeans`; beats amplitude culling at every $K$ on supp-doc Experiment C |
| `greedy`        | Runnalls hierarchical (lazy-heap, incremental) | none           | **quality-leading** at small $N$/$K$ (lowest rel-L² in practice). Now ~$O(Nk\log Nk)$, usable into the tens of thousands |
| `greedy_lloyd`  | Runnalls hierarchical (lazy-heap, incremental) | vectorised cost-increment Lloyd | quality-leaning option; greedy is often already near-optimal so Lloyd adds little  |

The method names keep their `kmeans` prefix for API stability; the warm
start is the `O(N log N)` Morton (Z-order) space-filling-curve partition,
**not** a global k-means++. In the substitutive regime there are
`M = N/K` bins, so a global k-means++ init is `O(M·N) = O(N²/K)` — it took
**minutes-to-hours at N ≈ 256K** and was the dominant cost. Morton-sort +
chunk gives the same `M` spatially-coherent, balanced, empty-bin-free bins
in `O(N log N)` (sub-second at 256K).

### Performance

- The warm-start partition and the per-bin merge are **fully vectorised
  segment reductions** (`torch.Tensor.index_add_` keyed by bin id) — no
  Python per-splat or per-bin loops. They run on PyTorch via
  `device='auto'` (CUDA / MPS / CPU).
- Lloyd refinement is **vectorised and monotone**: each pass reassigns all
  splats synchronously to the template they best project onto (candidates
  are the current bins of a splat's Morton-curve neighbours — an
  `O(N·k)` gather, not a spatial-hash kNN), rebuilds templates, and keeps
  the pass only if the global projection energy `P = Σ_b ⟨f,Ḡ⟩²/‖Ḡ‖²`
  does not decrease. So the result is never worse than the warm start.
- A reduction of **256K splats → 64K builds in ~3–4 s on CPU** (vs.
  ~1 hr before); the full 6-level ladder builds in ~5 s. Quality is high:
  a 4× reduction reconstructs at ~46 dB PSNR vs. the full set (Lloyd adds
  ~3 dB over the raw Morton partition).
- Greedy uses a **lazy-deletion priority queue** (Runnalls): each candidate
  pair cost is computed once (batched), stale heap entries are tagged by
  per-cluster version counters, and after each merge only the new cluster's
  ~`k` neighbour edges are recomputed (clusters live in fixed slots with an
  active free-list — no array splicing). Roughly `O(N·k·log(N·k))` vs. the
  former `O(N²·k)` full re-scan (which took ~57 s at N=200; now ~0.2 s, and
  ~9 s at N=10K). It is heavier per-merge than the Morton warm start, so
  `kmeans_lloyd` is still the default workhorse for very large `N`.

### Out of scope (this round)

- Joint relaxation polish (gradient descent over all $M$ Gaussians).
- Cauchy–Schwarz divergence and $W_2$ alternative cost metrics
  (the supp doc settles on $L^2$ as the primary objective).
- Cross-cell parameter deduplication via `splats/shared/` — deferred to a
  future v2.1; revisit after empirical disk-footprint evidence on real
  combined pyramids.

## References

- Supp. Doc. `additive_lod` (`luxar-paper/supp_doc/additive_lod/`):
  formal derivation, $(1-1/e)$ bound, complexity, real-data experiments.
- Supp. Doc. `substitutive_lod` (`luxar-paper/supp_doc/substitutive_lod/`):
  $K$-wise moment match, $L^2$-optimal amplitude, cost-increment Lloyd,
  spectral lower bound; Experiment C settles the algorithm choice.
- Nemhauser, Wolsey & Fisher (1978) — submodular greedy approximation.
- Mallat & Zhang (1993) — matching pursuit.
- Minoux (1978) — accelerated greedy via marginal-gain laziness.
- Lloyd (1982) — k-means clustering iteration.
- Runnalls (2007) — Gaussian mixture reduction via greedy hierarchical merging.
