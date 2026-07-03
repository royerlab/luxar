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

### Module layout

| File | Role |
|------|------|
| `additive.py` | additive axis: ordering + ladder (`make_additive_lod`, `compute_additive_order`) |
| `substitutive.py` | substitutive axis orchestrator (`make_substitutive_lod`, `_reduce_one_level`, `_pack_level`) |
| `pyramid.py` | `make_lod_pyramid` — chains substitutive (outer) × additive (inner) |
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
> See `gsplats/calibration.py` and the "Calibration (Blind-Spot CV)"
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
ladder.save("additive_lod.gsplats.zarr")            # v3.1 leaf with additive_<i>/ subgroups

# ── Substitutive: ceil(N/K^L) splats per level, replacement hierarchy
pyramid = make_substitutive_lod(data, compression_factor=4, levels=3)
# pyramid is a GSplatData with 4 substitutive levels (index 0 = finest)
pyramid.save("substitutive_pyramid.gsplats.zarr")   # v3.1 kind=lod group
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
full.save("pyramid.gsplats.zarr")       # v3.1 kind=lod group of additive-ladder leaves
```

## API

```python
make_additive_lod(
    data: GSplatData,
    n_lods: int = 4,
    *,
    method: str = "auto",         # see "Methods" below
    breakpoints = "equal-count",   # or list[int] | list[float]
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: int | None = None,
) -> GSplatData
```

```python
compute_additive_order(
    data: GSplatData,
    method: str = "auto",
    *,
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: int | None = None,
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
| `auto`        | size-adaptive: `greedy` if `N ≤ 5000` else `self_energy`  | (see chosen)  | (see chosen)      | (see chosen)        |

`auto` is the default. It resolves (via `resolve_additive_method`) to `greedy`
for `N ≤ _AUTO_ADDITIVE_MAX_N` (= 5000) and to `self_energy` above it. Rationale:
`greedy`/`spectral` need a sparse Gram, whose construction (`_build_sparse_gram`,
a pure-Python per-pair loop) is `O(nnz)` and scales with *overlap density* (avg
neighbours per splat), not `N` alone — so it becomes the bottleneck on large or
dense fits (the lazy-greedy heap pass itself is cheap). Empirically (`additive_lod`
Experiment C) `self_energy` trails greedy by only 2–10% AUC on real Luxar datasets
and is the right `O(N log N)` choice above the threshold. Pass an explicit method
to override `auto`; `greedy` remains the quality reference at small `N`.

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

## Complexity

| Step                      | Cost                                          |
|---------------------------|-----------------------------------------------|
| Sparse Gram (3σ pruning)  | $O(N \log N + N \cdot \bar{c})$ where $\bar{c}$ = avg neighbourhood size |
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
- Output is written as a v3.1 `.gsplats.zarr` node tree (v3.0 still readable): a leaf with
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
    refine_iters: int = 120,            # Adam steps per refined level
    device: str = "auto",               # auto | cpu | cuda | mps
    seed: int | None = None,
    verbose: bool = False,              # per-level Arbol logging
) -> GSplatData
```

Returns a single `GSplatData` with `n_substitutive = levels + 1`, one
additive sub-LOD per substitutive level, and splat counts
`[N, ⌈N/K⌉, ⌈N/K²⌉, …, ⌈N/K^L⌉]`. The on-disk container is a v3.1
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
