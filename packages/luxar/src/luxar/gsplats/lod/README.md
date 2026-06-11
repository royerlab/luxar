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
| `_substitutive/` | private support subpackage for `substitutive.py`: `warm_start.py` (Morton partition), `kmeans_lloyd.py` (cost-increment Lloyd), `greedy.py` (Runnalls lazy-heap merge) |

The public import paths (`luxar.gsplats.lod`, `lod.additive`,
`lod.substitutive`, `lod._kernels`) are unchanged; the three substitutive
algorithms are grouped under `_substitutive/` (leading underscore because
a module `substitutive.py` and a package `substitutive/` cannot coexist
in CPython).

> **Upstream step**: use `luxar gsplat cal` to pick a principled splat
> budget K\* before fitting. The canonical end-to-end pipeline is
> **`cal` → `fit --seeds K*` → `lod additive` (or `lod substitutive`)**.
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
ladder.save("additive_lod.gsplats.zarr")            # v3.0 leaf with additive_<i>/ subgroups

# ── Substitutive: ceil(N/K^L) splats per level, replacement hierarchy
pyramid = make_substitutive_lod(data, compression_factor=4, levels=3)
# pyramid is a GSplatData with 4 substitutive levels (index 0 = finest)
pyramid.save("substitutive_pyramid.gsplats.zarr")   # v3.0 kind=lod group
for s, lev in enumerate(pyramid.substitutive_levels):
    print(f"level {s}: {lev.n_splats_total} splats, K={lev.compression_factor}")

# ── Full pyramid: substitutive (outer) × additive (inner) in one call
full = make_lod_pyramid(
    data,
    compression_factor=4, levels=3,     # substitutive axis (4 levels)
    n_additive_lods=4,                  # additive axis (4 sublods per level)
)
full.save("pyramid.gsplats.zarr")       # v3.0 kind=lod group of additive-ladder leaves
```

## API

```python
make_additive_lod(
    data: GSplatData,
    n_lods: int = 4,
    *,
    method: str = "greedy",       # see "Methods" below
    breakpoints = "equal-count",   # or list[int] | list[float]
    truncation_sigmas: float = 3.0,
    max_n_dense: int = 2_000,
    seed: int | None = None,
) -> GSplatData
```

```python
compute_additive_order(
    data: GSplatData,
    method: str = "greedy",
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

`greedy` is the recommended default. Empirically (`additive_lod`
Experiment C) `self_energy` trails greedy by 2–10% AUC on real Luxar
datasets and is the best `O(N\log N)` fallback for very large `N` where
sparse-Gram construction becomes the bottleneck.

## Breakpoints

The `breakpoints` parameter selects how the ordered splats are sliced
into LOD levels:

- `"equal-count"` (default) — `n_lods` levels of (nearly-)equal size.
- `list[int]` — explicit cumulative splat counts. The list length sets
  the number of levels. Example: `[1000, 5000, 25000]`.
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

- Pure NumPy / SciPy. No GPU acceleration. Greedy on $N \gtrsim 10^6$ is
  not recommended; use `method="self_energy"` instead.
- Auto-fallback to `self_energy` is **not** enabled. Method choice is
  explicit so failure modes are loud.
- Output is written as a v3.0 `.gsplats.zarr` node tree: a leaf with
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
    device: str = "auto",               # auto | cpu | cuda | mps
    seed: int | None = None,
    verbose: bool = False,              # per-level Arbol logging
) -> GSplatData
```

Returns a single `GSplatData` with `n_substitutive = levels + 1`, one
additive sub-LOD per substitutive level, and splat counts
`[N, ⌈N/K⌉, ⌈N/K²⌉, …, ⌈N/K^L⌉]`. The on-disk container is a v3.0
`kind=lod` group (`child_<i>/` per level, coarsest→finest on disk)
— no `splats/substitutive_<s>/` wrapper.

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
