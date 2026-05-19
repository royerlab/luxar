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

> **Upstream step**: use `luxar gsplat cal` to pick a principled splat
> budget K\* before fitting. The canonical end-to-end pipeline is
> **`cal` → `fit --seeds K*` → `lod additive` (or `lod substitutive`)**.
> See `gsplats/calibration.py` and the "Calibration (Blind-Spot CV)"
> section in the parent `gsplats/README.md`.

## Quick start

```python
from luxar.gsplats.lod import make_additive_lod, make_substitutive_lod
from luxar.gsplats.gsplat_data import GSplatData

data = GSplatData.load("fit.gsplats.zarr")          # single-LOD

# ── Additive: same N splats, prefix-monotone
ladder = make_additive_lod(data, n_lods=4)          # multi-LOD GSplatData
ladder.save("additive_lod.gsplats.zarr")

# ── Substitutive: ceil(N/K^L) splats per level, replacement hierarchy
levels = make_substitutive_lod(data, compression_factor=4, levels=3)
# levels is a list[GSplatData]; each is flat (n_lods=1)
for L, lev in enumerate(levels):
    lev.save(f"sub_lod_level_{L}.gsplats.zarr")
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
`up_to_lod(k)` returns a valid additive prefix.

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
- The on-disk format is unchanged — the multi-LOD output is consumed by
  the existing viewer-side progressive loader without modification.

## Substitutive LOD (`substitutive.py`)

Each coarser level synthesises $M = \lceil N/K \rceil$ representative
splats per the supp doc `substitutive_lod.tex`:

```python
make_substitutive_lod(
    data: GSplatData,
    *,
    compression_factor: int = 4,        # K
    levels: int = 3,                    # L (coarser levels to produce)
    method: str = "kmeans_lloyd",       # see "Substitutive methods" below
    lloyd_iterations: int = 5,
    candidate_bins_k: int = 12,
    device: str = "auto",               # auto | cpu | cuda | mps
    seed: int | None = None,
) -> list[GSplatData]
```

Returns `levels + 1` flat `GSplatData` objects:
`[level_0=data, level_1, ..., level_L]` with splat counts
`[N, ⌈N/K⌉, ⌈N/K²⌉, …, ⌈N/K^L⌉]`. Each level is a standalone single-LOD
zarr (the on-disk multi-level container is a future scene-graph node).

### Substitutive methods

| Method          | Warm start        | Refinement        | Recommended for                                |
|-----------------|-------------------|-------------------|------------------------------------------------|
| `kmeans`        | k-means++ on means | none              | ablation only — *worse than amplitude culling on real anisotropic data*      |
| `kmeans_lloyd`  | k-means++ on means | cost-increment Lloyd | **recommended workhorse**. Beats amplitude culling at every $K$ on supp-doc Experiment C |
| `greedy`        | Runnalls hierarchical | none           | small/medium $N$ where $\mathcal{O}(N^2)$ is acceptable. Quality-leading at small $K$ |
| `greedy_lloyd`  | Runnalls hierarchical | cost-increment Lloyd | quality-leaning option for small/medium $N$  |

### Performance

- All warm-start k-means and per-iteration top-k bin-candidate lookups
  run on PyTorch via `device='auto'` and the
  `BatchedSpatialHashGrid` from `luxar.utils.spatial_hash` (CUDA / MPS /
  CPU; conservative fallback on OOM).
- Lloyd's per-splat sequential update is honest about its cost: the
  per-iteration work is $\mathcal{O}(N \cdot k \cdot \bar{K}^2)$ with
  $k$ = candidate bins and $\bar{K} = N/M$ the average bin size. For
  typical Luxar workloads ($N \approx 10^5{-}10^6$, $K = 4$,
  $\bar K = 4$), this is fast.
- Greedy with spatial-hash candidate pruning is still $\mathcal{O}(N^2)$
  worst case; use only at small/medium $N$ (≤ a few thousand).

### Out of scope (this round)

- Joint relaxation polish (gradient descent over all $M$ Gaussians).
- Cauchy–Schwarz divergence and $W_2$ alternative cost metrics
  (the supp doc settles on $L^2$ as the primary objective).
- Scene-graph node integration of the multi-level container.

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
