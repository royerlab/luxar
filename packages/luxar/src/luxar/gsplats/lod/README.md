# Levels-of-Detail for Gaussian Splats

Post-processing operators that turn a fitted `GSplatData` into a streamable
LOD ladder. Currently provides **additive** LOD; **substitutive** LOD is
planned (see `additive_lod.tex` and `substitutive_lod.tex` supplementary
documents).

## Overview

Two complementary LOD axes:

- **Additive** — same `N` splats, ordered so that the prefix sum at any
  `k` splats is the best $L^2$ approximation of the full scene at that
  budget. Implemented in `additive.py`.
- **Substitutive** — synthesised representative splats per coarser level
  via mixture reduction. Not yet implemented.

This module is a **pure post-process**. Fitting (single-pass or
progressive) returns a single flattened `GSplatData`; an LOD ladder is
built only on demand.

## Quick start

```python
from luxar.gsplats.lod import make_additive_lod
from luxar.gsplats.gsplat_data import GSplatData

data = GSplatData.load("fit.gsplats.zarr")          # single-LOD
lod  = make_additive_lod(data, n_lods=4)            # 4-level ladder
lod.save("lod.gsplats.zarr")
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

## References

- Supp. Doc. `additive_lod` (`luxar-paper/supp_doc/additive_lod/`):
  formal derivation, $(1-1/e)$ bound, complexity, real-data experiments.
- Nemhauser, Wolsey & Fisher (1978) — submodular greedy approximation.
- Mallat & Zhang (1993) — matching pursuit.
- Minoux (1978) — accelerated greedy via marginal-gain laziness.
