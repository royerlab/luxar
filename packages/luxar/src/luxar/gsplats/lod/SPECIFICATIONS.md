# LOD: Levels-of-Detail for Gaussian Splats

**Version**: 0.1.0
**Last Updated**: 2026-05-06
**Scope**: Additive LOD only. Substitutive LOD is planned and lives at the
scene-graph node level (out of scope for this submodule's first cut).

## Overview

Given a fitted `GSplatData` of `N` splats, this module produces a
multi-level `GSplatData` whose levels are valid prefixes of a principled
ordering of the original splats. The ladder is consumed by the viewer's
existing multi-LOD progressive loader without format changes; only the
*order* of splats inside the LOD groups is changed.

The implementation derives directly from the supplementary document
`additive_lod` (`luxar-paper/supp_doc/additive_lod/additive_lod.tex`),
which formalises the additive-LOD problem and proves a
$(1-1/e)$-approximation guarantee for the greedy / matching-pursuit
ordering.

**Related Specifications**:

- Splat container and serialization: `../gsplat_data.py` and
  `../io/SPECIFICATIONS.md`.
- Substitutive LOD (planned): `substitutive_lod.tex` supp doc; will live
  at the scene-graph node level, not in this module.
- Removal LOD: `../culling.py` (`cull_by_contribution`) — distinct
  operator, kept as-is.

## Dependencies

**Core**:

- NumPy
- SciPy (`scipy.spatial.cKDTree`, `scipy.sparse`,
  `scipy.sparse.linalg.eigsh`)

**Internal reuse**:

- `luxar.gsplats.gsplat_data.GSplatData` / `GSplatLOD`
- `luxar.gsplats.gsplat_data._SplatArrayMixin._cholesky_diag_elements`
- `luxar.gsplats.utils.trils.unpack_tril`

No PyTorch, no GPU. Pure CPU.

## Mathematical Formulation

A Gaussian splat dataset represents a non-negative scalar field

$$
f(x) = \sum_{i=1}^{N} \phi_i(x), \qquad
\phi_i(x) = a_i \exp\bigl(-\tfrac{1}{2}(x - \mu_i)^\top \Sigma_i^{-1} (x - \mu_i)\bigr)
$$

with peak amplitudes $a_i > 0$, centres $\mu_i \in \mathbb{R}^D$, and
positive-definite covariances $\Sigma_i = L_i L_i^\top$ stored as
packed lower-triangular Cholesky factors.

### The additive-LOD problem

Given a permutation $\pi: \{1,\dots,N\} \to \{1,\dots,N\}$, define the
prefix and residual at step $k$:

$$
g_k = \sum_{j=1}^{k} \phi_{\pi(j)}, \qquad
r_k = f - g_k = \sum_{j=k+1}^{N} \phi_{\pi(j)}.
$$

The squared $L^2$ residual energy is

$$
E_k(\pi) = \|r_k\|_{L^2}^2 = \mathbf{1}_{S_k^c}^\top \mathbf{G}\, \mathbf{1}_{S_k^c}
$$

where $\mathbf{G}_{ij} = \langle \phi_i, \phi_j\rangle_{L^2}$ has the
closed-form Gaussian inner product

$$
\mathbf{G}_{ij} = a_i a_j (2\pi)^{D/2}
  \sqrt{\frac{|\Sigma_i| |\Sigma_j|}{|\Sigma_i + \Sigma_j|}}\,
  \exp\!\left(-\tfrac{1}{2}(\mu_i - \mu_j)^\top (\Sigma_i + \Sigma_j)^{-1} (\mu_i - \mu_j)\right).
$$

Diagonal: $\mathbf{G}_{ii} = a_i^2 \pi^{D/2} |\Sigma_i|^{1/2}$.

### Submodularity and the greedy guarantee

For unsigned splats $\mathbf{G}_{ij} \geq 0$ for all $i, j$, so the
cumulative-energy utility $U(S) = \|f\|^2 - \|f - \sum_{i \in S} \phi_i\|^2$
is monotone submodular. Nemhauser–Wolsey–Fisher (1978) then gives:

$$
U(S_k^{\text{greedy}}) \geq (1 - 1/e)\,U(S_k^{\star})
\qquad \text{at every } k = 1,\dots,N.
$$

The greedy rule picks at each step the splat that maximises the marginal
gain

$$
\Delta_i(S) = U(S \cup \{i\}) - U(S) = 2\sigma_i^{(|S|)} - \mathbf{G}_{ii},
\quad \sigma_i^{(k)} = \sum_{j \notin S_k} \mathbf{G}_{ij}.
$$

After selecting $i^\star$, the tail-row-sum updates as
$\sigma_j \leftarrow \sigma_j - \mathbf{G}_{j, i^\star}$ for all $j$.

### $3\sigma$ truncation and sparse Gram

Two splats whose $3\sigma$ ellipsoids do not overlap have
$\mathbf{G}_{ij}$ that is exactly zero under the truncation convention
used at render time, and negligibly small in any case. We therefore
build a *sparse* symmetric Gram matrix by:

1. Computing per-splat truncation radii $r_i = \sigma\sqrt{\lambda_{\max}(\Sigma_i)}$.
2. Querying a k-d tree on the splat centres for all $j$ with
   $\|\mu_i - \mu_j\| \leq r_i + r_{\max}$.
3. Pruning to the exact condition $\|\mu_i - \mu_j\| \leq r_i + r_j$.
4. Computing $\mathbf{G}_{ij}$ in closed form for the surviving pairs.

Empirical Gram density on the supp doc's reference datasets ranges from
0.3% (`tribolium_embryo`, $N = 41{,}099$) to 13% (`dapi_nuclei`,
$N = 1{,}274$).

## Parameterization

### `make_additive_lod(data, n_lods, *, method, breakpoints, truncation_sigmas, max_n_dense, seed)`

| Parameter           | Type                       | Default        | Notes |
|---------------------|----------------------------|----------------|-------|
| `data`              | `GSplatData`               | (required)     | Single-LOD or multi-LOD; flattened internally before ordering. |
| `n_lods`            | `int`                       | `4`            | Number of levels when `breakpoints='equal-count'`. Ignored when `breakpoints` is a list. |
| `method`            | `str`                       | `'greedy'`     | One of `greedy`, `self_energy`, `mass`, `amplitude`, `spectral`, `random`. |
| `breakpoints`       | `'equal-count'` / `list[int]` / `list[float]` | `'equal-count'` | See "Breakpoints" below. |
| `truncation_sigmas` | `float`                     | `3.0`          | $\sigma$ multiplier for sparse-Gram pruning. Overestimating costs runtime; underestimating drops valid pairs. |
| `max_n_dense`       | `int`                       | `2000`         | At $N \leq$ this, dense Gram + scan-greedy is faster than the heap-based lazy variant. |
| `seed`              | `int \| None`               | `None`         | Used only when `method='random'`. |

### Breakpoints

| Form                            | Semantics |
|---------------------------------|-----------|
| `'equal-count'`                 | `n_lods` levels with cumulative cuts at $\lceil k N / n_\text{lods}\rceil$. |
| `[c_1, c_2, \dots, c_L]` (ints) | Explicit cumulative splat counts. Must be strictly increasing; the list length sets the number of levels; if $c_L < N$, an implicit final cut at $N$ is added. |
| `[f_1, f_2, \dots, f_L]` (floats in $(0, 1]$) | Cumulative energy fractions. The smallest $k$ whose cumulative-utility crosses each $f$ is used. |

## Implementation Architecture

### Public functions

- `compute_additive_order(data, method, ...) -> np.ndarray` — length-`N`
  int64 permutation. Operates on the flattened view; the input may be
  multi-LOD.
- `make_additive_lod(data, n_lods, ..., breakpoints, ...) -> GSplatData` —
  computes order, resolves breakpoints to cutpoints, slices the
  ordered splat arrays into per-level `GSplatLOD` instances, and
  assembles a multi-LOD `GSplatData` whose `up_to_lod(k)` is a valid
  additive prefix.

### Private helpers (in `additive.py`)

- `_self_energy_score(data)`, `_mass_score(data)`, `_det_L(data)` —
  scalar-invariant scores; constants drop out of the ordering.
- `_truncation_radii(data, sigmas)` — per-splat $r_i$ via
  `np.linalg.eigvalsh`.
- `_build_sparse_gram(data, sigmas)` — `scipy.spatial.cKDTree` queries
  + closed-form $\mathbf{G}_{ij}$, returns `csr_matrix`.
- `_dense_greedy(gram)` — $O(N^2)$ scan-greedy, used at small $N$.
- `_lazy_greedy(gram_csr)` — heap-based greedy (Minoux 1978).
- `_spectral_order(gram_csr)` — leading eigenvector via `eigsh(k=1)`.
- `_residual_energy_curve(gram_csr, order)` — $E_0, \dots, E_N$ along
  the order.
- `_resolve_breakpoints(n, n_lods, breakpoints)` and
  `_energy_fraction_cuts(fracs, gram_csr, order)` — breakpoint resolution.

## Algorithm Pseudocode

### Greedy (scan, dense, $O(N^2)$)

```
σ_i ← Σ_j G_ij                            # tail-row-sum init
chosen ← {}
for k = 1..N:
    Δ_i ← 2σ_i - G_ii  for i ∉ chosen
    i* ← argmax_i Δ_i
    π(k) ← i*
    chosen ← chosen ∪ {i*}
    σ_j ← σ_j - G_{j, i*}  for all j
return π
```

### Lazy greedy (heap, sparse)

Maintains a max-heap keyed on stale upper-bound estimates of $\Delta_i$.
At each step, pop the top; if its `stale` step matches the current step,
accept. Otherwise re-evaluate $\Delta_i$ using the *current* $\sigma_i$
and push back. By submodularity (Minoux 1978), $\Delta_i$ is monotone
non-increasing, so the cached upper bound is always conservative.

### Sparse Gram via 3σ + k-d tree

```
r_i ← σ √λ_max(Σ_i) for all i
tree ← cKDTree(centers)
G ← empty CSR
for i = 1..N:
    candidates ← tree.query_ball_point(centers[i], r=r_i + r_max)
    for j in candidates with j > i:
        if ||μ_i - μ_j|| > r_i + r_j: continue
        G_ij ← closed-form K_ij
        if G_ij > 0: insert (i, j, G_ij) and (j, i, G_ij)
return G as CSR
```

## Usage Patterns

### Build a 4-level ladder by greedy

```python
from luxar.gsplats.lod import make_additive_lod

ladder = make_additive_lod(data, n_lods=4)
ladder.save("ladder.gsplats.zarr")
```

### Energy-fraction cutpoints (50 / 90 / 99 / 100%)

```python
ladder = make_additive_lod(data, breakpoints=[0.5, 0.9, 0.99, 1.0])
```

### Cheap fallback for very large datasets

```python
ladder = make_additive_lod(data, method="self_energy", n_lods=4)
```

## Expected Behavior

- **Permutation invariant**: every method returns a permutation of
  `range(N)` (verified in tests).
- **Monotone $E_k$**: for any returned order, the residual energy curve
  is non-increasing (theorem-level guarantee for greedy; held empirically
  for the heuristics on real data).
- **Greedy beats random and amplitude on AUC** in synthetic and real
  experiments (supp doc Experiments B & C).
- **Empty input**: `n_splats == 0` returns the input unchanged
  (single, empty LOD).

## Validation & Testing

`packages/luxar/src/luxar/gsplats/tests/test_additive_lod.py`:

- Permutation invariants for all six methods.
- Monotone $E_k$ along the produced order.
- Greedy AUC < random AUC on a fixed-seed mixture.
- Self-energy diagonal matches the closed-form $a_i^2 \pi^{D/2} |\Sigma_i|^{1/2}$.
- Equal-count breakpoints yield the expected cuts on $N=20, n_\text{lods}=4$.
- Energy-fraction breakpoints produce monotone increasing cumulative-energy LODs.
- Explicit count breakpoints respected.
- Empty input handled.

## Performance Considerations

- Pure-Python sparse-Gram construction is the bottleneck at large $N$.
  Reference: 33 s total at $N \approx 79{,}000$ on a typical Luxar
  microscopy dataset (0.66% Gram density).
- For $N \gtrsim 10^6$ recommend `method="self_energy"`.
- The dense path at $N \leq 2000$ uses ~32 MB Gram; well within memory.

## Future Extensions

- **GPU sparse-Gram**: PyTorch / CUDA implementation of the closed-form
  $\mathbf{G}_{ij}$ formula and lazy-greedy. Out of scope for this
  module's first cut.
- **Substitutive LOD**: at the scene-graph node level. Out of scope here.
- **View-conditioned ordering**: pair greedy with view-frustum
  filtering for interactive viewers (PRoGS-style).
- **Stochastic greedy** (Mirzasoleiman et al. 2015): trades a slightly
  weaker $1-1/e-\epsilon$ bound for near-linear time.

## References

- Supp. Doc. `additive_lod` (`luxar-paper/supp_doc/additive_lod/`).
- Nemhauser, Wolsey & Fisher (1978).
- Mallat & Zhang (1993) — matching pursuit.
- Minoux (1978) — accelerated greedy.

## Glossary

- **Additive LOD** — same `N` splats, ordered for prefix-monotone
  fidelity. This module.
- **Substitutive LOD** — `M < N` synthesised representative splats per
  coarser level. Not implemented here.
- **Removal LOD** — subset selection without synthesis or ordering;
  implemented in `cull_by_contribution`.
- **Gram matrix** — pairwise $L^2$ inner products of the splats.
- **Tail-row-sum** $\sigma_i$ — running sum of the unchosen entries in
  row $i$ of the Gram matrix; the marginal gain of adding splat $i$ is
  $2\sigma_i - \mathbf{G}_{ii}$.
