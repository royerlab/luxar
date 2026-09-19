# luxar.gsplats.lod._substitutive

Private support subpackage for the substitutive-LOD orchestrator
[`lod/substitutive.py`](../substitutive.py). The orchestrator
(`make_substitutive_lod`) reduces a fitted splat set to coarser
*representative* levels by Gaussian-mixture reduction; the three
partition algorithms that decide **which** splats merge into **which**
bins live here, behind a leading underscore.

The leading underscore is structural, not just convention: CPython cannot
host both a module `substitutive.py` and a package `substitutive/` in the
same directory, so the public import path `luxar.gsplats.lod.substitutive`
stays a file and its helpers live in `_substitutive/`.

## Overview

A substitutive level is built by **partitioning** the input splats into
`M = N / K` bins, then **synthesising one representative splat per bin**
via moment-matching + an L²-optimal amplitude (the per-bin merge math is
shared with the additive axis and lives in
[`lod/_kernels.py`](../_kernels.py)). This subpackage owns only the
partition step — producing a length-`N` int64 assignment tensor with
values in `[0, M)`. The orchestrator turns that assignment into
representatives.

Two partition strategies are implemented, plus a shared warm start:

| Module | Strategy | Complexity | Best regime |
|--------|----------|-----------|-------------|
| `warm_start.py` | Morton (Z-order) space-filling-curve chunking | `O(N log N)` | Always (seeds the `kmeans*` methods) |
| `kmeans_lloyd.py` | Morton warm start → vectorised cost-increment Lloyd | `O(N·k)` per pass | Large `N` (the workhorse) |
| `greedy.py` | Bottom-up Runnalls hierarchical merge (lazy-heap) | `~O(N·k·log(N·k))` | Small `N` / `K` (quality-leading) |

The orchestrator's `method="auto"` resolves per level: `greedy` at or
below ~5000 input splats, `kmeans_lloyd` above. See
[`lod/substitutive.py`](../substitutive.py) and [`lod/README.md`](../README.md)
for the method-selection policy and the supp-doc algorithm references.

## File Structure

```
_substitutive/
├── __init__.py        # _TINY numerical floor; module-level docstring
├── warm_start.py      # _morton_order, _morton_partition
├── kmeans_lloyd.py    # _segment_templates, _build_representatives_vectorized,
│                      #   _score_and_pick, _cost_increment_lloyd_vectorized
└── greedy.py          # _pair_merge_costs, _merge_two_clusters,
                       #   _greedy_partition, _estimate_cell_size
```

`__init__.py` exposes a single constant, `_TINY = 1e-30`, the numerical
floor used across the subpackage to guard divisions by per-bin mass or
template norm.

## Components

### `warm_start.py` — Morton space-filling-curve partition

- **`_morton_order(centres) -> np.ndarray`** — length-`N` int64 indices
  that sort splats along a Morton (Z-order) curve so that `centres[order]`
  is spatially coherent (curve neighbours are spatial neighbours).
  Normalises centres to a `2**bits_per_dim` grid (`bits_per_dim =
  min(21, 64 // ndim)`) and stable-argsorts the encoded codes via
  `luxar.io.ordering.morton_encode_nd`. `O(N log N)`.
- **`_morton_partition(centres, M, *, order=None) -> torch.Tensor`** —
  the `O(N log N)` warm start: sort along the curve, then chunk the sorted
  sequence into `M` contiguous, balanced bins of ~`N/M` splats each via
  `bin_of_pos = (arange(N) * M) // N`. Balanced and surjective onto
  `[0, M)`, so it is empty-bin-free for `N >= M`. Replaces a former global
  k-means++ warm start whose `O(M·N) = O(N²/K)` initialisation was
  intractable in the substitutive regime where `M = N/K` is large.

### `kmeans_lloyd.py` — vectorised cost-increment Lloyd refinement

Every per-bin quantity is a segment reduction
(`torch.Tensor.index_add_`) keyed by the bin assignment — no Python
per-splat or per-bin loops.

- **`_segment_templates(...)`** — per-bin moment-matched template plus the
  L²-amplitude ingredients, batched over all `M` bins at once. Returns
  `(mu_bar (M,D), Sigma_bar (M,D,D), inner (M,), norm_sq (M,),
  weights (N,))` where `inner = ⟨f_Sb, Ḡb⟩` and `norm_sq = ‖Ḡb‖²`. The
  per-bin L²-optimal amplitude is `inner / norm_sq`; the projection energy
  is `inner² / norm_sq`. `Sigma_bar` combines intra-bin spread
  (`Σ_i w_i Σ_i`) and inter-bin spread (`Σ_i w_i δδᵀ`).
- **`_build_representatives_vectorized(...)`** — vectorised construction of
  the per-bin representative splats: `(mu_bar, L_bar, a_star, new_colors)`.
  `Sigma_bar` is Cholesky-factored with a `1e-6·I` ridge; non-PD / empty
  bins fall back to identity and are culled by their zeroed amplitude.
  Colors are mass-weighted-averaged per bin.
- **`_score_and_pick(...)`** — for each splat, picks the candidate bin
  maximising the normalised Gaussian inner product `K(i, Ḡ_b) / ‖Ḡ_b‖`.
  Chunked over splats (default 50 000) to bound the `(chunk, C, D, D)`
  working set.
- **`_cost_increment_lloyd_vectorized(...)`** — the monotone refinement
  loop. Each pass reassigns every splat synchronously to its best
  candidate bin, rebuilds templates, and recomputes the global projection
  energy `P = Σ_b ⟨f,Ḡ⟩² / ‖Ḡ‖²`. A pass is **committed only if it
  strictly increases `P`** (relative `1e-9` margin), so the result is
  monotone and never worse than the warm start in projection energy `P`;
  the first non-improving pass stops iteration. Candidate bins for a splat are
  the *current* bins of its Morton-order neighbours (`±candidate_bins_k//2` on
  the curve),
  re-gathered against live assignments each pass — an `O(N·k)` gather that
  replaces a per-iteration spatial-hash kNN.

> **Determinism note.** The `1e-9` acceptance margin is calibrated for the
> deterministic float64 CPU path that `make_substitutive_lod` forces.
> `index_add_` segment reductions are non-deterministic on CUDA (atomic
> adds), so on a CUDA device a genuine improvement below the atomic-add
> noise floor may be rejected, stopping iteration early — only ever a
> marginally coarser refinement, never a wrong or NaN result. Set
> `torch.use_deterministic_algorithms(True)` in the caller for bit-exact
> CUDA behaviour.

### `greedy.py` — bottom-up Runnalls hierarchical merge

Repeatedly merges the active cluster pair with the smallest closed-form
merge residual energy (supp-doc Algorithm 4.2) until `M_target` clusters
remain — the same sequential cheapest-pair decisions as the classic
Runnalls reduction, but with an efficient incremental implementation.

- **`_pair_merge_costs(...)`** — batched residual energy
  `E* = ‖f‖² − ⟨f, Ḡ⟩² / ‖Ḡ‖²` (supp-doc Eq. 2.5) over `P` aligned
  candidate pairs, the vectorised equivalent of the scalar per-pair cost.
- **`_merge_two_clusters(...)`** — moment-matches two clusters into one
  representative `(mu_bar, L_bar, a_star)`, with the same `1e-6·I` ridge
  fallback on a failed Cholesky.
- **`_greedy_partition(centres, L, amps, M_target) -> torch.Tensor`** —
  the driver. Clusters live in fixed slots `[0, N)` with an `active`
  free-list (no array splicing). A lazy-deletion min-heap holds candidate
  edges tagged with per-cluster version counters; stale entries are
  discarded on pop. After a merge, only the `O(k)` edges from the new
  cluster to the union of the two parents' neighbours are recomputed.
  Candidate neighbours come from a spatial-hash kNN graph
  (`luxar.gsplats.spatial_hash.BatchedSpatialHashGrid`, `k = min(8, n)`),
  rebuilt only if the heap drains before the target; a disconnected
  residue is force-merged by Euclidean nearest-neighbour to guarantee
  progress. Roughly `O(N·k·log(N·k))` vs. the former `O(N²·k)` full
  re-scan.
- **`_estimate_cell_size(centres) -> float`** — spatial-hash cell size from
  a uniform-density approximation `(bbox_volume / N)^(1/D) · 2`, so the
  typical nearest-neighbour distance fits comfortably in shell 1.

## Invariants

- Partition functions return a length-`N` int64 assignment tensor with
  values in `[0, M)`; the orchestrator owns representative synthesis.
- `_morton_partition` is empty-bin-free for `N >= M`.
- Lloyd refinement is monotone in projection energy `P` (never worse than
  the warm start).
- The supported / verified path runs on **CPU with float64**, forced by
  `make_substitutive_lod` (MPS lacks float64).

## Dependencies

**Internal:**
- [`lod/_kernels.py`](../_kernels.py) — shared L² / Gram / moment-match /
  residual-energy kernels (Gaussian inner products, `sqrt_det_from_cholesky`,
  `kwise_moment_match_torch`, `l2_optimal_amplitude_torch`,
  `template_squared_norm_torch`).
- `luxar.io.ordering` — `morton_encode_nd`, `normalize_coords_to_grid`.
- `luxar.gsplats.spatial_hash` — `BatchedSpatialHashGrid` (greedy kNN graph).

**External:** `numpy`, `torch`.

## See Also

- [`lod/substitutive.py`](../substitutive.py) — orchestrator that calls
  these partitioners and synthesises representatives.
- [`lod/README.md`](../README.md) — LOD package overview, method-selection
  policy, and supp-doc algorithm references.
- [`lod/_kernels.py`](../_kernels.py) — shared merge math.
- [`gsplats/README.md`](../../README.md) — Gaussian splatting package.
