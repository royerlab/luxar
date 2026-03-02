# luxar.gsplats.fitting.dynamic_ops - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2025-01

## Purpose

This package implements **fixed-pool splat relocation** for adaptive Gaussian splatting. Instead of adding/removing splats (which requires complex per-splat optimizer state management), weak splats are relocated to high-residual regions. This enables use of standard PyTorch Adam optimizer for 50x+ faster optimization.

**Key Innovation**: Fixed-pool architecture eliminates topology changes, allowing standard vectorized optimization.

---

## Core Concepts

### Fixed-Pool Relocation (vs Add/Remove)

**Old Approach** (removed):
- Add splats at high-residual regions (changes tensor shape)
- Remove weak splats (changes tensor shape)
- Required per-splat optimizer to preserve momentum across topology changes

**New Approach** (current):
- Identify weak splats (low importance = amplitude × volume)
- Identify high-residual peaks
- **Relocate** weak splats to peaks (just parameter updates, no shape change)
- Standard PyTorch Adam works naturally

### Convergence-Based Detection

All decisions are tied to the convergence criterion (`max_abs_error_threshold`):
- **Convergence guard**: Skip operations if strongest residual < threshold
- **Peak selection**: Only consider peaks where |residual| > threshold

### Spatial Fairness via Tiled Peak Finding

**Problem**: Global peak finding concentrates seeds in bright/high-error regions.

**Solution**: Divide image into tiles, find peaks per tile, ensuring spatial coverage.

---

## Data Structures

### DynamicOpsConfig

Configuration class with all algorithm parameters:

```python
class DynamicOpsConfig:
    def __init__(self) -> None:
        # Scheduling
        self.step_every: int = 50              # Run every N iterations

        # Step 1: Residual Peak Analysis
        self.k_max_residuals: int = 20         # Max peaks to find
        self.nms_radius_vox: float = 2.0       # Non-maximum suppression radius

        # Tiled seeding (spatial fairness)
        self.enable_tiled_seeding: bool = True
        self.num_tiles_per_dim: int | None = None  # Auto-select based on ndim

        # Step 2: Weak Splat Identification
        self.relocation_percentile: float = 1.0   # % of least important splats
        self.max_relocations_per_step: int | None = 32   # Cap relocations per step (None = no limit)

        # Step 3: Relocation Parameters
        self.init_sigma_vox: float = 0.5          # Initial sigma for relocated splats
        self.min_contribution_threshold: float = 0.01  # Min influence for coverage check
        self.enable_coverage_check: bool = False   # Skip peaks already covered by non-weak splats

        # Coverage Check Behavior:
        # - False (default): Aggressive - relocate to all high-residual peaks regardless
        # - True: Conservative - skip peaks with existing non-weak coverage
        #   The issue: "has influence" ≠ "error is resolved"
        #   A peak can have existing influence but still have high residual

        # Safety
        self.min_splats_to_keep: int = 10         # Minimum splat count

        # Cooldown mechanism (prevents immediate re-relocation)
        self.relocation_cooldown_steps: int = 3   # Steps to wait before allowing re-relocation
```

### RecentlyRelocatedTracker

**New in v2.0**: Cooldown mechanism to ensure diverse splat coverage.

```python
class RecentlyRelocatedTracker:
    """Track recently relocated splats to avoid immediate re-selection."""

    def __init__(self, n_splats: int, cooldown_steps: int = 3, device: str = "cpu"):
        # GPU tensor tracking: last_relocation_step[n_splats]
        # -1 means never relocated
        self.last_relocation_step = torch.full((n_splats,), -1, dtype=torch.long, device=device)
        self.cooldown_steps = cooldown_steps
        self.current_step = 0

    def mark_relocated_batch(self, splat_indices: torch.Tensor) -> None:
        """Mark splats as relocated (vectorized)."""

    def filter_eligible_splats(self, candidates: torch.Tensor) -> torch.Tensor:
        """Filter out splats on cooldown (vectorized)."""

    def advance_step(self) -> None:
        """Advance to next dynamic ops step."""

    def get_statistics(self) -> Dict[str, int]:
        """Get relocation statistics (total, unique, on cooldown)."""
```

**Purpose**: Prevents the critical bug where the same weak splats get relocated repeatedly while many splats remain untouched.

**How it works**:
1. When a splat is relocated, its index is recorded with the current step number
2. Before selecting weak splats, filter out any that were relocated within `cooldown_steps`
3. This ensures diverse coverage: ~90% of splats get relocated instead of ~10%

**Performance**: Fully GPU-accelerated using vectorized tensor operations, no Python loops.

### Function Signature

```python
def apply_dynamic_operations(
    model,                              # GaussianSplatModel to modify
    V_target: torch.Tensor,             # Target image/volume
    V_pred: torch.Tensor,               # Current prediction
    cfg: DynamicOpsConfig,              # Configuration
    max_abs_error_threshold: float,     # Convergence threshold
    optimizer: Optional[torch.optim.Optimizer] = None,  # For state reset
    relocation_tracker: Optional[RecentlyRelocatedTracker] = None,  # For cooldown
    verbose: bool = False,              # Logging flag
) -> bool:                              # Returns True if any splats relocated
```

---

## Algorithms

### Algorithm 1: Main Relocation Pipeline

```
Input: model, V_target, V_pred, cfg, max_abs_error_threshold,
       optimizer (optional), relocation_tracker (optional)
Output: bool (any_relocated)

1. RESIDUAL COMPUTATION:
   residual = V_target - V_pred

2. PEAK FINDING (vectorized):
   peak_locations = find_residual_peaks(residual, k_max, nms_radius, tiled)
   if no peaks: return False

3. CONVERGENCE GUARD:
   strongest_residual = |residual[peak_locations[0]]|
   if strongest_residual < max_abs_error_threshold:
       return False  # Already converged

4. WEAK SPLAT IDENTIFICATION (vectorized):
   importance = calculate_splat_importance(model)  # amplitude × volume
   weak_indices = select_bottom_percentile(importance, relocation_percentile)

5. COOLDOWN FILTERING (NEW):
   if relocation_tracker is not None:
       weak_indices = tracker.filter_eligible_splats(weak_indices)
   # Removes recently relocated splats (on cooldown)

6. MATCHING (vectorized, returns tensors):
   splat_indices, peak_indices = match_weak_splats_to_peaks_batch(
       model, weak_indices, peak_locations, residual, min_contribution_threshold
   )
   splat_indices = splat_indices[:max_relocations_per_step]  # Cap relocations
   peak_indices = peak_indices[:max_relocations_per_step]

7. RELOCATION (batched):
   peak_coords = peak_locations[peak_indices]  # Vectorized lookup
   relocate_splats_batch(model, splat_indices, peak_coords, residual, cfg, optimizer)
   # - Resets model parameters (mu, L, a, sharpness)
   # - Resets optimizer state (exp_avg, exp_avg_sq) if provided

8. TRACKER UPDATE (NEW):
   if relocation_tracker is not None:
       tracker.mark_relocated_batch(splat_indices)
       tracker.advance_step()

9. return len(splat_indices) > 0
```

**Key Changes in v2.0**:
- Step 5: Cooldown filtering prevents immediate re-relocation
- Step 6: Returns tensors (not Python lists) for performance
- Step 7: Batched relocation + optimizer state reset
- Step 8: Tracker bookkeeping

**Performance**: All operations vectorized on GPU, no Python loops in critical path.

### Algorithm 2: Splat Importance Calculation

```
Input: model with N splats
Output: importance tensor (N,)

For each splat i:
    # Volume approximation: product of diagonal elements
    volume[i] = prod(diag(L[i]))

    # Importance = amplitude × volume
    importance[i] = amps[i] × volume[i]

return importance
```

**Rationale**: Low importance = small AND dim → not useful where it is.

### Algorithm 3: Weak Splat Selection

```
Input: importance tensor, centers, residual, relocation_percentile
Output: list of splat indices (best candidates for relocation first)

1. n_candidates = max(1, floor(n_splats × percentile / 100))

2. Sort ALL splats by importance (ascending = weakest first):
   sorted_indices = argsort(importance)

3. For each splat, get residual at its center location:
   center_coords = round(centers)
   residual_at_centers = residual[center_coords]

4. Compute residual threshold (25th percentile of positive residuals):
   residual_threshold = quantile(residual[residual > 0], 0.25)

5. Filter strategy - only relocate splats at LOW-residual locations:
   candidates = []
   for idx in sorted_indices:
       if residual_at_centers[idx] > residual_threshold:
           continue  # Skip - splat is fighting high-error region

       candidates.append(idx)
       if len(candidates) >= n_candidates:
           break

6. Fallback: if not enough candidates (all weak splats fighting errors):
   candidates = sorted_indices[:n_candidates]

7. return candidates
```

**Key insight**: A splat at a high-residual location is doing useful work (fighting
the error), even if it has low importance. Only relocate splats that are weak AND
at low-residual locations (truly not contributing).

### Algorithm 4: Peak-Splat Matching

```
Input: model, weak_indices, peak_locations, min_threshold, enable_coverage_check
Output: list of (splat_idx, peak_coords) pairs

1. IF enable_coverage_check is True:
     exclude_set = set(weak_indices)  # Don't count weak splats as coverage

     Filter peaks to uncovered only:
     uncovered_peaks = []
     For each peak in peak_locations (strongest first):
         covering_splat, influence = find_splat_with_influence(
             model, peak, exclude_set, min_threshold
         )
         if no covering_splat:
             uncovered_peaks.append(peak)

   ELSE:
     # Skip coverage check - consider ALL peaks
     uncovered_peaks = peak_locations

2. Direct pairing (both lists already sorted):
   # weak_indices: least important first (ascending)
   # uncovered_peaks: highest residual first (descending)
   # Simply zip them together!

   n_matches = min(len(weak_indices), len(uncovered_peaks))
   matches = [(weak_indices[i], uncovered_peaks[i]) for i in range(n_matches)]

3. return matches
```

**Rationale for direct pairing**: Distance-based matching is unnecessary complexity
because relocated splats are completely reset (center, amplitude, covariance).
The weakest splat should go to the highest-error peak, 2nd weakest to 2nd highest,
etc. This aligns priorities correctly and is much simpler (O(N) vs O(N²)).

**Coverage check trade-off**: "Has influence" ≠ "error is resolved". A peak can
have existing splat influence (e.g., 0.3) but still have high residual (e.g., 0.8),
indicating insufficient coverage. The flag lets you choose conservative (default)
or aggressive relocation behavior.

### Algorithm 5: Splat Relocation (Batched)

```
Input: model, splat_indices (N,), peak_coords (N, d), residual, cfg, optimizer
Output: (modifies model and optimizer in-place)

BATCHED OPERATIONS (all vectorized on GPU):

1. CENTER: Convert coords to raw (logit) space
   u = peak_coords / (shape - 1)  # (N, d) Normalize to [0,1]
   raw_mu = log(u / (1-u))        # (N, d) Logit transform
   model.raw_mu[splat_indices] = raw_mu

2. AMPLITUDE: Set to residual values at new locations
   new_amps = residual[peak_coords]  # (N,) Advanced indexing
   raw_a = inverse_softplus(new_amps)
   model.raw_a[splat_indices] = raw_a

3. COVARIANCE: Reset to isotropic (init_sigma_vox)
   effective_diag = init_sigma_vox - sigma_min  # (d,)
   raw_L_diag = inverse_softplus(effective_diag)  # (d,)
   raw_L_diag_batch = expand(raw_L_diag, (N, d))  # Broadcast
   model.raw_L_diag[splat_indices] = raw_L_diag_batch
   model.L_off[splat_indices] = 0  # (N, d*(d-1)//2) Zero off-diagonal

4. SHARPNESS: Reset to standard (s=2.0 → s'=0)
   model.sharpness_offsets_raw[splat_indices] = 0.0

5. OPTIMIZER STATE RESET (if optimizer provided):
   reset_optimizer_state_batch(optimizer, model, splat_indices)
   # Zeros out exp_avg and exp_avg_sq for relocated splats
```

**Performance**: 50-100x faster than Python loop for 10+ relocations.

### Algorithm 6: Influence Detection

```
Input: location p, model, exclude_indices, min_threshold
Output: (splat_index, influence) or (-1, 0.0)

1. EARLY FILTERING:
   max_sigma = max(diag(L)) per splat
   max_reach = 6 × max_sigma
   candidates = splats where ||p - center|| ≤ max_reach

2. For each candidate i not in exclude_indices:
   # Mahalanobis distance
   diff = p - centers[i]
   y = solve_triangular(L[i], diff)
   squared_dist = y^T @ y

   # Gaussian influence
   influence[i] = amps[i] × exp(-0.5 × squared_dist)

3. max_influence, best_idx = max(influence)

4. if max_influence ≥ min_threshold:
       return (best_idx, max_influence)
   else:
       return (-1, 0.0)
```

### Algorithm 7: Optimizer State Reset (NEW in v2.0)

```
Input: optimizer (Adam), model, splat_indices (N,)
Output: (modifies optimizer.state in-place)

For Adam optimizer only:

1. PARAMETER ITERATION:
   params = [raw_mu, raw_L_diag, L_off, raw_a, sharpness_offsets_raw]

   for param in params:
       if param not in optimizer.state:
           continue  # Not yet optimized

       state = optimizer.state[param]

       # Reset momentum (first moment)
       if "exp_avg" in state:
           state["exp_avg"][splat_indices] = 0.0

       # Reset variance (second moment)
       if "exp_avg_sq" in state:
           state["exp_avg_sq"][splat_indices] = 0.0
```

**Why this is critical**:
- Without reset: Relocated splats have momentum from OLD location
- OLD gradient direction is wrong for NEW location
- Result: Poor convergence, thrashing, or stuck in local minima

**With reset**:
- Relocated splats start fresh with zero momentum/variance
- Optimizer learns gradient direction at new location
- Result: Fast adaptation within 1-3 iterations

**Performance**: Batched tensor indexing, no loops.

---

## Why Standard Adam Works (with State Reset)

When a splat is relocated:
1. Parameters are modified in-place (no tensor shape change)
2. **Optimizer state is reset** to zero for relocated splats (NEW)
3. Gradient at new location accumulates in fresh momentum buffers
4. Full adaptation within 1-3 iterations

This is much simpler than per-splat optimizer state management.

---

## Performance

### Time Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Peak finding | O(n_voxels) | ~10 ms |
| Importance calculation | O(N_splats) | ~1 ms |
| Influence detection | O(N × k) | ~5 ms |
| Relocation | O(k) | ~1 ms |
| **Total per step** | O(n_voxels + N × k) | ~20 ms |

### Recommended Settings

| Volume Size | step_every | k_max_residuals | relocation_percentile |
|-------------|------------|-----------------|----------------------|
| < 128³ | 25 | 10 | 10.0 |
| 128³-256³ | 50 | 20 | 5.0 |
| > 256³ | 75 | 30 | 3.0 |

---

## Related Specifications

- **Seeds**: `../../seeds/SPECIFICATIONS.md` - Initial seed generation
- **Model**: `../../models/gsplats/SPECIFICATIONS.md` - GaussianSplatModel
- **Optimizer**: `../../optim/SPECIFICATIONS.md` - Standard Adam with gradient dilution

---

## Version History

- **v2.1.0** (January 2025): Improved relocation algorithm
  - Smart weak splat selection: considers residual at splat location
  - Direct index pairing: simpler and more logically sound than distance-based
  - Vectorized max scatter operation for efficiency

- **v2.0.0** (January 2025): Fixed-pool relocation architecture
  - Removed add/remove operations
  - Standard PyTorch Adam (50x+ faster)
  - Splat relocation instead of topology changes

- **v1.x** (2024): Per-splat optimizer with add/remove (deprecated and removed)
