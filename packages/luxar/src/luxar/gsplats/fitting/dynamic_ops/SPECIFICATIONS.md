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
        self.relocation_percentile: float = 5.0   # % of least important splats
        self.max_relocations_per_step: int = 10   # Cap relocations per step

        # Step 3: Relocation Parameters
        self.init_sigma_vox: float = 0.5          # Initial sigma for relocated splats
        self.min_contribution_threshold: float = 0.01  # Min influence for coverage check

        # Safety
        self.min_splats_to_keep: int = 10         # Minimum splat count
```

### Function Signature

```python
def apply_dynamic_operations(
    model,                              # GaussianSplatModel to modify
    V_target: torch.Tensor,             # Target image/volume
    V_pred: torch.Tensor,               # Current prediction
    cfg: DynamicOpsConfig,              # Configuration
    max_abs_error_threshold: float,     # Convergence threshold
    verbose: bool = False,              # Logging flag
) -> bool:                              # Returns True if any splats relocated
```

---

## Algorithms

### Algorithm 1: Main Relocation Pipeline

```
Input: model, V_target, V_pred, cfg, max_abs_error_threshold
Output: bool (any_relocated)

1. RESIDUAL COMPUTATION:
   residual = V_target - V_pred

2. PEAK FINDING:
   peak_locations = find_residual_peaks(residual, k_max, nms_radius, tiled)
   if no peaks: return False

3. CONVERGENCE GUARD:
   strongest_residual = |residual[peak_locations[0]]|
   if strongest_residual < max_abs_error_threshold:
       return False  # Already converged

4. WEAK SPLAT IDENTIFICATION:
   importance = calculate_splat_importance(model)  # amplitude × volume
   weak_indices = select_bottom_percentile(importance, relocation_percentile)

5. MATCHING:
   matches = match_weak_splats_to_peaks(
       model, weak_indices, peak_locations, residual, min_contribution_threshold
   )
   matches = matches[:max_relocations_per_step]  # Cap relocations

6. RELOCATION:
   for (splat_idx, peak_coords) in matches:
       relocate_splat(model, splat_idx, peak_coords, residual, cfg)

7. return len(matches) > 0
```

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
Input: importance tensor, relocation_percentile
Output: list of splat indices (least important first)

1. n_candidates = max(1, floor(n_splats × percentile / 100))
2. sorted_indices = argsort(importance)  # Ascending
3. return sorted_indices[:n_candidates]
```

### Algorithm 4: Peak-Splat Matching

```
Input: model, weak_indices, peak_locations, residual, min_threshold
Output: list of (splat_idx, peak_coords) pairs

1. available_weak = set(weak_indices)
2. exclude_set = set(weak_indices)  # Don't count weak splats as coverage

3. For each peak in peak_locations (strongest first):
   if no available weak splats: break

   # Check if non-weak splat already covers this peak
   covering_splat, influence = find_splat_with_influence(
       model, peak, exclude_set, min_threshold
   )

   if covering_splat exists:
       continue  # Peak already covered

   # Find closest available weak splat
   closest = argmin(||centers[weak] - peak||) for weak in available_weak

   matches.append((closest, peak))
   available_weak.remove(closest)

4. return matches
```

### Algorithm 5: Splat Relocation

```
Input: model, splat_idx, new_center_coords, residual, cfg
Output: (modifies model in-place)

1. CENTER: Convert coords to raw (logit) space
   u = coords / (shape - 1)  # Normalize to [0,1]
   raw_mu = log(u / (1-u))   # Logit transform
   model.raw_mu[splat_idx] = raw_mu

2. AMPLITUDE: Set to residual value at new location
   new_amp = |residual[coords]|
   raw_a = inverse_softplus(new_amp)
   model.raw_a[splat_idx] = raw_a

3. COVARIANCE: Reset to isotropic (init_sigma_vox)
   effective_diag = init_sigma_vox - sigma_min
   raw_L_diag = inverse_softplus(effective_diag)
   model.raw_L_diag[splat_idx] = raw_L_diag
   model.L_off[splat_idx] = 0  # Zero off-diagonal

4. SHARPNESS: Keep unchanged (optimizer will adjust)
```

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

---

## Why Standard Adam Works

When a splat is relocated:
1. Parameters are modified in-place (no tensor shape change)
2. Adam's momentum buffers remain at same indices
3. Gradient at new location overwrites stale momentum
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

- **Seeds**: `../seeds/SPECIFICATIONS.md` - Initial seed generation
- **Model**: `../../models/gsplats/SPECIFICATIONS.md` - GaussianSplatModel
- **Optimizer**: `../../optim/SPECIFICATIONS.md` - Standard Adam with gradient dilution

---

## Version History

- **v2.0.0** (January 2025): Fixed-pool relocation architecture
  - Removed add/remove operations
  - Standard PyTorch Adam (50x+ faster)
  - Splat relocation instead of topology changes

- **v1.x** (2024): Per-splat optimizer with add/remove (deprecated and removed)
