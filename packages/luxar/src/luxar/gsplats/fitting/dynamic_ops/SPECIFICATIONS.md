# luxar.gsplats.fitting.dynamic_ops - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-01-10

## Purpose

This package implements convergence-driven dynamic operations for adaptive Gaussian splatting. During optimization, it automatically adds splats where reconstruction error is high (seeding) and removes splats that contribute minimally (pruning).

**Key Innovation**: Operations are driven by convergence criteria rather than arbitrary thresholds, ensuring seeding/pruning decisions align with the optimization goal.

---

## Core Concepts

### Convergence-Based Detection

Unlike traditional approaches that use fixed thresholds, this implementation ties all decisions to the convergence criterion (`max_abs_error_threshold`):

- **Seeding**: Only seed where `|residual| > max_abs_error_threshold`
- **Pruning**: Only prune if removal doesn't cause local `|residual| > max_abs_error_threshold`

This ensures dynamic operations work toward the same goal as the optimizer.

### Spatial Fairness via Tiled Seeding

**Problem**: Global peak finding concentrates seeds in bright/high-error regions.

**Solution**: Divide image into tiles, find peaks per tile, use probabilistic selection to maintain expected seed count while ensuring spatial fairness.

### Three-Step Algorithm

1. **Residual Peak Analysis**: Find k strongest residual peaks (globally or tiled)
2. **Convergence-Based Seeding**: Add splats only where coverage is insufficient
3. **Principled Pruning**: Remove splats that don't impact reconstruction quality

---

## Data Structures

### DynamicOpsConfig

Configuration dataclass with all algorithm parameters:

```python
@dataclass
class DynamicOpsConfig:
    # Scheduling
    step_every: int = 50              # Run every N iterations

    # Step 1: Residual Peak Analysis
    k_max_residuals: int = 20         # Expected number of seeds
    nms_radius_vox: float = 2.0       # Non-maximum suppression radius

    # Tiled seeding (spatial fairness)
    enable_tiled_seeding: bool = True
    num_tiles_per_dim: int | None = None  # Auto-select based on ndim

    # Step 2: Adaptive Operations
    min_contribution_threshold: float = 0.05   # Minimum influence for existing splat
    relative_contribution_factor: float = 0.1  # Adaptive amplitude threshold
    lr_boost_factor: float = 1.5               # LR multiplier for boosting
    boost_influence_threshold: float = 0.05   # Minimum influence to boost

    # Step 3: Pruning
    pruning_percentile: float = 5.0   # % of least important to consider
    min_splats_to_keep: int = 10      # Minimum splat count

    # Seeding parameters
    init_sigma_vox: float = 0.5       # Initial sigma for new splats
```

### Input/Output Types

```python
def apply_dynamic_operations(
    model: GaussianSplatModel,           # Model to modify
    optimizer: PerSplatAdam,             # Per-splat optimizer
    scheduler: PerSplatScheduler,        # Learning rate scheduler
    V_target: torch.Tensor,              # Target image/volume
    V_pred: torch.Tensor,                # Current prediction
    cfg: DynamicOpsConfig,               # Configuration
    current_lr: float,                   # Current learning rate
    max_abs_error_threshold: float,      # Convergence threshold
    device: torch.device,                # Compute device
    verbose: bool = False,               # Logging flag
) -> Tuple[Optimizer, Scheduler, bool]:  # Returns updated optimizer, scheduler, topology_changed
```

---

## Algorithms

### Algorithm 1: Residual Peak Finding

#### Global Mode (`enable_tiled_seeding=False`)

```
Input: residual tensor R, k_max_residuals, nms_radius_vox
Output: List of peak coordinates sorted by magnitude (descending)

1. Compute |R| (absolute residual)
2. Apply max-pooling with kernel_size = 2 * nms_radius_vox + 1
3. Find local maxima: peaks where |R| == max_pooled AND |R| > 0
4. Sort by magnitude (descending)
5. Return top k_max_residuals peaks
```

**Complexity**: O(n_voxels) for max-pooling, O(k log k) for sorting

#### Tiled Mode (`enable_tiled_seeding=True`)

```
Input: residual R, k_max_residuals, nms_radius_vox, num_tiles_per_dim
Output: List of peak coordinates with expected count ≈ k_max_residuals

1. Calculate total_tiles = num_tiles_per_dim^ndim
2. Calculate k_per_tile = k_max_residuals / total_tiles

3. For each tile:
   a. Extract tile region from |R|
   b. Find local maxima within tile (NMS with nms_radius_vox)

   c. If k_per_tile >= 1.0 (deterministic mode):
      - Keep top floor(k_per_tile) peaks from tile

   d. If k_per_tile < 1.0 (probabilistic mode):
      - Find best peak in tile
      - Strong peaks (≥ 75th percentile of global |R|): always keep
      - Weak peaks: keep with probability k_per_tile

4. Combine all selected peaks
5. Sort by magnitude (descending)
6. Return peaks list
```

**Auto-scaling for num_tiles_per_dim**:
- 2D: 16 (256 tiles)
- 3D: 6 (216 tiles)
- 4D: 4 (256 tiles)
- 5D+: 2 (2^d tiles)

**Rationale**: Maintains ~200-300 tiles across dimensions for consistent granularity.

### Algorithm 2: Convergence-Based Seeding

```
Input: peak_locations, residual R, model, cfg, max_abs_error_threshold
Output: Number of splats seeded

1. CONVERGENCE GUARD:
   strongest_residual = |R[peak_locations[0]]|
   if strongest_residual < max_abs_error_threshold:
       return 0  # Already converged

2. For each peak location p in peak_locations:
   local_residual = |R[p]|

   a. COVERAGE CHECK:
      if local_residual <= max_abs_error_threshold:
          continue  # Coverage sufficient at this location

   b. INFLUENCE CHECK:
      influential_splat, influence = find_splat_with_influence_at(p, model)

      if influential_splat exists AND influence >= boost_influence_threshold:
          # Existing splat covers this area - boost its LR instead
          boost_learning_rate(optimizer, influential_splat, lr_boost_factor)
          continue

      elif influential_splat exists:
          # Existing splat present but influence too low
          continue  # Let optimizer adjust existing splat

   c. SEEDING (no existing coverage):
      min_amp = local_residual * relative_contribution_factor

      if |R[p]| >= min_amp AND |R[p]| >= min_contribution_threshold:
          # Create new splat at peak location
          center = p
          amplitude = |R[p]|
          L = I * init_sigma_vox  # Isotropic covariance
          sharpness = 2.0         # Standard Gaussian

          add_splat(model, center, L, amplitude, sharpness)
          seeded_count += 1

3. Return seeded_count
```

### Algorithm 3: Influence Detection

```
Input: location p, model with N splats
Output: (splat_index, influence_value) or (-1, 0.0)

1. Get model parameters: centers, Ls, amps, sharpnesses

2. EARLY FILTERING (Euclidean distance):
   max_sigma = max(diag(L)) for each splat
   max_reach = 6 * max_sigma
   candidates = splats where ||p - center|| <= max_reach

3. For each candidate splat i:
   # Compute Mahalanobis distance
   diff = p - centers[i]
   y = solve_triangular(L[i], diff)  # L @ y = diff
   squared_dist = y^T @ y

   # Gaussian influence
   influence[i] = amps[i] * exp(-0.5 * squared_dist)

4. max_influence, best_idx = argmax(influence)

5. If max_influence >= min_influence_threshold:
   return (best_idx, max_influence)
   else:
   return (-1, 0.0)
```

**Complexity**: O(N) worst case, but early filtering reduces to O(k) where k << N

### Algorithm 4: Principled Pruning

```
Input: model, V_target, V_pred, cfg, max_abs_error_threshold
Output: List of splat indices to remove

1. MINIMUM CHECK:
   if model.n_splats() <= min_splats_to_keep:
       return []

2. IMPORTANCE CALCULATION:
   For each splat i:
       # Importance = amplitude × volume
       importance[i] = amps[i] * prod(diag(L[i]))

3. CANDIDATE SELECTION:
   n_candidates = floor(n_splats * pruning_percentile / 100)
   candidates = argsort(importance)[:n_candidates]  # Least important

4. LOCAL IMPACT TESTING:
   removable = []
   For each candidate splat i:
       # Compute splat's influence region (3σ ellipse)
       influence_mask = compute_influence_region(centers[i], L[i], radius=3.0)

       # Estimate prediction without this splat
       single_contribution = render_single_splat(i)
       pred_without = V_pred[influence_mask] - single_contribution[influence_mask]

       # Check local convergence
       local_residual = |pred_without - V_target[influence_mask]|
       max_local_error = max(local_residual)

       if max_local_error <= max_abs_error_threshold:
           removable.append(i)  # Safe to remove

5. Return removable[:n_splats - min_splats_to_keep]  # Respect minimum
```

**Key Optimization**: Uses single-splat subtraction instead of full re-render, reducing complexity from O(n_splats × n_voxels) to O(n_voxels).

### Algorithm 5: n-Dimensional Max Pooling

For dimensions > 3 where native max_poolN doesn't exist:

```
Input: data tensor (any dimension), kernel_size
Output: max-pooled tensor (same shape)

1. result = data
2. For each dimension d in 0..ndim:
   a. Permute to move dimension d to last position
   b. Reshape to (batch, 1, length) for max_pool1d
   c. Apply max_pool1d with kernel_size, stride=1, padding=kernel_size//2
   d. Reshape back to original dimension order
   e. Permute back to original axis order

3. Return result
```

This separable approach is equivalent to full nD max pooling for local maxima detection.

---

## Validation Rules

### Input Validation

1. **Residual tensor**: Must match target shape
2. **k_max_residuals**: Must be > 0
3. **nms_radius_vox**: Must be > 0
4. **num_tiles_per_dim**: If provided, must be >= 2
5. **pruning_percentile**: Must be in [0, 100]

### Invariants

1. **Splat count**: Always >= min_splats_to_keep after pruning
2. **Convergence alignment**: Operations only where |residual| > threshold
3. **Spatial constraint**: New splats separated by >= nms_radius_vox

### Error Handling

- **Device mismatch**: Caught and logged, operation skipped
- **Invalid coordinates**: Clamped to valid range
- **Singular matrices**: Graceful fallback to 0 influence

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Typical Time (512³, 1000 splats) |
|-----------|------------|----------------------------------|
| Peak finding (global) | O(n_voxels) | ~10 ms |
| Peak finding (tiled) | O(n_voxels + n_tiles) | ~15 ms |
| Influence detection | O(N × candidates) | ~5 ms |
| Single splat render | O(n_voxels) | ~20 ms |
| Pruning analysis | O(k × n_voxels) | ~100 ms |
| **Total per step** | O(n_voxels + k × N) | ~150 ms |

### Space Complexity

- Residual tensor: O(n_voxels) - shared with loss computation
- Peak coordinates: O(k_max_residuals × ndim)
- Influence mask: O(n_voxels) - reused
- **Peak overhead**: ~1.5× single tensor

### Recommended Settings by Data Size

| Volume Size | step_every | k_max_residuals | pruning_percentile |
|-------------|------------|-----------------|-------------------|
| < 128³ | 25 | 10 | 10.0 |
| 128³-256³ | 50 | 20 | 5.0 |
| 256³-512³ | 75 | 30 | 3.0 |
| > 512³ | 100 | 50 | 2.0 |

---

## Cross-Language Compatibility

This is a Python-only component (runs during optimization). No cross-language constraints.

---

## Related Specifications

- **Initial Seeding**: `seeds/SPECIFICATIONS.md` - Pre-optimization seed generation
- **Model**: `models/gsplats/SPECIFICATIONS.md` - GaussianSplatModel structure
- **Optimizer**: `optim/SPECIFICATIONS.md` - PerSplatAdam optimizer
- **Rendering**: `models/gsplats/rendering_core.py` - Gaussian rendering

---

## Design Decisions

### Why Convergence-Based Detection?

**Problem**: Fixed thresholds (e.g., "seed if residual > 0.1") don't adapt to:
- Different data intensity ranges
- Varying convergence requirements
- Progress during optimization

**Solution**: Use the convergence criterion itself as the threshold. If the user specifies `max_abs_error=0.01`, seeding only happens where error > 0.01.

### Why Tile-Based Seeding?

**Problem**: Bright regions dominate global peak finding, leaving dim regions under-seeded.

**Solution**: Tiled seeding with probabilistic selection ensures all spatial regions get attention regardless of brightness. The probabilistic scheme maintains expected seed count while providing spatial fairness.

### Why Single-Voxel Splat Initialization?

**Problem**: Large initial splats (σ > 1) cause error spikes in neighboring pixels during seeding.

**Solution**: Initialize with σ = 0.5 (single-voxel scale). The optimizer can grow splats as needed, but initial impact is localized.

**Quantitative**: σ=0.5 gives 13% influence on neighbors vs 60% for σ=1.5.

### Why LR Boosting Instead of Splitting?

**Problem**: Splitting splats adds complexity and can create unstable configurations.

**Solution**: When an existing splat covers a problematic region but hasn't converged, boost its learning rate instead of adding new splats. This is simpler and more stable.

---

## Changelog

- **v1.1.0** (2026-01-10): Complete specification with algorithms and rationale
- **v1.0.0** (2026-01-02): Initial placeholder specification