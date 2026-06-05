# Dynamic Operations for Adaptive Gaussian Splatting

Fixed-pool splat relocation that adaptively redistributes Gaussian splats during optimization.

## Purpose

Dynamic operations use **fixed-pool splat relocation** instead of add/remove operations. This enables use of standard PyTorch Adam optimizer (50x+ faster) by keeping tensor shapes constant throughout optimization.

### How It Works

1. **Find residual peaks**: Detect high-error regions using non-maximum suppression
2. **Identify weak splats**: Find splats with low importance (amplitude x volume)
3. **Filter by cooldown**: Skip recently relocated splats (prevents re-relocation)
4. **Relocate**: Move weak splats to high-residual peaks (parameter updates only)
5. **Reset optimizer state**: Zero out Adam momentum/variance for relocated splats

## Key Classes

### `DynamicOpsConfig`
Dataclass containing all parameters for the relocation algorithm: scheduling, peak finding, tiled seeding, weak splat identification, relocation parameters, cooldown, and safety limits.

### `RecentlyRelocatedTracker`
Tracks recently relocated splats using a cooldown mechanism to prevent immediate re-relocation. Uses GPU tensors for bulk filtering. Key methods:
- `mark_relocated_batch(splat_indices)` - Mark splats as relocated
- `filter_eligible_splats(candidate_indices)` - Filter out splats still on cooldown
- `advance_step()` - Advance to next dynamic ops step
- `get_statistics()` - Get relocation statistics

### Public Functions (re-exported from `__init__.py`)

- `apply_dynamic_operations()` - Main entry point: orchestrates the full relocation pipeline (in `operations.py`)
- `_find_residual_peaks()` - Find strongest residual peaks with NMS (global or tiled mode; in `peak_finding.py`)
- `_calculate_splat_importance()` - Compute importance = amplitude x prod(diag(L)) (in `operations.py`)
- `_select_weak_splats()` - Select weak splats safe to relocate (low importance + low residual + off cooldown; in `operations.py`)

### Internal Functions (module-level, imported directly from their module in tests)

In `peak_finding.py`:
- `_find_residual_peaks_global()` - Global peak finding (all regions compete)
- `_find_residual_peaks_tiled()` - Tiled peak finding for spatial fairness
- `_find_peaks_in_tile()` - Find up to `k_max` peaks within a single tile via NMS
- `_separable_nd_max_pool()` - Efficient nD max pooling via separable 1D passes

In `operations.py`:
- `_compute_peak_coverage_batch()` - Vectorized coverage check for all peaks
- `_match_weak_splats_to_peaks_batch()` - Vectorized peak-splat matching
- `_match_weak_to_peaks_direct()` - Direct pairing: weakest splat -> highest-error peak
- `_relocate_splats_batch()` - Batched relocation with optimizer state reset
- `_reset_optimizer_state_batch()` - Zero out Adam momentum/variance for relocated splats

## Key Features

### Fixed-Pool Architecture (50x+ Faster)

**Problem**: Traditional add/remove operations change tensor shapes, requiring complex per-splat optimizer state management.

**Solution**: Keep splat pool size fixed and relocate weak splats instead of adding/removing.

### Tile-Based Peak Finding for Spatial Fairness (Default)

**Problem**: Global peak finding concentrates relocations in bright/high-error regions.

**Solution**: Divide image into tiles and find peaks per tile, using probabilistic or deterministic selection depending on k_per_tile.

### Cooldown Mechanism

After a splat is relocated, it cannot be relocated again for N dynamic ops steps (default: 1). This ensures diverse splat coverage instead of repeatedly relocating the same splats.

### Convergence Guard

Only relocates if strongest residual peak exceeds convergence threshold:
```python
if strongest_residual < max_abs_error_threshold:
    skip_all_relocations()  # Already converged enough
```

## Configuration

```python
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

config = DynamicOpsConfig()

# Scheduling
config.step_every = 50                    # Run every N iterations

# Peak finding
config.k_max_residuals = 40               # Max peaks to analyze
config.nms_radius_vox = 2.0               # Spatial exclusion radius

# Tile-based seeding (enabled by default)
config.enable_tiled_seeding = True        # Use spatial fairness
config.num_tiles_per_dim = None           # Auto: 16 for 2D, 6 for 3D, 4 for 4D, 2 for 5D+

# Weak splat identification
config.relocation_percentile = 1.0        # % of weakest splats eligible
config.max_relocations_per_step = 64      # Cap relocations per step (None = no limit)

# Relocation parameters
config.init_sigma_vox = 0.5               # Initial sigma for relocated splats
config.min_contribution_threshold = 0.01  # Min influence to consider peak "covered"
config.enable_coverage_check = False      # If True, skip peaks already covered

# Coverage check behavior:
# - False (DEFAULT): Relocate to all high-residual peaks regardless of coverage
#   Rationale: If residual is high, existing coverage is clearly insufficient
# - True: Conservative mode - skip peaks where non-weak splats have influence
#   May leave persistent high-error regions unaddressed

# Cooldown mechanism
config.relocation_cooldown_steps = 1      # Steps to wait before re-relocation

# Safety
config.min_splats_to_keep = 10            # Minimum to retain
```

## Usage

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

# Use default configuration
result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,  # Enabled by default
)

# Custom configuration
config = DynamicOpsConfig()
config.k_max_residuals = 30              # More peaks to analyze
config.relocation_percentile = 10.0      # More aggressive relocation

result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,
    dynamic_config=config,
    dynamic_ops_verbose=True,            # See what's happening
)
```

## Algorithm Details

### Splat Importance Calculation

```python
importance = amplitude * prod(diag(L))  # Approximates Gaussian "mass"
```

Low importance = small AND dim. However, not all low-importance splats
should be relocated.

**Smart Selection**: Only relocates splats that are:
- Low importance (amplitude x volume)
- At locations with LOW residual (not fighting errors)
- Off cooldown (not recently relocated)

**Key insight**: A splat at a high-error location is doing useful work
(fighting the error), even if it has low importance. We only want to relocate
splats that are truly useless - weak AND at low-residual locations.

### Peak-Splat Matching

1. Process peaks in order of residual magnitude (strongest first)
2. Optionally check if any non-weak splat already has significant influence there (when `enable_coverage_check=True`)
3. Direct pairing: weakest splat -> highest-error peak, 2nd weakest -> 2nd highest, etc.
4. Cap relocations at `max_relocations_per_step`

**Why direct pairing?** Distance-based matching is unnecessary since relocated
splats are completely reset anyway. The weakest splat should target the worst
error, aligning priorities correctly.

### Coverage Check Control

The `enable_coverage_check` flag controls whether to skip peaks already "covered"
by existing non-weak splats:

**When `False` (default)**:
- Relocates to ALL high-residual peaks regardless of existing coverage
- More aggressive - assumes if residual is high, coverage is insufficient
- Can help when existing splats have influence but error remains high

**When `True`**:
- Skips peaks where existing splats already have significant influence
- Conservative approach
- May leave high-error peaks unaddressed if they have *some* coverage

### Relocation

For each matched (splat_idx, peak_coords) pair:
1. **CENTER**: Convert coords to raw (logit) space
2. **AMPLITUDE**: Set to residual value at new location
3. **COVARIANCE**: Reset to isotropic (`init_sigma_vox`)
4. **OPTIMIZER STATE**: Reset Adam momentum/variance for relocated parameters

## Package Structure

```
dynamic_ops/
├── __init__.py         # Public API exports
├── config.py           # DynamicOpsConfig dataclass
├── peak_finding.py     # Global and tiled peak finding with NMS
├── operations.py       # RecentlyRelocatedTracker, apply_dynamic_operations, relocation logic
├── README.md           # This file
└── tests/
    ├── __init__.py
    ├── README.md
    ├── test_dynamic_ops.py         # Config, peak finding, importance, relocation tests
    └── test_relocation_tracker.py  # Cooldown mechanism tests
```

## Performance

- **50x+ faster**: Standard PyTorch Adam vs per-splat alternatives
- **Tile-based overhead**: Negligible (~1-2% of iteration time)
- **Overall**: Dynamic ops add ~5-10% to total optimization time
- **GPU-accelerated**: All tensor operations run on GPU when available

## References

- See `operations.py` for relocation logic
- See `peak_finding.py` for tiling algorithm implementation
- See `tests/test_dynamic_ops.py` for usage examples

## Version History

- **v2.0.0** (January 2025): Fixed-pool splat relocation
  - Replaced add/remove with relocation (50x+ faster)
  - Standard PyTorch Adam optimizer
  - Simplified architecture
- **v0.3** (2025-01): Probabilistic unified parameter scheme
- **v0.2** (2025-01): Tile-based seeding for spatial fairness
- **v0.1** (2024): Initial dynamic operations (add/remove, deprecated)
