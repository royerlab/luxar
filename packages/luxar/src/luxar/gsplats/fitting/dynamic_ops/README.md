# Dynamic Operations for Adaptive Gaussian Splatting

Fixed-pool splat relocation that adaptively redistributes Gaussian splats during optimization.

## Overview

Dynamic operations use **fixed-pool splat relocation** instead of add/remove operations. This enables use of standard PyTorch Adam optimizer (50x+ faster) by keeping tensor shapes constant throughout optimization.

### How It Works

1. **Identify weak splats**: Find splats with low importance (amplitude × volume)
2. **Find residual peaks**: Detect high-error regions using non-maximum suppression
3. **Relocate**: Move weak splats to high-residual peaks (parameter updates only)
4. **Standard Adam adapts**: Optimizer momentum quickly adapts to new locations

## Key Features

### Fixed-Pool Architecture (50x+ Faster)

**Problem**: Traditional add/remove operations change tensor shapes, requiring complex per-splat optimizer state management.

**Solution**: Keep splat pool size fixed and relocate weak splats instead of adding/removing.

**Benefits**:
- ✅ Standard PyTorch Adam works naturally (50x+ faster)
- ✅ No complex state management
- ✅ Simpler architecture
- ✅ Adam's momentum adapts within 1-3 iterations

### Tile-Based Peak Finding for Spatial Fairness (Default)

**Problem**: Global peak finding concentrates relocations in bright/high-error regions.

**Solution**: Divide image into tiles and find peaks per tile.

**Benefits**:
- ✅ All regions get fair attention regardless of brightness
- ✅ Dim regions can't be out-competed by bright regions
- ✅ Guaranteed spatial coverage across entire image

### Convergence Guard

Only relocates if strongest residual peak exceeds convergence threshold:
```python
if strongest_residual < max_abs_error_threshold:
    skip_all_relocations()  # Already converged enough
```

## Configuration

```python
from luxar.gsplats.fitting.config import DynamicOpsConfig

config = DynamicOpsConfig()

# Scheduling
config.step_every = 50                    # Run every N iterations

# Peak finding
config.k_max_residuals = 20               # Max peaks to analyze
config.nms_radius_vox = 2.0               # Spatial exclusion radius

# Tile-based seeding (enabled by default)
config.enable_tiled_seeding = True        # Use spatial fairness
config.num_tiles_per_dim = None           # Auto: 16 for 2D, 6 for 3D, 4 for 4D

# Relocation parameters
config.relocation_percentile = 1.0        # % of weakest splats eligible
config.max_relocations_per_step = 32      # Cap relocations per step (None = no limit)
config.init_sigma_vox = 0.5               # Initial sigma for relocated splats
config.enable_coverage_check = False      # Skip peaks already covered by non-weak splats

# Coverage check behavior:
# - False (DEFAULT): Relocate to all high-residual peaks regardless of coverage
#   Rationale: If residual is high, existing coverage is clearly insufficient
# - True: Conservative mode - skip peaks where non-weak splats have influence
#   May leave persistent high-error regions unaddressed

# Safety
config.min_splats_to_keep = 10            # Minimum to retain
```

## Usage

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.config import DynamicOpsConfig

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
importance = amplitude × volume
volume = prod(diag(L))  # Approximates Gaussian volume
```

Low importance = small AND dim. However, not all low-importance splats
should be relocated.

**Smart Selection**: Only relocates splats that are:
- Low importance (amplitude × volume)
- At locations with LOW residual (not fighting errors)

**Key insight**: A splat at a high-error location is doing useful work
(fighting the error), even if it has low importance. We only want to relocate
splats that are truly useless - weak AND at low-residual locations.

### Peak-Splat Matching

1. Process peaks in order of residual magnitude (strongest first)
2. For each peak:
   - Check if any non-weak splat already has significant influence there
   - If yes, skip this peak (existing coverage)
   - If no, add to uncovered peaks list
3. Direct pairing: weakest splat → highest-error peak, 2nd weakest → 2nd highest, etc.
4. Cap relocations at `max_relocations_per_step`

**Why direct pairing?** Distance-based matching is unnecessary since relocated
splats are completely reset anyway. The weakest splat should target the worst
error, aligning priorities correctly.

### Coverage Check Control (Experimental)

The `enable_coverage_check` flag controls whether to skip peaks already "covered"
by existing non-weak splats:

**When `True` (default)**:
- Skips peaks where existing splats already have significant influence
- Original behavior - conservative approach
- May leave high-error peaks unaddressed if they have *some* coverage

**When `False`**:
- Relocates to ALL high-residual peaks regardless of existing coverage
- More aggressive - assumes if residual is high, coverage is insufficient
- Can help when existing splats have influence but error remains high

**Trade-off**: The issue is that "has influence" ≠ "error is resolved".
A peak with residual=0.8 and existing influence=0.3 clearly needs more coverage,
but the default behavior would skip it. Disabling coverage check addresses this.

**Recommendation**: Try both settings and compare convergence behavior for your data.

### Relocation

For each matched (splat_idx, peak_coords) pair:
1. **CENTER**: Convert coords to raw (logit) space
2. **AMPLITUDE**: Set to residual value at new location
3. **COVARIANCE**: Reset to isotropic (`init_sigma_vox`)
4. **SHARPNESS**: Keep unchanged (optimizer will adjust)

## Package Structure

```
dynamic_ops/
├── config.py           # DynamicOpsConfig (in parent fitting/config.py)
├── peak_finding.py     # Global and tiled peak finding
├── operations.py       # Main apply_dynamic_operations
├── SPECIFICATIONS.md   # Technical specification
└── tests/              # Comprehensive test suite
```

## Performance

- **50x+ faster**: Standard PyTorch Adam vs per-splat alternatives
- **Tile-based overhead**: Negligible (~1-2% of iteration time)
- **Overall**: Dynamic ops add ~5-10% to total optimization time

## References

- See `SPECIFICATIONS.md` for complete algorithm details
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
