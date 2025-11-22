# Dynamic Operations for Adaptive Gaussian Splatting

Convergence-driven dynamic operations that adaptively add and remove Gaussian splats during optimization.

## Overview

Dynamic operations address reconstruction deficiencies by analyzing the residual image (target - prediction) and performing three core operations:

1. **Residual Peak Analysis**: Find locations with highest reconstruction error
2. **Adaptive Seeding**: Add new splats where coverage is insufficient
3. **Principled Pruning**: Remove splats that contribute minimally

## Key Features

### Tile-Based Seeding for Spatial Fairness (Default)

**Problem**: Global peak finding concentrates seeds in bright/high-error regions, ignoring dimmer areas.

**Solution**: Divide image into tiles and process each independently.

**Benefits**:
- ✅ All regions get fair attention regardless of brightness
- ✅ Dim regions can't be out-competed by bright regions
- ✅ Guaranteed spatial coverage across entire image

### Probabilistic Unified Parameter Scheme

**Single parameter controls everything:**
```python
k_max_residuals = 10  # Expected number of seeds per iteration
```

**Auto-calculation** based on number of tiles:
```python
k_per_tile = k_max_residuals / (num_tiles_per_dim ** ndim)

if k_per_tile >= 1.0:
    # Deterministic: Keep floor(k_per_tile) peaks per tile
    # Example: k=512, 256 tiles → 2 per tile → 512 total
else:
    # Probabilistic: Keep each tile's peak with probability k_per_tile
    # Example: k=10, 256 tiles → p=0.039 → ~10 total
```

**Benefits**:
- Unified parameter across global and tiled modes
- Expected count ≈ k_max_residuals regardless of tiling
- Randomness prevents bias toward bright regions

### Single-Voxel Splat Initialization

New splats initialized with minimal neighborhood impact:
- **Amplitude**: Residual value at peak location
- **Sigma**: 0.5 voxels (single-pixel scale)
- **Shape**: Isotropic (identity covariance matrix)

**Benefit**: 13% influence on neighbors (vs 60% with σ=1.5), reducing error spikes when seeding.

## Configuration

```python
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

config = DynamicOpsConfig()

# Scheduling
config.step_every = 50                    # Run every N iterations

# Peak finding (controls both global and tiled modes)
config.k_max_residuals = 10               # Expected seeds per iteration
config.nms_radius_vox = 2.0               # Spatial exclusion radius

# Tile-based seeding (enabled by default)
config.enable_tiled_seeding = True        # Use spatial fairness
config.num_tiles_per_dim = None           # Auto: 16 for 2D, 6 for 3D, 4 for 4D, 2 for 5D+

# Seeding parameters
config.init_sigma_vox = 0.5               # Single-voxel scale splats

# Pruning parameters
config.pruning_percentile = 5.0           # % of least important to consider
config.min_splats_to_keep = 10            # Minimum to retain
```

## How It Works

### Tiling Strategy

**Auto-scaling by dimension** to maintain ~200-300 total tiles:
- **2D**: 16×16 = 256 tiles
- **3D**: 6×6×6 = 216 tiles
- **4D**: 4×4×4×4 = 256 tiles
- **5D+**: 2^d tiles

### Selection Modes

**Deterministic** (k_per_tile >= 1):
```python
k_max = 512, num_tiles = 256
k_per_tile = 512 / 256 = 2.0
→ Keep top 2 peaks per tile deterministically
→ Total: exactly 512 peaks
```

**Probabilistic** (k_per_tile < 1):
```python
k_max = 10, num_tiles = 256
k_per_tile = 10 / 256 = 0.039
→ Keep each tile's best peak with 3.9% probability
→ Expected: ~10 peaks (variance: 5-18)
```

### Seeding Process

For each peak above convergence threshold:
1. Create isotropic Gaussian at peak location
2. Amplitude = residual value at peak
3. Sigma = 0.5 voxels (single-voxel scale)
4. Add to model with current learning rate

### Convergence Guard

Only seeds if strongest peak's residual exceeds convergence threshold:
```python
if strongest_residual < max_abs_error_threshold:
    skip_all_seeding()  # Already converged enough
```

## Usage

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig

# Use default (tiled seeding with probabilistic fairness)
result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,  # Enabled by default
)

# Custom configuration
config = DynamicOpsConfig()
config.k_max_residuals = 20              # More seeds per iteration
config.num_tiles_per_dim = 8             # Custom tiling

result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,
    dynamic_config=config,
    dynamic_ops_verbose=True,            # See what's happening
)

# Disable tiled seeding (use global mode)
config = DynamicOpsConfig()
config.enable_tiled_seeding = False      # Global peak finding

result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,
    dynamic_config=config,
)
```

## Package Structure

```
dynamic_ops/
├── config.py           # DynamicOpsConfig
├── peak_finding.py     # Global and tiled peak finding
├── operations.py       # Main apply_dynamic_operations
└── tests/             # Comprehensive test suite
```

## Performance

- **Tile-based overhead**: Negligible (~1-2% of iteration time)
- **Probabilistic selection**: O(num_tiles) - very fast
- **Deterministic selection**: O(num_tiles × k_per_tile)
- **Overall**: Dynamic ops add ~5-10% to total optimization time

## References

- See `config.py` for all configuration parameters
- See `peak_finding.py` for tiling algorithm implementation
- See `operations.py` for seeding and pruning logic
- See `tests/test_dynamic_ops.py` for usage examples

## Version History

- v0.3 (2025-01): Probabilistic unified parameter scheme
- v0.2 (2025-01): Tile-based seeding for spatial fairness
- v0.1 (2024): Initial dynamic operations implementation
