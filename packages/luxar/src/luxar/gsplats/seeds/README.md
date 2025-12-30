# Seed Generation for Gaussian Splatting

This document describes the seed generation methods available in `luxar.gsplats.seeds` for seeding Gaussian splat fitting.

## Overview

Seed generation is the first step in Gaussian splat fitting - identifying potential locations where Gaussian splats should be placed. Good seed selection is crucial for:
- **Convergence speed**: Starting near optimal positions reduces iterations
- **Reconstruction quality**: Better coverage of important features
- **Computational efficiency**: Fewer redundant splats

The module provides two complementary approaches:

1. **Multiscale Gaussian Detection** (`seed_from_gaussian`): Multi-method detection with spatial redundancy
2. **Decomposition-Based** (`seed_from_decomposition`): Scale-hierarchical detection via image decomposition

## Methods

## Unified Entry Point: `generate_seeds()`

**Purpose**: Simplified interface for seed generation with automatic method selection.

**Function Signature**:
```python
def generate_seeds(
    V: np.ndarray,
    method: str = "both",
    **method_kwargs
) -> np.ndarray
```

**Parameters**:
- `V` (np.ndarray): Input n-dimensional image/volume
- `method` (str): Seed generation method
  - `"decomposition"`: Hierarchical scale decomposition-based detection - **DEFAULT**
  - `"gaussian"`: Fast multiscale Gaussian peak detection
  - `"both"`: Combines decomposition + gaussian methods
  - `"moments"`: Full covariance estimation for anisotropic features
- `**kwargs`: Method-specific parameters passed to underlying functions

**Returns**:
- `GSplatData`: Seed Gaussian splats with scale-informed shapes:
  - `centers`: (N, ndim) float - peak positions
  - `amplitudes`: (N,) float - peak intensities
  - `cholesky_factors`: (N, ndim*(ndim+1)//2) float - scale-informed Cholesky factors
  - `sharpnesses`: (N,) float - all set to 2.0 (standard Gaussian)

**Examples**:
```python
from luxar.gsplats.seeds import generate_seeds

# Automatic method selection (recommended) - returns GSplatData
seeds = generate_seeds(image)
print(f"Generated {len(seeds.centers)} seed splats")

# Gaussian method with custom parameters
seeds = generate_seeds(image, method="gaussian", scales=(1.0, 2.0, 4.0))

# Decomposition method with custom parameters
seeds = generate_seeds(image, method="decomposition",
                      scales=[1, 2, 4, 8], ignore_finest_k=1)
```

**Method Selection Guide**:
- **"decomposition"** (default): Most principled scale separation, recommended for most use cases
- **"gaussian"**: Fast generation when speed is critical
- **"moments"**: Full covariance estimation for anisotropic features
- **"both"**: Combines decomposition and gaussian methods for comprehensive coverage

**Integration with fit_gaussian_splats()**:
```python
from luxar.gsplats import fit_gaussian_splats

# Method 1: Automatic (most common)
result = fit_gaussian_splats(image)  # seeds auto-generated internally

# Method 2: Explicit seed generation - pass GSplatData directly
seeds = generate_seeds(image, method="gaussian")
result = fit_gaussian_splats(image, seeds=seeds)  # Uses full geometry

# Method 3: Custom seed_method parameter
result = fit_gaussian_splats(
    image,
    seed_method="decomposition",  # Override default
    seed_kwargs={"scales": [1, 2, 4, 8], "ignore_finest_k": 2}
)
```


**Seed Method Defaults**

The default method is "decomposition" for principled scale separation:

- **Default**: `method="decomposition"` - Most accurate scale information
- **Alternative**: `method="gaussian"` - Faster but less principled scale mapping
- **Full coverage**: `method="both"` - Combines both methods
- **Anisotropic**: `method="moments"` - Full covariance for elliptical features

**Note**: All methods return `GSplatData` with scale-informed Gaussian shapes.

**Configuration with seed_kwargs**:

All parameters from `generate_seeds()` can be passed through the API via `**seed_kwargs`:

```python
# Default (hybrid: decomposition + Gaussian for best convergence)
result = fit_gaussian_splats(image)  # seed_method="both" is the default

# Faster with decomposition only
result = fit_gaussian_splats(
    image,
    seed_method="decomposition",  # Most principled scale separation
    seed_kwargs={}                # Additional parameters passed to generate_seeds()
)

# Custom multiscale Gaussian parameters
result = fit_gaussian_splats(
    image,
    seed_method="gaussian",
    seed_kwargs={
        "scales": (1.0, 2.0, 4.0),
        "peaks_per_scale": 2000
    }
)

# Custom decomposition parameters
result = fit_gaussian_splats(
    image,
    seed_method="decomposition",
    seed_kwargs={
        "scales": [1, 2, 4, 8],
        "ignore_finest_k": 2
    }
)
```

See docstring for `fit_gaussian_splats()` for complete `seed_kwargs` options.

---

---


### 1. Multiscale Gaussian Seed Generation

**Function**: `seed_from_gaussian()`

**Strategy**: Combine multiple detection methods to create a rich, overcomplete set of seeds.

**Detection Methods**:
1. **CLAHE Preprocessing** (optional): Enhances local contrast for balanced detection in heterogeneous data
2. **Multiscale Gaussian-Blurred Peaks**: Detects blob-like structures at various scales
4. **Intensity-Weighted Grid Sampling**: Ensures spatial coverage in high-intensity regions

**When to Use**:
- Standard use case for most images
- When you want comprehensive feature detection
- When computational cost of many seeds is acceptable
- For complex scenes with mixed feature types

**Key Parameters**:
```python
seeds = seed_from_gaussian(
    V,                                    # Input image/volume
    scales=(0.7, 1.0, 1.4, 2.0, 2.8, 4.0),  # Detection scales (in voxels)
    peaks_per_scale=1000,                 # Max peaks per scale
    percentile_thresh=70.0,               # Intensity threshold (0-100)
    min_distance=2.0,                         # Min distance between candidates
    apply_clahe=True,                     # Enable CLAHE preprocessing
    add_intensity_grid=True,              # Add grid sampling
)
```

**Example**:
```python
from luxar.gsplats.seeds import seed_from_gaussian
from skimage import data
import numpy as np

# Load image
image = data.cell().astype(np.float32)

# Generate seeds with CLAHE
seeds = seed_from_gaussian(
    image,
    scales=(0.7, 1.0, 1.4, 2.0, 2.8, 4.0),
    apply_clahe=True,
    clahe_tile_size=32,
    clahe_clip_limit=16.0,
    min_distance=2.0,
)

print(f"Generated {len(seeds.centers)} seeds with scale info")
# seeds.centers shape: (N, 2) for 2D image
# seeds.cholesky_factors contains scale-informed sigmas
```

**Advantages**:
- Comprehensive detection across multiple methods
- CLAHE preprocessing for heterogeneous data
- Proven track record in production use
- Handles complex scenes well

**Limitations**:
- Can produce many redundant candidates
- No explicit scale hierarchy
- Computationally intensive with many scales

---

### 2. Decomposition-Based Seed Generation

**Function**: `seed_from_decomposition()`

**Strategy**: Decompose image into scale hierarchy, then find local maxima in each scale (excluding finest k scales for noise suppression).

**How It Works**:
1. **Multi-scale decomposition**: `V = Σₖ upsample(Vₖ)` where energy is distributed hierarchically
2. **Peak detection per scale**: Find local maxima in each scale image
3. **Scale filtering**: Ignore finest k scales (default k=1) to suppress noise
4. **Coordinate mapping**: Map scale coordinates to full resolution
5. **Spatial deduplication**: Remove close candidates using farthest-first selection
6. **Energy sorting**: Return candidates sorted by energy (descending)

**When to Use**:
- When you want principled scale separation
- For noisy images (ignore_finest_k suppresses noise)
- When you need hierarchical structure (coarse-to-fine)
- For sparse representations with fewer candidates

**Key Parameters**:
```python
seeds = seed_from_decomposition(
    V,                                    # Input image/volume
    scales=[1, 2, 4, 8, 16, 32],         # Scale factors for decomposition
    ignore_finest_k=1,                    # Skip finest k scales (noise suppression)
    peaks_per_scale=None,                 # Max peaks per scale (None = unlimited)
    min_distance=2.0,                     # Min distance between seeds
    threshold_rel=0.1,                    # Relative threshold (0.0-1.0)
    decompose_kwargs=None,                # Additional args for decompose_image()
)
```

**Example**:
```python
from luxar.gsplats.seeds import seed_from_decomposition
from skimage import data
import numpy as np

# Load image
image = data.cell().astype(np.float32)

# Generate seeds using decomposition
seeds = seed_from_decomposition(
    image,
    scales=[1, 2, 4, 8, 16],
    ignore_finest_k=1,              # Skip full-resolution scale (noise)
    min_distance=3.0,
    threshold_rel=0.15,             # Peaks must be 15% of scale max
    decompose_kwargs={
        'n_iters': 500,             # Decomposition iterations
        'energy_weight': 0.01,       # Hierarchical energy penalty
        'loss_type': 'l1',          # Robust loss function
    },
    verbose=True,
)

print(f"Generated {len(seeds.centers)} seeds with scale info")
# seeds.amplitudes sorted by energy (descending)
```

**Advantages**:
- **Principled scale separation**: Energy explicitly distributed across scales
- **Noise suppression**: Ignoring finest scales removes high-frequency noise
- **Hierarchical structure**: Natural coarse-to-fine ordering
- **Sparse representation**: Fewer, higher-quality seeds
- **Energy-based quality**: Seeds sorted by importance

**Limitations**:
- Requires decomposition step (adds computation time)
- May miss very fine details if ignore_finest_k is too large
- Sensitive to decomposition convergence

**Parameter Guide**:
- `ignore_finest_k=1`: Standard setting, good noise suppression
- `ignore_finest_k=2`: Aggressive noise suppression for very noisy data
- `ignore_finest_k=0`: Include all scales (no noise filtering)
- `threshold_rel=0.1`: Standard setting (10% of scale maximum)
- `threshold_rel=0.05`: More permissive (find more seeds)
- `threshold_rel=0.2`: More selective (fewer, stronger seeds)

---

## Comparison: Multiscale Gaussian vs Decomposition

| Aspect | Multiscale Gaussian | Decomposition |
|--------|-------------|---------------|
| **Philosophy** | Multiscale Gaussian peak detection | Hierarchical scale-based selection |
| **Candidate Count** | Many (1000-10000+) | Fewer (100-1000) |
| **Scale Treatment** | Gaussian peaks at multiple scales | Optimized scale decomposition |
| **Noise Handling** | CLAHE preprocessing | Scale filtering (ignore_finest_k) |
| **Computation Time** | Fast (seconds) | Slower (decomposition overhead) |
| **Quality** | High coverage, some redundancy | Sparse, principled selection |
| **Best For** | General use, complex scenes | Noisy data, hierarchical structure |

## Integration with fit_gaussian_splats

Both methods seamlessly integrate with the main fitting API:

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.seeds import seed_from_decomposition

# Step 1: Generate seeds (returns GSplatData with scale-informed shapes)
seeds = seed_from_decomposition(
    image,
    scales=[1, 2, 4, 8],
    ignore_finest_k=1,
)

# Step 2: Fit Gaussian splats using full seed geometry
result = fit_gaussian_splats(
    image,
    seeds=seeds,  # Pass GSplatData - uses scale-informed initialization
    n_iters=1000,
    enable_dynamic_ops=True,  # Can still add/prune splats during optimization
)

print(f"Started with {len(seeds.centers)} seeds")
print(f"Final splats: {len(result.amplitudes)}")
```

**Note**: Even with explicit seeds, `enable_dynamic_ops=True` allows the optimizer to add/remove splats during fitting based on residual analysis.

## Advanced Usage

### Combining Methods

Use `generate_seeds(method="both")` to combine methods automatically:

```python
from luxar.gsplats.seeds import generate_seeds

# Automatic combination with deduplication
seeds = generate_seeds(image, method="both", min_distance=3.0)

# Use combined seeds (returns GSplatData)
result = fit_gaussian_splats(image, seeds=seeds)
```

Alternatively, combine manually using `combine_seeds`:

```python
from luxar.gsplats.seeds import seed_from_gaussian, seed_from_decomposition, combine_seeds

# Generate seeds from both methods (returns GSplatData)
seeds_gaussian = seed_from_gaussian(image, min_distance=3.0)
seeds_decomp = seed_from_decomposition(image, scales=[1, 2, 4, 8])

# Combine positions with deduplication
combined_centers = combine_seeds(
    seeds_decomp.centers,  # Decomposition first (global structure)
    seeds_gaussian.centers,  # Then gaussian (local features)
    min_distance=3.0,
)

# Use with fitter (positions only - fitter will initialize shapes)
result = fit_gaussian_splats(image, seeds=combined_centers)
```

### Tuning for Different Data Types

**For noisy microscopy images**:
```python
seeds = seed_from_decomposition(
    image,
    scales=[1, 2, 4, 8],
    ignore_finest_k=2,          # Aggressive noise filtering
    threshold_rel=0.15,         # Higher threshold
    decompose_kwargs={'loss_type': 'l1'},  # Robust to outliers
)
```

**For clean synthetic data**:
```python
seeds = seed_from_gaussian(
    image,
    scales=(0.5, 0.7, 1.0, 1.4, 2.0),  # Include fine scales
    apply_clahe=False,          # No preprocessing needed
    percentile_thresh=60.0,     # Lower threshold (find more)
)
```

**For large sparse features**:
```python
seeds = seed_from_decomposition(
    image,
    scales=[2, 4, 8, 16, 32],   # Skip finest scales entirely
    ignore_finest_k=0,          # Process all provided scales
    threshold_rel=0.2,          # Higher threshold (stronger peaks only)
)
```

## Implementation Details

### Peak Detection Algorithm

Both methods use `_local_maxima()` for peak detection:
- L∞ (Chebyshev) neighborhood with configurable radius
- Threshold-based filtering
- Top-k limiting for computational efficiency
- Works in arbitrary dimensions (1D, 2D, 3D, 4D+)

### Deduplication Strategy

Uses `_dedupe_farthest_first()` for spatial deduplication:
1. Sort candidates by energy/intensity (highest first)
2. Keep first (highest energy) candidate
3. Iteratively select candidate farthest from all selected
4. Only keep candidates satisfying min_distance constraint
5. Ensures maximum spatial diversity with quality priority

### Coordinate Precision

- Multiscale Gaussian method: Sub-voxel precision via intensity-weighted centroid refinement
- Decomposition method: Sub-voxel precision via scale interpolation
- Both return float coordinates for downstream optimization

## Testing

Comprehensive test suite available in `tests/test_candidates.py`:

```bash
# Run all candidate tests
hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_candidates.py::TestDecompositionCandidates -v

# Run specific test
hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_candidates.py::TestDecompositionCandidates::test_basic_functionality_2d -v
```

Tests cover:
- Basic functionality (1D, 2D, 3D)
- Parameter validation
- Edge cases (empty results, uniform images)
- Parameter effects (ignore_finest_k, threshold_rel, min_distance)
- Integration with decompose_image

## Performance Considerations

### Computational Cost

**Multiscale Gaussian Method**:
- **Gaussian filtering**: O(n_scales × n_voxels × kernel_size)
- **Peak detection**: O(n_scales × n_voxels)
- **Total**: ~1-10 seconds for typical images

**Decomposition Method**:
- **Decomposition**: O(n_iters × n_scales × n_voxels) - dominant cost
- **Peak detection**: O(n_scales × n_voxels) - negligible
- **Total**: ~10-60 seconds for typical images

### Memory Usage

Both methods have modest memory requirements:
- Multiscale Gaussian: ~2-3× input size (filtered images)
- Decomposition: ~4-6× input size (scale images at different resolutions)
- Final candidates: ~8 bytes per coordinate × ndim × N_candidates

### Optimization Tips

1. **Reduce decomposition iterations**: `decompose_kwargs={'n_iters': 200}` (faster, slightly lower quality)
2. **Limit peaks per scale**: `peaks_per_scale=500` (faster deduplication)
3. **Use fewer scales**: `scales=[1, 4, 16]` (faster, coarser hierarchy)
4. **Increase min_distance**: `min_distance=5.0` (fewer candidates, faster downstream)

## References

- **Specification**: See `SPECIFICATIONS.md` for comprehensive technical details
- **Main API**: See `fit_gsplats.py` for integration with fitting pipeline
- **Decomposition**: See `multiscale/decompose.py` for multi-scale decomposition details
- **Tests**: See `tests/test_candidates.py` for usage examples and validation

## Version History

- **v0.1 (2025-01-16)**: Initial implementation of decomposition-based candidate generation
  - Added `seed_from_decomposition()` function
  - Comprehensive test suite
  - Integration with existing overcomplete method
