# Seed Generation for Gaussian Splatting

This document describes the seed generation methods available in `luxar.gsplats.seeds` for seeding Gaussian splat fitting.

## Overview

Seed generation is the first step in Gaussian splat fitting - identifying potential locations where Gaussian splats should be placed. Good seed selection is crucial for:
- **Convergence speed**: Starting near optimal positions reduces iterations
- **Reconstruction quality**: Better coverage of important features
- **Computational efficiency**: Fewer redundant splats

The module provides four complementary approaches:

1. **Decomposition-Based** (`seed_from_decomposition`): Scale-hierarchical detection via image decomposition
2. **Grid-Based** (`seed_from_grid`): Uniform spatial coverage with isotropic shapes
3. **Edge-Based** (`seed_from_edges`): Boundary detection with isotropic shapes
4. **Peaks-Based** (`seed_from_peaks`): Intensity-weighted sampling at non-zero voxels (ideal for sparse residuals in progressive fitting)

## Unified Entry Point: `generate_seeds()`

**Purpose**: Simplified interface for seed generation with automatic method selection.

**Function Signature**:
```python
def generate_seeds(
    V: np.ndarray,
    method: str = "auto",
    **method_kwargs
) -> GSplatData
```

**Parameters**:
- `V` (np.ndarray): Input n-dimensional image/volume
- `method` (str): Seed generation method
  - `"auto"`: Fast edges + grid combination - **DEFAULT, RECOMMENDED**
  - `"decomposition"`: Hierarchical scale decomposition-based detection (slow)
  - `"grid"`: Uniform grid seeding for spatial coverage
  - `"edges"`: Edge-based seeding along detected boundaries
  - `"peaks"`: Intensity-weighted sampling at non-zero voxels
  - Comma-separated: e.g., `"decomposition,edges,grid"` for specific combination
- `device` (str, optional): PyTorch device for GPU acceleration
  - `None` (default): CPU using scipy.ndimage
  - `'auto'`: Auto-detect (cuda > mps > cpu)
  - `'cpu'`: Force CPU
  - `'cuda'`: NVIDIA GPU (if available)
  - `'mps'`: Apple Metal (if available)
  - `'cuda:0'`, `'cuda:1'`: Specific GPU device
- `**kwargs`: Method-specific parameters passed to underlying functions

**Returns**:
- `GSplatData`: Seed Gaussian splats with scale-informed shapes:
  - `centers`: (N, ndim) float - peak positions
  - `amplitudes`: (N,) float - peak intensities
  - `cholesky_factors`: (N, ndim*(ndim+1)//2) float - scale-informed Cholesky factors

**Examples**:
```python
from luxar.gsplats.seeds import generate_seeds

# Automatic method selection (recommended) - returns GSplatData
seeds = generate_seeds(image)
print(f"Generated {len(seeds.centers)} seed splats")

# Decomposition method with custom parameters
seeds = generate_seeds(image, method="decomposition",
                      scales=[1, 2, 4, 8], ignore_finest_k=1)

# Grid method with custom spacing
seeds = generate_seeds(image, method="grid", spacing=10.0)

# Edge-based seeding
seeds = generate_seeds(image, method="edges", edge_threshold_rel=0.15)

# GPU acceleration (much faster for large volumes; speedup depends on GPU)
seeds = generate_seeds(image, device='auto')  # Auto-detect GPU
seeds = generate_seeds(image, device='cuda')  # Explicit NVIDIA GPU
seeds = generate_seeds(image, device='mps')   # Apple Metal GPU
```

**Method Selection Guide**:
- **"auto"** (default): Fast edges + grid combination - recommended for most use cases
- **"decomposition"**: Best for blob-like features with principled scale separation (slow)
- **"grid"**: Fast uniform coverage, good for textures or as baseline
- **"edges"**: Best for images with clear boundaries

**Auto Mode Budget Allocation**:
The "auto" method combines edges + grid (decomposition excluded for speed):
- Edges: 60% (boundary detection with Sobel gradients)
- Grid: 40% (coverage for gaps)

Use `method="decomposition,edges,grid"` to include all methods explicitly.

**Integration with fit_gaussian_splats()**:
```python
from luxar.gsplats import fit_gaussian_splats

# Method 1: Automatic (most common)
result = fit_gaussian_splats(image)  # seeds auto-generated internally with "auto" method

# Method 2: Explicit seed generation - pass GSplatData directly
seeds = generate_seeds(image, method="decomposition")
result = fit_gaussian_splats(image, seeds=seeds)  # Uses full geometry

# Method 3: Custom seed_method parameter
result = fit_gaussian_splats(
    image,
    seed_method="edges",  # Override default
    seed_kwargs={"edge_threshold_rel": 0.15}
)
```

**Note**: All methods return `GSplatData` with scale-informed Gaussian shapes.

---

### 1. Decomposition-Based Seed Generation

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

**Advantages**:
- **Principled scale separation**: Energy explicitly distributed across scales
- **Noise suppression**: Ignoring finest scales removes high-frequency noise
- **Hierarchical structure**: Natural coarse-to-fine ordering
- **Sparse representation**: Fewer, higher-quality seeds
- **Energy-based quality**: Seeds sorted by importance

---

### 2. Grid-Based Seed Generation

**Function**: `seed_from_grid()`

**Strategy**: Place seeds on a uniform grid with optional jitter and intensity filtering.

**How It Works**:
1. Generate uniform grid coordinates based on spacing
2. Apply optional jitter for randomness
3. Filter by intensity threshold
4. Sample amplitudes from image
5. Create isotropic Cholesky factors (sigma * I)

**When to Use**:
- When you want uniform spatial coverage
- For textures and uniform regions
- As a baseline or fallback method
- When speed is important

**Key Parameters**:
```python
seeds = seed_from_grid(
    V,                                    # Input image/volume
    spacing=None,                         # Grid spacing (None = auto, aspect-ratio-aware)
    jitter=0.0,                           # Jitter fraction (0.0-0.5)
    sigma=None,                           # Gaussian sigma (None = spacing/2)
    exclude_below=None,                   # Absolute intensity threshold
    exclude_below_percentile=None,        # Percentile threshold (0-100)
)
```

**Advantages**:
- **Fast**: Simple grid generation
- **Predictable**: Uniform coverage guaranteed
- **Easy to tune**: Just adjust spacing
- **Good baseline**: Works for any image type

---

### 3. Edge-Based Seed Generation

**Function**: `seed_from_edges()`

**Strategy**: Detect edges using Sobel gradients and place seeds along boundaries using Poisson disk sampling with isotropic shapes (sigma=1.0).

**How It Works**:
1. Compute nD Sobel gradient magnitude
2. Threshold to get edge mask (relative to maximum gradient)
3. Poisson disk sampling weighted by edge response
4. Assign isotropic sigma=1.0 to all seeds
5. Build isotropic Cholesky factors

**When to Use**:
- For images with clear boundaries
- When you need dense seeds along edges
- For capturing boundary structure

**Key Parameters**:
```python
seeds = seed_from_edges(
    V,                                    # Input image/volume
    n_seeds=None,                         # Target number (None = auto)
    min_distance=2.0,                     # Min distance between seeds
    edge_threshold_rel=0.1,               # Relative edge threshold (0.0-1.0)
    device=None,                          # GPU device (None=CPU, 'auto', 'cuda', 'mps')
)
```

**Advantages**:
- **Edge-aware**: Seeds concentrated along boundaries
- **Gradient-informed**: Uses Sobel gradient magnitude for weighting
- **Boundary-focused**: Seeds placed along edges via Poisson disk sampling

---

### 4. Peaks-Based Seed Generation

**Function**: `seed_from_peaks()`

**Strategy**: Sample seeds at non-zero voxels weighted by intensity. Every seed lands on actual signal, with splats starting at single-voxel size so the optimizer grows them as needed.

**How It Works**:
1. Find all non-zero voxels in the input volume
2. Sample `n_seeds` locations with probability proportional to intensity
3. Sort by intensity (brightest first)
4. Assign isotropic Gaussians with auto-scaled sigma based on inter-seed spacing

**When to Use**:
- For sparse residuals in progressive fitting
- When signal has varying shape (peaks, plateaus, edges)
- When peak-detection would miss non-extremal structures
- As a complement to other methods for residual fitting

**Key Parameters**:
```python
seeds = seed_from_peaks(
    V,                                    # Input volume (any dimensionality)
    n_seeds=None,                         # Number of seeds (None = one per non-zero voxel)
    init_sigma=None,                      # Gaussian sigma (None = auto-scaled)
    device=None,                          # GPU device ('cuda', 'mps', 'auto', None)
)
```

**Advantages**:
- **Signal-focused**: Seeds placed only on actual signal, no wasted seeds on background
- **Intensity-weighted**: Brighter voxels more likely to receive seeds
- **Auto-scaling**: Sigma automatically computed from inter-seed spacing
- **GPU-accelerated**: Uses PyTorch for fast multinomial sampling

---

## Comparison: Methods Overview

| Aspect | Decomposition | Grid | Edges | Peaks |
|--------|---------------|------|-------|-------|
| **Philosophy** | Scale-hierarchical | Uniform coverage | Boundary detection | Intensity-weighted sampling |
| **Shape Type** | Isotropic | Isotropic | Isotropic | Isotropic |
| **Best For** | Blob-like features | Textures, coverage | Boundaries | Sparse residuals |
| **Noise Handling** | ignore_finest_k | Intensity threshold | Edge threshold | Zero-voxel exclusion |
| **Speed** | Slower (decomposition) | Fast | Medium | Fast |

## Integration with fit_gaussian_splats

All methods seamlessly integrate with the main fitting API:

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

## GPU Acceleration

All seeding methods support GPU acceleration using PyTorch for substantial speedups on large volumes (>100³) — often orders of magnitude depending on GPU and problem size.

### Quick Start

```python
from luxar.gsplats.seeds import generate_seeds

# Auto-detect best device (cuda > mps > cpu)
seeds = generate_seeds(volume, device='auto')

# Explicit CUDA GPU
seeds = generate_seeds(volume, device='cuda')

# Specific GPU device
seeds = generate_seeds(volume, device='cuda:1')
```

### GPU Operations

> Per-operation speedup ranges below are typical observed values on test volumes; actual results vary by GPU model, problem size, and dimensionality. Treat them as indicative, not guarantees.

GPU acceleration is applied to:
- **Sobel gradients**: 10-50x faster (supports arbitrary dimensions)
- **Peak detection**: 20-100x faster (2D/3D only)
- **Soft blur**: 5-20x faster (all dimensions via separable 1D convolution)
- **Interpolation**: 10-30x faster (2D/3D only)
- **Deduplication**: Always CPU (SpatialHashGrid based, `device` parameter ignored)

### Dimension Support

| Operation | 1D | 2D | 3D | 4D+ |
|-----------|----|----|----|----|
| Sobel gradients | ✅ GPU | ✅ GPU | ✅ GPU | ✅ GPU |
| Peak detection | ❌ CPU | ✅ GPU | ✅ GPU | ❌ CPU |
| Interpolation | ❌ CPU | ✅ GPU | ✅ GPU | ❌ CPU |
| Deduplication | ✅ CPU | ✅ CPU | ✅ CPU | ✅ CPU |

**Note**: For unsupported dimensions, the system automatically falls back to CPU with a warning.

### Performance Optimization

```python
# For large 3D volumes (512³+), GPU provides substantial speedup (GPU-dependent)
seeds = generate_seeds(large_volume, method='edges', device='cuda')

# Small volumes (<50³) auto-use CPU (GPU overhead not worth it)
seeds = generate_seeds(small_volume, device='auto')  # Uses CPU

# Integration with fitting pipeline
from luxar.gsplats import fit_gaussian_splats
result = fit_gaussian_splats(
    volume,
    device='cuda',  # Propagates to seeding automatically
    seed_method='edges',
)
```

### Backward Compatibility

GPU acceleration is fully backward compatible:
```python
# Old code works unchanged (CPU path)
seeds = generate_seeds(volume)

# New code opts-in to GPU
seeds = generate_seeds(volume, device='cuda')
```

## Advanced Usage

### Combining Methods

Use `generate_seeds(method="auto")` to combine edges + grid automatically:

```python
from luxar.gsplats.seeds import generate_seeds

# Fast combination of edges + grid (default)
seeds = generate_seeds(image, method="auto", min_distance=3.0)

# Or combine specific methods
seeds = generate_seeds(image, method="decomposition,edges", min_distance=3.0)

# Use combined seeds (returns GSplatData)
result = fit_gaussian_splats(image, seeds=seeds)
```

Alternatively, combine manually using `combine_seeds`:

```python
from luxar.gsplats.seeds import (
    seed_from_decomposition,
    seed_from_grid,
    seed_from_edges,
    combine_seeds,
)

# Generate seeds from individual methods (returns GSplatData)
seeds_decomp = seed_from_decomposition(image, scales=[1, 2, 4, 8])
seeds_grid = seed_from_grid(image, spacing=10.0)
seeds_edges = seed_from_edges(image, min_distance=3.0)

# Combine positions with deduplication
combined_centers = combine_seeds(
    seeds_decomp.centers,  # Decomposition first (global structure)
    seeds_edges.centers,   # Then edges (boundaries)
    seeds_grid.centers,    # Finally grid (coverage)
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
seeds = seed_from_grid(
    image,
    spacing=5.0,                # Dense grid
    exclude_below_percentile=50.0,  # Skip low-intensity
)
```

**For images with strong edges**:
```python
seeds = seed_from_edges(
    image,
    edge_threshold_rel=0.2,     # Higher edge threshold
    min_distance=3.0,           # Larger spacing between seeds
)
```

## Implementation Details

### Amplitude Scaling

All methods sample seed amplitudes from the input volume and scale them by
`SEED_AMPLITUDE_SCALE = 0.9` (defined in `utils.py`). Starting seeds slightly
below the target intensity avoids initial over-prediction when splats overlap,
which would otherwise trigger the asymmetric over-prediction penalty and risk
divergence. The optimizer raises amplitudes as needed during fitting.

### Peak Detection Algorithm

The decomposition method uses `local_maxima()` for peak detection:
- L∞ (Chebyshev) neighborhood with configurable radius
- Threshold-based filtering
- Top-k limiting for computational efficiency
- Works in arbitrary dimensions (1D, 2D, 3D, 4D+)

### Deduplication Strategy

Uses `dedupe_farthest_first()` for spatial deduplication. The implementation uses a simple greedy approach, which is ~50x faster and produces equivalent results for Gaussian splatting:
1. Sort candidates by energy/intensity (highest first)
2. Keep first (highest energy) candidate
3. Iterate through remaining candidates in intensity order
4. Keep each candidate if it satisfies min_distance from all already-selected seeds
5. Uses SpatialHashGrid for O(1) amortized nearest-neighbor queries

### Edge Seeding (Sobel + Poisson Disk)

The edges method uses Sobel gradients for edge detection:
- Computes nD Sobel gradient magnitude for edge strength
- Thresholds at `edge_threshold_rel * max_gradient`
- Poisson disk sampling weighted by gradient magnitude
- Assigns isotropic sigma=1.0 to all seeds (orientation learned during fitting)

## Testing

Comprehensive test suite available:

```bash
# Run all seeds tests
hatch run pytest packages/luxar/src/luxar/gsplats/seeds/tests/ -v

# Run specific test files
hatch run pytest packages/luxar/src/luxar/gsplats/seeds/tests/test_grid.py -v
hatch run pytest packages/luxar/src/luxar/gsplats/seeds/tests/test_edges.py -v
hatch run pytest packages/luxar/src/luxar/gsplats/seeds/tests/test_generate_seeds.py -v
```

## Module Layout

| File | Contents |
|------|----------|
| `generate.py` | `generate_seeds()` unified entry point and `_auto_combine()` (edges + grid) |
| `multiscale_decomposition.py` | `seed_from_decomposition()` |
| `grid.py` | `seed_from_grid()` |
| `edges.py` | `seed_from_edges()`, plus shared `_compute_nd_sobel_magnitude()` and `_sample_amplitudes()` helpers |
| `peaks.py` | `seed_from_peaks()` (PyTorch, GPU-accelerated multinomial sampling) |
| `utils.py` | `local_maxima()`, `dedupe_farthest_first()`, `combine_seeds()`, `sigmas_to_cholesky_isotropic()`, `SEED_AMPLITUDE_SCALE`; re-exports `SpatialHashGrid` from `luxar.utils.spatial_hash` |
| `gpu_ops.py` | PyTorch GPU kernels (`_get_device()`, `should_use_gpu()`, Sobel/blur/max-pool/grid-sample) shared by the CPU/GPU dispatch paths |

## References

- **Main API**: See `fit_gsplats.py` for integration with fitting pipeline
- **Decomposition**: See `multiscale/decompose.py` for multi-scale decomposition details
- **Performance**: See `SEEDING_PERFORMANCE_ISSUE.md` for the superlinear scaling of edge detection at high seed counts
- **Tests**: See `tests/` for usage examples and validation

## Version History

- **v2.3 (2025-01)**: Peaks-based seeding
  - Added `seed_from_peaks()` for intensity-weighted sampling at non-zero voxels
  - Ideal for sparse residuals in progressive fitting
  - Auto-scaling sigma based on inter-seed spacing
  - GPU-accelerated via PyTorch multinomial sampling

- **v2.2 (2025-01)**: GPU acceleration
  - Added PyTorch GPU acceleration for all seeding methods
  - Substantial speedup for large volumes (>100³), often orders of magnitude depending on GPU
  - Automatic fallback for unsupported dimensions
  - Backward compatible (device=None defaults to CPU)
  - Pure PyTorch implementation (no external dependencies)

- **v2.1 (2025-01)**: Speed optimization
  - Changed "auto" default from all methods to edges + grid only
  - Decomposition excluded by default for speed (use explicitly if needed)
  - Budget allocation: 60% edges, 40% grid

- **v2.0 (2025-01)**: API update
  - Removed `seed_from_gaussian()` and `seed_from_moments()` methods
  - Added `seed_from_grid()` for uniform coverage
  - Added `seed_from_edges()` for isotropic edge seeding
  - Changed default method from "decomposition" to "auto"

- **v0.1 (2025-01)**: Initial implementation
  - Added `seed_from_decomposition()` function
  - Comprehensive test suite
