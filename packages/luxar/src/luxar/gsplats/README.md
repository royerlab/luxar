# Gaussian Splatting Package for Luxar

High-performance n-dimensional oriented Gaussian splatting with automatic optimization for efficient image and volume reconstruction.

## Overview

This package implements a sophisticated Gaussian splatting system that fits collections of oriented Gaussian functions (splats) to reconstruct n-dimensional data. Each splat is a multivariate Gaussian characterized by position, full covariance matrix, and amplitude. The implementation features automatic convergence detection, adaptive learning, and GPU acceleration.

## Installation

Gaussian splatting requires optional dependencies:

```bash
pip install "luxar[gsplats]"
```

## Key Features

- **N-dimensional Support**: Works seamlessly with 2D images, 3D volumes, and 4D+ hypercubes (validated to 4D) with automatic gradient dilution compensation
- **Efficient Cholesky Parameterization**: Covariance matrices via Cholesky decomposition with batched triangular solve
- **Device Optimized**: CUDA acceleration with automatic device selection (CPU preferred on Apple Silicon)
- **Oriented Gaussians**: Full covariance matrices via Cholesky decomposition for arbitrary orientations
- **Convergence-Driven Dynamic Operations**: Adaptive splat management based on convergence criteria with seeding and pruning
- **Asymmetric Loss Functions**: Configurable penalty for over-prediction addresses additive model constraints (MSE and Poisson)
- **Standard PyTorch Adam**: Fast vectorized optimization with gradient dilution compensation (significantly faster than per-splat alternatives)
- **Adaptive Thresholds**: Amplitude validation scales with local residual magnitude to prevent optimization plateaus
- **Automatic Optimization**: Early stopping saves a substantial fraction of iterations without quality loss
- **GPU Acceleration**: CUDA support with batched operations, improved MPS compatibility
- **Robust Initialization**: Multiscale candidate detection with DoG and peak finding
- **Memory Efficient**: Truncated rendering, optional mixed precision, pre-allocated buffers
- **Tiled Fitting**: Overlapping tiles with Hann cosine apodization for volumes that exceed GPU memory
- **Quality Metrics**: Built-in PSNR, SSIM, and MSE computation on GPU tensors

## Quick Example

```python
from luxar.gsplats import fit_gaussian_splats

# Fit splats to your volume
result = fit_gaussian_splats(volume, n_iters=1000)

# Post-processing transformations
result = result.center_at_centroid()     # Center for easier viewing
result = result.scale_intensity(0.1)     # Reduce brightness 10x
result = result.translate([10, 20, 30])  # Shift in space

# Save or visualize
result.save("output.gsplats.zarr")
```

## How It Works

### 1. Seed Generation
The system starts by finding initial seed positions using multiscale seed generation:
- **Gaussian filtering** at multiple scales to detect blob-like structures
- **Difference of Gaussians (DoG)** for edge and boundary detection
- **Intensity-weighted grid sampling** for spatial coverage
- **Spatial deduplication of seeds** to remove redundant seeds

### 2. Model Architecture
Each Gaussian splat is parameterized using covariance matrix representation:

**Covariance Matrix Parameterization**
- **Centers (μ)**: Sigmoid-bounded to stay within image domain
- **Covariance (Σ)**: Cholesky decomposition `Σ = L @ L^T` ensures positive definiteness
- **Amplitudes (a)**: Softplus activation for non-negativity
- **Sharpness (s)**: Exponential mapping `s = 2 * exp(s')` controls edge falloff (default s=2 for standard Gaussian)

Mathematical form (shifted Gaussian for C⁰ continuity at truncation boundary):
```
C     = exp(-0.5 * T²)              // boundary value (T = truncation radius)
scale = 1 / (1 - C)                 // peak-preserving rescale
f(x)  = a * scale * max(0, exp(-0.5 * ||y||²) - C)
```
where `y = Σ^(-1/2) @ (x-μ)`. The shift ensures zero intensity at the truncation boundary (no discontinuity from hard truncation).

The implementation avoids explicit matrix inversion by solving the triangular system `L @ y = (x-μ)`
and computing the quadratic form as `||y||²`.

### 3. Optimization Process
The fitting uses PyTorch with advanced optimization strategies:
- **Standard PyTorch Adam**: Fast vectorized optimizer with automatic gradient dilution compensation
- **Convergence-Based Dynamic Operations**: Fixed-pool splat relocation instead of add/remove operations
- **Asymmetric Loss Functions**: Configurable penalty for over-prediction addresses additive model constraints
- **Adaptive Amplitude Thresholds**: Scale with local residual magnitude to prevent plateaus
- **Early Stopping**: Maximum absolute error convergence criterion
- **Adaptive Learning Rate**: ReduceLROnPlateau scheduler for automatic LR adjustment
- **Regularization**: Optional L1 penalties on amplitudes and diagonal elements for sparsity and shape control

### 4. Rendering Pipeline
Efficient rendering using batched operations with two computational approaches:

**Efficient Rendering Pipeline:**
- **AABB Truncation**: Each splat rendered only within `truncate * σ` radius
- **Batched Triangular Solve**: Avoids explicit matrix inversion via `L @ y = (x-μ)`
- **Specialized 2D/3D Paths**: Optimized renderers with explicit forward-substitution
- **Amplitude-aware Culling**: Reduces computation for weak splats
- **Scatter-Add Accumulation**: Efficient GPU memory operations

## Performance Optimizations

The implementation includes several key optimizations that provide significant speedup:

1. **2D/3D Specialized Paths**: Optimized renderers with explicit forward-substitution for common cases
2. **Convergence Detection**: Automatically stops when loss plateaus (saves a substantial fraction of iterations)
3. **Adaptive Learning**: Reduces learning rate on plateaus for better convergence
4. **Device-Aware Selection**: Automatic selection of best available device (CUDA > MPS > CPU)
5. **Cached Computations**: Reuses grids and strides for repeated operations
6. **Optional Enhancements**:
   - Model compilation with `torch.compile` (PyTorch 2.0+, CUDA only)
   - Mixed precision training (FP16 on CUDA)
   - Pre-allocated buffers for memory efficiency

## Quick Start

### Simple One-Step API

The easiest way to use Gaussian splatting with intelligent defaults:

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy

# 1. One-step fitting with intelligent defaults
result = fit_gaussian_splats(
    image,
    # seeds auto-generated with volume-proportional scaling
    # norm_percentile=0.0 by default (full range normalization)
    n_iters=300,                          # Maximum iterations
    lr=0.01,                              # Stable learning rate
    asymmetric_penalty=1.0,               # Over-prediction penalty factor (default)
    loss_type="mse",                      # "mse", "poisson", or "l1" for robust features
    # max_abs_error auto-set to 0.01 (1% of normalized range)
    # l1_amp auto-set to 0.1 * lr for proportional amplitude regularization
    # l1_diag auto-set to 0.01 * lr for mild diagonal regularization
    # Standard Gaussian (s=2) is hardcoded
    # enable_dynamic_ops=True by default for optimal results
)

# 2. Access results and render directly - clean and simple!
# Result contains: centers, amplitudes, cholesky_factors, stats
reconstruction = render_gaussians_numpy(image.shape, result, truncate=3.0)

```

**Note**: `fit_gaussian_splats()` returns a `GSplatData` dataclass with:
- `centers`: np.ndarray, shape (N, d) - Splat center positions
- `amplitudes`: np.ndarray, shape (N,) - Non-negative amplitudes
- `cholesky_factors`: np.ndarray, shape (N, d*(d+1)//2) - Packed Cholesky factors
- `colors`: Optional[np.ndarray], shape (N, 3) - RGB colors (uint8 or float32 for HDR)
- `stats`: Dict[str, Any] - Optimization statistics

The result can be directly passed to `render_gaussians_numpy()` or `render_gaussians_pytorch()` for rendering, or added to a Scene (see below).

### Tiled Fitting for Large Volumes

For volumes that exceed GPU memory, tiled fitting splits the data into overlapping tiles with cosine (Hann) apodization and fits each tile independently:

```python
from luxar.gsplats import fit_tiled

# Fit a large volume using tiles
result = fit_tiled(
    large_volume,            # np.ndarray or zarr.Array (lazy loading supported)
    tile_size=256,           # Tile size per axis (int or per-axis tuple)
    overlap=32,              # Overlap width for cosine blending
    n_iters=1000,            # Forwarded to fit_gaussian_splats per tile
    device="cuda",           # GPU for each tile
    verbose=True,            # Per-tile progress logging
)

# Result is a single GSplatData with all splats in global coordinates
print(f"Total splats: {result.n_splats:,}")
print(f"Tiles fitted: {result.stats['num_tiles']}")
print(f"Splats per tile: {result.stats['splats_per_tile']}")

result.save("tiled_output.gsplats.zarr")
```

**Individual tile fitting** (for Slurm or distributed workflows):

```python
from luxar.gsplats import compute_tile_specs, fit_tile

# 1. Compute tile grid (deterministic, identical on all workers)
specs = compute_tile_specs(volume.shape, tile_size=256, overlap=32)

# 2. Fit a single tile (e.g., on a Slurm job array)
tile_result = fit_tile(volume, specs[tile_index], device="cuda")

# 3. Merge results from all tiles
merged = GSplatData.concatenate(all_tile_results)
```

**Tiling module** (`tiling.py`):

- `TileSpec` -- Frozen dataclass describing one tile: grid index, slices, origin, shape, border flags, and actual per-axis overlap with neighbors.
- `compute_tile_specs(volume_shape, tile_size, overlap)` -- Deterministic grid of overlapping tiles in row-major order. Edge tiles are clamped to volume boundaries.
- `cosine_window(spec)` -- Builds an nD separable Hann apodization window from a `TileSpec`. Two adjacent windows sum to exactly 1.0 in the overlap zone (partition-of-unity property).

**Key properties:**
- Overlap must satisfy `overlap <= tile_size // 2` to avoid triple tile overlap.
- Cosine windows guarantee seamless blending without post-merge pruning.
- `fit_tile` rejects explicit seed arrays (use int count, float ratio, or None).
- zarr arrays are supported for out-of-core processing -- only one tile is materialized at a time.

### Advanced Usage: Custom Candidates

For specialized use cases requiring custom candidate generation:

```python
from luxar.gsplats.seeds import generate_seeds, seed_from_edges

# Custom seed generation with specific method
custom_seeds = seed_from_edges(
    image,
    n_seeds=1000,                # Custom density
    min_distance=2.0,            # Minimum seed spacing
    edge_threshold_rel=0.1,      # Edge detection threshold
)

# Or use unified entry point
custom_seeds = generate_seeds(image, method="edges")

result = fit_gaussian_splats(
    image, seeds=custom_seeds
)
```

### Controlling Seed Count

The `seeds` parameter supports multiple input types for flexible control:

```python
# Option 1: Auto-generate (default)
result = fit_gaussian_splats(image)  # ~1% of voxels

# Option 2: Exact count (NEW!)
result = fit_gaussian_splats(image, seeds=1000)  # Exactly 1000 splats
# - Auto-generates with low threshold if needed
# - Subsamples with spatial diversity + intensity if too many
# - Grid fallback ensures target is reached

# Option 3: Compression ratio (splat floats / image floats)
result = fit_gaussian_splats(image, seeds=0.1)  # Target 10% of original storage

# Option 4: Explicit seed array
custom_seeds = np.array([[10, 20], [30, 40]])  # (N, ndim)
result = fit_gaussian_splats(image, seeds=custom_seeds)
```

### Auto-Seed Generation Features

- **Universal scales**: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) detect features from fine details to large structures
- **Volume-proportional density**: Automatically scales candidate count with image size (~0.2% of pixels)
- **Inclusive detection**: 70% percentile threshold for comprehensive feature coverage
- **Dimension-agnostic**: Works seamlessly with 2D images, 3D volumes, and 4D+ hypercubes (validated to 4D)

### Outlier-Robust Normalization

Control how outliers and noise are handled during normalization:

```python
# Full range (default) - maximum dynamic range
result = fit_gaussian_splats(image, norm_percentile=0.0)

# Robust to mild outliers - ignore bottom/top 1%
result = fit_gaussian_splats(image, norm_percentile=1.0)

# Very robust to noise - ignore bottom/top 5%
result = fit_gaussian_splats(image, norm_percentile=5.0)
```

**When to use:**
- **norm_percentile=0.0**: Clean data without outliers (maximum sensitivity)
- **norm_percentile=1.0**: Typical datasets with occasional outliers
- **norm_percentile=5.0**: Noisy data or datasets with many outliers

### Important Limitations

- **DC Component**: Gaussian splatting cannot represent uniform background intensities (DC components)
- **Focus**: The method excels at approximating variations, structures, and patterns, not constant baselines
- **Reconstruction**: Results preserve relative intensity relationships but may have different absolute baseline

### Comprehensive Logging

The system provides detailed optimization progress logging with best state tracking:

```
Convergence criterion: max absolute error < 0.010000
Maximum iterations: 1000
Auto-generating candidates: 131 peaks/scale for 65,536 pixels
Generated [0-9]* seed centers

[   1/1000] loss=0.078185  relL2=0.7969  maxAbsErr=1.1798  N=212
    ★ New best state: iteration 1, loss=0.078185  max_abs_error=0.978093
[  10/1000] loss=0.056279  relL2=0.7099  maxAbsErr=0.99667  N=212
    ★ New best state: iteration 8, loss=0.056279  max_abs_error=0.045821
...
✓ CONVERGENCE ACHIEVED at iteration 127
  Max absolute error: 0.009854 < threshold: 0.010000

★ Restored best state from iteration 125 (loss=0.056279  max_abs_error=0.009854)
Rescaled amplitudes to original intensity range (factor: 0.9075)
```

### Quality Guarantee Features

- **Best state tracking**: Always returns the splat configuration with lowest loss
- **Smart logging**: Reports significant improvements and early progress
- **State restoration**: Uses best quality achieved, not potentially suboptimal final state
- **Non-monotonic protection**: Handles optimization fluctuations and dynamic operations gracefully

## Asymmetric Loss Functions

The implementation includes asymmetric loss functions that address the fundamental constraints of additive Gaussian models:

### Why Asymmetric Loss?

**The Problem**: Gaussian splatting uses non-negative additive models: `prediction = Σ(positive_gaussians)`
- **Under-prediction** (`pred < target`): Easy to fix by adding more Gaussians
- **Over-prediction** (`pred > target`): Hard to fix, requires reducing/moving existing splats

**The Solution**: Asymmetric loss with configurable penalty for over-prediction (default: 1.0)
- **MSE**: `mean(where(pred > target, F * (pred - target)², (pred - target)²))` where F = `asymmetric_penalty`
- **Poisson**: Similar penalty applied to Poisson deviance

### Benefits

- **Better Optimization**: Avoids hard-to-correct over-prediction errors
- **Stable Convergence**: Reduces oscillations and improves trajectory
- **Model Alignment**: Reflects additive nature of Gaussian splatting
- **Enhanced Quality**: Enables better late-stage reconstruction improvements

## Loss Function Selection

The package supports three loss functions, each optimized for different data characteristics:

### **MSE (Mean Squared Error)** - Default
- **Best for**: Smooth data with Gaussian noise, general-purpose reconstruction
- **Characteristics**: Fast convergence, well-behaved gradients, penalizes large errors heavily
- **Use when**: Working with natural images, smooth volumetric data, or when in doubt

### **Poisson**
- **Best for**: Count/photon data, fluorescence microscopy, low-light imaging
- **Characteristics**: Optimal for Poisson noise statistics, handles non-negative intensities naturally
- **Use when**: Working with camera data, microscopy images, or any counting processes

### **L1 (Mean Absolute Error)**
- **Best for**: Data with outliers, sharp features, challenging datasets requiring robustness
- **Characteristics**: Robust to outliers, preserves edges, encourages sparse residuals
- **Use when**: Images with sharp boundaries, noisy data, or when MSE over-smooths features
- **Special synergy**: L1 + asymmetric penalty provides exceptional stability

**All loss functions support asymmetric penalties** for optimal performance with additive Gaussian models.

## Regularization Controls

The system includes two types of L1 regularization to control model complexity and shape characteristics:

### **Amplitude Regularization (`l1_amp`)**
- **Purpose**: Promotes sparsity by penalizing splat amplitudes
- **Default**: `0.1 * lr` (10% of learning rate)
- **Effect**: Encourages fewer, higher-quality splats by suppressing weak contributions
- **Use when**: You want to reduce the number of active splats for efficiency or interpretability

### **Diagonal Regularization (`l1_diag`)**
- **Purpose**: Encourages smaller, more isotropic (circular/spherical) splats
- **Default**: `0.01 * lr` (1% of learning rate)
- **Effect**: Regularizes the Cholesky diagonal elements, promoting smaller, rounder splats
- **Mathematical insight**: When diagonal parameters → 0, splats become small and axis-aligned

### **Combined Usage**
```python
result = fit_gaussian_splats(
    image,
    l1_amp=0.02,      # Strong amplitude sparsity (2% of lr)
    l1_diag=0.005,    # Mild shape regularization (0.5% of lr)
    lr=0.01,
    n_iters=300
)
```

**Benefits:**
- **Model parsimony**: Fewer, cleaner splats with controlled shapes
- **Overfitting prevention**: Regularization prevents excessive model complexity
- **Interpretable results**: Simpler splat shapes are easier to understand
- **Automatic scaling**: Both regularization terms scale proportionally with learning rate

## Fixed-Pool Splat Relocation

The system uses **fixed-pool splat relocation** instead of add/remove operations for optimal performance:

### **Problem**: Traditional Add/Remove Approach
- Adding/removing splats changes tensor shapes
- Requires complex per-splat optimizer state management
- Significantly slower than standard vectorized optimization

### **Solution**: Fixed-Pool Relocation
- Identify weak splats (low importance = amplitude × volume)
- Identify high-residual peaks
- **Relocate** weak splats to peaks (just parameter updates, no shape change)
- Standard PyTorch Adam works naturally

### **Benefits**
- **Significantly faster**: Standard vectorized Adam optimizer
- **Simple architecture**: No complex state management
- **Natural adaptation**: Adam's momentum quickly adapts to relocated splats
- **Spatial coverage**: Weak splats are reused for uncovered regions

## Gradient Dilution Compensation

The system automatically handles nD optimization challenges through intelligent learning rate scaling:

### **Problem**: Multi-Factor Gradient Dilution in Higher Dimensions
- **Parameter growth**: 2D (5 params), 3D (9 params), 4D (14 params) per splat
- **Signal dilution**: Same loss gradient distributed across more parameters
- **Spatial complexity**: 4D optimization landscape much more challenging than 2D/3D
- **Optimization difficulty**: Higher dimensions converge much slower without proper compensation

### **Solution**: Automatic Learning Rate Scaling
```python
# Applied automatically by create_optimizer_and_scheduler()
gradient_dilution_factor = calculate_gradient_dilution_factor(d)
effective_lr = base_lr × gradient_dilution_factor

# Result: 2D: lr × 1.0, 3D: lr × 1.8, 4D: lr × 8.5
```

### **Benefits**
- **Dimensional fairness**: Each dimension optimizes with equivalent effectiveness
- **Automatic scaling**: No manual learning rate tuning needed for different dimensions
- **Mathematical foundation**: Compensates for fundamental gradient dilution effect
- **Validated performance**: Enables excellent 4D convergence with standard lr=0.01 input

## Dynamic Operations

The implementation features **fixed-pool splat relocation** that automatically improves coverage based on reconstruction quality:

### Fixed-Pool Relocation

- **Weak splat detection**: Identify splats with low importance (amplitude × volume)
- **Residual peak detection**: Find high-error regions using non-maximum suppression
- **Relocation**: Move weak splats to high-residual peaks (no tensor shape changes)
- **Standard Adam**: Works naturally since tensor shapes are fixed

### Key Benefits

- **Significantly faster**: Standard vectorized PyTorch Adam optimizer
- **Convergence Alignment**: Operations directly serve optimization goals
- **Simple Architecture**: No complex per-splat state management
- **Quality Focus**: Continuous improvement throughout optimization

### Enabling Dynamic Operations

```python
from luxar.gsplats import DynamicOpsConfig

# Create configuration
config = DynamicOpsConfig()
config.step_every = 50              # Run every 50 iterations
config.k_max_residuals = 40         # Max peaks to find
config.nms_radius_vox = 2.0         # Non-maximum suppression radius
config.relocation_percentile = 1.0  # % of weakest splats to relocate
config.max_relocations_per_step = 64  # Cap relocations per step

# Fit with dynamic operations
result = fit_gaussian_splats(
    image,
    enable_dynamic_ops=True,
    dynamic_config=config,
    n_iters=300
)
```

### Configuration Guidelines

**Key Parameters:**
- `step_every`: Operation frequency (default: 50 iterations)
- `k_max_residuals`: Maximum peaks to analyze per step
- `relocation_percentile`: Percentage of weakest splats eligible for relocation
- `max_relocations_per_step`: Cap on relocations per dynamic ops step

**Performance vs Quality:**
- More frequent operations: Better adaptation, slightly more overhead
- Higher relocation percentile: More aggressive redistribution
- Lower max_relocations: More conservative, stable optimization

## Standard Optimizer with Fixed-Pool Architecture

The implementation uses **standard PyTorch Adam** with a fixed-pool splat architecture for optimal performance:

### Key Benefits

**Significantly Faster**: Standard vectorized Adam is dramatically faster than per-splat alternatives.

**Natural Momentum Adaptation**: When splats are relocated, Adam's momentum buffers at those indices quickly adapt to the new location as new gradients overwrite stale momentum.

**Simple Architecture**: No complex state management - just standard PyTorch optimizer with fixed tensor shapes.

### How It Works

The fixed-pool architecture keeps tensor shapes constant:

1. **No topology changes**: Splat pool size is fixed throughout optimization
2. **Relocation = parameter update**: Just modifies values, not tensor shapes
3. **Momentum adaptation**: Stale momentum at relocated splat quickly overwritten by new gradients
4. **Full adaptation**: Within 1-3 iterations after relocation

```python
# Standard optimizer is always used
from luxar.gsplats.optim import create_optimizer_and_scheduler

optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=0.01,  # Base LR, auto-compensated for gradient dilution
    scheduler_type="plateau"
)
```

### Technical Details

- **Gradient dilution compensation**: Learning rate automatically scaled based on dimensionality
- **Standard Adam buffers**: Uses PyTorch's built-in momentum and squared gradient tracking
- **Scheduler support**: ReduceLROnPlateau or ExponentialLR for learning rate adaptation

## Advanced Usage

For detailed control and statistics, use the class interface:

```python
from luxar.gsplats.fit_gsplats import GaussianSplatFitter

# Initialize fitter with specific device and options
fitter = GaussianSplatFitter(
    device="cuda",              # or "mps", "cpu"
    compile_model=True,         # torch.compile (CUDA only)
    use_mixed_precision=True,   # FP16 (CUDA only)
)

# Fit with detailed statistics
    result = fitter.fit(
        image,
        seeds=candidates,
        n_iters=500,
        early_stopping=True,
        early_stop_patience=200,
        sigma_min_diag=[0.5, 0.5],  # Minimum splat size
        sigma_max_diag=[10.0, 10.0], # Maximum splat size
    )

# Access optimization statistics from result.stats
print(f"Time: {result.stats['time_seconds']:.2f}s")
print(f"Iterations: {result.stats['iterations']}/{500}")
print(f"Converged: {result.stats['converged']}")
print(f"Final loss: {result.stats['final_loss']:.5g}")
```

## Quality Metrics

The `metrics` module provides GPU-accelerated quality metrics for evaluating Gaussian splat reconstructions. All heavy computation stays on the input device (CUDA/MPS/CPU); only scalar results are moved to CPU.

### Available Metrics

```python
from luxar.gsplats.metrics import compute_quality_metrics, compute_psnr, compute_ssim

# Compute all metrics at once
metrics = compute_quality_metrics(pred_tensor, target_tensor)
print(f"PSNR: {metrics['psnr_db']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"MSE:  {metrics['mse']:.6e}")
print(f"Relative L2: {metrics['rel_l2']:.4f}")
print(f"Max absolute error: {metrics['max_abs_error']:.4f}")

# Or compute individual metrics
psnr = compute_psnr(pred_tensor, target_tensor)
ssim = compute_ssim(pred_tensor, target_tensor, window_size=11)
```

### Functions

- `compute_psnr(pred, target, data_range=None)` -- Peak Signal-to-Noise Ratio in dB. Returns `float('inf')` when MSE is zero.
- `compute_ssim(pred, target, window_size=11, data_range=None)` -- Structural Similarity Index using nD Gaussian-weighted convolution. Supports 2D, 3D, and higher (averages over 3D sub-volumes for >3D).
- `compute_quality_metrics(pred, target, data_range=None, ssim_window_size=11)` -- Computes all metrics in one call. Returns a dict with keys: `mse`, `psnr_db`, `ssim`, `rel_l2`, `max_abs_error`.

### Quality Metrics in GSplatData.stats

When fitting completes, quality metrics may be stored in the `stats` dict of the returned `GSplatData`:

- `stats['psnr_db']` -- PSNR of the reconstruction vs. original volume (dB)
- `stats['ssim']` -- SSIM of the reconstruction vs. original volume
- `stats['mse']` -- Mean squared error of the reconstruction

These fields are populated automatically when metrics are computed during or after fitting, enabling easy comparison across fitting runs.

### Typical Usage with Rendering

```python
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.rendering import render_to_volume_tensor
from luxar.gsplats.metrics import compute_quality_metrics

# Fit and render back to a tensor (stays on GPU)
result = fit_gaussian_splats(volume, n_iters=1000, device="cuda")
rendered = render_to_volume_tensor(result, shape=volume.shape, device="cuda")

# Compute metrics without GPU-CPU round-trip
import torch
target = torch.from_numpy(volume).to("cuda")
metrics = compute_quality_metrics(rendered, target)
print(f"PSNR={metrics['psnr_db']:.1f} dB, SSIM={metrics['ssim']:.4f}")
```

## Rendering

The `rendering` module provides GPU-accelerated volume rendering with automatic backend selection (CUDA > MPS > CPU).

### render_to_volume

Renders splats to a NumPy array:

```python
from luxar.gsplats.rendering import render_to_volume

volume = render_to_volume(gsplat_data, shape=(128, 128, 128))
```

### render_to_volume_tensor

Renders splats to a `torch.Tensor` on the rendering device, avoiding an unnecessary GPU-to-CPU copy when the result feeds into further GPU operations (e.g., quality metric computation):

```python
from luxar.gsplats.rendering import render_to_volume_tensor

tensor = render_to_volume_tensor(gsplat_data, shape=(128, 128, 128), device="cuda")
# tensor is on CUDA -- pass directly to metrics or further processing
```

**Parameters** (shared by both functions):
- `gsplat_data`: GSplatData to render
- `shape`: Output volume shape, e.g. `(128, 128, 128)`
- `device`: `"cuda"`, `"mps"`, `"cpu"`, or `None` (auto-detect)
- `truncate`: Truncation radius in standard deviations (default 3.0)
- `intensity_floor`: Amplitude-aware culling threshold (default 1e-5)
- `chunk_size`: Optional chunk size for memory management on large volumes

## Saving, Loading, and Scene Integration

### Saving GSplatData

Save fitted splats to disk in the `.gsplats.zarr` format for later use:

```python
from luxar.gsplats import fit_gaussian_splats

# Fit splats
result = fit_gaussian_splats(image, n_iters=1000)

# Save to file with spatial ordering for efficient access
result.save('fitted.gsplats.zarr',
           ordering='hilbert',  # or 'morton', 'none'
           color_mode='sdr')    # 'sdr' (uint8) or 'hdr' (float32)

# Colors are automatically saved if present in result
```

### Loading GSplatData

Load previously saved splats back into a GSplatData object:

```python
from luxar.gsplats.io import load_gsplats

# Load from disk
result = load_gsplats('fitted.gsplats.zarr')

# Access all fields
print(f"Loaded {result.centers.shape[0]} splats")
print(f"Has colors: {result.colors is not None}")

# Render loaded splats
from luxar.gsplats.models.gsplats.rendering_wrappers import render_gaussians_numpy
reconstruction = render_gaussians_numpy(image.shape, result, truncate=3.0)
```

### Adding to Luxar Scenes

Integrate fitted Gaussian splats directly into Luxar scenes for visualization:

**Option 1: Add from GSplatData (no intermediate save)**
```python
from luxar import LuxarZarrCompiler, Dimensions
from luxar.gsplats import fit_gaussian_splats

# Fit splats
image = np.random.rand(100, 100).astype(np.float32)
result = fit_gaussian_splats(image, n_iters=1000)

# Add directly to scene
with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data('fitted', result)
    print(f"Added {gsplats.n_splats} splats with colors={gsplats.has_colors}")
```

**Option 2: Add from saved .gsplats.zarr file**
```python
# First save result
result.save('fitted.gsplats.zarr', ordering='hilbert')

# Later, load into a scene
with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_file('loaded', 'fitted.gsplats.zarr')
    print(f"Loaded {gsplats.n_splats} splats")
```

**Benefits:**
- Seamless integration with Luxar's scene graph system
- Preserves all splat data (centers, amplitudes, covariance, colors)
- Enables hierarchical organization with transforms
- Works with Luxar viewer for interactive visualization

### Color Support

GSplatData now supports optional per-splat RGB colors:

```python
# Create result with colors
result = GSplatData(
    centers=centers,
    amplitudes=amplitudes,
    cholesky_factors=cholesky_factors,
    colors=np.random.rand(n_splats, 3).astype(np.float32),  # Add colors
    stats={}
)

# Colors are preserved during save/load
result.save('colored.gsplats.zarr', color_mode='sdr')  # uint8 [0-255]
result.save('hdr_colored.gsplats.zarr', color_mode='hdr')  # float32 HDR

# Load preserves colors
loaded = load_gsplats('colored.gsplats.zarr')
assert loaded.colors is not None
assert loaded.colors.shape == (n_splats, 3)

# Scene integration preserves colors
with LuxarZarrCompiler('scene.zarr') as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data('colored', result)
    assert gsplats.has_colors == True
```

## API Reference

### Main Functions

#### `fit_gaussian_splats(V, seeds, **kwargs)`
Main fitting function with automatic optimizations.

**Key Parameters:**
- `V`: Input n-dimensional array to reconstruct
- `seeds`: Initial candidate positions (N, d), int count, or float compression ratio
- `n_iters`: Maximum iterations (default: 1000)
- `lr`: Learning rate (default: 0.01)
- `loss_type`: "mse", "poisson", or "l1" (default: "mse")
- `l1_amp`: L1 regularization on amplitudes (default: 0.1 * lr)
- `l1_diag`: L1 regularization on diagonal elements (default: 0.01 * lr)
- `sigma_min_diag`: Minimum Gaussian size per axis
- `sigma_max_diag`: Maximum Gaussian size per axis
- `early_stopping`: Enable convergence detection (default: True)
- `device`: PyTorch device (auto-detect if None)

**Returns:** `GSplatData` with fields:
- `centers`: np.ndarray, shape (N, d) - Splat center positions
- `amplitudes`: np.ndarray, shape (N,) - Non-negative amplitudes
- `cholesky_factors`: np.ndarray, shape (N, d*(d+1)//2) - Packed Cholesky factors
- `colors`: Optional[np.ndarray], shape (N, 3) - RGB colors
- `stats`: Dict[str, Any] - Optimization statistics

#### `fit_tiled(volume, tile_size=256, overlap=32, **fit_kwargs)`
Tiled fitting for large volumes that exceed GPU memory. Splits the volume into
overlapping tiles with Hann cosine apodization, fits each tile independently,
and concatenates results. The partition-of-unity property eliminates seam artifacts.

**Key Parameters:**
- `tile_size`: Tile size per axis (int or tuple). Must satisfy `overlap <= tile_size // 2`.
- `overlap`: Overlap width per axis for cosine blending.
- `voxel_size`: Physical voxel spacing (optional), forwarded to per-tile fitting.
- `output_space`: `"real"` or `"voxel"` coordinate space for output centers.
- `verbose`: Print per-tile progress (default: True).
- `**fit_kwargs`: All parameters from `fit_gaussian_splats` (seeds, n_iters, device, etc.)

**Returns:** `GSplatData` with all splats in global coordinates. Hilbert curve resorting happens automatically on `save()`.

#### `fit_tile(volume, spec, **fit_kwargs)`
Fit a single tile (Slurm-ready). Takes a `TileSpec` from `compute_tile_specs()` and
returns `GSplatData` with centers already in global coordinates.

#### `generate_seeds(V, method="auto", **kwargs)`
Unified entry point for all seed generation methods.
Returns `GSplatData` with scale-informed Gaussian shapes.

**Key Parameters:**
- `method`: `"auto"` (edges+grid), `"edges"`, `"grid"`, `"decomposition"`, or comma-separated combo
- `**kwargs`: Passed to the selected seeding method(s)

#### `seed_from_edges(V, n_seeds=None, min_distance=2.0, edge_threshold_rel=0.1, device=None, ...)`
Edge-based seeding using Sobel gradients with Poisson disk sampling.

#### `seed_from_grid(V, spacing=None, jitter=0.0, sigma=None, ...)`
Uniform grid seeding for spatial coverage with optional jitter.

#### `seed_from_decomposition(V, scales=..., ignore_finest_k=1, ...)`
Scale-hierarchical detection via optimized image decomposition.

### Tiling Functions

#### `compute_tile_specs(volume_shape, tile_size, overlap)`
Compute a deterministic grid of overlapping tiles covering a volume. Returns a list of `TileSpec` in row-major order (identical inputs always produce identical output).

#### `cosine_window(spec)`
Build an nD cosine (Hann) apodization window for a tile. Boundary faces stay at 1.0; interior faces are tapered over the actual overlap with the neighboring tile.

#### `TileSpec`
Frozen dataclass with fields: `index`, `grid_index`, `slices`, `origin`, `shape`, `border_low`, `border_high`, `overlap_low`, `overlap_high`.

### Quality Metrics Functions

#### `compute_quality_metrics(pred, target, data_range=None, ssim_window_size=11)`
Compute all quality metrics (MSE, PSNR, SSIM, relative L2, max absolute error) between two tensors. Returns a dict.

#### `compute_psnr(pred, target, data_range=None)`
Peak Signal-to-Noise Ratio in dB. Returns `float('inf')` when MSE is zero.

#### `compute_ssim(pred, target, window_size=11, data_range=None)`
Structural Similarity Index via nD Gaussian-weighted convolution. Supports 2D, 3D, and higher dimensions.

### Rendering Functions

#### `render_to_volume(gsplat_data, shape, device=None, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`
Render Gaussian splats to a NumPy array.

#### `render_to_volume_tensor(gsplat_data, shape, device=None, truncate=3.0, intensity_floor=1e-5, chunk_size=None)`
Render Gaussian splats to a `torch.Tensor` on the rendering device. Avoids GPU-to-CPU copy for downstream GPU operations.

## Device Support and Performance

The implementation supports multiple PyTorch devices with performance-aware auto-selection:

| Device | Auto-Selected | Performance | Notes |
|--------|---------------|-------------|-------|
| **CPU** | ✓ (default on Mac) | Baseline | Reliable, well-optimized |
| **MPS** (Apple Silicon) | Manual only | Slower than CPU | Compatible but has overhead issues |
| **CUDA** | ✓ (when available) | Significant speedup | Best performance, full features |

### Device Selection Logic:
```python
# Auto-detection priority: CUDA → MPS → CPU
fitter = GaussianSplatFitter()  # Uses best available

# Manual device selection:
fitter = GaussianSplatFitter(device="mps")    # Force MPS
fitter = GaussianSplatFitter(device="cuda")   # Force CUDA
fitter = GaussianSplatFitter(device="cpu")    # Force CPU
```

### Apple Silicon Performance Notes:

**PyTorch MPS Issues**: The PyTorch MPS backend has significant overhead for `torch.linalg.solve_triangular` operations, making CPU substantially faster than MPS on Apple Silicon chips.

**MLX Alternative Investigated**: Apple's MLX framework was evaluated as a potential solution. However, MLX's `solve_triangular` operation is currently CPU-only (not GPU-accelerated) and slower than PyTorch's CPU implementation.

**Current Optimal Strategy**: Auto-select CPU on Apple Silicon, which provides the best performance available. This will automatically benefit from future improvements in either PyTorch MPS or MLX GPU acceleration.

### Metal Backend (Apple Silicon GPU Acceleration)

For 3D volumes on Apple Silicon, a native **Metal compute shader** backend is available that provides a **significant speedup** over CPU rendering:

```python
from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

# Drop-in replacement for GaussianSplatModel
model = GaussianSplatModelMetal(
    shape=(64, 64, 64),
    centers0=centers,
    L0=L,
    amps0=amps,
    truncate=3.0,
    device='mps'  # Must be MPS
)

# Use like normal PyTorch model
output = model()
loss = criterion(output, target)
loss.backward()  # Gradients computed via Metal kernels
```

**Requirements**: macOS with Apple Silicon (M1/M2/M3/M4), full Xcode installation.

**Compilation**: The Metal backend requires one-time compilation:

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
python setup.py build_ext --inplace
```

See [metal/README.md](models/gsplats/metal/README.md) for detailed installation and troubleshooting.

## Package Structure

```
gsplats/
├── __init__.py                    # Public API exports (lazy import with graceful fallback)
├── fit_gsplats.py                 # Main fitting interface (orchestrates modular pipeline)
├── fit_progressive_gsplats.py     # Progressive fitting (iterative refinement)
├── fit_tiled_gsplats.py           # Tiled fitting for large volumes (fit_tile, fit_tiled)
├── tiling.py                      # Tile geometry and cosine apodization (TileSpec, cosine_window)
├── gsplat_data.py                 # GSplatData / GSplatLOD dataclasses, save/load, transforms
├── culling.py                     # Contribution-based splat culling (CullResult, cull_by_contribution)
├── metrics.py                     # Quality metrics (PSNR, SSIM, MSE, relative L2)
├── gpu_profile.py                 # GPU benchmark profile management
│
├── seeds/                         # Seed generation subpackage
│   ├── generate.py                # Unified entry point (generate_seeds)
│   ├── edges.py                   # Edge-based seeding (Sobel + Poisson disk)
│   ├── grid.py                    # Grid-based seeding with optional jitter
│   ├── multiscale_decomposition.py # Scale-hierarchical seed detection
│   ├── peaks.py                   # Peak detection utilities
│   ├── gpu_ops.py                 # GPU-accelerated seeding operations
│   └── utils.py                   # Seeding helper utilities
│
├── fitting/                       # Modular fitting pipeline
│   ├── config.py                  # Configuration dataclasses (FitConfig, PreprocessedData, etc.)
│   ├── validation.py              # Input validation and parameter checking
│   ├── preprocessing.py           # Data normalization and candidate generation
│   ├── initialization.py          # Model and optimizer initialization
│   ├── losses.py                  # Loss function creation (MSE, Poisson, L1)
│   ├── optimization.py            # Main optimization loop and convergence logic
│   ├── results.py                 # Result finalization and statistics
│   ├── sorting.py                 # Spatial sorting (Hilbert, Morton)
│   ├── downscale.py               # Volume downscaling utilities
│   ├── visualization.py           # Movie recording and compression analysis
│   └── dynamic_ops/               # Convergence-driven splat relocation
│       ├── config.py              # DynamicOpsConfig dataclass
│       ├── operations.py          # Relocation logic
│       └── peak_finding.py        # Residual peak detection
│
├── rendering/                     # Volume rendering module
│   ├── __init__.py                # Exports render_to_volume, render_to_volume_tensor
│   └── volume_rendering.py        # GPU-accelerated rendering with auto backend selection
│
├── optim/                         # Optimizer utilities
│   └── integration.py             # Standard Adam with gradient dilution compensation
│
├── models/
│   ├── gsplats/
│   │   ├── gsplat_model.py        # PyTorch model definition
│   │   ├── rendering_core.py      # Core rendering kernels
│   │   ├── rendering_wrappers.py  # NumPy/PyTorch rendering convenience wrappers
│   │   ├── cuda/                  # CUDA kernels for GPU-accelerated splatting
│   │   └── metal/                 # Metal compute shaders for Apple Silicon
│   └── utils/
│       ├── inverse_softplus.py    # Numerical utilities
│       └── lt_solver.py           # Triangular system solver
│
├── io/                            # I/O subpackage
│   ├── save_gsplats.py            # Save GSplatData to .gsplats.zarr
│   ├── load_gsplats.py            # Load GSplatData from .gsplats.zarr
│   └── inspect_gsplats.py         # Inspect and summarize .gsplats.zarr files
│
├── batch/                         # HPC batch fitting (Slurm integration)
│   ├── manifest.py                # Batch job manifest management
│   ├── slurm_gen.py               # Slurm script generation
│   ├── merge_orchestrator.py      # Tile merge orchestration
│   ├── status.py                  # Job status tracking
│   ├── time_estimate.py           # Fitting time estimation
│   └── env_capture.py             # Environment capture for reproducibility
│
├── clahe/                         # CLAHE (adaptive histogram equalization)
│   └── clahe_core.py              # Core CLAHE implementation
│
├── preprocessing/                 # Volume preprocessing (denoising, calibration)
│   ├── nlm_core.py                # Non-Local Means denoising (NumPy)
│   ├── nlm_pytorch.py             # Non-Local Means denoising (PyTorch)
│   ├── denoise_pipeline.py        # Denoising pipeline orchestration
│   ├── calibration.py             # Noise calibration
│   └── cuda/                      # CUDA-accelerated preprocessing
│
├── multiscale/                    # Multiscale volume decomposition
│   └── decompose.py               # Scale-space decomposition
│
├── utils/
│   └── trils.py                   # Triangular matrix packing/unpacking
│
├── demos/                         # Interactive demonstrations
│   ├── demo_basic_fitting.py      # Simple API introduction
│   ├── demo_performance_metrics.py # Convergence and quality metrics
│   ├── demo_tiled_fitting.py      # Tiled fitting for large volumes
│   ├── demo_progressive_fitting.py # Progressive fitting demo
│   ├── demo_2d_synthetic_blobs.py # 2D compression analysis
│   ├── demo_3d_synthetic_phantom.py # 3D volumetric compression
│   ├── demo_3d_dapi_microscopy.py # Real DAPI microscopy from IDR (remote zarr)
│   ├── demo_3d_celegans_confocal.py # C. elegans confocal microscopy
│   ├── demo_4d_hypercube.py       # 4D hypercube - nD algorithm validation
│   ├── demo_splats_astronaut.py   # Astronaut photo compression analysis
│   ├── demo_splats_coins.py       # Coins image compression
│   ├── demo_splats_mitosis.py     # Mitosis histology compression
│   └── ...                        # Additional progressive/culling/seeding demos
│
└── tests/
    ├── test_gsplats_integration.py # Comprehensive integration tests
    ├── test_fit_gsplats.py         # Fitting function tests
    ├── test_gsplat_data.py         # GSplatData dataclass tests
    ├── test_culling.py             # Culling tests
    ├── test_metrics.py             # Quality metrics tests
    ├── test_tiled_fitting.py       # Tiled fitting tests
    ├── test_progressive_fitting.py # Progressive fitting tests
    ├── test_batch.py               # Batch fitting tests
    └── ...                         # Additional test files
```

### Modular Fitting Architecture

The fitting pipeline uses a modular architecture with focused, maintainable modules:

- **fit_gsplats.py**: Clean orchestration method that coordinates the pipeline
- **fitting/ modules**: Each handles a specific aspect of the fitting process
  - Individual components are independently testable
  - Clear separation of concerns across six pipeline stages
  - Type-safe configuration via dataclasses (FitConfig, PreprocessedData, etc.)
  - See `fitting/README.md` for the full pipeline architecture diagram

## Running Demos

**Standard execution (with napari visualization):**
```bash
# Tiled fitting for large volumes
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_tiled_fitting.py

# Getting started - simple API demos
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_basic_fitting.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_performance_metrics.py

# Dimensional progression - 2D → 3D → 4D compression analysis
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_2d_synthetic_blobs.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_3d_synthetic_phantom.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_4d_hypercube.py

# Real data demos
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_3d_dapi_microscopy.py  # Remote zarr loading

# Biological data demonstration
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_mitosis.py
```

**Headless execution (for testing/CI):**
```bash
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_performance_metrics.py --no-napari
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_4d_hypercube.py --no-napari  # 4D validation
```

### 4D Hypercube Validation

The `demo_4d_hypercube.py` demonstrates complete nD algorithm validation:

**4D Test Results:**
- **Hypercube data**: Synthetic 4D Gaussian blobs in a multi-frame volume
- **Auto-candidate generation**: Volume-proportional scaling with automatic density tuning
- **4D splat fitting**: Successfully generates 4D splats with full covariance parameterization
- **Strong compression**: Substantial bit reduction compared to raw voxel storage
- **Best state tracking**: Quality guarantee with restoration from optimal iteration
- **Interactive 4D visualization**: Full napari navigation with dimension sliders
- **nD algorithms validated**: All features working correctly in 4D space

## Testing

The gsplats package has comprehensive test coverage organized into unit tests (per subpackage) and integration tests:

```bash
# Run all gsplats tests
hatch run pytest packages/luxar/src/luxar/gsplats/ -v

# Run integration tests only
hatch run pytest packages/luxar/src/luxar/gsplats/tests/ -v

# Run fitting pipeline unit tests
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/ -v

# Run quality metrics tests
hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_metrics.py -v

# Run tiled fitting tests
hatch run pytest packages/luxar/src/luxar/gsplats/tests/test_tiled_fitting.py -v
```

**Test Organization:**
- `fitting/tests/` - Modular fitting pipeline
- `optim/tests/` - Optimizer integration
- `models/*/tests/` - Model and utility functions
- `multiscale/tests/` - Multiscale decomposition
- `tests/` - Integration tests for complete pipelines
  - Includes tests for quality metrics, tiled fitting, culling, batch, and progressive fitting

**Coverage:**
- Unit tests for all pipeline components (validation, preprocessing, losses, optimization, etc.)
- 2D/3D/nD reconstruction pipelines
- Loss functions (MSE, Poisson, L1) with asymmetric penalties
- Dynamic operations (seeding, pruning, merging, splitting)
- Quality metrics (PSNR, SSIM, MSE) for 2D and 3D data
- Tiled fitting with cosine apodization and tile merging
- Device compatibility (CPU, CUDA, MPS)
- Edge cases and error handling
- No interactive windows during tests (napari properly mocked)

## Performance Tips

1. **Device Selection**: Use GPU when available (significant speedup)
2. **Early Stopping**: Keep enabled for substantial iteration reduction
3. **Candidate Tuning**: Balance quality vs speed with `peaks_per_scale`
4. **Loss Function**: Use Poisson for photon/count data, MSE for general
5. **Regularization**: Add L1 penalty for sparser, faster solutions
6. **Compilation**: Enable on CUDA for additional speedup
7. **Tiled Fitting**: Use `fit_tiled()` for volumes exceeding GPU memory
8. **GPU Tensor Rendering**: Use `render_to_volume_tensor()` instead of `render_to_volume()` when feeding results into further GPU operations (avoids GPU-CPU round-trip)

## Troubleshooting

### Common Issues

**PyTorch Warning about requires_grad**
- Harmless warning from learning rate scheduler, can be ignored

**Early Stopping Not Triggering**
- Simple images may converge late
- Try reducing `early_stop_patience` for more aggressive stopping
- Check that `early_stopping=True` is set

**Out of Memory**
- Reduce `peaks_per_scale` in candidate generation
- Enable `use_mixed_precision=True` on CUDA
- Use smaller images or downsample
- Use `fit_tiled()` to process the volume in overlapping tiles

**Poor Reconstruction Quality**
- Increase `n_iters` or disable early stopping for maximum quality
- Add more candidates (increase `peaks_per_scale`)
- Adjust learning rate (`lr`)
- Try different `loss_type` for your data
- Check quality with `compute_quality_metrics()` to get PSNR/SSIM numbers

## Implementation Details

### Numerical Stability
- Cholesky decomposition ensures positive definite covariances
- Sigmoid bounding prevents centers from leaving image domain
- Softplus activation guarantees non-negative amplitudes
- Triangular solve avoids explicit matrix inversion

### Memory Efficiency
- Truncated rendering (default 3σ radius)
- Batched operations for GPU parallelism
- Optional FP16 mixed precision
- Pre-allocated buffers where possible
- Amplitude-aware culling for weak splats

### Convergence Strategy
- Monitors loss over sliding window
- Detects both plateaus and oscillations
- Adaptive learning rate on plateaus
- Best state tracking with restoration

## Interactive Demos

The package includes several interactive napari demos that showcase Gaussian splatting on real-world data:

### **Biological Imaging: Mitosis Demo**
```bash
python demo_splats_mitosis.py
```
- **Dataset**: Human mitosis histology from scikit-image
- **Focus**: Biological cell structures and morphology
- **Features**: Optimized for microscopy data with fine cellular details

### **Photography: Astronaut Demo**
```bash
python demo_splats_astronaut.py
```
- **Dataset**: Classic astronaut photograph from scikit-image
- **Focus**: Complex photographic content with faces, textures, and spatial details
- **Features**: Demonstrates regularization effects (`l1_diag` parameter) on natural images

### **Metallic Textures: Coins Demo**
```bash
python demo_splats_coins.py
```
- **Dataset**: Classic coins image from scikit-image
- **Focus**: Metallic surface textures, circular objects, and illumination gradients
- **Features**: Gold-themed UI, optimized parameters for metallic surfaces and coin boundaries

### **What the Demos Show**
- **Interactive compression analysis**: Slider to explore quality vs file size trade-offs
- **Oriented ellipse visualization**: See how splats capture image structure
- **Real-time statistics**: Bits per pixel, compression ratios, and reconstruction errors
- **Regularization effects**: Compare different L1 penalty settings
- **Dynamic operations**: Watch splat management during optimization

**Usage**: Each demo supports `--no-napari` flag for testing without the GUI.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{luxar2024,
  title = {Luxar: High-Performance Visualization Framework},
  year = {2024},
  url = {https://github.com/royerlab/luxar}
}
```

## License

MIT License - See repository for details
