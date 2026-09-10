# Gaussian Splatting Package for Luxar

High-performance n-dimensional oriented Gaussian splatting with automatic optimization for efficient image and volume reconstruction.

## Overview

This package implements a sophisticated Gaussian splatting system that fits collections of oriented Gaussian functions (splats) to reconstruct n-dimensional data. Each splat is a multivariate Gaussian characterized by position, full covariance matrix, and amplitude. The implementation features automatic convergence detection, adaptive learning, and GPU acceleration.

## Installation

Fitting Gaussian splats requires optional dependencies:

```bash
pip install "luxar[gsplats]"
```

> **Core-only carve-out.** The pure-NumPy half of this package works on a plain
> `pip install luxar`: the `utils.trils` Cholesky helpers, and `GSplatData` —
> construct, `.save()`, `.load()`, plus the geometric `translate` / `transform` /
> `center_at_centroid` — so an existing `.gsplats.zarr` can be authored and
> grafted into a scene (`scene.add_gsplats` / `add_gsplats_from_data` /
> `add_gsplats_from_file`) without the extra. Fitting, calibration, culling, LOD
> construction and intensity editing all need it; the `luxar.gsplats` package
> docstring states exactly where the line falls.

> **⚠️ torch ABI coupling.** The optional CUDA extension
> (`models/gsplats/cuda/cuda_splatting_backend*.so`) is compiled against a
> specific PyTorch ABI. **After any `torch` upgrade you must rebuild it** with
> `make build-cuda`, otherwise importing the backend fails with an ABI/symbol
> mismatch. The `torch>=2.2.0,<3.0` pin in `pyproject.toml` exists to prevent a
> silent major-version jump from breaking the prebuilt extension.

## Key Features

- **N-dimensional Support**: Works seamlessly with 2D images, 3D volumes, and 4D+ hypercubes (validated to 4D) with automatic gradient dilution compensation
- **Efficient Cholesky Parameterization**: Covariance matrices via Cholesky decomposition with batched triangular solve
- **Device Optimized**: Automatic device selection (CUDA → MPS → CPU); custom CUDA kernels on NVIDIA and an optional Metal backend on Apple Silicon
- **Oriented Gaussians**: Full covariance matrices via Cholesky decomposition for arbitrary orientations
- **Convergence-Driven Dynamic Operations**: Fixed-pool splat relocation based on convergence criteria (constant splat count — weak splats move to high-residual regions)
- **Asymmetric Loss Functions**: Configurable penalty for over-prediction (applies to all three losses: L1, MSE, Poisson) addresses additive model constraints
- **Standard PyTorch Adam**: Fast vectorized optimization with gradient dilution compensation (significantly faster than per-splat alternatives)
- **Adaptive Thresholds**: Amplitude validation scales with local residual magnitude to prevent optimization plateaus
- **Automatic Optimization**: Early stopping saves a substantial fraction of iterations without quality loss
- **GPU Acceleration**: CUDA support with batched operations, improved MPS compatibility
- **Robust Initialization**: Multiscale candidate detection with DoG and peak finding
- **Memory Efficient**: Truncated rendering and pre-allocated buffers
- **Tiled Fitting**: Overlapping tiles with Hann cosine apodization for volumes that exceed GPU memory
- **Quality Metrics**: Built-in PSNR, foreground PSNR, SSIM, and MSE computation on GPU tensors

## Quick Example

```python
from luxar.gsplats import fit_gaussian_splats

# Fit splats to your volume
result = fit_gaussian_splats(volume, n_iters=1000)

# Post-processing transformations
result = result.center_at_centroid()  # Center for easier viewing
result = result.scale_intensity(0.1)  # Reduce brightness 10x
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

These three (centers, Cholesky factors, amplitudes) are the only optimized
parameters — the falloff exponent is fixed at 2 (a standard Gaussian). The
normalized `[0, 1]` sharpness attribute found on Points/Lines (mapping to a
super-Gaussian exponent `β = 2^(6s−2)`, with `s = 0.5` ⇒ Gaussian) is a viewer
rendering knob for those geometry types and is not part of the gsplat fitting
model.

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
6. **Automatic CUDA Enhancements**:
   - Loss kernels are compiled with `torch.compile` automatically on CUDA (PyTorch 2.0+), with an eager fallback if compilation is unavailable
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
    n_iters=300,  # Maximum iterations
    lr=0.01,  # Stable learning rate
    asymmetric_penalty=1.0,  # Over-prediction penalty factor (default)
    loss_type="l1",  # default; alternatives: "mse", "poisson"
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
- `colors`: Optional[np.ndarray], shape (N, 3) RGB or (N, 4) RGBA (uint8 or float32 for HDR) — the optional alpha channel is per-splat opacity in [0, 1] (consumed by every blending mode; mapped to optical depth in `volumetric`)
- `stats`: Dict[str, Any] - Optimization statistics

The result can be directly passed to `render_gaussians_numpy()` or `render_gaussians_pytorch()` for rendering, or added to a Scene (see below).

### Tiled Fitting for Large Volumes

For volumes that exceed GPU memory, tiled fitting splits the data into overlapping tiles with cosine (Hann) apodization and fits each tile independently:

```python
from luxar.gsplats import fit_tiled

# Fit a large volume using tiles
result = fit_tiled(
    large_volume,  # np.ndarray or zarr.Array (lazy loading supported)
    tile_size=256,  # Tile size per axis (int or per-axis tuple)
    overlap=32,  # Overlap width for cosine blending
    n_iters=1000,  # Forwarded to fit_gaussian_splats per tile
    device="cuda",  # GPU for each tile
    verbose=True,  # Per-tile progress logging
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
- `grid_bsp_tree(specs, *, scale=None)` -- Serialized `bsp_tree` over a uniform tile grid, so the viewer paints tile-parts back-to-front instead of by centroid (#1555). `scale` is a per-axis, finite and strictly positive factor mapping the specs' voxel frame onto the frame the splats actually live in (see below).
- `resolve_grid_scale(ndim, *, downscale_factors=None, voxel_size=None, output_space="real")` -- Builds that `scale`. The two terms COMPOSE: a `--downscale` grid is computed on the decimated shape while the workers rescale their splats back to full resolution, and a `voxel_size` fit with `output_space="real"` emits physical centers. Returns `None` when both are no-ops (#1587). Raises on an `output_space` outside `("real", "voxel")` -- silently reading a typo as "voxel" would drop the spacing term, which is the very mismatch this closes -- on a `voxel_size` that is neither a scalar nor length-`ndim`, and on a resolved factor that is not finite and positive (a YAML `.nan` passes the fitter's own `<= 0` test and would otherwise become a `NaN` split plane). Every producer of a uniform tile grid's `bsp_tree` goes through it: `fit_tiled`, `dispatch_parallel_tiled`, and the `batch-fit` planner (which resolves the workers' `--preset`/`--config` once and records the answer as the manifest's `grid_scale`, for the merge to read).

**Key properties:**
- Overlap must satisfy `overlap <= tile_size // 2` to avoid triple tile overlap.
- The background floor (`floor`, default `"auto"`) is resolved once against the whole volume (never per tile) and subtracted from each tile before apodization — on the DENOISED basis when `--denoise` is active, since that is the data the level is subtracted from: `resolve_volume_floor_denoised` keeps the whole-volume level and adds the shift measured on a small bounded denoise probe, which reproduces the non-tiled path's denoised estimate exactly whenever the volume fits the probe budget. Above that budget the shift is applied for a `pNN` spec only; the default `auto` keeps its raw-basis level and says so, because the histogram-mode shift is not measurable on a bounded crop (#1178). On the floor-subtracted data the cosine windows guarantee seamless blending without post-merge pruning. The content-adaptive planner resolves it the same way — one whole-volume level, handed to the density scan and to every box (#1174) — and every box combines it with the shared normalization-range low endpoint, so boxes lying entirely above the pedestal still use the same effective lower bound.
- Without denoising, the intensity scale is resolved whole-volume in the same spirit: `resolve_volume_norm_range` resolves one `(image_min, image_max)` against the raw whole volume and every tile or content box normalizes with it, so the optimiser's absolute criteria — convergence tolerance, seeding and culling thresholds, `amp_max` — mean the same thing across the decomposition. Batch denoising is the exception: an automatically sampled raw range is not forwarded, so each task resolves on the data it fits — denoise-corrected input for an on-the-fly uniform tile, or the denoised store in `preprocess` mode (including content boxes). An explicit configured `norm_range` remains an intentional override. Uniform tiles shift a shared raw range into their floor-subtracted basis and pin `image_min` at 0 (where apodized tile data starts); content boxes keep the raw pair and combine it with the shared floor. A full-range scale carries no ceiling, since it is a bounded *sample* and a brighter voxel is real signal. A region far dimmer than the volume maximum is therefore held to the same absolute tolerance as the rest of the volume, and converges earlier than it would have on its own scale. Because that unclipped scale can put a voxel above 1.0, the auto `amp_max` follows the normalized peak instead of capping at 1.0 (it stays 1.0 exactly whenever the range came from the array itself, so a whole-volume fit is unaffected). Under uniform tiling, the zero-pinned bottom means `norm_percentile > 0` keeps its bright-outlier clipping but not its low-end clipping. Where the tile measurement carries no usable scale — a non-finite top, a top with no positive extent, or a top the applied floor reaches — uniform tiles fall back to their own scale with a printed note rather than dividing by a zero, negative or epsilon range.
- `fit_tile` rejects explicit seed arrays (use int count, float ratio, or None).
- zarr arrays are supported for out-of-core processing -- only one tile is materialized at a time.

### Advanced Usage: Custom Candidates

For specialized use cases requiring custom candidate generation:

```python
from luxar.gsplats.seeds import generate_seeds, seed_from_edges

# Custom seed generation with specific method
custom_seeds = seed_from_edges(
    image,
    n_seeds=1000,  # Custom density
    min_distance=2.0,  # Minimum seed spacing
    edge_threshold_rel=0.1,  # Edge detection threshold
)

# Or use unified entry point
custom_seeds = generate_seeds(image, method="edges")

result = fit_gaussian_splats(image, seeds=custom_seeds)
```

### Controlling Seed Count

The `seeds` parameter supports multiple input types for flexible control:

```python
# Option 1: Auto-generate (default)
result = fit_gaussian_splats(
    image
)  # auto seed budget (see "Auto-Seed Generation" below)

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

- **Edges + grid by default**: `method="auto"` combines edge detection and grid coverage; multiscale decomposition is excluded by default for speed
- **Decomposition scales**: `[1, 2, 4, 8, 16, 32, 64]` span fine details to large structures (used only when decomposition is requested explicitly)
- **Volume-proportional density**: the auto seed budget is `max(100, int(total_voxels ** (1 / ndim) / 2))`, capped at 10,000
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
- **L1**: `mean(where(pred > target, F * |pred - target|, |pred - target|))` where F = `asymmetric_penalty`
- **MSE**: `mean(where(pred > target, F * (pred - target)², (pred - target)²))`
- **Poisson**: Similar penalty applied to Poisson deviance

### Benefits

- **Better Optimization**: Avoids hard-to-correct over-prediction errors
- **Stable Convergence**: Reduces oscillations and improves trajectory
- **Model Alignment**: Reflects additive nature of Gaussian splatting
- **Enhanced Quality**: Enables better late-stage reconstruction improvements

## Loss Function Selection

The package supports three loss functions, each optimized for different data characteristics:

### **L1 (Mean Absolute Error)** - Default
- **Best for**: Microscopy and most image-reconstruction tasks where the held-out (signal-recovery) metric matters
- **Characteristics**: Robust to outliers, preserves edges, encourages sparse residuals
- **Use when**: General-purpose default — verified empirically (Supp. Doc. 5) to reach equal-or-higher held-out PSNR than MSE on every microscopy dataset tested
- **Special synergy**: L1 + asymmetric penalty provides exceptional stability

### **Poisson (Deviance)**
- **Best for**: Count/photon data, fluorescence microscopy, low-light imaging — and any setting where convergence speed is the dominant constraint
- **Characteristics**: Optimal for Poisson noise statistics; uses 1.1-10× fewer iterations than MSE on most datasets
- **Use when**: Working with camera data, microscopy images, or counting processes; or when fitting time matters and the up-to-0.5 dB held-out PSNR gap vs L1 on noisy data is acceptable

### **MSE (Mean Squared Error)**
- **Best for**: Sanity checks, comparison baselines, smooth/dense data where outlier sensitivity is not a concern
- **Characteristics**: Standard textbook loss; PSNR-optimal *at* a critical point of the objective (which finite-iteration Adam fits do not reach)
- **Use when**: You have a specific reason to want MSE — note that L1 typically reaches equal-or-higher PSNR in this fitting regime

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
    l1_amp=0.02,  # Strong amplitude sparsity (2% of lr)
    l1_diag=0.005,  # Mild shape regularization (0.5% of lr)
    lr=0.01,
    n_iters=300,
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
config.step_every = 50  # Run every 50 iterations
config.k_max_residuals = 40  # Max peaks to find
config.nms_radius_vox = 2.0  # Non-maximum suppression radius
config.relocation_percentile = 1.0  # % of weakest splats to relocate
config.max_relocations_per_step = 64  # Cap relocations per step

# Fit with dynamic operations
result = fit_gaussian_splats(
    image, enable_dynamic_ops=True, dynamic_config=config, n_iters=300
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
    scheduler_type="plateau",
)
```

### Technical Details

- **Gradient dilution compensation**: Learning rate automatically scaled based on dimensionality
- **Standard Adam buffers**: Uses PyTorch's built-in momentum and squared gradient tracking
- **Scheduler support**: ReduceLROnPlateau or ExponentialLR for learning rate adaptation

## Advanced Usage

For detailed control and statistics, use the class interface:

```python
from luxar.gsplats import FitParameters, GaussianSplatFitter

# Initialize fitter with specific device and options
fitter = GaussianSplatFitter(
    device="cuda",  # or "mps", "cpu"; None/"auto" selects CUDA → MPS → CPU
    enable_dynamic_ops=True,  # fixed-pool splat relocation during fitting (default)
)

# Fit with detailed statistics
result = fitter.fit(
    FitParameters(
        V=image,
        seeds=8000,
        seed_method="decomposition",
        n_iters=500,
        early_stop_patience=200,  # iterations without improvement before stopping
        sigma_min_diag=[0.5, 0.5],  # Minimum splat size
        sigma_max_diag=[10.0, 10.0],  # Maximum splat size
        seed_kwargs={"num_scales": 3},  # Extra seed-generator options
    )
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
print(f"PSNR (foreground): {metrics['foreground_psnr_db']:.2f} dB "
      f"over {metrics['foreground_fraction'] * 100:.2f}% of voxels")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"MSE:  {metrics['mse']:.6e}")
print(f"Relative L2: {metrics['rel_l2']:.4f}")
print(f"Max absolute error: {metrics['max_abs_error']:.4f}")

# Or compute individual metrics
psnr = compute_psnr(pred_tensor, target_tensor)
ssim = compute_ssim(pred_tensor, target_tensor, window_size=11)
```

### Read the foreground number, not just the global one

On sparse volumes the global PSNR is dominated by background: on a synthetic
99.9%-empty volume (the shape a light-sheet stack has), a fit that discards 90%
of the signal still scores **37 dB globally** while scoring **0.9 dB on the
foreground**. Quote both --
the global figure alone is close to a report on how well the emptiness was
reproduced.

Foreground is `target > otsu(target)`, defined on the *target* so a fit that
hallucinates structure is still scored where the signal actually is. The error
is averaged over foreground voxels only, but `data_range` comes from the whole
volume, matching `calibration.metrics.held_out_psnr_foreground` so the two are
comparable; using the foreground's own (narrower) range would silently inflate
the result. `foreground_fraction` is reported alongside because a PSNR over
0.01% of a volume means something very different from one over 40%.

### Functions

- `compute_psnr(pred, target, data_range=None)` -- Peak Signal-to-Noise Ratio in dB. Returns `float('inf')` when MSE is zero.
- `compute_foreground_psnr(pred, target, data_range=None, threshold=None)` -- PSNR over foreground voxels only. Returns `(psnr_db, threshold, fraction)`; `psnr_db` is `nan` when the foreground is empty.
- `otsu_threshold(target, bins=256)` -- Otsu's threshold, on the input device. Reimplemented rather than delegating to scikit-image (a `demos` extra) so the foreground definition does not depend on which extras are installed; pinned to match `skimage.filters.threshold_otsu` exactly.
- `compute_ssim(pred, target, window_size=11, data_range=None)` -- Structural Similarity Index using nD Gaussian-weighted convolution. Supports 2D, 3D, and higher (averages over 3D sub-volumes for >3D).
- `compute_quality_metrics(pred, target, data_range=None, ssim_window_size=11)` -- Computes all metrics in one call. Returns a dict with keys: `mse`, `psnr_db`, `ssim`, `rel_l2`, `max_abs_error`, `foreground_psnr_db`, `foreground_threshold`, `foreground_fraction`.

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

## Calibration (Blind-Spot Cross-Validation)

The `calibration` module implements the manuscript's Noise2Self model-selection protocol — sweep splat count K, fit each against a 5%-donut-median-filled volume, and pick the K that maximises *held-out* PSNR. As a free byproduct, the module estimates the per-dataset noise floor (Laplacian + Haar HH + background MAD ensemble), giving an absolute PSNR ceiling for the dataset.

Held-out PSNR is fundamentally a *capacity*-selection criterion (across K), not an *iteration*-selection one. Within a single fit at fixed K, bounded splat parameters and L1 amplitude regularisation prevent the held-out trajectory from peak-and-declining over iterations — the patience-based early stop in `fit_gaussian_splats` already handles that regime. The calibration module therefore wraps `fit_gaussian_splats` unchanged at each K rather than augmenting it.

### Quick Example

```python
from pathlib import Path

from luxar.gsplats.calibration import calibrate, build_k_grid

ks = build_k_grid(n_points=10, k_min=1_000, k_max=512_000)  # manuscript-style sweep
result = calibrate(volume, k_grid=ks, fit_kwargs={"device": "cuda"})

print(f"Recommended K* = {result.held_out_peak.k_star:,}")
print(
    f"Curve type     = {result.held_out_peak.type}"
)  # peak | plateau | signal_limited
print(f"Noise floor σ̂ = {result.noise_floor.sigma_hat:.4f}")
print(f"PSNR ceiling   = {result.noise_floor.psnr_max_db:.1f} dB")

# Persist or re-load the sweep curves
result.to_json(Path("cal.json"))
```

### CLI

```bash
luxar gsplat cal volume.tiff cal.json                            # default 10-point sweep
luxar gsplat cal volume.zarr cal.json --n-grid 5 --k-max 128000  # faster
luxar gsplat cal volume.zarr cal.json --k-grid '1000,4000,16000,64000,256000'
luxar gsplat cal volume.tiff cal.json --pdf cal.pdf              # multi-page report
luxar gsplat cal volume.tiff cal.json --pdf cal.pdf --keep-fits fits/  # + slice montage
```

After calibration, re-fit at the recommended budget: `luxar gsplat fit volume.zarr out.zarr --seeds <K*>`.

### Functions

- `cv_mask(shape, fraction=0.05, seed=42)` — deterministic Bernoulli held-out mask.
- `donut_median_fill(V, mask, radius=1)` — replace masked voxels with the median of their *unmasked* `(2r+1)^D` donut neighbours (centre and other held-out voxels excluded), expanding the radius when a clustered mask hides every local donor. The filled volume depends on unmasked voxels only; an all-`True` mask raises `ValueError`. Operates on arrays of any dimensionality.
- `held_out_psnr(V_hat, V_original, mask, data_range=None)` — PSNR at masked positions against the *original* (pre-fill) values.
- `estimate_noise_floor(V) -> NoiseFloor` — ensemble of Laplacian MAD (Immerkaer 1996), Haar HH-subband MAD (Donoho & Johnstone 1994), and background-region MAD; the median across estimators is robust to one outlier on the low side (typical when the dark tail is quantised).
- `build_k_grid(explicit=None, n_points=10, k_min=1_000, k_max=512_000, progression="exp", power=2)` — exponential (geometric/log-spaced) or polynomial K grid; `explicit` takes precedence when given.
- `find_k_star(k_values, held_out_psnr_values) -> HeldOutPeak` — hybrid peak-detection rule from `splat_count_vs_quality §4.2`: returns the argmax when both flanks are ≥ 0.1 dB below; the smallest K within 0.3 dB of the max for a plateau; the largest K when the curve is monotone-rising in range (signal-limited).
- `calibrate(V, k_grid, *, fit_kwargs=None, mask_seed=42, mask_fraction=0.05, donut_radius=1, keep_fits=None, progress_callback=None) -> CalibrationResult` — top-level driver.

### Result Container

`CalibrationResult` carries the full sweep:

- `k_values_requested`, `k_values_effective` — splat counts before and after the post-fit cull.
- `held_out_psnr_db`, `train_psnr_db`, `held_out_mse` — at each K, against the original volume.
- `full_psnr_db`, `full_ssim` — over the whole volume against the original (cross-run comparable).
- `held_out_peak` (`HeldOutPeak`) — recommended `k_star`, curve `type`, `confidence_db`.
- `noise_floor` (`NoiseFloor`) — `sigma_hat`, the three component estimators, `psnr_max_db`.
- `fit_times_seconds`, `splat_paths` (when `keep_fits` is set), `mask_seed`, `mask_fraction`, `donut_radius`, `fit_config`, `volume_shape`, `timestamp`.
- `to_json(path)` / `CalibrationResult.from_json(path)` — round-trip serialisation; non-finite floats become `null`.

### PDF Report (optional)

`luxar.gsplats.calibration_report.render_calibration_report(result, volume, output_path, splat_paths=None)` produces a 3-page matplotlib PDF mirroring the manuscript's per-dataset figures:

1. **Rate-distortion** — PSNR (train / held-out / full) vs K, SSIM vs K, fit-time vs K, train-vs-held-out gap, with K\* annotated and the noise-floor PSNR ceiling overlaid.
2. **Blind-spot cross-validation** — train + held-out PSNR with the overfitting region shaded.
3. **Reconstruction slice montages** — target / K_min / K\* / K_max + per-pixel error map. Requires the per-K fits to have been persisted via `--keep-fits`; otherwise the page degrades to a placeholder.

The report module imports matplotlib lazily so it does not inflate cold-start cost when `--pdf` is not set.

## Level of Detail (LOD) Ladders

The `lod` subpackage builds streaming-ready LOD hierarchies from a fitted `GSplatData`. Two qualitatively different operators live side-by-side:

- **Additive** — same `N` splats, *reordered* so that the prefix sum at any `k ≤ N` splats is the best L² approximation of the full scene. Output: a single multi-LOD `GSplatData` where each level *extends* the previous one. Use this when you want progressive streaming: the viewer can stop loading at any point and the partial reconstruction is principled.
- **Substitutive** — synthesise `M < N` representative splats per coarser level via Gaussian mixture reduction (k-means + cost-increment Lloyd refinement, optionally hierarchical greedy). The recommended `kmeans_lloyd` method seeds k-means with **shape-aware** features (splat centres augmented with covariance / marginal-σ descriptors) so spatially-near but differently-shaped anisotropic splats aren't forced into the same bin; plain `kmeans` stays centres-only as a baseline. Output: a single `GSplatData` with `n_substitutive = levels + 1` substitutive levels (each *replacing* the previous one). Use this when you want fixed-budget coarse mip-levels for view-dependent rendering.

Both operators are pure post-processes on a fitted dataset; fitting (single-pass or progressive) returns one flat container, and an LOD hierarchy is built on demand.

### Quick Example

```python
from luxar.gsplats import (
    fit_gaussian_splats,
    make_additive_lod,
    make_lod_pyramid,
    make_substitutive_lod,
)

# 1. Fit (or use cal upstream to find K*; see "Calibration" section)
data = fit_gaussian_splats(volume, seeds=32_000)

# 2a. Additive: same N splats, reordered into 4 prefix-monotone levels
additive = make_additive_lod(data, n_lods=4, method="greedy")
# additive.additive_prefix(2) → best L² approximation using levels 0+1+2

# 2b. Substitutive: 3 coarser levels with 4x compression each
pyramid = make_substitutive_lod(
    data,
    compression_factor=4,
    levels=3,
    method="kmeans_lloyd",
)
# pyramid.substitutive_levels[0] = original; [3] = coarsest (≈ N / 64 splats)

# 2c. Full 2-D pyramid (substitutive × additive) in one call
matrix = make_lod_pyramid(
    data,
    compression_factor=4,
    levels=3,  # outer (substitutive) axis
    n_additive_lods=4,  # inner (additive) axis
)
```

### CLI

```bash
# One command, one `--recipe` flag (REQUIRED); output is a standalone v3.4 .gsplats.zarr.
# flat / stream — single leaf, optionally with an additive (prefix-sum) ladder
luxar gsplat lod fit.gsplats.zarr stream.gsplats.zarr --recipe stream --n-lods 4
luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe stream --add-method self_energy   # cheap O(N log N)
luxar gsplat lod fit.gsplats.zarr out.gsplats.zarr --recipe stream -m mass -b counts:500,2000,10000
# `radial` = the REVEAL: concentric shells around the bbox centre, so a streaming
# prefix grows outward from the object's middle. Authoring only — no viewer changes.
# Deliberately carries no energy stamps (a reveal must not be 1/e(k)-brightened).
luxar gsplat lod fit.gsplats.zarr reveal.gsplats.zarr --recipe stream -m radial --n-lods 6

# tiles / overview — BSP parts each with an additive ladder (large data);
# overview adds a coarse substitutive cap above the tiled fine branch
luxar gsplat lod fit.gsplats.zarr part.gsplats.zarr --recipe tiles --max-elements 250000
luxar gsplat lod fit.gsplats.zarr ms.gsplats.zarr --recipe overview --compression-factor 8

# levels — synthesised representative levels; v3.4 kind=lod group
luxar gsplat lod fit.gsplats.zarr levels.gsplats.zarr --recipe levels -L 3 -K 4
luxar gsplat lod fit.gsplats.zarr pyramid.gsplats.zarr --recipe levels -K 4 -L 3 --n-lods 4

# Collapse any tree (LOD/partition/nested) into one flat matrix-shaped leaf
luxar gsplat flatten partitioned.gsplats.zarr flat.gsplats.zarr

# Migrate legacy v1.0 / v1.1 / v2.0 / pre-v2.0 substitutive-directory /
# v3.0-v3.1 (pre-v3.2 pixel_size selector attrs) layouts → v3.4
luxar gsplat migrate-format legacy.gsplats.zarr v3.gsplats.zarr
```

### Recommended Workflow

`cal` → `fit` → `lod` is the canonical end-to-end pipeline:

```bash
# 1. Calibrate splat budget K* via blind-spot CV
luxar gsplat cal volume.tiff cal.json

# 2. Fit at the recommended K* (read from cal.json or its stdout)
luxar gsplat fit volume.tiff fitted.gsplats.zarr --seeds <K*>

# 3. Build a streaming LOD ladder (pick a recipe by dataset scale)
luxar gsplat lod fitted.gsplats.zarr scene.gsplats.zarr --recipe stream --n-lods 4
```

See `lod/README.md` for algorithm details (greedy vs self-energy vs mass vs amplitude ordering, breakpoint specs, performance notes, and the mathematical derivations / complexity analyses).

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
- `device`: `"cuda"`, `"mps"`, `"cpu"`, `"auto"`, or `None` (auto-detect)
- `truncate`: Truncation radius in standard deviations (default `DEFAULT_TRUNCATION_RADIUS` = 2.75)
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
result.save("fitted.gsplats.zarr", ordering="hilbert")  # or 'morton', 'none'

# Colors are automatically saved if present; SDR (uint8) vs HDR (geolog_perchannel_u16)
# is auto-detected from the values — there is no explicit color_mode knob.
```

### Loading GSplatData

Load previously saved splats back into a GSplatData object:

```python
from luxar.gsplats.io import load_gsplats

# Load from disk
result = load_gsplats("fitted.gsplats.zarr")

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
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data("fitted", result)
    print(f"Added {gsplats.n_splats} splats with colors={gsplats.has_colors}")
```

**Option 2: Add from saved .gsplats.zarr file**
```python
# First save result
result.save("fitted.gsplats.zarr", ordering="hilbert")

# Later, load into a scene
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_file("loaded", "fitted.gsplats.zarr")
    print(f"Loaded {gsplats.n_splats} splats")
```

**Benefits:**
- Seamless integration with Luxar's scene graph system
- Preserves all splat data (centers, amplitudes, covariance, colors)
- Enables hierarchical organization with transforms
- Works with Luxar viewer for interactive visualization

> **Multi-substitutive input → `kind=lod` group.** When the `GSplatData`
> carries more than one substitutive level (e.g. the output of
> `luxar gsplat lod --recipe levels`), `add_gsplats_from_data`
> (and `add_gsplats_from_file`) route it by default into a `kind=lod`
> scene group — one gsplats child per substitutive level, with
> `coverage_fraction` thresholds derived by SCREEN-OCCUPANCY HALVING (a
> dimensionless, viewport-relative value; coarsest = 0.0, finest = 0.5 for a
> whole-object ladder — a partition-bound one anchors its finest at 1.0 instead,
> see `partitioned_coverage_fractions`; the per-level splat counts set only the
> ladder's length), so the
> viewer view-switches between levels identically on any monitor. In v3.4 a saved `.gsplats.zarr`
> is already a `kind=lod` group on disk; scene embedding grafts that subtree
> directly. No substitutive work is discarded. Pass `lod_group=False` to
> collapse to the finest level, or `lod_group=dict(coverage_fractions=[...])`
> to set the switch thresholds explicitly.

### Color Support

GSplatData now supports optional per-splat RGB colors:

```python
# Create result with colors
result = GSplatData(
    centers=centers,
    amplitudes=amplitudes,
    cholesky_factors=cholesky_factors,
    colors=np.random.rand(n_splats, 3).astype(np.float32),  # Add colors
    stats={},
)

# Colors are preserved during save/load; SDR (uint8 [0-255]) vs HDR (values
# > 1 → geolog_perchannel_u16, decoded back to float32) is auto-detected —
# no color_mode knob.
result.save("colored.gsplats.zarr")

# Load preserves colors
loaded = load_gsplats("colored.gsplats.zarr")
assert loaded.colors is not None
assert loaded.colors.shape == (n_splats, 3)

# Scene integration preserves colors
with LuxarZarrCompiler("scene.luxar.zarr") as compiler:
    scene = compiler.create_scene(dimensions=Dimensions.default_3d())
    gsplats = scene.add_gsplats_from_data("colored", result)
    assert gsplats.has_colors == True
```

## API Reference

### Main Functions

#### `fit_gaussian_splats(V, seeds, **kwargs)`
Main fitting function with automatic optimizations.

**Key Parameters:**
- `V`: Input n-dimensional array to reconstruct
- `seeds`: Initial candidate positions (N, d), int count, float compression ratio, or a `GSplatData` to warm-start from
- `seed_amps_background_relative`: Which amplitude convention a `GSplatData` warm start carries — `False` (default) for `generate_seeds()` output (raw image intensities, background pedestal included), `True` for a previous fit's output (pedestal already removed). Getting it wrong is silent: a fit's output taken as raw has an active `floor` subtracted a second time, so every seed dimmer than the floor starts at 0
- `n_iters`: Maximum iterations (default: 1000)
- `lr`: Learning rate (default: 0.01)
- `loss_type`: "l1" (default), "mse", or "poisson" — see Supp. Doc. 5 for the empirical comparison that motivates the L1 default
- `l1_amp`: L1 regularization on amplitudes (default: 0.1 * lr)
- `l1_diag`: L1 regularization on diagonal elements (default: 0.01 * lr)
- `sigma_min_diag`: Minimum Gaussian size per axis
- `sigma_max_diag`: Maximum Gaussian size per axis
- `early_stop_patience`: Iterations without improvement before stopping (default: 300)
- `device`: PyTorch device (auto-detect if `None` or `"auto"`)

**Returns:** `GSplatData` with fields:
- `centers`: np.ndarray, shape (N, d) - Splat center positions
- `amplitudes`: np.ndarray, shape (N,) - Non-negative amplitudes
- `cholesky_factors`: np.ndarray, shape (N, d*(d+1)//2) - Packed Cholesky factors
- `colors`: Optional[np.ndarray], shape (N, 3) RGB or (N, 4) RGBA — the optional alpha channel is per-splat opacity in [0, 1]
- `stats`: Dict[str, Any] - Optimization statistics

#### `fit_tiled(volume, tile_size=256, overlap=32, **fit_kwargs)`
Tiled fitting for large volumes that exceed GPU memory. Splits the volume into
overlapping tiles with Hann cosine apodization, fits each tile independently,
and concatenates results. The background floor is resolved once against the
whole volume and subtracted from each tile before windowing (floor
subtraction and apodization do not commute) — on the denoised basis when
per-tile denoising is active, so `--denoise` removes the same pedestal here as
it does on the non-tiled path: exactly so within the bounded probe's budget,
approximately above it for a `pNN` spec, and not at all above it for the default
`auto` (the raw-basis level is kept, with a note, because the mode shift is not
measurable on a bounded crop) (#1178); the partition-of-unity property
then eliminates seam artifacts. The normalization range is resolved once
against the whole volume too (`resolve_volume_norm_range`), so every tile is
fitted on one shared intensity scale — see the tiling key-properties list
above.

**Key Parameters:**
- `tile_size`: Tile size per axis (int or tuple). Must satisfy `overlap <= tile_size // 2`.
- `overlap`: Overlap width per axis for cosine blending.
- `voxel_size`: Physical voxel spacing (optional), forwarded to per-tile fitting.
- `output_space`: `"real"` or `"voxel"` coordinate space for output centers.
- `verbose`: Print per-tile progress (default: True).
- `source_shape` / `source_dtype`: Grid and stored element type of the ACQUISITION, when `volume` is already a preprocessed copy of it (a caller that decimated before tiling must declare the grid, or the merged result records the working copy as its source). Taken explicitly rather than through `**fit_kwargs` because they describe the MERGED result and are applied at the merge: forwarded to the tiles, each would claim the whole acquisition as its own crop's source. `source_shape=None` measures `volume`, which is right whenever nothing was preprocessed.
- `**fit_kwargs`: All parameters from `fit_gaussian_splats` (seeds, n_iters, device, etc.)

`seeds` is handed to **every** tile as-is, so an integer here is a *per-tile* count, not a whole-volume budget: N tiles fit ~N × `seeds` splats. This differs from the CLI, where `--seeds` **is** a whole-volume budget that `luxar gsplat fit` divides by the non-empty tile count before calling this function (`cli.gsplat_ops.fitting.fit_utils.split_seeds_across_tiles`). If you are fitting at a K\* from `gsplat cal` (see the calibration sections above), divide it yourself — or pass a float compression ratio, which is scale-free and needs no adjustment.

**Merged quality metrics:** per-region scores describe apodized tile crops or halo-padded content boxes and do not compose into the merged result, so uniform and content-planned merges render the reconstruction against the whole volume and attach `psnr_db` / `ssim` / `mse` / `foreground_*` to the merged result (#1669, #1703, #1733, #1858). The CLI save path publishes that block at the archive root; a library caller that writes a returned partition node directly must pass its `meta["fit_stats"]` through `split_fitting_info` to `write_gsplats_tree`. A partition is rendered as the sum of its surviving parts, exactly as the parts compose, without flattening the splats: uniform tiles retain overlapping Hann-apodized contributions, while content boxes retain only splats centered in disjoint cores. Before scoring, the selected volume is shifted to `clip(V - image_min, 0, None)`, the background-relative basis reconstructed by the merged fit and used by the non-tiled path and `gsplat compare` (#1173). Scoring materializes the whole reference on the host (the fit itself may only read it region by region) and renders on the resolved torch device, so admission checks both resources: the host copies stay under half of free physical RAM and the render/metric peak stays under half of free CUDA memory, each held under a 24 GiB ceiling. Concurrent local batch workers divide the default host allowance across the whole run and the default CUDA allowance across the workers on their card. `LUXAR_TILED_QUALITY_MAX_GB` overrides both checks (`0` declines outright, an unreadable value falls back with a note). Over budget, or on a failure, it says so even when `verbose=False` — an archive that silently carries no PSNR is the failure this exists to end.

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

#### `render_to_volume(gsplat_data, shape, device=None, truncate=2.75, intensity_floor=1e-5, chunk_size=None)`
Render Gaussian splats to a NumPy array.

#### `render_to_volume_tensor(gsplat_data, shape, device=None, truncate=2.75, intensity_floor=1e-5, chunk_size=None)`
Render Gaussian splats to a `torch.Tensor` on the rendering device. Avoids GPU-to-CPU copy for downstream GPU operations.

## Device Support and Performance

The implementation supports multiple PyTorch devices with performance-aware auto-selection:

| Device | Auto-Selected | Performance | Notes |
|--------|---------------|-------------|-------|
| **CUDA** (NVIDIA) | ✓ (when available) | Significant speedup | Best performance, custom kernels |
| **MPS** (Apple Silicon) | ✓ (when available, no CUDA) | GPU-accelerated | Metal backend for 3D volumes; PyTorch renderer otherwise |
| **CPU** | ✓ (fallback) | Baseline | Reliable; fitting warns it can be orders of magnitude slower than GPU |

### Device Selection Logic:
```python
# Auto-detection priority: CUDA → MPS → CPU
fitter = GaussianSplatFitter()  # Uses best available

# Manual device selection:
fitter = GaussianSplatFitter(device="mps")  # Force MPS
fitter = GaussianSplatFitter(device="cuda")  # Force CUDA
fitter = GaussianSplatFitter(device="cpu")  # Force CPU
```

### Apple Silicon Performance Notes:

**MPS auto-selection**: With no CUDA device present, fitting auto-selects MPS on Apple Silicon (`use_metal=True` by default). For 3D volumes the native Metal backend (below) accelerates rendering; other shapes fall back to the PyTorch MPS path.

**PyTorch MPS caveat**: The pure PyTorch MPS backend has significant overhead for `torch.linalg.solve_triangular`, which is why the Metal compute-shader backend is used for 3D volumes to sidestep it.

**MLX Alternative Investigated**: Apple's MLX framework was evaluated as a potential solution. However, MLX's `solve_triangular` operation is currently CPU-only (not GPU-accelerated) and slower than PyTorch's CPU implementation.

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
    sigma_min_diag=[0.5, 0.5, 0.5],  # required: per-axis minimum splat size
    truncate=3.0,
    device="mps",  # Must be MPS
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
├── fit_basis.py                   # Align raw references with background-relative fits
├── fit_progressive_gsplats.py     # Progressive fitting (iterative refinement)
├── fit_tiled_gsplats.py           # Tiled fitting for large volumes (fit_tile, fit_tiled)
├── restride.py                    # Strided stacked-axis re-authoring
├── tiling.py                      # Tile geometry and cosine apodization (TileSpec, cosine_window)
├── gsplat_data.py                 # GSplatData container (construction/repr) + AdditiveSubLOD / SubstitutiveLevel
├── _data/                         # Internal GSplatData domain mixins (see _data/README.md)
│   ├── base.py                    # _GSplatDataOps cross-mixin base + shared color/read-only helpers
│   ├── metrics.py                 # _SplatArrayMixin: computed metrics shared with AdditiveSubLOD
│   ├── lod_views.py               # LOD matrix views, per-level constructors, node-tree bridge
│   ├── composition.py             # concatenate / combine_as_new_dimension / embed / partition builders
│   ├── transforms.py              # transform / translate / center_at_centroid (+ per-level map helpers)
│   ├── intensity.py               # amplitude & color edits (scale/normalize/clamp/reweight/with_colors)
│   ├── filtering.py               # filter / filter_by / slice_by
│   ├── culling.py                 # cull (heuristic + contribution-based)
│   ├── io_adapter.py              # save / load adapter onto luxar.gsplats.io
│   └── render.py                  # render_to_volume adapter onto luxar.gsplats.rendering
├── culling.py                     # Contribution-based splat culling (CullResult, cull_by_contribution)
├── metrics.py                     # Quality metrics (PSNR, SSIM, MSE, relative L2)
├── calibration.py                 # Blind-spot CV calibration (cv_mask, donut_median_fill,
│                                  #   estimate_noise_floor, build_k_grid, find_k_star, calibrate)
├── calibration_report.py          # Optional matplotlib PDF report for `luxar gsplat cal --pdf`
├── gpu_profile.py                 # GPU benchmark profile management
├── spatial_hash.py                # Online and batched nD proximity-query grids
│
├── lod/                            # Level-of-Detail post-processing (additive + substitutive)
│   ├── additive.py                 # compute_additive_order, make_additive_lod
│   ├── substitutive.py             # make_substitutive_lod orchestrator (+ _reduce_one_level, _pack_level)
│   ├── pyramid.py                  # make_lod_pyramid (substitutive × additive)
│   ├── _kernels.py                 # Shared L² / Gram / merge kernels
│   └── _substitutive/              # substitutive algorithms: warm_start / kmeans_lloyd / greedy
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
│   ├── config.py                  # Configuration dataclasses (FitParameters, FitConfig, etc.)
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
│   ├── inspect_gsplats.py         # Inspect and summarize .gsplats.zarr files
│   └── migrate.py                 # Migrate legacy v1.0 / v1.1 / v2.0 / substitutive-dir / v3.0-v3.1 (pre-v3.2 pixel_size attrs) layouts → v3.4
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
│   ├── trils.py                   # Triangular matrix packing/unpacking
│   └── device.py                  # PyTorch device-selection helpers (CUDA > MPS > CPU)
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
    ├── test_restride.py            # Stacked-axis restriding tests
    ├── test_batch.py               # Batch fitting tests
    └── ...                         # Additional test files
```

### Spatial Hash Queries

`spatial_hash.py` provides an online `SpatialHashGrid` for incremental
deduplication and a batched `BatchedSpatialHashGrid` for repeated k-NN and
radius queries. The module docstring records the query guarantees and backend
fallback contract.

```python
import numpy as np

from luxar.gsplats.spatial_hash import BatchedSpatialHashGrid

points = np.random.randn(10_000, 3).astype(np.float32)
queries = np.random.randn(1_000, 3).astype(np.float32)

grid = BatchedSpatialHashGrid.from_points(points, cell_size=0.5, device="auto")
distances, indices = grid.query_knn(queries, k=8)
neighbours = grid.query_radius(queries, radius=0.4)
```

### Modular Fitting Architecture

The fitting pipeline uses a modular architecture with focused, maintainable modules:

- **fit_gsplats.py**: Clean orchestration method that coordinates the pipeline
- **fitting/ modules**: Each handles a specific aspect of the fitting process
  - Individual components are independently testable
  - Clear separation of concerns across six pipeline stages
  - Type-safe configuration via dataclasses (FitParameters, FitConfig, etc.)
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
- Dynamic operations (fixed-pool splat relocation)
- Quality metrics (PSNR, SSIM, MSE) for 2D and 3D data
- Tiled fitting with cosine apodization and tile merging
- Device compatibility (CPU, CUDA, MPS)
- Edge cases and error handling
- No interactive windows during tests (napari properly mocked)

## Performance Tips

1. **Device Selection**: Use GPU when available (significant speedup)
2. **Early Stopping**: Keep enabled for substantial iteration reduction
3. **Candidate Tuning**: Balance quality vs speed with `peaks_per_scale`
4. **Loss Function**: L1 is the default; switch to Poisson when convergence speed matters
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
- Note that early stopping is on by default (controlled by `early_stop_patience`, default 300)

**Out of Memory**
- Reduce `peaks_per_scale` in candidate generation
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
- Truncated rendering (2.75σ default radius when fitting; 3σ for the standalone model/renderer)
- Batched operations for GPU parallelism
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

BSD 3-Clause License - See the repository `LICENSE` file for details
