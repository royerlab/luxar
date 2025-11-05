# Gaussian Splatting Package for Luxar

High-performance n-dimensional oriented Gaussian splatting with automatic optimization for efficient image and volume reconstruction.

## Overview

This package implements a sophisticated Gaussian splatting system that fits collections of oriented Gaussian functions (splats) to reconstruct n-dimensional data. Each splat is a multivariate Gaussian characterized by position, full covariance matrix, and amplitude. The implementation features automatic convergence detection, adaptive learning, and GPU acceleration.

## Key Features

- **N-dimensional Support**: Works seamlessly with 2D images, 3D volumes, and 4D+ hypercubes (validated to 4D) with automatic gradient dilution compensation
- **Efficient Cholesky Parameterization**: Covariance matrices via Cholesky decomposition with batched triangular solve
- **Device Optimized**: CUDA acceleration with automatic device selection (CPU preferred on Apple Silicon)
- **Oriented Gaussians**: Full covariance matrices via Cholesky decomposition for arbitrary orientations
- **Convergence-Driven Dynamic Operations**: Adaptive splat management based on convergence criteria with seeding and pruning
- **Asymmetric Loss Functions**: 10x penalty for over-prediction addresses additive model constraints (MSE and Poisson)
- **Per-Splat Optimization**: Individual learning rates and momentum preservation for each Gaussian splat
- **Adaptive Thresholds**: Amplitude validation scales with local residual magnitude to prevent optimization plateaus
- **Automatic Optimization**: Early stopping saves 20-60% of iterations without quality loss
- **GPU Acceleration**: CUDA support with batched operations, improved MPS compatibility
- **Robust Initialization**: Multiscale candidate detection with DoG and peak finding
- **Memory Efficient**: Truncated rendering, optional mixed precision, pre-allocated buffers

## How It Works

### 1. Candidate Generation
The system starts by finding initial splat positions using multiscale analysis:
- **Gaussian filtering** at multiple scales to detect blob-like structures
- **Difference of Gaussians (DoG)** for edge and boundary detection  
- **Intensity-weighted grid sampling** for spatial coverage
- **Spatial deduplication** to remove redundant candidates

### 2. Model Architecture
Each Gaussian splat is parameterized using covariance matrix representation:

**Covariance Matrix Parameterization**
- **Centers (μ)**: Sigmoid-bounded to stay within image domain
- **Covariance (Σ)**: Cholesky decomposition `Σ = L @ L^T` ensures positive definiteness
- **Amplitudes (a)**: Softplus activation for non-negativity
- **Sharpness (s)**: Exponential mapping `s = 2 * exp(s')` controls edge falloff (default s=2 for standard Gaussian)

Mathematical form: `f(x) = a * exp(-0.5 * ||y||^s)` where `y = Σ^(-1/2) @ (x-μ)` and s controls edge sharpness

The implementation avoids explicit matrix inversion by solving the triangular system `L @ y = (x-μ)` 
and computing the quadratic form as `||y||²`.

### 3. Optimization Process
The fitting uses PyTorch with advanced optimization strategies:
- **Per-Splat Optimization**: Individual learning rates and momentum for each Gaussian
- **Convergence-Based Dynamic Operations**: Residual-driven splat management with seeding, splitting, and pruning
- **Asymmetric Loss Functions**: 10x penalty for over-prediction addresses additive model constraints
- **Adaptive Amplitude Thresholds**: Scale with local residual magnitude to prevent plateaus
- **Early Stopping**: Maximum absolute error convergence criterion
- **Adaptive Learning Rate**: ReduceLROnPlateau scheduler with per-splat rates
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
2. **Convergence Detection**: Automatically stops when loss plateaus (saves 20-60% iterations)
3. **Adaptive Learning**: Reduces learning rate on plateaus for better convergence  
4. **Device-Aware Selection**: Automatic selection of best available device (CUDA > CPU > MPS)
5. **Cached Computations**: Reuses grids and strides for repeated operations
6. **Optional Enhancements**: 
   - Model compilation with `torch.compile` (PyTorch 2.0+, CUDA only)
   - Mixed precision training (FP16 on CUDA)
   - Pre-allocated buffers for memory efficiency

## Quick Start

### Simple One-Step API

The easiest way to use Gaussian splatting with intelligent defaults:

```python
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

# 1. One-step fitting with intelligent defaults
params, amps, stats = fit_gaussian_splats(
    image,
    # seeds auto-generated with volume-proportional scaling
    # norm_percentile=0.0 by default (full range normalization)
    n_iters=300,                          # Maximum iterations
    lr=0.01,                              # Stable learning rate
    asymmetric_penalty=10.0,              # 10x penalty for over-prediction (default)
    loss_type="l1",                       # "mse", "poisson", or "l1" for robust features
    # max_abs_error auto-set to 0.01 (1% of normalized range)
    # l1_amp auto-set to 0.1 * lr for proportional amplitude regularization
    # l1_diag auto-set to 0.01 * lr for mild diagonal regularization
    # l1_sharpness auto-set to 0.01 * lr for standard Gaussian regularization
    # enable_dynamic_ops=True by default for optimal results
)

# 2. Render reconstruction
# params includes all parameters (centers, Cholesky, sharpness) - auto-extracted
reconstruction = render_gaussians_numpy(
    image.shape, params, amps, truncate=3.0
)

```

**Note**: `fit_gaussian_splats()` returns params with shape `(N, d + d*(d+1)//2 + 1)` where:
- First `d` columns: centers
- Next `d*(d+1)//2` columns: packed Cholesky factors
- Last column: per-splat sharpness values

`render_gaussians_numpy()` automatically extracts all parameters from params, treating sharpness the same as centers and Cholesky.

### Multi-Scale Fitting for Large Datasets

For large images and volumes, multi-scale fitting provides 10-100× speedup by leveraging multi-scale decomposition:

```python
from luxar.gsplats import fit_multiscale_gaussian_splats

# Multi-scale fitting with intelligent defaults
params, amps, stats = fit_multiscale_gaussian_splats(
    large_volume,                      # 3D volume or 2D image
    scales=[1, 2, 4, 8],              # Scale factors (default)
    base_init_sigma=1.5,              # Base sigma (scaled per level)
    n_iters_decomp=1000,              # Decomposition iterations
    n_iters_per_scale=500,            # Iterations per scale
    loss_type="l1",                    # Loss function
    max_abs_error=0.1,                # Convergence threshold
    verbose=True,                      # Show progress
)

# Access speedup statistics
print(f"Computational speedup: {stats['computational_speedup']:.1f}×")
print(f"Splats per scale: {stats['n_splats_per_scale']}")
print(f"Total time: {stats['total_time_seconds']:.2f}s")
```

**How it works:**
1. **Decompose** image into multiple scales (coarse to fine)
2. **Fit independently** on each scale (fewer voxels = faster)
3. **Scale parameters** back to full resolution
4. **Combine** all splats from all scales

**Key benefits:**
- **Massive speedup**: 8× scale in 3D = 512× fewer voxels per scale
- **Hierarchical**: Coarse scales capture large structures, fine scales capture details
- **Quality**: Similar or better reconstruction than single-scale
- **Scalable**: Enables fitting on very large volumes (1024³+)

**When to use:**
- Large 3D/4D datasets where single-scale is slow
- Data with hierarchical structure (coarse + fine features)
- Need explicit scale separation
- Want 10-100× speedup without quality loss

**Visualization options:**
```python
# Enable per-scale visualization and decomposition movie
params, amps, stats = fit_multiscale_gaussian_splats(
    image,
    scales=[1, 2, 4, 8],
    visualize_per_scale=True,  # Show splat locations and reconstructions per scale
    napari_movie=True,          # Record decomposition convergence animation
    movie_every=50,             # Record every 50 iterations
)

# Access per-scale visualization data
for vis in stats['per_scale_visualizations']:
    scale = vis['scale_factor']
    centers = vis['centers']           # Splat locations at full resolution
    recon = vis['reconstruction']      # Full resolution reconstruction
    residual = vis['residual']         # Full resolution error map
    print(f"Scale {scale}×: {vis['n_splats']} splats, MSE={vis['error_mse']:.6e}")
```

### Advanced Usage: Custom Candidates

For specialized use cases requiring custom candidate generation:

```python
from luxar.gsplats.candidates import find_candidates_overcomplete_nd

# Custom candidate generation with specific parameters
custom_candidates = find_candidates_overcomplete_nd(
    image,
    scales=(1.0, 2.0, 4.0),      # Custom scales
    peaks_per_scale=1000,        # Custom density
    percentile_thresh=95,        # Custom selectivity
)

params_custom, amps_custom, _ = fit_gaussian_splats(
    image, seeds=custom_candidates
)
```

### Auto-Candidate Generation Features

- **Universal scales**: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) detect features from fine details to large structures
- **Volume-proportional density**: Automatically scales candidate count with image size (~0.2% of pixels)
- **Inclusive detection**: 70% percentile threshold for comprehensive feature coverage
- **Dimension-agnostic**: Works seamlessly with 2D images, 3D volumes, and 4D+ hypercubes (validated to 4D)

### Outlier-Robust Normalization

Control how outliers and noise are handled during normalization:

```python
# Full range (default) - maximum dynamic range
params, amps, _ = fit_gaussian_splats(image, norm_percentile=0.0)

# Robust to mild outliers - ignore bottom/top 1%
params, amps, _ = fit_gaussian_splats(image, norm_percentile=1.0)

# Very robust to noise - ignore bottom/top 5%
params, amps, _ = fit_gaussian_splats(image, norm_percentile=5.0)
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
Generated 212 candidate centers

[   1/1000] loss=0.078185  relL2=0.7969  maxAbsErr=1.1798  N=212
    ★ New best state: iteration 1, max_abs_error=0.978093
[  10/1000] loss=0.056279  relL2=0.7099  maxAbsErr=0.99667  N=212
    ★ New best state: iteration 8, max_abs_error=0.045821
...
✓ CONVERGENCE ACHIEVED at iteration 127
  Max absolute error: 0.009854 < threshold: 0.010000

★ Restored best state from iteration 125 (improved from 0.010123 to 0.009854)
Rescaled amplitudes to original intensity range (factor: 0.9075)
```

### Quality Guarantee Features

- **Best state tracking**: Always returns the splat configuration with lowest max absolute error
- **Smart logging**: Reports significant improvements and early progress
- **State restoration**: Uses best quality achieved, not potentially suboptimal final state
- **Non-monotonic protection**: Handles optimization fluctuations and dynamic operations gracefully

## Asymmetric Loss Functions

The implementation includes asymmetric loss functions that address the fundamental constraints of additive Gaussian models:

### Why Asymmetric Loss?

**The Problem**: Gaussian splatting uses non-negative additive models: `prediction = Σ(positive_gaussians)`
- **Under-prediction** (`pred < target`): Easy to fix by adding more Gaussians
- **Over-prediction** (`pred > target`): Hard to fix, requires reducing/moving existing splats

**The Solution**: Asymmetric loss with 10x penalty for over-prediction
- **MSE**: `mean(where(pred > target, 10 * (pred - target)², (pred - target)²))`
- **Poisson**: Similar 10x penalty applied to Poisson deviance

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
params, amps, stats = fit_gaussian_splats(
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

## Splat Proliferation Prevention

The system prevents runaway splat multiplication through sophisticated parameter-type-specific learning rates:

### **Problem**: Splat Migration and Proliferation
- Newly seeded splats migrate away from problematic regions during optimization
- Regions become uncovered again, triggering more seeding
- Results in splat proliferation without quality improvement

### **Solution**: Parameter-Type-Specific Learning Rates
```python
# Hard-coded in PerSplatAdam optimizer:
Position parameters (μ):     ×0.1  # Slow movement, keeps splats spatially stable
Variance parameters (L):     ×1.0  # Normal adaptation for shape and orientation
Amplitude parameters (a):    ×2.0  # Fast intensity matching for better convergence
```

### **Benefits**
- **Spatial stability**: Splats stay near seeded locations (×0.1 position updates)
- **Shape optimization**: Normal covariance evolution for local structure fitting
- **Fast convergence**: Accelerated amplitude adaptation reduces estimation failures
- **Proliferation prevention**: Eliminates runaway seeding cycles

## Gradient Dilution Compensation

The system automatically handles nD optimization challenges through intelligent learning rate scaling:

### **Problem**: Multi-Factor Gradient Dilution in Higher Dimensions
- **Parameter growth**: 2D (5 params), 3D (10 params), 4D (15 params) per splat
- **Signal dilution**: Same loss gradient distributed across more parameters
- **Spatial complexity**: 4D optimization landscape much more challenging than 2D/3D
- **Volume effects**: Higher dimensional spaces require more aggressive optimization
- **Optimization difficulty**: Higher dimensions converge much slower without proper compensation

### **Solution**: Enhanced Dimensional and Parameter Complexity Scaling
```python
# Enhanced gradient dilution compensation
dimensional_complexity = d ** 0.8  # Moderate spatial complexity scaling
parameter_complexity = params_current / 5  # Parameter dilution factor

gradient_dilution_factor = dimensional_complexity × parameter_complexity
effective_lr = base_lr × gradient_dilution_factor
# 2D: lr × 1.0, 3D: lr × 5.2, 4D: lr × 12.0
```

### **Benefits**
- **Dimensional fairness**: Each dimension optimizes with equivalent effectiveness
- **Automatic scaling**: No manual learning rate tuning needed for different dimensions
- **Mathematical foundation**: Compensates for fundamental gradient dilution effect
- **Validated performance**: Enables excellent 4D convergence with standard lr=0.01 input

## Dynamic Operations

The implementation features convergence-driven dynamic operations that automatically adjust splat topology based on reconstruction quality and convergence criteria:

### Convergence-Based Operations

- **Seeding**: Add new splats where residual exceeds convergence thresholds (ultra-simple: amplitude = residual value, isotropic shape)
- **Pruning**: Remove ineffective splats using importance-based analysis and quality validation
- **Adaptive Thresholds**: Validation scales with local residual magnitude

### Key Improvements

- **Convergence Alignment**: Operations directly serve optimization goals
- **Plateau Prevention**: Adaptive thresholds eliminate optimization plateaus
- **Adaptive Learning Rates**: Boost learning rates for splats covering problematic regions to "unfreeze" adaptation
- **Parameter-Type-Specific Learning Rates**: Different rates for position (×0.1), variance (×1.0), and amplitude (×2.0) parameters
- **Stable Evolution**: Slow position updates prevent splat migration while fast amplitude updates improve convergence
- **Quality Focus**: Continuous improvement throughout optimization

### Enabling Dynamic Operations

```python
from luxar.gsplats.dynamic_ops import DynamicOpsConfig

# Create configuration for 2D data
config = DynamicOpsConfig()
config.step_every = 10          # Run every 10 iterations
config.max_add_per_step = 40    # Allow moderate seeding
config.max_merges_per_step = 30 # Allow moderate merging
config.residual_quantile = 0.94 # Selective seeding threshold
config.merge_dist_vox = 2.0     # Merge distance in voxels
config.amp_abs_min = 1e-4       # Prune threshold
config.do_prune = True          # Enable all operations
config.do_seed = True
config.do_merge = True
config.do_split = True

# Fit with dynamic operations
params, amps, stats = fit_gaussian_splats(
    image,
    seeds=candidates,
    enable_dynamic_ops=True,
    dynamic_config=config,
    n_iters=300
)
```

### Configuration Guidelines

**Key Parameters:**
- `step_every`: Operation frequency (2D: 8-15, 3D: 12-20)
- `residual_quantile`: Seeding selectivity (0.85-0.98, higher = more selective)
- `merge_dist_vox`: Merge distance threshold (2D: 1.5-2.5, 3D: 1.0-2.0)
- `amp_abs_min`: Pruning threshold (1e-5 to 1e-3)

**Performance vs Quality:**
- More frequent operations: Better quality, slower
- Higher seeding threshold: Fewer but better-placed splats
- Aggressive merging/pruning: Faster inference, potential quality loss

## Per-Splat Optimization

The implementation includes a specialized per-splat Adam optimizer that maintains individual learning rates and momentum states for each Gaussian splat, enabling seamless integration with dynamic operations.

### Key Benefits

**Momentum Preservation**: Unlike standard optimizers that lose ALL momentum during dynamic operations, the per-splat optimizer preserves momentum for unchanged splats, preventing optimization disruption.

**Individual Learning Rates**: Each splat can have its own learning rate, allowing new splats to learn faster initially while preserving stable optimization for existing splats.

**Dynamic Integration**: Seamlessly handles:
- Adding new splats (seeding/splitting) with fresh optimizer states
- Removing splats (pruning) without affecting others
- Merging splats by combining their optimizer states

### Usage

The per-splat optimizer is automatically used when dynamic operations are enabled:

```python
# Per-splat optimizer is used automatically with dynamic operations
params, amps, stats = fit_gaussian_splats(
    image,
    seeds=candidates,
    enable_dynamic_ops=True,  # Automatically uses per-splat optimizer
    n_iters=300
)
```

### Technical Details

- **State Management**: Maintains exp_avg (momentum) and exp_avg_sq (squared gradient) per parameter per splat
- **Bias Correction**: Proper Adam bias correction with per-splat step counting
- **AMSGrad Support**: Optional AMSGrad variant for improved convergence
- **Memory Efficient**: Only allocates states for active splats

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
params, amps, stats = fitter.fit(
    image,
    seeds=candidates,
    n_iters=500,
    early_stopping=True,
    early_stop_patience=20,
    sigma_min_diag=[0.5, 0.5],  # Minimum splat size
    sigma_max_diag=[10.0, 10.0], # Maximum splat size
)

# Access optimization statistics
print(f"Time: {stats['time_seconds']:.2f}s")
print(f"Iterations: {stats['iterations']}/{500}")
print(f"Converged: {stats['converged']}")
print(f"Final loss: {stats['final_loss']:.5g}")
```

## API Reference

### Main Functions

#### `fit_gaussian_splats(V, seeds, **kwargs)`
Main fitting function with automatic optimizations.

**Key Parameters:**
- `V`: Input n-dimensional array to reconstruct
- `seeds`: Initial candidate positions (N, d)
- `n_iters`: Maximum iterations (default: 300)
- `lr`: Learning rate (default: 0.2)
- `loss_type`: "mse" or "poisson" (default: "mse")
- `l1_amp`: L1 regularization on amplitudes (default: 0.1 * lr)
- `l1_diag`: L1 regularization on diagonal elements (default: 0.01 * lr)
- `sigma_min_diag`: Minimum Gaussian size per axis
- `sigma_max_diag`: Maximum Gaussian size per axis
- `early_stopping`: Enable convergence detection (default: True)
- `device`: PyTorch device (auto-detect if None)

**Returns:**
- `params`: (N, d + d*(d+1)/2) array of [centers, packed_cholesky]
- `amps`: (N,) array of amplitudes

#### `find_candidates_overcomplete_nd(V, **kwargs)`
Generate initial splat positions using multiscale detection.

**Key Parameters:**
- `scales`: Gaussian filter scales (default: (0.7, 1.0, 1.4, 2.0, 2.8, 4.0))
- `peaks_per_scale`: Maximum peaks per scale (default: 1000)
- `percentile_thresh`: Intensity threshold percentile (default: 70.0)
- `min_dist`: Minimum distance between candidates (default: 2.0)

## Device Support and Performance

The implementation supports multiple PyTorch devices with performance-aware auto-selection:

| Device | Auto-Selected | Performance | Notes |
|--------|---------------|-------------|-------|
| **CPU** | ✓ (default on Mac) | Baseline | Reliable, well-optimized |
| **MPS** (Apple Silicon) | Manual only | ~0.1x (slower) | Compatible but has overhead issues |
| **CUDA** | ✓ (when available) | 5-10x faster | Best performance, full features |

### Device Selection Logic:
```python
# Auto-detection priority: CUDA → CPU (skips MPS due to performance)
fitter = GaussianSplatFitter()  # Uses best available

# Manual device selection:
fitter = GaussianSplatFitter(device="mps")    # Force MPS
fitter = GaussianSplatFitter(device="cuda")   # Force CUDA
fitter = GaussianSplatFitter(device="cpu")    # Force CPU
```

### Apple Silicon Performance Notes:

**PyTorch MPS Issues**: The PyTorch MPS backend (as of 2024-2025) has significant overhead for `torch.linalg.solve_triangular` operations, making CPU ~10x faster than MPS on M1/M2/M3/M4 chips.

**MLX Alternative Investigated**: Apple's MLX framework was evaluated as a potential solution. However, MLX's `solve_triangular` operation is currently CPU-only (not GPU-accelerated) and ~1.7x slower than PyTorch's CPU implementation as of MLX v0.29.0.

**Current Optimal Strategy**: Auto-select CPU on Apple Silicon, which provides the best performance available. This will automatically benefit from future improvements in either PyTorch MPS or MLX GPU acceleration.

## Package Structure

```
gsplats/
├── fit_gsplats.py              # Main fitting interface (refactored to use modular pipeline)
├── fit_multiscale_gsplats.py   # Multi-scale fitting for large datasets (NEW)
├── candidates.py               # Multiscale candidate detection
├── dynamic_ops.py              # Adaptive topology operations (prune, seed, merge, split)
├── fitting/                    # Modular fitting pipeline (NEW - refactored components)
│   ├── __init__.py            # Exports for main interface
│   ├── config.py              # Configuration dataclasses (FitConfig, PreprocessedData, etc.)
│   ├── validation.py          # Input validation and parameter checking
│   ├── preprocessing.py       # Data normalization and candidate generation
│   ├── initialization.py      # Model and optimizer initialization
│   ├── losses.py              # Loss function creation (MSE, Poisson, L1)
│   ├── optimization.py        # Main optimization loop and convergence logic
│   ├── results.py             # Result finalization and statistics
│   └── visualization.py       # Movie recording and compression analysis
├── optim/                      # Per-splat optimization algorithms
│   ├── per_splat_adam.py      # Per-splat Adam optimizer with momentum preservation
│   └── per_splat_scheduler.py # Individual learning rate scheduling
├── models/
│   ├── gsplats/
│   │   ├── gsplat_model.py    # PyTorch model definition
│   │   ├── gsplats_batched_render.py  # GPU batched renderer
│   │   └── gsplats_render.py  # CPU fallback renderer
│   └── utils/
│       ├── inverse_softplus.py # Numerical utilities
│       └── lt_solver.py       # Triangular system solver
├── utils/
│   └── trils.py               # Triangular matrix packing/unpacking
├── demos/                      # Interactive demonstrations
│   ├── demo_multiscale_fitting.py # Multi-scale vs single-scale comparison (NEW)
│   ├── demo_performance.py      # Performance showcase with dynamic ops
│   ├── demo_splats_fit.py       # Main fitting demo with simplified API
│   ├── demo_splats_2d_napari.py # Interactive 2D compression analysis
│   ├── demo_splats_3d_napari.py # Interactive 3D volumetric visualization
│   ├── demo_splats_4d_napari.py # 4D hypercube validation (nD algorithms)
│   └── demo_splats_mitosis.py   # Biological data with L1 loss
└── tests/
    ├── test_multiscale_fitting.py   # Multi-scale fitting tests (NEW)
    └── test_gsplats_integration.py  # Comprehensive tests
```

### Refactored Architecture

The fitting pipeline has been refactored from a monolithic 480+ line method into focused, maintainable modules:

- **fit_gsplats.py**: Now contains a clean orchestration method that coordinates the pipeline
- **fitting/ modules**: Each handles a specific aspect of the fitting process
  - Improved testability with individual components
  - Better separation of concerns
  - Enhanced maintainability and readability
  - Type-safe configuration objects

## Running Demos

**Standard execution (with napari visualization):**
```bash
# Multi-scale fitting comparison (NEW)
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_multiscale_fitting.py

# Main demos with simplified one-step API
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_performance.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_fit.py

# Interactive compression analysis
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_2d_napari.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_3d_napari.py
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_4d_napari.py  # 4D validation

# Biological data demonstration
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_mitosis.py
```

**Headless execution (for testing/CI):**
```bash
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_performance.py --no-napari
hatch run python packages/luxar/src/luxar/gsplats/demos/demo_splats_4d_napari.py --no-napari  # 4D validation
```

### 4D Hypercube Validation

The `demo_splats_4d_napari.py` demonstrates complete nD algorithm validation:

**4D Test Results:**
- **Hypercube data**: (8×64×64×64) = 262K hypervoxels with synthetic 4D Gaussian blobs
- **Auto-candidate generation**: Volume-proportional scaling (262K → 524 peaks/scale, perfect 0.2% density)
- **4D splat fitting**: Successfully generates 787 4D splats (15 parameters each)
- **Outstanding compression**: 95.5% bit reduction (8.3M → 377K bits)
- **Best state tracking**: Quality guarantee with restoration from optimal iteration
- **Interactive 4D visualization**: Full napari navigation with dimension sliders
- **✅ nD algorithms validated**: All features working correctly in 4D space

## Testing

The gsplats package has comprehensive test coverage with 325 tests organized into unit tests (per subpackage) and integration tests:

```bash
# Run all gsplats tests
hatch run pytest packages/luxar/src/luxar/gsplats/ -v

# Run integration tests only
hatch run pytest packages/luxar/src/luxar/gsplats/tests/ -v

# Run fitting pipeline unit tests
hatch run pytest packages/luxar/src/luxar/gsplats/fitting/tests/ -v
```

**Test Organization:**
- `fitting/tests/` - 84 tests for modular fitting pipeline (100% module coverage)
- `optim/tests/` - 17 tests for per-splat optimizer
- `models/*/tests/` - 53 tests for model and utility functions
- `multiscale/tests/` - 30 tests for multiscale decomposition
- `tests/` - 131 integration tests for complete pipelines
  - Includes 11 new tests for multi-scale fitting

**Coverage:**
- Unit tests for all pipeline components (validation, preprocessing, losses, optimization, etc.)
- 2D/3D/nD reconstruction pipelines
- Loss functions (MSE, Poisson, L1) with asymmetric penalties
- Dynamic operations (seeding, pruning, merging, splitting)
- Device compatibility (CPU, CUDA, MPS)
- Edge cases and error handling
- No interactive windows during tests (napari properly mocked)

## Performance Tips

1. **Device Selection**: Use GPU when available (5-10x speedup)
2. **Early Stopping**: Keep enabled for 20-60% iteration reduction
3. **Candidate Tuning**: Balance quality vs speed with `peaks_per_scale`
4. **Loss Function**: Use Poisson for photon/count data, MSE for general
5. **Regularization**: Add L1 penalty for sparser, faster solutions
6. **Compilation**: Enable on CUDA for additional 20-30% speedup

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

**Poor Reconstruction Quality**
- Increase `n_iters` or disable early stopping for maximum quality
- Add more candidates (increase `peaks_per_scale`)
- Adjust learning rate (`lr`)
- Try different `loss_type` for your data

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