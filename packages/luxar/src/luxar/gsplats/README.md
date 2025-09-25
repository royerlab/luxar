# Gaussian Splatting Package for Luxar

High-performance n-dimensional oriented Gaussian splatting with automatic optimization for efficient image and volume reconstruction.

## Overview

This package implements a sophisticated Gaussian splatting system that fits collections of oriented Gaussian functions (splats) to reconstruct n-dimensional data. Each splat is a multivariate Gaussian characterized by position, full covariance matrix, and amplitude. The implementation features automatic convergence detection, adaptive learning, and GPU acceleration.

## Key Features

- **N-dimensional Support**: Works seamlessly with 2D images, 3D volumes, and higher dimensions
- **Efficient Cholesky Parameterization**: Covariance matrices via Cholesky decomposition with batched triangular solve
- **Device Optimized**: CUDA acceleration with automatic device selection (CPU preferred on Apple Silicon)
- **Oriented Gaussians**: Full covariance matrices via Cholesky decomposition for arbitrary orientations
- **Convergence-Driven Dynamic Operations**: Adaptive splat management based on convergence criteria with seeding, splitting, and pruning
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

Mathematical form: `f(x) = a * exp(-0.5 * (x-μ)^T @ Σ^{-1} @ (x-μ))`

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
- **Regularization**: Optional L1 penalty on amplitudes for sparsity

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
    # centers_overcomplete auto-generated with volume-proportional scaling
    # norm_percentile=0.0 by default (full range normalization)
    n_iters=300,                          # Maximum iterations
    lr=0.01,                              # Stable learning rate
    asymmetric_penalty=10.0,              # 10x penalty for over-prediction (default)
    loss_type="l1",                       # "mse", "poisson", or "l1" for robust features
    # max_abs_error auto-set to 0.01 (1% of normalized range)
    # enable_dynamic_ops=True by default for optimal results
)

# 2. Render reconstruction
reconstruction = render_gaussians_numpy(
    image.shape, params, amps, truncate=3.0
)

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
    image, centers_overcomplete=custom_candidates
)
```

### Auto-Candidate Generation Features

- **Universal scales**: (0.5, 1.0, 2.0, 4.0, 8.0, 16.0) detect features from fine details to large structures
- **Volume-proportional density**: Automatically scales candidate count with image size (~0.2% of pixels)
- **Inclusive detection**: 70% percentile threshold for comprehensive feature coverage
- **Dimension-agnostic**: Works seamlessly with 2D images, 3D volumes, and higher dimensions

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

## Dynamic Operations

The implementation features convergence-driven dynamic operations that automatically adjust splat topology based on reconstruction quality and convergence criteria:

### Convergence-Based Operations

- **Seeding**: Add new splats where residual exceeds convergence thresholds
- **Splitting**: Divide large elongated splats with poor geometric properties
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
    centers_overcomplete=candidates,
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
    centers_overcomplete=candidates,
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
    centers_overcomplete=candidates,
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

#### `fit_gaussian_splats(V, centers_overcomplete, **kwargs)`
Main fitting function with automatic optimizations.

**Key Parameters:**
- `V`: Input n-dimensional array to reconstruct
- `centers_overcomplete`: Initial candidate positions (N, d)
- `n_iters`: Maximum iterations (default: 300)
- `lr`: Learning rate (default: 0.2)
- `loss_type`: "mse" or "poisson" (default: "mse")
- `l1_amp`: L1 regularization strength (default: 0.0)
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
├── fit_gsplats.py              # Main fitting implementation with optimizations
├── candidates.py               # Multiscale candidate detection
├── dynamic_ops.py              # Adaptive topology operations (prune, seed, merge, split)
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
├── demo/
│   ├── demo_performance.py    # Performance showcase with dynamic ops
│   ├── demo_splats_fit.py     # Main fitting demo with dynamic ops
│   ├── demo_splats_mitosis.py # Biological data demo with dynamic ops
│   └── demo_splats_3d_*.py    # 3D examples with dynamic ops
└── tests/
    └── test_gsplats_integration.py  # Comprehensive tests
```

## Running Demos

```bash
# Performance demonstration with visualization
python -m luxar.gsplats.demo.demo_performance

# Interactive compression demo
python -m luxar.gsplats.demo.demo_splats

# Per-splat optimizer demonstration
python -m luxar.gsplats.demo.demo_per_splat_quick

# Quick performance benchmark
python -m luxar.gsplats.demo.quick_benchmark

# 3D volume visualization
python -m luxar.gsplats.demo.demo_splats_3d_napari_example

# Dynamic operations are enabled by default in all demos
python -m luxar.gsplats.demos.demo_performance
```

## Testing

Run comprehensive tests to verify correctness:

```bash
pytest packages/luxar/src/luxar/gsplats/tests/test_gsplats_integration.py -v
```

Tests cover:
- 2D/3D reconstruction pipelines
- Early stopping convergence
- Loss functions (MSE, Poisson)
- Regularization effects
- Sigma constraints
- Device compatibility
- Edge cases and error handling

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