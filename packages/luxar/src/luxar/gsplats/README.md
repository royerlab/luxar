# Gaussian Splatting Package for Luxar

High-performance n-dimensional oriented Gaussian splatting with automatic optimization for efficient image and volume reconstruction.

## Overview

This package implements a sophisticated Gaussian splatting system that fits collections of oriented Gaussian functions (splats) to reconstruct n-dimensional data. Each splat is a multivariate Gaussian characterized by position, full covariance matrix, and amplitude. The implementation features automatic convergence detection, adaptive learning, and GPU acceleration.

## Key Features

- **N-dimensional Support**: Works seamlessly with 2D images, 3D volumes, and higher dimensions
- **Efficient Cholesky Parameterization**: Covariance matrices via Cholesky decomposition with batched triangular solve
- **Device Optimized**: CUDA acceleration with automatic device selection (CPU preferred on Apple Silicon)
- **Oriented Gaussians**: Full covariance matrices via Cholesky decomposition for arbitrary orientations
- **Dynamic Operations**: Adaptive splat management with pruning, seeding, merging, and splitting during optimization
- **Per-Splat Optimization**: Individual learning rates and momentum preservation for each Gaussian splat
- **Automatic Optimization**: Early stopping saves 20-60% of iterations without quality loss
- **GPU Acceleration**: CUDA support with batched operations, improved MPS compatibility
- **Multiple Loss Functions**: MSE for general data, Poisson for photon/count data
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
- **Dynamic Operations**: Adaptive model topology with automatic pruning, seeding, merging, and splitting
- **Early Stopping**: Monitors loss history to detect convergence
- **Adaptive Learning Rate**: ReduceLROnPlateau scheduler
- **Best State Tracking**: Restores optimal parameters if loss increases
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

```python
from luxar.gsplats.candidates import find_candidates_overcomplete_nd
from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.models.gsplats.gsplat_model import render_gaussians_numpy

# 1. Generate candidate centers
candidates = find_candidates_overcomplete_nd(
    image,
    scales=(0.8, 1.2, 1.8, 2.6),  # Multiscale detection
    peaks_per_scale=500,           # Max peaks per scale
    min_dist=2.0                   # Minimum separation
)

# 2. Fit Gaussian splats (precision parameterization by default)
params, amps, stats = fit_gaussian_splats(
    image,
    centers_overcomplete=candidates,
    n_iters=300,                          # Maximum iterations
    lr=0.2,                               # Learning rate
    loss_type="mse",                      # or "poisson" for count data
    l1_amp=0.01,                         # Sparsity regularization
    early_stopping=True,                  # Stop when converged (default)
    enable_dynamic_ops=True,              # Enable adaptive topology (optional)
    dynamic_config=None,                  # Use default config or provide custom
)

# 3. Render reconstruction
reconstruction = render_gaussians_numpy(
    image.shape, params, amps, truncate=3.0
)
```

## Dynamic Operations

The implementation supports adaptive model topology through dynamic operations that automatically adjust the number and properties of Gaussian splats during optimization:

### Dynamic Operations Overview

- **Pruning**: Remove weak splats with low amplitude
- **Seeding**: Add new splats in high-error regions  
- **Merging**: Combine nearby similar splats to reduce redundancy
- **Splitting**: Divide large anisotropic splats for better detail capture

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
│   ├── demo_performance.py    # Performance showcase
│   ├── demo_splats.py         # Compression visualization
│   ├── demo_per_splat_quick.py # Per-splat optimizer demonstration
│   ├── enable_dynamic_ops_guide.py # Guide for adding dynamic ops to existing demos
│   └── demo_splats_3d_*.py    # 3D examples
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

# Dynamic operations integration guide
python -m luxar.gsplats.demo.enable_dynamic_ops_guide
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