# Multi-Scale Image Decomposition

Efficient n-dimensional multi-scale image decomposition for hierarchical Gaussian splat fitting.

## Overview

This package decomposes images into non-negative components at different scales, enabling:

- **10-50× faster fitting** for large splats (fit against downsampled images)
- **Better optimization** by separating frequency bands
- **Hierarchical representation** matching natural image statistics

## Quick Start

```python
from luxar.gsplats.multiscale import decompose_image
import numpy as np

# Load your nD image
V = np.load("data.npy")  # Works for 2D, 3D, 4D+

# Decompose into scales
scales_list, stats = decompose_image(
    V,
    scales=[1, 2, 4, 8],  # Full, half, quarter, eighth resolution
    n_iters=500
)

# Use scale components
V_full = scales_list[0]      # Full resolution (1x)
V_half = scales_list[1]      # Half resolution (2x)
V_quarter = scales_list[2]   # Quarter resolution (4x)
V_eighth = scales_list[3]    # Eighth resolution (8x)
```

## How It Works

The decomposition optimizes:

```
V = Σₖ upsample(Vₖ)
```

where each `Vₖ` is a non-negative image at scale `k`. A hierarchical energy loss pushes low-frequency content toward coarse scales:

```
Loss = L1(reconstruction, target) + λ × Σₖ(αᵏ × ∫Vₖ)
```

## Parameters

### Key Parameters

- **`scales`** (list): Scale factors, e.g., `[1, 2, 4, 8]`
  - `1` = full resolution, `2` = half resolution, etc.
- **`n_iters`** (int): Optimization iterations (default: 500)
- **`alpha`** (float): Energy penalty growth (default: 1.5)
  - Higher = more energy to coarse scales
  - Try 2.0-3.0 if energy stays in finest scale
- **`energy_weight`** (float): Energy penalty strength (default: 0.01)
  - Higher = stronger coarse preference
  - Increase if alpha alone isn't enough

### Advanced Parameters

- **`init_method`** (str): Initialization method (default: "coarse")
  - **"coarse"** (coarse-weighted initialization) - **STRONGLY RECOMMENDED**
    - Energy weighted toward coarse scales proportional to scale factor
    - For scales [1, 2, 4, 8]: energy is [1x, 2x, 4x, 8x] respectively
    - **Empirically provides the best convergence and final quality**
    - Strongly aligns with the optimization objective (pushing energy to coarse scales)
    - Default and recommended for all use cases
  - **"pyramid"** (Gaussian pyramid)
    - Energy distributed across scales from coarse to fine using Gaussian pyramid
    - Natural frequency decomposition
    - Available for experimentation but generally inferior to "coarse"
  - **"uniform"** (uniform energy split)
    - Energy split equally across all scales (when upsampled)
    - Each scale gets 1/K of total energy
    - Balanced starting point
    - Available for experimentation but generally inferior to "coarse"
  - **"finest"** (finest scale initialization)
    - All energy starts in finest (highest resolution) scale
    - Other scales start at near-zero values
    - Useful for visualizing energy redistribution during optimization
    - Generally converges slower and achieves worse final quality
- **`loss_type`** (str): Type of reconstruction loss (default: "l1")
  - **"l1"** (Mean Absolute Error) - **RECOMMENDED DEFAULT**
    - Most robust to outliers
    - Provides stable gradients throughout optimization
    - Excellent for most image decomposition tasks
  - **"mse"** (Mean Squared Error)
    - Penalizes large errors more heavily than L1
    - May provide slightly better reconstruction quality in some cases
    - More sensitive to outliers
  - **"poisson"** (Poisson Deviance)
    - Appropriate for count data (photon counts, particle counts)
    - Models Poisson noise statistics
- **`asymmetric_penalty`** (float): Over-prediction penalty factor (default: 10.0)
  - Penalizes regions where reconstruction > target by this factor
  - Default 10× penalty strongly discourages overshooting
  - Addresses fundamental asymmetry: under-prediction is easier to fix than over-prediction
  - Works with all loss types (L1, MSE, Poisson)
  - Set to `None` to disable (symmetric loss)
- **`max_abs_error_threshold`** (float, optional): Convergence threshold (default: None → auto)
  - Stops optimization when max|reconstruction - target| < threshold
  - **Auto-convergence** (None): Uses 1% of image value range
  - **Manual**: Set specific value for custom convergence criteria
  - **Quality guarantee**: Always returns best result found, not final iteration
  - Example: `max_abs_error_threshold=0.001` for strict convergence
- **`interpolation`** (str): Interpolation method for upsampling (default: 'cubic')
  - **'nearest'**: Fastest, blocky (good for debugging)
  - **'linear'**: Fast, smooth (still fastest for 3D when speed is critical)
  - **'cubic'**: Highest quality, now practical for 3D (Keys cubic convolution is 27-43× faster than old torch-interpol)
  - See [Interpolation Methods](#interpolation-methods) section for detailed benchmarks
- **`napari_movie`** (bool): Enable recording of optimization progress (default: False)
  - Records target, reconstruction, and residual at each frame
  - Use `show_optimization_movie()` to visualize after optimization
- **`movie_every`** (int): Record frame every N iterations (default: 1)
  - Only relevant if `napari_movie=True`
- **`movie_max_frames`** (int or None): Maximum frames to store (default: None → 10000)
  - Prevents unbounded memory growth
  - Oldest frames discarded when limit reached
- **`lr`** (float): Learning rate (default: 0.01)
- **`device`** (str): 'cpu', 'cuda', or 'mps' (auto-detects by default)
- **`verbose`** (bool): Print progress (default: True)

## Examples

### 2D Image Decomposition

```python
from luxar.gsplats.multiscale import decompose_image
import matplotlib.pyplot as plt

# Load 2D image
image = plt.imread("photo.png")[:, :, 0]  # Grayscale

# Decompose
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4],
    n_iters=500
)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
for i, (scale, img) in enumerate(zip([1, 2, 4], scales_list)):
    axes[i+1].imshow(img, cmap='viridis')
    energy = stats['energy_distribution'][i]
    axes[i+1].set_title(f'Scale {scale}x ({energy:.1%})')
plt.show()
```

### 3D Volume Decomposition

```python
# Load 3D volume
volume = np.load("volume.npy")  # Shape: (128, 128, 128)

# Decompose with aggressive coarse preference
scales_list, stats = decompose_image(
    volume,
    scales=[1, 2, 4, 8],
    n_iters=300,
    alpha=2.0,             # Stronger coarse preference
    energy_weight=0.01,    # Higher penalty
    interpolation='cubic'  # Cubic is now practical for 3D (Keys cubic is fast!)
)

print(f"Energy distribution: {stats['energy_distribution']}")
# Expected: [0.05, 0.12, 0.25, 0.58] (fine → coarse)
```

### Interpolation Method Selection

```python
# 2D image: cubic is fine (good quality and speed)
image_2d = np.load("image.npy")  # Shape: (256, 256)
scales_list, stats = decompose_image(
    image_2d,
    scales=[1, 2, 4],
    interpolation='cubic'  # Default, good quality/speed for 2D
)

# 3D volume: cubic is now practical with Keys cubic convolution!
volume_3d = np.load("volume.npy")  # Shape: (128, 128, 128)
scales_list, stats = decompose_image(
    volume_3d,
    scales=[1, 2, 4],
    interpolation='cubic'  # Keys cubic: ~27-43× faster than old torch-interpol
)

# Linear is still fastest if speed is critical
scales_list, stats = decompose_image(
    volume_3d,
    scales=[1, 2, 4],
    interpolation='linear'  # Fastest option, smooth results
)
```

### Optimization Movie Visualization

```python
from luxar.gsplats.multiscale import decompose_image, show_optimization_movie

# Decompose with movie recording enabled
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4],
    n_iters=500,
    napari_movie=True,      # Enable movie recording
    movie_every=5,          # Record every 5 iterations
    movie_max_frames=100    # Store up to 100 frames
)

# Display convergence movie in napari
if stats['movie_frames'] is not None:
    # Pass interpolation mode from stats to ensure movie matches optimization
    interpolation = stats.get('interpolation', 'cubic')
    show_optimization_movie(stats['movie_frames'], image.shape, interpolation=interpolation)
    # Use time slider to scrub through optimization
    # Compare target, reconstruction, all scales, and residual over time
    # Toggle scale layers to see energy evolution across scales
```

### Initialization Methods

```python
# Coarse-weighted initialization (default and STRONGLY RECOMMENDED)
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    init_method='coarse',  # Default - empirically the best method
    n_iters=500
)
# Provides best convergence and final quality
# Strongly aligns with optimization objective

# Other methods available for experimentation (generally inferior):

# Gaussian pyramid initialization
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    init_method='pyramid',
    n_iters=500
)
# Natural frequency decomposition, but generally worse than 'coarse'

# Uniform energy split
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    init_method='uniform',
    n_iters=500
)
# Balanced starting point, but generally worse than 'coarse'

# Finest scale initialization (mainly for visualization)
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    init_method='finest',
    n_iters=1000,           # Requires more iterations
    napari_movie=True       # Watch energy redistribute
)
# Useful for visualizing optimization dynamics
# Converges slower and achieves worse final quality than 'coarse'
```

### Convergence and Early Stopping

```python
# Auto-convergence (recommended): stops when max error < 1% of image range
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    n_iters=1000,  # Max iterations
    # max_abs_error_threshold=None  # Default: auto (1% of range)
)

# Check convergence
if stats['converged']:
    print(f"✓ Converged after {stats['actual_iters']} iterations")
    print(f"  Best max error: {stats['best_max_abs_error']:.6f}")
    print(f"  Best iteration: {stats['best_iteration']}")
else:
    print(f"⚠ Did not converge after {stats['actual_iters']} iterations")

# Custom convergence threshold
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    n_iters=1000,
    max_abs_error_threshold=0.001,  # Strict convergence criterion
)

# Disable convergence check (run all iterations)
scales_list, stats = decompose_image(
    image,
    scales=[1, 2, 4, 8],
    n_iters=500,
    max_abs_error_threshold=1e-10,  # Effectively disabled
)
```

**Key features:**
- **Auto-convergence**: Adaptive threshold based on image range
- **Quality guarantee**: Always returns best result, not final iteration
- **Early stopping**: Saves computation when converged
- **Convergence tracking**: Check `stats['converged']` and `stats['best_iteration']`

### Multi-Scale Splat Fitting

```python
from luxar.gsplats import fit_gaussian_splats

# Step 1: Decompose
scales_list, _ = decompose_image(V, scales=[1, 2, 4, 8])

# Step 2: Fit splats at each scale
all_results = []

for scale_factor, V_scale in zip([1, 2, 4, 8], scales_list):
    # Fit with scale-appropriate sigma
    result = fit_gaussian_splats(
        V_scale,
        init_sigma_vox=1.5 * scale_factor,  # Larger splats for coarse scales
        n_iters=1000
    )

    # Scale parameters back to full resolution
    d = V.ndim
    if scale_factor > 1:
        result.centers[:, :d] *= scale_factor  # Scale centers
        result.cholesky_factors *= scale_factor  # Scale Cholesky factors

    all_results.append(result)

# Combine centers, amplitudes, etc. from all results
import numpy as np
centers_combined = np.vstack([r.centers for r in all_results])
amps_combined = np.concatenate([r.amplitudes for r in all_results])
```

## Tuning Guide

### Energy Distribution Goals

For 4 scales `[1, 2, 4, 8]` with default parameters:

**Good distribution:**
```
Scale 8x (coarsest): ~60%
Scale 4x:            ~25%
Scale 2x:            ~10%
Scale 1x (finest):   ~5%
```

**Warning signs:**
- Finest scale > 50%: Trivial solution, increase `alpha` or `energy_weight`
- All scales ~25%: Good balance, but may not achieve speedup goals
- Coarsest scale > 90%: Too aggressive, reduce `alpha` or `energy_weight`

### Parameter Tuning Strategy

1. **Start conservative:** `alpha=1.5`, `energy_weight=0.001`
2. **Check energy distribution** in stats
3. **If energy too fine:**
   - First try `alpha=2.0`
   - Then try `energy_weight=0.01`
4. **If reconstruction poor:** Increase `n_iters` or decrease penalties

## Running Demos

### 2D Astronaut Demo

```bash
cd packages/luxar/src/luxar/gsplats/multiscale/demos
python demo_decompose_2d.py
```

Demonstrates decomposition on the astronaut image with visualization.

### 2D Mitosis Demo

```bash
python demo_decompose_2d_mitosis.py
```

Demonstrates decomposition on human mitosis biological histology data, showing how cellular features (overall cell shapes, nuclear boundaries, chromatin patterns) are distributed across different scales.

### 3D Demo

```bash
python demo_decompose_3d.py
```

Demonstrates 3D volume decomposition with optional napari visualization using synthetic data.

### 3D DAPI Nuclei Demo (Real Data)

```bash
python demo_decompose_3d_dapi_nuclei.py
```

Demonstrates multi-scale decomposition on real DAPI-stained nuclei from the Image Data Resource (IDR). This demo loads full-resolution microscopy data from a remote zarr store and decomposes it into multiple scales, showing how nuclear structures are distributed hierarchically. **Note:** This demo requires internet connectivity to load the remote zarr dataset.

### Initialization Method Comparison

```bash
python demo_compare_initialization.py
```

Compares convergence speed and quality of all four initialization methods (pyramid, uniform, coarse, finest) on the astronaut image. Generates console-friendly ASCII plots showing:
- Reconstruction loss curves over iterations
- Summary table with final losses and timing
- Energy distribution bar charts for each method

This demo helps you understand which initialization method works best for your data and optimization parameters.

## Running Tests

```bash
# Run all tests
pytest packages/luxar/src/luxar/gsplats/multiscale/tests/

# Run specific test file
pytest packages/luxar/src/luxar/gsplats/multiscale/tests/test_decomposition_basic.py -v

# Run with coverage
pytest packages/luxar/src/luxar/gsplats/multiscale/tests/ --cov=luxar.gsplats.multiscale
```

## API Reference

### Main Function

```python
decompose_image(
    V: np.ndarray,
    scales: List[int] = [1, 2, 4, 8, 16, 32],
    n_iters: int = 500,
    lr: float = 0.01,
    energy_weight: float = 0.01,
    alpha: float = 1.5,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0,
    init_method: str = "coarse",
    max_abs_error_threshold: Optional[float] = None,
    interpolation: str = 'cubic',
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], Dict[str, Any]]
```

**Returns:**
- `scales_list`: List of non-negative images at each scale (best result, not final iteration)
- `stats`: Dictionary with optimization statistics
  - `'final_error'`: Final reconstruction MSE (from best result)
  - `'best_error'`: Best reconstruction loss achieved
  - `'best_max_abs_error'`: Best maximum absolute error achieved
  - `'converged'`: Boolean indicating if convergence criterion was met
  - `'best_iteration'`: Iteration where best result was achieved
  - `'actual_iters'`: Actual number of iterations run (may be less than n_iters if converged early)
  - `'energy_distribution'`: Energy fraction per scale
  - `'maxima_per_scale'`: Local-maxima count per scale (for downstream seed distribution; always >= 1)
  - `'scales'`: Scale factors actually used (after filtering scales too large for the image)
  - `'history'`: Per-iteration statistics
  - `'time_seconds'`: Total optimization time
  - `'interpolation'`: Interpolation mode used ('nearest', 'linear', or 'cubic')
  - `'movie_frames'`: Movie data (if napari_movie=True), or None

### Model Class

```python
class MultiScaleDecomposer(nn.Module):
    """PyTorch model for multi-scale decomposition."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        scales: List[int],
        interpolation: str = 'cubic'
    )
    def forward() -> Tuple[List[Tensor], List[Tensor], Tensor]
    def initialize_from_pyramid(target: Tensor) -> None
    def initialize_uniform(target: Tensor) -> None
    def initialize_coarse(target: Tensor) -> None
    def initialize_finest_scale(target: Tensor) -> None
```

**Initialization Methods:**
- `initialize_coarse()`: Energy weighted toward coarse scales proportional to scale factor (**strongly recommended**, default)
- `initialize_from_pyramid()`: Distribute energy across scales from coarse to fine
- `initialize_uniform()`: Split energy equally across all scales (balanced approach)
- `initialize_finest_scale()`: Put all energy in finest scale, other scales near zero
- `initialize_zero()`: All scales start at near-zero (worst-case baseline, research only)

### Loss Function

```python
decomposition_loss(
    model: MultiScaleDecomposer,
    target: torch.Tensor,
    energy_weight: float = 0.001,
    alpha: float = 1.5,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0
) -> Tuple[torch.Tensor, Dict[str, float]]
```

### Visualization

```python
upsample_for_visualization(
    img: np.ndarray,
    target_shape: Tuple[int, ...],
    interpolation: str = 'cubic'
) -> np.ndarray
```

Upsample a numpy array to the target shape for visualization purposes, using the same interpolation methods as the optimization to ensure visual consistency. The `interpolation` parameter should match what was used during `decompose_image()`.

```python
show_optimization_movie(
    movie_frames: Dict[str, Any],
    shape: Tuple[int, ...],
    interpolation: str = 'cubic'
) -> None
```

Display napari viewer with optimization movie showing convergence progress including target, reconstruction, all individual scale components, and residual over time. The `movie_frames` dictionary should come from `decompose_image()` with `napari_movie=True`.

The `interpolation` parameter should match the mode used during optimization (available from `stats['interpolation']`) to ensure the movie visualization matches what was optimized.

Individual scale components are all visible by default - toggle their visibility to reduce clutter if needed. This allows you to see how energy evolves across scales during optimization.

## Performance

### Computational Cost

For image size S^d with K scales, per iteration:
- **Forward pass:** O(K × S^d) for upsampling
- **Loss:** O(S^d) reconstruction + O(sum of scales) energy
- **Backward:** O(K × S^d)

**Typical:** 256³ volume, 4 scales, 500 iterations ≈ 2-5 minutes on GPU

### Memory Usage

- **Model parameters:** O(S^d) total across all scales
- **Gradients:** Same as parameters
- **Intermediate tensors:** O(K × S^d)

**Typical:** 256³ volume, 4 scales ≈ 2-3 GB GPU memory

### Interpolation Methods

The decomposition supports three interpolation methods for upsampling scale components:

```python
scales_list, stats = decompose_image(
    V,
    scales=[1, 2, 4, 8],
    interpolation='cubic',  # 'nearest', 'linear', or 'cubic' (default)
)
```

#### Available Methods

1. **`'nearest'`** - Nearest-neighbor interpolation
   - **Speed**: Fastest
   - **Quality**: Blocky, no smoothing
   - **Use case**: Quick prototyping, debugging

2. **`'linear'`** - Linear interpolation
   - **Speed**: Medium
   - **Quality**: Smooth, no overshoot/undershoot
   - **Use case**: General purpose, when non-negativity is critical
   - **Implementation**: Bilinear (2D), Trilinear (3D)

3. **`'cubic'`** - Cubic interpolation (default)
   - **Speed**: Slower than linear but practical for both 2D and 3D (Keys cubic is 27-43× faster than old torch-interpol)
   - **Quality**: Highest quality, smoothest
   - **Use case**: Default choice for quality; linear is still faster when speed is critical
   - **Implementation**: Bicubic (2D, PyTorch), Keys cubic convolution (3D+, custom vectorized)
   - **Note**: Can produce small negative values (undershoot) which are automatically clamped to zero

#### Performance Benchmarks

Benchmarks were run on an M1 Mac with both CPU and MPS (Metal Performance Shaders) acceleration using the new Keys cubic convolution implementation:

**2D Images (100 iterations, scales [1, 2, 4]):**

| Size | Device | Cubic (Keys) | Final Error | Notes |
|------|--------|--------------|-------------|-------|
| 128² | CPU | 0.25s ± 0.15s | 1.65e-03 | Good performance |
| 128² | MPS | 0.55s ± 0.14s | 2.35e-03 | CPU competitive for small 2D |
| 256² | CPU | 0.49s ± 0.03s | 1.71e-03 | Scales well |
| 256² | MPS | 0.42s ± 0.02s | 1.72e-03 | MPS advantage for larger 2D |

**3D Volumes (50 iterations, scales [1, 2, 4]):**

| Size | Device | Cubic (Keys) | Final Error | Notes |
|------|--------|--------------|-------------|-------|
| 64³ | CPU | 0.98s ± 0.11s | 1.47e-02 | **~27× faster than old torch-interpol!** |
| 64³ | MPS | 0.77s ± 0.05s | 1.50e-02 | **~34× faster than old torch-interpol!** |
| 96³ | CPU | 2.90s ± 0.05s | 1.46e-02 | **~31× faster than old torch-interpol!** |
| 96³ | MPS | 1.16s ± 0.14s | 1.50e-02 | **~43× faster than old torch-interpol!** |

**Comparison with Previous Implementation:**

The old torch-interpol B-spline cubic implementation was extremely slow for 3D:
- 64³ volume: **18.7s CPU** (old) → **0.98s CPU** (new) = **19× speedup**
- 64³ volume: **2.6s MPS** (old) → **0.77s MPS** (new) = **3.4× speedup**

**Key Findings:**

1. **2D performance**: Cubic is practical for 2D images
   - ~0.25-0.55s for 128² images (100 iterations)
   - CPU and MPS are competitive for small images
   - MPS shows advantage for larger images (256²)

2. **3D performance**: Keys cubic convolution is now practical for 3D!
   - **27-43× faster** than old torch-interpol implementation
   - Sub-second performance for 64³ volumes
   - GPU (MPS) acceleration provides additional 2-2.5× speedup
   - Cubic is now a viable option for 3D decomposition

3. **Device comparison**:
   - MPS (GPU) provides consistent 1.3-2.5× speedup over CPU
   - Larger volumes benefit more from GPU acceleration
   - Keys cubic implementation is fully GPU-accelerated

4. **Implementation details**:
   - No external dependencies (torch-interpol removed)
   - Vectorized PyTorch operations with torch.unfold
   - Separable filters for efficient nD processing
   - Maintains non-negativity through automatic clamping

#### Practical Recommendations

**For 2D images:**
```python
# Default cubic is fine - overhead is small
scales_list, stats = decompose_image(image_2d, interpolation='cubic')
```

**For 3D volumes:**
```python
# Linear is still fastest
scales_list, stats = decompose_image(volume_3d, interpolation='linear')

# Cubic is now practical for 3D (27-45× faster than old torch-interpol)
scales_list, stats = decompose_image(volume_3d, interpolation='cubic')  # Uses Keys cubic
```

**For debugging/prototyping:**
```python
# Use nearest for fastest iteration
scales_list, stats = decompose_image(data, interpolation='nearest', n_iters=50)
```

#### Implementation Details

- **All modes** use native PyTorch operations
- **Cubic 3D+** uses custom vectorized Keys cubic convolution (no external dependencies)
- **Cubic 2D** uses PyTorch's bicubic interpolation
- **Linear** uses PyTorch's bilinear/trilinear interpolation
- **Nearest** uses PyTorch's nearest-neighbor interpolation

## Troubleshooting

### Issue: All energy in finest scale

**Solution:**
```python
scales_list, stats = decompose_image(
    V,
    alpha=2.5,           # Increase from 1.5
    energy_weight=0.01   # Increase from 0.001
)
```

### Issue: Poor reconstruction quality

**Solution:**
```python
scales_list, stats = decompose_image(
    V,
    n_iters=1000,        # Increase iterations
    alpha=1.2,           # Decrease penalty
    energy_weight=0.0001 # Decrease penalty
)
```

### Issue: Optimization too slow

**Solution:**
```python
scales_list, stats = decompose_image(
    V,
    scales=[1, 2, 4],    # Fewer scales
    n_iters=300,         # Fewer iterations
    device='cuda'        # Use GPU
)
```

## Citation

If you use this package in your research, please cite:

```bibtex
@software{luxar_multiscale,
  title = {Multi-Scale Image Decomposition for Gaussian Splatting},
  author = {Luxar Development Team},
  year = {2025},
  url = {https://github.com/royerlab/luxar}
}
```

## See Also

- [Main Luxar Documentation](../../README.md) - Luxar Python package
- [Gaussian Splat Fitting](../README.md) - Core fitting algorithms
