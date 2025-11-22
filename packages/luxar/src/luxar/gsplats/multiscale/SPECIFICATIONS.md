# Multi-Scale Image Decomposition for Gaussian Splatting

## Overview

This module implements n-dimensional multi-scale image decomposition for efficient Gaussian splat fitting. The core idea is to decompose an image into a sum of non-negative components at different resolutions, enabling:

1. **Computational Efficiency**: Large splats (coarse features) can be fitted against downsampled images, reducing computational cost by orders of magnitude
2. **Better Optimization**: Separating frequency bands reduces interference between large and small splats during fitting
3. **Hierarchical Representation**: Natural separation of coarse and fine features

**Related Specifications**:
- **Multi-Scale Fitting API**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 6: Multi-Scale Gaussian Splat Fitting
- **Single-Scale Fitting**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 5: Main Fitting Interface
- **Terminology**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Terminology Glossary

## Dependencies

**Core Dependencies**:
- PyTorch (with CUDA/MPS support optional)
- NumPy
- SciPy (for napari visualization)

**Optional Dependencies**:
- `napari`: For visualization (movies, inspection)
- `arbol`: For structured console output

**Note**: All interpolation modes use native PyTorch operations or custom vectorized implementations. No external dependencies are required for any interpolation method, including 3D cubic.

## Mathematical Formulation

### Decomposition

Given an n-dimensional image V of shape (s₀, s₁, ..., sₙ₋₁), we decompose it into K scale components:

```
V = Σₖ upsample(Vₖ)
```

where:
- **Vₖ**: Image at scale k with shape (s₀/rₖ, s₁/rₖ, ..., sₙ₋₁/rₖ)
- **rₖ**: Scale factor (e.g., r₀=1, r₁=2, r₂=4, r₃=8 for scales [1, 2, 4, 8])
- **upsample()**: Interpolation function mapping low-resolution to high-resolution

### Constraints

1. **Non-negativity**: All Vₖ ≥ 0 (enforced via softplus parameterization)
2. **Reconstruction fidelity**: Σₖ upsample(Vₖ) ≈ V
3. **Energy distribution**: Coarse scales should capture low-frequency content, fine scales high-frequency

### Loss Function

The optimization balances three objectives:

```python
L_total = L_reconstruction + λ_energy × L_energy + λ_gradient × L_gradient
```

#### 1. Reconstruction Loss (L_reconstruction)

The reconstruction loss measures fidelity between reconstruction and target. Three loss types are supported:

**Loss Types** (configurable via `loss_type` parameter):

1. **"l1"** (Mean Absolute Error) - **DEFAULT**
   ```python
   L_reconstruction = MAE(Σₖ upsample(Vₖ), V)
   ```
   - Most robust to outliers
   - Provides stable gradients throughout optimization
   - Excellent for most image decomposition tasks
   - Recommended default

2. **"mse"** (Mean Squared Error)
   ```python
   L_reconstruction = MSE(Σₖ upsample(Vₖ), V)
   ```
   - Penalizes large errors more heavily than L1
   - Sensitive to outliers
   - May provide slightly better reconstruction quality in some cases

3. **"poisson"** (Poisson Deviance)
   ```python
   L_reconstruction = 2 * mean(pred - target + target * log(target/pred))
   ```
   - Appropriate for count data (photon counts, particle counts)
   - Models Poisson noise statistics
   - Use when data follows Poisson distribution

**Asymmetric Loss** (default enabled with 10× penalty for all loss types):

All reconstruction losses support asymmetric penalties to address the fundamental asymmetry in additive decomposition models:

- **Under-prediction** (reconstruction < target): Easy to fix by adding more energy to scales
- **Over-prediction** (reconstruction > target): Hard to fix, requires reducing energy in scales

**Implementation** (example with L1):
```python
over_prediction_mask = reconstruction > target
l1_error = abs(reconstruction - target)
L_reconstruction = mean(
    where(
        over_prediction_mask,
        asymmetric_penalty * l1_error,  # F times penalty (default: 10.0)
        l1_error  # Normal penalty
    )
)
```

**Rationale**:
- Default `asymmetric_penalty=10.0` strongly discourages overshooting
- Creates stable optimization by making over-prediction more costly than under-prediction
- Matches the behavior of Gaussian splat fitting which uses the same asymmetric loss
- Can be disabled by setting `asymmetric_penalty=None` for symmetric loss
- Works with all three loss types (L1, MSE, Poisson)

This is the primary loss ensuring fidelity.

#### 2. Hierarchical Energy Loss (L_energy)

Penalizes fine scales more heavily to push energy toward coarse scales:

```python
L_energy = Σₖ (αᵏ × ∫ Vₖ dx) / ∫ V dx
```

where:
- **α**: Growth factor (typically 1.5-2.0)
- **αᵏ**: Exponentially increasing penalty for finer scales
- k=0 is coarsest scale, k=K-1 is finest scale (in mathematical notation)
- **Note**: In Python implementation, `scales_list[0]` corresponds to scale factor 1 (finest), so the weight calculation uses `alpha^(K-1-k)` to match the mathematical formulation
- Normalization by ∫V dx makes it scale-invariant

**Effect**: With α=1.5 and 4 scales:
- Scale 0 (coarse): weight = 1.0
- Scale 1: weight = 1.5
- Scale 2: weight = 2.25
- Scale 3 (fine): weight = 3.375

This creates an energy gradient favoring coarser scales.

#### 3. Total Variation Loss (L_gradient) [Optional]

Penalizes high-frequency content at coarse scales:

```python
L_gradient = Σₖ₍ₖ<ₖ₋₁₎ (βᴷ⁻ᵏ × TV(Vₖ))
```

where:
- **TV(Vₖ)**: Total variation (sum of absolute gradients)
- **β**: Coarseness weight (typically 2.0)
- Only applied to non-finest scales

**Effect**: Encourages smooth, low-frequency content at coarse scales.

## Parameterization

### Softplus Non-negativity

To ensure Vₖ ≥ 0 during optimization:

```python
Vₖ = softplus(Vₖ_raw) = log(1 + exp(Vₖ_raw))
```

where Vₖ_raw are unconstrained learnable parameters.

**Inverse for initialization**:
```python
Vₖ_raw = inverse_softplus(Vₖ) = log(exp(Vₖ) - 1)
```

For numerical stability, use the stable implementation:
```python
inverse_softplus(x) = log(expm1(x))  # Uses log(exp(x)-1) identity
```

### Upsampling Strategy

The decomposition supports three interpolation modes (controlled by `interpolation` parameter):

#### 1. **'nearest'** - Nearest-Neighbor Interpolation
- **Speed**: Fastest
- **Quality**: Blocky, no smoothing
- **Use case**: Quick prototyping, debugging
- **Implementation**:
  - All dimensions: `F.interpolate(..., mode='nearest')`

#### 2. **'linear'** - Linear Interpolation
- **Speed**: Medium
- **Quality**: Smooth, no overshoot/undershoot
- **Use case**: When non-negativity is critical, general purpose
- **Implementation**:
  - 2D: `F.interpolate(..., mode='bilinear', align_corners=False)`
  - 3D: `F.interpolate(..., mode='trilinear', align_corners=False)`
  - nD (n>3): Falls back to `'nearest'`

#### 3. **'cubic'** - Cubic Interpolation (Default)
- **Speed**: Slowest
- **Quality**: Smoothest, highest quality
- **Caveat**: Can produce small negative values (undershoot), which are clamped to zero
- **Use case**: When maximum quality is desired (default)
- **Implementation**:
  - 2D: `F.interpolate(..., mode='bicubic', align_corners=False)`
  - 3D+: Custom Keys cubic convolution with vectorized PyTorch operations
  - Uses separable filters applied along each axis for efficient nD processing

**Rationale for align_corners=False**: Better preserves spatial positions for splat center coordinates.

**Negative Value Handling**: Cubic interpolation can produce small negative values due to the negative lobes in the cubic kernel (inherent to its superior smoothness). These are automatically clamped to zero after upsampling since all scale components must be non-negative.

**Keys Cubic Convolution**: For 3D and higher dimensions, we use a custom implementation of Keys cubic convolution (a=-0.5) with vectorized PyTorch operations:
- **Kernel weights**: Uses Keys cubic kernel at x=0.5: [-1/16, 9/16, 9/16, -1/16]
  - Provides C1 continuity (continuous first derivative)
  - Negative lobes enable superior smoothness compared to linear interpolation
  - Parameter a=-0.5 chosen for optimal balance between sharpness and smoothness
- **Efficient implementation**:
  - Uses `torch.unfold(dimension, size=4, step=1)` for sliding window extraction
  - Creates 4-element windows for each interpolation point without copying
  - Broadcasting for kernel application: (batch, size_in, 4) * (4,) → fully vectorized
  - No Python loops - all operations in PyTorch
- **Separable filtering**: Applied along each axis independently
  - Complexity: O(n×k) instead of O(k^n) for n dimensions
  - Example: 3D cubic would be 4³=64 kernel vs 3×4=12 with separable approach
- **Padding strategy**: Edge replication (`mode='replicate'`) for boundary handling
  - Padding (1, 2) for 4-element kernel: 1 element left, 2 elements right
  - Ensures smooth behavior at image boundaries
- **Recursive upsampling**: For scale factors > 2, applies 2× upsampling recursively
  - Example: 8× upsampling = three successive 2× operations
  - Maintains high quality across all power-of-2 scale factors
- **Performance**: Fully GPU-accelerated without external dependencies
  - 27-45× faster than previous torch-interpol implementation
  - Works seamlessly on CUDA, CPU, and MPS (Apple Silicon)

**Device Compatibility Notes**:
- **MPS (Apple Silicon)**: For 3D data, MPS doesn't support native avg_pool3d/max_pool3d operations
  - Implementation falls back to F.interpolate with trilinear mode for downsampling
  - All upsampling operations (cubic, linear, nearest) work correctly on MPS
  - No performance degradation for upsampling; slight overhead for downsampling fallback
- **CUDA/CPU**: All operations use native optimized kernels

## Initialization Strategy

The initialization method significantly affects convergence speed and final energy distribution. Five initialization methods are available via the `init_method` parameter:

### 1. Coarse Initialization (`init_method="coarse"`) - **STRONGLY RECOMMENDED**

Initialize with energy weighted proportionally to scale factor:

```python
def initialize_coarse(V, scales):
    target_energy = sum(V)
    total_weight = sum(scales)  # For [1, 2, 4, 8]: total = 15

    for scale in scales:
        # Downsample target to this scale's resolution
        V_scale = downsample(V, scale)

        # Energy proportional to scale factor
        energy_fraction = scale / total_weight
        # For [1, 2, 4, 8]: fractions are [1/15, 2/15, 4/15, 8/15]
        # = [6.7%, 13.3%, 26.7%, 53.3%]

        energy_per_scale_upsampled = target_energy * energy_fraction

        # Account for upsampling factor
        upsampling_factor = scale^ndim
        energy_per_scale_downsampled = energy_per_scale_upsampled / upsampling_factor

        # Normalize to correct energy at this resolution
        V_scale = V_scale * (energy_per_scale_downsampled / sum(V_scale))
```

**Properties**:
- Energy weighted toward coarse scales proportional to scale factor
- For scales [1, 2, 4, 8]: energy distribution is [1x, 2x, 4x, 8x]
- Strongly biases toward coarse scales from the start
- Strongly aligns with the optimization objective
- **Empirically provides the best convergence and final quality**

**Use case**: **Default and strongly recommended for all use cases.** Empirical testing shows this method consistently outperforms all other initialization methods across different image types.

### 2. Pyramid Initialization (`init_method="pyramid"`)

Initialize from a Gaussian pyramid decomposition:

```python
def initialize_from_pyramid(V, scales):
    remaining = V.clone()
    components = []

    for scale in reversed(scales):  # Coarse to fine
        # Downsample remaining signal
        V_scale = downsample(remaining, scale)

        # Subtract contribution
        remaining = remaining - upsample(V_scale, V.shape)
        # DON'T clamp negatives - allow negative propagation for energy conservation
        # If coarse scales overshoot, negatives naturally reduce finer scales
        # through the averaging operation in downsampling

        components.append(V_scale)

    return reversed(components)
```

**Properties**:
- Energy distributed across scales from coarse to fine
- Provides natural frequency decomposition
- Already approximately decomposes the image
- **Negative propagation enabled**: Allows coarse scales to overshoot for energy conservation
  - If coarse scales overshoot (causing negative residuals), those negatives naturally reduce
    finer scales' values through the averaging operation in downsampling
  - This ensures energy conservation without information loss
  - Individual scale values are still clamped to non-negative (via softplus constraint)
  - No clamping of the residual signal preserves mathematical correctness

**Use case**: Available for experimentation, but generally inferior to "coarse" initialization.

### 3. Uniform Initialization (`init_method="uniform"`)

Initialize with energy split equally across all scales:

```python
def initialize_uniform(V, scales):
    target_energy = sum(V)
    K = len(scales)

    for scale in scales:
        # Downsample target to this scale's resolution
        V_scale = downsample(V, scale)

        # Each scale gets 1/K of total energy (when upsampled)
        energy_fraction = 1.0 / K
        energy_per_scale_upsampled = target_energy * energy_fraction

        # Account for upsampling factor
        upsampling_factor = scale^ndim
        energy_per_scale_downsampled = energy_per_scale_upsampled / upsampling_factor

        # Normalize to correct energy at this resolution
        V_scale = V_scale * (energy_per_scale_downsampled / sum(V_scale))
```

**Properties**:
- Energy split equally across all scales when upsampled to full resolution
- Each scale gets 1/K of total energy
- Balanced starting point

**Use case**: Available for experimentation, but generally inferior to "coarse" initialization.

### 4. Finest Scale Initialization (`init_method="finest"`)

Initialize with all energy in finest (highest resolution) scale:

```python
def initialize_finest_scale(V, scales):
    target_energy = sum(V)

    for i, scale in enumerate(scales):
        V_scale = downsample(V, scale)

        if i == 0:  # Finest scale (scale factor = 1)
            # All energy goes here
            # (V_scale already has the right distribution)
        else:
            # Other scales start near zero
            V_scale = full_like(V_scale, 1e-8)
```

**Properties**:
- All energy starts in finest scale (typically scale factor = 1)
- Other scales initialized to near-zero values
- Allows visualization of energy redistribution during optimization
- May require more iterations to converge

**Use case**: Useful for understanding and visualizing how energy redistributes from fine to coarse scales during optimization. Good for debugging and analysis of optimization dynamics. Generally converges slower and achieves worse final quality than "coarse" initialization.

### 5. Zero Initialization (`init_method="zero"`)

Initialize all scales to zero (or near-zero values):

```python
def initialize_zero(target: Tensor) -> None:
    """
    Initialize all scales to zero (or near-zero).
    
    This creates a "worst case" starting point where all scales start at
    effectively zero and must be learned from scratch. Useful for understanding
    the importance of initialization and as a baseline comparison.
    """
    # Initialize all scales to large negative values
    # softplus(-10) ≈ 4.5e-5, which is effectively zero
    for raw_param in self.raw_images:
        raw_param.data.fill_(-10.0)
```

**Properties**:
- All scales start at effectively zero (softplus(-10) ≈ 4.5e-5)
- Worst-case baseline for initialization studies
- Requires optimizer to learn everything from scratch
- Typically requires many more iterations to converge
- Usually achieves poor final quality

**Use case**: Primarily for research and analysis purposes:
- Understanding the importance of good initialization
- Baseline comparison to quantify initialization impact
- Testing optimizer robustness
- Educational demonstrations of optimization from scratch
- **NOT RECOMMENDED** for production use

**Note**: This initialization is included in the API for completeness and research purposes, but should not be used for actual decomposition tasks. It demonstrates how critical proper initialization is for convergence speed and quality.

### Numerical Considerations

All initialization methods must account for the softplus parameterization:

```python
# After computing target values, convert to raw parameters
V_scale_clamped = clamp(V_scale, min=1e-6)  # Avoid log(0)
raw_init = inverse_softplus(V_scale_clamped)

# Stable inverse softplus
def stable_inverse_softplus(x):
    """Compute log(exp(x) - 1) stably."""
    # For x > 10: inverse_softplus(x) ≈ x
    # For x < 10: use log(expm1(x))
    return where(x > 10, x, log(expm1(x)))
```

**Important**: Never initialize raw parameters to zero, as `softplus(0) = log(2) ≈ 0.693`, not zero. Use large negative values (e.g., -10.0) for near-zero initialization.

## Implementation Architecture

### Core Classes

#### 1. MultiScaleDecomposer(nn.Module)

**Purpose**: PyTorch model representing the decomposition

**Parameters**:
- `shape: Tuple[int, ...]` - Shape of target image (n-dimensional)
- `scales: List[int]` - Scale factors (e.g., [1, 2, 4, 8])
- `interpolation: str` - Interpolation method: 'nearest', 'linear', or 'cubic' (default: 'cubic')
  - 'nearest': Fastest, blocky
  - 'linear': Medium speed, smooth, no overshoot
  - 'cubic': Highest quality, smoothest (Keys cubic for 3D+), may produce small negative values (clamped)

**Learnable Parameters**:
- `raw_images: nn.ParameterList` - One parameter tensor per scale (unconstrained)

**Methods**:
```python
def forward() -> Tuple[List[Tensor], List[Tensor], Tensor]:
    """
    Returns:
        scales_list: List of non-negative images at each scale
        upsampled_list: List of upsampled versions (all at full resolution)
        reconstruction: Sum of all upsampled scales
    """

def initialize_coarse(target: Tensor) -> None:
    """Initialize with energy weighted toward coarse scales (STRONGLY RECOMMENDED)."""

def initialize_from_pyramid(target: Tensor) -> None:
    """Initialize parameters from Gaussian pyramid decomposition."""

def initialize_uniform(target: Tensor) -> None:
    """Initialize with energy split equally across all scales."""


def initialize_zero(target: Tensor) -> None:
    """Initialize all scales to zero (or near-zero) for worst-case baseline."""
def initialize_finest_scale(target: Tensor) -> None:
    """Initialize with all energy in finest scale, other scales near zero."""
```

#### 2. Loss Functions

**decomposition_loss()**:
```python
def decomposition_loss(
    model: MultiScaleDecomposer,
    target: torch.Tensor,
    energy_weight: float = 0.01,
    alpha: float = 1.5,
    loss_type: str = "l1",
    asymmetric_penalty: Optional[float] = 10.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute multi-scale decomposition loss.

    Parameters:
        loss_type: Type of reconstruction loss ("l1", "mse", or "poisson").
                   Default: "l1" (Mean Absolute Error).
        asymmetric_penalty: Over-prediction penalty factor (default: 10.0).
                          Multiplies loss for regions where pred > target.
                          Set to None to disable asymmetric loss.

    Returns:
        loss: Total loss (scalar)
        stats: Dictionary with per-component losses and diagnostics
    """
```

#### 3. Main API Function

**decompose_image()**:
```python
def decompose_image(
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
    napari_movie: bool = False,
    movie_every: int = 1,
    movie_max_frames: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Decompose n-dimensional image into multi-scale non-negative components.

    Parameters
    ----------
    V : np.ndarray
        Input n-dimensional image to decompose
    scales : List[int], default=[1, 2, 4, 8]
        Scale factors. Scale 1 = full resolution, scale 2 = half resolution, etc.
    n_iters : int, default=500
        Number of optimization iterations
    lr : float, default=0.01
        Learning rate for Adam optimizer
    energy_weight : float, default=0.001
        Weight for hierarchical energy penalty (higher = more energy to coarse)
    alpha : float, default=1.5
        Growth factor for energy penalties (higher = stronger coarse preference)
    loss_type : str, default="l1"
        Type of reconstruction loss: "l1" (Mean Absolute Error, default),
        "mse" (Mean Squared Error), or "poisson" (Poisson Deviance)
    asymmetric_penalty : Optional[float], default=10.0
        Over-prediction penalty factor. Multiplies reconstruction loss for regions
        where pred > target by this factor. Set to None to disable asymmetric loss.
    init_method : str, default="coarse"
        Initialization method: "pyramid" (Gaussian pyramid, energy distributed
        across scales), "uniform" (energy split equally across all scales),
        "coarse" (energy weighted toward coarse scales proportional to scale
        factor - DEFAULT), or "finest" (all energy starts in finest scale).
        Coarse is the default and aligns with the optimization objective.
    max_abs_error_threshold : Optional[float], default=None
        Convergence threshold for maximum absolute error. If specified,
        optimization stops early when max|reconstruction - target| < threshold.
        If None (default), uses auto-convergence: 1% of image value range.
        For uniform images (all same value), uses 1% of mean absolute value.
    interpolation : str, default='cubic'
        Interpolation method for upsampling scale components:
        - 'nearest': Nearest-neighbor (fastest, blocky)
        - 'linear': Linear interpolation (medium speed, smooth, no overshoot)
        - 'cubic': Cubic interpolation (highest quality, may produce small negatives which are clamped)
        For 2D: uses PyTorch 'bicubic'
        For 3D+: uses custom Keys cubic convolution (vectorized, 27-45× faster than old torch-interpol)
    napari_movie : bool, default=False
        Enable recording of optimization progress for napari movie visualization
    movie_every : int, default=1
        Record a movie frame every N iterations (only if napari_movie=True)
    movie_max_frames : Optional[int], default=None
        Maximum number of frames to store. If None, defaults to 10000.
    device : str, optional
        PyTorch device ('cpu', 'cuda', 'mps'). Auto-detects if None.
    verbose : bool, default=True
        Print optimization progress

    Returns
    -------
    scales_list : List[np.ndarray]
        List of K non-negative images at each scale.
        scales_list[i] has shape (s₀/rᵢ, s₁/rᵢ, ..., sₙ₋₁/rᵢ)
    stats : dict
        Optimization statistics:
        - 'history': List of per-iteration loss components
        - 'final_error': Final reconstruction MSE
        - 'best_error': Best reconstruction MSE achieved
        - 'best_max_abs_error': Best maximum absolute error achieved (quality guarantee metric)
        - 'converged': Boolean indicating if convergence criterion was met
        - 'best_iteration': Iteration where best result was achieved
        - 'actual_iters': Actual number of iterations run (may be less than n_iters if converged early)
        - 'energy_distribution': Fraction of total energy per scale
        - 'scales': Scale factors used
        - 'time_seconds': Total optimization time
        - 'interpolation': Interpolation mode used ('nearest', 'linear', or 'cubic')
        - 'movie_frames': Dict with movie data (if napari_movie=True), or None
            - 'target': List of target frames
            - 'reconstruction': List of reconstruction frames
            - 'residual': List of residual frames
            - 'scales': List of lists (one list of scale components per frame)
            - 'iterations': List of iteration numbers

    Examples
    --------
    >>> import numpy as np
    >>> from luxar.gsplats.multiscale import decompose_image
    >>>
    >>> # 2D example
    >>> V = np.random.rand(256, 256)
    >>> scales_list, stats = decompose_image(V, scales=[1, 2, 4])
    >>> print([s.shape for s in scales_list])
    [(256, 256), (128, 128), (64, 64)]
    >>>
    >>> # 3D example
    >>> V = np.random.rand(128, 128, 128)
    >>> scales_list, stats = decompose_image(V, scales=[1, 2, 4, 8])
    >>> print(f"Energy distribution: {stats['energy_distribution']}")
    Energy distribution: [0.62, 0.23, 0.11, 0.04]

    Notes
    -----
    - Uses coarse initialization by default (strongly recommended for all use cases)
    - Four initialization methods available: "coarse" (default and best), "pyramid", "uniform", "finest"
    - All output images are guaranteed non-negative
    - Reconstruction: V ≈ Σₖ upsample(scales_list[k])
    - Higher alpha values push more energy to coarse scales
    """
```

## Algorithm Pseudocode

```python
# Initialization
model = MultiScaleDecomposer(shape=V.shape, scales=[1, 2, 4, 8])

# Choose initialization method
if init_method == "coarse":  # Default and strongly recommended
    model.initialize_coarse(V)
elif init_method == "pyramid":
    model.initialize_from_pyramid(V)
elif init_method == "uniform":
    model.initialize_uniform(V)
elif init_method == "finest":
    model.initialize_finest_scale(V)

optimizer = Adam(model.parameters(), lr=0.01)

# Optimization loop
for iteration in range(n_iters):
    # Forward pass
    scales_list, upsampled_list, reconstruction = model()

    # Compute losses
    L_recon = L1(reconstruction, V)  # L1 loss is default (not MSE)
    L_energy = sum(alpha^k * sum(scales_list[k]) for k in range(K)) / sum(V)
    L_total = L_recon + energy_weight * L_energy

    # Backward pass
    L_total.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Logging
    if verbose:
        energy_dist = [sum(s) / sum(V) for s in scales_list]
        print(f"Iter {iteration}: recon={L_recon:.5f}, energy={energy_dist}")

# Return final decomposition
return [s.detach().cpu().numpy() for s in scales_list]
```

## Usage Patterns

### Pattern 1: Simple Decomposition

```python
from luxar.gsplats.multiscale import decompose_image
import numpy as np

# Load image
V = np.load("data.npy")  # Shape: (256, 256, 128)

# Decompose into 4 scales
scales_list, stats = decompose_image(
    V,
    scales=[1, 2, 4, 8],
    n_iters=500
)

# Use scale components
V_full = scales_list[0]      # Full resolution: (256, 256, 128)
V_half = scales_list[1]      # Half resolution: (128, 128, 64)
V_quarter = scales_list[2]   # Quarter resolution: (64, 64, 32)
V_eighth = scales_list[3]    # Eighth resolution: (32, 32, 16)
```

### Pattern 2: Multi-Scale Gaussian Splat Fitting

**Note**: Multi-scale Gaussian splat fitting is now specified in the main gsplats package. Please refer to:

**[Main SPECIFICATIONS.md](../SPECIFICATIONS.md)** → Section 6: Multi-Scale Gaussian Splat Fitting

This section provides comprehensive specification for:
- Mathematical formulation and parameter scaling rules
- Complete API design for `fit_multiscale_gaussian_splats()`
- Thin wrapper architecture using `fit_gaussian_splats()` as building block
- Computational complexity analysis and expected speedups
- Implementation details and usage examples

The multi-scale decomposition provided by this package (`decompose_image()`) is used as a building block for the multi-scale fitting feature in the main gsplats package.

### Pattern 3: Custom Loss Weights

```python
# Aggressive coarse preference
scales_list, _ = decompose_image(
    V,
    energy_weight=0.01,   # Strong energy penalty
    alpha=2.5             # Steep penalty growth
)

# Conservative (closer to equal distribution)
scales_list, _ = decompose_image(
    V,
    energy_weight=0.0001,  # Weak energy penalty
    alpha=1.2              # Gentle penalty growth
)
```

### Pattern 4: Initialization Method Selection (5 Methods Available)

```python
# Coarse initialization (default and STRONGLY RECOMMENDED)
# Empirically the best method for all image types
scales_list, _ = decompose_image(
    V,
    init_method="coarse",  # Default - best convergence and quality
    energy_weight=0.001
)

# Other methods available for experimentation (generally inferior):

# Pyramid initialization
# Natural frequency decomposition
scales_list, _ = decompose_image(
    V,
    init_method="pyramid",
    n_iters=500
)

# Uniform initialization
# Balanced starting point
scales_list, _ = decompose_image(
    V,
    init_method="uniform",
    n_iters=500
)

# Finest initialization
# Mainly for visualizing energy redistribution
scales_list, _ = decompose_image(
    V,
    init_method="finest",
    n_iters=1000,      # Requires more iterations
    napari_movie=True  # Watch energy flow from fine to coarse
)
```

## Expected Behavior

### Energy Distribution Goals

For well-behaved decompositions with 4 scales [1, 2, 4, 8]:

**Good distribution** (alpha=1.5, energy_weight=0.001):
- Scale 8x (coarsest): ~50-70% of total energy
- Scale 4x: ~20-30%
- Scale 2x: ~10-15%
- Scale 1x (finest): ~5-10%

**Failure mode** (insufficient penalty):
- All scales < 10% except finest scale ~90%
- Indicates trivial solution, increase alpha or energy_weight

### Reconstruction Quality

Target final reconstruction error:
- **MSE < 1e-4** for normalized [0,1] images
- **Relative error < 1%** compared to input

If reconstruction error is poor (>1e-3), increase n_iters or decrease energy_weight.

### Convergence

Typical convergence behavior:
1. **Iterations 1-50**: Rapid reconstruction loss decrease
2. **Iterations 50-200**: Energy redistribution toward coarse scales
3. **Iterations 200-500**: Fine-tuning, gradual improvement

## Convergence Tracking and Quality Guarantee

### Auto-Convergence

The decomposition implements automatic convergence detection based on maximum absolute error:

**Convergence Criterion**:
```python
converged = max|reconstruction - target| < threshold
```

**Threshold Selection**:
- **Auto-mode (default)**: `threshold = 0.01 × (V.max() - V.min())`
  - Adapts to image scale (1% of value range)
  - For uniform images (all same value): `threshold = 0.01 × max(mean(|V|), 1e-6)`
- **Manual mode**: User specifies `max_abs_error_threshold`

**Benefits**:
1. **Saves computation**: Stops when target quality achieved
2. **Consistent quality**: Threshold adapts to data scale
3. **Handles edge cases**: Uniform images get sensible thresholds

### Best State Tracking (Quality Guarantee)

The optimization tracks and restores the **best** result seen during optimization, not the final iteration. This provides a quality guarantee even if optimization overshoots or becomes unstable.

**Implementation**:
```python
# During optimization
best_max_abs_error = infinity
best_state = None

for iteration in range(n_iters):
    # Forward pass
    reconstruction = model()
    current_max_abs_error = max|reconstruction - target|

    # Track best state based on max absolute error
    if current_max_abs_error < best_max_abs_error:
        best_max_abs_error = current_max_abs_error
        best_iteration = iteration

        # Save state (deep copy)
        best_state = {
            "raw_images": [p.detach().clone() for p in model.raw_images],
            "scales_list": [s.detach().clone() for s in scales_list],
            "reconstruction": reconstruction.detach().clone(),
            "iteration": iteration,
            "max_abs_error": current_max_abs_error,
            "recon_loss": reconstruction_loss
        }

    # Check for convergence
    if current_max_abs_error < threshold:
        break  # Early stopping

# Restore best state before returning
if best_state is not None:
    for param, best_param in zip(model.raw_images, best_state["raw_images"]):
        param.data.copy_(best_param)
```

**Key Features**:
1. **Quality guarantee**: Always returns best result, not final iteration
2. **Prevents regression**: If optimization overshoots, restores earlier state
3. **Smart logging**: Only logs significant improvements (> 5% reduction)
   - Uses threshold: `current_error < previous_best * 0.95` (5% improvement)
   - Prevents log spam during fine-tuning phase
   - Always logs first 10 iterations for debugging
4. **Deep copies**: Saves state without interfering with gradient computation

**Statistics Returned**:
- `converged`: Boolean indicating if convergence criterion was met
- `best_iteration`: Iteration where best result was achieved
- `actual_iters`: Actual number of iterations run (may be less if converged early)
- `best_max_abs_error`: Best maximum absolute error achieved
- `best_error`: Best reconstruction loss achieved
- `final_error`: Final reconstruction MSE (from best result)

### Early Stopping Example

```python
scales_list, stats = decompose_image(
    V,
    scales=[1, 2, 4],
    n_iters=1000,  # Maximum iterations
    # max_abs_error_threshold=None  # Default: auto (1% of range)
)

if stats['converged']:
    print(f"✓ Converged after {stats['actual_iters']} iterations")
    print(f"  Best max error: {stats['best_max_abs_error']:.6f}")
    print(f"  Best iteration: {stats['best_iteration']}")
else:
    print(f"⚠ Did not converge after {stats['actual_iters']} iterations")
    print(f"  Best max error achieved: {stats['best_max_abs_error']:.6f}")
```

### Edge Cases Handled

1. **Uniform images**: Uses 1% of mean absolute value when value range is zero
2. **Zero iterations**: Handles case where n_iters=0 (returns initialization)
3. **Convergence on first iteration**: Properly handles immediate convergence
4. **Numerical stability**: All comparisons use stable floating-point arithmetic

## Validation & Testing

### Unit Tests

1. **test_decomposition_basic.py**:
   - Non-negativity constraint
   - Reconstruction accuracy
   - Output shapes
   - Device handling (CPU/CUDA/MPS)

2. **test_decomposition_nd.py**:
   - 2D images
   - 3D volumes
   - 4D+ data
   - Edge cases (small images, single scale)

3. **test_energy_distribution.py**:
   - Verify energy moves to coarse scales
   - Alpha parameter effect
   - Energy weight parameter effect

4. **test_initialization.py** (integrated in test_decomposition_basic.py):
   - All five initialization methods: coarse (default), pyramid, uniform, finest, zero
   - Energy distribution verification for each method
   - Convergence properties comparison
   - Numerical stability of initialization

### Integration Tests

1. **test_multiscale_fitting.py**:
   - Full pipeline: decompose → fit splats → combine
   - Verify splat count scaling
   - Computational efficiency gains

### Demos

1. **demo_decompose_2d.py**:
   - Astronaut image decomposition
   - Visualization of scale components
   - Energy distribution plots

2. **demo_decompose_mitosis.py**:
   - Human mitosis biological histology data
   - Shows cellular feature distribution across scales
   - Demonstrates scale-specific biological structures

3. **demo_decompose_3d.py**:
   - 3D volume decomposition
   - Napari visualization
   - Per-scale energy analysis

4. **demo_multiscale_fitting.py** (future):
   - Complete multi-scale fitting pipeline
   - Comparison with single-scale baseline
   - Performance benchmarks

## Performance Considerations

### Memory Usage

For image of size S^d at K scales:
- Model parameters: O(S^d + (S/2)^d + (S/4)^d + ...) ≈ O(S^d)
- Gradients: Same as parameters
- Intermediate tensors: O(K × S^d) for upsampled versions

**Typical**: 256³ volume with 4 scales ≈ 2-3 GB GPU memory

### Computational Cost

Per iteration:
- Forward pass: O(K × S^d) for upsampling
- Loss computation: O(S^d) for reconstruction + O(sum of scales) for energy
- Backward pass: O(K × S^d)

**Typical**: 256³ volume, 500 iterations ≈ 1-5 minutes on GPU

### Optimization Tips

1. **Start with few scales**: [1, 2, 4] before adding [1, 2, 4, 8]
2. **Reduce iterations for large volumes**: 300-500 usually sufficient
3. **Use CUDA**: 10-50× speedup over CPU
4. **Lower precision**: `torch.float16` can reduce memory (experimental)

## Optimization Movie Visualization

### Purpose

The optimization movie feature records target, reconstruction, and residual frames during decomposition to visualize convergence behavior over time.

### Usage

```python
from luxar.gsplats.multiscale import decompose_image, show_optimization_movie

# Enable movie recording
scales_list, stats = decompose_image(
    V,
    napari_movie=True,      # Enable recording
    movie_every=5,          # Record every 5 iterations
    movie_max_frames=100    # Max 100 frames (prevents unbounded memory)
)

# Display movie in napari
if stats['movie_frames'] is not None:
    # Pass interpolation mode from stats to ensure movie matches optimization
    interpolation = stats.get('interpolation', 'cubic')
    show_optimization_movie(stats['movie_frames'], V.shape, interpolation=interpolation)
```

### Movie Data Structure

```python
movie_frames = {
    "target": [frame_0, frame_1, ...],           # Target image (constant, one per frame)
    "reconstruction": [frame_0, frame_1, ...],   # Reconstruction at each iteration
    "residual": [frame_0, frame_1, ...],        # Absolute residual at each iteration
    "scales": [[scale_0_0, scale_1_0, ...],     # Nested list: outer=frames, inner=scales
               [scale_0_1, scale_1_1, ...], ...], # Each inner list contains K numpy arrays
    "iterations": [iter_0, iter_1, ...]         # Iteration numbers (one per frame)
}

# Note: Each scale component in "scales" is stored as a numpy array at its native resolution
# (not upsampled). The show_optimization_movie() function upsamples them for visualization.
# This saves memory while preserving the ability to visualize individual scale evolution.
```

### Memory Management

- `movie_max_frames` limits total frames stored (default: 10000)
- FIFO removal: oldest frames discarded when limit reached
- Each frame stores full resolution images as numpy arrays
- Memory per frame ≈ `V.size * 4 bytes * (3 + K)` where K = number of scales
  - 3 = target + reconstruction + residual
  - K scale components (each upsampled to full resolution for visualization)

**Example memory usage:**
- 256² image, 100 frames, 4 scales ≈ 45 MB
- 256³ volume, 100 frames, 4 scales ≈ 11 GB

### Visualization Features

The `show_optimization_movie()` function displays:
- Time slider for scrubbing through optimization
- Multiple synchronized layers: target, reconstruction, all scale components, residual
- Individual scale components (hidden by default, toggle to view)
- Shared contrast limits for fair comparison across layers
- Iteration metadata overlay
- Scale evolution over time (see how energy shifts between scales)

## Future Extensions

### Potential Enhancements (Not in Initial Version)

1. **Adaptive scale selection**: Automatically determine optimal scales
2. **Gradient-based routing**: TV loss for frequency separation
3. **Learned upsampling**: Replace fixed interpolation with learned filters
4. **Perceptual losses**: Use feature-based losses for better quality
5. **Anisotropic decomposition**: Different scale factors per axis
6. **Sparse priors**: L1 penalties on scale components for sparsity
7. **Movie: Per-scale visualization**: Add individual scale components to movie

## References

### Related Work

1. **Laplacian Pyramids**: Burt & Adelson (1983) - Inspiration for multi-scale decomposition
2. **Gaussian Splatting**: Original work on splat-based reconstruction
3. **Scale-Space Theory**: Koenderink, Lindeberg - Mathematical foundation for multi-scale analysis

## Glossary

**Note**: For general Gaussian splatting terminology, see [Main GLOSSARY.md](../GLOSSARY.md)

**Multi-scale specific terms**:

- **Scale factor (r)**: Downsampling ratio (r=2 means half resolution, r=4 means quarter resolution)
- **Energy**: Sum of all pixel/voxel intensities in an image: ∫ V dx
- **Coarse scale**: Low-resolution component with high scale factor (e.g., r=8)
- **Fine scale**: High-resolution component with low scale factor (typically r=1, full resolution)
- **Upsampling**: Interpolation from low resolution to high resolution
- **Downsampling**: Reduction from high resolution to low resolution (e.g., averaging, pooling)
- **Total Variation (TV)**: Sum of absolute gradients, measures high-frequency content
- **Interpolation modes**: `'nearest'` (fastest), `'linear'` (smooth), `'cubic'` (highest quality, default)

## Version History

- **v0.1.0** (2025-01-XX): Initial specification with hierarchical energy loss
  - Basic decomposition with softplus constraints
  - Gaussian pyramid initialization
  - MSE reconstruction + energy penalty losses
  - Support for nD images (n ≥ 2)
