# Gaussian Splat Models Specification

**Version**: 1.0.0
**Last Updated**: 2025-11-27

## Overview

The models package contains PyTorch model implementations and rendering engines for n-dimensional oriented Gaussian splatting. The main components are:

1. **GaussianSplatModel**: PyTorch nn.Module for optimizable Gaussian splat parameters
2. **Rendering Core**: High-performance rendering engine with 2D/3D fast paths
3. **Rendering Wrappers**: Convenience functions for numpy/torch interfaces
4. **Utility Functions**: Numerical stability helpers (inverse softplus, LT solver)

**Prerequisite Reading**: See [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) for complete model specification

**Related Specifications**:
- **Complete Model Specification**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2: Gaussian Splat Model
- **Rendering Algorithm**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2: Rendering Function
- **Optimizers**: [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md)
- **Matrix Utilities**: [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md)
- **Metal Backend**: [gsplats/metal/SPECIFICATIONS.md](gsplats/metal/SPECIFICATIONS.md) - Apple Silicon GPU acceleration

## Package Structure

```
models/
├── gsplats/
│   ├── gsplat_model.py         # Main PyTorch model class
│   ├── rendering_core.py       # Core rendering engine
│   ├── rendering_wrappers.py   # NumPy/PyTorch wrappers
│   ├── __init__.py
│   └── metal/                  # Apple Silicon GPU acceleration
│       ├── gsplat_model_metal.py  # Metal-accelerated model
│       ├── SPECIFICATIONS.md      # Metal backend specification
│       ├── README.md              # Installation and usage
│       └── src/
│           ├── kernels.metal      # Metal compute shaders
│           └── bindings.mm        # C++ dispatcher
└── utils/
    ├── inverse_softplus.py     # Stable inverse softplus
    ├── lt_solver.py            # Lower-triangular solver compatibility wrapper
    └── __init__.py
```

## Detailed Specifications

**Note**: The comprehensive specification for Gaussian Splat models and rendering is documented in:

**`/packages/luxar/src/luxar/gsplats/SPECIFICATIONS.md`** → **Section 2: Gaussian Splat Model**

This includes:
- Complete mathematical formulation
- Parameter parameterizations (centers, covariances, amplitudes, sharpness)
- Rendering algorithm with AABB truncation
- Fast paths for 2D/3D with explicit forward substitution
- Memory management and chunking
- Sharpness feature specification (generalized Gaussians)

Please refer to the main SPECIFICATIONS.md file for complete implementation details.

## Summary of Key Components

### 1. GaussianSplatModel (`gsplats/gsplat_model.py`)

**Purpose**: PyTorch model representing a collection of oriented Gaussian functions.

**Parameters**:
- `raw_mu`: Centers in logit space → `sigmoid(raw_mu)` gives normalized coordinates
- `raw_L_diag`: Diagonal Cholesky elements via softplus → positive definiteness
- `L_off`: Off-diagonal Cholesky elements (unconstrained)
- `raw_a`: Amplitudes via softplus → non-negativity
- `sharpness_offsets_raw`: Sharpness offsets → `s = 2 * exp(s')` for generalized Gaussians

**Key Methods**:
```python
def current_params() -> Tuple[centers, Ls, amps, sharpness]:
    """Extract transformed parameters from raw learnable params."""

def forward() -> torch.Tensor:
    """Render all splats to output image/volume."""

def prune_(keep_mask: torch.Tensor) -> None:
    """Remove splats by boolean mask."""

def append_(centers_new, Ls_new, amps_new, sharpness_new) -> None:
    """Add new splats to the model."""

def replace_with(centers, Ls, amps, sharpness) -> None:
    """Replace all splats (complete reset)."""

def n_splats() -> int:
    """Return current number of splats."""
```

**Initialization**:
- Centers: Transformed to logit space to constrain within image bounds
  - Normalized coordinates clamped to [1e-6, 1-1e-6] to avoid sigmoid saturation
  - `raw_mu = log(u) - log(1-u)` where `u = centers / max(shape-1, 1)`
- Cholesky factors: Initialized with minimum diagonal constraints
  - Uses `stable_inverse_softplus` to convert initial diagonals to raw parameters
  - Off-diagonal elements directly initialized (unconstrained)
- Amplitudes: Initialized using `stable_inverse_softplus` from initial values
- Sharpness: Initialized to 0 → `s = 2 * exp(0) = 2` (standard Gaussian)

**Device Selection**:
- Auto-detects best device: CUDA → CPU
- MPS supported via standard PyTorch path
- **Metal backend** (Apple Silicon): For 3D volumes on MPS, use `GaussianSplatModelMetal` for 10-50× speedup. See [metal/SPECIFICATIONS.md](gsplats/metal/SPECIFICATIONS.md)
- Explicitly set via `device` parameter if needed

**Parameter Constraints**:
- Optional `sigma_max_diag`: Maximum diagonal values to prevent over-smoothing
- Applied via `torch.minimum(diag, sigma_max_diag)` in `_build_L()`

**Sharpness Clamping**:
- In `current_params()`, sharpness offsets clamped to `[-2.5, 2.5]` for stability
- This bounds actual sharpness to `s ∈ [0.164, 24.47]`
- Prevents numerical overflow in exponential while allowing wide variation

**Dynamic Operations**:
- `_to_internal_params()`: Converts external (μ, L, a, s) to raw learnable parameters
  - Handles inverse transformations: logit, inverse_softplus, log(s/2)
  - Clamps values to avoid log(0) and other numerical issues
  - Defaults sharpness to s=2 if not provided
- All dynamic operations (prune, append, replace) preserve sharpness parameter
- Empty splat additions (numel==0) are safely handled (no-op)

### 2. Rendering Core (`gsplats/rendering_core.py`)

**Purpose**: High-performance rendering engine with specialized fast paths.

**Main Function**:
```python
def render_gaussians(
    shape: Sequence[int],
    centers: torch.Tensor,      # (N, d)
    Ls: torch.Tensor,           # (N, d, d) lower-triangular
    amps: torch.Tensor,         # (N,)
    sharpness: torch.Tensor,    # (N,)
    truncate: float = 3.0,
    intensity_floor: float = 1e-5,
    chunk_size: Optional[int] = None
) -> torch.Tensor:
    """
    Render Gaussian splats to output image/volume.

    Algorithm:
    1. Compute sharpness-adjusted AABB per splat
    2. Optional amplitude-aware AABB shrinking
    3. Group splats by AABB shape for grid reuse
    4. For each group:
       a. Generate coordinate grid
       b. Process in memory chunks
       c. Solve L*y = (x-μ) (avoid matrix inversion)
       d. Apply generalized Gaussian: exp(-0.5 * ||y||^(s/2))
       e. Accumulate into output
    """
```

**Fast Paths**:
- `_render_gaussians_2d()`: Explicit 2×2 forward substitution
- `_render_gaussians_3d()`: Explicit 3×3 forward substitution
- 10-50× faster than generic nD path due to explicit operations
- Auto-selected based on dimensionality (d==2 or d==3)

**Memory Management**:
```python
def calculate_optimal_chunk_size(K, d, device, dtype) -> int:
    """
    Calculate optimal chunk size based on available memory.

    Memory Estimates:
    - CUDA: 60% of torch.cuda.mem_get_info(), fallback 2GB if unavailable
    - MPS: 4GB conservative estimate (unified memory)
    - CPU: 8GB conservative estimate

    Memory Footprint Calculation:
    - Per-element footprint: K * (2*d + 2) * bytes_per_element
    - Accounts for: delta (K,d,P), y (K,d,P), expo (K,P), vals (K,P)
    - Includes 1.5x overhead factor for intermediate tensors

    Returns:
    - Clamped to [1024, 1048576] for kernel efficiency
    - Fallback: 131072 if calculation fails or seems unreasonable
    """
```

**Grid Caching**:
- Process-wide cache `_GRID_CACHE` for base grids and linear offsets
- Keyed by `(device.type, str(dtype), tuple(strides), tuple(box_shape))`
- Returns cached `(base, lin_offsets)` for AABB shapes seen before
- Eliminates redundant meshgrid computations for splats with same AABB dimensions
- `base`: (d, P) coordinates in [0, h_i-1] for each dimension
- `lin_offsets`: (P,) row-major flat indices for indexing into output

**Helper Functions**:

```python
def linear_strides(shape: Sequence[int], device) -> torch.Tensor:
    """
    Compute row-major linear strides for nD tensor.

    Returns: (d,) long tensor where stride[i] = product(shape[i+1:])
    Example: shape (10, 20, 30) → strides [600, 30, 1]
    """

def group_by_box(lo: torch.Tensor, hi: torch.Tensor) -> Dict[Tuple[int, ...], torch.Tensor]:
    """
    Group splats by AABB shape using torch.unique on GPU.

    Returns: Dict mapping box_shape tuple → tensor of splat indices
    Minimizes GPU-CPU synchronization by doing grouping on GPU
    Only transfers small unique array to CPU for dict keys
    """

def group_by_box_gpu(lo: torch.Tensor, hi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-friendly grouping by AABB size (returns uniq_sizes, inv).

    MPS Compatibility: Falls back to CPU for torch.unique since MPS
    doesn't support unique with dim argument. Returns tensors on original device.
    """
```

**Validation and Edge Cases**:
- After AABB calculation, validates `hi > lo` for all dimensions
- Filters out splats with invalid AABBs (empty boxes)
- Early return with zeros if no valid splats remain
- Safely handles empty input (N=0 splats)

**Power Function Clamping**:
- In sharpness application: `expo_safe = torch.clamp(expo, min=1e-10)`
- Prevents `log(0)` in gradients of `pow(expo, s/2)` for backpropagation
- Ensures numerical stability during optimization

### 3. Sharpness Feature (Generalized Gaussians)

**Mathematical Formulation**:
- Standard Gaussian: `I(x) = a * exp(-0.5 * ||y||²)`
- Generalized Gaussian: `I(x) = a * exp(-0.5 * ||y||^s)` where `s` is sharpness
- Computational form: `exp(-0.5 * expo^(s/2))` where `expo = ||y||²`

**Sharpness Parameter s**:
- `s = 2`: Standard Gaussian (smooth exponential falloff)
- `s > 2`: Sharper edges, more compact support
- `s < 2`: Softer edges, heavier tails
- `s → ∞`: Approaches box function
- `s → 0`: Approaches uniform

**Exponential Parameterization**:
```
s = 2 * exp(s')
```
where `s'` is the learned **sharpness offset** parameter.

**Benefits**:
1. Zero-centered learning: `s' = 0` → `s = 2` (standard Gaussian default)
2. Symmetric exploration: Can increase/decrease sharpness from baseline
3. Always positive: `s > 0` guaranteed
4. L1 regularization friendly: Encourages standard Gaussians unless beneficial
5. Smooth gradients: Exponential provides stable optimization

**Inverse Transformation**:
```
s' = log(s / 2)
```

**Practical Bounds**:
- Raw parameter: `s' ∈ [-2.5, 2.5]` (clamped in `current_params()`)
- Actual sharpness: `s ∈ [0.164, 24.47]`
- Wide range while preventing numerical issues

**AABB Adjustment**:
For generalized Gaussian, effective radius is adjusted:
```
effective_truncate = truncate^(2/s)
```

**Rationale**: For same intensity threshold as standard Gaussian:
- `s=2.0` → `3^1 = 3` (unchanged)
- `s=1.5` → `3^1.33 ≈ 4.73` (larger for soft splats)
- `s=3.0` → `3^0.67 ≈ 2.08` (smaller for sharp splats)

**Amplitude-Aware Shrinking**:
When `intensity_floor > 0`, compute threshold per splat:
```
# Solve: a * exp(-0.5 * t^s) = intensity_floor
# Result: t = (2 * log(a / intensity_floor))^(1/s)
tmax = torch.pow(log_ratio, 1.0 / sharpness)
```
Then shrink AABB radii to minimum of truncate-based and amplitude-based limits.

### 4. Rendering Wrappers (`gsplats/rendering_wrappers.py`)

**Purpose**: User-friendly interfaces for numpy and torch tensors using `GSplatData` objects.

**API Change (v1.1.0, Nov 2025)**: Rendering wrappers now accept `GSplatData` dataclass instead of raw parameter arrays for type safety and clarity.

**render_gaussians_numpy**:
```python
def render_gaussians_numpy(
    shape: Sequence[int],
    result: GSplatData,  # Gaussian splat parameters
    truncate: float = 3.0,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    NumPy interface for rendering Gaussian splats.

    Parameters:
        shape: Output volume dimensions
        result: GSplatData from fit_gaussian_splats() or loaded from disk
        truncate: Truncation radius in standard deviations (default: 3σ)
        chunk_size: Process in chunks for memory efficiency (default: auto)

    Returns: NumPy array with shape `shape`, dtype float32

    Example:
        result = fit_gaussian_splats(volume)
        reconstruction = render_gaussians_numpy(
            shape=volume.shape,
            result=result,
            truncate=3.0
        )
    """
```

**render_gaussians_torch**:
```python
def render_gaussians_torch(
    shape: Sequence[int],
    result: GSplatData,  # Gaussian splat parameters
    truncate: float = 3.0,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None
) -> torch.Tensor:
    """
    PyTorch interface for rendering Gaussian splats.

    Parameters:
        shape: Output volume dimensions
        result: GSplatData object
        truncate: Truncation radius in standard deviations
        device: Target device (default: auto-detect CUDA→CPU)
        chunk_size: Batch size for memory efficiency

    Returns: Torch tensor on specified device

    Example:
        result = fit_gaussian_splats(volume)
        reconstruction = render_gaussians_torch(
            shape=(128, 128, 128),
            result=result,
            device=torch.device('cuda')
        )
    """
```

**render_gaussians_batched**:
```python
def render_gaussians_batched(
    shape: Sequence[int],
    result: GSplatData,  # Gaussian splat parameters
    model: GaussianSplatModel,    # Pre-initialized model
    truncate: float = 3.0,
    batch_size: int = 100,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Batch rendering for large datasets (GPU memory constrained).

    Parameters:
        shape: Output volume dimensions
        result: GSplatData object
        model: Pre-constructed GaussianSplatModel (avoids recreation per batch)
        truncate: Truncation radius in standard deviations
        batch_size: Splats per batch (default: 100)
        device: Target device

    Returns: Torch tensor, accumulated across batches

    Example:
        from luxar.gsplats.models import GaussianSplatModel

        model = GaussianSplatModel(n_splats=len(result.centers), d=result.centers.shape[1])
        reconstruction = render_gaussians_batched(
            shape=(256, 256, 256),
            result=result,
            model=model,
            batch_size=100
        )

    Note: Requires pre-initialized model for efficiency.
    """
```

**GSplatData Structure**:
```python
@dataclass
class GSplatData:
    centers: np.ndarray          # (N, d) float32 - Gaussian centers
    amplitudes: np.ndarray       # (N,) float32 - Amplitudes
    cholesky_factors: np.ndarray # (N, d*(d+1)/2) float32 - Packed Cholesky factors
    sharpnesses: np.ndarray      # (N,) float32 - Sharpness values (generalized Gaussian)
    stats: Dict[str, Any]        # Fitting statistics (loss history, convergence, etc.)
```

**Persistence** (see `gsplat_data.py`):
```python
# Save result to .gsplats.zarr format
result.save('fitted.gsplats.zarr')

# Load result
from luxar.gsplats import GSplatData
result = GSplatData.load('fitted.gsplats.zarr')
```

**Migration from Old API** (deprecated Nov 2025):
```python
# OLD API (no longer supported):
render_gaussians_numpy(shape, params_full, amps)

# NEW API:
result = fit_gaussian_splats(volume)  # Returns GSplatData
render_gaussians_numpy(shape, result)
```

### 5. Numerical Stability Utilities

**Stable Inverse Softplus** (`utils/inverse_softplus.py`):
```python
def stable_inverse_softplus(y: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Compute inverse softplus: x such that softplus(x) = y.

    Mathematical relationship:
        softplus(x) = (1/beta) * log(1 + exp(beta*x))
        inverse_softplus(y) = (1/beta) * log(exp(beta*y) - 1)
                            = (1/beta) * log(expm1(beta*y))  # stable form

    Algorithm:
    1. For large beta*y (>= 50): use asymptotic approximation y
    2. For normal values: use expm1 to avoid catastrophic cancellation
    3. Preserves input dtype (float32/float64)

    Warnings:
    - Issues RuntimeWarning if input contains non-positive values
    - Handles edge cases gracefully

    Returns: Same dtype as input
    """
```

**Lower Triangular Solver** (`utils/lt_solver.py`):
```python
def solve_lower_triangular(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Cross-version compatibility wrapper for PyTorch triangular solve.

    Modern PyTorch (>= 2.0): Uses torch.linalg.solve_triangular
    Legacy PyTorch (1.x): Uses torch.triangular_solve (returns tuple)

    Parameters:
    - L: (d, d) or (N, d, d) lower triangular matrices
    - B: (d, P) or (N, d, P) right-hand side

    Returns: Solution X such that L @ X = B

    Note: This solver is primarily for reference/testing.
    The main rendering code uses explicit forward substitution
    for better performance.
    """
```

**Clamping in Forward Substitution**:
```python
# 2D example - prevents division by near-zero diagonal elements
y0 = d0 / torch.clamp(l11, min=1e-12)
y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-12)

# 3D example
y0 = d0 / torch.clamp(l11, min=1e-12)
y1 = (d1 - l21 * y0) / torch.clamp(l22, min=1e-12)
y2 = (d2 - l31 * y0 - l32 * y1) / torch.clamp(l33, min=1e-12)
```

**Sigmoid Saturation Prevention**:
```python
# In GaussianSplatModel initialization
u0 = np.clip(centers0 / np.maximum(shape_arr - 1.0, 1.0), 1e-6, 1 - 1e-6)

# In current_params()
centers = u * torch.clamp(shape - 1.0, min=1.0)
```

## Rendering Algorithm Details

### AABB Truncation

**Purpose**: Avoid computing Gaussian contributions far from center (negligible).

**Implementation**:
1. Compute covariance diagonal: `sigma_diag = sum(L * L, dim=2)`
2. Calculate effective truncation: `effective_truncate = truncate^(2/s)` (per-splat)
3. Calculate radii: `radii = ceil(effective_truncate * sqrt(sigma_diag))`
4. Define AABB: `lo = clamp(floor(center - radii), min=0)`, `hi = min(ceil(center + radii) + 1, shape)`
5. Only render within AABB bounds

**Sharpness Adjustment**:
- Sharper splats (s>2) have smaller effective radius
- Softer splats (s<2) have larger effective radius
- Adjustment formula: `effective_truncate = truncate^(2/s)`

**Amplitude-Aware Shrinking** (optional, when `intensity_floor > 0`):
- Solves for radius where intensity drops to `intensity_floor`
- `tmax = (2 * log(a / intensity_floor))^(1/s)`
- Takes minimum of truncate-based and amplitude-based radii
- Prevents large AABBs for low-amplitude splats

**Edge Cases**:
- Handles splats near boundaries (clamps AABB to image bounds)
- Filters splats with invalid AABBs (`hi <= lo`)
- Returns zeros if no valid splats remain after filtering

### Grid Generation and Reuse

**Strategy**: Group splats by AABB shape to reuse coordinate grids.

**Algorithm**:
```python
# Step 1: Group by AABB dimensions
groups = group_by_box(lo, hi)  # Dict[box_shape -> splat_indices]

for box_shape, indices in groups.items():
    # Step 2: Generate or retrieve cached grid for this shape
    base, lin_offsets = cached_base_and_offsets(box_shape, strides, device)
    # base: (d, P) coordinates in [0, h_i-1]
    # lin_offsets: (P,) flat indices for output accumulation

    # Step 3: Compute base flat indices for each splat in group
    base_idx = (lo[indices] * strides).sum(dim=1)  # (K,)

    # Step 4: Process in memory chunks
    for p0 in range(0, P, P_chunk):
        p1 = min(P, p0 + P_chunk)
        base_chunk = base[:, p0:p1]  # (d, Pc)

        # Step 5: Compute deltas: x - μ (broadcast to all splats in group)
        delta = base_chunk[None, :, :] + lo_f[:, :, None] - mu[:, :, None]  # (K, d, Pc)

        # Step 6: Solve L*y = delta (batched)
        if d == 2:
            expo = fwd_norm2_2d(L, delta[0], delta[1])
        elif d == 3:
            expo = fwd_norm2_3d(L, delta[0], delta[1], delta[2])
        else:
            y = torch.linalg.solve_triangular(L, delta, upper=False)
            expo = torch.sum(y * y, dim=1)  # (K, Pc)

        # Step 7: Apply generalized Gaussian with sharpness
        expo_safe = torch.clamp(expo, min=1e-10)
        vals = torch.exp(-0.5 * torch.pow(expo_safe, s[:, None] / 2.0)) * a[:, None]

        # Step 8: Accumulate into output
        idx_flat = (base_idx[:, None] + lin_offsets[p0:p1][None, :]).reshape(-1)
        out_flat.index_add_(0, idx_flat, vals.reshape(-1))
```

**Performance Benefits**:
- Grid generation is expensive (meshgrid for all P points)
- Reusing grids for splats with same AABB shape provides significant speedup
- Typical datasets have many splats with identical or similar AABB dimensions
- Cache persists across forward passes (process-wide)

### Forward Substitution (2D Example)

**Problem**: Solve `L*y = delta` where L is lower-triangular.

**Explicit Solution**:
```python
# L = [[l11, 0  ],
#      [l21, l22]]

# Forward substitution (explicit)
y0 = delta[0] / torch.clamp(l11, min=1e-12)
y1 = (delta[1] - l21 * y0) / torch.clamp(l22, min=1e-12)

# Squared norm
expo = y0.mul(y0).add_(y1.mul(y1))  # y0^2 + y1^2 (fused multiply-add)
```

**3D Solution**:
```python
# L = [[l11, 0,   0  ],
#      [l21, l22, 0  ],
#      [l31, l32, l33]]

y0 = delta[0] / torch.clamp(l11, min=1e-12)
y1 = (delta[1] - l21 * y0) / torch.clamp(l22, min=1e-12)
y2 = (delta[2] - l31 * y0 - l32 * y1) / torch.clamp(l33, min=1e-12)

expo = y0.mul(y0).add_(y1.mul(y1)).add_(y2.mul(y2))
```

**Benefits**:
- No linalg kernel calls (avoids overhead)
- Explicit operations better for autograd (simpler computational graph)
- Fused multiply-add for norm computation
- 10-50× speedup vs generic solver for 2D/3D cases

**Generic nD Fallback**:
```python
# For d > 3, use PyTorch's triangular solver
try:
    y = torch.linalg.solve_triangular(L, delta, upper=False)
except Exception:
    # Legacy PyTorch fallback
    y, _ = torch.triangular_solve(delta, L, upper=False)

expo = torch.sum(y * y, dim=1)
```

## Testing Requirements

### Unit Tests

**gsplat_model.py**:
- Parameter initialization correctness (raw → transformed)
- Forward pass (rendering) produces correct output
- Dynamic operations (prune, append, replace) preserve correctness
- State dict serialization/deserialization works
- Sharpness parameter handling (default, explicit, dynamic ops)
- Device placement (CUDA, CPU, MPS)
- Bounds constraints (centers stay in bounds, diagonals positive)

**rendering_core.py**:
- 2D/3D fast paths produce identical results to generic path
- nD generic path correctness for various dimensions (4D, 5D, etc.)
- Memory chunking preserves correctness (compare chunked vs unchunked)
- Grid caching returns correct grids and offsets
- AABB calculation accuracy (various shapes, scales)
- Sharpness-adjusted AABB correctness (verify effective_truncate formula)
- Amplitude-aware shrinking correctness
- Edge cases: empty input, boundary splats, invalid AABBs

**rendering_wrappers.py**:
- Parameter unpacking correctness (with/without sharpness)
- Format detection (raises error on invalid column count)
- Backward compatibility (old format without sharpness)
- NumPy and PyTorch wrappers produce consistent results

**utils/inverse_softplus.py**:
- Numerical stability for large values (asymptotic approximation)
- Numerical stability for small values (expm1 usage)
- Round-trip consistency: `softplus(inverse_softplus(y)) ≈ y`
- Dtype preservation (float32, float64)
- Warning on non-positive input

**utils/lt_solver.py**:
- Correctness vs direct linalg.solve
- Compatibility with legacy PyTorch versions
- Batched solving (3D tensors)

### Integration Tests

**Complete Pipeline**:
- Model → current_params() → render_gaussians → Compare with ground truth
- Dynamic operations preserve rendering correctness:
  - Prune half splats → render → verify expected output
  - Append new splats → render → verify additive contribution
  - Replace all splats → render → verify complete reset
- Gradient flow through rendering:
  - Backward pass on render output
  - Verify gradients w.r.t. all parameters (centers, L, amps, sharpness)
  - Check gradient magnitudes are reasonable (not vanishing/exploding)

**Multi-Device Tests**:
- Same input produces same output on CPU, CUDA, MPS
- Memory management works on all devices
- Chunk size calculation reasonable for each device type

### Property Tests

**Mathematical Invariants**:
- Cholesky factors remain positive definite:
  - `diag(L) > 0` after `_build_L()`
  - `Σ = L @ L^T` is positive semi-definite
- Centers remain within image bounds:
  - `0 <= centers < shape` (approximately, with small epsilon)
- Amplitudes remain non-negative:
  - `amps >= 0` always
- Sharpness remains positive:
  - `sharpness > 0` always
  - Within expected bounds `[0.164, 24.47]` after clamping
- AABB bounds are valid:
  - `lo <= hi` for all dimensions
  - `lo >= 0` and `hi <= shape`

**Rendering Properties**:
- Additivity: `render([splats_A]) + render([splats_B]) ≈ render([splats_A + splats_B])`
  - (Approximate due to floating point, but should be close)
- Scaling: `render(amps * k) = k * render(amps)` for positive scalar k
- Translation: Moving splat center shifts rendered peak location
- Sharpness effect: Higher sharpness → more compact rendered splat

## Performance Characteristics

### Computational Complexity

**Per Forward Pass**:
- AABB calculation: `O(N_splats × d)` - diagonal computation and radius calculation
- Grouping: `O(N_splats × log(N_splats))` - torch.unique for GPU grouping
- Rendering per group: `O(K_group × P_box)` where:
  - `K_group` = number of splats in group
  - `P_box = product(box_shape)` = number of voxels in AABB
- Total: `O(N_splats × avg_P_box)` - dominates for typical workloads

**Memory Usage**:
- Model parameters: `O(N_splats × (d + d*(d+1)//2 + 1 + 1))`
  - Centers: `N × d`
  - Cholesky (diagonal): `N × d`
  - Cholesky (off-diagonal): `N × (d*(d-1)//2)`
  - Amplitudes: `N`
  - Sharpness: `N`
- Grid cache: `O(unique_box_shapes × max_P_box × d)`
  - Typically small (few unique shapes, cached persistently)
- Intermediate tensors (per chunk): `O(K_group × d × P_chunk)`
  - Delta: `K × d × P_chunk`
  - Y (solution): `K × d × P_chunk`
  - Expo: `K × P_chunk`
  - Vals: `K × P_chunk`
  - Total with overhead: ~1.5× base estimate

**Typical Workload Scaling**:
- 1K splats, 2D (512×512): ~100ms on GPU
- 10K splats, 3D (128×128×128): ~500ms on GPU
- Memory-limited by chunk size calculation (automatic)

### Optimization Tips

1. **Use fast paths**: Let renderer auto-detect 2D/3D for speedup
   - 10-50× faster than generic nD path
   - No manual intervention needed (automatic dispatch)

2. **Adjust chunk_size**: Smaller for memory-constrained devices
   - Default auto-calculation usually optimal
   - Manual override if experiencing OOM errors
   - Smaller chunks = slower but more memory-efficient

3. **Enable grid caching**: Already enabled by default
   - Cache persists across forward passes
   - Particularly beneficial when many splats have similar sizes

4. **Tune truncate**: Larger = more accurate but slower
   - Default 3.0 is good balance (99.7% of Gaussian mass)
   - Increase to 4.0 for higher accuracy (better tails)
   - Decrease to 2.0 for speed (acceptable for most cases)

5. **Use intensity_floor**: Enables amplitude-aware culling
   - Default 1e-5 prevents large AABBs for dim splats
   - Increase for more aggressive culling (faster, less accurate)
   - Set to 0 to disable (maximum accuracy, slower)

6. **Batch similar operations**: Dynamic splat modifications
   - Batch multiple prune/append operations when possible
   - Avoid frequent small modifications during optimization

## Extension Points

### Adding New Parameterizations

Modify `GaussianSplatModel.__init__()` and `current_params()` to support new parameter types:

```python
# Example: Adding anisotropic sharpness (per-axis)
def __init__(self, ..., init_sharpness_per_axis=None):
    # Initialize: (N, d) sharpness values instead of (N,)
    self.sharpness_offsets_raw = nn.Parameter(
        torch.zeros((N, d), dtype=torch.float32, device=device)
    )

def current_params(self):
    # Return: sharpness_per_axis instead of scalar sharpness
    sharpness_per_axis = 2.0 * torch.exp(
        torch.clamp(self.sharpness_offsets_raw, min=-2.5, max=2.5)
    )
    return centers, L, amps, sharpness_per_axis
```

Then update rendering core to handle per-axis sharpness.

### Adding New Rendering Modes

Implement specialized renderers following fast path pattern:

```python
def _render_gaussians_custom(shape, centers, Ls, amps, sharpness, ...):
    """
    Custom rendering mode (e.g., with different accumulation strategy).
    """
    device = centers.device
    out = torch.zeros(tuple(shape), dtype=torch.float32, device=device)

    # Custom AABB calculation
    # ...

    # Custom accumulation (e.g., max instead of sum)
    for each_splat:
        vals = compute_gaussian_values(...)
        out = torch.maximum(out, vals)  # Max accumulation

    return out
```

Add dispatch in main `render_gaussians()`:
```python
def render_gaussians(..., mode='sum'):
    if mode == 'sum':
        # existing implementation
    elif mode == 'max':
        return _render_gaussians_max(...)
```

### Adding New Activations

Replace softplus/sigmoid with custom activations in parameterization:

```python
# Example: Using ELU+1 instead of softplus for amplitudes
def current_params(self):
    # Original: amps = F.softplus(self.raw_a)
    # New: amps = F.elu(self.raw_a) + 1.0  # Ensures non-negativity

    # Corresponding initialization needs inverse:
    # raw_a0 = inverse_elu_plus_one(amps0)
```

Note: Ensure new activation has appropriate inverse for initialization.

## See Also

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2 - Complete model specification
- [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) - Matrix utilities (pack/unpack)
- [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md) - Optimizer integration
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and parameter naming

**External References**:
- PyTorch documentation: torch.linalg.solve_triangular
- NumPy documentation: expm1 (for numerical stability)
- Gaussian splatting literature (various papers on 3D reconstruction)

## Changelog

- **v1.0.0** (January 2025): Initial implementation
  - nD Gaussian splatting with full covariance
  - 2D/3D fast paths with explicit forward substitution
  - Per-splat sharpness (generalized Gaussians)
  - Memory-optimized chunking and grid caching
  - Dynamic operations support (prune/append/replace)
  - Cross-device support (CUDA, CPU, MPS)
