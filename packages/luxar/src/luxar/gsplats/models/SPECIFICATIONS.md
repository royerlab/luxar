# Gaussian Splat Models Specification

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

## Package Structure

```
models/
├── gsplats/
│   ├── gsplat_model.py         # Main PyTorch model class
│   ├── rendering_core.py       # Core rendering engine
│   ├── rendering_wrappers.py   # NumPy/PyTorch wrappers
│   └── __init__.py
└── utils/
    ├── inverse_softplus.py     # Stable inverse softplus
    ├── lt_solver.py            # Lower-triangular solver (if needed)
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

def append_(centers_new, Ls_new, amps_new, sharpness_new=None) -> None:
    """Add new splats to the model."""

def replace_with(centers, Ls, amps, sharpness=None) -> None:
    """Replace all splats (complete reset)."""

def n_splats() -> int:
    """Return current number of splats."""
```

**Initialization**:
- Centers: Transformed to logit space to constrain within image bounds
- Cholesky factors: Initialized with minimum diagonal constraints
- Amplitudes: Sampled from image at center locations
- Sharpness: Initialized to 0 (standard Gaussian, s=2)

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

**Memory Management**:
```python
def _calculate_optimal_chunk_size(K, d, device, dtype) -> int:
    """
    Calculate optimal chunk size based on available memory.

    CUDA: 60% of torch.cuda.mem_get_info(), fallback 2GB
    MPS: 4GB conservative estimate
    CPU: 8GB conservative estimate

    Clamps to [1024, 1048576] for kernel efficiency.
    """
```

**Grid Caching**:
- Process-wide cache for base grids and linear offsets
- Keyed by `(device, dtype, strides, box_shape)`
- Reuses grids for splats with same AABB shape

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

**AABB Adjustment**:
For generalized Gaussian, effective radius is adjusted:
```
effective_truncate = truncate^(2/s)
```

**Rationale**: For same intensity threshold as standard Gaussian:
- `s=2.0` → `3^1 = 3` (unchanged)
- `s=1.5` → `3^1.33 ≈ 4.73` (larger for soft splats)
- `s=3.0` → `3^0.67 ≈ 2.08` (smaller for sharp splats)

### 4. Numerical Stability

**Stable Inverse Softplus** (`utils/inverse_softplus.py`):
```python
def stable_inverse_softplus(y: Union[float, np.ndarray], beta: float = 1.0):
    """
    Compute inverse softplus: x such that softplus(x) = y.

    For beta*y >= 50: return y (asymptotic approximation)
    Otherwise: return log(expm1(beta*y)) / beta

    Avoids overflow/underflow for large values.
    """
```

**Clamping in Forward Substitution**:
```python
# Prevent division by near-zero diagonal elements
y0 = d0 / torch.clamp(l11, min=1e-12)
```

**Sigmoid Saturation Prevention**:
```python
# Clamp normalized coordinates to avoid gradient vanishing
u = np.clip(u, 1e-6, 1 - 1e-6)
```

## Rendering Algorithm Details

### AABB Truncation

**Purpose**: Avoid computing Gaussian contributions far from center (negligible).

**Implementation**:
1. Compute covariance diagonal: `sigma_diag = sum(L * L, dim=2)`
2. Calculate radii: `radii = ceil(effective_truncate * sqrt(sigma_diag))`
3. Define AABB: `lo = ceil(center - radii)`, `hi = floor(center + radii) + 1`
4. Only render within AABB bounds

**Sharpness Adjustment**:
- Sharper splats (s>2) have smaller effective radius
- Softer splats (s<2) have larger effective radius
- Adjustment formula: `effective_truncate = truncate^(2/s)`

### Grid Generation and Reuse

**Strategy**: Group splats by AABB shape to reuse coordinate grids.

**Algorithm**:
```python
groups = group_by_box(lo, hi)  # Dict[box_shape -> splat_indices]

for box_shape, indices in groups.items():
    # Generate grid once for this shape
    base, lin_offsets = _cached_base_and_offsets(box_shape, strides, device)

    # Process all splats with this AABB shape
    for splat_idx in indices:
        # Compute deltas: x - μ
        delta = base - centers[splat_idx][:, None]  # (d, P)

        # Solve L*y = delta
        if d == 2:
            expo = _fwd_norm2_2d(L[splat_idx], delta[0], delta[1])
        elif d == 3:
            expo = _fwd_norm2_3d(L[splat_idx], delta[0], delta[1], delta[2])
        else:
            expo = _fwd_norm2_nd(L[splat_idx], delta)

        # Apply generalized Gaussian
        vals = torch.exp(-0.5 * torch.pow(expo, sharpness[splat_idx] / 2.0)) * amps[splat_idx]

        # Accumulate
        out_flat.index_add_(0, lin_offsets, vals)
```

### Forward Substitution (2D Example)

**Problem**: Solve `L*y = delta` where L is lower-triangular.

**Explicit Solution**:
```python
# L = [[l11, 0  ],
#      [l21, l22]]

y0 = delta[0] / l11
y1 = (delta[1] - l21 * y0) / l22
expo = y0^2 + y1^2
```

**Benefits**:
- No linalg kernel calls (faster)
- Explicit operations better for autograd
- 10-50× speedup vs generic solver

## Testing Requirements

### Unit Tests

**gsplat_model.py**:
- Parameter initialization correctness
- Forward pass (rendering) correctness
- Dynamic operations (prune, append, replace)
- State dict serialization/deserialization
- Sharpness parameter handling

**rendering_core.py**:
- 2D/3D fast paths correctness
- nD generic path correctness
- Memory chunking correctness
- Grid caching correctness
- AABB calculation accuracy
- Sharpness-adjusted AABB correctness

**utils/**:
- Inverse softplus numerical stability
- Lower-triangular solver correctness

### Integration Tests

**Complete Pipeline**:
- Model → Render → Compare with ground truth
- Dynamic operations preserve correctness
- Gradient flow through rendering

### Property Tests

**Mathematical Invariants**:
- Cholesky factors remain positive definite
- Centers remain within image bounds
- Amplitudes remain non-negative
- Sharpness remains positive
- AABB bounds are valid (lo <= hi)

## Performance Characteristics

### Computational Complexity

**Per Forward Pass**:
- AABB calculation: `O(N_splats × d)`
- Grouping: `O(N_splats × log(N_splats))`
- Rendering per group: `O(K_group × P_box)` where P_box = product(box_shape)
- Total: `O(N_splats × avg_P_box)`

**Memory Usage**:
- Model parameters: `O(N_splats × (d + d*(d+1)//2 + 1 + 1))` (centers + Cholesky + amplitude + sharpness)
- Grid cache: `O(unique_box_shapes × max_P_box)`
- Intermediate tensors: `O(chunk_size × d)`

### Optimization Tips

1. **Use fast paths**: Let renderer auto-detect 2D/3D for speedup
2. **Adjust chunk_size**: Smaller for memory-constrained devices
3. **Enable grid caching**: Already enabled by default
4. **Tune truncate**: Larger = more accurate but slower

## Extension Points

### Adding New Parameterizations

Modify `GaussianSplatModel.__init__()` and `current_params()` to support new parameter types.

### Adding New Rendering Modes

Implement specialized renderers following fast path pattern:
```python
def _render_gaussians_custom(shape, centers, Ls, amps, sharpness, ...):
    # Custom rendering logic
    ...
```

### Adding New Activations

Replace softplus/sigmoid with custom activations in parameterization.

## See Also

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2 - Complete model specification
- [utils/SPECIFICATIONS.md](../utils/SPECIFICATIONS.md) - Matrix utilities (pack/unpack)
- [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md) - Optimizer integration
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and parameter naming

## Version History

- **v1.0.0** (January 2025): Initial implementation
  - nD Gaussian splatting with full covariance
  - 2D/3D fast paths with explicit forward substitution
  - Per-splat sharpness (generalized Gaussians)
  - Memory-optimized chunking and grid caching
  - Dynamic operations support (prune/append/replace)
