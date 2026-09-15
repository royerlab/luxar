# Gsplats Utilities

Utility functions for Gaussian splat parameter handling and optimization.

## Overview

This package provides low-level utilities for working with Gaussian splat parameters, particularly for packing/unpacking Cholesky factors and calculating gradient dilution compensation for nD optimization.

**Key Functions**:
- `tril_size(d)` - Calculate size of lower triangular matrix
- `pack_tril(L)` - Pack batch of lower triangular matrices to flat arrays
- `unpack_tril(v, d)` - Unpack flat arrays to batch of lower triangular matrices
- `calculate_gradient_dilution_factor(d)` - Compute gradient compensation for nD spaces
- `validate_cholesky_shape(cholesky_factors, ndim, ...)` - Validate packed Cholesky factor shapes
- `permute_cholesky_packed(packed, d, perm)` - Reorder dimensions of packed Cholesky factors
- `embed_cholesky_packed(packed, d_src, d_dst, dim_mapping, ...)` - Embed lower-dim Cholesky into higher-dim space
- `resolve_torch_device(device, use_cuda=True, use_metal=True)` - Shared PyTorch device auto-selection helper; `"auto"` is equivalent to `None` **(needs `luxar[gsplats]` — see below)**
- `is_mps_available()` - Robustly detect a working Apple Metal (MPS) backend **(needs `luxar[gsplats]` — see below)**

## Installation

Part of the `luxar.gsplats` package, but importable on a plain `pip install luxar`: everything in `trils.py` is pure NumPy, and the core scene-authoring path depends on that (`add_gsplats` reaches `split_tril` through the compiler on every call).

`worker_memory.py` is also dependency-free so core CLI planning can share fit-memory policy without importing `torch`.

The two `device.py` helpers — `resolve_torch_device` and `is_mps_available` — are the exception: they import `torch`, which ships only in the optional `gsplats` extra. They are therefore resolved lazily by a module `__getattr__` (PEP 562), so importing this package costs nothing on a core-only install and `from luxar.gsplats.utils import resolve_torch_device` raises `ModuleNotFoundError: No module named 'torch'` only when the name is actually touched. Install the extra to use them:

```bash
pip install 'luxar[gsplats]'
```

## Quick Start

```python
from luxar.gsplats.utils import (
    calculate_gradient_dilution_factor,
    pack_tril,
    unpack_tril,
)
import numpy as np

# Pack a batch of Cholesky factors for storage
L = np.array([[[1.0, 0.0], [0.5, 0.8]]])  # Shape (1, 2, 2) - batch of one 2x2 matrix
packed = pack_tril(L)  # Returns [[1.0, 0.5, 0.8]], shape (1, 3)

# Unpack for computation
L_restored = unpack_tril(packed, d=2)  # Returns shape (1, 2, 2)

# Calculate gradient dilution for 3D optimization
factor = calculate_gradient_dilution_factor(3)  # Returns ~1.8

# Everything above runs on a plain `pip install luxar`. Everything below needs
# `pip install 'luxar[gsplats]'` — hence the separate import block: these two
# names resolve lazily, so on a core-only install even importing them raises.
from luxar.gsplats.utils import is_mps_available, resolve_torch_device

# Resolve a PyTorch device with accelerator opt-out flags
fit_device = resolve_torch_device(None, use_cuda=False, use_metal=True)

# Or probe the Metal backend directly (robust against older PyTorch builds)
on_metal = is_mps_available()
```

## Core Functions

### tril_size(d: int) -> int

Calculate the number of elements in a d×d lower triangular matrix.

**Formula**: `d × (d + 1) / 2`

**Example**:
```python
from luxar.gsplats.utils import tril_size

tril_size(2)   # Returns 3: elements (0,0), (1,0), (1,1)
tril_size(3)   # Returns 6
tril_size(10)  # Returns 55
```

**Usage**: Compute parameter counts for Cholesky-parameterized covariance matrices.

---

### pack_tril(L: np.ndarray) -> np.ndarray

Pack batch of lower triangular matrices into flat arrays using row-major ordering.

**Parameters**:
- `L`: Batch of lower triangular matrices, shape (N, d, d) - upper triangle values are ignored
- Returns: Array of shape (N, d*(d+1)/2)

**Row-Major Ordering**:
```python
# For 3x3 matrix:
[[L00,   0,   0],
 [L10, L11,   0],
 [L20, L21, L22]]

# Packed as: [L00, L10, L11, L20, L21, L22]
```

**Example**:
```python
L = np.array([[
    [1.0, 0.0, 0.0],
    [0.5, 0.8, 0.0],
    [0.2, 0.3, 0.6]
]])  # Shape (1, 3, 3)
packed = pack_tril(L)
# Returns: [[1.0, 0.5, 0.8, 0.2, 0.3, 0.6]], shape (1, 6)
```

**Dtype Preservation**: Output has same dtype as input

**Use Case**: Efficient storage of Cholesky factors for Gaussian splats

---

### unpack_tril(v: np.ndarray, d: int) -> np.ndarray

Unpack batch of flat arrays to lower triangular matrices.

**Parameters**:
- `v`: Array of shape (N, d*(d+1)/2)
- `d`: Matrix dimension
- Returns: Batch of lower triangular matrices, shape (N, d, d) with zeros in upper triangle

**Example**:
```python
packed = np.array([[1.0, 0.5, 0.8, 0.2, 0.3, 0.6]])  # Shape (1, 6)
L = unpack_tril(packed, d=3)
# Returns shape (1, 3, 3):
# [[[1.0, 0.0, 0.0],
#   [0.5, 0.8, 0.0],
#   [0.2, 0.3, 0.6]]]
```

**Use Case**: Reconstruct Cholesky factors from storage for rendering or analysis

---

### calculate_gradient_dilution_factor(d: int) -> float

Calculate gradient dilution compensation factor for nD optimization.

**Purpose**: In higher dimensions, gradients become diluted across more parameters. This function computes a scaling factor to compensate and maintain convergence rates across dimensions.

**Algorithm**:
- **For d ≤ 3** (Conservative): `params_current / params_2d`
- **For d > 3** (Enhanced): `d^0.8 × (params_current / params_2d)`

Where:
- `params_2d = 2 + tril_size(2) = 5` (baseline: 2D position + 3 Cholesky elements)
- `params_current = d + tril_size(d)` (current dimension parameter count)

**Expected Values**:
- 2D: 1.0 (baseline, no compensation)
- 3D: 1.8 (conservative scaling)
- 4D: 8.5 (enhanced scaling)
- 5D: 14.1 (enhanced scaling)

**Example**:
```python
from luxar.gsplats.utils import calculate_gradient_dilution_factor

# 3D optimization
factor_3d = calculate_gradient_dilution_factor(3)
adjusted_lr = base_lr * factor_3d  # Scale learning rate

# 4D optimization
factor_4d = calculate_gradient_dilution_factor(4)
# Use factor to scale variance/covariance learning rates
```

**Integration**: Used by `create_optimizer_and_scheduler()` to automatically scale learning rates based on dimensionality. The function multiplies the base learning rate by the dilution factor before creating the standard PyTorch Adam optimizer.

**See Also**:
- `../optim/README.md` - Optimizer factory and gradient dilution integration
- `../README.md` - Gradient dilution rationale

---

### permute_cholesky_packed(packed, d, perm) -> np.ndarray

Reorder the dimensions of packed Cholesky factors according to a permutation.

**Purpose**: When dimensions need to be reordered (e.g., swapping X and Y axes), the packed Cholesky factors must be updated to reflect the new ordering. This function recomputes Cholesky factors for the permuted covariance matrix.

**Parameters**:
- `packed`: np.ndarray, shape (N, k) where k = d*(d+1)//2 -- Packed lower-triangular Cholesky factors
- `d`: int -- Number of dimensions
- `perm`: sequence of int -- Permutation of dimension indices. `perm[new_i] = old_i`. E.g., `[2, 0, 1]` means new dim 0 was old dim 2

**Returns**: np.ndarray, shape (N, k) -- Packed Cholesky factors with permuted dimensions

**Algorithm**:
1. Unpack to full lower-triangular matrices L (in float64 for numerical stability)
2. Compute covariance: Sigma = L @ L^T
3. Permute: Sigma_new[i,j] = Sigma[perm[i], perm[j]]
4. Re-decompose: L_new = cholesky(Sigma_perm)
5. Pack and cast back to input dtype

**Example**:
```python
from luxar.gsplats.utils import permute_cholesky_packed

# Swap X and Y in 2D
packed = np.array([[1.0, 0.5, 2.0]])  # L00, L10, L11
swapped = permute_cholesky_packed(packed, 2, [1, 0])
```

**Complexity**: O(N * d^3) due to Cholesky decomposition

---

### embed_cholesky_packed(packed, d_src, d_dst, dim_mapping, fill_sigma=None) -> np.ndarray

Embed lower-dimensional packed Cholesky factors into a higher-dimensional space.

**Purpose**: When lifting splats from a lower-dimensional space (e.g., 2D) into a higher-dimensional space (e.g., 3D), the Cholesky factors need to be embedded. Mapped dimensions carry over the original covariance; unmapped dimensions get independent Gaussian variance (diagonal only, no cross-terms).

**Parameters**:
- `packed`: np.ndarray, shape (N, k_src) where k_src = d_src*(d_src+1)//2 -- Packed Cholesky factors in source dimensionality
- `d_src`: int -- Source dimensionality
- `d_dst`: int -- Target dimensionality (must be >= d_src)
- `dim_mapping`: list of int, length d_src -- Maps source dimension i to target dimension dim_mapping[i]. E.g., `[1, 2, 3]` maps src dims 0,1,2 to dst dims 1,2,3
- `fill_sigma`: dict of {target_dim_index: sigma_value}, optional -- Standard deviations for unmapped target dimensions. Unmapped dims not in fill_sigma default to 1.0. A value of 0 is replaced with 1e-7 to maintain positive-definiteness

**Returns**: np.ndarray, shape (N, k_dst) where k_dst = d_dst*(d_dst+1)//2 -- Packed Cholesky factors in target dimensionality

**Algorithm**:
1. Unpack source Cholesky and compute source covariance (in float64)
2. Build target covariance matrix (d_dst x d_dst) initialized to zeros
3. Copy source covariance block into mapped positions
4. Fill unmapped diagonal positions with sigma^2
5. Cholesky-decompose the target covariance (with regularization for degenerate cases)
6. Pack and cast back to input dtype

**Example**:
```python
from luxar.gsplats.utils import embed_cholesky_packed

# Embed 2D into 3D: src dims [0,1] -> dst dims [0,1], new dim 2 has sigma=0.5
packed_2d = np.array([[1.0, 0.0, 1.0]])  # isotropic 2D
packed_3d = embed_cholesky_packed(packed_2d, 2, 3, [0, 1], fill_sigma={2: 0.5})
```

**Complexity**: O(N * d_dst^3) due to Cholesky decomposition

---

### validate_cholesky_shape(cholesky_factors, ndim, n_splats=None, allow_uniform=True) -> Tuple[bool, int]

Validate the shape of packed Cholesky factors for Gaussian splats.

**Purpose**: Verify that packed Cholesky factors have the correct shape for the given dimensionality. This helps catch shape errors early and ensures data consistency.

**Parameters**:
- `cholesky_factors`: np.ndarray to validate
- `ndim`: int - Number of dimensions (determines expected packed size k = ndim*(ndim+1)/2)
- `n_splats`: int, optional - Expected number of splats (validates first dimension if provided)
- `allow_uniform`: bool, default=True - Whether to allow uniform (1D) Cholesky factors

**Returns**: Tuple[bool, int]
- `is_uniform`: bool - True if factors are uniform (shape (k,)), False if per-splat (shape (N, k))
- `actual_n_splats`: int - Actual number of splats inferred from shape (0 for uniform)

**Raises**: ValueError if shape is invalid

**Valid Shapes**:
- **Per-splat**: (N, k) where k = d*(d+1)/2
- **Uniform**: (k,) when allow_uniform=True (shared by all splats)

**Example**:
```python
from luxar.gsplats.utils import validate_cholesky_shape, tril_size
import numpy as np

# Valid per-splat for 2D (k=3)
chol = np.random.rand(100, 3)
is_uniform, n = validate_cholesky_shape(chol, ndim=2, n_splats=100)
print(f"Uniform: {is_uniform}, N: {n}")  # Output: Uniform: False, N: 100

# Valid uniform for 3D (k=6)
chol = np.random.rand(6)
is_uniform, n = validate_cholesky_shape(chol, ndim=3)
print(f"Uniform: {is_uniform}, N: {n}")  # Output: Uniform: True, N: 0

# Invalid shape raises ValueError
try:
    chol = np.random.rand(100, 5)  # Wrong k for 2D (expected 3)
    validate_cholesky_shape(chol, ndim=2)
except ValueError as e:
    print(f"Error: {e}")
```

**Common Use Cases**:
- **Validate user input** before fitting or rendering
- **Debug shape mismatches** in splat pipelines
- **Ensure data consistency** when loading from files
- **Detect uniform vs per-splat** storage modes

**Design Note**: Returns both uniform status and count to enable different handling for uniform (broadcasted) vs per-splat data.

---

## Testing

**Test Files**: `tests/test_trils.py`, `tests/test_device.py`

**`test_trils.py` classes**:
- `TestCalculateGradientDilutionFactor` - Gradient dilution factor for 1D through 8D, monotonic increase, type checks
- `TestTrilSize` - Formula validation for various dimensions
- `TestPackTril` - Single/batch packing, upper triangle ignored, dtype preservation
- `TestUnpackTril` - Single/batch unpacking, dtype preservation
- `TestPackUnpackRoundTrip` - Round-trip for dimensions 1-5 with various batch sizes
- `TestEdgeCases` - 1x1 matrices, consistency with numpy.tril_indices
- `TestValidateCholeskShape` - Per-splat/uniform validation, error cases
- `TestEmbedCholeskyPackedNoNan` - Regression tests for degenerate inputs (zeros, near-singular)

**`test_device.py` functions** (no test classes — flat layout):
- `test_resolve_explicit_device_overrides_accelerator_flags` - Explicit device wins over `use_cuda=False/use_metal=False`
- `test_resolve_honors_accelerator_opt_outs` - Both accelerators disabled → CPU
- `test_resolve_prefers_cuda_over_mps_when_both_enabled` - CUDA priority on hosts with both
- `test_resolve_uses_mps_when_cuda_disabled` - MPS fallback on macOS when CUDA off
- `test_is_mps_available_handles_missing_backend` - Older PyTorch without `torch.backends.mps`

---

## Implementation Details

**Package Structure**:
```
gsplats/utils/
├── __init__.py              # Public API exports
├── device.py                # PyTorch device auto-selection helpers
├── trils.py                 # Cholesky pack/unpack and gradient dilution
├── tests/
│   ├── test_device.py       # Device-resolution tests
│   └── test_trils.py        # Comprehensive trils test suite
└── README.md                # This file
```

**Module Exports**:
```python
from luxar.gsplats.utils import (
    tril_size,
    pack_tril,
    unpack_tril,
    calculate_gradient_dilution_factor,
    validate_cholesky_shape,
    permute_cholesky_packed,    # Reorder dimensions of packed Cholesky factors
    embed_cholesky_packed,      # Embed lower-dim Cholesky into higher-dim space
    is_mps_available,           # Robustly detect a working MPS backend (torch)
    resolve_torch_device,       # CUDA > MPS > CPU device auto-selection (torch)
)
```

The last two are lazy (PEP 562 `__getattr__` + `__dir__`) and require the
`gsplats` extra; the rest are eager and pure NumPy.

---

## Performance

Every function documented above is implemented in pure NumPy (the torch-backed
`device` helpers are the package's only exception, and do no array work):
- `tril_size`: O(1) arithmetic
- `pack_tril`: O(N * d^2) element copy
- `unpack_tril`: O(N * d^2) element copy with zero-filling
- `calculate_gradient_dilution_factor`: O(1) arithmetic
- `validate_cholesky_shape`: O(1) shape checks
- `permute_cholesky_packed`: O(N * d^3) - requires Cholesky decomposition
- `embed_cholesky_packed`: O(N * d_dst^3) - requires Cholesky decomposition

---

## Design Rationale

**Row-Major Ordering**: Matches storage conventions for direct zarr writing and efficient memory access.

**Gradient Dilution**: Critical for nD optimization convergence. Without compensation, higher dimensions converge much slower due to gradient distribution across more parameters.

**Dtype Preservation**: Maintains numerical precision of input (float32/float64) for downstream compatibility.

---

## See Also

- **Optimization Integration**: [../optim/README.md](../optim/README.md)
- **Main Gsplats Package**: [../README.md](../README.md)
- **Test Suite**: [tests/test_trils.py](./tests/test_trils.py)

---

## Version

**Package**: luxar.gsplats.utils

## License

Part of the Luxar project. See main repository for license information.
