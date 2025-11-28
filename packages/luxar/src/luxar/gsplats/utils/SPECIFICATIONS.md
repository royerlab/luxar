# Gaussian Splatting Utilities Specification

**Version**: 1.0.0
**Last Updated**: 2025-11-27

## Overview

The utils package provides essential mathematical utilities for Gaussian splatting implementation. These utilities handle:

1. **Matrix Operations**: Packing/unpacking lower-triangular matrices (Cholesky factors)
2. **Gradient Dilution**: Calculating compensation factors for higher-dimensional optimization
3. **Future Extensions**: Placeholder for additional utility functions

**Related Specifications**:
- **Gradient Dilution Usage**: [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md) → Gradient Dilution Compensation
- **Cholesky Parameterization**: [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) → Section 2: Gaussian Splat Model
- **Model Implementation**: [models/SPECIFICATIONS.md](../models/SPECIFICATIONS.md)

## Package Structure

```
utils/
├── trils.py                # Lower-triangular matrix operations and gradient dilution
└── __init__.py
```

## Core Functions

### 1. Lower-Triangular Size Calculation

```python
def tril_size(d: int) -> int:
    """
    Calculate number of elements in lower-triangular portion of d×d matrix.

    Includes all elements on and below the main diagonal.

    Parameters:
        d: Dimension of square matrix

    Returns:
        Number of lower-triangular elements: d*(d+1)/2

    Examples:
        tril_size(2) = 3  # Elements: (0,0), (1,0), (1,1)
        tril_size(3) = 6  # Elements: (0,0), (1,0), (1,1), (2,0), (2,1), (2,2)
        tril_size(4) = 10 # 4×5/2 = 10 elements

    Use Cases:
        - Calculating storage requirements for Cholesky factors
        - Determining parameter counts for optimization
        - Allocating packed representation arrays
    """
    return d * (d + 1) // 2
```

**Mathematical Background**:
- Cholesky decomposition: `Σ = L @ L^T` where L is lower-triangular
- Only `d*(d+1)//2` elements of L are non-zero (below/on diagonal)
- Packed storage saves memory and transmission bandwidth

### 2. Gradient Dilution Factor Calculation

```python
def calculate_gradient_dilution_factor(d: int) -> float:
    """
    Calculate gradient dilution compensation factor for higher dimensions.

    Gradient dilution occurs because higher dimensions have more parameters
    per splat, spreading gradients thinner across more values. This function
    computes the multiplication factor to compensate.

    Parameters:
        d: Dimensionality of the model

    Returns:
        Gradient dilution compensation factor (multiply base LR by this)

    Algorithm:
        1. Calculate parameter counts:
           - params_2d = 2 + tril_size(2) = 2 + 3 = 5 (baseline)
           - params_current = d + tril_size(d)

        2. For d ≤ 3 (Conservative scaling):
           factor = params_current / params_2d

        3. For d > 3 (Enhanced scaling):
           dimensional_complexity = d^0.8
           parameter_complexity = params_current / params_2d
           factor = dimensional_complexity × parameter_complexity

    Examples:
        2D: factor = 5/5 = 1.0× (baseline, no compensation needed)
        3D: factor = 9/5 = 1.8× (80% increase in learning rate)
        4D: factor = 4^0.8 × 14/5 = 3.03 × 2.8 ≈ 8.5×

    Rationale:
        - Higher dimensions have more parameters diluting gradients
        - d^0.8 term captures spatial complexity scaling
        - Parameter ratio captures direct dilution effect
        - Combined approach provides empirically validated compensation

    Usage:
        Automatically applied by PerSplatAdam optimizer during initialization.
        Users don't need to call this directly.
    """
```

**Mathematical Derivation**:

Parameter counts per splat:
- 2D: 2 position + 3 covariance = 5 parameters
- 3D: 3 position + 6 covariance = 9 parameters
- 4D: 4 position + 10 covariance = 14 parameters
- nD: `d + d*(d+1)//2` parameters

**Gradient Dilution Phenomenon**:
- Loss gradient ∂L/∂θ is distributed across all parameters
- More parameters → each parameter receives smaller gradient fraction
- Effective learning rate decreases with dimension
- Compensation factor restores effective learning rate

**Conservative vs Enhanced Scaling**:
- **d ≤ 3**: Conservative scaling maintains existing quality
  - Well-tested on 2D/3D datasets
  - Simple parameter ratio suffices
- **d > 3**: Enhanced scaling addresses empirical findings
  - Higher dimensions need more aggressive compensation
  - Spatial complexity (d^0.8) + parameter dilution
  - Empirically validated on 4D datasets

**Validation**:
```python
# Expected compensation factors
assert calculate_gradient_dilution_factor(2) == 1.0
assert calculate_gradient_dilution_factor(3) == 1.8
assert abs(calculate_gradient_dilution_factor(4) - 8.5) < 0.2  # ≈ 8.5
```

### 3. Pack Lower-Triangular Matrices

```python
def pack_tril(L: np.ndarray) -> np.ndarray:
    """
    Pack lower-triangular portion of matrices into compact vector representation.

    Extracts and concatenates lower-triangular elements (including diagonal)
    from a batch of square matrices in row-major order.

    Parameters:
        L: Batch of square matrices, shape (N, d, d)
           Only elements where i >= j are used (on/below main diagonal)
           Upper triangular elements are ignored

    Returns:
        Packed vectors, shape (N, d*(d+1)//2)
        Elements ordered as: [L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2], ...]

    Algorithm:
        for each row i from 0 to d-1:
            for each column j from 0 to i:
                packed[k++] = L[i, j]

    Example:
        L = [[[1, 0], [2, 3]]]  # Shape (1, 2, 2)
        pack_tril(L) → [[1, 2, 3]]  # Shape (1, 3): [L00, L10, L11]

        L = [[[1, 0, 0], [2, 3, 0], [4, 5, 6]]]  # Shape (1, 3, 3)
        pack_tril(L) → [[1, 2, 3, 4, 5, 6]]  # Shape (1, 6)

    Use Cases:
        - Efficient storage of Cholesky factors
        - Network transmission (reduced bandwidth)
        - Parameter export for external tools
        - Final result packaging (fit_gaussian_splats return value)

    Storage Savings:
        - 2×2: 4 → 3 elements (25% savings)
        - 3×3: 9 → 6 elements (33% savings)
        - 4×4: 16 → 10 elements (37.5% savings)
        - d×d: d² → d*(d+1)/2 (approaches 50% for large d)
    """
```

**Implementation**:
```python
def pack_tril(L: np.ndarray) -> np.ndarray:
    N, d, _ = L.shape
    out = np.zeros((N, tril_size(d)), dtype=L.dtype)
    k = 0
    for i in range(d):
        for j in range(i + 1):  # j <= i (lower triangle)
            out[:, k] = L[:, i, j]
            k += 1
    return out
```

### 4. Unpack Lower-Triangular Matrices

```python
def unpack_tril(v: np.ndarray, d: int) -> np.ndarray:
    """
    Unpack compact vector representation into lower-triangular matrices.

    Inverse operation of pack_tril(). Reconstructs square matrices from
    their packed lower-triangular representations, filling upper triangle
    with zeros.

    Parameters:
        v: Packed vectors, shape (N, d*(d+1)//2)
           Elements in row-major order
        d: Dimension of square matrices to reconstruct

    Returns:
        Lower-triangular matrices, shape (N, d, d)
        Zeros above diagonal, packed elements on/below diagonal

    Algorithm:
        for each row i from 0 to d-1:
            for each column j from 0 to i:
                L[i, j] = packed[k++]
            for each column j from i+1 to d-1:
                L[i, j] = 0  # Upper triangle is zero

    Example:
        v = [[1, 2, 3]]  # Shape (1, 3)
        unpack_tril(v, 2) → [[[1, 0], [2, 3]]]  # Shape (1, 2, 2)

        v = [[1, 2, 3, 4, 5, 6]]  # Shape (1, 6)
        unpack_tril(v, 3) → [[[1, 0, 0], [2, 3, 0], [4, 5, 6]]]  # Shape (1, 3, 3)

    Use Cases:
        - Loading saved Cholesky factors
        - Importing parameters from external tools
        - Reconstructing covariance matrices: Σ = L @ L^T

    Validation:
        - pack_tril(unpack_tril(v, d)) == v (lossless round-trip)
        - unpack_tril(pack_tril(L), d) == L (for lower-triangular L)
    """
```

**Implementation**:
```python
def unpack_tril(v: np.ndarray, d: int) -> np.ndarray:
    N = v.shape[0]
    L = np.zeros((N, d, d), dtype=v.dtype)
    k = 0
    for i in range(d):
        for j in range(i + 1):  # j <= i (lower triangle)
            L[:, i, j] = v[:, k]
            k += 1
    return L
```

## Mathematical Foundations

### Cholesky Decomposition

**Definition**: For positive definite matrix Σ, Cholesky decomposition gives:
```
Σ = L @ L^T
```
where L is lower-triangular with positive diagonal elements.

**Properties**:
- Unique decomposition (given positive diagonal constraint)
- Numerically stable (avoids matrix inversion)
- Compact storage (only `d*(d+1)//2` non-zero elements)
- Efficient computation of `Σ^{-1} @ x` via forward/backward substitution

**Gaussian Splatting Application**:
- Each splat has covariance matrix Σ
- Parameterize via Cholesky factor L
- Ensures Σ is always positive definite during optimization
- Avoids expensive eigenvalue constraints

### Row-Major Order

**Convention**: Elements stored in row-by-row order.

**Example** (3×3 lower-triangular):
```
Matrix:          Packed Order:
[a, 0, 0]        [a, b, c, d, e, f]
[b, c, 0]         ^  ^  ^  ^  ^  ^
[d, e, f]         |  |  |  |  |  |
                 [0,0] [1,0] [1,1] [2,0] [2,1] [2,2]
```

**Index Mapping**:
```
Linear index k → (row i, col j):
k = 0: (0, 0)
k = 1: (1, 0)
k = 2: (1, 1)
k = 3: (2, 0)
k = 4: (2, 1)
k = 5: (2, 2)

Formula: k = i*(i+1)//2 + j for j <= i
```

### Gradient Dilution Theory

**Problem Statement**:
Given loss function L(θ) where θ ∈ ℝ^n:
- Gradient: ∇L = [∂L/∂θ₁, ..., ∂L/∂θₙ]
- Gradient magnitude typically scales with 1/√n
- Effective learning rate decreases as n increases

**Compensation Strategy**:
Scale learning rate by factor proportional to parameter count:
```
α_effective = α_base × f(n)
```
where f(n) grows with parameter count n.

**Empirical Formula** (d ≤ 3):
```
f(n) = n / n_baseline
```
Maintains constant effective learning rate.

**Enhanced Formula** (d > 3):
```
f(n) = d^0.8 × (n / n_baseline)
```
Accounts for both spatial complexity and parameter dilution.

**Validation Method**:
Compare convergence rates across dimensions with/without compensation.

## Implementation Details

### NumPy dtype Preservation

**Requirement**: All functions preserve input dtype.

**Implementation**:
```python
out = np.zeros((N, tril_size(d)), dtype=L.dtype)  # Preserve input dtype
```

**Supported dtypes**:
- `np.float32`: Standard for Gaussian splatting (memory efficient)
- `np.float64`: High precision (research/debugging)

### Index Bounds Checking

**Pack/Unpack Functions**:
- No explicit bounds checking (rely on NumPy)
- Shape validation via assertions
- Raises IndexError on invalid shapes

**Gradient Dilution**:
- No bounds checking needed (pure math function)
- Input d ≥ 1 assumed (validated by caller)

### Performance Considerations

**Packing/Unpacking**:
- Time complexity: O(N × d²) (iterate all lower-triangular elements)
- Space complexity: O(N × d²) (output allocation)
- Vectorized over batch dimension N

**Gradient Dilution**:
- Time complexity: O(1) (simple arithmetic)
- Space complexity: O(1) (single float return)

**Optimization Opportunities**:
- Use NumPy indexing tricks for pack/unpack (future)
- Precompute tril_size for common dimensions (future)

## Testing Requirements

### Unit Tests

**tril_size()**:
- Correctness for d=1,2,3,4,5,10
- Formula: d*(d+1)//2

**calculate_gradient_dilution_factor()**:
- Correctness for d=1,2,3,4,5,10
- Expected values: 2D=1.0, 3D=1.8, 4D≈8.5
- Conservative vs enhanced scaling boundary at d=3

**pack_tril()**:
- Correctness on known matrices
- Dtype preservation
- Batch processing correctness
- Edge cases: d=1, empty batch

**unpack_tril()**:
- Correctness on known vectors
- Dtype preservation
- Round-trip: pack(unpack(v, d)) == v
- Round-trip: unpack(pack(L), d) == L

### Property Tests

**Mathematical Invariants**:
- `tril_size(d) = d*(d+1)//2`
- `pack_tril(unpack_tril(v, d)) == v` (lossless)
- `unpack_tril(pack_tril(L), d) == L` (for lower-triangular L)
- `calculate_gradient_dilution_factor(2) == 1.0` (baseline)

### Integration Tests

**Complete Pipeline**:
- Pack parameters → Save → Load → Unpack
- Gradient dilution → Optimization convergence
- Parameter export → Import in external tool

## Extension Points

### Adding New Matrix Operations

Follow same pattern for other matrix structures:
```python
def pack_symmetric(S: np.ndarray) -> np.ndarray:
    """Pack symmetric matrix (upper or lower triangle)."""
    ...

def unpack_symmetric(v: np.ndarray, d: int) -> np.ndarray:
    """Unpack symmetric matrix."""
    ...
```

### Adding New Compensation Strategies

Implement alternative gradient dilution formulas:
```python
def calculate_gradient_dilution_adaptive(d: int, convergence_rate: float) -> float:
    """Adaptive compensation based on observed convergence."""
    ...
```

## Usage Examples

### Parameter Export/Import

```python
import numpy as np
from luxar.gsplats.utils import pack_tril, unpack_tril

# After optimization
centers, Ls, amps, sharpness = model.current_params()
Ls_np = Ls.detach().cpu().numpy()  # Shape: (N, d, d)

# Pack for storage
Ls_packed = pack_tril(Ls_np)  # Shape: (N, d*(d+1)//2)

# Save to disk
np.savez("splats.npz", centers=centers, Ls=Ls_packed, amps=amps, sharpness=sharpness)

# Load and unpack
data = np.load("splats.npz")
Ls_restored = unpack_tril(data["Ls"], d=3)  # Shape: (N, 3, 3)
```

### Gradient Dilution in Optimizer

```python
from luxar.gsplats.utils import calculate_gradient_dilution_factor

# In PerSplatAdam.__init__()
d = len(model.shape)
gradient_dilution_factor = calculate_gradient_dilution_factor(d)
self.effective_lr = self.base_lr * gradient_dilution_factor

# Result: 3D model gets 1.8× learning rate automatically
```

### Parameter Count Calculation

```python
from luxar.gsplats.utils import tril_size

d = 4  # 4D Gaussian splatting
params_per_splat = d + tril_size(d) + 1 + 1  # centers + Cholesky + amplitude + sharpness
print(f"Parameters per splat: {params_per_splat}")  # Output: 4 + 10 + 1 + 1 = 16
```

## See Also

**Related Specifications**:
- [Main SPECIFICATIONS.md](../SPECIFICATIONS.md) - Core Gaussian splatting concepts
- [optim/SPECIFICATIONS.md](../optim/SPECIFICATIONS.md) - Gradient dilution usage in optimizer
- [models/SPECIFICATIONS.md](../models/SPECIFICATIONS.md) - Cholesky parameterization
- [GLOSSARY.md](../GLOSSARY.md) - Terminology and naming conventions

## Changelog

- **v1.0.0** (January 2025): Initial implementation
  - Lower-triangular packing/unpacking utilities
  - Gradient dilution compensation calculation
  - Conservative (d≤3) and enhanced (d>3) scaling formulas
  - Comprehensive testing and documentation
