# Metal Backend Specification

**Version**: 1.0.0
**Last Updated**: 2025-12-23

## Overview

The Metal backend provides GPU-accelerated Gaussian splatting for Apple Silicon (M1/M2/M3/M4) Macs using Metal compute shaders. It replaces the PyTorch rendering path for 3D volumes with a highly optimized pixel-parallel implementation.

**Prerequisites**:
- macOS with Apple Silicon
- Xcode (full installation, not just Command Line Tools)
- PyTorch with MPS support

**Related Specifications**:
- **Main Rendering**: [models/SPECIFICATIONS.md](../SPECIFICATIONS.md) - PyTorch rendering specification
- **Fitting Pipeline**: [fitting/SPECIFICATIONS.md](../../../fitting/SPECIFICATIONS.md) - Integration with fitting

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Layer                                  │
│  GaussianSplatModelMetal → MetalSplatFunction                   │
│  (torch.autograd.Function for gradient computation)              │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              PyTorch: L → Conic Conversion                       │
│  cholesky_inverse(L) → Σ⁻¹  [O(N), CPU/MPS]                     │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              C++ Dispatcher (bindings.mm)                        │
│  dispatch_forward_3d() / dispatch_backward_3d()                  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                Metal Compute Kernels                             │
│  Forward:  preprocess → bin → rasterize_fwd_3d                   │
│  Backward: rasterize_bwd_3d (with tile data from forward)        │
└─────────────────────────────────────────────────────────────────┘
```

## Coordinate Conventions

**Critical**: The Metal backend handles coordinate transformations between PyTorch and Metal conventions.

### PyTorch/NumPy Convention: [Z, Y, X]
- Array indexing: `volume[z, y, x]`
- Centers: `[z_coord, y_coord, x_coord]`
- Cholesky L: Row indices correspond to [Z, Y, X]
- Conic upper triangle: `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]`

### Metal Kernel Convention: [X, Y, Z] for distance computation
- Grid dispatch: `gid = (x, y, z)`
- Pixel position: `float3 px = float3(gid.z, gid.y, gid.x)` converts to [Z,Y,X]
- Conic reordered: `[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]`

### Conversion Mapping
```python
# PyTorch [Z,Y,X] → Metal [X,Y,Z] conic reorder
# PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] indices [0,1,2,3,4,5]
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] indices [5,4,2,3,1,0]
conic_metal = conic_pytorch[:, [5, 4, 2, 3, 1, 0]]

# This permutation is self-inverse:
conic_pytorch = conic_metal[:, [5, 4, 2, 3, 1, 0]]
```

**Note**: Centers remain in [Z,Y,X] order throughout - only conic matrices are reordered.

## Metal Kernels

### Forward Pass Kernels

1. **preprocess_3d**: Count splats per tile based on AABB
   - Computes axis-aligned bounding boxes
   - Counts splats overlapping each tile
   - O(N × tiles)

2. **bin_3d**: Build per-tile splat lists
   - Allocates tile content buffer
   - Populates splat IDs per tile
   - O(N × tiles)

3. **rasterize_fwd_3d**: Pixel-parallel rendering
   - One thread per voxel
   - Iterates through tile's splat list
   - Accumulates Gaussian contributions
   - O(P × splats_per_tile)

### Backward Pass Kernel

**rasterize_bwd_3d**: Gradient computation

Computes gradients for all parameters:
- Centers (μ)
- Conic matrix (Σ⁻¹)
- Amplitudes (a)
- Sharpness (s)

**Mathematical Formulation**:

The Gaussian intensity is:
```
I(x) = a × exp(-0.5 × D^(s/2))
```
where `D² = d^T × Σ⁻¹ × d` (Mahalanobis distance squared) and `d = x - μ`.

**Gradient Derivations**:

1. **Amplitude gradient**:
   ```
   ∂I/∂a = exp(inner) = I/a
   ```

2. **Sharpness gradient**:
   ```
   ∂I/∂s = I × inner × 0.5 × ln(D²)
   ```
   where `inner = -0.5 × D^(s/2)`

3. **Distance gradient**:
   ```
   ∂I/∂D² = I × (-0.25s) × D^(s/2-1)
   ```

4. **Center gradient** (CRITICAL - this was the bug fix location):
   ```
   ∂D²/∂d = 2 × Σ⁻¹ × d
   ∂d/∂μ = -I  (negative identity matrix)

   Therefore:
   ∂I/∂μ = ∂I/∂D² × ∂D²/∂d × ∂d/∂μ
         = grad_dist × (2 × Σ⁻¹ × d) × (-1)
   ```

   **All three dimensions get multiplied by -1** because `d = x - μ` implies `∂d/∂μ = -I`.

5. **Conic gradient**:
   ```
   ∂D²/∂c_ij = d_i × d_j  (for diagonal elements)
             = 2 × d_i × d_j  (for off-diagonal elements)
   ```

### Gradient Bug Fix (2025-12-23)

**Issue**: The original implementation had incorrect signs for Y and X center gradients.

**Root Cause**: The code incorrectly used `+1.0f` instead of `-1.0f` for Y and X dimensions in the chain rule for `∂d/∂μ`.

**Fix Location**: `kernels.metal` lines 469-474

**Before** (incorrect):
```metal
val_centers.x = grad_dist * d_D2_d_d.x * -1.0f;  // Z: correct
val_centers.y = grad_dist * d_D2_d_d.y * +1.0f;  // Y: WRONG
val_centers.z = grad_dist * d_D2_d_d.z * +1.0f;  // X: WRONG
```

**After** (correct):
```metal
val_centers.x = grad_dist * d_D2_d_d.x * -1.0f;  // Z
val_centers.y = grad_dist * d_D2_d_d.y * -1.0f;  // Y
val_centers.z = grad_dist * d_D2_d_d.z * -1.0f;  // X
```

**Symptom**: During optimization, splats would become extremely elongated in Y/X dimensions and appear in incorrect locations.

**Guard Tests**: Three new tests in `test_metal_numerical.py` prevent regression:
- `test_gradient_sign_correctness`: Verifies gradient signs point toward target
- `test_gradient_values_match_cpu_reference`: Compares Metal vs CPU gradients
- `test_optimization_convergence`: Verifies optimization converges correctly

## Python Interface

### GaussianSplatModelMetal

Drop-in replacement for `GaussianSplatModel` that uses Metal rendering.

```python
from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

model = GaussianSplatModelMetal(
    shape=(64, 64, 64),          # 3D volume shape
    centers0=centers,            # Initial centers (N, 3)
    L0=L,                        # Initial Cholesky factors (N, 3, 3)
    amps0=amps,                  # Initial amplitudes (N,)
    sigma_min_diag=(1.0, 1.0, 1.0),  # Minimum diagonal values
    sigma_max_diag=None,         # Maximum diagonal values (optional)
    truncate=3.0,                # Truncation radius in sigmas
    intensity_floor=1e-5,        # Early culling threshold
    tile_size=4,                 # Tile size for binning (4³ = 64 voxels)
    use_metal_conic=False,       # Use Metal for L→Conic conversion
    device='mps',                # Must be MPS or CPU
)
```

### MetalSplatFunction

Custom `torch.autograd.Function` that handles forward/backward with Metal.

```python
output = MetalSplatFunction.apply(
    centers,      # (N, 3) tensor, requires_grad=True
    Ls,           # (N, 3, 3) Cholesky factors
    amps,         # (N,) amplitudes
    sharpness,    # (N,) sharpness values
    shape,        # (D, H, W) output shape
    truncate,     # Truncation radius
    intensity_floor,  # Early culling threshold
    tile_size,    # Tile size for binning
    use_metal_conic,  # Whether to use Metal conic computation
)
```

## Performance Characteristics

### Computational Complexity

**Forward Pass**:
- Preprocessing: O(N × tiles)
- Binning: O(N × tiles)
- Rasterization: O(P × avg_splats_per_tile)

**Backward Pass**:
- Rasterization: O(P × avg_splats_per_tile) with atomic gradient accumulation

### Expected Speedups (vs CPU PyTorch)

| Chip | Speedup Range |
|------|--------------|
| M4 Max | 10-50× |
| M3 | 8-30× |
| M2 | 5-20× |
| M1 | 3-15× |

Actual speedup depends on volume size, splat count, and splat density.

### Tile Size Optimization

The `tile_size` parameter affects binning granularity:
- `tile_size=4`: 64 voxels/tile (default, good balance)
- `tile_size=8`: 512 voxels/tile (fewer tiles, more splats/tile)
- `tile_size=2`: 8 voxels/tile (more tiles, fewer splats/tile)

## Limitations

1. **3D only**: Metal backend requires exactly 3 dimensions. For nD, PyTorch fallback is used.
2. **MPS device required**: Tensors must be on MPS device.
3. **float32 only**: No mixed precision support.
4. **Maximum splats**: Limited by GPU memory for tile data structures.

## Testing Requirements

### Unit Tests (`tests/test_metal_numerical.py`)

1. **Gradient correctness**:
   - `test_gradient_sign_correctness`: Verify gradient signs
   - `test_gradient_values_match_cpu_reference`: Compare with CPU
   - `test_optimization_convergence`: Verify optimization works

2. **Forward pass**:
   - `test_forward_matches_cpu`: Output matches PyTorch rendering

3. **Backward pass**:
   - `test_gradcheck_3d`: PyTorch gradcheck (may skip on GPU)

### Coordinate Transform Tests (`tests/test_coordinate_transforms.py`)

- `test_centers_reorder_roundtrip`: [Z,Y,X] → [X,Y,Z] → [Z,Y,X]
- `test_conic_reorder_roundtrip`: Conic reorder is self-inverse
- `test_conic_computation_consistency`: Metal vs PyTorch conic

## Changelog

- **v1.0.0** (2025-12-23): Initial documented release
  - Fixed center gradient sign bug (Y/X dimensions)
  - Added comprehensive gradient tests
  - Full coordinate convention documentation
  - Performance benchmarks for M-series chips
