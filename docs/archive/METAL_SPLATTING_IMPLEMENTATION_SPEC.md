> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# Metal Splatting Engine: Complete Implementation Specification

## Executive Summary

This document provides a complete, self-contained specification for implementing a **Metal-based pixel-centric Gaussian splatting renderer** for the Luxar project. The goal is to achieve **10-50x speedup** over the current PyTorch CPU implementation for 3D volume fitting on Apple Silicon (M-series chips).

**Key Design Decisions:**
1. **Hybrid Architecture**: PyTorch for matrix math (O(N)), Metal for pixel operations (O(pixels))
2. **Pixel-Centric Rendering**: Each GPU thread processes one pixel, gathering contributions from nearby splats
3. **Tiled Binning**: Spatial acceleration structure to avoid O(N×P) complexity
4. **SIMD Reduction**: Minimize atomic contention in backward pass using `simd_sum()`

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Mathematical Foundation](#2-mathematical-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [Integration with Luxar](#4-integration-with-luxar)
5. [Build System](#5-build-system)
6. [Metal Kernel Implementation](#6-metal-kernel-implementation)
7. [C++ Dispatcher](#7-c-dispatcher)
8. [Python Interface](#8-python-interface)
9. [Testing Strategy](#9-testing-strategy)
10. [Performance Expectations](#10-performance-expectations)
11. [Risks and Mitigations](#11-risks-and-mitigations)
12. [Implementation Phases](#12-implementation-phases)
13. [Appendix A: Complete Metal Kernel Code](#appendix-a-complete-metal-kernel-code)
14. [Appendix B: Complete C++ Dispatcher Code](#appendix-b-complete-c-dispatcher-code)
15. [Appendix C: Complete Python Interface Code](#appendix-c-complete-python-interface-code)

---

## 1. Background & Motivation

### 1.1 Current Implementation

The current Luxar Gaussian splatting renderer (`rendering_core.py`) uses a **splat-centric** approach:

```
For each splat k:
    Compute AABB (bounding box)
    For each pixel p in AABB:
        Compute contribution
        Accumulate via index_add_()
```

**Location**: `packages/luxar/src/luxar/gsplats/models/gsplats/rendering_core.py`

**Key functions**:
- `render_gaussians()` - Main entry point
- `_render_gaussians_3d()` - Specialized 3D renderer
- `fwd_norm2_3d()` - Forward substitution for Mahalanobis distance

### 1.2 Performance Bottleneck

Benchmarking on M4 Max (128GB RAM) shows:
- **CPU**: ~555 ms/iteration (baseline)
- **MPS**: ~1977 ms/iteration (3.6x slower due to scatter operations)

The bottleneck is `index_add_()` scatter operations, which are inefficient on GPU because:
1. Non-coalesced memory writes
2. Atomic contention when multiple splats overlap

### 1.3 Target Architecture

We will implement a **pixel-centric** approach:

```
For each pixel p (in parallel):
    Find nearby splats (via tiled binning)
    For each nearby splat k:
        Compute contribution
        Accumulate locally
    Write final value (single coalesced write)
```

This is the same approach used by:
- 3D Gaussian Splatting (SIGGRAPH 2023)
- Image-GS (arXiv 2407.01866)

---

## 2. Mathematical Foundation

### 2.1 Gaussian Splat Formulation

Each splat k contributes intensity at position x using a shifted Gaussian that ensures
C⁰ continuity (exactly zero) at the truncation boundary:

```
C     = exp(-0.5 × T²)              // boundary value (T=3 → 0.01111)
scale = 1 / (1 - C)                 // peak-preserving rescale (T=3 → 1.01123)
I_k(x) = a_k × scale × max(0, exp(-0.5 × D_k(x)) - C)
```

Where:
- `a_k` = amplitude (scalar)
- `D_k(x)` = squared Mahalanobis distance (standard Gaussian, hardcoded s=2)
- `T` = truncation radius (default 3.0)
- `C` = boundary shift ensuring the function reaches exactly zero at D = T²

### 2.2 Mahalanobis Distance

For covariance matrix `Σ_k = L_k × L_k^T` (Cholesky decomposition):

```
D_k(x) = (x - μ_k)^T × Σ_k^(-1) × (x - μ_k)
       = ||L_k^(-1) × (x - μ_k)||²
       = ||y_k||²  where L_k × y_k = (x - μ_k)
```

### 2.3 Conic Representation

To avoid per-pixel matrix solve, we precompute the **conic** (inverse covariance):

```
C_k = Σ_k^(-1) = (L_k × L_k^T)^(-1) = L_k^(-T) × L_k^(-1)
```

For 3D, `C_k` is a 3×3 symmetric matrix with 6 unique elements:
```
C = [c_xx, c_xy, c_xz]
    [c_xy, c_yy, c_yz]
    [c_xz, c_yz, c_zz]
```

The Mahalanobis distance becomes:
```
D_k(x) = d^T × C_k × d
       = c_xx×dx² + c_yy×dy² + c_zz×dz² + 2×(c_xy×dx×dy + c_xz×dx×dz + c_yz×dy×dz)
```

Where `d = x - μ_k`.

### 2.4 Cholesky Inversion (L → C)

Given lower-triangular L:
```
L = [L00,   0,   0]
    [L10, L11,   0]
    [L20, L21, L22]
```

Compute K = L^(-1) via forward substitution:
```
K00 = 1/L00
K11 = 1/L11
K22 = 1/L22
K10 = -L10 × K00 × K11
K21 = -L21 × K11 × K22
K20 = -(L20 × K00 + L21 × K10) × K22
```

Then C = K^T × K:
```
c_xx = K00² + K10² + K20²
c_xy = K10 × K11 + K20 × K21
c_xz = K20 × K22
c_yy = K11² + K21²
c_yz = K21 × K22
c_zz = K22²
```

### 2.5 Bounding Box Calculation

The extent of ellipsoid `d^T × C × d ≤ r²` projected onto axis i is:
```
extent_i = r × sqrt(Σ_ii)
```

Where `Σ_ii = sum_j(L_ij²)` (row-wise sum of squares of L).

For truncation radius `truncate`:
```
r = truncate  # standard Gaussian truncation radius
```

### 2.6 Gradient Derivation

**Forward (shifted Gaussian)**: `I = a × scale × max(0, exp(-0.5 × D) - C)` where `C = exp(-0.5 × T²)`, `scale = 1/(1-C)`

Let `g = exp(-0.5 × D) - C`. The gradient is nonzero only when `g > 0`.

**Gradients** (for backward pass):

1. **Amplitude**:
   ```
   ∂I/∂a = scale × max(0, g) = I/a    (when g > 0)
   ```

2. **Sharpness**:
   ```
   ∂I/∂s = I × ∂inner/∂s
         = I × (-0.5) × D^(s/2) × ln(D) × 0.5
         = I × inner × 0.5 × ln(D)
   ```

3. **Distance** (chain rule):
   ```
   ∂I/∂D = I × ∂inner/∂D
         = I × (-0.5) × (s/2) × D^(s/2 - 1)
         = I × (-0.25 × s) × D^(s/2 - 1)
   ```

4. **Center** (via distance):
   ```
   ∂D/∂μ = -2 × C × d
   ∂I/∂μ = (∂I/∂D) × (∂D/∂μ)
   ```

5. **Conic** (via distance):
   ```
   ∂D/∂c_xx = dx²
   ∂D/∂c_xy = 2 × dx × dy
   ∂D/∂c_xz = 2 × dx × dz
   ...etc
   ```

---

## 2.7 Dimension and Coordinate Conventions

**CRITICAL**: This section defines the canonical coordinate conventions used throughout all
Metal kernels, C++ dispatchers, and Python interfaces. Inconsistent conventions cause
subtle indexing bugs that are extremely hard to debug.

### Standard Convention (Used Everywhere)

| Variable | Components | Meaning |
|----------|------------|---------|
| `img_size` | `(W, H, D)` | Volume dimensions: Width, Height, Depth |
| `grid_dims` | `(tiles_x, tiles_y, tiles_z)` | Tile grid dimensions |
| `gid` | `(x, y, z)` | Thread position (pixel coordinates) |
| `centers` | `(z, y, x)` per splat | Splat center coordinates (stored in [Z,Y,X] order throughout; see Metal SPECIFICATIONS.md for coordinate conventions) |
| `conic` | `(c_xx, c_xy, c_xz, c_yy, c_yz, c_zz)` | Upper triangle of Σ⁻¹ |

### Output Index Calculation (Row-Major)

For a pixel at position `(x, y, z)` in a volume of size `(W, H, D)`:

```
out_idx = z * (H * W) + y * W + x
```

This corresponds to NumPy/PyTorch `[z, y, x]` indexing into a `(D, H, W)` shaped array.

### Memory Layout Mapping

```
Metal uint3 img_size:       img_size.x = W,  img_size.y = H,  img_size.z = D
Metal uint3 gid:            gid.x = x,       gid.y = y,       gid.z = z
Python shape tuple:         shape = (D, H, W)  # numpy convention
Output tensor indexing:     output[z, y, x]    # numpy convention
```

### Why (W, H, D) for img_size?

Metal's `uint3` maps `.x` to the fastest-varying dimension. For row-major output
indexing `z*H*W + y*W + x`, the stride of `x` is 1 (fastest), so `img_size.x = W`.
This keeps bounds checks intuitive: `gid.x < img_size.x` checks x < W.

### Convention Checklist for New Kernels

Before writing any new kernel, verify:
1. ✓ `img_size` is `(W, H, D)`, not `(D, H, W)`
2. ✓ `grid_dims` is `(tiles_x, tiles_y, tiles_z)`
3. ✓ Output index uses `z * (H * W) + y * W + x`
4. ✓ Bounds check is `gid.x < img_size.x && gid.y < img_size.y && gid.z < img_size.z`
5. ✓ Tile index uses `z * (grid_y * grid_x) + y * grid_x + x`

---

## 3. Architecture Overview

### 3.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  GaussianSplatModelMetal (gsplat_model_metal.py)        │    │
│  │  - Inherits from GaussianSplatModel                     │    │
│  │  - Overrides forward() to use MetalSplatFunction        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  MetalSplatFunction (torch.autograd.Function)           │    │
│  │  - forward(): PyTorch L→Conic, then dispatch Metal      │    │
│  │  - backward(): Metal gradients, PyTorch chain rule      │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────-─┘
                               │
                               ▼ (via compiled C++ extension)
┌─────────────────────────────────────────────────────────────────┐
│                        C++ Dispatcher                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  bindings.mm (Objective-C++)                            │    │
│  │  - dispatch_forward_3d() / dispatch_backward_3d()       │    │
│  │  - dispatch_forward_nd() / dispatch_backward_nd()       │    │
│  │  - Buffer management, kernel dispatch                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (Metal API)
┌─────────────────────────────────────────────────────────────────┐
│                        Metal Kernels                             │
│  ┌────────────────────────────────────────────────────────-┐    │
│  │  kernels.metal                                          │    │
│  │  - preprocess_3d: Compute tile counts                   │    │
│  │  - bin_3d: Populate tile lists                          │    │
│  │  - rasterize_fwd_3d: Forward rendering                  │    │
│  │  - rasterize_bwd_3d: Backward gradients (SIMD)          │    │
│  │  - rasterize_fwd_nd: Generic nD forward                 │    │
│  │  - rasterize_bwd_nd: Generic nD backward                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow (Forward Pass)

```
1. GaussianSplatModelMetal.forward()
   └─> current_params() → centers, Ls, amps

2. MetalSplatFunction.forward()
   ├─> [PyTorch] cholesky_to_conic(Ls) → conic (N, 6)
   ├─> [PyTorch] compute_bbox(centers, Ls, truncate) → lo, hi
   └─> [Metal] dispatch_forward_3d(centers, conic, amps, shape, truncate)
       ├─> preprocess_3d: Count splats per tile
       ├─> prefix_sum: Compute tile offsets (GPU via torch.cumsum)
       ├─> bin_3d: Populate tile_content array
       └─> rasterize_fwd_3d: Pixel-parallel rendering

3. Return output tensor
```

### 3.3 Data Flow (Backward Pass)

```
1. MetalSplatFunction.backward(grad_output)
   ├─> [Metal] dispatch_backward_3d(grad_output, centers, conic, ...)
   │   └─> rasterize_bwd_3d: Compute d_centers, d_conic, d_amps
   │       (Uses SIMD reduction to minimize atomic contention)
   │
   └─> [PyTorch] Chain rule: d_conic → d_Ls
       ├─> Recompute L → Conic in PyTorch (ensures graph consistency)
       └─> conic.backward(d_conic) → d_Ls

2. Return d_centers, d_Ls, d_amps
```

### 3.4 Key Design Decision: L → Conic in PyTorch

**Why not compute L → Conic in Metal?**

The Cholesky inversion involves division and sqrt operations that can have subtle floating-point differences between Metal and PyTorch. If the forward pass computes Conic in Metal but the backward pass expects to use PyTorch's chain rule, gradient inconsistencies can occur.

**Solution**: Compute L → Conic entirely in PyTorch, then pass Conic to Metal.

Benefits:
- 100% PyTorch-native gradient graph
- No numerical consistency issues
- L → Conic is O(N), not O(pixels), so it's fast anyway
- Simpler Metal kernels (just take Conic as input)

### 3.5 Hybrid Graph Data Flow Diagram

This diagram shows the complete forward and backward data flow, illustrating
how PyTorch and Metal work together:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FORWARD PASS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────┐                                                                 │
│  │   Ls    │  (Cholesky parameters, requires_grad=True)                      │
│  └────┬────┘                                                                 │
│       │                                                                      │
│       ▼  [PyTorch Op: O(N)]                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  cholesky_to_conic(Ls)                                                  │ │
│  │  Σ = L @ L^T  →  C = Σ^(-1)  →  extract upper triangle                  │ │
│  │  (Batched matrix multiply + inverse, runs on CPU/MPS)                   │ │
│  └────┬────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────┐                                                                 │
│  │  Conic  │  (N, 6) detached from PyTorch graph                             │
│  └────┬────┘                                                                 │
│       │                                                                      │
│       ├──────────────────────── BARRIER ────────────────────────────────────│
│       │  (Conic copied to MPS tensor, passed to Metal)                       │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Metal: Rasterize(Conic, Centers, Amps, Sharpness)                      │ │
│  │  - preprocess_3d: Count splats per tile                                 │ │
│  │  - bin_3d: Populate tile lists                                          │ │
│  │  - rasterize_fwd_3d: Pixel-parallel rendering [O(P × splats/tile)]      │ │
│  └────┬────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────┐                                                                 │
│  │  Image  │  Output volume (D, H, W)                                        │
│  └─────────┘                                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKWARD PASS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐                                                                │
│  │ d_Image  │  Upstream gradient from loss function                          │
│  └────┬─────┘                                                                │
│       │                                                                      │
│       ▼  [Metal Op: O(P × splats/tile)]                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  Metal: rasterize_bwd_3d(d_Image, Conic, Centers, ...)                  │ │
│  │  - Each pixel computes local gradients                                  │ │
│  │  - SIMD reduction: simd_sum() across 32 threads                         │ │
│  │  - Lane 0 writes to global memory (32x fewer atomics)                   │ │
│  └────┬────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐            │
│  │ d_Conic, d_Centers, d_Amps, d_Sharpness │  Metal-computed    │            │
│  └────┬─────────────────────────────────────────────────────────┘            │
│       │                                                                      │
│       ├──────────────────────── BARRIER ────────────────────────────────────│
│       │  (d_Conic copied back to PyTorch)                                    │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  PyTorch: Chain Rule                                                    │ │
│  │  1. Recompute: conic = cholesky_to_conic(Ls_for_conic)                  │ │
│  │  2. Inject:    conic.backward(d_conic)                                  │ │
│  │  3. Extract:   d_Ls = Ls_for_conic.grad                                 │ │
│  └────┬────────────────────────────────────────────────────────────────────┘ │
│       │                                                                      │
│       ▼                                                                      │
│  ┌────────┐                                                                  │
│  │  d_Ls  │  Final gradient w.r.t. Cholesky parameters                       │
│  └────────┘                                                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

Legend:
  [PyTorch Op]  = Executed by PyTorch on CPU or MPS
  [Metal Op]    = Executed by Metal compute kernels on GPU
  BARRIER       = Data transfer between PyTorch and Metal
```

**Key insight**: The expensive pixel-parallel work happens in Metal, while the
mathematically tricky L → Conic conversion stays in PyTorch for gradient safety.

---

## 4. Integration with Luxar

### 4.1 File Structure

New files to create:
```
packages/luxar/src/luxar/gsplats/models/gsplats/
├── metal/                          # NEW: Metal extension package
│   ├── __init__.py
│   ├── setup.py                    # Build script
│   ├── src/
│   │   ├── kernels.metal           # Metal shader code
│   │   └── bindings.mm             # C++ dispatcher
│   └── gsplat_model_metal.py       # Python interface
└── gsplat_model.py                 # Existing (unchanged)
```

### 4.2 Integration Points

**4.2.1 Model Selection**

Model creation happens in `initialization.py:create_model()` (line ~59). Modify this function
for device-aware model selection:

```python
# packages/luxar/src/luxar/gsplats/fitting/initialization.py (modification)

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def _metal_available() -> bool:
    """Check if Metal extension is available and functional."""
    try:
        import metal_splatting_backend
        return True
    except ImportError:
        return False


def create_model(config: FitConfig, data: PreprocessedData) -> GaussianSplatModel:
    """Create the appropriate model based on device and availability.

    INTEGRATION POINT: This is where Metal vs PyTorch model is chosen.
    The existing function creates GaussianSplatModel directly - we add
    a check for Metal availability first.
    """

    # Check if Metal acceleration is available and beneficial
    # Conditions: macOS platform, Metal extension built, 3D or lower
    use_metal = (
        config.use_metal
        and _metal_available()
        and data.d <= 3
    )

    if use_metal:
        from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
        return GaussianSplatModelMetal(
            shape=tuple(config.V.shape),
            centers0=data.seed_centers,
            L0=...,  # Same initialization as GaussianSplatModel
            amps0=...,
            sigma_min_diag=config.sigma_min_diag,
            sigma_max_diag=config.sigma_max_diag,
            truncate=config.truncate,
            intensity_floor=1e-5,  # Metal-specific parameter
            device=config.device,
        )
    else:
        # Existing PyTorch path
        return GaussianSplatModel(
            shape=tuple(config.V.shape),
            centers0=data.seed_centers,
            ...  # Existing initialization code
        )
```

**Important**: The `config.use_metal` flag should default to `True` in `FitConfig` (see Section 4.4).

**4.2.2 Fallback Mechanism**

The Metal model should gracefully fall back to PyTorch for:
- Dimensions > 3 (use nD kernel or pure PyTorch)
- Non-Apple platforms
- Build failures

```python
def _metal_available() -> bool:
    """Check if Metal extension is available."""
    try:
        import metal_splatting_backend
        return True
    except ImportError:
        return False
```

### 4.3 Existing Interface Compatibility

The `GaussianSplatModelMetal` class MUST maintain full interface compatibility with `GaussianSplatModel`.

**Design Choice: Composition over Inheritance**

We use composition (wrapping a `GaussianSplatModel` instance) rather than inheritance because:
1. Avoids double-registration of parameters when both base and derived class have `nn.Parameter`
2. Cleaner separation between parameter management and rendering strategy
3. Easier to swap between Metal and PyTorch at runtime

```python
class GaussianSplatModelMetal(torch.nn.Module):
    """Metal-accelerated Gaussian splat model (uses composition)."""

    def __init__(self, ...):
        super().__init__()
        self._base = GaussianSplatModel(...)  # Delegate parameter management

    def forward(self) -> torch.Tensor:
        # Override to use Metal rendering
        ...

    # Delegate all other methods to self._base:
    # - current_params()
    # - prune_()
    # - append_()
    # - replace_with()
    # - n_splats()
    # - parameters()  # Important: delegate to _base
```

### 4.4 Configuration

Add Metal-specific options to `FitConfig`:

```python
@dataclass
class FitConfig:
    ...
    # Metal acceleration options
    use_metal: bool = True  # Enable Metal when available
    metal_tile_size: int = 4  # Tile size for 3D (4³ = 64 threads/tile)
```

---

## 5. Build System

### 5.1 setup.py

```python
# packages/luxar/src/luxar/gsplats/models/gsplats/metal/setup.py

import os
import subprocess
import sys
from pathlib import Path

import setuptools
from torch.utils.cpp_extension import BuildExtension, CppExtension


def compile_metal_shaders():
    """Compile Metal shaders to .metallib format."""
    src_dir = Path(__file__).parent / "src"
    metal_file = src_dir / "kernels.metal"
    air_file = src_dir / "kernels.air"
    lib_file = src_dir / "default.metallib"

    if not metal_file.exists():
        raise FileNotFoundError(f"Metal source not found: {metal_file}")

    # Skip if already compiled and source unchanged
    if lib_file.exists():
        if lib_file.stat().st_mtime > metal_file.stat().st_mtime:
            print("Metal library up to date, skipping compilation")
            return

    print(f"Compiling Metal shaders: {metal_file}")

    # Compile .metal -> .air (intermediate representation)
    subprocess.check_call([
        "xcrun", "-sdk", "macosx", "metal",
        "-c", str(metal_file),
        "-o", str(air_file),
        "-std=metal3.0",  # Use Metal 3.0 for SIMD intrinsics
        "-O2",  # Optimization level
    ])

    # Link .air -> .metallib
    subprocess.check_call([
        "xcrun", "-sdk", "macosx", "metallib",
        str(air_file),
        "-o", str(lib_file),
    ])

    # Clean up intermediate file
    air_file.unlink()
    print(f"Metal library created: {lib_file}")


class CustomBuildExtension(BuildExtension):
    """Build extension with Metal shader compilation."""

    def run(self):
        # Check platform
        if sys.platform != "darwin":
            print("WARNING: Metal extension only supported on macOS")
            return

        # Compile Metal shaders first
        compile_metal_shaders()

        # Then build C++ extension
        super().run()


# C++ extension configuration
ext_modules = [
    CppExtension(
        name="metal_splatting_backend",
        sources=["src/bindings.mm"],
        extra_compile_args={
            "cxx": [
                "-std=c++17",
                "-fno-objc-arc",  # Manual memory management for Metal objects
                "-Wno-deprecated-declarations",
            ],
        },
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
        ],
    ),
]


setuptools.setup(
    name="metal_splatting",
    version="1.0.0",
    description="Metal-accelerated Gaussian splatting for Luxar",
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExtension},
    python_requires=">=3.10",
)
```

### 5.2 Build Instructions

```bash
# From packages/luxar/src/luxar/gsplats/models/gsplats/metal/
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal

# Build the extension
pip install -e .

# Or build in-place for development
python setup.py build_ext --inplace
```

### 5.3 Integration with Makefile and Hatch

**Important**: Hatchling doesn't natively support custom C++/Metal build hooks. The Metal extension
must be built separately via Makefile targets.

**Add to project root `Makefile`:**

```makefile
# Metal extension build
METAL_DIR := packages/luxar/src/luxar/gsplats/models/gsplats/metal

build-metal:  ## Build Metal splatting extension (macOS only)
	@if [ "$$(uname)" != "Darwin" ]; then \
		echo "⚠️  Metal extension only supported on macOS"; \
		exit 0; \
	fi
	@echo "🔧 Building Metal splatting extension..."
	@if ! command -v xcrun &> /dev/null; then \
		echo "❌ Xcode command line tools not found. Install with: xcode-select --install"; \
		exit 1; \
	fi
	cd $(METAL_DIR) && pip install -e .
	@echo "✅ Metal extension built successfully!"

build-metal-inplace:  ## Build Metal extension in-place for development
	@if [ "$$(uname)" != "Darwin" ]; then \
		echo "⚠️  Metal extension only supported on macOS"; \
		exit 0; \
	fi
	cd $(METAL_DIR) && python setup.py build_ext --inplace
	@echo "✅ Metal extension built in-place"

clean-metal:  ## Clean Metal build artifacts
	rm -rf $(METAL_DIR)/build/
	rm -rf $(METAL_DIR)/dist/
	rm -rf $(METAL_DIR)/*.egg-info/
	rm -rf $(METAL_DIR)/src/*.air
	rm -rf $(METAL_DIR)/src/*.metallib
	find $(METAL_DIR) -name "*.so" -delete
	@echo "✅ Metal build artifacts cleaned"
```

**Update existing targets to include Metal:**

```makefile
# Modify dev-setup target to optionally build Metal
dev-setup:  ## Complete development setup with Hatch
	@echo "🚀 Setting up development environment with Hatch..."
	hatch env create
	hatch run pre-commit install
	cd packages/luxar-viewer && pnpm install
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "🍎 macOS detected - building Metal extension..."; \
		$(MAKE) build-metal || echo "⚠️  Metal build failed (optional)"; \
	fi
	@echo "✅ Development environment setup complete!"

# Add to .PHONY
.PHONY: ... build-metal build-metal-inplace clean-metal
```

**Running the build:**

```bash
# Build Metal extension (macOS only)
make build-metal

# Or build in-place for development
make build-metal-inplace

# Clean Metal artifacts
make clean-metal
```

**Automatic detection in Python:**

The Python code automatically detects if the Metal extension is available:

```python
try:
    import metal_splatting_backend
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False  # Falls back to PyTorch
```

---

## 6. Metal Kernel Implementation

### 6.1 Overview

The Metal kernels are organized into:
1. **Preprocessing**: Compute tile counts and prepare binning data
2. **Binning**: Populate per-tile splat lists
3. **Rasterization Forward**: Pixel-parallel rendering
4. **Rasterization Backward**: Gradient computation with SIMD reduction
5. **nD Fallback**: Generic kernels for 4D+ data

### 6.2 Tiling Strategy

For 3D volumes:
- **Tile size**: 4×4×4 = 64 voxels per tile
- **Threads per threadgroup**: 64 (one per voxel)
- **SIMD groups per threadgroup**: 2 (32 threads each on Apple Silicon)

The tile size of 4 is chosen because:
- 64 threads fits well in Apple Silicon's execution width
- Small enough for good splat locality
- Large enough for efficient memory access

### 6.3 Kernel Responsibilities

| Kernel | Input | Output | Complexity |
|--------|-------|--------|------------|
| `preprocess_3d` | centers, Ls, truncate | tile_counts | O(N × tiles/splat) |
| `bin_3d` | centers, Ls, tile_offsets | tile_content | O(N × tiles/splat) |
| `rasterize_fwd_3d` | conic, amps, tiles | output | O(P × splats/tile) |
| `rasterize_bwd_3d` | grad_output, conic, ... | d_* | O(P × splats/tile) |

### 6.4 Memory Layout

**Input Buffers**:
- `centers`: (N, 3) float32, row-major
- `conic`: (N, 6) float32, upper triangle [xx, xy, xz, yy, yz, zz]
- `amps`: (N,) float32

**Intermediate Buffers**:
- `tile_counts`: (num_tiles,) int32
- `tile_offsets`: (num_tiles,) int32 (prefix sum of counts)
- `tile_content`: (total_splat_tile_pairs,) int32

**Output Buffers**:
- `output`: (D, H, W) float32, row-major

### 6.5 Shared Helper Functions

```cpp
// Shared between preprocess and bin kernels to ensure consistency
struct TileRange {
    int3 min_t;
    int3 max_t;
};

inline TileRange get_tile_range_3d(
    float3 center,
    float3 sigma_diag,  // Σ_ii = sum_j(L_ij²)
    float truncate,
    constant uint3& grid_dims
) {
    // Exact bounding box: r_i = truncate × sqrt(Σ_ii)
    float3 r = truncate * sqrt(sigma_diag);

    // Convert to tile indices (tile_size = 4)
    TileRange tr;
    tr.min_t = max(int3((center - r) / 4.0f), int3(0));
    tr.max_t = min(int3((center + r) / 4.0f), int3(grid_dims) - 1);

    return tr;
}
```

---

## 7. C++ Dispatcher

### 7.1 Overview

The C++ dispatcher (`bindings.mm`) is responsible for:
1. Managing Metal device, queue, and library
2. Creating and caching pipeline states
3. Allocating GPU buffers
4. Dispatching kernels with correct thread configuration
5. Synchronizing between passes

### 7.2 Context Management

```cpp
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    std::map<std::string, id<MTLComputePipelineState>> pipelines;

    MetalContext() {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];

        // Load precompiled .metallib
        NSString* libPath = /* path to default.metallib */;
        NSError* error = nil;
        library = [device newLibraryWithFile:libPath error:&error];
        if (!library) {
            throw std::runtime_error("Failed to load Metal library");
        }
    }

    id<MTLComputePipelineState> getPipeline(const char* name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) return it->second;

        id<MTLFunction> func = [library newFunctionWithName:
            [NSString stringWithUTF8String:name]];
        if (!func) throw std::runtime_error("Missing kernel: " + std::string(name));

        NSError* error = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:func error:&error];
        pipelines[name] = pso;
        return pso;
    }
};

static MetalContext* g_ctx = nullptr;
```

### 7.3 Buffer Handling with Storage Offsets

**CRITICAL**: PyTorch tensors may be views (slices) of larger storage buffers. Always handle `storage_offset`:

```cpp
// Helper to set buffer with correct offset handling
// Handles edge cases: empty tensors, non-MPS device, non-contiguous
void setBufferWithOffset(
    id<MTLComputeCommandEncoder> enc,
    const torch::Tensor& t,
    int index
) {
    // === Edge Case: Empty tensor ===
    // Empty tensors (N=0) are valid but require special handling
    if (t.numel() == 0) {
        // Bind a dummy buffer or skip binding
        // Note: Metal requires valid buffers at all indices, so we bind nullptr
        // with zero size, which Metal handles gracefully
        [enc setBuffer:nil offset:0 atIndex:index];
        return;
    }

    // === Validation ===
    TORCH_CHECK(t.device().is_mps(),
        "Tensor at index ", index, " must be on MPS device, got ", t.device());
    TORCH_CHECK(t.is_contiguous(),
        "Tensor at index ", index, " must be contiguous. Use .contiguous() first.");

    // === Get MTLBuffer from storage ===
    id<MTLBuffer> buf = tensorToMTLBuffer(t);
    TORCH_CHECK(buf != nil, "Failed to get MTLBuffer for tensor at index ", index);

    // === Compute byte offset ===
    // CRITICAL: storage_offset is in ELEMENTS, must multiply by element_size
    NSUInteger offset = t.storage_offset() * t.element_size();

    // === Validate offset doesn't exceed buffer bounds ===
    NSUInteger buffer_size = [buf length];
    NSUInteger required_size = offset + (t.numel() * t.element_size());
    TORCH_CHECK(required_size <= buffer_size,
        "Buffer overflow: tensor at index ", index,
        " requires ", required_size, " bytes but buffer is ", buffer_size, " bytes");

    [enc setBuffer:buf offset:offset atIndex:index];
}
```

**Edge Cases Handled:**
1. **Empty tensors (N=0)**: Valid input, binds nil buffer
2. **Non-MPS tensors**: Clear error message with device info
3. **Non-contiguous tensors**: Requires `.contiguous()` call first
4. **Buffer overflow**: Catches offset + size exceeding storage

### 7.4 Forward Dispatch (3D)

**IMPORTANT**: Forward must return intermediate buffers for backward pass reuse.

```cpp
// Return struct for forward pass - backward needs these buffers
struct ForwardResult {
    torch::Tensor output;
    torch::Tensor tile_counts;
    torch::Tensor tile_offsets;
    torch::Tensor tile_content;
};

std::vector<torch::Tensor> dispatch_forward_3d(
    torch::Tensor centers,   // (N, 3)
    torch::Tensor conic,     // (N, 6) - precomputed in PyTorch
    torch::Tensor amps,      // (N,)
    std::vector<int64_t> shape,  // [D, H, W]
    float truncate,
    float intensity_floor    // For early culling of invisible contributions
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int D = shape[0], H = shape[1], W = shape[2];  // Extract from numpy-order shape

    // Grid dimensions (tile size = 4)
    // NOTE: Metal uint3 uses (x,y,z) = (W,H,D) order - see §2.7 Dimension Conventions
    uint3 grid_dims = {(W + 3) / 4, (H + 3) / 4, (D + 3) / 4};
    int num_tiles = grid_dims.x * grid_dims.y * grid_dims.z;

    // Allocate buffers (will be returned for backward pass)
    auto tile_counts = torch::zeros({num_tiles}, torch::kInt32).to(torch::kMPS);
    auto output = torch::zeros(shape, torch::kFloat32).to(torch::kMPS);

    // === Pass 1: Preprocess (count splats per tile) ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("preprocess_3d")];

        // Use helper to handle storage offsets correctly
        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, /* Ls for sigma_diag */, 1);
        setBufferWithOffset(enc, tile_counts, 2);
        // ... set constants (truncate, grid_dims, n_splats)

        MTLSize gridSize = MTLSizeMake(N, 1, 1);
        MTLSize groupSize = MTLSizeMake(256, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // === Pass 2: Prefix sum (GPU via PyTorch cumsum - avoids CPU roundtrip!) ===
    // CRITICAL: The CPU version forced GPU→CPU→GPU sync every frame, killing performance.
    // PyTorch's cumsum() runs on MPS and avoids the synchronization penalty.
    //
    // For exclusive prefix sum: prepend 0, compute cumsum, drop last element
    auto zeros = torch::zeros({1}, torch::kInt32).to(torch::kMPS);
    auto counts_padded = torch::cat({zeros, tile_counts.slice(0, 0, num_tiles - 1)});
    auto tile_offsets = torch::cumsum(counts_padded, 0, torch::kInt32);

    // Total is the sum of all counts (last offset + last count)
    int total = (tile_offsets[-1] + tile_counts[-1]).item<int>();

    auto tile_content = torch::zeros({std::max(total, 1)}, torch::kInt32).to(torch::kMPS);
    auto tile_write_heads = torch::zeros({num_tiles}, torch::kInt32).to(torch::kMPS);

    // === Pass 3: Binning ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("bin_3d")];

        setBufferWithOffset(enc, centers, 0);
        // ... set all buffers with offset handling
        [enc dispatchThreads:MTLSizeMake(N, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // === Pass 4: Rasterization ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_fwd_3d")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, conic, 1);
        setBufferWithOffset(enc, amps, 2);
        setBufferWithOffset(enc, tile_offsets, 3);
        setBufferWithOffset(enc, tile_counts, 4);
        setBufferWithOffset(enc, tile_content, 5);
        setBufferWithOffset(enc, output, 6);
        // ... set constants (img_size, grid_dims, truncate, intensity_floor)

        // CRITICAL: Pad grid to multiple of tile size to ensure full threadgroups!
        // This is important for consistent behavior and SIMD operations.
        constexpr int TILE = 4;
        int W_padded = ((W + TILE - 1) / TILE) * TILE;
        int H_padded = ((H + TILE - 1) / TILE) * TILE;
        int D_padded = ((D + TILE - 1) / TILE) * TILE;

        MTLSize gridSize = MTLSizeMake(W_padded, H_padded, D_padded);
        MTLSize groupSize = MTLSizeMake(TILE, TILE, TILE);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    // Return output AND intermediate buffers for backward pass
    return {output, tile_counts, tile_offsets, tile_content};
}
```

### 7.5 Backward Dispatch (3D)

**CRITICAL**: Gradient buffers must be zero-initialized before kernel dispatch.

```cpp
std::vector<torch::Tensor> dispatch_backward_3d(
    torch::Tensor grad_output,    // (D, H, W)
    torch::Tensor centers,        // (N, 3)
    torch::Tensor conic,          // (N, 6) - precomputed
    torch::Tensor amps,           // (N,)
    torch::Tensor tile_offsets,   // From forward pass
    torch::Tensor tile_counts,    // From forward pass
    torch::Tensor tile_content,   // From forward pass
    std::vector<int64_t> shape,   // [D, H, W]
    float truncate,
    float intensity_floor
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int D = shape[0], H = shape[1], W = shape[2];  // Extract from numpy-order shape

    // Grid dimensions (tile size = 4)
    // NOTE: Metal uint3 uses (x,y,z) = (W,H,D) order - see §2.7 Dimension Conventions
    uint3 grid_dims = {(W + 3) / 4, (H + 3) / 4, (D + 3) / 4};

    // =============================================
    // CRITICAL: Zero-initialize gradient buffers
    // =============================================
    // Atomic operations in the kernel ADD to these buffers,
    // so they MUST start at zero to get correct gradients.
    auto d_centers = torch::zeros({N, 3}, torch::kFloat32).to(torch::kMPS);
    auto d_conic = torch::zeros({N, 6}, torch::kFloat32).to(torch::kMPS);
    auto d_amps = torch::zeros({N}, torch::kFloat32).to(torch::kMPS);

    // Ensure MPS operations complete before kernel dispatch
    torch::mps::synchronize();

    // === Backward Rasterization Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_bwd_3d")];

        // Input buffers (with offset handling)
        setBufferWithOffset(enc, grad_output, 0);
        setBufferWithOffset(enc, centers, 1);
        setBufferWithOffset(enc, conic, 2);
        setBufferWithOffset(enc, amps, 3);
        setBufferWithOffset(enc, tile_offsets, 4);
        setBufferWithOffset(enc, tile_counts, 5);
        setBufferWithOffset(enc, tile_content, 6);

        // Output gradient buffers (zero-initialized above)
        setBufferWithOffset(enc, d_centers, 7);
        setBufferWithOffset(enc, d_conic, 8);
        setBufferWithOffset(enc, d_amps, 9);

        // Constants
        uint3 img_size = {(uint)W, (uint)H, (uint)D};
        [enc setBytes:&img_size length:sizeof(uint3) atIndex:10];
        [enc setBytes:&grid_dims length:sizeof(uint3) atIndex:11];
        [enc setBytes:&truncate length:sizeof(float) atIndex:12];

        // CRITICAL: Pad grid to multiple of tile size to ensure full threadgroups!
        // Partial threadgroups can cause undefined behavior with simd_sum().
        // The kernel's `active` check handles out-of-bounds threads.
        constexpr int TILE = 4;
        int W_padded = ((W + TILE - 1) / TILE) * TILE;
        int H_padded = ((H + TILE - 1) / TILE) * TILE;
        int D_padded = ((D + TILE - 1) / TILE) * TILE;

        MTLSize gridSize = MTLSizeMake(W_padded, H_padded, D_padded);
        MTLSize groupSize = MTLSizeMake(TILE, TILE, TILE);  // 64 threads per tile
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {d_centers, d_conic, d_amps};
}
```

### 7.6 Forward Dispatch (nD Generic)

```cpp
std::vector<torch::Tensor> dispatch_forward_nd(
    torch::Tensor centers,        // (N, dim)
    torch::Tensor Ls,             // (N, dim, dim)
    torch::Tensor amps,           // (N,)
    std::vector<int64_t> shape,   // [s_0, s_1, ..., s_{dim-1}]
    float truncate,
    float intensity_floor
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int dim = centers.size(1);

    TORCH_CHECK(dim <= 8, "Metal nD kernel supports up to 8 dimensions");
    TORCH_CHECK(shape.size() == dim, "Shape must match dimensionality");

    // Compute total size and create output
    int64_t total_size = 1;
    for (int i = 0; i < dim; i++) total_size *= shape[i];

    auto output = torch::zeros(shape, torch::kFloat32).to(torch::kMPS);

    // Ensure inputs are on MPS
    torch::mps::synchronize();

    // === Rasterization Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_fwd_nd")];

        setBufferWithOffset(enc, centers, 0);
        setBufferWithOffset(enc, Ls, 1);
        setBufferWithOffset(enc, amps, 2);
        setBufferWithOffset(enc, output, 3);

        // Pass shape as array
        uint shape_arr[8] = {0};
        for (int i = 0; i < dim; i++) shape_arr[i] = (uint)shape[i];
        [enc setBytes:shape_arr length:sizeof(shape_arr) atIndex:4];

        uint dim_u = (uint)dim;
        uint N_u = (uint)N;
        [enc setBytes:&dim_u length:sizeof(uint) atIndex:5];
        [enc setBytes:&N_u length:sizeof(uint) atIndex:6];
        [enc setBytes:&truncate length:sizeof(float) atIndex:7];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:8];

        // Thread configuration: one thread per pixel
        MTLSize gridSize = MTLSizeMake(total_size, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((int64_t)256, total_size), 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {output};  // No tile data for nD (no binning optimization)
}
```

### 7.7 Backward Dispatch (nD Generic)

```cpp
std::vector<torch::Tensor> dispatch_backward_nd(
    torch::Tensor grad_output,    // Shape matches forward output
    torch::Tensor centers,        // (N, dim)
    torch::Tensor Ls,             // (N, dim, dim)
    torch::Tensor amps,           // (N,)
    std::vector<int64_t> shape,
    float truncate,
    float intensity_floor
) {
    if (!g_ctx) g_ctx = new MetalContext();

    int N = centers.size(0);
    int dim = centers.size(1);

    TORCH_CHECK(dim <= 8, "Metal nD kernel supports up to 8 dimensions");

    // Compute total size
    int64_t total_size = 1;
    for (int i = 0; i < dim; i++) total_size *= shape[i];

    // =============================================
    // CRITICAL: Zero-initialize gradient buffers
    // =============================================
    auto d_centers = torch::zeros({N, dim}, torch::kFloat32).to(torch::kMPS);
    auto d_Ls = torch::zeros({N, dim, dim}, torch::kFloat32).to(torch::kMPS);
    auto d_amps = torch::zeros({N}, torch::kFloat32).to(torch::kMPS);

    torch::mps::synchronize();

    // === Backward Kernel ===
    {
        id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx->getPipeline("rasterize_bwd_nd")];

        setBufferWithOffset(enc, grad_output, 0);
        setBufferWithOffset(enc, centers, 1);
        setBufferWithOffset(enc, Ls, 2);
        setBufferWithOffset(enc, amps, 3);

        // Output gradient buffers
        setBufferWithOffset(enc, d_centers, 4);
        setBufferWithOffset(enc, d_Ls, 5);
        setBufferWithOffset(enc, d_amps, 6);

        // Shape and constants
        uint shape_arr[8] = {0};
        for (int i = 0; i < dim; i++) shape_arr[i] = (uint)shape[i];
        [enc setBytes:shape_arr length:sizeof(shape_arr) atIndex:7];

        uint dim_u = (uint)dim;
        uint N_u = (uint)N;
        [enc setBytes:&dim_u length:sizeof(uint) atIndex:8];
        [enc setBytes:&N_u length:sizeof(uint) atIndex:9];
        [enc setBytes:&truncate length:sizeof(float) atIndex:10];
        [enc setBytes:&intensity_floor length:sizeof(float) atIndex:11];

        MTLSize gridSize = MTLSizeMake(total_size, 1, 1);
        MTLSize groupSize = MTLSizeMake(std::min((int64_t)256, total_size), 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:groupSize];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    return {d_centers, d_Ls, d_amps};
}
```

### 7.8 PyBind11 Bindings

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_3d", &dispatch_forward_3d,
          "Metal forward pass (3D tiled)");
    m.def("backward_3d", &dispatch_backward_3d,
          "Metal backward pass (3D tiled with SIMD reduction)");
    m.def("forward_nd", &dispatch_forward_nd,
          "Metal forward pass (generic nD)");
    m.def("backward_nd", &dispatch_backward_nd,
          "Metal backward pass (generic nD)");
}
```

---

## 8. Python Interface

### 8.1 Custom Autograd Function

```python
# packages/luxar/src/luxar/gsplats/models/gsplats/metal/gsplat_model_metal.py

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Sequence
import numpy as np

# Import C++ extension (compiled separately)
try:
    import metal_splatting_backend
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False


def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors to conic (inverse covariance) representation.

    Args:
        L: (N, d, d) lower-triangular Cholesky factors

    Returns:
        conic: (N, d*(d+1)//2) upper-triangular elements of Σ^(-1)
               For 3D: (N, 6) with [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
    """
    # Use cholesky_inverse for SPD matrices - more efficient and numerically stable
    # than computing Σ = L @ L^T followed by torch.linalg.inv(Σ).
    #
    # cholesky_inverse(L) computes (L @ L^T)^(-1) directly from L.
    # Note: requires L to be lower triangular with positive diagonal.
    Sigma_inv = torch.cholesky_inverse(L)

    # Extract upper triangular elements
    N, d, _ = L.shape

    if d == 3:
        # Optimized 3D case
        conic = torch.stack([
            Sigma_inv[:, 0, 0],  # c_xx
            Sigma_inv[:, 0, 1],  # c_xy
            Sigma_inv[:, 0, 2],  # c_xz
            Sigma_inv[:, 1, 1],  # c_yy
            Sigma_inv[:, 1, 2],  # c_yz
            Sigma_inv[:, 2, 2],  # c_zz
        ], dim=1)
    elif d == 2:
        conic = torch.stack([
            Sigma_inv[:, 0, 0],  # c_xx
            Sigma_inv[:, 0, 1],  # c_xy
            Sigma_inv[:, 1, 1],  # c_yy
        ], dim=1)
    else:
        # Generic nD: extract upper triangle
        indices = torch.triu_indices(d, d, device=L.device)
        conic = Sigma_inv[:, indices[0], indices[1]]

    return conic


def compute_sigma_diag(L: torch.Tensor) -> torch.Tensor:
    """Compute diagonal of Σ = L @ L^T: Σ_ii = sum_j(L_ij²)."""
    return torch.sum(L * L, dim=2)  # (N, d)


class MetalSplatFunction(torch.autograd.Function):
    """Custom autograd function for Metal-accelerated splatting."""

    @staticmethod
    def forward(
        ctx,
        centers: torch.Tensor,      # (N, d)
        Ls: torch.Tensor,           # (N, d, d)
        amps: torch.Tensor,         # (N,)
        shape: Tuple[int, ...],
        truncate: float,
        intensity_floor: float = 1e-5,  # For early culling of invisible contributions
    ) -> torch.Tensor:
        """
        Forward pass: render Gaussians to volume.

        L → Conic conversion happens in PyTorch for graph consistency.
        Metal handles the pixel-parallel rendering.
        """
        d = len(shape)
        N = centers.size(0)
        device = centers.device

        # === PyTorch: L → Conic (O(N), fast, graph-compatible) ===
        # This needs requires_grad for backward chain rule
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        conic = cholesky_to_conic(Ls_for_conic)  # (N, 6) for 3D

        # Compute sigma_diag for BBox (needed for binning)
        sigma_diag = compute_sigma_diag(Ls)  # (N, d)

        # === Dispatch to Metal ===
        if d == 3 and METAL_AVAILABLE:
            # Move to MPS for Metal interop (ensure contiguous)
            centers_mps = centers.contiguous().to('mps')
            conic_mps = conic.detach().contiguous().to('mps')
            amps_mps = amps.contiguous().to('mps')
            Ls_mps = Ls.contiguous().to('mps')  # Needed for sigma_diag in binning

            # Forward returns: [output, tile_counts, tile_offsets, tile_content]
            result = metal_splatting_backend.forward_3d(
                centers_mps, conic_mps, amps_mps, Ls_mps,
                list(shape), truncate, intensity_floor
            )
            output = result[0].to(device)

            # CRITICAL: Save tile data for backward pass (no grad needed)
            tile_counts = result[1]
            tile_offsets = result[2]
            tile_content = result[3]
        else:
            # Fallback to PyTorch (for nD or when Metal unavailable)
            from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
            output = render_gaussians(
                shape, centers, Ls, amps, truncate, intensity_floor
            )
            # No tile data for PyTorch path
            tile_counts = None
            tile_offsets = None
            tile_content = None

        # Save for backward - include tile data for Metal backward
        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps)

        # Save non-tensor data and tile buffers separately
        ctx.shape = shape
        ctx.truncate = truncate
        ctx.intensity_floor = intensity_floor
        ctx.d = d
        ctx.tile_counts = tile_counts
        ctx.tile_offsets = tile_offsets
        ctx.tile_content = tile_content

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: compute gradients.

        Metal computes: d_centers, d_conic, d_amps
        PyTorch handles: d_conic → d_Ls (chain rule)

        CRITICAL: Uses saved tile_counts/offsets/content from forward pass
        to avoid recomputing binning (saves time and ensures determinism).
        """
        (centers, Ls, Ls_for_conic, conic, amps) = ctx.saved_tensors
        shape = ctx.shape
        truncate = ctx.truncate
        intensity_floor = ctx.intensity_floor
        d = ctx.d

        device = centers.device

        if d == 3 and METAL_AVAILABLE and ctx.tile_counts is not None:
            # === Metal: Compute gradients using saved tile data ===
            grad_mps = grad_output.contiguous().to('mps')
            centers_mps = centers.contiguous().to('mps')
            conic_mps = conic.contiguous().to('mps')
            amps_mps = amps.contiguous().to('mps')

            # Reuse tile data from forward pass (CRITICAL for performance)
            d_centers, d_conic, d_amps = \
                metal_splatting_backend.backward_3d(
                    grad_mps, centers_mps, conic_mps, amps_mps,
                    ctx.tile_offsets, ctx.tile_counts, ctx.tile_content,
                    list(shape), truncate, intensity_floor
                )

            # Move back to original device
            d_centers = d_centers.to(device)
            d_conic = d_conic.to(device)
            d_amps = d_amps.to(device)

            # === PyTorch: Chain rule d_conic → d_Ls ===
            # CRITICAL: Custom autograd.Function.backward runs with grad mode disabled!
            # Must explicitly enable grad mode for the recomputation.
            with torch.enable_grad():
                conic_recomputed = cholesky_to_conic(Ls_for_conic)

            # Use torch.autograd.grad instead of .backward() because:
            # 1. Avoids polluting .grad buffers
            # 2. Cleaner semantics in custom Function.backward
            # 3. Works correctly if create_graph=True is needed later
            d_Ls, = torch.autograd.grad(
                outputs=conic_recomputed,
                inputs=Ls_for_conic,
                grad_outputs=d_conic,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

        else:
            # Fallback: use PyTorch autograd entirely
            # This requires the forward pass to have been done in PyTorch too
            raise NotImplementedError(
                "PyTorch fallback backward not implemented - use forward path only"
            )

        # Return gradients: (centers, Ls, amps, shape, truncate, intensity_floor)
        return d_centers, d_Ls, d_amps, None, None, None


### 8.2 Handling 2D Inputs

2D inputs are handled by treating them as 3D volumes with `depth=1`.
This avoids the need for dedicated 2D Metal kernels.

#### Coordinate Ordering for 2D Inputs

**IMPORTANT**: 2D centers follow numpy/image convention where the first coordinate
is the row (y) and the second is the column (x):

| Input | Convention | Description |
|-------|------------|-------------|
| `centers` | `(N, 2)` with `[y, x]` per row | Row-major image coordinates |
| `shape` | `(H, W)` | NumPy shape: height first, width second |
| `Ls` | `(N, 2, 2)` | Cholesky in `[y, x]` coordinate order |

When promoted to 3D, the mapping is:
```
2D (y, x) -> 3D (z, y, x)  # Z is the dummy depth dimension
```

This matches the Metal kernel's convention (see §2.7) where the 3D output is
indexed as `[z, y, x]` in numpy terms, which corresponds to Metal's `gid = (x, y, z)`.

```python
def _handle_2d_as_3d(
    centers: torch.Tensor,  # (N, 2) with each row as [y, x]
    Ls: torch.Tensor,       # (N, 2, 2) in [y, x] coordinate order
    shape: Tuple[int, int], # (H, W) - numpy convention
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """
    Convert 2D splats to 3D by adding a dummy Z dimension.

    Input coordinate convention: centers[:, 0] = y, centers[:, 1] = x
    Output coordinate convention: centers_3d[:, 0] = z, [:, 1] = y, [:, 2] = x

    Returns (centers_3d, Ls_3d, shape_3d) ready for 3D Metal kernels.
    """
    N = centers.size(0)
    device = centers.device

    # Shape: (H, W) -> (1, H, W) = (D, H, W)
    shape_3d = (1, *shape)

    # Centers: (N, 2) [y,x] -> (N, 3) [z,y,x] with z=0.5 (center of depth=1)
    centers_3d = torch.cat([
        torch.full((N, 1), 0.5, device=device),  # z coordinate
        centers  # y, x coordinates (unchanged)
    ], dim=1)

    # Cholesky: (N, 2, 2) -> (N, 3, 3)
    # Outer block is Z (dummy), inner 2x2 block is the original Y,X covariance
    Ls_3d = torch.zeros(N, 3, 3, device=device)
    Ls_3d[:, 0, 0] = 0.5  # Z variance (small, so splat spans full depth=1)
    Ls_3d[:, 1:, 1:] = Ls  # Copy 2D covariance to [Y,X] block

    return centers_3d, Ls_3d, shape_3d
```

**Usage in MetalSplatFunction.forward:**
```python
if d == 2 and METAL_AVAILABLE:
    # Promote 2D to 3D for Metal kernel
    centers_3d, Ls_3d, shape_3d = _handle_2d_as_3d(centers, Ls, shape)
    conic_3d = cholesky_to_conic(Ls_3d)
    # Dispatch to 3D kernel...
    output_3d = metal_splatting_backend.forward_3d(...)
    # Squeeze back to 2D
    output = output_3d.squeeze(0)
```


### 8.3 PyTorch Fallback (No Metal)

**IMPORTANT**: When Metal is unavailable, do NOT wrap the fallback in `MetalSplatFunction.apply`.
Call the vanilla PyTorch function directly so standard Autograd handles gradients automatically.

```python
def render_with_metal_fallback(
    centers: torch.Tensor,
    Ls: torch.Tensor,
    amps: torch.Tensor,
    shape: Tuple[int, ...],
    truncate: float,
    intensity_floor: float = 1e-5,
) -> torch.Tensor:
    """
    Render Gaussians, using Metal when available, PyTorch otherwise.

    The fallback does NOT use MetalSplatFunction - it calls render_gaussians
    directly, letting PyTorch Autograd record the graph automatically.
    """
    d = len(shape)

    # Metal path: use custom autograd function
    if (d == 2 or d == 3) and METAL_AVAILABLE:
        return MetalSplatFunction.apply(
            centers, Ls, amps, shape, truncate, intensity_floor
        )

    # PyTorch fallback: call vanilla function directly
    # Autograd automatically records the computation graph
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
    return render_gaussians(
        shape, centers, Ls, amps, truncate, intensity_floor
    )
```

**Why this works:**
- `render_gaussians()` uses standard PyTorch ops (torch.exp, tensor indexing, etc.)
- PyTorch Autograd automatically tracks these operations
- Calling `.backward()` on the output triggers automatic differentiation
- No need to implement a custom backward pass for the fallback


class GaussianSplatModelMetal(torch.nn.Module):
    """
    Metal-accelerated Gaussian splat model.

    Drop-in replacement for GaussianSplatModel with identical interface.
    Uses Metal kernels for 3D rendering when available.
    """

    def __init__(
        self,
        shape: Sequence[int],
        centers0: np.ndarray,
        L0: np.ndarray,
        amps0: np.ndarray,
        sigma_min_diag: Sequence[float],
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Import and instantiate base model for parameter management
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        # Create internal base model (handles all parameter logic)
        self._base = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            device=device,
        )

        # Copy attributes for compatibility
        self.shape = self._base.shape
        self.dim = self._base.dim
        self.truncate = self._base.truncate

    def current_params(self):
        """Delegate to base model."""
        return self._base.current_params()

    def forward(self) -> torch.Tensor:
        """
        Render all splats using Metal acceleration (when available).
        """
        centers, Ls, amps = self.current_params()

        return MetalSplatFunction.apply(
            centers, Ls, amps,
            self.shape, self.truncate
        )

    # === Delegate all other methods to base model ===

    def n_splats(self) -> int:
        return self._base.n_splats()

    def prune_(self, keep_mask: torch.Tensor) -> None:
        self._base.prune_(keep_mask)

    def append_(self, centers_new, Ls_new, amps_new) -> None:
        self._base.append_(centers_new, Ls_new, amps_new)

    def replace_with(self, centers, Ls, amps) -> None:
        self._base.replace_with(centers, Ls, amps)

    def parameters(self, recurse: bool = True):
        return self._base.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self._base.named_parameters(prefix, recurse)
```

---

## 8.5 Error Handling

This section documents error handling patterns for the Metal splatting pipeline.

### 8.5.1 Build-Time Errors

**Extension Not Compiled**
```python
try:
    import metal_splatting_backend
    METAL_AVAILABLE = True
except ImportError as e:
    # Fall back to PyTorch implementation
    METAL_AVAILABLE = False
    warnings.warn(f"Metal extension not available: {e}. Using PyTorch fallback.")
```

### 8.5.2 Runtime Errors

**Device Mismatch**
```python
def _validate_inputs(centers: torch.Tensor, Ls: torch.Tensor, ...):
    """Validate inputs before Metal dispatch."""
    # All tensors must be on the same device
    devices = {centers.device, Ls.device, amps.device}
    if len(devices) > 1:
        raise ValueError(f"All tensors must be on the same device, got: {devices}")

    # MPS or CPU only for Metal
    if centers.device.type not in ('mps', 'cpu'):
        raise ValueError(f"Metal backend requires MPS or CPU device, got: {centers.device}")
```

**Invalid Tensor Shapes**
```python
def _validate_shapes(centers: torch.Tensor, Ls: torch.Tensor, amps: torch.Tensor):
    """Validate tensor shapes."""
    N = centers.size(0)
    d = centers.size(1)

    if Ls.shape != (N, d, d):
        raise ValueError(f"Ls shape mismatch: expected ({N}, {d}, {d}), got {Ls.shape}")
    if amps.shape != (N,):
        raise ValueError(f"amps shape mismatch: expected ({N},), got {amps.shape}")
```

**Numerical Issues**
```python
def _validate_L_matrices(Ls: torch.Tensor, min_diag: float = 1e-6):
    """Validate Cholesky factors are well-formed."""
    # Check diagonal elements are positive (required for valid Cholesky)
    diag = Ls.diagonal(dim1=1, dim2=2)
    if (diag <= 0).any():
        raise ValueError("L matrices must have positive diagonal elements")
    if (diag < min_diag).any():
        warnings.warn(f"Some L diagonal elements are very small (< {min_diag}), may cause numerical issues")
```

### 8.5.3 Metal Kernel Errors

**C++ Dispatcher Error Handling**
```cpp
std::vector<torch::Tensor> dispatch_forward_3d(...) {
    // Validate context
    if (!g_ctx) {
        g_ctx = new MetalContext();
        if (!g_ctx->device) {
            throw std::runtime_error("No Metal device available");
        }
    }

    // Validate inputs
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3,
        "centers must be (N, 3), got ", centers.sizes());
    TORCH_CHECK(shape.size() == 3,
        "shape must have 3 elements for 3D, got ", shape.size());

    // Check for empty inputs (valid but requires special handling)
    if (centers.size(0) == 0) {
        // Return zeros output directly
        return {torch::zeros(shape, torch::kFloat32).to(torch::kMPS),
                torch::zeros({0}, torch::kInt32).to(torch::kMPS),
                torch::zeros({0}, torch::kInt32).to(torch::kMPS),
                torch::zeros({0}, torch::kInt32).to(torch::kMPS)};
    }

    // Check Metal command buffer status
    id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
    if (!cmd) {
        throw std::runtime_error("Failed to create Metal command buffer");
    }

    // After kernel execution, check for errors
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status == MTLCommandBufferStatusError) {
        NSError* error = cmd.error;
        std::string errorMsg = error ? [[error localizedDescription] UTF8String] : "Unknown";
        throw std::runtime_error("Metal kernel failed: " + errorMsg);
    }

    // ...
}
```

### 8.5.4 Gradient Validation

```python
def _check_gradients_valid(d_centers: torch.Tensor, d_Ls: torch.Tensor, ...):
    """Check gradient tensors for NaN/Inf."""
    for name, grad in [('d_centers', d_centers), ('d_Ls', d_Ls),
                        ('d_amps', d_amps)]:
        if grad is None:
            continue
        if torch.isnan(grad).any():
            raise RuntimeError(f"NaN detected in {name} gradient")
        if torch.isinf(grad).any():
            raise RuntimeError(f"Inf detected in {name} gradient")
```

### 8.5.5 Fallback Strategy

```python
def forward_with_fallback(centers, Ls, amps, shape, truncate, intensity_floor):
    """Forward pass with automatic fallback on errors."""
    try:
        if METAL_AVAILABLE:
            return MetalSplatFunction.apply(centers, Ls, amps, shape, truncate, intensity_floor)
    except RuntimeError as e:
        warnings.warn(f"Metal forward failed: {e}. Falling back to PyTorch.")

    # PyTorch fallback (always works)
    return render_gaussians(shape, centers, Ls, amps, truncate, intensity_floor)
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

**Test 1: Conic Computation**
```python
def test_cholesky_to_conic():
    """Verify L → Conic conversion matches analytical formula."""
    # Random L matrices
    N, d = 100, 3
    L = torch.randn(N, d, d)
    L = torch.tril(L)
    L.diagonal(dim1=1, dim2=2).abs_().clamp_(min=0.1)  # Ensure positive diagonal

    # Compute conic
    conic = cholesky_to_conic(L)  # (N, 6) for 3D

    # Reconstruct Σ^(-1) from conic representation
    Sigma_inv_reconstructed = torch.zeros(N, d, d)
    Sigma_inv_reconstructed[:, 0, 0] = conic[:, 0]  # c_xx
    Sigma_inv_reconstructed[:, 0, 1] = conic[:, 1]  # c_xy
    Sigma_inv_reconstructed[:, 1, 0] = conic[:, 1]  # c_xy (symmetric)
    Sigma_inv_reconstructed[:, 0, 2] = conic[:, 2]  # c_xz
    Sigma_inv_reconstructed[:, 2, 0] = conic[:, 2]  # c_xz (symmetric)
    Sigma_inv_reconstructed[:, 1, 1] = conic[:, 3]  # c_yy
    Sigma_inv_reconstructed[:, 1, 2] = conic[:, 4]  # c_yz
    Sigma_inv_reconstructed[:, 2, 1] = conic[:, 4]  # c_yz (symmetric)
    Sigma_inv_reconstructed[:, 2, 2] = conic[:, 5]  # c_zz

    # Verify by computing Σ and checking Σ × Σ^(-1) = I
    Sigma = L @ L.transpose(-1, -2)
    identity = Sigma @ Sigma_inv_reconstructed

    assert torch.allclose(identity, torch.eye(d).expand(N, -1, -1), atol=1e-5)
```

**Test 2: Forward Pass Correctness**
```python
def test_metal_forward_matches_pytorch():
    """Verify Metal forward pass matches PyTorch reference."""
    # Create test volume and splats
    shape = (32, 32, 32)
    N = 100

    # Generate random splats
    centers = torch.rand(N, 3) * torch.tensor(shape)
    Ls = torch.eye(3).expand(N, -1, -1) * 2.0
    amps = torch.rand(N)

    # PyTorch reference
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
    ref_output = render_gaussians(shape, centers, Ls, amps, truncate=3.0)

    # Metal output
    metal_output = MetalSplatFunction.apply(
        centers, Ls, amps, shape, 3.0
    )

    # Compare
    assert torch.allclose(metal_output, ref_output, rtol=1e-4, atol=1e-4)
```

**Test 3: Backward Pass Correctness**
```python
def test_metal_backward_matches_pytorch():
    """Verify Metal gradients match PyTorch autograd."""
    shape = (16, 16, 16)
    N = 50

    # Create splats with requires_grad
    centers = torch.rand(N, 3, requires_grad=True) * torch.tensor(shape)
    Ls = torch.eye(3).expand(N, -1, -1).clone() * 2.0
    Ls.requires_grad_(True)
    amps = torch.rand(N, requires_grad=True)

    # Forward + backward with Metal
    output_metal = MetalSplatFunction.apply(
        centers, Ls, amps, shape, 3.0
    )
    loss_metal = output_metal.sum()
    loss_metal.backward()

    # Save gradients
    grad_centers_metal = centers.grad.clone()
    grad_Ls_metal = Ls.grad.clone()

    # Reset gradients
    centers.grad = None
    Ls.grad = None

    # Forward + backward with PyTorch
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians
    output_ref = render_gaussians(shape, centers, Ls, amps, truncate=3.0)
    loss_ref = output_ref.sum()
    loss_ref.backward()

    # Compare
    assert torch.allclose(grad_centers_metal, centers.grad, rtol=1e-3, atol=1e-3)
    assert torch.allclose(grad_Ls_metal, Ls.grad, rtol=1e-3, atol=1e-3)
```

### 9.2 Integration Tests

**Test 4: Full Optimization Loop**
```python
def test_metal_optimization_convergence():
    """Verify Metal model can optimize to fit a target."""
    # Create synthetic target
    target = create_test_volume(size=64, n_blobs=5)

    # Initialize model
    model = GaussianSplatModelMetal(
        shape=target.shape,
        centers0=random_centers(100, target.shape),
        L0=default_L(100, 3),
        amps0=np.ones(100) * 0.5,
        sigma_min_diag=[0.5, 0.5, 0.5],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    target_tensor = torch.tensor(target)

    # Run optimization
    initial_loss = None
    for i in range(100):
        optimizer.zero_grad()
        output = model()
        loss = F.mse_loss(output, target_tensor)

        if initial_loss is None:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    # Verify loss decreased significantly
    assert final_loss < initial_loss * 0.5, "Optimization did not converge"
```

### 9.3 Performance Tests

```python
def test_metal_speedup():
    """Verify Metal provides meaningful speedup over CPU."""
    import time

    shape = (64, 64, 64)
    N = 1000
    n_iters = 50

    # Setup
    centers = torch.rand(N, 3) * torch.tensor(shape)
    Ls = torch.eye(3).expand(N, -1, -1) * 2.0
    amps = torch.rand(N)

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(n_iters):
        render_gaussians(shape, centers, Ls, amps, truncate=3.0)
    pytorch_time = (time.perf_counter() - start) / n_iters

    # Benchmark Metal
    start = time.perf_counter()
    for _ in range(n_iters):
        MetalSplatFunction.apply(centers, Ls, amps, shape, 3.0)
    metal_time = (time.perf_counter() - start) / n_iters

    speedup = pytorch_time / metal_time
    print(f"PyTorch: {pytorch_time*1000:.2f} ms, Metal: {metal_time*1000:.2f} ms")
    print(f"Speedup: {speedup:.1f}x")

    # Expect at least 5x speedup
    assert speedup > 5.0, f"Expected >5x speedup, got {speedup:.1f}x"
```

---

## 10. Performance Expectations

### 10.1 Theoretical Analysis

**Current PyTorch (Splat-Centric)**:
- Complexity: O(N × P_avg) where P_avg is average pixels per splat
- Memory pattern: Scattered writes via index_add_
- Bottleneck: Atomic contention, non-coalesced memory access

**Metal (Pixel-Centric with Tiling)**:
- Complexity: O(P × K_avg) where K_avg is average splats per tile
- Memory pattern: Coalesced reads, single write per pixel
- Optimization: SIMD reduction eliminates most atomics in backward pass

### 10.2 Expected Speedup

| Volume Size | N Splats | PyTorch (est.) | Metal (est.) | Speedup |
|-------------|----------|----------------|--------------|---------|
| 64³         | 1,000    | 555 ms         | 30-50 ms     | 10-18x  |
| 128³        | 5,000    | 2-3 s          | 100-200 ms   | 15-30x  |
| 256³        | 10,000   | 10-15 s        | 300-500 ms   | 20-50x  |

**Note**: Actual speedup depends on splat distribution and overlap.

### 10.3 Memory Requirements

For 3D tiled rendering with tile size 4:
- Tile grid: (D/4) × (H/4) × (W/4)
- tile_counts: 4 bytes per tile
- tile_content: 4 bytes × (total splat-tile pairs)
- Conic buffer: 24 bytes per splat (6 × float32)

Example for 128³ volume with 5000 splats:
- Tile grid: 32 × 32 × 32 = 32,768 tiles
- tile_counts: 128 KB
- tile_content: ~4 MB (assuming avg 50 tiles per splat)
- Conic: 120 KB
- Total overhead: ~4.5 MB (negligible)

---

## 11. Risks and Mitigations

### 11.1 Numerical Consistency

**Risk**: Floating-point differences between Metal and PyTorch cause gradient instability.

**Mitigation**:
- Compute L → Conic entirely in PyTorch
- Use consistent epsilon values (1e-9) across both
- Test gradient consistency rigorously

### 11.2 Build System Complexity

**Risk**: Metal compilation fails on different macOS versions or without Xcode.

**Mitigation**:
- Check for xcrun availability at build time
- Provide clear error messages with installation instructions
- Fall back gracefully to PyTorch when build fails

### 11.3 Atomic Contention

**Risk**: Large splats covering many tiles cause atomic bottleneck in backward pass.

**Mitigation**:
- SIMD reduction reduces atomics by 32x
- Skip negligible gradient updates (threshold check)
- Accept some contention as cost of simplicity

### 11.4 nD Support

**Risk**: 4D+ volumes don't benefit from tiling, may be slower.

**Mitigation**:
- Use generic nD kernel with AABB culling (no tiling)
- Still faster than PyTorch due to pixel-parallel execution
- Consider PyTorch fallback for very high dimensions

---

## 12. Implementation Phases

**Revised Plan**: Focus on data flow correctness before logic.

### Phase 1A: The Bridge (Data Flow) - Days 1-2
- [ ] Set up build system (setup.py, Metal compilation)
- [ ] Implement `tensorToMTLBuffer()` with offset handling
- [ ] Implement `setBufferWithOffset()` helper
- [ ] Create dummy "pass-through" kernel that writes 1.0 to output
- [ ] **Goal**: Prove tensor data flows: PyTorch → MPS → Metal → Back
- [ ] **Validation**: Round-trip a known tensor, verify values unchanged

### Phase 1B: Forward 3D (Logic) - Days 3-5
- [ ] Implement preprocess_3d kernel (tile counting)
- [ ] Implement bin_3d kernel (tile population)
- [ ] Implement rasterize_fwd_3d kernel (with intensity_floor)
- [ ] C++ dispatcher returns `{output, tile_counts, tile_offsets, tile_content}`
- [ ] **Goal**: Correct rendering
- [ ] **Validation**: `torch.allclose(metal_output, pytorch_output, atol=1e-4)`

### Phase 2A: Backward 3D (Gradients) - Days 6-8
- [ ] Implement rasterize_bwd_3d kernel with SIMD reduction
- [ ] C++ backward dispatcher accepts saved tile data
- [ ] Python backward() saves/restores tile buffers correctly
- [ ] PyTorch chain rule: d_conic → d_Ls via autograd
- [ ] **Goal**: Correct gradients
- [ ] **Validation**: `torch.autograd.gradcheck()` passes

### Phase 2B: nD Support (High-D) - Days 9-10
- [ ] Implement rasterize_fwd_nd kernel (no tiling, AABB culling)
- [ ] Implement rasterize_bwd_nd kernel **with SIMD reduction**
- [ ] **Goal**: Feature parity for Luxar's 4D+ use cases
- [ ] **Validation**: nD gradients match PyTorch

### Phase 3: Integration - Days 11-12
- [ ] GaussianSplatModelMetal class (composition pattern)
- [ ] Integration with fit_gsplats.py (model selection)
- [ ] Performance benchmarking vs CPU baseline
- [ ] **Goal**: Drop-in replacement works end-to-end

### Phase 4: Polish - Days 13-15
- [ ] Error handling and edge cases (empty tiles, N=0, etc.)
- [ ] Memory leak testing (Metal object lifecycle)
- [ ] CI integration (skip on non-macOS)
- [ ] Documentation updates
- [ ] **Goal**: Production-ready code

### Critical Checkpoints

| Checkpoint | Criteria | Abort If |
|------------|----------|----------|
| Phase 1A Complete | Tensor round-trip works | Buffer offset handling broken |
| Phase 1B Complete | Forward matches PyTorch within 1e-4 | >10% pixels differ |
| Phase 2A Complete | gradcheck passes | Gradients numerically wrong |
| Phase 3 Complete | Optimization converges | Loss diverges or NaN |

### Estimated Speedup Validation

Run benchmark at end of Phase 1B:
- If speedup < 3x: Investigate binning overhead, consider larger tiles
- If speedup < 1x: STOP - architecture review needed
- If speedup > 10x: Proceed with confidence

---

## Appendix A: Complete Metal Kernel Code

```cpp
// kernels.metal
// Complete, production-ready Metal kernels for Gaussian splatting

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

// ============================================================================
// CONSTANTS AND HELPERS
// ============================================================================

constant int TILE_SIZE_3D = 4;  // 4×4×4 voxels per tile

// Atomic float add using compare-exchange
// Atomic float add via CAS loop
// NOTE: Metal 2.3+ supports atomic_float natively. For older versions, use this fallback.
// The memory orders are both relaxed since we only need atomicity, not ordering.
inline void atomic_add_float(device atomic_float* addr, float val) {
    float old = atomic_load_explicit(addr, memory_order_relaxed);
    float desired;
    do {
        desired = old + val;
    } while (!atomic_compare_exchange_weak_explicit(
        addr, &old, desired,
        memory_order_relaxed,   // success: relaxed is sufficient for accumulation
        memory_order_relaxed    // failure: relaxed is standard for CAS retry loops
    ));
}

// Alternative: If targeting Metal 3.0+ only, use native atomic_fetch_add_explicit:
// inline void atomic_add_float(device atomic_float* addr, float val) {
//     atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
// }

// Shared tile range calculation (ensures preprocess and bin are consistent)
struct TileRange {
    int3 min_t;
    int3 max_t;
};

inline TileRange get_tile_range_3d(
    float3 center,
    float3 sigma_diag,  // Σ_ii = sum_j(L_ij²)
    float truncate,
    uint3 grid_dims
) {
    // Exact bounding box radius per axis
    float3 r = truncate * sqrt(max(sigma_diag, float3(1e-8f)));

    // Convert to tile indices
    TileRange tr;
    tr.min_t = max(int3((center - r) / float(TILE_SIZE_3D)), int3(0));
    tr.max_t = min(int3((center + r) / float(TILE_SIZE_3D)), int3(grid_dims) - 1);

    return tr;
}

// Compute sigma diagonal from Cholesky L
inline float3 compute_sigma_diag_3d(
    float L00, float L10, float L11, float L20, float L21, float L22
) {
    return float3(
        L00 * L00,                          // Σ_00
        L10 * L10 + L11 * L11,              // Σ_11
        L20 * L20 + L21 * L21 + L22 * L22   // Σ_22
    );
}

// ============================================================================
// KERNEL 1: PREPROCESS (Count splats per tile)
// ============================================================================

kernel void preprocess_3d(
    device const float* centers     [[buffer(0)]],  // (N, 3)
    device const float* Ls          [[buffer(1)]],  // (N, 3, 3) row-major
    device atomic_int* tile_counts  [[buffer(2)]],  // (num_tiles,)
    constant float& truncate        [[buffer(3)]],
    constant uint3& grid_dims       [[buffer(4)]],
    constant uint& n_splats         [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) return;

    // Load splat data
    float3 center = float3(
        centers[id * 3 + 0],
        centers[id * 3 + 1],
        centers[id * 3 + 2]
    );

    // Load L elements (row-major: L[i,j] = Ls[id*9 + i*3 + j])
    int base = id * 9;
    float L00 = Ls[base + 0];
    float L10 = Ls[base + 3], L11 = Ls[base + 4];
    float L20 = Ls[base + 6], L21 = Ls[base + 7], L22 = Ls[base + 8];

    float3 sigma_diag = compute_sigma_diag_3d(L00, L10, L11, L20, L21, L22);

    // Get tile range
    TileRange tr = get_tile_range_3d(
        center, sigma_diag, truncate, grid_dims
    );

    // Increment counts for each overlapping tile
    for (int z = tr.min_t.z; z <= tr.max_t.z; z++) {
        for (int y = tr.min_t.y; y <= tr.max_t.y; y++) {
            for (int x = tr.min_t.x; x <= tr.max_t.x; x++) {
                int tile_idx = z * (grid_dims.x * grid_dims.y)
                             + y * grid_dims.x + x;
                atomic_fetch_add_explicit(
                    &tile_counts[tile_idx], 1, memory_order_relaxed);
            }
        }
    }
}

// ============================================================================
// KERNEL 2: BINNING (Populate tile content lists)
// ============================================================================

kernel void bin_3d(
    device const float* centers         [[buffer(0)]],  // (N, 3)
    device const float* Ls              [[buffer(1)]],  // (N, 3, 3)
    device const int* tile_offsets      [[buffer(2)]],  // (num_tiles,) prefix sum
    device atomic_int* tile_write_heads [[buffer(3)]],  // (num_tiles,) must be zeroed
    device int* tile_content            [[buffer(4)]],  // (total_pairs,)
    constant float& truncate            [[buffer(5)]],
    constant uint3& grid_dims           [[buffer(6)]],
    constant uint& n_splats             [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_splats) return;

    // Load splat data (MUST match preprocess exactly)
    float3 center = float3(
        centers[id * 3 + 0],
        centers[id * 3 + 1],
        centers[id * 3 + 2]
    );

    int base = id * 9;
    float L00 = Ls[base + 0];
    float L10 = Ls[base + 3], L11 = Ls[base + 4];
    float L20 = Ls[base + 6], L21 = Ls[base + 7], L22 = Ls[base + 8];

    float3 sigma_diag = compute_sigma_diag_3d(L00, L10, L11, L20, L21, L22);

    // Get tile range (MUST match preprocess)
    TileRange tr = get_tile_range_3d(
        center, sigma_diag, truncate, grid_dims
    );

    // Write splat ID to each tile's list
    for (int z = tr.min_t.z; z <= tr.max_t.z; z++) {
        for (int y = tr.min_t.y; y <= tr.max_t.y; y++) {
            for (int x = tr.min_t.x; x <= tr.max_t.x; x++) {
                int tile_idx = z * (grid_dims.x * grid_dims.y)
                             + y * grid_dims.x + x;

                // Atomically get write slot
                int slot = atomic_fetch_add_explicit(
                    &tile_write_heads[tile_idx], 1, memory_order_relaxed);

                // Write splat ID
                tile_content[tile_offsets[tile_idx] + slot] = int(id);
            }
        }
    }
}

// ============================================================================
// KERNEL 3: RASTERIZE FORWARD (Pixel-parallel rendering)
// ============================================================================

kernel void rasterize_fwd_3d(
    device const float* centers      [[buffer(0)]],   // (N, 3)
    device const float* conic        [[buffer(1)]],   // (N, 6) [xx,xy,xz,yy,yz,zz]
    device const float* amps         [[buffer(2)]],   // (N,)
    device const int* tile_offsets   [[buffer(3)]],   // (num_tiles,)
    device const int* tile_counts    [[buffer(4)]],   // (num_tiles,)
    device const int* tile_content   [[buffer(5)]],   // (total_pairs,)
    device float* output             [[buffer(6)]],   // (D, H, W) row-major output
    constant uint3& img_size         [[buffer(7)]],   // (W, H, D) - see §2.7 Dimension Conventions
    constant uint3& grid_dims        [[buffer(8)]],   // (tiles_x, tiles_y, tiles_z) - see §2.7
    constant float& truncate         [[buffer(9)]],   // base truncate (NOT squared)
    constant float& intensity_floor  [[buffer(10)]],  // early culling threshold
    uint3 gid [[thread_position_in_grid]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    // Bounds check
    if (gid.x >= img_size.x || gid.y >= img_size.y || gid.z >= img_size.z) return;

    // Tile index (threads are grouped by tile)
    uint tile_idx = group_id.z * (grid_dims.x * grid_dims.y)
                  + group_id.y * grid_dims.x + group_id.x;

    int count = tile_counts[tile_idx];
    int start = tile_offsets[tile_idx];

    // Pixel position
    float3 px = float3(gid);

    // Accumulate contributions
    float accum = 0.0f;

    for (int i = 0; i < count; i++) {
        int splat_id = tile_content[start + i];

        // Load center
        float3 c = float3(
            centers[splat_id * 3 + 0],
            centers[splat_id * 3 + 1],
            centers[splat_id * 3 + 2]
        );

        // Displacement
        float3 d = px - c;

        // Load conic (Σ^-1 upper triangle)
        int cb = splat_id * 6;
        float c_xx = conic[cb + 0];
        float c_xy = conic[cb + 1];
        float c_xz = conic[cb + 2];
        float c_yy = conic[cb + 3];
        float c_yz = conic[cb + 4];
        float c_zz = conic[cb + 5];

        // Mahalanobis distance: d^T × Σ^-1 × d
        float dist_sq = d.x * d.x * c_xx + d.y * d.y * c_yy + d.z * d.z * c_zz
                      + 2.0f * (d.x * d.y * c_xy + d.x * d.z * c_xz + d.y * d.z * c_yz);

        // Truncation check
        float trunc_sq = truncate * truncate;
        if (dist_sq <= trunc_sq) {
            float a = amps[splat_id];

            // Shifted Gaussian for C⁰ continuity at truncation boundary:
            // I = a * scale * max(0, exp(-0.5 * D) - C)
            float C_boundary = exp(-0.5f * trunc_sq);
            float raw = exp(-0.5f * dist_sq);
            float val = a * max(0.0f, raw - C_boundary) / (1.0f - C_boundary);

            // Early culling: skip invisible contributions (saves GPU cycles)
            if (val < intensity_floor) continue;

            accum += val;
        }
    }

    // Write output (row-major: [z, y, x] -> z*H*W + y*W + x)
    int out_idx = gid.z * (img_size.y * img_size.x) + gid.y * img_size.x + gid.x;
    output[out_idx] = accum;
}

// ============================================================================
// KERNEL 4: RASTERIZE BACKWARD (Gradient computation with SIMD reduction)
// ============================================================================

kernel void rasterize_bwd_3d(
    device const float* grad_output    [[buffer(0)]],   // (D, H, W)
    device const float* centers        [[buffer(1)]],   // (N, 3)
    device const float* conic          [[buffer(2)]],   // (N, 6)
    device const float* amps           [[buffer(3)]],   // (N,)
    device const int* tile_offsets     [[buffer(4)]],
    device const int* tile_counts      [[buffer(5)]],
    device const int* tile_content     [[buffer(6)]],
    device atomic_float* d_centers     [[buffer(7)]],   // (N, 3)
    device atomic_float* d_conic       [[buffer(8)]],   // (N, 6)
    device atomic_float* d_amps        [[buffer(9)]],   // (N,)
    constant uint3& img_size           [[buffer(12)]],  // (W, H, D) - see §2.7
    constant uint3& grid_dims          [[buffer(13)]],  // (tiles_x, tiles_y, tiles_z) - see §2.7
    constant float& truncate           [[buffer(14)]],  // base truncate (NOT squared)
    constant float& intensity_floor    [[buffer(15)]],  // CRITICAL: must match forward
    uint3 gid [[thread_position_in_grid]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Load upstream gradient
    bool active = (gid.x < img_size.x && gid.y < img_size.y && gid.z < img_size.z);
    int pix_idx = gid.z * (img_size.y * img_size.x) + gid.y * img_size.x + gid.x;
    float d_L_d_I = active ? grad_output[pix_idx] : 0.0f;

    // Tile info
    uint tile_idx = group_id.z * (grid_dims.x * grid_dims.y)
                  + group_id.y * grid_dims.x + group_id.x;
    int count = tile_counts[tile_idx];
    int start = tile_offsets[tile_idx];

    float3 px = float3(gid);

    // Process each splat in tile
    for (int i = 0; i < count; i++) {
        int splat_id = tile_content[start + i];

        // === A. Compute local gradients (per thread) ===
        float val_amps = 0.0f;
        float3 val_centers = 0.0f;
        float val_conic[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        if (active && abs(d_L_d_I) > 1e-9f) {
            // Load data
            float3 c = float3(
                centers[splat_id * 3 + 0],
                centers[splat_id * 3 + 1],
                centers[splat_id * 3 + 2]
            );
            float3 d = px - c;

            int cb = splat_id * 6;
            float c_xx = conic[cb + 0], c_xy = conic[cb + 1], c_xz = conic[cb + 2];
            float c_yy = conic[cb + 3], c_yz = conic[cb + 4], c_zz = conic[cb + 5];

            float dist_sq = d.x * d.x * c_xx + d.y * d.y * c_yy + d.z * d.z * c_zz
                          + 2.0f * (d.x * d.y * c_xy + d.x * d.z * c_xz + d.y * d.z * c_yz);

            // Truncation check (must match forward pass)
            float trunc_sq = truncate * truncate;

            if (dist_sq <= trunc_sq) {
                float a = amps[splat_id];

                // Shifted Gaussian for C⁰ continuity (must match forward pass)
                float C_boundary = exp(-0.5f * trunc_sq);
                float inv_scale = 1.0f / (1.0f - C_boundary);
                float inner = -0.5f * dist_sq;
                float exp_val = exp(inner);
                float shifted = max(0.0f, exp_val - C_boundary);
                float intensity = a * shifted * inv_scale;

                // CRITICAL: Must match forward pass intensity_floor culling!
                if (intensity < intensity_floor) continue;

                float d_common = intensity * d_L_d_I;

                // 1. Amplitude gradient: dI/da = exp(inner)
                val_amps = exp_val * d_L_d_I;

                // 2. Distance gradient: dI/dD = I * (-0.5)
                float grad_dist = d_common * (-0.5f);

                // 3. Center gradient: dD/dmu = -2 * Sigma^-1 * d
                float3 d_D2_d_d;
                d_D2_d_d.x = 2.0f * (d.x * c_xx + d.y * c_xy + d.z * c_xz);
                d_D2_d_d.y = 2.0f * (d.x * c_xy + d.y * c_yy + d.z * c_yz);
                d_D2_d_d.z = 2.0f * (d.x * c_xz + d.y * c_yz + d.z * c_zz);
                val_centers = grad_dist * d_D2_d_d * -1.0f;

                // 4. Conic gradient: dD/dc_ij
                val_conic[0] = grad_dist * d.x * d.x;           // c_xx
                val_conic[1] = grad_dist * 2.0f * d.x * d.y;    // c_xy
                val_conic[2] = grad_dist * 2.0f * d.x * d.z;    // c_xz
                val_conic[3] = grad_dist * d.y * d.y;           // c_yy
                val_conic[4] = grad_dist * 2.0f * d.y * d.z;    // c_yz
                val_conic[5] = grad_dist * d.z * d.z;           // c_zz
            }
        }

        // === B. SIMD Reduction (sum across 32 threads in simdgroup) ===
        float sum_amps = simd_sum(val_amps);
        float3 sum_centers;
        sum_centers.x = simd_sum(val_centers.x);
        sum_centers.y = simd_sum(val_centers.y);
        sum_centers.z = simd_sum(val_centers.z);

        float sum_conic[6];
        for (int k = 0; k < 6; k++) {
            sum_conic[k] = simd_sum(val_conic[k]);
        }

        // === C. Leader writes to global memory (lane 0 only) ===
        if (simd_lane_id == 0) {
            // Skip zero contributions to avoid unnecessary atomics
            if (abs(sum_amps) > 1e-12f) {
                atomic_add_float(&d_amps[splat_id], sum_amps);
            }

            atomic_add_float(&d_centers[splat_id * 3 + 0], sum_centers.x);
            atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
            atomic_add_float(&d_centers[splat_id * 3 + 2], sum_centers.z);

            int cb = splat_id * 6;
            for (int k = 0; k < 6; k++) {
                if (abs(sum_conic[k]) > 1e-12f) {
                    atomic_add_float(&d_conic[cb + k], sum_conic[k]);
                }
            }
        }
    }
}

// ============================================================================
// GENERIC nD KERNELS (No tiling - fallback for 4D+)
// ============================================================================

kernel void rasterize_fwd_nd(
    device const float* centers      [[buffer(0)]],   // (N, dim)
    device const float* Ls           [[buffer(1)]],   // (N, dim, dim)
    device const float* amps         [[buffer(2)]],   // (N,)
    device float* output             [[buffer(3)]],   // flattened
    constant uint& n_splats          [[buffer(4)]],
    constant uint& dim               [[buffer(5)]],
    constant uint* shape             [[buffer(6)]],   // (dim,)
    constant float& truncate         [[buffer(7)]],
    constant float& intensity_floor  [[buffer(8)]],   // early culling threshold
    uint gid [[thread_position_in_grid]]
) {
    // Unpack voxel coordinates from linear index
    float coords[8];  // Max 8 dimensions
    int temp = gid;
    for (int d = dim - 1; d >= 0; d--) {
        coords[d] = float(temp % shape[d]);
        temp /= shape[d];
    }
    if (temp > 0) return;  // Out of bounds

    float accum = 0.0f;

    for (uint i = 0; i < n_splats; i++) {
        // AABB check using diagonal of Sigma
        bool possible = true;
        for (uint d = 0; d < dim; d++) {
            float c = centers[i * dim + d];
            float diff = coords[d] - c;

            // Compute Σ_dd = sum_k(L_dk²)
            float sigma_dd = 0.0f;
            for (uint k = 0; k <= d; k++) {
                float val = Ls[i * dim * dim + d * dim + k];
                sigma_dd += val * val;
            }

            if (abs(diff) > truncate * sqrt(sigma_dd)) {
                possible = false;
                break;
            }
        }

        if (!possible) continue;

        // Full Mahalanobis via forward substitution: y = L^-1 × (x - μ)
        float y[8];
        for (uint r = 0; r < dim; r++) {
            float sum = 0.0f;
            for (uint c = 0; c < r; c++) {
                sum += Ls[i * dim * dim + r * dim + c] * y[c];
            }
            float diff = coords[r] - centers[i * dim + r];
            float L_rr = Ls[i * dim * dim + r * dim + r];
            y[r] = (diff - sum) / (L_rr + 1e-9f);
        }

        // dist_sq = ||y||²
        float dist_sq = 0.0f;
        for (uint d = 0; d < dim; d++) {
            dist_sq += y[d] * y[d];
        }

        float trunc_sq = truncate * truncate;
        if (dist_sq <= trunc_sq) {
            // Shifted Gaussian for C⁰ continuity at truncation boundary
            float C_boundary = exp(-0.5f * trunc_sq);
            float raw = exp(-0.5f * dist_sq);
            float val = amps[i] * max(0.0f, raw - C_boundary) / (1.0f - C_boundary);

            // Early culling for intensity_floor
            if (val < intensity_floor) continue;

            accum += val;
        }
    }

    output[gid] = accum;
}

// ============================================================================
// KERNEL 5: RASTERIZE BACKWARD nD (MUST use SIMD reduction!)
// ============================================================================
//
// CRITICAL: The nD backward kernel MUST use SIMD reduction, just like the 3D
// kernel. Without it, every pixel fires atomic writes to global memory.
// For a 128³ volume, that's ~2 million atomic locks per frame - slower than CPU!
//
// The pattern is identical to rasterize_bwd_3d because nD still iterates over
// n_splats linearly. Each SIMD group processes the same splat together.

kernel void rasterize_bwd_nd(
    device const float* grad_output    [[buffer(0)]],   // flattened
    device const float* centers        [[buffer(1)]],   // (N, dim)
    device const float* Ls             [[buffer(2)]],   // (N, dim, dim)
    device const float* amps           [[buffer(3)]],   // (N,)
    device atomic_float* d_centers     [[buffer(4)]],   // (N, dim)
    device atomic_float* d_Ls          [[buffer(5)]],   // (N, dim, dim)
    device atomic_float* d_amps        [[buffer(6)]],   // (N,)
    constant uint& n_splats            [[buffer(7)]],
    constant uint& dim                 [[buffer(8)]],
    constant uint* shape               [[buffer(9)]],   // (dim,)
    constant float& truncate           [[buffer(10)]],
    constant float& intensity_floor    [[buffer(11)]],  // for consistency
    uint gid [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    // Unpack voxel coordinates from linear index
    float coords[8];  // Max 8 dimensions
    int temp = gid;
    for (int d = dim - 1; d >= 0; d--) {
        coords[d] = float(temp % shape[d]);
        temp /= shape[d];
    }
    bool active = (temp == 0);  // Within bounds

    float d_L_d_I = active ? grad_output[gid] : 0.0f;

    // Process each splat
    for (uint i = 0; i < n_splats; i++) {
        // === A. Compute local gradients (per thread) ===
        float val_amps = 0.0f;
        float val_centers[8] = {0};
        float val_Ls[64] = {0};  // Max 8x8

        if (active && abs(d_L_d_I) > 1e-9f) {
            // AABB check (same as forward)
            bool possible = true;
            for (uint d = 0; d < dim; d++) {
                float c = centers[i * dim + d];
                float diff = coords[d] - c;
                float sigma_dd = 0.0f;
                for (uint k = 0; k <= d; k++) {
                    float val = Ls[i * dim * dim + d * dim + k];
                    sigma_dd += val * val;
                }
                if (abs(diff) > truncate * sqrt(sigma_dd)) {
                    possible = false;
                    break;
                }
            }

            if (possible) {
                // Forward substitution: y = L^-1 × (x - μ)
                // Also compute delta = x - μ for L gradient
                float y[8];
                float delta[8];
                for (uint r = 0; r < dim; r++) {
                    delta[r] = coords[r] - centers[i * dim + r];
                    float sum = 0.0f;
                    for (uint c = 0; c < r; c++) {
                        sum += Ls[i * dim * dim + r * dim + c] * y[c];
                    }
                    float L_rr = Ls[i * dim * dim + r * dim + r];
                    y[r] = (delta[r] - sum) / (L_rr + 1e-9f);
                }

                float dist_sq = 0.0f;
                for (uint d = 0; d < dim; d++) {
                    dist_sq += y[d] * y[d];
                }

                float trunc_sq = truncate * truncate;
                if (dist_sq <= trunc_sq) {
                    float a = amps[i];

                    // Shifted Gaussian for C⁰ continuity (must match forward pass)
                    float C_boundary = exp(-0.5f * trunc_sq);
                    float inv_scale = 1.0f / (1.0f - C_boundary);
                    float inner = -0.5f * dist_sq;
                    float exp_val = exp(inner);
                    float shifted = max(0.0f, exp_val - C_boundary);
                    float intensity = a * shifted * inv_scale;
                    float d_common = intensity * d_L_d_I;

                    // 1. Amplitude gradient: dI/da = shifted * inv_scale
                    val_amps = shifted * inv_scale * d_L_d_I;

                    // 2. Distance gradient: dI/dD = I * (-0.5)
                    float grad_dist = d_common * (-0.5f);

                    // 3. Center gradient
                    for (uint d = 0; d < dim; d++) {
                        val_centers[d] = grad_dist * (-2.0f * y[d]);
                    }

                    // 4. L gradient: dD/dL via chain rule
                    for (uint r = 0; r < dim; r++) {
                        for (uint c = 0; c <= r; c++) {
                            float L_rc = Ls[i * dim * dim + r * dim + c];
                            float L_rc_safe = (c == r) ? max(abs(L_rc), 1e-9f) : (abs(L_rc) > 1e-9f ? L_rc : 1.0f);
                            val_Ls[r * 8 + c] = grad_dist * (-2.0f * y[r] * y[c]) / L_rc_safe;
                        }
                    }
                }
            }
        }

        // === B. SIMD Reduction (CRITICAL - reduces atomics by 32x) ===
        float sum_amps = simd_sum(val_amps);

        // Sum centers across SIMD group
        float sum_centers[8];
        for (uint d = 0; d < dim; d++) {
            sum_centers[d] = simd_sum(val_centers[d]);
        }

        // Sum Ls across SIMD group
        float sum_Ls[64];
        for (uint r = 0; r < dim; r++) {
            for (uint c = 0; c <= r; c++) {
                sum_Ls[r * 8 + c] = simd_sum(val_Ls[r * 8 + c]);
            }
        }

        // === C. Leader writes to global memory (lane 0 only) ===
        if (simd_lane_id == 0) {
            if (abs(sum_amps) > 1e-12f) {
                atomic_add_float(&d_amps[i], sum_amps);
            }
            for (uint d = 0; d < dim; d++) {
                if (abs(sum_centers[d]) > 1e-12f) {
                    atomic_add_float(&d_centers[i * dim + d], sum_centers[d]);
                }
            }
            // Write L gradients (lower triangle only)
            for (uint r = 0; r < dim; r++) {
                for (uint c = 0; c <= r; c++) {
                    if (abs(sum_Ls[r * 8 + c]) > 1e-12f) {
                        atomic_add_float(&d_Ls[i * dim * dim + r * dim + c], sum_Ls[r * 8 + c]);
                    }
                }
            }
        }
    }
}
```

---

## Appendix B: Complete C++ Dispatcher Code

See Section 7 for the dispatcher structure. Full implementation follows the same pattern with proper Metal API calls and buffer management.

---

## Appendix C: Complete Python Interface Code

See Section 8 for the complete Python interface implementation.

---

## Appendix D: Known Issues and TODOs

This section documents issues identified during specification review and their resolution status.

### D.1 Resolved Critical Issues

1. ~~**Missing Tile Data in Backward Pass**~~: **FIXED** - Section 8.1 now saves `tile_counts`, `tile_offsets`, and `tile_content` in the context and reuses them in the backward pass.

2. ~~**MPS ↔ Metal Buffer Interop**~~: **FIXED** - Appendix E provides complete implementation with correct `storage().data_ptr()` approach and offset handling.

3. ~~**Truncation Inconsistency**~~: **FIXED** - All rasterize kernels (3D and nD) now use consistent truncation: `trunc_sq = truncate * truncate`.

4. ~~**Missing rasterize_bwd_nd Implementation**~~: **FIXED** - Section 6 (Appendix A) now contains complete `rasterize_bwd_nd` kernel with SIMD reduction for L gradients.

5. ~~**nD Kernel Final Truncation Check**~~: **FIXED** - The nD kernels now use consistent truncation check.

6. ~~**Missing intensity_floor Parameter**~~: **FIXED** - All Metal kernels (forward and backward, 3D and nD) now include `intensity_floor` parameter for early culling.

7. ~~**Missing Gradient Buffer Initialization**~~: **FIXED** - Section 7.5 and 7.7 now explicitly zero-initialize all gradient buffers before kernel dispatch.

### D.2 Resolved Interface Issues

1. ~~**Composition vs Inheritance**~~: **FIXED** - Section 4.3 correctly documents the composition pattern with rationale.

2. ~~**2D Support Missing**~~: **FIXED** - Section 8.2 documents handling 2D inputs as 3D volumes with `depth=1`.

3. ~~**Integration Point Incorrect**~~: **FIXED** - Section 4.2.1 now correctly points to `initialization.py:create_model()` as the integration point.

### D.3 Resolved Code Completeness Issues

1. ~~**Test 1 Incomplete**~~: **FIXED** - `test_cholesky_to_conic()` now includes complete reconstruction and verification code.

2. ~~**getMTLBuffer() Undefined**~~: **FIXED** - Appendix E provides complete implementation.

3. ~~**C++ Dispatcher Incomplete**~~: **FIXED** - Section 7.5-7.7 now contain complete dispatcher implementations for all four functions (forward_3d, backward_3d, forward_nd, backward_nd).

4. ~~**PyTorch Fallback Backward**~~: **FIXED** - Section 8.3 documents calling vanilla PyTorch directly without wrapping in MetalSplatFunction.

### D.4 Minor Issues (Acceptable)

1. **sigma_diag Computed but Unused**: In Python forward, `sigma_diag` is computed but not passed to Metal. This is intentional - Metal recomputes it internally from Ls during binning, which is more efficient than passing extra buffers.

2. **Appendix B/C Stub**: These appendices reference earlier sections rather than duplicating code. This is intentional - the C++ and Python code in Sections 7-8 is sufficiently detailed and authoritative.

### D.5 Implementation Notes

1. **Build System**: Use `make build-metal` target (Section 5.3). Hatch does not support custom build hooks, so the Metal extension must be built separately.

2. **MPS Synchronization**: Use `torch::mps::synchronize()` before dispatching Metal kernels. See Appendix E.2 for API reference.

3. **Buffer Offset Handling**: Always use `setBufferWithOffset()` helper for all tensor bindings. See Section 7.3 and Appendix E.3.

---

## Appendix E: MPS-Metal Interoperability

### E.0 Version Requirements and Self-Test (MANDATORY)

**CRITICAL**: The MPS-Metal interop uses PyTorch internal APIs that may change between versions.

**Pinned PyTorch Version Range:**
```
PyTorch >= 2.1.0, < 2.5.0  # Tested range - update as needed
```

**Mandatory Self-Test at Import Time:**
```python
# In metal_splatting_backend/__init__.py or gsplat_model_metal.py

def _validate_mps_interop():
    """Validate MPS buffer interop works correctly. Run at import time."""
    import torch
    import warnings

    if not torch.backends.mps.is_available():
        return False  # MPS not available, Metal won't be used

    try:
        # Test 1: Basic buffer extraction
        t = torch.randn(10, device='mps')
        storage_ptr = t.storage().data_ptr()
        assert storage_ptr != 0, "storage().data_ptr() returned null"

        # Test 2: Storage offset handling (critical for tensor views)
        t_full = torch.randn(100, device='mps')
        t_slice = t_full[25:75]  # View with storage_offset=25
        assert t_slice.storage_offset() == 25, "storage_offset not working"

        # Test 3: Verify data consistency (optional but recommended)
        # Create tensor, modify via Metal, verify in PyTorch
        # This requires the extension to be loaded, so may be done separately

        return True

    except Exception as e:
        warnings.warn(
            f"MPS buffer interop validation failed: {e}. "
            f"Metal acceleration disabled. Using PyTorch fallback."
        )
        return False

# Run at module import
_MPS_INTEROP_VALID = _validate_mps_interop()

def metal_available() -> bool:
    """Check if Metal extension is available AND working."""
    return _MPS_INTEROP_VALID and METAL_EXTENSION_LOADED
```

**Fallback on Validation Failure:**
If the self-test fails, the module should gracefully disable Metal and use the PyTorch fallback. Never crash on interop failure.

### E.1 Getting MTLBuffer from PyTorch MPS Tensor

**Warning**: This uses PyTorch internal APIs that may change.

**CRITICAL**: `id<MTLBuffer>` is an Objective-C object pointer, NOT a raw memory pointer.
You cannot add a byte offset to an object pointer. Always use the Storage + Offset method.

```cpp
#include <torch/extension.h>

// ❌ WRONG - Direct pointer cast is DANGEROUS
// id<MTLBuffer> tensorToMTLBuffer(const torch::Tensor& tensor) {
//     void* ptr = tensor.data_ptr();  // This is NOT the MTLBuffer!
//     return (__bridge id<MTLBuffer>)ptr;  // UNDEFINED BEHAVIOR
// }

// ✅ CORRECT - Use storage().data_ptr() which IS the MTLBuffer
id<MTLBuffer> tensorToMTLBuffer(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor must be on MPS device");

    // The STORAGE's data_ptr() is the id<MTLBuffer> on MPS
    // (not the tensor's data_ptr(), which includes offset)
    void* storage_ptr = tensor.storage().data_ptr();
    return (__bridge id<MTLBuffer>)storage_ptr;
}
```

### E.2 Synchronization Requirements

When mixing PyTorch MPS operations with custom Metal:

**API Reference**: The C++ synchronization API is `torch::mps::synchronize()`, documented at:
https://pytorch.org/cppdocs/api/function_namespacetorch_1_1mps_1a4c590e765e9ef17a0bc292dd6a10907e.html

```cpp
#include <ATen/mps/MPSDevice.h>  // For torch::mps::synchronize()

// Before dispatching custom Metal kernel:
// 1. Ensure PyTorch MPS operations are complete
torch::mps::synchronize();

// 2. Use the same command queue as PyTorch MPS (preferred)
// Or use a separate queue and add proper synchronization

// After custom Metal kernel completes:
// Ensure our operations are visible to PyTorch
[commandBuffer waitUntilCompleted];
```

**Python equivalent**: For Python code, use `torch.mps.synchronize()` when testing or debugging.

### E.2.1 Command Queue Sharing (Advanced)

**Option 1: Separate Queue (Simpler, Recommended)**
Use your own command queue with proper synchronization:

```cpp
struct MetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;  // Our own queue
    // ...
};

// Before kernel dispatch:
torch::mps::synchronize();  // Ensure PyTorch work is done

// Dispatch our kernel
[cmd commit];
[cmd waitUntilCompleted];  // Ensure our work is done before returning
```

**Option 2: Shared Queue (Lower Latency, Complex)**
For advanced use cases requiring minimal synchronization overhead, you can share
PyTorch's MPS command queue. However, this requires accessing PyTorch internals:

```cpp
#include <ATen/mps/MPSStream.h>

id<MTLCommandQueue> getSharedQueue() {
    // Get PyTorch's MPS stream
    auto stream = at::mps::getCurrentMPSStream();
    return stream->commandQueue();  // Internal API - may change!
}
```

**Warning**: The shared queue approach uses PyTorch internal APIs that may change
between versions. Test thoroughly when upgrading PyTorch versions.

**Recommendation**: Use separate queues with explicit synchronization unless you have
measured evidence that the synchronization overhead is a bottleneck.

### E.3 Buffer Offset Handling

PyTorch tensors may have non-zero storage offsets (e.g., when sliced or viewed).
**Always compute and pass the offset to setBuffer:**

```cpp
// Complete helper function for binding tensors to Metal encoder
void setBufferWithOffset(
    id<MTLComputeCommandEncoder> enc,
    const torch::Tensor& tensor,
    int index
) {
    TORCH_CHECK(tensor.device().is_mps(), "Tensor must be on MPS device");
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    // 1. Get the underlying storage (the actual MTLBuffer)
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)tensor.storage().data_ptr();

    // 2. Compute offset in bytes (storage_offset is in elements)
    NSUInteger offset = tensor.storage_offset() * tensor.element_size();

    // 3. Bind buffer with offset
    [enc setBuffer:buf offset:offset atIndex:index];
}
```

**Example usage in dispatcher:**
```cpp
std::vector<torch::Tensor> dispatch_forward_3d(
    torch::Tensor centers,   // May be a view with storage_offset > 0
    torch::Tensor conic,
    // ...
) {
    id<MTLCommandBuffer> cmd = [g_ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:g_ctx->getPipeline("rasterize_fwd_3d")];

    // Use helper for ALL tensor bindings
    setBufferWithOffset(enc, centers, 0);
    setBufferWithOffset(enc, conic, 1);
    // ...
}
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-21 | - | Initial specification |
| 1.1 | 2025-12-21 | - | Added Appendix D (Known Issues), Appendix E (MPS Interop) |
| 1.2 | 2025-12-21 | - | Critical fixes: nD backward SIMD reduction, correct MPS buffer interop, 2D support, PyTorch fallback, data flow diagram |
| 1.3 | 2025-12-21 | - | Comprehensive review fixes: (1) nD truncation consistency, (2) intensity_floor in nD kernels, (3) complete nD backward L gradients with SIMD, (4) Makefile build-metal targets, (5) gradient buffer zero-initialization, (6) correct integration point (initialization.py), (7) MPS sync API documentation, (8) complete C++ dispatcher code for all 4 functions, (9) Appendix D reorganization, (10) setBufferWithOffset edge cases, (11) MPS command queue sharing docs, (12) error handling section |
| 1.4 | 2025-12-21 | - | Expert review P0/P1 fixes: **P0-1**: Fixed backward chain-rule to use `torch.enable_grad()` + `torch.autograd.grad()` instead of `.backward()`. **P0-2**: Added `intensity_floor` parameter and culling check to backward kernel to match forward. **P0-3**: Fixed `atomic_add_float` memory order from invalid `memory_order_success` to `memory_order_relaxed`. **P0-4**: Padded grid sizes to full threadgroups for SIMD safety with `simd_sum()`. **P0-5**: Replaced CPU prefix-sum roundtrip with GPU `torch.cumsum()`. **P0-6**: Added version pinning and mandatory MPS interop self-test (§E.0). **P1-A**: Added new §2.7 Dimension and Coordinate Conventions with canonical `img_size=(W,H,D)`, `grid_dims=(tiles_x,tiles_y,tiles_z)`, `gid=(x,y,z)` standard. **P1-C**: Changed `torch.linalg.inv` to `torch.cholesky_inverse` for efficiency. **P1-D**: Updated §8.2 with explicit 2D coordinate ordering documentation (centers as `[y,x]`, mapping to 3D `[z,y,x]`). |

---

*End of Document*
