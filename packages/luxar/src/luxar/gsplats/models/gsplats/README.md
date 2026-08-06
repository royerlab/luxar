# Gaussian Splat Models and Rendering

## Purpose

Core Gaussian splatting model and rendering engine for nD oriented Gaussian splats. This package provides the reference PyTorch implementation used for fitting and volume reconstruction, along with optimized rendering routines.

## Key Classes

### `GaussianSplatModel` (`gsplat_model.py`)

PyTorch `nn.Module` for optimizing collections of oriented Gaussian functions. Each splat is parameterized by:

- **Center position** -- sigmoid-constrained to stay within the image domain
- **Covariance matrix** -- Cholesky decomposition `L` where Sigma = L @ L^T, with configurable min/max diagonal and eccentricity constraints
- **Amplitude** -- non-negative via softplus activation, with optional max clamping

Key methods:

| Method | Description |
|--------|-------------|
| `forward()` | Render all splats to a volume using AABB truncation |
| `current_params()` | Extract (centers, L, amps) from raw learnable parameters |
| `prune_(mask)` | Remove splats by boolean mask |
| `append_(centers, Ls, amps)` | Add new splats |
| `replace_with(centers, Ls, amps)` | Replace all splats |
| `n_splats()` | Return current splat count |

### Rendering Functions (`rendering_core.py`)

Low-level rendering engine with specialized fast paths:

| Function | Description |
|----------|-------------|
| `render_gaussians(shape, centers, Ls, amps, ...)` | Main rendering entry point (dispatches to 2D/3D fast paths or generic nD) |
| `fwd_norm2_2d(L, d0, d1)` | 2D forward substitution (specialized) |
| `fwd_norm2_3d(L, d0, d1, d2)` | 3D forward substitution (specialized) |
| `compute_aabb_with_intensity_floor(...)` | AABB computation with intensity-based truncation |
| `group_by_box(aabb_lo, aabb_hi, shape)` | CPU spatial grouping for chunk processing |
| `group_by_box_gpu(aabb_lo, aabb_hi, shape)` | GPU spatial grouping |
| `cached_base_and_offsets(lo, hi, device)` | Cached coordinate grid generation |
| `calculate_optimal_chunk_size(K, d, device, dtype)` | Memory-aware chunk size selection |
| `clear_grid_cache()` | Clear byte-budgeted coordinate grid caches |
| `get_grid_cache_stats()` | Inspect per-device grid-cache occupancy and budgets |
| `linear_strides(shape)` | Compute linear memory strides |

### Rendering Wrappers (`rendering_wrappers.py`)

User-friendly wrappers that accept `GSplatData` objects:

| Function | Description |
|----------|-------------|
| `render_gaussians_numpy(shape, result, ...)` | CPU NumPy output (no gradients) |
| `render_gaussians_pytorch(shape, result, ...)` | PyTorch tensor output (supports gradients) |

## Usage

```python
import numpy as np
from luxar.gsplats.models.gsplats import GaussianSplatModel, render_gaussians

# Create model
model = GaussianSplatModel(
    shape=(128, 128, 128),
    centers0=centers,       # (N, 3) array
    L0=L,                   # (N, 3, 3) lower-triangular
    amps0=amps,             # (N,) array
    sigma_min_diag=(0.5, 0.5, 0.5),
    truncate=3.0,
)

# Forward pass (renders volume)
output = model()

# Direct rendering without model
volume = render_gaussians(
    shape=(128, 128, 128),
    centers=centers_tensor,
    Ls=L_tensor,
    amps=amps_tensor,
    truncate=3.0,
)
```

## Grid Cache Tuning

The PyTorch renderer caches local support grids by shape, stride, dtype, and
device so repeated AABB shapes can reuse coordinate tensors. The cache is
byte-budgeted per device instead of entry-count-limited:

- CUDA default: `min(10% of currently free VRAM, 2 GiB)`
- MPS default: `512 MiB`
- CPU default: `min(5% of available RAM, 4 GiB)` when available, otherwise `512 MiB`
- Default per-entry cap: 75% of the total budget

HPC or large-volume jobs can tune or disable the cache explicitly:

```bash
# Disable support-grid caching entirely
export LUXAR_GSPLAT_GRID_CACHE_MAX_BYTES=0

# Allow up to 8 GiB per device and up to 6 GiB for one repeated support grid
export LUXAR_GSPLAT_GRID_CACHE_MAX_GB=8
export LUXAR_GSPLAT_GRID_CACHE_MAX_ENTRY_GB=6
```

Large entries that exceed the active cap are still used for the current render,
but are not retained as long-lived CPU/GPU tensors. Use `clear_grid_cache()`
between unrelated large jobs, and `get_grid_cache_stats()` to inspect current
occupancy.

## Accelerated Backends

For GPU-accelerated rendering, see the backend-specific subpackages:

- **CUDA** (`cuda/`): NVIDIA GPU acceleration (2D-8D), substantial speedup (often orders of magnitude, GPU-dependent)
- **Metal** (`metal/`): Apple Silicon acceleration for 3D MPS volumes using splat-centric Metal kernels. The unconstrained 3D training path consumes raw model parameters directly in Metal and returns raw gradients; constrained 3D paths use post-activation Cholesky tensors, and other supported MPS dimensions use PyTorch rendering. Speedup is chip- and workload-dependent.

Both backends provide model classes (`GaussianSplatModelCUDA`, `GaussianSplatModelMetal`) with matching parameter-management APIs. CUDA provides custom kernels for 2D-8D; Metal accepts 2D-8D MPS models but only dispatches to custom splat-centric Metal kernels for 3D float32 tensors.

## File Structure

```
gsplats/
├── __init__.py              # Package exports
├── gsplat_model.py          # GaussianSplatModel (PyTorch nn.Module)
├── rendering_core.py        # Core rendering engine (2D/3D fast paths, nD generic)
├── rendering_wrappers.py    # NumPy/PyTorch/batched wrappers
├── README.md                # This file
├── cuda/                    # CUDA backend (NVIDIA GPUs)
├── metal/                   # Metal backend (Apple Silicon)
└── tests/
    ├── __init__.py
    ├── test_gsplat_model.py # Model unit tests
    └── test_rendering.py    # Rendering correctness tests
```
