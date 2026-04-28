# Seed Generation for Gaussian Splatting

**Version**: 2.0.0
**Last Updated**: 2026-02-28

## Overview

This subpackage provides seed Gaussian splat generation from n-dimensional images. Seeds serve as initial positions and shapes for Gaussian splat fitting, determining where splats should be placed to approximate image features. All seeding methods return `GSplatData` with scale-informed Gaussian shapes.

**Architecture: 4-Method Design**

1. **`generate_seeds()`** (unified entry point): Dispatches to one or more methods, combines results with deduplication
2. **`seed_from_edges()`**: Edge-based seeding via nD Sobel gradients with Poisson disk sampling
3. **`seed_from_grid()`**: Uniform grid seeding for spatial coverage
4. **`seed_from_decomposition()`**: Multi-scale decomposition peak detection for blob-like features

**Shared Utilities:**
- L-infinity neighborhood peak detection
- Greedy spatial deduplication with `SpatialHashGrid` (O(1) amortized proximity queries)
- Isotropic sigma to Cholesky factor conversion
- Seed combination and merging

**GPU Acceleration:**
- All methods accept `device` parameter for PyTorch GPU acceleration
- Substantial speedup for large volumes (>100^3) on CUDA/MPS devices — often orders of magnitude depending on GPU and problem size

## Package Structure

```
seeds/
├── __init__.py                    # Public API exports
├── generate.py                    # Unified entry point (generate_seeds)
├── edges.py                       # Edge-based seeding (seed_from_edges)
├── grid.py                        # Grid seeding (seed_from_grid)
├── multiscale_decomposition.py    # Decomposition-based seeding (seed_from_decomposition)
├── gpu_ops.py                     # GPU-accelerated operations (PyTorch)
├── utils.py                       # Shared utilities (peak detection, deduplication, Cholesky)
├── tests/                         # Unit tests
├── README.md                      # User-facing documentation
└── SPECIFICATIONS.md              # This file (technical specification)
```

## Public API Exports

From `__init__.py`:

```python
# Primary public API
generate_seeds          # Unified entry point (recommended)

# Individual seeding methods
seed_from_decomposition
seed_from_grid
seed_from_edges

# Utility functions (for advanced usage)
sigmas_to_cholesky_isotropic
local_maxima
dedupe_farthest_first
combine_seeds
```

---

## 1. Unified Entry Point: `generate_seeds()`

**Module**: `generate.py`

### Signature

```python
def generate_seeds(
    V: np.ndarray,
    method: str = "auto",
    **kwargs,
) -> GSplatData:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `V` | `np.ndarray` | (required) | Input n-dimensional image/volume |
| `method` | `str` | `"auto"` | Seeding method(s) to use |

**Method options:**
- `"auto"`: Fast edges + grid combination (default, recommended)
- `"edges"`: Edge-based boundary detection with Sobel gradients
- `"grid"`: Uniform grid for spatial coverage
- `"decomposition"`: Multi-scale decomposition for blob-like features (slow)
- Comma-separated: e.g. `"decomposition,edges,grid"` to include all

**Common kwargs (apply to multiple methods):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_distance` | `float` | `2.0` | Minimum Euclidean distance between seeds (voxels) |
| `target_seeds` | `int` or `None` | `None` | Target seed count for "auto" mode |
| `device` | `str` or `None` | `None` | PyTorch device (`None`, `'cpu'`, `'cuda'`, `'mps'`, `'auto'`) |

**Decomposition kwargs** (method="decomposition"):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scales` | `list[int]` | `[1,2,4,8,16,32,64]` | Scale factors for decomposition |
| `ignore_finest_k` | `int` | `1` | Number of finest scales to ignore |
| `threshold_rel` | `float` | `0.1` | Relative threshold for peak detection (0.0-1.0) |
| `peaks_per_scale` | `int` or `None` | `None` | Maximum peaks per scale |
| `decompose_kwargs` | `dict` or `None` | `None` | Additional kwargs for `decompose_image()` |
| `verbose` | `bool` | `False` | Print progress |

**Grid kwargs** (method="grid"):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spacing` | `float` or `Sequence[float]` or `None` | `None` | Grid spacing (None = aspect-ratio-aware auto) |
| `jitter` | `float` | `0.0` | Jitter fraction (0.0-0.5) |
| `sigma` | `float` or `None` | `None` | Gaussian sigma (None = spacing/2) |
| `exclude_below` | `float` or `None` | `None` | Absolute intensity threshold |
| `exclude_below_percentile` | `float` or `None` | `None` | Percentile intensity threshold (0-100) |

**Edge kwargs** (method="edges"):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_seeds` | `int` or `None` | `None` | Target number of edge seeds |
| `edge_threshold_rel` | `float` | `0.1` | Relative edge threshold (0.0-1.0) |

### Returns

`GSplatData` with:
- `centers`: Peak/centroid positions (N, ndim)
- `amplitudes`: Peak intensities (N,), scaled by `SEED_AMPLITUDE_SCALE` (0.9)
- `cholesky_factors`: Scale-informed packed Cholesky factors (N, tril_size)

### Algorithm: Auto Mode

When `method="auto"`, `generate_seeds()` uses `_auto_combine()`:

1. **Budget allocation**: 60% edges, 40% grid (decomposition excluded for speed)
2. **Target estimation**: If `target_seeds` is None, heuristic based on volume size:
   - `target = max(100, int(total_voxels^(1/ndim) / 2))`, capped at 10,000
3. **Phase 1 - Edge seeds**: Calls `seed_from_edges(V, n_seeds=budget_edges, ...)`
4. **Phase 2 - Grid seeds**: Auto-computes spacing from budget, calls `seed_from_grid(V, spacing=..., ...)`
5. **Deduplication**: Combines results via `_combine_gsplatdata()` with `dedupe_farthest_first()`

### Algorithm: Multi-Method Combination

When comma-separated methods are specified (e.g. `"decomposition,grid"`):

1. **Parameter routing**: Each kwarg is routed to applicable methods based on parameter sets
2. **Sequential generation**: Each method runs independently
3. **Combination**: Results concatenated and deduplicated via `_combine_gsplatdata()`

### Internal Functions

**`_empty_gsplatdata(ndim: int) -> GSplatData`**
- Creates empty GSplatData with correct array shapes

**`_auto_combine(V, target_seeds, min_distance, ...) -> GSplatData`**
- Implements the auto mode budget allocation strategy

**`_combine_gsplatdata(results, min_distance, device, ndim) -> GSplatData`**
- Concatenates arrays from multiple GSplatData objects
- Deduplicates using `dedupe_farthest_first()` sorted by amplitude (highest priority)
- Returns deduplicated GSplatData with O(1) index-based attribute lookup

---

## 2. Edge-Based Seeding: `seed_from_edges()`

**Module**: `edges.py`

### Signature

```python
def seed_from_edges(
    V: np.ndarray,
    n_seeds: Optional[int] = None,
    min_distance: float = 2.0,
    edge_threshold_rel: float = 0.1,
    device: Optional[str] = None,
) -> GSplatData:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `V` | `np.ndarray` | (required) | Input n-dimensional image/volume |
| `n_seeds` | `int` or `None` | `None` | Target number of seeds (None = auto-estimate) |
| `min_distance` | `float` | `2.0` | Minimum distance between seeds (voxels) |
| `edge_threshold_rel` | `float` | `0.1` | Relative edge threshold (0.0-1.0), fraction of max edge response |
| `device` | `str` or `None` | `None` | PyTorch device for GPU acceleration |

### Algorithm

1. **Sobel gradient magnitude**: Compute nD Sobel gradients via `_compute_nd_sobel_magnitude()`.
   - CPU path: `scipy.ndimage.sobel()` applied along each axis, summed as RSS
   - GPU path: Separable convolution with differentiation kernel `[-1, 0, 1]` and smoothing kernel `[1, 2, 1]`
2. **Normalize**: Scale edge response to [0, 1]
3. **Threshold**: Apply `edge_threshold_rel` mask
4. **Poisson disk sampling**: `_poisson_disk_sample_weighted()` selects `n_seeds` points weighted by edge response, enforcing `min_distance` via `SpatialHashGrid`
5. **Isotropic initialization**: All seeds get sigma=1.0 via `sigmas_to_cholesky_isotropic()`
6. **Amplitude sampling**: Interpolate from V at seed positions via `_sample_amplitudes()`, scaled by `SEED_AMPLITUDE_SCALE` (0.9)

### Auto-estimation of n_seeds

When `n_seeds` is None:
```
n_seeds = max(50, int(total_voxels^(1/ndim) / 4))
n_seeds = min(n_seeds, 5000)
```

### Poisson Disk Sampling: `_poisson_disk_sample_weighted()`

```python
def _poisson_disk_sample_weighted(
    density: np.ndarray,   # Sampling density
    mask: np.ndarray,      # Valid locations
    n_samples: int,        # Target count
    min_distance: float,   # Minimum spacing
) -> np.ndarray:
```

Algorithm:
1. Extract valid coordinates and density values from mask
2. Normalize density to probability distribution
3. Oversample: draw `min(len(valid), n_samples * 10)` candidates weighted by density
4. Greedy selection with `SpatialHashGrid` distance checks:
   - For each candidate, query spatial hash grid for nearest selected seed
   - Accept if distance >= `min_distance`
   - O(1) amortized proximity queries
   - For last 3 slots, use simple distance check (avoids overhead)
5. Uses fixed random seed (42) for reproducibility

**Complexity**: O(N_candidates) amortized with `SpatialHashGrid` queries

### Internal Functions

**`_compute_nd_sobel_magnitude(V, device) -> np.ndarray`**
- nD Sobel gradient magnitude computation
- Dispatches to GPU when device is not None/cpu and volume is large enough

**`_sample_amplitudes(V, coords, device) -> np.ndarray`**
- Interpolates amplitude values from V at given coordinates
- CPU: `scipy.ndimage.map_coordinates` with order=1, mode='nearest'
- GPU: `torch.nn.functional.grid_sample` (2D/3D only, falls back to CPU for other dims)

---

## 3. Grid Seeding: `seed_from_grid()`

**Module**: `grid.py`

### Signature

```python
def seed_from_grid(
    V: np.ndarray,
    spacing: Optional[Union[float, Sequence[float]]] = None,
    jitter: float = 0.0,
    sigma: Optional[float] = None,
    exclude_below: Optional[float] = None,
    exclude_below_percentile: Optional[float] = None,
    device: Optional[str] = None,
) -> GSplatData:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `V` | `np.ndarray` | (required) | Input n-dimensional image/volume |
| `spacing` | `float`, `Sequence[float]`, or `None` | `None` | Grid spacing (None = aspect-ratio-aware auto) |
| `jitter` | `float` | `0.0` | Jitter fraction (0.0-0.5) |
| `sigma` | `float` or `None` | `None` | Gaussian sigma (None = mean(spacing)/2) |
| `exclude_below` | `float` or `None` | `None` | Absolute intensity threshold |
| `exclude_below_percentile` | `float` or `None` | `None` | Percentile threshold (0-100) |
| `device` | `str` or `None` | `None` | PyTorch device for GPU acceleration |

`exclude_below` and `exclude_below_percentile` are mutually exclusive.

### Algorithm

1. **Compute spacing** (if None):
   - Aspect-ratio-aware: spacing proportional to each dimension's size
   - `s_old = max(2.0, min_dim * 0.05)` (old isotropic baseline)
   - `k = geometric_mean(shape) / s_old`
   - `spacing[i] = shape[i] / k`, clamped to minimum 2.0
   - Example: 1000x1000x10 image produces [136, 136, 1.4] spacing (not [29, 29, 29])
2. **Compute sigma** (if None): `sigma = mean(spacing) / 2`
3. **Generate grid**: Create meshgrid from `[spacing/2, shape - spacing/2]` per dimension
4. **Apply jitter**: Random offset in `[-jitter*spacing, +jitter*spacing]` (seed=42), clipped to bounds
5. **Sample amplitudes**: Via `_sample_amplitudes()` from `edges.py`, scaled by `SEED_AMPLITUDE_SCALE` (0.9)
6. **Intensity filtering**: Apply `exclude_below` or `exclude_below_percentile` mask
7. **Cholesky factors**: `sigmas_to_cholesky_isotropic(sigma, ndim)` for all seeds

---

## 4. Decomposition-Based Seeding: `seed_from_decomposition()`

**Module**: `multiscale_decomposition.py`

### Signature

```python
def seed_from_decomposition(
    V: np.ndarray,
    scales: Optional[List[int]] = None,
    ignore_finest_k: int = 1,
    peaks_per_scale: Optional[int] = None,
    min_distance: float = 2.0,
    threshold_rel: float = 0.1,
    decompose_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    device: Optional[str] = None,
) -> GSplatData:
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `V` | `np.ndarray` | (required) | Input n-dimensional image/volume |
| `scales` | `List[int]` or `None` | `[1,2,4,8,16,32,64]` | Scale factors for decomposition |
| `ignore_finest_k` | `int` | `1` | Finest scales to skip (noise suppression) |
| `peaks_per_scale` | `int` or `None` | `None` | Max peaks per scale |
| `min_distance` | `float` | `2.0` | Minimum distance between seeds (voxels) |
| `threshold_rel` | `float` | `0.1` | Relative threshold for peak detection (0.0-1.0) |
| `decompose_kwargs` | `dict` or `None` | `None` | Extra kwargs for `decompose_image()` |
| `verbose` | `bool` | `False` | Print progress |
| `device` | `str` or `None` | `None` | PyTorch device for GPU acceleration |

### Algorithm

1. **Decompose image**: Call `decompose_image(V, scales=scales, **decompose_kwargs)` to produce scale images
2. **Process scales** (coarse to fine, skipping finest `ignore_finest_k`):
   - For each scale: compute threshold as `threshold_rel * max_intensity`
   - Find local maxima using `local_maxima(scale_img, radius=1, thresh=threshold, top_k=peaks_per_scale)`
   - Map peak coordinates to full resolution: `position = peak * scale_factor + scale_factor / 2`
   - Record scale factor and energy (intensity) for each peak
3. **Combine all seeds**: Concatenate across scales
4. **Deduplicate**: `_dedupe_with_scales_and_energies()` - energy-based priority, greedy spatial deduplication
5. **Amplitudes**: Sample from original image V at (rounded) seed positions, scaled by `SEED_AMPLITUDE_SCALE`
6. **Cholesky factors**: `sigmas_to_cholesky_isotropic(scale_factors, ndim)` - sigma equals detection scale

### Internal Functions

**`_dedupe_with_scales_and_energies(coords, scales, energies, min_distance)`**
- Sort by energy (highest first)
- Greedy O(N^2) deduplication: for each seed, mark all seeds within `min_distance` as rejected
- Returns (coords, scales, energies) tuple after filtering

---

## 5. GPU Acceleration

**Module**: `gpu_ops.py`

All operations use pure PyTorch (no kornia/faiss dependencies).

### Device Resolution: `_get_device(device)`

| Input | Output | Notes |
|-------|--------|-------|
| `None` | `'cpu'` | Backward-compatible default |
| `'auto'` | Best available | cuda > mps > cpu |
| `'cpu'` | `'cpu'` | Force CPU |
| `'cuda'` | `'cuda'` or `'cpu'` | Fallback with warning if unavailable |
| `'mps'` | `'mps'` or `'cpu'` | Fallback with warning if unavailable |
| `'cuda:N'` | `'cuda:N'` | Specific GPU device |

### Size Threshold: `should_use_gpu(V, device)`

- Returns `False` for `device='cpu'`
- Returns `False` for volumes smaller than 50^3 voxels (GPU overhead not worth it)
- Returns `True` otherwise

### GPU Operations

**`_compute_nd_sobel_magnitude_gpu(V_tensor) -> torch.Tensor`**
- Separable nD Sobel gradient computation
- Differentiation kernel: `[-1, 0, 1]` along target axis
- Smoothing kernel: `[1, 2, 1]` (unnormalized) along all perpendicular axes
- Matches `scipy.ndimage.sobel` output
- Works for arbitrary dimensions via `_conv1d_along_axis()`

**`local_maxima_gpu(img, radius, thresh, top_k) -> np.ndarray`**
- Peak detection via `F.max_pool2d` / `F.max_pool3d`
- **Supports 2D and 3D only** (raises `NotImplementedError` for other dims)
- Replicate padding to match scipy 'nearest' mode
- Returns integer coordinates (N, ndim)

**`soft_blur_nd_gpu(img) -> torch.Tensor`**
- Separable blur with kernel `[0.25, 0.5, 0.25]`
- Applied along each axis via `_conv1d_along_axis()`
- Works for arbitrary dimensions

**`sample_amplitudes_gpu(V, coords, mode) -> torch.Tensor`**
- Volume interpolation via `F.grid_sample`
- **Supports 2D and 3D only** (raises `NotImplementedError` for other dims)
- Coordinate system: reverses axis order (scipy row,col -> grid_sample x,y)
- Normalization: voxel coords [0, shape-1] mapped to [-1, 1]
- Padding mode: 'border' (matches scipy 'nearest')

**`_conv1d_along_axis(x, kernel, axis, padding) -> torch.Tensor`**
- Applies 1D convolution along a specific axis of nD tensor
- Strategy: permute target axis to last, reshape for conv1d, apply, reshape back
- Supports 'same' padding via replicate mode

### Memory Estimation

**`estimate_gpu_memory_needed(V, operation) -> int`**
- Rough memory estimates: Sobel 5x, blur 3x, maxpool 3x, interpolation 2x of input size

**`check_gpu_memory(V, device, operation) -> bool`**
- Conservative check using 80% of GPU total memory as threshold
- CUDA only (MPS doesn't expose memory info, assumed OK)

---

## 6. Shared Utilities

**Module**: `utils.py`

### Constants

**`SEED_AMPLITUDE_SCALE = 0.9`**
- Multiplier for seed amplitude initialization
- Starting at 90% avoids initial over-prediction when splats overlap
- Prevents triggering asymmetric over-prediction penalty and divergence

### `sigmas_to_cholesky_isotropic(sigmas, ndim) -> np.ndarray`

Convert per-seed isotropic sigmas to packed Cholesky factors.

**Parameters:**
- `sigmas`: (N,) array of isotropic sigma values
- `ndim`: number of spatial dimensions

**Returns:** (N, ndim*(ndim+1)//2) packed lower-triangular Cholesky factors

**Packed format:** Row-major lower triangular: `[L00, L10, L11, L20, L21, L22, ...]`

For isotropic: only diagonal positions are non-zero. Diagonal index for dimension k: `k*(k+3)//2`

Example (3D, sigma=s): `[s, 0, s, 0, 0, s]` representing `L = diag(s, s, s)`

### `local_maxima(img, radius, thresh, top_k) -> np.ndarray`

Find local maxima in n-dimensional image using L-infinity (Chebyshev) neighborhood.

**Parameters:**
- `img`: n-dimensional image array
- `radius`: half-width of hypercube neighborhood (minimum 1)
- `thresh`: minimum intensity threshold
- `top_k`: maximum number of strongest peaks (None = all)

**Algorithm:**
1. Apply `scipy.ndimage.maximum_filter` with `size=[2*radius+1]*ndim`
2. Peak criteria: `(img == max_filtered) & (img >= thresh)`
3. Extract coordinates via `np.argwhere`
4. Optionally select top_k by intensity (argsort, take last k)

**Returns:** (N, ndim) integer coordinates

**Note:** Uses `size` parameter instead of `footprint` for memory efficiency. For 3D radius=5, `size=[11,11,11]` vs. footprint with 11^3=1331 boolean elements.

### `soft_blur_nd(img) -> np.ndarray`

Separable blur with tent kernel `[0.25, 0.5, 0.25]` applied via `scipy.ndimage.convolve1d` along each axis. Complexity: O(3*ndim*N) vs O(3^ndim*N) for direct convolution.

### `count_local_maxima(img, radius, threshold_rel, blur) -> int`

Count local maxima with optional soft blur preprocessing. Returns integer count.

### `dedupe_farthest_first(coords, min_distance, intensities, device) -> tuple[np.ndarray, np.ndarray]`

Greedy spatial deduplication with `SpatialHashGrid` acceleration (O(1) amortized proximity queries).

**Parameters:**
- `coords`: (N, ndim) seed coordinates
- `min_distance`: minimum Euclidean distance between kept seeds
- `intensities`: optional (N,) quality values for priority sorting
- `device`: ignored (always uses CPU `SpatialHashGrid`)

**Returns:** `(deduped_coords, kept_indices)` tuple
- `deduped_coords`: (M, ndim) deduplicated coordinates
- `kept_indices`: (M,) indices into original coords array for O(1) attribute lookup

**Algorithm:**
1. For small inputs (<50 seeds): `_dedupe_simple()` with O(N^2) greedy
2. For larger inputs:
   - Sort by intensity (highest first) if provided
   - Pre-allocate output arrays
   - Start with strongest seed
   - For each candidate: query `SpatialHashGrid` for nearest selected, keep if >= min_distance
   - O(1) amortized proximity queries via spatial hash grid
3. Map back to original indices if sorted

**Complexity:** O(N) amortized where N is input count, using `SpatialHashGrid`

**Note:** GPU deduplication was removed because CPU with `SpatialHashGrid` is faster for typical seed counts (<100K). CPU completes 16K seeds in ~0.77s.

### `_dedupe_simple(coords, min_distance, intensities) -> tuple[np.ndarray, np.ndarray]`

O(N^2) greedy deduplication for <50 seeds. Uses boolean mask to mark nearby seeds as used.

### `combine_seeds(*candidate_arrays, min_distance, method) -> np.ndarray`

Combine seed coordinate arrays from multiple methods.

**Parameters:**
- `*candidate_arrays`: variable number of (N_i, ndim) arrays
- `min_distance`: optional deduplication distance (None = no dedup)
- `method`: `"union"` only (concatenation)

**Returns:** (M, ndim) combined coordinates

---

## 7. Return Type: GSplatData

All seeding methods return `GSplatData` (from `luxar.gsplats.gsplat_data`):

| Field | Shape | Dtype | Description |
|-------|-------|-------|-------------|
| `centers` | (N, ndim) | float32 | Seed positions |
| `amplitudes` | (N,) | float32 | Intensity values (scaled by 0.9) |
| `cholesky_factors` | (N, tril_size) | float32 | Packed lower-triangular Cholesky factors |

Where `tril_size = ndim * (ndim + 1) // 2`.

---

## 8. Usage Examples

```python
from luxar.gsplats.seeds import generate_seeds
from luxar.gsplats import fit_gaussian_splats
import numpy as np

# Auto mode (recommended) - fast edges + grid combination
seeds = generate_seeds(image)

# Single method
seeds = generate_seeds(image, method="decomposition")
seeds = generate_seeds(image, method="grid", spacing=10.0)
seeds = generate_seeds(image, method="edges", n_seeds=1000)

# All methods combined
seeds = generate_seeds(image, method="decomposition,edges,grid")

# GPU acceleration
seeds = generate_seeds(image, method="auto", device="cuda")
seeds = generate_seeds(image, method="edges", device="auto")

# Use with fitter
seeds = generate_seeds(image, method="auto")
result = fit_gaussian_splats(image, seeds=seeds)
```

---

## Changelog

### 2.0.0 (2026-02-28)
- **Complete rewrite** to match actual implementation
- Documented 4-method architecture: generate_seeds, edges, grid, decomposition
- Removed references to non-existent files: `multiscale_gaussian.py`, `moment_seeding.py`
- Removed references to non-existent function: `seed_from_gaussian()`
- Added full documentation for GPU acceleration (`gpu_ops.py`)
- Added full documentation for shared utilities (`utils.py`)
- Documented parameter routing in `generate_seeds()`
- Documented Poisson disk sampling algorithm in edges.py
- Documented aspect-ratio-aware grid spacing in grid.py

### 1.3.0 (2026-01-11)
- Previous version (severely outdated, documented non-existent code)
