# luxar.gsplats.rendering - Technical Specification

**Version**: 1.0.1
**Last Updated**: 2026-05-02

## Purpose

`luxar.gsplats.rendering` renders `GSplatData` objects into dense NumPy volumes or Torch tensors for quality comparison, previews, and downstream metrics.

---

## Core Concepts

### Gaussian Splat Evaluation

Each splat contributes a truncated oriented Gaussian to voxels near its center:

```text
I(x) = amplitude * shifted_gaussian(mahalanobis_distance(x, center, L), truncate)
```

The Cholesky factor `L` represents covariance through `Σ = L @ L.T`. Rendering avoids explicit covariance inversion and delegates to the same lower-level rendering core used by fitting when possible.

### Coordinate System

Rendering samples output voxels at integer voxel coordinates. A center of
`(0, 0, 0)` is evaluated at the first voxel sample, and a shape `(Z, Y, X)` spans
valid sample coordinates `0..Z-1`, `0..Y-1`, and `0..X-1` in array order. The
public rendering helpers do not apply scene graph transforms; callers that need
world-space rendering must transform centers/Cholesky factors into the output
volume coordinate frame before calling `render_to_volume*()`.

### Backend Selection

Rendering supports explicit devices (`cuda`, `mps`, `cpu`) and automatic selection:

```text
CUDA extension available -> CUDA backend
else CUDA/MPS/CPU torch device -> PyTorch fallback
```

The CUDA backend is used only when compiled and compatible. Otherwise, PyTorch rendering is used on the selected device.

---

## Data Structures

### Input

```text
GSplatData:
  centers: float[N, D]
  cholesky_factors: float[N, D, D] or packed equivalent
  amplitudes: float[N]
  metadata: optional units/dimensions/LOD information
```

### Output

```text
render_to_volume: numpy.ndarray[shape]
render_to_volume_tensor: torch.Tensor[shape]
```

**Invariants**:
- `shape` length must match the spatial dimensionality being rendered.
- Output dtype is float32.
- Tensor output remains on the rendering device.

---

## Algorithms

### Render to Tensor

**Purpose**: Produce a dense tensor reconstruction from splats.

**Inputs**:
- `gsplat_data`: splats to render
- `shape`: output volume shape
- `device`: explicit device or `None` for auto
- `truncate`: truncation radius in standard deviations
- `chunk_size`: optional memory-control chunk size

**Algorithm**:
1. Resolve device.
2. Convert centers, amplitudes, and Cholesky factors to tensors.
3. Select CUDA fast path when available and requested.
4. Otherwise call PyTorch rendering core with chunking when needed.
5. Return a float32 tensor.

**Complexity**: proportional to the number of splats times the number of voxels inside each splat's truncated support.

### Splat Support and Accumulation

For each splat, the renderer computes an axis-aligned support box from the
truncation radius and the Cholesky-derived scale. Bounds are clamped to the
output shape. Voxels outside the support box are skipped. Contributions from
multiple splats are additive; no implicit normalization by total weight is
applied. Negative amplitudes are allowed by the math path, but fitting and
validation code may constrain amplitudes depending on the caller's config.

The PyTorch path groups splats by support-box shape so one local grid can be
reused per group. Local grid coordinates are shifted by each splat's lower bound,
subtracted from the splat center, evaluated through the Cholesky solve, and
accumulated into the flattened output tensor with row-major offsets.

### Memory Management

The PyTorch path chunks the local-grid point dimension (`P`) to avoid allocating
`K × D × P` intermediates for an entire support box at once. If `chunk_size` is
not supplied, the rendering core estimates a safe chunk size from available CUDA
or MPS memory and falls back to conservative CPU defaults. `clear_grid_cache()`
can be used by lower-level callers to release cached support grids after large
shape changes.

### Render to NumPy

**Purpose**: Convenience wrapper for CPU consumers.

**Algorithm**:
1. Call `render_to_volume_tensor()`.
2. Move result to CPU if needed.
3. Convert to NumPy float32.

---

## Validation Rules

- `shape` must contain positive integers.
- `truncate` must be positive.
- Device strings must resolve to a valid Torch device.
- CUDA backend failures must fall back only when correctness is preserved; unsupported hard failures should raise clear errors.

---

## Related Specifications

- `luxar.gsplats` - splat representation (`../SPECIFICATIONS.md`)
- `luxar.gsplats.models.gsplats` - optimization model and rendering core (`../models/gsplats/SPECIFICATIONS.md`)

---

## Changelog

- **v1.0.1** (2026-05-02): Documented voxel-coordinate convention, transform boundary, additive accumulation, and PyTorch memory-management behavior.
- **v1.0.0** (2026-05-02): Initial specification.
