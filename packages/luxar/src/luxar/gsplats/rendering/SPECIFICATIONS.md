# luxar.gsplats.rendering - Technical Specification

**Version**: 1.0.0
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

- **v1.0.0** (2026-05-02): Initial specification.
