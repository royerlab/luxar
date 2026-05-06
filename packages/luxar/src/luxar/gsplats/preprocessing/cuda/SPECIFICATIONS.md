# luxar.gsplats.preprocessing.cuda - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-02

## Purpose

`luxar.gsplats.preprocessing.cuda` provides an optional compiled CUDA backend for Non-Local Means denoising. It accelerates preprocessing while preserving the public preprocessing API's fallback behavior.

---

## Core Concepts

### Optional Extension

The CUDA extension is optional. Importing the package must not fail when the extension is absent. Instead, `NLM_CUDA_AVAILABLE` reports availability and callers fall back to PyTorch or scikit-image backends.

### Supported Kernel Parameters

The CUDA backend supports a bounded set of patch and search sizes selected to fit shared-memory constraints:

```text
patch_size in {3, 5}
search_distance in {5, 7, 9, 11, 13, 15}
```

Unsupported combinations raise `ValueError` with a fallback suggestion.

---

## Data Structures

### Kernel Input

```text
volume: torch.Tensor[H, W] or torch.Tensor[D, H, W], float32, CUDA device
h: float
patch_size: int
search_distance: int
```

### Kernel Output

```text
denoised: torch.Tensor with same shape, dtype, and device as input
```

**Invariants**:
- Input must be 2D or 3D.
- Input tensor must be contiguous or converted before kernel launch.
- Output shape equals input shape.

---

## Algorithms

### CUDA NLM Wrapper

**Purpose**: Validate parameters and dispatch to the compiled CUDA kernel.

**Algorithm**:
1. Verify extension availability.
2. Validate tensor dtype, dimensionality, and device.
3. Convert `patch_size` to patch radius.
4. Validate `search_distance` against compiled kernel support.
5. Check shared-memory feasibility for large 3D search windows.
6. Launch 2D or 3D CUDA kernel.
7. Return denoised tensor.

**Complexity**: O(N * search_window * patch_window) where N is voxel count.

### Build Script

**Purpose**: Compile CUDA/C++ extension reproducibly.

**Algorithm**:
1. Detect CUDA toolkit and PyTorch extension build configuration.
2. Select supported GPU architectures.
3. Compile kernels and bindings with optimization flags.
4. Copy extension artifact into the package directory.
5. Write build metadata for diagnostics.

---

## Validation Rules

- Extension import failures must not crash package import.
- Unsupported parameters must raise clear `ValueError` messages.
- Shared-memory limits must be checked before kernel launch.
- CPU tensors must not be passed to CUDA kernels.

---

## Related Specifications

- `luxar.gsplats.preprocessing` - high-level denoising pipeline (`../SPECIFICATIONS.md`)
- `luxar.gsplats.models.gsplats.cuda` - compiled CUDA backend conventions (`../../models/gsplats/cuda/SPECIFICATIONS.md`)

---

## Changelog

- **v1.0.0** (2026-05-02): Initial specification.
