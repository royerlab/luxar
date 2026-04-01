# CUDA Backend Specification - PyTorch Integration & Implementation

**Version**: 0.1.0
**Status**: Implementation Complete
**Last Updated**: 2026-03-31

> **Note**: This is Part 2 of the CUDA Backend Specification (PyTorch Integration & Implementation).
> See also:
> - [Part 1: Core Algorithms](SPECIFICATIONS.md) - Architecture, kernel design, memory optimization
> - [Part 3: Testing Strategy](SPECIFICATIONS_TESTING.md) - Comprehensive testing guide

## Table of Contents

8. [PyTorch Integration](#8-pytorch-integration)
   - 8.1 [Extension Structure](#81-extension-structure)
   - 8.2 [Backend Protocol (for Testability)](#82-backend-protocol-for-testability)
   - 8.3 [Python Model Class](#83-python-model-class)
   - 8.4 [Multi-Stream Support (Phase 5)](#84-multi-stream-support-phase-5)
   - 8.5 [Custom Autograd Function](#85-custom-autograd-function)
   - 8.6 [Modern torch.library API (Recommended for PyTorch 2.0+)](#86-modern-torchlibrary-api-recommended-for-pytorch-20)
9. [Performance Targets](#9-performance-targets)
   - 9.1 [Benchmark Scenarios](#91-benchmark-scenarios)
   - 9.2 [Memory Footprint Targets](#92-memory-footprint-targets)
   - 9.3 [Latency Breakdown (Target)](#93-latency-breakdown-target)
   - 9.4 [Occupancy Analysis](#94-occupancy-analysis)
   - 9.5 [Memory Bandwidth Analysis](#95-memory-bandwidth-analysis)
   - 9.6 [Profiling Methodology](#96-profiling-methodology)
10. [Implementation Phases](#10-implementation-phases)
    - 10.6 [Compilation Strategy (REQUIRED)](#106-compilation-strategy-required)
    - 10.7 [Memory Allocation Strategy (REQUIRED)](#107-memory-allocation-strategy-required)
12. [References](#12-references)

**Appendices**:
- [Appendix A: CUDA Error Handling](#appendix-a-cuda-error-handling)
- [Appendix B: Coordinate Conventions](#appendix-b-coordinate-conventions)
- [Appendix C: Error Codes](#appendix-c-error-codes)

---

## 8. PyTorch Integration

### 8.1 Extension Structure

```
cuda/
├── src/
│   ├── cuda_splatting.cu      # Dispatch layer: forward/backward entry points,
│   │                          #   template instantiations, input validation
│   ├── cuda_splatting.h       # Public API: forward(), backward(), BinningState,
│   │                          #   forward_fp16(), backward_fp16()
│   ├── bindings.cpp           # pybind11 bindings: forward_wrapper(), backward_wrapper()
│   ├── kernels_core.cuh       # Core kernels: preprocess, bin, rasterize fwd/bwd (tile-based),
│   │                          #   rasterize_forward_splat_centric_kernel,
│   │                          #   rasterize_backward_splat_centric_kernel
│   ├── kernels_global.cuh     # Global splat kernels (pixel-parallel, for legacy path)
│   ├── kernel_launchers.cuh   # Launch wrappers for all kernels (grid/block config)
│   ├── utils.cuh              # Umbrella header (includes all sub-headers below)
│   ├── math_utils.cuh         # Mahalanobis distance, Gaussian intensity, shift params
│   ├── tile_utils.cuh         # AABB struct, tile indexing, grid optimization
│   ├── reduction_utils.cuh    # Warp reduction, gradient helpers, 2D/3D backward specializations
│   └── dtype_traits.cuh       # DTypeTraits for FP16/FP32 load abstraction
├── gsplat_model_cuda.py       # Python model class (GaussianSplatModelCUDA)
├── build.py                   # Build script (uses torch.utils.cpp_extension)
├── setup.py                   # Legacy setuptools config (optional)
├── benchmark.py               # Performance benchmarks
├── __init__.py
├── tests/
│   ├── conftest.py            # Pytest fixtures and configuration
│   ├── test_cuda_forward.py   # Forward pass correctness tests
│   ├── test_cuda_backward.py  # Backward pass correctness tests
│   ├── test_cuda_gradcheck.py # Gradient correctness (autograd comparison)
│   ├── test_cuda_numerical.py # Numerical precision tests
│   ├── test_cuda_comparison.py # CUDA vs PyTorch reference comparison
│   ├── test_cuda_model.py     # Full model integration tests
│   ├── test_cuda_nd.py        # nD (4D-8D) tests
│   ├── test_cuda_fp16.py      # FP16 mode tests
│   ├── test_cuda_performance.py # Performance benchmarks
│   └── test_cuda_review_fixes.py # Regression tests for specific bug fixes
├── SPECIFICATIONS.md                    # Core algorithms spec
├── SPECIFICATIONS_PYTORCH_INTEGRATION.md  # This file
├── SPECIFICATIONS_TESTING.md            # Testing strategy spec
├── OPTIMIZATION_REPORT.md               # Tile-based → splat-centric transition
├── OPTIMIZATION_ROADMAP.md              # Future optimization plans
└── README.md
```

### 8.2 Backend Protocol (Actual Signatures)

The actual C++ bindings are defined in `forward_wrapper()` and `backward_wrapper()`
in `bindings.cpp`. These are the Python-facing signatures exposed as
`cuda_splatting_backend.forward()` and `cuda_splatting_backend.backward()`.

**Forward signature** (see `forward_wrapper()` in `bindings.cpp`):

```python
def forward(
    centers: Tensor,         # (N, d) float32 - splat centers in voxel coords
    conic: Tensor,           # (N, d*(d+1)/2) float32 - packed upper-tri inverse covariance
    amps: Tensor,            # (N,) float32 - amplitudes
    L_row_norms: Tensor,     # (N, d) float32 - per-axis std dev from Cholesky row norms
    shape: list[int],        # target volume shape (d elements)
    truncate: float,         # base truncation radius in std devs
    intensity_floor: float,  # minimum intensity threshold for culling
    tile_size: int,          # tile size for spatial binning (legacy, still required)
    batch_size: int = 128,   # shared memory batch size (32, 128, or 256; legacy)
    use_fp16: bool = False,  # use FP16 precision for inputs
    output_buffer: Optional[Tensor] = None,  # pre-zeroed buffer to reuse
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Returns 7-tuple:
        output:           (prod(shape),) float32 - rendered volume (flattened)
        tile_counts:      (num_tiles,) int32 - splats per tile (diagnostic)
        tile_offsets:     (num_tiles,) int64 - exclusive prefix sum (diagnostic)
        tile_content:     (0,) int32 - empty (tile binning eliminated)
        global_splat_ids: (num_global,) int32 - global splat IDs
        shape_tensor:     (d,) int32 - volume shape on device (for backward reuse)
        tile_dims_tensor: (d,) int32 - tile dims on device (for backward reuse)
    """
```

**Backward signature** (see `backward_wrapper()` in `bindings.cpp`):

```python
def backward(
    grad_output: Tensor,             # (prod(shape),) float32 - upstream gradient
    centers: Tensor,                 # (N, d) float32 - from forward
    conic: Tensor,                   # (N, d*(d+1)/2) float32 - from forward
    amps: Tensor,                    # (N,) float32 - from forward
    tile_offsets: Tensor,            # from forward (unused by splat-centric path)
    tile_counts: Tensor,             # from forward (unused by splat-centric path)
    tile_content: Tensor,            # from forward (unused by splat-centric path)
    global_splat_ids: Tensor,        # from forward (unused by splat-centric path)
    shape: list[int],                # target volume shape
    truncate: float,                 # base truncation radius
    intensity_floor: float,          # minimum intensity threshold
    tile_size: int,                  # tile size (legacy)
    batch_size: int = 128,           # shared memory batch size (legacy)
    use_fp16: bool = False,          # must match forward
    shape_tensor_cached: Optional[Tensor] = None,     # cached from forward
    tile_dims_tensor_cached: Optional[Tensor] = None,  # cached from forward
    output_to_zero: Optional[Tensor] = None,           # forward output to zero as side effect
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns 3-tuple:
        d_centers: (N, d) float32 - center gradients
        d_conic:   (N, d*(d+1)/2) float32 - conic gradients
        d_amps:    (N,) float32 - amplitude gradients
    """
```

> **Note on legacy parameters**: `tile_offsets`, `tile_counts`, `tile_content`,
> `global_splat_ids`, `tile_size`, and `batch_size` are retained in the backward
> signature for API compatibility. The splat-centric backward kernel ignores them
> entirely -- it recomputes the AABB from the conic. The `shape_tensor_cached` and
> `tile_dims_tensor_cached` parameters enable reuse of device tensors from the
> forward pass, avoiding redundant host-to-device copies.

> **Note on sharpness**: The original protocol included a `sharpness` parameter.
> The current implementation uses a fixed standard Gaussian (s=2), so `sharpness`
> has been removed from the CUDA backend interface. The shifted Gaussian formulas
> in the kernels use `exp(-0.5 * D^2)` directly.

> **Note on return tuple sizes**: Forward returns 7 tensors (up from the original
> spec's 4), and backward returns 3 tensors (down from 4, since `d_sharpness` is
> removed). The extra forward outputs (`global_splat_ids`, `shape_tensor`,
> `tile_dims_tensor`) support backward pass optimizations.

**Benefits of Backend Protocol**:

| Scenario | Backend | GPU Required |
|----------|---------|--------------|
| Production | `cuda_splatting_backend` | Yes |
| Python logic tests | `MockSplattingBackend` | No |
| Numerical validation | `GaussianSplatModel` (PyTorch) | Optional |
| CI without GPU | `MockSplattingBackend` | No |
| Gradcheck | Real backend | Yes |

### 8.3 Python Model Class

```python
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

# Import CUDA backend (compiled from C++/CUDA)
import cuda_splatting_backend

# Import base model for parameter management
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factor L to packed conic (Σ⁻¹).

    Args:
        L: (N, DIM, DIM) lower triangular Cholesky factor

    Returns:
        conic: (N, DIM*(DIM+1)/2) packed upper-triangular inverse covariance
    """
    Sigma = torch.bmm(L, L.transpose(-1, -2))
    Sigma_inv = torch.linalg.inv(Sigma)

    N, DIM = L.shape[0], L.shape[1]
    conic_size = (DIM * (DIM + 1)) // 2

    # Extract and pack upper triangle
    indices = torch.triu_indices(DIM, DIM)
    conic = Sigma_inv[:, indices[0], indices[1]].reshape(N, conic_size)

    return conic.contiguous()


class GaussianSplatModelCUDA(torch.nn.Module):
    """
    CUDA-accelerated Gaussian splat model for NVIDIA GPUs.

    Drop-in replacement for GaussianSplatModel with identical API.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        centers0: np.ndarray,
        L0: np.ndarray,
        amps0: np.ndarray,
        sigma_min_diag: Sequence[float],
        sigma_max_diag: Optional[Sequence[float]] = None,
        truncate: float = 3.0,
        intensity_floor: float = 1e-5,
        tile_size: Optional[int] = None,  # Auto-select based on dimension
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # === Input Validation ===

        # Device validation
        device = device or torch.device("cuda")
        if not str(device).startswith("cuda"):
            raise ValueError(f"CUDA backend requires CUDA device (got {device})")

        # Dimension validation
        self.dim = len(shape)
        if self.dim < 2 or self.dim > 8:
            raise ValueError(f"CUDA backend supports 2D-8D (got {self.dim}D)")

        # Shape validation
        for i, s in enumerate(shape):
            if s <= 0:
                raise ValueError(f"shape[{i}] must be positive (got {s})")

        # Input array shape validation
        N = centers0.shape[0]
        if centers0.shape != (N, self.dim):
            raise ValueError(
                f"centers0 shape mismatch: expected ({N}, {self.dim}), "
                f"got {centers0.shape}"
            )
        if L0.shape != (N, self.dim, self.dim):
            raise ValueError(
                f"L0 shape mismatch: expected ({N}, {self.dim}, {self.dim}), "
                f"got {L0.shape}"
            )
        if amps0.shape != (N,):
            raise ValueError(
                f"amps0 shape mismatch: expected ({N},), got {amps0.shape}"
            )

        # Dtype validation
        if centers0.dtype != np.float32:
            centers0 = centers0.astype(np.float32)
        if L0.dtype != np.float32:
            L0 = L0.astype(np.float32)
        if amps0.dtype != np.float32:
            amps0 = amps0.astype(np.float32)

        # Finite value check
        if not np.all(np.isfinite(centers0)):
            raise ValueError("centers0 contains non-finite values (inf or nan)")
        if not np.all(np.isfinite(L0)):
            raise ValueError("L0 contains non-finite values (inf or nan)")
        if not np.all(np.isfinite(amps0)):
            raise ValueError("amps0 contains non-finite values (inf or nan)")

        # Parameter range validation
        if truncate <= 0:
            raise ValueError(f"truncate must be positive (got {truncate})")
        if intensity_floor < 0:
            raise ValueError(f"intensity_floor must be non-negative (got {intensity_floor})")
        if len(sigma_min_diag) != self.dim:
            raise ValueError(
                f"sigma_min_diag length mismatch: expected {self.dim}, "
                f"got {len(sigma_min_diag)}"
            )
        if sigma_max_diag is not None and len(sigma_max_diag) != self.dim:
            raise ValueError(
                f"sigma_max_diag length mismatch: expected {self.dim}, "
                f"got {len(sigma_max_diag)}"
            )

        # Tile size validation (if provided)
        # NOTE: tile_size=32 is valid for 2D (32×32 = 1024 threads, max block size)
        if tile_size is not None:
            if tile_size not in (2, 4, 8, 16, 32):
                raise ValueError(
                    f"tile_size must be 2, 4, 8, 16, or 32 (got {tile_size})"
                )
            # Additional check: 32 only valid for 2D (32^3 = 32768 > max threads)
            if tile_size == 32 and self.dim > 2:
                raise ValueError(
                    f"tile_size=32 only valid for 2D (got {self.dim}D)"
                )

        # === End Input Validation ===

        # Create base model for parameter management
        self._base = GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=sigma_min_diag,
            sigma_max_diag=sigma_max_diag,
            truncate=truncate,
            device=device or torch.device("cuda"),
        )

        # Auto-select tile size based on dimension
        if tile_size is None:
            tile_size = self._auto_tile_size()

        self._shape = shape
        self._truncate = truncate
        self._intensity_floor = intensity_floor
        self._tile_size = tile_size

    def _auto_tile_size(self) -> int:
        """
        Select tile size automatically based on dimension.

        Strategy: Maximize occupancy while staying within thread limits.
        tile_size^dim <= 1024 (max threads per block)

        Returns:
            Tile size (power of 2)
        """
        # Map dimension to optimal tile size (see table in Section 4.3)
        tile_size_map = {
            2: 16,   # 16² = 256 threads (could use 32 for 1024)
            3: 8,    # 8³ = 512 threads
            4: 4,    # 4⁴ = 256 threads
            5: 4,    # 4⁵ = 1024 threads (max)
            6: 2,    # 2⁶ = 64 threads
            7: 2,    # 2⁷ = 128 threads
            8: 2,    # 2⁸ = 256 threads
        }
        return tile_size_map.get(self.dim, 2)

    def forward(self) -> torch.Tensor:
        centers, Ls, amps, sharpness = self._base.current_params()

        return CUDASplatFunction.apply(
            centers,
            Ls,
            amps,
            sharpness,
            self._shape,
            self._truncate,
            self._intensity_floor,
            self._tile_size,
        )
```

### 8.4 Multi-Stream Support (Phase 5)

Modern PyTorch uses CUDA streams for async execution. For optimal performance,
integrate with PyTorch's stream management:

```cpp
// Get current stream from PyTorch
cudaStream_t get_current_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// All kernel launches use the current stream
void forward_pass(
    const torch::Tensor& centers,
    const torch::Tensor& conic,
    // ...
    torch::Tensor& output
) {
    cudaStream_t stream = get_current_stream();

    // All operations on same stream for correct ordering
    preprocess_nd<DIM><<<grid1, block1, 0, stream>>>(...);
    cub::DeviceScan::ExclusiveSum(..., stream);
    bin_nd<DIM><<<grid2, block2, 0, stream>>>(...);
    rasterize_fwd_nd<DIM><<<grid3, block3, smem_size, stream>>>(...);
}
```

**Async Memory Operations**:
```cpp
// Use async memset for gradient zeroing
cudaMemsetAsync(d_centers, 0, N * DIM * sizeof(float), stream);
cudaMemsetAsync(d_amps, 0, N * sizeof(float), stream);

// Use async copies for total_pairs computation (see Section 5.1)
cudaMemcpyAsync(&host_val, &device_val, sizeof(int), cudaMemcpyDeviceToHost, stream);
```

**Overlap Opportunities** (10-20% speedup potential):
- Preprocess kernel can overlap with gradient zeroing (different memory)
- Host-side total_pairs copy can overlap with binning kernel

### 8.5 Custom Autograd Function

```python
class CUDASplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, Ls, amps, sharpness, shape, truncate, intensity_floor, tile_size):
        # Convert L to conic (Σ⁻¹)
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        conic = cholesky_to_conic(Ls_for_conic)

        # Compute exact L_row_norms for AABB computation
        L_row_norms = torch.sqrt(torch.sum(Ls * Ls, dim=2))

        # Dispatch to CUDA
        output, tile_counts, tile_offsets, tile_content = cuda_splatting_backend.forward(
            centers.contiguous(),
            conic.contiguous(),
            amps.contiguous(),
            sharpness.contiguous(),
            L_row_norms.contiguous(),
            list(shape),
            truncate,
            intensity_floor,
            tile_size,
        )

        # Save for backward
        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps, sharpness)
        ctx.shape = shape
        ctx.truncate = truncate
        ctx.intensity_floor = intensity_floor
        ctx.tile_size = tile_size
        ctx.tile_counts = tile_counts
        ctx.tile_offsets = tile_offsets
        ctx.tile_content = tile_content

        return output

    @staticmethod
    def backward(ctx, grad_output):
        centers, Ls, Ls_for_conic, conic, amps, sharpness = ctx.saved_tensors

        # CUDA backward for d_centers, d_conic, d_amps, d_sharpness
        d_centers, d_conic, d_amps, d_sharpness = cuda_splatting_backend.backward(
            grad_output.contiguous(),
            centers,
            conic,
            amps,
            sharpness,
            ctx.tile_offsets,
            ctx.tile_counts,
            ctx.tile_content,
            list(ctx.shape),
            ctx.truncate,
            ctx.intensity_floor,
            ctx.tile_size,
        )

        # Chain rule: d_conic → d_Ls
        with torch.enable_grad():
            conic_recomputed = cholesky_to_conic(Ls_for_conic)

        d_Ls, = torch.autograd.grad(
            outputs=conic_recomputed,
            inputs=Ls_for_conic,
            grad_outputs=d_conic,
        )

        return d_centers, d_Ls, d_amps, d_sharpness, None, None, None, None
```

### 8.6 Modern torch.library API (Recommended for PyTorch 2.0+)

The `torch.library` API provides better integration with PyTorch 2.x features like
`torch.compile`, functorch transforms, and vmap. This is the **recommended approach**
for new code.

#### 8.6.1 Defining the Custom Op

```python
import torch
from torch import Tensor
from torch.library import custom_op, register_fake

# Define the custom op namespace
LIBRARY = torch.library.Library("luxar_cuda", "DEF")

# Forward op
@custom_op("luxar_cuda::gaussian_splat_forward", mutates_args=())
def gaussian_splat_forward(
    centers: Tensor,
    conic: Tensor,
    amps: Tensor,
    sharpness: Tensor,
    L_row_norms: Tensor,
    shape: list[int],
    truncate: float,
    intensity_floor: float,
    tile_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Forward pass returning (output, tile_counts, tile_offsets, tile_content)."""
    return cuda_splatting_backend.forward(
        centers.contiguous(),
        conic.contiguous(),
        amps.contiguous(),
        sharpness.contiguous(),
        L_row_norms.contiguous(),
        shape,
        truncate,
        intensity_floor,
        tile_size,
    )


# Backward op (exposed as separate op for flexibility)
@custom_op("luxar_cuda::gaussian_splat_backward", mutates_args=())
def gaussian_splat_backward(
    grad_output: Tensor,
    centers: Tensor,
    conic: Tensor,
    amps: Tensor,
    sharpness: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    tile_content: Tensor,
    shape: list[int],
    truncate: float,
    intensity_floor: float,
    tile_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backward pass returning (d_centers, d_conic, d_amps, d_sharpness)."""
    return cuda_splatting_backend.backward(
        grad_output.contiguous(),
        centers,
        conic,
        amps,
        sharpness,
        tile_offsets,
        tile_counts,
        tile_content,
        shape,
        truncate,
        intensity_floor,
        tile_size,
    )
```

#### 8.6.2 Fake Tensor Registration (for torch.compile)

Fake tensors enable tracing without actual computation. Required for `torch.compile`:

```python
@register_fake("luxar_cuda::gaussian_splat_forward")
def gaussian_splat_forward_fake(
    centers: Tensor,
    conic: Tensor,
    amps: Tensor,
    sharpness: Tensor,
    shape: list[int],
    truncate: float,
    intensity_floor: float,
    tile_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return fake tensors with correct shapes/dtypes for tracing."""
    import math

    device = centers.device
    dtype = centers.dtype

    # Output shape
    output_numel = math.prod(shape)
    output = torch.empty(output_numel, device=device, dtype=dtype)

    # Tile metadata shapes (approximate - exact depends on spatial distribution)
    DIM = centers.shape[1]
    num_tiles = math.prod((s + tile_size - 1) // tile_size for s in shape)

    tile_counts = torch.empty(num_tiles, device=device, dtype=torch.int32)
    tile_offsets = torch.empty(num_tiles, device=device, dtype=torch.int64)

    # tile_content size is data-dependent, use conservative estimate
    N = centers.shape[0]
    avg_tiles_per_splat = min(8, num_tiles)  # Conservative estimate
    max_pairs = N * avg_tiles_per_splat
    tile_content = torch.empty(max_pairs, device=device, dtype=torch.int32)

    return output, tile_counts, tile_offsets, tile_content


@register_fake("luxar_cuda::gaussian_splat_backward")
def gaussian_splat_backward_fake(
    grad_output: Tensor,
    centers: Tensor,
    conic: Tensor,
    amps: Tensor,
    sharpness: Tensor,
    tile_offsets: Tensor,
    tile_counts: Tensor,
    tile_content: Tensor,
    shape: list[int],
    truncate: float,
    intensity_floor: float,
    tile_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return fake gradient tensors with correct shapes."""
    d_centers = torch.empty_like(centers)
    d_conic = torch.empty_like(conic)
    d_amps = torch.empty_like(amps)
    d_sharpness = torch.empty_like(sharpness)
    return d_centers, d_conic, d_amps, d_sharpness
```

#### 8.6.3 Autograd Setup

Register the backward pass using `setup_context` and `backward`:

```python
def setup_context(ctx, inputs, output):
    """Save tensors needed for backward."""
    centers, conic, amps, sharpness, shape, truncate, intensity_floor, tile_size = inputs
    output_tensor, tile_counts, tile_offsets, tile_content = output

    ctx.save_for_backward(centers, conic, amps, sharpness, tile_offsets, tile_counts, tile_content)
    ctx.shape = shape
    ctx.truncate = truncate
    ctx.intensity_floor = intensity_floor
    ctx.tile_size = tile_size


def backward(ctx, grad_output, _grad_tile_counts, _grad_tile_offsets, _grad_tile_content):
    """Compute gradients."""
    centers, conic, amps, sharpness, tile_offsets, tile_counts, tile_content = ctx.saved_tensors

    d_centers, d_conic, d_amps, d_sharpness = torch.ops.luxar_cuda.gaussian_splat_backward(
        grad_output,
        centers,
        conic,
        amps,
        sharpness,
        tile_offsets,
        tile_counts,
        tile_content,
        ctx.shape,
        ctx.truncate,
        ctx.intensity_floor,
        ctx.tile_size,
    )

    # Return gradients for each input (None for non-tensor args)
    return d_centers, d_conic, d_amps, d_sharpness, None, None, None, None


# Register autograd formula
torch.library.register_autograd(
    "luxar_cuda::gaussian_splat_forward",
    backward,
    setup_context=setup_context,
)
```

#### 8.6.4 Usage with torch.compile

The torch.library approach enables seamless use with `torch.compile`:

```python
# Model using torch.library ops
class GaussianSplatModelCUDA(torch.nn.Module):
    def forward(self):
        centers, Ls, amps, sharpness = self._base.current_params()
        conic = cholesky_to_conic(Ls)

        # Use the registered op
        output, _, _, _ = torch.ops.luxar_cuda.gaussian_splat_forward(
            centers, conic, amps, sharpness,
            list(self._shape), self._truncate, self._intensity_floor, self._tile_size
        )
        return output.view(self._shape)


# Compile the model for additional speedup
model = GaussianSplatModelCUDA(...)
compiled_model = torch.compile(model, mode="reduce-overhead")

# Use normally - backward works automatically
output = compiled_model()
loss = (output - target).pow(2).mean()
loss.backward()  # Uses registered backward
```

#### 8.6.5 Benefits of torch.library vs torch.autograd.Function

| Feature | torch.autograd.Function | torch.library |
|---------|------------------------|---------------|
| torch.compile support | Limited | Full |
| vmap support | Manual | Automatic (with register_fake) |
| AOT autograd | No | Yes |
| Debugging | Harder | Better tracing |
| Future compatibility | Legacy | Recommended |

**Recommendation**: Use torch.library for new development. Keep torch.autograd.Function
as fallback for PyTorch <2.0 compatibility if needed.

---

## 9. Performance Targets

### 9.1 Benchmark Scenarios

| Scenario | Dimensions | Volume Size | Splats | Target Speedup (vs CPU) |
|----------|------------|-------------|--------|------------------------|
| Small 2D | 2 | 512×512 | 1K | 20× |
| Medium 2D | 2 | 2048×2048 | 10K | 50× |
| Small 3D | 3 | 128³ | 1K | 30× |
| Medium 3D | 3 | 256³ | 10K | 50× |
| Large 3D | 3 | 512³ | 100K | 100× |
| 4D Hyper | 4 | 64⁴ | 5K | 40× |

### 9.2 Memory Footprint Targets

| Component | Memory Formula | 3D 256³, 10K splats |
|-----------|---------------|---------------------|
| Tile counts | `num_tiles × 4B` | 128KB (32³ tiles) |
| Tile offsets | `num_tiles × 4B` | 128KB |
| Tile content | `avg_splats_per_tile × num_tiles × 4B` | ~1MB |
| Gradient buffers | `N × (DIM + conic_size + 2) × 4B` | ~400KB |
| **Total overhead** | | ~2MB |

### 9.3 Latency Breakdown (Target)

For 256³ volume, 10K splats:

| Stage | Target Time | % of Total |
|-------|-------------|------------|
| Preprocess | 0.2ms | 5% |
| Prefix sum | 0.1ms | 2% |
| Binning | 0.3ms | 8% |
| Rasterize fwd | 2.0ms | 50% |
| Rasterize bwd | 1.4ms | 35% |
| **Total** | **4ms** | 100% |

### 9.4 Occupancy Analysis

GPU occupancy affects performance significantly. High-dimensional kernels use more registers,
limiting occupancy.

**Register Usage Estimates (per thread)**:

| Variable | Size | 2D | 3D | 4D | 8D |
|----------|------|----|----|----|----|
| `px[DIM]` | DIM | 2 | 3 | 4 | 8 |
| `d[DIM]` | DIM | 2 | 3 | 4 | 8 |
| `mu[DIM]` | DIM | 2 | 3 | 4 | 8 |
| `C[conic]` | DIM*(DIM+1)/2 | 3 | 6 | 10 | 36 |
| `g_centers[DIM]` | DIM | 2 | 3 | 4 | 8 |
| `g_conic[conic]` | DIM*(DIM+1)/2 | 3 | 6 | 10 | 36 |
| Scalars (~20) | 20 | 20 | 20 | 20 | 20 |
| **Total** | | 34 | 44 | 56 | 124 |

**Occupancy Limits (Ampere - 65536 regs/SM, 255 max/thread)**:

| Dimension | Regs/Thread | Max Threads/SM | Blocks/SM (256 threads) | Occupancy |
|-----------|-------------|----------------|-------------------------|-----------|
| 2D | ~40 | 1638 | 6 | 75% |
| 3D | ~50 | 1310 | 5 | 62.5% |
| 4D | ~64 | 1024 | 4 | 50% |
| 8D | ~128 | 512 | 2 | 25% |

**CRITICAL: Register Pressure Mitigation (REQUIRED for D>3)**

The compiler is aggressive with loop unrolling. In 5D+, if you unroll `mahalanobis_distance`,
that code bloat combined with template instantiation for `iterate_tile_range` will spill to
Local Memory (LMEM), killing performance. **LMEM spills can cause 10× slowdown.**

1. **REQUIRED: Use `__launch_bounds__`** with dimension-specific limits:
   ```cuda
   // 2D/3D: High occupancy, more registers allowed
   template<>
   __global__ __launch_bounds__(256, 4)  // 64 regs/thread max
   void rasterize_fwd_nd<2>(...) { ... }

   template<>
   __global__ __launch_bounds__(256, 4)
   void rasterize_fwd_nd<3>(...) { ... }

   // 4D: Moderate occupancy
   template<>
   __global__ __launch_bounds__(256, 2)  // 128 regs/thread max
   void rasterize_fwd_nd<4>(...) { ... }

   // 5D+: Accept lower occupancy, prevent spills
   template<>
   __global__ __launch_bounds__(128, 2)  // 256 regs/thread max, smaller blocks
   void rasterize_fwd_nd<5>(...) { ... }
   ```

2. **REQUIRED: Prevent unrolling for D>4**:
   ```cuda
   // In mahalanobis_distance for high-D:
   template<int DIM>
   __device__ __forceinline__ float mahalanobis_distance(
       const float* d, const float* C
   ) {
       float result = 0.0f;
       int idx = 0;

       // For D<=3: Full unroll (fast)
       // For D>3:  NO UNROLL - instruction overhead cheaper than LMEM spills
       #if DIM <= 3
       #pragma unroll
       #else
       #pragma unroll 1  // PREVENT unrolling
       #endif
       for (int i = 0; i < DIM; i++) {
           result += d[i] * d[i] * C[idx++];
           #if DIM <= 3
           #pragma unroll
           #else
           #pragma unroll 1
           #endif
           for (int j = i + 1; j < DIM; j++) {
               result += 2.0f * d[i] * d[j] * C[idx++];
           }
       }
       return result;
   }
   ```

3. **REQUIRED: Verify no LMEM spills** in Nsight Compute:
   ```bash
   ncu --metrics lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum \
       --kernel-name "rasterize_fwd_nd" ./your_kernel

   # If lts__t_sectors_* > 0 for your kernel, you have spills. Fix immediately.
   ```

4. **Spill to local memory** ONLY for rarely-used arrays (gradient buffers in backward pass)
   that are written once at end, not accessed in inner loops.

5. **Reduce block size** for high-D: 128 threads instead of 256 (as shown in launch_bounds).

6. **Kernel specialization**: 2D/3D are fully unrolled templates; 5D+ use loop-based
   generic kernel (see Section 10 Compilation Strategy).

### 9.5 Memory Bandwidth Analysis

**Forward Pass (per pixel)**:

| Access | Size | Type | Notes |
|--------|------|------|-------|
| tile_content | 4B × splats_in_tile | Global | Sequential, cacheable |
| centers | 4B × DIM | Global | Per-splat, cached in L2 |
| conic | 4B × conic_size | Global | Per-splat, cached in L2 |
| amps + sharpness | 8B | Global | Per-splat |
| output (write) | 4B | Global | One write per pixel |

**Arithmetic Intensity** (FLOPs per byte):
- Mahalanobis distance: ~DIM² FLOPs
- Gaussian exp: ~20 FLOPs
- Total per splat: ~DIM² + 30 FLOPs
- Bytes per splat: ~4×(DIM + conic_size + 2) ≈ 4×DIM²

**AI ≈ (DIM² + 30) / (4×DIM²) ≈ 0.25** (memory-bound)

**Optimization**: Shared memory batching amortizes global memory reads,
effectively increasing arithmetic intensity.

### 9.6 Profiling Methodology

**Key Nsight Compute Metrics**:

```bash
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    sm__sass_average_data_bytes_per_sector_mem_shared_op_ld,\
    smsp__sass_average_branch_targets_threads_uniform.pct \
    ./your_kernel
```

| Metric | Target | Action if Below |
|--------|--------|-----------------|
| Occupancy | >50% | Reduce register pressure |
| Memory Throughput | >70% | Improve coalescing |
| Shared Memory BW | >80% | Check bank conflicts |
| Branch Efficiency | >95% | Reduce warp divergence |

**NVTX Markers** (for Nsight Systems):
```cpp
#include <nvtx3/nvToolsExt.h>

void forward_pass(...) {
    nvtxRangePush("Preprocess");
    preprocess_nd<<<...>>>(...);
    nvtxRangePop();

    nvtxRangePush("PrefixSum");
    cub::DeviceScan::ExclusiveSum(...);
    nvtxRangePop();

    // ... etc
}
```

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure

- [x] Setup CUDA extension build system (build.py using torch.utils.cpp_extension)
- [ ] Implement pybind11 bindings skeleton
- [ ] Create `GaussianSplatModelCUDA` Python class (with CPU fallback)
- [ ] Implement `cholesky_to_conic` for 2D/3D (CUDA kernel)
- [ ] Unit tests for L → conic conversion

### Phase 2: Forward Pass

- [ ] Implement `preprocess_nd` kernel (3D first)
- [ ] Implement CUB prefix sum integration
- [ ] Implement `bin_nd` kernel
- [ ] Implement `rasterize_fwd_nd` kernel (3D)
- [ ] Numerical validation against PyTorch reference
- [ ] Extend to 2D
- [ ] Benchmark forward pass performance

### Phase 3: Backward Pass

- [ ] Implement `rasterize_bwd_nd` kernel (3D)
- [ ] Add warp-level reduction optimization
- [ ] Chain rule integration for L gradients (Python)
- [ ] Gradient validation with `torch.autograd.gradcheck`
- [ ] Extend to 2D
- [ ] Benchmark backward pass performance

### Phase 4: Optimization & nD

- [ ] Shared memory batch loading (BalanceGS pattern)
- [ ] Memory coalescing optimization (AoS layout)
- [ ] Extend to 4D-8D dimensions
- [ ] Add NVTX profiling markers
- [ ] Performance tuning for different GPU architectures
- [ ] Documentation and examples

### Phase 5: Production Hardening

- [ ] Error handling and input validation
- [ ] Multi-stream support
- [ ] Memory pool for workspace allocation
- [ ] Comprehensive test suite
- [ ] CI/CD integration
- [ ] Benchmarking suite

### 10.6 Compilation Strategy (REQUIRED)

**Problem**: Templating `dispatch_forward<DIM>` for DIM=2..8 generates a cartesian product of
kernels. If you also template `block_size` or `tile_size`, `nvcc` compilation time skyrockets
(10+ minutes) and the `.so` binary can be hundreds of MBs.

**REQUIRED Strategy: Hybrid Template + Generic**

```cpp
// Pre-compile fully-optimized templates for common dimensions:
template void dispatch_forward<2>(...);  // Fully unrolled, fast
template void dispatch_forward<3>(...);  // Fully unrolled, fast
template void dispatch_forward<4>(...);  // Partially unrolled

// For 5D-8D: Single generic kernel with runtime dimension
// Uses loop-based (non-unrolled) logic, passing actual_dim as argument
template<int DIM_MAX = 8>
__global__ __launch_bounds__(128, 2)
void rasterize_fwd_generic(
    int actual_dim,  // Runtime dimension (5-8)
    const float* __restrict__ centers,
    const float* __restrict__ conic,
    // ... rest of params
) {
    // Use loop-based iteration, NOT template recursion
    float px[DIM_MAX];
    float d[DIM_MAX];
    float C[DIM_MAX * (DIM_MAX + 1) / 2];

    // Compute pixel coords using actual_dim (runtime)
    for (int dim = 0; dim < actual_dim; dim++) {
        // ... loop-based coordinate computation
    }

    // Mahalanobis with runtime dimension (NO unrolling)
    float dist_sq = 0.0f;
    int idx = 0;
    for (int i = 0; i < actual_dim; i++) {
        dist_sq += d[i] * d[i] * C[idx++];
        for (int j = i + 1; j < actual_dim; j++) {
            dist_sq += 2.0f * d[i] * d[j] * C[idx++];
        }
    }
    // ... rest of kernel
}

// Dispatcher selects optimized template or generic
void dispatch_forward(int dim, ...) {
    switch (dim) {
        case 2: dispatch_forward_impl<2>(...); break;
        case 3: dispatch_forward_impl<3>(...); break;
        case 4: dispatch_forward_impl<4>(...); break;
        default:
            // 5D-8D use generic kernel
            rasterize_fwd_generic<8><<<grid, 128>>>(dim, ...);
    }
}
```

**Trade-offs**:

| Dimension | Strategy | Performance | Compile Time |
|-----------|----------|-------------|--------------|
| 2D | Full template | 100% | ~2 min |
| 3D | Full template | 100% | ~2 min |
| 4D | Partial unroll | 95% | ~2 min |
| 5D-8D | Generic kernel | 80-90% | Shared (~30s) |

**Total compile time**: ~7 minutes (vs 30+ minutes for full 8-way templates).

**Binary size impact**: ~50% reduction by avoiding 5D-8D template explosion.

### 10.7 Memory Allocation Strategy (REQUIRED)

**Problem**: Raw `cudaMalloc` for persistent `BinningState` buffers causes issues:
1. Reference counting conflict with PyTorch tensor lifecycle
2. Memory fragmentation in long training runs
3. OOM when other processes need GPU memory

**REQUIRED: Use PyTorch Caching Allocator**

```cpp
#include <c10/cuda/CUDACachingAllocator.h>

// BAD: Raw CUDA allocation (don't do this)
// cudaMalloc(&state.tile_counts, num_tiles * sizeof(int));

// GOOD: Use PyTorch's caching allocator
struct BinningState {
    torch::Tensor tile_counts;       // (num_tiles,)
    torch::Tensor tile_offsets;      // (num_tiles,)
    torch::Tensor tile_content;      // (max_pairs,)
    torch::Tensor tile_write_heads;  // (num_tiles,)
    torch::Tensor global_splat_flags; // (N,)
    torch::Tensor global_splat_ids;   // (max_global_splats,)

    // CUB temp storage - also use PyTorch allocation
    torch::Tensor scan_temp_storage;

    void allocate(int num_tiles, int N, int max_pairs, torch::Device device) {
        auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
        auto options_i64 = torch::TensorOptions().dtype(torch::kInt64).device(device);

        tile_counts = torch::empty({num_tiles}, options);
        tile_offsets = torch::empty({num_tiles}, options_i64);
        tile_content = torch::empty({max_pairs}, options);
        tile_write_heads = torch::empty({num_tiles}, options);
        global_splat_flags = torch::empty({N}, options);
        global_splat_ids = torch::empty({N}, options);  // Upper bound

        // Query CUB temp storage size
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            nullptr, temp_bytes,
            tile_counts.data_ptr<int>(),
            tile_offsets.data_ptr<int64_t>(),
            num_tiles
        );
        scan_temp_storage = torch::empty(
            {static_cast<int64_t>(temp_bytes)},
            torch::TensorOptions().dtype(torch::kUInt8).device(device)
        );
    }

    void zero_counters(cudaStream_t stream) {
        // Use cudaMemsetAsync on tensor data pointers
        cudaMemsetAsync(
            tile_counts.data_ptr<int>(), 0,
            tile_counts.numel() * sizeof(int), stream
        );
        cudaMemsetAsync(
            tile_write_heads.data_ptr<int>(), 0,
            tile_write_heads.numel() * sizeof(int), stream
        );
    }
};

// In GaussianSplatModelCUDA:
class CUDAWorkspace {
    BinningState state_;

public:
    void ensure_capacity(int num_tiles, int N, int max_pairs, torch::Device device) {
        // Only reallocate if capacity insufficient
        if (!state_.tile_counts.defined() ||
            state_.tile_counts.size(0) < num_tiles) {
            state_.allocate(num_tiles, N, max_pairs, device);
        }
    }

    // Raw pointers for kernel launches
    int* tile_counts_ptr() { return state_.tile_counts.data_ptr<int>(); }
    int64_t* tile_offsets_ptr() { return state_.tile_offsets.data_ptr<int64_t>(); }
    // ... etc
};
```

**Benefits**:
1. **Memory coexistence**: PyTorch's allocator handles pressure from other tensors
2. **Automatic cleanup**: Tensors are freed when model is deleted
3. **Memory pooling**: Repeated allocations reuse cached blocks
4. **Debugging**: Memory shows up in `torch.cuda.memory_stats()`

**Alternative for CUB**: If CUB requires raw pointers:
```cpp
// Get raw pointer from PyTorch-allocated tensor
void* temp_ptr = scan_temp_storage.data_ptr<uint8_t>();
cub::DeviceScan::ExclusiveSum(temp_ptr, temp_bytes, ...);
```

---

## 12. References

### Primary Sources

1. **gsplat**: [github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)
   - CUDA accelerated rasterization, 4× memory reduction

2. **diff-gaussian-rasterization**: [github.com/graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
   - Original INRIA implementation, tile binning + radix sort

3. **FlashGS**: [arxiv.org/html/2408.07967v2](https://arxiv.org/html/2408.07967v2)
   - Warp divergence elimination, pipeline optimization

4. **BalanceGS**: [arxiv.org/html/2510.14564](https://arxiv.org/html/2510.14564)
   - Memory coalescing, shared memory buffering

5. **DISTWAR**: [arxiv.org/html/2505.18764v1](https://arxiv.org/html/2505.18764v1)
   - Warp-level gradient pre-accumulation

### CUDA Programming

6. **NVIDIA Warp-Level Primitives**: [developer.nvidia.com/blog/using-cuda-warp-level-primitives/](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

7. **Warp-Aggregated Atomics**: [developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

8. **CUB Library**: [nvidia.github.io/cccl/cub/](https://nvidia.github.io/cccl/cub/)
   - DeviceRadixSort, DeviceScan

9. **PyTorch Custom CUDA Operators**: [pytorch.org/tutorials/advanced/cpp_custom_ops.html](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)

### Additional Resources

10. **LichtFeld Studio**: [github.com/MrNeRF/gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda)
    - Modern C++23/CUDA 12.8+ implementation

11. **Onesweep Radix Sort**: [arxiv.org/pdf/2206.01784](https://arxiv.org/pdf/2206.01784)
    - State-of-the-art GPU sorting algorithm

12. **GPU Prefix Sums**: [github.com/b0nes164/GPUPrefixSums](https://github.com/b0nes164/GPUPrefixSums)
    - Comprehensive prefix sum implementations

---

## Appendix A: CUDA Error Handling

All CUDA API calls should be wrapped with error checking to fail fast on errors:

```cpp
// Error checking macro - throws on any CUDA error
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(err);                           \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

// Check last kernel launch error
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            std::ostringstream oss;                                            \
            oss << "CUDA kernel error at " << __FILE__ << ":" << __LINE__      \
                << " - " << cudaGetErrorString(err);                           \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

// Usage example:
void forward_pass(...) {
    CUDA_CHECK(cudaMalloc(&ptr, size));

    preprocess_nd<DIM><<<grid, block, 0, stream>>>(...);
    CUDA_CHECK_LAST();

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
```

**Debug vs Release**:
- Debug builds: Full error checking after every operation
- Release builds: Optionally disable per-kernel checks, keep allocation checks

```cpp
#ifdef NDEBUG
    #define CUDA_CHECK_KERNEL()  // No-op in release
#else
    #define CUDA_CHECK_KERNEL() CUDA_CHECK_LAST()
#endif
```

---

## Appendix B: Coordinate Conventions

### NumPy/PyTorch Convention: [Z, Y, X] (or [D₀, D₁, ..., Dₙ₋₁])

- Array indexing: `volume[z, y, x]` or `volume[d0, d1, ..., dn]`
- Centers: `[z_coord, y_coord, x_coord]`
- Cholesky L: Row indices correspond to dimension order
- Conic upper triangle (3D): `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]`

### Memory Layout

```
For 3D volume shape (D, H, W) = (depth, height, width):
- Strides: (H*W, W, 1)
- Linear index: z*H*W + y*W + x
```

### Conic Packing (Upper Triangle)

For dimension D, the conic (Σ⁻¹) is stored as upper triangle:
```
D=2: [c_00, c_01, c_11]                    (3 elements)
D=3: [c_00, c_01, c_02, c_11, c_12, c_22]  (6 elements)
D=4: [c_00, c_01, c_02, c_03, c_11, c_12, c_13, c_22, c_23, c_33]  (10 elements)

General: D*(D+1)/2 elements
```

---

## Appendix C: Error Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | Operation completed successfully |
| 1 | CUDA_ERROR | CUDA runtime error |
| 2 | INVALID_DIM | Unsupported dimension (must be 2-8) |
| 3 | INVALID_DEVICE | Tensor not on CUDA device |
| 4 | SIZE_MISMATCH | Tensor sizes don't match |
| 5 | OUT_OF_MEMORY | Failed to allocate GPU memory |
| 6 | TILE_OVERFLOW | Tile content buffer too small |

---

*Last updated: 2026-01-10*
