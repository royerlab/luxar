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
   - 8.4 [Custom Autograd Function](#84-custom-autograd-function)
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
│   ├── cuda_splatting.h       # Public API: forward(), backward(), ForwardState,
│   │                          #   forward_fp16(), backward_fp16()
│   ├── bindings.cpp           # pybind11 bindings: forward_wrapper(), backward_wrapper()
│   ├── kernels_core.cuh       # Core kernels: splat-centric forward and backward
│   ├── kernel_launchers.cuh   # Launch wrappers for splat-centric kernels
│   ├── utils.cuh              # Umbrella header (includes all sub-headers below)
│   ├── math_utils.cuh         # Mahalanobis distance, Gaussian intensity, shift params
│   ├── voxel_utils.cuh        # Voxel coordinate conversion (voxel_to_linear)
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
    shape: list[int],        # target volume shape (d elements)
    truncate: float,         # base truncation radius in std devs
    intensity_floor: float,  # minimum intensity threshold for culling
    use_fp16: bool = False,  # use FP16 precision for inputs
    output_buffer: Optional[Tensor] = None,  # pre-zeroed buffer to reuse (optional)
) -> Tuple[Tensor, Tensor]:
    """
    Returns 2-tuple:
        output:       (prod(shape),) float32 - rendered volume (flattened)
        shape_tensor: (d,) int32 - volume shape on device (for backward reuse)
    """
```

**Backward signature** (see `backward_wrapper()` in `bindings.cpp`):

```python
def backward(
    grad_output: Tensor,             # (prod(shape),) float32 - upstream gradient
    centers: Tensor,                 # (N, d) float32 - from forward
    conic: Tensor,                   # (N, d*(d+1)/2) float32 - from forward
    amps: Tensor,                    # (N,) float32 - from forward
    shape: list[int],                # target volume shape
    truncate: float,                 # base truncation radius
    intensity_floor: float,          # minimum intensity threshold
    use_fp16: bool = False,          # must match forward
    shape_tensor_cached: Optional[Tensor] = None,  # cached from forward (avoids H2D copy)
    output_to_zero: Optional[Tensor] = None,       # forward output to zero as side effect
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns 3-tuple:
        d_centers: (N, d) float32 - center gradients
        d_conic:   (N, d*(d+1)/2) float32 - conic gradients
        d_amps:    (N,) float32 - amplitude gradients
    """
```

> **Note on `shape_tensor_cached`**: The `shape_tensor_cached` parameter enables
> reuse of the device tensor from the forward pass, avoiding a redundant
> host-to-device copy in the backward pass.

> **Note on sharpness**: The original protocol included a `sharpness` parameter.
> The current implementation uses a fixed standard Gaussian (s=2), so `sharpness`
> has been removed from the CUDA backend interface. The shifted Gaussian formulas
> in the kernels use `exp(-0.5 * D^2)` directly.

> **Note on return tuple sizes**: Forward returns 2 tensors (`output` and
> `shape_tensor`), backward returns 3 tensors (`d_centers`, `d_conic`, `d_amps`).
> The `shape_tensor` from forward is reused in backward to avoid a host-to-device
> copy. Tile-related outputs were removed with the splat-centric refactor.

**Benefits of Backend Protocol**:

| Scenario | Backend | GPU Required |
|----------|---------|--------------|
| Production | `cuda_splatting_backend` | Yes |
| Python logic tests | `GaussianSplatModel` (PyTorch reference) | No |
| Numerical validation | `GaussianSplatModel` (PyTorch) | Optional |
| CI without GPU | `GaussianSplatModel` (PyTorch reference) | No |
| Gradcheck | Real backend | Yes |

### 8.3 Python Model Class

See `gsplat_model_cuda.py` for the actual implementation. Key design points:

- `GaussianSplatModelCUDA` wraps `GaussianSplatModel` (base model for parameter management)
- `CUDASplatFunction(torch.autograd.Function)` handles forward/backward dispatch
- L -> conic conversion is done in PyTorch (with `torch.compile`) for autograd chain rule
- CUDA kernels handle splat-centric rendering (forward) and gradient computation (backward)
- FP16 support via `use_fp16` flag and `torch.autocast()` detection
- Output buffer allocation uses `torch::zeros` on the C++ side (PyTorch caching allocator)

### 8.4 Custom Autograd Function

The actual implementation uses `torch.autograd.Function` with the splat-centric API.
See `gsplat_model_cuda.py` for the full implementation. Key design:

```python
class CUDASplatFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, centers, Ls, amps, shape, truncate, intensity_floor, use_fp16):
        # L → conic conversion in PyTorch (for autograd chain rule)
        Ls_for_conic = Ls.detach().clone().requires_grad_(True)
        conic = _cholesky_to_conic_compiled(Ls_for_conic)

        # Dispatch to CUDA
        result = cuda_splatting_backend.forward(
            centers_kernel, conic_kernel, amps_kernel,
            list(shape), truncate, intensity_floor, use_fp16,
        )
        output = result[0]
        shape_tensor_cached = result[1]

        ctx.save_for_backward(centers, Ls, Ls_for_conic, conic, amps)
        ctx.shape_tensor_cached = shape_tensor_cached
        # ... other non-tensor context
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # CUDA backward for d_centers, d_conic, d_amps
        d_centers, d_conic, d_amps = cuda_splatting_backend.backward(
            grad_output.contiguous(),
            ctx.centers_kernel, ctx.conic_kernel, ctx.amps_kernel,
            list(shape), truncate, intensity_floor, use_fp16,
            shape_tensor_cached=ctx.shape_tensor_cached,
        )

        # Chain rule: d_conic → d_Ls via torch.autograd.grad
        with torch.enable_grad():
            conic_recomputed = _cholesky_to_conic_compiled(Ls_for_conic)
        (d_Ls,) = torch.autograd.grad(
            outputs=conic_recomputed, inputs=Ls_for_conic, grad_outputs=d_conic,
        )

        return d_centers, d_Ls, d_amps, None, None, None, None
```

### 8.5 Usage with torch.compile

```python
# The L→conic conversion (cholesky_to_conic) is already compiled via
# @torch.compile. The full model can also be compiled for additional speedup:
model = GaussianSplatModelCUDA(...)
compiled_model = torch.compile(model, mode="reduce-overhead")

# Use normally - backward works automatically
output = compiled_model()
loss = (output - target).pow(2).mean()
loss.backward()
```

#### 8.6 Benefits of torch.library vs torch.autograd.Function

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
| Output volume | `prod(shape) × 4B` | 64MB |
| shape_tensor | `DIM × 4B` | 12B |
| Gradient buffers | `N × (DIM + conic_size + 1) × 4B` | ~400KB |
| **Total overhead** | | ~400KB (excludes output) |

### 9.3 Latency Breakdown (Target)

For 256³ volume, 10K splats:

| Stage | Target Time | % of Total |
|-------|-------------|------------|
| L→conic (torch.compile) | 0.3ms | 10% |
| Splat-centric forward | 0.6ms | 20% |
| Splat-centric backward | 1.4ms | 45% |
| d_conic→d_Ls chain rule | 0.3ms | 10% |
| Optimizer step | 0.5ms | 15% |
| **Total** | **~3ms** | 100% |

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
   void rasterize_forward_splat_centric_kernel<2>(...) { ... }

   template<>
   __global__ __launch_bounds__(256, 4)
   void rasterize_forward_splat_centric_kernel<3>(...) { ... }

   // 4D: Moderate occupancy
   template<>
   __global__ __launch_bounds__(256, 2)  // 128 regs/thread max
   void rasterize_forward_splat_centric_kernel<4>(...) { ... }

   // 5D+: Accept lower occupancy, prevent spills
   template<>
   __global__ __launch_bounds__(128, 2)  // 256 regs/thread max, smaller blocks
   void rasterize_forward_splat_centric_kernel<5>(...) { ... }
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
       --kernel-name "rasterize_forward_splat_centric_kernel" ./your_kernel

   # If lts__t_sectors_* > 0 for your kernel, you have spills. Fix immediately.
   ```

4. **Spill to local memory** ONLY for rarely-used arrays (gradient buffers in backward pass)
   that are written once at end, not accessed in inner loops.

5. **Reduce block size** for high-D: 128 threads instead of 256 (as shown in launch_bounds).

6. **Kernel specialization**: 2D/3D are fully unrolled templates; 5D+ use loop-based
   generic kernel (see Section 10 Compilation Strategy).

### 9.5 Memory Bandwidth Analysis

**Forward Pass (splat-centric, per block/splat)**:

| Access | Size | Type | Notes |
|--------|------|------|-------|
| centers | 4B × DIM | Global → Shared | Loaded once per splat |
| conic | 4B × conic_size | Global → Shared | Loaded once per splat |
| amps | 4B | Global → Shared | Loaded once per splat |
| output (write) | 4B per pixel | Global | atomicAdd per pixel in AABB |

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
    nvtxRangePush("SplatForward");
    launch_rasterize_forward_splat_centric<DIM>(...);
    nvtxRangePop();
}
```

---

## 10. Implementation Phases

### Phase 1: Core Infrastructure (DONE)

- [x] Setup CUDA extension build system (build.py using torch.utils.cpp_extension)
- [x] Implement pybind11 bindings (`bindings.cpp`)
- [x] Create `GaussianSplatModelCUDA` Python class (with CPU fallback)
- [x] Implement `cholesky_to_conic` for 2D/3D/nD (PyTorch + `torch.compile`)
- [x] Unit tests for L → conic conversion

### Phase 2: Forward Pass (DONE)

- [x] Implement splat-centric forward kernel (1 block per splat, atomicAdd)
- [x] Support 2D-8D via DIM template parameter
- [x] Numerical validation against PyTorch reference
- [x] FP16 input support (FP32 output)
- [x] Benchmark forward pass performance

### Phase 3: Backward Pass (COMPLETE)

- [x] Implement splat-centric backward kernel (1 block per splat, warp reduction)
- [x] Zero-atomics gradient accumulation (warp → block → single global write)
- [x] Chain rule integration for L gradients (Python)
- [x] Gradient validation with `torch.autograd.gradcheck`
- [x] Extend to 2D
- [x] Benchmark backward pass performance

### Phase 4: Optimization & nD (COMPLETE)

- [x] Shared memory batch loading (BalanceGS pattern)
- [x] Memory coalescing optimization (AoS layout)
- [x] Extend to 4D-8D dimensions
- [x] Add NVTX profiling markers
- [x] Performance tuning for different GPU architectures
- [x] Documentation and examples

### Phase 5: Production Hardening (COMPLETE)

- [x] Error handling and input validation
- [x] Multi-stream support
- [x] Memory pool for workspace allocation
- [x] Comprehensive test suite
- [x] CI/CD integration
- [x] Benchmarking suite

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

**Rule**: Always use PyTorch's caching allocator (`torch::zeros`, `torch::empty`) for
GPU memory. Never use raw `cudaMalloc` — it conflicts with PyTorch's memory management
and causes fragmentation.

The current splat-centric pipeline has minimal state. The only persistent structure is
`ForwardState`, which caches the shape tensor for backward reuse:

```cpp
struct ForwardState {
    torch::Tensor shape_tensor;  // (dim,) int32 - volume shape on device
};
```

**Output buffer allocation** uses `torch::zeros` for each forward call. The CUDA caching
allocator efficiently reuses freed blocks, making repeated allocations essentially free.
Pre-allocated buffer reuse was benchmarked (2026-04-08) but rejected — PyTorch's autograd
requires `.clone()` on reused buffers, which negates the savings. See the comment in
`gsplat_model_cuda.py:GaussianSplatModelCUDA.forward()` for details.

**Gradient buffers** (`d_centers`, `d_conic`, `d_amps`) are allocated via `torch::zeros`
in `backward_impl()` and freed after each backward pass.

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
    launch_rasterize_forward_splat_centric<DIM>(
        centers_ptr, conic_ptr, amps_ptr, N,
        shape_ptr, truncate, intensity_floor, output_ptr, stream);
    CUDA_CHECK_LAST();
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

---

*Last updated: 2026-04-08*
