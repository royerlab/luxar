> **HISTORICAL DOCUMENT**: This plan was written for the tile-based backward kernel architecture. The current implementation uses a **splat-centric architecture** with completely different memory layout and kernel design. See `OPTIMIZATION_REPORT.md` for the current state. This document is retained for historical reference only.

# Extended Shared Memory for Modern GPUs (A100/H100)

## Summary

Enable larger batch sizes (BATCH=256) on modern GPUs (Ampere, Ada, Hopper) by implementing extended shared memory support via `cudaFuncSetAttribute()`. This requires converting from static to dynamic shared memory allocation.

**Goal:** Reduce memory trips by processing more splats per batch on GPUs with >48KB shared memory.

## Background

The current implementation uses **static shared memory** (`__shared__` arrays), which is limited to 48KB per block on all GPUs. Modern GPUs support extended shared memory:

| GPU | SM | Max Shared/Block (default) | Max Shared/Block (opt-in) |
|-----|-----|---------------------------|---------------------------|
| Turing (TITAN RTX) | 7.5 | 48KB | 48KB (no extension) |
| Ampere (A100) | 8.0 | 48KB | 163KB |
| Ampere (RTX 3090) | 8.6 | 48KB | 100KB |
| Ada (RTX 4090) | 8.9 | 48KB | 100KB |
| Hopper (H100) | 9.0 | 48KB | 227KB |

**Key constraint:** `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)` only works with **dynamic** shared memory, not static.

## Current State

### Static Shared Memory (kernels_core.cuh)
```cuda
// Backward kernel currently uses static arrays:
__shared__ float s_centers[BATCH_SIZE * CENTER_STRIDE];
__shared__ float s_conic[BATCH_SIZE * CONIC_SIZE];
__shared__ float s_amps[BATCH_SIZE];
__shared__ float s_sharpness[BATCH_SIZE];
__shared__ int s_splat_ids[BATCH_SIZE];
__shared__ float s_truncate_sq[BATCH_SIZE];
__shared__ float s_d_centers_tile[BATCH_SIZE * CENTER_STRIDE];
__shared__ float s_d_conic_tile[BATCH_SIZE * CONIC_SIZE];
__shared__ float s_d_amps_tile[BATCH_SIZE];
__shared__ float s_d_sharpness_tile[BATCH_SIZE];
__shared__ float s_grad_output[MAX_TILE_PIXELS];
```

### Current Template Instantiations (cuda_splatting.cu)
- BATCH=32: All dims (2-8)
- BATCH=128: Dims 2-6 only (DIM=7,8 exceed 48KB)
- BATCH=256: Dims 2-4 only (DIM≥5 exceed 48KB)

## Solution Design

### Phase 1: Convert to Dynamic Shared Memory

Replace static `__shared__` arrays with a single dynamic allocation:

```cuda
// New approach: dynamic shared memory with manual offsets
template <int DIM, int BATCH_SIZE, typename InputDType>
__global__ void rasterize_backward_kernel(...) {
    // Declare dynamic shared memory
    extern __shared__ char shared_mem[];

    // Compute offsets (same formula as static, but at runtime)
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;
    constexpr int CONIC_SIZE = DIM * (DIM + 1) / 2;
    constexpr int MAX_TILE_PIXELS = /* dimension-specific */;

    // Cast shared memory to typed pointers
    float* s_centers = (float*)shared_mem;
    float* s_conic = s_centers + BATCH_SIZE * CENTER_STRIDE;
    float* s_amps = s_conic + BATCH_SIZE * CONIC_SIZE;
    // ... etc
}
```

### Phase 2: Add Shared Memory Size Computation

Add a constexpr function to compute total shared memory needed:

```cuda
template <int DIM, int BATCH_SIZE>
constexpr size_t compute_backward_smem_size() {
    constexpr int CENTER_STRIDE = (DIM == 3) ? 4 : DIM;
    constexpr int CONIC_SIZE = DIM * (DIM + 1) / 2;
    constexpr int MAX_TILE_PIXELS = /* dimension-specific */;

    return (
        BATCH_SIZE * CENTER_STRIDE * sizeof(float) +  // s_centers
        BATCH_SIZE * CONIC_SIZE * sizeof(float) +      // s_conic
        BATCH_SIZE * sizeof(float) +                   // s_amps
        BATCH_SIZE * sizeof(float) +                   // s_sharpness
        BATCH_SIZE * sizeof(int) +                     // s_splat_ids
        BATCH_SIZE * sizeof(float) +                   // s_truncate_sq
        BATCH_SIZE * CENTER_STRIDE * sizeof(float) +  // s_d_centers_tile
        BATCH_SIZE * CONIC_SIZE * sizeof(float) +      // s_d_conic_tile
        BATCH_SIZE * sizeof(float) +                   // s_d_amps_tile
        BATCH_SIZE * sizeof(float) +                   // s_d_sharpness_tile
        MAX_TILE_PIXELS * sizeof(float)                // s_grad_output
    );
}
```

### Phase 3: Configure Extended Shared Memory at Launch

In `kernel_launchers.cuh`, add `cudaFuncSetAttribute` calls:

```cuda
template <int DIM, int BATCH_SIZE, typename InputDType>
void launch_rasterize_backward(...) {
    constexpr size_t smem_size = compute_backward_smem_size<DIM, BATCH_SIZE>();

    // Configure extended shared memory if needed (>48KB)
    if constexpr (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            rasterize_backward_kernel<DIM, BATCH_SIZE, InputDType>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    // Launch with dynamic shared memory size
    rasterize_backward_kernel<DIM, BATCH_SIZE, InputDType>
        <<<grid, block_size, smem_size, stream>>>(...);
}
```

### Phase 4: Add New Template Instantiations

Enable BATCH=256 for higher dimensions (with extended shared memory):

```cpp
// cuda_splatting.cu - Add new instantiations
// BATCH=256 for dims 5-6 (requires ~51-70KB, fits in 100KB on RTX 3090/4090)
INSTANTIATE_RASTERIZE(5, 256)
INSTANTIATE_RASTERIZE(6, 256)

// BATCH=128 for dims 7-8 (requires ~47-52KB, fits with extended smem)
// Already instantiated but previously unused due to 48KB limit
```

### Phase 5: Update Python Batch Size Selection

In `gsplat_model_cuda.py`, query extended shared memory support:

```python
def _compute_optimal_batch_size(self, shared_mem_available: int, d: int) -> int:
    sm_version = self._gpu_caps["sm_version"]

    # Query max shared memory per block with opt-in extension
    max_smem_optin = self._gpu_caps.get("max_shared_memory_per_block_optin", 48 * 1024)

    # Architecture-specific selection with extended shared memory
    if sm_version >= 90:  # Hopper
        # H100: 227KB available, use BATCH=256 for all practical dims
        return self._select_batch_for_limit(max_smem_optin, d, [256, 128, 32])
    elif sm_version >= 80:  # Ampere/Ada
        # A100: 163KB, RTX 3090/4090: 100KB
        return self._select_batch_for_limit(max_smem_optin, d, [256, 128, 32])
    else:  # Turing and older
        # 48KB limit, stay conservative
        return self._select_batch_for_limit(48 * 1024, d, [32, 128])
```

## Files to Modify

| File | Changes |
|------|---------|
| `kernels_core.cuh` | Convert to dynamic shared memory, add size computation |
| `kernel_launchers.cuh` | Add `cudaFuncSetAttribute` calls, pass smem_size to kernel |
| `cuda_splatting.cu` | Add BATCH=256 instantiations for DIM=5,6 |
| `gsplat_model_cuda.py` | Query extended smem capability, update batch selection |

## Shared Memory Requirements

### BATCH=256 Shared Memory Usage

| DIM | CENTER_STRIDE | CONIC_SIZE | MAX_TILE_PIXELS | Total smem |
|-----|---------------|------------|-----------------|------------|
| 2 | 2 | 3 | 256 | ~8 KB |
| 3 | 4 | 6 | 512 | ~28 KB |
| 4 | 4 | 10 | 256 | ~33 KB |
| 5 | 5 | 15 | 243 | ~51 KB |
| 6 | 6 | 21 | 729 | ~70 KB |

### Expected Performance Impact

With BATCH=256 on Hopper (vs BATCH=128):
- ~2x fewer batch iterations
- ~1.3-1.5x fewer memory transactions
- Expected 10-30% speedup for backward pass

## Implementation Order

1. **Step 1:** Add `compute_backward_smem_size()` function to `kernels_core.cuh`
2. **Step 2:** Convert backward kernel to dynamic shared memory (keep forward kernel unchanged for now)
3. **Step 3:** Update `kernel_launchers.cuh` to:
   - Pass smem_size to kernel launch
   - Call `cudaFuncSetAttribute` for kernels needing >48KB
4. **Step 4:** Add BATCH=256 instantiations for DIM=5,6 in `cuda_splatting.cu`
5. **Step 5:** Update `gsplat_model_cuda.py` to query and use extended smem
6. **Step 6:** Build, test, and benchmark

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Performance regression from dynamic smem | Profile before/after; revert if needed |
| Compile-time errors from constexpr complexity | Test all template instantiations |
| Runtime errors on older GPUs | Guard with SM version check |
| Alignment issues with manual offsets | Use 16-byte alignment for all arrays |

## Verification

### Build and Test
```bash
make clean-cuda && make build-cuda
make test-cuda  # All 164+ tests should pass
```

### Benchmark on Modern GPU
```bash
# If you have access to A100/H100:
make benchmark-cuda

# Compare BATCH=128 vs BATCH=256 for 3D 512³ 100K splats
```

### Key Metrics
- Backward pass time for DIM=3, N=100K splats
- No accuracy regression (test suite validates correctness)

## Alternative Considered: Keep Static Shared Memory

We considered keeping static shared memory and simply adding more template instantiations, but:
1. Static shared memory **cannot exceed 48KB** regardless of `cudaFuncSetAttribute`
2. The 48KB limit is hardcoded in the CUDA compiler for static allocations
3. Only dynamic shared memory supports the opt-in extended limit

Therefore, converting to dynamic shared memory is **required** to use extended shared memory.
