# CUDA GSplat 3D Optimization Roadmap

This document lists all available optimizations for the CUDA Gaussian Splatting backend,
ranked by a composite score considering:

---

## Implementation Status

| ID | Optimization | Status | Notes |
|----|-------------|--------|-------|
| 1.1 | Remove Unnecessary atomicAdd | ✅ Complete | Direct write in forward pass |
| 1.2 | Precompute effective_truncate_sq | ✅ Complete | s_truncate_sq + early rejection |
| 1.3 | Hardcoded 3D Mahalanobis Distance | ✅ Complete | Template specialization |
| 1.4 | Bitwise Tile-Local Pixel Indexing | ✅ Complete | 8×8×8 and 16×16 fast paths |
| 1.5 | Read-Only Cache (__ldg) | ✅ Complete | Used for global reads |
| 1.6 | Fast Math Intrinsics | ✅ Complete | __expf, __powf, __logf |
| 2.1 | 3D Grid Launch Configuration | ✅ Complete | dim3 grid for 2D/3D |
| 2.2 | Explicit 3D Gradient Formulas | ✅ Complete | backward_pixel_splat_3d/2d |
| 2.3 | Shared Memory Bank Conflict Avoidance | ✅ Complete | CENTER_STRIDE=4 for 3D |
| 2.4 | Vectorized Memory Loads (float4) | ⏳ Pending | Would require input padding |
| 2.5 | Standard Gaussian Fast Path (s=2) | ❌ Skipped | User prefers sharpness |
| 2.6 | Cache shape/tile_dims Device Tensors | ✅ Complete | BinningState caching |
| 2.7 | Persist CUB Scan Temporary Storage | ✅ Complete | BinningState.scan_temp_storage |

---

- **Impact**: Expected performance improvement
- **Simplicity**: Implementation effort and code complexity
- **Certainty**: Confidence in the performance gain (based on GPU architecture principles)
- **Risk**: Likelihood of introducing bugs or regressions

Optimizations are grouped into tiers. **Implement Tier 1 first** — these provide the best
return on investment.

---

## Tier 1: High Impact, Simple, Certain ✅

These optimizations are low-hanging fruit with predictable, significant gains.

### 1.1 Remove Unnecessary `atomicAdd` in Forward Pass

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐⭐ (15-25%) |
| Simplicity | ⭐⭐⭐⭐⭐ (1 line change) |
| Certainty | ⭐⭐⭐⭐⭐ (guaranteed) |
| Risk | ⭐ (very low) |

**Current code** (`cuda_splatting.cu:493`):
```cuda
atomicAdd(&output[global_px_idx], intensity_sum);
```

**Optimized**:
```cuda
output[global_px_idx] = intensity_sum;  // Direct write
```

**Why it works**: Each pixel belongs to exactly one tile. Tiles are disjoint partitions,
so no two blocks ever write to the same pixel. `atomicAdd` has 10-50 cycle latency;
direct write is 1 cycle.

**Caveat**: This assumes global splats (which could contribute from multiple tiles) are
handled separately. Currently global splats are dropped, so this is safe.

---

### 1.2 Precompute `effective_truncate_sq` Per-Splat

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (10-15%) |
| Simplicity | ⭐⭐⭐⭐⭐ (add to batch load) |
| Certainty | ⭐⭐⭐⭐⭐ (eliminates powf from hot loop) |
| Risk | ⭐ (very low) |

**Current code** (computed per-pixel, per-splat):
```cuda
float effective_truncate_radius_sq = powf(truncate, 4.0f / s);
```

**Optimized** (compute once during batch load):
```cuda
// In shared memory load:
s_truncate_sq[tid] = __powf(truncate, 4.0f / s_sharpness[tid]);

// In inner loop:
if (dist_sq <= s_truncate_sq[i]) { ... }  // Just a load!
```

**Why it works**: `powf()` is ~25-50 cycles. Each splat is evaluated against all 512
pixels in a tile. Moving this to batch load amortizes the cost 512×.

---

### 1.3 Hardcoded 3D Mahalanobis Distance

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐⭐ (20-30%) |
| Simplicity | ⭐⭐⭐⭐ (template specialization) |
| Certainty | ⭐⭐⭐⭐⭐ (eliminates loops, enables FMA) |
| Risk | ⭐ (low, just math) |

**Current code** (generic loop):
```cuda
float result = 0.0f;
int idx = 0;
for (int i = 0; i < DIM; i++) {
    result += d[i] * d[i] * conic[idx++];
    for (int j = i + 1; j < DIM; j++) {
        result += 2.0f * d[i] * d[j] * conic[idx++];
    }
}
```

**Optimized 3D** (explicit formula):
```cuda
template<>
__device__ __forceinline__ float mahalanobis_distance_sq<3>(
    const float* d, const float* c
) {
    // c = [c00, c01, c02, c11, c12, c22]
    return c[0]*d[0]*d[0] + c[3]*d[1]*d[1] + c[5]*d[2]*d[2]
         + 2.0f * (c[1]*d[0]*d[1] + c[2]*d[0]*d[2] + c[4]*d[1]*d[2]);
}
```

**Why it works**:
- Eliminates loop overhead (branch, counter increment, bounds check)
- Compiler can fuse into 9 FMA instructions
- Reduces register pressure (no loop variables)

---

### 1.4 Bitwise Tile-Local Pixel Indexing (Power-of-2 Tiles)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (10-15%) |
| Simplicity | ⭐⭐⭐⭐ (simple bit ops) |
| Certainty | ⭐⭐⭐⭐⭐ (bitwise >> division) |
| Risk | ⭐ (low) |

**Current code** (integer division):
```cuda
int remaining = local_px_idx;
for (int d = DIM - 1; d >= 0; d--) {
    voxel_coords[d] = tile_origin[d] + (remaining % tile_extent[d]);
    remaining /= tile_extent[d];
}
```

**Optimized 3D** (8×8×8 tile):
```cuda
// threadIdx.x = 0..511 for 8×8×8 tile
int local_z = threadIdx.x & 0x7;           // % 8 via AND
int local_y = (threadIdx.x >> 3) & 0x7;    // / 8 % 8
int local_x = threadIdx.x >> 6;            // / 64
```

**Why it works**: Bitwise AND/shift are single-cycle instructions. Integer division
is 20-40 cycles on GPU.

---

### 1.5 `__ldg()` for Global Memory Reads

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (5-15%) |
| Simplicity | ⭐⭐⭐⭐⭐ (wrapper change) |
| Certainty | ⭐⭐⭐⭐ (texture cache path) |
| Risk | ⭐ (very low) |

**Current code**:
```cuda
centers_smem[tid] = centers[splat_id * 3 + d];
```

**Optimized**:
```cuda
centers_smem[tid] = __ldg(&centers[splat_id * 3 + d]);
```

**Why it works**: `__ldg` uses the read-only texture cache, which has:
- Better bandwidth for broadcast patterns (same splat read by many threads)
- Separate cache from L1, reducing cache thrashing

---

### 1.6 Use Fast Math Intrinsics

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (5-10%) |
| Simplicity | ⭐⭐⭐⭐⭐ (function rename) |
| Certainty | ⭐⭐⭐⭐ (documented speedup) |
| Risk | ⭐⭐ (slightly reduced precision) |

**Current code**:
```cuda
float intensity = amplitude * expf(-0.5f * dist_sq);
float dist_pow = powf(dist_sq, s * 0.5f);
```

**Optimized**:
```cuda
float intensity = amplitude * __expf(-0.5f * dist_sq);
float dist_pow = __powf(dist_sq, s * 0.5f);
```

**Why it works**: Fast intrinsics (`__expf`, `__powf`, `__logf`) trade precision
(~2 ULP error vs ~1 ULP) for speed (~2× faster).

---

## Tier 2: High Impact, Medium Complexity ⚡

These require more code changes but provide substantial gains.

### 2.1 3D Grid Launch Configuration

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (15-25%) |
| Simplicity | ⭐⭐⭐ (refactor launch + indexing) |
| Certainty | ⭐⭐⭐⭐ (cache locality) |
| Risk | ⭐⭐ (medium - coordinate mapping) |

**Current code**:
```cuda
// 1D grid, manual tile coordinate extraction
<<<num_tiles, 512>>>
int tile_idx = blockIdx.x;
linear_to_tile_coords<DIM>(tile_idx, tile_dims, tile_coords);
```

**Optimized**:
```cuda
// 3D grid matching volume topology
dim3 grid(tile_dims[2], tile_dims[1], tile_dims[0]);
dim3 block(512);  // Or (8, 8, 8) for better coalescing

// Direct indexing - no arithmetic
int tile_x = blockIdx.x;
int tile_y = blockIdx.y;
int tile_z = blockIdx.z;
```

**Why it works**:
- GPU L2 cache and memory prefetcher exploit spatial locality
- Adjacent blocks in 3D grid access adjacent memory regions
- Eliminates division/modulo operations

---

### 2.2 Explicit 3D Gradient Formulas (Backward Pass)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐⭐ (25-35%) |
| Simplicity | ⭐⭐⭐ (template specialization) |
| Certainty | ⭐⭐⭐⭐⭐ (eliminates nested loops) |
| Risk | ⭐⭐ (must match forward exactly) |

**Current code** (nested loops for ∂D²/∂d):
```cuda
for (int i = 0; i < DIM; i++) {
    float sum = 0.0f;
    int row_start = 0;
    for (int k = 0; k < i; k++) { row_start += DIM - k; }
    sum += conic[row_start] * d[i];
    // ... more iterations for off-diagonals
    dD2_dd[i] = 2.0f * sum;
}
```

**Optimized 3D**:
```cuda
// ∂D²/∂d = 2 × Σ⁻¹ × d (matrix-vector multiply)
// conic = [c00, c01, c02, c11, c12, c22]
dD2_dd[0] = 2.0f * (c[0]*d[0] + c[1]*d[1] + c[2]*d[2]);
dD2_dd[1] = 2.0f * (c[1]*d[0] + c[3]*d[1] + c[4]*d[2]);
dD2_dd[2] = 2.0f * (c[2]*d[0] + c[4]*d[1] + c[5]*d[2]);
```

**Why it works**: Backward pass is compute-bound. Eliminating nested loops with
explicit formulas reduces instruction count by ~70%.

---

### 2.3 Shared Memory Bank Conflict Avoidance

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (5-15%) |
| Simplicity | ⭐⭐⭐ (layout restructure) |
| Certainty | ⭐⭐⭐⭐ (bank conflict elimination) |
| Risk | ⭐⭐ (must update all access patterns) |

**Current layout** (stride-3 causes conflicts):
```cuda
__shared__ float s_centers[BATCH_SIZE * 3];  // s_centers[i * 3 + d]
```

**Optimized options**:

Option A: Pad to power-of-2 stride:
```cuda
__shared__ float s_centers[BATCH_SIZE * 4];  // Pad to 4
// Access: s_centers[i * 4 + d] (wastes 25% but no conflicts)
```

Option B: Separate arrays:
```cuda
__shared__ float s_centers_x[BATCH_SIZE];
__shared__ float s_centers_y[BATCH_SIZE];
__shared__ float s_centers_z[BATCH_SIZE];
```

**Why it works**: Shared memory has 32 banks. Stride-3 access means threads 0,11,22
hit the same bank, serializing access. Power-of-2 strides avoid this.

---

### 2.4 Vectorized Memory Loads (float4)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (10-20%) |
| Simplicity | ⭐⭐⭐ (requires padding) |
| Certainty | ⭐⭐⭐⭐ (documented bandwidth gain) |
| Risk | ⭐⭐ (alignment requirements) |

**Current code** (3 separate loads):
```cuda
s_centers[tid * 3 + 0] = centers[splat_id * 3 + 0];
s_centers[tid * 3 + 1] = centers[splat_id * 3 + 1];
s_centers[tid * 3 + 2] = centers[splat_id * 3 + 2];
```

**Optimized** (pad to float4, single 16-byte load):
```cuda
// Requires centers stored as (N, 4) with padding
float4 c = __ldg(reinterpret_cast<const float4*>(&centers[splat_id * 4]));
s_centers_x[tid] = c.x;
s_centers_y[tid] = c.y;
s_centers_z[tid] = c.z;
```

**Trade-off**: +33% memory footprint, but 2-4× better memory bandwidth utilization.

---

### 2.5 Standard Gaussian Fast Path (s=2)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (10-20% when s=2) |
| Simplicity | ⭐⭐⭐⭐ (branch on sharpness) |
| Certainty | ⭐⭐⭐⭐ (eliminates powf) |
| Risk | ⭐⭐ (branch divergence possible) |

**Current code**:
```cuda
float dist_pow_s = powf(dist_sq, sharpness * 0.5f);
float intensity = amplitude * expf(-0.5f * dist_pow_s);
```

**Optimized**:
```cuda
float intensity;
if (sharpness == 2.0f) {
    // Standard Gaussian: D^s = D² when s=2
    intensity = amplitude * __expf(-0.5f * dist_sq);
} else {
    intensity = amplitude * __expf(-0.5f * __powf(dist_sq, sharpness * 0.5f));
}
```

**Why it works**: When s=2 (standard Gaussian), `dist_sq^(s/2) = dist_sq^1 = dist_sq`.
No `powf` needed. Many fitting workflows use s=2 by default.

**Alternative**: Precompute a flag `is_standard_gaussian` per batch.

---

### 2.6 Cache shape/tile_dims Device Tensors

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (100-500µs per iteration) |
| Simplicity | ⭐⭐⭐⭐ (add to state struct) |
| Certainty | ⭐⭐⭐⭐⭐ (eliminates allocation) |
| Risk | ⭐ (very low) |

**Current code** (allocates every forward pass):
```cpp
auto shape_tensor = torch::tensor(std::vector<int>(shape.begin(), shape.end()),
    torch::TensorOptions().dtype(torch::kInt32).device(device));
```

**Optimized**:
```cpp
// In BinningState initialization (once):
state.shape_tensor = torch::tensor(...).to(device);
state.tile_dims_tensor = torch::tensor(...).to(device);

// In forward (reuse):
// Just use state.shape_tensor directly
```

**Why it works**: `cudaMalloc` is ~100-500µs per call. For 1000 iterations of fitting,
this saves 0.1-0.5 seconds of pure overhead.

---

### 2.7 Persist CUB Scan Temporary Storage

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (100-500µs per iteration) |
| Simplicity | ⭐⭐⭐⭐ (allocate once) |
| Certainty | ⭐⭐⭐⭐⭐ (documented overhead) |
| Risk | ⭐ (very low) |

**Current code** (queries + allocates each pass):
```cpp
size_t temp_bytes = 0;
cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, ...);  // Query
state.scan_temp_storage = torch::empty({temp_bytes}, ...);  // Allocate
```

**Optimized**:
```cpp
// In initialization:
size_t max_temp_bytes = compute_max_temp_bytes(max_tiles);
state.scan_temp_storage = torch::empty({max_temp_bytes}, ...);

// In forward:
// Just use pre-allocated buffer
```

---

## Tier 3: Medium Impact, Higher Complexity 🔧

These provide good gains but require more careful implementation.

### 3.1 Double-Buffered Async Prefetching (Ampere+)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (20-30%) |
| Simplicity | ⭐⭐ (async primitives) |
| Certainty | ⭐⭐⭐⭐ (hides memory latency) |
| Risk | ⭐⭐⭐ (synchronization bugs) |

**Current code** (synchronous):
```cuda
// Load batch N
for (int i = threadIdx.x; i < batch_count; i += blockDim.x) {
    s_data[i] = global_data[offset + i];
}
__syncthreads();

// Process batch N
for (int i = 0; i < batch_count; i++) {
    process(s_data[i]);
}
__syncthreads();
```

**Optimized** (overlap load N+1 with processing N):
```cuda
__shared__ float s_data[2][BATCH_SIZE * DATA_SIZE];  // Double buffer
cuda::pipeline<cuda::thread_scope_block> pipe = ...;

for (int batch = 0; batch < num_batches; batch++) {
    int curr = batch % 2;
    int next = (batch + 1) % 2;

    // Start async load of NEXT batch
    if (batch + 1 < num_batches) {
        cuda::memcpy_async(&s_data[next], &global[batch+1], size, pipe);
    }
    pipe.producer_commit();

    // Process CURRENT batch (while next loads)
    for (int i = 0; i < batch_count; i++) {
        process(s_data[curr][i]);
    }

    pipe.consumer_wait();  // Ensure next batch ready
}
```

**Why it works**: Memory latency is ~400 cycles. Processing a batch takes ~200-400 cycles.
Overlapping hides most of the memory stall.

**Requires**: CUDA 11+, Compute Capability 8.0+ (Ampere)

---

### 3.2 Per-Tile Gradient Accumulation (Backward)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (20-30%) |
| Simplicity | ⭐⭐ (restructure backward) |
| Certainty | ⭐⭐⭐⭐ (reduces global atomics) |
| Risk | ⭐⭐⭐ (gradient correctness) |

**Current code** (warp reduce + global atomic per warp):
```cuda
// Each warp does a reduction, then atomic to global
float warp_sum = warp_reduce_sum(local_grad);
if (lane == 0) {
    atomicAdd(&d_centers[splat_id * 3 + d], warp_sum);  // 16 atomics per splat per tile
}
```

**Optimized** (tile-local accumulation):
```cuda
// Per-splat-in-batch gradient accumulators in shared memory
__shared__ float s_grad_centers[BATCH_SIZE * 3];
__shared__ float s_grad_conic[BATCH_SIZE * 6];

// Initialize to zero
for (int i = threadIdx.x; i < BATCH_SIZE * 3; i += blockDim.x) {
    s_grad_centers[i] = 0.0f;
}
__syncthreads();

// Accumulate locally (shared memory atomics are faster)
atomicAdd(&s_grad_centers[local_splat_idx * 3 + d], local_grad);
__syncthreads();

// Single global write per splat per tile
if (threadIdx.x < batch_count) {
    int splat_id = s_splat_ids[threadIdx.x];
    atomicAdd(&d_centers[splat_id * 3 + 0], s_grad_centers[threadIdx.x * 3 + 0]);
    atomicAdd(&d_centers[splat_id * 3 + 1], s_grad_centers[threadIdx.x * 3 + 1]);
    atomicAdd(&d_centers[splat_id * 3 + 2], s_grad_centers[threadIdx.x * 3 + 2]);
}
```

**Why it works**: Reduces global atomics from (512 pixels / 32 warp × splats) to
(1 × splats) per tile. Shared memory atomics are 10× faster than global.

---

### 3.3 Warp-Level Early Termination

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (5-15%, data-dependent) |
| Simplicity | ⭐⭐⭐ (voting primitives) |
| Certainty | ⭐⭐⭐ (depends on splat distribution) |
| Risk | ⭐⭐ (edge cases) |

**Current code**:
```cuda
if (dist_sq <= truncate_sq && intensity >= floor) {
    intensity_sum += intensity;
}
```

**Optimized**:
```cuda
bool contributes = (dist_sq <= truncate_sq);
unsigned int warp_mask = __ballot_sync(0xFFFFFFFF, contributes);

if (warp_mask == 0) {
    continue;  // Skip splat entirely for this warp
}

// Only compute expensive exp() if at least one thread needs it
if (contributes) {
    float intensity = amp * __expf(-0.5f * dist_sq);
    if (intensity >= floor) {
        intensity_sum += intensity;
    }
}
```

**Why it works**: If a splat is far from all 32 pixels in a warp, we skip the
expensive `expf()` for all of them.

---

### 3.4 Persistent Kernel Approach

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (10-20%) |
| Simplicity | ⭐⭐ (kernel restructure) |
| Certainty | ⭐⭐⭐ (reduces launch overhead) |
| Risk | ⭐⭐⭐ (workload balancing) |

**Current code**:
```cuda
// Many short-lived blocks
rasterize_kernel<<<num_tiles, 512>>>();
```

**Optimized**:
```cuda
// Fewer long-lived blocks that process multiple tiles
__global__ void rasterize_persistent(...) {
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        process_tile(tile);
        __syncthreads();  // Ensure tile complete before next
    }
}

// Launch fewer blocks (e.g., 2× number of SMs)
int num_sms;
cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
rasterize_persistent<<<num_sms * 2, 512>>>();
```

**Why it works**:
- Reduces kernel launch overhead (~5-10µs per launch)
- Blocks stay resident, maintaining warm L1/shared memory
- Better occupancy control

---

### 3.5 Fuse Preprocess + Bin Kernels

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (10-15%) |
| Simplicity | ⭐⭐ (major restructure) |
| Certainty | ⭐⭐⭐ (eliminates redundant AABB computation) |
| Risk | ⭐⭐⭐ (prefix sum dependency) |

**Current approach**: Two separate kernels both compute AABBs:
1. `preprocess_kernel`: Compute AABB, increment tile_counts
2. `bin_kernel`: Compute AABB again, write to tile_content

**Optimized approach**: Single kernel with two phases:

```cuda
__global__ void preprocess_and_bin(...) {
    // Phase 1: Compute AABB, count tiles
    AABB aabb = compute_aabb(...);
    int n_tiles = aabb.num_tiles();

    // Block-level reduction to compute local prefix sum
    __shared__ int s_counts[BLOCK_SIZE];
    s_counts[threadIdx.x] = n_tiles;
    __syncthreads();

    // Intra-block prefix sum
    int local_offset = block_prefix_sum(s_counts, threadIdx.x);

    // Global atomic to get block's base offset
    __shared__ int block_base;
    if (threadIdx.x == 0) {
        block_base = atomicAdd(&global_write_head, s_counts[BLOCK_SIZE - 1]);
    }
    __syncthreads();

    // Phase 2: Write to tile_content at known offset
    iterate_tiles(aabb, [&](int tile_idx, int local_tile) {
        tile_content[block_base + local_offset + local_tile] = splat_idx;
        // Need to also record which tile...
    });
}
```

**Challenge**: This requires restructuring the binning to not use per-tile offsets.
May need a different data structure (e.g., global list with tile tags).

---

### 3.6 Implement Global Splat Rendering ✅ COMPLETE

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐⭐ (correctness fix) |
| Simplicity | ⭐⭐ (new kernel) |
| Certainty | ⭐⭐⭐⭐⭐ (implemented) |
| Risk | ⭐⭐ (must integrate properly) |

**Status**: ✅ IMPLEMENTED in `dispatch_forward` lines 1920-1971 and dedicated kernels
`rasterize_forward_global_kernel` (lines 1189-1341) and `rasterize_backward_global_kernel`
(lines 1347-1489).

**Implementation**: Separate forward and backward kernels handle global splats:
- `rasterize_forward_global_kernel` (lines 1189-1341): Processes all global splats per pixel
- `rasterize_backward_global_kernel` (lines 1347-1489): Gradient computation for global splats
- `dispatch_forward` (lines 1920-1971): Detects global splats and routes to appropriate kernel

---

## Tier 4: Architecture-Specific (Ampere/Hopper) 🚀

These require specific GPU generations but provide significant additional gains.

### 4.1 Thread Block Clusters (Hopper SM 9.0+)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (15-25%) |
| Simplicity | ⭐ (new paradigm) |
| Certainty | ⭐⭐⭐ (depends on splat overlap) |
| Risk | ⭐⭐⭐⭐ (Hopper-only) |

Adjacent tiles often process the same splats. With clusters, they can share data
via distributed shared memory:

```cuda
__global__ __cluster_dims__(2, 2, 2)
void rasterize_clustered(...) {
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    // This block's shared memory
    __shared__ float my_splats[BATCH_SIZE * 11];

    // Access neighbor block's shared memory directly (no global round-trip!)
    float* neighbor_splats = cluster.map_shared_rank(my_splats, neighbor_rank);
}
```

**When useful**: Large splats spanning 2×2×2 tile regions can be loaded once
and shared among 8 blocks.

---

### 4.2 Tensor Memory Accelerator (Hopper SM 9.0+)

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (10-20%) |
| Simplicity | ⭐ (PTX-level) |
| Certainty | ⭐⭐⭐ (bulk transfer optimization) |
| Risk | ⭐⭐⭐⭐ (Hopper-only, complex) |

TMA provides hardware-accelerated async copy for structured data:

```cuda
#if __CUDA_ARCH__ >= 900
// Setup tensor map (once at init)
CUtensorMap tensor_map;
cuTensorMapEncode(&tensor_map, ...);

// In kernel: bulk copy without using threads
asm volatile (
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3, %4}];"
    :: "r"(smem_addr), "l"(tensor_map), "r"(x), "r"(y), "r"(z)
);
#endif
```

---

### 4.3 Runtime Architecture Dispatch

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (cumulative of above) |
| Simplicity | ⭐⭐ (multiple code paths) |
| Certainty | ⭐⭐⭐⭐ (best of each arch) |
| Risk | ⭐⭐⭐ (maintenance burden) |

Single binary optimized for multiple architectures:

```cpp
void dispatch_rasterize(int dim, /* args */) {
    int sm = get_sm_version();

    if (sm >= 90) {
        // Hopper: clusters + TMA + async
        rasterize_clustered_3d<<<...>>>(...);
    } else if (sm >= 80) {
        // Ampere: async pipeline + warp specialization
        rasterize_pipelined_3d<<<...>>>(...);
    } else if (sm >= 70) {
        // Volta/Turing: standard optimized
        rasterize_optimized_3d<<<...>>>(...);
    } else {
        // Fallback: generic
        rasterize_generic<3><<<...>>>(...);
    }
}
```

---

## Tier 5: Speculative / Situational ⚠️

These may help in specific scenarios but have uncertain or limited benefit.

### 5.1 Visibility Cache for Iterative Optimization

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐⭐ (20% amortized) |
| Simplicity | ⭐ (cache invalidation logic) |
| Certainty | ⭐⭐ (depends on convergence rate) |
| Risk | ⭐⭐⭐⭐ (stale cache = wrong results) |

When splats move slowly between iterations, cache tile assignments:

```cpp
struct VisibilityCache {
    torch::Tensor tile_splat_lists;  // Per-tile splat IDs
    torch::Tensor splat_positions_hash;  // Hash of last positions
    int last_valid_iteration;
};

bool is_cache_valid(const torch::Tensor& current_positions) {
    // Check if positions changed by more than threshold
    auto diff = (current_positions - cached_positions).abs().max();
    return diff.item<float>() < position_threshold;
}
```

**When useful**: Late-stage fitting with small learning rate, interactive editing.

**Risk**: Incorrect cache invalidation causes silent rendering errors.

---

### 5.2 Half-Precision (FP16) Computation

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐⭐ (up to 2× on Tensor Cores) |
| Simplicity | ⭐⭐ (mixed precision handling) |
| Certainty | ⭐⭐ (precision-sensitive for gradients) |
| Risk | ⭐⭐⭐⭐ (numerical stability) |

```cuda
// Use half precision for distance computation
__half2 d_half = __floats2half2_rn(d[0], d[1]);
// ... compute in FP16 ...
float result = __half2float(result_half);
```

**Challenge**: Gradient computation is sensitive to precision. May need to keep
gradients in FP32 while using FP16 for forward.

---

### 5.3 Splat Sorting by Tile Locality

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐ (5-10%) |
| Simplicity | ⭐ (preprocessing step) |
| Certainty | ⭐⭐ (depends on initial ordering) |
| Risk | ⭐⭐⭐ (index mapping overhead) |

Reorder splats by Morton code for better cache locality:

```python
def sort_splats_by_morton(centers, L, amps, sharpness, shape):
    morton_codes = compute_morton_codes(centers, shape)
    sorted_indices = torch.argsort(morton_codes)
    return (
        centers[sorted_indices],
        L[sorted_indices],
        amps[sorted_indices],
        sharpness[sorted_indices],
        sorted_indices  # For gradient remapping
    )
```

**Trade-off**: Adds sorting overhead, requires index remapping for gradients.

---

### 5.4 Cooperative Groups for Block-Wide Operations

| Metric | Rating |
|--------|--------|
| Impact | ⭐⭐ (5-10%) |
| Simplicity | ⭐⭐ (API learning curve) |
| Certainty | ⭐⭐⭐ (cleaner reductions) |
| Risk | ⭐⭐ (may not outperform manual) |

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void rasterize(...) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp-level reduction using cooperative groups
    float warp_sum = cg::reduce(warp, local_value, cg::plus<float>());
}
```

---

## Summary: Implementation Priority

### Phase 1: Quick Wins (1-2 days, ~50% speedup)
1. ✅ Remove atomicAdd in forward (1.1) - **DONE**
2. ✅ Precompute effective_truncate_sq (1.2) - **DONE** (utils.cuh: `effective_truncate_sq()`)
3. ✅ Hardcoded 3D Mahalanobis (1.3) - **DONE** (utils.cuh: template specialization)
4. ✅ Bitwise tile indexing (1.4) - **DONE** (utils.cuh: `local_idx_to_coords_3d_fast()`)
5. ✅ `__ldg` intrinsics (1.5) - **DONE** (cuda_splatting.cu: batch loading with `__ldg()`)
6. ✅ Fast math intrinsics (1.6) - **DONE** (utils.cuh: `__expf`, `__powf`, `__logf`)

### Phase 2: Core Optimizations (3-5 days, additional ~40% speedup)
7. ✅ 3D grid launch (2.1) - **DONE** (cuda_splatting.cu: `dim3` grid for 2D/3D with `get_tile_info_3d`)
8. ✅ Explicit 3D backward gradients (2.2) - **DONE** (cuda_splatting.cu: `compute_pixel_gradients<3>`)
9. ✅ Bank conflict avoidance (2.3) - **DONE** (CENTER_STRIDE=4 for 3D)
10. ✅ Cache device tensors (2.6) - **DONE** (BinningState: shape_tensor, tile_dims_tensor)
11. ✅ Persist CUB storage (2.7) - **DONE** (BinningState: scan_temp_storage)

### Phase 3: Advanced (1-2 weeks, additional ~30% speedup)
12. 🔧 Double-buffered prefetch (3.1) - Ampere+
13. 🔧 Per-tile gradient accumulation (3.2) - In progress
14. ✅ Global splat rendering fix (3.6) - **DONE** (dispatch_forward lines 1920-1971)
15. 🔧 Fuse preprocess+bin kernels (3.5) - In progress
16. 🔧 Vectorized FP16 loads - In progress

### Phase 4: Architecture-Specific (ongoing)
15. 🚀 Hopper optimizations (4.1, 4.2)
16. 🚀 Multi-arch dispatch (4.3)

---

## Expected Cumulative Speedup

| After Phase | Forward | Backward | Total |
|-------------|---------|----------|-------|
| Baseline | 1.0× | 1.0× | 1.0× |
| Phase 1 | 1.8× | 1.6× | 1.7× |
| Phase 2 | 2.8× | 3.2× | 3.0× |
| Phase 3 | 4.0× | 5.5× | 4.8× |
| Phase 4 (Hopper) | 5.0× | 7.0× | 6.0× |

For a typical 256³ volume with 100K splats:
- **Baseline**: ~60ms/iteration
- **After all optimizations**: ~10-12ms/iteration
- **On Hopper with all optimizations**: ~8-10ms/iteration