# CUDA Backend for Gaussian Splatting

**Status**: Implementation Complete (MVP)

This module provides GPU-accelerated Gaussian splatting for NVIDIA GPUs using custom CUDA kernels.

## Overview

The CUDA backend is designed to provide 10-100x speedup over CPU PyTorch for Gaussian splatting operations. It supports:

- **2D-8D rendering**: Not just 3D, but arbitrary dimensions up to 8D
- **Tile-based rasterization**: Efficient spatial binning with configurable tile sizes
- **Optimized gradient computation**: Warp-level reduction to minimize atomic operations
- **Memory efficiency**: 4x less memory than naive implementations

## Architecture

```
Python Layer (GaussianSplatModelCUDA)
           │
           ▼
PyTorch C++ Extension (pybind11)
           │
           ▼
CUDA Kernel Dispatcher
           │
           ▼
┌──────────────────────────────────────┐
│         CUDA Compute Kernels         │
│  preprocess → bin → rasterize_fwd    │
│           rasterize_bwd              │
└──────────────────────────────────────┘
```

## Requirements

- CUDA 11.8+ (CUDA 12.x recommended)
- NVIDIA GPU with Compute Capability 7.0+ (Volta, Turing, Ampere, Ada, Hopper)
- PyTorch 2.0+ with CUDA support
- CUB library (bundled with CUDA Toolkit)

## Installation

### Recommended: Using Make (from project root)

```bash
# Check dependencies and get installation guidance
make check-cuda-deps

# Full setup: install dependencies + build extension (may prompt for sudo)
make setup-cuda

# Or just build (if dependencies already installed)
make build-cuda
```

### Manual Installation

```bash
# Ensure PyTorch with CUDA is installed in hatch environment (from project root)
hatch run pip install torch --index-url https://download.pytorch.org/whl/cu121

# Build the extension (from project root)
hatch run python packages/luxar/src/luxar/gsplats/models/gsplats/cuda/build.py

# The .so file will be created in the cuda/ directory
```

## Usage

```python
from luxar.gsplats.models.gsplats.cuda import GaussianSplatModelCUDA

# Create model (same API as GaussianSplatModel)
model = GaussianSplatModelCUDA(
    shape=(128, 128, 128),       # Volume shape (supports 2D-8D)
    centers0=centers,            # Initial centers (N, d)
    L0=L,                        # Initial Cholesky factors (N, d, d)
    amps0=amps,                  # Initial amplitudes (N,)
    sigma_min_diag=(0.5, 0.5, 0.5),
    truncate=3.0,
    device='cuda',
)

# Forward pass
output = model()  # Uses CUDA kernels

# Backward pass (automatic via autograd)
loss = criterion(output, target)
loss.backward()  # Gradients computed via CUDA kernels
```

## Key Optimizations

### 1. Tile-Based Binning

Instead of checking all splats for every pixel, we:
1. Compute axis-aligned bounding boxes (AABBs) for each splat
2. Bin splats into spatial tiles based on AABB overlap
3. Each pixel only processes splats in its tile

### 2. Warp-Level Gradient Reduction

To minimize atomic operation contention:
```cuda
// Sum across 32 threads in warp
float grad_sum = warp_reduce_sum(local_grad);

// Only lane 0 writes to global memory
if (lane_id == 0) {
    atomicAdd(&grad_global[splat_id], grad_sum);
}
```

This reduces atomic operations by 32x.

### 3. Shared Memory Batch Loading

Following BalanceGS patterns:
```cuda
// Cooperative load into shared memory
__shared__ float splat_data[BATCH_SIZE];
for (int i = tid; i < batch_size; i += blockDim.x) {
    splat_data[i] = global_data[batch_start + i];
}
__syncthreads();

// Process from fast shared memory
for (int i = 0; i < batch_size; i++) {
    process(splat_data[i]);
}
```

## Performance Targets

| Scenario | Volume Size | Splats | Target Speedup |
|----------|-------------|--------|----------------|
| Small 3D | 128³ | 1K | 30× |
| Medium 3D | 256³ | 10K | 50× |
| Large 3D | 512³ | 100K | 100× |

## Testing

```bash
# Run all CUDA tests (from project root)
make test-cuda

# Run performance benchmarks
make benchmark-cuda

# Run specific test file
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/test_cuda_backend.py -v
```

## File Structure

```
cuda/
├── src/
│   ├── cuda_splatting.cu      # CUDA kernels
│   ├── cuda_splatting.h       # Header declarations
│   ├── bindings.cpp           # pybind11 bindings
│   └── utils.cuh              # Device utilities
├── gsplat_model_cuda.py       # Python model class
├── build.py                   # Build script (uses torch.utils.cpp_extension)
├── setup.py                   # Legacy setuptools config (optional)
├── benchmark.py               # Performance benchmarks
├── __init__.py
├── tests/
│   ├── test_cuda_backend.py
│   ├── test_cuda_numerical.py
│   └── test_cuda_performance.py
├── SPECIFICATIONS.md                    # Core algorithms spec
├── SPECIFICATIONS_PYTORCH_INTEGRATION.md  # PyTorch integration spec
├── SPECIFICATIONS_TESTING.md            # Testing strategy spec
├── OPTIMIZATION_ROADMAP.md              # Future optimization plans
└── README.md                            # This file
```

## Documentation

The technical specification is split into three files:

- **[SPECIFICATIONS.md](SPECIFICATIONS.md)** - Core algorithms: architecture, kernel designs, memory optimization, gradient computation
- **[SPECIFICATIONS_PYTORCH_INTEGRATION.md](SPECIFICATIONS_PYTORCH_INTEGRATION.md)** - PyTorch integration, performance targets, implementation phases, references
- **[SPECIFICATIONS_TESTING.md](SPECIFICATIONS_TESTING.md)** - Comprehensive testing strategy, unit tests, benchmarks

## References

Key implementations studied:
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Nerfstudio's CUDA rasterizer
- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) - INRIA's original
- [FlashGS](https://arxiv.org/html/2408.07967v2) - Warp divergence elimination
- [BalanceGS](https://arxiv.org/html/2510.14564) - Memory coalescing optimization

## Status

- [x] Design specification
- [x] Core infrastructure (build.py, bindings)
- [x] Forward pass kernels (2D, 3D)
- [x] Backward pass kernels
- [x] nD extension (4D-8D)
- [x] Standard Gaussian (s=2) fast path optimization
- [ ] Global splat handling (large splats currently skipped)
- [ ] Additional performance optimizations (see OPTIMIZATION_ROADMAP.md)

## Known Limitations

1. **Global splats**: Splats covering >10% of tiles are silently skipped. See OPTIMIZATION_ROADMAP.md section 3.6.
2. **Dimension limit**: Maximum 8 dimensions supported (template instantiation limit).
3. **Precision**: Uses float32 throughout; float16 not yet supported.
