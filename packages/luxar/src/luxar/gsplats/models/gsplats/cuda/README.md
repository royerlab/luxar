# CUDA Backend for Gaussian Splatting

**Status**: Implementation Complete (MVP)

This module provides GPU-accelerated Gaussian splatting for NVIDIA GPUs using custom CUDA kernels.

## Overview

The CUDA backend provides substantial speedup over CPU PyTorch for Gaussian splatting operations — often orders of magnitude, depending on GPU model and problem size. It supports:

- **2D-8D rendering**: Not just 3D, but arbitrary dimensions up to 8D
- **Splat-centric architecture**: Each CUDA block processes one splat (no tile binning)
- **3.16x training speedup**: Over the original tile-based pipeline (see `OPTIMIZATION_REPORT.md`)
- **Zero global atomics in backward**: Block-local gradient reduction
- **Optional FP16 mode**: Reduced memory bandwidth with FP16 inputs (output stays FP32)

## Architecture

```
Python Layer (GaussianSplatModelCUDA)
  torch.compile(cholesky_to_conic) fuses L→conic into 1 kernel
           │
           ▼
PyTorch C++ Extension (pybind11)
  forward_wrapper() / backward_wrapper()  [bindings.cpp]
           │
           ▼
CUDA Dispatch Layer  [cuda_splatting.cu]
  dispatch_forward_impl<InputDType>() / dispatch_backward_impl<>()
  DIM_DISPATCH macro for D=2..8 template instantiation
           │
           ▼
┌──────────────────────────────────────────────┐
│        Splat-Centric CUDA Kernels            │
│  Forward:  output.zero_() → splat_fwd        │
│            (1 kernel, N blocks, atomicAdd)    │
│  Backward: splat_bwd                          │
│            (1 kernel, N blocks, block-local)  │
│                                               │
│  No tile binning. No prefix sum. No sorting.  │
└──────────────────────────────────────────────┘
```

## Requirements

- CUDA 11.8+ (CUDA 12.x recommended)
- NVIDIA GPU with Compute Capability 7.5+ (Turing, Ampere, Ada, Hopper, Blackwell)
- PyTorch 2.2+ with CUDA support
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
hatch run pip install torch --index-url https://download.pytorch.org/whl/cu128

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
    use_fp16=False,              # Optional: use FP16 for reduced memory bandwidth
)

# Forward pass
output = model()  # Uses CUDA kernels

# Backward pass (automatic via autograd)
loss = criterion(output, target)
loss.backward()  # Gradients computed via CUDA kernels
```

## Key Optimizations

### 1. Splat-Centric Architecture

Instead of tile-based binning (preprocess -> prefix_sum -> bin -> rasterize), each
CUDA block processes exactly one splat:
1. Thread 0 loads splat data and computes the AABB
2. All 256 threads cooperatively iterate over voxels in the AABB
3. Forward: `atomicAdd` contributions to the output volume
4. Backward: block-local reduction, single write per splat (no global atomics)

This eliminates ~13 tensor allocations, 4 kernel launches, and all tile binning.

### 2. Block-Local Gradient Reduction (Backward)

Each block exclusively owns its splat, so gradients are accumulated in registers
and reduced via warp shuffles (`warp_reduce_sum()` in `reduction_utils.cuh`),
then written once to global memory. Zero global `atomicAdd` contention.

### 3. torch.compile Fusion

`torch.compile(cholesky_to_conic)` fuses the 12 kernel launches of the L->conic
forward+backward into a single fused kernel, saving ~4% of training time.

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
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/test_cuda_forward.py -v
```

## File Structure

```
cuda/
├── src/
│   ├── cuda_splatting.cu                  # Dispatch layer: entry points, template instantiations
│   ├── cuda_splatting.h                   # Public API: forward(), backward()
│   ├── bindings.cpp                       # pybind11: forward_wrapper(), backward_wrapper()
│   ├── kernels_core.cuh                   # Core kernels: splat-centric forward and backward
│   ├── kernel_launchers.cuh               # Launch wrappers for splat-centric kernels
│   ├── utils.cuh                          # Umbrella header (includes all sub-headers)
│   ├── math_utils.cuh                     # Mahalanobis distance, Gaussian intensity, shift params
│   ├── voxel_utils.cuh                    # Voxel coordinate conversion utilities
│   ├── reduction_utils.cuh                # Warp reduction, gradient helpers
│   └── dtype_traits.cuh                   # DTypeTraits for FP16/FP32 load abstraction
├── gsplat_model_cuda.py                   # Python model class (GaussianSplatModelCUDA)
├── build.py                               # Build script (torch.utils.cpp_extension)
├── setup.py                               # Optional setuptools config
├── benchmark.py                           # Performance benchmarks
├── __init__.py
├── tests/
│   ├── __init__.py                        # Package marker
│   ├── conftest.py                        # Pytest fixtures and configuration
│   ├── test_cuda_forward.py               # Forward pass correctness
│   ├── test_cuda_backward.py              # Backward pass correctness
│   ├── test_cuda_gradcheck.py             # Gradient correctness (autograd comparison)
│   ├── test_cuda_numerical.py             # Numerical precision
│   ├── test_cuda_comparison.py            # CUDA vs PyTorch reference
│   ├── test_cuda_model.py                 # Full model integration
│   ├── test_cuda_nd.py                    # nD (4D-8D) tests
│   ├── test_cuda_fp16.py                  # FP16 mode tests
│   ├── test_cuda_performance.py           # Performance benchmarks
│   └── test_cuda_kernel_safety.py         # Kernel safety/numerical-invariant regression tests
├── OPTIMIZATION_REPORT.md                 # CUDA optimization results
└── README.md                              # This file
```

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
- [x] Global splat handling (detected as side effect in splat-centric kernel)
- [x] FP16 support (true FP16 kernels with direct global memory load)
- [x] Splat-centric architecture (3.16x speedup, see OPTIMIZATION_REPORT.md)
- [x] torch.compile fusion for L->conic conversion

## Known Limitations

1. **Dimension limit**: Maximum 8 dimensions supported (template instantiation limit).
2. **`__expf` fast math precision**: ~2 ULP error for ~15% speedup. Causes center and off-diagonal L gradients to have ~59% and ~75% relative error vs PyTorch reference. Convergence verified by multi-iteration stability tests.
3. **2D 4096^2 50K inference**: Minor regression (+6%) from `atomicAdd` scatter in splat-centric forward (training time still improved by 25%).

## Mixed-Precision Training with AMP (Recommended)

The model automatically detects `torch.autocast()` context and uses FP16 kernels
for optimal training performance while maintaining FP32 master weights:

```python
# Mixed-precision training (RECOMMENDED)
model = GaussianSplatModelCUDA(..., use_fp16=False)  # FP32 params
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler('cuda')

for batch in dataloader:
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        output = model()  # Auto-uses FP16 kernels!
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**How it works**:
- Parameters stay FP32 (master weights) for stable gradient updates
- Forward/backward use FP16 kernels (2x memory bandwidth)
- GradScaler prevents gradient underflow
- Output and gradients are FP32

**Performance**: ~1.0-1.5x speedup with full training stability.

## FP16 Inference Mode

For inference with pre-trained models, `use_fp16=True` stores parameters
directly in FP16 for maximum bandwidth optimization:

```python
# FP16 inference mode (inference only!)
model = GaussianSplatModelCUDA(..., use_fp16=True)
output = model()  # Fast inference with FP16 params
```

**When to use `use_fp16=True`**:
- Inference with pre-trained models
- Evaluation/visualization (single forward passes)
- Memory-constrained deployments

**Do NOT use `use_fp16=True` for training** - parameters will overflow to inf/nan.

## FP16 Technical Details

**Architecture**:
- **Direct FP16 loading**: FP16 data loaded from global memory (2x bandwidth)
- **Fused conversion**: FP16→FP32 in shared memory via `DTypeTraits::load()`
- **FP32 computation**: All math uses FP32 for numerical stability
- **FP32 output**: Output and gradients always FP32

**Performance characteristics**:
- Speedups are workload-dependent (1.0-1.5x when memory-bound)
- Best for medium-to-large volumes (1M+ voxels)
- ~0.04% relative error vs FP32 (typical)
