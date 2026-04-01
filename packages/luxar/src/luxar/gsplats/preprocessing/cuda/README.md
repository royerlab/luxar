# luxar.gsplats.preprocessing.cuda

CUDA-accelerated Non-Local Means (NLM) denoising backend. Provides custom CUDA kernels for 2D and 3D NLM that are significantly faster than the pure-PyTorch fallback. When the extension is not compiled, `NLM_CUDA_AVAILABLE` is `False` and the PyTorch backend is used transparently.

## Key Exports

- **`NLM_CUDA_AVAILABLE`** — Boolean flag indicating whether the compiled CUDA extension is loaded.
- **`nlm_cuda_denoise(volume, h, patch_size, search_distance)`** — Python wrapper for the CUDA NLM kernels. Handles parameter translation (e.g., `patch_size` to `patch_half`) and validates supported parameter combinations.

## Module Structure

| File | Description |
|------|-------------|
| `__init__.py` | Extension loading and `NLM_CUDA_AVAILABLE` flag |
| `nlm_cuda_wrapper.py` | Python wrapper with parameter validation and shared memory checks |
| `build.py` | Build script using `torch.utils.cpp_extension` (multi-arch gencode) |
| `src/nlm_cuda.cu` | CUDA kernel implementations for 2D and 3D NLM |
| `src/nlm_cuda.h` | Kernel header declarations |
| `src/bindings.cpp` | PyTorch C++ extension bindings |

## Supported Parameters

- **`patch_size`**: 3 or 5 (i.e., `patch_half` = 1 or 2)
- **`search_distance`**: 5, 7, 9, 11, 13, or 15

Unsupported combinations raise `ValueError` with a message suggesting `backend='pytorch'` as a fallback. For 3D volumes with large `search_distance` (>= 9), the wrapper checks GPU shared memory limits and falls back gracefully if exceeded.

## Building

```bash
# From project root
make build-nlm-cuda

# Or directly
hatch run python packages/luxar/src/luxar/gsplats/preprocessing/cuda/build.py
```

The build script auto-detects GPU architectures (sm_70+), compiles with `-O3 --use_fast_math`, and copies the resulting `.so` to this directory. A `nlm_build_info.json` metadata file is written alongside for environment reproducibility.

## Usage

The CUDA backend is used automatically when available:

```python
from luxar.gsplats.preprocessing import denoise_nlm

# Auto-selects CUDA backend if extension is compiled and device is CUDA
denoised = denoise_nlm(volume, h=0.04, device='cuda')
```

Direct usage (advanced):

```python
from luxar.gsplats.preprocessing.cuda import NLM_CUDA_AVAILABLE
from luxar.gsplats.preprocessing.cuda.nlm_cuda_wrapper import nlm_cuda_denoise

if NLM_CUDA_AVAILABLE:
    denoised = nlm_cuda_denoise(volume_tensor, h=0.04, patch_size=3, search_distance=5)
```
