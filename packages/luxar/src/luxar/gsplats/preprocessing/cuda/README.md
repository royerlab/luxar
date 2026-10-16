# luxar.gsplats.preprocessing.cuda

CUDA-accelerated Non-Local Means (NLM) denoising backend. Provides custom CUDA kernels for 2D and 3D NLM that are significantly faster than the pure-PyTorch fallback. The kernels (`src/nlm_kernels.cuh`) reach the GPU as the ahead-of-time extension built by `make build-nlm-cuda` when it imports, otherwise compiled at runtime by NVRTC on first use and cached on disk (`nlm_nvrtc.py`, the same mechanism and `LUXAR_CUDA_BACKEND=aot|nvrtc|torch` override as the splatting kernels; see `models/gsplats/cuda/README.md`). Each `(ndim, patch_size, search_distance)` kernel compiles on its own the first time it is used. When neither path is usable, `nlm_cuda_available()` is `False` and the PyTorch backend is used transparently.

## Key Exports

- **`nlm_cuda_available()`** — whether a CUDA NLM kernel backend (AOT extension or NVRTC) can be used.
- **`get_nlm_backend()`** — the module providing `nlm_denoise_2d` / `nlm_denoise_3d` (the extension or `nlm_nvrtc`), or `None`.
- **`NLM_AOT_AVAILABLE`** — whether the `make build-nlm-cuda` extension imports.
- **`nlm_cuda_denoise(volume, h, patch_size, search_distance)`** — Python wrapper for the CUDA NLM kernels. Handles parameter translation (e.g., `patch_size` to `patch_half`) and validates supported parameter combinations.

## Module Structure

| File | Description |
|------|-------------|
| `__init__.py` | Backend selection (`get_nlm_backend`, `nlm_cuda_available`) |
| `nlm_nvrtc.py` | NVRTC runtime-compiled backend (port of the extension's host code) |
| `nlm_cuda_wrapper.py` | Python wrapper with parameter validation and shared memory checks |
| `build.py` | Build script using `torch.utils.cpp_extension` (multi-arch gencode) |
| `src/nlm_kernels.cuh` | The 2D and 3D NLM kernels (torch-free; shared by both backends) |
| `src/nlm_cuda.cu` | Extension host code: padding, launch dispatch |
| `src/nlm_cuda.h` | Kernel header declarations |
| `src/bindings.cpp` | PyTorch C++ extension bindings |

## Supported Parameters

- **`patch_size`**: 3 or 5 (i.e., `patch_half` = 1 or 2)
- **`search_distance`**: 5, 7, 9, 11, 13, or 15

Unsupported combinations raise `ValueError` with a message suggesting `backend='pytorch'` as a fallback. For 3D volumes with large `search_distance` (>= 9), the wrapper checks GPU shared memory limits and falls back gracefully if exceeded.

## Building (optional: the AOT extension)

```bash
# From project root
make build-nlm-cuda

# Or directly
hatch run python packages/luxar/src/luxar/gsplats/preprocessing/cuda/build.py
```

The build script auto-detects GPU architectures (sm_75 through sm_120, covering Turing → Blackwell; Blackwell archs require CUDA 12.8+), compiles with `-O3 --use_fast_math`, and copies the resulting `.so` to this directory. A `nlm_build_info.json` metadata file is written alongside for environment reproducibility.

## Usage

The CUDA backend is used automatically when available:

```python
from luxar.gsplats.preprocessing import denoise_nlm

# Auto-selects the CUDA kernels (extension, else NVRTC) when the device is CUDA
denoised = denoise_nlm(volume, h=0.04, device='cuda')
```

Direct usage (advanced):

```python
from luxar.gsplats.preprocessing.cuda import nlm_cuda_available
from luxar.gsplats.preprocessing.cuda.nlm_cuda_wrapper import nlm_cuda_denoise

if nlm_cuda_available():
    denoised = nlm_cuda_denoise(volume_tensor, h=0.04, patch_size=3, search_distance=5)
```
