# luxar.gsplats.preprocessing

GPU-accelerated volume preprocessing for Gaussian splat fitting, with a focus on Non-Local Means (NLM) denoising. Noisy data wastes splats on background artifacts; denoising first leads to more efficient, higher-quality fits.

## Key Functions

- **`denoise_nlm(volume, h, ...)`** — Non-Local Means denoising for 2D images and 3D volumes with automatic backend dispatch (CUDA > PyTorch > skimage).
- **`calibrate_nlm_h(volume, ...)`** — Find optimal filtering strength `h` via Noise2Self (J-invariant) cross-validation, requiring no clean reference image.
- **`denoise_volume_array(volume, h, ...)`** — High-level pipeline: normalize to [0,1], denoise, denormalize back. Supports 2D slice-by-slice and 3D modes with auto-chunking for large volumes.
- **`calibrate_all_channels(input_path, ...)`** — Calibrate `h` across multiple channels by sampling equidistant timepoints and taking the median.
- **`calibrate_h_for_channel(input_path, channel, ...)`** — Calibrate `h` for a single channel from sampled timepoints.
- **`normalize_volume()` / `denormalize_volume()`** — Consistent [0,1] normalization with reversibility.

## Module Structure

| File | Description |
|------|-------------|
| `nlm_core.py` | Public `denoise_nlm()` dispatcher with backend resolution and validation |
| `nlm_pytorch.py` | Pure-PyTorch NLM implementation with 2D/3D support and optional chunked processing |
| `calibration.py` | Noise2Self J-invariant calibration: grid search over `h` using masked cross-validation |
| `denoise_pipeline.py` | High-level pipeline: normalization, multi-channel calibration, volume denoising |
| `cuda/` | CUDA-accelerated NLM backend (see `cuda/README.md`) |
| `tests/` | Test suite covering all backends, calibration, and the denoise pipeline |

## Backends

Three backends are available, selected automatically or via the `backend` parameter:

| Backend | Device | Speed | Notes |
|---------|--------|-------|-------|
| `cuda` | CUDA GPU | Fastest | Requires compiled extension (`make build-nlm-cuda`) |
| `pytorch` | Any (CUDA, MPS, CPU) | Medium | Pure PyTorch, supports chunked 3D processing |
| `skimage` | CPU | Slowest | scikit-image reference, always available |

Auto-selection (`backend="auto"`): CUDA kernel if available on CUDA device, PyTorch on GPU devices, skimage on CPU.

## Usage

```python
import torch
from luxar.gsplats.preprocessing import denoise_nlm, calibrate_nlm_h

volume = torch.from_numpy(noisy_volume)

# Auto-calibrate denoising strength (no clean reference needed)
h = calibrate_nlm_h(volume, device='cuda')

# Denoise with best available backend
denoised = denoise_nlm(volume, h=h, device='cuda')

# High-level pipeline (handles normalization automatically)
from luxar.gsplats.preprocessing import denoise_volume_array
denoised_np = denoise_volume_array(noisy_np, h=0.04, backend='auto')
```

## Testing

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/preprocessing/tests/ -v
```
