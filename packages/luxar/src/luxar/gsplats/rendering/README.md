# luxar.gsplats.rendering

GPU-accelerated volume rendering of Gaussian splats, with automatic backend selection (CUDA > MPS > CPU).

## Key Functions

- **`render_to_volume(gsplat_data, shape, ...)`** — Render Gaussian splats to a NumPy volume array. This is the primary entry point for quality comparison and visualization.
- **`render_to_volume_tensor(gsplat_data, shape, ...)`** — Same as above but returns a `torch.Tensor` on the rendering device, avoiding a GPU-to-CPU copy when the result feeds into further GPU operations (e.g., metric computation).
- **`auto_detect_device()`** — Select the best available device (CUDA > MPS > CPU).

## Module Structure

| File | Description |
|------|-------------|
| `volume_rendering.py` | Volume rendering implementation with CUDA fast path and PyTorch fallback |

## Backend Selection

1. **CUDA splatting backend** (fastest) — Used when `device="cuda"` and the compiled CUDA extension is available. Uses tiled, memory-efficient rendering.
2. **Pure PyTorch** (fallback) — Works on any device (CUDA, MPS, CPU). Supports nD rendering with automatic chunking to prevent OOM.

## Usage

```python
from luxar.gsplats.rendering import render_to_volume, render_to_volume_tensor
from luxar.gsplats.gsplat_data import GSplatData

gsplats = GSplatData.load("splats.gsplats.zarr")

# Render to NumPy array (auto-selects best device)
volume = render_to_volume(gsplats, shape=(128, 128, 128))

# Render to GPU tensor (avoids CPU copy)
tensor = render_to_volume_tensor(gsplats, shape=(128, 128, 128), device="cuda")

# Compare reconstruction quality
from luxar.gsplats.rendering import render_to_volume
rendered = render_to_volume(gsplats, shape=original.shape)
mse = ((rendered - original) ** 2).mean()
```

## Parameters

- **`truncate`** (default 3.0) — Truncation radius in standard deviations. Gaussians are evaluated within this radius from their centers.
- **`intensity_floor`** (default 1e-5) — Minimum intensity threshold for amplitude-aware culling.
- **`chunk_size`** — Optional chunk size for memory management on large volumes.
