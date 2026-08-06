# Gaussian Splat Models

## Purpose

PyTorch model implementations and rendering engines for n-dimensional oriented Gaussian splatting. This package provides the core model class, rendering engine, hardware-accelerated backends, and numerical utilities.

## Package Structure

```
models/
├── __init__.py
├── README.md               # This file
├── gsplats/                 # Core model and rendering
│   ├── gsplat_model.py      # GaussianSplatModel (nn.Module)
│   ├── rendering_core.py    # Rendering engine (2D/3D fast paths, nD generic)
│   ├── rendering_wrappers.py # NumPy/PyTorch wrappers
│   ├── cuda/                # CUDA backend (NVIDIA GPUs)
│   └── metal/               # Metal backend (Apple Silicon)
└── utils/                   # Numerical utilities
    ├── inverse_softplus.py  # Stable inverse softplus (NumPy + PyTorch)
    └── lt_solver.py         # Lower-triangular system solver
```

## Public API (short imports)

All key symbols are re-exported from `luxar.gsplats.models` for convenience:

```python
from luxar.gsplats.models import (
    # Core model
    GaussianSplatModel,
    # Rendering
    render_gaussians,
    render_gaussians_numpy,
    render_gaussians_pytorch,
    # Numerical utilities
    stable_inverse_softplus,
    stable_inverse_softplus_torch,
    solve_lower_triangular,
)
```

The longer subpackage paths (`luxar.gsplats.models.gsplats`, `luxar.gsplats.models.utils.*`) still work and additionally expose internal helpers.

## Key Components

### GaussianSplatModel (`gsplats/gsplat_model.py`)

PyTorch `nn.Module` for optimizable Gaussian splats. Each splat is parameterized by:

- **Center position**: Sigmoid-constrained to stay within the image domain
- **Covariance matrix**: Cholesky decomposition `L` where Sigma = L @ L^T, with configurable min/max diagonal and eccentricity constraints
- **Amplitude**: Non-negative via softplus activation, with optional max clamping

Key methods:

| Method | Description |
|--------|-------------|
| `forward()` | Render all splats to a volume |
| `current_params()` | Returns `(centers, L, amps)` tensors |
| `prune_(keep_mask)` | Remove splats by boolean mask |
| `append_(centers, Ls, amps)` | Add new splats |
| `replace_with(centers, Ls, amps)` | Replace all splats |
| `n_splats()` | Return current splat count |

### Rendering Engine (`gsplats/rendering_core.py`)

High-performance renderer with specialized 2D/3D fast paths:

- `render_gaussians(shape, centers, Ls, amps, truncate, intensity_floor)` -- main entry point
- 2D/3D explicit forward substitution (10-50x faster than generic nD solver)
- Automatic memory-aware chunking
- AABB truncation with amplitude-aware shrinking

### Rendering Wrappers (`gsplats/rendering_wrappers.py`)

User-friendly wrappers that accept `GSplatData` objects:

- `render_gaussians_numpy()` -- CPU NumPy output (no gradients)
- `render_gaussians_pytorch()` -- PyTorch tensor output

### Accelerated Backends

- **CUDA** (`gsplats/cuda/`): NVIDIA GPU acceleration (2D-8D), substantial speedup (often orders of magnitude, GPU-dependent). See `gsplats/cuda/README.md`.
- **Metal** (`gsplats/metal/`): Apple Silicon acceleration (3D only), substantial speedup (chip-dependent). See `gsplats/metal/README.md`.

### Numerical Utilities (`utils/`)

- `stable_inverse_softplus()` -- Numerically stable inverse of softplus (NumPy and PyTorch variants)
- `solve_lower_triangular()` -- Cross-version compatible triangular solver

## Quick Start

```python
from luxar.gsplats.models import GaussianSplatModel, render_gaussians
import numpy as np

# Initialize parameters
N, d = 100, 3
centers0 = np.random.rand(N, d) * 50 + 7
L0 = np.zeros((N, d, d))
L0[:, 0, 0] = L0[:, 1, 1] = L0[:, 2, 2] = 3.0
amps0 = np.ones(N) * 0.3

# Create model
model = GaussianSplatModel(
    shape=(64, 64, 64),
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.8, 0.8, 0.8],
)

# Render
volume = model()  # Returns torch.Tensor of shape (64, 64, 64)

# Get current parameters
centers, Ls, amps = model.current_params()
```

## Optimization Example

```python
import torch.optim as optim

model = GaussianSplatModel(
    shape=(64, 64),
    centers0=centers0,
    L0=L0,
    amps0=amps0,
    sigma_min_diag=[0.5, 0.5],
)

target = torch.tensor(target_image, dtype=torch.float32, device=model.raw_mu.device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

for i in range(1000):
    optimizer.zero_grad()
    rendered = model()
    loss = torch.nn.functional.mse_loss(rendered, target)
    loss.backward()
    optimizer.step()
```

## Parameterization Details

1. **Centers**: `centers = sigmoid(raw_mu) * (shape - 1)` -- keeps centers within bounds
2. **Cholesky diagonal**: `L_diag = softplus(raw_L_diag) + sigma_min` -- ensures positive definiteness
3. **Off-diagonal**: `L_off` -- unconstrained (optionally bounded by eccentricity constraint)
4. **Amplitudes**: `amps = softplus(raw_a)` -- ensures non-negativity (optionally clamped by `amp_max`)

## Testing

```bash
# Run all model tests
hatch run pytest packages/luxar/src/luxar/gsplats/models/ -v

# Run specific test file
hatch run pytest packages/luxar/src/luxar/gsplats/models/gsplats/tests/test_gsplat_model.py -v
```

## References

- [Main Gaussian Splats Package](../README.md) -- Higher-level fitting API
- [Utils Package](../utils/README.md) -- Matrix utilities (pack/unpack)
- [Optimization Package](../optim/README.md) -- Optimizer integration
