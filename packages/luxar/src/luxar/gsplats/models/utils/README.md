# Model Utilities

## Purpose

Shared numerical utilities used by Gaussian splat models. These are low-level math functions that support the parameterization and solving in the model layer.

## Key Functions

### `stable_inverse_softplus(y, beta=1.0)` (`inverse_softplus.py`)

Numerically stable inverse of the softplus function using `expm1`. Used to initialize raw learnable parameters from desired activation values (e.g., converting desired diagonal values back to raw parameter space). Available in both NumPy and PyTorch variants:

- `stable_inverse_softplus(y)` -- NumPy version (CPU)
- `stable_inverse_softplus_torch(y)` -- PyTorch version (GPU-compatible, avoids CPU-GPU transfers)

### `solve_lower_triangular(L, B)` (`lt_solver.py`)

Cross-version compatible solver for lower triangular systems `L @ X = B`. Prefers `torch.linalg.solve_triangular` (PyTorch 1.9+) with fallback to `torch.triangular_solve` for older versions.

## Usage

```python
import numpy as np
from luxar.gsplats.models.utils.inverse_softplus import (
    stable_inverse_softplus,
    stable_inverse_softplus_torch,
)
from luxar.gsplats.models.utils.lt_solver import solve_lower_triangular

# NumPy: convert desired sigma values to raw parameters
raw_params = stable_inverse_softplus(desired_values - sigma_min)

# PyTorch (on GPU): same operation without CPU transfer
raw_params = stable_inverse_softplus_torch(desired_tensor - sigma_min_tensor)

# Solve L @ X = B for lower-triangular L
X = solve_lower_triangular(L, B)
```

## File Structure

```
utils/
├── __init__.py
├── inverse_softplus.py     # Stable inverse softplus (NumPy + PyTorch)
├── lt_solver.py            # Lower-triangular system solver
├── README.md               # This file
└── tests/
    ├── __init__.py
    ├── test_inverse_softplus.py
    └── test_lt_solver.py
```
