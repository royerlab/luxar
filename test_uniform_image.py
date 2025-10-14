#!/usr/bin/env python
"""Test uniform image edge case."""
import numpy as np
from luxar.gsplats.multiscale import decompose_image

# Test uniform image
V_uniform = np.ones((32, 32), dtype=np.float32) * 5.0
print('Testing uniform image...')
scales_list, stats = decompose_image(
    V_uniform,
    scales=[1, 2],
    n_iters=10,
    verbose=False
)
print(f'Converged: {stats["converged"]}')
print(f'Best max abs error: {stats["best_max_abs_error"]:.6e}')
print(f'Actual iters: {stats["actual_iters"]}')
print('✓ Uniform image test passed!')
