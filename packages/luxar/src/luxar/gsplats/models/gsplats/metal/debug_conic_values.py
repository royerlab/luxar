#!/usr/bin/env python
"""
Debug script to test conic permutation logic.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

# Simple test case: identity covariance (σ=2 in all directions)
L = torch.eye(3) * 2.0
L = L.unsqueeze(0)  # [1, 3, 3]

print("=" * 80)
print("CONIC VALUE DEBUG")
print("=" * 80)

print(f"\nInput L matrix ([Z,Y,X] order):")
print(L[0])

# Compute conic manually
L_tensor = L
conic = cholesky_to_conic(L_tensor)

print(f"\nConic (PyTorch [Z,Y,X] order):")
print(f"  [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]")
print(f"  {conic[0].numpy()}")

# Apply permutation [2, 3, 4, 1, 5, 0]
conic_permuted = conic[:, [2, 3, 4, 1, 5, 0]]

print(f"\nPermuted conic (Metal [X,Y,Z] order):")
print(f"  [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]")
print(f"  {conic_permuted[0].numpy()}")

# The conic matrix in full form (PyTorch [Z,Y,X])
c = conic[0].numpy()
conic_matrix_zyx = np.array([
    [c[0], c[5], c[4]],  # [c_zz, c_yz, c_xz]
    [c[5], c[1], c[3]],  # [c_yz, c_yy, c_xy]
    [c[4], c[3], c[2]]   # [c_xz, c_xy, c_xx]
])

print(f"\nFull conic matrix ([Z,Y,X] order):")
print(conic_matrix_zyx)

# The conic matrix expected by Metal ([X,Y,Z])
c_perm = conic_permuted[0].numpy()
conic_matrix_xyz = np.array([
    [c_perm[0], c_perm[1], c_perm[2]],  # [c_xx, c_xy, c_xz]
    [c_perm[1], c_perm[3], c_perm[4]],  # [c_xy, c_yy, c_yz]
    [c_perm[2], c_perm[4], c_perm[5]]   # [c_xz, c_yz, c_zz]
])

print(f"\nFull conic matrix ([X,Y,Z] order - Metal expects):")
print(conic_matrix_xyz)

# Test distance computation
test_point_zyx = np.array([16.0, 16.0, 16.0])  # Target location in [Z,Y,X]
center_zyx = np.array([20.0, 16.0, 16.0])  # Splat location in [Z,Y,X]
d_zyx = test_point_zyx - center_zyx  # [-4, 0, 0]

dist_sq_zyx = d_zyx @ conic_matrix_zyx @ d_zyx
print(f"\nDistance² computation ([Z,Y,X]):")
print(f"  d = {d_zyx}")
print(f"  D² = {dist_sq_zyx:.6f}")

# Same computation but in [X,Y,Z] order
test_point_xyz = np.array([16.0, 16.0, 20.0])  # Target in [X,Y,Z]
center_xyz = np.array([16.0, 16.0, 20.0])  # Splat in [X,Y,Z]
d_xyz = test_point_xyz - center_xyz  # [0, 0, 0]

dist_sq_xyz = d_xyz @ conic_matrix_xyz @ d_xyz
print(f"\nDistance² computation ([X,Y,Z]):")
print(f"  d = {d_xyz}")
print(f"  D² = {dist_sq_xyz:.6f}")

print("\n" + "=" * 80)
print("If permutation is correct, both distance computations should give same result")
print("for corresponding points.")
print("=" * 80)
