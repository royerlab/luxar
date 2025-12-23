#!/usr/bin/env python
"""
Inspect the actual conic values being passed to Metal kernel.
This will tell us if the permutation is correct or if we're sending wrong data.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

print("=" * 80)
print("INSPECT METAL CONIC INPUT")
print("=" * 80)

# Diagonal L matrix
sigma = 2.0
L = torch.eye(3) * sigma
L = L.unsqueeze(0)  # [1, 3, 3]

print(f"\nInput L (diagonal, σ={sigma}):")
print(L[0])

# Compute conic using the function
conic = cholesky_to_conic(L)

print(f"\nConic from cholesky_to_conic:")
print(f"  Raw output: {conic[0]}")
print(f"  Shape: {conic.shape}")

# Interpret as [Z,Y,X] upper triangle
c_values = conic[0].numpy()
print(f"\nInterpreted as [Z,Y,X] upper triangle:")
print(f"  [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]")
print(f"  [{c_values[0]:.4f}, {c_values[1]:.4f}, {c_values[2]:.4f}, {c_values[3]:.4f}, {c_values[4]:.4f}, {c_values[5]:.4f}]")

# Expected for diagonal with σ=2: c_zz=c_yy=c_xx=0.25, off-diagonal=0
expected = [0.25, 0.0, 0.0, 0.25, 0.0, 0.25]
print(f"\nExpected (diagonal σ=2):")
print(f"  [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]")
print(f"  [{expected[0]:.4f}, {expected[1]:.4f}, {expected[2]:.4f}, {expected[3]:.4f}, {expected[4]:.4f}, {expected[5]:.4f}]")

matches = np.allclose(c_values, expected, atol=1e-6)
print(f"\nMatches expected: {matches}")

if not matches:
    print(f"  Difference: {c_values - expected}")

# Now apply the permutation [5, 4, 2, 3, 1, 0] that gets sent to Metal
perm = [5, 4, 2, 3, 1, 0]
conic_reordered = conic[:, perm]

print(f"\n" + "=" * 80)
print(f"AFTER PERMUTATION [5, 4, 2, 3, 1, 0]")
print(f"=" * 80)

c_reordered = conic_reordered[0].numpy()
print(f"\nReordered conic (sent to Metal):")
print(f"  {c_reordered}")

# Metal expects [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
print(f"\nMetal interprets as [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]:")
print(f"  c_xx (pos 0): {c_reordered[0]:.4f}  (expect 0.25)")
print(f"  c_xy (pos 1): {c_reordered[1]:.4f}  (expect 0.00)")
print(f"  c_xz (pos 2): {c_reordered[2]:.4f}  (expect 0.00)")
print(f"  c_yy (pos 3): {c_reordered[3]:.4f}  (expect 0.25)")
print(f"  c_yz (pos 4): {c_reordered[4]:.4f}  (expect 0.00)")
print(f"  c_zz (pos 5): {c_reordered[5]:.4f}  (expect 0.25)")

# Check if any unexpected non-zero values
print(f"\nChecking for unexpected non-zero off-diagonal terms:")
off_diagonal_correct = (abs(c_reordered[1]) < 1e-6 and
                       abs(c_reordered[2]) < 1e-6 and
                       abs(c_reordered[4]) < 1e-6)

if off_diagonal_correct:
    print(f"  ✅ All off-diagonal terms are ~0 (correct)")
else:
    print(f"  ❌ Some off-diagonal terms are non-zero (WRONG!):")
    if abs(c_reordered[1]) > 1e-6:
        print(f"     c_xy = {c_reordered[1]:.6f} (should be 0)")
    if abs(c_reordered[2]) > 1e-6:
        print(f"     c_xz = {c_reordered[2]:.6f} (should be 0)")
    if abs(c_reordered[4]) > 1e-6:
        print(f"     c_yz = {c_reordered[4]:.6f} (should be 0)")

# Verify the full matrix in [X,Y,Z] order
print(f"\n" + "=" * 80)
print(f"FULL CONIC MATRIX (Metal's [X,Y,Z] order)")
print(f"=" * 80)

conic_matrix_xyz = np.array([
    [c_reordered[0], c_reordered[1], c_reordered[2]],  # [c_xx, c_xy, c_xz]
    [c_reordered[1], c_reordered[3], c_reordered[4]],  # [c_xy, c_yy, c_yz]
    [c_reordered[2], c_reordered[4], c_reordered[5]]   # [c_xz, c_yz, c_zz]
])

print(f"\nConic matrix (should be diagonal 0.25):")
print(conic_matrix_xyz)

is_diagonal = (np.abs(conic_matrix_xyz - np.diag(np.diag(conic_matrix_xyz))) < 1e-6).all()
print(f"\nIs diagonal: {is_diagonal}")

if is_diagonal:
    print(f"✅ Conic is diagonal as expected")
    print(f"   Diagonal values: [{conic_matrix_xyz[0,0]:.4f}, {conic_matrix_xyz[1,1]:.4f}, {conic_matrix_xyz[2,2]:.4f}]")
else:
    print(f"❌ Conic is NOT diagonal!")
    print(f"   Off-diagonal terms detected - permutation is WRONG")

print("\n" + "=" * 80)
