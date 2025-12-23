#!/usr/bin/env python
"""
Test to verify conic permutation is correct.

This checks that the permutation [2, 3, 4, 1, 5, 0] correctly maps
PyTorch [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] to
Metal [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz].
"""

import numpy as np
import torch

# Define a known conic matrix in [Z,Y,X] coordinates
# Let's use a diagonal matrix with distinct values so we can track them
c_zz, c_yy, c_xx = 1.0, 2.0, 3.0
c_xy, c_xz, c_yz = 4.0, 5.0, 6.0

# PyTorch conic storage order [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]
pytorch_conic = torch.tensor([[c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]], dtype=torch.float32)

print("=" * 80)
print("CONIC PERMUTATION TEST")
print("=" * 80)

print(f"\nOriginal conic (PyTorch [Z,Y,X] order):")
print(f"  [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]")
print(f"  [{c_zz}, {c_yy}, {c_xx}, {c_xy}, {c_xz}, {c_yz}]")
print(f"  Tensor: {pytorch_conic}")

# Expected Metal order [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
expected_metal = torch.tensor([[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]], dtype=torch.float32)

print(f"\nExpected Metal order [X,Y,Z]:")
print(f"  [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]")
print(f"  [{c_xx}, {c_xy}, {c_xz}, {c_yy}, {c_yz}, {c_zz}]")
print(f"  Tensor: {expected_metal}")

# Apply our permutation [2, 3, 4, 1, 5, 0]
permuted_conic = pytorch_conic[:, [2, 3, 4, 1, 5, 0]]

print(f"\nActual permuted conic:")
print(f"  Permutation: [2, 3, 4, 1, 5, 0]")
print(f"  Tensor: {permuted_conic}")

# Check if they match
matches = torch.allclose(permuted_conic, expected_metal)

print(f"\n" + "=" * 80)
if matches:
    print("✅ PERMUTATION IS CORRECT!")
    print("   PyTorch → Metal mapping works as expected")
else:
    print("❌ PERMUTATION IS WRONG!")
    print(f"   Expected: {expected_metal}")
    print(f"   Got:      {permuted_conic}")
    print(f"   Difference: {expected_metal - permuted_conic}")

# Also test the inverse permutation [5, 3, 0, 1, 2, 4]
print(f"\n" + "=" * 80)
print("TESTING INVERSE PERMUTATION")
print("=" * 80)

# Apply inverse to get back to PyTorch order
inverse_permuted = permuted_conic[:, [5, 3, 0, 1, 2, 4]]

print(f"\nInverse permutation: [5, 3, 0, 1, 2, 4]")
print(f"Result: {inverse_permuted}")
print(f"Original: {pytorch_conic}")

inverse_matches = torch.allclose(inverse_permuted, pytorch_conic)

if inverse_matches:
    print("✅ INVERSE PERMUTATION IS CORRECT!")
else:
    print("❌ INVERSE PERMUTATION IS WRONG!")
    print(f"   Difference: {pytorch_conic - inverse_permuted}")

print("\n" + "=" * 80)
