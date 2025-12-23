#!/usr/bin/env python
"""
Verify the actual indexing of cholesky_to_conic output.
"""

import torch

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import cholesky_to_conic

# Create a diagonal L matrix with distinct values
# If L is in [Z,Y,X] row order:
#   Row 0 = Z, Row 1 = Y, Row 2 = X
L = torch.zeros(1, 3, 3)
L[0, 0, 0] = 1.0  # L_zz
L[0, 1, 1] = 2.0  # L_yy
L[0, 2, 2] = 3.0  # L_xx

print("Input L (diagonal):")
print(L[0])
print("\nL[0,0] (row 0, col 0) = Z,Z =", L[0, 0, 0].item())
print("L[1,1] (row 1, col 1) = Y,Y =", L[0, 1, 1].item())
print("L[2,2] (row 2, col 2) = X,X =", L[0, 2, 2].item())

# Compute Σ = L @ L^T manually
Sigma = L[0] @ L[0].T
print("\nΣ = L @ L^T:")
print(Sigma)
print("Σ[0,0] (Z,Z) =", Sigma[0, 0].item())
print("Σ[1,1] (Y,Y) =", Sigma[1, 1].item())
print("Σ[2,2] (X,X) =", Sigma[2, 2].item())

# Compute Σ^{-1} manually
Sigma_inv = torch.linalg.inv(Sigma)
print("\nΣ^{-1}:")
print(Sigma_inv)
print("Σ^{-1}[0,0] (Z,Z) =", Sigma_inv[0, 0].item())
print("Σ^{-1}[1,1] (Y,Y) =", Sigma_inv[1, 1].item())
print("Σ^{-1}[2,2] (X,X) =", Sigma_inv[2, 2].item())

# Now use cholesky_to_conic
conic = cholesky_to_conic(L)
print("\ncholesky_to_conic output:")
print(conic[0])

# Match elements
print("\nElement matching:")
print(f"conic[0] = {conic[0, 0]:.6f} should be Σ^{{-1}}[i,j]")
print(f"conic[1] = {conic[0, 1]:.6f} should be Σ^{{-1}}[i,j]")
print(f"conic[2] = {conic[0, 2]:.6f} should be Σ^{{-1}}[i,j]")
print(f"conic[3] = {conic[0, 3]:.6f} should be Σ^{{-1}}[i,j]")
print(f"conic[4] = {conic[0, 4]:.6f} should be Σ^{{-1}}[i,j]")
print(f"conic[5] = {conic[0, 5]:.6f} should be Σ^{{-1}}[i,j]")

# Try to match
if abs(conic[0, 0] - Sigma_inv[0, 0]) < 1e-6:
    print("\n✓ conic[0] = Σ^{-1}[0,0] = c_zz (Z,Z)")
elif abs(conic[0, 0] - Sigma_inv[2, 2]) < 1e-6:
    print("\n✓ conic[0] = Σ^{-1}[2,2] = c_xx (X,X)")
else:
    print(f"\n✗ conic[0] = {conic[0, 0]:.6f} doesn't match any diagonal element")

if abs(conic[0, 3] - Sigma_inv[1, 1]) < 1e-6:
    print("✓ conic[3] = Σ^{-1}[1,1] = c_yy (Y,Y)")
else:
    print(f"✗ conic[3] = {conic[0, 3]:.6f} doesn't match Σ^{{-1}}[1,1] = {Sigma_inv[1, 1]:.6f}")

if abs(conic[0, 5] - Sigma_inv[2, 2]) < 1e-6:
    print("✓ conic[5] = Σ^{-1}[2,2] = c_xx (X,X)")
elif abs(conic[0, 5] - Sigma_inv[0, 0]) < 1e-6:
    print("✓ conic[5] = Σ^{-1}[0,0] = c_zz (Z,Z)")
else:
    print(f"✗ conic[5] = {conic[0, 5]:.6f} doesn't match any diagonal element")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If conic[0] = c_zz, then order is [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]")
print("If conic[0] = c_xx, then order is [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]")
print("=" * 80)
