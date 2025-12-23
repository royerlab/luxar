#!/usr/bin/env python
"""
Test with DIAGONAL conic only (no off-diagonal terms) to simplify debugging.

With diagonal conic: c_xy = c_xz = c_yz = 0
The gradient formula simplifies to:
  ∂D²/∂z = 2 * c_zz * dz
  ∂D²/∂y = 2 * c_yy * dy
  ∂D²/∂x = 2 * c_xx * dx

This makes it much easier to verify correctness.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("DIAGONAL CONIC GRADIENT TEST")
print("=" * 80)

# Use DIAGONAL covariance (identity scaled)
# This gives diagonal conic with c_zz = c_yy = c_xx = 1/σ²
shape = (32, 32, 32)
sigma = 2.0
L = np.eye(3) * sigma  # Diagonal L
L = L[np.newaxis, :, :]  # Add batch dimension

# Test case: splat offset from target in Z only
splat_center = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
target_center = [16, 16, 16]

amps = np.array([1.0], dtype=np.float32)

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nTest Configuration:")
print(f"  L (diagonal): σ = {sigma}")
print(f"  Splat center: {splat_center[0]}")
print(f"  Target center: {target_center}")
print(f"  Expected: Z gradient should dominate (dz = 4, dy = dx = 0)")

# Compute expected conic
Sigma = L[0] @ L[0].T
Sigma_inv = np.linalg.inv(Sigma)
print(f"\nΣ^(-1) (conic):")
print(Sigma_inv)
print(f"  Diagonal: [{Sigma_inv[0,0]:.4f}, {Sigma_inv[1,1]:.4f}, {Sigma_inv[2,2]:.4f}]")
print(f"  Off-diagonal: all should be ~0")

# Metal model
print(f"\n" + "=" * 80)
print("METAL MODEL")
print("=" * 80)

model_metal = GaussianSplatModelMetal(
    shape=shape,
    centers0=splat_center.copy(),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='mps'
)

output_metal = model_metal()
loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

# Extract raw_mu gradient
metal_grad = None
for name, param in model_metal.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        metal_grad = param.grad.detach().cpu().numpy()[0]
        break

print(f"\nMetal raw_mu gradient: {metal_grad}")
print(f"  Norm: {np.linalg.norm(metal_grad):.6e}")

# CPU model
print(f"\n" + "=" * 80)
print("CPU MODEL (Reference)")
print("=" * 80)

model_cpu = GaussianSplatModel(
    shape=shape,
    centers0=splat_center.copy(),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='cpu'
)

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

# Extract raw_mu gradient
cpu_grad = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        cpu_grad = param.grad.detach().cpu().numpy()[0]
        break

print(f"\nCPU raw_mu gradient: {cpu_grad}")
print(f"  Norm: {np.linalg.norm(cpu_grad):.6e}")

# Compare
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = metal_grad - cpu_grad
print(f"\nDifference (Metal - CPU): {diff}")
print(f"  Absolute: {np.abs(diff)}")
print(f"  Relative: {np.abs(diff / (cpu_grad + 1e-10))}")

# Check signs
signs_match = np.sign(metal_grad) == np.sign(cpu_grad)
print(f"\nSigns match: {signs_match}")
for i, dim in enumerate(['Z', 'Y', 'X']):
    if not signs_match[i]:
        print(f"  ✗ {dim}: Metal={np.sign(metal_grad[i]):.0f}, CPU={np.sign(cpu_grad[i]):.0f}")

# With diagonal conic, we can verify the gradient magnitude
# Expected: gradient should be proportional to distance offset
# dz = 20 - 16 = 4, dy = 0, dx = 0
# So Z gradient should dominate
print(f"\nExpected behavior (diagonal conic):")
print(f"  Z gradient should be LARGEST (dz = 4)")
print(f"  Y gradient should be ~0 (dy = 0)")
print(f"  X gradient should be ~0 (dx = 0)")

metal_abs = np.abs(metal_grad)
cpu_abs = np.abs(cpu_grad)

print(f"\nMetal absolute: {metal_abs}")
print(f"  Z dominates: {metal_abs[0] > 10 * metal_abs[1] and metal_abs[0] > 10 * metal_abs[2]}")

print(f"\nCPU absolute: {cpu_abs}")
print(f"  Z dominates: {cpu_abs[0] > 10 * cpu_abs[1] and cpu_abs[0] > 10 * cpu_abs[2]}")

if np.allclose(metal_grad, cpu_grad, rtol=0.01):
    print(f"\n✅ Gradients MATCH for diagonal conic")
else:
    print(f"\n❌ Gradients DIFFER even with diagonal conic!")
    print(f"   This narrows down the bug location.")

print("\n" + "=" * 80)
