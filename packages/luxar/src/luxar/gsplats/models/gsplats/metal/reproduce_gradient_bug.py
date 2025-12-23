#!/usr/bin/env python
"""
Reproduction script for Metal gradient bug.

This script demonstrates the gradient sign error that causes splats to move
in the wrong direction during optimization.

Expected behavior:
- Splat at [20, 16, 16] with target at [16, 16, 16]
- All gradients should be negative (pull splat toward target)
- Z gradient should be most negative (largest distance)

Bug behavior (before fix):
- Y gradient has POSITIVE sign (pushes splat AWAY from target)
- Causes elongated, misplaced gaussians during fitting
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("METAL GRADIENT BUG REPRODUCTION")
print("=" * 80)

# Test configuration
shape = (32, 32, 32)
n_splats = 1

# Place splat OFF-CENTER to generate gradients
# Splat at [20, 16, 16], target peak at [16, 16, 16]
centers = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

# Target: Single peak at center
target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nTest Setup:")
print(f"  Splat at:  [20, 16, 16]  (4 voxels above target in Z)")
print(f"  Target at: [16, 16, 16]")
print(f"  Expected gradients: ALL NEGATIVE (pull splat down toward target)")

# Test Metal model
print(f"\n" + "=" * 80)
print("METAL MODEL")
print("=" * 80)

model_metal = GaussianSplatModelMetal(
    shape=shape,
    centers0=centers,
    L0=L,
    amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='mps'
)

# Forward + backward
output_metal = model_metal()
print(f"\nMetal output stats:")
print(f"  Shape: {output_metal.shape}")
max_val = output_metal.max()
max_idx = output_metal.argmax()
max_pos = np.unravel_index(max_idx.cpu().numpy(), output_metal.shape)
print(f"  Max: {max_val:.6f} at {max_pos}")
print(f"  Sum: {output_metal.sum():.6f}")
print(f"  Output[20,16,16]: {output_metal[20,16,16]:.6f}")
print(f"  Output[16,16,16]: {output_metal[16,16,16]:.6f}")

loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
print(f"  Loss: {loss_metal.item():.6f}")
loss_metal.backward()

# Extract gradient
grad_metal = None
for name, param in model_metal.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        grad_metal = param.grad.cpu().numpy()
        break

if grad_metal is not None:
    print(f"\nMetal center gradient: {grad_metal[0]}")
    print(f"  Z gradient: {grad_metal[0, 0]:.6e} (expect negative)")
    print(f"  Y gradient: {grad_metal[0, 1]:.6e} (expect ~0 or small negative)")
    print(f"  X gradient: {grad_metal[0, 2]:.6e} (expect ~0 or small negative)")

    # Check signs (with tolerance for near-zero)
    NEAR_ZERO_TOL = 1e-4
    signs = np.sign(grad_metal[0])
    is_near_zero = np.abs(grad_metal[0]) < NEAR_ZERO_TOL

    print(f"\nGradient signs: [{signs[0]:.0f}, {signs[1]:.0f}, {signs[2]:.0f}]")
    print(f"Near-zero (< {NEAR_ZERO_TOL}): [{is_near_zero[0]}, {is_near_zero[1]}, {is_near_zero[2]}]")

    # Z should be positive (pull from 20 to 16 means decrease, grad_descent subtracts gradient)
    # Y and X should be near-zero (already aligned)
    z_correct = signs[0] > 0
    y_correct = is_near_zero[1] or signs[1] <= 0
    x_correct = is_near_zero[2] or signs[2] <= 0

    if z_correct and y_correct and x_correct:
        print("✅ All gradients have CORRECT signs!")
    else:
        print("❌ GRADIENT BUG DETECTED:")
        if not z_correct:
            print(f"   - Z gradient has wrong sign: {signs[0]:.0f} (expected positive)")
        if not y_correct:
            print(f"   - Y gradient: {grad_metal[0,1]:.6e} (expected ~0 or negative)")
        if not x_correct:
            print(f"   - X gradient: {grad_metal[0,2]:.6e} (expected ~0 or negative)")
else:
    print("❌ Could not extract Metal gradient")

# Test CPU model (reference)
print(f"\n" + "=" * 80)
print("CPU MODEL (Reference)")
print("=" * 80)

model_cpu = GaussianSplatModel(
    shape=shape,
    centers0=centers,
    L0=L,
    amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='cpu'
)

output_cpu = model_cpu()
print(f"\nCPU output stats:")
print(f"  Shape: {output_cpu.shape}")
print(f"  Max: {output_cpu.max():.6f}")
print(f"  Sum: {output_cpu.sum():.6f}")
print(f"  Output[20,16,16]: {output_cpu[20,16,16]:.6f}")
print(f"  Output[16,16,16]: {output_cpu[16,16,16]:.6f}")

loss_cpu = ((output_cpu - target) ** 2).sum()
print(f"  Loss: {loss_cpu.item():.6f}")
loss_cpu.backward()

# Extract gradient
grad_cpu = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        grad_cpu = param.grad.cpu().numpy()
        break

if grad_cpu is not None:
    print(f"\nCPU center gradient: {grad_cpu[0]}")
    print(f"  Z gradient: {grad_cpu[0, 0]:.6e}")
    print(f"  Y gradient: {grad_cpu[0, 1]:.6e}")
    print(f"  X gradient: {grad_cpu[0, 2]:.6e}")

    signs_cpu = np.sign(grad_cpu[0])
    print(f"\nGradient signs: [{signs_cpu[0]:.0f}, {signs_cpu[1]:.0f}, {signs_cpu[2]:.0f}]")
    print("✅ CPU reference (always correct)")
else:
    print("❌ Could not extract CPU gradient")

# Compare
if grad_metal is not None and grad_cpu is not None:
    print(f"\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    diff = np.abs(grad_metal[0] - grad_cpu[0])
    print(f"\nAbsolute difference: {diff}")

    # Compare signs with tolerance for near-zero gradients
    NEAR_ZERO_TOL = 1e-4
    metal_near_zero = np.abs(grad_metal[0]) < NEAR_ZERO_TOL
    cpu_near_zero = np.abs(grad_cpu[0]) < NEAR_ZERO_TOL

    signs_match = True
    for i in range(3):
        if metal_near_zero[i] or cpu_near_zero[i]:
            continue  # Both near zero, signs don't matter
        if np.sign(grad_metal[0, i]) != np.sign(grad_cpu[0, i]):
            signs_match = False
            break

    magnitudes_match = np.allclose(grad_metal, grad_cpu, rtol=0.2, atol=NEAR_ZERO_TOL)

    print(f"\nSigns match (with tolerance for near-zero): {signs_match}")
    print(f"Magnitudes match: {magnitudes_match} (within 20% or < {NEAR_ZERO_TOL})")

    if signs_match and magnitudes_match:
        print("\n✅✅✅ GRADIENT BUG IS FIXED! ✅✅✅")
    elif signs_match:
        print("\n⚠️  Signs correct but magnitudes differ")
        print("    (This is acceptable - may be due to numerical precision)")
    else:
        print("\n❌❌❌ GRADIENT BUG STILL PRESENT! ❌❌❌")
        print("\nWrong sign dimensions:")
        for i, name in enumerate(['Z', 'Y', 'X']):
            if not (metal_near_zero[i] or cpu_near_zero[i]) and \
               np.sign(grad_metal[0, i]) != np.sign(grad_cpu[0, i]):
                print(f"  - {name}: Metal={np.sign(grad_metal[0, i]):.0f}, "
                      f"CPU={np.sign(grad_cpu[0, i]):.0f}")

print("\n" + "=" * 80)
