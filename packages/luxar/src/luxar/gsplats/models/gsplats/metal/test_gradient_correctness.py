#!/usr/bin/env python
"""
CRITICAL: Test gradient correctness in detail.

If gradients are wrong, optimization will move splats to wrong locations.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("CRITICAL GRADIENT VALIDATION")
print("=" * 80)

# Simple test case: ONE splat, move it and check gradient direction
shape = (32, 32, 32)

# Place splat ABOVE center - gradient should push it DOWN (toward higher values)
center_z, center_y, center_x = 20, 16, 16  # Off-center in Z
centers = np.array([[center_z, center_y, center_x]], dtype=np.float32)
L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

# Create target: peak at center of volume
target = torch.zeros(shape)
target[16, 16, 16] = 1.0  # Peak at center

print(f"\nSetup:")
print(f"  Splat at: [{center_z}, {center_y}, {center_x}] (numpy [Z,Y,X])")
print(f"  Target at: [16, 16, 16]")
print(f"  Expected: Gradient should pull splat TOWARD [16,16,16]")

# Test Metal model
model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], device='mps'
)

# Test CPU model (reference)
model_cpu = GaussianSplatModel(
    shape=shape, centers0=centers, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], device='cpu'
)

# Forward and backward
output_metal = model_metal()
loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

# Get center gradients
grad_metal = None
grad_cpu = None

for name, param in model_metal.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        grad_metal = param.grad.cpu().numpy()
        break

for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        grad_cpu = param.grad.cpu().numpy()
        break

print(f"\n" + "=" * 80)
print("CENTER GRADIENTS")
print("=" * 80)

if grad_metal is not None and grad_cpu is not None:
    print(f"\nMetal gradient (raw_mu): {grad_metal[0]}")
    print(f"CPU gradient (raw_mu):   {grad_cpu[0]}")
    print(f"Difference: {np.abs(grad_metal[0] - grad_cpu[0])}")

    # Check signs
    print(f"\nGradient signs (should match):")
    print(f"  Metal: [{np.sign(grad_metal[0][0]):.0f}, {np.sign(grad_metal[0][1]):.0f}, {np.sign(grad_metal[0][2]):.0f}]")
    print(f"  CPU:   [{np.sign(grad_cpu[0][0]):.0f}, {np.sign(grad_cpu[0][1]):.0f}, {np.sign(grad_cpu[0][2]):.0f}]")

    # Check if signs match
    signs_match = np.allclose(np.sign(grad_metal), np.sign(grad_cpu))
    magnitudes_match = np.allclose(grad_metal, grad_cpu, rtol=0.1)

    print(f"\n" + "=" * 80)
    if signs_match and magnitudes_match:
        print("✓ GRADIENTS CORRECT - signs and magnitudes match!")
    elif signs_match:
        print("⚠️  Signs correct but magnitudes differ")
        print(f"   Max ratio: {np.max(np.abs(grad_metal / (grad_cpu + 1e-10))):.2f}")
    else:
        print("🚨 CRITICAL BUG: Gradient signs are WRONG!")
        print("   This will cause optimization to move splats in wrong direction!")

        # Check which dimensions are wrong
        for i, name in enumerate(['Z', 'Y', 'X']):
            if np.sign(grad_metal[0][i]) != np.sign(grad_cpu[0][i]):
                print(f"   ✗ Dimension {name} has WRONG sign!")
else:
    print("✗ Could not extract gradients")

# Also check L gradients
print(f"\n" + "=" * 80)
print("CHECKING L GRADIENTS")
print("=" * 80)

grad_L_metal = None
grad_L_cpu = None

for name, param in model_metal.named_parameters():
    if 'L_off' in name and param.grad is not None:
        grad_L_metal = param.grad.cpu().numpy()
        break

for name, param in model_cpu.named_parameters():
    if 'L_off' in name and param.grad is not None:
        grad_L_cpu = param.grad.cpu().numpy()
        break

if grad_L_metal is not None and grad_L_cpu is not None:
    print(f"Metal L gradient: {grad_L_metal[0]}")
    print(f"CPU L gradient:   {grad_L_cpu[0]}")
    print(f"Signs match: {np.allclose(np.sign(grad_L_metal), np.sign(grad_L_cpu))}")
