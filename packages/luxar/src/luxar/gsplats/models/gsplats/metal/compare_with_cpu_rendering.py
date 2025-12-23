#!/usr/bin/env python
"""
Compare Metal d_centers with CPU rendering gradients DIRECTLY.

This bypasses our Python reference and uses the actual CPU implementation.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import MetalSplatFunction, cholesky_to_conic
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

print("=" * 80)
print("METAL VS CPU RENDERING GRADIENT COMPARISON")
print("=" * 80)

# Test case
shape = (32, 32, 32)
centers_np = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
L_np = (np.eye(3) * 2.0)[np.newaxis, :, :].astype(np.float32)
amps_np = np.array([1.0], dtype=np.float32)
sharpness_val = 2.0

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

# ===============================
# CPU rendering with autograd
# ===============================
print(f"\nCPU Rendering:")

centers_cpu = torch.tensor(centers_np, requires_grad=True)
L_cpu = torch.tensor(L_np, requires_grad=True)
amps_cpu = torch.tensor(amps_np, requires_grad=True)
sharpness_cpu = torch.ones(1, requires_grad=True) * sharpness_val

output_cpu = render_gaussians(
    shape, centers_cpu, L_cpu, amps_cpu, sharpness_cpu,
    truncate=3.0, intensity_floor=1e-5
)

print(f"  Output sum: {output_cpu.sum().item():.6f}")
print(f"  Output max: {output_cpu.max().item():.6f}")

loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

cpu_d_centers = centers_cpu.grad.cpu().numpy()[0]
cpu_d_L = L_cpu.grad.cpu().numpy()[0]

print(f"  CPU d_centers: {cpu_d_centers}")
print(f"  CPU d_L diagonal: [{cpu_d_L[0,0]:.3e}, {cpu_d_L[1,1]:.3e}, {cpu_d_L[2,2]:.3e}]")

# ===============================
# Metal rendering
# ===============================
print(f"\nMetal Rendering:")

centers_metal = torch.tensor(centers_np, requires_grad=True, device='mps')
L_metal = torch.tensor(L_np, requires_grad=True, device='mps')
amps_metal = torch.tensor(amps_np, requires_grad=True, device='mps')
sharpness_metal = torch.ones(1, requires_grad=True, device='mps') * sharpness_val

# Call MetalSplatFunction which returns d_centers directly
output_metal = MetalSplatFunction.apply(
    centers_metal, L_metal, amps_metal, sharpness_metal,
    shape, 3.0, 1e-5, 4, False
)

print(f"  Output sum: {output_metal.sum().item():.6f}")
print(f"  Output max: {output_metal.max().item():.6f}")

loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

metal_d_centers = centers_metal.grad.cpu().numpy()[0]
metal_d_L = L_metal.grad.cpu().numpy()[0]

print(f"  Metal d_centers: {metal_d_centers}")
print(f"  Metal d_L diagonal: [{metal_d_L[0,0]:.3e}, {metal_d_L[1,1]:.3e}, {metal_d_L[2,2]:.3e}]")

# ===============================
# Direct comparison
# ===============================
print(f"\n" + "=" * 80)
print("DIRECT COMPARISON")
print("=" * 80)

diff_centers = metal_d_centers - cpu_d_centers
ratio_centers = metal_d_centers / (cpu_d_centers + 1e-15)

print(f"\nd_centers:")
print(f"  CPU:   {cpu_d_centers}")
print(f"  Metal: {metal_d_centers}")
print(f"  Diff:  {diff_centers}")
print(f"  Ratio: {ratio_centers}")

signs_match = np.sign(metal_d_centers) == np.sign(cpu_d_centers)
mag_match = np.allclose(metal_d_centers, cpu_d_centers, rtol=0.05, atol=1e-7)

print(f"\n  Signs match: {signs_match}")
print(f"  Magnitudes match (5%): {mag_match}")

if not signs_match.all():
    print(f"\n  ❌ SIGN ERRORS:")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        if not signs_match[i]:
            print(f"     {dim}: Metal={np.sign(metal_d_centers[i]):.0f}, CPU={np.sign(cpu_d_centers[i]):.0f}")

if not mag_match:
    print(f"\n  ❌ MAGNITUDE ERRORS:")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        if abs(ratio_centers[i] - 1.0) > 0.05:
            print(f"     {dim}: Metal={metal_d_centers[i]:.3e}, CPU={cpu_d_centers[i]:.3e}, Ratio={ratio_centers[i]:.1f}x")

print(f"\n" + "=" * 80)
if signs_match.all() and mag_match:
    print("✅✅✅ METAL GRADIENTS MATCH CPU! BUG IS FIXED! ✅✅✅")
else:
    print("❌ GRADIENTS STILL DIFFER - MORE WORK NEEDED")
print("=" * 80 + "\n")
