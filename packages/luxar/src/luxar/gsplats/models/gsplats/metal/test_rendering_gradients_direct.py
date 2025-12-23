#!/usr/bin/env python
"""
Test rendering gradients DIRECTLY without model wrapper.

This bypasses all parameter transformations and tests if Metal and CPU
produce the same gradients for the same inputs.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import MetalSplatFunction
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

print("=" * 80)
print("DIRECT RENDERING GRADIENT TEST")
print("=" * 80)

# Test inputs (NO parameter transformations)
shape = (32, 32, 32)
centers = torch.tensor([[20.0, 16.0, 16.0]], dtype=torch.float32, requires_grad=True)
L = torch.eye(3).unsqueeze(0) * 2.0
L.requires_grad = True
amps = torch.ones(1, requires_grad=True)
sharpness = torch.ones(1) * 2.0
sharpness.requires_grad = True

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nTest Configuration:")
print(f"  Centers: {centers[0]}")
print(f"  L: diagonal, σ=2")
print(f"  Target: peak at [16, 16, 16]")

# ==================================================
# Metal rendering
# ==================================================
print(f"\n" + "=" * 80)
print("METAL RENDERING (via MetalSplatFunction)")
print("=" * 80)

centers_metal = centers.clone().detach().requires_grad_(True).to('mps')
L_metal = L.clone().detach().requires_grad_(True).to('mps')
amps_metal = amps.clone().detach().requires_grad_(True).to('mps')
sharpness_metal = sharpness.clone().detach().requires_grad_(True).to('mps')

output_metal = MetalSplatFunction.apply(
    centers_metal, L_metal, amps_metal, sharpness_metal,
    shape, 3.0, 1e-5, 4, False  # truncate, floor, tile_size, use_metal_conic
)

loss_metal = ((output_metal - target.to('mps')) ** 2).sum()

print(f"Forward:")
print(f"  Output max: {output_metal.max().item():.6f}")
print(f"  Output sum: {output_metal.sum().item():.6f}")
print(f"  Loss: {loss_metal.item():.6f}")

# Temporarily disable debug output for cleaner display
import os
debug_was_on = os.environ.get('DEBUG_METAL_GRADIENTS')
if debug_was_on:
    del os.environ['DEBUG_METAL_GRADIENTS']

loss_metal.backward()

if debug_was_on:
    os.environ['DEBUG_METAL_GRADIENTS'] = '1'

print(f"\nGradients:")
print(f"  d_centers: {centers_metal.grad.cpu().numpy()[0]}")
print(f"  d_L[0,0]:  {L_metal.grad.cpu().numpy()[0, 0, 0]:.6f} (Z,Z)")
print(f"  d_L[1,1]:  {L_metal.grad.cpu().numpy()[0, 1, 1]:.6f} (Y,Y)")
print(f"  d_L[2,2]:  {L_metal.grad.cpu().numpy()[0, 2, 2]:.6f} (X,X)")
print(f"  d_amps:    {amps_metal.grad.cpu().numpy()[0]:.6f}")

# ==================================================
# CPU rendering
# ==================================================
print(f"\n" + "=" * 80)
print("CPU RENDERING (via render_gaussians)")
print("=" * 80)

centers_cpu = centers.clone().detach().requires_grad_(True)
L_cpu = L.clone().detach().requires_grad_(True)
amps_cpu = amps.clone().detach().requires_grad_(True)
sharpness_cpu = sharpness.clone().detach().requires_grad_(True)

output_cpu = render_gaussians(
    shape, centers_cpu, L_cpu, amps_cpu, sharpness_cpu,
    truncate=3.0, intensity_floor=1e-5
)

loss_cpu = ((output_cpu - target) ** 2).sum()

print(f"Forward:")
print(f"  Output max: {output_cpu.max().item():.6f}")
print(f"  Output sum: {output_cpu.sum().item():.6f}")
print(f"  Loss: {loss_cpu.item():.6f}")

loss_cpu.backward()

print(f"\nGradients:")
print(f"  d_centers: {centers_cpu.grad.cpu().numpy()[0]}")
print(f"  d_L[0,0]:  {L_cpu.grad.cpu().numpy()[0, 0, 0]:.6f} (Z,Z)")
print(f"  d_L[1,1]:  {L_cpu.grad.cpu().numpy()[0, 1, 1]:.6f} (Y,Y)")
print(f"  d_L[2,2]:  {L_cpu.grad.cpu().numpy()[0, 2, 2]:.6f} (X,X)")
print(f"  d_amps:    {amps_cpu.grad.cpu().numpy()[0]:.6f}")

# ==================================================
# Comparison
# ==================================================
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

d_centers_metal = centers_metal.grad.cpu().numpy()[0]
d_centers_cpu = centers_cpu.grad.cpu().numpy()[0]

print(f"\nd_centers:")
print(f"  Metal: {d_centers_metal}")
print(f"  CPU:   {d_centers_cpu}")
print(f"  Diff:  {d_centers_metal - d_centers_cpu}")

signs_match = np.sign(d_centers_metal) == np.sign(d_centers_cpu)
mag_match = np.allclose(d_centers_metal, d_centers_cpu, rtol=0.05, atol=1e-6)

print(f"\n  Signs match: {signs_match}")
print(f"  Magnitudes match (5% tol): {mag_match}")

if not signs_match.all():
    print(f"\n❌ SIGNS DIFFER:")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        if not signs_match[i]:
            print(f"     {dim}: Metal={np.sign(d_centers_metal[i]):.0f}, CPU={np.sign(d_centers_cpu[i]):.0f}")

# Check L gradients too
d_L_metal = L_metal.grad.cpu().numpy()[0]
d_L_cpu = L_cpu.grad.cpu().numpy()[0]

print(f"\nd_L diagonal:")
print(f"  Metal: [{d_L_metal[0,0]:.6f}, {d_L_metal[1,1]:.6f}, {d_L_metal[2,2]:.6f}]")
print(f"  CPU:   [{d_L_cpu[0,0]:.6f}, {d_L_cpu[1,1]:.6f}, {d_L_cpu[2,2]:.6f}]")

L_diag_match = np.allclose(np.diag(d_L_metal), np.diag(d_L_cpu), rtol=0.05)
print(f"  Diagonal match: {L_diag_match}")

print(f"\n" + "=" * 80)
if signs_match.all() and mag_match and L_diag_match:
    print("✅✅✅ METAL AND CPU GRADIENTS MATCH! ✅✅✅")
    print("    The bug is NOT in Metal - it's elsewhere!")
else:
    print("❌❌❌ GRADIENTS DIFFER ❌❌❌")
    if not signs_match.all():
        print("    → Metal produces wrong gradient SIGNS")
    if not mag_match:
        print("    → Metal gradient MAGNITUDES differ by >5%")
print("=" * 80 + "\n")
