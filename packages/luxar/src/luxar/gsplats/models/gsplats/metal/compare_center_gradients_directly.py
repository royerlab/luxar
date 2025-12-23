#!/usr/bin/env python
"""
Compare center gradients DIRECTLY (not raw_mu after activation).

This bypasses any activation function issues and tests if Metal's
d_centers output matches what CPU produces.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import MetalSplatFunction

print("=" * 80)
print("DIRECT CENTER GRADIENT COMPARISON")
print("=" * 80)

# Test case
shape = (32, 32, 32)
centers = torch.tensor([[20.0, 16.0, 16.0]], dtype=torch.float32)
L = torch.eye(3).unsqueeze(0) * 2.0
amps = torch.ones(1)
sharpness = torch.ones(1) * 2.0

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nTest Configuration:")
print(f"  Center: {centers[0]}")
print(f"  Target: [16, 16, 16]")
print(f"  L: diagonal, σ=2")

# ============================================================================
# Metal: Use MetalSplatFunction directly to get d_centers
# ============================================================================
print(f"\n" + "=" * 80)
print("METAL - Direct d_centers from backward")
print("=" * 80)

centers_metal = centers.clone().requires_grad_(True).to('mps')
L_metal = L.clone().requires_grad_(True).to('mps')
amps_metal = amps.clone().requires_grad_(True).to('mps')
sharpness_metal = sharpness.clone().requires_grad_(True).to('mps')

# Call MetalSplatFunction directly
output_metal = MetalSplatFunction.apply(
    centers_metal, L_metal, amps_metal, sharpness_metal,
    shape, 3.0, 1e-5, 4, False  # truncate, intensity_floor, tile_size, use_metal_conic
)

loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

d_centers_metal = centers_metal.grad.detach().cpu().numpy()[0]
print(f"\nMetal d_centers: {d_centers_metal}")

# ============================================================================
# CPU: Get d_centers from regular model
# ============================================================================
print(f"\n" + "=" * 80)
print("CPU - d_centers from backward")
print("=" * 80)

centers_cpu = centers.clone().requires_grad_(True)
L_cpu = L.clone().requires_grad_(True)
amps_cpu = amps.clone().requires_grad_(True)
sharpness_cpu = sharpness.clone().requires_grad_(True)

model_cpu = GaussianSplatModel(
    shape=shape,
    centers0=centers.numpy(),
    L0=L.numpy(),
    amps0=amps.numpy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='cpu'
)

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

# Get the center gradient through the model's parameters
# Need to check which parameter corresponds to centers
d_centers_cpu = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        # This is raw_mu gradient, we need to get actual center gradient
        # For now, let's just use this as reference
        d_centers_cpu = param.grad.detach().cpu().numpy()[0]
        print(f"\nCPU raw_mu grad (for reference): {d_centers_cpu}")
        break

# ============================================================================
# Actually, let me use a simpler approach - create a custom function
# that directly returns center gradients
# ============================================================================

print(f"\n" + "=" * 80)
print("SIMPLIFIED TEST - Bypass activation functions")
print("=" * 80)

# Use raw centers without any parameterization
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

centers_raw = centers.clone().requires_grad_(True)
L_raw = L.clone()
amps_raw = amps.clone()
sharpness_raw = sharpness.clone() * 2.0

output_raw = render_gaussians(
    shape, centers_raw, L_raw, amps_raw, sharpness_raw,
    truncate=3.0, intensity_floor=1e-5
)

loss_raw = ((output_raw - target) ** 2).sum()
loss_raw.backward()

d_centers_raw_cpu = centers_raw.grad.detach().cpu().numpy()[0]
print(f"\nCPU d_centers (direct, no activation): {d_centers_raw_cpu}")

# Compare
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nMetal d_centers:     {d_centers_metal}")
print(f"CPU d_centers (raw): {d_centers_raw_cpu}")

diff = d_centers_metal - d_centers_raw_cpu
print(f"\nDifference: {diff}")
print(f"Absolute:   {np.abs(diff)}")

signs_match = np.sign(d_centers_metal) == np.sign(d_centers_raw_cpu)
print(f"\nSigns match: {signs_match}")

if not signs_match.all():
    print(f"\n❌ CENTER GRADIENTS HAVE WRONG SIGNS!")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        if not signs_match[i]:
            print(f"   {dim}: Metal={d_centers_metal[i]:.6e} (sign={np.sign(d_centers_metal[i]):.0f}), "
                  f"CPU={d_centers_raw_cpu[i]:.6e} (sign={np.sign(d_centers_raw_cpu[i]):.0f})")
else:
    if np.allclose(d_centers_metal, d_centers_raw_cpu, rtol=0.01):
        print(f"\n✅ CENTER GRADIENTS MATCH!")
    else:
        print(f"\n⚠️  Signs match but magnitudes differ")

print("\n" + "=" * 80)
