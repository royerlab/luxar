#!/usr/bin/env python
"""
Compare raw gradients between Metal and CPU to verify they match.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("RAW GRADIENT COMPARISON: Metal vs CPU")
print("=" * 80)

# Test case
shape = (32, 32, 32)
center = np.array([[25.0, 16.0, 16.0]], dtype=np.float32)
L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

# Metal model
model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=center, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], device='mps'
)

output_metal = model_metal()
loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

# CPU model
model_cpu = GaussianSplatModel(
    shape=shape, centers0=center, L0=L, amps0=amps,
    sigma_min_diag=[0.5, 0.5, 0.5], device='cpu'
)

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

# Compare ALL gradients
print(f"\nGradient Comparison:")
print("=" * 80)

metal_grads = {}
for name, param in model_metal.named_parameters():
    if param.grad is not None:
        metal_grads[name] = param.grad.detach().cpu().numpy()

cpu_grads = {}
for name, param in model_cpu.named_parameters():
    if param.grad is not None:
        cpu_grads[name] = param.grad.detach().cpu().numpy()

for name in sorted(metal_grads.keys()):
    if name in cpu_grads:
        metal_g = metal_grads[name]
        cpu_g = cpu_grads[name]

        metal_norm = np.linalg.norm(metal_g)
        cpu_norm = np.linalg.norm(cpu_g)
        diff_norm = np.linalg.norm(metal_g - cpu_g)

        match = np.allclose(metal_g, cpu_g, rtol=0.01, atol=1e-7)

        print(f"\n{name}:")
        print(f"  Metal norm: {metal_norm:.6e}")
        print(f"  CPU norm:   {cpu_norm:.6e}")
        print(f"  Diff norm:  {diff_norm:.6e}")
        print(f"  Match:      {'✅' if match else '❌'}")

        if name == 'raw_mu':
            print(f"  Metal raw_mu grad: {metal_g[0]}")
            print(f"  CPU raw_mu grad:   {cpu_g[0]}")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if np.allclose(metal_grads['raw_mu'], cpu_grads['raw_mu'], rtol=0.01):
    print("\n✅ Metal and CPU raw_mu gradients MATCH")
    print("   → The gradient computation is CORRECT")
    print("   → The problem is SATURATED SIGMOID, not Metal gradients!")
    print("\n🔍 Root cause:")
    print("   Centers near volume boundaries → sigmoid saturated → tiny gradients")
    print("   Meanwhile L and amplitude have large gradients → gaussians elongate")
    print("   Result: 'Elongated gaussians in out-of-place locations'")
else:
    print("\n❌ Metal and CPU raw_mu gradients DIFFER")
    print("   → There IS a Metal gradient bug!")

print("\n" + "=" * 80)
