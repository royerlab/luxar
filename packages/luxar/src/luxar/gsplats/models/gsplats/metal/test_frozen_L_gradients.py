#!/usr/bin/env python
"""
Test with FROZEN L matrix to isolate where the gradient bug occurs.

If gradients match with L frozen: Bug is in d_conic → d_Ls chain rule
If gradients differ with L frozen: Bug is in d_centers handling
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("FROZEN L GRADIENT TEST")
print("=" * 80)

# Test case
shape = (32, 32, 32)
centers_init = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
L_init = np.eye(3) * 2.0
L_init = L_init[np.newaxis, :, :]
amps_init = np.array([1.0], dtype=np.float32)

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nTest Configuration:")
print(f"  Centers: {centers_init[0]}")
print(f"  L: diagonal, σ=2 (FROZEN)")
print(f"  Target: peak at [16, 16, 16]")
print(f"\nThis eliminates d_conic → d_Ls chain rule")
print(f"Only d_centers → raw_mu gradient remains")

# ==================================================
# Metal with FROZEN L
# ==================================================
print(f"\n" + "=" * 80)
print("METAL MODEL (L frozen)")
print("=" * 80)

model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='mps'
)

# Freeze ALL L-related parameters
n_frozen = 0
for name, param in model_metal.named_parameters():
    if 'L' in name or 'raw_L' in name:
        param.requires_grad = False
        n_frozen += 1
        print(f"  Froze: {name}")

print(f"\nTotal L parameters frozen: {n_frozen}")

output_metal = model_metal()
loss_metal = ((output_metal - target.to('mps')) ** 2).sum()

print(f"\nForward:")
print(f"  Loss: {loss_metal.item():.6f}")

loss_metal.backward()

metal_raw_mu_grad = None
for name, param in model_metal.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        metal_raw_mu_grad = param.grad.cpu().numpy()[0]

print(f"\nMetal raw_mu gradient (L frozen): {metal_raw_mu_grad}")

# ==================================================
# CPU with FROZEN L
# ==================================================
print(f"\n" + "=" * 80)
print("CPU MODEL (L frozen)")
print("=" * 80)

model_cpu = GaussianSplatModel(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='cpu'
)

# Freeze ALL L-related parameters
n_frozen_cpu = 0
for name, param in model_cpu.named_parameters():
    if 'L' in name or 'raw_L' in name:
        param.requires_grad = False
        n_frozen_cpu += 1

print(f"Total L parameters frozen: {n_frozen_cpu}")

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()

print(f"\nForward:")
print(f"  Loss: {loss_cpu.item():.6f}")

loss_cpu.backward()

cpu_raw_mu_grad = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        cpu_raw_mu_grad = param.grad.cpu().numpy()[0]

print(f"\nCPU raw_mu gradient (L frozen):   {cpu_raw_mu_grad}")

# ==================================================
# Comparison
# ==================================================
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

diff = metal_raw_mu_grad - cpu_raw_mu_grad
diff_pct = 100 * np.abs(diff / (cpu_raw_mu_grad + 1e-10))

print(f"\nAbsolute difference: {diff}")
print(f"Relative difference: {diff_pct} %")

signs_match = np.sign(metal_raw_mu_grad) == np.sign(cpu_raw_mu_grad)
mag_match = np.allclose(metal_raw_mu_grad, cpu_raw_mu_grad, rtol=0.05, atol=1e-6)

print(f"\nSigns match: {signs_match}")
print(f"Magnitudes match (5%): {mag_match}")

print(f"\n" + "=" * 80)
if signs_match.all() and mag_match:
    print("✅ GRADIENTS MATCH WITH L FROZEN!")
    print("   → Bug is in d_conic → d_Ls chain rule or how d_Ls affects raw_mu")
    print("   → This would be very strange since they're separate parameters")
elif signs_match.all():
    print("⚠️  SIGNS MATCH but magnitudes differ")
    print("   → Might be numerical precision or scaling issue")
else:
    print("❌ GRADIENTS STILL DIFFER WITH L FROZEN!")
    print("   → Bug is in d_centers computation or how Metal returns d_centers")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        if not signs_match[i]:
            print(f"   ✗ {dim}: Metal={np.sign(metal_raw_mu_grad[i]):.0f}, "
                  f"CPU={np.sign(cpu_raw_mu_grad[i]):.0f}")
print("=" * 80 + "\n")
