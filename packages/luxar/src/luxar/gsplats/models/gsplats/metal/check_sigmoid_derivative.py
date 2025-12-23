#!/usr/bin/env python
"""
Check if Metal and CPU models have different raw_mu values,
which would cause different sigmoid derivatives and explain the sign flip.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("SIGMOID DERIVATIVE CHECK")
print("=" * 80)

# Test case
shape = (32, 32, 32)
centers_init = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
L_init = np.eye(3) * 2.0
L_init = L_init[np.newaxis, :, :]
amps_init = np.array([1.0], dtype=np.float32)

# Metal model
model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='mps'
)

# CPU model
model_cpu = GaussianSplatModel(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='cpu'
)

# Get raw_mu values
print(f"\nChecking raw_mu values:")

metal_raw_mu = None
for name, param in model_metal.named_parameters():
    if 'raw_mu' in name:
        metal_raw_mu = param.detach().cpu().numpy()[0]
        print(f"  Metal raw_mu: {metal_raw_mu}")
        break

cpu_raw_mu = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name:
        cpu_raw_mu = param.detach().cpu().numpy()[0]
        print(f"  CPU raw_mu:   {cpu_raw_mu}")
        break

if metal_raw_mu is not None and cpu_raw_mu is not None:
    diff = metal_raw_mu - cpu_raw_mu
    print(f"  Difference:   {diff}")

    if not np.allclose(metal_raw_mu, cpu_raw_mu, atol=1e-6):
        print(f"\n❌ raw_mu values DIFFER between Metal and CPU!")
        print(f"   This explains different sigmoid' values")
    else:
        print(f"\n✅ raw_mu values are SAME")

    # Compute sigmoid and sigmoid'
    print(f"\nSigmoid values:")
    sigmoid_metal = 1 / (1 + np.exp(-metal_raw_mu))
    sigmoid_cpu = 1 / (1 + np.exp(-cpu_raw_mu))
    print(f"  Metal sigmoid: {sigmoid_metal}")
    print(f"  CPU sigmoid:   {sigmoid_cpu}")

    print(f"\nSigmoid derivative (σ'(x) = σ(x) * (1 - σ(x))):")
    sigmoid_prime_metal = sigmoid_metal * (1 - sigmoid_metal)
    sigmoid_prime_cpu = sigmoid_cpu * (1 - sigmoid_cpu)
    print(f"  Metal σ': {sigmoid_prime_metal}")
    print(f"  CPU σ':   {sigmoid_prime_cpu}")

    if not np.allclose(sigmoid_prime_metal, sigmoid_prime_cpu, rtol=0.01):
        print(f"\n⚠️  sigmoid' values differ!")
        print(f"   This could explain gradient magnitude differences")

    # Check the transformation: centers = sigmoid(raw_mu) * (shape - 1)
    print(f"\nCenter computation check:")
    centers_from_metal = sigmoid_metal * (np.array(shape) - 1)
    centers_from_cpu = sigmoid_cpu * (np.array(shape) - 1)
    print(f"  Metal centers: {centers_from_metal}")
    print(f"  CPU centers:   {centers_from_cpu}")
    print(f"  Expected:      {centers_init[0]}")

    if not np.allclose(centers_from_metal, centers_init[0], atol=0.1):
        print(f"\n❌ Metal centers don't match expected!")
    if not np.allclose(centers_from_cpu, centers_init[0], atol=0.1):
        print(f"\n❌ CPU centers don't match expected!")

print("\n" + "=" * 80)
