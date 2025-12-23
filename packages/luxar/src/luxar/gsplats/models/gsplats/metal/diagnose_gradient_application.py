#!/usr/bin/env python
"""
Diagnose why center gradients aren't being applied during optimization.
"""

import numpy as np
import torch
import torch.optim as optim

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("GRADIENT APPLICATION DIAGNOSTIC")
print("=" * 80)

# Simple test case
shape = (32, 32, 32)
initial_center = np.array([[25.0, 16.0, 16.0]], dtype=np.float32)
L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

# Test Metal model
print(f"\nTesting Metal Model")
print("=" * 80)

model = GaussianSplatModelMetal(
    shape=shape,
    centers0=initial_center.copy(),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='mps'
)

optimizer = optim.SGD(model.parameters(), lr=0.1)

print(f"\nInitial state:")
centers, Ls, amps_out, sharpness = model._base.current_params()
print(f"  Center: {centers.detach().detach().cpu().numpy()[0]}")
print(f"  Amplitude: {amps_out.detach().detach().cpu().numpy()[0]:.6f}")

print(f"\nStep 1:")
optimizer.zero_grad()
output = model()
loss = ((output - target.to('mps')) ** 2).sum()
print(f"  Loss: {loss.item():.4f}")

loss.backward()

# Check gradients
print(f"\n  Gradients:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        param_norm = param.norm().item()
        print(f"    {name:20s}: grad_norm={grad_norm:.6e}, param_norm={param_norm:.6e}")
        if 'raw_mu' in name:
            print(f"      raw_mu grad: {param.grad.detach().cpu().numpy()[0]}")
            print(f"      raw_mu value: {param.detach().cpu().numpy()[0]}")

optimizer.step()

# Check state after step
centers_after, Ls_after, amps_after, sharpness_after = model._base.current_params()
print(f"\n  After optimizer.step():")
print(f"    Center: {centers_after.detach().cpu().numpy()[0]}")
print(f"    Center changed: {not np.allclose(centers.detach().cpu().numpy()[0], centers_after.detach().cpu().numpy()[0])}")

# Check raw_mu
for name, param in model.named_parameters():
    if 'raw_mu' in name:
        print(f"    raw_mu after step: {param.detach().cpu().numpy()[0]}")

print(f"\nStep 2 (to see if change accumulates):")
optimizer.zero_grad()
output = model()
loss = ((output - target.to('mps')) ** 2).sum()
print(f"  Loss: {loss.item():.4f}")

loss.backward()
optimizer.step()

centers_step2, _, _, _ = model._base.current_params()
print(f"  Center after step 2: {centers_step2.detach().cpu().numpy()[0]}")
print(f"  Total movement: {centers_step2.detach().cpu().numpy()[0] - initial_center[0]}")

# Test if the issue is with raw_mu → center transformation
print(f"\n" + "=" * 80)
print("CHECKING RAW_MU → CENTER TRANSFORMATION")
print("=" * 80)

model2 = GaussianSplatModelMetal(
    shape=shape,
    centers0=np.array([[20.0, 16.0, 16.0]], dtype=np.float32),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='mps'
)

print(f"\nModel with center at [20, 16, 16]:")
for name, param in model2._base.named_parameters():
    if 'raw_mu' in name:
        print(f"  raw_mu: {param.detach().cpu().numpy()[0]}")

centers_20, _, _, _ = model2._base.current_params()
print(f"  Actual center: {centers_20.detach().cpu().numpy()[0]}")

# Manually modify raw_mu
print(f"\nManually changing raw_mu...")
for name, param in model2._base.named_parameters():
    if 'raw_mu' in name:
        with torch.no_grad():
            param[0, 0] += 1.0  # Increase Z by modifying raw_mu
        print(f"  raw_mu after manual change: {param.detach().cpu().numpy()[0]}")

centers_modified, _, _, _ = model2._base.current_params()
print(f"  Center after manual raw_mu change: {centers_modified.detach().cpu().numpy()[0]}")
print(f"  Center changed by: {centers_modified.detach().cpu().numpy()[0] - centers_20.detach().cpu().numpy()[0]}")

print("\n" + "=" * 80)
