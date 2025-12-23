#!/usr/bin/env python
"""
Test full optimization loop to catch elongated/misplaced gaussian issues.

This simulates actual training to see if splats move correctly toward the target.
"""

import numpy as np
import torch
import torch.optim as optim

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("FULL OPTIMIZATION TEST")
print("=" * 80)

# Test configuration
shape = (32, 32, 32)
n_steps = 100
lr = 0.1

# Place splat FAR from target to see movement
initial_center = np.array([[25.0, 16.0, 16.0]], dtype=np.float32)
target_center = np.array([16.0, 16.0, 16.0])

L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

# Target: peak at center
target = torch.zeros(shape)
target[16, 16, 16] = 1.0

print(f"\nSetup:")
print(f"  Initial center: {initial_center[0]}")
print(f"  Target center:  {target_center}")
print(f"  Steps: {n_steps}, LR: {lr}")

# Test Metal model
print(f"\n" + "=" * 80)
print("METAL OPTIMIZATION")
print("=" * 80)

model_metal = GaussianSplatModelMetal(
    shape=shape,
    centers0=initial_center.copy(),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='mps'
)

optimizer_metal = optim.SGD(model_metal.parameters(), lr=lr)

metal_centers = []
metal_losses = []

for step in range(n_steps):
    optimizer_metal.zero_grad()
    output = model_metal()
    loss = ((output - target.to('mps')) ** 2).sum()
    loss.backward()
    optimizer_metal.step()

    # Record center position
    with torch.no_grad():
        centers, _, _, _ = model_metal._base.current_params()
        center = centers.cpu().numpy()[0].copy()
        metal_centers.append(center)

    metal_losses.append(loss.item())

    if step % 20 == 0:
        print(f"  Step {step:3d}: Loss={loss.item():8.4f}, Center={metal_centers[-1]}")

print(f"\n  Final center: {metal_centers[-1]}")
print(f"  Distance from target: {np.linalg.norm(metal_centers[-1] - target_center):.4f}")

# Test CPU model (reference)
print(f"\n" + "=" * 80)
print("CPU OPTIMIZATION (Reference)")
print("=" * 80)

model_cpu = GaussianSplatModel(
    shape=shape,
    centers0=initial_center.copy(),
    L0=L.copy(),
    amps0=amps.copy(),
    sigma_min_diag=[0.5, 0.5, 0.5],
    device='cpu'
)

optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=lr)

cpu_centers = []
cpu_losses = []

for step in range(n_steps):
    optimizer_cpu.zero_grad()
    output = model_cpu()
    loss = ((output - target) ** 2).sum()
    loss.backward()
    optimizer_cpu.step()

    # Record center position
    with torch.no_grad():
        centers, _, _, _ = model_cpu.current_params()
        center = centers.cpu().numpy()[0].copy()
        cpu_centers.append(center)

    cpu_losses.append(loss.item())

    if step % 20 == 0:
        print(f"  Step {step:3d}: Loss={loss.item():8.4f}, Center={cpu_centers[-1]}")

print(f"\n  Final center: {cpu_centers[-1]}")
print(f"  Distance from target: {np.linalg.norm(cpu_centers[-1] - target_center):.4f}")

# Compare trajectories
print(f"\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

metal_centers = np.array(metal_centers)
cpu_centers = np.array(cpu_centers)

# Check if both moved toward target
metal_moved_correctly = metal_centers[-1][0] < initial_center[0, 0]  # Z should decrease
cpu_moved_correctly = cpu_centers[-1][0] < initial_center[0, 0]

print(f"\nMovement toward target:")
print(f"  Metal: Z moved from {initial_center[0,0]:.2f} to {metal_centers[-1][0]:.2f} ({'✓' if metal_moved_correctly else '✗'})")
print(f"  CPU:   Z moved from {initial_center[0,0]:.2f} to {cpu_centers[-1][0]:.2f} ({'✓' if cpu_moved_correctly else '✗'})")

# Check trajectory similarity
trajectory_diff = np.abs(metal_centers - cpu_centers)
max_diff = np.max(trajectory_diff, axis=0)
final_diff = trajectory_diff[-1]

print(f"\nTrajectory differences:")
print(f"  Max diff [Z, Y, X]:   {max_diff}")
print(f"  Final diff [Z, Y, X]: {final_diff}")

# Check if Metal converged in right direction
metal_final_dist = np.linalg.norm(metal_centers[-1] - target_center)
cpu_final_dist = np.linalg.norm(cpu_centers[-1] - target_center)

print(f"\nFinal distance from target:")
print(f"  Metal: {metal_final_dist:.4f}")
print(f"  CPU:   {cpu_final_dist:.4f}")
print(f"  Ratio: {metal_final_dist / cpu_final_dist:.2f}x")

# Check for "elongated" issue - get covariance info
with torch.no_grad():
    # Get sigma (standard deviations) instead of L
    metal_sigma = model_metal._base.sigma_diag().cpu().numpy()[0]
    cpu_sigma = model_cpu.sigma_diag().cpu().numpy()[0]

    print(f"\nFinal sigma (std deviations, check for elongation):")
    print(f"  Metal sigma: [{metal_sigma[0]:.3f}, {metal_sigma[1]:.3f}, {metal_sigma[2]:.3f}]")
    print(f"  CPU sigma:   [{cpu_sigma[0]:.3f}, {cpu_sigma[1]:.3f}, {cpu_sigma[2]:.3f}]")

    metal_ratio = np.max(metal_sigma) / np.min(metal_sigma)
    cpu_ratio = np.max(cpu_sigma) / np.min(cpu_sigma)

    print(f"  Metal aspect ratio: {metal_ratio:.2f}")
    print(f"  CPU aspect ratio:   {cpu_ratio:.2f}")

    if metal_ratio > 5.0:
        print(f"  ⚠️  Metal gaussian is ELONGATED (ratio > 5)")
    if cpu_ratio > 5.0:
        print(f"  ⚠️  CPU gaussian is ELONGATED (ratio > 5)")

# Final verdict
print(f"\n" + "=" * 80)
trajectories_match = np.allclose(metal_centers, cpu_centers, rtol=0.1, atol=0.5)
distances_match = abs(metal_final_dist - cpu_final_dist) < 0.5

if trajectories_match and distances_match and metal_moved_correctly:
    print("✅✅✅ OPTIMIZATION TEST PASSED ✅✅✅")
    print("    Metal and CPU converge similarly")
else:
    print("❌❌❌ OPTIMIZATION TEST FAILED ❌❌❌")
    if not metal_moved_correctly:
        print("    ✗ Metal moved in WRONG direction!")
    if not trajectories_match:
        print("    ✗ Trajectories differ significantly")
    if not distances_match:
        print("    ✗ Final distances differ")
print("=" * 80)
