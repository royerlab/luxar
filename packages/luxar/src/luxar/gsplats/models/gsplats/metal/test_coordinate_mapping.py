#!/usr/bin/env python
"""
Test if Metal is interpreting pixel coordinates correctly.

We'll create a test where we can verify the exact gradient contribution
from a specific known pixel.
"""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal

print("=" * 80)
print("COORDINATE MAPPING TEST")
print("=" * 80)

# Create a TINY volume to make debugging easier
shape = (8, 8, 8)
D, H, W = shape

# Splat at exact center
splat_center = np.array([[4.0, 4.0, 4.0]], dtype=np.float32)
L = np.eye(3) * 1.0  # Small sigma for tight gaussian
L = L[np.newaxis, :, :]
amps = np.array([1.0], dtype=np.float32)

# Target with gradient at ONE specific pixel
target = torch.zeros(shape)
target[5, 4, 4] = 1.0  # Pixel at [Z=5, Y=4, X=4] (one voxel in +Z from center)

print(f"\nConfiguration:")
print(f"  Shape: {shape}")
print(f"  Splat at: {splat_center[0]}")
print(f"  Target peak at: [5, 4, 4] (one voxel away in Z)")
print(f"\nExpected:")
print(f"  Output should have peak at [4, 4, 4] (splat center)")
print(f"  Gradient should pull splat toward [5, 4, 4]")
print(f"  → Z gradient should be POSITIVE")
print(f"  → Y and X gradients should be ~0 (aligned)")

# Metal model
model = GaussianSplatModelMetal(
    shape=shape, centers0=splat_center, L0=L, amps0=amps,
    sigma_min_diag=[0.3, 0.3, 0.3], device='mps'
)

# Forward
output = model()
print(f"\nForward pass:")
print(f"  Output at [4,4,4] (splat): {output[4,4,4].item():.6f}")
print(f"  Output at [5,4,4] (target): {output[5,4,4].item():.6f}")
print(f"  Output max: {output.max().item():.6f} at {tuple(np.unravel_index(output.argmax().cpu().numpy(), shape))}")

# Backward
loss = ((output - target.to('mps')) ** 2).sum()
loss.backward()

# Get gradient
raw_mu_grad = None
for name, param in model.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        raw_mu_grad = param.grad.cpu().numpy()[0]

print(f"\nMetal raw_mu gradient: {raw_mu_grad}")
print(f"  Signs: [Z={np.sign(raw_mu_grad[0]):.0f}, Y={np.sign(raw_mu_grad[1]):.0f}, X={np.sign(raw_mu_grad[2]):.0f}]")

# Manual calculation of expected gradient
print(f"\n" + "=" * 80)
print(f"MANUAL GRADIENT CALCULATION")
print(f"=" * 80)

# The main contribution should come from pixel [4,4,4] where output is high
# and pixel [5,4,4] where target is high

px_splat = [4, 4, 4]  # Output peak
px_target = [5, 4, 4]  # Target peak

print(f"\nPixel [4,4,4] (splat peak, output≈1, target=0):")
print(f"  grad_output ≈ 2*(1-0) = 2 (positive)")
print(f"  d = [0, 0, 0] (at splat center)")
print(f"  → gradient contribution ≈ 0")

print(f"\nPixel [5,4,4] (target peak, output≈small, target=1):")
print(f"  grad_output ≈ 2*(small-1) ≈ -2 (negative)")
print(f"  d = [5-4, 4-4, 4-4] = [1, 0, 0]")
print(f"  → Z gradient ∝ -2 * (-2) * c_zz * 1 = positive ✓")
print(f"  → Y gradient ∝ -2 * (-2) * c_yz * 1 = 0 (c_yz=0) ✓")

print(f"\nExpected signs:")
print(f"  Z: POSITIVE (pull splat in +Z direction)")
print(f"  Y, X: ~0 (already aligned)")

if raw_mu_grad is not None:
    if raw_mu_grad[0] > 0 and abs(raw_mu_grad[1]) < 1e-5 and abs(raw_mu_grad[2]) < 1e-5:
        print(f"\n✅ Metal gradients have CORRECT signs and reasonable magnitudes!")
    else:
        print(f"\n❌ Metal gradients are still WRONG")
        print(f"   Z should be positive: {raw_mu_grad[0]:.3e}")
        print(f"   Y should be ~0: {raw_mu_grad[1]:.3e}")
        print(f"   X should be ~0: {raw_mu_grad[2]:.3e}")

print("\n" + "=" * 80)
