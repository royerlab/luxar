#!/usr/bin/env python
"""
Python reference implementation that exactly mimics Metal kernel backward pass.

This lets us see pixel-by-pixel what the Metal kernel should be computing
and compare with what it actually produces.
"""

import numpy as np
import torch

print("=" * 80)
print("PYTHON REFERENCE BACKWARD PASS")
print("Mimicking Metal kernel logic exactly")
print("=" * 80)

# Test configuration (matches our diagonal conic test)
shape = (32, 32, 32)
D, H, W = shape
splat_center = np.array([20.0, 16.0, 16.0])  # [Z, Y, X]
conic_diag = np.array([0.25, 0.25, 0.25])  # [c_zz, c_yy, c_xx]
conic_off = np.array([0.0, 0.0, 0.0])  # [c_yz, c_xz, c_xy]
amp = 1.0
sharpness = 2.0
truncate = 3.0
intensity_floor = 1e-5

# Target (same as our test)
target = np.zeros(shape)
target[16, 16, 16] = 1.0

# Compute forward output (simplified - just for this one splat)
print(f"\nConfiguration:")
print(f"  Splat center: {splat_center}")
print(f"  Conic diagonal: {conic_diag}")
print(f"  Target: peak at [16, 16, 16]")

# Forward pass - compute output (use float32 like Metal)
output = np.zeros(shape, dtype=np.float32)
c_zz, c_yy, c_xx = conic_diag.astype(np.float32)
c_yz, c_xz, c_xy = conic_off.astype(np.float32)
splat_center = splat_center.astype(np.float32)

for z in range(D):
    for y in range(H):
        for x in range(W):
            # Distance
            dz = z - splat_center[0]
            dy = y - splat_center[1]
            dx = x - splat_center[2]

            # Mahalanobis distance²
            dist_sq = (dx*dx*c_xx + dy*dy*c_yy + dz*dz*c_zz +
                      2.0 * (dx*dy*c_xy + dx*dz*c_xz + dy*dz*c_yz))

            # Truncation
            s = sharpness
            effective_truncate_sq = truncate ** (4.0 / s)

            if dist_sq <= effective_truncate_sq:
                dist_sq_safe = max(dist_sq, 1e-10)
                inner = -0.5 * (dist_sq_safe ** (s / 2))
                intensity = amp * np.exp(inner)

                if intensity >= intensity_floor:
                    output[z, y, x] = intensity

print(f"\nForward pass:")
print(f"  Output sum: {output.sum():.6f}")
print(f"  Output max: {output.max():.6f} at {np.unravel_index(output.argmax(), shape)}")

# Backward pass - compute gradients (use float32)
grad_output = (2.0 * (output - target)).astype(np.float32)  # ∂loss/∂output for MSE loss

d_centers = np.zeros(3, dtype=np.float32)  # [Z, Y, X]
d_conic_xyz = np.zeros(6, dtype=np.float32)  # [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] in [X,Y,Z] order

contribution_count = {'Z': 0, 'Y': 0, 'X': 0}

print(f"\nBackward pass (pixel-by-pixel):")
print(f"Processing all pixels...")

for z in range(D):
    for y in range(H):
        for x in range(W):
            d_L_d_I = grad_output[z, y, x]

            if abs(d_L_d_I) < 1e-9:
                continue

            # Distance
            dz = z - splat_center[0]
            dy = y - splat_center[1]
            dx = x - splat_center[2]

            # Mahalanobis distance²
            dist_sq = (dx*dx*c_xx + dy*dy*c_yy + dz*dz*c_zz +
                      2.0 * (dx*dy*c_xy + dx*dz*c_xz + dy*dz*c_yz))

            # Truncation
            s = sharpness
            effective_truncate_sq = truncate ** (4.0 / s)

            if dist_sq <= effective_truncate_sq:
                # Recompute forward values
                dist_sq_safe = max(dist_sq, 1e-10)
                half_s = s * 0.5
                inner = -0.5 * (dist_sq_safe ** half_s)
                exp_val = np.exp(inner)
                intensity = amp * exp_val

                # Intensity floor culling
                if intensity < intensity_floor:
                    continue

                d_common = intensity * d_L_d_I

                # Distance gradient
                d_inner_d_D2 = -0.25 * s * (dist_sq_safe ** (half_s - 1.0))
                grad_dist = d_common * d_inner_d_D2

                # Center gradients
                d_D2_d_z = 2.0 * (dz * c_zz + dy * c_yz + dx * c_xz)
                d_D2_d_y = 2.0 * (dz * c_yz + dy * c_yy + dx * c_xy)
                d_D2_d_x = 2.0 * (dz * c_xz + dy * c_xy + dx * c_xx)

                d_centers[0] += grad_dist * d_D2_d_z * -1.0
                d_centers[1] += grad_dist * d_D2_d_y * -1.0
                d_centers[2] += grad_dist * d_D2_d_x * -1.0

                # Count contributions
                contribution_count['Z'] += 1
                if abs(grad_dist * d_D2_d_y * -1.0) > 1e-12:
                    contribution_count['Y'] += 1
                if abs(grad_dist * d_D2_d_x * -1.0) > 1e-12:
                    contribution_count['X'] += 1

                # Conic gradients (in [X,Y,Z] order)
                d_conic_xyz[0] += grad_dist * dx * dx  # c_xx
                d_conic_xyz[1] += grad_dist * 2.0 * dx * dy  # c_xy
                d_conic_xyz[2] += grad_dist * 2.0 * dx * dz  # c_xz
                d_conic_xyz[3] += grad_dist * dy * dy  # c_yy
                d_conic_xyz[4] += grad_dist * 2.0 * dy * dz  # c_yz
                d_conic_xyz[5] += grad_dist * dz * dz  # c_zz

print(f"\nPixels contributing gradients:")
print(f"  Z: {contribution_count['Z']} pixels")
print(f"  Y: {contribution_count['Y']} pixels (non-zero Y gradient)")
print(f"  X: {contribution_count['X']} pixels (non-zero X gradient)")

print(f"\nPython reference d_centers:")
print(f"  [Z, Y, X] = [{d_centers[0]:.6e}, {d_centers[1]:.6e}, {d_centers[2]:.6e}]")

print(f"\nPython reference d_conic ([X,Y,Z] order):")
print(f"  [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]")
print(f"  {d_conic_xyz}")

# Compare with Metal output (from our debug run)
metal_d_centers = np.array([2.735e-01, -4.135e-07, -1.555e-07])
metal_d_conic_xyz = np.array([-88.84, -2.66e-06, 2.34e-06, -88.84, 1.35e-06, -86.67])

print(f"\n" + "=" * 80)
print(f"COMPARISON WITH METAL")
print(f"=" * 80)

print(f"\nd_centers comparison:")
print(f"  Python: {d_centers}")
print(f"  Metal:  {metal_d_centers}")
print(f"  Ratio (Metal/Python): {metal_d_centers / (d_centers + 1e-10)}")

if np.allclose(d_centers, metal_d_centers, rtol=0.05):
    print(f"  ✅ MATCH - Metal is correct!")
else:
    print(f"  ❌ DIFFER:")
    for i, dim in enumerate(['Z', 'Y', 'X']):
        ratio = metal_d_centers[i] / (d_centers[i] + 1e-15)
        print(f"     {dim}: Python={d_centers[i]:.3e}, Metal={metal_d_centers[i]:.3e}, Ratio={ratio:.1f}x")

print(f"\n" + "=" * 80)
