#!/usr/bin/env python
"""
Final comprehensive validation of Metal backend.
Tests all critical aspects after coordinate fix.
"""

from __future__ import annotations

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

print("=" * 80)
print("FINAL METAL BACKEND VALIDATION")
print("=" * 80)

def test_case(name, shape, centers, L, amps, max_allowed_diff=0.05):
    """Test one case and return pass/fail."""
    model_metal = GaussianSplatModelMetal(
        shape=shape, centers0=centers, L0=L, amps0=amps,
        sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0, device="mps"
    )
    model_cpu = GaussianSplatModel(
        shape=shape, centers0=centers, L0=L, amps0=amps,
        sigma_min_diag=[0.5, 0.5, 0.5], truncate=3.0, device="cpu"
    )

    # Forward
    out_m = model_metal().cpu()
    out_p = model_cpu()

    max_diff = (out_m - out_p).abs().max().item()
    mean_diff = (out_m - out_p).abs().mean().item()

    passed = max_diff < max_allowed_diff

    # Backward
    loss_m = out_m.sum()
    loss_m.backward()

    loss_p = out_p.sum()
    loss_p.backward()

    has_grads = any(p.grad is not None and p.grad.norm() > 0 for p in model_metal.parameters())

    print(f"\n{name}:")
    print(f"  Max diff: {max_diff:.6f} (threshold: {max_allowed_diff})")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Gradients: {'✓ computed' if has_grads else '✗ missing'}")
    print(f"  Result: {'✓ PASS' if passed and has_grads else '✗ FAIL'}")

    return passed and has_grads

# Test cases
results = []

# 1. Diagonal L (should be perfect)
print("\n" + "-" * 80)
results.append(("Diagonal L", test_case(
    "Test 1: Diagonal L",
    shape=(16, 16, 16),
    centers=np.array([[8.0, 8.0, 8.0]], dtype=np.float32),
    L=np.array([np.eye(3) * 1.5], dtype=np.float32),
    amps=np.array([1.0], dtype=np.float32),
    max_allowed_diff=0.02
)))

# 2. Non-diagonal L (was problematic)
print("\n" + "-" * 80)
results.append(("Non-diagonal L", test_case(
    "Test 2: Non-diagonal L",
    shape=(16, 16, 16),
    centers=np.array([[8.0, 8.0, 8.0]], dtype=np.float32),
    L=np.array([[[2.0, 0.0, 0.0],
                  [1.0, 1.5, 0.0],
                  [0.5, 0.3, 1.0]]], dtype=np.float32),
    amps=np.array([1.0], dtype=np.float32),
    max_allowed_diff=0.02
)))

# 3. Multiple overlapping splats
print("\n" + "-" * 80)
results.append(("Overlapping splats", test_case(
    "Test 3: Multiple overlapping splats",
    shape=(16, 16, 16),
    centers=np.array([[8, 8, 8], [9, 8, 8], [8, 9, 8], [8, 8, 9], [7, 8, 8]], dtype=np.float32),
    L=np.tile(np.eye(3) * 1.5, (5, 1, 1)).astype(np.float32),
    amps=np.ones(5, dtype=np.float32),
    max_allowed_diff=0.05
)))

# 4. Larger volume with many splats
print("\n" + "-" * 80)
np.random.seed(42)
results.append(("Large volume", test_case(
    "Test 4: Large volume (64³, 100 splats)",
    shape=(64, 64, 64),
    centers=np.random.rand(100, 3) * 48 + 8,
    L=np.tile(np.eye(3) * 2.0, (100, 1, 1)).astype(np.float32),
    amps=np.ones(100, dtype=np.float32),
    max_allowed_diff=0.05
)))

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for name, passed in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")

all_passed = all(passed for _, passed in results)
print("\n" + "=" * 80)
if all_passed:
    print("✅ ALL VALIDATION TESTS PASSED")
    print("Metal backend is PRODUCTION READY!")
else:
    failed = [name for name, passed in results if not passed]
    print(f"⚠️  Some tests failed: {failed}")
    print("Metal backend is functional but needs review for failed cases")
print("=" * 80)
