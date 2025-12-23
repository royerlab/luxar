# Metal Gradient Bug - Detailed Findings and Proposed Fix

**Date:** 2025-12-22
**Status:** Bug located in d_conic computation or chain rule

---

## Executive Summary of Findings

Through systematic debugging with instrumented code, we have narrowed down the bug:

1. ✅ **Metal's d_centers is plausible:** [0.2735, -4.14e-07, -1.56e-07]
2. ❌ **Metal's d_conic appears wrong:** c_xx and c_yy gradients are large (-88.84) when they should likely be smaller or different
3. ❌ **Final raw_mu gradients are wrong:** Y gradient is 200x larger in Metal than CPU

---

## Debug Output Analysis

### Test Case
- Diagonal conic: σ = 2.0 → Σ⁻¹ diagonal = [0.25, 0.25, 0.25]
- Splat at [20, 16, 16], Target at [16, 16, 16]
- Only Z offset (dz=4), Y and X aligned (dy=0, dx=0 at target position)

### Metal Output (from debug probes)

**d_centers (raw from Metal kernel):**
```
[Z=2.735e-01, Y=-4.135e-07, X=-1.555e-07]
```
- Z: Large positive (correct - pulls splat toward target)
- Y, X: Near-zero (correct - already aligned)

**d_conic before reorder ([X,Y,Z] order):**
```
[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
[-88.84, -2.66e-06, 2.34e-06, -88.84, 1.35e-06, -86.67]
```

**d_conic after reorder ([Z,Y,X] order):**
```
[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]
[-86.67, 1.35e-06, 2.34e-06, -88.84, -2.66e-06, -88.84]
```

**conic_recomputed (forward conic for chain rule):**
```
[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]
[0.25, 0., 0., 0.25, 0., 0.25]
```

**d_Ls (from chain rule):**
```
[[21.67,  0,       0    ]
 [-1.69e-07, 22.21, 0    ]
 [-2.92e-07, 3.33e-07, 22.21]]
```

**Final raw_mu gradient:**
```
Metal: [ 1.941, -3.20e-06, -1.20e-06]
CPU:   [ 1.922, -1.44e-08, -5.48e-07]
```

---

## Analysis

### Problem 1: d_conic Values

For a **diagonal conic** with the gaussian spread isotropically, the d_conic gradients come from:

```metal
val_conic[0] = grad_dist * dx * dx;  // c_xx
val_conic[3] = grad_dist * dy * dy;  // c_yy
val_conic[5] = grad_dist * dz * dz;  // c_zz
```

**Accumulated over all pixels:**
- Pixels with dx ≠ 0 contribute to c_xx
- Pixels with dy ≠ 0 contribute to c_yy
- Pixels with dz ≠ 0 contribute to c_zz

Since the gaussian is isotropic (σ=2 in all directions) and spreads to ~same number of pixels in each direction:
- We expect c_xx ≈ c_yy ≈ c_zz (similar magnitudes)
- **Observed:** c_xx = c_yy = -88.84, c_zz = -86.67 ✓ Similar!

**This suggests d_conic might actually be CORRECT!**

### Problem 2: Chain Rule Amplification

The chain rule computes:
```python
d_Ls = ∂conic/∂L · d_conic
```

For diagonal L (and thus diagonal conic), the Jacobian ∂conic/∂L should also be mostly diagonal. The d_Ls output shows:
```
Diagonal: [21.67, 22.21, 22.21]
Off-diagonal: ~1e-07 (near zero)
```

This looks reasonable - diagonal gradients are large, off-diagonal tiny.

### Problem 3: Where Does the Error Come From?

**Hypothesis:** The error might be in how d_Ls gradients affect the raw_mu gradient through the full autograd graph.

The full computation graph is:
```
raw_mu → sigmoid → centers → [Metal rendering] → output → loss
   ↑                ↑         ↑
   └─ needs grad ───┴─────────┘
```

But ALSO:
```
raw_L_diag, L_off → L matrix → cholesky_to_conic → conic → [Metal rendering] → output
   ↑                                                   ↑
   └─────────── needs grad ────────────────────────────┘
```

**The key question:** When Metal returns both d_centers AND d_Ls, how does PyTorch combine them to get raw_mu gradient?

**Answer:** It shouldn't combine them! d_centers flows to raw_mu, d_Ls flows to raw_L_diag/L_off. They're separate parameters.

But there might be an interaction if the `current_params()` function somehow couples centers and L...

Let me check if current_params has any coupling between raw_mu and the L parameters.

---

## Next Debugging Step

**Create a test that uses FROZEN L matrix** (no gradients for L), only optimize centers:

```python
# Freeze L - only let centers vary
L.requires_grad = False
```

This will eliminate the d_conic → d_Ls chain rule entirely. If the raw_mu gradients STILL differ between Metal and CPU with frozen L, then the bug is purely in how d_centers is handled.

If the gradients MATCH with frozen L, then the bug is in how d_Ls affects raw_mu (which would be strange, since they should be independent).

---

## Code to Add

### Test with Frozen L

```python
#!/usr/bin/env python
"""Test with frozen L to isolate d_centers bug."""

import numpy as np
import torch

from luxar.gsplats.models.gsplats.metal import GaussianSplatModelMetal
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

# Test case
shape = (32, 32, 32)
centers_init = np.array([[20.0, 16.0, 16.0]], dtype=np.float32)
L_init = np.eye(3) * 2.0
L_init = L_init[np.newaxis, :, :]
amps_init = np.array([1.0], dtype=np.float32)

target = torch.zeros(shape)
target[16, 16, 16] = 1.0

# Metal with FROZEN L
model_metal = GaussianSplatModelMetal(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='mps'
)

# Freeze L parameters
for name, param in model_metal.named_parameters():
    if 'L' in name:
        param.requires_grad = False

output_metal = model_metal()
loss_metal = ((output_metal - target.to('mps')) ** 2).sum()
loss_metal.backward()

metal_raw_mu_grad = None
for name, param in model_metal.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        metal_raw_mu_grad = param.grad.cpu().numpy()[0]

print(f"Metal raw_mu gradient (L frozen): {metal_raw_mu_grad}")

# CPU with FROZEN L
model_cpu = GaussianSplatModel(
    shape=shape, centers0=centers_init, L0=L_init, amps0=amps_init,
    sigma_min_diag=[0.5, 0.5, 0.5], device='cpu'
)

for name, param in model_cpu.named_parameters():
    if 'L' in name:
        param.requires_grad = False

output_cpu = model_cpu()
loss_cpu = ((output_cpu - target) ** 2).sum()
loss_cpu.backward()

cpu_raw_mu_grad = None
for name, param in model_cpu.named_parameters():
    if 'raw_mu' in name and param.grad is not None:
        cpu_raw_mu_grad = param.grad.cpu().numpy()[0]

print(f"CPU raw_mu gradient (L frozen):   {cpu_raw_mu_grad}")

# Compare
if np.allclose(metal_raw_mu_grad, cpu_raw_mu_grad, rtol=0.01):
    print("\n✅ MATCH with L frozen → Bug is in d_Ls / chain rule")
else:
    print("\n❌ DIFFER with L frozen → Bug is in d_centers handling")
    print(f"   Difference: {metal_raw_mu_grad - cpu_raw_mu_grad}")
```

---

## Conclusion So Far

We know:
1. ✅ Metal kernel d_centers values are plausible
2. ❓ Metal d_conic values look large but might be correct (need CPU comparison)
3. ❌ Final raw_mu gradients differ significantly in Y and X

**Next step:** Test with frozen L to isolate whether bug is in d_centers path or d_Ls path.

