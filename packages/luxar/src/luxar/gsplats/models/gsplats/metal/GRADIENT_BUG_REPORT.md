# CRITICAL: Metal Backend Gradient Bug Report

**Status:** 🚨 **BLOCKING** - Metal backend produces incorrect gradients, causing optimization failures
**Impact:** Elongated gaussians in wrong locations during fitting
**Priority:** P0 - Must fix before Metal backend can be used for training

---

## Executive Summary

The Metal-accelerated Gaussian splatting backward pass computes **incorrect center gradients**, particularly with wrong signs in the Y dimension. This causes optimization to move splats in the wrong direction, resulting in:
- Extremely elongated gaussians
- Splats appearing in out-of-place locations
- Failed convergence during fitting

**Evidence:**
```
Test case: Splat at [25,25,25], target peak at [8,8,8]
Expected: All negative gradients (pull splat toward target)

Metal:  [-5.77e-07, +4.28e-07, -4.15e-07]  ← Y is POSITIVE (WRONG!)
CPU:    [-1.23e-06, -2.88e-07, -5.95e-07]  ← All negative (CORRECT)
```

The Y gradient has the **opposite sign**, causing the splat to move **away** from the target instead of toward it.

---

## 1. Mathematical Background

### 1.1 Gaussian Splat Rendering

A 3D Gaussian splat is defined by:
- **Center μ** ∈ ℝ³ (position in [Z,Y,X] coordinates)
- **Covariance Σ** ∈ ℝ³ˣ³ (shape and orientation)
- **Amplitude α** ∈ ℝ (intensity)
- **Sharpness s** ∈ ℝ (exponent scaling)

The contribution of a splat to voxel **p** is:

```
I(p) = α · exp(-s · D²(p, μ, Σ))
```

Where **D²** is the squared Mahalanobis distance:

```
D²(p, μ, Σ) = (p - μ)ᵀ · Σ⁻¹ · (p - μ)
```

### 1.2 Conic Representation

For efficiency, we work with **Σ⁻¹** (called "conic"), which is a 3×3 symmetric matrix:

```
Σ⁻¹ = [c_zz  c_yz  c_xz]
      [c_yz  c_yy  c_xy]
      [c_xz  c_xy  c_xx]
```

Stored as 6 values: `[c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]`

The squared distance becomes:

```
D²(p, μ) = dz·(c_zz·dz + c_yz·dy + c_xz·dx) +
           dy·(c_yz·dz + c_yy·dy + c_xy·dx) +
           dx·(c_xz·dz + c_xy·dy + c_xx·dx)
```

Where `d = p - μ = [dz, dy, dx]`

### 1.3 Gradient Formulas (from Spec §2.6)

#### Center Gradient
The gradient of D² with respect to the center μ is:

```
∂D²/∂μ = -2 · Σ⁻¹ · d
```

Expanded in components:

```
∂D²/∂z = -2 · (c_zz·dz + c_yz·dy + c_xz·dx)
∂D²/∂y = -2 · (c_yz·dz + c_yy·dy + c_xy·dx)
∂D²/∂x = -2 · (c_xz·dz + c_xy·dy + c_xx·dx)
```

**Critical point:** The negative sign means that when `dz > 0` (splat below target), the gradient should be **negative** to pull the splat upward.

#### Loss Propagation
Given `grad_loss` (gradient of loss w.r.t. output intensity), the gradient w.r.t. center is:

```
d_center = grad_loss · dI/dp · ∂D²/∂μ

Where: dI/dp = -α · s · exp(-s·D²) · ∂D²/∂p
```

---

## 2. Coordinate Convention

**CRITICAL:** The codebase uses **NumPy/PyTorch [Z,Y,X]** convention, NOT Cartesian [X,Y,Z].

### 2.1 Tensor Storage Order

```python
# In Python (NumPy/PyTorch)
centers.shape = (N, 3)  # centers[:, 0] = Z, centers[:, 1] = Y, centers[:, 2] = X
conic.shape = (N, 6)    # [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]
```

### 2.2 Metal float3 Mapping

```metal
// In Metal kernels - CONFUSING MAPPING!
float3 center;
center.x = Z_value  // ← NOT X! This is DEPTH
center.y = Y_value  // ← Height (correct)
center.z = X_value  // ← NOT Z! This is WIDTH

// Example from spec:
float z = center.x;  // Extract Z (depth)
float y = center.y;  // Extract Y (height)
float x = center.z;  // Extract X (width)
```

### 2.3 Conic Reordering

PyTorch computes conic in [Z,Y,X] order, but Metal expects [X,Y,Z]:

```python
# Python side (gsplat_model_metal.py:220-223)
conic = cholesky_to_conic(Ls)  # Returns [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]

# Reorder to [X,Y,Z] for Metal
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
# Result: [c_xx, c_yy, c_zz, c_yz, c_xz, c_xy]
```

**Permutation explained:**
- Index 0 (c_zz) → Index 2 in Metal
- Index 1 (c_yy) → Index 1 in Metal
- Index 2 (c_xx) → Index 0 in Metal
- Index 3 (c_xy) → Index 5 in Metal
- Index 4 (c_xz) → Index 4 in Metal
- Index 5 (c_yz) → Index 3 in Metal

---

## 3. Metal Backward Implementation

### 3.1 Current Code (kernels.metal:460-470)

```metal
// Extract conic elements (in [X,Y,Z] order as received from Python)
float c_xx = conic_data[i * 6 + 0];
float c_yy = conic_data[i * 6 + 1];
float c_zz = conic_data[i * 6 + 2];
float c_yz = conic_data[i * 6 + 3];
float c_xz = conic_data[i * 6 + 4];
float c_xy = conic_data[i * 6 + 5];

// ... (D² computation) ...

// 4. Center gradient: ∂D/∂μ = -2 × Σ^-1 × d
// d = [dz, dy, dx] (already extracted above), conic in [X,Y,Z]
float3 d_D2_d_d;  // Result will be in [Z,Y,X] order for Python

// THIS IS THE SUSPECTED BUG LOCATION:
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x

// Apply chain rule with loss gradient
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

### 3.2 Analysis

**Formula used:**
```
∂D²/∂z = +2 · (c_zz·dz + c_yz·dy + c_xz·dx)
∂D²/∂y = +2 · (c_yz·dz + c_yy·dy + c_xy·dx)
∂D²/∂x = +2 · (c_xz·dz + c_xy·dy + c_xx·dx)

Then multiplied by -1.0f at the end
```

**Expected formula (from spec):**
```
∂D²/∂z = -2 · (c_zz·dz + c_yz·dy + c_xz·dx)
∂D²/∂y = -2 · (c_yz·dz + c_yy·dy + c_xy·dx)
∂D²/∂x = -2 · (c_xz·dz + c_xy·dy + c_xx·dx)
```

**Apparent discrepancy:**
The code uses `+2.0f` then multiplies by `-1.0f` at the end, which should be equivalent to `-2.0f`. However, the test results show this produces wrong signs.

**Possible issues:**
1. **Coordinate mixing:** The code comment says "conic in [X,Y,Z]" but uses indices assuming [Z,Y,X]
2. **Sign error in chain rule:** The `grad_dist` term may have wrong sign
3. **Incorrect conic element mapping:** The indices may not correspond to the correct matrix elements

---

## 4. PyTorch CPU Reference Implementation

### 4.1 Forward Pass (gsplat_model.py:85-110)

```python
def forward(self) -> torch.Tensor:
    # Get current centers [N, d] in [Z,Y,X] order
    centers = self.mu()

    # Compute Cholesky L [N, d, d]
    L = self.L()

    # Compute conic (Σ⁻¹) from L [N, d*(d+1)//2]
    conic = self._cholesky_to_conic(L)  # [N, 6] = [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]

    # Render using custom CUDA or fallback
    return render_forward(centers, conic, amps, sharpness, shape, truncate)
```

### 4.2 CPU Fallback Rendering (gsplat_model.py:160-195)

```python
# For each voxel
for idx in range(total_voxels):
    p = coords[idx]  # Voxel position [Z, Y, X]

    for i in range(N):
        # Compute delta d = p - μ
        d = p - centers[i]  # [dz, dy, dx]

        # Compute D² = dᵀ · Σ⁻¹ · d
        c = conic[i]  # [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]
        D2 = (d[0] * (c[0]*d[0] + c[5]*d[1] + c[4]*d[2]) +
              d[1] * (c[5]*d[0] + c[1]*d[1] + c[3]*d[2]) +
              d[2] * (c[4]*d[0] + c[3]*d[1] + c[2]*d[2]))

        # Compute intensity
        if D2 < truncate * truncate:
            I = amps[i] * exp(-sharpness[i] * D2)
            output[idx] += I
```

### 4.3 PyTorch Autograd

The CPU implementation relies on PyTorch's autograd to compute gradients:

```python
# In fitting loop
output = model()  # Forward pass
loss = ((output - target) ** 2).sum()
loss.backward()  # Autograd computes all gradients

# Gradients available in model.raw_mu.grad, model.L_off.grad, etc.
```

PyTorch automatically computes correct gradients for the entire chain:
```
loss → output → exp(-s·D²) → D² → centers
```

---

## 5. Test Evidence

### 5.1 Test Setup (test_gradient_correctness.py)

```python
# Simple test: ONE splat, check gradient direction
shape = (32, 32, 32)

# Place splat ABOVE center
center_z, center_y, center_x = 20, 16, 16
centers = np.array([[center_z, center_y, center_x]], dtype=np.float32)
L = np.array([np.eye(3) * 2.0], dtype=np.float32)
amps = np.array([1.0], dtype=np.float32)

# Target: peak at center
target = torch.zeros(shape)
target[16, 16, 16] = 1.0

# Expected: Gradient should pull splat DOWN (toward target)
# Therefore: d_z should be NEGATIVE (since z=20 > 16)
```

### 5.2 Test Results

```
Metal gradient (raw_mu): [-5.77e-07, +4.28e-07, -4.15e-07]
CPU gradient (raw_mu):   [-1.23e-06, -2.88e-07, -5.95e-07]

Gradient signs:
  Metal: [-1.0, +1.0, -1.0]  ← Y is POSITIVE (WRONG!)
  CPU:   [-1.0, -1.0, -1.0]  ← All negative (CORRECT)
```

**Analysis:**
- **Z gradient:** Both negative ✓
- **Y gradient:** Metal is **positive**, CPU is **negative** ✗
- **X gradient:** Metal is **negative**, CPU is **negative** ✓ (but magnitude differs)

The Y gradient having the **opposite sign** will cause the optimizer to move the splat in the **wrong direction**.

### 5.3 Impact on Optimization

With wrong gradients, SGD optimizer does:
```python
# SGD update rule
new_center = old_center - learning_rate * gradient

# With wrong Y gradient:
new_y = 16 - lr * (+4.28e-07)  # MOVES DOWN (wrong direction!)

# Should be:
new_y = 16 - lr * (-2.88e-07)  # MOVES UP (toward target at y=16)
```

This explains the user's observation of "extremely elongated gaussians in out-of-place locations."

---

## 6. Gradient Chain Rule

### 6.1 Metal Backward Chain

```python
# In gsplat_model_metal.py:MetalSplatFunction.backward()

# Step 1: Metal computes raw gradients
d_centers, d_conic, d_amps, d_sharpness = metal_splatting_backend.backward_3d(
    grad_output,  # ∂L/∂output
    centers_mps, conic_mps, amps_mps, sharpness_mps, Ls_mps,
    saved_tile_data, shape, truncate, intensity_floor, tile_size
)
# Returns:
#   d_centers: ∂L/∂centers (BUGGED!)
#   d_conic: ∂L/∂conic

# Step 2: Reorder d_conic from [X,Y,Z] back to [Z,Y,X]
d_conic = d_conic[:, [5, 4, 2, 3, 1, 0]]

# Step 3: Chain rule to get d_Ls
with torch.enable_grad():
    Ls_for_conic.requires_grad_(True)
    conic_recomputed = cholesky_to_conic(Ls_for_conic)

    # Backprop through conic computation
    d_Ls, = torch.autograd.grad(
        outputs=conic_recomputed,
        inputs=Ls_for_conic,
        grad_outputs=d_conic,
        retain_graph=False
    )

return d_centers, d_Ls, d_amps, d_sharpness
```

### 6.2 The Bug Location

The bug is in **Step 1** - the Metal kernel `rasterize_bwd_3d` computes incorrect `d_centers`.

**Evidence it's not a chain rule issue:**
- Test used `use_metal_conic=False`, so PyTorch computed conic gradients
- Only Metal's center gradients had wrong signs
- The bug persists regardless of whether Metal or PyTorch computes conic

---

## 7. Debugging Hypotheses

### 7.1 Hypothesis 1: Coordinate System Confusion

**Issue:** The Metal kernel may be mixing [Z,Y,X] and [X,Y,Z] conventions.

**Evidence:**
```metal
// kernels.metal:460
// Comment says: "conic in [X,Y,Z]"
// But indices used:
d_D2_d_d.x = ... (c_zz·dz + c_yz·dy + c_xz·dx)  // Correct for Z
d_D2_d_d.y = ... (c_yz·dz + c_yy·dy + c_xy·dx)  // Correct for Y
d_D2_d_d.z = ... (c_xz·dz + c_xy·dy + c_xx·dx)  // Correct for X
```

The formulas appear correct, but the `float3` assignment may be wrong:
- `d_D2_d_d.x` should be ∂D²/∂z (Z gradient)
- `d_D2_d_d.y` should be ∂D²/∂y (Y gradient)
- `d_D2_d_d.z` should be ∂D²/∂x (X gradient)

**But:** `float3` in Metal uses `.x/.y/.z` which may map to [X,Y,Z], not [Z,Y,X]!

### 7.2 Hypothesis 2: Incorrect Conic Element Access

**Issue:** The conic is reordered to [X,Y,Z] before Metal, but the kernel accesses it incorrectly.

**Reordering (Python side):**
```python
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
# From: [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]  ([Z,Y,X] order)
# To:   [c_yz, c_xz, c_xx, c_xy, c_yy, c_zz]  ([X,Y,Z] order??)
```

**Wait - this permutation is suspicious!**

Let me trace the permutation:
- Original index 0 (c_zz) → Take from index 5 → **c_yz** ✗ WRONG!

**The permutation `[5,4,2,3,1,0]` is incorrect!**

To convert [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] to [X,Y,Z] order [c_xx, c_yy, c_zz, c_xy, c_xz, c_yz]:
```
Index 0: c_zz → should be c_xx → take from index 2
Index 1: c_yy → should be c_yy → take from index 1
Index 2: c_xx → should be c_zz → take from index 0
Index 3: c_xy → should be c_xy → take from index 3
Index 4: c_xz → should be c_xz → take from index 4
Index 5: c_yz → should be c_yz → take from index 5

Correct permutation: [2, 1, 0, 3, 4, 5]
```

**Currently used:** `[5, 4, 2, 3, 1, 0]` ← **WRONG PERMUTATION!**

### 7.3 Hypothesis 3: Sign Error in grad_dist

**Issue:** The `grad_dist` term may have wrong sign due to chain rule error.

**Chain rule:**
```
∂L/∂center = ∂L/∂I · ∂I/∂D² · ∂D²/∂center

Where:
  ∂I/∂D² = -α·s·exp(-s·D²)  (always negative for positive I)
  ∂D²/∂center = -2·Σ⁻¹·d  (negative sign in formula)

Combined:
  ∂L/∂center = grad_output · (-α·s·exp(-s·D²)) · (-2·Σ⁻¹·d)
             = grad_output · α·s·exp(-s·D²) · 2·Σ⁻¹·d
```

The two negatives should cancel, giving a positive coefficient!

**But the Metal code does:**
```metal
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

Where `grad_dist = grad_output · (-α·s·exp(-s·D²))` (negative)

So:
```
val_centers = (negative) * (+2·Σ⁻¹·d) * (-1)
            = (negative) * (negative) * (Σ⁻¹·d)
            = positive * (Σ⁻¹·d)
```

**This seems correct in principle**, but may have sign error depending on what `grad_dist` actually contains.

---

## 8. Recommended Debugging Steps

### 8.1 Immediate Checks

1. **Verify conic permutation:**
   ```python
   # Print conic before and after permutation
   print("Original conic [Z,Y,X]:", conic[0])
   print("Reordered conic:", conic_reordered[0])

   # Expected for identity Σ = diag(σ², σ², σ²):
   # Original: [σ⁻², σ⁻², σ⁻², 0, 0, 0]
   # Reordered should have same diagonal, same zeros
   ```

2. **Check Metal conic extraction:**
   ```metal
   // Add debug prints in backward kernel
   printf("Splat %d conic: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
          i, c_xx, c_yy, c_zz, c_xy, c_xz, c_yz);
   ```

3. **Verify gradient sign:**
   ```metal
   // Print intermediate values
   printf("d = [%.6f, %.6f, %.6f]\n", dz, dy, dx);
   printf("d_D2_d_d = [%.6f, %.6f, %.6f]\n",
          d_D2_d_d.x, d_D2_d_d.y, d_D2_d_d.z);
   printf("grad_dist = %.6f\n", grad_dist);
   printf("final gradient = [%.6f, %.6f, %.6f]\n",
          val_centers.x, val_centers.y, val_centers.z);
   ```

### 8.2 Fix Candidate 1: Correct Conic Permutation

**Current (WRONG):**
```python
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
```

**Proposed fix:**
```python
# Correct permutation [Z,Y,X] → [X,Y,Z]
# [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] → [c_xx, c_yy, c_zz, c_xy, c_xz, c_yz]
conic_reordered = conic[:, [2, 1, 0, 3, 4, 5]]
```

**Apply same fix to backward:**
```python
# In MetalSplatFunction.backward()
# Current: d_conic = d_conic[:, [5, 4, 2, 3, 1, 0]]
# Fixed:
d_conic = d_conic[:, [2, 1, 0, 3, 4, 5]]
```

### 8.3 Fix Candidate 2: Correct float3 Mapping

If the permutation is actually correct (needs verification), then the issue may be in float3 assignment:

**Current:**
```metal
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x
```

**Proposed fix (if float3 maps to [X,Y,Z]):**
```metal
d_D2_d_d.x = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x → .x
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y → .y
d_D2_d_d.z = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z → .z
```

### 8.4 Fix Candidate 3: Add Explicit Return Mapping

To avoid ambiguity, return gradients in a guaranteed order:

```metal
// In rasterize_bwd_3d kernel
// Compute gradients in clear variables
float grad_z = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);
float grad_y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);
float grad_x = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);

// Apply chain rule
float3 d_center;
d_center.x = grad_dist * grad_z * -1.0f;  // Z gradient
d_center.y = grad_dist * grad_y * -1.0f;  // Y gradient
d_center.z = grad_dist * grad_x * -1.0f;  // X gradient

// Write to buffer in explicit [Z,Y,X] order for Python
d_centers_ptr[i * 3 + 0] = d_center.x;  // Z
d_centers_ptr[i * 3 + 1] = d_center.y;  // Y
d_centers_ptr[i * 3 + 2] = d_center.z;  // X
```

---

## 9. Test Plan

### 9.1 Unit Test: Conic Permutation

```python
def test_conic_permutation():
    """Verify conic reordering is correct."""
    # Create identity covariance: Σ = I
    L = torch.eye(3).unsqueeze(0)  # [1, 3, 3]

    # Compute conic in [Z,Y,X] order
    conic = cholesky_to_conic(L)  # [1, 6]
    # Should be: [1, 1, 1, 0, 0, 0] = [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]

    # Apply current permutation
    conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
    print("Current permutation:", conic_reordered[0])
    # Should still be: [1, 1, 1, 0, 0, 0] for identity

    # Apply proposed fix
    conic_fixed = conic[:, [2, 1, 0, 3, 4, 5]]
    print("Proposed permutation:", conic_fixed[0])
    # Should be: [1, 1, 1, 0, 0, 0] = [c_xx, c_yy, c_zz, c_xy, c_xz, c_yz]

    # Both should be identical for identity matrix
    assert torch.allclose(conic_reordered, conic_fixed)
```

### 9.2 Gradient Check: torch.autograd.gradcheck

```python
def test_metal_gradients():
    """Numerical gradient verification."""
    from torch.autograd import gradcheck

    # Simple case: 1 splat
    centers = torch.randn(1, 3, requires_grad=True, device='mps')
    Ls = torch.eye(3).unsqueeze(0).requires_grad_(True).to('mps')
    amps = torch.ones(1, requires_grad=True, device='mps')
    sharpness = torch.ones(1, requires_grad=True, device='mps')

    # Use MetalSplatFunction
    func = MetalSplatFunction.apply
    inputs = (centers, Ls, amps, sharpness, (32,32,32), 3.0, 1e-5, 4, False)

    # gradcheck compares analytical vs numerical gradients
    test = gradcheck(func, inputs, eps=1e-4, atol=1e-3)
    assert test, "Gradient check failed!"
```

### 9.3 Integration Test: Simple Optimization

```python
def test_simple_optimization():
    """Test that optimization moves splat toward target."""
    # Setup: splat at [20, 16, 16], target at [16, 16, 16]
    centers = torch.tensor([[20.0, 16.0, 16.0]], requires_grad=True, device='mps')
    Ls = torch.eye(3).unsqueeze(0).requires_grad_(True).to('mps')
    amps = torch.ones(1, requires_grad=True, device='mps')
    sharpness = torch.ones(1, requires_grad=True, device='mps')

    target = torch.zeros(32, 32, 32, device='mps')
    target[16, 16, 16] = 1.0

    # One optimization step
    output = MetalSplatFunction.apply(centers, Ls, amps, sharpness,
                                      (32,32,32), 3.0, 1e-5, 4, False)
    loss = ((output - target) ** 2).sum()
    loss.backward()

    # Check gradient direction
    print(f"Center: {centers}")
    print(f"Gradient: {centers.grad}")

    # Z gradient should be NEGATIVE (pull toward 16 from 20)
    assert centers.grad[0, 0] < 0, f"Z gradient wrong sign: {centers.grad[0, 0]}"

    # Y and X gradients should be ~0 (already aligned)
    assert abs(centers.grad[0, 1]) < 1e-4, f"Y gradient should be ~0: {centers.grad[0, 1]}"
    assert abs(centers.grad[0, 2]) < 1e-4, f"X gradient should be ~0: {centers.grad[0, 2]}"
```

---

## 10. References

### Specification
- `METAL_SPLATTING_IMPLEMENTATION_SPEC.md` §2.6 "Backward Pass"
- Gradient formulas: ∂D²/∂μ = -2·Σ⁻¹·d

### Code Files
- `kernels.metal:460-470` - Suspected bug location
- `gsplat_model_metal.py:314-343` - Backward chain rule
- `gsplat_model_metal.py:220-223` - Conic reordering (forward)
- `gsplat_model_metal.py:335` - Conic reordering (backward)

### Tests
- `test_gradient_correctness.py` - Demonstrates the bug
- `test_metal_accuracy.py:test_backward_gradients()` - Should be expanded

---

## 11. Summary

**The Bug:**
Metal backward pass computes center gradients with incorrect signs, particularly in Y dimension.

**Most Likely Cause:**
Incorrect conic permutation using `[5, 4, 2, 3, 1, 0]` instead of `[2, 1, 0, 3, 4, 5]`.

**Impact:**
- Optimization moves splats in wrong directions
- Results in elongated, misplaced gaussians
- Makes Metal backend unusable for training

**Next Steps:**
1. Verify and fix conic permutation
2. Run gradient check tests
3. Verify with simple optimization
4. Re-run full fitting pipeline

**Status:**
🚨 **BLOCKING** - Metal backend disabled until fixed.
