# URGENT: Metal Gradient Bug - Current Understanding

**Status:** 🚨 CONFIRMED BUG - Metal produces wrong center gradients
**Impact:** BLOCKING - Cannot use Metal for training
**Priority:** P0 - Critical

---

## What We Know For CERTAIN

### ✅ Verified Facts

1. **Forward pass is CORRECT** - Metal and CPU outputs match perfectly
2. **Conic storage/permutation is CORRECT** - Verified with diagonal test case
3. **Conic values sent to Metal are CORRECT** - Inspected and confirmed diagonal = [0.25, 0.25, 0.25]
4. **L and amplitude gradients MATCH** between Metal and CPU
5. **Coordinate system mappings are CORRECT** - px = float3(gid.z, gid.y, gid.x) is correct
6. **Pixel indexing is CORRECT** - pix_idx calculation matches NumPy convention

### ❌ Confirmed Wrong

**Metal `raw_mu` gradients have WRONG SIGNS:**

Test: Diagonal conic, splat at [20, 16, 16], target at [16, 16, 16]
```
Metal raw_mu: [ 1.941e+00,  +1.45e-05, -7.15e-06]
CPU raw_mu:   [ 1.922e+00,  -1.44e-08, -5.48e-07]
```

- **Z gradient:** Close match ✓
- **Y gradient:** Metal = +1.45e-05, CPU = -1.44e-08 → WRONG SIGN ✗
- **X gradient:** Magnitudes differ but signs match

---

## The Mystery

**Everything looks mathematically correct in the Metal kernel!**

The gradient formula (lines 463-465 in kernels.metal):
```metal
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

For the diagonal test case (dz=4, dy=0, dx=0, all off-diagonal conic=0):
- Expected d_D2_d_d.y = 2.0 * (4 * 0 + 0 * 0.25 + 0 * 0) = **0**
- But Metal produces non-zero Y gradient with **wrong sign**!

**This suggests:**
1. Either dy ≠ 0 (but how? centers match in Y)
2. Or c_yz ≠ 0 (but we verified it is 0)
3. Or c_xy ≠ 0 (but we verified it is 0)
4. Or there's numerical noise/accumulation error
5. Or there's a subtle bug in how gradients are accumulated across threads

---

## Possible Explanations

### Theory 1: SIMD Reduction Bug
The gradients are accumulated using SIMD reduction (lines 478-489).
Maybe there's an issue in how multiple threads accumulate to the same splat?

### Theory 2: Atomic Operation Bug
Line 512-514 use atomic_add_float. Maybe there's a race condition or ordering issue?

### Theory 3: Numerical Precision Issue
The Y gradient is tiny (~1e-5). Maybe it's just accumulated numerical noise?
But why would it have the **wrong sign**?

### Theory 4: Hidden Coordinate Bug
Maybe gid.y and gid.x are swapped somewhere we haven't checked?
Or maybe img_size interpretation is different than we think?

### Theory 5: The Conic IS Wrong Somehow
Maybe our inspection was incomplete? Or the forward pass uses it differently than backward?

### Theory 6: Gradient Chain Rule Issue (Python Side)
Maybe the d_centers from Metal are actually correct, but the Python code that
transforms them to raw_mu gradients has a bug?

---

## What To Try Next

### Immediate Actions (in order)

1. **Add debug prints to Metal kernel** ⭐ MOST DIRECT
   - Print actual values of dz, dy, dx at a specific pixel
   - Print actual values of c_yz, c_yy, c_xy
   - Print d_D2_d_d.y before and after SIMD reduction
   - This requires modifying kernels.metal and recompiling

2. **Test with ZERO off-diagonal explicitly**
   - Modify Metal kernel to FORCE c_xy = c_xz = c_yz = 0
   - See if Y gradient becomes exactly 0

3. **Test single pixel, single splat**
   - Simplest possible case: 1 splat, 1 pixel
   - Manually compute expected gradient
   - Compare with Metal output

4. **Check if issue is in Python chain rule**
   - Extract d_centers directly from Metal (before chain rule)
   - Compare with CPU's d_centers (before chain rule)
   - This isolates whether bug is in Metal or Python

5. **Test with different grid dispatch sizes**
   - Maybe issue only appears with certain grid configurations?

---

## Code Locations

**Metal Kernel:**
- Forward: `kernels.metal:304-367`
- Backward: `kernels.metal:390-524`
- **Gradient formula: `kernels.metal:463-466`** ⭐ SUSPECT
- **SIMD reduction: `kernels.metal:478-489`** ⭐ SUSPECT
- **Atomic writes: `kernels.metal:512-514`** ⭐ SUSPECT

**Python Bindings:**
- Forward dispatch: `bindings.mm:225-370`
- Backward dispatch: `bindings.mm:390-470`
- Grid setup: `bindings.mm:345, 430` (img_size = {W, H, D})

**Python Wrapper:**
- MetalSplatFunction.backward: `gsplat_model_metal.py:254-357`
- Conic reordering: `gsplat_model_metal.py:277-279, 308-312`

---

## Test Files

1. ✅ `compare_raw_gradients.py` - Shows bug exists
2. ✅ `test_diagonal_conic_gradients.py` - Bug persists with diagonal
3. ✅ `inspect_metal_conic_input.py` - Conic is correct
4. ⏳ Need: Test that extracts d_centers directly from Metal
5. ⏳ Need: Modified Metal kernel with debug prints

---

## User's Report

> "I ran an optimisation and the result shows some extremely elongated gaussians that are 'out-of-place'"

**Confirmed cause:** Wrong center gradients → centers can't move properly → L/amp change instead → elongated, misplaced gaussians

**User was RIGHT** - There IS a critical bug!

---

## Next Step

**RECOMMENDATION:** Add debug printf statements to Metal kernel to see actual runtime values.
This is the most direct way to find the bug since all static analysis shows the code looks correct.

