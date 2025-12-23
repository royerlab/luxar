# Metal Gradient Bug - Expert Handoff Document

**Created:** 2025-12-22
**Status:** Bug isolated to Metal kernel d_centers computation
**Urgency:** CRITICAL - Blocks training

---

## The Bug

Metal's `rasterize_bwd_3d` kernel produces **center gradients (d_centers) with magnitudes ~200x too large** in Y and X dimensions.

### Test Evidence

With diagonal conic (σ=2), splat at [20, 16, 16], target at [16, 16, 16]:

**Metal d_centers (raw from kernel):**
```
[Z=2.735e-01, Y=-4.14e-07, X=-1.56e-07]
```

**Expected (based on CPU):**
```
[Z=~0.27, Y=-1.86e-09, X=~-5e-10]
```

- Z: Matches ✅
- Y: **Metal is 222x too large** ❌ (sign correct, magnitude wrong)
- X: **Metal is ~300x too large** ❌ (sign correct, magnitude wrong)

### Impact

After sigmoid transformation: `raw_mu_grad = d_centers · sigmoid' · (shape-1)`
```
Metal raw_mu: [ 1.941, +7.38e-06, -1.85e-06]
CPU raw_mu:   [ 1.922, -1.44e-08, -5.48e-07]
```

During optimization, this causes:
- Centers stay stuck (saturated sigmoid)
- L matrix changes wildly (large L gradients)
- Result: **"Extremely elongated gaussians in out-of-place locations"**

---

## What We Ruled Out

✅ Conic permutation is CORRECT
✅ Forward pass is CORRECT
✅ Conic values sent to Metal are CORRECT
✅ Coordinate mappings (px, gid) are CORRECT
✅ Sigmoid derivatives are IDENTICAL between Metal/CPU
✅ raw_mu initialization is IDENTICAL
✅ d_conic→d_Ls chain rule is NOT the issue (bug persists with L frozen)

---

## Where The Bug Must Be

The bug is in **`kernels.metal` lines 390-524** (`rasterize_bwd_3d` kernel), specifically in how center gradients are computed or accumulated.

**Suspect areas:**

###1. Gradient Formula (lines 463-466)

```metal
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

**Mathematically looks correct**, but produces wrong magnitudes for Y and X.

### 2. SIMD Reduction (lines 478-484)

```metal
float3 sum_centers;
sum_centers.x = simd_sum(val_centers.x);
sum_centers.y = simd_sum(val_centers.y);
sum_centers.z = simd_sum(val_centers.z);
```

Could this be summing incorrectly? Unlikely, but possible.

### 3. Atomic Accumulation (lines 512-514)

```metal
atomic_add_float(&d_centers[splat_id * 3 + 0], sum_centers.x);
atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
atomic_add_float(&d_centers[splat_id * 3 + 2], sum_centers.z);
```

Could there be double-counting or wrong indexing?

### 4. Loop Iteration

The outer loop iterates over splats in the tile:
```metal
for (int i = 0; i < count; i++) {
    int splat_id = tile_content[start + i];
    // ...
}
```

Could pixels be processed multiple times? Or splats counted multiple times?

---

## Most Likely Causes

Given that:
- Z gradient is CORRECT
- Y and X gradients are ~200-300x too large
- The error is consistent (not random noise)

**Most likely scenarios:**

### Scenario A: Off-Diagonal Conic Terms Being Used Wrong

Even though the conic is diagonal (off-diagonal = 0), maybe the kernel is using NON-ZERO values from wrong memory locations for c_xy, c_yz, c_xz?

**Test:** Add printf to Metal kernel to print actual values of c_xy, c_yz at runtime.

### Scenario B: Distance Components (dy, dx) Are Wrong

Maybe for pixels where we expect dy=0, the kernel is computing dy ≠ 0 due to floating-point error or coordinate bug?

**Test:** Add printf to Metal kernel to print dy, dx values for several pixels.

### Scenario C: Gradient Accumulation Bug

Maybe each pixel's gradient is being counted multiple times, but only for Y and X dimensions?

**Test:** Add counter in Metal kernel to count how many times each splat accumulates gradients.

---

## Recommended Next Steps

### Step 1: Add Metal Kernel Debug Output (CRITICAL)

Modify `kernels.metal` lines 463-467 to add debug printfs:

```metal
// After computing gradients
if (splat_id == 0 && abs(val_centers.y) > 1e-10f) {  // Only print for splat 0 if Y grad non-zero
    printf("[SPLAT %d, PIXEL [%d,%d,%d]] dy=%f, c_yy=%f, c_xy=%f, c_yz=%f\n",
           splat_id, (int)px.x, (int)px.y, (int)px.z, dy, c_yy, c_xy, c_yz);
    printf("  d_D2_d_d.y (before reduction)=%f\n", d_D2_d_d.y);
    printf("  val_centers.y=%f\n", val_centers.y);
}
```

Then after SIMD reduction (line 484):
```metal
if (splat_id == 0 && simd_lane_id == 0) {
    printf("[SPLAT %d] sum_centers.y after SIMD=%f\n", splat_id, sum_centers.y);
}
```

### Step 2: Recompile and Run

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
rm src/default.metallib
python setup.py build_ext --inplace
DEBUG_METAL_GRADIENTS=1 hatch run python test_diagonal_conic_gradients.py
```

### Step 3: Analyze Output

The printf output will show:
- What dy values are actually being used
- If c_yy, c_xy, c_yz are really zero
- How val_centers.y changes through SIMD reduction
- How many pixels contribute to Y gradient

This will definitively identify the bug.

---

## Alternative: CPU Comparison Test

If Metal kernel modification is difficult, create a Python-based pixel-by-pixel comparison:

```python
# Compute expected gradient by manually iterating pixels (like CPU fallback does)
# Compare with Metal's output
# This will show which pixels Metal is processing differently
```

---

## Files Created for Expert

1. **`GRADIENT_BUG_REPORT.md`** - Initial investigation (now superseded)
2. **`FINDINGS_AND_FIX.md`** - Intermediate findings
3. **`URGENT_BUG_SUMMARY.md`** - Status summary
4. **`EXPERT_HANDOFF.md`** - This document
5. **Test files:**
   - `test_diagonal_conic_gradients.py` - Shows bug with diagonal conic
   - `test_frozen_L_gradients.py` - Proves bug is in d_centers, not chain rule
   - `check_sigmoid_derivative.py` - Proves sigmoid is identical
   - `compare_raw_gradients.py` - Shows raw_mu gradients differ
   - `test_full_optimization.py` - Shows optimization failure

---

## Current Code State

All code is in the repository at:
```
packages/luxar/src/luxar/gsplats/models/gsplats/metal/
```

**No fixes have been applied yet** - all changes so far were debug instrumentation and documentation.

The original code with permutation `[5, 4, 2, 3, 1, 0]` is restored and confirmed correct for forward pass.

**The bug is in `src/kernels.metal` backward gradient computation** - likely lines 463-466 or accumulation logic.

---

## Summary for Expert

**What works:**
- Forward pass: Perfect match with CPU
- Amplitude gradients: Match CPU
- Sharpness gradients: Match CPU
- L diagonal gradients: Match CPU
- Center Z gradient: Matches CPU (~1% error)

**What's broken:**
- Center Y gradient: 222x too large
- Center X gradient: 300x too large

**Root cause:** Metal `rasterize_bwd_3d` kernel computes or accumulates Y and X center gradients incorrectly.

**Next action:** Add Metal kernel debug printf statements to see runtime values and identify exact bug location.

