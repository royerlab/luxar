# Metal Gradient Bug - Complete Investigation Summary for Expert

**Date:** 2025-12-22
**Status:** 🚨 **BUG CONFIRMED - Metal produces gradients with OPPOSITE SIGNS**
**Urgency:** CRITICAL - Blocks all training

---

## The Bug in One Sentence

Metal's `rasterize_bwd_3d` kernel produces center gradients with **opposite signs in Y and X dimensions** compared to CPU, causing optimization to move splats in the wrong direction.

---

## Test Evidence

**Test case:** Diagonal conic (σ=2), splat at [20, 16, 16], target at [16, 16, 16]

**CPU d_centers (CORRECT):**
```
[Z=2.707e-01, Y=+2.310e-07, X=+1.863e-09]
```

**Metal d_centers (WRONG):**
```
[Z=2.707e-01, Y=-4.470e-06, X=-9.034e-07]
```

**Analysis:**
- **Z gradient:** ✅ Matches perfectly (1.000x ratio)
- **Y gradient:** ❌ **OPPOSITE SIGN** (Metal negative, CPU positive) + 19x too large
- **X gradient:** ❌ **OPPOSITE SIGN** (Metal negative, CPU positive) + 485x too large

---

## What This Causes

During optimization:
```python
new_center_y = old_center_y - lr * gradient_y

# CPU (correct): gradient_y = +small → center decreases slightly or stays
# Metal (wrong): gradient_y = -small → center INCREASES

→ Metal moves splats in WRONG DIRECTION!
```

Result: **"Extremely elongated gaussians in out-of-place locations"** (user's original report)

---

## Investigation Summary

### Wh at We Verified ✅

1. Forward pass: Metal and CPU outputs match perfectly
2. Conic storage order: Confirmed [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]
3. Conic permutation [5,4,2,3,1,0]: Mathematically correct
4. Conic values sent to Metal: Verified correct (diagonal = [0.25, 0.25, 0.25])
5. Coordinate system (px, gid): Mapped correctly
6. Z gradient: Perfect match between Metal and CPU
7. L diagonal gradients: Match between Metal and CPU
8. Amplitude gradients: Match between Metal and CPU

### What's Wrong ❌

**Only center Y and X gradients have opposite signs + wrong magnitudes.**

This is EXTREMELY SPECIFIC - not a random bug, but a systematic error affecting only Y and X center gradients.

---

## Code Changes Attempted

### Change 1: Removed SIMD Reduction

**File:** `kernels.metal` lines 476-508
**What:** Replaced SIMD reduction + atomic pattern with direct per-thread atomic adds
**Result:** ❌ Did not fix the bug
**Why attempted:** Suspected race condition due to SIMD synchronization
**Why failed:** Bug is not in accumulation, but in gradient computation itself

### Change 2: Fixed Misleading Docstrings

**File:** `gsplat_model_metal.py` lines 27-41
**What:** Documented actual conic storage order
**Result:** ⚠️ Clarification only, no functional change

### Change 3: Added Debug Probes

**File:** `gsplat_model_metal.py` lines 309-399
**What:** Added conditional debug output with `DEBUG_METAL_GRADIENTS=1`
**Result:** ✅ Revealed the exact nature of the bug

---

## The Mystery: Why Only Y and X?

**Critical observation:** Z gradient is PERFECT, but Y and X are completely wrong.

**What's special about Z?**
- In the test, splat is offset only in Z (at Z=20, target at Z=16)
- Y and X are aligned (both at 16)

**Possible explanations:**

### Theory A: Coordinate System Swap

Maybe `gid.y` and `gid.x` meanings are swapped somewhere? But this would affect ALL dimensions, not just Y and X.

### Theory B: Conic Element Misordering

Maybe c_yy and c_xy are being loaded from wrong positions? But we verified the permutation is correct.

### Theory C: Formula Sign Error

Maybe there's a subtle sign error in the formula that only affects off-center dimensions?

### Theory D: Distance Calculation Bug

Maybe `dy` and `dx` are computed with wrong signs?

**Check:** In line 427:
```metal
float dz = d.x, dy = d.y, dx = d.z;
```

And `d = px - c` (line 419).

For pixel at [z, y, x] and splat at [20, 16, 16]:
- d.x = px.x - c.x = z - 20
- d.y = px.y - c.y = y - 16
- d.z = px.z - c.z = x - 16

This looks correct...

---

## Recommended Next Steps for Expert

Given the specificity of the bug (only Y and X, not Z), I recommend:

### 1. Check Coordinate Extraction (Line 427)

```metal
float dz = d.x, dy = d.y, dx = d.z;  // Line 427
```

**Verify:** Is this extraction correct for float3 in [Z,Y,X] semantic order?

**Test:** Try swapping to:
```metal
float dz = d.z, dy = d.y, dx = d.x;  // Swap Z and X
```

If this fixes Y and X but breaks Z, we know the float3 component mapping is wrong.

### 2. Check px Construction (Line 399)

```metal
float3 px = float3(gid.z, gid.y, gid.x);  // [Z,Y,X]
```

**Verify:** Does float3(a, b, c) actually put 'a' in .x, 'b' in .y, 'c' in .z?

**Test:** Try:
```metal
float3 px = float3(gid.x, gid.y, gid.z);  // Try [X,Y,Z] instead
```

### 3. Check Center Loading (Lines 413-417)

```metal
float3 c = float3(
    centers[splat_id * 3 + 0],  // Z
    centers[splat_id * 3 + 1],  // Y
    centers[splat_id * 3 + 2]   // X
);
```

**Verify:** Are centers stored as [Z, Y, X] in the buffer?

**Test:** Print centers buffer content and verify order.

---

## Most Likely Root Cause

Based on the evidence, **Theory D (Distance Calculation)** seems most likely:

**The signs of `dy` or `dx` might be inverted.**

If `dy` has the wrong sign, then:
- Pixels with y > center contribute gradient with wrong sign
- Pixels with y < center contribute gradient with wrong sign
- These don't cancel out → net wrong sign

**Proposed Fix:** Add explicit sign check or recompute d components:

```metal
// Instead of this (line 427):
float dz = d.x, dy = d.y, dx = d.z;

// Try this:
float dz = px.x - c.x;  // Explicit: Z_pixel - Z_center
float dy = px.y - c.y;  // Explicit: Y_pixel - Y_center
float dx = px.z - c.z;  // Explicit: X_pixel - X_center
```

This bypasses any potential float3 component confusion.

---

## Files for Expert Review

All investigation in: `packages/luxar/src/luxar/gsplats/models/gsplats/metal/`

**Key documents:**
1. `FINAL_STATUS_FOR_EXPERT.md` - This document
2. `RACE_CONDITION_FIX.md` - Attempted fix (didn't work)
3. `EXPERT_HANDOFF.md` - Earlier analysis

**Critical test files:**
1. `compare_with_cpu_rendering.py` - Shows opposite signs
2. `test_diagonal_conic_gradients.py` - Simplified test case
3. `python_reference_backward.py` - Manual calculation

**Code locations:**
- Bug location: `src/kernels.metal` lines 427 or 463-466
- Suspected: float3 component extraction or distance calculation

---

## Summary

**Status:** After extensive investigation and one attempted fix, the bug persists.

**The bug:** Metal center gradients for Y and X dimensions have **opposite signs** and wrong magnitudes.

**Most likely cause:** Error in how `dy` or `dx` are extracted from float3, or how `px` is constructed.

**Next action:** Try the proposed coordinate calculation fix above.

