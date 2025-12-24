# Metal Gradient Bug Investigation - Complete Code Changes Summary

**Investigation Date:** 2025-12-22
**Status:** Bug NOT fixed - Metal Y and X gradients have opposite signs from CPU
**Files Modified:** 2 files
**Tests Created:** 15 files

---

## Code Changes Made

### File 1: `gsplat_model_metal.py`

#### Change 1.1: Fixed Misleading Docstring (Lines 27-41)

**Before:**
```python
def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        conic: (N, 6) with [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]  # WRONG!
    """
```

**After:**
```python
def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        conic: (N, 6) with [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)

    WARNING: Variable names in the 3D implementation use c_xx, c_yy, c_zz but these
             actually correspond to matrix indices [0,0], [1,1], [2,2] which are
             Z,Z, Y,Y, X,X respectively! The variable names are backwards.
    """
```

**Purpose:** Prevent future confusion about conic storage order

#### Change 1.2: Updated Conic Reordering Comments (Lines 195-200)

**Before:** Incorrect comment showing wrong PyTorch order

**After:**
```python
# Conic from PyTorch is in [Z,Y,X], need to reorder to [X,Y,Z]
# PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (X,Y,Z upper triangle)
# Mapping: [ 0,   1,    2,    3,    4,    5  ] → [ 5,  4,  2,  3,  1,  0]
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
```

**Purpose:** Document correct permutation mapping

#### Change 1.3: Added Debug Instrumentation (Lines 309-399)

**Added code:**
```python
# === DEBUG PROBE: Check raw Metal output ===
import os
if os.environ.get('DEBUG_METAL_GRADIENTS'):
    print("DEBUG: RAW METAL BACKWARD OUTPUT")
    # ...prints d_centers, d_conic before/after reorder, chain rule inputs/outputs...
```

**Purpose:** Enable runtime debugging with `DEBUG_METAL_GRADIENTS=1`

**Note:** This code is CONDITIONAL (only runs when env var is set) and can be removed for production.

---

### File 2: `src/kernels.metal`

#### Change 2.1: Removed SIMD Reduction (Lines 476-508)

**Before:**
```metal
// === B. SIMD Reduction (sum across 32 threads in simdgroup) ===
float sum_amps = simd_sum(val_amps);
// ...
sum_centers.y = simd_sum(val_centers.y);
// ...

// === C. Leader writes to global memory (lane 0 only) ===
if (simd_lane_id == 0) {
    atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
}
```

**After:**
```metal
// === DIRECT ATOMIC WRITE (no SIMD reduction) ===
// Each thread atomically adds its own gradient contribution
// Guard against NaN
if (!has_nan) {
    atomic_add_float(&d_centers[splat_id * 3 + 1], val_centers.y);
    // ...all threads do atomic_add, not just lane 0...
}
```

**Purpose:** Attempted to fix suspected race condition
**Result:** ❌ Did not fix the bug

#### Change 2.2: Explicit Distance Calculation (Line 427-429)

**Before:**
```metal
float3 d = px - c;
float dz = d.x, dy = d.y, dx = d.z;
```

**After:**
```metal
float3 d = px - c;
// EXPLICIT distance calculation (avoid float3 component confusion)
float dz = px.x - c.x;  // Z_pixel - Z_center
float dy = px.y - c.y;  // Y_pixel - Y_center
float dx = px.z - c.z;  // X_pixel - X_center
```

**Purpose:** Avoid potential float3 component mapping issues
**Result:** ❌ Did not fix the bug

---

## Test Files Created (15 total)

**Core diagnostic tests:**
1. `compare_with_cpu_rendering.py` - Direct Metal vs CPU gradient comparison ⭐ **KEY TEST**
2. `test_diagonal_conic_gradients.py` - Simplified diagonal conic test
3. `test_frozen_L_gradients.py` - Isolates d_centers bug (freezes L)
4. `python_reference_backward.py` - Python implementation of Metal logic

**Supporting tests:**
5. `compare_raw_gradients.py` - Shows raw_mu gradient differences
6. `test_full_optimization.py` - Demonstrates optimization failure
7. `diagnose_gradient_application.py` - Shows saturated sigmoid issue
8. `reproduce_gradient_bug.py` - Original bug reproduction
9. `test_conic_permutation.py` - Verifies permutation correctness
10. `verify_conic_indexing.py` - Confirms conic storage order
11. `inspect_metal_conic_input.py` - Validates conic values sent to Metal
12. `debug_conic_values.py` - Conic value inspection
13. `check_sigmoid_derivative.py` - Verifies sigmoid transformation
14. `test_coordinate_mapping.py` - Tests coordinate system
15. `test_gradient_correctness.py` - Original gradient validation

**Documentation files:**
16. `GRADIENT_BUG_REPORT.md` - Initial investigation
17. `GRADIENT_FIX_SUMMARY.md` - First fix attempt (later found wrong)
18. `CURRENT_STATUS.md` - Investigation status
19. `URGENT_BUG_SUMMARY.md` - Summary for quick reference
20. `FINDINGS_AND_FIX.md` - Detailed findings
21. `EXPERT_HANDOFF.md` - Handoff document
22. `RACE_CONDITION_FIX.md` - Race condition analysis
23. `FINAL_STATUS_FOR_EXPERT.md` - Final analysis
24. `CODE_CHANGES_SUMMARY.md` - This document

---

## Current Bug Status

### What's Wrong

**Metal produces center gradients with OPPOSITE SIGNS in Y and X:**

```
Test: Diagonal conic, splat [20, 16, 16], target [16, 16, 16]

CPU d_centers:   [Z=2.707e-01, Y=+2.310e-07, X=+1.863e-09]  ← CORRECT
Metal d_centers: [Z=2.707e-01, Y=-1.213e-05, X=-1.699e-06]  ← WRONG

Errors:
- Y: OPPOSITE SIGN (Metal negative, should be positive) + 52x too large
- X: OPPOSITE SIGN (Metal negative, should be positive) + 912x too large
- Z: Perfect match ✅
```

### Impact

During SGD optimization:
```python
center_y_new = center_y - lr * gradient_y

CPU:  center_y_new = 16 - 0.1 * (+2.3e-07) ≈ 16 (barely moves) ✓
Metal: center_y_new = 16 - 0.1 * (-1.2e-05) = 16 + 1.2e-06 (moves WRONG way) ✗
```

Over many iterations, wrong signs cause:
- Centers move away from targets
- L matrix compensates → extreme elongation
- Result: "Elongated gaussians in out-of-place locations"

---

## What We Ruled Out

✅ Conic permutation - verified correct with multiple tests
✅ Conic values - inspected and confirmed correct
✅ Forward pass - matches CPU perfectly
✅ SIMD reduction race - removed it, bug persists
✅ Float3 component extraction - made explicit, bug persists
✅ Coordinate system (px, gid) - verified correct
✅ Grid dispatch - verified correct
✅ Sigmoid transformation - identical between Metal/CPU
✅ L diagonal gradients - match perfectly
✅ Amplitude gradients - match perfectly

---

## What We Have NOT Ruled Out

### Hypothesis 1: Gradient Formula Sign Error

Maybe there's a subtle sign error in lines 463-465 that only affects Y and X?

**Current formula:**
```metal
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
val_centers.y = grad_dist * d_D2_d_d.y * -1.0f;
```

**Proposed test:** Try inverting the final sign:
```metal
val_centers.y = grad_dist * d_D2_d_d.y * +1.0f;  // Remove the -1.0f
```

If this fixes Y and X but breaks Z, we know there's a sign convention issue.

### Hypothesis 2: Conic Matrix Order Mismatch

Maybe when computing (Σ⁻¹ · d)_y, we need different matrix elements than we think?

**Current:** Assumes conic in [X,Y,Z] order, multiplies by d in [Z,Y,X] order
**Maybe:** There's a mismatch in how matrix-vector product is computed?

### Hypothesis 3: PyTorch Autograd vs Manual Gradient

Maybe PyTorch's autograd computes gradients differently than our manual formulas?

**Evidence:** Z gradient matches → formula works for some cases
**Counter:** But Y and X don't match → formula fails for other cases?

---

## Recommended Next Action

**Try changing line 466 from:**
```metal
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

**To:**
```metal
val_centers.x = grad_dist * d_D2_d_d.x * -1.0f;  // Keep Z with -1.0
val_centers.y = grad_dist * d_D2_d_d.y * +1.0f;  // Try Y with +1.0
val_centers.z = grad_dist * d_D2_d_d.z * +1.0f;  // Try X with +1.0
```

This will test if there's a sign convention difference for Y and X vs Z.

---

## How to Test the Next Fix

```bash
cd /Users/loic.royer/workspace/python/luxar
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
# 1. Edit src/kernels.metal (apply proposed change)
# 2. Recompile
python setup.py build_ext --inplace
# 3. Test
cd /Users/loic.royer/workspace/python/luxar
hatch run python packages/luxar/src/luxar/gsplats/models/gsplats/metal/compare_with_cpu_rendering.py
```

**Success criteria:**
- Y and X signs match CPU
- Y and X magnitudes within 5x of CPU
- Z gradient still matches

---

## Summary for Expert

**Problem:** Metal center gradients for Y and X have opposite signs from CPU

**Root cause:** Unknown - formula looks mathematically correct but produces wrong signs

**Attempted fixes:**
1. Removed SIMD reduction → no effect
2. Explicit distance calculation → no effect

**Remaining hypothesis:** Sign convention error in gradient formula for non-primary dimensions

**Files to examine:**
- `src/kernels.metal` lines 463-466 (gradient formula)
- `src/kernels.metal` line 427-429 (distance extraction)

**Next test:** Try inverting sign for Y and X gradients only (see recommended action above)

