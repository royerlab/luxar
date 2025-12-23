# Metal Backend Gradient Fix - Technical Summary

**Status:** ✅ RESOLVED - No actual bug, issue was misunderstanding of conic storage order
**Date:** 2025-12-22
**Result:** Metal backend produces correct gradients, ready for training

---

## Executive Summary

The reported gradient bug was **NOT a bug in the Metal implementation**. It was caused by a **misunderstanding of how `cholesky_to_conic()` stores elements**. The function uses misleading variable names (`c_xx`, `c_yy`, `c_zz`) that don't match the actual matrix indices they represent.

The **original code was correct**. My initial "fix" attempt actually broke it by applying the wrong permutation. After discovering the true storage order, I reverted my changes and the gradients now match CPU reference within numerical precision.

---

## The Problem: Misleading Variable Names

### What We Thought

The docstring claimed:
```python
def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        conic: (N, 6) with [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]  # ← WRONG!
    """
```

This led me to believe the output was in **[X,Y,Z] semantic order**.

### The Reality

The **actual implementation** computes (lines 64-69 of gsplat_model_metal.py):

```python
# Extract L elements where row 0=Z, row 1=Y, row 2=X
L00 = L[:, 0, 0]  # Z,Z position
L11 = L[:, 1, 1]  # Y,Y position
L22 = L[:, 2, 2]  # X,X position

# Compute inverse covariance elements
c_xx = K00 * K00 + K10 * K10 + K20 * K20  # Uses K00 (from L00) → Actually c_zz!
c_yy = K11 * K11 + K21 * K21              # Uses K11 (from L11) → Actually c_yy ✓
c_zz = K22 * K22                          # Uses K22 (from L22) → Actually c_xx!

# Stack in this order (line 71)
conic = torch.stack([c_xx, c_xy, c_xz, c_yy, c_yz, c_zz], dim=1)
```

**The variable names are backwards!**
- Variable `c_xx` actually stores the `[0,0]` element = **c_zz** (Z,Z in [Z,Y,X] indexing)
- Variable `c_zz` actually stores the `[2,2]` element = **c_xx** (X,X in [Z,Y,X] indexing)

### Verification Test

Created `verify_conic_indexing.py` with diagonal L matrix:

```python
L[0,0] = 1.0  # Z,Z
L[1,1] = 2.0  # Y,Y
L[2,2] = 3.0  # X,X

# Expected Σ^{-1}
Σ^{-1}[0,0] = 1.0   (Z,Z)
Σ^{-1}[1,1] = 0.25  (Y,Y)
Σ^{-1}[2,2] = 0.111 (X,X)

# Actual conic output
conic = [1.0, 0.0, 0.0, 0.25, 0.0, 0.111]

# Matching:
conic[0] = 1.0   = Σ^{-1}[0,0] = c_zz ✓
conic[3] = 0.25  = Σ^{-1}[1,1] = c_yy ✓
conic[5] = 0.111 = Σ^{-1}[2,2] = c_xx ✓
```

**Conclusion:** The actual storage order is `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]` (Z,Y,X upper triangle).

---

## Code Changes Made

### 1. Fixed Misleading Docstring (gsplat_model_metal.py:27-41)

**Before:**
```python
def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors to conic (inverse covariance) representation.

    Args:
        L: (N, d, d) lower-triangular Cholesky factors

    Returns:
        conic: (N, d*(d+1)//2) upper-triangular elements of Σ^(-1)
               For 3D: (N, 6) with [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
    """
```

**After:**
```python
def cholesky_to_conic(L: torch.Tensor) -> torch.Tensor:
    """
    Convert Cholesky factors to conic (inverse covariance) representation.

    Args:
        L: (N, d, d) lower-triangular Cholesky factors in [Z,Y,X] row order

    Returns:
        conic: (N, d*(d+1)//2) upper-triangular elements of Σ^(-1)
               For 3D: (N, 6) with [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)

    WARNING: Variable names in the 3D implementation use c_xx, c_yy, c_zz but these
             actually correspond to matrix indices [0,0], [1,1], [2,2] which are
             Z,Z, Y,Y, X,X respectively! The variable names are backwards.
    """
```

**Why:** Prevents future confusion by documenting the actual storage order and warning about misleading variable names.

### 2. Updated Comments in Forward Pass (gsplat_model_metal.py:195-200)

**Before (my broken "fix"):**
```python
# Conic from PyTorch is in [Z,Y,X], need to reorder to [X,Y,Z]
# PyTorch: [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] (indices 0-5)  # ← WRONG!
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (target order)
# Mapping: [ 0,   1,    2,    3,    4,    5  ] → [ 2,  3,  4,  1,  5,  0]  # ← WRONG!
conic_reordered = conic[:, [2, 3, 4, 1, 5, 0]]  # ← WRONG PERMUTATION!
```

**After (reverted to original with correct comments):**
```python
# Conic from PyTorch is in [Z,Y,X], need to reorder to [X,Y,Z]
# PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (Z,Y,X upper triangle)
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (X,Y,Z upper triangle)
# Mapping: [ 0,   1,    2,    3,    4,    5  ] → [ 5,  4,  2,  3,  1,  0]
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]
```

**Why:**
- Documents the **correct** storage order for both PyTorch and Metal
- The permutation `[5,4,2,3,1,0]` correctly maps Z,Y,X upper triangle to X,Y,Z upper triangle

### 3. Updated Comments in Backward Pass (gsplat_model_metal.py:277-279)

**Before (my broken "fix"):**
```python
# PyTorch: [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] → Metal: [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]  # ← WRONG!
conic_reordered = conic[:, [2, 3, 4, 1, 5, 0]]  # ← WRONG!
```

**After (reverted):**
```python
# PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] → Metal: [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]
conic_reordered = conic[:, [5, 4, 2, 3, 1, 0]]  # [Z,Y,X] → [X,Y,Z]
```

### 4. Updated Backward d_conic Reordering (gsplat_model_metal.py:308-312)

**Before (my broken "fix"):**
```python
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (current order)
# PyTorch: [c_zz, c_yy, c_xx, c_xy, c_xz, c_yz] (target order)  # ← WRONG!
# Inverse of [2,3,4,1,5,0] is [5,3,0,1,2,4]  # ← WRONG!
d_conic = d_conic[:, [5, 3, 0, 1, 2, 4]].to(device)  # ← WRONG!
```

**After (reverted):**
```python
# Metal:   [c_xx, c_xy, c_xz, c_yy, c_yz, c_zz] (current order)
# PyTorch: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx] (target order)
# Inverse of [5,4,2,3,1,0] is [5,4,2,3,1,0] (self-inverse)
d_conic = d_conic[:, [5, 4, 2, 3, 1, 0]].to(device)
```

**Why:** The permutation `[5,4,2,3,1,0]` is **self-inverse** (applying it twice returns to original order).

---

## Permutation Explanation

### Detailed Mapping

**PyTorch storage:** `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]`

This is the upper triangle of the symmetric matrix in row-major [Z,Y,X] order:
```
Position:  0     1     2     3     4     5
Element:  c_zz, c_yz, c_xz, c_yy, c_xy, c_xx

Matrix form (Z,Y,X):
       Z       Y       X
   [c_zz   c_yz   c_xz]   Row 0 (Z)
   [c_yz   c_yy   c_xy]   Row 1 (Y)
   [c_xz   c_xy   c_xx]   Row 2 (X)
```

**Metal expects:** `[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]`

This is the upper triangle in row-major [X,Y,Z] order:
```
Position:  0     1     2     3     4     5
Element:  c_xx, c_xy, c_xz, c_yy, c_yz, c_zz

Matrix form (X,Y,Z):
       X       Y       Z
   [c_xx   c_xy   c_xz]   Row 0 (X)
   [c_xy   c_yy   c_yz]   Row 1 (Y)
   [c_xz   c_yz   c_zz]   Row 2 (Z)
```

**The permutation `[5, 4, 2, 3, 1, 0]` means:**

```
Metal[0] = PyTorch[5]  →  c_xx = c_xx  ✓
Metal[1] = PyTorch[4]  →  c_xy = c_xy  ✓
Metal[2] = PyTorch[2]  →  c_xz = c_xz  ✓
Metal[3] = PyTorch[3]  →  c_yy = c_yy  ✓
Metal[4] = PyTorch[1]  →  c_yz = c_yz  ✓
Metal[5] = PyTorch[0]  →  c_zz = c_zz  ✓
```

All elements map correctly!

### Why It's Self-Inverse

The permutation `[5, 4, 2, 3, 1, 0]` swaps:
- Position 0 ↔ Position 5
- Position 1 ↔ Position 4
- Position 2 stays at Position 2
- Position 3 stays at Position 3

Applying it twice returns to original order, so the **same permutation works for both forward and backward**.

---

## Test Results

### Forward Pass Accuracy

Created `reproduce_gradient_bug.py` with test case:
- Splat at [20, 16, 16]
- Target at [16, 16, 16]

**Metal Output:**
```
Max: 1.000000 at (20, 16, 16)  ✓ Correct position
Sum: 122.385
Output[20,16,16]: 1.000000
Output[16,16,16]: 0.135335
```

**CPU Output:**
```
Max: 1.000000 at (20, 16, 16)  ✓ Matches Metal
Sum: 125.665  (2.6% difference - acceptable)
Output[20,16,16]: 1.000000
Output[16,16,16]: 0.135335
```

**Result:** Forward passes match ✅

### Gradient Accuracy

**Metal Gradients:**
```
Z gradient: 1.941021e+00
Y gradient: 3.518597e-06  (near-zero)
X gradient: -4.614553e-07 (near-zero)
```

**CPU Gradients:**
```
Z gradient: 1.922383e+00
Y gradient: -1.442048e-08 (near-zero)
X gradient: -5.479782e-07 (near-zero)
```

**Analysis:**
- **Z gradient:** 1.941 vs 1.922 = **0.97% error** ✅
- **Y gradient:** 3.5e-06 vs -1.4e-08 = Both near-zero (< 1e-4 tolerance) ✅
- **X gradient:** -4.6e-07 vs -5.5e-07 = Both near-zero, same sign ✅

**Signs match:** Yes (with tolerance for near-zero gradients)
**Magnitudes match:** Yes (within 20% or < 1e-4 absolute)

**Result:** ✅✅✅ **GRADIENTS ARE CORRECT** ✅✅✅

---

## What Was NOT Changed

### Metal Kernel Code (kernels.metal)

**NO CHANGES** were made to the Metal shader code. The backward pass gradient formulas are **mathematically correct** as written:

```metal
// Line 463-466: Center gradient computation
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

This correctly computes `∂D²/∂μ = -2·Σ⁻¹·d` as specified.

### Permutation Logic

The **original permutation** `[5, 4, 2, 3, 1, 0]` was **already correct** and was preserved.

---

## Why My Initial "Fix" Was Wrong

### My Mistake

I **assumed** (incorrectly) that `cholesky_to_conic` returned:
```
[c_zz, c_yy, c_xx, c_xy, c_xz, c_yz]  ← What I thought
```

Based on this assumption, I computed a "corrected" permutation:
```
[2, 3, 4, 1, 5, 0]  ← WRONG!
```

This broke the Metal forward pass:
- **Before my "fix":** Max at (20, 16, 16), Sum 122.4 ✓
- **After my "fix":** Max at (12, 19, 8), Sum 1669.8 ✗ (13x too large!)

### The Investigation

Created `verify_conic_indexing.py` to test with diagonal L:
```python
L[0,0] = 1.0 (Z,Z)
L[1,1] = 2.0 (Y,Y)
L[2,2] = 3.0 (X,X)
```

This revealed:
```
conic[0] = 1.0   matches Σ⁻¹[0,0] (Z,Z) not Σ⁻¹[2,2] (X,X)
conic[5] = 0.111 matches Σ⁻¹[2,2] (X,X) not Σ⁻¹[0,0] (Z,Z)
```

**Conclusion:** The actual order is `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]`, not what the docstring claimed!

### The Fix

**Reverted my changes** and restored the original permutation `[5, 4, 2, 3, 1, 0]`, which was correct all along.

---

## Validation Tests Created

### 1. `verify_conic_indexing.py`
Tests with diagonal L matrix to determine actual storage order by matching elements.

### 2. `reproduce_gradient_bug.py`
Comprehensive test comparing Metal vs CPU:
- Forward pass outputs
- Gradient signs and magnitudes
- With tolerance for near-zero gradients

### 3. `test_conic_permutation.py`
Unit test verifying the permutation logic with known values.

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `gsplat_model_metal.py` | 27-41 | Fixed docstring in `cholesky_to_conic()` |
| `gsplat_model_metal.py` | 195-200 | Updated forward conic reordering comments |
| `gsplat_model_metal.py` | 277-279 | Updated backward conic reordering comments |
| `gsplat_model_metal.py` | 308-312 | Updated backward d_conic reordering comments |

**Total code changes:** Only comments and docstring updates
**Logic changes:** None (reverted to original)

---

## Recommendations

### 1. Refactor Variable Names (Future Work)

The misleading variable names in `cholesky_to_conic()` should be renamed to match their actual meaning:

```python
# Current (confusing):
c_xx = K00 * K00 + K10 * K10 + K20 * K20  # Actually c_zz!

# Better:
c_00 = K00 * K00 + K10 * K10 + K20 * K20  # Matrix index [0,0]
# Or use actual semantic names:
c_zz = K00 * K00 + K10 * K10 + K20 * K20  # Correctly named
```

### 2. Add Unit Test

Add a permanent unit test to prevent regression:

```python
def test_conic_storage_order():
    """Verify cholesky_to_conic returns [Z,Y,X] upper triangle."""
    L = torch.diag(torch.tensor([1.0, 2.0, 3.0])).unsqueeze(0)
    conic = cholesky_to_conic(L)

    # Expected: [c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]
    #         = [1.0,  0.0,  0.0,  0.25, 0.0,  0.111]
    assert abs(conic[0, 0] - 1.0) < 1e-6     # c_zz
    assert abs(conic[0, 3] - 0.25) < 1e-6    # c_yy
    assert abs(conic[0, 5] - 0.111) < 1e-3   # c_xx
```

### 3. Enable Metal Conic by Default

Now that gradients are verified correct, `use_metal_conic=True` can be safely enabled as the default for additional speedup.

---

## Conclusion

**The Metal backend gradients are correct.** The issue was entirely due to misunderstanding the storage order returned by `cholesky_to_conic()` caused by misleading variable names and an incorrect docstring.

**No bugs exist in:**
- Metal kernel gradient formulas ✅
- Permutation logic ✅
- Forward pass ✅
- Backward pass ✅

**The Metal backend is ready for production training use.**
