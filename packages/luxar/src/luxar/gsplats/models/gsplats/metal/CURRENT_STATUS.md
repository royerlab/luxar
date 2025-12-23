# Metal Backend Gradient Bug - Current Status

**Date:** 2025-12-22
**Status:** 🚨 **CONFIRMED BUG** - Metal produces incorrect raw_mu gradients

---

## Summary

The Metal backward pass produces **incorrect center gradients** with **wrong signs**, confirmed by direct comparison with CPU reference:

```
Test case: Center at [25, 16, 16], Target at [16, 16, 16]

Metal raw_mu gradient: [-5.8e-07, -9.1e-06, +4.4e-06]
CPU raw_mu gradient:   [-1.6e-06, +1.3e-06, +2.9e-08]

Signs differ in Y and X dimensions!
```

This causes optimization failure: gaussians become elongated and stay in wrong positions because center gradients have wrong signs while L/amplitude gradients are correct and large.

---

## What We Know

### ✅ Verified CORRECT

1. **Forward pass:** Metal and CPU outputs match
2. **Conic storage order:** Confirmed `[c_zz, c_yz, c_xz, c_yy, c_xy, c_xx]` (Z,Y,X upper triangle)
3. **Conic permutation:** `[5, 4, 2, 3, 1, 0]` correctly maps to Metal's expected `[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]`
4. **L and amplitude gradients:** Match between Metal and CPU
5. **Sharpness gradients:** Match between Metal and CPU

### ❌ BROKEN

1. **raw_mu (center) gradients:** Metal produces wrong signs in Y and X
2. **L_off gradients:** Also differ (small values, might be related)

---

## Investigation Needed

The bug is in **Metal's backward pass center gradient computation**. Suspects:

### 1. Gradient Formula in kernels.metal (lines 463-466)

```metal
d_D2_d_d.x = 2.0f * (dz * c_zz + dy * c_yz + dx * c_xz);  // ∂D²/∂z
d_D2_d_d.y = 2.0f * (dz * c_yz + dy * c_yy + dx * c_xy);  // ∂D²/∂y
d_D2_d_d.z = 2.0f * (dz * c_xz + dy * c_xy + dx * c_xx);  // ∂D²/∂x
val_centers = grad_dist * d_D2_d_d * -1.0f;
```

**Possible issues:**
- Conic elements accessed in wrong order?
- Float3 component assignment incorrect?
- Sign error in formula?

### 2. Conic Element Loading (lines 422-424)

```metal
int cb = splat_id * 6;
float c_xx = conic[cb + 0], c_xy = conic[cb + 1], c_xz = conic[cb + 2];
float c_yy = conic[cb + 3], c_yz = conic[cb + 4], c_zz = conic[cb + 5];
```

Is Metal actually receiving conic in the expected order `[c_xx, c_xy, c_xz, c_yy, c_yz, c_zz]`?

### 3. Distance Vector (d) Computation (line 419, 427)

```metal
float3 d = px - c;  // Line 419
float dz = d.x, dy = d.y, dx = d.z;  // Line 427
```

Are dz, dy, dx extracted correctly from float3?

### 4. Gradient Output Ordering (lines 512-514)

```metal
atomic_add_float(&d_centers[splat_id * 3 + 0], sum_centers.x);
atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
atomic_add_float(&d_centers[splat_id * 3 + 2], sum_centers.z);
```

Does Python expect gradients in [Z,Y,X] order at positions [0,1,2]?

---

## Debugging Strategy

### Phase 1: Isolate the Bug (In Progress)

1. ✅ Confirm bug exists (done - gradients differ)
2. ⏳ Test with simple diagonal conic (no off-diagonal terms)
3. ⏳ Test each dimension independently
4. ⏳ Print intermediate Metal kernel values (d, conic elements, d_D2_d_d)

### Phase 2: Fix and Validate

1. Apply fix to Metal kernel
2. Re-run gradient comparison
3. Test full optimization loop
4. Validate with real fitting workload

---

## Test Files Created

1. **`compare_raw_gradients.py`** - Shows Metal vs CPU gradient mismatch ✅
2. **`reproduce_gradient_bug.py`** - Tests single gradient step
3. **`test_full_optimization.py`** - Shows optimization failure
4. **`diagnose_gradient_application.py`** - Reveals saturated sigmoid issue
5. **`verify_conic_indexing.py`** - Confirms conic storage order

---

## Next Steps

1. **Add debug prints to Metal kernel** to see actual values during backward pass
2. **Test with diagonal conic only** to simplify (set all c_xy, c_xz, c_yz = 0)
3. **Test each coordinate independently** (vary only Z, then only Y, then only X)
4. **Compare intermediate values** between Metal kernel and CPU implementation

---

## Key Files

- **Metal kernel:** `packages/luxar/src/luxar/gsplats/models/gsplats/metal/src/kernels.metal`
  - Forward: lines 304-367
  - Backward: lines 390-524
  - Gradient formula: lines 463-466

- **Python wrapper:** `packages/luxar/src/luxar/gsplats/models/gsplats/metal/gsplat_model_metal.py`
  - Forward conic reordering: lines 195-200
  - Backward conic reordering: lines 277-279
  - Gradient chain rule: lines 314-343

- **CPU reference:** `packages/luxar/src/luxar/gsplats/models/gsplats/gsplat_model.py`

---

## User's Original Report

> "I ran an optimisation and the result shows some extremely elongated gaussians that are 'out-of-place'. This tells me that there are still bugs!"

**Explanation:** Wrong center gradients → centers can't move correctly → L matrix changes wildly → elongated gaussians in wrong locations.

**Status:** User was RIGHT - there IS a bug in the Metal gradients!
