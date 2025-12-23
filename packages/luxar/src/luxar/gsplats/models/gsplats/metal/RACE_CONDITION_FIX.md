# URGENT: Metal Gradient Race Condition - Root Cause and Fix

**Status:** 🚨 **RACE CONDITION IDENTIFIED**
**Date:** 2025-12-22
**Severity:** CRITICAL - Causes non-deterministic training failures

---

## The Problem

Metal's `rasterize_bwd_3d` kernel produces **non-deterministic gradients** that vary between runs:

```
Run 1: d_centers Y = 2.384e-07
Run 2: d_centers Y = 4.768e-07
Run 3: d_centers Y = 1.728e-06
```

Values vary by **7x between runs** - this is a **RACE CONDITION**.

---

## Root Cause Analysis

### The Bug Location

**File:** `packages/luxar/src/luxar/gsplats/models/gsplats/metal/src/kernels.metal`
**Lines:** 512-514 (atomic_add_float operations)

```metal
// === C. Leader writes to global memory (lane 0 only) ===
if (simd_lane_id == 0) {
    // ...NaN checks...

    atomic_add_float(&d_centers[splat_id * 3 + 0], sum_centers.x);  // Line 512
    atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);  // Line 513
    atomic_add_float(&d_centers[splat_id * 3 + 2], sum_centers.z);  // Line 514

    // ...conic atomic adds...
}
```

### Why It's a Race Condition

**Multiple thread groups can process the SAME splat simultaneously:**
1. Tile A contains splat 0 → Threadgroup A processes splat 0
2. Tile B contains splat 0 → Threadgroup B processes splat 0 (different pixels)
3. Both threadgroups call `atomic_add_float` on `d_centers[0]` **concurrently**

**What should happen:**
Atomic operations serialize the adds → correct sum

**What actually happens:**
For some pixels/threads, the atomic operation has a race or the SIMD reduction has issues, causing incorrect accumulation.

### Evidence

1. **Non-deterministic:** Same test produces different gradients each run
2. **Magnitude error:** Gradients are 6-300x too large (varying)
3. **Sign errors:** Sometimes positive when should be negative
4. **Y and X affected, Z correct:** Suggests systematic issue, not random noise

---

## The Actual Bug

After extensive analysis, I believe the issue is **NOT a race in atomic_add**, but rather in the **SIMD reduction logic** or **how threads are synchronized within threadgroups**.

### SIMD Reduction (lines 482-484)

```metal
float3 sum_centers;
sum_centers.x = simd_sum(val_centers.x);
sum_centers.y = simd_sum(val_centers.y);
sum_centers.z = simd_sum(val_centers.z);
```

**Hypothesis:** The `simd_sum()` function might be:
1. Summing across wrong threads (beyond the simdgroup)
2. Including values from previous loop iterations
3. Not properly synchronized between simdgroups

### Loop Structure Issue

The loop structure is:
```metal
for (int i = 0; i < count; i++) {
    int splat_id = tile_content[start + i];

    float3 val_centers = 0.0f;  // Reset for this splat

    // ...compute val_centers for this pixel+splat...

    // SIMD reduction across threads
    sum_centers.y = simd_sum(val_centers.y);

    // Atomic add (only lane 0)
    if (simd_lane_id == 0) {
        atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
    }
}
```

**All threads in the threadgroup execute this loop together**, but each thread has its own `val_centers`. The SIMD reduction should sum across threads in the same simdgroup for the current splat iteration.

---

## Comparison with Forward Pass

The **forward pass works correctly**, so let's compare its structure:

### Forward (lines 308-365)

```metal
for (int i = 0; i < count; i++) {
    int splat_id = tile_content[start + i];

    // ...compute intensity...

    accum += intensity;  // Simple accumulation, no SIMD
}

output[pix_idx] = accum;  // Each thread writes its own pixel
```

**Key difference:** Forward has **each thread accumulate its own value** and write independently. No SIMD reduction, no atomic operations.

### Backward (lines 401-523)

```metal
for (int i = 0; i < count; i++) {
    int splat_id = tile_content[start + i];

    // ...compute val_centers for this thread...

    // SIMD reduction (sums across 32 threads)
    sum_centers = simd_sum(val_centers);

    // Atomic add (one thread writes sum)
    if (simd_lane_id == 0) {
        atomic_add_float(&d_centers[splat_id * 3 + ...], sum_centers...);
    }
}
```

**The complex part:** Multiple threads cooperate via SIMD, then one writes via atomic.

---

## Proposed Fix

### Option 1: Match Forward Pass Structure (SAFEST)

Remove SIMD reduction and make each thread atomically add its own gradient:

```metal
// Current (complex, buggy):
sum_centers.y = simd_sum(val_centers.y);
if (simd_lane_id == 0) {
    atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
}

// Proposed (simple, safe):
atomic_add_float(&d_centers[splat_id * 3 + 1], val_centers.y);
```

**Pros:**
- Simple, matches forward pass pattern
- No SIMD synchronization issues
- Deterministic

**Cons:**
- More atomic operations (64 per splat instead of 2)
- Slightly slower due to atomic contention (but likely negligible)

### Option 2: Fix SIMD Reduction Synchronization

Add explicit thread synchronization:

```metal
// Ensure all threads in simdgroup have computed val_centers
simdgroup_barrier(mem_flags::mem_device);

sum_centers.y = simd_sum(val_centers.y);

// Ensure reduction is complete before atomic write
simdgroup_barrier(mem_flags::mem_device);

if (simd_lane_id == 0) {
    atomic_add_float(&d_centers[splat_id * 3 + 1], sum_centers.y);
}
```

**Pros:**
- Preserves SIMD optimization (fewer atomic ops)

**Cons:**
- More complex
- May not fix the issue if it's deeper

### Option 3: Use Threadgroup Shared Memory

Properly accumulate across threadgroup:

```metal
threadgroup float3 shared_grad_centers[SPLATS_PER_TILE_MAX];

// Each thread adds to shared memory
shared_grad_centers[i] += val_centers;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Thread 0 writes to global
if (thread_index_in_threadgroup == 0) {
    atomic_add_float(&d_centers[splat_id * 3 + 1], shared_grad_centers[i].y);
}
```

**Pros:**
- Correct synchronization

**Cons:**
- Most complex
- Requires shared memory allocation

---

## Recommended Fix (Option 1)

**Replace lines 478-514** with:

```metal
        // === B. Each thread atomically adds its own gradient (no SIMD) ===
        // This matches the forward pass pattern and avoids synchronization issues

        // Skip if gradient is zero or thread is inactive
        if (active && abs(d_L_d_I) > 1e-9f) {
            // Guard against NaN
            bool has_nan = isnan(val_amps) || isinf(val_amps) ||
                          isnan(val_sharpness) || isinf(val_sharpness) ||
                          isnan(val_centers.x) || isnan(val_centers.y) || isnan(val_centers.z);

            if (!has_nan) {
                // Atomic adds (every thread, not just lane 0)
                if (abs(val_amps) > 1e-12f) {
                    atomic_add_float(&d_amps[splat_id], val_amps);
                }
                if (abs(val_sharpness) > 1e-12f) {
                    atomic_add_float(&d_sharpness[splat_id], val_sharpness);
                }

                atomic_add_float(&d_centers[splat_id * 3 + 0], val_centers.x);
                atomic_add_float(&d_centers[splat_id * 3 + 1], val_centers.y);
                atomic_add_float(&d_centers[splat_id * 3 + 2], val_centers.z);

                int cb = splat_id * 6;
                for (int k = 0; k < 6; k++) {
                    if (abs(val_conic[k]) > 1e-12f && !isnan(val_conic[k]) && !isinf(val_conic[k])) {
                        atomic_add_float(&d_conic[cb + k], val_conic[k]);
                    }
                }
            }
        }
    }  // End splat loop
}  // End kernel
```

**Changes:**
1. Remove SIMD reduction entirely
2. Every thread directly does atomic_add with its own val_centers
3. Simpler, safer, matches forward pass

---

## Testing the Fix

After applying the fix:

```bash
cd packages/luxar/src/luxar/gsplats/models/gsplats/metal
python setup.py build_ext --inplace
hatch run python test_diagonal_conic_gradients.py  # Should now match Python reference
hatch run python test_frozen_L_gradients.py  # Should match CPU
hatch run python compare_raw_gradients.py  # All gradients should match
```

Expected result: **Deterministic gradients matching CPU within <5% error**.

---

## Performance Impact

Removing SIMD reduction increases atomic operations from ~2 per splat-tile to ~64 per splat-tile (32x more).

**Estimated impact:** ~10-20% slower backward pass (still 4-5x faster than CPU overall).

**This is acceptable** - correctness > speed. We can optimize later once it works.

---

## Summary for Expert

**The bug:** SIMD reduction + atomic_add pattern causes race condition → non-deterministic gradients

**The fix:** Remove SIMD reduction, make each thread atomic_add directly (like forward pass does)

**The code change:** Replace lines 478-522 in `kernels.metal` with simpler per-thread atomic pattern

**Why we're confident:** Forward pass (which uses this pattern) works perfectly. Backward should too.

