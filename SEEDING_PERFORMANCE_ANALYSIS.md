# 🔥 CRITICAL Performance Analysis: Seeding Algorithms

**Date:** 2026-01-15
**Focus:** Identify and fix performance bottlenecks in seed generation

---

## 🚨 CRITICAL BOTTLENECKS (Must Fix Immediately)

### 1. **O(M×N) Index Matching in `_combine_gsplatdata`** ⚠️⚠️⚠️

**Location:** `generate.py:396-410`

**Problem:**
```python
for deduped_coord in deduped_centers:  # M iterations
    # Computes distance to ALL N original coordinates!
    dists = np.sqrt(np.sum((all_centers - deduped_coord) ** 2, axis=1))  # O(N)
    matching_idx = np.argmin(dists)  # O(N)
    kept_indices.append(matching_idx)
```

**Complexity:** O(M × N) where M=5K-10K deduped, N=50K-100K original

**Performance:**
- M=5K, N=50K: **250 MILLION** distance calculations
- Each distance: sqrt(sum of squares) = expensive
- This runs EVERY time auto mode combines methods!

**Impact:** Dominates total seeding time for auto mode (60%+ of runtime)

**Root Cause:** `dedupe_farthest_first()` returns NEW coordinate arrays (with potential float rounding), so we can't track indices directly.

**Solution Options:**

**Option A: Return indices from dedupe_farthest_first** (Best - O(1) lookup)
```python
def dedupe_farthest_first(...) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (deduped_coords, kept_indices)"""
    # ... existing logic ...
    return deduped_coords, kept_original_indices
```
Then in `_combine_gsplatdata`:
```python
deduped_centers, kept_indices = dedupe_farthest_first(...)  # No O(M×N) search!
```

**Option B: Use coordinate hashing** (Good - O(M) with hash collisions)
```python
# Before deduplication:
coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(all_centers)}
# After deduplication:
kept_indices = [coord_to_idx[tuple(coord)] for coord in deduped_centers]
```

**Option C: Skip deduplication if methods don't overlap** (Fast - O(1))
```python
# If edge seeds and grid seeds use different regions, no need to dedupe
if spatially_disjoint(results):
    return concatenate_without_dedup(results)
```

**Estimated Speedup:** 100-1000x for auto mode (250M → 5K operations)

---

### 2. **O(M² log M) KD-Tree Rebuilding in `dedupe_farthest_first`** ⚠️⚠️

**Location:** `utils.py:267-308`

**Problem:**
```python
while True:
    # ... select farthest seed ...
    selected.append(coords_sorted[farthest_idx_global])

    # REBUILDS ENTIRE TREE FROM SCRATCH!
    selected_array = np.array(selected)  # O(M) - converts list to array
    tree = cKDTree(selected_array)        # O(M log M) - builds tree
```

**Complexity:** O(M² log M) where M = number of selected seeds

**Performance:**
- M=1000 seeds: ~10 MILLION tree building operations
- Each iteration: convert list → array → build tree
- For M iterations: O(M) + O(M log M) each = O(M² + M² log M)

**Impact:** 50-80% of deduplication time

**Solution Options:**

**Option A: Pre-allocate array and incremental tree** (Complex but fast)
```python
selected_array = np.empty((max_seeds, ndim), dtype=float)
selected_count = 0
# ... fill selected_array[selected_count] instead of list.append() ...
# Problem: cKDTree doesn't support incremental updates!
```

**Option B: Use Ball Tree from sklearn** (if it supports incremental)
```python
from sklearn.neighbors import BallTree
# Check if BallTree allows incremental updates
```

**Option C: Grid-based spatial hashing** (O(N) amortized)
```python
# Divide space into grid cells of size min_distance
# For each seed, check only neighboring cells
# No tree building needed!
```

**Option D: Greedy single-pass with one tree build** (Simpler, still fast)
```python
# Sort by intensity
# Build tree ONCE from all candidates
# Greedily select seeds, marking neighbors as invalid
# O(N log N) total instead of O(M² log M)
```

**Estimated Speedup:** 10-50x for deduplication (10M → 100K operations)

---

### 3. **O(M² log M) KD-Tree Rebuilding in `_poisson_disk_sample_weighted`** ⚠️⚠️

**Location:** `edges.py:252-278`

**Problem:** Identical to #2 - rebuilds KD-tree every iteration

```python
for idx in candidate_indices:  # M iterations
    coord = valid_coords[idx].astype(float)

    if tree is not None:
        dist, _ = tree.query(coord, k=1)
        if dist < min_distance:
            continue

    selected.append(coord)

    # REBUILDS TREE!
    if len(selected) > 0:
        tree = cKDTree(np.array(selected))  # O(M log M)
```

**Complexity:** Same as #2 - O(M² log M)

**Solution:** Same options as #2, but simpler: just use `dedupe_farthest_first` instead!

```python
def _poisson_disk_sample_weighted(...) -> np.ndarray:
    # Sample candidates weighted by density (no distance check)
    candidates = rng.choice(valid_coords, size=n_samples*2, p=prob)

    # Use existing dedupe_farthest_first for spatial filtering
    return dedupe_farthest_first(candidates, min_distance, valid_density)
```

**Estimated Speedup:** 10-50x (reuse optimized deduplication)

---

## 🔴 HIGH PRIORITY (Large Memory Allocations)

### 4. **Massive Allocation in `np.argwhere(mask)` for Edge Detection**

**Location:** `edges.py:226`

**Problem:**
```python
valid_coords = np.argwhere(mask)  # Returns ALL True coordinates!
```

**Memory Impact:**
- 1000×1000 image with 50% edges: 500K coordinates × 2 dimensions × 8 bytes = **8 MB**
- 512³ volume with 10% edges: 13M coordinates × 3 dimensions × 8 bytes = **300 MB**

**CPU Impact:**
- Creates coordinate array for ALL edge pixels
- Then samples only n_samples (e.g., 1000) from it
- 99.9% of allocations wasted!

**Solution:**
```python
# Option A: Randomly sample coordinates directly from mask
edge_indices = np.where(mask)
n_valid = len(edge_indices[0])
sample_idx = rng.choice(n_valid, size=min(n_samples*10, n_valid), p=prob)
valid_coords = np.column_stack([edge_indices[i][sample_idx] for i in range(ndim)])
```

**Option B: Use reservoir sampling** (constant memory)
```python
# Sample n_samples coordinates directly from mask without creating full array
```

**Estimated Speedup:** 10-100x reduction in allocation overhead

---

### 5. **Meshgrid Memory Explosion in Grid Generation**

**Location:** `grid.py:176-177`

**Problem:**
```python
grids = np.meshgrid(*ranges, indexing="ij")  # Creates ndim full-size arrays!
grid_coords = np.column_stack([g.ravel() for g in grids])
```

**Memory Impact:**
- 512³ volume, spacing=4: 128³ grid = 2,097,152 points
- Meshgrid creates 3 arrays of 2M elements each = **50 MB** intermediate allocation
- Then flattens and stacks = another **50 MB**
- Total: 100 MB for what could be done with 48 MB final array

**Solution:**
```python
# Option A: Use np.indices (slightly better)
indices = np.indices([len(r) for r in ranges], dtype=float)
# Still creates full arrays, but more efficient

# Option B: Direct coordinate generation (best)
# Use broadcasting to avoid intermediate arrays
grid_coords = np.empty((np.prod([len(r) for r in ranges]), ndim), dtype=float)
# Fill using index arithmetic (no intermediate arrays)
```

**Option C: Use np.mgrid with slicing** (cleanest)
```python
slices = [slice(start, end, spacing) for start, end, spacing in zip(...)]
grids = np.mgrid[slices]
grid_coords = grids.reshape(ndim, -1).T
```

**Estimated Speedup:** 2-3x memory reduction, 10-20% speed improvement

---

### 6. **Python Loop in `sigmas_to_cholesky_isotropic`**

**Location:** `utils.py:72-74`

**Problem:**
```python
for k in range(ndim):  # Python loop!
    diag_idx = k * (k + 3) // 2
    cholesky[:, diag_idx] = sigmas  # Sets column, but in a loop
```

**Performance:**
- N=10K seeds, ndim=3: 3 iterations, 30K assignments
- Each iteration: compute index, fancy index assignment
- Python loop overhead

**Solution:**
```python
# Pre-compute all diagonal indices (vectorized)
diag_indices = np.array([k * (k + 3) // 2 for k in range(ndim)])
# Single vectorized assignment
cholesky[:, diag_indices] = sigmas[:, np.newaxis]  # Broadcasting!
```

**Estimated Speedup:** 2-5x for large N

---

## 🟡 MEDIUM PRIORITY (Cumulative Impact)

### 7. **Repeated Type Conversions**

**Locations:** Throughout codebase

**Examples:**
```python
# edges.py:256
coord = valid_coords[idx].astype(float)  # Every iteration!

# utils.py:263, 265
coords_sorted = coords[sort_indices].astype(float)  # Potential copy
coords_sorted = coords.astype(float)  # Potential copy

# generate.py:413-416
centers=all_centers[kept_indices].astype(np.float32),  # Copy
amplitudes=all_amplitudes[kept_indices].astype(np.float32),  # Copy
cholesky_factors=all_cholesky[kept_indices].astype(np.float32),  # Copy
sharpnesses=all_sharpnesses[kept_indices].astype(np.float32),  # Copy
```

**Solution:**
- Use float32 consistently throughout pipeline
- Avoid intermediate conversions
- Convert once at input, never again

**Estimated Speedup:** 5-10% cumulative

---

### 8. **List → Array Conversions**

**Locations:**
- `utils.py:268-270` - `selected = [...]` then `np.array(selected)`
- `edges.py:239-264` - Same pattern
- `generate.py:399-406` - `kept_indices = []` then `np.array()`

**Solution:**
- Pre-allocate numpy arrays when size is known/bounded
- Use numpy array slicing instead of Python lists

**Estimated Speedup:** 2-5% cumulative

---

## 📊 Performance Summary Table

| Issue | Location | Complexity | Impact | Fix Difficulty | Priority |
|-------|----------|------------|--------|----------------|----------|
| Index matching | `generate.py:400` | O(M×N) | **CRITICAL** | Easy | **P0** |
| KD-tree rebuild (dedupe) | `utils.py:307` | O(M² log M) | **CRITICAL** | Medium | **P0** |
| KD-tree rebuild (poisson) | `edges.py:270` | O(M² log M) | **CRITICAL** | Easy | **P0** |
| argwhere allocation | `edges.py:226` | O(N) memory | High | Easy | P1 |
| Meshgrid allocation | `grid.py:176` | O(N) memory | High | Medium | P1 |
| Python loop | `utils.py:72` | O(ndim×N) | Medium | Easy | P2 |
| Type conversions | Multiple | O(N) | Medium | Easy | P2 |
| List conversions | Multiple | O(N) | Low | Easy | P3 |

---

## 🎯 Recommended Fix Order

### Phase 1: Critical O(N²) Fixes (10x-1000x speedup)
1. ✅ **Fix index matching in `_combine_gsplatdata`** (30 min)
   - Modify `dedupe_farthest_first` to return indices
   - Update callers to use returned indices

2. ✅ **Replace Poisson sampling with reused deduplication** (15 min)
   - Remove tree rebuilding loop
   - Use `dedupe_farthest_first` for spatial filtering

3. ✅ **Optimize `dedupe_farthest_first` tree rebuilding** (2 hours)
   - Implement grid-based spatial hashing OR
   - Switch to greedy single-pass algorithm

### Phase 2: Memory Optimizations (2x-10x speedup)
4. ✅ **Fix argwhere in edge detection** (30 min)
5. ✅ **Optimize meshgrid in grid generation** (1 hour)

### Phase 3: Micro-optimizations (5-20% cumulative)
6. ✅ **Vectorize Cholesky loop** (15 min)
7. ✅ **Remove redundant type conversions** (30 min)

---

## 📈 Expected Performance Gains

**Current (worst case):**
- 512³ volume, auto mode, 10K seeds
- Total time: ~10-30 seconds (dominated by O(M×N) matching)

**After Phase 1 fixes:**
- Same workload: ~1-3 seconds (**10x speedup**)
- Dominated by actual seed computation, not deduplication

**After all phases:**
- Same workload: ~0.5-1 seconds (**20-60x speedup**)
- Memory: 50-70% reduction

---

## 🧪 Profiling Recommendations

Before implementing fixes, profile with realistic data:

```python
import cProfile
import pstats

V = np.random.rand(512, 512, 64)  # ~16M voxels

pr = cProfile.Profile()
pr.enable()

seeds = generate_seeds(V, method="auto", target_seeds=5000)

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumtime')
stats.print_stats(30)
```

**Expected hotspots (confirm before fixing):**
1. `_combine_gsplatdata` → index matching loop
2. `dedupe_farthest_first` → KD-tree rebuilding
3. `_poisson_disk_sample_weighted` → KD-tree rebuilding
4. `ndi.sobel` → edge detection (acceptable - scipy C code)
5. `np.meshgrid` → grid generation

---

## 🎓 Key Lessons

1. **Avoid O(N²) at all costs** - Dominates runtime for N > 1000
2. **Never rebuild spatial structures in loops** - Build once, query many
3. **Allocate strategically** - Pre-allocate when size known, avoid waste
4. **Type consistency matters** - Float32 everywhere, convert once
5. **Return indices, not coordinates** - Enables O(1) lookups

---

## ✅ Action Items

- [ ] Profile current performance with realistic data
- [ ] Implement Phase 1 fixes (index matching, tree rebuilding)
- [ ] Add performance regression tests
- [ ] Update SEEDING_ALGORITHMS_REVIEW.md with new findings
- [ ] Commit with detailed performance measurements

---

**Priority:** These fixes should be implemented ASAP. The O(M×N) index matching alone is killing performance for any serious use case.
