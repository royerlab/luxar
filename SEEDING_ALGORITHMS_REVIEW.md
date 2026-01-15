# Critical Review: Gaussian Splat Seeding Algorithms

**Date:** 2026-01-15
**Reviewer:** Claude Code
**Scope:** `/packages/luxar/src/luxar/gsplats/seeds/`

---

## Executive Summary

The seeding algorithm implementation is **well-structured and documented** with good separation of concerns, but contains **significant performance issues** and **unused code** that should be addressed. Most critically:

- ✅ **Good**: Modular design, comprehensive tests, clear API
- ⚠️ **Concerning**: O(N²) algorithms without spatial acceleration
- ❌ **Critical**: ~120 lines of completely dead code in edges.py
- ⚠️ **Performance**: Multiple inefficiencies that compound at scale

**Overall Grade: B-** (Good architecture, needs optimization)

---

## 🔴 CRITICAL ISSUES

### 1. Dead Code in Edge Seeding (HIGH PRIORITY)

**Location:** `edges.py:282-435` (~153 lines)

**Problem:** The entire structure tensor calculation is disabled and replaced with a simple `σ=1.0` initialization:

```python
# Line 145-148 in edges.py
# Step 5: Use simple isotropic σ=1 initialization (ignore structure tensor estimates)
# The fancy anisotropic shapes from eigendecomposition don't help in practice
sigmas_one = np.ones(len(centers), dtype=np.float32)
cholesky_factors = sigmas_to_cholesky_isotropic(sigmas_one, ndim)
```

**Impact:**
- 153 lines of unused code
- API parameters (`structure_radius`, `min_sigma`, `max_sigma`) that do nothing
- Maintenance burden for code that never executes
- Misleading docstrings claiming "anisotropic Gaussian shapes"

**Functions affected (all dead code):**
- `_compute_structure_tensor_cholesky()` (lines 282-403)
- `_isotropic_cholesky()` (lines 406-416)
- `_pack_lower_triangular()` (lines 419-435)

**Recommendation:**
- ✅ **Remove dead code entirely** OR
- ⚠️ **Re-enable structure tensor with a flag** if there's research value
- Document why anisotropic initialization "doesn't help in practice"

---

### 2. O(N²) Poisson Disk Sampling

**Location:** `edges.py:206-279` (`_poisson_disk_sample_weighted`)

**Problem:** For each candidate point, checks distance against ALL previously selected points:

```python
# Lines 264-269
for idx in candidate_indices:
    coord = valid_coords[idx].astype(float)
    if len(selected) > 0:
        selected_arr = np.array(selected)
        distances = np.sqrt(np.sum((selected_arr - coord) ** 2, axis=1))  # O(M)
        if np.min(distances) < min_distance:
            continue
```

**Complexity:** O(N × M) where M = number of selected points (worst case O(N²))

**Performance Impact:**
- For `n_seeds=5000`: ~12.5M distance calculations
- Each iteration recreates `selected_arr` (memory allocation overhead)
- No spatial acceleration structure

**Comparison:** `dedupe_farthest_first()` in utils.py uses KD-tree for O(N log N) performance

**Recommendation:**
```python
# Use KD-tree for O(N log M) performance
from scipy.spatial import cKDTree

tree = None
for idx in candidate_indices:
    coord = valid_coords[idx].astype(float)
    if tree is not None:
        dist, _ = tree.query(coord, k=1)
        if dist < min_distance:
            continue
    selected.append(coord)
    tree = cKDTree(np.array(selected))  # Rebuild tree
```

**Alternative:** Use `dedupe_farthest_first()` after initial sampling instead of rejection during sampling.

---

### 3. O(N²) Deduplication in Combination

**Location:** `generate.py:398-408` (`_combine_gsplatdata`)

**Problem:** Naive nested loop for deduplication:

```python
# Lines 401-408
for i in range(len(centers_sorted)):
    if not kept_mask[i]:
        continue
    diffs = centers_sorted[i + 1 :] - centers_sorted[i]  # O(N)
    distances = np.sqrt(np.sum(diffs**2, axis=1))        # O(N)
    nearby = distances < min_distance
    kept_mask[i + 1 :][nearby] = False
```

**Complexity:** O(N²) for N seeds

**Performance Impact:**
- For 10,000 combined seeds: ~50M distance calculations
- Vectorized but still quadratic
- **Alternative already exists**: `dedupe_farthest_first()` uses KD-tree!

**Recommendation:**
```python
# Replace lines 398-415 with:
from luxar.gsplats.seeds.utils import dedupe_farthest_first

kept_centers = dedupe_farthest_first(
    all_centers,
    min_distance=min_distance,
    intensities=all_amplitudes  # Quality-priority ordering
)
# Then filter other arrays by kept indices
```

---

## ⚠️ PERFORMANCE ISSUES

### 4. Unnecessary 2x Oversampling in Auto Mode

**Location:** `generate.py:335`

```python
edge_kwargs = {**edges_kwargs, "n_seeds": budget_edges * 2}  # Over-sample
```

**Problem:**
- Allocates 2x memory for edge detection
- Performs 2x edge computations
- Comment says "Over-sample" but deduplication happens anyway

**Impact:** For budget=1000, computes 2000 edge seeds then throws away half

**Recommendation:**
- Remove the `* 2` factor
- If oversampling is needed, document WHY and make it configurable
- Consider: `oversample_factor = kwargs.get('oversample_factor', 1.5)`

---

### 5. Inefficient Random Seed Management

**Location:** Multiple files

**Problem:** Hardcoded `seed=42` in multiple places:
- `edges.py:247` - Poisson sampling
- `grid.py:171` - Grid jitter

**Issues:**
- Non-reproducible across different execution paths
- No way for user to control randomness
- Makes debugging harder

**Recommendation:**
```python
# Add to all seed functions:
def seed_from_edges(..., random_state=None):
    rng = np.random.default_rng(seed=random_state)
    # Use rng throughout
```

---

### 6. Broad Exception Catching

**Location:** `generate.py:339-355`

```python
try:
    seeds_edges = seed_from_edges(V, **edge_kwargs)
except ImportError:
    pass
except Exception:  # ❌ Too broad!
    pass
```

**Problem:**
- Silently swallows all errors (ValueError, RuntimeError, MemoryError, etc.)
- Makes debugging impossible
- Hides real bugs

**Recommendation:**
```python
try:
    seeds_edges = seed_from_edges(V, **edge_kwargs)
except (ImportError, ValueError) as e:
    if verbose:
        warnings.warn(f"Edge seeding failed: {e}")
```

---

### 7. Grid Spacing Ignores Anisotropy

**Location:** `generate.py:348-349`

```python
target_grid_density = budget_grid / total_voxels
spacing = max(2.0, (1.0 / target_grid_density) ** (1.0 / ndim))
```

**Problem:** Uses isotropic spacing even for highly anisotropic images

**Example:** For 1000×1000×10 microscopy image:
- Total voxels: 10M
- Computed spacing: 21.5 voxels (same for X, Y, Z)
- Should be: ~21×21×2 to respect aspect ratio

**Recommendation:**
```python
# Per-dimension spacing based on shape
shape = np.array(V.shape)
vol_per_seed = total_voxels / budget_grid
spacing_per_dim = shape * (vol_per_seed / total_voxels) ** (1.0 / ndim)
```

---

## 🔶 CODE QUALITY ISSUES

### 8. Magic Numbers Throughout

**Locations:**
- `generate.py:326-327`: Budget allocation (60% edges, 40% grid)
- `generate.py:323`: Seed target cap (10,000)
- `edges.py:158`, `grid.py:186`: Amplitude scaling (0.9)
- `grid.py:114`: Spacing default (5% of min dimension)

**Problem:** Unexplained constants make tuning difficult

**Recommendation:** Create a configuration class:

```python
class SeedingConfig:
    AUTO_BUDGET_EDGES = 0.60
    AUTO_BUDGET_GRID = 0.40
    AUTO_MAX_SEEDS = 10000
    AUTO_MIN_SEEDS = 100
    AMPLITUDE_SCALE = 0.9
    GRID_SPACING_FACTOR = 0.05
```

---

### 9. Inconsistent Amplitude Scaling

**Problem:** Three different implementations of the same concept:
- `multiscale_decomposition.py:23`: `_SEED_AMPLITUDE_SCALE = 0.9`
- `edges.py:158`: `* 0.9` (inline)
- `grid.py:186`: `* 0.9` (inline)

**Recommendation:** Move to shared constant in utils.py:
```python
# utils.py
SEED_AMPLITUDE_SCALE = 0.9  # Avoid overlap over-prediction
```

---

### 10. Parameter Validation Duplication

**Problem:** Every seeding function re-validates similar parameters:
```python
# Repeated in 3 files:
V = np.asarray(V, dtype=float)
if V.size == 0:
    raise ValueError("Input array V cannot be empty")
if V.ndim == 0:
    raise ValueError("Input array V must have at least 1 dimension")
```

**Recommendation:** Create `_validate_input()` helper in utils.py

---

## 🟡 ALGORITHMIC CONCERNS

### 11. Farthest-First May Not Be Optimal

**Location:** `utils.py:203-303` (`dedupe_farthest_first`)

**Concern:** Prioritizes spatial diversity over intensity:

```python
# Among valid seeds, pick the FARTHEST one (maximum distance)
valid_distances = distances[valid_mask]
farthest_idx = np.argmax(valid_distances)
```

**Tradeoff:**
- ✅ Guarantees spatial coverage
- ❌ May discard high-intensity peaks in crowded regions
- ❌ May keep low-intensity peaks in sparse regions

**Current behavior:** If sorting by intensity, picks highest-intensity first, then farthest among remaining

**Alternative algorithms to consider:**
1. **Weighted farthest-first:** `score = distance * intensity`
2. **Importance sampling:** Probability ∝ intensity × distance²
3. **Hierarchical clustering:** Group similar seeds, keep centroid

**Recommendation:** Document the tradeoff and consider making it configurable.

---

### 12. Auto Mode Heuristics Are Arbitrary

**Location:** `generate.py:320-323`

```python
# Heuristic: ~1 seed per 100 voxels^(1/ndim), minimum 100
total_voxels = float(np.prod(V.shape))
target_seeds = max(100, int(total_voxels ** (1.0 / ndim) / 2))
target_seeds = min(target_seeds, 10000)  # Cap at 10k
```

**Questions:**
- Why divide by 2?
- Why cap at 10k?
- Why min 100?

**Recommendation:** Add research notes explaining the scaling law, or make it configurable:
```python
def _estimate_seed_count(shape, scale_factor=0.5, min_seeds=100, max_seeds=10000):
    """
    Estimate seed count using empirical scaling law.

    scale_factor=0.5: ~1 seed per 2×spacing^ndim
    """
```

---

## ✅ POSITIVE ASPECTS

### What's Working Well:

1. **Modular Design** - Clean separation between methods (decomposition, grid, edges)
2. **Unified API** - `generate_seeds()` provides consistent interface
3. **Comprehensive Tests** - Good test coverage in tests/ directory
4. **Type Hints** - Proper typing throughout
5. **Documentation** - Excellent docstrings and README.md
6. **GSplatData Abstraction** - Clean return type with all necessary data
7. **nD Support** - Handles arbitrary dimensions correctly
8. **KD-Tree Optimization** - `dedupe_farthest_first()` is well-optimized (just not used everywhere!)

---

## 📊 PERFORMANCE PROFILING RECOMMENDATIONS

To validate performance concerns, profile with:

```python
import cProfile
import pstats

# Test with realistic dataset
V = np.random.rand(512, 512, 64)  # 16M voxels

pr = cProfile.Profile()
pr.enable()

seeds = generate_seeds(V, method="auto", target_seeds=5000)

pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

**Expected hotspots:**
1. `_poisson_disk_sample_weighted` - O(N²) loop
2. `_combine_gsplatdata` - O(N²) deduplication
3. `sobel` operations in edge detection

---

## 🔧 RECOMMENDED FIXES (Priority Order)

### High Priority (Do First):

1. **Remove dead code in edges.py** (lines 282-435)
   - Effort: 30 min
   - Impact: Code clarity, reduced maintenance

2. **Fix O(N²) deduplication in `_combine_gsplatdata`**
   - Use existing `dedupe_farthest_first()`
   - Effort: 15 min
   - Impact: 100-1000x speedup for large datasets

3. **Fix O(N²) Poisson sampling in edges.py**
   - Add KD-tree acceleration
   - Effort: 1 hour
   - Impact: 10-100x speedup for edge seeding

### Medium Priority:

4. **Remove 2x oversampling in auto mode** (line 335)
   - Effort: 5 min
   - Impact: 2x memory reduction, cleaner code

5. **Unify amplitude scaling** to shared constant
   - Effort: 15 min
   - Impact: Consistency, easier tuning

6. **Fix broad exception catching** in generate.py
   - Effort: 15 min
   - Impact: Better debugging

7. **Add random_state parameter** to all functions
   - Effort: 30 min
   - Impact: Reproducibility

### Low Priority (Nice to Have):

8. **Create SeedingConfig class** for magic numbers
9. **Add anisotropic grid spacing**
10. **Document algorithmic tradeoffs** (farthest-first, budget allocation)
11. **Add input validation helper** to reduce duplication

---

## 📈 ESTIMATED PERFORMANCE GAINS

**Before optimization:**
- 10K seeds on 512³ volume: ~5-10 seconds
- O(N²) algorithms dominate: `_combine_gsplatdata`, `_poisson_disk_sample_weighted`

**After optimization:**
- Same workload: ~0.5-1 seconds (5-20x speedup)
- Scales to 50K+ seeds without issues

**Memory reduction:**
- Remove 2x oversampling: 50% reduction in peak memory
- KD-tree: More cache-friendly than O(N²) loops

---

## 🧪 TESTING RECOMMENDATIONS

Add performance regression tests:

```python
def test_seeding_performance_large():
    """Ensure seeding scales to large datasets."""
    V = np.random.rand(512, 512, 64)  # 16M voxels

    import time
    start = time.time()
    seeds = generate_seeds(V, method="auto", target_seeds=5000)
    elapsed = time.time() - start

    assert elapsed < 2.0, f"Seeding took {elapsed:.2f}s, should be <2s"
    assert len(seeds.centers) > 4000, "Should generate ~5000 seeds"
```

---

## 🎯 CONCLUSION

The seeding implementation has **strong fundamentals** (good architecture, tests, docs) but needs **performance optimization** and **dead code removal**. The issues are localized and fixable with ~2-4 hours of focused work.

**Key takeaways:**
1. Replace O(N²) algorithms with KD-tree acceleration
2. Remove ~150 lines of dead structure tensor code
3. Unify constants and improve configurability
4. Add random_state for reproducibility

**Risk assessment:** LOW - Changes are isolated to specific functions, comprehensive tests exist.

---

## 📝 NEXT STEPS

1. **Immediate:** Fix critical O(N²) performance issues
2. **Short-term:** Remove dead code, unify constants
3. **Long-term:** Profile with real datasets, add performance tests

Would you like me to implement any of these fixes?
