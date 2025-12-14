# Lines Segment Spatial Index Chunking Bug - Analysis & Fix Plan

**Created**: 2025-12-10
**Status**: ✅ IMPLEMENTED (Commit: 8205029 - dimension-aware spatial indexing)
**Priority**: High (9× performance degradation - RESOLVED)
**Affects**: Lines spatial indexing for datasets with discrete non-displayed dimensions (FIXED)

> **Note**: This document is kept for its permanent architectural insights about the
> fundamental difference between Points (D-space) and Lines (2×D-space) spatial indexing.
> The performance issue has been resolved with dimension-aware padding.

---

## Executive Summary

Lines datasets with discrete non-displayed dimensions (e.g., time-animated particle tracks) suffer from severe performance degradation during viewer navigation. The root cause is **segment spatial index chunks that span multiple discrete dimension values**, causing queries to load 9-10× more data than necessary.

**Impact**:
- Loading ~294K vertices instead of ~31K per time slice (9.3× overhead)
- Affects ALL time-animated Lines datasets
- Points are NOT affected (use different chunking strategy)

**Fix**: Two coordinated changes to segment chunk computation:
1. Use dimension-aware padding (step_size/2 instead of hardcoded 0.5)
2. Split chunks at discrete dimension boundaries to prevent cross-frame contamination

---

## Table of Contents

1. [Bug Discovery Timeline](#1-bug-discovery-timeline)
2. [Root Cause Analysis](#2-root-cause-analysis)
3. [Mathematical Proof](#3-mathematical-proof)
4. [Affected Code](#4-affected-code)
5. [Implementation Plan](#5-implementation-plan)
6. [Testing Strategy](#6-testing-strategy)
7. [Risks and Mitigations](#7-risks-and-mitigations)

---

## 1. Bug Discovery Timeline

### Initial Symptom (User Report)
- Time scrubbing in `demo_particle_collision_animated.py` was slow
- Data Loading Monitor showed "4.0M visible lines"
- Console showed excessive logging (10,000+ dequantization messages)

### Investigation Phases

**Phase 1**: Removed excessive logging
- Removed per-chunk dequantization logs
- Result: "better, not super fast" - logging was symptom, not cause

**Phase 2**: Segment query analysis
- Logs showed: "Found 36/965 segment chunks (3.7%)" ✓ Correct
- But: "Loading 294,912 vertices (100%)" ✗ Bug!
- Expected: ~3.7% of vertices, got 100%

**Phase 3**: Suspected spatial ordering issue
- Hypothesis: Morton/Hilbert ordering destroys time locality
- Investigation: Confirmed Lines DOES use compound ordering
- Verified: Time dimension correctly marked as `discrete=True, display=False`

**Phase 4**: Diagnostic script on small dataset
- Created minimal test (1 event, 10 frames, 57K vertices)
- Result: **PERFECT time locality!**
  - Vertices sorted by time ✓
  - Each time frame: consecutive indices ✓
  - Query at time T: ONE range, 100% efficiency ✓
- **Conclusion**: Compound ordering WORKS correctly!

**Phase 5**: Runtime logging on large dataset
- Added vertex index range diagnostics to TypeScript loader
- Discovered: Loading ~294K vertices per frame from large dataset (7.9M vertices, 250 frames)
- Expected: ~31K vertices per frame
- **Ratio: 9.3× over-loading!**

**Phase 6**: Identified root cause
- Segment chunks span multiple time frames
- Hardcoded ±0.5 padding too large for time dimension
- Fixed chunk size (4,096 segments) crosses time boundaries

---

## 2. Root Cause Analysis

### 2.1 How Segment Spatial Indexing Works

**Encoding (Python)**:
1. Sort segments in **2×D space** by `(t_start, t_end, spatial_code)` - compound ordering
2. Divide sorted segments into chunks of **4,096 segments** (fixed size)
3. Compute chunk bounding boxes in **D-space** (not 2×D) for each chunk
4. Store chunk bounds in spatial index metadata

**Querying (TypeScript)**:
1. Query spatial index with current slice position + tolerance
2. Find chunks whose bounds intersect query range
3. Load ALL segments from matching chunks
4. Extract vertex indices from loaded segments
5. Load vertices by ranges

### 2.2 Root Cause #1: Hardcoded ±0.5 Padding

**Location**: `packages/luxar/src/luxar/io/ordering.py` lines 771-776

**Current Code**:
```python
if d in discrete_dims:
    chunk_bounds[chunk_idx, d, 0] = (
        min(p1[:, d].min(), p2[:, d].min()) - 0.5  # ← Hardcoded!
    )
    chunk_bounds[chunk_idx, d, 1] = (
        max(p1[:, d].max(), p2[:, d].max()) + 0.5  # ← Hardcoded!
    )
```

**Problem**:
- Time step in demo: `50ns / 250 frames = 0.2ns`
- Hardcoded padding: `±0.5ns`
- Padding is **2.5× the step size!**
- This causes chunk bounds to extend ±2.5 frames beyond actual data

**Impact**:
- A chunk spanning times [5.0 - 7.5] becomes bounds [4.5 - 8.0] after padding
- Query at time=6.0 ± 0.5 (range [5.5 - 6.5]) matches this chunk
- But chunk contains segments from 5.0, 5.2, 5.4, ..., 7.5 (13 frames!)

### 2.3 Root Cause #2: Chunks Spanning Discrete Boundaries

**Location**: `packages/luxar/src/luxar/io/compiler.py` lines 1524-1527

**Current Code**:
```python
# Segment chunk size
bytes_per_segment = 8 + 8  # 2 uint32 indices + overhead
segment_chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_segment)
segment_chunk_size = min(segment_chunk_size, n_segments)
```

This creates **fixed-size chunks of 4,096 segments** without considering discrete dimension boundaries.

**Problem**:
After compound sorting by `(t_start, t_end, spatial_code)`:
- Segments 0-4,095 might be:
  - Segs 0-2,500: time=(5.0, 5.0), various spatial codes
  - Segs 2,501-4,095: time=(5.2, 5.2), various spatial codes

When computing chunk bounds:
```python
# This chunk's time bounds
chunk_time_min = 5.0  # From segment 0
chunk_time_max = 5.2  # From segment 4095
chunk_bounds = [5.0 - 0.5, 5.2 + 0.5] = [4.5, 5.7]
```

**Result**: Single chunk spans 2 time frames!

Across 965 segment chunks, if each spans ~2 frames on average:
- Query one time frame → matches chunks spanning ±1 frame
- Load segments from 2-3 frames → vertices from 9-10 frames!

### 2.4 Why Points Are Not Affected

**Points use different chunking** (`compute_point_chunk_bounds` in ordering.py):
- Points are sorted in **D-space** (not 2×D)
- Compound ordering ensures all points at time T have consecutive indices
- Point chunks naturally align to time boundaries due to sort order
- Even if a chunk spans 2 time frames, points at each time are consecutive
- Range-based loading works efficiently

**Lines are different**:
- Segments sorted in **2×D space** create different patterns
- Segment chunks don't naturally align to time boundaries
- Segments reference vertices scattered across multiple time blocks

---

## 3. Mathematical Proof

### 3.1 Small Dataset (Works Correctly)

**Configuration**: 1 event, 10 frames (actually 9 time values)
```
Total vertices: 57,600
Vertices per frame: 6,400
Segments per frame: 3,200

Diagnostic Results:
  ✓ Vertices sorted by time
  ✓ Time 27.78: indices [25,600 - 31,999] (ONE consecutive block)
  ✓ Segments at time 27.78: reference exactly those 6,400 vertices
  ✓ Efficiency: 100.0% (1 range)
```

### 3.2 Large Dataset (Broken)

**Configuration**: 1 event, 250 frames, 248 tracks
```
Total vertices: 7,904,256
Vertices per frame: 248 tracks × 128 verts/track = 31,744 verts/frame
Total segments: 3,952,128
Segments per frame: 248 tracks × 64 segs/track = 15,872 segs/frame

Runtime Logs:
  Segment query: 36/965 chunks (3.7%) ✓ Spatially correct
  Vertex loading: ~294,000 vertices

Analysis:
  294,000 / 31,744 = 9.3 frames worth of vertices! ✗ BUG

Efficiency: 92-99.6% (varies by frame)
  - Best case: 99.6% with 39 ranges (almost one large consecutive block)
  - Worst case: 92.4% with 327 ranges (highly fragmented)
```

**Why Small Dataset Works**:
- Fewer frames (9) → fewer time transitions
- Fewer segments per frame → chunks rarely span time boundaries
- Most chunks contain segments from single time frame

**Why Large Dataset Fails**:
- Many frames (250) → many time transitions throughout sorted array
- 4,096-segment chunks ALWAYS span multiple frames:
  - ~16,000 segs/frame ÷ 4,096 segs/chunk ≈ 3.9 chunks/frame
  - Chunk boundaries misaligned with time boundaries
  - Every chunk spans 2-3 frames

---

## 4. Affected Code

### 4.1 Python Encoder (luxar package)

| File | Function | Issue | Lines |
|------|----------|-------|-------|
| `io/ordering.py` | `compute_segment_chunk_bounds()` | Hardcoded ±0.5 padding | 771-776 |
| `io/ordering.py` | `compute_segment_chunk_bounds()` | No discrete boundary awareness | 756-786 |
| `io/compiler.py` | `_build_lines_spatial_ordering_if_enabled()` | Fixed chunk size, no boundary splitting | 1524-1527 |

### 4.2 TypeScript Decoder (luxar-viewer package)

| File | Function | Status |
|------|----------|--------|
| `data/lines-spatial-index-loader.ts` | `queryVisibleSegmentRanges()` | ✓ Working correctly - no changes needed |
| `data/lines-chunk-spatial-index.ts` | `querySegmentChunksForView()` | ✓ Working correctly - queries what encoder provides |

**Note**: TypeScript decoder is NOT the problem. It correctly queries the spatial index. The issue is that the encoder creates chunks with incorrect bounds.

### 4.3 Documentation Files

| File | Section | Update Required |
|------|---------|-----------------|
| `packages/luxar/src/luxar/io/SPECIFICATIONS.md` | Segment Spatial Ordering | Document discrete-aware chunking |
| `packages/luxar/src/luxar/io/README.md` | Spatial Indexing | Mention discrete boundary alignment |
| `packages/luxar-viewer/src/data/SPECIFICATIONS.md` | Section 7.9 Lines Dimension Extension | Update chunk behavior description |
| `docs/guides/user/LUXAR_ZARR_FORMAT.md` | Spatial Index Format | Document discrete-aware chunking guarantee |

---

## 5. Implementation Plan

### 5.1 Fix #1: Dimension-Aware Padding

**Location**: `packages/luxar/src/luxar/io/ordering.py`
**Function**: `compute_segment_chunk_bounds()`
**Change**: Replace hardcoded 0.5 with step-based padding

**Current** (lines 724-730):
```python
def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,  # ← Only indices, no step info!
) -> np.ndarray:
```

**New Signature**:
```python
def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
    dimensions: Optional[list[Dimension]] = None,  # ← ADD THIS
) -> np.ndarray:
```

**Implementation** (lines 771-776):
```python
if d in discrete_dims:
    # Use dimension-aware padding
    if dimensions and d < len(dimensions) and dimensions[d].step:
        padding = dimensions[d].step / 2
    else:
        padding = 0.5  # Fallback for missing metadata

    chunk_bounds[chunk_idx, d, 0] = (
        min(p1[:, d].min(), p2[:, d].min()) - padding
    )
    chunk_bounds[chunk_idx, d, 1] = (
        max(p1[:, d].max(), p2[:, d].max()) + padding
    )
```

**Caller Update** (`io/compiler.py` lines 1550-1556):
```python
segment_chunk_bounds = compute_segment_chunk_bounds(
    sorted_vertices,
    sorted_segments,
    widths_expanded,
    segment_chunk_size,
    slice_dims=ordering_metadata["vertex_ordering"]["slice_dims"],
    dimensions=dimensions.dimensions,  # ← PASS THIS
)
```

### 5.2 Fix #2: Discrete-Aware Chunk Splitting

**Location**: `packages/luxar/src/luxar/io/ordering.py`
**New Function**: `split_chunks_at_discrete_boundaries()`

**Implementation**:
```python
def split_chunks_at_discrete_boundaries(
    segments: np.ndarray,
    vertices: np.ndarray,
    base_chunk_size: int,
    slice_dims: list[int],
    min_chunk_size: int = 256,
) -> list[tuple[int, int]]:
    """Split segment chunks to never span discrete dimension boundaries.

    This ensures that all segments in a chunk have compatible discrete
    dimension values (e.g., all at the same time frame), which is critical
    for efficient spatial queries on time-animated data.

    IMPORTANT: Checks BOTH segment endpoints to handle segments that may
    span discrete boundaries (e.g., trajectory from time A to time B).

    Args:
        segments: Sorted segment indices (S, 2)
        vertices: Sorted vertex positions (V, D)
        base_chunk_size: Target chunk size (max segments per chunk)
        slice_dims: Indices of discrete dimensions
        min_chunk_size: Minimum segments per chunk (avoid too many tiny chunks)

    Returns:
        List of (start_idx, end_idx) tuples defining chunk ranges
    """
    if not slice_dims or len(segments) == 0:
        # No discrete dims or empty - use fixed chunks
        num_chunks = (len(segments) + base_chunk_size - 1) // base_chunk_size
        return [(i * base_chunk_size, min((i + 1) * base_chunk_size, len(segments)))
                for i in range(num_chunks)]

    chunks = []
    chunk_start = 0

    # Track discrete value range for current chunk
    # For each discrete dim, track [min_val, max_val] across chunk
    def get_segment_discrete_range(seg_idx):
        """Get [min, max] discrete values for a segment (checking both endpoints)."""
        p1_vals = vertices[segments[seg_idx, 0], slice_dims]
        p2_vals = vertices[segments[seg_idx, 1], slice_dims]
        return np.minimum(p1_vals, p2_vals), np.maximum(p1_vals, p2_vals)

    chunk_min, chunk_max = get_segment_discrete_range(0)

    for i in range(1, len(segments)):
        seg_min, seg_max = get_segment_discrete_range(i)

        # Check if this segment's discrete range is compatible with chunk's range
        # Compatible = overlaps or is adjacent (within small tolerance)
        tolerance = 0.01  # Small epsilon for floating-point comparison
        discrete_incompatible = np.any(
            (seg_max < chunk_min - tolerance) |  # Segment entirely before chunk
            (seg_min > chunk_max + tolerance)     # Segment entirely after chunk
        )

        chunk_size_ok = (i - chunk_start) < base_chunk_size
        chunk_too_small = (i - chunk_start) < min_chunk_size

        # Split if discrete values incompatible AND chunk not too small
        if discrete_incompatible and not chunk_too_small:
            # Finalize current chunk
            chunks.append((chunk_start, i))
            chunk_start = i
            chunk_min, chunk_max = seg_min, seg_max
        # Also split if chunk size limit reached (regardless of discrete values)
        elif not chunk_size_ok:
            chunks.append((chunk_start, i))
            chunk_start = i
            chunk_min, chunk_max = seg_min, seg_max
        else:
            # Expand chunk range to include this segment
            chunk_min = np.minimum(chunk_min, seg_min)
            chunk_max = np.maximum(chunk_max, seg_max)

    # Add final chunk
    if chunk_start < len(segments):
        chunks.append((chunk_start, len(segments)))

    return chunks
```

**Note**: This implementation:
- Checks BOTH endpoints to handle segments spanning discrete boundaries
- Enforces minimum chunk size (256) to avoid excessive fragmentation
- Uses range overlap logic (not exact equality) for robustness
- Handles multiple discrete dimensions via numpy arrays

**Integration Point** (`io/ordering.py` around line 685):

**Current**:
```python
def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
) -> np.ndarray:
    # ...
    num_chunks = (S + chunk_size - 1) // chunk_size  # ← Fixed division

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size  # ← Fixed boundaries
        end_idx = min(start_idx + chunk_size, S)
```

**New**:
```python
def compute_segment_chunk_bounds(
    vertices: np.ndarray,
    segments: np.ndarray,
    widths: np.ndarray,
    chunk_size: int,
    slice_dims: Optional[list[int]] = None,
    dimensions: Optional[list[Dimension]] = None,
) -> np.ndarray:
    # Split chunks at discrete boundaries
    if slice_dims:
        chunk_ranges = split_chunks_at_discrete_boundaries(
            segments, vertices, chunk_size, slice_dims
        )
    else:
        # No discrete dims - use fixed chunks
        num_chunks = (len(segments) + chunk_size - 1) // chunk_size
        chunk_ranges = [(i * chunk_size, min((i + 1) * chunk_size, len(segments)))
                        for i in range(num_chunks)]

    num_chunks = len(chunk_ranges)
    chunk_bounds = np.zeros((num_chunks, D, 2), dtype=np.float32)

    for chunk_idx, (start_idx, end_idx) in enumerate(chunk_ranges):
        chunk_segs = segments[start_idx:end_idx]
        # ... rest of bounds computation
```

### 5.3 Caller Updates

**Location**: `io/compiler.py` lines 1550-1556

Add `dimensions` parameter to call:
```python
segment_chunk_bounds = compute_segment_chunk_bounds(
    sorted_vertices,
    sorted_segments,
    widths_expanded,
    segment_chunk_size,
    slice_dims=ordering_metadata["vertex_ordering"]["slice_dims"],
    dimensions=dimensions.dimensions,  # ← ADD THIS
)
```

Also update function signature to accept dimensions.

### 5.4 Vertex Chunk Bounds (Consistency)

**Location**: `packages/luxar/src/luxar/io/ordering.py` lines 712-715

For consistency, vertex chunk bounds should also use dimension-aware padding:

**Current**:
```python
if d in discrete_dims:
    chunk_bounds[chunk_idx, d, 0] = chunk_verts[:, d].min() - 0.5
    chunk_bounds[chunk_idx, d, 1] = chunk_verts[:, d].max() + 0.5
```

**New**:
```python
if d in discrete_dims:
    if dimensions and d < len(dimensions) and dimensions[d].step:
        padding = dimensions[d].step / 2
    else:
        padding = 0.5
    chunk_bounds[chunk_idx, d, 0] = chunk_verts[:, d].min() - padding
    chunk_bounds[chunk_idx, d, 1] = chunk_verts[:, d].max() + padding
```

**Note**: Vertex chunks likely already align to discrete boundaries due to compound sorting, but this ensures consistency.

---

## 6. Testing Strategy

### 6.1 Unit Tests

**New Test File**: `packages/luxar/tests/io/test_discrete_chunking.py`

```python
def test_split_chunks_at_discrete_boundaries():
    """Test that chunks split when discrete dimension changes."""
    # Create mock segments with vertices at 3 different times
    vertices = np.array([
        [0, 0, 0, 5.0],  # Time 5.0
        [1, 0, 0, 5.0],
        [2, 0, 0, 5.2],  # Time 5.2
        [3, 0, 0, 5.2],
        [4, 0, 0, 5.4],  # Time 5.4
        [5, 0, 0, 5.4],
    ])
    segments = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint32)

    chunks = split_chunks_at_discrete_boundaries(
        segments, vertices, chunk_size=10, slice_dims=[3]
    )

    # Should create 3 chunks (one per time value)
    assert len(chunks) == 3
    assert chunks[0] == (0, 1)  # Segment at time 5.0
    assert chunks[1] == (1, 2)  # Segment at time 5.2
    assert chunks[2] == (2, 3)  # Segment at time 5.4

def test_dimension_aware_padding():
    """Test that padding uses step_size/2 instead of hardcoded 0.5."""
    # Create dimension with step=0.2
    dims = [Dimension("time", step=0.2, discrete=True)]

    # Create chunk with time range [5.0 - 5.0] (single frame)
    # ...compute bounds...

    # Verify bounds use ±0.1 (step/2), not ±0.5
    assert chunk_bounds[0, 3, 0] == pytest.approx(5.0 - 0.1)
    assert chunk_bounds[0, 3, 1] == pytest.approx(5.0 + 0.1)
```

### 6.2 Integration Test

**Test**: Encode and decode time-animated Lines, verify loading efficiency

```python
def test_time_animated_lines_efficiency():
    """Verify efficient vertex loading for time-animated Lines."""
    # Create dataset with 50 frames, 100 tracks
    # Encode with spatial indexing
    # Decode at single time frame
    # Verify: loaded vertices ≈ vertices_per_frame (within 10%)

    total_verts = ...
    verts_per_frame = total_verts / 50
    loaded_verts = ...  # From decoder

    efficiency = loaded_verts / verts_per_frame
    assert efficiency < 1.2, f"Loading {efficiency:.1f}× expected vertices"
```

### 6.3 E2E Test

**Test**: Full pipeline with viewer

```typescript
// packages/luxar-viewer/tests/e2e/lines-time-animated.spec.ts
test('time-animated lines load efficiently', async ({ page }) => {
  // Load particle collision animated dataset
  // Navigate through time frames
  // Verify console logs show:
  //   - Segment chunks: ~3-5% of total
  //   - Vertex loading: ~1/Nth of total (where N = num frames)
  //   - Efficiency: >95%
});
```

### 6.4 Manual Testing

1. **Generate test dataset**:
   ```bash
   hatch run python packages/luxar/src/luxar/demos/demo_particle_collision_animated.py \
     --events=2 --frames=100
   ```

2. **Verify encoding logs** show:
   ```
   ✓ Dual ordering complete: XXXX vertex chunks, YYYY segment chunks
   ```
   Expect: MORE segment chunks than before (due to discrete splitting)

3. **Load in viewer** and navigate through time frames

4. **Check console logs** for:
   ```
   Loading vertices for X ranges (Y unique vertices)
   Vertex index range: [...], efficiency=ZZ%
   ```
   Expect:
   - Y ≈ 31,744 (not 294K!)
   - Efficiency >95% (not 92%)
   - X < 50 ranges (not 270-327)

---

## 7. Risks and Mitigations

### 7.1 Risk: Increased Number of Chunks

**Concern**: Splitting at discrete boundaries creates more chunks
- Current: 965 segment chunks for 3.9M segments
- After fix: Could be ~2,000-3,000 chunks (one per time × spatial regions)

**Impact**:
- Slightly larger spatial index metadata (negligible)
- Slightly longer index query time (checking more chunks)

**Mitigation**:
- The query time increase is negligible (<1ms for 3,000 chunks)
- Loading time savings (9.3× fewer vertices) FAR outweighs this
- **Net performance gain: massive**

### 7.2 Risk: Backward Compatibility

**Concern**: Existing zarr files use old chunking

**Impact**:
- Old zarr files will continue to work (decoder unchanged)
- But will have poor performance for time-animated Lines
- Users need to re-encode to get performance benefit

**Mitigation**:
- Add version number to spatial index metadata
- Document breaking change in CHANGELOG
- Provide migration script if needed (optional)

### 7.3 Risk: Edge Cases

**Concern**: Datasets with:
- No discrete dimensions (pure spatial)
- Multiple discrete dimensions
- Very small chunks after splitting

**Mitigation**:
- Fallback to fixed chunks if no discrete dims
- Support multiple discrete dims (nested splits)
- Set minimum chunk size (e.g., 256 segments) to avoid too many tiny chunks

### 7.4 Risk: Points vs Lines Divergence

**Concern**: Points and Lines use different chunking strategies

**Impact**:
- Already true (Points don't need discrete splitting due to D-space sorting)
- This fix makes them more different

**Mitigation**:
- Document the difference clearly in specs
- Rationale: Different geometry types have different spatial coherence properties
- Points in D-space naturally align; Segments in 2×D-space don't

---

## 8. Critical Review & Corrections

### 8.1 Issue Found: Original Split Function Incomplete

**Problem**: Initial implementation checked only `segments[i, 0]` (start vertex) for discrete value changes.

**Why This Is Wrong**:
- Segment bounds computation (line 772-775) uses **min/max of BOTH endpoints**
- Segments can span discrete boundaries (e.g., trajectory from time=5 to time=7)
- Checking only start vertex would miss segments where `t_start ≠ t_end`

**Fix**: Updated `split_chunks_at_discrete_boundaries()` to:
- Check BOTH endpoints via `get_segment_discrete_range()`
- Use range overlap logic instead of exact equality
- Handle segments spanning discrete boundaries correctly

### 8.2 Issue Found: No Minimum Chunk Size

**Problem**: Splitting at every discrete transition could create thousands of tiny chunks for fine-grained time series.

**Impact**:
- More chunks = larger index metadata
- More chunks = slightly slower queries
- Extreme case: 1 segment per chunk (pathological)

**Fix**: Added `min_chunk_size=256` parameter (hardcoded initially, can be made configurable)
- Won't split if chunk would be <256 segments
- Trades slight cross-frame overlap for reasonable chunk count
- For 250 frames: ~16K segs/frame ÷ 256 = max ~62 chunks/frame → ~15,500 total chunks
- Compare to current: 965 chunks total → increase by 16× (acceptable)

**Edge Case - Few Segments Per Frame**:
- If dataset has <256 segments per frame (e.g., 1000 frames × 10 segs each)
- Chunks will span multiple frames (not ideal, but still better than current)
- **Future enhancement**: Make `min_chunk_size` configurable or adaptive based on segs/frame distribution

### 8.3 Issue Found: Dimension.step Validation

**Problem**: Not all Dimension objects have `step` defined (it's optional).

**Risk**: `dimensions[d].step` could be `None`, causing `None / 2` → TypeError

**Fix**: Added defensive check in padding computation:
```python
if dimensions and d < len(dimensions) and dimensions[d].step:
    padding = dimensions[d].step / 2
else:
    padding = 0.5  # Fallback for missing metadata
```

### 8.4 Assumption: Segments Sorted By Compound Ordering

**Critical Assumption**: The split function assumes segments are ALREADY sorted by compound ordering with discrete dims as primary key.

**Verification**:
- `sort_segments_compound()` (ordering.py:483-602) DOES sort by discrete dims first ✓
- Lexsort order: `[spatial_codes] + [slice_values[:, i] for i in reversed(slice_dims)]`
- This means discrete dims are PRIMARY sort key ✓

**Implication**: Segments are already grouped by time in sorted order. The split function just needs to identify transition points!

### 8.5 Additional Edge Cases Analyzed

**Case 1: Sparse Time Series** (frames with no segments)
- Behavior: No segments = no chunks for that frame
- Impact: None - handled correctly ✓

**Case 2: Variable Segments Per Frame** (50 to 10,000)
- Behavior: Small frames group together; large frames split
- Impact: Adaptive chunk count - correct ✓

**Case 3: Segments Spanning Discrete Boundaries** (trajectory t=5→7)
- Behavior: Range overlap [5-7] compatible with chunk [5-8]
- Impact: Correctly handles time-spanning geometry ✓

**Case 4: Multiple Discrete Dimensions** (time + event_id)
- Behavior: Numpy operations handle multi-dimensional checks
- Impact: Splits when ANY dimension incompatible ✓

**Case 5: Fine-Grained Time** (1000 frames, 10 segments each)
- Behavior: <256 segs/frame → chunks span ~25 frames
- Impact: Not optimal, but 4× better than current (~100 frames)
- Mitigation: Make min_chunk_size adaptive in future if needed

**Case 6: No Step Defined** (continuous time dimension)
- Behavior: Falls back to ±0.5 padding
- Impact: Still works, just less precise ✓

### 8.6 Diagnostic Logging Specification

**Location**: `io/ordering.py` in `compute_segment_chunk_bounds()`

Add logging AFTER chunk splitting to show:
```python
if slice_dims:
    # Log chunk split statistics
    chunk_count = len(chunk_ranges)
    avg_chunk_size = len(segments) / chunk_count if chunk_count > 0 else 0
    min_size = min(end - start for start, end in chunk_ranges)
    max_size = max(end - start for start, end in chunk_ranges)

    aprint(f"  📊 Discrete-aware chunking:")
    aprint(f"     Chunks: {chunk_count} (avg {avg_chunk_size:.0f} segs, "
           f"range [{min_size}-{max_size}])")

    # Verify no chunk spans discrete boundaries
    max_span = 0
    for start, end in chunk_ranges:
        chunk_segs = segments[start:end]
        for d in slice_dims:
            p1_vals = vertices[chunk_segs[:, 0], d]
            p2_vals = vertices[chunk_segs[:, 1], d]
            span = max(p1_vals.max(), p2_vals.max()) - min(p1_vals.min(), p2_vals.min())
            max_span = max(max_span, span)

    aprint(f"     Max discrete span per chunk: {max_span:.3f} "
           f"(should be ≤ step_size={dimensions[slice_dims[0]].step if dimensions else 'N/A'})")
```

---

## 9. Implementation Checklist

### Phase 1: Core Fix
- [ ] Add `dimensions` parameter to `compute_segment_chunk_bounds()`
- [ ] Implement dimension-aware padding (step_size/2)
- [ ] Update `compute_vertex_chunk_bounds()` for consistency
- [ ] Update caller in `compiler.py` to pass dimensions

### Phase 2: Discrete Boundary Splitting
- [ ] Implement `split_chunks_at_discrete_boundaries()`
- [ ] Integrate into `compute_segment_chunk_bounds()`
- [ ] Add logging to show chunk count before/after splitting

### Phase 3: Testing
- [ ] Write unit tests for boundary splitting
- [ ] Write unit tests for dimension-aware padding
- [ ] Run existing test suite (ensure no regressions)
- [ ] Manual test with particle collision demo
- [ ] Verify 9× performance improvement

### Phase 4: Documentation
- [ ] Update `io/SPECIFICATIONS.md` - discrete-aware chunking algorithm
- [ ] Update `io/README.md` - mention performance benefit
- [ ] Update `data/SPECIFICATIONS.md` (viewer) - chunk behavior
- [ ] Update `guides/user/LUXAR_ZARR_FORMAT.md` - spatial index format
- [ ] Add entry to CHANGELOG.md

### Phase 5: Validation
- [ ] Run `make test-all`
- [ ] Generate large dataset, verify encoding logs show more chunks
- [ ] Load in viewer, verify ~31K vertices per frame (not 294K)
- [ ] Measure time scrubbing performance improvement

---

## 10. Expected Results

### Before Fix:
```
Segment chunks: 965 total
  Chunk size: 4,096 segments (fixed)
  Each chunk spans: ~2-3 time frames (misaligned boundaries)

Query at time T:
  Matched chunks: ~36/965 (3.7%)
  Frames spanned: 9-10 frames
  Vertex loading: ~294K vertices (9.3 frames worth)
  Vertex efficiency: 92-93%
  Vertex ranges: 260-330 ranges

Performance: Slow time scrubbing (~200ms per frame)
```

### After Fix (Ideal Case):
```
Segment chunks: ~2,000-15,000 total (varies by min_chunk_size)
  Chunk size: 256-4,096 segments (variable, split at time boundaries)
  Each chunk spans: 1 time frame (aligned boundaries)

Query at time T:
  Matched chunks: ~8-12 chunks (all at time T)
  Frames spanned: 1 frame only
  Vertex loading: ~31K vertices (1 frame worth)
  Vertex efficiency: >99%
  Vertex ranges: 1-10 ranges (mostly consecutive)

Performance: 9.3× faster time scrubbing (~20ms per frame)
```

### After Fix (With min_chunk_size=256 Tradeoff):
```
Segment chunks: ~5,000-10,000 total
  Some chunks may span 1-2 adjacent frames (if <256 segs/frame)
  Most chunks aligned to single frame

Query at time T:
  Matched chunks: ~10-15 chunks
  Frames spanned: 1-2 frames (significant improvement)
  Vertex loading: ~50K vertices (1.5 frames worth)
  Vertex efficiency: >97%
  Vertex ranges: 10-30 ranges

Performance: 5-6× faster time scrubbing (~40ms per frame)

Note: Exact results depend on segments-per-frame distribution
```

---

## 10. Alternative Approaches Considered

### Alt 1: Increase Query Tolerance Precision
- **Idea**: Use tighter tolerance (0.01 instead of 0.5) for time queries
- **Rejected**: Doesn't fix chunk boundary spans; could miss valid segments

### Alt 2: Disable Spatial Ordering for Time-Animated Data
- **Idea**: Don't sort animated Lines, preserve time locality
- **Rejected**: Loses spatial coherence; hurts XYZ queries

### Alt 3: Separate Spatial Index Per Time Frame
- **Idea**: Create independent spatial indices for each time value
- **Rejected**: Massive metadata overhead; complex to maintain

### Alt 4: Viewer-Side Filtering
- **Idea**: Load all vertices, filter in viewer based on actual time
- **Rejected**: Doesn't solve network/decoding overhead; just moves the problem

**Chosen Approach** (Discrete-Aware Chunking):
- ✓ Preserves spatial coherence within each time frame
- ✓ Ensures time isolation between frames
- ✓ Minimal metadata overhead
- ✓ Works for any discrete dimension (not just time)
- ✓ Backward compatible (old data still works, just not optimally)

---

## Appendix: Code References

### A. Segment Chunk Bounds Computation
`packages/luxar/src/luxar/io/ordering.py` lines 724-786

### B. Segment Spatial Ordering
`packages/luxar/src/luxar/io/ordering.py` lines 616-682

### C. Compiler Integration
`packages/luxar/src/luxar/io/compiler.py` lines 1457-1570

### D. Viewer Segment Query
`packages/luxar-viewer/src/data/lines-spatial-index-loader.ts` lines 251-320

### E. Diagnostic Evidence
- Small dataset: `/Users/loic.royer/workspace/python/luxar/delme/diagnose_vertex_order.py` output
- Large dataset: Browser console logs from viewer testing
- Viewer diagnostic logging: `lines-spatial-index-loader.ts` lines 190-218
