# GSplats 4D Timelapse Visibility Bug — Root Cause Analysis

## Status: ACTIVE INVESTIGATION

**Date**: 2026-02-28
**Dataset**: `demo_gsplats_4d_celegans_tracking.py` (50 gsplat nodes, one per timepoint)
**Symptom**: Only first 1-2 timepoints visible; later timepoints dark despite data loading successfully

---

## Key Finding: Amplitudes ~100x Too Low After nD→3D Processing

The data loads correctly. The spatial index finds the right chunks. The worker processes all splats. But the **amplitudes in the final geometry are ~100x too low** — too dim to see on screen.

### Measured Values (browser diagnostic at slider position ~23)

```
t0022: ampMax = 0.0000039  (should be ~0.1)
t0023: ampMax = 0.0009     (should be ~0.1)  ← this is the ON-SLICE node
t0024: ampMax = 0.0000036  (should be ~0.1)
```

### Confirmed Facts

- `slicePosition[3] = 23` (confirmed via `sceneDimsManager.getDims().currentStep`)
- `instanceCount > 0` in mesh geometry (data IS reaching GPU buffers)
- Worker reports "1791/1791 visible" — all splats pass the 1e-6 threshold
- On-disk data is correct (Python zarr verification shows correct time coordinates)

---

## Two Contributing Factors

### Factor 1: Architectural Mismatch — Gaussian Attenuation on Discrete Time Dimension

**This is likely the fundamental design issue.**

Each gsplat node represents a **snapshot at a single discrete timepoint**. The splats are fitted in 3D and have **no natural extent in time**. But the demo uses:

```python
fill_sigma={"time": 0.3}
```

This artificially gives every splat a Gaussian extent in the time dimension with sigma=0.3. Combined with the discrete time step of 1.0, this creates an extremely narrow Gaussian:

| Time offset from center | Attenuation (sharpness=2.0) |
|---|---|
| 0.0 (exact) | 1.000 |
| 0.1 | 0.946 |
| 0.3 | 0.607 |
| 0.5 | 0.249 |
| 1.0 | 0.004 |

**The problem**: Any tiny mismatch between the loaded time coordinate and the slicePosition causes severe attenuation. With sigma=0.3, a mismatch of just 0.92 time units produces attenuation of 0.009 — killing 99.1% of the signal.

**Why this matters**: The viewer's nD→3D processing treats ALL hidden dimensions with continuous Gaussian attenuation. There's no concept of "this dimension is discrete — skip attenuation." For discrete time steps with one-node-per-timepoint, the correct behavior is:
1. Use the spatial index to load only the relevant timepoint's data (already works)
2. **Do NOT attenuate** based on time distance — loaded splats should be fully visible

### Factor 2: Possible Data Corruption From zarrita 404 Errors

The ChunkPrefetcher (`chunk-prefetcher.ts:168-206`) generates adjacent chunk requests (±1 per dimension) **without bounds validation**, causing 404 errors for non-existent chunks. While zarrita's own `BasicIndexer` computes correct chunk coordinates, the prefetcher's invalid requests may interfere with data assembly.

**Evidence that 404s may corrupt data**:
- The back-computed time offset (0.92) doesn't match any clean integer or half-integer offset
- If zarrita's internal buffer includes zero-filled regions from 404'd chunks, the data strides could be wrong
- The "first timepoint works" pattern is consistent with data corruption: at time=0, random corrupted values near 0 produce less attenuation

**Evidence against data corruption**:
- zarrita's `BasicIndexer` source code correctly computes chunk coordinates
- The prefetcher requests go through different cache keys than the main data path
- zarrita handles missing chunks gracefully (fills with fill_value)

**Verdict**: Data corruption is still possible but unconfirmed. The diagnostic logging added to the codebase will reveal the truth when the data is reloaded.

---

## Proposed Fixes

### Fix A: Skip Attenuation for Discrete Dimensions (Architectural Fix)

In `processGSplatsTo3D`, when computing hidden dimension attenuation, **skip dimensions marked as `discrete`** in the dimension metadata. Splats loaded for discrete dimensions should be fully visible.

This requires:
1. Passing dimension metadata through to the processing function
2. Filtering `hiddenDims` to exclude discrete dimensions before computing Mahalanobis distance
3. The spatial index AABB query already handles discrete dimension filtering (tolerance=0.5)

**Impact**: Correct behavior for all discrete-dimension scenarios. Loaded splats at discrete positions are either loaded (visible) or not loaded (invisible). No intermediate attenuation.

### Fix B: Use Large fill_sigma for Discrete Dimensions (Quick Workaround)

Change the demo from `fill_sigma={"time": 0.3}` to `fill_sigma={"time": 10.0}` or larger.

With sigma=10.0:
- diff=1.0 → attenuation = exp(-0.5 * (1/10)²) = 0.995 (essentially invisible reduction)
- diff=5.0 → attenuation = 0.88 (still bright enough)

**Downsides**:
- Chunk bounds extend ±30 time units → more nodes loaded per query → performance impact
- Not a real fix, just masks the symptom

### Fix C: Use `extend_to_all=["time"]` Per Node (Wrong Approach)

This would make ALL nodes visible at ALL times — 50 timepoints rendered simultaneously. Not correct for time-lapse visualization.

### Fix D: Fix ChunkPrefetcher Bounds Checking

Add bounds validation to `getAdjacentChunks()` in `chunk-prefetcher.ts`:

```typescript
// Before generating adjacent chunk, check bounds
if (maxIndices && newIndices[dim] >= maxIndices[dim]) continue;
```

This eliminates the 404 noise but may not fix the data corruption (if any).

### Fix E: Don't Embed fill_sigma at All for Discrete Dimensions

On the Python side, when `fill={"time": t}` is used for a discrete dimension, don't add any Cholesky extent in that dimension. Instead, mark the dimension in metadata as "discrete fill" so the viewer knows to skip attenuation.

This is the cleanest long-term solution but requires changes to both Python encoder and viewer processor.

---

## 404 Error Details

### Source: ChunkPrefetcher (NOT zarrita)

The 404 errors come from `chunk-prefetcher.ts:168-206` (`getAdjacentChunks`), which generates ±1 in each dimension without checking array bounds:

```typescript
for (let dim = 0; dim < indices.length; dim++) {
  for (const delta of [-1, 1]) {
    const newIndices = [...indices];
    newIndices[dim] += delta;
    if (newIndices[dim] < 0) continue;  // Only checks negative, NOT upper bound!
    adjacent.push(generateKey(newIndices));
  }
}
```

zarrita's own `BasicIndexer` (`zarrita/src/indexing/indexer.ts`) correctly computes:
- `dim_chunk_ix_from = Math.floor(start / chunk_len)`
- `dim_chunk_ix_to = Math.ceil(stop / chunk_len)`

This produces the correct chunk set. The extra invalid requests come solely from the prefetcher.

### Pattern Examples

| Array | Shape | Chunks | Valid Chunks | Prefetcher 404s |
|-------|-------|--------|-------------|-----------------|
| centers | (2096, 4) | (1024, 4) | 0.0, 1.0, 2.0 | 0.1, 1.1, 2.1, 3.0 |
| chunk_bounds | (3, 4, 2) | (3, 4, 2) | 0.0.0 | 0.0.1, 0.1.0, 1.0.0 |
| sharpnesses | (1,) | (1,) | 0 | 1 |
| colors | (1, 3) | (1, 3) | 0.0 | 0.1, 1.0 |

---

## What We Verified Is NOT The Problem

1. **Viewer code logic**: Exhaustive trace of spatial query, nD→3D processing, Cholesky factorization, GPU buffer pool — all mathematically correct
2. **Python encoding**: On-disk data verified correct — chunks aligned, encoding metadata present, content_hash set
3. **Consolidated metadata (.zmetadata)**: Correct chunk sizes (`[1024, 4]`) in the consolidated file
4. **OPFS/HTTP caching**: Cleared via `?clear-cache`, same issue on fresh machines
5. **GPU buffer pool**: Disabled and tested without — same issue
6. **THREE.js rendering**: Mesh has correct `instanceCount > 0`, geometry attributes allocated, material compiled
7. **zarrita chunk computation**: `BasicIndexer` correctly computes chunk coordinates from shape/chunks metadata

---

## Diagnostic Logging Added

### In `gsplats-spatial-index-loader.ts` (after data loading)

Logs raw loaded values including time coordinate:
```
DATA CHECK /gsplats_t0023: 2096 splats loaded, ndim=4,
  first_center=[x, y, z, TIME], time_col[0]=23.000, amp[0]=0.100000, ampMax100=0.100000
```

If `time_col[0]` is NOT 23.0, the data is corrupted during loading.
If `time_col[0]` IS 23.0 but post-processing amplitudes are low, the processing has a bug.

### In `scene-loader.ts` (after nD→3D processing)

Logs pre/post amplitude comparison:
```
GSplats /gsplats_t0023: 2096/2096 visible, ampMax pre=0.100000 post=0.100000
  (attenuation=1.0000), slice=[0, 0, 0, 23], worker=true
```

If `attenuation` is not ~1.0 for the on-slice node, the processing is wrong.

---

## Investigation Timeline

### Phase 1: Code Review (gsplats-processor.ts)
- Fixed performance bugs: double Cholesky computation, per-splat allocations, workspace reuse
- Added 3 regression tests for workspace safety
- **Result**: Code is correct

### Phase 2: Diagnostic Logging
- Added warning log for silent `updateGSplatsGeometry` skip
- Added nD→3D filtering result log (loaded vs visible splat counts)
- **Result**: Data loads and processes "successfully" but with wrong amplitude values

### Phase 3: Runtime Diagnostics (browser console)
- Confirmed `instanceCount > 0` in geometry (data IS reaching GPU)
- Confirmed `slicePosition[3] = 23` (integer, correct)
- Captured actual amplitude values: **ampMax = 0.0009** (should be 0.1)
- **Result**: Attenuation is ~0.009 instead of 1.0 for on-slice splats

### Phase 4: 404 Error Source Identification
- Traced 404s to `ChunkPrefetcher.getAdjacentChunks()` (not zarrita core)
- zarrita's `BasicIndexer` computes correct chunks
- Prefetcher lacks upper-bound checking on chunk indices
- **Result**: 404s are prefetcher noise, but data corruption mechanism still unclear

### Phase 5: Architectural Analysis
- Identified fundamental mismatch: Gaussian attenuation applied to discrete time dimension
- `fill_sigma={"time": 0.3}` creates extremely narrow Gaussian (sigma < step)
- Viewer has no concept of "discrete dimension = no attenuation"
- **Result**: Even with correct data, this architecture is fragile for discrete time-lapse

---

## Key Diagnostic Commands

### Check amplitude values (browser console)
```javascript
// Set up monitor BEFORE moving slider
const iv=setInterval(()=>{window.__luxarDebug?.scene?.traverse(o=>{
  if(o.name?.startsWith('/gsplats_t002')&&o.geometry?.instanceCount>0){
    const g=o.geometry;
    const c=g.getAttribute('aCenter');
    const a=g.getAttribute('aAmplitude');
    console.log('CAUGHT',o.name,'ic:',g.instanceCount,
      'center0:',c.getX(0),c.getY(0),c.getZ(0),
      'amp0:',a.getX(0),
      'ampMax:',Math.max(...Array.from(a.array.slice(0,100))));
    clearInterval(iv)}})},50);
// Then move slider
```

### Check slice position
```javascript
JSON.stringify(window.__luxarDebug?.sceneDimsManager?.getDims()?.currentStep)
```

### Check mesh state
```javascript
const r=[]; window.__luxarDebug?.scene?.traverse(o=>{
  if(o.name?.startsWith('/gsplats_t002')){
    r.push({name:o.name,
      ic:o.geometry?.instanceCount,
      ac:o.geometry?.getAttribute('aCenter')?.count})}});
console.table(r);
```

### Verify on-disk data (Python)
```python
import zarr
store = zarr.open_group("path/to/data.zarr", "r")
g = store["gsplats_t0023"]
centers = g["centers"][:]
print(f"Time column: min={centers[:,3].min()}, max={centers[:,3].max()}")
# Should print: Time column: min=23.0, max=23.0
```

---

## Files Modified During Investigation

| File | Changes |
|------|---------|
| `packages/luxar-viewer/src/data/gsplats-processor.ts` | Performance: eliminated double Cholesky computation, pre-allocated workspace buffers, cached attenuation values |
| `packages/luxar-viewer/src/data/scene-loader.ts` | Diagnostic: warning log for silent update skip, pre/post amplitude comparison log |
| `packages/luxar-viewer/src/data/gsplats-spatial-index-loader.ts` | Diagnostic: DATA CHECK log showing raw loaded values including time coordinate |
| `packages/luxar-viewer/src/tests/unit/data/gsplats-processor.test.ts` | 3 new tests: workspace reuse safety, cross-correlated Cholesky, mixed visibility |
| `packages/luxar-viewer/src/config/index.ts` | Temporary: GPU buffer pool disabled for testing (should be re-enabled) |

---

## Next Steps

1. **Regenerate data** and check the new diagnostic logs (`DATA CHECK` and `ampMax pre/post`)
2. If time coordinates load correctly but attenuation is still wrong: **implement Fix A** (skip attenuation for discrete dims)
3. If time coordinates are corrupted: investigate zarrita's data assembly with prefetcher active
4. **Fix the ChunkPrefetcher** bounds checking (Fix D) regardless — it's a clear bug causing unnecessary 404 traffic
5. Consider **Fix E** long-term: don't embed fill_sigma for discrete dimensions
