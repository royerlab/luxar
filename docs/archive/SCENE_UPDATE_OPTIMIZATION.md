> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# Scene Update Optimization Design

This document captures optimization ideas for the THREE.js scene update pipeline and the profiling infrastructure needed to measure them.

---

## 1. Current State Summary

### Update Pipeline (per slider move)

```
Slider Move
    │
    ├─► Points Updates (parallel per node)
    │     ├─► loader.updateView()
    │     │     ├─► Spatial index query
    │     │     ├─► Chunk loading (L1/L2 cache or network)
    │     │     ├─► Decompression
    │     │     └─► Data accumulation
    │     └─► updatePointsGeometry()
    │           ├─► GPU pool acquire/grow
    │           ├─► TypedArray copy
    │           ├─► Bounding box computation
    │           └─► needsUpdate = true
    │
    ├─► Lines Updates (parallel per node)
    │     ├─► loader.updateView()
    │     ├─► buildInstanceBuffers() (nD clipping)
    │     └─► updateLinesGeometry()
    │
    └─► GSplats Updates (parallel per node)
          ├─► loader.updateView()
          ├─► processGSplats() (nD slicing)
          ├─► packCholeskyForShader()
          └─► updateGSplatsGeometry()
```

### Current Inefficiencies

| Issue | Description | Impact |
|-------|-------------|--------|
| No query caching | Same slice position re-queries spatial index | CPU waste |
| No data fingerprinting | Can't detect "same data as before" | GPU upload waste |
| No partial updates | Entire geometry rebuilt even for small changes | Memory bandwidth |
| Redundant bounding box | Temp geometry allocated every update | GC pressure |
| Material over-creation | Different scales create separate materials | Memory |

---

## 2. Profiling Infrastructure (Priority: HIGH)

**Goal**: Replace the broken performance graph with a hierarchical timing breakdown.

### 2.1 Proposed UI: Hierarchical Timing Panel

```
┌─────────────────────────────────────────────────────────────────┐
│ PERFORMANCE                                          [Collapse] │
├─────────────────────────────────────────────────────────────────┤
│ Last Update: 47.3ms                        Avg (10): 52.1ms     │
├─────────────────────────────────────────────────────────────────┤
│ ▼ Total Update                              47.3ms    52.1ms    │
│   ▼ Points (/scene/nuclei)                  32.1ms    35.4ms    │
│     ├─ Spatial Query                         2.3ms     2.8ms    │
│     ├─ Chunk Load (3 chunks)                18.2ms    20.1ms    │
│     │   ├─ L1 Cache Hit (2)                  0.1ms     0.1ms    │
│     │   └─ Network Fetch (1)                18.1ms    20.0ms    │
│     ├─ Decompress                            4.2ms     5.1ms    │
│     ├─ Accumulate                            1.1ms     1.2ms    │
│     └─ GPU Upload                            6.3ms     6.2ms    │
│         ├─ Pool Acquire                      0.1ms     0.1ms    │
│         ├─ TypedArray Copy                   2.8ms     2.9ms    │
│         └─ Bounding Box                      3.4ms     3.2ms    │
│   ▼ Lines (/scene/tracks)                   12.4ms    13.8ms    │
│     ├─ Spatial Query                         1.1ms     1.2ms    │
│     ├─ Segment Load                          5.2ms     5.8ms    │
│     ├─ nD Clipping                           2.1ms     2.4ms    │
│     └─ GPU Upload                            4.0ms     4.4ms    │
│   ► GSplats (/scene/gaussians)               2.8ms     2.9ms    │
│   ─ Skip: extend_to_all (/scene/detector)      -         -      │
├─────────────────────────────────────────────────────────────────┤
│ Points: 1.2M visible (32%)  Lines: 45K segs  GSplats: 8K        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Model

```typescript
/**
 * Hierarchical timing entry
 */
interface TimingEntry {
  name: string;
  lastMs: number;
  avgMs: number;
  count: number;           // For averaging
  children: TimingEntry[];
  metadata?: {
    chunks?: number;
    cacheHits?: number;
    cacheMisses?: number;
    points?: number;
    segments?: number;
    splats?: number;
    skipped?: boolean;
    skipReason?: string;
  };
}

/**
 * Performance profiler for scene updates
 */
interface UpdateProfiler {
  // Start a new update cycle (called on slider move)
  beginUpdate(): UpdateSession;

  // Get aggregated timing data
  getTimings(): TimingEntry;

  // Reset averages
  reset(): void;
}

/**
 * Single update session (one slider move)
 */
interface UpdateSession {
  // Hierarchical timing - returns child session
  begin(name: string): UpdateSession;
  end(): void;

  // Add metadata to current entry
  setMetadata(meta: Partial<TimingEntry['metadata']>): void;

  // Mark as skipped (extend_to_all, etc.)
  markSkipped(reason: string): void;
}
```

### 2.3 Usage Pattern

```typescript
// In SceneLoader.updateView()
const session = this.profiler.beginUpdate();

// Points updates
for (const [path, loader] of this.loaders) {
  const pointsSession = session.begin(`Points (${path})`);

  // Check extend_to_all skip
  if (isFullyExtended) {
    pointsSession.markSkipped('extend_to_all');
    pointsSession.end();
    continue;
  }

  // Spatial query
  const querySession = pointsSession.begin('Spatial Query');
  const ranges = await loader.queryRanges(viewState);
  querySession.end();

  // Chunk loading
  const loadSession = pointsSession.begin('Chunk Load');
  loadSession.setMetadata({ chunks: ranges.length });
  const data = await loader.loadChunks(ranges);
  loadSession.setMetadata({
    cacheHits: data.cacheHits,
    cacheMisses: data.cacheMisses
  });
  loadSession.end();

  // ... etc

  pointsSession.setMetadata({ points: data.count });
  pointsSession.end();
}

session.end();
```

### 2.4 Implementation Location

| Component | File | Changes |
|-----------|------|---------|
| Profiler class | `src/profiling/update-profiler.ts` (new) | Core timing logic |
| UI component | `src/profiling/update-profiler.ts` (new) | Hierarchical tree view |
| Integration | `src/data/scene-loader.ts` | Add profiler calls |
| Integration | `src/data/*-loader.ts` | Add profiler calls to loaders |
| Monitor update | `src/ui/data-loading-monitor.ts` | Replace perf graph with tree |

### 2.5 Key Measurements

**Per-Update Timings:**
- Total update time
- Per-node breakdown (Points, Lines, GSplats)
- Spatial index query time
- Chunk load time (with cache hit/miss breakdown)
- Decompression time
- Data processing time (accumulation, nD clipping, Cholesky packing)
- GPU upload time (pool acquire, TypedArray copy, bounding box)

**Metadata:**
- Number of chunks loaded
- Cache hit rate (L1, L2, network)
- Points/segments/splats visible
- Skip reasons (extend_to_all, etc.)

---

## 3. Optimization Ideas (For Later)

### 3.1 Query Result Caching

**Idea**: Cache the result of spatial index queries keyed by slice position.

```typescript
// Cache key
const queryKey = slicePosition.map(v => v.toFixed(3)).join(',');

// Check cache
if (this.queryCache.has(queryKey)) {
  return this.queryCache.get(queryKey); // Skip query entirely
}
```

**Benefit**: Avoid re-querying when returning to a previously visited slice position.

**Complexity**: Low - simple Map cache with LRU eviction.

### 3.2 Data Fingerprinting

**Idea**: Hash the chunk indices to detect "same visible data".

```typescript
const dataFingerprint = hashChunkIndices(visibleChunks);

if (dataFingerprint === this.lastFingerprint) {
  return; // Skip GPU upload entirely
}
```

**Benefit**: Avoid GPU upload when slider moves but visible chunks don't change.

**Complexity**: Medium - need efficient hashing, careful invalidation.

### 3.3 Material Consolidation

**Idea**: Move radius scaling entirely to shader, use single material per blend mode.

```typescript
// Before: Different scales = different materials
const key = `point_${blend}_o${opacity}_r${radiusScale}`;

// After: Scales in uniform, single material per blend mode
const key = `point_${blend}_o${opacity}`;
material.uniforms.radiusScale.value = radiusScale;
```

**Benefit**: Fewer materials, simpler caching.

**Complexity**: Low - shader already supports uniforms.

### 3.4 Incremental Bounding Box

**Idea**: Track min/max during TypedArray copy instead of separate computation.

```typescript
// During copy
for (let i = 0; i < count; i++) {
  const x = positions[i * 3];
  const y = positions[i * 3 + 1];
  const z = positions[i * 3 + 2];
  min.x = Math.min(min.x, x);
  // ... etc
}
// No need for temp geometry or computeBoundingBox()
```

**Benefit**: Eliminate temp geometry allocation and extra iteration.

**Complexity**: Low - straightforward implementation.

### 3.5 GPU Buffer Pool Observability

**Idea**: Add pool stats to the performance panel.

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU BUFFER POOL                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Reuse Rate: 94.2%  (847 reuses / 901 total)                     │
│ Active: 12 buffers  Pooled: 8 buffers  Memory: ~48MB            │
│ Allocations: 54  Evictions: 3  Growths: 7                       │
└─────────────────────────────────────────────────────────────────┘
```

**Benefit**: Understand pool effectiveness, tune eviction policy.

**Complexity**: Low - stats already tracked, just need UI.

---

## 4. Implementation Priority

| Priority | Item | Reason |
|----------|------|--------|
| **1** | Hierarchical Profiler | Need measurements before optimizing |
| **2** | GPU Pool Observability | Low effort, high insight |
| **3** | Query Result Caching | High impact, low complexity |
| **4** | Material Consolidation | Low effort, cleaner code |
| **5** | Data Fingerprinting | Medium effort, high impact for nD navigation |
| **6** | Incremental Bounding Box | Low impact, nice cleanup |

---

## 5. Open Questions

1. **Profiler overhead**: Should profiling be opt-in (via URL param) or always-on?

2. **Averaging window**: Rolling average of last N updates, or exponential decay?

3. **Persistence**: Should timing data persist across page reloads for comparison?

4. **Export**: Should we support exporting timing data as JSON for analysis?

5. **Alerts**: Should the panel highlight slow operations (>16ms for 60fps budget)?

---

## Changelog

- **2025-12-26**: Initial document created
  - Documented current update pipeline
  - Proposed hierarchical profiling UI
  - Listed optimization ideas for future work
