> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](../README.md) for status labels and retention policy.

# Phase 4: GPU Buffer Pool - Implementation Plan

**Date**: 2025-12-24
**Status**: ✅ **ALL PREREQUISITES MET, READY TO IMPLEMENT**
**Context**: Built on verified Phases 1-3 foundation

---

## 🎯 Implementation Status

### ✅ Prerequisites Complete

**Architecture Understanding:**
- ✅ Current geometry creation patterns analyzed
- ✅ Update flow for all three types documented
- ✅ Integration points identified in scene-loader.ts
- ✅ Material attribute layouts verified
- ✅ Memory lifecycle understood

**Phases 1-3 Foundation:**
- ✅ 1375 tests passing (1365 TS + 10 Rust)
- ✅ Worker infrastructure tested
- ✅ WASM module built (17KB, optimized)
- ✅ All loaders support async worker queries

---

## 📋 Phase 4 Deliverables

Per spec v3.6.0 (section 9, lines 3115-3127):

1. **`gpu-buffer-pool.ts`** with all three geometry types
   - PooledBuffer interface
   - GPUBufferPool class
   - Size-based bucketing
   - LRU eviction (300 frames)
   - Capacity growth (1.5x)

2. **SceneManager Integration**
   - Modify `scene-loader.ts:updatePointsGeometry()`
   - Modify `scene-loader.ts:updateLinesGeometry()`
   - Modify `scene-loader.ts:updateGSplatsGeometry()`
   - Add buffer pool initialization/disposal

3. **WebGL Context Loss Handling**
   - Dispose all pooled geometries on context loss
   - Reinitialize pool on context restore

4. **Performance Validation**
   - Unit tests for pool logic
   - Integration tests for reuse
   - E2E tests for actual performance

---

## 🔍 Current Architecture Analysis

### Points Geometry Update (scene-loader.ts:1339-1381)

**Current Flow:**
```typescript
updatePointsGeometry(path: string, data: PointsData): void {
  const points = this.rootGroup.getObjectByName(path);
  const oldGeometry = points.geometry;

  // 1. DISPOSE old geometry
  oldGeometry.dispose();

  // 2. CREATE new geometry from scratch
  const newGeometry = this.createGeometry(data);

  // 3. ASSIGN new geometry
  points.geometry = newGeometry;
}
```

**Issues:**
- ❌ GPU memory allocation every update
- ❌ CPU overhead creating BufferAttributes
- ❌ No buffer reuse across updates

**With GPU Buffer Pool:**
```typescript
updatePointsGeometry(path: string, data: PointsData): void {
  const points = this.rootGroup.getObjectByName(path);
  const oldGeometry = points.geometry;

  // 1. RETURN old geometry to pool (or dispose if oversized)
  if (oldGeometry) {
    this.bufferPool.releasePointsGeometry(path, oldGeometry);
  }

  // 2. ACQUIRE from pool (reuses if available, allocates if needed)
  const geometry = this.bufferPool.acquirePointsGeometry(path, data.metadata.loadedPoints);

  // 3. UPDATE attributes in place (no allocation)
  this.updatePointsAttributesInPlace(geometry, data);

  // 4. ASSIGN (might be same geometry if size unchanged)
  points.geometry = geometry;
}
```

**Benefits:**
- ✅ Zero GPU allocation if geometry reused
- ✅ In-place attribute updates via `.set()`
- ✅ ~2-3ms per update vs ~15-20ms current

### Lines Geometry Update (scene-loader.ts:474-534)

**Current Flow:**
```typescript
updateLinesGeometry(path: string, data: LoadedLinesData, viewState): void {
  const mesh = this.rootGroup.getObjectByName(path);
  const oldGeometry = mesh.geometry;

  // 1. DISPOSE old
  oldGeometry.dispose();

  // 2. CREATE new via createInstancedLinesMesh()
  const processed = buildInstanceBuffers(data, viewState);
  const newMesh = createInstancedLinesMesh(processed, material);

  // 3. COPY geometry reference
  mesh.geometry = newMesh.geometry;

  // 4. CLEANUP temp mesh
  newMesh.geometry = new THREE.BufferGeometry(); // Prevent double disposal
}
```

**Issues:**
- ❌ Creates temporary mesh just to get geometry
- ❌ Allocates all instance attributes every time
- ❌ Complex geometry juggling

**With GPU Buffer Pool:**
```typescript
updateLinesGeometry(path: string, data: LoadedLinesData, viewState): void {
  const mesh = this.rootGroup.getObjectByName(path);

  // 1. RETURN old to pool
  if (mesh.geometry) {
    this.bufferPool.releaseLinesGeometry(path, mesh.geometry);
  }

  // 2. PROCESS data (nD → per-segment attributes)
  const processed = buildInstanceBuffers(data, viewState);

  // 3. ACQUIRE from pool
  const geometry = this.bufferPool.acquireLinesGeometry(path, processed.segmentCount);

  // 4. UPDATE instance attributes in place
  this.updateLinesInstanceAttributesInPlace(geometry, processed);

  // 5. ASSIGN
  mesh.geometry = geometry;
}
```

**Benefits:**
- ✅ No temporary mesh creation
- ✅ Reuses InstancedBufferGeometry
- ✅ In-place instance attribute updates
- ✅ ~5-8ms per update vs ~25-30ms current

### GSplats Geometry Update (scene-loader.ts:545-590)

**Current Flow:**
Already partially optimized via `updateInstancedGSplatsMesh()` in gsplat-material.ts:
- If size unchanged: Updates in place ✓
- If size changed: Recreates geometry ✗

**With GPU Buffer Pool:**
Even more optimized - always uses pool, eliminating size-change reallocations.

---

## 🏗️ GPU Buffer Pool Architecture

### Class Structure

```typescript
export class GPUBufferPool {
  // Type-specific pools (size-bucketed)
  private pointBuffers: Map<number, PooledBuffer[]>;
  private lineBuffers: Map<number, PooledBuffer[]>;
  private gsplatBuffers: Map<number, PooledBuffer[]>;

  // Active assignments (nodeId → geometry)
  private activeBuffers: Map<string, PooledBuffer>;

  // Frame tracking for LRU eviction
  private frameCount: number;

  // Configuration
  private maxPoolSize: number;
  private evictionFrames: number;

  // Statistics
  private stats: {
    allocations: number;
    reuses: number;
    evictions: number;
    capacityGrowths: number;
  };

  // Methods
  acquirePointsGeometry(nodeId: string, count: number): THREE.BufferGeometry
  acquireLinesGeometry(nodeId: string, count: number): THREE.InstancedBufferGeometry
  acquireGSplatsGeometry(nodeId: string, count: number): THREE.InstancedBufferGeometry

  releasePointsGeometry(nodeId: string, geometry: THREE.BufferGeometry): void
  releaseLinesGeometry(nodeId: string, geometry: THREE.InstancedBufferGeometry): void
  releaseGSplatsGeometry(nodeId: string, geometry: THREE.InstancedBufferGeometry): void

  updatePointsGeometry(geometry: THREE.BufferGeometry, data: PointsData): void
  updateLinesGeometry(geometry: THREE.InstancedBufferGeometry, data: ProcessedLinesData): void
  updateGSplatsGeometry(geometry: THREE.InstancedBufferGeometry, data: PackedGSplatsData): void

  evictUnused(): number
  dispose(): void
  getStats(): PoolStats
}
```

### Pooling Strategy

**Size Bucketing:**
- Points: Buckets at 1K, 5K, 10K, 50K, 100K, 500K, 1M
- Lines: Buckets at 500, 2K, 10K, 50K, 100K
- GSplats: Buckets at 500, 2K, 10K, 50K, 100K

**Capacity Matching:**
- If needed ≤ capacity: REUSE
- If needed > capacity: GROW (reallocate attributes at 1.5x)
- If needed << capacity (>2x waste): ALLOCATE new from smaller bucket

**LRU Eviction:**
- Track lastUsedFrame for each buffer
- When pool exceeds maxPoolSize:
  - Find buffers unused for >300 frames
  - Dispose geometry and remove from pool

---

## 🧪 Testing Strategy

### Unit Tests (`tests/unit/rendering/gpu-buffer-pool.test.ts`)

**Core Pool Logic:**
- Buffer acquisition and reuse
- Size bucketing
- Capacity growth
- LRU eviction
- Statistics tracking

**Type-Specific Tests:**
- Points geometry creation and update
- Lines instanced geometry handling
- GSplats instanced geometry handling

**Edge Cases:**
- Empty geometry (0 count)
- Size changes (grow/shrink)
- Rapid updates
- Pool exhaustion
- Context loss simulation

### Integration Tests

**SceneLoader Integration:**
- Verify geometries reused across updates
- Verify attributes updated correctly
- Verify old geometries returned to pool
- Verify no memory leaks

---

## 📊 Expected Performance Improvements

| Metric | Before Phase 4 | After Phase 4 | Improvement |
|--------|----------------|---------------|-------------|
| GPU buffer allocation (Points) | Every update (~15-20ms) | 0ms (reuse) | **Eliminate** |
| GPU buffer allocation (Lines) | Every update (~20-25ms) | 0ms (reuse) | **Eliminate** |
| Attribute upload (if size unchanged) | Full upload | Partial update | **2-3x faster** |
| Memory spikes | Old + new held briefly | Pool-managed | **Smoother** |
| VRAM usage (1M points) | ~300MB peak | ~200MB stable | **33% reduction** |

---

## 🔧 Integration Plan

### Step 1: Create GPUBufferPool Class
- Implement all acquire/release/update methods
- Add size bucketing and LRU eviction
- Comprehensive error handling

### Step 2: Add Logging Module
```typescript
// src/utils/log.ts
GPU_BUFFER_POOL: 'GPUBufferPool'
```

### Step 3: Integrate into scene-loader.ts
- Initialize pool in constructor
- Modify updatePointsGeometry to use pool
- Modify updateLinesGeometry to use pool
- Modify updateGSplatsGeometry to use pool
- Add pool disposal in cleanup

### Step 4: Handle WebGL Context Loss
- Listen to context loss events
- Dispose all pooled geometries
- Clear pools
- Re-initialize on context restore

### Step 5: Test Thoroughly
- Unit tests for pool logic
- Integration tests with actual scene loading
- E2E tests in browser
- Performance benchmarks

---

## ⚠️ Implementation Considerations

### Critical Details from Analysis

1. **Attribute Types Matter:**
   - Points: Float16/Float32/Uint8/Uint16 (need conversion + normalization)
   - Lines: All Float32 instance attributes
   - GSplats: All Float32 instance attributes

2. **Normalization Flags:**
   - Uint8/Uint16 arrays use `BufferAttribute(..., normalized: true)`
   - GPU auto-normalizes [0, 255] → [0, 1]
   - Shader applies radiusScale/sharpnessScale uniforms

3. **Lines Complexity:**
   - Uses `THREE.InstancedBufferGeometry` (not `THREE.InstancedMesh`)
   - Base quad geometry shared across all instances
   - 11 instance attributes per segment (aStartPos, aEndPos, etc.)

4. **GSplats Complexity:**
   - Already uses `updateInstancedGSplatsMesh()` with conditional updates
   - Build on existing pattern rather than replacing

5. **Memory Safety:**
   - Always dispose unused geometries (no leaks)
   - Respect maxPoolSize limit
   - Monitor VRAM usage

---

## 🎯 Success Criteria for Phase 4

Per spec:
- [x] Architecture analyzed and understood
- [ ] `gpu-buffer-pool.ts` implementation (based on spec lines 2050-2960)
- [ ] All three geometry types supported
- [ ] Unit tests created and passing
- [ ] Integrated into scene-loader.ts
- [ ] WebGL context loss handled
- [ ] Performance validated (GPU allocation → 0ms on reuse)

---

## 📝 Recommendation

**Phase 4 is ready to implement** with full context and understanding.

**Estimated effort**: 4-6 hours for complete implementation, testing, and integration.

**Risk level**: LOW - Solid foundation from Phases 1-3, clear integration points identified, existing patterns to build on.

**Next action**: Implement `gpu-buffer-pool.ts` following spec, create comprehensive tests, integrate carefully into scene-loader.ts.

All prerequisites are met. Implementation can proceed with confidence.
