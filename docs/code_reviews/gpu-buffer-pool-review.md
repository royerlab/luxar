# GPU Buffer Pool - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/rendering/gpu-buffer-pool.ts`
- `packages/luxar-viewer/src/tests/unit/rendering/gpu-buffer-pool.test.ts`
- `packages/luxar-viewer/src/tests/unit/integration/gpu-pool-integration.test.ts`
- `packages/luxar-viewer/src/tests/unit/performance/gpu-pool-performance.test.ts`
- Integration in `scene-loader.ts` and `geometry-update-manager.ts`

---

## Executive Summary

The GPU Buffer Pool is well-designed for geometry reuse, with type-aware matching and capacity bucketing. However, the review found **one critical issue** where release methods are never called from application code, making the pool's recycling mechanism dead code. Several high-severity issues were also identified.

---

## Critical Findings

### CRITICAL: Release methods are never called — pool recycling is dead code

**Severity**: CRITICAL
**File**: All consumers (`scene-loader.ts`, `geometry-update-manager.ts`)

`releasePointsGeometry()`, `releaseLinesGeometry()`, and `releaseGSplatsGeometry()` are **never called** from application code. They are only called from the pool's own `acquirePointsGeometry()` (when types change) and from test files.

**Impact**: The pool's recycling mechanism is dead code. Geometries are acquired and stay "active" forever. The `pointBuffers`, `lineBuffers`, and `gsplatBuffers` Maps remain empty during normal operation. LRU eviction never runs because there are no pooled buffers to evict.

The pool still provides value via **same-node reuse** (returning the same active geometry when requested again with the same nodeId), but cross-node recycling never occurs.

**Recommendation**: Add `release*Geometry()` calls when scene nodes are removed or scenes change.

---

## HIGH Findings

### HIGH: scene-loader.dispose() does not dispose the GPU buffer pool

**File**: `scene-loader.ts`, lines 2488-2531

The `dispose()` method cleans up loaders, caching stores, and L0 cache, but never disposes the GPU buffer pool. The `_gpuBufferPool` is left dangling with all its active geometries unreleased.

**Fix needed**: Add `this._gpuBufferPool?.dispose()` to `scene-loader.ts` dispose method.

### HIGH: frameCount increments per acquire call, not per render frame

**File**: `gpu-buffer-pool.ts`, lines 186, 513, 728

Each `acquire*()` call increments `this.frameCount++`. With N scene nodes, `frameCount` advances by N per actual render frame. The eviction threshold of 300 "frames" becomes ~300/N real frames (~17 frames with 18 nodes).

Currently masked by the release-never-called issue, but will become real once lifecycle is fixed.

**Recommendation**: Use a separate `tickFrame()` method called once per render frame.

### HIGH: No validation that count <= capacity in update methods

**File**: `gpu-buffer-pool.ts`, lines 449-503, 659-717, 863-910

`update*Geometry()` methods are public and can be called independently of `acquire*Geometry()`. If `count` exceeds the geometry's buffer capacity, `TypedArray.set()` throws a `RangeError`.

---

## MEDIUM Findings

### MEDIUM: Pool lookup is O(N) across all buckets

Despite bucketing by size, the pool search iterates across ALL buckets. With `maxPoolSize` of 20, impact is negligible.

### MEDIUM: No buckets above 1M elements

`getBucket()` maps everything above 500K to the 1M bucket. A 2M and 10M geometry share the same bucket, causing potential 5x memory waste.

### MEDIUM: Temporary geometry allocation in bounding box computation

`updateLinesGeometry` and `updateGSplatsGeometry` create temporary `Float32Array` + `BufferGeometry` on every update to compute bounding boxes — the exact allocation the pool is designed to avoid.

### MEDIUM: Float16Array not handled in detectAttributeTypes

`PositionArray` types include `Float16Array`, but `detectAttributeTypes` always assumes `Float32Array` for non-Uint8 data.

---

## LOW Findings

### LOW: computeBoundingBox called twice for Points

`updatePointsGeometry()` calls `computeBoundingBox()` internally, then scene-loader immediately overwrites it with `data.metadata.bounds.clone()`.

### LOW: No zero-count early return in update methods

When `count=0`, update methods still run through all attribute updates.

---

## Test Coverage Gaps

1. No test coverage for `updateLinesGeometry` or `updateGSplatsGeometry`
2. No buffer overflow protection test
3. No zero-count geometry test
4. No dispose correctness verification test
5. No type-change path test
6. No bounding box correctness test
