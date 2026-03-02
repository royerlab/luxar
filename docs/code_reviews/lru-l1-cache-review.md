# LRU Cache & L1 Segmented LRU Cache - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/cache/lru-cache.ts`
- `packages/luxar-viewer/src/cache/segmented-lru-cache.ts`
- `packages/luxar-viewer/src/tests/unit/cache/lru-cache.test.ts`
- `packages/luxar-viewer/src/tests/unit/cache/segmented-lru-cache.test.ts`

---

## Executive Summary

The LRU cache is a clean, well-documented implementation using JavaScript Map for O(1) operations. The Segmented LRU adds intelligent metadata/chunks routing. Three bugs were found and **all three fixed**.

---

## Bugs Found and Fixed

### Bug 1: `SegmentedLRUCache.get()` double-counts misses — **FIXED**

**Severity**: CRITICAL
**File**: `segmented-lru-cache.ts`, lines 43-57

The original `get()` checked metadata segment first, then chunks. For chunk lookups (95%+ of traffic), the metadata segment recorded a spurious miss before the chunks segment was even checked. This inflated miss counts by ~2x, causing the DataLoadingMonitor to report ~45% hit rate when the true rate was ~90%.

**Fix**: Route `get()` to the correct segment using `isMetadataFile()`, matching the `set()` method's behavior.

```typescript
// BEFORE (broken):
get(key) {
  const metadataHit = this.metadata.get(key);  // Records miss for chunk keys!
  if (metadataHit) return metadataHit;
  return this.chunks.get(key);
}

// AFTER (fixed):
get(key) {
  if (SegmentedLRUCache.isMetadataFile(key)) return this.metadata.get(key);
  return this.chunks.get(key);
}
```

### Bug 2: Negative `chunksSize` when `totalSize < MIN_METADATA_SIZE` — **FIXED**

**Severity**: HIGH
**File**: `segmented-lru-cache.ts`, line 28

When `totalSize < 10MB`, `metadataSize = 10MB` (floor), `chunksSize = totalSize - 10MB = negative`. A negative `maxSize` in LRU cache causes every `set()` to immediately evict the previous entry.

**Fix**: `Math.max(0, totalSize - metadataSize)`.

### Bug 3: `LRUCache.delete()` uses truthiness check instead of `!== undefined` — **FIXED**

**Severity**: MEDIUM
**File**: `lru-cache.ts`, line 120

`if (value)` fails for falsy values like `0`, `""`, `false`. While current usage stores `Uint8Array` objects (always truthy), this is a latent bug in the generic API.

**Fix**: Changed to `if (value !== undefined)`.

---

## Additional Findings

### `LRUCache.set()` allowed oversized items — **FIXED**

**Severity**: HIGH — Items larger than `maxSize` would empty the cache and still be added, causing `currentSize` to permanently exceed `maxSize`. Fixed by rejecting oversized items.

### Class header claims O(1) but `set()` is O(k)

**Severity**: LOW — `set()` is O(k) where k = evictions needed (typically 0-2). This is documented in the `@performance` tag but not in the class-level comment.

### `clear()` resets hit/miss/eviction counters

**Severity**: LOW — Inconsistent with `TwoLevelCachingStore` which does NOT reset network stats on clear. Minor UX inconsistency in the monitoring UI.

### Missing test coverage

- Falsy value deletion (bug 3)
- `has()` non-promotion verification
- `totalSize < 10MB` (bug 2)
- Cross-segment miss inflation (bug 1)

---

## Positive Observations

1. **Map-based LRU** — Correct O(1) for get/delete with insertion-order reordering
2. **Clean statistics API** — hit/miss/eviction tracking with getter accessors
3. **Well-chosen metadata patterns** — `.zmetadata`, `.zarray`, `.zattrs`, `zarr.json` covers all zarr versions
