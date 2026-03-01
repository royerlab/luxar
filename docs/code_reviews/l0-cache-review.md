# L0 Decompressed Chunk Cache - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/cache/decompressed-chunk-cache.ts`
- `packages/luxar-viewer/src/cache/cached-zarr-array.ts`
- `packages/luxar-viewer/src/cache/lru-cache.ts` (dependency)
- `packages/luxar-viewer/src/tests/unit/cache/decompressed-chunk-cache.test.ts`
- `packages/luxar-viewer/src/tests/unit/cache/cached-zarr-array.test.ts`

---

## Executive Summary

The L0 decompressed chunk cache is well-designed and well-implemented. The core LRU cache has correct O(1) operations, the ES6 Proxy wrapper is carefully crafted to handle zarrita's private field access patterns, and the test coverage is solid. However, the review found **one genuine concurrency bug** (duplicate decompression on concurrent getChunk calls), several medium-severity design concerns, and a handful of minor issues.

**Issue Count**: 1 CRITICAL, 2 HIGH, 5 MEDIUM, 4 LOW

---

## CRITICAL Issues

### 1.1 Concurrent `getChunk()` calls cause duplicate decompression (thundering herd)

**Severity**: CRITICAL
**File**: `cached-zarr-array.ts`, lines 94-124
**Status**: Not yet fixed (requires careful design)

When two concurrent callers request the same chunk simultaneously, both see a cache miss and both call `target.getChunk()`, triggering duplicate Blosc decompression:

```typescript
// Caller A: cache.get(key) -> undefined (miss)
// Caller B: cache.get(key) -> undefined (miss)  <-- race: both see miss
// Caller A: await target.getChunk(chunkCoords)   <-- decompresses
// Caller B: await target.getChunk(chunkCoords)   <-- decompresses AGAIN
```

**Real-world severity**: Limited with current architecture (each zarr array is wrapped independently with unique `arrayPath` values). The thundering herd only occurs if the same wrapped array's `getChunk()` is called concurrently for identical `chunkCoords` — unlikely but possible during rapid re-renders.

**Recommended Fix**: Add an in-flight promise map to deduplicate concurrent requests:
```typescript
const inFlight = new Map<string, Promise<ChunkResult>>();
// Check inFlight before calling target.getChunk()
```

---

## HIGH Issues

### 2.1 LRU cache allowed items larger than maxSize

**Severity**: HIGH — **FIXED**
**File**: `lru-cache.ts`, line 99

Items larger than `maxSize` would empty the cache via eviction, then be added unconditionally, causing `currentSize` to permanently exceed `maxSize`.

**Fix applied**: Added early return when `size > this.maxSize`.

### 2.2 `parseKey` with colon separator in array paths

**Severity**: HIGH (downgraded on analysis)
**File**: `decompressed-chunk-cache.ts`, lines 219-230

The `lastIndexOf(':')` approach correctly handles colons in array paths (e.g., `http://localhost:8080/data:0,1`). This is actually correct behavior but deserves a documentation comment.

---

## MEDIUM Issues

### 3.1 Shared mutable data: cached chunks are not defensively copied

Consumers could mutate the returned TypedArrays in-place, corrupting subsequent cache hits. Document as a contract: "consumers MUST NOT mutate returned arrays."

### 3.2 Proxy property whitelist is fragile and version-dependent

The hardcoded list of 10 properties will break silently if zarrita adds new getters with private field access. Consider pinning zarrita version and adding upgrade notes.

### 3.3 `LRUCache.delete()` mishandled falsy stored values

**FIXED**: Changed `if (value)` to `if (value !== undefined)`.

### 3.4 `has()` does not update LRU order

Design choice consistent with Redis. Worth documenting.

### 3.5 `DEFAULT_MAX_SIZE` evaluated at module load time

Evaluated once at import time from `config.cache.l0MaxSizeMB`. Not a practical issue since production code passes the size explicitly.

---

## LOW Issues

### 4.1 `parseKey` does not roundtrip with empty chunk coordinates

`makeKey('/data', [])` produces `"/data:"`, but `parseKey('/data:')` returns `null`.

### 4.2 `METADATA_OVERHEAD` constant underestimates actual overhead

Set to 64 bytes, actual is ~180-280 bytes. Negligible at scale (~0.4% of budget).

### 4.3 Missing test: concurrent `getChunk()` for same key

No test exercises the thundering herd scenario.

### 4.4 Missing test: retry after `getChunk` error

Error test verifies propagation but not that a retry succeeds.

---

## Positive Observations

1. **Excellent Proxy design** — `Reflect.get(target, prop, target)` with property whitelist handles ES6 Proxy + private field interactions correctly
2. **Clean separation of concerns** — `LRUCache` (generic), `DecompressedChunkCache` (domain logic), `wrapWithCache` (zarrita integration)
3. **Double-wrap prevention** — `CACHE_MARKER` symbol prevents accidental double-wrapping
4. **Good error propagation** — Errors from `getChunk` propagate without poisoning the cache
