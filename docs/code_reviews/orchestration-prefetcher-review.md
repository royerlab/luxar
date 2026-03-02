# Two-Level Caching Store & Chunk Prefetcher - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/cache/two-level-caching-store.ts`
- `packages/luxar-viewer/src/cache/chunk-prefetcher.ts`
- `packages/luxar-viewer/src/tests/unit/cache/two-level-caching-store.test.ts`
- `packages/luxar-viewer/src/cache/chunk-prefetcher.test.ts`

---

## Executive Summary

The orchestration layer is well-structured with clean L1→L2→L3 cascade. One critical prefetch amplification bug was found and **fixed**.

---

## Bugs Found and Fixed

### CRITICAL: Cascading prefetch amplification — **FIXED**

**Severity**: CRITICAL
**File**: `chunk-prefetcher.ts`, `onAccess()` method

When the prefetcher calls `store.get(key)` for a prefetch, and the store fetches from HTTP (L3), the store calls `this.prefetcher?.onAccess(key)` on the *prefetched* chunk. This triggers neighbor expansion on the prefetched chunk, cascading until the entire dataset is fetched.

**Cascade analysis** (3D dataset, N chunks per dimension):
```
User requests [1,1,1]
  → onAccess enqueues 6 neighbors
  → Each neighbor fetched → onAccess → 6 more neighbors each
  → Cascades to O(N^3) fetches instead of the intended 6
```

The existing `queue + inFlight` deduplication only prevents duplicates within a single wave. Once a prefetch completes, its neighbors trigger new waves.

**Fix applied**: Added a `seen` set that tracks all keys whose neighbors have already been expanded. Once a key's neighbors are enqueued, `onAccess()` returns immediately for that key on subsequent calls.

---

## Other Findings

### HIGH: No concurrent get() deduplication

**Severity**: HIGH
**File**: `two-level-caching-store.ts`, `get()` method

Two concurrent callers requesting the same key both miss L1, both miss L2, and both issue independent HTTP fetches. Happens when zarrita reads shared metadata or when prefetcher and user request race.

**Recommendation**: Add an in-flight promise map to coalesce concurrent requests.

### HIGH: No init() guard against double initialization

**Severity**: HIGH
**File**: `two-level-caching-store.ts`, `init()` method

No guard prevents `init()` from being called concurrently. Could create multiple `OPFSStore` instances or race between `clearAll()` and `validateCache()`.

### HIGH: `?no-cache` still reads L1 unconditionally

**Severity**: HIGH
**File**: `two-level-caching-store.ts`, line 239

When `?no-cache` is set, L1 reads are still performed (but never populated). If `?no-cache` is added mid-session, previously cached L1 entries would still be served.

### MEDIUM: L2 write errors silently swallowed

`this.l2Store.set(key, data).catch(() => {})` makes L2 write failures invisible.

### MEDIUM: Bandwidth is lifetime average

After initial load burst, reported bandwidth decreases toward zero indefinitely. A sliding window would be more useful.

### MEDIUM: Prefetch explosion for high-dimensional data

For D-dimensional data, each access generates 2*D adjacent chunks. For 16D data that's 32 prefetches per access. Consider capping or only prefetching along navigation dimensions.

### LOW: L1 not cleared after content hash mismatch

`validateCache()` clears L2 but not L1 on hash mismatch. Not a concern for normal page reloads (L1 is session-only) but could serve stale data if validation occurs mid-session.

---

## Positive Observations

1. **Correct cascade logic** — L1→L2→L3 ordering with proper promotion/population
2. **Content hash bypass** — Fetches `.zattrs` directly from HTTP for validation
3. **Clean AsyncReadable interface** — Seamless zarrita integration
4. **Race-condition safe queue processing** — `processing` flag + `queueMicrotask` pattern
5. **Bounds checking** — `registerArrayBounds()` prevents out-of-range prefetch
