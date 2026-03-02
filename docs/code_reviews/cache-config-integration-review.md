# Cache Configuration, Types & Integration - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/cache/types.ts`
- `packages/luxar-viewer/src/cache/index.ts`
- `packages/luxar-viewer/src/config/index.ts` (cache section)
- `packages/luxar-viewer/src/config/types.ts` (CacheConfig)
- `packages/luxar-viewer/src/data/scene-loader.ts` (cache init)
- `packages/luxar-viewer/src/core/app.ts` (debug API, cleanup)
- `packages/luxar-viewer/src/tests/e2e/cache-system.spec.ts`

---

## Executive Summary

The cache configuration system is well-architected with sensible defaults and URL parameter overrides. Two critical issues were identified: the SegmentedLRU miss double-counting (**already fixed** in this review) and cache not being disposed on page unload.

---

## Critical Findings

### CRITICAL: SegmentedLRUCache.get() double-counts misses — **FIXED**

(See lru-l1-cache-review.md for details. Fix applied to `segmented-lru-cache.ts`.)

### CRITICAL: Cache not disposed on page unload

**File**: `app.ts`, lines 616-675

`LuxarApp.cleanup()` is called on `beforeunload` but does NOT dispose cache resources. `SceneLoaderManager.reset()` or `destroyAll()` is never called, meaning:
1. `OPFSStore.dispose()` never runs
2. Pending debounced metadata saves (1s delay) are lost
3. `_cache_meta.json` may not be written

**Recommendation**: Add cache disposal to `cleanup()`.

---

## HIGH Findings

### HIGH: L0 can serve stale data after L2 cache invalidation

When `validateCache()` detects a content hash mismatch and clears L2, it does NOT clear L0 (managed separately by SceneLoader). If validation occurs within a session, L0 could hold stale decompressed chunks.

### HIGH: OPFS metadata save may be lost on page close

The 1-second debounce means metadata writes pending at page close are lost. Even `dispose()` in `beforeunload` may be interrupted.

---

## MEDIUM Findings

### MEDIUM: Dual disabling mechanisms unclear

`config.cache.enabled = false` prevents TwoLevelCachingStore creation. `?no-cache` URL param disables L2 inside an existing store. L0 only checks `l0Enabled && !noCache`, not `config.cache.enabled`. The semantics are correct but confusing.

### MEDIUM: No guard against pre-init get() calls

`TwoLevelCachingStore.get()` called before `init()` silently skips L2 (since `l2Store` is `null`), degrading to L1-only without warning.

### MEDIUM: `as any` for private member access in debug API

`app.ts` uses `(loader as any).cachingStore` and `(loader as any).l0Cache` to access private members. Should use public getter methods.

### MEDIUM: `ExtendedCacheStats` type is dead code

Defined in `types.ts` and exported from `index.ts` but never imported elsewhere.

---

## Configuration Coherence

The defaults are well-chosen:
- L0: 200MB (decompressed, ~5x larger than compressed)
- L1: 100MB (compressed, adequate for typical sessions)
- L2: 2GB (OPFS, generous but within browser quotas)

URL parameter handling is consistent across all levels:
| Parameter | L0 | L1 | L2 | Prefetch |
|-----------|----|----|----|---------:|
| `?no-cache` | Disables | N/A | Disables writes | N/A |
| `?cache-debug` | Enables logging | N/A | Enables logging | N/A |
| `?clear-cache` | N/A | N/A | Clears on init | N/A |
| `?no-prefetch` | N/A | N/A | N/A | Disables |

---

## Test Strategy Assessment

**Well-covered**: Unit tests for all 7 cache modules. E2E tests cover 11 scenarios.

**Missing**:
1. L0-L1-L2 cascade with real decompression
2. Content hash invalidation end-to-end (data purge, not just console logs)
3. Concurrent `loadScene()` calls
4. OPFS quota exceeded degradation
5. Page unload metadata persistence
