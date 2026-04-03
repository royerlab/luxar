# Cache System Master Review Summary

**Date**: 2026-02-28
**Reviewer**: Claude Opus 4.6
**Scope**: All caching mechanisms in luxar-viewer

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Data Pipeline                      │
│                                                      │
│  L0 DecompressedChunkCache (200MB, ~1μs)            │
│    ↓ miss                                            │
│  L1 SegmentedLRUCache (100MB, ~1μs)                 │
│    ↓ miss                                            │
│  L2 OPFSStore (2GB, ~1ms)                           │
│    ↓ miss                                            │
│  L3 HTTP Fetch (~100ms)                              │
│    → Populate L1 + L2 + trigger prefetch             │
│                                                      │
│  ChunkPrefetcher: ±1 adjacent chunks per dimension   │
├──────────────────────────────────────────────────────┤
│                  Rendering Layer                      │
│                                                      │
│  GPUBufferPool: THREE.BufferGeometry reuse           │
│    Type-aware matching, capacity bucketing            │
└──────────────────────────────────────────────────────┘
```

---

## Bugs Fixed (5 fixes applied)

| # | Severity | Fix | File |
|---|----------|-----|------|
| 1 | **CRITICAL** | SegmentedLRU `get()` double-counted misses — now routes to correct segment | `segmented-lru-cache.ts` |
| 2 | **CRITICAL** | Prefetch cascade amplification — added `seen` set to prevent unbounded expansion | `chunk-prefetcher.ts` |
| 3 | **CRITICAL** | OPFS `data.buffer` wrote entire underlying ArrayBuffer — now writes `data` directly | `opfs-store.ts` |
| 4 | **HIGH** | LRU `delete()` used truthiness check — now uses `!== undefined` | `lru-cache.ts` |
| 5 | **HIGH** | LRU `set()` accepted oversized items — now rejects `size > maxSize` | `lru-cache.ts` |
| 6 | **HIGH** | SegmentedLRU negative `chunksSize` when `totalSize < 10MB` — now clamped to 0 | `segmented-lru-cache.ts` |

---

## Outstanding Issues (not fixed, need design decisions)

### CRITICAL

| Issue | Location | Description |
|-------|----------|-------------|
| L0 thundering herd | `cached-zarr-array.ts` | Concurrent `getChunk()` for same key triggers duplicate decompression. Needs in-flight promise map. |
| Cache not disposed on unload | `app.ts` cleanup() | Page unload doesn't flush OPFS metadata. Risk of metadata desync. |
| GPU pool recycling dead code | `gpu-buffer-pool.ts` | `release*Geometry()` never called from app code — pool only does same-node reuse. |

### HIGH

| Issue | Location | Description |
|-------|----------|-------------|
| No concurrent get() dedup | `two-level-caching-store.ts` | Duplicate HTTP fetches for same key on concurrent calls. |
| No init() guard | `two-level-caching-store.ts` | Double init() can create duplicate OPFSStore instances. |
| `?no-cache` still reads L1 | `two-level-caching-store.ts` | Previously-cached L1 entries served even with no-cache. |
| scene-loader missing pool dispose | `scene-loader.ts` | GPU buffer pool not disposed in dispose(). |
| GPU frameCount per-acquire | `gpu-buffer-pool.ts` | Frame counter inflated by N nodes/frame, eviction too aggressive. |
| OPFS metadata lost on page close | `opfs-store.ts` | 1s debounce + async save = lost on beforeunload. |
| L0 stale after hash invalidation | `two-level-caching-store.ts` | L2 cleared but L0 retains stale decompressed chunks. |

### MEDIUM

| Issue | Location | Description |
|-------|----------|-------------|
| OPFS eviction O(N) | `opfs-store.ts` | Linear scan for LRU entry; use min-heap for O(log N). |
| Bandwidth is lifetime average | `two-level-caching-store.ts` | Trends toward zero after load burst; use sliding window. |
| High-dim prefetch explosion | `chunk-prefetcher.ts` | 2*D prefetches per access; cap for >6D data. |
| Proxy whitelist fragile | `cached-zarr-array.ts` | Hardcoded zarrita property list; breaks on updates. |
| GPU pool O(N) lookup | `gpu-buffer-pool.ts` | Iterates all buckets; bucketing provides no perf benefit. |
| No GPU update count validation | `gpu-buffer-pool.ts` | count > capacity causes RangeError. |

---

## Review Files

| Review | File |
|--------|------|
| L0 Decompressed Chunk Cache | `code_reviews/l0-cache-review.md` |
| LRU + L1 Segmented Cache | `code_reviews/lru-l1-cache-review.md` |
| L2 OPFS Persistent Cache | `code_reviews/l2-opfs-cache-review.md` |
| Orchestration + Prefetcher | `code_reviews/orchestration-prefetcher-review.md` |
| GPU Buffer Pool | `code_reviews/gpu-buffer-pool-review.md` |
| Config & Integration | `code_reviews/cache-config-integration-review.md` |

---

## Overall Assessment

**Architecture**: Excellent. Clean four-level hierarchy with proper separation of concerns. Each layer has a clear responsibility and well-defined interface.

**Code Quality**: High. Well-documented with JSDoc, good error handling, consistent patterns. The Proxy-based L0 wrapper and bucket-sharded OPFS are particularly well-engineered.

**Testing**: Good unit coverage, solid E2E. Missing: concurrent access scenarios, lifecycle/dispose tests, cross-level integration tests.

**Key Risk Areas**:
1. **Data integrity** — The OPFS `data.buffer` bug (now fixed) could have caused silent data corruption
2. **Performance** — The prefetch cascade (now fixed) could have fetched entire datasets on cold cache
3. **Statistics accuracy** — The miss double-counting (now fixed) was reporting ~50% hit rate instead of ~90%
4. **Resource cleanup** — Multiple disposal gaps (cache unload, GPU pool) can leak memory over time
