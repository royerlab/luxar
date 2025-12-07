# Luxar Viewer Cache Package

Two-level caching system with intelligent prefetching for zarr chunks enabling offline viewing, instant reloads, and reduced bandwidth.

## Overview

This package implements a transparent caching and prefetching layer for zarr datasets:

- **L1 (Memory)**: 100MB segmented LRU cache with metadata protection
- **L2 (OPFS)**: 2GB persistent storage surviving browser restarts
- **Intelligent Prefetching**: Proactive loading of adjacent chunks to hide network latency
- **Content-hash validation**: Automatic cache invalidation when data changes
- **Zero overhead**: Stores raw compressed chunks (no re-compression)

## Quick Start

The cache is **enabled by default** and works transparently through zarrita:

```typescript
import { TwoLevelCachingStore } from '../cache';
import * as zarr from 'zarrita';

// Create caching store
const store = new TwoLevelCachingStore('https://example.com/dataset.zarr', {
  l1MaxSize: 100 * 1024 * 1024, // 100MB (optional)
  l2MaxSize: 2 * 1024 * 1024 * 1024, // 2GB (optional)
  debug: false, // Set true for verbose logging (optional)
});

await store.init();

// Use with zarrita - completely transparent!
const zarrStore = await zarr.tryWithConsolidated(store);
const root = await zarr.open(zarrStore);

// First load: HTTP → L2 → L1 → zarrita
// Second load: L1 → zarrita (~1μs!)
// After reload: L2 → L1 → zarrita (~1ms)

// Clean up when done
await store.dispose();
```

## URL Parameters

Override cache behavior via URL parameters:

- `?no-cache` - Disable all caching for this session
- `?cache-debug` - Enable verbose cache logging
- `?clear-cache` - Clear OPFS cache before loading dataset
- `?no-prefetch` - Disable prefetching (cache still active)
- `?prefetch-debug` - Enable verbose prefetch logging

Example:

```
http://localhost:5173/?src=http://example.com/data.zarr&cache-debug&prefetch-debug
```

## Intelligent Prefetching

The cache includes an **intelligent prefetching system** that proactively loads adjacent chunks to hide network latency.

### How It Works

When a chunk is accessed from L2 (OPFS) or L3 (HTTP):

1. Parse chunk indices from the key (e.g., `positions/1.2.3` → `[1, 2, 3]`)
2. Calculate adjacent chunks (±1 in each dimension)
3. Queue them for background prefetching
4. Limit to 4 concurrent prefetches (leaves bandwidth for normal requests)

### Example

```
User requests: positions/1.2.3
             ↓
Cache prefetches (in background):
  positions/0.2.3  (dim 0: -1)
  positions/2.2.3  (dim 0: +1)
  positions/1.1.3  (dim 1: -1)
  positions/1.3.3  (dim 1: +1)
  positions/1.2.2  (dim 2: -1)
  positions/1.2.4  (dim 2: +1)
```

### Benefits

- **Reduced Latency**: Adjacent chunks pre-loaded before explicit request
- **Transparent**: No API changes required, works automatically
- **Bandwidth-Friendly**: Limited concurrency prevents congestion
- **Non-Blocking**: Fire-and-forget pattern, never blocks normal requests

### Configuration

Prefetching is **enabled by default** and configurable via ChunkPrefetcher options:

```typescript
import { ChunkPrefetcher } from '../cache';

const prefetcher = new ChunkPrefetcher(store, {
  maxConcurrent: 4, // Max concurrent prefetch requests
  enabled: true, // Enable/disable prefetching
  debug: false, // Enable debug logging
});

store.setPrefetcher(prefetcher);
```

For full details, see [`docs/CACHE_PREFETCHING_SPEC.md`](../../../../docs/CACHE_PREFETCHING_SPEC.md).

## API Reference

### TwoLevelCachingStore

Main class implementing zarrita's `AsyncReadable` interface.

#### Constructor

```typescript
new TwoLevelCachingStore(baseUrl: string, options?: {
  l1MaxSize?: number;      // L1 size in bytes (default: 100MB)
  l2MaxSize?: number;      // L2 size in bytes (default: 2GB)
  debug?: boolean;         // Enable debug logging (default: false)
  urlParams?: URLSearchParams;  // For testing (optional)
})
```

#### Methods

**`async init(): Promise<void>`**

Initialize OPFS storage and validate cache. Must be called before first use.

**`async get(key: string): Promise<Uint8Array | undefined>`**

Get a zarr chunk with L1 → L2 → HTTP cascade. Implements zarrita's AsyncReadable interface.

**`getStats(): { l1, l2 }`**

Get cache statistics:

```typescript
{
  l1: {
    metadataSize: number,    // Bytes in metadata segment
    chunksSize: number,      // Bytes in chunks segment
    metadataCount: number,   // Files in metadata segment
    chunksCount: number      // Files in chunks segment
  },
  l2: {
    size: number,            // Total bytes in OPFS
    count: number            // Total files in OPFS
  }
}
```

**`clearL1(): void`**

Clear L1 memory cache only (L2 persists).

**`async clearL2(): Promise<void>`**

Clear L2 OPFS cache only (L1 untouched).

**`async clearAll(): Promise<void>`**

Clear both L1 and L2 caches.

**`setPrefetcher(prefetcher: ChunkPrefetcher | null): void`**

Attach a prefetcher to enable intelligent adjacent chunk prefetching. Pass `null` to disable.

```typescript
const prefetcher = new ChunkPrefetcher(store, { maxConcurrent: 4 });
store.setPrefetcher(prefetcher); // Enable prefetching
```

**`async dispose(): Promise<void>`**

Flush pending metadata writes, clear prefetcher reference, and clear L1. Call when navigating away.

**`async listDatasets(): Promise<Array<{...}>>`**

List all cached datasets in OPFS:

```typescript
[
  {
    url: "https://example.com/dataset.zarr",
    hash: "8649296f56b76790",
    size: 52428800,  // bytes
    count: 1234      // number of cached chunks
  },
  ...
]
```

### ChunkPrefetcher

Intelligent prefetcher for proactive adjacent chunk loading.

#### Constructor

```typescript
new ChunkPrefetcher(store: TwoLevelCachingStore, options?: {
  maxConcurrent?: number;       // Max concurrent prefetches (default: 4)
  enabled?: boolean;            // Enable/disable (default: true)
  useFetchPriority?: boolean;   // Use Fetch Priority API (default: true, not yet implemented)
  debug?: boolean;              // Enable debug logging (default: false)
  urlParams?: URLSearchParams;  // For testing (optional)
})
```

#### Methods

**`onAccess(key: string): void`**

Called by store when a chunk is accessed from L2 or L3. Enqueues adjacent chunks for prefetching. Called automatically - not for direct use.

**`getStats(): { queued, inFlight, enabled }`**

Get prefetch queue statistics:

```typescript
{
  queued: number,    // Chunks waiting to be prefetched
  inFlight: number,  // Chunks currently being prefetched
  enabled: boolean   // Whether prefetching is enabled
}
```

### Modules

**TwoLevelCachingStore** - Main orchestrator implementing AsyncReadable interface

**ChunkPrefetcher** - Intelligent adjacent chunk prefetcher (enabled by default)

**LRUCache** - Generic LRU cache with O(1) operations

**SegmentedLRUCache** - 20/80 metadata/chunks split with name-based routing

**OPFSStore** - OPFS persistence layer with LRU eviction

## Cache Behavior

### First Visit

```
User loads dataset
└─→ HTTP fetches all chunks (~100ms each)
    └─→ Stores in L2 (OPFS, persists across sessions)
    └─→ Stores in L1 (memory, session only)
    └─→ Renders
```

### Same Session

```
User navigates/zooms
└─→ L1 hit (~1μs) - instant!
    └─→ Renders
```

### Browser Restart

```
User returns later
└─→ L1 miss (cleared on page load)
    └─→ L2 hit (~1ms) - fast!
        └─→ Promotes to L1
        └─→ Renders
```

### Dataset Updated

```
Dataset content changed (new content_hash)
└─→ Cache validation detects mismatch
    └─→ Clears L2 automatically
    └─→ Fetches fresh data
```

## Content Hash System

All Luxar-generated datasets include hierarchical content hashes:

**Python side** (automatic):

```python
# In LuxarZarrCompiler.finalize():
# - Computes xxhash64 for each node
# - Children hashed before parents (post-order)
# - Stored in .zattrs['content_hash']
```

**TypeScript side** (automatic):

```typescript
// On TwoLevelCachingStore.init():
// - Reads root .zattrs['content_hash']
// - Compares with cached hash
// - Clears cache if mismatch
```

**External datasets** (non-Luxar):

- If no `content_hash` attribute → validation skipped
- Cache remains functional but won't auto-invalidate

## Performance

### Expected Improvements

| Metric                   | HTTP Only        | With Cache                  |
| ------------------------ | ---------------- | --------------------------- |
| First load               | 100ms/chunk      | 100ms/chunk (same)          |
| Reload (same session)    | 100ms/chunk      | 1μs/chunk (100,000x faster) |
| Reload (new session)     | 100ms/chunk      | 1ms/chunk (100x faster)     |
| Offline viewing          | Impossible       | Fully functional            |
| Bandwidth (10MB dataset) | 10MB every visit | 10MB once                   |

### Memory Budget

**100MB L1** compressed ≈ **300MB-1GB** decoded coverage (depends on compression ratio)

**Typical session** (100K points, 4D):

- RangeCache: ~150MB (decoded arrays)
- L1 Cache: ~100MB (compressed chunks)
- Three.js: ~50MB (geometries)
- **Total**: ~300MB (well within browser limits)

## Browser Support

**OPFS Available**:

- ✅ Chrome 86+ (October 2020)
- ✅ Firefox 111+ (March 2023)
- ✅ Safari 15.2+ (December 2021)

**Graceful Degradation**:

- Older browsers → L1-only mode (still faster than no cache)
- OPFS quota exceeded → Skip L2, use L1 + HTTP
- Network offline → Use cached data if available

## Debug API

When `?debug` URL parameter is present, cache management available at:

```typescript
await window.__luxarDebug.cache.getStats(); // Get L1 + L2 statistics
await window.__luxarDebug.cache.listDatasets(); // List all cached datasets
await window.__luxarDebug.cache.clearL1(); // Clear L1 memory cache only
await window.__luxarDebug.cache.clearL2(); // Clear L2 OPFS cache only
await window.__luxarDebug.cache.clearAll(); // Clear both L1 and L2
```

Example usage:

```typescript
// Check cache usage
const stats = await window.__luxarDebug.cache.getStats();
console.log('L1 Memory:', stats.l1.metadataCount, 'metadata,', stats.l1.chunksCount, 'chunks');
console.log('L2 OPFS:', stats.l2.size, 'bytes,', stats.l2.count, 'files');

// List all cached datasets
const datasets = await window.__luxarDebug.cache.listDatasets();
console.log('Cached datasets:', datasets);

// Clear caches during development
await window.__luxarDebug.cache.clearAll();
```

## Implementation Details

See [SPECIFICATIONS.md](./SPECIFICATIONS.md) for detailed algorithms and design rationale.

## Related Packages

- **RangeCache** (`../data/range-cache.ts`) - Caches decoded Float32Array data
- **Scene Loader** (`../data/scene-loader.ts`) - Uses TwoLevelCachingStore transparently
- **Configuration** (`../config/`) - Cache configuration options
