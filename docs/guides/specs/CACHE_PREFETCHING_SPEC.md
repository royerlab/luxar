# Cache Prefetching Specification

**Version**: 1.2.0
**Last Updated**: 2026-07-13
**Status**: Implemented

> **Current architecture (2026-07).** The store class is
> `MultiLevelCachingStore` (`packages/luxar-viewer/src/cache/multi-level-caching-store.ts`).
> The L1 (memory LRU) / L2 (OPFS) / L3 (remote HTTP) tiers described below are
> the **compressed-chunk** path the prefetcher operates on; the full cache
> system has since grown additional tiers ABOVE it — an **L0 decompressed-chunk
> cache** (decoded TypedArrays, skips Blosc) and the **S-cache**
> (`SliceCache`, fully decoded per-slice geometry ladders) — which prefetching
> does not touch. See `packages/luxar-viewer/src/cache/README.md` for the live
> S-cache/L0/L1/L2 + HTTP picture and heap-aware budgets. The
> `ChunkPrefetcher` part of this spec matches the implementation
> (`packages/luxar-viewer/src/cache/chunk-prefetcher.ts`).

## Overview

This document specifies a prefetching system for the Luxar viewer's multi-level cache. The goal is to reduce perceived latency by proactively fetching adjacent zarr chunks before they are explicitly requested.

### Motivation

When viewing zarr data, access patterns exhibit strong spatial and temporal locality:
- **Spatial locality**: Viewing a 3D region means adjacent chunks are likely needed next
- **Temporal locality**: nD slicing often iterates sequentially through time/channel dimensions
- **Network latency**: L3 (HTTP) fetches are ~100ms, making latency hiding valuable

By prefetching adjacent chunks after cache hits, we can hide network latency and improve perceived performance.

---

## Architecture

### Design Decision: Separate PrefetchManager

The prefetching logic is implemented as a **separate `ChunkPrefetcher` class** rather than inline in `MultiLevelCachingStore`. This provides:

1. **Separation of concerns** - Store remains a simple zarr backend
2. **Testability** - Prefetch logic can be unit tested in isolation
3. **Configurability** - Prefetching can be disabled/tuned independently
4. **Extensibility** - Smarter heuristics can be added without touching the store

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     MultiLevelCachingStore                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │ L1 Cache │───►│ L2 Cache │───►│ L3 HTTP  │                  │
│  │ (Memory) │    │ (OPFS)   │    │ (Remote) │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│        │               │               │                        │
│        └───────────────┴───────────────┘                        │
│                        │                                        │
│                   onAccess(key, hitLevel)                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ChunkPrefetcher                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Parse Chunk  │───►│ Calculate    │───►│ Enqueue      │      │
│  │ Indices      │    │ Adjacent     │    │ Prefetch     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                │                │
│                                                ▼                │
│                                    ┌──────────────────┐        │
│                                    │ Process Queue    │        │
│                                    │ (max 4 concurrent)│        │
│                                    └──────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Trigger Points

Prefetching is triggered at the **L2/L3 boundary**, not on L1 hits:

| Event | Action | Rationale |
|-------|--------|-----------|
| L1 hit | No prefetch | Data already fast (~1μs), not worth overhead |
| L2 hit | Prefetch adjacent from L3 → L2 + L1 | Hide future L3 latency |
| L3 fetch | Prefetch adjacent from L3 → L2 + L1 | Already paying network cost, batch it |

### Why Not Prefetch on L1 Hit?

L1 hits are extremely fast (~1μs). The overhead of calculating adjacent chunks and checking if they need prefetching would exceed any potential benefit. Additionally, L1 hits indicate hot data that's likely already been prefetched.

---

## Adjacent Chunk Calculation

### Chunk Key Formats

Zarr chunks use two naming conventions:

```
Zarr v2 (dot notation):  points/positions/0.1.2
Zarr v3 (path notation): points/positions/c/0/1/2
```

Both encode the same information: array path + chunk indices.

### Adjacency Definition

For a chunk at indices `[i, j, k, ...]`, adjacent chunks are defined as **symmetric ±1 in each dimension**:

```
Current:  [2, 3, 1]
Adjacent: [1, 3, 1], [3, 3, 1],  // ±1 in dim 0
          [2, 2, 1], [2, 4, 1],  // ±1 in dim 1
          [2, 3, 0], [2, 3, 2]   // ±1 in dim 2
```

**Design Decision**: Symmetric (±1) rather than forward-only (+1) because:
1. Most arrays are 1D anyway (positions, colors, radii)
2. Chunking reduces effective dimensionality
3. User navigation can go in either direction
4. Symmetric is simpler to implement and reason about

### Boundary Handling

- Negative indices are skipped (no chunk at [-1, 0, 0])
- Upper bounds are NOT checked (let HTTP 404 handle non-existent chunks)
- This avoids needing to know array shape at prefetch time

### Chunk Index Parsing

```typescript
/**
 * Parse chunk indices from a zarr chunk key.
 *
 * @example
 * parseChunkIndices('points/positions/0.1.2') → [0, 1, 2]
 * parseChunkIndices('points/positions/c/0/1/2') → [0, 1, 2]
 * parseChunkIndices('.zattrs') → null (not a chunk)
 */
function parseChunkIndices(key: string): number[] | null {
  // Check for v2 dot notation: ends with digits separated by dots
  const v2Match = key.match(/\/(\d+(?:\.\d+)*)$/);
  if (v2Match) {
    return v2Match[1].split('.').map(Number);
  }

  // Check for v3 path notation: /c/ followed by path segments
  const v3Match = key.match(/\/c\/(\d+(?:\/\d+)*)$/);
  if (v3Match) {
    return v3Match[1].split('/').map(Number);
  }

  return null; // Not a chunk key (metadata file)
}
```

### Adjacent Key Generation

```typescript
/**
 * Generate adjacent chunk keys (±1 in each dimension).
 *
 * @example
 * getAdjacentChunks('points/positions/1.2.3')
 * → ['points/positions/0.2.3', 'points/positions/2.2.3',
 *    'points/positions/1.1.3', 'points/positions/1.3.3',
 *    'points/positions/1.2.2', 'points/positions/1.2.4']
 */
function getAdjacentChunks(key: string): string[] {
  const indices = parseChunkIndices(key);
  if (!indices) return [];

  // Extract base path (everything before the indices)
  const isV3 = key.includes('/c/');
  const basePath = isV3
    ? key.replace(/\/c\/[\d/]+$/, '')
    : key.replace(/\/[\d.]+$/, '');

  const adjacent: string[] = [];

  for (let dim = 0; dim < indices.length; dim++) {
    for (const delta of [-1, 1]) {
      const newIndices = [...indices];
      newIndices[dim] += delta;

      // Skip negative indices
      if (newIndices[dim] < 0) continue;

      // Generate key in same format as input
      const indexStr = isV3
        ? 'c/' + newIndices.join('/')
        : newIndices.join('.');

      adjacent.push(`${basePath}/${indexStr}`);
    }
  }

  return adjacent;
}
```

---

## Priority Mechanism

### The Challenge

JavaScript is single-threaded - there's no OS-level thread priority. "Lower priority" must be achieved through **code behavior**, not scheduler hints.

### How Priority Works

| Aspect | Normal Request | Prefetch Request |
|--------|---------------|------------------|
| Blocking | `await fetch()` - caller waits | Fire-and-forget - no await |
| Concurrency | Unlimited (browser limit) | Limited to 4 concurrent |
| Error handling | Propagate to caller | Silently ignore |
| On dispose | Must complete | Complete harmlessly in background |
| User impact | Blocks UI/data display | Invisible to user |

### JavaScript Async Model

JavaScript achieves "parallel" I/O through **non-blocking operations**, not threads:

```
┌─────────────────────────────────────────────────────────────────┐
│                       Browser Process                           │
│                                                                 │
│  ┌─────────────────┐         ┌─────────────────────────────┐   │
│  │   JS Thread     │         │   Browser Internals          │   │
│  │   (single)      │         │   (C++, truly multi-threaded)│   │
│  │                 │         │                               │   │
│  │  fetch(url) ────┼────────►│  - Network stack             │   │
│  │  // returns     │         │  - DNS resolution            │   │
│  │  // immediately │         │  - TLS handshake             │   │
│  │                 │         │  - HTTP request/response     │   │
│  │  // JS is free  │         │                               │   │
│  │  // to do other │◄────────┼── Queues callback when done  │   │
│  │  // work        │         │                               │   │
│  └─────────────────┘         └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight**: `fetch()` returns immediately. The browser's network stack (which IS multi-threaded) handles the actual HTTP connection. When the response arrives, a callback is queued for the JS event loop.

```typescript
// These run "in parallel" (all requests in flight simultaneously)
// but no JS code executes in parallel
const p1 = fetch(url1);  // Starts request, returns immediately
const p2 = fetch(url2);  // Starts request, returns immediately
const p3 = fetch(url3);  // Starts request, returns immediately

// JS thread is FREE here - all 3 requests are in flight
// Browser handles actual network I/O

// Later, collect results
const [r1, r2, r3] = await Promise.all([p1, p2, p3]);
```

### Concurrency Limiting

Normal requests go directly to `store.get()` with no limit. Prefetch requests are queued and limited:

```typescript
class ChunkPrefetcher {
  private store: MultiLevelCachingStore;
  private maxConcurrent = 4;
  private inFlight = new Set<string>();  // Track in-flight keys
  private queue = new Set<string>();
  private processing = false;

  /**
   * Called by store after L2 hit or L3 fetch.
   */
  onAccess(key: string): void {
    const adjacent = this.getAdjacentChunks(key);
    for (const adjKey of adjacent) {
      // Full deduplication: not in queue, not in-flight
      if (!this.queue.has(adjKey) && !this.inFlight.has(adjKey)) {
        this.queue.add(adjKey);
      }
    }
    this.processQueue();  // Fire-and-forget (not awaited)
  }

  private async processQueue(): Promise<void> {
    // Prevent concurrent processing
    if (this.processing) return;
    this.processing = true;

    try {
      while (this.queue.size > 0 && this.inFlight.size < this.maxConcurrent) {
        const key = this.queue.values().next().value;
        this.queue.delete(key);
        this.inFlight.add(key);

        // Fire-and-forget with cleanup
        this.store.get(key)
          .catch(() => {})  // Ignore errors (404s, network failures)
          .finally(() => {
            this.inFlight.delete(key);
            // Trigger queue processing when slot frees up
            if (this.queue.size > 0) {
              queueMicrotask(() => this.processQueue());
            }
          });
      }
    } finally {
      this.processing = false;
    }
  }
}
```

This ensures:
- At most 4 prefetch requests in flight
- Full deduplication (no duplicate prefetches in queue or in-flight)
- Race-condition safe (re-triggers processing when slots free up)
- Normal requests face no artificial limit
- Browser's connection pool (~6 for HTTP/1.1, ~100 streams for HTTP/2) has capacity for normal requests

### Fetch Priority API (Optional Enhancement)

Modern browsers support the Fetch Priority API as a hint:

```typescript
// Normal request - default priority
fetch(url);

// Prefetch request - low priority hint
fetch(url, { priority: 'low' });
```

Browser support:
- Chrome/Edge 101+: Supported
- Firefox: Not supported (ignored)
- Safari: Not supported (ignored)

This is a **hint**, not a guarantee. Our concurrency limiting is the primary mechanism.

---

## Integration

### Store Modifications Required

The current `MultiLevelCachingStore` needs these additions to support prefetching:

```typescript
// Add to MultiLevelCachingStore class:

private prefetcher: ChunkPrefetcher | null = null;

/**
 * Attach a prefetcher to receive access notifications.
 */
setPrefetcher(prefetcher: ChunkPrefetcher | null): void {
  this.prefetcher = prefetcher;
}

// Modify get() method - add prefetch triggers after L2 hit and L3 fetch:
async get(key: string, _options?: any): Promise<Uint8Array | undefined> {
  // L1: Memory check (fastest, ~1μs)
  const l1Hit = this.l1Cache.get(key);
  if (l1Hit) {
    // No prefetch on L1 hit - data is already fast
    return l1Hit;
  }

  // L2: OPFS check (~1ms)
  if (this.enabled && this.l2Store) {
    const l2Hit = await this.l2Store.get(key);
    if (l2Hit) {
      this.l1Cache.set(key, l2Hit);
      // NEW: Trigger prefetch on L2 hit
      this.prefetcher?.onAccess(key);
      return l2Hit;
    }
  }

  // L3: Remote fetch (~100ms)
  try {
    const response = await fetch(`${this.baseUrl}/${key}`);
    if (!response.ok) return undefined;

    const data = new Uint8Array(await response.arrayBuffer());
    this.l1Cache.set(key, data);
    if (this.enabled && this.l2Store) {
      this.l2Store.set(key, data).catch(() => {});
    }

    // NEW: Trigger prefetch on L3 fetch
    this.prefetcher?.onAccess(key);

    return data;
  } catch {
    return undefined;
  }
}

// Modify dispose() to clear prefetcher reference:
async dispose(): Promise<void> {
  // Clear prefetcher reference (in-flight requests will complete harmlessly)
  this.prefetcher = null;

  if (this.l2Store) {
    await this.l2Store.dispose();
  }
  this.clearL1();
}
```

### Lifecycle Management

The prefetcher lifecycle is tied to the `MultiLevelCachingStore`:

```typescript
// Creation (in scene-loader.ts or similar)
const store = new MultiLevelCachingStore(baseUrl, options);
await store.init();

// Create prefetcher with reference to store
const prefetcher = new ChunkPrefetcher(store, {
  maxConcurrent: 4,
  enabled: true,
});

// Connect bidirectionally
store.setPrefetcher(prefetcher);

// Usage - normal, prefetching happens automatically
const chunk = await store.get('points/positions/0.0.0');

// Cleanup - dispose store (which cancels prefetches)
await store.dispose();
```

---

## Configuration

### Options

```typescript
interface ChunkPrefetcherOptions {
  /** Maximum concurrent prefetch requests (default: 4) */
  maxConcurrent?: number;

  /** Enable/disable prefetching (default: true) */
  enabled?: boolean;

  /** Use Fetch Priority API hint (default: true) */
  useFetchPriority?: boolean;
}
```

### URL Parameters

For debugging and testing:

| Parameter | Effect |
|-----------|--------|
| `?no-prefetch` | Disable prefetching entirely |
| `?prefetch-debug` | Log prefetch operations to console |

---

## Performance Considerations

### Memory Impact

Prefetched data competes for L1 cache space. However:
- LRU eviction naturally handles unused prefetches
- Prefetched-but-unused data gets evicted first (hasn't been accessed recently)
- L1 size (100MB default) is large enough to absorb reasonable prefetch

### Network Impact

With `maxConcurrent = 4`:
- HTTP/1.1 (6 connections): 2 slots reserved for normal requests
- HTTP/2 (~100 streams): Plenty of capacity

### When Prefetching Hurts

Prefetching can hurt performance when:
1. **Random access patterns** - Prefetched data is never used
2. **Bandwidth-constrained** - Prefetch competes with needed data
3. **High-latency connections** - Queue fills up, stale prefetches

The `?no-prefetch` parameter allows disabling for these cases.

---

## Testing Strategy

### Unit Tests

1. **Chunk index parsing**: Test both v2 and v3 formats
2. **Adjacent calculation**: Test various dimensions, boundary cases
3. **Queue management**: Test concurrency limiting, deduplication

### Integration Tests

1. **L2 hit triggers prefetch**: Mock store, verify prefetcher called
2. **L3 fetch triggers prefetch**: Mock store, verify prefetcher called
3. **L1 hit skips prefetch**: Verify prefetcher NOT called

### E2E Tests (Playwright)

1. **Prefetch reduces latency**: Measure time to load adjacent chunks
2. **No resource leaks**: Verify in-flight requests complete without errors
3. **Debug parameter works**: Verify `?no-prefetch` disables feature

---

## Future Enhancements

### Directional Prediction

Track recent access patterns to predict prefetch direction:

```typescript
// If recent accesses are [0,0,0] → [0,0,1] → [0,0,2]
// Predict next: [0,0,3], prefetch more aggressively in dim 2
```

### Adaptive Concurrency

Adjust `maxConcurrent` based on network conditions:

```typescript
// If latency is high, reduce prefetch concurrency
// If bandwidth is plentiful, increase it
```

### Prefetch Buffer

Separate prefetched data from L1 to prevent evicting hot data:

```typescript
// Prefetched data → prefetchBuffer (smaller, separate)
// On access → promote to L1
// Prevents prefetch from evicting actively-used chunks
```

---

## Changelog

- **v1.2.0** (2026-07-13): Sync with current cache architecture
  - Class renamed to match the implementation: `TwoLevelCachingStore` →
    `MultiLevelCachingStore`
  - Added "Current architecture (2026-07)" note: L0 decompressed cache and
    S-cache (SliceCache) sit above the L1/L2/L3 compressed-chunk path this
    spec covers; see `packages/luxar-viewer/src/cache/README.md`

- **v1.1.0** (2025-01-06): Critical review and simplification
  - Fixed race condition in `processQueue()` (re-trigger via queueMicrotask when slots free up)
  - Added full deduplication (check both queue and in-flight before adding)
  - **Removed cancellation entirely** - in-flight requests complete harmlessly on dispose
  - Simplified `inFlight` from `Map<string, AbortController>` to `Set<string>`
  - Clarified store modifications required for integration
  - Updated priority comparison table (removed cancellation row)

- **v1.0.0** (2025-01-06): Initial specification
  - Separate ChunkPrefetcher architecture
  - L2/L3 trigger points
  - Symmetric ±1 adjacency
  - Concurrency-based priority mechanism
  - Full async model documentation
