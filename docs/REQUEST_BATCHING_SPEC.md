# Request Batching Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-04
**Status**: Proposed

## Problem Statement

When loading zarr datasets with a large number of chunks (30K+), browsers fail with `ERR_INSUFFICIENT_RESOURCES`. This occurs because:

1. Each zarr chunk is stored as a separate file requiring an individual HTTP request
2. Browsers limit concurrent connections to ~6 per domain (HTTP/1.1 standard)
3. The browser's internal request queue has resource limits (~64K pending requests)
4. When the viewer queries visible data, it can trigger thousands of fetch() calls
5. These pending fetch() calls exhaust browser memory/resources before completing

### Concrete Example

The `demo_4d_fractals.py` demo generates:
- 57 million points
- 5D data (x, y, z, w, fractal_id)
- Chunk size: ~1,820 points (based on 64KB target / 36 bytes per point)
- Total chunks: **31,383**

Loading this dataset triggers 31,383+ HTTP requests, causing immediate browser failure.

## Root Cause Analysis

### Chunk Size Calculation

```python
# In compiler.py
TARGET_CHUNK_BYTES = 65_536  # 64KB
bytes_per_point = n_dims * 4 + 16  # Conservative estimate with overhead
chunk_size = max(1024, TARGET_CHUNK_BYTES // bytes_per_point)

# For 5D data:
# bytes_per_point = 5 * 4 + 16 = 36 bytes
# chunk_size = 65536 // 36 = 1,820 points per chunk
```

### Request Flow (Current)

```
User loads dataset
    ↓
SceneLoader opens zarr store (1 request for .zmetadata)
    ↓
PointSpatialIndexLoader loads chunk_bounds (1 request)
    ↓
queryVisibleRanges() returns N ranges to load
    ↓
loadRanges() iterates sequentially:
    for each range:
        await get(array, slice)  ← HTTP request
    ↓
Browser queues 31K+ requests
    ↓
ERR_INSUFFICIENT_RESOURCES
```

## Proposed Solution

### Architecture Overview

Introduce a `RequestQueue` that sits between application code and the browser's fetch API:

```
┌─────────────────────────────────────┐
│   Application Code                   │
│   (PointSpatialIndexLoader)          │
└──────────────┬──────────────────────┘
               │ Unlimited requests accepted
               ▼
┌─────────────────────────────────────┐
│   RequestQueue                       │
│   - JavaScript-side queue            │
│   - Limits concurrent to MAX (50)    │
│   - FIFO processing                  │
│   - Graceful backpressure            │
└──────────────┬──────────────────────┘
               │ Throttled (max 50 concurrent)
               ▼
┌─────────────────────────────────────┐
│   Browser Fetch API                  │
│   (~6 concurrent TCP connections)    │
└─────────────────────────────────────┘
```

### Key Insight

**The browser cannot queue 31K requests, but JavaScript can.**

By intercepting fetch operations before they reach the browser's native queue, we can:
1. Accept unlimited requests into our JavaScript queue
2. Release them to the browser at a controlled rate
3. Prevent resource exhaustion while ensuring all requests complete

## Detailed Specification

### Component 1: RequestQueue Class

**Location**: `packages/luxar-viewer/src/data/request-queue.ts`

**Purpose**: Limit concurrent HTTP requests to prevent browser resource exhaustion

**Interface**:

```typescript
interface RequestQueueOptions {
  maxConcurrent?: number;  // Default: 50
}

class RequestQueue {
  constructor(options?: RequestQueueOptions);

  /**
   * Execute an async operation with concurrency limiting.
   * If at capacity, waits until a slot is available.
   *
   * @param fn - Async function to execute (typically contains fetch/zarr get)
   * @returns Promise resolving to fn's return value
   */
  get<T>(fn: () => Promise<T>): Promise<T>;

  /**
   * Wrapper for fetch() with concurrency limiting.
   *
   * @param url - URL to fetch
   * @param options - Standard fetch options
   * @returns Promise<Response>
   */
  fetch(url: string, options?: RequestInit): Promise<Response>;

  /**
   * Current number of active requests.
   */
  readonly activeCount: number;

  /**
   * Current number of queued (waiting) requests.
   */
  readonly pendingCount: number;
}
```

**Algorithm**:

```
function get(fn):
    if activeCount >= maxConcurrent:
        add resolver to pending queue
        await promise (blocked until released)

    activeCount++
    try:
        return await fn()
    finally:
        activeCount--
        if pending queue not empty:
            release next waiting request
```

**Constants**:

| Constant | Value | Rationale |
|----------|-------|-----------|
| `MAX_CONCURRENT_REQUESTS` | 50 | Safe for all browsers, balances throughput vs. resource usage |
| `MIN_CONCURRENT_REQUESTS` | 6 | Matches browser connection limit |
| `DEFAULT_CONCURRENT_REQUESTS` | 50 | Optimized for HTTP/2 multiplexing |

### Component 2: Integration Points

**File**: `packages/luxar-viewer/src/data/point-spatial-index-loader.ts`

**Current Pattern** (problematic):

```typescript
// Line ~716 - Sequential fetches, each await triggers HTTP request
for (const range of ranges) {
  const sliceSpec = [slice(range.start, range.end), slice(null)];
  const chunkData = await get(array, sliceSpec);  // Immediate fetch
  output.set(chunkData.data, destOffset);
}
```

**Proposed Pattern**:

```typescript
import { globalRequestQueue } from './request-queue';

// Wrap zarr get() with queue - waits if at capacity
for (const range of ranges) {
  const sliceSpec = [slice(range.start, range.end), slice(null)];
  const chunkData = await globalRequestQueue.get(() =>
    get(array, sliceSpec)
  );
  output.set(chunkData.data, destOffset);
}
```

### Component 3: Parallel Array Loading (Optimization)

**Current Pattern** (sequential):

```typescript
// Each array waits for previous to complete
const positions = await this.loadRanges('positions', ranges);
const colors = await this.loadRanges('colors', ranges);
const radii = await this.loadRanges('radii', ranges);
const sharpness = await this.loadRanges('sharpness', ranges);
```

**Proposed Pattern** (parallel):

```typescript
// All arrays load concurrently, queue handles throttling
const [positions, colors, radii, sharpness] = await Promise.all([
  this.loadRanges('positions', ranges),
  this.arrays.colors ? this.loadRanges('colors', ranges) : null,
  this.arrays.radii ? this.loadRanges('radii', ranges) : null,
  this.arrays.sharpness ? this.loadRanges('sharpness', ranges) : null,
]);
```

**Impact**: 4x potential speedup (4 sequential → 1 parallel batch)

### Component 4: Python-Side Warning

**Location**: `packages/luxar/src/luxar/io/compiler.py`

**Purpose**: Warn users at dataset generation time when chunk count may cause browser issues

**Constant**:

```python
# In typing_utils/constants.py
MAX_RECOMMENDED_CHUNKS: Final[int] = 10_000
```

**Implementation**:

```python
# In _finalize_spatial_index() after computing chunk count
if n_chunks > MAX_RECOMMENDED_CHUNKS:
    aprint(f"Warning: {n_chunks:,} chunks may cause browser resource issues")
    aprint(f"  Consider: increasing chunk size, reducing dimensions, or ")
    aprint(f"  using a smaller dataset. Browser limit is ~10K chunks.")
```

## Performance Characteristics

### Expected Results

| Metric | Before | After |
|--------|--------|-------|
| Max pending browser requests | 31,000+ | 50 |
| Browser errors | `ERR_INSUFFICIENT_RESOURCES` | None |
| Memory for promises | ~31K Promise objects | ~50 Promise objects |
| Load time (31K chunks) | Fails | ~30-60 seconds |
| Load time (3K chunks) | ~5 seconds | ~5 seconds (no change) |

### Throughput Analysis

With 50 concurrent requests and average latency of 50ms:
- Theoretical throughput: 1,000 requests/second
- 31K chunks: ~31 seconds minimum
- Actual (with HTTP/2): faster due to multiplexing

### Memory Impact

- Before: 31K pending Promise objects + fetch state
- After: 50 active + N queued (lightweight resolver functions)
- Reduction: ~600x less memory pressure

## Trade-offs

### Approach Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Request batching** (proposed) | Works for any chunk count, no Python changes | Slower total load for huge datasets |
| **Larger chunk size** | Fewer requests | Worse spatial query granularity |
| **HTTP/2 only** | 100+ streams | Still hits fetch() API limits |
| **WebSocket streaming** | Efficient | Major architecture change |

### Why Request Batching?

1. **Universal**: Works regardless of dataset size or dimension count
2. **Non-breaking**: No changes to Python encoding or zarr format
3. **Proven pattern**: Standard solution for rate limiting
4. **Graceful degradation**: Large datasets load slowly but correctly

## Configuration

### Recommended Defaults

```typescript
// src/config/data-loading.ts
export const DATA_LOADING_CONFIG = {
  /** Maximum concurrent HTTP requests for zarr chunks */
  maxConcurrentRequests: 50,

  /**
   * Warn in console if dataset has more chunks than this.
   * Note: Python compiler has its own warning at 10K.
   */
  chunkCountWarningThreshold: 10_000,
};
```

### Tuning Guidelines

| Scenario | Recommended `maxConcurrent` |
|----------|----------------------------|
| Standard browser | 50 |
| Mobile/low-memory | 20 |
| HTTP/2 server | 100 |
| Local development | 100 |

## Testing Strategy

### Unit Tests

1. **Concurrency enforcement**: Verify never exceeds `maxConcurrent`
2. **FIFO ordering**: Verify requests complete in submission order
3. **Completion guarantee**: All queued requests eventually complete
4. **Error propagation**: Errors in `fn()` propagate correctly

### Integration Tests

1. **Large dataset loading**: 30K+ chunks loads without browser errors
2. **Performance regression**: Small datasets not slowed down
3. **Memory stability**: No memory leaks during extended loading

### E2E Tests

1. Regenerate `demo_4d_fractals.py` with `grid_size=100` (31K chunks)
2. Load in browser via Playwright
3. Assert no console errors
4. Assert points eventually render

## Implementation Checklist

- [ ] Create `packages/luxar-viewer/src/data/request-queue.ts`
- [ ] Add `globalRequestQueue` singleton export
- [ ] Modify `point-spatial-index-loader.ts` to use queue
- [ ] Add parallel array loading with `Promise.all()`
- [ ] Add `MAX_RECOMMENDED_CHUNKS` constant to Python
- [ ] Add warning in `compiler.py` when chunk count exceeds threshold
- [ ] Add unit tests for RequestQueue
- [ ] Add E2E test with large dataset
- [ ] Update data loading documentation

## Future Considerations

### Potential Enhancements

1. **Priority queue**: Load visible/nearby chunks first
2. **Request cancellation**: Cancel pending requests on view change
3. **Adaptive concurrency**: Adjust based on error rates
4. **Progress reporting**: Track loading progress for UI

### Alternative Long-term Solutions

1. **Server-side aggregation**: API endpoint that merges chunks
2. **WebSocket streaming**: Push-based chunk delivery
3. **IndexedDB caching**: Persist chunks locally
4. **Service Worker**: Background chunk pre-fetching

---

## Changelog

- **v1.0.0** (2025-01-04): Initial specification
  - Defined RequestQueue architecture
  - Specified integration points
  - Documented Python-side warning
  - Added performance analysis