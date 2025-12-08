# luxar-viewer.cache - Technical Specification

**Version**: 1.2.1
**Last Updated**: 2025-12-08

## Purpose

The cache package provides a two-level caching system with intelligent prefetching for zarr chunks, enabling offline viewing, instant reloads, reduced bandwidth, and proactive latency hiding through adjacent chunk prefetching.

---

## Architecture

### Data Pipeline Position

```
HTTP Request → TwoLevelCachingStore → zarrita decompression → ArrayDecoder → RangeCache → Rendering
                ↑                                                              ↑
            Caches compressed chunks                              Caches decoded arrays
            (no overlap - different pipeline stages)
```

**Key Insight**: This cache operates on **raw compressed zarr chunks** (Uint8Array), complementing the existing RangeCache which operates on **decoded Float32Array data**. No memory competition.

### Two-Level Architecture

```
Request → L1 (Memory) → L2 (OPFS) → Remote HTTP
              ↓             ↓            ↓
          ~1μs          ~1ms         ~100ms
```

| Level | Storage | Speed  | Size      | Persistence  | Eviction      |
| ----- | ------- | ------ | --------- | ------------ | ------------- |
| L1    | Memory  | ~1μs   | 100MB     | Session only | Segmented LRU |
| L2    | OPFS    | ~1ms   | 2GB       | Permanent    | LRU           |
| L3    | Remote  | ~100ms | Unlimited | N/A          | N/A           |

---

## Core Algorithms

### 1. LRU Cache (O(1) Operations)

**Algorithm**: Least Recently Used eviction using JavaScript Map

**Key Properties**:

- Map maintains insertion order
- Delete + re-insert moves item to end (most recently used)
- First item is always least recently used

**Operations**:

- `get(key)`: O(1) - Return value and move to end
- `set(key, value)`: O(1) - Add/update and evict LRU if needed
- `delete(key)`: O(1) - Remove entry
- Eviction: O(k) where k = items to evict (typically 1)

**Eviction Strategy**:

```
while (currentSize + newItemSize > maxSize && cache.size > 0):
    oldestKey = cache.keys().next().value  // O(1) - first in Map
    evict(oldestKey)                        // O(1)
```

**Size Tracking**:

- Each value has computed size via `getSize(v)` function
- `currentSize` maintained incrementally
- Accurate byte-level accounting

### 2. Segmented LRU (Name-Based Protection)

**Algorithm**: Route cache entries to dedicated segments by file name pattern

**Segments**:

1. **Metadata segment** (20% of cache, min 10MB)
   - Files: `.zmetadata`, `.zarray`, `.zattrs`, `zarr.json`
   - Protected from eviction by data chunks

2. **Chunks segment** (80% of cache)
   - Everything else (data chunks)
   - Standard LRU eviction

**Routing Logic**:

```
if key.endsWith('.zmetadata') or key.endsWith('.zarray') or
   key.endsWith('.zattrs') or key.endsWith('zarr.json') or key == any_of_those:
    → metadata segment
else:
    → chunks segment
```

**Benefits**:

- No cold start - metadata protected from first access
- Deterministic - we KNOW which files are metadata
- Simple - no frequency tracking needed

**Size Calculation**:

```
metadataSize = max(totalSize * 0.2, 10MB)
chunksSize = totalSize - metadataSize
```

### 3. OPFS Persistence Layer (L2) with Shallow Bucketing

**Storage**: Browser's Origin Private File System (OPFS)

**Problem**: With a 2GB cache and 32KB chunks, up to 65,000 files could be stored. A single directory with 65,000 files causes performance issues in some browsers/filesystems.

**Solution**: Shallow bucketing distributes files across 256 subdirectories.

**Bucket Hash Algorithm**:

```
function getBucket(key: string): string {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << 5) - hash + key.charCodeAt(i)) | 0;
  }
  return (hash & 0xff).toString(16).padStart(2, '0');  // "00" to "ff"
}
```

**Properties**:

- Simple djb2-style hash
- 256 buckets (00-ff)
- Distributes ~65,000 files into ~250 per bucket
- Deterministic (same key always maps to same bucket)

**File Name Encoding**:

```
function keyToFileName(key: string): string {
  const base64 = btoa(key);
  return base64.replace(/\//g, '_').replace(/=/g, '-').replace(/\+/g, '.');
}
```

**Data Structure**:

```
OPFS root/
└── zarr-cache-{url-hash}/
    ├── 00/                           # Bucket directories (256 total)
    │   ├── cG9pbnRz...               # Base64-encoded file
    │   └── ...
    ├── 01/
    │   └── ...
    ├── ...
    ├── ff/
    └── _cache_meta.json              # Index + metadata (at root)
```

**Example**:

```
Key: "points/positions/0.0.0"
  → Bucket: "23" (from hash)
  → Filename: "cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA"
  → Path: 23/cG9pbnRzL3Bvc2l0aW9ucy8wLjAuMA
```

**Bucket Handle Caching**:

```
bucketHandles = Map<string, FileSystemDirectoryHandle>()

async getBucketHandle(bucket, create):
  if bucketHandles.has(bucket):
    return bucketHandles.get(bucket)  // O(1) cache hit

  handle = await opfsRoot.getDirectoryHandle(bucket, { create })
  bucketHandles.set(bucket, handle)
  return handle
```

**Benefits**:

- Max ~250 files per directory instead of 65,000
- Only 256 bucket handles to cache (trivial memory)
- 2 async calls per file access (bucket + file) vs 1 - negligible overhead
- Efficient clear(): iterates 256 directories, not 65,000 files
- Bucket handles cached → subsequent accesses to same bucket are O(1)

**Metadata Structure** (\_cache_meta.json):

```json
{
  "baseUrl": "https://example.com/dataset.zarr",
  "entries": [
    [".zmetadata", { "size": 2048, "order": 1 }],
    ["points/positions/0.0.0", { "size": 32768, "order": 2 }]
  ],
  "totalSize": 34816,
  "orderCounter": 3,
  "contentHash": "8649296f56b76790..."
}
```

Note: The index stores original keys (e.g., `points/positions/0.0.0`), not bucketed paths. Bucket/filename are computed on access.

**LRU Tracking**:

- Each entry has `order` number (monotonically increasing)
- Lowest order = least recently used
- Finding LRU: O(n) scan over index Map
- Eviction triggered when `totalSize + newSize > maxSize`

**Debounced Saves**:

- Metadata saves debounced with 1 second delay
- Prevents excessive OPFS writes during rapid access
- Flushed on dispose() for clean shutdown

### 4. Content-Hash Validation

**Algorithm**: Hierarchical xxhash64 hashing for cache invalidation

**Python Side** (Computation):

```
For each zarr node (post-order traversal):
  hasher = xxhash64()

  1. Hash own datasets (sorted):
     for each array in node.array_keys() (sorted):
       hasher.update(array[:].tobytes())

  2. Hash metadata (excluding content_hash):
     attrs = {k:v for k,v in node.attrs if k != 'content_hash'}
     hasher.update(json.dumps(attrs, sort_keys=True))

  3. Hash child hashes (sorted):
     for each child in node.group_keys() (sorted):
       child_hash = compute_hash_recursive(child)
       hasher.update(child_hash.encode())

  content_hash = hasher.hexdigest()
  node.attrs['content_hash'] = content_hash

  return content_hash
```

**TypeScript Side** (Validation):

```
On cache init():
  1. Read root .zattrs (from cache or network)
  2. Extract content_hash attribute
  3. Compare with cached hash from _cache_meta.json
  4. If mismatch → clear L2 cache
  5. If match or no hash → keep cache
  6. Update cached hash
```

**Properties**:

- **Deterministic**: Same data always produces same hash
- **Hierarchical**: Parent hash includes all child hashes
- **Content-based**: Detects actual changes, not just timestamps
- **Offline-friendly**: Hash read from cached `.zattrs`, no network needed
- **Zarr-native**: Stored in standard zarr attributes

### 5. Quota Management

**Strategy**: Check-before-write with 10% safety margin

**Algorithm**:

```
async checkQuota(requiredBytes):
  estimate = await navigator.storage.estimate()
  available = estimate.quota - estimate.usage
  return available > requiredBytes * 1.1  // 10% buffer
```

**Fallback**: If quota exceeded, skip L2 write but continue with L1 and HTTP

### 6. Intelligent Chunk Prefetching

**Algorithm**: Proactive loading of adjacent chunks to hide network latency

**Trigger Points**:

- L1 hit: NO prefetch (too fast, ~1μs overhead not worth it)
- L2 hit: Prefetch adjacent chunks from L3 → L2 + L1
- L3 fetch: Prefetch adjacent chunks from L3 → L2 + L1

**Adjacent Chunk Calculation**:

```
For chunk at indices [i, j, k, ...]:
  adjacent = []
  for each dimension d:
    for delta in [-1, +1]:
      new_indices = copy(indices)
      new_indices[d] += delta

      if new_indices[d] < 0:
        skip  // No negative indices

      adjacent.append(format_chunk_key(new_indices))

  return adjacent
```

**Example** (3D chunk at [1, 2, 3]):

- Generates 6 neighbors: [0,2,3], [2,2,3], [1,1,3], [1,3,3], [1,2,2], [1,2,4]
- 4D chunk generates 8 neighbors, nD generates 2n neighbors

**Chunk Key Formats**:

```
Zarr v2 (dot notation):  points/positions/0.1.2
Zarr v3 (path notation): points/positions/c/0/1/2
```

**Parsing Algorithm**:

```
function parseChunkIndices(key):
  // CRITICAL: Check v3 FIRST (v2 regex can match v3 paths!)

  // v3: /c/ followed by slash-separated digits
  v3Match = key.match(/\/c\/(\d+(?:\/\d+)*)$/)
  if v3Match:
    return v3Match[1].split('/').map(Number)

  // v2: slash followed by dot-separated digits
  v2Match = key.match(/\/(\d+(?:\.\d+)*)$/)
  if v2Match:
    return v2Match[1].split('.').map(Number)

  return null  // Metadata file, skip prefetching
```

**Concurrency Control**:

```
class ChunkPrefetcher:
  queue = Set()           // Pending prefetch requests
  inFlight = Set()        // Currently fetching
  maxConcurrent = 4       // Limit concurrent prefetches
  processing = false      // Prevent concurrent queue processing

  onAccess(key):
    adjacent = getAdjacentChunks(key)
    for adjKey in adjacent:
      // Full deduplication
      if adjKey not in queue and adjKey not in inFlight:
        queue.add(adjKey)

    processQueue()  // Fire-and-forget (not awaited)

  async processQueue():
    if processing: return  // Already running
    processing = true

    try:
      while queue.size > 0 and inFlight.size < maxConcurrent:
        key = queue.pop()
        if not key: break

        inFlight.add(key)

        // Fire-and-forget with cleanup
        store.get(key)
          .catch(() => {})  // Ignore errors
          .finally(() => {
            inFlight.delete(key)
            if queue.size > 0:
              queueMicrotask(processQueue)  // Re-trigger when slot frees
          })
    finally:
      processing = false
```

**Properties**:

- **Non-blocking**: Fire-and-forget pattern, never blocks normal requests
- **Concurrency-limited**: Max 4 concurrent prefetch requests
- **Deduplication**: Checks both queue and inFlight sets
- **Race-safe**: queueMicrotask ensures no stranded queue items
- **Error-tolerant**: Silently ignores 404s and network failures

**Performance Impact**:

- Memory: Prefetched data competes for L1 space (LRU handles naturally)
- Network: With max 4 concurrent, leaves bandwidth for normal requests (HTTP/1.1: 2/6 slots, HTTP/2: 4/100+ streams)
- CPU: Minimal (Set operations are O(1))

For complete details, see: [`docs/CACHE_PREFETCHING_SPEC.md`](../../../../docs/CACHE_PREFETCHING_SPEC.md)

---

## Data Flow

### First Load (Cold Cache)

```
1. zarrita requests chunk "points/positions/0.0.0"
2. TwoLevelCachingStore.get()
   a. Check L1 → miss
   b. Check L2 → miss
   c. HTTP fetch → 32KB compressed chunk
   d. Store in L1 (sync)
   e. Store in L2 (async)
   f. Return to zarrita
3. zarrita decompresses → Float32Array
4. Rendering
```

### Subsequent Load (Warm L1)

```
1. zarrita requests chunk
2. TwoLevelCachingStore.get()
   a. Check L1 → HIT! (~1μs)
   b. Return immediately
3. zarrita decompresses
4. Rendering
```

### Page Reload (Warm L2, Cold L1)

```
1. zarrita requests chunk
2. TwoLevelCachingStore.get()
   a. Check L1 → miss (cleared on page load)
   b. Check L2 → HIT! (~1ms)
   c. Promote to L1
   d. Return
3. zarrita decompresses
4. Rendering
```

---

## Cache Invalidation

**Trigger**: Root content_hash mismatch detected on init()

**Process**:

```
1. Fetch root .zattrs DIRECTLY from HTTP (bypass cache!)
   - CRITICAL: Uses getRemoteContentHash() not get()
   - Prevents circular reference (comparing cached hash to itself)
2. Extract content_hash from fresh server response
3. Compare with L2 cached hash
4. If different:
   - Log warning
   - Clear L2 completely
   - Fetch fresh data
5. Update cached hash
```

**Why Direct Fetch?**: The validation MUST bypass cache to detect true server state changes. If validation used `get()`, it would read the cached `.zattrs` with the old hash, compare it to itself, and always validate successfully (false positive). This bug was discovered when switching datasets on the same port showed stale cached data.

**External Datasets**:

- If no content_hash present → skip validation
- Cache remains usable but won't auto-invalidate
- All Luxar-generated datasets include hashes

---

## Performance Characteristics

### Time Complexity

| Operation | L1 (LRUCache) | L1 (Segmented) | L2 (OPFSStore)                 |
| --------- | ------------- | -------------- | ------------------------------ |
| get()     | O(1)          | O(1)           | O(log n) OPFS lookup           |
| set()     | O(k) evict    | O(k) evict     | O(n) LRU scan + O(log n) write |
| delete()  | O(1)          | O(1)           | O(log n)                       |
| has()     | O(1)          | O(1)           | O(1) Map lookup                |

Where:

- k = number of items to evict (typically 1)
- n = number of cached entries
- OPFS operations include file system navigation

### Space Complexity

**L1 Memory**:

- Metadata: max(20% of L1, 10MB) = 20MB for default 100MB
- Chunks: 80MB for default 100MB
- Overhead: ~50 bytes per Map entry

**L2 OPFS**:

- Data: Up to 2GB compressed chunks
- Metadata: ~100 bytes per entry in \_cache_meta.json
- Filesystem overhead: Varies by browser

**Compression Ratio Impact**:

- Zarr chunks compressed with blosc+zstd: typically 3-10x
- 100MB L1 compressed ≈ 300MB-1GB decoded coverage
- Actual ratio depends on data characteristics

---

## Error Handling

### Graceful Degradation

**OPFS Not Available** (older browsers):

```
try:
  opfsRoot = await navigator.storage.getDirectory()
catch:
  opfsRoot = null  // L1-only mode
```

**Quota Exceeded**:

```
if (!checkQuota(size)):
  log warning
  return early  // Skip L2, use L1 + HTTP
```

**Corrupted Cache Data**:

```
data = await readFromOPFS(key)
if data.byteLength != expectedSize:
  delete(key)
  return undefined  // Refetch from HTTP
```

**Network Offline**:

- Content hash validation fails → use cached data anyway
- HTTP fetch fails → return undefined (zarrita handles gracefully)

---

## Browser Compatibility

**OPFS Support**:

- ✅ Chrome 86+ (October 2020)
- ✅ Firefox 111+ (March 2023)
- ✅ Safari 15.2+ (December 2021)

**Fallback**: L1-only mode on older browsers

---

## Configuration

**URL Parameters** (runtime overrides):

- `?no-cache` - Disable all caching
- `?cache-debug` - Enable verbose logging
- `?clear-cache` - Clear cache before loading
- `?no-prefetch` - Disable prefetching (cache still active)
- `?prefetch-debug` - Enable verbose prefetch logging

**Cache Config Options**:

```typescript
{
  enabled: boolean,       // Default: true
  l1MaxSizeMB: number,    // Default: 100
  l2MaxSizeMB: number,    // Default: 2048
  debug: boolean          // Default: false
}
```

**Prefetch Config Options**:

```typescript
{
  maxConcurrent: number,  // Default: 4
  enabled: boolean,       // Default: true
  debug: boolean          // Default: false
}
```

---

## Implementation Notes

### Why Map for LRU?

JavaScript Map maintains insertion order natively:

- First item = oldest (LRU)
- Last item = newest (MRU)
- Delete + re-insert = move to end
- No need for doubly-linked list

### Why xxhash64 over SHA256?

- 5-10x faster than SHA256
- Sufficient collision resistance for cache validation
- Already in Luxar dependencies
- 64-bit hash = 16 hex characters

### Why Segmented LRU?

Prevents metadata thrashing:

- Small datasets: 20% may exceed 10MB, metadata fully protected
- Large datasets: 10MB floor ensures .zmetadata never evicted
- Data chunks cycle normally through LRU

### Why Content Hash over Last-Modified?

- Works offline (no HTTP HEAD request)
- Content-based (detects actual changes)
- Zarr-native (stored in attributes)
- Future: hierarchical invalidation (per-subtree)

---

## Related Specifications

**Luxar viewer data loading**:

- RangeCache - Decoded array caching (see `../data/README.md`)
- Spatial index loading (see `../data/README.md`)

**Zarr format**:

- Consolidated metadata (see `@zarrita` documentation)
- Chunked array storage
- Compression with blosc+zstd

**Browser APIs**:

- OPFS (Origin Private File System)
- File System Access API
- Storage quota API

---

## Future Enhancements

1. **Hierarchical invalidation**: Use per-node content hashes to invalidate only changed subtrees
2. **Compression stats tracking**: Monitor bandwidth savings and compression ratios
3. **Adaptive sizing**: Automatically adjust L1/L2 based on dataset size and usage patterns
4. **Directional prefetching**: Track access patterns to predict navigation direction
5. **Fetch Priority API**: Use browser's fetch priority hints for prefetch requests
6. **Service Worker integration**: True offline-first PWA with background sync
7. **Cross-tab coordination**: BroadcastChannel for multi-tab cache sharing

---

## Changelog

- **v1.2.1** (2025-12-08): Cache validation bypass fix
  - **CRITICAL BUG FIX**: `validateCache()` now bypasses cache when fetching content_hash
  - Added `getRemoteContentHash()` method that fetches `.zattrs` directly from HTTP
  - Removed `getRootAttrs()` method (was reading from cache, causing false positives)
  - Fixes issue where switching datasets on same port showed stale data
  - Added 2 comprehensive unit tests verifying bypass behavior
  - See: `two-level-caching-store.ts:172-192`

- **v1.2.0** (2025-12-06): Shallow bucketing for OPFS storage
  - Added 256-bucket directory structure to distribute files
  - Prevents filesystem performance issues with 65,000+ files
  - Simple djb2-style hash for bucket assignment
  - Bucket handle caching for efficient repeated access
  - Max ~250 files per directory instead of 65,000
  - Minimal overhead: 2 async calls vs 1 per file access
  - Efficient clear(): iterates 256 directories, not all files
  - Index still stores original keys (bucket computed on access)

- **v1.1.0** (2025-01-06): Intelligent chunk prefetching
  - Added ChunkPrefetcher class for transparent adjacent chunk prefetching
  - Symmetric ±1 adjacency calculation in all dimensions
  - Zarr v2 and v3 chunk key parsing support
  - Concurrency-limited (max 4) fire-and-forget pattern
  - Full deduplication (queue + in-flight sets)
  - Race-condition safe queue processing with queueMicrotask
  - L2/L3 trigger points (not L1 - too fast for overhead)
  - URL parameters: ?no-prefetch, ?prefetch-debug
  - Auto-enabled in scene loader by default
  - Comprehensive test suite (21 tests)
  - See: `docs/CACHE_PREFETCHING_SPEC.md` for complete specification

- **v1.0.0** (2025-01-06): Initial specification
  - Two-level caching architecture (L1: memory, L2: OPFS)
  - Segmented LRU with 20/80 metadata/chunks split
  - Content-hash validation for cache invalidation
  - O(1) LRU operations using JavaScript Map
  - OPFS persistence with quota checking
  - Hierarchical xxhash64 hashing (Python-side)
  - zarrita AsyncReadable interface implementation
  - URL parameter overrides (?no-cache, ?cache-debug, ?clear-cache)
