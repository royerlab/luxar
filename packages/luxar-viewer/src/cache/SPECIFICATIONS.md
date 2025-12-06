# luxar-viewer.cache - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2025-01-06

## Purpose

The cache package provides a two-level caching system for zarr chunks, enabling offline viewing, instant reloads, and reduced bandwidth by caching compressed zarr data in browser memory (L1) and OPFS persistent storage (L2).

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

### 3. OPFS Persistence Layer (L2)

**Storage**: Browser's Origin Private File System (OPFS)

**Data Structure**:

```
OPFS root/
└── zarr-cache-{url-hash}/
    ├── _cache_meta.json        // Index + metadata
    ├── .zmetadata              // Cached files
    ├── .zattrs
    ├── points/
    │   ├── positions/
    │   │   ├── 0.0.0          // Cached chunks
    │   │   └── 0.0.1
    │   └── colors/
    │       └── 0.0.0
    └── ...
```

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
1. Read root .zattrs (through cache layers)
2. Extract content_hash
3. Compare with L2 cached hash
4. If different:
   - Log warning
   - Clear L2 completely
   - Clear L1
   - Fetch fresh data
5. Update cached hash
```

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

**Config Options**:

```typescript
{
  enabled: boolean,       // Default: true
  l1MaxSizeMB: number,    // Default: 100
  l2MaxSizeMB: number,    // Default: 2048
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
4. **Prefetching**: Background download of adjacent chunks based on navigation patterns
5. **Service Worker integration**: True offline-first PWA with background sync
6. **Cross-tab coordination**: BroadcastChannel for multi-tab cache sharing

---

## Changelog

- **v1.0.0** (2025-01-06): Initial specification
  - Two-level caching architecture (L1: memory, L2: OPFS)
  - Segmented LRU with 20/80 metadata/chunks split
  - Content-hash validation for cache invalidation
  - O(1) LRU operations using JavaScript Map
  - OPFS persistence with quota checking
  - Hierarchical xxhash64 hashing (Python-side)
  - zarrita AsyncReadable interface implementation
  - URL parameter overrides (?no-cache, ?cache-debug, ?clear-cache)
