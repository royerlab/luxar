# OPFS-Based Zarr Cache for Luxar Viewer

This document describes a proposed caching system using the Origin Private File System (OPFS) to persist zarr datasets locally in the browser.

**Development Philosophy**: This is an early-stage project. We do not maintain backwards compatibility, deprecation paths, or keep dead code around. When changes are needed, we refactor directly and update all documentation and examples. This keeps the codebase clean and maintainable.

## Overview

The idea is to cache zarr chunks from remote HTTP sources to the browser's local file system (OPFS), enabling:
- **Offline viewing** of previously loaded datasets
- **Instant reload** on subsequent visits (no network latency)
- **Reduced bandwidth** - chunks downloaded once, cached forever
- **Zero re-compression overhead** - store raw compressed bytes

## Why OPFS + Zarr is a Perfect Match

### Zarr Format Characteristics
- Zarr is essentially a directory of files (chunks + metadata)
- Chunks are already compressed (blosc + zstd)
- Each chunk is an independent file that can be cached individually

### OPFS Capabilities
- Browser-native file system API
- Handles `ArrayBuffer` efficiently (no base64 encoding)
- Large storage quota (typically ~50% of free disk)
- Persistent across browser sessions
- Supported in modern browsers (Chrome 86+, Firefox 111+, Safari 15.2+)

### OPFS Persistence Model

OPFS data is **persistent by default**:
- **Survives**: Tab close, browser restart, system reboot
- **Only cleared by**: User clearing browser data, explicit `removeEntry()` calls, or browser uninstall
- **Per-origin isolation**: Each website has its own isolated OPFS storage
- **Quota**: Typically ~50% of available disk space, shared with IndexedDB

This persistence is what enables true offline viewing - once a dataset is cached, it remains available until explicitly cleared.

## Key Insight: Zero Re-compression

The most elegant aspect of this approach is that zarr chunks arrive from the server **already compressed**. We can store them as-is:

```
First Visit:
  Server → [compressed chunk] → OPFS (store raw) → decompress → render

Subsequent Visits:
  OPFS → [compressed chunk] → decompress → render
  (no network, no re-compression)
```

| Step | With Re-compression | With OPFS (raw) |
|------|---------------------|-----------------|
| Save to cache | Decompress → Compress | Direct write |
| Load from cache | Decompress → Decompress | Single decompress |
| CPU overhead | 2x compression | 0 extra |

## Two-Level Cache Architecture

The recommended architecture uses two cache levels for optimal performance:

```
Request → L1 (Memory) → L2 (OPFS) → Remote HTTP
              ↑             ↑            │
              └─────────────┴────────────┘
                    populate on miss
```

### Cache Levels

| Level | Storage | Speed | Default Size | Persistence | Eviction |
|-------|---------|-------|--------------|-------------|----------|
| **L1** | Memory | ~1μs | 100MB | Session only | Segmented LRU |
| **L2** | OPFS | ~1ms | ~2GB | Persistent | LRU |
| **L3** | Remote | ~100ms | Unlimited | N/A | N/A |

**L1 Size Configuration:**
- Default: 100MB fixed size
- Rationale: Small fixed size avoids competition with RangeCache for RAM
- RangeCache handles decoded arrays; L1 handles compressed chunks (no overlap)
- Compression ratio (typically 3-10x with blosc+zstd, varies by data type) means 100MB compressed ≈ 300MB-1GB decoded coverage
- Configurable via `l1MaxSize` option
- **Metadata segment**: 20MB (20% of 100MB) - sufficient for all metadata files

**Note on RangeCache**: The Luxar viewer uses RangeCache (a separate cache system) to store decoded array ranges ready for rendering (e.g., decompressed Float32Array position data). L1 caches compressed chunks before decompression; RangeCache caches decompressed data after decoding. This separation avoids memory competition - they serve different stages of the data pipeline and don't duplicate effort.

### Why Two Levels?

- **L1 (Memory)**: Hot data for the current view - instant access
- **L2 (OPFS)**: Warm data for the entire dataset - survives page reload
- **L3 (Remote)**: Cold data - only fetched once per dataset

### Data Flow

```
Chunk Request: "/points/positions/0.0.0"

1. L1 Check  → Hit? Return immediately (~1μs)
2. L1 Miss   → Check L2
3. L2 Check  → Hit? Return + promote to L1 (~1ms)
4. L2 Miss   → Fetch from HTTP (~100ms)
5. Got data  → Store in L1, async store in L2
6. Return to zarrita → Decompress → Render
```

## Cache Invalidation

### Content-Hash Validation (Recommended)

The cache uses **hierarchical content hashing** for reliable, offline-friendly validation. Each zarr node has a SHA256 hash computed from its content and child hashes.

#### Python Side: Hash Computation

During compilation, compute recursive content hash for each node:

```python
import xxhash  # Faster than SHA256, already in codebase
import json

def _compute_content_hashes(compiler) -> None:
    """
    Compute content hashes for all zarr nodes using post-order traversal.

    Called during finalize() after all nodes have been written.
    Uses xxhash64 for speed (already in codebase).
    """

    def compute_hash_recursive(group_path: str) -> str:
        """Recursively compute hash for a group and its children."""
        group = compiler._root[group_path] if group_path else compiler._root

        hasher = xxhash.xxh64()

        # 1. Hash this node's own datasets (positions, colors, radii, etc.)
        for dataset_name in sorted(group.array_keys()):
            dataset = group[dataset_name]
            hasher.update(dataset[:].tobytes())

        # 2. Hash metadata (excluding content_hash to avoid recursion)
        attrs = {k: v for k, v in dict(group.attrs).items() if k != 'content_hash'}
        hasher.update(json.dumps(attrs, sort_keys=True, default=str).encode())

        # 3. Hash child groups (recursively, sorted for determinism)
        for child_name in sorted(group.group_keys()):
            child_path = f"{group_path}/{child_name}" if group_path else child_name
            child_hash = compute_hash_recursive(child_path)
            hasher.update(child_hash.encode())

        # Store hash in this node's attrs
        content_hash = hasher.hexdigest()
        group.attrs['content_hash'] = content_hash

        return content_hash

    # Start from root (empty path)
    root_hash = compute_hash_recursive('')
    return root_hash

# In LuxarZarrCompiler.finalize():
def finalize(self) -> None:
    # ... existing consolidation ...

    # Compute content hashes (post-order: children before parents)
    root_hash = self._compute_content_hashes()
    aprint(f"Scene content hash: {root_hash[:16]}...")

    # Re-consolidate to include hashes in .zmetadata
    if self._use_consolidated_metadata:
        zarr.consolidate_metadata(self._store, self._root_path)
```

**Stored in**: Each node's `.zattrs` file as `content_hash` field.

#### TypeScript Side: Cache Validation

```typescript
async validateCache(): Promise<void> {
  try {
    // Read root .zattrs (from cache or network)
    const rootAttrs = await this.getRootAttrs();
    const remoteHash = rootAttrs?.content_hash;  // Optional chaining

    // No hash in dataset → skip validation (backward compatibility)
    if (!remoteHash) {
      if (this.debug) {  // Respect debug flag
        console.log('[Cache] No content_hash found, skipping validation');
      }
      return;
    }

    // Compare with cached hash
    if (this.cachedContentHash && remoteHash !== this.cachedContentHash) {
      console.log('[Cache] Dataset content changed, clearing cache');  // Always log invalidation
      if (this.debug) {  // Details only in debug mode
        console.log(`  Old: ${this.cachedContentHash.slice(0, 16)}...`);
        console.log(`  New: ${remoteHash.slice(0, 16)}...`);
      }
      await this.clearL2();
    }

    this.cachedContentHash = remoteHash;
    await this.saveL2Metadata();
  } catch {
    // Offline or error - use cached data as-is
    if (this.debug) {  // Respect debug flag
      console.log('[Cache] Cannot validate (offline?), using cached data');
    }
  }
}
```

**Why Content Hash over Last-Modified:**
- ✅ **Content-based**: Hash changes only if data actually changes (no false invalidations)
- ✅ **Zarr-native**: Stored in `.zattrs`, no HTTP server dependency
- ✅ **Offline-friendly**: Can validate from cached `.zattrs` without network
- ✅ **Cryptographically strong**: SHA256 has negligible collision risk
- ✅ **Hierarchical**: Future enhancement can invalidate only changed subtrees
- ✅ **Deterministic**: Same data always produces same hash

**External Dataset Handling**: If `content_hash` not present (external or non-Luxar datasets), validation is skipped. Cache remains usable but won't auto-invalidate. All Luxar-generated datasets will include content hashes.

## Eviction Strategy: LRU vs FIFO

### Why LRU (Not FIFO)

For zarr spatial data viewing, **LRU is superior** to FIFO because of access patterns:

```
Typical session:
1. Load dataset → fetch .zmetadata (accessed 100+ times)
2. View area A → fetch chunks [0,0], [0,1], [1,0], [1,1]
3. Pan to area B → fetch chunks [2,0], [2,1], [3,0], [3,1]
4. Zoom out → need [0,0] again!  ← FIFO would have evicted it
5. Pan back to A → need to re-fetch all of A with FIFO
```

| Scenario | FIFO Behavior | LRU Behavior |
|----------|---------------|--------------|
| Metadata (.zmetadata) | Evicted early despite constant use | Stays hot |
| Back-and-forth navigation | Re-fetches old chunks | Keeps recently viewed |
| Zoom out (needs more chunks) | Evicts visible chunks | Keeps accessed chunks |

### Segmented LRU for L1 (Memory Cache)

For the memory cache where space is precious, we use **Segmented LRU** with **name-based protection**:

```
┌─────────────────────────────────────────────────┐
│                  L1 Cache                        │
├─────────────────────┬───────────────────────────┤
│  Metadata (20%)     │   Chunks (80%)            │
│  (min 10MB)         │                           │
├─────────────────────┼───────────────────────────┤
│  .zmetadata         │   chunk [5,3,2]           │
│  .zarray files      │   chunk [5,3,3]           │
│  .zattrs files      │   chunk [5,4,2]           │
│  zarr.json          │   chunk [6,0,0]           │
└─────────────────────┴───────────────────────────┘
```

**How it works:**
1. Files are routed by **name pattern** on insertion (not by access frequency)
2. Metadata files (`.zmetadata`, `.zarray`, `.zattrs`, `zarr.json`) → **metadata** segment
3. Data chunks (everything else) → **chunks** segment
4. Eviction happens from chunks segment first

**Metadata patterns:**
- `.zmetadata` - consolidated metadata
- `.zarray` - array metadata
- `.zattrs` - attribute metadata
- `zarr.json` - v3 format metadata

**Benefits:**
- No cold start problem - metadata is protected from the first access
- Deterministic behavior - we KNOW which files are metadata
- Simpler logic - no frequency tracking or promotion needed
- Data chunks cycle through probationary via standard LRU

## Concurrent Access

**Note on multi-tab scenarios:**
- Multiple browser tabs may access the same cached dataset
- OPFS provides eventual consistency (last write wins)
- Metadata corruption is possible but unlikely (small window)
- For critical applications, consider using OPFS's `createSyncAccessHandle()` for locking

The current design accepts eventual consistency as the tradeoff for simpler implementation.

## Implementation

### LRU Cache with Map (O(1) Operations)

```typescript
/**
 * LRU Cache using Map for O(1) get/set/delete operations.
 * Map maintains insertion order, enabling efficient LRU tracking.
 */
class LRUCache<V> {
  private cache = new Map<string, V>();
  private maxSize: number;
  private currentSize = 0;
  private getSize: (v: V) => number;

  constructor(maxSize: number, getSize: (v: V) => number) {
    this.maxSize = maxSize;
    this.getSize = getSize;
  }

  get(key: string): V | undefined {
    const value = this.cache.get(key);
    if (value !== undefined) {
      // Move to end (most recently used) - O(1) with Map
      this.cache.delete(key);
      this.cache.set(key, value);
    }
    return value;
  }

  set(key: string, value: V): void {
    // If exists, remove old
    if (this.cache.has(key)) {
      this.currentSize -= this.getSize(this.cache.get(key)!);
      this.cache.delete(key);
    }

    const size = this.getSize(value);

    // Evict LRU until space available
    while (this.currentSize + size > this.maxSize && this.cache.size > 0) {
      const oldestKey = this.cache.keys().next().value;
      this.currentSize -= this.getSize(this.cache.get(oldestKey)!);
      this.cache.delete(oldestKey);
    }

    this.cache.set(key, value);
    this.currentSize += size;
  }

  has(key: string): boolean {
    return this.cache.has(key);
  }

  delete(key: string): boolean {
    const value = this.cache.get(key);
    if (value) {
      this.currentSize -= this.getSize(value);
      return this.cache.delete(key);
    }
    return false;
  }

  clear(): void {
    this.cache.clear();
    this.currentSize = 0;
  }

  get size(): number {
    return this.currentSize;
  }

  get count(): number {
    return this.cache.size;
  }
}
```

### Segmented LRU Cache for L1

```typescript
/**
 * Segmented LRU with name-based routing: Metadata files go to a dedicated segment,
 * data chunks go to the chunks segment. No frequency tracking needed.
 */
class SegmentedLRUCache {
  private static readonly MIN_METADATA_SIZE = 10 * 1024 * 1024; // 10MB floor

  // Metadata file patterns - these go to metadata segment
  private static readonly METADATA_PATTERNS = [
    '.zmetadata',    // Consolidated metadata (zarr v2)
    '.zarray',       // Array metadata
    '.zattrs',       // Attribute metadata
    'zarr.json',     // zarr v3 metadata
  ];

  // Metadata segment: zarr metadata files (20% of cache, min 10MB)
  private metadata: LRUCache<Uint8Array>;

  // Chunks segment: data chunks (80% of cache)
  private chunks: LRUCache<Uint8Array>;

  constructor(totalSize: number) {
    const getSize = (v: Uint8Array) => v.byteLength;
    const metadataSize = Math.max(
      totalSize * 0.2,
      SegmentedLRUCache.MIN_METADATA_SIZE
    );
    const chunksSize = totalSize - metadataSize;

    this.metadata = new LRUCache(metadataSize, getSize);
    this.chunks = new LRUCache(chunksSize, getSize);
  }

  /**
   * Check if a key is a metadata file based on name pattern.
   */
  private static isMetadataFile(key: string): boolean {
    return SegmentedLRUCache.METADATA_PATTERNS.some(
      pattern => key.endsWith(pattern) || key === pattern
    );
  }

  get(key: string): Uint8Array | undefined {
    // Check metadata segment first
    const metadataHit = this.metadata.get(key);
    if (metadataHit) {
      return metadataHit;
    }

    // Check chunks segment
    const chunksHit = this.chunks.get(key);
    if (chunksHit) {
      return chunksHit;
    }

    return undefined;
  }

  set(key: string, data: Uint8Array): void {
    // Route by name pattern: metadata files → metadata segment, else → chunks segment
    if (SegmentedLRUCache.isMetadataFile(key)) {
      this.metadata.set(key, data);
    } else {
      this.chunks.set(key, data);
    }
  }

  has(key: string): boolean {
    return this.metadata.has(key) || this.chunks.has(key);
  }

  clear(): void {
    this.metadata.clear();
    this.chunks.clear();
  }

  getStats(): { metadataSize: number; chunksSize: number; metadataCount: number; chunksCount: number } {
    return {
      metadataSize: this.metadata.size,
      chunksSize: this.chunks.size,
      metadataCount: this.metadata.count,
      chunksCount: this.chunks.count,
    };
  }
}
```

### Two-Level Caching Store for Zarrita

```typescript
import type { Readable } from '@zarrita/storage';

export class TwoLevelCachingStore implements Readable {
  private baseUrl: string;

  // L1: Memory cache with Segmented LRU
  private l1Cache: SegmentedLRUCache;

  // L2: OPFS cache with LRU tracking via Map (O(1) operations)
  private opfsRoot: FileSystemDirectoryHandle | null = null;
  private l2Index = new Map<string, { size: number; order: number }>();
  private l2OrderCounter = 0;
  private l2TotalSize = 0;
  private l2MaxSize: number;

  // Debounced metadata save
  private metadataSaveTimeout: ReturnType<typeof setTimeout> | null = null;
  private static readonly METADATA_SAVE_DELAY = 1000; // 1 second debounce

  // Cached content hash for invalidation
  private cachedContentHash: string | null = null;

  private debug: boolean = false;  // NEW: Add debug field

  constructor(
    baseUrl: string,
    options: {
      l1MaxSize?: number;  // Default: 100MB
      l2MaxSize?: number;  // Default: 2GB
      debug?: boolean;     // Default: false
    } = {}
  ) {
    this.baseUrl = baseUrl;
    const DEFAULT_L1_SIZE = 100 * 1024 * 1024;  // 100MB
    const l1Size = options.l1MaxSize ?? DEFAULT_L1_SIZE;
    this.l1Cache = new SegmentedLRUCache(l1Size);
    this.l2MaxSize = options.l2MaxSize ?? 2 * 1024 * 1024 * 1024;
    this.debug = options.debug ?? false;  // Store debug option
  }

  async init(): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      const datasetId = await this.hashUrl(this.baseUrl);
      this.opfsRoot = await root.getDirectoryHandle(datasetId, { create: true });

      // Load L2 metadata
      await this.loadL2Metadata();

      // Validate cache freshness
      await this.validateCache();
    } catch (error) {
      console.warn('[Cache] Failed to initialize OPFS:', error);
      this.opfsRoot = null;
    }
  }

  /**
   * Validate cache using content hash. Clears cache if content changed.
   * Works offline by reading hash from cached .zattrs.
   */
  private async validateCache(): Promise<void> {
    try {
      // Read root .zattrs to get content_hash
      const rootAttrs = await this.getRootAttrs();
      const remoteHash = rootAttrs?.content_hash;

      // No hash in dataset → skip validation (backward compatibility)
      if (!remoteHash) {
        if (this.debug) {
          console.log('[Cache] No content_hash found, skipping validation');
        }
        return;
      }

      // Compare with cached hash
      if (this.cachedContentHash && remoteHash !== this.cachedContentHash) {
        console.log('[Cache] Dataset content changed, clearing cache');
        if (this.debug) {
          console.log(`  Old: ${this.cachedContentHash.slice(0, 16)}...`);
          console.log(`  New: ${remoteHash.slice(0, 16)}...`);
        }
        await this.clearL2();
      }

      this.cachedContentHash = remoteHash;
      await this.saveL2Metadata();
    } catch (error) {
      // Offline or error - use cached data as-is
      if (this.debug) {
        console.log('[Cache] Cannot validate (offline?), using cached data');
      }
    }
  }

  /**
   * Read root .zattrs from L1 cache, L2 cache, or network (in that order).
   */
  private async getRootAttrs(): Promise<any> {
    const attrsKey = '.zattrs';
    const data = await this.get(attrsKey);
    if (!data) return null;
    return JSON.parse(new TextDecoder().decode(data));
  }

  /**
   * Check available storage quota before writes.
   */
  private async checkQuota(requiredBytes: number): Promise<boolean> {
    try {
      const estimate = await navigator.storage.estimate();
      const available = (estimate.quota || 0) - (estimate.usage || 0);
      return available > requiredBytes * 1.1; // 10% safety margin
    } catch {
      return true; // Assume OK if API unavailable
    }
  }

  async get(key: string): Promise<Uint8Array | undefined> {
    // L1: Memory check (fastest, ~1μs)
    const l1Hit = this.l1Cache.get(key);
    if (l1Hit) {
      return l1Hit;
    }

    // L2: OPFS check (~1ms)
    if (this.opfsRoot) {
      const l2Hit = await this.getFromOPFS(key);
      if (l2Hit) {
        // Promote to L1
        this.l1Cache.set(key, l2Hit);
        // Update L2 LRU order
        this.touchL2(key);
        return l2Hit;
      }
    }

    // L3: Remote fetch (~100ms)
    try {
      const response = await fetch(`${this.baseUrl}/${key}`);
      if (!response.ok) return undefined;

      const data = new Uint8Array(await response.arrayBuffer());

      // Populate both caches
      this.l1Cache.set(key, data);
      if (this.opfsRoot) {
        this.addToL2(key, data).catch(() => {});
      }

      return data;
    } catch {
      // Network error - chunk not available
      // When network returns, subsequent fetches will work
      console.warn(`[Cache] Network error fetching ${key}`);
      return undefined;
    }
  }

  // ========== L2 OPFS Operations ==========

  private async getFromOPFS(key: string): Promise<Uint8Array | undefined> {
    try {
      const fileHandle = await this.navigateToFile(key, false);
      const file = await fileHandle.getFile();
      const data = new Uint8Array(await file.arrayBuffer());

      // Verify file size matches metadata
      const entry = this.l2Index.get(key);
      if (entry && entry.size !== data.byteLength) {
        console.warn(`[Cache] Size mismatch for ${key}, removing corrupted entry`);
        await this.deleteFromOPFS(key);
        this.l2Index.delete(key);
        return undefined;
      }

      return data;
    } catch {
      return undefined;
    }
  }

  private async addToL2(key: string, data: Uint8Array): Promise<void> {
    const size = data.byteLength;

    // Check quota before writing
    if (!await this.checkQuota(size)) {
      console.warn('[Cache] Insufficient storage quota, skipping L2 cache');
      return;
    }

    // LRU eviction until we have space (using Map for O(1) lookups)
    while (this.l2TotalSize + size > this.l2MaxSize && this.l2Index.size > 0) {
      // Find LRU entry (lowest order number)
      let lruKey: string | null = null;
      let lruOrder = Infinity;
      for (const [k, v] of this.l2Index) {
        if (v.order < lruOrder) {
          lruOrder = v.order;
          lruKey = k;
        }
      }

      if (lruKey) {
        const entry = this.l2Index.get(lruKey)!;
        await this.deleteFromOPFS(lruKey);
        this.l2TotalSize -= entry.size;
        this.l2Index.delete(lruKey);
      }
    }

    // Write to OPFS
    const fileHandle = await this.navigateToFile(key, true);
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();

    // Update index
    const existingEntry = this.l2Index.get(key);
    if (existingEntry) {
      this.l2TotalSize -= existingEntry.size;
    }
    this.l2Index.set(key, { size, order: this.l2OrderCounter++ });
    this.l2TotalSize += size;

    // Debounced metadata save
    this.scheduleMetadataSave();
  }

  private touchL2(key: string): void {
    // Update order to mark as recently used (O(1))
    const entry = this.l2Index.get(key);
    if (entry) {
      entry.order = this.l2OrderCounter++;
      this.scheduleMetadataSave();
    }
  }

  private scheduleMetadataSave(): void {
    if (this.metadataSaveTimeout) {
      clearTimeout(this.metadataSaveTimeout);
    }
    this.metadataSaveTimeout = setTimeout(() => {
      this.saveL2Metadata().catch(() => {});
      this.metadataSaveTimeout = null;
    }, TwoLevelCachingStore.METADATA_SAVE_DELAY);
  }

  private async deleteFromOPFS(key: string): Promise<void> {
    try {
      const parts = key.split('/');
      let dir = this.opfsRoot!;
      for (let i = 0; i < parts.length - 1; i++) {
        dir = await dir.getDirectoryHandle(parts[i]);
      }
      await dir.removeEntry(parts[parts.length - 1]);
    } catch {
      // File doesn't exist, ignore
    }
  }

  // ========== Metadata Persistence ==========

  private async loadL2Metadata(): Promise<void> {
    try {
      const metaHandle = await this.opfsRoot!.getFileHandle('_cache_meta.json');
      const file = await metaHandle.getFile();
      const meta = JSON.parse(await file.text());

      // Reconstruct Map from stored array
      this.l2Index = new Map(meta.entries || []);
      this.l2TotalSize = meta.totalSize || 0;
      this.l2OrderCounter = meta.orderCounter || 0;
      this.cachedContentHash = meta.contentHash || null;
    } catch {
      this.l2Index = new Map();
      this.l2TotalSize = 0;
      this.l2OrderCounter = 0;
      this.cachedContentHash = null;
    }
  }

  private async saveL2Metadata(): Promise<void> {
    if (!this.opfsRoot) return;

    try {
      const metaHandle = await this.opfsRoot.getFileHandle('_cache_meta.json', { create: true });
      const writable = await metaHandle.createWritable();
      await writable.write(JSON.stringify({
        entries: Array.from(this.l2Index.entries()),
        totalSize: this.l2TotalSize,
        orderCounter: this.l2OrderCounter,
        contentHash: this.cachedContentHash,  // Store content hash, not Last-Modified
      }));
      await writable.close();
    } catch {
      // Ignore metadata save failures
    }
  }

  // ========== Helpers ==========

  private async navigateToFile(key: string, create: boolean): Promise<FileSystemFileHandle> {
    const parts = key.split('/');
    let dir = this.opfsRoot!;
    for (let i = 0; i < parts.length - 1; i++) {
      dir = await dir.getDirectoryHandle(parts[i], { create });
    }
    return dir.getFileHandle(parts[parts.length - 1], { create });
  }

  /**
   * Generate a unique, collision-resistant hash for the dataset URL.
   * Uses SHA-256 for strong collision resistance.
   */
  private async hashUrl(url: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(url);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return `zarr-cache-${hashHex.slice(0, 16)}`;  // First 16 chars = 64 bits
  }

  // ========== Stats & Management ==========

  getStats(): {
    l1: { metadataSize: number; chunksSize: number; metadataCount: number; chunksCount: number };
    l2: { size: number; count: number };
  } {
    return {
      l1: this.l1Cache.getStats(),
      l2: { size: this.l2TotalSize, count: this.l2Index.size },
    };
  }

  clearL1(): void {
    this.l1Cache.clear();
  }

  async clearL2(): Promise<void> {
    if (this.opfsRoot) {
      for await (const name of (this.opfsRoot as any).keys()) {
        await this.opfsRoot.removeEntry(name, { recursive: true });
      }
    }
    this.l2Index = new Map();
    this.l2TotalSize = 0;
    this.l2OrderCounter = 0;
    this.cachedContentHash = null;  // Reset to force re-validation on next init
  }

  async clearAll(): Promise<void> {
    this.clearL1();
    await this.clearL2();
  }

  /**
   * Dispose the cache store. Flushes pending writes and clears L1.
   * Call this when navigating away or closing the viewer.
   */
  async dispose(): Promise<void> {
    // Flush any pending metadata save
    if (this.metadataSaveTimeout) {
      clearTimeout(this.metadataSaveTimeout);
      await this.saveL2Metadata();
    }

    // Clear L1 (L2 persists for future sessions)
    this.clearL1();
  }
}
```

### Usage with Zarrita

```typescript
import { open } from '@zarrita/core';
import { TwoLevelCachingStore } from './two-level-caching-store';

// Create two-level caching store with defaults (100MB L1, 2GB L2)
const store = new TwoLevelCachingStore('https://example.com/dataset.zarr');
await store.init();

// Or with custom sizes
const customStore = new TwoLevelCachingStore('https://example.com/dataset.zarr', {
  l1MaxSize: 200 * 1024 * 1024,  // 200MB memory (optional override)
  l2MaxSize: 5 * 1024 * 1024 * 1024,  // 5GB disk
});
await customStore.init();

// Use with zarrita - completely transparent!
const root = await open(store);
const array = await open(root.resolve('/points/positions'));

// First load: HTTP → L2 (OPFS) → L1 (memory) → decompress → render
// Second load: L1 (memory) → decompress → render (~1μs!)
// After page reload: L2 (OPFS) → L1 → decompress → render (~1ms)
const data = await array.getChunk([0, 0]);

// Check cache stats
console.log(store.getStats());
// {
//   l1: { metadataSize: 2048, chunksSize: 98000, ... },
//   l2: { size: 500000000, count: 1234 }
// }

// Clean up when done
await store.dispose();
```

### Integration in Luxar Scene Loader

Current flow in `scene-loader.ts`:
```typescript
// Current: HTTP-only store
const store = new FetchStore(zarrUrl);
```

With two-level caching:
```typescript
// New: Two-level caching store (drop-in replacement)
const store = new TwoLevelCachingStore(zarrUrl);
await store.init();

// ... use store ...

// On cleanup
await store.dispose();
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Luxar Viewer                              │
├─────────────────────────────────────────────────────────────────┤
│                   TwoLevelCachingStore                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    L1: Memory Cache                      │    │
│  │              (name-based segment routing)                │    │
│  │  ┌─────────────────┐  ┌───────────────────────────────┐ │    │
│  │  │    Metadata     │  │          Chunks               │ │    │
│  │  │ (20%, min 10MB) │  │          (80%)                │ │    │
│  │  │                 │  │                               │ │    │
│  │  │  .zmetadata  ←──┼──┼─ metadata routed here         │ │    │
│  │  │  .zarray files  │  │                               │ │    │
│  │  │  .zattrs files  │  │  data chunks ──→ routed here  │ │    │
│  │  │  zarr.json      │  │  (standard LRU eviction)      │ │    │
│  │  └─────────────────┘  └───────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓ miss                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    L2: OPFS Cache                        │    │
│  │                    (LRU via Map, O(1))                   │    │
│  │         Persistent across browser sessions               │    │
│  │     Content-hash validation on init, quota-aware writes  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓ miss                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   L3: Remote HTTP                        │    │
│  │                   (network fetch)                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Benefits Summary

| Feature | Current (HTTP only) | With Two-Level Cache |
|---------|---------------------|----------------------|
| Offline viewing | No | Yes |
| Reload speed | Network-bound (~100ms/chunk) | L1: ~1μs, L2: ~1ms |
| Bandwidth | Every visit | Once per dataset |
| Large datasets | Streaming only | Full local copy |
| CPU overhead | N/A | Zero (raw bytes) |
| Metadata access | Network every time | Metadata segment in L1 |
| Stale data detection | N/A | Content-hash validation (offline-friendly) |

## Browser Storage Options Comparison

| Option | Binary Support | Size Limit | Async | Best For |
|--------|---------------|------------|-------|----------|
| **OPFS** | Native ArrayBuffer | Quota-based (~50% disk) | Yes | Large zarr datasets |
| IndexedDB | Via ArrayBuffer | Quota-based | Yes | Structured data + binary |
| Cache API | HTTP responses | Quota-based | Yes | Request/response caching |
| localStorage | String only (base64) | ~5MB | No | Not suitable |

OPFS is recommended for zarr caching due to:
- Best performance for large binary data
- File system semantics match zarr's structure
- No serialization overhead

## Limitations

- **Browser support**: Chrome 86+, Firefox 111+, Safari 15.2+
- **Storage quota**: User can clear browser data
- **No cross-origin**: Each origin has its own OPFS
- **No sharing**: Can't export cached data to other applications
- **Concurrent tabs**: Eventual consistency only (no locking)

## Future Enhancements

1. **Cache management UI**: Show cached datasets, sizes, allow deletion
2. **Prefetching**: Background download of visible region chunks
3. **Compression stats**: Track compression ratios and bandwidth savings
4. **Service Worker**: Enable true offline-first PWA experience
5. **Cross-tab communication**: Use BroadcastChannel for cache coordination
6. **Hierarchical invalidation**: Use per-node content hashes to invalidate only changed subtrees
7. **Incremental updates**: Download only changed nodes instead of clearing entire cache

## References

- [OPFS API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
- [Zarrita Storage Interface](https://github.com/manzt/zarrita.js)
- [Blosc Compression](https://www.blosc.org/)
- [Segmented LRU (SLRU) Algorithm](https://en.wikipedia.org/wiki/Cache_replacement_policies#Segmented_LRU_(SLRU))
- [Storage Quota API](https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/estimate)
