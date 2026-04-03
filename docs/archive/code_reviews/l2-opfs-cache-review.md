# L2 OPFS Persistent Cache - Code Review

**Reviewer**: Claude Opus 4.6
**Date**: 2026-02-28
**Files Reviewed**:
- `packages/luxar-viewer/src/cache/opfs-store.ts`
- `packages/luxar-viewer/src/cache/types.ts`

---

## Executive Summary

The OPFS store is well-architected with 256-bucket sharding, debounced metadata persistence, and LRU eviction. One critical data corruption bug was found and **fixed**.

---

## Bugs Found and Fixed

### CRITICAL: `data.buffer` writes entire underlying ArrayBuffer — **FIXED**

**Severity**: CRITICAL
**File**: `opfs-store.ts`, line 135

```typescript
// BEFORE (broken):
await writable.write(data.buffer as ArrayBuffer);
```

If the `Uint8Array` is a view on a larger `ArrayBuffer` (e.g., from a sub-slice or offset view), `data.buffer` writes the **entire** underlying buffer, not just the view's portion. This causes silent data corruption where stored files are larger than expected and contain garbage bytes.

**Fix**: Write `data` directly (the `Uint8Array` itself), which `FileSystemWritableFileStream.write()` accepts natively.

```typescript
// AFTER (fixed):
await writable.write(data);
```

---

## Other Findings

### HIGH: Metadata save may be lost on page close

**Severity**: HIGH
**File**: `opfs-store.ts`, lines 330-337

`scheduleMetadataSave()` uses a 1-second debounce. If the page closes within that window, the metadata update is lost. The `dispose()` method calls `saveMetadata()`, but `beforeunload` handlers may be interrupted by the browser before async operations complete.

**Impact**: Metadata could desync from actual OPFS files on next load. Mitigated by the size verification in `get()` which catches corrupted/mismatched entries.

### MEDIUM: LRU eviction is O(N) per eviction

**Severity**: MEDIUM
**File**: `opfs-store.ts`, lines 114-128

Finding the LRU entry requires a full scan of the `index` Map to find the minimum `order`. For a cache with 10,000 entries, each eviction is O(10,000). A min-heap or sorted structure would give O(log N).

**Practical impact**: Low — evictions are infrequent and the async OPFS I/O dominates the cost.

### MEDIUM: `delete()` swallows all errors silently

**Severity**: MEDIUM
**File**: `opfs-store.ts`, line 177

Bare `catch {}` swallows all errors including permission failures and corrupted file handles. A debug-level log would help troubleshooting.

### LOW: Bucket distribution correctness

The `getBucket()` hash function (djb2 variant) produces good distribution across 256 buckets. However, keys with identical prefixes (common in zarr paths like `positions/0.0.0`, `positions/0.0.1`) may cluster slightly. Not a significant issue at scale.

### LOW: No validation of metadata consistency on load

`loadMetadata()` trusts `_cache_meta.json` blindly. If metadata references files that were deleted externally or have wrong sizes, the index becomes stale. The `get()` size verification partially mitigates this.

---

## Positive Observations

1. **256-bucket sharding** — Excellent choice to avoid OPFS per-directory file limits
2. **Debounced metadata saves** — Prevents write amplification during burst operations
3. **Content hash invalidation** — Proper cache invalidation mechanism
4. **Quota checking** — 10% safety margin prevents quota exceeded errors
5. **Base64 key encoding** — Filesystem-safe with special character replacement
