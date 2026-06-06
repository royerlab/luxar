# Decompressed Chunk Cache — zarr.Array Proxy Wrapper

Single-file leaf that adds L0 caching to a `zarr.Array` via an ES6 `Proxy`.
The `DecompressedChunkCache` class itself lives one level up
(`../decompressed-chunk-cache.ts`); this folder hosts the wrapper that
splices it onto a real zarr array without touching zarrita's source.

## Overview

`wrapWithCache(array, cache, arrayPath, getProbe?)` returns a
`Proxy<zarr.Array>` that intercepts `getChunk()`:

1. Build a key via `DecompressedChunkCache.makeKey(arrayPath, coords)`.
2. On L0 hit, return the cached `{ data, shape, stride }` immediately
   (~1μs, no Blosc decode).
3. On miss, await the original `getChunk()`, clone the result's
   `ArrayBufferView`, store it in the cache, and return the original.
4. **Same-chunk decode coalescing** — a per-wrapper `Map<key, Promise>`
   ensures concurrent `getChunk()` calls for the same key share one
   underlying decompression instead of running Blosc twice.
5. **Residency reporting** — when an optional `getProbe` accessor is
   supplied, every `getChunk()` calls `getProbe()?.record(hit)` against
   the currently-active `ResidencyProbe` (see `../residency-probe`). L0
   hits and coalesced waits count as hits (no fresh Blosc work); genuine
   misses count as misses. `getProbe` returning `null` (the default, or
   when no load is in flight) disables reporting, keeping prefetch
   traffic out of a demand load's signal.

All other property access passes through unchanged. See
[`../README.md`](../README.md) (the "L0 Decompressed Chunk Cache" section)
for how L0 fits into the L0 → L1 → L2 → Remote hierarchy and the
read-only chunk contract that this wrapper upholds via cloning.

## File Structure

```
decompressed-chunk-cache/
└── cached-zarr-array.ts   # Proxy wrapper, marker symbols, and clone helper
```

## API

| Export                                              | Purpose                                                                                                                                                                    |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wrapWithCache(array, cache, arrayPath, getProbe?)` | Wrap a `zarr.Array` with L0 caching. Idempotent — already-wrapped arrays are returned as-is. Optional `getProbe` accessor reports hit/miss to the active `ResidencyProbe`. |
| `isCachedArray(array)`                              | Detect the wrapper via a private `Symbol` marker.                                                                                                                          |
| `unwrapCachedArray(array)`                          | Recover the original unwrapped array (or pass through if not wrapped).                                                                                                     |
| `cloneArrayBufferView(view)` _(internal)_           | Clone a `TypedArray` or `DataView` to a fresh underlying buffer. Exported only for unit tests.                                                                             |

## Invariants

- **Direct access for zarrita private-field getters.** `attrs`, `shape`,
  `dtype`, `chunks`, `order`, `fill_value`/`fillValue`, `dimensionNames`,
  `compressor`, `filters`, `codec`, `codecs` are read directly off
  `target` — bypassing `Reflect.get` — because zarrita's getters touch
  `#metadata` and the property descriptor coming from the proxy breaks
  the private-field lookup.
- **`Reflect.get(target, prop, target)` for everything else.** The
  receiver MUST be `target`, not the proxy; otherwise zarrita methods
  hit `Cannot read private member #e` on `#store`/`#e` access.
- **Clone on cache insert.** The wrapper clones `chunk.data` via
  `cloneArrayBufferView` before storing so a caller mutating their view
  cannot corrupt the cached copy. Subsequent L0 hits return the cached
  view by reference — downstream loaders must treat L0 chunks as
  read-only (see "L0 read-only chunk contract" in `../README.md`).
- **No double-wrap.** `wrapWithCache` checks `isCachedArray` first and
  returns the input unchanged when already wrapped.

## See Also

- [`../decompressed-chunk-cache.ts`](../decompressed-chunk-cache.ts) —
  the L0 cache class itself (LRU, key format, `makeKey`).
- [`../README.md`](../README.md) — full cache package overview,
  including the L0 read-only chunk contract and end-to-end flow.
