# Decompressed Chunk Cache — zarr.Array Proxy Wrapper

Single-file leaf that adds L0 caching to a `zarr.Array` via an ES6 `Proxy`.
The `DecompressedChunkCache` class itself lives one level up
(`../decompressed-chunk-cache.ts`); this folder hosts the wrapper that
splices it onto a real zarr array without touching zarrita's source.

## Overview

`wrapWithCache(array, cache, arrayPath, hooks?)` (hooks = `{ getProbe?,
getSignal?, getOrigin? }`; a deprecated `aliasOnMiss` is accepted and
ignored) returns a
`Proxy<zarr.Array>` that intercepts `getChunk()` (and adds `warmChunk()`):

0. **Per-update abort chokepoint** — when an optional `getSignal` accessor is
   supplied, the interceptor calls `getSignal()?.throwIfAborted()` BEFORE the
   L0 lookup. A superseded `updateView` (newer view-state queued) therefore
   bails on the hit, coalesced-pending, and miss paths alike — zarrita's own
   `throwIfAborted` only fires between chunks of a multi-chunk selection, so
   this is what makes warm-cache scrubs cancel. The accessor reads the owning
   loader's per-call `ActiveLoadContext` signal, so it never aborts a
   coalesced chunk another live caller awaits.
1. Build a key via `DecompressedChunkCache.makeKey(arrayPath, coords)`.
2. On L0 hit, return the cached `{ data, shape, stride }` immediately
   (~1μs, no Blosc decode).
3. On miss, run the original `getChunk()`, store the result's
   `ArrayBufferView` in the cache as is (copied only when it does not span its
   whole `ArrayBuffer`), and return it.
4. **Same-chunk decode coalescing, cache-wide** — the in-flight map lives on
   the `DecompressedChunkCache` (`getInflight`/`setInflight`/`deleteInflight`,
   keyed by the full L0 key), NOT on the proxy, so every proxy over one cache —
   the foreground loaders and the SlicePrefetcher's shadow loaders wrap the
   same arrays through different proxies — shares one decode per chunk.
   **Abort isolation:** the shared decode runs under the entry's OWN
   `AbortController`; each waiter races it against its own signal (the call's
   `options.signal`, else `getSignal()`), rejecting only itself on abort. The
   decode is cancelled only once EVERY waiter has aborted; it stays registered
   until it settles, so a caller arriving meanwhile takes its result if the
   cancellation came too late to stop it, and starts afresh if it was really
   cancelled (rejected with the abort reason, or an `AbortError`). A decode
   that is not fully abandoned completes and is cached. Only a
   still-registered entry commits, so `cache.clear()` also orphans in-flight
   decodes.
5. **Residency reporting** — when an optional `getProbe` accessor is
   supplied, every `getChunk()` calls `getProbe()?.record(hit)` against
   the currently-active `ResidencyProbe` (see `../residency-probe`). L0
   hits and coalesced waits count as hits (no fresh Blosc work); genuine
   misses count as misses. `getProbe` returning `null` (the default, or
   when no load is in flight) disables reporting, keeping prefetch
   traffic out of a demand load's signal.
6. **Perf counters** (`../../profiling/perf-counters.ts`) — `l0.hits`,
   `l0.misses`, `l0.coalesced`, `l0.cloneBytes`, and per completed miss
   decode `decode.count`, `decode.count.<origin>`, `decode.bytes`, and
   `decode.duplicates` (the same L0 cache instance + key decoded again within
   3 s; a module-level history bounded to 65,536 keys). The origin is the
   call signal's tag (`tagSignalOrigin` in `decode-origin.ts` — the shadow
   prefetcher, ladder lookahead, and `prefetchRangesIntoCache` tag their
   signals), else `hooks.getOrigin()`, else `'foreground'`.
7. **`warmChunk(coords, { signal?, origin? })`** (`warm-chunk.ts`) — the
   cache-warming entry point `prefetchRangesIntoCache` uses: L0 lookup + the
   same shared decode, but with the caller's own signal only (never
   `getSignal`), no residency-probe record, no output assembly, and origin
   `'prefetch'` by default. The standalone `warmChunk(array, …)` helper falls
   back to a bare `getChunk` for an unwrapped array.
8. **One read-only buffer per chunk** — the miss that decoded a chunk, the
   waiters that joined its decode and every later hit all receive the SAME
   buffer L0 holds; there is no defensive clone on any path (a miss-only
   clone protected one caller while every other shared the buffer anyway).
   zarrita `get()` keeps the contract by copying every chunk into its own
   output; a direct `getChunk()` caller must copy before writing. The
   `aliasOnMiss` hook that used to opt into this is now a no-op (the
   spatial-index loaders still pass it as `L0_ALIAS_ON_MISS`).

All other property access passes through unchanged. See
[`../README.md`](../README.md) (the "L0 Decompressed Chunk Cache" section)
for how L0 fits into the L0 → L1 → L2 → Remote hierarchy and the
read-only chunk contract every consumer of this wrapper must keep.

## File Structure

```
decompressed-chunk-cache/
├── cached-zarr-array.ts   # Proxy wrapper, shared-decode core, marker symbols, clone helper
├── decode-origin.ts      # AbortSignal -> decode-origin tags (perf counters)
└── warm-chunk.ts         # warmChunk: dependency-free cache-warming entry point
```

## API

| Export                                           | Purpose                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wrapWithCache(array, cache, arrayPath, hooks?)` | Wrap a `zarr.Array` with L0 caching. Idempotent — already-wrapped arrays are returned as-is. Optional `hooks.getProbe` reports hit/miss to the active `ResidencyProbe`; `hooks.getSignal` supplies the per-update `AbortSignal` checked at `getChunk` entry so superseded loads bail; `hooks.getOrigin` names a miss decode's origin for the perf counters. |
| `tagSignalOrigin(signal, origin)`                | Attribute miss decodes of reads carrying `signal` to `decode.count.<origin>` (re-exported from `decode-origin.ts`).                                                                                                                                                                                                                                         |
| `warmChunk(array, coords, options?)`             | Decode + cache one chunk with no probe record and no output assembly (proxy method, or a bare `getChunk` fallback for an unwrapped array). Re-exported from `warm-chunk.ts`.                                                                                                                                                                                |
| `isCachedArray(array)`                           | Detect the wrapper via a private `Symbol` marker.                                                                                                                                                                                                                                                                                                           |
| `unwrapCachedArray(array)`                       | Recover the original unwrapped array (or pass through if not wrapped).                                                                                                                                                                                                                                                                                      |
| `cloneArrayBufferView(view)` _(internal)_        | Clone a `TypedArray` or `DataView` to a fresh underlying buffer. Exported only for unit tests.                                                                                                                                                                                                                                                              |

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
- **Shared, read-only chunks.** The decoded `chunk.data` is stored as is —
  `cloneArrayBufferView` runs only for a view into a larger buffer — and every
  caller (miss, coalesced waiter, hit) gets that one view by reference, so
  downstream loaders must treat L0 chunks as read-only (see "L0 read-only
  chunk contract" in `../README.md`).
- **No double-wrap.** `wrapWithCache` checks `isCachedArray` first and
  returns the input unchanged when already wrapped.

## See Also

- [`../decompressed-chunk-cache.ts`](../decompressed-chunk-cache.ts) —
  the L0 cache class itself (LRU, key format, `makeKey`).
- [`../README.md`](../README.md) — full cache package overview,
  including the L0 read-only chunk contract and end-to-end flow.
