# Range Loader Encodings

Per-encoding loader bodies for `RangeLoader`. The parent
`../range-loader.ts` is a thin dispatcher + context builder; each encoding's
actual range-loading logic lives in one file here, keeping the dispatcher
small and each encoding independently testable.

## Overview

`RangeLoader` is the single entry point shared by the Points, Lines, and
GSplats spatial-index loaders — **all** chunk reads for every geometry type go
through it (plus the L0 `wrapWithCache` proxy), so there is one place that calls
`zarr.get()` and one place the per-update abort signal lives:

- **`loadRanges()` / `loadRangesResolvingRef()`** detect the encoding written by
  the Python `luxar.encoding` layer and delegate to the matching `load*`
  function below. Encoded decoders (lut/quantized/perchannel/broadcasted)
  always produce a `Float32Array`.
- **`loadDirectTyped()`** is the direct-read entry for callers that allocate a
  natively-typed output buffer and branch on encoding _before_ dispatch — Points
  non-color attributes, the direct/`rgb_*` color path (`color-loader.ts`), and
  the Lines `segments` connectivity array. It wraps `loadDirect`, which copies
  **preserving** the output buffer's dtype (see `copyDirectChunk`).

Both report how many elements were written.

**Ranges load in parallel.** Each per-encoding loader (direct/lut/quantized)
precomputes every range's destination offset up front (`rangeDestOffsets` —
element counts are deterministic per range) and issues all `zarr.get()` calls
concurrently via `Promise.all`; ranges write into disjoint output spans, so
resolution order doesn't matter. Network concurrency stays bounded by the
global fetch gate (`utils/fetch-concurrency.ts`, 64-wide) and decode
concurrency by the worker pool. A decoded chunk whose length mismatches the
precomputed span is clamped with a warning (`clampRangeData`) — over-long data
is truncated (never corrupts a neighbour's span), short data leaves the tail
of its span zeroed, mirroring the historical graceful-fallback behavior.

CPU-heavy decodes (broadcast replication, LUT lookup, dequantization) are
offloaded to the worker pool when `config.dataLoading.performance.useWebWorkers`
is set and the element count exceeds `workerThreshold` (default 1000), with a
main-thread fallback through `ArrayDecoder` on worker failure. `WorkerAbortError`
is re-thrown rather than swallowed so cancelled loads propagate cleanly.

## File Structure

```
range-loader/
├── encoding-types.ts    # Shared types, config defaults, slice helper, copyDirectChunk
├── detect-encoding.ts   # ArrayMetadata → EncodingType (priority-ordered)
├── direct.ts            # loadDirect    — raw values, dtype-preserving copy
├── quantized.ts         # loadQuantized — uint8/uint16 → float (linear or log)
├── perchannel.ts        # loadPerChannel — per-column (col_lo/col_hi) dequant → float32
├── lut.ts               # loadLUT       — index → palette row/scalar lookup
├── broadcasted.ts       # loadBroadcasted — one value replicated to N items
├── array-ref.ts         # loadArrayRef  — unresolved ref guard (throws)
├── ref-resolution.ts    # resolveArrayRef — opens an array_ref target array
└── shared-instance.ts   # getSharedRangeLoader / getSharedRefRegistry singletons
```

## Encodings

Encoding detection follows the same priority order as the Python encoder
(`detect-encoding.ts`):

| Encoding      | Loader            | Stored form                              | Decode                                                         |
| ------------- | ----------------- | ---------------------------------------- | -------------------------------------------------------------- |
| `broadcasted` | `loadBroadcasted` | single value + repeat count              | Replicate one value across all items (`elementsPerItem` wide). |
| `array_ref`   | `loadArrayRef`    | reference to another array               | Must be pre-resolved upstream — reaching the loader throws.    |
| `lut`         | `loadLUT`         | small `uint8`/`uint16` indices + palette | Map each index through the LUT (`row` or `scalar` mode).       |
| `quantized`   | `loadQuantized`   | `uint8`/`uint16` quantized values        | Dequantize via bounds (linear) or `maxLog` (log-space).        |
| `perchannel`  | `loadPerChannel`  | `*_perchannel_*` uint8/uint16 levels     | Per-column dequant via `col_lo`/`col_hi` (log / signed-log / linear / geolog) to float32. |
| `direct`      | `loadDirect`      | raw values in any numeric dtype          | Slice the requested ranges, copy preserving the output dtype.  |

`direct` is also the fallback when an array has no `encoding` metadata.

## Loading model

Encoded loaders take a `LoadRange[]` over the first axis (`{ start, end }`) and a
`Float32Array` output (decoding always produces floats). The direct reader
(`loadDirect` / `loadDirectTyped`) takes a `DirectOutputBuffer` —
`Float32Array | Float16Array | Uint8Array | Uint16Array | Uint32Array` — so the
caller's allocated dtype is preserved. `firstAxisRangeSlice` (in
`encoding-types.ts`) builds the zarr slice spec — a slice on axis 0 and full
slices on the trailing axes — so a range over an `[N, D]` array fetches
`[start:end, :]`. Each range is fetched through `zarr.get()` (with the abort
signal), written at the running `destOffset`, and the final offset is returned.

`copyDirectChunk` performs the dtype-preserving copy: a same-kind source is a
zero-conversion `TypedArray.set`; a different numeric kind is converted by `set`
(e.g. `Float32` output from a `Uint8` source — and crucially the reverse,
keeping `uint8` colors as bytes for THREE.js 0–255→0–1 normalization); and
`BigInt64Array`/`BigUint64Array` sources are widened element-by-element via
`Number()` (a plain `set` of a BigInt array into a numeric buffer would throw).

## Array refs

`array_ref` encodings point one array at another (the encoder deduplicates
identical arrays). They must be resolved **before** reaching `RangeLoader`:

- `resolveArrayRef` (called by the spatial-index loaders) opens the target via
  `zarr.root(store).resolve(target)`, reads its attrs, and derives
  `elementsPerItem` from the target shape (product of the trailing axes, or 1 for
  1-D). Returns `null` when no resolution is needed.
- `loadArrayRef` is a guard: if an unresolved `array_ref` ever reaches the
  dispatcher it throws with the offending `target`/`hash`, signalling a code path
  that bypassed resolution.

## Shared instance

`shared-instance.ts` provides process-wide singletons so the geometry loaders
share one `RangeLoader` and one `ArrayRefRegistry` (the registry backs
`array_ref` deduplication):

```typescript
import { getSharedRangeLoader, getSharedRefRegistry } from './range-loader/shared-instance';

const loader = getSharedRangeLoader(); // lazily constructs RangeLoader + registry
const registry = getSharedRefRegistry();
```

The optional `registry` argument to `getSharedRangeLoader` only takes effect on
the first call. `resetSharedRangeLoader()` clears both singletons (test-only).

## Configuration

`RangeLoaderConfig` (`encoding-types.ts`) is small:

| Field             | Default                        | Meaning                                    |
| ----------------- | ------------------------------ | ------------------------------------------ |
| `workerThreshold` | `1000`                         | Min elements before a decode uses workers. |
| `logModule`       | `Modules.SPATIAL_INDEX_LOADER` | Log module tag for verbose output.         |

`EncodingType` is the union `'broadcasted' | 'quantized' | 'lut' | 'array_ref' | 'perchannel' | 'direct'`.

## See Also

- `../range-loader.ts` — the dispatcher that builds the per-encoding `Ctx`
  objects and calls these loaders.
- `../../array-decoder/` — `ArrayDecoder` / `ArrayRefRegistry`; the main-thread
  decode fallback and metadata helpers (`getLUTMetadata`,
  `getQuantizationMetadata`, `isArrayRef`).
- `../../../../workers/worker-pool.ts` — worker pool used for offloaded decodes.
- `../README.md` — the unified loader infrastructure overview.
