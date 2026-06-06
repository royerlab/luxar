# Range Loader Encodings

Per-encoding loader bodies for `RangeLoader`. The parent
`../range-loader.ts` is a thin dispatcher + context builder; each encoding's
actual range-loading logic lives in one file here, keeping the dispatcher
small and each encoding independently testable.

## Overview

`RangeLoader.loadRanges()` is the single entry point shared by the Points,
Lines, and GSplats spatial-index loaders. It detects the encoding written by
the Python `luxar.encoding` layer, then delegates to the matching `load*`
function in this folder. Every loader writes its dequantized/decoded result
into a caller-supplied `Float32Array` output buffer and reports how many
elements it wrote.

CPU-heavy decodes (broadcast replication, LUT lookup, dequantization) are
offloaded to the worker pool when `config.dataLoading.performance.useWebWorkers`
is set and the element count exceeds `workerThreshold` (default 1000), with a
main-thread fallback through `ArrayDecoder` on worker failure. `WorkerAbortError`
is re-thrown rather than swallowed so cancelled loads propagate cleanly.

## File Structure

```
range-loader/
├── encoding-types.ts    # Shared types, config defaults, slice/convert helpers
├── detect-encoding.ts   # ArrayMetadata → EncodingType (priority-ordered)
├── direct.ts            # loadDirect    — raw values, dtype → Float32
├── quantized.ts         # loadQuantized — uint8/uint16 → float (linear or log)
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
| `direct`      | `loadDirect`      | raw values in any numeric dtype          | Slice the requested ranges, convert to `Float32Array`.         |

`direct` is also the fallback when an array has no `encoding` metadata.

## Loading model

Every loader takes a `LoadRange[]` over the first axis (`{ start, end }`) and an
output `Float32Array`. `firstAxisRangeSlice` (in `encoding-types.ts`) builds the
zarr slice spec — a slice on axis 0 and full slices on the trailing axes — so a
range over an `[N, D]` array fetches `[start:end, :]`. Each range is fetched
through `zarr.get()`, decoded, written at the running `destOffset`, and the final
offset is returned as the element count.

`numericArrayToFloat32` normalises any zarr dtype to `Float32Array`, including
`BigUint64Array`/`BigInt64Array` (converted element-by-element via `Number()`).

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

`EncodingType` is the union `'broadcasted' | 'quantized' | 'lut' | 'array_ref' | 'direct'`.

## See Also

- `../range-loader.ts` — the dispatcher that builds the per-encoding `Ctx`
  objects and calls these loaders.
- `../../array-decoder/` — `ArrayDecoder` / `ArrayRefRegistry`; the main-thread
  decode fallback and metadata helpers (`getLUTMetadata`,
  `getQuantizationMetadata`, `isArrayRef`).
- `../../../../workers/worker-pool.ts` — worker pool used for offloaded decodes.
- `../README.md` — the unified loader infrastructure overview.
