# Array Decoder

> Decoders for the Python `luxar.encoding` array formats consumed by the viewer.

## Overview

Arrays in a Luxar zarr archive may be stored in one of several encoded forms
to reduce on-disk size and avoid duplication. This folder owns the
TypeScript counterpart of the Python encoder: it reads `encoding` metadata
from `.zattrs`, dispatches to the correct decoder, and returns a plain
`Float32Array` to the rest of the viewer.

The decoder MUST decode every format written by the Python `ArrayEncoder`
or scenes will render incorrectly. The canonical reference for what each
encoding means lives in `packages/luxar/src/luxar/encoding/README.md`.

## File Structure

```
array-decoder/
├── decoder.ts            # ArrayDecoder class — priority-dispatch body
├── ref-registry.ts       # ArrayRefRegistry — hash → Float32Array cache for array_ref
├── load-and-decode.ts    # loadAndDecodeOptionalArray helper for sibling attribute arrays
└── types.ts              # ArrayMetadata + EncodingMetadata schema (.zattrs shape)
```

`decoder.ts` re-exports `ArrayRefRegistry`, the metadata types, and the
`loadAndDecodeOptionalArray` helper so consumers can import everything
from one entry point.

## Priority-Dispatch Order

`ArrayDecoder.decode()` checks encoding modes in a fixed order — the
order MUST match the Python spec or behavior diverges:

1. **broadcasted** — single value replicated to `n_elements` × `k`. Reads
   one row from zarr, replicates in place, registers under `enc.hash` for
   later `array_ref` reuse.
2. **array_ref** — delegates to `decodeArrayRef()`: hash-cache lookup
   first, otherwise resolves `enc.target` against `zarrRootLoc`, opens
   the target zarr array, and recursively decodes it (the target may
   itself be LUT- or quantization-encoded). Result is cached by hash.
3. **LUT** (`lut_uint8`, `lut_uint16`) — indices into `enc.lut`. Two
   modes: `row` (one index → `k` values) and `scalar` (one index → 1
   value). `k` comes from `enc.original_shape[1]`.
4. **log_scalar** (`log_scalar_uint8`, `log_scalar_uint16`) — checked
   BEFORE generic quantization because the name contains `uint`. Decodes
   via `expm1(normalized × max_log)`. Used for radii and other
   wide-dynamic-range positive scalars.
5. **quantized** (`rgb_uint8`, `rgb_uint16`, `bounded_scalar_uint8`,
   `bounded_scalar_uint16`) — linear dequantization to `[min, max]`.
   Bounds resolved from `enc.bounds`, then `enc.min`/`enc.max`, then
   inferred (only `rgb_*` is inferrable → `[0, 1]`).
6. **direct** — `undefined` / `'none'` / `float16` / `float32` /
   `uint8` / `uint16` / `uint32` / `uint64` — raw zarr buffer converted
   to `Float32Array`. Registered under `enc.hash` if present.

Empty zarr arrays (`shape` contains 0) short-circuit to
`new Float32Array(0)` — zarrita's `get()` cannot materialize them in
Node. `array_ref` is handled before this guard because its physical
zarr shape is also empty.

## Encoding Metadata Schema

`types.ts` mirrors what the Python encoder writes under the
`encoding` key of `.zattrs`:

| Field                               | Used by           | Notes                                                                    |
| ----------------------------------- | ----------------- | ------------------------------------------------------------------------ |
| `name`                              | dispatch          | One of the encoding modes above; missing → direct.                       |
| `n_elements`                        | broadcasted       | Logical broadcast count. Required for `broadcasted`.                     |
| `lut`, `lut_mode`, `original_shape` | LUT               | `lut_mode` defaults to `'row'`. `k` from `original_shape[1]`.            |
| `original_dtype`                    | LUT + quantized   | Required by validation; consumers restore native dtype after decode.     |
| `bounds` / `min` / `max`            | quantized         | Linear quantization range.                                               |
| `max_log`                           | log_scalar        | Inverse-log1p scale factor; must be finite and `> 0`.                    |
| `target`, `hash`                    | array_ref + dedup | `target` resolved via `zarrRootLoc.resolve()`; `hash` keys the registry. |

`ArrayDecoder.validateEncodingMetadata()` runs ahead of dispatch and
rejects metadata that's structurally wrong (missing `name`, `target`
without `array_ref`, bounds on a non-quantized encoding, mismatched
`min`/`max`, unknown encoding name, …) with a clear `[ArrayDecoder]`
error.

## ArrayRefRegistry

`ref-registry.ts` is a small `Map<string, Float32Array>` wrapper used to
deduplicate identical buffers across the scene. `ArrayDecoder` registers
into it from both the broadcasted and direct paths, and reads from it
in `decodeArrayRef()` before touching zarr. The registry lives in its
own module so consumers that only need the type (SceneLoader,
`loaders/base-types.ts`, `scene-loader/loaders/loader-factory.ts`, the three
spatial-index loaders, and `RangeLoader`) don't import the full
~970-line decoder body.

## Range-Loader Hooks

The `loaders/spatial-query/range-loader.ts` path needs to decode a _slice_ of a
quantized or LUT array without loading the whole thing. Three static /
instance helpers on `ArrayDecoder` support that:

- `getLUTMetadata(attrs)` — extract `{lut, lutMode, k}` for slice-time use.
- `getQuantizationMetadata(attrs, zarrDtype)` — extract
  `{bounds, dtype, isLogSpace}`; `zarrDtype` is the actual array dtype
  (Python writes the quantized dtype on the zarr array, not in
  `attrs.dtype`).
- `decodeLUTIndices(indices, lutMetadata)` and
  `dequantizeRange(quantizedData, quantMetadata)` — decode the slice
  with the metadata from above.

Classification helpers (`isEncoded`, `isLUTEncoded`, `isBroadcasted`,
`isArrayRef`, `isDirectEncodingName`, `isLUTEncodingName`,
`isLogScalarEncodingName`, `isQuantizedEncoding`, `isQuantizedEncodingName`,
`isKnownEncodingName`, `getEncodingMode`) let callers choose the right
loading strategy without parsing `enc.name` themselves.

## Optional-Array Helper

`loadAndDecodeOptionalArray(location, arrayName, decoder, expected?)`
opens a sibling zarr array (e.g. `colors`, `radii`, `sharpness`), reads
its attrs, runs `decoder.decode()`, and returns the float buffer.
Missing arrays surface as `null` — the open error is swallowed so
optional attributes are truly optional. Used by loader internals; not
part of the public `data/` API.

## See Also

- [../loaders/README.md](../loaders/README.md) —
  `spatial-query/range-loader.ts` consumes the range-decode helpers above.
- `packages/luxar/src/luxar/encoding/README.md` — Python encoder spec
  that this module mirrors.
