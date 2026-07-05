# Worker Decode Entry Points

> WASM-accelerated decoders that turn raw zarr chunk bytes into `Float32Array`
> buffers inside the data worker.

## Overview

The main thread fetches encoded zarr buffers, then hands them to a
`DataWorker` for off-thread decoding via these entry points. Each function
calls into the compiled WASM kernel that mirrors the corresponding Python
`luxar.encoding` format, then ships the result back to the main thread with
`Comlink.transfer` so the underlying `ArrayBuffer` moves zero-copy.

These are the worker-side counterparts of `src/data/array-decoder/`. The
main-thread `ArrayDecoder` owns dispatch on encoding metadata and may run
in-process for small arrays; for large arrays it forwards the raw bytes
to one of these functions so the work happens on a worker. The two paths
MUST stay numerically identical or scenes render incorrectly — the
canonical encoding spec lives in
`packages/luxar/src/luxar/encoding/README.md`.

## File Structure

```
decode/
├── quantized.ts     — Linear dequantization: uint8/uint16 → float32 in [min, max]
├── log-scalar.ts    — Inverse-log1p dequantization for wide-dynamic-range positives
├── geolog-scalar.ts — Min/max-anchored geometric-log dequantization (reserved zero level)
├── lut.ts           — Lookup-table decode in 'row' (k values per index) or 'scalar' mode
└── broadcasted.ts   — Replicate a single k-vector to numPoints × k
```

Each entry point takes a shared `state: WasmCtx` (defined in
`../state.ts`) and a `params` object, calls `requireWasm(ctx)` to assert
the WASM module is initialized, validates the JS-side inputs, invokes
the matching Rust kernel, and `transfer`s the resulting `Float32Array`.

## Encoding Correspondence

| File             | Python encoding name(s)                                                    | WASM kernel(s)                                  |
| ---------------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `quantized.ts`   | `rgb_uint8`, `rgb_uint16`, `bounded_scalar_uint8`, `bounded_scalar_uint16` | `decode_quantized_u8`, `decode_quantized_u16`   |
| `log-scalar.ts`  | `log_scalar_uint8`, `log_scalar_uint16`                                    | `decode_log_scalar_u8`, `decode_log_scalar_u16` |
| `geolog-scalar.ts` | `geolog_scalar_uint8`, `geolog_scalar_uint16`                            | `decode_geolog_scalar_u8`, `decode_geolog_scalar_u16` |
| `lut.ts`         | `lut_uint8`, `lut_uint16` (modes: `row`, `scalar`)                         | `decode_lut_{scalar,row}_{u8,u16}`              |
| `broadcasted.ts` | `broadcasted`                                                              | `decode_broadcasted`                            |

`array_ref` and direct (`float32`, `float16`, raw `uint*`) encodings are
handled entirely on the main thread by `ArrayDecoder`; they never reach
the worker.

## Validation Contract

`quantized`, `log-scalar`, and `lut` route every JS-side precondition
through `validateDecodeArgs` (in `../validation.ts`) — finite scalars,
positive-integer `k`, non-empty + minimum-length LUT, and `bounds[1] >
bounds[0]`. `lut.ts` additionally:

- rejects `lutMode` values other than `'row'` / `'scalar'`,
- requires `lut.length % k === 0` in row mode,
- scans every index against the computed entry count before the WASM
  call — Rust kernels index `lut[indices[i]]` directly with no bounds
  check, so an out-of-range index would otherwise trap inside WASM.

`broadcasted.ts` runs its own inline checks (non-negative integer
`numPoints`, positive integer `elementsPerPoint`, `value.length ≥
elementsPerPoint`) and skips `validateDecodeArgs` because it has no
quantized input buffer.

## Output Transfer

Every function ends with `return transfer(result, [result.buffer])`.
This hands the `Float32Array`'s backing buffer to the main thread with
no copy; the worker's view becomes detached after the call. Callers on
the main thread receive a fresh, owned `Float32Array` ready to upload
to a WebGL buffer or hand to a geometry builder.

## See Also

- [`../../README.md`](../../README.md) — worker pool architecture, RPC
  surface, and how decode requests are routed.
- [`../validation.ts`](../validation.ts) — `validateDecodeArgs` and the
  rationale for guarding the JS→WASM boundary.
- [`../../../data/array-decoder/README.md`](../../../data/array-decoder/README.md)
  — main-thread `ArrayDecoder` that decides whether to decode in-process
  or forward to one of these worker entry points.
- `packages/luxar/src/luxar/encoding/README.md` — Python encoder spec
  these decoders mirror.
