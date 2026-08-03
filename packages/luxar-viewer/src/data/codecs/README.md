# data/codecs — Zarr codec extensions

**Purpose**: Luxar-specific zarrita `array_to_array` codecs that operate on entire chunks before sub-chunk slicing.

This directory holds codec implementations registered with the zarrita codec registry at module scope. Each codec is a **stateless** per-chunk transform applied inside `zarrita.getChunk()` (before the range-loader slices sub-chunk element ranges), so the viewer's per-element decode kernels (WASM/TS) and range-based loading pipeline remain untouched.

## Modules

| File             | Codec ID         | Status       | Description                                                                                                                     |
| ---------------- | ---------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `luxar-delta.ts` | `luxar_delta_v1` | **Internal** | Columnar per-chunk delta + zigzag filter for quantized uint8/uint16 codes. Registered by `../zarr.ts` module-scope initializer. |

## Delta Codec (`luxar_delta_v1`)

### Wire Format

Per-chunk modular delta encoding with zigzag mapping to unsigned values, laid out **column-major** within each chunk (all column-0 residuals, then column-1, …):

```
encode:  d  = (code - prev) mod 2^bits          // prev = 0 at chunk start
         s  = d >= 2^(bits-1) ? d - 2^bits : d  // signed interpretation
         zz = (s << 1) ^ (s >> (bits-1))        // zigzag -> uint
decode:  s    = (zz >>> 1) ^ -(zz & 1)
         code = (prev + s) mod 2^bits
```

Spatial ordering (Hilbert/Morton) makes consecutive codes a smooth ramp; the residuals compress 12–16% better whole-store under Blosc/zstd. See the 2026-07 compression-transfer campaign / SOG comparison for measured gains.

### Format Version

The current **standalone gsplat format is v3.3** (`GSPLATS_FORMAT_VERSION` in `src/types/format-contract.ts`; codegen'd from `format-contract/contract.yaml`). Supported versions: 3.0, 3.1, 3.2, 3.3.

### Optional & Lossless

The filter is **optional**: the Python writer (`packages/luxar/src/luxar/encoding/_encoders/delta_codec.py`) runs an encode-time probe that compresses a representative chunk both plain and delta-filtered, then enables the filter only when the plain size exceeds the delta size by `DELTA_PROBE_MIN_GAIN` (1.02x). Arrays where delta is marginal or negative are stored exactly as before (no filter entry in `.zarray`).

The filter is **lossless**: a perfect round-trip (byte-exact reconstruction). The cumulative modular arithmetic and zigzag mapping preserve all bits.

### Python ↔ TypeScript Parity

The Python twin lives in `packages/luxar/src/luxar/encoding/_encoders/delta_codec.py` (registered as `numcodecs.luxar_delta_v1`). Both implementations share **identical** hand-computed wire-format test vectors (locked in `luxar/encoding/tests/test_delta_codec.py::TestWireFormat` and `src/tests/unit/data/codecs/luxar-delta.test.ts`) — if either side changes bytes, both suites fail.

### Registration

The codec is registered at module scope by `../zarr.ts` (the sole zarrita import boundary):

```typescript
import { LuxarDeltaCodec } from './codecs/luxar-delta';

// codecRegistry is zarr.ts's own re-export of zarrita.registry (not an
// import from 'zarrita'):
//   export const codecRegistry = zarrita.registry;
// The registry key is the zarrita codec name `numcodecs.luxar_delta_v1`
// (a v2 `.zarray` filter `{id: "luxar_delta_v1"}` maps to it), and the
// value is a lazy thunk resolving to the codec class:
codecRegistry.set('numcodecs.luxar_delta_v1', () => Promise.resolve(LuxarDeltaCodec));
```

Every context that opens zarr arrays (main thread, data workers) imports `../zarr.ts`, so the codec is provisioned globally.

### Lifecycle Invariants

- The codec operates on **whole chunks** (zarrita calls `decode(chunk)` after fetch, before sub-chunk slicing).
- Chunk size must be a multiple of `cols` — the writer never splits columns, and the codec raises if a chunk is ragged.
- `fromConfig(config, meta)` validates `cols` against the chunk shape (fail-loud on corruption).
- The encoded chunk layout is **column-major** (all column-0 residuals, then column-1, …); row-major would be weak on coordinates and negative on Cholesky codes.

## See Also

- [`../zarr.ts`](../zarr.ts) — Luxar zarrita facade: the sole production boundary that imports zarrita directly; registers all codecs.
- [`../array-decoder/`](../array-decoder/README.md) — Per-element decode kernels (quantization, LUT, broadcasting) that run **after** chunk-level codecs.
- [`../../tests/unit/data/codecs/`](../../tests/unit/data/codecs/) — Unit tests for codec wire format and round-trip correctness.
- `packages/luxar/src/luxar/encoding/_encoders/delta_codec.py` — Python twin (write side); cross-language test vectors.
- `docs/specs/GSPLATS_ZARR_FORMAT.md` — Standalone gsplat format spec (v3.3).
