# luxar.encoding._encoders

**Internal mixin package** for the encoding layer. The mixins here implement the domain-specific encoding logic that `ArrayEncoder` (in `../encoder.py`) inherits directly or transitively. Application code uses `ArrayEncoder` rather than the mixin classes. `delta_codec.LuxarDelta` is not a mixin and is re-exported publicly from `luxar.encoding`.

## Purpose

Break the monolithic `ArrayEncoder` into focused, testable mixins, each owning one semantic-type family:

- **BaseEncoderMixin** — Shared quantization primitives and instance-attribute stubs
- **StructuralEncoderMixin** — Passthrough, broadcast, array_ref, LUT, index, and custom encodings
- **PerChannelEncoderMixin** — Per-column and scalar quantizers (coordinates, colors, bounded/positive scalars, geolog)
- **CholeskyEncoderMixin** — Cholesky split-pair policy with encode-time covariance certificate
- **delta_codec.py** — zarr v2 filter for columnar delta + zigzag pre-filtering

`ArrayEncoder` directly inherits `StructuralEncoderMixin`, `PerChannelEncoderMixin`, and `CholeskyEncoderMixin`; each of those inherits `BaseEncoderMixin`. Its `encode(...)` dispatch follows the priority order defined on the orchestrator.

## Module Ownership

| Module | Semantic Types Handled | Key Entry Points |
|--------|------------------------|------------------|
| **base.py** | (foundation) | `_is_uniform`, `_compute_quantization_bits`, `_quantize_normalized_clip`, `_quantize_per_column`, `_geolog_forward/inverse`, `_perchannel_log_forward/scales/roundtrip`, `_perchannel_geolog_scales`, `_validate_input`, `_write_float` |
| **structural.py** | INDEX, (all via structural transforms) | `_lut_plan` (two-tier LUT eligibility + payload), `_write_passthrough`, `_encode_broadcasted`, `_encode_array_ref`, `_encode_lut`, `_encode_index`, `_encode_custom` (+ dispatch handlers) |
| **perchannel.py** | COORDINATE, COLOR (SDR/HDR), BOUNDED_SCALAR, POSITIVE_SCALAR | `_encode_coordinate`, `_encode_color`, `_encode_bounded_scalar`, `_encode_positive_scalar`, `_encode_geolog_scalar`, `_encode_linear_perchannel`, `_encode_log_perchannel`, `_encode_signed_log_perchannel`, `_encode_geolog_perchannel` |
| **cholesky.py** | CHOLESKY_DIAG, CHOLESKY_OFFDIAG | `encode_cholesky_split`, `_cov_relf_p95`, `_sigma_from_split`, `_relf_p95`, `_encode_cholesky` (legacy CHOLESKY passthrough) |
| **delta_codec.py** | (zarr filter, not a semantic type) | `LuxarDelta` (numcodecs codec class), `probe_delta_filter` |

## Call Flow

**Priority order** (as defined in `ArrayEncoder.encode`):

1. **Scalar input**: `_scalar_to_array` → `_encode_broadcasted_scalar` (shape `(1,)` or `(1, d)` with `n_elements` metadata). Checked first: a non-`ndarray` input is converted and returns before the empty-array check.
2. **Empty array**: passthrough (shape `(0,)` or `(0, d)`, no encoding). Runs after the scalar branch but before the uniformity check (`_is_uniform` indexes `data[0]` and would raise on empty input).
3. **Uniform array**: `_is_uniform` → `_encode_broadcasted` (stores first row/element only)
4. **Deduplicate**: `_registry.check` → `_encode_array_ref` (if `deduplicate=True` and a byte-identical array was already written)
5. **LUT encoding**: `_lut_plan` → `_encode_lut` (if ≤65,536 unique values pass benefit rules; two index tiers: uint8 ≤256, uint16 257..65,536 row-mode colors only)
6. **Dtype encoding**: dispatch to `_encode_dtype` → semantic-type handlers (quantization based on mode + semantic type)

**Cholesky split-pair** bypasses the priority ladder: `encode_cholesky_split` is the direct public entry point (called by gsplat writers); it encodes both halves through the full priority ladder with a certified precision tier, then stamps the certificate in the attrs.

**Delta filter** is probe-gated at the write site (inside the per-channel and scalar encoders): `probe_delta_filter` compresses one representative chunk both plain and delta-filtered; the filter is applied only where it wins (≥ `DELTA_PROBE_MIN_GAIN` = 1.02× advantage). The viewer's zarrita pipeline auto-decodes it (registered as `numcodecs.luxar_delta_v1`).

## Key Invariants

1. **Instance attribute contract** (BaseEncoderMixin): `_registry`, `_broadcast_rtol`, `_broadcast_atol`, `_float16_allowed`, `_lut_json_max_bytes` are assigned by `ArrayEncoder.__init__` and read by mixins. The two `encode` / `_encode_dtype` stubs are declared on `BaseEncoderMixin` but implemented on `ArrayEncoder` so cross-mixin calls (`self.encode(...)`) type-check.

2. **Uniform detection** (`_is_uniform`): For 1D arrays, checks if all elements equal `data[0]`; for 2D, checks if all rows equal `data[0]`. Uses `np.all(data == data[0])` when tolerances are zero (exact equality, safe for all dtypes); uses `np.allclose` when non-zero tolerances are set (approximate, opt-in).

3. **Cholesky certificate** (`encode_cholesky_split`): AUTO mode round-trips a bounded sample (up to `COV_CERT_SAMPLE_MAX` = 262,144 evenly-spaced rows) through the exact quantization transform, rebuilds Σ = L·Lᵀ from both halves, and measures the p95 per-splat relative Frobenius error. u8 is kept when the error is ≤ `COV_CERT_RELF_P95_MAX` (0.05); otherwise AUTO escalates to u16, then (practically unreachable) float32. MEMORY is u8 unconditionally. The measured certificate is written into each array's own `encoding` attrs as provenance (`{"metric": "cov_relf_p95", "value", "threshold", "tier"}`) — decode never needs it. Both halves always share one tier.

4. **Per-channel quantization** (BaseEncoderMixin `_quantize_per_column`, `_quantize_perchannel_zero_level`): Each column is quantized over its own `[lo, hi]` to `2**bits` uniform levels. The `_perchannel_log_*` family anchors scales at each column's **nonzero** min/max and reserves code 0 for exact zeros (the rescale-first, zero-safe layout matching `geolog_scalar`); the reserved level makes exact zeros round-trip exactly (an axis-aligned splat keeps zero correlations) and lets the scales come from the nonzero entries only.

5. **LUT eligibility** (`_lut_plan`): Decides LUT encoding and builds the payload in ONE `np.unique` pass (the previous split design ran `unique` twice and could diverge). Two index tiers:
   - **uint8** (K ≤ 256): legacy rules preserved verbatim — row mode requires `N ≥ 2K` for uint8 colors; everything else requires `size ≥ 4K`.
   - **uint16** (257 ≤ K ≤ 65,536): ROW MODE (colors) ONLY. Scalar-mode u16 indices cost exactly what quantized scalars cost, so the LUT JSON would be pure overhead (measured ~2× store regression). Row mode genuinely wins (one 2-B index covers all channels vs ≥3 B/row for quantized color). The LUT values live as JSON in `.zattrs` and are DUPLICATED by consolidated `.zmetadata` (parsed at scene-open for every node), so element-count heuristics lie. Accept only when the doubled JSON costs at most half the raw byte savings over the cheapest quantized color (1 B/channel) AND stays under `lut_json_max_bytes` (default 512 KiB). Accepted uint16 LUTs are therefore always strictly smaller than even the lossy alternative — while being EXACT.
   - **INDEX arrays never LUT-encode** (either tier): the viewer's line segments loader reads them RAW with no encoding dispatch, so a LUT would silently corrupt connectivity — and the smallest-uint INDEX encoding is already within one byte of what LUT indices would cost.
   - **64-bit integer values beyond ±2^53** never LUT-encode (they would not survive the JSON round-trip).

6. **Delta filter wire format** (`delta_codec.py`): Columnar per-chunk modular delta + zigzag. Per chunk of `rows × cols` codes, per column, modular `2**bits` arithmetic:
   ```
   encode:  d  = (code - prev) mod 2^bits          # prev = 0 at chunk start
            s  = d >= 2^(bits-1) ? d - 2^bits : d  # signed interpretation
            zz = (s << 1) ^ (s >> (bits-1))        # zigzag -> uint
   decode:  s    = (zz >> 1) ^ -(zz & 1)
            code = (prev + s) mod 2^bits
   ```
   Layout is **columnar** (all column-0 residuals, then column-1, ...) — this is what unlocks the gain (interleaved row-major is weak on coordinates, negative on Cholesky codes). Chunks must hold whole rows (Luxar chunking never splits columns); the codec raises if chunk size is not a multiple of `cols`.

## Testing

The mixins are NOT unit-tested in isolation (they have no standalone API). The shared test suites in `encoding/tests/` exercise them through `ArrayEncoder`:

- **test_encoder.py** — Broadcasting, LUT, array_ref, semantic-type dispatch, mode selection
- **test_roundtrip_encoding.py** — Full encode→decode→re-encode roundtrips per semantic type
- **test_edge_cases.py** — Error paths, boundary conditions, unusual inputs
- **test_dynamic_range.py** — Dynamic-range-based dtype selection

The Cholesky certificate is exercised by gsplat end-to-end tests (`gsplats/io/tests/test_save_load.py`, e.g. `test_roundtrip_dims_and_modes`). The delta filter is verified by hand-computed byte vectors in both the Python codec (`encoding/tests/test_delta_codec.py`) and the TypeScript twin (`luxar-viewer/src/tests/unit/data/codecs/luxar-delta.test.ts`) — kept in 1:1 sync.

## See Also

- `../encoder.py` — `ArrayEncoder` orchestrator that inherits all mixins
- `../decoder.py` — `ArrayDecoder` that inverts these transforms
- `../README.md` — Encoding package overview
- `../../io/_compiler/dataset_writers/` — Callsites (positions, colors, scalars)
- `packages/luxar-viewer/src/data/codecs/luxar-delta.ts` — TypeScript delta filter twin
