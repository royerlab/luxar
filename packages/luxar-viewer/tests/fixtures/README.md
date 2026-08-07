# Viewer Test Fixtures

Python-generated zarr datasets used by the viewer's unit tests (Vitest) and
E2E specs (Playwright) to verify the Python encoder ↔ TypeScript decoder
contract end-to-end.

The fixtures themselves are **not** tracked in git — only the two generator
scripts are. Every `test_*.zarr` directory in this folder and the
`roundtrip_expectations.json` snapshot beside them are produced on demand by
running the generators below. Both paths are explicitly listed in the repo
`.gitignore`.

## Generators

| File | Purpose |
|------|---------|
| `generate_test_data.py` | Single source of truth for the fixture set. Each fixture corresponds to one `generate_*` function and writes a `FIXTURES_DIR / "test_*.zarr"` archive using the real Python encoder (`luxar.LuxarZarrCompiler` / `luxar.encoding.ArrayEncoder`). Compression is disabled (`compressor=None`) and `float16_allowed=False` so the output is consumable from Node.js without blosc/numcodecs WASM bindings. |
| `generate_expectations.py` | Walks every `test_*.zarr` directory, decodes each numeric array with Python's `ArrayDecoder`, and writes `roundtrip_expectations.json` — flat-array shapes, SHA-256 hashes, sample values, stats, and representative first-axis range slices. The Vitest contract tests cross-check the TypeScript `ArrayDecoder` against this snapshot in pure Node (no browser, no GPU). |

The fixture list is parsed at test-startup time from the `FIXTURE_NAMES`
declaration at the top of `generate_test_data.py` — see
`src/tests/global-setup.ts`. Adding or renaming a fixture in the Python
script is sufficient, provided `FIXTURE_NAMES` is updated alongside the
new `generate_*()` function; no separate TypeScript manifest needs
updating.

## Generating fixtures

```bash
# Preferred — from packages/luxar-viewer/:
pnpm test:generate-fixtures       # runs both scripts in order
pnpm test:with-fixtures           # generate then run unit tests
pnpm test                         # global-setup regenerates missing/stale ones automatically

# Equivalent direct invocations (from repo root):
hatch run python packages/luxar-viewer/tests/fixtures/generate_test_data.py
hatch run python packages/luxar-viewer/tests/fixtures/generate_expectations.py
```

`src/tests/global-setup.ts` runs once before the Vitest suite, detects
missing `test_*.zarr` archives or an out-of-date `roundtrip_expectations.json`
(checked by mtime against the generators and fixtures), and re-runs the
relevant generator. The expectations file is regenerated whenever any
fixture or either generator script has changed.

## Fixture matrix

Each entry below corresponds to one `generate_*` function in
`generate_test_data.py` and exercises a specific encoding/scene-graph
behavior. Hover-tooltip and `test-fixtures-rendering.spec.ts` E2E specs
load several of these directly via `?src=...`; the contract tests in
`src/tests/unit/data/array-decoder/` consume them through Python-derived
expectations.

| Fixture | Exercises |
|---------|-----------|
| `test_broadcasting.luxar.zarr` | Broadcasted (uniform) per-point values — `(1, 3)` color and `(1,)` radius expand to N points. |
| `test_lut.luxar.zarr` | LUT encoding for color arrays with ≤256 unique values (10 distinct colors over 1000 points). |
| `test_quantization.luxar.zarr` | uint8 quantization of bounded scalar attributes (radii in `[0.1, 2.0]`, colors in `[0, 1]`). |
| `test_array_refs.luxar.zarr` | Array-reference deduplication — two groups sharing identical positions/colors collapse to one stored array. |
| `test_array_ref_broadcasting.luxar.zarr` | Mix of `array_ref` positions with scalar/broadcast attributes in the same scene. |
| `test_encoding_edge_cases.luxar.zarr` | Raw `ArrayEncoder` outputs covering edge cases that bypass scene-level validation. |
| `test_encoding_contract_matrix.luxar.zarr` | Declarative cross-product of encodings × dtypes × shapes × semantic types — the broadest single fixture, used to drive the contract-matrix tests. |
| `test_mixed.luxar.zarr` | Multiple encoding modes (raw, quantized, broadcast, LUT) coexisting within a single scene. |
| `test_4d.luxar.zarr` | 4D data with a time dimension — exercises nD slicing in the viewer and the `extend_to_all` path. |
| `test_hierarchical_transforms.luxar.zarr` | Nested groups with per-node 4x4 transforms — verifies composition order and the row/column-major guard. |
| `test_integer_colors.luxar.zarr` | Direct uint8 / uint16 SDR color arrays (no quantization round-trip). |
| `test_hdr_colors.luxar.zarr` | HDR colors with values > 1.0 — verifies float32 color preservation through the pipeline. |
| `test_log_scalar.luxar.zarr` | Wide-dynamic-range radii encoded as log-scalar uint8/uint16. |
| `test_4d_scalar_lut.luxar.zarr` | 4D positions combined with scalar LUT encoding. |
| `test_uint16_quantization.luxar.zarr` | uint16 quantization for attributes whose dynamic range exceeds uint8 precision. |
| `test_sharpness_range.luxar.zarr` | 32 points sampling the full normalized `[0, 1]` sharpness knob range — verifies decoded sharpness handling. |
| `test_nd_transforms.luxar.zarr` | `nd_transform` metadata on non-displayed dimensions — exercises the viewer's inverse-query path. |
| `test_lines.luxar.zarr` | Lines geometry (vertices, widths, optional colors and segments). |
| `test_gsplats.luxar.zarr` | GSplats (Gaussian Splats) geometry (centers, amplitudes, Cholesky factors, colors). |
| `test_labelled_points.luxar.zarr` | Small labelled-points dataset used by the `hover-tooltip.spec.ts` E2E spec. |
| `test_line_joins.luxar.zarr` | Five polyline-joint cases in separate world-Y bands (smooth curve, 90° zigzag, thin and thick straights, nine-ray indexed hub) under a photometry-grade pinned viewer config. Acceptance fixture for `line-join-artifact.spec.ts` (issues #780 / #785 / #790). |

## Files

```
fixtures/
├── generate_test_data.py        # Fixture generators (one function per test_*.zarr)
├── generate_expectations.py     # Builds roundtrip_expectations.json from the fixtures
├── roundtrip_expectations.json  # Generated; gitignored. Python decoder snapshot.
└── test_*.zarr/                 # Generated; gitignored. Created on demand.
```

## See Also

- `../../src/tests/global-setup.ts` — Vitest global setup that parses the
  fixture list and regenerates missing/stale outputs.
- `../../src/tests/unit/data/array-decoder/` — Vitest contract tests that
  consume `roundtrip_expectations.json`.
- `../../src/tests/e2e/test-fixtures-rendering.spec.ts` and
  `hover-tooltip.spec.ts` — Playwright specs that load the fixtures in
  the browser.
- `../../README.md` — Viewer package overview (Available Scripts section
  documents `test:generate-fixtures` and `test:with-fixtures`).
