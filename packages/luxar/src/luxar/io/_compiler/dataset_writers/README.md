# dataset_writers

Per-attribute zarr array serializers used by the compiler. Each function takes a
zarr group plus one attribute array (positions, colors, radii, ...), routes it
through the shared `ArrayEncoder`, and logs the resulting encoding. These are the
canonical writers shared across all three geometry types — Points, Lines, and
GSplats — so a given semantic type (e.g. `COLOR`, `POSITIVE_SCALAR`) is encoded
identically everywhere.

## Overview

The compiler writes a geometry node by calling one writer per attribute. The
writers themselves are thin and stateless: the encoding policy (precision
selection, deduplication, LUT/broadcast detection) lives in `ArrayEncoder`, and
the chunk layout is computed by `..chunking.calculate_intelligent_chunks` so
chunk boundaries align with the spatial index when one is present.

All writers receive a frozen `DatasetCtx` (`..context.DatasetCtx`) carrying the
shared `encoder`, `encoding_mode`, and `compressor`. None of them hold a
back-pointer to the orchestrator.

```
LuxarZarrCompiler / gsplat_assembly
        │  (one call per attribute)
        ▼
  write_positions   → semantic_type = COORDINATE
  write_colors      → semantic_type = COLOR
  write_positive_scalar → POSITIVE_SCALAR   (radii / widths / amplitudes / scalars)
  write_bounded_scalar  → BOUNDED_SCALAR    (sharpnesses)
        │
        ▼
  ArrayEncoder.encode(...)  →  zarr array + "encoding" attrs
```

## File Structure

```
dataset_writers/
├── __init__.py     # empty (writers imported by module path)
├── positions.py    # write_positions          (COORDINATE)
├── colors.py       # write_colors             (COLOR; SDR/HDR detection)
└── scalars.py      # write_positive_scalar, write_bounded_scalar,
                    #   write_radii, write_sharpness, write_scalars
```

## API

### `positions.py`

- **`write_positions(group, positions, spatial_index_data, ctx)`**
  Writes the `positions` array as the `COORDINATE` semantic type. Chunks are
  computed from the position shape and the optional spatial index. Logs whether
  positions were stored as `float32` or `float16` (the latter under MEMORY mode).

### `colors.py`

- **`write_colors(group, colors, spatial_index_data, n_elements, ctx)`**
  Writes the `colors` array (or a single broadcast tuple/list) as the `COLOR`
  semantic type. Detects SDR vs HDR from the data (`color_mode = "hdr"` when any
  value exceeds 1.0) and forwards it to the encoder. Records
  `group.attrs["color_data_range"]` (min/max of the original data) for the
  viewer's layer controls, and logs the actual encoding chosen
  (`broadcasted`, `array_ref`, `lut_uint8`, `lut_uint16`, `rgb_uint8`, `float32`, ...).

### `scalars.py`

The two canonical scalar writers plus geometry-named convenience wrappers and the
colormap-scalars writer:

- **`write_positive_scalar(group, data, name, spatial_index_data, n_elements, ctx, log_label_singular=None) -> float`**
  Canonical `POSITIVE_SCALAR` writer used by Points (`radii`), Lines (`widths`),
  and GSplats (`amplitudes`). Accepts an array or a single broadcast value.
  Returns the maximum value so callers can cache it for layer-control metadata
  without recomputing. `log_label_singular` controls the log wording
  (`"radius"` / `"width"` / `"amplitude"`); defaults to `name`.

- **`write_bounded_scalar(group, data, name, bounds, spatial_index_data, n_elements, ctx, log_label_singular=None) -> float`**
  Canonical `BOUNDED_SCALAR` writer used for `sharpnesses`. The `bounds` tuple is
  forwarded to the encoder, which quantizes the data to Uint8 normalised to that
  range when the encoding mode allows. Also returns the maximum value.

- **`write_radii(group, radii, spatial_index_data, n_points, ctx) -> float`**
  Points-specific wrapper over `write_positive_scalar` (name `"radii"`,
  label `"radius"`). Kept only for readability in `write_points`; new geometries
  should call `write_positive_scalar` directly.

- **`write_sharpness(group, sharpness, spatial_index_data, n_points, ctx) -> float`**
  Points-specific wrapper over `write_bounded_scalar` (name `"sharpnesses"`,
  bounds `(0.0, SHARPNESS_MAX)`, label `"sharpness"`).

- **`write_scalars(group, scalars, spatial_index_data, n_elements, ctx)`**
  Writes the `scalars` array used for colormap lookup (encoded as
  `POSITIVE_SCALAR`). Requires the group to already contain a position array
  (`positions`, `vertices`, or `centers`) and raises `RuntimeError` otherwise.
  Records `group.attrs["scalar_data_range"]` (min/max) for layer controls.

## Invariants

- **`n_elements` is the logical count, never the physical zarr shape.**
  Deduplicated positions/vertices can be stored as an `array_ref` with physical
  shape `(0, D)`, so the element count must be passed in by the caller. The
  scalar/color writers use it to size broadcast and `array_ref` arrays; passing
  the wrong count silently desyncs an attribute from its geometry.

- **A single broadcast value (or a length-1 array) is written as a broadcast
  encoding**, with `n_elements` forwarded and `chunks=None`. Full-length arrays
  get intelligent chunks instead and `n_elements=None`.

- **One semantic type, one writer.** A new geometry type reuses these writers
  with its own dataset name rather than re-implementing precision/dedup logic.

## Dependencies

**Internal:**
- `..context.DatasetCtx` — encoder / mode / compressor bundle
- `..chunking.calculate_intelligent_chunks` — spatial-index-aligned chunking
- `....encoding.SemanticType` / `ArrayEncoder` — encoding policy
- `....typing_utils.constants.SHARPNESS_MAX` — sharpness upper bound

**External:**
- `numpy`, `zarr`, `arbol` (`aprint` for encoding logs)

## Callers

- `luxar.io.compiler` (`write_points` / `write_lines` paths) imports
  `write_positions`, `write_colors`, and the scalar writers.
- `luxar.io._compiler.gsplat_assembly` imports `write_colors` and
  `write_positive_scalar` for amplitudes.

Direct unit tests live in
`luxar/io/tests/_compiler/test_datasets.py`.

## See Also

- [../context.py](../context.py) — `DatasetCtx` and the other narrow `Ctx` objects
- [encoding/README.md](../../../encoding/README.md) — semantic types and `ArrayEncoder`
- [core/README.md](../../../core/README.md) — geometry node attributes
