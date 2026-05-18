# luxar.colormaps.tests - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-10

## Purpose

The `luxar.colormaps.tests` package verifies the public colormap registry and colormap resolution behavior used by Python writers and the TypeScript viewer.

---

## Core Concepts

### Built-in Colormap Registry

Built-in colormaps are exposed as named lookup tables. Tests verify names, array shape, dtype, copy semantics, and expected endpoint behavior.

### Custom Colormap Resolution

`resolve_colormap()` accepts built-in names and user-provided arrays. Tests cover float and integer inputs, resampling to the expected LUT size, validation failures, and optional matplotlib/colorcet fallback behavior.

### Cross-Language LUT Contract

Python-authored colormap LUTs are consumed by the viewer as 256-entry RGB or RGBA byte arrays. Tests protect the writer-side shape and dtype expectations that the viewer relies on.

---

## Data Structures

### Resolved Colormap

```text
ResolvedColormap:
  lut: numpy.ndarray
  shape: (256, 3) or (256, 4)
  dtype: uint8
```

**Invariants**:

- Built-in lookups return independent arrays so callers cannot mutate registry state.
- Float custom arrays are in the range [0, 1] before conversion.
- Integer custom arrays use supported unsigned byte-compatible values.
- Invalid names and malformed arrays raise clear exceptions.

---

## Algorithms

### Built-in Registry Test

**Purpose**: Verify built-in colormap data is available and immutable from caller perspective.

**Inputs**:

- Built-in colormap names.

**Outputs**:

- Assertions on LUT shape, dtype, and copy semantics.

**Algorithm**:

```text
1. Resolve a built-in colormap by name.
2. Assert shape and dtype.
3. Mutate the returned array.
4. Resolve the same name again.
5. Assert registry data was not mutated.
```

### Custom Array Resolution Test

**Purpose**: Verify custom colormap inputs normalize to the writer/viewer LUT contract.

**Inputs**:

- Custom float or uint8 arrays.
- Optional target number of entries.

**Outputs**:

- Normalized uint8 LUT with expected shape.

**Algorithm**:

```text
1. Pass custom input to resolve_colormap().
2. Validate or resample entries as needed.
3. Convert to uint8 LUT.
4. Assert shape, dtype, range, and representative values.
```

---

## Validation Rules

- Unknown colormap names raise errors unless an optional fallback provider resolves them.
- Custom arrays must be 2D with 3 or 4 channels.
- Float arrays must stay within [0, 1].
- Arrays with too few entries or unsupported dtypes are rejected.
- Optional fallback tests are skipped when their dependency is unavailable.

---

## Cross-Language Compatibility

Python colormap LUTs are serialized into Luxar zarr metadata/arrays and loaded by `packages/luxar-viewer/src/rendering/colormap-textures.ts`. The test suite protects the 256-entry RGB/RGBA byte layout expected by the viewer.

---

## Related Specifications

- `luxar.colormaps` - Colormap registry and resolution behavior (see `../SPECIFICATIONS.md`).
- Luxar zarr format - Colormap metadata and LUT arrays (see `../../../../../../docs/guides/user/LUXAR_ZARR_FORMAT.md`).

---

## Changelog

- **v1.0.0** (2026-05-10): Initial specification for colormap tests.
