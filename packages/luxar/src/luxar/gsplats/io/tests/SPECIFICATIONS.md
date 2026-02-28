# luxar.gsplats.io.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Gaussian splatting I/O module: saving and loading `.gsplats.zarr` files, spatial ordering for efficient streaming, and format compliance verification.

---

## Test Files

| File | Description |
|------|-------------|
| `test_save_load.py` | Save and load functions (round-trip through `.gsplats.zarr`) |
| `test_ordering.py` | Spatial ordering module (Z-order/Hilbert curve indexing) |
| `test_format.py` | `.gsplats.zarr` format compliance (validates saved files match spec exactly) |

**Total**: 3 test files

---

## Key Test Patterns

- Save/load tests verify round-trip fidelity of all GSplatData attributes.
- Format tests validate zarr hierarchy structure, attribute names, array shapes, and dtypes against the format specification.
- Ordering tests verify spatial locality properties of the chosen ordering.

---

## Related Specifications

- `luxar.gsplats.io` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (3 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
