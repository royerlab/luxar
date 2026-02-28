# luxar.io.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Luxar I/O module: zarr reading/writing, the LuxarZarrCompiler, spatial indexing, ordering strategies (points and lines), progressive writing, nD chunking, metadata handling, and round-trip verification.

---

## Test Files

| File | Description |
|------|-------------|
| `test_roundtrip.py` | Comprehensive round-trip tests verifying all aspects of the Luxar zarr format are correctly written and read back |
| `test_compiler_integration.py` | Integration tests for the progressive writing API using LuxarZarrCompiler |
| `test_compiler_improvements.py` | Tests for compiler improvements and fixes |
| `test_io_metadata.py` | Metadata handling in I/O operations |
| `test_ordering_points.py` | Points chunk bounds computation with discrete dimensions |
| `test_ordering_lines.py` | Lines spatial indexing with dual ordering |
| `test_progressive_writing.py` | Progressive writing architecture with LuxarZarrCompiler |
| `test_zarr_nd_chunking.py` | nD zarr data handling and chunking optimization |
| `test_writer_parent_parameter.py` | Zarr hierarchy creation with nested groups and points |
| `test_reader_nodes.py` | Reader node collection (verifying no duplicates in any scenario) |

**Total**: 10 test files

---

## Key Test Patterns

- Round-trip tests write a scene via the compiler, then read it back and verify all attributes match.
- Compiler integration tests verify progressive writing (multiple `add_points` calls building up a scene).
- Ordering tests verify spatial index correctness for efficient chunk-based loading.
- nD chunking tests verify correct behavior for 4D+ datasets.

---

## Related Specifications

- `luxar.io` package: `../SPECIFICATIONS.md`
- Zarr format spec: `docs/guides/user/LUXAR_ZARR_FORMAT.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (10 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
