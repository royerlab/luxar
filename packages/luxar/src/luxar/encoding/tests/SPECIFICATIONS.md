# luxar.encoding.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Luxar encoding pipeline: array encoding (broadcasting, LUT, quantization, log-space, array references), decoding, the encoding registry, scalar input support, dynamic range dtype selection, and edge cases.

---

## Test Files

| File | Description |
|------|-------------|
| `test_encoder.py` | ArrayEncoder class: all semantic types, encoding modes, and special encodings |
| `test_decoder.py` | ArrayDecoder class: all decoding methods including broadcasting, LUT, array references, quantization, and log-space |
| `test_registry.py` | ArrayRefRegistry: duplicate detection, hashing, and registry lifecycle |
| `test_scalar_input.py` | Scalar input support in ArrayEncoder (v0.6.0 feature) |
| `test_dynamic_range.py` | Dynamic range-based dtype selection (uint8 vs uint16 vs float32) |
| `test_edge_cases.py` | Error paths, boundary conditions, and unusual inputs |

**Total**: 6 test files

---

## Encoding Modes Tested

- **Broadcasting**: Uniform values (1 value replicated to N points)
- **LUT**: Lookup table for <=256 unique values
- **Scalar LUT**: Per-channel LUT encoding
- **Quantization**: uint8/uint16 compressed representation
- **Array References**: Deduplication via shared arrays
- **Log-space**: Wide dynamic range encoding
- **Direct/no encoding**: Pass-through

---

## Cross-Language Compatibility

These tests are critical because Python encodes data and TypeScript decodes it. The corresponding TypeScript tests are in `packages/luxar-viewer/src/tests/unit/data/array-decoder.test.ts` and must stay in sync with the Python encoder behavior.

---

## Related Specifications

- `luxar.encoding` package: `../SPECIFICATIONS.md`
- Viewer array decoder tests: `packages/luxar-viewer/src/tests/unit/data/array-decoder.test.ts`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory and encoding mode documentation.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
