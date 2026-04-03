# luxar.encoding.tests

Unit tests for the `luxar.encoding` package covering array encoding, decoding, registry, and roundtrip behavior.

## What's Tested

- **ArrayEncoder** - Semantic types, encoding modes, broadcasting, LUT, dtype, special encodings
- **ArrayDecoder** - Broadcasting expansion, LUT decode, array references, quantized inverse transforms
- **ArrayRefRegistry** - Duplicate detection, two-stage hashing, lifecycle (check/clear)
- **Roundtrip** - Full encode-decode cycles with numerical tolerance per semantic type
- **Dynamic range** - Range computation and dtype mapping for POSITIVE_SCALAR
- **Scalar input** - Scalar/tuple convenience inputs (v0.6.0 feature)
- **Edge cases** - Error paths, boundary conditions, empty arrays, unusual inputs

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/encoding/tests/
hatch run pytest packages/luxar/src/luxar/encoding/tests/test_encoder.py  # Single file
```
