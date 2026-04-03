# luxar.gsplats.io.tests

Tests for Gaussian splat I/O (save, load, inspect) and spatial ordering.

## What is Tested

- **Save** (`test_save_load.py`): basic save, colors, Morton/Hilbert ordering, encoding modes, fitting info, validation errors, compression, chunk capping
- **Load** (`test_save_load.py`): basic load, encoded arrays, stats, missing file, invalid format
- **Round-trip** (`test_save_load.py`): basic, with ordering, with quantization, GSplatData methods, color modes (SDR/HDR/uint8), multi-LOD
- **Inspect** (`test_save_load.py`): basic metadata, fitting info, formatted output
- **Ordering** (`test_ordering.py`): Morton encoding (2D/3D/nD), Hilbert encoding, coordinate normalization, auto-resolution, chunk bounds
- **Format compliance** (`test_format.py`): root attributes, splats group structure, array shapes, encoding metadata

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/io/tests/ -v
```
