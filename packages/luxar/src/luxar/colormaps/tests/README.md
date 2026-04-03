# luxar.colormaps.tests

Unit tests for the `luxar.colormaps` package covering built-in colormap data, resolution, and custom array handling.

## What's Tested

- **Built-in colormaps** - LUT shape/dtype, copy semantics, ramp endpoints, unknown name errors
- **resolve_colormap()** - Name resolution, custom float/uint8 arrays, resampling, dtype conversion
- **Validation** - Wrong shape, bad dtype, out-of-range floats, too few entries, invalid types
- **Optional fallbacks** - matplotlib and colorcet integration (skipped if not installed)

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/colormaps/tests/
hatch run pytest packages/luxar/src/luxar/colormaps/tests/test_registry.py  # Single file
```
