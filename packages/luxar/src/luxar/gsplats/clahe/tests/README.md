# luxar.gsplats.clahe.tests

Tests for the CLAHE (Contrast Limited Adaptive Histogram Equalization) implementation.

## What is Tested

- **Basic functionality**: uniform image unchanged, range/shape/dtype preservation
- **Contrast enhancement**: low-contrast region enhancement, heterogeneous balancing, clip_limit effect
- **nD support**: 1D, 2D, 3D, 4D, non-square shapes
- **Edge cases**: small images, single tile, exact/inexact tile division, near-uniform, all-zero
- **Sampling probabilities**: probability properties, sampling, uniform image probabilities
- **Parameter validation**: various tile sizes, clip limits, nbins values
- **Numerical stability**: extreme values, negative values, mixed signs, small range

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/clahe/tests/ -v
```
