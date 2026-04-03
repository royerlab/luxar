# luxar.gsplats.utils.tests

Tests for Gaussian splat utility functions (Cholesky packing, gradient dilution, embedding).

## What is Tested

- **Gradient dilution factor**: 1D-8D values, monotonic increase, conservative vs aggressive boundary, return type
- **tril_size**: formula correctness for dimensions 0-9
- **pack_tril / unpack_tril**: single/batch matrices, dtype preservation, empty batch, round-trip for dimensions 1-5
- **validate_cholesky_shape**: per-splat/uniform for 2D/3D, invalid shapes, n_splats mismatch, uniform not allowed
- **embed_cholesky_packed**: well-conditioned, degenerate zeros, all-zero, near-singular, empty, single splat

## How to Run

```bash
hatch run pytest packages/luxar/src/luxar/gsplats/utils/tests/ -v
```
