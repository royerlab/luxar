# luxar.gsplats.models.gsplats.metal.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for the Metal (Apple GPU) backend for Gaussian splatting: backend initialization, L-to-Conic computation, numerical accuracy, performance benchmarks, coordinate transforms (PyTorch ZYX to Metal XYZ), and optimizer compatibility.

---

## Test Files

| File | Description |
|------|-------------|
| `test_metal_backend.py` | Metal backend initialization, device detection, and basic operations |
| `test_metal_conic.py` | Metal L-to-Conic computation (Cholesky factor to conic matrix) |
| `test_metal_numerical.py` | Numerical accuracy and precision tests for Metal backend |
| `test_metal_performance.py` | Performance benchmark tests for Metal backend |
| `test_coordinate_transforms.py` | Coordinate transformations between PyTorch [Z,Y,X] and Metal [X,Y,Z] |
| `test_optimizer_compatibility.py` | Optimizer compatibility with Metal backend (gradient flow) |

**Total**: 6 test files

---

## Key Test Patterns

- Metal tests are conditionally skipped on non-macOS systems or systems without MPS support.
- Numerical tests compare Metal results against CPU reference implementations with tolerance thresholds.
- Performance tests measure throughput and compare against CPU baseline.
- Coordinate transform tests verify axis ordering conventions are handled correctly.

---

## Related Specifications

- `luxar.gsplats.models.gsplats.metal` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (6 files).
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
