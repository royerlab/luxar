# luxar.gsplats.models.gsplats.metal.tests - Technical Specification

**Version**: 2.0.0
**Last Updated**: 2026-05-02

## Purpose

Tests for the Apple Metal backend: extension initialization, 3D splat-centric
forward/backward correctness, native `[Z,Y,X]` coordinate conventions,
L-to-conic computation, API parity, performance smoke coverage, and optimizer
compatibility.

---

## Test Files

| File | Description |
|------|-------------|
| `test_metal_backend.py` | Metal backend initialization, device detection, forward/backward basics |
| `test_metal_conic.py` | Metal L-to-Conic computation in native `[Z,Y,X]` packed order |
| `test_metal_interface.py` | CUDA/base API parity, supported-dimension PyTorch rendering, CPU/device-transfer rejection, dtype rejection, clean state dicts, dynamic splat ops, and zero-splat pruning |
| `test_metal_numerical.py` | Numerical accuracy, gradient signs/values, and optimization convergence |
| `test_metal_performance.py` | Performance benchmark smoke tests |
| `test_coordinate_transforms.py` | Native `[Z,Y,X]` coordinate convention and end-to-end asymmetric-splat checks |
| `test_optimizer_compatibility.py` | Optimizer compatibility and gradient flow |

---

## Key Test Patterns

- Metal tests are skipped on non-macOS systems or systems without MPS support.
- Numerical tests compare Metal results against CPU/PyTorch references with
  tolerance thresholds appropriate for GPU float32 math.
- Gradient tests verify both signs and magnitudes for center gradients and a
  short optimization convergence loop.
- Interface tests ensure the MPS/float32 invariants are enforced consistently.
- Performance tests are smoke/characterization tests; they print timings but do
  not require a specific speedup because Apple GPU generations vary.

---

## Related Specifications

- Metal backend package: `../SPECIFICATIONS.md`

---

## Changelog

- **v2.0.0** (2026-05-02): Updated for the splat-centric Metal rewrite: native
  `[Z,Y,X]` conics, no tile-binned hot path, and no packed-conic reorder tests.
- **v1.2.0** (2026-05-02): Added interface-parity tests and documented
  CPU/device-transfer rejection, dtype rejection, clean nested state dicts, and
  2D/4D MPS PyTorch rendering coverage.
- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory.
- **v1.0.0** (2026-01-02): Initial specification.
