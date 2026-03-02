# luxar.gsplats.seeds.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Tests for Gaussian splatting seed generation: the unified `generate_seeds()` entry point, individual seeding methods (grid, edges, multiscale decomposition), GPU-accelerated operations, utility functions, and integration tests.

---

## Test Files

| File | Description |
|------|-------------|
| `test_generate_seeds.py` | `generate_seeds()` unified entry point function |
| `test_grid.py` | `seed_from_grid()` uniform grid seeding |
| `test_edges.py` | `seed_from_edges()` edge-based seeding |
| `test_multiscale_decomposition.py` | `seed_from_decomposition` scale-hierarchical detection |
| `test_gpu_ops.py` | GPU-accelerated seed generation operations |
| `test_utils.py` | Seed detection utility functions |
| `test_seeds_integration.py` | Integration tests for the seeds sub-package |

**Total**: 7 test files

---

## Key Test Patterns

- `generate_seeds` tests verify that the unified API correctly dispatches to specific seeding methods.
- Grid tests verify uniform spacing and boundary handling.
- Edge tests verify that seeds are placed at intensity gradients.
- GPU ops tests are conditionally skipped when CUDA/MPS is not available.
- Integration tests verify the complete seeding pipeline on synthetic volumes.

---

## GPU Testing

GPU tests (`test_gpu_ops.py`) test:
- Sobel gradients on GPU (all dimensions)
- Peak detection (2D/3D only, auto-fallback for others)
- Interpolation (2D/3D only, auto-fallback for others)
- Deduplication (all dimensions)

Tests are marked with `@pytest.mark.skipif` for environments without GPU support.

---

## Related Specifications

- `luxar.gsplats.seeds` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (7 files) and GPU testing documentation.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
