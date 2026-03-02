# luxar.gsplats.tests - Technical Specification

**Version**: 1.1.0
**Last Updated**: 2026-02-28

## Purpose

Top-level tests for the Gaussian splatting module: high-level fitting API (`fit_gaussian_splats`), GSplatData class, end-to-end integration tests, multi-scale fitting, and Cholesky dimension operations.

---

## Test Files

| File | Description |
|------|-------------|
| `test_fit_gsplats.py` | `fit_gaussian_splats` function and optimization pipeline |
| `test_gsplat_data.py` | GSplatData class methods |
| `test_gsplats_integration.py` | End-to-end pipeline tests (candidate generation to fitting to rendering) |
| `test_multiscale_fitting.py` | Multi-scale Gaussian splat fitting |
| `test_cholesky_dim_ops.py` | Cholesky dimension permutation and embedding utilities (`permute_cholesky_packed`, `embed_cholesky_packed`, `pack_tril`, `unpack_tril`) |

**Total**: 5 test files

---

## Key Test Patterns

- Integration tests verify the complete pipeline from volume data through seed generation, fitting, and rendering output.
- Cholesky tests verify that dimension permutations preserve covariance matrices.
- Multi-scale tests verify that hierarchical decomposition produces valid Gaussian representations.

---

## Subpackage Test Suites

Each gsplats subpackage has its own test directory:
- `gsplats.clahe.tests` - CLAHE implementation
- `gsplats.fitting.tests` - Fitting pipeline components
- `gsplats.fitting.dynamic_ops.tests` - Dynamic operations (split/merge/relocate)
- `gsplats.io.tests` - GSplats I/O format
- `gsplats.models.gsplats.tests` - GSplat model
- `gsplats.models.gsplats.metal.tests` - Metal backend
- `gsplats.models.utils.tests` - Model utilities
- `gsplats.multiscale.tests` - Multiscale decomposition
- `gsplats.optim.tests` - Optimizer integration
- `gsplats.seeds.tests` - Seed generation
- `gsplats.utils.tests` - Utility functions

---

## Related Specifications

- `luxar.gsplats` package: `../SPECIFICATIONS.md`

---

## Changelog

- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (5 files) and subpackage directory listing.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
