# luxar.gsplats.tests - Technical Specification

**Version**: 1.2.0
**Last Updated**: 2026-03-31

## Purpose

Top-level tests for the Gaussian splatting module: high-level fitting API (`fit_gaussian_splats`), GSplatData class, end-to-end integration tests, and Cholesky dimension operations.

---

## Test Files

| File | Description |
|------|-------------|
| `test_batch.py` | Batch fitting infrastructure (manifest, env, slurm, time, merge) |
| `test_cholesky_dim_ops.py` | Cholesky dimension permutation and embedding utilities (`permute_cholesky_packed`, `embed_cholesky_packed`, `pack_tril`, `unpack_tril`) |
| `test_culling.py` | Contribution-based Gaussian splat culling |
| `test_fit_gsplats.py` | `fit_gaussian_splats` function and optimization pipeline |
| `test_gpu_profile.py` | GPU profile management (multi-GPU registry) |
| `test_gsplat_data.py` | GSplatData class methods |
| `test_gsplats_integration.py` | End-to-end pipeline tests (candidate generation to fitting to rendering) |
| `test_metrics.py` | Quality metrics (PSNR, SSIM, MSE, rel_l2, max_abs_error) |
| `test_progressive_fitting.py` | Progressive Gaussian splat fitting |
| `test_spatial_volume_filter.py` | Spatial volume computation and specific-brightness filtering |
| `test_tiled_fitting.py` | Tiled Gaussian splat fitting |

**Total**: 11 test files

---

## Key Test Patterns

- Integration tests verify the complete pipeline from volume data through seed generation, fitting, and rendering output.
- Cholesky tests verify that dimension permutations preserve covariance matrices.

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

- **v1.2.0** (2026-03-31): Updated test file inventory (4 -> 11 files). Added test_batch, test_culling, test_gpu_profile, test_metrics, test_progressive_fitting, test_spatial_volume_filter, test_tiled_fitting.
- **v1.1.0** (2026-02-28): Rewritten with actual test file inventory (5 files) and subpackage directory listing.
- **v1.0.0** (2026-01-02): Initial specification (boilerplate).
