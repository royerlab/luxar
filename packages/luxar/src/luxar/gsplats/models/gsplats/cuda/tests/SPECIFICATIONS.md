# luxar.gsplats.models.gsplats.cuda.tests - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-10

## Purpose

The `luxar.gsplats.models.gsplats.cuda.tests` package verifies correctness, numerical stability, and integration behavior of the CUDA Gaussian splat backend.

---

## Core Concepts

### Backend Parity

CUDA kernels must match the reference PyTorch implementation within documented tolerances. Tests compare rendered outputs and gradients for representative splat shapes, covariance structures, and image/volume sizes.

### Gradient Correctness

Backward tests validate gradients for positions, amplitudes, Cholesky factors, and other differentiable model parameters. Gradcheck tests use small inputs and double precision where practical.

### Environment Gating

Tests that require CUDA are skipped when CUDA or the compiled extension is unavailable. Skips must reflect missing environment capabilities, not hidden test failures.

---

## Data Structures

### CUDA Test Fixture

```text
CudaFixture:
  device: torch.device('cuda')
  splat_parameters: tensors on CUDA
  render_shape: tuple[int, ...]
  reference_output: tensor from PyTorch reference path
```

**Invariants**:

- Input tensors are on the CUDA device unless a test explicitly validates host/device transfer behavior.
- Cholesky factors represent positive-definite covariance matrices.
- Reference and CUDA outputs have the same shape and dtype before comparison.

---

## Algorithms

### Forward Parity Test

**Purpose**: Ensure CUDA rendering matches the reference implementation.

**Inputs**:

- Synthetic splat parameters.
- Render shape.
- Reference renderer.
- CUDA renderer.

**Outputs**:

- Assertion that max/mean absolute error is within tolerance.

**Algorithm**:

```text
1. Build deterministic synthetic splat parameters.
2. Render with the reference path.
3. Render with the CUDA path.
4. Compare shapes and dtypes.
5. Assert numerical error thresholds.
```

**Complexity**: O(number of splats × rendered voxels/pixels) for the compared render.

### Backward Parity Test

**Purpose**: Ensure CUDA gradients match reference autograd behavior.

**Inputs**:

- Differentiable splat parameters.
- Scalar loss derived from rendered output.

**Outputs**:

- Assertion that CUDA gradients and reference gradients agree within tolerance.

**Algorithm**:

```text
1. Clone inputs for reference and CUDA paths.
2. Enable gradients on differentiable tensors.
3. Render both paths and reduce to scalar losses.
4. Backpropagate.
5. Compare each gradient tensor with tolerance appropriate for dtype.
```

**Complexity**: O(forward + backward kernel work).

---

## Validation Rules

- CUDA-only tests must use pytest skips for unavailable CUDA devices or unbuilt extensions.
- Randomized tests must set deterministic seeds.
- Numerical tolerances must account for dtype and operation order, especially FP16 tests.
- Performance tests are smoke tests; they should not encode hardware-specific hard limits unless marked accordingly.

---

## Cross-Language Compatibility

Not applicable. These tests validate Python/PyTorch and native CUDA extension behavior within the Python package.

---

## Related Specifications

- `luxar.gsplats.models.gsplats` - Gaussian splat model behavior (see `../SPECIFICATIONS.md`).
- CUDA build and HPC instructions (see repository `AGENTS.md` and `Makefile` CUDA targets).

---

## Changelog

- **v1.0.0** (2026-05-10): Initial specification for CUDA backend tests.
