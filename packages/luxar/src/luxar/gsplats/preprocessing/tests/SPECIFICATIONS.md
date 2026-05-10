# luxar.gsplats.preprocessing.tests - Technical Specification

**Version**: 1.0.0
**Last Updated**: 2026-05-10

## Purpose

The `luxar.gsplats.preprocessing.tests` package verifies preprocessing utilities used before Gaussian splat fitting, with emphasis on non-local means (NLM) denoising backend selection and output correctness.

---

## Core Concepts

### Backend Dispatch

The preprocessing package can select among CUDA, PyTorch, and scikit-image implementations depending on availability, input dimensionality, and user configuration. Tests assert that dispatch decisions are explicit and that fallbacks preserve behavior.

### Denoising Contract

NLM denoising must preserve array shape, produce finite numeric output, and keep dtype/range behavior consistent with the selected backend's documented contract.

### Optional CUDA Extension

CUDA tests are conditional on a CUDA-capable environment and the preprocessing CUDA extension. Missing CUDA support should skip CUDA-specific tests while leaving CPU/PyTorch coverage active.

---

## Data Structures

### Denoise Test Case

```text
DenoiseCase:
  input: numpy.ndarray | torch.Tensor
  h: float
  patch_size: int
  patch_distance: int
  backend: 'cuda' | 'pytorch' | 'skimage' | 'auto'
  expected_shape: tuple[int, ...]
```

**Invariants**:

- Inputs are finite numeric arrays.
- Output shape matches input shape.
- Backend-specific outputs are finite and suitable for downstream fitting.

---

## Algorithms

### Backend Dispatch Test

**Purpose**: Ensure preprocessing selects the intended backend or fallback.

**Inputs**:

- Backend preference.
- Environment capability flags/mocks.
- Representative input array.

**Outputs**:

- Assertion that the selected backend and result are correct.

**Algorithm**:

```text
1. Configure backend preference and capability state.
2. Run the preprocessing entry point.
3. Inspect selected backend or observable output path.
4. Assert output shape, finiteness, and fallback behavior.
```

**Complexity**: O(input voxel count × backend-specific search window) for denoising calls.

### Pipeline Denoise Test

**Purpose**: Validate the denoise pipeline as users call it from CLI/API paths.

**Inputs**:

- Small synthetic volume.
- Denoise parameters.

**Outputs**:

- Denoised array or saved output file with correct metadata/shape.

**Algorithm**:

```text
1. Create deterministic synthetic noisy input.
2. Run the denoise pipeline with a selected backend.
3. Verify output exists when writing to disk.
4. Verify output shape and numeric sanity.
5. Verify errors are informative for invalid parameters.
```

---

## Validation Rules

- Tests must not require CUDA unless explicitly marked/skipped based on CUDA availability.
- Synthetic data should be deterministic.
- Backend comparisons should use tolerances appropriate to algorithmic differences.
- Pipeline tests should use temporary directories and avoid persistent generated data.

---

## Cross-Language Compatibility

Not applicable. These tests validate Python preprocessing implementations and optional native extensions.

---

## Related Specifications

- `luxar.gsplats.preprocessing` - Preprocessing algorithms and backend behavior (see `../SPECIFICATIONS.md`).
- `luxar.gsplats` - Fitting pipeline that consumes preprocessed volumes (see `../../SPECIFICATIONS.md`).

---

## Changelog

- **v1.0.0** (2026-05-10): Initial specification for preprocessing tests.
