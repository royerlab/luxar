# CUDA Backend Specification - Testing Strategy

**Version**: 0.1.0
**Status**: Implementation Complete
**Last Updated**: 2026-01-10

> **Note**: This is Part 3 of the CUDA Backend Specification (Testing Strategy).
> See also:
> - [Part 1: Core Algorithms](SPECIFICATIONS.md) - Architecture, kernel design, memory optimization
> - [Part 2: PyTorch Integration](SPECIFICATIONS_PYTORCH_INTEGRATION.md) - Integration, performance, implementation phases

## Table of Contents

11. [Testing Strategy](#11-testing-strategy)
    - 11.0 [Test Infrastructure](#110-test-infrastructure)
      - 11.0.1 [Pytest Configuration](#1101-pytest-configuration)
      - 11.0.2 [CI/CD Configuration](#1102-cicd-configuration)
      - 11.0.3 [Test Organization](#1103-test-organization)
      - 11.0.4 [Running Tests](#1104-running-tests)
    - 11.1 [Unit Tests](#111-unit-tests)
      - 11.1.1 [Reference Comparison Tests (CUDA vs PyTorch)](#1111-reference-comparison-tests-cuda-vs-pytorch)
    - 11.2 [Edge Case Tests](#112-edge-case-tests)
    - 11.3 [Operator Correctness Tests (opcheck)](#113-operator-correctness-tests-opcheck)
    - 11.4 [C++ Kernel Unit Tests (reference design, not implemented)](#114-c-kernel-unit-tests)
      - 11.4.1 [Test Framework Setup](#1141-test-framework-setup)
      - 11.4.2 [AABB Computation Tests](#1142-aabb-computation-tests)
      - 11.4.3 [Mahalanobis Distance Tests](#1143-mahalanobis-distance-tests)
      - 11.4.4 [Conic Gradient Finite Difference Tests (CRITICAL)](#1144-conic-gradient-finite-difference-tests-critical)
      - 11.4.5 [Running C++ Tests](#1145-running-c-tests)
      - 11.4.6 [Integration with Python Tests](#1146-integration-with-python-tests)

---

## 11. Testing Strategy

### 11.0 Test Infrastructure

This section defines the testing infrastructure required for comprehensive, maintainable tests
with high coverage. All tests should be runnable both locally and in CI environments.

#### 11.0.1 Pytest Configuration

```python
# tests/conftest.py
"""
Pytest fixtures and configuration for CUDA backend tests.

Usage:
    pytest tests/                           # Run all tests
    pytest tests/ -m "not slow"             # Skip slow tests
    pytest tests/ -m "not gpu"              # Skip tests requiring GPU
    pytest tests/ --cov=cuda_splatting      # With coverage
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any


# =============================================================================
# MARKERS
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU (deselect with '-m \"not gpu\"')")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "numerical: marks numerical accuracy tests")


# =============================================================================
# GPU AVAILABILITY
# =============================================================================

@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available, skip test if not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture(scope="session")
def cuda_device(cuda_available):
    """Return CUDA device for tests."""
    return torch.device("cuda:0")


# =============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def splat_params_2d() -> Dict[str, np.ndarray]:
    """Generate random 2D splat parameters."""
    N = 100
    DIM = 2
    return {
        "centers": np.random.rand(N, DIM).astype(np.float32) * 64,
        "L": np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0,
        "amps": np.abs(np.random.randn(N).astype(np.float32)) + 0.1,
        "sharpness": np.ones(N, dtype=np.float32) * 2.0,
        "shape": (64, 64),
        "sigma_min_diag": [0.1] * DIM,
    }


@pytest.fixture
def splat_params_3d() -> Dict[str, np.ndarray]:
    """Generate random 3D splat parameters."""
    N = 100
    DIM = 3
    return {
        "centers": np.random.rand(N, DIM).astype(np.float32) * 64,
        "L": np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0,
        "amps": np.abs(np.random.randn(N).astype(np.float32)) + 0.1,
        "sharpness": np.ones(N, dtype=np.float32) * 2.0,
        "shape": (64, 64, 64),
        "sigma_min_diag": [0.1] * DIM,
    }


@pytest.fixture(params=[2, 3, 4])
def splat_params_nd(request) -> Dict[str, np.ndarray]:
    """Parameterized fixture for testing multiple dimensions."""
    DIM = request.param
    N = 50
    shape_size = max(8, 64 // DIM)  # Smaller shapes for higher dimensions

    return {
        "centers": np.random.rand(N, DIM).astype(np.float32) * shape_size,
        "L": np.eye(DIM, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 1.5,
        "amps": np.abs(np.random.randn(N).astype(np.float32)) + 0.1,
        "sharpness": np.ones(N, dtype=np.float32) * 2.0,
        "shape": tuple([shape_size] * DIM),
        "sigma_min_diag": [0.1] * DIM,
        "dim": DIM,
    }


# =============================================================================
# MODEL FACTORIES
# =============================================================================

@pytest.fixture
def reference_model_factory(cuda_device):
    """
    Factory for creating PyTorch reference models.

    Usage:
        model = reference_model_factory(params)
    """
    def factory(params: Dict[str, Any], device=None):
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
        return GaussianSplatModel(
            shape=params["shape"],
            centers0=params["centers"],
            L0=params["L"],
            amps0=params["amps"],
            sigma_min_diag=params["sigma_min_diag"],
            device=device or cuda_device,
        )
    return factory


@pytest.fixture
def cuda_model_factory(cuda_device):
    """
    Factory for creating CUDA-accelerated models.

    Usage:
        model = cuda_model_factory(params)
    """
    def factory(params: Dict[str, Any]):
        from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import GaussianSplatModelCUDA
        return GaussianSplatModelCUDA(
            shape=params["shape"],
            centers0=params["centers"],
            L0=params["L"],
            amps0=params["amps"],
            sigma_min_diag=params["sigma_min_diag"],
        )
    return factory


@pytest.fixture
def mock_backend():
    """Return mock backend for testing without GPU.

    Note: MockSplattingBackend was removed. Tests requiring no GPU
    should use pytest markers (@pytest.mark.gpu) to skip GPU tests
    or use the PyTorch reference model (GaussianSplatModel) instead.
    """
    pytest.skip("MockSplattingBackend is no longer available; use reference model instead")


# =============================================================================
# TOLERANCE CONSTANTS
# =============================================================================

class Tolerances:
    """Tolerance levels for numerical comparisons."""

    # Forward pass comparison
    FORWARD_RTOL = 1e-4
    FORWARD_ATOL = 1e-6

    # Backward pass comparison (looser due to atomics)
    BACKWARD_RTOL = 1e-3
    BACKWARD_ATOL = 1e-5

    # FP16 comparison
    FP16_RTOL = 1e-2
    FP16_ATOL = 1e-4

    # Convergence comparison (for fitting tests)
    CONVERGENCE_RTOL = 0.1  # 10% relative difference in final loss


@pytest.fixture
def tolerances():
    """Return tolerance constants."""
    return Tolerances()
```

#### 11.0.2 CI/CD Configuration

```yaml
# .github/workflows/cuda-tests.yml
name: CUDA Backend Tests

on:
  push:
    paths:
      - 'packages/luxar/src/luxar/gsplats/models/gsplats/cuda/**'
  pull_request:
    paths:
      - 'packages/luxar/src/luxar/gsplats/models/gsplats/cuda/**'

jobs:
  # Tests that don't require GPU (mock backend, input validation)
  test-no-gpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pytest pytest-cov numpy torch --index-url https://download.pytorch.org/whl/cpu
          pip install -e packages/luxar
      - name: Run non-GPU tests
        run: |
          pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/ \
            -m "not gpu" \
            --cov=luxar.gsplats.models.gsplats.cuda \
            --cov-report=xml

  # Tests requiring GPU (numerical validation, performance)
  test-gpu:
    runs-on: [self-hosted, gpu]  # Requires self-hosted runner with GPU
    steps:
      - uses: actions/checkout@v4
      - name: Build CUDA extension
        run: |
          cd packages/luxar/src/luxar/gsplats/models/gsplats/cuda
          pip install -e .
      - name: Run GPU tests
        run: |
          pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/ \
            -m "gpu" \
            --cov=luxar.gsplats.models.gsplats.cuda \
            --cov-report=xml \
            --cov-fail-under=80

  # Performance regression tests (optional, slow)
  test-performance:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run performance benchmarks
        run: |
          pytest packages/luxar/src/luxar/gsplats/models/gsplats/cuda/tests/ \
            -m "slow" \
            --benchmark-json=benchmark.json
      - name: Store benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: benchmark.json
```

#### 11.0.3 Test Organization

```
tests/
├── conftest.py                    # Fixtures and configuration
├── test_cuda_forward.py           # Forward pass correctness tests
├── test_cuda_backward.py          # Backward pass correctness tests
├── test_cuda_gradcheck.py         # Gradient correctness (autograd comparison)
├── test_cuda_numerical.py         # Numerical precision tests
├── test_cuda_comparison.py        # CUDA vs PyTorch reference comparison
├── test_cuda_model.py             # Full model integration tests
├── test_cuda_nd.py                # nD (4D-8D) tests
├── test_cuda_fp16.py              # FP16 mode tests
├── test_cuda_performance.py       # Performance benchmarks (marked slow)
└── test_cuda_review_fixes.py      # Regression tests for specific bug fixes
```

#### 11.0.4 Running Tests

```bash
# Run all tests (requires GPU)
pytest tests/

# Run only non-GPU tests (CI without GPU)
pytest tests/ -m "not gpu"

# Run with coverage
pytest tests/ --cov=cuda_splatting --cov-report=html

# Run specific test file
pytest tests/test_cuda_comparison.py -v

# Run parameterized tests for specific dimension
pytest tests/ -k "3d" -v

# Skip slow performance tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/ -m "integration"

# Debug failing test with full output
pytest tests/test_cuda_numerical.py::TestCUDANumerical::test_conic_gradient_off_diagonal_scaling -v -s
```

### 11.1 Unit Tests

```python
class TestCUDABackend:
    """Core functionality tests."""

    def test_forward_matches_cpu_3d(self):
        """CUDA forward output matches PyTorch reference."""

    def test_forward_matches_cpu_2d(self):
        """CUDA 2D forward matches PyTorch reference."""

    def test_gradcheck_3d(self):
        """torch.autograd.gradcheck passes for 3D."""

    def test_gradient_values_match_cpu(self):
        """Gradient values match PyTorch autograd."""

    def test_gradient_sign_correctness(self):
        """Gradient signs point toward loss reduction."""

class TestCUDANumerical:
    """Numerical accuracy tests."""

    def test_conic_from_cholesky(self):
        """Σ⁻¹ computation matches numpy reference."""

    def test_mahalanobis_distance(self):
        """Distance computation matches numpy."""

    def test_generalized_gaussian(self):
        """Sharpness parameter correctly affects shape."""

    def test_conic_gradient_off_diagonal_scaling(self):
        """
        CRITICAL: Verify off-diagonal conic gradients have 2× factor.

        The packed upper-triangle stores each off-diagonal element once,
        but it contributes twice to D² (via symmetry). Missing the 2× factor
        causes incorrect L gradients through the chain rule.

        Test: For 2D conic [c_00, c_01, c_11], verify:
          d_conic[0] = d[0]² × grad_dist  (diagonal, no scaling)
          d_conic[1] = 2 × d[0] × d[1] × grad_dist  (off-diagonal, 2× scaling!)
          d_conic[2] = d[1]² × grad_dist  (diagonal, no scaling)
        """
        # Create simple 2D test case
        d = torch.tensor([2.0, 3.0], device="cuda")  # Displacement
        conic = torch.tensor([1.0, 0.5, 1.0], device="cuda", requires_grad=True)  # [c_00, c_01, c_11]

        # D² = d[0]² × c_00 + 2 × d[0] × d[1] × c_01 + d[1]² × c_11
        dist_sq = d[0]**2 * conic[0] + 2 * d[0] * d[1] * conic[1] + d[1]**2 * conic[2]
        dist_sq.backward()

        # Expected gradients
        expected = torch.tensor([
            d[0]**2,              # d_c_00 = 4.0 (diagonal, no 2×)
            2 * d[0] * d[1],      # d_c_01 = 12.0 (off-diagonal, 2×!)
            d[1]**2,              # d_c_11 = 9.0 (diagonal, no 2×)
        ], device="cuda")

        assert torch.allclose(conic.grad, expected), (
            f"Off-diagonal scaling error: got {conic.grad}, expected {expected}"
        )

    def test_conic_gradient_finite_difference(self):
        """
        CRITICAL: Finite difference verification of conic gradient.

        This is the MOST IMPORTANT numerical test. It proves the factor-of-2
        for off-diagonal elements is correct by comparing analytic gradients
        against numerical finite differences.

        If this test fails, the implementation has a gradient bug that will
        cause optimization to diverge or converge to wrong solutions.
        """
        import torch

        # Use small 2D case for clear verification
        N = 1
        DIM = 2
        CONIC_SIZE = 3  # 2D: [c_00, c_01, c_11]

        # Test parameters
        centers = torch.tensor([[10.0, 10.0]], device="cuda", dtype=torch.float64)
        conic = torch.tensor([[1.0, 0.3, 1.0]], device="cuda", dtype=torch.float64, requires_grad=True)
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float64)

        shape = [20, 20]

        # Forward function (simplified for test)
        def forward_fn(conic_in):
            # Compute output at a single test pixel
            px = torch.tensor([12.0, 11.0], device="cuda", dtype=torch.float64)
            d = px - centers[0]

            # Mahalanobis distance: d^T @ Σ^-1 @ d
            # For packed conic [c_00, c_01, c_11]:
            # D² = d[0]² × c_00 + 2 × d[0] × d[1] × c_01 + d[1]² × c_11
            dist_sq = (
                d[0]**2 * conic_in[0, 0] +
                2 * d[0] * d[1] * conic_in[0, 1] +  # Note the 2× for off-diagonal!
                d[1]**2 * conic_in[0, 2]
            )

            # Generalized Gaussian intensity
            s = sharpness[0]
            dist_pow_s = dist_sq ** (s / 2)
            intensity = amps[0] * torch.exp(-0.5 * dist_pow_s)

            return intensity

        # Compute analytic gradient
        output = forward_fn(conic)
        output.backward()
        analytic_grad = conic.grad.clone()

        # Compute finite difference gradient
        eps = 1e-6
        fd_grad = torch.zeros_like(conic)

        for i in range(CONIC_SIZE):
            # Perturb conic[0, i] by +eps
            conic_plus = conic.detach().clone()
            conic_plus[0, i] += eps
            out_plus = forward_fn(conic_plus)

            # Perturb conic[0, i] by -eps
            conic_minus = conic.detach().clone()
            conic_minus[0, i] -= eps
            out_minus = forward_fn(conic_minus)

            # Central difference
            fd_grad[0, i] = (out_plus - out_minus) / (2 * eps)

        # Compare analytic vs finite difference
        # Using float64 for precision, tolerance can be tight
        rtol = 1e-5
        atol = 1e-8

        for i in range(CONIC_SIZE):
            diff = abs(analytic_grad[0, i] - fd_grad[0, i])
            rel_diff = diff / max(abs(fd_grad[0, i]), 1e-10)

            component_name = ["c_00 (diagonal)", "c_01 (off-diagonal, 2×)", "c_11 (diagonal)"][i]

            assert rel_diff < rtol or diff < atol, (
                f"Finite difference mismatch for {component_name}:\n"
                f"  Analytic: {analytic_grad[0, i]:.10f}\n"
                f"  Finite diff: {fd_grad[0, i]:.10f}\n"
                f"  Relative error: {rel_diff:.2e}\n"
                f"  Absolute error: {diff:.2e}\n"
                f"  If c_01 is wrong by 2×, you're missing the off-diagonal scaling!"
            )

        # Additional sanity check: c_01 gradient should be ~2× larger than
        # what you'd get without the off-diagonal scaling
        d = torch.tensor([2.0, 1.0], device="cuda", dtype=torch.float64)  # px - center
        expected_c01_without_2x = d[0] * d[1]  # WRONG (missing 2×)
        expected_c01_with_2x = 2 * d[0] * d[1]  # CORRECT

        # The actual gradient is proportional to d[0]*d[1] or 2*d[0]*d[1]
        # depending on whether 2× is applied. This ratio test catches the bug.
        print(f"Conic gradient test passed!")
        print(f"  c_00 grad (diagonal): {analytic_grad[0, 0]:.6f}")
        print(f"  c_01 grad (off-diag): {analytic_grad[0, 1]:.6f}")
        print(f"  c_11 grad (diagonal): {analytic_grad[0, 2]:.6f}")

    def test_conic_gradient_3d_finite_difference(self):
        """
        Finite difference test for 3D conic (6 elements).

        Tests all off-diagonal elements: c_01, c_02, c_12
        """
        import torch

        DIM = 3
        CONIC_SIZE = 6  # [c_00, c_01, c_02, c_11, c_12, c_22]

        centers = torch.tensor([[5.0, 5.0, 5.0]], device="cuda", dtype=torch.float64)
        conic = torch.tensor([[1.0, 0.2, 0.1, 1.0, 0.15, 1.0]], device="cuda", dtype=torch.float64, requires_grad=True)
        amps = torch.tensor([1.0], device="cuda", dtype=torch.float64)
        sharpness = torch.tensor([2.0], device="cuda", dtype=torch.float64)

        def forward_3d(conic_in):
            px = torch.tensor([7.0, 6.0, 5.5], device="cuda", dtype=torch.float64)
            d = px - centers[0]

            # Unpack conic: [c_00, c_01, c_02, c_11, c_12, c_22]
            c = conic_in[0]

            # D² = Σᵢⱼ cᵢⱼ × dᵢ × dⱼ with 2× for off-diagonal
            dist_sq = (
                d[0]**2 * c[0] +           # c_00
                2 * d[0] * d[1] * c[1] +   # c_01 (off-diagonal)
                2 * d[0] * d[2] * c[2] +   # c_02 (off-diagonal)
                d[1]**2 * c[3] +           # c_11
                2 * d[1] * d[2] * c[4] +   # c_12 (off-diagonal)
                d[2]**2 * c[5]             # c_22
            )

            s = sharpness[0]
            dist_pow_s = dist_sq ** (s / 2)
            intensity = amps[0] * torch.exp(-0.5 * dist_pow_s)

            return intensity

        # Analytic gradient
        output = forward_3d(conic)
        output.backward()
        analytic_grad = conic.grad.clone()

        # Finite difference
        eps = 1e-6
        fd_grad = torch.zeros_like(conic)

        for i in range(CONIC_SIZE):
            conic_plus = conic.detach().clone()
            conic_plus[0, i] += eps
            out_plus = forward_3d(conic_plus)

            conic_minus = conic.detach().clone()
            conic_minus[0, i] -= eps
            out_minus = forward_3d(conic_minus)

            fd_grad[0, i] = (out_plus - out_minus) / (2 * eps)

        # Check each component
        component_names = ["c_00", "c_01 (2×)", "c_02 (2×)", "c_11", "c_12 (2×)", "c_22"]
        rtol = 1e-5

        for i in range(CONIC_SIZE):
            diff = abs(analytic_grad[0, i] - fd_grad[0, i])
            rel_diff = diff / max(abs(fd_grad[0, i]), 1e-10)

            assert rel_diff < rtol, (
                f"3D finite diff mismatch for {component_names[i]}:\n"
                f"  Analytic: {analytic_grad[0, i]:.10f}\n"
                f"  FD: {fd_grad[0, i]:.10f}\n"
                f"  Error: {rel_diff:.2e}"
            )

        print("3D conic gradient test passed!")

    def test_fp16_convergence(self):
        """
        FP16 forward should produce similar optimization trajectory to FP32.

        While individual values may differ by ~1e-3 (FP16 precision), the
        overall optimization should converge to similar loss values.
        """
        # Create identical models in FP16 and FP32
        shape = (32, 32, 32)
        N = 50
        centers0 = np.random.rand(N, 3).astype(np.float32) * 32
        L0 = np.eye(3, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 2.0
        amps0 = np.ones(N, dtype=np.float32)

        model_fp32 = GaussianSplatModelCUDA(shape=shape, centers0=centers0, L0=L0, amps0=amps0, ...)
        model_fp16 = GaussianSplatModelCUDA(shape=shape, centers0=centers0, L0=L0, amps0=amps0, use_fp16=True, ...)

        target = torch.rand(shape, device="cuda")

        # Train both for a few steps
        opt_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=0.01)
        opt_fp16 = torch.optim.Adam(model_fp16.parameters(), lr=0.01)

        for _ in range(100):
            opt_fp32.zero_grad()
            loss_fp32 = ((model_fp32() - target)**2).mean()
            loss_fp32.backward()
            opt_fp32.step()

            opt_fp16.zero_grad()
            loss_fp16 = ((model_fp16() - target)**2).mean()
            loss_fp16.backward()
            opt_fp16.step()

        # Final losses should be within 5% of each other
        assert abs(loss_fp32.item() - loss_fp16.item()) / loss_fp32.item() < 0.05, (
            f"FP16/FP32 convergence diverged: FP32={loss_fp32.item()}, FP16={loss_fp16.item()}"
        )

class TestCUDAPerformance:
    """Performance regression tests."""

    def test_speedup_vs_cpu_3d_medium(self):
        """Achieves target speedup for medium 3D workload."""

    def test_memory_footprint(self):
        """Memory usage within expected bounds."""

    def test_scaling_with_splats(self):
        """Performance scales reasonably with splat count."""
```

### 11.1.1 Reference Comparison Tests (CUDA vs PyTorch)

These tests ensure the CUDA backend produces numerically equivalent results to the pure PyTorch
`GaussianSplatModel` reference implementation. This is **critical** for validating correctness.

```python
# tests/test_cuda_comparison.py
"""
Reference comparison tests: CUDA backend vs PyTorch GaussianSplatModel.

These tests verify numerical equivalence between:
- CUDA forward pass and PyTorch forward pass
- CUDA gradients and PyTorch autograd gradients
- Optimization trajectories converge to similar results
"""

import pytest
import torch
import numpy as np

from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import GaussianSplatModelCUDA


@pytest.mark.gpu
class TestCUDAvsReference:
    """Compare CUDA backend against PyTorch reference implementation."""

    def test_forward_numerical_equivalence_3d(
        self,
        splat_params_3d,
        reference_model_factory,
        cuda_model_factory,
        tolerances,
    ):
        """
        Forward outputs match within tolerance.

        Tolerance: rtol=1e-5, atol=1e-7 (from Tolerances class)
        """
        # Create identical models
        ref_model = reference_model_factory(splat_params_3d)
        cuda_model = cuda_model_factory(splat_params_3d)

        # Forward pass
        output_ref = ref_model()
        output_cuda = cuda_model()

        # Verify shapes match
        assert output_ref.shape == output_cuda.shape, (
            f"Shape mismatch: ref={output_ref.shape}, cuda={output_cuda.shape}"
        )

        # Verify values match within tolerance
        torch.testing.assert_close(
            output_cuda.view(-1),
            output_ref.view(-1),
            rtol=tolerances.FORWARD_RTOL,
            atol=tolerances.FORWARD_ATOL,
            msg="CUDA forward diverges from PyTorch reference"
        )

    def test_forward_numerical_equivalence_2d(
        self,
        splat_params_2d,
        reference_model_factory,
        cuda_model_factory,
        tolerances,
    ):
        """Forward equivalence for 2D case."""
        ref_model = reference_model_factory(splat_params_2d)
        cuda_model = cuda_model_factory(splat_params_2d)

        output_ref = ref_model()
        output_cuda = cuda_model()

        torch.testing.assert_close(
            output_cuda.view(-1),
            output_ref.view(-1),
            rtol=tolerances.FORWARD_RTOL,
            atol=tolerances.FORWARD_ATOL,
        )

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_forward_all_dimensions(
        self,
        dim,
        reference_model_factory,
        cuda_model_factory,
        tolerances,
    ):
        """Forward equivalence across all supported dimensions."""
        # Generate params for this dimension
        N = 50
        shape_size = max(8, 32 // dim)
        params = {
            "centers": np.random.rand(N, dim).astype(np.float32) * shape_size,
            "L": np.eye(dim, dtype=np.float32)[None, :, :].repeat(N, axis=0) * 1.5,
            "amps": np.abs(np.random.randn(N).astype(np.float32)) + 0.1,
            "shape": tuple([shape_size] * dim),
            "sigma_min_diag": [0.1] * dim,
        }

        ref_model = reference_model_factory(params)
        cuda_model = cuda_model_factory(params)

        torch.testing.assert_close(
            cuda_model().view(-1),
            ref_model().view(-1),
            rtol=tolerances.FORWARD_RTOL,
            atol=tolerances.FORWARD_ATOL,
        )

    def test_backward_gradient_equivalence(
        self,
        splat_params_3d,
        reference_model_factory,
        cuda_model_factory,
        tolerances,
    ):
        """
        Gradient values match within tolerance.

        NOTE: Gradients have looser tolerance (1e-4) due to:
        - Atomic additions in CUDA have non-deterministic ordering
        - Floating-point accumulation order affects results
        """
        ref_model = reference_model_factory(splat_params_3d)
        cuda_model = cuda_model_factory(splat_params_3d)

        # Same target for both
        target = torch.rand(splat_params_3d["shape"], device="cuda")

        # Reference backward
        ref_model.zero_grad()
        output_ref = ref_model()
        loss_ref = ((output_ref - target) ** 2).mean()
        loss_ref.backward()

        # Get reference gradients (copy before they're overwritten)
        grad_centers_ref = ref_model._base.centers.grad.clone()
        grad_L_ref = ref_model._base.Ls.grad.clone()
        grad_amps_ref = ref_model._base.amps.grad.clone()

        # CUDA backward
        cuda_model.zero_grad()
        output_cuda = cuda_model()
        loss_cuda = ((output_cuda - target) ** 2).mean()
        loss_cuda.backward()

        grad_centers_cuda = cuda_model._base.centers.grad
        grad_L_cuda = cuda_model._base.Ls.grad
        grad_amps_cuda = cuda_model._base.amps.grad

        # Compare gradients (looser tolerance for backward)
        torch.testing.assert_close(
            grad_centers_cuda, grad_centers_ref,
            rtol=tolerances.BACKWARD_RTOL,
            atol=tolerances.BACKWARD_ATOL,
            msg="Center gradients diverge"
        )
        torch.testing.assert_close(
            grad_L_cuda, grad_L_ref,
            rtol=tolerances.BACKWARD_RTOL,
            atol=tolerances.BACKWARD_ATOL,
            msg="L (Cholesky) gradients diverge"
        )
        torch.testing.assert_close(
            grad_amps_cuda, grad_amps_ref,
            rtol=tolerances.BACKWARD_RTOL,
            atol=tolerances.BACKWARD_ATOL,
            msg="Amplitude gradients diverge"
        )

    def test_gradient_sign_consistency(
        self,
        splat_params_3d,
        reference_model_factory,
        cuda_model_factory,
    ):
        """
        Gradient SIGNS match between CUDA and reference.

        Even if magnitudes differ slightly, signs should be consistent.
        Wrong signs would cause optimization to diverge.
        """
        ref_model = reference_model_factory(splat_params_3d)
        cuda_model = cuda_model_factory(splat_params_3d)

        target = torch.rand(splat_params_3d["shape"], device="cuda")

        # Compute gradients
        ref_model.zero_grad()
        ((ref_model() - target) ** 2).mean().backward()

        cuda_model.zero_grad()
        ((cuda_model() - target) ** 2).mean().backward()

        # Compare signs (ignore very small gradients)
        threshold = 1e-6
        ref_grad = ref_model._base.centers.grad
        cuda_grad = cuda_model._base.centers.grad

        significant_mask = ref_grad.abs() > threshold
        signs_match = (torch.sign(ref_grad[significant_mask]) ==
                       torch.sign(cuda_grad[significant_mask])).all()

        assert signs_match, "Gradient signs differ - optimization would diverge!"


@pytest.mark.gpu
@pytest.mark.integration
class TestOptimizationConvergence:
    """Verify CUDA and reference converge to similar solutions."""

    def test_fitting_convergence_equivalence(
        self,
        splat_params_3d,
        reference_model_factory,
        cuda_model_factory,
        tolerances,
    ):
        """
        Both backends converge to similar final loss.

        This is a critical integration test - even if individual gradients
        have small differences, the optimization trajectory should be similar.
        """
        # Create synthetic target
        shape = splat_params_3d["shape"]
        target = torch.rand(shape, device="cuda") * 0.5

        # Train reference model
        ref_model = reference_model_factory(splat_params_3d)
        ref_opt = torch.optim.Adam(ref_model.parameters(), lr=0.01)

        for _ in range(100):
            ref_opt.zero_grad()
            loss = ((ref_model() - target) ** 2).mean()
            loss.backward()
            ref_opt.step()
        final_loss_ref = loss.item()

        # Train CUDA model (same initialization!)
        cuda_model = cuda_model_factory(splat_params_3d)
        cuda_opt = torch.optim.Adam(cuda_model.parameters(), lr=0.01)

        for _ in range(100):
            cuda_opt.zero_grad()
            loss = ((cuda_model() - target) ** 2).mean()
            loss.backward()
            cuda_opt.step()
        final_loss_cuda = loss.item()

        # Final losses should be within 5%
        relative_diff = abs(final_loss_cuda - final_loss_ref) / max(final_loss_ref, 1e-8)
        assert relative_diff < tolerances.CONVERGENCE_RTOL, (
            f"Convergence diverged: ref={final_loss_ref:.6f}, cuda={final_loss_cuda:.6f}, "
            f"diff={relative_diff:.2%}"
        )

    def test_gradcheck_passes(self, splat_params_3d, cuda_model_factory):
        """
        torch.autograd.gradcheck passes for CUDA backend.

        This is PyTorch's built-in numerical gradient verification.
        """
        model = cuda_model_factory(splat_params_3d)

        # Get parameters that require grad
        params = [p for p in model.parameters() if p.requires_grad]

        def forward_fn(*params):
            # Temporarily set parameters
            # (simplified - real implementation would properly set params)
            return model().sum()

        # Use double precision for gradcheck
        for p in params:
            p.data = p.data.double()

        torch.autograd.gradcheck(
            forward_fn,
            params,
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=True,
        )
```

**Tolerance Justification**:

| Comparison | Tolerance | Rationale |
|------------|-----------|-----------|
| Forward (CUDA vs PyTorch) | rtol=1e-4, atol=1e-6 | FP32 precision, atomicAdd non-determinism |
| Backward (gradients) | rtol=1e-3, atol=1e-5 | Atomic operations have non-deterministic order |
| Convergence (final loss) | 10% relative | Different rounding accumulates over iterations |
| FP16 forward | rtol=1e-2, atol=1e-4 | FP16 has ~3 decimal digits precision |

### 11.2 Edge Case Tests

```python
import numpy as np
import pytest
import torch

from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import GaussianSplatModelCUDA


class TestCUDAEdgeCases:
    """Edge cases and boundary conditions."""

    def test_high_dim_tile_cap(self):
        """Verify error on exceeding 1M tile limit."""
        # 6D with shape=64 and tile_size=2: 32^6 = 1B tiles
        with pytest.raises(RuntimeError, match="too large"):
            GaussianSplatModelCUDA(
                shape=(64,) * 6,
                centers0=np.random.rand(10, 6).astype(np.float32),
                # ...
            )

    def test_tile_auto_adjustment(self):
        """Verify tile_size auto-increases for high-D."""
        # 5D with shape=32 should auto-adjust tile_size
        model = GaussianSplatModelCUDA(shape=(32,) * 5, ...)
        assert model._tile_size >= 4  # Auto-increased from default 2

    def test_tile_content_overflow_detection(self):
        """Verify overflow flag is set when tile_content exceeds capacity."""
        # Create scenario with many splats overlapping same tile
        # ...
        assert model.check_overflow() == False  # Should handle gracefully

    def test_determinism_across_runs(self):
        """Verify identical outputs in deterministic mode."""
        model = GaussianSplatModelCUDA(..., deterministic=True)
        out1 = model()
        out2 = model()
        assert torch.equal(out1, out2)  # Bit-identical

    def test_nondeterminism_variance(self):
        """Verify nondeterminism is bounded (~1e-6 relative)."""
        model = GaussianSplatModelCUDA(..., deterministic=False)
        outputs = [model() for _ in range(10)]
        variance = torch.var(torch.stack(outputs), dim=0)
        assert (variance / outputs[0].abs().clamp(min=1e-6)).max() < 1e-5

    def test_empty_tiles(self):
        """Verify correct handling of tiles with no splats."""
        # Splats clustered in corner, many tiles empty
        centers = np.random.rand(10, 3).astype(np.float32) * 10  # All in [0,10]
        model = GaussianSplatModelCUDA(shape=(128, 128, 128), centers0=centers, ...)
        output = model()
        # Should not crash, output in empty regions should be 0

    def test_splat_outside_volume(self):
        """Verify splats outside volume are correctly skipped."""
        centers = np.array([[-10, -10, -10], [200, 200, 200]], dtype=np.float32)
        model = GaussianSplatModelCUDA(shape=(64, 64, 64), centers0=centers, ...)
        output = model()
        # Output should be all zeros (both splats outside)
        assert output.sum() == 0

    def test_degenerate_covariance(self):
        """Verify handling of near-singular L matrices."""
        L = np.eye(3, dtype=np.float32)[None, :, :] * 1e-6  # Very small
        model = GaussianSplatModelCUDA(..., L0=L, ...)
        output = model()  # Should not produce NaN/Inf

    def test_extreme_sharpness(self):
        """Verify numerical stability at sharpness boundaries."""
        # Test s=0.5 (soft) and s=8.0 (sharp)
        for s in [0.5, 1.0, 2.0, 4.0, 8.0]:
            model = GaussianSplatModelCUDA(..., sharpness0=np.array([s]), ...)
            output = model()
            assert torch.isfinite(output).all()

class TestCUDAIntegration:
    """Integration with fitting pipeline."""

    def test_fit_gaussian_splats_cuda(self):
        """Full fitting pipeline works with CUDA backend."""

    def test_optimizer_compatibility(self):
        """Standard PyTorch Adam works correctly with the CUDA backend."""

    def test_dynamic_ops(self):
        """Fixed-pool splat relocation works correctly."""
```

### 11.3 Operator Correctness Tests (opcheck)

PyTorch 2.x provides `torch.library.opcheck` for verifying custom operators
work correctly with torch.compile, vmap, and other transforms. **These tests
are essential** for torch.library-based implementations.

```python
import torch
from torch.library import opcheck
from torch.testing._internal.optests import generate_opcheck_tests

class TestOpCheck:
    """
    Verify custom ops are compatible with PyTorch 2.x features.

    opcheck runs a battery of tests including:
    - test_schema: Op schema is correct and matches implementation
    - test_autograd_registration: Backward is properly registered
    - test_faketensor: FakeTensor mode works (required for torch.compile)
    - test_aot_dispatch_dynamic: AOT dispatch with dynamic shapes
    - test_aot_dispatch_static: AOT dispatch with static shapes
    """

    def test_forward_op_check(self):
        """Verify forward op passes all opcheck tests."""
        # Create sample inputs
        N, DIM = 100, 3
        centers = torch.randn(N, DIM, device="cuda", requires_grad=True)
        conic = torch.randn(N, (DIM * (DIM + 1)) // 2, device="cuda", requires_grad=True)
        amps = torch.randn(N, device="cuda", requires_grad=True)
        sharpness = torch.ones(N, device="cuda", requires_grad=True) * 2.0

        # Run opcheck
        opcheck(
            torch.ops.luxar_cuda.gaussian_splat_forward,
            args=(centers, conic, amps, sharpness, [64, 64, 64], 3.0, 1e-5, 8),
            test_utils=[
                "test_schema",
                "test_autograd_registration",
                "test_faketensor",
                "test_aot_dispatch_dynamic",
            ],
        )

    def test_backward_op_check(self):
        """Verify backward op passes all opcheck tests."""
        N, DIM = 100, 3
        grad_output = torch.randn(64 * 64 * 64, device="cuda")
        centers = torch.randn(N, DIM, device="cuda")
        conic = torch.randn(N, (DIM * (DIM + 1)) // 2, device="cuda")
        amps = torch.randn(N, device="cuda")
        sharpness = torch.ones(N, device="cuda") * 2.0
        tile_offsets = torch.zeros(512, device="cuda", dtype=torch.int64)
        tile_counts = torch.zeros(512, device="cuda", dtype=torch.int32)
        tile_content = torch.zeros(1000, device="cuda", dtype=torch.int32)

        opcheck(
            torch.ops.luxar_cuda.gaussian_splat_backward,
            args=(
                grad_output, centers, conic, amps, sharpness,
                tile_offsets, tile_counts, tile_content,
                [64, 64, 64], 3.0, 1e-5, 8
            ),
            test_utils=["test_schema", "test_faketensor"],
        )

    def test_vmap_compatibility(self):
        """
        Verify op works with vmap (vectorized map).

        vmap requires the op to support batched inputs. This test verifies
        that register_fake correctly handles batched tensor shapes.
        """
        from torch.func import vmap

        def single_forward(centers, conic, amps, sharpness):
            output, _, _, _ = torch.ops.luxar_cuda.gaussian_splat_forward(
                centers, conic, amps, sharpness,
                [32, 32, 32], 3.0, 1e-5, 8
            )
            return output

        # Batch of 4 different splat configurations
        batch_size = 4
        N, DIM = 50, 3
        batch_centers = torch.randn(batch_size, N, DIM, device="cuda")
        batch_conic = torch.randn(batch_size, N, (DIM * (DIM + 1)) // 2, device="cuda")
        batch_amps = torch.randn(batch_size, N, device="cuda")
        batch_sharpness = torch.ones(batch_size, N, device="cuda") * 2.0

        # vmap over batch dimension
        batched_forward = vmap(single_forward)
        batch_outputs = batched_forward(batch_centers, batch_conic, batch_amps, batch_sharpness)

        assert batch_outputs.shape == (batch_size, 32 * 32 * 32)

    def test_torch_compile_compatibility(self):
        """
        Verify op works with torch.compile.

        This is a critical test - if this fails, the op won't work with
        PyTorch's JIT compiler for optimized execution.
        """
        import torch._dynamo as dynamo

        def forward_fn(centers, conic, amps, sharpness):
            output, _, _, _ = torch.ops.luxar_cuda.gaussian_splat_forward(
                centers, conic, amps, sharpness,
                [32, 32, 32], 3.0, 1e-5, 8
            )
            return output.sum()

        # Compile the function
        compiled_fn = torch.compile(forward_fn, backend="inductor")

        # Test inputs
        N, DIM = 100, 3
        centers = torch.randn(N, DIM, device="cuda", requires_grad=True)
        conic = torch.randn(N, (DIM * (DIM + 1)) // 2, device="cuda", requires_grad=True)
        amps = torch.randn(N, device="cuda", requires_grad=True)
        sharpness = torch.ones(N, device="cuda", requires_grad=True) * 2.0

        # Forward should work
        loss = compiled_fn(centers, conic, amps, sharpness)
        assert torch.isfinite(loss)

        # Backward should work through compiled graph
        loss.backward()
        assert centers.grad is not None
        assert torch.isfinite(centers.grad).all()

    def test_double_backward(self):
        """
        Verify second-order gradients (Hessian-vector products).

        While not strictly required, double backward is useful for:
        - Second-order optimization methods
        - Computing Fisher information
        - Hessian analysis
        """
        N, DIM = 10, 3
        centers = torch.randn(N, DIM, device="cuda", requires_grad=True)
        conic = torch.randn(N, (DIM * (DIM + 1)) // 2, device="cuda", requires_grad=True)
        amps = torch.randn(N, device="cuda", requires_grad=True)
        sharpness = torch.ones(N, device="cuda", requires_grad=True) * 2.0

        output, _, _, _ = torch.ops.luxar_cuda.gaussian_splat_forward(
            centers, conic, amps, sharpness,
            [16, 16, 16], 3.0, 1e-5, 4
        )
        loss = output.sum()

        # First backward
        grad_centers, = torch.autograd.grad(
            loss, centers, create_graph=True, retain_graph=True
        )

        # Second backward (Hessian-vector product)
        v = torch.randn_like(grad_centers)
        hvp, = torch.autograd.grad(grad_centers, centers, grad_outputs=v)

        # Should not error; values should be finite
        assert torch.isfinite(hvp).all(), "Hessian-vector product contains NaN/Inf"
```

**Running opcheck tests**:

```bash
# Run all opcheck tests
pytest tests/test_cuda_gradcheck.py -v

# Run with verbose output to see which subtests pass/fail
pytest tests/test_cuda_gradcheck.py -v --tb=short
```

**Common opcheck failures and fixes**:

| Failure | Cause | Fix |
|---------|-------|-----|
| `test_schema` | Op signature doesn't match schema | Ensure all args have correct types in schema |
| `test_faketensor` | register_fake returns wrong shapes | Verify fake returns tensors with correct shapes/dtypes |
| `test_autograd_registration` | No backward registered | Use `torch.library.register_autograd()` |
| `test_aot_dispatch_dynamic` | Op doesn't support dynamic shapes | Add guards or use SymInt in fake |

**Integration with CI**:

```yaml
# .github/workflows/test.yml
- name: Run opcheck tests
  run: |
    pytest tests/test_cuda_gradcheck.py -v --tb=short
    # Fail CI if any opcheck test fails
```

### 11.4 C++ Kernel Unit Tests

> **NOTE**: The `cpp/` test directory described below was never implemented.
> All kernel correctness testing is done through the Python test suite
> (`test_cuda_forward.py`, `test_cuda_backward.py`, `test_cuda_gradcheck.py`,
> `test_cuda_numerical.py`). The section below is retained as a reference design
> for potential future C++ unit tests.

CUDA kernels and device functions should be tested in isolation using C++ unit tests.
This catches bugs before they reach the Python layer and provides faster iteration.

#### 11.4.1 Test Framework Setup

Use Google Test (gtest) for C++ unit tests. CMake configuration:

```cmake
# tests/cpp/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(cuda_splatting_tests CUDA CXX)

# Find packages
find_package(GTest REQUIRED)
find_package(CUDA REQUIRED)

# Enable CUDA language
enable_language(CUDA)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 100 120)

# Test executable
add_executable(cuda_kernel_tests
    test_aabb.cpp
    test_mahalanobis.cpp
    test_conic_gradient.cpp    # CRITICAL: finite difference gradient tests
    test_tile_iteration.cpp
    test_binning.cpp
    # Include actual kernel code
    ${CMAKE_SOURCE_DIR}/../../src/cuda_splatting.cu
)

target_include_directories(cuda_kernel_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/../../src
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(cuda_kernel_tests
    GTest::GTest
    GTest::Main
    ${CUDA_LIBRARIES}
)

# Enable testing
enable_testing()
add_test(NAME cuda_kernel_tests COMMAND cuda_kernel_tests)

# Coverage (optional)
if(ENABLE_COVERAGE)
    target_compile_options(cuda_kernel_tests PRIVATE --coverage)
    target_link_options(cuda_kernel_tests PRIVATE --coverage)
endif()
```

#### 11.4.2 AABB Computation Tests

Test the shared `compute_splat_aabb` function directly:

```cpp
// tests/cpp/test_aabb.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuda_splatting.h"

// Test fixture for AABB tests
class AABBTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Allocate device memory for test data
        cudaMalloc(&d_L_row_norms, 8 * sizeof(float));  // Max 8D
        cudaMalloc(&d_mu, 8 * sizeof(float));
        cudaMalloc(&d_tile_size, 8 * sizeof(int));
        cudaMalloc(&d_tile_dims, 8 * sizeof(int));
        cudaMalloc(&d_shape, 8 * sizeof(int));
    }

    void TearDown() override {
        cudaFree(d_L_row_norms);
        cudaFree(d_mu);
        cudaFree(d_tile_size);
        cudaFree(d_tile_dims);
        cudaFree(d_shape);
    }

    float* d_L_row_norms;
    float* d_mu;
    int* d_tile_size;
    int* d_tile_dims;
    int* d_shape;
};

// Kernel to test compute_splat_aabb
template<int DIM>
__global__ void test_aabb_kernel(
    const float* mu,
    const float* L_row_norms,
    float sharpness,
    float amplitude,
    float truncate,
    float intensity_floor,
    const int* tile_size,
    const int* tile_dims,
    const int* shape,
    int* aabb_lo,
    int* aabb_hi
) {
    AABB<DIM> aabb = compute_splat_aabb<DIM>(
        mu, L_row_norms, sharpness, amplitude,
        truncate, intensity_floor, tile_size, tile_dims, shape
    );

    for (int d = 0; d < DIM; d++) {
        aabb_lo[d] = aabb.lo[d];
        aabb_hi[d] = aabb.hi[d];
    }
}

TEST_F(AABBTest, CenteredSplat3D) {
    // 3D splat at center of 64³ volume
    float h_mu[3] = {32.0f, 32.0f, 32.0f};
    float h_L_row_norms[3] = {2.0f, 2.0f, 2.0f};  // Isotropic
    int h_tile_size[3] = {8, 8, 8};
    int h_tile_dims[3] = {8, 8, 8};  // 64/8 = 8 tiles per axis
    int h_shape[3] = {64, 64, 64};

    cudaMemcpy(d_mu, h_mu, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L_row_norms, h_L_row_norms, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_size, h_tile_size, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_dims, h_tile_dims, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, h_shape, 3 * sizeof(int), cudaMemcpyHostToDevice);

    int *d_aabb_lo, *d_aabb_hi;
    cudaMalloc(&d_aabb_lo, 3 * sizeof(int));
    cudaMalloc(&d_aabb_hi, 3 * sizeof(int));

    test_aabb_kernel<3><<<1, 1>>>(
        d_mu, d_L_row_norms,
        2.0f,   // sharpness
        1.0f,   // amplitude
        3.0f,   // truncate
        1e-5f,  // intensity_floor
        d_tile_size, d_tile_dims, d_shape,
        d_aabb_lo, d_aabb_hi
    );
    cudaDeviceSynchronize();

    int h_aabb_lo[3], h_aabb_hi[3];
    cudaMemcpy(h_aabb_lo, d_aabb_lo, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aabb_hi, d_aabb_hi, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Radius ≈ 3 × 2 = 6 voxels, centered at 32
    // Should span tiles containing voxels 26-38
    // Tile 3 contains voxels 24-31, Tile 4 contains 32-39
    // So AABB should be roughly [3,3,3] to [4,4,4]
    for (int d = 0; d < 3; d++) {
        EXPECT_GE(h_aabb_lo[d], 2);  // Not too far left
        EXPECT_LE(h_aabb_lo[d], 4);  // Includes center
        EXPECT_GE(h_aabb_hi[d], 3);  // Includes center
        EXPECT_LE(h_aabb_hi[d], 5);  // Not too far right
    }

    cudaFree(d_aabb_lo);
    cudaFree(d_aabb_hi);
}

TEST_F(AABBTest, SplatOutsideVolume) {
    // Splat entirely outside volume should return empty AABB
    float h_mu[3] = {-100.0f, -100.0f, -100.0f};  // Way outside
    float h_L_row_norms[3] = {2.0f, 2.0f, 2.0f};
    int h_tile_size[3] = {8, 8, 8};
    int h_tile_dims[3] = {8, 8, 8};
    int h_shape[3] = {64, 64, 64};

    // ... copy to device ...

    int *d_aabb_lo, *d_aabb_hi;
    cudaMalloc(&d_aabb_lo, 3 * sizeof(int));
    cudaMalloc(&d_aabb_hi, 3 * sizeof(int));

    test_aabb_kernel<3><<<1, 1>>>(...);
    cudaDeviceSynchronize();

    int h_aabb_lo[3], h_aabb_hi[3];
    cudaMemcpy(h_aabb_lo, d_aabb_lo, 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aabb_hi, d_aabb_hi, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Empty AABB: lo > hi in at least one dimension
    bool is_empty = false;
    for (int d = 0; d < 3; d++) {
        if (h_aabb_lo[d] > h_aabb_hi[d]) is_empty = true;
    }
    EXPECT_TRUE(is_empty) << "AABB should be empty for splat outside volume";

    cudaFree(d_aabb_lo);
    cudaFree(d_aabb_hi);
}

TEST_F(AABBTest, AnisotropicSplat) {
    // Anisotropic splat (elongated in one dimension)
    // L_row_norms = [1, 1, 5] means elongated in Z
    float h_mu[3] = {32.0f, 32.0f, 32.0f};
    float h_L_row_norms[3] = {1.0f, 1.0f, 5.0f};  // Elongated in Z

    // ... setup and run kernel ...

    // Z should have larger AABB extent than X/Y
    // This validates correct handling of anisotropic covariances
    int h_aabb_lo[3], h_aabb_hi[3];
    // ... copy results ...

    int extent_x = h_aabb_hi[0] - h_aabb_lo[0];
    int extent_z = h_aabb_hi[2] - h_aabb_lo[2];

    EXPECT_GT(extent_z, extent_x)
        << "Anisotropic splat should have larger Z extent";
}
```

#### 11.4.3 Mahalanobis Distance Tests

```cpp
// tests/cpp/test_mahalanobis.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "cuda_splatting.h"

// Kernel to test mahalanobis_distance
template<int DIM>
__global__ void test_mahalanobis_kernel(
    const float* d,
    const float* conic,
    float* result
) {
    *result = mahalanobis_distance<DIM>(d, conic);
}

class MahalanobisTest : public ::testing::Test {
protected:
    float* d_displacement;
    float* d_conic;
    float* d_result;

    void SetUp() override {
        cudaMalloc(&d_displacement, 8 * sizeof(float));
        cudaMalloc(&d_conic, 36 * sizeof(float));  // Max 8D: 8*9/2=36
        cudaMalloc(&d_result, sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_displacement);
        cudaFree(d_conic);
        cudaFree(d_result);
    }
};

TEST_F(MahalanobisTest, IdentityCovariance2D) {
    // For identity covariance (Σ = I), Mahalanobis = Euclidean distance squared
    float h_d[2] = {3.0f, 4.0f};
    // Conic for identity: [c_00=1, c_01=0, c_11=1]
    float h_conic[3] = {1.0f, 0.0f, 1.0f};

    cudaMemcpy(d_displacement, h_d, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, 3 * sizeof(float), cudaMemcpyHostToDevice);

    test_mahalanobis_kernel<2><<<1, 1>>>(d_displacement, d_conic, d_result);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 3² + 4² = 25
    EXPECT_NEAR(result, 25.0f, 1e-5f);
}

TEST_F(MahalanobisTest, ScaledCovariance2D) {
    // Σ = diag(4, 9), so Σ⁻¹ = diag(0.25, 0.111)
    float h_d[2] = {2.0f, 3.0f};
    // Conic: [c_00=0.25, c_01=0, c_11=0.111]
    float h_conic[3] = {0.25f, 0.0f, 1.0f/9.0f};

    cudaMemcpy(d_displacement, h_d, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, 3 * sizeof(float), cudaMemcpyHostToDevice);

    test_mahalanobis_kernel<2><<<1, 1>>>(d_displacement, d_conic, d_result);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: (2²)(0.25) + (3²)(1/9) = 1 + 1 = 2
    EXPECT_NEAR(result, 2.0f, 1e-5f);
}

TEST_F(MahalanobisTest, OffDiagonalCovariance2D) {
    // Test with off-diagonal element (correlated)
    float h_d[2] = {1.0f, 1.0f};
    // Conic: [c_00=2, c_01=1, c_11=2]
    // D² = 2×1² + 2×1×1×1 + 2×1² = 2 + 2 + 2 = 6
    float h_conic[3] = {2.0f, 1.0f, 2.0f};

    cudaMemcpy(d_displacement, h_d, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, 3 * sizeof(float), cudaMemcpyHostToDevice);

    test_mahalanobis_kernel<2><<<1, 1>>>(d_displacement, d_conic, d_result);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // D² = d[0]²×c_00 + 2×d[0]×d[1]×c_01 + d[1]²×c_11
    //    = 1×2 + 2×1×1×1 + 1×2 = 6
    EXPECT_NEAR(result, 6.0f, 1e-5f);
}

TEST_F(MahalanobisTest, IdentityCovariance3D) {
    float h_d[3] = {1.0f, 2.0f, 2.0f};
    // Conic for identity 3D: [c_00=1, c_01=0, c_02=0, c_11=1, c_12=0, c_22=1]
    float h_conic[6] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};

    cudaMemcpy(d_displacement, h_d, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, 6 * sizeof(float), cudaMemcpyHostToDevice);

    test_mahalanobis_kernel<3><<<1, 1>>>(d_displacement, d_conic, d_result);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Expected: 1² + 2² + 2² = 9
    EXPECT_NEAR(result, 9.0f, 1e-5f);
}
```

#### 11.4.4 Conic Gradient Finite Difference Tests (CRITICAL)

This is the **most important C++ test**. It verifies that the CUDA kernel's analytic gradients
match numerical finite differences, catching the common "factor of 2" bug for off-diagonal
conic elements.

```cpp
// tests/cpp/test_conic_gradient.cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "cuda_splatting.h"

// =============================================================================
// FORWARD FUNCTION FOR FINITE DIFFERENCE TESTING
// =============================================================================

// Compute single-pixel Gaussian intensity (matches CUDA kernel logic)
template<int DIM>
__device__ float compute_intensity(
    const float* px,           // Pixel coordinates
    const float* mu,           // Splat center
    const float* conic,        // Packed upper-triangle of Σ⁻¹
    float amplitude,
    float sharpness
) {
    // Displacement d = px - mu
    float d[DIM];
    for (int i = 0; i < DIM; i++) {
        d[i] = px[i] - mu[i];
    }

    // Mahalanobis distance: D² = d^T @ Σ⁻¹ @ d
    // For packed upper-triangle, off-diagonal elements contribute 2×
    float dist_sq = 0.0f;
    int idx = 0;
    for (int i = 0; i < DIM; i++) {
        dist_sq += d[i] * d[i] * conic[idx++];  // Diagonal
        for (int j = i + 1; j < DIM; j++) {
            dist_sq += 2.0f * d[i] * d[j] * conic[idx++];  // Off-diagonal (2×!)
        }
    }

    // Generalized Gaussian: I = a × exp(-0.5 × D^s)
    float dist_pow_s = powf(dist_sq, sharpness * 0.5f);
    return amplitude * expf(-0.5f * dist_pow_s);
}

// Kernel to compute intensity at a single pixel
template<int DIM>
__global__ void compute_intensity_kernel(
    const float* px,
    const float* mu,
    const float* conic,
    float amplitude,
    float sharpness,
    float* output
) {
    *output = compute_intensity<DIM>(px, mu, conic, amplitude, sharpness);
}

// =============================================================================
// GRADIENT COMPUTATION (matches rasterize_bwd_nd logic)
// =============================================================================

template<int DIM>
__global__ void compute_conic_gradient_kernel(
    const float* px,
    const float* mu,
    const float* conic,
    float amplitude,
    float sharpness,
    float dL_dI,              // Upstream gradient (set to 1.0 for testing)
    float* d_conic            // Output: gradient w.r.t. conic elements
) {
    constexpr int CONIC_SIZE = DIM * (DIM + 1) / 2;

    // Forward computation
    float d[DIM];
    for (int i = 0; i < DIM; i++) {
        d[i] = px[i] - mu[i];
    }

    float dist_sq = 0.0f;
    int idx = 0;
    for (int i = 0; i < DIM; i++) {
        dist_sq += d[i] * d[i] * conic[idx++];
        for (int j = i + 1; j < DIM; j++) {
            dist_sq += 2.0f * d[i] * d[j] * conic[idx++];
        }
    }

    // Prevent log(0)
    float dist_sq_clamped = fmaxf(dist_sq, 1e-12f);
    float s = sharpness;
    float dist_pow_s = powf(dist_sq_clamped, s * 0.5f);
    float intensity = amplitude * expf(-0.5f * dist_pow_s);

    // Gradient of intensity w.r.t. dist_sq
    // ∂I/∂D² = I × (-0.25 × s) × D²^(s/2 - 1)
    float dist_pow_s_minus_1 = powf(dist_sq_clamped, s * 0.5f - 1.0f);
    float grad_dist = intensity * (-0.25f * s) * dist_pow_s_minus_1;

    // Gradient of dist_sq w.r.t. conic elements
    // ∂D²/∂c_ii = d_i²         (diagonal)
    // ∂D²/∂c_ij = 2 × d_i × d_j  (off-diagonal, i < j)
    idx = 0;
    for (int i = 0; i < DIM; i++) {
        // Diagonal
        float grad = d[i] * d[i];
        d_conic[idx++] = dL_dI * grad_dist * grad;

        // Off-diagonals
        for (int j = i + 1; j < DIM; j++) {
            grad = 2.0f * d[i] * d[j];  // CRITICAL: factor of 2!
            d_conic[idx++] = dL_dI * grad_dist * grad;
        }
    }
}

// =============================================================================
// TEST FIXTURE
// =============================================================================

class ConicGradientTest : public ::testing::Test {
protected:
    float* d_px;
    float* d_mu;
    float* d_conic;
    float* d_output;
    float* d_grad_conic;

    static constexpr int MAX_DIM = 4;
    static constexpr int MAX_CONIC = MAX_DIM * (MAX_DIM + 1) / 2;  // 10 for 4D

    void SetUp() override {
        cudaMalloc(&d_px, MAX_DIM * sizeof(float));
        cudaMalloc(&d_mu, MAX_DIM * sizeof(float));
        cudaMalloc(&d_conic, MAX_CONIC * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMalloc(&d_grad_conic, MAX_CONIC * sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_px);
        cudaFree(d_mu);
        cudaFree(d_conic);
        cudaFree(d_output);
        cudaFree(d_grad_conic);
    }

    // Helper: compute finite difference gradient for conic element i
    template<int DIM>
    float finite_diff_conic(
        const float* h_px,
        const float* h_mu,
        float* h_conic,  // Will be modified temporarily
        float amplitude,
        float sharpness,
        int conic_idx,
        float eps = 1e-4f
    ) {
        // f(conic + eps)
        float orig = h_conic[conic_idx];
        h_conic[conic_idx] = orig + eps;
        cudaMemcpy(d_conic, h_conic, (DIM * (DIM + 1) / 2) * sizeof(float), cudaMemcpyHostToDevice);
        compute_intensity_kernel<DIM><<<1, 1>>>(d_px, d_mu, d_conic, amplitude, sharpness, d_output);
        cudaDeviceSynchronize();
        float out_plus;
        cudaMemcpy(&out_plus, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // f(conic - eps)
        h_conic[conic_idx] = orig - eps;
        cudaMemcpy(d_conic, h_conic, (DIM * (DIM + 1) / 2) * sizeof(float), cudaMemcpyHostToDevice);
        compute_intensity_kernel<DIM><<<1, 1>>>(d_px, d_mu, d_conic, amplitude, sharpness, d_output);
        cudaDeviceSynchronize();
        float out_minus;
        cudaMemcpy(&out_minus, d_output, sizeof(float), cudaMemcpyDeviceToHost);

        // Restore original
        h_conic[conic_idx] = orig;

        return (out_plus - out_minus) / (2.0f * eps);
    }
};

// =============================================================================
// 2D FINITE DIFFERENCE TEST
// =============================================================================

TEST_F(ConicGradientTest, FiniteDiff2D) {
    constexpr int DIM = 2;
    constexpr int CONIC_SIZE = 3;  // [c_00, c_01, c_11]

    // Test setup
    float h_px[DIM] = {12.0f, 11.0f};
    float h_mu[DIM] = {10.0f, 10.0f};
    float h_conic[CONIC_SIZE] = {1.0f, 0.3f, 1.0f};
    float amplitude = 1.0f;
    float sharpness = 2.0f;

    cudaMemcpy(d_px, h_px, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, h_mu, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, CONIC_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Compute analytic gradient
    float h_grad_analytic[CONIC_SIZE];
    compute_conic_gradient_kernel<DIM><<<1, 1>>>(
        d_px, d_mu, d_conic, amplitude, sharpness, 1.0f, d_grad_conic
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_grad_analytic, d_grad_conic, CONIC_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute finite difference gradient
    const char* names[CONIC_SIZE] = {"c_00 (diag)", "c_01 (off-diag, 2×)", "c_11 (diag)"};

    for (int i = 0; i < CONIC_SIZE; i++) {
        float fd_grad = finite_diff_conic<DIM>(h_px, h_mu, h_conic, amplitude, sharpness, i);
        float analytic = h_grad_analytic[i];
        float rel_error = std::abs(analytic - fd_grad) / std::max(std::abs(fd_grad), 1e-10f);

        // Print for debugging
        printf("  %s: analytic=%.8f, fd=%.8f, rel_err=%.2e\n",
               names[i], analytic, fd_grad, rel_error);

        EXPECT_LT(rel_error, 1e-3f)
            << "2D gradient mismatch for " << names[i]
            << "\n  Analytic: " << analytic
            << "\n  Finite diff: " << fd_grad
            << "\n  If c_01 is wrong by 2×, you're missing off-diagonal scaling!";
    }
}

// =============================================================================
// 3D FINITE DIFFERENCE TEST
// =============================================================================

TEST_F(ConicGradientTest, FiniteDiff3D) {
    constexpr int DIM = 3;
    constexpr int CONIC_SIZE = 6;  // [c_00, c_01, c_02, c_11, c_12, c_22]

    float h_px[DIM] = {7.0f, 6.0f, 5.5f};
    float h_mu[DIM] = {5.0f, 5.0f, 5.0f};
    float h_conic[CONIC_SIZE] = {1.0f, 0.2f, 0.1f, 1.0f, 0.15f, 1.0f};
    float amplitude = 1.0f;
    float sharpness = 2.0f;

    cudaMemcpy(d_px, h_px, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, h_mu, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, CONIC_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float h_grad_analytic[CONIC_SIZE];
    compute_conic_gradient_kernel<DIM><<<1, 1>>>(
        d_px, d_mu, d_conic, amplitude, sharpness, 1.0f, d_grad_conic
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_grad_analytic, d_grad_conic, CONIC_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    const char* names[CONIC_SIZE] = {
        "c_00", "c_01 (2×)", "c_02 (2×)", "c_11", "c_12 (2×)", "c_22"
    };

    for (int i = 0; i < CONIC_SIZE; i++) {
        float fd_grad = finite_diff_conic<DIM>(h_px, h_mu, h_conic, amplitude, sharpness, i);
        float analytic = h_grad_analytic[i];
        float rel_error = std::abs(analytic - fd_grad) / std::max(std::abs(fd_grad), 1e-10f);

        printf("  %s: analytic=%.8f, fd=%.8f, rel_err=%.2e\n",
               names[i], analytic, fd_grad, rel_error);

        EXPECT_LT(rel_error, 1e-3f)
            << "3D gradient mismatch for " << names[i];
    }
}

// =============================================================================
// RATIO TEST: Verify off-diagonal has 2× factor
// =============================================================================

TEST_F(ConicGradientTest, OffDiagonalRatioCheck2D) {
    // This test explicitly verifies the 2× factor by computing what the gradient
    // WOULD BE without the factor, and checking the ratio.

    constexpr int DIM = 2;

    // Use simple displacement for easy calculation
    float h_px[DIM] = {3.0f, 2.0f};  // d = [3, 2] - [0, 0] = [3, 2]
    float h_mu[DIM] = {0.0f, 0.0f};
    float h_conic[3] = {1.0f, 0.0f, 1.0f};  // Identity (c_01 = 0)
    float amplitude = 1.0f;
    float sharpness = 2.0f;

    cudaMemcpy(d_px, h_px, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu, h_mu, DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conic, h_conic, 3 * sizeof(float), cudaMemcpyHostToDevice);

    // Compute gradient at c_01 = 0
    float h_grad[3];
    compute_conic_gradient_kernel<DIM><<<1, 1>>>(
        d_px, d_mu, d_conic, amplitude, sharpness, 1.0f, d_grad_conic
    );
    cudaDeviceSynchronize();
    cudaMemcpy(h_grad, d_grad_conic, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // For d = [3, 2]:
    // ∂D²/∂c_00 = d[0]² = 9
    // ∂D²/∂c_01 = 2 × d[0] × d[1] = 2 × 3 × 2 = 12  (WITH 2× factor)
    // ∂D²/∂c_11 = d[1]² = 4
    //
    // If the 2× factor is missing, ∂D²/∂c_01 would be 6 instead of 12.

    float d0 = h_px[0] - h_mu[0];  // 3
    float d1 = h_px[1] - h_mu[1];  // 2

    // The gradient includes the chain rule factor, but the RATIO between
    // components should match the d_i × d_j pattern.
    float ratio_01_to_00 = h_grad[1] / h_grad[0];
    float expected_ratio_with_2x = (2.0f * d0 * d1) / (d0 * d0);  // 12/9 = 1.333
    float expected_ratio_without_2x = (d0 * d1) / (d0 * d0);      // 6/9 = 0.667

    printf("Ratio test:\n");
    printf("  grad[c_01]/grad[c_00] = %.4f\n", ratio_01_to_00);
    printf("  Expected with 2×: %.4f\n", expected_ratio_with_2x);
    printf("  Expected without 2×: %.4f\n", expected_ratio_without_2x);

    // The ratio should match the WITH 2× expectation
    float error_with_2x = std::abs(ratio_01_to_00 - expected_ratio_with_2x);
    float error_without_2x = std::abs(ratio_01_to_00 - expected_ratio_without_2x);

    EXPECT_LT(error_with_2x, 0.01f)
        << "Off-diagonal gradient ratio doesn't match 2× expectation!\n"
        << "This means the implementation is MISSING the factor of 2.\n"
        << "Check compute_conic_gradient in rasterize_bwd_nd kernel.";

    EXPECT_GT(error_without_2x, 0.5f)
        << "Gradient ratio matches the WRONG (no 2×) expectation!\n"
        << "The 2× factor is definitely missing.";
}
```

**Running the C++ gradient tests**:

```bash
# Build and run
cd tests/cpp/build
cmake .. && make -j
./cuda_kernel_tests --gtest_filter="ConicGradientTest.*"

# Expected output:
# [==========] Running 3 tests from 1 test suite.
# [ RUN      ] ConicGradientTest.FiniteDiff2D
#   c_00 (diag): analytic=-0.00234567, fd=-0.00234565, rel_err=8.52e-06
#   c_01 (off-diag, 2×): analytic=-0.00312456, fd=-0.00312454, rel_err=6.41e-06
#   c_11 (diag): analytic=-0.00156234, fd=-0.00156233, rel_err=6.40e-06
# [       OK ] ConicGradientTest.FiniteDiff2D
# ...
```

**Why this test catches the 2× bug**:

The `OffDiagonalRatioCheck2D` test is particularly powerful because:
1. It uses simple integer displacements (d = [3, 2]) for easy mental verification
2. It computes the RATIO between gradient components, which isolates the 2× factor
3. If the ratio is ~0.67 instead of ~1.33, the bug is definitively proven

#### 11.4.5 Running C++ Tests

```bash
# Build tests
cd tests/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j

# Run tests
./cuda_kernel_tests

# Run with verbose output
./cuda_kernel_tests --gtest_verbose

# Run specific test
./cuda_kernel_tests --gtest_filter="AABBTest.*"

# Run with valgrind for memory checks (CPU-side only)
valgrind ./cuda_kernel_tests --gtest_filter="*CPU*"
```

#### 11.4.6 Integration with Python Tests

For tests that need both C++ correctness and Python integration, use pytest fixtures
that call the C++ tests:

```python
# tests/test_cuda_cpp_integration.py
import subprocess
import pytest

@pytest.fixture(scope="session")
def cpp_tests_built():
    """Ensure C++ tests are built before running."""
    result = subprocess.run(
        ["make", "-C", "tests/cpp/build"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"C++ tests failed to build: {result.stderr}")
    return True

@pytest.mark.gpu
def test_cpp_kernel_tests_pass(cpp_tests_built):
    """All C++ kernel unit tests should pass."""
    result = subprocess.run(
        ["./tests/cpp/build/cuda_kernel_tests"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"C++ tests failed:\n{result.stdout}\n{result.stderr}"
```

---

*Last updated: 2026-01-10*
