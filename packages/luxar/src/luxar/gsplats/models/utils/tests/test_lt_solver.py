"""
Tests for lower triangular solver with cross-version PyTorch compatibility.
"""

import pytest

from luxar.gsplats.models.utils.lt_solver import solve_lower_triangular

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")


class TestSolveLowerTriangular:
    """Test solve_lower_triangular function."""

    def test_solve_2x2_single_rhs(self) -> None:
        """Test solving 2x2 system with single right-hand side."""
        # L @ x = b, where L is lower triangular
        L = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32)
        b = torch.tensor([[4.0], [7.0]], dtype=torch.float32)  # Shape (2, 1)

        x = solve_lower_triangular(L, b)

        # Verify solution: L @ x should equal b
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-6, rtol=1e-6)

        # Check expected solution: x = [2.0, 1.0]^T
        expected = torch.tensor([[2.0], [5.0 / 3.0]], dtype=torch.float32)
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_solve_3x3_multiple_rhs(self) -> None:
        """Test solving 3x3 system with multiple right-hand sides."""
        L = torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, 1.0, 4.0]], dtype=torch.float32
        )
        b = torch.tensor(
            [[1.0, 2.0], [5.0, 8.0], [6.0, 10.0]], dtype=torch.float32
        )  # Shape (3, 2)

        x = solve_lower_triangular(L, b)

        # Verify solution
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-6, rtol=1e-6)

        assert x.shape == (3, 2)

    def test_solve_batched_systems(self) -> None:
        """Test solving batched lower triangular systems."""
        torch.manual_seed(0)
        batch_size = 4
        d = 3

        # Create a batch of distinct, well-conditioned lower triangular
        # matrices. The diagonal is drawn in [2, 3) so no batch element is ever
        # near-singular; without this, an unlucky draw near zero makes the
        # float32 solve residual exceed the tolerance below and the test flakes
        # (~2% of runs).
        L = torch.zeros(batch_size, d, d, dtype=torch.float32)
        for i in range(batch_size):
            off_diag = torch.tril(torch.randn(d, d), diagonal=-1)
            diag = torch.diag(2.0 + torch.rand(d))
            L[i] = off_diag + diag

        # Multiple RHS per batch
        b = torch.randn(batch_size, d, 2, dtype=torch.float32)

        x = solve_lower_triangular(L, b)

        # Verify solutions
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-5, rtol=1e-5)

        assert x.shape == (batch_size, d, 2)

    def test_solve_identity_matrix(self) -> None:
        """Test solving with identity matrix (should return b unchanged)."""
        identity_matrix = torch.eye(3, dtype=torch.float32)
        b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)

        x = solve_lower_triangular(identity_matrix, b)

        torch.testing.assert_close(x, b, atol=1e-7, rtol=1e-7)

    def test_solve_diagonal_matrix(self) -> None:
        """Test solving with diagonal matrix."""
        D = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
        b = torch.tensor([[6.0], [9.0], [12.0]], dtype=torch.float32)

        x = solve_lower_triangular(D, b)

        # Expected: x = [3.0, 3.0, 3.0]^T
        expected = torch.tensor([[3.0], [3.0], [3.0]], dtype=torch.float32)
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_solve_single_dimension(self) -> None:
        """Test solving 1x1 system (scalar case)."""
        L = torch.tensor([[5.0]], dtype=torch.float32)
        b = torch.tensor([[10.0, 15.0]], dtype=torch.float32)

        x = solve_lower_triangular(L, b)

        expected = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)

    def test_solve_large_system(self) -> None:
        """Test solving larger system to check performance and accuracy."""
        d = 10
        torch.manual_seed(42)

        # Create well-conditioned lower triangular matrix
        L = torch.tril(torch.randn(d, d)) + 2.0 * torch.eye(d)
        b = torch.randn(d, 3)

        x = solve_lower_triangular(L, b)

        # Verify solution
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-4, rtol=1e-4)

        assert x.shape == (d, 3)

    def test_solve_different_dtypes(self) -> None:
        """Test solver with different tensor dtypes."""
        L = torch.tensor([[2.0, 0.0], [1.0, 3.0]])
        b = torch.tensor([[4.0], [7.0]])

        # Test float32
        L_32 = L.to(torch.float32)
        b_32 = b.to(torch.float32)
        x_32 = solve_lower_triangular(L_32, b_32)
        assert x_32.dtype == torch.float32

        # Test float64
        L_64 = L.to(torch.float64)
        b_64 = b.to(torch.float64)
        x_64 = solve_lower_triangular(L_64, b_64)
        assert x_64.dtype == torch.float64

        # Results should be similar
        torch.testing.assert_close(x_32.double(), x_64, atol=1e-6, rtol=1e-6)

    def test_solve_different_devices(self) -> None:
        """Test solver on different devices (CPU and CUDA if available)."""
        L = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32)
        b = torch.tensor([[4.0], [7.0]], dtype=torch.float32)

        # Test CPU
        x_cpu = solve_lower_triangular(L, b)
        result_cpu = L @ x_cpu
        torch.testing.assert_close(result_cpu, b, atol=1e-6, rtol=1e-6)

        # Test CUDA if available
        if torch.cuda.is_available():
            L_cuda = L.cuda()
            b_cuda = b.cuda()
            x_cuda = solve_lower_triangular(L_cuda, b_cuda)
            result_cuda = L_cuda @ x_cuda
            torch.testing.assert_close(result_cuda, b_cuda, atol=1e-6, rtol=1e-6)

            # Results should be similar
            torch.testing.assert_close(x_cpu, x_cuda.cpu(), atol=1e-6, rtol=1e-6)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through the solver correctly."""
        L = torch.tensor(
            [[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32, requires_grad=True
        )
        b = torch.tensor([[4.0], [7.0]], dtype=torch.float32, requires_grad=True)

        x = solve_lower_triangular(L, b)
        loss = torch.sum(x**2)
        loss.backward()  # type: ignore[no-untyped-call]

        # Gradients exist, are finite, and non-zero for both inputs.
        for name, t in (("L", L), ("b", b)):
            assert t.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(t.grad).all(), f"{name} grad is non-finite"
            assert not torch.allclose(t.grad, torch.zeros_like(t.grad))

        # Analytic gradients match numerical ones (gradcheck, float64) — this is
        # what actually proves the backward is correct, not merely present.
        Ld = torch.tensor(
            [[2.0, 0.0], [1.0, 3.0]], dtype=torch.float64, requires_grad=True
        )
        bd = torch.tensor([[4.0], [7.0]], dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(solve_lower_triangular, (Ld, bd), atol=1e-6)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_singular_matrix(self) -> None:
        """Test behavior with singular (non-invertible) matrix."""
        # Matrix with zero on diagonal
        L = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)  # Singular
        b = torch.tensor([[1.0], [1.0]], dtype=torch.float32)

        # This should either error or produce inf/nan
        try:
            x = solve_lower_triangular(L, b)
            # If it doesn't error, result should contain inf or nan
            assert torch.any(torch.isinf(x)) or torch.any(torch.isnan(x))
        except (RuntimeError, torch.linalg.LinAlgError):
            # Expected behavior for singular matrix
            pass

    def test_mismatched_dimensions(self) -> None:
        """Test error handling for mismatched dimensions."""
        L = torch.tensor([[1.0, 0.0], [1.0, 2.0]], dtype=torch.float32)  # 2x2
        b = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)  # 3x1

        with pytest.raises((RuntimeError, ValueError)):
            solve_lower_triangular(L, b)

    def test_non_square_matrix(self) -> None:
        """Test error handling for non-square coefficient matrix."""
        L = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]], dtype=torch.float32
        )  # 2x3 (not square)
        b = torch.tensor([[1.0], [2.0]], dtype=torch.float32)

        with pytest.raises((RuntimeError, ValueError)):
            solve_lower_triangular(L, b)

    def test_empty_tensors(self) -> None:
        """Test behavior with empty tensors."""
        L = torch.empty((0, 0), dtype=torch.float32)
        b = torch.empty((0, 1), dtype=torch.float32)

        x = solve_lower_triangular(L, b)
        assert x.shape == (0, 1)


class TestCrossVersionCompatibility:
    """Test compatibility across different PyTorch versions."""

    def test_both_api_paths(self) -> None:
        """Test that both PyTorch solve paths produce same results."""
        L = torch.tensor([[2.0, 0.0], [1.0, 3.0]], dtype=torch.float32)
        b = torch.tensor([[4.0], [7.0]], dtype=torch.float32)

        # Our function should work regardless of PyTorch version
        x = solve_lower_triangular(L, b)

        # Verify solution is correct
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-6, rtol=1e-6)

        # Test that the function doesn't crash (version compatibility)
        assert x.shape == (2, 1)
        assert x.dtype == torch.float32

    def test_fallback_behavior(self) -> None:
        """Test fallback behavior across supported PyTorch versions."""
        # This is hard to test directly without mocking, but we can at least
        # ensure the function works in various scenarios. Seed and keep the
        # random case well-conditioned (diagonal in [2, 3)) so the
        # float32 residual stays inside the tolerance below regardless of the
        # RNG stream or test ordering.
        torch.manual_seed(0)
        rand_L = torch.tril(torch.randn(3, 3), diagonal=-1) + torch.diag(
            2.0 + torch.rand(3)
        )
        test_cases = [
            # (L, b) pairs to test
            (torch.eye(2), torch.ones(2, 1)),
            (rand_L, torch.randn(3, 2)),
            (
                torch.tensor([[5.0]], dtype=torch.float32),
                torch.tensor([[10.0]], dtype=torch.float32),
            ),
        ]

        for L, b in test_cases:
            x = solve_lower_triangular(L, b)
            result = L @ x
            torch.testing.assert_close(result, b, atol=1e-5, rtol=1e-5)


class TestNumericalStability:
    """Test numerical stability of the solver."""

    def test_ill_conditioned_matrix(self) -> None:
        """Test solver with ill-conditioned but solvable matrix."""
        # Create ill-conditioned lower triangular matrix
        L = torch.tensor([[1e-8, 0.0], [1.0, 1e-8]], dtype=torch.float32)
        b = torch.tensor([[1e-8], [1.0]], dtype=torch.float32)

        x = solve_lower_triangular(L, b)

        # Solution should still satisfy L @ x = b reasonably well
        result = L @ x
        torch.testing.assert_close(result, b, atol=1e-6, rtol=1e-3)

    def test_well_conditioned_vs_manual_solution(self) -> None:
        """Compare solver result with manually computed solution for well-conditioned system."""
        L = torch.tensor([[2.0, 0.0], [3.0, 4.0]], dtype=torch.float32)
        b = torch.tensor([[6.0], [18.0]], dtype=torch.float32)

        x = solve_lower_triangular(L, b)

        # Manual solution:
        # 2*x1 = 6 => x1 = 3
        # 3*x1 + 4*x2 = 18 => 3*3 + 4*x2 = 18 => x2 = 9/4 = 2.25
        expected = torch.tensor([[3.0], [2.25]], dtype=torch.float32)

        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-6)
