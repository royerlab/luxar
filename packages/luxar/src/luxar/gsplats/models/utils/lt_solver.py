from typing import cast

import torch


def solve_lower_triangular(L: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Solve lower triangular linear system L @ X = B for X.

    This function provides cross-version compatibility for PyTorch's triangular
    solve functionality, preferring the newer torch.linalg.solve_triangular when
    available, falling back to torch.triangular_solve for older versions.

    Parameters
    ----------
    L : torch.Tensor, shape (d, d) or (N, d, d)
        Lower triangular coefficient matrix. Upper triangular elements are ignored.
        For batched operation, first dimension is batch size.
    B : torch.Tensor, shape (d, P) or (N, d, P)
        Right-hand side matrix with P solution vectors in columns.
        Must have compatible batch dimensions with L.

    Returns
    -------
    torch.Tensor, shape (d, P) or (N, d, P)
        Solution matrix X such that L @ X = B.

    Notes
    -----
    This solver is numerically stable and efficient for lower triangular systems,
    commonly arising from Cholesky decomposition. The operation is performed via
    forward substitution.
    """
    # PyTorch Version Compatibility Fallback
    # ========================================
    # torch.linalg.solve_triangular: Added in PyTorch 1.9 (2021), stable in PyTorch 2.0+
    # torch.triangular_solve: Deprecated in PyTorch 1.9, removed in PyTorch 2.0 (2023)
    #
    # RATIONALE FOR FALLBACK:
    # - Supports users on PyTorch 1.12-1.13 (deprecated API still exists)
    # - Graceful degradation for older PyTorch installations
    # - We require torch>=1.12.0 (pyproject.toml), so solve_triangular should be available
    #
    # MPS (APPLE SILICON) NOTES:
    # - Both functions have IDENTICAL 10× overhead on MPS vs CPU (as of PyTorch 2.5, 2024)
    # - This fallback does NOT improve MPS performance (same underlying operation)
    # - MPS had device check bugs in solve_triangular (fixed Dec 2024, PyTorch PR #142477)
    # - CPU is currently 10× faster than MPS for this operation on Apple Silicon
    # - See README.md "Apple Silicon Performance Notes" for details
    #
    # FUTURE: Consider removing fallback when PyTorch 1.x support is no longer needed
    try:
        # Modern PyTorch (>= 1.9): use torch.linalg.solve_triangular
        return cast(torch.Tensor, torch.linalg.solve_triangular(L, B, upper=False))
    except Exception:
        # Legacy PyTorch (1.12-1.13): use deprecated torch.triangular_solve
        # Note: triangular_solve returns (solution, cloned_L_matrix) tuple
        X, _ = torch.triangular_solve(B, L, upper=False)
        return X
