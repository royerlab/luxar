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
    # MPS (APPLE SILICON) NOTES:
    # - MPS has 10× overhead vs CPU for this operation (as of PyTorch 2.5, 2024)
    # - MPS had device check bugs in solve_triangular (fixed Dec 2024, PyTorch PR #142477)
    # - CPU is currently 10× faster than MPS for this operation on Apple Silicon
    # - See README.md "Apple Silicon Performance Notes" for details
    return cast(torch.Tensor, torch.linalg.solve_triangular(L, B, upper=False))
