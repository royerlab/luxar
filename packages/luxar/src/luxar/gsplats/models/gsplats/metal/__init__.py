"""
Metal-accelerated Gaussian splatting renderer for Apple Silicon.

This package provides a high-performance Metal compute backend for Gaussian splatting,
achieving 3-7x speedup over CPU PyTorch on M-series chips.
"""

from __future__ import annotations

import sys
import warnings


def _validate_mps_interop() -> bool:
    """
    MANDATORY: Validate MPS-Metal buffer interop works correctly.

    Tests that PyTorch MPS tensors can be accessed as Metal buffers.
    This uses PyTorch internal APIs that may change between versions.

    Returns:
        True if interop validated, False otherwise
    """
    try:
        import torch

        if not torch.backends.mps.is_available():
            return False  # MPS not available, Metal won't work

        # Test 1: Basic buffer extraction
        t = torch.randn(10, device="mps")
        # Use untyped_storage() to avoid deprecation warning
        storage_ptr = t.untyped_storage().data_ptr()
        if not storage_ptr:
            raise RuntimeError("untyped_storage().data_ptr() returned null")

        # Test 2: Storage offset handling (critical for tensor views)
        t_full = torch.randn(100, device="mps")
        t_slice = t_full[25:75]  # View with storage_offset=25
        if t_slice.storage_offset() != 25:
            raise RuntimeError("storage_offset not working correctly")

        # Test 3: Contiguous check
        t_contig = t_full.contiguous()
        if not t_contig.is_contiguous():
            raise RuntimeError("contiguous() failed")

        return True

    except Exception as e:
        warnings.warn(
            f"MPS-Metal interop validation failed: {e}. "
            f"Metal acceleration disabled. Using PyTorch fallback.",
            RuntimeWarning,
        )
        return False


# Check if Metal extension is available
_metal_available = False
_mps_interop_valid = False

if sys.platform == "darwin":
    # First validate MPS interop works
    _mps_interop_valid = _validate_mps_interop()

    if _mps_interop_valid:
        try:
            # Import torch first to load PyTorch libraries (needed for rpath)
            import torch

            # Import from current package directory
            import os

            _current_dir = os.path.dirname(__file__)
            if _current_dir not in sys.path:
                sys.path.insert(0, _current_dir)

            import metal_splatting_backend

            _metal_available = True
        except (ImportError, OSError) as e:
            warnings.warn(
                f"Metal backend import failed: {e}. Using PyTorch fallback.",
                RuntimeWarning,
            )
            _metal_available = False


def is_metal_available() -> bool:
    """
    Check if Metal splatting backend is available and working.

    Returns True only if:
    1. Running on macOS
    2. MPS-Metal interop validated
    3. Metal extension successfully imported
    """
    return _metal_available and _mps_interop_valid


# Only expose GaussianSplatModelMetal if Metal is available
__all__ = ["is_metal_available"]

if _metal_available:
    from .gsplat_model_metal import GaussianSplatModelMetal

    __all__.append("GaussianSplatModelMetal")
