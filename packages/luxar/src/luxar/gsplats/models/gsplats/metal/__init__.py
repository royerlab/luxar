"""
Metal-accelerated Gaussian splatting renderer for Apple Silicon.

This package provides a high-performance Metal compute backend for Gaussian splatting,
achieving 10-50x speedup over CPU PyTorch on M-series chips.

The Metal extension is automatically compiled on first use if not already built.
Requirements:
- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode (full installation, not just Command Line Tools)
- PyTorch with MPS support
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

# Package directory
_METAL_DIR = Path(__file__).parent
_SRC_DIR = _METAL_DIR / "src"
_EXTENSION_PATTERN = "metal_splatting_backend.cpython-*.so"


def _find_extension() -> Optional[Path]:
    """Find the compiled extension file if it exists."""
    extensions = list(_METAL_DIR.glob(_EXTENSION_PATTERN))
    if extensions:
        return extensions[0]
    return None


def _check_xcode_setup() -> Tuple[bool, str]:
    """
    Check if Xcode is properly set up for Metal compilation.

    Returns:
        (is_valid, message) tuple
    """
    # Check 1: Is xcrun available?
    try:
        result = subprocess.run(
            ["xcrun", "--find", "metal"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, _XCODE_NOT_FOUND_MESSAGE
    except FileNotFoundError:
        return False, _XCODE_NOT_FOUND_MESSAGE
    except subprocess.TimeoutExpired:
        return False, "Xcode tools check timed out. Please verify Xcode installation."

    # Check 2: Is xcode-select pointing to Xcode.app (not CommandLineTools)?
    try:
        result = subprocess.run(
            ["xcode-select", "-p"], capture_output=True, text=True, timeout=10
        )
        xcode_path = result.stdout.strip()
        if "CommandLineTools" in xcode_path:
            return False, _COMMAND_LINE_TOOLS_MESSAGE.format(current_path=xcode_path)
    except Exception:
        pass  # Non-fatal, continue

    return True, "Xcode setup OK"


_XCODE_NOT_FOUND_MESSAGE = """
================================================================================
METAL BACKEND: Xcode Not Found
================================================================================

The Metal backend requires Apple's Xcode to compile Metal shaders.

To install Xcode:

  1. Install Xcode from the Mac App Store (or developer.apple.com)
     Note: This is a large download (~12GB)

  2. After installation, open Xcode once to accept the license, or run:

     sudo xcodebuild -license accept

  3. Point xcode-select to Xcode.app:

     sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

  4. Verify the Metal compiler is available:

     xcrun --find metal

After completing these steps, restart Python and import again.

For more details, see:
  packages/luxar/src/luxar/gsplats/models/gsplats/metal/README.md
================================================================================
"""

_COMMAND_LINE_TOOLS_MESSAGE = """
================================================================================
METAL BACKEND: Wrong Xcode Path
================================================================================

The Metal compiler requires the full Xcode.app, but xcode-select is pointing to:

  {current_path}

This typically means only Command Line Tools are active, not the full Xcode.

To fix this:

  1. Ensure Xcode.app is installed (from Mac App Store or developer.apple.com)

  2. Switch xcode-select to Xcode.app:

     sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer

  3. Verify the Metal compiler is now available:

     xcrun --find metal

After completing these steps, restart Python and import again.
================================================================================
"""

_BUILD_FAILED_MESSAGE = """
================================================================================
METAL BACKEND: Build Failed
================================================================================

The Metal extension failed to compile. This is usually due to:

  1. Missing Xcode or wrong xcode-select path (see above)
  2. Missing PyTorch or incompatible version
  3. Permissions issues in the package directory

Build output:
{build_output}

To manually build:

  cd {metal_dir}
  python setup.py build_ext --inplace

For troubleshooting, see:
  packages/luxar/src/luxar/gsplats/models/gsplats/metal/README.md
================================================================================
"""

_BUILD_SUCCESS_MESSAGE = """
Metal backend compiled successfully!
  Extension: {extension_path}

The extension will be used automatically for future imports.
"""


def _auto_build_extension() -> Tuple[bool, str]:
    """
    Automatically build the Metal extension if not present.

    Returns:
        (success, message) tuple
    """
    from arbol import aprint, asection

    # Check if already built
    existing = _find_extension()
    if existing and existing.exists():
        return True, f"Using existing extension: {existing.name}"

    with asection("Metal Backend: Auto-compiling (first-time setup)"):
        aprint("This only happens once. Future imports will be instant.")
        aprint("")

        # Check Xcode setup first
        xcode_ok, xcode_msg = _check_xcode_setup()
        if not xcode_ok:
            aprint(xcode_msg)
            return False, xcode_msg

        aprint("Xcode setup verified")
        aprint(f"Building in: {_METAL_DIR}")
        aprint("")

        # Run setup.py build_ext --inplace
        try:
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=str(_METAL_DIR),
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if result.returncode != 0:
                error_output = result.stderr or result.stdout or "No output"
                msg = _BUILD_FAILED_MESSAGE.format(
                    build_output=error_output[:2000],  # Truncate if very long
                    metal_dir=_METAL_DIR,
                )
                aprint(msg)
                return False, msg

            # Verify build succeeded
            built_extension = _find_extension()
            if built_extension and built_extension.exists():
                msg = _BUILD_SUCCESS_MESSAGE.format(extension_path=built_extension)
                aprint(msg)
                return True, msg
            else:
                msg = (
                    f"Build appeared to succeed but extension not found in {_METAL_DIR}"
                )
                aprint(msg)
                return False, msg

        except subprocess.TimeoutExpired:
            msg = "Build timed out after 2 minutes. Try building manually."
            aprint(msg)
            return False, msg
        except Exception as e:
            msg = f"Build failed with exception: {e}"
            aprint(msg)
            return False, msg


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


def _try_import_extension() -> bool:
    """Try to import the Metal extension."""
    try:
        # Import torch first to load PyTorch libraries (needed for rpath)
        import torch

        # Add current directory to path for import
        if str(_METAL_DIR) not in sys.path:
            sys.path.insert(0, str(_METAL_DIR))

        import metal_splatting_backend

        return True

    except (ImportError, OSError):
        # Don't warn here - caller will handle
        return False


# Check if Metal extension is available
_metal_available = False
_mps_interop_valid = False
_init_message = ""

if sys.platform == "darwin":
    # First validate MPS interop works
    _mps_interop_valid = _validate_mps_interop()

    if _mps_interop_valid:
        # Try to import existing extension
        if _try_import_extension():
            _metal_available = True
            _init_message = "Metal backend loaded successfully"
        else:
            # Extension not found - try auto-build
            build_success, build_msg = _auto_build_extension()

            if build_success:
                # Try import again after build
                if _try_import_extension():
                    _metal_available = True
                    _init_message = "Metal backend compiled and loaded"
                else:
                    _init_message = "Build succeeded but import failed. Check Python version compatibility."
                    warnings.warn(_init_message, RuntimeWarning)
            else:
                _init_message = build_msg
                # Don't warn here - the build function already printed detailed help


def is_metal_available() -> bool:
    """
    Check if Metal splatting backend is available and working.

    Returns True only if:
    1. Running on macOS
    2. MPS-Metal interop validated
    3. Metal extension successfully imported

    If False, the package will use PyTorch CPU/MPS fallback instead.
    """
    return _metal_available and _mps_interop_valid


def get_metal_status() -> str:
    """
    Get a human-readable status message about the Metal backend.

    Returns:
        Status message explaining current Metal backend state
    """
    if sys.platform != "darwin":
        return "Metal backend only available on macOS"
    if not _mps_interop_valid:
        return "MPS-Metal interop validation failed. PyTorch MPS may not be available."
    if _metal_available:
        return "Metal backend available and loaded"
    return _init_message or "Metal backend not available"


# Only expose GaussianSplatModelMetal if Metal is available
__all__ = ["is_metal_available", "get_metal_status"]

if _metal_available:
    from .gsplat_model_metal import GaussianSplatModelMetal

    __all__.append("GaussianSplatModelMetal")
