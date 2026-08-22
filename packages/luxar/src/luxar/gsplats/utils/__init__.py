"""
Utilities for Gaussian splat fitting.

This package provides utility functions for:
- Lower-triangular matrix operations (pack/unpack Cholesky factors)
- Cholesky factor validation for Gaussian splats
- Cholesky dimension permutation and embedding for cross-dimensional scenes
- Gradient dilution compensation for higher-dimensional optimization
- Triangle matrix size calculations
"""

from typing import TYPE_CHECKING, Any

from luxar.gsplats.utils.trils import (
    calculate_gradient_dilution_factor,
    diag_indices,
    embed_cholesky_packed,
    merge_tril,
    offdiag_indices,
    pack_tril,
    permute_cholesky_packed,
    split_tril,
    tril_size,
    unpack_tril,
    validate_cholesky_shape,
)

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from luxar.gsplats.utils.device import is_mps_available, resolve_torch_device

__all__ = [
    "is_mps_available",
    "resolve_torch_device",
    "tril_size",
    "calculate_gradient_dilution_factor",
    "pack_tril",
    "unpack_tril",
    "validate_cholesky_shape",
    "permute_cholesky_packed",
    "embed_cholesky_packed",
    "diag_indices",
    "offdiag_indices",
    "split_tril",
    "merge_tril",
]

# Names that live in the torch-only ``device`` submodule.
# Guarded by ``luxar/tests/test_lazy_imports.py``, which runs the whole
# authoring path with the ``gsplats`` extra blocked.
_DEVICE_EXPORTS = frozenset({"is_mps_available", "resolve_torch_device"})


def __getattr__(name: str) -> Any:
    """Resolve the torch-backed exports lazily (PEP 562).

    ``luxar.gsplats.utils.device`` imports ``torch``, which ships only in the
    optional ``gsplats`` extra — but the CORE scene-authoring path reaches this
    package for the pure-numpy ``trils`` helpers. The UNCONDITIONAL route is the
    compiler: ``write_gsplat_arrays``
    (``luxar/io/_compiler/gsplat_assembly.py``) does ``from
    ...gsplats.utils.trils import split_tril`` on every ``add_gsplats``.
    (``core/group/dim_order.py`` -> ``embed_cholesky_packed`` is a second route,
    but only when the caller passes ``dim_order=``.) Importing a submodule
    executes this ``__init__``, so an eager ``from ... device import ...`` here
    would make ``scene.add_gsplats(...)`` fail with ``ModuleNotFoundError: No
    module named 'torch'`` on a core-only install. Keep these lazy: only fitting
    code (which already requires torch) touches them, and ``from
    luxar.gsplats.utils import resolve_torch_device`` still works unchanged.
    """
    if name in _DEVICE_EXPORTS:
        from luxar.gsplats.utils import device

        return getattr(device, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Keep the lazy exports visible to ``dir()`` (and hence to Sphinx).

    ``automodule ... :members:`` collects from ``dir()``, so without this the
    two ``device`` names silently vanish from the API reference — no warning,
    and the docs ratchet stays green. Unlike ``luxar/__init__.py``, whose lazy
    ``__getattr__`` hands back non-raising stubs, the names listed here still
    raise ``ModuleNotFoundError`` when resolved on a core-only install — so
    ``help()`` / ``inspect.getmembers()`` / ``import *`` raise here rather than
    degrading. That is deliberate: for a direct ``from luxar.gsplats.utils
    import resolve_torch_device`` the explicit missing-torch error is far more
    useful than a vague ``AttributeError``.
    """
    return sorted(set(globals()) | _DEVICE_EXPORTS)
