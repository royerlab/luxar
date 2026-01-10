from __future__ import annotations

from typing import Any, Optional

_GSPLATS_IMPORT_ERROR: Optional[ImportError] = None

try:
    from luxar.gsplats import clahe, seeds
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
    from luxar.gsplats.fit_multiscale_gsplats import fit_multiscale_gaussian_splats
    from luxar.gsplats.fit_result import GSplatData
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.seeds import (
        generate_seeds,
        seed_from_decomposition,
        seed_from_edges,
        seed_from_grid,
    )
except ImportError as exc:
    _GSPLATS_IMPORT_ERROR = exc

    def _raise_gsplats_import_error(_exc: ImportError = exc) -> None:
        raise ImportError(
            "luxar.gsplats requires optional dependencies. "
            "Install with: pip install 'luxar[gsplats]'"
        ) from _exc

    def fit_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()

    def fit_multiscale_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()

    class GaussianSplatFitter:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _raise_gsplats_import_error()

    class GSplatData:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _raise_gsplats_import_error()

    class DynamicOpsConfig:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _raise_gsplats_import_error()

    def generate_seeds(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()

    def seed_from_decomposition(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()

    def seed_from_grid(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()

    def seed_from_edges(*_args: Any, **_kwargs: Any) -> Any:
        _raise_gsplats_import_error()


__all__ = [
    # Fitting functions
    "fit_gaussian_splats",
    "fit_multiscale_gaussian_splats",
    "GaussianSplatFitter",
    "GSplatData",
    "DynamicOpsConfig",
    # Seeding functions
    "generate_seeds",
    "seed_from_decomposition",
    "seed_from_grid",
    "seed_from_edges",
    # Submodules
    "seeds",
    "clahe",
]
