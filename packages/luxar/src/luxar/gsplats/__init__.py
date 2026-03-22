from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

_GSPLATS_IMPORT_ERROR: Optional[ImportError] = None

if TYPE_CHECKING:
    from luxar.gsplats import clahe, seeds
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
    from luxar.gsplats.fit_multiscale_gsplats import fit_multiscale_gaussian_splats
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
    from luxar.gsplats.fit_tiled_gsplats import fit_tile, fit_tiled
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD
    from luxar.gsplats.seeds import (
        generate_seeds,
        seed_from_decomposition,
        seed_from_edges,
        seed_from_grid,
    )
    from luxar.gsplats.tiling import TileSpec, compute_tile_specs, cosine_window

    fit_tiled_gaussian_splats = fit_tiled
else:
    try:
        from luxar.gsplats import clahe, seeds
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
        from luxar.gsplats.fit_multiscale_gsplats import fit_multiscale_gaussian_splats
        from luxar.gsplats.fit_progressive_gsplats import (
            fit_progressive_gaussian_splats,
        )
        from luxar.gsplats.fit_tiled_gsplats import fit_tile, fit_tiled
        from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
        from luxar.gsplats.gsplat_data import GSplatData, GSplatLOD
        from luxar.gsplats.seeds import (
            generate_seeds,
            seed_from_decomposition,
            seed_from_edges,
            seed_from_grid,
        )
        from luxar.gsplats.tiling import TileSpec, compute_tile_specs, cosine_window

        fit_tiled_gaussian_splats = fit_tiled
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

        def fit_progressive_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        class GaussianSplatFitter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class GSplatData:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class GSplatLOD:
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

        def fit_tile(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def fit_tiled(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        fit_tiled_gaussian_splats = fit_tiled

        def compute_tile_specs(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def cosine_window(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        class TileSpec:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()


__all__ = [
    # Fitting functions
    "fit_gaussian_splats",
    "fit_multiscale_gaussian_splats",
    "fit_progressive_gaussian_splats",
    "fit_tiled_gaussian_splats",
    "fit_tile",
    "fit_tiled",
    "GaussianSplatFitter",
    "GSplatData",
    "GSplatLOD",
    "DynamicOpsConfig",
    # Tiling
    "TileSpec",
    "compute_tile_specs",
    "cosine_window",
    # Seeding functions
    "generate_seeds",
    "seed_from_decomposition",
    "seed_from_grid",
    "seed_from_edges",
    # Submodules
    "seeds",
    "clahe",
]
