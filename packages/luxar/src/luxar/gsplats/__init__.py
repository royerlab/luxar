from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

_GSPLATS_IMPORT_ERROR: Optional[ImportError] = None

if TYPE_CHECKING:
    from luxar.gsplats import clahe, lod, preprocessing, seeds
    from luxar.gsplats.calibration import (
        CalibrationResult,
        HeldOutPeak,
        NoiseFloor,
        build_k_grid,
        calibrate,
        cv_mask,
        donut_median_fill,
        estimate_noise_floor,
        find_k_star,
        held_out_psnr,
    )
    from luxar.gsplats.culling import CullResult, cull_by_contribution
    from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
    from luxar.gsplats.fit_progressive_gsplats import fit_progressive_gaussian_splats
    from luxar.gsplats.fit_tiled_gsplats import fit_tile, fit_tiled
    from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
    from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
    from luxar.gsplats.lift import (
        coarse_substitutive_levels,
        compute_ray_integral_factor,
        lift_lines_to_gsplats,
        lift_points_to_gsplats,
        render_light,
    )
    from luxar.gsplats.lod import (
        compute_additive_order,
        make_additive_lod,
        make_lod_pyramid,
        make_substitutive_lod,
    )
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
        from luxar.gsplats import clahe, lod, preprocessing, seeds
        from luxar.gsplats.calibration import (
            CalibrationResult,
            HeldOutPeak,
            NoiseFloor,
            build_k_grid,
            calibrate,
            cv_mask,
            donut_median_fill,
            estimate_noise_floor,
            find_k_star,
            held_out_psnr,
        )
        from luxar.gsplats.culling import CullResult, cull_by_contribution
        from luxar.gsplats.fit_gsplats import GaussianSplatFitter, fit_gaussian_splats
        from luxar.gsplats.fit_progressive_gsplats import (
            fit_progressive_gaussian_splats,
        )
        from luxar.gsplats.fit_tiled_gsplats import fit_tile, fit_tiled
        from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
        from luxar.gsplats.gsplat_data import (
            AdditiveSubLOD,
            GSplatData,
            SubstitutiveLevel,
        )
        from luxar.gsplats.lift import (
            coarse_substitutive_levels,
            compute_ray_integral_factor,
            lift_lines_to_gsplats,
            lift_points_to_gsplats,
            render_light,
        )
        from luxar.gsplats.lod import (
            compute_additive_order,
            make_additive_lod,
            make_lod_pyramid,
            make_substitutive_lod,
        )
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
                "luxar.gsplats requires optional dependencies (torch, scipy, etc.). "
                "Install with: pip install 'luxar[gsplats]'\n"
                f"Original error: {_exc}"
            ) from _exc

        def cull_by_contribution(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        class CullResult:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        def fit_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def fit_progressive_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        class GaussianSplatFitter:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class GSplatData:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class AdditiveSubLOD:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class SubstitutiveLevel:
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

        # Calibration (blind-spot CV)
        def cv_mask(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def donut_median_fill(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def held_out_psnr(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def estimate_noise_floor(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def build_k_grid(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def find_k_star(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def calibrate(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        class NoiseFloor:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class HeldOutPeak:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class CalibrationResult:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        # LOD (luxar.gsplats.lod)
        lod = None  # type: ignore[assignment]

        def compute_additive_order(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def make_additive_lod(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def make_substitutive_lod(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def make_lod_pyramid(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        # Lift (luxar.gsplats.lift) — points -> gsplats
        def coarse_substitutive_levels(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def compute_ray_integral_factor(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def lift_lines_to_gsplats(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def lift_points_to_gsplats(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()

        def render_light(*_args: Any, **_kwargs: Any) -> Any:
            _raise_gsplats_import_error()


__all__ = [
    # Culling
    "cull_by_contribution",
    "CullResult",
    # Fitting functions
    "fit_gaussian_splats",
    "fit_progressive_gaussian_splats",
    "fit_tiled_gaussian_splats",
    "fit_tile",
    "fit_tiled",
    "GaussianSplatFitter",
    "GSplatData",
    "AdditiveSubLOD",
    "SubstitutiveLevel",
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
    # Calibration (blind-spot CV)
    "cv_mask",
    "donut_median_fill",
    "held_out_psnr",
    "estimate_noise_floor",
    "build_k_grid",
    "find_k_star",
    "calibrate",
    "NoiseFloor",
    "HeldOutPeak",
    "CalibrationResult",
    # LOD (additive + substitutive + pyramid)
    "compute_additive_order",
    "make_additive_lod",
    "make_lod_pyramid",
    "make_substitutive_lod",
    # Lift (points -> gsplats)
    "coarse_substitutive_levels",
    "compute_ray_integral_factor",
    "lift_lines_to_gsplats",
    "lift_points_to_gsplats",
    "render_light",
    # Submodules
    "seeds",
    "clahe",
    "preprocessing",
    "lod",
]
