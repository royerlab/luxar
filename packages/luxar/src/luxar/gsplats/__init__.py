"""Gaussian splatting subsystem: fitting, calibration, LOD, tiling, and lifting.

Public entry points for turning a volume into oriented Gaussian splats
(``fit_gaussian_splats`` / ``GaussianSplatFitter``), calibrating the splat count
``K`` (``calibrate``), building level-of-detail topologies (``make_additive_lod``,
``make_substitutive_lod``, ``make_lod_pyramid``), tiling large volumes, seeding,
and lifting points/lines to splats. The heavy dependencies (torch, scipy) are
optional and loaded lazily: the package imports without them, and the entry
points that need them fail only when first used (the callable exports raise a
clear install hint; an unexpected internal import error propagates loudly).

The docstrings on the fallback definitions below describe what each real symbol
does; when the ``gsplats`` extra is absent these names resolve to stubs that
raise :class:`ImportError` (via :func:`_raise_gsplats_import_error`) on first
use rather than at import time.
"""

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
        # Only treat a genuinely MISSING OPTIONAL dependency (torch/scipy) as
        # "gsplats extra not installed" and fall back to stubs. An internal
        # import error (a typo'd/moved symbol inside luxar.gsplats.*) must
        # propagate loudly instead of masquerading as a missing extra — else a
        # real packaging bug is silently hidden behind the install hint.
        _OPTIONAL_ROOTS = ("torch", "scipy")
        _missing = (exc.name or "").split(".", 1)[0]
        if _missing not in _OPTIONAL_ROOTS:
            raise
        _GSPLATS_IMPORT_ERROR = exc

        def _raise_gsplats_import_error(_exc: ImportError = exc) -> None:
            """Raise a clear ImportError pointing at ``pip install luxar[gsplats]``."""
            raise ImportError(
                "luxar.gsplats requires optional dependencies (torch, scipy, etc.). "
                "Install with: pip install 'luxar[gsplats]'\n"
                f"Original error: {_exc}"
            ) from _exc

        def cull_by_contribution(*_args: Any, **_kwargs: Any) -> Any:
            """Cull splats by their rendered contribution (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        class CullResult:
            """Result of a contribution-based cull (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        def fit_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
            """Fit oriented Gaussian splats to a volume (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def fit_progressive_gaussian_splats(*_args: Any, **_kwargs: Any) -> Any:
            """Fit splats in iterative residual passes (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        class GaussianSplatFitter:
            """Stateful Gaussian-splat fitter (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class GSplatData:
            """In-memory Gaussian-splat dataset / node tree (needs gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class AdditiveSubLOD:
            """One additive (prefix-sum) LOD sub-level (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class SubstitutiveLevel:
            """One substitutive (coarse->fine) LOD level (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class DynamicOpsConfig:
            """Config for fixed-pool splat-relocation ops (needs gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        def generate_seeds(*_args: Any, **_kwargs: Any) -> Any:
            """Generate initial splat seeds from a volume (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def seed_from_decomposition(*_args: Any, **_kwargs: Any) -> Any:
            """Seed splats from a multi-scale image decomposition (gsplats extra)."""
            _raise_gsplats_import_error()

        def seed_from_grid(*_args: Any, **_kwargs: Any) -> Any:
            """Seed splats on a regular grid (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def seed_from_edges(*_args: Any, **_kwargs: Any) -> Any:
            """Seed splats from volume edges / gradients (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def fit_tile(*_args: Any, **_kwargs: Any) -> Any:
            """Fit a single spatial tile of a large volume (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def fit_tiled(*_args: Any, **_kwargs: Any) -> Any:
            """Fit a large volume tile-by-tile and stitch it (needs gsplats extra)."""
            _raise_gsplats_import_error()

        fit_tiled_gaussian_splats = fit_tiled

        def compute_tile_specs(*_args: Any, **_kwargs: Any) -> Any:
            """Plan the tile grid for a tiled fit (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def cosine_window(*_args: Any, **_kwargs: Any) -> Any:
            """Hann/cosine apodization window for seamless tiling (gsplats extra)."""
            _raise_gsplats_import_error()

        class TileSpec:
            """Geometry spec for one tile of a tiled fit (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        # Calibration (blind-spot CV)
        def cv_mask(*_args: Any, **_kwargs: Any) -> Any:
            """Build the blind-spot cross-validation mask (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def donut_median_fill(*_args: Any, **_kwargs: Any) -> Any:
            """Donut-median fill of masked voxels for blind-spot CV (gsplats extra)."""
            _raise_gsplats_import_error()

        def held_out_psnr(*_args: Any, **_kwargs: Any) -> Any:
            """Held-out PSNR at masked voxels for K calibration (gsplats extra)."""
            _raise_gsplats_import_error()

        def estimate_noise_floor(*_args: Any, **_kwargs: Any) -> Any:
            """Estimate the image noise floor / PSNR ceiling (needs gsplats extra)."""
            _raise_gsplats_import_error()

        def build_k_grid(*_args: Any, **_kwargs: Any) -> Any:
            """Build the splat-count (K) sweep grid for calibration (gsplats extra)."""
            _raise_gsplats_import_error()

        def find_k_star(*_args: Any, **_kwargs: Any) -> Any:
            """Locate the held-out PSNR peak K* in a sweep (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def calibrate(*_args: Any, **_kwargs: Any) -> Any:
            """Calibrate the splat count K via blind-spot CV (needs gsplats extra)."""
            _raise_gsplats_import_error()

        class NoiseFloor:
            """Estimated noise floor from calibration (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class HeldOutPeak:
            """Held-out PSNR peak (K*) from a sweep (needs the gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        class CalibrationResult:
            """Full calibration result: K*, noise floor, curve (needs gsplats extra)."""

            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                _raise_gsplats_import_error()

        # LOD (luxar.gsplats.lod)
        lod = None  # type: ignore[assignment]

        def compute_additive_order(*_args: Any, **_kwargs: Any) -> Any:
            """Order splats for an additive LOD ladder (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def make_additive_lod(*_args: Any, **_kwargs: Any) -> Any:
            """Build an additive (prefix-sum) LOD ladder (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def make_substitutive_lod(*_args: Any, **_kwargs: Any) -> Any:
            """Build substitutive coarse->fine LOD levels (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def make_lod_pyramid(*_args: Any, **_kwargs: Any) -> Any:
            """Build a full LOD pyramid over a fit (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        # Lift (luxar.gsplats.lift) — points -> gsplats
        def coarse_substitutive_levels(*_args: Any, **_kwargs: Any) -> Any:
            """Coarsen splats into substitutive levels (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def compute_ray_integral_factor(*_args: Any, **_kwargs: Any) -> Any:
            """Ray-integral calibration factor used when lifting (gsplats extra)."""
            _raise_gsplats_import_error()

        def lift_lines_to_gsplats(*_args: Any, **_kwargs: Any) -> Any:
            """Lift line geometry to Gaussian splats (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def lift_points_to_gsplats(*_args: Any, **_kwargs: Any) -> Any:
            """Lift point geometry to Gaussian splats (needs the gsplats extra)."""
            _raise_gsplats_import_error()

        def render_light(*_args: Any, **_kwargs: Any) -> Any:
            """Total emitted light of a splat set under sum projection.

            Σ aᵢ·|det Lᵢ| (needs the gsplats extra).
            """
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
