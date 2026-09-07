"""The :class:`CalibrationResult` container and its JSON (de)serialisation.

Output of :func:`luxar.gsplats.calibration.calibrate`. All fields are plain
data; non-finite floats round-trip through JSON as ``null`` and back to
``nan``/``inf`` on load, and every additive field default-hydrates so old
``cal.json`` files keep loading.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .curve_analysis import HeldOutPeak
from .noise_floor import NoiseFloor


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate`. Serialisable to JSON."""

    k_values_requested: List[int]
    """K values passed to the sweep."""

    k_values_effective: List[int]
    """Post-cull splat counts actually realised at each K."""

    held_out_psnr_db: List[float]
    """PSNR at masked positions, against the original pre-fill values."""

    train_psnr_db: List[float]
    """PSNR at unmasked positions, against the original values."""

    held_out_mse: List[float]
    """MSE at masked positions, against the original values."""

    full_psnr_db: List[float]
    """PSNR over the whole volume against the original — for cross-run comparison."""

    full_ssim: List[float]
    """SSIM over the whole volume against the original."""

    held_out_peak: HeldOutPeak
    """The recommended K* and curve type."""

    noise_floor: NoiseFloor
    """Ensemble noise-floor estimate for the input volume."""

    fit_times_seconds: List[float]
    """Wall-clock time per fit, in seconds."""

    splat_paths: Optional[List[str]]
    """Per-K ``.gsplats.zarr`` paths when ``--keep-fits`` is set; else ``None``."""

    mask_seed: int
    mask_fraction: float
    donut_radius: int

    fit_config: Dict[str, Any]
    """Fit kwargs that were applied (sans the per-K ``seeds`` value)."""

    volume_shape: List[int]
    volume_dtype: str
    timestamp: str

    # --- regime-robust extensions (all optional; old cal.json still loads) ---
    held_out_psnr_fg_db: List[float] = field(default_factory=list)
    """Foreground-restricted held-out PSNR (background-domination removed)."""
    held_out_psnr_fg_weighted_db: List[float] = field(default_factory=list)
    """Held-out PSNR with controlled foreground/background total weight."""
    foreground_mask_fraction: float = float("nan")
    """Fraction selected by the smoothed Otsu foreground mask."""
    foreground_otsu_threshold: float = float("nan")
    """Otsu cut on the lightly smoothed, floor-subtracted calibration volume."""
    fg_bg_ratio: float = 1.0
    """Foreground:background total-weight ratio for the weighted metric."""
    held_out_gain_db: List[float] = field(default_factory=list)
    """dB the fit beats the predict-zero baseline (plateaus meaningfully)."""
    predict_zero_baseline_mse: float = float("nan")
    """MSE of the trivial all-zeros reconstruction at masked voxels."""
    k_star_metric: str = "psnr_minmax"
    """Metric used for ``held_out_peak_selected``."""
    held_out_peak_selected: Optional[HeldOutPeak] = None
    """K* under ``k_star_metric`` (None when it equals the default psnr_minmax)."""
    calibration_region: Optional[Dict[str, Any]] = None
    """Provenance when an auto-selected sub-region was calibrated (else None)."""
    original_volume_shape: Optional[List[int]] = None
    """Shape of the full input before any region crop (disambiguates cropped PSNR)."""
    splat_density: Optional[Dict[str, Any]] = None
    """Transferable :class:`SplatDensity` (as dict) for the planner."""
    rd_model: Optional[Dict[str, Any]] = None
    """Parametric :class:`RDModel` (as dict) of held-out error vs K."""
    not_converged: bool = False
    """True when the held-out curve was still climbing at K_max (RD model)."""
    exponent_fit: Optional[Dict[str, Any]] = None
    """Multi-scale :class:`ExponentFit` (as dict) when ``cal --fit-exponent`` ran;
    its ``alpha`` is also written into ``splat_density.saturation_exponent``."""

    def to_json(self, path: Path) -> None:
        """Serialise to JSON. Non-finite floats become ``null``."""

        def _safe(v: Any) -> Any:
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
            if isinstance(v, np.generic):
                return _safe(v.item())
            if isinstance(v, (list, tuple)):
                return [_safe(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _safe(x) for k, x in v.items()}
            return v

        data = asdict(self)
        data = _safe(data)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationResult":
        """Load from JSON. ``null`` floats become ``nan``."""
        raw = json.loads(Path(path).read_text())

        def _hydrate_float_list(xs: List[Any]) -> List[float]:
            return [float("nan") if x is None else float(x) for x in xs]

        def _peak_from(d: dict) -> HeldOutPeak:
            # Additive fields hydrate with defaults; k_knee falls back to k_star
            # for cal.json written before the operating-point fields existed.
            k_star = int(d["k_star"])
            return HeldOutPeak(
                k_star=k_star,
                type=d["type"],
                confidence_db=float(d["confidence_db"]),
                k_knee=int(d.get("k_knee", k_star) or k_star),
                knee_idx=int(d.get("knee_idx", -1)),
                drop_after_peak_db=float(d.get("drop_after_peak_db", 0.0)),
                tail_rise_db=float(d.get("tail_rise_db", 0.0)),
                plateau_spread_db=float(d.get("plateau_spread_db", 0.0)),
                total_rise_db=float(d.get("total_rise_db", 0.0)),
                still_climbing=bool(d.get("still_climbing", False)),
                knee_margin_db=float(d.get("knee_margin_db", 0.3)),
            )

        peak = _peak_from(raw["held_out_peak"])
        nf_raw = raw["noise_floor"]
        nf = NoiseFloor(
            sigma_hat=float(nf_raw["sigma_hat"])
            if nf_raw["sigma_hat"] is not None
            else float("nan"),
            sigma_laplacian=float(nf_raw["sigma_laplacian"])
            if nf_raw["sigma_laplacian"] is not None
            else float("nan"),
            sigma_haar=float(nf_raw["sigma_haar"])
            if nf_raw["sigma_haar"] is not None
            else float("nan"),
            sigma_background=float(nf_raw["sigma_background"])
            if nf_raw["sigma_background"] is not None
            else float("nan"),
            psnr_max_db=float(nf_raw["psnr_max_db"])
            if nf_raw["psnr_max_db"] is not None
            else float("inf"),
        )
        sel_raw = raw.get("held_out_peak_selected")
        peak_selected = _peak_from(sel_raw) if sel_raw else None
        baseline = raw.get("predict_zero_baseline_mse")
        return cls(
            k_values_requested=[int(x) for x in raw["k_values_requested"]],
            k_values_effective=[int(x) for x in raw["k_values_effective"]],
            held_out_psnr_db=_hydrate_float_list(raw["held_out_psnr_db"]),
            train_psnr_db=_hydrate_float_list(raw["train_psnr_db"]),
            held_out_mse=_hydrate_float_list(raw["held_out_mse"]),
            full_psnr_db=_hydrate_float_list(raw["full_psnr_db"]),
            full_ssim=_hydrate_float_list(raw["full_ssim"]),
            held_out_peak=peak,
            noise_floor=nf,
            fit_times_seconds=_hydrate_float_list(raw["fit_times_seconds"]),
            splat_paths=raw.get("splat_paths"),
            mask_seed=int(raw["mask_seed"]),
            mask_fraction=float(raw["mask_fraction"]),
            donut_radius=int(raw["donut_radius"]),
            fit_config=dict(raw.get("fit_config", {})),
            volume_shape=[int(x) for x in raw["volume_shape"]],
            volume_dtype=str(raw["volume_dtype"]),
            timestamp=str(raw["timestamp"]),
            # --- regime-robust extensions (default-hydrated for old files) ---
            held_out_psnr_fg_db=_hydrate_float_list(raw.get("held_out_psnr_fg_db", [])),
            held_out_psnr_fg_weighted_db=_hydrate_float_list(
                raw.get("held_out_psnr_fg_weighted_db", [])
            ),
            foreground_mask_fraction=float(
                raw.get("foreground_mask_fraction")
                if raw.get("foreground_mask_fraction") is not None
                else float("nan")
            ),
            foreground_otsu_threshold=float(
                raw.get("foreground_otsu_threshold")
                if raw.get("foreground_otsu_threshold") is not None
                else float("nan")
            ),
            fg_bg_ratio=float(raw.get("fg_bg_ratio", 1.0)),
            held_out_gain_db=_hydrate_float_list(raw.get("held_out_gain_db", [])),
            predict_zero_baseline_mse=float("nan")
            if baseline is None
            else float(baseline),
            k_star_metric=str(raw.get("k_star_metric", "psnr_minmax")),
            held_out_peak_selected=peak_selected,
            calibration_region=raw.get("calibration_region"),
            original_volume_shape=(
                [int(x) for x in raw["original_volume_shape"]]
                if raw.get("original_volume_shape") is not None
                else None
            ),
            splat_density=_rehydrate_nan_dict(raw.get("splat_density")),
            rd_model=_rehydrate_nan_dict(raw.get("rd_model")),
            not_converged=bool(raw.get("not_converged", False)),
            exponent_fit=_rehydrate_nan_dict(raw.get("exponent_fit")),
        )


def _rehydrate_nan_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Restore float-symmetry on reload: ``to_json`` writes NaN/inf as ``null``,
    so a ``None`` inside a nested ``splat_density`` / ``rd_model`` dict means a
    non-finite float. Map it back to ``nan`` so consumers (planner) don't choke
    on ``float(None)``. Keys (all-string) are never None, so this is safe."""
    if d is None:
        return None
    return {k: (float("nan") if v is None else v) for k, v in d.items()}


def _json_safe(v: Any) -> bool:
    """Return True iff ``v`` survives ``json.dumps`` (used to filter fit_config)."""
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False
