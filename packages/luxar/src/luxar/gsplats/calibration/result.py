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
    """SSIM over the whole volume against the original, on the full range
    (``max - min``); see ``full_ssim_robust`` for the robust-peak figure."""

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
    """K* under ``k_star_metric``.

    None when the metric is the default ``psnr_minmax``, or when the selected
    curve was undefined and K* fell back to min-max PSNR.
    """
    calibration_region: Optional[Dict[str, Any]] = None
    """Provenance when an auto-selected sub-region was calibrated (else None)."""
    original_volume_shape: Optional[List[int]] = None
    """Shape of the loaded input before any ``--auto-region`` crop.

    Disambiguates cropped PSNR. Under ``cal --region`` the load itself is
    cropped, so this is the ``--region`` crop's shape (the full input is not read);
    ``input_frame["source_region"]`` says where that crop sits in the input.
    """
    input_frame: Optional[Dict[str, Any]] = None
    """The voxel spacing and ``--region`` crop the sweep ran on (``cal`` CLI).

    Keys as in a fitted store's ``fitting/`` group: ``voxel_size``,
    ``voxel_unit``, ``voxel_size_source``, ``physical_coordinates`` (always
    false: calibration fits stay on the voxel grid) and ``source_region``.
    ``None`` for a plain voxel-unit, whole-volume calibration.
    """
    splat_density: Optional[Dict[str, Any]] = None
    """Transferable :class:`SplatDensity` (as dict) for the planner."""
    rd_model: Optional[Dict[str, Any]] = None
    """Parametric :class:`RDModel` (as dict) of held-out error vs K."""
    not_converged: bool = False
    """True when the held-out curve was still climbing at K_max (RD model)."""
    exponent_fit: Optional[Dict[str, Any]] = None
    """Multi-scale :class:`ExponentFit` (as dict) when ``cal --fit-exponent`` ran;
    its ``alpha`` is also written into ``splat_density.saturation_exponent``."""
    sweep: Optional[Dict[str, Any]] = None
    """What the sweep did beyond the grid it was handed (``SweepControl`` as a
    dict): whether early stopping ended it and which K were skipped, the points
    added below the grid when K* landed on its bottom edge, and where the grid
    came from (explicit / bounds / data-derived ``auto``, with provenance).
    ``None`` when the caller ran a plain fixed grid."""
    status: str = "complete"
    """``complete``, ``partial`` (a per-point checkpoint written while the sweep
    runs — what a crash leaves behind) or ``interrupted`` (Ctrl-C; the points
    measured so far)."""
    held_out_psnr_robust_db: List[float] = field(default_factory=list)
    """Held-out PSNR on the ROBUST peak (``p99.99 - min`` of the calibrated
    volume) — the convention ``fit`` and ``compare`` headline. A constant offset
    from ``held_out_psnr_db`` (the full-range ``max - min`` peak), so it selects
    the same K*."""
    held_out_psnr_fg_robust_db: List[float] = field(default_factory=list)
    """Foreground held-out PSNR on the robust peak (offset of
    ``held_out_psnr_fg_db``)."""
    data_range: float = float("nan")
    """Full-range peak (``max - min``) of the calibrated volume."""
    robust_data_range: float = float("nan")
    """Robust peak (``p99.99 - min``) of the calibrated volume."""
    full_ssim_robust: List[float] = field(default_factory=list)
    """Whole-volume SSIM on the ROBUST peak, the SSIM ``fit`` / ``compare``
    headline. ``full_ssim`` keeps the full range, which one hot pixel inflates
    (SSIM's constants scale with the range). Empty in older files."""
    k_values_fitted: List[int] = field(default_factory=list)
    """Splats each fit ended with, before any closing reduction."""
    k_values_stored: List[int] = field(default_factory=list)
    """Splats ``fit --seeds K`` would STORE for that fit: the default speck
    pruning (``min_extent``) then the cumulative cull (``cull_retention``).
    ``k_values_effective`` is after the cull only (the scored model)."""
    k_star_stored: Optional[int] = None
    """``k_values_stored`` at the selected K*: what ``fit --seeds K*`` keeps.
    ``K*`` itself is a SEEDING budget, not a stored count."""
    noise_sigma_rms: float = float("nan")
    """RMS noise sigma of the calibrated volume, in its intensity units.

    The in-plane Haar HH band binned by local intensity and pooled as an RMS
    (:func:`~.noise_floor.estimate_masked_noise` over the whole volume), measured
    on the floor-subtracted volume the held-out scores use as their reference.
    It is what a held-out MSE averages:
    fluorescence noise grows with the signal, so the bright voxels set it, while
    ``noise_floor.sigma_hat`` (a median) reads the dark background. ``nan`` for a
    ``cal.json`` that predates it."""
    noise_scale_ratio: float = float("nan")
    """Coarse- over fine-scale sigma of :attr:`noise_sigma_rms`; above
    :data:`~.noise_floor.MASKED_SIGMA_MAX_SCALE_RATIO` the finest band also holds
    signal, so the sigma reads high and the ceiling below reads low."""
    psnr_ceiling_robust_db: float = float("nan")
    """Held-out PSNR ceiling on the robust peak (the table's basis):
    ``20 log10(robust_data_range / noise_sigma_rms)``. A held-out score cannot
    sit far above it, since the held-out error includes the noise itself."""
    spot_count: Optional[int] = None
    """Puncta detected on the floor-subtracted volume (:mod:`.spots`); ``None``
    for a ``cal.json`` that predates the check or a volume with no spread."""
    spot_peak_ratio: List[float] = field(default_factory=list)
    """Per K: median rendered/reference peak at the detected puncta (blurred,
    ``sigma = 1`` voxel). Report-only; ``cal`` warns when K* reaches under 95% of
    the sweep's best on a punctate volume (``luxar.gsplats.calibration.spots``)."""
    whole_volume_seeds: Optional[Dict[str, Any]] = None
    """For a crop calibration (``cal --region`` / ``--auto-region``): the
    whole-input ``fit --seeds`` budget — the content planner's per-box density
    transfer summed over crop-sized boxes of the input — with its provenance
    (``seeds``, ``n_boxes``, feature counts, ``rule``). ``None`` otherwise."""

    def robust_offset_db(self) -> float:
        """dB to ADD to a full-range PSNR to restate it on the robust peak.

        PSNR is ``10 log10(peak^2 / MSE)`` and the peak is one constant per
        volume, so moving from the full range (``max - min``) to the robust
        peak (``p99.99 - min``, the ``fit``/``compare`` headline) is a constant
        ``20 log10(robust / full)``: every curve shifts together and K* does
        not move. ``0`` for a ``cal.json`` that predates the robust peak.
        """
        full, robust = self.data_range, self.robust_data_range
        if math.isfinite(full) and math.isfinite(robust) and full > 0 and robust > 0:
            return 20.0 * math.log10(robust / full)
        return 0.0

    def incomplete_warning(self) -> Optional[str]:
        """A loud warning when this calibration did not finish, else ``None``.

        A ``partial`` file is a per-point checkpoint (a crashed or still
        running ``cal``); an ``interrupted`` one holds the points measured
        before Ctrl-C. Either way K*, the density and its saturation cap come
        from a truncated sweep, so a consumer must say so.
        """
        if self.status == "complete":
            return None
        what = (
            "a per-point checkpoint of a calibration that crashed or is still running"
            if self.status == "partial"
            else "a calibration interrupted by Ctrl-C"
        )
        return (
            f"⚠ cal.json status is {self.status!r}: {what}, measured at K = "
            f"{self.k_values_requested}. Its K* and splat density come from that "
            "truncated sweep; re-run `gsplat cal` to completion for a reliable "
            "budget."
        )

    def to_json(self, path: Path) -> None:
        """Serialise to JSON (atomically). Non-finite floats become ``null``."""

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
        _atomic_write_text(Path(path), json.dumps(data, indent=2))

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
                warnings=[str(w) for w in d.get("warnings", []) or []],
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
            input_frame=raw.get("input_frame"),
            splat_density=_rehydrate_nan_dict(raw.get("splat_density")),
            rd_model=_rehydrate_nan_dict(raw.get("rd_model")),
            not_converged=bool(raw.get("not_converged", False)),
            exponent_fit=_rehydrate_nan_dict(raw.get("exponent_fit")),
            sweep=raw.get("sweep"),
            status=str(raw.get("status", "complete")),
            held_out_psnr_robust_db=_hydrate_float_list(
                raw.get("held_out_psnr_robust_db", [])
            ),
            held_out_psnr_fg_robust_db=_hydrate_float_list(
                raw.get("held_out_psnr_fg_robust_db", [])
            ),
            data_range=_opt_float(raw.get("data_range")),
            robust_data_range=_opt_float(raw.get("robust_data_range")),
            full_ssim_robust=_hydrate_float_list(raw.get("full_ssim_robust", [])),
            k_values_fitted=[int(x) for x in raw.get("k_values_fitted", [])],
            k_values_stored=[int(x) for x in raw.get("k_values_stored", [])],
            whole_volume_seeds=raw.get("whole_volume_seeds"),
            spot_count=(
                int(raw["spot_count"]) if raw.get("spot_count") is not None else None
            ),
            spot_peak_ratio=_hydrate_float_list(raw.get("spot_peak_ratio", [])),
            noise_sigma_rms=_opt_float(raw.get("noise_sigma_rms")),
            noise_scale_ratio=_opt_float(raw.get("noise_scale_ratio")),
            psnr_ceiling_robust_db=_opt_float(raw.get("psnr_ceiling_robust_db")),
            k_star_stored=(
                int(raw["k_star_stored"])
                if raw.get("k_star_stored") is not None
                else None
            ),
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via a temp file in the same directory.

    ``gsplat cal`` rewrites ``cal.json`` after every grid point; a crash or a
    Ctrl-C mid-write must leave the previous complete file, never a truncated
    one, so the bytes land in a sibling temp file that ``os.replace`` (atomic
    on one filesystem) swaps in.
    """
    import os
    import secrets
    import stat

    tmp = path.with_name(f".{path.name}.{secrets.token_hex(12)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _opt_float(x: Any) -> float:
    """``null`` (a non-finite float on write) back to ``nan``."""
    return float("nan") if x is None else float(x)


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
