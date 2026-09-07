"""PDF report for ``luxar gsplat cal`` results.

Produces a multi-page matplotlib PDF mirroring the per-dataset figures
of ``manuscript/supp_doc/splat_count_vs_quality/``:

* Page 1 — Rate-distortion: PSNR / SSIM / fit time / train-vs-held-out
  gap, all vs K, with K* annotated and the noise-floor PSNR ceiling
  overlaid.
* Page 2 — Blind-spot cross-validation: train + held-out PSNR with the
  overfitting region shaded.
* Page 3 — Slice montages (only when per-K fits were persisted via
  ``--keep-fits``): target / reconstruction at ``K_min`` / ``K*`` /
  ``K_max`` plus a per-pixel error map. Skipped gracefully otherwise.

This module is loaded only when ``--pdf`` is set so matplotlib does not
inflate the cold-start cost of every CLI invocation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from luxar.gsplats.calibration import CalibrationResult, HeldOutPeak

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _safe_log_x(ax: "Axes") -> None:
    """Use a log x-axis with sensible ticks for K values."""
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)


def _selected_metric(
    result: CalibrationResult,
) -> tuple[np.ndarray, str, HeldOutPeak, str]:
    """Return the selected curve, label, peak, and resolved metric name."""
    curves = {
        "psnr_minmax": (result.held_out_psnr_db, "held-out PSNR"),
        "psnr_foreground": (
            result.held_out_psnr_fg_db,
            "foreground-only held-out PSNR",
        ),
        "psnr_fg_weighted": (
            result.held_out_psnr_fg_weighted_db,
            "foreground-weighted held-out PSNR",
        ),
        "gain": (result.held_out_gain_db, "held-out gain over predict-zero"),
    }
    resolved_metric = (
        result.k_star_metric if result.k_star_metric in curves else "psnr_minmax"
    )
    values, label = curves[resolved_metric]
    if (
        result.k_star_metric != "psnr_minmax" and result.held_out_peak_selected is None
    ) or len(values) != len(result.k_values_requested):
        values, label = result.held_out_psnr_db, "held-out PSNR"
        peak = result.held_out_peak
        resolved_metric = "psnr_minmax"
    else:
        peak = result.held_out_peak_selected or result.held_out_peak
    return np.asarray(values, dtype=float), label, peak, resolved_metric


def _knee_display_idx(result: CalibrationResult) -> Optional[int]:
    """Index of the operating point (``k_knee``) in the sweep, or None.

    Returns None when the knee coincides with ``k_star`` (peak/plateau, or
    legacy cal.json hydrated with the fallback) — the report then shows only
    the K* marker, mirroring the CLI which prints the operating point only
    when it differs.
    """
    _curve, _label, peak, _resolved_metric = _selected_metric(result)
    if not peak.k_knee or peak.k_knee == peak.k_star:
        return None
    if peak.k_knee not in result.k_values_requested:
        return None
    return result.k_values_requested.index(peak.k_knee)


def _plot_rate_distortion(
    fig: "Figure",
    ax_psnr: "Axes",
    ax_ssim: "Axes",
    ax_time: "Axes",
    ax_gap: "Axes",
    result: CalibrationResult,
) -> None:
    ks = np.asarray(result.k_values_effective, dtype=float)
    held = np.asarray(result.held_out_psnr_db, dtype=float)
    selected, selected_label, selected_peak, resolved_metric = _selected_metric(result)
    train = np.asarray(result.train_psnr_db, dtype=float)
    full = np.asarray(result.full_psnr_db, dtype=float)
    ssim = np.asarray(result.full_ssim, dtype=float)
    fit_t = np.asarray(result.fit_times_seconds, dtype=float)
    k_star = selected_peak.k_star
    psnr_ceiling = result.noise_floor.psnr_max_db

    # PSNR panel
    ax_psnr.plot(ks, held, "o-", color="C1", label="held-out")
    if resolved_metric not in ("psnr_minmax", "gain"):
        ax_psnr.plot(ks, selected, "d-", color="C6", label=selected_label)
    ax_psnr.plot(ks, train, "s--", color="C0", label="train")
    ax_psnr.plot(ks, full, "v:", color="C2", label="full volume")
    if math.isfinite(psnr_ceiling):
        ax_psnr.axhline(psnr_ceiling, color="grey", linestyle=":", linewidth=1)
        ax_psnr.text(
            ks[0],
            psnr_ceiling,
            f"  noise floor = {psnr_ceiling:.1f} dB",
            va="bottom",
            ha="left",
            fontsize=8,
            color="grey",
        )
    star_idx = result.k_values_requested.index(k_star)
    displayed_selected = held if resolved_metric == "gain" else selected
    ax_psnr.plot(
        ks[star_idx],
        displayed_selected[star_idx],
        marker="*",
        color="C3",
        markersize=18,
        zorder=10,
        label=f"K*={k_star:,}",
    )
    knee_idx = _knee_display_idx(result)
    if knee_idx is not None:
        ax_psnr.plot(
            ks[knee_idx],
            displayed_selected[knee_idx],
            marker="D",
            color="C6",
            markersize=10,
            zorder=9,
            label=f"knee={selected_peak.k_knee:,}",
        )
    _safe_log_x(ax_psnr)
    ax_psnr.set_xlabel("Effective splat count K")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("Reconstruction PSNR vs K")
    ax_psnr.legend(loc="best", fontsize=9)

    # SSIM panel
    ax_ssim.plot(ks, ssim, "o-", color="C2")
    ax_ssim.plot(
        ks[star_idx], ssim[star_idx], marker="*", color="C3", markersize=18, zorder=10
    )
    _safe_log_x(ax_ssim)
    ax_ssim.set_xlabel("Effective splat count K")
    ax_ssim.set_ylabel("SSIM")
    ax_ssim.set_title("Reconstruction SSIM vs K")
    ax_ssim.set_ylim(min(0.0, ssim.min() - 0.05), 1.01)

    # Fit time panel
    ax_time.plot(ks, fit_t, "o-", color="C4")
    _safe_log_x(ax_time)
    ax_time.set_xlabel("Effective splat count K")
    ax_time.set_ylabel("Fit time (s)")
    ax_time.set_title("Wall-clock fit time vs K")

    # Train vs held-out gap (overfitting diagnostic)
    gap = train - held
    ax_gap.plot(ks, gap, "o-", color="C5")
    ax_gap.axhline(0.0, color="grey", linestyle="-", linewidth=0.8)
    ax_gap.fill_between(ks, gap, 0.0, where=gap > 0, alpha=0.2, color="C3")
    ax_gap.plot(
        ks[star_idx], gap[star_idx], marker="*", color="C3", markersize=18, zorder=10
    )
    _safe_log_x(ax_gap)
    ax_gap.set_xlabel("Effective splat count K")
    ax_gap.set_ylabel("PSNR_train − PSNR_held-out (dB)")
    ax_gap.set_title("Train/held-out gap (positive = overfitting)")


def _plot_blind_spot(fig: "Figure", ax: "Axes", result: CalibrationResult) -> None:
    ks = np.asarray(result.k_values_effective, dtype=float)
    held, held_label, selected_peak, resolved_metric = _selected_metric(result)
    train = np.asarray(result.train_psnr_db, dtype=float)
    psnr_ceiling = result.noise_floor.psnr_max_db
    k_star = selected_peak.k_star
    star_idx = result.k_values_requested.index(k_star)

    if resolved_metric == "psnr_minmax":
        ax.plot(ks, train, "s-", color="C0", label="train")
    ax.plot(ks, held, "o-", color="C1", label=f"{held_label} (model selection)")
    if resolved_metric == "psnr_minmax":
        ax.fill_between(
            ks, train, held, where=(train > held).tolist(), alpha=0.15, color="C3"
        )

    if resolved_metric == "psnr_minmax" and math.isfinite(psnr_ceiling):
        ax.axhline(psnr_ceiling, color="grey", linestyle=":", linewidth=1)
        ax.text(
            ks[0],
            psnr_ceiling,
            f"  noise floor = {psnr_ceiling:.1f} dB",
            va="bottom",
            ha="left",
            fontsize=8,
            color="grey",
        )

    ax.plot(
        ks[star_idx],
        held[star_idx],
        marker="*",
        color="C3",
        markersize=22,
        zorder=10,
        label=f"K* = {k_star:,} ({selected_peak.type})",
    )
    knee_idx = _knee_display_idx(result)
    if knee_idx is not None:
        ax.plot(
            ks[knee_idx],
            held[knee_idx],
            marker="D",
            color="C6",
            markersize=12,
            zorder=9,
            label=f"operating point (knee) = {selected_peak.k_knee:,}",
        )
    _safe_log_x(ax)
    ax.set_xlabel("Effective splat count K")
    ax.set_ylabel("Gain (dB)" if resolved_metric == "gain" else "PSNR (dB)")
    ax.set_title("Blind-spot cross-validation")
    ax.legend(loc="best", fontsize=10)


def _render_splat_path(splat_path: str, shape: Sequence[int]) -> Optional[np.ndarray]:
    """Load a persisted ``.gsplats.zarr`` and render to numpy. Returns None on failure."""
    try:
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.rendering.volume_rendering import render_to_volume
    except ImportError:
        return None
    try:
        data = GSplatData.load(splat_path, include_stats=False)
        rendered = render_to_volume(data, shape=tuple(shape), device="cpu")
        return np.asarray(rendered)
    except Exception:
        return None


def _mid_slices(V: np.ndarray) -> List[np.ndarray]:
    """For 3D V, return three orthogonal mid-slices. For 2D, a single slice."""
    if V.ndim == 2:
        return [V]
    if V.ndim == 3:
        z, y, x = (s // 2 for s in V.shape)
        return [V[z], V[:, y], V[..., x]]
    # Higher D: project to 3D by taking the middle along leading axes
    while V.ndim > 3:
        V = V[V.shape[0] // 2]
    return _mid_slices(V)


def _plot_slice_montage(
    fig: "Figure",
    axes_grid: Any,
    volume: np.ndarray,
    rendered_low: Optional[np.ndarray],
    rendered_star: Optional[np.ndarray],
    rendered_high: Optional[np.ndarray],
    k_low: int,
    k_star: int,
    k_high: int,
) -> None:
    """Five-column montage: target / K_low / K* / K_high / error map.

    ``axes_grid`` is a 2D array of axes returned by ``plt.subplots``,
    typically with ``nrows=3`` (one per orthogonal slice) and ``ncols=5``.
    Missing renderings are filled with placeholder text.
    """
    target_slices = _mid_slices(volume)
    rec_low_slices = _mid_slices(rendered_low) if rendered_low is not None else None
    rec_star_slices = _mid_slices(rendered_star) if rendered_star is not None else None
    rec_high_slices = _mid_slices(rendered_high) if rendered_high is not None else None
    n_rows = len(target_slices)

    vmin = float(volume.min())
    vmax = float(volume.max())

    titles = [
        "Target",
        f"K = {k_low:,}",
        f"K* = {k_star:,}",
        f"K = {k_high:,}",
        "abs(K* − target)",
    ]

    for r in range(n_rows):
        cols = [
            target_slices[r],
            rec_low_slices[r] if rec_low_slices is not None else None,
            rec_star_slices[r] if rec_star_slices is not None else None,
            rec_high_slices[r] if rec_high_slices is not None else None,
        ]
        for c, panel in enumerate(cols):
            ax = axes_grid[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if panel is None:
                ax.text(
                    0.5,
                    0.5,
                    "fit not\npersisted\n(use --keep-fits)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
            else:
                ax.imshow(panel, vmin=vmin, vmax=vmax, cmap="magma")
            if r == 0:
                ax.set_title(titles[c], fontsize=9)

        # Error column
        ax = axes_grid[r, 4]
        ax.set_xticks([])
        ax.set_yticks([])
        if rec_star_slices is None:
            ax.text(
                0.5,
                0.5,
                "needs --keep-fits",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
        else:
            err = np.abs(rec_star_slices[r] - target_slices[r])
            ax.imshow(err, cmap="inferno")
        if r == 0:
            ax.set_title(titles[4], fontsize=9)


def render_calibration_report(
    result: CalibrationResult,
    volume: np.ndarray,
    output_path: Path,
    splat_paths: Optional[List[str]] = None,
) -> None:
    """Generate a multi-page PDF calibration report.

    Parameters
    ----------
    result
        Output of :func:`luxar.gsplats.calibration.calibrate`.
    volume
        The original (pre-mask) input volume — used as the "target" panel
        in the slice montage.
    output_path
        Destination ``.pdf``.
    splat_paths
        Optional list of per-K ``.gsplats.zarr`` paths in the same order
        as ``result.k_values_requested``. When provided, the third page
        of the PDF includes reconstruction slice montages at K_min, K*,
        K_max. When ``None``, page 3 is replaced with a placeholder.

    Raises
    ------
    ImportError
        If matplotlib is not installed. The CLI handler catches this and
        prints a hint to install the optional dependency.
    """
    import matplotlib.pyplot as plt  # noqa: F401  (raises ImportError if missing)
    from matplotlib.backends.backend_pdf import PdfPages

    if len(result.k_values_requested) < 1:
        raise ValueError("CalibrationResult contains no K values")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _selected_curve, _selected_label, selected_peak, resolved_metric = _selected_metric(
        result
    )
    knee_idx = _knee_display_idx(result)

    with PdfPages(output_path) as pdf:
        # ── Page 1: rate-distortion (4 panels) ────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        suptitle = (
            f"Calibration: {tuple(result.volume_shape)} volume — "
            f"K* = {selected_peak.k_star:,} "
            f"(metric: {resolved_metric}, type: {selected_peak.type})"
        )
        if knee_idx is not None:
            suptitle += f" — operating point = {selected_peak.k_knee:,}"
        if resolved_metric != "psnr_minmax":
            suptitle += (
                f"\npsnr_minmax K* = {result.held_out_peak.k_star:,} "
                f"(type: {result.held_out_peak.type})"
            )
        fig.suptitle(suptitle, fontsize=12)
        _plot_rate_distortion(
            fig,
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
            result,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: blind-spot cross-validation ────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        _plot_blind_spot(fig, ax, result)
        # Annotation: peak detection summary
        nf = result.noise_floor
        annotation = (
            f"K* = {selected_peak.k_star:,}  "
            f"(metric: {resolved_metric}, type: {selected_peak.type}, "
            f"confidence: {selected_peak.confidence_db:.2f} dB)\n"
        )
        if knee_idx is not None:
            annotation += (
                f"operating point (diminishing returns) = "
                f"{selected_peak.k_knee:,} splats\n"
            )
        if resolved_metric != "psnr_minmax":
            annotation += (
                f"psnr_minmax K* = {result.held_out_peak.k_star:,} "
                f"(type: {result.held_out_peak.type})\n"
            )
        annotation += (
            f"σ̂ = {nf.sigma_hat:.4f}, "
            f"ceiling = {'>60' if math.isinf(nf.psnr_max_db) else f'{nf.psnr_max_db:.1f}'} dB"
        )
        fig.text(
            0.02,
            0.02,
            annotation,
            fontsize=9,
            family="monospace",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: slice montages (gracefully skipped without --keep-fits) ──
        if splat_paths is not None and len(splat_paths) == len(
            result.k_values_requested
        ):
            ks_req = result.k_values_requested
            k_low = ks_req[0]
            k_high = ks_req[-1]
            star_idx = ks_req.index(selected_peak.k_star)
            shape = tuple(result.volume_shape)

            rendered_low = _render_splat_path(splat_paths[0], shape)
            rendered_star = _render_splat_path(splat_paths[star_idx], shape)
            rendered_high = _render_splat_path(splat_paths[-1], shape)

            n_rows = 1 if volume.ndim == 2 else 3
            fig, axes_grid = plt.subplots(
                n_rows,
                5,
                figsize=(15, 3.2 * n_rows),
                squeeze=False,
            )
            _plot_slice_montage(
                fig,
                axes_grid,
                volume,
                rendered_low,
                rendered_star,
                rendered_high,
                k_low,
                selected_peak.k_star,
                k_high,
            )
            fig.suptitle(
                f"Reconstruction slices  —  target vs K = {k_low:,} / "
                f"K* = {selected_peak.k_star:,} / K = {k_high:,}",
                fontsize=11,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(fig)
            plt.close(fig)
        else:
            # Placeholder page
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "Slice montages require per-K fits to be persisted.\n\n"
                "Re-run with `--keep-fits <dir>` to enable this page.",
                ha="center",
                va="center",
                fontsize=12,
            )
            pdf.savefig(fig)
            plt.close(fig)

        # PDF metadata
        d = pdf.infodict()
        d["Title"] = f"Luxar calibration — {tuple(result.volume_shape)}"
        d["Subject"] = (
            f"Recommended K = {selected_peak.k_star} "
            f"({resolved_metric}, {selected_peak.type})"
        )
        d["Creator"] = "luxar gsplat cal"
