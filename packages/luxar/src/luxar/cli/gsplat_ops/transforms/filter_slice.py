"""Implementation helpers for filter/slice gsplat edit commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ...utils import format_memory_size
from ..encoding import _resolve_encoding_mode
from .parsing import (
    parse_axis_list,
    parse_bbox,
    parse_slices,
    parse_threshold,
)


def run_filter_dataset(
    *,
    input_path: Path,
    output_path: Path,
    bbox: Optional[str],
    volume_min: Optional[str],
    volume_max: Optional[str],
    volume_normalized: bool,
    scale_min: Optional[str],
    scale_max: Optional[str],
    scale_normalized: bool,
    amplitude_min: Optional[str],
    amplitude_max: Optional[str],
    amplitude_normalized: bool,
    eccentricity_min: Optional[str],
    eccentricity_max: Optional[str],
    mass_min: Optional[str],
    mass_max: Optional[str],
    mass_normalized: bool,
    sigma_axis: Optional[int],
    sigma_min: Optional[str],
    sigma_max: Optional[str],
    isolation_max: Optional[str],
    min_neighbors: Optional[int],
    neighbor_radius: Optional[float],
    spatial_dims: Optional[str],
    soft_highpass: Optional[str],
    soft_lowpass: Optional[str],
    soft_width: float,
    dry_run: bool,
    truncate: Optional[float],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run filter command implementation."""
    try:
        import numpy as np

        from luxar.cli.gsplat_ops.loading import load_matrix_gsplats

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Filtering: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = load_matrix_gsplats(
                    input_path,
                    include_stats=True,
                    command="filter",
                )
                n_original = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_original:,} splats ({ndim}D)")

            if truncate is None:
                truncate = data.truncation_radius

            bbox_parsed = parse_bbox(bbox, ndim) if bbox is not None else None

            # Parse thresholds ("pNN"/"NN%" → percentile flag) once.
            vmin, vmin_p = parse_threshold(volume_min, "volume-min")
            vmax, vmax_p = parse_threshold(volume_max, "volume-max")
            scmin, scmin_p = parse_threshold(scale_min, "scale-min")
            scmax, scmax_p = parse_threshold(scale_max, "scale-max")
            amin, amin_p = parse_threshold(amplitude_min, "amplitude-min")
            amax, amax_p = parse_threshold(amplitude_max, "amplitude-max")
            emin, emin_p = parse_threshold(eccentricity_min, "eccentricity-min")
            emax, emax_p = parse_threshold(eccentricity_max, "eccentricity-max")
            mmin, mmin_p = parse_threshold(mass_min, "mass-min")
            mmax, mmax_p = parse_threshold(mass_max, "mass-max")
            smin, smin_p = parse_threshold(sigma_min, "sigma-min")
            smax, smax_p = parse_threshold(sigma_max, "sigma-max")
            imax, imax_p = parse_threshold(isolation_max, "isolation-max")

            axes = (
                parse_axis_list(spatial_dims, ndim, "spatial-dims")
                if spatial_dims is not None
                else None
            )

            # A single percentile flag per attribute is shared by its min & max
            # (mirrors the existing *_normalized convention). Mixing modes on one
            # attribute (e.g. --volume-min p90 --volume-max 0.5) would silently
            # read the absolute value as a percentile, so reject it up front.
            for name, lo, lo_p, hi, hi_p in (
                ("volume", vmin, vmin_p, vmax, vmax_p),
                ("scale", scmin, scmin_p, scmax, scmax_p),
                ("amplitude", amin, amin_p, amax, amax_p),
                ("eccentricity", emin, emin_p, emax, emax_p),
                ("mass", mmin, mmin_p, mmax, mmax_p),
                ("sigma", smin, smin_p, smax, smax_p),
            ):
                if lo is not None and hi is not None and lo_p != hi_p:
                    raise typer.BadParameter(
                        f"--{name}-min and --{name}-max must use the same mode: "
                        f"both percentile ('pNN') or both absolute."
                    )

            # A single percentile flag per attribute (min/max share it).
            filter_kwargs: dict[str, Any] = dict(
                bbox=bbox_parsed,
                volume_min=vmin,
                volume_max=vmax,
                volume_normalized=volume_normalized,
                volume_percentile=vmin_p or vmax_p,
                scale_min=scmin,
                scale_max=scmax,
                scale_normalized=scale_normalized,
                scale_percentile=scmin_p or scmax_p,
                amplitude_min=amin,
                amplitude_max=amax,
                amplitude_normalized=amplitude_normalized,
                amplitude_percentile=amin_p or amax_p,
                eccentricity_min=emin,
                eccentricity_max=emax,
                eccentricity_percentile=emin_p or emax_p,
                mass_min=mmin,
                mass_max=mmax,
                mass_normalized=mass_normalized,
                mass_percentile=mmin_p or mmax_p,
                sigma_axis=sigma_axis,
                sigma_min=smin,
                sigma_max=smax,
                sigma_percentile=smin_p or smax_p,
                isolation_max=imax,
                isolation_percentile=imax_p,
                min_neighbors=min_neighbors,
                neighbor_radius=neighbor_radius,
                spatial_dims=axes,
                truncate=truncate,
            )

            with asection("Filtering"):
                filtered_data = data.filter_by(**filter_kwargs)
                n_filtered = filtered_data.n_splats
                n_removed = n_original - n_filtered

                # Impact vs the ORIGINAL (mass = integrated brightness; the key
                # GSIP signal is mass-removed ≫ amplitude-removed = diffuse haze).
                orig_mass = float(data.masses().sum())
                orig_amp = float(np.asarray(data.amplitudes).sum())
                kept_mass = float(filtered_data.masses().sum()) if n_filtered else 0.0
                kept_amp = (
                    float(np.asarray(filtered_data.amplitudes).sum())
                    if n_filtered
                    else 0.0
                )

                def pct(a: float, b: float) -> float:
                    return 100.0 * (1.0 - a / b) if b > 0 else 0.0

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Filtered splats: {n_filtered:,}")
                aprint(
                    f"  Removed:         {n_removed:,} "
                    f"({100 * n_removed / max(n_original, 1):.2f}%)"
                )
                aprint(f"  Mass removed:      {pct(kept_mass, orig_mass):.2f}%")
                aprint(f"  Amplitude removed: {pct(kept_amp, orig_amp):.2f}%")

            # Soft reweighting (attenuate, not remove) — resolve percentile
            # cutoffs against the post-hard-filter scale distribution.
            soft_active = soft_highpass is not None or soft_lowpass is not None
            if soft_active and n_filtered > 0:
                scl = filtered_data.scale(axes=axes)
                hp_v, hp_p = parse_threshold(soft_highpass, "soft-highpass")
                lp_v, lp_p = parse_threshold(soft_lowpass, "soft-lowpass")
                hp = (
                    float(np.percentile(scl, hp_v))
                    if hp_p and hp_v is not None
                    else hp_v
                )
                lp = (
                    float(np.percentile(scl, lp_v))
                    if lp_p and lp_v is not None
                    else lp_v
                )
                with asection("Soft reweighting"):
                    pre_mass = float(filtered_data.masses().sum())
                    filtered_data = filtered_data.soft_scale_filter(
                        highpass=hp, lowpass=lp, width=soft_width, spatial_dims=axes
                    )
                    post_mass = float(filtered_data.masses().sum())
                    aprint(
                        f"  soft highpass={hp} lowpass={lp} width={soft_width} → "
                        f"mass attenuated {pct(post_mass, pre_mass):.2f}%"
                    )

            if dry_run:
                aprint("\n(dry-run: nothing written)")
                return

            from luxar.gsplats.io.load_gsplats import read_rebuild_root_attrs

            with asection(f"Saving to {output_path.name}"):
                if n_filtered == 0:
                    aprint("⚠ No splats remain after filtering — skipping save")
                else:
                    filtered_data.save(
                        output_path,
                        amplitude_bits="auto",
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                        root_attrs=read_rebuild_root_attrs(input_path),
                    )
                    aprint(f"Saved filtered dataset: {output_path}")
                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:
        exit_with_error(f"❌ Error: {e}", e)


def run_slice_dataset(
    *,
    input_path: Path,
    output_path: Path,
    ranges: str,
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run slice command implementation."""
    try:
        from luxar.cli.gsplat_ops.loading import load_matrix_gsplats

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Slicing: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = load_matrix_gsplats(
                    input_path,
                    include_stats=True,
                    command="slice",
                )
                n_original = data.n_splats
                aprint(f"Loaded {n_original:,} splats ({data.ndim}D)")

            # Parse ranges
            slices = parse_slices(ranges, data.ndim)
            with asection("Ranges"):
                for i, s in enumerate(slices):
                    lo = s.start if s.start is not None else "-inf"
                    hi = s.stop if s.stop is not None else "inf"
                    aprint(f"  dim {i}: [{lo}, {hi}]")

            # Slice
            with asection("Slicing"):
                sliced_data = data.slice_by(slices)
                n_sliced = sliced_data.n_splats
                n_removed = n_original - n_sliced

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Sliced splats:   {n_sliced:,}")
                aprint(
                    f"  Removed:         {n_removed:,} ({100 * n_removed / max(n_original, 1):.1f}%)"
                )

            # Save
            from luxar.gsplats.io.load_gsplats import read_rebuild_root_attrs

            with asection(f"Saving to {output_path.name}"):
                if n_sliced == 0:
                    aprint("⚠ No splats remain after slicing — skipping save")
                else:
                    sliced_data.save(
                        output_path,
                        amplitude_bits="auto",
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                        root_attrs=read_rebuild_root_attrs(input_path),
                    )
                    aprint(f"Saved sliced dataset: {output_path}")

                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"❌ Error: {e}", e)
