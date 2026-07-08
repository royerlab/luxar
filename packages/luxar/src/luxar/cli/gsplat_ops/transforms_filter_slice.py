"""Implementation helpers for filter/slice gsplat edit commands."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from ..utils import format_memory_size
from .encoding import _resolve_encoding_mode
from .transforms_parsing import parse_bbox, parse_slices


def run_filter_dataset(
    *,
    input_path: Path,
    output_path: Path,
    bbox: Optional[str],
    volume_min: Optional[float],
    volume_max: Optional[float],
    volume_normalized: bool,
    amplitude_min: Optional[float],
    amplitude_max: Optional[float],
    amplitude_normalized: bool,
    eccentricity_min: Optional[float],
    eccentricity_max: Optional[float],
    mass_min: Optional[float],
    mass_max: Optional[float],
    mass_normalized: bool,
    sigma_axis: Optional[int],
    sigma_min: Optional[float],
    sigma_max: Optional[float],
    truncate: Optional[float],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run filter command implementation."""
    try:
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Filtering: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
                n_original = data.n_splats
                ndim = data.ndim
                aprint(f"Loaded {n_original:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Parse bbox
            bbox_parsed = None
            if bbox is not None:
                bbox_parsed = parse_bbox(bbox, ndim)

            # Show active criteria
            with asection("Active criteria"):
                if bbox_parsed is not None:
                    aprint(f"  bbox: {bbox_parsed}")
                if amplitude_min is not None or amplitude_max is not None:
                    norm = " (normalized)" if amplitude_normalized else ""
                    aprint(f"  amplitude: [{amplitude_min}, {amplitude_max}]{norm}")
                if volume_min is not None or volume_max is not None:
                    norm = " (normalized)" if volume_normalized else ""
                    aprint(
                        f"  volume: [{volume_min}, {volume_max}]{norm} (truncate={truncate})"
                    )
                if eccentricity_min is not None or eccentricity_max is not None:
                    aprint(f"  eccentricity: [{eccentricity_min}, {eccentricity_max}]")
                if mass_min is not None or mass_max is not None:
                    norm = " (normalized)" if mass_normalized else ""
                    aprint(f"  mass: [{mass_min}, {mass_max}]{norm}")
                if sigma_axis is not None:
                    aprint(f"  sigma axis {sigma_axis}: [{sigma_min}, {sigma_max}]")

            # Filter
            with asection("Filtering"):
                filtered_data = data.filter_by(
                    bbox=bbox_parsed,
                    volume_min=volume_min,
                    volume_max=volume_max,
                    volume_normalized=volume_normalized,
                    amplitude_min=amplitude_min,
                    amplitude_max=amplitude_max,
                    amplitude_normalized=amplitude_normalized,
                    eccentricity_min=eccentricity_min,
                    eccentricity_max=eccentricity_max,
                    mass_min=mass_min,
                    mass_max=mass_max,
                    mass_normalized=mass_normalized,
                    sigma_axis=sigma_axis,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    truncate=truncate,
                )
                n_filtered = filtered_data.n_splats
                n_removed = n_original - n_filtered

                aprint("\nResults:")
                aprint(f"  Original splats: {n_original:,}")
                aprint(f"  Filtered splats: {n_filtered:,}")
                aprint(
                    f"  Removed:         {n_removed:,} ({100 * n_removed / max(n_original, 1):.1f}%)"
                )

            # Save
            with asection(f"Saving to {output_path.name}"):
                if n_filtered == 0:
                    aprint("⚠ No splats remain after filtering — skipping save")
                else:
                    filtered_data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                    aprint(f"Saved filtered dataset: {output_path}")

                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


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
        from luxar.gsplats.gsplat_data import GSplatData

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Slicing: {input_path.name}"):
            # Load
            with asection("Loading dataset"):
                data = GSplatData.load(input_path, include_stats=True)
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
            with asection(f"Saving to {output_path.name}"):
                if n_sliced == 0:
                    aprint("⚠ No splats remain after slicing — skipping save")
                else:
                    sliced_data.save(
                        output_path,
                        encoding_mode=encoding_mode_obj,
                        include_fitting_info=True,
                        compress=compress,
                    )
                    aprint(f"Saved sliced dataset: {output_path}")

                    if output_path.exists():
                        aprint(
                            f"  Size: {format_memory_size(output_path.stat().st_size)}"
                        )

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
