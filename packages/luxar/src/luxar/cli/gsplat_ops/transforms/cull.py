"""Implementation helper for gsplat cull command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ..encoding import _resolve_encoding_mode

if TYPE_CHECKING:
    import numpy as np


def _print_cull_compression(
    source_bytes: Optional[int], input_path: Path, output_path: Path
) -> None:
    """Report compression before -> after on the one shared basis.

    Decoded source bytes over each store's bytes ON DISK (the definition
    `gsplat info` quotes), measured on the written output so the encoding the
    save chose is what is counted. Silent when the source size is unknown.
    """
    from luxar.gsplats.fitting.visualization import (
        compression_ratio,
        format_compression_ratio,
    )

    from ...utils import store_size

    before = compression_ratio(source_bytes, store_size(input_path))
    after = compression_ratio(source_bytes, store_size(output_path))
    if before is None or after is None:
        return
    aprint(
        f"Compression: {format_compression_ratio(before)} -> "
        f"{format_compression_ratio(after)} vs raw voxels"
    )


def _target_on_fit_basis(
    target: "np.ndarray", stats: Optional[Mapping[str, Any]]
) -> "np.ndarray":
    from luxar.gsplats.fit_basis import (
        MISSING_BASIS_HINT,
        fit_image_min,
        reference_in_amplitude_units,
        reference_on_fit_basis,
        stored_intensity_scale,
    )

    level = fit_image_min(stats)
    if level is None:
        aprint(f"WARNING: {MISSING_BASIS_HINT}")
    elif level > 0.0:
        target = reference_on_fit_basis(target, level)
        aprint(f"Target shifted onto the fit's basis (image_min={level:.6g})")
    # The culled store keeps its edited amplitudes, so the target moves into
    # their units rather than the splats into the target's.
    scale = stored_intensity_scale(stats)
    if scale != 1.0:
        # In place: the caller hands over the volume it just loaded (or the
        # shifted copy above), which nothing else holds.
        target = reference_in_amplitude_units(target, scale, in_place=True)
        aprint(
            f"Target multiplied by intensity_scale={scale:.6g} (a transform "
            "--scale-intensity edit) to match the stored amplitudes"
        )
    return target


def run_cull_dataset(
    *,
    input_path: Path,
    output_path: Path,
    method: str,
    target_path: Optional[Path],
    volume_shape: Optional[str],
    error_percentile: float,
    error_tolerance: float,
    redundancy_threshold: float,
    retention: float,
    amplitude_percentile: float,
    volume_percentile: float,
    truncate: Optional[float],
    max_iters: int,
    device: Optional[str],
    channel: Optional[int],
    timepoint: Optional[int],
    encoding_mode: Literal["auto", "precision", "memory"],
    compress: Optional[Literal["zip", "tar.gz"]],
) -> None:
    """Run cull command implementation."""
    try:
        from luxar.cli.gsplat_ops.loading import load_matrix_gsplats

        encoding_mode_obj = _resolve_encoding_mode(encoding_mode)

        with asection(f"Culling: {input_path.name}"):
            with asection("Loading dataset"):
                data = load_matrix_gsplats(
                    input_path,
                    include_stats=True,
                    command="cull",
                )
                n_original = data.n_splats
                aprint(f"Loaded {n_original:,} splats ({data.ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            # Load target volume if provided
            target_np = None
            # Decoded source size, the numerator of the one compression ratio
            # (`compression_ratio`): the fit's own record first, else the
            # target volume in its stored element type.
            source_bytes = (data.stats or {}).get("source_bytes")
            if target_path is not None:
                from luxar.cli.gsplat_config import load_volume

                with asection("Loading target volume"):
                    target_info: dict = {}
                    target_np = load_volume(
                        target_path,
                        channel=channel,
                        timepoint=timepoint,
                        info=target_info,
                    )
                    aprint(f"Target shape: {target_np.shape}")
                    if not source_bytes:
                        import numpy as np

                        dtype = target_info.get("source_dtype") or target_np.dtype
                        source_bytes = int(target_np.size) * np.dtype(dtype).itemsize
                    if len(target_np.shape) != data.ndim:
                        aprint(
                            f"Dimension mismatch: gsplats are {data.ndim}D "
                            f"but target is {len(target_np.shape)}D"
                        )
                        raise typer.Exit(1)

                    # The error budget is measured against this target, and the
                    # splats reconstruct `V - image_min`. Left raw, the budget is
                    # spent on background the splats never claimed to represent,
                    # which biases retention toward whichever splats reproduce
                    # haze (#1177). Shift it onto the fit's basis first.
                    target_np = _target_on_fit_basis(target_np, data.stats)

            # Parse --shape if provided
            parsed_shape = None
            if volume_shape is not None:
                parsed_shape = tuple(int(x.strip()) for x in volume_shape.split(","))

            # Resolve method
            resolved = method
            if resolved == "auto":
                if target_np is not None:
                    resolved = "error_budget"
                elif parsed_shape is not None:
                    resolved = "redundancy"
                else:
                    resolved = "cumulative"

            with asection(f"Culling (method={resolved})"):
                culled_data = data.cull(
                    target=target_np,
                    method=resolved,
                    shape=parsed_shape,
                    truncate=truncate,
                    error_percentile=error_percentile,
                    error_tolerance=error_tolerance,
                    redundancy_threshold=redundancy_threshold,
                    max_binary_search_iters=max_iters,
                    device=device,
                    retention=retention,
                    amplitude_percentile=amplitude_percentile,
                    volume_percentile=volume_percentile,
                    verbose=True,
                )

                n_culled = n_original - culled_data.n_splats
                aprint("\nResults:")
                aprint(f"  Original: {n_original:,} splats")
                aprint(
                    f"  Removed:  {n_culled:,} ({100 * n_culled / max(n_original, 1):.1f}%)"
                )
                aprint(f"  Kept:     {culled_data.n_splats:,} splats")
                aprint(f"  Method:   {resolved}")

                if resolved in ("error_budget", "redundancy"):
                    aprint(
                        f"  Error budget: {culled_data.stats.get('error_budget', 'N/A')}"
                    )
                    aprint(
                        f"  Joint check iters: {culled_data.stats.get('phase2_iterations', 'N/A')}"
                    )
                if resolved in ("cumulative", "amplitude_percentile", "combined"):
                    amp_ret = culled_data.stats.get("amplitude_retention")
                    if amp_ret is not None:
                        aprint(f"  Amplitude retention: {100 * amp_ret:.2f}%")

            from luxar.gsplats.io.load_gsplats import read_rebuild_root_attrs

            with asection("Saving"):
                culled_data.save(
                    output_path,
                    amplitude_bits="auto",
                    encoding_mode=encoding_mode_obj,
                    compress=compress,
                    root_attrs=read_rebuild_root_attrs(input_path),
                )
                aprint(f"Saved to {output_path}")
                _print_cull_compression(source_bytes, input_path, output_path)

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)
