"""Implementation helpers for denoise/render fitting commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ...utils import format_memory_size

if TYPE_CHECKING:
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.tree import GSplatLeaf


def _default_output_shape(
    default_leaves: list[GSplatLeaf], ndim: int
) -> tuple[int, ...]:
    """Return the auto-sized output shape for the leaves the renderer will draw."""
    import numpy as np

    from luxar.gsplats.tree import center_bounds

    leaf_bounds = [
        bounds for leaf in default_leaves if (bounds := center_bounds(leaf)) is not None
    ]
    if not leaf_bounds:
        raise ValueError("GSplat tree has no splat centers")
    mins = np.min(np.stack([bounds[0] for bounds in leaf_bounds]), axis=0)
    maxs = np.max(np.stack([bounds[1] for bounds in leaf_bounds]), axis=0)
    return tuple(int(maxs[index] - mins[index]) + 1 for index in range(ndim))


def _render_default_leaves(
    default_leaves: list[GSplatLeaf],
    first_data: GSplatData,
    *,
    output_shape: tuple[int, ...],
    device: Optional[str],
    truncate: float,
) -> np.ndarray:
    """Render one leaf directly, or accumulate a multi-leaf default selection."""
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData

    if len(default_leaves) == 1:
        return first_data.render_to_volume(
            shape=output_shape,
            device=device,
            truncate=truncate,
        )

    volume = np.zeros(output_shape, dtype=np.float32)
    for index, leaf in enumerate(default_leaves):
        leaf_data = first_data if index == 0 else GSplatData.from_tree(leaf)
        volume += leaf_data.render_to_volume(
            shape=output_shape,
            device=device,
            truncate=truncate,
        )
    return volume


def run_denoise_volume_cmd(
    *,
    input_path: Path,
    output_path: Path,
    h: Optional[float],
    patch_size: int,
    search_distance: int,
    backend: str,
    device: Optional[str],
    denoise_2d: bool,
    channel: Optional[int],
    timepoint: Optional[int],
    array_key: Optional[str],
) -> None:
    """Run denoise command implementation."""
    try:
        import numpy as np

        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.preprocessing.denoise_pipeline import (
            denoise_volume_array,
            normalize_volume,
        )

        with asection("Loading volume"):
            volume = load_volume(
                input_path, channel=channel, timepoint=timepoint, array_key=array_key
            )
            aprint(f"Shape: {volume.shape}, dtype: {volume.dtype}")

        # Calibrate h if not provided
        effective_h: float
        if h is not None:
            effective_h = h
            aprint(f"Using manual h={effective_h:.4f}")
        else:
            import torch

            from luxar.gsplats.preprocessing import calibrate_nlm_h
            from luxar.gsplats.utils.device import resolve_torch_device

            with asection("Auto-calibrating h (Noise2Self)"):
                norm_vol, _, _ = normalize_volume(volume)
                t_vol = torch.from_numpy(norm_vol)
                # Auto-select CUDA > MPS > CPU when --device is omitted.
                dev = resolve_torch_device(device) if device else resolve_torch_device()
                effective_h = calibrate_nlm_h(
                    t_vol,
                    patch_size=patch_size,
                    search_distance=search_distance,
                    backend=backend,
                    device=dev,
                    use_2d_slice=True,
                )
                aprint(f"Calibrated h={effective_h:.4f}")

        with asection("Denoising (NLM)"):
            denoised = denoise_volume_array(
                volume,
                h=effective_h,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
                device=device,
                use_2d=denoise_2d,
            )
            aprint(f"Denoised shape: {denoised.shape}")

        # Save
        with asection(f"Saving to {output_path.name}"):
            suffix = output_path.suffix.lower()
            if suffix == ".npy":
                np.save(output_path, denoised)
            elif suffix in (".zarr",):
                import zarr

                from luxar._zarr_compat import zarr_format

                # `zarr_format` is NOT optional here even though zarr's own
                # default now matches Luxar's: the write format is overridable
                # (LUXAR_ZARR_FORMAT / `set_zarr_format`), so an unpinned `save`
                # would ignore the override and emit a format-3 store in a run
                # the user asked to be format 2. Called through the accessor
                # rather than importing `ZARR_FORMAT` by value, which is what the
                # facade asks callers to do: a `from ... import ZARR_FORMAT`
                # binds whatever was current when the import ran, so hoisting it
                # to module scope would silently stop honouring the override.
                zarr.save(str(output_path), denoised, zarr_format=zarr_format())
            else:
                np.save(output_path, denoised)
            aprint("Done")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)


def run_render_to_file(
    *,
    input_path: Path,
    output_path: Path,
    shape: Optional[str],
    device: Optional[str],
    truncate: Optional[float],
) -> None:
    """Run render command implementation."""
    try:
        import numpy as np

        from luxar.cli.gsplat_config import parse_shape
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import iter_default_leaves, total_splats

        with asection(f"Rendering: {input_path.name}"):
            with asection("Loading gsplat dataset"):
                node, _ = load_gsplat_node(input_path, include_stats=False)
                default_leaves = list(iter_default_leaves(node))
                if not default_leaves:
                    raise ValueError("GSplat tree has no default-rendered leaves")
                first_leaf = default_leaves[0]
                first_data = GSplatData.from_tree(first_leaf)
                ndim = first_data.ndim
                n_splats = sum(total_splats(leaf) for leaf in default_leaves)
                aprint(f"Loaded {n_splats:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = first_data.truncation_radius

            if shape is not None:
                output_shape = parse_shape(shape)
            else:
                output_shape = _default_output_shape(default_leaves, ndim)
                aprint(f"Auto shape from bounding box: {output_shape}")

            with asection(f"Rendering to {output_shape}"):
                volume = _render_default_leaves(
                    default_leaves,
                    first_data,
                    output_shape=output_shape,
                    device=device,
                    truncate=truncate,
                )
                aprint(
                    f"Rendered: {volume.shape}, "
                    f"range [{volume.min():.4f}, {volume.max():.4f}]"
                )

            suffix = output_path.suffix.lower()
            with asection(f"Saving to {output_path.name}"):
                if suffix == ".npy":
                    np.save(str(output_path), volume)
                elif suffix in (".tiff", ".tif"):
                    try:
                        import tifffile
                    except ImportError:
                        aprint("tifffile not installed.")
                        aprint("Install with: pip install luxar[io]")
                        raise typer.Exit(1)
                    tifffile.imwrite(str(output_path), volume)
                else:
                    aprint(f"Unknown extension '{suffix}', saving as NumPy .npy")
                    np.save(str(output_path), volume)

                if output_path.exists():
                    aprint(
                        f"File size: {format_memory_size(output_path.stat().st_size)}"
                    )

        aprint(f"\nSaved: {output_path}")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)
