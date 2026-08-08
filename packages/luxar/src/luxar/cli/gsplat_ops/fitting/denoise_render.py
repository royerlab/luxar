"""Implementation helpers for denoise/render fitting commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from arbol import aprint, asection

from ...utils import format_memory_size


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

                zarr.save(str(output_path), denoised)
            else:
                np.save(output_path, denoised)
            aprint("Done")

    except typer.Exit:
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        raise typer.Exit(1) from e


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

        with asection(f"Rendering: {input_path.name}"):
            with asection("Loading gsplat dataset"):
                data = GSplatData.load(input_path, include_stats=False)
                ndim = data.ndim
                aprint(f"Loaded {data.n_splats:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = data.truncation_radius

            if shape is not None:
                output_shape = parse_shape(shape)
            else:
                mins = data.centers.min(axis=0)
                maxs = data.centers.max(axis=0)
                output_shape = tuple(int(maxs[i] - mins[i]) + 1 for i in range(ndim))
                aprint(f"Auto shape from bounding box: {output_shape}")

            with asection(f"Rendering to {output_shape}"):
                volume = data.render_to_volume(
                    shape=output_shape,
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
