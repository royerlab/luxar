"""Implementation helpers for denoise/render fitting commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import typer
from arbol import aprint, asection

from ..._traceback import exit_with_error
from ...utils import format_memory_size, store_size

if TYPE_CHECKING:
    import numpy as np

    from luxar.gsplats.gsplat_data import GSplatData


def _bounding_box_grid(
    datas: list[GSplatData], ndim: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(origin, shape)`` of the voxel grid spanning every splat center.

    The origin is the floor of the lowest center, NOT index 0: a store whose
    splats start at, say, z=40 used to be rendered on a grid sized by its extent
    but anchored at 0, which clipped the upper part of the data and left the
    lower part of the grid empty.
    """
    import numpy as np

    populated = [d.centers for d in datas if d.n_splats > 0]
    if not populated:
        raise ValueError("GSplat tree has no splat centers")
    mins = np.min(np.stack([c.min(axis=0) for c in populated]), axis=0)
    maxs = np.max(np.stack([c.max(axis=0) for c in populated]), axis=0)
    origin = tuple(int(np.floor(mins[i])) for i in range(ndim))
    shape = tuple(int(np.floor(maxs[i])) - origin[i] + 1 for i in range(ndim))
    return origin, shape


def _source_grid_shape(
    stats: dict, ndim: int, dimension_metadata: Any
) -> Optional[tuple[int, ...]]:
    """The fitted volume's shape, when it describes the centers' grid.

    Not on a STACKED store (a batch-fit timelapse, ``merge --as-dimension``):
    there ``source_shape`` is in source order (time first) or describes one
    component, while the centers are spatial-first with the stacked axis last.
    """
    from luxar.gsplats.scoring_grid import is_stacked_store

    shape = (stats or {}).get("source_shape")
    if not isinstance(shape, (list, tuple)) or len(shape) != ndim:
        return None
    if is_stacked_store(stats, ndim, dimension_metadata):
        return None
    return tuple(int(n) for n in shape)


def _on_source_grid(
    datas: list[GSplatData], stats: dict, ndim: int, dimension_metadata: Any
) -> Optional[tuple[list[GSplatData], tuple[int, ...]]]:
    """``(leaves, shape)`` on the fit's source grid, or ``None`` if it does not fit.

    A ``--region`` crop fit keeps full-volume centers with a crop-shaped source
    grid, so the crop start is subtracted. The grid is trusted only when most
    centers land inside it.
    """
    import numpy as np

    from luxar.gsplats.scoring_grid import (
        fraction_inside_grid,
        map_to_grid,
        source_region_start,
    )

    source = _source_grid_shape(stats, ndim, dimension_metadata)
    if source is None:
        return None
    start = source_region_start(stats, ndim)
    if start is not None:
        datas = [map_to_grid(d, origin=start) for d in datas]
    populated = [d.centers for d in datas if d.n_splats > 0]
    inside = (
        fraction_inside_grid(np.concatenate(populated), source) if populated else 0.0
    )
    if inside < 0.5:
        aprint(
            f"The fit's source grid {source} holds only {inside:.0%} of the splat "
            "centers; using the bounding box instead"
        )
        return None
    aprint(f"Auto shape from the fit's source grid: {source}")
    if start is not None:
        aprint(f"Crop fit: rendering the crop (start {list(start)})")
    return datas, source


def _resolve_output_grid(
    datas: list[GSplatData],
    stats: dict,
    ndim: int,
    shape: Optional[str],
    dimension_metadata: Any = None,
) -> tuple[list[GSplatData], tuple[int, ...]]:
    """The render grid, and the leaves translated onto it.

    ``--shape`` renders at index 0. Otherwise a fit renders on its own source
    grid (``fitting/source_shape``), so the output lines up voxel-for-voxel with
    the volume that was fitted; any other store (stacked stores included) on the
    bounding box of its centers, with the origin reported.
    """
    from luxar.cli.gsplat_config import parse_shape
    from luxar.gsplats.scoring_grid import map_to_grid

    if shape is not None:
        return datas, tuple(parse_shape(shape))
    on_source = _on_source_grid(datas, stats, ndim, dimension_metadata)
    if on_source is not None:
        return on_source
    origin, box = _bounding_box_grid(datas, ndim)
    aprint(f"Auto shape from bounding box: {box} (origin {origin})")
    if any(origin):
        datas = [map_to_grid(d, origin=origin) for d in datas]
    return datas, box


def _warn_if_off_grid(datas: list[GSplatData], shape: tuple[int, ...]) -> None:
    import numpy as np

    from ..compare_ops import warn_if_outside_grid

    populated = [d.centers for d in datas if d.n_splats > 0]
    if populated:
        warn_if_outside_grid(np.concatenate(populated), shape)


def _render_default_leaves(
    datas: list[GSplatData],
    *,
    output_shape: tuple[int, ...],
    device: Optional[str],
    truncate: float,
) -> np.ndarray:
    """Render one leaf directly, or accumulate a multi-leaf default selection."""
    import numpy as np

    if len(datas) == 1:
        return datas[0].render_to_volume(
            shape=output_shape,
            device=device,
            truncate=truncate,
        )

    volume = np.zeros(output_shape, dtype=np.float32)
    for leaf_data in datas:
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


def _load_render_leaves(
    input_path: Path, voxel_size: Optional[str]
) -> tuple[list[GSplatData], dict, int, Any]:
    """The default-rendered leaves on the voxel grid, root stats, ndim, axis names."""
    from luxar.gsplats.gsplat_data import GSplatData
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.scoring_grid import map_to_grid
    from luxar.gsplats.tree import iter_default_leaves

    from ..compare_ops import resolve_voxel_size

    node, stats = load_gsplat_node(input_path, include_stats=True)
    leaves = list(iter_default_leaves(node))
    if not leaves:
        raise ValueError("GSplat tree has no default-rendered leaves")
    datas = [GSplatData.from_tree(leaf) for leaf in leaves]
    ndim = datas[0].ndim
    scale, source = resolve_voxel_size(stats, ndim, voxel_size)
    if scale is not None:
        aprint(
            f"Physical splats mapped to voxels by voxel_size={list(scale)} "
            f"(from {source})"
        )
        datas = [map_to_grid(d, scale=scale) for d in datas]
    return datas, stats, ndim, node.meta.get("dimension_metadata")


def _save_volume(volume: np.ndarray, output_path: Path) -> None:
    import numpy as np

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
            aprint(f"File size: {format_memory_size(store_size(output_path))}")


def run_render_to_file(
    *,
    input_path: Path,
    output_path: Path,
    shape: Optional[str],
    device: Optional[str],
    truncate: Optional[float],
    voxel_size: Optional[str] = None,
) -> None:
    """Run render command implementation."""
    try:
        with asection(f"Rendering: {input_path.name}"):
            with asection("Loading gsplat dataset"):
                datas, stats, ndim, dim_meta = _load_render_leaves(
                    input_path, voxel_size
                )
                n_splats = sum(d.n_splats for d in datas)
                aprint(f"Loaded {n_splats:,} splats ({ndim}D)")

            # Resolve truncation radius from dataset if not explicitly set
            if truncate is None:
                truncate = datas[0].truncation_radius

            datas, output_shape = _resolve_output_grid(
                datas, stats, ndim, shape, dim_meta
            )
            _warn_if_off_grid(datas, output_shape)

            with asection(f"Rendering to {output_shape}"):
                volume = _render_default_leaves(
                    datas,
                    output_shape=output_shape,
                    device=device,
                    truncate=truncate,
                )
                aprint(
                    f"Rendered: {volume.shape}, "
                    f"range [{volume.min():.4f}, {volume.max():.4f}]"
                )

            _save_volume(volume, output_path)

        aprint(f"\nSaved: {output_path}")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)
