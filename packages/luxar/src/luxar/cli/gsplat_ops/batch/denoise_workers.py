"""Implementation helpers for hidden batch denoise worker commands."""

from __future__ import annotations

from pathlib import Path

import typer
from arbol import aprint, asection

from luxar._zarr_compat import create_array, open_group
from luxar.encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor

from ..._traceback import exit_with_error


def resolve_deferred_batch_floor(output_dir: Path) -> None:
    """Resolve a denoise-dependent global floor and persist it for fit workers."""
    import json

    from luxar.cli.gsplat_ops.batch.planning import resolve_batch_floor
    from luxar.gsplats.batch.manifest import load_manifest, save_manifest
    from luxar.gsplats.batch.slurm_gen import _preprocessed_axes

    manifest = load_manifest(output_dir)
    if not manifest.floor_deferred:
        return

    if manifest.denoise_mode == "preprocess":
        source = Path(manifest.denoised_zarr_path or output_dir / "denoised.zarr")
        canonical_axes = _preprocessed_axes(manifest)
        axes_labels = canonical_axes.split(",") if canonical_axes is not None else None
        level, forward = resolve_batch_floor(
            source,
            manifest.floor_spec,
            n_timepoints=manifest.n_timepoints,
            n_channels=manifest.n_channels,
            array_key="data",
            axes=canonical_axes,
            axes_labels=axes_labels,
            channel_shape=(manifest.n_channels,),
            spatial_shape=tuple(manifest.spatial_shape),
        )
    else:
        from luxar.cli.gsplat_config import discover_ome_zarr_shape

        if manifest.denoise_h is not None:
            channels = manifest.channel_indices or list(range(manifest.n_channels))
            numeric_h = {channel: manifest.denoise_h for channel in channels}
        else:
            h_values = manifest.denoise_h_values
            if h_values is None:
                h_path = output_dir / "denoise_h_values.json"
                if not h_path.exists():
                    raise RuntimeError("denoise_h_values.json not found")
                h_values = json.loads(h_path.read_text())
            numeric_h = {int(key): float(value) for key, value in h_values.items()}
        axes_override = manifest.axes.split(",") if manifest.axes else None
        source_info = discover_ome_zarr_shape(
            Path(manifest.input_path),
            axes_override=axes_override,
            array_key=manifest.array_key,
        )
        level, forward = resolve_batch_floor(
            Path(manifest.input_path),
            manifest.floor_spec,
            n_timepoints=source_info.n_timepoints,
            n_channels=source_info.n_channels,
            array_key=manifest.array_key,
            axes=manifest.axes,
            axes_labels=list(source_info.axes),
            channel_shape=tuple(manifest.channel_shape),
            spatial_shape=tuple(manifest.spatial_shape),
            denoise_h_values=numeric_h,
            denoise_params={
                "patch_size": manifest.denoise_patch_size,
                "search_distance": manifest.denoise_search_distance,
                "backend": manifest.denoise_backend,
                "device": manifest.fit_args.get("device"),
                "use_2d": manifest.denoise_2d,
            },
        )

    (output_dir / "floor_level.json").write_text(
        json.dumps(
            {
                "level": level if isinstance(forward, (int, float)) else None,
                "forward": forward,
            },
            indent=2,
        )
    )
    manifest.floor_level = level if isinstance(forward, (int, float)) else None
    manifest.fit_args["floor"] = forward
    manifest.floor_deferred = False
    save_manifest(manifest, output_dir)


def run_batch_resolve_floor_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """[Internal] Resolve the batch floor after denoise prerequisites finish."""
    try:
        with asection("Resolving denoised batch floor"):
            resolve_deferred_batch_floor(output_dir)
    except typer.Exit:
        raise
    except Exception as exc:
        exit_with_error(f"Error: {exc}", exc)


def run_batch_denoise_calibrate_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
) -> None:
    """[Internal] Run NLM calibration for batch denoise pipeline.

    Reads manifest, calibrates h per channel, writes results back.
    Called by the calibration Slurm job.
    """
    try:
        import json

        from luxar.gsplats.batch.manifest import load_manifest, save_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import calibrate_all_channels

        manifest = load_manifest(output_dir)

        if not manifest.denoise:
            aprint("Error: denoise not enabled in manifest")
            raise typer.Exit(1)

        with asection("NLM Calibration"):
            h_values = calibrate_all_channels(
                input_path=Path(manifest.input_path),
                n_timepoints=manifest.n_timepoints,
                n_channels=manifest.n_channels,
                channel_indices=(
                    manifest.channel_indices
                    if manifest.channel_indices
                    else list(range(manifest.n_channels))
                ),
                timepoint_indices=manifest.timepoint_indices,
                array_key=manifest.array_key,
                axes=manifest.axes,
                calibration_samples=manifest.calibration_samples,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                h_override=manifest.denoise_h,
            )

            # Write h_values to manifest (string keys for JSON)
            manifest.denoise_h_values = {str(k): v for k, v in h_values.items()}
            save_manifest(manifest, output_dir)

            # Also write standalone JSON for easy reading by other jobs
            h_path = output_dir / "denoise_h_values.json"
            h_path.write_text(json.dumps(h_values, indent=2))

            aprint(f"Calibrated h values: {h_values}")
            aprint(f"Saved to {h_path}")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)


def run_batch_denoise_preprocess_cmd(
    output_dir: Path = typer.Argument(..., exists=True, help="Batch output directory"),
    task_id: int = typer.Argument(..., help="Array task ID (encodes T*n_c + C)"),
) -> None:
    """[Internal] Denoise one (T,C) volume for batch preprocess pipeline.

    Called by the denoise Slurm array job, one task per (timepoint, channel).
    """
    try:
        import json

        from luxar.cli.gsplat_config import load_volume
        from luxar.gsplats.batch.manifest import load_manifest
        from luxar.gsplats.preprocessing.denoise_pipeline import denoise_volume_array

        manifest = load_manifest(output_dir)

        if manifest.denoise_h is None:
            h_path = output_dir / "denoise_h_values.json"
            if not h_path.exists():
                aprint("Error: denoise_h_values.json not found. Run calibration first.")
                raise typer.Exit(1)
            h_values = json.loads(h_path.read_text())
        else:
            h_values = {}

        # Decode task_id -> (t_idx, c_idx) within selected indices
        n_c = manifest.n_channels
        t_idx = task_id // n_c
        c_idx = task_id % n_c

        # Map to real dataset indices
        t_indices = manifest.timepoint_indices or list(range(manifest.n_timepoints))
        c_indices = manifest.channel_indices or list(range(manifest.n_channels))
        t_real = t_indices[t_idx]
        c_real = c_indices[c_idx]

        h = (
            manifest.denoise_h
            if manifest.denoise_h is not None
            else h_values.get(str(c_real), 0.04)
        )

        with asection(f"Denoising T={t_real} C={c_real} (h={h:.4f})"):
            # Load volume
            volume = load_volume(
                Path(manifest.input_path),
                channel=(
                    c_real
                    if manifest.n_channels > 1 or manifest.channel_indices is not None
                    else None
                ),
                timepoint=(
                    t_real
                    if manifest.n_timepoints > 1
                    or manifest.timepoint_indices is not None
                    else None
                ),
                array_key=manifest.array_key,
                axes=manifest.axes,
            )
            aprint(f"Loaded: shape={volume.shape}")

            # Denoise
            denoised = denoise_volume_array(
                volume,
                h=h,
                patch_size=manifest.denoise_patch_size,
                search_distance=manifest.denoise_search_distance,
                backend=manifest.denoise_backend,
                use_2d=manifest.denoise_2d,
            )

            # Write to denoised.zarr
            zarr_path = output_dir / "denoised.zarr"
            # Through the facade: this is the store's CREATING open (the first
            # task to arrive makes it), and a bare `zarr.open` would create it at
            # zarr 3's default format 3 — which the `data` array below then
            # inherits from its parent, whatever `create_array` is told.
            store = open_group(zarr_path, mode="a")

            spatial = denoised.shape
            full_shape = (len(t_indices), len(c_indices), *spatial)
            chunks = (1, 1, *[min(s, 128) for s in spatial])

            if "data" not in store:
                create_array(
                    store,
                    "data",
                    shape=full_shape,
                    chunks=chunks,
                    # string dtype: numpy is only imported under TYPE_CHECKING
                    # in this module (dtype=np.float32 here was a latent
                    # NameError before this change).
                    dtype="float32",
                    compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, "float32"),
                )
            store["data"][t_idx, c_idx] = denoised
            aprint(f"Written to denoised.zarr[{t_idx}, {c_idx}]")

    except typer.Exit:
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)
