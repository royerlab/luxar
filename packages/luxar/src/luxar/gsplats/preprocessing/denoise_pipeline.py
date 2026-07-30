"""High-level denoising pipeline: normalization, multi-channel calibration, denoising.

Composes the low-level ``denoise_nlm`` and ``calibrate_nlm_h`` functions with
consistent [0,1] normalization and per-channel calibration across multiple
timepoints.  Designed to be called from the CLI and batch pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from arbol import aprint, asection

# ── Normalization ────────────────────────────────────────────────


def normalize_volume(
    volume: np.ndarray,
    value_range: "tuple[float, float] | None" = None,
) -> tuple[np.ndarray, float, float]:
    """Normalize float32 volume to [0, 1] range.

    Parameters
    ----------
    volume : np.ndarray
        Float32 input volume.
    value_range : tuple[float, float], optional
        Fixed ``(vmin, vmax)`` to normalize against instead of the volume's
        own min/max. Used by tiled callers so a fixed NLM ``h`` means the same
        smoothing strength in every tile (each tile is normalized against the
        WHOLE-volume range, not its own extent).

    Returns
    -------
    normalized : np.ndarray
        Volume scaled to [0, 1].
    vmin : float
        Minimum actually used (for denormalization).
    vmax : float
        Maximum actually used (for denormalization).
    """
    if value_range is not None:
        vmin, vmax = float(value_range[0]), float(value_range[1])
    else:
        vmin = float(volume.min())
        vmax = float(volume.max())
    if vmax > vmin:
        return (volume - vmin) / (vmax - vmin), vmin, vmax
    # Constant volume — return zeros
    return np.zeros_like(volume), vmin, vmax


def denormalize_volume(volume: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Reverse [0, 1] normalization."""
    if vmax > vmin:
        return volume * (vmax - vmin) + vmin
    return np.full_like(volume, vmin)


# ── Single-volume denoising ─────────────────────────────────────


def _auto_chunk_size(
    shape: tuple[int, ...], search_distance: int, patch_size: int
) -> Optional[int]:
    """Compute chunk_size for 3D NLM to keep CPU memory under ~16 GB.

    Returns None if the volume is small enough to process in one shot.
    """
    if len(shape) != 3:
        return None
    d, h, w = shape
    slice_bytes = h * w * 4  # float32
    # Each chunk needs ~5x its size in memory (input, output, weights, etc.)
    mem_per_slice = slice_bytes * 5
    target_mem = 16 * 1024**3  # 16 GB target
    max_slices = max(1, int(target_mem / mem_per_slice))
    # Add halo to effective chunk size
    halo = search_distance + patch_size // 2
    if max_slices + 2 * halo >= d:
        return None  # Volume fits in memory, no chunking needed
    return max_slices


def denoise_volume_array(
    volume: np.ndarray,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: Optional[str] = None,
    use_2d: bool = False,
    chunk_size: Optional[int] = None,
    norm_range: "tuple[float, float] | None" = None,
) -> np.ndarray:
    """Denoise a single 3D volume (or 2D image) with NLM.

    Normalizes to [0,1], denoises, denormalizes back.

    Parameters
    ----------
    volume : np.ndarray
        Float32 input volume (2D or 3D).
    h : float
        NLM filtering strength (calibrated in [0,1] normalized space).
    use_2d : bool
        If True, denoise slice-by-slice (2D) instead of full 3D.
    chunk_size : int, optional
        Process 3D volumes in overlapping chunks of this many Z-slices.
        Auto-computed to keep memory under ~16 GB if not specified.
        Only used with the ``pytorch`` backend for 3D volumes.
    norm_range : tuple[float, float], optional
        Fixed whole-volume ``(vmin, vmax)`` to normalize against instead of
        this array's own min/max. Tiled callers pass the global range so a
        fixed ``h`` yields scale-consistent smoothing across all tiles.
    """
    import time as _time

    import torch

    from .nlm_core import denoise_nlm

    volume = volume.astype(np.float32)
    norm_vol, vmin, vmax = normalize_volume(volume, value_range=norm_range)

    # Auto-detect device: prefer CUDA, then MPS, then CPU.
    from luxar.gsplats.utils.device import resolve_torch_device

    dev = resolve_torch_device(device)

    # Resolve and log backend
    from .nlm_core import _resolve_backend

    effective_device = dev
    resolved_backend = _resolve_backend(backend, effective_device)
    mode_str = (
        "2D slice-by-slice" if (use_2d and norm_vol.ndim == 3) else f"{norm_vol.ndim}D"
    )
    aprint(
        f"NLM denoise: {mode_str}, shape={norm_vol.shape}, "
        f"h={h:.4f}, backend={resolved_backend}, device={effective_device}"
    )

    # Auto-compute chunk_size for large 3D volumes
    if chunk_size is None and not use_2d and norm_vol.ndim == 3:
        chunk_size = _auto_chunk_size(norm_vol.shape, search_distance, patch_size)
        if chunk_size is not None:
            aprint(f"Auto-chunking: {chunk_size} Z-slices per chunk")

    t0 = _time.monotonic()

    if use_2d and norm_vol.ndim == 3:
        # Slice-by-slice 2D NLM
        result_slices = []
        for z in range(norm_vol.shape[0]):
            t_slice = torch.from_numpy(norm_vol[z])
            denoised_slice = denoise_nlm(
                t_slice,
                h=h,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
                device=dev,
            )
            result_slices.append(denoised_slice.cpu().numpy())
        denoised_np = np.stack(result_slices, axis=0)
    else:
        t_vol = torch.from_numpy(norm_vol)
        denoised = denoise_nlm(
            t_vol,
            h=h,
            patch_size=patch_size,
            search_distance=search_distance,
            backend=backend,
            device=dev,
            chunk_size=chunk_size,
        )
        denoised_np = denoised.cpu().numpy()

    elapsed = _time.monotonic() - t0
    aprint(f"NLM denoise completed in {elapsed:.1f}s")

    return denormalize_volume(denoised_np, vmin, vmax)


# ── Calibration ─────────────────────────────────────────────────


def _pick_sample_timepoints(
    n_timepoints: int,
    n_samples: int,
    timepoint_indices: Optional[list[int]] = None,
) -> list[int]:
    """Pick equidistant timepoints for calibration.

    Returns actual timepoint indices into the dataset.
    """
    if timepoint_indices is not None:
        available = timepoint_indices
    else:
        available = list(range(n_timepoints))

    n = min(n_samples, len(available))
    if n <= 1:
        return [available[len(available) // 2]]

    step = max(1, (len(available) - 1) / (n - 1))
    return [available[int(round(i * step))] for i in range(n)]


def calibrate_h_for_channel(
    input_path: Path,
    channel: int,
    sample_timepoints: list[int],
    array_key: Optional[str] = None,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: Optional[str] = None,
) -> float:
    """Calibrate NLM h for one channel by sampling timepoints.

    Loads the central 2D slice at each sample timepoint, normalizes to [0,1],
    runs Noise2Self calibration, and returns the median h across timepoints.
    """
    import torch

    from .calibration import calibrate_nlm_h

    h_values = []
    for tp in sample_timepoints:
        try:
            from luxar.io.volume import load_volume

            vol = load_volume(
                input_path,
                channel=channel,
                timepoint=tp,
                array_key=array_key,
            )
            # Normalize to [0,1]
            norm_vol, _, _ = normalize_volume(vol)
            t_vol = torch.from_numpy(norm_vol)

            dev = torch.device(device) if device else None
            h = calibrate_nlm_h(
                t_vol,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
                device=dev,
                use_2d_slice=True,  # Fast: calibrate on central 2D slice
            )
            h_values.append(h)
            aprint(f"  T={tp}: h={h:.4f}")

            # Free memory between calibration iterations
            del vol, norm_vol, t_vol
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            aprint(f"  T={tp}: calibration failed ({e}), skipping")

    if not h_values:
        # Fallback: use middle of default h_range
        aprint("  Warning: all calibration samples failed, using default h=0.04")
        return 0.04

    median_h = float(np.median(h_values))
    return median_h


def calibrate_all_channels(
    input_path: Path,
    n_timepoints: int,
    n_channels: int,
    channel_indices: Optional[list[int]] = None,
    timepoint_indices: Optional[list[int]] = None,
    array_key: Optional[str] = None,
    calibration_samples: int = 5,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: Optional[str] = None,
    h_override: Optional[float] = None,
) -> dict[int, float]:
    """Calibrate NLM h for all channels.

    Returns
    -------
    dict[int, float]
        Mapping from channel index to calibrated h value.
    """
    channels = (
        channel_indices if channel_indices is not None else list(range(n_channels))
    )
    sample_tps = _pick_sample_timepoints(
        n_timepoints, calibration_samples, timepoint_indices
    )

    if h_override is not None:
        aprint(f"Using manual h={h_override:.4f} for all {len(channels)} channels")
        return {c: h_override for c in channels}

    aprint(
        f"Calibrating h for {len(channels)} channel(s), "
        f"sampling {len(sample_tps)} timepoint(s): {sample_tps}"
    )

    h_per_channel: dict[int, float] = {}
    for c in channels:
        with asection(f"Channel {c}"):
            h = calibrate_h_for_channel(
                input_path,
                channel=c,
                sample_timepoints=sample_tps,
                array_key=array_key,
                patch_size=patch_size,
                search_distance=search_distance,
                backend=backend,
                device=device,
            )
            h_per_channel[c] = h
            aprint(f"Median h={h:.4f}")

    return h_per_channel
