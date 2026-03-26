"""High-level denoising pipeline: normalization, multi-channel calibration, denoising.

Composes the low-level ``denoise_nlm`` and ``calibrate_nlm_h`` functions with
consistent [0,1] normalization and per-channel calibration across multiple
timepoints.  Designed to be called from the CLI and batch pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from arbol import aprint, asection

# ── Normalization ────────────────────────────────────────────────


def normalize_volume(volume: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Normalize float32 volume to [0, 1] range.

    Returns
    -------
    normalized : np.ndarray
        Volume scaled to [0, 1].
    vmin : float
        Original minimum (for denormalization).
    vmax : float
        Original maximum (for denormalization).
    """
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


def denoise_volume_array(
    volume: np.ndarray,
    h: float,
    patch_size: int = 3,
    search_distance: int = 5,
    backend: str = "auto",
    device: Optional[str] = None,
    use_2d: bool = False,
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
    """
    from .nlm_core import denoise_nlm

    volume = volume.astype(np.float32)
    norm_vol, vmin, vmax = normalize_volume(volume)

    dev = torch.device(device) if device else None

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
        )
        denoised_np = denoised.cpu().numpy()

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
    from .calibration import calibrate_nlm_h

    h_values = []
    for tp in sample_timepoints:
        try:
            from luxar.cli.gsplat_config import load_volume

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
