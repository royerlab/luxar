"""Shared ``--show-roundtrip`` visualisation for the multichannel gsplat demos.

Several ``demo_gsplats_*`` demos offer a ``--show-roundtrip`` flag that renders
the fitted splats back to a volume and shows original / reconstruction /
absolute-difference slices side by side, with per-channel PSNR and MSE. That
figure-building code was duplicated verbatim across five demos, so it lives
here once, parameterised on the two things the copies closed over: the channel
names and the render device.

Not a demo itself (no ``demo_`` prefix), so the demo import smoke test skips it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import numpy as np
from arbol import aprint, asection

from ._dependencies import MissingDependencyError, require_module

if TYPE_CHECKING:
    from luxar.gsplats.gsplat_data import GSplatData


def show_roundtrip_comparison(
    volumes: list[np.ndarray],
    gsplats_list: list[GSplatData],
    channel_names: Sequence[str],
    *,
    device: Optional[str] = None,
) -> None:
    """Show original vs round-trip reconstructed volumes side by side.

    Renders each channel's fitted splats back to its original volume shape,
    reports PSNR/MSE per channel, and draws an ``n_channels x 3`` matplotlib
    figure (original | reconstruction | absolute difference) through the mid
    z-slice. Returns quietly with a console notice if matplotlib is not
    installed, so the flag never breaks a demo run.

    Args:
        volumes: Per-channel original volumes, normalised to ``[0, 1]``. This
            is what sizes the figure; nothing is drawn if it is empty.
        gsplats_list: Per-channel fitted splats, parallel to ``volumes``.
        channel_names: Display names for the channels. The figure has one row
            per entry in ``volumes``, so extra names are simply unused (callers
            may pass a whole channel table); too few is an error.
        device: Render device passed to ``render_to_volume`` ("cuda", "mps",
            "cpu"). ``None`` auto-detects the fastest available backend.

    Raises:
        ValueError: If ``gsplats_list`` or ``channel_names`` has fewer entries
            than ``volumes``. Both are consumed by ``zip``, which would
            truncate silently and leave reserved-but-blank rows in the figure
            plus an understated splat total in the suptitle.
    """
    n_channels = len(volumes)
    short = [
        f"{label} has {length}"
        for label, length in (
            ("gsplats_list", len(gsplats_list)),
            ("channel_names", len(channel_names)),
        )
        if length < n_channels
    ]
    if short:
        raise ValueError(
            f"every channel needs its own splats and name: {n_channels} "
            f"volumes but {', '.join(short)}"
        )

    if not volumes:
        aprint("Skipping --show-roundtrip: no volumes to compare")
        return

    try:
        plt = require_module("matplotlib.pyplot")
    except MissingDependencyError as exc:
        aprint(f"Skipping --show-roundtrip: {exc}")
        return

    with asection("Round-trip reconstruction comparison"):
        reconstructions = []
        for i, (volume, gsplats, name) in enumerate(
            zip(volumes, gsplats_list, channel_names)
        ):
            with asection(f"Rendering Ch{i}: {name}"):
                recon = gsplats.render_to_volume(shape=volume.shape, device=device)
                reconstructions.append(recon)
                mse = float(np.mean((volume - recon) ** 2))
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
                aprint(f"  PSNR: {psnr:.2f} dB, MSE: {mse:.6g}")

        fig, axes = plt.subplots(
            n_channels, 3, figsize=(14, 4.5 * n_channels), squeeze=False
        )

        for i, (volume, recon, name) in enumerate(
            zip(volumes, reconstructions, channel_names)
        ):
            mid_z = volume.shape[0] // 2
            orig_slice = volume[mid_z]
            recon_slice = recon[mid_z]
            diff_slice = np.abs(orig_slice - recon_slice)

            mse = float(np.mean((volume - recon) ** 2))
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

            axes[i, 0].imshow(orig_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title(f"Original — {name}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(recon_slice, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed (PSNR {psnr:.1f} dB)")
            axes[i, 1].axis("off")

            im = axes[i, 2].imshow(diff_slice, cmap="inferno", vmin=0, vmax=0.3)
            axes[i, 2].set_title("|Difference|")
            axes[i, 2].axis("off")
            fig.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Round-Trip Comparison — z-slice {mid_z}  "
            f"({sum(len(g.amplitudes) for g in gsplats_list):,} total splats)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.show()
