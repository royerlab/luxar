#!/usr/bin/env python3
"""Dataset registry for splat count vs quality analysis.

Each dataset is a callable returning ``(volume, metadata)`` where *volume*
is a float32 ndarray normalised to [0, 1] and *metadata* carries provenance
info used in figure titles and TSV headers.

Adding a new dataset is a single ``@register("name")`` decorated function.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np
from arbol import aprint, asection

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, Callable[[], tuple[np.ndarray, dict[str, Any]]]] = {}


def register(name: str):
    """Decorator that registers a dataset loader under *name*."""

    def decorator(fn: Callable[[], tuple[np.ndarray, dict[str, Any]]]):
        DATASETS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_opencell_map4"


def _ensure_opencell_tiff() -> Path:
    """Download the OpenCell MAP4 TIFF if not already cached.

    Returns the local path to the TIFF file.
    """
    TIFF_URL = (
        "https://czb-opencell.s3.amazonaws.com/microscopy/raw/"
        "MAP4_ENSG00000047849/"
        "OC-FOV_MAP4_ENSG00000047849_CID000828_FID00002848_stack.tif"
    )
    cached = CACHE_DIR / "opencell_map4_stack.tif"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cached.exists():
        aprint(f"Using cached TIFF: {cached}")
        return cached

    with asection("Downloading OpenCell MAP4 TIFF (~70 MB)"):
        aprint(f"URL: {TIFF_URL}")
        fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tif.tmp")
        os.close(fd)
        try:
            urllib.request.urlretrieve(TIFF_URL, tmp_path)
            os.replace(tmp_path, cached)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        aprint(f"Saved to {cached}")

    return cached


def _load_opencell_channel(channel: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one channel of the OpenCell MAP4 z-stack.

    Parameters
    ----------
    channel : int
        Channel index (0 = MAP4-GFP, 1 = Hoechst nuclei).

    Returns
    -------
    volume : np.ndarray
        Float32 volume normalised to [0, 1], shape (Z, Y, X).
    metadata : dict
        Provenance information.
    """
    import tifffile

    tiff_path = _ensure_opencell_tiff()

    with asection(f"Loading OpenCell channel {channel}"):
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        # Expected layout: (Z=51, C=2, Y=600, X=600)
        if data.ndim == 4 and data.shape[1] == 2:
            V = np.array(data[:, channel], dtype=np.float32)
        elif data.ndim == 4 and data.shape[0] == 2:
            V = np.array(data[channel], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")

        del data

        # Robust normalisation — percentile clipping (matches demo)
        p_low, p_high = np.percentile(V, [1, 99.5])
        V = np.clip(V, p_low, p_high)
        V = (V - p_low) / (p_high - p_low + 1e-8)

        aprint(f"Volume shape: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": f"OpenCell MAP4 ch{channel}",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "OpenCell",
        "citation": "Cho et al. (2022) Science 375(6585)",
        "channel": channel,
    }


# ---------------------------------------------------------------------------
# Registered datasets
# ---------------------------------------------------------------------------


@register("opencell_map4_ch0")
def load_opencell_map4_ch0() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell MAP4-GFP channel (microtubule network)."""
    return _load_opencell_channel(0)


@register("opencell_map4_ch1")
def load_opencell_map4_ch1() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell Hoechst channel (nuclei)."""
    return _load_opencell_channel(1)
