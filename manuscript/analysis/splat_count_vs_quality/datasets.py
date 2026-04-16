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

# Human-readable display names for figures and tables
DATASET_LABELS: dict[str, str] = {
    "opencell_map4_ch0": "MAP4 (Hoechst) — Nuclei",
    "opencell_map4_ch1": "MAP4 (GFP) — Microtubules",
    "kidney_dapi": "Mouse Kidney — DAPI (Nuclei)",
    "kidney_actin": "Mouse Kidney — Phalloidin (Actin)",
    "organoid_ch0": "Organoid — Channel 0",
    "celegans_t100": "C. elegans Embryo — t=100",
    "tribolium": "Tribolium Embryo (Light-Sheet)",
    "opencell_lmnb1_ch0": "LMNB1 (Hoechst) — Nuclei",
    "opencell_lmnb1_ch1": "LMNB1 (GFP) — Nuclear Lamina",
    "cells3d_nuclei": "HeLa Cells — Nuclei",
    "cells3d_membrane": "HeLa Cells — Membrane",
    "acto3d_heart_nuclei": "Mouse Heart — Nuclei (Light-Sheet)",
}


def register(name: str):
    """Decorator that registers a dataset loader under *name*."""

    def decorator(fn: Callable[[], tuple[np.ndarray, dict[str, Any]]]):
        DATASETS[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ANALYSIS_CACHE = Path.home() / ".cache" / "luxar"


def _download_cached(url: str, cache_dir: Path, filename: str, label: str = "") -> Path:
    """Download a file if not already cached. Returns local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / filename
    if cached.exists():
        aprint(f"Using cached: {cached}")
        return cached

    with asection(f"Downloading {label or filename}"):
        aprint(f"URL: {url}")
        fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
        os.close(fd)
        try:
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, cached)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        aprint(f"Saved to {cached}")
    return cached


def _normalize_volume(V: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Robust percentile normalisation to [0, 1]."""
    lo, hi = np.percentile(V, [p_low, p_high])
    V = np.clip(V, lo, hi)
    return ((V - lo) / (hi - lo + 1e-8)).astype(np.float32)


def _load_opencell_channel(channel: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one channel of the OpenCell MAP4 z-stack (51x600x600)."""
    import tifffile

    cache_dir = ANALYSIS_CACHE / "gsplats_opencell_map4"
    tiff_path = _download_cached(
        "https://czb-opencell.s3.amazonaws.com/microscopy/raw/"
        "MAP4_ENSG00000047849/"
        "OC-FOV_MAP4_ENSG00000047849_CID000828_FID00002848_stack.tif",
        cache_dir, "opencell_map4_stack.tif", "OpenCell MAP4 TIFF (~70 MB)",
    )

    with asection(f"Loading OpenCell channel {channel}"):
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        if data.ndim == 4 and data.shape[1] == 2:
            V = np.array(data[:, channel], dtype=np.float32)
        elif data.ndim == 4 and data.shape[0] == 2:
            V = np.array(data[channel], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")
        del data

        V = _normalize_volume(V, 1.0, 99.5)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    ch_names = {0: "Hoechst (Nuclei)", 1: "MAP4-GFP (Microtubules)"}
    return V, {
        "name": f"OpenCell MAP4 — {ch_names.get(channel, f'ch{channel}')}",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "OpenCell",
        "citation": "Cho et al. (2022) Science 375(6585)",
    }


# ---------------------------------------------------------------------------
# Registered datasets
# ---------------------------------------------------------------------------


# ---- OpenCell MAP4 (spinning-disk confocal, HEK293T) ----

@register("opencell_map4_ch0")
def load_opencell_map4_ch0() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell Hoechst channel (nuclei). 51x600x600."""
    return _load_opencell_channel(0)


@register("opencell_map4_ch1")
def load_opencell_map4_ch1() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell MAP4-GFP channel (microtubule network). 51x600x600."""
    return _load_opencell_channel(1)


# ---- Mouse kidney (confocal, scikit-image) ----

@register("kidney_dapi")
def load_kidney_dapi() -> tuple[np.ndarray, dict[str, Any]]:
    """Mouse kidney DAPI nuclei (confocal). 16x512x512."""
    with asection("Loading kidney DAPI from scikit-image"):
        from skimage.data import kidney

        raw = kidney()  # (16, 512, 512, 3) uint16
        aprint(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
        V = raw[:, :, :, 0].astype(np.float32)
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "Mouse Kidney — DAPI (Nuclei)",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "scikit-image",
        "citation": "van der Walt et al. (2014) PeerJ 2:e453",
    }


@register("kidney_actin")
def load_kidney_actin() -> tuple[np.ndarray, dict[str, Any]]:
    """Mouse kidney Phalloidin actin (confocal). 16x512x512."""
    with asection("Loading kidney Phalloidin from scikit-image"):
        from skimage.data import kidney

        raw = kidney()
        V = raw[:, :, :, 2].astype(np.float32)
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "Mouse Kidney — Phalloidin (Actin)",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "scikit-image",
        "citation": "van der Walt et al. (2014) PeerJ 2:e453",
    }


# ---- Organoid (confocal, IDR remote Zarr) ----

@register("organoid_ch0")
def load_organoid_ch0() -> tuple[np.ndarray, dict[str, Any]]:
    """Mouse intestinal organoid channel 0 (confocal, IDR). Full-res, center crop if needed."""
    import fsspec
    import zarr

    zarr_url = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
    max_voxels = 20_000_000  # ~20M voxels max to keep fitting tractable

    with asection("Loading organoid from IDR Zarr"):
        aprint(f"Source: {zarr_url}")
        mapper = fsspec.get_mapper(zarr_url)
        store = zarr.open_group(mapper, mode="r")
        data = store["0"]  # Full resolution — no downsampling
        aprint(f"Full shape: {data.shape}")  # (T, C, Z, Y, X)

        _, _, z_size, y_size, x_size = data.shape
        total_voxels = z_size * y_size * x_size
        aprint(f"Spatial: {z_size}x{y_size}x{x_size} = {total_voxels:,} voxels")

        if total_voxels <= max_voxels:
            V = np.array(data[0, 0, :, :, :], dtype=np.float32)
            aprint(f"Loaded full volume: {V.shape}")
        else:
            # Crop only along the largest spatial axis to reduce volume while
            # keeping full FOV in the other two axes. This preserves biological
            # structure (e.g., organoid geometry).
            shape = [z_size, y_size, x_size]
            largest = int(np.argmax(shape))
            target_len = max_voxels // (
                shape[(largest + 1) % 3] * shape[(largest + 2) % 3]
            )
            target_len = min(target_len, shape[largest])
            start = (shape[largest] - target_len) // 2
            slc = [slice(None)] * 3
            slc[largest] = slice(start, start + target_len)
            axis_name = "ZYX"[largest]
            aprint(
                f"Cropping {axis_name}-axis: {shape[largest]} -> {target_len} "
                f"(center, keeping full FOV in other axes)"
            )
            V = np.array(
                data[0, 0, slc[0], slc[1], slc[2]],
                dtype=np.float32,
            )
            aprint(f"Cropped: {V.shape} ({V.size:,} voxels)")

        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        V = V.astype(np.float32)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "Organoid — Channel 0 (full-res)",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "IDR idr0062",
        "citation": "Blin et al. (2019) / Williams et al. (2017) Nat Methods 14:775",
    }


# ---- C. elegans embryo (confocal, single timepoint from 4D movie) ----

@register("celegans_t100")
def load_celegans_t100() -> tuple[np.ndarray, dict[str, Any]]:
    """C. elegans embryo, single mid-movie timepoint t=100. 41x512x512."""
    import zipfile

    import tifffile

    cache_dir = ANALYSIS_CACHE / "gsplats_celegans"
    timepoint = 100  # Mid-movie (~400 total)
    sample_name = "mskcc_confocal_s1"

    # 1. Check if the demo already extracted this timepoint's TIFF
    extract_dir = cache_dir / "extracted"
    existing_tiffs = sorted(extract_dir.rglob("*.tif")) if extract_dir.exists() else []
    sample_tiffs = [f for f in existing_tiffs if sample_name in str(f)]

    tiff_path = None
    if sample_tiffs and timepoint < len(sample_tiffs):
        tiff_path = sample_tiffs[timepoint]
        aprint(f"Found cached TIFF from demo: {tiff_path.name}")

    # 2. If not found, locate the ZIP (demo cache or download)
    if tiff_path is None:
        zip_path = cache_dir / "mskcc_confocal.zip"
        if not zip_path.exists():
            from luxar.utils.download import robust_download

            cache_dir.mkdir(parents=True, exist_ok=True)
            with asection("Downloading C. elegans dataset from Zenodo (~26 GB)"):
                robust_download(
                    "https://zenodo.org/api/records/6460303/files/"
                    "mskcc_confocal.zip/content",
                    zip_path,
                    max_retries=5,
                    timeout=600,
                )

        # Extract just the one timepoint
        with asection(f"Extracting timepoint {timepoint}"):
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = sorted(
                    m for m in zf.namelist()
                    if sample_name in m and m.lower().endswith((".tif", ".tiff"))
                )
                aprint(f"Found {len(members)} TIFFs for {sample_name}")
                if timepoint >= len(members):
                    raise ValueError(
                        f"Timepoint {timepoint} out of range "
                        f"(max {len(members) - 1})"
                    )
                member = members[timepoint]
                target = extract_dir / member
                if not target.exists():
                    zf.extract(member, extract_dir)
                tiff_path = target
                aprint(f"Extracted: {member}")

    with asection("Loading timepoint volume"):
        V = tifffile.imread(str(tiff_path)).astype(np.float32)
        if V.ndim == 4 and V.shape[0] <= 4:
            V = V[0]
        elif V.ndim == 4 and V.shape[-1] <= 4:
            V = V[..., 0]
        # Robust normalisation (P1-P99.999 — matches demo)
        V = _normalize_volume(V, 1.0, 99.999)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": f"C. elegans Embryo — t={timepoint}",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "Zenodo 6460303",
        "citation": "Bao et al. (2006) PNAS / Murray et al. (2006)",
    }


# ---- Tribolium embryo (light-sheet, whole embryo) ----

@register("tribolium")
def load_tribolium() -> tuple[np.ndarray, dict[str, Any]]:
    """Tribolium castaneum embryo (light-sheet). 965x1871x991, cropped along largest axis."""
    import zipfile

    import tifffile

    cache_dir = ANALYSIS_CACHE / "gsplats_tribolium"
    max_voxels = 100_000_000  # ~100M voxels — gives ~104 Y-slices from 1871

    # 1. Check if the demo already has the ZIP cached
    zip_path = cache_dir / "Supplemental_File_2.zip"
    if not zip_path.exists():
        from luxar.utils.download import robust_download

        cache_dir.mkdir(parents=True, exist_ok=True)
        with asection("Downloading Tribolium dataset from Zenodo (~2.6 GB)"):
            robust_download(
                "https://zenodo.org/api/records/5270323/files/"
                "Supplemental_File_2.zip/content",
                zip_path,
                max_retries=5,
                timeout=600,
            )

    # 2. Extract TIFFs
    extract_dir = cache_dir / "extracted"
    with asection("Extracting Tribolium volume"):
        tiff_files = sorted(
            f for f in extract_dir.rglob("*") if f.suffix.lower() in (".tif", ".tiff")
        ) if extract_dir.exists() else []
        if not tiff_files:
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                tiff_members = [
                    m for m in zf.namelist()
                    if m.lower().endswith((".tif", ".tiff"))
                ]
                aprint(f"Extracting {len(tiff_members)} TIFFs...")
                for m in tiff_members:
                    zf.extract(m, extract_dir)
            tiff_files = sorted(
                f for f in extract_dir.rglob("*") if f.suffix.lower() in (".tif", ".tiff")
            )
        aprint(f"Found {len(tiff_files)} TIFF files")

    # 3. Load volume
    with asection("Loading Tribolium volume"):
        if len(tiff_files) == 1:
            V = tifffile.imread(str(tiff_files[0])).astype(np.float32)
        else:
            V = tifffile.imread([str(f) for f in tiff_files]).astype(np.float32)

        if V.ndim == 4 and V.shape[0] <= 4:
            V = V[0]
        elif V.ndim == 4 and V.shape[-1] <= 4:
            V = V[..., 0]

        aprint(f"Raw shape: {V.shape}")

        # 4. Crop along largest axis only (preserves full FOV in other two)
        total = V.size
        if total > max_voxels:
            shape = list(V.shape)
            largest = int(np.argmax(shape))
            target_len = max_voxels // (
                shape[(largest + 1) % 3] * shape[(largest + 2) % 3]
            )
            target_len = min(target_len, shape[largest])
            start = (shape[largest] - target_len) // 2
            slc = [slice(None)] * 3
            slc[largest] = slice(start, start + target_len)
            axis_name = "ZYX"[largest]
            aprint(
                f"Cropping {axis_name}-axis: {shape[largest]} -> {target_len} "
                f"(center, full FOV in other axes)"
            )
            V = V[tuple(slc)]
        aprint(f"Volume after crop: {V.shape} ({V.size:,} voxels)")

        # 5. Normalize (min-max)
        V = _normalize_volume(V, 0.0, 100.0)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "Tribolium Embryo (light-sheet, cropped)",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "Zenodo 5270323",
        "citation": "Cell Tracking Challenge / Stegmaier et al.",
    }


# ---- OpenCell LMNB1 (spinning-disk confocal, nuclear lamina) ----

@register("opencell_lmnb1_ch0")
def load_opencell_lmnb1_ch0() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell LMNB1 Hoechst (nuclei). 51x600x600."""
    return _load_opencell_target("LMNB1", "ENSG00000113368", "CID000892", "FID00003884", 0)


@register("opencell_lmnb1_ch1")
def load_opencell_lmnb1_ch1() -> tuple[np.ndarray, dict[str, Any]]:
    """OpenCell LMNB1-GFP (nuclear lamina). 51x600x600."""
    return _load_opencell_target("LMNB1", "ENSG00000113368", "CID000892", "FID00003884", 1)


def _load_opencell_target(
    gene: str, ensg: str, cid: str, fid: str, channel: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a channel from any OpenCell target."""
    import tifffile

    cache_dir = ANALYSIS_CACHE / f"gsplats_opencell_{gene.lower()}"
    filename = f"OC-FOV_{gene}_{ensg}_{cid}_{fid}_stack.tif"
    url = (
        f"https://czb-opencell.s3.amazonaws.com/microscopy/raw/"
        f"{gene}_{ensg}/{filename}"
    )
    tiff_path = _download_cached(url, cache_dir, filename, f"OpenCell {gene} TIFF (~70 MB)")

    with asection(f"Loading OpenCell {gene} channel {channel}"):
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        if data.ndim == 4 and data.shape[1] == 2:
            V = np.array(data[:, channel], dtype=np.float32)
        elif data.ndim == 4 and data.shape[0] == 2:
            V = np.array(data[channel], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected TIFF shape: {data.shape}")
        del data

        V = _normalize_volume(V, 1.0, 99.5)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    ch_names = {0: "Hoechst (Nuclei)", 1: f"{gene}-GFP"}
    return V, {
        "name": f"OpenCell {gene} — {ch_names.get(channel, f'ch{channel}')}",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "OpenCell",
        "citation": "Cho et al. (2022) Science 375(6585)",
    }


# ---- HeLa cells3d (confocal, scikit-image) ----

@register("cells3d_nuclei")
def load_cells3d_nuclei() -> tuple[np.ndarray, dict[str, Any]]:
    """HeLa cells nuclei channel (confocal). 60x256x256."""
    with asection("Loading cells3d nuclei from scikit-image"):
        from skimage.data import cells3d

        raw = cells3d()  # (60, 2, 256, 256) uint16
        aprint(f"Raw shape: {raw.shape}, dtype: {raw.dtype}")
        V = raw[:, 1, :, :].astype(np.float32)  # Channel 1 = nuclei
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "HeLa Cells — Nuclei",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "scikit-image cells3d",
        "citation": "van der Walt et al. (2014) PeerJ 2:e453",
    }


@register("cells3d_membrane")
def load_cells3d_membrane() -> tuple[np.ndarray, dict[str, Any]]:
    """HeLa cells membrane channel (confocal). 60x256x256."""
    with asection("Loading cells3d membrane from scikit-image"):
        from skimage.data import cells3d

        raw = cells3d()
        V = raw[:, 0, :, :].astype(np.float32)  # Channel 0 = membrane
        V = (V - V.min()) / (V.max() - V.min() + 1e-8)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "HeLa Cells — Membrane",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "scikit-image cells3d",
        "citation": "van der Walt et al. (2014) PeerJ 2:e453",
    }


# ---- Acto3D mouse heart (light-sheet, single channel, cropped) ----

@register("acto3d_heart_nuclei")
def load_acto3d_heart_nuclei() -> tuple[np.ndarray, dict[str, Any]]:
    """Mouse embryo heart nuclei (light-sheet, SYTOX Green). Cropped from 597x960x960."""
    import tifffile

    cache_dir = ANALYSIS_CACHE / "gsplats_acto3d_heart"
    tiff_name = "acto3d_heart_E13_5.tif"
    tiff_path = cache_dir / tiff_name
    max_voxels = 100_000_000

    # Download from Google Drive if not cached
    if not tiff_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Direct download approach (simpler than loading the demo module)

        # Direct download approach
        gdrive_id = "1VHiLkK2O1ZrWoWX4ahPwnZfNgDQ242Kz"
        with asection("Downloading Acto3D heart from Google Drive (~1.65 GB)"):
            import requests
            url = f"https://drive.usercontent.google.com/download?id={gdrive_id}&confirm=t"
            aprint("Downloading from Google Drive...")
            resp = requests.get(url, stream=True)
            resp.raise_for_status()
            tmp = tiff_path.with_suffix(".tmp")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
            tmp.rename(tiff_path)
            aprint(f"Saved to {tiff_path}")

    with asection("Loading Acto3D heart"):
        data = tifffile.imread(str(tiff_path))
        aprint(f"Raw shape: {data.shape}, dtype: {data.dtype}")

        # The TIFF is interleaved: (Z*C, Y, X) or (C, Z, Y, X) or (Z, C, Y, X)
        if data.ndim == 3:
            # Interleaved Z*C — 3 channels
            n_channels = 3
            n_planes = data.shape[0] // n_channels
            data = data.reshape(n_planes, n_channels, data.shape[1], data.shape[2])
        if data.ndim == 4:
            if data.shape[1] <= 4:  # (Z, C, Y, X)
                V = data[:, 0, :, :].astype(np.float32)
            elif data.shape[0] <= 4:  # (C, Z, Y, X)
                V = data[0, :, :, :].astype(np.float32)
            else:
                V = data[:, 0, :, :].astype(np.float32)
        else:
            V = data.astype(np.float32)
        del data

        aprint(f"Channel 0 (nuclei) shape: {V.shape}")

        # Crop along largest axis if too big
        total = V.size
        if total > max_voxels:
            shape = list(V.shape)
            largest = int(np.argmax(shape))
            target_len = max_voxels // (
                shape[(largest + 1) % 3] * shape[(largest + 2) % 3]
            )
            target_len = min(target_len, shape[largest])
            start = (shape[largest] - target_len) // 2
            slc = [slice(None)] * 3
            slc[largest] = slice(start, start + target_len)
            axis_name = "ZYX"[largest]
            aprint(f"Cropping {axis_name}-axis: {shape[largest]} -> {target_len}")
            V = V[tuple(slc)]

        V = _normalize_volume(V, 0.0, 100.0)
        aprint(f"Volume: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

    return V, {
        "name": "Mouse Heart — Nuclei (light-sheet, cropped)",
        "shape": V.shape,
        "ndim": V.ndim,
        "source": "Acto3D / Google Drive",
        "citation": "Acto3D project (https://github.com/Acto3D/Acto3D)",
    }
