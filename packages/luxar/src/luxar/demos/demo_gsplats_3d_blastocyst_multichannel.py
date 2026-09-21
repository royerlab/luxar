#!/usr/bin/env python3
"""GSplats Demo: Multi-Channel Mouse Blastocyst (IDR) - Full Compute Pipeline

Visualises multi-channel 3D microscopy data as Gaussian splats.
By default, uses precomputed gsplats from Git LFS (fast).
Use --recompute to fetch data from IDR and fit from scratch (slow, needs GPU).

================================================================================
MULTI-CHANNEL GAUSSIAN SPLATTING - FULL PIPELINE
================================================================================

This demo shows the complete workflow for multi-channel Gaussian splatting:
- Fetching real microscopy data from Image Data Resource (IDR)
- Fitting Gaussian splats to each channel independently
- Merging channels with distinct colors for visualization

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Image ID: 6001240 (idr6001240)
Source: Image Data Resource (IDR) - https://idr.openmicroscopy.org/
Study: idr0062-blin-nuclearsegmentation - nuclear segmentation benchmark
Format: OME-ZARR 5D (Time × Channel × Z × Y × X)
Data Type: High-resolution 3D confocal microscopy of a mouse blastocyst (E3.5),
           imaged for segmentation benchmarking

Original Authors & Study:
--------------------------
Principal Investigator: Sally Lowell
Institution: University of Edinburgh (data published by the University of Dundee)

The image is one of the benchmark volumes behind Nessys, a nuclear-segmentation
method for dense 3D tissue.

How to Cite:
------------
If you use this dataset, please cite:

1. Original Research (what the credit in DEMO_META names):
   Blin, G., Sadurska, D., Portero Migueles, R., Chen, N., Watson, J.A.,
   Lowell, S. (2019). "Nessys: A new set of tools for the automated detection
   of nuclei within intact tissues and dense 3D cultures." PLoS Biology.
   DOI: 10.1371/journal.pbio.3000388 (CC BY 4.0)

2. Image Data Resource (IDR) — the repository, not the data's authors:
   Williams, E. et al. (2017). "The Image Data Resource: a bioimage data
   integration and publication platform."
   Nature Methods, 14(8), 775-781.
   DOI: 10.1038/nmeth.4326

3. Data Accession:
   IDR study idr0062, Image 6001240
   URL: https://idr.openmicroscopy.org/webclient/?show=image-6001240

WORKFLOW:
=========

1. **Load multi-channel data** from Image Data Resource (IDR)
   - Channel 0: Lamin B1 immunostain (ab16048) - the nuclear LAMINA, so it
     draws the envelope AROUND each nucleus. This is the shell-like channel.
   - Channel 1: DAPI (DNA stain) - FILLS each nucleus.
   Both labels are read off the OME-Zarr's own `omero.channels` metadata and
   IDR's protocol annotation, not inferred from the render.

2. **Fit each channel independently**
   - Each channel gets its own set of Gaussian splats
   - Captures channel-specific structures

3. **Add each channel as a separate layer**
   - Magenta: Lamin B1 (nuclear envelope)
   - Cyan: DAPI (DNA)
   - Each channel is a toggleable layer in the viewer

4. **Visualize** in the Luxar viewer
   - Additive blending sums the superimposed channels, so overlap reads as a
     colour mix (volumetric compositing would make one channel occlude the other)
   - Toggle layers to inspect individual channels

USAGE:
======
    python demo_gsplats_3d_blastocyst_multichannel.py [--recompute] [--no-serve] [--no-napari]

Options:
    --recompute:      Force re-fitting from scratch (download + GPU fitting)
    --no-serve:       Don't auto-launch viewer after scene creation
    --no-napari:      Skip napari visualization (useful for headless/CI)
    --serve-only:     Skip fitting, just serve existing scene
    --show-roundtrip: Show matplotlib comparison of original vs reconstructed volumes
    --synthetic:      Use clearly labelled procedural stand-in channels, not microscopy

By default, precomputed GSplats are loaded from package data (Git LFS).
Use --recompute to re-fit from scratch (requires network + GPU).

"""

DEMO_META = {
    "key": "gsplats_3d_blastocyst_multichannel",
    "title": "Multichannel 3D Mouse Blastocyst",
    "description": "Two-channel confocal mouse blastocyst (Lamin B1 + DAPI, IDR idr0062) as colored Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 1,
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_multichannel"],
    "outputs": ["gsplats_3d_blastocyst_multichannel"],
    # The study that produced the image, not the repository that hosts it:
    # IDR's own record for idr0062 names Blin et al. and the PLoS Biology DOI,
    # and crediting the IDR platform paper instead would attribute someone
    # else's data to the archive it happens to sit in.
    "citation": {
        "short": "Blin et al. 2019",
        "doi": "10.1371/journal.pbio.3000388",
        "license": "CC BY 4.0",
    },
}

# Enable MPS→CPU fallback for unsupported PyTorch ops (must be before torch import)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
from pathlib import Path

import numpy as np
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimensions, LuxarZarrCompiler
from luxar.colormaps import resolve_colormap
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    launch_viewer,
    load_dataset_gsplats,
    load_local_fit_gsplats,
    local_fit_path,
    parse_demo_flags,
    stamp_input_digests,
    warn_if_no_cuda_gpu,
)
from luxar.demos._lod_policy import save_with_lod
from luxar.demos._roundtrip_common import show_roundtrip_comparison
from luxar.encoding import EncodingMode
from luxar.gsplats import fit_gaussian_splats
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.models.gsplats.metal import is_metal_available
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZARR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.2/6001240.zarr"
TARGET_SIZE = 256  # Edge length of explicitly requested synthetic volumes.
TIME_POINT = 0  # First time point

# Channel configuration with colors.
#
# The channel identities are not guesswork: the OME-Zarr's own `omero.channels`
# labels them `LaminB1` and `Dapi` in that order, and IDR's bulk annotation for
# this image records the protocol — "Immunostaining: LaminB1 antibody: ab16048
# (dilution 1:1000)". Lamin B1 is a nuclear LAMINA protein, so channel 0 outlines
# the nuclear envelope of every cell; that is the membrane-looking channel, and
# it is the one Nessys segments on (DAPI fills nuclei, which is exactly what
# makes touching nuclei hard to separate — the envelope is what separates them).
CHANNELS = [
    {
        "index": 0,
        "name": "Lamin B1",
        "colormap": "magenta",
        "blurb": "nuclear envelope (immunostain, ab16048)",
    },
    {
        "index": 1,
        "name": "DAPI",
        "colormap": "cyan",
        "blurb": "DNA — fills each nucleus",
    },
]

# Fit parameters (fixed-K, seeds=K*)
MAX_SPLATS = 22000
DEVICE = None  # Auto-detect (cuda/mps/cpu)

# Manifest dataset + the files it pins, one per channel. A local refit is OUR
# artifact, not a copy of the hosted one, so it lives in the demo's local-fit
# namespace (~/.cache/luxar/<name>/local/, see `local_fit_path`). Writing it to
# ~/.cache/luxar/<name>/<file> — the path the manifest fetch owns — got it
# quarantined on the next launch for failing the pinned sha256 (#1618).
DEMO_NAME = "gsplats_multichannel"
# Renamed with the rest. These are the pinned artifact names in
# `data_manifest.json` and on the published Zenodo record; `sha256` / `bytes`
# describe the record copies directly.
GSPLATS_FILES = [
    "blastocyst_ch0.gsplats.zarr.zip",
    "blastocyst_ch1.gsplats.zarr.zip",
]
# The synthetic cache names, defined ONCE. The writer and the reader derived
# them separately before, so the writer stored synthetic_* and the reader asked
# for the real names: the synthetic cache was write-only and every --synthetic
# run refit from scratch.
SYNTHETIC_GSPLATS_FILES = [f"synthetic_{name}" for name in GSPLATS_FILES]

# Parse command-line flags
FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
NO_NAPARI = "--no-napari" in sys.argv
SHOW_ROUNDTRIP = "--show-roundtrip" in sys.argv
# Opt-in ONLY; never reached by a failure (acquisition errors raise).
SYNTHETIC = "--synthetic" in sys.argv
SCENE_STEM = (
    "gsplats_3d_blastocyst_multichannel_SYNTHETIC"
    if SYNTHETIC
    else "gsplats_3d_blastocyst_multichannel"
)

# Setup
Arbol.max_depth = 10


# =============================================================================
# Data Loading
# =============================================================================


def load_multichannel_data():
    """Load and preprocess multi-channel microscopy data from IDR.

    Returns the IDR volumes at native resolution (no zoom resample) — this
    matches the manuscript's supp_doc dataset, which records the same volume
    under the old ``organoid_ch0`` id. That id is quoted rather than corrected
    because it is what is written in a document this repo does not own; the
    volume is a mouse BLASTOCYST channel, and the manuscript should be fixed
    too when it is next touched.  An earlier
    version of this loader force-resampled to TARGET_SIZE^3 via bilinear
    ``scipy.ndimage.zoom``; that smoothed away high-frequency noise and
    pushed the held-out PSNR ceiling ~5 dB above the paper's reference,
    biasing K* upward.  Native resolution at this image (≤20M voxels) is
    plenty tractable.

    Returns:
        tuple: ``(volumes, source_dtype)`` -- one volume per channel, and the
        element type the store holds, to be declared to the fit so compression is
        quoted against the acquisition and not the float32 working copy.

    Data Source: Image Data Resource (IDR) study idr0062, Image 6001240
    Original Authors: Blin et al., Lowell lab (University of Edinburgh)
    Citation: Blin et al. (2019), PLoS Biology, doi:10.1371/journal.pbio.3000388
    """
    with asection("Loading multi-channel microscopy data"):
        aprint(f"Source: {ZARR_URL}")
        aprint("Dataset: IDR idr0062, Image 6001240 (Blin et al. 2019, Lowell lab)")
        aprint("Resolution: native (no zoom resample)")

        try:
            import fsspec

            # Open remote zarr
            mapper = fsspec.get_mapper(ZARR_URL)
            store = zarr.open_group(mapper, mode="r")
            data = store["0"]  # Highest resolution
            full_shape = data.shape

            aprint(f"Full data shape: {full_shape}")

            if len(full_shape) != 5:
                raise ValueError(
                    f"Expected 5D data (T×C×Z×Y×X), got shape {full_shape}"
                )

            n_time, n_channels, z_size, y_size, x_size = full_shape
            aprint("Format: OME-ZARR 5D (T×C×Z×Y×X)")
            aprint(f"  Time points: {n_time}")
            aprint(f"  Channels: {n_channels}")
            aprint(f"  Spatial: {z_size}×{y_size}×{x_size}")

            # Load each channel at native resolution.
            volumes = []
            for ch_config in CHANNELS:
                ch_idx = ch_config["index"]
                ch_name = ch_config["name"]

                if ch_idx >= n_channels:
                    aprint(
                        f"Warning: Channel {ch_idx} not available (only {n_channels} channels)"
                    )
                    continue

                aprint(f"Loading T={TIME_POINT}, C={ch_idx} ({ch_name})...")
                V = np.array(data[TIME_POINT, ch_idx, :, :, :], dtype=np.float32)

                # Normalize to [0, 1]
                V = (V - V.min()) / (V.max() - V.min() + 1e-8)
                V = V.astype(np.float32)

                volumes.append(V)
                aprint(f"  {ch_name}: {V.shape}, range [{V.min():.3f}, {V.max():.3f}]")

            aprint(f"Loaded {len(volumes)} channels")
            return volumes, str(data.dtype)

        except Exception as exc:
            # FAIL, rather than fabricate. See the sibling DAPI demo for the
            # full reasoning: this used to catch everything and return
            # procedurally generated blobs, which the scene then published
            # under Blin et al.'s DOI and CC BY 4.0 notice with a description
            # of a Leica SP8 acquisition. Fabricated imagery carrying real
            # biological provenance is the one failure this tool must not have.
            raise DatasetUnavailable(
                f"could not load IDR idr0062 image 6001240 from {ZARR_URL}: {exc}. "
                "Check the network and that `fsspec` is installed. Pass "
                "--synthetic for clearly-labelled stand-in channels with no "
                "citation, which are not a substitute for this dataset."
            ) from exc


def synthesize_channel_volumes(seed: int = 42):
    """Procedurally generate per-channel blobs. NOT microscopy.

    Reachable only via ``--synthetic``. Cache files, scene name, title,
    description, caption and citation are all switched by the same flag, so
    these arrays cannot inherit the real dataset's identity.

    Returns ``(volumes, None)``: synthesized here, so they ARE their own source.
    """
    with asection("Synthesizing stand-in channels (NOT real microscopy)"):
        volumes = []
        shape = (TARGET_SIZE, TARGET_SIZE, TARGET_SIZE)
        grids = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")

        for ch_idx, ch_config in enumerate(CHANNELS):
            V = np.zeros(shape, dtype=np.float32)
            rng = np.random.default_rng(seed + ch_idx)  # reproducible per channel
            for _ in range(10 + ch_idx * 5):
                center = [rng.uniform(15, s - 15) for s in shape]
                sigma = rng.uniform(4, 10)
                amplitude = rng.uniform(0.5, 1.0)
                dist_sq = sum((g - c) ** 2 for g, c in zip(grids, center))
                V += amplitude * np.exp(-dist_sq / (2 * sigma**2))

            V = np.clip(V, 0, 1).astype(np.float32)
            volumes.append(V)
            aprint(f"  {ch_config['name']}: {V.shape} (synthetic)")

        return volumes, None


# =============================================================================
# GSplats Fitting
# =============================================================================


def fit_channel(volume, channel_name, cache_file, source_dtype=None):
    """Fit gsplats to a single channel (always fits — caller handles precomputed)."""
    # Auto-detect best device
    global DEVICE
    if DEVICE is None:
        import torch

        if is_metal_available() and torch.backends.mps.is_available():
            DEVICE = "mps"
            aprint("Using MPS device (Metal acceleration)")
        elif torch.cuda.is_available():
            DEVICE = "cuda"
            aprint("Using CUDA device")
        else:
            DEVICE = "cpu"
            aprint("Using CPU device")

    # Fit gsplats progressively
    aprint(f"Fitting {channel_name} (fixed-K joint fit: seeds={MAX_SPLATS})...")

    result = fit_gaussian_splats(
        volume,
        seeds=MAX_SPLATS,
        # The grid is the acquisition's; only the element type was changed
        # on the way here, and that is the denominator of the ratio.
        source_dtype=source_dtype,
        device=DEVICE,
        verbose=True,
    )

    n_splats = len(result.amplitudes)
    aprint(f"  Fitted {n_splats} splats")

    # Cache result in compressed zarr format
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Caching to {cache_file}")
    save_with_lod(
        result,
        cache_file,
        recipe="stream",
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )

    return GSplatData.load(cache_file, include_stats=False)


def fit_all_channels(volumes, source_dtype=None):
    """Fit gsplats to all channels."""
    with asection("Fitting GSplats per channel"):
        gsplats_list = []

        for i, (volume, ch_config) in enumerate(zip(volumes, CHANNELS)):
            ch_name = ch_config["name"]
            # Separate cache namespace under --synthetic: sharing it would let
            # one offline run poison every later online run through the cache.
            cache_file = local_fit_path(
                DEMO_NAME,
                (SYNTHETIC_GSPLATS_FILES if SYNTHETIC else GSPLATS_FILES)[i],
            )

            with asection(f"Channel {i}: {ch_name}"):
                gsplats = fit_channel(
                    volume, ch_name, cache_file, source_dtype=source_dtype
                )
                gsplats_list.append(gsplats)

        return gsplats_list


# =============================================================================
# Napari Viewing
# =============================================================================


def view_with_napari(volumes, gsplats_list, channel_configs):
    """Open original volumes and gsplat renderings in napari for comparison."""
    try:
        import napari
    except ImportError:
        aprint("napari not installed, skipping napari view")
        aprint("Install with: pip install 'napari[all]>=0.8'")
        return

    def _colormap_for_channel(idx, name):
        if idx == 0:
            return "magenta"
        if idx == 1:
            return "cyan"
        return "gray"

    with asection("Opening in napari"):
        aprint("Preparing napari visualization...")

        rendered_volumes = []
        for idx, (volume, gsplats, ch_config) in enumerate(
            zip(volumes, gsplats_list, channel_configs)
        ):
            ch_name = ch_config["name"]
            aprint(f"Rendering gsplats for {ch_name}...")
            rendered = gsplats.render_to_volume(
                shape=tuple(dim_len * 2 for dim_len in volume.shape)
            )
            rendered_volumes.append(rendered)

        aprint("Launching napari...")
        viewer = napari.Viewer(title="GSplats vs Original - Blastocyst Channels")

        for idx, (volume, rendered, ch_config) in enumerate(
            zip(volumes, rendered_volumes, channel_configs)
        ):
            ch_name = ch_config["name"]
            cmap = _colormap_for_channel(idx, ch_name)

            viewer.add_image(
                volume,
                name=f"Original {ch_name}",
                colormap=cmap,
                opacity=1.0,
                blending="additive",
            )
            viewer.add_image(
                rendered,
                name=f"GSplats {ch_name}",
                colormap=cmap,
                opacity=1.0,
                blending="additive",
            )

        aprint("Napari opened - toggle layers to compare channels")
        napari.run()


# =============================================================================
# Scene Creation
# =============================================================================


def legend_css(colormap_name: str) -> str:
    """The CSS colour of a colormap's top end, for a legend swatch.

    Read out of the LUT the renderer will actually use rather than restated as a
    literal, so the legend cannot drift away from the layer it labels.
    """
    top = np.asarray(resolve_colormap(colormap_name))[-1]
    return f"rgb({int(top[0])},{int(top[1])},{int(top[2])})"


def create_luxar_scene(gsplats_list, output_path: Path | None = None):
    """Create Luxar scene with per-channel gsplat layers."""
    if output_path is None:
        output_path = get_demos_output_dir() / f"{SCENE_STEM}.luxar.zarr"

    with asection("Creating Luxar Scene"):
        aprint(f"Output: {output_path.name}")

        # Compute shared centroid across all channels (amplitude-weighted)
        all_centers = [g.centers for g in gsplats_list]
        all_amps = [g.amplitudes for g in gsplats_list]
        total_amp = sum(a.sum() for a in all_amps)
        if total_amp > 0:
            shared_centroid = (
                sum(c.T @ a for c, a in zip(all_centers, all_amps)) / total_amp
            )
        else:
            shared_centroid = np.mean(np.concatenate(all_centers, axis=0), axis=0)

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # ACES, set explicitly, for the per-channel colormaps
            # (ACES is the house default; it shifts LUT hues slightly, which is
            # the accepted trade for its highlight rolloff).
            scene = compiler.create_scene(
                dimensions=Dimensions.default_3d(),
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
                # A DOI asserts provenance; synthetic blobs have none.
                citation=None if SYNTHETIC else DEMO_META["citation"],
            )
            stamp_input_digests(scene)

            # Add scene metadata
            scene.attrs["title"] = (
                "GSplats: SYNTHETIC two-channel stand-in (not microscopy)"
                if SYNTHETIC
                else "GSplats: Mouse Blastocyst, Two Channels (Lamin B1 + DAPI)"
            )
            if SYNTHETIC:
                scene.attrs["description"] = """
Synthetic Multi-Channel Gaussian Splats — NOT Microscopy
========================================================

This scene contains two procedurally generated intensity volumes fitted as
Gaussian splats. Nothing here was measured, no biological specimen or imaging
instrument is represented, and the channels do not correspond to stains,
antibodies or molecular structures.

It exists only to exercise the multi-channel fitting, layering and rendering
pipeline without network access. Run without --synthetic for the real dataset.

Toggle layers in the viewer (press L) to inspect the generated channels.

Controls:
- Mouse drag to rotate
- Scroll to zoom
- Right-click drag to pan
- 'C' to toggle fly controls
                """
            else:
                scene.attrs["description"] = """
Multi-Channel Gaussian Splatting — Mouse Blastocyst (E3.5)
===========================================================

A wild-type mouse blastocyst at embryonic day 3.5, imaged on a Leica SP8
confocal (HC PL APO 40x/1.30 Oil) in two channels and fitted as Gaussian
splats, one independent fit per channel, each shown as its own toggleable
layer.

The two channels are complementary, not redundant:
- Magenta — Lamin B1, immunostained (ab16048). Lamin B1 is a nuclear LAMINA
  protein, so this channel draws the ENVELOPE around each nucleus. It is the
  shell-like channel, and it is what makes touching nuclei separable.
- Cyan — DAPI, which binds DNA and so FILLS each nucleus.

That pairing is the point of the source dataset: it is a benchmark volume for
Nessys, a nuclear-segmentation method that works from the envelope rather than
from the DNA, precisely because densely packed nuclei merge in a DAPI channel
but stay individually outlined in a lamina channel.

Data Source:
  - Image Data Resource (IDR) study idr0062, Image 6001240 (B1_C1.tif)
  - Original research: Blin et al. (2019), PLoS Biology (Lowell lab, Edinburgh),
    doi:10.1371/journal.pbio.3000388, CC BY 4.0

Toggle layers in the viewer (press L) to inspect individual channels.

Controls:
- Mouse drag to rotate
- Scroll to zoom
- Right-click drag to pan
- 'C' to toggle fly controls
                """

            # Add each channel as a separate layer
            for i, (gsplats, ch_config) in enumerate(
                zip(gsplats_list, CHANNELS[: len(gsplats_list)])
            ):
                ch_name = ch_config["name"]
                colormap = ch_config["colormap"]

                # Center using shared centroid and reduce brightness
                centered = gsplats.translate(-shared_centroid)
                centered = centered.scale_intensity(0.1)

                aprint(
                    f"Adding {ch_name} ({len(centered.amplitudes)} splats, "
                    f"colormap={colormap})..."
                )
                scene.add_gsplats_from_data(
                    name=f"ch{i}_{ch_name.lower().replace(' ', '_')}",
                    result=centered,
                    opacity=1.0,
                    # Additive, not volumetric: the two channels are
                    # SUPERIMPOSED over the same volume, and volumetric
                    # compositing makes whichever layer draws first occlude the
                    # other, so channel overlap reads as one channel hiding the
                    # rest instead of the two colors mixing.
                    blending_mode="additive",
                    layer=True,
                    colormap=colormap,
                )
            # ── Overlays ────────────────────────────────────────────────────
            # Canonical title (top-left) + data-source caption (bottom-right),
            # matching the other gsplat demos. The caption used to read
            # "Light-sheet microscopy", which is wrong for this image: IDR's
            # protocol annotation records a Leica SP8 point-scanning confocal.
            scene.add_text(
                "SYNTHETIC stand-in • not microscopy"
                if SYNTHETIC
                else "Mouse Blastocyst • Lamin B1 + DAPI",
                position=(0.02, 0.02),
                font_size=0.048,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                f"SYNTHETIC data • {len(gsplats_list)} generated channels"
                if SYNTHETIC
                else f"Leica SP8 confocal • {len(gsplats_list)} channels • IDR idr0062",
                None if SYNTHETIC else DEMO_META.get("citation"),
            )

            # Explanatory panel, placed below the title so the two don't overlap
            # — same slot and styling as the LOD demos' panel.
            scene.add_text(
                (
                    "Procedurally generated intensity blobs.\n"
                    "Two independent channels are shown as\n"
                    "toggleable layers for pipeline testing.\n"
                    "Nothing here was measured and neither\n"
                    "channel represents a stain, antibody,\n"
                    "specimen or biological structure.\n"
                    "Press L for the Layers panel."
                    if SYNTHETIC
                    else "Mouse blastocyst (E3.5). Two channels, fitted\n"
                    "independently, shown as toggleable layers.\n"
                    "Lamin B1 is a nuclear LAMINA protein, so it\n"
                    "draws the envelope AROUND each nucleus; DAPI\n"
                    "binds DNA and FILLS it. Densely packed nuclei\n"
                    "merge in the DAPI channel but stay separable\n"
                    "in the envelope one — which is why this image\n"
                    "is a nuclear-segmentation benchmark.\n"
                    "Press L for the Layers panel."
                ),
                position=(0.02, 0.10),
                font_size=0.020,
                font="mono",
                color="white",
                width=0.46,
                line_height=1.45,
                background="rgba(0,0,0,0.55)",
                padding=0.012,
            )

            # Per-channel legend, each line tinted to match its layer. Laid out
            # upward from a fixed bottom so it stays on-screen for any channel
            # count.
            spacing = 0.034
            start_y = 0.94 - spacing * (len(gsplats_list) - 1)
            for i, ch_config in enumerate(CHANNELS[: len(gsplats_list)]):
                scene.add_text(
                    (
                        f"● Generated channel {i + 1}: procedural intensity blobs"
                        if SYNTHETIC
                        else f"● {ch_config['name']}: {ch_config['blurb']}"
                    ),
                    position=(0.02, start_y + spacing * i),
                    font_size=0.020,
                    font="mono",
                    color=legend_css(ch_config["colormap"]),
                    stroke_color="black",
                    stroke_width=0.0018,
                )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def resolve_gsplats() -> list[GSplatData] | None:
    """Resolve the selected mode's fit cache, or return None to build it.

    Synthetic mode consults only its local namespace. Real mode tries the
    manifest before this machine's earlier refit; only ``DatasetUnavailable``
    falls through to that local door. Manifest/configuration faults still raise.
    """
    if SYNTHETIC:
        # See the DAPI sibling: consulting the manifest here returned the real
        # pinned artifact and published it under the synthetic identity. Read
        # only the synthetic namespace, and read the SAME names fit_all_channels
        # writes — this previously wrote synthetic_*.zarr.zip and read back the
        # real names, so the synthetic cache was write-only and every run refit.
        if RECOMPUTE:
            return None
        return load_local_fit_gsplats(DEMO_NAME, SYNTHETIC_GSPLATS_FILES)

    try:
        precomputed = load_dataset_gsplats(
            DEMO_NAME,
            GSPLATS_FILES,
            recompute=RECOMPUTE,
        )
    except DatasetUnavailable as exc:
        aprint(f"Manifest fetch unavailable ({exc}).")
        precomputed = None
    if precomputed is None and not RECOMPUTE:
        # A fit this machine built earlier, in its own namespace — checked
        # BEFORE refitting, which is what makes the refit one-time.
        precomputed = load_local_fit_gsplats(DEMO_NAME, GSPLATS_FILES)
    return precomputed


def acquire_volumes():
    """Return ``(volumes, source_dtype)`` from IDR, or synthesize under --synthetic.

    The one place the choice is made, so `main` carries no branch for it.
    """
    return synthesize_channel_volumes() if SYNTHETIC else load_multichannel_data()


def announce_data_provenance() -> None:
    """Say, up front, whether this run is real microscopy or a stand-in."""
    if SYNTHETIC:
        aprint("⚠ SYNTHETIC MODE — procedurally generated stand-in channels.")
        aprint("  This is NOT microscopy and the scene carries no citation.")
    else:
        aprint("Per-channel fitting + colormap layers + Web visualization")


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: Multi-Channel Mouse Blastocyst (Lamin B1 + DAPI)")
    aprint("=" * 70)
    announce_data_provenance()
    aprint("")

    # Determine output path
    output_path = get_demos_output_dir() / f"{SCENE_STEM}.luxar.zarr"

    # Serve only mode
    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
            return
        else:
            aprint(f"Scene not found: {output_path}")
            aprint("Run without --serve-only to generate first")
            return

    # Try the manifest-driven fetch (checksum-verified cache -> in-repo -> Zenodo)
    precomputed = resolve_gsplats()

    volumes = None

    if precomputed is not None:
        gsplats_list = precomputed
    else:
        # --recompute path (or no data to be had): download raw data, fit from
        # scratch, and cache the fits in the local-fit namespace.
        warn_if_no_cuda_gpu()
        volumes, source_dtype = acquire_volumes()

        if len(volumes) < 2:
            aprint("Error: Need at least 2 channels for this demo")
            return

        gsplats_list = fit_all_channels(volumes, source_dtype=source_dtype)

    # Optional round-trip visualisation
    if SHOW_ROUNDTRIP:
        if volumes is not None:
            show_roundtrip_comparison(
                volumes,
                gsplats_list,
                [c["name"] for c in CHANNELS],
                device=DEVICE,
            )
        else:
            aprint(
                "Cannot show round-trip: original volumes not available "
                "(loaded from precomputed cache). Re-run with --recompute."
            )

    # Create scene (centering + intensity scaling happen inside)
    scene_path = create_luxar_scene(gsplats_list, output_path)

    # Summary (only if volumes were loaded)
    if volumes is not None:
        total_splats = sum(len(g.amplitudes) for g in gsplats_list)
        total_voxels = sum(v.size for v in volumes)
        volume_bytes = total_voxels * 4  # float32
        # 11 floats per splat (no per-splat color, using colormaps)
        splats_bytes = total_splats * 11 * 4
        compression = volume_bytes / splats_bytes

        aprint("\n" + "=" * 70)
        aprint("Multi-Channel Compression Summary")
        aprint("=" * 70)
        aprint(f"Channels: {len(volumes)}")
        aprint(f"Total voxels: {total_voxels:,}")
        aprint(f"Total splats: {total_splats:,}")
        aprint(f"Raw size: {volume_bytes / 1024 / 1024:.2f} MB")
        aprint(f"Splat size: {splats_bytes / 1024:.2f} KB")
        aprint(f"Compression ratio: {compression:.1f}:1")
        aprint("=" * 70)

    # Open in napari for visual comparison
    if not NO_NAPARI and volumes is not None:
        view_with_napari(volumes, gsplats_list, CHANNELS[: len(gsplats_list)])

    # Launch viewer
    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer in 2 seconds...")
        time.sleep(2)
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
