#!/usr/bin/env python3
"""GSplats Demo: 3-Colour MCFO Fly Brain Neurons (FlyLight / FISBe)

Visualises a three-channel MCFO (MultiColor FlpOut) confocal volume of a
*Drosophila* central brain as Gaussian splats — long, thin, widely branching
neurons traced through a whole brain.

================================================================================
LONG-RANGE THIN FILAMENTS — THE HARD CASE FOR A GAUSSIAN BASIS
================================================================================

Most microscopy gsplat demos in this repo fit *blobby* content: nuclei, cells,
tissue. This one is deliberately the opposite. MCFO neurons are:

  - **Thin** — single neurites approach the optical resolution limit
  - **Long-range** — a single arbor spans the entire brain
  - **Sparse** — roughly 0.1% of voxels carry signal
  - **Interwoven** — several neurons pass through the same neuropil

That combination is what makes FISBe a benchmark dataset, and it makes an
honest stress test for an anisotropic Gaussian basis: a Gaussian elongated
along a neurite is a very good local model of a filament, so the fit
compresses hard — but only if the splats land on neurons rather than on the
neuropil haze they float in.

Three things about this dataset are easy to get wrong, and all three were
measured rather than assumed. They are worth reading before adapting this
demo to other MCFO data.

--------------------------------------------------------------------------------
1. ``gsplat cal`` CANNOT calibrate K here — the data is noise-free
--------------------------------------------------------------------------------

The blind-spot cross-validation protocol behind ``luxar gsplat cal`` needs
image noise to locate a held-out peak. FISBe's raw has effectively none: the
Janelia Workstation stitches and distortion-corrects the tiles, which scrubs
the pixel noise even though values remain 12-bit. A sweep on this data returns
``curve_type=signal_limited``, ``still_climbing=True`` and sigma_hat ~1e-10, so
its "K*" is simply the top of whatever grid you supplied. Luxar warns about
this and points at ``--k-star-metric gain``; we drove K off a measured quality
curve instead.

--------------------------------------------------------------------------------
2. GLOBAL PSNR IS THE WRONG NUMBER — score the foreground
--------------------------------------------------------------------------------

Only ~0.2% of this volume's energy lies inside the annotated neurons, so global
PSNR mostly scores how well the fit reproduces empty space. Measured on
channel 0 (390x1058x907, RTX PRO 6000), where ``fg`` is PSNR restricted to the
ground-truth instance mask and ``energy`` is the share of foreground brightness
the reconstruction reproduces:

        K   global     fg   energy    fit
   30,000   42.29   24.70    0.51x   104s   <- neurites break into beads
  100,000   43.58   27.33    0.69x    84s
  300,000   45.53   28.54    0.76x   120s
  600,000   46.37   28.91    0.78x   181s   <- continuity recovers
1,200,000   47.10   29.63    0.81x   240s
2,400,000   47.49   30.46    0.84x   345s

A ~17 dB global-to-foreground gap. At K=30,000 the fit still reads 42 dB
globally while dropping *half* the neurite brightness, and the failure is
qualitative rather than a smooth loss of dB: on a MIP the neurites break into
disconnected beads where the original is a continuous process. No global
metric flags that; only looking does.

--------------------------------------------------------------------------------
3. FIT THE COMPOSITE ONCE, THEN COLOUR — do not fit three channels separately
--------------------------------------------------------------------------------

MCFO is a *stochastic* multicolour label: a neuron's colour is a fixed ratio of
the three channels **at the same voxels**. Fitting each channel independently
produces three splat sets that do not co-locate, and the composite becomes
candy-stripe — adjacent red, green and blue splats along a single axon that
should be one uniform hue. (This is a real thing that happened; it is visible
the moment you zoom to a single neurite.)

So: fit the per-voxel channel maximum once, which makes co-location structural,
then read each splat's colour from the three channels at its own centre. The
channels are balanced by their own robust maxima first — they are different
fluorophores with several-fold different gains, and compositing them raw makes
the brightest channel win 96% of splats and the whole brain read red.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  FISBe (FlyLight Instance Segmentation Benchmark)
         Zenodo record 10875063 -- https://doi.org/10.5281/zenodo.10875063
Sample:  VT047848-20171020_66_I3  (``completely`` split, train)
Origin:  Janelia FlyLight Gen1 MCFO collection
Shape:   (3, 390, 1058, 907) -- (C, Z, Y, X), uint16 (12-bit valued)
Voxel:   0.44 um isotropic  (the "40x Gen1" subset; FISBe paper Sec. 3)
Extent:  ~172 x 466 x 399 um  (Z x Y x X)
Optics:  Zeiss LSM 710/780 confocal, Plan-Apochromat 40x/1.3 Oil DIC M27
Genotype: VT047848  BJD_118E08_AE_01  (female)
License: CC BY 4.0

How to Cite:
------------
If you use this dataset, please cite all three:

  Mais, Hirsch, Managan, Kandarpa, Rumberger, Reinke, Maier-Hein, Ihrke,
  Kainmueller. "FISBe: A real-world benchmark dataset for instance
  segmentation of long-range thin filamentous structures." CVPR 2024.
  arXiv:2404.00130

  Meissner et al. "A searchable image resource of Drosophila GAL4 driver
  expression patterns with single neuron resolution."
  eLife (2023) 12:e80660

  Tirian & Dickson. "The VT GAL4, LexA, and split-GAL4 driver line
  collections for targeted expression in the Drosophila nervous system."
  bioRxiv (2017) doi:10.1101/198648

Credit the FlyLight Project Team, Janelia Research Campus, HHMI.

WORKFLOW:
=========

1. **Fetch** one sample (~415 MB) out of the 7.1 GB Zenodo archive using HTTP
   range requests -- the archive is never downloaded whole
2. **Combine** the three channels by per-voxel maximum (geometry)
3. **Fit** one splat set to the composite, with aggressive floor suppression
4. **Colour** each splat from the three channels at its own centre
5. **Visualise** in the Luxar web viewer

USAGE:
======
    python demo_gsplats_3d_flylight_mcfo_neurons.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:      Force re-fitting from scratch (fetch + GPU fitting)
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --sample=NAME:    Fit a different FISBe sample from the 'completely' split

Output:
    - Scene saved to: datasets/demos/gsplats_3d_flylight_mcfo_neurons.luxar.zarr
    - Automatically opens in your browser on the demo's own derived port
"""

DEMO_META = {
    "key": "gsplats_3d_flylight_mcfo_neurons",
    "title": "3D FlyLight MCFO Neurons",
    "description": "3-colour MCFO fly brain neurons — long thin filaments as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        # One FISBe sample is range-extracted from the Zenodo archive; the
        # 7.1 GB archive itself is never downloaded whole.
        "download_mb": 415,
        "compute": "heavy",
        # The source is redistributable (CC BY 4.0) but is not hosted by us
        # yet, so a first run fetches and refits — which needs a GPU.
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_flylight_mcfo"],
    "outputs": ["gsplats_3d_flylight_mcfo_neurons"],
}

import io
import sys
from pathlib import Path

import numpy as np
import requests
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    launch_viewer,
    parse_demo_flags,
    warn_if_no_cuda_gpu,
)
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

ZENODO_RECORD = "10875063"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
ARCHIVE_NAME = "fisbe_v1.0_completely.zip"
SAMPLE_LIST_URL = f"{ZENODO_BASE}/sample_list_per_split.txt?download=1"

# The hero sample: several neurons whose arbors span the whole central brain.
DEFAULT_SAMPLE = "VT047848-20171020_66_I3"
SAMPLE_MEMBER_PREFIX = "completely/train"

# Voxel spacing (Z, Y, X) in micrometres; FISBe "40x Gen1" is isotropic 0.44 um.
VOXEL_SIZE_ZYX = (0.44, 0.44, 0.44)

# Seed count for the single composite fit. This is a SEED count, not the final
# splat count: the optimiser prunes splats that end up carrying no mass, and
# with FLOOR below the great majority do. 1.2M seeds settles at ~19K splats on
# the default sample — the neurons, and almost nothing else.
SEEDS = 1_200_000

# Aggressive floor suppression is what makes this demo work at all.
#
# The default "auto" floor estimates a background level of ~0.0012 here, which
# is correct for a *pedestal* but leaves the neuropil autofluorescence intact —
# and that haze fills the brain silhouette. Fitting it produces ~1.1M splats
# that are overwhelmingly background, and the render is a solid saturated blob
# with the neurons buried inside it, in every blending mode (additive sums the
# haze along each ray; peak projection keeps it because it is genuinely bright).
#
# Signal here occupies ~0.1-0.25% of voxels, so subtracting the 99th percentile
# (~0.0337 on the default sample) cuts the haze and leaves the neurons.
FLOOR = "p99"

# Percentile used to balance the three detection channels before compositing
# colour. They are different fluorophores with several-fold different gains.
COLOR_BALANCE_PERCENTILE = 99.99

CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_flylight_mcfo"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

SAMPLE = DEFAULT_SAMPLE
for _arg in sys.argv:
    if _arg.startswith("--sample="):
        SAMPLE = _arg.split("=", 1)[1]

Arbol.max_depth = 6
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = None


# =============================================================================
# Remote zip access (HTTP range requests)
# =============================================================================


class _HttpRangeFile(io.RawIOBase):
    """A minimal seekable read-only file over an HTTP resource.

    ``zipfile.ZipFile`` only needs ``seek``/``read``/``tell``, and it seeks to
    the end of the file to find the central directory. Serving those seeks with
    HTTP range requests means one member can be extracted from a multi-gigabyte
    remote archive while transferring only that member plus the directory.
    """

    def __init__(self, url: str, size: int, session, chunk_size: int = 1 << 20):
        self._url = url
        self._size = size
        self._session = session
        self._pos = 0
        self._chunk_size = chunk_size
        # One-block read-ahead: zipfile does many small adjacent reads while
        # parsing the central directory, and without this each becomes its own
        # HTTP round trip.
        self._cache_start = 0
        self._cache = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""

        out = bytearray()
        while size > 0:
            block = self._block_for(self._pos)
            offset = self._pos - self._cache_start
            take = min(size, len(block) - offset)
            if take <= 0:  # pragma: no cover - defensive
                break
            out += block[offset : offset + take]
            self._pos += take
            size -= take
        return bytes(out)

    def readinto(self, b) -> int:  # noqa: D102 - RawIOBase contract
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def _block_for(self, pos: int) -> bytes:
        if self._cache and self._cache_start <= pos < self._cache_start + len(
            self._cache
        ):
            return self._cache
        start = pos
        end = min(start + self._chunk_size, self._size) - 1
        resp = self._session.get(
            self._url, headers={"Range": f"bytes={start}-{end}"}, timeout=60
        )
        if resp.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206 (partial content) from {self._url}, got "
                f"{resp.status_code}. The host has stopped honouring range "
                "requests, so a single sample can no longer be extracted "
                "without downloading the whole archive."
            )
        self._cache_start = start
        self._cache = resp.content
        return self._cache


def _open_remote_zip(url: str):
    """Open a remote zip for partial extraction. Returns ``(ZipFile, handle)``."""
    import zipfile

    session = requests.Session()

    # Probe with a one-byte ranged GET rather than trusting HEAD: Zenodo's HEAD
    # answers 200 with no ``Accept-Ranges`` header even though ranged GETs are
    # honoured, so a HEAD-based check would refuse a host that works fine. The
    # 206 also carries the total size in ``Content-Range``, so this is one
    # request instead of two.
    probe = session.get(
        url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60
    )
    probe.raise_for_status()
    content_range = probe.headers.get("Content-Range", "")
    if probe.status_code != 206 or "/" not in content_range:
        raise RuntimeError(
            f"{url} did not honour a byte-range request (status "
            f"{probe.status_code}), so extracting one sample would require "
            "downloading the full 7.1 GB archive. Download it manually and "
            f"extract the sample into {CACHE_DIR} as <sample>.zarr instead."
        )
    size = int(content_range.rsplit("/", 1)[1])

    handle = _HttpRangeFile(probe.url, size, session)
    return zipfile.ZipFile(io.BufferedReader(handle, buffer_size=1 << 20)), handle


def fetch_sample(sample: str) -> Path:
    """Range-extract one FISBe sample's zarr store into the demo cache."""
    target = CACHE_DIR / f"{sample}.zarr"
    # The store is renamed into place only once every member is written, so a
    # present target directory always means a complete extraction.
    if target.exists() and not RECOMPUTE:
        aprint(f"Sample already extracted: {target.name}")
        return target

    url = f"{ZENODO_BASE}/{ARCHIVE_NAME}?download=1"
    member_prefix = f"{SAMPLE_MEMBER_PREFIX}/{sample}.zarr/"

    with asection(f"Range-extracting {sample} from {ARCHIVE_NAME}"):
        aprint("The 7.1 GB archive is NOT downloaded whole — only this sample.")
        zf, _handle = _open_remote_zip(url)
        try:
            members = [n for n in zf.namelist() if n.startswith(member_prefix)]
            if not members:
                raise RuntimeError(
                    f"Sample {sample!r} not found in {ARCHIVE_NAME}. "
                    f"See {SAMPLE_LIST_URL} for valid names (this demo reads "
                    "the 'completely' split)."
                )
            payload = sum(zf.getinfo(n).compress_size for n in members)
            aprint(f"{len(members)} members, {payload / 1e6:.0f} MB compressed")

            tmp = target.with_suffix(".zarr.partial")
            for i, name in enumerate(members):
                rel = name[len(member_prefix) :]
                if not rel:
                    continue
                dest = tmp / rel
                if name.endswith("/"):
                    dest.mkdir(parents=True, exist_ok=True)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(name) as src:
                    dest.write_bytes(src.read())
                if i % 200 == 0:
                    aprint(f"  {i}/{len(members)} members")
            tmp.rename(target)
        finally:
            zf.close()

    aprint(f"Extracted to {target}")
    return target


# =============================================================================
# Data Loading
# =============================================================================


def load_fisbe_sample(sample: str):
    """Load the three MCFO signal channels of one FISBe sample.

    Returns:
        tuple: ``(channels, combined)`` where ``channels`` is a list of three
        (Z, Y, X) float32 volumes scaled to [0, 1] by a shared maximum, and
        ``combined`` is their per-voxel maximum (the fit target).
    """
    store_path = fetch_sample(sample)

    with asection(f"Loading {sample}"):
        raw = zarr.open(str(store_path), mode="r")["volumes"]["raw"]
        aprint(f"raw: shape={raw.shape} dtype={raw.dtype}")
        if raw.shape[0] != 3:
            raise RuntimeError(f"Expected 3 MCFO channels, got {raw.shape[0]}")

        channels = [np.asarray(raw[c]).astype(np.float32) for c in range(3)]
        shared_max = max(float(v.max()) for v in channels) or 1.0
        channels = [v / shared_max for v in channels]

        for c, v in enumerate(channels):
            aprint(f"  ch{c}: mean={v.mean():.6f} p99.9={np.percentile(v, 99.9):.5f}")

        combined = np.maximum(np.maximum(channels[0], channels[1]), channels[2])
        extent = tuple(round(n * s, 1) for n, s in zip(combined.shape, VOXEL_SIZE_ZYX))
        aprint(f"  composite mean={combined.mean():.6f}")
        aprint(f"  physical extent (Z, Y, X): {extent} um")
        return channels, combined


# =============================================================================
# Fitting and colouring
# =============================================================================


def fit_composite(combined, cache_file: Path):
    """Fit one splat set to the composite volume, using the cache when present."""
    if cache_file.exists() and not RECOMPUTE:
        aprint("Loading cached composite fit")
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats")
            return result
        except Exception as exc:
            aprint(f"  Cache load failed: {exc}, re-fitting...")

    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting composite (seeds={SEEDS:,}, floor={FLOOR})...")
    result = fit_gaussian_splats(
        combined,
        seeds=SEEDS,
        floor=FLOOR,
        device=DEVICE,
        verbose=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )
    aprint(f"  Fitted {len(result.amplitudes):,} splats (from {SEEDS:,} seeds)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    return result


def splat_colors(centers, channels, voxel_size=VOXEL_SIZE_ZYX):
    """Read each splat's MCFO colour out of the three channel volumes.

    Args:
        centers: (N, 3) splat centres in PHYSICAL units (Z, Y, X micrometres).
        channels: three (Z, Y, X) float32 volumes.
        voxel_size: (Z, Y, X) micrometres per voxel.

    Returns:
        (N, 3) float32 RGB in [0, 1].
    """
    shape = channels[0].shape
    # ``fit_gaussian_splats(..., voxel_size=...)`` returns centres in physical
    # units, so they must be divided back out before they can index the volume.
    # Indexing microns as voxels silently samples the wrong places (compressed
    # toward the origin by 1/voxel_size) and quietly corrupts every colour.
    idx = np.rint(centers / np.asarray(voxel_size, dtype=np.float32)).astype(np.int64)
    for d in range(3):
        np.clip(idx[:, d], 0, shape[d] - 1, out=idx[:, d])

    triplet = np.empty((len(centers), 3), dtype=np.float32)
    for c, vol in enumerate(channels):
        # Balance channels before compositing — see the module docstring.
        gain = float(np.percentile(vol, COLOR_BALANCE_PERCENTILE)) or 1.0
        triplet[:, c] = vol[idx[:, 0], idx[:, 1], idx[:, 2]] / gain
        aprint(f"  ch{c} balance gain (p{COLOR_BALANCE_PERCENTILE}) = {gain:.5f}")

    # Hue only: brightness is already carried by the splat amplitude, so
    # normalising each splat by its own strongest channel keeps the MCFO colour
    # ratio without double-counting intensity.
    peak = triplet.max(axis=1, keepdims=True)
    rgb = np.divide(triplet, peak, out=np.zeros_like(triplet), where=peak > 0)
    rgb = np.clip(rgb, 0.0, 1.0)

    dominant = triplet.argmax(axis=1)
    aprint(
        "  dominant channel: "
        + ", ".join(f"ch{c}={100 * (dominant == c).mean():.1f}%" for c in range(3))
    )
    return rgb


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(gsplats, rgb, output_path=None):
    """Create the 3D scene: one splat set carrying per-splat MCFO colour."""
    if output_path is None:
        output_path = (
            get_demos_output_dir() / "gsplats_3d_flylight_mcfo_neurons.luxar.zarr"
        )

    with asection("Creating 3D Luxar Scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="um", display=True),
                Dimension("y", unit="um", display=True),
                Dimension("z", unit="um", display=True),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )

            scene.attrs["title"] = "GSplats: 3-Colour MCFO Fly Brain Neurons"
            scene.attrs["description"] = """
3-Colour MCFO Fly Brain Neurons (FlyLight / FISBe)
==================================================

Long-range, thin, widely branching Drosophila neurons fitted as anisotropic
Gaussian splats — the hard case for a Gaussian basis, and the reason FISBe
exists as a benchmark.

One splat set is fitted to the per-voxel maximum of the three MCFO detection
channels, then each splat is coloured from those channels at its own centre.
MCFO is a stochastic label, so a neuron's colour is a ratio of the three
channels at the same voxels — fitting the channels separately would break that
co-location and stripe each axon.

Data Source:
  - FISBe v1.0, Zenodo 10.5281/zenodo.10875063 (CC BY 4.0)
  - Sample VT047848-20171020_66_I3, 'completely' split
  - Janelia FlyLight Gen1 MCFO collection
  - Zeiss LSM 710/780 confocal, 40x/1.3 Oil, 0.44 um isotropic

Cite: Mais et al. (FISBe, CVPR 2024); Meissner et al. (eLife 2023
12:e80660); Tirian & Dickson (2017). Credit the FlyLight Project Team,
Janelia Research Campus, HHMI.

Controls:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
  - Press L for the Layers panel
            """

            centroid = (gsplats.centers.T @ gsplats.amplitudes) / max(
                float(gsplats.amplitudes.sum()), 1e-9
            )
            aprint(f"Centroid: {centroid}")

            scene.add_gsplats(
                name="gsplats_mcfo",
                centers=gsplats.centers - centroid,
                amplitudes=gsplats.amplitudes,
                cholesky_factors=gsplats.cholesky_factors,
                colors=rgb,
                dim_order=["z", "y", "x"],
                opacity=1.0,
                # Additive: with the neuropil haze removed by FLOOR the scene is
                # a sparse set of bright filaments on black, which is exactly
                # the case additive compositing is for. (It is NOT safe before
                # floor suppression — see the FLOOR comment.)
                blending_mode="additive",
                layer=True,
            )
            aprint(f"Added {len(gsplats.amplitudes):,} splats with per-splat colour")

            scene.add_text(
                "MCFO Fly Brain Neurons",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Confocal • 0.44 μm isotropic • FISBe / FlyLight",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 3-Colour MCFO Fly Brain Neurons (FlyLight / FISBe)")
    aprint("=" * 70)
    aprint(f"Sample: {SAMPLE}")
    aprint("Long-range thin filamentous neurons as anisotropic Gaussian splats")
    aprint("")

    output_path = get_demos_output_dir() / "gsplats_3d_flylight_mcfo_neurons.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    warn_if_no_cuda_gpu()

    # Colouring needs the individual channels, so the sample is loaded even when
    # the fit itself is cached.
    channels, combined = load_fisbe_sample(SAMPLE)

    cache_file = CACHE_DIR / f"{SAMPLE}_composite.gsplats.zarr.zip"
    with asection("Fitting composite"):
        gsplats = fit_composite(combined, cache_file)

    with asection("Colouring splats from MCFO channels"):
        rgb = splat_colors(gsplats.centers, channels)

    with asection("Summary"):
        n = len(gsplats.amplitudes)
        voxels = combined.size
        aprint(f"  Splats: {n:,} (from {SEEDS:,} seeds, floor={FLOOR})")
        aprint(f"  Voxels: {voxels:,}")
        # 11 floats per splat: 3 centers + 6 Cholesky + amplitude + pad.
        aprint(f"  Compression: {(voxels * 4) / (n * 11 * 4):.0f}:1")

    scene_path = create_luxar_scene(gsplats, rgb, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
