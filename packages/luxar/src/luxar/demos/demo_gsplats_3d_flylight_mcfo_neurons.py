#!/usr/bin/env python3
"""GSplats Demo: MCFO Fly Brain Neurons in a Volume-Rendered Neuropil

Labelled *Drosophila* neurons — long, thin, widely branching — threaded through
the brain they live in, as a single volume-rendered Gaussian splat cloud.

================================================================================
WHAT THIS DEMO IS FOR
================================================================================

Two things, and they pull in opposite directions:

1. **Thin filaments are the hard case for a Gaussian basis.** Most microscopy
   gsplat demos here fit *blobby* content — nuclei, cells, tissue. MCFO neurons
   are thin (single neurites approach the resolution limit), long-range (one
   arbor spans the brain), sparse (~0.1% of voxels), and interwoven. That is
   what makes FISBe a benchmark dataset.

2. **The neuropil is the showcase for volume rendering.** The counterstained
   brain is a dense, semi-transparent medium. Under ``blending_mode=
   "volumetric"`` (emission-absorption, Max 1995) it composites front-to-back
   with real occlusion: the brain reads as a solid body, and neurites genuinely
   pass behind it and are dimmed by it, rather than glowing through as they
   would under additive. Sparse filaments inside a dense medium, in one node,
   is exactly the case that separates volumetric from additive compositing.

--------------------------------------------------------------------------------
1. ``gsplat cal`` CANNOT calibrate K here — the data is noise-free
--------------------------------------------------------------------------------

The blind-spot cross-validation behind ``luxar gsplat cal`` needs image noise
to locate a held-out peak. FISBe's raw has effectively none: the Janelia
Workstation stitches and distortion-corrects the tiles, which scrubs the pixel
noise even though values remain 12-bit. A sweep returns
``curve_type=signal_limited``, ``still_climbing=True``, sigma_hat ~1e-10 — its
"K*" is just the top of whatever grid you supplied. Luxar warns and points at
``--k-star-metric gain``; we drove K off a measured quality curve instead.

--------------------------------------------------------------------------------
2. GLOBAL PSNR IS THE WRONG NUMBER — score the foreground
--------------------------------------------------------------------------------

Only ~0.2% of the raw volume's energy lies inside the annotated neurons (41%
after floor suppression), so global PSNR mostly scores empty space. Measured on
the default sample, ``fg`` restricted to the ground-truth instance mask:

        K   global     fg   energy
   30,000   42.29   24.70    0.51x   <- neurites break into beads
  100,000   43.58   27.33    0.69x
  300,000   45.53   28.54    0.76x
  600,000   46.37   28.91    0.78x   <- continuity recovers
1,200,000   47.10   29.63    0.81x

A ~17 dB global-to-foreground gap. At K=30,000 the fit reads 42 dB globally
while dropping *half* the neurite brightness, and the failure is qualitative:
on a MIP the neurites break into disconnected beads where the original is a
continuous process. No global metric flags that; only looking does.

--------------------------------------------------------------------------------
3. SEEDS ARE NOT THE SPLAT COUNT — ``cull_retention`` is
--------------------------------------------------------------------------------

``fit_gaussian_splats`` optimises a fixed pool of ``seeds`` splats, then culls
by cumulative amplitude mass at the end (``cull_retention``, default 0.95). On
floor-suppressed data a handful of splats carry most of the mass, so the
default throws away nearly everything: 1.2M seeds settle at ~19K splats.

The trap is fixing that by *lowering seeds*. Measured, same floor and volume:

  1.2M seeds, retention 0.95   ->   18,623 splats   fg 27.60   energy 0.81x
  128K seeds, retention 1.0    ->  128,000 splats   fg 25.97   energy 0.78x  (!)
  1.2M seeds, retention 0.99   ->   26,238 splats   fg 26.79   energy 0.81x
  1.2M seeds, retention 0.999  ->   32,709 splats   fg 27.80   energy 0.84x  <- used

**128,000 splats scored worse than 18,623**, and its axon was visibly *more*
beaded. Seeds set the optimiser's search pool; retention sets the output size.
Shrinking seeds to hit a target count throws away the search that makes the
surviving splats good. Keep seeds high, raise retention.

--------------------------------------------------------------------------------
4. ONE NODE, and alpha is OPTICAL DEPTH
--------------------------------------------------------------------------------

Neurons and neuropil occupy the same volume. As two nodes there is no correct
draw order — whichever draws first occludes the other — so they are merged into
a single node with per-splat RGBA and depth-sorted together. Alpha keeps them
distinguishable: the neuropil gets a low per-splat alpha, the neurons a high
one.

Under volumetric compositing alpha is *optical depth*, and it accumulates: with
~356K neuropil splats spanning ~350 um, alpha 0.12 makes the brain effectively
opaque and buries interior neurites. That is the intended look here — a solid
body with neurons emerging from it. Lower ``NEUROPIL_ALPHA`` toward ~0.01 for a
translucent haze with every neurite visible; both are one constant apart and
worth trying.

The cost of one node: the Layers panel can no longer fade the neuropil
independently, because there is no second layer. That trade is deliberate.

--------------------------------------------------------------------------------
5. FIT THE COMPOSITE ONCE, THEN COLOUR
--------------------------------------------------------------------------------

MCFO is a *stochastic* multicolour label: a neuron's colour is a fixed ratio of
the three signal channels **at the same voxels**. Fitting each channel
separately produces splat sets that do not co-locate, and the composite becomes
candy-stripe — adjacent red, green and blue splats along a single axon that
should be one hue. So: fit the per-voxel channel maximum once, then read each
splat's colour from the channels at its own centre, balancing the channels by
their own robust maxima first (unbalanced, the brightest channel wins 96% of
splats and the whole brain reads red).

DATA SOURCE & CITATIONS:
========================

Neurons: FISBe (FlyLight Instance Segmentation Benchmark)
         Zenodo 10875063 -- https://doi.org/10.5281/zenodo.10875063
         Sample VT047848-20171020_66_I3 (``completely`` split, train)
         (C, Z, Y, X) = (3, 390, 1058, 907), uint16, 12-bit valued

Neuropil: Janelia FlyLight Gen1 MCFO, ``janelia-flylight-imagery`` on S3.
         FISBe distributes only the three signal channels; this sample's
         ``channel_spec`` is ``sssr`` -- three **s**ignal plus one
         **r**eference channel. That reference (neuropil counterstain) is what
         makes the image read as neurons *inside a brain*, and it exists only
         in the FlyLight release.

Voxel:   0.44 um isotropic, confirmed two ways -- FISBe paper Sec. 3, and the
         H5J's own root attrs (``voxel_size``, ``unit='micron'``)
Optics:  Zeiss LSM 710/780 confocal, Plan-Apochromat 40x/1.3 Oil DIC M27
Genotype: VT047848 BJD_118E08_AE_01 (female)
License: CC BY 4.0 (both sources)

Where the data comes from (both downloads are automatic):

  FISBe dataset page   https://kainmueller-lab.github.io/fisbe
  FISBe archive        https://doi.org/10.5281/zenodo.10875063
  FlyLight Gen1 MCFO   https://gen1mcfo.janelia.org
  FlyLight imagery     s3://janelia-flylight-imagery  (anonymous HTTP works)
  FlyLight project     https://www.janelia.org/project-team/flylight

How to Cite:
------------
If you use this data, cite all three. The first covers the benchmark and its
annotations; the second the imagery; the third the driver line.

  Mais L, Hirsch P, Managan C, Kandarpa R, Rumberger JL, Reinke A,
  Maier-Hein L, Ihrke G, Kainmueller D. "FISBe: A real-world benchmark
  dataset for instance segmentation of long-range thin filamentous
  structures." CVPR 2024.
  https://arxiv.org/abs/2404.00130

  Meissner GW, et al. "A searchable image resource of Drosophila GAL4
  driver expression patterns with single neuron resolution."
  eLife (2023) 12:e80660.
  https://doi.org/10.7554/eLife.80660

  Tirian L, Dickson BJ. "The VT GAL4, LexA, and split-GAL4 driver line
  collections for targeted expression in the Drosophila nervous system."
  bioRxiv (2017).
  https://doi.org/10.1101/198648

Credit the FlyLight Project Team, Janelia Research Campus, HHMI. Both
sources are CC BY 4.0, which requires attribution and that changes be
indicated — this demo fits splats to the imagery, which is a change.

WORKFLOW:
=========

1. **Fetch** one FISBe sample (~415 MB) from the 7.1 GB Zenodo archive by HTTP
   range request -- the archive is never downloaded whole
2. **Fetch + decode** the reference channel from the FlyLight H5J (~58 MB;
   HEVC streams inside HDF5, needs ffmpeg -- skipped gracefully if absent)
3. **Fit** the MCFO composite (neurons) and the reference channel (neuropil)
4. **Colour** each neuron splat from the three channels at its own centre
5. **Merge** into one node with per-splat RGBA and render volumetrically

USAGE:
======
    python demo_gsplats_3d_flylight_mcfo_neurons.py [--recompute] [--no-serve] [--serve-only]

Options:
    --recompute:      Force re-fetch and re-fit from scratch
    --no-serve:       Generate scene without launching viewer
    --serve-only:     Just serve a previously generated scene
    --sample=NAME:    Fit a different FISBe sample from the 'completely' split
    --no-neuropil:    Skip the reference channel (neurons on black)

Output:
    - Scene saved to: datasets/demos/gsplats_3d_flylight_mcfo_neurons.luxar.zarr
    - Automatically opens in your browser on the demo's own derived port
"""

DEMO_META = {
    "key": "gsplats_3d_flylight_mcfo_neurons",
    "title": "3D FlyLight MCFO Neurons",
    "description": "MCFO fly brain neurons in a volume-rendered neuropil, as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        # ~415 MB range-extracted from the FISBe archive (never fetched whole)
        # plus a ~58 MB H5J for the reference channel.
        "download_mb": 473,
        "compute": "heavy",
        # Redistributable (CC BY 4.0) but not hosted by us yet, so a first run
        # fetches and refits — which needs a GPU.
        "gpu": "required",
        "local_data": None,
    },
    "caches": ["gsplats_flylight_mcfo"],
    "outputs": ["gsplats_3d_flylight_mcfo_neurons"],
}

import hashlib
import importlib.util
import io
import ntpath
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import zarr
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    launch_viewer,
    parse_demo_flags,
    require_module,
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

DEFAULT_SAMPLE = "VT047848-20171020_66_I3"
SAMPLE_MEMBER_PREFIX = "completely/train"

# The FlyLight H5J carrying the reference (neuropil) channel FISBe drops.
FLYLIGHT_BUCKET = "https://janelia-flylight-imagery.s3.amazonaws.com"
FLYLIGHT_H5J = {
    DEFAULT_SAMPLE: (
        "Gen1 MCFO/VT047848/"
        "VT047848-20171020_66_I3-f-40x-brain-GAL4-unaligned_stack.h5j"
    ),
}

VOXEL_SIZE_ZYX = (0.44, 0.44, 0.44)

# Neurons. SEEDS is the optimiser's pool, CULL_RETENTION decides how much of it
# survives — see finding 3. Do NOT lower SEEDS to shrink the output.
SEEDS = 1_200_000
CULL_RETENTION = 0.999
FLOOR = "p99"

# Neuropil. Fewer splats than the neurons need, because it is a smooth medium.
NEUROPIL_SEEDS = 600_000
NEUROPIL_ALPHA = 0.12  # optical depth per splat — see finding 4
NEUROPIL_AMP = 0.6
NEUROPIL_RGB = (0.15, 0.25, 1.0)

COLOR_BALANCE_PERCENTILE = 99.99

CACHE_DIR = Path.home() / ".cache" / "luxar" / "gsplats_flylight_mcfo"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
NO_NEUROPIL = "--no-neuropil" in sys.argv

# FISBe sample names are plain basenames (``VT047848-20171020_66_I3``,
# ``JRC_SS04989-20160318_24_B1``). Pinning that shape keeps --sample out of the
# path-traversal business: the value lands in cache paths, the archive member
# prefix and the output filename, so a ``../`` or an absolute path would let it
# read and write outside the demo's directories — and --serve-only would then
# happily serve whatever .luxar.zarr it pointed at.
_SAMPLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_sample_name(sample: str) -> str:
    """Return ``sample`` if it is a bare FISBe sample name, else raise."""
    if not _SAMPLE_RE.match(sample):
        raise ValueError(
            f"Invalid --sample {sample!r}: expected a bare FISBe sample name "
            "matching [A-Za-z0-9][A-Za-z0-9_-]* (no path separators, no '..'). "
            f"See {SAMPLE_LIST_URL} for valid names."
        )
    return sample


SAMPLE = DEFAULT_SAMPLE
for _arg in sys.argv:
    if _arg.startswith("--sample="):
        SAMPLE = validate_sample_name(_arg.split("=", 1)[1])

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
        """Wrap ``url`` as a seekable file of ``size`` bytes served by ranges."""
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
        """This file is read-only, and readable."""
        return True

    def seekable(self) -> bool:
        """Seeking is what makes remote zip access possible."""
        return True

    def tell(self) -> int:
        """Return the current byte offset."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the read position, clamping to the bounds of the resource."""
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
        """Read ``size`` bytes (or to EOF), fetching blocks as needed."""
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

    def readinto(self, b) -> int:
        """Read into a pre-allocated buffer (the ``RawIOBase`` contract)."""
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def _block_for(self, pos: int) -> bytes:
        """Return the cached block covering ``pos``, range-fetching it if needed."""
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


def _safe_extract_path(root: Path, rel: str, member: str) -> Path:
    """Resolve ``rel`` under ``root``, refusing anything that escapes it.

    Archive member names are attacker-controlled in the general case, and a
    member spelled ``../../…`` (or with an absolute path) would otherwise be
    written outside the cache directory — the classic zip-slip. Refuse rather
    than sanitise, so a malformed archive is loud instead of silently partial.
    """
    if PurePosixPath(rel).is_absolute() or ntpath.isabs(rel):
        raise RuntimeError(f"Refusing absolute path in archive member: {member!r}")
    dest = (root / rel).resolve()
    if dest != root and root not in dest.parents:
        raise RuntimeError(
            f"Refusing archive member that escapes the extraction root: {member!r}"
        )
    return dest


def _extract_members(zf, members, member_prefix: str, tmp: Path) -> None:
    """Write every archive member below ``member_prefix`` into ``tmp``."""
    root = tmp.resolve()
    for i, name in enumerate(members):
        rel = name[len(member_prefix) :]
        if not rel:
            continue
        dest = _safe_extract_path(root, rel, name)
        if name.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src:
            dest.write_bytes(src.read())
        if i % 200 == 0:
            aprint(f"  {i}/{len(members)} members")


def _install_store(tmp: Path, target: Path) -> None:
    """Move a freshly extracted store into place, preserving any existing one.

    Renaming onto a populated directory raises ENOTEMPTY, so an existing store
    is moved aside first — and restored if installing the new one fails.
    Deleting it unconditionally would lose BOTH copies on failure, leaving no
    usable sample at all.
    """
    if not target.exists():
        tmp.rename(target)
        return

    stale = target.with_suffix(".zarr.stale")
    if stale.exists():
        shutil.rmtree(stale)
    target.rename(stale)
    try:
        tmp.rename(target)
    except BaseException:
        if not target.exists():
            stale.rename(target)
        raise
    shutil.rmtree(stale, ignore_errors=True)


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

            # A previous interrupted run can leave a stale partial directory,
            # and --recompute reaches here with `target` already populated.
            tmp = target.with_suffix(".zarr.partial")
            if tmp.exists():
                shutil.rmtree(tmp)

            _extract_members(zf, members, member_prefix, tmp)
            _install_store(tmp, target)
        finally:
            zf.close()

    aprint(f"Extracted to {target}")
    return target


# =============================================================================
# Reference (neuropil) channel — FlyLight H5J
# =============================================================================


def decode_h5j_channel(h5j_path: Path, channel: int) -> np.ndarray:
    """Decode one channel of an H5J stack to a (Z, Y, X) uint8 volume.

    H5J stores each channel as an HEVC video stream inside a 1-D uint8 HDF5
    dataset. Frames are padded up to the codec's block size (``pad_right`` /
    ``pad_bottom``), so decoded frames must be cropped back to the stated size.
    """
    h5py = require_module("h5py")

    with h5py.File(h5j_path, "r") as f:
        grp = f["Channels"]
        w = int(grp.attrs["width"][0])
        h = int(grp.attrs["height"][0])
        n = int(grp.attrs["frames"][0])
        pad_r = int(grp.attrs["pad_right"][0])
        pad_b = int(grp.attrs["pad_bottom"][0])
        spec = f.attrs["channel_spec"].decode()
        blob = grp[f"Channel_{channel}"][:].tobytes()

    ew, eh = w + pad_r, h + pad_b
    aprint(f"channel {channel} of spec {spec!r}: {w}x{h}x{n} (encoded {ew}x{eh})")

    stream = h5j_path.with_suffix(f".ch{channel}.hevc")
    stream.write_bytes(blob)
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, path from our cache
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(stream),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:500]}")
        raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    finally:
        stream.unlink(missing_ok=True)

    # Refuse a short or ragged decode rather than silently returning a
    # thinner volume: the neuropil would then span a different physical extent
    # from the neurons and the merged scene would be misregistered — a much
    # harder thing to notice than an exception here.
    expected = n * ew * eh
    if raw.size != expected:
        raise RuntimeError(
            f"H5J channel {channel} decoded to {raw.size} bytes, expected "
            f"{expected} ({n} frames of {ew}x{eh}). The stream is truncated or "
            "ffmpeg dropped frames; the neuropil would be misregistered "
            "against the neurons."
        )

    vol = raw.reshape(n, eh, ew)[:, :h, :w]
    aprint(f"  decoded {n} frames -> {vol.shape}, mean={vol.mean():.2f}")
    return vol


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` via a temporary sibling + rename."""
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """``np.save`` to a temporary sibling, then rename into place.

    Saving through an open handle rather than a path: ``np.save`` appends
    ``.npy`` to any *name* that does not already end in it, so a temporary
    called ``vol.npy.part`` would silently be written as ``vol.npy.part.npy``
    and the rename would then fail on a missing file.
    """
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        np.save(fh, array)
    tmp.replace(path)


def fetch_neuropil(sample: str):
    """Fetch and decode the reference channel, or return None if unavailable.

    Missing ffmpeg (or an unmapped sample) degrades to a neurons-only scene
    rather than failing: the demo is still worth running without the neuropil,
    it just loses the volume-rendered context.
    """
    key = FLYLIGHT_H5J.get(sample)
    if key is None:
        aprint(f"No FlyLight H5J mapped for {sample}; skipping the neuropil.")
        return None
    # Check the decoded cache FIRST: loading it needs neither ffmpeg nor h5py,
    # so gating on those beforehand would throw away a perfectly good warm
    # cache and silently drop to a neurons-only scene.
    cached = CACHE_DIR / f"{sample}_neuropil.npy"
    if cached.exists() and not RECOMPUTE:
        aprint(f"Neuropil already decoded: {cached.name}")
        return np.load(cached)

    # Both remaining dependencies are optional, and BOTH must degrade the same
    # way: the documented fallback is a neurons-only scene, so a missing one
    # must not raise out of a demo that promises to keep going.
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (H5J stores the channel as HEVC)")
    if importlib.util.find_spec("h5py") is None:
        missing.append("h5py (pip install 'luxar[demos]')")
    if missing:
        aprint(
            f"Skipping the neuropil channel — missing {', '.join(missing)}. "
            "The neurons render fine without it; install the above for the "
            "volume-rendered brain."
        )
        return None

    h5j = CACHE_DIR / f"{sample}.h5j"
    if not h5j.exists() or RECOMPUTE:
        url = f"{FLYLIGHT_BUCKET}/{requests.utils.quote(key)}"
        with asection("Fetching FlyLight H5J (reference channel)"):
            aprint(url)
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            # Publish atomically: an interrupted write would otherwise leave a
            # truncated file that every later run trusts and fails to decode,
            # recoverable only by knowing to pass --recompute.
            _atomic_write(h5j, resp.content)
            aprint(f"  {len(resp.content) / 1e6:.0f} MB")

    # ``channel_spec`` is 'sssr': three signal channels then the reference.
    vol = decode_h5j_channel(h5j, 3)
    _atomic_save_npy(cached, vol)
    return vol


# =============================================================================
# Data Loading
# =============================================================================


def load_fisbe_sample(sample: str):
    """Load the three MCFO signal channels and their per-voxel maximum."""
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


def fit_cache_key(seeds: int, floor, retention: float) -> str:
    """Short digest of everything that changes a fit's result.

    Cache filenames keyed only by sample would silently reuse an incompatible
    fit whenever one of the tuning constants above is edited — and those
    constants are exactly what this demo invites you to tune. Folding them into
    the name means changing one produces a different file, so the refit is
    automatic rather than dependent on remembering ``--recompute``.
    """
    payload = f"v1|{seeds}|{floor}|{retention}|{tuple(VOXEL_SIZE_ZYX)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _fit_cache_path(component: str, seeds: int, floor, retention: float) -> Path:
    """Cache path for one fitted component, keyed by its fit parameters."""
    key = fit_cache_key(seeds, floor, retention)
    return CACHE_DIR / f"{SAMPLE}_{component}_{key}.gsplats.zarr.zip"


def _device():
    """Resolve (and memoise) the fitting device."""
    global DEVICE
    if DEVICE is None:
        from luxar.demos import detect_device

        DEVICE = detect_device()
    return DEVICE


def fit_volume(
    volume, cache_file: Path, seeds: int, floor, retention: float, label: str
):
    """Fit one volume to splats, using the cache when present."""
    if cache_file.exists() and not RECOMPUTE:
        aprint(f"Loading cached {label} fit")
        try:
            result = GSplatData.load(cache_file, include_stats=False)
            aprint(f"  Loaded {len(result.amplitudes):,} cached splats")
            return result
        except Exception as exc:
            aprint(f"  Cache load failed: {exc}, re-fitting...")

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"Fitting {label} (seeds={seeds:,}, floor={floor}, retention={retention})")
    result = fit_gaussian_splats(
        volume,
        seeds=seeds,
        floor=floor,
        cull_retention=retention,
        device=_device(),
        verbose=True,
        voxel_size=VOXEL_SIZE_ZYX,
    )
    aprint(f"  Fitted {len(result.amplitudes):,} splats (from {seeds:,} seeds)")

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
        # Balance channels before compositing — see finding 5.
        gain = float(np.percentile(vol, COLOR_BALANCE_PERCENTILE)) or 1.0
        triplet[:, c] = vol[idx[:, 0], idx[:, 1], idx[:, 2]] / gain
        aprint(f"  ch{c} balance gain (p{COLOR_BALANCE_PERCENTILE}) = {gain:.5f}")

    # Hue only: brightness is carried by the splat amplitude, so normalising
    # each splat by its own strongest channel keeps the MCFO colour ratio
    # without double-counting intensity.
    peak = triplet.max(axis=1, keepdims=True)
    rgb = np.clip(
        np.divide(triplet, peak, out=np.zeros_like(triplet), where=peak > 0), 0.0, 1.0
    )
    dominant = triplet.argmax(axis=1)
    aprint(
        "  dominant channel: "
        + ", ".join(f"ch{c}={100 * (dominant == c).mean():.1f}%" for c in range(3))
    )
    return rgb


def merge_for_render(neurons, neuron_rgb, neuropil):
    """Merge neurons and neuropil into one splat set with per-splat RGBA.

    Returns ``(centers, amplitudes, cholesky, rgba)``. The neuropil is emitted
    first purely for readability; ordering within a node is resolved by the
    renderer's depth sort, which is the whole reason for using one node.
    """
    n = len(neurons.amplitudes)
    neuron_rgba = np.concatenate(
        [neuron_rgb, np.ones((n, 1), dtype=np.float32)], axis=1
    ).astype(np.float32)
    if neuropil is None:
        return (
            neurons.centers,
            neurons.amplitudes,
            neurons.cholesky_factors,
            neuron_rgba,
        )

    p = len(neuropil.amplitudes)
    neuropil_rgba = np.empty((p, 4), dtype=np.float32)
    neuropil_rgba[:, :3] = np.asarray(NEUROPIL_RGB, dtype=np.float32)
    neuropil_rgba[:, 3] = NEUROPIL_ALPHA

    return (
        np.concatenate([neuropil.centers, neurons.centers]).astype(np.float32),
        np.concatenate([neuropil.amplitudes * NEUROPIL_AMP, neurons.amplitudes]).astype(
            np.float32
        ),
        np.concatenate([neuropil.cholesky_factors, neurons.cholesky_factors]).astype(
            np.float32
        ),
        np.concatenate([neuropil_rgba, neuron_rgba]).astype(np.float32),
    )


# =============================================================================
# Scene Creation
# =============================================================================


def create_luxar_scene(centers, amplitudes, cholesky, rgba, output_path=None):
    """Create the 3D scene: one volume-rendered node with per-splat RGBA."""
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
                # Cinematic mode (ACES + a subtle wide bloom, detector noise,
                # vignette, chromatic lens distortion, 35mm FOV) suits this
                # scene: it IS a microscope image, so film-grain and lens
                # character read as photographic rather than as decoration, and
                # the bloom gives the bright neurites the glow they have in the
                # raw data. ACES is set explicitly too — cinematic mode selects
                # it, but stating it keeps the intent legible if the preset ever
                # changes.
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    cinematic_mode=True,
                ),
            )

            scene.attrs["title"] = "GSplats: MCFO Fly Brain Neurons"
            scene.attrs["sample"] = SAMPLE
            scene.attrs["description"] = f"""
MCFO Fly Brain Neurons in a Volume-Rendered Neuropil
====================================================

Labelled Drosophila neurons — long, thin, widely branching — threaded through
the brain they live in, as one volume-rendered Gaussian splat cloud.

The neuropil counterstain is a dense semi-transparent medium; under volumetric
(emission-absorption) compositing it occludes front-to-back, so the brain reads
as a solid body and neurites genuinely pass behind it. Neurons and neuropil
share ONE node with per-splat RGBA — as two nodes covering the same volume
there would be no correct draw order.

Data Source:
  - Neurons: FISBe v1.0, Zenodo 10.5281/zenodo.10875063 (CC BY 4.0)
    Sample {SAMPLE}, 'completely' split
  - Neuropil: Janelia FlyLight Gen1 MCFO reference channel (CC BY 4.0),
    which FISBe does not distribute
  - Zeiss LSM 710/780 confocal, 40x/1.3 Oil, 0.44 um isotropic

Cite: Mais et al. (FISBe, CVPR 2024); Meissner et al. (eLife 2023
12:e80660); Tirian & Dickson (2017). Credit the FlyLight Project Team,
Janelia Research Campus, HHMI.

Controls:
  - Mouse drag to rotate, scroll to zoom, right-click drag to pan
            """

            centroid = (centers.T @ amplitudes) / max(float(amplitudes.sum()), 1e-9)
            aprint(f"Centroid: {centroid}")

            scene.add_gsplats(
                name="gsplats_mcfo",
                centers=centers - centroid,
                amplitudes=amplitudes,
                cholesky_factors=cholesky,
                colors=rgba,
                dim_order=["z", "y", "x"],
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
                layer=True,
            )
            aprint(f"Added {len(amplitudes):,} splats with per-splat RGBA")

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
    aprint("GSplats Demo: MCFO Fly Brain Neurons (FlyLight / FISBe)")
    aprint("=" * 70)
    aprint(f"Sample: {SAMPLE}")
    aprint("Thin filaments + a volume-rendered neuropil, in one splat cloud")
    aprint("")

    # A non-default --sample writes beside the default scene rather than
    # silently replacing it, so the two are never confused for one another.
    stem = "gsplats_3d_flylight_mcfo_neurons"
    if SAMPLE != DEFAULT_SAMPLE:
        stem = f"{stem}_{SAMPLE}"
    output_path = get_demos_output_dir() / f"{stem}.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    warn_if_no_cuda_gpu()

    # Colouring needs the individual channels, so the sample is loaded even
    # when the fit itself is cached.
    channels, combined = load_fisbe_sample(SAMPLE)

    with asection("Fitting neurons"):
        neurons = fit_volume(
            combined,
            _fit_cache_path("neurons", SEEDS, FLOOR, CULL_RETENTION),
            SEEDS,
            FLOOR,
            CULL_RETENTION,
            "neurons",
        )

    neuropil = None
    if not NO_NEUROPIL:
        with asection("Neuropil (reference channel)"):
            ref = fetch_neuropil(SAMPLE)
            if ref is not None:
                neuropil = fit_volume(
                    ref.astype(np.float32) / 255.0,
                    _fit_cache_path("neuropil", NEUROPIL_SEEDS, "auto", 0.95),
                    NEUROPIL_SEEDS,
                    "auto",
                    0.95,
                    "neuropil",
                )

    with asection("Colouring neurons from MCFO channels"):
        neuron_rgb = splat_colors(neurons.centers, channels)

    centers, amps, chol, rgba = merge_for_render(neurons, neuron_rgb, neuropil)

    with asection("Summary"):
        aprint(f"  Neurons:  {len(neurons.amplitudes):,} splats")
        if neuropil is not None:
            aprint(f"  Neuropil: {len(neuropil.amplitudes):,} splats")
        else:
            aprint("  Neuropil: skipped")
        aprint(f"  Total:    {len(amps):,} splats")
        # 11 floats per splat: 3 centers + 6 Cholesky + amplitude + pad.
        aprint(f"  Compression: {(combined.size * 4) / (len(amps) * 11 * 4):.0f}:1")

    scene_path = create_luxar_scene(centers, amps, chol, rgba, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
