#!/usr/bin/env python3
"""Self-Contained Demo: CytoSelf Protein Localization 3D UMAP

Visualizes ~114K per-image CytoSelf embeddings from the OpenCell dataset as a
3D UMAP point cloud. Each point is a single fluorescence microscopy crop of an
endogenously tagged protein, colored by subcellular localization or protein
identity.

Run behavior:
  First run: Download embeddings from Google Drive, compute 3D UMAP (~10-30 min)
  Subsequent runs: Load cached results instantly
  Interrupted runs: Every download is streamed into a sibling ".part" file and
    renamed onto its cache name only after the bytes are verified, so a killed
    run never leaves a truncated file behind. Image thumbnails are cached per
    source Image_data file, so a re-run only re-fetches and re-encodes the
    pieces that are missing.

Download size (exact, from Range probes of the Drive files; decimal GB/MB):
  185.8 GB with hover thumbnails, which are ON by default:
    Global_representation.npy   4.23 GB
    label.csv                   6.55 MB
    Label_data00..09.csv        64.7 MB total (3.95-8.74 MB each)
    Image_data00..09.npy        181.5 GB total (11.28-23.60 GB each)
  Pass --without-images for a ~4.24 GB run: the embeddings and label.csv only,
  since the Label_data CSVs are needed solely to place the thumbnails.
  Budget ~190 GB of free disk rather than 186: the downloads stay cached, and
  the encoded thumbnails — one archive per source file, the assembled bundle,
  and the staging copy the bundle is written through — sit alongside them.
  Plan on a 32 GB machine: the thumbnail pass reads one Image_data archive
  whole and the largest is 23.6 GB, with the selected crops, the thumbnails
  encoded so far and the label tables all live on top of it. --without-images
  peaks at ~16 GB, for the UMAP over the embeddings.

Data source: OpenCell / CytoSelf (CC BY-SA 4.0 -- ShareAlike; the manifest
records the OpenCell MAP4 imagery as cc-by-sa-4.0, and the primary AWS Open
Data registry entry is https://registry.opendata.aws/czb-opencell/)
  - Embeddings: Global VQ-VAE-2 representations (9,216-dim per image)
  - Labels: Protein name, subcellular localization
  - 114,806 images across ~1,311 proteins

References:
  - CytoSelf: Kobayashi et al., Nature Methods 2022
    https://doi.org/10.1038/s41592-022-01541-z
  - OpenCell: Cho et al., Science 2022
    https://doi.org/10.1126/science.abi6983
  - GitHub: https://github.com/royerlab/cytoself

Usage:
    python -m luxar.demos.demo_cytoself_protein_landscape
    python -m luxar.demos.demo_cytoself_protein_landscape --no-serve
    python -m luxar.demos.demo_cytoself_protein_landscape --recompute
    python -m luxar.demos.demo_cytoself_protein_landscape --without-images

    --recompute rebuilds BOTH the 3D UMAP and the hover-thumbnail bundle.
    --without-images skips the (large) image download; hover then shows a
    text-only tooltip.

Dependencies:
    pip install 'luxar[demos]'   # includes umap-learn, pandas, Pillow
"""

DEMO_META = {
    "key": "cytoself_protein_landscape",
    "title": "cytoself Protein Landscape",
    "description": "~114K CytoSelf image embeddings (OpenCell) as a 3D UMAP, colored by subcellular localization.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        # 185,838,193,451 B = 185,838 MB, from Range probes of the Drive files:
        # 4.23 GB embeddings, 71.3 MB of CSVs, 181.5 GB of Image_data crops (ten
        # files of 11.28-23.60 GB). Hover thumbnails are ON by default, so this
        # is the figure `--max-download-mb` must screen against; --without-images
        # fetches only the embeddings and label.csv, ~4240 MB.
        "download_mb": 186000,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["cytoself"],
    "outputs": ["cytoself_protein_landscape", "cytoself_landscape"],
    "citation": {
        "short": "OpenCell (Cho et al. 2022); embeddings by cytoself (Kobayashi et al. 2022)",
        "ref": "Cho / Kobayashi et al. 2022",
        "doi": "10.1126/science.abi6983",
    },
}

import errno
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    MissingDependencyError,
    add_demo_caption,
    cache_computed,
    launch_viewer,
    quarantine_file,
    require_module,
)
from luxar.demos._support._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.paths import get_demos_output_dir

_T = TypeVar("_T")

# =============================================================================
# Configuration
# =============================================================================

GDRIVE_EMBEDDINGS_ID = "1s9TL72912HH947SFWcO6tClF91xDOEEi"  # Global_representation.npy
GDRIVE_LABELS_ID = "1fl0lcrZCOkdN2vwXQSHe7i1MC04RXCiE"  # label.csv

# Image data: 10 numpy files with shape (batch, 100, 100, 4)
# Channels: [target_protein_GFP, nucleus_Hoechst, nuclear_distance, nuclear_segmentation]
# These contain ALL crops (train+val+test, ~1.1M total). The embeddings and
# label.csv are the TEST split only (~114K). We use the paired Label_data CSVs
# to identify which image rows correspond to the test split.
GDRIVE_IMAGE_IDS = {
    "Image_data00.npy": "15_CHBPT-p5JG44acP6D2hKd8jAacZatp",
    "Image_data01.npy": "1m7Cj2OALiZTIiHpvb9zFPG_I3j1wRnzK",
    "Image_data02.npy": "17nknzqlcYO3n9bAe4FwGVPkU-mJAhQ4j",
    "Image_data03.npy": "1vEsddF68dyOda-hwI-ptAL4vShBGl98Y",
    "Image_data04.npy": "1aB7WaRuhobG_IDl0l_PPeSJAxCYy-Pye",
    "Image_data05.npy": "1qb0waKcLprDtuFAdCec3WegWkmd-U45A",
    "Image_data06.npy": "1y-1vlfZ4eNhvTvpuqTZVL8DvSwYX3CH_",
    "Image_data07.npy": "1ejcPdh-d5lB1OcZ6x8SJx61pEUioZvB2",
    "Image_data08.npy": "1DOicAkruNsU5F4DWLzO2QrV6xU4kuVxs",
    "Image_data09.npy": "1a5YyHeRSRdJStG3KnFe2vsNjrsit9zbf",
}

# Label data: 10 CSV files paired with Image_data (same row ordering).
# Used to match test-split embeddings back to their image crops.
GDRIVE_LABEL_DATA_IDS = {
    "Label_data00.csv": "1CVwvXW2KhVBbTBixwRXIIiMhrlGDXz-4",
    "Label_data01.csv": "1mTYe5icvWXNfY5wEsuQUhSwgtefBJpjg",
    "Label_data02.csv": "1HckmktklyPo6qbakrwtERsCT34mRdn7l",
    "Label_data03.csv": "1GBxDmWcl_o49i4lGujA8EgIn5G4htkBr",
    "Label_data04.csv": "1G4FpJnlqB3ejmdw3SF2w3DFYt8Wnq0fT",
    "Label_data05.csv": "1Vo1J09qP2TAoXwltCF84socz2TPV92JU",
    "Label_data06.csv": "1d7gJjLTQhOw-e9KZJY9pr6KOCIN8NBvp",
    "Label_data07.csv": "1kr5EF0RA3ZwSXmoaBFwFDVnrokh2EaOE",
    "Label_data08.csv": "1mXyedmLezzty2LSSH3asw0LQeu-ie9mz",
    "Label_data09.csv": "1Vdv1cD75VhvC3FdKTen-5rqLJnWpHvmb",
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "luxar" / "cytoself"

# Staging suffixes — `.part` for an in-flight download, `.tmp` for an atomic
# rewrite of a local file — and the completion sidecar written next to a
# download once its bytes are verified. Both staging names are process-private
# (see `_part_path` / `_tmp_sibling`), so two runs sharing this cache directory
# cannot write over each other's in-flight bytes.
PART_SUFFIX = ".part"
TMP_SUFFIX = ".tmp"
COMPLETE_SUFFIX = ".complete"

# Absolute floor below which no CytoSelf artifact can be genuine, and the HTML
# markers a Google Drive quota/error page starts with (those pages are 2-3 KB,
# so a size floor alone waves them through).
MIN_DOWNLOAD_BYTES = 1_000
HTML_SNIFF_BYTES = 512
HTML_MARKERS = (b"<!doctype html", b"<html", b"<head")

# Errnos that describe the ENVIRONMENT rather than the bytes on disk: descriptor
# exhaustion, permissions, allocation failure. A truncated or garbage artifact
# surfaces from numpy/pandas as ValueError / BadZipFile / ParserError with no
# errno at all, so nothing here can mask a real corruption — while treating an
# EMFILE as one would quarantine a healthy 23 GB file and re-download it into
# the same wall. Consumed by :func:`_is_environment_failure`.
_ENVIRONMENT_ERRNOS = frozenset(
    {errno.EACCES, errno.EPERM, errno.EMFILE, errno.ENFILE, errno.ENOMEM}
)

# Exact size of every artifact, in bytes, measured two independent ways that
# agree to the byte: HTTP Range probes of the Drive files, and `stat` on a fully
# warmed cache. They sum to the 185,838,193,451 B quoted at the top of the file.
#
# PER FILE, not per family. One floor shared by ten `Image_data` files has to
# sit under the smallest of them (11.28 GB) or it rejects a genuine download —
# which leaves a 10 GB fragment of the largest (23.60 GB) looking complete, and
# a fragment is exactly what a stream cut with no declared content-length
# produces. The `Label_data` CSVs are the same hole and the worse one to fall
# into: pandas parses a truncated CSV without complaining, so the row mapping
# quietly shifts and every thumbnail after the short file lands on the wrong
# point, with nothing downstream able to notice.
#
# How much slack each user allows depends on what else it knows. A stream that
# declared no content-length has no other completeness signal at all, so it is
# held to the full size — a truncation there is invisible otherwise, and the
# CSVs make it silent (see :func:`_verify_staged_download`). A stream whose
# declared length matched allows 10% under: slack for a re-upload a few bytes
# different, without the 2x hole a family-wide floor leaves. The legacy
# sidecar-less cache check splits the same way, on whether the loader would
# catch what the size missed — 10% under for the `.npy` files, whose truncation
# `np.load` raises on, and the full size for the CSVs, whose truncation pandas
# swallows (see :func:`_cached_file_is_complete`). A file that really did shrink
# fails loudly, with the manual download URL.
ARTIFACT_SIZES = {
    "Global_representation.npy": 4_232_208_512,
    "label.csv": 6_553_361,
    "Label_data00.csv": 7_677_236,
    "Label_data01.csv": 6_046_321,
    "Label_data02.csv": 8_742_330,
    "Label_data03.csv": 6_786_224,
    "Label_data04.csv": 5_997_384,
    "Label_data05.csv": 7_414_652,
    "Label_data06.csv": 6_534_876,
    "Label_data07.csv": 5_831_517,
    "Label_data08.csv": 5_728_784,
    "Label_data09.csv": 3_950_974,
    "Image_data00.npy": 21_488_640_128,
    "Image_data01.npy": 18_204_800_128,
    "Image_data02.npy": 23_603_040_128,
    "Image_data03.npy": 18_694_080_128,
    "Image_data04.npy": 17_121_760_128,
    "Image_data05.npy": 20_590_560_128,
    "Image_data06.npy": 18_229_760_128,
    "Image_data07.npy": 15_774_080_128,
    "Image_data08.npy": 16_545_280_128,
    "Image_data09.npy": 11_282_720_128,
}

# Per-Image_data thumbnail part caches and the assembled test-aligned bundle.
# Both are versioned so a change to the encoding invalidates them by name.
# The unversioned bundle is what the pre-versioning code wrote; its contents are
# byte-identical to v1, so it is adopted rather than rebuilt (see
# `load_cytoself_images`).
THUMBNAIL_PART_TEMPLATE = "thumbs_part{index:02d}_v1.npz"
THUMBNAIL_CACHE_NAME = "image_labels_test_webp_v1.npz"
LEGACY_THUMBNAIL_CACHE_NAME = "image_labels_test_webp.npz"

# The files the test-row -> image-row mapping is derived from. A digest of them
# is stored inside the assembled bundle so the bundle is keyed to the mapping it
# was built under, the way each part cache is keyed by its stored
# `test_indices`. Without it the bundle short-circuits every mapping check: a
# `Label_data` CSV repaired or re-ordered without changing label.csv's row count
# leaves a bundle of the right LENGTH whose every blob sits on the wrong point,
# and nothing on disk ever repairs it.
MAPPING_INPUT_NAMES = ("label.csv", *GDRIVE_LABEL_DATA_IDS)

# Regression tripwire, NOT a quality bar: unmatched test rows are normal (they
# get a placeholder), but a match rate this low means the composite-key row
# matching itself has broken. Below it the assembled thumbnails are still
# returned — they are simply not frozen into the cache.
MIN_MATCH_FRACTION = 0.5

# Per-cell sphere radius in scene units. Deliberately LARGER than the ~0.4x
# median-nearest-neighbour rule the other embedding demos follow (median NN here
# is ~0.016, so that rule would give ~0.0065). This landscape is 115k cells on
# thin filaments rather than a dense ball: shrinking to 0.0065 cut lit coverage
# from 16% to 6% of frame and left the localization clusters too faint to read,
# with no gain in colour fidelity. The spacing rule screens for fusion; it is not
# a lower bound on legibility for sparse, filamentary clouds.
POINT_RADIUS = 0.02


# =============================================================================
# Google Drive Download
# =============================================================================


def _sidecar_path(output_path: Path) -> Path:
    """Path of the completion sidecar that records a verified download's size."""
    return output_path.with_name(output_path.name + COMPLETE_SUFFIX)


def _part_path(output_path: Path) -> Path:
    """Process-private staging path for an in-flight download of *output_path*.

    The cache directory is shared, so a fixed ``<name>.part`` would let two
    concurrent runs write over each other's bytes (and one of them promote the
    other's partial file). Keying on the PID gives each run its own staging
    file, which it is also the only one to delete — see
    :func:`_sweep_dead_staging_files` for how the strays are reclaimed.
    """
    return output_path.with_name(f"{output_path.name}.{os.getpid()}{PART_SUFFIX}")


def _pid_is_alive(pid: int) -> bool:
    """Whether *pid* names a live process (conservatively ``True`` if unsure).

    POSIX only. On Windows ``os.kill(pid, 0)`` does not probe, it TERMINATES,
    so there the answer is always "alive" and no staging file is ever swept.
    """
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Most often PermissionError: the process exists, someone else owns it.
        return True
    return True


def _sweep_dead_staging_files(target: Path, suffix: str) -> None:
    """Delete ``<target>.<pid><suffix>`` files left behind by dead runs.

    Process-private staging names fix the concurrent-clobber problem but move
    the leak: every interrupted run would otherwise strand its own staging file
    for good, and nothing else ever looks at them again. So the writers reclaim
    the ones whose owning process is gone first. A live owner's file, and any
    name we cannot parse a PID out of, are left strictly alone.

    Called for the multi-gigabyte leaks — ``.part`` downloads and ``.tmp`` npz
    archives. The completion sidecar's own ``.tmp`` is a few dozen bytes and is
    NOT swept; that stray is not worth a directory scan per download.
    """
    prefix = f"{target.name}."
    for candidate in target.parent.glob(f"{target.name}.*{suffix}"):
        raw_pid = candidate.name[len(prefix) : -len(suffix)]
        try:
            pid = int(raw_pid)
        except ValueError:
            continue
        if pid == os.getpid() or _pid_is_alive(pid):
            continue
        aprint(f"  Reclaiming abandoned staging file: {candidate.name}")
        candidate.unlink(missing_ok=True)


def _parse_content_length(headers: Any) -> int | None:
    """Parse a *positive* Content-Length, else ``None`` (unknown).

    Mirrors the ``_parse_len`` closure inside
    :func:`luxar.demos.robust_download`, which is nested and so cannot
    be imported: a duplicated header arrives as ``"100, 100"`` and a chunked
    host sends ``0``. Both mean *unknown* and must never be read as a real size.
    """
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        size = int(raw)
    except (TypeError, ValueError):
        return None
    return size if size > 0 else None


def _tmp_sibling(path: Path) -> Path:
    """Process-private staging sibling for an atomic write of *path*.

    Same reasoning as :func:`_part_path`: a shared ``<name>.tmp`` in a shared
    cache directory lets two runs write one file and rename the other's bytes
    into place — which for the thumbnail bundle produces a perfectly VALID npz
    holding the wrong number of blobs, so nothing downstream ever flags it.
    """
    return path.with_name(f"{path.name}.{os.getpid()}{TMP_SUFFIX}")


def _write_completion_sidecar(output_path: Path, size: int) -> None:
    """Record *size* as the verified byte count of *output_path* (atomically)."""
    sidecar = _sidecar_path(output_path)
    tmp_path = _tmp_sibling(sidecar)
    tmp_path.write_text(json.dumps({"size": int(size)}), encoding="utf-8")
    tmp_path.replace(sidecar)


def _cached_file_is_complete(path: Path, expected_min_size: int) -> bool:
    """Decide whether a cached download can be trusted without re-fetching.

    A sidecar written by this module is authoritative: the file is accepted only
    when the recorded size still matches the file on disk, so a copy truncated
    behind the sidecar's back is re-fetched instead of trusted.

    Without a sidecar (a cache written before staging existed) the file's own
    measured size is the only signal there is, and how much slack it gets depends
    on whether the LOADER would catch what the size missed.

    ``np.load`` would: a ``.npy`` header declares the shape, so a short file
    raises rather than yielding a short array, and
    :func:`_load_downloaded_artifact` quarantines and re-fetches it. Those are
    also the 4-23 GB artifacts, so they keep the ``> expected * 0.9`` floor —
    pinning them to today's exact sizes would re-download gigabytes the moment a
    file is re-uploaded a byte different.

    ``pd.read_csv`` would NOT: it parses a truncated CSV without complaining. A
    legacy ``Label_data*.csv`` cut by less than 10% therefore clears the floor,
    parses, and silently shortens the concatenated label table — which shifts
    every global image index after it and lands the thumbnails on the wrong
    points, at a match rate far above the tripwire. Nothing downstream can
    notice, so the CSVs are held to their full measured size instead, the same
    rule :func:`_verify_staged_download` applies when nothing else proved
    wholeness. The re-download that costs is 4-9 MB, not gigabytes.
    """
    if not path.is_file():
        return False

    size = path.stat().st_size
    sidecar = _sidecar_path(path)
    if sidecar.exists():
        try:
            recorded = int(json.loads(sidecar.read_text(encoding="utf-8"))["size"])
        except Exception:
            # An unreadable sidecar says nothing about the file: do not trust it.
            return False
        return recorded == size

    if path.suffix.lower() == ".csv":
        return size >= expected_min_size
    return size > expected_min_size * 0.9


def _looks_like_html(head: bytes) -> bool:
    """True when the leading bytes of a file look like an HTML document."""
    lowered = head.lower()
    return any(marker in lowered for marker in HTML_MARKERS)


def _reject_staged_download(
    part_path: Path, reason: str, file_id: str, output_path: Path
) -> RuntimeError:
    """Discard a staged download and build the error explaining why."""
    part_path.unlink(missing_ok=True)
    return RuntimeError(
        f"{reason} Try downloading manually from: "
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing "
        f"and place it at {output_path}"
    )


def _quarantine_download(path: Path, reason: str) -> None:
    """Move a rejected cached artifact aside and drop its completion sidecar.

    Dropping the sidecar matters as much as the rename: a stale sidecar left
    behind would keep describing a file that is no longer there and could shadow
    the freshly written one after a re-fetch.

    The ORDER is the load-bearing part. The rename goes first: while the bad file
    still sits at the canonical name, its sidecar is the only thing that marks it
    bad, so a rename that fails (a cross-device cache, an open handle on Windows)
    must not also leave the file looking like a sidecar-less legacy cache that
    the size heuristic then trusts. The reverse leftover is harmless — a sidecar
    with no file fails :func:`_cached_file_is_complete` on the missing file and
    is overwritten by the next successful download.
    """
    try:
        quarantine_file(path, reason=reason)
    except FileNotFoundError:
        # Already gone — either never written, or a concurrent run quarantined
        # the same artifact between our check and theirs. Either way it is out
        # from under the canonical name, which is all this needs to guarantee.
        pass
    _sidecar_path(path).unlink(missing_ok=True)


def _is_environment_failure(exc: BaseException) -> bool:
    """Whether *exc* describes the environment rather than the bytes on disk.

    Every cache reader in this module reacts to a read failure by throwing the
    artifact away, and the artifacts are expensive: 4-23 GB per download, and a
    thumbnail bundle whose rebuild walks all ten ``Image_data`` files. So the
    faults that say nothing about the file have to be told apart from the ones
    that do, and they are the same set everywhere:
    :class:`MemoryError` (the crops and the assembled bundle are large enough to
    hit it on a loaded machine), :class:`ImportError` (a reader reaching for an
    engine that is not installed is a broken environment, not a broken file) and
    an :class:`OSError` carrying one of :data:`_ENVIRONMENT_ERRNOS` — a
    descriptor limit, a permission, an allocation.

    Real corruption arrives as ValueError / BadZipFile / ParserError with no
    errno at all, so nothing here can mask it.
    """
    if isinstance(exc, (MemoryError, ImportError)):
        return True
    return isinstance(exc, OSError) and exc.errno in _ENVIRONMENT_ERRNOS


def _load_downloaded_artifact(
    path: Path,
    loader: Callable[[Path], _T],
    file_id: str,
    expected_min_size: int = 0,
) -> _T:
    """Load a downloaded artifact, self-healing once if it cannot be read.

    A cached file that survives :func:`_cached_file_is_complete` but still fails
    to parse (a legacy truncated copy, a half-written pre-fix download) would
    otherwise crash every subsequent run with an opaque error. Here it is
    quarantined, re-downloaded once, and re-read; a second failure propagates.

    ONLY parse/integrity failures count as corruption. Faults that say nothing
    about the bytes on disk are re-raised untouched — see
    :func:`_is_environment_failure`; here that spares a healthy 23 GB file from
    being thrown away and re-fetched straight into the same wall (and the 4.23 GB
    embeddings, which need ~16 GB of RAM to load, from a re-download that would
    fail identically).
    """
    try:
        return loader(path)
    except Exception as exc:
        if _is_environment_failure(exc):
            raise
        aprint(f"  ⚠ Cached file {path.name} could not be read ({exc})")
        _quarantine_download(path, reason="unreadable cached download")
        _download_from_google_drive(file_id, path, expected_min_size=expected_min_size)
        return loader(path)


def _open_drive_stream(session: Any, url: str, file_id: str) -> Any:
    """Get a streamed response for *file_id*, past the virus-scan confirmation.

    Google Drive answers a large-file download with an interstitial instead of
    the bytes, and which interstitial depends on the file and the day, so all
    four escalations below are needed. Each step only runs while the response is
    still ``text/html`` — the first one that yields a body short-circuits the
    rest. A response that is STILL html when this returns is not an error here:
    the caller's HTML sniff rejects it after staging, with the manual URL.
    """
    # Strategy 1: Direct download with confirm=t
    response = session.get(url, params={"confirm": "t"}, stream=True, timeout=60)

    # Strategy 2: Check cookies for download_warning token
    if response.headers.get("content-type", "").startswith("text/html"):
        aprint("Trying cookie-based confirmation...")
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break
        if confirm_token:
            response = session.get(
                url,
                params={"confirm": confirm_token},
                stream=True,
                timeout=60,
            )

    # Strategy 3: Parse the HTML confirmation page
    if response.headers.get("content-type", "").startswith("text/html"):
        import html as html_mod
        import re

        aprint("Parsing confirmation page for download form...")
        page_html = response.text

        action_match = re.search(r'action="([^"]*)"', page_html)
        form_inputs = dict(
            re.findall(
                r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"',
                page_html,
            )
        )

        if action_match and form_inputs:
            action_url = html_mod.unescape(action_match.group(1))
            aprint(f"Found download form with {len(form_inputs)} params")
            response = session.get(
                action_url,
                params=form_inputs,
                stream=True,
                timeout=60,
            )
        else:
            # Strategy 4: Try the usercontent endpoint
            aprint("Trying usercontent endpoint...")
            uc_url = (
                f"https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t"
            )
            response = session.get(uc_url, stream=True, timeout=60)

    return response


def _stream_to_part(response: Any, part_path: Path, total_size: int | None) -> int:
    """Stream *response* into *part_path*, reporting progress; return the size."""
    downloaded = 0
    last_report_mb = 0
    start_time = time.time()
    chunk_size = 1024 * 1024  # 1 MB

    with open(part_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)

            progress_mb = downloaded / (1024 * 1024)
            if progress_mb - last_report_mb < 100:
                continue
            elapsed = time.time() - start_time
            rate = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
            if total_size is not None:
                pct = downloaded / total_size * 100
                aprint(
                    f"  {downloaded / (1024**3):.2f} / "
                    f"{total_size / (1024**3):.2f} GB "
                    f"({pct:.0f}%) - {rate:.1f} MB/s"
                )
            else:
                aprint(f"  {downloaded / (1024**3):.2f} GB - {rate:.1f} MB/s")
            last_report_mb = progress_mb

    return part_path.stat().st_size


def _verify_staged_download(
    part_path: Path,
    final_size: int,
    total_size: int | None,
    expected_min_size: int,
    file_id: str,
    output_path: Path,
) -> None:
    """Raise (and discard the staging file) unless the staged bytes look real."""
    with open(part_path, "rb") as f:
        head = f.read(HTML_SNIFF_BYTES)
    if _looks_like_html(head):
        raise _reject_staged_download(
            part_path,
            "Downloaded file is an HTML page, not data — Google Drive most "
            "likely returned a quota or permission error.",
            file_id,
            output_path,
        )

    if final_size < MIN_DOWNLOAD_BYTES:
        raise _reject_staged_download(
            part_path,
            f"Downloaded file is too small ({final_size} bytes) — likely a "
            "Google Drive error page.",
            file_id,
            output_path,
        )

    if total_size is not None and final_size != total_size:
        raise _reject_staged_download(
            part_path,
            f"Download is incomplete: got {final_size} bytes but the "
            f"server declared {total_size}.",
            file_id,
            output_path,
        )

    # The expected-size gate runs whether or not a length was declared, but how
    # tight it is depends on what the declared length already proved.
    #
    # NO declared length — Drive's chunked/connection-close path, and the usual
    # one for the multi-gigabyte archives — means nothing so far says the body
    # arrived WHOLE: a stream cut short simply ends. The measured size is then
    # the only completeness signal there is, so it is required in full. Slack
    # here is the dangerous window and buys almost nothing: 8 MB of the 8.74 MB
    # `Label_data02.csv` clears a 90% bar, pandas parses that truncation without
    # complaining, and every image index after the short file shifts.
    #
    # WITH a declared length the body is provably whole (checked just above), so
    # a size that disagrees with the table means a DIFFERENT file rather than a
    # truncated one — a re-upload, or one of Drive's small non-HTML "cannot
    # access this file" bodies, which agree with their own length and would
    # otherwise be promoted AND certified by a sidecar. A 10% floor is the right
    # shape there: it waves a re-upload a few bytes different through and still
    # rejects the error bodies.
    too_short = (
        final_size < expected_min_size
        if total_size is None
        else final_size <= expected_min_size * 0.9
    )
    if too_short:
        raise _reject_staged_download(
            part_path,
            f"Download is too short: got {final_size} bytes, under the "
            f"{expected_min_size} bytes expected for this file.",
            file_id,
            output_path,
        )


def _download_from_google_drive(
    file_id: str, output_path: Path, expected_min_size: int = 0
) -> Path:
    """Download a file from Google Drive, handling the virus-scan confirmation.

    Only the confirmation dance below is genuinely custom — Google Drive's
    four-way "are you sure" flow has no equivalent in
    :mod:`luxar.demos`, which is why ``robust_download`` cannot be used
    directly. The FINISH, however, now matches ``robust_download``'s contract:
    the body is streamed into a process-private sibling ``.part`` file, verified
    (HTML sniff, absolute floor, declared content-length, expected-size floor),
    and only then renamed onto *output_path*, followed by a completion sidecar
    recording the verified size. Staging files abandoned by dead runs are swept
    first — see :func:`_sweep_dead_staging_files`.

    Args:
        file_id: Google Drive file ID.
        output_path: Where to save the downloaded file.
        expected_min_size: Minimum expected file size in bytes (for cache check).

    Returns:
        Path to downloaded file.

    Raises:
        RuntimeError: The response was an HTML error page, was truncated, or
            disagreed with its declared content-length. Nothing is cached.
    """
    # Check if already downloaded
    if _cached_file_is_complete(output_path, expected_min_size):
        aprint(f"File already downloaded: {output_path.name}")
        aprint(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
        return output_path

    # `requests` is a CORE Luxar dependency, so it needs no gate (only the
    # `demos`-extra modules below are optional).
    import requests

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Stage into a sibling ".part" so an interruption can never leave a
    # truncated file at the destination. No HTTP resume: the confirmation dance
    # makes a byte-range continuation unreliable, so our own stale part is
    # discarded and the stream restarted. Any existing sidecar is deliberately
    # LEFT ALONE — it is the only evidence that the file already at the
    # destination is bad, and deleting it before the replacement exists would
    # promote that bad file to "plausible legacy cache" if this fetch fails.
    # `_write_completion_sidecar` overwrites it atomically on success.
    part_path = _part_path(output_path)
    part_path.unlink(missing_ok=True)
    _sweep_dead_staging_files(output_path, PART_SUFFIX)

    session = requests.Session()
    # `requests` advertises `gzip, deflate` by default and `iter_content`
    # DECODES the body, while Content-Length describes the COMPRESSED bytes — a
    # gzipped text/csv would then look truncated and be rejected even though it
    # arrived intact. `_support.downloads.download._force_identity_encoding`
    # codifies this precondition, but its "unless the caller already set it"
    # guard would preserve the Session's own gzip default, so the header is set outright.
    # Setting it on the session makes all four strategies below inherit it.
    session.headers["Accept-Encoding"] = "identity"

    with asection(f"Downloading {output_path.name} from Google Drive"):
        aprint(f"File ID: {file_id}")

        response = _open_drive_stream(session, url, file_id)
        response.raise_for_status()

        total_size = _parse_content_length(response.headers)
        if total_size is not None:
            aprint(f"Download size: {total_size / (1024**2):.1f} MB")

        final_size = _stream_to_part(response, part_path, total_size)
        aprint(f"Download complete: {final_size / (1024**2):.1f} MB")

        # Verify the staged bytes BEFORE they take the destination name.
        _verify_staged_download(
            part_path, final_size, total_size, expected_min_size, file_id, output_path
        )

        # Verified: take the destination name atomically, then record the size
        # so the next run can tell a complete cache from a truncated one.
        part_path.replace(output_path)
        _write_completion_sidecar(output_path, final_size)

    return output_path


# =============================================================================
# Data Loading & UMAP
# =============================================================================


def load_cytoself_data(
    cache_dir: Path | None = None,
    recompute: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Load CytoSelf embeddings and compute 3D UMAP.

    Downloads Global_representation.npy (4.23 GB) and label.csv from Google Drive,
    then runs UMAP dimensionality reduction (cached after first run).

    Args:
        cache_dir: Directory for caching downloads and UMAP results.
        recompute: If True, recompute UMAP even if cache exists.

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of 3D UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    # Gated here, not in main(): label.csv parsing needs pandas on every run.
    pd = require_module("pandas")

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Download raw data ---
    embeddings_path = cache_dir / "Global_representation.npy"
    labels_path = cache_dir / "label.csv"

    with asection("Loading CytoSelf Data"):
        _download_from_google_drive(
            GDRIVE_EMBEDDINGS_ID,
            embeddings_path,
            expected_min_size=ARTIFACT_SIZES[embeddings_path.name],
        )
        _download_from_google_drive(
            GDRIVE_LABELS_ID,
            labels_path,
            expected_min_size=ARTIFACT_SIZES[labels_path.name],
        )

    # --- Extract attributes from labels ---
    with asection("Processing Labels"):
        df = _load_downloaded_artifact(
            labels_path,
            pd.read_csv,
            GDRIVE_LABELS_ID,
            expected_min_size=ARTIFACT_SIZES[labels_path.name],
        )
        aprint(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

        attributes: dict[str, np.ndarray] = {}
        category_maps: dict[str, list] = {}

        # Columns: ensg, name, loc_grade1, loc_grade2, loc_grade3, protein_id, FOV_id
        # loc_grade1 = primary localization (semicolon-separated multi-labels)
        # name = protein name (~1,311 unique)

        # Extract primary localization (first label before semicolon, NaN -> "unknown")
        if "loc_grade1" in df.columns:
            primary_loc = df["loc_grade1"].fillna("unknown").str.split(";").str[0]
            cat = pd.Categorical(primary_loc)
            attributes["localization"] = cat.codes.astype(np.int32)
            category_maps["localization"] = list(cat.categories)
            aprint(f"  localization: {len(cat.categories)} unique values")

        # Protein name
        if "name" in df.columns:
            cat = pd.Categorical(df["name"])
            attributes["protein_name"] = cat.codes.astype(np.int32)
            category_maps["protein_name"] = list(cat.categories)
            aprint(f"  protein_name: {len(cat.categories)} unique values")

        if not attributes:
            raise RuntimeError(
                f"No usable columns found in label.csv. "
                f"Available columns: {list(df.columns)}"
            )

    # --- Compute or load cached 3D UMAP ---
    def _compute_umap3d() -> np.ndarray:
        # Gated here, not in main(): a warm cytoself UMAP cache never calls this.
        UMAP = require_module("umap").UMAP

        with asection("Computing 3D UMAP (this may take 10-30 minutes)"):
            aprint("Loading embeddings into memory...")
            embeddings = _load_downloaded_artifact(
                embeddings_path,
                np.load,
                GDRIVE_EMBEDDINGS_ID,
                expected_min_size=ARTIFACT_SIZES[embeddings_path.name],
            )
            aprint(f"Embeddings shape: {embeddings.shape}")
            aprint("Parameters: n_neighbors=15, min_dist=0.1, metric=cosine")

            reducer = UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                n_jobs=-1,
                low_memory=True,
                verbose=True,
            )
            coordinates = reducer.fit_transform(embeddings).astype(np.float32)
            coordinates -= coordinates.mean(axis=0)
            aprint(f"UMAP complete: {coordinates.shape}")

        return coordinates

    # The CytoSelf embeddings are a single fixed Google-Drive dataset and the
    # UMAP params above are fixed, so a static key is safe. --recompute forces
    # a rebuild. Cached under ~/.cache/luxar/cytoself/umap3d_v1.pkl.
    coordinates = cache_computed(
        "cytoself", "umap3d", _compute_umap3d, version=1, recompute=recompute
    )

    aprint(f"Coordinates: {len(coordinates):,} points")
    aprint(f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]")
    aprint(f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
    aprint(f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]")

    return coordinates, attributes, category_maps


def _encode_crops_to_webp(
    images: np.ndarray,
) -> list[bytes]:
    """Encode (N, 100, 100, 4) image crops to WebP thumbnails.

    Takes channels 0 (protein GFP) and 1 (nucleus Hoechst), normalizes
    per-image to uint8, composites as green+blue RGB, and encodes to WebP.
    """
    import io

    # Optional dep, gated per helper: the soft check in main() lives in a
    # different function, so calling this directly would raise a bare error.
    PILImage = require_module("PIL.Image")

    n_crops = images.shape[0]

    # Extract channels: 0 = protein (GFP), 1 = nucleus (Hoechst)
    protein = images[:, :, :, 0].astype(np.float32)
    nucleus = images[:, :, :, 1].astype(np.float32)

    # Per-image min-max normalization to uint8
    def normalize(ch: np.ndarray) -> np.ndarray:
        flat = ch.reshape(ch.shape[0], -1)
        ch_min = flat.min(axis=1)[:, np.newaxis, np.newaxis]
        ch_max = flat.max(axis=1)[:, np.newaxis, np.newaxis]
        return ((ch - ch_min) / np.maximum(ch_max - ch_min, 1e-8) * 255).astype(
            np.uint8
        )

    protein_u8 = normalize(protein)
    nucleus_u8 = normalize(nucleus)

    # RGB composite: green = protein, blue = nucleus
    rgb = np.zeros((n_crops, 100, 100, 3), dtype=np.uint8)
    rgb[:, :, :, 1] = protein_u8
    rgb[:, :, :, 2] = nucleus_u8

    blobs: list[bytes] = []
    for i in range(n_crops):
        buf = io.BytesIO()
        PILImage.fromarray(rgb[i], mode="RGB").save(buf, format="webp", quality=85)
        blobs.append(buf.getvalue())

    return blobs


def _build_test_index_mapping(
    cache_dir: Path,
) -> tuple[dict[int, int], int]:
    """Build a mapping from test-split row index to global image row index.

    Downloads the 10 Label_data CSVs (small files), concatenates them to get
    the full ~1.1M-row label table, then matches rows from label.csv (the
    test split) to find their positions in the full dataset.

    The Image_data .npy files are row-aligned with the Label_data CSVs, so
    the resulting mapping tells us which image rows to encode.

    Args:
        cache_dir: Directory for caching downloads.

    Returns:
        Tuple of (mapping, n_test) where mapping is
        test_row_index -> global_image_row_index and n_test is the total
        number of test rows.
    """
    # Gated here, not in main(): only the image-thumbnail path reaches this.
    pd = require_module("pandas")

    with asection("Building test-to-image index mapping"):
        # Download the 10 Label_data CSVs (small: ~5-10 MB each)
        label_dfs = []
        for filename, file_id in GDRIVE_LABEL_DATA_IDS.items():
            csv_path = cache_dir / filename
            _download_from_google_drive(
                file_id, csv_path, expected_min_size=ARTIFACT_SIZES[filename]
            )
            df = _load_downloaded_artifact(
                csv_path,
                lambda p: pd.read_csv(p, header=None),
                file_id,
                expected_min_size=ARTIFACT_SIZES[filename],
            )
            label_dfs.append(df)

        full_labels = pd.concat(label_dfs, ignore_index=True)
        aprint(
            f"Full label table: {len(full_labels):,} rows x {full_labels.shape[1]} cols"
        )

        # Load test-split labels
        test_labels = _load_downloaded_artifact(
            cache_dir / "label.csv",
            pd.read_csv,
            GDRIVE_LABELS_ID,
            expected_min_size=ARTIFACT_SIZES["label.csv"],
        )
        aprint(f"Test labels: {len(test_labels):,} rows")

        # The Label_data CSVs have no header; label.csv has headers.
        # Assign column names from label.csv to the full table.
        if full_labels.shape[1] == len(test_labels.columns):
            full_labels.columns = test_labels.columns
        else:
            aprint(
                f"  ⚠ Column count mismatch: full={full_labels.shape[1]}, "
                f"test={len(test_labels.columns)}"
            )
            # Try using the first N columns that match
            full_labels.columns = [
                test_labels.columns[i] if i < len(test_labels.columns) else f"col_{i}"
                for i in range(full_labels.shape[1])
            ]

        # Build composite key for matching. Use all available columns for
        # an exact row match. Fill NaN consistently so keys match.
        key_cols = list(test_labels.columns)
        for col in key_cols:
            full_labels[col] = full_labels[col].fillna("__NA__").astype(str)
            test_labels[col] = test_labels[col].fillna("__NA__").astype(str)

        full_labels["_key"] = full_labels[key_cols].agg("|".join, axis=1)
        test_labels["_key"] = test_labels[key_cols].agg("|".join, axis=1)

        # Build lookup: key -> list of global indices (duplicates possible)
        from collections import defaultdict

        key_to_global: dict[str, list[int]] = defaultdict(list)
        for global_idx, key in enumerate(full_labels["_key"]):
            key_to_global[key].append(global_idx)

        # Match each test row. For duplicate keys, consume indices in order.
        key_usage: dict[str, int] = defaultdict(int)
        mapping: dict[int, int] = {}
        unmatched = 0

        for test_idx, key in enumerate(test_labels["_key"]):
            candidates = key_to_global.get(key, [])
            usage = key_usage[key]
            if usage < len(candidates):
                mapping[test_idx] = candidates[usage]
                key_usage[key] = usage + 1
            else:
                unmatched += 1

        aprint(
            f"Matched {len(mapping):,}/{len(test_labels):,} test rows "
            f"to image indices ({unmatched:,} unmatched)"
        )

    return mapping, len(test_labels)


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    """Write an ``.npz`` archive so readers never see a half-written file.

    ``np.savez`` appends ``.npz`` to a *path* that lacks the suffix, which would
    defeat a ``foo.npz.tmp`` staging name — so the archive is written through an
    open file handle and the finished file is renamed into place. The staging
    name is process-private (see :func:`_tmp_sibling`).
    """
    _sweep_dead_staging_files(path, TMP_SUFFIX)
    tmp_path = _tmp_sibling(path)
    try:
        with open(tmp_path, "wb") as handle:
            np.savez(handle, **arrays)
        tmp_path.replace(path)
    except BaseException:
        # A failure here is most often ENOSPC on the ~114k-blob bundle. Leaving
        # a full-size stray behind would waste the space exactly when it is
        # scarcest, and nothing else ever reclaims it.
        tmp_path.unlink(missing_ok=True)
        raise


def _thumbnail_part_path(cache_dir: Path, index: int) -> Path:
    """Cache path of the encoded thumbnails from one ``Image_data`` file."""
    return cache_dir / THUMBNAIL_PART_TEMPLATE.format(index=index)


def _as_webp_blob(entry: object) -> bytes:
    """Return one cached thumbnail entry as WebP bytes, or raise if it is not one.

    ``bytes()`` alone is too weak to be the validity check: it converts a
    NUMERIC entry silently (a float yields the 8 bytes of its IEEE encoding, an
    int that many NULs), so an archive of the right length but the wrong dtype
    would sail past the guard and feed the viewer garbage images forever. Every
    entry this cache ever writes is WebP — the encoder's output or the 1x1
    placeholder — so the container signature is the honest test.

    Raising is the contract: both readers (:func:`_load_thumbnail_part` and
    :func:`_read_bundle_blobs`) call this INSIDE their own guard, so a
    non-WebP entry is quarantined and rebuilt like any other unusable archive.
    Checking the parts as well as the bundle is what keeps that terminating: a
    garbage part would otherwise be reassembled into a garbage bundle, which
    the bundle check then quarantines, on every single run.
    """
    if not isinstance(entry, (bytes, bytearray)):
        raise TypeError(f"blob entry is {type(entry).__name__}, not bytes")
    blob = bytes(entry)
    if blob[:4] != b"RIFF" or blob[8:12] != b"WEBP":
        raise ValueError("blob entry is not a WebP image")
    return blob


def _expected_test_indices(
    global_to_test: dict[int, list[int]], global_offset: int, n_crops: int
) -> list[int]:
    """The test rows an ``Image_data`` file at *global_offset* must contribute.

    Built in exactly the order the encode path appends them, so a part cache's
    stored ``test_indices`` can be compared against it element by element. That
    equality is what keys a part cache to the CURRENT row mapping: a part built
    against a different mapping — even one of the same length — disagrees here
    and is rebuilt rather than pasted onto the wrong rows.
    """
    expected: list[int] = []
    for local_idx in range(n_crops):
        expected.extend(global_to_test.get(global_offset + local_idx, ()))
    return expected


def _load_thumbnail_part(path: Path) -> tuple[list[bytes], list[int], int] | None:
    """Load a per-file thumbnail part cache.

    Returns ``(blobs, test_indices, n_crops)``, or ``None`` when the part has
    not been built yet or could not be read (in which case it is quarantined so
    the caller rebuilds it).

    An environment failure (:func:`_is_environment_failure`) propagates instead:
    rebuilding a part means re-reading an 11-23 GB ``Image_data`` file into RAM,
    so discarding a healthy part cache because the machine is out of memory or
    descriptors trades a good cache for a job that is about to fail harder.
    """
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=True) as data:
            blobs = [_as_webp_blob(b) for b in data["blobs"]]
            test_indices = [int(i) for i in data["test_indices"]]
            n_crops = int(data["n_crops"])
        if len(blobs) != len(test_indices):
            raise ValueError("blobs and test_indices have different lengths")
    except Exception as exc:
        if _is_environment_failure(exc):
            raise
        aprint(f"  ⚠ Thumbnail part cache unreadable ({exc}) — rebuilding")
        _quarantine_download(path, reason="unreadable thumbnail part cache")
        return None

    return blobs, test_indices, n_crops


def _mapping_fingerprint(cache_dir: Path) -> str:
    """Digest the CSVs the row mapping is derived from, ``""`` when unknowable.

    The empty string is returned as soon as one input is missing, and it means
    "cannot be checked", never "does not match": a user who pruned the CSVs to
    reclaim disk must not have a valid 114k-thumbnail bundle thrown away and
    181 GB of ``Image_data`` re-fetched to rebuild it. Content is hashed rather
    than size/mtime so that re-downloading an identical file — which the
    self-heal path does routinely — keeps the same fingerprint.
    """
    digest = hashlib.blake2b(digest_size=16)
    for name in MAPPING_INPUT_NAMES:
        path = cache_dir / name
        if not path.is_file():
            return ""
        digest.update(name.encode("utf-8"))
        with open(path, "rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _stored_mapping_fingerprint(data: Any) -> str:
    """The fingerprint recorded in a bundle, ``""`` for one written without."""
    if "mapping_fingerprint" not in data.files:
        return ""
    return str(data["mapping_fingerprint"])


def _read_bundle_blobs(
    path: Path, expected_count: int | None, expected_fingerprint: str = ""
) -> list[bytes] | None:
    """Read an assembled thumbnail bundle, or ``None`` if it cannot be used.

    Three ways a bundle is unusable, and all of them quarantine it so the caller
    rebuilds rather than crashing or degrading forever:

    * it does not parse — a truncated npz used to crash every subsequent run
      until the user deleted it by hand. A bundle that opens fine but whose
      entries are not WebP images is the same dead end, so the decode
      (:func:`_as_webp_blob`) happens INSIDE this guard rather than after it;
    * it parses but holds the wrong number of blobs. That is the *silent*
      failure, and the one a bundle cannot self-report: the loser of the
      pre-staging rename race is a perfectly valid npz whose blob count belongs
      to another run's mapping. The scene builder refuses a mismatched count
      (it cannot align the thumbnails to the points), so such a bundle costs
      every future run its hover images with nothing on disk ever repairing it.
      *expected_count* is the caller's point count; ``None`` skips the check.
    * it holds the right number of blobs but was built under a DIFFERENT row
      mapping — a repaired ``Label_data`` CSV shifts which image row each test
      row points at without changing how many there are, so the count check is
      structurally blind to it and every thumbnail lands on the wrong point.
      *expected_fingerprint* is :func:`_mapping_fingerprint` for the current
      cache; an empty one on either side means "cannot be checked" and the
      bundle is accepted, since a false rebuild costs a 181 GB re-download.

    None of that applies to an environment failure
    (:func:`_is_environment_failure`), which propagates: the bundle holds ~114k
    blobs, so it is precisely the read that a memory- or descriptor-starved
    machine fails on, and quarantining it there would trade a good cache for the
    most expensive rebuild in the demo on the strength of a fault that says
    nothing about its bytes.
    """
    try:
        with np.load(path, allow_pickle=True) as data:
            blobs = [_as_webp_blob(b) for b in data["blobs"]]
            stored_fingerprint = _stored_mapping_fingerprint(data)
    except Exception as exc:
        if _is_environment_failure(exc):
            raise
        aprint(f"  ⚠ Cached thumbnail bundle unreadable ({exc}) — rebuilding")
        _quarantine_download(path, reason="unreadable thumbnail bundle")
        return None

    if expected_count is not None and len(blobs) != expected_count:
        aprint(
            f"  ⚠ Cached thumbnail bundle holds {len(blobs):,} thumbnails but "
            f"{expected_count:,} are needed — rebuilding"
        )
        _quarantine_download(path, reason="thumbnail bundle with a stale blob count")
        return None

    comparable = expected_fingerprint and stored_fingerprint
    if comparable and expected_fingerprint != stored_fingerprint:
        aprint(
            "  ⚠ Cached thumbnail bundle was built from different label CSVs "
            "— rebuilding"
        )
        _quarantine_download(path, reason="thumbnail bundle from a stale row mapping")
        return None

    return blobs


def _legacy_bundle_is_adoptable() -> bool:
    """Whether a pre-versioning bundle is still readable under the current name.

    The literal comparison is the enforcement, not a note: bump the encoding to
    v2 and this goes False by itself, so a stale bundle is rebuilt instead of
    laundered into the new name. Shared with :func:`_surviving_bundle` so that
    "can be adopted" and "would be read by a later run" can never drift apart.
    """
    return THUMBNAIL_CACHE_NAME == "image_labels_test_webp_v1.npz"


def _surviving_bundle(cache_dir: Path) -> Path | None:
    """The bundle a later run would read, or ``None`` if there is none.

    BOTH names have to be considered. Adoption normally leaves only the
    versioned one, but its un-renameable branch leaves the legacy file in place
    and still serving — so a message that looked only at the versioned name
    would fall silent in exactly the case where something did survive.
    Precedence matches :func:`_load_cached_thumbnails`: v1 first.
    """
    if (cache_dir / THUMBNAIL_CACHE_NAME).exists():
        return cache_dir / THUMBNAIL_CACHE_NAME
    legacy = cache_dir / LEGACY_THUMBNAIL_CACHE_NAME
    if _legacy_bundle_is_adoptable() and legacy.exists():
        return legacy
    return None


def _adopt_legacy_bundle(
    cache_dir: Path, expected_count: int | None
) -> list[bytes] | None:
    """Rename a pre-versioning bundle onto the versioned name; ``None`` if none.

    A bundle written before the name carried a version has contents
    byte-identical to v1, so renaming is sound and spares an existing user a
    multi-gigabyte re-download for what is only a rename.

    Callers rely on the SIDE EFFECT as much as the return value: after a
    successful rename only one bundle name exists, so nothing is orphaned under
    the other one.

    Taking the versioned name is refused while that name is occupied, and the
    refusal is enforced twice over. The cheap ``exists()`` check below spares a
    200 MB read when a v1 is already there, and it is enforced HERE rather than
    only by the callers: at :func:`_load_cached_thumbnails` the precondition
    otherwise holds by a non-local accident (every rejecting branch of
    :func:`_read_bundle_blobs` happens to quarantine the versioned file first),
    so one cheap reject-without-quarantining pre-check added there would turn
    adoption into a clobber. The ``os.link`` that finishes the job is what makes
    the refusal RELIABLE — a check followed by a rename is only a check, and the
    validating read in between is long enough for a concurrent run to publish a
    fresh v1 into the gap.

    The un-linkable branch leaves the legacy file where it is: on the normal
    path that is harmless (its blobs are returned, no v1 is written, and the
    next run retries the adoption), but a caller that goes on to write a v1
    anyway makes that leftover unreachable for good.
    """
    legacy_cache = cache_dir / LEGACY_THUMBNAIL_CACHE_NAME
    if not _legacy_bundle_is_adoptable() or not legacy_cache.exists():
        return None
    if (cache_dir / THUMBNAIL_CACHE_NAME).exists():
        return None

    thumbnails_cache = cache_dir / THUMBNAIL_CACHE_NAME
    with asection("Adopting pre-versioning thumbnail cache"):
        # Validated BEFORE the rename: a legacy bundle is exactly the vintage
        # that can carry a raced blob count, and adopting one under the v1 name
        # would make that permanent.
        blobs = _read_bundle_blobs(
            legacy_cache, expected_count, _mapping_fingerprint(cache_dir)
        )
        if blobs is None:
            return None
        try:
            # Link-then-unlink rather than `replace()`, because `replace()`
            # OVERWRITES and the check above is only a check: reading and
            # validating a 200 MB bundle takes long enough for a concurrent run
            # to finish a rebuild and publish a fresh, fingerprinted v1 in the
            # gap, which the rename would then destroy in favour of an
            # unverifiable legacy one. `os.link` refuses an occupied name
            # atomically, so the loser of that race keeps its hands off. Dying
            # between the two calls leaves both names pointing at one inode —
            # no extra bytes, and v1 takes precedence everywhere.
            os.link(legacy_cache, thumbnails_cache)
        except FileExistsError:
            # Another run published a versioned bundle while this one was
            # validating. Theirs is at least as good and carries a fingerprint;
            # the blobs read here are still fine to return.
            aprint(
                f"  ⚠ Another run published {THUMBNAIL_CACHE_NAME} first — "
                "using the legacy bundle for this run and leaving both in place"
            )
        except OSError as exc:
            # Un-linkable (an open handle on Windows, a read-only cache, a
            # filesystem without hard links). The bytes are good, so use them;
            # the next run tries the adoption again.
            aprint(
                f"  ⚠ Could not rename the legacy bundle ({exc}) — using it in place"
            )
        else:
            legacy_cache.unlink(missing_ok=True)
            aprint(
                f"Adopted {len(blobs):,} thumbnails from "
                f"{LEGACY_THUMBNAIL_CACHE_NAME} as {THUMBNAIL_CACHE_NAME}"
            )
        return blobs


def _load_cached_thumbnails(
    cache_dir: Path, expected_count: int | None
) -> list[bytes] | None:
    """Return the assembled thumbnails already on disk, or ``None`` to rebuild.

    Looks first for the versioned bundle, then falls back to adopting a
    pre-versioning one (:func:`_adopt_legacy_bundle`).

    Both bundles are checked against the row mapping the current label CSVs
    produce — see :func:`_read_bundle_blobs`. This is the only place that check
    can happen: the bundle short-circuits the whole per-file pipeline, so
    without it the mapping-keyed part caches guard nothing on a warm cache.
    """
    thumbnails_cache = cache_dir / THUMBNAIL_CACHE_NAME
    if thumbnails_cache.exists():
        with asection("Loading cached image thumbnails"):
            blobs = _read_bundle_blobs(
                thumbnails_cache, expected_count, _mapping_fingerprint(cache_dir)
            )
            if blobs is not None:
                aprint(f"Loaded {len(blobs):,} cached thumbnails (test-aligned)")
                return blobs

    return _adopt_legacy_bundle(cache_dir, expected_count)


def _reusable_thumbnail_part(
    part_path: Path,
    filename: str,
    global_to_test: dict[int, list[int]],
    global_offset: int,
) -> tuple[list[bytes], list[int], int] | None:
    """A part cache for *filename* that the CURRENT row mapping can reuse.

    A part cache holds everything one ``Image_data`` file contributes,
    INCLUDING its crop count — the caller advances a running global offset by
    that count, so a skipped file must still move the offset or every later
    file would match the wrong rows.
    """
    part = _load_thumbnail_part(part_path)
    if part is None:
        return None
    if part[1] != _expected_test_indices(global_to_test, global_offset, part[2]):
        # The part was built against a different row mapping — a repaired
        # Label_data CSV shifts it without changing its length. Reusing it would
        # paste every blob onto the wrong row and then freeze that into the
        # bundle, so rebuild instead.
        aprint(f"  ⚠ Stale thumbnail part cache for {filename} — rebuilding")
        _quarantine_download(part_path, reason="stale thumbnail part cache")
        return None
    return part


def _build_thumbnail_part(
    cache_dir: Path,
    filename: str,
    file_id: str,
    position: str,
    global_to_test: dict[int, list[int]],
    global_offset: int,
) -> tuple[list[bytes], list[int], int]:
    """Download one ``Image_data`` file and encode its matched crops.

    The result is checkpointed by the caller, immediately — so a failure on a
    later file costs only that file's work rather than the whole multi-hour
    encode. The write is the CALLER's because the two steps fail over different
    artifacts and each has to be able to name its own: an ENOSPC writing
    ``thumbs_partNN`` must not be reported against the 23 GB download that
    produced it. See :func:`_thumbnail_part_for`.

    Deliberately not given the checkpoint path: a function that cannot name the
    file cannot quietly grow a write back into itself, which would put that
    ENOSPC back under the download's name and undo the split.
    """
    img_path = cache_dir / filename
    _download_from_google_drive(
        file_id, img_path, expected_min_size=ARTIFACT_SIZES[filename]
    )

    with asection(f"Processing {filename} ({position})"):
        arr = _load_downloaded_artifact(
            img_path, np.load, file_id, expected_min_size=ARTIFACT_SIZES[filename]
        )

        n_crops = arr.shape[0]
        aprint(f"Shape: {arr.shape}, dtype: {arr.dtype}")

        # Find which local indices in this file are needed
        local_indices = []
        local_to_test_map: list[tuple[int, int]] = []
        for local_idx in range(n_crops):
            global_idx = global_offset + local_idx
            if global_idx in global_to_test:
                local_indices.append(local_idx)
                for test_idx in global_to_test[global_idx]:
                    local_to_test_map.append((local_idx, test_idx))

        new_blobs: list[bytes] = []
        new_test_indices: list[int] = []
        if local_indices:
            # Extract and encode only the needed crops
            unique_local = sorted(set(local_indices))
            encoded = _encode_crops_to_webp(arr[unique_local])

            # Map encoded blobs back to test indices
            local_to_encoded = {li: ei for ei, li in enumerate(unique_local)}
            for local_idx, test_idx in local_to_test_map:
                new_blobs.append(encoded[local_to_encoded[local_idx]])
                new_test_indices.append(test_idx)

            aprint(
                f"Encoded {len(unique_local):,} matched crops (of {n_crops:,} total)"
            )
        else:
            aprint(f"No matched crops in this file ({n_crops:,} total)")

        del arr

    return new_blobs, new_test_indices, n_crops


def _write_thumbnail_part(
    part_path: Path, part: tuple[list[bytes], list[int], int]
) -> None:
    """Checkpoint one file's encoded contribution, crop count included."""
    new_blobs, new_test_indices, n_crops = part
    _write_npz_atomic(
        part_path,
        blobs=np.array(new_blobs, dtype=object),
        test_indices=np.asarray(new_test_indices, dtype=np.int64),
        n_crops=np.int64(n_crops),
    )


def _report_bundle_not_cached(cache_dir: Path, reason: str) -> None:
    """Print *reason*, then say what declining to cache left on disk.

    A usable bundle can still be on disk here, because ``--recompute``
    deliberately does not delete one before a replacement exists. Saying so is
    the whole point: silence read as "the escape hatch worked" when in fact the
    next run will serve that same bundle again, and the user had no way to tell.

    What it promises is deliberately weak. The survivor is NOT validated here —
    validating means quarantining, which is the destruction this whole path
    exists to avoid — so all that can honestly be said is that the next run
    re-checks it. On the ``wrong_count`` branch that re-check will in fact
    quarantine it; on the tripwire branch it usually will not. Either way the
    user is told the file is still there and how to remove it themselves.
    """
    aprint(reason)
    survivor = _surviving_bundle(cache_dir)
    if survivor is not None:
        aprint(
            f"    The bundle already on disk ({survivor.name}) was left in "
            "place — nothing here throws away a working artifact to make room "
            "for one that did not pass. The next run re-checks it as usual and "
            "rebuilds if its own count/mapping checks reject it; delete it by "
            "hand to force a rebuild from the per-file caches."
        )


def _thumbnail_part_for(
    cache_dir: Path,
    part_path: Path,
    filename: str,
    file_id: str,
    position: str,
    global_to_test: dict[int, list[int]],
    global_offset: int,
) -> tuple[list[bytes], list[int], int]:
    """One ``Image_data`` file's contribution: reused from cache, or rebuilt.

    Failures are relabelled with the artifact that was in hand when they
    happened. main()'s handler has to stay broad — this can fail with a
    download error, a ``MemoryError`` from ``np.load`` on an 11-23 GB array, a
    PIL encode failure, an environment fault reading the small local part cache,
    or an ENOSPC writing one — and without a name the user cannot tell WHICH
    file to look at. *stage* tracks that name because the three steps touch
    different artifacts: blaming a 23 GB ``Image_data`` download for an
    ``EMFILE`` reading ``thumbs_partNN``, or for an ENOSPC writing it, points at
    the one file the user has no reason to touch — and main()'s advice on that
    failure is to re-run and refetch it.

    The relabelling sits OUTSIDE both helpers on purpose: their own
    quarantine/self-heal logic inspects exception types (see
    :func:`_is_environment_failure`) and has already run — and been given its
    chance to recover — by the time anything reaches here.
    """
    stage = part_path.name
    try:
        part = _reusable_thumbnail_part(
            part_path, filename, global_to_test, global_offset
        )
        if part is not None:
            aprint(
                f"Reusing {len(part[0]):,} cached thumbnails for {filename} "
                f"({position})"
            )
            return part

        stage = filename
        built = _build_thumbnail_part(
            cache_dir,
            filename,
            file_id,
            position,
            global_to_test,
            global_offset,
        )

        stage = part_path.name
        _write_thumbnail_part(part_path, built)
        return built
    except MissingDependencyError:
        # Has its own handler (and its own pip/extra hint) in main();
        # rebranding it as RuntimeError would route it to the download advice
        # instead.
        raise
    except Exception as exc:
        # Keep the original type in the message: main() prints
        # `type(e).__name__`, which is now always RuntimeError, and
        # `str(MemoryError())` is empty — a bare trailing colon.
        raise RuntimeError(f"{stage}: {type(exc).__name__}: {exc}") from exc


def _existing_thumbnail_bundle(
    cache_dir: Path, expected_count: int | None, recompute: bool
) -> list[bytes] | None:
    """The assembled bundle to reuse, or ``None`` once there is none to reuse.

    ``recompute`` skips the READ rather than deleting: this module's whole
    argument — the one that keeps a `MemoryError` from quarantining a healthy
    23 GB download — is that a working artifact is never traded for a
    hypothetical one. A bundle removed before its replacement exists is that
    same mistake, and on a legacy-only cache with no ``Image_data`` files left
    on disk an interrupted rebuild would cost a 181 GB re-download to recover
    exactly what the user already had. A VERSIONED bundle is therefore never
    even opened here, so it always survives to be replaced atomically.

    A pre-versioning one is not quite so untouched, and the asymmetry is worth
    stating: adoption VALIDATES before it renames, and every rejecting branch of
    :func:`_read_bundle_blobs` quarantines. So a legacy bundle that fails the
    count or mapping check is moved aside in this prelude even though the
    rebuild has not run yet. That is the same thing a plain run does to it — the
    validation is one policy with one implementation, and giving the recompute
    path a quieter second copy of that rename is exactly what produced a
    clobbering bug once already.

    Legacy ADOPTION still runs, for its side effect, but ONLY when the
    versioned name is free — the same precedence :func:`_load_cached_thumbnails`
    applies, and for a stronger reason here. Adoption takes the versioned name,
    so running it over a live v1 would put an older unverifiable bundle where a
    fresh fingerprinted one was, before any replacement exists — erasing the
    very fingerprint that would have caught the mismatch, and leaving an
    interrupted run serving thumbnails pasted onto the wrong points. Adoption
    refuses that by itself too; this guard keeps the recompute path from even
    reading a bundle it must not take.

    When the versioned name IS free, adopting is what keeps a pre-versioning
    bundle reachable: the rename leaves exactly one bundle, which the rebuild
    below replaces atomically, and if the rebuild never finishes those bytes sit
    under the canonical name where the normal path picks them up. Skipping it
    would leave the legacy file untouched and unreachable forever, since once a
    v1 exists no later run looks for it again. The un-renameable branch is the
    one case this cannot rescue — the rebuild goes on to write a v1 and the
    leftover legacy file becomes unreachable — but that branch already means a
    cache directory that cannot be renamed in, where the rebuild's own write is
    just as likely to fail.
    """
    if not recompute:
        return _load_cached_thumbnails(cache_dir, expected_count)

    if not (cache_dir / THUMBNAIL_CACHE_NAME).exists():
        _adopt_legacy_bundle(cache_dir, expected_count)

    survivor = _surviving_bundle(cache_dir)
    if survivor is not None:
        # Names the WRITE target rather than promising the survivor is replaced:
        # in the un-renameable branch the survivor is the legacy file and the
        # rebuild writes the versioned name beside it, so "replaces it" would be
        # the one message contradicting the caveat above.
        aprint(
            f"Rebuilding thumbnails (--recompute): ignoring {survivor.name} — "
            f"kept, since nothing is deleted before {THUMBNAIL_CACHE_NAME} is "
            "written"
        )
    return None


def load_cytoself_images(
    cache_dir: Path | None = None,
    expected_count: int | None = None,
    recompute: bool = False,
) -> list[bytes]:
    """Load and encode CytoSelf image crops as WebP thumbnails.

    Downloads the 10 Label_data CSVs to identify which of the ~1.1M image
    crops correspond to the 114K test-split embeddings, then downloads and
    encodes only the matched crops from the Image_data .npy files.

    Work is checkpointed per ``Image_data`` file, so an interrupted run resumes
    where it stopped instead of discarding every file it had already encoded.

    Args:
        cache_dir: Directory for caching downloads and encoded thumbnails.
        expected_count: How many thumbnails the caller needs, one per point. A
            cached bundle holding a different number is unusable (the scene
            builder cannot align it) and is rebuilt rather than returned, and a
            freshly assembled one is not written to disk. ``None`` disables both
            checks.
        recompute: If True, ignore the assembled thumbnail bundle and rebuild
            it. The automatic checks above only catch a bundle that is unusable
            on its face; this is the escape hatch for one that is merely wrong,
            which the cache would otherwise short-circuit on forever. It
            deliberately reaches no further than the bundle: the ``Image_data``
            downloads and the per-file part caches are the expensive artifacts,
            they are each keyed to the current row mapping, and a run that asks
            for a rebuild is promised they are reused — so this costs a
            reassembly, not a 181 GB re-fetch.

    Returns:
        List of WebP-encoded bytes aligned with label.csv / embeddings. If
        fewer than :data:`MIN_MATCH_FRACTION` of the test rows matched a crop
        the thumbnails are still returned, but they are NOT cached: that match
        rate reads as a regression in the row matching, and freezing it would
        make it permanent.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached = _existing_thumbnail_bundle(cache_dir, expected_count, recompute)
    if cached is not None:
        return cached

    # Step 1: Build index mapping (test row -> global image row)
    mapping, n_test = _build_test_index_mapping(cache_dir)

    # Fingerprint the CSVs the mapping was JUST derived from, not whatever is on
    # disk hours later when the bundle is written. The encode pass below runs for
    # hours on the real dataset, and a `Label_data` CSV repaired while it runs
    # would otherwise stamp thumbnails built under the OLD mapping with the NEW
    # mapping's digest — certifying the exact mismatch the digest exists to
    # catch, permanently. Stamping the captured value instead means the next run
    # sees the disagreement and rebuilds.
    mapping_fingerprint = _mapping_fingerprint(cache_dir)

    # Invert: global_index -> list of test_indices (for per-file processing)
    global_to_test: dict[int, list[int]] = {}
    for test_idx, global_idx in mapping.items():
        global_to_test.setdefault(global_idx, []).append(test_idx)

    # Step 2: Process Image_data files, encoding only matched crops
    result_blobs: list[bytes | None] = [None] * n_test
    global_offset = 0
    matched_count = 0

    with asection("Downloading and encoding matched image crops"):
        for i, (filename, file_id) in enumerate(GDRIVE_IMAGE_IDS.items()):
            part_blobs, part_test_indices, n_crops = _thumbnail_part_for(
                cache_dir,
                _thumbnail_part_path(cache_dir, i),
                filename,
                file_id,
                f"{i + 1}/{len(GDRIVE_IMAGE_IDS)}",
                global_to_test,
                global_offset,
            )
            for blob, test_idx in zip(part_blobs, part_test_indices):
                result_blobs[test_idx] = blob
                matched_count += 1
            global_offset += n_crops

        aprint(f"Total matched: {matched_count:,}/{n_test:,}")

    # Count the DISTINCT test rows that got a real crop, straight off the result
    # array rather than from the running counter — that way the figure survives
    # a corrupted offset chain, which is exactly the condition under which it
    # needs to be believed.
    matched_unique = sum(1 for blob in result_blobs if blob is not None)
    match_fraction = matched_unique / n_test if n_test else 0.0
    matching_looks_broken = match_fraction < MIN_MATCH_FRACTION

    # Fill any unmatched slots with a 1x1 transparent placeholder
    import io

    # Optional dep, gated per helper: the soft check in main() lives in a
    # different function, so calling this directly would raise a bare error.
    PILImage = require_module("PIL.Image")

    placeholder = io.BytesIO()
    PILImage.new("RGB", (1, 1), (0, 0, 0)).save(placeholder, format="webp")
    placeholder_bytes = placeholder.getvalue()
    n_filled = 0
    for idx in range(n_test):
        if result_blobs[idx] is None:
            result_blobs[idx] = placeholder_bytes
            n_filled += 1
    if n_filled:
        aprint(f"  Filled {n_filled:,} unmatched slots with placeholder")

    final_blobs: list[bytes] = [b for b in result_blobs if b is not None]

    # Cache the test-aligned thumbnails — unless the match rate says the row
    # matching has regressed, in which case the thumbnails are still returned
    # (a partly-working tooltip beats none) but never frozen onto disk, where
    # they would be reused unquestioned for good.
    wrong_count = expected_count is not None and len(final_blobs) != expected_count
    if matching_looks_broken:
        # This tripwire fires when OUR row matching regressed, which makes an
        # older bundle — assembled back when it worked — the MORE trustworthy of
        # the two. So the new assembly is refused and any existing bundle is left
        # exactly where it is; `_report_bundle_not_cached` says so out loud.
        _report_bundle_not_cached(
            cache_dir,
            f"⚠️  Only {matched_unique:,}/{n_test:,} test rows "
            f"({match_fraction:.1%}) matched an image crop, below the "
            f"{MIN_MATCH_FRACTION:.0%} tripwire — the label row matching looks "
            "broken, so the thumbnails were NOT cached. The per-file caches "
            "were kept, so a retry only redoes the assembly.",
        )
    elif wrong_count:
        # label.csv and the embeddings disagree on how many rows there are, so
        # the scene builder will refuse these thumbnails anyway. Writing the
        # bundle would only hand the next run a cache it has to quarantine —
        # the per-file caches keep the reassembly cheap either way.
        _report_bundle_not_cached(
            cache_dir,
            f"⚠️  Assembled {len(final_blobs):,} thumbnails but the scene needs "
            f"{expected_count:,} — label.csv and the embeddings disagree, so "
            "the thumbnails were NOT cached.",
        )
    else:
        thumbnails_cache = cache_dir / THUMBNAIL_CACHE_NAME
        with asection("Caching test-aligned thumbnails"):
            # Stamped with the mapping these blobs were assembled under — the
            # digest captured before the encode pass, not a fresh one — so a
            # later run whose label CSVs have changed rebuilds instead of
            # pasting every thumbnail onto the wrong point.
            _write_npz_atomic(
                thumbnails_cache,
                blobs=np.array(final_blobs, dtype=object),
                mapping_fingerprint=np.array(mapping_fingerprint),
            )
            aprint(f"Cached {len(final_blobs):,} thumbnails to {thumbnails_cache}")

    return final_blobs


# =============================================================================
# Scene Construction
# =============================================================================


def _resolve_image_labels(
    image_labels: list[bytes] | None,
    *,
    n_points: int,
    n_views: int,
    images_expected: bool,
) -> list[bytes] | None:
    """Replicate the hover thumbnails per attribute view, or explain their absence.

    Only used if the count matches the embeddings — the image .npy files may
    contain more crops than the embedding/label rows. Returns None whenever the
    scene has to fall back to the text-only hover tooltip.
    """
    if image_labels is None:
        if images_expected:
            # The caller (main()) has already explained WHY they are missing
            # and how to get them back — only state the consequence here so
            # the advice is printed exactly once.
            aprint("  ⚠ No image labels — hover tooltip will be text-only")
        else:
            aprint("  Building without hover thumbnails (--without-images)")
        return None

    if len(image_labels) == n_points:
        return image_labels * n_views

    aprint(
        f"  ⚠ Skipping image labels: count mismatch "
        f"({len(image_labels):,} images vs {n_points:,} embeddings)"
    )
    aprint("    Hover falls back to the text-only tooltip. A run through")
    aprint("    main() hands the point count to load_cytoself_images, which")
    aprint("    rebuilds a stale bundle by itself — so reaching here means")
    aprint("    label.csv and the embeddings themselves disagree, and no")
    aprint("    rebuild will fix it. A caller that passed no expected_count")
    aprint("    can force one by deleting the bundle and re-running:")
    # The thumbnails may have come from a caller-supplied cache_dir, so name
    # the file and mark the directory as the default rather than asserting a
    # path this function cannot know. BOTH names are offered: adoption renames
    # the legacy bundle onto the versioned one, but a rename that fails leaves
    # the legacy file serving the blobs, so naming only the v1 one would send
    # that user to delete a file that is not in effect.
    aprint(f"      {THUMBNAIL_CACHE_NAME} — or {LEGACY_THUMBNAIL_CACHE_NAME},")
    aprint(f"      whichever is present, in {DEFAULT_CACHE_DIR} by default")
    aprint("    (--recompute does the same, but also throws away the")
    aprint("    cached UMAP: a 10-30 min recompute.)")
    return None


def _protein_link_attrs(
    attributes: dict,
    category_maps: dict | None,
    available_attrs: list[str],
    n_points: int,
) -> dict[str, object]:
    """Build aligned Human Protein Atlas link attributes when names exist."""
    if not category_maps or "protein_name" not in available_attrs:
        return {}
    names = category_maps.get("protein_name", [])
    if not names:
        return {}
    per_cell_keys = []
    for i in range(n_points):
        code = int(attributes["protein_name"][i])
        per_cell_keys.append(str(names[code]) if 0 <= code < len(names) else "")
    return {
        "keys": per_cell_keys * len(available_attrs),
        "link": "https://www.proteinatlas.org/search/{hover_key}",
        "copy": "{hover_key}",
    }


def create_cytoself_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
    category_maps: dict | None = None,
    image_labels: list[bytes] | None = None,
    *,
    images_expected: bool = True,
) -> int:
    """Create Luxar scene with categorical attribute visualization.

    Args:
        output_path: Where to write Luxar zarr
        coordinates: (N, 3) UMAP coordinates
        attributes: Dict of attribute arrays
        category_maps: Dict of attribute name -> list of category labels
        image_labels: Optional list of WebP-encoded image blobs (one per point)
        images_expected: Whether the caller INTENDED to supply thumbnails.
            False for a deliberate ``--without-images`` run, which then builds
            quietly instead of warning about an absence the user asked for.

    Returns:
        Number of points
    """
    n_points = len(coordinates)

    with asection("Building Multi-Attribute Scene"):
        attr_types = ["localization", "protein_name"]
        category_labels = ["Localization", "Protein"]

        # Create one copy of points per attribute type
        all_positions = []
        all_colors = []
        available_attrs = [name for name in attr_types if name in attributes]

        for attr_idx, attr_name in enumerate(available_attrs):
            colors = attribute_to_color(attributes[attr_name], attr_name)

            positions_4d = np.column_stack(
                [
                    np.full(n_points, attr_idx, dtype=np.float32),
                    coordinates[:, 0],
                    coordinates[:, 1],
                    coordinates[:, 2],
                ]
            )

            all_positions.append(positions_4d)
            all_colors.append(colors)

            n_unique = len(np.unique(attributes[attr_name]))
            aprint(f"  Attribute {attr_idx} ({attr_name}): {n_unique} unique values")

        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"Created {len(available_attrs)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Filter category_labels to match available attributes
        available_labels = [
            label
            for label, name in zip(category_labels, attr_types)
            if name in attributes
        ]

        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=available_labels,
                    display=False,
                    description="Color coding attribute for CytoSelf embeddings",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(cinematic_mode=True),
            )

            total_points = len(positions_combined)
            radii = np.full(total_points, POINT_RADIUS, dtype=np.float32)
            sharpnesses = np.full(total_points, 0.6, dtype=np.float32)

            # Hover labels: resolve codes to category names, repeated per view
            per_cell_labels = []
            if category_maps:
                for i in range(n_points):
                    parts = []
                    for attr_name in available_attrs:
                        code = int(attributes[attr_name][i])
                        cats = category_maps.get(attr_name, [])
                        name = str(cats[code]) if 0 <= code < len(cats) else str(code)
                        parts.append(name)
                    per_cell_labels.append("\n".join(parts))
            labels = per_cell_labels * len(available_attrs) if per_cell_labels else None

            # Click a cell to open its protein in the Human Protein Atlas,
            # right-click to copy the name (#1917). The Atlas rather than a gene
            # database because this demo IS subcellular localization, and that
            # is the page which shows it.
            #
            # The label joins every available attribute with newlines —
            # localization and protein name together — so the URL needs the bare
            # name from `keys=`. Tiled per view exactly like the labels: the
            # protein a cell shows does not change with the active attribute.
            link_attrs = _protein_link_attrs(
                attributes, category_maps, available_attrs, n_points
            )

            all_image_labels = _resolve_image_labels(
                image_labels,
                n_points=n_points,
                n_views=len(available_attrs),
                images_expected=images_expected,
            )

            scene.add_points(
                "Images",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
                intensity=0.18,
                labels=labels,
                image_labels=all_image_labels,
                **link_attrs,
                layer=True,
            )

            # --- Overlays ---
            scene.add_text(
                "CytoSelf Protein Localization UMAP",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            attr_labels = available_labels
            attr_keys = available_attrs
            for attr_id, (label, attr_key) in enumerate(zip(attr_labels, attr_keys)):
                scene.add_text(
                    f"Colored by: {label}",
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"attribute": attr_id},
                    transition="fade",
                    transition_duration=0.2,
                )

                if category_maps and attr_key in category_maps:
                    legend_html = build_legend_html(
                        attr_key, category_maps[attr_key], attributes.get(attr_key)
                    )
                    if legend_html:
                        scene.add_html(
                            legend_html,
                            position=(0.98, 0.5),
                            anchor="center-right",
                            opacity=0.9,
                            visible_range={"attribute": attr_id},
                            transition="fade",
                            transition_duration=0.2,
                        )

            # Custom hover overlays. Defining ANY hover=True overlay
            # suppresses the auto-injected default, so this block owns the
            # whole hover layout and has to cover BOTH shapes:
            #
            #  * WITH thumbnails — the bespoke two-panel layout: the image
            #    panel top-right at 0.98, and the text label at x=0.82 so it
            #    sits immediately to its LEFT (two lines via the \n separator
            #    in the labels).
            #  * WITHOUT thumbnails — x=0.82 would leave the text floating
            #    beside an image panel that does not exist, and the injected
            #    default is the same top-right corner, only 16% of the
            #    viewport further into it. So the label moves to the
            #    centre-left slot, which no other overlay in this scene uses
            #    (the legend is center-RIGHT at 0.98/0.5) and which is the
            #    house convention for a text-only hover tooltip: parameters
            #    copied from demo_chromatrace_choir_umap.py, whose overlay
            #    layout is otherwise identical to this one. Note the viewer's
            #    control rail is docked at the same left-centre edge and
            #    paints above the overlay layer, so the first glyph or two
            #    can sit behind it — house-wide, and matching the siblings
            #    beats diverging from them.
            if all_image_labels is not None:
                scene.add_html(
                    "{hover_image_label}",
                    position=(0.98, 0.02),
                    anchor="top-right",
                    opacity=1.0,
                    transition="fade",
                    transition_duration=0.15,
                    hover=True,
                    hover_image_size=(0.15, 0.20),
                )
                if labels is not None:
                    scene.add_text(
                        "{hover_label}",
                        position=(0.82, 0.02),
                        anchor="top-right",
                        font_size=0.018,
                        color="white",
                        width=0.12,
                        text_align="right",
                        background="rgba(0,0,0,0.7)",
                        padding=0.008,
                        opacity=1.0,
                        transition="fade",
                        transition_duration=0.15,
                        hover=True,
                    )
            elif labels is not None:
                scene.add_text(
                    "{hover_label}",
                    position=(0.02, 0.5),
                    anchor="center-left",
                    font_size=0.022,
                    color="white",
                    background="rgba(0,0,0,0.72)",
                    padding=0.01,
                    text_align="left",
                    opacity=1.0,
                    transition="fade",
                    transition_duration=0.15,
                    hover=True,
                )

            add_demo_caption(
                scene,
                f"{n_points:,} images \u2022 OpenCell \u2022 3D UMAP \u2022 Kobayashi et al., Nat Methods 2022",
                DEMO_META.get("citation"),
            )

        aprint(f"Scene created with {n_points:,} points")

    return n_points


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("CYTOSELF PROTEIN LOCALIZATION LANDSCAPE")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing ~114K fluorescence microscopy image embeddings")
    aprint("from the OpenCell proteome-scale dataset.")
    aprint("")
    aprint("Each point is a single cell crop, positioned by CytoSelf's")
    aprint("self-supervised VQ-VAE-2 embedding, reduced to 3D via UMAP.")
    aprint("Points cluster by subcellular localization.")
    aprint("")
    aprint("References:")
    aprint("  CytoSelf: Kobayashi et al., Nature Methods 2022")
    aprint("  OpenCell: Cho et al., Science 2022")
    aprint("  GitHub:   https://github.com/royerlab/cytoself")
    aprint("")
    aprint("NOTE: First run downloads 185.8 GB — 4.23 GB of embeddings, 71 MB")
    aprint("      of label CSVs, and ten Image_data files of 11.3-23.6 GB each")
    aprint("      (181.5 GB) for the hover thumbnails — and computes UMAP")
    aprint("      (~10-30 min). Needs ~190 GB of free disk — the downloads stay")
    aprint("      cached, and the encoded thumbnails (per source file, plus the")
    aprint("      assembled bundle and its staging copy) sit alongside them —")
    aprint("      and a 32 GB machine, since the thumbnail pass reads one whole")
    aprint("      Image_data archive at a time (the largest is 23.6 GB) and the")
    aprint("      crops, encoded thumbnails and label tables sit on top of it.")
    aprint("      --without-images needs ~16 GB, for the UMAP.")
    aprint("      Subsequent runs load from cache; an interrupted run resumes")
    aprint("      per file. Use --without-images for the ~4.24 GB run.")
    aprint("")

    recompute = "--recompute" in sys.argv
    without_images = "--without-images" in sys.argv

    # Load data (downloads + UMAP on first run, cached thereafter)
    coordinates, attributes, category_maps = load_cytoself_data(
        recompute=recompute,
    )

    # Load image labels (unless opted out)
    image_labels: list[bytes] | None = None
    if not without_images:
        try:
            require_module("PIL.Image")
            # The point count is what makes a cached bundle usable or not, so
            # pass it in: a bundle with a different blob count is rebuilt rather
            # than silently costing every future run its hover images.
            image_labels = load_cytoself_images(
                expected_count=len(coordinates),
                recompute=recompute,
            )
        except MissingDependencyError as exc:
            # `exc` already names the package and the extra that provides it,
            # and a re-run without it fails identically — so no other advice.
            aprint(f"WARNING: hover thumbnails unavailable — {exc}")
        except Exception as e:
            # Broad on purpose: a download/decode failure must not kill the
            # demo. But it must be honest about WHAT failed and how to fix it.
            aprint(
                f"WARNING: failed to load hover thumbnails — {type(e).__name__}: {e}"
            )
            aprint("  Re-running SKIPS whole files that already finished")
            aprint(f"  downloading under {DEFAULT_CACHE_DIR}, so only the")
            aprint("  missing Image_data*.npy is refetched (a file that died")
            aprint("  part-way is refetched from the start — there is no")
            aprint("  byte-level resume). The WebP encodes are checkpointed per")
            aprint("  source file too, so only the file that failed is redone.")
            aprint("  Pass --without-images to skip the images deliberately.")
    else:
        aprint("Skipping image labels (--without-images)")

    # Generate legend images
    generate_all_legends(
        attributes,
        category_maps,
        prefix="cytoself",
        attr_display_names={
            "localization": "Localization",
            "protein_name": "Protein",
        },
    )

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "cytoself_protein_landscape.luxar.zarr"
        _n_points = create_cytoself_scene(
            output_path,
            coordinates,
            attributes,
            category_maps,
            image_labels=image_labels,
            images_expected=not without_images,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_cytoself_") as tmpdir:
        output_path = Path(tmpdir) / "cytoself_landscape.luxar.zarr"

        _n_points = create_cytoself_scene(
            output_path,
            coordinates,
            attributes,
            category_maps,
            image_labels=image_labels,
            images_expected=not without_images,
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate to explore UMAP structure")
        aprint("  - Zoom in to see individual images")
        # Mirror create_cytoself_scene's count-mismatch guard: thumbnails only
        # made it into the scene if they were loaded AND aligned 1:1.
        if image_labels is not None and len(image_labels) == len(coordinates):
            aprint("  - Hover over a point to see its fluorescence image")
        else:
            aprint("  - Hover over a point to see its localization and protein")
        aprint("")
        aprint("  Press '1' to select ATTRIBUTE VIEW, then use [/]:")
        aprint("     0: Localization (subcellular compartment)")
        aprint("     1: Protein (~1,311 unique proteins)")
        aprint("")
        aprint("  Same structure, different colors reveal different biology!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
