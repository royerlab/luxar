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

Data source: OpenCell / CytoSelf (CC BY 4.0)
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
}

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
from luxar.demos import (
    MissingDependencyError,
    cache_computed,
    launch_viewer,
    require_module,
)
from luxar.utils._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.download import quarantine_file
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

# Per-artifact size floors. Exact sizes, from HTTP Range probes of the Drive
# files themselves (185,838,193,451 B for the whole set):
#   Global_representation.npy  4,232,208,512 B                       (4.23 GB)
#   label.csv                      6,553,361 B                       (6.55 MB)
#   Label_data*.csv            3,950,974 - 8,742,330 B each   (64.7 MB total)
#   Image_data*.npy      11,282,720,128 - 23,603,040,128 each (181.5 GB total)
# Both users of these floors compare against `floor * 0.9` (the legacy
# sidecar-less cache check, and the promotion gate for a stream that declared no
# content-length), so each floor is chosen with that factor in mind: the 0.9 bar
# must land BELOW the smallest real file of its kind — with margin, since these
# are the sizes today and a re-upload could shrink them slightly — while still
# being high enough to refuse a large fragment. Guessing low is what let a
# 500 MB fragment of a 23.6 GB file pass; guessing high would reject the real
# thing and brick the demo, so the smallest file in each family is what matters.
MIN_SIZE_EMBEDDINGS = 4_000_000_000
MIN_SIZE_LABELS_CSV = 5_000_000
MIN_SIZE_LABEL_DATA_CSV = 3_000_000
MIN_SIZE_IMAGE_DATA = 10_000_000_000

# Per-Image_data thumbnail part caches and the assembled test-aligned bundle.
# Both are versioned so a change to the encoding invalidates them by name.
# The unversioned bundle is what the pre-versioning code wrote; its contents are
# byte-identical to v1, so it is adopted rather than rebuilt (see
# `load_cytoself_images`).
THUMBNAIL_PART_TEMPLATE = "thumbs_part{index:02d}_v1.npz"
THUMBNAIL_CACHE_NAME = "image_labels_test_webp_v1.npz"
LEGACY_THUMBNAIL_CACHE_NAME = "image_labels_test_webp.npz"

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
    :func:`luxar.utils.download.robust_download`, which is nested and so cannot
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

    Without a sidecar (a cache written before staging existed) we fall back to
    the ``size > expected_min_size * 0.9`` heuristic. The floors it reads are now
    the measured ones, but the rule stays a floor rather than an equality: the
    exact sizes are today's, and pinning a legacy cache to them would re-download
    gigabytes the moment a file is re-uploaded a byte different. The residual
    risk — a legacy file truncated by less than 10% — is covered on the consumer
    side by :func:`_load_downloaded_artifact`, which quarantines an unreadable
    artifact and re-downloads it once.
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
    """
    _sidecar_path(path).unlink(missing_ok=True)
    try:
        quarantine_file(path, reason=reason)
    except FileNotFoundError:
        # Already gone — either never written, or a concurrent run quarantined
        # the same artifact between our check and theirs. Either way it is out
        # from under the canonical name, which is all this needs to guarantee.
        pass


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
    about the bytes on disk are re-raised untouched: ``MemoryError`` (the
    embeddings need ~16 GB of RAM, so re-downloading 4.23 GB would fail
    identically) and ``ImportError`` (a reader reaching for an engine that is
    not installed — ``pd.read_csv`` does this — is a broken environment, not a
    broken file).
    """
    try:
        return loader(path)
    except (MemoryError, ImportError):
        raise
    except Exception as exc:
        aprint(f"  ⚠ Cached file {path.name} could not be read ({exc})")
        _quarantine_download(path, reason="unreadable cached download")
        _download_from_google_drive(file_id, path, expected_min_size=expected_min_size)
        return loader(path)


def _download_from_google_drive(
    file_id: str, output_path: Path, expected_min_size: int = 0
) -> Path:
    """Download a file from Google Drive, handling the virus-scan confirmation.

    Only the confirmation dance below is genuinely custom — Google Drive's
    four-way "are you sure" flow has no equivalent in
    :mod:`luxar.utils.download`, which is why ``robust_download`` cannot be used
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
    # arrived intact. `luxar.utils.download._force_identity_encoding` codifies
    # this precondition, but its "unless the caller already set it" guard would
    # preserve the Session's own gzip default, so the header is set outright.
    # Setting it on the session makes all four strategies below inherit it.
    session.headers["Accept-Encoding"] = "identity"

    with asection(f"Downloading {output_path.name} from Google Drive"):
        aprint(f"File ID: {file_id}")

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

        response.raise_for_status()

        total_size = _parse_content_length(response.headers)
        if total_size is not None:
            aprint(f"Download size: {total_size / (1024**2):.1f} MB")

        downloaded = 0
        last_report_mb = 0
        start_time = time.time()
        chunk_size = 1024 * 1024  # 1 MB

        with open(part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    progress_mb = downloaded / (1024 * 1024)
                    if progress_mb - last_report_mb >= 100:
                        elapsed = time.time() - start_time
                        rate = (
                            downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                        )
                        if total_size is not None:
                            pct = downloaded / total_size * 100
                            aprint(
                                f"  {downloaded / (1024**3):.2f} / "
                                f"{total_size / (1024**3):.2f} GB "
                                f"({pct:.0f}%) - {rate:.1f} MB/s"
                            )
                        else:
                            aprint(
                                f"  {downloaded / (1024**3):.2f} GB - {rate:.1f} MB/s"
                            )
                        last_report_mb = progress_mb

        final_size = part_path.stat().st_size
        aprint(f"Download complete: {final_size / (1024**2):.1f} MB")

        # --- Verify the staged bytes before they take the destination name ---

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

        # The expected-size floor is checked whether or not a length was
        # declared. An honest content-length only proves the body arrived whole,
        # not that it is the body we asked for: Drive also serves small
        # non-HTML "cannot access this file" responses, which agree with their
        # own length and would otherwise be promoted AND certified by a sidecar
        # — where the old heuristic at least re-fetched them every run.
        if final_size <= expected_min_size * 0.9:
            raise _reject_staged_download(
                part_path,
                f"Download is too short: got {final_size} bytes, well under the "
                f"{expected_min_size} bytes expected for this file.",
                file_id,
                output_path,
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
            expected_min_size=MIN_SIZE_EMBEDDINGS,
        )
        _download_from_google_drive(
            GDRIVE_LABELS_ID,
            labels_path,
            expected_min_size=MIN_SIZE_LABELS_CSV,
        )

    # --- Extract attributes from labels ---
    with asection("Processing Labels"):
        df = _load_downloaded_artifact(
            labels_path,
            pd.read_csv,
            GDRIVE_LABELS_ID,
            expected_min_size=MIN_SIZE_LABELS_CSV,
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
                expected_min_size=MIN_SIZE_EMBEDDINGS,
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
                file_id, csv_path, expected_min_size=MIN_SIZE_LABEL_DATA_CSV
            )
            df = _load_downloaded_artifact(
                csv_path,
                lambda p: pd.read_csv(p, header=None),
                file_id,
                expected_min_size=MIN_SIZE_LABEL_DATA_CSV,
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
            expected_min_size=MIN_SIZE_LABELS_CSV,
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
    """
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=True) as data:
            blobs = [bytes(b) for b in data["blobs"]]
            test_indices = [int(i) for i in data["test_indices"]]
            n_crops = int(data["n_crops"])
        if len(blobs) != len(test_indices):
            raise ValueError("blobs and test_indices have different lengths")
    except Exception as exc:
        aprint(f"  ⚠ Thumbnail part cache unreadable ({exc}) — rebuilding")
        _quarantine_download(path, reason="unreadable thumbnail part cache")
        return None

    return blobs, test_indices, n_crops


def load_cytoself_images(
    cache_dir: Path | None = None,
) -> list[bytes]:
    """Load and encode CytoSelf image crops as WebP thumbnails.

    Downloads the 10 Label_data CSVs to identify which of the ~1.1M image
    crops correspond to the 114K test-split embeddings, then downloads and
    encodes only the matched crops from the Image_data .npy files.

    Work is checkpointed per ``Image_data`` file, so an interrupted run resumes
    where it stopped instead of discarding every file it had already encoded.

    Args:
        cache_dir: Directory for caching downloads and encoded thumbnails.

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

    # Check for cached encoded thumbnails (test-aligned)
    thumbnails_cache = cache_dir / THUMBNAIL_CACHE_NAME
    if thumbnails_cache.exists():
        with asection("Loading cached image thumbnails"):
            try:
                with np.load(thumbnails_cache, allow_pickle=True) as data:
                    blobs = [bytes(b) for b in data["blobs"]]
                aprint(f"Loaded {len(blobs):,} cached thumbnails (test-aligned)")
                return blobs
            except Exception as exc:
                # A truncated bundle used to crash every run until the user
                # deleted it by hand; quarantine it and rebuild instead.
                aprint(f"  ⚠ Cached thumbnail bundle unreadable ({exc}) — rebuilding")
                _quarantine_download(
                    thumbnails_cache, reason="unreadable thumbnail bundle"
                )

    # Adopt a bundle written before the name carried a version. The v1 contents
    # are byte-identical to the unversioned ones, so renaming is sound and
    # spares an existing user a multi-gigabyte re-download for a pure rename.
    # The literal below is the enforcement, not a note: bump the encoding to v2
    # and this stops matching, so the stale bundle is rebuilt instead of
    # laundered into the new name.
    legacy_cache = cache_dir / LEGACY_THUMBNAIL_CACHE_NAME
    if (
        THUMBNAIL_CACHE_NAME == "image_labels_test_webp_v1.npz"
        and legacy_cache.exists()
    ):
        with asection("Adopting pre-versioning thumbnail cache"):
            try:
                with np.load(legacy_cache, allow_pickle=True) as data:
                    blobs = [bytes(b) for b in data["blobs"]]
                legacy_cache.replace(thumbnails_cache)
            except Exception as exc:
                # Unreadable, or un-renameable (an open handle on Windows).
                # Either way, fall through and rebuild.
                aprint(
                    f"  ⚠ Legacy thumbnail bundle not adoptable ({exc}) — rebuilding"
                )
                _quarantine_download(
                    legacy_cache, reason="unusable legacy thumbnail bundle"
                )
            else:
                aprint(
                    f"Adopted {len(blobs):,} thumbnails from "
                    f"{LEGACY_THUMBNAIL_CACHE_NAME} as {THUMBNAIL_CACHE_NAME}"
                )
                return blobs

    # Step 1: Build index mapping (test row -> global image row)
    mapping, n_test = _build_test_index_mapping(cache_dir)

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
            img_path = cache_dir / filename
            part_path = _thumbnail_part_path(cache_dir, i)

            # A part cache holds everything this file contributes, INCLUDING its
            # crop count — the loop advances `global_offset` by that count, so a
            # skipped file must still move the offset or every later file would
            # match the wrong rows.
            part = _load_thumbnail_part(part_path)
            if part is not None and part[1] != _expected_test_indices(
                global_to_test, global_offset, part[2]
            ):
                # The part was built against a different row mapping — a repaired
                # Label_data CSV shifts it without changing its length. Reusing
                # it would paste every blob onto the wrong row and then freeze
                # that into the bundle, so rebuild instead.
                aprint(f"  ⚠ Stale thumbnail part cache for {filename} — rebuilding")
                _quarantine_download(part_path, reason="stale thumbnail part cache")
                part = None
            if part is not None:
                part_blobs, part_test_indices, n_crops = part
                for blob, test_idx in zip(part_blobs, part_test_indices):
                    result_blobs[test_idx] = blob
                    matched_count += 1
                aprint(
                    f"Reusing {len(part_blobs):,} cached thumbnails for {filename} "
                    f"({i + 1}/{len(GDRIVE_IMAGE_IDS)})"
                )
                global_offset += n_crops
                continue

            _download_from_google_drive(
                file_id, img_path, expected_min_size=MIN_SIZE_IMAGE_DATA
            )

            with asection(f"Processing {filename} ({i + 1}/{len(GDRIVE_IMAGE_IDS)})"):
                arr = _load_downloaded_artifact(
                    img_path, np.load, file_id, expected_min_size=MIN_SIZE_IMAGE_DATA
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
                    subset = arr[unique_local]
                    encoded = _encode_crops_to_webp(subset)

                    # Map encoded blobs back to test indices
                    local_to_encoded = {li: ei for ei, li in enumerate(unique_local)}
                    for local_idx, test_idx in local_to_test_map:
                        encoded_idx = local_to_encoded[local_idx]
                        result_blobs[test_idx] = encoded[encoded_idx]
                        new_blobs.append(encoded[encoded_idx])
                        new_test_indices.append(test_idx)
                        matched_count += 1

                    aprint(
                        f"Encoded {len(unique_local):,} matched crops "
                        f"(of {n_crops:,} total)"
                    )
                else:
                    aprint(f"No matched crops in this file ({n_crops:,} total)")

                # Checkpoint this file's contribution before moving on, so a
                # failure on a later file costs only that file's work.
                _write_npz_atomic(
                    part_path,
                    blobs=np.array(new_blobs, dtype=object),
                    test_indices=np.asarray(new_test_indices, dtype=np.int64),
                    n_crops=np.int64(n_crops),
                )

                global_offset += n_crops
                del arr

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
    if matching_looks_broken:
        aprint(
            f"⚠️  Only {matched_unique:,}/{n_test:,} test rows "
            f"({match_fraction:.1%}) matched an image crop, below the "
            f"{MIN_MATCH_FRACTION:.0%} tripwire — the label row matching looks "
            "broken, so the thumbnails were NOT cached. The per-file caches "
            "were kept, so a retry only redoes the assembly."
        )
    else:
        with asection("Caching test-aligned thumbnails"):
            _write_npz_atomic(
                thumbnails_cache, blobs=np.array(final_blobs, dtype=object)
            )
            aprint(f"Cached {len(final_blobs):,} thumbnails to {thumbnails_cache}")

    return final_blobs


# =============================================================================
# Scene Construction
# =============================================================================


def create_cytoself_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
    category_maps: dict | None = None,
    image_labels: list[bytes] | None = None,
) -> int:
    """Create Luxar scene with categorical attribute visualization.

    Args:
        output_path: Where to write Luxar zarr
        coordinates: (N, 3) UMAP coordinates
        attributes: Dict of attribute arrays
        category_maps: Dict of attribute name -> list of category labels
        image_labels: Optional list of WebP-encoded image blobs (one per point)

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
            scene = compiler.create_scene(dimensions=dims)

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
                        name = str(cats[code]) if code < len(cats) else str(code)
                        parts.append(name)
                    per_cell_labels.append("\n".join(parts))
            labels = per_cell_labels * len(available_attrs) if per_cell_labels else None

            # Image labels: replicate per attribute view (same as text labels).
            # Only use if count matches embeddings — the image .npy files may
            # contain more crops than the embedding/label rows.
            all_image_labels = None
            if image_labels is not None:
                if len(image_labels) == n_points:
                    all_image_labels = image_labels * len(available_attrs)
                else:
                    aprint(
                        f"  ⚠ Skipping image labels: count mismatch "
                        f"({len(image_labels):,} images vs {n_points:,} embeddings)"
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

            # Custom hover overlays: image top-right, text to its left
            # (two lines via \n separator in labels). Defining these
            # suppresses the auto-injected default hover overlay.
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

            scene.add_text(
                f"{n_points:,} images \u2022 OpenCell \u2022 3D UMAP \u2022 Kobayashi et al., Nat Methods 2022",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
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
    aprint("      (~10-30 min). Needs ~16 GB RAM and ~186 GB of free disk.")
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
            image_labels = load_cytoself_images()
        except MissingDependencyError as exc:
            aprint(f"WARNING: skipping image labels — {exc}")
        except Exception as e:
            aprint(f"WARNING: Failed to load images — skipping: {e}")
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
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate to explore UMAP structure")
        aprint("  - Zoom in to see individual images")
        aprint("  - Hover over a point to see its fluorescence image")
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
