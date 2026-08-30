#!/usr/bin/env python3
"""Self-Contained Demo: arXiv / bioRxiv / medRxiv Paper Embeddings (Kaggle)

Visualize the WHOLE preprint corpus of the Kaggle "openai-arxiv-embeddings"
dataset in 3D — 3,286,365 papers with pre-computed OpenAI embeddings.

================================================================================
DATASET
================================================================================

Source: https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings

`vectors.dat` is 40,382,853,120 bytes of unit-norm ``float32[3072]`` rows —
exactly 3,286,365 of them, one per paper, from `text-embedding-3-large`.
`papers.csv` names them, in ascending-ID order, and records which preprint
server each came from:

    arXiv    2,902,228
    bioRxiv    308,367
    medRxiv     75,770

…through 2025-12. Titles and categories come from the separate Cornell arXiv
metadata snapshot (~1.8 GB), also on Kaggle; papers it does not cover fall back
to their preprint server (`biorxiv` / `medrxiv`), which is a real category
rather than a grey "other".

Dates do NOT come from the snapshot, which records `update_date` — the LAST
REVISION. Nothing in it predates 2007, so reading years from it would paint the
400,803 papers submitted from 1991 through 2006 as 2007-or-later and squeeze 35
years of arXiv into 19. The year is instead decoded from the identifier
itself: `0704.0001` and `hep-lat/0506004` both carry their submission month, and
a bioRxiv/medRxiv DOI carries a full date. Span: **1991 to 2025**.

HOW THE FULL CORPUS FITS IN MEMORY
==================================

It does not — `N x 3072 x 4 B` is 40 GB — so it is never held. `vectors.dat` is
streamed out of the ZIP in blocks (measured 204 MiB/s, ~3 min for the whole
thing) and projected on the fly through a PCA basis down to `--pca-dim`
(default 128, capturing ~58% of the variance) fitted on a 300k-row uniform
subsample. Only that `(N, 128) float32` matrix — 1.6 GB — is cached and handed
to UMAP. The PCA cache is keyed on `--pca-dim` alone, so the 3072-D vectors are
not re-read; `papers.csv` still comes from the ZIP, so keep the archive.

DOWNLOAD & CACHING
==================

FIRST RUN (one-time) — budget ~39 GB of disk, not the ~32 GB downloaded:
  * embeddings ZIP     30.4 GB -> ~/.cache/luxar/arxiv_embeddings.zip
  * metadata ZIP        1.7 GB -> ~/.cache/luxar/arxiv_kaggle/arxiv-metadata.zip
  * metadata JSON       4.7 GB -> ~/.cache/luxar/arxiv_metadata.json  (extracted)
  * metadata lookup     0.3 GB -> ~/.cache/luxar/arxiv_metadata_lookup.pkl
  * PCA matrix          1.6 GB -> ~/.cache/luxar/arxiv_kaggle/pca128_all.npy
  No Kaggle credentials are needed — both are public dataset URLs.
  `luxar demo cache clear arxiv_papers_kaggle` reclaims the ~3.6 GB under the
  `arxiv_kaggle` namespace; the embeddings ZIP and the two metadata files sit
  at the cache ROOT rather than inside it, so they survive and must be deleted
  by hand.

SUBSEQUENT RUNS
  * warm `arxiv_kaggle` bundle -> seconds
  * warm PCA cache, new UMAP   -> minutes (GPU) to hours (CPU, full corpus)
  When using `hatch run`, override its one-thread defaults for CPU work, e.g.
  `OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 hatch run python ...`.

Usage:
    python demo_arxiv_embeddings_kaggle.py [OPTIONS]

    --sample=N        Uniform RANDOM sample of N papers (default: the whole
                      corpus). Random, not a prefix: `papers.csv` is sorted by
                      ID, so a prefix is a date slice, not a sample. Spell the
                      whole corpus `all`, not its size — the bundle cache is
                      keyed on the spelling, so the two cache separately.
    --seed=S          Seed for that sample (default 0), and for fitting a cold
                      PCA basis. A warm `pca<dim>_*` cache is reused regardless.
    --pca-dim=D       PCA components fed to UMAP (default 128).
    --device=auto|cpu|gpu
                      `gpu` runs cuML's UMAP (RAPIDS) when importable; `auto`
                      uses it if present, else umap-learn on the CPU.

Requirements:
    - Install: pip install 'luxar[demos]'   # umap-learn, scikit-learn
    - Optional: pip install 'luxar[gsplats]'   # torch + scipy, for Points LOD
      coarsening; without it the scene is a flat (fully viewable) point cloud
    - Optional: a RAPIDS cuML install, for GPU UMAP on the full corpus

Controls:
    - Explore clusters of related research
    - Color = arXiv category / preprint server, or publication year
    - Size = recency (newer papers larger; undated papers use the midpoint)
    - Click a paper to open it (via doi.org, so arXiv, bioRxiv and medRxiv all
      resolve); right-click to copy its DOI
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "arxiv_papers_kaggle",
    "title": "arXiv Papers (Kaggle / OpenAI)",
    "description": "All 3.29M arXiv/bioRxiv/medRxiv preprints embedded with OpenAI text-embedding-3-large, shown as a 3D UMAP.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 32000,
        "compute": "heavy",
        "gpu": "optional",
        "local_data": "kaggle-auth",
    },
    "caches": ["arxiv_kaggle"],
    "outputs": ["arxiv_papers_kaggle"],
    "citation": {
        "short": "arXiv metadata by Cornell University; embeddings by tomtum",
        "ref": "Cornell / tomtum",
        "url": "https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings",
    },
}

import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cache_computed,
    cached_download,
    hsv_to_rgb,
    launch_viewer,
    require_module,
    stack_colorings,
)
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

#: ``None`` = the whole corpus. ``--sample=N`` takes a uniform RANDOM subset.
DEFAULT_SAMPLE_SIZE: int | None = None

#: Width of one ``vectors.dat`` row (OpenAI ``text-embedding-3-large``).
EMBEDDING_DIM = 3072

#: PCA components fed to UMAP. 128 captures ~58% of the variance of the full
#: corpus (64 -> ~45%, 32 -> ~34%), and shrinks the UMAP working set from
#: ``N x 3072 x 4 B`` (40 GB at full scale) to ``N x 128 x 4 B`` (1.6 GB).
DEFAULT_PCA_DIM = 128

#: Rows drawn uniformly at random to FIT the PCA basis. 300k x 3072 float32 is
#: 3.7 GB, held once — that is what bounds the REDUCTION stage, which is the
#: point of streaming. It is not the pipeline's peak: the Cornell lookup dict is
#: ~3 GB and the stacked per-point hover labels are larger again, so a
#: whole-corpus build was measured at ~12 GB RSS overall.
PCA_FIT_ROWS = 300_000

#: Rows decoded per read from the ZIP. 8192 x 3072 x 4 B = 100 MB per block.
STREAM_BLOCK_ROWS = 8192

# Per-paper radius ramp (older -> newer), expressed as MULTIPLES OF THE MEASURED
# MEDIAN NEAREST-NEIGHBOUR DISTANCE of the reduced cloud rather than as scene
# units. A constant cannot be right at two sizes: median NN spacing falls
# roughly as N^(-1/3), so the 500k cloud's measured 0.021 becomes ~0.011 over the
# full 3.29M corpus and any fixed radius is then ~1.9x too large. The failure is
# not subtle — an earlier fixed ramp ran 2.5x the spacing, one sphere covered a
# median of 9 papers (p90 38), and the cloud clipped to white, making the
# category colours this demo exists to show unreadable at every zoom. These two
# fractions reproduce the hand-tuned 500k ramp exactly (0.005/0.021 = 0.238,
# 0.016/0.021 = 0.762) and now hold at any N.
RADIUS_MIN_NN_FRACTION = 0.238
RADIUS_MAX_NN_FRACTION = 0.762

#: Points sampled to estimate the median nearest-neighbour distance. The tree is
#: built over ALL points — sampling the QUERIES is unbiased, sampling the tree
#: would inflate the spacing by the subsample factor^(1/3).
NN_PROBE_POINTS = 50_000

# ArXiv category colors (comprehensive coverage of all major categories)
CATEGORY_COLORS = {
    # Computer Science
    "cs": np.array([0.3, 0.9, 0.9]),  # Cyan
    # Mathematics
    "math": np.array([0.3, 1.0, 0.5]),  # Green
    # Physics - Astrophysics & Cosmology
    "astro-ph": np.array([0.2, 0.4, 1.0]),  # Deep Blue
    # Physics - Condensed Matter
    "cond-mat": np.array([0.7, 0.3, 0.9]),  # Purple
    # Physics - General Relativity
    "gr-qc": np.array([0.4, 0.6, 0.9]),  # Light Blue
    # Physics - High Energy Physics
    "hep-ph": np.array([1.0, 0.5, 0.2]),  # Orange (Phenomenology)
    "hep-th": np.array([1.0, 0.7, 0.3]),  # Light Orange (Theory)
    "hep-ex": np.array([0.9, 0.4, 0.1]),  # Dark Orange (Experiment)
    "hep-lat": np.array([0.8, 0.6, 0.3]),  # Tan (Lattice)
    # Physics - Nuclear
    "nucl-th": np.array([0.8, 0.4, 0.6]),  # Mauve (Theory)
    "nucl-ex": np.array([0.9, 0.3, 0.5]),  # Pink (Experiment)
    # Physics - Quantum
    "quant-ph": np.array([0.6, 0.3, 1.0]),  # Violet
    # Physics - General
    "physics": np.array([0.5, 0.7, 1.0]),  # Pale Blue
    # Nonlinear Sciences
    "nlin": np.array([0.4, 0.9, 0.7]),  # Teal
    # Biology
    "q-bio": np.array([1.0, 0.3, 0.3]),  # Red
    # Statistics & Finance
    "stat": np.array([0.9, 0.3, 0.9]),  # Magenta
    "q-fin": np.array([1.0, 0.8, 0.2]),  # Gold
    "econ": np.array([0.9, 0.9, 0.4]),  # Yellow
    # Engineering
    "eess": np.array([0.8, 0.9, 0.5]),  # Pale Yellow
    # Preprint servers outside arXiv. The Cornell metadata snapshot covers arXiv
    # only, so without these 384,137 bioRxiv/medRxiv papers — 11.7% of the
    # corpus, and the whole of its life-science half — collapse into grey
    # "other". `papers.csv` names their server in its `journal` column.
    "biorxiv": np.array([1.0, 0.25, 0.55]),  # Crimson Pink
    "medrxiv": np.array([0.55, 0.95, 1.0]),  # Ice Blue
    # Other/Unmatched
    "other": np.array([0.5, 0.5, 0.5]),  # Dark Gray (not white!)
}

#: Colour for a paper whose publication year could not be determined, used in
#: the YEAR view only. 68,138 legacy bioRxiv accessions (`10.1101/001891`) carry
#: no date anywhere, and painting them at either end of the ramp would assert a
#: date the data does not have.
UNKNOWN_YEAR_COLOR = np.array([0.35, 0.35, 0.35], dtype=np.float32)


# =============================================================================
# Dataset Download and Caching
# =============================================================================


def download_kaggle_dataset(
    output_path: Path,
    url: str = "https://www.kaggle.com/api/v1/datasets/download/tomtum/openai-arxiv-embeddings",
) -> Path:
    """Download Kaggle dataset with robust retry and resume capability.

    Uses the robust_download utility which provides:
    - Automatic retry on network errors (up to 3 attempts)
    - Resume capability for partial downloads
    - Progress tracking with ETA
    - File size verification

    Args:
        output_path: Where to save the ZIP file
        url: Kaggle dataset download URL

    Returns:
        Path to downloaded file
    """
    from luxar.demos import robust_download

    with asection("Downloading Kaggle ArXiv Embeddings Dataset"):
        aprint("URL: https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings")
        aprint(f"Destination: {output_path}")
        aprint("")
        aprint("⏱️  This is a ONE-TIME download (~30 GB, 10-15 minutes)")
        aprint("⏱️  Subsequent runs will use cached file instantly!")
        aprint(
            "⏱️  Download will auto-retry on network errors and can resume if interrupted"
        )
        aprint("")

        try:
            output_path = robust_download(
                url=url,
                output_path=output_path,
                max_retries=3,  # Retry up to 3 times on network errors
                timeout=600,  # 10 minute initial connection timeout for large file
                chunk_size=1024 * 1024,  # 1MB chunks
                verify_size=True,  # Verify final size matches Content-Length
            )

            aprint("✓ Download complete and verified!")
            return output_path

        except Exception as e:
            aprint(f"❌ Download failed after all retries: {e}")
            if output_path.exists():
                partial_size = output_path.stat().st_size
                aprint(f"   Partial download saved: {partial_size / (1024**2):.1f} MB")
                aprint("   Run again to resume from this point")
            raise

    return output_path


def load_metadata_lookup(metadata_path: Path, cache_path: Path | None = None) -> dict:
    """Load arXiv metadata and create ID lookup dictionary.

    Args:
        metadata_path: Path to arxiv-metadata-oai-snapshot.json
        cache_path: Optional path to cache the processed lookup dict

    Returns:
        Dictionary mapping paper_id -> {category, title, year}
    """
    import json
    import pickle

    # Check cache first
    if cache_path and cache_path.exists():
        with asection("Loading cached metadata lookup"):
            aprint(f"Cache: {cache_path}")
            with open(cache_path, "rb") as f:
                metadata_by_id = pickle.load(f)
            aprint(f"✓ Loaded {len(metadata_by_id):,} papers instantly!")
        return metadata_by_id

    with asection("Building metadata lookup (one-time, ~15 seconds)"):
        aprint(f"Metadata file: {metadata_path}")
        aprint(f"Size: {metadata_path.stat().st_size / (1024**3):.1f} GB")
        aprint("Loading metadata (full 2M+ papers, may take 1-2 minutes)...")

        metadata_by_id = {}

        with open(metadata_path, "r") as f:
            for i, line in enumerate(f):
                if i % 100000 == 0 and i > 0:
                    aprint(f"  Loaded {i:,} papers...")

                try:
                    paper = json.loads(line)
                    paper_id = paper["id"]
                    categories = paper.get("categories", "")

                    # Get primary category
                    primary_cat = categories.split()[0] if categories else "unknown"

                    meta_info = {
                        "category": primary_cat,
                        "title": paper.get("title", "Unknown")[:100],
                        "year": paper.get("update_date", "2020-01-01")[:4],
                    }

                    # Store with original ID
                    metadata_by_id[paper_id] = meta_info

                    # Also store variants (handle format differences)
                    if paper_id.startswith("0"):
                        metadata_by_id[paper_id.lstrip("0")] = meta_info

                except (json.JSONDecodeError, KeyError):
                    continue

        aprint(f"✓ Loaded metadata for {len(metadata_by_id):,} papers")

        # Cache for future runs
        if cache_path:
            aprint("Saving to cache...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(metadata_by_id, f)
            aprint(f"✓ Cached to {cache_path}")
            aprint("  Future runs will load instantly (<1 second)!")

    return metadata_by_id


# =============================================================================
# Dataset Loading — streaming decode + PCA pre-reduction
# =============================================================================


def read_paper_index(zip_path: Path) -> tuple[list[str], list[str]]:
    """Read every row of ``papers.csv`` (paper id + preprint server).

    ``papers.csv`` is row-aligned with ``vectors.dat``: row ``i`` of the CSV
    names the paper whose 3072 floats start at byte ``i * 3072 * 4``. It is 92 MB
    of text and is read whole — the vectors are what cannot be.

    Args:
        zip_path: The downloaded ``openai-arxiv-embeddings`` ZIP.

    Returns:
        ``(ids, journals)``, both of length ``n_papers``.
    """
    import csv
    import io
    import zipfile

    with asection("Reading papers.csv"):
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("papers.csv") as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"))
                ids: list[str] = []
                journals: list[str] = []
                for row in reader:
                    ids.append(row["id"])
                    journals.append(row.get("journal") or "arxiv")
        counts: dict[str, int] = {}
        for j in journals:
            counts[j] = counts.get(j, 0) + 1
        aprint(f"✓ {len(ids):,} papers")
        for j, c in sorted(counts.items(), key=lambda x: -x[1]):
            aprint(f"  {j}: {c:,}")

    return ids, journals


def _stream_vectors(zip_path: Path, n_rows: int, on_block) -> int:
    """Decode ``vectors.dat`` in blocks and hand each one to ``on_block``.

    The member is deflate-compressed, so it cannot be seeked — every pass costs
    a full decompression (measured 204 MiB/s, ~3 min for 37.6 GiB). What it must
    NOT cost is a Python-level loop per paper: ``np.frombuffer`` over a
    multi-row block is ~3 orders of magnitude faster than a ``struct.unpack``
    per row, and is the difference between the full corpus being a 3-minute read
    and being impossible.

    Args:
        zip_path: The embeddings ZIP.
        n_rows: Total rows in ``vectors.dat``.
        on_block: Called as ``on_block(start_row, block)`` with a read-only
            ``(rows, EMBEDDING_DIM) float32`` view.

    Returns:
        The number of rows actually decoded.
    """
    import time
    import zipfile

    row_bytes = EMBEDDING_DIM * 4
    row = 0
    t0 = time.time()

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("vectors.dat") as f:
            while row < n_rows:
                want = min(STREAM_BLOCK_ROWS, n_rows - row)
                buf = f.read(want * row_bytes)
                got = len(buf) // row_bytes
                if got == 0:
                    break
                block = np.frombuffer(
                    buf, dtype="<f4", count=got * EMBEDDING_DIM
                ).reshape(got, EMBEDDING_DIM)
                on_block(row, block)
                row += got
                if row % (STREAM_BLOCK_ROWS * 100) == 0:
                    elapsed = time.time() - t0
                    eta = elapsed * (n_rows / row - 1) / 60
                    aprint(f"  {row:,}/{n_rows:,}  {elapsed:.0f}s  eta {eta:.1f} min")

    return row


def vector_row_count(zip_path: Path) -> int:
    """Number of papers in the ZIP, from ``vectors.dat``'s uncompressed size."""
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zf:
        size = zf.getinfo("vectors.dat").file_size
    if size % (EMBEDDING_DIM * 4):
        raise ValueError(
            f"vectors.dat is {size} bytes, not a whole number of "
            f"{EMBEDDING_DIM}-float rows — the download is truncated"
        )
    return size // (EMBEDDING_DIM * 4)


def build_pca_matrix(
    zip_path: Path,
    cache_dir: Path,
    pca_dim: int = DEFAULT_PCA_DIM,
    seed: int = 0,
) -> np.ndarray:
    """Reduce every paper's 3072-D vector to ``pca_dim``, streaming, and cache it.

    Two passes over the ZIP:

    1. collect a uniform ``PCA_FIT_ROWS`` subsample and fit a randomized PCA;
    2. project every row into an on-disk ``(n_papers, pca_dim) float32`` memmap.

    The raw matrix is never materialized — at full scale it is 40 GB — and peak
    RSS is bounded by the fit subsample (3.7 GB). The result is cached under
    ``cache_dir`` keyed on ``pca_dim`` only, so later UMAP runs do not re-read the
    3072-D vectors. The caller still reads ``papers.csv`` from the ZIP.

    Args:
        zip_path: The embeddings ZIP.
        cache_dir: Directory for ``pca<dim>_basis.npz`` / ``pca<dim>_all.npy``.
        pca_dim: Number of components to keep.
        seed: Seed for the fit subsample when fitting a new basis. A warm
            ``pca<dim>_*`` cache is reused regardless of the seed.

    Returns:
        A read-only ``(n_papers, pca_dim) float32`` memmap.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    basis_path = cache_dir / f"pca{pca_dim}_basis.npz"
    basis_tmp_path = basis_path.with_suffix(".npz.tmp")
    proj_path = cache_dir / f"pca{pca_dim}_all.npy"

    if proj_path.exists():
        aprint(f"✓ Using cached PCA matrix: {proj_path}")
        return np.load(proj_path, mmap_mode="r")

    n_rows = vector_row_count(zip_path)

    if not basis_path.exists():
        # Gated here rather than at import: a warm PCA cache never needs sklearn.
        PCA = require_module("sklearn.decomposition").PCA

        with asection(f"Fitting PCA basis ({EMBEDDING_DIM}D → {pca_dim}D)"):
            fit_rows = min(PCA_FIT_ROWS, n_rows)
            rng = np.random.default_rng(seed)
            mask = np.zeros(n_rows, dtype=bool)
            mask[rng.choice(n_rows, size=fit_rows, replace=False)] = True

            subsample = np.empty((fit_rows, EMBEDDING_DIM), dtype=np.float32)
            filled = 0

            def collect(start: int, block: np.ndarray) -> None:
                nonlocal filled
                take = mask[start : start + len(block)]
                k = int(take.sum())
                if k:
                    subsample[filled : filled + k] = block[take]
                    filled += k

            aprint(f"Collecting a {fit_rows:,}-row uniform subsample...")
            _stream_vectors(zip_path, n_rows, collect)
            if filled != fit_rows:
                # A short pass means the stream ended early, so what we hold is
                # a uniform sample of a PREFIX — and `papers.csv` is date-
                # ordered, so that is a date-biased basis. Refusing here matters
                # more than it looks: the basis is cached, so accepting one
                # would silently skew every future projection too.
                raise ValueError(
                    f"vectors.dat ended after {filled:,} of {fit_rows:,} "
                    "subsample rows — the download is truncated"
                )

            pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
            pca.fit(subsample[:filled])
            evr = float(pca.explained_variance_ratio_.sum())
            aprint(f"✓ PCA fitted on {filled:,} rows — explains {evr:.1%} of variance")

            # Use a handle: np.savez would append ".npz" to the temp path.
            with basis_tmp_path.open("wb") as basis_file:
                np.savez(
                    basis_file,
                    components=pca.components_.astype(np.float32),
                    mean=pca.mean_.astype(np.float32),
                    explained_variance_ratio=pca.explained_variance_ratio_.astype(
                        np.float32
                    ),
                )
            basis_tmp_path.rename(basis_path)
            del subsample

    basis = np.load(basis_path)
    components = np.ascontiguousarray(basis["components"].T)  # (EMBEDDING_DIM, dim)
    mean = basis["mean"]

    with asection(f"Projecting {n_rows:,} papers → {pca_dim}D"):
        # Write to a .tmp and rename, so an interrupted pass never leaves a
        # short-but-plausible cache behind for the next run to trust.
        tmp_path = proj_path.with_suffix(".npy.tmp")
        out = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(n_rows, pca_dim)
        )

        # `dest` is bound as a default rather than captured: the memmap is
        # deleted below to close it before the rename, and a closure over a
        # deleted name is a NameError waiting to happen.
        def project(start: int, block: np.ndarray, dest=out) -> None:
            dest[start : start + len(block)] = (block - mean) @ components

        got = _stream_vectors(zip_path, n_rows, project)
        out.flush()
        del out
        if got != n_rows:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"vectors.dat ended after {got:,} of {n_rows:,} rows")
        tmp_path.rename(proj_path)
        gb = proj_path.stat().st_size / 1024**3
        aprint(f"✓ Cached {proj_path.name} ({gb:.2f} GB)")

    return np.load(proj_path, mmap_mode="r")


def select_sample(n_rows: int, sample_size: int | None, seed: int = 0) -> np.ndarray:
    """Choose which papers to visualize.

    ``papers.csv`` is sorted by ascending paper ID, so the first N rows are the
    OLDEST N papers, not a sample of the corpus: taking 500,000 of 3,286,365 that
    way yields arXiv submissions from 2007-04 to ~2012-03 and only 8.7% computer
    science, because the CS/ML growth happens entirely after the cut. A subset
    must therefore be drawn uniformly at random.

    Args:
        n_rows: Corpus size.
        sample_size: Papers wanted, or ``None`` / ``>= n_rows`` for all of them.
        seed: Seed, so a given ``--sample`` is reproducible.

    Returns:
        Ascending row indices (ascending keeps the memmap gather sequential).
    """
    if sample_size is None or sample_size >= n_rows:
        if sample_size is not None:
            # Same result, different cache key. The bundle is keyed on the
            # SPELLING because the corpus size is not known until the ZIP is
            # opened, and opening it here would cost a warm-bundle run its
            # whole point (a 30 GB download for someone who reclaimed the
            # disk). So say so instead of silently caching a second copy.
            aprint(
                f"ℹ️  --sample={sample_size:,} is the whole corpus; "
                "`--sample=all` is the canonical spelling and caches under "
                "one key (this run adds a second bundle of the same content)."
            )
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


#: An arXiv identifier encodes its own submission month, in one of two styles:
#: ``YYMM.NNNNN`` since 2007-04 (``0704.0001``), and ``archive/YYMMNNN`` before
#: that (``hep-lat/0506004``, ``math.AG/0601001``). Every one of the corpus's
#: 2,902,228 arXiv rows parses under one of these.
_ARXIV_ID_NEW = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}$")
_ARXIV_ID_OLD = re.compile(r"^[a-zA-Z.\-]+/(\d{2})(\d{2})\d{3}$")

#: bioRxiv/medRxiv DOIs carry a full date: ``10.1101/2020.03.03.20030890``.
_PREPRINT_DOI_DATE = re.compile(r"/(\d{4})\.\d{2}\.\d{2}\.")


def paper_doi(paper_id: str) -> str:
    """The DOI for a row of ``papers.csv``, whichever server it came from.

    Backs the demo's click-through: one ``https://doi.org/{hover_key}`` template
    resolves all three preprint servers, so the mixed corpus needs no per-server
    branching at the node level (a node carries a single ``link``).

    bioRxiv and medRxiv rows already carry a DOI in the id column
    (``10.1101/2020.03.03.20030890``, ``10.64898/...``); arXiv rows carry a bare
    id, and arXiv has retroactively minted ``10.48550/arXiv.<id>`` for every
    paper it holds.

    Dispatches on the id's own shape rather than the ``journal`` column, so a row
    whose journal is blank or unexpected still resolves — ``read_paper_index``
    defaults a missing journal to ``"arxiv"`` and
    :func:`resolve_paper_metadata` buckets anything unrecognised as ``"other"``,
    neither of which says anything about the id.

    The viewer percent-encodes the WHOLE substituted key (its path-traversal
    guard), so what reaches doi.org has even the DOI's own prefix separator
    escaped — ``https://doi.org/10.48550%2FarXiv.hep-th%2F9901001``, two escaped
    slashes for an old-style arXiv id. doi.org unescapes before resolving, which
    is not something to take on trust: checked against the live resolver, that
    URL and ``10.64898%2F2025.12.05.25341689`` both 302 to the right paper.

    Args:
        paper_id: An identifier from ``papers.csv``.

    Returns:
        A DOI with no scheme or ``doi.org/`` prefix — the bare value the
        ``link`` template substitutes and right-click copies.
    """
    if paper_id.startswith("10."):
        return paper_id
    return f"10.48550/arXiv.{paper_id}"


def arxiv_submission_year(paper_id: str) -> int:
    """Submission year encoded in an arXiv ID, or ``0`` if it is not an arXiv ID.

    Preferred over the Cornell snapshot's ``update_date``, which is the LAST
    REVISION date: no record in the snapshot predates 2007, so reading years
    from it paints the 400,803 papers submitted from 1991 through 2006 as
    2007-or-later and compresses 35 years of arXiv into 19.

    This deliberately re-crosses a fence. An ID decode existed in 3638ba8bf and
    was dropped by 5e34d3e27 ("integrate arXiv metadata for real categories,
    titles, and years"), whose actual subject was categories and titles — which
    an identifier cannot supply and which still come from the snapshot. The year
    was collateral. The decode it removed also never handled old-style IDs (it
    guessed 2010) and fell back to a literal 2015; this one handles both styles
    and returns 0 rather than inventing a date, so restoring it is not a revert
    to what was removed.

    Args:
        paper_id: An identifier from ``papers.csv``.

    Returns:
        A four-digit year, or ``0`` when the ID is not an arXiv identifier.
    """
    m = _ARXIV_ID_NEW.match(paper_id) or _ARXIV_ID_OLD.match(paper_id)
    if not m:
        return 0
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return 0
    # arXiv's identifiers begin in 1991 (the corpus's earliest is 9107), so a
    # two-digit year of 91..99 is 19xx and everything else is 20xx. The rule
    # stays unambiguous until 2091.
    return 1900 + yy if yy >= 91 else 2000 + yy


def resolve_paper_metadata(
    ids: list[str],
    journals: list[str],
    metadata_lookup: dict,
) -> tuple[list[str], list[str], list[int]]:
    """Attach a title, a category and a year to every selected paper.

    Titles and categories come from the Cornell snapshot, which covers arXiv
    only. Years do NOT: the snapshot records ``update_date``, so the year is
    taken from the arXiv ID itself (see :func:`arxiv_submission_year`) and the
    snapshot is only a fallback. A bioRxiv/medRxiv row instead takes its
    category from its preprint server and its year from the date embedded in
    its DOI (``10.1101/2020.03.03.20030890``). Legacy bioRxiv accessions
    (``10.1101/001891``) carry no date at all and get year ``0``, which the
    caller renders as :data:`UNKNOWN_YEAR_COLOR` rather than guessing.

    Args:
        ids: Paper IDs, in visualization order.
        journals: Matching ``papers.csv`` ``journal`` values.
        metadata_lookup: ``paper_id -> {category, title, year}`` from Cornell.

    Returns:
        ``(titles, categories, years)``; ``years`` is 0 where unknown.
    """
    titles: list[str] = []
    categories: list[str] = []
    years: list[int] = []
    matched = 0
    undated = 0

    for pid, journal in zip(ids, journals):
        meta = metadata_lookup.get(pid)
        if meta is not None:
            titles.append(meta["title"])
            cat = meta["category"]
            categories.append(cat.split(".")[0] if "." in cat else cat)
            # The ID is the submission date; `meta["year"]` is only the last
            # revision, so it is the fallback, not the source.
            years.append(arxiv_submission_year(pid) or int(meta["year"]))
            matched += 1
            continue

        # Not in the arXiv snapshot: fall back to the preprint server itself.
        server = journal if journal in ("biorxiv", "medrxiv") else "other"
        categories.append(server)
        titles.append(f"{server}:{pid}" if server != "other" else f"arXiv:{pid}")
        m = _PREPRINT_DOI_DATE.search(pid)
        if m:
            years.append(int(m.group(1)))
        else:
            years.append(arxiv_submission_year(pid))
            if years[-1] == 0:
                undated += 1

    n = len(ids)
    matched_percent = matched / n * 100 if n else 0.0
    aprint(
        f"✓ Matched {matched:,}/{n:,} papers to arXiv metadata ({matched_percent:.1f}%)"
    )
    aprint(
        f"✓ Of {n - matched:,} papers without snapshot metadata, "
        f"{n - matched - undated:,} dated from their identifier and "
        f"{undated:,} left undated"
    )
    aprint(f"✓ {len(set(categories))} distinct categories")

    return titles, categories, years


def median_nearest_neighbor_distance(
    positions: np.ndarray, n_probe: int = NN_PROBE_POINTS, seed: int = 0
) -> float:
    """Median distance from a point to its nearest neighbour in the cloud.

    The tree is built over EVERY point and only the queries are subsampled:
    subsampling the tree instead would report the spacing of a sparser cloud,
    inflated by roughly ``(n / n_probe) ** (1/3)``.

    Args:
        positions: ``(N, 3)`` reduced coordinates.
        n_probe: Query points drawn at random.
        seed: Seed for that draw.

    Returns:
        The median nearest-neighbour distance, or ``0.0`` for a degenerate cloud.
    """
    if len(positions) < 2:
        return 0.0

    # Gated here, not at import: only the radius calibration needs a KD-tree, and
    # a warm bundle carries the measurement rather than recomputing it.
    KDTree = require_module("sklearn.neighbors").KDTree

    tree = KDTree(np.ascontiguousarray(positions, dtype=np.float64))
    rng = np.random.default_rng(seed)
    probe = positions[
        rng.choice(len(positions), size=min(n_probe, len(positions)), replace=False)
    ]
    # k=2: the first neighbour of a point in the tree is itself.
    dist, _ = tree.query(np.ascontiguousarray(probe, dtype=np.float64), k=2)
    return float(np.median(dist[:, 1]))


# =============================================================================
# UMAP Dimensionality Reduction
# =============================================================================


def _cuml_umap():
    """Return cuML's ``UMAP`` class, or ``None`` when RAPIDS is not installed.

    Not routed through :func:`require_module`: cuML is genuinely optional (it is
    in no Luxar extra and has no CPU-only wheel), so its absence must be a quiet
    fall-back to ``umap-learn``, not an install instruction.
    """
    try:
        from cuml.manifold import UMAP as CuUMAP  # type: ignore[import-not-found]
    except Exception:
        return None
    return CuUMAP


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    device: str = "auto",
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 3D using UMAP.

    Args:
        embeddings: ``(n_samples, n_features)`` array — the PCA-reduced matrix,
            not the raw 3072-D vectors.
        n_components: Target dimensions (3 for visualization).
        n_neighbors: UMAP neighbourhood size.
        device: ``"gpu"`` to require cuML, ``"cpu"`` to require umap-learn,
            ``"auto"`` (default) to prefer cuML when it imports. At full corpus
            scale the difference is minutes against hours.

    Returns:
        ``(n_samples, n_components)`` coordinates, centred at their barycentre.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"device must be auto/cpu/gpu, got {device!r}")

    CuUMAP = None if device == "cpu" else _cuml_umap()
    if device == "gpu" and CuUMAP is None:
        raise RuntimeError(
            "--device=gpu needs RAPIDS cuML (pip install cuml-cu12); "
            "use --device=cpu or --device=auto for umap-learn"
        )

    backend = "cuML (GPU)" if CuUMAP is not None else "umap-learn (CPU)"
    with asection(f"Reducing {embeddings.shape[1]}D → {n_components}D with UMAP"):
        aprint(f"Input: {embeddings.shape}  backend: {backend}")
        aprint(f"Parameters: n_neighbors={n_neighbors}, metric=cosine")

        # A memmap is fine for umap-learn but cuML wants a contiguous host array.
        matrix = np.ascontiguousarray(embeddings, dtype=np.float32)

        if CuUMAP is not None:
            reducer = CuUMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric="cosine",
                verbose=True,
            )
        else:
            # Gated here, not in main(): a warm arxiv_kaggle cache never needs UMAP.
            UMAP = require_module("umap").UMAP
            reducer = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric="cosine",
                n_jobs=-1,  # Use ALL CPU cores for massive speedup!
                low_memory=False,  # Speed optimization (uses more RAM)
                # Note: random_state removed to enable parallel processing
                # Results will vary slightly between runs but be much faster!
                verbose=True,
            )

        reduced = np.asarray(reducer.fit_transform(matrix), dtype=np.float32)

        aprint("✓ UMAP complete")
        aprint(f"  Output: {reduced.shape}")
        aprint(f"  Range: [{reduced.min():.2f}, {reduced.max():.2f}]")

        # Center at barycenter (center of mass) for easier exploration
        centroid = reduced.mean(axis=0)
        reduced = reduced - centroid
        aprint("✓ Centered at barycenter")
        aprint(f"  New range: [{reduced.min():.2f}, {reduced.max():.2f}]")

    return reduced.astype(np.float32)


# =============================================================================
# Cache Provisioning
# =============================================================================


def ensure_embeddings_zip() -> Path:
    """Return the ~30 GB embeddings ZIP, downloading it if it is not cached.

    A short size check catches a half-finished download that would otherwise
    read as a valid (but truncated) corpus, and a marker file keeps two
    concurrent demo runs from downloading over each other.

    Returns:
        Path to a complete ``arxiv_embeddings.zip``.

    Raises:
        RuntimeError: Another process is mid-download.
    """
    dataset_cache = Path.home() / ".cache" / "luxar" / "arxiv_embeddings.zip"
    expected_emb_size_gb = 30
    download_marker = dataset_cache.parent / ".arxiv_downloading"

    if download_marker.exists():
        aprint("⚠️  Download already in progress in another process!")
        aprint("   Please wait for it to complete or delete the marker:")
        aprint(f"   rm {download_marker}")
        raise RuntimeError("Download in progress")

    if dataset_cache.exists():
        size_gb = dataset_cache.stat().st_size / (1024**3)
        if size_gb < expected_emb_size_gb * 0.9:  # Allow 10% variance
            aprint(
                f"⚠️  Cached embeddings incomplete ({size_gb:.1f} GB / "
                f"~{expected_emb_size_gb} GB)"
            )
            aprint("   Deleting and re-downloading...")
            dataset_cache.unlink()
        else:
            aprint(f"✓ Using cached embeddings: {dataset_cache}")
            aprint(f"  Size: {size_gb:.1f} GB")

    if not dataset_cache.exists():
        aprint("Dataset not in cache, downloading...")
        download_marker.parent.mkdir(parents=True, exist_ok=True)
        download_marker.touch()
        try:
            download_kaggle_dataset(dataset_cache)
        finally:
            # Remove marker when done (or on failure)
            if download_marker.exists():
                download_marker.unlink()

    return dataset_cache


def ensure_metadata_lookup() -> dict:
    """Return the Cornell arXiv ``paper_id -> {category, title, year}`` lookup.

    Downloads and extracts the ~1.8 GB snapshot on first use. This covers arXiv
    only — bioRxiv/medRxiv rows are described by
    :func:`resolve_paper_metadata` from ``papers.csv`` instead.

    Returns:
        The lookup dict (~3 GB of small dicts at full scale).
    """
    metadata_cache = Path.home() / ".cache" / "luxar" / "arxiv_metadata.json"

    if not metadata_cache.exists():
        aprint("Metadata not in cache, downloading...")
        # Cache the metadata ZIP under ~/.cache/luxar/arxiv_kaggle instead of
        # polluting the user's ~/Downloads (skip-if-present built in).
        meta_zip = cached_download(
            "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv",
            "arxiv_kaggle",
            "arxiv-metadata.zip",
        )

        import zipfile

        aprint("Extracting metadata...")
        with zipfile.ZipFile(meta_zip, "r") as zf:
            zf.extract("arxiv-metadata-oai-snapshot.json", metadata_cache.parent)
            extracted = metadata_cache.parent / "arxiv-metadata-oai-snapshot.json"
            if extracted.exists():
                extracted.rename(metadata_cache)
        aprint(f"✓ Metadata extracted to {metadata_cache}")

    return load_metadata_lookup(
        metadata_cache,
        cache_path=Path.home() / ".cache" / "luxar" / "arxiv_metadata_lookup.pkl",
    )


# =============================================================================
# Visualization Generation
# =============================================================================


def generate_paper_landscape(
    output_path: Path,
    sample_size: int | None = DEFAULT_SAMPLE_SIZE,
    pca_dim: int = DEFAULT_PCA_DIM,
    device: str = "auto",
    seed: int = 0,
) -> int:
    """Generate the 3D landscape of the preprint corpus.

    Args:
        output_path: Where to write the scene.
        sample_size: Papers to visualize; ``None`` means the whole corpus.
        pca_dim: PCA components fed to UMAP.
        device: UMAP backend — ``auto`` / ``cpu`` / ``gpu`` (cuML).
        seed: Seed for both the PCA fit subsample and the paper sample.

    Returns:
        Number of papers visualized.
    """

    def _compute_bundle() -> dict:
        dataset_cache = ensure_embeddings_zip()
        metadata_lookup = ensure_metadata_lookup()

        # Stream the 40 GB of vectors through a PCA basis ONCE, cached on disk
        # under `pca<dim>_all.npy`, then sample rows out of that matrix. Ordering
        # matters: reducing first and sampling second means a different
        # `--sample` costs a memmap gather, not another pass over the ZIP.
        ids, journals = read_paper_index(dataset_cache)
        pca_matrix = build_pca_matrix(
            dataset_cache,
            Path.home() / ".cache" / "luxar" / "arxiv_kaggle",
            pca_dim=pca_dim,
            seed=seed,
        )
        if len(pca_matrix) != len(ids):
            raise ValueError(
                f"papers.csv has {len(ids):,} rows but vectors.dat has "
                f"{len(pca_matrix):,} — the two files are not row-aligned"
            )

        selected = select_sample(len(pca_matrix), sample_size, seed=seed)
        if len(selected) == 0:
            return {
                "positions": np.zeros((0, 3), dtype=np.float32),
                "categories": [],
                "years": [],
                "titles": None,
                "ids": [],
                "median_nn": 0.0,
            }
        aprint(f"✓ Visualizing {len(selected):,} of {len(pca_matrix):,} papers")

        embeddings = np.ascontiguousarray(pca_matrix[selected])
        with asection("Resolving titles, categories and years"):
            titles, categories, years = resolve_paper_metadata(
                [ids[i] for i in selected],
                [journals[i] for i in selected],
                metadata_lookup,
            )
        del metadata_lookup  # ~3 GB of dicts; UMAP wants the room

        # Reduce to 3D
        positions = reduce_embeddings_umap(embeddings, n_components=3, device=device)

        # Measured here rather than at scene-build time so a warm bundle needs
        # no KD-tree (and no scikit-learn) to size its points.
        with asection("Measuring cloud spacing"):
            median_nn = median_nearest_neighbor_distance(positions, seed=seed)
            aprint(f"✓ Median nearest-neighbour distance: {median_nn:.5f}")

        return {
            "positions": positions,
            "categories": list(categories),
            "years": list(years),
            "titles": list(titles) if titles is not None else None,
            # The paper's own identifier, for the DOI link (#1917). Already read
            # from `papers.csv` and sliced for `resolve_paper_metadata` just
            # above, then dropped — and the hover label is a truncated title, so
            # nothing downstream could reconstruct it.
            "ids": [ids[i] for i in selected],
            "median_nn": median_nn,
        }

    # UMAP + matched metadata cached under ~/.cache/luxar/arxiv_kaggle. The
    # backend is deliberately absent from the key: either backend produces a
    # valid embedding, and recomputing 3.29M points just to switch is wasteful.
    # version=3: v1 bundles were a date-ordered PREFIX of the corpus with no
    # `median_nn`; v2 bundles carry `update_date` years, which
    # `resolve_paper_metadata` no longer produces. Neither may be reused — the
    # stored `years` are part of this computation's output, so changing how they
    # are derived invalidates the cache exactly as changing the UMAP would.
    # Still version=3, deliberately, even though `_compute_bundle` gained an
    # `ids` field after v3 bundles were written. Bumping would invalidate every
    # warm bundle and force a 40 GB PCA stream plus a UMAP over 3.29M points on
    # each machine — the cost this cache exists to avoid. A pre-ids bundle
    # instead degrades to a decorated-label search (see `link_attrs` below), so
    # the pick stays useful without anyone paying for a recompute. Regenerate
    # explicitly if you want the real DOI deep links locally.
    cache_key = (
        f"umap3d_n{'all' if sample_size is None else sample_size}"
        f"_pca{pca_dim}_seed{seed}"
    )
    bundle = cache_computed("arxiv_kaggle", cache_key, _compute_bundle, version=3)
    positions = bundle["positions"]
    categories = list(bundle["categories"])
    years = list(bundle["years"])
    titles = bundle["titles"]
    # `.get`, and deliberately NO version bump: `cache_computed` pickles the
    # whole dict under `<key>_v<version>`, so an added field reads back fine and
    # a bundle written before this simply lacks it. Bumping to v4 would force
    # every warm cache to re-stream 40 GB of vectors through PCA and re-run UMAP
    # over 3.29M points, and the bar for that — set by the version note above —
    # is a change to what the computation OUTPUTS. Adding a field changes none
    # of it, so a pre-existing bundle ships without links instead of triggering
    # hours of recomputation.
    paper_ids = list(bundle.get("ids") or [])
    median_nn = float(bundle.get("median_nn") or 0.0)

    if len(positions) == 0:
        aprint("❌ No papers loaded")
        return 0

    # Generate visualization
    with asection("Generating visualization"):
        n_papers = len(positions)
        year_array = np.array(years, dtype=np.float32)
        # Year 0 = "no date anywhere in the record" (legacy bioRxiv accessions),
        # not "year zero". It is excluded from the ramp's endpoints so a handful
        # of undated papers cannot stretch the whole gradient, and painted
        # separately below.
        dated = year_array > 0
        if dated.any():
            yr_lo = float(year_array[dated].min())
            yr_hi = float(year_array[dated].max())
        else:
            yr_lo = yr_hi = 0.0
        yr_rng = yr_hi - yr_lo
        yr_t = (
            np.clip((year_array - yr_lo) / yr_rng, 0.0, 1.0)
            if yr_rng > 0
            else np.zeros_like(year_array)
        )

        # Two switchable coloring views: arXiv category / preprint server
        # (categorical) and a cool→warm sequential ramp over publication year.
        category_colors = np.array(
            [CATEGORY_COLORS.get(c, CATEGORY_COLORS["other"]) for c in categories],
            dtype=np.float32,
        )
        year_colors = hsv_to_rgb(0.66 * (1.0 - yr_t))  # older=blue → newer=red
        year_colors[~dated] = UNKNOWN_YEAR_COLOR

        cat_counts: dict[str, int] = {}
        for cat in categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        aprint("✓ Papers by category (colored by category / year):")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
            aprint(f"  {cat}: {count:,}")
        aprint(
            f"  Year range: {int(yr_lo)} to {int(yr_hi)}"
            f" ({int((~dated).sum()):,} undated)"
        )

        # Per-point radii by recency (newer=larger), scaled to the cloud's OWN
        # measured spacing so the same code is correctly sized at 100k and at
        # 3.29M papers. A degenerate cloud with no measurable spacing falls back
        # to the ramp that was hand-tuned at 500k.
        if median_nn > 0:
            radius_base = RADIUS_MIN_NN_FRACTION * median_nn
            radius_gain = (RADIUS_MAX_NN_FRACTION - RADIUS_MIN_NN_FRACTION) * median_nn
        else:
            radius_base, radius_gain = 0.005, 0.011
        aprint(
            f"✓ Radii {radius_base:.5f} → {radius_base + radius_gain:.5f} "
            f"(median NN spacing {median_nn:.5f})"
        )
        radii_pp = (radius_base + radius_gain * yr_t).astype(np.float32)
        # Either end of the ramp would assign a date the neutral colour refuses to.
        radii_pp[~dated] = radius_base + 0.5 * radius_gain

        def _title(i: int) -> str:
            if titles is not None:
                return titles[i][:60] + ("…" if len(titles[i]) > 60 else "")
            return str(categories[i])

        def _year(i: int) -> str:
            return str(years[i]) if years[i] else "year unknown"

        category_labels = [
            f"{_title(i)} ({_year(i)}, {categories[i]})" for i in range(n_papers)
        ]
        year_labels = [f"{_title(i)} ({_year(i)})" for i in range(n_papers)]

        have_ids = len(paper_ids) == n_papers
        if not have_ids:
            aprint(
                "  ⓘ Cached bundle predates stored paper ids — "
                "using Scholar label search"
            )

        stacked = stack_colorings(
            positions,
            [
                {
                    "label": "Category",
                    "colors": category_colors,
                    "labels": category_labels,
                },
                {"label": "Year", "colors": year_colors, "labels": year_labels},
            ],
            keys=[paper_doi(p) for p in paper_ids] if have_ids else None,
        )
        link_attrs = (
            {
                "keys": stacked.keys,
                "link": "https://doi.org/{hover_key}",
                "copy": "{hover_key}",
            }
            if stacked.keys is not None
            # No stored ids (a cached bundle predating them) used to mean NO
            # click-through at all, which is the worst outcome: every point
            # already carries a title as its hover label, so the pick was
            # actionable and silently did nothing. Fall back to a label search
            # instead — same shape as the label-search links cellxgene
            # (EBI OLS) and dmri_tractography (Wikipedia) already use. Scholar
            # rather than arXiv search because this set is arXiv + bioRxiv +
            # medRxiv, and an arXiv-only search misses the two preprint servers.
            else {
                "link": "https://scholar.google.com/scholar?q={hover_label}",
                "copy": "{hover_label}",
            }
        )
        radii = np.tile(radii_pp, len(stacked.categories)).astype(np.float32)

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension(
                    "coloring",
                    unit="",
                    categories=stacked.categories,
                    display=False,
                    description="Color scheme: category or preprint server / publication year",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True),
            )

            # Additive ladder only — no substitutive levels. 3.29M papers x 2
            # colorings is 6.57M stored, but the colorings stack on the hidden
            # `coloring` axis, so the resident slice is 3,286,365 against a
            # 5,591,040 Points cap. The coarse levels served a framing the
            # screen-area selector never picks (finest anchored at half-screen
            # occupancy; this demo opens auto-fitted): 18 groups -> 9.
            #
            # NOTE for anyone tempted to add `partition=` here instead: the
            # hidden `coloring` dim is FIRST in this scene, so displayDims is
            # [1,2,3]. `spatial_bsp_tree` always splits on positions columns 0-2
            # whatever they mean, and the viewer discards a `bsp_tree` whose
            # split axis is not displayed — so a partition would quietly revert
            # to centroid ordering. It would need the stack axis moved LAST, as
            # demo_nuclear_pore_complex documents.
            #
            # Retires the substitutive_lod_or_flat gate: it existed because the
            # coarsening write path imports torch+scipy, and an additive ladder
            # imports neither, so a warm-cache run builds the REAL scene rather
            # than a degraded flat one.
            scene.add_points(
                "arxiv_papers_kaggle",
                positions=stacked.positions,
                colors=stacked.colors,
                radii=radii,
                sharpness=np.full(len(stacked.positions), 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.1,
                labels=stacked.labels,
                **link_attrs,
                layer=True,
                additive_lod=stream_ladder(
                    len(stacked.positions),
                    slices=hidden_axis_stops(stacked.positions, dims.non_displayed),
                ),
            )

            # --- Overlays ---
            scene.add_text(
                "arXiv · bioRxiv · medRxiv Papers",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Category legend (view 0) + year gradient caption (view 1).
            _cat_legend = (
                '<div style="font-size:1.2vh;line-height:1.5;background:rgba(0,0,0,0.5);'
                'padding:0.5vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">Category</div>'
            )
            legend_categories = [
                cat for cat, _ in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]
            ]
            for preprint_server in ("biorxiv", "medrxiv"):
                if (
                    preprint_server in cat_counts
                    and preprint_server not in legend_categories
                ):
                    legend_categories.append(preprint_server)
            for cat in legend_categories:
                r, g, b = (
                    int(round(v * 255))
                    for v in CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
                )
                _cat_legend += f'<div><span style="color:#{r:02x}{g:02x}{b:02x}">█</span> {cat}</div>'
            _cat_legend += "</div>"
            # x=0.05, not 0.02: the full-corpus legend has ten-plus rows and
            # its lower entries otherwise run under the viewer's icon rail.
            scene.add_html(
                _cat_legend,
                position=(0.05, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 0},
                transition="fade",
                transition_duration=0.3,
            )
            scene.add_html(
                '<div style="font-size:1.3vh;background:rgba(0,0,0,0.5);padding:0.6vh;'
                'border-radius:3px;color:#ccc">Year: '
                '<span style="color:#4d80ff">█</span> older → '
                '<span style="color:#ff4d4d">█</span> newer &nbsp; '
                '<span style="color:#595959">█</span> undated</div>',
                position=(0.05, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 1},
                transition="fade",
                transition_duration=0.3,
            )

            # Info + source
            add_demo_caption(
                scene,
                f"{n_papers:,} papers • OpenAI text-embedding-3-large • "
                f"PCA-{pca_dim} → UMAP 3D",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Visualization created with {n_papers:,} papers")

    return n_papers


# =============================================================================
# Main Entry Point
# =============================================================================


def _positive(flag: str, raw: str) -> int:
    """An integer of at least 1, or a message naming the flag."""
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{flag} needs an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{flag} must be at least 1, got {value}")
    return value


def _parse_sample(raw: str) -> int | None:
    """``all`` / ``full`` mean the whole corpus; anything else is a count."""
    return (
        None if raw.strip().lower() in ("all", "full") else _positive("--sample", raw)
    )


def _parse_pca_dim(raw: str) -> int:
    """At least 1, and at most the width of a stored vector.

    The upper bound is not cosmetic: PCA cannot produce more components than the
    data has features, so a larger value fails inside the fit — after a full
    streaming pass has already been paid for.
    """
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"--pca-dim needs an integer, got {raw!r}") from None
    if not 1 <= value <= EMBEDDING_DIM:
        # One message for both bounds: either way the useful thing to tell the
        # user is the range, not which end they missed.
        raise ValueError(
            f"--pca-dim must be between 1 and {EMBEDDING_DIM}, got {value}"
        )
    return value


def _parse_device(raw: str) -> str:
    device = raw.lower()
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"--device must be auto, cpu or gpu, got {device!r}")
    return device


def _parse_seed(raw: str) -> int:
    try:
        seed = int(raw)
    except ValueError:
        raise ValueError(f"--seed needs an integer, got {raw!r}") from None
    # `np.random.default_rng` rejects a negative seed, and is not reached until
    # the sampling step.
    if seed < 0:
        raise ValueError(f"--seed must be non-negative, got {seed}")
    return seed


#: ``--flag`` -> (result key, value parser). Table-driven so adding a flag
#: cannot forget to validate it, and so `parse_args` stays branch-free.
_FLAGS: dict[str, tuple[str, Any]] = {
    "--sample=": ("sample_size", _parse_sample),
    "--pca-dim=": ("pca_dim", _parse_pca_dim),
    "--device=": ("device", _parse_device),
    "--seed=": ("seed", _parse_seed),
}


def parse_args(argv: list[str]) -> tuple[int | None, int, str, int]:
    """Parse the demo's ``--flag=value`` arguments.

    Every value is validated HERE rather than where it is first used. The two
    consumers sit behind a multi-minute streaming pass over a 30 GB ZIP (hours
    on a cold cache), so a typo in ``--device`` used to surface only after all
    of it, and a non-positive ``--sample`` / ``--pca-dim`` either crashed there
    with a bare numpy or sklearn message or — for ``--sample=0`` — produced an
    empty scene with no error at all.

    Unrecognised arguments are ignored: ``luxar demo run`` forwards its own
    (``--no-serve``).

    Args:
        argv: Arguments without the program name.

    Returns:
        ``(sample_size, pca_dim, device, seed)``; ``sample_size`` is ``None``
        for the whole corpus (the default, and what ``--sample=all`` spells).

    Raises:
        ValueError: A flag carries an unusable value.
    """
    opts: dict[str, Any] = {
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "pca_dim": DEFAULT_PCA_DIM,
        "device": "auto",
        "seed": 0,
    }
    for arg in argv:
        for prefix, (name, parse) in _FLAGS.items():
            if arg.startswith(prefix):
                opts[name] = parse(arg[len(prefix) :])
                break

    return opts["sample_size"], opts["pca_dim"], opts["device"], opts["seed"]


def main() -> None:
    """Main demo entry point."""
    try:
        sample_size, pca_dim, device, seed = parse_args(sys.argv[1:])
    except ValueError as e:
        # Match how a build failure is reported below: a typo is a user error,
        # not a crash, and does not deserve a traceback.
        aprint(f"Error: {e}")
        sys.exit(2)

    aprint("=" * 70)
    aprint("PREPRINT EMBEDDINGS - PRE-COMPUTED FROM KAGGLE")
    aprint("=" * 70)
    aprint("")
    aprint("Dataset: OpenAI ArXiv Embeddings (Kaggle)")
    aprint("https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings")
    aprint("")
    aprint("What this shows:")
    aprint("  • 3,286,365 arXiv + bioRxiv + medRxiv papers, through 2025-12")
    aprint("  • OpenAI text-embedding-3-large vectors (3,072D), streamed and")
    aprint(f"    PCA-reduced to {pca_dim}D, then projected to 3D by UMAP")
    aprint("  • Papers cluster by research topic automatically")
    aprint("  • Color = category / preprint server, or publication year")
    aprint("  • Size = recency (newer papers are larger)")
    aprint("")
    aprint("Parameters:")
    aprint(
        "  Sample size: "
        + ("whole corpus" if sample_size is None else f"{sample_size:,} papers")
    )
    aprint(f"  PCA dim: {pca_dim}   UMAP device: {device}   seed: {seed}")
    if sample_size is None:
        aprint(
            "  First run: budget ~39 GB of disk and hours on CPU; "
            "use --sample=N to bound UMAP RAM/time."
        )
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "arxiv_papers_kaggle.luxar.zarr"
        try:
            n_papers = generate_paper_landscape(
                output_path,
                sample_size=sample_size,
                pca_dim=pca_dim,
                device=device,
                seed=seed,
            )
            if n_papers == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total papers: {n_papers:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_arxiv_kaggle_") as tmpdir:
        output_path = Path(tmpdir) / "arxiv_papers_kaggle.luxar.zarr"

        try:
            n_papers = generate_paper_landscape(
                output_path,
                sample_size=sample_size,
                pca_dim=pca_dim,
                device=device,
                seed=seed,
            )

            if n_papers == 0:
                return

        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Explore the knowledge landscape:")
        aprint("  - Zoom out: See overall structure of scientific fields")
        aprint("  - Zoom in: Explore specific research topics")
        aprint("  - Look for clusters: Papers on same topic group together")
        aprint("  - Boundaries: Interdisciplinary research")
        aprint("")
        aprint(f"Total papers: {n_papers:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
