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

…through 2025-12. Titles, categories and dates come from the separate Cornell
arXiv metadata snapshot (~1.8 GB), also on Kaggle; papers it does not cover fall
back to their preprint server (`biorxiv` / `medrxiv`), which is a real category
rather than a grey "other".

HOW THE FULL CORPUS FITS IN MEMORY
==================================

It does not — `N x 3072 x 4 B` is 40 GB — so it is never held. `vectors.dat` is
streamed out of the ZIP in blocks (measured 204 MiB/s, ~3 min for the whole
thing) and projected on the fly through a PCA basis down to `--pca-dim`
(default 128, capturing ~58% of the variance) fitted on a 300k-row uniform
subsample. Only that `(N, 128) float32` matrix — 1.6 GB — is cached and handed
to UMAP. Re-running UMAP at other settings costs nothing: the PCA cache is keyed
on `--pca-dim` alone and the 30 GB ZIP is not touched again.

DOWNLOAD & CACHING
==================

FIRST RUN (one-time)
  * embeddings ZIP  ~30 GB  ->  ~/.cache/luxar/arxiv_embeddings.zip
  * metadata ZIP    ~1.8 GB ->  ~/.cache/luxar/arxiv_kaggle/
  * PCA matrix      1.6 GB  ->  ~/.cache/luxar/arxiv_kaggle/pca128_all.npy
  No Kaggle credentials are needed — both are public dataset URLs.

SUBSEQUENT RUNS
  * warm `arxiv_kaggle` bundle -> seconds
  * warm PCA cache, new UMAP   -> minutes (GPU) to hours (CPU, full corpus)

Usage:
    python demo_arxiv_embeddings_kaggle.py [OPTIONS]

    --sample=N        Uniform RANDOM sample of N papers (default: the whole
                      corpus). Random, not a prefix: `papers.csv` is sorted by
                      ID, so a prefix is a date slice, not a sample.
    --seed=S          Seed for that sample (default 0).
    --pca-dim=D       PCA components fed to UMAP (default 128).
    --device=auto|cpu|gpu
                      `gpu` runs cuML's UMAP (RAPIDS) when importable; `auto`
                      uses it if present, else umap-learn on the CPU.

Requirements:
    - Install: pip install 'luxar[demos]'   # umap-learn, scikit-learn, pandas
    - Optional: pip install 'luxar[gsplats]'   # torch + scipy, for Points LOD
      coarsening; without it the scene is a flat (fully viewable) point cloud
    - Optional: a RAPIDS cuML install, for GPU UMAP on the full corpus

Controls:
    - Explore clusters of related research
    - Color = arXiv category / preprint server, or publication year
    - Size = recency (newer papers larger)
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
        "url": "https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings",
    },
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import (
    cache_computed,
    cached_download,
    hsv_to_rgb,
    launch_viewer,
    require_module,
    stack_colorings,
    substitutive_lod_or_flat,
)
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
#: 3.7 GB held once, which is what bounds this pipeline's peak RSS.
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
    from luxar.utils.download import robust_download

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
    ``cache_dir`` keyed on ``pca_dim`` only, so re-running UMAP at other settings
    never re-reads the 30 GB ZIP.

    Args:
        zip_path: The embeddings ZIP.
        cache_dir: Directory for ``pca<dim>_basis.npz`` / ``pca<dim>_all.npy``.
        pca_dim: Number of components to keep.
        seed: Seed for the fit subsample.

    Returns:
        A read-only ``(n_papers, pca_dim) float32`` memmap.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    basis_path = cache_dir / f"pca{pca_dim}_basis.npz"
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

            pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed)
            pca.fit(subsample[:filled])
            evr = float(pca.explained_variance_ratio_.sum())
            aprint(f"✓ PCA fitted on {filled:,} rows — explains {evr:.1%} of variance")

            np.savez(
                basis_path,
                components=pca.components_.astype(np.float32),
                mean=pca.mean_.astype(np.float32),
                explained_variance_ratio=pca.explained_variance_ratio_.astype(
                    np.float32
                ),
            )
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
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def resolve_paper_metadata(
    ids: list[str],
    journals: list[str],
    metadata_lookup: dict,
) -> tuple[list[str], list[str], list[int]]:
    """Attach a title, a category and a year to every selected paper.

    The Cornell snapshot covers arXiv only. A bioRxiv/medRxiv row instead takes
    its category from its preprint server and its year from the date embedded in
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
    import re

    doi_date = re.compile(r"/(\d{4})\.\d{2}\.\d{2}\.")

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
            years.append(int(meta["year"]))
            matched += 1
            continue

        # Not in the arXiv snapshot: fall back to the preprint server itself.
        server = journal if journal in ("biorxiv", "medrxiv") else "other"
        categories.append(server)
        titles.append(f"{server}:{pid}" if server != "other" else f"arXiv:{pid}")
        m = doi_date.search(pid)
        if m:
            years.append(int(m.group(1)))
        else:
            years.append(0)
            undated += 1

    n = len(ids)
    aprint(
        f"✓ Matched {matched:,}/{n:,} papers to arXiv metadata "
        f"({matched / n * 100:.1f}%)"
    )
    aprint(
        f"✓ {n - matched - undated:,} dated from their preprint DOI, "
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
            "median_nn": median_nn,
        }

    # UMAP + matched metadata cached under ~/.cache/luxar/arxiv_kaggle, keyed on
    # everything that changes the result, so a second identical run is instant.
    # version=2: v1 bundles were a date-ordered PREFIX of the corpus with no
    # `median_nn`, and must not be reused.
    cache_key = (
        f"umap3d_n{'all' if sample_size is None else sample_size}"
        f"_pca{pca_dim}_seed{seed}"
    )
    bundle = cache_computed("arxiv_kaggle", cache_key, _compute_bundle, version=2)
    positions = bundle["positions"]
    categories = list(bundle["categories"])
    years = list(bundle["years"])
    titles = bundle["titles"]
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
        # 3.29M papers. A stale bundle (no measurement) falls back to the ramp
        # that was hand-tuned at 500k.
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
                citation=DEMO_META["citation"], dimensions=dims
            )

            # Substitutive Points LOD for the multi-million-paper cloud —
            # coarse merged levels when zoomed out (census-style wiring; coarse
            # splats stay pure per coloring via the `coloring` barrier). Gated
            # through substitutive_lod_or_flat so a warm-cache run without
            # torch/scipy still builds a (flat) viewable scene.
            scene.add_points(
                "arxiv_papers_kaggle",
                positions=stacked.positions,
                colors=stacked.colors,
                radii=radii,
                sharpness=np.full(len(stacked.positions), 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.1,
                labels=stacked.labels,
                layer=True,
                substitutive_lod=substitutive_lod_or_flat(
                    dict(compression_factor=8, levels=3, device="auto")
                ),
            )

            # --- Overlays ---
            scene.add_text(
                "arXiv · bioRxiv · medRxiv — 3.3M Papers",
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
            for cat, _ in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
                r, g, b = (
                    int(round(v * 255))
                    for v in CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])
                )
                _cat_legend += f'<div><span style="color:#{r:02x}{g:02x}{b:02x}">█</span> {cat}</div>'
            _cat_legend += "</div>"
            # x=0.05, not 0.02: the legend grew to ten rows over the full
            # corpus and its lower entries ran under the viewer's icon rail.
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
            scene.add_text(
                f"{n_papers:,} papers • OpenAI text-embedding-3-large • "
                f"PCA-{pca_dim} → UMAP 3D",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Visualization created with {n_papers:,} papers")

    return n_papers


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_args(argv: list[str]) -> tuple[int | None, int, str, int]:
    """Parse the demo's ``--flag=value`` arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        ``(sample_size, pca_dim, device, seed)``; ``sample_size`` is ``None``
        for the whole corpus (the default, and what ``--sample=all`` spells).
    """
    sample_size = DEFAULT_SAMPLE_SIZE
    pca_dim = DEFAULT_PCA_DIM
    device = "auto"
    seed = 0

    for arg in argv:
        if arg.startswith("--sample="):
            value = arg.split("=", 1)[1]
            sample_size = None if value in ("all", "full") else int(value)
        elif arg.startswith("--pca-dim="):
            pca_dim = int(arg.split("=", 1)[1])
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
        elif arg.startswith("--seed="):
            seed = int(arg.split("=", 1)[1])

    return sample_size, pca_dim, device, seed


def main() -> None:
    """Main demo entry point."""
    sample_size, pca_dim, device, seed = parse_args(sys.argv[1:])

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
