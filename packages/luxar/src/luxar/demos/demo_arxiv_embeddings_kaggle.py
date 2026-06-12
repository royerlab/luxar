#!/usr/bin/env python3
"""Self-Contained Demo: ArXiv Paper Embeddings from Kaggle Dataset

Visualize scientific papers from arXiv using pre-computed OpenAI embeddings
from the Kaggle "openai-arxiv-embeddings" dataset.

================================================================================
DATASET: OpenAI ArXiv Embeddings
================================================================================

Source: https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings

This dataset contains:
- ALL arXiv papers (2M+) with pre-computed embeddings
- OpenAI text-embedding-3-large (3,072 dimensions)
- Title, abstract, categories, authors, dates
- Total size: 32.6 GB (we'll sample a subset!)

WHY THIS IS FASTER:
- No need to download papers from Semantic Scholar API
- No need to compute embeddings (already done!)
- Just download → UMAP → visualize!

DOWNLOAD & CACHING STRATEGY:
============================

FIRST RUN (one-time, ~8-15 minutes):
  ✓ Downloads full 30.4 GB dataset from Kaggle
  ✓ mlcroissant caches to: ~/.cache/mlcroissant/
  ✓ ONE-TIME cost for PERMANENT access to 2M+ papers!
  ✓ Download at ~60MB/s (your network speed)

SUBSEQUENT RUNS (super fast!):
  ✓ Reads from cached dataset (no re-download!)
  ✓ Samples N papers instantly
  ✓ UMAP reduction: 1-3 minutes for 50k papers
  ✓ With --use-cache: <30 seconds total!

THIS IS WORTH IT:
  - Download once → visualize ANY subset forever
  - 10k papers? Instant!
  - 100k papers? ~2 minutes
  - 500k papers? ~10 minutes
  - 1M papers? ~20 minutes (just UMAP)
  - ALL 2M papers? ~45 minutes (epic!)

================================================================================

QUICK START:
============
1. Download the dataset (one-time, ~30GB, 10-15 min):
   curl -L -o ~/Downloads/openai-arxiv-embeddings.zip \\
     https://www.kaggle.com/api/v1/datasets/download/tomtum/openai-arxiv-embeddings

2. Run the demo:
   python demo_arxiv_embeddings_kaggle.py --sample=50000 --use-cache

3. Subsequent runs are INSTANT (uses cached data + UMAP)!

Usage:
    python demo_arxiv_embeddings_kaggle.py [--sample=N]

    Options:
    --sample=N       Number of papers to sample (default: 50000)
    --categories=X   Filter by arXiv category (e.g., cs.AI, physics.atom-ph)
    --use-cache      Use cached UMAP coordinates (RECOMMENDED!)

Requirements:
    - Install: pip install umap-learn

NO AUTHENTICATION NEEDED!
    Downloads directly from Kaggle API (no login required).
    First run downloads 30GB dataset (~10-15 min).
    Cached to ~/.cache/luxar/ for instant subsequent runs!

Controls:
    - Explore clusters of related research
    - Color = arXiv category
    - Size = recency (newer papers larger)
    - Ctrl+C to stop
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 500000  # 500k papers

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
    # Other/Unmatched
    "other": np.array([0.5, 0.5, 0.5]),  # Dark Gray (not white!)
}


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
# Dataset Loading
# =============================================================================


def load_arxiv_dataset_local(
    zip_path: Path,
    metadata_lookup: dict,
    sample_size: int = 50000,
) -> tuple[list, list, list, list]:
    """Load arXiv embeddings and match with metadata.

    The Kaggle embeddings dataset contains:
    - papers.csv: paper IDs
    - vectors.dat: binary embeddings (float32, 3072-dim per paper)

    Args:
        zip_path: Path to downloaded openai-arxiv-embeddings.zip
        metadata_lookup: Dictionary mapping paper_id -> {category, title, year}
        sample_size: Number of papers to load

    Returns:
        Tuple of (embeddings, titles, categories, years)
    """
    import struct
    import zipfile

    import pandas as pd

    with asection("Loading arXiv embeddings from local ZIP"):
        aprint(f"ZIP file: {zip_path}")
        aprint(f"Size: {zip_path.stat().st_size / (1024**3):.1f} GB")

        with zipfile.ZipFile(zip_path, "r") as zf:
            aprint(f"Files in ZIP: {', '.join(zf.namelist())}")

            # Load metadata from CSV
            aprint("Loading papers.csv...")
            with zf.open("papers.csv") as f:
                papers_df = pd.read_csv(f, nrows=sample_size)

            aprint(f"✓ Loaded {len(papers_df):,} paper metadata entries")

            # Load embeddings from binary file
            aprint("Loading vectors.dat (binary embeddings)...")
            with zf.open("vectors.dat") as f:
                # Read binary data
                # Format: each embedding is 3072 float32 values (12,288 bytes)
                embedding_dim = 3072
                bytes_per_embedding = embedding_dim * 4  # float32 = 4 bytes

                embeddings = []
                for i in range(min(sample_size, len(papers_df))):
                    # Read one embedding
                    emb_bytes = f.read(bytes_per_embedding)
                    if len(emb_bytes) < bytes_per_embedding:
                        break

                    # Unpack floats
                    emb = struct.unpack(f"{embedding_dim}f", emb_bytes)
                    embeddings.append(list(emb))

                    if (i + 1) % 10000 == 0:
                        aprint(f"  Loaded {i + 1:,} embeddings...")

            aprint(f"✓ Loaded {len(embeddings):,} embeddings")

        # Match paper IDs with metadata to get real categories
        paper_ids = papers_df["id"].tolist()[: len(embeddings)]

        aprint(f"Matching {len(paper_ids):,} paper IDs with metadata...")
        titles = []
        categories = []
        years = []
        matched = 0

        for pid in paper_ids:
            pid_str = str(pid)

            if pid_str in metadata_lookup:
                meta = metadata_lookup[pid_str]
                titles.append(meta["title"])
                # Extract main category (e.g., "cs.AI" -> "cs")
                cat = (
                    meta["category"].split(".")[0]
                    if "." in meta["category"]
                    else meta["category"]
                )
                categories.append(cat)
                years.append(int(meta["year"]))
                matched += 1
            else:
                # Fallback for unmatched
                titles.append(f"arXiv:{pid_str}")
                categories.append("other")
                years.append(2015)

        match_rate = matched / len(paper_ids) * 100 if paper_ids else 0
        aprint(f"✓ Matched {matched:,}/{len(paper_ids):,} papers ({match_rate:.1f}%)")
        aprint(f"✓ Found {len(set(categories))} unique categories")
        aprint(f"✓ Year range: {min(years)} - {max(years)}")

    return embeddings, titles, categories, years


# =============================================================================
# UMAP Dimensionality Reduction
# =============================================================================


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 3D using UMAP.

    Args:
        embeddings: (n_samples, n_features) array
        n_components: Target dimensions (3 for visualization)
        n_neighbors: UMAP parameter

    Returns:
        (n_samples, n_components) reduced coordinates
    """
    try:
        from umap import UMAP
    except ImportError:
        aprint("❌ Error: umap-learn not installed")
        aprint("")
        aprint("Install with:")
        aprint("  pip install umap-learn")
        raise

    with asection(f"Reducing {embeddings.shape[1]}D → {n_components}D with UMAP"):
        aprint(f"Input: {embeddings.shape}")
        aprint(f"Parameters: n_neighbors={n_neighbors}, metric=cosine")

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

        reduced = reducer.fit_transform(embeddings)

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
# Visualization Generation
# =============================================================================


def generate_paper_landscape(
    output_path: Path,
    sample_size: int = 50000,
    category_filter: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Generate 3D landscape of arXiv papers.

    Args:
        output_path: Where to write zarr
        sample_size: Number of papers to sample
        category_filter: Optional category filter
        cache_dir: Optional cache for UMAP results

    Returns:
        Number of papers visualized
    """
    # Check cache for UMAP results
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"umap_{sample_size}_{category_filter or 'all'}.npz"

    if cache_file and cache_file.exists():
        with asection("Loading cached UMAP coordinates"):
            cached = np.load(cache_file, allow_pickle=True)
            positions = cached["positions"]
            categories = list(cached["categories"])
            years = list(cached["years"])
            titles = list(cached["titles"]) if "titles" in cached else None
            aprint(f"✓ Loaded {len(positions):,} papers from cache")
    else:
        # Check for cached embeddings dataset
        dataset_cache = Path.home() / ".cache" / "luxar" / "arxiv_embeddings.zip"
        metadata_cache = Path.home() / ".cache" / "luxar" / "arxiv_metadata.json"
        expected_emb_size_gb = 30  # Expected embeddings size
        download_marker = dataset_cache.parent / ".arxiv_downloading"

        # Check if download is in progress
        if download_marker.exists():
            aprint("⚠️  Download already in progress in another process!")
            aprint("   Please wait for it to complete or delete the marker:")
            aprint(f"   rm {download_marker}")
            raise RuntimeError("Download in progress")

        # Check if embeddings file exists and is complete
        if dataset_cache.exists():
            size_gb = dataset_cache.stat().st_size / (1024**3)
            if size_gb < expected_emb_size_gb * 0.9:  # Allow 10% variance
                aprint(
                    f"⚠️  Cached embeddings incomplete ({size_gb:.1f} GB / ~{expected_emb_size_gb} GB)"
                )
                aprint("   Deleting and re-downloading...")
                dataset_cache.unlink()
            else:
                aprint(f"✓ Using cached embeddings: {dataset_cache}")
                aprint(f"  Size: {size_gb:.1f} GB")

        if not dataset_cache.exists():
            aprint("Dataset not in cache, downloading...")
            # Create marker file
            download_marker.parent.mkdir(parents=True, exist_ok=True)
            download_marker.touch()
            try:
                download_kaggle_dataset(dataset_cache)
            finally:
                # Remove marker when done (or on failure)
                if download_marker.exists():
                    download_marker.unlink()

        # Download and load metadata
        if not metadata_cache.exists():
            aprint("Metadata not in cache, downloading...")
            # First download to Downloads, then extract
            meta_zip = Path.home() / "Downloads" / "arxiv-metadata.zip"
            if not meta_zip.exists():
                aprint("Downloading arXiv metadata (1.5GB compressed)...")
                download_kaggle_dataset(
                    meta_zip,
                    url="https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv",
                )

            # Extract metadata
            import zipfile

            aprint("Extracting metadata...")
            with zipfile.ZipFile(meta_zip, "r") as zf:
                zf.extract("arxiv-metadata-oai-snapshot.json", metadata_cache.parent)
                # Rename to cache location
                extracted = metadata_cache.parent / "arxiv-metadata-oai-snapshot.json"
                if extracted.exists():
                    extracted.rename(metadata_cache)
            aprint(f"✓ Metadata extracted to {metadata_cache}")

        # Load metadata lookup (with caching!)
        metadata_lookup_cache = (
            Path.home() / ".cache" / "luxar" / "arxiv_metadata_lookup.pkl"
        )
        metadata_lookup = load_metadata_lookup(
            metadata_cache, cache_path=metadata_lookup_cache
        )

        # Load from cached ZIP with metadata matching
        embeddings_list, titles, categories, years = load_arxiv_dataset_local(
            dataset_cache, metadata_lookup, sample_size
        )

        embeddings = np.array(embeddings_list, dtype=np.float32)

        if len(embeddings) == 0:
            aprint("❌ No papers loaded")
            return 0

        # Reduce to 3D
        positions = reduce_embeddings_umap(embeddings, n_components=3)

        # Cache UMAP results for instant future runs
        aprint(
            f"[DEBUG] About to cache. cache_file={cache_file}, cache_dir={cache_dir}"
        )
        with asection("Saving UMAP cache"):
            if cache_file:
                aprint(f"Cache path: {cache_file}")
                np.savez(
                    cache_file,
                    positions=positions,
                    categories=np.array(categories),
                    years=np.array(years),
                    titles=np.array(titles, dtype=object),
                )
                aprint("✓ UMAP cached successfully!")
                aprint("  Next run with same sample size will be INSTANT!")
            else:
                aprint("⚠️  Cache not enabled (--use-cache flag needed)")

    # Generate visualization
    with asection("Generating visualization"):
        n_papers = len(positions)

        # Colors by category
        colors = np.zeros((n_papers, 3), dtype=np.float32)
        for i, cat in enumerate(categories):
            colors[i] = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])

        # Count categories
        cat_counts = {}
        for cat in categories:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        aprint("✓ Papers by category:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
            aprint(f"  {cat}: {count:,}")

        # Size by recency (newer papers = larger)
        year_array = np.array(years, dtype=np.float32)
        year_norm = (year_array - year_array.min()) / (
            year_array.max() - year_array.min() + 1
        )
        radii = (0.03 + 0.07 * year_norm).astype(np.float32)

        aprint("✓ Sizes by year:")
        aprint(f"  Year range: {int(year_array.min())} to {int(year_array.max())}")

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            sharpness = np.full(n_papers, 0.6, dtype=np.float32)

            # Hover labels: title + year + category
            if titles is not None:
                paper_labels = [
                    f"{titles[i][:60]}{'…' if len(titles[i]) > 60 else ''} ({years[i]}, {categories[i]})"
                    for i in range(n_papers)
                ]
            else:
                paper_labels = [
                    f"{categories[i]} ({years[i]})" for i in range(n_papers)
                ]

            scene.add_points(
                "arxiv_papers",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.9,
                intensity=0.1,
                labels=paper_labels,
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "ArXiv Paper Embeddings",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info + source
            scene.add_text(
                f"{n_papers:,} papers • OpenAI embeddings • Kaggle dataset",
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


def main() -> None:
    """Main demo entry point."""
    sample_size = DEFAULT_SAMPLE_SIZE
    category_filter = None
    _use_cache = True  # Caching is ALWAYS on by default (reserved for future use)

    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
        elif arg.startswith("--category="):
            category_filter = arg.split("=")[1]

    aprint("=" * 70)
    aprint("ARXIV PAPER EMBEDDINGS - PRE-COMPUTED FROM KAGGLE")
    aprint("=" * 70)
    aprint("")
    aprint("Dataset: OpenAI ArXiv Embeddings (Kaggle)")
    aprint("https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings")
    aprint("")
    aprint("What this shows:")
    aprint("  • 2M+ arXiv papers with pre-computed OpenAI embeddings")
    aprint("  • 3D UMAP projection of 3,072-dimensional vectors")
    aprint("  • Papers cluster by research topic automatically")
    aprint("  • Color = arXiv category (cs, physics, math, etc.)")
    aprint("  • Size = recency (newer papers are larger)")
    aprint("")
    aprint("Parameters:")
    aprint(f"  Sample size: {sample_size:,} papers")
    if category_filter:
        aprint(f"  Category filter: {category_filter}")
    aprint("")

    # Check dependencies
    try:
        import umap  # noqa: F401
    except ImportError:
        aprint("Missing dependency: umap-learn")
        aprint("")
        aprint("Install with:")
        aprint("  pip install umap-learn")
        aprint("")
        sys.exit(1)

    # ALWAYS use cache
    cache_dir = Path.home() / ".cache" / "luxar" / "arxiv_umap"

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "arxiv_papers_kaggle.luxar.zarr"
        try:
            n_papers = generate_paper_landscape(
                output_path,
                sample_size=sample_size,
                category_filter=category_filter,
                cache_dir=cache_dir,
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
        output_path = Path(tmpdir) / "arxiv_papers.luxar.zarr"

        try:
            n_papers = generate_paper_landscape(
                output_path,
                sample_size=sample_size,
                category_filter=category_filter,
                cache_dir=cache_dir,
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
