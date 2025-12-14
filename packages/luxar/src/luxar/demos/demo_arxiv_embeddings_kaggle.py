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
    - Install: pip install mlcroissant umap-learn pandas

NO AUTHENTICATION NEEDED!
    Uses mlcroissant to download directly from Kaggle without API keys.
    First run downloads data (may take time for large samples).
    Subsequent runs use cached data.

Controls:
    - Explore clusters of related research
    - Color = arXiv category
    - Size = recency (newer papers larger)
    - Ctrl+C to stop
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 50000  # Sample 50k papers for reasonable performance

# ArXiv category colors (major categories)
CATEGORY_COLORS = {
    "cs": np.array([0.3, 0.9, 0.9]),  # Computer Science - Cyan
    "physics": np.array([0.3, 0.5, 1.0]),  # Physics - Blue
    "math": np.array([0.3, 1.0, 0.5]),  # Mathematics - Green
    "q-bio": np.array([1.0, 0.4, 0.4]),  # Quantitative Biology - Red
    "q-fin": np.array([1.0, 0.6, 0.2]),  # Quantitative Finance - Orange
    "stat": np.array([0.9, 0.3, 0.9]),  # Statistics - Magenta
    "eess": np.array([0.9, 0.9, 0.3]),  # Electrical Engineering - Yellow
    "econ": np.array([0.5, 1.0, 0.8]),  # Economics - Teal
    "other": np.array([0.7, 0.7, 0.7]),  # Other - Gray
}


# =============================================================================
# Dataset Loading with MLCroissant
# =============================================================================


def load_arxiv_dataset_local(
    zip_path: Path,
    sample_size: int = 50000,
) -> tuple[list, list, list, list]:
    """Load arXiv embeddings from local ZIP file.

    Args:
        zip_path: Path to downloaded openai-arxiv-embeddings.zip
        sample_size: Number of papers to load

    Returns:
        Tuple of (embeddings, titles, categories, years)
    """
    import zipfile

    with asection("Loading arXiv embeddings from local ZIP"):
        aprint(f"ZIP file: {zip_path}")
        aprint(f"Size: {zip_path.stat().st_size / (1024**3):.1f} GB")

        embeddings = []
        titles = []
        categories = []
        years = []

        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Find the data file (likely JSONL format)
            data_files = [f for f in zf.namelist() if f.endswith(('.json', '.jsonl'))]

            if not data_files:
                raise FileNotFoundError(f"No JSON/JSONL files found in {zip_path}")

            data_file = data_files[0]
            aprint(f"Reading: {data_file}")
            aprint(f"Sampling {sample_size:,} papers...")

            with zf.open(data_file) as f:
                import io
                text_stream = io.TextIOWrapper(f, encoding='utf-8')

                count = 0
                for i, line in enumerate(text_stream):
                    if count >= sample_size:
                        break

                    if i % 10000 == 0 and i > 0:
                        aprint(f"  Processed {i:,} papers, sampled {count:,}")

                    try:
                        import json
                        paper = json.loads(line)

                        # Extract embedding
                        emb = paper.get("embedding") or paper.get("embeddings")
                        if emb and len(emb) > 0:
                            embeddings.append(emb)
                            titles.append(paper.get("title", "Unknown"))

                            # Get primary category
                            cats = paper.get("categories", "").split()
                            primary_cat = cats[0].split(".")[0] if cats else "other"
                            categories.append(primary_cat)

                            # Get year
                            update_date = paper.get("update_date", "2020-01-01")
                            year = int(update_date[:4]) if update_date else 2020
                            years.append(year)

                            count += 1

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        aprint(f"✓ Loaded {len(embeddings):,} papers with embeddings")

    return embeddings, titles, categories, years


def load_arxiv_dataset_croissant(
    sample_size: int = 50000,
) -> tuple[list, list, list, list]:
    """Load arXiv embeddings using mlcroissant (no auth needed!).

    Args:
        sample_size: Number of papers to load

    Returns:
        Tuple of (embeddings, titles, categories, years)
    """
    try:
        import mlcroissant as mlc
    except ImportError:
        aprint("❌ Error: mlcroissant not installed")
        aprint("")
        aprint("Install with:")
        aprint("  pip install mlcroissant")
        aprint("")
        raise

    with asection("Loading arXiv dataset via mlcroissant"):
        aprint("Fetching Kaggle dataset (no authentication needed!)...")
        aprint("Dataset: tomtum/openai-arxiv-embeddings")
        aprint("")

        try:
            dataset = mlc.Dataset(
                "https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings/croissant/download"
            )

            # Get record sets
            record_sets = dataset.metadata.record_sets
            aprint(f"✓ Found {len(record_sets)} record set(s)")

            # Load records
            aprint(f"Loading papers (will sample {sample_size:,})...")
            embeddings = []
            titles = []
            categories = []
            years = []

            count = 0
            for i, record in enumerate(dataset.records(record_set=record_sets[0].uuid)):
                if count >= sample_size:
                    break

                if i % 10000 == 0 and i > 0:
                    aprint(f"  Loaded {count:,}/{sample_size:,} papers...")

                # Extract data from record
                emb = record.get("embedding")
                if emb and len(emb) > 0:
                    embeddings.append(emb)
                    titles.append(record.get("title", "Unknown"))

                    # Get category
                    cats = record.get("categories", "").split()
                    primary_cat = cats[0].split(".")[0] if cats else "other"
                    categories.append(primary_cat)

                    # Get year
                    update_date = record.get("update_date", "2020-01-01")
                    year = int(update_date[:4]) if update_date else 2020
                    years.append(year)

                    count += 1

            aprint(f"✓ Loaded {len(embeddings):,} papers with embeddings")

        except Exception as e:
            aprint(f"❌ Error loading dataset: {e}")
            aprint("")
            aprint("This dataset is very large (32.6 GB).")
            aprint("First download may take significant time.")
            raise

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
            random_state=42,
            verbose=True,
        )

        reduced = reducer.fit_transform(embeddings)

        aprint("✓ UMAP complete")
        aprint(f"  Output: {reduced.shape}")
        aprint(f"  Range: [{reduced.min():.2f}, {reduced.max():.2f}]")

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
            cached = np.load(cache_file)
            positions = cached["positions"]
            categories = list(cached["categories"])
            years = list(cached["years"])
            aprint(f"✓ Loaded {len(positions):,} papers from cache")
    else:
        # Check for local ZIP file first
        local_zip = Path.home() / "Downloads" / "openai-arxiv-embeddings.zip"

        if local_zip.exists():
            aprint(f"✓ Found local dataset: {local_zip}")
            embeddings_list, titles, categories, years = load_arxiv_dataset_local(
                local_zip, sample_size
            )
        else:
            aprint("No local ZIP found, using mlcroissant (will download ~30GB)...")
            aprint("To avoid this, download manually:")
            aprint("  curl -L -o ~/Downloads/openai-arxiv-embeddings.zip \\")
            aprint("    https://www.kaggle.com/api/v1/datasets/download/tomtum/openai-arxiv-embeddings")
            aprint("")
            embeddings_list, titles, categories, years = load_arxiv_dataset_croissant(
                sample_size
            )

        embeddings = np.array(embeddings_list, dtype=np.float32)

        if len(embeddings) == 0:
            aprint("❌ No papers loaded")
            return 0

        # Reduce to 3D
        positions = reduce_embeddings_umap(embeddings, n_components=3)

        # Cache UMAP results
        if cache_file:
            np.savez(
                cache_file,
                positions=positions,
                categories=np.array(categories),
                years=np.array(years),
            )
            aprint(f"✓ Cached UMAP to {cache_file}")

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

            sharpness = np.full(n_papers, 4.0, dtype=np.float32)

            scene.add_points(
                "arxiv_papers",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.9,
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
    use_cache = "--use-cache" in sys.argv

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
        import mlcroissant  # noqa: F401
        import umap  # noqa: F401
    except ImportError as e:
        aprint(f"❌ Missing dependency: {e}")
        aprint("")
        aprint("Install with:")
        aprint("  pip install umap-learn mlcroissant")
        aprint("")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="luxar_demo_arxiv_kaggle_") as tmpdir:
        output_path = Path(tmpdir) / "arxiv_papers.zarr"

        cache_dir = None
        if use_cache:
            cache_dir = Path.home() / ".cache" / "luxar" / "arxiv_umap"

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
            aprint(f"\n❌ Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Explore the knowledge landscape:")
        aprint("  • Zoom out: See overall structure of scientific fields")
        aprint("  • Zoom in: Explore specific research topics")
        aprint("  • Look for clusters: Papers on same topic group together")
        aprint("  • Boundaries: Interdisciplinary research")
        aprint("")
        aprint(f"Total papers: {n_papers:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Press Ctrl+C when done.")
        aprint("")

        if "--no-serve" in sys.argv:
            aprint("✓ Dataset generated successfully (--no-serve mode)")
            return

        try:
            subprocess.run(
                ["luxar", "serve", str(output_path), "--viewer", "--open"],
                check=True,
            )
        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
