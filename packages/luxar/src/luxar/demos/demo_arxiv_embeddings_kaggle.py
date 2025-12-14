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

SAMPLING STRATEGY:
Since the full dataset is 32.6 GB, we'll sample:
- Download the dataset file
- Load and sample N papers (default: 50,000)
- Apply UMAP to reduce 3072D → 3D
- Visualize!

================================================================================

Usage:
    python demo_arxiv_embeddings_kaggle.py [--sample=N]

    Options:
    --sample=N       Number of papers to sample (default: 50000)
    --categories=X   Filter by arXiv category (e.g., cs.AI, physics.atom-ph)
    --use-cache      Use cached UMAP coordinates

Requirements:
    - Kaggle API credentials (kaggle.json in ~/.kaggle/)
    - Install: pip install kaggle umap-learn

Setup Kaggle API:
    1. Go to https://www.kaggle.com/settings
    2. Create API token (downloads kaggle.json)
    3. Place in ~/.kaggle/kaggle.json
    4. chmod 600 ~/.kaggle/kaggle.json

ALTERNATIVE (No Kaggle Auth):
If you don't have Kaggle credentials, manually:
    1. Download from https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings
    2. Extract the ZIP
    3. Run: python demo_arxiv_embeddings_kaggle.py --local=path/to/data

Controls:
    - Explore clusters of related research
    - Color = arXiv category
    - Size = recency (newer papers larger)
    - Ctrl+C to stop
"""

import json
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
# Kaggle Dataset Download
# =============================================================================


def download_kaggle_dataset(dataset_name: str, download_path: Path) -> Path:
    """Download dataset from Kaggle using kaggle CLI.

    Requires kaggle API credentials in ~/.kaggle/kaggle.json

    Args:
        dataset_name: Kaggle dataset identifier
        download_path: Where to download

    Returns:
        Path to downloaded data
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        aprint("❌ Error: kaggle package not installed")
        aprint("")
        aprint("Install with:")
        aprint("  pip install kaggle")
        aprint("")
        aprint("Then setup API credentials:")
        aprint("  1. Go to https://www.kaggle.com/settings")
        aprint("  2. Create API token (downloads kaggle.json)")
        aprint("  3. Place in ~/.kaggle/kaggle.json")
        aprint("  4. chmod 600 ~/.kaggle/kaggle.json")
        raise

    with asection(f"Downloading Kaggle dataset: {dataset_name}"):
        download_path.mkdir(parents=True, exist_ok=True)

        aprint("Downloading from Kaggle...")
        aprint("⚠️  This dataset is 32.6 GB - download may take 10-30 minutes")
        aprint("   Consider using --sample to limit the amount loaded")

        try:
            import kaggle.api
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                dataset_name, path=str(download_path), unzip=True
            )
            aprint(f"✓ Downloaded to {download_path}")

        except Exception as e:
            aprint(f"❌ Error: {e}")
            aprint("")
            aprint("Alternatives:")
            aprint("  1. Download manually from Kaggle website")
            aprint("  2. Use --local=path/to/data flag")
            raise

    return download_path


# =============================================================================
# Data Loading and Sampling
# =============================================================================


def load_and_sample_embeddings(
    data_path: Path,
    sample_size: int = 50000,
    category_filter: str | None = None,
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Load embeddings and metadata from downloaded dataset.

    Args:
        data_path: Path to extracted dataset
        sample_size: Number of papers to sample
        category_filter: Optional arXiv category filter (e.g., "cs.AI")

    Returns:
        Tuple of (embeddings, titles, categories, years)
    """
    with asection("Loading arXiv embeddings from dataset"):
        # Find the data file (might be JSON, parquet, or CSV)
        data_files = list(data_path.glob("*.json")) + list(
            data_path.glob("*.parquet")
        )

        if not data_files:
            aprint(f"❌ No data files found in {data_path}")
            raise FileNotFoundError(f"No data files in {data_path}")

        aprint(f"Found data file: {data_files[0].name}")
        aprint(f"Size: {data_files[0].stat().st_size / (1024**3):.2f} GB")

        # Load data (this part depends on actual file format)
        # For now, assuming JSON Lines format
        embeddings = []
        titles = []
        categories = []
        years = []

        aprint(f"Loading and sampling {sample_size:,} papers...")

        if data_files[0].suffix == ".json":
            with open(data_files[0]) as f:
                count = 0
                for line_num, line in enumerate(f):
                    if count >= sample_size:
                        break

                    if line_num % 10000 == 0:
                        aprint(f"  Processed {line_num:,} lines, sampled {count:,} papers")

                    try:
                        paper = json.loads(line)

                        # Filter by category if specified
                        if category_filter:
                            paper_cats = paper.get("categories", "").split()
                            if not any(
                                category_filter in cat for cat in paper_cats
                            ):
                                continue

                        # Extract data
                        emb = paper.get("embedding")
                        if emb and len(emb) > 0:
                            embeddings.append(emb)
                            titles.append(paper.get("title", "Unknown"))
                            # Get primary category
                            cats = paper.get("categories", "").split()
                            primary_cat = cats[0] if cats else "other"
                            categories.append(primary_cat.split(".")[0])  # Main category
                            # Parse year from update_date
                            year_str = paper.get("update_date", "2020-01-01")
                            year = int(year_str[:4]) if year_str else 2020
                            years.append(year)
                            count += 1

                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        aprint(f"✓ Loaded {len(embeddings):,} papers with embeddings")

    return (
        np.array(embeddings, dtype=np.float32),
        titles,
        categories,
        years,
    )


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
    data_path: Path,
    sample_size: int = 50000,
    category_filter: str | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Generate 3D landscape of arXiv papers.

    Args:
        output_path: Where to write zarr
        data_path: Path to Kaggle dataset
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
        # Load data
        embeddings, titles, categories, years = load_and_sample_embeddings(
            data_path, sample_size, category_filter
        )

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
    local_path = None
    use_cache = "--use-cache" in sys.argv

    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
        elif arg.startswith("--category="):
            category_filter = arg.split("=")[1]
        elif arg.startswith("--local="):
            local_path = Path(arg.split("=")[1])

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
        aprint("❌ Missing dependency: umap-learn")
        aprint("")
        aprint("Install with:")
        aprint("  pip install umap-learn")
        sys.exit(1)

    # Determine data path
    if local_path:
        data_path = local_path
        aprint(f"Using local data: {data_path}")
    else:
        # Check if kaggle is installed
        try:
            import kaggle  # noqa: F401
        except ImportError:
            aprint("❌ Error: kaggle package not installed")
            aprint("")
            aprint("Option 1 - Install Kaggle CLI:")
            aprint("  pip install kaggle")
            aprint("  Setup credentials: https://www.kaggle.com/docs/api")
            aprint("")
            aprint("Option 2 - Manual download:")
            aprint("  1. Download: https://www.kaggle.com/datasets/tomtum/openai-arxiv-embeddings")
            aprint("  2. Extract ZIP file")
            aprint("  3. Run: python demo_arxiv_embeddings_kaggle.py --local=path/to/data")
            aprint("")
            sys.exit(1)

        # Download dataset
        download_dir = Path.home() / ".cache" / "luxar" / "kaggle_datasets"
        data_path = download_dir / "openai-arxiv-embeddings"

        if not (data_path / "embeddings.json").exists():
            try:
                download_kaggle_dataset("tomtum/openai-arxiv-embeddings", download_dir)
            except Exception as e:
                aprint(f"\n❌ Download failed: {e}")
                aprint("\nTry manual download instead (see instructions above)")
                sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="luxar_demo_arxiv_kaggle_") as tmpdir:
        output_path = Path(tmpdir) / "arxiv_papers.zarr"

        cache_dir = None
        if use_cache:
            cache_dir = Path.home() / ".cache" / "luxar" / "arxiv_umap"

        try:
            n_papers = generate_paper_landscape(
                output_path,
                data_path,
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
