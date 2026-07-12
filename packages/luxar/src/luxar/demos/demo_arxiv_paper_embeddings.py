#!/usr/bin/env python3
"""Self-Contained Demo: ArXiv Paper Embeddings - Landscape of Scientific Knowledge

Visualize scientific papers from arXiv in 3D embedding space, showing how
different research areas cluster together based on semantic similarity.

================================================================================
SCIENTIFIC KNOWLEDGE AS A LANDSCAPE
================================================================================

This visualization shows scientific papers as points in 3D space, where:
- **Distance = Semantic similarity** (closer papers are more related)
- **Color = Research field** (physics, CS, biology, math, etc.)
- **Size = Citation count** (larger = more influential)

The 3D coordinates are computed using:
1. **Embeddings**: Convert paper abstracts to 768D vectors using Sentence-BERT
2. **UMAP**: Reduce 768D → 3D while preserving semantic relationships

WHAT YOU'LL SEE:
- Papers naturally cluster by topic (ML, quantum physics, genomics, etc.)
- Interdisciplinary papers appear between clusters
- Citation-heavy papers stand out with larger size
- Beautiful "topology" of human knowledge

DATA SOURCE:
- Semantic Scholar API (https://api.semanticscholar.org/)
- ArXiv papers across multiple fields
- Open access, no authentication required

EMBEDDING METHOD:
We use Sentence-BERT (all-MiniLM-L6-v2) to embed paper abstracts.
This creates 384-dimensional vectors that capture semantic meaning.

UMAP (Uniform Manifold Approximation and Projection) then reduces these
to 3D while preserving the neighborhood structure - papers on similar
topics remain close together.

FIELDS OF STUDY:
- Computer Science (cyan)
- Physics (blue)
- Mathematics (green)
- Biology (red)
- Medicine (orange)
- And more!

Usage:
    python demo_arxiv_paper_embeddings.py [--papers=N] [--fields=LIST]

    Options:
    --papers=N          Number of papers per field (default: 1000)
    --fields=cs,physics Fields to include (default: cs,physics,biology,medicine,math)

    Embeddings + 3D UMAP are cached automatically under ~/.cache/luxar/arxiv_paper
    (keyed on fields + papers-per-field), so repeat runs are instant.

Controls:
    - Rotate to explore the knowledge landscape
    - Zoom in to see individual research clusters
    - Different colors show different disciplines
    - Larger points = more citations
    - Ctrl+C to stop and cleanup

NOTES:
- First run downloads papers and computes embeddings
  * Small scale (5k papers): ~2-5 minutes
  * Large scale (100k papers): ~30-60 minutes
  * Very large scale (1M papers): Use bulk dataset API recommended!
- Subsequent runs load cached embeddings automatically
- Requires internet connection for Semantic Scholar API
- Required packages: sentence-transformers, umap-learn
- API rate limits: ~100 requests/second (use delays for large queries)

FOR 1M PAPERS:
Consider using Semantic Scholar's bulk datasets instead of API:
https://api.semanticscholar.org/datasets/v1/release/
Download pre-computed embeddings and metadata directly!
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import cache_computed, hsv_to_rgb, launch_viewer, stack_colorings
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"

# Default fields to query
DEFAULT_FIELDS = ["Computer Science", "Physics", "Biology", "Medicine", "Mathematics"]

# Papers per field
DEFAULT_PAPERS_PER_FIELD = 200000  # 200k per field → 1M total for 5 fields!

# Field colors (rainbow across disciplines)
FIELD_COLORS = {
    "Computer Science": np.array([0.3, 0.9, 0.9]),  # Cyan
    "Physics": np.array([0.3, 0.5, 1.0]),  # Blue
    "Mathematics": np.array([0.3, 1.0, 0.5]),  # Green
    "Biology": np.array([1.0, 0.4, 0.4]),  # Red
    "Medicine": np.array([1.0, 0.6, 0.2]),  # Orange
    "Chemistry": np.array([0.9, 0.3, 0.9]),  # Magenta
    "Engineering": np.array([0.9, 0.9, 0.3]),  # Yellow
    "Other": np.array([0.7, 0.7, 0.7]),  # Gray
}


# =============================================================================
# Semantic Scholar API
# =============================================================================


def search_papers_by_field(
    field: str,
    limit: int = 500,
    min_citations: int = 10,
) -> list[dict]:
    """Search for papers in a specific field using Semantic Scholar API.

    Args:
        field: Field of study (e.g., "Computer Science", "Physics")
        limit: Maximum papers to return
        min_citations: Minimum citation count filter

    Returns:
        List of paper dictionaries with title, abstract, citations, year, etc.
    """
    papers = []
    offset = 0
    batch_size = 100  # API limit per request

    with asection(f"Querying {field} papers"):
        aprint(
            f"Searching for papers (limit={limit}, min_citations={min_citations})..."
        )

        while len(papers) < limit:
            try:
                # Query parameters
                params = {
                    "query": field,
                    "fields": "title,abstract,citationCount,year,fieldsOfStudy,paperId",
                    "offset": offset,
                    "limit": min(batch_size, limit - len(papers)),
                }

                url = f"{S2_API_BASE}/paper/search"
                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 429:
                    aprint("  Rate limited, waiting 2 seconds...")
                    time.sleep(2)
                    continue

                response.raise_for_status()
                data = response.json()

                if "data" not in data or len(data["data"]) == 0:
                    break

                for paper in data["data"]:
                    # Filter by citations and abstract availability
                    if (
                        paper.get("abstract")
                        and paper.get("citationCount", 0) >= min_citations
                    ):
                        papers.append(paper)

                aprint(f"  Progress: {len(papers)}/{limit} papers")

                offset += batch_size

                # Rate limiting - be nice to API
                time.sleep(0.5)

                if len(papers) >= limit:
                    break

            except requests.exceptions.RequestException as e:
                aprint(f"  ⚠️  API error: {e}")
                break

        aprint(f"✓ Found {len(papers)} papers in {field}")

    return papers[:limit]


# =============================================================================
# Embedding Generation
# =============================================================================


def compute_text_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings for text using Sentence-BERT.

    Uses a lightweight model (all-MiniLM-L6-v2) that's fast and effective.

    Args:
        texts: List of text strings to embed

    Returns:
        Array of shape (n_texts, embedding_dim) with embeddings
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        aprint("❌ Error: sentence-transformers not installed")
        aprint("")
        aprint("Install with:")
        aprint("  pip install sentence-transformers")
        aprint("")
        raise

    with asection("Computing text embeddings"):
        aprint("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        aprint("✓ Model loaded (embedding dim: 384)")

        aprint(f"Encoding {len(texts)} texts...")
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )

        aprint(f"✓ Computed {len(embeddings)} embeddings")
        aprint(f"  Shape: {embeddings.shape}")

    return embeddings.astype(np.float32)


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 3D using UMAP.

    UMAP preserves both local and global structure, making it ideal
    for visualizing semantic relationships.

    Args:
        embeddings: (n_samples, n_features) array
        n_components: Target dimensions (3 for visualization)
        n_neighbors: UMAP parameter (smaller = focus on local structure)

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
        aprint("")
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
        aprint(f"  Output shape: {reduced.shape}")
        aprint(f"  Range: [{reduced.min():.2f}, {reduced.max():.2f}]")

    return reduced.astype(np.float32)


# =============================================================================
# Paper Processing
# =============================================================================


def download_papers_across_fields(
    fields: list[str],
    papers_per_field: int = 500,
) -> list[dict]:
    """Download papers from multiple fields using Semantic Scholar API.

    Args:
        fields: List of field names
        papers_per_field: Number of papers to get per field

    Returns:
        List of paper dictionaries with metadata
    """
    all_papers = []

    with asection(f"Downloading papers from {len(fields)} fields"):
        for field in fields:
            papers = search_papers_by_field(field, limit=papers_per_field)
            all_papers.extend(papers)

            aprint(f"  {field}: {len(papers)} papers")

        aprint(f"\n✓ Total: {len(all_papers)} papers")

    return all_papers


def prepare_paper_data(
    papers: list[dict],
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Extract relevant data from papers.

    Args:
        papers: List of paper dictionaries

    Returns:
        Tuple of (abstracts, primary_fields, citation_counts, years)
    """
    abstracts = []
    primary_fields = []
    citation_counts = []
    years = []

    for paper in papers:
        abstract = paper.get("abstract", "")
        if not abstract:
            continue

        # Get primary field of study
        fields = paper.get("fieldsOfStudy", [])
        if fields and len(fields) > 0:
            primary_field = fields[0]
        else:
            primary_field = "Other"

        abstracts.append(abstract)
        primary_fields.append(primary_field)
        citation_counts.append(paper.get("citationCount", 0))
        years.append(paper.get("year", 2020))

    return abstracts, primary_fields, citation_counts, years


# =============================================================================
# Visualization Generation
# =============================================================================


def generate_paper_landscape(
    output_path: Path,
    fields: list[str] = None,
    papers_per_field: int = 500,
) -> int:
    """Generate 3D landscape of scientific papers.

    Args:
        output_path: Where to write zarr
        fields: List of fields to include
        papers_per_field: Papers to download per field

    Returns:
        Total number of papers visualized
    """
    if fields is None:
        fields = DEFAULT_FIELDS

    def _compute_bundle() -> dict:
        # Download papers from Semantic Scholar
        papers = download_papers_across_fields(fields, papers_per_field)

        if len(papers) == 0:
            return {
                "papers_clean": [],
                "embeddings_3d": np.zeros((0, 3), dtype=np.float32),
                "fields": [],
                "citations": [],
            }

        # Prepare data
        with asection("Preparing paper data"):
            abstracts, primary_fields, citation_counts, years = prepare_paper_data(
                papers
            )
            aprint(f"✓ Prepared {len(abstracts)} papers with abstracts")

        # Compute embeddings
        embeddings = compute_text_embeddings(abstracts)

        # Reduce to 3D with UMAP
        embeddings_3d = reduce_embeddings_umap(embeddings, n_components=3)

        # Update papers with extracted data
        papers_clean = []
        for i, paper in enumerate(papers):
            if i < len(abstracts):
                papers_clean.append(
                    {
                        "title": paper.get("title", "Unknown"),
                        "field": primary_fields[i],
                        "citations": citation_counts[i],
                        "year": years[i],
                    }
                )

        return {
            "papers_clean": papers_clean,
            "embeddings_3d": embeddings_3d,
            "fields": primary_fields,
            "citations": citation_counts,
        }

    # Embeddings + UMAP are cached ON BY DEFAULT under ~/.cache/luxar/arxiv_paper,
    # keyed on the query (fields + papers-per-field), version=1.
    cache_key = f"embed3d_{'_'.join(fields)}_n{papers_per_field}"
    bundle = cache_computed("arxiv_paper", cache_key, _compute_bundle, version=1)
    papers_clean = bundle["papers_clean"]
    embeddings_3d = np.asarray(bundle["embeddings_3d"], dtype=np.float32)
    primary_fields = bundle["fields"]
    citation_counts = bundle["citations"]

    if len(embeddings_3d) == 0:
        aprint("❌ No papers found")
        return 0

    # Generate colors and sizes
    with asection("Generating visualization attributes"):
        n_papers = len(embeddings_3d)
        positions = embeddings_3d

        citation_array = np.array(citation_counts, dtype=np.float32)
        log_citations = np.log1p(citation_array)  # log(1 + x) to handle 0 citations
        years = np.array(
            [
                int(papers_clean[i].get("year", 2015)) if i < len(papers_clean) else 2015
                for i in range(n_papers)
            ],
            dtype=np.float32,
        )

        def _norm(a: np.ndarray) -> np.ndarray:
            rng = float(a.max() - a.min())
            return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

        # Three switchable coloring views: research field (categorical), plus
        # cool→warm sequential ramps over publication year and citation count.
        field_colors = np.array(
            [FIELD_COLORS.get(primary_fields[i], FIELD_COLORS["Other"]) for i in range(n_papers)],
            dtype=np.float32,
        )
        year_colors = hsv_to_rgb(0.66 * (1.0 - _norm(years)))  # old=blue → new=red
        citation_colors = hsv_to_rgb(0.66 * (1.0 - _norm(log_citations)))

        field_counts: dict[str, int] = {}
        for field in primary_fields:
            field_counts[field] = field_counts.get(field, 0) + 1
        aprint("✓ Colored by field / year / citations:")
        for field, count in sorted(field_counts.items(), key=lambda x: -x[1])[:10]:
            aprint(f"  {field}: {count} papers")

        # Per-point radii by citation count (log scale); tiled across views below.
        radii_pp = (0.02 + 0.08 * (log_citations / max(log_citations.max(), 1e-9))).astype(
            np.float32
        )

        def _title(i: int) -> str:
            t = (
                papers_clean[i].get("title", "Unknown")
                if i < len(papers_clean)
                else "Unknown"
            )
            return t[:60] + ("…" if len(t) > 60 else "")

        field_labels = [
            f"{_title(i)} ({int(citation_array[i])} cites, {primary_fields[i]})"
            for i in range(n_papers)
        ]
        year_labels = [f"{_title(i)} ({int(years[i])})" for i in range(n_papers)]
        citation_labels = [
            f"{_title(i)} ({int(citation_array[i])} cites)" for i in range(n_papers)
        ]

        stacked = stack_colorings(
            positions,
            [
                {"label": "Field", "colors": field_colors, "labels": field_labels},
                {"label": "Year", "colors": year_colors, "labels": year_labels},
                {"label": "Citations", "colors": citation_colors, "labels": citation_labels},
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
                    description="Color scheme: research field / year / citations",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Substitutive Points LOD for the (potentially large) paper cloud —
            # coarse merged levels when zoomed out (census-style wiring; coarse
            # splats stay pure per coloring via the `coloring` barrier).
            scene.add_points(
                "papers",
                positions=stacked.positions,
                colors=stacked.colors,
                radii=radii,
                sharpness=np.full(len(stacked.positions), 0.6, dtype=np.float32),
                opacity=0.9,
                intensity=0.1,
                labels=stacked.labels,
                substitutive_lod=dict(compression_factor=8, levels=3, device="auto"),
            )

            # --- Overlays ---
            # Title
            scene.add_text(
                "ArXiv Paper Landscape",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Field color legend (bottom-left)
            scene.add_html(
                '<div style="font-size:1.3vh;line-height:1.6;background:rgba(0,0,0,0.5);padding:0.6vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.4vh">Fields of Study</div>'
                '<div><span style="color:#4de6e6">\u2588</span> Computer Science</div>'
                '<div><span style="color:#4d80ff">\u2588</span> Physics</div>'
                '<div><span style="color:#4dff80">\u2588</span> Mathematics</div>'
                '<div><span style="color:#ff6666">\u2588</span> Biology</div>'
                '<div><span style="color:#ff9933">\u2588</span> Medicine</div>'
                "</div>",
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 0},
                transition="fade",
                transition_duration=0.3,
            )
            scene.add_html(
                '<div style="font-size:1.3vh;background:rgba(0,0,0,0.5);padding:0.6vh;'
                'border-radius:3px;color:#ccc">Year: '
                '<span style="color:#4d80ff">\u2588</span> older \u2192 '
                '<span style="color:#ff4d4d">\u2588</span> newer</div>',
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 1},
                transition="fade",
                transition_duration=0.3,
            )
            scene.add_html(
                '<div style="font-size:1.3vh;background:rgba(0,0,0,0.5);padding:0.6vh;'
                'border-radius:3px;color:#ccc">Citations: '
                '<span style="color:#4d80ff">\u2588</span> few \u2192 '
                '<span style="color:#ff4d4d">\u2588</span> many</div>',
                position=(0.02, 0.97),
                anchor="bottom-left",
                visible_range={"coloring": 2},
                transition="fade",
                transition_duration=0.3,
            )

            # Info + source
            scene.add_text(
                f"{n_papers:,} papers • Semantic Scholar API • SentenceBERT embeddings",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Scene created with {n_papers:,} papers")

    return n_papers


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    # Parse arguments
    papers_per_field = DEFAULT_PAPERS_PER_FIELD
    field_list = DEFAULT_FIELDS

    for arg in sys.argv[1:]:
        if arg.startswith("--papers="):
            papers_per_field = int(arg.split("=")[1])
        elif arg.startswith("--fields="):
            field_str = arg.split("=")[1]
            # Map short names to full field names
            field_map = {
                "cs": "Computer Science",
                "physics": "Physics",
                "bio": "Biology",
                "med": "Medicine",
                "math": "Mathematics",
                "chem": "Chemistry",
                "eng": "Engineering",
            }
            field_list = [
                field_map.get(f.strip(), f.strip()) for f in field_str.split(",")
            ]

    aprint("=" * 70)
    aprint("ARXIV PAPER EMBEDDINGS - LANDSCAPE OF KNOWLEDGE")
    aprint("=" * 70)
    aprint("")
    aprint("Visualize scientific papers in 3D semantic space!")
    aprint("")
    aprint("What this shows:")
    aprint("  • Each point = one scientific paper")
    aprint("  • Position = semantic similarity (close = related topics)")
    aprint("  • Color = field of study")
    aprint("  • Size = citation count (bigger = more influential)")
    aprint("")
    aprint("Method:")
    aprint("  1. Download papers from Semantic Scholar API")
    aprint("  2. Compute embeddings from abstracts (Sentence-BERT)")
    aprint("  3. Reduce 384D → 3D using UMAP")
    aprint("  4. Visualize the knowledge landscape!")
    aprint("")
    aprint("Parameters:")
    aprint(f"  Fields: {', '.join(field_list)}")
    aprint(f"  Papers per field: {papers_per_field}")
    aprint(f"  Total target: ~{len(field_list) * papers_per_field:,} papers")
    aprint("")
    aprint("⏱️  Expected time:")
    aprint("  • First run: 2-5 minutes (download + compute embeddings)")
    aprint("  • Cached run: <30 seconds (results cached automatically)")
    aprint("")

    # Check dependencies
    try:
        import sentence_transformers  # noqa: F401
        import umap  # noqa: F401
    except ImportError as e:
        aprint(f"Missing dependency: {e}")
        aprint("")
        aprint("Install required packages:")
        aprint("  pip install sentence-transformers umap-learn")
        aprint("")
        sys.exit(1)

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "arxiv_papers.luxar.zarr"
        try:
            n_papers = generate_paper_landscape(
                output_path,
                fields=field_list,
                papers_per_field=papers_per_field,
            )
            if n_papers == 0:
                aprint("\nNo papers generated")
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            aprint("\nPossible issues:")
            aprint("  - Network connection failed")
            aprint("  - API rate limit exceeded")
            aprint("  - Missing dependencies (sentence-transformers, umap-learn)")
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total papers: {n_papers:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_arxiv_") as tmpdir:
        output_path = Path(tmpdir) / "arxiv_papers.luxar.zarr"

        try:
            n_papers = generate_paper_landscape(
                output_path,
                fields=field_list,
                papers_per_field=papers_per_field,
            )

            if n_papers == 0:
                aprint("\nNo papers generated")
                return

        except Exception as e:
            aprint(f"\nError: {e}")
            aprint("\nPossible issues:")
            aprint("  - Network connection failed")
            aprint("  - API rate limit exceeded")
            aprint("  - Missing dependencies (sentence-transformers, umap-learn)")
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS - EXPLORE THE KNOWLEDGE LANDSCAPE")
        aprint("=" * 70)
        aprint("")
        aprint("What to look for:")
        aprint("  - Clusters of related papers (same topic)")
        aprint("  - Boundaries between fields (interdisciplinary zones)")
        aprint("  - Large points = highly cited landmark papers")
        aprint("  - Colors show different research disciplines")
        aprint("")
        aprint("Try this:")
        aprint("  1. Zoom out: See the overall structure of knowledge")
        aprint("  2. Zoom in: Explore specific research clusters")
        aprint("  3. Rotate: See how fields relate in 3D space")
        aprint("")
        aprint("Field colors:")
        for field, color in FIELD_COLORS.items():
            if field != "Other":
                rgb = (color * 255).astype(int)
                aprint(f"  {field}: RGB({rgb[0]}, {rgb[1]}, {rgb[2]})")
        aprint("")
        aprint(f"Total papers: {n_papers:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Explore the landscape of human knowledge!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")
    aprint("")
    aprint("Performance tips:")
    aprint("  - Embeddings + UMAP are cached automatically (instant re-runs)")
    aprint("  - Reduce --papers=N for faster generation")
    aprint("  - Limit --fields=cs,physics for focused exploration")
    aprint("")


if __name__ == "__main__":
    main()
