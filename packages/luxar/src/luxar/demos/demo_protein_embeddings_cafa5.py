#!/usr/bin/env python3
"""Self-Contained Demo: Protein Function Landscape (ProtT5 Embeddings)

Visualize 142k proteins in 3D embedding space, showing how proteins with
similar functions cluster together.

================================================================================
PROTEIN FUNCTION PREDICTION & EMBEDDINGS
================================================================================

CAFA (Critical Assessment of Functional Annotation) is a community challenge
for predicting protein function from sequence. The CAFA5 dataset contains
142,246 proteins with known functional annotations.

WHAT ARE PROTEIN EMBEDDINGS?
-----------------------------
Protein embeddings are learned vector representations that capture a protein's
sequence, structure, and function in a high-dimensional space. Similar to how
word embeddings (Word2Vec) place semantically similar words close together,
protein embeddings place functionally similar proteins nearby.

PROTT5 MODEL:
-------------
ProtT5 is a protein language model based on T5 (Text-To-Text Transfer Transformer)
architecture, trained on billions of protein sequences. It learns to:
- Understand amino acid patterns
- Recognize functional motifs
- Capture evolutionary relationships
- Predict structural features

The model generates 1,024-dimensional embeddings that encode a protein's
characteristics.

GENE ONTOLOGY (GO) ANNOTATIONS:
--------------------------------
Proteins are annotated with GO terms describing:
- **Molecular Function**: What the protein does (e.g., "kinase activity")
- **Biological Process**: What pathway it's involved in (e.g., "cell division")
- **Cellular Component**: Where it's located (e.g., "mitochondrion")

VISUALIZATION STRATEGY:
-----------------------
This demo:
1. Loads 142k pre-computed ProtT5 embeddings (1,024D)
2. Reduces to 3D using UMAP (preserves functional relationships)
3. Colors proteins by function category
4. Shows beautiful clusters of proteins with similar roles!

WHAT YOU'LL SEE:
- Enzymes (kinases, proteases, etc.) clustering together
- Structural proteins in distinct regions
- Membrane proteins separate from cytoplasmic ones
- DNA-binding proteins grouped by function
- Beautiful functional landscape of the proteome!

DATASET: CAFA5 (Kaggle)
-----------------------
Source: https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5
- 142,246 proteins with ProtT5-XL embeddings
- 1,024 dimensions per protein
- 540 MB download (compressed)
- NumPy format (easy loading!)

Usage:
    python demo_protein_embeddings_cafa5.py [--sample=N]

    Options:
    --sample=N       Number of proteins to visualize (default: all 142k)
    --no-serve       Generate dataset without launching viewer

Requirements:
    - pip install umap-learn pandas

Controls:
    - Explore clusters of functionally similar proteins
    - Color = protein function category
    - Size = sequence length or confidence
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

DEFAULT_SAMPLE_SIZE = None  # Use all 142k proteins by default

# Protein function category colors
# Based on broad functional categories OR k-means clusters
FUNCTION_COLORS = {
    # GO-based categories
    "enzyme": np.array([1.0, 0.5, 0.2]),  # Orange
    "transporter": np.array([0.3, 0.7, 1.0]),  # Blue
    "receptor": np.array([0.9, 0.3, 0.9]),  # Magenta
    "structural": np.array([0.5, 0.9, 0.5]),  # Green
    "regulator": np.array([1.0, 0.8, 0.2]),  # Gold
    "binding": np.array([0.6, 0.3, 1.0]),  # Purple
    "signaling": np.array([1.0, 0.4, 0.4]),  # Red
    "membrane": np.array([0.3, 0.9, 0.9]),  # Cyan
    "nucleic_acid": np.array([0.9, 0.6, 0.3]),  # Tan
    "catalytic": np.array([1.0, 0.7, 0.3]),  # Light Orange
    # K-means clusters (rainbow)
    "cluster_0": np.array([1.0, 0.3, 0.3]),  # Red
    "cluster_1": np.array([1.0, 0.6, 0.2]),  # Orange
    "cluster_2": np.array([1.0, 0.9, 0.2]),  # Yellow
    "cluster_3": np.array([0.5, 1.0, 0.3]),  # Lime
    "cluster_4": np.array([0.2, 1.0, 0.5]),  # Green
    "cluster_5": np.array([0.2, 0.9, 0.9]),  # Cyan
    "cluster_6": np.array([0.3, 0.5, 1.0]),  # Blue
    "cluster_7": np.array([0.6, 0.3, 1.0]),  # Purple
    "cluster_8": np.array([0.9, 0.3, 0.8]),  # Magenta
    "cluster_9": np.array([1.0, 0.4, 0.6]),  # Pink
    # Fallback
    "other": np.array([0.5, 0.5, 0.5]),  # Gray
}


# =============================================================================
# Dataset Download
# =============================================================================


def download_cafa5_dataset(output_dir: Path) -> Path:
    """Download CAFA5 ProtT5 embeddings from Kaggle with robust retry and resume.

    Uses the robust_download utility which provides:
    - Automatic retry on network errors (up to 3 attempts)
    - Resume capability for partial downloads
    - Progress tracking with ETA
    - File size verification

    Args:
        output_dir: Where to save the dataset

    Returns:
        Path to downloaded/extracted dataset directory
    """
    import zipfile

    from luxar.utils.download import robust_download

    dataset_zip = output_dir / "cafa5_prott5.zip"

    with asection("Downloading CAFA5 ProtT5 Embeddings"):
        aprint("Dataset: CAFA5 Protein Function Prediction")
        aprint(
            "URL: https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5"
        )
        aprint("")
        aprint("Download: 540 MB (pre-computed embeddings)")
        aprint("Features: Auto-retry on errors, resume on interruption")
        aprint("")

        if dataset_zip.exists():
            aprint(f"✓ Dataset already downloaded: {dataset_zip}")
        else:
            url = "https://www.kaggle.com/api/v1/datasets/download/horikitasaku/prott5-embedding-for-cafa5"

            try:
                dataset_zip = robust_download(
                    url=url,
                    output_path=dataset_zip,
                    max_retries=3,  # Retry up to 3 times
                    timeout=300,  # 5 minute initial connection timeout
                    chunk_size=1024 * 1024,  # 1MB chunks
                    verify_size=True,  # Verify final size
                )

                aprint("✓ Download complete and verified!")

            except Exception as e:
                aprint(f"❌ Download failed after all retries: {e}")
                if dataset_zip.exists():
                    partial_size = dataset_zip.stat().st_size
                    aprint(
                        f"   Partial download saved: {partial_size / (1024**2):.0f} MB"
                    )
                    aprint("   Run again to resume from this point")
                raise

        # Extract if needed
        extracted_dir = output_dir / "cafa5_data"
        if not extracted_dir.exists():
            aprint("Extracting...")
            with zipfile.ZipFile(dataset_zip, "r") as zf:
                zf.extractall(extracted_dir)
            aprint(f"✓ Extracted to {extracted_dir}")

    return extracted_dir


# =============================================================================
# Data Loading
# =============================================================================


def load_go_annotations(data_dir: Path) -> dict:
    """Load GO term annotations from TSV file.

    Args:
        data_dir: CAFA5 data directory

    Returns:
        Dictionary mapping protein_id -> list of GO terms
    """
    import pandas as pd

    tsv_files = list(data_dir.rglob("*terms.tsv"))

    if not tsv_files:
        aprint("⚠️  No GO annotation files found")
        return {}

    with asection("Loading GO term annotations"):
        go_file = tsv_files[0]
        aprint(f"File: {go_file.name}")

        df = pd.read_csv(go_file, sep="\t")
        aprint(f"✓ Loaded {len(df):,} GO annotations")
        aprint(f"  Columns: {list(df.columns)}")

        # Group by protein ID
        protein_to_go = {}
        for _, row in df.iterrows():
            protein_id = row["EntryID"]
            go_term = row["term"]

            if protein_id not in protein_to_go:
                protein_to_go[protein_id] = []
            protein_to_go[protein_id].append(go_term)

        aprint(f"✓ Annotations for {len(protein_to_go):,} unique proteins")

    return protein_to_go


def load_protein_embeddings(
    data_dir: Path,
    sample_size: int | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load ProtT5 embeddings and metadata from CAFA5 dataset.

    Args:
        data_dir: Directory containing extracted CAFA5 data
        sample_size: Optional number of proteins to sample

    Returns:
        Tuple of (embeddings, protein_ids, functions)
    """
    with asection("Loading CAFA5 protein embeddings"):
        # Find the embedding files
        npy_files = list(data_dir.rglob("*.npy"))

        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

        # Find the LARGEST train_embeddings.npy (main dataset, not subsets)
        embedding_candidates = [
            f for f in npy_files if "embeddings" in f.name and "train" in f.name
        ]
        ids_candidates = [f for f in npy_files if "ids" in f.name and "train" in f.name]

        if not embedding_candidates:
            raise FileNotFoundError("Could not find train_embeddings.npy")

        # Use the largest embedding file (full dataset)
        embedding_file = max(embedding_candidates, key=lambda f: f.stat().st_size)
        ids_file = (
            max(ids_candidates, key=lambda f: f.stat().st_size)
            if ids_candidates
            else None
        )

        aprint(
            f"Selected embeddings file: {embedding_file.name} ({embedding_file.stat().st_size / (1024**2):.1f} MB)"
        )

        aprint(f"Loading embeddings: {embedding_file.name}")
        embeddings = np.load(embedding_file)
        aprint(f"✓ Loaded {len(embeddings):,} embeddings (shape: {embeddings.shape})")

        # Load protein IDs
        if ids_file:
            aprint(f"Loading protein IDs: {ids_file.name}")
            protein_ids = np.load(ids_file).tolist()
            aprint(f"✓ Loaded {len(protein_ids):,} protein IDs")
        else:
            protein_ids = [f"Protein_{i}" for i in range(len(embeddings))]

        # Load GO annotations
        protein_to_go = load_go_annotations(data_dir)

        # Try GO-based classification
        aprint("Attempting GO-based classification...")
        functions = []
        matched_go = 0
        for pid in protein_ids:
            if pid in protein_to_go:
                go_terms = protein_to_go[pid]
                func = classify_go_term(go_terms[0]) if go_terms else "other"
                matched_go += 1
            else:
                func = "other"
            functions.append(func)

        aprint(
            f"  GO matches: {matched_go}/{len(protein_ids)} ({matched_go / len(protein_ids) * 100:.1f}%)"
        )

        # If no GO matches, use k-means clustering for coloring!
        if matched_go < len(protein_ids) * 0.01:  # Less than 1% annotated
            aprint("⚠️  Very few GO annotations, using k-means clustering instead!")
            from sklearn.cluster import KMeans

            with asection("Clustering proteins by embedding similarity"):
                n_clusters = 10  # 10 functional groups
                aprint(f"Running k-means with {n_clusters} clusters...")

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)

                # Map cluster labels to function names
                cluster_names = [f"cluster_{i}" for i in range(n_clusters)]
                functions = [cluster_names[label] for label in cluster_labels]

                aprint(f"✓ Created {n_clusters} functional clusters")
                for i in range(min(5, n_clusters)):
                    count = sum(1 for label in cluster_labels if label == i)
                    aprint(f"  Cluster {i}: {count:,} proteins")
        else:
            func_counts = {}
            for f in functions:
                func_counts[f] = func_counts.get(f, 0) + 1
            aprint(f"✓ Function distribution: {len(func_counts)} categories")
            for func, count in sorted(func_counts.items(), key=lambda x: -x[1])[:5]:
                aprint(f"  {func}: {count:,}")

        # Sample if requested
        if sample_size and sample_size < len(embeddings):
            aprint(f"Sampling {sample_size:,} proteins...")
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            protein_ids = [protein_ids[i] for i in indices]
            functions = [functions[i] for i in indices]

        aprint(f"✓ Final dataset: {len(embeddings):,} proteins")
        aprint(f"  Function categories: {len(set(functions))} unique")

    return embeddings, protein_ids, functions


def classify_go_term(go_id: str) -> str:
    """Classify GO term ID into broad functional category.

    Uses GO term number ranges to classify:
    - GO:0003xxx: Molecular Function
    - GO:0008xxx: Biological Process
    - GO:0005xxx: Cellular Component

    Args:
        go_id: GO term ID (e.g., "GO:0003700")

    Returns:
        Broad category name
    """
    try:
        # Extract numeric part
        go_num = int(go_id.split(":")[1])

        # Molecular Functions (catalytic activities, binding)
        if 3000 <= go_num < 6000:
            if 3700 <= go_num < 3800:  # Transcription factors
                return "regulator"
            elif 4000 <= go_num < 5000:  # Enzyme activities
                return "enzyme"
            elif 5000 <= go_num < 6000:  # Binding
                return "binding"
            else:
                return "catalytic"

        # Biological Processes
        elif 6000 <= go_num < 9000 or 40000 <= go_num < 100000:
            if 6350 <= go_num < 6400:  # DNA/RNA processes
                return "nucleic_acid"
            elif 6800 <= go_num < 7000:  # Signal transduction
                return "signaling"
            elif 6900 <= go_num < 7000:  # Transport
                return "transporter"
            else:
                return "binding"  # General biological process

        # Cellular Components (location-based)
        elif 5000 <= go_num < 6000 or 9000 <= go_num < 10000:
            if 5886 == go_num:  # Membrane
                return "membrane"
            elif 5840 <= go_num < 5850:  # Ribosome
                return "structural"
            else:
                return "membrane"

        else:
            return "other"

    except (ValueError, IndexError):
        return "other"


# =============================================================================
# UMAP Dimensionality Reduction
# =============================================================================


def reduce_embeddings_umap(
    embeddings: np.ndarray,
    functions: list[str],
    cache_path: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Reduce protein embeddings to 3D using UMAP.

    Args:
        embeddings: (n_proteins, n_features) array
        functions: List of function categories for each protein
        cache_path: Optional path to cache UMAP results

    Returns:
        Tuple of (positions, functions) - both cached together!
    """
    from umap import UMAP

    # Check cache first (loads positions AND functions together!)
    if cache_path and cache_path.exists():
        with asection("Loading cached UMAP coordinates"):
            aprint(f"Cache: {cache_path}")
            cached = np.load(cache_path, allow_pickle=True)
            positions = cached["positions"]
            functions = list(cached["functions"])
            aprint(f"✓ Loaded {len(positions):,} proteins from cache (INSTANT!)")
            return positions, functions

    if embeddings is None:
        raise ValueError("Embeddings required when not loading from cache")

    with asection(f"Reducing {embeddings.shape[1]}D → 3D with UMAP"):
        aprint(
            f"Input: {embeddings.shape[0]:,} proteins × {embeddings.shape[1]} dimensions"
        )
        aprint("Parameters: n_neighbors=15, metric=cosine, all CPU cores")

        reducer = UMAP(
            n_components=3,
            n_neighbors=15,
            metric="cosine",
            n_jobs=-1,  # All cores
            low_memory=False,  # Speed optimization
            verbose=True,
        )

        reduced = reducer.fit_transform(embeddings)

        aprint("✓ UMAP complete")
        aprint(f"  Output: {reduced.shape}")
        aprint(f"  Range: [{reduced.min():.2f}, {reduced.max():.2f}]")

        # Center at barycenter
        centroid = reduced.mean(axis=0)
        reduced = reduced - centroid
        aprint("✓ Centered at barycenter")
        aprint(f"  New range: [{reduced.min():.2f}, {reduced.max():.2f}]")

        # Cache results (positions AND functions together!)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                positions=reduced,
                functions=np.array(functions, dtype=object),  # Store functions too!
            )
            aprint(f"✓ Cached UMAP + functions to {cache_path}")
            aprint("  Future runs will load instantly!")

    return reduced.astype(np.float32), functions


# =============================================================================
# Visualization Generation
# =============================================================================


def generate_protein_landscape(
    output_path: Path,
    sample_size: int | None = None,
) -> int:
    """Generate 3D protein function landscape.

    Args:
        output_path: Where to write zarr
        sample_size: Optional number of proteins to sample

    Returns:
        Number of proteins visualized
    """
    # Setup caching
    cache_dir = Path.home() / ".cache" / "luxar" / "protein_embeddings"
    dataset_cache = cache_dir / "cafa5_data"
    umap_cache = cache_dir / f"umap_{sample_size or 'all'}.npz"

    # Download dataset if needed
    if not dataset_cache.exists():
        dataset_cache = download_cafa5_dataset(cache_dir)

    # Check UMAP cache first (complete early exit if cached!)
    if umap_cache.exists():
        positions, functions = reduce_embeddings_umap(None, None, cache_path=umap_cache)
    else:
        # Load embeddings (only if UMAP not cached)
        embeddings, protein_ids, functions = load_protein_embeddings(
            dataset_cache,
            sample_size=sample_size,
        )

        if len(embeddings) == 0:
            aprint("❌ No proteins loaded")
            return 0

        # Reduce to 3D with UMAP and cache
        positions, functions = reduce_embeddings_umap(
            embeddings, functions, cache_path=umap_cache
        )

    # Generate visualization
    with asection("Generating visualization"):
        n_proteins = len(positions)

        # Colors by function
        colors = np.zeros((n_proteins, 3), dtype=np.float32)
        for i, func in enumerate(functions):
            colors[i] = FUNCTION_COLORS.get(func, FUNCTION_COLORS["other"])

        # Count functions
        func_counts = {}
        for func in functions:
            func_counts[func] = func_counts.get(func, 0) + 1

        aprint("✓ Proteins by function:")
        for func, count in sorted(func_counts.items(), key=lambda x: -x[1])[:10]:
            aprint(f"  {func}: {count:,}")

        # Size: annotated proteins are 3x larger to stand out!
        radii = np.zeros(n_proteins, dtype=np.float32)
        for i, func in enumerate(functions):
            if func == "other":
                radii[i] = 0.007  # Small gray background points
            else:
                radii[i] = 0.02  # 3x larger for annotated proteins!

        n_annotated = sum(1 for f in functions if f != "other")
        aprint(
            f"✓ Radii: {n_annotated:,} annotated (large), {n_proteins - n_annotated:,} unannotated (small)"
        )

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

            sharpness = np.full(n_proteins, 0.55, dtype=np.float32)

            # Hover labels: function category for each protein
            protein_labels = [f"{functions[i]}" for i in range(n_proteins)]

            scene.add_points(
                "proteins",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.9,
                intensity=0.124,
                labels=protein_labels,
            )

            # --- Overlays ---
            scene.add_text(
                "Protein Function Landscape (CAFA5)",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Function color legend
            legend_items = [
                ("Enzyme", "#ff8033"),
                ("Transporter", "#4db3ff"),
                ("Receptor", "#e64de6"),
                ("Structural", "#80e680"),
                ("Regulator", "#ffcc33"),
                ("Binding", "#994dff"),
                ("Signaling", "#ff6666"),
                ("Membrane", "#4de6e6"),
            ]
            legend_html = (
                '<div style="font-size:1.3vh;line-height:1.7;background:rgba(0,0,0,0.5);padding:0.5vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">Function</div>'
            )
            for name, color in legend_items:
                legend_html += (
                    f'<div><span style="color:{color}">\u2588</span> {name}</div>'
                )
            legend_html += "</div>"

            scene.add_html(
                legend_html,
                position=(0.02, 0.97),
                anchor="bottom-left",
            )

            scene.add_text(
                f"{n_proteins:,} proteins • ProtT5 embeddings • 3D UMAP • Zhou et al. 2024",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"✓ Visualization created with {n_proteins:,} proteins")

    return n_proteins


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    sample_size = DEFAULT_SAMPLE_SIZE

    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("PROTEIN FUNCTION LANDSCAPE - PROTT5 EMBEDDINGS")
    aprint("=" * 70)
    aprint("")
    aprint("Dataset: CAFA5 Protein Function Prediction")
    aprint("https://www.kaggle.com/datasets/horikitasaku/prott5-embedding-for-cafa5")
    aprint("")
    aprint("What this shows:")
    aprint("  • 142k proteins with ProtT5-XL embeddings (1,024D)")
    aprint("  • 3D UMAP projection showing functional relationships")
    aprint("  • Proteins with similar functions cluster together")
    aprint("  • Color = protein function category")
    aprint("  • Size = sequence complexity")
    aprint("")
    aprint("Parameters:")
    aprint(
        f"  Proteins: {sample_size:,}" if sample_size else "  Proteins: All (142,246)"
    )
    aprint("")

    # Check dependencies
    try:
        import pandas  # noqa: F401
        import umap  # noqa: F401
    except ImportError as e:
        aprint(f"Missing dependency: {e}")
        aprint("")
        aprint("Install with:")
        aprint("  pip install umap-learn pandas")
        sys.exit(1)

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "protein_landscape.zarr"
        try:
            n_proteins = generate_protein_landscape(
                output_path,
                sample_size=sample_size,
            )
            if n_proteins == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total proteins: {n_proteins:,}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_proteins_") as tmpdir:
        output_path = Path(tmpdir) / "protein_landscape.zarr"

        try:
            n_proteins = generate_protein_landscape(
                output_path,
                sample_size=sample_size,
            )

            if n_proteins == 0:
                return

        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS - EXPLORE THE PROTEOME")
        aprint("=" * 70)
        aprint("")
        aprint("What to look for:")
        aprint("  - Enzyme clusters (kinases, proteases, etc.)")
        aprint("  - Structural protein regions")
        aprint("  - Membrane protein groups")
        aprint("  - DNA/RNA binding protein clusters")
        aprint("  - Functional boundaries and overlaps")
        aprint("")
        aprint("Try this:")
        aprint("  1. Zoom out: See overall functional organization")
        aprint("  2. Zoom in: Explore specific protein families")
        aprint("  3. Look for: Tight clusters = very similar function")
        aprint("")
        aprint(f"Total proteins: {n_proteins:,}")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Explore the landscape of protein function!")
        aprint("Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
