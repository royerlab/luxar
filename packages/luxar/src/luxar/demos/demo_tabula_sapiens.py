"""
Tabula Sapiens — Human Single-Cell Atlas UMAP
==============================================

Visualizes the Tabula Sapiens single-cell transcriptomics atlas as a 3D UMAP
embedding. Each point is a cell, colored by organ of origin, with hover labels
showing cell type and tissue.

The Tabula Sapiens is a benchmark, first-draft human cell atlas of nearly
500,000 cells from 24 tissues of 15 human donors (Science, 2022).

Data source: CZ CELLxGENE Discover (open access, CC BY 4.0)
Paper: https://doi.org/10.1126/science.abl4896

Usage:
    python -m luxar.demos.demo_tabula_sapiens
    python -m luxar.demos.demo_tabula_sapiens --no-serve
    python -m luxar.demos.demo_tabula_sapiens --sample=50000

Dependencies:
    pip install luxar[demos] anndata cellxgene-census
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

DEFAULT_SAMPLE_SIZE = 100000  # 100K cells (from ~500K total)

# Organ/tissue colors — distinct hues for each organ
ORGAN_COLORS: dict[str, tuple[float, float, float]] = {
    "blood": (0.9, 0.2, 0.2),
    "bone marrow": (0.8, 0.3, 0.3),
    "lung": (0.3, 0.7, 0.9),
    "heart": (0.9, 0.3, 0.4),
    "liver": (0.6, 0.3, 0.1),
    "kidney": (0.7, 0.4, 0.6),
    "pancreas": (0.5, 0.7, 0.3),
    "skin": (0.9, 0.7, 0.5),
    "eye": (0.4, 0.6, 0.9),
    "bladder": (0.6, 0.5, 0.8),
    "fat": (0.9, 0.8, 0.4),
    "large intestine": (0.4, 0.5, 0.3),
    "small intestine": (0.5, 0.6, 0.3),
    "lymph node": (0.7, 0.8, 0.3),
    "mammary": (0.8, 0.5, 0.6),
    "muscle": (0.6, 0.2, 0.2),
    "prostate": (0.5, 0.4, 0.6),
    "salivary gland": (0.7, 0.6, 0.5),
    "spleen": (0.8, 0.4, 0.5),
    "thymus": (0.4, 0.7, 0.6),
    "tongue": (0.7, 0.5, 0.4),
    "trachea": (0.5, 0.7, 0.8),
    "uterus": (0.8, 0.5, 0.7),
    "vasculature": (0.6, 0.3, 0.5),
}

DEFAULT_COLOR = (0.5, 0.5, 0.5)


# =============================================================================
# Data Loading
# =============================================================================


def load_tabula_sapiens(
    cache_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load Tabula Sapiens via cellxgene-census or cached h5ad.

    Returns:
        Tuple of (umap_2d, cell_types, tissues)
    """
    processed_cache = cache_dir / f"tabula_sapiens_{sample_size}.npz"

    if processed_cache.exists():
        with asection("Loading cached Tabula Sapiens data"):
            cached = np.load(processed_cache, allow_pickle=True)
            umap_coords = cached["umap_coords"]
            cell_types = list(cached["cell_types"])
            tissues = list(cached["tissues"])
            aprint(f"✓ Loaded {len(cell_types):,} cells from cache")
            return umap_coords, cell_types, tissues

    # Try cellxgene-census first (official API)
    try:
        return _load_via_census(cache_dir, sample_size, processed_cache)
    except ImportError:
        pass

    # Fallback: try loading from local h5ad
    try:
        return _load_via_h5ad(cache_dir, sample_size, processed_cache)
    except (ImportError, FileNotFoundError):
        pass

    aprint("❌ Could not load Tabula Sapiens data.")
    aprint("")
    aprint("Install one of these options:")
    aprint("  Option 1 (recommended): pip install cellxgene-census")
    aprint("  Option 2: pip install anndata")
    aprint("    Then download h5ad from: https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5")
    aprint(f"    Place it at: {cache_dir / 'tabula_sapiens.h5ad'}")
    sys.exit(1)


def _load_via_census(
    cache_dir: Path,
    sample_size: int,
    processed_cache: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load via CZ CELLxGENE Census API (recommended)."""
    import cellxgene_census

    with asection("Loading Tabula Sapiens via CELLxGENE Census"):
        aprint("Opening census (this may take a moment on first run)...")

        with cellxgene_census.open_soma() as census:
            # Query Tabula Sapiens collection
            aprint("Querying Tabula Sapiens cells...")

            # Get cell metadata
            obs_df = cellxgene_census.get_obs(
                census,
                "Homo sapiens",
                value_filter="dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'",
                column_names=[
                    "cell_type",
                    "tissue_general",
                ],
            )

            aprint(f"✓ Found {len(obs_df):,} cells")

            # Subsample
            if sample_size < len(obs_df):
                obs_df = obs_df.sample(n=sample_size, random_state=42)
                aprint(f"  Subsampled to {len(obs_df):,} cells")

            cell_types = obs_df["cell_type"].astype(str).tolist()
            tissues = obs_df["tissue_general"].astype(str).tolist()

            # Get UMAP embedding for these cells
            aprint("Fetching UMAP embeddings...")
            embedding = cellxgene_census.get_embedding(
                "CxG-contrib-1",
                census["census_data"]["homo_sapiens"],
                obs_soma_joinids=obs_df["soma_joinid"].tolist(),
                obsm_layer="emb",
            )
            # Use first 2 components as UMAP
            umap_coords = embedding[:, :2].astype(np.float32)

    aprint(f"  Unique cell types: {len(set(cell_types))}")
    aprint(f"  Unique tissues: {len(set(tissues))}")

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        processed_cache,
        umap_coords=umap_coords,
        cell_types=np.array(cell_types, dtype=object),
        tissues=np.array(tissues, dtype=object),
    )
    aprint(f"✓ Cached to {processed_cache}")

    return umap_coords, cell_types, tissues


def _load_via_h5ad(
    cache_dir: Path,
    sample_size: int,
    processed_cache: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load from a local h5ad file."""
    import anndata as ad

    h5ad_path = cache_dir / "tabula_sapiens.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found at {h5ad_path}")

    with asection(f"Loading h5ad from {h5ad_path}"):
        adata = ad.read_h5ad(h5ad_path, backed="r")
        aprint(f"✓ {adata.n_obs:,} cells, {adata.n_vars:,} genes")

        # Subsample
        if sample_size < adata.n_obs:
            indices = np.random.default_rng(42).choice(adata.n_obs, sample_size, replace=False)
            indices.sort()
        else:
            indices = np.arange(adata.n_obs)

        # Extract UMAP
        umap_key = "X_umap"
        if umap_key not in adata.obsm:
            available = list(adata.obsm.keys())
            aprint(f"⚠️  '{umap_key}' not found. Available: {available}")
            umap_key = available[0] if available else None
            if umap_key is None:
                raise ValueError("No embedding found in obsm")

        umap_coords = np.array(adata.obsm[umap_key][indices]).astype(np.float32)

        # Extract metadata
        # Try common column names
        cell_type_col = None
        for col in ["cell_ontology_class", "cell_type", "celltype", "cell_type_ontology_term_id"]:
            if col in adata.obs.columns:
                cell_type_col = col
                break
        if cell_type_col is None:
            cell_type_col = adata.obs.columns[0]

        tissue_col = None
        for col in ["organ_tissue", "tissue", "tissue_general", "organ"]:
            if col in adata.obs.columns:
                tissue_col = col
                break
        if tissue_col is None:
            tissue_col = adata.obs.columns[1] if len(adata.obs.columns) > 1 else cell_type_col

        cell_types = adata.obs[cell_type_col].iloc[indices].astype(str).tolist()
        tissues = adata.obs[tissue_col].iloc[indices].astype(str).tolist()

        aprint(f"  Cell type column: {cell_type_col} ({len(set(cell_types))} unique)")
        aprint(f"  Tissue column: {tissue_col} ({len(set(tissues))} unique)")

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        processed_cache,
        umap_coords=umap_coords,
        cell_types=np.array(cell_types, dtype=object),
        tissues=np.array(tissues, dtype=object),
    )
    aprint(f"✓ Cached to {processed_cache}")

    return umap_coords, cell_types, tissues


# =============================================================================
# Scene Generation
# =============================================================================


def generate_tabula_sapiens(
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    cache_dir: Path | None = None,
) -> int:
    """Generate 3D UMAP landscape of Tabula Sapiens."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "luxar" / "tabula_sapiens"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data (2D UMAP + metadata)
    umap_2d, cell_types, tissues = load_tabula_sapiens(cache_dir, sample_size)
    n_cells = len(cell_types)

    if n_cells == 0:
        aprint("❌ No cells loaded")
        return 0

    # Lift 2D UMAP to 3D (add a computed z dimension or use UMAP 3D)
    umap_3d_cache = cache_dir / f"umap3d_{n_cells}.npz"
    if umap_3d_cache.exists():
        positions = np.load(umap_3d_cache)["positions"]
        aprint(f"✓ Loaded 3D UMAP from cache ({len(positions):,} cells)")
    else:
        with asection("Computing 3D embedding from 2D UMAP"):
            # Use 2D UMAP as x,y; compute z from local density
            from scipy.spatial import cKDTree

            tree = cKDTree(umap_2d)
            # z = log(local density) — gives depth to clusters
            k = min(30, n_cells - 1)
            dists, _ = tree.query(umap_2d, k=k + 1)
            avg_dist = dists[:, 1:].mean(axis=1)  # skip self
            # Invert: dense regions → low z, sparse → high z
            z = np.log1p(avg_dist)
            z = (z - z.mean()) * 2.0  # center and scale

            positions = np.column_stack([umap_2d, z.astype(np.float32)])
            positions -= positions.mean(axis=0)
            aprint(f"✓ 3D embedding: {positions.shape}")

        np.savez(umap_3d_cache, positions=positions)

    # Generate visualization
    with asection("Generating visualization"):
        # Colors by tissue
        colors = np.zeros((n_cells, 3), dtype=np.float32)
        for i, tissue in enumerate(tissues):
            t_lower = tissue.lower()
            matched = False
            for key, color in ORGAN_COLORS.items():
                if key in t_lower or t_lower in key:
                    colors[i] = color
                    matched = True
                    break
            if not matched:
                colors[i] = DEFAULT_COLOR

        # Count tissues
        tissue_counts: dict[str, int] = {}
        for t in tissues:
            tissue_counts[t] = tissue_counts.get(t, 0) + 1
        aprint("✓ Cells by tissue:")
        for t, count in sorted(tissue_counts.items(), key=lambda x: -x[1])[:12]:
            aprint(f"  {t}: {count:,}")

        # Uniform radii and sharpness
        radii = np.full(n_cells, 0.015, dtype=np.float32)

        # Hover labels: cell type (tissue)
        labels = [
            f"{cell_types[i]} ({tissues[i]})" for i in range(n_cells)
        ]

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="density", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            sharpness = np.full(n_cells, 4.0, dtype=np.float32)

            scene.add_points(
                "cells",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.85,
                intensity=0.15,
                labels=labels,
            )

            # Title overlay
            scene.add_text(
                "Tabula Sapiens — Human Cell Atlas",
                position=(0.5, 0.02),
                font_size=0.026,
                anchor="top-center",
                color="rgba(255,255,255,0.85)",
                stroke_color="black",
                stroke_width=0.002,
            )

            # Info overlay
            n_types = len(set(cell_types))
            n_tissues = len(set(tissues))
            scene.add_text(
                f"{n_cells:,} cells • {n_types} cell types • {n_tissues} tissues • color = organ",
                position=(0.5, 0.97),
                font_size=0.014,
                anchor="bottom-center",
                color="rgba(200,200,200,0.6)",
            )

    aprint(f"✓ Wrote {n_cells:,} cells to {output_path}")
    return n_cells


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("TABULA SAPIENS — Human Single-Cell Atlas")
    aprint("=" * 70)
    aprint("")
    aprint("~500K cells from 24 tissues of 15 human donors.")
    aprint("Hover to see cell type and tissue of origin.")
    aprint("")

    # Parse sample size
    sample_size = DEFAULT_SAMPLE_SIZE
    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
            aprint(f"Sample size: {sample_size:,}")

    cache_dir = Path.home() / ".cache" / "luxar" / "tabula_sapiens"

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "tabula_sapiens.zarr"
        try:
            n_cells = generate_tabula_sapiens(
                output_path, sample_size=sample_size, cache_dir=cache_dir
            )
            if n_cells == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total cells: {n_cells:,}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_tabula_sapiens_") as tmpdir:
        output_path = Path(tmpdir) / "tabula_sapiens.zarr"

        try:
            n_cells = generate_tabula_sapiens(
                output_path, sample_size=sample_size, cache_dir=cache_dir
            )
            if n_cells == 0:
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
        aprint("Explore the human cell atlas:")
        aprint("  - Clusters = cell types with similar transcriptomic profiles")
        aprint("  - Colors = organ/tissue of origin")
        aprint("  - Hover over any cell to see its type and tissue")
        aprint("")
        aprint(f"Total cells: {n_cells:,}")
        aprint("")
        aprint("Press Ctrl+C when done.")

        launch_viewer(output_path)

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
