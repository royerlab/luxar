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
    pip install luxar[demos]   # includes pandas, umap-learn, scipy
"""

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 500000  # All ~500K cells

# CELLxGENE collection for Tabula Sapiens
COLLECTION_ID = "e5f58829-1a66-40b5-a624-9046778e74f5"
CELLXGENE_API = "https://api.cellxgene.cziscience.com/curation/v1"

# Organ/tissue-specific dataset titles to download (sorted smallest first).
# These are organ-level datasets that together give multi-tissue diversity.
# Cell-type compartment datasets (Immune, Stromal, etc.) are excluded.
ORGAN_DATASETS = {
    "Tabula Sapiens - Ear",
    "Tabula Sapiens - Testis",
    "Tabula Sapiens - Germline",
    "Tabula Sapiens - Kidney",
    "Tabula Sapiens - Skin",
    "Tabula Sapiens - Pancreas",
    "Tabula Sapiens - Liver",
    "Tabula Sapiens - Uterus",
    "Tabula Sapiens - Prostate",
    "Tabula Sapiens - Trachea",
    "Tabula Sapiens - Heart",
    "Tabula Sapiens - Stomach",
    "Tabula Sapiens - Large_Intestine",
    "Tabula Sapiens - Bone_Marrow",
    "Tabula Sapiens - Eye",
    "Tabula Sapiens - Mammary",
    "Tabula Sapiens - Small_Intestine",
    "Tabula Sapiens - Salivary_Gland",
    "Tabula Sapiens - Tongue",
    "Tabula Sapiens - Muscle",
    "Tabula Sapiens - Ovary",
    "Tabula Sapiens - Blood",
    "Tabula Sapiens - Bladder",
    "Tabula Sapiens - Lung",
    "Tabula Sapiens - Spleen",
    "Tabula Sapiens - Thymus",
    "Tabula Sapiens - Lymph_Node",
    "Tabula Sapiens - Fat",
    "Tabula Sapiens - Vasculature",
    "Tabula Sapiens - Endothelium",
    "Tabula Sapiens - Neural",
}

# Organ/tissue colors — distinct hues for each organ.
# Keys are normalized: lowercase, underscores replaced with spaces.
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
    "ear": (0.6, 0.7, 0.4),
    "testis": (0.4, 0.4, 0.7),
    "ovary": (0.9, 0.5, 0.8),
    "stomach": (0.7, 0.6, 0.3),
    "endothelium": (0.5, 0.3, 0.6),
    "neural": (0.3, 0.5, 0.8),
    "germline": (0.6, 0.6, 0.3),
}

DEFAULT_COLOR = (0.5, 0.5, 0.5)


def _tissue_color(tissue: str) -> tuple[float, float, float]:
    """Map a tissue name to its color, normalizing for underscores/case."""
    normalized = tissue.lower().replace("_", " ")
    # Exact match first
    if normalized in ORGAN_COLORS:
        return ORGAN_COLORS[normalized]
    # Substring match (longer keys first to avoid "ear" matching before "heart")
    for key in sorted(ORGAN_COLORS, key=len, reverse=True):
        if key in normalized:
            return ORGAN_COLORS[key]
    return DEFAULT_COLOR


# =============================================================================
# h5py helpers for reading h5ad files without anndata
# =============================================================================


def _ensure_h5py():
    """Import h5py, auto-installing if necessary."""
    try:
        import h5py

        return h5py
    except ImportError:
        import subprocess

        aprint("Installing h5py (one-time, ~10 seconds)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "h5py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import h5py

        return h5py


def _read_h5ad_column(h5: Any, column: str) -> list[str]:
    """Read a string/categorical column from h5ad obs group using raw h5py.

    h5ad stores categoricals as a group with two sub-datasets:
      /obs/<column>/codes        -> integer codes
      /obs/<column>/categories   -> string category labels

    Plain string columns are stored directly as /obs/<column> (dataset).
    """
    import h5py

    item = h5["obs"][column]

    # Categorical encoding: group with codes + categories sub-datasets
    if isinstance(item, h5py.Group):
        codes = item["codes"][:]
        cats_raw = item["categories"][:]
        categories = [c.decode() if isinstance(c, bytes) else str(c) for c in cats_raw]
        return [categories[c] for c in codes]

    # Direct string/bytes dataset
    values = item[:]
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


# =============================================================================
# Data Loading — three paths, in order of preference
# =============================================================================


def _discover_tissue_assets() -> list[dict]:
    """Query CELLxGENE API for Tabula Sapiens organ-level h5ad download URLs.

    Returns list of dicts sorted by filesize (smallest first), each with:
        title, cell_count, url, filesize
    """
    import requests

    with asection("Querying CELLxGENE API for Tabula Sapiens tissue datasets"):
        resp = requests.get(f"{CELLXGENE_API}/collections/{COLLECTION_ID}", timeout=30)
        resp.raise_for_status()
        collection = resp.json()

        assets = []
        for ds in collection["datasets"]:
            title = ds.get("title", "")
            if title not in ORGAN_DATASETS:
                continue
            for asset in ds.get("assets", []):
                if asset.get("filetype") == "H5AD":
                    assets.append(
                        {
                            "title": title,
                            "cell_count": ds.get("cell_count", 0),
                            "url": asset["url"],
                            "filesize": asset.get("filesize", 0),
                        }
                    )

        # Sort by file size — download smallest tissues first
        assets.sort(key=lambda a: a["filesize"])
        aprint(f"✓ Found {len(assets)} tissue datasets")
        return assets


def load_tabula_sapiens(
    cache_dir: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    """Load Tabula Sapiens data with three-tier fallback.

    Priority:
        1. Processed NPZ cache (instant)
        2. CELLxGENE Census API (fast streaming, needs cellxgene-census)
        3. Auto-download tissue h5ad files from CELLxGENE (needs only requests+h5py)
        4. Local h5ad file via anndata (manual fallback)

    Returns:
        Tuple of (umap_2d, pca_or_scvi, cell_types, tissues)
        pca_or_scvi may be None if only 2D UMAP was available.
    """
    processed_cache = cache_dir / f"tabula_sapiens_{sample_size}.npz"

    # --- Fast path: processed cache ---
    if processed_cache.exists():
        with asection("Loading cached Tabula Sapiens data"):
            cached = np.load(processed_cache, allow_pickle=True)
            umap_coords = cached["umap_coords"]
            pca = cached["pca"] if "pca" in cached else None
            cell_types = list(cached["cell_types"])
            tissues = list(cached["tissues"])
            aprint(f"✓ Loaded {len(cell_types):,} cells from cache")
            if pca is not None:
                aprint(f"  PCA/scVI embeddings: {pca.shape}")
            return umap_coords, pca, cell_types, tissues

    # --- Path 1: CELLxGENE Census API (best: fast streaming) ---
    try:
        return _load_via_census(cache_dir, sample_size, processed_cache)
    except ImportError:
        aprint("cellxgene-census not installed, trying direct download...")
    except Exception as e:
        aprint(f"Census API failed ({e}), trying direct download...")

    # --- Path 2: Auto-download tissue h5ad files (fully automated) ---
    try:
        return _load_via_h5ad_download(cache_dir, sample_size, processed_cache)
    except Exception as e:
        aprint(f"Direct h5ad download failed: {e}")
        import traceback

        traceback.print_exc()

    # --- Path 3: Local h5ad via anndata ---
    try:
        return _load_via_h5ad(cache_dir, sample_size, processed_cache)
    except (ImportError, FileNotFoundError):
        pass

    aprint("❌ Could not load Tabula Sapiens data.")
    aprint("")
    aprint("Install demo dependencies with:")
    aprint("  pip install luxar[demos]")
    sys.exit(1)


def _load_via_census(
    cache_dir: Path,
    sample_size: int,
    processed_cache: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load via CZ CELLxGENE Census API (recommended, fast streaming)."""
    import cellxgene_census

    with asection("Loading Tabula Sapiens via CELLxGENE Census"):
        aprint("Opening census (this may take a moment on first run)...")

        with cellxgene_census.open_soma() as census:
            aprint("Querying Tabula Sapiens cells...")

            obs_df = cellxgene_census.get_obs(
                census,
                "Homo sapiens",
                value_filter="dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'",
                column_names=["cell_type", "tissue_general"],
            )

            aprint(f"✓ Found {len(obs_df):,} cells")

            if sample_size < len(obs_df):
                obs_df = obs_df.sample(n=sample_size, random_state=42)
                aprint(f"  Subsampled to {len(obs_df):,} cells")

            cell_types = obs_df["cell_type"].astype(str).tolist()
            tissues = obs_df["tissue_general"].astype(str).tolist()

            aprint("Fetching UMAP embeddings...")
            embedding = cellxgene_census.get_embedding(
                "CxG-contrib-1",
                census["census_data"]["homo_sapiens"],
                obs_soma_joinids=obs_df["soma_joinid"].tolist(),
                obsm_layer="emb",
            )
            umap_coords = embedding[:, :2].astype(np.float32)

    aprint(f"  Unique cell types: {len(set(cell_types))}")
    aprint(f"  Unique tissues: {len(set(tissues))}")

    _save_processed_cache(processed_cache, cache_dir, umap_coords, cell_types, tissues)
    return umap_coords, None, cell_types, tissues


def _load_via_h5ad_download(
    cache_dir: Path,
    sample_size: int,
    processed_cache: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Fully automated: download tissue h5ad files from CELLxGENE, read with h5py.

    Downloads the smallest organ-level h5ad files from CELLxGENE until enough
    cells are collected. Only the UMAP coordinates and cell metadata are
    extracted; the large expression matrix is never loaded into memory.
    """
    from luxar.utils.download import robust_download

    h5py = _ensure_h5py()

    # Discover available tissue datasets from CELLxGENE
    assets = _discover_tissue_assets()
    if not assets:
        raise RuntimeError("No tissue datasets found via CELLxGENE API")

    h5ad_dir = cache_dir / "h5ad"
    h5ad_dir.mkdir(parents=True, exist_ok=True)

    all_umap: list[np.ndarray] = []
    all_pca: list[np.ndarray] = []
    all_cell_types: list[str] = []
    all_tissues: list[str] = []
    total_cells = 0
    has_pca = True

    # Download smallest tissues first until we have 2x sample_size
    # (extra margin so subsampling gives good tissue diversity)
    target = sample_size * 2

    with asection(f"Downloading tissue h5ad files (target: {target:,} cells)"):
        for asset in assets:
            if total_cells >= target:
                break

            tissue_name = asset["title"].replace("Tabula Sapiens - ", "")
            size_mb = asset["filesize"] / (1024**2)
            h5ad_path = h5ad_dir / f"{tissue_name.lower().replace(' ', '_')}.h5ad"

            with asection(
                f"{tissue_name} ({asset['cell_count']:,} cells, {size_mb:.0f} MB)"
            ):
                # robust_download handles skip-if-complete and resume-if-truncated
                robust_download(
                    asset["url"],
                    h5ad_path,
                    expected_size=asset["filesize"] or None,
                )

                # Read UMAP + metadata with h5py (no anndata needed)
                try:
                    with h5py.File(str(h5ad_path), "r") as f:
                        # Read UMAP coordinates
                        if "obsm" not in f or "X_umap" not in f["obsm"]:
                            aprint(f"  ⚠ No X_umap in {tissue_name}, skipping")
                            continue
                        umap = np.array(f["obsm"]["X_umap"]).astype(np.float32)

                        # Read PCA or scVI embeddings for proper 3D UMAP
                        pca_local = None
                        for emb_key in ["X_scvi", "X_pca"]:
                            if emb_key in f["obsm"]:
                                pca_local = np.array(f["obsm"][emb_key]).astype(
                                    np.float32
                                )
                                break

                        # Read cell type
                        cell_type_col = None
                        for col in [
                            "cell_type",
                            "cell_ontology_class",
                            "celltype",
                        ]:
                            if col in f["obs"]:
                                cell_type_col = col
                                break
                        if cell_type_col is None:
                            aprint(f"  ⚠ No cell_type column in {tissue_name}")
                            continue
                        cell_types = _read_h5ad_column(f, cell_type_col)

                        # Read tissue
                        tissue_col = None
                        for col in [
                            "tissue_in_publication",
                            "tissue_general",
                            "organ_tissue",
                            "tissue",
                            "organ",
                        ]:
                            if col in f["obs"]:
                                tissue_col = col
                                break
                        if tissue_col is None:
                            # Use the dataset title as tissue label
                            tissues_local = [tissue_name] * len(cell_types)
                        else:
                            tissues_local = _read_h5ad_column(f, tissue_col)

                    all_umap.append(umap)
                    if pca_local is not None:
                        all_pca.append(pca_local)
                    else:
                        has_pca = False
                    all_cell_types.extend(cell_types)
                    all_tissues.extend(tissues_local)
                    total_cells += len(cell_types)
                    aprint(
                        f"  ✓ Extracted {len(cell_types):,} cells (total: {total_cells:,})"
                    )

                except Exception as e:
                    aprint(f"  ⚠ Failed to read {tissue_name}: {e}")
                    continue

    if total_cells == 0:
        raise RuntimeError("No cells extracted from any tissue h5ad")

    # Combine across tissues
    umap_coords = np.concatenate(all_umap, axis=0)
    pca_combined = np.concatenate(all_pca, axis=0) if has_pca and all_pca else None
    aprint(f"✓ Combined {total_cells:,} cells from {len(all_umap)} tissues")
    if pca_combined is not None:
        aprint(f"  PCA/scVI embeddings: {pca_combined.shape}")

    # Subsample to requested size
    if sample_size < total_cells:
        rng = np.random.default_rng(42)
        indices = rng.choice(total_cells, sample_size, replace=False)
        indices.sort()
        umap_coords = umap_coords[indices]
        if pca_combined is not None:
            pca_combined = pca_combined[indices]
        all_cell_types = [all_cell_types[i] for i in indices]
        all_tissues = [all_tissues[i] for i in indices]
        aprint(f"  Subsampled to {len(all_cell_types):,} cells")

    aprint(f"  Unique cell types: {len(set(all_cell_types))}")
    aprint(f"  Unique tissues: {len(set(all_tissues))}")

    _save_processed_cache(
        processed_cache,
        cache_dir,
        umap_coords,
        all_cell_types,
        all_tissues,
        pca=pca_combined,
    )
    return umap_coords, pca_combined, all_cell_types, all_tissues


def _load_via_h5ad(
    cache_dir: Path,
    sample_size: int,
    processed_cache: Path,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load from a local h5ad file via anndata."""
    import anndata as ad

    h5ad_path = cache_dir / "tabula_sapiens.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"H5AD file not found at {h5ad_path}")

    with asection(f"Loading h5ad from {h5ad_path}"):
        adata = ad.read_h5ad(h5ad_path, backed="r")
        aprint(f"✓ {adata.n_obs:,} cells, {adata.n_vars:,} genes")

        if sample_size < adata.n_obs:
            indices = np.random.default_rng(42).choice(
                adata.n_obs, sample_size, replace=False
            )
            indices.sort()
        else:
            indices = np.arange(adata.n_obs)

        umap_key = "X_umap"
        if umap_key not in adata.obsm:
            available = list(adata.obsm.keys())
            aprint(f"⚠️  '{umap_key}' not found. Available: {available}")
            umap_key = available[0] if available else None
            if umap_key is None:
                raise ValueError("No embedding found in obsm")

        umap_coords = np.array(adata.obsm[umap_key][indices]).astype(np.float32)

        cell_type_col = None
        for col in [
            "cell_ontology_class",
            "cell_type",
            "celltype",
            "cell_type_ontology_term_id",
        ]:
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
            tissue_col = (
                adata.obs.columns[1] if len(adata.obs.columns) > 1 else cell_type_col
            )

        cell_types = adata.obs[cell_type_col].iloc[indices].astype(str).tolist()
        tissues = adata.obs[tissue_col].iloc[indices].astype(str).tolist()

        aprint(f"  Cell type column: {cell_type_col} ({len(set(cell_types))} unique)")
        aprint(f"  Tissue column: {tissue_col} ({len(set(tissues))} unique)")

    # Extract PCA/scVI if available
    pca = None
    for emb_key in ["X_scvi", "X_pca"]:
        if emb_key in adata.obsm:
            pca = np.array(adata.obsm[emb_key][indices]).astype(np.float32)
            aprint(f"  Extracted {emb_key}: {pca.shape}")
            break

    _save_processed_cache(
        processed_cache, cache_dir, umap_coords, cell_types, tissues, pca=pca
    )
    return umap_coords, pca, cell_types, tissues


def _save_processed_cache(
    processed_cache: Path,
    cache_dir: Path,
    umap_coords: np.ndarray,
    cell_types: list[str],
    tissues: list[str],
    pca: np.ndarray | None = None,
) -> None:
    """Save processed UMAP + PCA + metadata as NPZ for fast reload."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "umap_coords": umap_coords,
        "cell_types": np.array(cell_types, dtype=object),
        "tissues": np.array(tissues, dtype=object),
    }
    if pca is not None:
        save_kwargs["pca"] = pca
    np.savez(processed_cache, **save_kwargs)
    aprint(f"✓ Cached to {processed_cache}")


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

    # Load data (2D UMAP + PCA/scVI embeddings + metadata)
    umap_2d, pca, cell_types, tissues = load_tabula_sapiens(cache_dir, sample_size)
    n_cells = len(cell_types)

    if n_cells == 0:
        aprint("❌ No cells loaded")
        return 0

    # Compute proper 3D UMAP from PCA/scVI embeddings
    umap_3d_cache = cache_dir / f"umap3d_{n_cells}.npz"
    if umap_3d_cache.exists():
        positions = np.load(umap_3d_cache)["positions"]
        aprint(f"✓ Loaded 3D UMAP from cache ({len(positions):,} cells)")
    elif pca is not None:
        from umap import UMAP

        with asection(
            f"Computing 3D UMAP from {pca.shape[1]}D embeddings ({n_cells:,} cells)"
        ):
            aprint("This may take a few minutes for large datasets...")
            reducer = UMAP(
                n_components=3,
                n_neighbors=30,
                min_dist=0.3,
                metric="euclidean",
                n_jobs=-1,
                verbose=True,
            )
            positions = reducer.fit_transform(pca).astype(np.float32)
            positions -= positions.mean(axis=0)
            aprint(f"✓ 3D UMAP complete: {positions.shape}")

        np.savez(umap_3d_cache, positions=positions)
    else:
        # Fallback: flat 2D UMAP with z=0 (no PCA available)
        aprint("⚠ No PCA/scVI embeddings — using flat 2D UMAP (z=0)")
        positions = np.column_stack([umap_2d, np.zeros(n_cells, dtype=np.float32)])
        positions -= positions.mean(axis=0)

    # Generate visualization
    with asection("Generating visualization"):
        # Colors by tissue
        colors = np.zeros((n_cells, 3), dtype=np.float32)
        for i, tissue in enumerate(tissues):
            colors[i] = _tissue_color(tissue)

        # Count tissues
        tissue_counts: dict[str, int] = {}
        for t in tissues:
            tissue_counts[t] = tissue_counts.get(t, 0) + 1
        aprint("✓ Cells by tissue:")
        for t, count in sorted(tissue_counts.items(), key=lambda x: -x[1])[:12]:
            aprint(f"  {t}: {count:,}")

        radii = np.full(n_cells, 0.015, dtype=np.float32)

        labels = [f"{cell_types[i]} ({tissues[i]})" for i in range(n_cells)]

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

            sharpness = np.full(n_cells, 0.6, dtype=np.float32)

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

            scene.add_text(
                "Tabula Sapiens — Human Cell Atlas",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            n_types = len(set(cell_types))
            n_tissues = len(set(tissues))
            scene.add_text(
                f"{n_cells:,} cells • {n_types} cell types • {n_tissues} tissues • The Tabula Sapiens Consortium 2022",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
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
