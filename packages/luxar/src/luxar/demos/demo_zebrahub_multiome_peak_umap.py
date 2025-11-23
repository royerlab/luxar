#!/usr/bin/env python3
"""Self-Contained Demo: Zebrahub Multiome Peak 3D UMAP Visualization

This demo demonstrates:
- Loading biological data from remote public zarr store
- 3D UMAP embedding of 640k single-cell ATAC-seq peaks
- Color-coded by cell type (30 cell types)
- Navigation through timepoints (6 developmental stages)
- Real scientific dataset from Zebrahub project
- Complete workflow: download → build scene → serve → cleanup

Data Source:
    CZ Biohub - Zebrahub Multiome Project
    3D UMAP of single-cell chromatin accessibility peaks
    URL: https://public.czbiohub.org/royerlab/zebrahub/...

    References:
    - Zebrahub: https://zebrahub.org
    - Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

Dataset Structure:
    - 640,830 points in 3D UMAP space
    - 30 distinct cell types
    - 6 developmental timepoints
    - 6 lineages
    - 4 peak types

Usage:
    python demo_zebrahub_multiome_peak_umap.py

Controls:
    - Rotate to explore UMAP structure
    - Press '4' to navigate through timepoints
    - Different cell types shown in different colors
    - Ctrl+C to stop
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import zarr
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler


def load_zebrahub_umap_data(
    base_url: str = "https://public.czbiohub.org/royerlab/zebrahub/sequencing/3d-umaps/peak_umap_3d_annotated_v6",
) -> tuple[np.ndarray, dict]:
    """Load 3D UMAP data from Zebrahub public zarr store.

    Args:
        base_url: URL to zarr store

    Returns:
        Tuple of (coordinates, attributes) where:
        - coordinates: (N, 3) array of UMAP positions
        - attributes: dict of attribute arrays
    """
    with asection("Downloading Zebrahub 3D UMAP Data"):
        # Load coordinates
        aprint("Loading 3D UMAP coordinates...")
        coords_store = fsspec.get_mapper(f"{base_url}/coords.zarr")
        coords_flat = zarr.open(coords_store, mode="r")[:]

        # Reshape to (N, 3)
        n_points = len(coords_flat) // 3
        coordinates = coords_flat.reshape(n_points, 3)

        aprint(f"✓ Loaded {n_points:,} points")
        aprint(f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]")
        aprint(f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
        aprint(f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]")

        # Load attributes
        aprint("\nLoading cell annotations...")
        attributes = {}

        for attr_name in ["celltype", "lineage", "timepoint", "peak_type"]:
            try:
                attr_store = fsspec.get_mapper(f"{base_url}/attribute_{attr_name}.zarr")
                attr_data = zarr.open(attr_store, mode="r")[:]
                attributes[attr_name] = attr_data

                n_unique = len(np.unique(attr_data))
                aprint(f"  {attr_name}: {n_unique} unique values")
            except Exception as e:
                aprint(f"  {attr_name}: Failed to load - {e}")

    return coordinates, attributes


def celltype_to_color(celltype_ids: np.ndarray, n_celltypes: int = 30) -> np.ndarray:
    """Convert cell type IDs to distinct colors.

    Args:
        celltype_ids: Cell type indices
        n_celltypes: Total number of cell types

    Returns:
        RGB colors (N, 3)
    """
    colors = np.zeros((len(celltype_ids), 3), dtype=np.float32)

    # Generate distinct colors for each cell type using HSV
    for ct in np.unique(celltype_ids):
        mask = celltype_ids == ct

        # Hue varies with cell type
        hue = ct / n_celltypes

        # Convert to RGB (simplified HSV to RGB)
        h = hue * 6.0
        c = 1.0  # Full saturation
        x = c * (1 - np.abs(h % 2 - 1))

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors[mask] = [r, g, b]

    return colors


def attribute_to_color(
    attribute_values: np.ndarray,
    attribute_name: str,
) -> np.ndarray:
    """Convert any attribute to distinct colors.

    Args:
        attribute_values: Attribute array
        attribute_name: Name for logging

    Returns:
        RGB colors
    """
    n_unique = len(np.unique(attribute_values))
    colors = np.zeros((len(attribute_values), 3), dtype=np.float32)

    # Generate distinct colors using HSV
    for i, val in enumerate(np.unique(attribute_values)):
        mask = attribute_values == val
        hue = i / n_unique

        # HSV to RGB
        h = hue * 6.0
        c = 1.0
        x = c * (1 - np.abs(h % 2 - 1))

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors[mask] = [r, g, b]

    return colors


def create_zebrahub_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
) -> int:
    """Create Luxar scene with categorical attribute visualization.

    Args:
        output_path: Where to write Luxar zarr
        coordinates: (N, 3) UMAP coordinates
        attributes: Dict of attribute arrays

    Returns:
        Number of points
    """
    n_points = len(coordinates)

    with asection("Building Multi-Attribute Scene"):
        # Define attribute types for categorical navigation
        attr_types = ["celltype", "chromosome", "leiden_coarse", "leiden_fine",
                      "lineage", "peak_type", "timepoint"]

        # Create one copy of points per attribute type
        all_positions = []
        all_colors = []

        for attr_idx, attr_name in enumerate(attr_types):
            if attr_name in attributes:
                # Generate colors for this attribute
                colors = attribute_to_color(attributes[attr_name], attr_name)

                # Create 4D positions: [attribute_view, x, y, z]
                positions_4d = np.column_stack([
                    np.full(n_points, attr_idx, dtype=np.float32),
                    coordinates[:, 0],
                    coordinates[:, 1],
                    coordinates[:, 2],
                ])

                all_positions.append(positions_4d)
                all_colors.append(colors)

                n_unique = len(np.unique(attributes[attr_name]))
                aprint(f"  Attribute {attr_idx} ({attr_name}): {n_unique} unique values")

        # Combine all attribute views
        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"✓ Created {len(attr_types)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Define dimensions with categorical attribute selector
        dims = Dimensions([
            Dimension(
                "attribute",
                unit="view",
                range=(0, len(attr_types) - 1),
                step=1,
                display=False,
                discrete=True,
                description="Attribute visualization (0=celltype, 1=chromosome, 2=leiden_coarse, 3=leiden_fine, 4=lineage, 5=peak_type, 6=timepoint)",
            ),
            Dimension("x", unit="UMAP", display=True),
            Dimension("y", unit="UMAP", display=True),
            Dimension("z", unit="UMAP", display=True),
        ])

        # Create scene
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add points with small radii for dense point cloud
            total_points = len(positions_combined)
            radii = np.full(total_points, 0.02, dtype=np.float32)
            sharpnesses = np.full(total_points, 4.0, dtype=np.float32)

            scene.add_points(
                "Cells",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
            )

        aprint(f"✓ Scene created with {n_points:,} points")

    return n_points


def main():
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("ZEBRAHUB MULTIOME PEAK 3D UMAP DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing single-cell chromatin accessibility from zebrafish!")
    aprint("Data: 3D UMAP of 640k peaks across developmental stages")
    aprint("")
    aprint("📚 References:")
    aprint("   • Zebrahub: https://zebrahub.org")
    aprint("   • Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1")
    aprint("")
    aprint("What you'll see:")
    aprint("  • 640k points representing chromatin accessibility peaks")
    aprint("  • Colors indicate 30 different cell types")
    aprint("  • Points cluster by cell type in UMAP space")
    aprint("  • Navigate timepoints to see developmental changes")
    aprint("")

    # Load data from remote zarr
    coordinates, attributes = load_zebrahub_umap_data()

    # Use temporary directory
    with tempfile.TemporaryDirectory(prefix="luxar_demo_zebrahub_") as tmpdir:
        output_path = Path(tmpdir) / "zebrahub_umap.zarr"

        # Create scene
        n_points = create_zebrahub_scene(output_path, coordinates, attributes)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  • Rotate to explore UMAP structure")
        aprint("  • Zoom in to see individual cells")
        aprint("")
        aprint("  🔑 Press '1' to select ATTRIBUTE VIEW, then use [/]:")
        aprint("     0: Cell Type (30 types)")
        aprint("     1: Chromosome (genomic location)")
        aprint("     2: Leiden Coarse (broad clusters)")
        aprint("     3: Leiden Fine (detailed clusters)")
        aprint("     4: Lineage (6 lineages)")
        aprint("     5: Peak Type (4 types)")
        aprint("     6: Timepoint (6 developmental stages)")
        aprint("")
        aprint("  → Same structure, different colors reveal different biology!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        try:
            # luxar CLI automatically finds available ports
            subprocess.run(
                ["luxar", "serve", str(output_path), "--viewer", "--open"],
                check=True,
            )
        except KeyboardInterrupt:
            aprint("\n🛑 Stopping demo...")
        except subprocess.CalledProcessError as e:
            aprint(f"\n❌ Error: {e}")
            aprint("💡 Make sure viewer is built")
            sys.exit(1)
        except FileNotFoundError:
            aprint("\n❌ Error: 'luxar' command not found")
            sys.exit(1)

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
