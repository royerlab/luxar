#!/usr/bin/env python3
"""Self-Contained Demo: Human Multiome Peak 3D UMAP Visualization

This demo demonstrates:
- Loading biological data from local parquet file
- 3D UMAP embedding of ~1M single-cell ATAC-seq peaks
- Color-coded by cell type, lineage, timepoint, and other attributes
- Navigation through categorical attributes
- Complete workflow: load -> build scene -> serve -> cleanup

Dataset Structure:
    - ~1M points in 3D UMAP space
    - Multiple cell types
    - Developmental timepoints
    - Multiple lineages
    - Peak types and chromosomes

Usage:
    python demo_human_multiome_peak_umap.py

Controls:
    - Rotate to explore UMAP structure
    - Use dropdown to switch between biological attributes
    - Press '1' to select attribute dimension, then '['/']' to navigate
    - Different values shown in different colors
    - Ctrl+C to stop
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer, require_local_data
from luxar.utils._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.paths import get_demos_output_dir


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def load_human_umap_data() -> tuple[np.ndarray, dict, dict]:
    """Load 3D UMAP data from local parquet file.

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    with asection("Loading Human 3D UMAP Data"):
        data_path = require_local_data(get_data_dir() / "3d_umap_coords_human.parquet")
        aprint(f"Loading from {data_path}...")

        df = pd.read_parquet(data_path)
        aprint(f"Loaded {len(df):,} points")

        # Extract coordinates
        coordinates = df[["UMAP_1", "UMAP_2", "UMAP_3"]].values.astype(np.float32)
        aprint(
            f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]"
        )
        aprint(
            f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]"
        )
        aprint(
            f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]"
        )

        # Convert categorical columns to numeric indices
        aprint("\nProcessing annotations...")
        attributes = {}
        category_maps = {}

        categorical_cols = [
            "celltype",
            "chromosome",
            "leiden_coarse",
            "lineage",
            "peak_type",
            "timepoint",
        ]

        for col in categorical_cols:
            if col not in df.columns:
                continue

            # Convert to categorical and get codes
            cat = pd.Categorical(df[col])
            attributes[col] = cat.codes.astype(np.int32)
            category_maps[col] = list(cat.categories)

            n_unique = len(cat.categories)
            aprint(f"  {col}: {n_unique} unique values")

    return coordinates, attributes, category_maps


def create_human_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
    category_maps: dict | None = None,
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
        attr_types = [
            "celltype",
            "chromosome",
            "leiden_coarse",
            "lineage",
            "peak_type",
            "timepoint",
        ]

        # Filter to available attributes
        available_attrs = [a for a in attr_types if a in attributes]

        # Create one copy of points per attribute type
        all_positions = []
        all_colors = []

        for attr_idx, attr_name in enumerate(available_attrs):
            colors = attribute_to_color(attributes[attr_name], attr_name)

            # Create 4D positions: [attribute_view, x, y, z]
            positions_4d = np.column_stack(
                [
                    np.full(n_points, attr_idx, dtype=np.float32),
                    coordinates[:, 0],
                    coordinates[:, 1],
                    coordinates[:, 2],
                ]
            )

            all_positions.append(positions_4d)
            all_colors.append(colors)

            n_unique = len(np.unique(attributes[attr_name]))
            aprint(f"  Attribute {attr_idx} ({attr_name}): {n_unique} unique values")

        # Combine all attribute views
        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"Created {len(available_attrs)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Define dimensions with categorical attribute selector
        category_labels = [
            "Cell Type",
            "Chromosome",
            "Leiden Coarse",
            "Lineage",
            "Peak Type",
            "Timepoint",
        ]
        # Filter to match available attributes
        category_labels = category_labels[: len(available_attrs)]

        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=category_labels,
                    display=False,
                    description="Biological attribute for color coding",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        # Create scene
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            total_points = len(positions_combined)
            radii = np.full(total_points, 0.02, dtype=np.float32)
            sharpnesses = np.full(total_points, 0.6, dtype=np.float32)

            # Hover labels: resolve integer codes to category names, repeated per view
            per_cell_labels = []
            if category_maps:
                for i in range(n_points):
                    parts = []
                    for attr_name in available_attrs:
                        code = int(attributes[attr_name][i])
                        cats = category_maps.get(attr_name, [])
                        name = str(cats[code]) if code < len(cats) else str(code)
                        parts.append(name)
                    per_cell_labels.append("\n".join(parts))
            labels = per_cell_labels * len(available_attrs) if per_cell_labels else None

            scene.add_points(
                "Cells",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
                intensity=0.11,
                labels=labels,
            )

            # --- Overlays ---
            scene.add_text(
                "Human Multiome Peak UMAP",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Dimension-aware attribute labels + HTML legends
            attr_keys = [
                "celltype",
                "chromosome",
                "leiden_coarse",
                "lineage",
                "peak_type",
                "timepoint",
            ]
            for attr_id, (label, attr_key) in enumerate(
                zip(category_labels, attr_keys)
            ):
                scene.add_text(
                    f"Colored by: {label}",
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"attribute": attr_id},
                    transition="fade",
                    transition_duration=0.2,
                )

                if category_maps and attr_key in category_maps:
                    legend_html = build_legend_html(
                        attr_key, category_maps[attr_key], attributes.get(attr_key)
                    )
                    if legend_html:
                        scene.add_html(
                            legend_html,
                            position=(0.98, 0.5),
                            anchor="center-right",
                            opacity=0.9,
                            visible_range={"attribute": attr_id},
                            transition="fade",
                            transition_duration=0.2,
                        )

            scene.add_text(
                f"{n_points:,} peaks • Human scATAC-seq • 3D UMAP",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene created with {n_points:,} points")

    return n_points


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("HUMAN MULTIOME PEAK 3D UMAP DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing single-cell chromatin accessibility from human samples!")
    aprint("Data: 3D UMAP of ~1M peaks across developmental stages")
    aprint("")
    aprint("What you'll see:")
    aprint("  - ~1M points representing chromatin accessibility peaks")
    aprint("  - Colors indicate different cell types")
    aprint("  - Points cluster by cell type in UMAP space")
    aprint("  - Navigate attributes to see different biological features")
    aprint("")

    # Load data from local parquet
    coordinates, attributes, category_maps = load_human_umap_data()

    # Generate legend images for all attributes
    generate_all_legends(attributes, category_maps, prefix="human")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "human_multiome_peak_umap.luxar.zarr"
        _n_points = create_human_scene(
            output_path, coordinates, attributes, category_maps
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_human_") as tmpdir:
        output_path = Path(tmpdir) / "human_umap.luxar.zarr"

        # Create scene
        _n_points = create_human_scene(
            output_path, coordinates, attributes, category_maps
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate to explore UMAP structure")
        aprint("  - Zoom in to see individual cells")
        aprint("")
        aprint("  Press '1' to select ATTRIBUTE VIEW, then use [/]:")
        aprint("     0: Cell Type")
        aprint("     1: Chromosome")
        aprint("     2: Leiden Coarse")
        aprint("     3: Lineage")
        aprint("     4: Peak Type")
        aprint("     5: Timepoint")
        aprint("")
        aprint("  Same structure, different colors reveal different biology!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
