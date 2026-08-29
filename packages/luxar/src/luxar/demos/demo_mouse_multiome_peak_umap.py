#!/usr/bin/env python3
"""Self-Contained Demo: Mouse Multiome Peak 3D UMAP Visualization

This demo demonstrates:
- Loading biological data from local parquet file
- 3D UMAP embedding of ~192k single-cell ATAC-seq peaks
- Color-coded by cell type, lineage, timepoint, and other attributes
- Navigation through categorical attributes
- Complete workflow: load -> build scene -> serve -> cleanup

Dataset Structure:
    - ~192k points in 3D UMAP space
    - Multiple cell types
    - Embryonic developmental timepoints (E7.5-E8.75)
    - Multiple lineages
    - Peak types and chromosomes

Usage:
    python demo_mouse_multiome_peak_umap.py

Controls:
    - Rotate to explore UMAP structure
    - Use dropdown to switch between biological attributes
    - Press '1' to select attribute dimension, then '['/']' to navigate
    - Different values shown in different colors
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "mouse_multiome_peak_umap",
    "title": "Mouse Multiome Peak UMAP",
    "description": "A 3D UMAP of ~192k mouse single-cell ATAC-seq peaks, colored by cell type and lineage.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "light",
        "gpu": "none",
        "local_data": "manual-file",
    },
    "citation": {
        "short": "Argelaguet et al. 2022; peak-UMAP analysis Kim et al. 2024",
        "ref": "Argelaguet / Kim et al. 2022–2024",
        "doi": "10.1101/2022.06.15.496239",
        "license": "CC BY 4.0",
    },
    "caches": [],
    "outputs": ["mouse_multiome_peak_umap", "mouse_umap"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    launch_viewer,
    require_local_data,
)
from luxar.demos._lod_policy import stream_ladder
from luxar.demos._support._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.paths import get_demos_output_dir


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent / "data"


def load_mouse_umap_data() -> tuple[np.ndarray, dict, dict]:
    """Load 3D UMAP data from local parquet file.

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    with asection("Loading Mouse 3D UMAP Data"):
        data_path = require_local_data(get_data_dir() / "3d_umap_coords_mouse.parquet")
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


def _celltype_link_attrs(
    attributes: dict,
    category_maps: dict | None,
    available_attrs: list[str],
    n_points: int,
) -> dict[str, object]:
    """Build aligned OLS link attributes when cell-type names exist."""
    if not category_maps or "celltype" not in available_attrs:
        return {}
    categories = category_maps.get("celltype", [])
    if not categories:
        return {}
    per_cell_keys = []
    for i in range(n_points):
        code = int(attributes["celltype"][i])
        per_cell_keys.append(
            str(categories[code]) if 0 <= code < len(categories) else ""
        )
    return {
        "keys": per_cell_keys * len(available_attrs),
        "link": "https://www.ebi.ac.uk/ols4/search?q={hover_key}",
        "copy": "{hover_key}",
    }


def create_mouse_scene(
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
        category_maps: Dict of attribute name -> list of category labels

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
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(cinematic_mode=True),
            )

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
                        name = str(cats[code]) if 0 <= code < len(cats) else str(code)
                        parts.append(name)
                    per_cell_labels.append("\n".join(parts))
            labels = per_cell_labels * len(available_attrs) if per_cell_labels else None

            # Click a cell to look its type up in the EBI Ontology Lookup
            # Service, right-click to copy the term (#1917). The label joins
            # every attribute view with newlines — chromosome and peak type
            # among them — so the query needs the bare cell type from `keys=`,
            # tiled per view exactly like the labels.
            #
            # A SEARCH, not a term page: these annotations are each paper's own
            # clustering, so a value may be an ontology term or free text.
            # Search resolves either.
            link_attrs = _celltype_link_attrs(
                attributes, category_maps, available_attrs, n_points
            )

            scene.add_points(
                "Cells",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
                intensity=0.25,
                labels=labels,
                **link_attrs,
                layer=True,
                # Additive ladder only — no substitutive levels. This is ONE
                # object shown whole, and the node is stacked on the hidden
                # `attribute` axis, so the slice the viewer makes resident is
                # 192,251 points against a 5,591,040 Points cap: 29x under it.
                # The coarse levels were ~17% of the store (68K + 464K + 2.9M of
                # 20M) serving a framing the screen-area selector never picks —
                # the finest level is anchored at half-screen occupancy and this
                # demo opens auto-fitted, so they were bytes nobody fetched. The
                # ladder also drops the store from 13 groups to 7, and hosted
                # first paint costs roughly one request per node.
                additive_lod=stream_ladder(len(positions_combined)),
            )

            # --- Overlays ---
            scene.add_text(
                "Mouse Embryo Multiome UMAP",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            for attr_id, (label, attr_key) in enumerate(
                zip(category_labels, available_attrs)
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

            # #1810 recorded the mouse citation in DEMO_META, so the stale
            # zebrafish credit removed in #1744 no longer leaves this uncited.
            add_demo_caption(
                scene,
                f"{n_points:,} peaks • Mouse E7.5–E8.75 • 3D UMAP",
                DEMO_META.get("citation"),
            )

        aprint(f"Scene created with {n_points:,} points")

    return n_points


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("MOUSE MULTIOME PEAK 3D UMAP DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing single-cell chromatin accessibility from mouse embryos!")
    aprint("Data: 3D UMAP of ~192k peaks across developmental stages (E7.5-E8.75)")
    aprint("")
    aprint("What you'll see:")
    aprint("  - ~192k points representing chromatin accessibility peaks")
    aprint("  - Colors indicate different cell types")
    aprint("  - Points cluster by cell type in UMAP space")
    aprint("  - Navigate attributes to see different biological features")
    aprint("")

    # Load data from local parquet
    coordinates, attributes, category_maps = load_mouse_umap_data()

    # Generate legend images for all attributes
    generate_all_legends(attributes, category_maps, prefix="mouse")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "mouse_multiome_peak_umap.luxar.zarr"
        _n_points = create_mouse_scene(
            output_path, coordinates, attributes, category_maps
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_mouse_") as tmpdir:
        output_path = Path(tmpdir) / "mouse_umap.luxar.zarr"

        # Create scene
        _n_points = create_mouse_scene(
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
