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
    - Use dropdown to switch between 7 biological attributes (Cell Type, Chromosome, etc.)
    - Press '1' to select attribute dimension, then '['/']' to navigate
    - Different values shown in different colors
    - Ctrl+C to stop
"""

DEMO_META = {
    "key": "zebrahub_multiome_peak_umap",
    "title": "Zebrahub Multiome Peak UMAP",
    "description": "3D UMAP of 640k single-cell ATAC-seq peaks (Zebrahub), colored by cell type across 6 timepoints.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 50,  # approx
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["zebrahub_multiome_peak"],
    "outputs": ["zebrahub_multiome_peak_umap", "zebrahub_umap"],
    "citation": {
        # First author of the Zebrahub-Multiome preprint this demo cites above.
        "short": "Kim et al. 2024",
        "doi": "10.1101/2024.10.18.618987",
    },
}

import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import zarr
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cache_computed,
    launch_viewer,
)
from luxar.demos._lod_policy import stream_ladder
from luxar.demos._support._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.paths import get_demos_output_dir

# Per-cell sphere radius in scene units, sized against the measured local
# spacing: the 641k-cell cloud has a median nearest-neighbour distance of
# ~0.025, so ~0.4x that keeps adjacent cells just short of touching. At the
# previous 0.02 the filament cores washed toward white (61% of lit pixels lost
# their hue at the opening framing), hiding most of the 27 cell-type colours.
POINT_RADIUS = 0.010


def load_zebrahub_umap_data(
    base_url: str = "https://public.czbiohub.org/royerlab/zebrahub/sequencing/3d-umaps/peak_umap_3d_annotated_v6",
) -> tuple[np.ndarray, dict, dict]:
    """Load 3D UMAP data from Zebrahub public zarr store.

    Args:
        base_url: URL to zarr store

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    # Attributes fetched from the remote zarr (fold into the cache key so a
    # changed attribute set never reuses a stale cache).
    attr_names = [
        "celltype",
        "chromosome",
        "leiden_coarse",
        "leiden_fine",
        "lineage",
        "peak_type",
        "timepoint",
    ]

    def _fetch() -> tuple[np.ndarray, dict, dict]:
        # Load coordinates
        aprint("Loading 3D UMAP coordinates...")
        try:
            coords_store = fsspec.get_mapper(f"{base_url}/coords.zarr")
            coords_flat = zarr.open(coords_store, mode="r")[:]
        except Exception as e:
            aprint(f"❌ Failed to load data from {base_url}")
            aprint(f"   Error: {e}")
            aprint("")
            aprint("💡 Possible reasons:")
            aprint("   • Network connection issues")
            aprint("   • Remote server unavailable")
            aprint("   • Data URL has changed")
            aprint("")
            aprint("Please check your internet connection and try again.")
            raise

        # Reshape to (N, 3)
        n_points = len(coords_flat) // 3
        coordinates = coords_flat.reshape(n_points, 3)

        aprint(f"✓ Loaded {n_points:,} points")
        aprint(
            f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]"
        )
        aprint(
            f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]"
        )
        aprint(
            f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]"
        )

        # Load attributes and their category mappings
        aprint("\nLoading cell annotations...")
        attributes = {}
        category_maps = {}

        for attr_name in attr_names:
            try:
                attr_store = fsspec.get_mapper(f"{base_url}/attribute_{attr_name}.zarr")
                z = zarr.open(attr_store, mode="r")
                attr_data = z[:]
                attributes[attr_name] = attr_data

                # Extract category mapping from zarr attrs
                if "map" in z.attrs:
                    category_maps[attr_name] = list(z.attrs["map"])

                n_unique = len(np.unique(attr_data))
                aprint(f"  {attr_name}: {n_unique} unique values")
            except Exception as e:
                aprint(f"  {attr_name}: Failed to load - {e}")

        return coordinates, attributes, category_maps

    # Cache the 640k×N remote fetch under ~/.cache/luxar/zebrahub_multiome_peak
    # (keyed on the selected attribute set) so repeat runs are offline.
    cache_key = "coords_attrs_" + "_".join(attr_names)
    with asection("Downloading Zebrahub 3D UMAP Data"):
        return cache_computed("zebrahub_multiome_peak", cache_key, _fetch, version=1)


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


def create_zebrahub_scene(
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
        # Attribute types for categorical navigation, with display labels.
        attr_display_labels = {
            "celltype": "Cell Type",
            "chromosome": "Chromosome",
            "leiden_coarse": "Leiden Coarse",
            "leiden_fine": "Leiden Fine",
            "lineage": "Lineage",
            "peak_type": "Peak Type",
            "timepoint": "Timepoint",
        }
        attr_types = list(attr_display_labels.keys())

        # Only attributes that actually loaded become views. The view coordinate
        # is the index within available_attrs (NOT the full attr_types list), so
        # a failed-to-load attribute does not leave a hole in the view axis or
        # shift the category labels / overlays out of sync.
        available_attrs = [name for name in attr_types if name in attributes]

        # Create one copy of points per available attribute type
        all_positions = []
        all_colors = []
        for view_idx, attr_name in enumerate(available_attrs):
            # Generate colors for this attribute
            colors = attribute_to_color(attributes[attr_name], attr_name)

            # Create 4D positions: [attribute_view, x, y, z]
            positions_4d = np.column_stack(
                [
                    np.full(n_points, view_idx, dtype=np.float32),
                    coordinates[:, 0],
                    coordinates[:, 1],
                    coordinates[:, 2],
                ]
            )

            all_positions.append(positions_4d)
            all_colors.append(colors)

            n_unique = len(np.unique(attributes[attr_name]))
            aprint(f"  Attribute {view_idx} ({attr_name}): {n_unique} unique values")

        # Combine all attribute views
        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"✓ Created {len(available_attrs)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Define dimensions with categorical attribute selector
        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=[attr_display_labels[name] for name in available_attrs],
                    display=False,
                    description="Biological attribute for color coding cells in UMAP space",
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

            # Add points with small radii for dense point cloud
            total_points = len(positions_combined)
            radii = np.full(total_points, POINT_RADIUS, dtype=np.float32)
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
                intensity=0.067,
                labels=labels,
                **link_attrs,
                layer=True,
                # Additive ladder only — no substitutive levels. Stacked over 7
                # attribute views on a hidden axis, so 4,485,810 total is 640,830
                # RESIDENT against a 5,591,040 Points cap — 8.7x under it. The
                # coarse levels served a framing the screen-area selector never
                # picks (finest anchored at half-screen occupancy, and this demo
                # opens auto-fitted). 17 groups -> 11. Same wiring as the census
                # demo.
                additive_lod=stream_ladder(len(positions_combined)),
            )

            # --- Overlays ---
            scene.add_text(
                "Zebrahub Multiome Peak UMAP",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            for attr_id, attr_key in enumerate(available_attrs):
                label = attr_display_labels[attr_key]
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

                # Generate HTML color legend from category maps
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

            add_demo_caption(
                scene,
                f"{n_points:,} peaks • Zebrafish • 3D UMAP • Kim et al. 2024",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Scene created with {n_points:,} points")

    return n_points


def main() -> None:
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
    coordinates, attributes, category_maps = load_zebrahub_umap_data()

    # Generate legend images for all attributes
    generate_all_legends(attributes, category_maps)

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "zebrahub_multiome_peak_umap.luxar.zarr"
        _n_points = create_zebrahub_scene(
            output_path, coordinates, attributes, category_maps
        )
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_zebrahub_") as tmpdir:
        output_path = Path(tmpdir) / "zebrahub_umap.luxar.zarr"

        # Create scene
        _n_points = create_zebrahub_scene(
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
        aprint("     0: Cell Type (30 types)")
        aprint("     1: Chromosome (genomic location)")
        aprint("     2: Leiden Coarse (broad clusters)")
        aprint("     3: Leiden Fine (detailed clusters)")
        aprint("     4: Lineage (6 lineages)")
        aprint("     5: Peak Type (4 types)")
        aprint("     6: Timepoint (6 developmental stages)")
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
