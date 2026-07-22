#!/usr/bin/env python3
"""Self-Contained Demo: Zebrahub Integrated Cells 3D UMAP

Visualize 95k integrated single cells from zebrafish with categorical attribute navigation.

Data: CZ Biohub Zebrahub - https://zebrahub.org
Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

Navigate between Cell Type and Timepoint views using the categorical dimension dropdown!
"""

DEMO_META = {
    "key": "zebrahub_multiome",
    "title": "Zebrahub Integrated Cells 3D UMAP",
    "description": "3D UMAP of ~95k integrated zebrafish single cells with categorical attribute navigation.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 5,  # approx
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["zebrahub_multiome"],
    "outputs": ["zebrahub_multiome", "cells"],
}

import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import zarr
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import cache_computed, launch_viewer
from luxar.utils.paths import get_demos_output_dir


def load_cells_data():  # type: ignore[no-untyped-def]
    """Load integrated cells UMAP from Zebrahub.

    Returns coords, attrs (integer codes), and category maps (code → name).
    The remote fetch+parse is cached under ~/.cache/luxar/zebrahub_multiome so
    a second run is fully offline.
    """
    base = "https://public.czbiohub.org/royerlab/zebrahub/sequencing/3d-umaps/integrated_umap_3d_annotated"

    def _fetch():  # type: ignore[no-untyped-def]
        try:
            # Coordinates
            coords_flat = zarr.open(fsspec.get_mapper(f"{base}/coords.zarr"), mode="r")[
                :
            ]
            coords = coords_flat.reshape(-1, 3)
            aprint(f"✓ {len(coords):,} cells loaded")

            # Attributes (integer codes) + category maps from zarr attrs
            attrs = {}
            category_maps = {}
            for name in ["celltype", "timepoint"]:
                z = zarr.open(
                    fsspec.get_mapper(f"{base}/attribute_{name}.zarr"), mode="r"
                )
                attrs[name] = z[:]
                category_maps[name] = list(z.attrs.get("map", []))
                aprint(f"  {name}: {len(np.unique(attrs[name]))} unique")
                if category_maps[name]:
                    aprint(f"    categories: {category_maps[name][:5]}...")
        except Exception as e:
            aprint(f"❌ Failed to load data from {base}")
            aprint(f"   Error: {e}")
            aprint("")
            aprint("💡 Possible reasons:")
            aprint("   • Network connection issues")
            aprint("   • Remote server unavailable")
            aprint("   • Data URL has changed")
            aprint("")
            aprint("Please check your internet connection and try again.")
            raise

        return coords, attrs, category_maps

    with asection("Loading Zebrahub Multiome Integrated Cells"):
        return cache_computed("zebrahub_multiome", "coords_attrs", _fetch, version=1)


def attr_to_colors(values):  # type: ignore[no-untyped-def]
    """Generate distinct colors per unique value."""
    unique_vals = np.unique(values)
    n_unique = len(unique_vals)
    colors = np.zeros((len(values), 3), dtype=np.float32)

    for i, val in enumerate(unique_vals):
        hue = i / n_unique
        h = hue * 6.0
        c, x = 1.0, 1.0 * (1 - abs(h % 2 - 1))

        r, g, b = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)][
            int(h)
        ]
        colors[values == val] = [r, g, b]

    return colors


def main() -> None:
    aprint("=" * 70)
    aprint("ZEBRAHUB MULTIOME — Integrated 3D UMAP")
    aprint("=" * 70)
    aprint("95k cells • 32 cell types • 6 timepoints")
    aprint("https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1")
    aprint("")

    coords, attrs, category_maps = load_cells_data()

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output = get_demos_output_dir() / "zebrahub_multiome.luxar.zarr"
        with asection("Building Scene"):
            # Create 2 views (celltype, timepoint)
            all_pos, all_col = [], []

            for idx, (name, data) in enumerate(attrs.items()):
                pos_4d = np.column_stack([np.full(len(coords), idx), coords])
                all_pos.append(pos_4d)
                all_col.append(attr_to_colors(data))
                aprint(f"  View {idx}: {name}")

            positions = np.vstack(all_pos)
            colors = np.vstack(all_col)

            # Hover labels: resolve integer codes to human-readable names
            ct_map = category_maps.get("celltype", [])
            tp_map = category_maps.get("timepoint", [])
            per_cell_labels = [
                f"{ct_map[attrs['celltype'][i]] if attrs['celltype'][i] < len(ct_map) else attrs['celltype'][i]}"
                f" @ {tp_map[attrs['timepoint'][i]] if attrs['timepoint'][i] < len(tp_map) else attrs['timepoint'][i]}"
                for i in range(len(coords))
            ]
            labels = per_cell_labels * len(attrs)

            dims = Dimensions(
                [
                    Dimension(
                        "view",
                        unit="",
                        categories=["Cell Type", "Timepoint"],
                        display=False,
                        description="Categorical view: color by cell type or developmental timepoint",
                    ),
                    Dimension("x", unit="UMAP", display=True),
                    Dimension("y", unit="UMAP", display=True),
                    Dimension("z", unit="UMAP", display=True),
                ]
            )

            with LuxarZarrCompiler(output) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "Cells",
                    positions,
                    colors=colors,
                    radii=np.full(len(positions), 0.02, dtype=np.float32),
                    sharpness=np.full(len(positions), 0.6, dtype=np.float32),
                    opacity=0.8,
                    intensity=0.25,
                    labels=labels,
                )

                # --- Overlays ---
                # Title
                scene.add_text(
                    "Zebrahub Multiome — Integrated 3D UMAP",
                    position=(0.02, 0.02),
                    font_size=0.055,
                    anchor="top-left",
                    color="rgba(255,255,255,0.6)",
                    blend_mode="difference",
                )

                # Dimension-aware labels for view
                for i, label in enumerate(["Cell Type", "Timepoint"]):
                    scene.add_text(
                        label,
                        position=(0.02, 0.97),
                        font_size=0.015,
                        anchor="bottom-left",
                        color="#ffcc44",
                        visible_range={"view": float(i)},
                        transition="fade",
                        transition_duration=0.15,
                    )

                # Info + citation
                scene.add_text(
                    "95K cells • 32 cell types • Lange et al., Cell 2024",
                    position=(0.98, 0.97),
                    font_size=0.012,
                    anchor="bottom-right",
                    color="rgba(200,200,200,0.45)",
                )

            aprint(f"{len(positions):,} total points ({len(coords):,} per view)")

        aprint(f"Dataset generated at {output}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_zebrahub_multiome_") as tmpdir:
        output = Path(tmpdir) / "cells.luxar.zarr"

        with asection("Building Scene"):
            # Create 2 views (celltype, timepoint)
            all_pos, all_col = [], []

            for idx, (name, data) in enumerate(attrs.items()):
                pos_4d = np.column_stack([np.full(len(coords), idx), coords])
                all_pos.append(pos_4d)
                all_col.append(attr_to_colors(data))
                aprint(f"  View {idx}: {name}")

            positions = np.vstack(all_pos)
            colors = np.vstack(all_col)

            # Hover labels: resolve integer codes to human-readable names
            ct_map = category_maps.get("celltype", [])
            tp_map = category_maps.get("timepoint", [])
            per_cell_labels = [
                f"{ct_map[attrs['celltype'][i]] if attrs['celltype'][i] < len(ct_map) else attrs['celltype'][i]}"
                f" @ {tp_map[attrs['timepoint'][i]] if attrs['timepoint'][i] < len(tp_map) else attrs['timepoint'][i]}"
                for i in range(len(coords))
            ]
            labels = per_cell_labels * len(attrs)

            dims = Dimensions(
                [
                    Dimension(
                        "view",
                        unit="",
                        categories=["Cell Type", "Timepoint"],
                        display=False,
                        description="Categorical view: color by cell type or developmental timepoint",
                    ),
                    Dimension("x", unit="UMAP", display=True),
                    Dimension("y", unit="UMAP", display=True),
                    Dimension("z", unit="UMAP", display=True),
                ]
            )

            with LuxarZarrCompiler(output) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "Cells",
                    positions,
                    colors=colors,
                    radii=np.full(len(positions), 0.02, dtype=np.float32),
                    sharpness=np.full(len(positions), 0.6, dtype=np.float32),
                    opacity=0.8,
                    intensity=0.25,
                    labels=labels,
                )

                # --- Overlays ---
                # Title
                scene.add_text(
                    "Zebrahub Multiome — Integrated 3D UMAP",
                    position=(0.02, 0.02),
                    font_size=0.055,
                    anchor="top-left",
                    color="rgba(255,255,255,0.6)",
                    blend_mode="difference",
                )

                # Dimension-aware labels for view
                for i, label in enumerate(["Cell Type", "Timepoint"]):
                    scene.add_text(
                        label,
                        position=(0.02, 0.97),
                        font_size=0.015,
                        anchor="bottom-left",
                        color="#ffcc44",
                        visible_range={"view": float(i)},
                        transition="fade",
                        transition_duration=0.15,
                    )

                # Info + citation
                scene.add_text(
                    "95K cells • 32 cell types • Lange et al., Cell 2024",
                    position=(0.98, 0.97),
                    font_size=0.012,
                    anchor="bottom-right",
                    color="rgba(200,200,200,0.45)",
                )

            aprint(f"{len(positions):,} total points ({len(coords):,} per view)")

        aprint("")
        aprint("NAVIGATION: Press '1' then [/] to switch:")
        aprint("   0: Cell Type (32 types)")
        aprint("   1: Timepoint (6 stages)")
        aprint("")

        launch_viewer(output)

    aprint("Done")


if __name__ == "__main__":
    main()
