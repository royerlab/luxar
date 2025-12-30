#!/usr/bin/env python3
"""Self-Contained Demo: Zebrahub Integrated Cells 3D UMAP

Visualize 95k integrated single cells from zebrafish with categorical attribute navigation.

Data: CZ Biohub Zebrahub - https://zebrahub.org
Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

Navigate between Cell Type and Timepoint views using the categorical dimension dropdown!
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
from luxar.utils.paths import get_demos_output_dir


def load_cells_data():  # type: ignore[no-untyped-def]
    """Load integrated cells UMAP from Zebrahub."""
    base = "https://public.czbiohub.org/royerlab/zebrahub/sequencing/3d-umaps/integrated_umap_3d_annotated"

    with asection("Loading Zebrahub Integrated Cells"):
        try:
            # Coordinates
            coords_flat = zarr.open(fsspec.get_mapper(f"{base}/coords.zarr"), mode="r")[
                :
            ]
            coords = coords_flat.reshape(-1, 3)
            aprint(f"✓ {len(coords):,} cells loaded")

            # Attributes
            attrs = {}
            for name in ["celltype", "timepoint"]:
                attrs[name] = zarr.open(
                    fsspec.get_mapper(f"{base}/attribute_{name}.zarr"), mode="r"
                )[:]
                aprint(f"  {name}: {len(np.unique(attrs[name]))} unique")
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

    return coords, attrs


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
    aprint("ZEBRAHUB INTEGRATED CELLS - 3D UMAP")
    aprint("=" * 70)
    aprint("95k cells • 32 cell types • 6 timepoints")
    aprint("https://zebrahub.org")
    aprint("")

    coords, attrs = load_cells_data()

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output = get_demos_output_dir() / "zebrahub_integrated_cells.zarr"
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
                    sharpness=np.full(len(positions), 4.0, dtype=np.float32),
                    opacity=0.8,
                )

            aprint(f"{len(positions):,} total points ({len(coords):,} per view)")

        aprint(f"Dataset generated at {output}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_zebrahub_cells_") as tmpdir:
        output = Path(tmpdir) / "cells.zarr"

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
                    sharpness=np.full(len(positions), 4.0, dtype=np.float32),
                    opacity=0.8,
                )

            aprint(f"{len(positions):,} total points ({len(coords):,} per view)")

        aprint("")
        aprint("NAVIGATION: Press '1' then [/] to switch:")
        aprint("   0: Cell Type (32 types)")
        aprint("   1: Timepoint (6 stages)")
        aprint("")

        try:
            subprocess.run(
                ["luxar", "serve", str(output), "--viewer", "--open"], check=True
            )
        except (KeyboardInterrupt, subprocess.CalledProcessError, FileNotFoundError):
            pass

    aprint("Done")


if __name__ == "__main__":
    main()
