#!/usr/bin/env python
"""GSplats Demo: Zebrafish Embryo, Single h2afva Stack (Light-Sheet).

One 3D stack out of the **zebrahub** *h2afva* recording — a zebrafish embryo
whose every nucleus is labelled with a histone-H2A variant fusion, imaged on a
light-sheet microscope. This is timepoint 234 of 253, late enough that the body
axis has formed and the embryo wraps around the yolk.

Where the Drosophila companion demo is a single compact leaf, this one is the
**large-data** shape: 1.65 million splats organised as a ``kind=partition`` of
41 spatial parts, each carrying its own progressive (stream) ladder. The viewer
frustum-culls whole parts you are not looking at and refines the rest
progressively, which is what makes a stack this size interactive.

A second, toggleable layer draws that structure: one coloured wireframe box per
part (press **L** for the Layers panel to show or hide it). The boxes are each
part's content bounds — the extent the viewer actually culls against — so they
overlap wherever neighbouring parts share splats.

DATA SOURCE & CITATIONS:
    Royer lab, CZ Biohub San Francisco (zebrahub). Raw acquisition:
    ``h2afva/fused`` — 253 timepoints of 407 x 2048 x 2048 voxels, fused and
    deconvolved.

    Please cite the zebrahub resource when using this data.

ANISOTROPY:
    The raw voxels are anisotropic by a factor of **4** along Z; the fitted
    splats are scaled accordingly (``gsplat transform --scale 4,1,1``), matching
    the convention used for the full 253-timepoint fits. After scaling, the
    volume has the embryo's correct proportions.

    The acquisition records no voxel size, so the calibration comes from the
    instrument and from prior work on this dataset. It was imaged on a
    SiMView-type light-sheet, whose detection optics (16x onto a 6.5 um sCMOS)
    give **0.40625 um** laterally; the z:xy ratio of **4** is the one the
    full-timelapse fitting campaign established for isotropy, so the axial step
    is 4 x 0.40625 = **1.625 um**. At that calibration the embryo measures
    657 x 802 x 827 um, the right envelope for this stage.

    The ratio is the softer of the two numbers — it is prior convention rather
    than recorded metadata, and unlike the Drosophila companion demo it cannot
    be checked against the specimen's geometry: a zebrafish embryo at this stage
    is a yolk sphere with the body wrapped around it, so no axis can be asserted
    equal to another, and light-sheet attenuation truncates the measurable depth
    (a threshold-based extent finds 233 of the 407 z-slices the fitted splats
    actually occupy). Re-stamp both numbers if the acquisition settings surface.

PIPELINE (how the bundled gsplats were produced — provenance, NOT re-run here):
    1. Select timepoint 234 from ``h2afva/fused``.
    2. Reuse the calibrated K* from the full-timelapse campaign
       (Noise2Self blind-spot sweep, K* = 128,000, held-out peak 42.5 dB).
    3. ``gsplat fit --tiling content --cal … --recipe stream --n-lods 6``
       -> 1,653,405 splats across 41 content-balanced boxes, each laddered.
    4. ``gsplat transform --scale 4,1,1 --normalize-intensity 1.0``
       -> isotropic proportions, amplitudes normalised to a 0-1 range.

USAGE:
    python demo_gsplats_3d_h2afva_stack.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_3d_h2afva_stack.luxar.zarr
    - Opens in the browser; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_h2afva_stack",
    "title": "3D Zebrafish Embryo (h2afva stack)",
    "description": (
        "A zebrafish embryo's nuclei as 1.65M Gaussian splats in 41 frustum-culled, "
        "progressively-streamed spatial parts."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 27,
        "compute": "light",
        "gpu": "none",
        # Pending the Zenodo upload (R17 step 2) the data resolves from the
        # local cache only, so this is a manual-file demo today. It becomes a
        # plain download (local_data None) once the record URL is populated.
        "local_data": "manual-file",
    },
    "caches": ["gsplats_3d_h2afva_stack"],
    "outputs": ["gsplats_3d_h2afva_stack"],
}

import colorsys
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import ensure_dataset, launch_viewer, parse_demo_flags
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_3d_h2afva_stack"

SCENE_NAME = "gsplats_3d_h2afva_stack.luxar.zarr"

#: Wireframe width for the partition-box layer, in scene units. The stack is
#: ~2000 units across, so this reads as a hairline at full-object framing.
BOX_LINE_WIDTH = 4.0

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]


# =============================================================================
# Partition-box wireframes
# =============================================================================
#: The 12 edges of a box, as index pairs into the 8 corners produced by
#: :func:`_corners` (bit i of the corner index selects max over min on axis i).
_BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),  # along X (bit 0)
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),  # along Y (bit 1)
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # along Z (bit 2)
)


def _corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box, ordered so bit i selects max on axis i."""
    return np.array(
        [
            [
                bmax[0] if (i & 4) else bmin[0],
                bmax[1] if (i & 2) else bmin[1],
                bmax[2] if (i & 1) else bmin[2],
            ]
            for i in range(8)
        ],
        dtype=np.float32,
    )


def partition_box_lines(node: Any) -> tuple[np.ndarray, np.ndarray]:
    """Wireframe boxes for every part of a ``kind=partition`` gsplat subtree.

    Returns ``(vertices, colors)`` for ``line_type="segments"`` — consecutive
    vertex PAIRS are independent edges, so each box contributes 12 edges = 24
    vertices and no spurious edge ever joins one box to the next.

    The box drawn is each part's **content bounds** (the axis-aligned extent of
    the splat centers it holds), which is what the viewer frustum-culls against.
    It is deliberately NOT the planner's box: a content-tiled fit does not store
    its plan, and its boxes were built with an overlap margin, so the drawn boxes
    genuinely do overlap where neighbouring parts share splats. Each part gets a
    distinct hue so adjacent boxes stay tellable apart.
    """
    verts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    children = list(node.children)
    for i, child in enumerate(children):
        bounds = center_bounds(child)
        if bounds is None:  # an empty part contributes no box
            continue
        corners = _corners(bounds[0], bounds[1])
        verts.append(np.stack([corners[a] for e in _BOX_EDGES for a in e]))

        # Golden-ratio hue stepping: neighbouring part indices land far apart on
        # the colour wheel, so spatially adjacent boxes never share a hue.
        hue = (i * 0.61803398875) % 1.0
        rgb = np.array(colorsys.hsv_to_rgb(hue, 0.75, 1.0), dtype=np.float32)
        # Demo colours are authored sRGB but consumed as LINEAR light.
        cols.append(np.tile(rgb**2.2, (len(_BOX_EDGES) * 2, 1)))

    return np.concatenate(verts).astype(np.float32), np.concatenate(cols).astype(
        np.float32
    )


# =============================================================================
# Data loading
# =============================================================================
def resolve_data() -> Path:
    """Resolve the fitted gsplats: cache -> in-repo copy -> Zenodo."""
    with asection("Resolving h2afva gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 3D scene by grafting the 41-part partition subtree."""
    with asection("Creating h2afva zebrafish scene"):
        node, _ = load_gsplat_node(str(data_path))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds: min={np.round(bmin, 1)} max={np.round(bmax, 1)}")

        # Center columns are (Z, Y, X), in physical microns: the shipped data
        # carries the x4 z-scaling AND the 0.40625 um lateral pitch (see the
        # module docstring's ANISOTROPY note).
        dims = Dimensions(
            [
                Dimension(
                    "Z",
                    unit="µm",
                    display=True,
                    range=(float(bmin[0]), float(bmax[0])),
                ),
                Dimension(
                    "Y",
                    unit="µm",
                    display=True,
                    range=(float(bmin[1]), float(bmax[1])),
                ),
                Dimension(
                    "X",
                    unit="µm",
                    display=True,
                    range=(float(bmin[2]), float(bmax[2])),
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )
            scene.attrs["title"] = "GSplats: Zebrafish Embryo (h2afva stack)"
            scene.attrs["description"] = (
                "Zebrafish embryo nuclei (histone H2A variant label), one stack from "
                "the zebrahub h2afva light-sheet recording, fitted as 1.65M Gaussian "
                "splats in 41 spatial parts. Each part carries a progressive ladder "
                "and is frustum-culled independently. Press L for the Layers panel."
            )

            with asection("Adding gsplats (41-part partition)"):
                # Attributes land on the wrapper node; the viewer inherits them
                # down the partition subtree.
                scene.add_gsplats_from_file(
                    name="zebrafish_nuclei",
                    path=str(data_path),
                    # Same appearance reasoning as the Drosophila companion demo
                    # (see its comments): alpha-over keeps the surface nuclei
                    # crisp where a summing mode integrates the far side of the
                    # embryo through the near side, and the midtones are shaped
                    # with `gamma` rather than `intensity`, whose top clips.
                    # Measured on THIS dataset: raising intensity to 3 or 6
                    # washes out the left flank while adding nothing, so the
                    # window stays at 1.0 even though these amplitudes are much
                    # sparser (mean 0.023 of full scale) than the Drosophila fit.
                    blending_mode="normal",
                    colormap="viridis",
                    intensity=1.0,
                    gamma=2.2,
                    layer=True,
                )

            with asection("Adding partition-box wireframes"):
                box_verts, box_colors = partition_box_lines(node)
                n_boxes = len(box_verts) // (len(_BOX_EDGES) * 2)
                scene.add_lines(
                    name="partition_boxes",
                    vertices=box_verts,
                    # Independent edges, NOT one polyline: a polyline would draw
                    # a spurious edge from each box's last corner to the next
                    # box's first.
                    line_type="segments",
                    widths=np.full(len(box_verts), BOX_LINE_WIDTH, dtype=np.float32),
                    colors=box_colors,
                    # `normal`, not additive: additive wireframes have no
                    # occlusion, so every box's far edges bleed through the
                    # embryo and the overlay reads as a hairball.
                    blending_mode="normal",
                    opacity=0.55,
                    layer=True,
                )
                aprint(f"Added {n_boxes} box outlines ({len(box_verts):,} vertices)")

            scene.add_text(
                "Zebrafish • h2afva",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                "Light-sheet • histone-labelled nuclei",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the data, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: Zebrafish Embryo (h2afva single stack)")
    aprint("=" * 70)
    aprint("1.65M splats • 41 frustum-culled parts • progressive ladders")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    data_path = resolve_data()
    scene_path = create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer…")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
