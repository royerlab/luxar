#!/usr/bin/env python3
"""Mesh Demo: Isosurfaces of a 3D Fluorescence Volume (scikit-image cells3d)

Marching-cubes **isosurfaces** of the two-channel scikit-image ``cells3d``
volume — cell membranes and nuclei — added as two independent, toggleable
``layer=True`` mesh nodes and shaded by Luxar's view-anchored headlight.

This is the reference demo for the **Mesh** geometry type, and it is deliberately
the same dataset as ``demo_gsplats_3d_cells3d_multichannel``: run both and you
are comparing the two ways Luxar can represent one volume. Gaussian splats
approximate the whole intensity field and need no threshold; an isosurface picks
ONE level set and renders it as an opaque surface with real occlusion and
silhouettes. Neither is a better answer — they answer different questions, and
seeing the same nuclei both ways is the fastest way to feel the difference.

Isosurfaces and segmentation boundaries are the named target data for Mesh
(``docs/specs/MESH_NODE_SPEC.md`` §1), which is why this demo exists rather than
a synthetic shape: it is the routine output of the same pipelines Luxar already
serves, and before Mesh it could only be approximated by a dense point cloud.

================================================================================
WHAT THIS DEMO SHOWS
================================================================================

**Shading.** Mesh is the only Luxar geometry type that shades — the other three
are soft, emissive, per-element primitives with no surface orientation. The
marching-cubes gradient normals are written as the node's ``normals`` with an
explicit ``normal_dims``, so the surfaces come out smooth-shaded rather than
faceted. Compare with ``shading="flat"`` (which reads screen-space derivatives
instead) by editing ``SHADING`` below.

**Opaque by default.** Mesh defaults to ``opaque``, unlike the other three types'
``additive`` — the only blending mode that is unconditionally correct without
per-triangle depth sorting, and what a surface should look like. Nuclei sitting
*inside* membranes are therefore genuinely occluded, which is the whole point of
a surface representation.

**Layers.** Press **L** for the Layers panel: per-channel visibility, opacity,
and the mesh-only **Ambient** / **Shade falloff** sliders. Turning the membrane
layer off is the fastest way to see the nuclei it encloses.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source: scikit-image sample data (``skimage.data.cells3d()``)
Shape: (60, 2, 256, 256) — (Z, Channel, Y, X), uint16
Channel 0: Cell membranes
Channel 1: Cell nuclei (fluorescent stain)
Origin: Allen Institute for Cell Science
Voxel size: (0.29, 0.26, 0.26) µm in (Z, Y, X)

How to Cite:
------------
scikit-image: image processing in Python.
van der Walt et al. (2014). PeerJ 2:e453. DOI: 10.7717/peerj.453

WORKFLOW:
=========

1. **Load** cells3d from scikit-image (downloads ~1 MB on first use)
2. **Smooth** each channel lightly — marching cubes on raw microscopy produces a
   surface dominated by shot noise, not biology
3. **Extract** an isosurface per channel with ``skimage.measure.marching_cubes``,
   in micrometres via its ``spacing`` argument
4. **Add** each as a ``layer=True`` mesh node with its gradient normals
5. **Visualize** — toggle layers in the Layers panel (press L)

No GPU and no fitting step, which makes this the cheapest end-to-end demo of any
Luxar geometry type: marching cubes is CPU-only and takes a few seconds.

USAGE:
======
    python demo_mesh_isosurface_cells3d.py [--no-serve] [--serve-only]

Options:
    --no-serve:   Don't auto-launch the viewer after scene creation
    --serve-only: Skip extraction, just serve the existing scene

Output:
    - Scene saved to: demos/mesh_isosurface_cells3d.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "mesh_isosurface_cells3d",
    "title": "Cells3D Isosurfaces",
    "description": (
        "Marching-cubes isosurfaces of the scikit-image cells3d volume "
        "(membranes + nuclei) as shaded, toggleable mesh layers."
    ),
    "category": "microscopy",
    "geometry": "mesh",
    "requirements": {
        "download_mb": 1,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["mesh_isosurface_cells3d"],
}

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.utils.demos import launch_viewer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

#: Voxel spacing (Z, Y, X) in micrometres, passed straight to marching_cubes so
#: the emitted vertices are already in physical units.
VOXEL_SIZE_ZYX = (0.29, 0.26, 0.26)

#: Gaussian smoothing sigma in VOXELS, applied per channel before extraction.
#: Not cosmetic: marching cubes on raw fluorescence traces every shot-noise
#: excursion across the level, producing millions of triangles of noise sheet.
#: 1.0 removes that while leaving the cell boundaries where they are.
SMOOTH_SIGMA = 1.0

#: Isolevel as a fraction of the robust intensity span (1st–99.5th percentile).
#: Percentiles rather than min/max because a single hot pixel would otherwise set
#: the scale for the whole volume. 0.35 sits above the background haze and below
#: the bright cores for BOTH channels — a threshold each, tuned per channel, is
#: the first thing to reach for if you retarget this to other data.
ISOLEVEL_FRACTION = 0.35

#: Per-channel appearance. Colours are flat per layer on purpose: with a uniform
#: albedo, everything you see is the SHADING, which is what distinguishes mesh
#: from the three emissive types.
CHANNELS = [
    {
        "index": 0,
        "name": "membranes",
        "label": "Cell membranes",
        # Warm orange, echoing the bop_orange LUT its gsplat twin uses.
        "color": (1.0, 0.55, 0.15),
        "opacity": 1.0,
    },
    {
        "index": 1,
        "name": "nuclei",
        "label": "Cell nuclei",
        # Cool blue, echoing bop_blue.
        "color": (0.30, 0.60, 1.0),
        "opacity": 1.0,
    },
]

#: "smooth" uses the marching-cubes gradient normals; "flat" ignores them and
#: shades from screen-space derivatives (faceted — one normal per triangle).
SHADING = "smooth"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

Arbol.max_depth = 5


# =============================================================================
# Isosurface extraction
# =============================================================================


def extract_isosurface(volume: np.ndarray, name: str) -> tuple:
    """Smooth `volume`, then return `(vertices, faces, normals)` in micrometres.

    Vertices come back in the array's own (Z, Y, X) axis order, scaled by
    ``VOXEL_SIZE_ZYX`` through marching_cubes' ``spacing`` — so the scene's
    dimensions are declared (z, y, x) to match rather than transposing a
    multi-hundred-thousand-row array for cosmetics.

    The returned normals are the interpolated intensity GRADIENT, pointing toward
    DECREASING intensity (marching_cubes' ``gradient_direction="descent"``
    default) — i.e. outward from a bright object, which is the orientation a
    surface enclosing signal wants.
    """
    ndimage = require_module("scipy.ndimage")
    measure = require_module("skimage.measure")

    with asection(f"Extracting {name} isosurface"):
        v = volume.astype(np.float32)
        v = ndimage.gaussian_filter(v, SMOOTH_SIGMA)

        lo, hi = np.percentile(v, [1.0, 99.5])
        level = float(lo + ISOLEVEL_FRACTION * (hi - lo))
        aprint(f"intensity span (p1–p99.5): {lo:.0f}–{hi:.0f} → isolevel {level:.0f}")

        vertices, faces, normals, _values = measure.marching_cubes(
            v, level=level, spacing=VOXEL_SIZE_ZYX
        )
        aprint(f"{len(vertices):,} vertices, {len(faces):,} triangles")

        return (
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.uint32),
            np.ascontiguousarray(normals, dtype=np.float32),
        )


# =============================================================================
# Scene creation
# =============================================================================


def create_scene(output_path) -> None:
    """Extract both isosurfaces and write them as two mesh layers."""
    data_mod = require_module("skimage.data")

    with asection("Loading cells3d"):
        volume = data_mod.cells3d()
        aprint(f"shape {volume.shape} (Z, C, Y, X), dtype {volume.dtype}")

    surfaces = []
    for channel in CHANNELS:
        vertices, faces, normals = extract_isosurface(
            volume[:, channel["index"]], channel["name"]
        )
        surfaces.append((channel, vertices, faces, normals))

    # Centre on the volume's midpoint so the scene opens framed on the cells
    # rather than on a corner. One shared offset across both channels, or the
    # membranes and the nuclei would drift apart.
    all_vertices = np.vstack([s[1] for s in surfaces])
    centre = (all_vertices.min(axis=0) + all_vertices.max(axis=0)) / 2.0
    extent = all_vertices.max(axis=0) - all_vertices.min(axis=0)
    aprint(
        f"volume extent (z, y, x): {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} µm"
    )

    # Axis order matches the array: marching_cubes emits (Z, Y, X).
    dims = Dimensions(
        [
            Dimension("z", unit="um", display=True),
            Dimension("y", unit="um", display=True),
            Dimension("x", unit="um", display=True),
        ]
    )

    with asection("Writing scene"):
        with LuxarZarrCompiler(
            output_path,
            encoding_mode=EncodingMode.MEMORY,
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # ACES set explicitly — the house default, and stating it keeps
                    # the compiler's "nothing was chosen" LUT notice quiet.
                    tone_mapping="ACES",
                    # Baked pose: a three-quarter view of the slab. Load-bearing
                    # rather than cosmetic — the scene's first axis is Z, which for
                    # cells3d is the SHORT one (17 µm against 66 µm), so the
                    # default up-axis convention frames the volume edge-on and it
                    # reads as a line. `up` along Z puts the slab face-on.
                    camera=CameraConfig(
                        position=(
                            float(extent[0]) * 3.0,
                            float(extent[1]) * 0.9,
                            float(extent[2]) * 1.1,
                        ),
                        target=(0.0, 0.0, 0.0),
                        up=(1.0, 0.0, 0.0),
                    ),
                ),
            )

            for channel, vertices, faces, normals in surfaces:
                scene.add_mesh(
                    channel["name"],
                    vertices - centre,
                    faces,
                    normals=normals,
                    # The three axes the normals describe. Explicit rather than
                    # implied: for an nD mesh "the first three dimensions" is
                    # exactly the wrong guess (a (t, x, y, z) mesh's normals
                    # describe x/y/z, not t/x/y), so the attr is required.
                    normal_dims=[0, 1, 2],
                    colors=np.array([channel["color"]], dtype=np.float32),
                    shading=SHADING,
                    opacity=channel["opacity"],
                    # Toggleable in the Layers panel (press L).
                    layer=True,
                )
                aprint(f"added '{channel['name']}' ({channel['label']})")

    aprint(f"Scene written to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Extract the isosurfaces (unless ``--serve-only``) and open the viewer.

    No cache branch, unlike the heavier demos: extraction is a couple of seconds of
    CPU, so re-running from scratch is cheaper than the machinery to avoid it.
    """
    output_path = get_demos_output_dir() / "mesh_isosurface_cells3d.luxar.zarr"

    with asection("Mesh Isosurface Demo — cells3d"):
        if SERVE_ONLY:
            if not output_path.exists():
                aprint(f"No scene at {output_path} — run without --serve-only first.")
                return
            aprint("Serving existing scene (--serve-only)")
        else:
            create_scene(output_path)

        if not NO_SERVE:
            launch_viewer(output_path)


if __name__ == "__main__":
    main()
