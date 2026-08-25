#!/usr/bin/env python3
"""Mesh Demo: Isosurfaces of a 3D Fluorescence Volume (scikit-image cells3d)

Marching-cubes **isosurfaces** of the two-channel scikit-image ``cells3d``
volume — cell membranes and nuclei — added as two independent, toggleable
``layer=True`` mesh nodes and shaded by Luxar's view-anchored offset key.

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

**Baked occlusion.** The material's key light says which way a face *turns*, not
how *enclosed* it is, so the crevice where two cells touch comes out as brightly
lit as the open outer wall. ``luxar.shading.bake_ambient_occlusion`` supplies the
missing term, using the same marching-cubes normals to restrict the integral to
the hemisphere each vertex faces, and it is multiplied into the per-vertex albedo.
It is a property of the geometry rather than of the view, so it holds as the
camera orbits. Set ``AO_STRENGTH = 0.0`` to see the surfaces without it.

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
4. **Bake** per-vertex ambient occlusion from those same normals
5. **Add** each as a ``layer=True`` mesh node with its gradient normals
6. **Visualize** — toggle layers in the Layers panel (press L)

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
    # scikit-image ships the sample, but the images are the Allen Institute's
    # (see the Dataset block above and skimage's own cells3d docstring). The
    # van der Walt PeerJ paper credits the LIBRARY, so citing it here would
    # attribute someone else's microscopy to the software that loads it; the
    # header keeps that software citation where it belongs.
    "citation": {
        "short": "Allen Institute for Cell Science (scikit-image cells3d)",
        "ref": "Allen Institute for Cell Science",
    },
}

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    launch_viewer,
    parse_demo_flags,
    require_module,
)
from luxar.demos._cinematic_camera import pull_in
from luxar.encoding import EncodingMode
from luxar.shading import bake_ambient_occlusion
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

#: Per-channel base colour. One hue per layer, so the only variation across a
#: surface comes from the two shading terms — the material's runtime key light
#: and the baked occlusion multiplied into the albedo by `_occluded_albedo`.
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

#: Ambient-occlusion radius, in µm. The mesh material's key light is a single
#: direction, so it shades which way a face turns but not how *enclosed* it is:
#: the gap between two touching cells and the open outer wall come out equally
#: lit. Occlusion is the missing term. 4 µm is well under a cell diameter, so it
#: responds to the contacts between cells rather than to the slab as a whole.
AO_RADIUS_UM = 4.0

#: Fraction of the ambient illumination that is DIRECT, and so occludable;
#: `1 - AO_STRENGTH` is the indirect, multiply-scattered ambient reaching even a
#: fully enclosed vertex. Lower than the library default because mesh is the one
#: type that ALREADY has a direct light: the material's key term supplies its own
#: directional component, so this stands in only for the ambient part of the
#: illumination rather than for all of it. At full strength the contacts go black.
#:
#: Left alone when the library's auto target was retuned for stronger contrast,
#: which is a checked decision rather than an oversight: because this demo
#: normalizes against the term's own maximum, the retune moved its measured
#: contrast by only ~12% (membranes 0.115 -> 0.140), and every LOWER strength
#: moved it further from the appearance that was already reviewed. Points needed
#: the extra push; a shaded surface, which puts one element in each pixel, did not.
AO_STRENGTH = 0.45

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

Arbol.max_depth = 5


# =============================================================================
# Isosurface extraction
# =============================================================================


def extract_isosurface(volume: np.ndarray, name: str) -> tuple:
    """Smooth `volume`, then return `(vertices, faces, normals)`.

    Vertices come back in micrometres, in the array's own (Z, Y, X) axis order,
    scaled by ``VOXEL_SIZE_ZYX`` through marching_cubes' ``spacing`` — so the
    scene's dimensions are declared (z, y, x) to match rather than transposing a
    multi-hundred-thousand-row array for cosmetics. (Normals are unitless
    directions; skimage does not apply ``spacing`` to them, so under this
    dataset's mild anisotropy they are tilted a few degrees off the true
    physical-surface normal — immaterial for shading.)

    The returned normals are the interpolated NEGATIVE intensity gradient,
    pointing toward DECREASING intensity — i.e. outward from a bright object,
    which is the orientation a surface enclosing signal wants. marching_cubes
    computes those normals the same way regardless of ``gradient_direction``; the
    flag only controls FACE WINDING. Its ``"descent"`` default does
    ``np.fliplr(faces)``, which leaves each triangle's right-handed winding
    opposite the outward normals — so exterior triangles render back-facing and
    the lighting gradient inverts. We pass ``"ascent"`` to keep the winding
    consistent with the outward normals.

    NB: skimage documents ``"ascent"`` as "exterior was greater than object",
    which is the opposite of this bright-signal data — but that label is written
    for a LEFT-hand-rule consumer (its source comment: "MC implementation is
    right-handed, but gradient_direction is left-handed"). Luxar's renderer is
    right-handed (CCW = front-facing), so the unflipped ``"ascent"`` winding is
    the one that matches our outward normals. In skimage 0.26 the flag only
    toggles the ``np.fliplr(faces)`` winding; the winding test guards against a
    well-meaning revert to ``"descent"``.
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
            v, level=level, spacing=VOXEL_SIZE_ZYX, gradient_direction="ascent"
        )
        aprint(f"{len(vertices):,} vertices, {len(faces):,} triangles")

        return (
            np.ascontiguousarray(vertices, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.uint32),
            np.ascontiguousarray(normals, dtype=np.float32),
        )


# =============================================================================
# Baked occlusion
# =============================================================================


def _occluded_albedo(
    channel: dict, vertices: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Per-vertex albedo: the channel colour darkened where the surface is enclosed.

    Baked **per channel**, so a surface is only occluded by itself. Letting the
    two channels occlude each other would be more physical — nuclei do sit inside
    membranes — but each channel is an independently toggleable layer, and a
    cross-baked nucleus would keep wearing membrane-shaped shadows after the
    membrane layer was switched off. Same reasoning as baking a timelapse
    per-timepoint rather than across the time axis.

    The marching-cubes gradient normals are passed straight through, which puts
    the bake on the cosine-hemisphere path: this is a surface, and on a surface
    the full sphere is diluted by the in-plane material every vertex shares.

    Args:
        channel: One :data:`CHANNELS` entry.
        vertices: ``(V, 3)`` vertex positions in (z, y, x) µm.
        normals: ``(V, 3)`` unit outward normals, same frame.

    Returns:
        ``(V, 3)`` float32 linear-light albedo.
    """
    occlusion = bake_ambient_occlusion(
        vertices,
        normals=normals,
        radius=AO_RADIUS_UM,
        strength=AO_STRENGTH,
    )

    # Rescale so the most exposed vertex keeps the channel colour untouched.
    # Occlusion always costs mean brightness, and letting it fall on the whole
    # surface would dim the demo relative to the exposure it was authored at;
    # normalizing against the term's own top end spends it on CONTRAST instead.
    normalized = occlusion / max(float(occlusion.max()), 1e-6)
    aprint(
        f"  {channel['name']}: occlusion raw "
        f"[{occlusion.min():.3f}, {occlusion.max():.3f}] "
        f"-> normalized [{normalized.min():.3f}, 1.000]"
    )
    base = np.asarray(channel["color"], dtype=np.float32)
    return (base[None, :] * normalized[:, None]).astype(np.float32)


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
                    cinematic_mode=True,
                    # ACES set explicitly — the house default, and stating it keeps
                    # the compiler's "nothing was chosen" LUT notice quiet.
                    tone_mapping="ACES",
                    # Baked pose: a three-quarter view of the slab. Load-bearing
                    # rather than cosmetic — the scene's first axis is Z, which for
                    # cells3d is the SHORT one (17 µm against 66 µm), so the
                    # default up-axis convention frames the volume edge-on and it
                    # reads as a line. `up` along Z puts the slab face-on.
                    camera=CameraConfig(
                        # The extent multiples are the framing this pose was
                        # tuned at; `pull_in` restates that framing for the
                        # cinematic preset's wider 35 mm lens, which the scene
                        # takes whole by pinning no `fov` of its own.
                        position=pull_in(
                            (
                                float(extent[0]) * 3.0,
                                float(extent[1]) * 0.9,
                                float(extent[2]) * 1.1,
                            )
                        ),
                        target=(0.0, 0.0, 0.0),
                        up=(1.0, 0.0, 0.0),
                    ),
                ),
                citation=DEMO_META["citation"],
            )

            for channel, vertices, faces, normals in surfaces:
                scene.add_mesh(
                    channel["name"],
                    vertices - centre,
                    faces,
                    colors=_occluded_albedo(channel, vertices, normals),
                    normals=normals,
                    # The three axes the normals describe. Explicit rather than
                    # implied: for an nD mesh "the first three dimensions" is
                    # exactly the wrong guess (a (t, x, y, z) mesh's normals
                    # describe x/y/z, not t/x/y), so the attr is required.
                    normal_dims=[0, 1, 2],
                    shading=SHADING,
                    opacity=channel["opacity"],
                    # Toggleable in the Layers panel (press L).
                    layer=True,
                )
                aprint(f"added '{channel['name']}' ({channel['label']})")
            add_demo_caption(
                scene,
                "scikit-image cells3d • isosurfaces",
                DEMO_META.get("citation"),
            )

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
