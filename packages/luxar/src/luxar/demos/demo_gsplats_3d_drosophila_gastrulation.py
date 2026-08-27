#!/usr/bin/env python
"""GSplats Demo: Drosophila Gastrulation (SiMView Light-Sheet).

A single 3D light-sheet stack of a *Drosophila melanogaster* embryo caught at
**gastrulation** — the ~20-minute window when the blastoderm's uniform shell of
nuclei buckles into the first body-plan structures. Two of them are visible in
this stack: the **cephalic furrow** (the transverse groove near the anterior
third) and the **posterior midgut invagination** (the pit at the posterior pole).

The embryo carries a ubiquitous histone label (``His2Av::mRFP1``), so every
nucleus is a bright blob — roughly 6,000 of them at this stage. That is exactly
the structure Gaussian splatting represents well: a dense field of compact,
locally-ellipsoidal emitters. The fit below spends ~200,000 splats on it.

DATA SOURCE & CITATIONS:
    Royer & Keller labs. Imaged on a SiMView multi-view light-sheet microscope
    under the AutoPilot adaptive framework:

      Royer, L.A., Lemon, W.C., Chhetri, R.K., Wan, Y., Coleman, M., Myers, E.W.
      & Keller, P.J. "Adaptive light-sheet microscopy for long-term,
      high-resolution imaging in living organisms."
      Nat. Biotechnol. 34, 1267-1278 (2016). doi:10.1038/nbt.3708

    Specimen: ``w; His2Av::mRFP1; +`` (Bloomington stock #23560).

VOXEL CALIBRATION (why the Z scale is what it is):
    The raw stack is anisotropic: 108 z-slices x 1352 x 532 px. The lateral
    pixel size follows from the instrument — Nikon 16x/0.8 detection objectives
    on Hamamatsu Orca Flash 4.0 cameras (6.5 um pitch), i.e. 6.5/16 =
    **0.40625 um/px** (Royer et al. 2016, Online Methods).

    The axial step is not recorded in the array metadata, so it was measured
    from the embryo's own geometry: a Drosophila embryo is a prolate ellipsoid,
    so its mid-length cross-section must be circular. The section spans 91
    z-slices and 433 lateral px, giving a ratio of 4.76 and therefore a z-step
    of 4.76 x 0.40625 = **1.93 um** (the standard ~2 um SiMView sampling, and
    consistent with the paper's 1.75 um detection depth of focus).

    Cross-checks, both independent of that ratio: at this calibration the embryo
    measures 521 um long and 190 um wide, against a textbook Drosophila embryo of
    ~500 x 180 um; and after scaling, the two cross-sectional axes agree to 2.5%
    (195 um axial vs 190 um lateral) — i.e. the section really is circular.

PIPELINE (how the bundled gsplats were produced — provenance, NOT re-run here):
    1. Extract timepoint 150 from ``DrosophilaHistone.zarr.zip [data]``, the
       500-timepoint recording.
    2. Calibrate K* by Noise2Self blind-spot cross-validation (``gsplat cal``):
       the curve is ``signal_limited`` (K* = 512,000) with diminishing returns
       flagged at 256,000 — the operating point used here.
    3. ``gsplat fit --tiling none --seeds 256000`` -> 200,023 splats.
       (The recorded run pinned ``--tiling none``, which at the time was
       required: ``--seeds`` was then applied *per tile*, so auto-tiling this
       stack into 21 tiles multiplied the calibrated budget by 21. Since #1556
       an integer ``--seeds`` is a whole-volume budget that a tiled fit divides
       across its tiles, so the pin is no longer needed to keep the budget
       honest — it is kept here only because it is what actually produced the
       numbers below.)
    4. Quality vs the original volume: **41.65 dB PSNR**, **37.07 dB foreground
       PSNR**. SSIM was not re-measured for this refit.
    5. ``gsplat transform --scale 1.93,0.40625,0.40625`` -> physical microns.
    6. ``gsplat lod --recipe stream --target-ms 200`` -> progressive ladder.

USAGE:
    python demo_gsplats_3d_drosophila_gastrulation.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_3d_drosophila_gastrulation.luxar.zarr
    - Opens in the browser; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_drosophila_gastrulation",
    "title": "3D Drosophila Gastrulation",
    "description": (
        "A Drosophila embryo at gastrulation (cephalic furrow + posterior midgut "
        "invagination) as 200k Gaussian splats, from SiMView light-sheet."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "light",
        "gpu": "none",
        # Pending the Zenodo upload (R17 step 2) the data resolves from the
        # local cache only, so this is a manual-file demo today. It becomes a
        # plain download (local_data None) once the record URL is populated.
        "local_data": "manual-file",
    },
    "caches": ["gsplats_3d_drosophila_gastrulation"],
    "outputs": ["gsplats_3d_drosophila_gastrulation"],
    "citation": {
        "short": "Royer et al. 2016",
        "doi": "10.1038/nbt.3708",
        "license": "CC BY 4.0",
    },
}

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
)
from luxar.demos._cinematic_camera import VIEWER_DEFAULT_FOV_DEG, pull_in
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import GSplatNode, center_bounds, is_matrix_shaped, iter_leaves
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_3d_drosophila_gastrulation"

SCENE_NAME = "gsplats_3d_drosophila_gastrulation.luxar.zarr"

# Physical voxel size (Z, Y, X) in microns — already baked into the centers by
# the fit pipeline (step 5 above). Kept here because the scene's dimension
# ranges are read off the data, but the UNITS are ours to declare.
VOXEL_UM = (1.93, 0.40625, 0.40625)

# The fitted archive stores amplitudes as RAW DETECTOR COUNTS (min 5.0, p50 70,
# p99.9 512, max 798). This scene normalises them at authoring time so the
# display window below reads as a plain fraction of a robust data scale rather
# than as detector counts. The 99.9th percentile maps to 1.0; the handful of
# hotter splats remain above 1.0 for the authored window to clip. One shared
# factor spans the WHOLE ladder (see `normalize_amplitudes`) — a per-sub-LOD
# factor would rescale the rungs against each other and make every streaming
# prefix render at a different exposure.
#
# This is also the scene's primary exposure change: the stored amplitude feeds
# both emitted radiance and volumetric optical depth. The display window only
# compensates the LUT lookup. `GSPLAT_OPACITY` is a further trim on top.
AMPLITUDE_NORM_RANGE = (0.0, 1.0)
AMPLITUDE_REFERENCE_PERCENTILE = 99.9

# Display window top, on the robust normalised scale. The original 0.737 window
# was chosen in a live Layers-panel session on the approved render. On this fit,
# the compensated top maps back to 5 + 1.153 * (512 - 5) ~= 589 detector counts:
# deliberately above p99.9 512, before clipping the hotter outliers toward 798.
DISPLAY_WINDOW_TOP = 0.737 * (798.0 - 5.0) / (512.0 - 5.0)

# In `volumetric` blending each pixel accumulates emission along the whole ray,
# so what you see is a sum over every splat behind it. Robust amplitude
# normalisation sets that scale; opacity trims it further. The inverse ratio
# compensates for the 798-max -> 512-p99.9 normalisation change, preserving the
# current fit's accumulated radiance and optical depth while making exposure
# outlier-stable.
#
# Absorption then only has to supply the depth cue — how much the near shell
# occludes the far one — so it can stay well below 1.
GSPLAT_OPACITY = 0.41 * (512.0 - 5.0) / (798.0 - 5.0)
GSPLAT_ABSORPTION = 0.57

# Opening camera. The embryo is a prolate ellipsoid whose long axis is centre
# column 1 -> world Y, i.e. already screen-vertical, and auto-rotation orbits
# about the world up axis, so the spin runs about the embryo's own long axis.
# Framing is authored at the VIEWER's default FOV and then pulled in for the
# cinematic 63 mm lens (see `_cinematic_camera.pull_in`).
CAMERA_FRAME_FILL = 0.85  # share of the half-frame the long axis subtends
AUTO_ROTATE_SPEED = 2.5

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]


# =============================================================================
# Data loading
# =============================================================================
def resolve_data() -> Path:
    """Resolve the fitted gsplats: cache -> in-repo copy -> Zenodo.

    ``ensure_dataset`` verifies the manifest sha256 at every step, so a partial
    or corrupted copy is never handed back.
    """
    with asection("Resolving Drosophila gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


# =============================================================================
# Scene construction
# =============================================================================
def normalize_amplitudes(node: GSplatNode) -> tuple[float, float]:
    """Robustly normalise the tree's amplitudes in place.

    Returns the ``(lo, reference_hi)`` count range that was mapped to
    :data:`AMPLITUDE_NORM_RANGE`. Values above the reference percentile remain
    above the output range for the display window to clip.

    ONE shared factor across every sub-LOD of every leaf, derived from the pooled
    minimum and 99.9th percentile: the additive rungs are prefixes of one splat
    set, so rescaling them independently would change their relative brightness
    and make each prefix render as a different exposure. A robust upper reference
    keeps one hot splat in a refit from darkening the whole scene.

    Rewritten IN PLACE into the loaded arrays rather than rebuilt into a fresh
    ``GSplatData``: the sub-LOD containers are frozen and reconstructing one
    through the ``additive_sublods=`` constructor drops the authored per-rung meta
    that the ``_node`` fast path carries straight off disk (coverage_fraction,
    the energy stamps the viewer's LOD upgrades read). Mutating the amplitude
    arrays touches only the values being normalised.
    """
    sublods = [s for leaf in iter_leaves(node) for s in leaf.additive_sublods]
    pooled = np.concatenate(
        [np.asarray(s.amplitudes, dtype=np.float64) for s in sublods]
    )
    lo = float(pooled.min())
    hi = float(np.percentile(pooled, AMPLITUDE_REFERENCE_PERCENTILE))
    out_lo, out_hi = AMPLITUDE_NORM_RANGE
    span = hi - lo
    if not span > 0:
        data_hi = float(pooled.max())
        if data_hi > lo:
            hi = data_hi
            span = hi - lo
        else:
            for s in sublods:
                amps = s.amplitudes
                if not amps.flags.writeable:
                    amps = np.array(amps, copy=True)
                    object.__setattr__(s, "amplitudes", amps)
                amps.fill(out_hi)
            return lo, hi
    scale = (out_hi - out_lo) / span
    for s in sublods:
        amps = s.amplitudes
        if not amps.flags.writeable:
            amps = np.array(amps, copy=True)
            object.__setattr__(s, "amplitudes", amps)
        np.subtract(amps, np.float32(lo), out=amps, casting="unsafe")
        np.multiply(amps, scale, out=amps, casting="unsafe")
        if out_lo != 0.0:
            amps += np.float32(out_lo)
    return lo, hi


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 3D scene from the pre-fitted, physically-scaled gsplats."""
    with asection("Creating Drosophila gastrulation scene"):
        node, _ = load_gsplat_node(str(data_path))
        bmin, bmax = center_bounds(node)
        amp_lo, amp_hi = normalize_amplitudes(node)
        aprint(
            f"Amplitudes normalised: [{amp_lo:.1f}, p99.9 {amp_hi:.1f}] counts -> "
            f"{AMPLITUDE_NORM_RANGE}; hotter splats remain above 1"
        )
        aprint(f"Scene bounds (um): min={np.round(bmin, 1)} max={np.round(bmax, 1)}")
        aprint(
            f"Embryo extent: {bmax[1] - bmin[1]:.0f} um long, "
            f"{bmax[2] - bmin[2]:.0f} um wide"
        )

        # Opening pose. Centre column 1 is the embryo's long axis and maps to
        # world Y, so it is already screen-vertical and auto-rotation (which
        # orbits the up axis) spins the embryo about its own length. The camera
        # backs off along +world Z, so the silhouette shows the full length and
        # width, far enough that the long axis subtends
        # CAMERA_FRAME_FILL of the half-frame. Solved from the data's own bbox
        # rather than pinned, so a refit or a re-scaled fit reframes itself.
        centre = tuple(float(v) for v in (bmin + bmax) / 2.0)
        half_len = float(bmax[1] - bmin[1]) / 2.0
        half_fov = np.radians(VIEWER_DEFAULT_FOV_DEG / 2.0)
        cam_dist = (half_len / CAMERA_FRAME_FILL) / float(np.tan(half_fov))
        cam_eye = (centre[0], centre[1], centre[2] + cam_dist)
        aprint(f"Camera: target {np.round(centre, 1)} distance {cam_dist:.0f} um")

        # Center columns are (Z, Y, X) — the fit's array order — so the
        # Dimensions list must follow that exact order. All three are spatial
        # and displayed; the stack is a single timepoint, so there is no
        # hidden axis.
        dims = Dimensions(
            [
                Dimension(
                    "Z", unit="µm", display=True, range=(float(bmin[0]), float(bmax[0]))
                ),
                Dimension(
                    "Y", unit="µm", display=True, range=(float(bmin[1]), float(bmax[1]))
                ),
                Dimension(
                    "X", unit="µm", display=True, range=(float(bmin[2]), float(bmax[2]))
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # exposure is in LOG2 STOPS and stays neutral: with the emission
                # itself scaled by `GSPLAT_OPACITY` the frame already lands inside
                # the tone curve, so there is no stop to make up.
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    exposure=0.0,
                    auto_rotate=True,
                    auto_rotate_speed=AUTO_ROTATE_SPEED,
                    camera=CameraConfig(
                        position=pull_in(cam_eye, centre), target=centre
                    ),
                ),
                citation=DEMO_META["citation"],
            )
            scene.attrs["title"] = "GSplats: Drosophila Gastrulation (SiMView)"
            scene.attrs["description"] = (
                "Drosophila melanogaster embryo (His2Av::mRFP1) at gastrulation, "
                "imaged on a SiMView adaptive light-sheet microscope and fitted as "
                "200k Gaussian splats. The transverse groove is the cephalic furrow; "
                "the posterior pit is the midgut invagination. Voxels are physical "
                "microns (Z 1.93, XY 0.40625). Press L for the Layers panel."
            )

            with asection("Adding gsplats"):
                appearance = dict(
                    # `volumetric` emission-absorption, which is what a single
                    # fluorescence channel wants. Opacity and absorption do
                    # different jobs and are not interchangeable: amplitudes set
                    # the primary emission/optical-depth scale, opacity trims both,
                    # and absorption governs how quickly that depth accumulates.
                    blending_mode="volumetric",
                    absorption=GSPLAT_ABSORPTION,
                    opacity=GSPLAT_OPACITY,
                    colormap="magma",
                    # On a COLORMAPPED node `intensity`/`offset` are the scalar
                    # display WINDOW, not a post-LUT gain: the viewer recovers
                    # it as `[-offset/i, (1-offset)/i]`. Amplitudes arrive
                    # normalised to [0, 1], so this states the window
                    # `[0, DISPLAY_WINDOW_TOP]`.
                    intensity=1.0 / DISPLAY_WINDOW_TOP,
                    offset=0.0,
                    gamma=1.0,
                    layer=True,
                )
                if is_matrix_shaped(node):
                    scene.add_gsplats_from_data(
                        name="drosophila_nuclei",
                        result=GSplatData.from_tree(node),
                        **appearance,
                    )
                else:
                    from luxar.core.group.gsplats_pipeline.from_io import (
                        graft_gsplat_node,
                    )

                    graft_gsplat_node(
                        scene,
                        name="drosophila_nuclei",
                        node=node,
                        **appearance,
                    )

            scene.add_text(
                "Drosophila • gastrulation",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene, "SiMView light-sheet • His2Av::mRFP1", DEMO_META.get("citation")
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the data, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: Drosophila Gastrulation (SiMView Light-Sheet)")
    aprint("=" * 70)
    aprint("200k splats • cephalic furrow + posterior midgut invagination")
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
