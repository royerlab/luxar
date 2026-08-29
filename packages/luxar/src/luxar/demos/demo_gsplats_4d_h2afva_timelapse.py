#!/usr/bin/env python
"""GSplats Demo: Zebrafish Embryogenesis, h2afva Timelapse (Light-Sheet).

Fifty-one timepoints of the **zebrahub** *h2afva* recording — a zebrafish embryo
whose every nucleus carries a histone-H2A variant fusion, imaged on a light-sheet
microscope — as one 4D Gaussian-splat scene. Step the Time axis and the embryo
develops: the blastoderm spreads over the yolk, the body axis extends, and the
nuclei divide and stream past each other.

Where :mod:`demo_gsplats_3d_h2afva_stack` shows ONE stack from this recording
(timepoint 234) and :mod:`demo_gsplats_3d_decimation_study` compares fits at
four splat budgets, this is the recording as a TIMELAPSE — the axis the other
two hold fixed.

STRUCTURE:
    One 4D gsplat leaf with a twelve-step progressive stream ladder. There are
    no spatial partitions or substitutive LOD levels; the viewer progressively
    streams the ladder. The stacked time axis is a hard coarsening barrier, so
    no coarse splat ever blends two timepoints together.

DATA SOURCE & CITATIONS:
    Royer lab, CZ Biohub San Francisco (zebrahub). Raw acquisition:
    ``h2afva/fused`` — 253 timepoints of 407 x 2048 x 2048 voxels, fused and
    deconvolved. Please cite the zebrahub resource when using this data.

WHICH TIMEPOINTS:
    Every fifth frame of the 253-timepoint recording — original indices
    0, 5, 10, ... 250 — giving 51 frames. The demo fixes that sampling with
    ``SOURCE_STRIDE = 5``. If an archive records ``source_stride`` or
    ``source_timepoints``, scene creation cross-checks those attributes; the
    pinned archive does not retain them. Its stacked axis is renumbered 0..50
    so the viewer's discrete navigation grid lands exactly on stored values.

    The Time axis is therefore a FRAME INDEX, not minutes. The acquisition
    interval is not recorded anywhere in this dataset or its metadata, and
    inventing one would put a fabricated number on a slider that looks
    authoritative. One step here is five original timepoints.

ANISOTROPY AND UNITS:
    The raw voxels are anisotropic by a factor of **4** along Z and the fitted
    splats carry that scaling, so the embryo has its correct proportions. The
    centers are in **lateral-pixel units**, NOT microns: bounds run to
    1624 x 2038 x 2046, i.e. 407 z-slices x 4 and 2048 lateral.

    The single-stack companion ships microns, because its archive had the
    lateral pitch (0.40625 um) folded in as a second uniform scale. This
    archive does not. Converting would mean a
    ``gsplat transform --scale 0.40625,0.40625,0.40625,1`` pass over the whole
    1.12 GB fit, which changes its bytes and therefore its published checksum.
    That is a data-side change, not a scene-authoring one, so the axes here are
    honest about being pixels. At the companion's calibration (0.40625 um
    laterally, 1.625 um axially) this envelope is 660 x 828 x 831 um.

PIPELINE (provenance of the bundled gsplats — NOT re-run here):
    1. Fit the full 253-timepoint ``h2afva/fused`` timelapse
       (``h2afva_253tp.gsplats.zarr``).
    2. Slice every fifth timepoint and renumber the stacked axis
       -> ``h2afva_51tp.gsplats.zarr``, one leaf with a twelve-step stream ladder.
    3. ``gsplat transform --scale 4,1,1,1`` -> isotropic proportions.

    The 51-frame fit is the manifest's default variant precisely because the
    full 253-frame one is 9.25 GB; this is the same recording at a fifth of the
    download.

USAGE:
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --no-serve
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --serve-only
"""

DEMO_META = {
    "key": "gsplats_4d_h2afva_timelapse",
    "title": "4D Zebrafish Embryogenesis (h2afva timelapse)",
    "description": (
        "Zebrafish embryogenesis as a 4D Gaussian-splat timelapse: 51 timepoints "
        "of histone-labelled nuclei in one progressively streamed twelve-step "
        "detail ladder."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 1064,
        "compute": "light",
        "gpu": "none",
        # Hosted-only: the fitted archive lives on Zenodo record 21912284, which
        # is still an unpublished draft with no base_url, so `ensure_dataset`
        # can build no URL for it. Until that record is published this resolves
        # from a hand-placed cache copy only — the same state the single-stack
        # companion is in.
        "local_data": "manual-file",
    },
    "caches": ["h2afva"],
    "outputs": ["gsplats_4d_h2afva_timelapse"],
    "citation": {
        "short": "Lange et al. 2024 (Zebrahub)",
        "ref": "Lange et al. 2024",
        "doi": "10.1016/j.cell.2024.09.047",
        "license": "CC BY 4.0",
    },
}

from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import read_node_attrs
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
)
from luxar.gsplats.io._archive import read_archive_root_attrs
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "h2afva"

SCENE_NAME = "gsplats_4d_h2afva_timelapse.luxar.zarr"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

#: Original-recording stride: one step of the Time axis is five acquisition
#: timepoints. Cross-checked against the archive when one records it.
SOURCE_STRIDE = 5


def _read_source_attrs(data_path: Path) -> dict:
    """Read source-root attrs from a directory or compressed archive."""
    if data_path.is_dir():
        return read_node_attrs(data_path) or {}
    return read_archive_root_attrs(data_path)


def resolve_data() -> Path:
    """Resolve the fitted 51-timepoint gsplats: cache -> in-repo -> Zenodo."""
    with asection("Resolving h2afva 51tp gsplats"):
        paths = ensure_dataset(DATASET, variant="51tp")
        aprint(f"Data: {paths[0]}")
        return paths[0]


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 4D scene from the progressive-ladder leaf."""
    with asection("Creating h2afva timelapse scene"):
        node, _ = load_gsplat_node(str(data_path))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds: min={np.round(bmin, 1)} max={np.round(bmax, 1)}")

        n_frames = int(round(float(bmax[3]) - float(bmin[3]))) + 1
        archive_attrs = _read_source_attrs(data_path)
        recorded_stride = archive_attrs.get("source_stride")
        if recorded_stride is not None and recorded_stride != SOURCE_STRIDE:
            raise ValueError(
                "h2afva archive source_stride does not match the demo: "
                f"expected {SOURCE_STRIDE}, got {recorded_stride!r}"
            )
        source_timepoints = archive_attrs.get("source_timepoints")
        if source_timepoints is not None and len(source_timepoints) != n_frames:
            raise ValueError(
                "h2afva archive source_timepoints do not match its time bounds: "
                f"expected {n_frames}, got {len(source_timepoints)}"
            )
        aprint(f"Timepoints: {n_frames} (every {SOURCE_STRIDE}th of the recording)")

        # Center columns are (Z, Y, X, T). Spatial units are lateral pixels —
        # see the module docstring's ANISOTROPY AND UNITS note for why this
        # demo does not claim microns while its single-stack companion does.
        dims = Dimensions(
            [
                Dimension(
                    "Z",
                    unit="px",
                    display=True,
                    range=(float(bmin[0]), float(bmax[0])),
                ),
                Dimension(
                    "Y",
                    unit="px",
                    display=True,
                    range=(float(bmin[1]), float(bmax[1])),
                ),
                Dimension(
                    "X",
                    unit="px",
                    display=True,
                    range=(float(bmin[2]), float(bmax[2])),
                ),
                # A frame index, deliberately not minutes: the acquisition
                # interval is not recorded for this dataset. `step=1` puts the
                # viewer's discrete navigation exactly on stored values, since
                # the stacked axis was renumbered 0..50 at slice time.
                Dimension(
                    "Time",
                    unit="frame",
                    display=False,
                    discrete=True,
                    step=1.0,
                    range=(float(bmin[3]), float(bmax[3])),
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
            )
            scene.attrs["title"] = "GSplats: Zebrafish Embryogenesis (h2afva timelapse)"
            scene.attrs["description"] = (
                f"Zebrafish embryo nuclei (histone H2A variant label) across "
                f"{n_frames} timepoints of the zebrahub h2afva light-sheet "
                f"recording — every {SOURCE_STRIDE}th frame of 253 — fitted as "
                "one Gaussian-splat leaf with a twelve-step progressive ladder. "
                "The time axis is a coarsening barrier, so no coarse splat blends "
                "two timepoints. Step Time to watch the body axis form. Press L "
                "for the Layers panel."
            )

            with asection(f"Adding gsplats (12-step ladder, {n_frames} frames)"):
                # Keep the additive mode used for the validated gallery build:
                # unlike volumetric compositing, it is order-independent as the
                # progressive rungs arrive.
                scene.add_gsplats_from_file(
                    name="zebrafish_nuclei_4d",
                    path=str(data_path),
                    blending_mode="additive",
                    # `plasma`, as on the single-stack companion: its
                    # yellow-to-magenta ramp keeps the bright nuclei distinct
                    # from the dimmer body signal behind them.
                    colormap="plasma",
                    # On a COLORMAPPED node `intensity`/`offset` ARE the scalar
                    # window feeding the LUT (`intensity = 1/(hi-lo)`,
                    # `offset = -lo/(hi-lo)`), not a colour gain.
                    #
                    # Measured on this fit's DECODED amplitudes, not copied from
                    # the companion: p50 0.00050, p90 0.0036, p99 0.032,
                    # p99.9 0.057, peak 0.200. Far more skewed than the
                    # companion's data, so reusing its relative window
                    # (0.001-0.111 of a 0.146 peak) put p99.9 in the bottom
                    # third of the LUT and the embryo rendered near-black.
                    #
                    # Window = p50 .. p99.9, chosen by rendering both.
                    #
                    # A p90 floor was tried on the theory that the dim 90% was
                    # haze flattening the silhouette at tile size. It rendered
                    # WORSE — dimmer and bluer, with less structure. The dim
                    # splats are body signal, not a veil to remove. Measured
                    # amplitudes: p50 0.00050, p90 0.0036, p99 0.032,
                    # p99.9 0.057, peak 0.200.
                    intensity=17.57,
                    offset=-0.00879,
                    gamma=2.2,
                    # Keep layer ownership explicit at the scene boundary rather
                    # than relying on archive attrs. `test_demo_layers` also reads
                    # the module source to enforce that contract.
                    layer=True,
                )

            scene.add_text(
                "Zebrafish • h2afva timelapse",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                f"Light-sheet • {n_frames} timepoints • histone-labelled nuclei",
                DEMO_META.get("citation"),
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


def main() -> None:
    """Resolve the data, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: Zebrafish Embryogenesis (h2afva timelapse)")
    aprint("=" * 70)
    aprint("51 timepoints • one leaf • 12-step progressive ladder")
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
