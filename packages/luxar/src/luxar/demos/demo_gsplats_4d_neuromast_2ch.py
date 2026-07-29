#!/usr/bin/env python
"""GSplats Demo: 4D Two-Channel Zebrafish Neuromast Timelapse.

Visualises a 4D (3D + time), two-channel light-sheet/iSIM recording of a
developing zebrafish lateral-line **neuromast** as Gaussian splats, with each
channel exposed as an independently-toggleable layer.

The two channels are the two fluorescent markers of the
``she:GFP; cldnb:lyn-mScarlet`` line:

  - **Membranes** (mScarlet, iSIM 561/605) — rendered with the ``bop_blue`` LUT
  - **Nuclei**    (GFP,      iSIM 488/525) — rendered with the ``bop_orange`` LUT

Each channel is a separate ``layer=True`` gsplats node, so the viewer's Layers
panel (press **L**) gives per-channel visibility, display range, gamma and
blending controls. Play the **Time** dimension to scrub the 100-timepoint
developmental sequence; both channels animate together (they are co-registered
and share one 4D coordinate space).

DATA SOURCE & CITATIONS:
    Adrian Jacobo lab (CZ Biohub / Rockefeller). iSIM, Richardson–Lucy
    deconvolved, motion-aligned. Original volumes on the CZ Biohub HPC:
    ``…/04192022_she_gfp_cldn_mscarlet_Timelapse3_3dpf/S1/{Membranes,Nuclei}``.

PIPELINE (how the bundled gsplats were produced — for provenance, NOT re-run
by this demo):
    1. Assemble each channel's 100 deconvolved timepoints into a
       ``time,z,y,x`` volume; subtract a single global background floor.
    2. Calibrate K* per channel (Noise2Self blind-spot sweep) → K* = 64,000.
    3. ``batch-fit`` each channel: 100 timepoints, 64k seeds, ``n2s`` preset,
       barrier-aware ``stream`` LOD merge (time axis a hard coarsening barrier).
    4. Anisotropy: scale Z ×2.5 (raw voxels are anisotropic) + normalise
       intensity, per channel.
    5. Redundancy-cull each timepoint (``--redundancy-threshold 0.20``,
       SSIM-flat) → ~11% lighter with imperceptible quality loss.

DATA STORAGE (important):
    These fitted gsplats are ~220 MB and are **not bundled with the repo** and
    **not yet hosted** for download. For now they live in a local store on this
    machine (see ``DATA_DIR`` below). This is the outstanding follow-up: upload
    the two ``.gsplats.zarr`` to the demo data host (as the other gsplat demos
    do via ``load_precomputed_gsplats``) and switch ``load_neuromast_gsplats``
    to fetch from there. Until then the demo runs only where ``DATA_DIR`` is
    populated.

USAGE:
    python demo_gsplats_4d_neuromast_2ch.py [--no-serve] [--serve-only]

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_4d_neuromast_2ch.luxar.zarr
    - Opens in the browser; press L for the Layers panel, play the Time slider.
"""

DEMO_META = {
    "key": "gsplats_4d_neuromast_2ch",
    "title": "4D Neuromast (2-channel timelapse)",
    "description": "4D two-channel zebrafish neuromast timelapse (membranes + nuclei) as Gaussian splats.",
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 220,  # approx (local store, not bundled/hosted)
        "compute": "medium",
        "gpu": "none",
        "local_data": "manual-file",
    },
    "caches": [],
    "outputs": ["gsplats_4d_neuromast_2ch"],
}

import os
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.demos import launch_viewer, parse_demo_flags
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

# Local store for the fitted gsplats. NOT bundled/hosted yet (see the module
# docstring's DATA STORAGE note). Override with $LUXAR_NEUROMAST_DATA_DIR.
DATA_DIR = Path(
    os.environ.get(
        "LUXAR_NEUROMAST_DATA_DIR",
        str(Path.home() / "luxar_demo_data" / "gsplats_neuromast_2ch"),
    )
)

# Channel configuration — each becomes an independently-toggleable layer.
# Named colormaps (not baked RGB) so the viewer applies the LUT at display
# time and the Layers panel can switch it interactively.
CHANNELS = [
    {
        "name": "membranes",
        "file": "neuromast_membranes.gsplats.zarr",
        "colormap": "bop_blue",  # mScarlet membranes, iSIM 561/605
        "marker": "cldnb:lyn-mScarlet (membranes)",
        # Membranes are a dense diffuse shell that otherwise dominates and hides
        # the nuclei — render at half opacity so both channels read.
        "opacity": 0.5,
    },
    {
        "name": "nuclei",
        "file": "neuromast_nuclei.gsplats.zarr",
        "colormap": "bop_orange",  # GFP nuclei, iSIM 488/525
        "marker": "she:GFP (nuclei)",
        "opacity": 1.0,
    },
]

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]

SCENE_NAME = "gsplats_4d_neuromast_2ch.luxar.zarr"


# =============================================================================
# Data loading
# =============================================================================
def resolve_channel_paths() -> list[Path]:
    """Resolve the per-channel gsplat paths in the local store.

    Returns the list of existing ``.gsplats.zarr`` paths (channel order), or
    raises with an actionable message if the local store isn't populated —
    since the data is not yet hosted for automatic download.
    """
    paths = [DATA_DIR / ch["file"] for ch in CHANNELS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Neuromast gsplat data not found in the local store:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nThis demo's fitted gsplats (~220 MB) are not bundled with the "
            f"repo and not yet hosted for download.\nPopulate {DATA_DIR} with the two "
            "`.gsplats.zarr` (or set $LUXAR_NEUROMAST_DATA_DIR to their location).\n"
            "See the module docstring's PIPELINE / DATA STORAGE notes."
        )
    return paths


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(channel_paths: list[Path], output_path: Path) -> Path:
    """Build the 4D two-channel scene: one layer-enabled gsplats node per marker.

    The gsplats are pre-fit 4D (``z, y, x, time``), already anisotropy-corrected
    (Z ×2.5), intensity-normalised and redundancy-culled, so we simply graft each
    channel with its LUT and ``layer=True``. Both channels share identical 4D
    bounds → they co-register and animate together over the Time dimension.
    """
    with asection("Creating 4D two-channel neuromast scene"):
        # Explicit, named 4D dims (not the generic dim0..dim3 from
        # build_dimensions_from_data). The fitted gsplat center columns are
        # ordered (Z, Y, X, Time) — Z first, from the fit's axes=time,z,y,x —
        # so the Dimensions list must follow that exact order. The three
        # spatial axes are displayed; Time is a DISCRETE (step=1) hidden axis
        # that drives the playback slider.
        node, _ = load_gsplat_node(str(channel_paths[0]))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds: min={np.round(bmin, 2)} max={np.round(bmax, 2)}")
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
                Dimension(
                    "Time",
                    unit="frame",
                    discrete=True,
                    step=1.0,
                    display=False,
                    range=(float(bmin[3]), float(bmax[3])),
                ),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(tone_mapping="ACES"),
            )
            scene.attrs["title"] = "GSplats: 4D Two-Channel Neuromast Timelapse"
            scene.attrs["description"] = (
                "Zebrafish lateral-line neuromast (she:GFP; cldnb:lyn-mScarlet), "
                "iSIM, 100 timepoints. Two toggleable layers — membranes (bop_blue) "
                "+ nuclei (bop_orange). Press L for the Layers panel; play the Time "
                "slider to scrub development."
            )

            for ch, path in zip(CHANNELS, channel_paths):
                with asection(f"Adding {ch['name']} layer ({ch['marker']})"):
                    scene.add_gsplats_from_file(
                        name=ch["name"],
                        path=str(path),
                        opacity=ch.get("opacity", 1.0),
                        # kappa at the slider's smallest non-zero step: the two
                        # channels are superimposed over the same neuromast, so
                        # meaningful absorption makes whichever layer draws
                        # first occlude the other. Near-zero kappa keeps
                        # volumetric's bounded accumulation without the
                        # occlusion.
                        absorption=0.05,
                        blending_mode="volumetric",
                        layer=True,
                        colormap=ch["colormap"],
                    )
                    aprint(f"  colormap={ch['colormap']}")

            scene.add_text(
                "Neuromast • membranes + nuclei",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Demo: 4D Two-Channel Neuromast Timelapse")
    aprint("=" * 70)
    aprint("Membranes (bop_blue) + Nuclei (bop_orange) • 100 timepoints")
    aprint("Press L for the Layers panel; play the Time slider.")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    channel_paths = resolve_channel_paths()
    scene_path = create_luxar_scene(channel_paths, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer… (press L for the Layers panel)")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
