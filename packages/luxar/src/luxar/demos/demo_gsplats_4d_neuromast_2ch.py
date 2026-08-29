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

PIPELINE — reproducible per channel with ``--recompute``:
    1. Assemble the channel's 100 deconvolved timepoints into one
       ``time,z,y,x`` array, and subtract a single global background floor
       measured once for that channel (105.991 membranes / 103.888 nuclei).
       ``--recompute`` starts from the assembled array and does the
       subtraction; assembling from the microscope's per-timepoint files is
       upstream of this demo.
    2. Calibrate K* per channel (Noise2Self blind-spot sweep) → K* = 64,000.
       Recorded, not re-run: the sweep is hours and its answer is stable.
    3. ``batch-fit run``: 100 timepoints, one uniform tile each, ``n2s`` preset,
       64k seeds, ``--floor auto``, no fit-time cull.
    4. Redundancy-cull every per-timepoint tile
       (``-m redundancy --redundancy-threshold 0.20``) → ~11% lighter at
       SSIM-flat quality. Per TILE, before the merge: culling the merged
       timelapse would need the whole 4D reconstruction in memory at once.
    5. ``batch-fit merge --recipe stream`` over the culled tiles → ONE leaf with
       an 8-rung progressive ladder. The stacked time axis is a hard coarsening
       barrier, so no rung blends two timepoints.
    6. ``gsplat transform --scale 2.5,1,1,1 --normalize-intensity 1.0``
       → isotropic Z, amplitudes on a 0-1 scale.

    Step 3 must be run with ``--jobs-per-gpu 12``, not ``auto``. On this box
    ``auto`` sized 100 concurrent workers for 100 tasks and every one of them
    was OOM-killed (``exit -9``) before a single tile landed — an
    over-subscribed GPU here fails all-at-once rather than degrading.

DATA STORAGE (important):
    These fitted gsplats are ~220 MB unzipped and are **not bundled with the
    repo**. Both channels are uploaded to the ``cc-by`` Zenodo record and pinned
    by SHA-256 in ``demos/data_manifest.json`` (as the 133 MB
    ``.gsplats.zarr.zip`` pair), but that record is still an unsubmitted draft —
    it carries ``published: false``, so the fetch leg builds no URL and nothing
    is downloadable yet. For now the data lives in a local store on this machine
    (see ``DATA_DIR`` below), so the demo runs only where ``DATA_DIR`` is
    populated. Once the record is published, switch ``resolve_channel_paths`` to
    ``ensure_dataset("gsplats_4d_neuromast_2ch")`` (as the other gsplat demos do
    via ``load_precomputed_gsplats``) and drop the local store.

USAGE:
    python demo_gsplats_4d_neuromast_2ch.py [--no-serve] [--serve-only]
    python demo_gsplats_4d_neuromast_2ch.py --recompute \
        --source-membranes /path/to/membranes.zarr \
        --source-nuclei    /path/to/nuclei.zarr

    --no-serve:    Build the scene but don't launch the viewer.
    --serve-only:  Skip the build, just serve the already-built scene.
    --recompute:   Refit both channels from their assembled source arrays
                   (needs a CUDA GPU and roughly two hours).

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
    "citation": {
        "short": "Jacobo lab, CZ Biohub San Francisco",
        "ref": "Jacobo lab / CZ Biohub",
        "license": "CC BY 4.0",
    },
}

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    launch_viewer,
    parse_demo_flags,
    parse_path_arg,
    run_luxar_cli,
)
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds, iter_leaves
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
        # ---- recompute recipe, per channel ----
        #: ``--source-<name> PATH``: the assembled (time, z, y, x) array.
        "source_flag": "source-membranes",
        #: Global background floor, measured ONCE on this channel and recorded.
        #: Re-measuring would drift, and the fit's own `--floor auto` runs on top
        #: of the subtraction rather than replacing it.
        "background_floor": 105.9911880493164,
        #: What the recipe must reproduce, from the merge that built the shipped
        #: archive ("Wrote single stream lod: 5,864,440 splats, 4D").
        "expected_splats": 5_864_440,
    },
    {
        "name": "nuclei",
        "file": "neuromast_nuclei.gsplats.zarr",
        "colormap": "bop_orange",  # GFP nuclei, iSIM 488/525
        "marker": "she:GFP (nuclei)",
        "opacity": 1.0,
        "source_flag": "source-nuclei",
        "background_floor": 103.88801574707031,
        "expected_splats": 5_530_300,
    },
]

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: Per-channel ``--source-*`` paths, resolved once at import like the other flags.
SOURCE_ARGS = {ch["name"]: parse_path_arg(str(ch["source_flag"])) for ch in CHANNELS}

SCENE_NAME = "gsplats_4d_neuromast_2ch.luxar.zarr"

# ---- Recorded fit recipe, shared by both channels ---------------------------
#: Axis labels of the assembled source array. Passed explicitly: the fit records
#: them, and getting them wrong silently fits the time axis as a spatial one.
SOURCE_AXES = "time,z,y,x"
#: Full extent both channels must have; asserted before the GPU is touched.
SOURCE_SHAPE = (100, 84, 580, 576)
#: Calibrated splat budget per timepoint (Noise2Self blind-spot sweep).
SEEDS = 64_000
#: Fitting preset. `n2s` is the noise-aware one that pairs with the calibration.
PRESET = "n2s"
#: One uniform tile per timepoint: 640 exceeds every spatial extent above, so
#: the volume stays whole and there are no tile seams to apodize.
TILE_SIZE = 640
#: Concurrent fit workers per GPU. NOT `auto` — see the docstring's warning.
JOBS_PER_GPU = 12
#: Redundancy cull threshold, chosen from a measured 0.02/0.05/0.10/0.20/0.35
#: sweep as the most aggressive setting still SSIM-flat.
REDUNDANCY_THRESHOLD = 0.20
#: Voxel anisotropy: z is 2.5x the lateral pitch on this instrument.
VOXEL_SCALE = (2.5, 1.0, 1.0, 1.0)
#: Amplitudes normalised to a unit peak, so appearance does not depend on the
#: recording's absolute intensity scale.
NORMALIZE_INTENSITY = 1.0
#: Progressive rungs the merge produced for each channel.
EXPECTED_RUNGS = 8


# =============================================================================
# Data loading
# =============================================================================
#: Array name inside the background-subtracted intermediate group. Named rather
#: than left at the store root because the writer facade creates arrays UNDER a
#: group, and rooting an array directly would mean bypassing it.
BGSUB_ARRAY_KEY = "volume"


@contextmanager
def _open_source(path: Path) -> Iterator[Any]:
    """Yield a channel source's 4D array and close any zip store afterwards.

    The zip branch is load-bearing: zarr-python 3 no longer sniffs the `.zip`
    suffix, and the membrane channel ships as a 7.5 GB zipped array, so handing
    it straight to ``zarr.open`` raises GroupNotFoundError and blames a file that
    is perfectly good.
    """
    import zarr

    store = None
    try:
        if path.suffix == ".zip":
            from zarr.storage import ZipStore

            # The caller slices per timepoint, so keep the store open for the
            # yielded array's lifetime rather than loading the archive.
            store = ZipStore(str(path), mode="r")
            yield zarr.open(store=store, mode="r")
        else:
            yield zarr.open(str(path), mode="r")
    finally:
        if store is not None:
            store.close()


def _subtract_background(src: Path, out: Path, floor: float) -> Path:
    """Write ``src - floor`` (clipped at 0) under ``out``, one timepoint at a time.

    Streamed rather than vectorised over the whole array because a channel is
    100 x 84 x 580 x 576 float32 = 11.2 GB, and one expression would hold the
    input, the shifted copy and the clipped copy at once.

    Returns the path of the written group; the array is ``out/<BGSUB_ARRAY_KEY>``.
    """
    from luxar._zarr_compat import create_array, open_group

    with _open_source(src) as array:
        shape = tuple(getattr(array, "shape", ()) or ())
        if shape != SOURCE_SHAPE:
            raise SystemExit(
                f"{src.name} has shape {shape or 'no array at its root'}, want "
                f"{SOURCE_SHAPE}. Both channels of this recording share that extent, "
                f"so a mismatch means the wrong file or a partial assembly."
            )
        with asection(f"Subtracting background floor {floor:.4f} -> {out.name}"):
            group = open_group(out, mode="w")
            dest = create_array(
                group,
                BGSUB_ARRAY_KEY,
                shape=SOURCE_SHAPE,
                dtype="float32",
                # One timepoint per chunk: every consumer downstream (the fit, the
                # cull, this loop) reads exactly one timepoint at a time.
                chunks=(1,) + SOURCE_SHAPE[1:],
                compressor=None,
            )
            for t in range(SOURCE_SHAPE[0]):
                frame = np.asarray(array[t], dtype=np.float32)
                np.subtract(frame, np.float32(floor), out=frame)
                np.clip(frame, 0.0, None, out=frame)
                dest[t] = frame
            dest.attrs["axes"] = SOURCE_AXES
            dest.attrs["background_floor"] = float(floor)
            dest.attrs["source"] = str(src)
    return out


def _validate_rebuilt_channel(channel: dict, leaves: Sequence[Any]) -> int:
    """Validate the merged leaf count and progressive-rung contract."""
    name = str(channel["name"])
    if len(leaves) != 1:
        raise RuntimeError(f"{name}: merge produced {len(leaves)} leaves, expected one")
    rungs = int(leaves[0].n_additive_sublods)
    if rungs != EXPECTED_RUNGS:
        raise RuntimeError(
            f"{name}: merge produced {rungs} progressive rungs, expected {EXPECTED_RUNGS}"
        )
    return int(leaves[0].n_splats)


def _cull_tiles(fit_dir: Path, culled_dir: Path) -> int:
    """Redundancy-cull every per-timepoint tile into a parallel batch directory.

    The manifest is copied verbatim so ``batch-fit merge`` reads the same plan;
    only the tile stores differ. Culling per tile rather than after the merge is
    a memory constraint: the redundancy metric renders the splats to measure
    each one's local contribution, and doing that on the merged 4D result would
    mean reconstructing all 100 timepoints at once.
    """
    tiles = sorted((fit_dir / "tiles").glob("*.gsplats.zarr"))
    if not tiles:
        raise SystemExit(
            f"no tiles under {fit_dir / 'tiles'} -- the fit produced nothing to cull"
        )
    shutil.rmtree(culled_dir, ignore_errors=True)
    (culled_dir / "tiles").mkdir(parents=True)
    shutil.copy2(fit_dir / "manifest.json", culled_dir / "manifest.json")
    with asection(f"Culling {len(tiles)} tiles (threshold {REDUNDANCY_THRESHOLD})"):
        for tile in tiles:
            run_luxar_cli(
                "gsplat",
                "cull",
                str(tile),
                str(culled_dir / "tiles" / tile.name),
                "-m",
                "redundancy",
                "--shape",
                ",".join(str(v) for v in SOURCE_SHAPE[1:]),
                "--redundancy-threshold",
                str(REDUNDANCY_THRESHOLD),
            )
    return len(tiles)


def recompute_channel(channel: dict, source: Path, work_dir: Path) -> Path:
    """Refit one channel end to end and return its finished archive."""
    name = str(channel["name"])
    with asection(f"Recomputing the {name} channel"):
        bgsub = work_dir / f"{name}_bgsub.zarr"
        fit_dir = work_dir / f"{name}_fit"
        culled_dir = work_dir / f"{name}_fit_culled"
        final = work_dir / str(channel["file"])

        _subtract_background(source, bgsub, float(channel["background_floor"]))

        run_luxar_cli(
            "gsplat",
            "batch-fit",
            "run",
            str(bgsub),
            str(fit_dir),
            "--array-key",
            BGSUB_ARRAY_KEY,
            "--axes",
            SOURCE_AXES,
            "--tiling",
            "uniform",
            "--tile-size",
            str(TILE_SIZE),
            "--preset",
            PRESET,
            "--seeds",
            str(SEEDS),
            "--floor",
            "auto",
            # No fit-time cull: the redundancy cull below is the one that was
            # measured, and stacking a second criterion on top of it would make
            # the retained count depend on two thresholds instead of one.
            "--cull-retention",
            "0.0",
            "--jobs-per-gpu",
            str(JOBS_PER_GPU),
        )
        n_tiles = _cull_tiles(fit_dir, culled_dir)
        aprint(f"culled {n_tiles} tiles")

        # `batch-fit run` above already merged the UNCULLED tiles; that result is
        # discarded. Merging here is what produces the shipped ladder.
        run_luxar_cli(
            "gsplat",
            "batch-fit",
            "merge",
            str(culled_dir),
            "--recipe",
            "stream",
        )
        merged = culled_dir / "merged" / "final.gsplats.zarr"
        if not merged.exists():
            raise SystemExit(f"merge produced no {merged}")

        # Anisotropy + normalisation LAST: this is what takes the archive off the
        # voxel grid, and the fit's PSNR stamps are only comparable to the source
        # before it happens.
        run_luxar_cli(
            "gsplat",
            "transform",
            str(merged),
            str(final),
            "--scale",
            ",".join(str(v) for v in VOXEL_SCALE),
            "--normalize-intensity",
            str(NORMALIZE_INTENSITY),
        )

        node, _ = load_gsplat_node(str(final))
        got = _validate_rebuilt_channel(channel, list(iter_leaves(node)))
        expected = int(channel["expected_splats"])
        drift = abs(got - expected) / expected
        aprint(f"{name}: rebuilt {got:,} splats (recorded {expected:,}, {drift:.3%})")
        # A tolerance, not equality: nothing pins the reduction order of a
        # 12-worker parallel fit, so exact reproduction is not achievable. 1% is
        # far tighter than any recipe change and far looser than float noise.
        if drift >= 0.01:
            raise RuntimeError(
                f"{name}: rebuilt {got:,} splats against a recorded {expected:,} "
                f"({drift:.2%} drift). Over 1% means the recipe changed, not "
                f"float noise -- check the source array and the cull threshold."
            )
        return final


def recompute_channel_paths(work_dir: Path) -> list[Path]:
    """Refit both channels. Each is independent -- no shared fit, no shared cull."""
    missing = [
        str(ch["source_flag"]) for ch in CHANNELS if SOURCE_ARGS[ch["name"]] is None
    ]
    if missing:
        raise SystemExit(
            "--recompute needs a source array per channel: "
            + ", ".join(f"--{flag} PATH" for flag in missing)
            + ".\nEach is the channel's assembled (time, z, y, x) recording. The "
            "two channels were acquired and assembled separately — the membrane "
            "one ships as a single zipped array, the nuclei one was assembled "
            "from per-timepoint HPC files — so there is no single --source."
        )
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for channel in CHANNELS:
        source = SOURCE_ARGS[channel["name"]]
        assert source is not None  # guarded above
        source = source.expanduser()
        if not source.exists():
            raise FileNotFoundError(
                f"--{channel['source_flag']} does not exist: {source}"
            )
        paths.append(recompute_channel(channel, source, work_dir))
    return paths


def resolve_channel_paths() -> list[Path]:
    """Resolve the per-channel gsplat paths in the local store.

    Returns the list of existing ``.gsplats.zarr`` paths (channel order), or
    raises with an actionable message if the local store isn't populated — the
    files are pinned in the manifest but their Zenodo record is still an
    unpublished draft, so there is nothing to download yet.
    """
    paths = [DATA_DIR / ch["file"] for ch in CHANNELS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Neuromast gsplat data not found in the local store:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\n\nThis demo's fitted gsplats (~220 MB) are not bundled with the "
            f"repo, and their Zenodo record is still an unpublished draft, so "
            f"they cannot be fetched yet.\nPopulate {DATA_DIR} with the two "
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
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
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
                        # One global order slot per node cannot interleave the
                        # two co-located volumes. Near-zero kappa only hid that
                        # limitation; additive is order-independent.
                        blending_mode="additive",
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
            add_demo_caption(
                scene,
                "Light-sheet / iSIM • membranes + nuclei",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the channel data, build the 4D scene, and optionally serve it."""
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

    if RECOMPUTE:
        channel_paths = recompute_channel_paths(
            get_demos_output_dir() / "_neuromast_recompute"
        )
    else:
        channel_paths = resolve_channel_paths()
    scene_path = create_luxar_scene(channel_paths, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer… (press L for the Layers panel)")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
