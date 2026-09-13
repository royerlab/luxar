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
    deconvolved, motion-aligned. Original volumes are under ``HPC_SOURCE_ROOT``
    in each channel's recorded ``hpc_source_dir``.

PIPELINE — reproducible per channel with ``--recompute``:
    1. To rebuild either channel, read its deconvolved TIFFs from the channel's
       ``hpc_source_dir`` using ``SOURCE_FILE_PATTERN`` and
       ``SOURCE_TIMEPOINT_LABELS``. Order ``t1`` through ``t100`` NUMERICALLY
       (not lexicographically), treat each TIFF as ``z,y,x``, and stack them
       into one ``time,z,y,x`` array. The nuclei array used for the recorded run
       was assembled this way; the membranes array in hand ships as a single
       zipped array. ``--source-*`` expects either assembled array, not the
       per-timepoint directory.
    2. Measure (and only record) the camera pedestal per channel: on the ten
       zero-based frames in ``CAMERA_PEDESTAL_SAMPLE_INDICES``, take the centre
       of the peak bin in a 512-bin histogram over values at or below that
       frame's 95th percentile, then take the median of those ten modes. The
       per-frame step matches ``estimate_floor(frame, method="mode")`` in
       ``luxar.gsplats.calibration.noise_floor``; that helper additionally
       excludes exact zeros and caps at the frame median, neither of which
       affected these frames. The recorded results are 105.991 (membranes) and
       103.888 (nuclei), kept in ``CHANNELS[...]["camera_pedestal"]``.
       The pedestal is NOT subtracted before the fit. Measured 2026-09-10 on
       frames 50 and 99: a fit of the exactly pedestal-subtracted, zero-clipped
       frame loses the low-contrast background modulation (actin ruffles at
       0.2-1 % of the frame maximum: correlation 0.16 / 0.12 against the frame,
       36 % / 65 % of the seeds dead), while the same fit of the raw frame keeps
       it (0.55 / 0.43, under 2 % dead) -- seeds that land on exactly-zero
       background have nothing to fit and die, and relaxing the amplitude
       sparsity changes nothing. The pedestal costs a constant haze that the
       display window removes; the recorded values are provenance, and the basis
       for any fidelity score (compare a raw-frame render with the raw frame).
    3. Calibrate K* per channel (Noise2Self blind-spot sweep) → K* = 64,000.
       Recorded, not re-run: the sweep is hours and its answer is stable.
    4. ``batch-fit run`` on the RAW assembled array: 100 timepoints, one uniform
       tile each, ``n2s`` preset, 64k seeds, ``--floor none`` (every frame's
       minimum is 0, so nothing is subtracted; see step 2), no fit-time cull.
    5. (Removed 2026-09.) The 2026-08 build redundancy-culled every tile at
       threshold 0.20, validated by SSIM. Per-frame PSNR against the raw frame
       showed the cull removing bright nuclear splats as "redundant" with the
       pedestal splats beneath them, at a 17-26 dB foreground cost; SSIM is
       blind to that (an empty render scores 0.82). No post-fit cull now.
    6. ``batch-fit run --merge-recipe stream --merge-n-lods 8`` merges the
       (unculled) tiles into ONE leaf with the recorded 8-rung progressive
       ladder (the run's own merge; the run's default would be 4 rungs).
    7. ``gsplat transform --scale 0.25,0.1083,0.1083,1.0
       --normalize-intensity 1.0`` → convert the three spatial centre and
       covariance axes from voxels to microns using the measured MetaMorph
       voxel size; amplitudes stay on a 0-1 scale.
       (The 2026-08 pair on the record was transformed with the historical 2.5
       — an 8.3 % Z stretch — and stays pinned in ``data_manifest.json``;
       the corrected rebuild is ready, but publishing it still needs the record
       upload and manifest repin, which this recipe change does not do.)

    Step 4 must be run with ``--jobs-per-gpu 12``, not ``auto``. On this box
    ``auto`` sized 100 concurrent workers for 100 tasks and every one of them
    was OOM-killed (``exit -9``) before a single tile landed — an
    over-subscribed GPU here fails all-at-once rather than degrading.

DATA STORAGE:
    These fitted gsplats are ~220 MB unzipped and are **not bundled with the
    repo**. Both channels live on the published ``cc-by`` Zenodo record as the
    130 MB ``.gsplats.zarr.zip`` pair, pinned by SHA-256 in
    ``demos/data_manifest.json``, so ``resolve_channel_paths`` fetches them on
    demand through ``ensure_dataset("gsplats_4d_neuromast_2ch")``: the pair is
    verified against those digests, cached under ``~/.cache/luxar/`` and
    expanded to a temporary directory on read. The ``.zip`` suffix is why
    fetched paths differ from each channel's ``file`` key, which names the
    *unzipped* store that ``--recompute`` and the local fallback use.

    ``DATA_DIR`` (below) remains that local fallback — the acquisition machine
    and any hand-placed copy, and the one way to run this demo from a manifest
    that cannot build a download URL for the record.

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
        "download_mb": 130,  # the zipped pair on the cc-by record
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_4d_neuromast_2ch"],
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
    DatasetUnavailable,
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
    parse_path_arg,
    run_luxar_cli,
    stamp_input_digests,
)
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds, iter_leaves
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

#: Manifest key for the hosted pair (``cc-by`` record, SHA-256 pinned).
DATASET_NAME = "gsplats_4d_neuromast_2ch"

# Optional local store of the UNZIPPED pair: the acquisition machine, a
# hand-placed copy, or a manifest that cannot resolve the record. The normal
# path is the fetch above. Override with $LUXAR_NEUROMAST_DATA_DIR.
DATA_DIR = Path(
    os.environ.get(
        "LUXAR_NEUROMAST_DATA_DIR",
        str(Path.home() / "luxar_demo_data" / "gsplats_neuromast_2ch"),
    )
)

HPC_SOURCE_ROOT = (
    "/hpc/projects/jacobo_group/Adrian/RU_Processed_Data/No_Ablations_Aligned/"
    "04192022_she_gfp_cldn_mscarlet_Timelapse3_3dpf/S1"
)
#: Applied inside each channel directory for every recorded timepoint label.
SOURCE_FILE_PATTERN = "*_t{timepoint}.tiff"
#: Axis order of every deconvolved per-timepoint TIFF before stacking.
SOURCE_FRAME_AXES = "z,y,x"
#: Ten evenly spaced zero-based frames (rounded linspace 0..99) used for pedestals.
CAMERA_PEDESTAL_SAMPLE_INDICES = (0, 11, 22, 33, 44, 55, 66, 77, 88, 99)
#: Per-frame mode measurement: histogram the low-intensity bulk through p95.
CAMERA_PEDESTAL_HISTOGRAM_PERCENTILE = 95.0
CAMERA_PEDESTAL_HISTOGRAM_BINS = 512
#: Combine the ten per-frame modes into the one pedestal pinned per channel.
CAMERA_PEDESTAL_REDUCTION = "median"

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
        # Display window (Layers-panel range) and gamma, set by eye on the pinned
        # historical-2.5, lateral-pixel-unit store (re-tuned 2026-09-10 under
        # ADDITIVE compositing, see the graft).
        # The window is authored as intensity/offset: intensity = 1 / (hi - lo),
        # offset = -lo / (hi - lo), which the viewer maps back to [lo, hi] on a
        # colormapped node. A gamma below 1 lifts the dim membrane shell.
        "window": (0.0, 1.719),
        "gamma": 0.71,
        # The enclosing structure, so it composites FIRST and the nuclei read on
        # top of it. This deliberately does NOT match the order the viewer would
        # infer: containment goes by bounding-sphere radius, and the hosted fit
        # gives the nuclei channel a 1.7 % larger sphere (a property of where the
        # splats landed, not of the anatomy), so the inference draws nuclei first.
        # A membrane shell
        # enclosing nuclei is the biology; state it.
        "layer_order": 10,
        # ---- recompute recipe, per channel ----
        #: ``--source-<name> PATH``: the assembled (time, z, y, x) array.
        "source_flag": "source-membranes",
        #: Durable upstream TIFF tree.
        "hpc_source_dir": f"{HPC_SOURCE_ROOT}/Membranes/Deconvolved",
        #: Camera pedestal, measured ONCE on this channel and recorded (step 2).
        #: Provenance and scoring basis only: it stays IN the data, because a fit
        #: of the pedestal-subtracted frame loses the faint background structure.
        "camera_pedestal": 105.9911880493164,
        #: What the recipe must reproduce. The unculled 2026-09 rebuild keeps every
        #: seed (64,000 x 100 frames); the 2026-08 redundancy-culled build had
        #: 5,864,440 ("Wrote single stream lod: 5,864,440 splats, 4D").
        "expected_splats": 6_400_000,
    },
    {
        "name": "nuclei",
        "file": "neuromast_nuclei.gsplats.zarr",
        "colormap": "bop_orange",  # GFP nuclei, iSIM 488/525
        "marker": "she:GFP (nuclei)",
        "opacity": 0.47,
        "window": (0.0, 0.459),
        "gamma": 0.69,
        #: Inside the membrane shell, so it composites last (on top).
        "layer_order": 20,
        "source_flag": "source-nuclei",
        "hpc_source_dir": f"{HPC_SOURCE_ROOT}/Nuclei/Deconvolved",
        "camera_pedestal": 103.88801574707031,
        # 2026-09 unculled rebuild: every seed kept (2026-08 culled build: 5,530,300).
        "expected_splats": 6_400_000,
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
#: One TIFF per one-based acquisition timepoint, stacked in this numeric order.
SOURCE_TIMEPOINT_LABELS = tuple(range(1, SOURCE_SHAPE[0] + 1))
#: Calibrated splat budget per timepoint (Noise2Self blind-spot sweep).
SEEDS = 64_000
#: Fitting preset. `n2s` is the noise-aware one that pairs with the calibration.
PRESET = "n2s"
#: One uniform tile per timepoint: 640 exceeds every spatial extent above, so
#: the volume stays whole and there are no tile seams to apodize.
TILE_SIZE = 640
#: Concurrent fit workers per GPU. NOT `auto` — see the docstring's warning.
JOBS_PER_GPU = 12
#: Physical voxel size in fitted-axis order (Z, Y, X, Time). The raw MetaMorph
#: headers and paper registry record voxel_um=(0.25, 0.1083, 0.1083); the
#: stacked time coordinate is already in frames and must not be scaled.
#: The 2026-08 pair on the record carries the historical Z-only 2.5 (step 7).
VOXEL_SCALE = (0.25, 0.1083, 0.1083, 1.0)
#: Amplitudes normalised to a unit peak, so appearance does not depend on the
#: recording's absolute intensity scale.
NORMALIZE_INTENSITY = 1.0

#: Progressive rungs the merge produced for each channel.
EXPECTED_RUNGS = 8

# ---- Appearance, shared by both channels -----------------------------------
#: Scene exposure in log2 stops (Rendering Controls > HDR > Exposure).
EXPOSURE_STOPS = -3.4


# =============================================================================
# Data loading
# =============================================================================
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


def _validate_source(src: Path) -> tuple[int, ...]:
    """Refuse a wrong-shaped or array-less source before the GPU is touched.

    The fit reads the RAW assembled array directly (docstring step 2: the camera
    pedestal is kept), so this is the only step between the source and
    ``batch-fit run`` and the only place a wrong file is caught cheaply.
    """
    with _open_source(src) as array:
        shape = tuple(getattr(array, "shape", ()) or ())
    if shape != SOURCE_SHAPE:
        raise SystemExit(
            f"{src.name} has shape {shape or 'no array at its root'}, want "
            f"{SOURCE_SHAPE}. Both channels of this recording share that extent, "
            f"so a mismatch means the wrong file or a partial assembly."
        )
    return shape


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


def recompute_channel(channel: dict, source: Path, work_dir: Path) -> Path:
    """Refit one channel end to end and return its finished archive."""
    name = str(channel["name"])
    with asection(f"Recomputing the {name} channel"):
        fit_dir = work_dir / f"{name}_fit"
        final = work_dir / str(channel["file"])

        _validate_source(source)

        # The RAW array, root-level (no --array-key): the pedestal stays in.
        run_luxar_cli(
            "gsplat",
            "batch-fit",
            "run",
            str(source),
            str(fit_dir),
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
            # Nothing is subtracted (docstring steps 2 and 4).
            "--floor",
            "none",
            # No fit-time cull (0 keeps every splat): the fit's own pruning of
            # zero-amplitude seeds is the only reduction; see docstring step 5 for
            # why the 2026-08 post-fit redundancy cull was dropped.
            "--cull-retention",
            "0.0",
            "--jobs-per-gpu",
            str(JOBS_PER_GPU),
            # The streaming ladder is built by the run's own merge; there is no
            # post-fit cull any more (docstring step 5).
            "--merge-recipe",
            "stream",
            # Pinned: `batch-fit run`'s merge defaults to a 4-rung ladder while the
            # 2026-08 archive (built by a standalone `batch-fit merge`) has 8; the
            # rung count is part of the recorded recipe and validated below.
            "--merge-n-lods",
            str(EXPECTED_RUNGS),
        )
        merged = fit_dir / "merged" / "final.gsplats.zarr"
        if not merged.exists():
            raise SystemExit(f"merge produced no {merged}")

        # Physical voxel scaling + normalisation LAST: the fit's PSNR stamps are
        # only comparable to the voxel-grid source before this conversion. The
        # uniform 0.1083x shrink relative to the anisotropy-corrected rebuild
        # leaves the amplitude-space window/gamma and post-multiply opacity
        # unchanged. Additive ray integration is world-scale dependent, however:
        # the micron rebuild is 9.23x dimmer (-3.21 stops). Keep -3.4 for the
        # pinned pixel-unit pair; the repin must render-check roughly -0.2.
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
                f"float noise -- check the source array."
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
    """Resolve the per-channel gsplats, in ``CHANNELS`` order.

    The manifest is the normal path: the record is published, so the archive
    pair is downloaded, SHA-256 verified and cached on first use. Only its
    specific ``DatasetUnavailable`` absence — a manifest that can build no
    download URL — falls back to the local store, which is what keeps the
    acquisition machine and hand-placed copies working.

    Paired by NAME rather than by position. ``ensure_dataset`` returns the
    manifest's file order, which happens to match ``CHANNELS`` today; relying on
    that would silently swap the two markers' colormaps and opacities if either
    list were ever reordered, and a swapped-channel render looks plausible.
    """
    try:
        fetched = {path.name: path for path in ensure_dataset(DATASET_NAME)}
    except DatasetUnavailable as unavailable:
        aprint(f"Manifest fetch unavailable ({unavailable}).")
        paths = [DATA_DIR / channel["file"] for channel in CHANNELS]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Neuromast gsplat data is unavailable from the manifest and "
                "missing from the local store:\n"
                + "\n".join(f"  - {path}" for path in missing)
                + f"\n\nPopulate {DATA_DIR} with the two `.gsplats.zarr` stores "
                "(or set $LUXAR_NEUROMAST_DATA_DIR to their location)."
            ) from unavailable
        return paths

    # ``file`` names the unzipped store that ``--recompute`` writes; the record
    # hosts it zipped.
    wanted = [f"{channel['file']}.zip" for channel in CHANNELS]
    missing = [name for name in wanted if name not in fetched]
    if missing:
        raise FileNotFoundError(
            f"{DATASET_NAME!r} did not provide: {missing}\n"
            f"It provided: {sorted(fetched)}\n"
            f"One archive per channel was expected: {wanted}\n"
            "The manifest entry and this demo's CHANNELS list have diverged."
        )
    return [fetched[name] for name in wanted]


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(channel_paths: list[Path], output_path: Path) -> Path:
    """Build the 4D two-channel scene: one layer-enabled gsplats node per marker.

    The gsplats are pre-fit 4D (``z, y, x, time``), already scaled to microns and
    intensity-normalised, so we simply graft each channel with its LUT and
    ``layer=True``. The pinned pair was scaled by the historical Z-only 2.5, not
    ``VOXEL_SCALE`` (step 7), so its spatial coordinates remain in lateral-pixel
    units and Z is also stretched by 8.3 %; a ``--recompute`` build has physical
    spatial units. Both channels co-register and animate over the Time dimension.
    """
    with asection("Creating 4D two-channel neuromast scene"):
        # Explicit, named 4D dims (not the generic dim0..dim3 from
        # build_dimensions_from_data). The fitted gsplat center columns are
        # ordered (Z, Y, X, Time) — Z first, from the fit's axes=time,z,y,x —
        # so the Dimensions list must follow that exact order. The three
        # spatial axes are displayed; Time is a DISCRETE (step=1) hidden axis
        # that drives the playback slider.
        channel_bounds = []
        for path in channel_paths:
            node, _ = load_gsplat_node(str(path))
            bounds = center_bounds(node)
            if bounds is None:
                raise ValueError(f"Channel has no centre bounds: {path}")
            channel_bounds.append(bounds)
        channel_mins = np.stack([bounds[0] for bounds in channel_bounds])
        channel_maxs = np.stack([bounds[1] for bounds in channel_bounds])
        bmin = np.min(channel_mins, axis=0)
        bmax = np.max(channel_maxs, axis=0)
        endpoint_delta = np.maximum(
            np.ptp(channel_mins, axis=0), np.ptp(channel_maxs, axis=0)
        )
        quantization_step = np.nextafter((bmax - bmin) / 65535, np.inf)
        divergent = endpoint_delta > quantization_step
        if np.any(divergent):
            axes = np.asarray(("Z", "Y", "X", "Time"))[divergent]
            raise ValueError(
                "Neuromast channels do not co-register within one uint16 "
                f"coordinate step on axes {', '.join(axes)}: "
                f"endpoint deltas={endpoint_delta[divergent]}, "
                f"tolerances={quantization_step[divergent]}"
            )
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
                # Exposure pulled down 3.4 stops (set by eye with the channel
                # windows): at the default camera distance the rosette core
                # otherwise saturates to white under both additive layers.
                viewer_config=ViewerConfig(
                    cinematic_mode=True, tone_mapping="ACES", exposure=EXPOSURE_STOPS
                ),
            )
            stamp_input_digests(scene)
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
                        # Cross-layer draw order, stated rather than inferred
                        # from bounding-sphere radii (see CHANNELS above). Kept
                        # under additive compositing (where order is moot) so a
                        # switch to a depth-sorted mode preserves anatomy.
                        layer_order=ch["layer_order"],
                        blending_mode="additive",
                        gamma=ch["gamma"],
                        intensity=1.0 / (ch["window"][1] - ch["window"][0]),
                        offset=-ch["window"][0] / (ch["window"][1] - ch["window"][0]),
                        layer=True,
                        colormap=ch["colormap"],
                    )
                    aprint(
                        f"  colormap={ch['colormap']} window={ch['window']} "
                        f"gamma={ch['gamma']} opacity={ch.get('opacity', 1.0)}"
                    )

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
