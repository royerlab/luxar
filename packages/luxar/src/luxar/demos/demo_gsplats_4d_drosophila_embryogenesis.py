#!/usr/bin/env python
"""GSplats Demo: Drosophila Embryogenesis, 4h10m as Gaussian Splats.

The whole recording the ``gsplats_3d_drosophila_gastrulation`` demo takes a
single frame from: **500 timepoints at 30-second resolution**, a *Drosophila
melanogaster* embryo carried from the uniform blastoderm shell, through
gastrulation, into a fully extended germband. Every nucleus in the embryo is
labelled (``His2Av::mRFP1``), so what moves on screen is the morphogenesis
itself rather than a sparse subset of tracked cells.

One 4D Gaussian-splat node, time as a real axis of the data. The Time slider
walks it.

================================================================================
WHAT YOU ARE LOOKING AT
================================================================================

    t = 0 min      a uniform shell of nuclei in regular hexagonal packing —
                   the cellular blastoderm, before any body plan exists
    t = 75 min     gastrulation: the cephalic furrow cuts in near the anterior
                   third, the posterior midgut invaginates at the pole
    t = 100-175    germband extension — the trunk elongates and folds back over
                   itself along the dorsal side
    t = 175-250    fully extended germband; nuclei crowd and the surface
                   resolves into distinct segmental texture

The 30-second interval is the recording's own (Royer et al. 2016 imaged
*Drosophila* embryos "with 30-s temporal resolution throughout development").
It is stated in the paper rather than recorded in the array metadata — the
source zarr carries no interval and every raw frame file shares one mtime — but
it is corroborated by the biology: at 30 s/frame the cephalic furrow appears at
t = 150, which is 75 minutes after a syncytial-blastoderm t = 0, and that is
when it appears in a real embryo.

================================================================================
WHY THIS IS ONE NODE AND NOT 500
================================================================================

Every timepoint is fitted on its own — that is what keeps each fit small and
separately schedulable across a GPU — and the fits are then stacked into ONE 4D
``GSplatData`` whose fourth centre column is time, with ``sigma = 0`` on that
axis so a splat is instantaneous and cannot smear into the next frame.

The payoff is not tidiness, it is streaming. The writer treats a categorical
centre column as an ordering BARRIER: splats are lexsorted by time first and the
Hilbert curve is computed over the three spatial columns only, so a chunk never
straddles two timepoints. The viewer's spatial index then fetches only the
chunks whose bounds intersect the current slice.

Measured in a real browser on the shipped 500-frame store on 2026-08-30,
advancing one timepoint costs **0-2.3 MB in 0-4 requests**. First paint does not
share that locality: it costs **47 MB in 115 requests** for a timepoint that
needs 3.16 MB, a 15x read amplification from the 1 MB archive chunks. At this
length every coarse LOD level holds all 500 timepoints in one chunk, so drawing
one frame downloads the whole level. An earlier 20-frame pilot measured about
2 MB to first paint; that result does not survive the 25x scale-up. See #2374.

A consequence worth knowing: the writer warns that this node holds more splats
than a single gsplats node can render on a 4096-class GPU. That warning is
expected here and not a defect. It is about the RESIDENT set, and an nD node
sliced on a hidden dimension commits only its current slice — roughly 160k
splats, some 25x under the cap.

================================================================================
THE TIME AXIS IS A TRAP, AND THIS IS HOW IT IS DISARMED
================================================================================

Time is a DISCRETE viewer dimension: navigation snaps to ``t_min + k*step``, and
a chunk is only fetched within a QUARTER STEP of the snapped position. A
timepoint sitting off that grid produces a slider stop that renders nothing at
all, silently.

So the axis is in minutes with ``step = 0.5``, which is binary-exact — 30
seconds is exactly representable, unlike the 2.0001470947-minute interval that
forced the zebrafish demo to round. And the build asserts every stored timepoint
lands on the grid rather than trusting that it does.

================================================================================
CULLING: WHY THE 2026-08 ARCHIVE'S AMPLITUDE CUTOFF WAS DROPPED (2026-09)
================================================================================

The fit uses a nominal seed budget of 256,000 splats per frame — a ceiling of
128 million in total — and the merge measured exactly 128,000,000. The 2026-08
archive then kept only 83,221,420 of them: a hard raw-amplitude floor at
33.40462112 (originally intended as cumulative-amplitude retention 0.960; the
pre-#2260 float32 accumulator kept 12.4 %, the corrected float64 one 55.3 %,
and the shipped root stats record the fixed cutoff instead). It was validated
by flicker test on the companion ``gsplats_3d_culling_study`` demo.

That validation was blind to what the cull cost. Scored per frame against the
raw recording (manuscript SD14, five frames), the culled archive sat 5-8 dB
below a fresh fit of the same frames, and an uncelled tile of the batch fit
scored identically to the fresh fit (39.45 vs 39.46 dB): the WHOLE gap was the
cull. The dim tail is haze in a flicker test but signal in a PSNR, so the
recipe now ships the uncelled merge. The archive is correspondingly larger.

WHERE THE DATA COMES FROM
================================================================================

This demo downloads **one fitted archive** and builds a scene from it. That
archive is the only input, and nothing else is needed to run the demo:

    gsplats_4d_drosophila_embryogenesis  ->  drosophila_embryogenesis_500tp.gsplats.zarr.zip
    824 MB, 83,221,420 splats, 500 timepoints
    Zenodo record 10.5281/zenodo.22118695

THE RAW IMAGERY IS NOT PUBLICLY DEPOSITED, and that is worth stating plainly
rather than leaving as a gap. The source is a 500-timepoint SiMView light-sheet
recording, ``(500, 108, 1352, 532)`` uint16 — 19.3 GB compressed, 77.7 GB
decoded — held by the Royer and Keller labs. No public archive holds it; it is
why Philipp Keller is credited as a data collector on the Zenodo record above
rather than the imagery being cited from a repository. To obtain it, approach
the authors of the publication cited below.

The consequence for reproducibility is honest but limited: **the fit cannot be
re-run from scratch by a stranger**, only by someone who already holds the raw
recording. The published archive is therefore the reproducible artifact, and the
exact commands that produced it are recorded below so that anyone who does hold
the raw data gets the same result.

DATA SOURCE & CITATIONS:
    Keller lab, HHMI Janelia Research Campus (L. A. Royer was then a postdoctoral
    fellow there). Imaged on a SiMView multi-view light-sheet microscope
    under the AutoPilot adaptive framework:

      Royer, L.A., Lemon, W.C., Chhetri, R.K., Wan, Y., Coleman, M., Myers, E.W.
      & Keller, P.J. "Adaptive light-sheet microscopy for long-term,
      high-resolution imaging in living organisms."
      Nat. Biotechnol. 34, 1267-1278 (2016). doi:10.1038/nbt.3708

    Specimen: ``w; His2Av::mRFP1; +`` (Bloomington stock #23560).

    The single-frame ``gsplats_3d_drosophila_gastrulation`` demo is frame 150 of
    this same recording, fitted independently and published on a different record
    (10.5281/zenodo.21912280). Same specimen, same instrument, same acquisition.

VOXEL CALIBRATION:
    Identical to the single-frame demo, and derived the same way. Lateral pixel
    size follows from the instrument (Nikon 16x/0.8 onto Hamamatsu Orca Flash
    4.0, 6.5 um pitch): 6.5/16 = **0.40625 um/px**. The axial step is not in the
    metadata and was measured from the embryo's own geometry — a prolate
    ellipsoid's mid-length cross-section must be circular, which gives
    **1.93 um**. Cross-checks: the embryo measures ~521 x 190 um against a
    textbook ~500 x 180, and the two cross-sectional axes agree to 2.5%.

REPRODUCING THE ARCHIVE FROM THE RAW RECORDING
================================================================================

Every step is a stock ``luxar`` command; there are no private scripts. Given
``DrosophilaHistone.zarr.zip`` (array key ``data``), on one CUDA GPU:

    # 1. Fit all 500 timepoints. One tile per volume (tile-size >= max spatial
    #    extent), so the merge emits a SINGLE STACKED 4D LEAF rather than a
    #    partition. ~9.3 h at 1.12 min/timepoint on an RTX PRO 6000.
    luxar gsplat batch-fit run DrosophilaHistone.zarr.zip out/ \
        --array-key data --axes time,z,y,x \
        --tiling uniform --tile-size 1400 --overlap 0 \
        --preset standard --iters 1500 --seeds 256000 --floor 8 \
        --gpus 0 --jobs-per-gpu 2 \
        --merge-recipe stream --merge-target-ms 200

    # 2. Physical microns, straight from the uncelled merge (no amplitude cull,
    #    see CULLING above). Time is left as an INTEGER FRAME INDEX on purpose —
    #    see `normalise_time_axis` below for why a scaled axis does not survive
    #    the centres encoder.
    luxar gsplat transform out/merged/final.gsplats.zarr um.gsplats.zarr \
        --scale 1.93,0.40625,0.40625,1

    # 3. Re-chunk for streaming. Measured on this store: 173 -> about 3 requests
    #    per timepoint step, and a 7.7% smaller zip. The tradeoff at 500 frames
    #    is 47 MB / 115 requests for first paint; `--profile hosting` measures
    #    18 MB / 68 requests instead, but about 12 requests per timepoint step.
    luxar optimise um.gsplats.zarr \
        drosophila_embryogenesis_500tp.gsplats.zarr.zip --profile archive

Three choices in there are measured rather than conventional:

``--iters 1500`` — 5,000 iterations scores identically on this data (41.8 /
37.2 dB foreground either way; the loss is flat past ~1,000). Iterations are not
the lever; seeding is, at 57-70 s of a ~115 s timepoint.

``--floor 8`` and the merge's single global normalisation range are resolved
ONCE across all 500 frames rather than per timepoint. That is what keeps the
exposure from drifting as the embryo brightens, and it is why the appearance
constants below are portable across the whole recording.

``--merge-target-ms 200`` is sized per DISPLAYED TIME SLICE since #2374/#2376:
the merge multiplies the download budget by the 500 slices it observes, so rung 0
holds ~10.4 M splats (~20,800 per timepoint, ~200 ms of the visible frame) and
the geometric ladder has 5 rungs. The 2026-08 archive was laddered before that
fix: the same flag sized the budget against the whole node, its first rung held a
median 45 splats per timepoint (p05 7; one timepoint empty) and the ladder had 14
rungs. ``EXPECTED_RUNGS`` pins the corrected 5-rung ladder; a 14-rung result
means the old, starved sizing came back.

The fit directory is preserved across recompute attempts so ``batch-fit run`` can
resume completed timepoints. Only the derived filter, transform, and archive
outputs are replaced on a retry. Delete ``fit/`` manually after changing fit
constants such as ``ITERS``; otherwise the default resume path can reuse the old
tiles and merged output.


USAGE:
    python demo_gsplats_4d_drosophila_embryogenesis.py [--no-serve] [--serve-only]
    python demo_gsplats_4d_drosophila_embryogenesis.py --recompute \
        --source /path/to/DrosophilaHistone.zarr.zip

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_4d_drosophila_embryogenesis.luxar.zarr
    - Press 1 then [ / ] to walk Time; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_4d_drosophila_embryogenesis",
    "title": "4D Drosophila Embryogenesis",
    "description": (
        "A Drosophila embryo from blastoderm through gastrulation to germband "
        "extension — 500 timepoints at 30 s, as one 4D Gaussian-splat node."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        # The largest payload in the corpus by some margin: 863,811,020 bytes
        # MEASURED on the shipped artifact, not projected. A one-time download,
        # after which advancing one timepoint costs 0-2.3 MB in 0-4 requests
        # (the store is re-chunked with `optimise --profile archive`, without
        # which it would be 173 requests per step).
        "download_mb": 824,
        # Not a fit — but the scene build loads ~82M splats into memory
        # (~3.6 GB) to stack and write them, which is not a laptop-idle task.
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_4d_drosophila_embryogenesis"],
    "outputs": ["gsplats_4d_drosophila_embryogenesis"],
    "citation": {
        "short": "Royer et al. 2016",
        "doi": "10.1038/nbt.3708",
        "license": "CC BY 4.0",
    },
}

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
    parse_path_arg,
    run_luxar_cli,
    stamp_input_digests,
)
from luxar.demos._cinematic_camera import VIEWER_DEFAULT_FOV_DEG, pull_in
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds, is_matrix_shaped, iter_leaves
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_4d_drosophila_embryogenesis"
SCENE_NAME = "gsplats_4d_drosophila_embryogenesis.luxar.zarr"

#: Acquisition interval, in minutes. 30 s exactly — and 0.5 is binary-exact, so
#: the discrete Time axis's stops land where the range input can reach them.
#: See the docstring: an off-grid timepoint renders nothing, silently.
FRAME_INTERVAL_MIN = 0.5

#: Appearance, tuned live in the Layers panel on the hosted render
#: (2026-09-10). The non-zero window floor lifts residual haze and buys
#: contrast; the window top came down from 1.153 to 0.868 (brighter nuclei) and
#: absorption went UP from 0.33 to 1.19 so the near nuclei screen the far ones
#: and the embryo reads as a solid rather than a haze — the two move together
#: (a higher kappa dims, the tighter window compensates). The window is stored
#: as an intensity/offset PAIR, not a gain: ``intensity = 1/(hi-lo)``,
#: ``offset = -lo/(hi-lo)``.
DISPLAY_WINDOW = (0.059, 0.868)
GSPLAT_OPACITY = 0.20
GSPLAT_ABSORPTION = 1.19

#: Share of the half-frame the embryo's long axis subtends at the opening pose.
CAMERA_FRAME_FILL = 0.62

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: ``--source PATH`` — the unpublished raw recording. The file name was checked
#: against the copy on the acquisition box rather than inferred from the parent
#: directory name ``DrosophilaHistoneRaw``.
SOURCE_ARG = parse_path_arg("source")

# ---- Recorded fit recipe ----------------------------------------------------
#: Verified source name and group member on the acquisition box.
SOURCE_FILENAME = "DrosophilaHistone.zarr.zip"
SOURCE_ARRAY_KEY = "data"
#: Deliberately differs from the manifest-pinned download name: a local rebuild
#: must never squat the checksum-owned fetch path or be quarantined next launch.
REBUILT_FILENAME = "drosophila_embryogenesis_500tp_rebuilt.gsplats.zarr.zip"
#: Full recording contract checked before a roughly 9.3-hour fit starts.
SOURCE_SHAPE = (500, 108, 1352, 532)
SOURCE_DTYPE = np.dtype("uint16")
#: Explicit because a wrong time-axis label silently fits time as space.
SOURCE_AXES = "time,z,y,x"
#: Larger than every spatial extent, keeping each timepoint in one seam-free tile.
TILE_SIZE = 1400
OVERLAP = 0
#: The measured run converged by 1,500 iterations; 5,000 scored identically.
PRESET = "standard"
ITERS = 1500
#: Per-timepoint operating point chosen from the calibration sweep. Because
#: ``TILE_SIZE`` exceeds every spatial extent, each task has one tile and the
#: whole-volume seed budget is not divided across tiles.
SEEDS = 256_000
#: One global fixed floor keeps exposure comparable across all 500 frames.
FLOOR = 8
#: Recorded acquisition-box scheduling; two workers kept the GPU occupied.
GPUS = "0"
JOBS_PER_GPU = 2
#: Progressive first-paint ladder produced during the streaming merge (sized
#: per time slice, see the docstring; 5 rungs on this recording).
MERGE_RECIPE = "stream"
MERGE_TARGET_MS = 200
#: Physical (z, y, x) microns; the stacked frame-index axis is unchanged here.
VOXEL_SCALE = (1.93, 0.40625, 0.40625, 1.0)
#: One-megabyte chunks measured at 0-2.3 MB in 0-4 requests per timepoint step.
CHUNK_PROFILE = "archive"
#: Nominal whole-recording seed budget; a refit may drift as dynamic ops run.
NOMINAL_FITTED_SPLATS = SOURCE_SHAPE[0] * SEEDS
EXPECTED_RUNGS = 5
#: Parallel fit reductions can move a threshold count, but not by recipe scale.
SPLAT_COUNT_TOLERANCE = 0.01


# =============================================================================
# Recompute recipe
# =============================================================================
@contextmanager
def _open_source(path: Path) -> Iterator[Any]:
    """Yield the source root and close a zip store after validation.

    zarr-python 3 does not infer :class:`ZipStore` from a ``.zip`` suffix. The
    explicit branch is required for the real 19.3 GB source; opening its path
    directly otherwise raises ``GroupNotFoundError`` for a valid recording.
    Unlike the neuromast opener, this yields the root group so the named source
    array and the available-array diagnostics can both be validated here.
    """
    import zarr

    store = None
    try:
        if path.suffix == ".zip":
            from zarr.storage import ZipStore

            try:
                store = ZipStore(str(path), mode="r")
                root = zarr.open(store=store, mode="r")
            except Exception as exc:  # noqa: BLE001 - normalize storage errors
                raise ValueError(
                    f"Could not read {path.name} as a Zarr recording. Expected "
                    f"the DrosophilaHistone recording. Underlying error: {exc}"
                ) from exc
        else:
            try:
                root = zarr.open(str(path), mode="r")
            except Exception as exc:  # noqa: BLE001 - normalize zarr errors
                raise ValueError(
                    f"Could not read {path.name} as a Zarr recording. Expected "
                    f"the DrosophilaHistone recording. Underlying error: {exc}"
                ) from exc
        yield root
    finally:
        if store is not None:
            store.close()


def _validate_source(path: Path) -> None:
    """Refuse a wrong or partial recording before starting the GPU fit."""
    with _open_source(path) as root:
        try:
            array_keys = tuple(root.array_keys())
            group_keys = tuple(root.group_keys())
        except Exception as exc:  # noqa: BLE001 - a bare array has no array_keys
            available = "(this store is a bare array, not a group)"
            raise ValueError(
                f"{path.name} has no {SOURCE_ARRAY_KEY!r} array; found: {available}. "
                f"Expected the DrosophilaHistone recording, a group whose "
                f"{SOURCE_ARRAY_KEY!r} member is (time, z, y, x). Underlying error: "
                f"{exc}"
            ) from exc
        if SOURCE_ARRAY_KEY not in array_keys:
            arrays = ", ".join(sorted(array_keys)) or "(none)"
            groups = ", ".join(sorted(group_keys)) or "(none)"
            raise ValueError(
                f"{path.name} has no {SOURCE_ARRAY_KEY!r} array; arrays: {arrays}; "
                f"groups: {groups}. "
                f"Expected the DrosophilaHistone recording, a group whose "
                f"{SOURCE_ARRAY_KEY!r} member is (time, z, y, x)."
            )
        array = root[SOURCE_ARRAY_KEY]

        shape = tuple(array.shape)
        dtype = np.dtype(array.dtype)
        aprint(f"source array [{SOURCE_ARRAY_KEY}]: {shape} {dtype}")
        if shape != SOURCE_SHAPE:
            raise ValueError(
                f"{path.name} has shape {shape}, expected {SOURCE_SHAPE}. Refusing "
                f"to fit a wrong or partial recording."
            )
        if dtype != SOURCE_DTYPE:
            raise ValueError(
                f"{path.name} has dtype {dtype}, expected {SOURCE_DTYPE}. Refusing "
                f"to fit a converted source with a different intensity contract."
            )
    aprint(f"source validated: {SOURCE_SHAPE} {SOURCE_DTYPE}")


def _only_leaf(node: Any, label: str) -> Any:
    """Return the sole leaf or reject an unexpected tree topology."""
    leaves = list(iter_leaves(node))
    if len(leaves) != 1:
        raise RuntimeError(f"{label} produced {len(leaves)} leaves, expected one")
    return leaves[0]


def _validate_rebuilt_archive(node: Any) -> int:
    """Validate the merged leaf count and progressive-rung contract."""
    leaf = _only_leaf(node, "rebuild")
    rungs = int(leaf.n_additive_sublods)
    if rungs != EXPECTED_RUNGS:
        raise RuntimeError(
            f"rebuild produced {rungs} progressive rungs, expected {EXPECTED_RUNGS}"
        )
    return int(leaf.n_splats)


def _validate_fitted_splat_count(node: Any) -> None:
    """Reject an incomplete or misconfigured merged fit before transforming it."""
    got = int(_only_leaf(node, "fitted merge").n_splats)
    drift = abs(got - NOMINAL_FITTED_SPLATS) / NOMINAL_FITTED_SPLATS
    if drift >= SPLAT_COUNT_TOLERANCE:
        raise RuntimeError(
            f"fitted merge produced {got:,} splats against the nominal "
            f"{NOMINAL_FITTED_SPLATS:,}-splat seed budget ({drift:.2%} drift). "
            f"Over 1% means the fit recipe changed -- check the source, "
            f"timepoints, and fit settings."
        )
    aprint(
        f"fitted merge validated: {got:,} splats "
        f"(nominal {NOMINAL_FITTED_SPLATS:,}, {drift:.3%} drift)"
    )


def recompute_archive(work_dir: Path) -> Path:
    """Refit, filter, physically scale, and re-chunk the 500-frame archive."""
    with asection("Recomputing the Drosophila embryogenesis archive"):
        if SOURCE_ARG is None:
            raise SystemExit(
                "--recompute needs --source PATH pointing at the unpublished raw "
                f"recording ({SOURCE_FILENAME})."
            )
        source = SOURCE_ARG.expanduser()
        if not source.exists():
            raise FileNotFoundError(f"--source does not exist: {source}")
        _validate_source(source)

        work_dir.mkdir(parents=True, exist_ok=True)
        fit = work_dir / "fit"
        culled = (
            work_dir / "culled.gsplats.zarr"
        )  # legacy intermediate, removed if present
        scaled = work_dir / "um.gsplats.zarr"
        final = work_dir / REBUILT_FILENAME
        for derived in (culled, scaled, final):
            if derived.is_dir():
                shutil.rmtree(derived)
            else:
                derived.unlink(missing_ok=True)

        run_luxar_cli(
            "gsplat",
            "batch-fit",
            "run",
            str(source),
            str(fit),
            "--array-key",
            SOURCE_ARRAY_KEY,
            "--axes",
            SOURCE_AXES,
            "--tiling",
            "uniform",
            "--tile-size",
            str(TILE_SIZE),
            "--overlap",
            str(OVERLAP),
            "--preset",
            PRESET,
            "--iters",
            str(ITERS),
            "--seeds",
            str(SEEDS),
            "--floor",
            str(FLOOR),
            "--gpus",
            GPUS,
            "--jobs-per-gpu",
            str(JOBS_PER_GPU),
            "--merge-recipe",
            MERGE_RECIPE,
            "--merge-target-ms",
            str(MERGE_TARGET_MS),
        )
        merged = fit / "merged" / "final.gsplats.zarr"
        fitted_node, _ = load_gsplat_node(merged)
        _validate_fitted_splat_count(fitted_node)
        del fitted_node
        # No post-fit amplitude cull (2026-09): the shipped 2026-08 archive was the
        # fit filtered at AMPLITUDE_MIN (65 % of the splats removed) and scored 5-8 dB
        # below a fresh fit of the same frames; the uncelled merge IS the recipe.
        run_luxar_cli(
            "gsplat",
            "transform",
            str(merged),
            str(scaled),
            "--scale",
            ",".join(f"{value:g}" for value in VOXEL_SCALE),
        )
        run_luxar_cli(
            "optimise",
            str(scaled),
            str(final),
            "--profile",
            CHUNK_PROFILE,
        )

        node, _ = load_gsplat_node(str(final))
        got = _validate_rebuilt_archive(node)
        aprint(
            f"rebuilt {got:,} splats (uncelled recipe; nominal {NOMINAL_FITTED_SPLATS:,})"
        )
        aprint(f"rebuilt: {final}")
        return final


# =============================================================================
# Scene construction
# =============================================================================
def resolve_data() -> Path:
    """Resolve the fitted 4D archive: cache -> in-repo -> Zenodo."""
    with asection("Resolving the Drosophila timelapse gsplats"):
        paths = ensure_dataset(DATASET)
        aprint(f"Data: {paths[0]}")
        return paths[0]


def normalise_time_axis(node) -> tuple[float, float, int]:
    """Bring the stored time column to MINUTES, whichever form it arrives in.

    Two forms are accepted, because the centres encoder is not consistent about
    which one survives a write (#2259):

    * **frame indices** ``0, 1, ... N-1``   -> multiplied by the interval here
    * **minutes**       ``0, s, ... (N-1)s`` -> already correct, left alone

    Anything else raises. This is deliberately not a heuristic with a fallback:
    both forms are exactly recognisable, and a third would mean the archive is
    not what this demo thinks it is.

    Why it has to cope with both. The archive is written with an integer frame
    index on purpose — centres are quantised to per-axis uint16, and for the
    stacked axis the encoder sometimes writes an IDENTITY mapping over the raw
    uint16 range (measured: ``col_hi = 65535.0``), which silently discards
    whatever real-world scale the column carried. An integer index survives that
    exactly. But it does not ALWAYS discard it: two archives from the same
    command chain, differing only in splat count, came back one as ``0..499``
    (indices, scale lost) and one as ``0..249.5`` (minutes, scale kept). So the
    demo cannot assume either, and guessing wrong mislabels the axis by 2x with
    no error anywhere. Hence: recognise both, refuse the rest.
    """
    from luxar.gsplats.tree import iter_leaves

    subs = [s for leaf in iter_leaves(node) for s in leaf.additive_sublods]
    stored = np.unique(np.concatenate([np.asarray(s.centers)[:, 3] for s in subs]))
    n = stored.size

    as_frames = np.arange(n, dtype=np.float64)
    as_minutes = as_frames * FRAME_INTERVAL_MIN

    if np.allclose(stored, as_frames, atol=1e-6):
        scale = FRAME_INTERVAL_MIN
        aprint(
            f"Time column: frame indices 0..{n - 1} -> scaling by "
            f"{FRAME_INTERVAL_MIN} min/frame"
        )
    elif np.allclose(stored, as_minutes, atol=1e-6):
        scale = 1.0
        aprint(f"Time column: already minutes 0..{as_minutes[-1]:g}")
    else:
        raise RuntimeError(
            f"the stored time column is neither consecutive frame indices "
            f"0..{n - 1} nor minutes 0..{as_minutes[-1]:g}: got min={stored.min()} "
            f"max={stored.max()} distinct={n} "
            f"step(s)={np.unique(np.round(np.diff(stored), 6))[:4]}. Refusing to "
            f"guess — mislabelling this axis is silent and doubles the stated "
            f"duration."
        )

    if scale != 1.0:
        for s in subs:
            cen = s.centers
            if not cen.flags.writeable:
                cen = np.array(cen, copy=True)
                object.__setattr__(s, "centers", cen)
            cen[:, 3] *= np.float32(scale)

    lo, hi = 0.0, float((n - 1) * FRAME_INTERVAL_MIN)
    # Stops must land on `lo + k*step` or the viewer fetches nothing within a
    # quarter-step of them, silently. Integer index x 0.5 is binary-exact.
    assert np.array_equal(np.arange(n) * FRAME_INTERVAL_MIN, np.linspace(lo, hi, n)), (
        "time axis off-grid"
    )
    return lo, hi, n


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 4D scene from the pre-fitted, culled, physically-scaled data."""
    with asection("Creating the Drosophila embryogenesis scene"):
        node, _ = load_gsplat_node(str(data_path))

        # Centre columns are (Z, Y, X, Time) — the fit's own order, with the
        # stacked axis last. Time arrives as frame indices or minutes depending
        # on how the centres encoder quantised the stacked axis;
        # normalise_time_axis recognises both.
        t_lo, t_hi, n_t = normalise_time_axis(node)
        bmin, bmax = center_bounds(node)
        aprint(
            f"Time: {n_t} stops, {t_lo:g}-{t_hi:g} min "
            f"({(t_hi - t_lo) / 60:.2f} h) at {FRAME_INTERVAL_MIN * 60:g} s"
        )
        aprint(
            f"Embryo: {bmax[1] - bmin[1]:.0f} um long, {bmax[2] - bmin[2]:.0f} um wide"
        )

        centre = tuple(float(v) for v in (bmin[:3] + bmax[:3]) / 2.0)
        half_len = float(bmax[1] - bmin[1]) / 2.0
        half_fov = np.radians(VIEWER_DEFAULT_FOV_DEG / 2.0)
        cam_dist = (half_len / CAMERA_FRAME_FILL) / float(np.tan(half_fov))
        cam_eye = (centre[0], centre[1], centre[2] + cam_dist)

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
                    unit="min",
                    display=False,
                    discrete=True,
                    range=(t_lo, t_hi),
                    step=FRAME_INTERVAL_MIN,
                ),
            ]
        )

        lo, hi = DISPLAY_WINDOW
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                citation=DEMO_META["citation"],
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    exposure=0.0,
                    camera=CameraConfig(
                        position=pull_in(cam_eye, centre), target=centre
                    ),
                ),
            )
            stamp_input_digests(scene)
            scene.attrs["title"] = "GSplats: Drosophila Embryogenesis (SiMView)"
            scene.attrs["description"] = (
                "Drosophila melanogaster embryo (His2Av::mRFP1) over 4 h 10 min of "
                "development, from cellular blastoderm through gastrulation to a "
                "fully extended germband. 500 timepoints at 30 s, imaged on a "
                "SiMView adaptive light-sheet microscope and fitted as one 4D "
                "Gaussian-splat node. Press 1 then [ / ] to walk Time."
            )

            appearance = dict(
                blending_mode="volumetric",
                absorption=GSPLAT_ABSORPTION,
                opacity=GSPLAT_OPACITY,
                colormap="magma",
                intensity=1.0 / (hi - lo),
                offset=-lo / (hi - lo),
                gamma=1.0,
                layer=True,
                # normalize_amplitudes left at the default. The archive stores
                # raw detector counts; one insertion of one stacked node gives
                # ONE factor across every timepoint and every ladder rung, which
                # is exactly what keeps the exposure from stepping as you scrub.
            )
            with asection("Adding gsplats"):
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
                        scene, name="drosophila_nuclei", node=node, **appearance
                    )

            # House overlay style, matching the sibling 4D timelapses
            # (zebrafish, h2afva, neuromast, nexrad all use exactly this):
            # title at (0.02, 0.02), font 0.05, rgba(255,255,255,0.6), and
            # `difference` blending so it stays legible over both the dark
            # background and a bright specimen. The sub-lines are the house
            # mono block at 0.018 starting at y=0.10.
            scene.add_text(
                "Drosophila embryogenesis",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            # ONE add_text PER LINE: a "\n" inside an overlay string renders as
            # a single space, so a multi-line block written as one call comes
            # out as one run-on line. (The zebrafish demo has that bug.)
            for i, line in enumerate(
                (
                    "blastoderm → gastrulation → germband extension",
                    f"{n_t} timepoints · {FRAME_INTERVAL_MIN * 60:g} s apart "
                    f"· {(t_hi - t_lo) / 60:.2f} h",
                )
            ):
                scene.add_text(
                    line,
                    position=(0.02, 0.10 + i * 0.028),
                    font_size=0.018,
                    font="mono",
                    anchor="top-left",
                    color="rgba(255,255,255,0.55)",
                )
            add_demo_caption(
                scene,
                "SiMView light-sheet • His2Av::mRFP1",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Demo: Drosophila Embryogenesis (SiMView Light-Sheet)")
    aprint("=" * 70)
    aprint("500 timepoints • 4 h 10 min • blastoderm to extended germband")
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
        data_path = recompute_archive(
            get_demos_output_dir() / "_drosophila_embryogenesis_recompute"
        )
    else:
        data_path = resolve_data()
    scene_path = create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer…")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
