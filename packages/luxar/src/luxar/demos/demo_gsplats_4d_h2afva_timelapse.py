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
    The pinned archive is one leaf of 121,163,285 splats with a 12-rung
    progressive ladder. The SCENE does not graft that leaf as-is: at build time
    it is re-authored (no refit — the splats are untouched) into a
    ``kind=partition`` of **one part per timepoint**, 51 parts, each carrying
    its own capped progressive (stream) ladder: a 39,062-splat first rung
    (~200 ms at 25 Mbps), doubling, increments capped at 900,000. No
    substitutive levels anywhere.

    Why time parts (measured 2026-09-10 against the single sliced leaf and
    against 44 spatial parts, all at the same chunking):

      - A time step is exactly one part. Nothing from other timepoints is
        fetched, the part's first rung paints at once, and its ladder is sized
        against that timepoint alone — so no slice is ever starved. The single
        leaf's global ladder re-streamed from its bottom on every slice and left
        3-27% of the frame resident while a step loaded; spatial parts kept
        more on screen but paid ~30 MB per step because each part was still
        sliced by time.
      - The parts double as the frustum-independent grouping the viewer's
        per-node element cap wants; only one of them is ever resident.

    The re-authored partition is cached next to the download
    (``~/.cache/luxar/h2afva/51tp_timeparts_v<N>/``) and rebuilt when the
    archive or the recipe changes. Building it holds the whole leaf in memory
    once (~8 GB at 121M splats) and takes ~10 minutes of CPU.

    HISTORY: this archive USED to be a ``kind=partition`` of 44 SPATIAL parts,
    each an LOD group of four substitutive levels; both were dropped
    (2026-08) because a partition has no viewer-side selector and the
    substitutive levels were pure download weight for a node never seen
    whole. Time parts are a different thing: each is a complete frame, and the
    viewer's hidden-axis slice selects exactly one.

DATA SOURCE & CITATIONS:
    Royer lab, CZ Biohub San Francisco (zebrahub). Raw acquisition:
    ``h2afva/fused`` — 253 timepoints of 407 x 2048 x 2048 voxels, fused and
    deconvolved. Please cite the zebrahub resource when using this data.

WHICH TIMEPOINTS:
    Every fifth frame of the 253-timepoint recording — original indices
    0, 5, 10, ... 250 — giving 51 frames. If an archive records
    ``source_stride`` or ``source_timepoints``, scene creation cross-checks those
    attributes; the pinned archive does not retain them. Its stacked axis is
    renumbered 0..50 so the viewer's discrete navigation grid lands exactly on
    stored values.

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
    archive does not, and grafting cannot apply one — converting would mean a
    ``gsplat transform --scale 0.40625,0.40625,0.40625,1`` pass over the whole
    1.12 GB fit, which changes its bytes and therefore its published checksum.
    That is a data-side change, not a scene-authoring one, so the axes here are
    honest about being pixels. At the companion's calibration (0.40625 um
    laterally, 1.625 um axially) this envelope is 660 x 828 x 831 um.

PIPELINE — reproducible with ``--recompute``:
    1. Fit the full 253-timepoint ``h2afva/fused`` timelapse and scale it
       isotropic: ``h2afva_253tp.gsplats.zarr``, which this demo takes as its
       PARENT (``--parent PATH``) rather than refitting. That fit is 5.87 GB and
       hours of GPU; re-deriving the 51-frame variant from it is minutes of CPU.
    2. ``restride_stacked_axis(stride=5)`` — keep original timepoints
       0, 5, ... 250 and renumber them to 0..50, preserving the parent's tree
       shape. Not ``gsplat slice``: that takes ranges, not a stride, and refuses
       a partition.
    3. ``gsplat flatten`` — collapse to one leaf, keeping the finest level.
    4. ``gsplat lod --recipe stream --target-ms 200`` — put the progressive
       ladder back (flatten drops it), sized for a ~200 ms first paint.
    5. ``luxar optimise --profile archive`` — re-chunk. LAST, because step 4
       adds arrays that also want the 1 MB layout.

    Steps 2 and 3 are in that order for a memory reason, not a stylistic one:
    the parent is 602M splats at its finest level and flattening it whole peaked
    at 115 GB. Striding first cuts it to 121M before anything loads flat.

    The 51-frame variant is the manifest's default precisely because the
    253-frame one is 5.87 GB; this is the same recording at about a fifth of the
    download.

USAGE:
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --no-serve
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --serve-only
    python -m luxar.demos.demo_gsplats_4d_h2afva_timelapse --recompute \
        --parent /path/to/h2afva_253tp.gsplats.zarr
"""

DEMO_META = {
    "key": "gsplats_4d_h2afva_timelapse",
    "title": "4D Zebrafish Embryogenesis (h2afva timelapse)",
    "description": (
        "Zebrafish embryogenesis as a 4D Gaussian-splat timelapse: 51 timepoints "
        "of histone-labelled nuclei, 121M splats streamed progressively over a "
        "12-rung ladder."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 1064,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
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

import hashlib
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import read_node_attrs
from luxar.core.group.lod.group import (
    DEFAULT_LADDER_BYTES_PER_ELEMENT,
    DEFAULT_LADDER_TARGET_MS,
)
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    ensure_dataset,
    launch_viewer,
    parse_demo_flags,
    parse_path_arg,
    run_luxar_cli,
    stamp_input_digests,
)
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io._archive import read_archive_root_attrs
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.lod.additive import make_additive_lod
from luxar.gsplats.restride import restride_stacked_axis
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatPartition,
    center_bounds,
    iter_leaves,
)
from luxar.utils.lod_breakpoints import (
    DEFAULT_BANDWIDTH_MBPS,
    DEFAULT_MAX_ADDITIVE_COMMIT,
    capped_stream_cuts,
    streaming_chunk_splats,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "h2afva"

SCENE_NAME = "gsplats_4d_h2afva_timelapse.luxar.zarr"

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: ``--parent PATH`` — the 253-timepoint parent archive this variant is derived
#: from. Required for ``--recompute``. Not a raw microscope recording: the
#: expensive fit already happened, and re-deriving 51 frames from its output is
#: minutes of CPU rather than hours of GPU. It is also the only honest input,
#: since a fresh fit of the parent would not reproduce these bytes.
PARENT_ARG = parse_path_arg("parent")

#: Original-recording stride: one step of the Time axis is five acquisition
#: timepoints. Cross-checked against the archive before authoring the caption.
SOURCE_STRIDE = 5

# ---- Recorded re-authoring recipe (see the module docstring's PIPELINE) ------
#: Center column holding the stacked time axis. The fit puts spatial dims first
#: and the stacked axis LAST, so this is 3 for a (Z, Y, X, T) fit — NOT 0, which
#: is where a time-first source array would have it.
TIME_COL = 3
#: Frames the stride must yield out of the parent's 253.
EXPECTED_FRAMES = 51
#: Parent's recorded finest-level total. ``EXPECTED_SPLATS`` is sanity-checked
#: against it in the tests; the parent root exposes no finest-only count that can
#: be validated before the walk.
PARENT_FINEST_SPLATS = 602_580_152
#: What the recipe must reproduce, exactly: every step after the fit is
#: deterministic (a stride selection, a flatten that keeps the finest level, a
#: ladder built by a fixed rule), so unlike a refit this admits no drift
#: tolerance. A mismatch means the parent or the stride changed.
EXPECTED_SPLATS = 121_163_285
#: Progressive rungs recorded on the shipped one-leaf archive.
EXPECTED_RUNGS = 12
#: Progressive-ladder budget: the first rung is sized to roughly this many
#: milliseconds of download at the CLI's default assumed bandwidth.
LADDER_TARGET_MS = 200
#: Chunk profile. `archive` (1 MB) rather than `hosting` (256 KB) because this
#: node is read a whole timepoint at a time, and the measured cost of the
#: as-built layout was 173 requests per timepoint step against 2 after.
CHUNK_PROFILE = "archive"
#: Appearance authored into the archive root, matching what the shipped archive
#: carries. The scene re-states its own when grafting, because grafting carries
#: no root attrs -- so these govern only a direct `luxar gsplat view`, and the
#: demo's `blending_mode` below deliberately differs (see the graft call).
ARCHIVE_BLENDING_MODE = "volumetric"
ARCHIVE_ABSORPTION = 1.34

# ---- Scene-time re-authoring into time parts (variant B, 2026-09-10) --------
#: Bump when the time-part recipe changes (ladder rule, encoding, part layout);
#: the cache directory name carries it, so an old build is never reused.
TIME_PARTS_VERSION = 1
#: First rung of every part's ladder: the shared ~200 ms download budget
#: (39,062 splats at 25 Mbps and 16 B/splat). A time part is ONE frame, not a
#: sliced node, so the budget rule applies rather than the sliced share floor.
FIRST_RUNG_SPLATS = streaming_chunk_splats(
    DEFAULT_LADDER_TARGET_MS, DEFAULT_BANDWIDTH_MBPS, DEFAULT_LADDER_BYTES_PER_ELEMENT
)
#: Contribution ordering inside each part's ladder: the brightest nuclei first,
#: so the 1/e(k) brightening of an incomplete ladder shows a dimmer version of
#: the SAME frame rather than a random subset of it.
LADDER_METHOD = "self_energy"

# ---- Layer appearance (Loic's Layers-panel values, 2026-09-10, on variant B) --
#: DISPLAY RANGE 0.001 - 0.025 on a colormapped node is the LUT window, stored
#: as intensity = 1/(hi-lo) and offset = -lo/(hi-lo).
DISPLAY_WINDOW = (0.001, 0.025)
LAYER_GAMMA = 2.2
LAYER_OPACITY = 0.55
LAYER_ABSORPTION = 1.09
LAYER_BLENDING = "volumetric"
LAYER_COLORMAP = "plasma"


def _validate_rebuilt_archive(node: Any) -> int:
    """Require the recorded one-leaf, twelve-rung archive shape."""
    leaves = list(iter_leaves(node))
    if len(leaves) != 1:
        raise RuntimeError(f"rebuild produced {len(leaves)} leaves, expected one")
    rungs = int(leaves[0].n_additive_sublods)
    if rungs != EXPECTED_RUNGS:
        raise RuntimeError(
            f"rebuild produced {rungs} progressive rungs, expected {EXPECTED_RUNGS}"
        )
    return int(leaves[0].n_splats)


def _read_source_attrs(data_path: Path) -> dict:
    """Read source-root attrs from a directory or compressed archive."""
    if data_path.is_dir():
        return read_node_attrs(data_path) or {}
    return read_archive_root_attrs(data_path)


def _validate_parent(path: Path) -> None:
    """Assert ``--parent`` is the 253-timepoint archive this recipe expects.

    Worth doing up front: the strided slice walks the parent's entire tree, so a
    wrong ``--parent`` costs the better part of an hour before its splat count
    disagrees. The checks below reject the single-timepoint sibling, an already
    sliced variant, and a count-compatible parent missing the recorded Z scale.
    """
    attrs = _read_source_attrs(path)
    kind = attrs.get("kind")
    if kind != "partition":
        raise SystemExit(
            f"{path.name} declares kind={kind!r}; --parent must be the 253tp "
            f"partition archive. A flattened copy has already lost the "
            f"per-part structure this reproduces."
        )
    bounds = attrs.get("position_bounds") or {}
    maximum = bounds.get("max") or []
    stacked_max = maximum[TIME_COL] if len(maximum) > TIME_COL else None
    if stacked_max is None:
        raise SystemExit(
            f"{path.name} declares no position_bounds maximum for stacked column "
            f"{TIME_COL}, so its timepoint count cannot be checked. Refusing to "
            f"walk 602M splats on faith."
        )
    if float(maximum[0]) <= 1000:
        raise SystemExit(
            f"{path.name} has Z maximum {maximum[0]}, so it is missing the "
            f"recorded x4 Z anisotropy. Refusing to rebuild a geometrically "
            f"flattened embryo."
        )
    frames = int(round(float(stacked_max))) + 1
    # Check what the stride WILL yield, not the parent's exact length: 253
    # timepoints (0..252) and 251 both give 51 frames at stride 5, and pinning
    # one of them would reject the real archive. Rounding the bound first because
    # it is stored float32, so 252 reads back as 251.99998.
    yielded = -(-frames // SOURCE_STRIDE)
    if yielded != EXPECTED_FRAMES:
        raise SystemExit(
            f"{path.name} spans {frames} timepoints, which at stride "
            f"{SOURCE_STRIDE} yields {yielded} frames, not {EXPECTED_FRAMES}. "
            f"This is the check that catches the single-timepoint sibling fit "
            f"and the already-sliced 51-frame variant."
        )
    aprint(f"parent verified: kind=partition, {frames} timepoints")


def recompute_archive(work_dir: Path) -> Path:
    """Re-derive the 51-frame archive from its 253-frame parent.

    Four stages. The stride comes FIRST and the flatten second, which is a
    memory constraint rather than a preference: the parent holds 602M splats at
    its finest level and flattening it whole peaked at 115 GB, while striding
    first cuts it to 121M before anything is held flat.
    """
    with asection("Re-deriving the h2afva 51tp archive"):
        if PARENT_ARG is None:
            raise SystemExit(
                "--recompute needs --parent PATH pointing at the 253-timepoint "
                "archive (h2afva_253tp.gsplats.zarr). This variant is a strided "
                "slice of that fit, not an independent one — refitting the "
                "recording would not reproduce these bytes."
            )
        parent = PARENT_ARG.expanduser()
        if not parent.exists():
            raise FileNotFoundError(f"--parent does not exist: {parent}")
        if not parent.is_dir():
            raise SystemExit(
                f"--parent must be an unpacked directory store, got {parent.name}. "
                f"zarr-python 3 no longer sniffs the .zip suffix, and the walk "
                f"reads array by array — unzip it first."
            )
        _validate_parent(parent)

        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        strided = work_dir / "h2afva_51tp_strided.gsplats.zarr"
        flat = work_dir / "h2afva_51tp_flat.gsplats.zarr"
        laddered = work_dir / "h2afva_51tp_stream.gsplats.zarr"
        final = work_dir / "h2afva_51tp.gsplats.zarr"

        summary = restride_stacked_axis(
            parent,
            strided,
            stride=SOURCE_STRIDE,
            time_col=TIME_COL,
            # Appearance goes on the ROOT, which is the only node whose attrs a
            # caller can set -- the writer stamps leaves and groups from its own
            # key set. That is also where the shipped archive carries them, so a
            # bare `luxar gsplat view` of the result composites as intended.
            root_attrs={
                "blending_mode": ARCHIVE_BLENDING_MODE,
                "absorption": ARCHIVE_ABSORPTION,
            },
        )
        if summary["frames"] != EXPECTED_FRAMES:
            raise RuntimeError(
                f"stride {SOURCE_STRIDE} yielded {summary['frames']} frames, "
                f"want {EXPECTED_FRAMES}"
            )

        # Flatten drops the additive ladder along with the levels, so the ladder
        # is rebuilt after it — not before, or the flatten would discard it.
        run_luxar_cli("gsplat", "flatten", str(strided), str(flat))
        run_luxar_cli(
            "gsplat",
            "lod",
            str(flat),
            str(laddered),
            "--recipe",
            "stream",
            "--target-ms",
            str(LADDER_TARGET_MS),
        )
        # Re-chunk LAST: the ladder above adds arrays that also want the archive
        # layout, and leaving them as-built is the 173-requests-per-step case.
        run_luxar_cli(
            "optimise",
            str(laddered),
            str(final),
            "--profile",
            CHUNK_PROFILE,
        )

        node, _ = load_gsplat_node(str(final))
        got = _validate_rebuilt_archive(node)
        aprint(f"rebuilt {got:,} splats (recorded {EXPECTED_SPLATS:,})")
        if got != EXPECTED_SPLATS:
            # Exact, unlike the refit demos: every step here is deterministic
            # given the parent, so there is no reduction-order noise to absorb.
            raise RuntimeError(
                f"rebuilt {got:,} splats against a recorded {EXPECTED_SPLATS:,}. "
                f"Every step of this recipe is deterministic given --parent, so "
                f"this is not float noise: either the parent is a different "
                f"generation or the stride changed."
            )
        aprint(f"rebuilt: {final}")
        return final


def resolve_data() -> Path:
    """Resolve the fitted 51-timepoint gsplats: cache -> in-repo -> Zenodo."""
    with asection("Resolving h2afva 51tp gsplats"):
        paths = ensure_dataset(DATASET, variant="51tp")
        aprint(f"Data: {paths[0]}")
        return paths[0]


def window_attrs(window: tuple[float, float]) -> dict[str, float]:
    """The intensity/offset pair a colormapped Layers-panel window is stored as."""
    lo, hi = window
    return {"intensity": 1.0 / (hi - lo), "offset": -lo / (hi - lo)}


def _finest_data(node: Any) -> list[GSplatData]:
    """Finest-level content of every matrix-shaped subtree, as flat data.

    A leaf or a ``kind=lod`` group yields one entry (the group's finest level);
    a partition recurses. The pinned archive is a single leaf, the historical
    44-part archive a partition of lod groups — both are accepted.
    """
    if isinstance(node, (GSplatLeaf, GSplatLodGroup)):
        return [GSplatData.from_tree(node)]
    if isinstance(node, GSplatPartition):
        out: list[GSplatData] = []
        for child in node.children:
            out.extend(_finest_data(child))
        return out
    raise TypeError(f"unsupported gsplat node {type(node).__name__}")


def time_part_ladders(n_per_part: list[int]) -> list[list[int]]:
    """Cumulative ladder cuts per part: budget first rung, doubling, capped."""
    return [
        capped_stream_cuts(n, FIRST_RUNG_SPLATS, DEFAULT_MAX_ADDITIVE_COMMIT)
        for n in n_per_part
    ]


def time_parts_cache_dir(data_path: Path) -> Path:
    """Where the re-authored partition lives, keyed on the archive it came from."""
    stat = data_path.stat()
    key = hashlib.sha1(
        f"{data_path.name}|{stat.st_size}|{int(stat.st_mtime)}|"
        f"{TIME_PARTS_VERSION}|{FIRST_RUNG_SPLATS}|{LADDER_METHOD}".encode()
    ).hexdigest()[:10]
    return data_path.parent / f"51tp_timeparts_v{TIME_PARTS_VERSION}_{key}"


def build_time_parts(data_path: Path, out_path: Path) -> Path:
    """Re-author the archive as one laddered part per timepoint (no refit).

    Reads the finest content, splits it on the stacked time column, gives every
    part its own capped stream ladder and writes a ``kind=partition`` archive.
    Splat count is preserved exactly; only the grouping and the ladders change.
    """
    with asection("Re-authoring into time parts"):
        node, _ = load_gsplat_node(str(data_path))
        finest = _finest_data(node)
        centers = np.concatenate([np.asarray(d.centers) for d in finest], axis=0)
        amplitudes = np.concatenate([np.asarray(d.amplitudes) for d in finest])
        cholesky = np.concatenate(
            [np.asarray(d.cholesky_factors) for d in finest], axis=0
        )
        has_colors = all(d.colors is not None for d in finest)
        colors = (
            np.concatenate([np.asarray(d.colors) for d in finest], axis=0)
            if has_colors
            else None
        )
        truncation = float(finest[0].truncation_radius)
        total = int(centers.shape[0])
        del finest
        times = np.unique(centers[:, TIME_COL])
        aprint(f"{total:,} splats over {times.size} timepoints")

        parts: list[Any] = []
        counts: list[int] = []
        for t in times:
            mask = centers[:, TIME_COL] == t
            part = GSplatData(
                centers=centers[mask],
                amplitudes=amplitudes[mask],
                cholesky_factors=cholesky[mask],
                colors=None if colors is None else colors[mask],
                truncation_radius=truncation,
            )
            counts.append(part.n_splats)
            laddered = make_additive_lod(
                part,
                method=LADDER_METHOD,
                breakpoints=time_part_ladders([part.n_splats])[0],
            )
            parts.append(laddered.tree)
        if sum(counts) != total:
            raise RuntimeError(
                f"time parts hold {sum(counts):,} splats, source {total:,}"
            )
        root_attrs = _read_source_attrs(data_path)
        keep = ("source_stride", "source_timepoints", "stacked_axis_renumbered")
        meta = {k: root_attrs[k] for k in keep if k in root_attrs}
        if out_path.exists():
            shutil.rmtree(out_path)
        write_gsplats_tree(
            out_path,
            GSplatPartition(children=parts, max_elements=max(counts), meta=meta),
            barrier_dims=[TIME_COL],
            pipeline_info={
                "recipe": "time_parts_stream",
                "source": data_path.name,
                "first_rung": FIRST_RUNG_SPLATS,
                "ladder_method": LADDER_METHOD,
                "parts": len(parts),
            },
            root_attrs={
                **meta,
                "blending_mode": ARCHIVE_BLENDING_MODE,
                "absorption": ARCHIVE_ABSORPTION,
            },
        )
        aprint(f"{len(parts)} time parts, largest {max(counts):,} splats -> {out_path}")
        return out_path


def resolve_time_parts(data_path: Path, cache_dir: Path | None = None) -> Path:
    """Return the cached time-part partition for ``data_path``, building it once."""
    cache = time_parts_cache_dir(data_path) if cache_dir is None else cache_dir
    out = cache / "h2afva_51tp_timeparts.gsplats.zarr"
    if out.exists() and not RECOMPUTE:
        aprint(f"time parts cached: {out}")
        return out
    cache.mkdir(parents=True, exist_ok=True)
    return build_time_parts(data_path, out)


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 4D scene: re-author into time parts, then graft the partition."""
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
            stamp_input_digests(scene)
            scene.attrs["title"] = "GSplats: Zebrafish Embryogenesis (h2afva timelapse)"
            scene.attrs["description"] = (
                f"Zebrafish embryo nuclei (histone H2A variant label) across "
                f"{n_frames} timepoints of the zebrahub h2afva light-sheet "
                f"recording — every {SOURCE_STRIDE}th frame of 253 — fitted as "
                "121M Gaussian splats authored as one part per timepoint, each "
                "with its own progressive ladder, so every frame paints its "
                "brightest nuclei immediately and refines. Step Time to watch "
                "the body axis form. Press L for the Layers panel."
            )

            parts_path = resolve_time_parts(data_path)
            with asection(f"Adding gsplats ({n_frames} time parts, laddered)"):
                window = window_attrs(DISPLAY_WINDOW)
                scene.add_gsplats_from_file(
                    name="zebrafish_nuclei_4d",
                    path=str(parts_path),
                    # Loic's Layers-panel values on the time-part build
                    # (2026-09-10): volumetric compositing with a moderate
                    # absorption so the near half of the embryo reads in front of
                    # the far half, `plasma` as on the single-stack companion.
                    blending_mode=LAYER_BLENDING,
                    absorption=LAYER_ABSORPTION,
                    opacity=LAYER_OPACITY,
                    colormap=LAYER_COLORMAP,
                    # On a COLORMAPPED node `intensity`/`offset` ARE the scalar
                    # window feeding the LUT (`intensity = 1/(hi-lo)`,
                    # `offset = -lo/(hi-lo)`), not a colour gain. The window is
                    # DISPLAY_WINDOW = 0.001 .. 0.025 of the decoded amplitudes
                    # (p50 0.00050, p90 0.0036, p99 0.032, p99.9 0.057).
                    intensity=window["intensity"],
                    offset=window["offset"],
                    gamma=LAYER_GAMMA,
                    # Grafting carries none of the archive's root attrs, so the
                    # layer flag must be authored here. `test_demo_layers` also
                    # reads the module source to enforce that contract.
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
    aprint("51 timepoints • 121M splats • one laddered part per timepoint")
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
        data_path = recompute_archive(get_demos_output_dir() / "_h2afva_51tp_recompute")
    else:
        data_path = resolve_data()
    scene_path = create_luxar_scene(data_path, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer…")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
