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
    One leaf of 121,163,285 splats carrying a 12-rung progressive (stream)
    ladder. The viewer paints the first rung immediately and refines as the rest
    arrive; the stacked time axis is a hard coarsening barrier, so no splat ever
    blends two timepoints together.

    This archive USED to be a ``kind=partition`` of 44 parts, each an LOD group
    of four substitutive levels. Both structures were removed deliberately:

      - A partition has no viewer-side selector — ``load-partition-group-node``
        keeps every child visible — so 44 parts cost 44+ requests per view and
        bought culling the viewer never performed. Parts are load-bearing only
        against the per-node element cap, and this node is sliced on the hidden
        time axis, so only one timepoint's ~2.4M splats are ever resident.
      - The substitutive levels were the same splats at coarser merges, i.e.
        pure download weight for a node that is never seen whole.

    Dropping both and re-chunking took the archive from 1.87 GB to 1.06 GB and
    from 173 requests per timepoint step to 2, at identical finest-level content.

DATA SOURCE & CITATIONS:
    Royer lab, CZ Biohub San Francisco (zebrahub). Raw acquisition:
    ``h2afva/fused`` — 253 timepoints of 407 x 2048 x 2048 voxels, fused and
    deconvolved. Please cite the zebrahub resource when using this data.

WHICH TIMEPOINTS:
    Every fifth frame of the 253-timepoint recording — original indices
    0, 5, 10, ... 250 — giving 51 frames. The archive records this itself
    (``source_archive``, ``source_stride``, ``source_timepoints``), and the
    stacked axis is renumbered 0..50 so the viewer's discrete navigation grid
    lands exactly on stored values.

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
    1.87 GB fit, which changes its bytes and therefore its published checksum.
    That is a data-side change, not a scene-authoring one, so the axes here are
    honest about being pixels. At the companion's calibration (0.40625 um
    laterally, 1.625 um axially) this envelope is 660 x 828 x 831 um.

PIPELINE — reproducible with ``--recompute``:
    1. Fit the full 253-timepoint ``h2afva/fused`` timelapse and scale it
       isotropic: ``h2afva_253tp.gsplats.zarr``, which this demo takes as its
       PARENT (``--parent PATH``) rather than refitting. That fit is 9.25 GB and
       hours of GPU; re-deriving the 51-frame variant from it is minutes of CPU.
    2. ``restride_stacked_axis(stride=5)`` — keep original timepoints
       0, 5, ... 250 and renumber them to 0..50, preserving the parent's tree
       shape. Not ``gsplat slice``: that takes ranges, not a stride, and refuses
       a partition.
    3. ``gsplat flatten`` — collapse to one leaf, keeping the finest level.
    4. ``gsplat lod --recipe stream --target-ms 200`` — put the progressive
       ladder back (flatten drops it), sized for a ~200 ms first paint.
    5. ``gsplat optimise --profile archive`` — re-chunk. LAST, because step 4
       adds arrays that also want the 1 MB layout.

    Steps 2 and 3 are in that order for a memory reason, not a stylistic one:
    the parent is 602M splats at its finest level and flattening it whole peaked
    at 115 GB. Striding first cuts it to 121M before anything loads flat.

    The 51-frame variant is the manifest's default precisely because the
    253-frame one is 9.25 GB; this is the same recording at a ninth of the
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

import shutil
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
    parse_path_arg,
    run_luxar_cli,
)
from luxar.gsplats.io._archive import read_archive_root_attrs
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.restride import restride_stacked_axis
from luxar.gsplats.tree import center_bounds, iter_leaves
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
#: Parent's finest-level total, asserted before an hour of work: the strided
#: slice reads the parent's whole tree, so a wrong ``--parent`` is worth catching
#: at the door.
PARENT_FINEST_SPLATS = 602_580_152
#: What the recipe must reproduce, exactly: every step after the fit is
#: deterministic (a stride selection, a flatten that keeps the finest level, a
#: ladder built by a fixed rule), so unlike a refit this admits no drift
#: tolerance. A mismatch means the parent or the stride changed.
EXPECTED_SPLATS = 121_163_285
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


def _read_source_attrs(data_path: Path) -> dict:
    """Read source-root attrs from a directory or compressed archive."""
    if data_path.is_dir():
        return read_node_attrs(data_path) or {}
    return read_archive_root_attrs(data_path)


def _validate_parent(path: Path) -> None:
    """Assert ``--parent`` is the 253-timepoint archive this recipe expects.

    Worth doing up front: the strided slice walks the parent's entire tree, so a
    wrong ``--parent`` costs the better part of an hour before its splat count
    disagrees. Both checks below discriminate against a real sibling — the
    single-timepoint tp234 fit is also a partition with the same appearance
    attrs, and differs exactly in these two numbers.
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
    stacked_max = (bounds.get("max") or [None] * 4)[TIME_COL]
    if stacked_max is None:
        raise SystemExit(
            f"{path.name} declares no position_bounds, so its timepoint count "
            f"cannot be checked. Refusing to walk 602M splats on faith."
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
            "gsplat",
            "optimise",
            str(laddered),
            str(final),
            "--profile",
            CHUNK_PROFILE,
        )

        node, _ = load_gsplat_node(str(final))
        got = sum(int(leaf.n_splats) for leaf in iter_leaves(node))
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


def create_luxar_scene(data_path: Path, output_path: Path) -> Path:
    """Build the 4D scene by grafting the progressively-laddered leaf."""
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
                "121M Gaussian splats streamed over a 12-rung progressive "
                "ladder, so the first frame paints immediately and refines. The "
                "time axis is a coarsening barrier, so no splat blends two "
                "timepoints. Step Time to watch the body axis form. Press L for "
                "the Layers panel."
            )

            with asection(f"Adding gsplats (streamed leaf, {n_frames} frames)"):
                # Keep the additive mode used for the validated gallery build.
                # It mattered more when this was a 44-part partition with no BSP
                # split planes (parts could only be centroid-ordered, and
                # volumetric compositing popped at their seams); on one leaf the
                # reason is simply that the gallery stills were approved on it.
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
    aprint("51 timepoints • 121M splats • 12-rung progressive ladder")
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
