#!/usr/bin/env python
"""GSplats Demo: What Does Culling Actually Remove? (a measured study).

A fitted Gaussian-splat dataset is almost always carrying splats that cost bytes
and contribute nothing you can see. This demo is about *which* splats those are,
and about how far you can go before it starts to matter.

The subject is the same frame as the ``gsplats_3d_drosophila_gastrulation``
demo — timepoint 150 of a SiMView light-sheet recording of a *Drosophila*
embryo at gastrulation. Ten cull levels of that one fit are stacked on a
categorical ``Cull`` dimension, so picking an entry swaps the level **in place
at a fixed camera**. That matters more than it sounds: side by side, the eye has
to carry a memory across a gap and small differences vanish. Stacked, stepping
the selector is a flicker test, and a flicker test is the only way a volumetric
difference this subtle is visible at all. Every level is a single flat leaf —
no partition, no LOD tree — so what you see is exactly the splats the selector
names.

================================================================================
WHAT "CUMULATIVE RETENTION" MEANS
================================================================================

``luxar gsplat cull --method cumulative --retention R`` sorts the splats by
amplitude, walks from the brightest down, and keeps the prefix that carries a
fraction ``R`` of the fit's TOTAL amplitude. Everything below that line is
discarded. So ``retention=0.99`` does not mean "keep 99% of the splats" — it
means "keep whatever number of splats accounts for 99% of the light", which
was 93% of them on the superseded pre-refit archive generation measured below.

That asymmetry IS the finding. Amplitude in a fitted stack is heavy-tailed: a
minority of splats sit on nuclei and carry most of the signal, while a long tail
of dim, diffuse splats models background and out-of-focus haze. Cumulative
culling removes the tail first, in amplitude order — which is very nearly the
same order as "least visible first".

MEASURED ON THE SUPERSEDED PRE-REFIT ARCHIVE GENERATION
(an excerpt of its 200,155-splat table; the runtime overlay is recomputed from
whichever archive is loaded):

    retention    splats   % kept     size   vs the raw stack
      1.000     200,155    100.0%   1.97 MB       75:1
      0.990     186,956     93.4%   1.83 MB       81:1
      0.900     139,933     69.9%   1.38 MB      108:1
      0.750      95,160     47.5%   0.95 MB      156:1

**Giving up a quarter of the total amplitude halves the file**, and the
amplitude given up is spread across the dimmest splats in the scene rather than
concentrated anywhere you are looking.

One caveat the measured archive generation itself illustrates. Its fitter's
post-fit pass had ALREADY culled 256,000 seeds to 200,155 splats, so its dim tail
was partly gone before this study started and each retention kept more than it
would on a raw fit. On an unculled fit of the same specimen,
``retention=0.999`` alone removes 23% of the splats for one thousandth of the
light. So the numbers above are a *lower* bound on what culling can recover, not
an upper one, and a fit that has never been culled has much more slack than this
table suggests.

================================================================================
WHY THIS IS NOT A FREE LUNCH, AND HOW TO TELL
================================================================================

Culling is deletion, not summarisation. Nothing is merged and no mass is
conserved, so every level below 1.000 is strictly dimmer in total than the one
above it — the question is only whether the light removed was structure or haze.
Two habits keep you honest:

1. **Judge it on a render at a fixed camera, not on a number.** Global PSNR on a
   mostly-empty stack mostly measures how well you reproduce black. This demo
   deliberately gives you the flicker test instead of a dB column, because on
   this data the dB column would move long before your eye did.
2. **Re-check on the busiest frame, not the prettiest one.** A cull level chosen
   at gastrulation is not automatically safe once the germband extends and the
   nuclei crowd together.

The counterpart study is ``gsplats_3d_decimation_study``, which asks the same
"how far can you go" question with MERGING rather than deleting, and finds
merging worth 3-4 dB at equal splat count below the halfway mark. The two are
complementary: merge when you need the count down and the mass kept; cull when
what you want gone is genuinely background.

DATA SOURCE & CITATIONS:
    Keller lab, HHMI Janelia Research Campus (L. A. Royer was then a postdoctoral
    fellow there). Imaged on a SiMView multi-view light-sheet microscope
    under the AutoPilot adaptive framework:

      Royer, L.A., Lemon, W.C., Chhetri, R.K., Wan, Y., Coleman, M., Myers, E.W.
      & Keller, P.J. "Adaptive light-sheet microscopy for long-term,
      high-resolution imaging in living organisms."
      Nat. Biotechnol. 34, 1267-1278 (2016). doi:10.1038/nbt.3708

    Specimen: ``w; His2Av::mRFP1; +`` (Bloomington stock #23560).

    This demo reuses the ``gsplats_3d_drosophila_gastrulation`` archive rather
    than shipping its own, so it adds no hosted bytes.

USAGE:
    python demo_gsplats_3d_culling_study.py [--no-serve] [--serve-only]

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_3d_culling_study.luxar.zarr
    - Press 1 then [ / ] to step the Cull selector; press L for the Layers panel.
"""

DEMO_META = {
    "key": "gsplats_3d_culling_study",
    "title": "GSplat Culling Study",
    "description": (
        "What does culling actually remove? Ten cull levels of one Drosophila "
        "gastrulation fit in a single selector, with splat count and compression "
        "ratio for each."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 3,
        "compute": "light",
        "gpu": "none",
        # Reuses the single-frame Drosophila archive; see `caches` below.
        "local_data": None,
    },
    "caches": ["gsplats_3d_drosophila_gastrulation"],
    "outputs": ["gsplats_3d_culling_study"],
    "citation": {
        "short": "Royer et al. 2016",
        "doi": "10.1038/nbt.3708",
        "license": "CC BY 4.0",
    },
}

import shutil
import tempfile
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
    stamp_input_digests,
)
from luxar.demos._cinematic_camera import VIEWER_DEFAULT_FOV_DEG, pull_in
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import iter_leaves
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_3d_drosophila_gastrulation"
SCENE_NAME = "gsplats_3d_culling_study.luxar.zarr"

#: Cumulative-amplitude retention for each level, gentlest first.
RETENTIONS = [1.0, 0.999, 0.995, 0.99, 0.98, 0.96, 0.94, 0.90, 0.85, 0.75]

#: The acquisition this fit represents: one timepoint of the 500-frame
#: recording, 108 x 1352 x 532 uint16. DECLARED rather than read — the shipped
#: archive predates source-grid stamping and carries no ``source_shape``, so the
#: figure comes from the acquisition itself (verified against the source zarr)
#: and not from the file. The ratios below are therefore against the DECODED
#: acquisition; the same timepoint inside the compressed source zarr is ~38.6 MB,
#: roughly a quarter of this, so a ratio quoted against what you would actually
#: download is ~4x smaller. Both are legitimate, they answer different questions,
#: and quoting one without saying which is how compression numbers get inflated.
RAW_SHAPE = (108, 1352, 532)
RAW_ITEMSIZE = 2  # uint16
RAW_BYTES = int(np.prod(RAW_SHAPE)) * RAW_ITEMSIZE  # 155,361,024

# Appearance carried from the approved single-frame demo — same data, same look.
# Amplitudes arrive in raw detector counts and are normalised to a robust 1.0 by
# `add_gsplats_from_data` (one factor for the whole stacked node), which is what
# makes these opacity/absorption values portable numbers rather than magic
# constants. See core/group/gsplats_pipeline/amplitude_norm.py.
# The carried 5 / 512 / 798 values came from the sibling demo's earlier archive
# generation (minimum / p99.9 reference / maximum detector counts). This
# pure-scale transfer is approximate because its manual normalization subtracts 5.
GSPLAT_OPACITY = 0.41 * (512.0 - 5.0) / (798.0 - 5.0)
GSPLAT_ABSORPTION = 0.57
DISPLAY_WINDOW_TOP = 0.737 * (798.0 - 5.0) / (512.0 - 5.0)

CAMERA_FRAME_FILL = 0.72

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]


# =============================================================================
# Data
# =============================================================================
def load_base_fit() -> GSplatData:
    """The shipped single-frame fit, collapsed to one flat splat set.

    The archive carries an additive ladder; its rungs are disjoint prefixes of
    one set, so their union IS the fit and concatenating them loses nothing.
    """
    with asection("Resolving the Drosophila gastrulation fit"):
        path = ensure_dataset(DATASET)[0]
        aprint(f"Data: {path}")
        node, _ = load_gsplat_node(str(path))
        subs = [s for leaf in iter_leaves(node) for s in leaf.additive_sublods]
        data = GSplatData(
            centers=np.concatenate([np.asarray(s.centers) for s in subs]),
            amplitudes=np.concatenate([np.asarray(s.amplitudes) for s in subs]),
            cholesky_factors=np.concatenate(
                [np.asarray(s.cholesky_factors) for s in subs]
            ),
        )
        aprint(f"Base fit: {data.n_splats:,} splats, {data.ndim}D")
        return data


def measure_stored_bytes(data: GSplatData, workdir: Path, tag: str) -> int:
    """Bytes this level actually occupies once written as a ``.gsplats.zarr``.

    Written and measured rather than estimated from a bytes-per-splat constant:
    the encoder picks its quantisation per array from the data's own range, so
    bytes/splat is not constant across cull levels and an estimate would quietly
    misreport exactly the quantity this demo exists to show.
    """
    out = workdir / f"{tag}.gsplats.zarr"
    data.save(str(out))
    return sum(p.stat().st_size for p in out.rglob("*") if p.is_file())


def build_levels(base: GSplatData) -> tuple[list[GSplatData], list[dict]]:
    """Cull the base fit at every retention; return the levels and their stats."""
    levels: list[GSplatData] = []
    rows: list[dict] = []
    workdir = Path(tempfile.mkdtemp(prefix="luxar-cull-study-"))
    try:
        with asection("Culling"):
            for r in RETENTIONS:
                data = (
                    base
                    if r >= 1.0
                    else base.cull(method="cumulative", retention=float(r))
                )
                nbytes = measure_stored_bytes(data, workdir, f"r{r:.3f}")
                ratio = RAW_BYTES / nbytes
                rows.append(
                    {
                        "retention": r,
                        "splats": data.n_splats,
                        "pct": 100.0 * data.n_splats / base.n_splats,
                        "mb": nbytes / (1024 * 1024),
                        "ratio": ratio,
                    }
                )
                levels.append(data)
                aprint(
                    f"retention {r:.3f}: {data.n_splats:>8,} splats "
                    f"({rows[-1]['pct']:5.1f}%)  {rows[-1]['mb']:6.2f} MB  "
                    f"{ratio:6.1f}:1 vs the {RAW_BYTES / 1048576:.0f} MB raw stack"
                )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return levels, rows


def level_label(row: dict) -> str:
    """One selector entry: everything you need to judge it, in one line."""
    return (
        f"{row['retention']:.3f} · {row['splats'] / 1000:.0f}k splats · "
        f"{row['pct']:.0f}% · {row['ratio']:.0f}:1"
    )


def scene_description(rows: list[dict]) -> str:
    """Describe the culling study using the fit loaded for this build."""
    return (
        "Ten cumulative-amplitude cull levels of one Drosophila "
        f"gastrulation fit ({rows[0]['splats']:,} splats), stacked on a Cull selector so "
        "they swap in place at a fixed camera. Culling removes the dimmest "
        "splats first, which on this data is background haze rather than "
        "nuclei. Press 1 then [ / ] to step the selector."
    )


# =============================================================================
# Scene
# =============================================================================
def create_luxar_scene(output_path: Path) -> Path:
    base = load_base_fit()
    levels, rows = build_levels(base)

    with asection("Creating the culling-study scene"):
        # sigma = 0 on the new axis: a level is a discrete choice and must not
        # smear into its neighbour, which would show two levels at once and
        # defeat the comparison.
        stacked = GSplatData.combine_as_new_dimension(
            levels, values=[float(i) for i in range(len(levels))], sigma=0.0
        )
        aprint(f"Stacked: {stacked.n_splats:,} splats, {stacked.ndim}D")

        centers = np.asarray(stacked.centers)
        bmin, bmax = centers.min(axis=0), centers.max(axis=0)
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
                # Categorical, so the viewer offers named levels rather than a
                # continuous slider and cannot land between two of them.
                Dimension(
                    "Cull",
                    display=False,
                    categories=[level_label(row) for row in rows],
                ),
            ]
        )

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
            scene.attrs["title"] = "GSplats: What Does Culling Actually Remove?"
            scene.attrs["description"] = scene_description(rows)

            scene.add_gsplats_from_data(
                name="drosophila_nuclei",
                result=stacked,
                blending_mode="volumetric",
                absorption=GSPLAT_ABSORPTION,
                opacity=GSPLAT_OPACITY,
                colormap="magma",
                intensity=1.0 / DISPLAY_WINDOW_TOP,
                offset=0.0,
                gamma=1.0,
                layer=True,
                # Left at the default ON PURPOSE. One insertion of one stacked
                # node means ONE factor across every cull level — which is what
                # makes the comparison fair. Normalising each level separately
                # would rescale the heavily-culled ones brighter and hide the
                # very dimming the study is measuring.
            )

            # Overlays. One add_text PER LINE: a "\n" inside an overlay string
            # renders as a single space, so a multi-line block written as one
            # call comes out as one long run-on line.
            _add_overlay(scene, rows)

            add_demo_caption(
                scene,
                "SiMView light-sheet • His2Av::mRFP1 • cumulative-amplitude culling",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


#: Measured-table typography. Smaller than the explainer so the table's left
#: edge stays clear of the explainer's right edge (see `_add_overlay`).
TABLE_HEADER_FONT = 0.015
TABLE_ROW_FONT = 0.0155
TABLE_TOP = 0.078
TABLE_ROW_PITCH = 0.0215


def _add_overlay(scene, rows: list[dict]) -> None:
    """Title, the point of the demo, and the measured table.

    Two layout constraints learned the hard way, both visible the moment you
    open the scene rather than in any test:

    * **Left overlays must clear the icon rail**, which occupies roughly the
      first 5% of the viewport width. Text at x=0.02 renders UNDER it.
    * **The overlay font is proportional, so space-padded columns do not
      line up.** A table written with ``{:>9}`` alignment comes out ragged.
      Middle-dot separated fields read cleanly at any width instead, and the
      table is anchored top-RIGHT where nothing collides with it.
    """
    white = "rgba(255,255,255,0.92)"
    dim = "rgba(255,255,255,0.62)"
    faint = "rgba(255,255,255,0.45)"
    left = 0.055  # clear of the icon rail

    # The explainer and the table share the top band, so their widths are
    # budgeted: the explainer lines are kept short enough to end well before
    # x ~ 0.62 at this size, and the table is small enough to start past
    # x ~ 0.75 (2026-09-10: at 0.019 the second explainer line ran under the
    # table's second row on a 16:9 view).
    for size, y, text, colour in [
        (0.028, 0.05, "What does culling actually remove?", white),
        (
            0.017,
            0.085,
            "Cumulative culling keeps the brightest splats carrying a fraction R "
            "of the total amplitude.",
            dim,
        ),
        (
            0.017,
            0.107,
            "The dim, diffuse tail goes first — background and haze — so the count "
            "falls faster than the look.",
            dim,
        ),
        (
            0.017,
            0.136,
            "Press 1 then [ / ] to step the Cull selector: same camera, same "
            "pixels — a flicker test.",
            faint,
        ),
    ]:
        scene.add_text(
            text, position=(left, y), font_size=size, anchor="top-left", color=colour
        )

    scene.add_text(
        "retention · splats · kept · size · vs raw",
        position=(0.98, 0.05),
        font_size=TABLE_HEADER_FONT,
        anchor="top-right",
        color=faint,
    )
    for i, row in enumerate(rows):
        scene.add_text(
            f"{row['retention']:.3f} · {row['splats']:,} · {row['pct']:.0f}% · "
            f"{row['mb']:.2f} MB · {row['ratio']:.0f}:1",
            position=(0.98, TABLE_TOP + i * TABLE_ROW_PITCH),
            font_size=TABLE_ROW_FONT,
            anchor="top-right",
            color=dim,
        )
    scene.add_text(
        f"ratio vs the decoded acquisition — {RAW_SHAPE[0]}x{RAW_SHAPE[1]}"
        f"x{RAW_SHAPE[2]} uint16 = {RAW_BYTES / 1048576:.0f} MB",
        position=(0.98, TABLE_TOP + len(rows) * TABLE_ROW_PITCH + 0.012),
        font_size=0.014,
        anchor="top-right",
        color=faint,
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    aprint("=" * 70)
    aprint("GSplats Demo: What Does Culling Actually Remove?")
    aprint("=" * 70)
    aprint("Ten cull levels of one Drosophila gastrulation fit, in one selector")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    scene_path = create_luxar_scene(output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer…")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
