#!/usr/bin/env python
"""GSplats Demo: How Far Can You Decimate? (a measured study).

A fitted gsplat dataset is almost always bigger than it needs to be. This demo
is about *how to find out by how much* — and about the fact that WHICH reduction
scheme you use is worth more than the splat count you pick.

The subject is one stack of the zebrahub *h2afva* zebrafish recording, fitted to
**1,653,405 splats**. Four detail levels of the same embryo are stacked on a
categorical ``Detail`` dimension, so picking an entry swaps the level **in
place at a fixed camera** — the only way to actually perceive a 3 dB difference.
Every level here is a single flat leaf: no partition, no LOD tree, so what you
see is exactly the splats named in the selector and nothing is being swapped in
behind your back.

================================================================================
THE MEASUREMENT
================================================================================

Each candidate was rendered back into the source volume's own 407x2048x2048 grid
and scored against the real acquisition. The number that matters is **foreground
PSNR** — PSNR restricted to signal voxels — because the stack is 97.8% empty and
a global PSNR mostly measures how well a scheme reproduces black, which they all
do perfectly. Judged globally the 1% variant scores 45.6 dB and looks excellent,
while the embryo itself has lost 16 dB. The foreground mask is Otsu on the
intensity histogram (threshold 597 of a 33,888 range; 2.21% of voxels), chosen
so no variant can grade its own homework with a tuned threshold.

Three reduction families, foreground PSNR in dB:

    splats      %     prefix-mass  prefix-additive  merge-levels
    1,653,405  100%      48.93         48.93            —
      826,702   50%      42.98        [45.48]          44.52
      413,351   25%      37.19         39.11          [41.66]
      165,340   10%      33.42         34.46          [38.28]
       82,670    5%      31.81         32.39          [36.21]
       41,335  2.5%      30.66         30.93          [34.61]
       16,534    1%      29.49         29.58          [33.10]

  prefix-mass      keep the top-N splats by mass (amplitude x volume) — the
                   cheapest thing anyone tries first.
  prefix-additive  keep the first N of the additive ORDERING already stored in
                   the file (``self_energy``), which is chosen to COVER the
                   object rather than to keep the brightest parts of it.
  merge-levels     substitutive reduction (``kmeans_lloyd``): neighbouring
                   splats are MERGED into representatives carrying their
                   combined mass. Nothing is discarded — it is summarised.

WHAT THE NUMBERS SAY:

  1. **Merging beats deleting by 3-4 dB at equal splat count**, everywhere below
     50%. A prefix throws splats away and dims the embryo; a merge conserves
     mass. Concretely, merge at 10% matches what a prefix needs ~40% to reach,
     so choosing the scheme well is worth another ~4x reduction for free.
  2. **Except at mild reduction:** at 50% the additive prefix (45.48) beats the
     merge (44.52). Above about half there is little redundancy left to
     summarise and merging only blurs — and the prefix is free, since that
     ordering is already in the file.
  3. **There is no knee.** Quality falls smoothly at ~3-4 dB per halving, so
     "how far can we go" is a judgement about acceptable quality, not a free
     lunch waiting to be found.

The shipped levels use ``merge-levels`` (the winner in the range that matters).
10% — 165K splats, 38.3 dB, 26 MB -> 2.9 MB — is the recommended operating point.

REPRODUCING IT:

    luxar gsplat flatten parent.gsplats.zarr flat.gsplats.zarr
    luxar gsplat decimate flat.gsplats.zarr quarter.gsplats.zarr \
        -f 0.25 -m merge --device cpu
    # Repeat at -f 0.10 and -f 0.05 for the other shipped levels.
    # score against the source volume
    luxar gsplat compare quarter.gsplats.zarr orig_tp234.npy --device cuda

  The shipped counts were produced by this flatten + direct merge-decimation
  recipe. ``--recompute`` is the supported rebuild and reproduces their counts
  within 0.02%. ``compare`` reports GLOBAL PSNR; the foreground figures above
  come from the same render with the MSE restricted to Otsu-masked voxels.

DATA SOURCE & CITATIONS:
    Royer lab, CZ Biohub San Francisco (zebrahub). ``h2afva/fused``, timepoint
    234 of 253, 407 x 2048 x 2048, fused and deconvolved. Please cite the
    zebrahub resource when using this data. Axes are in physical microns: a
    SiMView-type light-sheet gives 0.40625 um laterally, and the z:xy ratio of 4
    established by the full-timelapse campaign puts the axial step at 1.625 um
    (see the h2afva stack demo for why the ratio, not the pitch, is the softer
    of the two numbers).

USAGE:
    python demo_gsplats_3d_decimation_study.py [--recompute] [--parent PATH] [--no-serve] [--serve-only]

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_3d_decimation_study.luxar.zarr
    - Opens in the browser; use the **Detail** selector to compare levels.
"""

DEMO_META = {
    "key": "gsplats_3d_decimation_study",
    "title": "GSplat Decimation Study",
    "description": (
        "How far can a gsplat fit be decimated? Four measured detail levels of one "
        "zebrafish embryo in one selector, with foreground PSNR for each."
    ),
    "category": "microscopy",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 27,
        "compute": "light",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["gsplats_3d_h2afva_decimation"],
    "outputs": ["gsplats_3d_decimation_study"],
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
from luxar.gsplats.io.load_gsplats import load_gsplat_node
from luxar.gsplats.tree import center_bounds
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DATASET = "gsplats_3d_h2afva_decimation"

SCENE_NAME = "gsplats_3d_decimation_study.luxar.zarr"

#: The shipped levels, coarsest LAST so the selector reads full -> reduced.
#: ``fg_psnr`` is the measured foreground PSNR (see the module docstring); it is
#: recorded here so each entry's label states the cost of that step rather than
#: leaving the viewer to guess.
LEVELS = [
    {
        "file": "h2afva_full.gsplats.zarr.zip",
        "splats": 1_653_405,
        "fg_psnr": 48.93,
        "label": "100% · 1.65M · 48.9 dB",
    },
    {
        "file": "h2afva_quarter.gsplats.zarr.zip",
        "splats": 411_180,
        "fg_psnr": 41.66,
        "label": "25% · 411K · 41.7 dB",
    },
    {
        "file": "h2afva_tenth.gsplats.zarr.zip",
        "splats": 165_276,
        "fg_psnr": 38.28,
        "label": "10% · 165K · 38.3 dB",
    },
    {
        "file": "h2afva_twentieth.gsplats.zarr.zip",
        "splats": 82_660,
        "fg_psnr": 36.21,
        "label": "5% · 83K · 36.2 dB",
    },
]

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: ``--parent PATH`` — the tp234 fit the levels are derived from. Optional: with
#: no flag the parent is fetched via ``ensure_dataset`` like any other demo
#: dataset. It exists because the parent is a lab fit rather than a public
#: download, so a rebuild from scratch has to be able to name a local copy.
PARENT_ARG = parse_path_arg("parent")

#: Reduction fractions, coarsest LAST, matched to LEVELS[1:].
#: `merge` (substitutive) NOT `prefix`: at these ratios merging beats deleting by
#: 3-4 dB because it conserves mass instead of discarding it. Verified against
#: the shipped files rather than assumed -- see RECOMPUTE_NOTES below.
DECIMATION_FRACTIONS = [0.25, 0.10, 0.05]

RECOMPUTE_NOTES = """\
Derived, not fitted: no source volume and no GPU are needed, only the tp234
parent. Chain, with every step verified against the shipped archives:

    tp234 parent   1,653,405 splats, 41 spatial tiles   (a kind=partition)
      flatten   ->  h2afva_full, single flat leaf, SAME count
      decimate  ->  quarter / tenth / twentieth at -f 0.25 / 0.10 / 0.05,
                    method=merge, device=cpu

`--device cpu` pins CUDA-capable hosts to the CPU path used for the recorded
rebuild; `auto` otherwise prefers CUDA when available. MPS already falls back to
CPU, so the explicit flag only silences that warning there. The measured CPU run
at -f 0.10 gave 165,271 splats against the shipped 165,276 (0.003%). No
CPU-versus-CUDA count comparison was made.

NOT REPRODUCED HERE: the shipped files went through a further encoding pass --
every family member differs in size between the 2026-08-12 build and the pinned
record copy (full 18,070,045 -> 17,812,123; twentieth 1,036,456 -> 1,010,680),
which is the v3/3.4 reencode. Byte-identity with the shipped archives is
therefore NOT expected from this path; splat counts and quality are.
"""


# =============================================================================
# Recompute (--recompute): rebuild the four levels from the tp234 parent
# =============================================================================
def _resolve_parent() -> Path:
    """The tp234 fit: an explicit --parent, else the shipped stack dataset."""
    if PARENT_ARG is not None:
        parent = PARENT_ARG.expanduser()
        if not parent.exists():
            raise FileNotFoundError(f"--parent does not exist: {parent}")
        aprint(f"parent from --parent: {parent}")
        return parent
    aprint("no --parent given; resolving the shipped tp234 stack")
    (parent,) = ensure_dataset("gsplats_3d_h2afva_stack")
    return parent


def recompute_levels(work_dir: Path) -> list[Path]:
    """Rebuild all four levels; returns them in LEVELS order.

    Writes into ``work_dir`` and returns directory stores (not zips) -- the
    scene builder loads either, and zipping is a packaging step that belongs to
    whoever publishes, not to a rebuild.
    """
    with asection("Recomputing decimation levels"):
        aprint(RECOMPUTE_NOTES)
        parent = _resolve_parent()
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Flatten the partition into the single flat leaf the study needs.
        #    Every level here must be ONE leaf: the demo's whole point is that
        #    what you see is the splats named in the selector, with no LOD tree
        #    swapping content in behind the camera.
        full = work_dir / LEVELS[0]["file"].replace(".zip", "")
        run_luxar_cli("gsplat", "flatten", str(parent), str(full))

        out = [full]
        for level, fraction in zip(LEVELS[1:], DECIMATION_FRACTIONS, strict=True):
            dest = work_dir / level["file"].replace(".zip", "")
            run_luxar_cli(
                "gsplat",
                "decimate",
                str(full),
                str(dest),
                "-f",
                str(fraction),
                "-m",
                "merge",
                "--device",
                "cpu",
            )
            out.append(dest)

        # Report the counts against what the labels claim. A level whose count
        # drifts far from its label makes the on-screen "25% / 411K / 41.7 dB"
        # a lie, and the label is the only thing telling a viewer what they are
        # looking at.
        drifted = []
        for level, path in zip(LEVELS, out, strict=True):
            node, _stats = load_gsplat_node(str(path))
            got = int(node.n_splats)
            want = level["splats"]
            drift = abs(got - want) / max(want, 1)
            flag = "ok" if drift < 0.01 else "DRIFT"
            aprint(f"  [{flag}] {path.name}: {got:,} splats (label says {want:,})")
            if drift >= 0.01:
                drifted.append(f"{path.name}: {got:,} instead of {want:,}")
        if drifted:
            details = "; ".join(drifted)
            raise ValueError(
                "--parent does not reproduce the counts behind the shipped labels "
                f"({details}); expected the tp234 h2afva fit, got {parent}"
            )
        return out


# =============================================================================
# Data loading
# =============================================================================
def resolve_data() -> list[Path]:
    """Resolve every detail level: cache -> in-repo copy -> Zenodo."""
    with asection("Resolving decimation levels"):
        paths = ensure_dataset(DATASET)
        by_name = {p.name: p for p in paths}
        ordered = [by_name[level["file"]] for level in LEVELS]
        for level, path in zip(LEVELS, ordered):
            aprint(f"{level['label']:26s} {path.name}")
        return ordered


# =============================================================================
# Scene construction
# =============================================================================
def create_luxar_scene(level_paths: list[Path], output_path: Path) -> Path:
    """Build the 4D scene: three spatial dims + a categorical Detail axis."""
    with asection("Creating decimation-study scene"):
        # Bounds come from the FULL level; every level is the same embryo, so
        # they share one coordinate space and the camera never moves between
        # them — which is the whole point of putting them on one axis.
        node, _ = load_gsplat_node(str(level_paths[0]))
        bmin, bmax = center_bounds(node)
        aprint(f"Scene bounds: min={np.round(bmin, 1)} max={np.round(bmax, 1)}")

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
                # Categorical, so the viewer offers a DROPDOWN of named levels
                # (not a continuous slider) and cannot land between two of
                # them, which would show both at once and defeat the
                # comparison.
                Dimension(
                    "Detail",
                    display=False,
                    categories=[level["label"] for level in LEVELS],
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
            scene.attrs["title"] = "GSplats: How Far Can You Decimate?"
            scene.attrs["description"] = (
                "One zebrafish embryo (zebrahub h2afva) at four measured detail "
                "levels, reduced by substitutive merging. Pick a level in the Detail "
                "selector to swap it in place. Labels give splat count and the measured "
                "FOREGROUND PSNR against the source volume — global PSNR runs 15 dB "
                "higher on this 97.8%-empty stack and would flatter every level."
            )

            for i, (level, path) in enumerate(zip(LEVELS, level_paths)):
                with asection(f"Adding level {level['label']}"):
                    scene.add_gsplats_from_file(
                        name=f"detail_{i}",
                        path=str(path),
                        # The data is 3D and the scene is 4D: `dim_order` names
                        # which scene dimensions the three center columns are,
                        # and `fill` pins this level's coordinate on the
                        # remaining (Detail) axis. Without `dim_order` the
                        # column count is checked against the scene's and the
                        # add is refused before `fill` is ever consulted.
                        dim_order=["Z", "Y", "X"],
                        fill={"Detail": float(i)},
                        # BOTH of the next two are load-bearing, and omitting
                        # either silently draws all four levels at once — the
                        # scene still loads, the selector still reads correctly,
                        # and nothing errors.
                        #
                        # `extend_to_all=[]`: a dimension absent from `dim_order`
                        # is "unmapped", and an unmapped dimension is
                        # AUTO-extended to all positions unless extend_to_all is
                        # set explicitly. Supplying `fill` does not count as
                        # mapping it. The empty list is the opt-out — the whole
                        # point here is that a level is NOT visible everywhere.
                        # `fill_sigma`: a splat is visible where its nD extent
                        # meets the slice, and an unmapped dim's sigma defaults
                        # to 1.0, which at 3-sigma truncation reaches across all
                        # four categories (spaced 1 apart). 0.1 keeps a level's
                        # reach (+/-0.3) inside its own category.
                        extend_to_all=[],
                        fill_sigma={"Detail": 0.1},
                        blending_mode="normal",
                        colormap="viridis",
                        intensity=1.0,
                        gamma=2.2,
                        layer=True,
                    )
                    aprint(f"  {level['splats']:,} splats @ {level['fg_psnr']} dB")

            scene.add_text(
                "Decimation study • pick a level in the Detail selector",
                position=(0.02, 0.02),
                font_size=0.045,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "zebrahub h2afva • merge-levels reduction • foreground PSNR",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve every level, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: How Far Can You Decimate? (a measured study)")
    aprint("=" * 70)
    aprint("Four detail levels of one embryo • pick one in the Detail selector")
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
        level_paths = recompute_levels(get_demos_output_dir() / "decimation_recompute")
    else:
        level_paths = resolve_data()
    scene_path = create_luxar_scene(level_paths, output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer… (use the Detail selector to compare levels)")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
