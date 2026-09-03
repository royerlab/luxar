#!/usr/bin/env python3
"""GSplats Demo: 4D Zebrafish Embryo Time-Lapse (Zenodo 1211599).

A 4D (3D + time) confocal recording of a living zebrafish embryo during
gastrulation, as **one** Gaussian-splat node with time as a real axis of the
data: 151 timepoints over five hours, while the fluorescently labelled
endodermal cells spread from a tight cluster over the whole yolk sphere.

The specimen occupies well under 2% of the imaged voxels, so a **wireframe cage
with a 100 um grid** is drawn around the acquisition volume as its own
toggleable layer. Without it there is nothing to judge scale, depth or drift
against — the cells float in an unmarked void, and the migration that is the
whole point of the recording is impossible to read.

================================================================================
READ THIS FIRST: what this demo is trying to teach
================================================================================

This file is longer than a demo needs to be, on purpose. The dataset is
**extremely sparse** — 98.7 to 99.8% of its voxels are exactly zero — and almost
every automatic default in the fitting pipeline is calibrated for data that is
not. Three separate defaults INVERT on it, each in the same way: an estimator
that asks "what value best predicts a held-out voxel?" answers "zero", because
zero is almost always right. Follow those estimators and you ship a dataset with
the specimen filtered out of it.

The general lesson, and the reason for the tables below: **on sparse data,
measure the thing you actually care about, not the thing that is easy to
measure.** Every number in this docstring is a measurement, each with the
command that produced it implied by the surrounding constant. Four rules
recur, and they generalise to any near-empty volume:

1. **Global PSNR is the reward for predicting empty space.** On a 99%-empty
   stack it barely moves when the specimen is destroyed. Always score a
   foreground — and where the "foreground" itself contains noise, score the
   part you care about (here: voxels inside real connected components).
2. **Never score a preprocessed fit against its own preprocessed input.** A
   floored or denoised fit compared to the floored or denoised volume looks
   excellent by construction. Score against the raw acquisition, and when the
   preprocessing was deliberate (denoising), split the score into what it
   risked and what it bought rather than collapsing both into one number.
3. **Reproduced ENERGY is the honest sparse-data metric.** The share of the
   frame's intensity a fit puts back is impossible to fake and immediately
   exposes a filter that deleted signal — it was 44% when the background floor
   was on, and nobody noticed from the dB.
4. **An automatic calibration that answers at the edge of its own grid has
   told you nothing.** Widen the grid and look again; if the answer keeps
   moving, the estimator does not apply to your data. The ``cal`` K* sweep
   fails this way here: it answers the SECOND point of its own grid, with a
   confidence margin smaller than the spacing between points. The NLM
   calibrator did too, back when this demo used NLM — default range capped at
   0.08, answer 0.08 — which is part of why it no longer does.

WHAT MAKES THIS A 4D NODE (and not 151 nodes in a trench coat):
    Every timepoint is still fitted on its own — that is what keeps each fit
    small and separately cacheable — but the fits are then
    ``combine_as_new_dimension``-stacked into ONE 4D ``GSplatData`` whose fourth
    centre column is time (sigma = 0: a splat is instantaneous and must not
    smear across frames). That single node then gets a substitutive LOD ladder
    with time as a hard coarsening barrier, and the viewer's Time slider walks
    the axis. The previous version of this demo instead wrote one flat gsplats
    node per timepoint with ``fill={"time": t}`` — 64 sibling nodes, no LOD, no
    shared appearance, one Layers row each.

    Why it matters beyond tidiness: a stacked node is what makes a LOD ladder
    possible at all (there is one object to coarsen), what gives the whole
    recording one appearance to edit in the Layers panel, and what lets the
    viewer's spatial index cull by time. The cost is that time becomes a
    coordinate you must protect — see ``LOD_COARSEN_DIMS``.

DATA SOURCE & CITATIONS:
    Zenodo record 1211599 — ``cxcr4aMO2_290112.lsm`` (Zeiss LSM, 2.1 GB),
    Danio rerio (zebrafish) cxcr4a morphant, endoderm label, confocal
    laser-scanning microscopy. DOI 10.5281/zenodo.1211599, CC BY-SA 4.0.

    Please cite the record if you use this data. ShareAlike binds whoever
    receives a scene built from it, which is why the licence is carried into
    the store rather than only mentioned here.

ACQUISITION (read from the LSM's own metadata, not assumed):
    151 timepoints x 44 x 512 x 512 uint8, voxel 7.1847 x 1.6774 x 1.6774 um
    (ZYX) — anisotropic by 4.28x along Z — one frame every 120.01 s, so the
    recording spans 5 h 00 m. Each fit runs on the voxel grid and is then scaled
    by that voxel size (an exact diagonal affine, carried into the Cholesky
    factors), so the shipped centres are in microns and the embryo has its true
    proportions; the imaged block is 316 x 859 x 859 um.

SPLAT BUDGET (measured on this data, 2026-08-22, RTX PRO 6000):
    The previous version asked for 2,000 seeds and shipped a few hundred splats,
    because it also took the fitter's bare defaults: 1,000 iterations (below the
    ``draft`` preset's 2,000) and ``cull_retention=0.95``, which discards most of
    the splats along with the last 5% of amplitude. Every arm below is frame 110
    with the floor OFF (see FLOOR), scored against the raw frame; "energy" is the
    share of the frame's total intensity the reconstruction puts back.

    | fit                                   | splats | global | foreground | energy |
    |---------------------------------------|--------|--------|------------|--------|
    | shipped config (2k, 1000 it, cull .95)|    767 |  33.92 |  10.10 dB  |  0.74  |
    | 8k seeds, 5000 it, cull .9999         |  7,877 |  37.37 |  14.98 dB  |  0.90  |
    | 16k seeds, 5000 it, cull .9999        | 12,439 |  39.66 |  18.88 dB  |  0.98  |
    | 32k seeds, 5000 it, cull .9999        | 12,940 |  40.68 |  20.30 dB  |  0.98  |
    | 64k seeds, 5000 it, cull .9999        | 13,006 |  40.81 |  20.46 dB  |  0.98  |
    | 32k seeds, 5000 it, cull .95          |  9,889 |  39.17 |  16.90 dB  |  0.91  |

    32,000 seeds is the plateau, not a compromise: doubling it again moves the
    reconstruction by 0.13 dB and adds 66 splats, because the post-fit cull —
    not the seed budget — is what sets the final count on data this
    heavy-tailed. The last row isolates that cull: at the SAME seeds, retaining
    0.9999 of the amplitude instead of 0.95 is worth +1.5 dB global and +3.4 dB
    foreground, since on a volume this sparse the discarded 5% of amplitude *is*
    the dim cells. Against the config that shipped, the fit this demo now does
    is +6.8 dB global, +10.2 dB foreground and 17x the splats. Side-by-side MIPs
    at matched zoom agree: at a few hundred splats the cells are blurred blobs
    with many missing entirely; by ~13,000 the population is complete and
    further splats change nothing visible.

    ``cal`` is deliberately not consulted. Its blind-spot sweep put K* at 1,852
    for this dataset — the second point of its own grid, with a 0.17 dB
    confidence margin, measured on the XY-halved copy the demo used to fit.

FLOOR (why this demo turns off a default the house rule says to keep):
    ``--floor auto`` estimates the histogram mode of the NON-ZERO voxels, capped
    at their median. That is right on a stack with a camera pedestal, where the
    non-zero population is background. Here 98.7-99.8% of voxels are exactly
    zero, so the non-zero population IS the specimen and the level lands inside
    it — and climbs as the embryo brightens. At 32k seeds, scored against the
    unfloored frames:

    | frame | non-zero | level auto picked | auto (global / fg / energy) | none  |
    |-------|----------|-------------------|-----------------------------|-------|
    |     0 |    0.20% |            26/255 | 46.37 / 16.28 dB / 0.73     | 48.18 / 19.31 / 1.01 |
    |    75 |    0.35% |            63/255 | 38.37 / 11.51 dB / 0.55     | 45.94 / 22.14 / 1.00 |
    |   150 |    1.35% |            91/255 | 29.23 /  8.13 dB / 0.44     | 36.73 / 18.82 / 0.97 |

    Up to +7.6 dB global (frame 75) and +10.7 dB foreground (frame 150) for
    turning it off, and reproduced energy goes from 44-73% back to ~100%. Read
    the energy column: global PSNR barely moves at frame 0 and hides how much
    was being deleted.

SUPPRESSING THE SHOT NOISE (and a metric that lied about it):
    The noise here is not a nuisance at the margin. At t=0, 32% of the frame's
    total energy sits in isolated voxels, and the MEDIAN object in the whole
    frame is ONE voxel -- p90 is 1-2 at every timepoint. An unfiltered fit
    spends its budget accordingly.

    So each timepoint has its sub-4-voxel 6-connected components deleted before
    it is fitted. Not smoothed -- deleted. Every 6-connected object at or above
    the threshold passes through bit-identical.

    HOW THIS SECTION GOT REWRITTEN, because the mistake is the lesson:

    The first attempt used non-local means at h=0.05, chosen by sweeping h and
    scoring "the share of the shot-noise voxels' energy the fit still
    reproduces". That metric said the noise collapsed 12-30x. It was wrong, and
    wrong in a way worth recognising: **NLM is a neighbourhood average, so it
    SPREADS a spike rather than deleting it.** Energy that moved one voxel out
    of its original spike left the mask the metric was watching and scored as
    removed, while remaining perfectly visible on screen as a softer, wider
    blob. The metric measured displacement and reported destruction.

    It took someone looking at the render and saying "that doesn't look
    denoised" to catch it. No number in the sweep would have.

    The replacement metric -- energy outside a DILATED cell mask -- was also
    wrong, in the opposite direction: at the busy timepoints NLM's own halo
    pushes real cell energy past the mask boundary, scoring 1.049 (worse than no
    filter at all) for what is actually signal. Two metrics, two directions,
    both plausible. The general lesson is not "use this metric" but: **when a
    filter MOVES things, any mask fixed on the unfiltered data measures the
    movement, not the removal.** Prefer an operation that does not move
    anything, or score by looking.

    WHY A SIZE FILTER, measured with 6-connectivity against a cell definition
    (components >= 27 voxels) deliberately STRICTER than any threshold under
    test, so no arm is judged against its own definition:

    | threshold | frame energy removed (t=0) | cell energy | mean cell peak |
    |-----------|----------------------------|-------------|----------------|
    | NLM h=.05 |                      ~0.32 |       0.953 |     **0.783**  |
    |         2 |                     0.2959 |      1.0000 |       1.0000   |
    |         4 |                     0.3208 |      1.0000 |       1.0000   |
    |         8 |                     0.3225 |      1.0000 |       1.0000   |
    |        12 |                     0.3225 |      1.0000 |       1.0000   |

    There is no trade-off to tune: a component filter's cell columns are exactly
    1.0000 at every threshold, by construction, while NLM dims the cells it is
    supposed to protect by 6-22% depending on timepoint. The only thing the
    threshold changes is how much noise goes, and that curve is flat past 4. So
    4 takes essentially all of the reduction while deleting nothing larger than
    3 voxels.

    The energy removed falls with time -- 32% at t=0, 13% at t=40, 2.7% at t=150
    -- not because the noise changes but because the specimen brightens around
    it. A fixed filter, a moving proportion.

    WHAT IT DOES TO THE FIT, which is the part that settles the argument. Both
    arms rendered back and scored INSIDE the 6-connected cells of the RAW frame
    -- the one reference neither arm touched:

    | frame | arm | splats | cell PSNR | cell energy | out-of-cell energy |
    |-------|-----|--------|-----------|-------------|--------------------|
    |     0 | mc4 |  1,369 |     19.51 |       0.992 |            0.011   |
    |     0 | NLM | 23,478 |     19.66 |       0.987 |            0.503   |
    |    10 | mc4 |  1,809 |     20.07 |       1.002 |            0.012   |
    |    10 | NLM | 21,398 |     19.96 |       0.997 |            0.488   |
    |    75 | mc4 |  7,730 |     22.33 |       0.989 |            0.108   |
    |    75 | NLM | 12,837 |     21.75 |       0.971 |            0.325   |
    |   120 | mc4 | 14,090 |     20.53 |       0.965 |            0.076   |
    |   120 | NLM | 17,740 |     20.50 |       0.962 |            0.243   |
    |   150 | mc4 | 18,754 |     18.77 |       0.956 |            0.071   |
    |   150 | NLM | 20,732 |     19.04 |       0.957 |            0.216   |

    Equal or better cell fidelity almost everywhere, from a small fraction of
    the splats at the early timepoints. The last column says where the rest
    went: NLM's fits put back 22-50% of the energy OUTSIDE the cells against
    1-11% here, because a smeared spike is still something to model and the
    0.9999 amplitude retention keeps modelling it. Deleting the spike instead
    leaves the budget nothing to spend on but cells.

    NLM wins cell PSNR in two rows, and both are left in rather than dropped. At
    t=0 it buys 0.15 dB (19.66 against 19.51), but uses 17x the splats and puts
    back 46x the out-of-cell energy. At t=150 it buys 0.27 dB (19.04 against
    18.77) with 10% more splats. By then only 2.7% of the frame's energy is
    noise, so there is little for a size filter to remove and the comparison is
    nearly two unfiltered fits; the gain is the cost of deleting a few small
    real objects along with the specks. That is the honest shape of this trade,
    next to the 3x lower out-of-cell energy in the same row.

    Note how the splat count tracks the specimen rather than the noise: 1,369 at
    t=0 where a third of the energy was noise, 18,754 at t=150 where almost none
    of it is. The whole archive is 1,270,233 splats against NLM's 2,689,314.

    This is also the answer to "did we lose detail by shipping fewer splats".
    Splat count is not detail. It is only detail once you know what the splats
    are sitting on.

THE LOD LADDER, AND A MEASUREMENT TRAP INSIDE IT:
    The stacked node carries a substitutive ladder (``levels``: 3 coarse levels,
    each 4x smaller) with ``coarsen_dims=(0, 1, 2)`` — the three SPATIAL centre
    columns. Column 3 is time, and leaving it out makes it a **hard barrier**:
    coarsening may merge two splats that are near each other in space at the
    same timepoint, never two at different timepoints. Without the barrier a
    coarse level would average frame 40 into frame 41 and the migration would
    smear into a blur that gets worse the further out you zoom. This is the
    general rule for any stacked axis — time, channel, condition: a coarsening
    that crosses it is mixing measurements that were never simultaneous.

    The trap: when checking whether the ladder LOST anything, do not reach for
    ``volumes()`` or the amplitude sum. A stacked axis has sigma = 0, so a
    splat's nD "volume" is zero (or a degenerate product) and any total built
    from it is meaningless — twice during this work it produced a convincing
    "the coarse level lost 40% of the mass" alarm that was purely an artefact of
    the sigma-0 axis. The check that actually works is to **render each level to
    a volume and compare brightness**: the first coarse level holds 95.8% of the
    finest level's, which is what conservation is supposed to look like. When a
    derived quantity has a degenerate factor in it, measure the observable
    instead.

AUTHORING FOR THE VIEWER (three traps that only a browser reveals):
    None of these show up in a test, a PSNR, or a scene-graph dump. All three
    were found by loading the built scene and using it, and all three generalise
    to any 4D demo.

    1. A DISCRETE dimension's ``step`` must be exactly representable in binary.
       The viewer turns a discrete axis into an ``<input type="range">``, whose
       value is snapped onto ``min + k*step``. Hand it the LSM's measured
       interval — 2.0001470947 min — and the browser computes ``150 * step`` a
       hair above ``max`` and clamps to 149, so the FINAL timepoint cannot be
       selected at all. The axis therefore steps by exactly 2.0 min, a nominal
       value, at a cost of 1.3 s of drift accumulated over five hours. Nominal
       but reachable beats recorded but not. See ``AXIS_STEP_MIN``.

    2. The ORDER of the ``Dimensions`` list is what the viewer maps onto screen
       x/y/z. It does not have to match the centre-column order — ``dim_order``
       maps those by NAME — and here it deliberately does not: listing Z first
       shows an 859 x 859 x 316 um slab edge-on, as a tall narrow column. The
       two long axes go first so the opening view is the one the microscope was
       pointed at. See ``create_luxar_scene``.

    3. An additive reference layer must lose every contest for attention. The
       cage's first colours were bright enough to bury the specimen it exists to
       measure — 2% of the frame, drawn over by a glowing box. Roughly halving
       them fixed it. See ``BOX_EDGE_COLOR``.

REPRODUCING ANY OF THIS:
    Every table here came from the same shape of experiment: hold everything
    fixed, sweep one knob, and score each arm against the RAW frames. The
    scoring helper that matters is the connected-component split. Both masks
    use 6-connectivity: the threshold sweep calls components of >= 27 voxels
    cells, deliberately stricter than every threshold it tests; the rendered-fit
    table uses >= 8 voxels. Everything smaller is shot noise for that table.
    This split separates "the filter removed noise" from "the filter removed
    signal", which no single PSNR can do.

    The fits are cached per timepoint under a key that includes every knob that
    changes the result (seeds, iterations, patience, cull retention, floor,
    minimum component size and connectivity), so re-running a sweep cannot be
    served a stale answer. That keying is not bookkeeping — an earlier version
    of this demo keyed only on the frame index, and the tables above are exactly
    the sort of sweep it would have silently invalidated.

USAGE:
    python demo_gsplats_4d_zebrafish_timelapse.py [--recompute] [--no-serve]
        [--refit-all] [--serve-only] [--max-timepoints=N]

    --recompute:        Ignore the shipped archive; download the LSM and build
                        the 4D fit locally (needs a GPU). Per-timepoint fits are
                        still read from the cache, so an interrupted run of all
                        151 frames RESUMES where it stopped.
    --refit-all:        With --recompute, also ignore the per-timepoint cache
                        and fit every frame again. Rarely wanted: the cache key
                        already carries every constant that changes a fit.
    --no-serve:         Build the scene without launching the viewer.
    --serve-only:       Serve an already-built scene.
    --max-timepoints=N: Fit only N evenly spaced timepoints (refit path only).

OUTPUT:
    - Scene saved to:  datasets/demos/gsplats_4d_zebrafish_timelapse.luxar.zarr
    - Opens in the browser; press L for the Layers panel, play the Time slider.
"""

DEMO_META = {
    "key": "gsplats_4d_zebrafish_timelapse",
    "title": "4D Zebrafish Gastrulation (endoderm)",
    "description": (
        "Five hours of zebrafish gastrulation as one 4D Gaussian-splat node, "
        "inside a gridded cage that gives the sparse embryo a scale."
    ),
    "category": "microscopy",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 19,  # the 19,229,817-byte archive, and nothing else
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["gsplats_zebrafish"],
    "outputs": ["gsplats_4d_zebrafish_timelapse"],
    "citation": {
        "short": "Aanstad 2018",
        "doi": "10.5281/zenodo.1211599",
        # ShareAlike: worth carrying into the store, since it binds whoever
        # receives a scene built from this data.
        "license": "CC BY-SA 4.0",
    },
}

import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    DatasetUnavailable,
    add_demo_caption,
    detect_device,
    launch_viewer,
    load_dataset_gsplats,
    load_local_fit_gsplats_at,
    local_fit_path,
    parse_demo_flags,
    parse_int_arg,
    require_module,
    warn_if_no_cuda_gpu,
)
from luxar.demos.registry import DEMO_CACHE_ROOT
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.merged_quality import collect_part_provenance
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME = "gsplats_zebrafish"
GSPLATS_FILE = "zebrafish_4d.gsplats.zarr.zip"
SCENE_NAME = "gsplats_4d_zebrafish_timelapse.luxar.zarr"

#: The raw acquisition, only needed on the ``--recompute`` path.
ZENODO_URL = "https://zenodo.org/api/records/1211599/files/cxcr4aMO2_290112.lsm/content"
LSM_BYTES = 2_080_484_264

#: The acquisition geometry, as the LSM records it. Held here rather than only
#: read from the file because the SCENE needs it and the scene is normally built
#: from the shipped archive, with no LSM in reach: the cage must be the imaged
#: BLOCK, and a box fitted to the splats instead would shrink around a specimen
#: that fills under 2% of it — a ruler cut to the size of the thing it measures.
#: :func:`open_lsm` checks the file against these on the refit path, so they
#: cannot drift silently.
ACQUISITION_SHAPE_ZYX = (44, 512, 512)
VOXEL_SIZE_ZYX_UM = (7.184696827249337, 1.6774389429296399, 1.6774389429296399)
FRAME_INTERVAL_S = 120.00882789993899

#: The Time axis step, in minutes: the recorded interval ROUNDED to exactly 2.
#:
#: Not cosmetic. A discrete dimension becomes an ``<input type="range">`` whose
#: value is snapped onto ``min + k*step``, and with the raw 2.0001470947... the
#: browser's sanitiser computes ``150 * step`` a hair above ``max`` and clamps to
#: 149 — so the FINAL timepoint cannot be selected with the slider at all. Two is
#: exact in binary and every multiple of it up to 300 is too.
#:
#: The cost is 0.0001471 min per frame, 1.3 s of drift accumulated over the whole
#: five hours, against a frame interval of two minutes. The axis is therefore
#: nominal-but-exact rather than recorded-but-unreachable.
AXIS_STEP_MIN = 2.0

CACHE_DIR = DEMO_CACHE_ROOT / DEMO_NAME
LSM_PATH = CACHE_DIR / "cxcr4aMO2_290112.lsm"
#: Per-timepoint fits, keyed on EVERY knob of the fit schedule so retuning
#: cannot hit a stale entry. All of them change the result while leaving the
#: frame index identical, and the cull is the sharpest: the SPLAT BUDGET table
#: is the proof — same 32k seeds, retention .95 gives 9,889 splats and .9999
#: gives 12,940. A cache keyed on the frame alone would serve the old fits back
#: to anyone re-running that sweep.
FITS_DIR = CACHE_DIR / "fits"
#: The stacked 4D archive this machine built, in the demo's own namespace. It is
#: a different artifact from the hosted file of the same purpose, so it must NOT
#: live at the manifest path — ``ensure_dataset`` hashes whatever it finds there
#: and quarantines a mismatch, which would refit on every launch (#1618).
LOCAL_FIT = local_fit_path(DEMO_NAME, GSPLATS_FILE)

#: Fit schedule. See the module docstring's SPLAT BUDGET table for the sweep
#: these came from. The seed ceiling was not re-swept after filtering; retaining
#: 32,000 is harmless because the amplitude cull, not the ceiling, sets the final
#: count.
#:
#: ``seeds`` proposes and ``cull_retention`` disposes. The seed budget is a
#: CEILING, not a target: the post-fit cull keeps splats only until 99.99% of the
#: amplitude is accounted for, so the count that ships is whatever the frame
#: needs, and after the component filter that is a few thousand rather than
#: the tens of thousands an unfiltered or NLM-smoothed frame produced. Both
#: numbers matter, and the second is the one people forget: raising `seeds`
#: past the plateau changes nothing, while moving `cull_retention` from the
#: fitter's default 0.95 to 0.9999 was worth +3.4 dB of foreground on its own,
#: because on data this sparse the discarded 5% of amplitude IS the dim cells.
SEEDS = 32_000
N_ITERS = 5_000
EARLY_STOP_PATIENCE = 500
CULL_RETENTION = 0.9999

#: Delete connected components smaller than this before fitting.
#:
#: This stack's shot noise is not a nuisance at the margin, it is a third of the
#: signal at the dim end: at t=0, 32% of the frame's total energy sits in
#: isolated voxels, and the MEDIAN object in the frame is ONE voxel (p90 is 1-2
#: at every timepoint). An unfiltered fit spends its budget accordingly.
#:
#: A size filter is chosen over a smoothing one because it matches that noise
#: model exactly. NLM was tried first and is the wrong tool here: a neighbourhood
#: average SPREADS a single-voxel spike instead of deleting it, so the spike
#: survives as a softer, wider blob, and the same averaging dims the cells --
#: mean cell peak fell to 0.78 of raw at t=0. A component filter cannot do
#: either: it removes objects below the threshold and leaves every object above
#: it bit-identical.
#:
#: 4 is measured. Against a deliberately STRICTER cell definition than the
#: threshold itself (components >= 27 voxels, so the test cannot be circular),
#: every threshold from 2 to 12 keeps cell energy AND mean cell peak at exactly
#: 1.0000 -- there is no signal cost to trade off. What varies is only how much
#: noise goes, and that curve is flat past 4 (at t=0: 29.6% of frame energy
#: removed at 2, 32.1% at 4, 32.25% at 12). So 4 takes essentially all of the
#: available reduction while only ever deleting 1-3 voxel objects, which at
#: p90=1-2 is unambiguously the noise population. Going higher buys ~0.2% more
#: and starts risking genuinely small, dim cells at the early timepoints where
#: cells are few and faint.
MIN_COMPONENT_VOXELS = 4

#: SciPy connectivity 1: face-connected neighbours only (6-neighbour in 3D).
#: Connectivity changes which touching voxels form one object, so it is explicit
#: and participates in the fit cache key rather than arriving as a library default.
COMPONENT_CONNECTIVITY = 1

#: NO background floor, stated rather than defaulted. The house rule is to stay
#: on ``auto`` unless you have measured otherwise; this is a dataset where
#: measuring says otherwise, by up to 10.7 dB of foreground. See the module
#: docstring's FLOOR section for the table and why `auto` inverts here.
FLOOR = "none"

#: Substitutive LOD: three coarse levels, each 4x lighter than the last, every
#: level carrying its own progressive ladder. Coarsening is restricted to the
#: three SPATIAL centre columns — time (column 3) is a hard barrier, because
#: merging across it would blend one timepoint's cells into the next and smear
#: the whole recording into a single haze.
#:
#: The archive is built with ``build_recipe`` rather than through
#: ``luxar.demos._lod_policy.save_with_lod``, for the reason its 4D sibling
#: ``demo_gsplats_4d_cell_tracking_challenge`` has: the helper writes exactly one
#: topology per demo and takes no barrier argument, while this demo writes two
#: kinds of store — the per-timepoint fits, which are scratch consumed by the
#: stack and want no ladder at all, and the one 4D archive it ships.
LOD_COMPRESSION_FACTOR = 4
LOD_LEVELS = 3
LOD_N_LODS = 4
LOD_COARSEN_DIMS = (0, 1, 2)

#: Appearance. Volumetric with a modest kappa: the specimen is a thin shell of
#: cells on a sphere, so a little occlusion separates near from far, while a
#: large one would hide the far side entirely. ``inferno`` starts at (0, 0, 4)
#: rather than viridis's dark purple, so the 98% of the volume that is empty
#: reads as true black instead of a haze.
COLORMAP = "inferno"
VOLUME_OPACITY = 1.0
VOLUME_ABSORPTION = 0.5

#: The acquisition cage: wireframe box + grid, in microns.
GRID_STEP_UM = 100.0
BOX_EDGE_WIDTH_UM = 2.0
GRID_LINE_WIDTH_UM = 1.0
#: Dim, and dimmer still for the rulings. These are ADDITIVE lines drawn over
#: a specimen that occupies 2% of the frame: at the first values tried
#: ((0.42, 0.62, 0.72) / (0.16, 0.26, 0.32)) the cage read as a solid glowing
#: box and buried the cells it exists to give a scale to. A reference must be
#: legible and lose every contest for attention.
BOX_EDGE_COLOR = (0.20, 0.30, 0.36)
GRID_LINE_COLOR = (0.075, 0.115, 0.145)

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

#: Force every timepoint to be fitted again, ignoring the per-timepoint cache.
#:
#: ``--recompute`` deliberately does NOT do this. The cache key carries every
#: constant that changes a fit (there is a test enumerating them, plus a guard
#: that fails when a new one reaches the fitter without being added), so a
#: cached entry under the current key IS what refitting would produce — and
#: honouring it makes an interrupted run of all 151 timepoints RESUME instead of
#: starting over. That is not hypothetical: this refit died at frame 37 of 151
#: and the bypass would have thrown away forty minutes of GPU time.
#:
#: The escape hatch exists for the case the key cannot cover — a cache you
#: suspect was written by a different build of the fitter itself.
REFIT_ALL = "--refit-all" in sys.argv

#: ``None`` = every timepoint in the recording. Only consulted when refitting.
MAX_TIMEPOINTS = parse_int_arg("max-timepoints", None, sys.argv)

Arbol.max_depth = 5

DEVICE: Optional[str] = None


# =============================================================================
# Acquisition
# =============================================================================
def open_lsm() -> Any:
    """Download the LSM if needed, check its geometry, and open it lazily.

    Returns a lazy ``(T, Z, Y, X)`` zarr view: the movie is 1.7 GB of voxels and
    only one timepoint is ever needed at a time, so it is never read whole.

    The LSM's own voxel size and frame interval are compared against the module
    constants above rather than merely read, because the scene uses the
    constants — a record re-uploaded with a different calibration would
    otherwise mislabel every axis of a rebuilt archive without a word.
    """
    import zarr  # a core dependency, unlike tifffile

    tifffile = require_module("tifffile")

    from luxar.demos import robust_download

    with asection("Zebrafish LSM (Zenodo 1211599)"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        robust_download(
            ZENODO_URL, LSM_PATH, max_retries=5, timeout=900, expected_size=LSM_BYTES
        )

        with tifffile.TiffFile(str(LSM_PATH)) as tif:
            meta = getattr(tif, "lsm_metadata", None) or {}
            voxel = tuple(
                float(meta.get(k, 0.0)) * 1e6
                for k in ("VoxelSizeZ", "VoxelSizeY", "VoxelSizeX")
            )
            interval = float(meta.get("TimeIntervall", 0.0))

        # series 0 is the image data; series 1 is the embedded RGB thumbnail.
        array = zarr.open(
            tifffile.imread(str(LSM_PATH), aszarr=True, series=0), mode="r"
        )
        aprint(f"Acquisition: {array.shape} {array.dtype}")
        aprint(f"Voxel (Z,Y,X): {voxel[0]:.4f} x {voxel[1]:.4f} x {voxel[2]:.4f} um")
        aprint(
            f"Frame interval: {interval:.2f} s "
            f"({array.shape[0] * interval / 3600:.2f} h total)"
        )

        drift = [
            f"{what}: file says {got}, this demo assumes {want}"
            for what, got, want, ok in (
                (
                    "grid",
                    tuple(array.shape[1:]),
                    ACQUISITION_SHAPE_ZYX,
                    tuple(array.shape[1:]) == ACQUISITION_SHAPE_ZYX,
                ),
                (
                    "voxel size (um)",
                    voxel,
                    VOXEL_SIZE_ZYX_UM,
                    np.allclose(voxel, VOXEL_SIZE_ZYX_UM, rtol=1e-4),
                ),
                (
                    "frame interval (s)",
                    interval,
                    FRAME_INTERVAL_S,
                    abs(interval - FRAME_INTERVAL_S) < 1e-3,
                ),
            )
            if not ok
        ]
        if drift:
            raise RuntimeError(
                "The downloaded LSM does not match the acquisition this demo is "
                "written against:\n  "
                + "\n  ".join(drift)
                + "\nUpdate ACQUISITION_SHAPE_ZYX / VOXEL_SIZE_ZYX_UM / "
                "FRAME_INTERVAL_S, and the numbers quoted in the docstring."
            )
        return array


def select_timepoints(n_total: int, limit: Optional[int]) -> list[int]:
    """Frame indices to fit — every frame, or a uniform subsample of ``limit``.

    The shipped archive takes every frame, which is the change that matters:
    the previous version asked for 64 of 151 and computed ``stride =
    n_total // n_use`` = 2, so it stopped at frame 126 and the last 50 minutes
    of gastrulation — the part where the endoderm has actually spread — never
    made it into the demo at all.

    A subsample is UNIFORM by construction, not merely evenly spread. Time
    becomes a discrete viewer dimension whose navigation snaps to multiples of
    one ``step``, so unevenly spaced frames would land between stops and those
    stops would silently render nothing.

    The stride is SEARCHED rather than computed, because neither closed form
    is good enough. Rounding ``(n - 1) / (limit - 1)`` down and truncating to
    ``limit`` — what this demo used to do — takes frames 0-99 for
    ``--max-timepoints=100`` of 151 and stops two thirds of the way through
    gastrulation. Rounding it up fixes that case and breaks a smaller one: 3 of
    10 frames becomes 2, spanning 56%. Scanning every stride and keeping the
    most frames that fit in the budget (ties to the one reaching furthest)
    dominates both — measured over n = 2..199 and limit = 2..59, its worst span
    is 80% against 50.4% for rounding down and 50.3% for rounding up. (At the
    151 frames this demo actually has, the search never drops below 95.3%.)

    The scan is over at most ``n_total`` strides on a list of timepoints, so it
    is free next to a single fit.
    """
    if limit is None or limit >= n_total:
        return list(range(n_total))
    if limit < 2:
        raise ValueError(f"--max-timepoints must be at least 2, got {limit}")
    best: list[int] = []
    for stride in range(1, n_total):
        frames = list(range(0, n_total, stride))
        if len(frames) > limit:
            continue
        if (len(frames), frames[-1]) > ((len(best), best[-1]) if best else (0, 0)):
            best = frames
    return best


# =============================================================================
# Fitting
# =============================================================================
def _fit_cache_path(frame: int) -> Path:
    return FITS_DIR / (
        f"f{frame:04d}_k{SEEDS}_i{N_ITERS}_p{EARLY_STOP_PATIENCE}"
        f"_c{CULL_RETENTION}_{FLOOR}_mc{MIN_COMPONENT_VOXELS}"
        f"_conn{COMPONENT_CONNECTIVITY}.gsplats.zarr.zip"
    )


def denoise(volume: np.ndarray) -> np.ndarray:
    """Delete connected components below ``MIN_COMPONENT_VOXELS`` rather than smoothing.

    Pure deletion, not smoothing: every object at or above the threshold comes
    through bit-identical, so cell peaks and cell energy are untouched by
    construction rather than by measurement. That is the property a smoothing
    filter cannot offer and the reason this replaced NLM here.

    Cheap too -- one ``ndimage.label`` pass per frame, no GPU, against the tens
    of seconds per frame the NLM path cost.
    """
    if not MIN_COMPONENT_VOXELS or MIN_COMPONENT_VOXELS < 2:
        return volume
    from scipy import ndimage

    structure = ndimage.generate_binary_structure(volume.ndim, COMPONENT_CONNECTIVITY)
    labels, _ = ndimage.label(volume > 0, structure=structure)
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes < MIN_COMPONENT_VOXELS)
    # Excluding label 0 is a COST guard, not a correctness one: background is
    # already zero, so writing zeros over it changes nothing -- but on a volume
    # that is 98.7% empty it would make the mask span nearly every voxel.
    small = small[small != 0]
    if not small.size:
        return volume
    out = volume.copy()
    out[np.isin(labels, small)] = 0.0
    return out


def fit_timepoint(volume: np.ndarray, frame: int, acquisition: tuple):
    """Fit one timepoint, caching the result.

    The cached store is what gets stacked, not the in-memory fit: the cache is
    written under a lossy encoding, so returning the unquantized object would
    make a cold run and a warm run build different scenes. ``include_stats=True``
    on the read keeps the fit's normalization provenance (floor / image range),
    which ``combine_as_new_dimension`` only propagates into the stacked result
    when every part reports one (#1175).
    """
    global DEVICE
    cache_file = _fit_cache_path(frame)
    if cache_file.exists() and not REFIT_ALL:
        try:
            return GSplatData.load(cache_file, include_stats=True)
        except Exception as exc:  # noqa: BLE001
            aprint(f"  frame {frame}: cache unreadable ({exc}); refitting")
            cache_file.unlink(missing_ok=True)

    if DEVICE is None:
        DEVICE = detect_device()
    from luxar.gsplats import fit_gaussian_splats

    volume = denoise(volume)
    src_shape, src_dtype = acquisition
    result = fit_gaussian_splats(
        volume,
        seeds=SEEDS,
        n_iters=N_ITERS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        cull_retention=CULL_RETENTION,
        floor=FLOOR,
        # Fit on the VOXEL grid, and apply the microns afterwards (see
        # `to_microns`). Handing the fitter `voxel_size` instead puts the
        # optimizer in physical space, which costs the same wall clock (A/B at
        # 32k seeds: 38.1 s voxel vs 32.6 s physical on frame 110) but scores
        # slightly worse on both frames tried — 34.26/10.44 dB against
        # 34.00/10.08 global/foreground at t=110, 46.37/16.29 against
        # 46.12/16.07 at t=0 — plausibly because a 4.28x-anisotropic grid makes
        # the fitter's isotropic seed shapes a worse starting guess in microns
        # than in voxels. It is also the space the SPLAT BUDGET table was
        # measured in. The scale itself is exact either way: a diagonal affine,
        # which `GSplatData.transform` carries into the Cholesky factors.
        output_space="voxel",
        # One timepoint of the stored uint8 stack is the source; `volume` is a
        # normalized float32 copy of it, so without this the compression ratio
        # would be quoted against a denominator 4x too large.
        source_shape=src_shape,
        source_dtype=src_dtype,
        device=DEVICE,
        verbose=False,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    return GSplatData.load(cache_file, include_stats=True)


def fit_all_timepoints(array, frames: list[int]) -> list[GSplatData]:
    """Fit every selected timepoint, reading one frame of the movie at a time."""
    acquisition = (tuple(int(s) for s in array.shape[1:]), str(array.dtype))
    scale = (
        float(np.iinfo(array.dtype).max)
        if np.issubdtype(array.dtype, np.integer)
        else 1.0
    )
    fits: list[GSplatData] = []
    with asection(
        f"Fitting {len(frames)} timepoints (seeds={SEEDS:,}, {N_ITERS} iters, "
        f"min component {MIN_COMPONENT_VOXELS} voxels)"
    ):
        for i, frame in enumerate(frames):
            volume = np.asarray(array[frame]).astype(np.float32) / scale
            fits.append(fit_timepoint(volume, frame, acquisition))
            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                aprint(
                    f"  {i + 1}/{len(frames)} — frame {frame}: "
                    f"{fits[-1].n_splats:,} splats"
                )
        report_fit_quality(fits)
    return fits


def report_fit_quality(fits: list[GSplatData]) -> None:
    """Print what the fits actually achieved, from their own stamps.

    Each fit scores itself against the array it was handed, which since the
    component filter is the FILTERED frame — not the acquisition. So these say
    how faithfully the fit represents what it was asked to fit, and are NOT
    quotable as fidelity to the microscope. Labelled accordingly, because an
    unlabelled PSNR here would end up on a Zenodo record meaning something it
    does not.

    The gap is smaller than it was under NLM, though, and for a reason worth
    knowing: a component filter only ever sets voxels to zero, so the filtered
    frame agrees with the acquisition EXACTLY wherever a cell exists. The
    difference between these numbers and raw-referenced ones is confined to
    voxels that held deleted noise.

    Both columns are reported: on a volume this sparse, global PSNR is mostly
    the reward for predicting empty space correctly, and the foreground figure
    is the one that moves when the fit gets better or worse.
    """

    def column(key: str) -> Optional[np.ndarray]:
        vals = [g.stats.get(key) for g in fits]
        good = np.array([float(v) for v in vals if isinstance(v, (int, float))])
        return good if good.size else None

    counts = np.array([g.n_splats for g in fits])
    aprint(
        f"Splats per timepoint: median {int(np.median(counts)):,}, "
        f"range {counts.min():,}-{counts.max():,}, total {counts.sum():,}"
    )
    vs = "vs the filtered input" if MIN_COMPONENT_VOXELS >= 2 else "vs the raw frame"
    for label, key in (("global", "psnr_db"), ("foreground", "foreground_psnr_db")):
        col = column(key)
        if col is None:
            # A cache written before the fit stamped quality, or a fitter path
            # that skipped it — say so rather than print a silent blank.
            aprint(f"{label.capitalize()} PSNR: not stamped on these fits")
        else:
            aprint(
                f"{label.capitalize()} PSNR ({vs}): median {np.median(col):.2f} "
                f"dB, range {col.min():.2f}-{col.max():.2f} dB"
            )


def acquisition_box_um() -> tuple[np.ndarray, np.ndarray]:
    """The imaged block in microns, centred on the origin: ``(bmin, bmax)``.

    Every timepoint shares one grid, so this is a constant of the recording —
    which is what lets the splats be recentred on it and the cage drawn around
    it without either needing the other in hand.
    """
    half = 0.5 * np.asarray(VOXEL_SIZE_ZYX_UM) * np.asarray(ACQUISITION_SHAPE_ZYX)
    # float32, because that is the width the cage vertices are STORED at. Left in
    # float64 the declared dimension range and the written vertex disagree in the
    # last bit, and the compiler correctly warns that a vertex sits outside its
    # own axis — a warning that would be noise here and hides real ones.
    half = half.astype(np.float32)
    return -half, half


def to_microns(fit: GSplatData) -> GSplatData:
    """Put one voxel-space fit into microns, centred on the acquisition box.

    The scale is the LSM's own voxel size — a diagonal affine, so
    ``GSplatData.transform`` carries it exactly into the Cholesky factors and
    the 4.28x axial anisotropy comes out corrected rather than merely stretched
    at display time.

    The recentring is on the *box* centre, not on the fit's own
    amplitude-weighted centroid. That matters more here than in a static demo:
    the labelled endoderm migrates across the yolk over five hours, so a
    per-frame centroid would chase it and the embryo would appear to swim on the
    spot while the cage slid past it.
    """
    scale = np.diag(np.asarray(VOXEL_SIZE_ZYX_UM, dtype=np.float64))
    # +0.5 voxel: a fitted centre is a voxel INDEX, and voxel i occupies
    # [i, i+1) of the block, so its middle is i + 0.5. Without the shift the
    # mapping is asymmetric — index 0 lands exactly ON bmin with no slack while
    # a whole voxel goes spare at the far face, so any splat the optimizer moves
    # to a slightly negative index falls outside the range the scene declares.
    # With it, both faces keep half a voxel and index -0.5 lands exactly on bmin.
    offset = 0.5 * np.asarray(VOXEL_SIZE_ZYX_UM, dtype=np.float64)
    # The box's own half-extent IS the centre offset, so the splats and the cage
    # cannot drift apart: both read it from the same place.
    _, half = acquisition_box_um()
    return fit.transform(scale).translate(offset - half)


def combine_to_4d(
    per_timepoint: list[GSplatData], times_min: list[float]
) -> GSplatData:
    """Stack per-timepoint 3D fits into one 4D (ZYX + time) dataset.

    Amplitudes are left alone. Every frame of this recording is an 8-bit stack
    whose brightest cells sit at 255, so the fits already share a scale, and
    normalising each frame to a fixed peak — the usual defence against a
    brightness pop — would flatten the real signal growth from 0.2% to 1.4%
    occupancy that is the developmental story.
    """
    # Preserve the as-fitted stamps before the archived splats are transformed.
    part_provenance = collect_part_provenance(
        per_timepoint,
        values=times_min,
        fit_reference={
            "kind": "preprocessed",
            "note": f"connected-component filter, minimum {MIN_COMPONENT_VOXELS} voxels",
        },
    )
    stacked = GSplatData.combine_as_new_dimension(
        [to_microns(g) for g in per_timepoint],
        values=times_min,
        sigma=0.0,  # a splat is instantaneous; it must not smear across frames
        part_provenance=part_provenance,
    )
    aprint(f"Stacked {stacked.n_splats:,} splats over {len(per_timepoint)} timepoints")
    return stacked


def build_lod(stacked: GSplatData) -> GSplatData:
    """Give the 4D archive its substitutive ladder, with time a hard barrier."""
    from luxar.gsplats.lod import RecipeParams, build_recipe

    return build_recipe(
        stacked,
        "levels",
        RecipeParams(
            compression_factor=LOD_COMPRESSION_FACTOR,
            levels=LOD_LEVELS,
            n_lods=LOD_N_LODS,
            coarsen_dims=LOD_COARSEN_DIMS,
            device=DEVICE or "auto",
        ),
    )


# =============================================================================
# The acquisition cage
# =============================================================================
#: The 12 edges of a box, as index pairs into the 8 corners of :func:`_corners`
#: (axis a is bit ``2 - a`` of the corner index: axis 0 is the HIGH bit).
_BOX_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),  # along axis 2
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),  # along axis 1
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),  # along axis 0
)


def _corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """The 8 corners of an axis-aligned box; axis a is bit ``2 - a``."""
    return np.array(
        [
            [bmax[a] if (i >> (2 - a)) & 1 else bmin[a] for a in range(3)]
            for i in range(8)
        ],
        dtype=np.float32,
    )


def cage_lines(
    bmin: np.ndarray, bmax: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wireframe box plus a grid ruled on its six faces.

    Returns ``(vertices, colors, widths)`` for ``line_type="segments"`` —
    consecutive vertex PAIRS are independent edges, so nothing ever joins the
    end of one grid line to the start of the next.

    The grid is anchored at 0 in scene coordinates (the box centre), not at the
    box minimum, so the rulings are symmetric about the specimen and a reading
    off them is a signed distance from the middle of the acquisition. Lines that
    land within a hair of a face are dropped, since the box edge already draws
    there and a doubled line reads as a brighter one.

    Colours are authored in sRGB and returned linearized, because the viewer
    composites in linear light.
    """
    segs: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    widths: list[float] = []

    corners = _corners(bmin, bmax)
    for a, b in _BOX_EDGES:
        segs.append(np.stack([corners[a], corners[b]]))
        cols.append(np.tile(np.float32(BOX_EDGE_COLOR) ** 2.2, (2, 1)))
        widths += [BOX_EDGE_WIDTH_UM] * 2

    eps = 1e-3 * float(np.max(bmax - bmin))
    for face_axis in range(3):
        u, v = [a for a in range(3) if a != face_axis]
        for face_at in (bmin[face_axis], bmax[face_axis]):
            # Rule the face along u (spanning v), then along v (spanning u).
            for ruled, spanned in ((u, v), (v, u)):
                lo = np.ceil(bmin[ruled] / step) * step
                for pos in np.arange(lo, bmax[ruled] + eps, step):
                    if min(abs(pos - bmin[ruled]), abs(pos - bmax[ruled])) < eps:
                        continue  # coincides with a box edge already drawn
                    a = np.empty(3, dtype=np.float32)
                    a[face_axis] = face_at
                    a[ruled] = pos
                    b = a.copy()
                    a[spanned], b[spanned] = bmin[spanned], bmax[spanned]
                    segs.append(np.stack([a, b]))
                    cols.append(np.tile(np.float32(GRID_LINE_COLOR) ** 2.2, (2, 1)))
                    widths += [GRID_LINE_WIDTH_UM] * 2

    return (
        np.concatenate(segs).astype(np.float32),
        np.concatenate(cols).astype(np.float32),
        np.asarray(widths, dtype=np.float32),
    )


# =============================================================================
# Data resolution
# =============================================================================
def load_or_build_gsplats() -> GSplatData:
    """Return the 4D stacked splats: hosted archive, local rebuild, or a refit."""
    if not RECOMPUTE:
        try:
            hosted = load_dataset_gsplats(DEMO_NAME, [GSPLATS_FILE])
            if hosted is not None:
                return hosted[0]
        except DatasetUnavailable:
            aprint("Precomputed 4D fit unavailable (Git LFS asset not pulled).")
        local = load_local_fit_gsplats_at([LOCAL_FIT], label=DEMO_NAME)
        if local is not None:
            return local[0]
        aprint(f"Falling back to download + fit (one-time; cached at {LOCAL_FIT}).")

    warn_if_no_cuda_gpu()
    array = open_lsm()
    frames = select_timepoints(int(array.shape[0]), MAX_TIMEPOINTS)
    fits = fit_all_timepoints(array, frames)
    times_min = [frame * AXIS_STEP_MIN for frame in frames]
    stacked = combine_to_4d(fits, times_min)

    with asection("Building the LOD ladder"):
        laddered = build_lod(stacked)
        LOCAL_FIT.parent.mkdir(parents=True, exist_ok=True)
        laddered.save(
            LOCAL_FIT,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=True,
            compress="zip",
            zip_deflate=True,
        )
    loaded = load_local_fit_gsplats_at([LOCAL_FIT], label=DEMO_NAME)
    assert loaded is not None, LOCAL_FIT  # just written
    return loaded[0]


# =============================================================================
# Scene
# =============================================================================
def create_luxar_scene(stacked: GSplatData, output_path: Path) -> Path:
    """Build the 4D scene: one gsplats node, one cage layer, overlays."""
    times = np.unique(stacked.centers[:, 3])
    n_timepoints = int(times.size)
    if n_timepoints < 2:
        # One timepoint has no interval to derive a step from, and what follows
        # would divide by zero on its way to `Dimension(step=0.0)`, which is
        # refused several frames later by a message that does not mention time.
        raise ValueError(
            f"a timelapse needs at least 2 timepoints, got {n_timepoints}; "
            "this archive is a single stack, not a recording"
        )
    t_min, t_max = float(times.min()), float(times.max())
    step_min = (t_max - t_min) / (n_timepoints - 1)

    # Time is a DISCRETE viewer dimension: navigation snaps to
    # `t_min + k * step`, and a chunk is only fetched within a quarter-step of
    # the snapped position. A timepoint sitting off that grid produces a slider
    # stop that renders nothing at all — silently. Cheap to check here,
    # invisible if it ever stops holding.
    offsets = times - t_min
    off_grid = np.abs(offsets - np.round(offsets / step_min) * step_min).max()
    if off_grid > 0.01 * step_min:
        raise RuntimeError(
            f"timepoints are not uniformly spaced (worst offset {off_grid:.4g} "
            f"min against a {step_min:.4g} min step); the viewer's discrete "
            "navigation would land on empty stops"
        )

    bmin, bmax = acquisition_box_um()

    with asection("Creating the 4D zebrafish scene"):
        aprint(f"Splats: {stacked.n_splats:,} over {n_timepoints} timepoints")
        aprint(f"Time: 0 to {t_max:.0f} min, step {step_min:.2f} min")
        aprint(f"Box: {' x '.join(f'{s:.0f}' for s in (bmax - bmin))} um")

        # X, Y, Z — the ORDER of this list is what the viewer maps onto screen
        # x/y/z, and it does not have to match the centre-column order (which
        # `dim_order` maps by name). Listing Z first shows the specimen edge-on:
        # the 316 um axial extent becomes screen-x and the two 859 um lateral
        # axes become screen-y and depth, so the opening view is a tall narrow
        # column of a block that is actually a wide flat slab. Lateral first
        # gives the en-face view the microscope was pointed at.
        dims = Dimensions(
            [
                Dimension(
                    "X", unit="um", display=True, range=(float(bmin[2]), float(bmax[2]))
                ),
                Dimension(
                    "Y", unit="um", display=True, range=(float(bmin[1]), float(bmax[1]))
                ),
                Dimension(
                    "Z", unit="um", display=True, range=(float(bmin[0]), float(bmax[0]))
                ),
                # Real minutes, not a frame index: the LSM records a 120.01 s
                # interval, so the slider can read in the unit the biology
                # happens in. Values sit exactly on `t_min + k * step`, which
                # is the viewer's discrete navigation grid.
                Dimension(
                    "Time",
                    unit="min",
                    display=False,
                    discrete=True,
                    step=step_min,
                    range=(t_min, t_max),
                ),  # step is AXIS_STEP_MIN; see why it is rounded, there
            ]
        )

        # AUTO, the house default, rather than the PRECISION this demo used to
        # ask for. PRECISION stores every channel as float32, which on a 4D node
        # of this size is ~60 bytes a splat before compression — a scene the
        # viewer has to stream. AUTO picks per-axis fixed point instead, and the
        # margins here are wide: 859 um of extent across uint16 is a 0.013 um
        # step against a 1.68 um voxel, and the time axis is on an exact grid,
        # which the encoder snaps to rather than quantizes.
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(cinematic_mode=True, tone_mapping="ACES"),
            )
            denoise_clause = (
                f", with connected components smaller than {MIN_COMPONENT_VOXELS} "
                "voxels removed from each timepoint before fitting"
                if MIN_COMPONENT_VOXELS >= 2
                else ""
            )
            scene.attrs["title"] = "GSplats: Zebrafish Gastrulation, 4D Time-Lapse"
            scene.attrs["description"] = (
                f"A living zebrafish embryo (cxcr4a morphant) imaged by confocal "
                f"laser-scanning microscopy through gastrulation: {n_timepoints} "
                f"timepoints over {t_max / 60:.1f} hours, {stacked.n_splats:,} "
                f"Gaussian splats in one 4D node with time as its fourth centre "
                f"column{denoise_clause}. The labelled endodermal cells start as a tight "
                f"cluster and spread over the yolk; the wireframe cage is the "
                f"imaged volume, ruled every {GRID_STEP_UM:.0f} um. Play the Time "
                f"slider to run the recording; press L for per-layer controls. "
                f"Zenodo 1211599, DOI 10.5281/zenodo.1211599, CC BY-SA 4.0."
            )

            scene.add_gsplats_from_data(
                "endoderm",
                stacked,
                lod_group=True,
                dim_order=["Z", "Y", "X", "Time"],
                extend_to_all=[],
                colormap=COLORMAP,
                opacity=VOLUME_OPACITY,
                absorption=VOLUME_ABSORPTION,
                blending_mode="volumetric",
                layer=True,
            )

            verts, colors, widths = cage_lines(bmin, bmax, GRID_STEP_UM)
            scene.add_lines(
                "acquisition cage",
                vertices=verts,
                widths=widths,
                colors=colors,
                line_type="segments",
                dim_order=["Z", "Y", "X"],
                # The cage is the instrument, not the specimen: it is stored
                # once and shown at every timepoint.
                extend_to_all=["Time"],
                blending_mode="additive",
                opacity=0.9,
                layer=True,
            )

            scene.add_text(
                "Zebrafish gastrulation",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            scene.add_text(
                f"{n_timepoints} timepoints over {t_max / 60:.1f} h\n"
                f"{stacked.n_splats:,} Gaussian splats\n"
                f"cage {bmax[2] - bmin[2]:.0f} x {bmax[1] - bmin[1]:.0f} x "
                f"{bmax[0] - bmin[0]:.0f} um, grid {GRID_STEP_UM:.0f} um",
                position=(0.02, 0.10),
                font_size=0.018,
                font="mono",
                color="white",
                line_height=1.45,
            )
            add_demo_caption(
                scene,
                "Confocal laser-scanning microscopy • endoderm label",
                DEMO_META.get("citation"),
            )

    aprint(f"Scene saved: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    """Resolve the 4D splats, build the scene, and optionally serve it."""
    aprint("=" * 70)
    aprint("GSplats Demo: 4D Zebrafish Embryo Time-Lapse")
    aprint("=" * 70)
    aprint("One 4D node • time as a real centre column • gridded acquisition cage")
    aprint("")

    output_path = get_demos_output_dir() / SCENE_NAME

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: launching viewer…")
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    scene_path = create_luxar_scene(load_or_build_gsplats(), output_path)

    if not NO_SERVE:
        aprint("\nLaunching viewer… (press L for the Layers panel)")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
