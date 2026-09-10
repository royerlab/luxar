#!/usr/bin/env python3
"""Data-Driven Demo: Human White-Matter Tractography (HCP-1065 dMRI atlas)

The wiring of the human brain, drawn as curves. Every one of the 87 named
white-matter tracts of the HCP-1065 population atlas is rendered as a bundle of
connected polylines in ICBM 2009a stereotaxic space, coloured by the standard
diffusion-MRI direction convention. The corpus callosum arches between the
hemispheres, the corticospinal tracts fan from the brainstem up into the motor
cortex, and the arcuate fasciculus hooks around the Sylvian fissure.

================================================================================
WHAT TRACTOGRAPHY IS
================================================================================
Diffusion MRI measures how water molecules diffuse in each voxel of the brain.
Inside a white-matter fibre bundle, water diffuses much more freely *along* the
axons than across them, so the diffusion profile points down the fibre. Chaining
those local orientations voxel to voxel — *tractography* — reconstructs the
long-range pathways that connect one region of cortex to another.

The result is not a picture of axons (an axon is ~1 um across; a voxel is ~1 mm).
It is a reconstruction of the dominant fibre geometry, at millimetre scale, and
it is the only non-invasive way to see the brain's structural wiring in a living
person.

THE DIRECTION COLOUR CONVENTION
-------------------------------
Every point is coloured by the absolute value of the local tangent direction,
the near-universal convention in the diffusion-imaging literature:

    RED   = left-right          (commissural fibres: the corpus callosum)
    GREEN = anterior-posterior  (association fibres: IFOF, ILF, arcuate)
    BLUE  = inferior-superior   (projection fibres: corticospinal tract)

So the colour is not decoration — the corpus callosum is red *because* it runs
left-right, and the corticospinal tract is blue *because* it runs up and down.
Reading the picture is reading the anatomy.

================================================================================
WHY THIS IS A LINES DEMO
================================================================================
Most volumetric science has to be *converted* into a Luxar geometry type. A
tractogram does not: a streamline is already a 3D polyline, so this is Luxar's
Lines geometry applied to data that is natively made of curves. Each of the 87
bundles becomes its own Lines node with ``layer=True``, so the Layers panel
(press **L**) gives per-tract visibility, display range, gamma and blending.

HOVER LABELS, ON EVERY LEVEL
----------------------------
The atlas names its bundles with terse codes (`AF_L`, `IFOF_R`, `DRTT_L`), which
are what the Layers panel shows. Hovering a streamline expands the code into the
full anatomical name plus a one-line gloss of what the tract does, e.g.

    AF_L — Arcuate fasciculus (left) · association · frontal (Broca) and
    temporal (Wernicke) language areas

Every LOD level carries the labels. The coarse levels are SUBSAMPLED STREAMLINES
of the same bundle (see LEVEL OF DETAIL below), so a fibre on a coarse level is
still a fibre of that tract and gets the same string — hover works from the
whole-brain opening pose, not only after zooming a bundle to half the screen.

Lines labels are stored *per vertex*, so a bundle holds one copy of its string
per vertex — 168,000 copies on the finest level at the defaults, plus a quarter
and a sixteenth of that on the two coarse levels. Labels here run ~124 bytes on
average (AF_L's is 107, the longest, C_PHP_L's, ~148), so a finest-level blob is
~18-25 MB raw. That is cheap in two of the three places it could hurt and
unavoidable in the third: 168,000 identical strings compress to tens of KB on
disk, and the viewer's label loader collapses a consecutive run of equal
labels to a single decoded string, so the tract name is retained once rather
than 168,000 times. The cost is the *fetch*: `write_labels_csr` chunks
`label_bytes` at 65,536 bytes, so one bundle is ~280-380 chunks (its ~18-25 MB
over 64 KiB, plus 3 for `label_offsets`) and its first hover issues that many
small requests, together a few tens of KB over the wire once compressed.
Round-trip count, not decode and not memory, is what you wait for — and across
87 bundles it is also ~30,000 extra files in the store, which is worth knowing
before `luxar export` writes them all to an offline folder or you put the
scene on static hosting.

LEVEL OF DETAIL — SUBSAMPLED STREAMLINES, NOT LIFTED SPLATS
-----------------------------------------------------------
Each tract is a `kind=lod` group of three Lines levels, selected on screen
occupancy (`selector="screen-area"`, finest anchored at half the screen, one
level coarser per halving of occupied area). A coarse level keeps a seeded
1-in-K subset of the bundle's STREAMLINES with their widths multiplied by K, so
the summed line width along every view ray — what `additive` blending
integrates — is the same at every level, and a coarse level looks like fewer
fibres of the same anatomy rather than like something else.

This replaces the `substitutive_lod=` default for Lines, which lifts every
segment to Gaussian "beads" and merges those into fewer, fatter splats. On the
hosted demo that coarse representation rendered the whole-brain opening pose as
a blown-out pastel blob with no fibre structure (2026-09-10): a merged
Gaussian's sigma follows the merged spread so it is fattest where the bundle is
sparse, the lift's bead count is set by arc length rather than segment count
(K had to be 256 before a coarse level was even smaller than the fine one), and
the sum-projection light rescale does not survive the 75x display attenuation
tuned for thin lines. Fewer fibres are the right coarse level for fibres; the
API version of this recipe is #2679 (the Lines twin of #2660 for Points).

RENDERING NOTE — `additive`, AND WHY NOT `normal`
-------------------------------------------------
A tractogram is a *solid* object: ~6.8M segments packed into a 180 mm skull,
many hundreds deep along any view ray. That makes the choice of blending mode
the single biggest visual decision in this demo, and it is not the obvious one.

`normal` (alpha-over) at opacity 1.0 looks right in a still: depthWrite is on,
so each pixel shows the nearest fibre and the near/far hemispheres separate
cleanly. But alpha-blended nodes render in the *transparent* pass, which THREE
sorts back-to-front **per object**, every frame. With 87 mutually-overlapping
bundles whose centroids interleave, that sort order flips as the camera orbits
and whole tracts visibly pop in front of each other. The still is fine; the
interaction is not, and this scene is meant to be orbited.

`additive` has no such failure mode, because addition is commutative: the frame
is the same whatever order the 87 nodes draw in, so there is nothing to sort and
nothing to pop. The cost is that it ignores depth, which is why the naive
settings blow out — the accumulated sum clips to white long before the far side
of the brain has been drawn.

The fix is to make each sample contribute *little*: opacity 0.24 and a display
window of [0, 74.976] (a ~75x attenuation on the colour). Hundreds of faint
samples then integrate into a glowing, X-ray-like volume in which the internal
architecture — the callosal fan, the arcuate's hook, the cerebellar peduncles —
is visible *through* the surface fibres rather than hidden behind them.

If you ever do want hard occlusion back without the popping, the mode to reach
for is `opaque`, not `normal`: it is depth-tested and depth-written but never
enters the transparent pass, so it is also order-independent.

DATA SOURCE & CITATIONS:
========================
Dataset:
--------
Source:  HCP-1065 population-averaged tractography atlas
         https://brain.labsolver.org/hcp_trk_atlas.html
Archive: hcp1065_avg_tracts_trk.zip (588 MB, 87 bundles in TRK format)
Space:   ICBM 2009a Nonlinear Asymmetric, 1 mm isotropic
Built:   automatic + augmented fibre tracking over 1,065 Human Connectome
         Project young-adult subjects
License: Creative Commons Attribution-ShareAlike 4.0 International

Citation:
---------
Yeh, F-C. Population-based tract-to-region connectome of the human brain and
its hierarchical topology. Nature Communications 13, 4933 (2022).
https://doi.org/10.1038/s41467-022-32595-4

Data were provided in part by the Human Connectome Project, WU-Minn Consortium
(Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657),
funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for
Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at
Washington University.

Because the source is ShareAlike, nothing derived from it is committed to this
repository — the atlas is downloaded and cached locally on first run.

Usage:
    python -m luxar.demos.demo_dmri_tractography
    python -m luxar.demos.demo_dmri_tractography --no-serve
    python -m luxar.demos.demo_dmri_tractography --recompute
    python -m luxar.demos.demo_dmri_tractography --keep-stale
    python -m luxar.demos.demo_dmri_tractography --per-bundle=3000 --points=20

Controls:
    - Press 'L' to open the Layers panel: one row per tract, 87 in total
    - Hover a tract for its full name and what it does — every LOD level
      carries the labels, so this works from the whole-brain opening pose
    - Ctrl+C to stop and cleanup
"""

from __future__ import annotations

DEMO_META = {
    "key": "dmri_tractography",
    "title": "Human White-Matter Tractography (HCP-1065)",
    "description": (
        "The human brain's 87 named white-matter tracts as "
        "directionally-coloured streamlines."
    ),
    "category": "medical",
    "geometry": "lines",
    "requirements": {
        "download_mb": 588,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["dmri_tractography"],
    "outputs": ["dmri_tractography"],
    "citation": {
        "short": "Yeh 2022",
        "doi": "10.1038/s41467-022-32595-4",
        "license": "CC BY-SA 4.0",
    },
}

import gzip
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import is_consolidated
from luxar.core.group.lod.group import coverage_fractions
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cached_download,
    demo_source_fingerprint,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
    require_module,
)
from luxar.demos._cinematic_camera import pull_in
from luxar.encoding import EncodingMode
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME: Final = "dmri_tractography"

ATLAS_URL: Final = (
    "https://github.com/data-others/atlas/releases/download/hcp1065/"
    "hcp1065_avg_tracts_trk.zip"
)
ATLAS_ZIP: Final = "hcp1065_avg_tracts_trk.zip"
ATLAS_ZIP_BYTES: Final = 587_869_457

#: The five anatomical divisions the atlas ships, in the order they are added to
#: the scene. These are the zip's top-level directories — note the space in
#: "cranial nerve", which is a real directory name and not a typo.
DIVISIONS: Final = (
    "association",
    "projection",
    "commissural",
    "cerebellum",
    "cranial nerve",
)

#: Atlas code -> (full anatomical name, one-line functional gloss).
#:
#: The bundle names and the code -> tract mapping follow the DSI-Studio HCP-1065
#: atlas documentation and Yeh, F-C. "Population-based tract-to-region connectome
#: of the human brain and its hierarchical topology", Nat. Commun. 13, 4933
#: (2022). The one-line glosses are standard neuroanatomy, not from either
#: source — several draw on the wider literature (the callosal fibre count, the
#: DRTT as the tremor target, the Broca/Wernicke framing of the arcuate).
#:
#: Keyed on the BASE code with any ``_L`` / ``_R`` hemisphere suffix stripped, so
#: 46 entries cover all 87 bundles; the pairing is applied by ``tract_label``
#: rather than duplicated here.
BUNDLE_INFO: Final[dict[str, tuple[str, str]]] = {
    # -- association -------------------------------------------------------
    "AF": (
        "Arcuate fasciculus",
        "frontal (Broca) and temporal (Wernicke) language areas",
    ),
    "C_FP": (
        "Cingulum, frontal-parietal segment",
        "dorsal cingulate bundle linking frontal and parietal cortex",
    ),
    "C_FPH": (
        "Cingulum, frontal-parahippocampal segment",
        "long cingulate arc from frontal cortex to the parahippocampal gyrus",
    ),
    "C_PH": (
        "Cingulum, parahippocampal segment",
        "temporal cingulum along the hippocampus; memory circuit",
    ),
    "C_PHP": (
        "Cingulum, parahippocampal-parietal segment",
        "retrosplenial link from the parahippocampal gyrus to parietal cortex",
    ),
    "C_PO": (
        "Cingulum, parolfactory segment",
        "subgenual cingulum curving under the genu of the corpus callosum",
    ),
    "EMC": (
        "Extreme capsule",
        "ventral fronto-temporal pathway between insula and claustrum",
    ),
    "FAT": (
        "Frontal aslant tract",
        "pre-SMA to inferior frontal gyrus; speech initiation and fluency",
    ),
    "IFOF": (
        "Inferior fronto-occipital fasciculus",
        "longest association bundle; frontal to occipital, semantic processing",
    ),
    "ILF": (
        "Inferior longitudinal fasciculus",
        "occipital to anterior temporal; object and face recognition",
    ),
    "MdLF": (
        "Middle longitudinal fasciculus",
        "superior temporal gyrus to parietal and occipital cortex; auditory and "
        "language",
    ),
    "PAT": (
        "Parietal aslant tract",
        "superior to inferior parietal lobule; vertical link across parietal "
        "association cortex",
    ),
    "SLF1": (
        "Superior longitudinal fasciculus I",
        "superior parietal to superior frontal; dorsal spatial and motor stream",
    ),
    "SLF2": (
        "Superior longitudinal fasciculus II",
        "inferior parietal to dorsolateral prefrontal; spatial attention",
    ),
    "SLF3": (
        "Superior longitudinal fasciculus III",
        "supramarginal gyrus to ventral premotor; articulation and praxis",
    ),
    "UF": (
        "Uncinate fasciculus",
        "hooks the anterior temporal lobe to orbitofrontal cortex; emotion and memory",
    ),
    "VOF": (
        "Vertical occipital fasciculus",
        "connects dorsal and ventral occipital visual streams",
    ),
    # -- cerebellum --------------------------------------------------------
    # DSI-Studio calls this member "Cerebellum"; expanded here because CB_L IS
    # the left cerebellar hemisphere and "Cerebellum (left)" answers "what is
    # this?" with "the cerebellum". The one deviation from atlas nomenclature in
    # this table. Note too that outside this atlas `CB` conventionally
    # abbreviates the CINGULUM BUNDLE, so the row is easy to misread.
    "CB": (
        "Cerebellar hemisphere",
        "hemispheric white matter (arbor vitae); coordination and timing of "
        "limb movement",
    ),
    "ICP": (
        "Inferior cerebellar peduncle",
        "chiefly olivocerebellar, spinal and vestibular input to the cerebellum",
    ),
    "MCP": (
        "Middle cerebellar peduncle",
        "pontine nuclei to cerebellar cortex; the cortico-ponto-cerebellar relay",
    ),
    "SCP": (
        "Superior cerebellar peduncle",
        "main cerebellar output to red nucleus and thalamus",
    ),
    "V": (
        "Vermis",
        "midline cerebellar fibres coordinating trunk and posture",
    ),
    # -- commissural -------------------------------------------------------
    "AC": (
        "Anterior commissure",
        "small ventral commissure joining the temporal lobes and olfactory regions",
    ),
    "CC": (
        "Corpus callosum",
        "the great commissure; ~200 million fibres joining the hemispheres",
    ),
    # -- cranial nerve -----------------------------------------------------
    # The roman numeral is left out of the name: the code already leads the
    # tooltip, so "CNII_L — Optic nerve (CN II) (left)" doubles the
    # parenthetical against the hemisphere qualifier.
    "CNII": (
        "Optic nerve",
        "retina to the optic chiasm; all visual input from one eye",
    ),
    "CNIII": (
        "Oculomotor nerve",
        "midbrain to most extraocular muscles; eye movement and pupil",
    ),
    "CNV": (
        "Trigeminal nerve",
        "face sensation and the muscles of mastication",
    ),
    "CNVII": (
        "Facial nerve",
        "facial expression, taste from the anterior tongue, lacrimation",
    ),
    "CNVIII": (
        "Vestibulocochlear nerve",
        "hearing and balance from the inner ear",
    ),
    # -- projection --------------------------------------------------------
    "AR": (
        "Acoustic radiation",
        "medial geniculate body to Heschl's gyrus, the primary auditory cortex",
    ),
    "CBT": (
        "Corticobulbar tract",
        "motor cortex to brainstem cranial-nerve nuclei; face, tongue, swallowing",
    ),
    # "<Region> corticopontine tract", not "Frontopontine tract": the code says
    # CPT, so the name expands it, and the shape then matches the CS_* / TR_*
    # siblings ("Anterior corticostriatal tract"). A "(frontopontine)" alias
    # would collide with the hemisphere qualifier the same way the cranial
    # nerves' roman numerals did.
    "CPT_F": (
        "Frontal corticopontine tract",
        "frontal cortex to pontine nuclei; cortico-ponto-cerebellar relay",
    ),
    "CPT_O": (
        "Occipital corticopontine tract",
        "occipital cortex to pontine nuclei; visual input to the cerebellum",
    ),
    "CPT_P": (
        "Parietal corticopontine tract",
        "parietal association cortex to pontine nuclei; somatosensory and "
        "visuospatial input to the cerebellum",
    ),
    "CST": (
        "Corticospinal tract",
        "motor cortex to spinal cord; the pyramidal tract's spinal component, "
        "carrying voluntary movement",
    ),
    "CS_A": (
        "Anterior corticostriatal tract",
        "prefrontal cortex to the caudate and putamen",
    ),
    "CS_P": (
        "Posterior corticostriatal tract",
        "posterior cortex to the striatum",
    ),
    "CS_S": (
        "Superior corticostriatal tract",
        "superior and motor cortex to the striatum",
    ),
    "DRTT": (
        "Dentato-rubro-thalamic tract",
        "cerebellar dentate nucleus to red nucleus and thalamus; the tremor target",
    ),
    "F": (
        "Fornix",
        "hippocampus to mammillary bodies and septum; the main hippocampal output",
    ),
    "ML": (
        "Medial lemniscus",
        "dorsal-column nuclei to thalamus; fine touch and proprioception",
    ),
    "OR": (
        "Optic radiation",
        "lateral geniculate nucleus to primary visual cortex; its anterior "
        "fibres detour through the temporal lobe as Meyer's loop",
    ),
    "RST": (
        "Reticulospinal tract",
        "brainstem reticular formation to spinal cord; posture and locomotion",
    ),
    "TR_A": (
        "Anterior thalamic radiation",
        "anterior and mediodorsal thalamus to prefrontal cortex",
    ),
    "TR_P": (
        "Posterior thalamic radiation",
        "pulvinar and posterior thalamus to parietal and occipital cortex",
    ),
    "TR_S": (
        "Superior thalamic radiation",
        "ventral thalamus to the pre- and postcentral gyri; sensorimotor relay",
    ),
}

#: Points per streamline after arc-length resampling. The source is sampled at
#: ~0.4 mm (a ~10 cm tract carries ~270 points), far finer than any rendered
#: line width; 28 points keeps every tract's curvature while cutting the vertex
#: count ~10x.
DEFAULT_POINTS: Final = 28

#: Streamlines kept per bundle. Two independent viewer limits bound this, and
#: the tighter one wins:
#:
#:   * ``scripts/check_demo_ladders.py`` fails any un-laddered lines leaf above
#:     200,000 vertices, so a node must stay under 200_000 / POINTS streamlines.
#:   * The element data texture holds 6 texels per segment, capping a node at
#:     ``682 * maxTextureSize`` segments — 2,793,472 on a 4096-class GPU. Over
#:     that the viewer silently clamps and the tail never renders.
#:
#: At 6,000 x 28 a node peaks at 168,000 vertices / 162,000 segments, clearing
#: both with margin. Only ~16 of the 87 bundles are large enough to be capped.
DEFAULT_PER_BUNDLE: Final = 6_000

#: The tighter of the two limits above: ``check_demo_ladders`` fails an
#: un-laddered lines leaf past this, so ``points x per_bundle`` must stay under
#: it. Exceeded only by an explicit ``--points`` / ``--per-bundle`` override.
LADDER_GATE_VERTICES: Final = 200_000

#: Deterministic subsample of the over-large bundles, so a rebuild is identical.
SUBSAMPLE_SEED: Final = 0

LINE_WIDTH: Final = 0.32  # mm; the brain is ~180 mm across
LINE_OPACITY: Final = 0.24
#: Gain on the per-vertex colour, i.e. a display window of ``[0, 74.976]``
#: (``intensity = 1 / (max - min)``, see the viewer's
#: ``rendering/display-range.ts``). A ~75x attenuation looks extreme written
#: down, but it is what additive compositing of a *solid* object needs: with
#: hundreds of segments along every view ray, each one may contribute only a
#: percent or so before the sum clips. Tuned in the Layers panel, then baked.
#:
#: NOTE: this must not be exactly 1.0. The line material compiles with a
#: ``LUXAR_NO_GOG`` define when gain/offset/gamma are all identity, which
#: strips the gain path out of the shader entirely — an authored 1.0 cannot
#: then be recovered at runtime.
LINE_INTENSITY: Final = 1.0 / 74.976

#: Substitutive LOD: each level replaces the finer one with FEWER STREAMLINES of
#: the same bundle, so zooming out costs less instead of drawing every fibre.
#: Three levels, 1-in-4 per step: a 6,000-streamline bundle ships 375 / 1,500 /
#: 6,000 streamlines. Coarse levels keep a seeded uniform subset of whole
#: streamlines (never a subset of vertices — that would break the joints) with
#: widths multiplied by the subsampling factor, so the summed width along any
#: ray is conserved and the additive integral matches across the LOD seam.
#:
#: Hand-built rather than `substitutive_lod=` (whose Lines coarse levels are
#: lifted Gaussian beads) — see LEVEL OF DETAIL in the module docstring for
#: why, and #2679 for the API that should eventually replace this recipe.
LOD_LEVELS: Final = 3
LOD_COMPRESSION: Final = 4
#: Hover-link template for every tract level (the demo link gate requires one
#: unshadowed module-level constant per template).
TRACT_LINK_TEMPLATE: Final = (
    "https://en.wikipedia.org/wiki/Special:Search?search={hover_key}"
)

# NOTE — `additive_lod=False` on every level is deliberate. This indexed layout
# qualifies for a composed ladder (every streamline is an ascending simple path,
# so the writer's fabricated per-component chain is faithful — see
# `indexed_components_are_chains`), but the shipped sizing would add three rungs
# to each of 87 already-small nodes (about 261 groups). Every level stays below
# the 200K-vertex un-laddered-leaf gate and the nodes already stream
# independently, so that metadata and traversal cost buys little. Keeping each
# level single-shot also leaves its per-vertex hover-label CSR directly on the
# level rather than moving it to an additive rung parent.

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
KEEP_STALE = FLAGS["keep_stale"]

#: Identifies the builder and Luxar writer that wrote the scene, alongside the
#: sizing knobs and authored schema version in :data:`SCENE_MARKER`.
FINGERPRINT: Final = demo_source_fingerprint(__file__)

POINTS_PER_STREAMLINE = parse_int_arg("points", DEFAULT_POINTS)
PER_BUNDLE = parse_int_arg("per-bundle", DEFAULT_PER_BUNDLE)

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / DEMO_NAME

#: What the scene currently on disk was built from. The output stem is fixed
#: (the registry declares one output per demo), so existence alone cannot tell a
#: default scene from one built with ``--points`` / ``--per-bundle``.
SCENE_MARKER: Final = CACHE_DIR / "scene_build.json"

#: Bumped whenever the scene's *content* changes in a way the sizing knobs do
#: not capture, so a warm output directory is rebuilt instead of silently
#: served. v2 added per-tract hover labels; a v1 scene has no ``version`` key
#: at all, which reads as stale and rebuilds.
#:
#: Note what this does and does not reach.
#: ``scripts/gallery/generate_gallery_datasets.py`` skips on the dataset
#: existing and never invokes the demo at all, so the bump cannot rescue a warm
#: gallery box on the default path — only ``--force`` gets there. What the bump
#: buys is that such a regen (and any plain re-run against a warm
#: ``datasets/demos/``) actually rebuilds, instead of the demo short-circuiting
#: on its own marker and re-serving the label-less scene.
SCENE_SCHEMA_VERSION: Final = 2

Arbol.max_depth = 4


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def check_sizing(points: int, per_bundle: int) -> str | None:
    """Validate the two sizing knobs, up front rather than 588 MB later.

    Args:
        points: ``--points``, vertices per resampled streamline.
        per_bundle: ``--per-bundle``, streamlines kept per tract.

    Returns:
        A warning to print when the pair exceeds the un-laddered leaf gate
        (still built — the override is deliberate), else ``None``.

    Raises:
        ValueError: For values that cannot produce geometry at all.
    """
    if points < 2:
        raise ValueError(
            f"--points must be >= 2 (a segment needs two ends), got {points}"
        )
    if per_bundle < 1:
        raise ValueError(f"--per-bundle must be >= 1, got {per_bundle}")
    if points * per_bundle > LADDER_GATE_VERTICES:
        return (
            f"--points {points} x --per-bundle {per_bundle} = "
            f"{points * per_bundle:,} vertices in the largest node, over the "
            f"{LADDER_GATE_VERTICES:,}-vertex un-laddered leaf gate: the scene "
            "will render but `hatch run check` will fail on it."
        )
    return None


def split_hemisphere(bundle: str) -> tuple[str, str]:
    """Split an atlas code into its base code and hemisphere word.

    Only ``_L`` / ``_R`` are hemisphere suffixes, and only when what remains is
    itself a known base code. That lookup guard is what makes the split safe:
    several base codes end in ``_F``, ``_O``, ``_P``, ``_A``, ``_S`` (``CPT_F``,
    ``CS_A``, ``C_PO``, ...), and ``CPT_F_L`` must resolve to ``CPT_F``, never to
    a naive "strip the last underscore group".

    Args:
        bundle: An atlas bundle code, e.g. ``"AF_L"``, ``"CPT_F_L"``, ``"MCP"``.

    Returns:
        ``(base_code, hemisphere)`` where ``hemisphere`` is ``"left"``,
        ``"right"``, or ``""`` for an unpaired (midline) bundle. ``base_code`` is
        ``bundle`` unchanged when no hemisphere suffix applies.
    """
    for suffix, side in (("_L", "left"), ("_R", "right")):
        if bundle.endswith(suffix) and bundle[: -len(suffix)] in BUNDLE_INFO:
            return bundle[: -len(suffix)], side
    return bundle, ""


def is_named_tract(bundle: str) -> bool:
    """Whether ``bundle`` has a :data:`BUNDLE_INFO` entry.

    The single source of truth for "will :func:`tract_label` fall back?", so the
    build-time coverage report and the label itself cannot drift apart.

    Args:
        bundle: An atlas bundle code, e.g. ``"AF_L"``.

    Returns:
        ``True`` when the label will be fully expanded.
    """
    return split_hemisphere(bundle)[0] in BUNDLE_INFO


def tract_label(bundle: str, division: str) -> str:
    """Hover label for one tract: code, full name, division, functional gloss.

    The Layers panel shows the raw atlas code, so the label leads with it and
    then expands it — the two surfaces name the same thing and read together.

    Args:
        bundle: The atlas bundle code, e.g. ``"AF_L"``.
        division: The anatomical division the bundle belongs to, one of
            :data:`DIVISIONS`.

    Returns:
        ``"AF_L — Arcuate fasciculus (left) · association · frontal (Broca) and
        temporal (Wernicke) language areas"``. An unrecognized base code (a
        future atlas revision) is NOT an error: it falls back to
        ``f"{bundle} — {division}"``, which ``build_scene`` detects via
        :func:`is_named_tract` and reports once.
    """
    base, side = split_hemisphere(bundle)
    info = BUNDLE_INFO.get(base)
    if info is None:
        return f"{bundle} — {division}"
    name, gloss = info
    hemi = f" ({side})" if side else ""
    return f"{bundle} — {name}{hemi} · {division} · {gloss}"


def tract_key(bundle: str) -> str:
    """The searchable anatomical name for a bundle, or ``""`` if unknown.

    Backs the click-through (#1917). The hover label leads with the atlas code
    and trails into division and gloss, so a search built from it would carry
    three fields nobody typed; this is the name alone — "Arcuate Fasciculus" —
    which is what an encyclopaedia can resolve.

    Hemisphere is deliberately dropped: left and right arcuate share one
    article, and "(left)" only narrows the search away from it.

    A bundle missing from :data:`BUNDLE_INFO` returns ``""``, and the viewer
    suppresses a link whose template has an empty substitution — so an
    unrecognised tract is simply not clickable, matching how
    :func:`tract_label` already degrades for it.

    Args:
        bundle: The atlas bundle code, e.g. ``"AF_L"``.

    Returns:
        The bundle's full anatomical name, or the empty string.
    """
    base, _side = split_hemisphere(bundle)
    info = BUNDLE_INFO.get(base)
    return info[0] if info is not None else ""


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Arc-length resample an ``(M, 3)`` streamline to exactly ``(n, 3)``.

    Interpolates each coordinate against the cumulative chord-length parameter,
    so the output points are evenly spaced *along the curve* rather than evenly
    spaced in the input index. That matters because tractography step sizes are
    not uniform across bundles: index-space resampling would bunch points where
    the tracker happened to take small steps.

    Endpoints are preserved exactly (the first and last parameter values are the
    ends of the interval), so a resampled tract still starts and stops where the
    original did.

    Args:
        points: ``(M, 3)`` streamline vertices, ``M >= 2``.
        n: Number of output points, ``>= 2``.

    Returns:
        ``(n, 3)`` float32 resampled streamline.

    Raises:
        ValueError: If ``points`` is not ``(M >= 2, 3)`` or ``n < 2``.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (M, 3), got {pts.shape}")
    if pts.shape[0] < 2:
        raise ValueError(f"points must have M >= 2, got {pts.shape[0]}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(step)])
    total = float(dist[-1])
    if total <= 0.0:
        # A degenerate zero-length streamline (all vertices coincident): every
        # output point is that single location. np.interp would divide by zero.
        return np.repeat(pts[:1], n, axis=0).astype(np.float32)

    want = np.linspace(0.0, total, n)
    out = np.empty((n, 3), dtype=np.float32)
    for axis in range(3):
        out[:, axis] = np.interp(want, dist, pts[:, axis])
    return out


def direction_colors(paths: np.ndarray) -> np.ndarray:
    """Per-vertex RGB from the local tangent direction (the dMRI convention).

    ``R = |dx|`` (left-right), ``G = |dy|`` (anterior-posterior),
    ``B = |dz|`` (inferior-superior), from the normalized central-difference
    tangent. The absolute value is what makes the encoding orientation-only:
    a tract looks the same whichever end you traced it from.

    Args:
        paths: ``(P, V, 3)`` resampled streamlines in **anatomical RAS** order
            — the colour convention is defined on anatomical axes, so this must
            be called before any scene-space axis remap.

    Returns:
        ``(P, V, 3)`` uint8 RGB.

    Raises:
        ValueError: If ``paths`` is not ``(P, V >= 2, 3)``.
    """
    arr = np.asarray(paths, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"paths must be (P, V, 3), got {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"paths must have V >= 2, got {arr.shape[1]}")

    # Central differences interior, one-sided at the two ends.
    tangent = np.empty_like(arr)
    tangent[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
    tangent[:, 0] = arr[:, 1] - arr[:, 0]
    tangent[:, -1] = arr[:, -1] - arr[:, -2]

    norm = np.linalg.norm(tangent, axis=2, keepdims=True)
    # A coincident-vertex pair gives a zero tangent; leave it black rather than
    # dividing by zero and painting a NaN.
    unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 0.0)
    return np.clip(np.abs(unit) * 255.0, 0.0, 255.0).astype(np.uint8)


def ras_to_scene(ras: np.ndarray) -> np.ndarray:
    """Rotate anatomical RAS into the viewer's Y-up scene convention.

    RAS is ``(+x right, +y anterior, +z superior)``; the viewer orbits about a
    Y-up axis. Mapping ``(x, y, z) -> (x, z, -y)`` puts superior on +Y, so the
    default orbit behaves and the brain is upright without a custom camera up.

    Args:
        ras: ``(..., 3)`` coordinates in RAS millimetres.

    Returns:
        ``(..., 3)`` float32 in scene space.
    """
    arr = np.asarray(ras, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"last axis must be 3, got {arr.shape}")
    return np.stack([arr[..., 0], arr[..., 2], -arr[..., 1]], axis=-1).astype(
        np.float32
    )


def polyline_segment_indices(n_paths: int, n_vertices: int) -> np.ndarray:
    """Indices joining consecutive vertices WITHIN each path (never across).

    Sharing a vertex at each joint lets the line material render a seamless join
    instead of two overlapping end-caps; excluding the path boundaries stops the
    last vertex of one streamline connecting to the first of the next.

    Returns:
        ``(2 * n_paths * (n_vertices - 1),)`` uint32, consecutive pairs.
    """
    if n_vertices < 2:
        raise ValueError(f"n_vertices must be >= 2, got {n_vertices}")
    base = (np.arange(n_paths, dtype=np.int64) * n_vertices)[:, None]
    starts = base + np.arange(n_vertices - 1, dtype=np.int64)[None, :]
    return np.stack([starts, starts + 1], axis=-1).reshape(-1).astype(np.uint32)


def subsample_indices(n_available: int, n_keep: int, *, seed: int) -> np.ndarray:
    """Sorted indices of ``n_keep`` streamlines drawn without replacement.

    Sorted so the kept streamlines stay in file order (readable diffs, stable
    caches), and seeded so a rebuild reproduces the same tract exactly.
    """
    if n_keep >= n_available:
        return np.arange(n_available, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_available, size=n_keep, replace=False))


def lod_streamline_counts(
    n_paths: int, *, levels: int = LOD_LEVELS, compression: int = LOD_COMPRESSION
) -> list[int]:
    """Streamlines kept per LOD level, coarsest first, finishing at ``n_paths``.

    ``n / K**(L-1), …, n / K, n`` with every entry at least 1; a level that would
    repeat its coarser neighbour (tiny bundles) is dropped, so the result is
    strictly increasing and may be shorter than ``levels``. A single-entry result
    means the bundle is too small to ladder at all.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    counts: list[int] = []
    for level in range(levels):
        count = max(1, n_paths // compression ** (levels - 1 - level))
        if level == levels - 1:
            count = n_paths
        if not counts or count > counts[-1]:
            counts.append(count)
    return counts


def brain_camera(radius: float) -> CameraConfig:
    """Left-lateral opening pose, the conventional view of a tractogram.

    Looks down the +X (right) axis at the left hemisphere, slightly raised and
    pulled forward, which is how these atlases are shown in the literature.
    """
    return CameraConfig(
        position=pull_in(
            (-2.6 * radius, 0.55 * radius, 0.85 * radius), from_fov_deg=38.0
        ),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
    )


# =============================================================================
# Data loading
# =============================================================================


def _download_atlas() -> Path:
    """Fetch (once) the 588 MB TRK atlas archive into the demo cache."""
    return cached_download(
        ATLAS_URL,
        DEMO_NAME,
        ATLAS_ZIP,
        expected_size=ATLAS_ZIP_BYTES,
    )


def _bundle_members(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Return ``(division, member_path)`` for every bundle, in DIVISIONS order.

    Within a division the bundles are sorted by name so the Layers panel lists
    them predictably (AF_L before AF_R before C_FP_L ...).
    """
    members = [n for n in archive.namelist() if n.endswith(".trk.gz")]
    out: list[tuple[str, str]] = []
    for division in DIVISIONS:
        prefix = f"{division}/"
        out.extend(
            (division, name)
            for name in sorted(m for m in members if m.startswith(prefix))
        )
    missing = set(members) - {m for _, m in out}
    if missing:
        raise ValueError(
            f"{len(missing)} bundle(s) outside the known divisions {DIVISIONS}: "
            f"{sorted(missing)[:3]}"
        )
    return out


def _load_bundle(raw_gz: bytes, *, per_bundle: int, points: int) -> tuple:
    """Decode one ``.trk.gz`` into resampled paths + direction colours.

    Returns ``(scene_xyz, rgb, n_kept, n_total)`` where ``scene_xyz`` is
    ``(n_kept * points, 3)`` float32 in scene space and ``rgb`` is the matching
    ``(n_kept * points, 3)`` uint8.
    """
    # Imported at the point of use, not in a preflight: a warm-cache run never
    # needs nibabel at all. The submodule inherits nibabel's tabled spec, so the
    # error message still advertises `nibabel>=5.0.0` and the `demos` extra.
    nib_streamlines = require_module("nibabel.streamlines")

    trk = nib_streamlines.TrkFile.load(
        io.BytesIO(gzip.decompress(raw_gz)), lazy_load=False
    )
    streamlines = trk.tractogram.streamlines  # RAS+ millimetres
    n_total = len(streamlines)
    if n_total == 0:
        raise ValueError("bundle contains no streamlines")

    keep = subsample_indices(n_total, per_bundle, seed=SUBSAMPLE_SEED)
    paths = np.empty((len(keep), points, 3), dtype=np.float32)
    for row, idx in enumerate(keep):
        paths[row] = resample_polyline(np.asarray(streamlines[int(idx)]), points)

    rgb = direction_colors(paths).reshape(-1, 3)
    scene_xyz = ras_to_scene(paths).reshape(-1, 3)
    return scene_xyz, rgb, len(keep), n_total


def load_or_build_bundles(*, per_bundle: int, points: int) -> dict:
    """Return the assembled per-bundle arrays, using the ``.npz`` cache if fresh.

    The cache key includes the two sizing knobs, so ``--per-bundle`` /
    ``--points`` sweeps do not collide with each other.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npz = CACHE_DIR / f"bundles_p{points}_n{per_bundle}.npz"

    if cache_npz.exists() and not RECOMPUTE:
        aprint(f"Using cached bundles: {cache_npz.name}")
        with np.load(cache_npz, allow_pickle=False) as data:
            names = [str(x) for x in data["names"]]
            return {
                "names": names,
                "divisions": [str(x) for x in data["divisions"]],
                "positions": [data[f"pos_{i}"] for i in range(len(names))],
                "colors": [data[f"rgb_{i}"] for i in range(len(names))],
            }

    zip_path = _download_atlas()
    names: list[str] = []
    divisions: list[str] = []
    positions: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    kept_total = 0
    source_total = 0

    with asection("Decoding 87 tract bundles"):
        with zipfile.ZipFile(zip_path) as archive:
            for division, member in _bundle_members(archive):
                bundle = Path(member).name.removesuffix(".trk.gz")
                xyz, rgb, n_kept, n_src = _load_bundle(
                    archive.read(member), per_bundle=per_bundle, points=points
                )
                names.append(bundle)
                divisions.append(division)
                positions.append(xyz)
                colors.append(rgb)
                kept_total += n_kept
                source_total += n_src
                capped = " (capped)" if n_kept < n_src else ""
                aprint(
                    f"{division}/{bundle}: {n_kept:,} of {n_src:,} streamlines{capped}"
                )

    aprint(
        f"{len(names)} bundles, {kept_total:,} streamlines kept of "
        f"{source_total:,}, {kept_total * points:,} vertices"
    )

    # Center on the atlas centroid so the brain sits at the origin.
    centroid = np.concatenate(positions).mean(axis=0)
    positions = [(p - centroid).astype(np.float32) for p in positions]

    payload = {f"pos_{i}": p for i, p in enumerate(positions)}
    payload.update({f"rgb_{i}": c for i, c in enumerate(colors)})
    payload["names"] = np.array(names)
    payload["divisions"] = np.array(divisions)
    np.savez_compressed(cache_npz, **payload)
    aprint(f"Cached bundles: {cache_npz.name}")

    return {
        "names": names,
        "divisions": divisions,
        "positions": positions,
        "colors": colors,
    }


# =============================================================================
# Scene
# =============================================================================


def _add_tract(
    group: Any,
    name: str,
    division: str,
    xyz: np.ndarray,
    rgb: np.ndarray,
    *,
    points: int,
    seed: int,
) -> None:
    """Write one tract as a ``kind=lod`` group of subsampled-streamline levels.

    Coarsest first. Every level is an indexed Lines node of WHOLE streamlines
    drawn without replacement (``subsample_indices``), widths scaled by the
    subsampling factor so the additive integral is conserved across the seam,
    and every level carries the tract's label/key so hover works at any zoom.
    A bundle too small to ladder is written as one flat Lines node with the
    same compositing.

    The two ``add_lines`` calls spell every keyword out (no ``**`` spreads): the
    corpus-wide element-cap gate reads geometry calls by AST and refuses spreads
    it cannot see through.
    """
    n_paths = len(xyz) // points
    label = tract_label(name, division)
    key = tract_key(name)
    counts = lod_streamline_counts(n_paths)

    def level_arrays(
        kept: np.ndarray,
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, int]:
        keep = (kept[:, None] * points + np.arange(points)[None, :]).ravel()
        # Width x subsampling factor: sum(width) along a ray is what additive
        # blending integrates, so the coarse level emits the light of the
        # fibres it stands in for.
        width = float(LINE_WIDTH * (n_paths / kept.size))
        return (
            xyz[keep],
            width,
            rgb[keep],
            polyline_segment_indices(int(kept.size), points),
            len(keep),
        )

    if len(counts) == 1:
        verts, width, cols, idx, n_verts = level_arrays(np.arange(n_paths))
        group.add_lines(
            name,
            vertices=verts,
            widths=width,
            colors=cols,
            indices=idx,
            line_type="indexed",
            labels=[label] * n_verts,
            keys=[key] * n_verts,
            link=TRACT_LINK_TEMPLATE,
            copy="{hover_key}",
            additive_lod=False,
            blending_mode="additive",
            opacity=LINE_OPACITY,
            intensity=LINE_INTENSITY,
            layer=True,
        )
        return

    # `layer=True` on the wrapper, not the levels: one Layers-panel row per
    # tract, and every level inherits the same look from here. `additive` —
    # see the module docstring's RENDERING NOTE: order-independent, so 87
    # mutually-overlapping nodes never pop as the camera moves.
    tract = group.add_lod_group(
        name,
        selector="screen-area",
        blending_mode="additive",
        opacity=LINE_OPACITY,
        intensity=LINE_INTENSITY,
        layer=True,
    )
    permutation = subsample_indices(n_paths, n_paths, seed=seed)
    for level, (count, cover) in enumerate(
        zip(counts, coverage_fractions(counts), strict=True)
    ):
        kept = np.sort(permutation[:count])
        verts, width, cols, idx, n_verts = level_arrays(kept)
        # `indexed`, NOT `segments`: interior joints must share a vertex index
        # or thick lines render as chains of beads. Lines labels are PER VERTEX
        # and every vertex of a bundle belongs to the same tract, so the one
        # tract string is broadcast across the level; the line picker reports a
        # SEGMENT slot fed unremapped into this per-vertex array, and every
        # entry holds the same string, so the tooltip is right regardless.
        # Click a tract to read about it, right-click to copy its name (#1917).
        tract.add_lines(
            f"child_{level}",
            vertices=verts,
            widths=width,
            colors=cols,
            indices=idx,
            line_type="indexed",
            labels=[label] * n_verts,
            keys=[key] * n_verts,
            link=TRACT_LINK_TEMPLATE,
            copy="{hover_key}",
            coverage_fraction=float(cover),
            additive_lod=False,
        )


def build_scene(bundles: dict, output_path: Path, *, points: int) -> Path:
    """Write the 87-node tractography scene."""
    names = bundles["names"]
    divisions = bundles["divisions"]
    positions = bundles["positions"]
    colors = bundles["colors"]

    extent = float(np.abs(np.concatenate(positions)).max())

    with asection("Writing scene"):
        # Reported BEFORE the write: a bundle missing from BUNDLE_INFO is a
        # table gap to fix, and saying so after 87 nodes are already on disk
        # (and after finalize's banner) buries it.
        unknown = [n for n in names if not is_named_tract(n)]
        aprint(f"Hover labels: {len(names) - len(unknown)}/{len(names)} tracts named")
        if unknown:
            aprint(
                f"WARNING: {len(unknown)} bundle(s) missing from BUNDLE_INFO, "
                f"labelled by code only: {', '.join(sorted(unknown))}"
            )

        dims = Dimensions(
            [
                Dimension("x", unit="mm", display=True),
                Dimension("y", unit="mm", display=True),
                Dimension("z", unit="mm", display=True),
            ]
        )
        with LuxarZarrCompiler(output_path, encoding_mode=EncodingMode.PRECISION) as c:
            scene = c.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # Thin lines lose detail at CSS resolution; see ViewerConfig.allow_high_dpr.
                    allow_high_dpr=True,
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    camera=brain_camera(extent),
                ),
                citation=DEMO_META["citation"],
            )
            scene.attrs["title"] = (
                "Human White-Matter Tractography — HCP-1065 population atlas"
            )

            groups = {d: scene.add_group(d.replace(" ", "_")) for d in DIVISIONS}
            total_segments = 0

            for bundle_idx, (name, division, xyz, rgb) in enumerate(
                zip(names, divisions, positions, colors, strict=True)
            ):
                n_paths = len(xyz) // points
                total_segments += n_paths * (points - 1)
                _add_tract(
                    groups[division],
                    name,
                    division,
                    xyz,
                    rgb,
                    points=points,
                    seed=1_000 * bundle_idx,
                )

            scene.add_text(
                "Human White-Matter Tractography",
                position=(0.02, 0.02),
                font_size=0.042,
                anchor="top-left",
                color="rgba(255,255,255,0.75)",
                blend_mode="difference",
            )
            add_demo_caption(
                scene,
                "HCP-1065 atlas (CC BY-SA 4.0) — 87 tracts",
                DEMO_META.get("citation"),
            )
            # Tells the viewer that hover does something — labels ride every
            # LOD level, so it does from the opening pose. Bottom-left is the
            # only free corner: title top-left, credit bottom-right, and the
            # auto-injected hover box top-right.
            scene.add_text(
                "Hover a tract to identify it",
                position=(0.02, 0.97),
                font_size=0.015,
                anchor="bottom-left",
                color="rgba(200,200,220,0.5)",
            )
            # NO explicit hover overlay, deliberately: the compiler
            # auto-injects one top-right as soon as a node carries labels, and
            # that corner is the free one. Do not "improve" it to the left or
            # top-left — the control rail and Layers panel are fixed DOM at
            # z-index 1000/1500, far above the overlay layer's 5, so a hover
            # box there is occluded.

        aprint(f"{len(names)} lines nodes, {total_segments:,} segments")
        aprint(f"Scene saved: {output_path}")
    return output_path


def scene_marker_matches(marker: Path, *, points: int, per_bundle: int) -> bool:
    """Whether the scene on disk was built with these knobs AND this schema.

    A missing, unreadable or stale marker reads as "no", which costs one
    rebuild — the safe direction. The alternative (existence alone) silently
    serves ``--points 20`` geometry to a later default run, and vice versa, or
    keeps serving a scene built before a content change such as hover labels.
    """
    try:
        record = json.loads(marker.read_text())
    except (OSError, ValueError):
        return False
    return (
        record.get("builder") == FINGERPRINT
        and record.get("version") == SCENE_SCHEMA_VERSION
        and record.get("points") == points
        and record.get("per_bundle") == per_bundle
    )


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene path, regenerating on a fresh system.

    Reuse is keyed on this builder's fingerprint, the sizing knobs and
    :data:`SCENE_SCHEMA_VERSION` as well as on the file existing: the scene path
    is fixed, so nothing else distinguishes a scene built at the defaults from
    one built with ``--points`` / ``--per-bundle``, or from one built before a
    source or schema change.
    """
    if (
        output_path.exists()
        and is_consolidated(output_path)
        and not RECOMPUTE
        and (
            KEEP_STALE
            or scene_marker_matches(
                SCENE_MARKER, points=POINTS_PER_STREAMLINE, per_bundle=PER_BUNDLE
            )
        )
    ):
        aprint(f"Using existing scene: {output_path}")
        return output_path

    bundles = load_or_build_bundles(per_bundle=PER_BUNDLE, points=POINTS_PER_STREAMLINE)
    scene_path = build_scene(bundles, output_path, points=POINTS_PER_STREAMLINE)
    SCENE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SCENE_MARKER.write_text(
        json.dumps(
            {
                "builder": FINGERPRINT,
                "version": SCENE_SCHEMA_VERSION,
                "points": POINTS_PER_STREAMLINE,
                "per_bundle": PER_BUNDLE,
            }
        )
    )
    return scene_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    aprint("=" * 70)
    aprint("Demo: Human White-Matter Tractography — HCP-1065 dMRI atlas")
    aprint("=" * 70)

    output_path = get_demos_output_dir() / f"{DEMO_NAME}.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint("No scene found. Run without --serve-only first.")
        return

    warning = check_sizing(POINTS_PER_STREAMLINE, PER_BUNDLE)
    if warning:
        aprint(f"WARNING: {warning}")

    scene_path = load_or_build_scene(output_path)

    if NO_SERVE:
        aprint(f"Scene ready at {scene_path}")
    else:
        aprint("Data credit: HCP-1065 tractography atlas (Yeh 2022, CC BY-SA 4.0)")
        aprint("Press 'L' in the viewer for per-tract visibility.")
        aprint(
            "Fly the camera into the tractogram (or zoom a bundle to fill the "
            "view), then hover a tract to see its name and function."
        )
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
