#!/usr/bin/env python3
"""Demo: Biodiversity at Planetary Scale — GBIF occurrences + animal migrations.

Every dot is a real recorded encounter between a human and another species: a
bird counted at dawn, a beetle pinned in a drawer in 1912, a fungus photographed
on a phone. GBIF pools 3.7 **billion** of them. This demo draws a stratified,
license-clean sample of that record onto a Blue Marble globe, and threads
through it the actual paths of tracked animals — white storks flying Europe to
the Sahel, turkey vultures the length of the Americas, blue and humpback whales
crossing ocean basins.

Two of Luxar's four geometry types, plus two non-displayed dimensions:

  * **Occurrences (Points)** — GBIF records placed on the globe from
    ``decimallatitude`` / ``decimallongitude``, coloured by major taxonomic
    group.
  * **Migrations (Lines)** — CC0 Movebank tracks as great-circle-resampled
    polylines whose **time coordinate advances along the chain**.
  * **taxon** and **period** are non-displayed categorical dimensions, so
    ``[`` / ``]`` scrubs the sample by taxonomic group and by decade. Each has a
    leading "all" slot (``All life`` / ``All years``) where the always-on summary
    layers live, and the scene opens there.

================================================================================
WHAT THIS SHOWS THAT A DENSITY MAP CANNOT
================================================================================

GBIF's own web maps, and every other planetary-scale rendering of this dataset
(GBIF's density tiles, GBIF Globe's H3 hexes), **aggregate before they draw** —
they ship pre-binned counts, never the records. This demo ships the records. The
interesting consequence is that the dataset's own pathologies stay visible
instead of being smoothed away: you can see roads and rail lines in the
observation record, national-park boundaries, the hard edge of a country's
recording effort at its border, and grid lattices where coordinates were rounded.

================================================================================
SCALING: WHERE LOD HELPS, AND WHERE IT MUST NOT BE USED
================================================================================

A single Points node holds at most ``floor(4096/3) * maxTextureSize`` points —
**5,591,040** on a 4096-class GPU — and overflow is a *silent clamp*, not an
error. So every large layer is split into spatial BSP tiles. Tiling alone does
not bound cost, though: a ``kind=partition`` group renders **every** part (the
GPU only frustum-culls at draw), and a ``stream:`` additive ladder is
*progressive*, so it converges to 100% of the layer regardless of camera
distance. Built that way, whole-globe framing held all **18.1M** elements.

The occurrence cloud therefore uses a ``kind=partition`` wrapper whose every
child is a per-tile ``kind=lod`` ladder — the Points counterpart of the gsplat
``adaptive`` recipe. ``coverage_fraction`` is evaluated per tile against that
tile's own screen size and non-default levels are lazily attached, so a tile that
is small on screen never fetches its fine levels. Whole-globe framing holds
~0.2M of its 15M; zooming a tile walks it to the full 1,875,000-point level while
its neighbours stay coarse.

**The globe deliberately gets none of that.** A textured shell cannot survive
Gaussian merging: at K=4/levels=2 the coarsest level is 46k merged splats per
750k-point tile, and those coarse ellipsoids cannot reproduce a continuous
textured shell — the planet becomes a pile of blobs. Coarse levels are
meaningful for a diffuse point cloud (they read as *density*, which is exactly
what the occurrence layer wants) and meaningless for a continuous surface. The
globe is instead a fixed-resolution backdrop, sized (``N_GLOBE``) so it never
needs reducing.

**Calibrating the occurrence thresholds** took two measured corrections, both
worth knowing before reusing the recipe. The default
``coverage_fraction = sqrt(N_i/N_finest)`` is calibrated for a *single* lod group
that fills the screen; split into T tiles, each tile's projected diagonal at
whole-globe framing is only ~0.6 of the viewport diagonal, which against the
default ladder still selects a mid level. And a threshold placed *on* that
measured metric makes the tiles flap: two levels stay simultaneously visible,
cross-faded, both resident, because the selector's hysteresis is 10% and
downgrade-only. The metric also varies per tile (nearer tiles project larger), so
thresholds must clear the *largest* per-tile value, not the mean.

PERSISTENT CONTEXT vs. SLICED SELECTION
--------------------------------------
The scene mixes two kinds of layer, and they use two different mechanisms.

**Context — ``extend_to_all``.** The globe must survive every scrub, or the moment
you move a slider the selected records are left floating in black with no
geographic reference. That is exactly what ``extend_to_all`` is for, and it was
broken: a node whose extended dims covered
all non-displayed dims was never queried at all (royerlab/luxar#1157 — arrays
never fetched, ladder frozen, gsplat levels of a ``substitutive_lod`` filtered
out entirely). This demo carried a workaround for it: a 25k-point globe
replicated into all 139 slots, 3.5M elements to show 25k coarse points at a time.

**#1157 is fixed** (a fully-extended node now derives a slice-invariant query and
flows through the normal load path), verified here before the workaround was
removed:

* the issue's own control scene now loads its extended layer to the full
  1,200,000 points and holds it byte-identically across every scrub, where it
  previously fetched *nothing*;
* an ``extend_to_all`` + ``substitutive_lod`` partition-of-LOD holds 318,686
  elements identically at every ``(taxon, period)`` combination with the coarse
  **gsplat** level intact and no "filtered out during nD->3D processing"
  warnings.

So the globe is now ONE ``extend_to_all`` layer of ``N_GLOBE`` points at a fixed
resolution, and the replicated context globe is gone.

Only the globe is extended. Extending the always-on track layer as well was tried
and reverted: 105,662 always-visible ribbons buried the selection (7,283 fish
records vanished under them). Persistent context has to be quiet enough to sit
*behind* the thing being selected.

**Selection — real coordinate slots.** The scrubbable layers do the opposite: they
carry real ``(taxon, period)`` coordinates so scrubbing isolates. Both dimensions
get a leading "all" category, and **every reachable slot is materialised** — the
9 taxon marginals, the 13 period marginals, and all 117 joint cells (139 slots).
That is not an ``extend_to_all`` matter but a consequence of the viewer showing
the *intersection* of the non-displayed slices: a layer stored only at
``(real taxon, real period)`` occupies just the joint cells, so moving ONE slider
lands on a slot holding nothing.

Cell density drove two further choices, both measured rather than assumed:

* **Decades, not years.** Censused on the real 1.44M scrubbable sample at year
  granularity: 1,132 of 1,143 cells populated, but the median populated cell held
  **244 points** and 688 held under 500 -- ``Fishes``/``1961`` was 147 points
  scattered over a globe. By decade: 108 of 108 cells populated, median **2,390**.
  Ten times denser for a tenth the slider steps. Categorical rather than a
  discrete decade axis because an "all" step and decade steps cannot share one
  numeric step grid.
* **Stratify the reservoirs over the cross product during the read.** Sampling
  joint cells from a per-taxon reservoir instead gave ``Birds``/``1960s`` just
  258 points: bird records are overwhelmingly recent eBird, so a uniform sample of
  the taxon barely touches an old decade. With one reservoir per slot every joint
  cell fills to its own cap. The marginals keep their own uniform-per-taxon and
  uniform-per-period reservoirs -- deriving them from the period-balanced cells
  would over-weight sparse decades and misrepresent the taxon.

================================================================================
APPEARANCE — TUNED IN THE LAYERS PANEL, THEN BAKED
================================================================================

The rendering settings were found interactively in the viewer's Layers panel and
then transcribed, rather than guessed. Reading them back matters because the
panel's DISPLAY RANGE ``[lo, hi]`` is a **window, not a gain**:
``intensity = 1/(hi-lo)`` and ``offset = -lo/(hi-lo)``
(``packages/luxar-viewer/src/rendering/display-range.ts``).

**Absorption and brightness are a coupled pair.** Raising ``absorption`` is what
turns a cloud of points into something that reads as an opaque *material* — and
one that can still be made slightly transparent on demand, which a truly opaque
mode cannot. But absorption also makes a layer very dim, almost black. The fix is
to push brightness up in the same move by lowering the display-range max, which
raises ``intensity``. The globe showed this: tuned as ``absorption=10`` with a
display max of 0.041 (a 24x gain), then re-tuned to a 0.205 window (a 4.88x gain)
once the switch to ``opaque`` removed the absorption. The gains land almost
exactly 5x apart (24.39 / 4.88 = 5.0), a neat near-exact coincidence — but the
switch to ``opaque`` also changed the compositing (now unblended and
opacity-independent), not only the absorption term, so read it as a useful
mnemonic rather than a proof that the whole gain was absorption.

**A backdrop cannot be a transparent mode, though.** The globe ultimately ships
``opaque`` rather than the ``volumetric`` it was tuned to, because a backdrop
needs two things no transparent mode provides: to be drawn BEFORE the data, and
to occlude the far hemisphere. See ``GLOBE_BLENDING`` for the measured draw order
that forced this and for royerlab/luxar#1227 (since resolved). The tuned
brightness survives (``intensity`` and ``gamma`` still apply); the absorption
term does not.

The scrubbable records layer takes the opposite treatment — ``opaque``
(depth-tested, unblended: ``transparent: false`` disables framebuffer blending,
so the fragment's emitted alpha is discarded and ``opacity`` has no effect) with
a hard brightness push — so it reads as crisp discrete marks sitting *on* the lit
globe. An additive selection washes out over a bright surface.

Two things that only show up on a real build:

* ``intensity`` is capped at 100 by ``validate_intensity``; the panel reading of
  250 fails the build. It costs nothing here, because both saturate — the dimmest
  taxon colour channel clips to 1.0 by intensity ~8.
* Only ``opacity`` propagates from a ``kind=lod`` group to its children's
  materials (``intensity``/``gamma`` leave the child uniforms at 1). Compositing
  attrs therefore ride on the node marked ``layer=True`` — the partition wrapper —
  which is the node the Layers panel reads and pushes down to every descendant.

The records also had to be lifted clear of the shell (``OCCURRENCE_LIFT``). With
``opaque`` blending the depth test makes an intersecting dot wink in and out along
the terrain, and the globe's rendered footprint is wider than its nominal point
radius.

================================================================================
TIME VARIES ALONG EACH MIGRATION WORLDLINE
================================================================================

Track vertices carry the **decade** of their originating fix, and that value
advances *along* the polyline — which no other 4D Lines demo in this repo does
(they all hold ``t`` constant per chain). It works: the Liang-Barsky clipper
draws a segment straddling the slab **clipped** rather than dropping it, so a
decade-boundary segment reads as a whisker that grows and shrinks as you scrub.

Verified arithmetically on a 40-track prototype: at one slice, 520 of 14,840
segments = 440 within-slice (40 tracks x 11) + 80 straddlers, exactly two per
track. Then on real data: 6,853 track segments at one slice, 3,345 at another,
0 in a decade before the tags existed.

Binning to a whole decade rather than a fractional year is deliberate. The
compiler warns that off-grid discrete values are only fetched within a
**quarter**-step of the grid while membership is a half-step, so off-grid
vertices can sit in chunks that are never requested — it would work on a small
scene and fail silently on a large one. An A/B in the browser showed on-grid and
off-grid rendering *identically* (520 segments vs 520), so there was nothing to
lose.

================================================================================
DATA HYGIENE — WHAT IS FILTERED AND WHY
================================================================================

GBIF at scale is not clean, and the mess is well characterised (all figures
measured against the live index, 2026-08):

* **Aves is 60-62% of all GBIF records** and eBird alone is ~45%, so any
  uniform sample is mostly bird checklists. The always-on layer keeps whatever
  proportion the sampling actually produces rather than reweighting, and the
  scrubbable ``By taxon & period`` layer caps each group instead, so rare groups
  stay explorable.

  Note carefully what the resulting number is and is not. Measured on the
  default 250-part read of the 2026-08-01 snapshot: 97.9M rows read, 76.1M
  passed the filters, and of those **73.0% are birds** -- higher than GBIF's
  own ~60%. That is not a sampling error: the filters here are *not
  taxon-neutral*. Requiring a coordinate, a year >= 1900, a mappable
  kingdom/phylum/class, an uncertainty under 100 km, and a non-NonCommercial
  license all favour exactly the large, recent, well-georeferenced
  bird-observation datasets. The honest claim is therefore "73% of these
  license-clean georeferenced records are birds", not "73% of GBIF is birds".
  For reference, the rest of that measured composition: flowering plants
  10.3%, insects 5.1%, fishes 3.7%, other animals 3.4%, other plants/algae/
  microbes 2.6%, fungi 1.0%, mammals 0.5%, reptiles & amphibians 0.4%.
* **Only ~38% of coordinate pairs are unique**, and 15% of records carry
  ``COORDINATE_ROUNDED``. Raw plotting stacks hundreds of records on one pixel
  and paints visible lattices. Each record is therefore jittered by its own
  ``coordinateuncertaintyinmeters`` (floored at 300 m, capped at 25 km).
* **Null island** (``lat=lon=0``, 1.42M records) is dropped.
* **Country centroids** are *not* flagged by GBIF. Records with an uncertainty
  above 100 km are dropped, which removes most of them cheaply.
* **License is a per-record column.** ``CC_BY_NC_4_0`` (16.8%) is excluded, so
  the sample is CC_BY_4_0 + CC0_1_0 only. The AWS registry labels the whole
  snapshot CC-BY-NC because that is its most restrictive member; filtering the
  column is the defensible read.
* Records whose kingdom/phylum/class cannot be mapped to one of the nine groups
  are dropped (the count is reported).

Geographic imbalance is *not* corrected: North America and Europe really are
~76% of all georeferenced records, and that is the single most important thing
the map has to say.

================================================================================
SELF-CONTAINED / REGENERATING (no LFS asset)
================================================================================
Ships only code. The first run reads a few hundred parquet parts straight out of
GBIF's AWS Open Data snapshot (anonymous, no account, ~200 MB on the wire thanks
to column projection), downloads ~124 MB of CC0 tracks and a 5 MB NASA texture,
then builds the scene. Sources are cached under
``~/.cache/luxar/biodiversity_planetary_scale/``; ``--recompute`` rebuilds.

DATA SOURCES & CITATIONS
------------------------
GBIF Occurrence Snapshot on AWS Open Data (Registry of Open Data on AWS),
    ``s3://gbif-open-data-us-east-1/occurrence/<snapshot>/``. Each snapshot
    carries its own DOI in ``citation.txt``, which this demo fetches and prints.
    Records here are filtered to CC_BY_4_0 and CC0_1_0.
    https://github.com/gbif/occurrence/blob/master/aws-public-data.md
    GBIF.org — https://www.gbif.org/citation-guidelines
Movebank Data Repository (all CC0 1.0, DOI'd, no account required):
    Bildstein K. et al. — Turkey vultures in North and South America.
        https://doi.org/10.5441/001/1.46ft1k05
    Fiedler W., Wikelski M. et al. — MPIAB Argos white stork tracking 1991-2017.
        https://doi.org/10.5441/001/1.k29d81dh
    Mate B. — Blue whales Eastern North Pacific 1993-2008 Argos.
    Australia east-coast humpback whales. https://doi.org/10.5441/001/1.294
NASA Blue Marble: Next Generation ("land_shallow_topo"), NASA Earth Observatory
    (Reto Stockli). Public domain. https://visibleearth.nasa.gov/

USAGE
-----
    luxar demo run biodiversity_planetary_scale
    python demo_biodiversity_planetary_scale.py [--n-points N] [--n-parts K]
        [--recompute] [--no-serve] [--serve-only]

Controls:
    Mouse drag rotate, scroll zoom, right-drag pan.
    '1' selects the taxon dimension, '2' the period dimension; '[' / ']' step.
"""

from __future__ import annotations

DEMO_META = {
    "key": "biodiversity_planetary_scale",
    "title": "Biodiversity at Planetary Scale",
    "description": (
        "GBIF occurrence records and CC0 animal-migration tracks on a Blue "
        "Marble globe, sliceable by taxon and period."
    ),
    "category": "geoscience",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 330,
        "compute": "heavy",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["biodiversity_planetary_scale"],
    "outputs": ["biodiversity_planetary_scale"],
}

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.group.compositing import position_bounds_from_array
from luxar.core.group.partition import median_bsp_partition
from luxar.core.viewer_config import (
    CameraConfig,
    DimensionsConfig,
    ViewerConfig,
)
from luxar.demos import require_module, substitutive_lod_or_flat
from luxar.encoding import EncodingMode
from luxar.utils.demos import (
    cache_computed,
    cached_download,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
    print_data_provenance,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME: Final = "biodiversity_planetary_scale"
CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / DEMO_NAME

R_EARTH_KM: Final = 6371.0
RADIUS: Final = 100.0  # globe radius in scene units

# --- GBIF snapshot -----------------------------------------------------------
GBIF_BUCKET: Final = "gbif-open-data-us-east-1"
GBIF_REGION: Final = "us-east-1"
# Pinned for reproducibility; falls back to the newest listed snapshot (with a
# notice) once GBIF prunes this one. Snapshots are monthly, YYYY-MM-01.
GBIF_SNAPSHOT: Final = "2026-08-01"
GBIF_COLUMNS: Final = [
    "decimallatitude",
    "decimallongitude",
    "kingdom",
    "phylum",
    "class",
    "year",
    "license",
    "coordinateuncertaintyinmeters",
    "datasetkey",
]
# Measured on the 2026-08-01 snapshot: 9,705 parts, 3.72e9 rows, mean 379k
# rows/part. Measured over 40 random parts with THIS demo's filter set
# (coordinates + license + year window + mappable taxon + uncertainty):
# 16,227,558 rows read -> 12,273,021 kept = 75.6%, i.e. ~307k kept per part,
# at 3.1 parts/s on 48 threads. Rounded down for headroom.
GBIF_USABLE_ROWS_PER_PART: Final = 300_000
# Read this many times more records than the point target. Almost all of this
# is bought for DIVERSITY, not headroom: at 40 parts the sample read 82.8%
# Aves and 0.55% Insecta against GBIF's true ~60% / ~8.6%, because a handful of
# huge single-publisher eBird parts dominate a narrow read.
OVERSAMPLE_FACTOR: Final = 5.0
GBIF_READ_THREADS: Final = 48
#: The ONLY licenses kept. An allowlist, deliberately, not a blocklist that
#: rejects `CC_BY_NC_4_0`: `_dictionary_codes` maps a null or unrecognised value
#: to UNMAPPED, so a blocklist silently retains every record whose license GBIF
#: did not populate or spells differently — and this demo claims, in its
#: docstring and on screen, that the sample is CC BY 4.0 + CC0 only. A claim
#: about licensing has to be enforced by construction.
#:
#: On the 2026-08-01 snapshot the two forms happen to agree exactly (76,057,458
#: rows either way): GBIF populates `license` with one of its three values for
#: every record, so there is nothing for a blocklist to miss *today*. That is
#: precisely why the blocklist was worth replacing -- it was correct by luck.
ALLOWED_LICENSES: Final = ("CC0_1_0", "CC_BY_4_0")
MAX_COORD_UNCERTAINTY_M: Final = 100_000.0  # drops most country centroids
JITTER_FLOOR_M: Final = 300.0
JITTER_CEIL_M: Final = 25_000.0

YEAR_MIN: Final = 1900
YEAR_MAX: Final = 2026

# --- Sampling ----------------------------------------------------------------
DEFAULT_N_POINTS: Final = 15_000_000
SAMPLE_SEED: Final = 20260803

# --- Scene / rendering -------------------------------------------------------
#: The globe carries NO substitutive LOD, and is sized so it does not need one.
#:
#: A textured shell cannot survive Gaussian merging. At K=4/levels=2 the coarsest
#: level is 46k merged splats per 750k-point tile, and those coarse ellipsoids
#: cannot reproduce a continuous textured shell -- the planet becomes a pile of
#: blobs. Coarse levels are meaningful for a diffuse point cloud (they
#: read as density, which is the `All life` layer's whole point) and meaningless
#: for a continuous surface.
#:
#: So the globe is a fixed-resolution backdrop: partition + a `stream:` ladder for
#: fast first paint, and no view-dependent reduction at all. 700k keeps it always
#: resident inside the ~1M whole-globe budget (700k + ~210k occurrences + ~106k
#: tracks) while still giving 0.24-degree spacing -- about 27 km at Earth scale.
N_GLOBE: Final = 700_000
#: ~0.78x the mean point spacing, the ratio that seals the shell without
#: over-drawing (0.29 spacing at 700k points on a radius-100 globe).
GLOBE_RADII: Final = 0.23
#: The Blue Marble texture is multiplied down HARD. It has to be: the globe is a
#: sealed shell, so at full brightness the continents (bright green
#: Europe, tan Sahara, saturated blue ocean) carry more contrast than the data
#: drawn on top of them and the occurrence colours stop reading as data at all.
#: Baked colour dimming is the lever that drops the texture's contrast so the
#: data reads on top. Layer `opacity` is not even an option here: under `opaque`
#: it is inert (`transparent: false` disables framebuffer blending, so the
#: fragment's emitted alpha is discarded), so it can neither dim the shell nor
#: make it translucent.
#:
#: 0.12, not the 0.20 that first looked right: 0.20 was chosen against a sparse
#: early globe, and once the shell sealed at N_GLOBE it was bright enough that a
#: 7,283-record selection stopped reading against it.
GLOBE_DIM: Final = 0.12
OCCURRENCE_RADII: Final = 0.062
OCCURRENCE_OPACITY: Final = 0.75
#: Baked colour scale for the occurrence records. KEPT at 0.85 even though the
#: display-range window now supplies the gain: the DISPLAY RANGE values below
#: were tuned in the Layers panel against a scene whose colours already carried
#: this scale, so changing it would invalidate them.
OCCURRENCE_COLOR_SCALE: Final = 0.85

# --- Appearance, transcribed from the Layers panel -------------------------
# Found interactively in the viewer and then baked, rather than guessed. The
# panel's DISPLAY RANGE [lo, hi] is a WINDOW, not a gain:
# `intensity = 1/(hi-lo)`, `offset = -lo/(hi-lo)` (rendering/display-range.ts).
# Both layers window from 0, so `offset` stays 0 and only `intensity` is needed.
#
# THE GENERAL RULE, worth carrying to any volumetric layer: absorption and
# brightness are a COUPLED PAIR, and pushing absorption alone is a trap.
# Raising absorption is what turns a cloud of points into a material that reads
# as an opaque surface (and can still be made slightly transparent on demand) --
# but it also makes the layer very dim, almost black. The fix is to push
# brightness up by the same move: LOWER the display-range max, which raises
# `intensity`. The globe was originally tuned exactly that way -- absorption 10
# with a display max of 0.041 (a 24x gain), either number alone looking wrong --
# before it shipped `opaque`, and that tuning is what drove this observation.
#
# The scrubbable records layer takes the opposite treatment: `opaque`
# (depth-tested, unblended) so it reads as crisp dots sitting on the lit globe.
# An additive selection washes out over a bright surface.
#: Globe: DISPLAY RANGE 0-0.205 -> 1/0.205.
#:
#: 4.88, re-tuned after the switch to `opaque`. It was 24.39 (a 0-0.041 window)
#: while the globe was `volumetric` with `absorption=10`, and the ratio is neat:
#: 24.39 / 4.88 = 5.0. Most of that 5x was compensating for the darkening the
#: absorption term imposed, but the switch to `opaque` also changed the
#: compositing (unblended, opacity-independent), not only the absorption term --
#: so read the exact 5.0 as a near-exact coincidence, not a proof that the whole
#: gain was absorption. Either way, carrying the old 24.39 over to `opaque` left
#: the planet blown out, so the gain had to come down.
GLOBE_INTENSITY: Final = 4.88
#: Inert under `opaque` (the absorption term belongs to the emission-absorption
#: `volumetric` integral). Kept as the reference value to restore, together with
#: GLOBE_INTENSITY 24.39, if a future change ever makes a `volumetric` backdrop
#: viable (the backdrop-ordering problem was worked in royerlab/luxar#1227, now
#: resolved).
GLOBE_ABSORPTION: Final = 10.0
#: `opaque`, not the `volumetric` this was originally tuned to.
#:
#: The globe is a BACKDROP, and only `opaque` gets a backdrop's two required
#: properties. It is the one mode with `transparent: false`
#: (`rendering/blending-state.ts`), so THREE puts it in the opaque bucket, drawn
#: before every transparent layer; and it is the only mode that unconditionally
#: sets `depthWrite: true`, so it actually occludes.
#:
#: Both mattered at the time. Measured draw order with a volumetric globe, read
#: off `onBeforeRender` in the live scene:
#:
#:     3-10. All life  (8 tiles)  transparent  depthWrite=0
#:     11.   Earth                transparent  depthWrite=0   <- AFTER the data
#:     12.   Migration highways   transparent  depthWrite=0
#:
#: 1. The globe ORIGINALLY composited ON TOP of the 15M-record layer, multiplying
#:    it by the shell's transmittance -- at absorption 10, most of the way to
#:    erasing it. The cause was the containment rule in
#:    `rendering/depth-sort-coordinator/render-order.ts`, which hoists a group
#:    whose bounding sphere contains another's so that "embedded content
#:    composites on top". Written for a small marker inside a huge cloud; here the
#:    geometry is inverted -- the data sits on a shell OUTSIDE the globe, and its
#:    8-tile bounding sphere is a loose upper bound -- so the DATA was classified
#:    as the container and the BACKDROP as embedded content. Filed as
#:    royerlab/luxar#1227 and since RESOLVED at the engine level:
#:    `orderGroupsWithContainment` now requires a containment edge to hold under
#:    BOTH the Ritter and the legacy centroid bounds, which drops this scene's
#:    false edge (its own comment names it). So the draw-ORDER half no longer
#:    needs `opaque`.
#: 2. `volumetric` never writes depth, so nothing occluded anything: far-side
#:    records and track ribbons showed straight through the planet. Nor would a
#:    `normal` globe have occluded: `getPointBlendingState` forces
#:    `depthWrite: false` for a `normal` POINTS layer at ANY opacity (issue
#:    #1002 -- the `normalModeDepthWrite` / opacity >= 0.99 predicate governs
#:    only Lines). Tightening the draw order (reason 1) does not make any
#:    transparent mode write depth, so `opaque` is STILL required here: it is the
#:    ONLY mode that makes the globe occlude.
#:
#: The cost is that the shell no longer self-shades as a participating medium.
#: `intensity` and `gamma` still apply, so the tuned brightness survives.
GLOBE_BLENDING: Final = "opaque"
#: The SCRUBBABLE records layer's own opacity. Separate from
#: OCCURRENCE_OPACITY (which belongs to the untuned `All life` summary layer).
#: 1.0 is the natural value for crisp opaque dots, but it is effectively inert:
#: under `opaque`, `transparent: false` disables framebuffer blending, so the
#: fragment's emitted alpha is discarded and the layer opacity never reaches the
#: pixel. A no-op recorded for intent / in case the mode changes. The panel
#: reading was 1.00.
RECORDS_OPACITY: Final = 1.0
#: Occurrences: DISPLAY RANGE 0-0.004 -> 1/0.004 = 250, GAMMA 0.82.
#:
#: Written as 100.0, not 250.0, because `validate_intensity` caps the attr at
#: 100 and a 250 build fails outright. The cap costs nothing here: both values
#: saturate. The dimmest taxon colour channel is ~0.13 after
#: OCCURRENCE_COLOR_SCALE, so any intensity above ~8 already clips it to 1.0 —
#: 100 and 250 produce identical pixels. The intent of the setting is "crisp,
#: maximally visible dots", and it is met.
OCCURRENCE_INTENSITY: Final = 100.0
OCCURRENCE_GAMMA: Final = 0.82
OCCURRENCE_BLENDING: Final = "opaque"

#: Fractional radial lift for the occurrence layers, i.e. how far the records
#: float above the globe shell.
#:
#: 0.010 (= 1.0 scene unit at RADIUS 100), not the 0.0012 first used. A globe
#: point is `GLOBE_RADII` 0.23 wide and its soft edge spreads wider still, so a
#: 0.12-unit lift left the records buried inside the shell's rendered
#: footprint. With `opaque` blending the depth test makes that intersection
#: unmistakable: dots wink in and out along the terrain. 1% of the globe radius
#: is imperceptible as displacement but clears the shell everywhere.
OCCURRENCE_LIFT: Final = 0.010

TRACK_WIDTH: Final = 0.30
TRACK_LIFT: Final = 0.006
TRACK_HIGHWAY_WIDTH: Final = 0.075
#: The always-on track layer is context, not subject: it must not out-shout 15M
#: occurrence records. Dimmed hard and drawn thin.
TRACK_HIGHWAY_COLOR_SCALE: Final = 0.28
TRACK_HIGHWAY_OPACITY: Final = 0.5
GREAT_CIRCLE_MAX_STEP_DEG: Final = 1.5

# Points share the element texture at 3 texels each, so one Points node holds
# at most floor(4096/3) * maxTextureSize = 1365 * maxTextureSize points --
# 5,591,040 on a 4096-class GPU. Overflow is a SILENT clamp (one console
# warning, tail never rendered), so every leaf is asserted below this.
MAX_POINTS_PER_NODE: Final = 5_000_000
# Target finest-level size per spatial tile. 2M keeps each tile a comfortable
# WebGL batch and puts the 15M default at 8 tiles, the 100M ceiling at 64.
TARGET_TILE_POINTS: Final = 2_000_000
MAX_LINE_VERTICES_PER_NODE: Final = 2_500_000
#: The scrubbable track layer holds three slot-copies of every vertex (taxon
#: marginal + period marginal + joint), which took it from 106k to 318k. An
#: indexed-Lines node cannot carry an additive ladder -- rebuilding connected
#: components as plain chains would invent or drop edges, so the composed ladder
#: is refused -- and `check_demo_ladders` fails any un-laddered leaf above
#: 200,000 ("will block the main thread on load"). Partitioning below that
#: threshold is therefore the only lever, and it happens to be the natural one:
#: the copies split cleanly into ~106k parts.
MAX_TRACK_VERTICES_PER_NODE: Final = 150_000
# The globe is partitioned for a DIFFERENT reason than the occurrence layer: not
# the 5.59M texture bound but the ladder's LAST commit. A `stream:20000`
# geometric ladder doubles, so its final chunk is ~half the layer, and
# `check_demo_ladders` fails any commit above 1,000,000 (a single commit that
# large blocks the main thread). At the current N_GLOBE this cap is slack -- the
# globe stays one part -- and it is the guard that catches a future raise of
# N_GLOBE rather than letting that land as a main-thread stall.
MAX_GLOBE_POINTS_PER_NODE: Final = 1_000_000

#: View-dependent LOD for the two big summary layers. A `stream:` ladder alone is
#: NOT enough: it is *progressive*, so it converges to 100% of the layer no
#: matter how far the camera is. Whole-globe framing ended up with all 18.1M
#: elements resident. `substitutive_lod` is what makes detail view-dependent — a
#: `kind=lod` group picks ONE child per frame from that tile's own screen size.
#:
#: `coverage_fractions` is overridden rather than left to the default
#: `sqrt(N_i/N_finest)`, and the override is the crux. That default is calibrated
#: for a SINGLE lod group that fills the screen. Here each layer is split into T
#: spatial tiles, so at whole-globe framing a tile's projected diagonal is only
#: ~0.6 of the viewport diagonal — measured in-browser, not estimated. Against
#: the default thresholds that metric still lands on a mid level (tiles sat on
#: 117k and 469k, ~3.6M total).
#:
#: MEASURED WHOLE-GLOBE COVERAGE METRIC ~= 0.60. Every threshold is therefore
#: kept well clear of it. A first attempt put the first threshold AT 0.60 and the
#: tiles never settled: `child_0` and `child_1` stayed simultaneously visible
#: with cross-fade opacities summing to 1.0, so BOTH levels were resident and the
#: layer cost 1.07M instead of 186k. The selector's hysteresis is 10% and
#: downgrade-only, which cannot damp a metric sitting exactly on a boundary.
#: The metric also VARIES BY TILE — a tile nearer the camera projects larger, so
#: on a globe the front-facing tiles run well above the mean. At thresholds of
#: 0.78/0.82 two of the four globe tiles still crossed into their middle level and
#: the globe alone cost 513k of a 1.02M total. The thresholds are therefore set
#: above the LARGEST per-tile metric at whole-globe framing, not the mean.
#: Refinement then begins only once you have zoomed in appreciably, which is the
#: intended behaviour: cheap overview, detail on demand.
OCCURRENCE_LOD_LEVELS: Final = 3
OCCURRENCE_COVERAGE: Final = (0.0, 0.88, 0.96, 1.0)

STREAM_LOD: Final = dict(counts="stream:20000", method="random", seed=0)

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]
# NOTE: parse_int_arg prepends the "--" itself. Passing "--n-points" here makes
# it look for "----n-points", which never matches, so the flag is silently
# ignored and every run builds at the default size.
N_POINTS = parse_int_arg("n-points", DEFAULT_N_POINTS)
N_PARTS = parse_int_arg("n-parts", 0)  # 0 -> derived from N_POINTS

Arbol.max_depth = 5


# =============================================================================
# Taxonomy — the nine display groups
# =============================================================================

# NINE display groups: a lay-legible partition of GBIF's kingdoms/phyla/classes,
# each with a distinct, roughly equal-luminance hue. The viewer renders a
# categorical dimension with 2 categories as a toggle, 3-9 as a dropdown, and
# >=10 as a slider (ui/dimension-sliders.ts); with the leading ``All life``
# summary slot the taxon dimension is 10 categories, so it renders as a slider
# (see ``TAXON_CATEGORIES`` below).
#: The "everything at once" slots for the SCRUBBABLE (selection) layers. The
#: viewer shows the INTERSECTION of the non-displayed slices, so a layer stored
#: only at real ``(taxon, period)`` pairs occupies just the joint cells -- move a
#: single slider off "all" and it lands on an empty slot. So "all" is modelled as
#: a real coordinate: ``taxon`` gets an extra leading category and ``period`` an
#: extra leading category; the summary layers live there, the per-taxon/per-period
#: layers live in the real slots, and every node is an ordinary sliced node whose
#: streaming ladder advances normally.
#:
#: This is a different mechanism from ``extend_to_all`` (used for the persistent
#: globe -- see the module docstring): that keeps a context layer visible at
#: *every* slice, and since the #1157 fix (#1167 on main) a fully-extended node
#: is queried at every slice as intended rather than skipped and frozen.
ALL_LIFE_SLOT: Final = 0
PERIOD_ALL_SLOT: Final = 0

#: Time is a DECADE, not a year, and both non-displayed dims carry an explicit
#: "all" slot. Both choices come from the same measurement.
#:
#: The viewer shows the INTERSECTION of the non-displayed slices, so a layer
#: stored at (real taxon, real period) occupies only the *joint* cells. Two things
#: went wrong with that:
#:
#: 1. Moving ONE slider emptied the scene, because nothing occupied
#:    (Fishes, all-years) at all. Hence the marginal slots below.
#: 2. The joint cells were far too thin. Censused on the real 1.44M scrubbable
#:    sample: 1,132 of 1,143 (taxon, year) cells are populated, but the MEDIAN
#:    populated cell holds just 244 points and 688 cells hold under 500 —
#:    Fishes/1961 is 147 points spread over a globe, which reads as nothing.
#:    Re-binned by decade: 108 of 108 cells populated, median 2,390,
#:    Fishes/1960s = 2,655. A 10x density win for one tenth the slider steps.
#:
#: Categorical rather than a discrete decade axis on purpose: an "all" step plus
#: decade steps cannot both sit on one numeric step grid, and the compiler is
#: right to warn about off-grid discrete values. Categories also read as
#: "period: 1970s" in the dimension readout instead of a bare number.
PERIOD_DECADE_START: Final = 1900
N_PERIODS: Final = 13  # 1900s .. 2020s
PERIOD_CATEGORIES: Final[List[str]] = ["All years"] + [
    f"{PERIOD_DECADE_START + 10 * i}s" for i in range(N_PERIODS)
]

#: Per-cell caps for the scrubbable layer. Marginals are the slots a user reaches
#: by moving a single slider, so they get the bigger budget; joint cells are the
#: 117-way cross product and only need to be legible.
MARGINAL_CELL_CAP: Final = 45_000
JOINT_CELL_CAP: Final = 12_000

TAXON_GROUP_NAMES: Final[List[str]] = [
    "Birds",
    "Insects",
    "Flowering plants",
    "Mammals",
    "Fishes",
    "Reptiles & amphibians",
    "Other animals",
    "Fungi & lichens",
    "Plants, algae & microbes",
]

# Distinct, roughly equal-luminance hues that stay legible over a dark globe.
TAXON_GROUP_COLORS: Final = np.array(
    [
        [1.00, 0.55, 0.15],  # Birds — orange
        [0.75, 0.90, 0.25],  # Insects — yellow-green
        [0.25, 0.85, 0.45],  # Flowering plants — green
        [0.95, 0.30, 0.30],  # Mammals — red
        [0.25, 0.80, 0.95],  # Fishes — cyan
        [0.70, 0.45, 0.95],  # Reptiles & amphibians — violet
        [0.95, 0.40, 0.75],  # Other animals — magenta
        [0.90, 0.75, 0.45],  # Fungi & lichens — tan
        [0.35, 0.75, 0.70],  # Plants, algae & microbes — teal
    ],
    dtype=np.float32,
)

#: What the viewer's categorical dimension actually offers: the summary slot
#: first, then the nine groups. Ten categories means the control renders as a
#: slider rather than a dropdown (the cutover is at 10) -- worth it, because the
#: alternative is a layer that silently shows 2% of its data.
TAXON_CATEGORIES: Final[List[str]] = ["All life", *TAXON_GROUP_NAMES]

_BIRDS, _INSECTS, _FLOWERING, _MAMMALS, _FISHES = 0, 1, 2, 3, 4
_HERPS, _OTHER_ANIMALS, _FUNGI, _OTHER_PLANTS = 5, 6, 7, 8

#: GBIF ``class`` -> group. Checked first; the most specific signal available.
CLASS_TO_GROUP: Final[Dict[str, int]] = {
    "Aves": _BIRDS,
    "Insecta": _INSECTS,
    "Magnoliopsida": _FLOWERING,
    "Liliopsida": _FLOWERING,
    "Mammalia": _MAMMALS,
    # Fishes (a grade, not a clade — grouped as a lay reader expects)
    "Actinopterygii": _FISHES,
    "Actinopteri": _FISHES,
    "Teleostei": _FISHES,
    "Chondrichthyes": _FISHES,
    "Elasmobranchii": _FISHES,
    "Holocephali": _FISHES,
    "Sarcopterygii": _FISHES,
    "Myxini": _FISHES,
    "Petromyzonti": _FISHES,
    "Cephalaspidomorphi": _FISHES,
    "Cyclostomata": _FISHES,
    # Reptiles & amphibians
    "Amphibia": _HERPS,
    "Reptilia": _HERPS,
    "Squamata": _HERPS,
    "Testudines": _HERPS,
    "Crocodylia": _HERPS,
    "Lepidosauria": _HERPS,
    "Sphenodontia": _HERPS,
    "Rhynchocephalia": _HERPS,
    # Other animals — non-insect invertebrates and unplaced chordates
    "Arachnida": _OTHER_ANIMALS,
    "Pycnogonida": _OTHER_ANIMALS,
    "Merostomata": _OTHER_ANIMALS,
    "Malacostraca": _OTHER_ANIMALS,
    "Maxillopoda": _OTHER_ANIMALS,
    "Hexanauplia": _OTHER_ANIMALS,
    "Copepoda": _OTHER_ANIMALS,
    "Ostracoda": _OTHER_ANIMALS,
    "Branchiopoda": _OTHER_ANIMALS,
    "Thecostraca": _OTHER_ANIMALS,
    "Cephalocarida": _OTHER_ANIMALS,
    "Remipedia": _OTHER_ANIMALS,
    "Chilopoda": _OTHER_ANIMALS,
    "Diplopoda": _OTHER_ANIMALS,
    "Pauropoda": _OTHER_ANIMALS,
    "Symphyla": _OTHER_ANIMALS,
    "Collembola": _OTHER_ANIMALS,
    "Entognatha": _OTHER_ANIMALS,
    "Diplura": _OTHER_ANIMALS,
    "Protura": _OTHER_ANIMALS,
    "Gastropoda": _OTHER_ANIMALS,
    "Bivalvia": _OTHER_ANIMALS,
    "Cephalopoda": _OTHER_ANIMALS,
    "Polyplacophora": _OTHER_ANIMALS,
    "Scaphopoda": _OTHER_ANIMALS,
    "Monoplacophora": _OTHER_ANIMALS,
    "Solenogastres": _OTHER_ANIMALS,
    "Caudofoveata": _OTHER_ANIMALS,
    "Clitellata": _OTHER_ANIMALS,
    "Polychaeta": _OTHER_ANIMALS,
    "Oligochaeta": _OTHER_ANIMALS,
    "Hirudinea": _OTHER_ANIMALS,
    "Anthozoa": _OTHER_ANIMALS,
    "Hydrozoa": _OTHER_ANIMALS,
    "Scyphozoa": _OTHER_ANIMALS,
    "Cubozoa": _OTHER_ANIMALS,
    "Staurozoa": _OTHER_ANIMALS,
    "Myxozoa": _OTHER_ANIMALS,
    "Asteroidea": _OTHER_ANIMALS,
    "Echinoidea": _OTHER_ANIMALS,
    "Holothuroidea": _OTHER_ANIMALS,
    "Ophiuroidea": _OTHER_ANIMALS,
    "Crinoidea": _OTHER_ANIMALS,
    "Demospongiae": _OTHER_ANIMALS,
    "Calcarea": _OTHER_ANIMALS,
    "Hexactinellida": _OTHER_ANIMALS,
    "Homoscleromorpha": _OTHER_ANIMALS,
    "Turbellaria": _OTHER_ANIMALS,
    "Trematoda": _OTHER_ANIMALS,
    "Cestoda": _OTHER_ANIMALS,
    "Monogenea": _OTHER_ANIMALS,
    "Chromadorea": _OTHER_ANIMALS,
    "Enoplea": _OTHER_ANIMALS,
    "Secernentea": _OTHER_ANIMALS,
    "Adenophorea": _OTHER_ANIMALS,
    "Gymnolaemata": _OTHER_ANIMALS,
    "Stenolaemata": _OTHER_ANIMALS,
    "Phylactolaemata": _OTHER_ANIMALS,
    "Ascidiacea": _OTHER_ANIMALS,
    "Thaliacea": _OTHER_ANIMALS,
    "Appendicularia": _OTHER_ANIMALS,
    "Leptocardii": _OTHER_ANIMALS,
    "Enteropneusta": _OTHER_ANIMALS,
    "Pterobranchia": _OTHER_ANIMALS,
    # Non-flowering plants, algae, diatoms, cyanobacteria
    "Bryopsida": _OTHER_PLANTS,
    "Polytrichopsida": _OTHER_PLANTS,
    "Sphagnopsida": _OTHER_PLANTS,
    "Andreaeopsida": _OTHER_PLANTS,
    "Takakiopsida": _OTHER_PLANTS,
    "Tetraphidopsida": _OTHER_PLANTS,
    "Jungermanniopsida": _OTHER_PLANTS,
    "Marchantiopsida": _OTHER_PLANTS,
    "Anthocerotopsida": _OTHER_PLANTS,
    "Haplomitriopsida": _OTHER_PLANTS,
    "Polypodiopsida": _OTHER_PLANTS,
    "Lycopodiopsida": _OTHER_PLANTS,
    "Equisetopsida": _OTHER_PLANTS,
    "Psilotopsida": _OTHER_PLANTS,
    "Marattiopsida": _OTHER_PLANTS,
    "Pinopsida": _OTHER_PLANTS,
    "Cycadopsida": _OTHER_PLANTS,
    "Ginkgoopsida": _OTHER_PLANTS,
    "Gnetopsida": _OTHER_PLANTS,
    "Charophyceae": _OTHER_PLANTS,
    "Chlorophyceae": _OTHER_PLANTS,
    "Ulvophyceae": _OTHER_PLANTS,
    "Bryopsidophyceae": _OTHER_PLANTS,
    "Trebouxiophyceae": _OTHER_PLANTS,
    "Klebsormidiophyceae": _OTHER_PLANTS,
    "Zygnematophyceae": _OTHER_PLANTS,
    "Florideophyceae": _OTHER_PLANTS,
    "Bangiophyceae": _OTHER_PLANTS,
    "Compsopogonophyceae": _OTHER_PLANTS,
    "Phaeophyceae": _OTHER_PLANTS,
    "Bacillariophyceae": _OTHER_PLANTS,
    "Coscinodiscophyceae": _OTHER_PLANTS,
    "Fragilariophyceae": _OTHER_PLANTS,
    "Mediophyceae": _OTHER_PLANTS,
    "Dinophyceae": _OTHER_PLANTS,
    "Cyanophyceae": _OTHER_PLANTS,
    "Euglenophyceae": _OTHER_PLANTS,
    "Chrysophyceae": _OTHER_PLANTS,
    "Xanthophyceae": _OTHER_PLANTS,
    "Raphidophyceae": _OTHER_PLANTS,
    "Cryptophyceae": _OTHER_PLANTS,
    "Prymnesiophyceae": _OTHER_PLANTS,
    "Haptophyta": _OTHER_PLANTS,
    "Glaucocystophyceae": _OTHER_PLANTS,
}

#: GBIF ``phylum`` -> group. Second choice, when ``class`` is null or unknown.
PHYLUM_TO_GROUP: Final[Dict[str, int]] = {
    "Arthropoda": _OTHER_ANIMALS,
    "Mollusca": _OTHER_ANIMALS,
    "Annelida": _OTHER_ANIMALS,
    "Cnidaria": _OTHER_ANIMALS,
    "Echinodermata": _OTHER_ANIMALS,
    "Porifera": _OTHER_ANIMALS,
    "Platyhelminthes": _OTHER_ANIMALS,
    "Nematoda": _OTHER_ANIMALS,
    "Nematomorpha": _OTHER_ANIMALS,
    "Bryozoa": _OTHER_ANIMALS,
    "Rotifera": _OTHER_ANIMALS,
    "Tardigrada": _OTHER_ANIMALS,
    "Onychophora": _OTHER_ANIMALS,
    "Brachiopoda": _OTHER_ANIMALS,
    "Nemertea": _OTHER_ANIMALS,
    "Sipuncula": _OTHER_ANIMALS,
    "Chaetognatha": _OTHER_ANIMALS,
    "Ctenophora": _OTHER_ANIMALS,
    "Hemichordata": _OTHER_ANIMALS,
    "Acanthocephala": _OTHER_ANIMALS,
    "Entoprocta": _OTHER_ANIMALS,
    "Gastrotricha": _OTHER_ANIMALS,
    "Kinorhyncha": _OTHER_ANIMALS,
    "Priapulida": _OTHER_ANIMALS,
    "Placozoa": _OTHER_ANIMALS,
    "Xenacoelomorpha": _OTHER_ANIMALS,
    "Chordata": _OTHER_ANIMALS,  # tunicates / lancelets / unplaced vertebrates
    "Magnoliophyta": _FLOWERING,
    "Tracheophyta": _OTHER_PLANTS,
    "Bryophyta": _OTHER_PLANTS,
    "Marchantiophyta": _OTHER_PLANTS,
    "Anthocerotophyta": _OTHER_PLANTS,
    "Chlorophyta": _OTHER_PLANTS,
    "Charophyta": _OTHER_PLANTS,
    "Rhodophyta": _OTHER_PLANTS,
    "Ochrophyta": _OTHER_PLANTS,
    "Myzozoa": _OTHER_PLANTS,
    "Ciliophora": _OTHER_PLANTS,
    "Cyanobacteria": _OTHER_PLANTS,
    "Basidiomycota": _FUNGI,
    "Ascomycota": _FUNGI,
    "Zygomycota": _FUNGI,
    "Chytridiomycota": _FUNGI,
    "Glomeromycota": _FUNGI,
    "Mucoromycota": _FUNGI,
    "Zoopagomycota": _FUNGI,
    "Blastocladiomycota": _FUNGI,
    "Neocallimastigomycota": _FUNGI,
    "Cryptomycota": _FUNGI,
    "Microsporidia": _FUNGI,
    "Entomophthoromycota": _FUNGI,
    "Basidiobolomycota": _FUNGI,
}

#: GBIF ``kingdom`` -> group. Last resort. ``incertae sedis`` is deliberately
#: absent: an unplaceable record is dropped rather than filed under a lie.
KINGDOM_TO_GROUP: Final[Dict[str, int]] = {
    "Animalia": _OTHER_ANIMALS,
    "Plantae": _OTHER_PLANTS,
    "Fungi": _FUNGI,
    "Chromista": _OTHER_PLANTS,
    "Protozoa": _OTHER_PLANTS,
    "Bacteria": _OTHER_PLANTS,
    "Archaea": _OTHER_PLANTS,
    "Viruses": _OTHER_PLANTS,
}

UNMAPPED: Final = -1


# =============================================================================
# Pure helpers (unit-tested; no network / no IO)
# =============================================================================


def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray, relief: np.ndarray) -> np.ndarray:
    """Longitude/latitude in degrees -> Cartesian globe coordinates.

    ``y`` is the north-pole axis and longitude increases eastward. The ``-z``
    keeps the frame right-handed (East x North = outward), so the globe is NOT
    mirrored -- the same convention as ``demo_ocean_currents_earth`` and
    ``demo_global_rivers_earth``. (``demo_earthquakes_3d`` uses an older,
    non-negated variant and compensates by negating its longitudes.)

    Args:
        lon: Longitude in degrees, any range (cos/sin are periodic).
        lat: Latitude in degrees, ``[-90, 90]``.
        relief: FRACTIONAL radial offset (``elev_m / R_EARTH``-like), so
            ``0.0`` sits exactly on the shell of radius :data:`RADIUS`.

    Returns:
        ``(N, 3)`` float32 array of scene-unit coordinates.
    """
    la = np.radians(np.asarray(lat, dtype=np.float64))
    lo = np.radians(np.asarray(lon, dtype=np.float64))
    r = RADIUS * (1.0 + np.asarray(relief, dtype=np.float64))
    cl = np.cos(la)
    return np.column_stack(
        [r * cl * np.cos(lo), r * np.sin(la), -r * cl * np.sin(lo)]
    ).astype(np.float32)


def globe_camera(lon: float, lat: float, *, distance: float = 2.6) -> CameraConfig:
    """A camera looking straight down at ``(lon, lat)`` from ``distance x R``."""
    la, lo = math.radians(lat), math.radians(lon)
    cl = math.cos(la)
    normal = (cl * math.cos(lo), math.sin(la), -cl * math.sin(lo))
    return CameraConfig(
        position=tuple(n * RADIUS * distance for n in normal),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        fov=42.0,
        near=RADIUS * 0.02,
        far=RADIUS * 40.0,
    )


def period_slot(year: np.ndarray) -> np.ndarray:
    """Observation year -> stored ``period`` coordinate (1..N_PERIODS).

    Slot 0 is :data:`PERIOD_ALL_SLOT` ("All years"), so real decades start at 1.
    Years outside the decade range clamp to the first/last decade rather than
    being dropped -- the loader already restricts to ``[YEAR_MIN, YEAR_MAX]``, so
    a clamp here only ever folds the current partial decade into the last slot.
    """
    y = np.asarray(year, dtype=np.int64)
    idx = (y - PERIOD_DECADE_START) // 10
    return (np.clip(idx, 0, N_PERIODS - 1) + 1).astype(np.float32)


def taxon_slot(group_id: int) -> int:
    """Internal group id (0..8) -> stored ``taxon`` coordinate (1..9).

    The tables and :func:`taxon_group_of` keep the compact 0-based ids so the
    colour palette indexes directly; only the coordinate written into the scene
    is shifted past :data:`ALL_LIFE_SLOT`.
    """
    if not 0 <= group_id < len(TAXON_GROUP_NAMES):
        raise ValueError(f"group_id must be in 0..{len(TAXON_GROUP_NAMES) - 1}")
    return group_id + 1


def taxon_group_of(
    kingdom: Optional[str], phylum: Optional[str], class_: Optional[str]
) -> int:
    """Map a GBIF (kingdom, phylum, class) triple to a :data:`TAXON_GROUP_NAMES` id.

    Resolution order is most-specific-first: ``class``, then ``phylum``, then
    ``kingdom``. Returns :data:`UNMAPPED` (``-1``) when none of the three is
    recognised, which the loader treats as "drop this record".
    """
    if class_ is not None:
        g = CLASS_TO_GROUP.get(class_)
        if g is not None:
            return g
    if phylum is not None:
        g = PHYLUM_TO_GROUP.get(phylum)
        if g is not None:
            return g
    if kingdom is not None:
        g = KINGDOM_TO_GROUP.get(kingdom)
        if g is not None:
            return g
    return UNMAPPED


def jitter_sigma_deg(
    uncertainty_m: np.ndarray, lat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-record jitter sigma in DEGREES of latitude and longitude.

    Only ~38% of GBIF coordinate pairs are unique and 15% of records carry
    ``COORDINATE_ROUNDED``, so plotting raw coordinates stacks hundreds of
    records on one pixel and paints visible lattices wherever coordinates were
    rounded to a coarse grid. Each record is therefore displaced by a Gaussian
    whose width is the record's OWN stated positional uncertainty, clamped to
    ``[JITTER_FLOOR_M, JITTER_CEIL_M]`` (a null uncertainty takes the floor).

    Longitude sigma is divided by ``cos(lat)`` because a degree of longitude
    shrinks toward the poles; without it, high-latitude jitter would collapse
    into east-west streaks. The divisor is floored so it cannot blow up at the
    poles themselves.

    Returns:
        ``(sigma_lat_deg, sigma_lon_deg)``, both float32 of shape ``(N,)``.
    """
    unc = np.asarray(uncertainty_m, dtype=np.float64)
    unc = np.where(np.isfinite(unc), unc, JITTER_FLOOR_M)
    unc = np.clip(unc, JITTER_FLOOR_M, JITTER_CEIL_M)
    sigma_lat = np.degrees(unc / (R_EARTH_KM * 1000.0))
    cos_lat = np.maximum(np.cos(np.radians(np.asarray(lat, dtype=np.float64))), 0.05)
    return sigma_lat.astype(np.float32), (sigma_lat / cos_lat).astype(np.float32)


def great_circle_resample(
    lon: np.ndarray, lat: np.ndarray, max_step_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify a lon/lat track so no leg spans more than ``max_step_deg``.

    Consecutive Argos fixes can be thousands of kilometres apart. Joining them
    with a straight line in Cartesian globe space draws a CHORD, which for a
    long leg passes *through* the planet and surfaces on the far side -- the
    track appears to tunnel. Interpolating on the sphere instead (a slerp of
    the two unit vectors) keeps every leg on the surface.

    Returns:
        ``(lon_out, lat_out, source_index)``. ``source_index`` gives, for each
        emitted vertex, the index of the input fix at or before it, so per-fix
        attributes (year, colour) can be carried across without re-deriving.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lon and lat must be 1-D arrays of equal length")
    if lon.size == 0:
        raise ValueError("great_circle_resample needs at least one fix")
    if max_step_deg <= 0:
        raise ValueError(f"max_step_deg must be > 0, got {max_step_deg}")
    if lon.size == 1:
        return lon.copy(), lat.copy(), np.zeros(1, dtype=np.intp)

    la, lo = np.radians(lat), np.radians(lon)
    cl = np.cos(la)
    unit = np.column_stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)])

    dot = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
    ang = np.arccos(dot)  # central angle of each leg, radians
    n_sub = np.maximum(1, np.ceil(np.degrees(ang) / max_step_deg).astype(np.intp))

    out_lon: List[np.ndarray] = []
    out_lat: List[np.ndarray] = []
    out_src: List[np.ndarray] = []
    for i in range(unit.shape[0] - 1):
        k = int(n_sub[i])
        # t excludes the leg's endpoint; the next leg contributes it. Keeps the
        # concatenation free of duplicated vertices.
        t = np.arange(k, dtype=np.float64) / k
        theta = ang[i]
        if theta < 1e-9:
            pts = np.repeat(unit[i][None, :], k, axis=0)
        else:
            s = np.sin(theta)
            a = np.sin((1.0 - t) * theta) / s
            b = np.sin(t * theta) / s
            pts = a[:, None] * unit[i][None, :] + b[:, None] * unit[i + 1][None, :]
        nrm = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(nrm, 1e-12)
        out_lat.append(np.degrees(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0))))
        out_lon.append(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])))
        out_src.append(np.full(k, i, dtype=np.intp))
    out_lon.append(lon[-1:])
    out_lat.append(lat[-1:])
    out_src.append(np.array([lon.size - 1], dtype=np.intp))
    return (
        np.concatenate(out_lon),
        np.concatenate(out_lat),
        np.concatenate(out_src),
    )


def polyline_segment_indices(n_paths: int, n_vertices: int) -> np.ndarray:
    """Segment index pairs for ``n_paths`` equal-length chains.

    Same builder as ``demo_ocean_currents_earth`` / ``demo_dmri_tractography``:
    consecutive vertices within a path are joined, and paths are never joined
    to each other.
    """
    if n_paths < 1 or n_vertices < 2:
        raise ValueError(
            f"need n_paths >= 1 and n_vertices >= 2, got {n_paths}, {n_vertices}"
        )
    base = np.arange(n_paths, dtype=np.uint32)[:, None] * n_vertices
    within = np.arange(n_vertices - 1, dtype=np.uint32)[None, :]
    starts = (base + within).ravel()
    return np.column_stack([starts, starts + 1]).ravel()


def chain_segment_indices(lengths: Sequence[int]) -> np.ndarray:
    """Segment index pairs for consecutively-stored chains of RAGGED length.

    ``lengths[i]`` is the vertex count of chain ``i``; vertices are assumed
    stored back-to-back in that order. Chains shorter than 2 vertices
    contribute nothing (they have no segment) but still consume their slot in
    the vertex numbering.
    """
    out: List[np.ndarray] = []
    offset = 0
    for n in lengths:
        n = int(n)
        if n >= 2:
            s = offset + np.arange(n - 1, dtype=np.uint32)
            out.append(np.column_stack([s, s + 1]).ravel())
        offset += n
    if not out:
        return np.empty(0, dtype=np.uint32)
    return np.concatenate(out)


def stratified_cap(
    labels: np.ndarray, cap: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices keeping at most ``cap`` randomly-chosen members of each label.

    Used for the scrubbable ``By taxon & period`` layer, so a 0.4%-of-GBIF group
    is as explorable as the 62% one. Returned indices are sorted, which keeps
    downstream spatial ordering deterministic.
    """
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    keep: List[np.ndarray] = []
    for value in np.unique(labels):
        idx = np.flatnonzero(labels == value)
        if idx.size > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.append(idx)
    if not keep:
        return np.empty(0, dtype=np.intp)
    return np.sort(np.concatenate(keep)).astype(np.intp)


def tile_count_for(n_points: int, target_tile_points: int) -> int:
    """Number of BSP tiles for ``n_points``, as a power of two.

    The median BSP halves recursively, so the achievable part counts are powers
    of two; reporting the count up front lets the build log and the docstring
    agree with what the partition actually produces.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if target_tile_points < 1:
        raise ValueError(f"target_tile_points must be >= 1, got {target_tile_points}")
    if n_points <= target_tile_points:
        return 1
    return 1 << math.ceil(math.log2(n_points / target_tile_points))


def parts_needed_for(n_points: int, oversample: float = OVERSAMPLE_FACTOR) -> int:
    """How many GBIF parquet parts to read for ``n_points`` kept records.

    Reading MORE parts than the point target strictly needs is the whole
    strategy, for two reasons that both come from the same fact -- parts are
    clustered by publishing dataset, not sampled:

    * Per-part yield swings wildly. Measured on three random parts of the
      2026-08-01 snapshot: 16.7%, 99.1% and 0.9% of rows survived filtering.
    * A single part can be effectively one taxon (some are 100% Aves, one is
      100% bacteria), so a narrow read gives a composition that is an artifact
      of which parts were drawn rather than a property of GBIF.

    ``oversample`` therefore buys both a safety margin on the count and enough
    publisher diversity for the rare taxonomic groups to be explorable.
    """
    if oversample < 1.0:
        raise ValueError(f"oversample must be >= 1.0, got {oversample}")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    return max(1, math.ceil(n_points * oversample / GBIF_USABLE_ROWS_PER_PART))


class BottomKSampler:
    """Uniform random sample of a fixed size from a stream of batches.

    Every row gets an independent uniform key and the ``k`` smallest keys are
    kept; that is exactly a uniform sample of size ``k`` from everything seen,
    no matter how the stream was chunked. This matters more than it looks:

    * It removes any need to guess a per-part subsample probability up front,
      which is unknowable because per-part yield varies by two orders of
      magnitude.
    * It is order-independent. The naive alternative -- fill greedily from
      whichever parts finish first -- would populate a group's sample from one
      or two publishing datasets and so from one or two regions, producing a
      map with birds in Denmark and nowhere else.

    Batches are BUFFERED and the retained sample is only trimmed once the
    buffer is worth a trim. Trimming on every batch is what makes the default
    build expensive: a full 15M-row reservoir would be concatenated and
    ``argpartition``ed once per parquet part -- measured at 0.5 s each, ~100 s
    over the 250-part default read, and ~1.5 GB of copying per trim -- even
    though almost every incoming row loses. Buffering changes nothing about
    WHICH rows are selected (bottom-k over fixed keys does not care how the
    stream is chunked), only how often the reservoir is rewritten.
    """

    def __init__(self, k: int, rng: np.random.Generator) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")
        self.k = int(k)
        self._rng = rng
        self._keys: Optional[np.ndarray] = None
        self._cols: Optional[List[np.ndarray]] = None
        self._pending_keys: List[np.ndarray] = []
        self._pending_cols: List[List[np.ndarray]] = []
        self._pending_n = 0
        self._n_cols: Optional[int] = None
        #: Trim once the buffer holds a quarter of a reservoir. Bigger buffers
        #: keep getting faster, but each one is also held in memory alongside
        #: the retained sample; a quarter is where the default read stops
        #: paying for trims (122 s -> ~12 s) without growing the peak much.
        self._trim_at = max(1, self.k // 4)
        self._n_kept = 0
        self.n_seen = 0

    def add(
        self, columns: Sequence[np.ndarray], keys: Optional[np.ndarray] = None
    ) -> None:
        """Offer a batch of parallel columns (all the same length).

        ``keys`` lets the CALLER supply the uniform keys instead of drawing them
        from this sampler's RNG, and that is what makes a threaded read
        reproducible. Bottom-k over a FIXED key per row is order-independent by
        construction -- the k smallest keys are the k smallest however the
        batches arrive -- whereas drawing keys here consumes one shared RNG
        stream in `as_completed()` order, so network timing decided which part
        got which slice of the stream and a seeded rebuild produced a different
        sample.

        A batch is held by reference until the next trim, so do not mutate one
        after offering it.
        """
        if not columns:
            return
        n = int(columns[0].shape[0])
        if any(int(c.shape[0]) != n for c in columns):
            raise ValueError("all columns in a batch must have equal length")
        if keys is not None and int(keys.shape[0]) != n:
            raise ValueError("keys must have the same length as the columns")
        self.n_seen += n
        if n == 0 or self.k == 0:
            return
        keys = (
            self._rng.random(n) if keys is None else np.asarray(keys, dtype=np.float64)
        )
        if self._n_cols is None:
            self._n_cols = len(columns)
        elif len(columns) != self._n_cols:
            raise ValueError("batch column count changed between add() calls")
        self._pending_keys.append(keys)
        self._pending_cols.append([np.asarray(c) for c in columns])
        self._pending_n += n
        self._n_kept = min(self.k, self._n_kept + n)
        if self._pending_n >= self._trim_at:
            self._trim()

    def _trim(self) -> None:
        """Merge the buffered batches into the retained sample and cut to ``k``."""
        if not self._pending_cols:
            return
        key_parts = self._pending_keys
        col_parts = self._pending_cols
        if self._keys is not None:
            assert self._cols is not None
            key_parts = [self._keys, *key_parts]
            col_parts = [self._cols, *col_parts]
        self._pending_keys = []
        self._pending_cols = []
        self._pending_n = 0
        # A single-element concatenate still copies, so the retained columns
        # never alias a caller's array.
        keys = np.concatenate(key_parts)
        cols = [
            np.concatenate([part[j] for part in col_parts])
            for j in range(len(col_parts[0]))
        ]
        if keys.size > self.k:
            # argpartition is O(n): we only need the k smallest, unordered.
            sel = np.argpartition(keys, self.k - 1)[: self.k]
            keys = keys[sel]
            cols = [c[sel] for c in cols]
        self._keys = keys
        self._cols = cols

    def result(self) -> List[np.ndarray]:
        """The sampled columns, in the order they were offered."""
        self._trim()
        if self._cols is None:
            return []
        return self._cols

    @property
    def n_kept(self) -> int:
        """How many rows the sample holds -- without forcing a trim."""
        return self._n_kept


# =============================================================================
# GBIF — reading the AWS Open Data parquet snapshot
# =============================================================================


def _gbif_filesystem() -> Any:
    """An anonymous pyarrow S3 filesystem for the GBIF Open Data bucket.

    The snapshot is free and public: no account, no credentials, and NOT
    requester-pays. ``anonymous=True`` matters -- with ambient AWS credentials
    on the machine pyarrow would sign the requests and the bucket policy would
    reject them.
    """
    pafs = require_module("pyarrow.fs")
    return pafs.S3FileSystem(anonymous=True, region=GBIF_REGION)


def _resolve_snapshot(fs: Any) -> str:
    """Return the snapshot date to read: the pin, or the newest available.

    GBIF prunes old monthly snapshots, so a hard pin eventually 404s. Falling
    back to the newest keeps the demo working years from now; the notice makes
    the substitution visible rather than silent.
    """
    pafs = require_module("pyarrow.fs")
    selector = pafs.FileSelector(f"{GBIF_BUCKET}/occurrence", recursive=False)
    dates = sorted(
        Path(info.path).name
        for info in fs.get_file_info(selector)
        if info.type == pafs.FileType.Directory
    )
    if not dates:
        raise RuntimeError(
            f"No snapshots found under s3://{GBIF_BUCKET}/occurrence/ -- "
            "the bucket layout may have changed."
        )
    if GBIF_SNAPSHOT in dates:
        return GBIF_SNAPSHOT
    aprint(
        f"⚠️  Pinned GBIF snapshot {GBIF_SNAPSHOT} is gone (GBIF prunes old "
        f"ones); using the newest available: {dates[-1]}"
    )
    return dates[-1]


def _snapshot_citation(fs: Any, snapshot: str) -> str:
    """The snapshot's own DOI line, read from its ``citation.txt``."""
    path = f"{GBIF_BUCKET}/occurrence/{snapshot}/citation.txt"
    try:
        with fs.open_input_stream(path) as handle:
            return handle.readall().decode("utf-8", "replace").strip()
    except Exception as exc:  # noqa: BLE001 - provenance must never break a build
        return f"(citation.txt unavailable: {exc})"


def _list_parts(fs: Any, snapshot: str) -> List[str]:
    """Every parquet part path of a snapshot.

    Parts are 6-digit zero-padded names with NO ``.parquet`` extension, so
    extension-based filtering finds nothing.
    """
    pafs = require_module("pyarrow.fs")
    prefix = f"{GBIF_BUCKET}/occurrence/{snapshot}/occurrence.parquet"
    selector = pafs.FileSelector(prefix, recursive=False)
    return sorted(
        info.path
        for info in fs.get_file_info(selector)
        if info.type == pafs.FileType.File and info.size > 0
    )


def _dictionary_codes(column: Any, lookup: Dict[str, int]) -> np.ndarray:
    """Map a low-cardinality string column to group ids without per-row Python.

    ``class``/``phylum``/``kingdom`` have a few hundred distinct values across
    hundreds of millions of rows, so the dictionary is tiny. Encoding once and
    mapping the DICTIONARY (not the rows) keeps the Python work O(unique) and
    the per-row work a single numpy gather. A ``.to_pylist()`` over the rows
    instead would dominate the whole read.
    """
    pa = require_module("pyarrow")
    arr = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    encoded = arr.dictionary_encode()
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    values = encoded.dictionary.to_pylist()
    table = np.array(
        [lookup.get(v, UNMAPPED) if v is not None else UNMAPPED for v in values],
        dtype=np.int8,
    )
    idx = encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False)
    idx = np.asarray(idx, dtype=np.int64)
    out = np.full(idx.shape, UNMAPPED, dtype=np.int8)
    valid = idx >= 0
    if table.size:
        out[valid] = table[idx[valid]]
    return out


class DatasetRegistry:
    """Assigns a stable small integer id to each GBIF ``datasetkey``.

    The provenance sidecar has to count the records the scene ACTUALLY contains,
    not the candidates that were scanned, so ``datasetkey`` must survive
    sampling. Strings cannot ride through the numeric samplers, so each key gets
    an int32 id here and the ids travel with the coordinates.

    SHARED BY EVERY READER THREAD, hence the lock. "Look the key up, and if it
    is new take the next id and append it" is a read-modify-write, and the GIL
    does not make it atomic: two threads that each arrive with a different new
    key can both read the same ``len(self._keys)``, both claim that id, and the
    second append lands one slot further on -- so one publisher's records are
    credited to the other and the real key never appears in the sidecar at all.
    That is precisely the misattribution this registry exists to prevent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ids: Dict[str, int] = {}
        self._keys: List[str] = []

    def ids_for(self, names: Sequence[Optional[str]]) -> np.ndarray:
        """Map dataset keys to their int32 ids, registering any new ones.

        Feed it a part's DICTIONARY (its few hundred distinct keys), never its
        rows: a per-row Python loop over ~300k names would run in every reader
        thread with the GIL held and dominate the read.
        """
        out = np.empty(len(names), dtype=np.int32)
        with self._lock:
            for i, name in enumerate(names):
                key = name if name is not None else ""
                got = self._ids.get(key)
                if got is None:
                    got = len(self._keys)
                    self._ids[key] = got
                    self._keys.append(key)
                out[i] = got
        return out

    def key(self, dataset_id: int) -> str:
        return self._keys[dataset_id]

    def __len__(self) -> int:
        return len(self._keys)


def _dataset_counts(column: Any, keep_idx: np.ndarray) -> Dict[str, int]:
    """Per-``datasetkey`` kept-record counts over the SCANNED candidates.

    Retained for the build log only. It is deliberately NOT what the provenance
    sidecar reports -- see :func:`write_dataset_provenance`.

    GBIF's Data User Agreement asks that a redistributed subset credit its
    contributing publishers, which is what a registered Derived Dataset does
    (https://www.gbif.org/citation-guidelines#derivedDatasets). Collecting the
    counts costs almost nothing -- ``datasetkey`` is low-cardinality and
    dictionary-encoded on disk -- and it means the citation is available if the
    scene is ever shared.
    """
    if keep_idx.size == 0:
        return {}
    pa = require_module("pyarrow")
    taken = column.take(pa.array(keep_idx))
    if isinstance(taken, pa.ChunkedArray):
        taken = taken.combine_chunks()
    encoded = taken.dictionary_encode()
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    keys = encoded.dictionary.to_pylist()
    if not keys:
        return {}
    idx = np.asarray(
        encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    counts = np.bincount(idx[idx >= 0], minlength=len(keys))
    return {
        key: int(count) for key, count in zip(keys, counts) if key is not None and count
    }


def _dataset_ids(
    column: Any, keep_idx: np.ndarray, registry: "DatasetRegistry"
) -> np.ndarray:
    """Registry ids for the kept rows' ``datasetkey``, one per kept row.

    Goes through the column's DICTIONARY, exactly as :func:`_dictionary_codes`
    does for the taxonomy: a part has a few hundred distinct dataset keys and
    ~300k rows, so mapping the rows one by one in Python would put ~76M
    interpreted iterations, GIL-held, into the middle of a 48-thread read.
    """
    if keep_idx.size == 0:
        return np.empty(0, dtype=np.int32)
    pa = require_module("pyarrow")
    taken = column.take(pa.array(keep_idx))
    if isinstance(taken, pa.ChunkedArray):
        taken = taken.combine_chunks()
    encoded = taken.dictionary_encode()
    if isinstance(encoded, pa.ChunkedArray):
        encoded = encoded.combine_chunks()
    local = np.asarray(
        encoded.indices.fill_null(-1).to_numpy(zero_copy_only=False), dtype=np.int64
    )
    # The trailing None gives null keys a slot of their own, so the gather below
    # never has to special-case them (the registry maps None to "", which
    # `write_dataset_provenance` drops).
    dict_ids = registry.ids_for([*encoded.dictionary.to_pylist(), None])
    return dict_ids[np.where(local >= 0, local, len(dict_ids) - 1)].astype(np.int32)


class _PartResult:
    """Filtered columns from one parquet part, plus the counters for the log."""

    __slots__ = (
        "lat",
        "lon",
        "taxon",
        "year",
        "unc",
        "ds_id",
        "uid",
        "n_rows",
        "n_kept",
        "datasets",
    )

    def __init__(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        taxon: np.ndarray,
        year: np.ndarray,
        unc: np.ndarray,
        ds_id: np.ndarray,
        uid: np.ndarray,
        n_rows: int,
        datasets: Dict[str, int],
    ) -> None:
        self.lat = lat
        self.lon = lon
        self.taxon = taxon
        self.year = year
        self.unc = unc
        self.ds_id = ds_id
        self.uid = uid
        self.n_rows = n_rows
        self.n_kept = int(lat.size)
        self.datasets = datasets


def _read_part(
    fs: Any, path: str, ordinal: int, registry: "DatasetRegistry"
) -> _PartResult:
    """Read one part with column projection and apply every record filter.

    Only 9 of the snapshot's 50 columns are requested, which cuts a ~28 MB part
    to ~1.5-1.8 MB on the wire. Filtering happens in memory because the parts
    are written by parquet-mr 1.9.0, which emits **no statistics for string
    columns** -- there is no predicate pushdown available on ``license``,
    ``class`` or ``kingdom``, and each part is a single row group so there is
    nothing to prune within a part either.
    """
    pq = require_module("pyarrow.parquet")
    table = pq.read_table(path, columns=GBIF_COLUMNS, filesystem=fs)
    n_rows = table.num_rows

    lat = table.column("decimallatitude").to_numpy(zero_copy_only=False)
    lon = table.column("decimallongitude").to_numpy(zero_copy_only=False)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    year = np.asarray(
        table.column("year").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    unc = np.asarray(
        table.column("coordinateuncertaintyinmeters").to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )

    taxon = _dictionary_codes(table.column("class"), CLASS_TO_GROUP)
    from_phylum = _dictionary_codes(table.column("phylum"), PHYLUM_TO_GROUP)
    from_kingdom = _dictionary_codes(table.column("kingdom"), KINGDOM_TO_GROUP)
    # Most-specific-first, mirroring taxon_group_of().
    taxon = np.where(taxon == UNMAPPED, from_phylum, taxon)
    taxon = np.where(taxon == UNMAPPED, from_kingdom, taxon)

    license_codes = _dictionary_codes(
        table.column("license"), {name: 1 for name in ALLOWED_LICENSES}
    )

    keep = np.isfinite(lat) & np.isfinite(lon)
    keep &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)
    # Null island: 1.42M records sit at exactly (0, 0).
    keep &= ~((np.abs(lat) < 0.01) & (np.abs(lon) < 0.01))
    keep &= np.isfinite(year) & (year >= YEAR_MIN) & (year <= YEAR_MAX)
    # == 1, not != 1: keep ONLY the allowlisted licenses, so nulls and unknowns
    # are dropped rather than silently inherited (see ALLOWED_LICENSES).
    keep &= license_codes == 1
    keep &= taxon != UNMAPPED
    # GBIF does not flag country centroids; a huge stated uncertainty is the
    # cheapest proxy. A NULL uncertainty is kept (most records have none).
    keep &= ~(np.isfinite(unc) & (unc > MAX_COORD_UNCERTAINTY_M))

    idx = np.flatnonzero(keep)
    datasets = _dataset_counts(table.column("datasetkey"), idx)

    ds_id = _dataset_ids(table.column("datasetkey"), idx, registry)
    # A globally unique row id: part ordinal in the high bits, row index in the
    # low bits. Lets the provenance count deduplicate a record that a sampler
    # emitted into more than one (taxon, period) slot.
    uid = (np.int64(ordinal) << np.int64(32)) | idx.astype(np.int64)

    return _PartResult(
        lat[idx].astype(np.float32),
        lon[idx].astype(np.float32),
        taxon[idx],
        year[idx].astype(np.int16),
        unc[idx].astype(np.float32),
        ds_id,
        uid,
        n_rows,
        datasets,
    )


class GbifSample:
    """The pooled, sampled GBIF records plus everything needed to report them."""

    __slots__ = (
        "lat",
        "lon",
        "taxon",
        "year",
        "unc",
        "taxon_lat",
        "taxon_lon",
        "taxon_taxon",
        "taxon_year",
        "taxon_unc",
        "slot_taxon",
        "slot_period",
        "scene_datasets",
        "group_totals",
        "n_rows_read",
        "n_kept",
        "n_parts",
        "snapshot",
        "citation",
        "datasets",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs[key])


def _pool_gbif(n_points: int, n_parts: int, *, seed: int) -> GbifSample:
    """Read ``n_parts`` random snapshot parts and draw two uniform samples.

    Two samples come out of one pass, because the two layers want different
    things and neither is a subset of the other:

    * ``n_points`` records sampled uniformly over everything read. Uniform
      means the sample keeps GBIF's REAL composition -- bird-dominated,
      Europe/North-America-dominated. That is the honest picture and it is what
      the always-on layer draws.
    * one capped reservoir **per reachable (taxon, period) slot** --
      :data:`MARGINAL_CELL_CAP` for the marginals, :data:`JOINT_CELL_CAP` for
      the joint cells -- for the scrubbable layer. Capping per slot (rather
      than reweighting the main sample) is what makes a 0.1%-of-GBIF group
      explorable without misrepresenting how much of GBIF it actually is.
    """
    fs = _gbif_filesystem()
    snapshot = _resolve_snapshot(fs)
    citation = _snapshot_citation(fs, snapshot)
    parts = _list_parts(fs, snapshot)
    if not parts:
        raise RuntimeError(f"snapshot {snapshot} has no parquet parts")

    rng = np.random.default_rng(seed)
    n_parts = min(n_parts, len(parts))
    chosen = [parts[i] for i in rng.choice(len(parts), size=n_parts, replace=False)]
    aprint(
        f"Reading {n_parts:,} of {len(parts):,} parts from snapshot {snapshot} "
        f"({GBIF_READ_THREADS} threads, {len(GBIF_COLUMNS)} projected columns)"
    )

    main = BottomKSampler(n_points, np.random.default_rng(seed + 1))
    # ONE RESERVOIR PER REACHABLE SLOT — 9 taxon marginals + 13 period marginals
    # + 117 joint cells = 139, mirroring exactly what the scrubbable layer needs.
    #
    # Stratifying here rather than post-hoc is what makes the thin joint cells
    # usable. Drawing them from a per-taxon reservoir instead gave Birds/1960s
    # just 258 points: bird records are overwhelmingly recent (eBird), so an
    # honest uniform sample of the taxon barely touches an old decade. Reserving
    # per CELL lets each of the 117 fill to its own cap from the 76M rows read.
    #
    # The marginals keep their own uniform-per-taxon and uniform-per-period
    # reservoirs on purpose. Deriving them from the period-balanced cell
    # reservoirs would over-weight sparse decades and misrepresent the taxon.
    n_tax = len(TAXON_GROUP_NAMES)
    marg_taxon = [
        BottomKSampler(MARGINAL_CELL_CAP, np.random.default_rng(seed + 100 + g))
        for g in range(n_tax)
    ]
    marg_period = [
        BottomKSampler(MARGINAL_CELL_CAP, np.random.default_rng(seed + 200 + i))
        for i in range(N_PERIODS)
    ]
    joint = [
        [
            BottomKSampler(
                JOINT_CELL_CAP, np.random.default_rng(seed + 300 + g * N_PERIODS + i)
            )
            for i in range(N_PERIODS)
        ]
        for g in range(n_tax)
    ]
    group_totals = np.zeros(n_tax, dtype=np.int64)
    datasets: Dict[str, int] = {}
    registry = DatasetRegistry()
    n_rows_read = 0
    n_kept = 0
    n_done = 0

    with ThreadPoolExecutor(max_workers=GBIF_READ_THREADS) as pool:
        futures = {
            pool.submit(_read_part, fs, part, i, registry): (part, i)
            for i, part in enumerate(chosen)
        }
        for future in as_completed(futures):
            # POP, don't index: `as_completed` drops its own reference as it
            # yields, so this dict is the last thing holding a finished part's
            # columns. Keeping all 250 alive to the end of the read costs ~2 GB
            # on the default build for data already folded into the samplers.
            part, ordinal = futures.pop(future)
            try:
                res = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad part must not kill the read
                aprint(f"  ⚠️  part {Path(part).name} failed ({exc}); skipping")
                n_done += 1
                continue
            # Uniform keys derived from the part's ORDINAL, not from a shared
            # stream consumed in completion order. Bottom-k over fixed per-row
            # keys is order-independent, so a seeded rebuild reproduces the same
            # sample no matter how the threads interleave.
            part_keys = np.random.default_rng([seed, ordinal]).random(int(res.lat.size))
            n_rows_read += res.n_rows
            n_kept += res.n_kept
            group_totals += np.bincount(res.taxon.astype(np.int64), minlength=n_tax)
            for key, count in res.datasets.items():
                datasets[key] = datasets.get(key, 0) + count
            cols = (
                res.lat,
                res.lon,
                res.taxon,
                res.year,
                res.unc,
                res.ds_id,
                res.uid,
            )
            main.add(cols, keys=part_keys)
            pslot = period_slot(res.year).astype(np.int64)
            for g in range(n_tax):
                in_group = res.taxon == g
                if not in_group.any():
                    continue
                marg_taxon[g].add(
                    tuple(c[in_group] for c in cols), keys=part_keys[in_group]
                )
                for i in range(N_PERIODS):
                    cell = in_group & (pslot == i + 1)
                    if cell.any():
                        joint[g][i].add(
                            tuple(c[cell] for c in cols), keys=part_keys[cell]
                        )
            for i in range(N_PERIODS):
                in_period = pslot == i + 1
                if in_period.any():
                    marg_period[i].add(
                        tuple(c[in_period] for c in cols), keys=part_keys[in_period]
                    )
            n_done += 1
            if n_done % 20 == 0 or n_done == n_parts:
                aprint(
                    f"  {n_done:>4}/{n_parts} parts · {n_rows_read:,} rows read · "
                    f"{n_kept:,} kept · sample {main.n_kept:,}/{n_points:,}"
                )

    if main.n_kept == 0:
        raise RuntimeError("every GBIF part failed or was filtered empty")

    lat, lon, taxon, year, unc, main_ds, main_uid = main.result()

    # Flatten every populated slot into the scrubbable arrays. A record can
    # appear in up to three slots (taxon marginal, period marginal, joint cell);
    # those are three different coordinates, so each is its own element.
    cell_cols: List[List[np.ndarray]] = []
    cell_tax: List[np.ndarray] = []
    cell_per: List[np.ndarray] = []

    def _take(sampler: "BottomKSampler", t_slot: int, p_slot: int) -> None:
        if sampler.n_kept == 0:
            return
        got = sampler.result()
        cell_cols.append(got)
        cell_tax.append(np.full(sampler.n_kept, float(t_slot), dtype=np.float32))
        cell_per.append(np.full(sampler.n_kept, float(p_slot), dtype=np.float32))

    for g in range(n_tax):
        _take(marg_taxon[g], taxon_slot(g), PERIOD_ALL_SLOT)
    for i in range(N_PERIODS):
        _take(marg_period[i], ALL_LIFE_SLOT, i + 1)
    for g in range(n_tax):
        for i in range(N_PERIODS):
            _take(joint[g][i], taxon_slot(g), i + 1)

    if cell_cols:
        (
            t_lat,
            t_lon,
            t_taxon,
            t_year,
            t_unc,
            cell_ds,
            cell_uid,
        ) = (np.concatenate([c[k] for c in cell_cols]) for k in range(7))
        slot_taxon = np.concatenate(cell_tax)
        slot_period = np.concatenate(cell_per)
    else:  # pragma: no cover - only if every slot came back empty
        t_lat = t_lon = t_year = t_unc = np.empty(0, dtype=np.float32)
        t_taxon = np.empty(0, dtype=np.int8)
        slot_taxon = slot_period = np.empty(0, dtype=np.float32)
        cell_ds = np.empty(0, dtype=np.int32)
        cell_uid = np.empty(0, dtype=np.int64)
    # Provenance over the records the SCENE CONTAINS, not the ~76M scanned
    # candidates. A record can be emitted into several (taxon, period) slots, so
    # deduplicate on the stable row uid before counting -- otherwise a dataset's
    # count would depend on how many slices its records happen to land in.
    all_uid = np.concatenate([main_uid, cell_uid])
    all_ds = np.concatenate([main_ds, cell_ds])
    if all_uid.size:
        _, first = np.unique(all_uid, return_index=True)
        counts = np.bincount(all_ds[first].astype(np.int64), minlength=len(registry))
        scene_datasets = {
            registry.key(i): int(c)
            for i, c in enumerate(counts)
            if c and registry.key(i)
        }
    else:  # pragma: no cover - only if every part came back empty
        scene_datasets = {}

    aprint(
        f"scrubbable slots: {len(cell_cols)} populated of "
        f"{n_tax + N_PERIODS + n_tax * N_PERIODS}, {slot_taxon.size:,} elements"
    )

    return GbifSample(
        lat=lat,
        lon=lon,
        taxon=taxon,
        year=year,
        unc=unc,
        taxon_lat=t_lat,
        taxon_lon=t_lon,
        taxon_taxon=t_taxon,
        taxon_year=t_year,
        taxon_unc=t_unc,
        slot_taxon=slot_taxon,
        slot_period=slot_period,
        scene_datasets=scene_datasets,
        group_totals=group_totals,
        n_rows_read=n_rows_read,
        n_kept=n_kept,
        n_parts=n_parts,
        snapshot=snapshot,
        citation=citation,
        datasets=datasets,
    )


def load_gbif(n_points: int, n_parts: int) -> GbifSample:
    """Cached wrapper around :func:`_pool_gbif`.

    The cache key folds in every parameter that changes the output, so a
    different ``--n-points`` or ``--n-parts`` gets its own entry instead of
    silently reusing the wrong sample.
    """
    # The key must move whenever the READ changes shape, not just when its
    # parameters do: the licence gate became an allowlist (so nulls/unknowns are
    # now dropped) and the sampler switched to caller-supplied keys (so the
    # selected rows differ). A warm cache would otherwise serve the old sample.
    # The per-slot caps are in the key for the same reason -- they size the
    # scrubbable arrays this function returns.
    key = (
        f"gbif_{GBIF_SNAPSHOT}_p{n_parts}_n{n_points}_s{SAMPLE_SEED}"
        f"_m{MARGINAL_CELL_CAP}_j{JOINT_CELL_CAP}_allow_v3"
    )
    return cache_computed(
        DEMO_NAME,
        key,
        lambda: _pool_gbif(n_points, n_parts, seed=SAMPLE_SEED),
        version=1,
        recompute=RECOMPUTE,
    )


def write_dataset_provenance(sample: GbifSample) -> Path:
    """Write the contributing-publisher table next to the cached sample.

    Counts are over the records **actually present in the built scene**,
    deduplicated across the (taxon, period) slots a record may be emitted into --
    not over the ~76M candidates the read scanned. The distinction matters
    because this file is the raw material for registering a GBIF Derived Dataset
    (https://www.gbif.org/citation-guidelines#derivedDatasets), so counting
    scanned candidates would over-report every publisher and could list datasets
    that contribute no rendered record at all.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"gbif_datasets_{sample.snapshot}.csv"
    # Count first, then key: equal counts must not be ordered by the id the
    # registry happened to hand out, which follows thread completion order.
    rows = sorted(sample.scene_datasets.items(), key=lambda kv: (-kv[1], kv[0]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("datasetkey,records_in_scene\n")
        for key, count in rows:
            handle.write(f"{key},{count}\n")
    aprint(f"  📄 {len(rows):,} contributing GBIF datasets -> {path.name}")
    return path


# =============================================================================
# Movebank — CC0 animal tracks from the Data Repository
# =============================================================================

#: Each entry is one CC0 bitstream in the Movebank Data Repository. ``group`` is
#: the fallback taxon group; ``species`` overrides it per canonical name for the
#: multi-species sets. Download URLs are the DSpace 8 bitstream content route,
#: which needs no account and supports HTTP Range (so a partial download
#: resumes rather than restarting).
MOVEBANK_SOURCES: Final = (
    {
        "name": "White storks (Argos, 1991-2017)",
        "uuid": "f0147626-8366-4b23-9a55-b2a76e75acbc",
        "filename": "white_stork_argos.csv",
        "group": _BIRDS,
        "doi": "https://doi.org/10.5441/001/1.k29d81dh",
    },
    {
        "name": "Turkey vultures (N & S America)",
        "uuid": "d6a353a9-2291-4e2e-9323-5a39ed6976c2",
        "filename": "turkey_vultures.csv",
        "group": _BIRDS,
        "doi": "https://doi.org/10.5441/001/1.46ft1k05",
    },
    {
        "name": "Blue whales (E North Pacific)",
        "uuid": "bab61b80-771a-4568-bfb4-41ff356ee8c6",
        "filename": "blue_whales.csv",
        "group": _MAMMALS,
        "doi": "https://datarepository.movebank.org/",
    },
    {
        "name": "Humpback whales (E Australia)",
        "uuid": "aca1254b-7574-4677-a4d0-5fc23415859f",
        "filename": "humpback_whales.csv",
        "group": _MAMMALS,
        "doi": "https://doi.org/10.5441/001/1.294",
    },
)
MOVEBANK_URL: Final = (
    "https://datarepository.movebank.org/server/api/core/bitstreams/{uuid}/content"
)

TRACK_MIN_HOURS: Final = 6.0  # thin fixes to at most one per this many hours
TRACK_MAX_SPEED_KMH: Final = 150.0  # above this, a "move" is a bad fix
TRACK_MIN_FIXES: Final = 6  # drop individuals with fewer usable fixes
#: Argos location classes to reject outright. 3/2/1 are the accurate classes,
#: 0 and A are usable for a basin-scale visualisation, B is very poor and Z is
#: explicitly invalid.
ARGOS_REJECT_CLASSES: Final = frozenset({"B", "Z"})


def haversine_km(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in km between paired lon/lat points."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, np.float64))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def thin_and_despike(
    lon: np.ndarray,
    lat: np.ndarray,
    hours: np.ndarray,
    *,
    min_hours: float = TRACK_MIN_HOURS,
    max_speed_kmh: float = TRACK_MAX_SPEED_KMH,
) -> np.ndarray:
    """Indices of one chronologically-ordered track's fixes worth drawing.

    Does two jobs in a single forward pass, because they interact -- a rejected
    spike must not become the anchor the next fix is measured against:

    * **Thinning.** Argos and GPS tags fire far more often than a planetary
      view can show; keeping at most one fix per ``min_hours`` cuts vertex
      counts by an order of magnitude without changing the visible path.
    * **De-spiking.** Argos positions are derived from Doppler shift and the
      poor location classes scatter by hundreds of km. A fix implying a ground
      speed above ``max_speed_kmh`` is dropped -- no stork flies at 150 km/h
      sustained, so such a jump is instrumentation, not behaviour.

    Args:
        lon, lat: Fix coordinates in degrees, already sorted by time.
        hours: Fix times in hours since an arbitrary epoch, non-decreasing.

    Returns:
        Sorted indices into the inputs.
    """
    n = int(lon.size)
    if n == 0:
        return np.empty(0, dtype=np.intp)
    keep = [0]
    last = 0
    for i in range(1, n):
        dt = float(hours[i] - hours[last])
        if dt < min_hours:
            continue
        dist = float(
            haversine_km(
                lon[last : last + 1],
                lat[last : last + 1],
                lon[i : i + 1],
                lat[i : i + 1],
            )[0]
        )
        if dt > 0.0 and dist / dt > max_speed_kmh:
            # Reject the spike WITHOUT advancing `last`, so the next candidate
            # is still measured from the last trusted position.
            continue
        keep.append(i)
        last = i
    return np.asarray(keep, dtype=np.intp)


def _read_movebank_csv(path: Path) -> Dict[str, np.ndarray]:
    """Parse a Movebank export into the columns this demo needs.

    Only a handful of the (sometimes 30+) columns are read. ``visible`` is
    Movebank's own outlier flag and the repository documentation asks that it be
    honoured, so it is treated as authoritative rather than advisory.
    """
    pacsv = require_module("pyarrow.csv")
    pa = require_module("pyarrow")
    table = pacsv.read_csv(
        str(path),
        convert_options=pacsv.ConvertOptions(
            timestamp_parsers=["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"],
            strings_can_be_null=True,
        ),
    )
    names = set(table.column_names)
    required = {"timestamp", "location-long", "location-lat"}
    missing = required - names
    if missing:
        # The largest files in the repository are light-level geolocator or
        # accelerometer exports with no positions at all — fail loudly rather
        # than silently contributing zero tracks.
        raise ValueError(
            f"{path.name} has no location columns (missing {sorted(missing)}); "
            "this is a sensor-only Movebank export, not a track export."
        )

    def col(name: str) -> Optional[np.ndarray]:
        if name not in names:
            return None
        c = table.column(name)
        if isinstance(c, pa.ChunkedArray):
            c = c.combine_chunks()
        return c.to_numpy(zero_copy_only=False)

    lon = np.asarray(col("location-long"), dtype=np.float64)
    lat = np.asarray(col("location-lat"), dtype=np.float64)
    ts = col("timestamp")
    ident = col("individual-local-identifier")
    if ident is None:
        ident = col("tag-local-identifier")
    if ident is None:
        ident = np.zeros(lon.size, dtype=object)
    species = col("individual-taxon-canonical-name")

    ts = np.asarray(ts, dtype="datetime64[s]")
    keep = np.isfinite(lon) & np.isfinite(lat) & ~np.isnat(ts)
    keep &= (lat >= -90.0) & (lat <= 90.0) & (lon >= -180.0) & (lon <= 180.0)

    visible = col("visible")
    if visible is not None:
        vis = np.array([str(v).strip().lower() != "false" for v in visible], dtype=bool)
        keep &= vis
    for flag in ("algorithm-marked-outlier", "manually-marked-outlier"):
        marked = col(flag)
        if marked is not None:
            bad = np.array(
                [str(v).strip().lower() == "true" for v in marked], dtype=bool
            )
            keep &= ~bad
    lc = col("argos:lc")
    if lc is not None:
        rejected = np.array(
            [str(v).strip().upper() in ARGOS_REJECT_CLASSES for v in lc], dtype=bool
        )
        keep &= ~rejected

    idx = np.flatnonzero(keep)
    out = {
        "lon": lon[idx],
        "lat": lat[idx],
        "hours": (ts[idx].astype("int64") / 3600.0),
        "year": ts[idx].astype("datetime64[Y]").astype(int) + 1970,
        "ident": np.asarray([str(v) for v in np.asarray(ident)[idx]], dtype=object),
    }
    out["species"] = (
        np.asarray([str(v) for v in np.asarray(species)[idx]], dtype=object)
        if species is not None
        else np.array([""] * idx.size, dtype=object)
    )
    return out


class TrackSet:
    """Assembled migration polylines, ready to hand to ``add_lines``."""

    __slots__ = (
        "xyz",
        "taxon",
        "year",
        "colors",
        "lengths",
        "n_individuals",
        "n_fixes_raw",
        "n_fixes_kept",
        "sources",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key in self.__slots__:
            setattr(self, key, kwargs[key])


def build_tracks() -> TrackSet:
    """Download, clean, densify and concatenate every Movebank source."""
    chains_lon: List[np.ndarray] = []
    chains_lat: List[np.ndarray] = []
    chains_year: List[np.ndarray] = []
    chains_group: List[int] = []
    n_fixes_raw = 0
    n_fixes_kept = 0
    sources: List[str] = []

    for src in MOVEBANK_SOURCES:
        print_data_provenance(
            title=f"Movebank — {src['name']}",
            source="Movebank Data Repository",
            license="CC0 1.0 Universal (public domain dedication)",
            url=src["doi"],
        )
        path = cached_download(
            MOVEBANK_URL.format(uuid=src["uuid"]),
            DEMO_NAME,
            filename=src["filename"],
        )
        data = _read_movebank_csv(path)
        n_fixes_raw += int(data["lon"].size)
        n_ind = 0
        for ident in np.unique(data["ident"]):
            sel = np.flatnonzero(data["ident"] == ident)
            order = np.argsort(data["hours"][sel], kind="stable")
            sel = sel[order]
            keep = thin_and_despike(
                data["lon"][sel], data["lat"][sel], data["hours"][sel]
            )
            if keep.size < TRACK_MIN_FIXES:
                continue
            sel = sel[keep]
            lon = data["lon"][sel]
            lat = data["lat"][sel]
            year = data["year"][sel]
            d_lon, d_lat, srcidx = great_circle_resample(
                lon, lat, GREAT_CIRCLE_MAX_STEP_DEG
            )
            chains_lon.append(d_lon)
            chains_lat.append(d_lat)
            # Carry the WHOLE YEAR of the originating fix here; it is binned to a
            # decade `period` slot downstream via `period_slot` (see the module
            # docstring for why fractional years were rejected).
            chains_year.append(year[srcidx].astype(np.float32))
            species = str(data["species"][sel[0]]) if sel.size else ""
            chains_group.append(_movebank_group(species, int(src["group"])))
            n_fixes_kept += int(sel.size)
            n_ind += 1
        sources.append(f"{src['name']}: {n_ind} individuals")
        aprint(f"  🐦 {src['name']}: {n_ind} usable individuals")

    if not chains_lon:
        raise RuntimeError("no usable Movebank tracks were assembled")

    lengths = [int(c.size) for c in chains_lon]
    lon = np.concatenate(chains_lon)
    lat = np.concatenate(chains_lat)
    year = np.concatenate(chains_year)
    taxon = np.concatenate(
        [np.full(n, g, dtype=np.float32) for n, g in zip(lengths, chains_group)]
    )
    xyz = lonlat_to_xyz(lon, lat, np.full(lon.size, TRACK_LIFT))
    colors = TAXON_GROUP_COLORS[taxon.astype(np.intp)]

    return TrackSet(
        xyz=xyz,
        taxon=taxon,
        year=year,
        colors=colors,
        lengths=lengths,
        n_individuals=len(lengths),
        n_fixes_raw=n_fixes_raw,
        n_fixes_kept=n_fixes_kept,
        sources=sources,
    )


#: Canonical-name overrides for the multi-species Movebank sets.
MOVEBANK_SPECIES_GROUP: Final[Dict[str, int]] = {
    "Ciconia ciconia": _BIRDS,
    "Cathartes aura": _BIRDS,
    "Coragyps atratus": _BIRDS,
    "Gyps africanus": _BIRDS,
    "Anser fabalis": _BIRDS,
    "Balaenoptera musculus": _MAMMALS,
    "Megaptera novaeangliae": _MAMMALS,
    "Loxodonta africana": _MAMMALS,
    "Syncerus caffer": _MAMMALS,
    "Canis mesomelas": _MAMMALS,
    "Antidorcas marsupialis": _MAMMALS,
}


def _movebank_group(species: str, default: int) -> int:
    """Resolve a Movebank canonical species name to a taxon group id."""
    return MOVEBANK_SPECIES_GROUP.get(species.strip(), default)


# =============================================================================
# Globe — a jittered Fibonacci sphere sampled from NASA Blue Marble
# =============================================================================

BLUE_MARBLE_URL: Final = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/"
    "land_shallow_topo_2048.jpg"
)


def fibonacci_sphere(
    n: int, *, jitter: bool = True, seed: int = 1234
) -> Tuple[np.ndarray, np.ndarray]:
    """``(lon, lat)`` degrees for ``n`` near-uniform points on a sphere.

    The dither is not cosmetic. An undithered Fibonacci lattice is a *lattice*,
    and once the rendered point radius approaches the point spacing its spiral
    arms beat against themselves into visible moire "worms" across the whole
    globe. Displacing each point by up to half a mean cell trades that
    structure for unstructured noise, which reads as texture rather than as a
    rendering bug.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    golden = (1.0 + 5.0**0.5) / 2.0
    y = 1.0 - 2.0 * (i + 0.5) / n
    r_xy = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = 2.0 * np.pi * i / golden
    lat = np.degrees(np.arcsin(np.clip(y, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(r_xy * np.sin(theta), r_xy * np.cos(theta)))
    if jitter:
        cell = np.degrees(np.sqrt(4.0 * np.pi / n))  # mean angular spacing
        lat = np.clip(lat + rng.uniform(-0.5, 0.5, n) * cell, -89.999, 89.999)
        lon = lon + rng.uniform(-0.5, 0.5, n) * cell / np.maximum(
            np.cos(np.radians(lat)), 1e-2
        )
    return lon, lat


def sample_equirect(
    texture: np.ndarray, lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Bilinearly sample an equirectangular RGB texture at lon/lat.

    Row 0 of the image is +90 degrees latitude. Longitude wraps (so the
    antimeridian seam is continuous); latitude clamps.
    """
    h, w = texture.shape[:2]
    x = np.mod((np.asarray(lon, dtype=np.float64) + 180.0) / 360.0 * w, w)
    y = np.clip((90.0 - np.asarray(lat, dtype=np.float64)) / 180.0 * (h - 1), 0, h - 1)
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = (x0 + 1) % w
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    tex = texture.astype(np.float32) / 255.0
    top = tex[y0, x0] * (1.0 - fx) + tex[y0, x1] * fx
    bottom = tex[y1, x0] * (1.0 - fx) + tex[y1, x1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def build_globe() -> Tuple[np.ndarray, np.ndarray]:
    """``(positions, colors)`` for the Blue Marble shell."""
    image_module = require_module("PIL.Image")
    print_data_provenance(
        title="NASA Blue Marble: Next Generation (land_shallow_topo)",
        source="NASA Earth Observatory (Reto Stockli)",
        license="Public domain",
        url="https://visibleearth.nasa.gov/",
    )
    path = cached_download(BLUE_MARBLE_URL, DEMO_NAME, filename="blue_marble.jpg")
    texture = np.asarray(image_module.open(path).convert("RGB"))
    lon, lat = fibonacci_sphere(N_GLOBE)
    positions = lonlat_to_xyz(lon, lat, np.zeros(N_GLOBE))
    # Dim the shell so the occurrence colours read as data on top of it rather
    # than competing with the continents.
    colors = sample_equirect(texture, lon, lat) * GLOBE_DIM
    return positions, colors


# =============================================================================
# Scene assembly
# =============================================================================


def jittered_positions(
    lat: np.ndarray,
    lon: np.ndarray,
    unc: np.ndarray,
    rng: np.random.Generator,
    *,
    relief: float = OCCURRENCE_LIFT,
) -> np.ndarray:
    """Occurrence lat/lon -> globe xyz, displaced by per-record uncertainty."""
    s_lat, s_lon = jitter_sigma_deg(unc, lat)
    j_lat = np.clip(lat + rng.normal(0.0, 1.0, lat.size) * s_lat, -89.999, 89.999)
    j_lon = lon + rng.normal(0.0, 1.0, lon.size) * s_lon
    return lonlat_to_xyz(j_lon, j_lat, np.full(lat.size, relief))


def summary_positions(positions3: np.ndarray) -> np.ndarray:
    """(N, 3) globe coords -> (N, 5) at the "all taxa / all years" slots.

    Summary layers get REAL coordinates in the two non-displayed dimensions,
    pointing at the dedicated all-taxa / all-years slots, rather than a constant
    ``fill`` plus ``extend_to_all``. See :data:`ALL_LIFE_SLOT`.
    """
    n = positions3.shape[0]
    return np.column_stack(
        [
            positions3,
            np.full(n, float(ALL_LIFE_SLOT), dtype=np.float32),
            np.full(n, float(PERIOD_ALL_SLOT), dtype=np.float32),
        ]
    ).astype(np.float32)


def assert_tile_budget(max_elements: int) -> None:
    """Fail the BUILD if a tile cap could trip the viewer's silent point clamp.

    ``clampElementCapacity`` truncates an oversized node to the texture bound,
    logs one browser-console warning, and renders the rest of the scene as if
    nothing happened -- so an over-large leaf costs you points with no error and
    no CI gate anywhere. A check here is the only cheap place to catch it.
    """
    if max_elements > MAX_POINTS_PER_NODE:
        raise RuntimeError(
            f"max_elements={max_elements:,} exceeds the {MAX_POINTS_PER_NODE:,} "
            "per-node budget; the viewer would silently clamp each tile's tail."
        )


def add_lod_tiles(
    scene: Any,
    name: str,
    positions3: np.ndarray,
    colors: np.ndarray,
    *,
    radii: float,
    opacity: float,
    max_elements: int,
    levels: int,
    coverage: Sequence[float],
    sharpness: float = 0.55,
    compositing: Optional[Dict[str, Any]] = None,
) -> None:
    """Add a ``kind=partition`` wrapper whose children are per-tile LOD ladders.

    The Points counterpart of the gsplat ``adaptive`` recipe, and the reason
    whole-globe framing stays cheap: ``coverage_fraction`` is evaluated per tile
    against that tile's own screen size, and non-default levels are lazily
    attached, so a tile that is small on screen never fetches its fine levels.

    Two non-obvious details, both of which cost real time to find:

    * ``partition=`` is NOT forwarded to the per-tile calls -- not even as the
      documented ``partition=False`` "no-partition bypass" sentinel. The
      mutual-exclusion guard tests ``partition is not None``, so ``False``
      raises just like a dict would. Omitting it is also safe: the substitutive
      branch returns before the compiler's auto-partition heuristic is
      consulted, and the substitutive writer already forces ``partition=False``
      on the leaves it creates.
    * ``position_bounds`` is set on the wrapper to the union over all tiles, so
      picking and camera framing treat the layer as one entity rather than
      snapping to whichever tile was hit. Mirrors the library's own
      ``partition=`` path.
    """
    assert_tile_budget(max_elements)
    # Real coordinates in the "all" summary slots: the layer shows at the opening
    # slice and is sliced away when the user scrubs, which is what makes the
    # scrubbable layers legible.
    pos5 = summary_positions(positions3)
    parts = median_bsp_partition(pos5, max_elements)
    lod = substitutive_lod_or_flat(
        dict(
            compression_factor=4,
            levels=levels,
            device="auto",
            # Pin coarsening to the three SPATIAL columns. The default (all
            # dims) would let coarse Gaussians merge across taxon and period.
            coarsen_dims=[0, 1, 2],
            coverage_fractions=list(coverage),
        )
    )
    aprint(
        f"  🗺️  '{name}': {pos5.shape[0]:,} points -> {len(parts)} BSP tiles "
        f"(sizes {min(int(p.size) for p in parts):,}.."
        f"{max(int(p.size) for p in parts):,}), {levels} coarse levels each, "
        f"coverage={list(coverage)}"
    )
    # Compositing attrs ride on the WRAPPER, because the wrapper is the node
    # marked `layer=True` and therefore the one the Layers panel reads to seed
    # its display range / gamma / blend and then pushes down to every
    # descendant material. Putting them on the per-tile lod groups instead does
    # NOT work: of the compositing attrs only `opacity` reaches a lod group's
    # children (measured in-browser -- `intensity` and `gamma` leave the child
    # uniforms at 1).
    wrapper = scene.add_partition_group(
        name,
        display_type="points",
        max_elements=max_elements,
        layer=True,
        position_bounds=position_bounds_from_array(pos5),
        **(compositing or {}),
    )
    for i, idx in enumerate(parts):
        wrapper.add_points(
            f"part_{i}",
            pos5[idx],
            colors=colors[idx],
            radii=radii,
            sharpness=np.full(idx.size, sharpness, dtype=np.float32),
            blending_mode="normal",
            opacity=opacity,
            substitutive_lod=lod,
            additive_lod=STREAM_LOD,
        )


def build_scene(output_path: Path, sample: GbifSample, tracks: TrackSet) -> Path:
    """Write the whole five-layer scene."""
    rng = np.random.default_rng(SAMPLE_SEED + 99)

    with asection("Preparing occurrence geometry"):
        occ_xyz = jittered_positions(sample.lat, sample.lon, sample.unc, rng)
        occ_colors = (
            TAXON_GROUP_COLORS[sample.taxon.astype(np.intp)] * OCCURRENCE_COLOR_SCALE
        )
        aprint(f"always-on layer: {occ_xyz.shape[0]:,} points")

        taxon_xyz = jittered_positions(
            sample.taxon_lat, sample.taxon_lon, sample.taxon_unc, rng
        )
        taxon_pos = np.column_stack(
            [taxon_xyz, sample.slot_taxon, sample.slot_period]
        ).astype(np.float32)
        taxon_colors = (
            TAXON_GROUP_COLORS[sample.taxon_taxon.astype(np.intp)]
            * OCCURRENCE_COLOR_SCALE
        )
        aprint(f"scrubbable layer: {taxon_pos.shape[0]:,} elements")

    with asection(f"Building globe ({N_GLOBE:,} points)"):
        globe_xyz, globe_colors = build_globe()

    with asection("Assembling migration ribbons"):
        indices = chain_segment_indices(tracks.lengths)
        # Three copies of every track vertex: taxon marginal, period marginal
        # and the joint cell -- the same reason the occurrence layer needs them.
        # Vertex ORDER is preserved in each copy, so one `lengths` list repeated
        # three times rebuilds the chains correctly.
        t_period = period_slot(tracks.year)
        t_taxon = tracks.taxon + 1.0
        n_v = tracks.xyz.shape[0]
        all_tax = np.full(n_v, float(ALL_LIFE_SLOT), dtype=np.float32)
        all_per = np.full(n_v, float(PERIOD_ALL_SLOT), dtype=np.float32)
        track_pos = np.concatenate(
            [
                np.column_stack([tracks.xyz, t_taxon, all_per]),  # taxon marginal
                np.column_stack([tracks.xyz, all_tax, t_period]),  # period marginal
                np.column_stack([tracks.xyz, t_taxon, t_period]),  # joint
            ]
        ).astype(np.float32)
        track_colors = np.tile(tracks.colors, (3, 1))
        track_indices = chain_segment_indices(list(tracks.lengths) * 3)
        aprint(
            f"{tracks.n_individuals} individuals · {tracks.xyz.shape[0]:,} vertices"
            f" · {indices.size // 2:,} segments; scrubbable copies "
            f"{track_pos.shape[0]:,} vertices"
        )

    dims = Dimensions(
        [
            Dimension("x", unit="", display=True),
            Dimension("y", unit="", display=True),
            Dimension("z", unit="", display=True),
            Dimension(
                "taxon",
                unit="",
                categories=list(TAXON_CATEGORIES),
                display=False,
                description=(
                    "Major taxonomic group; the first slot shows every group at once"
                ),
            ),
            Dimension(
                "period",
                unit="",
                categories=list(PERIOD_CATEGORIES),
                display=False,
                description=(
                    "Decade the occurrence was recorded; the first slot shows "
                    "every decade at once"
                ),
            ),
        ]
    )

    with asection("Writing scene"):
        with LuxarZarrCompiler(
            str(output_path), encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    # Neutral, not ACES: the palette is a CATEGORICAL encoding of
                    # taxonomic group, and ACES's filmic rolloff shifts hues
                    # enough to blur the legend's identity with the globe.
                    tone_mapping="Neutral",
                    camera=globe_camera(10.0, 25.0, distance=2.95),
                    # Open on (All life, All years) -- the slots the summary
                    # layers occupy. Without this the scene would open on
                    # taxon=All life and period=All years anyway (both are
                    # category index 0, the range minima), but pinning it makes
                    # the intent explicit and survives any future reordering of
                    # the categories.
                    dimensions=DimensionsConfig(
                        current_step=[
                            0.0,
                            0.0,
                            0.0,
                            float(ALL_LIFE_SLOT),
                            float(PERIOD_ALL_SLOT),
                        ]
                    ),
                ),
            )
            scene.attrs["title"] = "Biodiversity at Planetary Scale"
            scene.attrs["gbif_snapshot"] = sample.snapshot
            scene.attrs["gbif_citation"] = sample.citation
            # The publishers behind the records the scene HOLDS, matching the
            # sidecar. `sample.datasets` counts the ~76M scanned candidates and
            # would credit publishers that contributed no rendered record.
            scene.attrs["gbif_datasets"] = len(sample.scene_datasets)

            # The globe: extended over every slice (so it is the persistent
            # geographic reference), partitioned to stay under the per-node
            # texture bound, laddered for a fast first paint -- and deliberately
            # WITHOUT substitutive LOD (see N_GLOBE).
            #
            # `extend_to_all` is inferred from the 3-column positions + `fill`.
            # Before royerlab/luxar#1157 was fixed a fully-extended node was never
            # queried at all, and this demo carried a 25k globe replicated into
            # all 139 slots as a workaround; the fix made that unnecessary.
            scene.add_points(
                "Earth",
                globe_xyz,
                colors=globe_colors,
                radii=GLOBE_RADII,
                dim_order=["x", "y", "z"],
                fill={
                    "taxon": float(ALL_LIFE_SLOT),
                    "period": float(PERIOD_ALL_SLOT),
                },
                blending_mode=GLOBE_BLENDING,
                intensity=GLOBE_INTENSITY,
                offset=0.0,
                gamma=1.0,
                opacity=1.0,
                layer=True,
                partition=dict(max_elements=MAX_GLOBE_POINTS_PER_NODE),
                additive_lod=STREAM_LOD,
            )

            add_lod_tiles(
                scene,
                "All life",
                occ_xyz,
                occ_colors,
                radii=OCCURRENCE_RADII,
                opacity=OCCURRENCE_OPACITY,
                max_elements=TARGET_TILE_POINTS,
                levels=OCCURRENCE_LOD_LEVELS,
                coverage=OCCURRENCE_COVERAGE,
                # `normal`, with no display-range push. The Layers-panel settings
                # transcribed below were tuned on `Earth` and on the SCRUBBABLE
                # records layer; this one was left alone, so it keeps its own
                # appearance rather than inheriting settings never chosen for it.
                compositing=dict(
                    blending_mode="normal",
                    opacity=OCCURRENCE_OPACITY,
                ),
            )

            scene.add_points(
                "By taxon & period",
                taxon_pos,
                colors=taxon_colors,
                radii=OCCURRENCE_RADII * 1.6,
                sharpness=np.full(taxon_pos.shape[0], 0.55, dtype=np.float32),
                blending_mode=OCCURRENCE_BLENDING,
                intensity=OCCURRENCE_INTENSITY,
                offset=0.0,
                gamma=OCCURRENCE_GAMMA,
                opacity=RECORDS_OPACITY,
                layer=True,
                # VISIBLE, deliberately. These layers occupy the per-taxon and
                # per-decade slots, so at the opening (All life, All years) slice
                # they contribute exactly nothing and the first paint is
                # unchanged. Shipping them hidden made both sliders look broken:
                # moving one correctly sliced the summary layers away and there
                # was nothing enabled to replace them, so the scene just went
                # black until you found the Layers panel.
                partition=dict(max_elements=TARGET_TILE_POINTS),
                additive_lod=STREAM_LOD,
            )

            scene.add_lines(
                "Migration highways",
                # NOT extended, deliberately. These tracks are 105,662 segments;
                # extending them over every slice buried the selection (7,283
                # fish records disappeared under the ribbons). The globe alone
                # supplies the geographic context that was actually missing, so
                # the tracks stay at the summary slot and scrubbing isolates.
                # `Migrations by slice` carries the per-slot copies.
                vertices=summary_positions(tracks.xyz),
                widths=TRACK_HIGHWAY_WIDTH,
                colors=tracks.colors * TRACK_HIGHWAY_COLOR_SCALE,
                indices=indices,
                line_type="indexed",
                # `normal`, NOT `additive`: additive ignores the depth buffer, so
                # tracks on the far side of the globe paint straight through the
                # continents and read as a broken land mask.
                blending_mode="normal",
                opacity=TRACK_HIGHWAY_OPACITY,
                layer=True,
                partition=dict(max_elements=MAX_LINE_VERTICES_PER_NODE),
            )

            scene.add_lines(
                "Migrations by slice",
                vertices=track_pos,
                widths=TRACK_WIDTH,
                colors=track_colors,
                indices=track_indices,
                line_type="indexed",
                blending_mode="normal",
                opacity=1.0,
                intensity=1.0,
                layer=True,
                # VISIBLE, deliberately. These layers occupy the per-taxon and
                # per-decade slots, so at the opening (All life, All years) slice
                # they contribute exactly nothing and the first paint is
                # unchanged. Shipping them hidden made both sliders look broken:
                # moving one correctly sliced the summary layers away and there
                # was nothing enabled to replace them, so the scene just went
                # black until you found the Layers panel.
                partition=dict(max_elements=MAX_TRACK_VERTICES_PER_NODE),
            )

            _add_overlays(scene, sample, tracks)

    aprint(f"Scene saved: {output_path}")
    return output_path


def _add_overlays(scene: Any, sample: GbifSample, tracks: TrackSet) -> None:
    """Title + one credit line, matching every other demo in the gallery.

    Deliberately just these two. The house convention (ocean_currents,
    global_rivers, desi_galaxies, gaia_milky_way, ...) is a difference-blended
    title at ``(0.02, 0.02)`` and a single one-line credit at ``(0.98, 0.97)``.
    An earlier version of this demo carried a stat block, a nine-row colour
    legend and a caption paragraph, which read as a different product from the
    rest of the gallery. The taxon colours are legible without a legend because
    the Dimension Navigation readout names the current group as you scrub it, and
    the composition caveats live in the module docstring where they can be
    stated properly.
    """
    scene.add_text(
        "Biodiversity at Planetary Scale",
        position=(0.02, 0.02),
        font_size=0.045,
        anchor="top-left",
        color="rgba(255,255,255,0.75)",
        blend_mode="difference",
    )
    scene.add_text(
        f"{sample.lat.size:,} GBIF occurrence records • "
        f"{tracks.n_individuals} tracked animals • "
        f"snapshot {sample.snapshot} • CC BY / CC0 records only",
        position=(0.98, 0.97),
        font_size=0.015,
        anchor="bottom-right",
        color="rgba(200,200,220,0.5)",
    )


# =============================================================================
# Build-parameter marker
# =============================================================================


def _build_params() -> Dict[str, Any]:
    return {
        "n_points": N_POINTS,
        "n_parts": N_PARTS or parts_needed_for(N_POINTS),
        "snapshot": GBIF_SNAPSHOT,
        "seed": SAMPLE_SEED,
        "n_globe": N_GLOBE,
        "tile_points": TARGET_TILE_POINTS,
        "marginal_cap": MARGINAL_CELL_CAP,
        "joint_cap": JOINT_CELL_CAP,
        "version": 1,
    }


def _marker_path(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.name}.build.json"


def scene_marker_matches(output_path: Path) -> bool:
    """True when an existing scene was built with the CURRENT flags.

    ``main()`` short-circuits on an existing output directory, so without this
    an earlier ``--n-points 200000`` smoke build would be silently reused for a
    later full run and there would be no diagnostic at all -- the scene would
    just be 75x smaller than asked for.
    """
    marker = _marker_path(output_path)
    if not marker.exists():
        return False
    try:
        with open(marker, encoding="utf-8") as handle:
            return json.load(handle) == _build_params()
    except (OSError, ValueError):
        return False


def write_scene_marker(output_path: Path) -> None:
    with open(_marker_path(output_path), "w", encoding="utf-8") as handle:
        json.dump(_build_params(), handle, indent=1, sort_keys=True)


# =============================================================================
# Main
# =============================================================================


def load_or_build_scene(output_path: Path) -> Path:
    """Return the built scene, regenerating it when flags or data changed."""
    if output_path.exists() and not RECOMPUTE and scene_marker_matches(output_path):
        aprint(f"Using existing scene: {output_path}")
        return output_path
    if output_path.exists() and not RECOMPUTE:
        aprint(
            "Existing scene was built with different parameters "
            f"({_marker_path(output_path).name} mismatch); rebuilding."
        )

    n_parts = N_PARTS or parts_needed_for(N_POINTS)
    print_data_provenance(
        title="GBIF Occurrence Snapshot (AWS Open Data)",
        source=f"s3://{GBIF_BUCKET}/occurrence/{GBIF_SNAPSHOT}/",
        license="Per-record; this demo keeps CC_BY_4_0 and CC0_1_0 only",
        url="https://www.gbif.org/citation-guidelines",
        note=(
            f"Reading {n_parts:,} randomly chosen parquet parts "
            f"({len(GBIF_COLUMNS)} of 50 columns projected)."
        ),
    )
    with asection("GBIF occurrence records"):
        sample = load_gbif(N_POINTS, n_parts)
        _report_sample(sample)
        write_dataset_provenance(sample)

    with asection("Movebank animal tracks"):
        tracks = cache_computed(
            DEMO_NAME,
            f"tracks_v1_h{TRACK_MIN_HOURS}_s{GREAT_CIRCLE_MAX_STEP_DEG}",
            build_tracks,
            version=1,
            recompute=RECOMPUTE,
        )
        aprint(
            f"{tracks.n_individuals} individuals · "
            f"{tracks.n_fixes_raw:,} raw fixes -> {tracks.n_fixes_kept:,} kept -> "
            f"{tracks.xyz.shape[0]:,} densified vertices"
        )

    path = build_scene(output_path, sample, tracks)
    write_scene_marker(output_path)
    return path


def _report_sample(sample: GbifSample) -> None:
    """Log what was actually read, so the numbers are never taken on faith."""
    total = int(sample.group_totals.sum())
    aprint(
        f"{sample.n_parts:,} parts · {sample.n_rows_read:,} rows read · "
        f"{sample.n_kept:,} passed filters "
        f"({100.0 * sample.n_kept / max(1, sample.n_rows_read):.1f}%)"
    )
    aprint(
        f"sample: {sample.lat.size:,} always-on, {sample.taxon_lat.size:,} scrubbable"
    )
    aprint(f"snapshot DOI: {sample.citation}")
    # NOT "GBIF's proportions": the filters are not taxon-neutral (see the
    # module docstring). This is the composition of the filtered subset.
    with asection("Composition of the filtered, license-clean records"):
        for gid, name in enumerate(TAXON_GROUP_NAMES):
            n = int(sample.group_totals[gid])
            aprint(f"{name:28s} {n:>12,}  {100.0 * n / max(1, total):6.2f}%")


def main() -> None:
    aprint("=" * 74)
    aprint("Demo: Biodiversity at Planetary Scale — GBIF + Movebank on a globe")
    aprint("=" * 74)

    output_path = get_demos_output_dir() / f"{DEMO_NAME}.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint("No scene found. Run without --serve-only first.")
        return

    scene_path = load_or_build_scene(output_path)

    if NO_SERVE:
        aprint(f"Scene ready at {scene_path}")
    else:
        aprint(
            "Data credit: GBIF.org occurrence snapshot (CC BY / CC0 records) · "
            "Movebank Data Repository (CC0) · NASA Blue Marble (public domain)"
        )
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
