#!/usr/bin/env python3
"""GSplats Demo: 4D NEXRAD Level II — El Reno Tornadic Supercell (2013-05-31)

Reconstructs the El Reno, Oklahoma tornadic supercell as a 4D (3D + time)
Gaussian-splat timelapse from WSR-88D Level II weather-radar volume scans,
regridded from the radar's native polar geometry into a Cartesian storm box.

================================================================================
A SUPERCELL THUNDERSTORM AS GAUSSIAN SPLATS
================================================================================

A supercell is a thunderstorm organised around a single deep, persistently
rotating updraft — a mesocyclone. That rotation gives it a 3D architecture no
ordinary storm has, and essentially all of it is invisible from the ground:

  * The HOOK ECHO — precipitation wrapped around the low-level mesocyclone by
    the rear-flank downdraft, curling into the comma shape that is the classic
    radar tornado signature.
  * The BOUNDED WEAK ECHO REGION (BWER, or "vault") — a column of *low*
    reflectivity punched through the storm's core, where the updraft is so
    violent that raindrops and hail are carried aloft faster than they can grow
    to detectable size. A hole in the middle of the storm, held open by wind.
  * The ECHO OVERHANG — the mid-level core leaning out over the inflow notch,
    suspended above almost nothing.
  * The OVERSHOOTING TOP — the updraft punching through the equilibrium level
    and bulging into the stratosphere, here past 15 km.

A magenta vertical line marks where the tornado actually was, on the frames when
it was on the ground. That position comes from the NWS ground damage survey — an
entirely separate source from the radar — and it lands right at the hook echo,
which is a satisfying confirmation that both are describing the same storm.

WHY THE DATA LOOKS THE WAY IT DOES

A WSR-88D does not sample a volume. It spins a ~0.95-degree pencil beam around
the horizon at one elevation angle, steps up, and spins again — 15 tilts from
0.5 to 19.5 degrees, about 4.5 minutes for a full volume. The result is a set of
nested CONES of samples, not a grid. Adjacent tilts are 0.5 degrees apart near
the ground but 3.2 degrees apart aloft, far wider than the beam itself, so the
raw gates render as discrete conical shells with gaps between them — venetian
blinds, not a cloud. Turning that into a volume is the job of an objective
analysis, here a Barnes distance-weighted interpolation onto a Cartesian box.

Gate positions are not simply range-and-angle either. The beam refracts in the
atmosphere's vertical density gradient and bends back toward the earth, while
the earth curves away beneath it. Both are folded into the standard
4/3-effective-earth-radius model used below.

THE STORM

On 2013-05-31 this supercell produced the widest tornado ever recorded — 4.2 km
across at peak, rated EF3. Touchdown was at 23:03 UTC, dissipation at 23:43 UTC.
The 82 volume scans here span 21:00 UTC through 03:00 UTC, so scrubbing the
Time slider carries you from a nearly empty sky, through initiation and
tornadogenesis, into the overnight mesoscale system that flooded Oklahoma
City. The first ~16 frames are almost bare: the storm has not formed yet.

DATA SOURCE & CITATIONS:
========================

Dataset:
--------
Source:  NEXRAD Level II archive (NOAA Open Data Dissemination)
Bucket:  unidata-nexrad-level2 (anonymous HTTPS, no credentials required)
         NOTE: the widely documented `noaa-nexrad-level2` bucket is RETIRED and
         now returns AccessDenied; `unidata-nexrad-level2` is the live archive.
Site:    KTLX - Twin Lakes, Oklahoma (35.33306 N, 97.27748 W, 369 m AMSL
         tower base + 19 m feedhorn AGL)
Date:    2013-05-31 21:00Z - 2013-06-01 03:00Z (82 volume scans, ~4.3 min
         cadence): initiation, the El Reno tornado, and the overnight growth
         into the MCS that flooded Oklahoma City
VCP:     212 (precipitation mode, 15 elevations 0.5-19.5 deg, split cuts)
Size:    ~850 MB total (82 files, ~10 MB each, gzipped Archive II)
License: U.S. Government work, public domain (17 U.S.C. 105). NOAA requests
         attribution and that modified data not be presented as original NOAA
         data. The shipped splat bundle is a DERIVED product (regridded and
         Gaussian-fitted), not original NOAA data.

How to Cite:
------------
NEXRAD on AWS was accessed on <DATE> from
https://registry.opendata.aws/noaa-nexrad.

Decoder:
--------
May, R. M., et al. MetPy: A Python Package for Meteorological Data. Unidata.
doi:10.5065/D6WW7G29

Tornado track:
--------------
NOAA/NWS Storm Prediction Center tornado database (public domain),
https://www.spc.noaa.gov/wcm/data/ — the 2013-05-31 Oklahoma EF3 record: 16.2 mi
path, 4,576 yd (4.18 km) max width, the widest tornado ever recorded.

Beam geometry:
--------------
Doviak, R. J., and D. S. Zrnic (1993). Doppler Radar and Weather Observations,
2nd ed. 4/3-effective-earth-radius model.

WORKFLOW:
=========

 1. Download  82 Level II volume scans over plain HTTPS (cached, ~850 MB)
 2. Decode    each with MetPy Level2File (reads the archived .gz directly)
 3. Select    one reflectivity sweep per true elevation (split-cut dedup)
 4. Geolocate every gate with the 4/3-effective-earth beam model
 5. Grid      polar gates into a fixed ANISOTROPIC Cartesian box (Barnes,
              kd-tree; coarser vertically because the radar samples the
              vertical far more sparsely -- see GRID_Z_M)
 6. Mask      cells outside the radar's real beam coverage
 7. Map       dBZ to a single GLOBAL non-negative intensity (no per-frame gain)
 8. Fit       each timepoint as 3D Gaussian splats (per-timepoint caching)
 9. Stack     all timepoints into one 4D dataset (time = last center column)
10. Mark      the surveyed tornado position as a vertical line, on the frames
              when it was on the ground (NWS damage survey via the SPC database)
11. Visualise - scrub the Time slider through the storm's life cycle

USAGE:
======
    python demo_gsplats_4d_nexrad_supercell.py [options]

Options:
    --recompute:         Re-download, re-grid and re-fit from scratch
    --no-serve:          Generate the scene without launching the viewer
    --serve-only:        Just serve a previously generated scene
    --max-timepoints=N:  Volume scans to use (default: all 82)
    --grid-m=N:          Horizontal grid spacing in metres (default: 750)
    --grid-z-m=N:        Vertical grid spacing in metres (default: 1500)
    --splats=N:          Fixed splats per timepoint (default 0 = adaptive)
    --dbz-floor=N:       dBZ below which a cell is transparent (default: 20)
    --vert-exag=N:       Vertical exaggeration for display (default: 1 = true)
    --relist:            Maintenance: print a fresh S3 listing and exit

Requirements:
    The default path loads precomputed splats from Git LFS and needs no
    network, no GPU and no radar decoder. Only --recompute needs the optional
    extras -- install them with:  pip install 'luxar[demos]'

Output:
    - Scene saved to: datasets/demos/gsplats_4d_nexrad_supercell.luxar.zarr
    - Automatically opens in browser
"""

DEMO_META = {
    "key": "gsplats_4d_nexrad_supercell",
    "title": "4D NEXRAD Supercell (El Reno)",
    "description": "Weather-radar reflectivity of the 2013 El Reno tornadic supercell as a 4D Gaussian-splat timelapse.",
    "category": "geoscience",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 12,
        "compute": "medium",
        "gpu": "optional",
        "local_data": "git-lfs",
    },
    "caches": ["gsplats_nexrad_supercell"],
    "outputs": ["gsplats_4d_nexrad_supercell"],
}

import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
from arbol import Arbol, aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import require_module
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.typing_utils.constants import DEFAULT_TRUNCATION_RADIUS
from luxar.utils.demos import (
    cached_download,
    detect_device,
    launch_viewer,
    load_precomputed_bundle,
    parse_demo_flags,
    parse_int_arg,
    print_data_provenance,
    warn_if_no_cuda_gpu,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

BUCKET_HOST = "https://unidata-nexrad-level2.s3.amazonaws.com"
SITE = "KTLX"

# Site constants, read from the VolConsts block of KTLX20130531_230523_V06.
SITE_LAT, SITE_LON = 35.33306, -97.27748

#: The full convective evening, 21:00 UTC on 2013-05-31 through 02:59 UTC on
#: 06-01 — 82 volume scans at ~4.3 min cadence. Spans initiation, the
#: El Reno tornado (touchdown 23:03, dissipation 23:43 UTC) and the overnight
#: growth into the flash-flooding MCS that hit Oklahoma City.
#:
#: Entries are (date_prefix, HHMMSS) because THE WINDOW CROSSES MIDNIGHT — a
#: bare time is ambiguous once 00Z belongs to the next radar-day, and the S3
#: prefix is keyed by date.
#:
#: PINNED rather than listed at runtime: the default path must touch no network
#: at all, the LFS bundle's inner filenames derive from these indices and must
#: stay stable, and the 2013 archive is closed so there is nothing to discover.
#: Refresh with --relist if the archive is ever re-ingested under other keys.
#:
#: Measured caveat: the first ~16 scans (the 21Z hour) hold almost NO echo in
#: this domain — the storm has not initiated yet, so the timelapse opens on a
#: nearly empty sky and the first cells appear around 22Z. That lead-in is
#: deliberate; trim it with --max-timepoints or --start-index.
VOLUME_SCANS: tuple[tuple[str, str], ...] = (
    ("2013/05/31", "210009"),
    ("2013/05/31", "210337"),
    ("2013/05/31", "210706"),
    ("2013/05/31", "211034"),
    ("2013/05/31", "211403"),
    ("2013/05/31", "211731"),
    ("2013/05/31", "212059"),
    ("2013/05/31", "212427"),
    ("2013/05/31", "212810"),
    ("2013/05/31", "213151"),
    ("2013/05/31", "213601"),
    ("2013/05/31", "213957"),
    ("2013/05/31", "214407"),
    ("2013/05/31", "214832"),
    ("2013/05/31", "215242"),
    ("2013/05/31", "215705"),
    ("2013/05/31", "220114"),
    ("2013/05/31", "220537"),
    ("2013/05/31", "221011"),
    ("2013/05/31", "221445"),
    ("2013/05/31", "221920"),
    ("2013/05/31", "222355"),
    ("2013/05/31", "222832"),
    ("2013/05/31", "223308"),
    ("2013/05/31", "223746"),
    ("2013/05/31", "224223"),
    ("2013/05/31", "224659"),
    ("2013/05/31", "225137"),
    ("2013/05/31", "225611"),
    ("2013/05/31", "230048"),
    ("2013/05/31", "230523"),
    ("2013/05/31", "230957"),
    ("2013/05/31", "231434"),
    ("2013/05/31", "231911"),
    ("2013/05/31", "232345"),
    ("2013/05/31", "232823"),
    ("2013/05/31", "233259"),
    ("2013/05/31", "233734"),
    ("2013/05/31", "234209"),
    ("2013/05/31", "234645"),
    ("2013/05/31", "235120"),
    ("2013/05/31", "235556"),
    ("2013/06/01", "000034"),
    ("2013/06/01", "000509"),
    ("2013/06/01", "000945"),
    ("2013/06/01", "001422"),
    ("2013/06/01", "001857"),
    ("2013/06/01", "002333"),
    ("2013/06/01", "002808"),
    ("2013/06/01", "003242"),
    ("2013/06/01", "003718"),
    ("2013/06/01", "004154"),
    ("2013/06/01", "004629"),
    ("2013/06/01", "005105"),
    ("2013/06/01", "005544"),
    ("2013/06/01", "010018"),
    ("2013/06/01", "010454"),
    ("2013/06/01", "010931"),
    ("2013/06/01", "011406"),
    ("2013/06/01", "011841"),
    ("2013/06/01", "012320"),
    ("2013/06/01", "012758"),
    ("2013/06/01", "013233"),
    ("2013/06/01", "013710"),
    ("2013/06/01", "014145"),
    ("2013/06/01", "014620"),
    ("2013/06/01", "015055"),
    ("2013/06/01", "015531"),
    ("2013/06/01", "020007"),
    ("2013/06/01", "020444"),
    ("2013/06/01", "020918"),
    ("2013/06/01", "021354"),
    ("2013/06/01", "021831"),
    ("2013/06/01", "022308"),
    ("2013/06/01", "022743"),
    ("2013/06/01", "023219"),
    ("2013/06/01", "023654"),
    ("2013/06/01", "024131"),
    ("2013/06/01", "024607"),
    ("2013/06/01", "025042"),
    ("2013/06/01", "025519"),
    ("2013/06/01", "025955"),
)

# --- Tornado track (NWS damage survey, via the SPC tornado database) ---------
#: The EF3 El Reno tornado, from the Storm Prediction Center's tornado database
#: (https://www.spc.noaa.gov/wcm/data/, the 1950-2023 CSV): 2013-05-31, Oklahoma,
#: magnitude 3, path length 16.2 miles, max width 4,576 YARDS = 4.18 km — the
#: widest tornado ever recorded. Its logged time is 17:03 in tz=3 (CST), i.e.
#: 23:03 UTC.
#:
#: Worth knowing as a cross-check: converting these damage-survey coordinates
#: into the radar frame puts the track at (-74, +17) -> (-52, +19) km, which is
#: exactly where the radar's hook echo appears. Two completely independent
#: sources — a ground damage survey and a reflectivity field — agreeing is the
#: strongest evidence available that the geolocation math here is right.
TORNADO_START_LATLON = (35.485, -98.096)
TORNADO_END_LATLON = (35.502, -97.848)
#: Touchdown and dissipation, UTC. The touchdown time is from the SPC record; the
#: 40-minute duration is from the NWS Norman damage survey narrative (the SPC
#: database carries no end time), so treat the END as documented-not-measured.
TORNADO_START_UTC = ("2013/05/31", "230300")
TORNADO_END_UTC = ("2013/05/31", "234300")
#: How tall to draw the marker, km. Not a physical claim about the vortex depth —
#: it is a pointer into the storm, tall enough to read against a 16 km cloud.
TORNADO_MARKER_TOP_KM = 12.0
#: Marker line width, km. Deliberately NOT the tornado's real 4.18 km width: a
#: 4 km wide, 12 km tall box reads as a slab and would falsely imply the vortex
#: was that deep. 0.2 km matches the hairline-to-scene ratio the other lines
#: demos use (rivers: 0.015 on a 200-unit globe) scaled to a 300 km domain.
TORNADO_MARKER_WIDTH_KM = 0.2
#: Marker colour. MUST be uint8. Passing a float triple like (255, 60, 200) is
#: silently wrong: float colours are HDR values in [0, 1], so 255 clips through
#: tone mapping and the marker renders pure WHITE instead of magenta.
TORNADO_MARKER_RGB = (255, 60, 200)
#: Width of the domain reference cube, km. Thinner than the tornado marker so
#: the furniture never competes with the subject.
BOX_WIRE_WIDTH_KM = 0.12

# --- Beam geometry ----------------------------------------------------------
#: 4/3-effective-earth radius in KILOMETRES. Everything downstream works in km,
#: which is also the unit MetPy reports gate ranges in -- see decode_volume.
RE_KM = 6371.0 * 4.0 / 3.0
#: WSR-88D half-power beamwidth (degrees).
BEAMWIDTH_DEG = 0.95
#: Split-cut clustering tolerance (degrees). Must sit BELOW VCP 212's minimum
#: true elevation spacing (0.4 deg, from 0.5 to 0.9) and ABOVE the measured
#: split-cut spread (0.12 deg). See _select_reflectivity_sweeps.
EL_CLUSTER_TOL_DEG = 0.25

# --- Cartesian storm box (FIXED for every timepoint) ------------------------
#: The supercell translates east-north-east all evening, so the box spans the
#: whole track rather than following the storm — a storm-following box would
#: pin the storm in place and make the ground appear to move instead.
#: The 300x300 km extent was MEASURED, not guessed. Pooling every >=20 dBZ gate
#: (the echo this demo actually renders) from four probe scans spanning 23Z to
#: 02Z, the retained fraction is:
#:     180x120 km (the original storm-scale box)  74.8%
#:     300x300 km (this)                          90.1%
#:     360x360 km                                 94.0%
#:     460x460 km (full surveillance disc)        97.1%
#: 300x300 buys the mesoscale context for ~5x the voxels; the last 7% costs
#: another 2.3x on top and lies at ranges where the beam is 3-4 km wide, so
#: there is little real structure left to resolve out there.
BOX_X_KM = (-160.0, 140.0)  # east of KTLX
BOX_Y_KM = (-110.0, 190.0)  # north of KTLX
BOX_Z_KM = (0.25, 18.25)  # above the feedhorn

# --- Barnes objective analysis ---------------------------------------------
ROI_KM = 1.6
BARNES_SIGMA_KM = ROI_KM / 2.0
#: Fixed-k truncation of the ROI ball. At sigma = 0.8 km the weight at the ROI
#: edge is exp(-2) = 0.135, so the nearest neighbours carry nearly all the mass;
#: truncation only bites where the sampling is already over-dense.
KNN = 32
#: Grid points per kd-tree query block, bounding peak memory.
QUERY_BLOCK = 262_144

# --- dBZ -> intensity -------------------------------------------------------
DBZ_CEIL = 70.0
#: Finite stand-in for "the radar saw nothing here", used once the gridded field
#: is cached (-inf does not round-trip cleanly through every reader). Any value
#: far below the floor works; it clips to zero intensity either way.
NO_DATA_DBZ = -999.0

# --- Fit / scene ------------------------------------------------------------
DEMO_NAME = "gsplats_nexrad_supercell"
CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME
_PRECOMPUTED_BUNDLE_NAME = "nexrad_supercell.gsplats.zarr.zip"
#: Peak amplitude after the single global rescale. This is 1.0 ON PURPOSE and is
#: NOT a brightness knob. On a colormapped node the scalar range is mapped into
#: the LUT through a display WINDOW that defaults to [0, 1]; amplitudes peaking
#: at (say) 0.2 therefore land in the bottom fifth of the ramp — every splat
#: comes out navy, and accumulating navy washes the storm to a flat pale cyan
#: with no hail core. Verified in the viewer. Normalising the peak to 1.0 makes
#: the whole colour ramp reachable; brightness is SCENE_OPACITY and the usable
#: sub-range is SCENE_WINDOW.
AMPLITUDE_PEAK = 1.0
#: Appearance, dialled in interactively in the viewer's Layers panel and then
#: baked here so the demo opens the way it was tuned. These four values are ONE
#: coherent look — changing any of them alone will not reproduce it.
#:
#: `turbo` maps blue -> cyan -> green -> yellow -> red almost band-for-band onto
#: the NWS reflectivity palette a meteorologist reads by eye, and because the
#: dBZ -> amplitude mapping is linear (see dbz_to_intensity) the colour bands land
#: at their conventional dBZ values. Colour here is therefore MEANINGFUL, not
#: decorative: light rain blue/green, heavy rain yellow, hail core red.
SCENE_COLORMAP = "turbo"
#: Very LOW opacity, and that is the trick that makes this work. Each splat
#: contributes only a sliver, so the hundreds that overlap along a view ray
#: accumulate into a smooth gradient instead of saturating to white — which is
#: exactly what a thick cloud does to light. Turning this up is the fastest way
#: to destroy the look.
SCENE_OPACITY = 0.05
#: Moderate optical depth, paired with the low opacity above. Absorption is what
#: makes near splats occlude far ones, so the image approaches per-splat colour
#: rather than a sum along the ray (a sum of colourmapped splats corresponds to no
#: dBZ value at all). Too little and the storm reads as a flat pale wash; too much
#: and its interior is buried behind its own surface.
SCENE_ABSORPTION = 2.11
#: Scalar display WINDOW, as authored `intensity`/`offset`. On a colormapped node
#: these are NOT a brightness gain — the viewer resolves them to the LUT window
#: `[-offset/intensity, (1 - offset)/intensity]` (see
#: `rendering/display-range.ts::computeDisplayRange`), which is why brightness is
#: SCENE_OPACITY's job and not theirs. These are the exact solution for the
#: window below:  intensity = 1 / (hi - lo),  offset = -lo * intensity.
#: The window spans essentially the whole amplitude range, so no reflectivity is
#: clipped out of the colour scale — the low opacity, not a narrow window, is
#: what keeps the dense cores from blowing out.
SCENE_WINDOW = (0.0, 0.88)
SCENE_INTENSITY = 1.0 / (SCENE_WINDOW[1] - SCENE_WINDOW[0])
SCENE_OFFSET = -SCENE_WINDOW[0] * SCENE_INTENSITY

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

MAX_TIMEPOINTS = min(
    parse_int_arg("max-timepoints", len(VOLUME_SCANS)), len(VOLUME_SCANS)
)
#: Cartesian grid spacing, metres. 750 m is chosen to MATCH THE RADAR'S REAL
#: RESOLUTION, not the gate spacing: the 0.95 deg beam is ~660 m across at 40 km
#: and ~1660 m at 100 km, which is the range band this storm occupies. A 500 m
#: grid therefore oversamples the beam by 2-3x — it invents detail the radar
#: never resolved, and costs 3.4x more voxels (2.42 M vs 0.71 M), which dominates
#: fit time because every iteration renders the whole volume.
GRID_M = parse_int_arg("grid-m", 750)
#: VERTICAL grid spacing, metres — deliberately COARSER than the horizontal one.
#: The radar's vertical sampling is far sparser than its horizontal sampling and
#: gets rapidly worse with height: at 60 km range adjacent VCP-212 tilts are
#: 0.42 km apart near the ground but 3.35 km apart aloft. Gridding the vertical
#: at the horizontal 750 m therefore over-samples it by 2-4x above ~5 km, and
#: each lone tilt smears down a whole column of cells — visible as vertical
#: banding in a side view. 1500 m is the honest compromise across the column.
GRID_Z_M = parse_int_arg("grid-z-m", 1500)
#: Splat budget per timepoint. 0 = ADAPTIVE, which is the default and matters
#: here: the system grows about 6x through the night (22k gates of >=45 dBZ echo
#: at 23Z against 131k at 02Z), so one fixed K would over-fit the opening frames
#: and badly under-resolve the late ones. Adaptive keeps a constant number of
#: occupied voxels per splat, so detail tracks the storm instead of the clock.
#: NOTE for maintenance sweeps: this demo is deliberately NOT registered in
#: scripts/update_demo_max_splats.py or scripts/calibrate_gsplat_demos.py. Both
#: exist to pick and write a single `MAX_SPLATS = N` per demo from a `gsplat cal`
#: K* sweep; this demo has no such constant because the budget is per-frame and
#: adaptive, so there is nothing for them to rewrite. Its budget was instead
#: validated by measuring reconstruction PSNR directly against the gridded
#: volume (table under VOXELS_PER_SEED), which is a more direct check than a
#: blind-spot K* estimate. Nor is it in scripts/add_additive_lod_to_demos.py:
#: that adds ladders to the per-frame LFS baselines, and
#: `combine_as_new_dimension` discards a per-frame ladder when it stacks the 4D
#: dataset, so the ladder has to be (and is) applied at scene-write time.
SPLATS_OVERRIDE = parse_int_arg("splats", 0)
#: Occupied (non-transparent) voxels per SEED under the adaptive budget.
#:
#: Named for seeds, not splats, because post-fit culling (cull_retention below)
#: removes roughly 70% of them: at 4 voxels/seed the DELIVERED density measures
#: 12-15 voxels per surviving splat. Treat this as a compute knob, not as the
#: reconstruction density.
#:
#: Measured on the busiest frame (224k occupied voxels), fitting in voxel space
#: and scoring the reconstruction against the gridded volume:
#:     seeds/voxel   seeds     splats    PSNR
#:        1/16       14,007     6,565   42.5 dB
#:        1/8        28,014    12,804   44.7 dB
#:        1/4 (this) 56,028    15,090   45.1 dB
#:        1/2       112,056    27,054   47.2 dB   <- best
#:        1/1       224,112    28,077   46.4 dB   <- regresses; culling caps it
#: So 4.0 leaves ~2 dB unclaimed and 2.0 is the optimum, at ~2.8x the fit time.
#: 45 dB already comfortably exceeds the other gsplat demos in this repo (the
#: TNG cosmic web ships at 34 dB), and at this demo's low opacity the difference
#: is not visible, so 4.0 is kept deliberately rather than by omission.
VOXELS_PER_SEED = 4.0
#: Floor and ceiling for the adaptive budget. The ceiling bounds fit time and GPU
#: memory; the floor stops a handful of voxels being fitted with one splat.
SPLATS_MIN, SPLATS_MAX = 50, 60_000
#: Occupancy below which a timepoint is treated as EMPTY SKY and skipped.
#:
#: This MUST be its own constant rather than reusing SPLATS_MIN. It was briefly
#: conflated with it, and the effect was measured: at a 1500-voxel threshold,
#: frames 7 and 8 (522 and 1,334 occupied voxels) were thrown away as "empty"
#: even though they hold the storm's FIRST CELLS — so the supercell popped into
#: existence fully formed instead of emerging. An occupancy floor and a splat
#: floor answer different questions; 50 keeps genuine initiation echo while
#: still skipping the truly bare 21Z frames (10-66 voxels of speckle).
MIN_OCCUPIED_VOXELS = 50
#: Gaussian truncation radius, in sigmas. Taken from the shared core constant and
#: passed to BOTH the fitter and the empty-sky placeholder, because
#: `GSplatData.concatenate` REFUSES to mix datasets with different radii
#: ("Truncation radius mismatch: dataset 0 has 3.0, dataset 9 has 2.75") — which
#: it did, blowing up the 4D stack at the very end of an 82-frame build when the
#: placeholder was built from the bare constructor. #1216 has since unified the
#: constructor and fitter defaults on this value, so the mismatch can no longer
#: arise; sourcing it from one place keeps that true if the default ever moves.
TRUNCATE_SIGMAS = DEFAULT_TRUNCATION_RADIUS
#: Vertical exaggeration applied AFTER fitting, as a mass- and
#: covariance-consistent diagonal transform. 1 = true to scale. The storm is a
#: 7.5:1 pancake (129 km wide, 17 km deep), which is real, so exaggeration is
#: off by default; 2-3 makes the vault and overshooting top far more legible.
VERT_EXAG = float(parse_int_arg("vert-exag", 1))
#: dBZ below which a cell is fully transparent. 20 dBZ is the conventional
#: "this is precipitation" threshold, and a rendered sweep over 5/10/15/20/25/30
#: picked it: at 5-15 the domain fills with clear-air and insect return plus
#: concentric ground-clutter rings around the radar, while 25+ starts eating the
#: anvil and forward-flank shield. 20 removes both artifacts and keeps the storm.
DBZ_FLOOR = float(parse_int_arg("dbz-floor", 20))
RELIST = "--relist" in sys.argv

Arbol.max_depth = 5
DEVICE: Optional[str] = None

# Memo for the coverage mask, which depends only on the box, the spacing and the
# VCP's elevation limits -- identical for every timepoint in the hour.
_COVERAGE_MEMO: dict[tuple, np.ndarray] = {}


# =============================================================================
# Download
# =============================================================================


def _list_volume_keys(site: str = SITE, date_prefix: str = "2013/05/31") -> list[str]:
    """List the volume-scan times available for one radar-day (maintenance only).

    The normal run never calls this -- VOLUME_SCANS is pinned. Used by --relist
    to regenerate that tuple, and as a diagnostic if a pinned key ever 404s.

    Returns:
        HHMMSS strings, ascending.
    """
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    times: list[str] = []
    token = ""
    while True:
        url = f"{BUCKET_HOST}/?list-type=2&prefix={date_prefix}/{site}/"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            root = ET.fromstring(resp.read())
        for contents in root.findall(f"{ns}Contents"):
            key = contents.findtext(f"{ns}Key") or ""
            name = key.rsplit("/", 1)[-1]
            if name.endswith("_V06.gz") and "_" in name:
                times.append(name.split("_")[1])
        if (root.findtext(f"{ns}IsTruncated") or "false").lower() != "true":
            break
        token = root.findtext(f"{ns}NextContinuationToken") or ""
        if not token:
            break
    return sorted(times)


def download_volume(date_prefix: str, hhmmss: str) -> Path:
    """Fetch one Level II volume scan into the demo cache."""
    filename = f"{SITE}{date_prefix.replace('/', '')}_{hhmmss}_V06.gz"
    url = f"{BUCKET_HOST}/{date_prefix}/{SITE}/{filename}"
    return cached_download(url, DEMO_NAME, filename)


def scan_label(index: int) -> str:
    """Human-readable UTC stamp for a pinned scan, e.g. ``06-01 00:09:45Z``."""
    date_prefix, hhmmss = VOLUME_SCANS[index]
    return (
        f"{date_prefix[5:].replace('/', '-')} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}Z"
    )


def domain_box_wireframe() -> tuple[np.ndarray, np.ndarray]:
    """The 8 corners and 12 edges of the analysis domain, for a reference cube.

    Gives the storm a frame of reference: without it there is no cue for how big
    the domain is, where the ground plane sits, or how much of the scene is empty
    sky rather than absent data. Mirrors the reference cube in
    ``demo_cosmicflows_laniakea``.

    Returns:
        ``(corners, edges)`` -- corners ``(8, 3)`` in ``(up, north, east)`` order
        to match the scene's ``dim_order``, and edges ``(12, 2)`` vertex-index
        pairs for ``line_type="indexed"``. Indexed rather than segments so the
        three edges meeting at each corner share one vertex row and join cleanly.
    """
    z0, z1 = BOX_Z_KM
    y0, y1 = BOX_Y_KM
    x0, x1 = BOX_X_KM
    corners = np.array(
        [
            [z0, y0, x0],
            [z0, y0, x1],
            [z0, y1, x1],
            [z0, y1, x0],  # bottom face
            [z1, y0, x0],
            [z1, y0, x1],
            [z1, y1, x1],
            [z1, y1, x0],  # top face
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # verticals
        ],
        dtype=np.uint32,
    )
    return corners, edges


def _utc_seconds(date_prefix: str, hhmmss: str) -> int:
    """Seconds since 2013-05-31 00:00 UTC. Handles the midnight rollover."""
    day = 86_400 if date_prefix == "2013/06/01" else 0
    return day + int(hhmmss[:2]) * 3600 + int(hhmmss[2:4]) * 60 + int(hhmmss[4:])


def scan_utc_seconds(index: int) -> int:
    """Absolute UTC seconds for a pinned scan."""
    return _utc_seconds(*VOLUME_SCANS[index])


def latlon_to_local_km(lat: float, lon: float) -> tuple[float, float]:
    """Ground position relative to the radar, in kilometres (east, north).

    A local tangent-plane approximation, which is all that is warranted: the
    tornado track spans ~23 km, over which the error against a proper geodesic is
    centimetres. Uses the standard metres-per-degree at the radar's latitude.
    """
    east = (lon - SITE_LON) * 111.320 * np.cos(np.deg2rad(SITE_LAT))
    north = (lat - SITE_LAT) * 110.574
    return float(east), float(north)


def tornado_position_km(utc_seconds: int) -> Optional[tuple[float, float]]:
    """Where the tornado was at a given instant, or None if it was not down.

    Linearly interpolates between the damage-survey start and end points. That is
    a deliberate simplification — the real path curved slightly and the forward
    speed varied — but the survey records only the two endpoints, so a straight
    interpolation is exactly as much as the source supports. Do not present the
    marker as a tracked vortex position; it is the surveyed path, sampled in time.
    """
    t0 = _utc_seconds(*TORNADO_START_UTC)
    t1 = _utc_seconds(*TORNADO_END_UTC)
    if not (t0 <= utc_seconds <= t1):
        return None
    frac = (utc_seconds - t0) / float(t1 - t0)
    e0, n0 = latlon_to_local_km(*TORNADO_START_LATLON)
    e1, n1 = latlon_to_local_km(*TORNADO_END_LATLON)
    return e0 + frac * (e1 - e0), n0 + frac * (n1 - n0)


def tornado_marker_segments(indices: list[int]) -> tuple[np.ndarray, list[int]]:
    """Vertical line segments marking the tornado, one per frame it was down.

    Returns ``(vertices, frames)`` where vertices are ``(2 * K, 4)`` in
    ``(up, north, east, time)`` order — consecutive PAIRS, for
    ``line_type="segments"``, so nothing connects one frame's marker to the next.
    """
    verts: list[list[float]] = []
    frames: list[int] = []
    for t, src in enumerate(indices):
        pos = tornado_position_km(scan_utc_seconds(src))
        if pos is None:
            continue
        east, north = pos
        verts.append([BOX_Z_KM[0], north, east, float(t)])
        verts.append([TORNADO_MARKER_TOP_KM, north, east, float(t)])
        frames.append(t)
    if not verts:
        return np.zeros((0, 4), dtype=np.float32), []
    return np.asarray(verts, dtype=np.float32), frames


def download_all_volumes(scan_times: list[tuple[str, str]]) -> list[Path]:
    """Fetch every requested volume scan, announcing provenance once."""
    print_data_provenance(
        title="NEXRAD Level II - KTLX, El Reno supercell (2013-05-31 21Z - 06-01 03Z)",
        source="NOAA/NWS WSR-88D via NOAA Open Data Dissemination",
        license="U.S. Government work, public domain (17 U.S.C. 105); NOAA attribution requested",
        url="https://registry.opendata.aws/noaa-nexrad",
        note=f"{len(scan_times)} volume scans, ~9.8 MB each, fetched anonymously.",
    )
    paths = []
    with asection(f"Downloading {len(scan_times)} volume scans"):
        for date_prefix, hhmmss in scan_times:
            paths.append(download_volume(date_prefix, hhmmss))
    return paths


# =============================================================================
# Beam geometry (4/3 effective earth radius)
# =============================================================================


def beam_height_and_arc(
    r_km: np.ndarray, el_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Height above the radar and great-circle arc distance for a radar gate.

    The standard 4/3-effective-earth-radius approximation: the beam is treated
    as straight over an earth inflated by 4/3, which reproduces normal
    atmospheric refraction to within a few tens of metres out to ~200 km.

    Args:
        r_km: Slant range from the radar, kilometres.
        el_deg: Beam elevation angle, degrees above horizontal.

    Returns:
        ``(z_km, s_km)`` -- height above the feedhorn and arc distance along
        the earth's surface, both in kilometres.
    """
    el = np.deg2rad(el_deg)
    z = np.sqrt(r_km**2 + RE_KM**2 + 2.0 * r_km * RE_KM * np.sin(el)) - RE_KM
    s = RE_KM * np.arcsin(r_km * np.cos(el) / (RE_KM + z))
    return z, s


def inverse_beam(s_km: np.ndarray, z_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slant range and elevation angle that reach a given arc distance / height.

    The exact inverse of :func:`beam_height_and_arc`, used to decide which
    Cartesian cells the radar could actually see.

    Returns:
        ``(r_km, el_deg)``.
    """
    theta = s_km / RE_KM
    el = np.arctan2(np.cos(theta) - RE_KM / (RE_KM + z_km), np.sin(theta))
    r = (RE_KM + z_km) * np.sin(theta) / np.cos(el)
    return r, np.rad2deg(el)


# =============================================================================
# Decode
# =============================================================================


def select_reflectivity_sweeps(sweeps: list) -> list[int]:
    """Indices of the sweeps to use -- one per TRUE elevation angle.

    VCP 212 uses SPLIT CUTS: the low elevations are scanned twice, once as a
    long-range surveillance cut and once as a shorter-range Doppler cut, and
    BOTH carry reflectivity. Keeping both double-counts every low tilt and
    biases the Barnes analysis toward the shorter-range cut.

    Deduplicating is fiddlier than it looks. The two halves differ slightly in
    mean elevation (0.80 vs 0.92 degrees for the nominal 0.9 degree cut), so
    rounding the angle splits them into different buckets; and the reported
    elevation NUMBER is a cut index, so it is not necessarily equal across the
    pair either. Instead cluster by mean elevation with a tolerance that sits
    below the VCP's minimum true spacing but above the split-cut spread, then
    keep the widest-range member of each cluster.

    Args:
        sweeps: ``Level2File.sweeps`` -- a list of lists of radials.

    Returns:
        Sweep indices, ascending in elevation.
    """
    candidates = []
    for idx, sweep in enumerate(sweeps):
        moments = sweep[0][4]
        if b"REF" not in moments:
            continue
        mean_el = float(np.mean([ray[0].el_angle for ray in sweep]))
        header = moments[b"REF"][0]
        candidates.append((mean_el, header.num_gates, idx))
    candidates.sort()

    kept: list[int] = []
    cluster: list[tuple[float, int, int]] = []
    for entry in candidates:
        if cluster and entry[0] - cluster[0][0] > EL_CLUSTER_TOL_DEG:
            kept.append(max(cluster, key=lambda c: c[1])[2])
            cluster = []
        cluster.append(entry)
    if cluster:
        kept.append(max(cluster, key=lambda c: c[1])[2])

    kept.sort(key=lambda i: float(np.mean([ray[0].el_angle for ray in sweeps[i]])))
    return kept


def decode_volume(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Decode one Level II volume scan into geolocated reflectivity gates.

    Args:
        path: A gzipped Archive II file.

    Returns:
        ``(xyz_km, dbz, meta)`` -- gate positions relative to the radar
        (east, north, up), their reflectivity, and scan metadata. Only finite
        gates inside the storm box (plus an ROI margin) are returned.
    """
    metpy_io = require_module("metpy.io")
    radar = metpy_io.Level2File(str(path))

    # The VOL constants block rides on the first radial of a cut, not every one.
    vol = next(
        (ray[1] for sweep in radar.sweeps for ray in sweep if ray[1] is not None), None
    )
    if vol is not None and abs(vol.lat - SITE_LAT) > 0.01:
        aprint(f"  WARNING: site latitude {vol.lat} != expected {SITE_LAT}")

    selected = select_reflectivity_sweeps(radar.sweeps)

    xs, ys, zs, vs = [], [], [], []
    el_angles, max_range_km = [], 0.0
    for idx in selected:
        sweep = radar.sweeps[idx]
        header = sweep[0][4][b"REF"][0]
        # MetPy reports gate_width and first_gate in KILOMETRES, not metres --
        # mixing them with a metre-based earth radius silently collapses the
        # whole storm into a sub-kilometre box.
        r_km = header.first_gate + np.arange(header.num_gates) * header.gate_width
        az = np.deg2rad([ray[0].az_angle for ray in sweep])[:, None]
        el = np.array([ray[0].el_angle for ray in sweep])[:, None]
        ref = np.array([ray[4][b"REF"][1] for ray in sweep], dtype=np.float32)

        n = min(ref.shape[1], r_km.size)
        ref, r = ref[:, :n], r_km[None, :n]

        z, s = beam_height_and_arc(r, el)
        # Azimuth is clockwise from true north, so sin -> east and cos -> north.
        xs.append((s * np.sin(az)).ravel())
        ys.append((s * np.cos(az)).ravel())
        zs.append(np.broadcast_to(z, ref.shape).ravel())
        vs.append(ref.ravel())
        el_angles.extend(el.ravel().tolist())
        max_range_km = max(max_range_km, float(r.max()))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    dbz = np.concatenate(vs)

    keep = np.isfinite(dbz)
    keep &= (x >= BOX_X_KM[0] - ROI_KM) & (x <= BOX_X_KM[1] + ROI_KM)
    keep &= (y >= BOX_Y_KM[0] - ROI_KM) & (y <= BOX_Y_KM[1] + ROI_KM)
    keep &= (z >= BOX_Z_KM[0] - ROI_KM) & (z <= BOX_Z_KM[1] + ROI_KM)

    xyz = np.column_stack([x[keep], y[keep], z[keep]]).astype(np.float64)
    meta = {
        "n_sweeps": len(selected),
        "el_min": float(np.min(el_angles)),
        "el_max": float(np.max(el_angles)),
        "r_max_km": max_range_km,
        "vcp": getattr(vol, "vcp", None),
    }
    return xyz, dbz[keep].astype(np.float32), meta


# =============================================================================
# Cartesian gridding
# =============================================================================


def grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three 1-D Cartesian axes of the storm box, in kilometres.

    ANISOTROPIC: the vertical step is ``GRID_Z_M``, the horizontal ones
    ``GRID_M``. See :data:`GRID_Z_M` for why the vertical must be coarser.
    """
    step_h = GRID_M / 1000.0
    step_v = GRID_Z_M / 1000.0
    return (
        np.arange(BOX_Z_KM[0], BOX_Z_KM[1], step_v),
        np.arange(BOX_Y_KM[0], BOX_Y_KM[1], step_h),
        np.arange(BOX_X_KM[0], BOX_X_KM[1], step_h),
    )


def _vertical_stretch() -> float:
    """Factor by which vertical distances are shrunk for neighbour search.

    The Barnes influence region has to be as anisotropic as the grid, otherwise
    a spherical ROI that comfortably spans 750 m horizontally falls short of the
    1-3 km vertical gaps between tilts, and the cells in between find no
    neighbour at all — the vertical banding this fixes. Dividing z by this
    factor before the kd-tree query makes the ROI an ellipsoid that is
    ``GRID_Z_M / GRID_M`` times taller than it is wide, in physical terms.
    """
    return GRID_Z_M / GRID_M


def coverage_mask(el_min: float, el_max: float, r_max_km: float) -> np.ndarray:
    """Boolean mask of cells the radar could actually see.

    Barnes interpolation happily extrapolates past the outermost tilt, which
    manufactures a hard sloping "ceiling" of invented echo above the storm and
    smears fill into the cone of silence below the lowest tilt. Rather than
    hiding those with a threshold, mask them geometrically: invert the beam
    model per cell and keep only cells that fall inside the swept elevation
    range (widened by half a beamwidth) and within the maximum range.

    Cached: the box, spacing and VCP are identical for every timepoint.
    """
    key = (el_min, el_max, r_max_km, GRID_M)
    if key in _COVERAGE_MEMO:
        return _COVERAGE_MEMO[key]

    z_ax, y_ax, x_ax = grid_axes()
    Z, Y, X = np.meshgrid(z_ax, y_ax, x_ax, indexing="ij")
    s = np.hypot(X, Y)
    r, el = inverse_beam(s, Z)
    half = 0.5 * BEAMWIDTH_DEG
    mask = (el <= el_max + half) & (el >= el_min - half) & (r <= r_max_km)

    _COVERAGE_MEMO[key] = mask
    return mask


def grid_volume(xyz_km: np.ndarray, dbz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Barnes-interpolate polar gates onto the Cartesian storm box.

    Uses a FIXED-K kd-tree query rather than a radius query: a ball query
    returns ragged neighbour lists, and reducing those needs a Python loop that
    dominates the runtime. A fixed-k query returns dense ``(M, k)`` arrays that
    reduce with two vectorised sums, and misses come back as ``inf`` so an
    empty neighbourhood is still unambiguous.

    Args:
        xyz_km: ``(N, 3)`` gate positions.
        dbz: ``(N,)`` reflectivity in dBZ.
        mask: Coverage mask from :func:`coverage_mask`.

    Returns:
        ``(nz, ny, nx)`` float32 reflectivity in dBZ, ``-inf`` where the radar
        saw nothing. The dBZ→intensity mapping is deliberately NOT applied here
        so that it stays a cheap post-cache knob — see :func:`dbz_to_intensity`.
    """
    spatial = require_module("scipy.spatial")

    z_ax, y_ax, x_ax = grid_axes()
    Z, Y, X = np.meshgrid(z_ax, y_ax, x_ax, indexing="ij")
    shape = Z.shape
    # Work in a space where vertical distances are divided by the vertical/
    # horizontal grid ratio, so the spherical ROI below becomes a physically
    # TALLER-than-wide ellipsoid matching how the radar actually samples.
    vz = _vertical_stretch()
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel() / vz])

    gates = xyz_km.copy()
    gates[:, 2] /= vz
    tree = spatial.cKDTree(gates)
    out = np.zeros(pts.shape[0], dtype=np.float32)

    for start in range(0, pts.shape[0], QUERY_BLOCK):
        block = pts[start : start + QUERY_BLOCK]
        dist, idx = tree.query(block, k=KNN, distance_upper_bound=ROI_KM, workers=-1)
        valid = np.isfinite(dist)
        # Misses index one past the end; clamp before gathering.
        idx = np.where(valid, idx, 0)
        w = np.where(valid, np.exp(-0.5 * (dist / BARNES_SIGMA_KM) ** 2), 0.0)
        den = w.sum(axis=1)
        num = (w * dbz[idx]).sum(axis=1)
        # np.where evaluates both branches, so guard the divide itself rather
        # than filtering afterwards -- otherwise every empty neighbourhood
        # raises an invalid-value warning.
        block_out = np.full(den.shape, -np.inf, dtype=np.float64)
        np.divide(num, den, out=block_out, where=den > 0)
        out[start : start + QUERY_BLOCK] = block_out

    grid = out.reshape(shape)
    grid[~mask] = -np.inf
    return grid


def dbz_to_intensity(dbz_grid: np.ndarray) -> np.ndarray:
    """Map a gridded dBZ field onto a non-negative intensity in [0, 1].

    Clips in dBZ SPACE rather than converting to linear reflectivity. Linear Z
    spans seven decades over the observed range, which would let the hail core
    consume the entire splat budget and leave the anvil numerically invisible.
    Clipping in dBZ is also what every NWS reflectivity palette does, so the
    colour bands land at their conventional values.

    The consequence worth knowing: the mapping is linear in dBZ and therefore
    LOGARITHMIC in returned power. A 50 dBZ core is 100x the power of a 30 dBZ
    region but only ~1.5x the amplitude. That is deliberate — it keeps the
    light rain and anvil visible so the storm reads as a structure (hook, vault,
    overshooting top) instead of one saturated blob.
    """
    span = DBZ_CEIL - DBZ_FLOOR
    intensity = np.clip(dbz_grid - DBZ_FLOOR, 0.0, span) / span
    return np.nan_to_num(intensity, nan=0.0, neginf=0.0).astype(np.float32)


def build_timepoint_grid(path: Path, index: int) -> np.ndarray:
    """Decode, geolocate and grid one volume scan to dBZ, caching the result.

    The cache holds the dBZ field, keyed on grid spacing only: re-tuning
    --splats or --dbz-floor then costs nothing, while changing --grid-m
    correctly invalidates it.
    """
    cache_file = CACHE_DIR / f"nexrad_dbz_{_geometry_token()}_frame{index:04d}.npz"
    if cache_file.exists() and not RECOMPUTE:
        with np.load(cache_file) as data:
            return data["dbz"]

    xyz, dbz, meta = decode_volume(path)
    aprint(
        f"  {meta['n_sweeps']} sweeps, VCP {meta['vcp']}, "
        f"{len(dbz):,} gates in box, el {meta['el_min']:.2f}-{meta['el_max']:.2f} deg"
    )
    mask = coverage_mask(meta["el_min"], meta["el_max"], meta["r_max_km"])
    grid = grid_volume(xyz, dbz, mask)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Replace -inf with a finite sentinel well below any real reflectivity, and
    # return the SAME array that a later cache hit would return -- otherwise the
    # cold and warm paths differ in a way that only shows up as a subtle
    # numerical discrepancy between the first run and every run after it.
    grid = np.where(np.isfinite(grid), grid, NO_DATA_DBZ).astype(np.float32)
    np.savez_compressed(cache_file, dbz=grid)
    return grid


# =============================================================================
# Fitting
# =============================================================================


def _geometry_token() -> str:
    """Cache-name fragment for the GRID GEOMETRY: box extent plus both spacings.

    The box MUST be in the key. It was briefly omitted, and the failure mode is
    nasty: widening the domain left the old grids and fits on disk under an
    unchanged name, so a re-run silently mixed volumes of different shapes and
    splat clouds in different coordinate frames. Grid spacing alone is not the
    geometry — the extent is half of it.
    """
    return (
        f"g{GRID_M}x{GRID_Z_M}"
        f"_b{int(BOX_X_KM[0])},{int(BOX_X_KM[1])}"
        f",{int(BOX_Y_KM[0])},{int(BOX_Y_KM[1])}"
        f",{int(BOX_Z_KM[0])},{int(BOX_Z_KM[1])}"
    )


def _budget_token() -> str:
    """Cache-name fragment identifying the splat-budget POLICY, not one K.

    Under the adaptive budget every frame gets its own K, so keying the cache on
    a single number would be a lie; key it on the policy that produced them.
    """
    if SPLATS_OVERRIDE:
        return f"k{SPLATS_OVERRIDE}"
    return f"vps{VOXELS_PER_SEED:g}"


def splat_budget(volume: np.ndarray) -> int:
    """Splats to fit for one timepoint (see :data:`SPLATS_OVERRIDE`)."""
    if SPLATS_OVERRIDE:
        return SPLATS_OVERRIDE
    occupied = int((volume > 0.0).sum())
    return int(np.clip(round(occupied / VOXELS_PER_SEED), SPLATS_MIN, SPLATS_MAX))


def _frame_cache_file(index: int) -> Path:
    """Per-frame splat cache path. Bundle inner names must match this exactly."""
    return CACHE_DIR / (
        f"nexrad_frame_{_geometry_token()}_{_budget_token()}_{index:04d}.gsplats.zarr.zip"
    )


def fit_timepoint(volume: np.ndarray, index: int, label: str) -> GSplatData:
    """Fit one gridded volume as 3D Gaussian splats, with caching."""
    global DEVICE

    cache_file = _frame_cache_file(index)
    if cache_file.exists() and not RECOMPUTE:
        result = GSplatData.load(cache_file, include_stats=False)
        aprint(f"  Loaded {result.n_splats:,} cached splats ({label})")
        return result

    n_splats = splat_budget(volume)
    occupied = int((volume > 0.0).sum())
    if occupied < MIN_OCCUPIED_VOXELS:
        # The 21Z frames are essentially empty (the storm has not initiated), and
        # fitting a near-blank volume wastes a GPU pass to produce noise. Emit a
        # single negligible splat so the timepoint still EXISTS on the time axis —
        # dropping it would silently renumber every later frame.
        aprint(f"  {label}: only {occupied:,} occupied voxels — empty-sky placeholder")
        result = GSplatData(
            centers=np.zeros((1, volume.ndim), dtype=np.float32),
            amplitudes=np.zeros(1, dtype=np.float32),
            cholesky_factors=np.tile(
                np.eye(volume.ndim, dtype=np.float32)[np.tril_indices(volume.ndim)],
                (1, 1),
            ).astype(np.float32),
            truncation_radius=TRUNCATE_SIGMAS,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.save(
            cache_file,
            encoding_mode=EncodingMode.MEMORY,
            include_fitting_info=False,
            compress="zip",
            zip_deflate=True,
        )
        return result

    if DEVICE is None:
        DEVICE = detect_device()

    from luxar.gsplats import fit_gaussian_splats

    aprint(f"  {label}: {occupied:,} occupied voxels -> {n_splats:,} splats")
    result = fit_gaussian_splats(
        volume,
        seeds=n_splats,
        n_iters=800,
        truncate=TRUNCATE_SIGMAS,
        device=DEVICE,
        # The dBZ clip already puts a hard zero below DBZ_FLOOR, so there is no
        # pedestal to estimate. "auto" would be a no-op here but makes the floor
        # frame-dependent in principle, and anything frame-dependent breaks a
        # timelapse.
        floor="none",
        # ANISOTROPIC voxel size (dz, dy, dx) in km, so fitted centers and
        # covariances come out in true kilometres despite the coarser
        # vertical grid. A scalar here would silently squash the vertical.
        voxel_size=(GRID_Z_M / 1000.0, GRID_M / 1000.0, GRID_M / 1000.0),
        # Keep more of the low-amplitude tail than the default 0.95: the anvil
        # and forward-flank shield are exactly that tail, and they are what make
        # the result read as a storm rather than a blob.
        cull_retention=0.98,
        verbose=True,
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.save(
        cache_file,
        encoding_mode=EncodingMode.MEMORY,
        include_fitting_info=True,
        compress="zip",
        zip_deflate=True,
    )
    return result


def fit_all_timepoints(paths: list[Path], indices: list[int]) -> list[GSplatData]:
    """Grid and fit every requested timepoint."""
    results = []
    with asection(f"Fitting {len(paths)} timepoints"):
        for path, index in zip(paths, indices):
            label = scan_label(index)
            with asection(f"Timepoint {index} ({label})"):
                volume = dbz_to_intensity(build_timepoint_grid(path, index))
                results.append(fit_timepoint(volume, index, label))
    return results


# =============================================================================
# 4D assembly
# =============================================================================


def combine_timepoints_to_4d(gsplats_list: list[GSplatData]) -> GSplatData:
    """Stack per-timepoint fits into one 4D dataset with time as the last axis.

    Two deliberate departures from the other 4D demos in this repo:

    * **No per-frame amplitude normalisation.** Zebrafish and C. elegans rescale
      each frame to a fixed peak because their signal strength varies for
      reasons that are not the subject. Here the storm's intensification and
      decay IS the subject, and a single global dBZ mapping already makes
      amplitudes comparable across frames, so one global scale is applied.
    * **No shared-centroid alignment.** Those demos subtract an
      amplitude-weighted centroid to stop frames jittering in their own voxel
      frames. Our grid box is identical for every timepoint, so there is no
      jitter -- and a centroid would actively introduce drift as the storm
      moves. A constant offset is used instead, so the axes read as true
      kilometres from the radar.
    """
    origin = np.array([BOX_Z_KM[0], BOX_Y_KM[0], BOX_X_KM[0]])  # centers are (z, y, x)
    global_max = max(float(g.amplitudes.max()) for g in gsplats_list)
    aprint(f"Global peak amplitude across timepoints: {global_max:.4f}")

    processed = [
        g.translate(origin).scale_intensity(AMPLITUDE_PEAK / global_max)
        for g in gsplats_list
    ]
    if VERT_EXAG != 1.0:
        # Applied AFTER the fit, never as an anisotropic voxel_size: `transform`
        # scales centers AND covariances together, so the splats stretch with the
        # geometry instead of the fit being solved in a distorted space. Centers
        # are (z, y, x), so the vertical is axis 0.
        stretch = np.diag([VERT_EXAG, 1.0, 1.0]).astype(np.float64)
        processed = [g.transform(stretch) for g in processed]
        aprint(f"Vertical exaggeration: {VERT_EXAG:g}x (post-fit, mass-preserving)")
    for t, g in enumerate(processed):
        aprint(
            f"  t={t:2d}  {g.n_splats:7,} splats  peak {float(g.amplitudes.max()):.4f}"
        )

    return GSplatData.combine_as_new_dimension(
        processed, values=[float(t) for t in range(len(processed))], sigma=0.0
    )


def create_luxar_scene(
    combined: GSplatData, output_path: Path, frame_indices: list[int]
) -> Path:
    """Write the 4D scene."""
    n_timepoints = int(combined.centers[:, -1].max()) + 1

    with asection("Creating Luxar scene"):
        dims = Dimensions(
            [
                Dimension(
                    "east",
                    unit="km",
                    display=True,
                    description="Distance east of KTLX (Twin Lakes, OK)",
                ),
                Dimension(
                    "north",
                    unit="km",
                    display=True,
                    description="Distance north of KTLX",
                ),
                Dimension(
                    "up",
                    unit="km",
                    display=True,
                    description="Height above the radar feedhorn (4/3-earth beam model)",
                ),
                Dimension(
                    "time",
                    unit="frame",
                    display=False,
                    discrete=True,
                    range=(0, n_timepoints - 1),
                    step=1.0,
                    description="Volume scan, ~4.5 min apart, 23:00-23:56 UTC",
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    camera=CameraConfig(
                        # WORLD UP MUST BE +Z HERE. The displayed dims map
                        # east->x, north->y, up->z, and the viewer's default up
                        # vector is (0, 1, 0) -- i.e. NORTH. Left at the default,
                        # orbiting side-on puts north on the screen vertical and
                        # lays the altitude axis sideways, so a storm 129 km wide
                        # and 17 km deep renders as a tall narrow sliver and the
                        # ground plane reads as a wall.
                        up=(0.0, 0.0, 1.0),
                        # Opening pose: OUTSIDE the domain, south-east and above,
                        # looking back at the storm. It must clear the box — the
                        # earlier pose was tuned for a 180x120 km domain and now
                        # sits inside the 300x300 km one, where the reference cube
                        # degenerates into a couple of stray edges crossing frame.
                        # ~300 km out, not the ~510 km that would frame the entire
                        # 425 km diagonal: the appearance above was tuned at a close
                        # view, and at full-domain distance the low opacity reads as
                        # faint. This keeps the storm substantial with most of the
                        # reference cube still in frame.
                        position=(84.0, -199.0, 132.0),
                        target=(-66.0, 26.0, 6.0),
                        fov=45.0,
                    ),
                ),
            )
            scene.attrs["title"] = (
                "El Reno Tornadic Supercell - NEXRAD Level II (2013-05-31)"
            )
            scene.attrs["description"] = (
                f"{n_timepoints} WSR-88D volume scans from KTLX (Twin Lakes, OK), "
                "23:00-23:56 UTC on 2013-05-31, regridded from polar to Cartesian "
                "and fitted as Gaussian splats. The tornado touched down at 23:03 "
                "and dissipated at 23:43 UTC. Colour is radar reflectivity: blue "
                "light rain, yellow heavy rain, red the hail core."
            )

            # Data columns are (z, y, x, time); map them onto the named scene
            # dims. Time MUST stay last in BOTH orderings -- the LOD barrier
            # below is expressed as center-column indices, while the compiler's
            # chunk-ordering barrier is read from the scene dimension list.
            scene.add_gsplats_from_data(
                name="reflectivity_4d",
                result=combined,
                dim_order=["up", "north", "east", "time"],
                extend_to_all=[],
                # NO substitutive (coarse-replacement) LOD here, deliberately.
                # A few hundred thousand splats is small enough to draw whole,
                # and the merged coarse levels actively hurt: their
                # mass-conserving amplitudes are averaged down, so the hail core
                # renders yellow-green instead of red, and the storm turns into
                # a blur. Measured in the viewer on a single-timepoint build.
                #
                # A streaming ladder IS kept — it is required, not cosmetic: a
                # single leaf this size trips scripts/check_demo_ladders.py, and
                # a ladder baked into the cached per-frame files does not
                # survive the 4D stack. Note the gsplats spelling is
                # `breakpoints=`; the `counts=` form the Points/Lines demos use
                # is rejected here.
                additive_lod=dict(breakpoints="stream:20000", seed=0),
                colormap=SCENE_COLORMAP,
                blending_mode="volumetric",
                absorption=SCENE_ABSORPTION,
                opacity=SCENE_OPACITY,
                intensity=SCENE_INTENSITY,
                offset=SCENE_OFFSET,
                gamma=1.0,
                layer=True,
            )

            # Domain wireframe: a faint reference cube around the whole analysis
            # box. `extend_to_all=["time"]` makes ONE cube persist across every
            # frame — replicating it per timepoint would write 82 copies and make
            # it blink under LOD streaming.
            corners, edges = domain_box_wireframe()
            scene.add_lines(
                "domain box",
                vertices=corners,
                widths=BOX_WIRE_WIDTH_KM,
                # Float colours here are HDR in [0, 1] — see TORNADO_MARKER_RGB
                # for what happens if you pass 0-255 floats by mistake.
                colors=(0.55, 0.62, 0.75),
                sharpness=0.8,
                indices=edges.ravel(),
                line_type="indexed",
                dim_order=["up", "north", "east"],
                fill={"time": 0.0},
                extend_to_all=["time"],
                opacity=0.18,
                blending_mode="normal",
                layer=True,
            )

            # Tornado marker: a vertical line at the surveyed tornado position,
            # present only on the frames when it was on the ground.
            verts, frames = tornado_marker_segments(frame_indices)
            if len(frames):
                aprint(
                    f"Tornado marker on {len(frames)} frames "
                    f"(t={frames[0]}..{frames[-1]}, 23:03-23:43 UTC)"
                )
                scene.add_lines(
                    name="tornado_track",
                    vertices=verts,
                    widths=TORNADO_MARKER_WIDTH_KM,
                    # Consecutive PAIRS are independent segments — a polyline
                    # would join one frame's marker top to the next frame's base.
                    line_type="segments",
                    dim_order=["up", "north", "east", "time"],
                    extend_to_all=[],
                    colors=np.tile(
                        np.asarray(TORNADO_MARKER_RGB, dtype=np.uint8), (len(verts), 1)
                    ),
                    # NOT additive: additive lines have no occlusion, so the
                    # marker would glow straight through the storm in front of it
                    # and read as a decal rather than an object in the scene.
                    blending_mode="normal",
                    opacity=1.0,
                    layer=True,
                )

            scene.add_text(
                "El Reno Supercell - NEXRAD Level II",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            for t in range(n_timepoints):
                stamp = scan_label(t) if t < len(VOLUME_SCANS) else ""
                scene.add_text(
                    f"2013-{stamp}" if stamp else "",
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"time": t},
                    transition="fade",
                    transition_duration=0.15,
                )
            scene.add_text(
                "NEXRAD Level II - KTLX Twin Lakes, OK - NOAA NODD (public domain)",
                position=(0.98, 0.97),
                font_size=0.015,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

    aprint(f"Scene written to {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================


def _select_frame_indices(n: int) -> list[int]:
    """Evenly subsample the pinned scan list down to ``n`` timepoints."""
    if n >= len(VOLUME_SCANS):
        return list(range(len(VOLUME_SCANS)))
    stride = max(1, len(VOLUME_SCANS) // n)
    return list(range(0, len(VOLUME_SCANS), stride))[:n]


def main() -> None:
    """Main demo execution."""
    aprint("=" * 70)
    aprint("GSplats Demo: 4D NEXRAD El Reno Tornadic Supercell (2013-05-31)")
    aprint("=" * 70)
    aprint("Weather radar -> Cartesian volume -> 4D Gaussian splats")
    aprint("")

    if RELIST:
        # The pinned window crosses midnight, so emit (date, time) pairs for both
        # radar-days in the paste-ready form VOLUME_SCANS expects.
        pairs: list[tuple[str, str]] = []
        for date_prefix, hours in (
            ("2013/05/31", range(21, 24)),
            ("2013/06/01", range(0, 3)),
        ):
            for t in _list_volume_keys(date_prefix=date_prefix):
                if int(t[:2]) in hours:
                    pairs.append((date_prefix, t))
        aprint(f"{len(pairs)} volume scans for {SITE} across the 21Z-03Z window:")
        aprint("VOLUME_SCANS: tuple[tuple[str, str], ...] = (")
        for i in range(0, len(pairs), 3):
            aprint("    " + " ".join(f'("{d}", "{t}"),' for d, t in pairs[i : i + 3]))
        aprint(")")
        return

    output_path = get_demos_output_dir() / "gsplats_4d_nexrad_supercell.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            aprint("Serve-only mode: Launching viewer...")
            launch_viewer(output_path)
        else:
            aprint(f"No scene found at {output_path}. Run without --serve-only first.")
        return

    indices = _select_frame_indices(MAX_TIMEPOINTS)
    file_names = [_frame_cache_file(i).name for i in indices]

    try:
        gsplats_list = load_precomputed_bundle(
            DEMO_NAME, _PRECOMPUTED_BUNDLE_NAME, file_names, recompute=RECOMPUTE
        )
    except FileNotFoundError:
        aprint(
            "Precomputed bundle unavailable (Git LFS asset not pulled). "
            "Falling back to recomputing from the NEXRAD archive."
        )
        gsplats_list = None

    if gsplats_list is None:
        warn_if_no_cuda_gpu()
        paths = download_all_volumes([VOLUME_SCANS[i] for i in indices])
        gsplats_list = fit_all_timepoints(paths, indices)

    combined = combine_timepoints_to_4d(gsplats_list)
    aprint(f"Combined 4D dataset: {combined.n_splats:,} splats, {combined.ndim}D")

    scene_path = create_luxar_scene(combined, output_path, indices)

    if not NO_SERVE:
        aprint("\nLaunching viewer...")
        launch_viewer(scene_path)

    aprint("\nDone!")


if __name__ == "__main__":
    main()
