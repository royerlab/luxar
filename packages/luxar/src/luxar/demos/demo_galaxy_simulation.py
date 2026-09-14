#!/usr/bin/env python3
"""Self-Contained Demo: Galaxy Simulation — a density-wave spiral, integrated

A barred-free Sb spiral galaxy built from its own gravity rather than drawn.
Every star sits on a real orbit in a real potential, the spiral arms are a
density wave rather than a shape, and time is an axis you can step through and
watch the two come apart.

================================================================================
WHY THIS IS NOT A DRAWN SPIRAL
================================================================================

The obvious way to make a spiral galaxy is to scatter points along a few
logarithmic curves and rotate the whole picture. That is what this demo used to
do, and it is wrong in a way that matters: a galaxy's disc does NOT rotate
rigidly. It rotates *differentially* — the inner disc goes round several times
for every outer revolution — so any arm made of a fixed set of stars shears into
a tight rag within two or three rotations. This is the **winding problem**, and
it is the observation that killed material-arm models.

The resolution (Lin & Shu 1964) is that arms are a **density wave**: a
quasi-stationary spiral pattern in the gravitational potential that rotates
rigidly at a single pattern speed, while individual stars orbit at their own
speed and simply crowd together as they pass through it. Stars flow *through*
the arms the way cars crowd through a slow patch of motorway.

This demo shows exactly that, because it computes both halves separately:

* every star's guiding centre advances at its own angular speed ``Omega(R)``,
  derived from the galaxy's mass model;
* the spiral pattern advances at a single fixed ``Omega_p``.

So as you step the time slider, the stars visibly **overtake** the arms inside
the corotation radius and **fall behind** them outside it — the signature of a
density wave, and the thing a rigidly rotated picture can never show.

================================================================================
WHAT IS ACTUALLY COMPUTED
================================================================================

**1. A mass model, and the rotation curve it implies.**
   Three components, each with its standard analytic potential:

   * a Hernquist bulge, ``M = 1.2e10 Msun``, ``a = 0.7 kpc``
   * a Miyamoto-Nagai disc, ``M = 6e10 Msun``, ``a = 3.5``, ``b = 0.25 kpc``
   * an NFW dark halo, ``r_s = 16 kpc``, ``c = 12``, ``v200 = 190 km/s``

   Their circular speeds add in quadrature, giving a curve that rises to
   ~260 km/s by 5 kpc and stays **flat** out past 30 kpc — the observation that
   dark haloes exist. Nothing about the shape is imposed; it falls out.

**2. The frequencies that follow from it.** ``Omega = v_c/R`` and the epicyclic
   frequency ``kappa^2 = (2 Omega / R) d(R^2 Omega)/dR``, differentiated
   numerically off the analytic curve. With the pattern speed set to
   ``Omega_p = 23 km/s/kpc`` these place the resonances at

   * inner Lindblad resonance  ~1.7 kpc
   * **corotation             ~11.4 kpc**
   * outer Lindblad resonance ~19.4 kpc

   Corotation is where the arms are printed on the disc — the one radius at
   which stars and pattern keep step. Look there while scrubbing time.

**3. Each star's orbit as a forced epicycle.** A star is not on a circle. It has
   a guiding radius ``R_g`` and oscillates about it radially at ``kappa`` and
   vertically at ``nu``, tracing an epicycle whose tangential-to-radial axis
   ratio is exactly ``2 Omega / kappa``. On top of that free motion it feels the
   spiral, and responds with the standard forced-oscillator amplitude carrying
   the resonant denominator ``kappa^2 - m^2 (Omega - Omega_p)^2``. That
   denominator is what makes the arms strong between the resonances and fade
   towards them; it is softened here so the demo does not divide by zero at the
   resonances themselves, where linear theory stops being valid anyway.

**4. Ages, and the age-velocity dispersion relation.** Disc stars are heated by
   scattering as they age: ``sigma_R ~ age^0.33``. A star's epicyclic amplitude
   is ``sigma_R / kappa``, so this is not a cosmetic choice — it *derives* the
   most recognisable fact about spiral galaxies, that **the arms are a young
   population**. Solo the youngest age layer and watch the arms sharpen to knife
   edges in the youngest bin and dissolve into a smooth disc in the oldest.
   Vertical thickness is heated the same way, so the old disc is also the thick
   one.

**5. Colour from physics, not from a palette.** A star's age sets an effective
   temperature; the temperature is turned into RGB by integrating the Planck
   spectrum against the CIE 1931 colour-matching functions and converting
   through the sRGB primaries. That is why the young arms come out genuinely
   blue-white and the bulge genuinely amber — those are the colours those
   temperatures *are*.

**6. Dust lanes.** Gas shocks on the upstream edge of the arm, which is the
   inner edge inside corotation, and that is where the dust sits — which is why
   real spiral arms have a dark thread just inside the bright one. Stars caught
   in the lane are attenuated with a wavelength-dependent extinction, so they go
   dim *and* red rather than just dim.

**7. HII regions.** The same shock triggers star formation, so the youngest
   clusters ionise the gas around them. They are drawn as their own layer in
   H-alpha magenta with a trace of [OIII] teal, and they pick out the arms more
   sharply than the stars do.

**8. Bulge, halo and globular clusters.** A Hernquist bulge of old amber stars;
   a sparse, pressure-supported stellar halo; and ~120 globular clusters on
   plunging orbits, each a small Plummer knot.

================================================================================
NAVIGATION
================================================================================

Four dimensions: three spatial plus **T** (time). Stellar age is exposed as
five LAYERS rather than as a fifth dimension — see ``AGE_BIN_LABELS`` for why.

* Press **1**, then **[** / **]** — step time over 480 Myr in 2 Myr frames.
  (The digit keys index the NAVIGABLE dimensions, so time is 1. Age is exposed
  as layers, not as a dimension.)
* Or drag the **T** slider, or press play. It is a DISCRETE axis, so it snaps to
  the 2 Myr frame grid — every stop is a frame that has stars in it, and at the
  viewer's default 10 fps the 241 frames play as a 24 s loop.
* Press **L** — the Layers panel: disc, HII regions, bulge, halo, globulars.

Usage:
    python demo_galaxy_simulation.py [--stars=N] [--frames=N] [--no-serve]
"""

DEMO_META = {
    "key": "galaxy_simulation",
    "title": "Galaxy Simulation",
    "description": (
        "A spiral galaxy integrated from its own mass model: flat rotation "
        "curve, density-wave arms that stars overtake, age-heated populations, "
        "dust lanes and HII regions."
    ),
    "category": "synthetic",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": [],
    "outputs": ["galaxy_simulation"],
    # Procedurally simulated: no external dataset. The physics is textbook and
    # the references are in the module docstring, so there is nothing to credit
    # as DATA.
    "citation": None,
}

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.demos._lod_policy import hidden_axis_stops, stream_ladder
from luxar.utils.lod_breakpoints import (
    DEFAULT_MAX_ADDITIVE_COMMIT,
    capped_stream_cuts,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Physical constants and the mass model
# =============================================================================

#: Gravitational constant in the galactic unit system: kpc (km/s)^2 / Msun.
G = 4.300917270e-6

#: km/s expressed in kpc/Gyr, so an angular speed quoted in the astronomers'
#: km/s/kpc can be multiplied straight into radians per Gyr.
KMS_TO_KPC_PER_GYR = 1.0227121650537

# Hernquist bulge.
M_BULGE, A_BULGE = 1.2e10, 0.7
# Miyamoto-Nagai disc (the POTENTIAL's scale lengths, not the light profile's).
M_DISC, A_DISC, B_DISC = 6.0e10, 3.5, 0.25
# NFW dark halo.
R_S_NFW, C_NFW, V200_NFW = 16.0, 12.0, 190.0

#: Exponential scale length of the STELLAR disc's surface density. Distinct from
#: the Miyamoto-Nagai `A_DISC` above, which parameterises the potential.
R_DISC_SCALE = 3.4
#: Outer truncation of the stellar disc.
R_DISC_MAX = 22.0
#: Inner radius below which the disc gives way to the bulge.
R_DISC_MIN = 0.7

# =============================================================================
# The spiral pattern
# =============================================================================

#: Azimuthal harmonic. m=2 is the two-armed grand design.
M_ARMS = 2
#: Pattern speed, km/s/kpc. Sets corotation at ~11.4 kpc on this rotation curve,
#: which is ~0.5 of the disc's optical extent — where measured pattern speeds
#: put it in real grand-design spirals.
OMEGA_P = 23.0
#: Pitch angle of the arms in degrees. 13 deg is a typical Sb; smaller is more
#: tightly wound (Sa), larger is flocculent (Sc).
PITCH_DEG = 13.0
#: Spiral forcing coefficient in kpc·(km/s/kpc)^2. After division by the
#: resonant denominator it produces a 0.18 kpc displacement at the reference
#: radius, and 4-6% of R across the disc: visible but still in the linear regime.
SPIRAL_FORCING = 620.0
#: Resonance softening: the forced-response denominator is never allowed below
#: this fraction of kappa^2. Linear theory diverges at the Lindblad resonances
#: and stops being valid there, so the honest thing is to cap the response
#: rather than to let it blow up.
RESONANCE_SOFTENING = 0.30

# =============================================================================
# Populations
# =============================================================================

#: Age of the disc, Gyr. Star formation is taken as continuous over this span.
DISC_AGE_MAX = 9.0
#: Radial velocity dispersion at 1 Gyr, km/s, and the heating exponent. The
#: exponent is the observed age-velocity dispersion slope for the solar
#: neighbourhood; the pair is what makes old stars leave the arms.
SIGMA_R_1GYR, SIGMA_HEATING_EXPONENT = 22.0, 0.33
#: Vertical-to-radial dispersion ratio for the disc.
SIGMA_Z_RATIO = 0.55

#: Age-bin edges in Gyr, and their labels.
#:
#: These are LAYERS, not a navigable dimension, and that is a deliberate
#: reversal. A star has exactly one age, so an Age *dimension* would slice: the
#: viewer opens on one bin and the galaxy shows up as the 4% of stars that
#: happen to be young (measured — the first build opened on "< 50 Myr" and drew
#: 22k of 6.5M points). As layers every bin is on at once, the default view is
#: the whole galaxy, and the Layers panel still lets you isolate any one of them
#: — which is the comparison the age split exists for.
AGE_BIN_EDGES = (0.0, 0.05, 0.3, 2.0, 6.0, 100.0)
AGE_BIN_LABELS = (
    "< 50 Myr",
    "50-300 Myr",
    "0.3-2 Gyr",
    "2-6 Gyr",
    "> 6 Gyr",
)

#: Rows rung 0 must budget PER FRAME on an age-bin layer, not per node.
#:
#: The bins are wildly uneven by construction — at the authored seed they hold
#: 110 / 966 / 12,455 / 44,569 / 41,900 stars, and every star is present in every
#: frame, so a layer is exactly ``stars x n_frames`` rows with an exactly uniform
#: per-frame occupancy. ``stream_ladder`` sizes rung 0 as a SHARE of the node
#: (``n/8``), which is the right contract for the three fat bins and useless for
#: ``50-300 Myr``: 966 x 241 = 232,806 rows is only 16% over the ladder gate's
#: 200,000 "needs a ladder at all" threshold, so its share rung stayed at the
#: unsliced 39,062-element download budget and spread to a MEASURED 144 rows at
#: the sparsest 5th-percentile frame — under the 250-element absolute
#: first-paint floor in ``scripts/check_demo_ladders.py``, which is the arm that
#: catches a played axis rendering nothing.
#:
#: 300 rather than 250 because the seeded random prefix is a multinomial over the
#: frames, not an even deal: its p05 frame lands ~8-11% below the mean (measured
#: 144 against a 162 mean at the old 39,062 rung). At 300 x 241 = 72,300 rows the
#: measured p05 frame is 275 — 10% clear of the floor — and the ladder resolves to
#: 3 rungs of 72,300 / 72,300 / 88,206, whose largest increment is 37.9% of the
#: node, inside the gate's 0.6 degeneracy bound.
#:
#: NOT fixed by merging the two thin bins. The five-way age split is the
#: comparison this demo exists for (see ``AGE_BIN_LABELS`` above), #2657 asks for
#: ladders "without changing scene content", and merging would barely move the
#: arithmetic anyway: ``< 50 Myr`` + ``50-300 Myr`` is 1,076 stars = 259,316 rows,
#: still over the 200,000 threshold and still needing ~28% of the node in rung 0
#: to put 250 rows in its sparsest frame.
DISC_FIRST_RUNG_ROWS_PER_FRAME = 300

# =============================================================================
# Dust
# =============================================================================

#: Peak V-band optical depth through a dust lane.
DUST_TAU_V = 1.35
#: Angular half-width of the lane in units of the arm's own phase.
DUST_WIDTH = 0.42
#: Phase offset of the dust lane from the stellar arm, radians. NEGATIVE puts it
#: on the upstream (inner) edge inside corotation, which is where the gas shocks
#: and therefore where the dark thread is in real spirals.
DUST_PHASE_OFFSET = -0.55
#: Extinction is steeper in the blue; these are A_lambda / A_V at roughly the
#: effective wavelengths of the three channels, from a standard R_V = 3.1 curve.
DUST_REDDENING = (0.75, 1.00, 1.42)

# =============================================================================
# Time axis
# =============================================================================

#: Span of the animation and its frame count. 480 Myr is about 2.5 turns at
#: 8 kpc against 1.8 turns of the pattern, so the overtaking is unmistakable.
#:
#: The STEP is set by how far the disc turns between frames, not by the span.
#: On this rotation curve, per frame:
#:
#:      radius     Omega          20 Myr       2 Myr
#:      2 kpc    102 km/s/kpc     120 deg      12 deg
#:      8 kpc     32 km/s/kpc      38 deg     3.8 deg
#:     20 kpc     13 km/s/kpc      15 deg     1.5 deg
#:
#: 20 Myr was not merely jumpy at 2 kpc, it was ALIASED: 120 deg per frame is
#: past the half-turn limit, so the inner disc could read as turning the wrong
#: way — the opposite of what a demo about differential rotation should show.
#: 2 Myr keeps every radius well inside it.
#:
#: The cost is linear in BOTH factors, because the whole disc is duplicated per
#: frame. 241 frames x 400k stars measured 118.6M points, 828 MB on disk, 7.4 min
#: to build and 14.4 GB peak RSS — too heavy for what it bought, so the budget is
#: spent on TIME instead: 241 frames x 100k stars is 29.7M points, 218 MB, 1.3 min
#: and 131k points on screen per frame. `density_scale` is what makes the lighter
#: disc look like the heavy one instead of merely dimmer: each population keeps
#: the additive-light invariant `N * gain * radius^2`. Both knobs stay exposed
#: (`--stars`, `--frames`); neither changes the physics.
#:
#: The count does NOT have to be odd: T is declared DISCRETE (see
#: `time_dimension`), so the slider and the keyboard both snap to `k * step`
#: and every reachable stop is a frame that exists. It is capped so adjacent
#: frames stay outside the viewer's inclusive +/-0.5 discrete membership gate.
#:
#: That is the whole point of the discrete flag here, and it was learned the hard
#: way. While T was declared continuous-and-spatial the viewer gave it a slider
#: with 1000 free positions and opened it at the MIDPOINT of its range, while a
#: point still only matched a slice it sat exactly on. So any stop between two
#: frames matched nothing: measured at 225.12 Myr (drag position 469/1000), all
#: five disc layers, the HII regions and the bulge went to ZERO points and the
#: 33,022 static halo/globular points were the entire scene. An odd frame count
#: rescued only the OPENING frame — the first mouse drag emptied it again.
T_SPAN_MYR = 480.0
T_FRAMES = 241
DISCRETE_MEMBERSHIP_TOLERANCE = 0.5
MAX_FRAME_COUNT = math.ceil(T_SPAN_MYR / DISCRETE_MEMBERSHIP_TOLERANCE)


def time_step(n_frames: int) -> float:
    """The frame interval, and the viewer's snap grid for T."""
    return T_SPAN_MYR / (n_frames - 1)


def frame_times(n_frames: int) -> np.ndarray:
    """Frame times in Myr, built as `k * step` so they sit EXACTLY on the grid.

    Not `linspace`: the viewer snaps a discrete dimension to
    `round(value / step) * step`, so the stored planes use the same `k * step`
    arithmetic. The last product can round just below or above `T_SPAN_MYR`;
    `time_dimension` therefore uses that actual last frame as the range maximum.
    """
    return np.arange(n_frames, dtype=np.float64) * time_step(n_frames)


def time_dimension(n_frames: int) -> Dimension:
    """The T axis: DISCRETE, so every slider stop is a frame that exists.

    `discrete=True` is what makes the slider an `n_frames`-stop track snapping
    to multiples of `step` instead of a 1000-position continuous scrub, and it is
    what puts the opening position on the first frame rather than at the
    midpoint of the range. The range ends on the LAST FRAME (not on
    `T_SPAN_MYR`) so the final stop is reachable for any frame count, whatever
    rounding `(n - 1) * step` picks up.
    """
    times = frame_times(n_frames)
    return Dimension(
        "T",
        unit="Myr",
        range=(0.0, float(times[-1])),
        step=time_step(n_frames),
        display=False,
        discrete=True,
        description=(
            "Time. Stars advance at their own Omega(R); the spiral "
            "pattern advances at Omega_p. Watch stars overtake the "
            "arms inside corotation."
        ),
    )


# =============================================================================
# Scene scale
# =============================================================================

#: Camera framing. Solved at the cinematic preset's own lens, so `fov` is left
#: unset on the CameraConfig (see `demos/_cinematic_camera.py`).
CAMERA_FILL = 0.617
#: A low, near-edge-on inclination shows the disc's thickness, the dust lanes
#: and the bulge all at once; face-on shows the arms best. 32 degrees is the
#: compromise most galaxy photographs are taken at.
CAMERA_INCLINATION_DEG = 32.0
AUTO_ROTATE_SPEED = 0.10


def galaxy_camera_position() -> tuple[float, float, float]:
    """Return the opening camera position solved at the cinematic lens."""
    distance = R_DISC_MAX / np.sin(np.radians(CINEMATIC_FOV_DEG * CAMERA_FILL))
    inclination = np.radians(CAMERA_INCLINATION_DEG)
    return (
        0.0,
        float(distance * np.sin(inclination)),
        float(distance * np.cos(inclination)),
    )


#: Authored per-point gain. Millions of additive points over a disc sum hard, so
#: this is small by construction; it is dialled against a render rather than
#: derived, and moving it means re-checking the frame is not clipping. The gain
#: and every radius below were dialled at `GAIN_REFERENCE_STARS`; `--stars`
#: rescales both through `density_scale`, so a lighter build looks the same
#: rather than dimmer and grainier.
STAR_GAIN = 0.030
GAIN_REFERENCE_STARS = 400_000
GLOBULAR_STARS_PER_CLUSTER = 90


def density_scale(n_disc: int) -> float:
    """Radius and gain multiplier that holds the picture fixed as N changes.

    A sample thinned by a factor f has its mean nearest-neighbour spacing grow
    as `f^(1/3)`, so radii scale that way to keep the fill factor — a disc of
    blobs, not a dust of pinpricks with gaps between them. Summed additive light
    then goes as `N * gain * radius^2`, i.e. as `gain * N^(1/3)`, so the SAME
    factor applied to the gain holds surface brightness too. One number, both
    jobs, and it is 1.0 at the reference count.
    """
    return float(GAIN_REFERENCE_STARS / max(n_disc, 1)) ** (1.0 / 3.0)


def globular_stars_per_cluster(n_disc: int) -> int:
    """Sample each globular at the same relative density as the disc."""
    relative_density = n_disc / GAIN_REFERENCE_STARS
    return max(round(GLOBULAR_STARS_PER_CLUSTER * relative_density), 1)


#: Scene exposure in LOG2 STOPS, on top of the gain above.
EXPOSURE_EV = -0.4


# =============================================================================
# Mass model -> rotation curve -> frequencies
# =============================================================================
def circular_speed(radius: np.ndarray) -> np.ndarray:
    """Circular speed in km/s at cylindrical radius ``radius`` (kpc).

    Bulge, disc and halo circular speeds add in quadrature because each
    component's contribution to the centripetal acceleration is additive. The
    resulting curve is flat past ~5 kpc, which is the whole reason the halo term
    is here: bulge and disc alone fall off Keplerian.
    """
    r = np.maximum(np.asarray(radius, dtype=np.float64), 1e-3)
    v2_bulge = G * M_BULGE * r / (r + A_BULGE) ** 2
    v2_disc = G * M_DISC * r**2 / (r**2 + (A_DISC + B_DISC) ** 2) ** 1.5
    g_c = np.log1p(C_NFW) - C_NFW / (1.0 + C_NFW)
    r200 = R_S_NFW * C_NFW
    x = r / R_S_NFW
    mass_fraction = (np.log1p(x) - x / (1.0 + x)) / g_c
    v2_halo = V200_NFW**2 * (r200 / r) * mass_fraction
    return np.sqrt(v2_bulge + v2_disc + v2_halo)


def frequency_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sampled ``(R, v_c, Omega, kappa)`` for interpolation.

    ``kappa`` — the epicyclic frequency, the rate at which a star oscillates
    radially about its guiding centre — is
    ``kappa^2 = (2 Omega / R) d(R^2 Omega)/dR``. It is differentiated
    numerically off the analytic curve rather than derived by hand, so changing
    the mass model above cannot leave a stale expression behind.
    """
    r = np.linspace(0.02, 40.0, 8000)
    v = circular_speed(r)
    omega = v / r
    d_specific_am = np.gradient(r**2 * omega, r)
    kappa = np.sqrt(np.maximum(2.0 * omega / r * d_specific_am, 1e-6))
    return r, v, omega, kappa


def resonance_radii(
    r: np.ndarray, omega: np.ndarray, kappa: np.ndarray
) -> dict[str, float]:
    """Where ``Omega``, ``Omega -+ kappa/m`` cross the pattern speed."""

    def crossing(curve: np.ndarray) -> float:
        idx = np.where(np.diff(np.sign(curve - OMEGA_P)))[0]
        return float(r[idx[0]]) if len(idx) else float("nan")

    return {
        "ILR": crossing(omega - kappa / M_ARMS),
        "corotation": crossing(omega),
        "OLR": crossing(omega + kappa / M_ARMS),
    }


# =============================================================================
# Colour: temperature -> RGB through the CIE observer
# =============================================================================
def _cie_lobe(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    """Asymmetric ("piecewise") Gaussian lobe used by the colour-matching fit."""
    s = np.where(x < mu, s1, s2)
    return np.exp(-0.5 * ((x - mu) / s) ** 2)


def blackbody_rgb(temperature_k: np.ndarray) -> np.ndarray:
    """Linear-light RGB of a blackbody at ``temperature_k``, peak-normalised.

    Planck spectrum integrated against the CIE 1931 2-degree colour-matching
    functions, then through the sRGB primaries. The colour-matching functions
    use the multi-lobe Gaussian fit of Wyman, Sloan & Shirley (2013), which is
    within about 1% of the tabulated curves and needs no data file.

    LINEAR light, deliberately: Luxar demo colours are linear, and applying an
    sRGB transfer curve here would wash the whole galaxy out.

    Sanity values it reproduces: 3000 K -> deep amber ``(1, .48, .16)``,
    5800 K -> near white ``(1, .88, .83)``, 12000 K -> blue ``(.53, .63, 1)``.
    """
    lam = np.arange(360.0, 831.0, 5.0)
    x_bar = (
        1.056 * _cie_lobe(lam, 599.8, 37.9, 31.0)
        + 0.362 * _cie_lobe(lam, 442.0, 16.0, 26.7)
        - 0.065 * _cie_lobe(lam, 501.1, 20.4, 26.2)
    )
    y_bar = 0.821 * _cie_lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _cie_lobe(
        lam, 530.9, 16.3, 31.1
    )
    z_bar = 1.217 * _cie_lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _cie_lobe(
        lam, 459.0, 26.0, 13.8
    )

    lam_m = lam[None, :] * 1e-9
    h, c_light, k_b = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    t = np.asarray(temperature_k, dtype=np.float64)[:, None]
    radiance = (2 * h * c_light**2 / lam_m**5) / np.expm1(
        h * c_light / (lam_m * k_b * t)
    )

    xyz = np.stack(
        [
            (radiance * x_bar).sum(1),
            (radiance * y_bar).sum(1),
            (radiance * z_bar).sum(1),
        ],
        axis=1,
    )
    xyz_to_srgb = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    rgb = np.clip(xyz @ xyz_to_srgb.T, 0.0, None)
    return (rgb / np.maximum(rgb.max(axis=1, keepdims=True), 1e-12)).astype(np.float32)


#: Age (Gyr) -> effective temperature (K) of the integrated stellar population.
#: A simple-stellar-population reddens as its main-sequence turnoff moves down;
#: these stops track that, from an O/B-dominated 15 kK at birth to a K-giant
#: 4.2 kK at 10 Gyr.
_AGE_GYR_STOPS = np.array([0.0, 0.05, 0.2, 0.7, 2.0, 5.0, 10.0])
_TEFF_STOPS = np.array([15000.0, 11000.0, 8500.0, 7000.0, 6000.0, 5000.0, 4200.0])


def age_to_temperature(age_gyr: np.ndarray) -> np.ndarray:
    """Effective temperature of a population of the given age."""
    return np.interp(np.asarray(age_gyr, dtype=np.float64), _AGE_GYR_STOPS, _TEFF_STOPS)


# =============================================================================
# Sampling the components
# =============================================================================
def sample_exponential_disc(n: int, rng: np.random.Generator) -> np.ndarray:
    """Guiding radii drawn from an exponential surface density.

    The disc's mass per unit radius is ``2 pi R Sigma(R)``, so the radial
    distribution to draw from is ``R exp(-R/R_d)`` — a Gamma(2) — not the
    exponential itself. Inverse-transform sampling on a fine CDF grid keeps it
    exact and truncated to ``[R_DISC_MIN, R_DISC_MAX]``.
    """
    grid = np.linspace(R_DISC_MIN, R_DISC_MAX, 4096)
    pdf = grid * np.exp(-grid / R_DISC_SCALE)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, grid)


def spiral_phase(
    radius: np.ndarray, azimuth: np.ndarray, time_gyr: float
) -> np.ndarray:
    """Phase of the m-armed logarithmic spiral wave at ``(R, phi, t)``.

    Zero on an arm crest. The ``ln R`` term is what makes the arm logarithmic:
    a constant pitch angle means the arm's azimuth advances by a fixed amount
    per e-folding of radius. The ``Omega_p * t`` term is the ONLY time
    dependence — the pattern turns rigidly, which is the whole point.
    """
    k_log = M_ARMS / np.tan(np.radians(PITCH_DEG))
    return M_ARMS * (
        azimuth - OMEGA_P * KMS_TO_KPC_PER_GYR * time_gyr
    ) + k_log * np.log(np.maximum(radius, 1e-3) / R_DISC_SCALE)


def forced_response(
    radius: np.ndarray, omega: np.ndarray, kappa: np.ndarray
) -> np.ndarray:
    """Amplitude of a star's forced radial displacement by the spiral, in kpc.

    A star in a weak spiral potential is a driven harmonic oscillator: it
    oscillates freely at ``kappa`` and is driven at the frequency it meets the
    arms, ``m (Omega - Omega_p)``. The response therefore carries the classic
    denominator

        Delta = kappa^2 - m^2 (Omega - Omega_p)^2

    which vanishes at the Lindblad resonances (where the driving matches the
    natural frequency) and is largest between them. That is why grand-design
    arms are strong across the mid-disc and fade at both ends, and it is a
    prediction of the theory rather than a taper anyone painted on.

    Near the resonances linear theory is simply not valid, so ``Delta`` is
    floored at ``RESONANCE_SOFTENING * kappa^2`` — the response saturates
    instead of diverging.
    """
    drive = M_ARMS * (omega - OMEGA_P)
    delta = kappa**2 - drive**2
    floor = RESONANCE_SOFTENING * kappa**2
    delta = np.where(np.abs(delta) < floor, np.sign(delta) * floor, delta)
    delta = np.where(delta == 0.0, floor, delta)
    # Forcing tapers at the disc edge, where there is no longer enough mass to
    # support a wave.
    taper = np.exp(-((radius / (0.85 * R_DISC_MAX)) ** 4))
    return SPIRAL_FORCING * taper / delta


def dust_attenuation(phase: np.ndarray) -> np.ndarray:
    """Per-channel transmission through the dust lane at spiral ``phase``.

    The lane is a Gaussian trough in phase, offset to the upstream edge of the
    arm. Extinction is applied per channel with a standard ``R_V = 3.1``
    reddening slope, so an obscured star goes red as well as dim — which is what
    distinguishes a dust lane from a gap.
    """
    wrapped = np.mod(phase - DUST_PHASE_OFFSET + np.pi, 2 * np.pi) - np.pi
    tau_v = DUST_TAU_V * np.exp(-0.5 * (wrapped / DUST_WIDTH) ** 2)
    slope = np.asarray(DUST_REDDENING, dtype=np.float32)[None, :]
    return np.exp(-tau_v[:, None] * slope).astype(np.float32)


# =============================================================================
# The disc population
# =============================================================================
def build_disc(n_stars: int, rng: np.random.Generator) -> dict:
    """Guiding centres, ages and epicycle parameters for the disc.

    Returns the time-INDEPENDENT state. Positions at a given time come from
    :func:`disc_positions`, which is called once per frame.
    """
    r_tab, _, omega_tab, kappa_tab = frequency_tables()

    r_g = sample_exponential_disc(n_stars, rng)
    omega = np.interp(r_g, r_tab, omega_tab) * KMS_TO_KPC_PER_GYR  # rad/Gyr
    kappa = np.interp(r_g, r_tab, kappa_tab) * KMS_TO_KPC_PER_GYR

    # Continuous star formation over the life of the disc, biased slightly to
    # early times the way a real declining SFH is.
    age = DISC_AGE_MAX * rng.random(n_stars) ** 0.75

    # Age-velocity dispersion relation -> epicyclic amplitude. This is the line
    # that makes the arms a young-star phenomenon.
    sigma_r = SIGMA_R_1GYR * np.maximum(age, 0.02) ** SIGMA_HEATING_EXPONENT
    epi_amplitude = sigma_r * KMS_TO_KPC_PER_GYR / kappa
    z_amplitude = SIGMA_Z_RATIO * sigma_r * KMS_TO_KPC_PER_GYR / (2.4 * omega)

    return {
        "r_g": r_g,
        "phi_0": rng.uniform(0.0, 2 * np.pi, n_stars),
        "omega": omega,
        "kappa": kappa,
        # Vertical oscillation frequency. For a thin disc nu is a couple of
        # times Omega; 2.4 is a good fit across this rotation curve.
        "nu": 2.4 * omega,
        "epi_amplitude": epi_amplitude,
        "epi_phase": rng.uniform(0.0, 2 * np.pi, n_stars),
        "z_amplitude": z_amplitude,
        "z_phase": rng.uniform(0.0, 2 * np.pi, n_stars),
        "age": age,
        "forced": forced_response(
            r_g,
            np.interp(r_g, r_tab, omega_tab),
            np.interp(r_g, r_tab, kappa_tab),
        ),
    }


def disc_positions(disc: dict, time_gyr: float) -> tuple[np.ndarray, np.ndarray]:
    """Disc star positions at ``time_gyr``, and their spiral phase there.

    Three superposed motions, in order of size:

    1. the guiding centre going round at the star's OWN ``Omega(R_g)``;
    2. the free epicycle, radial at ``kappa`` and vertical at ``nu``, with the
       tangential excursion exactly ``2 Omega / kappa`` times the radial one —
       the epicycle's true axis ratio, not a fudge;
    3. the forced response to the spiral, in phase with the wave.
    """
    phi_guide = disc["phi_0"] + disc["omega"] * time_gyr
    epi = disc["kappa"] * time_gyr + disc["epi_phase"]

    r_free = -disc["epi_amplitude"] * np.cos(epi)
    phi_free = (
        2.0
        * disc["omega"]
        / disc["kappa"]
        * disc["epi_amplitude"]
        / np.maximum(disc["r_g"], 1e-3)
        * np.sin(epi)
    )

    phase = spiral_phase(disc["r_g"], phi_guide, time_gyr)
    r = disc["r_g"] + r_free + disc["forced"] * np.cos(phase)
    phi = phi_guide + phi_free
    z = disc["z_amplitude"] * np.cos(disc["nu"] * time_gyr + disc["z_phase"])

    r = np.maximum(r, 0.05)
    xyz = np.column_stack([r * np.cos(phi), r * np.sin(phi), z]).astype(np.float32)
    return xyz, spiral_phase(r, phi, time_gyr)


# =============================================================================
# Bulge, halo, globular clusters
# =============================================================================
def build_bulge(n: int, rng: np.random.Generator) -> dict:
    """Hernquist bulge: old, amber, pressure-supported, slowly rotating.

    Radii are drawn by inverting the Hernquist cumulative mass profile
    ``M(<r)/M = r^2/(r+a)^2``, which has the closed form
    ``r = a sqrt(u)/(1 - sqrt(u))``. Directions are isotropic — sampling
    ``cos(theta)`` uniformly rather than ``theta``, or the poles get crowded.
    """
    # The Hernquist profile has an unbounded tail: r = a sqrt(u)/(1-sqrt(u))
    # diverges as u -> 1, and at u = 0.985 it already reaches 93 kpc, which
    # would put "bulge" stars out past the halo and blow the scene bounds up by
    # a factor of four. Truncate in RADIUS and convert that to the matching u
    # bound, u_max = (r_t/(r_t+a))^2, so the profile inside the truncation is
    # still exactly Hernquist.
    r_truncate = 3.5
    u_max = (r_truncate / (r_truncate + A_BULGE)) ** 2
    u = rng.random(n) * u_max
    r = A_BULGE * np.sqrt(u) / (1.0 - np.sqrt(u))
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi0 = rng.uniform(0.0, 2 * np.pi, n)
    # Slight flattening toward the disc plane, as real bulges have.
    return {
        "r_cyl": r * sin_theta,
        "z": (r * cos_theta * 0.75).astype(np.float32),
        "phi_0": phi0,
        # Slow, nearly solid-body figure rotation.
        "omega": np.full(n, 42.0 * KMS_TO_KPC_PER_GYR),
        "age": rng.uniform(7.5, 12.5, n),
    }


def build_halo(
    n: int, n_globulars: int, per_cluster: int, rng: np.random.Generator
) -> dict:
    """Stellar halo plus globular clusters, both old and metal-poor.

    The field halo follows ``rho ~ r^-3.5``, the observed slope; sampling it
    means inverting that power law, which is why the radii come out of a simple
    ``u^(1/(3-alpha))`` rather than a CDF grid. Globulars are Plummer knots
    scattered on the same envelope.
    """
    alpha = 3.5
    r_min, r_max = 2.0, 34.0
    exponent = 3.0 - alpha
    u = rng.random(n)
    r = (r_min**exponent + u * (r_max**exponent - r_min**exponent)) ** (1.0 / exponent)
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi0 = rng.uniform(0.0, 2 * np.pi, n)
    field = np.column_stack(
        [r * sin_theta * np.cos(phi0), r * sin_theta * np.sin(phi0), r * cos_theta]
    ).astype(np.float32)

    # Globular clusters: a Plummer sphere each, radius ~ a few parsecs scaled up
    # to stay visible at galaxy scale.
    u_c = rng.random(n_globulars)
    r_c = (r_min**exponent + u_c * (r_max**exponent - r_min**exponent)) ** (
        1.0 / exponent
    )
    cos_c = rng.uniform(-1.0, 1.0, n_globulars)
    sin_c = np.sqrt(1.0 - cos_c**2)
    phi_c = rng.uniform(0.0, 2 * np.pi, n_globulars)
    centres = np.column_stack(
        [r_c * sin_c * np.cos(phi_c), r_c * sin_c * np.sin(phi_c), r_c * cos_c]
    )
    plummer_a = 0.10
    offsets = rng.normal(0.0, plummer_a, (n_globulars * per_cluster, 3))
    clusters = (np.repeat(centres, per_cluster, axis=0) + offsets).astype(np.float32)

    return {"field": field, "clusters": clusters}


# =============================================================================
# Scene assembly
# =============================================================================
def age_bin_index(age_gyr: np.ndarray) -> np.ndarray:
    """Index of the age bin each star belongs to."""
    return np.clip(
        np.searchsorted(np.asarray(AGE_BIN_EDGES[1:-1]), age_gyr, side="right"),
        0,
        len(AGE_BIN_LABELS) - 1,
    ).astype(np.int64)


def generate_galaxy(output_path: Path, n_disc: int, n_frames: int) -> int:
    """Simulate the galaxy and write it as a 4D Luxar scene."""
    rng = np.random.default_rng(20260823)

    r_tab, _, omega_tab, kappa_tab = frequency_tables()
    res = resonance_radii(r_tab, omega_tab, kappa_tab)
    with asection("Mass model"):
        aprint(f"v_c(2 kpc)  = {circular_speed(2.0):7.1f} km/s")
        aprint(f"v_c(8 kpc)  = {circular_speed(8.0):7.1f} km/s")
        aprint(f"v_c(20 kpc) = {circular_speed(20.0):7.1f} km/s   (flat)")
        aprint(f"pattern speed  Omega_p = {OMEGA_P:.1f} km/s/kpc")
        aprint(f"  inner Lindblad resonance : {res['ILR']:6.2f} kpc")
        aprint(f"  corotation               : {res['corotation']:6.2f} kpc")
        aprint(f"  outer Lindblad resonance : {res['OLR']:6.2f} kpc")

    n_bulge = max(n_disc // 5, 1)
    n_halo = max(n_disc // 18, 1)
    n_globulars = 120
    per_cluster = globular_stars_per_cluster(n_disc)

    # Radii and gain both track the sample density (see `density_scale`).
    scale = density_scale(n_disc)
    gain = STAR_GAIN * scale
    if abs(scale - 1.0) > 1e-9:
        aprint(
            f"density scale {scale:.3f}x on radii and gain (vs {GAIN_REFERENCE_STARS:,} stars)"
        )

    with asection("Building populations"):
        disc = build_disc(n_disc, rng)
        bulge = build_bulge(n_bulge, rng)
        halo = build_halo(n_halo, n_globulars, per_cluster, rng)
        aprint(f"disc {n_disc:,} · bulge {n_bulge:,} · halo {n_halo:,} field")
        aprint(f"globular clusters: {n_globulars} ({len(halo['clusters']):,} stars)")

    # Static colours (age -> temperature -> RGB). Computed once; the dust
    # attenuation below is the only per-frame colour work.
    disc_rgb = blackbody_rgb(age_to_temperature(disc["age"]))
    bulge_rgb = blackbody_rgb(age_to_temperature(bulge["age"]))
    halo_rgb = blackbody_rgb(np.full(len(halo["field"]), 4600.0))
    cluster_rgb = blackbody_rgb(np.full(len(halo["clusters"]), 4900.0))

    # Young stars are intrinsically far more luminous than old ones; a rough
    # main-sequence scaling keeps the arms bright without touching their hue.
    disc_rgb = disc_rgb * (0.30 + 3.2 * np.exp(-disc["age"] / 0.30))[:, None].astype(
        np.float32
    )
    # Radii follow the same argument: the young are the giants.
    disc_radii_star = (scale * (0.050 + 0.11 * np.exp(-disc["age"] / 0.25))).astype(
        np.float32
    )

    bins = age_bin_index(disc["age"])
    hii_cut = np.quantile(disc["age"], 0.030)
    hii_mask = disc["age"] < hii_cut
    n_hii = int(hii_mask.sum())
    # H-alpha with a trace of [OIII]; linear light, HDR so they glow additively.
    hii_rgb = np.tile(
        np.array([1.9, 0.36, 0.90], dtype=np.float32), (n_hii, 1)
    ) * rng.uniform(0.55, 1.45, (n_hii, 1)).astype(np.float32)

    frames_myr = frame_times(n_frames)

    # Per-frame accumulators, one list per age-bin layer plus HII and bulge.
    disc_xyz: list[list[np.ndarray]] = [[] for _ in AGE_BIN_LABELS]
    disc_col: list[list[np.ndarray]] = [[] for _ in AGE_BIN_LABELS]
    disc_t: list[list[np.ndarray]] = [[] for _ in AGE_BIN_LABELS]
    hii_xyz, hii_col, hii_t = [], [], []
    bulge_xyz, bulge_t = [], []

    with asection(f"Integrating {n_frames} frames over {T_SPAN_MYR:.0f} Myr"):
        for frame, t_myr in enumerate(frames_myr):
            t_gyr = t_myr / 1000.0

            xyz, phase = disc_positions(disc, t_gyr)
            transmission = dust_attenuation(phase)
            attenuated = disc_rgb * transmission

            for b in range(len(AGE_BIN_LABELS)):
                sel = bins == b
                disc_xyz[b].append(xyz[sel])
                disc_col[b].append(attenuated[sel])
                disc_t[b].append(np.full(int(sel.sum()), t_myr, dtype=np.float32))

            hii_xyz.append(xyz[hii_mask])
            # HII gas sits in the same lane as the dust that made it, so it is
            # obscured too — just less, being on the near side of the shock.
            hii_col.append(hii_rgb * (0.45 + 0.55 * transmission[hii_mask]))
            hii_t.append(np.full(n_hii, t_myr, dtype=np.float32))

            phi_b = bulge["phi_0"] + bulge["omega"] * t_gyr
            bulge_xyz.append(
                np.column_stack(
                    [
                        bulge["r_cyl"] * np.cos(phi_b),
                        bulge["r_cyl"] * np.sin(phi_b),
                        bulge["z"],
                    ]
                ).astype(np.float32)
            )
            bulge_t.append(np.full(n_bulge, t_myr, dtype=np.float32))

            if (frame + 1) % 5 == 0 or frame == n_frames - 1:
                aprint(f"  frame {frame + 1}/{n_frames}  (t = {t_myr:5.0f} Myr)")

    def stack4(xyz_list, t_list) -> np.ndarray:
        """(N, 4) columns: X, Y, Z, T.

        Filled into ONE preallocated array rather than vstack-then-hstack. At
        241 frames a single age bin is tens of millions of rows, and the two
        intermediate copies of the naive version were the process's peak.
        """
        total_rows = sum(len(x) for x in xyz_list)
        out = np.empty((total_rows, 4), dtype=np.float32)
        row = 0
        for xyz, t in zip(xyz_list, t_list):
            end = row + len(xyz)
            out[row:end, :3] = xyz
            out[row:end, 3] = t
            row = end
        return out

    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("X", unit="kpc", range=(-36, 36), display=True),
                Dimension("Y", unit="kpc", range=(-36, 36), display=True),
                Dimension("Z", unit="kpc", range=(-36, 36), display=True),
                time_dimension(n_frames),
            ]
        )

        camera_pos = galaxy_camera_position()
        distance = float(np.linalg.norm(camera_pos))
        aprint(
            f"Camera distance {distance:.1f} kpc at {CAMERA_INCLINATION_DEG:.0f} deg"
        )

        total = 0
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    exposure=EXPOSURE_EV,
                    auto_rotate=True,
                    auto_rotate_speed=AUTO_ROTATE_SPEED,
                    camera=CameraConfig(
                        position=camera_pos,
                        target=(0.0, 0.0, 0.0),
                    ),
                ),
            )
            scene.attrs["title"] = "Galaxy Simulation — a density-wave spiral"
            scene.attrs["description"] = scene_description(res, n_frames)

            with asection("Adding layers"):
                # One layer per age bin. Every bin is visible by default, so the
                # opening frame is the whole galaxy; isolating a bin in the
                # Layers panel is what shows the arms sharpening young and
                # dissolving old.
                for b, label in enumerate(AGE_BIN_LABELS):
                    pos = stack4(disc_xyz[b], disc_t[b])
                    # Every layer's per-frame lists are released as soon as that
                    # layer is on disk — at 241 frames the accumulators, not the
                    # integration, are what the process's memory ceiling is made
                    # of, so they must not all be held to the end.
                    disc_xyz[b].clear()
                    disc_t[b].clear()
                    if len(pos) == 0:
                        disc_col[b].clear()
                        del pos
                        continue
                    col = np.concatenate(disc_col[b], axis=0).astype(
                        np.float32, copy=False
                    )
                    disc_col[b].clear()
                    rad = np.tile(disc_radii_star[bins == b], n_frames)
                    # T is PLAYED here, so rung 0 has to satisfy the absolute
                    # per-frame arm of the ladder gate as well as its share arm.
                    # `stream_ladder`'s share rung (n/8) covers the fat bins and
                    # starves the thin ones — see
                    # DISC_FIRST_RUNG_ROWS_PER_FRAME for the measured 144-row
                    # sparsest frame it leaves on `50-300 Myr`. So take whichever
                    # rung 0 is larger and re-resolve the SAME capped schedule
                    # around it; on the three fat bins the max picks the policy's
                    # own chunk and the ladder is byte-for-byte unchanged
                    # (verified: 3.00M / 10.74M / 10.10M rows all keep
                    # [n/8, n/4, n/2, n]). Reading rung 0 back off the spec
                    # rather than recomputing it is what keeps the budget-vs-share
                    # choice in one place; it is a cut LIST because these are
                    # Points — the Lines spelling of `counts` is a string.
                    stops = hidden_axis_stops(pos, dims.non_displayed)
                    ladder = stream_ladder(len(pos), slices=stops)
                    ladder["counts"] = capped_stream_cuts(
                        len(pos),
                        max(
                            ladder["counts"][0],
                            stops * DISC_FIRST_RUNG_ROWS_PER_FRAME,
                        ),
                        DEFAULT_MAX_ADDITIVE_COMMIT * stops,
                    )
                    scene.add_points(
                        f"Disc {label}",
                        positions=pos,
                        colors=col,
                        radii=rad,
                        sharpness=0.42,
                        opacity=1.0,
                        blending_mode="additive",
                        intensity=gain,
                        additive_lod=ladder,
                        layer=True,
                    )
                    total += len(pos)
                    aprint(f"  disc {label:<12} {len(pos):>10,} points")
                    del pos, col, rad

                hii_pos = stack4(hii_xyz, hii_t)
                hii_colors = np.concatenate(hii_col, axis=0).astype(
                    np.float32, copy=False
                )
                hii_xyz.clear()
                hii_col.clear()
                hii_t.clear()
                scene.add_points(
                    "HII regions",
                    positions=hii_pos,
                    colors=hii_colors,
                    radii=0.13 * scale,
                    sharpness=0.32,
                    opacity=1.0,
                    blending_mode="additive",
                    intensity=gain,
                    # No per-frame floor needed here or on the bulge below: both
                    # are single fat nodes, so the n/8 share rung already clears
                    # the 250-row absolute arm with room — measured p05 frames of
                    # 343 (HII, 90,375 of 723,000) and 2,423 (bulge, 602,500 of
                    # 4,820,000). Only the age-bin layers split thin enough to
                    # need DISC_FIRST_RUNG_ROWS_PER_FRAME.
                    additive_lod=stream_ladder(
                        len(hii_pos),
                        slices=hidden_axis_stops(hii_pos, dims.non_displayed),
                    ),
                    layer=True,
                )
                total += len(hii_pos)
                aprint(f"  HII regions      {len(hii_pos):>10,} points")
                del hii_pos, hii_colors

                bulge_pos = stack4(bulge_xyz, bulge_t)
                bulge_xyz.clear()
                bulge_t.clear()
                scene.add_points(
                    "Bulge",
                    positions=bulge_pos,
                    colors=np.tile(bulge_rgb, (n_frames, 1)).astype(np.float32),
                    radii=0.038 * scale,
                    sharpness=0.42,
                    opacity=1.0,
                    blending_mode="additive",
                    intensity=gain,
                    additive_lod=stream_ladder(
                        len(bulge_pos),
                        slices=hidden_axis_stops(bulge_pos, dims.non_displayed),
                    ),
                    layer=True,
                )
                total += len(bulge_pos)
                aprint(f"  bulge            {len(bulge_pos):>10,} points")
                del bulge_pos

                # Halo and globulars are pressure-supported on orbits far longer
                # than the animation, so they are static: `extend_to_all` pins
                # them to every T instead of duplicating them per frame.
                for name, pos3, col3, rad in (
                    ("Stellar halo", halo["field"], halo_rgb * 0.5, 0.05 * scale),
                    (
                        "Globular clusters",
                        halo["clusters"],
                        cluster_rgb * 1.1,
                        0.045 * scale,
                    ),
                ):
                    scene.add_points(
                        name,
                        positions=pos3,
                        colors=col3.astype(np.float32),
                        radii=rad,
                        sharpness=0.35,
                        opacity=1.0,
                        blending_mode="additive",
                        intensity=gain,
                        dim_order=["X", "Y", "Z"],
                        fill={"T": 0.0},
                        extend_to_all=["T"],
                        layer=True,
                    )
                    total += len(pos3)
                    aprint(f"  {name:<16} {len(pos3):>10,} points")

            add_overlays(scene, res)

    aprint(f"✓ Written to {output_path}  ({total:,} points)")
    return total


def scene_description(res: dict, n_frames: int) -> str:
    """The long-form scene description shown in the viewer's info panel."""
    step = time_step(n_frames)
    span = T_SPAN_MYR
    return f"""
Galaxy Simulation
=================

A spiral galaxy integrated from its own mass model rather than drawn.

Mass model: Hernquist bulge + Miyamoto-Nagai disc + NFW halo.
  v_c(2 kpc)  = {circular_speed(2.0):.0f} km/s
  v_c(8 kpc)  = {circular_speed(8.0):.0f} km/s
  v_c(20 kpc) = {circular_speed(20.0):.0f} km/s   (flat — this is the dark halo)

Spiral: an m={M_ARMS} density wave at Omega_p = {OMEGA_P:.0f} km/s/kpc, pitch
angle {PITCH_DEG:.0f} deg. Resonances on this rotation curve:
  inner Lindblad  {res["ILR"]:.1f} kpc
  COROTATION      {res["corotation"]:.1f} kpc
  outer Lindblad  {res["OLR"]:.1f} kpc

Every star's guiding centre turns at its OWN Omega(R). The pattern turns at
Omega_p. So inside corotation stars OVERTAKE the arms and outside it they fall
behind — step T and watch. A rigidly rotated picture cannot show this, and a
material arm would wind up within a couple of turns.

Stars are on forced epicycles: free radial oscillation at kappa, vertical at
nu, tangential excursion exactly 2*Omega/kappa times the radial one, plus the
spiral's forced response with its resonant denominator
kappa^2 - m^2 (Omega - Omega_p)^2.

Ages drive everything else. sigma_R grows as age^{SIGMA_HEATING_EXPONENT}, so
old stars have big epicycles and leave the arms — which is WHY spiral arms are
a young-star feature. Colour is the blackbody colour of the population's
effective temperature, integrated through the CIE 1931 observer.

Dust lanes sit on the upstream edge of each arm and redden as well as dim.
HII regions mark where the shock is making stars right now.

Navigation
----------
  Press 1 then [ / ]  — step time ({n_frames} frames of {step:.3g} Myr over {span:.0f} Myr)
  Press L             — Layers: five age bins, HII, bulge, halo, globulars.
                        Solo the youngest bin for knife-edge arms; solo the
                        oldest for a smooth, thick, featureless disc.
"""


def add_overlays(scene, res: dict) -> None:
    """Title, caption and the explanatory panel."""
    scene.add_text(
        "Galaxy Simulation",
        position=(0.02, 0.02),
        font_size=0.048,
        anchor="top-left",
        color="rgba(255,255,255,0.6)",
        blend_mode="difference",
    )
    add_demo_caption(
        scene,
        f"m={M_ARMS} density wave • pattern {OMEGA_P:.0f} km/s/kpc • "
        f"corotation {res['corotation']:.1f} kpc",
        DEMO_META.get("citation"),
    )
    scene.add_text(
        "Spiral arms are a DENSITY WAVE, not a set of\n"
        "stars. Each star orbits at its own \u03a9(R); the\n"
        f"pattern turns at \u03a9p = {OMEGA_P:.0f} km/s/kpc. Inside\n"
        f"corotation ({res['corotation']:.1f} kpc) stars OVERTAKE the\n"
        "arms; outside it they fall behind.\n"
        "\n"
        f"Press 1 then [ / ] to step {T_SPAN_MYR:.0f} Myr, or press play.\n"
        "Press L for layers: the five age bins show\n"
        "the arms sharpen young and dissolve old.",
        position=(0.02, 0.10),
        font_size=0.019,
        font="mono",
        color="white",
        width=0.38,
        line_height=1.45,
        background="rgba(0,0,0,0.55)",
        padding=0.012,
    )


# =============================================================================
# Main
# =============================================================================
def _int_arg(flag: str, default: int) -> int:
    for arg in sys.argv[1:]:
        if arg.startswith(f"--{flag}="):
            return int(arg.split("=", 1)[1])
    return default


def validate_frame_count(n_frames: int) -> int:
    """Reject frame grids that cannot be selected one frame at a time.

    T is discrete, so the count need not be odd, but adjacent frames must remain
    outside the viewer's inclusive +/-0.5 discrete membership tolerance.
    """
    if n_frames < 3:
        raise ValueError("--frames must be at least 3 to define a time step")
    if n_frames > MAX_FRAME_COUNT:
        raise ValueError(
            f"--frames must be at most {MAX_FRAME_COUNT} so adjacent frames stay "
            "outside the viewer's discrete tolerance"
        )
    return n_frames


def validate_star_count(n_stars: int) -> int:
    """Reject populations too small to produce an HII-region sample."""
    if n_stars < 2:
        raise ValueError(
            "--stars must be at least 2 to include a star below the 3rd age percentile"
        )
    return n_stars


def main() -> None:
    """Simulate the galaxy and open it in the viewer."""
    n_disc = validate_star_count(_int_arg("stars", 100_000))
    n_frames = validate_frame_count(_int_arg("frames", T_FRAMES))

    aprint("=" * 70)
    aprint("GALAXY SIMULATION — density-wave spiral")
    aprint("=" * 70)
    aprint(f"Disc stars: {n_disc:,}   Frames: {n_frames}   Span: {T_SPAN_MYR:.0f} Myr")
    aprint("")

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "galaxy_simulation.luxar.zarr"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_galaxy(output_path, n_disc, n_frames)
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "galaxy_simulation.luxar.zarr"
        generate_galaxy(output_path, n_disc, n_frames)
        launch_viewer(output_path)


if __name__ == "__main__":
    main()
