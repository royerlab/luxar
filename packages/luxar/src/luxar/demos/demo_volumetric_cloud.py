#!/usr/bin/env python3
"""Self-Contained Demo: An Evolving Volumetric Cloud (3D + time)

A convective cumulus lived through its whole cycle on a hidden time axis: a
low fragment at the condensation level, a cauliflower turret billowing upward,
a mature top leaning downwind, then the whole body pulling in and sinking back
as the thermals feeding it die.

This demo demonstrates:
- A 3D + time (4D) Points scene: x/y/z displayed, ``time`` a hidden discrete axis
- Lagrangian parcels advected by an analytic divergence-free velocity field
- 4D fractal noise (three spatial axes + time) in *material* coordinates
- A cauliflower silhouette built as the union of rising thermal bubbles
- Emission-absorption (``volumetric``) compositing, so the cloud is a body
- Baked sun, sky and ground light from optical depths through the cloud
- Complete workflow: generate -> serve -> view -> cleanup

The demo is completely self-contained - all generation code is in this file,
including the noise, the flow and the shading.

Mathematical Background:
    Two ingredients, kept deliberately separate.

    *Where the air goes* is an analytic velocity field built to be exactly
    divergence-free, so it stirs the parcels without ever compressing them:
    an axisymmetric convection roll written through a Stokes stream function
    (rises in the core, spreads at the top, sinks outside, converges at the
    base), plus a wind shear that leans the cloud downwind with height, plus a
    slow swirl about the vertical. Parcels are pushed through it with midpoint
    steps, so a point is a parcel of air and keeps its identity frame to frame.

    *Where the water is* is a condensation field evaluated at each parcel:
    multi-octave fractal noise sampled in MATERIAL coordinates (the parcel's
    label, i.e. where it started), multiplied by a thermodynamic envelope
    evaluated at the parcel's current WORLD position. Material coordinates are
    what make the texture ride with the flow — it stretches and folds the way
    a real cloud's structure does, instead of boiling in place. The envelope is
    what makes it a cumulus rather than a blob: liquid water only exists above
    the lifting condensation level, which is why cumulus have famously flat
    bases.

    The noise's time axis is handled by precomputing a handful of static 3D
    fields at fixed material coordinates and quintic-interpolating between them
    — this is exactly 4D value noise, evaluated for the price of a lerp, and
    the same trick as animating a cloud field by evolving a small state and
    redrawing attenuated particles [Dobashi et al. 2000]. Its
    temporal frequency grows only like 2^(2k/3) per octave rather than 2^k, in
    the spirit of Kolmogorov's eddy-turnover scaling: small eddies do turn over
    faster, but not as fast as a naive 4D noise would flicker them.

    Points are emissive, so lighting has to be baked in. Parcel condensate is
    splatted onto a coarse grid, cumulatively summed toward the light to get an
    optical depth, and sampled back per parcel — once along the sun's own ray
    and once straight up for skylight. The sun is deliberately off to one side:
    a light directly overhead illuminates every surface by its depth alone, so
    two lobes at the same altitude are lit identically however they face, and
    the relief that makes a cumulus legible disappears.

    That baked light is only coherent in a renderer that OCCLUDES, and getting
    this wrong produced the worst artefact this demo has had. A cumulus is
    optically thick — you see its surface. ``additive`` blending models the
    opposite regime, an optically thin emissive medium that ignores depth
    entirely, so the darkened interior was not hidden behind the lit shell but
    fully visible through it: the cloud rendered as a glowing archway with a
    hole in the middle, and from overhead as a ring. The node is therefore
    ``volumetric`` — emission composited against absorption, back to front.

References:
    Nothing here is invented. Each idea below is the standard one for its job,
    and is cited at the function that uses it.

    Procedural noise
        Perlin, K. (1985). "An Image Synthesizer." SIGGRAPH '85. The lattice
        noise this file's ``simple_noise_3d`` is a value-noise variant of.
        Perlin, K. (2002). "Improving Noise." ACM TOG 21(3). Source of the
        quintic fade ``6t^5 - 15t^4 + 10t^3`` used on all four axes, chosen
        over the older cubic because its second derivative also vanishes at
        the lattice, so no crease shows where cells meet.
        Ebert, D. S., Musgrave, F. K., Peachey, D., Perlin, K. & Worley, S.
        (2003). "Texturing & Modeling: A Procedural Approach", 3rd ed. fBm,
        persistence and lacunarity; the canonical procedural-cloud treatment.
        Collet, Y. "xxHash." The PRIME32 constants and the shift-multiply
        avalanche in ``hash_coords`` are xxHash32's (the same shape as
        MurmurHash3's ``fmix32``, with different constants). 0x9E3779B1 is
        2^32 divided by the golden ratio, the usual multiplicative-hash
        choice.

    Fluid flow
        Batchelor, G. K. (1967). "An Introduction to Fluid Dynamics." CUP.
        The Stokes stream function for axisymmetric incompressible flow that
        ``velocity`` is built from (pp. 78-79), and the Lagrangian-vs-Eulerian
        specification that the material/world split here rests on.
        Bridson, R., Houriham, J. & Nordenstam, M. (2007). "Curl-Noise for
        Procedural Fluid Flow." ACM TOG 26(3). Why a procedural flow field
        should be divergence-free rather than merely plausible.
        Frisch, U. (1995). "Turbulence: The Legacy of A. N. Kolmogorov." CUP.
        The eddy-turnover scaling tau(l) ~ l^(2/3) behind this file's choice
        of temporal frequency per octave.

    Cumulus convection
        Scorer, R. S. & Ludlam, F. H. (1953). "Bubble theory of penetrative
        convection." Quart. J. Roy. Meteor. Soc. 79. The thermal-bubble
        picture of cumulus growth that ``build_bubbles`` implements.
        Stommel, H. (1947). "Entrainment of air into a cumulus cloud."
        J. Meteorology 4. Lateral entrainment of dry air, which is what eats
        the flanks and ends the life cycle here.
        Rogers, R. R. & Yau, M. K. (1989). "A Short Course in Cloud Physics",
        3rd ed. The lifting condensation level, and hence the flat base.

    Implicit surfaces
        Ricci, A. (1973). "A Constructive Geometry for Computer Graphics."
        The Computer Journal 16(2). The p-norm soft union used to merge the
        thermals without the crease a hard max would leave.
        Blinn, J. F. (1982a). "A Generalization of Algebraic Surface
        Drawing." ACM TOG 1(3). Blobby models — summed radial fields read as
        a solid.

    Cloud rendering
        Blinn, J. F. (1982b). "Light Reflection Functions for Simulation of
        Clouds and Dusty Surfaces." SIGGRAPH '82. Single-scattering albedo
        against optical depth, the model ``shade`` approximates.
        Max, N. (1995). "Optical Models for Direct Volume Rendering." IEEE
        TVCG 1(2). The emission-absorption integral that the viewer's
        ``volumetric`` blending mode implements, and the reason ``additive``
        is a different optical regime rather than a dimmer version of it.
        Harris, M. J. & Lastra, A. (2001). "Real-Time Cloud Rendering."
        Computer Graphics Forum 20(3). Precompute each particle's
        illumination by accumulating optical depth toward the light, then
        composite the particles sorted from the eye — which is, grid-based
        rather than render-based, exactly the scheme used here.
        Dobashi, Y., Kaneda, K., Yamashita, H., Okita, T. & Nishita, T.
        (2000). "A Simple, Efficient Method for Realistic Animation of
        Clouds." SIGGRAPH '00. Evolving cloud fields drawn as attenuated
        billboards.

Usage:
    python demo_volumetric_cloud.py [--parcels=N] [--frames=N] [--no-serve]

Controls:
    - Opens already playing and rotating; K pauses, N toggles the sliders
    - Ctrl+C to stop and cleanup
    - Browser opens automatically
"""

DEMO_META = {
    "key": "cloud",
    "title": "Evolving Cloud",
    "description": (
        "A convective cumulus over its whole life cycle: air parcels advected "
        "by a divergence-free flow, condensing into cauliflower above a flat base."
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
    "outputs": ["cloud"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import math
import sys
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import (
    AnimationConfig,
    CameraConfig,
    DimensionsConfig,
    UIConfig,
    ViewerConfig,
)
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.utils.paths import get_demos_output_dir

# --- Scene scale ------------------------------------------------------------
# Everything below is expressed in these units; one unit is roughly 30 m if you
# want the cumulus to come out life-sized.
CLOUD_SIZE = 20.0  #: horizontal scale of the domain
BASE_Y = 0.0  #: lifting condensation level — the flat cloud base
TOP_MAX = 16.5  #: how high above the base the mature turret reaches
SEED_RADIUS = 10.0  #: parcels are seeded in a cylinder of this radius
SEED_Y = (-2.0, 18.0)  #: ...spanning this height range

# --- Flow ---------------------------------------------------------------
# Speeds are per unit PHASE — per whole sequence — not per frame, and the
# distinction is the difference between `--frames` being a resolution knob and
# being a physics knob. Integrating a fixed displacement once per frame makes
# the total distance travelled proportional to the frame count, so asking for
# twice the temporal resolution would silently give you a cloud that drifts
# twice as far. Multiplying by dt = 1/(n_frames-1) instead means every frame
# count samples the SAME evolution, just more finely.
UPDRAFT = 13.0  #: peak core updraft, scene units per unit phase
ROLL_A = 4.2  #: core radius of the convection roll
ROLL_H = 22.0  #: vertical wavelength of the roll (no flux through its ends)
ROLL_Y0 = 8.0  #: height of maximum updraft
SHEAR = 0.35  #: d(u_x)/dy — the wind shear that leans the cloud downwind
SWIRL = 0.59  #: rotation about the vertical, radians per unit phase
SWIRL_B = 6.0  #: radius over which the swirl decays
TILT = 0.22  #: static lean of the cloud's own axis, matching the shear
SUBSTEPS = 2  #: midpoint substeps per frame

# --- Condensation field -----------------------------------------------------
NOISE_OCTAVES = 6
NOISE_BASE_FREQ = 3.5  #: lattice cells across the domain at the first octave
NOISE_PERSISTENCE = 0.55
NOISE_TIME_SPAN = 3.2  #: noise-time units traversed over the whole sequence
NOISE_EVOLVING_OCTAVES = 4  #: finer octaves are frozen and merely advected
N_BUBBLES = 42  #: thermals whose union makes the cauliflower
BUBBLE_RADIUS = 3.1  #: peak radius of one thermal
BUBBLE_SPREAD = 1.9  #: lateral scatter of thermals about the leaning axis
BUBBLE_TAPER = 0.42  #: how much thinner a thermal is near the crown
ROOT_RADIUS = 3.0  #: half-width of the slab of cloud sitting on the base
ROOT_DEPTH = 4.6  #: how far up from the base that slab reaches
BUBBLE_FLATTEN = 0.82  #: thermals are squashed by their own drag
UNION_POWER = 3.0  #: p-norm exponent; a hard max creases where bubbles meet
BILLOW = 0.28  #: how strongly coarse noise bulges the silhouette
BILLOW_FREQ = 3.5  #: lattice cells across the domain for the silhouette lobes
BILLOW_SEED = 500_000  #: keeps the silhouette noise independent of the detail
BASE_SOFT = 0.4  #: how sharply condensate switches on at the base
BASE_WOBBLE = 0.30  #: coarse variation of the base altitude — flat, not razor-flat
BASE_LIFT = 0.55  #: how much the threshold drops at the base, keeping it solid
BASE_LIFT_DEPTH = 2.6  #: height over which that base boost fades out
SOLIDITY = 0.26  #: how far the core is pushed toward fully saturated
SOLID_REF = 0.78  #: envelope value at which the core counts as solid

# --- Appearance -------------------------------------------------------------
MIN_RADIUS = 0.10
MAX_RADIUS = 0.62
SHADE_GRID = 44  #: cells per axis of the optical-depth grid
EXTINCTION = 1.0  #: extinction per unit of water column (see optical_depth)
#: Sun position, as a direction from the cloud toward the light. Deliberately
#: off to one side and only moderately high: a sun directly overhead lights
#: every lobe by depth alone and the cauliflower relief vanishes.
SUN_DIRECTION = (-0.62, 0.68, 0.39)
SKY_FALLOFF = 0.55  #: skylight is hemispherical, so it penetrates further
#: Radiances, in LINEAR light, and their sum is the point of the numbers.
#: Under emission-absorption the accumulated radiance of an optically thick
#: medium tends to emission/extinction, i.e. to the parcel colour itself — so a
#: fully lit parcel IS the brightest pixel the cloud can produce. Summing three
#: terms that each looked reasonable gave 1.4, every lit face clipped to flat
#: white, and the shading that took all this work became invisible. They now
#: sum to ~0.40. The cinematic preset's bloom threshold is 0.01, so the WHOLE
#: cloud blooms onto itself rather than just its highlights; exposing for ACES
#: alone left every lit face clipped to flat white with the shading invisible.
SUN_COLOR = np.array([0.30, 0.288, 0.265], dtype=np.float32)
SKY_COLOR = np.array([0.085, 0.110, 0.165], dtype=np.float32)
GROUND_COLOR = np.array([0.014, 0.012, 0.010], dtype=np.float32)
INTENSITY = 1.0  #: node gain; volumetric emission is driven by per-point alpha
ABSORPTION = 1.6  #: kappa — how strongly a parcel hides what is behind it
ALPHA_FLOOR = 0.10  #: opacity of the thinnest emitted parcel
ALPHA_GAIN = 0.62  #: extra opacity carried by the densest

# --- Defaults ---------------------------------------------------------------
DEFAULT_PARCELS = 600_000
DEFAULT_FRAMES = 120
TARGET_POINTS_PER_FRAME = 35_000
DENSITY_POWER = 1.9  #: emission probability ~ (water/gate)^this
OPENING_PHASE = 0.55  #: the scene opens on the mature cloud, not on frame 0
CALIBRATION_PHASES = (0.2, 0.35, 0.5, 0.65, 0.8)  #: probes for the point budget
FRAMING_MARGIN = 1.30  #: headroom around the cloud in the opening shot
FRAMING_QUANTILE = 0.995  #: share of the cloud the opening shot must contain
MIN_FRAMING_RADIUS = 0.05  #: keeps a degenerate cloud from collapsing the pose
#: Opening view direction, from the front, right and a little below — the
#: angle you actually see a cumulus from, and the one that puts the flat
#: base edge-on instead of hiding it underneath.
VIEW_DIRECTION = (0.62, -0.16, 1.0)
#: Turntable speed for the opening orbit. The viewer's slider spans 0.1 to 5.0
#: and defaults to 0.25, which is a revolution every few minutes — too slow to
#: read as motion. At OrbitControls' convention this is roughly one revolution
#: every 26 seconds: enough for the cauliflower relief and the shear-leaned top
#: to come round and be seen from more than the one angle the opening pose
#: gives, and slow enough not to fight scrubbing the time axis at the same time.
AUTO_ROTATE_SPEED = 2.3
#: Timepoints per second ASKED FOR. It is a ceiling, not a promise: the viewer
#: throttles dimension playback to what chunk streaming can keep up with, and
#: at ~25k points a frame this scene measures about 4 fps in practice — call it
#: half a minute for a full life cycle. Asking for more than that costs
#: nothing and lets a faster machine, or a smaller `--parcels`, run nearer the
#: rate the motion was designed at.
PLAYBACK_FPS = 15.0


def simple_noise_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Trilinearly interpolated VALUE noise on the integer lattice.

    Value noise, not gradient noise: each lattice corner is assigned a scalar
    straight from a hash of its coordinates, and the eight corners of the
    containing cell are trilinearly blended. Perlin's original assigns
    gradients instead and interpolates a dot product [Perlin 1985], which
    avoids the faint axis-aligned bias value noise has. That bias is invisible
    here because the field is only ever seen through a threshold and a cloud
    envelope, and value noise is a good deal cheaper — one hash per corner
    rather than a gradient lookup and a dot product.

    The fade is Perlin's quintic ``6t^5 - 15t^4 + 10t^3`` [Perlin 2002] rather
    than the older cubic ``3t^2 - 2t^3``: both kill the first derivative at the
    lattice, only the quintic also kills the second, and without that the cell
    boundaries show as faint creases once the field is differentiated by a
    threshold.

    Not cryptographically secure, and does not need to be.

    Args:
        x, y, z: Coordinate arrays (same shape)
        seed: Random seed for reproducibility

    Returns:
        Noise values in approximate range [-1, 1]
    """
    # This creates smooth interpolated noise from grid coordinates.
    # Note: this is a deterministic positional hash (see hash_coords below) - it
    # does not touch the global RNG, so no np.random.seed() call is needed here.

    # Get integer grid coordinates
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    zi = np.floor(z).astype(np.int64)

    # Fractional parts for interpolation
    xf = x - xi
    yf = y - yi
    zf = z - zi

    # Smooth interpolation (fade function: 6t^5 - 15t^4 + 10t^3)
    u = xf * xf * xf * (xf * (xf * 6 - 15) + 10)
    v = yf * yf * yf * (yf * (yf * 6 - 15) + 10)
    w = zf * zf * zf * (zf * (zf * 6 - 15) + 10)

    # Generate pseudo-random values for the cube corners from a position hash.
    #
    # The mixing steps below are not decoration. A bare
    # `(xi*C1 + yi*C2 + zi*C3) % M` leaves the low bits of the product sum
    # almost untouched, so the "random" value stays correlated with the
    # magnitude of the coordinates — near the lattice origin every corner
    # comes back close to the same value, and the field acquires a smooth
    # radial bias that no amount of octave stacking removes. What that looks
    # like downstream is a cloud with a systematically empty core. The
    # finalizer here is xxHash32's avalanche [Collet, xxHash] — two
    # xor-shift/multiply rounds over its PRIME32 constants, the same shape as
    # MurmurHash3's fmix32 — which makes every output bit depend on every
    # input bit. Still not cryptographically secure, and does not need to be.
    def hash_coords(xi, yi, zi):  # type: ignore[no-untyped-def]
        h = (
            xi.astype(np.uint64) * np.uint64(374761393)
            + yi.astype(np.uint64) * np.uint64(668265263)
            + zi.astype(np.uint64) * np.uint64(1274126177)
            + np.uint64(seed & 0xFFFFFFFF) * np.uint64(2654435761)
        ) & np.uint64(0xFFFFFFFF)
        h ^= h >> np.uint64(15)
        h = (h * np.uint64(2246822519)) & np.uint64(0xFFFFFFFF)
        h ^= h >> np.uint64(13)
        h = (h * np.uint64(3266489917)) & np.uint64(0xFFFFFFFF)
        h ^= h >> np.uint64(16)
        return h.astype(np.float32) * np.float32(2.0 / 0xFFFFFFFF) - np.float32(1.0)

    # Get gradient values at cube corners (8 corners)
    # This is simplified - real Perlin uses gradient vectors
    n000 = hash_coords(xi, yi, zi)
    n001 = hash_coords(xi, yi, zi + 1)
    n010 = hash_coords(xi, yi + 1, zi)
    n011 = hash_coords(xi, yi + 1, zi + 1)
    n100 = hash_coords(xi + 1, yi, zi)
    n101 = hash_coords(xi + 1, yi, zi + 1)
    n110 = hash_coords(xi + 1, yi + 1, zi)
    n111 = hash_coords(xi + 1, yi + 1, zi + 1)

    # Trilinear interpolation
    # Interpolate along x
    nx00 = n000 * (1 - u) + n100 * u
    nx01 = n001 * (1 - u) + n101 * u
    nx10 = n010 * (1 - u) + n110 * u
    nx11 = n011 * (1 - u) + n111 * u

    # Interpolate along y
    nxy0 = nx00 * (1 - v) + nx10 * v
    nxy1 = nx01 * (1 - v) + nx11 * v

    # Interpolate along z
    nxyz = nxy0 * (1 - w) + nxy1 * w

    return nxyz  # type: ignore[no-any-return]


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, clamped outside ``[edge0, edge1]``."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)  # type: ignore[no-any-return]


def _smoothstep_scalar(edge0: float, edge1: float, x: float) -> float:
    """Scalar smoothstep, for the life-cycle curves."""
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return float(t * t * (3.0 - 2.0 * t))


# ---------------------------------------------------------------------------
# 4D fractal noise in material coordinates
# ---------------------------------------------------------------------------


def build_noise_series(
    material: np.ndarray,
    octaves: int = NOISE_OCTAVES,
    base_frequency: float = NOISE_BASE_FREQ,
    evolving_octaves: int = NOISE_EVOLVING_OCTAVES,
    time_span: float = NOISE_TIME_SPAN,
    seed_offset: int = 0,
) -> List[Tuple[np.ndarray, float]]:
    """Precompute the time keyframes of a 4D fractal noise at fixed points.

    The parcels' material coordinates never change, so the only thing the noise
    depends on is time — which means the whole 4D field collapses to a small
    stack of static 3D fields per octave, quintic-interpolated in between.
    That IS 4D value noise; it is simply the cheapest possible way to evaluate
    one when the spatial sample points are known up front.

    The temporal frequency of octave ``k`` grows as ``2^(2k/3)`` rather than the
    ``2^k`` a naive 4D noise would use. This is Kolmogorov's inertial-range
    scaling [Frisch 1995]: an eddy of size ``l`` turns over in a time
    ``tau(l) ~ l^(2/3)``, so halving the length scale speeds the evolution by
    ``2^(2/3)`` and not by 2. Small eddies really do turn over faster than
    large ones, but at the naive ``2^k`` rate the finest octaves read as
    shimmer rather than as turbulence. Octaves at or beyond
    ``evolving_octaves`` are frozen entirely and left to the flow to move.

    Args:
        material: ``(P, 3)`` material coordinates (each parcel's fixed label)
        octaves: Number of spatial octaves
        base_frequency: Lattice cells per unit at the first octave
        evolving_octaves: How many octaves carry a time axis
        time_span: Noise-time units traversed over the whole sequence
        seed_offset: Shifts every hash, giving a statistically independent field

    Returns:
        One ``(keyframes, temporal_frequency)`` pair per octave. A temporal
        frequency of zero marks a frozen octave with a single keyframe.
    """
    mx, my, mz = material[:, 0], material[:, 1], material[:, 2]
    series: List[Tuple[np.ndarray, float]] = []

    for octave in range(octaves):
        frequency = base_frequency * 2.0**octave
        # Kolmogorov-ish eddy turnover: tau ~ l^(2/3), so omega ~ l^(-2/3).
        temporal = 2.0 ** (2.0 * octave / 3.0) if octave < evolving_octaves else 0.0
        n_keys = int(np.ceil(time_span * temporal)) + 1 if temporal > 0.0 else 1

        keys = np.empty((n_keys, material.shape[0]), dtype=np.float32)
        for k in range(n_keys):
            keys[k] = simple_noise_3d(
                mx * frequency,
                my * frequency,
                mz * frequency,
                seed=seed_offset + 977 * octave + 31337 * k,
            )
        series.append((keys, temporal))

    return series


def sample_octave(
    series: List[Tuple[np.ndarray, float]], octave: int, tau: float
) -> np.ndarray:
    """Evaluate one octave of the precomputed noise at noise-time ``tau``."""
    keys, temporal = series[octave]
    if temporal == 0.0:
        return keys[0]

    s = tau * temporal
    k0 = min(int(np.floor(s)), keys.shape[0] - 1)
    k1 = min(k0 + 1, keys.shape[0] - 1)
    if k0 == k1:
        return keys[k0]

    f = s - k0
    # Same quintic fade as the spatial interpolation, so the time axis has
    # continuous first and second derivatives too.
    u = f * f * f * (f * (f * 6 - 15) + 10)
    mixed = keys[k0] * (1.0 - u) + keys[k1] * u

    # Variance-preserving normalization, and it is not a nicety. Blending two
    # INDEPENDENT keyframes with weights that sum to one gives a variance of
    # (1-u)^2 + u^2, which is 1 at a keyframe and only 1/2 halfway between —
    # so the field's contrast sags between keyframes and recovers at them.
    # Spatially this is invisible, because neighbouring samples sit at
    # different lattice phases and it averages out. Along TIME every parcel
    # shares the same u, so the whole cloud breathes in and out of focus in
    # step with the keyframe grid. Dividing by the norm of the weight vector
    # holds the variance flat; the quintic fade still kills the derivative at
    # the keyframes, so the result stays smooth.
    norm = np.sqrt((1.0 - u) ** 2 + u**2)
    return mixed / norm  # type: ignore[no-any-return]


def sample_noise_series(
    series: List[Tuple[np.ndarray, float]],
    tau: float,
    persistence: float = NOISE_PERSISTENCE,
) -> np.ndarray:
    """Evaluate the precomputed 4D noise at noise-time ``tau``.

    This is where the fractional Brownian motion is actually summed
    [Ebert et al. 2003]: each octave doubles in frequency (lacunarity 2) and
    is scaled by ``persistence``, so the spectrum falls off as a power law and
    the field is statistically self-similar the way real cloud edges are. The
    octaves themselves were evaluated once, at fixed material coordinates, by
    :func:`build_noise_series` — all that is left here is the weighted sum.

    Args:
        series: Output of :func:`build_noise_series`
        tau: Noise-time coordinate
        persistence: Amplitude falloff per octave

    Returns:
        Fractal noise values in approximately ``[-1, 1]``
    """
    out = np.zeros(series[0][0].shape[1], dtype=np.float32)
    amplitude, total = 1.0, 0.0

    for octave in range(len(series)):
        out += sample_octave(series, octave, tau) * amplitude
        total += amplitude
        amplitude *= persistence

    return out / total  # type: ignore[no-any-return]


class NoiseField(NamedTuple):
    """The two noise fields the cloud is made of.

    ``detail`` textures the interior; ``billow`` shapes the silhouette. They
    are deliberately *independent* fields rather than two reads of the same
    one. Reusing the coarsest detail octave as the silhouette modulator seems
    thrifty and is in fact a trap: wherever it dips, the envelope narrows AND
    the noise falls below threshold, so the two effects multiply and carve the
    turret into hollow prongs instead of rounding it into lobes.

    Attributes:
        detail: Multi-octave series for the condensate texture
        billow: Single coarse octave for the silhouette lobes
    """

    detail: List[Tuple[np.ndarray, float]]
    billow: List[Tuple[np.ndarray, float]]


def build_noise_field(material: np.ndarray) -> NoiseField:
    """Build the detail and silhouette noise fields.

    Args:
        material: ``(P, 3)`` material coordinates

    Returns:
        The assembled :class:`NoiseField`
    """
    return NoiseField(
        detail=build_noise_series(material),
        billow=build_noise_series(
            material,
            octaves=1,
            base_frequency=BILLOW_FREQ,
            evolving_octaves=1,
            seed_offset=BILLOW_SEED,
        ),
    )


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def velocity(positions: np.ndarray, updraft: float) -> np.ndarray:
    """An analytic, exactly divergence-free velocity field, in units per frame.

    Three superposed pieces, each solenoidal on its own:

    1. An axisymmetric convection roll, written through a Stokes stream
       function [Batchelor 1967, pp. 78-79]
       ``psi = U r^2 exp(-r^2 / 2a^2) cos(k (y - y0))``. Taking
       ``u_r = -(1/r) dpsi/dy`` and ``u_y = (1/r) dpsi/dr`` makes the
       divergence identically zero by construction, and the resulting cell
       rises in the core, diverges at the top, sinks outside and converges at
       the base — a cumulus turning itself over. The phase is clamped to
       ``[-pi/2, pi/2]`` so there is no flux through the roll's ends.
    2. A wind shear ``u_x += S (y - base)``. Its divergence vanishes because
       ``u_x`` has no ``x`` dependence. This is what leans the cloud downwind.
    3. A swirl about the vertical whose strength depends only on the radius,
       so the two cross terms of its divergence cancel exactly.

    Divergence-freeness is not decoration [Bridson et al. 2007]: the parcels
    are a Monte Carlo sample of a uniform density, and only a solenoidal field
    keeps that sample uniform as it deforms. A field that compressed would pile
    parcels up and read as brightness drifting where no water went. Writing the
    flow through a stream function gets this exactly, by construction, rather
    than approximately — the divergence is identically zero in the algebra, not
    small in a measurement.

    Args:
        positions: ``(P, 3)`` world positions
        updraft: Strength of the convection roll at this instant

    Returns:
        ``(P, 3)`` displacement per frame
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    r2 = x * x + z * z
    falloff = np.exp(-r2 / (2.0 * ROLL_A**2))
    k = np.pi / ROLL_H
    phase = np.clip(k * (y - ROLL_Y0), -np.pi / 2.0, np.pi / 2.0)

    # u_y = (1/r) dpsi/dr, u_r = -(1/r) dpsi/dy — the radial one divided by r
    # up front so the axis (r = 0) stays finite.
    u_y = updraft * (2.0 - r2 / ROLL_A**2) * falloff * np.cos(phase)
    u_r_over_r = updraft * falloff * k * np.sin(phase)
    u_x = u_r_over_r * x
    u_z = u_r_over_r * z

    # Wind shear: leans the column downwind with height.
    u_x = u_x + SHEAR * (y - BASE_Y)

    # Swirl about the vertical.
    omega = SWIRL * np.exp(-r2 / (2.0 * SWIRL_B**2))
    u_x = u_x - omega * z
    u_z = u_z + omega * x

    return np.stack([u_x, u_y, u_z], axis=1).astype(np.float32)


def advect(
    positions: np.ndarray,
    updraft: float,
    dt: float,
    substeps: int = SUBSTEPS,
) -> None:
    """Push the parcels forward by ``dt`` of phase, in place.

    Explicit midpoint (RK2): evaluate the field, step half way on that
    estimate, and take the full step using the velocity found there. Second
    order globally, so halving ``dt`` cuts the trajectory error roughly
    fourfold — which is what lets the frame count be a pure resolution choice
    rather than a change to the motion (see
    ``test_the_flow_is_frame_rate_independent``). Plain forward Euler would be
    first order and would systematically fling parcels outward on the roll's
    curved streamlines, thinning the core over the course of a run.

    Args:
        positions: ``(P, 3)`` world positions, modified in place
        updraft: Strength of the convection roll at this instant
        dt: Phase advanced by this call, i.e. ``1 / (n_frames - 1)``
        substeps: Midpoint steps used to cover ``dt``
    """
    step = dt / substeps
    for _ in range(substeps):
        mid = positions + velocity(positions, updraft) * (0.5 * step)
        positions += velocity(mid, updraft) * step


# ---------------------------------------------------------------------------
# The thermodynamic envelope and the life cycle
# ---------------------------------------------------------------------------


class LifeCycle(NamedTuple):
    """The cloud's state at one instant of its life cycle.

    Attributes:
        amplitude: Overall condensate scale
        top: Cloud-top altitude
        width: Multiplier on the turret's half-width
        sigmas: Noise threshold, in standard deviations above the field mean
    """

    amplitude: float
    top: float
    width: float
    sigmas: float


def life_cycle(phase: float) -> LifeCycle:
    """The cloud's state at a given phase of its life.

    The arc is: a low ragged fragment at the condensation level, a turret
    growing up and out, a mature sheared cloud, then entrainment eating it back
    into rags. Size carries both ends of it — the body grows and then pulls
    back in — while the threshold adds raggedness at the extremes, crisp in the
    middle of the life and shredded at either end.

    Every curve here is deliberately shallow, and the reason is worth stating
    because it is easy to undo. Emission probability goes as the condensate to
    the power :data:`DENSITY_POWER`, and the envelope's volume goes as roughly
    width squared times height, so a life cycle that looks gentle written down
    compounds into a point count that swings by two orders of magnitude. A
    cloud that empties at either end leaves dead frames at exactly the two
    timepoints a viewer reaches first (``Home`` and ``End``), which reads as a
    broken dataset rather than as a life cycle.

    Args:
        phase: Normalized time in ``[0, 1]``

    Returns:
        The :class:`LifeCycle` state
    """
    grow = _smoothstep_scalar(0.0, 0.34, phase)
    swell = _smoothstep_scalar(0.0, 0.55, phase)
    decay = 1.0 - _smoothstep_scalar(0.70, 1.0, phase)

    amplitude = 0.80 + 0.20 * grow * decay
    # Both the top and the width carry the decay, and that is a consequence of
    # rendering the cloud as an opaque body rather than a glow. Dissipation used
    # to be expressed by raising the noise threshold, which erodes the INTERIOR
    # — invisible once you can only see the surface. What a viewer can actually
    # see shrink is the silhouette, so the turret has to sag and the body pull
    # in for the end of the life cycle to read at all.
    top = BASE_Y + TOP_MAX * (0.70 + 0.30 * swell) * (0.86 + 0.14 * decay)
    width = (0.80 + 0.20 * swell) * (0.74 + 0.26 * decay)
    # Low threshold = solid cloud; high threshold = shredded filaments. A young
    # turret is compact and crisp rather than wispy, so only a little of the
    # early raggedness comes from here — most of it is simply being small.
    sigmas = -0.10 + 0.10 * (1.0 - grow) + 0.26 * _smoothstep_scalar(0.66, 1.0, phase)
    return LifeCycle(amplitude, top, width, sigmas)


def cloud_base(billow: np.ndarray) -> np.ndarray:
    """Altitude of the cloud base per parcel.

    Condensation is a sharp thermodynamic threshold, which is why every cumulus
    in a field shares one altitude for its bottom. The coarse noise is allowed
    only a slight say in it — enough that the base reads as a real surface
    rather than a machined plane, not enough to lose the flatness that makes
    the cloud legible as a cumulus.

    Args:
        billow: ``(P,)`` coarse noise, roughly unit variance

    Returns:
        ``(P,)`` base altitude
    """
    return BASE_Y + BASE_WOBBLE * billow  # type: ignore[no-any-return]


class Bubbles(NamedTuple):
    """The rising thermal bubbles the cumulus is built out of.

    Attributes:
        offset: ``(K, 2)`` lateral offset of each bubble from the leaning axis
        radius: ``(K,)`` peak radius
        birth: ``(K,)`` phase at which the bubble starts rising
        life: ``(K,)`` how long, in phase, it takes to rise and dissolve
        ceiling: ``(K,)`` fraction of the cloud top this bubble climbs to
    """

    offset: np.ndarray
    radius: np.ndarray
    birth: np.ndarray
    life: np.ndarray
    ceiling: np.ndarray


def build_bubbles(rng: np.random.Generator, count: int = N_BUBBLES) -> Bubbles:
    """Lay out the thermals whose union is the cloud.

    A cumulus is not a shape, it is a PROCESS: a succession of buoyant bubbles
    punching up through the condensation level, each overshooting a little less
    than the last, mixing away at its edges as it goes. This is the bubble
    theory of penetrative convection [Scorer & Ludlam 1953], and rendering the
    union of those bubbles is what produces the cauliflower — the lobes, the
    crevices between them, and a silhouette that changes as new thermals
    arrive. Their flanks are eroded by dry air mixing in [Stommel 1947], which
    is what the noise threshold stands in for and what ends the life cycle.

    The previous version drove the outline from a single surface of revolution,
    which is a lathe, and it looked like one: one smooth mass with no lobes for
    the light to catch. No amount of noise on the radius fixed that, because
    noise perturbs a shape whereas cauliflower IS the shape.

    Births are spread from before the sequence starts to near its end, so the
    opening frame already has mature bubbles rather than opening on an empty
    sky. Radii are drawn independently, but the CEILING rises with birth order:
    later thermals climb into air that earlier ones have already moistened, and
    a run of ceilings clustered low leaves the column squat.

    Args:
        rng: Seeded generator, so a given demo run is reproducible
        count: How many bubbles

    Returns:
        The :class:`Bubbles` layout
    """
    birth = np.linspace(-0.85, 0.85, count) + rng.uniform(-0.02, 0.02, count)
    order = np.linspace(0.0, 1.0, count)

    angle = rng.uniform(0.0, 2.0 * np.pi, count)
    spread = BUBBLE_SPREAD * np.sqrt(rng.random(count))
    offset = np.column_stack([spread * np.cos(angle), spread * np.sin(angle)])

    radius = BUBBLE_RADIUS * rng.uniform(0.72, 1.28, count)
    life = rng.uniform(0.60, 0.95, count)
    # Most thermals go most of the way up. Ceilings clustered low left the
    # column squat: the cloud never reached the altitude the life cycle was
    # raising its top to, so the growth arc had nothing to show.
    ceiling = np.clip(0.60 + 0.32 * order + rng.uniform(-0.07, 0.07, count), 0.45, 0.92)

    return Bubbles(
        offset.astype(np.float32),
        radius.astype(np.float32),
        birth.astype(np.float32),
        life.astype(np.float32),
        ceiling.astype(np.float32),
    )


def envelope(
    positions: np.ndarray, phase: float, billow: np.ndarray, bubbles: Bubbles
) -> np.ndarray:
    """Where the air is cool enough and moist enough to hold liquid water.

    The union of the thermals, cut off underneath at the condensation level.
    A cumulus is a cumulus because of that cut: below the lifting condensation
    level — the height a surface parcel must rise to before it saturates
    [Rogers & Yau 1989] — there is no liquid water at all, which is why every
    cumulus in a field shares one flat bottom at the same altitude.

    The union is Ricci's p-norm blend [Ricci 1973], ``(sum f_i^p)^(1/p)``,
    which tends to a hard ``max`` as p grows and to a plain sum as p tends to
    1. A hard max creases visibly where two bubbles meet and the crease reads
    as a seam in what is supposed to be one body of cloud; a plain sum bulges
    wherever bubbles merely overlap. The same family of summed radial fields
    is what blobby models are built from [Blinn 1982a].

    Args:
        positions: ``(P, 3)`` world positions
        phase: Normalized time in ``[0, 1]``
        billow: ``(P,)`` coarse noise, roughly unit variance
        bubbles: Output of :func:`build_bubbles`

    Returns:
        ``(P,)`` envelope in ``[0, 1]``
    """
    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
    state = life_cycle(phase)
    base = cloud_base(billow)

    accumulated = np.zeros(len(positions), dtype=np.float32)
    for k in range(len(bubbles.radius)):
        age = (phase - bubbles.birth[k]) / bubbles.life[k]
        if age <= 0.0 or age >= 1.0:
            continue

        # Rises from the base to its own ceiling, and swells then shrinks. Both
        # endpoints are zero radius, so a bubble fades in and out rather than
        # appearing at full size. The climb is LINEAR in age on purpose: an
        # age^0.7 climb bunches the live thermals into the middle of the
        # column, and the cloud comes out as a ball with an empty base and an
        # empty crown.
        height = BASE_Y + bubbles.ceiling[k] * state.top * age
        taper = 1.0 - BUBBLE_TAPER * (height - BASE_Y) / max(state.top, 1e-6)
        # The late fade is sharp on purpose. A gentle one leaves a thermal at
        # 60% of full size when it reaches its ceiling, sitting clear of the
        # crowd below with nothing to merge into — which renders as a detached
        # sphere floating above the cloud like a balloon.
        swell = np.sin(np.pi * age) ** 0.42
        swell *= 1.0 - _smoothstep_scalar(0.72, 1.0, age)
        radius = bubbles.radius[k] * state.width * taper * swell
        if radius <= 1e-3:
            continue

        cx = TILT * (height - BASE_Y) + bubbles.offset[k, 0]
        cz = bubbles.offset[k, 1]
        # Thermals are flattened by their own drag, not spherical.
        dx = (x - cx) / radius
        dy = (y - height) / (radius * BUBBLE_FLATTEN)
        dz = (z - cz) / radius
        falloff = np.exp(-(dx * dx + dy * dy + dz * dz) * 1.35)
        accumulated += falloff**UNION_POWER

    # A slab of cloud sitting ON the condensation level, always present while
    # the cloud is alive. Thermals alone leave the bottom ragged and holed —
    # each one is a sphere that has already left the base by the time it is
    # big — and the flat bottom is the single feature that says "cumulus".
    # Physically this is the layer being fed continuously from below.
    dxr = x - TILT * (y - BASE_Y)
    r_root = np.sqrt(dxr * dxr + z * z) / (ROOT_RADIUS * state.width)
    # A soft quadratic falloff, not a quartic. A quartic gives the slab a hard
    # rim, and a hard-rimmed disc wider than the tower above it reads as a
    # second, separate cloud rather than as the root of this one.
    root = np.exp(-(r_root**2)) * np.exp(-(((y - base) / ROOT_DEPTH) ** 2))
    accumulated += np.clip(root, 0.0, None) ** UNION_POWER

    union = np.clip(accumulated, 0.0, None) ** (1.0 / UNION_POWER)
    union *= 1.0 + BILLOW * billow

    e_base = _smoothstep(0.0, BASE_SOFT, y - base)
    return (state.amplitude * np.clip(union, 0.0, 1.0) * e_base).astype(np.float32)


def _standardize(
    noise: np.ndarray, weights: np.ndarray
) -> Tuple[np.float32, np.float32]:
    """Mean and spread of the noise over the parcels the threshold acts on.

    Three separate things make an absolute threshold the wrong instrument here,
    and each one only becomes visible once the previous is fixed.

    The spread of a multi-octave sum depends on the octave count and the
    persistence, so a fixed level tuned at one setting silently empties the
    cloud at another; that argues for measuring the field and working in
    standard deviations. Measuring it ONCE is then not enough, because the
    coarsest octave has only a few lattice cells across the domain and the time
    axis walks its spatial mean around — across a run the fraction of the field
    above a fixed level wandered from 0.37 to 0.64, and since every parcel
    shares one ``tau`` that wander is global, reading as the whole cloud
    thinning and thickening for no reason the life cycle asked for.

    Measuring per frame over the whole domain fixes that and still leaves the
    third. The texture is welded to the parcels, so the flow steadily replaces
    the air inside the envelope with air from elsewhere, whose noise is
    whatever it happens to be; the cloud's bulk water then follows a random
    walk driven by the large-scale structure of the material field, and the
    point count sags through the second half of the run. So the statistics are
    taken over the population the threshold is actually meant to divide —
    parcels weighted by how deep inside the envelope they sit — which leaves
    the life cycle as the only thing that moves the cloud.

    Args:
        noise: ``(P,)`` fractal field
        weights: ``(P,)`` envelope, used as the weighting

    Returns:
        ``(mean, standard deviation)``
    """
    total = float(weights.sum())
    if total <= 0.0:
        return np.float32(noise.mean()), np.float32(max(float(noise.std()), 1e-6))
    mean = float((weights * noise).sum() / total)
    var = float((weights * (noise - mean) ** 2).sum() / total)
    return np.float32(mean), np.float32(max(np.sqrt(var), 1e-6))


def condensate(
    positions: np.ndarray, field: NoiseField, phase: float, bubbles: Bubbles
) -> np.ndarray:
    """Liquid water content per parcel: 4D noise clipped by the envelope.

    Args:
        positions: ``(P, 3)`` world positions
        field: Output of :func:`build_noise_field`
        phase: Normalized time in ``[0, 1]``
        bubbles: Output of :func:`build_bubbles`

    Returns:
        ``(P,)`` condensate in ``[0, 1]``
    """
    sigmas = life_cycle(phase).sigmas
    tau = phase * NOISE_TIME_SPAN

    noise = sample_noise_series(field.detail, tau)
    coarse = sample_octave(field.billow, 0, tau)
    billow = coarse / max(float(coarse.std()), 1e-6)
    env = envelope(positions, phase, billow, bubbles)

    mean, std = _standardize(noise, env)

    # Everything measured in standard deviations, so the shape of the life
    # cycle survives a change of octave count or persistence.
    #
    # The threshold is relaxed in the first couple of units above the base.
    # Without it the noise eats the bottom into the same ragged fringe as the
    # rest of the surface, and the flat base — the one feature that says
    # "cumulus" at a glance — never appears.
    depth = (positions[:, 1] - cloud_base(billow)) / BASE_LIFT_DEPTH
    lift = BASE_LIFT * np.exp(-(np.clip(depth, 0.0, None) ** 2))
    threshold = mean + (sigmas - lift) * std
    supersaturation = np.clip((noise - threshold) / (1.6 * std), 0.0, 1.0)

    # Noise ERODES the body rather than composing it. Thresholded fractal noise
    # applied uniformly punches holes straight through the middle, and a
    # cumulus with daylight through its core is the one thing it never is:
    # it is optically thick within a few metres of its surface. So the core is
    # pushed toward saturation in proportion to how deep inside the envelope it
    # sits, and the noise keeps full authority only out at the ragged fringe —
    # which is where a real cloud's structure actually lives. A little is held
    # back everywhere so the interior does not go featureless.
    solid = np.clip(env / SOLID_REF, 0.0, 1.0) ** 1.5
    supersaturation += SOLIDITY * solid * (1.0 - supersaturation)

    return (env * supersaturation).astype(np.float32)


# ---------------------------------------------------------------------------
# Baked lighting
# ---------------------------------------------------------------------------


def _sun_frame(direction: Tuple[float, float, float]) -> np.ndarray:
    """Orthonormal basis whose THIRD axis points at the light."""
    w = np.array(direction, dtype=np.float64)
    w /= np.linalg.norm(w)
    seed = np.array([0.0, 0.0, 1.0]) if abs(w[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(seed, w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    return np.stack([u, v, w], axis=1)


def optical_depth(
    positions: np.ndarray,
    water: np.ndarray,
    parcel_density: float,
    direction: Tuple[float, float, float],
) -> np.ndarray:
    """Optical depth accumulated toward ``direction``.

    Points are emissive — nothing in the renderer knows the sun exists — so the
    only way to get a lit cloud is to bake it. Splat the condensate onto a grid
    aligned with the light, blur it once so the sampling does not band,
    cumulatively sum it from the lit side inward, and read it back per parcel.
    What comes out is the water column between each parcel and the light, which
    is exactly the argument of Beer-Lambert: transmittance ``exp(-tau)``.

    This is Harris & Lastra's cloud-shading scheme [Harris & Lastra 2001] —
    precompute each particle's illumination by accumulating optical depth
    toward the light, then composite the particles sorted from the eye — done
    on a grid rather than by rendering from the light's point of view, which
    suits a job that already has every parcel in memory as an array.

    The direction is a parameter rather than hard-coded to straight down, and
    that is a substantive choice. A light directly overhead illuminates every
    surface of the cloud by its DEPTH alone, so two lobes at the same altitude
    are lit identically no matter which way they face, and the relief that
    makes a cumulus legible disappears. Moving the sun off to one side means a
    lobe's near face is bright while its far face is not, which is what draws
    the cauliflower.

    Args:
        positions: ``(P, 3)`` world positions
        water: ``(P,)`` condensate
        parcel_density: Seeded parcels per unit volume
        direction: Unit-ish vector pointing from the cloud toward the light

    Returns:
        ``(P,)`` optical depth
    """
    local = positions @ _sun_frame(direction)

    lo = local.min(axis=0)
    hi = local.max(axis=0)
    span = np.maximum(hi - lo, 1e-3)
    cell = span / SHADE_GRID

    idx = np.clip(((local - lo) / cell).astype(np.int32), 0, SHADE_GRID - 1)
    flat = (idx[:, 0] * SHADE_GRID + idx[:, 1]) * SHADE_GRID + idx[:, 2]

    grid = np.zeros(SHADE_GRID**3, dtype=np.float32)
    np.add.at(grid, flat, water)
    grid = grid.reshape(SHADE_GRID, SHADE_GRID, SHADE_GRID)

    # One separable 3-tap blur: enough to stop the nearest-cell read-back from
    # showing the grid, cheap enough not to matter.
    for axis in (0, 1, 2):
        grid = (
            grid
            + 0.5 * np.roll(grid, 1, axis=axis)
            + 0.5 * np.roll(grid, -1, axis=axis)
        ) / 2.0

    # Convert the per-cell SUM of parcel water into a water DENSITY before
    # integrating. Skipping this and folding the geometry into one magic
    # constant works right up until somebody passes --parcels or edits
    # SHADE_GRID, at which point the whole cloud silently changes how it is
    # lit: the raw sum scales with both the parcel count and the cell volume.
    cell_volume = float(cell[0] * cell[1] * cell[2])
    density = grid / max(parcel_density * cell_volume, 1e-12)

    # Exclusive cumulative sum from the lit end (the far end of axis 2, which
    # the frame points at the light) inward.
    above = np.cumsum(density[:, :, ::-1], axis=2)[:, :, ::-1] - density
    tau = above * (float(cell[2]) * EXTINCTION)

    return tau[idx[:, 0], idx[:, 1], idx[:, 2]]  # type: ignore[no-any-return]


def shade(
    positions: np.ndarray, water: np.ndarray, parcel_density: float
) -> np.ndarray:
    """Per-parcel colour: sunlight, skylight and a little bounce off the ground.

    Three terms, because a cloud lit by one of them looks wrong in a way that
    is hard to place until you see all three. Each is a single-scattering
    approximation — albedo times the light that survives the journey in
    [Blinn 1982b] — with no multiple scattering, which is why the constants
    below are tuned rather than derived: the whiteness of a real cumulus comes
    largely from light bouncing many times inside it.

    ``sun`` is direct beam attenuated along the sun's own ray. On its own it
    renders everything the beam misses as pure black, which no cloud has ever
    been. ``sky`` is the hemisphere of blue light from above, attenuated along
    the vertical, and it is what actually fills a cumulus's shadowed flanks —
    and it is BLUE, which is why the underside of a real cumulus reads cool
    grey-blue rather than neutral grey. ``ground`` is a weak warm bounce that
    keeps the very bottom from going flat.

    Args:
        positions: ``(P, 3)`` world positions
        water: ``(P,)`` condensate
        parcel_density: Seeded parcels per unit volume

    Returns:
        ``(P, 3)`` linear RGB
    """
    tau_sun = optical_depth(positions, water, parcel_density, SUN_DIRECTION)
    tau_sky = optical_depth(positions, water, parcel_density, (0.0, 1.0, 0.0))

    direct = np.exp(-tau_sun)[:, None]
    # Skylight arrives from a whole hemisphere, so it is far less directional
    # than the beam and falls off more gently with depth.
    ambient = np.exp(-SKY_FALLOFF * tau_sky)[:, None]

    return (
        SUN_COLOR[None, :] * direct
        + SKY_COLOR[None, :] * ambient
        + GROUND_COLOR[None, :]
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Emission gate
# ---------------------------------------------------------------------------


def emission_odds(water: np.ndarray, gate: float) -> np.ndarray:
    """Probability that a parcel with this much water is emitted as a point.

    A parcel carries a *frozen* uniform draw ``u`` and is emitted whenever
    these odds clear it, so the expected point density in a region tracks the
    condensate there — point count carries the density of the cloud, on top of
    what radius and brightness already say. Freezing ``u`` per parcel rather
    than redrawing it each frame is the whole trick: a parcel switches on once
    and stays on while the water lasts, instead of flickering like a Bernoulli
    trial resampled sixty times.

    The power is what keeps the cloud from wearing a halo. Odds proportional to
    the water itself still emit a few percent of the parcels far out on the
    envelope's tail, and spread thinly over a large volume those few percent
    read as a dusting of spray around the cloud. Raising the ratio to a power
    leaves the core untouched and collapses that tail.

    Args:
        water: ``(P,)`` condensate
        gate: Gate value from :func:`calibrate_gate`

    Returns:
        ``(P,)`` probabilities in ``[0, 1]``
    """
    return np.clip(water / gate, 0.0, 1.0) ** DENSITY_POWER  # type: ignore[no-any-return]


def calibrate_gate(waters: List[np.ndarray], target_points: int) -> float:
    """Pick the gate that caps the BUSIEST frame at ``target_points``.

    Calibrating against a single sampled frame under-counts: the point budget
    then applies to whichever frame happened to be probed while the true peak
    of the life cycle runs over it. Bisecting on the maximum across several
    probe phases makes ``target_points`` a real ceiling.

    Args:
        waters: Condensate arrays sampled at several phases
        target_points: Wanted number of emitted points at the busiest frame

    Returns:
        The gate value
    """
    lo, hi = 1e-4, 4.0
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        busiest = max(float(emission_odds(w, mid).sum()) for w in waters)
        if busiest > target_points:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def compose_opening_camera(points: np.ndarray) -> CameraConfig:
    """Frame the cloud as it stands at the opening timepoint.

    Composed for the cinematic preset's 63° lens by construction rather than
    corrected for it afterwards: a perspective camera subtends a sphere of
    radius ``R`` at ``asin(R / D)``, so reading the preset's own FOV and
    inverting that leaves no legacy-FOV assumption anywhere in the pose.
    Authoring a position also suppresses the viewer's auto-framing, preserving
    the deliberate opening phase and quantile-based composition below.

    The radius comes from a high quantile of the distance to the centre, not
    from the bounding box. The cloud's outermost parcels are a sparse fringe
    with barely any light in them, and letting them set the distance pushes the
    camera back far enough to shrink the part anyone came to look at.

    Args:
        points: ``(N, 3)`` positions of the points visible at the opening frame

    Returns:
        The opening :class:`CameraConfig`
    """
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radial = float(
        np.quantile(np.linalg.norm(points - centre, axis=1), FRAMING_QUANTILE)
    )
    # The preset's field of view is VERTICAL, and this cloud is taller than it
    # is wide, so the height is what the framing is actually up against. A
    # radial quantile alone under-measures that — it averages the tall axis in
    # with two short ones — and the opening shot cut the base off the bottom of
    # the frame, which is the one feature the whole envelope exists to produce.
    half_height = float(np.quantile(np.abs(points[:, 1] - centre[1]), FRAMING_QUANTILE))
    # Floored, because a degenerate point set would otherwise put the camera
    # exactly ON its own target: `lookAt` along a zero-length direction is not
    # a view, it is a NaN. One point is enough to reach that, and a frame CAN
    # come down to one point once the emission gate is relaxed to keep the time
    # axis unholed.
    radius = max(radial, half_height, MIN_FRAMING_RADIUS)
    distance = FRAMING_MARGIN * radius / math.sin(math.radians(CINEMATIC_FOV_DEG / 2))

    direction = np.array(VIEW_DIRECTION, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    position = centre + direction * distance

    return CameraConfig(
        position=(float(position[0]), float(position[1]), float(position[2])),
        target=(float(centre[0]), float(centre[1]), float(centre[2])),
    )


def generate_evolving_cloud(
    output_path: Path,
    n_parcels: int = DEFAULT_PARCELS,
    n_frames: int = DEFAULT_FRAMES,
    target_points_per_frame: int = TARGET_POINTS_PER_FRAME,
) -> None:
    """Generate a 3D + time cumulus and write it as a Luxar scene.

    This function contains ALL the generation logic - completely self-contained.

    Args:
        output_path: Where to write the zarr store
        n_parcels: Air parcels carried through the flow (most stay invisible)
        n_frames: Timepoints on the hidden ``time`` axis
        target_points_per_frame: Emitted points at the mature frame
    """
    rng = np.random.default_rng(42)

    with asection(f"Seeding {n_parcels:,} air parcels"):
        # Uniform in a vertical cylinder: r = R sqrt(u) keeps the areal density
        # flat, which is what makes point density proportional to condensate
        # rather than to distance from the axis.
        radius = SEED_RADIUS * np.sqrt(rng.random(n_parcels))
        theta = rng.uniform(0.0, 2.0 * np.pi, n_parcels)
        positions = np.empty((n_parcels, 3), dtype=np.float32)
        positions[:, 0] = radius * np.cos(theta)
        positions[:, 1] = rng.uniform(SEED_Y[0], SEED_Y[1], n_parcels)
        positions[:, 2] = radius * np.sin(theta)

        # The parcel's fixed label. Everything about its texture is a function
        # of this, so the texture is welded to the fluid and travels with it.
        material = (positions / CLOUD_SIZE).astype(np.float32)

        # Frozen per-parcel emission threshold (see calibrate_gate).
        gate_u = rng.random(n_parcels).astype(np.float32)

        # Parcels per unit volume. The flow is divergence-free, so this stays
        # true for the whole run and the shading can use it as a fixed scale.
        seeded_volume = np.pi * SEED_RADIUS**2 * (SEED_Y[1] - SEED_Y[0])
        parcel_density = n_parcels / seeded_volume

        aprint(f"✓ Cylinder r <= {SEED_RADIUS}, y in [{SEED_Y[0]}, {SEED_Y[1]}]")
        aprint(f"✓ Parcel density: {parcel_density:.1f} per unit volume")

    with asection("Precomputing the 4D condensation noise"):
        field = build_noise_field(material)
        bubbles = build_bubbles(rng)
        for octave, (keys, temporal) in enumerate(field.detail):
            frequency = NOISE_BASE_FREQ * 2.0**octave
            state = f"{keys.shape[0]} keyframes" if temporal else "frozen (advected)"
            aprint(f"octave {octave}: {frequency:6.1f} cells/domain — {state}")
        total_mb = (
            (
                sum(k.nbytes for k, _ in field.detail)
                + sum(k.nbytes for k, _ in field.billow)
            )
            / 1024
            / 1024
        )
        aprint(f"✓ {len(field.detail)} detail octaves + 1 billow octave")
        aprint(f"✓ {total_mb:.0f} MB of keyframes")

    with asection("Calibrating the emission gate"):
        probes = [condensate(positions, field, p, bubbles) for p in CALIBRATION_PHASES]
        gate = calibrate_gate(probes, target_points_per_frame)
        counts = [int((emission_odds(w, gate) > gate_u).sum()) for w in probes]
        for p, n in zip(CALIBRATION_PHASES, counts):
            aprint(f"probe phase {p:.2f}: {n:,} points")
        # A reference condensate for the radius ramp, fixed once so point size
        # means the same thing in every frame.
        busiest = probes[int(np.argmax(counts))]
        emitted = busiest[emission_odds(busiest, gate) > gate_u]
        water_ref = float(np.quantile(emitted, 0.95)) if emitted.size else gate
        aprint(f"✓ Gate {gate:.4f} caps the busiest frame at {max(counts):,} points")
        aprint(f"✓ Radius reference condensate: {water_ref:.4f}")

    frames_pos: List[np.ndarray] = []
    frames_col: List[np.ndarray] = []
    frames_rad: List[np.ndarray] = []
    frames_shp: List[np.ndarray] = []
    per_frame_counts: List[int] = []

    with asection(f"Evolving the cumulus over {n_frames} frames"):
        # Phase advanced per frame. Everything else in the model is already a
        # function of phase, so this is the only place the frame count enters
        # the physics — and it enters it exactly once.
        dt = 1.0 / max(n_frames - 1, 1)

        for frame in range(n_frames):
            phase = frame / max(n_frames - 1, 1)

            water = condensate(positions, field, phase, bubbles)
            odds = emission_odds(water, gate)
            keep = odds > gate_u

            if not keep.any() and float(water.max()) > 0.0:
                # An empty frame is a HOLE in the time axis: the dimension
                # advertises a range, and part of that range then has nothing
                # behind it — the compiler says so out loud ("actual data ends
                # at ...") and the viewer shows a blank scene mid-scrub.
                #
                # The life cycle cannot cause this on its own (its amplitude
                # floor is 0.80), but the GATE can: it is calibrated once, from
                # a target point count, and a small enough `--parcels` or point
                # budget leaves a lean frame with nothing clearing it. So relax
                # the gate for this frame alone until something passes. Halving
                # only ever admits MORE parcels, and the recomputed odds keep
                # the fade-in margin positive for whatever it admits.
                relaxed = gate
                for _ in range(40):
                    relaxed *= 0.5
                    keep = emission_odds(water, relaxed) > gate_u
                    if keep.any():
                        odds = emission_odds(water, relaxed)
                        break

            n_kept = int(keep.sum())
            per_frame_counts.append(n_kept)

            if n_kept == 0:
                # Only reachable when the frame holds no condensate at all,
                # anywhere — which the envelope's amplitude floor makes very
                # hard. Report rather than raise: one blank timepoint is worth
                # less than losing the other hundred and nineteen.
                aprint(f"⚠️  frame {frame}: no condensate anywhere")
            else:
                kept_pos = positions[keep]
                kept_water = water[keep]

                # How far past its own threshold a parcel is. Fading the newly
                # condensed ones in over that margin is what keeps points from
                # popping into existence at full size.
                margin = np.clip((odds[keep] - gate_u[keep]) / 0.22, 0.0, 1.0)

                rgb = shade(kept_pos, kept_water, parcel_density)

                size = np.clip(kept_water / water_ref, 0.0, 1.0) ** 0.6

                # RGBA. Under volumetric blending the alpha column IS the
                # parcel's optical depth, so denser air hides more of what is
                # behind it — which is the whole reason the cloud reads as a
                # body rather than a glow. Newly condensed parcels fade in
                # through it instead of switching on at full opacity.
                alpha = np.clip(ALPHA_FLOOR + ALPHA_GAIN * size, 0.0, 1.0) * margin
                colors = np.concatenate(
                    [np.clip(rgb, 0.0, 1.0), alpha[:, None].astype(np.float32)], axis=1
                )
                radii = MIN_RADIUS + size * (MAX_RADIUS - MIN_RADIUS)
                radii *= 0.40 + 0.60 * margin
                radii *= rng.uniform(0.88, 1.12, n_kept)
                radii = np.clip(radii, 0.02, MAX_RADIUS).astype(np.float32)

                # Denser parcels are softer: more multiple scattering, less of
                # a defined edge. Matches the original demo's inverse ramp.
                sharpness = 0.20 + (1.0 - size) * 0.15
                sharpness *= rng.uniform(0.9, 1.1, n_kept)
                sharpness = np.clip(sharpness, 0.15, 0.4).astype(np.float32)

                frames_pos.append(
                    np.column_stack(
                        [kept_pos, np.full(n_kept, float(frame), dtype=np.float32)]
                    ).astype(np.float32)
                )
                frames_col.append(colors)
                frames_rad.append(radii)
                frames_shp.append(sharpness)

            # Advect into the next frame. The thermal pulses early and fades,
            # so the turret shoots up and then merely drifts.
            updraft = UPDRAFT * (
                0.35 + 0.65 * float(np.exp(-(((phase - 0.30) / 0.30) ** 2)))
            )
            advect(positions, updraft, dt)

        aprint(
            f"✓ Points per frame: min {min(per_frame_counts):,}, "
            f"max {max(per_frame_counts):,}, "
            f"mean {int(np.mean(per_frame_counts)):,}"
        )

    if not frames_pos:
        raise ValueError("No condensate was emitted; increase --parcels")

    all_positions = np.concatenate(frames_pos, axis=0)
    all_colors = np.concatenate(frames_col, axis=0)
    all_radii = np.concatenate(frames_rad, axis=0)
    all_sharpness = np.concatenate(frames_shp, axis=0)
    aprint(f"✓ {len(all_positions):,} points across {n_frames} timepoints")

    with asection("Writing to Zarr"):
        # x/y/z displayed, time hidden. Integer frame indices with step 1 is the
        # shape that makes scrubbing exact: the viewer's discrete membership
        # gate is an absolute +/- 0.5, so one frame is selected and only one.
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
                Dimension(
                    "time",
                    unit="frame",
                    display=False,
                    discrete=True,
                    range=(0, n_frames - 1),
                    step=1,
                    description="Cumulus life cycle: growth, maturity, decay",
                ),
            ]
        )

        # Derived, not hardcoded. `current_step` and `animation` are both
        # POSITIONAL — indexed by dimension — so a literal `3` and a
        # four-entry list silently retarget a displayed spatial axis the day
        # anyone inserts a dimension above `time`. Reading the index back out
        # of the dimension list makes that impossible instead of unlikely.
        names = [d.name for d in dims.dimensions]
        time_axis = names.index("time")

        opening_frame = int(round(OPENING_PHASE * (n_frames - 1)))
        populated_frames = np.flatnonzero(np.asarray(per_frame_counts) > 0)
        camera_frame = int(
            populated_frames[np.argmin(np.abs(populated_frames - opening_frame))]
        )
        opening_points = all_positions[all_positions[:, time_axis] == camera_frame, :3]
        viewer_config = ViewerConfig(
            cinematic_mode=True,
            camera=compose_opening_camera(opening_points),
            # Turntable on by default. A cumulus is a 3D body whose whole point
            # is that it looks different from every side — the sunlit flank,
            # the shadowed one, the lean of the top — and a still opening frame
            # shows exactly one of those. RenderingControls.applyZarrDefaults forwards
            # `autoRotate` / `autoRotateSpeed` to the controls manager.
            auto_rotate=True,
            auto_rotate_speed=AUTO_ROTATE_SPEED,
            # Playing on open. `animation` is indexed BY DIMENSION, so the
            # displayed axes take empty entries and only `time` is asked to
            # run. It loops rather than stopping at the end:
            # the life cycle is a cycle, and a scene that halts on its last
            # frame looks like it broke rather than like it finished.
            animation=[
                AnimationConfig(
                    playing=True,
                    target_fps=PLAYBACK_FPS,
                    loop="loop",
                    direction="forward",
                )
                if axis == time_axis
                else AnimationConfig()
                for axis in range(len(names))
            ],
            # Open on the mature cloud rather than on the opening wisps. This
            # is applied before playback starts, so the sequence opens on
            # the mature cloud and runs on from there rather than snapping
            # back to frame 0. The slider panel is opened so the time axis is
            # visible and scrubbable, and so `K` is discoverable for pausing.
            dimensions=DimensionsConfig(
                current_step=[
                    float(camera_frame) if axis == time_axis else 0.0
                    for axis in range(len(names))
                ],
                # NAVIGABLE position, not an absolute dimension index: the
                # viewer resolves this against the non-displayed dimensions
                # only (`getSelectedDimensionIndex`), which is why the number
                # keys start at `1` for the first hidden axis. This scene hides
                # exactly one, so 0 IS `time`. Writing 3 here would name a
                # fourth navigable axis that does not exist.
                #
                # The viewer does not read the field back today, which is the
                # same trap the `animation` block was in — so it is set
                # correctly rather than plausibly.
                selected_dimension=0,
            ),
            ui=UIConfig(show_dimensions=True),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            scene.add_points(
                "EvolvingCloud",
                all_positions,
                colors=all_colors,
                radii=all_radii,
                sharpness=all_sharpness,
                opacity=1.0,
                # A cumulus is optically THICK — you see its surface, not
                # through it. The viewer's `volumetric` mode implements the
                # emission-absorption integral [Max 1995]: each parcel adds
                # its own light AND attenuates everything behind it, composited
                # back to front. `additive` models the opposite regime (an
                # optically thin emissive medium: nothing occludes anything),
                # and combining it with shading derived from occlusion is
                # incoherent: the darkened interior is not hidden behind the
                # lit shell, it shows straight through it, and the cloud reads
                # as a glowing archway with a hole in the middle. Measured on
                # the additive build, the core carried the HIGHEST point
                # density (346 vs 51 per unit area at the rim) and the LOWEST
                # brightness (0.46 vs 0.85) — a hole in the light, not in the
                # geometry. Volumetric composites emission against absorption
                # back-to-front, which is the regime the baked sunlight was
                # computed for.
                blending_mode="volumetric",
                absorption=ABSORPTION,
                intensity=INTENSITY,
                layer=True,
            )

            # Overlay annotations
            scene.add_text(
                "Evolving Cloud",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )
            # One-line context (2026-09-10 review: the title alone said nothing
            # about what is shown). Word-wrapped; no `\n` in a non-hover overlay.
            scene.add_text(
                "A simulated cumulus through its whole life cycle: air parcels "
                "carried by an updraft, condensing where a 4D noise field puts "
                "the water, lit by sunlight baked through the cloud. Time plays "
                "on its own — K pauses, N shows the sliders.",
                position=(0.02, 0.10),
                font_size=0.02,
                anchor="top-left",
                color="rgba(255,255,255,0.45)",
                width=0.58,
                line_height=1.35,
            )
            add_demo_caption(
                scene,
                f"Cumulus life cycle • {n_frames} frames • K pauses",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Written {len(all_positions):,} points to {output_path}")
        aprint(
            "✓ Dataset size: "
            f"~{len(all_positions) * 44 / 1024 / 1024:.1f} MB (uncompressed)"
        )


def main() -> None:
    """Main demo entry point."""
    n_parcels = DEFAULT_PARCELS
    n_frames = DEFAULT_FRAMES
    for arg in sys.argv[1:]:
        if arg.startswith("--parcels="):
            n_parcels = max(1, int(arg.split("=")[1]))
        elif arg.startswith("--frames="):
            n_frames = max(1, int(arg.split("=")[1]))

    aprint("=" * 70)
    aprint("EVOLVING CLOUD DEMO (3D + time)")
    aprint("=" * 70)
    aprint("")
    aprint("A convective cumulus lived through its whole life cycle")
    aprint(f"Parcels: {n_parcels:,}   Frames: {n_frames}")
    aprint("Features:")
    aprint("  - Air parcels advected by an exactly divergence-free flow")
    aprint("  - Convection roll + wind shear + swirl, all analytic")
    aprint("  - 4D fractal noise sampled in material coordinates")
    aprint("  - Flat cloud base at the lifting condensation level")
    aprint("  - Baked off-axis sunlight, skylight, and ground bounce")
    aprint("  - Growth, mature sheared top, then dissipation into wisps")
    aprint("")

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "cloud.luxar.zarr"
        generate_evolving_cloud(output_path, n_parcels=n_parcels, n_frames=n_frames)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_cloud_") as tmpdir:
        output_path = Path(tmpdir) / "cloud.luxar.zarr"

        # Generate the dataset (all code in this file!)
        generate_evolving_cloud(output_path, n_parcels=n_parcels, n_frames=n_frames)

        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("The viewer will open in your browser automatically.")
        aprint("Press Ctrl+C when done to stop and cleanup.")
        aprint("")
        aprint("VIEWING TIPS:")
        aprint("   - Opens already playing and rotating; K pauses, N toggles sliders")
        aprint("   - Home/End jump to the first and last frame; Shift+Up/Down step")
        aprint("   - The scene opens on the mature cloud, mid-life-cycle")
        aprint("   - Watch the flat base: cumulus condense at one altitude")
        aprint("   - The top leans downwind — that is the wind shear")
        aprint("   - Texture stretches and folds instead of boiling: the noise")
        aprint("     is welded to the parcels, so the flow deforms it")
        aprint("   - Bright sunlit crown, blue-grey shadowed underside")
        aprint("   - Late frames shred into rags as dry air is entrained")
        aprint("")

        launch_viewer(output_path)

    # Cleanup happens automatically
    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
