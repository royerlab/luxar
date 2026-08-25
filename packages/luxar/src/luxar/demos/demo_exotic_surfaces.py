#!/usr/bin/env python3
"""Self-Contained Demo: Eighteen Exotic Mathematical Surfaces

A 3x3 grid of famous surfaces, with a hidden **family** axis that switches the
whole wall between two collections:

    family 0   Triply-periodic minimal surfaces — zero mean curvature, dividing
               space into two interpenetrating labyrinths that never touch
    family 1   Algebraic surfaces — zeros of a polynomial, most of them holding
               a RECORD for how singular a surface of their degree can be

Press the number key for the ``family`` dimension and step with ``[`` / ``]`` to
flip between them; the explanatory overlay changes with it. Every surface is
generated from its own closed-form equation in a few seconds — no downloads, no
fitting, no external data.

================================================================================
WHY THESE SURFACES
================================================================================

**Triply-periodic minimal surfaces (TPMS)** minimize area subject to their
boundary conditions, so mean curvature vanishes everywhere: every point is a
saddle, curving up in one direction exactly as much as it curves down in the
perpendicular one. They are not curiosities — block copolymers, lipid membranes,
butterfly-wing photonic crystals and lattice metamaterials all adopt these
shapes, because a TPMS is how nature packs two phases into one volume with the
least interface. Schwarz found the first in 1865; Schoen added the gyroid and
several others in a 1970 NASA report while looking for strong, light structures.

**Algebraic surfaces** are the zero sets of polynomials, and the ones here are
mostly extremal: for each degree there is a maximum number of singular points a
surface can have, and several of these attain it. Barth's sextic has 65 nodes
and no sextic can have more; the Clebsch cubic carries exactly 27 real lines,
the most a cubic can hold; Kummer's quartic has 16. They are the landmarks of
19th- and 20th-century algebraic geometry.

**A caveat that matters.** The TPMS here are *nodal approximations* — short
trigonometric series whose zero set closely resembles the true minimal surface
but does not exactly minimize area. This is the standard way to compute them
(von Schnering & Nesper 1991); do not measure curvature off these points and
report it as the minimal surface's.

================================================================================
WHAT THIS DEMO SHOWS
================================================================================

**Baked ambient occlusion doing real work.** Points are emissive: nothing in the
shader knows neighbouring geometry exists, so a folded surface renders as an even
glow and the folds vanish. Every surface here carries a per-point occlusion
multiplier from ``luxar.shading.bake_ambient_occlusion``, computed with the
surface's own analytic normals and the ``opaque`` occluder — the right model for
a surface, where one wall blocks a direction and a second behind it changes
nothing. It is what makes the channels legible. Set ``AO_STRENGTH = 0.0`` to see
what the geometry looks like without it.

**A hidden categorical dimension.** ``family`` is a two-category axis that no
geometry extends through, so stepping it swaps the entire wall and the overlay
with it. This is the same mechanism a timelapse uses for time.

DATA SOURCE & CITATIONS:
========================

Procedurally generated from published equations. Per-surface attribution appears
in the on-screen overlay for each family; the nodal-approximation technique used
for all nine TPMS is:

    von Schnering, H.G. & Nesper, R. (1991). "Nodal surfaces of Fourier series:
    fundamental invariants of structured matter." Zeitschrift für Physik B 83,
    407-412. DOI: 10.1007/BF01313411

Individual surfaces are credited to: Schwarz (1865, P/D/CLP), Neovius (1883),
Schoen (1970, gyroid/I-WP/F-RD), Lidin & Larsson (1990, lidinoid), Fischer &
Koch (1987, S), Clebsch (1871), Kummer (1864), Cayley (1869), Steiner (1844),
Chmutov (1992), Whitney (1943), Barth (1996), Taubin (1994, heart).

USAGE:
======
    python demo_exotic_surfaces.py [--resolution=N]

Controls:
    - Step the `family` dimension to swap between the two collections
    - Press L for the Layers panel
    - Ctrl+C to stop and cleanup
"""

DEMO_META = {
    "key": "exotic_surfaces",
    "title": "Exotic Mathematical Surfaces",
    "description": (
        "Eighteen famous surfaces on a 3x3 grid — minimal-surface labyrinths and "
        "record-holding algebraic singularities — on a switchable family axis."
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
    "outputs": ["exotic_surfaces"],
    # The surfaces are mathematics, but the trigonometric forms used for all
    # nine TPMS are nodal approximations from a specific paper, and that is a
    # method being reused rather than a fact of nature.
    "citation": {
        "short": "von Schnering & Nesper 1991 (TPMS nodal approximations)",
        "ref": "von Schnering & Nesper 1991",
        "doi": "10.1007/BF01313411",
    },
}

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import add_demo_caption, launch_viewer
from luxar.demos._cinematic_camera import pull_in
from luxar.shading import bake_ambient_occlusion
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Layout and appearance
# =============================================================================

#: Grid is 3x3 per family.
GRID = 3

#: Edge length each surface is normalized into, and the spacing between cell
#: centres. The gap keeps neighbouring surfaces from reading as one object; the
#: occlusion bake is per surface regardless, so they never shade each other.
CELL_SIZE = 1.0
CELL_PITCH = 1.28

#: Shell half-thickness as a multiple of the SAMPLE SPACING rather than an
#: absolute distance. Each surface has its own domain, so a fixed thickness would
#: give a hairline shell on the wide domains and a slab on the narrow ones.
SHELL_SPACINGS = 0.9

#: Render radius as a multiple of the (post-normalization) sample spacing. Below
#: 0.5 the sprites do not touch and the surface renders as a dot screen, whose
#: per-pixel noise drowns the occlusion gradient entirely.
SPRITE_OVERLAP = 0.7

#: Occlusion radius, as a fraction of a cell. Small enough to read the folds of a
#: labyrinth rather than the silhouette of the whole surface.
AO_RADIUS_FRACTION = 0.11

#: All of the ambient treated as direct, leaving no indirect floor. These are
#: synthetic objects with no scientific quantity to protect, and the shape IS the
#: subject, so the term runs at full strength.
AO_STRENGTH = 1.0

#: Every subject in this demo is a surface sampled as points, so one wall fully
#: blocks a direction rather than accumulating as an unbounded medium density.
OCCLUDER = "opaque"

#: Peak additive accumulation to expose for. Additive SUMS along the ray, and a
#: clipped frame flattens exactly the shading this demo depends on.
TARGET_PEAK = 0.7

#: Per-family albedo in LINEAR light. One flat colour per family, so every
#: variation across a surface is the occlusion term.
FAMILY_COLORS = (
    np.array([0.60, 0.74, 0.98], dtype=np.float32),  # minimal — cool
    np.array([0.98, 0.72, 0.46], dtype=np.float32),  # algebraic — warm
)

FAMILY_NAMES = ("Minimal surfaces", "Algebraic surfaces")


# =============================================================================
# Surface definitions
# =============================================================================


@dataclass(frozen=True)
class Surface:
    """One implicit surface: its field, its domain, and why it is interesting."""

    key: str
    title: str
    family: int
    #: The one property worth knowing, shown in the overlay.
    note: str
    #: Discoverer and year.
    credit: str
    #: Half-width of the sampling box, in the surface's own coordinates.
    extent: float
    #: ``f(points) -> values``; the surface is the zero set.
    field: Callable[[np.ndarray], np.ndarray]


def _xyz(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return p[:, 0], p[:, 1], p[:, 2]


# --- Triply-periodic minimal surfaces (nodal approximations, period 2*pi) -----


def _gyroid(p):
    """Gyroid: sin x cos y + sin y cos z + sin z cos x = 0."""

    x, y, z = _xyz(p)
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def _schwarz_p(p):
    """Schwarz primitive: cos x + cos y + cos z = 0."""

    x, y, z = _xyz(p)
    return np.cos(x) + np.cos(y) + np.cos(z)


def _schwarz_d(p):
    """Schwarz diamond: sin sin sin plus the three cos-pair permutations."""

    x, y, z = _xyz(p)
    return (
        np.sin(x) * np.sin(y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(z)
        + np.cos(x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(y) * np.sin(z)
    )


def _neovius(p):
    """Neovius: 3(cos x + cos y + cos z) + 4 cos x cos y cos z = 0."""

    x, y, z = _xyz(p)
    return 3.0 * (np.cos(x) + np.cos(y) + np.cos(z)) + 4.0 * np.cos(x) * np.cos(
        y
    ) * np.cos(z)


def _iwp(p):
    """Schoen I-WP: 2*sum(cos cos) - sum(cos 2t) = 0."""

    x, y, z = _xyz(p)
    return 2.0 * (
        np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x)
    ) - (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))


def _frd(p):
    """Schoen F-RD: 4 cos x cos y cos z - sum(cos 2t cos 2u) = 0."""

    x, y, z = _xyz(p)
    return 4.0 * np.cos(x) * np.cos(y) * np.cos(z) - (
        np.cos(2 * x) * np.cos(2 * y)
        + np.cos(2 * y) * np.cos(2 * z)
        + np.cos(2 * z) * np.cos(2 * x)
    )


def _lidinoid(p):
    """Lidinoid: the gyroid's hexagonal relative, with a 0.15 level shift."""

    x, y, z = _xyz(p)
    return (
        0.5
        * (
            np.sin(2 * x) * np.cos(y) * np.sin(z)
            + np.sin(2 * y) * np.cos(z) * np.sin(x)
            + np.sin(2 * z) * np.cos(x) * np.sin(y)
        )
        - 0.5
        * (
            np.cos(2 * x) * np.cos(2 * y)
            + np.cos(2 * y) * np.cos(2 * z)
            + np.cos(2 * z) * np.cos(2 * x)
        )
        + 0.15
    )


def _split_p(p):
    """Split P: Schwarz P doubled into two sheets."""

    x, y, z = _xyz(p)
    return (
        1.1
        * (
            np.sin(2 * x) * np.sin(z) * np.cos(y)
            + np.sin(2 * y) * np.sin(x) * np.cos(z)
            + np.sin(2 * z) * np.sin(y) * np.cos(x)
        )
        - 0.2
        * (
            np.cos(2 * x) * np.cos(2 * y)
            + np.cos(2 * y) * np.cos(2 * z)
            + np.cos(2 * z) * np.cos(2 * x)
        )
        - 0.4 * (np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z))
    )


def _fischer_koch_s(p):
    """Fischer-Koch S: cos 2x sin y cos z + two cyclic permutations."""

    x, y, z = _xyz(p)
    return (
        np.cos(2 * x) * np.sin(y) * np.cos(z)
        + np.cos(x) * np.cos(2 * y) * np.sin(z)
        + np.sin(x) * np.cos(y) * np.cos(2 * z)
    )


# --- Algebraic surfaces -------------------------------------------------------

_PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _barth_sextic(p):
    """Barth sextic: 4*prod(phi^2 a^2 - b^2) - (1+2 phi)(r^2-1)^2 = 0."""

    x, y, z = _xyz(p)
    r2 = x * x + y * y + z * z
    return (
        4.0
        * (_PHI**2 * x * x - y * y)
        * (_PHI**2 * y * y - z * z)
        * (_PHI**2 * z * z - x * x)
        - (1.0 + 2.0 * _PHI) * (r2 - 1.0) ** 2
    )


def _clebsch(p):
    """Clebsch diagonal cubic, in its symmetric three-variable form."""

    x, y, z = _xyz(p)
    return (
        81.0 * (x**3 + y**3 + z**3)
        - 189.0
        * (x * x * y + x * x * z + y * y * x + y * y * z + z * z * x + z * z * y)
        + 54.0 * x * y * z
        + 126.0 * (x * y + y * z + z * x)
        - 9.0 * (x * x + y * y + z * z)
        - 9.0 * (x + y + z)
        + 1.0
    )


def _kummer(p):
    """Kummer quartic: (r^2 - mu^2)^2 - lambda * prod(four planes) = 0."""

    x, y, z = _xyz(p)
    mu2 = 1.3**2
    lam = (3.0 * mu2 - 1.0) / (3.0 - mu2)
    root2 = np.sqrt(2.0)
    return (x * x + y * y + z * z - mu2) ** 2 - lam * (
        (1.0 - z - root2 * x)
        * (1.0 - z + root2 * x)
        * (1.0 + z + root2 * y)
        * (1.0 + z - root2 * y)
    )


def _cayley(p):
    """Cayley cubic: 4(x^2 + y^2 + z^2) + 16xyz - 1 = 0."""

    x, y, z = _xyz(p)
    return 4.0 * (x * x + y * y + z * z) + 16.0 * x * y * z - 1.0


def _roman(p):
    """Steiner Roman surface: x^2y^2 + y^2z^2 + z^2x^2 + xyz = 0."""

    x, y, z = _xyz(p)
    return x * x * y * y + y * y * z * z + z * z * x * x + x * y * z


def _chmutov(p):
    """Degree-6 Chmutov surface: T6(x) + T6(y) + T6(z) = 0."""

    def t6(t):
        t2 = t * t
        return 32.0 * t2 * t2 * t2 - 48.0 * t2 * t2 + 18.0 * t2 - 1.0

    x, y, z = _xyz(p)
    return t6(x) + t6(y) + t6(z)


def _whitney(p):
    """Whitney umbrella: x^2 - y^2 z = 0; the canonical pinch point."""

    x, y, z = _xyz(p)
    return x * x - y * y * z


def _heart(p):
    """Taubin's heart sextic, oriented to stand up in a z-vertical scene."""

    x, y, z = _xyz(p)
    # Taubin's heart, with the conventional y/z swap so it stands up in a
    # z-is-vertical scene rather than lying on its side.
    a = x * x + (9.0 / 4.0) * y * y + z * z - 1.0
    return a**3 - x * x * z**3 - (9.0 / 80.0) * y * y * z**3


def _tanglecube(p):
    """Tanglecube: x^4 - 5x^2 + y^4 - 5y^2 + z^4 - 5z^2 + 11.8 = 0."""

    x, y, z = _xyz(p)
    return x**4 - 5.0 * x * x + y**4 - 5.0 * y * y + z**4 - 5.0 * z * z + 11.8


#: Grid order is reading order: left to right, top to bottom.
SURFACES: List[Surface] = [
    # --- family 0: triply-periodic minimal surfaces ---
    Surface(
        "gyroid",
        "Gyroid",
        0,
        "No straight lines and no mirror symmetries — the only TPMS of its "
        "kind; found in butterfly-wing photonic crystals.",
        "Schoen 1970",
        np.pi,
        _gyroid,
    ),
    Surface(
        "schwarz_p",
        "Schwarz P",
        0,
        "The first TPMS ever described. Cubic symmetry, with open channels "
        "along all three axes.",
        "Schwarz 1865",
        np.pi,
        _schwarz_p,
    ),
    Surface(
        "schwarz_d",
        "Schwarz D (Diamond)",
        0,
        "Two interwoven diamond lattices; the shape a lipid bilayer takes "
        "in the cubic phase.",
        "Schwarz 1865",
        np.pi,
        _schwarz_d,
    ),
    Surface(
        "neovius",
        "Neovius",
        0,
        "Higher genus than Schwarz P, so a far more ornate cell for the "
        "same cubic symmetry.",
        "Neovius 1883",
        np.pi,
        _neovius,
    ),
    Surface(
        "iwp",
        "Schoen I-WP",
        0,
        "'Wrapped package' — a cage of struts around isolated pockets, so "
        "its two labyrinths are NOT congruent.",
        "Schoen 1970",
        np.pi,
        _iwp,
    ),
    Surface(
        "frd",
        "Schoen F-RD",
        0,
        "Face-centred rhombic dodecahedral: the densest packing symmetry "
        "of the family.",
        "Schoen 1970",
        np.pi,
        _frd,
    ),
    Surface(
        "lidinoid",
        "Lidinoid",
        0,
        "A gyroid relative found 20 years later, with hexagonal rather "
        "than cubic symmetry.",
        "Lidin & Larsson 1990",
        np.pi,
        _lidinoid,
    ),
    Surface(
        "split_p",
        "Split P",
        0,
        "Schwarz P split into a doubled sheet — a distinct surface, not a "
        "deformation of it.",
        "Schoen 1970",
        np.pi,
        _split_p,
    ),
    Surface(
        "fischer_koch_s",
        "Fischer-Koch S",
        0,
        "One of the balanced surfaces found by systematic enumeration of "
        "space groups rather than by construction.",
        "Fischer & Koch 1987",
        np.pi,
        _fischer_koch_s,
    ),
    # --- family 1: algebraic surfaces ---
    Surface(
        "barth",
        "Barth Sextic",
        1,
        "65 ordinary double points — the MAXIMUM possible for a degree-6 "
        "surface. Icosahedral symmetry.",
        "Barth 1996",
        1.9,
        _barth_sextic,
    ),
    Surface(
        "clebsch",
        "Clebsch Diagonal Cubic",
        1,
        "Contains exactly 27 real straight lines — every smooth cubic has "
        "27, and this is the one where all of them are real.",
        "Clebsch 1871",
        2.6,
        _clebsch,
    ),
    Surface(
        "kummer",
        "Kummer Quartic",
        1,
        "16 nodal singularities, the maximum for a quartic; self-dual, and "
        "the wave surface of 19th-century optics.",
        "Kummer 1864",
        2.4,
        _kummer,
    ),
    Surface(
        "cayley",
        "Cayley Cubic",
        1,
        "4 nodes — the most a cubic surface can have. Its 27 lines "
        "degenerate down to just 9.",
        "Cayley 1869",
        1.3,
        _cayley,
    ),
    Surface(
        "roman",
        "Roman (Steiner) Surface",
        1,
        "A one-sided immersion of the real projective plane, with three "
        "lines of self-intersection meeting at a triple point.",
        "Steiner 1844",
        1.1,
        _roman,
    ),
    Surface(
        "chmutov",
        "Chmutov Sextic",
        1,
        "Built from Chebyshev polynomials, which is how its singularities "
        "are placed by construction rather than found.",
        "Chmutov 1992",
        1.35,
        _chmutov,
    ),
    Surface(
        "whitney",
        "Whitney Umbrella",
        1,
        "The canonical pinch point: a self-intersection line that simply "
        "stops, which no smooth surface can do.",
        "Whitney 1943",
        1.6,
        _whitney,
    ),
    Surface(
        "heart",
        "Heart Surface",
        1,
        "A sextic with no mathematical record to its name, included "
        "because a closed-form heart is a good joke.",
        "Taubin 1994",
        1.5,
        _heart,
    ),
    Surface(
        "tanglecube",
        "Tanglecube",
        1,
        "A quartic whose four separate sheets link through one another "
        "without ever touching.",
        "Banchoff",
        3.1,
        _tanglecube,
    ),
]


# =============================================================================
# Sampling
# =============================================================================


def _central_difference(
    field: Callable[[np.ndarray], np.ndarray], points: np.ndarray, epsilon: float
) -> np.ndarray:
    """Unnormalized gradient of ``field``, by central differences.

    Kept separate from :func:`implicit_normals` because the shell test needs the
    gradient's MAGNITUDE, which normalizing throws away.
    """
    gradient = np.empty_like(points, dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = epsilon
        gradient[:, axis] = field(points + offset) - field(points - offset)
    return gradient


def implicit_normals(
    field: Callable[[np.ndarray], np.ndarray], points: np.ndarray, epsilon: float
) -> np.ndarray:
    """Unit surface normals from central differences of the implicit field.

    Numerical rather than analytic, and deliberately so: hand-deriving eighteen
    gradients — several of them degree-6 polynomials — would be eighteen chances
    to introduce a silent error, where one numerical routine can be validated
    once against a surface whose gradient IS known in closed form (see the
    gyroid check in the tests). Accuracy is ~1e-8 for these smooth fields, far
    below what the occlusion bake can resolve.

    Args:
        field: ``f(points) -> values``.
        points: ``(N, 3)`` positions.
        epsilon: Central-difference step.

    Returns:
        ``(N, 3)`` unit normals; zero rows where the gradient vanishes.
    """
    gradients = _central_difference(field, points, epsilon)
    lengths = np.linalg.norm(gradients, axis=1, keepdims=True)
    return np.divide(
        gradients, lengths, out=np.zeros_like(gradients), where=lengths > 1e-12
    )


def sample_surface(
    surface: Surface, resolution: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample points near a surface's zero set, normalized into one grid cell.

    Every surface is scaled to the same cell, so a wide-domain surface and a
    narrow-domain one read at comparable size in the grid.

    Args:
        surface: The surface to sample.
        resolution: Grid samples per axis.

    Returns:
        ``(positions, normals, spacing)`` — positions centred on the origin and
        fitting within :data:`CELL_SIZE`, unit normals, and the post-scaling
        sample spacing (what the render radius must span).
    """
    coords = np.linspace(-surface.extent, surface.extent, resolution)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    samples = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    spacing = float(coords[1] - coords[0])
    # Jitter off the lattice: a regular grid through a smooth surface produces
    # visible moire terracing.
    rng = np.random.default_rng(abs(hash(surface.key)) % (2**32))
    samples += rng.uniform(-0.3 * spacing, 0.3 * spacing, samples.shape)

    values = surface.field(samples)
    # |f| / |grad f| is the first-order distance to the zero set, so thresholding
    # it keeps a shell of even thickness instead of one that bulges wherever the
    # polynomial happens to be flat. Needs the gradient MAGNITUDE, so it is taken
    # from the raw differences rather than from `implicit_normals`, which has
    # already divided the magnitude out.
    gradient = _central_difference(surface.field, samples, 0.5 * spacing)
    magnitude = np.linalg.norm(gradient, axis=1) / spacing
    distance = np.abs(values) / np.maximum(magnitude, 1e-12)

    near = distance < SHELL_SPACINGS * spacing
    if not np.any(near):
        return (
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.float64),
            spacing,
        )

    positions = samples[near]
    normals = implicit_normals(surface.field, positions, 0.5 * spacing)

    # Normalize into the cell: centre, then scale by the longest axis so the
    # aspect ratio is preserved.
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    centre = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    scale = CELL_SIZE / max(span, 1e-9)
    return (
        ((positions - centre) * scale).astype(np.float32),
        normals,
        spacing * scale,
    )


def cell_offset(index: int) -> Tuple[float, float]:
    """(x, y) centre of grid cell ``index``, in reading order."""
    column, row = index % GRID, index // GRID
    origin = -0.5 * (GRID - 1) * CELL_PITCH
    # Rows run downward in reading order, so the first surface is top-left.
    return origin + column * CELL_PITCH, -(origin + row * CELL_PITCH)


def auto_exposure(positions: np.ndarray, radius: float) -> Tuple[float, int]:
    """Node gain holding the deepest additive sightline under white.

    Derived, not hardcoded: the deepest column grows with ``--resolution``, so a
    fixed gain either clips at one end of the range or is needlessly dim at the
    other. Approximate — it counts points per column, where the true per-point
    contribution also depends on opacity and the sprite profile — hence a
    :data:`TARGET_PEAK` well below 1.0.

    Returns:
        ``(intensity, deepest_column)``.
    """
    cell = 2.0 * radius
    keys = np.floor(positions[:, :2] / cell).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    deepest = max(int(counts.max()), 1)
    brightest = float(max(c.max() for c in FAMILY_COLORS))
    return TARGET_PEAK / (deepest * brightest), deepest


# =============================================================================
# Scene
# =============================================================================

_OVERLAY = {
    0: (
        "MINIMAL SURFACES  ·  triply-periodic, zero mean curvature\n"
        "Every point is a saddle: curving up in one direction exactly as much "
        "as it curves down in the perpendicular one. Each divides space into "
        "two interpenetrating labyrinths that never touch — which is how block "
        "copolymers, lipid membranes and butterfly-wing photonic crystals pack "
        "two phases with the least interface.\n"
        "Shown as nodal approximations (von Schnering & Nesper 1991), not exact "
        "minimal surfaces."
    ),
    1: (
        "ALGEBRAIC SURFACES  ·  zero sets of polynomials\n"
        "For each degree there is a maximum number of singular points a surface "
        "can have, and several of these attain it: Barth's sextic has 65 nodes, "
        "Kummer's quartic 16, Cayley's cubic 4. The Clebsch cubic instead holds "
        "a record for lines — all 27 of them real.\n"
        "These are the landmarks of 19th- and 20th-century algebraic geometry."
    ),
}


def _family_roster(family: int) -> str:
    """The nine surfaces of a family, in grid order, with credits."""
    lines = []
    for index, surface in enumerate(s for s in SURFACES if s.family == family):
        column, row = index % GRID, index // GRID
        where = "  ".join(
            [["left", "centre", "right"][column], ["top", "middle", "bottom"][row]]
        )
        lines.append(f"{surface.title} ({surface.credit}) — {where}\n   {surface.note}")
    return "\n".join(lines)


def generate_exotic_surfaces(output_path: Path, resolution: int = 112) -> int:
    """Generate the two-family surface grid.

    Args:
        output_path: Where to write the zarr store.
        resolution: Grid samples per axis, per surface.

    Returns:
        Total points written.
    """
    families: dict = {0: [], 1: []}

    with asection(f"Sampling {len(SURFACES)} surfaces at {resolution}³ each"):
        for surface in SURFACES:
            index = sum(
                1
                for s in SURFACES[: SURFACES.index(surface)]
                if s.family == surface.family
            )
            local, normals, spacing = sample_surface(surface, resolution)
            if len(local) == 0:
                aprint(f"⚠️  {surface.title}: no surface points — check its extent")
                continue

            ao = bake_ambient_occlusion(
                local,
                normals=normals,
                occluder=OCCLUDER,
                radius=AO_RADIUS_FRACTION * CELL_SIZE,
                strength=AO_STRENGTH,
            )
            offset_x, offset_y = cell_offset(index)
            placed = local.copy()
            placed[:, 0] += offset_x
            placed[:, 1] += offset_y

            # Radius is stored PER SURFACE, not averaged across the family: each
            # surface has its own normalization scale, so one mean radius would
            # leave the finely-scaled ones stippling and smear the coarse ones.
            families[surface.family].append(
                (placed, ao, np.full(len(placed), SPRITE_OVERLAP * spacing, np.float32))
            )
            aprint(
                f"  {surface.title:<28} {len(local):>7,} pts  "
                f"AO [{ao.min():.3f}, {ao.max():.3f}] contrast "
                f"{ao.std() / max(ao.mean(), 1e-9):.3f}"
            )

    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="units", display=True),
                Dimension("y", unit="units", display=True),
                Dimension("z", unit="units", display=True),
                # Categorical, so it is DISCRETE with exactly two stops and an
                # integer step. A continuous hidden axis would give a slider with
                # a thousand positions, almost all of which show an empty scene.
                Dimension(
                    "family",
                    display=False,
                    categories=list(FAMILY_NAMES),
                    description="Which collection of surfaces is on the wall",
                ),
            ]
        )

        total = 0
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                citation=DEMO_META["citation"],
                viewer_config=ViewerConfig(
                    cinematic_mode=True,
                    tone_mapping="ACES",
                    # The grid is a WALL in x-y, so it has to be viewed roughly
                    # face-on or the nine cells overlap into one mass. `pull_in`
                    # restates the framing for the preset's wider 35 mm lens,
                    # which the scene takes whole by pinning no `fov` of its own.
                    camera=CameraConfig(
                        position=pull_in((0.0, -0.6, 6.4)),
                        target=(0.0, 0.0, 0.0),
                        up=(0.0, 1.0, 0.0),
                    ),
                ),
            )

            for family, entries in families.items():
                if not entries:
                    continue
                positions = np.vstack([e[0] for e in entries])
                occlusion = np.concatenate([e[1] for e in entries])
                radii = np.concatenate([e[2] for e in entries])
                intensity, deepest = auto_exposure(positions, float(radii.max()))

                # The hidden axis: every point of a family sits at that family's
                # coordinate, and nothing extends through the axis, so stepping
                # it swaps the whole wall.
                nd = np.column_stack(
                    [positions, np.full(len(positions), family, dtype=np.float32)]
                ).astype(np.float32)

                colors = (FAMILY_COLORS[family][None, :] * occlusion[:, None]).astype(
                    np.float32
                )

                scene.add_points(
                    FAMILY_NAMES[family],
                    nd,
                    colors=np.ascontiguousarray(colors),
                    radii=radii,
                    opacity=0.9,
                    blending_mode="additive",
                    intensity=intensity,
                    layer=True,
                )
                total += len(nd)
                aprint(
                    f"✓ {FAMILY_NAMES[family]}: {len(nd):,} points, "
                    f"deepest column {deepest} -> intensity {intensity:.4f}"
                )

            for family in (0, 1):
                # `visible_range` ties each overlay to its own slice of the
                # hidden axis, so the explanation swaps with the geometry.
                scene.add_text(
                    _OVERLAY[family],
                    position=(0.02, 0.02),
                    font_size=0.019,
                    width=0.42,
                    anchor="top-left",
                    color="rgba(255,255,255,0.82)",
                    line_height=1.45,
                    visible_range={"family": family},
                    transition="fade",
                )
                scene.add_text(
                    _family_roster(family),
                    position=(0.98, 0.03),
                    font_size=0.0135,
                    width=0.33,
                    anchor="top-right",
                    text_align="right",
                    color="rgba(255,255,255,0.60)",
                    line_height=1.4,
                    visible_range={"family": family},
                    transition="fade",
                )

            add_demo_caption(
                scene,
                "18 surfaces • step `family` to switch collection",
                DEMO_META.get("citation"),
            )

        aprint(f"✓ Written to {output_path}")

    return total


def main() -> None:
    """Main demo entry point."""
    resolution = 112
    for arg in sys.argv[1:]:
        if arg.startswith("--resolution="):
            resolution = int(arg.split("=")[1])

    aprint("=" * 70)
    aprint("EXOTIC MATHEMATICAL SURFACES")
    aprint("=" * 70)
    aprint("")
    aprint("Two collections of nine, on a switchable `family` axis:")
    aprint("  family 0  triply-periodic minimal surfaces (zero mean curvature)")
    aprint("  family 1  algebraic surfaces (record-holding singularities)")
    aprint("")
    aprint(f"Resolution: {resolution}³ per surface, 18 surfaces")
    aprint("")

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "exotic_surfaces.luxar.zarr"
        if generate_exotic_surfaces(output_path, resolution) == 0:
            aprint("\n❌ No points generated")
            return
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_exotic_surfaces_") as tmpdir:
        output_path = Path(tmpdir) / "exotic_surfaces.luxar.zarr"
        if generate_exotic_surfaces(output_path, resolution) == 0:
            aprint("\n❌ No points generated")
            return

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("  • Step the `family` dimension to swap the whole wall")
        aprint("  • The overlay text changes with it")
        aprint("  • Zoom into a single cell — the labyrinths are worth it")
        aprint("  • Press L for the Layers panel")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete - temporary files removed")


if __name__ == "__main__":
    main()
