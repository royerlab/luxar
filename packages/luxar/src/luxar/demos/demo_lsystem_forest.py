#!/usr/bin/env python3
"""Self-Contained Demo: A Year in a Growing L-System Forest.

The flagship synthetic scene: a terrain-planted procedural forest you can
scrub through time on TWO axes — a ``growth`` dimension (seedling to gnarled
canopy, staggered per tree so maturity rolls across the field in waves) and a
``season`` dimension (spring blossom, summer green, autumn fire, winter
frost). Press play on either axis and the viewer animates a growth time-lapse
or a cycling year. All four Luxar geometry types share the frame:

- **Mesh** — an fBm heightfield terrain, shaded by the viewer's offset key,
  with per-season vertex colours (snow in winter). The forest floor has
  actual depth instead of a flat line grid.
- **Lines** — the trees: indexed line networks (joints and branch points
  share vertex indices, so thick trunks render as smooth tubes), one merged
  node per species so the Layers panel gets one toggleable row per family.
- **GSplats** — the foliage: anisotropic Gaussian splats oriented along
  their parent branch, ``volumetric`` blending so canopies read as soft
  occluding clouds rather than additive fuzz. Spring adds blossom splats;
  autumn thins and turns to fire; winter strips the deciduous species bare
  while the conifers keep frost-dusted needles.
- **Points** — the accents: summer fireflies, winter frost sparkle on the
  snow, spring petals drifting under the blossom trees — and a very light
  multiscale cloud cover over the whole forest whose coverage follows the
  season (summer wisps, a near-closed winter deck).

L-system grammar (turtle commands):
    F/G: move forward drawing a segment (G = apical leader, drawn longer)
    +/-: yaw around the up axis      ^/&: pitch around the left axis
    \\//: roll around the heading    [/]: push / pop state (branch)

The grammars go beyond angle jitter with the three ingredients from *The
Algorithmic Beauty of Plants* that separate "fractal twig" from
"recognisable tree":

- **Stochastic productions** — a symbol can carry several weighted
  productions; every derivation samples its own expansion, so no two trees
  of a species repeat.
- **Tropism** (ABOP §1.7) — after each drawn segment the heading is bent
  toward a tropism vector by one Rodrigues rotation: gravity droop for the
  willow and the palm fronds, upward phototropism for the columnar poplar.
  Applied only at branching depth >= 1, so trunks keep their negative
  gravitropism and stay straight.
- **Parametric apical dominance** — the ``G`` symbol draws a longer leader
  internode, giving the conifer its monopodial trunk (Honda's model): the
  whorls lower on the trunk are older, so they have expanded for more
  iterations and are naturally longer. No explicit taper rule needed.

Growth is modelled the botanically honest way: each of the six stages is a
genuine re-derivation of every tree at an increasing iteration count plus
maturity-scaled length/width — development IS successive derivation steps.
Saplings are short and thin, not masked-out subsets of the adult. Because
segment counts grow roughly geometrically per iteration, the sum over all
six stages costs only ~2.3x the final stage alone.

Usage:
    luxar demo run forest [-- --trees=800] [-- --iterations=5]

Controls:
    - Keys 1/2 select the season/growth dimension, [ and ] step it,
      and the dimension play button animates it.
    - Hover any trunk for the tree's species, instance, season and stage.
    - Ctrl+C to stop and cleanup.
"""

from __future__ import annotations

DEMO_META = {
    "key": "forest",
    "title": "L-System Forest",
    "description": (
        "A growing, seasonal L-system forest: shaded terrain mesh, per-species "
        "indexed-line trees, volumetric gsplat foliage and point accents, "
        "scrubbable through growth and season dimensions."
    ),
    "category": "synthetic",
    "geometry": "mixed",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["forest"],
    "outputs": ["forest"],
    # Procedurally generated: no external dataset, nothing to credit.
    "citation": None,
}

import html
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, DimensionsConfig, ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cache_computed,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
)
from luxar.demos._cinematic_camera import CINEMATIC_FOV_DEG
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# The two navigable dimensions
# =============================================================================

#: Season slots, in calendar order. Index = coordinate on the `season` dim.
SEASONS = ("Spring", "Summer", "Autumn", "Winter")
SPRING, SUMMER, AUTUMN, WINTER = range(4)

#: Growth stages. Index = coordinate on the `growth` dim.
GROWTH_STAGES = ("Seedling", "Sprout", "Sapling", "Young", "Mature", "Ancient")

#: Iteration offset of each stage relative to a species' final iteration
#: count. Adjacent stages may share a derivation depth (differentiated by
#: the maturity scaling below); the last two both derive at FULL depth so
#: the late stages read as densely crowned. Staggered trees reach the final
#: stage too — see :func:`_effective_stage`, which delays development
#: without truncating the endpoint. With geometric segment growth the
#: stage SUM is ~2.3x the final stage.
STAGE_ITER_OFFSETS = (-3, -3, -2, -1, 0, 0)

#: Maturity in [0, 1] per stage: scales segment lengths and widths so a
#: seedling is short and thin rather than a shrunken adult silhouette.
STAGE_MATURITY = (0.16, 0.30, 0.46, 0.64, 0.82, 1.0)

N_STAGES = len(GROWTH_STAGES)

#: Sigma written on the two stacked (season/growth) axes of the 5D foliage
#: Cholesky factors. It is NOT what confines a splat to one slot: both stacked
#: dims are CATEGORICAL, hence discrete, so the viewer gates them on a half-step
#: membership test against the splat CENTRE and never looks at the covariance
#: there. The value's only job is to keep the packed lower-triangular factor
#: non-singular. Don't copy this onto a CONTINUOUS stacked axis, where the
#: attenuation IS Gaussian: centres are stored as per-column uint16, so a
#: coordinate lands up to ~1.5e-5 off its authored value, which against a 1e-6
#: sigma is ~15 sigma — the whole node would slice away to nothing.
STACKED_AXIS_SIGMA = 1e-6

#: Bump when the generation logic changes in a way that must invalidate
#: cached bundles (also passed as `version=` to :func:`cache_computed`).
SCENE_VERSION = 11


def _maturity_length_scale(m: float) -> float:
    """Segment-length multiplier for maturity ``m`` (height keeps growing)."""
    return 0.5 + 0.5 * m


def _maturity_width_scale(m: float) -> float:
    """Trunk-width multiplier for maturity ``m`` (girth grows more than height)."""
    return 0.3 + 0.7 * m


# =============================================================================
# L-System grammar: stochastic productions, tropism, apical dominance
# =============================================================================

#: Rotation commands: char -> (frame index of the rotation axis, sign).
#: Frame indices: 0 = heading, 1 = left, 2 = up.
_ROTATIONS = {
    "+": (2, 1.0),
    "-": (2, -1.0),
    "^": (1, 1.0),
    "&": (1, -1.0),
    "\\": (0, 1.0),
    "/": (0, -1.0),
}

Vec3 = tuple[float, float, float]

#: A production is a plain string, or a list of (probability, production)
#: alternatives sampled per occurrence per derivation (stochastic L-system).
Production = Union[str, Sequence[tuple[float, str]]]


def _rotate_vec(v: Vec3, k: Vec3, c: float, s: float) -> Vec3:
    """Rodrigues rotation of ``v`` about unit axis ``k`` (c/s = cos/sin)."""
    kx, ky, kz = k
    vx, vy, vz = v
    dot = kx * vx + ky * vy + kz * vz
    cx = ky * vz - kz * vy
    cy = kz * vx - kx * vz
    cz = kx * vy - ky * vx
    oc = (1.0 - c) * dot
    return (
        vx * c + cx * s + kx * oc,
        vy * c + cy * s + ky * oc,
        vz * c + cz * s + kz * oc,
    )


_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """SplitMix64 finalizer: avalanche an integer into 64 well-mixed bits."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _branch_jitter(seed: int, branch_hash: int, ordinal: int) -> float:
    """Uniform [0, 1) as a PURE FUNCTION of (tree seed, branch path, ordinal).

    Angle jitter must NOT come from a sequential rng stream: a growth stage
    is a re-derivation at a deeper iteration count, and a stream's draw
    positions shift with the string length, re-rolling every branch angle
    between stages (visible popping in the growth time-lapse). Keyed on the
    branch's bracket PATH and the rotation's ordinal within that branch, so a
    branch that exists at two stages bends identically at both.

    The ordinal is only depth-stable when a production appends its recursion
    AFTER the commands already in the string — true of every shipped species
    except the palm, whose axiom is ``TC``: ``T -> F/T`` inserts one more
    top-level roll AHEAD of the crown per derivation step, so the crown's
    ordinals all shift by one and its fronds re-roll. Measured, that costs
    the palm a few degrees of crown jitter per stage, on top of the ~26
    deg/stage roll its trunk grammar intends anyway (phyllotaxis), so it is
    a wash visually — but do not read the property as universal. The scope
    is pinned by ``test_lsystem_growth_stages_keep_existing_branch_orientations``.
    """
    return _mix64(seed ^ _mix64(branch_hash ^ _mix64(ordinal))) / 2.0**64


def _rotate_frame_vec(v: Vec3, k: Vec3, c: float, s: float) -> Vec3:
    """:func:`_rotate_vec`, then re-normalize — for turtle FRAME vectors.

    Rodrigues assumes a unit axis, and the axis here is itself a frame
    vector. Without per-step normalization the ~1e-16 float error per
    rotation compounds GEOMETRICALLY (a slightly short axis shrinks the
    other two vectors, which later serve as slightly shorter axes...):
    measured, a random command string collapses the frame to zero within
    ~250 rotations. Pinning the norm after every step keeps the error
    additive and negligible for any realistic derivation.
    """
    x, y, z = _rotate_vec(v, k, c, s)
    inv = 1.0 / math.sqrt(x * x + y * y + z * z)
    return (x * inv, y * inv, z * inv)


def _bend_toward(frame: list[Vec3], tropism: Vec3, strength: float) -> None:
    """Bend the turtle frame toward ``tropism`` (ABOP §1.7), in place.

    The rotation axis is ``heading x tropism`` and the angle is proportional
    to its magnitude, so a heading already aligned with the tropism vector is
    a fixed point and the bend fades out smoothly as branches align.
    """
    hx, hy, hz = frame[0]
    tx, ty, tz = tropism
    ax = hy * tz - hz * ty
    ay = hz * tx - hx * tz
    az = hx * ty - hy * tx
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-9:
        return
    axis = (ax / norm, ay / norm, az / norm)
    angle = strength * norm
    c, s = math.cos(angle), math.sin(angle)
    for j in range(3):
        frame[j] = _rotate_frame_vec(frame[j], axis, c, s)


@dataclass
class LSystem:
    """L-System grammar with stochastic expansion, tropism and a leader symbol."""

    axiom: str
    rules: dict[str, Production]
    angle: float = np.radians(25)
    length: float = 1.0
    width: float = 0.15
    width_decay: float = 0.7
    length_decay: float = 0.9
    randomness: float = 0.15
    #: Unit tropism vector T; after each drawn segment at depth >= 1 the
    #: frame is bent toward T (gravity droop, phototropism, ...).
    tropism: Optional[Vec3] = None
    tropism_strength: float = 0.0
    #: Length multiplier of the ``G`` (apical leader) symbol relative to ``F``.
    leader_scale: float = 1.4

    def expand(self, iterations: int, rng: np.random.Generator) -> str:
        """Expand the axiom, sampling stochastic productions per occurrence."""
        current = self.axiom
        for _ in range(iterations):
            parts: list[str] = []
            for char in current:
                rule = self.rules.get(char)
                if rule is None:
                    parts.append(char)
                elif isinstance(rule, str):
                    parts.append(rule)
                else:
                    r = rng.random()
                    acc = 0.0
                    chosen = rule[-1][1]
                    for probability, production in rule:
                        acc += probability
                        if r < acc:
                            chosen = production
                            break
                    parts.append(chosen)
            current = "".join(parts)
        return current

    def interpret(
        self, string: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpret an expanded string as an indexed 3D line network.

        Vertices are deduplicated exactly during interpretation: the turtle
        carries the index of the vertex at its current position (pushed and
        popped with the branching stack), so consecutive segments and
        branches emanating from a branch point share the same vertex index
        with no position hashing. Shared indices let the viewer suppress
        joint caps, so thick trunks render as smooth tubes instead of bead
        chains.

        Angle jitter is drawn from :func:`_branch_jitter` — a pure function
        of (seed, bracket path, per-branch ordinal), NOT a sequential rng —
        so a branch that exists at two growth stages bends identically at
        both (see the growth-stage stability test).

        The hot loop runs on plain Python floats (tuples + ``math``) rather
        than small numpy arrays — the strings run to tens of thousands of
        commands per tree and small-array overhead dominates otherwise.

        Returns:
            Tuple of (vertices, edges, edge_depths, vertex_depths):
            vertices: (V, 3) float32 unique vertex positions
            edges: (E, 2) uint32 vertex-index pairs, one per segment
            edge_depths: (E,) int32 branching depth of each segment
            vertex_depths: (V,) int32 depth of the segment that first
                introduced each vertex (the parent side at joints)
        """
        frame: list[Vec3] = [(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
        pos: Vec3 = (0.0, 0.0, 0.0)
        depth = 0
        vertex_index = -1
        length = self.length
        tropism = self.tropism
        strength = self.tropism_strength
        branch_hash = _mix64(seed)
        rotation_ordinal = 0
        child_count = 0

        stack: list[
            tuple[Vec3, tuple[Vec3, Vec3, Vec3], int, int, float, int, int, int]
        ] = []
        vertices: list[Vec3] = []
        vertex_depths: list[int] = []
        edges: list[tuple[int, int]] = []
        edge_depths: list[int] = []

        for char in string:
            if char == "F" or char == "G":
                if tropism is not None and depth > 0 and strength > 0.0:
                    _bend_toward(frame, tropism, strength)
                if vertex_index < 0:
                    vertices.append(pos)
                    vertex_depths.append(depth)
                    vertex_index = len(vertices) - 1
                step = length * (self.leader_scale if char == "G" else 1.0)
                hx, hy, hz = frame[0]
                pos = (pos[0] + hx * step, pos[1] + hy * step, pos[2] + hz * step)
                vertices.append(pos)
                vertex_depths.append(depth)
                edges.append((vertex_index, len(vertices) - 1))
                edge_depths.append(depth)
                vertex_index = len(vertices) - 1
            elif char in _ROTATIONS:
                axis_index, sign = _ROTATIONS[char]
                u = _branch_jitter(seed, branch_hash, rotation_ordinal)
                rotation_ordinal += 1
                jitter = 1.0 + (u - 0.5) * 2.0 * self.randomness
                angle = self.angle * sign * jitter
                axis = frame[axis_index]
                c, s = math.cos(angle), math.sin(angle)
                for j in range(3):
                    if j != axis_index:
                        frame[j] = _rotate_frame_vec(frame[j], axis, c, s)
            elif char == "[":
                child_count += 1
                stack.append(
                    (
                        pos,
                        (frame[0], frame[1], frame[2]),
                        depth,
                        vertex_index,
                        length,
                        branch_hash,
                        rotation_ordinal,
                        child_count,
                    )
                )
                branch_hash = _mix64(branch_hash ^ child_count)
                rotation_ordinal = 0
                child_count = 0
                depth += 1
                length *= self.length_decay
            elif char == "]":
                if stack:
                    (
                        pos,
                        saved,
                        depth,
                        vertex_index,
                        length,
                        branch_hash,
                        rotation_ordinal,
                        child_count,
                    ) = stack.pop()
                    frame = list(saved)

        if not edges:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 2), dtype=np.uint32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
            )

        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(edges, dtype=np.uint32),
            np.asarray(edge_depths, dtype=np.int32),
            np.asarray(vertex_depths, dtype=np.int32),
        )


def derive_tree(
    lsystem: LSystem, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand + interpret one tree deterministically from its seed.

    Cross-stage coherence rests on two separated randomness sources: the
    expansion rng replays the same production choices for the shared
    derivation prefix when re-deriving at a HIGHER iteration count, and the
    turtle jitter is a pure function of each branch's bracket path (see
    :func:`_branch_jitter`), immune to the string growing around it. A
    branch that exists at two growth stages therefore keeps its exact
    orientation — growth adds geometry instead of re-rolling it. (The palm
    is the one exception; :func:`_branch_jitter` says why.)
    """
    rng = np.random.default_rng(seed)
    string = lsystem.expand(iterations, rng)
    return lsystem.interpret(string, seed)


def vary_lsystem(
    base: LSystem, rng: np.random.Generator, variation: float = 0.18
) -> LSystem:
    """A varied copy of an L-system: family resemblance, not clones."""
    return replace(
        base,
        angle=base.angle * (1.0 + rng.uniform(-variation, variation)),
        length=base.length * (1.0 + rng.uniform(-variation, variation)),
        width=base.width * (1.0 + rng.uniform(-variation * 0.4, variation * 0.4)),
        width_decay=float(
            np.clip(base.width_decay + rng.uniform(-0.08, 0.08), 0.5, 0.9)
        ),
        length_decay=float(
            np.clip(base.length_decay + rng.uniform(-0.06, 0.06), 0.55, 0.97)
        ),
        randomness=float(np.clip(base.randomness + rng.uniform(-0.1, 0.15), 0.1, 0.7)),
        tropism_strength=base.tropism_strength * (1.0 + rng.uniform(-0.3, 0.3)),
    )


# =============================================================================
# Species: eight families, each with its own grammar, palette and foliage
# =============================================================================

RGB = tuple[float, float, float]


def _linear_rgb(colors: np.ndarray) -> np.ndarray:
    """sRGB-intent -> linear-light for (N, 3) arrays. Luxar stores LINEAR
    colours (the classical-splat importers convert sRGB DC colours the same
    way); a palette authored as perceived-sRGB values and written raw comes
    out ~1/2.2-power brighter — pastel bark, washed-out ground. Applied
    exactly once per colour path, as the LAST step before the array lands
    in the bundle."""
    return np.clip(colors, 0.0, 1.0) ** 2.2


#: Autumn fire palette sampled per splat for deciduous foliage.
FIRE_COLORS = (
    (0.95, 0.45, 0.08),
    (0.92, 0.25, 0.08),
    (0.95, 0.72, 0.14),
    (0.85, 0.35, 0.10),
)

BLOSSOM_COLOR: RGB = (0.98, 0.74, 0.83)


@dataclass(frozen=True)
class Species:
    """One tree family: grammar + palettes + foliage behaviour + habitat."""

    key: str
    title: str
    lsystem: LSystem
    weight: float
    bark: RGB
    mid: RGB
    #: Branch-tip colour per season (index = season slot).
    season_tips: tuple[RGB, RGB, RGB, RGB]
    #: Foliage splat colour per season; None = bare that season.
    foliage: tuple[Optional[RGB], Optional[RGB], Optional[RGB], Optional[RGB]]
    #: Fraction of foliage splats kept per season (autumn thins, winter strips).
    foliage_keep: tuple[float, float, float, float] = (0.85, 1.0, 0.6, 0.0)
    #: Autumn foliage colour is sampled per splat from FIRE_COLORS.
    autumn_fire: bool = False
    #: Spring gets extra blossom splats.
    blossom: bool = False
    #: Foliage splats per tree at full maturity.
    foliage_max: int = 75
    #: Added to the scene-wide final iteration count (palms grow linearly,
    #: so they can afford — and need — deeper derivations).
    iter_bonus: int = 0
    base_scale: float = 0.9
    #: Placement bias: "any", "ridge", "wet" or "lowland".
    habitat: str = "any"


_DECIDUOUS_FOLIAGE = (
    (0.55, 0.78, 0.35),
    (0.16, 0.44, 0.13),
    (0.9, 0.45, 0.1),
    None,
)


SPECIES: tuple[Species, ...] = (
    Species(
        key="elegant",
        title="Maple",
        lsystem=LSystem(
            axiom="X",
            rules={
                "X": [
                    (0.5, "F[+X][-X][^X][&X]FX"),
                    (0.3, "F[+X][&X][-X]FX"),
                    (0.2, "F[-X][^X][+X]FX"),
                ],
                "F": "FF",
            },
            angle=np.radians(25),
            length=0.38,
            width=0.13,
            width_decay=0.7,
            length_decay=0.78,
            randomness=0.4,
        ),
        weight=0.16,
        bark=(0.30, 0.20, 0.12),
        mid=(0.42, 0.28, 0.15),
        season_tips=(
            (0.55, 0.72, 0.35),
            (0.25, 0.52, 0.18),
            (0.92, 0.38, 0.10),
            (0.78, 0.80, 0.86),
        ),
        foliage=_DECIDUOUS_FOLIAGE,
        autumn_fire=True,
        blossom=True,
    ),
    Species(
        key="fractal",
        title="Beech",
        lsystem=LSystem(
            axiom="FA",
            rules={
                "A": [
                    (0.65, "[+FA][-FA][^FA][&FA]"),
                    (0.35, "[+FA][&FA][-FA]"),
                ],
            },
            angle=np.radians(30),
            length=0.8,
            width=0.15,
            width_decay=0.65,
            length_decay=0.7,
            randomness=0.35,
        ),
        weight=0.11,
        bark=(0.33, 0.26, 0.18),
        mid=(0.44, 0.34, 0.22),
        season_tips=(
            (0.62, 0.78, 0.40),
            (0.30, 0.58, 0.22),
            (0.95, 0.68, 0.16),
            (0.75, 0.77, 0.84),
        ),
        foliage=_DECIDUOUS_FOLIAGE,
        autumn_fire=True,
    ),
    Species(
        key="willow",
        title="Willow",
        lsystem=LSystem(
            axiom="FFFFA",
            rules={
                "A": [
                    (0.55, "[&&&B][&&&&B]FA"),
                    (0.45, "[&&&B][&&&&B][&&&&&B]FA"),
                ],
                "B": [(0.7, "&&F[-F]BF[+F]"), (0.3, "&F[+F]B")],
            },
            angle=np.radians(18),
            length=0.5,
            width=0.09,
            width_decay=0.78,
            length_decay=0.88,
            randomness=0.45,
            tropism=(0.0, 0.0, -1.0),
            tropism_strength=0.2,
        ),
        weight=0.13,
        bark=(0.26, 0.19, 0.11),
        mid=(0.36, 0.28, 0.16),
        season_tips=(
            (0.60, 0.78, 0.42),
            (0.38, 0.65, 0.30),
            (0.88, 0.75, 0.25),
            (0.72, 0.75, 0.82),
        ),
        foliage=(
            (0.58, 0.80, 0.40),
            (0.30, 0.58, 0.24),
            (0.88, 0.72, 0.20),
            None,
        ),
        habitat="wet",
    ),
    Species(
        key="bush",
        title="Bush",
        lsystem=LSystem(
            axiom="FA",
            rules={
                "A": [
                    (0.6, "[++++FA][----FA][^^^^FA][&&&&FA]FA"),
                    (0.4, "[++++FA][^^^^FA][&&&&FA]FA"),
                ],
            },
            angle=np.radians(12),
            length=0.32,
            width=0.05,
            width_decay=0.8,
            length_decay=0.85,
            randomness=0.5,
        ),
        weight=0.13,
        bark=(0.24, 0.17, 0.10),
        mid=(0.34, 0.25, 0.13),
        season_tips=(
            (0.52, 0.70, 0.32),
            (0.24, 0.50, 0.16),
            (0.85, 0.30, 0.12),
            (0.70, 0.72, 0.78),
        ),
        foliage=_DECIDUOUS_FOLIAGE,
        autumn_fire=True,
        blossom=True,
        foliage_max=40,
        base_scale=0.65,
    ),
    Species(
        key="conifer",
        title="Conifer",
        lsystem=LSystem(
            # Honda's monopodial model: the trunk is an apical leader (G)
            # laying down one whorl of five branches per derivation step.
            # Older (lower) whorls have expanded for more iterations, so
            # they are naturally longer — the classic conical silhouette
            # with no explicit taper rule.
            axiom="A",
            rules={
                "A": "G[&&&B]///[&&&B]///[&&&B]///[&&&B]///[&&&B]//A",
                "B": [(0.7, "F[^F][&F]FB"), (0.3, "FF[&F]B")],
            },
            angle=np.radians(24),
            # Stature: without the longer internodes a "conifer" tops out
            # around 2.4 m while the maples reach 10 m — the ridge species
            # read as shrubs and the eco-zone silhouette collapses.
            length=0.58,
            width=0.14,
            width_decay=0.6,
            length_decay=0.82,
            randomness=0.18,
            tropism=(0.0, 0.0, -1.0),
            tropism_strength=0.06,
            leader_scale=1.7,
        ),
        weight=0.17,
        bark=(0.28, 0.18, 0.11),
        mid=(0.34, 0.24, 0.14),
        season_tips=(
            (0.16, 0.42, 0.22),
            (0.12, 0.38, 0.20),
            (0.12, 0.36, 0.19),
            (0.55, 0.66, 0.70),
        ),
        foliage=(
            (0.12, 0.36, 0.18),
            (0.10, 0.32, 0.16),
            (0.10, 0.30, 0.15),
            (0.30, 0.42, 0.36),
        ),
        foliage_keep=(0.9, 1.0, 0.95, 0.9),
        foliage_max=85,
        habitat="ridge",
    ),
    Species(
        key="poplar",
        title="Poplar",
        lsystem=LSystem(
            axiom="FA",
            rules={
                "A": [(0.7, "F[+B][-B][^B]/FA"), (0.3, "F[+B][&B]/FA")],
                "B": [(0.6, "F[+F]FB"), (0.4, "FFB")],
            },
            angle=np.radians(16),
            length=0.42,
            width=0.10,
            width_decay=0.68,
            length_decay=0.85,
            randomness=0.3,
            # Upward phototropism: side branches curl back toward vertical,
            # producing the tight columnar silhouette.
            tropism=(0.0, 0.0, 1.0),
            tropism_strength=0.22,
        ),
        weight=0.12,
        bark=(0.33, 0.30, 0.24),
        mid=(0.42, 0.39, 0.30),
        season_tips=(
            (0.58, 0.75, 0.38),
            (0.32, 0.60, 0.24),
            (0.95, 0.80, 0.20),
            (0.80, 0.82, 0.88),
        ),
        foliage=(
            (0.56, 0.78, 0.36),
            (0.24, 0.52, 0.18),
            (0.95, 0.78, 0.18),
            None,
        ),
        foliage_max=45,
    ),
    Species(
        key="palm",
        title="Palm",
        lsystem=LSystem(
            # A linear-growth rosette: the trunk adds one internode per
            # iteration (T -> F/T) while the crown symbol C unfolds once
            # into eight fronds whose arcs lengthen and droop with age
            # (P -> F&P plus gravity tropism).
            axiom="TC",
            rules={
                "T": "F/T",
                "C": "[&&P]//[&&P]//[&&P]//[&&P]//[&&P]//[&&P]//[&&P]//[&&P]",
                "P": "F&P",
            },
            angle=np.radians(26),
            length=0.55,
            width=0.16,
            width_decay=0.75,
            length_decay=0.95,
            randomness=0.25,
            tropism=(0.0, 0.0, -1.0),
            tropism_strength=0.12,
        ),
        weight=0.09,
        bark=(0.45, 0.36, 0.24),
        mid=(0.42, 0.38, 0.24),
        season_tips=(
            (0.30, 0.60, 0.28),
            (0.26, 0.56, 0.24),
            (0.30, 0.55, 0.22),
            (0.30, 0.50, 0.26),
        ),
        foliage=(
            (0.26, 0.56, 0.24),
            (0.22, 0.52, 0.20),
            (0.24, 0.50, 0.20),
            (0.24, 0.46, 0.22),
        ),
        foliage_keep=(0.8, 0.85, 0.8, 0.75),
        foliage_max=30,
        iter_bonus=3,
        habitat="lowland",
    ),
    Species(
        key="dead_oak",
        title="Dead Oak",
        lsystem=LSystem(
            axiom="FX",
            rules={
                "X": [
                    (0.4, "F[+X][-X]^X"),
                    (0.25, "F[&X]F[^X]X"),
                    (0.2, "F[+X]F[-X]X"),
                    (0.15, "FX"),
                ],
                "F": [(0.85, "FF"), (0.15, "F")],
            },
            angle=np.radians(32),
            length=0.35,
            width=0.14,
            width_decay=0.72,
            length_decay=0.82,
            randomness=0.6,
            tropism=(0.0, 0.0, -1.0),
            tropism_strength=0.05,
        ),
        weight=0.09,
        bark=(0.22, 0.19, 0.16),
        mid=(0.32, 0.28, 0.24),
        season_tips=(
            (0.45, 0.42, 0.38),
            (0.45, 0.42, 0.38),
            (0.42, 0.38, 0.34),
            (0.82, 0.84, 0.90),
        ),
        foliage=(None, None, None, None),
        foliage_keep=(0.0, 0.0, 0.0, 0.0),
        foliage_max=0,
    ),
)


# =============================================================================
# fBm terrain
# =============================================================================

FOREST_SIZE = 76.0  # trees are planted inside this square (m)
TERRAIN_SIZE = 96.0  # the terrain mesh extends a little beyond the trees
TERRAIN_GRID = 129  # vertices per side (128x128 cells, ~32k triangles)

_LATTICE = 64  # value-noise lattice resolution (wraps; never reached here)


def _lattice_noise(table: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smoothstep-interpolated value noise on a wrapped random lattice."""
    i0 = np.floor(u).astype(np.int64)
    j0 = np.floor(v).astype(np.int64)
    fu = u - i0
    fv = v - j0
    su = fu * fu * (3.0 - 2.0 * fu)
    sv = fv * fv * (3.0 - 2.0 * fv)
    i0 %= _LATTICE
    j0 %= _LATTICE
    i1 = (i0 + 1) % _LATTICE
    j1 = (j0 + 1) % _LATTICE
    top = table[i0, j0] * (1.0 - su) + table[i1, j0] * su
    bottom = table[i0, j1] * (1.0 - su) + table[i1, j1] * su
    return top * (1.0 - sv) + bottom * sv


class FBm:
    """Fractional Brownian motion over 2D value noise (shared by the terrain
    mesh, tree planting heights and the accent layers, so they agree exactly)."""

    def __init__(
        self,
        rng: np.random.Generator,
        octaves: int = 5,
        base_cell: float = 34.0,
        amplitude: float = 3.4,
        gain: float = 0.55,
        lacunarity: float = 2.0,
    ) -> None:
        self.tables = [
            rng.uniform(-1.0, 1.0, (_LATTICE, _LATTICE)) for _ in range(octaves)
        ]
        self.base_cell = base_cell
        self.amplitude = amplitude
        self.gain = gain
        self.lacunarity = lacunarity

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
        amp = self.amplitude
        cell = self.base_cell
        for table in self.tables:
            total += amp * _lattice_noise(table, x / cell + 17.31, y / cell + 5.77)
            amp *= self.gain
            cell /= self.lacunarity
        return total


@dataclass
class Terrain:
    """The heightfield bundle: mesh arrays + the continuous field samplers."""

    vertices: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (F, 3) uint32
    normals: np.ndarray  # (V, 3) float32
    season_colors: list[np.ndarray]  # 4 x (V, 3) float32
    height: FBm
    moisture: FBm
    h_min: float
    h_max: float

    def height_at(self, x: Any, y: Any) -> np.ndarray:
        return self.height(x, y)

    def h_norm_at(self, x: Any, y: Any) -> np.ndarray:
        span = max(self.h_max - self.h_min, 1e-6)
        return np.clip((self.height(x, y) - self.h_min) / span, 0.0, 1.0)

    def moisture_at(self, x: Any, y: Any) -> np.ndarray:
        return 0.5 + 0.5 * np.tanh(self.moisture(x, y) * 0.9)


def _terrain_season_colors(
    h_norm: np.ndarray, moisture: np.ndarray, slope: np.ndarray
) -> list[np.ndarray]:
    """Per-season vertex colours from height band, moisture and slope."""

    def lerp(a: RGB, b: RGB, t: np.ndarray) -> np.ndarray:
        t = t[:, None]
        return (
            np.asarray(a, dtype=np.float32) * (1.0 - t)
            + np.asarray(b, dtype=np.float32) * t
        )

    rockiness = np.clip(slope * 2.2 + (h_norm - 0.62) * 2.0, 0.0, 1.0)
    colors: list[np.ndarray] = []
    # Values are sRGB INTENTS (what the ground should look like) and are
    # linearized below; kept on the dark side regardless — a bright ground
    # plane fills half the frame, washing out the emissive trees above it
    # and feeding the bloom pass (measured, not guessed).
    grass_by_season = (
        ((0.28, 0.42, 0.20), (0.36, 0.48, 0.24)),  # spring: fresh, wetter = lusher
        ((0.20, 0.36, 0.15), (0.27, 0.43, 0.18)),  # summer: deep green
        ((0.42, 0.35, 0.18), (0.38, 0.38, 0.19)),  # autumn: dried tan-olive
        ((0.78, 0.81, 0.87), (0.72, 0.76, 0.84)),  # winter: snow
    )
    rock_by_season = (
        (0.42, 0.39, 0.35),
        (0.43, 0.40, 0.36),
        (0.42, 0.38, 0.33),
        (0.53, 0.57, 0.66),  # winter: wind-scoured blue-grey rock
    )
    for season in range(4):
        dry, wet = grass_by_season[season]
        grass = lerp(dry, wet, moisture)
        color = (
            grass * (1.0 - rockiness[:, None])
            + np.asarray(rock_by_season[season], dtype=np.float32) * rockiness[:, None]
        )
        if season == WINTER:
            # Snow shades toward blue in steep, shadowed folds.
            shadow = np.clip(slope * 1.6, 0.0, 0.5)[:, None]
            color = (
                color * (1.0 - shadow)
                + np.asarray((0.50, 0.57, 0.72), dtype=np.float32) * shadow
            )
        colors.append(_linear_rgb(color).astype(np.float32))
    return colors


def build_terrain(rng: np.random.Generator) -> Terrain:
    """Build the fBm heightfield mesh with per-season vertex colours."""
    height = FBm(rng)
    moisture = FBm(rng, octaves=3, base_cell=26.0, amplitude=1.0)

    n = TERRAIN_GRID
    axis = np.linspace(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, n)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    gz = height(gx, gy)
    vertices = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)

    # Two CCW-wound triangles per cell (front face toward +z).
    idx = np.arange(n * n, dtype=np.uint32).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[:-1, 1:].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.concatenate(
        [np.column_stack([a, b, d]), np.column_stack([a, d, c])]
    ).astype(np.uint32)

    # Area-weighted vertex normals.
    tri = vertices[faces.astype(np.int64)]
    face_normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner].astype(np.int64), face_normal)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.maximum(norm, 1e-12)).astype(np.float32)

    h_min = float(gz.min())
    h_max = float(gz.max())
    h_norm = (vertices[:, 2] - h_min) / max(h_max - h_min, 1e-6)
    slope = np.clip(1.0 - normals[:, 2], 0.0, 1.0)
    moist = 0.5 + 0.5 * np.tanh(moisture(vertices[:, 0], vertices[:, 1]) * 0.9)
    season_colors = _terrain_season_colors(h_norm, moist, slope)

    return Terrain(
        vertices=vertices,
        faces=faces,
        normals=normals,
        season_colors=season_colors,
        height=height,
        moisture=moisture,
        h_min=h_min,
        h_max=h_max,
    )


# =============================================================================
# Poisson-disk planting with eco-zone species selection
# =============================================================================


def poisson_disk_sampling(
    rng: np.random.Generator,
    width: float,
    height: float,
    min_dist: float,
    max_points: int,
    k: int = 30,
) -> list[tuple[float, float]]:
    """Bridson's algorithm for Poisson disk sampling.

    Generates points with guaranteed minimum distance between them,
    creating natural-looking distributions like forests.
    """
    cell_size = min_dist / np.sqrt(2)
    grid_width = int(np.ceil(width / cell_size))
    grid_height = int(np.ceil(height / cell_size))

    grid = np.full((grid_width, grid_height), -1, dtype=np.int32)
    points: list[tuple[float, float]] = []
    active: list[int] = []

    def grid_coords(x: float, y: float) -> tuple[int, int]:
        return int(x / cell_size), int(y / cell_size)

    def is_valid(x: float, y: float) -> bool:
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        gx, gy = grid_coords(x, y)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    index = grid[nx, ny]
                    if index >= 0:
                        px, py = points[index]
                        if (x - px) ** 2 + (y - py) ** 2 < min_dist**2:
                            return False
        return True

    x0 = rng.uniform(0, width)
    y0 = rng.uniform(0, height)
    points.append((x0, y0))
    gx, gy = grid_coords(x0, y0)
    grid[gx, gy] = 0
    active.append(0)

    while active and len(points) < max_points:
        index = rng.integers(0, len(active))
        px, py = points[active[index]]

        found = False
        for _ in range(k):
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(min_dist, 2 * min_dist)
            x = px + r * np.cos(angle)
            y = py + r * np.sin(angle)

            if is_valid(x, y):
                new_index = len(points)
                points.append((x, y))
                gx, gy = grid_coords(x, y)
                grid[gx, gy] = new_index
                active.append(new_index)
                found = True
                break

        if not found:
            active.pop(index)

    return points


@dataclass
class TreePlan:
    """Everything needed to derive and place one tree."""

    index: int  # instance number within its species
    species_index: int
    x: float
    y: float
    z: float
    seed: int
    scale: float
    rotation: float
    stage_offset: int  # this tree matures `offset` growth slots late
    tint_shift: np.ndarray  # (3,) additive per-tree colour shift
    brightness: float
    lsystem: LSystem


def _habitat_weight(species: Species, h_norm: float, moisture: float) -> float:
    """Eco-zone bias: conifers climb the ridges, palms and willows keep wet feet."""
    if species.habitat == "ridge":
        return species.weight * (0.4 + 1.8 * h_norm)
    if species.habitat == "wet":
        return species.weight * (0.35 + 1.8 * moisture)
    if species.habitat == "lowland":
        return species.weight * (0.3 + 2.4 * moisture * (1.0 - h_norm))
    return species.weight


def plant_trees(
    rng: np.random.Generator, terrain: Terrain, n_trees: int
) -> list[TreePlan]:
    """Place trees by Poisson disk, thin the ridges, assign species by habitat."""
    # Oversample: the ridge-thinning rejection below eats ~25% of the
    # candidates, and Poisson packing at this spacing saturates near the
    # requested count — without headroom the forest comes up short.
    raw = poisson_disk_sampling(
        rng,
        width=FOREST_SIZE,
        height=FOREST_SIZE,
        min_dist=2.05,
        max_points=int(n_trees * 1.9),
        k=30,
    )
    plans: list[TreePlan] = []
    per_species_counts = [0] * len(SPECIES)
    for x0, y0 in raw:
        if len(plans) >= n_trees:
            break
        x = x0 - FOREST_SIZE / 2
        y = y0 - FOREST_SIZE / 2
        h_norm = float(terrain.h_norm_at(x, y))
        moisture = float(terrain.moisture_at(x, y))
        # Ridges thin out; damp hollows crowd in.
        accept = 0.6 + 0.4 * moisture - 0.55 * max(0.0, h_norm - 0.72)
        if rng.random() > accept:
            continue
        weights = np.array([_habitat_weight(sp, h_norm, moisture) for sp in SPECIES])
        weights /= weights.sum()
        species_index = int(rng.choice(len(SPECIES), p=weights))
        species = SPECIES[species_index]
        plans.append(
            TreePlan(
                index=per_species_counts[species_index],
                species_index=species_index,
                x=x,
                y=y,
                z=float(terrain.height_at(x, y)) - 0.05,
                seed=int(rng.integers(0, 2**31 - 1)),
                scale=species.base_scale * (0.85 + 0.3 * rng.random()),
                rotation=float(rng.uniform(0, 2 * np.pi)),
                stage_offset=int(rng.choice([0, 0, 1, 1, 2])),
                tint_shift=rng.uniform(-0.05, 0.05, size=3).astype(np.float32),
                brightness=float(rng.uniform(0.88, 1.12)),
                lsystem=vary_lsystem(species.lsystem, rng),
            )
        )
        per_species_counts[species_index] += 1
    return plans


# =============================================================================
# Per-tree geometry across growth stages and seasons
# =============================================================================


def _depth_gradient(t: np.ndarray, bark: RGB, mid: RGB, tip: RGB) -> np.ndarray:
    """Vectorised colour gradient along normalised branch depth.

    The seasonal tip colour saturates by t=0.66, NOT t=1.0 — the deepest
    twigs are the thinnest (near-invisible) ones, so a gradient that only
    reaches the tip colour there leaves every crown bark-brown year-round.
    """
    stops_t = np.array([0.0, 0.32, 0.66, 1.0])
    stops_rgb = np.array([bark, mid, tip, tip], dtype=np.float32)
    out = np.empty((len(t), 3), dtype=np.float32)
    for channel in range(3):
        out[:, channel] = np.interp(t, stops_t, stops_rgb[:, channel])
    return out


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass
class SpeciesArrays:
    """Accumulators for one merged per-species Lines node."""

    vertices: list[np.ndarray] = field(default_factory=list)
    widths: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    sharpness: list[np.ndarray] = field(default_factory=list)
    edges: list[np.ndarray] = field(default_factory=list)
    label_runs: list[tuple[int, str]] = field(default_factory=list)
    n_vertices: int = 0
    n_segments: int = 0
    n_trees: int = 0


@dataclass
class FoliageArrays:
    """Accumulators for the single merged foliage GSplats node."""

    centers: list[np.ndarray] = field(default_factory=list)
    cholesky: list[np.ndarray] = field(default_factory=list)
    amplitudes: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    label_runs: list[tuple[int, str]] = field(default_factory=list)
    n_splats: int = 0


def _pack_spatial_cholesky_5d(spatial_l: np.ndarray) -> np.ndarray:
    """Embed (K, 3, 3) spatial lower-triangular factors into the 15-element
    packed lower-triangular layout of the 5D scene (season, growth, x, y, z).

    Packed order is row-major over ``np.tril_indices(5)``; the two stacked
    axes get the near-zero diagonal of :data:`STACKED_AXIS_SIGMA` (slot
    membership comes from the discrete gate, not from this — see there).
    """
    k = len(spatial_l)
    packed = np.zeros((k, 15), dtype=np.float32)
    packed[:, 0] = STACKED_AXIS_SIGMA  # (season, season)
    packed[:, 2] = STACKED_AXIS_SIGMA  # (growth, growth)
    packed[:, 5] = spatial_l[:, 0, 0]  # (x, x)
    packed[:, 8] = spatial_l[:, 1, 0]  # (y, x)
    packed[:, 9] = spatial_l[:, 1, 1]  # (y, y)
    packed[:, 12] = spatial_l[:, 2, 0]  # (z, x)
    packed[:, 13] = spatial_l[:, 2, 1]  # (z, y)
    packed[:, 14] = spatial_l[:, 2, 2]  # (z, z)
    return packed


def _foliage_cholesky(
    directions: np.ndarray, sigma_along: np.ndarray, sigma_perp: np.ndarray
) -> np.ndarray:
    """Anisotropic 3D covariances elongated along the parent branch."""
    d = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-9)
    eye = np.eye(3, dtype=np.float64)
    outer = d[:, :, None] * d[:, None, :]
    cov = (
        sigma_perp[:, None, None] ** 2 * eye
        + (sigma_along[:, None, None] ** 2 - sigma_perp[:, None, None] ** 2) * outer
    )
    return np.linalg.cholesky(cov)


def _tip_anchors(
    vertices: np.ndarray,
    edges: np.ndarray,
    edge_depths: np.ndarray,
    max_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Branch-tip positions and directions (foliage anchors)."""
    threshold = max(1.0, max_depth * 0.55)
    mask = edge_depths >= threshold
    starts = vertices[edges[mask, 0].astype(np.int64)]
    ends = vertices[edges[mask, 1].astype(np.int64)]
    return ends, ends - starts


def _stage_world_geometry(
    plan: TreePlan,
    derived: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    maturity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local derivation -> world-space vertices, widths, sharpness, depth-t."""
    vertices, _edges, _edge_depths, vertex_depths = derived
    max_depth = int(vertex_depths.max()) if len(vertex_depths) else 1
    t = vertex_depths / max(max_depth, 1)

    scale = plan.scale * _maturity_length_scale(maturity)
    world = (vertices @ _rotation_z(plan.rotation).T) * scale
    world = world + np.array([plan.x, plan.y, plan.z], dtype=np.float32)

    base_width = plan.lsystem.width * plan.scale * _maturity_width_scale(maturity)
    min_width = 0.006 * plan.scale
    widths = (min_width + (base_width - min_width) * np.exp(-t * 3.0)).astype(
        np.float32
    )
    sharpness = (0.65 - t * 0.35).astype(np.float32)
    return world.astype(np.float32), widths, sharpness, t.astype(np.float32)


def _append_tree_lines(
    out: SpeciesArrays,
    species: Species,
    plan: TreePlan,
    world: np.ndarray,
    widths: np.ndarray,
    sharpness: np.ndarray,
    t: np.ndarray,
    edges: np.ndarray,
    growth: int,
    effective_stage: int,
) -> None:
    """Append one (tree, growth slot) block for all four seasons."""
    n = len(world)
    n_edges = len(edges)
    stage_name = GROWTH_STAGES[effective_stage].lower()
    for season in range(4):
        block = np.empty((n, 5), dtype=np.float32)
        block[:, 0] = float(season)
        block[:, 1] = float(growth)
        block[:, 2:] = world
        colors = _depth_gradient(
            t, species.bark, species.mid, species.season_tips[season]
        )
        # Tint in sRGB-intent space (perceptually even), then linearize once.
        colors = np.clip(colors * plan.brightness + plan.tint_shift, 0.0, 1.0)
        colors = _linear_rgb(colors).astype(np.float32)
        label = (
            f"{species.title} #{plan.index + 1:03d} · {SEASONS[season].lower()} · "
            f"{stage_name} ({effective_stage + 1}/{N_STAGES}) · "
            f"{n_edges:,} segments"
        )
        out.vertices.append(block)
        out.widths.append(widths)
        out.colors.append(colors)
        out.sharpness.append(sharpness)
        out.edges.append(edges.astype(np.uint32) + np.uint32(out.n_vertices))
        out.label_runs.append((n, label))
        out.n_vertices += n
        out.n_segments += n_edges


def _append_tree_foliage(
    out: FoliageArrays,
    species: Species,
    plan: TreePlan,
    rng: np.random.Generator,
    tips: np.ndarray,
    tip_dirs: np.ndarray,
    growth: int,
    maturity: float,
) -> None:
    """Append this tree's foliage splats for one growth slot, per season."""
    n_target = int(round(species.foliage_max * maturity * maturity))
    if n_target < 3 or len(tips) == 0:
        return
    take = min(len(tips), n_target)
    order = rng.permutation(len(tips))[:take]
    anchors = tips[order]
    directions = tip_dirs[order]

    sigma_along = (
        0.4 * plan.scale * (0.4 + 0.6 * maturity) * rng.uniform(0.75, 1.3, take)
    )
    sigma_perp = sigma_along * rng.uniform(0.5, 0.7, take)
    spatial_l = _pack_spatial_cholesky_5d(
        _foliage_cholesky(directions.astype(np.float64), sigma_along, sigma_perp)
    )

    for season in range(4):
        base_rgb = species.foliage[season]
        keep = species.foliage_keep[season]
        if base_rgb is None or keep <= 0.0:
            continue
        mask = rng.random(take) < keep
        m = int(mask.sum())
        if m == 0:
            continue
        if season == AUTUMN and species.autumn_fire:
            picks = rng.integers(0, len(FIRE_COLORS), m)
            rgb = np.asarray(FIRE_COLORS, dtype=np.float32)[picks]
        else:
            rgb = np.tile(np.asarray(base_rgb, dtype=np.float32), (m, 1))
        rgb = np.clip(rgb * rng.uniform(0.85, 1.15, (m, 1)), 0.0, 1.0)
        rgb = _linear_rgb(rgb).astype(np.float32)

        centers = np.empty((m, 5), dtype=np.float32)
        centers[:, 0] = float(season)
        centers[:, 1] = float(growth)
        centers[:, 2:] = anchors[mask]
        out.centers.append(centers)
        out.cholesky.append(spatial_l[mask])
        out.amplitudes.append(rng.uniform(0.5, 1.1, m).astype(np.float32))
        out.colors.append(rgb)
        out.label_runs.append(
            (
                m,
                f"{species.title} #{plan.index + 1:03d} foliage · "
                f"{SEASONS[season].lower()}",
            )
        )
        out.n_splats += m

    # Spring blossom overlay: smaller, rounder, pink.
    if species.blossom:
        mask = rng.random(take) < 0.45
        m = int(mask.sum())
        if m == 0:
            return
        sigma = sigma_perp[mask] * 0.6
        blossom_l = _pack_spatial_cholesky_5d(
            np.eye(3, dtype=np.float64)[None] * sigma[:, None, None]
        )
        rgb = np.clip(
            np.tile(np.asarray(BLOSSOM_COLOR, dtype=np.float32), (m, 1))
            * rng.uniform(0.9, 1.1, (m, 1)),
            0.0,
            1.0,
        )
        rgb = _linear_rgb(rgb).astype(np.float32)
        centers = np.empty((m, 5), dtype=np.float32)
        centers[:, 0] = float(SPRING)
        centers[:, 1] = float(growth)
        centers[:, 2:] = anchors[mask]
        out.centers.append(centers)
        out.cholesky.append(blossom_l)
        out.amplitudes.append(rng.uniform(0.6, 1.2, m).astype(np.float32))
        out.colors.append(rgb)
        out.label_runs.append(
            (m, f"{species.title} #{plan.index + 1:03d} blossom · spring")
        )
        out.n_splats += m


def _effective_stage(growth: int, stage_offset: int) -> int:
    """A staggered tree's own stage at global growth slot ``growth``.

    The stagger delays development WITHOUT truncating the endpoint: a
    delayed tree starts late, lags through the middle slots, and catches up
    to reach the SAME final stage in the last slot (young trees grow fast).
    A plain ``growth - offset`` clamp would leave offset trees permanently
    short of the final stage — at the authored "ancient" poster slice most
    of the forest would never actually be ancient.
    """
    last = N_STAGES - 1
    if stage_offset <= 0:
        return int(np.clip(growth, 0, last))
    scaled = (growth - stage_offset) * last / (last - stage_offset)
    return int(np.clip(round(scaled), 0, last))


def _build_tree(
    plan: TreePlan,
    species_out: SpeciesArrays,
    foliage_out: FoliageArrays,
    final_iterations: int,
) -> None:
    """Derive one tree at every distinct depth and author its 24 slot blocks."""
    species = SPECIES[plan.species_index]
    foliage_rng = np.random.default_rng(plan.seed + 1)

    derivations: dict[int, tuple[np.ndarray, ...]] = {}
    for growth in range(N_STAGES):
        effective_stage = _effective_stage(growth, plan.stage_offset)
        iterations = max(1, final_iterations + STAGE_ITER_OFFSETS[effective_stage])
        if iterations not in derivations:
            derivations[iterations] = derive_tree(plan.lsystem, iterations, plan.seed)
        derived = derivations[iterations]
        vertices, edges, edge_depths, _vertex_depths = derived
        if len(edges) == 0:
            continue

        maturity = STAGE_MATURITY[effective_stage]
        world, widths, sharpness, t = _stage_world_geometry(plan, derived, maturity)
        _append_tree_lines(
            species_out,
            species,
            plan,
            world,
            widths,
            sharpness,
            t,
            edges,
            growth,
            effective_stage,
        )

        max_depth = int(edge_depths.max())
        # Depth >= 1 (not 2): the palm's fronds all live at branch depth 1 —
        # `C` opens a single bracket level and `P -> F&P` nests no further —
        # so a deeper gate would leave palms bare in every season.
        if max_depth >= 1 and iterations >= 2:
            local_tips, local_dirs = _tip_anchors(
                vertices, edges, edge_depths, max_depth
            )
            if len(local_tips):
                scale = plan.scale * _maturity_length_scale(maturity)
                rot = _rotation_z(plan.rotation).T
                tips = (local_tips @ rot) * scale + np.array(
                    [plan.x, plan.y, plan.z], dtype=np.float32
                )
                dirs = (local_dirs @ rot).astype(np.float64)
                _append_tree_foliage(
                    foliage_out,
                    species,
                    plan,
                    foliage_rng,
                    tips.astype(np.float32),
                    dirs,
                    growth,
                    maturity,
                )


# =============================================================================
# Accent point layers (each pinned to one season, extended over growth)
# =============================================================================


def _build_accents(
    rng: np.random.Generator, terrain: Terrain, plans: list[TreePlan]
) -> dict[str, dict[str, Any]]:
    """Fireflies (summer), frost sparkle (winter), petals (spring)."""
    accents: dict[str, dict[str, Any]] = {}
    half = FOREST_SIZE / 2

    n_fireflies = 350
    x = rng.uniform(-half, half, n_fireflies)
    y = rng.uniform(-half, half, n_fireflies)
    z = (
        terrain.height_at(x, y)
        + 0.4
        + np.minimum(rng.exponential(0.8, n_fireflies), 2.6)
    )
    accents["Fireflies"] = dict(
        season=SUMMER,
        positions=np.column_stack([x, y, z]).astype(np.float32),
        colors=_linear_rgb(
            np.tile(np.array([1.0, 0.85, 0.45], np.float32), (n_fireflies, 1))
            * rng.uniform(0.75, 1.05, (n_fireflies, 1))
        ).astype(np.float32),
        radii=rng.uniform(0.05, 0.09, n_fireflies).astype(np.float32),
        sharpness=0.3,
        blending_mode="luminous",
        intensity=1.4,
        label="Firefly · summer dusk",
    )

    n_frost = 6000
    x = rng.uniform(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, n_frost)
    y = rng.uniform(-TERRAIN_SIZE / 2, TERRAIN_SIZE / 2, n_frost)
    z = terrain.height_at(x, y) + 0.04
    accents["Frost Sparkle"] = dict(
        season=WINTER,
        positions=np.column_stack([x, y, z]).astype(np.float32),
        colors=_linear_rgb(
            np.tile(np.array([0.85, 0.92, 1.0], np.float32), (n_frost, 1))
            * rng.uniform(0.6, 1.0, (n_frost, 1))
        ).astype(np.float32),
        radii=rng.uniform(0.02, 0.05, n_frost).astype(np.float32),
        sharpness=0.85,
        blending_mode="luminous",
        intensity=1.1,
        label="Frost sparkle · winter",
    )

    blossom_plans = [p for p in plans if SPECIES[p.species_index].blossom]
    petal_blocks: list[np.ndarray] = []
    for plan in blossom_plans:
        n = int(rng.integers(6, 15))
        px = plan.x + rng.normal(0.0, 1.4 * plan.scale, n)
        py = plan.y + rng.normal(0.0, 1.4 * plan.scale, n)
        pz = terrain.height_at(px, py) + rng.uniform(0.1, 2.4 * plan.scale, n)
        petal_blocks.append(np.column_stack([px, py, pz]))
    if petal_blocks:
        petals = np.concatenate(petal_blocks).astype(np.float32)
        n_petals = len(petals)
        accents["Petals"] = dict(
            season=SPRING,
            positions=petals,
            colors=_linear_rgb(
                np.tile(np.asarray(BLOSSOM_COLOR, np.float32), (n_petals, 1))
                * rng.uniform(0.85, 1.1, (n_petals, 1))
            ).astype(np.float32),
            radii=rng.uniform(0.03, 0.055, n_petals).astype(np.float32),
            sharpness=0.5,
            blending_mode="luminous",
            intensity=0.9,
            label="Falling petal · spring",
        )
    return accents


# =============================================================================
# Seasonal cloud cover
# =============================================================================

#: Cloud deck floor above the highest terrain point (m). Well clear of the
#: tallest canopy so the veil reads as sky, not as fog in the trees.
CLOUD_DECK_HEIGHT = 26.0
#: Vertical thickness of the deck (m); the noise field also modulates it.
CLOUD_DECK_THICKNESS = 10.0
#: The deck extends past the terrain so its edge never shows in the opening view.
CLOUD_FIELD_SIZE = TERRAIN_SIZE * 1.6
#: Candidate spacing on the jittered sampling grid (m).
CLOUD_SPACING = 2.0
#: Hard cap per season, so four decks stay a light accent in the element budget.
CLOUD_MAX_POINTS = 3400
#: Coverage threshold on the normalised [0, 1] noise field, per season slot:
#: lower = more sky covered. Spring: scattered fair-weather clouds; summer: a
#: few high wisps; autumn: broken cover; winter: a heavy, mostly closed deck.
CLOUD_THRESHOLDS = (0.54, 0.66, 0.47, 0.40)
#: Deck tint per season (sRGB intent; linearised on write). Winter is grey-blue.
CLOUD_TINTS: tuple[RGB, RGB, RGB, RGB] = (
    (1.00, 0.97, 0.93),
    (1.00, 0.99, 0.96),
    (0.94, 0.92, 0.90),
    (0.80, 0.85, 0.92),
)
#: Per-season drift of the field (m) so the same sky does not sit still over
#: the year; the coverage change alone would only fade one cloud in and out.
CLOUD_DRIFT = ((0.0, 0.0), (23.0, -11.0), (47.0, 9.0), (70.0, -26.0))
#: How faint the veil is: additive points at this gain barely lift the sky
#: over the far canopy ("very very light" was the brief).
CLOUD_INTENSITY = 0.03


def cloud_field(fbm: FBm, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Multiscale value-noise cloud density in [0, 1] (0.5 = the field's mean)."""
    raw = fbm(x, y)
    # An fBm with amplitude 1 and gain 0.5 over 4 octaves spans about +-1.9.
    return np.clip(0.5 + raw / 3.8, 0.0, 1.0)


def _cloud_deck_for_season(
    rng: np.random.Generator, fbm: FBm, terrain: Terrain, season: int
) -> np.ndarray:
    """Cloud points (N, 3) for one season: a thresholded, edge-feathered field."""
    half = CLOUD_FIELD_SIZE / 2
    n_side = int(CLOUD_FIELD_SIZE / CLOUD_SPACING)
    axis = np.linspace(-half, half, n_side)
    gx, gy = np.meshgrid(axis, axis, indexing="ij")
    x = gx.ravel() + rng.uniform(-0.5, 0.5, gx.size) * CLOUD_SPACING
    y = gy.ravel() + rng.uniform(-0.5, 0.5, gy.size) * CLOUD_SPACING
    dx, dy = CLOUD_DRIFT[season]
    density = cloud_field(fbm, x + dx, y + dy)
    threshold = CLOUD_THRESHOLDS[season]
    # Feathered edges: a point survives with probability rising from 0 at the
    # threshold to 1 well inside a cloud, so the deck has no hard outline.
    inside = np.clip((density - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    keep = rng.random(x.size) < np.sqrt(inside)
    x, y, inside = x[keep], y[keep], inside[keep]
    if len(x) > CLOUD_MAX_POINTS:
        pick = rng.choice(len(x), CLOUD_MAX_POINTS, replace=False)
        x, y, inside = x[pick], y[pick], inside[pick]
    # Thicker where denser, floating on a gently undulating deck.
    z = (
        terrain.h_max
        + CLOUD_DECK_HEIGHT
        + inside * CLOUD_DECK_THICKNESS * rng.uniform(0.2, 1.0, len(x))
        + 2.5 * fbm(y * 0.5 + 31.0, x * 0.5 - 17.0)
    )
    return np.column_stack([x, y, z]).astype(np.float32)


def _build_cloud_cover(
    rng: np.random.Generator, terrain: Terrain
) -> dict[str, Any] | None:
    """Very light multiscale cloud cover above the forest, one deck per season.

    A single Points node stacked on the ``season`` axis (like the accents):
    the same fBm field is thresholded at a per-season coverage and drifted so
    the sky thins to summer wisps and closes over in winter.
    """
    fbm = FBm(rng, octaves=4, base_cell=42.0, amplitude=1.0, gain=0.5)
    blocks: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    labels: list[str] = []
    for season in range(4):
        xyz = _cloud_deck_for_season(rng, fbm, terrain, season)
        if len(xyz) == 0:
            continue
        n = len(xyz)
        blocks.append(np.column_stack([np.full(n, float(season), np.float32), xyz]))
        tint = np.asarray(CLOUD_TINTS[season], np.float32)
        colors.append(
            _linear_rgb(np.tile(tint, (n, 1)) * rng.uniform(0.85, 1.0, (n, 1)))
        )
        labels.extend([f"Cloud cover · {SEASONS[season].lower()}"] * n)
    if not blocks:
        return None
    positions = np.concatenate(blocks).astype(np.float32)
    return dict(
        positions=positions,
        colors=np.concatenate(colors).astype(np.float32),
        radii=rng.uniform(3.0, 6.0, len(positions)).astype(np.float32),
        labels=labels,
    )


# =============================================================================
# Bundle build (cached) and scene authoring
# =============================================================================


def _expand_label_runs(runs: list[tuple[int, str]]) -> list[str]:
    labels: list[str] = []
    for count, label in runs:
        labels.extend([label] * count)
    return labels


def build_forest_bundle(
    n_trees: int, iterations: int, seed: int = 123
) -> dict[str, Any]:
    """Compute every array the scene needs (the expensive, cacheable part)."""
    rng = np.random.default_rng(seed)

    with asection("Building fBm terrain"):
        terrain = build_terrain(rng)
        aprint(
            f"Terrain: {len(terrain.vertices):,} vertices, "
            f"{len(terrain.faces):,} triangles, "
            f"relief {terrain.h_min:.1f}..{terrain.h_max:.1f} m"
        )

    with asection("Planting trees (Poisson disk + eco-zones)"):
        plans = plant_trees(rng, terrain, n_trees)
        counts = {sp.title: 0 for sp in SPECIES}
        for plan in plans:
            counts[SPECIES[plan.species_index].title] += 1
        aprint(f"Planted {len(plans)} trees:")
        for title, count in counts.items():
            aprint(f"  {title}: {count}")

    final_iterations = iterations - 1
    species_arrays = [SpeciesArrays() for _ in SPECIES]
    foliage = FoliageArrays()

    with asection(
        f"Deriving {len(plans)} trees x {N_STAGES} growth stages x 4 seasons"
    ):
        for i, plan in enumerate(plans):
            species = SPECIES[plan.species_index]
            out = species_arrays[plan.species_index]
            out.n_trees += 1
            _build_tree(
                plan,
                out,
                foliage,
                max(1, final_iterations + species.iter_bonus),
            )
            if i % 100 == 0:
                aprint(f"  tree {i}/{len(plans)}")
        total_segments = sum(out.n_segments for out in species_arrays)
        aprint(
            f"Authored {total_segments:,} segments, {foliage.n_splats:,} "
            "foliage splats (all slots, all seasons)"
        )

    with asection("Scattering accents"):
        accents = _build_accents(rng, terrain, plans)
        for name, accent in accents.items():
            aprint(f"  {name}: {len(accent['positions']):,} points")

    with asection("Drawing seasonal cloud cover"):
        clouds = _build_cloud_cover(rng, terrain)
        if clouds is not None:
            per_season = np.bincount(clouds["positions"][:, 0].astype(int), minlength=4)
            aprint(
                "  Cloud points per season: "
                + ", ".join(
                    f"{SEASONS[i]} {int(n):,}" for i, n in enumerate(per_season)
                )
            )

    species_bundles = {}
    for species, out in zip(SPECIES, species_arrays):
        if out.n_segments == 0:
            continue
        species_bundles[species.key] = dict(
            title=species.title,
            vertices=np.concatenate(out.vertices),
            widths=np.concatenate(out.widths),
            colors=np.concatenate(out.colors),
            sharpness=np.concatenate(out.sharpness),
            edges=np.concatenate(out.edges),
            label_runs=out.label_runs,
            n_segments=out.n_segments,
            n_trees=out.n_trees,
        )

    foliage_bundle = None
    if foliage.n_splats:
        foliage_bundle = dict(
            centers=np.concatenate(foliage.centers),
            cholesky=np.concatenate(foliage.cholesky),
            amplitudes=np.concatenate(foliage.amplitudes),
            colors=np.concatenate(foliage.colors),
            label_runs=foliage.label_runs,
        )

    return dict(
        terrain=dict(
            vertices=terrain.vertices,
            faces=terrain.faces,
            normals=terrain.normals,
            season_colors=terrain.season_colors,
        ),
        species=species_bundles,
        foliage=foliage_bundle,
        accents=accents,
        clouds=clouds,
        total_segments=sum(out.n_segments for out in species_arrays),
    )


def _forest_dimensions() -> Dimensions:
    return Dimensions(
        [
            Dimension(
                "season",
                unit="",
                categories=list(SEASONS),
                display=False,
                description="Season of the year: recolours and re-dresses the "
                "same forest (blossom, green, fire, frost)",
            ),
            Dimension(
                "growth",
                unit="",
                categories=list(GROWTH_STAGES),
                display=False,
                description="Developmental stage: every tree is re-derived at "
                "increasing iteration depth, staggered per tree",
            ),
            Dimension("x", unit="m", display=True),
            Dimension("y", unit="m", display=True),
            Dimension("z", unit="m", display=True),
        ]
    )


def _viewer_config() -> ViewerConfig:
    camera_target = (4.0, 6.0, 4.5)
    view_from = np.asarray((-48.0, -42.0, 14.0))
    view_direction = view_from - np.asarray(camera_target)
    view_direction /= np.linalg.norm(view_direction)
    # Carry over the authored view direction, but solve the standoff from the
    # cinematic lens and planted span so the opening eye stays outside the trees.
    camera_distance = (FOREST_SIZE / 2.0) / math.tan(
        math.radians(CINEMATIC_FOV_DEG / 2.0)
    )
    camera_position = tuple(
        np.asarray(camera_target) + view_direction * camera_distance
    )
    return ViewerConfig(
        cinematic_mode=True,
        # ACES explicitly — the house default; the luminous accents +
        # emissive foliage mix is exactly what its filmic rolloff is for.
        tone_mapping="ACES",
        # Opening pose: low vantage outside the forest edge at canopy height,
        # looking into the field — the pose test locks that clearance.
        camera=CameraConfig(
            position=camera_position,
            target=camera_target,
            up=(0.0, 0.0, 1.0),
            near=0.5,
            far=800.0,
        ),
        bloom_enabled=True,
        bloom_strength=0.09,
        bloom_threshold=0.85,
        # Open on autumn fire at full maturity: the poster shot.
        dimensions=DimensionsConfig(
            current_step=[float(AUTUMN), float(N_STAGES - 1), 0.0, 0.0, 0.0]
        ),
    )


def _grammar_card_html() -> str:
    """A small card with each species' actual production rules — the
    intellectual heart of the demo, shown next to the forest it built."""
    rows = [
        '<div style="font-size:1.15vh;line-height:1.6;font-family:monospace;'
        'background:rgba(0,0,0,0.55);padding:0.6vh;border-radius:3px">'
        '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">'
        "L-system grammars</div>"
    ]
    for species in SPECIES:
        symbol, rule = next(iter(species.lsystem.rules.items()))
        production = rule if isinstance(rule, str) else rule[0][1]
        if len(production) > 30:
            production = production[:28] + "…"
        rows.append(
            f"<div><b>{html.escape(species.title)}</b> "
            f'<span style="opacity:0.75">{html.escape(species.lsystem.axiom)}'
            f" ; {html.escape(symbol)} → {html.escape(production)}</span></div>"
        )
    rows.append("</div>")
    return "".join(rows)


_SEASON_GLYPHS = ("🌸", "☀", "🍂", "❄")


def _add_overlays(scene: Any) -> None:
    scene.add_text(
        "L-System Forest",
        position=(0.02, 0.02),
        font_size=0.05,
        anchor="top-left",
        color="rgba(255,255,255,0.65)",
        blend_mode="difference",
    )
    scene.add_text(
        "a year in a growing forest",
        position=(0.02, 0.085),
        font_size=0.018,
        anchor="top-left",
        color="rgba(255,255,255,0.4)",
        blend_mode="difference",
    )

    for season, name in enumerate(SEASONS):
        scene.add_text(
            f"{_SEASON_GLYPHS[season]} {name}",
            position=(0.02, 0.9),
            font_size=0.022,
            anchor="bottom-left",
            color="#ffcc44",
            visible_range={"season": float(season)},
            transition="fade",
            transition_duration=0.2,
        )
    for growth, name in enumerate(GROWTH_STAGES):
        scene.add_text(
            f"Growth {growth + 1}/{N_STAGES} · {name}",
            position=(0.02, 0.94),
            font_size=0.016,
            anchor="bottom-left",
            color="rgba(220,220,220,0.75)",
            visible_range={"growth": float(growth)},
            transition="fade",
            transition_duration=0.2,
        )

    scene.add_html(
        _grammar_card_html(),
        position=(0.98, 0.5),
        anchor="center-right",
        opacity=0.9,
    )
    add_demo_caption(
        scene,
        "keys 1/2 select season/growth • [ ] step • play animates • hover a trunk",
        DEMO_META.get("citation"),
    )


def _author_scene(scene: Any, bundle: dict[str, Any]) -> None:
    """Write every node of the pre-computed bundle into the scene."""
    terrain = bundle["terrain"]
    n_terrain = len(terrain["vertices"])
    with asection("Adding terrain mesh (4 seasonal dressings)"):
        vertices4 = np.concatenate(
            [
                np.column_stack(
                    [np.full(n_terrain, float(season), np.float32), terrain["vertices"]]
                )
                for season in range(4)
            ]
        ).astype(np.float32)
        faces4 = np.concatenate(
            [
                terrain["faces"].astype(np.int64) + season * n_terrain
                for season in range(4)
            ]
        ).astype(np.uint32)
        # The one node WITHOUT hover labels, deliberately: the ground is
        # under the cursor almost everywhere, and a terrain tooltip would
        # fire constantly and compete with the per-tree labels.
        scene.add_mesh(
            "Terrain",
            vertices4,
            faces4,
            normals=np.tile(terrain["normals"], (4, 1)),
            # The three scene dims the normals describe (x, y, z).
            normal_dims=[2, 3, 4],
            colors=np.concatenate(terrain["season_colors"]),
            shading="smooth",
            dim_order=["season", "x", "y", "z"],
            fill={"growth": 0.0},
            extend_to_all=["growth"],
            layer=True,
        )
        aprint(f"Terrain: {len(faces4):,} triangles across 4 seasons")

    with asection("Adding per-species tree nodes"):
        for key, sp in bundle["species"].items():
            scene.add_lines(
                sp["title"],
                vertices=sp["vertices"],
                widths=sp["widths"],
                colors=sp["colors"],
                sharpness=sp["sharpness"],
                indices=sp["edges"],
                line_type="indexed",
                labels=_expand_label_runs(sp["label_runs"]),
                # `normal`, NOT the default `additive`: a forest needs
                # trunks in front to actually hide trunks behind.
                blending_mode="normal",
                opacity=1.0,
                layer=True,
            )
            aprint(
                f"  {sp['title']}: {sp['n_trees']} trees, {sp['n_segments']:,} segments"
            )

    if bundle["foliage"] is not None:
        with asection("Adding volumetric foliage gsplats"):
            fol = bundle["foliage"]
            scene.add_gsplats(
                "Foliage",
                centers=fol["centers"],
                amplitudes=fol["amplitudes"],
                cholesky_factors=fol["cholesky"],
                colors=fol["colors"],
                labels=_expand_label_runs(fol["label_runs"]),
                # Volumetric: canopies are soft occluding clouds, not
                # additive fuzz. VERY low opacity is the trick — dozens of
                # splats overlap along a view ray, so each may contribute
                # only a sliver or the canopy saturates to a milky veil;
                # absorption is what makes near foliage occlude far foliage.
                blending_mode="volumetric",
                absorption=1.8,
                opacity=0.09,
                layer=True,
            )
            aprint(f"  Foliage: {len(fol['centers']):,} splats")

    with asection("Adding accent point layers"):
        for name, accent in bundle["accents"].items():
            positions = accent["positions"]
            block = np.column_stack(
                [
                    np.full(len(positions), float(accent["season"]), np.float32),
                    positions,
                ]
            )
            scene.add_points(
                name,
                positions=block,
                colors=accent["colors"],
                radii=accent["radii"],
                sharpness=accent["sharpness"],
                labels=[accent["label"]] * len(positions),
                dim_order=["season", "x", "y", "z"],
                fill={"growth": 0.0},
                extend_to_all=["growth"],
                blending_mode=accent["blending_mode"],
                intensity=accent["intensity"],
                layer=True,
            )
            aprint(f"  {name}: {len(positions):,} points ({SEASONS[accent['season']]})")

    clouds = bundle.get("clouds")
    if clouds is not None:
        with asection("Adding seasonal cloud cover"):
            scene.add_points(
                "Cloud Cover",
                positions=clouds["positions"],
                colors=clouds["colors"],
                radii=clouds["radii"],
                # Very soft, very faint: a veil over the sky, not a ceiling.
                sharpness=0.05,
                labels=clouds["labels"],
                dim_order=["season", "x", "y", "z"],
                fill={"growth": 0.0},
                extend_to_all=["growth"],
                blending_mode="additive",
                intensity=CLOUD_INTENSITY,
                layer=True,
            )
            aprint(f"  Cloud Cover: {len(clouds['positions']):,} points (4 seasons)")

    _add_overlays(scene)


def generate_forest(
    output_path: Path,
    iterations: int = 5,
    n_trees: int = 800,
    recompute: bool = False,
    use_cache: Optional[bool] = None,
) -> int:
    """Generate the complete forest scene.

    Args:
        output_path: Destination ``.luxar.zarr`` store.
        iterations: Scene-wide derivation budget; the final growth stage
            derives at ``iterations - 1`` (plus per-species bonus).
        n_trees: Number of trees to plant.
        recompute: Force a rebuild of the cached geometry bundle.
        use_cache: Cache the computed bundle under ``~/.cache/luxar/forest``.
            Default (None) caches only non-toy sizes (>= 64 trees), so tests
            and tiny runs stay out of the user's cache.

    Returns:
        Total number of authored line segments (across all slots).
    """
    if use_cache is None:
        use_cache = n_trees >= 64

    def _build() -> dict[str, Any]:
        return build_forest_bundle(n_trees=n_trees, iterations=iterations)

    if use_cache:
        bundle = cache_computed(
            "forest",
            f"scene_t{n_trees}_i{iterations}",
            _build,
            version=SCENE_VERSION,
            recompute=recompute,
        )
    else:
        bundle = _build()

    with LuxarZarrCompiler(output_path) as compiler:
        scene = compiler.create_scene(
            dimensions=_forest_dimensions(), viewer_config=_viewer_config()
        )
        _author_scene(scene, bundle)

    return int(bundle["total_segments"])


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    flags = parse_demo_flags()
    iterations = parse_int_arg("iterations", 5)
    n_trees = parse_int_arg("trees", 800)

    aprint("=" * 70)
    aprint("L-SYSTEM FOREST — A YEAR IN A GROWING FOREST")
    aprint("=" * 70)
    aprint("")
    aprint("All four geometry types in one synthetic scene:")
    aprint("  Mesh    - fBm heightfield terrain, shaded, snowy in winter")
    aprint("  Lines   - eight tree species as merged indexed-line nodes")
    aprint("  GSplats - volumetric foliage clouds, re-dressed per season")
    aprint("  Points  - fireflies (summer), frost (winter), petals (spring),")
    aprint("            and a faint seasonal cloud deck over the canopy")
    aprint("")
    aprint("Two navigable dimensions:")
    aprint("  growth - six derivation stages, staggered per tree")
    aprint("  season - spring / summer / autumn / winter")
    aprint("")
    aprint(f"Iterations: {iterations} | Trees: {n_trees}")
    aprint("")

    output_path = get_demos_output_dir() / "forest.luxar.zarr"

    if flags["serve_only"] and not output_path.exists():
        aprint("--serve-only but no dataset on disk yet; generating it first.")
        flags = {**flags, "serve_only": False}

    if not flags["serve_only"]:
        if output_path.exists():
            shutil.rmtree(output_path)
        with asection("Generating forest"):
            total_segments = generate_forest(
                output_path,
                iterations=iterations,
                n_trees=n_trees,
                recompute=flags["recompute"],
            )
        aprint("")
        aprint(
            f"FOREST COMPLETE: {total_segments:,} line segments "
            f"(x{N_STAGES} growth stages x 4 seasons)"
        )
        aprint(f"Dataset generated at {output_path}")

    if flags["no_serve"]:
        return

    aprint("")
    aprint("Keys 1/2 select the season/growth dimension, [ ] step it, and the")
    aprint("dimension play button animates the year or the time-lapse.")
    aprint("Browser will open automatically. Press Ctrl+C when done.")
    aprint("")
    launch_viewer(output_path)


if __name__ == "__main__":
    main()
