#!/usr/bin/env python3
"""Self-Contained Demo: Quantum Atomic Orbitals as Gaussian Splats

Hydrogen-atom electron probability density |ψ|², rendered as *volumetric*
Gaussian splats instead of a point cloud. Each orbital is first evaluated on a
regular 3D voxel grid — a genuine analytic volume — and then fitted with
oriented Gaussians by the standard Luxar splat fitter. The result composites
with the ``volumetric`` blending mode, so the orbitals read as translucent
glowing clouds with real depth rather than a haze of discrete dots.

================================================================================
WHY SPLATS (AND WHY VOLUMETRIC)
================================================================================

A probability density is exactly the kind of object Gaussian splats represent
natively: a smooth, everywhere-positive scalar field with unbounded support. The
point-cloud version of this demo had to threshold |ψ|² and scatter
equal-radius spheres over the surviving voxels, which quantizes a continuous
cloud into visible grid structure. Fitting oriented Gaussians instead lets a few
thousand anisotropic ellipsoids absorb the smooth falloff analytically — fewer
primitives, no grid aliasing, and a density field the ``volumetric`` blending
mode integrates along the view ray the way a real emissive/absorbing medium
behaves.

PHASE COLORING
--------------
Splats are tinted by the **sign of the wavefunction** ψ at their center — warm
amber for ψ > 0, cool blue for ψ < 0 — the textbook convention. The density
|ψ|² is sign-blind, so this is the only way to see the nodal structure that
governs bonding: the radial node of 2s, the plane through the nucleus that
separates the two lobes of 2p, the alternating lobes of 3d.

REAL SPHERICAL HARMONICS
------------------------
The orbitals use *real* (tesseral) spherical harmonics, not the complex Y_lm.
This matters: |Y₁¹|² is a torus about z, so a complex-harmonic "2p_x" is not a
dumbbell along x at all. The real combinations are the ones chemistry uses, and
the ones the labels here promise.

TRUE RELATIVE SCALE
-------------------
Every orbital is generated inside its own box, sized to contain 99.5% of the
radial probability, but all eight land in one shared coordinate frame in Bohr
radii. So 1s really is several times smaller than 3d — the size hierarchy that
sets the size of atoms is visible, not normalized away.

Mathematical Background:
    ψ_nlm(r,θ,φ) = R_nl(r) × Y_lm(θ,φ)      (real Y for m ≠ 0)
    Probability density: ρ = |ψ|²

    n: Principal (1, 2, 3, ...) — energy level / size
    l: Angular momentum (0..n-1) — shape (s, p, d, f)
    m: Magnetic (-l..+l) — orientation

CACHING
-------
The fit runs once (~2 min for all eight orbitals at the default 96³ / 25k
seeds / 1500 iters, measured end to end on Apple-silicon MPS) and is cached to
``~/.cache/luxar/quantum_orbitals/``; later runs are instant. The cache file is
keyed by ``--grid`` / ``--seeds`` / ``--iters`` plus a digest of the whole fit
recipe — the orbital table, the tuning constants baked into the splats (box
size, normalization, culling, phase tints) and ``FIT_RECIPE_VERSION``. Changing
any of them refits instead of silently reusing the old fit, while switching back
to a previously-fitted combination is still instant. ``--recompute`` forces a
refit of the current parameter set.

Usage:
    python demo_quantum_orbitals.py [--grid N] [--seeds K] [--iters N]
                                    [--recompute] [--no-serve] [--serve-only]
    python demo_quantum_orbitals.py --recompute --grid 128 --seeds 40000
    python demo_quantum_orbitals.py --no-serve      # generate only, no viewer
    python demo_quantum_orbitals.py --serve-only    # re-serve the last scene

Controls:
    - Press '1' to select ORBITAL, then '['/']' to cycle 1s → 2s → … → 3dxy → 1s
    - Mouse drag: rotate,  Scroll: zoom,  Right-drag: pan,  'C': fly controls
"""

DEMO_META = {
    "key": "quantum_orbitals",
    "title": "Quantum Atomic Orbitals (Gaussian splats)",
    "description": "Hydrogen-atom probability-density orbitals (s/p/d) fitted as volumetric Gaussian splats, phase-colored, with state navigation.",
    "category": "synthetic",
    "geometry": "gsplats",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "optional",
        "local_data": None,
    },
    "caches": ["quantum_orbitals"],
    "outputs": ["quantum_orbitals"],
}

import hashlib
import math
import os
import shutil
from pathlib import Path

import numpy as np
from arbol import Arbol, aprint, asection
from scipy.special import genlaguerre, sph_harm_y

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.utils.demos import (
    detect_device,
    launch_viewer,
    parse_demo_flags,
    parse_int_arg,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEMO_NAME = "quantum_orbitals"
CACHE_DIR = Path.home() / ".cache" / "luxar" / DEMO_NAME

#: (n, l, m, label, quantum-number caption, one-line description).
#: ``m`` indexes the REAL spherical harmonics: +|m| is the cosine (x-like)
#: partner, -|m| the sine (y-like) one. So l=2, m=+2 is d_x²−y² and m=-2 is d_xy.
ORBITALS = [
    (1, 0, 0, "1s", "n=1, l=0, m=0", "Spherical ground state"),
    (2, 0, 0, "2s", "n=2, l=0, m=0", "Sphere with a radial node"),
    (2, 1, 0, "2pz", "n=2, l=1, m=0", "Dumbbell along z"),
    (2, 1, 1, "2px", "n=2, l=1, m=+1", "Dumbbell along x"),
    (3, 1, 0, "3pz", "n=3, l=1, m=0", "Larger dumbbell, one radial node"),
    (3, 2, 0, "3dz²", "n=3, l=2, m=0", "Dumbbell wearing a torus"),
    (3, 2, 1, "3dxz", "n=3, l=2, m=+1", "Cloverleaf in the xz plane"),
    (3, 2, -2, "3dxy", "n=3, l=2, m=-2", "Cloverleaf in the xy plane"),
]

#: Bump when the GENERATING LOGIC changes (wavefunction, fitter call, stacking)
#: in a way that alters the cached splats but leaves every tuning constant
#: below untouched — a value digest cannot see that, so this is the manual half
#: of the cache key. Mirrors the ``version=`` parameter of
#: :func:`luxar.utils.demos.cache_computed`.
FIT_RECIPE_VERSION = 1

#: Fraction of the radial probability each generation box must contain.
RADIAL_CONTAINMENT = 0.995

#: Percentile used to normalize |ψ|² before fitting. Peak density runs far above
#: the 99.9th percentile on the states with a sharp inner shell (measured
#: max/p99.9: 2s 32.6x, 3pz 6.6x, 1s 2.7x; the d states only 1.3x), so clipping
#: there keeps the diffuse cloud in a usable range instead of letting it be
#: crushed by a handful of hot voxels. It costs 0.1% of voxels on every state.
NORM_PERCENTILE = 99.9

#: Post-fit culling retention. Higher than the 0.95 default on purpose: the
#: faint outer cloud IS the orbital's character here, not background to discard.
CULL_RETENTION = 0.995

#: Phase tints (linear RGB). Warm = ψ > 0, cool = ψ < 0.
PHASE_POSITIVE = (1.00, 0.42, 0.18)
PHASE_NEGATIVE = (0.20, 0.52, 1.00)

#: Opening camera direction (normalized on use — a 3/4 view).
CAMERA_DIRECTION = np.array([0.575, 0.436, 0.693], dtype=np.float64)

#: Fraction of the vertical half-FOV the largest orbital should fill, so the
#: opening pose leaves ~18% air around it. The distance is DERIVED from this
#: rather than guessed as a multiple of the scene extent: a perspective camera
#: subtends a sphere of radius R at ``asin(R/D)``, not ``R/D``, and R is the
#: bounding-SPHERE radius (27.3 a₀), not the per-axis extent (26.0 a₀). Sizing
#: off the per-axis extent with the small-angle form put 3pz at 28.5° against a
#: 23.5° half-FOV — screen-filling and clipped at every edge.
CAMERA_FOV_FILL = 0.85

#: The viewer's default vertical field of view, in degrees. Only sets the
#: opening pose; changing the FOV preset in the viewer just re-frames.
VIEWER_FOV_DEGREES = 47.0

FLAGS = parse_demo_flags()
NO_SERVE = FLAGS["no_serve"]
SERVE_ONLY = FLAGS["serve_only"]
RECOMPUTE = FLAGS["recompute"]

GRID_SIZE = parse_int_arg("grid", 96)
SEEDS = parse_int_arg("seeds", 25_000)
ITERS = parse_int_arg("iters", 1_500)

Arbol.max_depth = 4


def recipe_digest() -> str:
    """Short digest of everything baked into the cached splats.

    Part of the cache key, alongside ``--grid`` / ``--seeds`` / ``--iters``.
    Two distinct hazards:

    * The :data:`ORBITALS` table. Dropping a state leaves ``Dimensions``
      declaring fewer categories than the cached data carries, and merely
      REORDERING two states keeps every count identical while pointing every
      on-screen label at the wrong splats. Only quantum numbers and labels are
      hashed — a description edit changes nothing about the fit.
    * The tuning constants. ``RADIAL_CONTAINMENT`` sets the box, and hence every
      coordinate; ``NORM_PERCENTILE`` sets the fit target; ``CULL_RETENTION``
      sets how many splats survive; the phase tints are literally stored in the
      cached colors. Editing any of them in-source leaves the CLI flags
      untouched, so without them here the demo would silently serve geometry or
      colors from the previous recipe until someone remembered ``--recompute``.

    :data:`FIT_RECIPE_VERSION` covers what a value digest cannot: a change to
    the generating *logic* (the wavefunction, the fitter call, the stacking).
    Bump it when you change how the splats are produced.

    Returns:
        First 8 hex characters of a SHA-256 over the recipe.
    """
    table = ";".join(
        f"{n},{l_quantum},{m},{label}" for n, l_quantum, m, label, _q, _d in ORBITALS
    )
    payload = "|".join(
        [
            f"v{FIT_RECIPE_VERSION}",
            table,
            f"containment={RADIAL_CONTAINMENT!r}",
            f"norm={NORM_PERCENTILE!r}",
            f"cull={CULL_RETENTION!r}",
            f"tints={PHASE_POSITIVE!r}{PHASE_NEGATIVE!r}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def css_rgb(tint: tuple[float, float, float]) -> str:
    """Render a linear-RGB tint constant as a CSS ``rgb()`` string.

    Lets the on-screen phase legend derive its swatches from
    :data:`PHASE_POSITIVE` / :data:`PHASE_NEGATIVE` instead of repeating the
    values, so the legend cannot drift from the splats it describes.

    Args:
        tint: RGB components in [0, 1].

    Returns:
        e.g. ``"rgb(255, 107, 46)"``.
    """
    r, g, b = (int(round(c * 255)) for c in tint)
    return f"rgb({r}, {g}, {b})"


def camera_distance_for_radius(radius: float) -> float:
    """Distance at which a sphere of ``radius`` fills :data:`CAMERA_FOV_FILL`.

    A perspective camera subtends a sphere of radius R at ``asin(R/D)`` — NOT
    ``R/D``, and not R measured per-axis. Getting either wrong put the largest
    orbital at 28.5° against a 23.5° half-FOV: screen-filling and clipped on
    every edge.

    Args:
        radius: Bounding-sphere radius of the subject, in scene units.

    Returns:
        Camera distance from the target, in the same units.
    """
    return radius / math.sin(math.radians(CAMERA_FOV_FILL * VIEWER_FOV_DEGREES / 2.0))


def cache_path(grid_size: int, seeds: int, iters: int) -> Path:
    """Cache file for one fit parameter set.

    The fit inputs live in the filename, not just the directory: the
    ``--grid`` / ``--seeds`` / ``--iters`` knobs plus :func:`recipe_digest`,
    which covers the orbital table, the tuning constants that are baked into
    the cached splats, and :data:`FIT_RECIPE_VERSION` for logic changes a value
    digest cannot see. A run with new values must never silently load an old
    one — the convention :func:`luxar.utils.demos.cache_computed` documents.
    Each parameter set keeps its own cache, so switching back and forth stays
    instant.

    Args:
        grid_size: Voxels per axis used to generate each orbital volume.
        seeds: Splat budget per orbital.
        iters: Optimization iterations per orbital.

    Returns:
        Path under :data:`CACHE_DIR`.
    """
    stem = f"orbitals_g{grid_size}_k{seeds}_i{iters}_{recipe_digest()}"
    return CACHE_DIR / f"{stem}.gsplats.zarr.zip"


# =============================================================================
# Quantum mechanics  (pure functions — unit-tested)
# =============================================================================


def hydrogen_radial_wavefunction(
    r: np.ndarray, n: int, l_quantum: int, a0: float = 1.0
) -> np.ndarray:
    """Radial part R_nl(r) of the hydrogen wavefunction (associated Laguerre).

    Args:
        r: Radial coordinates, in the same units as ``a0``.
        n: Principal quantum number (1, 2, 3, ...).
        l_quantum: Angular momentum quantum number (0 to n-1).
        a0: Bohr radius (length unit).

    Returns:
        R_nl(r), same shape as ``r``. Signed — R changes sign at radial nodes.
    """
    rho = 2.0 * r / (n * a0)
    norm = math.sqrt(
        (2.0 / (n * a0)) ** 3
        * math.factorial(n - l_quantum - 1)
        / (2 * n * math.factorial(n + l_quantum))
    )
    laguerre = genlaguerre(n - l_quantum - 1, 2 * l_quantum + 1)
    return norm * np.exp(-rho / 2.0) * (rho**l_quantum) * laguerre(rho)  # type: ignore[no-any-return]


def real_spherical_harmonic(
    l_quantum: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Real (tesseral) spherical harmonic Y_lm(θ, φ).

    The real combinations of the complex Y_l^{±|m|} — the orbitals chemistry
    actually uses. ``m > 0`` gives the cosine (x-like) partner, ``m < 0`` the
    sine (y-like) one, ``m == 0`` the already-real zonal harmonic.

    Args:
        l_quantum: Degree l >= 0.
        m: Order, -l <= m <= l.
        theta: Polar angle (0 at +z).
        phi: Azimuthal angle.

    Returns:
        Real-valued harmonic, same shape as ``theta``.
    """
    if m == 0:
        return np.asarray(sph_harm_y(l_quantum, 0, theta, phi).real)
    Y = sph_harm_y(l_quantum, abs(m), theta, phi)
    sign = math.sqrt(2.0) * (-1.0) ** m
    return np.asarray(sign * (Y.real if m > 0 else Y.imag))


def radial_containment_radius(
    n: int, l_quantum: int, fraction: float = RADIAL_CONTAINMENT
) -> float:
    """Smallest radius containing ``fraction`` of the radial probability.

    Sizes each orbital's generation box from the physics instead of a hand-tuned
    per-n constant, so 1s gets a small tight box and 3d a large one — and the
    boxes stay directly comparable in Bohr radii.

    Args:
        n: Principal quantum number.
        l_quantum: Angular momentum quantum number.
        fraction: Radial probability mass to enclose (0 < fraction < 1).

    Returns:
        Radius in Bohr radii.
    """
    r = np.linspace(1e-6, 40.0 * n, 20_000)
    p = (hydrogen_radial_wavefunction(r, n, l_quantum) ** 2) * r**2
    cumulative = np.cumsum(p)
    cumulative /= cumulative[-1]
    return float(r[np.searchsorted(cumulative, fraction)])


def orbital_wavefunction_volume(
    n: int, l_quantum: int, m: int, grid_size: int
) -> tuple[np.ndarray, float]:
    """Sample the signed wavefunction ψ_nlm on a cubic voxel grid.

    Args:
        n: Principal quantum number.
        l_quantum: Angular momentum quantum number.
        m: Magnetic quantum number (real-harmonic convention).
        grid_size: Voxels per axis (grid_size³ total).

    Returns:
        ``(psi, half_extent)`` — the signed wavefunction cube and the box
        half-width in Bohr radii, so the box spans ``[-half_extent, +half_extent]``
        on every axis.
    """
    half_extent = radial_containment_radius(n, l_quantum)
    coords = np.linspace(-half_extent, half_extent, grid_size)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")

    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(np.clip(Z / (r + 1e-12), -1.0, 1.0))
    phi = np.arctan2(Y, X)

    psi = hydrogen_radial_wavefunction(r, n, l_quantum) * real_spherical_harmonic(
        l_quantum, m, theta, phi
    )
    return psi.astype(np.float32), half_extent


def normalize_density(psi: np.ndarray) -> np.ndarray:
    """Turn a signed wavefunction into a fit-ready [0, 1] density cube.

    Args:
        psi: Signed wavefunction volume.

    Returns:
        ``|ψ|²`` divided by its :data:`NORM_PERCENTILE` percentile and clipped
        to [0, 1].
    """
    rho = psi.astype(np.float32) ** 2
    scale = float(np.percentile(rho, NORM_PERCENTILE))
    if scale <= 0.0:
        scale = float(rho.max()) or 1.0
    return np.clip(rho / scale, 0.0, 1.0).astype(np.float32)


def phase_colors(
    centers: np.ndarray, psi: np.ndarray, half_extent: float
) -> np.ndarray:
    """Tint each splat by the sign of ψ at its center (textbook phase coloring).

    Args:
        centers: Splat centers, shape (N, 3), in Bohr radii with the box origin
            at voxel (0, 0, 0) — i.e. before re-centering on the nucleus.
        psi: The signed wavefunction cube the splats were fitted to.
        half_extent: Box half-width in Bohr radii.

    Returns:
        Float32 RGB array of shape (N, 3).
    """
    grid_size = psi.shape[0]
    voxel = 2.0 * half_extent / (grid_size - 1)
    idx = np.clip(np.rint(centers / voxel).astype(np.int64), 0, grid_size - 1)
    sign_at_center = psi[idx[:, 0], idx[:, 1], idx[:, 2]]

    colors = np.empty((len(centers), 3), dtype=np.float32)
    colors[:] = PHASE_POSITIVE
    colors[sign_at_center < 0.0] = PHASE_NEGATIVE
    return colors


# =============================================================================
# Fitting
# =============================================================================


def fit_orbitals(grid_size: int, seeds: int, iters: int) -> GSplatData:
    """Generate every orbital volume, fit splats, and stack into one 4D dataset.

    Each orbital is fitted independently in its own physical box, phase-colored,
    re-centered on the nucleus, then embedded at its own coordinate along a new
    trailing ``orbital`` axis (sigma = 0, so a splat belongs to exactly one state
    and never bleeds into its neighbours).

    Args:
        grid_size: Voxels per axis for each orbital volume.
        seeds: Splat budget per orbital, before post-fit culling.
        iters: Optimization iterations per orbital.

    Returns:
        A single 4D :class:`GSplatData` with columns ``(x, y, z, orbital)``.

    Raises:
        ValueError: If any fit parameter is out of range. ``grid_size`` needs at
            least 2 voxels per axis — at 1 the voxel pitch ``2h/(grid-1)`` is a
            division by zero, which would otherwise surface as a bare
            ``ZeroDivisionError`` several frames deep.
    """
    from luxar.gsplats import fit_gaussian_splats

    if grid_size < 2:
        raise ValueError(f"--grid must be at least 2 voxels per axis, got {grid_size}")
    if seeds < 1:
        raise ValueError(f"--seeds must be at least 1, got {seeds}")
    if iters < 1:
        raise ValueError(f"--iters must be at least 1, got {iters}")

    # Deliberately NOT calling utils.demos.warn_if_no_cuda_gpu() here, though
    # the heavyweight gsplat demos all do: its banner warns that fitting "can
    # take hours instead of minutes" and points at "shipped precomputed data".
    # Neither applies — these eight 96³ fits measure ~2 min total on MPS and
    # this demo ships no precomputed asset. The fitter still prints its own
    # warning on a genuine CPU fallback, which is the case that actually hurts.
    device = detect_device()
    per_orbital: list[GSplatData] = []

    with asection(f"Fitting {len(ORBITALS)} orbitals ({grid_size}³ voxels each)"):
        aprint(f"Device: {device}   seeds={seeds:,}   iters={iters}")
        for n, l_quantum, m, label, quantum, _desc in ORBITALS:
            with asection(f"{label}  ({quantum})"):
                psi, half_extent = orbital_wavefunction_volume(
                    n, l_quantum, m, grid_size
                )
                density = normalize_density(psi)
                aprint(
                    f"Box: ±{half_extent:.2f} a₀   "
                    f"occupancy: {(density > 0.01).mean():.1%}"
                )

                fitted = fit_gaussian_splats(
                    density,
                    seeds=seeds,
                    n_iters=iters,
                    device=device,
                    verbose=False,
                    cull_retention=CULL_RETENTION,
                    voxel_size=2.0 * half_extent / (grid_size - 1),
                    output_space="real",
                )

                fitted = fitted.with_colors(
                    phase_colors(fitted.centers, psi, half_extent)
                )
                # Box origin → nucleus at the scene origin, so all eight states
                # share one coordinate frame at true relative scale.
                fitted = fitted.translate(np.full(3, -half_extent, dtype=np.float32))

                aprint(f"✓ {len(fitted.amplitudes):,} splats")
                per_orbital.append(fitted)

    with asection("Stacking orbitals into the navigable axis"):
        # sigma=0: each splat lives at exactly one integer orbital coordinate.
        combined = GSplatData.combine_as_new_dimension(per_orbital, sigma=0.0)
        aprint(f"✓ {len(combined.amplitudes):,} splats total, ndim={combined.ndim}")
        return combined


def load_or_build_orbitals() -> GSplatData:
    """Return the fitted 4D orbital stack, fitting and caching it on first run.

    Follows the cache contract :func:`luxar.utils.demos.cache_computed`
    establishes: an unreadable cache is quarantined (``.corrupt``) and refitted
    rather than crashing, and the write is atomic so an interrupted run never
    leaves a truncated file behind. Without the first half, a Ctrl-C during the
    save bricks the demo — every later run re-hits the same half-written zip and
    dies on a raw ``BadZipFile`` traceback with no hint that clearing the cache
    would fix it.
    """
    cache_file = cache_path(GRID_SIZE, SEEDS, ITERS)

    if not RECOMPUTE and cache_file.exists():
        with asection("Loading cached fit"):
            aprint(f"{cache_file}")
            try:
                return GSplatData.load(cache_file)
            except Exception as exc:  # truncated / incompatible archive
                from luxar.utils.download import quarantine_file

                quarantine_file(cache_file, reason=f"unreadable gsplats cache ({exc})")
                aprint("Refitting from scratch.")

    combined = fit_orbitals(GRID_SIZE, SEEDS, ITERS)

    with asection("Caching fit"):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Stage in a sibling directory so the archive's internal zarr name is
        # still derived from the FINAL filename, then rename into place — an
        # atomic operation on the same filesystem.
        staging = CACHE_DIR / f".staging-{os.getpid()}"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            staged = staging / cache_file.name
            combined.save(
                staged,
                encoding_mode=EncodingMode.MEMORY,
                include_fitting_info=True,
                compress="zip",
                zip_deflate=True,
            )
            staged.replace(cache_file)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        aprint(f"✓ {cache_file}")
    return combined


# =============================================================================
# Scene
# =============================================================================


def create_luxar_scene(orbitals: GSplatData, output_path: Path) -> Path:
    """Build the navigable 4D orbital scene.

    Args:
        orbitals: 4D stack from :func:`fit_orbitals` — columns (x, y, z, orbital).
        output_path: Destination ``.luxar.zarr``.

    Returns:
        ``output_path``.
    """
    with asection("Creating Luxar scene"):
        centers_xyz = orbitals.centers[:, :3]
        extent = float(np.abs(centers_xyz).max())
        radius = float(np.linalg.norm(centers_xyz, axis=1).max())
        distance = camera_distance_for_radius(radius)
        eye = CAMERA_DIRECTION / np.linalg.norm(CAMERA_DIRECTION) * distance
        aprint(
            f"Camera: bounding radius {radius:.2f} a₀ → distance {distance:.1f} "
            f"(subtends {math.degrees(math.asin(radius / distance)):.1f}° of the "
            f"{VIEWER_FOV_DEGREES / 2:.1f}° half-FOV)"
        )
        dims = Dimensions(
            [
                Dimension("x", unit="a₀", display=True, range=(-extent, extent)),
                Dimension("y", unit="a₀", display=True, range=(-extent, extent)),
                Dimension("z", unit="a₀", display=True, range=(-extent, extent)),
                Dimension(
                    "orbital",
                    unit="",
                    display=False,
                    discrete=True,
                    step=1.0,
                    range=(0, len(ORBITALS) - 1),
                    categories=[o[3] for o in ORBITALS],
                    cyclic=True,  # wrap around from 3dxy back to 1s
                    description="Hydrogen quantum state |n, l, m⟩",
                ),
            ]
        )

        with LuxarZarrCompiler(
            output_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            # ACES on purpose, not by omission: the density near the nucleus is
            # orders of magnitude above the diffuse cloud, and ACES' highlight
            # rolloff is what keeps the core from clipping into a flat white blob.
            scene = compiler.create_scene(
                dimensions=dims,
                viewer_config=ViewerConfig(
                    tone_mapping="ACES",
                    # A 3/4 view, not the auto-framed head-on one: looking
                    # straight down an axis stacks a cloverleaf's four lobes on
                    # top of each other in projection. Off-axis separates them
                    # and reads the z-aligned dumbbells at the same time. The
                    # distance frames the LARGEST state (3pz), so 1s still reads
                    # as a small bright core — that size gap is the point.
                    camera=CameraConfig(
                        position=(float(eye[0]), float(eye[1]), float(eye[2])),
                        target=(0.0, 0.0, 0.0),
                    ),
                ),
            )
            scene.attrs["title"] = "GSplats: Hydrogen Atom Orbitals"
            scene.attrs["description"] = (
                "Hydrogen |ψ|² for eight quantum states, each evaluated on a 3D "
                "voxel grid and fitted with oriented Gaussians. Splats are tinted "
                "by the sign of ψ. Press '1' then '[' / ']' to change state."
            )

            scene.add_gsplats_from_data(
                name="orbitals",
                result=orbitals,
                opacity=1.0,
                blending_mode="volumetric",
                # Moderate absorption: an orbital is a glowing probability cloud,
                # not occluding tissue. Enough kappa that the near lobe reads in
                # front of the far one; much more (measured up to kappa=4) and the
                # nearest lobe simply masks everything behind it, which destroys
                # the alternating-phase pattern that makes a d orbital legible.
                # Intensity is set below the clipping point on purpose — at 1.6 the
                # lobe cores flatten into featureless white even under ACES.
                absorption=0.6,
                intensity=0.8,
                layer=True,
            )

            # --- Overlays ---
            scene.add_text(
                "Hydrogen Atom Orbitals",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            for orb_id, (_n, _l, _m, label, quantum, desc) in enumerate(ORBITALS):
                scene.add_html(
                    f'<div style="font-size:1.5vh;font-weight:bold;color:#ffcc44">{label}</div>'
                    f'<div style="font-size:1.3vh;color:#aaa">{quantum}</div>'
                    f'<div style="font-size:1.3vh;color:#888;margin-top:0.3vh">{desc}</div>',
                    position=(0.02, 0.97),
                    anchor="bottom-left",
                    visible_range={"orbital": orb_id},
                    transition="fade",
                    transition_duration=0.2,
                )

            # HTML, not add_text: one `color=` paints the whole string, so both
            # swatches came out the same gray and the legend never actually said
            # which tint meant which sign — the only on-screen explanation of
            # this demo's headline feature. Swatch colours are derived from the
            # tint constants so they cannot drift from the splats they describe.
            scene.add_html(
                # nowrap: the overlay box is sized to its anchor, so without it
                # each line breaks after a word or two into a ragged column.
                f'<div style="font-size:1.5vh;color:rgba(200,200,200,0.75);'
                f'white-space:nowrap;text-align:right;line-height:1.5">'
                f'<span style="color:{css_rgb(PHASE_POSITIVE)}">█</span> ψ &gt; 0'
                f'&nbsp;&nbsp;<span style="color:{css_rgb(PHASE_NEGATIVE)}">█</span> ψ &lt; 0'
                f"<br>|ψ|² as volumetric Gaussian splats"
                f"<br>shown at true relative scale</div>",
                # Top-right, NOT the conventional bottom-centre: the Dimension
                # Navigation panel occupies bottom-centre by default and hid
                # this legend completely (it did in the point-cloud version
                # too — the text was in the DOM the whole time, painted behind
                # the panel). Top-right is the one large region no default
                # chrome claims: the rail is left, title top-left, state label
                # bottom-left.
                position=(0.98, 0.02),
                anchor="top-right",
            )

        aprint(f"Scene saved: {output_path}")
        return output_path


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("QUANTUM ATOMIC ORBITALS — GAUSSIAN SPLATS")
    aprint("=" * 70)
    aprint("Hydrogen |ψ|² on a voxel grid → oriented Gaussians → volumetric fog")
    aprint("")
    aprint("  • Real (tesseral) harmonics: 2px really is a dumbbell along x")
    aprint("  • Splats tinted by the sign of ψ — nodal structure made visible")
    aprint("  • All eight states share one frame at true relative scale")
    aprint("")
    aprint("Navigation: press '1' to select ORBITAL, then '[' / ']' to cycle:")
    aprint("  " + " → ".join(o[3] for o in ORBITALS) + " → 1s")
    aprint("")

    output_path = get_demos_output_dir() / "quantum_orbitals.luxar.zarr"

    if SERVE_ONLY:
        if output_path.exists():
            launch_viewer(output_path)
        else:
            aprint(f"No scene at {output_path}. Run without --serve-only first.")
        return

    orbitals = load_or_build_orbitals()
    scene_path = create_luxar_scene(orbitals, output_path)

    if NO_SERVE:
        aprint(f"Dataset generated at {scene_path}")
    else:
        launch_viewer(scene_path)


if __name__ == "__main__":
    main()
