"""Tests for the pure quantum-mechanics helpers in demo_quantum_orbitals.

These cover deterministic analytic functions only — no fitting, no cache, no
scene I/O. The demo is loaded by file path (see test_demo_ppi_flow_field for the
rationale: ``luxar.demos`` is aliased to ``luxar.utils.demos``).

The orientation tests are the point of this file: the demo's whole claim is that
it uses REAL (tesseral) harmonics, so ``2px`` is a dumbbell along x and ``3dxy``
a cloverleaf on the xy diagonals. With the complex ``Y_l^m`` those same labels
would render as tori about z, and nothing else in the pipeline would notice.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_quantum_orbitals.py"


def _load_demo_module():
    name = "_luxar_demo_quantum_orbitals_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
hydrogen_radial_wavefunction = _demo.hydrogen_radial_wavefunction
real_spherical_harmonic = _demo.real_spherical_harmonic
radial_containment_radius = _demo.radial_containment_radius
orbital_wavefunction_volume = _demo.orbital_wavefunction_volume
normalize_density = _demo.normalize_density
phase_colors = _demo.phase_colors
recipe_digest = _demo.recipe_digest
cache_path = _demo.cache_path
camera_distance_for_radius = _demo.camera_distance_for_radius
fit_orbitals = _demo.fit_orbitals
ORBITALS = _demo.ORBITALS


# ---------------------------------------------------------------------------
# Radial wavefunction
# ---------------------------------------------------------------------------


def test_radial_1s_matches_closed_form() -> None:
    """R_10(r) = 2 a0^(-3/2) exp(-r/a0)."""
    r = np.linspace(0.0, 10.0, 64)
    np.testing.assert_allclose(
        hydrogen_radial_wavefunction(r, 1, 0), 2.0 * np.exp(-r), rtol=1e-10
    )


def test_radial_2s_has_a_node_at_two_bohr() -> None:
    """R_20 changes sign exactly once, at r = 2 a0."""
    r = np.linspace(0.05, 12.0, 4000)
    R = hydrogen_radial_wavefunction(r, 2, 0)
    crossings = np.flatnonzero(np.diff(np.sign(R)) != 0)
    assert len(crossings) == 1
    assert r[crossings[0]] == pytest.approx(2.0, abs=0.01)


@pytest.mark.parametrize("n,l_quantum", [(1, 0), (2, 0), (2, 1), (3, 1), (3, 2)])
def test_radial_is_normalized(n: int, l_quantum: int) -> None:
    """∫ |R_nl|² r² dr = 1 over the full radial line."""
    r = np.linspace(1e-9, 80.0, 200_000)
    integral = np.trapezoid(
        hydrogen_radial_wavefunction(r, n, l_quantum) ** 2 * r**2, r
    )
    assert integral == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Real spherical harmonics
# ---------------------------------------------------------------------------


def _sphere_grid(n: int = 200):
    rng = np.random.default_rng(0)
    theta = np.arccos(rng.uniform(-1.0, 1.0, n))
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return theta, phi


def test_real_harmonics_are_real_valued() -> None:
    theta, phi = _sphere_grid()
    for l_quantum in range(4):
        for m in range(-l_quantum, l_quantum + 1):
            Y = real_spherical_harmonic(l_quantum, m, theta, phi)
            assert np.isrealobj(Y), f"l={l_quantum} m={m} is not real-valued"


@pytest.mark.parametrize(
    "l_quantum,m,expected",
    [
        # Angular parts, up to a positive normalization constant.
        (1, 0, lambda x, y, z: z),
        (1, 1, lambda x, y, z: x),
        (1, -1, lambda x, y, z: y),
        (2, 0, lambda x, y, z: 3.0 * z**2 - 1.0),
        (2, 1, lambda x, y, z: x * z),
        (2, -1, lambda x, y, z: y * z),
        (2, 2, lambda x, y, z: x**2 - y**2),
        (2, -2, lambda x, y, z: x * y),
    ],
)
def test_real_harmonic_matches_cartesian_form(l_quantum, m, expected) -> None:
    """Each real harmonic is proportional to its textbook Cartesian form.

    This is what pins ``2px`` to x and ``3dxy`` to the xy diagonals — the complex
    harmonics would pass none of the m != 0 rows here.
    """
    theta, phi = _sphere_grid()
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    Y = real_spherical_harmonic(l_quantum, m, theta, phi)
    target = expected(x, y, z)
    ratio = Y / target
    assert np.allclose(ratio, ratio[0], rtol=1e-9), f"l={l_quantum} m={m} not ∝ target"
    assert ratio[0] > 0.0, f"l={l_quantum} m={m} has an inverted sign"


# ---------------------------------------------------------------------------
# Volume generation
# ---------------------------------------------------------------------------


def test_containment_radius_grows_with_n() -> None:
    radii = [radial_containment_radius(n, 0) for n in (1, 2, 3)]
    assert radii == sorted(radii)
    # 1s is the tight one: 99.5% of its probability is inside ~5 a0.
    assert 3.0 < radii[0] < 6.0


def test_containment_radius_actually_contains_the_requested_mass() -> None:
    r_cut = radial_containment_radius(3, 2, fraction=0.9)
    r = np.linspace(1e-9, 200.0, 400_000)
    density = hydrogen_radial_wavefunction(r, 3, 2) ** 2 * r**2
    inside = np.trapezoid(density[r <= r_cut], r[r <= r_cut])
    total = np.trapezoid(density, r)
    assert inside / total == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize(
    "n,l_quantum,m,axis_of_peak",
    [
        (2, 1, 0, (0.0, 0.0, 1.0)),  # 2pz  → along z
        (2, 1, 1, (1.0, 0.0, 0.0)),  # 2px  → along x
        (3, 2, 0, (0.0, 0.0, 1.0)),  # 3dz² → along z
        (3, 2, 1, (1.0, 0.0, 1.0)),  # 3dxz → xz diagonal
        (3, 2, -2, (1.0, 1.0, 0.0)),  # 3dxy → xy diagonal
    ],
)
def test_density_peaks_on_the_expected_axis(
    n: int, l_quantum: int, m: int, axis_of_peak: tuple[float, float, float]
) -> None:
    """The brightest voxel sits on the direction the orbital's label promises."""
    psi, half_extent = orbital_wavefunction_volume(n, l_quantum, m, 48)
    coords = np.linspace(-half_extent, half_extent, 48)
    peak = np.unravel_index(np.argmax(psi**2), psi.shape)
    direction = np.array([coords[peak[0]], coords[peak[1]], coords[peak[2]]])
    direction /= np.linalg.norm(direction)

    expected = np.array(axis_of_peak) / np.linalg.norm(axis_of_peak)
    # Sign-blind: |psi|^2 peaks on both ends of every lobe pair.
    assert abs(float(direction @ expected)) > 0.95


def test_s_orbitals_are_spherically_symmetric() -> None:
    """An s orbital's density depends on r alone — invariant under axis swaps."""
    psi, _ = orbital_wavefunction_volume(2, 0, 0, 32)
    rho = psi**2
    np.testing.assert_allclose(rho, np.transpose(rho, (1, 2, 0)), rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(rho, rho[::-1], rtol=1e-5, atol=1e-12)


def test_2s_wavefunction_changes_sign_but_1s_does_not() -> None:
    """The radial node of 2s is what the phase coloring exists to show."""
    psi_1s, _ = orbital_wavefunction_volume(1, 0, 0, 32)
    psi_2s, _ = orbital_wavefunction_volume(2, 0, 0, 32)
    assert np.all(psi_1s > 0.0)
    assert psi_2s.min() < 0.0 < psi_2s.max()


# ---------------------------------------------------------------------------
# Fit-ready normalization and phase coloring
# ---------------------------------------------------------------------------


def test_normalize_density_is_bounded_and_nonnegative() -> None:
    psi, _ = orbital_wavefunction_volume(2, 1, 0, 32)
    rho = normalize_density(psi)
    assert rho.dtype == np.float32
    assert rho.min() >= 0.0
    assert rho.max() == pytest.approx(1.0, abs=1e-6)
    # Clipping is mild — most of the cloud survives un-clipped.
    assert (rho >= 1.0).mean() < 0.01


def test_phase_tints_are_actually_warm_and_cool() -> None:
    """Anchor the tints semantically, not against themselves.

    Every other phase assertion compares colors to ``PHASE_POSITIVE`` /
    ``PHASE_NEGATIVE``, so swapping the two constants would slip through all of
    them. The docstring promises "warm amber for ψ > 0, cool blue for ψ < 0" —
    that is the claim pinned here.
    """
    r_pos, _g_pos, b_pos = _demo.PHASE_POSITIVE
    r_neg, _g_neg, b_neg = _demo.PHASE_NEGATIVE
    assert r_pos > b_pos, "PHASE_POSITIVE must be the warm (red-dominant) tint"
    assert b_neg > r_neg, "PHASE_NEGATIVE must be the cool (blue-dominant) tint"


def test_phase_colors_follow_the_sign_of_psi() -> None:
    """Off-grid centers on a fine-grained sign pattern.

    3dxy (not 2pz) and a sub-voxel jitter on purpose: with centers landing
    exactly on grid points and a coarse two-lobe sign field, nearest-voxel
    rounding is unobservable, and ``rint`` → ``floor``/``ceil`` or a
    ``/(grid-1)`` → ``/grid`` pitch error all still pass. Splats never land on
    grid points in practice (measured sub-voxel drift ≈ 0.24 voxels), so this
    mirrors the real input and makes the lookup's arithmetic load-bearing.
    """
    grid_size = 48
    psi, half_extent = orbital_wavefunction_volume(3, 2, -2, grid_size)
    voxel = 2.0 * half_extent / (grid_size - 1)

    coords = np.linspace(-half_extent, half_extent, grid_size)
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    rng = np.random.default_rng(0)
    sample = grid.reshape(-1, 3)[::29]
    jitter = rng.uniform(-0.45, 0.45, sample.shape) * voxel
    centers = (sample + jitter + half_extent).astype(np.float32)

    colors = phase_colors(centers, psi, half_extent)

    # Independent expectation: nearest voxel to the SAME physical point.
    idx = np.clip(np.rint(centers / voxel).astype(int), 0, grid_size - 1)
    expected_negative = psi[idx[:, 0], idx[:, 1], idx[:, 2]] < 0.0

    positive = np.all(colors == np.float32(_demo.PHASE_POSITIVE), axis=1)
    negative = np.all(colors == np.float32(_demo.PHASE_NEGATIVE), axis=1)
    assert np.all(positive | negative)
    assert np.array_equal(negative, expected_negative)
    # The sample must actually straddle both phases, or the assert is vacuous.
    assert 0.2 < negative.mean() < 0.8


def test_phase_colors_use_the_full_box_pitch() -> None:
    """Pin the voxel pitch to ``2h/(grid-1)``, mid-box where nothing can hide it.

    Deliberately NOT tested at the far corner: there the wrong pitch ``2h/grid``
    overshoots to index ``grid``, the clamp pulls it straight back to
    ``grid-1``, and the bug is invisible. Mid-box the clamp is inert, so the
    index difference survives — grid 16 / h 3.0 gives pitch 0.4 vs 0.375, and a
    center at 4.0 lands on voxel 10 correctly but voxel 11 with the wrong pitch.
    """
    grid_size, half_extent = 16, 3.0
    assert round(4.0 / (2 * half_extent / (grid_size - 1))) == 10
    assert round(4.0 / (2 * half_extent / grid_size)) == 11  # the wrong pitch

    psi = np.zeros((grid_size,) * 3, dtype=np.float32)
    psi[10, 10, 10] = -1.0  # only the correctly-indexed voxel is negative
    centers = np.array([[4.0, 4.0, 4.0]], dtype=np.float32)

    colors = phase_colors(centers, psi, half_extent)
    assert np.array_equal(colors[0], np.float32(_demo.PHASE_NEGATIVE))


def test_phase_colors_far_corner_lands_on_the_last_voxel() -> None:
    """The box's far corner maps to ``grid-1`` — the upper boundary case."""
    grid_size, half_extent = 16, 3.0
    psi = np.zeros((grid_size,) * 3, dtype=np.float32)
    psi[-1, -1, -1] = -1.0
    far_corner = np.array([[2.0 * half_extent] * 3], dtype=np.float32)

    colors = phase_colors(far_corner, psi, half_extent)
    assert np.array_equal(colors[0], np.float32(_demo.PHASE_NEGATIVE))


def test_phase_colors_treat_exact_zero_as_positive() -> None:
    """The nodal surface itself (ψ == 0) takes the warm tint, per ``< 0.0``."""
    psi = np.zeros((4, 4, 4), dtype=np.float32)
    colors = phase_colors(np.zeros((3, 3), dtype=np.float32), psi, 1.0)
    assert np.all(colors == np.float32(_demo.PHASE_POSITIVE))


def test_phase_colors_clamp_out_of_box_centers() -> None:
    """A splat nudged outside the box during fitting must not index out of range."""
    psi, half_extent = orbital_wavefunction_volume(1, 0, 0, 16)
    centers = np.array([[-5.0, -5.0, -5.0], [1e6, 1e6, 1e6]], dtype=np.float32)
    colors = phase_colors(centers, psi, half_extent)
    assert colors.shape == (2, 3)
    assert np.isfinite(colors).all()


# ---------------------------------------------------------------------------
# Demo configuration
# ---------------------------------------------------------------------------


def test_orbital_table_uses_valid_quantum_numbers() -> None:
    for n, l_quantum, m, label, _quantum, _desc in ORBITALS:
        assert n >= 1, label
        assert 0 <= l_quantum <= n - 1, label
        assert -l_quantum <= m <= l_quantum, label


def test_orbital_labels_are_unique() -> None:
    labels = [o[3] for o in ORBITALS]
    assert len(set(labels)) == len(labels)


# ---------------------------------------------------------------------------
# Cache keying
# ---------------------------------------------------------------------------


def test_recipe_digest_is_stable_hex() -> None:
    digest = recipe_digest()
    assert len(digest) == 8
    assert all(c in "0123456789abcdef" for c in digest)
    assert digest == recipe_digest(), "digest must not vary between calls"


def test_recipe_digest_tracks_table_identity_and_order(monkeypatch) -> None:
    """Editing OR reordering the table must change the digest.

    The reorder case is the dangerous one: it keeps every count identical, so
    without the digest a reordered table silently reuses the old fit and every
    on-screen label points at the wrong splats.
    """
    baseline = recipe_digest()

    monkeypatch.setattr(_demo, "ORBITALS", ORBITALS[:-1])
    assert recipe_digest() != baseline, "dropping a state must change the digest"

    reordered = list(ORBITALS)
    reordered[2], reordered[3] = reordered[3], reordered[2]
    monkeypatch.setattr(_demo, "ORBITALS", reordered)
    assert recipe_digest() != baseline, "reordering must change the digest"

    monkeypatch.setattr(_demo, "ORBITALS", list(ORBITALS))
    assert recipe_digest() == baseline, "an identical table must reuse the cache"


def test_recipe_digest_ignores_description_edits(monkeypatch) -> None:
    """Prose-only edits must NOT force a multi-minute refit."""
    baseline = recipe_digest()
    reworded = [(*o[:5], o[5] + " (reworded)") for o in ORBITALS]
    monkeypatch.setattr(_demo, "ORBITALS", reworded)
    assert recipe_digest() == baseline


def test_recipe_digest_covers_labels(monkeypatch) -> None:
    """A label rename changes the digest — the documented, conservative choice.

    Labels do not affect the fitted geometry, so hashing them can force a
    refit that isn't strictly required. That is the deliberate trade: labels
    become the scene's dimension categories, so a rename usually rides along
    with a semantic change, and a spurious refit is far cheaper than silently
    serving splats under the wrong name.
    """
    baseline = recipe_digest()
    renamed = [(*ORBITALS[0][:3], "1s_renamed", *ORBITALS[0][4:]), *ORBITALS[1:]]
    monkeypatch.setattr(_demo, "ORBITALS", renamed)
    assert recipe_digest() != baseline


@pytest.mark.parametrize(
    "attr,value",
    [
        ("RADIAL_CONTAINMENT", 0.99),
        ("NORM_PERCENTILE", 99.5),
        ("CULL_RETENTION", 0.95),
        ("PHASE_POSITIVE", (0.9, 0.4, 0.2)),
        ("PHASE_NEGATIVE", (0.2, 0.5, 0.9)),
        ("FIT_RECIPE_VERSION", 2),
    ],
)
def test_recipe_digest_covers_everything_baked_into_the_cache(
    monkeypatch, attr, value
) -> None:
    """Constants that change the cached splats must change the cache key.

    None of these are CLI flags, so editing one in-source leaves
    ``--grid``/``--seeds``/``--iters`` untouched. The phase tints are the
    starkest case: they are stored inside the cached ``GSplatData`` colors, so a
    tint edit with an unchanged key would serve the previous colors forever.
    """
    baseline = recipe_digest()
    monkeypatch.setattr(_demo, attr, value)
    assert recipe_digest() != baseline, f"{attr} must participate in the cache key"


def test_cache_path_actually_embeds_the_digest(monkeypatch) -> None:
    """Tie the digest to the PATH, not just to its own function.

    Without this, deleting ``recipe_digest()`` from the filename template
    leaves every other cache test green while restoring the silent-stale-reuse
    bug the digest exists to prevent.
    """
    baseline = cache_path(96, 25000, 1500)
    assert recipe_digest() in baseline.name

    monkeypatch.setattr(_demo, "ORBITALS", ORBITALS[:-1])
    assert cache_path(96, 25000, 1500) != baseline, (
        "editing ORBITALS must change the cache file, not just the digest"
    )


@pytest.mark.parametrize(
    "a,b",
    [
        ((96, 25000, 1500), (128, 25000, 1500)),  # --grid
        ((96, 25000, 1500), (96, 60000, 1500)),  # --seeds
        ((96, 25000, 1500), (96, 25000, 3000)),  # --iters
    ],
)
def test_cache_path_distinguishes_every_fit_parameter(a, b) -> None:
    """Each knob that changes the fit must change the cache file."""
    assert cache_path(*a) != cache_path(*b)


def test_cache_path_is_stable_and_inside_the_cache_dir() -> None:
    p = cache_path(96, 25000, 1500)
    assert p == cache_path(96, 25000, 1500)
    assert p.parent == _demo.CACHE_DIR
    assert p.name.endswith(".gsplats.zarr.zip")
    # Only ints and a hex digest reach the name — nothing that could escape the dir.
    assert "/" not in p.name and ".." not in p.name


# ---------------------------------------------------------------------------
# Fit-parameter guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid,seeds,iters,expected",
    [
        (1, 25000, 1500, "--grid"),
        (0, 25000, 1500, "--grid"),
        (-4, 25000, 1500, "--grid"),
        (96, 0, 1500, "--seeds"),
        (96, 25000, 0, "--iters"),
    ],
)
def test_fit_orbitals_rejects_out_of_range_parameters(
    grid, seeds, iters, expected
) -> None:
    """Fail fast and legibly — grid=1 used to die on a bare ZeroDivisionError."""
    with pytest.raises(ValueError, match=expected):
        fit_orbitals(grid, seeds, iters)


# ---------------------------------------------------------------------------
# Opening camera
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("radius", [1.0, 4.34, 27.32, 1000.0])
def test_camera_distance_frames_the_subject(radius: float) -> None:
    """The subject must subtend strictly less than the half-FOV, at any scale."""
    distance = camera_distance_for_radius(radius)
    subtended = math.degrees(math.asin(radius / distance))
    half_fov = _demo.VIEWER_FOV_DEGREES / 2.0
    assert subtended < half_fov, f"radius {radius} would be clipped"
    assert subtended == pytest.approx(_demo.CAMERA_FOV_FILL * half_fov)


def test_camera_distance_beats_the_extent_multiple_that_shipped_clipped() -> None:
    """Regression guard for the framing bug this demo actually shipped with.

    The first version set ``distance = 2.2 * extent`` where ``extent`` was the
    per-AXIS maximum (26.04 a₀). The subject a perspective camera really
    subtends is the bounding SPHERE (27.32 a₀), and at distance 57.3 that is
    28.5° against a 23.5° half-FOV — 3pz filled the frame and clipped on every
    edge. Both mistakes are pinned here: the wrong radius and the wrong rule.
    """
    per_axis_extent, bounding_radius = 26.04, 27.32
    half_fov = _demo.VIEWER_FOV_DEGREES / 2.0

    shipped = 2.2 * per_axis_extent
    assert math.degrees(math.asin(bounding_radius / shipped)) > half_fov, (
        "the historical formula must be demonstrably clipping, "
        "or this regression guard proves nothing"
    )

    fixed = camera_distance_for_radius(bounding_radius)
    assert fixed > shipped
    assert math.degrees(math.asin(bounding_radius / fixed)) < half_fov


def test_camera_direction_is_offaxis_on_all_three_axes() -> None:
    """A 3/4 view: head-on down any axis stacks a cloverleaf's lobes."""
    d = _demo.CAMERA_DIRECTION
    assert d.shape == (3,)
    assert np.all(np.abs(d) > 0.1), "no component may be ~0 (that is an axis view)"
