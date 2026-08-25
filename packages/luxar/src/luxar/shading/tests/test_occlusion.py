"""Tests for :mod:`luxar.shading.occlusion`.

The comparative tests pass an explicit ``extinction`` rather than leaning on
``"auto"``. Auto-calibration pins the *median of the population it was given*, so
two separate calls land on two different scales and a cross-call comparison would
be measuring the calibration instead of the geometry. Where the calibration
itself is under test it is named as such.
"""

import numpy as np
import pytest

from luxar.shading import (
    bake_ambient_occlusion,
    directional_optical_depth,
    sphere_directions,
)
from luxar.shading import occlusion as occlusion_module
from luxar.shading.occlusion import AUTO_TARGET_TRANSMITTANCE


def _ball(n: int, seed: int = 0, radius: float = 1.0) -> np.ndarray:
    """Uniformly sampled solid ball."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n * 3, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    scale = rng.random(len(pts)) ** (1.0 / 3.0)
    pts *= scale[:, None] * radius
    return pts[:n]


def _truncated_ball(n: int, seed: int = 0) -> np.ndarray:
    """A ball with a flat face cut off, so its occlusion is not radially symmetric."""
    pts = _ball(n * 2, seed=seed)
    return pts[pts[:, 0] < 0.5][:n]


def _grid_plane(x_range, y_range, step, z=0.0):
    xs = np.arange(x_range[0], x_range[1] + 1e-9, step)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z, dtype=float)])


#: Trench wall offset in y. The grid must resolve this, so keep the scene's total
#: extent small enough that ``extent / grid_cells`` stays well under it — an
#: 8-unit-wide scene at 64 cells gives 0.125 cells and cannot see a 0.15 trench
#: at all.
_WALL_OFFSET = 0.15


def _sheet_and_trench(step: float = 0.025):
    """An open flat sheet and an identically-floored trench, side by side.

    Built in ONE array so a single bake covers both: they then share one
    extinction, one grid and one reference density, and the only thing that
    differs between the two probe points is the shape of their surroundings.

    Returns:
        ``(positions, sheet_index, trench_index)``.
    """
    sheet = _grid_plane((-1.6, -0.6), (-0.5, 0.5), step)
    trench_floor = _grid_plane((0.6, 1.6), (-0.5, 0.5), step)
    walls = [
        _grid_plane((0.6, 1.6), (y_wall, y_wall), step, z=z)
        for y_wall in (-_WALL_OFFSET, _WALL_OFFSET)
        for z in np.arange(step, 0.5, step)
    ]
    positions = np.vstack([sheet, trench_floor, *walls])

    sheet_index = int(np.argmin(np.linalg.norm(positions - [-1.1, 0.0, 0.0], axis=1)))
    trench_index = int(np.argmin(np.linalg.norm(positions - [1.1, 0.0, 0.0], axis=1)))
    return positions, sheet_index, trench_index


# ---------------------------------------------------------------------------
# The discriminating property: geometry, not local count
# ---------------------------------------------------------------------------


def test_trench_is_darker_than_flat_sheet():
    """A point in a trench is more occluded than one on an open sheet.

    Both regions are built in ONE call, so both share a single extinction and a
    single grid; the only difference between them is shape. This is the property
    the whole module exists for — an occlusion term that responded to depth alone
    would score these two the same.
    """
    positions, sheet_centre, trench_centre = _sheet_and_trench()
    shade = bake_ambient_occlusion(
        positions, radius=0.4, extinction=0.6, grid_cells=96, n_directions=12
    )

    assert shade[trench_centre] < shade[sheet_centre] - 0.05


def test_ball_interior_is_darker_than_its_shell():
    positions = _ball(40_000, seed=1)
    shade = bake_ambient_occlusion(
        positions, radius=0.35, extinction=0.5, grid_cells=48, n_directions=12
    )

    distance = np.linalg.norm(positions, axis=1)
    core = shade[distance < 0.3]
    shell = shade[distance > 0.9]

    assert core.mean() < shell.mean() - 0.1
    assert shell.mean() <= 1.0


def test_isolated_element_beyond_radius_is_unoccluded():
    """The window is finite, so mass further away than ``radius`` cannot darken."""
    blob = _ball(20_000, seed=2, radius=0.5)
    lonely = np.array([[8.0, 0.0, 0.0]])
    positions = np.vstack([blob, lonely])

    shade = bake_ambient_occlusion(
        positions, radius=0.4, extinction=1.5, grid_cells=64, n_directions=12
    )

    assert shade[-1] == pytest.approx(1.0, abs=1e-4)
    assert shade[:-1].min() < 0.9


# ---------------------------------------------------------------------------
# View independence — the claim that makes baking legitimate
# ---------------------------------------------------------------------------


def test_occlusion_is_rotation_invariant():
    """Rotating the data must not change how any element is shaded.

    This is what the fixed isotropic cell size buys. Sizing cells from each
    direction's own rotated bounding box would make the answer depend on the
    arbitrary orientation the caller happened to store.
    """
    positions = _truncated_ball(30_000, seed=3)

    angle = 0.7
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    cos, sin = np.cos(angle), np.sin(angle)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = cos * np.eye(3) + sin * cross + (1.0 - cos) * np.outer(axis, axis)

    kwargs = dict(radius=0.3, extinction=0.3, grid_cells=48, n_directions=12)
    shade = bake_ambient_occlusion(positions, **kwargs)
    shade_rotated = bake_ambient_occlusion(positions @ rotation.T, **kwargs)

    # Judged on ABSOLUTE deviation, which is what the eye sees. A correlation
    # floor is the wrong yardstick on its own: it is variance-normalized, and the
    # field's own variance shrinks with resolution, so a correlation threshold
    # would reward a coarser grid even where absolute agreement is worse.
    assert np.abs(shade - shade_rotated).mean() < 0.02
    assert np.abs(shade - shade_rotated).max() < 0.09
    assert np.corrcoef(shade, shade_rotated)[0, 1] > 0.95


def test_translation_does_not_change_occlusion():
    positions = _truncated_ball(20_000, seed=4)
    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=48, n_directions=6)

    shade = bake_ambient_occlusion(positions, **kwargs)
    shifted = bake_ambient_occlusion(positions + [100.0, -50.0, 7.5], **kwargs)

    assert np.abs(shade - shifted).mean() < 0.02


# ---------------------------------------------------------------------------
# Optional normals: hemisphere weighting
# ---------------------------------------------------------------------------


def test_normals_restrict_occlusion_to_the_facing_hemisphere():
    """A sheet's exposed side must brighten once it knows which way it faces.

    Two stacked sheets: every element of the lower one has the upper one directly
    above it, and open space below. Full-sphere AO averages both and darkens
    everything; pointing the normals DOWN, away from the neighbour, must recover
    most of the light.
    """
    step = 0.02
    lower = _grid_plane((-0.5, 0.5), (-0.5, 0.5), step, z=0.0)
    upper = _grid_plane((-0.5, 0.5), (-0.5, 0.5), step, z=0.06)
    positions = np.vstack([lower, upper])

    facing_away = np.tile([0.0, 0.0, -1.0], (len(positions), 1)).astype(float)
    facing_away[len(lower) :] = [0.0, 0.0, 1.0]

    kwargs = dict(radius=0.2, extinction=0.6, grid_cells=96, n_directions=24)
    sphere = bake_ambient_occlusion(positions, **kwargs)
    hemisphere = bake_ambient_occlusion(positions, normals=facing_away, **kwargs)

    assert hemisphere.mean() > sphere.mean() + 0.05


def test_normals_facing_the_occluder_are_darker_than_facing_away():
    """The weighting must follow the normal's direction, not merely exist."""
    step = 0.02
    lower = _grid_plane((-0.5, 0.5), (-0.5, 0.5), step, z=0.0)
    upper = _grid_plane((-0.5, 0.5), (-0.5, 0.5), step, z=0.06)
    positions = np.vstack([lower, upper])

    kwargs = dict(radius=0.2, extinction=0.6, grid_cells=96, n_directions=24)
    toward = np.tile([0.0, 0.0, 1.0], (len(positions), 1)).astype(float)
    away = np.tile([0.0, 0.0, -1.0], (len(positions), 1)).astype(float)

    # Judged on the LOWER sheet only, whose occluder is unambiguously above it.
    facing_up = bake_ambient_occlusion(positions, normals=toward, **kwargs)
    facing_down = bake_ambient_occlusion(positions, normals=away, **kwargs)

    assert facing_up[: len(lower)].mean() < facing_down[: len(lower)].mean() - 0.05


def test_degenerate_normals_fall_back_to_the_full_sphere():
    """A vanishing gradient must not divide by zero or blank the element."""
    positions = _ball(15_000, seed=23)
    normals = np.zeros((len(positions), 3))

    kwargs = dict(radius=0.3, extinction=0.5, grid_cells=48, n_directions=6)
    zeroed = bake_ambient_occlusion(positions, normals=normals, **kwargs)
    sphere = bake_ambient_occlusion(positions, **kwargs)

    assert np.all(np.isfinite(zeroed))
    np.testing.assert_allclose(zeroed, sphere, atol=1e-6)


def test_normals_need_not_be_unit_length():
    positions = _truncated_ball(10_000, seed=24)
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)

    kwargs = dict(radius=0.3, extinction=0.5, grid_cells=48, n_directions=6)
    unit = bake_ambient_occlusion(positions, normals=normals, **kwargs)
    scaled = bake_ambient_occlusion(positions, normals=normals * 7.5, **kwargs)

    np.testing.assert_allclose(unit, scaled, atol=1e-6)


def test_rejects_mismatched_normals():
    positions = _ball(100, seed=25)
    with pytest.raises(ValueError, match="normals must have shape"):
        bake_ambient_occlusion(positions, normals=np.zeros((100, 2)))


def test_normals_are_grouped_alongside_positions():
    """``group_by`` must slice normals in step with positions, not drop them."""
    ball = _ball(8_000, seed=26)
    shifted = ball + [0.7, 0.0, 0.0]
    positions = np.vstack([ball, shifted])
    normals = np.tile([0.0, 0.0, 1.0], (len(positions), 1)).astype(float)
    groups = np.concatenate([np.zeros(len(ball), int), np.ones(len(ball), int)])

    kwargs = dict(radius=0.3, extinction=0.5, grid_cells=48, n_directions=6)
    single = bake_ambient_occlusion(ball, normals=normals[: len(ball)], **kwargs)
    grouped = bake_ambient_occlusion(
        positions, normals=normals, group_by=groups, **kwargs
    )

    np.testing.assert_allclose(grouped[: len(ball)], single, atol=1e-6)
    np.testing.assert_allclose(grouped[len(ball) :], single, atol=1e-6)


# ---------------------------------------------------------------------------
# Auto calibration
# ---------------------------------------------------------------------------


def test_auto_extinction_is_independent_of_sampling_density():
    """The documented reason ``"auto"`` is the default.

    The same object sampled four times as densely must shade the same, or every
    bake would have to be retuned whenever the point budget changed.
    """
    sparse = _truncated_ball(15_000, seed=5)
    dense = _truncated_ball(60_000, seed=5)

    kwargs = dict(radius=0.3, grid_cells=48, n_directions=12)
    shade_sparse = bake_ambient_occlusion(sparse, **kwargs)
    shade_dense = bake_ambient_occlusion(dense, **kwargs)

    quantiles = [10, 50, 90]
    sparse_q = np.percentile(shade_sparse, quantiles)
    dense_q = np.percentile(shade_dense, quantiles)
    assert np.abs(sparse_q - dense_q).max() < 0.08


def test_auto_extinction_is_independent_of_mass_units():
    positions = _truncated_ball(20_000, seed=6)
    mass = np.abs(np.random.default_rng(6).normal(size=len(positions))) + 0.1

    kwargs = dict(radius=0.3, grid_cells=48, n_directions=6)
    shade = bake_ambient_occlusion(positions, mass=mass, **kwargs)
    shade_scaled = bake_ambient_occlusion(positions, mass=mass * 1000.0, **kwargs)

    assert np.abs(shade - shade_scaled).max() < 0.02


def test_auto_extinction_hits_its_declared_target():
    """The realized median must track :data:`AUTO_TARGET_TRANSMITTANCE`.

    Derived from the constant rather than hardcoded, so retuning the target
    cannot leave this test silently asserting the old value. Jensen's inequality
    on the exponential puts the realized median somewhat ABOVE the target, so the
    band is one-sided-ish and generous; what it pins is that the calibration is
    aimed at the declared number and lands nowhere near either rail.
    """
    positions = _ball(20_000, seed=7)
    shade = bake_ambient_occlusion(
        positions, radius=0.3, grid_cells=48, n_directions=12, strength=1.0
    )

    median = float(np.median(shade))
    assert AUTO_TARGET_TRANSMITTANCE - 0.05 < median < AUTO_TARGET_TRANSMITTANCE + 0.25
    assert 0.05 < median < 0.95


def test_lowering_the_auto_target_raises_contrast(monkeypatch):
    """The lever the target exists to provide, and its failure mode.

    Contrast must rise as the target falls — that is the whole point of the
    constant — but the 5th percentile must not be what is paying for it. A target
    low enough to crush the dark end to black shows more "contrast" while showing
    less structure, so both are asserted together.
    """
    positions = _truncated_ball(20_000, seed=27)
    kwargs = dict(radius=0.3, grid_cells=48, n_directions=12, strength=1.0)

    monkeypatch.setattr(occlusion_module, "AUTO_TARGET_TRANSMITTANCE", 0.5)
    timid = bake_ambient_occlusion(positions, **kwargs)
    monkeypatch.setattr(occlusion_module, "AUTO_TARGET_TRANSMITTANCE", 0.25)
    bold = bake_ambient_occlusion(positions, **kwargs)

    assert bold.std() / bold.mean() > 1.3 * (timid.std() / timid.mean())
    assert float(np.percentile(bold, 5)) > 0.1


def test_auto_extinction_on_data_sparser_than_the_radius():
    """Nothing within reach to occlude with must not divide by zero."""
    rng = np.random.default_rng(8)
    positions = rng.uniform(-100.0, 100.0, size=(12, 3))
    shade = bake_ambient_occlusion(positions, radius=0.01, grid_cells=64)
    assert np.all(np.isfinite(shade))
    assert shade == pytest.approx(np.ones(len(positions)), abs=1e-5)


# ---------------------------------------------------------------------------
# nD safety: occlusion must not cross a non-spatial axis
# ---------------------------------------------------------------------------


def test_group_by_isolates_timepoints():
    """Each timepoint must shade as its own object, not as the union of all of them.

    The two timepoints are OFFSET rather than coincident, and that is the point:
    exact duplicates are indistinguishable by construction, because occupancy is
    normalized by the mean mass of an occupied cell and duplicating every element
    scales numerator and reference together. Offsetting by less than the ball
    diameter makes the merged pair genuinely overlap, so the guard has something
    real to prevent.
    """
    ball = _ball(15_000, seed=9)
    shifted = ball + [0.7, 0.0, 0.0]
    positions = np.vstack([ball, shifted])
    groups = np.concatenate([np.zeros(len(ball), int), np.ones(len(ball), int)])

    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=48, n_directions=6)
    single = bake_ambient_occlusion(ball, **kwargs)
    grouped = bake_ambient_occlusion(positions, group_by=groups, **kwargs)
    merged = bake_ambient_occlusion(positions, **kwargs)

    # Each group reproduces the lone ball exactly — the other timepoint is invisible.
    np.testing.assert_allclose(grouped[: len(ball)], single, atol=1e-6)
    np.testing.assert_allclose(grouped[len(ball) :], single, atol=1e-6)

    # And the guard is load-bearing: without it the overlap region cross-occludes,
    # so the merged bake is measurably darker than the isolated one.
    overlap = ball[:, 0] > 0.35
    assert merged[: len(ball)][overlap].mean() < single[overlap].mean() - 0.02


def test_spatial_dims_selects_the_occluding_axes():
    """A 4D node's time column must be excluded, not treated as a third axis."""
    ball = _ball(12_000, seed=10)
    time_column = np.repeat([0.0, 5.0], len(ball) // 2)[: len(ball)]
    positions_4d = np.column_stack([ball, time_column])

    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=48, n_directions=6)
    from_3d = bake_ambient_occlusion(ball, **kwargs)
    from_4d = bake_ambient_occlusion(positions_4d, spatial_dims=(0, 1, 2), **kwargs)

    np.testing.assert_allclose(from_4d, from_3d, atol=1e-6)


def test_spatial_dims_accepts_a_non_leading_triple():
    ball = _ball(8_000, seed=11)
    padded = np.column_stack([np.zeros(len(ball)), ball])

    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=32, n_directions=6)
    np.testing.assert_allclose(
        bake_ambient_occlusion(padded, spatial_dims=(1, 2, 3), **kwargs),
        bake_ambient_occlusion(ball, **kwargs),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Look knobs
# ---------------------------------------------------------------------------


def test_strength_zero_is_a_no_op():
    positions = _ball(5_000, seed=12)
    shade = bake_ambient_occlusion(positions, strength=0.0, grid_cells=32)
    np.testing.assert_allclose(shade, np.ones(len(positions)), atol=1e-6)


def test_strength_scales_the_darkening_monotonically():
    positions = _ball(15_000, seed=13)
    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=48, n_directions=6)
    weak = bake_ambient_occlusion(positions, strength=0.3, **kwargs)
    strong = bake_ambient_occlusion(positions, strength=0.9, **kwargs)
    assert strong.mean() < weak.mean()
    assert np.all(strong <= weak + 1e-6)


def test_floor_clamps_the_darkest_elements():
    positions = _ball(15_000, seed=14)
    shade = bake_ambient_occlusion(
        positions,
        radius=0.5,
        extinction=20.0,
        grid_cells=48,
        n_directions=6,
        strength=1.0,
        floor=0.25,
    )
    assert shade.min() == pytest.approx(0.25, abs=1e-6)


def test_radius_sets_the_scale_of_structure_detected():
    """Radius selects WHICH structure registers, not just how dark things get.

    The trench walls stand ``_WALL_OFFSET`` from its floor, so a radius well
    inside that cannot reach them and the trench stops reading as enclosed at
    all — the contrast against the open sheet collapses by more than an order of
    magnitude, while both points merely get uniformly brighter.
    """
    positions, sheet_centre, trench_centre = _sheet_and_trench()

    kwargs = dict(extinction=0.6, grid_cells=96, n_directions=12)
    wide = bake_ambient_occlusion(positions, radius=0.4, **kwargs)
    narrow = bake_ambient_occlusion(positions, radius=_WALL_OFFSET / 3.0, **kwargs)

    wide_contrast = wide[sheet_centre] - wide[trench_centre]
    narrow_contrast = narrow[sheet_centre] - narrow[trench_centre]
    assert wide_contrast > 0.05
    assert narrow_contrast < wide_contrast / 2.0


def test_a_wider_window_never_lets_in_more_light():
    """More path means more material, never less — the column is a sum.

    Monotone in ``radius`` for every element, not merely on average: a windowed
    sum can only grow as the window widens, so no element may brighten.
    """
    positions = _ball(20_000, seed=15)
    kwargs = dict(extinction=0.5, grid_cells=48, n_directions=6)
    near = bake_ambient_occlusion(positions, radius=0.1, **kwargs)
    far = bake_ambient_occlusion(positions, radius=0.6, **kwargs)

    assert far.mean() < near.mean()
    assert np.all(far <= near + 1e-6)


def test_uniform_mass_rescale_is_a_no_op():
    """Scaling every element's mass equally must not change the shading.

    A property of the occupancy normalization, not just of ``"auto"`` — hence the
    explicit extinction. Without it, ``mass`` would silently double as a
    brightness knob and every dataset would need its own extinction.
    """
    positions = _ball(20_000, seed=16)
    kwargs = dict(radius=0.3, extinction=1.5, grid_cells=48, n_directions=6)
    light = bake_ambient_occlusion(
        positions, mass=np.full(len(positions), 0.1), **kwargs
    )
    heavy = bake_ambient_occlusion(
        positions, mass=np.full(len(positions), 10.0), **kwargs
    )
    np.testing.assert_allclose(heavy, light, atol=1e-6)


def test_heavier_material_occludes_more():
    """Mass is relative: within one bake, dense material casts more occlusion.

    Two separated balls in a single call, so both share one reference density.
    The heavy one's cells sit above that reference and the light one's below.
    """
    light_ball = _ball(20_000, seed=17)
    heavy_ball = _ball(20_000, seed=18) + [5.0, 0.0, 0.0]
    positions = np.vstack([light_ball, heavy_ball])
    mass = np.concatenate(
        [np.full(len(light_ball), 1.0), np.full(len(heavy_ball), 20.0)]
    )

    shade = bake_ambient_occlusion(
        positions,
        mass=mass,
        radius=0.3,
        extinction=1.5,
        grid_cells=64,
        n_directions=6,
    )
    assert shade[len(light_ball) :].mean() < shade[: len(light_ball)].mean() - 0.05


# ---------------------------------------------------------------------------
# Direction set
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("requested", [1, 2, 3, 6, 12, 13, 64])
def test_sphere_directions_are_paired_unit_vectors(requested):
    directions = sphere_directions(requested)

    assert len(directions) % 2 == 0
    assert len(directions) >= requested
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)

    half = len(directions) // 2
    np.testing.assert_allclose(directions[:half], -directions[half:], atol=1e-12)


def test_sphere_directions_are_roughly_isotropic():
    directions = sphere_directions(64)
    # Exactly zero by construction from the pairing; the real check is that no
    # single axis is over-represented in |component|.
    np.testing.assert_allclose(directions.sum(axis=0), 0.0, atol=1e-9)
    magnitudes = np.abs(directions).mean(axis=0)
    assert magnitudes.max() / magnitudes.min() < 1.5


def test_sphere_directions_rejects_zero():
    with pytest.raises(ValueError, match="n_directions"):
        sphere_directions(0)


# ---------------------------------------------------------------------------
# The directional primitive
# ---------------------------------------------------------------------------


def test_directional_optical_depth_is_larger_behind_a_slab():
    """Shadowed side vs lit side of a slab, along the sampled direction."""
    slab = _grid_plane((-1.0, 1.0), (-1.0, 1.0), 0.04, z=0.0)
    probes = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    positions = np.vstack([slab, probes])
    mass = np.ones(len(positions))

    tau = directional_optical_depth(
        positions, mass, (0.0, 0.0, 1.0), grid_cells=48, radius=None
    )

    lit, shadowed = tau[-2], tau[-1]
    assert shadowed > lit
    assert lit == pytest.approx(0.0, abs=1e-5)


def test_directional_optical_depth_window_truncates():
    slab = _grid_plane((-1.0, 1.0), (-1.0, 1.0), 0.04, z=0.0)
    probe = np.array([[0.0, 0.0, -0.5]])
    positions = np.vstack([slab, probe])
    mass = np.ones(len(positions))

    unbounded = directional_optical_depth(
        positions, mass, (0.0, 0.0, 1.0), grid_cells=48, radius=None
    )
    windowed = directional_optical_depth(
        positions, mass, (0.0, 0.0, 1.0), grid_cells=48, radius=0.1
    )

    assert unbounded[-1] > 0.0
    assert windowed[-1] == pytest.approx(0.0, abs=1e-5)


def test_directional_optical_depth_rejects_a_zero_direction():
    positions = _ball(100, seed=17)
    with pytest.raises(ValueError, match="non-zero"):
        directional_optical_depth(positions, np.ones(len(positions)), (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Degenerate input and validation
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty():
    shade = bake_ambient_occlusion(np.empty((0, 3), dtype=np.float32))
    assert shade.shape == (0,)
    assert shade.dtype == np.float32


def test_single_element_is_unoccluded():
    shade = bake_ambient_occlusion(np.array([[1.0, 2.0, 3.0]]))
    assert shade == pytest.approx(np.ones(1), abs=1e-5)


def test_all_elements_coincident_does_not_divide_by_zero():
    positions = np.zeros((500, 3))
    shade = bake_ambient_occlusion(positions, grid_cells=32, n_directions=6)
    assert np.all(np.isfinite(shade))


def test_result_dtype_and_range():
    positions = _ball(5_000, seed=18)
    shade = bake_ambient_occlusion(positions, grid_cells=32, n_directions=6)
    assert shade.dtype == np.float32
    assert shade.min() >= 0.0
    assert shade.max() <= 1.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spatial_dims": (0, 1)}, "exactly 3 axes"),
        ({"spatial_dims": (0, 1, 9)}, "out of range"),
        ({"spatial_dims": (0, 1, 1)}, "distinct"),
        ({"floor": 1.5}, "floor"),
        ({"strength": -0.1}, "strength"),
        ({"grid_cells": 2}, "grid_cells"),
        ({"radius": 0.0}, "radius"),
        ({"extinction": "sometimes"}, "extinction"),
        ({"extinction": -1.0}, "extinction"),
    ],
)
def test_rejects_invalid_arguments(kwargs, message):
    positions = _ball(200, seed=19)
    with pytest.raises(ValueError, match=message):
        bake_ambient_occlusion(positions, **kwargs)


def test_rejects_non_2d_positions():
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        bake_ambient_occlusion(np.zeros(10))


def test_rejects_mismatched_mass():
    positions = _ball(100, seed=20)
    with pytest.raises(ValueError, match="mass must have shape"):
        bake_ambient_occlusion(positions, mass=np.ones(99))


def test_rejects_negative_mass():
    positions = _ball(100, seed=21)
    mass = np.ones(len(positions))
    mass[0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        bake_ambient_occlusion(positions, mass=mass)


def test_rejects_mismatched_group_by():
    positions = _ball(100, seed=22)
    with pytest.raises(ValueError, match="group_by must have shape"):
        bake_ambient_occlusion(positions, group_by=np.zeros(50, dtype=int))
