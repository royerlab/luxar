"""Tests for :mod:`luxar.shading.occlusion`.

The comparative tests pass an explicit ``extinction`` rather than leaning on
``"auto"``. Auto-calibration pins the *median of the population it was given*, so
two separate calls land on two different scales and a cross-call comparison would
be measuring the calibration instead of the geometry. Where the calibration
itself is under test it is named as such.
"""

import tracemalloc

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


def _golden_inputs():
    rng = np.random.default_rng(2189)
    positions = rng.normal(size=(16, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= rng.uniform(0.15, 1.0, size=(16, 1))
    mass = rng.uniform(0.25, 1.75, size=16)
    normals = rng.normal(size=(16, 3))
    return positions, mass, normals


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


@pytest.mark.parametrize(
    "occluder, use_normals, expected",
    [
        (
            "density",
            False,
            [
                0.4772594,
                0.8658054,
                0.52359486,
                0.8584347,
                0.7205625,
                0.78561676,
                0.3946743,
                1.0,
                0.6091558,
                0.65593886,
                0.8584526,
                0.59058166,
                0.735902,
                0.59123194,
                0.74920523,
                0.8712187,
            ],
        ),
        (
            "density",
            True,
            [
                0.26036423,
                0.7944937,
                0.52302706,
                1.0,
                0.3638128,
                1.0,
                0.7787249,
                1.0,
                0.6810605,
                0.4389521,
                1.0,
                0.3851334,
                0.86871606,
                0.20319045,
                0.31647933,
                0.68358916,
            ],
        ),
        (
            "opaque",
            False,
            [
                0.4709193,
                0.85833335,
                0.528831,
                0.85833335,
                0.7166667,
                0.7831265,
                0.41334534,
                1.0,
                0.5815767,
                0.6709464,
                0.85833335,
                0.60642946,
                0.7166667,
                0.575,
                0.7455794,
                0.85833335,
            ],
        ),
        (
            "opaque",
            True,
            [
                0.2575248,
                0.7855057,
                0.5391116,
                1.0,
                0.35610747,
                1.0,
                0.7800412,
                1.0,
                0.66421187,
                0.4349221,
                1.0,
                0.3874846,
                0.86749125,
                0.168073,
                0.3089602,
                0.6577838,
            ],
        ),
    ],
)
def test_bake_matches_fixed_seed_golden_values(occluder, use_normals, expected):
    positions, mass, normals = _golden_inputs()
    got = bake_ambient_occlusion(
        positions,
        mass=mass,
        normals=normals if use_normals else None,
        occluder=occluder,
        radius=0.9,
        n_directions=6,
        grid_cells=8,
        strength=0.85,
        floor=0.1,
    )
    np.testing.assert_array_equal(got, np.asarray(expected, dtype=np.float32))


def test_shading_processes_columns_in_bounded_row_slabs(monkeypatch):
    monkeypatch.setattr(occlusion_module, "_ROW_SLAB_SIZE", 4, raising=False)
    rows = 5
    columns = np.full((rows, 2), 0.5, dtype=np.float32)
    weights = np.full((rows, 2), 0.75, dtype=np.float32)
    seen_rows = []
    real_transmittance = occlusion_module._transmittance

    def recording_transmittance(depth, occluder):
        seen_rows.append(len(depth))
        return real_transmittance(depth, occluder)

    monkeypatch.setattr(occlusion_module, "_transmittance", recording_transmittance)
    occlusion_module._shade_columns(
        columns,
        weights,
        occluder="density",
        extinction=0.8,
        strength=0.7,
        floor=0.0,
    )

    assert seen_rows == [4, 1]
    np.testing.assert_array_equal(columns, np.full_like(columns, 0.5))


def test_slabbed_helpers_are_bit_identical_across_boundaries(monkeypatch):
    monkeypatch.setattr(occlusion_module, "_ROW_SLAB_SIZE", 4)
    rng = np.random.default_rng(36)
    columns = rng.uniform(0.0, 3.0, size=(9, 6)).astype(np.float32)
    weights = rng.uniform(0.01, 1.0, size=(9, 6)).astype(np.float32)

    for occluder in ("density", "opaque"):
        for active_weights in (None, weights):
            mapped = columns.copy()
            mapped *= 0.73
            occlusion_module._transmittance(mapped, occluder)
            if active_weights is None:
                transmittance = mapped.mean(axis=1)
            else:
                transmittance = (mapped * active_weights).sum(
                    axis=1
                ) / active_weights.sum(axis=1)
            expected = np.clip(1.0 - 0.81 * (1.0 - transmittance), 0.13, 1.0).astype(
                np.float32
            )

            got = occlusion_module._shade_columns(
                columns,
                active_weights,
                occluder=occluder,
                extinction=0.73,
                strength=0.81,
                floor=0.13,
            )
            np.testing.assert_array_equal(got, expected)

    expected = (columns * weights).sum(axis=1) / weights.sum(axis=1)
    np.testing.assert_array_equal(occlusion_module._combine(columns, weights), expected)


def test_weighted_combine_does_not_materialize_a_full_product(monkeypatch):
    monkeypatch.setattr(occlusion_module, "_ROW_SLAB_SIZE", 1_024, raising=False)
    per_direction = np.full((100_000, 8), 0.5, dtype=np.float32)
    weights = np.full_like(per_direction, 0.75)

    tracemalloc.start()
    try:
        got = occlusion_module._combine(per_direction, weights)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    np.testing.assert_array_equal(got, np.full(len(per_direction), 0.5, np.float32))
    assert peak < per_direction.nbytes // 2


def test_ungrouped_bake_reuses_group_columns_result(monkeypatch):
    positions = _ball(8, seed=35)
    sentinel = np.zeros((len(positions), 2), dtype=np.float32)
    seen = []

    monkeypatch.setattr(occlusion_module, "sphere_directions", lambda _: np.eye(2, 3))
    monkeypatch.setattr(occlusion_module, "_group_columns", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        occlusion_module,
        "_resolve_extinction",
        lambda extinction, columns, weights, occluder: seen.append(columns) or 0.0,
    )

    bake_ambient_occlusion(positions, n_directions=2, grid_cells=4)

    assert len(seen) == 1
    assert seen[0] is sentinel


def test_direction_weight_clipping_reuses_matmul_result(monkeypatch):
    real_clip = np.clip
    clip_calls = []

    def recording_clip(values, lower, upper, **kwargs):
        clip_calls.append((values, kwargs.get("out")))
        return real_clip(values, lower, upper, **kwargs)

    monkeypatch.setattr(occlusion_module.np, "clip", recording_clip)
    occlusion_module._direction_weights(np.eye(3), np.eye(3))

    assert len(clip_calls) == 1
    assert clip_calls[0][1] is clip_calls[0][0]


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
    grouped = bake_ambient_occlusion(
        positions, normals=normals, group_by=groups, **kwargs
    )
    full_sphere = bake_ambient_occlusion(positions, group_by=groups, **kwargs)

    np.testing.assert_allclose(grouped[: len(ball)], grouped[len(ball) :], atol=1e-6)
    assert np.max(np.abs(grouped - full_sphere)) > 0.05


# ---------------------------------------------------------------------------
# Occluder model: medium vs surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "occluder, expected",
    [
        # Beer-Lambert: never reaches zero, always keeps falling.
        ("density", [1.0, np.exp(-1.0), np.exp(-1.5), np.exp(-15.0)]),
        # Saturating: hits zero at one wall's worth and stays there.
        ("opaque", [1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_transmittance_mapping_is_the_defining_difference(occluder, expected):
    """The invariant the mode exists for, tested where it actually lives.

    Asserted on the mapping rather than through a scene, because geometry cannot
    isolate it: in any real arrangement some grazing directions reach the far
    material WITHOUT passing the near material, so even the saturating mode keeps
    a little sensitivity. Here the claim is exact.
    """
    got = occlusion_module._transmittance(np.array([0.0, 1.0, 1.5, 15.0]), occluder)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_opaque_is_far_less_sensitive_to_added_depth_than_density():
    """The same difference as it survives into a real bake.

    One thin sheet against a stack of ten identical ones, probed from just below.
    Opaque does not go fully insensitive — grazing directions enter the stack from
    the side without crossing the first sheet, so they are not saturated by it —
    but it must move markedly less. Measured: density 0.188, opaque 0.109.
    """
    step = 0.02
    # Directly beneath the stack, so roughly a whole hemisphere is blocked and
    # the two mappings are compared where they actually differ. Held far enough
    # below the first sheet to clear the own-slice exclusion.
    probe = np.array([[0.0, 0.0, -0.06]])

    def sheets(count):
        stack = [
            _grid_plane((-0.6, 0.6), (-0.6, 0.6), step, z=0.02 * i)
            for i in range(count)
        ]
        return np.vstack([*stack, probe])

    thin, thick = sheets(1), sheets(10)
    kwargs = dict(radius=0.4, grid_cells=96, n_directions=12, strength=1.0)

    d_thin = bake_ambient_occlusion(thin, extinction=1.0, **kwargs)[-1]
    d_thick = bake_ambient_occlusion(thick, extinction=1.0, **kwargs)[-1]
    o_thin = bake_ambient_occlusion(thin, occluder="opaque", extinction=1.0, **kwargs)[
        -1
    ]
    o_thick = bake_ambient_occlusion(
        thick, occluder="opaque", extinction=1.0, **kwargs
    )[-1]

    # Density: ten walls are markedly darker than one.
    assert d_thin - d_thick > 0.15
    # Opaque: substantially less sensitive to the nine it cannot see past.
    assert abs(o_thin - o_thick) < 0.75 * (d_thin - d_thick)


def test_opaque_transmittance_never_goes_negative():
    """`max(0, 1 - depth)` must clamp, not wrap into negative light."""
    positions = _ball(15_000, seed=28)
    shade = bake_ambient_occlusion(
        positions,
        occluder="opaque",
        extinction=50.0,
        radius=0.4,
        grid_cells=48,
        n_directions=6,
        strength=1.0,
    )
    assert shade.min() >= 0.0
    assert np.all(np.isfinite(shade))


def test_both_occluders_hit_the_same_auto_target():
    """Auto calibration is inverted per mode, so neither aims somewhere else."""
    positions = _truncated_ball(20_000, seed=29)
    kwargs = dict(radius=0.3, grid_cells=48, n_directions=12, strength=1.0)

    for occluder in ("density", "opaque"):
        shade = bake_ambient_occlusion(positions, occluder=occluder, **kwargs)
        median = float(np.median(shade))
        assert (
            AUTO_TARGET_TRANSMITTANCE - 0.05
            < median
            < (AUTO_TARGET_TRANSMITTANCE + 0.25)
        ), f"{occluder} median {median:.3f}"


def test_opaque_carries_more_contrast_on_a_thin_shell():
    """Why a surface should prefer it: matched median, more spread.

    Both modes are auto-calibrated to the same median here, so this compares the
    SHAPE of the two mappings rather than their exposure.
    """
    positions, sheet_index, trench_index = _sheet_and_trench()
    kwargs = dict(radius=0.4, grid_cells=96, n_directions=12, strength=1.0)

    density = bake_ambient_occlusion(positions, **kwargs)
    opaque = bake_ambient_occlusion(positions, occluder="opaque", **kwargs)

    assert opaque.std() > density.std()
    # And the trench still reads as more enclosed than the open sheet.
    assert opaque[trench_index] < opaque[sheet_index]


def test_rejects_an_unknown_occluder():
    positions = _ball(200, seed=30)
    with pytest.raises(ValueError, match="occluder must be"):
        bake_ambient_occlusion(positions, occluder="opake")


def test_occluder_is_validated_before_the_grid_passes(monkeypatch):
    """A typo must fail fast, not after paying for every direction."""
    called = []
    real = occlusion_module._mass_grid
    monkeypatch.setattr(
        occlusion_module,
        "_mass_grid",
        lambda *a, **k: (called.append(1), real(*a, **k))[1],
    )
    with pytest.raises(ValueError, match="occluder must be"):
        bake_ambient_occlusion(_ball(500, seed=31), occluder="nope")
    assert not called, "grid passes ran before the argument was rejected"


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
    grouped = bake_ambient_occlusion(positions, group_by=groups, **kwargs)
    merged = bake_ambient_occlusion(positions, **kwargs)

    # The translated copies share calibration but remain geometrically isolated.
    np.testing.assert_allclose(grouped[: len(ball)], grouped[len(ball) :], atol=1e-6)

    # And the guard is load-bearing: without it the overlap region cross-occludes,
    # so the merged bake is measurably darker than the isolated one.
    overlap = ball[:, 0] > 0.35
    assert (
        merged[: len(ball)][overlap].mean()
        < grouped[: len(ball)][overlap].mean() - 0.02
    )


def test_group_by_shares_calibration_across_timepoints():
    """Grouping isolates geometry without flattening real density changes."""
    diffuse = _ball(12_000, seed=29)
    compact = diffuse * 0.25
    positions = np.vstack([diffuse, compact])
    groups = np.repeat([0, 1], len(diffuse))

    shade = bake_ambient_occlusion(
        positions,
        group_by=groups,
        grid_cells=48,
        n_directions=6,
    )

    diffuse_mean = float(shade[: len(diffuse)].mean())
    compact_mean = float(shade[len(diffuse) :].mean())
    assert compact_mean < 0.9 * diffuse_mean


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


@pytest.mark.parametrize(
    "positions, kwargs, message",
    [
        (np.zeros(10), {}, r"\(N, D\)"),
        (np.zeros((10, 3)), {"grid_cells": 0}, "grid_cells"),
        (np.zeros((10, 3)), {"radius": -1.0}, "radius"),
    ],
)
def test_directional_optical_depth_rejects_invalid_arguments(
    positions, kwargs, message
):
    with pytest.raises(ValueError, match=message):
        directional_optical_depth(
            positions,
            np.ones(len(positions)),
            (0.0, 0.0, 1.0),
            **kwargs,
        )


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


def test_default_direction_counts_match_the_converged_floors():
    positions = _ball(1_000, seed=32)
    normals = positions / np.linalg.norm(positions, axis=1, keepdims=True)
    kwargs = dict(radius=0.3, grid_cells=24)

    np.testing.assert_allclose(
        bake_ambient_occlusion(positions, **kwargs),
        bake_ambient_occlusion(positions, n_directions=24, **kwargs),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        bake_ambient_occlusion(positions, normals=normals, **kwargs),
        bake_ambient_occlusion(positions, normals=normals, n_directions=48, **kwargs),
        atol=1e-7,
    )


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


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_mass(bad_value):
    positions = _ball(100, seed=30)
    mass = np.ones(len(positions))
    mass[0] = bad_value
    with pytest.raises(ValueError, match="finite"):
        bake_ambient_occlusion(positions, mass=mass)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_spatial_positions(bad_value):
    positions = _ball(100, seed=31)
    positions[0, 1] = bad_value
    with pytest.raises(ValueError, match="finite"):
        bake_ambient_occlusion(positions)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_normals(bad_value):
    positions = _ball(100, seed=33)
    normals = positions.copy()
    normals[0, 1] = bad_value
    with pytest.raises(ValueError, match="normals must be finite"):
        bake_ambient_occlusion(positions, normals=normals)


def test_rejects_mismatched_group_by():
    positions = _ball(100, seed=22)
    with pytest.raises(ValueError, match="group_by must have shape"):
        bake_ambient_occlusion(positions, group_by=np.zeros(50, dtype=int))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_group_by(bad_value):
    positions = _ball(100, seed=34)
    groups = np.zeros(len(positions))
    groups[0] = bad_value
    with pytest.raises(ValueError, match="group_by must be finite"):
        bake_ambient_occlusion(positions, group_by=groups)
