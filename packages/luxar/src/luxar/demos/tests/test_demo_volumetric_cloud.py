"""Guards for the evolving-cumulus demo.

Every defect this file pins was found by MEASURING the demo, never by reading
it, and not one of them raised: each produced a plausible-looking cloud. The
original demo shipped for years emitting 790 points out of 800,000 candidates
because its noise hash correlated with coordinate magnitude and it sampled less
than one lattice cell — the scene rendered, the log said "7-octave fractal
noise with turbulence", and the result was a smudge. That is the failure mode
this module exists to catch, so the assertions here are on measured statistics
rather than on the shape of the code.

The thresholds are deliberately loose. They are tripwires for a defect
returning, not a pin on the current tuning, which should stay free to change.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar import LuxarScene

from .. import demo_volumetric_cloud as cloud

#: Enough parcels for the statistics to mean something, few enough to stay fast.
N_PARCELS = 40_000


@pytest.fixture(scope="module")
def parcels() -> np.ndarray:
    """Seeded parcels, laid out exactly as the demo lays them out."""
    rng = np.random.default_rng(42)
    radius = cloud.SEED_RADIUS * np.sqrt(rng.random(N_PARCELS))
    theta = rng.uniform(0.0, 2.0 * np.pi, N_PARCELS)
    positions = np.empty((N_PARCELS, 3), dtype=np.float32)
    positions[:, 0] = radius * np.cos(theta)
    positions[:, 1] = rng.uniform(cloud.SEED_Y[0], cloud.SEED_Y[1], N_PARCELS)
    positions[:, 2] = radius * np.sin(theta)
    return positions


@pytest.fixture(scope="module")
def field(parcels: np.ndarray) -> cloud.NoiseField:
    return cloud.build_noise_field((parcels / cloud.CLOUD_SIZE).astype(np.float32))


class TestNoiseHash:
    """The hash must actually mix. This is the 790-point bug."""

    def test_value_is_uncorrelated_with_coordinate_magnitude(self) -> None:
        # The old hash was `(xi*C1 + yi*C2 + zi*C3 + seed) % 1000000 / 500000 - 1`,
        # whose low bits barely move, so the value tracked |coordinate| and every
        # corner near the lattice origin came back close to -1. Sampled over a
        # few cells around the origin — which is exactly where the demo samples —
        # that shows up as a smooth radial ramp masquerading as noise.
        rng = np.random.default_rng(0)
        p = rng.uniform(-2.0, 2.0, (60_000, 3))
        value = cloud.simple_noise_3d(p[:, 0], p[:, 1], p[:, 2], seed=7)
        distance = np.linalg.norm(p, axis=1)

        correlation = float(np.corrcoef(value, distance)[0, 1])
        assert abs(correlation) < 0.05, (
            f"noise value correlates with distance from the lattice origin "
            f"(r={correlation:+.3f}); the hash has stopped mixing, and the "
            f"visible symptom is a cloud with a systematically empty core"
        )

    def test_distribution_covers_its_stated_range(self) -> None:
        rng = np.random.default_rng(1)
        p = rng.uniform(-8.0, 8.0, (60_000, 3))
        value = cloud.simple_noise_3d(p[:, 0], p[:, 1], p[:, 2], seed=3)
        assert -1.0 <= value.min() and value.max() <= 1.0
        assert value.max() - value.min() > 1.5, "noise barely varies"
        assert abs(float(value.mean())) < 0.05, "noise is biased away from zero"

    def test_seeds_give_independent_fields(self) -> None:
        rng = np.random.default_rng(2)
        p = rng.uniform(-4.0, 4.0, (40_000, 3))
        a = cloud.simple_noise_3d(p[:, 0], p[:, 1], p[:, 2], seed=0)
        b = cloud.simple_noise_3d(p[:, 0], p[:, 1], p[:, 2], seed=cloud.BILLOW_SEED)
        assert abs(float(np.corrcoef(a, b)[0, 1])) < 0.05


class TestNoiseFrequency:
    """Several lattice cells must span the cloud, or the octaves do nothing."""

    def test_the_base_frequency_resolves_more_than_one_cell(self) -> None:
        # The demo samples material coordinates of roughly +/- 0.5, so the
        # octave-0 lattice count across the domain IS the base frequency. Below
        # about two, every octave lives inside one cell and the "fractal" field
        # is a trilinear ramp.
        assert cloud.NOISE_BASE_FREQ >= 2.0

    def test_the_finest_octave_stays_above_the_point_size(self) -> None:
        # Detail finer than a point cannot be seen, only aliased into speckle.
        finest = cloud.NOISE_BASE_FREQ * 2 ** (cloud.NOISE_OCTAVES - 1)
        feature_size = cloud.CLOUD_SIZE / finest
        assert feature_size > 0.3 * cloud.MIN_RADIUS


class TestTimeInterpolation:
    """The time axis must be stationary — every parcel shares one tau."""

    def test_variance_does_not_breathe_between_keyframes(
        self, field: cloud.NoiseField
    ) -> None:
        # Lerping two independent keyframes with weights summing to one gives a
        # variance of (1-u)^2 + u^2, which halves halfway between. Spatially
        # that averages out; along time it makes the whole cloud pulse.
        octave = 2
        spreads = [
            float(cloud.sample_octave(field.detail, octave, tau).std())
            for tau in np.linspace(0.0, cloud.NOISE_TIME_SPAN, 25)
        ]
        swing = max(spreads) / min(spreads)
        assert swing < 1.15, (
            f"octave {octave} spread swings {swing:.2f}x over the sequence; the "
            f"variance-preserving normalization in sample_octave has been lost "
            f"and the cloud will breathe in and out of focus"
        )

    def test_the_field_stays_smooth_across_a_keyframe(
        self, field: cloud.NoiseField
    ) -> None:
        taus = np.linspace(0.0, cloud.NOISE_TIME_SPAN, 200)
        means = np.array(
            [float(cloud.sample_noise_series(field.detail, t).mean()) for t in taus]
        )
        jumps = np.abs(np.diff(means))
        assert jumps.max() < 8.0 * float(np.median(jumps)) + 1e-3, (
            "the field jumps at a keyframe boundary — the quintic fade is gone"
        )


class TestFlow:
    """Divergence-free is a requirement, not a description."""

    def test_the_velocity_field_is_solenoidal(self) -> None:
        # Central differences on the analytic field. The parcels are a Monte
        # Carlo sample of a uniform density and only a solenoidal field keeps
        # that sample uniform as it deforms; a compressive one piles parcels up
        # and reads as brightness drifting where no water went.
        rng = np.random.default_rng(5)
        p = np.column_stack(
            [
                rng.uniform(-8.0, 8.0, 4000),
                rng.uniform(-1.0, 16.0, 4000),
                rng.uniform(-8.0, 8.0, 4000),
            ]
        ).astype(np.float32)

        eps = 1e-2
        divergence = np.zeros(len(p))
        for axis in range(3):
            step = np.zeros(3, dtype=np.float32)
            step[axis] = eps
            plus = cloud.velocity(p + step, cloud.UPDRAFT)[:, axis]
            minus = cloud.velocity(p - step, cloud.UPDRAFT)[:, axis]
            divergence += (plus - minus) / (2 * eps)

        scale = float(np.abs(cloud.velocity(p, cloud.UPDRAFT)).mean())
        assert float(np.abs(divergence).max()) < 0.05 * scale / eps
        assert float(np.abs(divergence).mean()) < 1e-3 * scale / eps

    def test_advection_preserves_parcel_density_in_the_core(
        self, parcels: np.ndarray
    ) -> None:
        positions = parcels.copy()
        radius = np.hypot(positions[:, 0], positions[:, 2])
        before = int(
            ((radius < 6) & (positions[:, 1] > 0) & (positions[:, 1] < 13)).sum()
        )

        for frame in range(60):
            phase = frame / 59
            updraft = cloud.UPDRAFT * (
                0.35 + 0.65 * float(np.exp(-(((phase - 0.30) / 0.30) ** 2)))
            )
            cloud.advect(positions, updraft)

        radius = np.hypot(positions[:, 0], positions[:, 2])
        after = int(
            ((radius < 6) & (positions[:, 1] > 0) & (positions[:, 1] < 13)).sum()
        )
        assert 0.85 < after / before < 1.18, (
            f"core parcel count moved {before} -> {after} over the run; the flow "
            f"is draining or concentrating the region the cloud lives in"
        )


class TestEnvelope:
    """Shape invariants that were each visible only in a rendered frame."""

    def test_nothing_condenses_above_the_crown(self) -> None:
        # Flooring the turret radius at a small positive width kept the divide
        # safe and left a thin chimney of cloud running up the axis forever.
        state = cloud.life_cycle(0.5)
        above = state.top + 4.0
        positions = np.array(
            [[0.0, above, 0.0], [0.05, above, 0.0], [0.0, above + 6.0, 0.0]],
            dtype=np.float32,
        )
        billow = np.zeros(len(positions), dtype=np.float32)
        assert float(cloud.envelope(positions, 0.5, billow).max()) < 1e-3

    def test_nothing_condenses_below_the_base(self) -> None:
        positions = np.array(
            [[0.0, cloud.BASE_Y - 2.0, 0.0], [1.0, cloud.BASE_Y - 5.0, 1.0]],
            dtype=np.float32,
        )
        billow = np.zeros(len(positions), dtype=np.float32)
        assert float(cloud.envelope(positions, 0.5, billow).max()) < 1e-3

    def test_the_turret_profile_flares_then_domes(self) -> None:
        heights = np.linspace(0.0, 1.0, 200)
        radii = cloud.turret_radius(heights)
        peak = int(np.argmax(radii))
        assert 0.4 < heights[peak] < 0.95, "the widest point is not a shoulder"
        assert radii[-1] < 0.1 * radii[peak], "the crown does not close over"
        assert radii[0] < 0.8 * radii[peak], "the base is not narrower than the body"


class TestCondensate:
    """The life cycle should be the only thing that moves the cloud."""

    def test_every_frame_carries_condensate(
        self, parcels: np.ndarray, field: cloud.NoiseField
    ) -> None:
        # An empty frame is a hole in the time axis, and Home/End land on
        # exactly the two frames most at risk of being one.
        for phase in np.linspace(0.0, 1.0, 21):
            water = cloud.condensate(parcels, field, float(phase))
            assert float(water.max()) > 0.05, f"phase {phase:.2f} has no water"

    def test_the_point_count_does_not_collapse_at_either_end(
        self, parcels: np.ndarray, field: cloud.NoiseField
    ) -> None:
        rng = np.random.default_rng(11)
        gate_u = rng.random(N_PARCELS).astype(np.float32)
        probes = [cloud.condensate(parcels, field, p) for p in cloud.CALIBRATION_PHASES]
        gate = cloud.calibrate_gate(probes, N_PARCELS // 20)

        counts = [
            int(
                (
                    cloud.emission_odds(cloud.condensate(parcels, field, p), gate)
                    > gate_u
                ).sum()
            )
            for p in np.linspace(0.0, 1.0, 13)
        ]
        assert min(counts) > 0.08 * max(counts), (
            f"point count ranges {min(counts)}..{max(counts)} across the life "
            f"cycle; emission goes as water^{cloud.DENSITY_POWER} and envelope "
            f"volume as roughly width^2 x height, so a life cycle that reads "
            f"gently on the page compounds into empty frames"
        )

    def test_the_gate_caps_the_busiest_frame(
        self, parcels: np.ndarray, field: cloud.NoiseField
    ) -> None:
        probes = [cloud.condensate(parcels, field, p) for p in cloud.CALIBRATION_PHASES]
        target = N_PARCELS // 10
        gate = cloud.calibrate_gate(probes, target)
        busiest = max(float(cloud.emission_odds(w, gate).sum()) for w in probes)
        assert abs(busiest - target) < 0.02 * target


class TestShading:
    """The baked light has to have a range, and not depend on the sampling."""

    def test_optical_depth_spans_a_visible_range(
        self, parcels: np.ndarray, field: cloud.NoiseField
    ) -> None:
        density = N_PARCELS / (
            np.pi * cloud.SEED_RADIUS**2 * (cloud.SEED_Y[1] - cloud.SEED_Y[0])
        )
        water = cloud.condensate(parcels, field, cloud.OPENING_PHASE)
        keep = water > 0.4 * water.max()
        tau = cloud.optical_depth(parcels[keep], water[keep], float(density))

        light = cloud.AMBIENT + (1.0 - cloud.AMBIENT) * np.exp(-tau)
        contrast = float(np.quantile(light, 0.9)) / float(np.quantile(light, 0.1))
        assert contrast > 1.5, (
            f"top-to-base light contrast is only {contrast:.2f}x — the shading "
            f"is there but invisible, which is how EXTINCTION was mis-scaled"
        )

    def test_optical_depth_is_independent_of_parcel_count(
        self, parcels: np.ndarray, field: cloud.NoiseField
    ) -> None:
        # Folding the parcel count into the extinction constant means --parcels
        # silently relights the cloud.
        water = cloud.condensate(parcels, field, cloud.OPENING_PHASE)
        keep = water > 0.4 * water.max()
        volume = np.pi * cloud.SEED_RADIUS**2 * (cloud.SEED_Y[1] - cloud.SEED_Y[0])

        full = cloud.optical_depth(
            parcels[keep], water[keep], float(N_PARCELS / volume)
        )
        half = np.flatnonzero(keep)[::2]
        halved = cloud.optical_depth(
            parcels[half], water[half], float(N_PARCELS / 2 / volume)
        )
        assert abs(float(halved.mean()) - float(full.mean())) < 0.25 * float(
            full.mean()
        )


class TestScene:
    """What the written store has to promise the viewer."""

    def test_the_scene_is_4d_with_an_integer_frame_axis(self, tmp_path) -> None:
        path = tmp_path / "cloud.luxar.zarr"
        cloud.generate_evolving_cloud(
            path, n_parcels=20_000, n_frames=6, target_points_per_frame=2_000
        )

        scene = LuxarScene.load(path)
        assert scene.dimensions is not None
        dims = scene.dimensions.dimensions
        assert [d.name for d in dims] == ["x", "y", "z", "time"]

        time_dim = dims[3]
        assert time_dim.display is False
        assert time_dim.discrete is True
        # Integer frames with step 1: the viewer's per-point membership gate is
        # an absolute +/- 0.5, not 0.5 x step, so any other step either makes
        # neighbouring frames visible at once or narrower than a cell.
        assert float(time_dim.step) == 1.0
        assert tuple(time_dim.range) == (0, 5)

    def test_every_timepoint_is_populated(self, tmp_path) -> None:
        path = tmp_path / "cloud.luxar.zarr"
        cloud.generate_evolving_cloud(
            path, n_parcels=20_000, n_frames=6, target_points_per_frame=2_000
        )

        scene = LuxarScene.load(path)
        positions = np.asarray(scene.get_points("EvolvingCloud").positions)
        assert positions.shape[1] == 4

        frames, counts = np.unique(positions[:, 3], return_counts=True)
        assert list(frames) == [0, 1, 2, 3, 4, 5]
        assert counts.min() > 0

    def test_the_opening_camera_is_outside_the_cloud_and_frames_it(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(0.0, 3.0, (5000, 3)).astype(np.float32) + np.array(
            [1.0, 6.0, 0.0], dtype=np.float32
        )

        camera = cloud.compose_opening_camera(points)
        assert camera.position is not None and camera.target is not None

        position = np.array(camera.position)
        target = np.array(camera.target)
        distance = float(np.linalg.norm(position - target))
        radius = float(np.quantile(np.linalg.norm(points - target, axis=1), 0.98))

        assert distance > radius, "the camera opens inside the cloud"
        # Composed for the cinematic lens: a sphere of radius R subtends
        # asin(R / D), so the subject should fill most of the 63 degree frame.
        half_angle = np.degrees(np.arcsin(min(radius / distance, 1.0)))
        assert 15.0 < half_angle < cloud.CINEMATIC_FOV_DEG / 2
