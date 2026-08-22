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


@pytest.fixture(scope="module")
def bubbles() -> cloud.Bubbles:
    return cloud.build_bubbles(np.random.default_rng(42))


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

    def test_the_flow_is_frame_rate_independent(self, parcels: np.ndarray) -> None:
        """`--frames` must be a resolution knob, not a physics knob.

        Speeds are per unit PHASE and each frame advances by
        ``dt = 1/(n_frames-1)``. Integrating a fixed displacement once per
        frame instead would make total distance travelled proportional to the
        frame count, so doubling the temporal resolution would silently double
        how far the cloud drifts — the same sequence at a different sampling
        rate would be a different cloud.

        Midpoint integration is second order, so the two trajectories should
        agree far more closely than either agrees with the exact flow.
        """

        def run(n_frames: int) -> np.ndarray:
            positions = parcels.copy()
            dt = 1.0 / (n_frames - 1)
            for frame in range(n_frames - 1):
                phase = frame / (n_frames - 1)
                updraft = cloud.UPDRAFT * (
                    0.35 + 0.65 * float(np.exp(-(((phase - 0.30) / 0.30) ** 2)))
                )
                cloud.advect(positions, updraft, dt)
            return positions

        coarse = run(60)
        fine = run(120)

        drift = np.linalg.norm(fine - parcels, axis=1)
        disagreement = np.linalg.norm(fine - coarse, axis=1)
        assert float(drift.mean()) > 1.0, "the parcels barely moved; test is vacuous"
        assert float(disagreement.mean()) < 0.06 * float(drift.mean()), (
            f"60 and 120 frames disagree by {disagreement.mean():.3f} against a "
            f"mean drift of {drift.mean():.3f} — the frame count is changing "
            f"the physics, not just how finely it is sampled"
        )

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
            cloud.advect(positions, updraft, 1.0 / 59)

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

    @staticmethod
    def _env(positions: np.ndarray, phase: float, bubbles: cloud.Bubbles) -> np.ndarray:
        billow = np.zeros(len(positions), dtype=np.float32)
        return cloud.envelope(positions, phase, billow, bubbles)

    def test_nothing_condenses_above_the_crown(self, bubbles: cloud.Bubbles) -> None:
        state = cloud.life_cycle(0.5)
        above = state.top + 6.0
        positions = np.array(
            [[0.0, above, 0.0], [0.05, above, 0.0], [0.0, above + 8.0, 0.0]],
            dtype=np.float32,
        )
        assert float(self._env(positions, 0.5, bubbles).max()) < 1e-2

    def test_nothing_condenses_below_the_base(self, bubbles: cloud.Bubbles) -> None:
        positions = np.array(
            [[0.0, cloud.BASE_Y - 2.0, 0.0], [1.0, cloud.BASE_Y - 5.0, 1.0]],
            dtype=np.float32,
        )
        assert float(self._env(positions, 0.5, bubbles).max()) < 1e-3

    def test_the_body_is_one_connected_column(self, bubbles: cloud.Bubbles) -> None:
        """No horizontal gap between the base slab and the thermals above it.

        The root slab exists because thermals alone leave the bottom ragged —
        each is a sphere that has already left the base by the time it is big.
        But a slab that does not reach far enough up separates from the tower
        and renders as a second, unrelated cloud sitting underneath.
        """
        state = cloud.life_cycle(cloud.OPENING_PHASE)
        heights = np.linspace(0.3, state.top, 120)
        axis = np.column_stack(
            [
                cloud.TILT * (heights - cloud.BASE_Y),
                heights,
                np.zeros_like(heights),
            ]
        ).astype(np.float32)
        along = self._env(axis, cloud.OPENING_PHASE, bubbles)

        # `state.top` is the ceiling the thermals climb TOWARD, and a thermal
        # arriving there has already faded to nothing — so the realized crown
        # sits below it. Connectivity is only meaningful up to the real top.
        solid = np.flatnonzero(along > 0.15)
        assert len(solid) > 10, "no column found at all"
        crown = solid[-1]

        interior = along[: crown + 1]
        assert float(interior.min()) > 0.06, (
            f"envelope along the cloud's own axis dips to {interior.min():.3f} "
            f"below the crown at y={heights[crown]:.1f}; the tower has "
            f"separated from the slab at its base"
        )

    def test_the_column_is_taller_than_it_is_wide(self, bubbles: cloud.Bubbles) -> None:
        """A cumulus congestus stands up. A squat one reads as a cotton ball."""
        state = cloud.life_cycle(cloud.OPENING_PHASE)
        rng = np.random.default_rng(9)
        probe = np.column_stack(
            [
                rng.uniform(-14, 14, 60_000),
                rng.uniform(-1, state.top + 4, 60_000),
                rng.uniform(-14, 14, 60_000),
            ]
        ).astype(np.float32)
        inside = probe[self._env(probe, cloud.OPENING_PHASE, bubbles) > 0.25]
        assert len(inside) > 500

        height = inside[:, 1].max() - inside[:, 1].min()
        width = max(
            inside[:, 0].max() - inside[:, 0].min(),
            inside[:, 2].max() - inside[:, 2].min(),
        )
        assert height > width, f"cloud is {width:.1f} wide by {height:.1f} tall"

    def test_dissipation_shrinks_the_silhouette(self) -> None:
        """Decay has to be visible from OUTSIDE.

        While the cloud rendered additively, dissipation was expressed by
        raising the noise threshold, which erodes the interior. Once the medium
        is opaque the interior cannot be seen at all, so that reads as no
        change whatsoever; the silhouette itself has to come in.
        """
        mature = cloud.life_cycle(0.5)
        late = cloud.life_cycle(1.0)
        assert late.width < 0.9 * mature.width
        assert late.top < 0.98 * mature.top


class TestRenderingRegime:
    """The renderer has to be in the same optical regime as the shading.

    This is the class of bug that produced the worst artefact in the demo's
    history. A cumulus is optically THICK, so you see its surface; ``additive``
    blending models an optically THIN emissive medium and ignores depth
    entirely. Shading the parcels by how much water is above them, and then
    compositing them with a mode that cannot occlude, painted a dark interior
    that was fully visible THROUGH the lit shell — the cloud rendered as a
    glowing archway with a hole in the middle, and from overhead as a ring.
    Measured on that build, the core had the HIGHEST point density (346 vs 51
    per unit area at the rim) and the LOWEST brightness (0.46 vs 0.85).
    """

    def test_the_node_absorbs_what_is_behind_it(self) -> None:
        import ast

        tree = ast.parse(cloud.__file__ and open(cloud.__file__).read())
        modes = [
            kw.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for kw in node.keywords
            if kw.arg == "blending_mode" and isinstance(kw.value, ast.Constant)
        ]
        assert modes, "no blending_mode found"
        assert all(m == "volumetric" for m in modes), (
            f"cloud node uses {modes}; the baked sunlight is computed from an "
            f"occlusion integral and is only coherent in a mode that occludes"
        )

    def test_a_fully_lit_parcel_stays_inside_the_display_range(self) -> None:
        """Exposure guard.

        Under emission-absorption the accumulated radiance of a thick medium
        tends to emission/extinction — that is, to the parcel colour itself. So
        the brightest possible pixel is the brightest parcel colour, and if
        that already exceeds 1 the shading is invisible: every lit face clips
        to flat white. The preset's bloom threshold of 0.01 means the whole
        cloud blooms onto itself as well, so there has to be real headroom.
        """
        brightest = cloud.SUN_COLOR + cloud.SKY_COLOR + cloud.GROUND_COLOR
        assert float(brightest.max()) < 0.62, (
            f"a fully lit parcel emits {brightest}; ACES plus the preset's "
            f"all-over bloom will clip that to white"
        )
        # ...and the shadow side must not be black, or the cloud reads as a
        # cut-out. Skylight alone has to carry it.
        shadow = cloud.SKY_COLOR + cloud.GROUND_COLOR
        assert float(shadow.max()) > 0.04
        # Skylight is blue: that is why a real cumulus underside is cool grey.
        assert cloud.SKY_COLOR[2] > cloud.SKY_COLOR[0] * 1.4


class TestCondensate:
    """The life cycle should be the only thing that moves the cloud."""

    def test_every_frame_carries_condensate(
        self, parcels: np.ndarray, field: cloud.NoiseField, bubbles: cloud.Bubbles
    ) -> None:
        # An empty frame is a hole in the time axis, and Home/End land on
        # exactly the two frames most at risk of being one.
        for phase in np.linspace(0.0, 1.0, 21):
            water = cloud.condensate(parcels, field, float(phase), bubbles)
            assert float(water.max()) > 0.05, f"phase {phase:.2f} has no water"

    def test_the_point_count_does_not_collapse_at_either_end(
        self, parcels: np.ndarray, field: cloud.NoiseField, bubbles: cloud.Bubbles
    ) -> None:
        rng = np.random.default_rng(11)
        gate_u = rng.random(N_PARCELS).astype(np.float32)
        probes = [
            cloud.condensate(parcels, field, p, bubbles)
            for p in cloud.CALIBRATION_PHASES
        ]
        gate = cloud.calibrate_gate(probes, N_PARCELS // 20)

        counts = [
            int(
                (
                    cloud.emission_odds(
                        cloud.condensate(parcels, field, p, bubbles), gate
                    )
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
        self, parcels: np.ndarray, field: cloud.NoiseField, bubbles: cloud.Bubbles
    ) -> None:
        probes = [
            cloud.condensate(parcels, field, p, bubbles)
            for p in cloud.CALIBRATION_PHASES
        ]
        target = N_PARCELS // 10
        gate = cloud.calibrate_gate(probes, target)
        busiest = max(float(cloud.emission_odds(w, gate).sum()) for w in probes)
        assert abs(busiest - target) < 0.02 * target


class TestShading:
    """The baked light has to have a range, and not depend on the sampling."""

    def test_optical_depth_spans_a_visible_range(
        self, parcels: np.ndarray, field: cloud.NoiseField, bubbles: cloud.Bubbles
    ) -> None:
        density = N_PARCELS / (
            np.pi * cloud.SEED_RADIUS**2 * (cloud.SEED_Y[1] - cloud.SEED_Y[0])
        )
        water = cloud.condensate(parcels, field, cloud.OPENING_PHASE, bubbles)
        keep = water > 0.4 * water.max()
        rgb = cloud.shade(parcels[keep], water[keep], float(density))

        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        contrast = float(np.quantile(luma, 0.9)) / float(np.quantile(luma, 0.1))
        assert contrast > 1.8, (
            f"lit-to-shadow contrast is only {contrast:.2f}x — the shading is "
            f"there but invisible, which is how EXTINCTION was mis-scaled"
        )

    def test_the_sun_is_not_straight_overhead(self) -> None:
        """A vertical sun lights every lobe by depth alone.

        Two lobes at the same altitude then receive identical light however
        they face, so the relief that makes a cumulus legible disappears and
        the cloud renders as a flat cut-out. The whole reason
        :func:`optical_depth` takes a direction is to avoid that.
        """
        d = np.array(cloud.SUN_DIRECTION, dtype=np.float64)
        d /= np.linalg.norm(d)
        elevation = np.degrees(np.arcsin(d[1]))
        assert 20.0 < elevation < 70.0, (
            f"sun elevation {elevation:.0f} degrees is too close to vertical "
            f"(or below the horizon) to model the cloud"
        )

    def test_optical_depth_follows_its_direction(self) -> None:
        """A slab lit from the side is not lit like a slab lit from above."""
        rng = np.random.default_rng(4)
        slab = np.column_stack(
            [
                rng.uniform(-6, 6, 30_000),
                rng.uniform(0, 2, 30_000),
                rng.uniform(-6, 6, 30_000),
            ]
        ).astype(np.float32)
        water = np.full(len(slab), 0.5, dtype=np.float32)

        # A wide flat slab is deep along x and shallow along y, so a horizontal
        # ray through it accumulates far more than a vertical one.
        from_above = cloud.optical_depth(slab, water, 400.0, (0.0, 1.0, 0.0))
        from_side = cloud.optical_depth(slab, water, 400.0, (1.0, 0.0, 0.0))
        assert from_side.mean() > 2.0 * from_above.mean()

    def test_optical_depth_is_independent_of_parcel_count(self) -> None:
        """Folding the parcel count into the constant means --parcels relights it.

        Sampled on a dense synthetic ball rather than on the demo's own
        parcels: the estimator is a histogram, so at a few thousand points over
        a 44-cubed grid most cells hold nothing and the comparison measures
        shot noise instead of the scaling being guarded.
        """
        rng = np.random.default_rng(17)
        n = 240_000
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        ball = (d * (6.0 * rng.random((n, 1)) ** (1 / 3))).astype(np.float32)
        water = np.full(n, 0.5, dtype=np.float32)
        volume = 4.0 / 3.0 * np.pi * 6.0**3

        full = cloud.optical_depth(ball, water, float(n / volume), cloud.SUN_DIRECTION)
        half = slice(None, None, 2)
        halved = cloud.optical_depth(
            ball[half], water[half], float(n / 2 / volume), cloud.SUN_DIRECTION
        )
        assert abs(float(halved.mean()) - float(full.mean())) < 0.12 * float(
            full.mean()
        ), (
            f"halving the parcels moved mean tau {full.mean():.3f} -> "
            f"{halved.mean():.3f}; the sampling is leaking into the lighting"
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

    def test_the_points_carry_an_alpha_channel(self, tmp_path) -> None:
        """Volumetric blending reads optical depth off the alpha column.

        Written as RGB, every parcel would absorb identically regardless of how
        much water it stands for, and the cloud would lose the density gradient
        that makes its edge soft and its core solid.
        """
        path = tmp_path / "cloud.luxar.zarr"
        cloud.generate_evolving_cloud(
            path, n_parcels=20_000, n_frames=4, target_points_per_frame=2_000
        )
        colors = np.asarray(LuxarScene.load(path).get_points("EvolvingCloud").colors)
        assert colors.shape[1] == 4, f"colors are {colors.shape[1]}-channel, not RGBA"
        alpha = colors[:, 3]
        assert 0.0 <= alpha.min() and alpha.max() <= 1.0
        assert alpha.max() - alpha.min() > 0.2, "alpha carries no density gradient"

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
