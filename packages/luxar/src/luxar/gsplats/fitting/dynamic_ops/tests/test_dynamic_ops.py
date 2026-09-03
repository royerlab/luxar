"""
Tests for dynamic Gaussian splat operations with fixed-pool relocation.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fit_gsplats import fit_gaussian_splats
from luxar.gsplats.fitting.dynamic_ops import (
    DynamicOpsConfig,
    RecentlyRelocatedTracker,
    _calculate_splat_importance,
    _find_residual_peaks,
    _select_weak_splats,
    apply_dynamic_operations,
)
from luxar.gsplats.fitting.dynamic_ops.operations import _relocate_splats_batch
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.seeds import seed_from_grid


class TestDynamicOpsConfig:
    """Test dynamic operations configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        cfg = DynamicOpsConfig()
        assert cfg.step_every == 50
        assert cfg.k_max_residuals == 40
        assert cfg.nms_radius_vox == 2.0
        assert cfg.enable_tiled_seeding is True
        assert cfg.relocation_percentile == 1.0
        assert cfg.max_relocations_per_step == 64
        assert cfg.init_sigma_vox == 0.5
        assert cfg.min_contribution_threshold == 0.01
        assert cfg.enable_coverage_check is False
        assert cfg.min_splats_to_keep == 10

    def test_config_modification(self) -> None:
        """Test that config values can be modified."""
        cfg = DynamicOpsConfig()
        cfg.step_every = 25
        cfg.k_max_residuals = 5
        cfg.relocation_percentile = 10.0
        cfg.max_relocations_per_step = 5

        assert cfg.step_every == 25
        assert cfg.k_max_residuals == 5
        assert cfg.relocation_percentile == 10.0
        assert cfg.max_relocations_per_step == 5


class TestResidualPeakFinding:
    """Test residual peak finding functionality."""

    def test_find_residual_peaks_2d(self) -> None:
        """Test finding residual peaks in 2D images."""
        # Create a synthetic residual with clear peaks
        residual = torch.zeros((20, 20))
        residual[5, 5] = 1.0  # Peak 1
        residual[15, 15] = 0.8  # Peak 2
        residual[10, 10] = 0.6  # Peak 3

        peaks = _find_residual_peaks(residual, k_max_residuals=3, nms_radius_vox=2.0)

        assert len(peaks) <= 3
        assert len(peaks) > 0

        # Check that peaks contain the expected locations (peaks is now a tensor)
        peak_tuples = {tuple(p.tolist()) for p in peaks}
        assert (5, 5) in peak_tuples
        assert (15, 15) in peak_tuples

    def test_find_residual_peaks_3d(self) -> None:
        """Test finding residual peaks in 3D volumes."""
        # Create a synthetic 3D residual
        residual = torch.zeros((10, 10, 10))
        residual[5, 5, 5] = 1.0  # Peak 1
        residual[2, 2, 2] = 0.7  # Peak 2

        peaks = _find_residual_peaks(residual, k_max_residuals=2, nms_radius_vox=1.5)

        assert len(peaks) <= 2
        assert len(peaks) > 0

        # Check that we get 3D coordinates (peaks is now a tensor of shape (K, 3))
        assert peaks.shape[1] == 3

    def test_find_residual_peaks_empty(self) -> None:
        """Test behavior with no significant peaks."""
        residual = torch.zeros((10, 10))
        peaks = _find_residual_peaks(residual, k_max_residuals=5, nms_radius_vox=2.0)

        # Should return empty list for zero residual
        assert len(peaks) == 0


class TestTiledSeedingReproducibility:
    """Tiled probabilistic peak selection must be reproducible under a fixed seed.

    When k_per_tile < 1 the tiled seeder keeps each weak peak with probability
    k_per_tile via an RNG. Before seeding was threaded through, that draw used
    the unseeded global ``random`` module, so two fits with identical inputs
    could select different peaks. These tests pin the seeded contract.
    """

    @staticmethod
    def _spread_amplitude_residual() -> torch.Tensor:
        """64 peaks on an 8x8 grid with amplitudes 1..64.

        The amplitudes span the full range so the 75th-percentile "strong peak"
        threshold leaves a clear majority of weak peaks subject to the
        probabilistic keep decision — i.e. the seeded branch is actually
        exercised. Peaks are spaced 8 voxels apart so each survives NMS and
        lands in its own tile (auto tiling for 2D is 16x16 = 256 tiles).
        """
        residual = torch.zeros((64, 64))
        amp = 1
        for i in range(4, 64, 8):
            for j in range(4, 64, 8):
                residual[i, j] = float(amp)
                amp += 1
        return residual

    def _find(self, residual: torch.Tensor, seed: int | None) -> torch.Tensor:
        # k_max=40 over 256 tiles → keep_probability ≈ 0.156 → probabilistic mode.
        return _find_residual_peaks(
            residual,
            k_max_residuals=40,
            nms_radius_vox=1.0,
            enable_tiled=True,
            seed=seed,
        )

    def test_config_default_seed_is_42(self) -> None:
        """Dynamic seeding is reproducible by default (matches FPS seed convention)."""
        assert DynamicOpsConfig().seed == 42

    def test_same_seed_is_deterministic(self) -> None:
        """Identical input + identical seed → byte-identical peak selection."""
        residual = self._spread_amplitude_residual()
        first = self._find(residual, seed=123)
        second = self._find(residual, seed=123)
        assert torch.equal(first, second)

    def test_different_seeds_change_weak_peak_selection(self) -> None:
        """The seed actually drives the probabilistic draw, not just the API."""
        residual = self._spread_amplitude_residual()
        signatures = {
            tuple(sorted(tuple(p.tolist()) for p in self._find(residual, seed=s)))
            for s in (1, 2, 3, 4)
        }
        # With ~48 weak peaks each kept ~15.6% of the time, distinct seeds
        # almost surely disagree; a single signature would mean the seed is
        # being ignored.
        assert len(signatures) > 1

    def test_strong_peaks_always_kept_regardless_of_seed(self) -> None:
        """Above-threshold peaks bypass the probabilistic draw (invariant)."""
        residual = self._spread_amplitude_residual()
        # The global maximum (amplitude 64 at (60, 60)) is well above the 75th
        # percentile and must be retained under every seed, including unseeded.
        for seed in (1, 7, 999, None):
            peak_tuples = {tuple(p.tolist()) for p in self._find(residual, seed)}
            assert (60, 60) in peak_tuples

    def test_count_stays_within_candidate_bound(self) -> None:
        """Selection never invents peaks beyond the candidate set (invariant)."""
        residual = self._spread_amplitude_residual()
        peaks = self._find(residual, seed=42)
        # 64 candidate peaks total; cannot exceed that, and strong peaks
        # guarantee a non-empty result.
        assert 0 < len(peaks) <= 64


class TestSimplifiedSeeding:
    """Test ultra-simple seeding approach with direct amplitude and isotropic shape."""

    def test_simple_amplitude_estimation(self) -> None:
        """Test direct amplitude estimation from residual center value."""
        # Create synthetic residual with known peak
        residual = torch.zeros((20, 20))
        residual[10, 10] = 0.75  # Known residual value

        # Simple amplitude should equal residual value at center
        center = torch.tensor([10.0, 10.0])
        center_coords = torch.round(center).long()
        amplitude = torch.abs(residual[tuple(center_coords)])

        assert amplitude.item() == 0.75  # Should exactly match residual value
        assert amplitude.item() > 0

    def test_isotropic_shape_generation(self) -> None:
        """Test isotropic covariance matrix generation."""
        cfg = DynamicOpsConfig()
        center = torch.tensor([10.0, 10.0])
        d = len(center)

        # Simple isotropic covariance
        L = torch.eye(d, device=center.device) * cfg.init_sigma_vox

        assert L.shape == (2, 2)
        assert torch.allclose(L, torch.eye(2) * 0.5)  # Should be identity scaled


class TestSplatImportanceCalculation:
    """Test splat importance (amplitude × volume) calculation."""

    def test_calculate_importance(self) -> None:
        """Test importance calculation for varying splats."""
        # Create model with varying properties
        centers = np.array([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]])
        # Different sizes: small, medium, large
        L0 = np.stack(
            [
                np.eye(2) * 0.5,  # Small
                np.eye(2) * 1.0,  # Medium
                np.eye(2) * 2.0,  # Large
            ]
        )
        # Different amplitudes
        amps0 = np.array([1.0, 0.1, 0.5])

        model = GaussianSplatModel(
            shape=(20, 20),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.1],
        )

        # Get cached params and pass to importance function
        _, Ls, amps = model.current_params()
        importance = _calculate_splat_importance(Ls, amps)

        assert importance.shape == (3,)
        assert torch.all(importance >= 0)

        # Verify relative ordering: importance = amplitude × volume
        # Splat 0: 1.0 × (0.5 × 0.5) = 0.25
        # Splat 1: 0.1 × (1.0 × 1.0) = 0.1
        # Splat 2: 0.5 × (2.0 × 2.0) = 2.0
        # So ordering should be: splat 1 < splat 0 < splat 2
        sorted_indices = torch.argsort(importance)
        assert sorted_indices[0].item() == 1  # Least important
        assert sorted_indices[2].item() == 2  # Most important


class TestWeakSplatSelection:
    """Test weak splat selection for relocation."""

    def test_select_weak_splats(self) -> None:
        """Test that weak splats are correctly identified."""
        # Create random importance values
        importance = torch.tensor([0.5, 0.1, 0.9, 0.2, 0.8])
        centers = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )
        residual = torch.zeros((5, 5))

        # Select bottom 40% (2 splats)
        weak_indices = _select_weak_splats(
            importance, centers, residual, relocation_percentile=40.0
        )

        assert len(weak_indices) == 2  # 40% of 5 = 2
        assert weak_indices.tolist() == [1, 3]

    def test_select_weak_splats_is_reproducible_under_a_fixed_seed(self) -> None:
        """A seeded call must be bit-reproducible.

        The residual-percentile sample draws 10k voxels out of the volume. It
        used to draw from the global torch RNG, which made every default fit
        non-reproducible even though ``DynamicOpsConfig.seed`` documents a
        fixed default for exactly this reason. Regression for that.

        The residual must be LARGER than the 10k sample cap and have a spread
        of positive values, or the sampling branch is never taken and the test
        passes vacuously.
        """
        n_splats = 400
        torch.manual_seed(0)
        importance = torch.rand(n_splats)
        centers = torch.rand(n_splats, 3) * 20.0
        residual = torch.rand(21, 21, 41)  # 18_081 voxels > the 10_000 cap
        assert residual.numel() > 10_000, "sampling branch must be exercised"

        first = _select_weak_splats(
            importance, centers, residual, relocation_percentile=25.0, seed=1234
        )
        second = _select_weak_splats(
            importance, centers, residual, relocation_percentile=25.0, seed=1234
        )
        assert torch.equal(first, second)

        # Every seed must be reproducible, not just this one.
        other = _select_weak_splats(
            importance, centers, residual, relocation_percentile=25.0, seed=99
        )
        assert torch.equal(
            other,
            _select_weak_splats(
                importance, centers, residual, relocation_percentile=25.0, seed=99
            ),
        )
        # And the seed must actually reach the sampler. A different seed draws
        # a different residual sample, so the estimated 25th-percentile cutoff
        # moves and a different number of splats survives the filter. If this
        # ever stops differing, the parameter has been silently dropped.
        assert not torch.equal(first, other)

    def test_select_weak_splats_uses_global_rng_when_unseeded(self) -> None:
        """An unseeded call must consume PyTorch's global RNG stream."""
        data_rng = torch.Generator().manual_seed(7)
        n_splats = 400
        importance = torch.rand(n_splats, generator=data_rng)
        centers = torch.rand(n_splats, 3, generator=data_rng) * 20.0
        residual = torch.rand((21, 21, 41), generator=data_rng)
        assert residual.numel() > 10_000, "sampling branch must be exercised"

        torch.manual_seed(2)
        rng_state_before = torch.get_rng_state()
        _select_weak_splats(
            importance, centers, residual, relocation_percentile=25.0, seed=None
        )

        assert not torch.equal(torch.get_rng_state(), rng_state_before)

    def test_dynamic_ops_advances_its_seed_between_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The config seed reaches both samplers and advances reproducibly."""
        from luxar.gsplats.fitting.dynamic_ops import operations

        seen_peak_seeds = []
        seen_weak_seeds = []

        def fake_find_residual_peaks(*_args, seed=None, **_kwargs):
            seen_peak_seeds.append(seed)
            return torch.tensor([[0, 0]])

        def fake_select_weak_splats(*_args, seed=None, **_kwargs):
            seen_weak_seeds.append(seed)
            return torch.empty(0, dtype=torch.long)

        monkeypatch.setattr(
            operations, "_find_residual_peaks", fake_find_residual_peaks
        )
        monkeypatch.setattr(operations, "_select_weak_splats", fake_select_weak_splats)

        class Model:
            @staticmethod
            def current_params():
                return torch.zeros((1, 2)), torch.eye(2)[None], torch.ones(1)

        cfg = DynamicOpsConfig(seed=1234)
        tracker = RecentlyRelocatedTracker(1)
        target = torch.ones((2, 2))
        prediction = torch.zeros((2, 2))

        for _ in range(2):
            apply_dynamic_operations(
                Model(),
                target,
                prediction,
                cfg,
                max_abs_error_threshold=0.0,
                relocation_tracker=tracker,
            )
            tracker.advance_step()

        assert seen_peak_seeds == [1234, 1235]
        assert seen_weak_seeds == [1234, 1235]

    def test_select_weak_splats_minimum_one(self) -> None:
        """Test that at least one splat is always selected."""
        importance = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        centers = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
        )
        residual = torch.zeros((5, 5))

        # Even with very low percentile, should get at least 1
        weak_indices = _select_weak_splats(
            importance, centers, residual, relocation_percentile=1.0
        )

        assert weak_indices.tolist() == [0]


class TestGaussianSplatModel:
    """Test basic Gaussian splat model operations required for dynamic ops."""

    def test_n_splats(self) -> None:
        """Test counting splats."""
        centers = np.array([[5.0, 5.0], [10.0, 10.0]])
        L0 = np.eye(2)[None, :, :] * 1.0  # (1, 2, 2)
        L0 = np.repeat(L0, 2, axis=0)  # (2, 2, 2)
        amps0 = np.array([1.0, 1.0])
        model = GaussianSplatModel(
            shape=(20, 20),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )
        assert model.n_splats() == 2

    def test_current_params(self) -> None:
        """Test retrieving current parameters."""
        centers = np.array([[5.0, 5.0], [10.0, 10.0]])
        L0 = np.eye(2)[None, :, :] * 1.0  # (1, 2, 2)
        L0 = np.repeat(L0, 2, axis=0)  # (2, 2, 2)
        amps0 = np.array([1.0, 1.0])
        model = GaussianSplatModel(
            shape=(20, 20),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )

        centers_t, Ls_t, amps_t = model.current_params()

        assert centers_t.shape == (2, 2)
        assert Ls_t.shape == (2, 2, 2)
        assert amps_t.shape == (2,)


class TestDynamicOperationsIntegration:
    """Test the full dynamic operations pipeline."""

    def test_dynamic_ops_with_simple_model(self) -> None:
        """Test dynamic operations on a simple model."""
        # Create simple synthetic data
        V_target = torch.zeros((16, 16))
        V_target[8, 8] = 1.0  # Single bright spot

        # Create initial model with a few splats (one weak, one strong)
        centers = np.array([[7.0, 7.0], [9.0, 9.0], [3.0, 3.0]])  # Third is far away
        L0 = np.stack(
            [
                np.eye(2) * 2.0,  # Large
                np.eye(2) * 2.0,  # Large
                np.eye(2) * 0.3,  # Small (weak)
            ]
        )
        amps0 = np.array([1.0, 1.0, 0.01])  # Third is weak amplitude
        model = GaussianSplatModel(
            shape=(16, 16),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device("cpu"),
        )

        # Get current prediction
        V_pred = model()

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.k_max_residuals = 3
        cfg.relocation_percentile = 50.0  # Allow up to 50% for relocation
        cfg.min_contribution_threshold = 1e-6

        # Apply dynamic operations
        any_relocated = apply_dynamic_operations(
            model=model,
            V_target=V_target,
            V_pred=V_pred,
            cfg=cfg,
            max_abs_error_threshold=0.1,
            verbose=False,
        )

        # Check that we get valid return
        assert isinstance(any_relocated, bool)

    def test_fit_with_dynamic_ops(self) -> None:
        """Test full fitting pipeline with dynamic operations enabled."""
        # Create simple test data
        blob = np.zeros((32, 32))
        blob[16, 16] = 1.0
        blob[12, 12] = 0.8
        V = blob.astype(np.float32)

        # Find seeds (returns GSplatData with scale-informed shapes)
        seeds = seed_from_grid(V, spacing=5.0)

        if len(seeds.centers) == 0:
            pytest.skip("No seeds found for test data")

        # Configure dynamic operations
        cfg = DynamicOpsConfig()
        cfg.step_every = 5  # Run more frequently for testing

        # Run fitting with dynamic operations (GSplatData passed directly)
        result = fit_gaussian_splats(
            V,
            seeds=seeds,
            init_sigma_vox=1.5,
            n_iters=20,  # Short run for testing
            lr=0.1,
            enable_dynamic_ops=True,
            dynamic_config=cfg,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results
        assert result.centers.shape[0] > 0  # Should have some splats
        assert result.amplitudes.shape[0] == result.centers.shape[0]
        assert "final_loss" in result.stats  # Check for stats that actually exist

    def test_fit_without_dynamic_ops(self) -> None:
        """Test fitting without dynamic operations for comparison."""
        # Create test data with a broader blob (not single-pixel) so splats
        # can develop meaningful amplitudes above the noise floor in few iterations
        y, x = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        V = np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / (2 * 3**2)).astype(np.float32)

        # Find seeds (returns GSplatData)
        seeds = seed_from_grid(V, spacing=8.0)

        if len(seeds.centers) == 0:
            pytest.skip("No seeds found for test data")

        # Run fitting without dynamic operations
        result = fit_gaussian_splats(
            V,
            seeds=seeds,
            init_sigma_vox=1.5,
            n_iters=10,
            lr=0.1,
            enable_dynamic_ops=False,
            verbose=False,
            napari_movie=False,  # Disable movie for tests
        )

        # Check that we got valid results (post-fit culling may reduce count)
        assert result.amplitudes.shape[0] == result.centers.shape[0]
        assert "final_loss" in result.stats  # Check for stats that actually exist

    def test_weak_splat_identification(self) -> None:
        """Test the weak splat identification algorithm."""
        # Create test model with varying importance splats
        V = np.random.random((32, 32)).astype(np.float32)
        seeds = seed_from_grid(V, spacing=4.0)
        centers = seeds.centers

        # Create model with many splats to test selection
        L0 = np.eye(2)[None, :, :] * 1.0
        L0 = np.repeat(L0, len(centers), axis=0)
        amps0 = np.random.uniform(0.01, 1.0, len(centers)).astype(
            np.float32
        )  # Varying amplitudes

        model = GaussianSplatModel(
            shape=(32, 32),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
        )

        # Test importance calculation - get cached params first
        centers_t, Ls, amps_t = model.current_params()
        importance = _calculate_splat_importance(Ls, amps_t)
        assert importance.shape == (len(centers),)
        assert torch.all(importance >= 0)

        # Test weak splat selection
        residual = torch.full((32, 32), 0.5, device=centers_t.device)
        weak_indices = _select_weak_splats(
            importance, centers_t, residual, 10.0
        )  # Bottom 10%
        expected_candidates = max(1, int(len(centers) * 0.1))
        assert len(weak_indices) == expected_candidates

        sorted_indices = torch.argsort(importance).tolist()
        assert weak_indices.tolist() == sorted_indices[:expected_candidates]


class TestRelocationParameters:
    """Test parameter resets during relocation."""

    def test_relocation_respects_per_axis_sigma_min(self) -> None:
        """Ensure init sigma reset respects per-axis sigma_min_diag."""
        centers = np.array([[1.0, 1.0]], dtype=np.float32)
        L0 = np.eye(2, dtype=np.float32)[None, :, :] * 0.2
        amps0 = np.array([0.1], dtype=np.float32)

        model = GaussianSplatModel(
            shape=(8, 8),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.5],
            device=torch.device("cpu"),
        )

        residual = torch.zeros((8, 8), dtype=torch.float32)
        residual[3, 4] = 1.0

        cfg = DynamicOpsConfig()
        cfg.init_sigma_vox = 0.6

        # Use batched version with single splat
        splat_indices = torch.tensor([0], dtype=torch.long, device=torch.device("cpu"))
        peak_coords = torch.tensor(
            [[3.0, 4.0]], dtype=torch.float32, device=torch.device("cpu")
        )
        _relocate_splats_batch(model, splat_indices, peak_coords, residual, cfg)

        _, Ls, _ = model.current_params()
        diag = torch.diagonal(Ls[0], dim1=-2, dim2=-1)
        assert torch.allclose(
            diag, torch.tensor([0.6, 0.6], dtype=diag.dtype), atol=1e-6
        )

    def test_asymmetric_penalty_with_all_loss_types(self) -> None:
        """Test asymmetric penalty works with all loss functions."""
        V = np.random.random((24, 24)).astype(np.float32)
        seeds = seed_from_grid(V, spacing=5.0)

        for loss_type in ["mse", "poisson", "l1"]:
            result = fit_gaussian_splats(
                V,
                seeds=seeds,
                n_iters=10,
                loss_type=loss_type,
                asymmetric_penalty=5.0,  # Test with asymmetric penalty
                verbose=False,
                enable_dynamic_ops=False,
                napari_movie=False,
            )

            assert len(result.amplitudes) > 0, (
                f"{loss_type} with asymmetric penalty failed"
            )
            assert all(result.amplitudes >= 0), (
                f"{loss_type} produced negative amplitudes"
            )

    def test_relocation_behavior(self) -> None:
        """Test the splat relocation algorithm."""
        # Create test data where some splats should be relocatable
        V = np.ones((32, 32), dtype=np.float32) * 0.5  # Uniform background
        V[25, 25] = 2.0  # Add a bright spot that needs coverage

        # Create splats - one strong, two weak
        centers = np.array([[10, 10], [15, 15], [3, 3]], dtype=np.float32)
        L0 = np.stack(
            [
                np.eye(2) * 2.0,  # Large
                np.eye(2) * 0.3,  # Small (weak)
                np.eye(2) * 0.2,  # Very small (weak)
            ]
        )
        amps0 = np.array([0.8, 0.01, 0.005])  # Strong, weak, very weak

        model = GaussianSplatModel(
            shape=(32, 32),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.1],
            device=torch.device("cpu"),
        )

        V_target = torch.tensor(V, dtype=torch.float32, device=torch.device("cpu"))
        V_pred = model()

        cfg = DynamicOpsConfig()
        cfg.relocation_percentile = 70.0  # Allow more aggressive relocation for test
        cfg.max_relocations_per_step = 5

        # Apply dynamic operations with verbose output
        any_relocated = apply_dynamic_operations(
            model,
            V_target,
            V_pred,
            cfg,
            max_abs_error_threshold=0.01,
            verbose=True,  # Test verbose output
        )

        # Verify function returned successfully
        assert isinstance(any_relocated, bool)

    def test_auto_convergence_threshold_behavior(self) -> None:
        """Test auto-convergence threshold integration with dynamic operations."""
        V = np.random.random((24, 24)).astype(np.float32)
        seeds = seed_from_grid(V, spacing=4.0)

        # Test that auto-threshold works with dynamic operations
        result = fit_gaussian_splats(
            V,
            seeds=seeds,
            n_iters=50,
            max_abs_error=None,  # Should auto-set to 0.01
            loss_type="l1",
            enable_dynamic_ops=True,  # Enable to test interaction
            dynamic_ops_verbose=True,  # Test verbose output
            verbose=True,  # Should log auto-threshold
            napari_movie=False,
        )

        assert "converged" in result.stats
        assert len(result.amplitudes) > 0

    def test_compression_analysis_functionality(self) -> None:
        """Test compression ratio analysis functionality."""
        from luxar.gsplats.fitting.visualization import display_compression_analysis
        from luxar.gsplats.gsplat_data import GSplatData

        # Create simple test data
        V = np.random.random((16, 16)).astype(np.float32)
        d = 2
        N = 10

        # Create GSplatData for testing
        result = GSplatData(
            centers=np.random.random((N, d)).astype(np.float32),
            amplitudes=np.random.uniform(0.1, 1.0, N).astype(np.float32),
            cholesky_factors=np.random.random((N, 3)).astype(np.float32),  # 2D tril = 3
            stats={},
        )

        # Test compression analysis (should not raise exceptions)
        try:
            display_compression_analysis(V, result)
            compression_test_passed = True
        except Exception:
            compression_test_passed = False

        assert compression_test_passed, "Compression analysis failed"

    def test_convergence_guard_skips_when_converged(self) -> None:
        """Test that dynamic ops are skipped when residual is below threshold."""
        # Create a well-reconstructed image (low residual)
        V_target = torch.ones((16, 16)) * 0.5
        V_pred = V_target.clone()  # Perfect prediction

        centers = np.array([[8, 8]], dtype=np.float32)
        L0 = np.eye(2)[None, :, :] * 1.0
        amps0 = np.array([0.5])

        model = GaussianSplatModel(
            shape=(16, 16),
            centers0=centers,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.1, 0.1],
        )

        cfg = DynamicOpsConfig()
        cfg.relocation_percentile = 100.0  # Would relocate all if not for guard

        # Should return False (no relocation) because residual is zero
        any_relocated = apply_dynamic_operations(
            model,
            V_target,
            V_pred,
            cfg,
            max_abs_error_threshold=0.01,
            verbose=False,
        )

        assert any_relocated is False  # Should skip due to convergence guard
