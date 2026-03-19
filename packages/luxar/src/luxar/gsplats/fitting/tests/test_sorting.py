"""Tests for Z-order (Morton code) sorting of splats during fitting."""

import numpy as np
import torch

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig, RecentlyRelocatedTracker
from luxar.gsplats.fitting.initialization import initialize_optimization
from luxar.gsplats.fitting.sorting import sort_splats_by_morton_order


def _make_config_and_data(n_splats=20, shape=(32, 32)):
    """Create a FitConfig and PreprocessedData for testing."""
    d = len(shape)
    V = np.random.rand(*shape).astype(np.float32)

    # Generate random seed centers inside the volume
    seed_centers = np.column_stack(
        [np.random.uniform(1, s - 1, n_splats) for s in shape]
    ).astype(np.float32)

    config = FitConfig(
        V=V,
        seeds=seed_centers,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5] * d,
        sigma_max_diag=[10.0] * d,
        truncate=3.0,
        n_iters=10,
        lr=0.01,
        max_abs_error=0.01,
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="l1",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        scheduler_type="plateau",
        patience=10,
        lr_reduction_factor=0.5,
        early_stop_patience=None,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
    )

    preprocessed = PreprocessedData(
        V_normalized=V,
        V_tensor=torch.tensor(V),
        seed_centers=seed_centers,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        d=d,
        N=n_splats,
        max_abs_error=0.01,
    )

    return config, preprocessed


class TestSortSplats:
    def test_sort_preserves_forward_output(self):
        """Model output should be identical before and after sorting."""
        config, preprocessed = _make_config_and_data()
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        with torch.no_grad():
            output_before = model().clone()

        sort_splats_by_morton_order(model, optimizer)

        with torch.no_grad():
            output_after = model()

        torch.testing.assert_close(output_before, output_after)

    def test_sort_permutes_all_parameters(self):
        """All 4 model parameters should be reordered."""
        config, preprocessed = _make_config_and_data()
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        # Store original parameter values
        orig_params = {
            "raw_mu": model.raw_mu.data.clone(),
            "raw_a": model.raw_a.data.clone(),
            "raw_L_diag": model.raw_L_diag.data.clone(),
            "L_off": model.L_off.data.clone(),
        }

        sort_splats_by_morton_order(model, optimizer)

        # Parameters should contain the same values but potentially reordered
        # (set of rows should be the same)
        assert torch.allclose(
            model.raw_mu.data.sort(dim=0).values,
            orig_params["raw_mu"].sort(dim=0).values,
        )
        assert torch.allclose(
            model.raw_a.data.sort().values,
            orig_params["raw_a"].sort().values,
        )
        assert torch.allclose(
            model.raw_L_diag.data.sort(dim=0).values,
            orig_params["raw_L_diag"].sort(dim=0).values,
        )
        assert torch.allclose(
            model.L_off.data.sort(dim=0).values,
            orig_params["L_off"].sort(dim=0).values,
        )

    def test_sort_permutes_optimizer_state(self):
        """Adam optimizer state should be permuted to match parameters."""
        config, preprocessed = _make_config_and_data()
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        # Run a few optimizer steps to populate state
        for _ in range(3):
            optimizer.zero_grad()
            pred = model()
            loss = pred.sum()
            loss.backward()
            optimizer.step()

        # Verify state exists
        assert model.raw_mu in optimizer.state
        orig_exp_avg = optimizer.state[model.raw_mu]["exp_avg"].clone()

        sort_splats_by_morton_order(model, optimizer)

        # State should contain same values (possibly reordered)
        new_exp_avg = optimizer.state[model.raw_mu]["exp_avg"]
        assert torch.allclose(
            new_exp_avg.sort(dim=0).values,
            orig_exp_avg.sort(dim=0).values,
        )

    def test_sort_permutes_relocation_tracker(self):
        """Relocation tracker per-splat state should be permuted."""
        n_splats = 20
        config, preprocessed = _make_config_and_data(n_splats=n_splats)
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        tracker = RecentlyRelocatedTracker(
            n_splats=n_splats,
            cooldown_steps=3,
            device="cpu",
        )
        # Set some non-zero tracker state
        tracker.last_relocation_step[5] = 10
        tracker.last_relocation_step[15] = 20
        orig_values = tracker.last_relocation_step.clone()

        sort_splats_by_morton_order(model, optimizer, tracker)

        # Same values should be present (possibly reordered)
        assert torch.allclose(
            tracker.last_relocation_step.sort().values,
            orig_values.sort().values,
        )

    def test_sort_single_splat_noop(self):
        """Sorting a single splat should be a no-op."""
        config, preprocessed = _make_config_and_data(n_splats=1)
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        orig_mu = model.raw_mu.data.clone()
        sort_splats_by_morton_order(model, optimizer)
        torch.testing.assert_close(model.raw_mu.data, orig_mu)

    def test_sort_3d(self):
        """Sorting should work for 3D data."""
        config, preprocessed = _make_config_and_data(n_splats=30, shape=(16, 16, 16))
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        with torch.no_grad():
            output_before = model().clone()

        sort_splats_by_morton_order(model, optimizer)

        with torch.no_grad():
            output_after = model()

        torch.testing.assert_close(output_before, output_after)

    def test_sort_integration_with_fitting(self):
        """Fitting with sorting enabled should converge."""
        from luxar.gsplats import fit_gaussian_splats

        V = np.random.rand(16, 16).astype(np.float32)
        result = fit_gaussian_splats(
            V,
            seeds=10,
            n_iters=50,
            sort_splats_enabled=True,
            sort_splats_interval=20,
            verbose=False,
        )
        assert result.centers.shape[0] > 0
        assert result.stats["iterations"] > 0

    def test_sort_degenerate_identical_centers(self):
        """Splats at identical positions should be a no-op (no crash)."""
        config, preprocessed = _make_config_and_data(n_splats=10, shape=(32, 32))
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        # Force all centers to the same position
        with torch.no_grad():
            model.raw_mu.data[:] = model.raw_mu.data[0]

        orig_mu = model.raw_mu.data.clone()
        sort_splats_by_morton_order(model, optimizer)
        # All centers identical → argsort is identity → no-op
        torch.testing.assert_close(model.raw_mu.data, orig_mu)

    def test_sort_does_not_alias_permutation(self):
        """In-place permutation must not corrupt data via aliasing."""
        config, preprocessed = _make_config_and_data(n_splats=50, shape=(32, 32))
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        # Capture all parameter values before sort
        mu_before = model.raw_mu.data.clone()
        a_before = model.raw_a.data.clone()
        diag_before = model.raw_L_diag.data.clone()
        off_before = model.L_off.data.clone()

        sort_splats_by_morton_order(model, optimizer)

        # After sorting, every original row must appear exactly once
        # Check via sorted values (multiset equality)
        torch.testing.assert_close(
            model.raw_mu.data.sort(dim=0).values,
            mu_before.sort(dim=0).values,
        )
        torch.testing.assert_close(
            model.raw_a.data.sort().values,
            a_before.sort().values,
        )
        torch.testing.assert_close(
            model.raw_L_diag.data.sort(dim=0).values,
            diag_before.sort(dim=0).values,
        )
        torch.testing.assert_close(
            model.L_off.data.sort(dim=0).values,
            off_before.sort(dim=0).values,
        )

    def test_sort_4d_high_dimension(self):
        """Sorting should work for 4D data (bits_per_dim capped at 16)."""
        config, preprocessed = _make_config_and_data(n_splats=20, shape=(8, 8, 8, 8))
        components = initialize_optimization(config, preprocessed)
        model = components.model
        optimizer = components.optimizer

        with torch.no_grad():
            output_before = model().clone()

        sort_splats_by_morton_order(model, optimizer)

        with torch.no_grad():
            output_after = model()

        torch.testing.assert_close(output_before, output_after)
