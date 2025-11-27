"""Comprehensive tests for PerSplatAdam serialization, AMSGrad, and edge cases."""

import numpy as np
import pytest
import torch


class TestPerSplatAdamStateDictSerialization:
    """Test state_dict() and load_state_dict() methods."""

    def create_simple_model(self, n_splats=5, device="cpu"):  # type: ignore[no-untyped-def]
        """Helper to create a simple model for testing."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        shape = (32, 32)

        # Handle empty model case
        if n_splats == 0:
            centers0 = np.empty((0, 2), dtype=np.float32)
            L0 = np.empty((0, 2, 2), dtype=np.float32)
            amps0 = np.empty(0, dtype=np.float32)
        else:
            centers0 = np.random.uniform(5, 25, (n_splats, 2)).astype(np.float32)
            L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
            amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        return GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device(device),
        )

    def test_state_dict_basic(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test basic state_dict extraction."""

        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Get state dict
        state = optimizer.state_dict()

        # Verify all required keys are present
        assert "splat_states" in state
        assert "global_step" in state
        assert "base_lr" in state
        assert "betas" in state
        assert "eps" in state
        assert "weight_decay" in state
        assert "amsgrad" in state

        # Verify values
        assert state["global_step"] == 0
        assert state["base_lr"] == 0.01
        assert state["betas"] == (0.9, 0.999)
        assert state["eps"] == 1e-8
        assert state["weight_decay"] == 0.0
        assert state["amsgrad"] is False

    def test_state_dict_empty_model(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test state_dict with model that has no splats (edge case)."""
        model = self.create_simple_model(n_splats=0)
        optimizer = PerSplatAdam(model, lr=0.01)

        state = optimizer.state_dict()

        # Should still have all keys even with empty model
        assert "splat_states" in state
        assert len(state["splat_states"]) == 0
        assert state["global_step"] == 0

    def test_state_dict_large_model(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test state_dict with large model (many splats)."""
        model = self.create_simple_model(n_splats=100)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Run a few steps to populate state
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(3):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        state = optimizer.state_dict()

        # Verify large state is captured
        assert len(state["splat_states"]) == 100
        assert state["global_step"] == 3

    def test_state_dict_after_optimization(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that state_dict captures optimizer state after optimization steps."""
        model = self.create_simple_model(n_splats=5)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Run optimization to populate momentum buffers
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        state = optimizer.state_dict()

        # Check that state was populated
        assert state["global_step"] == 5
        assert len(state["splat_states"]) == 5

        # Check that momentum buffers exist for at least one splat
        for splat_state in state["splat_states"].values():
            assert "exp_avg_mu" in splat_state
            assert "exp_avg_sq_mu" in splat_state
            assert "step" in splat_state
            # Momentum should be non-zero after optimization
            assert splat_state["step"] == 5

    def test_load_state_dict_basic(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test basic state_dict loading."""
        model = self.create_simple_model(n_splats=5)
        optimizer1 = PerSplatAdam(model, lr=0.01)

        # Run some optimization
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(3):
            optimizer1.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer1.step()

        # Save state
        state = optimizer1.state_dict()

        # Create new optimizer and load state
        model2 = self.create_simple_model(n_splats=5)
        optimizer2 = PerSplatAdam(model2, lr=0.02)  # Different LR initially

        optimizer2.load_state_dict(state)

        # Verify state was loaded correctly
        assert optimizer2.global_step == 3
        assert optimizer2.base_lr == 0.01  # Should match loaded state
        assert len(optimizer2.splat_states) == 5

    def test_load_state_dict_preserves_momentum(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that loading state_dict preserves momentum buffers."""
        model1 = self.create_simple_model(n_splats=3)
        optimizer1 = PerSplatAdam(model1, lr=0.01)

        # Run optimization to build momentum
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(10):
            optimizer1.zero_grad()
            pred = model1()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer1.step()

        # Save and load state
        state = optimizer1.state_dict()
        model2 = self.create_simple_model(n_splats=3)
        optimizer2 = PerSplatAdam(model2, lr=0.01)
        optimizer2.load_state_dict(state)

        # Continue optimization with loaded optimizer
        loss1 = None
        for _ in range(5):
            optimizer2.zero_grad()
            pred = model2()
            loss1 = torch.nn.functional.mse_loss(pred, target)
            loss1.backward()  # type: ignore[no-untyped-call]
            optimizer2.step()

        # Verify optimization continues (loss should be finite, not NaN)
        assert not torch.isnan(loss1)
        assert optimizer2.global_step == 15  # 10 + 5 steps

    def test_state_dict_with_different_hyperparameters(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test state_dict with non-default hyperparameters."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(
            model,
            lr=0.001,
            betas=(0.95, 0.9995),
            eps=1e-10,
            weight_decay=0.01,
            amsgrad=False,
        )

        state = optimizer.state_dict()

        # Verify custom hyperparameters are saved
        assert state["base_lr"] == 0.001
        assert state["betas"] == (0.95, 0.9995)
        assert state["eps"] == 1e-10
        assert state["weight_decay"] == 0.01
        assert state["amsgrad"] is False

    def test_load_state_dict_with_mismatched_splats(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test loading state_dict into optimizer with different number of splats."""
        # Create optimizer with 5 splats
        model1 = self.create_simple_model(n_splats=5)
        optimizer1 = PerSplatAdam(model1, lr=0.01)

        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(3):
            optimizer1.zero_grad()
            pred = model1()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer1.step()

        state = optimizer1.state_dict()

        # Create optimizer with 3 splats
        model2 = self.create_simple_model(n_splats=3)
        optimizer2 = PerSplatAdam(model2, lr=0.01)

        # Load state (should work, but only relevant states will be used)
        optimizer2.load_state_dict(state)

        # Verify basic state was loaded
        assert optimizer2.global_step == 3
        assert optimizer2.base_lr == 0.01

        # Splat states will be mismatched, but optimizer should still function
        # (this tests robustness)
        optimizer2.zero_grad()
        pred = model2()
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer2.step()  # Should not crash

        assert optimizer2.global_step == 4


class TestPerSplatAdamAMSGrad:
    """Test AMSGrad variant functionality."""

    def create_simple_model(self, n_splats=5):
        """Helper to create a simple model."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        shape = (32, 32)
        centers0 = np.random.uniform(5, 25, (n_splats, 2)).astype(np.float32)
        L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
        amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        return GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device("cpu"),
        )

    def test_amsgrad_initialization(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that AMSGrad variant initializes correctly."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01, amsgrad=True)

        assert optimizer.amsgrad is True

        state = optimizer.state_dict()
        assert state["amsgrad"] is True

    def test_amsgrad_state_tracking(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that AMSGrad maintains max_exp_avg_sq buffers."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01, amsgrad=True)

        # Run optimization to populate AMSGrad state
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        # Check that max_exp_avg_sq buffers exist
        for splat_state in optimizer.splat_states.values():
            # AMSGrad should have max buffers
            assert "max_exp_avg_sq_mu" in splat_state
            assert "max_exp_avg_sq_L_diag" in splat_state
            assert "max_exp_avg_sq_L_off" in splat_state
            assert "max_exp_avg_sq_a" in splat_state

    def test_amsgrad_vs_standard_convergence(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that AMSGrad variant converges (basic functionality check)."""
        model_standard = self.create_simple_model(n_splats=3)
        model_amsgrad = self.create_simple_model(n_splats=3)

        optimizer_standard = PerSplatAdam(model_standard, lr=0.01, amsgrad=False)
        optimizer_amsgrad = PerSplatAdam(model_amsgrad, lr=0.01, amsgrad=True)

        target = torch.randn_like(torch.zeros((32, 32)))

        # Run both optimizers
        loss_standard = None
        loss_amsgrad = None

        for _ in range(10):
            # Standard Adam
            optimizer_standard.zero_grad()
            pred = model_standard()
            loss_standard = torch.nn.functional.mse_loss(pred, target)
            loss_standard.backward()  # type: ignore[no-untyped-call]
            optimizer_standard.step()

            # AMSGrad
            optimizer_amsgrad.zero_grad()
            pred = model_amsgrad()
            loss_amsgrad = torch.nn.functional.mse_loss(pred, target)
            loss_amsgrad.backward()  # type: ignore[no-untyped-call]
            optimizer_amsgrad.step()

        # Both should converge (losses should be finite)
        assert not torch.isnan(loss_standard)
        assert not torch.isnan(loss_amsgrad)
        assert loss_standard < 100.0  # Should have made progress
        assert loss_amsgrad < 100.0

    def test_amsgrad_state_dict_serialization(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test that AMSGrad state is properly serialized."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01, amsgrad=True)

        # Run optimization
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        # Save and load state
        state = optimizer.state_dict()
        assert state["amsgrad"] is True

        # Load into new optimizer
        model2 = self.create_simple_model(n_splats=3)
        optimizer2 = PerSplatAdam(model2, lr=0.01, amsgrad=True)
        optimizer2.load_state_dict(state)

        # Verify AMSGrad state was preserved
        assert optimizer2.amsgrad is True
        for splat_state in optimizer2.splat_states.values():
            assert "max_exp_avg_sq_mu" in splat_state


class TestPerSplatAdamEdgeCases:
    """Test edge cases and error handling."""

    def create_simple_model(self, n_splats=5):
        """Helper to create a simple model."""
        from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

        shape = (32, 32)

        # Handle empty model case
        if n_splats == 0:
            centers0 = np.empty((0, 2), dtype=np.float32)
            L0 = np.empty((0, 2, 2), dtype=np.float32)
            amps0 = np.empty(0, dtype=np.float32)
        else:
            centers0 = np.random.uniform(5, 25, (n_splats, 2)).astype(np.float32)
            L0 = np.stack([np.eye(2) * 1.5] * n_splats).astype(np.float32)
            amps0 = np.random.uniform(0.1, 1.0, n_splats).astype(np.float32)

        return GaussianSplatModel(
            shape=shape,
            centers0=centers0,
            L0=L0,
            amps0=amps0,
            sigma_min_diag=[0.5, 0.5],
            device=torch.device("cpu"),
        )

    def test_optimizer_with_zero_splats(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test optimizer creation and operation with zero splats."""
        model = self.create_simple_model(n_splats=0)
        optimizer = PerSplatAdam(model, lr=0.01)

        assert optimizer.model.n_splats() == 0
        assert len(optimizer.splat_states) == 0

        # Should be able to get state dict even with no splats
        state = optimizer.state_dict()
        assert len(state["splat_states"]) == 0

    def test_adding_splats_to_empty_model(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test adding splats to a model that started empty."""
        model = self.create_simple_model(n_splats=0)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Add splats
        centers_new = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
        Ls_new = torch.stack([torch.eye(2) * 1.0, torch.eye(2) * 1.2], dim=0)
        amps_new = torch.tensor([0.5, 0.7], dtype=torch.float32)
        sharpness_new = torch.tensor([2.0, 2.0], dtype=torch.float32)

        model.append_(centers_new, Ls_new, amps_new, sharpness_new)
        optimizer.add_splats(2, lr_new=0.1)

        assert model.n_splats() == 2
        assert len(optimizer.splat_states) == 2

    def test_extreme_learning_rate(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test optimizer with extreme learning rates."""
        model = self.create_simple_model(n_splats=3)

        # Very small LR
        optimizer_small = PerSplatAdam(model, lr=1e-10)
        assert optimizer_small.base_lr == 1e-10

        # Very large LR (might be unstable, but should initialize)
        model2 = self.create_simple_model(n_splats=3)
        optimizer_large = PerSplatAdam(model2, lr=100.0)
        assert optimizer_large.base_lr == 100.0

    def test_extreme_beta_values(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test optimizer with extreme beta values."""
        model = self.create_simple_model(n_splats=3)

        # Very low beta1 (less momentum)
        optimizer = PerSplatAdam(model, lr=0.01, betas=(0.1, 0.999))
        assert optimizer.betas == (0.1, 0.999)

        # Very high beta1 (more momentum)
        model2 = self.create_simple_model(n_splats=3)
        optimizer2 = PerSplatAdam(model2, lr=0.01, betas=(0.999, 0.9999))
        assert optimizer2.betas == (0.999, 0.9999)

    def test_weight_decay_nonzero(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test optimizer with non-zero weight decay."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01, weight_decay=0.01)

        assert optimizer.weight_decay == 0.01

        # Run optimization with weight decay
        target = torch.randn_like(torch.zeros((32, 32)))
        for _ in range(5):
            optimizer.zero_grad()
            pred = model()
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        # Should complete without errors
        assert optimizer.global_step == 5

    def test_get_effective_learning_rates(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test get_effective_learning_rates() method."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01)

        lrs = optimizer.get_effective_learning_rates()

        # Should have one LR per splat
        assert len(lrs) == 3

        # All should be positive
        assert all(lr > 0 for lr in lrs)

    def test_multiple_zero_grad_calls(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test calling zero_grad() multiple times."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01)

        # Should be safe to call multiple times
        optimizer.zero_grad()
        optimizer.zero_grad()
        optimizer.zero_grad()

        # No assertion needed, just checking it doesn't crash

    def test_step_without_backward(self) -> None:
        from luxar.gsplats.optim.per_splat_adam import PerSplatAdam

        """Test calling step() without backward() (edge case)."""
        model = self.create_simple_model(n_splats=3)
        optimizer = PerSplatAdam(model, lr=0.01)

        optimizer.zero_grad()
        # Don't call backward()
        # Calling step() should work (gradients will be None/zero)
        optimizer.step()

        # Global step should not increment without gradients (correct behavior)
        assert optimizer.global_step == 0  # No work done, no step increment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
