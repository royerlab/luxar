"""
Tests for fitting/losses.py module.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting.losses import (
    _compute_l1_loss,
    _compute_mse_loss,
    _compute_poisson_loss,
    create_loss_function,
)
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel


def create_test_config(V, **kwargs):
    """Helper to create a FitConfig with all required fields."""
    defaults = dict(
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5] * len(V.shape),
        sigma_max_diag=[10.0] * len(V.shape),
        truncate=3.0,
        n_iters=10,
        lr=0.01,
        max_abs_error=0.01,
        gradient_clip=None,
        loss_type="mse",
        asymmetric_penalty=None,
        l1_amp=None,
        l1_diag=None,
        l1_sharpness=None,
        scheduler_type="plateau",
        patience=10,
        factor=0.5,
        enable_dynamic_ops=False,
        dynamic_config=DynamicOpsConfig(),
        dynamic_ops_verbose=False,
        napari_movie=False,
        movie_every=1,
        movie_max_frames=100,
        device=torch.device("cpu"),
        verbose=False,
    )
    defaults.update(kwargs)
    return FitConfig(V=V, **defaults)


@pytest.fixture
def target_tensor():
    """Create a simple target tensor."""
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)


@pytest.fixture
def pred_tensor():
    """Create a prediction tensor."""
    return torch.tensor([[0.9, 2.2], [2.8, 4.1]], dtype=torch.float32)


@pytest.fixture
def basic_model():
    """Create a basic GaussianSplatModel for testing."""
    shape = (8, 8)
    N = 3
    d = 2
    centers0 = np.random.rand(N, d).astype(np.float32) * 6
    L0 = np.zeros((N, d, d), dtype=np.float32)
    for i in range(d):
        L0[:, i, i] = 1.0
    amps0 = np.ones(N, dtype=np.float32) * 0.5

    return GaussianSplatModel(
        shape=shape,
        centers0=centers0,
        L0=L0,
        amps0=amps0,
        sigma_min_diag=[0.5, 0.5],
        device="cpu",
    )


def test_mse_loss_basic(target_tensor, pred_tensor) -> None:
    """Test basic MSE loss without asymmetric penalty."""
    loss = _compute_mse_loss(pred_tensor, target_tensor, asymmetric_penalty=None)

    # Calculate expected MSE
    expected = torch.mean((pred_tensor - target_tensor) ** 2)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_mse_loss_asymmetric(target_tensor, pred_tensor) -> None:
    """Test MSE loss with asymmetric penalty."""
    asymmetric_penalty = 5.0
    loss = _compute_mse_loss(pred_tensor, target_tensor, asymmetric_penalty)

    # Loss should be higher than basic MSE due to penalties
    basic_loss = _compute_mse_loss(pred_tensor, target_tensor, None)
    assert loss >= basic_loss


def test_l1_loss_basic(target_tensor, pred_tensor) -> None:
    """Test basic L1 loss."""
    loss = _compute_l1_loss(pred_tensor, target_tensor, asymmetric_penalty=None)

    # Calculate expected L1
    expected = torch.mean(torch.abs(pred_tensor - target_tensor))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_l1_loss_asymmetric(target_tensor, pred_tensor) -> None:
    """Test L1 loss with asymmetric penalty."""
    asymmetric_penalty = 3.0
    loss = _compute_l1_loss(pred_tensor, target_tensor, asymmetric_penalty)

    # Loss should be higher than basic L1 due to over-prediction penalties
    basic_loss = _compute_l1_loss(pred_tensor, target_tensor, None)
    assert loss >= basic_loss


def test_poisson_loss_basic(target_tensor, pred_tensor) -> None:
    """Test basic Poisson loss."""
    loss = _compute_poisson_loss(pred_tensor, target_tensor, asymmetric_penalty=None)

    # Poisson loss should be positive
    assert loss > 0
    assert torch.isfinite(loss)


def test_poisson_loss_asymmetric(target_tensor, pred_tensor) -> None:
    """Test Poisson loss with asymmetric penalty."""
    asymmetric_penalty = 4.0
    loss = _compute_poisson_loss(pred_tensor, target_tensor, asymmetric_penalty)

    # Should be higher than basic Poisson
    basic_loss = _compute_poisson_loss(pred_tensor, target_tensor, None)
    assert loss >= basic_loss


def test_l1_regularization_amplitude(basic_model) -> None:
    """Test L1 regularization on amplitudes."""
    V = np.random.rand(8, 8).astype(np.float32)
    config = create_test_config(V, n_iters=10, l1_amp=0.1)

    V_tensor = torch.from_numpy(V).to(config.device)
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    loss_fn = create_loss_function(config, preprocessed_data, basic_model)
    pred = basic_model()
    loss_with_reg = loss_fn(pred)

    # Now test without regularization
    config_no_reg = create_test_config(V, n_iters=10, l1_amp=None)
    loss_fn_no_reg = create_loss_function(config_no_reg, preprocessed_data, basic_model)
    loss_without_reg = loss_fn_no_reg(pred)

    # Loss with regularization should be higher
    assert loss_with_reg > loss_without_reg


def test_l1_regularization_diagonal(basic_model) -> None:
    """Test L1 regularization on diagonal elements."""
    V = np.random.rand(8, 8).astype(np.float32)
    config = create_test_config(V, n_iters=10, l1_diag=0.05)

    V_tensor = torch.from_numpy(V).to(config.device)
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    loss_fn = create_loss_function(config, preprocessed_data, basic_model)
    pred = basic_model()
    loss_with_reg = loss_fn(pred)

    # Test without regularization
    config_no_reg = create_test_config(V, n_iters=10, l1_diag=None)
    loss_fn_no_reg = create_loss_function(config_no_reg, preprocessed_data, basic_model)
    loss_without_reg = loss_fn_no_reg(pred)

    assert loss_with_reg > loss_without_reg


def test_l1_regularization_sharpness(basic_model) -> None:
    """Test L1 regularization on sharpness."""
    V = np.random.rand(8, 8).astype(np.float32)
    config = create_test_config(V, n_iters=10, l1_sharpness=0.02)

    V_tensor = torch.from_numpy(V).to(config.device)
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    loss_fn = create_loss_function(config, preprocessed_data, basic_model)
    pred = basic_model()
    loss_with_reg = loss_fn(pred)

    # Test without regularization
    config_no_reg = create_test_config(V, n_iters=10, l1_sharpness=None)
    loss_fn_no_reg = create_loss_function(config_no_reg, preprocessed_data, basic_model)
    loss_without_reg = loss_fn_no_reg(pred)

    # Loss with regularization should be >= loss without regularization
    # (equal if sharpness parameters are zero/small)
    assert loss_with_reg >= loss_without_reg
    # Verify both losses are finite and positive
    assert torch.isfinite(loss_with_reg)
    assert torch.isfinite(loss_without_reg)


def test_combined_regularization(basic_model) -> None:
    """Test all regularizations combined."""
    V = np.random.rand(8, 8).astype(np.float32)
    config = create_test_config(
        V, n_iters=10, l1_amp=0.1, l1_diag=0.05, l1_sharpness=0.02
    )

    V_tensor = torch.from_numpy(V).to(config.device)
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    loss_fn = create_loss_function(config, preprocessed_data, basic_model)
    pred = basic_model()
    loss_combined = loss_fn(pred)

    # Compare with no regularization
    config_no_reg = create_test_config(V, n_iters=10)
    loss_fn_no_reg = create_loss_function(config_no_reg, preprocessed_data, basic_model)
    loss_no_reg = loss_fn_no_reg(pred)

    # Combined should be higher
    assert loss_combined > loss_no_reg


def test_loss_function_factory_returns_callable(basic_model) -> None:
    """Test that create_loss_function returns a callable."""
    V = np.random.rand(8, 8).astype(np.float32)
    config = create_test_config(V, n_iters=10)

    V_tensor = torch.from_numpy(V).to(config.device)
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    loss_fn = create_loss_function(config, preprocessed_data, basic_model)

    assert callable(loss_fn)

    # Test it works
    pred = basic_model()
    loss = loss_fn(pred)
    assert torch.isfinite(loss)


def test_different_loss_types(basic_model) -> None:
    """Test that different loss types produce different results."""
    V = np.random.rand(8, 8).astype(np.float32)
    V_tensor = torch.from_numpy(V).to("cpu")
    preprocessed_data = PreprocessedData(
        d=2,
        N=3,
        seed_centers=np.random.rand(3, 2).astype(np.float32) * 6,
        V_normalized=V,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    pred = basic_model()

    # Test MSE
    config_mse = create_test_config(V, n_iters=10, loss_type="mse")
    loss_fn_mse = create_loss_function(config_mse, preprocessed_data, basic_model)
    loss_mse = loss_fn_mse(pred)

    # Test L1
    config_l1 = create_test_config(V, n_iters=10, loss_type="l1")
    loss_fn_l1 = create_loss_function(config_l1, preprocessed_data, basic_model)
    loss_l1 = loss_fn_l1(pred)

    # Test Poisson
    config_poisson = create_test_config(V, n_iters=10, loss_type="poisson")
    loss_fn_poisson = create_loss_function(
        config_poisson, preprocessed_data, basic_model
    )
    loss_poisson = loss_fn_poisson(pred)

    # All should be finite
    assert torch.isfinite(loss_mse)
    assert torch.isfinite(loss_l1)
    assert torch.isfinite(loss_poisson)

    # They should generally be different
    # (though could theoretically be equal for some data)
    losses = [loss_mse.item(), loss_l1.item(), loss_poisson.item()]
    assert len(set(losses)) >= 2  # At least 2 different values
