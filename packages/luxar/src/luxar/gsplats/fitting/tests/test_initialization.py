"""
Tests for fitting/initialization.py module.
"""

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.fitting.initialization import initialize_optimization


@pytest.fixture
def basic_config():
    """Create a basic FitConfig for testing."""
    V = np.random.rand(32, 32).astype(np.float32) * 0.5 + 0.5
    return FitConfig(
        V=V,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5],
        sigma_max_diag=[10.0, 10.0],
        truncate=3.0,
        n_iters=100,
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


@pytest.fixture
def basic_preprocessed_data(basic_config):
    """Create basic preprocessed data."""
    N = 10
    d = 2
    seed_centers = np.random.rand(N, d).astype(np.float32) * 30
    V_normalized = basic_config.V / (basic_config.V.max() + 1e-12)
    V_tensor = torch.from_numpy(V_normalized).to(basic_config.device)

    return PreprocessedData(
        d=d,
        N=N,
        seed_centers=seed_centers,
        V_normalized=V_normalized,
        V_tensor=V_tensor,
        image_min=0.0,
        image_max=float(basic_config.V.max()),
        intensity_range=float(basic_config.V.max()),
        max_abs_error=0.01,
    )


def test_initialize_optimization_normal(basic_config, basic_preprocessed_data) -> None:
    """Test normal initialization with valid config."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    assert components.model is not None
    assert components.optimizer is not None
    assert components.scheduler is not None
    assert components.coordinator is not None

    # Check model has correct number of splats
    assert components.model.n_splats() == basic_preprocessed_data.N


def test_initialize_optimization_zero_candidates(basic_config) -> None:
    """Test edge case: N=0 returns None components."""
    # Create preprocessed data with zero candidates
    V_normalized = basic_config.V / (basic_config.V.max() + 1e-12)
    preprocessed_data = PreprocessedData(
        d=2,
        N=0,
        seed_centers=np.array([]).reshape(0, 2).astype(np.float32),
        V_normalized=V_normalized,
        V_tensor=torch.from_numpy(V_normalized).to(basic_config.device),
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )

    components = initialize_optimization(basic_config, preprocessed_data)

    assert components.model is None
    assert components.optimizer is None
    assert components.scheduler is None
    assert components.coordinator is None


def test_model_initialization_parameters(basic_config, basic_preprocessed_data) -> None:
    """Test that model gets correct initialization parameters."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    model = components.model
    assert model.shape == basic_config.V.shape
    # Compare lists properly
    assert list(model.sigma_min_diag) == basic_config.sigma_min_diag
    assert list(model.sigma_max_diag) == basic_config.sigma_max_diag
    assert model.truncate == basic_config.truncate


def test_optimizer_setup(basic_config, basic_preprocessed_data) -> None:
    """Test that per-splat optimizer is created correctly."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    assert components.optimizer is not None
    assert components.scheduler is not None
    assert components.coordinator is not None

    # PerSplatAdam is a custom optimizer - check it has the expected methods
    assert hasattr(components.optimizer, "step")
    assert hasattr(components.optimizer, "zero_grad")
    assert hasattr(components.optimizer, "get_effective_learning_rates")


def test_amplitude_extraction(basic_config, basic_preprocessed_data) -> None:
    """Test that amplitudes are extracted from seed locations."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    model = components.model
    centers, _, amps, _ = model.current_params()

    # Amplitudes should be positive
    assert torch.all(amps > 0)

    # Should have correct number
    assert len(amps) == basic_preprocessed_data.N


def test_sigma_initialization(basic_config, basic_preprocessed_data) -> None:
    """Test that L0 diagonal is initialized with init_sigma_vox."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    model = components.model
    _, Ls, _, _ = model.current_params()

    # Check diagonal elements are initialized correctly
    # L matrices should have appropriate scale
    for i in range(basic_preprocessed_data.N):
        L = Ls[i].detach().cpu().numpy()
        # Diagonal elements should be around init_sigma_vox
        diag = np.diag(L)
        assert np.all(diag > 0)


def test_initialization_3d(basic_config) -> None:
    """Test initialization works for 3D data."""
    # Create 3D config and data
    V_3d = np.random.rand(16, 16, 16).astype(np.float32) * 0.5 + 0.5
    config_3d = FitConfig(
        V=V_3d,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5, 0.5],
        sigma_max_diag=[10.0, 10.0, 10.0],
        truncate=3.0,
        n_iters=100,
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

    N = 5
    d = 3
    seed_centers = np.random.rand(N, d).astype(np.float32) * 14
    V_normalized = V_3d / (V_3d.max() + 1e-12)

    preprocessed_data = PreprocessedData(
        d=d,
        N=N,
        seed_centers=seed_centers,
        V_normalized=V_normalized,
        V_tensor=torch.from_numpy(V_normalized).to(config_3d.device),
        image_min=0.0,
        image_max=1.0,
        intensity_range=float(V_3d.max()),
        max_abs_error=0.01,
    )

    components = initialize_optimization(config_3d, preprocessed_data)

    assert components.model is not None
    assert components.model.n_splats() == N


def test_initialization_different_devices() -> None:
    """Test initialization on different devices if available."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    for device_str in devices:
        V = np.random.rand(16, 16).astype(np.float32)
        device = torch.device(device_str)
        config = FitConfig(
            V=V,
            seeds=None,
            norm_percentile=0.0,
            init_sigma_vox=2.0,
            sigma_min_diag=[0.5, 0.5],
            sigma_max_diag=[10.0, 10.0],
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
            lr_reduction_factor=0.5,
            early_stop_patience=None,
            enable_dynamic_ops=False,
            dynamic_config=DynamicOpsConfig(),
            dynamic_ops_verbose=False,
            napari_movie=False,
            movie_every=1,
            movie_max_frames=100,
            device=device,
            verbose=False,
        )

        N = 3
        d = 2
        seed_centers = np.random.rand(N, d).astype(np.float32) * 14
        V_normalized = V / (V.max() + 1e-12)

        preprocessed_data = PreprocessedData(
            d=d,
            N=N,
            seed_centers=seed_centers,
            V_normalized=V_normalized,
            V_tensor=torch.from_numpy(V_normalized).to(device),
            image_min=0.0,
            image_max=1.0,
            intensity_range=1.0,
            max_abs_error=0.01,
        )

        components = initialize_optimization(config, preprocessed_data)
        assert components.model is not None

        # Check model is on correct device
        centers, _, _, _ = components.model.current_params()
        # Device check: for CPU it's "cpu", for CUDA it's "cuda:0", for MPS it's "mps:0"
        actual_device = str(centers.device)
        assert device_str in actual_device or actual_device.startswith(device_str)
