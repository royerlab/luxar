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
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
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
    """Test normal initialization with valid config.

    Audit W1 fix: three consecutive `is not None` checks pass even
    when objects are the wrong type or in broken state. Pin the
    documented type of each component.
    """
    from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel

    components = initialize_optimization(basic_config, basic_preprocessed_data)

    assert isinstance(components.model, GaussianSplatModel)
    assert isinstance(components.optimizer, torch.optim.Optimizer)
    # scheduler can be None when scheduler_type is None — accept either
    # the documented subclass or None.
    assert components.scheduler is None or hasattr(components.scheduler, "step")

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
    """Test that optimizer is created correctly (standard Adam)."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    assert components.optimizer is not None
    assert components.scheduler is not None

    # Standard PyTorch Adam optimizer
    assert hasattr(components.optimizer, "step")
    assert hasattr(components.optimizer, "zero_grad")
    # Standard optimizer is torch.optim.Adam
    assert isinstance(components.optimizer, torch.optim.Adam)


def test_standard_optimizer_with_dynamic_ops(
    basic_config, basic_preprocessed_data
) -> None:
    """Test that standard optimizer is used even with dynamic ops enabled."""
    # Enable dynamic ops - should still use standard optimizer (no per-splat)
    basic_config.enable_dynamic_ops = True
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    assert components.optimizer is not None
    assert components.scheduler is not None

    # Should be standard PyTorch Adam (per-splat optimizer removed)
    assert isinstance(components.optimizer, torch.optim.Adam)
    assert hasattr(components.optimizer, "step")
    assert hasattr(components.optimizer, "zero_grad")


def test_amplitude_extraction(basic_config, basic_preprocessed_data) -> None:
    """Test that amplitudes are extracted from seed locations."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    model = components.model
    centers, _, amps = model.current_params()

    # Amplitudes should be positive
    assert torch.all(amps > 0)

    # Should have correct number
    assert len(amps) == basic_preprocessed_data.N


def test_sigma_initialization(basic_config, basic_preprocessed_data) -> None:
    """Test that L0 diagonal is initialized with init_sigma_vox."""
    components = initialize_optimization(basic_config, basic_preprocessed_data)

    model = components.model
    _, Ls, _ = model.current_params()

    # Check diagonal elements are initialized correctly
    # L matrices should have appropriate scale
    for i in range(basic_preprocessed_data.N):
        L = Ls[i].detach().cpu().numpy()
        # Diagonal elements should be around init_sigma_vox
        diag = np.diag(L)
        assert np.all(diag > 0)


@pytest.mark.parametrize(
    ("branch", "sigma_min_diag", "expected"),
    [
        ("explicit", [0.0, 0.0], [0.5, 0.5]),
        ("explicit", [1.0, 1.0], [1.1, 1.1]),
        ("physical", [0.0, 0.0], [1.6, 0.8]),
        ("physical", [1.0, 1.0], [1.6, 1.1]),
        ("auto", [0.0, 0.0], [1.6, 1.6]),
        ("auto", [2.0, 2.0], [2.1, 2.1]),
    ],
)
def test_fresh_seed_sigma_branches_and_floor(
    basic_config, basic_preprocessed_data, branch, sigma_min_diag, expected
) -> None:
    """Fresh-seed initialization preserves every scale branch and floor clamp."""
    basic_config.sigma_min_diag = sigma_min_diag
    if branch == "explicit":
        basic_config.init_sigma_vox = 0.5
        basic_config.voxel_size = None
    elif branch == "physical":
        basic_config.init_sigma_vox = None
        basic_config.voxel_size = np.array([1.0, 2.0], dtype=np.float32)
    else:
        basic_config.init_sigma_vox = None
        basic_config.voxel_size = None

    components = initialize_optimization(basic_config, basic_preprocessed_data)
    assert components.model is not None
    _, Ls, _ = components.model.current_params()

    np.testing.assert_allclose(
        torch.diagonal(Ls, dim1=1, dim2=2).detach().cpu().numpy(),
        np.tile(expected, (basic_preprocessed_data.N, 1)),
        atol=1e-6,
    )


def test_fresh_seed_verbose_reports_pre_floor_sigma_and_clamp(
    basic_config, basic_preprocessed_data, capsys
) -> None:
    """Verbose initialization distinguishes auto scale from the applied floor."""
    basic_config.init_sigma_vox = None
    basic_config.voxel_size = None
    basic_config.sigma_min_diag = [3.0, 3.0]
    basic_config.verbose = True

    components = initialize_optimization(basic_config, basic_preprocessed_data)
    assert components.model is not None

    output = capsys.readouterr().out
    assert "Auto-computed init sigma diag=[1.600000" in output
    assert "Clamped L0 diagonal to >= sigma_min_diag + 0.1" in output


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
        rel_l2_target=None,
        gradient_clip=None,
        loss_type="mse",
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
    import pytest

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")

    for device_str in devices:
        V = np.random.rand(16, 16).astype(np.float32)
        try:
            device = torch.device(device_str)
            # Test device is actually usable (not just reported as available)
            if device_str == "cuda":
                torch.zeros(1, device=device)  # Will fail if CUDA is busy/unavailable
        except (RuntimeError, torch.cuda.CudaError) as e:
            pytest.skip(
                f"Device {device_str} reported available but is not usable: {e}"
            )
        except Exception as e:
            # Catch torch.AcceleratorError and similar
            if "CUDA" in str(e) or "cuda" in str(e).lower():
                pytest.skip(f"Device {device_str} is not usable: {e}")
            raise

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
            rel_l2_target=None,
            gradient_clip=None,
            loss_type="mse",
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
        centers, _, _ = components.model.current_params()
        # Device check: for CPU it's "cpu", for CUDA it's "cuda:0", for MPS it's "mps:0"
        actual_device = str(centers.device)
        assert device_str in actual_device or actual_device.startswith(device_str)


def _with_normalized(config, peak: float, poison: float | None = None):
    """PreprocessedData whose normalized array peaks at ``peak``.

    Stands in for a tile normalized against a SUPPLIED whole-volume range: that
    range is a bounded-sample estimate, so the array can legitimately hold a
    voxel above it and ``_normalize_data`` deliberately leaves it unclipped.
    """
    V_normalized = (config.V / (config.V.max() + 1e-12) * peak).astype(np.float32)
    if poison is not None:
        V_normalized[0, 0] = poison
    return PreprocessedData(
        d=2,
        N=5,
        seed_centers=(np.random.rand(5, 2).astype(np.float32) * 30),
        V_normalized=V_normalized,
        V_tensor=torch.from_numpy(V_normalized).to(config.device),
        image_min=0.0,
        image_max=1.0,
        intensity_range=1.0,
        max_abs_error=0.01,
    )


def test_auto_amp_max_tracks_a_normalized_peak_above_one(basic_config) -> None:
    """A shared whole-volume scale can put a voxel above 1.0; the cap must follow.

    ``torch.clamp`` has zero gradient above its max, so a 1.0 cap would pin the
    brightest splats and stop them learning — flattening exactly the peak the
    unclipped shared scale exists to preserve.
    """
    components = initialize_optimization(
        basic_config, _with_normalized(basic_config, 1.4)
    )
    assert components.model is not None
    assert components.model.amp_max == pytest.approx(1.4, rel=1e-4)


def test_auto_amp_max_stays_one_for_a_self_normalized_array(basic_config) -> None:
    """Negative control: a range derived from the array itself peaks at 1.0."""
    components = initialize_optimization(
        basic_config, _with_normalized(basic_config, 1.0)
    )
    assert components.model is not None
    assert components.model.amp_max == pytest.approx(1.0)


def test_auto_amp_max_ignores_a_non_finite_peak(basic_config) -> None:
    """A NaN/inf voxel must not become the cap — clamping to NaN would turn
    every amplitude NaN."""
    for poison in (float("nan"), float("inf")):
        components = initialize_optimization(
            basic_config, _with_normalized(basic_config, 1.0, poison=poison)
        )
        assert components.model is not None
        assert components.model.amp_max == pytest.approx(1.0)


def test_explicit_amp_max_is_not_raised_by_the_peak(basic_config) -> None:
    """An explicit cap is the caller's call — the auto path must not override it."""
    import dataclasses

    config = dataclasses.replace(basic_config, amp_max=0.5)
    components = initialize_optimization(config, _with_normalized(config, 1.4))
    assert components.model is not None
    assert components.model.amp_max == pytest.approx(0.5)
