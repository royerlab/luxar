"""Selection of the native CPU backend: env vars, fallback, fit wiring, CLI."""

from __future__ import annotations

import sys
from typing import Iterator

import numpy as np
import pytest
import torch

from luxar.gsplats.fitting.config import FitConfig, PreprocessedData
from luxar.gsplats.fitting.dynamic_ops import DynamicOpsConfig
from luxar.gsplats.models.gsplats.cpu import backend as backend_mod
from luxar.gsplats.models.gsplats.cpu.backend import (
    BACKEND_ENV,
    THREADS_ENV,
    cpu_native_unavailable_reason,
    get_cpu_kernels,
    physical_core_count,
    requested_backend,
    resolve_thread_count,
)
from luxar.gsplats.models.gsplats.gsplat_model import GaussianSplatModel
from luxar.gsplats.utils import device as device_mod

_KERNELS_MODULE = "luxar.gsplats.models.gsplats.cpu.kernels"


@pytest.fixture(autouse=True)
def _fresh_selection(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.setenv(THREADS_ENV, "2")
    # The PyTorch-path CPU warning is once per process; re-arm it per test.
    monkeypatch.setattr(device_mod, "_CPU_WARNING_ISSUED", False)
    backend_mod.reset_for_tests()
    yield
    backend_mod.reset_for_tests()


@pytest.fixture
def numba_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import numba`` (and the kernel module) fail as on a bare install."""
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setitem(sys.modules, _KERNELS_MODULE, None)
    backend_mod.reset_for_tests()


def _config_3d(verbose: bool) -> tuple[FitConfig, PreprocessedData]:
    rng = np.random.default_rng(0)
    V = (rng.random((12, 14, 16)) * 0.5 + 0.5).astype(np.float32)
    config = FitConfig(
        V=V,
        seeds=None,
        norm_percentile=0.0,
        init_sigma_vox=2.0,
        sigma_min_diag=[0.5, 0.5, 0.5],
        sigma_max_diag=[10.0, 10.0, 10.0],
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
        device=torch.device("cpu"),
        verbose=verbose,
    )
    V_normalized = V / V.max()
    data = PreprocessedData(
        d=3,
        N=5,
        seed_centers=(rng.random((5, 3)) * 11).astype(np.float32),
        V_normalized=V_normalized,
        V_tensor=torch.from_numpy(V_normalized),
        image_min=0.0,
        image_max=1.0,
        intensity_range=float(V.max()),
        max_abs_error=0.01,
    )
    return config, data


# ----------------------------------------------------------------- env parsing


def test_requested_backend_defaults_to_auto_and_rejects_unknown_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert requested_backend() == "auto"
    monkeypatch.setenv(BACKEND_ENV, " Native ")
    assert requested_backend() == "native"
    monkeypatch.setenv(BACKEND_ENV, "openmp")
    with pytest.raises(ValueError, match=BACKEND_ENV):
        requested_backend()


def test_forced_torch_selects_no_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert get_cpu_kernels() is None
    assert cpu_native_unavailable_reason() == f"{BACKEND_ENV}=torch"


def test_thread_count_honours_the_env_var_and_ignores_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_thread_count() == 2
    for bad in ("zero", "0", "-3"):
        monkeypatch.setenv(THREADS_ENV, bad)
        assert resolve_thread_count() == physical_core_count()
    monkeypatch.delenv(THREADS_ENV)
    assert 1 <= resolve_thread_count() == physical_core_count()


# ------------------------------------------------------------- numba missing


@pytest.mark.usefixtures("numba_missing")
def test_auto_without_numba_falls_back_and_says_why() -> None:
    assert get_cpu_kernels() is None
    reason = cpu_native_unavailable_reason()
    assert reason is not None and "numba is not installed" in reason


@pytest.mark.usefixtures("numba_missing")
def test_forced_native_without_numba_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV, "native")
    with pytest.raises(RuntimeError, match="numba"):
        get_cpu_kernels()


@pytest.mark.usefixtures("numba_missing")
def test_fit_without_numba_uses_the_pytorch_model_and_hints_at_numba(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from luxar.gsplats.fitting.initialization import initialize_optimization

    config, data = _config_3d(verbose=True)
    with pytest.warns(UserWarning, match="running on CPU") as record:
        components = initialize_optimization(config, data)
    assert type(components.model) is GaussianSplatModel
    message = " ".join(str(w.message) for w in record)
    assert "native CPU kernels are unavailable" in message
    assert "pip install numba" in message
    assert device_mod.cpu_fallback_hint() in message  # WP2's honest reason
    out = capsys.readouterr().out
    assert "Model class: GaussianSplatModel (PyTorch, device=cpu)" in out


# ------------------------------------------------------------ fit wiring


def test_fit_on_cpu_selects_the_native_model_and_logs_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("numba")
    from luxar.gsplats.fitting.initialization import initialize_optimization
    from luxar.gsplats.models.gsplats.cpu import GaussianSplatModelCPUNative

    config, data = _config_3d(verbose=True)
    components = initialize_optimization(config, data)
    assert type(components.model) is GaussianSplatModelCPUNative
    out = capsys.readouterr().out
    assert "Model class: GaussianSplatModelCPUNative (numba, 2 threads" in out
    assert "CPU kernels " in out  # the first call in this process reports its JIT


def test_quiet_fit_summary_names_the_numba_kernels(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("numba")
    from luxar.gsplats.fitting import initialization

    monkeypatch.setattr(initialization, "_ANNOUNCED_BACKENDS", set())
    config, data = _config_3d(verbose=False)
    initialization.initialize_optimization(config, data)
    assert (
        "Fitting backend: GaussianSplatModelCPUNative (numba kernels, device=cpu)"
        in capsys.readouterr().out
    )


def test_forced_torch_fit_uses_the_pytorch_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting.initialization import initialize_optimization

    monkeypatch.setenv(BACKEND_ENV, "torch")
    config, data = _config_3d(verbose=False)
    with pytest.warns(UserWarning, match="running on CPU"):
        components = initialize_optimization(config, data)
    assert type(components.model) is GaussianSplatModel
