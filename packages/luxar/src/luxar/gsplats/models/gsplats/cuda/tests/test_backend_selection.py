"""CPU-runnable tests for kernel-backend selection and the torch.compile fallback.

No GPU needed: CUDA availability, the AOT module and NVRTC availability are
all monkeypatched, so these run on every CI runner.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
import torch

from luxar.gsplats.models.gsplats.cuda import backend as backend_mod
from luxar.gsplats.models.gsplats.cuda import gsplat_model_cuda
from luxar.gsplats.models.gsplats.cuda.backend import (
    BACKEND_ENV,
    SplattingBackend,
    get_splatting_backend,
    requested_backend,
    splatting_backend_unavailable_reason,
)


@pytest.fixture(autouse=True)
def _fresh_resolution(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    backend_mod.reset_for_tests()
    yield
    backend_mod.reset_for_tests()


def _fake_nvrtc(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    fake = SplattingBackend(
        "nvrtc", lambda *a, **k: None, lambda *a, **k: None, lambda d, dev=None: None
    )
    monkeypatch.setattr(
        backend_mod,
        "_nvrtc_backend",
        lambda: (fake, None) if available else (None, "libnvrtc not found"),
    )


def _env(
    monkeypatch: pytest.MonkeyPatch, *, cuda: bool, aot: bool, nvrtc: bool
) -> None:
    monkeypatch.setattr(backend_mod, "_cuda_available", lambda: cuda)
    aot_module = SimpleNamespace(
        forward=lambda *a: "aot-fwd", backward=lambda *a: "aot-bwd"
    )
    monkeypatch.setattr(backend_mod, "AOT_MODULE", aot_module if aot else None)
    _fake_nvrtc(monkeypatch, nvrtc)


# =============================================================================
# LUXAR_CUDA_BACKEND parsing
# =============================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [("", "auto"), ("  NVRTC ", "nvrtc"), ("aot", "aot"), ("torch", "torch")],
)
def test_requested_backend_parsing(raw: str, expected: str) -> None:
    assert requested_backend({BACKEND_ENV: raw}) == expected


def test_requested_backend_unset_is_auto() -> None:
    assert requested_backend({}) == "auto"


def test_requested_backend_rejects_typos() -> None:
    with pytest.raises(ValueError, match="nvrtcc"):
        requested_backend({BACKEND_ENV: "nvrtcc"})


# =============================================================================
# Selection order
# =============================================================================


def test_auto_prefers_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=True, nvrtc=True)
    chosen = get_splatting_backend()
    assert chosen is not None and chosen.name == "aot"
    assert chosen.forward() == "aot-fwd"


def test_auto_falls_to_nvrtc_without_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=False, nvrtc=True)
    chosen = get_splatting_backend()
    assert chosen is not None and chosen.name == "nvrtc"


def test_auto_falls_to_torch_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=False, nvrtc=False)
    assert get_splatting_backend() is None
    reason = splatting_backend_unavailable_reason()
    assert reason is not None and "libnvrtc not found" in reason


def test_auto_without_cuda_is_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=False, aot=True, nvrtc=True)
    assert get_splatting_backend() is None
    assert splatting_backend_unavailable_reason() == "no CUDA device"


def test_force_nvrtc_skips_aot(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=True, nvrtc=True)
    monkeypatch.setenv(BACKEND_ENV, "nvrtc")
    chosen = get_splatting_backend()
    assert chosen is not None and chosen.name == "nvrtc"


def test_force_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=True, nvrtc=True)
    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert get_splatting_backend() is None
    assert splatting_backend_unavailable_reason() == f"{BACKEND_ENV}=torch"


@pytest.mark.parametrize(
    "forced,env,match",
    [
        ("aot", dict(cuda=True, aot=False, nvrtc=True), "make build-cuda"),
        ("nvrtc", dict(cuda=True, aot=True, nvrtc=False), "libnvrtc not found"),
        ("nvrtc", dict(cuda=False, aot=False, nvrtc=True), "CUDA device"),
    ],
)
def test_forced_backend_that_cannot_run_raises(
    monkeypatch: pytest.MonkeyPatch, forced: str, env: dict[str, bool], match: str
) -> None:
    _env(monkeypatch, **env)
    monkeypatch.setenv(BACKEND_ENV, forced)
    with pytest.raises(RuntimeError, match=match):
        get_splatting_backend()


def test_resolution_is_cached_per_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, cuda=True, aot=False, nvrtc=True)
    calls = 0
    original = backend_mod._resolve

    def counting(choice: str) -> Any:
        nonlocal calls
        calls += 1
        return original(choice)

    monkeypatch.setattr(backend_mod, "_resolve", counting)
    first = get_splatting_backend()
    assert get_splatting_backend() is first
    assert calls == 1
    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert get_splatting_backend() is None
    assert calls == 2


# =============================================================================
# torch.compile first-call failure -> eager, remembered
# =============================================================================


def _double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def test_compile_failure_at_first_call_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inductor without a C compiler raises at the FIRST CALL, not at wrap time."""
    calls = 0

    def fake_compile(fn: Any) -> Any:
        def compiled(x: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            # What Inductor raises with no compiler: its own error wrapping the
            # failed compiler lookup.
            raise RuntimeError("InductorError") from FileNotFoundError(
                "[Errno 2] No such file or directory: '/nonexistent/cc'"
            )

        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    wrapped = gsplat_model_cuda._CompiledWithEagerFallback(_double)
    assert wrapped.uses_compiled

    with pytest.warns(RuntimeWarning, match="running it eagerly"):
        out = wrapped(torch.ones(3))
    torch.testing.assert_close(out, torch.full((3,), 2.0))
    assert not wrapped.uses_compiled

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the decision is remembered: no 2nd warning
        torch.testing.assert_close(wrapped(torch.ones(2)), torch.full((2,), 2.0))
    assert calls == 1  # the compiled path is never retried


def test_compile_failure_at_wrap_time_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_compile(fn: Any) -> Any:
        raise RuntimeError("no triton")

    monkeypatch.setattr(torch, "compile", broken_compile)
    with pytest.warns(RuntimeWarning, match="no triton"):
        wrapped = gsplat_model_cuda._CompiledWithEagerFallback(_double)
    assert not wrapped.uses_compiled
    torch.testing.assert_close(wrapped(torch.ones(1)), torch.full((1,), 2.0))


def _compiled_raising(exc: BaseException) -> Any:
    def fake_compile(fn: Any) -> Any:
        def compiled(x: torch.Tensor) -> torch.Tensor:
            raise exc

        return compiled

    return fake_compile


@pytest.mark.parametrize(
    "exc",
    [
        torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("mat1 and mat2 shapes cannot be multiplied"),
        ValueError("caller bug"),
    ],
    ids=["oom", "shape-error", "value-error"],
)
def test_non_compile_errors_propagate_and_keep_compiled_path(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    """OOM (benchmark sweeps into it on purpose) and real bugs are NOT compile failures."""
    monkeypatch.setattr(torch, "compile", _compiled_raising(exc))
    wrapped = gsplat_model_cuda._CompiledWithEagerFallback(_double)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "running it eagerly" warning either
        with pytest.raises(type(exc)):
            wrapped(torch.ones(1))
    assert wrapped.uses_compiled


def test_oom_wrapped_in_a_compile_error_still_propagates() -> None:
    try:
        try:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        except torch.cuda.OutOfMemoryError as oom:
            raise RuntimeError("InductorError") from oom
    except RuntimeError as exc:
        assert not gsplat_model_cuda.is_compile_stack_failure(exc)


def test_compile_stack_failures_are_recognized() -> None:
    import subprocess

    assert gsplat_model_cuda.is_compile_stack_failure(FileNotFoundError("cc"))
    assert gsplat_model_cuda.is_compile_stack_failure(
        subprocess.CalledProcessError(1, ["gcc", "driver.c"])
    )
    fake_inductor = type("InductorError", (RuntimeError,), {})
    fake_inductor.__module__ = "torch._inductor.exc"
    assert gsplat_model_cuda.is_compile_stack_failure(fake_inductor("no cc"))
    assert not gsplat_model_cuda.is_compile_stack_failure(RuntimeError("plain"))


class _BackwardCompileFails(torch.autograd.Function):
    """Identity-ish forward whose backward fails like a lazy AOTAutograd compile."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return gsplat_model_cuda.cholesky_to_conic(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> Any:
        raise RuntimeError("BackendCompilerFailed") from FileNotFoundError("cc")


def test_lazy_backward_compile_failure_falls_back_to_eager_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backward graph compiles inside autograd.grad; its failure must not abort."""
    monkeypatch.setattr(torch, "compile", lambda fn: _BackwardCompileFails.apply)
    helper = gsplat_model_cuda._CompiledWithEagerFallback(
        gsplat_model_cuda.cholesky_to_conic
    )
    monkeypatch.setattr(gsplat_model_cuda, "_cholesky_to_conic_compiled", helper)

    L = torch.eye(3).repeat(4, 1, 1) * 1.5
    L[:, 1, 0] = 0.2
    L = L.requires_grad_(True)
    d_conic = torch.rand(4, 6)
    with pytest.warns(RuntimeWarning, match="running it eagerly"):
        got = gsplat_model_cuda.conic_vjp(L, d_conic)
    assert not helper.uses_compiled

    L2 = L.detach().clone().requires_grad_(True)
    (expected,) = torch.autograd.grad(
        gsplat_model_cuda.cholesky_to_conic(L2), L2, grad_outputs=d_conic
    )
    torch.testing.assert_close(got, expected)


def test_lazy_backward_non_compile_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BackwardOOM(_BackwardCompileFails):
        @staticmethod
        def backward(ctx: Any, grad: torch.Tensor) -> Any:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch, "compile", lambda fn: _BackwardOOM.apply)
    helper = gsplat_model_cuda._CompiledWithEagerFallback(
        gsplat_model_cuda.cholesky_to_conic
    )
    monkeypatch.setattr(gsplat_model_cuda, "_cholesky_to_conic_compiled", helper)
    L = (torch.eye(3).repeat(2, 1, 1) * 2.0).requires_grad_(True)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        gsplat_model_cuda.conic_vjp(L, torch.ones(2, 6))
    assert helper.uses_compiled


def test_module_helper_uses_the_fallback_wrapper() -> None:
    assert isinstance(
        gsplat_model_cuda._cholesky_to_conic_compiled,
        gsplat_model_cuda._CompiledWithEagerFallback,
    )


# =============================================================================
# Fitting-side fallback (initialization._try_cuda_model)
# =============================================================================


def test_try_cuda_model_warns_and_returns_none_without_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting import initialization

    _env(monkeypatch, cuda=True, aot=False, nvrtc=False)
    with pytest.warns(UserWarning, match="PyTorch fallback"):
        assert (
            initialization._try_cuda_model(SimpleNamespace(verbose=False), {}) is None
        )  # type: ignore[arg-type]


def test_try_cuda_model_quiet_when_torch_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.fitting import initialization

    _env(monkeypatch, cuda=True, aot=True, nvrtc=True)
    monkeypatch.setenv(BACKEND_ENV, "torch")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            initialization._try_cuda_model(SimpleNamespace(verbose=False), {}) is None
        )  # type: ignore[arg-type]


def test_try_cuda_model_falls_back_when_nvrtc_compile_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats._nvrtc import NvrtcCompileError
    from luxar.gsplats.fitting import initialization
    from luxar.gsplats.models.gsplats import cuda as cuda_pkg

    _env(monkeypatch, cuda=True, aot=False, nvrtc=True)

    def failing_model(**kwargs: Any) -> Any:
        raise NvrtcCompileError("NVRTC failed to compile CUDA program", "error: boom")

    monkeypatch.setattr(cuda_pkg, "GaussianSplatModelCUDA", failing_model)
    with pytest.warns(UserWarning, match="nvrtc kernels failed to load"):
        assert (
            initialization._try_cuda_model(SimpleNamespace(verbose=False), {}) is None
        )  # type: ignore[arg-type]


def test_try_cuda_model_reraises_when_backend_is_forced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LUXAR_CUDA_BACKEND=nvrtc means 'fail if NVRTC fails', in fitting too."""
    from luxar.gsplats._nvrtc import NvrtcCompileError
    from luxar.gsplats.fitting import initialization
    from luxar.gsplats.models.gsplats import cuda as cuda_pkg

    _env(monkeypatch, cuda=True, aot=False, nvrtc=True)
    monkeypatch.setenv(BACKEND_ENV, "nvrtc")

    def failing_model(**kwargs: Any) -> Any:
        raise NvrtcCompileError("NVRTC failed to compile CUDA program", "error: boom")

    monkeypatch.setattr(cuda_pkg, "GaussianSplatModelCUDA", failing_model)
    with pytest.raises(NvrtcCompileError, match="boom"):
        initialization._try_cuda_model(SimpleNamespace(verbose=False), {})  # type: ignore[arg-type]


def _nlm_failing_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    from luxar.gsplats.preprocessing.cuda import nlm_cuda_wrapper

    def failing(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise RuntimeError("CUDA driver call cuLaunchKernel failed: boom")

    monkeypatch.setattr(nlm_cuda_wrapper, "nlm_cuda_denoise", failing)


def test_nlm_kernel_failure_falls_back_in_auto_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxar.gsplats.preprocessing import denoise_nlm

    _nlm_failing_kernel(monkeypatch)
    vol = torch.rand(12, 12)
    with pytest.warns(UserWarning, match="falling back to PyTorch"):
        out = denoise_nlm(vol, h=0.1, backend="cuda")
    assert out.shape == vol.shape


@pytest.mark.parametrize("forced", ["nvrtc", "aot"])
def test_nlm_kernel_failure_raises_when_backend_is_forced(
    monkeypatch: pytest.MonkeyPatch, forced: str
) -> None:
    from luxar.gsplats.preprocessing import denoise_nlm

    _nlm_failing_kernel(monkeypatch)
    monkeypatch.setenv(BACKEND_ENV, forced)
    with pytest.raises(RuntimeError, match="cuLaunchKernel failed"):
        denoise_nlm(torch.rand(12, 12), h=0.1, backend="cuda")


# =============================================================================
# NLM backend selection
# =============================================================================


def test_nlm_backend_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    from luxar.gsplats import _nvrtc
    from luxar.gsplats.preprocessing import cuda as nlm_cuda

    monkeypatch.setattr(nlm_cuda, "_resolved", {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nlm_cuda, "_aot_module", None)
    monkeypatch.setattr(_nvrtc, "nvrtc_unavailable_reason", lambda: None)
    chosen = nlm_cuda.get_nlm_backend()
    assert chosen is not None and chosen.__name__.endswith("nlm_nvrtc")

    monkeypatch.setattr(nlm_cuda, "_resolved", {})
    monkeypatch.setenv(BACKEND_ENV, "torch")
    assert not nlm_cuda.nlm_cuda_available()

    monkeypatch.setattr(nlm_cuda, "_resolved", {})
    monkeypatch.setenv(BACKEND_ENV, "aot")
    with pytest.raises(RuntimeError, match="make build-nlm-cuda"):
        nlm_cuda.get_nlm_backend()
