"""MPS tests for the runtime-compiled Metal backend (``torch.mps.compile_shader``).

Parity against the PyTorch reference renderer (forward and backward), against
the AOT extension when one is built, the int32 shape-binding contract, the
validation ported from bindings.mm, and the no-Xcode guarantee.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
from typing import Any

import pytest
import torch

from luxar.gsplats.models.gsplats.metal import backend as backend_mod
from luxar.gsplats.models.gsplats.metal import runtime_backend
from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

pytestmark = pytest.mark.skipif(
    runtime_backend.runtime_unavailable_reason() is not None,
    reason=f"runtime Metal backend unavailable: "
    f"{runtime_backend.runtime_unavailable_reason()}",
)

if runtime_backend.runtime_unavailable_reason() is None:
    from luxar.gsplats.models.gsplats.metal.gsplat_model_metal import (
        MetalSplatFunction,
        cholesky_to_conic,
    )


def _runtime_kernels() -> backend_mod.MetalKernelBackend:
    kernels, why = backend_mod._runtime_backend()
    assert kernels is not None, why
    return kernels


def _random_splats(
    shape: tuple[int, int, int], n: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(n, 3, generator=g) * torch.tensor(shape, dtype=torch.float32)
    L = torch.diag_embed(1.0 + 2.0 * torch.rand(n, 3, generator=g))
    L[:, 1, 0] = 0.3 * torch.randn(n, generator=g)
    L[:, 2, 0] = 0.3 * torch.randn(n, generator=g)
    L[:, 2, 1] = 0.3 * torch.randn(n, generator=g)
    amps = 0.1 + 0.9 * torch.rand(n, generator=g)
    return centers, L.contiguous(), amps


def _to_mps(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.to("mps").contiguous() for t in tensors]


# A NON-CUBIC shape is the int32 binding test: bound as int64, [D, H, W] would
# be read by `constant uint3&` as (D, 0, H), which renders nothing / garbage.
_SHAPES = [(13, 21, 34), (32, 32, 32)]


@pytest.mark.parametrize("shape", _SHAPES)
def test_forward_matches_the_pytorch_reference(shape: tuple[int, int, int]) -> None:
    centers, L, amps = _to_mps(*_random_splats(shape, 64))
    conic = cholesky_to_conic(L).contiguous()
    out = runtime_backend.forward_splat_3d(centers, conic, amps, list(shape), 3.0, 1e-5)
    ref = render_gaussians(shape, centers, L, amps, truncate=3.0, intensity_floor=1e-5)
    assert tuple(out.shape) == shape
    assert ref.abs().max().item() > 0.5, "reference must render something"
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=5e-5)


def test_forward_accepts_tuple_and_torch_size_shapes() -> None:
    shape = (9, 10, 11)
    centers, L, amps = _to_mps(*_random_splats(shape, 8))
    conic = cholesky_to_conic(L).contiguous()
    as_list = runtime_backend.forward_splat_3d(centers, conic, amps, list(shape), 3, 0)
    as_size = runtime_backend.forward_splat_3d(
        centers, conic, amps, torch.Size(shape), 3, 0
    )
    torch.testing.assert_close(as_list, as_size, rtol=1e-6, atol=1e-7)


def test_backward_matches_pytorch_autograd() -> None:
    shape = (17, 23, 29)
    centers, L, amps = _random_splats(shape, 24, seed=1)
    target = torch.rand(shape)

    ref_params = [t.to("mps").requires_grad_(True) for t in (centers, L, amps)]
    ref = render_gaussians(shape, *ref_params, truncate=3.0, intensity_floor=1e-5)
    ((ref - target.to("mps")) ** 2).sum().backward()

    rt_params = [t.to("mps").requires_grad_(True) for t in (centers, L, amps)]
    out = MetalSplatFunction.apply(
        *rt_params, shape, 3.0, 1e-5, False, _runtime_kernels()
    )
    ((out - target.to("mps")) ** 2).sum().backward()

    for name, ref_p, rt_p in zip(
        ("centers", "Ls", "amps"), ref_params, rt_params, strict=True
    ):
        assert ref_p.grad is not None and rt_p.grad is not None
        scale = ref_p.grad.abs().max().item()
        err = (ref_p.grad - rt_p.grad).abs().max().item()
        # The kernels' gradient differs from the PyTorch reference by ~0.1-0.3% of
        # its peak -- identically for the AOT build (measured), so this is the
        # kernel's own discretization, not the runtime path. 1% bounds it.
        assert err <= 1e-2 * scale + 1e-5, f"{name}: max err {err} vs scale {scale}"


def test_compute_conic_matches_the_pytorch_helper() -> None:
    _, L, _ = _random_splats((8, 8, 8), 100, seed=2)
    (Lm,) = _to_mps(L)
    torch.testing.assert_close(
        runtime_backend.compute_conic_metal(Lm),
        cholesky_to_conic(Lm),
        rtol=1e-5,
        atol=1e-6,
    )


def test_metal_conic_path_through_the_autograd_function() -> None:
    shape = (16, 16, 16)
    centers, L, amps = _to_mps(*_random_splats(shape, 16, seed=4))
    kernels = _runtime_kernels()
    a = MetalSplatFunction.apply(centers, L, amps, shape, 3.0, 1e-5, True, kernels)
    b = MetalSplatFunction.apply(centers, L, amps, shape, 3.0, 1e-5, False, kernels)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_empty_inputs_return_zeros() -> None:
    shape = [4, 5, 6]
    empty = torch.zeros((0, 3), device="mps")
    conic = torch.zeros((0, 6), device="mps")
    amps = torch.zeros((0,), device="mps")
    out = runtime_backend.forward_splat_3d(empty, conic, amps, shape, 3.0, 1e-5)
    assert tuple(out.shape) == (4, 5, 6) and out.abs().sum().item() == 0.0
    grads = runtime_backend.backward_splat_3d(
        torch.ones(shape, device="mps"), empty, conic, amps, shape, 3.0, 1e-5
    )
    assert [tuple(g.shape) for g in grads] == [(0, 3), (0, 6), (0,)]
    assert tuple(
        runtime_backend.compute_conic_metal(torch.zeros((0, 3, 3), device="mps")).shape
    ) == (0, 6)


def test_forward_output_is_zeroed_between_calls() -> None:
    shape = (12, 12, 12)
    centers, L, amps = _to_mps(*_random_splats(shape, 10, seed=5))
    conic = cholesky_to_conic(L).contiguous()
    first = runtime_backend.forward_splat_3d(centers, conic, amps, shape, 3.0, 1e-5)
    second = runtime_backend.forward_splat_3d(centers, conic, amps, shape, 3.0, 1e-5)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "mutate,error,match",
    [
        (lambda c, k, a: (c.double(), k, a), TypeError, "float32"),
        (lambda c, k, a: (c.cpu(), k, a), ValueError, "MPS"),
        (lambda c, k, a: (c[:, :2].contiguous(), k, a), ValueError, r"\(N, 3\)"),
        (lambda c, k, a: (c, k[:, :5].contiguous(), a), ValueError, r"\(N, 6\)"),
        (lambda c, k, a: (c, k, a[:-1]), ValueError, "batch dimensions"),
        (lambda c, k, a: (c.t().contiguous().t(), k, a), ValueError, "contiguous"),
    ],
)
def test_forward_validation_mirrors_the_aot_bindings(
    mutate: Any, error: type[Exception], match: str
) -> None:
    centers, L, amps = _to_mps(*_random_splats((8, 8, 8), 4))
    conic = cholesky_to_conic(L).contiguous()
    with pytest.raises(error, match=match):
        runtime_backend.forward_splat_3d(
            *mutate(centers, conic, amps), (8, 8, 8), 3.0, 0.0
        )


@pytest.mark.parametrize(
    "shape,match",
    [
        ((8, 8), r"\(D, H, W\)"),
        ((8, 0, 8), "positive"),
        ((2000, 2000, 1000), "capacity"),
    ],
)
def test_shape_validation(shape: tuple[int, ...], match: str) -> None:
    centers, L, amps = _to_mps(*_random_splats((8, 8, 8), 2))
    conic = cholesky_to_conic(L).contiguous()
    with pytest.raises(ValueError, match=match):
        runtime_backend.forward_splat_3d(centers, conic, amps, shape, 3.0, 0.0)


def test_backward_rejects_a_mismatched_grad_shape() -> None:
    centers, L, amps = _to_mps(*_random_splats((8, 8, 8), 2))
    conic = cholesky_to_conic(L).contiguous()
    with pytest.raises(ValueError, match="grad_output shape"):
        runtime_backend.backward_splat_3d(
            torch.zeros((8, 8, 9), device="mps"),
            centers,
            conic,
            amps,
            (8, 8, 8),
            3.0,
            0.0,
        )


def test_concurrent_first_use_compiles_once(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_backend.reset_for_tests()
    compiler = runtime_backend._shader_compiler()
    assert compiler is not None
    calls: list[int] = []

    def counting(source: str) -> Any:
        calls.append(1)
        return compiler(source)

    monkeypatch.setattr(runtime_backend, "_shader_compiler", lambda: counting)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            runtime_backend.prepare()
        except BaseException as exc:  # noqa: BLE001 -- surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and len(calls) == 1


def test_runtime_path_spawns_no_process_with_developer_dir_broken() -> None:
    """Compile + run with every subprocess launch forbidden and DEVELOPER_DIR broken.

    This proves the runtime path never shells out (so never reaches ``xcrun``)
    and does not consult the active developer directory. It does not uninstall
    Xcode from the machine running the test.
    """
    script = textwrap.dedent(
        """
        import subprocess, os
        def forbidden(*args, **kwargs):
            raise AssertionError(f"runtime Metal backend spawned a process: {args!r}")
        subprocess.Popen = forbidden
        subprocess.run = forbidden
        subprocess.check_call = forbidden
        subprocess.check_output = forbidden
        os.system = forbidden
        import torch
        from luxar.gsplats.models.gsplats.metal import get_metal_backend
        kernels = get_metal_backend()
        assert kernels is not None and kernels.name == "runtime", kernels
        c = torch.full((1, 3), 4.0, device="mps")
        k = torch.tensor([[1.0, 0, 0, 1, 0, 1]], device="mps")
        a = torch.ones(1, device="mps")
        out = kernels.forward_splat_3d(c, k, a, [8, 9, 10], 3.0, 1e-5)
        assert out.sum().item() > 1.0, out.sum().item()
        print("OK")
        """
    )
    env = dict(os.environ)
    env.update(
        DEVELOPER_DIR="/nonexistent",
        PATH="/usr/bin:/bin",
        LUXAR_METAL_BACKEND="runtime",
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter + inline script
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout


# =============================================================================
# Runtime vs AOT parity (only when the optional extension is built)
# =============================================================================


def _aot_kernels() -> backend_mod.MetalKernelBackend | None:
    kernels, _ = backend_mod._aot_backend()
    return kernels


@pytest.mark.skipif(
    sys.platform != "darwin" or backend_mod.find_extension() is None,
    reason="AOT Metal extension not built",
)
@pytest.mark.parametrize("shape", _SHAPES)
def test_runtime_matches_aot(shape: tuple[int, int, int]) -> None:
    aot = _aot_kernels()
    if aot is None:
        pytest.skip(f"AOT extension unusable: {backend_mod._aot_backend()[1]}")
    rt = _runtime_kernels()
    centers, L, amps = _to_mps(*_random_splats(shape, 64, seed=7))
    conic = cholesky_to_conic(L).contiguous()
    fwd_aot = aot.forward_splat_3d(centers, conic, amps, list(shape), 3.0, 1e-5)
    fwd_rt = rt.forward_splat_3d(centers, conic, amps, list(shape), 3.0, 1e-5)
    torch.testing.assert_close(fwd_rt, fwd_aot, rtol=1e-5, atol=1e-6)

    grad = torch.randn(shape).to("mps")
    bwd_aot = aot.backward_splat_3d(grad, centers, conic, amps, list(shape), 3.0, 1e-5)
    bwd_rt = rt.backward_splat_3d(grad, centers, conic, amps, list(shape), 3.0, 1e-5)
    for a, b in zip(bwd_rt, bwd_aot, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    # Same source, but the AOT metallib is built with `-O2 -std=metal3.0` and
    # the runtime compiler with its defaults: last-ulp differences only.
    torch.testing.assert_close(
        rt.compute_conic_metal(L), aot.compute_conic_metal(L), rtol=1e-6, atol=1e-6
    )
