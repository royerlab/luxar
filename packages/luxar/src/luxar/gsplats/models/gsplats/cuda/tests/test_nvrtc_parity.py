"""GPU parity of the NVRTC runtime-compiled splatting kernels.

Forward and backward of :mod:`..nvrtc_backend` against

- the pure-PyTorch reference renderer (``rendering_core.render_gaussians`` +
  autograd) -- the authoritative definition. Measured on an RTX 3070 and an
  RTX PRO 6000: FP32 forward rel-L2 ~5e-7, FP16 forward <=2.3e-3 (FP16 input
  quantization), gradients rel-L2 <=1.3e-2 with correlation >0.9999. The
  thresholds below sit a few x above those, not at the loose aggregate
  bounds the older comparison tests use; and
- the ahead-of-time extension when it is importable -- the SAME device code.
  Measured: forward rel-L2 ~5e-8 (float atomicAdd order only) and backward
  bit-identical.

Skips only when there is no CUDA device or no ``libnvrtc`` (environment).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from luxar.gsplats.models.gsplats.cuda.backend import AOT_MODULE

from .conftest import Tolerances, seed_all

pytestmark = [pytest.mark.gpu]


def _nvrtc_or_skip() -> Any:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from luxar.gsplats._nvrtc import nvrtc_unavailable_reason

    reason = nvrtc_unavailable_reason()
    if reason is not None:
        pytest.skip(f"NVRTC unavailable: {reason}")
    from luxar.gsplats.models.gsplats.cuda import nvrtc_backend

    return nvrtc_backend


SHAPES = {2: (48, 40), 3: (24, 20, 22), 4: (12, 10, 11, 9)}


def _problem(dim: int, n: int = 40, seed: int = 0) -> dict[str, Any]:
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import cholesky_to_conic

    seed_all(seed)
    shape = SHAPES[dim]
    dev = torch.device("cuda")
    lo = torch.tensor([2.0] * dim, device=dev)
    hi = torch.tensor([s - 3.0 for s in shape], device=dev)
    centers = lo + torch.rand(n, dim, device=dev) * (hi - lo)
    L = torch.zeros(n, dim, dim, device=dev)
    for i in range(dim):
        L[:, i, i] = 0.8 + 1.2 * torch.rand(n, device=dev)
        for j in range(i):
            L[:, i, j] = 0.3 * (torch.rand(n, device=dev) - 0.5)
    amps = 0.3 + 0.7 * torch.rand(n, device=dev)
    return {
        "centers": centers,
        "L": L,
        "conic": cholesky_to_conic(L).contiguous(),
        "amps": amps,
        "shape": shape,
    }


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        np.corrcoef(a.double().flatten().cpu(), b.double().flatten().cpu())[0, 1]
    )


# =============================================================================
# vs the pure-PyTorch reference
# =============================================================================


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_forward_matches_torch_reference(dim: int, dtype: torch.dtype) -> None:
    nvrtc = _nvrtc_or_skip()
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

    p = _problem(dim)
    use_fp16 = dtype == torch.float16
    out, shape_tensor = nvrtc.forward(
        p["centers"], p["conic"], p["amps"], list(p["shape"]), 3.0, 1e-5, use_fp16
    )
    ref = render_gaussians(p["shape"], p["centers"], p["L"], p["amps"], 3.0, 1e-5)

    assert out.dtype == torch.float32 and out.shape == (int(np.prod(p["shape"])),)
    assert (
        shape_tensor.tolist() == list(p["shape"]) and shape_tensor.dtype == torch.int32
    )
    out = out.view(p["shape"])
    limit = 1e-2 if use_fp16 else 1e-5
    assert _rel_l2(out, ref) < limit, f"rel L2 {_rel_l2(out, ref):.2e}"
    assert _corr(out, ref) > Tolerances.COMPARISON_MIN_CORRELATION


@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_backward_matches_torch_reference(
    dim: int, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gradients w.r.t. (centers, L, amps) through the autograd wrapper."""
    _nvrtc_or_skip()
    from luxar.gsplats.models.gsplats.cuda import backend as backend_mod
    from luxar.gsplats.models.gsplats.cuda.gsplat_model_cuda import CUDASplatFunction
    from luxar.gsplats.models.gsplats.rendering_core import render_gaussians

    monkeypatch.setenv(backend_mod.BACKEND_ENV, "nvrtc")
    p = _problem(dim)
    weight = torch.rand(p["shape"], device="cuda")

    def grads(render: Any) -> list[torch.Tensor]:
        c = p["centers"].clone().requires_grad_(True)
        L = p["L"].clone().requires_grad_(True)
        a = p["amps"].clone().requires_grad_(True)
        out = render(c, L, a).reshape(p["shape"])
        (out * weight).sum().backward()
        return [c.grad, L.grad, a.grad]

    if dtype == torch.float16:

        def kernel(c: torch.Tensor, L: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            with torch.autocast(
                "cuda", dtype=torch.float16
            ):  # the AMP route to FP16 kernels
                return CUDASplatFunction.apply(c, L, a, p["shape"], 3.0, 1e-5, False)
    else:

        def kernel(c: torch.Tensor, L: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            return CUDASplatFunction.apply(c, L, a, p["shape"], 3.0, 1e-5, False)

    got = grads(kernel)
    ref = grads(lambda c, L, a: render_gaussians(p["shape"], c, L, a, 3.0, 1e-5))
    assert backend_mod.get_splatting_backend().name == "nvrtc"  # type: ignore[union-attr]

    for name, g, r in zip(("centers", "L", "amps"), got, ref, strict=True):
        assert torch.isfinite(g).all(), name
        assert _rel_l2(g, r) < 0.04, f"{name}: rel L2 {_rel_l2(g, r):.2e}"
        assert _corr(g, r) > 0.999, f"{name}: correlation {_corr(g, r):.5f}"


# =============================================================================
# vs the AOT extension (same device code)
# =============================================================================

needs_aot = pytest.mark.skipif(
    AOT_MODULE is None, reason="AOT extension not built (make build-cuda)"
)


@needs_aot
@pytest.mark.parametrize("dim", [2, 3, 4])
@pytest.mark.parametrize("use_fp16", [False, True])
def test_forward_backward_match_aot(dim: int, use_fp16: bool) -> None:
    nvrtc = _nvrtc_or_skip()
    assert AOT_MODULE is not None
    p = _problem(dim, n=200)
    args = (p["centers"], p["conic"], p["amps"], list(p["shape"]), 3.0, 1e-5, use_fp16)
    out_n, st_n = nvrtc.forward(*args)
    out_a, st_a = AOT_MODULE.forward(*args)
    assert _rel_l2(out_n, out_a) < 1e-6
    assert torch.equal(st_n, st_a)

    grad = torch.rand_like(out_a)
    g_n = nvrtc.backward(grad, *args[:-1], use_fp16, shape_tensor_cached=st_n)
    g_a = AOT_MODULE.backward(grad, *args[:-1], use_fp16, shape_tensor_cached=st_a)
    for name, gn, ga in zip(("d_centers", "d_conic", "d_amps"), g_n, g_a, strict=True):
        assert gn.shape == ga.shape and gn.dtype == ga.dtype == torch.float32, name
        assert _rel_l2(gn, ga) < 1e-6, f"{name}: rel L2 {_rel_l2(gn, ga):.2e}"


# =============================================================================
# Host-glue contract (ported from cuda_splatting.cu / bindings.cpp)
# =============================================================================


def test_output_to_zero_and_output_buffer_round_trip() -> None:
    nvrtc = _nvrtc_or_skip()
    p = _problem(3)
    args = (p["centers"], p["conic"], p["amps"], list(p["shape"]), 3.0, 1e-5)
    out, st = nvrtc.forward(*args)
    reference = out.clone()
    assert out.abs().sum() > 0

    nvrtc.backward(torch.ones_like(out), *args, False, st, out)
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(out)) == 0  # zeroed as a backward side effect

    again, _ = nvrtc.forward(*args, False, out)  # reuses the pre-zeroed buffer
    assert again.data_ptr() == out.data_ptr()
    torch.testing.assert_close(again, reference, rtol=1e-5, atol=1e-7)


def test_zero_splats() -> None:
    nvrtc = _nvrtc_or_skip()
    empty = torch.zeros(0, 3, device="cuda")
    out, _ = nvrtc.forward(
        empty,
        torch.zeros(0, 6, device="cuda"),
        torch.zeros(0, device="cuda"),
        [4, 5, 6],
        3.0,
        1e-5,
    )
    assert out.shape == (120,) and int(torch.count_nonzero(out)) == 0
    grads = nvrtc.backward(
        torch.ones(120, device="cuda"),
        empty,
        torch.zeros(0, 6, device="cuda"),
        torch.zeros(0, device="cuda"),
        [4, 5, 6],
        3.0,
        1e-5,
    )
    assert [g.shape for g in grads] == [(0, 3), (0, 6), (0,)]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda p: {**p, "centers": p["centers"].cpu()},
            "centers must be on CUDA device",
        ),
        (
            lambda p: {**p, "conic": p["conic"][:, :5].contiguous()},
            r"conic must have shape \(N, 6\)",
        ),
        (lambda p: {**p, "amps": p["amps"][:-1]}, "amps must have shape"),
        (lambda p: {**p, "centers": p["centers"].double()}, "centers must be float32"),
        (
            lambda p: {**p, "conic": p["conic"].t().contiguous().t()},
            "conic must be contiguous",
        ),
        (lambda p: {**p, "shape": (8,) * 9}, "Dimension must be between 2 and 8"),
    ],
)
def test_validation_errors(mutate: Any, match: str) -> None:
    nvrtc = _nvrtc_or_skip()
    p = mutate(_problem(3))
    with pytest.raises(RuntimeError, match=match):
        nvrtc.forward(p["centers"], p["conic"], p["amps"], list(p["shape"]), 3.0, 1e-5)


def test_backward_rejects_wrong_sized_grad() -> None:
    nvrtc = _nvrtc_or_skip()
    p = _problem(3)
    with pytest.raises(RuntimeError, match="grad_output must have prod"):
        nvrtc.backward(
            torch.ones(7, device="cuda"),
            p["centers"],
            p["conic"],
            p["amps"],
            list(p["shape"]),
            3.0,
            1e-5,
        )


def test_launches_on_current_stream() -> None:
    nvrtc = _nvrtc_or_skip()
    p = _problem(3)
    args = (p["centers"], p["conic"], p["amps"], list(p["shape"]), 3.0, 1e-5)
    expected, _ = nvrtc.forward(*args)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out, _ = nvrtc.forward(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_fit_end_to_end_with_nvrtc(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short real fit through fit_gaussian_splats runs on the NVRTC kernels."""
    _nvrtc_or_skip()
    from luxar.gsplats import fit_gaussian_splats
    from luxar.gsplats.models.gsplats.cuda import backend as backend_mod

    monkeypatch.setenv(backend_mod.BACKEND_ENV, "nvrtc")
    seed_all(3)
    g = np.arange(32, dtype=np.float32)
    vol = np.zeros((32, 32, 32), np.float32)
    for c in ((10, 12, 20), (22, 18, 9), (16, 25, 24)):
        r2 = (
            (g[:, None, None] - c[0]) ** 2
            + (g[None, :, None] - c[1]) ** 2
            + (g[None, None, :] - c[2]) ** 2
        )
        vol += np.exp(-r2 / (2 * 2.0**2))
    result = fit_gaussian_splats(
        vol, seeds=60, n_iters=60, device="cuda", verbose=False
    )
    assert backend_mod.get_splatting_backend().name == "nvrtc"  # type: ignore[union-attr]
    assert len(result.centers) > 0
    assert np.isfinite(np.asarray(result.amplitudes)).all()
